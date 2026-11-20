"""One output contract for every external longitudinal denoiser in the benchmark.

An adapter runs the *authors'* software on one simulated dataset and returns a
relative-composition estimate on the v3.1 grid.  Nothing here fits a model: the
module only fixes the interface, validates it and records provenance, so that
`compare_methods.py` can put TGP-CODA, TEMPTED, microTensor, EMBED, PARAFAC2 and
MALLARD in one paired table scored by the single `score_composition()` in
`external_benchmarks.py`.

Contract (checked by `validate_comp_mean`):

* `comp_mean` is floating point with exactly `Y.shape`;
* it is finite, non-negative and sums to one along taxa at every scored visit;
* cells the method does not estimate are NaN -- never a silent zero;
* the taxon set is never shrunk: an adapter that drops taxa must fail the task;
* the adapter must not modify the simulated counts in place.

Rank selection (pre-registered rule B, see EXTERNAL_BENCHMARKS.md): every
low-rank method searches the same grid over the same method-independent
validation samples, scored by held-out multinomial cross-entropy computed from
observed counts alone.  Simulation truth is never read in this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

CONTRACT_VERSION = "1.0"
# Shared by every low-rank method; truncated per dataset by truncate_rank_grid.
DEFAULT_RANK_GRID = (2, 3, 5)
# Fraction of observed samples held out by the shared validation split.
DEFAULT_CV_FRACTION = 0.15
SUM_TOLERANCE = 1e-4


class ContractError(ValueError):
    """An adapter returned something the benchmark cannot score."""


class MethodUnsupported(RuntimeError):
    """The dataset is outside this method's documented applicability domain."""


# --------------------------------------------------------------------- arrays

def f32(value):
    return np.asarray(value, dtype=np.float32)


def stable_softmax(z, axis=-1):
    """exp(z - max z) / sum exp(z - max z): the inverse transform used by
    microTensor's helpers_evaluation.R and by the PARAFAC2 CLR reconstruction."""
    z = np.asarray(z, dtype=float)
    shifted = z - np.nanmax(z, axis=axis, keepdims=True)
    weight = np.exp(shifted)
    total = np.nansum(weight, axis=axis, keepdims=True)
    return np.divide(weight, total, out=np.full_like(weight, np.nan), where=total > 0)


def observed_visits(truth):
    """(subject, visit) pairs with observed counts.  `mask` is authoritative:
    a visit is either fully observed or entirely absent from the simulation.

    A partially masked visit would make "observed" ambiguous -- the adapters
    treat a visit as a unit and the scorer uses the full mask -- so it is
    rejected rather than resolved by a guess.
    """
    mask = np.asarray(truth["mask"])
    whole, any_ = mask.all(axis=2), mask.any(axis=2)
    if not np.array_equal(whole, any_):
        raise ContractError(
            f"{int((any_ & ~whole).sum())} visits are observed for some taxa only; this "
            "benchmark assumes whole-visit missingness (see simulate_microbiome.mask)")
    return whole


def counts_fingerprint(truth):
    """Hash of everything an adapter may read, so an in-place edit is detected
    (adapters are handed the shared per-process truth cache, not a copy)."""
    digest = hashlib.sha256()
    for key in ("Y", "mask", "depth", "times"):
        array = np.ascontiguousarray(truth[key])
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()[:32]


# -------------------------------------------------------------------- results

@dataclass
class ExternalResult:
    """What every adapter returns.  `comp_mean` is the only scored field."""
    comp_mean: np.ndarray
    meta: dict
    arrays: dict = field(default_factory=dict)   # extra npz payload
    warnings: list = field(default_factory=list)

    def with_warning(self, message):
        self.warnings.append(str(message))
        return self


def validate_comp_mean(comp_mean, truth, method, scored=None, tolerance=SUM_TOLERANCE):
    """Raise ContractError unless comp_mean can be scored beside every other method.

    `scored` restricts the check to the cells a run actually claims (e.g. the
    TGP-CODA subject subset); it defaults to the dataset's observation mask.
    """
    comp = np.asarray(comp_mean)
    shape = np.asarray(truth["Y"]).shape
    if comp.shape != shape:
        raise ContractError(f"{method}: comp_mean has shape {comp.shape}, expected {shape}")
    if comp.dtype.kind != "f":
        raise ContractError(f"{method}: comp_mean must be floating point, got {comp.dtype}")
    scored = observed_visits(truth) if scored is None else np.asarray(scored, bool)
    if scored.ndim == 3:
        scored = scored.all(axis=2)
    if not scored.any():
        raise ContractError(f"{method}: no scored visit -- nothing was estimated")
    values = comp[scored]
    if not np.isfinite(values).all():
        bad = int((~np.isfinite(values)).sum())
        raise ContractError(f"{method}: {bad} non-finite values at scored visits")
    if (values < 0).any():
        raise ContractError(f"{method}: comp_mean has negative values at scored visits")
    totals = values.sum(axis=-1)
    if not np.allclose(totals, 1.0, atol=tolerance, rtol=0):
        worst = float(np.max(np.abs(totals - 1.0)))
        raise ContractError(f"{method}: scored visits must sum to 1 along taxa "
                            f"(worst |sum-1| = {worst:.3g})")
    unscored = comp[~scored]
    if unscored.size and not np.isnan(unscored).all():
        wrong = int(np.sum(~np.isnan(unscored)))
        raise ContractError(f"{method}: {wrong} unscored cells are not NaN; a missing visit "
                            "must not be reported as a zero composition")
    return dict(scored_visits=int(scored.sum()), total_visits=int(scored.size),
                taxa=int(shape[2]), max_sum_error=float(np.max(np.abs(totals - 1.0))))


def check_taxa_preserved(method, n_in, n_out, dropped=()):
    """A method that filters taxa cannot be compared on the common taxon set."""
    if n_out != n_in or len(dropped):
        raise ContractError(
            f"{method}: returned {n_out} of {n_in} taxa (dropped {list(dropped)[:10]}). "
            "The benchmark scores one common taxon set; regenerate the plan with a "
            "method-independent filter instead of shrinking it here.")


# ---------------------------------------------------------------- provenance

def source_commit(root):
    """Pinned commit of a vendored source tree: `.commit` stamp, else git."""
    root = Path(root)
    stamp = root / ".commit"
    if stamp.is_file():
        return stamp.read_text(encoding="utf-8").strip()
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root), text=True,
                             capture_output=True, timeout=30)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def allocated_cpus():
    if os.environ.get("SLURM_CPUS_PER_TASK", "").isdigit():
        return int(os.environ["SLURM_CPUS_PER_TASK"])
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def children_cpu_seconds():
    try:
        import resource
    except ImportError:          # Windows: only reached by the local tests
        return 0.0
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def environment_info():
    return dict(python=sys.version.split()[0], platform=platform.platform(),
                numpy=np.__version__, executable=sys.executable,
                slurm_job=os.environ.get("SLURM_JOB_ID"),
                slurm_array_task=os.environ.get("SLURM_ARRAY_TASK_ID"))


def build_meta(method, *, method_version, source_repository, source_commit_id, settings,
               rank, seed, wall_seconds, alloc_cpus=None, cpu_hours_used=None,
               transform, inverse_transform, status, input_shape, output_shape,
               filtered_taxa=(), filtered_visits=(), warnings=(), **extra):
    """The provenance block written into every array npz and every task JSON."""
    cpus = allocated_cpus() if alloc_cpus is None else int(alloc_cpus)
    meta = dict(contract_version=CONTRACT_VERSION, method=method, method_version=method_version,
                source_repository=source_repository, source_commit=source_commit_id,
                settings=settings, rank=rank, seed=seed,
                wall_seconds=float(wall_seconds), alloc_cpus=cpus,
                cpu_hours_alloc=float(wall_seconds) * cpus / 3600.0,
                cpu_hours_used=None if cpu_hours_used is None else float(cpu_hours_used),
                transform=transform, inverse_transform=inverse_transform,
                status=status, input_shape=list(input_shape), output_shape=list(output_shape),
                filtered_taxa=list(filtered_taxa), filtered_visits=list(filtered_visits),
                warnings=[str(w) for w in warnings], environment=environment_info())
    meta.update(extra)
    return meta


# ----------------------------------------------------- shared rank selection

def truncate_rank_grid(truth, grid=DEFAULT_RANK_GRID, extra_limit=None):
    """Ranks a decomposition of this tensor can carry: below every mode size."""
    I, T, J = np.asarray(truth["Y"]).shape
    limit = min(I, T, J)
    if extra_limit:
        limit = min(limit, int(extra_limit))
    kept = sorted({int(r) for r in grid if 1 <= r < limit})
    if not kept:
        raise MethodUnsupported(f"no rank in {tuple(grid)} fits a {I}x{T}x{J} tensor")
    return kept


def validation_samples(truth, seed, fraction=DEFAULT_CV_FRACTION):
    """A fixed, method-independent set of held-out (subject, visit) samples.

    Only observed samples are eligible, a subject's first and last observed
    visits are always kept (every method needs its own time range), and at most
    a third of a subject's visits is held out.  The split depends on the dataset
    and the seed alone, so all methods and all ranks see exactly the same mask.
    Simulation truth is not consulted.
    """
    observed = observed_visits(truth)
    rng = np.random.default_rng(int(seed))
    held = np.zeros_like(observed)
    for i in range(observed.shape[0]):
        visits = np.flatnonzero(observed[i])
        if visits.size < 4:
            continue                                   # too short to hold anything out
        eligible = visits[1:-1]
        n = min(int(round(fraction * visits.size)), max(1, visits.size // 3))
        if n < 1 or eligible.size == 0:
            continue
        held[i, rng.choice(eligible, size=min(n, eligible.size), replace=False)] = True
    return held


def holdout_cross_entropy(truth, comp_mean, held):
    """Mean per-sample multinomial cross-entropy of held-out observed counts.

    Lower is better.  Uses observed counts only -- never `clean_composition` --
    so it is a legal model-selection criterion inside the benchmark.
    """
    Y = np.asarray(truth["Y"], dtype=float)
    comp = np.asarray(comp_mean, dtype=float)
    held = np.asarray(held, bool)
    if not held.any():
        return float("nan")
    scores = []
    for i, t in zip(*np.nonzero(held)):
        counts, p = Y[i, t], comp[i, t]
        total = counts.sum()
        if total <= 0 or not np.isfinite(p).all():
            continue
        floor = 1.0 / max(float(total) * comp.shape[2], 1.0)
        p = np.maximum(p, floor)
        p = p / p.sum()
        scores.append(float(-(counts * np.log(p)).sum() / total))
    return float(np.mean(scores)) if scores else float("nan")


def select_rank(fit_at_rank, truth, grid, seed, fraction=DEFAULT_CV_FRACTION):
    """Pre-registered rule B: lowest held-out cross entropy on the shared split.

    `fit_at_rank(rank, held)` must fit on the observed samples that are NOT held
    out and return a full (I, T, J) composition including the held-out samples
    (every method here carries a temporal component defined on the whole grid).
    Ties, and ranks whose fit fails, resolve towards the smaller rank.
    """
    held = validation_samples(truth, seed, fraction)
    scores, failures = {}, {}
    for rank in grid:
        try:
            estimate = fit_at_rank(rank, held)
            scores[rank] = holdout_cross_entropy(truth, estimate, held)
        except Exception as error:            # a rank that will not fit is not selectable
            failures[rank] = f"{type(error).__name__}: {error}"
    usable = {r: s for r, s in scores.items() if np.isfinite(s)}
    if not usable:
        raise MethodUnsupported(f"no rank in {grid} produced a usable validation fit: {failures}")
    best = min(usable, key=lambda r: (usable[r], r))
    return dict(rank=best,
                scores={str(k): (None if not np.isfinite(v) else float(v))
                        for k, v in scores.items()},
                failures=failures, held_out_samples=int(held.sum()),
                rule="masked-sample CV, shared grid, held-out multinomial cross-entropy",
                cv_seed=int(seed), cv_fraction=float(fraction))


# ----------------------------------------------------------- small utilities

class Timer:
    """Wall and CPU seconds of this process and its children."""

    def __enter__(self):
        self.wall = time.monotonic()
        self.cpu = children_cpu_seconds() + time.process_time()
        return self

    def __exit__(self, *exc):
        self.wall = time.monotonic() - self.wall
        self.cpu = children_cpu_seconds() + time.process_time() - self.cpu
        return False


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    # Path.write_text gained newline only in Python 3.10; QIIME2 uses 3.8.
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, default=str) + "\n")
    os.replace(tmp, path)


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
