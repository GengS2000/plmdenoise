"""DA extension of simple_v1 scenario 7; no work is performed on import."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from simulate_microbiome import SimulationConfig, _simulate_simple


METHODS = ("linda", "maaslin2", "glmmpql", "zigmm")
INPUTS = ("naive", "m4_flex", "tempted_gemelli", "microtensor", "embed", "parafac2")
# Inputs written at generation time, never by a denoiser. oracle_* are diagnostic
# references from the saved truth; they are never given to any denoiser.
GENERATED_INPUTS = ("naive", "oracle_clean", "oracle_abundance")
# Inputs derived from another fit's saved posterior: input -> producing fit.
DERIVED_INPUTS = {"m4_flex_abundance": "m4_flex"}
EFFECTS = {"strong": 1.0, "weak": 0.03}
# Pre-registered oracle/abundance grid (natural-log gamma). Chosen from the
# local mechanism check (pilot seeds 6100-6103, never confirmation seeds):
# 0.2-0.5 spans the abundance-oracle power transition, 0.4-0.75 the naive
# CLR transition, and 0.75 shows where closure starts to move uninjected taxa.
ORACLE_GRID = (0., .2, .3, .4, .5, .75)


def fit_source(source):
    """The denoiser whose completion marker covers this input (None if generated)."""
    if source in GENERATED_INPUTS:
        return None
    return DERIVED_INPUTS.get(source, source)


def closed(values):
    values = np.asarray(values, dtype=float)
    return values / values.sum(axis=-1, keepdims=True)


def abundance_composition(log_abundance):
    """exp(Z) closed over taxa, without the structural-zero channel (stable softmax)."""
    z = np.asarray(log_abundance, dtype=float)
    return closed(np.exp(z - z.max(axis=-1, keepdims=True)))


def simulation_config(seed: int, **overrides) -> SimulationConfig:
    # Same defaults as run_validation.SIMPLE_DEFAULTS, restricted to scene 7/base.
    return SimulationConfig(**(dict(
        design="simple_v1", scenario=7, n_subjects=40, n_visits=12, n_taxa=500,
        n_factors=1, n_batches=1, n_events=1, median_depth=30000.,
        depth_log_sd=0., batch_sd=0., effect_scale=float(np.log(2)),
        exposure_fraction=.8, response_fraction=.3, hetero_sd=0., seed=seed,
    ) | overrides))


def effect_cells(levels=(0, 1)):
    levels = sorted(set(levels))
    if not levels or any(not isinstance(s, int) or s < 0 or s > 5 for s in levels):
        raise ValueError("Effect levels must be integers in 0..5")
    cells = []
    if 0 in levels:
        cells.append(dict(series="null", s=0, strength=0.))
    for series, factor in EFFECTS.items():
        cells.extend(dict(series=series, s=s, strength=factor * s) for s in levels if s)
    return cells


def effect_grid(strengths):
    """Explicit natural-log effects, injected before zeros/counts/normalisation."""
    values = [float(value) for value in strengths]
    if (not values or len(set(values)) != len(values) or
            any(not math.isfinite(value) or not 0 <= value <= 5 for value in values) or 0 not in values):
        raise ValueError("Effect grid must contain distinct finite values in [0, 5], including zero")
    return [dict(series="null" if value == 0 else "dose", s=value, strength=value)
            for value in sorted(values)]


def simulate_da(config: SimulationConfig, strength: float):
    if config.design != "simple_v1":
        raise ValueError("DA is only defined for simple_v1")
    return _simulate_simple(config, da_strength=strength)


def write_csv(path: Path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def identifiers(shape):
    subjects, visits, taxa = shape
    taxon_ids = [f"taxon_{j + 1:04d}" for j in range(taxa)]
    sample_ids = [f"subject_{i + 1:03d}__visit_{t + 1:02d}"
                  for i in range(subjects) for t in range(visits)]
    return sample_ids, taxon_ids


def export_matrix(path: Path, values):
    values = np.asarray(values)
    if values.ndim != 3 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Expected finite nonnegative subject x visit x taxon values")
    samples, taxa = identifiers(values.shape)
    write_csv(path, ["sample_id"] + taxa,
              (dict(zip(["sample_id"] + taxa, [sid] + row.tolist()))
               for sid, row in zip(samples, values.reshape(-1, values.shape[-1]))))


def export_observed(directory: Path, truth):
    """R receives only observed composition and metadata, never simulation truth."""
    y = truth["Y"]
    total = y.sum(-1, keepdims=True)
    if (total <= 0).any():
        raise ValueError("All-zero sample: cannot compare composition methods")
    export_matrix(directory / "counts.csv", y)
    export_matrix(directory / "naive.csv", y / total)
    samples, taxa = identifiers(y.shape)
    rows = []
    for k, sid in enumerate(samples):
        i, t = divmod(k, y.shape[1])
        tau = float(truth["event_times"][i, 0])
        time = float(truth["times"][i, t])
        rows.append(dict(sample_id=sid, participant=f"subject_{i + 1:03d}",
                         X=int(truth["group"][i]), visit=t + 1, time=time,
                         event_post=int(np.isfinite(tau) and time >= tau),
                         exposed=int(np.isfinite(tau)),
                         exposure=float(truth["depth"][i, t]), library_size=int(total[i, t, 0])))
    write_csv(directory / "metadata.csv", list(rows[0]), rows)
    write_csv(directory / "truth.csv", ["taxon", "gamma", "assigned_sign", "is_differential"],
              (dict(taxon=taxon, gamma=float(truth["da_gamma"][j]),
                    assigned_sign=int(truth["da_sign"][j]),
                    is_differential=int(truth["da_is_differential"][j]))
               for j, taxon in enumerate(taxa)))


def bh_full_family(pvalues):
    """BH over the predeclared family; missing/failed tests count as p=1."""
    p = np.asarray(pvalues, dtype=float)
    valid = np.isfinite(p) & (p >= 0) & (p <= 1)
    safe = np.where(valid, p, 1.)
    order = np.argsort(safe, kind="stable")
    adjusted = np.minimum.accumulate((safe[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    q = np.empty(len(p))
    q[order] = np.minimum(adjusted, 1.)
    return q, valid


def score_results(truth_file: Path, result_file: Path, alpha: float):
    with truth_file.open(encoding="utf-8", newline="") as handle:
        truth = list(csv.DictReader(handle))
    with result_file.open(encoding="utf-8", newline="") as handle:
        results = list(csv.DictReader(handle))
    names = [row["taxon"] for row in results]
    expected = [row["taxon"] for row in truth]
    if len(set(names)) != len(names) or set(names) != set(expected):
        raise ValueError("DA output must contain every taxon exactly once")
    lookup = {row["taxon"]: row for row in results}
    ordered = [lookup[taxon] for taxon in expected]
    pvalues = []
    for row in ordered:
        try:
            value = float(row["pvalue"])
        except (ValueError, TypeError):
            value = math.nan
        pvalues.append(value if row["fit_status"] in ("ok", "warning") else math.nan)
    q, valid = bh_full_family(pvalues)
    actual = np.array([int(row["is_differential"]) != 0 for row in truth])
    called = valid & (q <= alpha)
    tp, fp = int(sum(called & actual)), int(sum(called & ~actual))
    fn, tn = int(sum(~called & actual)), int(sum(~called & ~actual))
    metrics = dict(n_taxa=len(actual), n_tested=int(sum(valid)), n_failed=int(sum(~valid)),
                   n_warning=sum(row["fit_status"] == "warning" for row in ordered),
                   boundary_fraction=sum(row.get("fit_engine") == "ZIGMM_boundary_pi0_nlme_ML"
                                         for row in ordered) / len(ordered),
                   n_true=int(sum(actual)), n_called=int(sum(called)), TP=tp, FP=fp, FN=fn, TN=tn,
                   # A dataset with zero returned tests is a failure, not FDR=0.
                   fdp=fp / max(tp + fp, 1) if valid.any() else None,
                   power=tp / sum(actual) if actual.any() and valid.any() else None,
                   precision=tp / (tp + fp) if tp + fp else None,
                   f1=2 * tp / (2 * tp + fp + fn) if actual.any() and valid.any() else None,
                   fpr=fp / sum(~actual) if (~actual).any() and valid.any() else None,
                   any_false_discovery=int(fp > 0) if valid.any() else None,
                   failure_fraction=float(np.mean(~valid)))
    scored = []
    for j, row in enumerate(ordered):
        scored.append(row | dict(gamma=truth[j]["gamma"], is_differential=int(actual[j]),
                                 qvalue=float(q[j]), significant=int(called[j])))
    return metrics, scored
