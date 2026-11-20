"""Run published longitudinal microbiome denoisers on a v3.1 simulation plan.

The adapters deliberately use the authors' implementations:

* TGP-CODA: ``gpmicrobiome.run_nuts`` (one independent fit per subject).
* LUMINATE: ``main.py estimate`` from https://github.com/tyjo/luminate.
  Retained for the feasibility report only; see ``luminate_feasibility.py``.
* TEMPTED, microTensor, EMBED, PARAFAC2 and the MALLARD pilot through
  ``adapters/``; each obeys ``external_method_contract`` and returns a full
  relative composition, validated before it is scored.

Only compositional recovery is scored.  None of these methods estimates the
absolute latent abundance, batch component, or event-response estimands of our
model, and a low-rank method's loadings are never treated as denoised
abundances: only the composition its own inverse transform produces is used.

The output directory is a store keyed by dataset id (a hash of the simulation
settings), so several plans that share datasets (e.g. p500_demo and the full
benchmark) reuse each other's fits.  TGP-CODA is scheduled per subject because
every subject is an independent NUTS fit; a dataset is assembled and scored once
all of its subjects are present.  Per-dataset arrays are written to
``arrays/<dataset>_<method>.npz`` next to the scored ``tasks/`` rows.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

from evaluate import _scale_block
import external_method_contract as contract
from simulate_microbiome import SimulationConfig, simulate


# Methods driven through adapters/ and the shared output contract.
ADAPTER_METHODS = ("tempted_gemelli", "microtensor", "embed", "parafac2", "mallard_pilot")
METHODS = ("tgp_coda", "luminate") + ADAPTER_METHODS
# Settings that name an executable, a source checkout or a wall-clock budget.
# They legitimately differ between machines, so they are recorded per run but
# excluded from the store's "one configuration per method" check.
ENVIRONMENT_KEYS = ("source_dir", "stan_dir", "rscript", "octave", "matlab",
                    "engine", "timeout", "cores")


def allocated_cpus():
    """CPUs this process may use: the Slurm allocation, else the affinity mask."""
    if os.environ.get("SLURM_CPUS_PER_TASK", "").isdigit():
        return int(os.environ["SLURM_CPUS_PER_TASK"])
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def children_cpu_seconds():
    try:
        import resource
    except ImportError:  # Windows: only used by local tests
        return 0.0
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def timing(wall, cpu_used, cpus):
    """Uniform cost fields shared by every method (see timing_report.py)."""
    return dict(wall_seconds=float(wall), alloc_cpus=int(cpus),
                cpu_hours_alloc=float(wall) * cpus / 3600.0,
                cpu_hours_used=None if cpu_used is None else float(cpu_used) / 3600.0)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + f".{os.getpid()}.tmp.npz")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def dataset_id(simulation):
    payload = json.dumps(simulation, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def datasets_from_plan(plan_dir):
    plan = json.loads((Path(plan_dir) / "plan.json").read_text(encoding="utf-8"))
    unique = {}
    for task in plan["tasks"]:
        simulation = task["simulation"]
        key = dataset_id(simulation)
        unique.setdefault(key, dict(dataset_id=key, scenario=task["scenario"],
                                    condition=task["condition"], seed=task["seed"],
                                    simulation=simulation))
    return [unique[key] for key in sorted(unique)]


def tgp_subjects(task, max_subjects=None):
    n = task["simulation"]["n_subjects"]
    return n if not max_subjects else min(n, max_subjects)


def job_list(datasets, methods, max_subjects=None):
    """(dataset, method, subject) units; subject is None for whole-dataset fits.

    max_subjects limits TGP-CODA to the first subjects of each dataset (lite
    timing runs); its subjects are independent fits, so this is not a different model.
    """
    jobs = []
    for task in datasets:
        for method in methods:
            if method == "tgp_coda":
                jobs.extend((task, method, i) for i in range(tgp_subjects(task, max_subjects)))
            else:
                jobs.append((task, method, None))
    return jobs


def _f32(value):
    return np.asarray(value, dtype=np.float32)


def truth_arrays(truth):
    """The observation and targets saved beside every external estimate."""
    return dict(Y=truth["Y"].astype(np.int32), mask=truth["mask"].astype(bool),
                depth=truth["depth"].astype(np.int64), times=_f32(truth["times"]),
                group=truth["group"], batch=truth["batch"], event_times=_f32(truth["event_times"]),
                clean_composition=_f32(truth["clean_composition"]), X_true=_f32(truth["X_true"]),
                S_true=truth["S_true"].astype(bool))


def score_composition(truth, estimate, method, task, elapsed):
    """Score only the estimand shared by all three approaches.

    The log floor matches evaluate.evaluate (1 / median depth) so comp_* rows
    are directly comparable with the v3.1 fits on the same dataset.
    """
    estimate = np.asarray(estimate, dtype=float)
    if estimate.shape != truth["Y"].shape or not np.isfinite(estimate[truth["mask"]]).all():
        raise ValueError("estimate must be finite on observed cells and match Y.shape")
    estimate = np.maximum(np.nan_to_num(estimate, nan=0.0), 0)
    total = estimate.sum(axis=2, keepdims=True)
    if np.any(total[truth["mask"].any(axis=2)] <= 0):
        raise ValueError("every observed visit must have positive estimated mass")
    estimate = np.divide(estimate, total, out=np.zeros_like(estimate), where=total > 0)
    floor = 1.0 / max(float(np.median(truth["depth"])), 1.0)
    row = dict(scenario=task["scenario"], scenario_name=json.loads(str(truth["metadata_json"]))["scenario_name"],
               condition=task["condition"], seed=task["seed"], method=method,
               estimand="relative_composition_only", fit_seconds=elapsed,
               dataset_id=task["dataset_id"])
    row.update(_scale_block(truth["clean_composition"], estimate, truth["mask"], floor,
                            "comp", compositional=True))
    return row


# --------------------------------------------------------------------------- TGP-CODA

def tgp_subject_path(out, task, i):
    return Path(out) / "work" / task["dataset_id"] / "tgp_coda" / f"subject_{i:03d}.npz"


def run_tgp_subject(truth, task, i, args):
    try:
        from jax import random
        import numpyro
        from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
        from gpmicrobiome import run_nuts
    except ImportError as error:
        raise RuntimeError("Install the tgp-coda environment (slurm/setup_external_envs.sh)") from error
    numpyro.enable_x64()
    J = truth["Y"].shape[2]
    visit = truth["mask"][i].all(axis=1)
    if visit.sum() < 2:
        raise ValueError(f"subject {i} has fewer than two observed visits")
    x = truth["times"][i, visit, None].astype(float)
    # The current author tutorial uses consecutive integer time points.
    # Express our arbitrary 0..horizon unit in median-visit intervals so
    # the package's default length-scale prior has the intended meaning.
    gaps = np.diff(x[:, 0])
    scale = float(np.median(gaps[gaps > 0])) if np.any(gaps > 0) else 1.0
    x = (x - x.min()) / scale
    y = truth["Y"][i, visit]
    key = random.PRNGKey(args.seed_offset + int(task["seed"]) + i)
    started, cpu_started = time.monotonic(), time.process_time()
    fit = run_nuts(key, x, y, num_basis=args.tgp_basis, num_warmup=args.tgp_warmup,
                   num_samples=args.tgp_samples, num_chains=args.tgp_chains)
    by_chain = fit.get_samples(group_by_chain=True)
    theta = np.asarray(by_chain["theta"])  # chains x draws x visits x taxa
    if theta.shape[2:] != (visit.sum(), J):
        raise ValueError(f"unexpected TGP-CODA theta shape {theta.shape}")
    flat = theta.reshape(-1, *theta.shape[2:])
    lower, upper = np.quantile(flat, [0.025, 0.975], axis=0)
    rhat = np.asarray(split_gelman_rubin(theta))
    ess = np.asarray(effective_sample_size(theta))
    diverging = np.asarray(fit.get_extra_fields()["diverging"])
    diagnostics = dict(subject=i, seconds=time.monotonic() - started,
                       cpu_seconds=time.process_time() - cpu_started, alloc_cpus=allocated_cpus(),
                       theta_rhat_max=float(np.nanmax(rhat)),
                       theta_rhat_frac_gt_1_05=float(np.mean(rhat > 1.05)),
                       theta_ess_min=float(np.nanmin(ess)), theta_ess_median=float(np.nanmedian(ess)),
                       divergences=int(diverging.sum()), draws=int(flat.shape[0]))
    b_mean = np.asarray(by_chain["b"]).reshape(-1, visit.sum(), J).mean(0) if "b" in by_chain else np.full((visit.sum(), J), np.nan)
    return dict(visit=np.flatnonzero(visit), mean=_f32(flat.mean(0)), lower=_f32(lower), upper=_f32(upper),
                b_mean=_f32(b_mean), diagnostics=np.array(json.dumps(diagnostics)))


def assemble_tgp(truth, task, args):
    """Combine per-subject fits into one dataset estimate; None if incomplete.

    With max_subjects (lite runs) only those subjects are fit and scored.
    """
    I, T, J = truth["Y"].shape
    settings = read_json(Path(args.out) / "settings_tgp_coda.json") or {}
    fitted = tgp_subjects(task, settings.get("max_subjects"))
    paths = [tgp_subject_path(args.out, task, i) for i in range(fitted)]
    if not all(p.exists() for p in paths):
        return None
    mean, lower, upper, b_mean = (np.full((I, T, J), np.nan, dtype=np.float32) for _ in range(4))
    diagnostics = []
    for i, path in enumerate(paths):
        with np.load(path) as part:
            v = part["visit"]
            mean[i, v], lower[i, v], upper[i, v], b_mean[i, v] = part["mean"], part["lower"], part["upper"], part["b_mean"]
            diagnostics.append(json.loads(str(part["diagnostics"])))
    elapsed = float(sum(d["seconds"] for d in diagnostics))
    scored = truth["mask"].copy()
    scored[fitted:] = False
    row = score_composition(dict(truth, mask=scored), mean, "tgp_coda", task, elapsed)
    # Subjects are independent jobs: wall time and CPU cost add up over subjects.
    row.update(timing(elapsed, sum(d.get("cpu_seconds", 0.0) for d in diagnostics), 1))
    row["cpu_hours_alloc"] = sum(d["seconds"] * d.get("alloc_cpus", 1) for d in diagnostics) / 3600.0
    row["alloc_cpus"] = max(d.get("alloc_cpus", 1) for d in diagnostics)
    row.update(subjects_fit=fitted, subjects_total=I,
               tgp_subject_seconds=[d["seconds"] for d in diagnostics])
    target, cell = truth["clean_composition"], scored
    row["comp_coverage95"] = float(np.mean((target[cell] >= lower[cell]) & (target[cell] <= upper[cell])))
    row["comp_width95_median"] = float(np.median((upper - lower)[cell]))
    row["tgp_rhat_max"] = max(d["theta_rhat_max"] for d in diagnostics)
    row["tgp_subjects_rhat_gt_1_05"] = sum(d["theta_rhat_max"] > 1.05 for d in diagnostics)
    row["tgp_rhat_frac_gt_1_05"] = float(np.mean([d["theta_rhat_frac_gt_1_05"] for d in diagnostics]))
    row["tgp_ess_min"] = min(d["theta_ess_min"] for d in diagnostics)
    row["tgp_divergences"] = sum(d["divergences"] for d in diagnostics)
    row["tgp_subject_seconds_max"] = max(d["seconds"] for d in diagnostics)
    atomic_npz(Path(args.out) / "arrays" / f"{task['dataset_id']}_tgp_coda.npz",
               **truth_arrays(truth), comp_mean=mean, comp_lower=lower, comp_upper=upper,
               tgp_b_mean=b_mean, meta=np.array(json.dumps(dict(task, method="tgp_coda", subjects_fit=fitted,
                                                                settings=settings,
                                                                subject_diagnostics=diagnostics))))
    return row


# --------------------------------------------------------------------------- LUMINATE

def write_luminate_table(path, truth):
    """Write the authors' two-header-row OTU-table format."""
    samples = []
    for i in range(truth["Y"].shape[0]):
        for t in np.flatnonzero(truth["mask"][i].all(axis=1)):
            samples.append((i, t))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sequence_id"] + [f"subject_{i}" for i, _ in samples])
        writer.writerow(["time"] + [repr(float(truth["times"][i, t])) for i, t in samples])
        for j in range(truth["Y"].shape[2]):
            writer.writerow([f"taxon_{j}"] + [int(truth["Y"][i, t, j]) for i, t in samples])
    return samples


def read_luminate_table(path, truth, samples, fill=0.0):
    table = np.loadtxt(path, delimiter=",", dtype=str)
    values = table[2:, 1:].astype(float).T
    if values.shape != (len(samples), truth["Y"].shape[2]):
        raise ValueError(f"unexpected LUMINATE output shape {values.shape}")
    estimate = np.full(truth["Y"].shape, fill, dtype=float)
    for value, (i, t) in zip(values, samples):
        estimate[i, t] = value
    return estimate


def luminate_commit(root):
    stamp = Path(root) / ".commit"  # written when the source was vendored without .git
    if stamp.is_file():
        return stamp.read_text(encoding="utf-8").strip()
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
                              capture_output=True, timeout=30).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def run_luminate(truth, task, args):
    root = Path(args.luminate_root).resolve()
    main = root / "main.py"
    if not main.is_file():
        raise FileNotFoundError(f"LUMINATE main.py not found under {root}")
    work = (Path(args.out) / "work" / task["dataset_id"] / "luminate").resolve()
    work.mkdir(parents=True, exist_ok=True)
    samples = write_luminate_table(work / "counts.csv", truth)
    # main.py strips "/" from -o, which turns absolute paths into relative
    # ones; run inside the work directory and pass "." instead.
    command = [args.luminate_python, str(main), "estimate", "counts.csv", "-o", "."]
    started, cpu_started = time.monotonic(), children_cpu_seconds()
    completed = subprocess.run(command, cwd=work, capture_output=True, encoding="utf-8", errors="replace",
                               timeout=args.luminate_timeout)
    elapsed = time.monotonic() - started
    cpu_used = children_cpu_seconds() - cpu_started
    (work / "stdout.txt").write_text(completed.stdout or "", encoding="utf-8")
    (work / "stderr.txt").write_text(completed.stderr or "", encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"LUMINATE exited {completed.returncode}; see {work / 'stderr.txt'}")
    estimate = read_luminate_table(work / "counts-est.csv", truth, samples, fill=np.nan)
    nonzero = read_luminate_table(work / "counts-nonzero-posterior-probs.csv", truth, samples, fill=np.nan)
    row = score_composition(truth, estimate, "luminate", task, elapsed)
    row.update(timing(elapsed, cpu_used if cpu_used > 0 else None, allocated_cpus()))
    row["luminate_outer_iterations"] = sum(line.strip().startswith("it:") for line in completed.stdout.splitlines())
    atomic_npz(Path(args.out) / "arrays" / f"{task['dataset_id']}_luminate.npz",
               **truth_arrays(truth), comp_mean=_f32(estimate), nonzero_probability=_f32(nonzero),
               meta=np.array(json.dumps(dict(task, method="luminate", settings=method_settings("luminate", args)))))
    for name in ("counts.csv", "counts-est.csv", "counts-nonzero-posterior-probs.csv", "P.pkl"):
        (work / name).unlink(missing_ok=True)  # the npz keeps everything needed
    return row


# ----------------------------------------------------------------- adapters

def adapter_overrides(args):
    """--set METHOD.KEY=VALUE, parsed as JSON where possible."""
    overrides = {}
    for item in getattr(args, "set", None) or []:
        if "=" not in item or "." not in item.split("=", 1)[0]:
            raise SystemExit(f"--set expects METHOD.KEY=VALUE, got {item!r}")
        target, raw = item.split("=", 1)
        method, key = target.split(".", 1)
        if method not in ADAPTER_METHODS:
            raise SystemExit(f"--set: {method!r} is not an adapter method {ADAPTER_METHODS}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        overrides.setdefault(method, {})[key] = value
    return overrides


def adapter_settings(method, args):
    import adapters
    settings = adapters.defaults(method)
    unknown = set(adapter_overrides(args).get(method, {})) - set(settings)
    if unknown:
        raise SystemExit(f"--set: {method} has no settings {sorted(unknown)}")
    settings.update(adapter_overrides(args).get(method, {}))
    return settings


def adapter_workdir(out, task, method):
    return Path(out) / "work" / task["dataset_id"] / method


def run_adapter(truth, task, method, args):
    """Fit one adapter method, validate the contract and score the composition."""
    import adapters
    module = adapters.load(method)
    settings = adapter_settings(method, args)
    workdir = adapter_workdir(args.out, task, method)
    workdir.mkdir(parents=True, exist_ok=True)
    seed = int(args.seed_offset) + int(task["seed"])
    before = contract.counts_fingerprint(truth)
    started, cpu_started = time.monotonic(), children_cpu_seconds()
    result = module.fit(truth, task, settings, dict(workdir=workdir, seed=seed,
                                                    alloc_cpus=allocated_cpus(),
                                                    commits=source_commits(args)))
    elapsed = time.monotonic() - started
    cpu_used = children_cpu_seconds() - cpu_started
    if contract.counts_fingerprint(truth) != before:
        raise contract.ContractError(f"{method}: the adapter modified the input counts in place")
    contract.validate_comp_mean(result.comp_mean, truth, method)
    row = score_composition(truth, result.comp_mean, method, task, elapsed)
    row.update(timing(elapsed, cpu_used if cpu_used > 0 else None, allocated_cpus()))
    meta = dict(result.meta)
    meta.setdefault("warnings", []).extend(result.warnings)
    row["rank"] = meta.get("rank")
    row["method_status"] = meta.get("status")
    row["n_warnings"] = len(meta.get("warnings", []))
    atomic_npz(Path(args.out) / "arrays" / f"{task['dataset_id']}_{method}.npz",
               **truth_arrays(truth), comp_mean=_f32(result.comp_mean),
               **{k: np.asarray(v) for k, v in result.arrays.items()},
               meta=np.array(json.dumps(dict(task, **meta), default=str)))
    # The npz holds everything that is scored; the exchange tables (which can be
    # hundreds of MB across the grid) go, the logs and diagnostics stay.
    for leftover in list(workdir.glob("counts_*")) + list(workdir.glob("comp_*")):
        leftover.unlink(missing_ok=True)
    return row


def source_commits(args):
    """Pinned commits of the vendored author sources, when they are on disk."""
    roots = dict(luminate=getattr(args, "luminate_root", None),
                 embed=getattr(args, "embed_root", None),
                 parafac2=getattr(args, "parafac2_root", None),
                 mallard=getattr(args, "mallard_root", None),
                 microtensor=getattr(args, "microtensor_root", None),
                 gemelli=getattr(args, "gemelli_root", None))
    return {name: contract.source_commit(root) for name, root in roots.items() if root}


# --------------------------------------------------------------------------- driver

def method_settings(method, args):
    if method == "tgp_coda":
        return dict(warmup=args.tgp_warmup, samples=args.tgp_samples, chains=args.tgp_chains,
                    basis=args.tgp_basis, seed_offset=args.seed_offset,
                    max_subjects=args.tgp_max_subjects or None)
    if method in ADAPTER_METHODS:
        settings = adapter_settings(method, args)
        return {k: v for k, v in sorted(settings.items()) if k not in ENVIRONMENT_KEYS}
    return dict(commit=luminate_commit(args.luminate_root))


def check_settings(args):
    """One store holds one configuration per method; refuse silent mixing."""
    for method in args.methods:
        path = Path(args.out) / f"settings_{method}.json"
        current = method_settings(method, args)
        previous = read_json(path)
        if previous is None:
            try:
                atomic_json(path, current)
            except FileExistsError:
                # Parallel filesystems can refuse rename-over-existing: a concurrent
                # array task created the file first.  Adopt theirs if it matches.
                previous = read_json(path) or {}
                if previous != current:
                    raise SystemExit(f"{path} records {previous}, this run uses {current}. Use a new --out.")
        elif previous != current:
            raise SystemExit(f"{path} records {previous}, this run uses {current}. Use a new --out.")


def task_path(out, task, method):
    return Path(out) / "tasks" / f"{task['dataset_id']}_{method}.json"


def is_complete(path):
    value = read_json(path)
    return bool(value) and value.get("status") == "complete"


def wave_job_indices(n_jobs, shard, shards, wave=0, waves=1):
    """Partition each shard's serial work into disjoint submission waves."""
    if shards < 1 or not 0 <= shard < shards or waves < 1 or not 0 <= wave < waves:
        raise ValueError("Invalid shard or wave")
    return (index for index in range(shard, n_jobs, shards)
            if (index // shards) % waves == wave)


def run(args):
    datasets = datasets_from_plan(args.plan)
    jobs = job_list(datasets, args.methods, args.tgp_max_subjects)
    waves, wave = getattr(args, "waves", 1), getattr(args, "wave", 0)
    indices = wave_job_indices(len(jobs), args.shard, args.shards, wave, waves)
    check_settings(args)
    failures, cache = 0, {}

    def truth_for(task):
        if task["dataset_id"] not in cache:
            cache.clear()
            cache[task["dataset_id"]] = simulate(SimulationConfig(**task["simulation"]))
        return cache[task["dataset_id"]]

    for index in indices:
        task, method, subject = jobs[index]
        path = task_path(args.out, task, method)
        if is_complete(path):
            continue
        if method == "tgp_coda" and tgp_subject_path(args.out, task, subject).exists():
            continue
        print(f"START {index}/{len(jobs)} {task['dataset_id']} {method} subject={subject}", flush=True)
        try:
            truth = truth_for(task)
            if method == "tgp_coda":
                atomic_npz(tgp_subject_path(args.out, task, subject), **run_tgp_subject(truth, task, subject, args))
                path.with_name(path.stem + f"_subject{subject:03d}.failed.json").unlink(missing_ok=True)
                row = assemble_tgp(truth, task, args)
                if row is None:
                    print(f"DONE subject {subject}", flush=True)
                    continue
            elif method in ADAPTER_METHODS:
                row = run_adapter(truth, task, method, args)
            else:
                row = run_luminate(truth, task, args)
            result = dict(status="complete", row=row)
        except Exception as error:
            failures += 1
            result = dict(status="failed", dataset_id=task["dataset_id"], method=method, subject=subject,
                          scenario=task.get("scenario"), condition=task.get("condition"),
                          seed=task.get("seed"),
                          error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(),
                          workdir=str(adapter_workdir(args.out, task, method))
                          if method in ADAPTER_METHODS else None,
                          logs=sorted(str(p) for p in
                                      adapter_workdir(args.out, task, method).glob("std*.txt"))
                          if method in ADAPTER_METHODS else [],
                          details=getattr(error, "details", None))
            print(result["traceback"], file=sys.stderr, flush=True)
            if subject is not None:
                path = path.with_name(path.stem + f"_subject{subject:03d}.failed.json")
        atomic_json(path, result)
        print(f"{result['status'].upper()} {index}", flush=True)
    return 1 if failures else 0


def finalise_pending(args):
    """Assemble TGP datasets whose last subject finished in a concurrent shard."""
    if args.plan is None or not (Path(args.out) / "work").exists():
        return
    for task in datasets_from_plan(args.plan):
        path = task_path(args.out, task, "tgp_coda")
        if is_complete(path) or not tgp_subject_path(args.out, task, 0).exists():
            continue
        truth = simulate(SimulationConfig(**task["simulation"]))
        try:
            row = assemble_tgp(truth, task, args)
        except Exception as error:
            atomic_json(path, dict(status="failed", dataset_id=task["dataset_id"], method="tgp_coda",
                                   error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc()))
            continue
        if row is not None:
            atomic_json(path, dict(status="complete", row=row))


def summarise(args):
    finalise_pending(args)
    rows, failures = [], []
    for path in sorted((Path(args.out) / "tasks").glob("*.json")):
        value = read_json(path) or dict(status="failed", error=f"unreadable {path.name}")
        (rows if value.get("status") == "complete" else failures).append(value.get("row", value))
    subject_parts = len(list((Path(args.out) / "work").glob("*/tgp_coda/subject_*.npz")))
    atomic_json(Path(args.out) / "metrics.json", rows)
    keys = ("scenario", "condition", "method")
    metrics = ("comp_log_rmse", "comp_log_mae", "comp_log_bias", "comp_pearson",
               "comp_spearman", "comp_slope", "comp_trajectory_r", "comp_bray_curtis",
               "comp_aitchison", "fit_seconds", "wall_seconds", "cpu_hours_alloc", "cpu_hours_used")
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    summary = []
    for key, group in sorted(groups.items()):
        record = dict(zip(keys, key), n=len(group), n_seeds=len({r["seed"] for r in group}))
        for metric in metrics:
            values = [r[metric] for r in group if isinstance(r.get(metric), (int, float))]
            record[metric] = float(np.mean(values)) if values else None
            record[metric + "_se"] = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None
        summary.append(record)
    atomic_json(Path(args.out) / "summary.json", summary)
    status = dict(complete=len(rows), failed=len(failures), tgp_subject_fits=subject_parts, failures=failures)
    if args.plan is not None:
        datasets = datasets_from_plan(args.plan)
        done = {(r["dataset_id"], r["method"]) for r in rows}
        status["plan_expected"] = {m: len(datasets) for m in args.methods}
        status["plan_complete"] = {m: sum((d["dataset_id"], m) in done for d in datasets) for m in args.methods}
    atomic_json(Path(args.out) / "status.json", status)
    print(json.dumps({k: v for k, v in status.items() if k != "failures"}, indent=2))


def add_fit_args(p):
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--seed-offset", type=int, default=100000)
    p.add_argument("--set", action="append", default=[], metavar="METHOD.KEY=VALUE",
                   help="override one adapter setting, e.g. --set microtensor.rank=3")
    for name in ("gemelli", "microtensor", "embed", "parafac2", "mallard"):
        p.add_argument(f"--{name}-root", type=Path,
                       help=f"vendored {name} source, used to record its pinned commit")
    p.add_argument("--tgp-warmup", type=int, default=1000)
    p.add_argument("--tgp-samples", type=int, default=1000)
    p.add_argument("--tgp-chains", type=int, default=4)
    p.add_argument("--tgp-basis", type=int, default=10)
    p.add_argument("--tgp-max-subjects", type=int, default=0, help="lite runs: fit only the first N subjects")
    p.add_argument("--luminate-root", type=Path)
    p.add_argument("--luminate-python", default=sys.executable)
    p.add_argument("--luminate-timeout", type=int, default=72000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("count", help="number of schedulable units (TGP-CODA counts subjects)")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--tgp-max-subjects", type=int, default=0)
    p = commands.add_parser("run")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--wave", type=int, default=0,
                   help="zero-based wave of serial jobs per shard")
    p.add_argument("--waves", type=int, default=1,
                   help="number of sequential waves; use the same value for every wave")
    add_fit_args(p)
    p = commands.add_parser("summarise")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--plan", type=Path, help="also assemble finished TGP datasets and report plan completeness")
    add_fit_args(p)
    args = parser.parse_args()
    if args.command == "count":
        print(len(job_list(datasets_from_plan(args.plan), args.methods, args.tgp_max_subjects)))
        return 0
    if args.command == "summarise":
        summarise(args)
        return 0
    if "luminate" in args.methods and args.luminate_root is None:
        parser.error("--luminate-root is required when running LUMINATE")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
