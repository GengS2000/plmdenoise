"""Frozen scenario-7 PCA/PERMANOVA study with full-refit subject bootstrap."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
import sys
import time
import traceback

from run_da import atomic_json, atomic_npz, digest, file_hash, artifact_done, finish

ROOT = Path(__file__).resolve().parent
SOURCE_FILES = (
    "run_dr.py", "dr_metrics.py", "plot_dr.py", "DR_SERVER_GUIDE.md",
    "requirements_dr.txt", "build_dr_bundle.py", "tests/test_dr.py", "tests/check_dr_slurm.sh",
    "slurm/submit_dr.sh", "slurm/dr_array.sbatch", "slurm/dr_merge.sbatch",
    "run_da.py", "r_process.py", "da_design.py", "simulate_microbiome.py", "event_aware_denoise.py",
)


def prepare(args):
    from da_design import simulation_config
    from event_aware_denoise import DenoiseConfig
    if not math.isfinite(args.effect) or args.effect <= 0 or args.replicates < 2 or args.bootstraps < 2:
        raise ValueError("Positive effect and at least two independent/bootstrap replicates required")
    if not 0 < args.alpha < 1 or not math.isfinite(args.pseudocount) or args.pseudocount <= 0:
        raise ValueError("Invalid alpha/pseudocount")
    if args.permutations < 19 or args.seed < 0 or args.seed + args.replicates >= 2**31:
        raise ValueError("Need >=19 permutations and valid seed range")
    if not 2 <= args.components < min(args.subjects, args.taxa):
        raise ValueError("Invalid PCA dimension")
    if args.subjects % 2 or round(.8 * args.subjects) % 2 or args.taxa % 4 or args.visits < 6:
        raise ValueError("Need balanced exposure/group cells, taxa divisible by 4, >=6 visits")
    sim = asdict(simulation_config(0, n_subjects=args.subjects, n_visits=args.visits, n_taxa=args.taxa))
    fit = asdict(DenoiseConfig(adaptive_budget=True, joint_rank=4, hierarchical_observation=True,
                              flexible_event=True, map_steps=600, map_starts=2, vi_steps=2000,
                              posterior_samples=400, early_stopping=True, verbose=False))
    fit.pop("seed")
    if args.smoke:
        fit.update(adaptive_budget=False, map_steps=2, map_starts=1, vi_steps=2,
                   posterior_samples=4, early_stopping=False)
    tasks = []
    for label, strength in (("null", 0.), ("effect", args.effect)):
        for rep in range(args.replicates):
            tasks.append(dict(task_id=f"{label}_mc_{rep:04d}", phase="mc", condition=label,
                              strength=strength, seed=args.seed + rep, replicate=rep))
        for rep in range(args.bootstraps):
            tasks.append(dict(task_id=f"{label}_boot_{rep:04d}", phase="bootstrap", condition=label,
                              strength=strength, seed=args.seed, replicate=rep,
                              parent=f"{label}_mc_0000", resample_seed=args.seed + 100000 + rep))
    plan = dict(schema="dr_v1", simulation=sim, fit=fit, tasks=tasks, smoke=args.smoke,
                effect=args.effect, replicates=args.replicates, bootstraps=args.bootstraps,
                components=args.components, pseudocount=args.pseudocount,
                permutations=args.permutations, alpha=args.alpha,
                methods=["naive", "m4_flex_abundance"], denoise_group="blind",
                bootstrap="subject blocks within group x exposure; full refit; fixed fit seed",
                inference="independent datasets only; group after exposure; within-exposure label permutations",
                endpoint="time-average CLR per subject, PCA without taxon variance scaling",
                effect_selection=("0.3 maximises prior LinDA complete F1 gain over naive (0.4899047169)"
                                  if args.effect == .3 else "explicit user-specified effect override"),
                source_hashes={name: file_hash(ROOT / name) for name in SOURCE_FILES})
    plan["plan_id"] = digest(plan)
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use an empty NEW OUT; resume uses the frozen source, not prepare")
    out.mkdir(parents=True, exist_ok=True)
    for name in SOURCE_FILES:
        dest = out / "source" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    atomic_json(out / "plan.json", plan)
    print(f"Prepared {2*args.replicates} independent + {2*args.bootstraps} full-refit bootstrap tasks; no jobs run.")


def load_plan(out):
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    if digest({k: v for k, v in plan.items() if k != "plan_id"}) != plan["plan_id"]:
        raise ValueError("Plan checksum mismatch")
    if ROOT != out / "source":
        raise ValueError("After prepare, execute OUT/source/run_dr.py")
    for name, checksum in plan["source_hashes"].items():
        if file_hash(out / "source" / name) != checksum:
            raise ValueError(f"Frozen source changed: {name}")
    return plan


def load_npz(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as saved:
        return {k: saved[k] for k in saved.files}


def run_task(out, plan, task, threads):
    import numpy as np
    import torch
    import scipy
    import sklearn
    from simulate_microbiome import SimulationConfig
    from da_design import simulate_da, abundance_composition
    from event_aware_denoise import DenoiseConfig, EventAwareDenoiser
    from dr_metrics import bootstrap_indices, representation, stability, METHODS

    directory = out / "tasks" / task["task_id"]
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "status.json"
    if artifact_done(marker, plan["plan_id"]):
        return
    started = time.monotonic()
    torch.set_num_threads(threads)
    bootstrap = task["phase"] == "bootstrap"
    dependencies = []
    if bootstrap:
        parent = out / "tasks" / task["parent"]
        if not artifact_done(parent / "status.json", plan["plan_id"]):
            raise RuntimeError("Bootstrap parent incomplete; resume MC stage first")
        saved = load_npz(parent / "data.npz")
        exposure = np.isfinite(saved["event_times"][:, 0]).astype(int)
        indices = bootstrap_indices(saved["group"], exposure, task["resample_seed"])
        truth = {key: value[indices] for key, value in saved.items()}
        dependencies = [parent / "status.json", parent / "data.npz"] + [parent / f"{m}.npz" for m in METHODS]
    else:
        sim = simulate_da(SimulationConfig(**(plan["simulation"] | dict(seed=task["seed"]))), task["strength"])
        truth = {key: sim[key] for key in ("Y", "times", "depth", "batch", "event_times", "group", "mask", "Z_true")}
        indices = np.arange(len(truth["group"]))
    # Each duplicate gets a NEW model subject axis; raw time/taxon structure is unchanged.
    atomic_npz(directory / "data.npz", truth)
    atomic_npz(directory / "resample.npz", dict(original_subject=indices))
    y = truth["Y"].astype(float)
    if np.any(y.sum(-1) <= 0):
        raise ValueError("All-zero visit cannot be closed")
    inputs = dict(naive=y / y.sum(-1, keepdims=True))
    estimator = EventAwareDenoiser(DenoiseConfig(**(plan["fit"] | dict(seed=task["seed"]))))
    estimator.fit(truth["Y"], truth["times"], truth["depth"], truth["batch"],
                  truth["event_times"], None, truth["mask"])
    posterior = estimator.posterior()
    inputs["m4_flex_abundance"] = abundance_composition(posterior["Z_mean"])
    atomic_npz(directory / "posterior.npz", {"Z_mean": posterior["Z_mean"]})
    z = truth["Z_true"]
    true_clr = z - z.mean(-1, keepdims=True)  # exact target; no truth-based preprocessing choices
    exposure = np.isfinite(truth["event_times"][:, 0]).astype(int)
    records, files = [], [directory / name for name in ("data.npz", "resample.npz", "posterior.npz")]
    for method, p in inputs.items():
        metrics, arrays = representation(p, true_clr, truth["group"], exposure,
                                        components=plan["components"], pseudocount=plan["pseudocount"],
                                        seed=task["seed"], permutations=plan["permutations"],
                                        inference=not bootstrap, has_effect=task["strength"] > 0)
        if bootstrap:
            reference = load_npz(parent / f"{method}.npz")
            extra, ids, aligned = stability(arrays, reference, indices)
            metrics.update(extra)
            arrays.update(stability_subject_ids=ids, stability_aligned_scores2=aligned)
        atomic_npz(directory / f"{method}.npz", arrays)
        files.append(directory / f"{method}.npz")
        records.append(dict(task_id=task["task_id"], phase=task["phase"], condition=task["condition"],
                            strength=task["strength"], replicate=task["replicate"], seed=task["seed"],
                            method=method, **metrics))
    atomic_json(directory / "metrics.json", records)
    files.append(directory / "metrics.json")
    finish(marker, plan, "complete", files, dependencies=dependencies, seconds=time.monotonic()-started,
           versions=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                         torch=torch.__version__, sklearn=sklearn.__version__), threads=threads,
           converged=bool(estimator.metadata_.get("map_stable") and estimator.metadata_.get("vi_stable")),
           diagnostics=estimator.metadata_, task=task)


def write_rows(path, rows):
    import csv
    fields = sorted(set().union(*(row.keys() for row in rows))) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summary(out, plan):
    import numpy as np
    from dr_metrics import METHODS, wilson
    rows, coverage = [], []
    for task in plan["tasks"]:
        d = out / "tasks" / task["task_id"]
        complete = artifact_done(d / "status.json", plan["plan_id"])
        status = json.loads((d / "status.json").read_text(encoding="utf-8")) if (d / "status.json").exists() else {}
        coverage.append(dict(task_id=task["task_id"], phase=task["phase"], condition=task["condition"],
                             complete=complete, status="complete" if complete else status.get("status", "missing"),
                             converged=status.get("converged", False), error=status.get("error", "")))
        if complete:
            records = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
            for record in records:
                record["converged"] = bool(status.get("converged"))
            rows.extend(records)
    result = out / "summary"
    result.mkdir(exist_ok=True)
    write_rows(result / "metrics.csv", rows)
    write_rows(result / "coverage.csv", coverage)
    inference, ranges, paired = [], [], []
    metadata = {"task_id", "phase", "condition", "strength", "replicate", "seed", "method", "converged"}
    for phase in ("mc", "bootstrap"):
        planned = plan["replicates"] if phase == "mc" else plan["bootstraps"]
        for condition in ("null", "effect"):
            for scope in ("all_complete", "converged_only"):
                subset = [r for r in rows if r["phase"] == phase and r["condition"] == condition
                          and (scope == "all_complete" or r["converged"])]
                for method in METHODS:
                    group = [r for r in subset if r["method"] == method]
                    if phase == "mc":
                        k = sum(r["permanova_p"] <= plan["alpha"] for r in group)
                        n = len(group)
                        low, high = wilson(k, n)
                        rate = k / n if n else None
                        inference.append(dict(condition=condition, method=method, scope=scope,
                                              estimand="type1_error" if condition == "null" else "power",
                                              planned=planned, valid=n, rejections=k, rate=rate,
                                              mcse=math.sqrt(rate*(1-rate)/n) if n else None,
                                              wilson95_low=low, wilson95_high=high,
                                              missing_outcome_lower=k/planned,
                                              missing_outcome_upper=(k+planned-n)/planned))
                    keys = sorted(set().union(*(r.keys() for r in group)) - metadata) if group else []
                    for key in keys:
                        vals = [r[key] for r in group if r.get(key) is not None and np.isfinite(r[key])]
                        if not vals:
                            continue
                        v = np.asarray(vals)
                        ranges.append(dict(phase=phase, condition=condition, scope=scope, method=method,
                                           metric=key, n=len(v), mean=float(v.mean()),
                                           sd=float(v.std(ddof=1)) if len(v)>1 else None,
                                           mean_mcse=float(v.std(ddof=1)/np.sqrt(len(v))) if len(v)>1 else None,
                                           percentile2_5=float(np.quantile(v, .025)), percentile97_5=float(np.quantile(v, .975)),
                                           interval="conditional bootstrap percentile stability range" if phase == "bootstrap" else "independent simulation distribution range"))
                by_method = [{r["task_id"]: r for r in subset if r["method"] == m} for m in METHODS]
                differences = {}
                for tid in sorted(set(by_method[0]) & set(by_method[1])):
                    a, b = by_method[0][tid], by_method[1][tid]
                    for key in set(a) - metadata:
                        if a.get(key) is not None and b.get(key) is not None and np.isfinite(a[key]) and np.isfinite(b[key]):
                            differences.setdefault(key, []).append(b[key] - a[key])
                    if phase == "mc":
                        differences.setdefault("rejection_rate", []).append(int(b["permanova_p"] <= plan["alpha"]) - int(a["permanova_p"] <= plan["alpha"]))
                for key, vals in differences.items():
                    v = np.asarray(vals)
                    # Across independent simulations: paired bootstrap CI for MEAN improvement.
                    # Within subject bootstrap: percentile range of refitted paired differences.
                    dist = np.random.default_rng(9031).choice(v, size=(2000, len(v)), replace=True).mean(1) if phase == "mc" else v
                    paired.append(dict(phase=phase, condition=condition, scope=scope, metric=key, n=len(v),
                                       difference="m4_flex_abundance minus naive", mean=float(v.mean()),
                                       low95=float(np.quantile(dist,.025)), high95=float(np.quantile(dist,.975)),
                                       interval="paired simulation-bootstrap mean CI" if phase == "mc" else "conditional subject-bootstrap percentile range"))
    write_rows(result / "inference.csv", inference)
    write_rows(result / "ranges.csv", ranges)
    write_rows(result / "paired.csv", paired)
    atomic_json(result / "status.json", dict(plan_id=plan["plan_id"], smoke=plan["smoke"],
                planned=len(plan["tasks"]), complete=sum(r["complete"] for r in coverage),
                converged=sum(r["complete"] and r["converged"] for r in coverage),
                incomplete=[r["task_id"] for r in coverage if not r["complete"]],
                note="Bootstrap p values/rejection rates deliberately not computed. Ranges conditional on the first MC dataset per condition."))
    from plot_dr import plot
    plot(out, rows, inference, plan)
    print(f"Summary: {sum(r['complete'] for r in coverage)}/{len(coverage)} complete; {result}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--effect", type=float, default=.3)
    p.add_argument("--replicates", type=int, default=100)
    p.add_argument("--bootstraps", type=int, default=100)
    p.add_argument("--seed", type=int, default=7100)
    p.add_argument("--subjects", type=int, default=40)
    p.add_argument("--visits", type=int, default=12)
    p.add_argument("--taxa", type=int, default=500)
    p.add_argument("--components", type=int, default=5)
    p.add_argument("--pseudocount", type=float, default=1e-6)
    p.add_argument("--permutations", type=int, default=999)
    p.add_argument("--alpha", type=float, default=.05)
    p.add_argument("--smoke", action="store_true", help="2-step fit for engineering checks ONLY")
    for name in ("count", "run", "summarize", "check"):
        p = sub.add_parser(name)
        p.add_argument("--out", type=Path, required=True)
        if name in ("count", "run"):
            p.add_argument("--phase", choices=("mc", "bootstrap"), required=True)
        if name == "run":
            p.add_argument("--shard", type=int, default=0)
            p.add_argument("--shards", type=int, default=1)
            p.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
        return
    out = args.out.resolve()
    plan = load_plan(out)
    if args.command == "count":
        print(sum(t["phase"] == args.phase for t in plan["tasks"]))
    elif args.command == "check":
        import numpy, scipy, torch, sklearn, matplotlib, threadpoolctl
        print(json.dumps({m.__name__: m.__version__ for m in (numpy, scipy, torch, sklearn, matplotlib, threadpoolctl)}))
        print(f"Verified source and plan {plan['plan_id']}; smoke={plan['smoke']}")
    elif args.command == "summarize":
        summary(out, plan)
    else:
        if not 0 <= args.shard < args.shards or args.threads < 1:
            raise ValueError("Invalid shard/thread settings")
        from threadpoolctl import threadpool_limits
        tasks = [t for t in plan["tasks"] if t["phase"] == args.phase][args.shard::args.shards]
        failed = 0
        for task in tasks:
            try:
                with threadpool_limits(limits=args.threads):
                    run_task(out, plan, task, args.threads)
            except Exception as error:
                failed += 1
                atomic_json(out / "tasks" / task["task_id"] / "status.json",
                            dict(plan_id=plan["plan_id"], status="failed", error=str(error), traceback=traceback.format_exc()))
                print(f"FAILED {task['task_id']}: {error}", file=sys.stderr)
        if failed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
