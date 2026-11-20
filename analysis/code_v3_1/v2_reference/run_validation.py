"""Versioned simulation plans, atomic per-fit results and paired summaries.

prepare creates an immutable plan AND a frozen Python source snapshot. run and
summarise verify that source before touching results. A failed fit never marks
other variants complete. Run --help and see SERVER_GUIDE.md for Slurm usage.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import warnings

import numpy as np
import torch

from event_aware_denoise import DenoiseConfig, EventAwareDenoiser, legacy_config, zinb_log_prob, nb_log_prob
from evaluate import evaluate, SUMMARY, format_table
from simulate_microbiome import SimulationConfig, simulate

SOURCE_FILES = ("event_aware_denoise.py", "simulate_microbiome.py", "evaluate.py", "run_validation.py")
VARIANTS = {
    "legacy": asdict(legacy_config()),
    "no_R": asdict(legacy_config(residual="none")),
    "no_d": asdict(legacy_config(individual_event="none")),
    "no_R_no_d": asdict(legacy_config(residual="none", individual_event="none")),
    "gain_legacy": asdict(legacy_config(residual="none", individual_event="gain", individual_covariance="diagonal")),
    "shrink_basis": asdict(DenoiseConfig(residual="none", individual_event="basis", individual_covariance="full")),
    "compact_no_R": asdict(DenoiseConfig(residual="none")),
    "compact_R": asdict(DenoiseConfig()),
    "legacy_nb": asdict(legacy_config(zero_inflated=False)),
    "compact_R_nb": asdict(DenoiseConfig(zero_inflated=False)),
    "compact_no_R_nb": asdict(DenoiseConfig(residual="none", zero_inflated=False)),
}
BASE_CONDITIONS = {"base": {}, "depth_500": {"median_depth": 500.},
                   "confound_complete": {"confounding": "complete"}}
STRESS_CONDITIONS = {
    "base": {}, "mismatch": {"basis_match": "mismatched", "random_lengths": True},
    "local_trend": {"trend_mode": "local"},
    "both_mismatch": {"basis_match": "mismatched", "trend_mode": "local", "random_lengths": True},
    "visits_6": {"n_visits": 6}, "visits_10": {"n_visits": 10},
    "R_zero": {"residual_multiplier": 0.}, "R_strong": {"residual_multiplier": 3.},
    "hetero_zero": {"heterogeneity": "gain", "hetero_sd": 0.},
    "hetero_high": {"heterogeneity": "gain", "hetero_sd": .5},
    "hetero_shared": {"heterogeneity": "shared", "hetero_sd": .5},
    "event_uncertain": {"event_time_error": .02},
    "events_overlap": {"n_events": 2}, "delayed_subjects": {"subject_delay_sd": .04},
    "irregular_missing": {"irregular": True, "missing_rate": .2},
}
STAGES = {
    "smoke": dict(scenarios=[3, 4, 6, 7], conditions={"base": {}}, seeds=[2026],
                  variants=["legacy", "no_R", "no_d", "no_R_no_d", "compact_no_R", "compact_R"]),
    "ablation": dict(scenarios=[3, 4, 5, 6, 7], conditions=BASE_CONDITIONS,
                     seeds=list(range(2026, 2036)), variants=["legacy", "no_R"]),
    "selection": dict(scenarios=[3, 4, 6, 7], conditions={k: BASE_CONDITIONS[k] for k in ("base", "depth_500")},
                      seeds=list(range(2026, 2031)),
                      variants=["no_R", "no_R_no_d", "gain_legacy", "shrink_basis", "compact_no_R", "compact_R"]),
    "stress": dict(scenarios=[3, 4, 6, 7], conditions=STRESS_CONDITIONS,
                   seeds=list(range(2040, 2050)), variants=["compact_no_R", "compact_R"]),
    "null": dict(scenarios=[1, 7], conditions={k: BASE_CONDITIONS[k] for k in ("base", "depth_500")},
                 seeds=list(range(2100, 2130)), variants=["compact_no_R", "compact_R"]),
    "scale": dict(scenarios=[4, 6], conditions={"base": {}, "T40": {"n_visits": 40},
                  "I100": {"n_subjects": 100}, "J300": {"n_taxa": 300},
                  "I20_T6_J300": {"n_subjects": 20, "n_visits": 6, "n_taxa": 300}},
                  seeds=list(range(2040, 2045)), variants=["compact_no_R", "compact_R"]),
    "observation": dict(scenarios=[1, 6, 7, 8], conditions={k: BASE_CONDITIONS[k] for k in ("base", "depth_500")},
                        seeds=list(range(2040, 2045)), variants=["legacy", "compact_R", "compact_R_nb"]),
}


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def digest(value):
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_hashes(root=None):
    root = Path(root or Path(__file__).parent)
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(clean(value), handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def make_plan(args):
    stage = STAGES[args.stage]
    variants = args.variants or stage["variants"]
    scenarios = args.scenarios or stage["scenarios"]
    seeds = args.seeds or stage["seeds"]
    conditions = stage["conditions"]
    if args.conditions:
        conditions = {name: (BASE_CONDITIONS | STRESS_CONDITIONS | stage["conditions"])[name] for name in args.conditions}
    tasks = []
    for scenario in scenarios:
        for condition, override in conditions.items():
            for seed in seeds:
                sim = asdict(SimulationConfig(scenario=scenario, seed=seed))
                if args.stage == "smoke":
                    sim.update(n_subjects=12, n_visits=10, n_taxa=20)
                sim.update({key: getattr(args, key) for key in ("n_subjects", "n_visits", "n_taxa") if getattr(args, key) is not None})
                sim.update(override)
                for variant in variants:
                    for repeat in range(args.fit_replicates):
                        fit = dict(VARIANTS[variant])
                        # Equal fixed budgets for structural comparisons by default.
                        fit.update(map_steps=300, map_starts=2, vi_steps=500, posterior_samples=200,
                                   early_stopping=False, seed=seed + repeat * 100003, verbose=False)
                        if args.stage == "smoke":
                            fit.update(map_steps=150, map_starts=1, vi_steps=300, posterior_samples=100)
                        if args.stage in {"selection", "stress", "null", "scale", "observation"}:
                            fit.update(map_steps=600, map_starts=2, vi_steps=2000, posterior_samples=400, early_stopping=True)
                        fit.update({key: getattr(args, key) for key in ("map_steps", "map_starts", "vi_steps", "posterior_samples") if getattr(args, key) is not None})
                        if args.early_stopping:
                            fit["early_stopping"] = True
                        for name in ("event_global_scale", "zero_global_scale", "event_slab_scale", "gain_scale", "event_decay", "residual_df"):
                            if getattr(args, name) is not None:
                                fit[name] = getattr(args, name)
                        if args.model_factors is not None:
                            fit["n_factors"] = args.model_factors
                        EventAwareDenoiser(DenoiseConfig(**fit))  # validate before server submission
                        if min(sim[k] for k in ("n_subjects", "n_visits", "n_taxa")) < 2 or seed < 0:
                            raise ValueError("Simulation dimensions must be >=2 and seeds nonnegative")
                        if fit["n_factors"] > min(sim["n_taxa"], sim["n_subjects"] * sim["n_visits"]):
                            raise ValueError("Model rank exceeds simulation dimensions")
                        task = dict(scenario=scenario, condition=condition, seed=seed, fit_repeat=repeat,
                                    variant=variant, simulation=sim, fit=fit,
                                    holdout_fraction=args.holdout_fraction)
                        task["task_id"] = digest(task)[:20]
                        tasks.append(task)
    if len({t["task_id"] for t in tasks}) != len(tasks):
        raise ValueError("Duplicate tasks; remove duplicate scenarios/seeds/variants")
    body = dict(schema_version=2, stage=args.stage, source_hashes=source_hashes(), tasks=tasks,
                notes="Independent atomic fit results; supported post-event ROPE tests, not PIP.")
    return clean(body | {"plan_id": digest(body)})


def prepare(args):
    if args.fit_replicates < 1 or not 0 <= args.holdout_fraction < .8:
        raise ValueError("fit_replicates >=1 and holdout_fraction in [0,.8) required")
    plan = make_plan(args)
    path = args.out / "plan.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != plan:
            raise ValueError("Output directory has a different plan/code. Use a new OUT; existing results were not changed.")
        if source_hashes(args.out / "source") != plan["source_hashes"]:
            raise ValueError("Frozen source changed; refusing to overwrite it")
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        snapshot = args.out / "source"
        snapshot.mkdir(exist_ok=True)
        for name in SOURCE_FILES:
            (snapshot / name).write_bytes((Path(__file__).parent / name).read_bytes())
        atomic_json(path, plan)
    print(f"{len(plan['tasks'])} fits; plan {plan['plan_id'][:12]}; {path}")
    return plan


def read_plan(out, verify=True):
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    if digest({k: v for k, v in plan.items() if k != "plan_id"}) != plan["plan_id"]:
        raise ValueError("Plan checksum mismatch")
    if verify and source_hashes() != plan["source_hashes"]:
        raise ValueError("Code differs from plan; use OUT/source/run_validation.py or create a new plan")
    return plan


def training_mask(truth, fraction, seed):
    mask = truth["mask"].copy()
    heldout = np.zeros_like(mask)
    if fraction:
        rng = np.random.default_rng(np.random.SeedSequence([seed, 314159]))
        for i in range(mask.shape[0]):
            observed = np.flatnonzero(mask[i].any(-1))
            # Interior contiguous visits; retain both endpoints and >=2 visits.
            count = min(max(1, round(fraction * len(observed))), len(observed) - 2)
            if count > 0:
                start = int(rng.integers(1, len(observed) - count))
                heldout[i, observed[start:start + count]] = mask[i, observed[start:start + count]]
        mask[heldout] = False
        # A completely withheld batch has no estimable effect. Keep one visit
        # for every batch (not its count values), rather than leaking its effect.
        for level in np.unique(truth["batch"]):
            locations = (truth["batch"] == level) & truth["mask"].any(-1)
            if locations.any() and not (locations & mask.any(-1)).any():
                i, t = np.argwhere(locations)[0]
                mask[i, t] = truth["mask"][i, t]
                heldout[i, t] = False
    return mask, heldout


@torch.no_grad()
def holdout_score(estimator, truth, result, heldout):
    if not heldout.any():
        return {"heldout_cells": 0}
    model = estimator.model_
    y = torch.as_tensor(np.where(truth["mask"], truth["Y"], 0), dtype=torch.float64, device=model.device)
    pick = torch.as_tensor(heldout, device=model.device)
    devices = [torch.device(model.device).index or 0] if str(model.device).startswith("cuda") else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(estimator.config.seed + 40000)
        scores = []
        for _ in range(64):
            d = model.decode(model.values(True))
            ll = (zinb_log_prob(y, d["log_mu"], d["size"], d["eta_logit"]) if estimator.config.zero_inflated else
                  nb_log_prob(y, d["log_mu"], d["size"]))
            scores.append(ll[pick])
        lp = torch.logsumexp(torch.stack(scores), dim=0) - np.log(len(scores))
    return {"heldout_cells": int(heldout.sum()), "heldout_count_nll": -float(lp.mean()),
            "heldout_Z_rmse": float(np.sqrt(np.mean((result["Z_mean"][heldout] - truth["Z_true"][heldout]) ** 2)))}


def run_task(task, out, plan_id, device="cpu", save_fits=False):
    truth = simulate(SimulationConfig(**task["simulation"]))
    mask, heldout = training_mask(truth, task["holdout_fraction"], task["seed"])
    c = DenoiseConfig(**(task["fit"] | {"device": device}))
    started = time.monotonic()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all diagnostics saved below
        estimator = EventAwareDenoiser(c).fit(truth["Y"], truth["times"], truth["depth"], truth["batch"],
                                             truth["event_times"], truth["group"], mask)
        result = estimator.posterior()
    # Primary recovery is on training observations; heldout results stay separate.
    scoring_truth = truth | {"mask": mask}
    rows = evaluate(scoring_truth, result, label=task["variant"], condition=task["condition"],
                    extra={"fit_repeat": task["fit_repeat"], "task_id": task["task_id"],
                           "fit_seed": c.seed, "fit_seconds": time.monotonic() - started})
    row = next(r for r in rows if r["method"] == task["variant"])
    row.update(holdout_score(estimator, truth, result, heldout))
    if save_fits:
        artifact = out / "fits" / task["task_id"]
        artifact.mkdir(parents=True, exist_ok=True)
        estimator.save_checkpoint(artifact / "fit.pt")
        np.savez_compressed(artifact / "posterior.npz", **result)
    return dict(status="complete", task_id=task["task_id"], plan_id=plan_id, row=row,
                baselines=[r for r in rows if r["method"] != task["variant"]],
                diagnostics=estimator.metadata_, runtime=dict(python=sys.version, torch=torch.__version__,
                numpy=np.__version__, platform=platform.platform(), threads=torch.get_num_threads(), device=device))


def completed(path, task_id, plan_id):
    if not path.exists():
        return False
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        row = result.get("row", {})
        required = ("scenario", "condition", "seed", "fit_repeat", "method", "Z_rmse", "count_log_rmse")
        return (result.get("status") == "complete" and result.get("task_id") == task_id
                and result.get("plan_id") == plan_id and isinstance(row, dict)
                and all(key in row for key in required)
                and all(isinstance(row[key], (int, float)) and np.isfinite(row[key]) for key in ("Z_rmse", "count_log_rmse")))
    except (json.JSONDecodeError, OSError):
        return False


def run(args):
    plan = read_plan(args.out)
    if not 0 <= args.shard < args.shards or args.threads < 1:
        raise ValueError("Invalid shard or thread count")
    torch.set_num_threads(args.threads)
    failures = 0
    for index in range(args.shard, len(plan["tasks"]), args.shards):
        task = plan["tasks"][index]
        path = args.out / "tasks" / (task["task_id"] + ".json")
        if completed(path, task["task_id"], plan["plan_id"]):
            continue
        print(f"START {index} {task['task_id']} sc{task['scenario']} {task['condition']} {task['variant']}", flush=True)
        try:
            result = run_task(task, args.out, plan["plan_id"], args.device, args.save_fits)
        except Exception as error:
            failures += 1
            result = dict(status="failed", task_id=task["task_id"], plan_id=plan["plan_id"],
                          error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
            print(result["traceback"], file=sys.stderr, flush=True)
        atomic_json(path, result)
        print(f"{result['status'].upper()} {index}", flush=True)
    return 1 if failures else 0


def numeric(values):
    return [float(v) for v in values if isinstance(v, (int, float)) and np.isfinite(v)]


def aggregate(rows, keys, metrics):
    """Average optimization repeats within data seed, then independent seeds."""
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    output = []
    for key, group in sorted(groups.items()):
        record = dict(zip(keys, key), n_fits=len(group), n_seeds=len({r["seed"] for r in group}))
        for metric in metrics:
            seed_values = []
            for seed in sorted({r["seed"] for r in group}):
                vals = numeric([r.get(metric) for r in group if r["seed"] == seed])
                if vals:
                    seed_values.append(float(np.mean(vals)))
            record[metric] = float(np.mean(seed_values)) if seed_values else None
            record[metric + "_n"] = len(seed_values)
            record[metric + "_se"] = float(np.std(seed_values, ddof=1) / np.sqrt(len(seed_values))) if len(seed_values) > 1 else None
        output.append(record)
    return output


def summarise(args):
    plan = read_plan(args.out)
    rows, failures, missing, unstable = [], [], [], []
    baselines = {}
    for index, task in enumerate(plan["tasks"]):
        path = args.out / "tasks" / (task["task_id"] + ".json")
        if completed(path, task["task_id"], plan["plan_id"]):
            result = json.loads(path.read_text(encoding="utf-8"))
            rows.append(result["row"])
            for baseline in result.get("baselines", []):
                key = (baseline["scenario"], baseline["condition"], baseline["seed"], baseline["method"])
                baselines.setdefault(key, baseline)
            if not result["row"].get("converged"):
                unstable.append(index)
        elif path.exists():
            failures.append(index)
        else:
            missing.append(index)
    status = dict(plan_id=plan["plan_id"], expected=len(plan["tasks"]), complete=len(rows),
                  failed_indices=failures, missing_indices=missing, unstable_indices=unstable,
                  all_tasks_complete=not failures and not missing,
                  note="Completion is not convergence. Fixed-budget fits often remain unstable; inspect diagnostics before claims.")
    atomic_json(args.out / "status.json", status)
    atomic_json(args.out / "metrics.json", rows)
    metrics = list(dict.fromkeys([m for m in SUMMARY if m != "method"] + ["detect_any_any_false_positive",
                  "detect_zero_fpr_50", "detect_plus_fdr", "heldout_count_nll", "heldout_Z_rmse",
                  "Z_rmse_near_event", "Z_rmse_far_event", "event_plus_r2_vs_fitted_population", "fit_seconds"]))
    keys = ["scenario", "condition", "method"]
    summary = aggregate(rows, keys, metrics)
    atomic_json(args.out / "baseline_summary.json", aggregate(list(baselines.values()), keys, metrics))
    atomic_json(args.out / "summary.json", summary)
    (args.out / "summary.md").write_text(format_table(summary, keys + ["n_seeds"] + [m for m in SUMMARY if m != "method"]) + "\n", encoding="utf-8")
    # Pair variants within scenario, condition, data seed AND optimization repeat.
    paired = []
    reference = args.reference or ("legacy" if any(r["method"] == "legacy" for r in rows) else
                                   "no_R" if any(r["method"] == "no_R" for r in rows) else "compact_no_R")
    ref = {(r["scenario"], r["condition"], r["seed"], r["fit_repeat"]): r for r in rows if r["method"] == reference}
    for row in rows:
        base = ref.get((row["scenario"], row["condition"], row["seed"], row["fit_repeat"]))
        if base is None or row["method"] == reference:
            continue
        delta = {key: row[key] for key in keys + ["seed", "fit_repeat"]}
        for metric in metrics:
            a, b = numeric([row.get(metric)]), numeric([base.get(metric)])
            delta[metric] = a[0] - b[0] if a and b else None
        paired.append(delta)
    atomic_json(args.out / "paired_deltas.json", dict(reference=reference,
                interpretation="candidate minus reference; lower is better for errors, higher for coverage/power",
                rows=aggregate(paired, keys, metrics)))
    print(json.dumps({k: v for k, v in status.items() if not k.endswith("indices")}, indent=2))
    print(f"failed={len(failures)} missing={len(missing)} unstable={len(unstable)}; see status.json")
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--stage", choices=STAGES, default="smoke")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--variants", choices=VARIANTS, nargs="+")
    p.add_argument("--scenarios", type=int, choices=range(1, 9), nargs="+")
    p.add_argument("--conditions", nargs="+")
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--fit-replicates", type=int, default=1)
    p.add_argument("--holdout-fraction", type=float, default=0.)
    for name in ("n_subjects", "n_visits", "n_taxa", "map_steps", "map_starts", "vi_steps", "posterior_samples"):
        p.add_argument("--" + name.replace("_", "-"), type=int)
    p.add_argument("--early-stopping", action="store_true")
    p.add_argument("--model-factors", type=int)
    p.add_argument("--residual-df", type=int)
    for name in ("event_global_scale", "zero_global_scale", "event_slab_scale", "gain_scale", "event_decay"):
        p.add_argument("--" + name.replace("_", "-"), type=float)
    p = commands.add_parser("run")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--device", default="cpu")
    p.add_argument("--save-fits", action="store_true")
    p = commands.add_parser("summarise")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reference")
    p = commands.add_parser("count")
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "run":
        return run(args)
    elif args.command == "summarise":
        summarise(args)
    else:
        print(len(read_plan(args.out)["tasks"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
