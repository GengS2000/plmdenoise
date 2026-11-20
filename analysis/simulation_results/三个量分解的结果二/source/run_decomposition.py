"""Frozen decomposition study: does m4_flex recover baseline/common/individual/event?

Data and fits are exactly those of simple450: the tasks are taken from
run_validation.make_plan(stage="simple32", variants=["m4_flex"]) unchanged
(450 datasets, seeds 4100-4114, simple32 m4_flex budget, group passed as before).
The generating model is docs/simple_simulation_and_metrics.tex eq. (18):

    Z = alpha_j + b_ij + lambda_j f_it + beta_j q(x_t) + e_i a_j h_j(x_t - tau_i)

whose individual differences are b_ij (baseline), lambda_j f_it (individual
dynamics) and e_i, tau_i (exposure and event time). Only m4_flex is fitted; the
raw data are the reference. See DECOMPOSITION_GUIDE.md.
`prepare --model-factors K` changes only the fitted latent rank (n_factors) of
every task; data, budget and all other m4_flex settings stay as in simple450.

    python run_decomposition.py prepare --out OUT [--model-factors K]
    python OUT/source/run_decomposition.py run --out OUT --shard K --shards N
    python OUT/source/run_decomposition.py summarise --out OUT
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
from types import SimpleNamespace
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE_FILES = (
    "run_decomposition.py", "decomposition.py", "event_aware_denoise.py", "simulate_microbiome.py",
    "evaluate.py", "run_validation.py", "protocol.py", "demo_report.py", "DECOMPOSITION_GUIDE.md",
    "slurm/submit_decomposition.sh", "slurm/decomposition_array.sbatch",
    "slurm/decomposition_merge.sbatch", "tests/test_decomposition.py",
)
VARIANT = "m4_flex"
REQUIRED = ("total_clr_all_rmse_m4", "struct_common_rmse_m4", "struct_individual_rmse_m4",
            "struct_baseline_rmse_m4", "canon_bg_common_rmse_m4")


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def digest(value):
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_hashes(root):
    return {name: hashlib.sha256((Path(root) / name).read_bytes()).hexdigest() for name in SOURCE_FILES}


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


def simple32_tasks(seeds=None, scenarios=None, conditions=None, model_factors=None):
    """The simple450 m4_flex tasks, built by run_validation itself; model_factors
    (run_validation's own override) is the only setting that may differ."""
    import run_validation
    args = SimpleNamespace(stage="simple32", variants=[VARIANT], scenarios=scenarios, seeds=seeds,
                           conditions=conditions, fit_replicates=None, holdout_fraction=None,
                           early_stopping=False, model_factors=model_factors,
                           **{k: None for k in ("n_subjects", "n_visits", "n_taxa", "map_steps", "map_starts",
                                                "vi_steps", "posterior_samples", "event_global_scale",
                                                "zero_global_scale", "event_slab_scale", "gain_scale",
                                                "event_decay", "residual_df")})
    return run_validation.make_plan(args)["tasks"]


def make_plan(args):
    model_factors = getattr(args, "model_factors", None)
    if model_factors is not None and model_factors < 1:
        raise ValueError("--model-factors must be a positive integer")
    tasks = simple32_tasks(args.seeds, args.scenarios, args.conditions, model_factors)
    if args.smoke:
        # Plumbing only: tiny data and a 3-step fit; never mixed with a formal OUT.
        for task in tasks:
            task["simulation"].update(n_subjects=10, n_visits=12, n_taxa=30)
            task["fit"].update(map_steps=3, map_starts=1, vi_steps=3, posterior_samples=8, validation_samples=2)
            task["task_id"] = digest({k: v for k, v in task.items() if k != "task_id"})[:20]
    if not tasks or len({t["task_id"] for t in tasks}) != len(tasks):
        raise ValueError("Empty or duplicated task list")
    body = dict(schema="decomposition_v1", smoke=bool(args.smoke), tasks=tasks,
                source_hashes=source_hashes(ROOT),
                notes=("simple450 data and m4_flex fits (tasks from run_validation simple32, unchanged); "
                       "raw log(Y+0.5) CLR is the reference. Components and inference follow "
                       "real_denoise/event_export.py and cohort_analysis.py."),
                **({} if model_factors is None else dict(model_factors=model_factors)))
    return clean(body | dict(plan_id=digest(body)))


def prepare(args):
    plan = make_plan(args)
    out = args.out.resolve()
    if (out / "plan.json").exists():
        saved = read_plan(out, verify_code=False)
        if saved != plan:
            raise ValueError("OUT holds a different plan or code; choose a new OUT")
    else:
        if out.exists() and any(p for p in out.iterdir() if p.name != "logs"):
            raise ValueError("New OUT must be empty")
        for name in SOURCE_FILES:
            dest = out / "source" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((ROOT / name).read_bytes())
        atomic_json(out / "plan.json", plan)
    print(f"{len(plan['tasks'])} fits; plan {plan['plan_id'][:12]}; {out / 'plan.json'}")
    return plan


def read_plan(out, verify_code=True):
    out = Path(out).resolve()
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    if digest({k: v for k, v in plan.items() if k != "plan_id"}) != plan["plan_id"]:
        raise ValueError("Plan checksum mismatch")
    if source_hashes(out / "source") != plan["source_hashes"]:
        raise ValueError("Frozen source changed; create a new OUT")
    if verify_code and ROOT != out / "source":
        raise ValueError("After prepare, run OUT/source/run_decomposition.py")
    return plan


def completed(path, task_id, plan_id):
    if not path.exists():
        return False
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        row = result.get("row", {})
        return (result.get("status") == "complete" and result.get("task_id") == task_id
                and result.get("plan_id") == plan_id
                and all(isinstance(row.get(k), (int, float)) and np.isfinite(row[k]) for k in REQUIRED))
    except (json.JSONDecodeError, OSError):
        return False


def run_task(task, out, plan_id, threads=1, save_components=False):
    import torch
    from decomposition import clr, evaluate_decomposition, posterior_decomposition, truth_components
    from evaluate import evaluate
    from event_aware_denoise import DenoiseConfig, EventAwareDenoiser
    from run_validation import training_mask
    from simulate_microbiome import SimulationConfig, simulate
    torch.set_num_threads(threads)
    # Same data, mask, inputs and settings as run_validation.run_task for this task.
    truth = simulate(SimulationConfig(**task["simulation"]))
    mask, _ = training_mask(truth, task["holdout_fraction"], task["seed"])
    config = DenoiseConfig(**task["fit"])
    started = time.monotonic()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # _prepare records them in diagnostics
        estimator = EventAwareDenoiser(config).fit(truth["Y"], truth["times"], truth["depth"], truth["batch"],
                                                   truth["event_times"], truth["group"], mask)
        result = estimator.posterior()
    fit_seconds = time.monotonic() - started
    truth = truth | {"mask": mask}
    standard = evaluate(truth, result, label=task["variant"], condition=task["condition"])
    row = next(r for r in standard if r["method"] == task["variant"])
    del result  # free posterior() arrays before the second streaming pass
    truth_clr = {k: clr(v) for k, v in truth_components(truth).items()}
    post = posterior_decomposition(estimator, truth["times"], truth["event_times"], truth_clr)
    decomposition, arrays = evaluate_decomposition(truth, post)
    overlap = set(decomposition) & set(row)
    if overlap:
        raise RuntimeError(f"Metric name clash: {sorted(overlap)[:5]}")
    row.update(decomposition)
    meta = estimator.metadata_
    row.update(task_id=task["task_id"], variant=task["variant"], fit_seconds=fit_seconds,
               total_seconds=time.monotonic() - started, map_stable=meta["map_stable"],
               vi_stable=meta["vi_stable"], map_steps_run=sum(h["steps"] for h in meta["map_histories"]),
               vi_steps_run=meta["vi_steps_run"], vi_recovery_count=len(meta["vi_recoveries"]),
               event_trend_max_corr=row.get("event_trend_max_corr"))
    if save_components:
        target = out / "components" / f"{task['task_id']}.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays, metadata_json=np.array(json.dumps(clean(task))))
        os.replace(temporary, target)
    diagnostics = {k: v for k, v in meta.items() if k not in ("map_histories", "vi_negative_elbo")}
    return dict(status="complete", task_id=task["task_id"], plan_id=plan_id, row=row,
                baselines=[r for r in standard if r["method"] != task["variant"]], diagnostics=diagnostics,
                runtime=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
                             platform=platform.platform(), threads=threads))


def run(args):
    plan = read_plan(args.out)
    if not 0 <= args.shard < args.shards or args.threads < 1:
        raise ValueError("Invalid shard or thread count")
    failures = 0
    for index in range(args.shard, len(plan["tasks"]), args.shards):
        task = plan["tasks"][index]
        path = args.out / "tasks" / f"{task['task_id']}.json"
        components = args.out / "components" / f"{task['task_id']}.npz"
        if completed(path, task["task_id"], plan["plan_id"]) and (not args.save_components or components.exists()):
            continue
        print(f"START {index} {task['task_id']} sc{task['scenario']} {task['condition']} "
              f"seed {task['seed']} {task['variant']}", flush=True)
        try:
            result = run_task(task, args.out, plan["plan_id"], args.threads, args.save_components)
        except Exception as error:
            failures += 1
            result = dict(status="failed", task_id=task["task_id"], plan_id=plan["plan_id"],
                          error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
            print(result["traceback"], file=sys.stderr, flush=True)
        atomic_json(path, result)
        print(f"{result['status'].upper()} {index}", flush=True)
    return 1 if failures else 0


# ----------------------------------------------------------------- summary
KEY_METRICS = (
    # 1. total
    "total_clr_all_rmse_m4", "total_clr_all_rmse_naive", "total_clr_all_rmse_naive_nz",
    "total_clr_all_rmse_delta", "total_clr_all_rmse_delta_nz",
    "total_clr_present_rmse_m4", "total_clr_present_rmse_naive", "total_clr_present_rmse_naive_nz",
    "total_clr_structural_zero_rmse_m4", "total_clr_structural_zero_rmse_naive",
    "total_traj_r_m4", "total_traj_r_naive", "total_traj_r_naive_nz",
    # 2. labelled components
    *[f"{f}_{k}_{s}_{e}" for f in ("struct", "struct_present")
      for k in ("baseline", "common", "individual", "event", "dynamic", "background")
      for s in ("r2", "rmse") for e in ("m4", "naive", "naive_nz")],
    *[f"leak_m4_{a}_from_{b}" for a in ("baseline", "common", "individual", "event")
      for b in ("baseline", "common", "individual", "event")],
    # 3. canonical
    *[f"canon_{f}_{k}_r2_{e}" for f in ("bg", "bg_present", "total") for k in ("baseline", "common", "individual")
      for e in ("m4", "naive", "naive_nz")],
    *[f"label_vs_canon_{k}_rmse_m4" for k in ("baseline", "common", "individual")],
    # 4-5. partition, calibration, coupling
    *[f"vp_{k}_share_{e}" for k in ("baseline", "common", "individual", "event") for e in ("truth", "m4", "naive", "naive_nz")],
    *[f"cov_{k}_m4" for k in ("baseline", "common", "individual", "event", "total")],
    "postcorr_common_individual_median_m4", "postcorr_common_event_median_m4",
    "postcorr_individual_event_median_m4",
    # 6. inference
    "traj_disc_m4_power", "traj_disc_m4_fdp", "traj_disc_naive_power", "traj_disc_naive_fdp",
    "traj_auc_m4", "traj_auc_naive", "traj_curve_pearson_m4",
    "evt_disc_m4_power", "evt_disc_m4_fdp", "evt_disc_m4_n", "evt_disc_naive_power", "evt_disc_naive_fdp",
    "evt_disc_naive_n", "evt_auc_m4", "evt_auc_naive", "evt_curve_pearson_m4",
    "gain_abs_mean_m4", "gain_sd_m4", "gain_subject_amplification_sd_m4", "evt_subject_rms_spearman_m4",
    "indiv_slope_pearson_m4", "indiv_slope_pearson_naive", "indiv_slope_pearson_naive_nz",
    "indiv_subject_rms_spearman_m4", "indiv_subject_rms_spearman_naive", "indiv_subject_rms_spearman_naive_nz",
    # fit health
    "fit_seconds", "map_stable", "vi_stable", "vi_recovery_count", "converged",
)


def numeric(values):
    out = []
    for v in values:
        if isinstance(v, bool):
            out.append(float(v))
        elif isinstance(v, (int, float)) and np.isfinite(v):
            out.append(float(v))
    return out


def aggregate(rows, keys, metrics):
    """Average conditions within a data seed first, then over seeds (seed = unit)."""
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    output = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        record = dict(zip(keys, key), n_fits=len(group), n_seeds=len({r["seed"] for r in group}))
        for metric in metrics:
            per_seed = []
            for seed in sorted({r["seed"] for r in group}):
                vals = numeric([r.get(metric) for r in group if r["seed"] == seed])
                if vals:
                    per_seed.append(float(np.mean(vals)))
            record[metric] = float(np.mean(per_seed)) if per_seed else None
            record[metric + "_se"] = (float(np.std(per_seed, ddof=1) / np.sqrt(len(per_seed)))
                                      if len(per_seed) > 1 else None)
            record[metric + "_n"] = len(per_seed)
        output.append(record)
    return output


def verdicts(summary):
    """Predeclared descriptive screen per component: m4 minus each raw reference,
    seed-level 95% CI, at least 10 seeds."""
    out = []
    for record in summary:
        scopes = ([("total_clr", "all"), ("total_clr", "present")]
                  + [(f, c) for f in ("struct", "struct_present")
                     for c in ("baseline", "common", "individual", "dynamic", "background")]
                  + [(f, c) for f in ("canon_bg", "canon_bg_present") for c in ("baseline", "common", "individual")])
        for family, k in scopes:
            for suffix, reference in (("delta", "naive"), ("delta_nz", "naive_nz")):
                metric = f"{family}_{k}_rmse_{suffix}"
                if not record.get(metric + "_n"):
                    continue  # present-cell scopes exist only where structural zeros occur
                mean, se = record.get(metric), record.get(metric + "_se")
                verdict = "insufficient"
                if mean is not None and se is not None and record[metric + "_n"] >= 10:
                    verdict = ("m4_closer" if mean + 1.96 * se < 0 else
                               f"{reference}_closer" if mean - 1.96 * se > 0 else "not_distinguishable")
                out.append({**{k2: record[k2] for k2 in ("scenario", "condition", "variant") if k2 in record},
                            "metric": metric, "reference": reference, "mean_delta": mean, "se": se,
                            "verdict": verdict})
    return out


def format_table(rows, columns, digits=3):
    def cell(v):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "-"
        return f"{v:.{digits}g}" if isinstance(v, float) else str(v)
    body = [[cell(r.get(c)) for c in columns] for r in rows]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    return "\n".join(lines + ["| " + " | ".join(b) + " |" for b in body])


def report(out, by_scenario, verdict_rows, n_factors=()):
    blocks = [
        ("1. Total: raw vs denoised (CLR RMSE to Z_true)",
         ["total_clr_all_rmse_naive", "total_clr_all_rmse_naive_nz", "total_clr_all_rmse_m4",
          "total_clr_present_rmse_naive", "total_clr_present_rmse_naive_nz", "total_clr_present_rmse_m4",
          "total_traj_r_naive", "total_traj_r_naive_nz", "total_traj_r_m4"]),
        ("2. Labelled components (R2 to generated component)",
         [f"struct_{k}_r2_{e}" for k in ("baseline", "common", "individual", "event") for e in ("naive", "naive_nz", "m4")]),
        ("2a. Same, present cells only (ZINB scenarios; raw data cannot reach Z at structural zeros)",
         [f"struct_present_{k}_r2_{e}" for k in ("baseline", "common", "individual") for e in ("naive", "naive_nz", "m4")]),
        ("2b. Sums robust to label swaps (R2)",
         [f"struct_{k}_r2_{e}" for k in ("dynamic", "background") for e in ("naive", "naive_nz", "m4")]),
        ("2c. Leakage: estimated common / individual regressed on generated parts",
         [f"leak_m4_{a}_from_{b}" for a in ("common", "individual") for b in ("common", "individual", "event")]),
        ("2d. Where the generated event goes (coefficient on generated event)",
         [f"leak_m4_{a}_from_event" for a in ("baseline", "common", "individual", "event")]),
        ("3. Canonical (convention-free) background split, R2",
         [f"canon_bg_{k}_r2_{e}" for k in ("baseline", "common", "individual") for e in ("naive", "naive_nz", "m4")]),
        ("3a. Canonical split, present cells only",
         [f"canon_bg_present_{k}_r2_{e}" for k in ("baseline", "common", "individual") for e in ("naive", "naive_nz", "m4")]),
        ("4. Variance partition shares",
         [f"vp_{k}_share_{e}" for k in ("common", "individual", "event") for e in ("truth", "naive", "naive_nz", "m4")]),
        ("5. 95% coverage (CLR) and posterior coupling",
         [f"cov_{k}_m4" for k in ("baseline", "common", "individual", "event", "total")]
         + ["postcorr_common_individual_median_m4"]),
        ("6a. Population trajectory taxa",
         ["traj_disc_naive_power", "traj_disc_naive_fdp", "traj_disc_m4_power", "traj_disc_m4_fdp",
          "traj_auc_naive", "traj_auc_m4"]),
        ("6b. Event taxa; spurious individual gain (true gain = 0)",
         ["evt_disc_naive_power", "evt_disc_naive_fdp", "evt_disc_m4_power", "evt_disc_m4_fdp",
          "evt_disc_m4_n", "evt_subject_rms_spearman_m4", "gain_abs_mean_m4", "gain_subject_amplification_sd_m4"]),
        ("6c. Individual dynamics (truth: lambda_j f_it)",
         ["indiv_slope_pearson_naive", "indiv_slope_pearson_naive_nz", "indiv_slope_pearson_m4",
          "indiv_subject_rms_spearman_naive", "indiv_subject_rms_spearman_naive_nz", "indiv_subject_rms_spearman_m4"]),
    ]
    lines = ["# Decomposition study: seed-level means by scenario (conditions pooled within seed)", "",
             "naive = raw log(Y+0.5) CLR; naive_nz = zeros treated as missing and filled from subject + visit means; "
             "both use the canonical split and have no event term. Seeds are the "
             "independent unit; see summary_by_condition.json for each condition and *_se columns.", "",
             f"Fitted m4_flex latent rank n_factors = {', '.join(map(str, n_factors)) or '?'}.", ""]
    for title, columns in blocks:
        lines += [f"## {title}", "", format_table(by_scenario, ["scenario", "variant", "n_seeds"] + columns), ""]
    lines += ["## Screen: m4 minus naive RMSE (seed-level 95% CI, by scenario x condition)", "",
              format_table(verdict_rows, ["scenario", "condition", "variant", "metric", "reference", "mean_delta", "se", "verdict"]), ""]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarise(args):
    plan = read_plan(args.out)
    rows, failed, missing = [], [], []
    for index, task in enumerate(plan["tasks"]):
        path = args.out / "tasks" / f"{task['task_id']}.json"
        if completed(path, task["task_id"], plan["plan_id"]):
            row = json.loads(path.read_text(encoding="utf-8"))["row"]
            rows.append(row | dict(scenario=task["scenario"], condition=task["condition"],
                                   seed=task["seed"], variant=task["variant"]))
        elif path.exists():
            failed.append(index)
        else:
            missing.append(index)
    status = dict(plan_id=plan["plan_id"], expected=len(plan["tasks"]), complete=len(rows),
                  failed_indices=failed, missing_indices=missing, all_tasks_complete=not failed and not missing,
                  converged=sum(bool(r.get("converged")) for r in rows),
                  note="Completion is not convergence; m4_flex runs a capped adaptive budget as in simple32.")
    atomic_json(args.out / "status.json", status)
    atomic_json(args.out / "metrics.json", rows)
    if not rows:
        print(json.dumps(status, indent=2))
        return status
    metrics = sorted({k for r in rows for k, v in r.items()
                      if isinstance(v, (int, float)) and k not in ("scenario", "seed")})
    by_condition = aggregate(rows, ["scenario", "condition", "variant"], metrics)
    by_scenario = aggregate(rows, ["scenario", "variant"], metrics)
    overall = aggregate([r | dict(scenario="all") for r in rows], ["scenario", "variant"], metrics)
    atomic_json(args.out / "summary_by_condition.json", by_condition)
    atomic_json(args.out / "summary_by_scenario.json", by_scenario + overall)
    verdict_rows = verdicts(by_condition)
    atomic_json(args.out / "verdicts.json", verdict_rows)
    keys = [m for m in KEY_METRICS if m in metrics]
    import csv
    for name, table in (("summary_by_condition.csv", by_condition), ("summary_by_scenario.csv", by_scenario + overall)):
        columns = ["scenario", "condition", "variant", "n_fits", "n_seeds"] + [x for m in keys for x in (m, m + "_se")]
        with (args.out / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(table)
    report(args.out, by_scenario + overall, verdict_rows, sorted({t["fit"]["n_factors"] for t in plan["tasks"]}))
    print(json.dumps({k: v for k, v in status.items() if not k.endswith("indices")}, indent=2))
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--scenarios", type=int, nargs="+", choices=range(1, 9))
    p.add_argument("--conditions", nargs="+")
    p.add_argument("--smoke", action="store_true", help="tiny data and 3-step fits: plumbing only")
    p.add_argument("--model-factors", type=int, help="fitted latent rank n_factors (default: m4_flex's 3)")
    p = commands.add_parser("run")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--save-components", action="store_true")
    for name in ("summarise", "count"):
        p = commands.add_parser(name)
        p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "run":
        return run(args)
    elif args.command == "summarise":
        summarise(args)
    else:
        print(len(read_plan(args.out, verify_code=False)["tasks"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
