"""Paired comparison: v3.1 fits vs every external method that ran, on one plan.

Only the shared estimand (relative composition, comp_* metrics, identical
scorer and log floor) is compared.  Methods that could not run on a dataset are
absent rather than imputed, so every table also reports coverage: how many
datasets each method finished, and on which applicability domain.  Three strata
are reported besides the overall table:

* `complete_grid` -- datasets with a shared visit grid and no missing visit,
  the domain every method here supports;
* `irregular` -- datasets with subject-specific times or missing visits, which
  only some methods accept;
* `common_subset` -- the datasets every requested method finished, which is the
  only place the word "paired" applies to all of them at once.

Uncertainty across simulation replicates:

* mean, SD and SE over seeds for every scenario x condition x method;
* percentile bootstrap 95% CI of the mean, resampling seeds;
* paired differences (competitor - v3.1) on the datasets both methods finished;
  correlated conditions are averaged within seed before the bootstrap CI,
  per scenario x condition, per scenario and overall. The better fraction is
  also averaged within seed.

The observed-proportion baseline ("naive", scored inside every v3.1 task) is
included as a reference.  The array index maps each dataset to its npz files:
OURS/demo/<task_id>.npz and EXTERNAL/arrays/<dataset_id>_<method>.npz.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from external_benchmarks import atomic_json, dataset_id, read_json, score_composition
from simulate_microbiome import SimulationConfig, simulate

METRICS = ("comp_log_rmse", "comp_log_mae", "comp_log_bias", "comp_pearson", "comp_spearman",
           "comp_slope", "comp_trajectory_r", "comp_bray_curtis", "comp_aitchison")
# Orientation used for "competitor better" fractions.
LOWER_BETTER = {"comp_log_rmse", "comp_log_mae", "comp_bray_curtis", "comp_aitchison"}
ABS_LOWER_BETTER = {"comp_log_bias"}
ONE_BETTER = {"comp_slope"}


def badness(metric, value):
    if metric in LOWER_BETTER:
        return value
    if metric in ABS_LOWER_BETTER:
        return abs(value)
    if metric in ONE_BETTER:
        return abs(value - 1)
    return -value


def domain(simulation):
    """Applicability stratum of a dataset, from its generating settings alone.

    `complete_grid` means a shared visit grid with every visit observed: the
    domain microTensor, EMBED and MALLARD are defined on.  Anything else is
    `irregular` and is only attempted by the methods that accept it.
    """
    irregular = bool(simulation.get("irregular"))
    missing = float(simulation.get("missing_rate") or 0) + float(simulation.get("dropout_rate") or 0)
    return "irregular" if irregular or missing > 0 else "complete_grid"


def load_rows(ours, external, reference):
    plan = json.loads((ours / "plan.json").read_text(encoding="utf-8"))
    rows, index, expected = [], {}, {}
    for task in plan["tasks"]:
        key = dataset_id(task["simulation"])
        expected[key] = dict(scenario=task["scenario"], condition=task["condition"],
                             seed=task["seed"], domain=domain(task["simulation"]))
        entry = index.setdefault(key, dict(dataset_id=key, **expected[key]))
        result = read_json(ours / "tasks" / f"{task['task_id']}.json")
        demo = ours / "demo" / f"{task['task_id']}.npz"
        entry[f"{task['variant']}_task_id"] = task["task_id"]
        entry[f"{task['variant']}_npz"] = str(demo) if demo.exists() else ""
        if not result or result.get("status") != "complete":
            continue
        for row in [result["row"]] + [r for r in result.get("baselines", []) if r["method"] in reference]:
            if row["method"] in reference and any(r["dataset_id"] == key and r["method"] == row["method"] for r in rows):
                continue  # baselines repeat across variants of one dataset
            rows.append(dict(expected[key], dataset_id=key, method=row["method"],
                             fit_seconds=row.get("fit_seconds"), converged=row.get("converged"),
                             **{m: row.get(m) for m in METRICS}))
    failures = []
    if external is not None:
        for path in sorted((external / "tasks").glob("*.json")):
            value = read_json(path)
            if not value:
                continue
            if value.get("status") != "complete":
                if value.get("dataset_id") in expected:
                    failures.append(dict(expected[value["dataset_id"]],
                                         dataset_id=value["dataset_id"],
                                         method=value.get("method"),
                                         error=value.get("error"),
                                         workdir=value.get("workdir"), task=str(path)))
                continue
            if value["row"]["dataset_id"] not in expected:
                continue
            row = value["row"]
            key = row["dataset_id"]
            rows.append(dict(expected[key], dataset_id=key, method=row["method"],
                             fit_seconds=row.get("fit_seconds"), rank=row.get("rank"),
                             method_status=row.get("method_status"),
                             tgp_rhat_max=row.get("tgp_rhat_max"),
                             tgp_ess_min=row.get("tgp_ess_min"),
                             tgp_divergences=row.get("tgp_divergences"),
                             **{m: row.get(m) for m in METRICS}))
            npz = external / "arrays" / f"{key}_{row['method']}.npz"
            index[key][f"{row['method']}_npz"] = str(npz) if npz.exists() else ""
    return rows, list(index.values()), failures


def coverage(rows, index, failures, methods):
    """Per method and domain: datasets finished, attempted and failed."""
    domains = {}
    for entry in index:
        domains.setdefault(entry["domain"], set()).add(entry["dataset_id"])
    done = {}
    for row in rows:
        done.setdefault((row["method"], row["domain"]), set()).add(row["dataset_id"])
    failed = {}
    for failure in failures:
        failed.setdefault((failure["method"], failure["domain"]), set()).add(failure["dataset_id"])
    out = []
    seen = sorted({row["method"] for row in rows} | {f["method"] for f in failures} | set(methods))
    for method in seen:
        for name, keys in sorted(domains.items()):
            complete = done.get((method, name), set())
            broke = failed.get((method, name), set()) - complete
            out.append(dict(method=method, domain=name, datasets=len(keys),
                            complete=len(complete), failed=len(broke),
                            not_attempted=len(keys) - len(complete) - len(broke),
                            success_fraction=len(complete) / len(keys) if keys else None))
    return out


def common_datasets(rows, methods):
    """Dataset ids where every requested method produced a scored row."""
    have = {}
    for row in rows:
        have.setdefault(row["dataset_id"], set()).add(row["method"])
    return {key for key, found in have.items() if set(methods) <= found}


def rescore_subjects(rows, ours, external, n_subjects, ours_method):
    """Re-score every method on the first n_subjects of each dataset from saved arrays.

    Used by lite runs, where TGP-CODA fits only a few subjects: all methods are
    then compared on exactly the same cells. v3.1 and LUMINATE were still fit on
    all subjects (they pool across subjects by design); only scoring is restricted.
    """
    plan = json.loads((ours / "plan.json").read_text(encoding="utf-8"))
    sources = {}
    for task in plan["tasks"]:
        if task["variant"] == ours_method:
            sources[dataset_id(task["simulation"])] = task
    kept = []
    for key, task in sources.items():
        dataset_rows = [r for r in rows if r["dataset_id"] == key]
        if not dataset_rows:
            continue
        truth = simulate(SimulationConfig(**task["simulation"]))
        scored = truth["mask"].copy()
        scored[n_subjects:] = False
        truth = truth | {"mask": scored}
        for row in dataset_rows:
            method = row["method"]
            if method == ours_method:
                path = ours / "demo" / f"{task['task_id']}.npz"
                name = "X_mean"
            elif method == "naive":
                path, name = None, None
            else:
                path, name = external / "arrays" / f"{key}_{method}.npz", "comp_mean"
            if path is None:
                estimate = np.where(truth["mask"], truth["Y"], 0) / truth["depth"][:, :, None]
            elif path.exists():
                with np.load(path) as saved:
                    estimate = saved[name].astype(float)
            else:
                continue  # no arrays: cannot be compared on the subset
            info = dict(task, dataset_id=key)
            new = score_composition(truth, estimate, method, info, row.get("fit_seconds"))
            kept.append(row | {m: new.get(m) for m in METRICS} | dict(scored_subjects=n_subjects))
    return kept


def bootstrap_ci(values, rng, reps):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return None, None
    means = values[rng.integers(0, len(values), size=(reps, len(values)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def describe(values, rng, reps):
    values = [v for v in values if isinstance(v, (int, float)) and np.isfinite(v)]
    if not values:
        return dict(n=0, mean=None, sd=None, se=None, ci_lo=None, ci_hi=None)
    sd = float(np.std(values, ddof=1)) if len(values) > 1 else None
    lo, hi = bootstrap_ci(values, rng, reps)
    return dict(n=len(values), mean=float(np.mean(values)), sd=sd,
                se=sd / np.sqrt(len(values)) if sd is not None else None, ci_lo=lo, ci_hi=hi)


def summarise_methods(rows, rng, reps):
    groups = {}
    for row in rows:
        groups.setdefault((row["scenario"], row["condition"], row["method"]), []).append(row)
    out = []
    for (scenario, condition, method), group in sorted(groups.items()):
        for metric in METRICS + ("fit_seconds",):
            out.append(dict(scenario=scenario, condition=condition, method=method, metric=metric,
                            **describe([r.get(metric) for r in group], rng, reps)))
    return out


def paired(rows, ours_method, rng, reps):
    by_key = {(r["dataset_id"], r["method"]): r for r in rows}
    competitors = sorted({r["method"] for r in rows} - {ours_method})
    datasets = {r["dataset_id"]: r for r in rows}
    out = []
    levels = [("scenario_condition", lambda d: (d["scenario"], d["condition"])),
              ("scenario", lambda d: (d["scenario"], "all")),
              ("domain", lambda d: (d.get("domain", "all"), "all")),
              ("overall", lambda d: ("all", "all"))]
    for level, grouping in levels:
        cells = {}
        for key, info in datasets.items():
            cells.setdefault(grouping(info), []).append(key)
        for (scenario, condition), keys in sorted(cells.items(), key=lambda kv: str(kv[0])):
            for competitor in competitors:
                both = [k for k in keys if (k, ours_method) in by_key and (k, competitor) in by_key]
                for metric in METRICS:
                    by_seed = {}
                    for k in both:
                        c, o = by_key[(k, competitor)].get(metric), by_key[(k, ours_method)].get(metric)
                        if not (isinstance(c, (int, float)) and isinstance(o, (int, float))
                                and np.isfinite(c) and np.isfinite(o)):
                            continue
                        seed = datasets[k]["seed"]
                        by_seed.setdefault(seed, []).append((c, o))
                    # Each simulated seed is one independent unit. Its correlated
                    # conditions are averaged before the seed bootstrap.
                    diffs = [float(np.mean([c - o for c, o in pairs])) for pairs in by_seed.values()]
                    stats = describe(diffs, rng, reps)
                    better = [np.mean([badness(metric, c) < badness(metric, o)
                                       for c, o in pairs]) for pairs in by_seed.values()]
                    out.append(dict(level=level, scenario=scenario, condition=condition,
                                    competitor=competitor, reference=ours_method, metric=metric,
                                    **{f"diff_{k}": v for k, v in stats.items()},
                                    n_datasets=sum(map(len, by_seed.values())),
                                    competitor_better_fraction=float(np.mean(better)) if better else None))
    return out


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def markdown(summary, comparisons, ours_method, metric="comp_log_rmse", cover=(), notes=()):
    fmt = lambda v: "" if v is None else f"{v:.3f}"
    lines = [f"# Method comparison ({metric})", ""]
    if cover:
        lines += ["## Coverage", "",
                  "Methods that could not run on a dataset are absent, never imputed.", "",
                  "| method | domain | datasets | complete | failed | not attempted |",
                  "|---|---|---|---|---|---|"]
        for c in cover:
            lines.append(f"| {c['method']} | {c['domain']} | {c['datasets']} | {c['complete']} | "
                         f"{c['failed']} | {c['not_attempted']} |")
        lines.append("")
    if notes:
        lines += ["## Notes", ""] + [f"- {n}" for n in notes] + [""]
    lines += ["## Per scenario x condition", "",
              "mean ± SE over seeds [bootstrap 95% CI of the mean]", "",
              "| scenario | condition | method | n | mean ± SE | 95% CI |", "|---|---|---|---|---|---|"]
    for r in summary:
        if r["metric"] == metric:
            lines.append(f"| {r['scenario']} | {r['condition']} | {r['method']} | {r['n']} | "
                         f"{fmt(r['mean'])} ± {fmt(r['se'])} | [{fmt(r['ci_lo'])}, {fmt(r['ci_hi'])}] |")
    lines += ["", f"## Paired difference (competitor − {ours_method}); positive = {ours_method} better for RMSE", "",
              "| level | scenario | condition | competitor | n | mean diff ± SE | 95% CI | competitor better |",
              "|---|---|---|---|---|---|---|---|"]
    for r in comparisons:
        if r["metric"] == metric and r["level"] != "scenario_condition":
            lines.append(f"| {r['level']} | {r['scenario']} | {r['condition']} | {r['competitor']} | {r['diff_n']} | "
                         f"{fmt(r['diff_mean'])} ± {fmt(r['diff_se'])} | [{fmt(r['diff_ci_lo'])}, {fmt(r['diff_ci_hi'])}] | "
                         f"{fmt(r['competitor_better_fraction'])} |")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ours", type=Path, required=True, help="v3.1 OUT with plan.json and tasks/")
    p.add_argument("--external", type=Path, help="external_benchmarks.py store")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ours-method", default="m4_flex")
    p.add_argument("--reference", nargs="*", default=["naive"], help="v3.1 in-task baselines to include")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260919)
    p.add_argument("--score-subjects", type=int, default=0,
                   help="lite runs: score all methods on the first N subjects only (from saved arrays)")
    p.add_argument("--expect", nargs="*", default=[],
                   help="methods that should be present; the common subset is taken over these "
                        "plus --ours-method, and missing ones are reported, not silently dropped")
    p.add_argument("--require-arrays", action="store_true",
                   help="fail if any completed scored method lacks its readable raw arrays")
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    rows, index, failures = load_rows(args.ours, args.external, set(args.reference))
    if args.require_arrays:
        entries = {entry["dataset_id"]: entry for entry in index}
        for row in rows:
            method = row["method"]
            if method in args.reference:
                continue  # computed from the saved observation, not a separate fit
            path = entries[row["dataset_id"]].get(f"{method}_npz")
            if not path:
                raise FileNotFoundError(f"{row['dataset_id']} {method}: missing raw arrays")
            with np.load(path, allow_pickle=False) as saved:
                required = {"Y", "clean_composition", "comp_mean"}
                if not required <= set(saved.files):
                    raise ValueError(f"{path}: missing {required - set(saved.files)}")
                for name in required:
                    if saved[name].shape != saved["Y"].shape:
                        raise ValueError(f"{path}: {name} has wrong shape")
    if args.score_subjects:
        rows = rescore_subjects(rows, args.ours, args.external, args.score_subjects, args.ours_method)
    args.out.mkdir(parents=True, exist_ok=True)
    summary = summarise_methods(rows, rng, args.bootstrap)
    comparisons = paired(rows, args.ours_method, rng, args.bootstrap)
    wanted = sorted(set(args.expect) | {args.ours_method}) if args.expect else \
        sorted({row["method"] for row in rows})
    common = common_datasets(rows, wanted)
    subset = [row for row in rows if row["dataset_id"] in common]
    for record in paired(subset, args.ours_method, rng, args.bootstrap):
        if record["level"] == "overall":
            comparisons.append(dict(record, level="common_subset"))
    cover = coverage(rows, index, failures, wanted)
    notes = [f"common subset: {len(common)} of {len(index)} datasets carry every method in "
             f"{wanted}; the `common_subset` rows are the only fully paired comparison",
             f"{len(failures)} recorded external failures (see comparison_failures.csv)"]
    write_csv(args.out / "coverage.csv", cover)
    write_csv(args.out / "comparison_failures.csv", failures)
    write_csv(args.out / "comparison_long.csv", rows)
    write_csv(args.out / "comparison_summary.csv", summary)
    write_csv(args.out / "comparison_paired.csv", comparisons)
    write_csv(args.out / "array_index.csv", index)
    if args.external is not None:
        paths = sorted({Path(value).relative_to(args.external).as_posix() for entry in index
                        for key, value in entry.items()
                        if key.endswith("_npz") and key != f"{args.ours_method}_npz"
                        and value and Path(value).is_relative_to(args.external)})
        (args.out / "external_arrays.list").write_text(
            "".join(path + "\n" for path in paths), encoding="utf-8")
    counts = {}
    for row in rows:
        counts[row["method"]] = counts.get(row["method"], 0) + 1
    missing = [m for m in wanted if m not in counts]
    atomic_json(args.out / "comparison.json", dict(datasets=len(index), rows_per_method=counts,
                                                   bootstrap=args.bootstrap, seed=args.seed,
                                                   scored_subjects=args.score_subjects or "all",
                                                   expected_methods=wanted, missing_methods=missing,
                                                   common_subset=sorted(common), coverage=cover,
                                                   failures=failures,
                                                   summary=summary, paired=comparisons))
    (args.out / "comparison.md").write_text(
        markdown(summary, comparisons, args.ours_method, cover=cover, notes=notes),
        encoding="utf-8", newline="\n")
    print(json.dumps(dict(datasets=len(index), rows_per_method=counts,
                          missing_methods=missing, common_subset=len(common),
                          failures=len(failures)), indent=2))


if __name__ == "__main__":
    main()
