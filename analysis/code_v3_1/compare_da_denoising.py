"""Score saved DA denoising outputs; never generate data or fit any model.

Uses the simulation1-8 relative-composition scorer, log floor and metrics.
Truth is read from each effect-added truth.npz, never regenerated from a seed.
comp_* scores against the clean composition (structural zeros included);
abund_* scores the same estimate against exp(Z_true) closed (no zero channel),
the scale on which the DA effect gamma is injected. Oracle inputs are not scored.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from compare_methods import METRICS as COMP_METRICS, badness as comp_badness, describe
from da_design import abundance_composition, fit_source
from da_io import identifiers
from evaluate import _scale_block
from external_benchmarks import score_composition
from run_da import artifact_done, atomic_json, file_hash, load_plan

METHODS = ("naive", "m4_flex", "tempted_gemelli", "microtensor", "embed", "parafac2")
ABUND_METRICS = tuple(m.replace("comp_", "abund_", 1) for m in COMP_METRICS)
METRICS = COMP_METRICS + ABUND_METRICS


def badness(metric, value):
    return comp_badness(metric.replace("abund_", "comp_", 1), value)


def scored_methods(plan):
    """Every non-oracle input of the frozen plan; naive first (the paired reference)."""
    methods = [m for m in plan["inputs"] if not m.startswith("oracle_")]
    if "naive" not in methods:
        raise ValueError("The plan has no naive input to pair against")
    return tuple(["naive"] + [m for m in methods if m != "naive"])


def read_composition(path, shape):
    """Require exact sample/taxon order; a silent reorder would corrupt scoring."""
    samples, taxa = identifiers(shape)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        if next(reader, None) != ["sample_id"] + taxa:
            raise ValueError("Taxon header/order does not match saved truth")
        records = list(reader)
    if len(records) != len(samples):
        raise ValueError("Sample count does not match saved truth")
    if any(len(row) != len(taxa) + 1 or row[0] != sample
           for row, sample in zip(records, samples)):
        raise ValueError("Sample IDs/order or number of taxon columns is invalid")
    values = np.asarray([row[1:] for row in records], dtype=float).reshape(shape)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Composition contains nonfinite or negative entries")
    if not np.allclose(values.sum(axis=-1), 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError("Composition rows must sum to one")
    return values


def write_csv(path, rows, required=()):
    fields = list(required)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def collect(out, plan, methods=METHODS):
    rows, failures, inputs = [], [], []
    for task in plan["tasks"]:
        directory = out / "datasets" / task["task_id"]
        base = dict(dataset_id=task["task_id"], scenario=9, base_scenario=7,
                    condition=f"{task['series']}_s{task['s']}", series=task["series"],
                    s=task["s"], strength=task["strength"], seed=task["seed"])
        try:
            if not artifact_done(directory / "generation.json", plan["plan_id"]):
                raise ValueError("Generation marker/files are missing or fail hash validation")
            with np.load(directory / "truth.npz", allow_pickle=False) as saved:
                truth = {key: saved[key] for key in
                         ("Y", "clean_composition", "mask", "depth", "metadata_json", "Z_true")}
            abundance = abundance_composition(truth["Z_true"])
        except Exception as error:
            failures.extend(dict(base, method=method, error=f"{type(error).__name__}: {error}")
                            for method in methods)
            continue
        truth_hash = file_hash(directory / "truth.npz")
        floor = 1.0 / max(float(np.median(truth["depth"])), 1.0)
        for method in methods:
            path = directory / f"{method}.csv"
            fit = fit_source(method)
            marker = directory / ("generation.json" if fit is None else f"denoise_{fit}.json")
            try:
                if not artifact_done(marker, plan["plan_id"], retry_partial=True):
                    raise ValueError("Completion marker/files are missing or fail hash validation")
                info = json.loads(marker.read_text(encoding="utf-8"))
                # Check that the CSV being scored is actually covered by its marker.
                if path.name not in info["artifacts"]:
                    raise ValueError("Composition CSV is not listed in completion marker")
                estimate = read_composition(path, truth["Y"].shape)
                scored = score_composition(truth, estimate, method, base,
                                           0.0 if method == "naive" else info.get("seconds"))
                scored.update(_scale_block(abundance, estimate, truth["mask"], floor, "abund", True))
                rows.append(dict(scored, **base, converged=info.get("converged")))
                inputs.append(dict(base, method=method, composition=str(path),
                                   composition_sha256=file_hash(path), truth_sha256=truth_hash,
                                   marker=str(marker)))
            except Exception as error:
                failures.append(dict(base, method=method, composition=str(path),
                                     error=f"{type(error).__name__}: {error}"))
    return rows, failures, inputs


def make_reports(rows, plan, rng, reps, methods=METHODS):
    lookup = {(row["dataset_id"], row["method"]): row for row in rows}
    conditions = sorted({f"{t['series']}_s{t['s']}" for t in plan["tasks"]})
    tasks_by_condition = {condition: [t for t in plan["tasks"]
                                     if f"{t['series']}_s{t['s']}" == condition]
                          for condition in conditions}
    summary, paired, deltas, coverage = [], [], [], []
    for condition, tasks in tasks_by_condition.items():
        keys = [task["task_id"] for task in tasks]
        common = {key for key in keys if all((key, m) in lookup for m in methods)}
        for method in methods:
            count = sum((key, method) in lookup for key in keys)
            coverage.append(dict(condition=condition, method=method, expected=len(keys),
                                 scored=count, missing_or_invalid=len(keys) - count,
                                 common_subset=len(common)))
        for scope in ("all_available", "common_subset"):
            for method in methods:
                selected = [lookup[key, method] for key in keys
                            if (key, method) in lookup and
                            (scope == "all_available" or key in common)]
                for metric in METRICS:
                    if not any(metric in r for r in selected):
                        continue
                    summary.append(dict(condition=condition, scope=scope, method=method,
                                        metric=metric, n_datasets=len(selected),
                                        **describe([r[metric] for r in selected], rng, reps)))
        for scope in ("pairwise", "common_subset"):
            for method in methods[1:]:
                for metric in METRICS:
                    values, improvements = [], []
                    for key in keys:
                        if (key, method) not in lookup or (key, "naive") not in lookup:
                            continue
                        if scope == "common_subset" and key not in common:
                            continue
                        candidate, naive = lookup[key, method], lookup[key, "naive"]
                        a, b = candidate.get(metric, np.nan), naive.get(metric, np.nan)
                        if not np.isfinite(a) or not np.isfinite(b):
                            continue
                        delta = a - b
                        improvement = badness(metric, b) - badness(metric, a)
                        values.append(delta)
                        improvements.append(improvement)
                        if scope == "pairwise":
                            deltas.append(dict(dataset_id=key, seed=candidate["seed"],
                                               condition=condition, method=method, reference="naive",
                                               metric=metric, method_value=a, naive_value=b,
                                               delta=delta, improvement=improvement))
                    stats = describe(values, rng, reps)
                    paired.append(dict(condition=condition, scope=scope, method=method,
                                       reference="naive", metric=metric,
                                       **{f"delta_{k}": v for k, v in stats.items()},
                                       improvement_mean=float(np.mean(improvements)) if improvements else None,
                                       better_fraction=float(np.mean(np.asarray(improvements) > 0))
                                       if improvements else None))
    # Conditions sharing a seed are correlated. Keep the three effect conditions
    # separate instead of treating 45 datasets as 45 independent replicates.
    return summary, paired, deltas, coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Existing DA experiment OUT")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261002, help="Bootstrap RNG only")
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    out = args.out.resolve()
    plan = load_plan(out, require_snapshot=False)
    methods = scored_methods(plan)
    report = out / "comparison_denoising"
    report.mkdir(exist_ok=True)
    rows, failures, inputs = collect(out, plan, methods)
    summary, paired, deltas, coverage = make_reports(
        rows, plan, np.random.default_rng(args.seed), args.bootstrap, methods)
    for name, records, required in (
        ("comparison_long.csv", rows, ("dataset_id", "condition", "method")),
        ("comparison_summary.csv", summary, ("condition", "scope", "method", "metric")),
        ("comparison_paired.csv", paired, ("condition", "scope", "method", "metric")),
        ("comparison_paired_deltas.csv", deltas, ("dataset_id", "method", "metric")),
        ("comparison_failures.csv", failures, ("dataset_id", "condition", "method", "error")),
        ("coverage.csv", coverage, ("condition", "method")),
        ("input_index.csv", inputs, ("dataset_id", "method")),
    ):
        write_csv(report / name, records, required)
    source_names = ("compare_da_denoising.py", "compare_methods.py", "external_benchmarks.py",
                    "evaluate.py", "simulate_microbiome.py", "da_io.py", "run_da.py",
                    "r_process.py", "external_method_contract.py", "da_design.py")
    atomic_json(report / "status.json", dict(
        plan_id=plan["plan_id"], expected=len(plan["tasks"]) * len(methods),
        scored=len(rows), missing_or_invalid=len(failures), bootstrap=args.bootstrap,
        bootstrap_seed=args.seed, methods=methods, metrics=METRICS,
        source_hashes={name: file_hash(Path(__file__).resolve().parent / name) for name in source_names},
        estimand="comp_*: effect-added clean relative composition (structural zeros kept); "
                 "abund_*: exp(Z_true) closed, the scale on which gamma is injected",
        log_floor="1 / median saved depth",
        delta="method - naive", improvement="badness(naive) - badness(method); positive means better",
        uncertainty="95% percentile bootstrap over seeds separately within each effect condition"))
    print(f"Scored {len(rows)}/{len(plan['tasks']) * len(methods)} outputs; "
          f"missing/invalid={len(failures)}. Reports: {report}")
    if not rows:
        raise SystemExit("No valid saved outputs were available; inspect comparison_failures.csv")


if __name__ == "__main__":
    main()
