"""Run the simulation matrix: simulate -> denoise -> score, one row per method.

One factor moves at a time away from the reference cell, so every row is
comparable to `base`. Datasets are regenerated from their config rather than
stored, the simulator being deterministic given a seed. Results append to
rows.jsonl, so an interrupted run resumes where it stopped.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import time
import traceback
import warnings

import numpy as np
import torch

from evaluate import SUMMARY, evaluate, format_table
from event_aware_denoise import DenoiseConfig, EventAwareDenoiser, legacy_config
from simulate_microbiome import SCENARIOS, SimulationConfig, simulate

AXES = {
    "signal": [("depth_2000", {"median_depth": 2000.}), ("depth_500", {"median_depth": 500.})],
    "robust": [("dropout_0.3", {"dropout_rate": .3}), ("outlier_0.1", {"outlier_rate": .1}),
               ("batch_drift_0.5", {"batch_drift": .5}), ("batch_dispersion", {"batch_dispersion": True}),
               ("overdispersed", {"nb_size": 2.})],
    "design": [("missing_0.2", {"missing_rate": .2}), ("irregular", {"irregular": True}),
               ("confound_partial", {"confounding": "partial"}), ("confound_complete", {"confounding": "complete"})],
}
METHODS = {"denoised": True, "denoised_nb": False}  # label -> zero_inflated


def conditions(axes):
    """The reference cell first; every other cell differs from it in one field."""
    return [("base", {})] + [cell for axis in axes for cell in AXES[axis]]


def run_cell(scenario, name, override, seed, args):
    """Simulate one dataset, fit each requested variant, and score them together.

    Baselines depend only on the truth, so they are emitted once per dataset.
    """
    config = SimulationConfig(scenario=scenario, n_subjects=args.n_subjects, n_visits=args.n_visits,
                              n_taxa=args.n_taxa, seed=seed, **override)
    truth = simulate(config)
    inputs = (truth["Y"], truth["times"], truth["depth"], truth["batch"],
              truth["event_times"], truth["group"], truth["mask"])
    rows = []
    for label in args.methods:
        denoise = legacy_config(n_factors=args.n_factors, zero_inflated=METHODS[label],
                                map_steps=args.map_steps, map_starts=args.map_starts,
                                vi_steps=args.vi_steps, posterior_samples=args.draws,
                                seed=seed, device=args.device, verbose=False)
        started = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # _prepare records them in diagnostics
            result = EventAwareDenoiser(denoise).fit(*inputs).posterior()
        scored = evaluate(truth, result, label=label, alpha=args.alpha, condition=name,
                          extra={"axis_override": json.dumps(override), "fit_seconds": round(time.time() - started, 1)})
        rows += scored if not rows else [r for r in scored if r["method"] == label]
    return rows or evaluate(truth, None, condition=name, extra={"axis_override": json.dumps(override)})


def read_jsonl(path):
    """A torn final line from a shard still appending is skipped, not fatal."""
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def load_rows(out):
    """Every shard part plus any single-process file. A shard resumes from its own
    part alone: having 200 tasks glob and read all parts would hammer the share."""
    return [row for path in [out / "rows.jsonl", *sorted((out / "rows").glob("part_*.jsonl"))]
            for row in read_jsonl(path)]


def summarise(rows, by, columns=SUMMARY):
    """Average each metric over seeds; non-numeric and missing entries are skipped."""
    columns = [c for c in columns if c not in by]
    groups = {}
    for row in rows:
        groups.setdefault(tuple(str(row.get(k)) for k in by), []).append(row)
    out = []
    for key, group in sorted(groups.items()):
        summary = dict(zip(by, key), n=len(group))
        for column in columns:
            values = [r[column] for r in group
                      if isinstance(r.get(column), (int, float)) and np.isfinite(r[column])]
            summary[column] = float(np.mean(values)) if values else float("nan")
        out.append(summary)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "experiments")
    parser.add_argument("--scenarios", type=int, nargs="+", default=[4, 5, 6, 7, 8], choices=list(SCENARIOS))
    parser.add_argument("--axes", nargs="+", default=list(AXES), choices=list(AXES))
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    parser.add_argument("--n-subjects", type=int, default=40)
    parser.add_argument("--n-visits", type=int, default=16)
    parser.add_argument("--n-taxa", type=int, default=80)
    parser.add_argument("--n-factors", type=int, default=3)
    parser.add_argument("--map-steps", type=int, default=300)
    parser.add_argument("--map-starts", type=int, default=2)
    parser.add_argument("--vi-steps", type=int, default=500)
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=.1, help="Target Bayesian FDR")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--summarise-only", action="store_true", help="Re-print tables from the collected rows")
    parser.add_argument("--list-cells", action="store_true", help="Print the cell count and exit (for array sizing)")
    parser.add_argument("--shard", type=int, help="Run cells shard, shard+shards, ... into their own part file")
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    cells = [(s, name, override, seed) for s in args.scenarios
             for name, override in conditions(args.axes) for seed in args.seeds]
    if args.list_cells:
        print(len(cells))
        return
    if args.shard is not None and not 0 <= args.shard < args.shards:
        parser.error("--shard must lie in [0, --shards)")
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    # One file per shard: 200 concurrent tasks appending to one file interleave.
    if args.shard is None:
        path = args.out / "rows.jsonl"
    else:
        (args.out / "rows").mkdir(exist_ok=True)
        path = args.out / "rows" / f"part_{args.shard:05d}.jsonl"
        cells = cells[args.shard::args.shards]
    rows = read_jsonl(path) if args.shard is not None else load_rows(args.out)
    done = {(r["scenario"], r["condition"], r["seed"], r["method"]) for r in rows if r.get("method") != "failed"}
    todo = [c for c in cells if any((c[0], c[1], c[3], method) not in done for method in args.methods)]
    if not args.summarise_only:
        print(f"{len(cells)} cells, {len(todo)} to run, {len(args.methods)} fit(s) each", flush=True)
        for index, (scenario, name, override, seed) in enumerate(todo, 1):
            started = time.time()
            try:
                produced = run_cell(scenario, name, override, seed, args)
            except Exception as error:  # one bad cell must not lose the run
                produced = [{"scenario": scenario, "condition": name, "seed": seed, "method": "failed",
                             "error": f"{type(error).__name__}: {error}"}]
                traceback.print_exc()
            with path.open("a", encoding="utf-8") as handle:
                for row in produced:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows += produced
            elapsed = time.time() - started
            print(f"[{index}/{len(todo)}] scenario {scenario} {name} seed {seed}: "
                  f"{elapsed:.0f}s, eta {(len(todo) - index) * elapsed / 60:.0f}min", flush=True)
    if args.shard is not None:
        print(f"shard {args.shard}/{args.shards} done -> {path}")
        return  # summaries would be partial, and 200 tasks must not race on one file
    rows = load_rows(args.out)
    live = [r for r in rows if r.get("method") != "failed"]
    for by in (("condition", "method"), ("scenario_name", "method")):
        table = format_table(summarise(live, by), None)
        print(f"\n### by {' x '.join(by)}\n{table}")
        (args.out / f"summary_by_{'_'.join(by)}.md").write_text(table + "\n", encoding="utf-8")
    failures = [r for r in rows if r.get("method") == "failed"]
    print(f"\n{len(live)} rows, {len(failures)} failed cells -> {args.out}")


if __name__ == "__main__":
    main()
