"""Collect finished window fits into one compact per-dataset file for local analysis.

Run this on the server once the array has finished. For every window dataset it
averages the posterior component means over seeds and writes a single npz holding
everything the downstream day-14 prediction analysis needs:

    subject_id, time, depth, sample_id, taxon_id   sample and feature keys
    raw_counts                                     the raw comparator (int32)
    total, baseline, common, individual, event     seed-averaged log-scale components
    total_seed_sd                                  across-seed dispersion of ``total``

``total`` is the denoised log positive-abundance value; the published analyses row
centre it to obtain the denoised CLR. The component identity of the frozen model is
``total = baseline + common + individual + event``.

Only fits whose done.json reports usable=true are exported; anything else is listed
in the index with its status so a hole is visible rather than silent.
"""
import sys

sys.dont_write_bytecode = True

import argparse
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

COMPONENTS = ("total", "baseline", "common", "individual", "event")


def task_dirs(out):
    return sorted(p for p in (out / "tasks").iterdir() if p.is_dir()) if (out / "tasks").is_dir() else []


def read_task(path):
    """Return (dataset, seed, status, usable) for one task directory."""
    done = path / "done.json"
    if not done.exists():
        failed = (path / "failure.json").exists()
        return None, None, "failed" if failed else "missing", False
    d = json.loads(done.read_text(encoding="utf-8"))
    task = d["identity"]["task"]
    return task["dataset"], int(task["seed"]), d.get("status", "unknown"), bool(d.get("usable"))


def export_dataset(dataset, paths, destination):
    """Average the component means of one dataset over its seeds."""
    stacks = {name: [] for name in COMPONENTS}
    reference = None
    for path in paths:
        with np.load(path / "components.npz", allow_pickle=False) as d:
            sample_id = d["sample_id"].astype(str)
            taxon_id = d["taxon_id"].astype(str)
            values = {name: d[f"{name}_mean"].astype(np.float32) for name in COMPONENTS}
        if reference is None:
            reference = (sample_id, taxon_id)
        elif not (np.array_equal(reference[0], sample_id) and np.array_equal(reference[1], taxon_id)):
            raise ValueError(f"{dataset}: seeds disagree on sample or taxon order")
        for name, value in values.items():
            stacks[name].append(value)

    sample_id, taxon_id = reference
    means = {name: np.mean(stacks[name], axis=0) for name in COMPONENTS}
    total_sd = (np.std(stacks["total"], axis=0, ddof=1).astype(np.float32)
                if len(stacks["total"]) > 1 else np.zeros_like(means["total"]))

    identity = means["total"] - sum(means[name] for name in ("baseline", "common", "individual", "event"))
    residual = float(np.abs(identity).max())

    samples = pd.read_csv(paths[0] / "samples.tsv", sep="\t", dtype={"sample_id": str, "subject_id": str})
    samples = samples.set_index(samples.sample_id.astype(str)).loc[sample_id]
    with np.load(paths[0] / "posterior_observed.npz", allow_pickle=False) as d:
        if not np.array_equal(d["sample_id"].astype(str), sample_id):
            raise ValueError(f"{dataset}: posterior_observed sample order differs from components")
        raw_counts = d["raw_counts"].astype(np.int32)

    np.savez_compressed(
        destination / f"{dataset}.npz",
        sample_id=sample_id, taxon_id=taxon_id,
        subject_id=samples.subject_id.to_numpy(dtype=str),
        time=samples.time.to_numpy(dtype=float), depth=samples.depth.to_numpy(dtype=float),
        raw_counts=raw_counts, total_seed_sd=total_sd,
        seeds=np.array([int(p.name.rsplit("_", 1)[-1]) for p in paths]),
        **means)
    return dict(dataset=dataset, n_seeds=len(paths), n_samples=len(sample_id), n_taxa=len(taxon_id),
                max_component_identity_residual=residual,
                median_total_seed_sd=float(np.median(total_sd)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True, help="the run directory passed to event_run.py --out")
    p.add_argument("--destination", type=Path, required=True, help="export directory (created)")
    p.add_argument("--bundle", type=Path, default=None, help="bundle, to copy its manifest alongside")
    p.add_argument("--tar", action="store_true", help="also write <destination>.tar.gz for download")
    a = p.parse_args()

    out, destination = a.out.resolve(), a.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)

    found, status_rows = {}, []
    for path in task_dirs(out):
        dataset, seed, status, usable = read_task(path)
        status_rows.append(dict(task_dir=path.name, dataset=dataset, seed=seed,
                                status=status, usable=usable))
        if usable:
            found.setdefault(dataset, []).append(path)

    rows = []
    for dataset in sorted(found):
        rows.append(export_dataset(dataset, sorted(found[dataset]), destination))
        print(f"exported {dataset:<42} seeds={rows[-1]['n_seeds']} "
              f"samples={rows[-1]['n_samples']:>4} taxa={rows[-1]['n_taxa']:>4} "
              f"identity_residual={rows[-1]['max_component_identity_residual']:.2e}", flush=True)

    pd.DataFrame(status_rows).to_csv(destination / "task_status.tsv", sep="\t", index=False)
    pd.DataFrame(rows).to_csv(destination / "windows_index.tsv", sep="\t", index=False)
    if a.bundle:
        for name in ("WINDOW_MANIFEST.json", "plan.json"):
            source = a.bundle.resolve() / name
            if source.exists():
                (destination / name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    incomplete = [r for r in status_rows if not r["usable"]]
    print(f"\n{len(rows)} datasets exported to {destination}")
    if incomplete:
        print(f"{len(incomplete)} task(s) not usable; see task_status.tsv")

    if a.tar:
        archive = destination.with_suffix(destination.suffix + ".tar.gz")
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(destination, arcname=destination.name)
        print(f"archive {archive}")


if __name__ == "__main__":
    main()
