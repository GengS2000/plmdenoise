"""Turn the raw window fit directories into one CSV per site x window x seed.

Run this on the server after the array finishes. Each fit directory holds about 20 MB
(fit.pt checkpoint, posterior draws, intervals for six components); the analysis needs
only the posterior means on the observed samples. This writes, under --destination:

    <site>/<window>_seed<seed>.csv.gz   one long table per fit, one row per sample x taxon
    <site>/taxa.csv                     taxonomy of the site's features (shared by all windows)
    task_index.csv                      one row per fit: status, usable, timing, memory, file

Columns of a fit table:

    sample_id, subject_id, visit, time, depth     sample keys (time in days, depth = library size)
    taxon_id
    raw_count                                     the observed count, the raw comparator
    total                                         denoised log positive abundance
    baseline, common, individual, event           its components; total = sum of the four
    composition                                   denoised relative abundance (posterior mean)
    absence_probability                           posterior probability the taxon is absent

``--intervals`` adds total_lower/total_upper. Row centring ``total`` within a sample
gives the denoised CLR. Values are written with 6 significant digits.

Only fits whose done.json says usable=true are organised unless --include-unusable is
given; every fit, organised or not, is listed in task_index.csv. Existing outputs are
skipped, so the script can be rerun while the array is still finishing.
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
SAMPLE_KEYS = ("sample_id", "subject_id", "visit", "time", "depth")


def split_dataset(dataset):
    """GINGIVITIS_saliva__W3 -> ('saliva', 'W3')."""
    site, window = dataset.rsplit("__", 1)
    return site.removeprefix("GINGIVITIS_"), window


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def describe(path):
    """Index row for one fit directory, whatever state it is in."""
    done = read_json(path / "done.json")
    if done:
        task = done["identity"]["task"]
        status, usable = done.get("status", "unknown"), bool(done.get("usable"))
    else:
        # 30_GINGIVITIS_saliva__W0_noevent_2026 -> recover the task from the directory name.
        _, rest = path.name.split("_", 1)
        dataset, arm, seed = rest.rsplit("_", 2)
        task = dict(dataset=dataset, arm=arm, seed=int(seed))
        status = "failed" if (path / "failure.json").exists() else (
            "running" if (path / "running.json").exists() else "missing")
        usable = False
    site, window = split_dataset(task["dataset"])
    diag = read_json(path / "diagnostics.json")
    quality = diag.get("quality", {})
    best = quality.get("vi_validation_first_last_best") or [np.nan] * 3
    return dict(task_dir=path.name, dataset=task["dataset"], site=site, window=window,
                seed=int(task["seed"]), arm=task["arm"], status=status, usable=usable,
                fit_hours=diag.get("fit_seconds", np.nan) / 3600 if diag else np.nan,
                peak_rss_mib=diag.get("peak_rss_mib_linux", np.nan),
                vi_best_neg_elbo_per_cell=best[2], vi_rollbacks=quality.get("vi_rollbacks", np.nan),
                csv="", n_rows=0)


def fit_table(path, intervals):
    """Long table of one fit: one row per observed sample x taxon."""
    with np.load(path / "components.npz", allow_pickle=False) as d:
        sample_id, taxon_id = d["sample_id"].astype(str), d["taxon_id"].astype(str)
        values = {name: d[f"{name}_mean"] for name in COMPONENTS}
        if intervals:
            values.update(total_lower=d["total_lower"], total_upper=d["total_upper"])
    with np.load(path / "posterior_observed.npz", allow_pickle=False) as d:
        if not (np.array_equal(d["sample_id"].astype(str), sample_id)
                and np.array_equal(d["taxon_id"].astype(str), taxon_id)):
            raise ValueError(f"{path.name}: posterior_observed order differs from components")
        raw = d["raw_counts"].astype(np.int64)
        values.update(composition=d["composition_mean"], absence_probability=d["absence_probability_mean"])

    identity = values["total"] - sum(values[n] for n in ("baseline", "common", "individual", "event"))
    if np.abs(identity).max() > 1e-3:
        raise ValueError(f"{path.name}: component identity broken ({np.abs(identity).max():.2e})")

    samples = pd.read_csv(path / "samples.tsv", sep="\t", dtype={"sample_id": str, "subject_id": str})
    samples = samples.set_index("sample_id", drop=False).loc[sample_id, list(SAMPLE_KEYS)]
    n_s, n_t = len(sample_id), len(taxon_id)
    table = samples.loc[samples.index.repeat(n_t)].reset_index(drop=True)
    table["taxon_id"] = np.tile(taxon_id, n_s)
    table["raw_count"] = raw.ravel()
    for name, value in values.items():
        table[name] = np.asarray(value, dtype=np.float64).ravel()  # sample-major, matches repeat/tile
    return table


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True, help="the run directory passed to event_run.py --out")
    p.add_argument("--destination", type=Path, required=True, help="organised output directory (created)")
    p.add_argument("--intervals", action="store_true", help="also write total_lower / total_upper")
    p.add_argument("--no-gzip", action="store_true", help="write plain .csv (about 4x larger)")
    p.add_argument("--include-unusable", action="store_true", help="organise fits not marked usable too")
    p.add_argument("--overwrite", action="store_true", help="rewrite tables that already exist")
    p.add_argument("--tar", action="store_true", help="also pack <destination>.tar for download")
    p.add_argument("--from-smoke", action="store_true", help=argparse.SUPPRESS)  # local testing only
    a = p.parse_args()

    source = a.out.resolve() / ("smoke" if a.from_smoke else "tasks")
    destination = a.destination.resolve()
    if not source.is_dir():
        raise SystemExit(f"No fit directories under {source}")
    destination.mkdir(parents=True, exist_ok=True)
    suffix = ".csv" if a.no_gzip else ".csv.gz"

    rows = []
    for path in sorted(q for q in source.iterdir() if q.is_dir()):
        row = describe(path)
        rows.append(row)
        if not (row["usable"] or (a.include_unusable and (path / "components.npz").exists())):
            print(f"skip     {path.name:<48} {row['status']}", flush=True)
            continue
        site_dir = destination / row["site"]
        site_dir.mkdir(exist_ok=True)
        target = site_dir / f"{row['window']}_seed{row['seed']}{suffix}"
        row["csv"] = str(target.relative_to(destination)).replace("\\", "/")
        taxa_file = site_dir / "taxa.csv"
        if not taxa_file.exists():
            taxa = pd.read_csv(path / "taxa.tsv", sep="\t", dtype=str)
            taxa.drop(columns=["representative_sequence"], errors="ignore").to_csv(taxa_file, index=False)
        if target.exists() and not a.overwrite:
            row["n_rows"] = -1  # present from an earlier run, not recounted
            print(f"exists   {row['csv']}", flush=True)
            continue
        table = fit_table(path, a.intervals)
        partial = target.with_name(target.name + ".partial")
        table.to_csv(partial, index=False, float_format="%.6g",
                     compression="gzip" if not a.no_gzip else None)
        partial.replace(target)  # never leave a truncated table under the final name
        row["n_rows"] = len(table)
        print(f"wrote    {row['csv']:<34} rows={len(table):>7} "
              f"{target.stat().st_size / 2**20:6.1f} MiB", flush=True)

    index = pd.DataFrame(rows).sort_values(["site", "window", "seed"])
    index.to_csv(destination / "task_index.csv", index=False, float_format="%.6g")
    organised = index.csv.ne("").sum()
    print(f"\n{organised}/{len(index)} fits organised under {destination}")
    not_ready = index[~index.usable]
    if len(not_ready):
        print(f"{len(not_ready)} fit(s) not usable yet: "
              + ", ".join(f"{r.site}/{r.window}/{r.seed}={r.status}" for r in not_ready.itertuples()))

    if a.tar:
        archive = destination.with_name(destination.name + ".tar")  # tables are already gzipped
        with tarfile.open(archive, "w") as tar:
            tar.add(destination, arcname=destination.name)
        print(f"archive  {archive}  {archive.stat().st_size / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
