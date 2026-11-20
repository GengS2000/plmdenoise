"""Refit lower-interproximal M2 and rank ASVs by three-class coefficient norms."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

import predict_full_microbiome as model
import predict_phenotype as old


SITE = "lower_jaw_interproximal"
WINDOWS = old.WINDOWS[:4]
CLASSES = ["Low", "Mid", "High"]
TUNING_SEED = 9100


def subject_draws(y, reps, seed):
    """Keep class counts fixed and sample whole subject rows with replacement."""
    rng = np.random.default_rng(seed)
    draws = []
    for c in old.CLASSES:
        indices = np.flatnonzero(y == c)
        draws.append(rng.choice(indices, size=(reps, len(indices)), replace=True))
    return np.sort(np.concatenate(draws, axis=1), axis=1)


def coefficient_norms(Xc, Xm, y, rows, weight, alpha, n_visits, verify=False):
    """Recover the primal coefficients of the original single-site dual fit."""
    kc, _, km, _ = model.blocks(Xc, [Xm], rows, rows)
    targets = np.eye(len(CLASSES))[y[rows]]
    dual = np.linalg.solve(kc + weight * km + alpha * np.eye(len(rows)),
                           targets - targets.mean(axis=0))
    A = Xm[rows] - Xm[rows].mean(axis=0)
    norm = max(float(np.mean(np.sum(A * A, axis=1))), 1e-12)
    beta = weight * (A.T @ dual) / norm
    if verify:
        np.testing.assert_allclose(A @ beta, weight * km @ dual, atol=1e-10)
    return np.linalg.norm(beta, axis=1).reshape(n_visits, -1).sum(axis=0)


def run_window(window, Xc, Xm, y, draws, taxa, out):
    days = old.WIN_DAYS[window]
    full = np.arange(len(y))
    inner = list(StratifiedKFold(3, shuffle=True, random_state=TUNING_SEED).split(Xc, y))
    _, weight, alpha, auc = model.tune_predict(Xc, [Xm], y, full, full, inner)
    observed = coefficient_norms(Xc, Xm, y, full, weight, alpha, len(days), verify=True)
    norms = np.empty((len(draws), len(taxa)))
    for b, rows in enumerate(draws):
        norms[b] = coefficient_norms(Xc, Xm, y, rows, weight, alpha, len(days))
    if not np.isfinite(norms).all() or not np.isfinite(observed).all():
        raise ValueError(f"{window}: non-finite ASV norms")

    pd.DataFrame(norms, columns=taxa, index=pd.RangeIndex(1, len(draws) + 1, name="bootstrap")) \
        .to_csv(out / f"{window}_bootstrap_norms.tsv.gz", sep="\t")
    low, median, high = np.quantile(norms, [0.025, 0.5, 0.975], axis=0)
    summary = pd.DataFrame(dict(asv=taxa, full_norm=observed, bootstrap_median=median,
                                bootstrap_q025=low, bootstrap_q975=high))
    summary["full_rank"] = summary.full_norm.rank(ascending=False, method="min").astype(int)
    summary.sort_values(["full_rank", "asv"]).to_csv(out / f"{window}_asv_summary.tsv",
                                                    sep="\t", index=False)
    print(f"{window}: {len(draws)} refits, {len(taxa)} ASVs, weight={weight:g}, alpha={alpha:g}",
          flush=True)
    return dict(days=days, n_asvs=len(taxa), micro_weight=weight, alpha=alpha, tuning_auc=auc)


def run(out, reps, seed):
    clin, labels = old.load_clinical()
    config = dict(site=SITE, model="M2", label="Rfplaque_formers_day14", classes=CLASSES,
                  source="existing centered mean log abundance of denoising seeds",
                  denoising_seeds=old.SEEDS, bootstrap_reps=reps, bootstrap_seed=seed,
                  sampling="paired across windows, stratified by class, with replacement by subject",
                  tuning=dict(folds=3, seed=TUNING_SEED, alphas=model.ALPHAS, weights=model.WEIGHTS,
                              fit="once per window on the full cohort, fixed across bootstrap refits"),
                  norm="sum across visits of the L2 norm of unscaled Low/Mid/High coefficients",
                  windows={})
    subjects = None
    for window in WINDOWS:
        meta, mats = old.load_window(SITE, window)
        window_subjects = sorted(set(meta.subject_id) & set(labels.index))
        if subjects is None:
            subjects = window_subjects
            y = labels.loc[subjects].to_numpy()
            draws = subject_draws(y, reps, seed)
        elif subjects != window_subjects:
            raise ValueError(f"{window}: subject cohort differs from {WINDOWS[0]}")
        taxa = mats["mean"].columns
        Xc = old.clinical_features(clin, subjects, window).to_numpy()
        Xm, imputed = old.flatten(meta, mats["mean"], subjects, old.WIN_DAYS[window])
        config["windows"][window] = run_window(window, Xc, Xm, y, draws, taxa, out)
        config["windows"][window]["imputed_visits"] = imputed

    config["subjects"] = dict(zip(subjects, [CLASSES[c] for c in y]))
    config["class_counts"] = dict(zip(CLASSES, np.bincount(y).tolist()))
    counts = np.zeros((reps, len(subjects)), dtype=int)
    np.add.at(counts, (np.arange(reps)[:, None], draws), 1)
    pd.DataFrame(counts, columns=subjects, index=pd.RangeIndex(1, reps + 1, name="bootstrap")) \
        .to_csv(out / "bootstrap_subject_counts.tsv.gz", sep="\t")
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=73091)
    parser.add_argument("--out", type=Path, default=old.HERE / "results_bootstrap_m2_asv_norms")
    args = parser.parse_args()
    if args.reps < 1 or args.seed < 0:
        parser.error("--reps must be positive and --seed must be nonnegative")
    out = args.out.resolve()
    if out.exists():
        parser.error(f"output already exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".bootstrap_m2_", dir=out.parent) as temp:
        staged = Path(temp) / "results"
        staged.mkdir()
        with threadpool_limits(limits=1):
            run(staged, args.reps, args.seed)
        staged.rename(out)
    print(f"Results written to {out}", flush=True)


if __name__ == "__main__":
    main()
