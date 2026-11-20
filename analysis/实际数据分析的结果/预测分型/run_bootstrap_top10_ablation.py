"""Evaluate the fixed bootstrap Top 10 panels with the original M2 CV."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

import predict_full_microbiome as model
import predict_phenotype as old
from bootstrap_m2_asv_norms import SITE, WINDOWS


BOOT = old.HERE / "results_bootstrap_m2_asv_norms"
OUT = old.HERE / "results_bootstrap_top10_ablation"
MODES = ("all", "keep10", "drop10")


def select_panel(window):
    norms = pd.read_csv(BOOT / f"{window}_bootstrap_norms.tsv.gz", sep="\t",
                        index_col="bootstrap")
    full = pd.read_csv(BOOT / f"{window}_asv_summary.tsv", sep="\t", index_col="asv")
    ranks = norms.rank(axis=1, ascending=False, method="min")
    if not ranks.le(10).sum(axis=1).eq(10).all():
        raise ValueError(f"{window}: ties at the bootstrap Top 10 boundary")
    panel = full[["full_rank"]].assign(
        selection_frequency=ranks.le(10).mean(),
        mean_bootstrap_norm=norms.mean(),
        median_bootstrap_rank=ranks.median(),
    ).sort_values(["selection_frequency", "median_bootstrap_rank", "full_rank"],
                  ascending=[False, True, True]).head(10).reset_index()
    return panel.assign(window=window, panel_rank=np.arange(1, 11))


def evaluate(window, panel, clin, labels, repeats):
    meta, mats = old.load_window(SITE, window)
    subjects = sorted(set(meta.subject_id) & set(labels.index))
    y = labels.loc[subjects].to_numpy()
    Xc = old.clinical_features(clin, subjects, window).to_numpy()
    days = old.WIN_DAYS[window]
    clr = mats["mean"]
    Xm, _ = old.flatten(meta, clr, subjects, days)
    tensor = Xm.reshape(len(subjects), len(days), len(clr.columns))
    keep = clr.columns.isin(panel.asv)
    if keep.sum() != 10:
        raise ValueError(f"{window}: missing selected ASV")
    designs = (Xm, tensor[:, :, keep].reshape(len(subjects), -1),
               tensor[:, :, ~keep].reshape(len(subjects), -1))
    rows = []
    for rep in range(repeats):
        outer = StratifiedKFold(5, shuffle=True, random_state=8400 + rep)
        scores = np.empty((len(MODES), len(y), 3))
        for fold, (tr, te) in enumerate(outer.split(Xc, y)):
            inner = list(StratifiedKFold(3, shuffle=True, random_state=9100 + rep * 5 + fold)
                         .split(Xc[tr], y[tr]))
            for i, X in enumerate(designs):
                scores[i, te] = model.tune_predict(Xc, [X], y, tr, te, inner)[0]
        for mode, X, scores_for_mode in zip(MODES, designs, scores):
            rows.append(dict(window=window, ablation=mode, repeat=rep, n=len(y),
                             n_asvs=X.shape[1] // len(days), auc=model.auc(y, scores_for_mode)))
    return pd.DataFrame(rows)


def main():
    original = old.HERE / "results_full_microbiome"
    reference = pd.read_csv(original / "metrics_all.tsv", sep="\t")
    reference = reference[(reference.site == SITE) & (reference.model == "M2")]
    repeats = reference["repeat"].nunique()
    clin, labels = old.load_clinical()
    panels, metrics = [], []
    for window in WINDOWS:
        panel = select_panel(window)
        result = evaluate(window, panel, clin, labels, repeats)
        all_asvs = result[result.ablation == "all"].set_index("repeat").auc
        saved = reference[reference.window == window].set_index("repeat").auc
        np.testing.assert_allclose(all_asvs, saved.reindex(all_asvs.index), atol=1e-12)
        panels.append(panel)
        metrics.append(result)
        print(window, result.groupby("ablation").auc.mean().round(4).to_dict(), flush=True)

    metrics = pd.concat(metrics, ignore_index=True)
    summary = metrics.groupby(["window", "ablation", "n", "n_asvs"], sort=False).auc.mean()
    summary = summary.rename("macro_ovr_auc").reset_index()
    OUT.mkdir(exist_ok=True)
    pd.concat(panels, ignore_index=True).to_csv(OUT / "top10_panels.tsv", sep="\t", index=False)
    metrics.to_csv(OUT / "repeat_metrics.tsv", sep="\t", index=False)
    summary.to_csv(OUT / "summary.tsv", sep="\t", index=False)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
