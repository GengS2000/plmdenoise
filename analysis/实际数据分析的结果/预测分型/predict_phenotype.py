"""Denoised vs raw microbiome for predicting the day-14 RF-plaque phenotype (GINGIVITIS).

Design: 预测分型方案（修订版）.
  label    Rfplaque_formers_day14 (0/1/2 = Low/Mid/High), subjects = microbiome cohort ∩ RFdata
  windows  W0..W3 (pre-day-14 visits), WF (all 7 visits, upper-bound reference only)
  features clinical (window visits × 5 vars, day ≥ 14 never used) + microbiome PCA (k = 5)
           on the per-subject flattened CLR (visit × taxon); missing visits filled with the
           subject's within-window taxon mean
  models   M0 clinical, M1 clinical + raw PCA, M2 clinical + denoised PCA
           (main M2 = mean of 3 seeds; each seed also refit separately)
           supplementary microbiome-only: P1 raw PCA, P2 denoised PCA (no clinical ceiling)
  classif  RF (500 trees, balanced, OOB) and SVM (scaled RBF, C = 1, balanced, LOO)

Usage: python predict_phenotype.py [--k 5] [--out results]
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HERE = Path(__file__).resolve().parent
DATA = HERE / "organized"
CLIN = HERE.parents[1] / "Make_Figure/第三步效应分解+应用/group_revision_sources/RFdata.csv"

SITES = ["saliva", "tongue", "upper_jaw_interproximal", "upper_jaw_lingual",
         "lower_jaw_interproximal", "lower_jaw_lingual"]
WINDOWS = ["W0", "W1", "W2", "W3", "WF"]
WIN_DAYS = {"W0": [-14, 0], "W1": [-14, 0, 2], "W2": [-14, 0, 2, 5],
            "W3": [-14, 0, 2, 5, 9], "WF": [-14, 0, 2, 5, 9, 14, 21]}
SEEDS = [2026, 2027, 2028]
CLIN_VARS = ["RF_area", "dRmax", "MQH", "SPS", "gingiva_RG"]
CLIN_MAX_DAY = 9  # day-14 / day-21 clinical values are never features (WF uses W3 clinical)
CLASSES = np.array([0, 1, 2])
RS = 0


# ------------------------------------------------------------------ inputs
def load_clinical():
    clin = pd.read_csv(CLIN, encoding="utf-8-sig")
    clin.columns = ["subjectid", "id", "fotonr", "day", "study_group", "rf_group", "MQH", "SPS",
                    "RF_area", "dRmax", "RF_area_x_dRmax", "gingiva_RG", "gingiva_RG_upper",
                    "gingiva_RG_lower"]
    label = clin.drop_duplicates("subjectid").set_index("subjectid").rf_group.astype(int)
    return clin, label


def clinical_features(clin, subjects, window):
    days = [d for d in WIN_DAYS[window] if d <= CLIN_MAX_DAY]
    wide = (clin[clin.day.isin(days)]
            .pivot_table(index="subjectid", columns="day", values=CLIN_VARS))
    wide.columns = [f"{v}_d{d}" for v, d in wide.columns]
    wide = wide.reindex(subjects)
    if wide.isna().any().any():
        raise ValueError(f"{window}: missing clinical values")
    return wide


def load_window(site, window):
    """Per-sample CLR matrices: raw, denoised per seed, denoised seed-mean."""
    frames = [pd.read_csv(DATA / site / f"{window}_seed{s}.csv.gz",
                          usecols=["sample_id", "subject_id", "time", "taxon_id", "raw_count", "total"],
                          dtype={"sample_id": str, "subject_id": str})
              for s in SEEDS]
    base = frames[0]
    for f in frames[1:]:
        if not (f[["sample_id", "taxon_id"]].values == base[["sample_id", "taxon_id"]].values).all():
            raise ValueError(f"{site} {window}: seed files misaligned")
        if not (f.raw_count.values == base.raw_count.values).all():
            raise ValueError(f"{site} {window}: raw counts differ across seeds")
    meta = base.drop_duplicates("sample_id")[["sample_id", "subject_id", "time"]].set_index("sample_id")

    def wide(values):
        m = base.assign(v=values).pivot(index="sample_id", columns="taxon_id", values="v")
        return m.loc[meta.index]

    def clr(m):
        return m.sub(m.mean(axis=1), axis=0)

    mats = {"raw": clr(np.log(wide(base.raw_count.values + 0.5)))}
    totals = [wide(f.total.values) for f in frames]
    for s, t in zip(SEEDS, totals):
        mats[f"seed{s}"] = clr(t)
    mats["mean"] = clr(sum(totals) / len(totals))
    return meta, mats


def flatten(meta, mat, subjects, days):
    """Subject × (visit × taxon); a missing visit takes the subject's within-window taxon mean."""
    rows, n_imputed = [], 0
    for sub in subjects:
        idx = meta.index[meta.subject_id == sub]
        m = mat.loc[idx].set_index(meta.loc[idx, "time"].values)
        fill = m.mean(axis=0)
        parts = []
        for d in days:
            if d in m.index:
                parts.append(m.loc[d].to_numpy())
            else:
                parts.append(fill.to_numpy())
                n_imputed += 1
        rows.append(np.concatenate(parts))
    return np.vstack(rows), n_imputed


def pca_scores(X, k):
    p = PCA(n_components=k, random_state=RS).fit(X)
    return p.transform(X), p.explained_variance_ratio_.sum()


# ------------------------------------------------------------------ models
def macro_auc(y, P):
    return roc_auc_score(y, P, multi_class="ovr", average="macro", labels=CLASSES)


def eval_rf(X, y):
    rf = RandomForestClassifier(n_estimators=500, class_weight="balanced", oob_score=True,
                                random_state=RS, n_jobs=-1).fit(X, y)
    P_tr = rf.predict_proba(X)
    P_cv = rf.oob_decision_function_
    pred_cv = rf.classes_[P_cv.argmax(1)]
    return dict(train_acc=accuracy_score(y, rf.predict(X)), train_auc=macro_auc(y, P_tr),
                cv_acc=accuracy_score(y, pred_cv), cv_auc=macro_auc(y, P_cv),
                cm=confusion_matrix(y, pred_cv, labels=CLASSES))


def svm():
    return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, class_weight="balanced",
                                               probability=True, random_state=RS))


def eval_svm(X, y):
    full = svm().fit(X, y)
    P_cv, pred_cv = np.zeros((len(y), 3)), np.zeros(len(y), dtype=int)
    for tr, te in LeaveOneOut().split(X):
        m = svm().fit(X[tr], y[tr])
        P_cv[te] = m.predict_proba(X[te])
        pred_cv[te] = m.predict(X[te])
    return dict(train_acc=accuracy_score(y, full.predict(X)), train_auc=macro_auc(y, full.predict_proba(X)),
                cv_acc=accuracy_score(y, pred_cv), cv_auc=macro_auc(y, P_cv),
                cm=confusion_matrix(y, pred_cv, labels=CLASSES))


CLASSIFIERS = {"RF": eval_rf, "SVM": eval_svm}


# ------------------------------------------------------------------ main loop
def run(k):
    clin, label = load_clinical()
    rows, pca_rows = [], []
    for site in SITES:
        for window in WINDOWS:
            meta, mats = load_window(site, window)
            subjects = sorted(set(meta.subject_id) & set(label.index))
            y = label.loc[subjects].to_numpy()
            Xc = clinical_features(clin, subjects, window).to_numpy()
            micro = {}
            for src, mat in mats.items():
                Xf, n_imp = flatten(meta, mat, subjects, WIN_DAYS[window])
                micro[src], ev = pca_scores(Xf, k)
                pca_rows.append(dict(site=site, window=window, source=src, n=len(subjects),
                                     n_features=Xf.shape[1], k=k, explained_var=ev,
                                     visits_imputed=n_imp))
            designs = {("M0", "-"): Xc, ("M1", "raw"): np.hstack([Xc, micro["raw"]]),
                       ("M2", "mean"): np.hstack([Xc, micro["mean"]]),
                       ("P1", "raw"): micro["raw"], ("P2", "mean"): micro["mean"]}
            for s in SEEDS:
                designs[("M2", f"seed{s}")] = np.hstack([Xc, micro[f"seed{s}"]])
                designs[("P2", f"seed{s}")] = micro[f"seed{s}"]
            for (model, src), X in designs.items():
                for cname, fn in CLASSIFIERS.items():
                    r = fn(X, y)
                    rows.append(dict(site=site, window=window, model=model, source=src,
                                     classifier=cname, n=len(subjects),
                                     n_low=int((y == 0).sum()), n_mid=int((y == 1).sum()),
                                     n_high=int((y == 2).sum()), n_features=X.shape[1],
                                     train_acc=r["train_acc"], train_auc=r["train_auc"],
                                     cv_type="OOB" if cname == "RF" else "LOO",
                                     cv_acc=r["cv_acc"], cv_auc=r["cv_auc"],
                                     confusion_cv=r["cm"].tolist()))
            print(f"{site:25s} {window} n={len(subjects)} done", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(pca_rows)


def null_baseline(k, reps=20):
    """OOB/LOO AUC on pure-noise features (k columns) for each cohort's label vector.

    Small-n LOO/OOB is biased below 0.5 under no signal (held-out class is under-represented
    in the training fold), so microbiome-only AUCs are read against this, not against 0.5.
    """
    _, label = load_clinical()
    rng = np.random.default_rng(RS)
    rows = []
    for site in SITES:
        meta, _ = load_window(site, "W0")
        subjects = sorted(set(meta.subject_id) & set(label.index))
        y = label.loc[subjects].to_numpy()
        if any(r["n"] == len(y) for r in rows):
            continue
        for cname, fn in CLASSIFIERS.items():
            aucs = [fn(rng.normal(size=(len(y), k)), y)["cv_auc"] for _ in range(reps)]
            rows.append(dict(n=len(y), cohort_site=site, classifier=cname, reps=reps,
                             null_auc_mean=np.mean(aucs), null_auc_sd=np.std(aucs),
                             null_auc_min=np.min(aucs), null_auc_max=np.max(aucs)))
    return pd.DataFrame(rows)


def deltas(res):
    key = ["site", "window", "classifier"]
    auc = res.pivot_table(index=key, columns=["model", "source"], values="cv_auc")
    acc = res.pivot_table(index=key, columns=["model", "source"], values="cv_acc")
    d = pd.DataFrame(index=auc.index)
    d["n"] = res.groupby(key).n.first()
    for name, t in [("auc", auc), ("acc", acc)]:
        d[f"M0_{name}"] = t[("M0", "-")]
        d[f"M1_{name}"] = t[("M1", "raw")]
        d[f"M2_{name}"] = t[("M2", "mean")]
        d[f"d_M2_M1_{name}"] = t[("M2", "mean")] - t[("M1", "raw")]
        d[f"d_M2_M0_{name}"] = t[("M2", "mean")] - t[("M0", "-")]
        d[f"d_M1_M0_{name}"] = t[("M1", "raw")] - t[("M0", "-")]
        d[f"P1_{name}"] = t[("P1", "raw")]
        d[f"P2_{name}"] = t[("P2", "mean")]
        d[f"d_P2_P1_{name}"] = t[("P2", "mean")] - t[("P1", "raw")]
    for tag, den, ref in [("", "M2", "M1"), ("_P", "P2", "P1")]:
        seed_cols = [f"d_{den}_{ref}_auc_seed{s}" for s in SEEDS]
        for s, c in zip(SEEDS, seed_cols):
            d[c] = auc[(den, f"seed{s}")] - auc[(ref, "raw")]
        d[f"seed_pos{tag}"] = (d[seed_cols] > 0).sum(1)
        d[f"seed_neg{tag}"] = (d[seed_cols] < 0).sum(1)
        d[f"seed_consistent{tag}"] = (d[f"seed_pos{tag}"] == 3) | (d[f"seed_neg{tag}"] == 3)
    return d.reset_index()


# ------------------------------------------------------------------ figures
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
MODEL_STYLE = {"M0": ("#2a78d6", "o", "M0 clinical"),
               "M1": ("#eb6834", "s", "M1 + raw PCA"),
               "M2": ("#1baf7a", "D", "M2 + denoised PCA"),
               "P1": ("#eb6834", "s", "P1 raw PCA only"),
               "P2": ("#1baf7a", "D", "P2 denoised PCA only")}
DIVERGING = LinearSegmentedColormap.from_list("bwr_gray", ["#e34948", "#f0efec", "#2a78d6"])
SITE_LABEL = {"saliva": "Saliva", "tongue": "Tongue", "upper_jaw_interproximal": "Upper interprox.",
              "upper_jaw_lingual": "Upper lingual", "lower_jaw_interproximal": "Lower interprox.",
              "lower_jaw_lingual": "Lower lingual"}


def style_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=INK2, labelsize=8)


def plot_heatmap(d, out, den="M2", ref="M1", tag=""):
    wins = ["W0", "W1", "W2", "W3"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    lim = 0.3
    for ax, cname in zip(axes, ["RF", "SVM"]):
        sub = d[(d.classifier == cname) & d.window.isin(wins)].set_index(["site", "window"])
        M = np.array([[sub.loc[(s, w), f"d_{den}_{ref}_auc"] for w in wins] for s in SITES])
        im = ax.imshow(M, cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
        for i, s in enumerate(SITES):
            for j, w in enumerate(wins):
                r = sub.loc[(s, w)]
                pos, neg = r[f"seed_pos{tag}"], r[f"seed_neg{tag}"]
                agree = f"{pos}/3" if pos >= neg else f"{neg}/3"
                ax.text(j, i, f"{M[i, j]:+.2f}\n{agree}", ha="center", va="center", fontsize=8,
                        color=INK, fontweight="bold" if r[f"seed_consistent{tag}"] else "normal")
        ax.set_xticks(range(len(wins)), wins)
        ax.set_yticks(range(len(SITES)), [f"{SITE_LABEL[s]} (n={sub.loc[(s, 'W0'), 'n']})" for s in SITES])
        ax.set_title(f"{cname} ({'OOB' if cname == 'RF' else 'LOO'}): ΔAUC = {den} − {ref}", fontsize=10, color=INK)
        ax.tick_params(colors=INK2, labelsize=8, length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    cb = fig.colorbar(im, ax=axes, shrink=0.8)
    cb.set_label("ΔAUC (denoised − raw)", fontsize=8, color=INK2)
    cb.ax.tick_params(labelsize=8, colors=INK2)
    fig.text(0.01, -0.02, "Cell text: ΔAUC, then number of seeds (of 3) sharing the majority sign; "
             "bold = all 3 seeds agree.", fontsize=7.5, color=INK2)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)


def plot_curves(res, out, models=("M0", "M1", "M2")):
    main = res[(res.source.isin(["-", "raw", "mean"]))]
    x_win = {"W0": 0, "W1": 1, "W2": 2, "W3": 3, "WF": 4.3}
    fig, axes = plt.subplots(2, 6, figsize=(16, 6), sharey=True, constrained_layout=True)
    for i, cname in enumerate(["RF", "SVM"]):
        for j, site in enumerate(SITES):
            ax = axes[i, j]
            style_axes(ax)
            ax.axhline(0.5, color=GRID, lw=1, zorder=0)
            sub = main[(main.classifier == cname) & (main.site == site)].set_index(["model", "window"])
            for model in models:
                col, mk, lab = MODEL_STYLE[model]
                ys = [sub.loc[(model, w), "cv_auc"] for w in ["W0", "W1", "W2", "W3"]]
                ax.plot(range(4), ys, color=col, lw=2, marker=mk, ms=6, label=lab,
                        markeredgecolor="#fcfcfb", markeredgewidth=1.2)
                yf = sub.loc[(model, "WF"), "cv_auc"]
                ax.plot([3, x_win["WF"]], [ys[-1], yf], color=col, lw=1.5, ls="--")
                ax.plot(x_win["WF"], yf, marker=mk, ms=6, color=col, markerfacecolor="#fcfcfb",
                        markeredgewidth=1.5)
            ax.set_xticks(list(x_win.values()), list(x_win.keys()))
            ax.set_ylim(0.0, 1.0)
            if i == 0:
                ax.set_title(f"{SITE_LABEL[site]} (n={sub.n.iloc[0]})", fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel(f"{cname} {'OOB' if cname == 'RF' else 'LOO'} macro AUC", fontsize=9, color=INK2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False, fontsize=9)
    fig.text(0.01, -0.03, "Solid: predictive windows W0–W3. Dashed/open marker: WF (includes day-14/21 "
             "microbiome; clinical capped at day 9) — upper-bound reference only. Grey line: chance (0.5).",
             fontsize=7.5, color=INK2)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)


def summary_table(d):
    t = d[d.window != "WF"].copy()
    cols = ["site", "classifier", "window", "n", "M0_auc", "M1_auc", "M2_auc",
            "d_M2_M1_auc", "d_M2_M0_auc", "d_M1_M0_auc", "seed_pos", "seed_neg",
            "P1_auc", "P2_auc", "d_P2_P1_auc", "seed_pos_P", "seed_neg_P"]
    return t[cols].sort_values(["classifier", "site", "window"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = HERE / a.out
    out.mkdir(exist_ok=True)
    res, pca = run(a.k)
    res.to_csv(out / "metrics_all.tsv", sep="\t", index=False)
    pca.to_csv(out / "pca_explained_variance.tsv", sep="\t", index=False)
    d = deltas(res)
    d.to_csv(out / "deltas.tsv", sep="\t", index=False)
    summary_table(d).to_csv(out / "summary_table.tsv", sep="\t", index=False, float_format="%.3f")
    plot_heatmap(d, out / "fig1_heatmap_dAUC_M2_vs_M1.png")
    plot_curves(res, out / "fig2_window_curves.png")
    plot_heatmap(d, out / "fig3_heatmap_dAUC_P2_vs_P1.png", den="P2", ref="P1", tag="_P")
    plot_curves(res, out / "fig4_window_curves_microbiome_only.png", models=("M0", "P1", "P2"))
    null_baseline(a.k).to_csv(out / "null_baseline.tsv", sep="\t", index=False)
    print("written to", out)


if __name__ == "__main__":
    main()
