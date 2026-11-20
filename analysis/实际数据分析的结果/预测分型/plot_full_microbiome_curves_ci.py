"""Redraw full-feature AUC curves with per-model conditional 95% intervals.

Intervals reuse predict_full_microbiome.paired_bootstrap on the saved fixed OOF
scores (2000 label-stratified subject bootstraps, mean of 10 repeat AUCs). They
reflect subject resampling only; no refit, inner tuning or denoiser uncertainty.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import predict_phenotype as old
from predict_full_microbiome import auc, paired_bootstrap
from summarize_full_microbiome import GROUPS, LABELS

HERE = Path(__file__).resolve().parent
RES = HERE / 'results_full_microbiome'
OUT_PNG = HERE.parent / '下游分析整理_20261101' / '第二部分_牙龈炎机器学习' / '08_已有预测曲线.png'
MODELS = ['M0', 'M1', 'M2']
WINDOWS = old.WINDOWS[:4]


def interval_table():
    contrasts = pd.read_csv(RES / 'contrasts_all.tsv', sep='\t')
    rows = []
    for site in GROUPS:
        for w in WINDOWS:
            oof = pd.read_csv(RES / f'{site}_{w}_oof.tsv', sep='\t')
            subjects = sorted(oof.subject.unique())
            y = oof.drop_duplicates('subject').set_index('subject').loc[subjects, 'y'].to_numpy()
            reps = sorted(oof['repeat'].unique())
            pred = np.zeros((len(MODELS), len(reps), len(subjects), 3))
            for mi, m in enumerate(MODELS):
                s = oof[oof.model == m].set_index(['repeat', 'subject'])
                for ri, r in enumerate(reps):
                    pred[mi, ri] = s.loc[r].loc[subjects, ['score_low', 'score_mid', 'score_high']].to_numpy()
            boot = paired_bootstrap(y, pred)
            ref = contrasts[(contrasts.site == site) & (contrasts.window == w)].iloc[0]
            for mi, m in enumerate(MODELS):
                point = np.mean([auc(y, p) for p in pred[mi]])
                assert abs(point - ref[f'{m}_auc']) < 1e-9, (site, w, m)
                lo, hi = np.quantile(boot[:, mi], [.025, .975])
                rows.append(dict(site=site, window=w, model=m, n=len(y), auc=point,
                                 ci_low_conditional=lo, ci_high_conditional=hi))
    return pd.DataFrame(rows)


def main():
    t = interval_table()
    t.to_csv(RES / 'model_auc_ci.tsv', sep='\t', index=False)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5), sharey=True, constrained_layout=True)
    styles = [('M0', '#555555', -.12), ('M1', '#d27a2c', 0), ('M2', '#176b91', .12)]
    for site, label, ax in zip(GROUPS, LABELS, axes.flat):
        s = t[t.site == site]
        for model, color, dx in styles:
            m = s[s.model == model].set_index('window').loc[WINDOWS]
            x = np.arange(4) + dx
            ax.errorbar(x, m.auc, yerr=[m.auc - m.ci_low_conditional, m.ci_high_conditional - m.auc],
                        marker='o', ms=5, capsize=3, lw=1.5, elinewidth=1, color=color, label=model)
        ax.set_title(f'{label} (n={s.n.iloc[0]})', fontsize=10)
        ax.set_xticks(range(4), WINDOWS)
        ax.set_ylim(.5, 1.0)
        ax.set_xlabel('Prediction window')
        ax.set_ylabel('Mean repeat macro OVR AUC')
        ax.grid(axis='y', alpha=.2)
    handles, _ = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, ['M0 clinical', 'M1 clinical + raw', 'M2 clinical + denoised'],
               loc='outside upper center', ncol=3)
    fig.supxlabel('Source: results_full_microbiome, 2026-10-30 | All CLR features | 10 x nested stratified 5-fold CV | Visits through day 9\n'
                  'Error bars: conditional 95% interval, 2000 label-stratified subject bootstraps of fixed OOF scores '
                  '(no refit; overlapping bars do not test paired differences)', fontsize=8)
    fig.savefig(OUT_PNG, dpi=180)
    plt.close(fig)
    print(t.round(3).to_string())


if __name__ == '__main__':
    main()
