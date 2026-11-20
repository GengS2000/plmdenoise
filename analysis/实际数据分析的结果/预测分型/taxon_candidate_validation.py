"""Which lower-interproximal taxa carry the denoised RF-phenotype gain?

Fold-internal candidate screening plus keep/drop/balance validation, on exactly the
cohort, label, clinical block, splits, grid and bootstrap of predict_full_microbiome.py
(lower_jaw_interproximal, n=26, RF phenotype).

Per outer training fold, separately for raw and denoised (seed mean):
  1. Rank taxa by the scaled ridge coefficient of a clinical + microbiome fit whose
     (weight>0, alpha) is chosen by the same inner CV on training subjects only;
     effect = |beta| * training SD summed over visits, direction = High - Low.
  2. Candidates = top K taxa (K = 10, 25, 50).
  3. Refit with full inner tuning and score the outer test fold:
     keep  = clinical + candidates only (subcomposition re-centred per visit),
     drop  = clinical + all other taxa (re-centred),
     ratio = clinical + per-visit log-ratio (mean CLR of High-direction candidates
             minus mean CLR of Low-direction candidates).
Seed-specific denoised data are ranked the same way for stability only.
Nothing is chosen with test subjects; the screening is descriptive, not causal.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
import predict_full_microbiome as model
import predict_phenotype as old

SITE = 'lower_jaw_interproximal'
OUT = old.HERE / 'results_taxon_candidates'
REPEATS = 10
KS = [10, 25, 50]
SEEDS = [f'seed{s}' for s in old.SEEDS]


def tune(Xc, X, y, tr, te, inner, weights):
    full = model.WEIGHTS
    model.WEIGHTS = weights
    try:
        return model.tune_predict(Xc, [X], y, tr, te, inner)
    finally:
        model.WEIGHTS = full


def rank_taxa(Xc, T, y, tr, te, inner):
    """Fold-internal ranking from the ridge fit restricted to microbiome weight > 0."""
    n, V, P = T.shape
    X = T.reshape(n, -1)
    _, w, alpha, _ = tune(Xc, X, y, tr, te, inner, [0.1, 1.0, 10.0])
    kc, _, km, _ = model.blocks(Xc, [X], tr, te)
    Y = np.eye(3)[y[tr]]
    dual = np.linalg.solve(kc + w * km + alpha * np.eye(len(tr)), Y - Y.mean(0))
    A = X[tr] - X[tr].mean(0)
    beta = w * A.T @ dual / max(np.mean(np.sum(A * A, axis=1)), 1e-12)
    sd = A.std(0)
    effect = (np.linalg.norm(beta, axis=1) * sd).reshape(V, P).sum(0)
    sign = ((beta[:, 2] - beta[:, 0]) * sd).reshape(V, P).sum(0)
    return effect, sign


def subcomposition(T, mask):
    Z = T[:, :, mask].copy()
    Z -= Z.mean(2, keepdims=True)
    return Z.reshape(len(T), -1)


def log_ratio(T, up, down):
    hi = T[:, :, up].mean(2) if up.any() else 0
    lo = T[:, :, down].mean(2) if down.any() else 0
    return np.asarray(hi - lo, dtype=float).reshape(len(T), -1)


def run_window(window, clin, labels, genus):
    meta, mats = old.load_window(SITE, window)
    subjects = sorted(set(meta.subject_id) & set(labels.index))
    taxa = np.array(mats['raw'].columns)
    days = old.WIN_DAYS[window]
    T = {s: old.flatten(meta, mats[s], subjects, days)[0].reshape(len(subjects), len(days), -1)
         for s in ['raw', 'mean'] + SEEDS}
    y = labels.loc[subjects].to_numpy()
    Xc = old.clinical_features(clin, subjects, window).to_numpy()
    names = ['M0', 'raw_all', 'den_all'] + [f'{src}_{kind}{k}' for src in ['raw', 'den']
                                             for k in KS for kind in ['keep', 'drop', 'ratio']]
    pred = np.zeros((len(names), REPEATS, len(y), 3))
    picks = []
    for rep in range(REPEATS):
        for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=8400 + rep).split(Xc, y)):
            inner = list(StratifiedKFold(3, shuffle=True, random_state=9100 + rep * 5 + fold).split(Xc[tr], y[tr]))
            designs = {'raw_all': T['raw'].reshape(len(y), -1), 'den_all': T['mean'].reshape(len(y), -1)}
            for src, key in [('raw', 'raw'), ('den', 'mean')] + [(s, s) for s in SEEDS]:
                effect, sign = rank_taxa(Xc, T[key], y, tr, te, inner)
                order = np.argsort(-effect)
                for r, j in enumerate(order[:max(KS)]):
                    picks.append(dict(window=window, source=src, repeat=rep, fold=fold, taxon=taxa[j],
                                      rank=r + 1, effect_share=effect[j] / effect.sum(), high_minus_low=sign[j]))
                if src not in ('raw', 'den'):
                    continue
                for k in KS:
                    mask = np.zeros(len(taxa), bool)
                    mask[order[:k]] = True
                    designs[f'{src}_keep{k}'] = subcomposition(T[key], mask)
                    designs[f'{src}_drop{k}'] = subcomposition(T[key], ~mask)
                    designs[f'{src}_ratio{k}'] = log_ratio(T[key], mask & (sign > 0), mask & (sign <= 0))
            for j, name in enumerate(names):
                X = designs.get(name, designs['raw_all'])
                pred[j, rep, te] = model.tune_predict(Xc, [X], y, tr, te, inner, name == 'M0')[0]
        print(window, 'repeat', rep, 'done', flush=True)
    auc = np.array([[model.auc(y, p) for p in pr] for pr in pred])
    boot = model.paired_bootstrap(y, pred)
    rows = []
    for j, name in enumerate(names):
        for ref in ['M0', 'raw_all', 'den_all']:
            r = names.index(ref)
            if r == j:
                continue
            lo, hi = np.quantile(boot[:, j] - boot[:, r], [.025, .975])
            rows.append(dict(window=window, n=len(y), model=name, reference=ref, auc=auc[j].mean(),
                             reference_auc=auc[r].mean(), delta_auc=(auc[j] - auc[r]).mean(),
                             ci_low_conditional=lo, ci_high_conditional=hi,
                             positive_repeats=int(((auc[j] - auc[r]) > 0).sum())))
    picks = pd.DataFrame(picks)
    picks['genus'] = picks.taxon.map(genus)
    return pd.DataFrame(rows), picks, subjects


def stability(picks, k):
    top = picks[picks['rank'] <= k]
    folds = picks.groupby(['window', 'source'])[['repeat', 'fold']].apply(lambda d: len(d.drop_duplicates()))
    g = top.groupby(['window', 'source', 'taxon', 'genus'], dropna=False).agg(
        n_selected=('rank', 'size'), mean_rank=('rank', 'mean'), mean_effect_share=('effect_share', 'mean'),
        high_direction_fraction=('high_minus_low', lambda s: float(np.mean(s > 0))),
        mean_high_minus_low=('high_minus_low', 'mean')).reset_index()
    g['selection_frequency'] = g.n_selected / g.set_index(['window', 'source']).index.map(folds)
    g['K'] = k
    return g.sort_values(['window', 'source', 'selection_frequency', 'mean_rank'], ascending=[True, True, False, True])


def main():
    OUT.mkdir(exist_ok=True)
    clin, labels = old.load_clinical()
    genus = pd.read_csv(old.DATA / SITE / 'taxa.csv').set_index('taxon_id').Genus
    reference = pd.read_csv(old.HERE / 'results_full_microbiome/contrasts_all.tsv', sep='\t')
    contrasts, picks = [], []
    for window in old.WINDOWS[:4]:
        c, p, _ = run_window(window, clin, labels, genus)
        ref = reference[(reference.site == SITE) & (reference.window == window) & (reference.contrast == 'M2-M1')].iloc[0]
        got = c[(c.model == 'den_all') & (c.reference == 'raw_all')].iloc[0]
        # Same splits, grid and data: the full models must reproduce the published run.
        assert np.isclose(got.auc, ref.M2_auc) and np.isclose(got.reference_auc, ref.M1_auc), (got, ref)
        contrasts.append(c); picks.append(p)
    pd.concat(contrasts).to_csv(OUT / 'keep_drop_contrasts.tsv', sep='\t', index=False)
    picks = pd.concat(picks)
    picks.to_csv(OUT / 'fold_candidates.tsv.gz', sep='\t', index=False)
    pd.concat([stability(picks, k) for k in KS]).to_csv(OUT / 'candidate_stability.tsv', sep='\t', index=False)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
