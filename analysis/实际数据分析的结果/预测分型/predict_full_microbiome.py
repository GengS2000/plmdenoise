"""All-feature, paired nested CV; existing denoising remains transductive.

No PCA or outcome-driven taxon selection. Clinical and each microbiome site
are normalized separately using training subjects only. Dual ridge on one-hot
targets is equivalent to linear ridge using ALL columns, with block penalties.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix
from threadpoolctl import threadpool_limits
import predict_phenotype as old

ALPHAS = [0.01, 0.1, 1.0, 10.0]
WEIGHTS = [0.0, 0.1, 1.0, 10.0]
SOURCES = ['raw', 'mean', 'seed2026', 'seed2027', 'seed2028']
MODELS = ['M0', 'M1', 'M2', 'M2_seed2026', 'M2_seed2027', 'M2_seed2028']


def occurrence_matrices(site, window, meta, taxa):
    """Observed presence vs posterior presence, each paired with full CLR."""
    frames = [pd.read_csv(old.DATA / site / f'{window}_seed{s}.csv.gz',
                         usecols=['sample_id', 'taxon_id', 'raw_count', 'absence_probability'])
              for s in old.SEEDS]
    def wide(f, values):
        return f.assign(v=values).pivot(index='sample_id', columns='taxon_id', values='v') \
            .reindex(index=meta.index, columns=taxa)
    result = {'raw': wide(frames[0], (frames[0].raw_count > 0).astype(float))}
    for seed, f in zip(old.SEEDS, frames):
        result[f'seed{seed}'] = wide(f, 1 - f.absence_probability)
    result['mean'] = sum(result[f'seed{s}'] for s in old.SEEDS) / 3
    assert all(np.isfinite(m.to_numpy()).all() and m.min().min() >= 0 and m.max().max() <= 1
               for m in result.values())
    return result


def auc(y, scores):
    """Macro one-vs-rest AUC on uncalibrated class scores, no softmax needed."""
    return float(np.mean([np.mean((scores[y == c, c, None] > scores[y != c, c])
                         + 0.5 * (scores[y == c, c, None] == scores[y != c, c]))
                         for c in range(3)]))


def kernel(X, tr, te, standardize=False):
    center = X[tr].mean(0)
    A, B = X[tr] - center, X[te] - center
    if standardize:
        scale = A.std(0)
        scale[scale < 1e-12] = 1
        A, B = A / scale, B / scale
    # Training mean squared row norm = 1, independent of feature count.
    norm = max(float(np.mean(np.sum(A * A, axis=1))), 1e-12)
    return A @ A.T / norm, B @ A.T / norm


def blocks(Xc, Xm, tr, te):
    kc, tc = kernel(Xc, tr, te, True)
    ms = [kernel(x, tr, te) for x in Xm]
    return kc, tc, sum(x[0] for x in ms) / len(ms), sum(x[1] for x in ms) / len(ms)


def predict(k, kt, y, alpha):
    Y = np.eye(3)[y]
    intercept = Y.mean(0)
    return kt @ np.linalg.solve(k + alpha * np.eye(len(y)), Y - intercept) + intercept


def tune_predict(Xc, Xm, y, tr, te, inner_splits, clinical_only=False):
    weights = [0.0] if clinical_only else WEIGHTS
    candidates = [(w, a) for w in weights for a in ALPHAS]
    values = np.zeros(len(candidates))
    for itr, iva in inner_splits:
        a, b = tr[itr], tr[iva]
        kc, tc, km, tm = blocks(Xc, Xm, a, b)
        for j, (w, reg) in enumerate(candidates):
            values[j] += auc(y[b], predict(kc + w * km, tc + w * tm, y[a], reg))
    # Fixed ordering favors weight zero on exact ties; no outer-test selection.
    best = int(np.argmax(values))
    w, reg = candidates[best]
    kc, tc, km, tm = blocks(Xc, Xm, tr, te)
    return predict(kc + w * km, tc + w * tm, y[tr], reg), w, reg, values[best] / len(inner_splits)


def paired_bootstrap(y, predictions, reps=2000):
    """Subject-paired stratified bootstrap of FIXED OOF scores, not refit CI.

    Predictions shape model x repeat x subject x class. Averaging pairwise
    comparison matrices gives exactly the mean repeat AUC under resampling.
    """
    rng = np.random.default_rng(73091)
    out = np.zeros((reps, len(predictions)))
    # Resample each label separately, same weights across models/repeats.
    weights = np.zeros((reps, len(y)))
    for c in range(3):
        ix = np.where(y == c)[0]
        weights[:, ix] = rng.multinomial(len(ix), np.ones(len(ix)) / len(ix), size=reps)
    for c in range(3):
        pos, neg = np.where(y == c)[0], np.where(y != c)[0]
        for m, p in enumerate(predictions):
            d = p[:, pos, c, None] - p[:, neg, c][:, None, :]
            comparisons = ((d > 0) + .5 * (d == 0)).mean(0)
            out[:, m] += np.einsum('bi,ij,bj->b', weights[:, pos], comparisons,
                                   weights[:, neg]) / (len(pos) * len(neg) * 3)
    return out


def evaluate(name, window, subjects, Xc, Xms, y, repeats, out):
    predictions = np.zeros((len(MODELS), repeats, len(y), 3))
    selections, metrics, oof = [], [], []
    for rep in range(repeats):
        outer = StratifiedKFold(5, shuffle=True, random_state=8400 + rep)
        for fold, (tr, te) in enumerate(outer.split(Xc, y)):
            inner = list(StratifiedKFold(3, shuffle=True, random_state=9100 + rep * 5 + fold)
                         .split(Xc[tr], y[tr]))
            assert not set(tr) & set(te)
            for mi, model in enumerate(MODELS):
                source = 'raw' if mi == 0 else SOURCES[mi - 1]
                p, w, reg, score = tune_predict(Xc, Xms[source], y, tr, te, inner, mi == 0)
                predictions[mi, rep, te] = p
                selections.append(dict(site=name, window=window, repeat=rep, fold=fold,
                                       model=model, micro_weight=w, alpha=reg, inner_auc=score))
                for j, subindex in enumerate(te):
                    oof.append(dict(site=name, window=window, repeat=rep, fold=fold,
                                    model=model, subject=subjects[subindex], y=int(y[subindex]),
                                    score_low=p[j, 0], score_mid=p[j, 1], score_high=p[j, 2]))
        for mi, model in enumerate(MODELS):
            p = predictions[mi, rep]
            metrics.append(dict(site=name, window=window, repeat=rep, model=model, n=len(y),
                                auc=auc(y, p), accuracy=accuracy_score(y, p.argmax(1)),
                                balanced_accuracy=balanced_accuracy_score(y, p.argmax(1)),
                                confusion=json.dumps(confusion_matrix(y, p.argmax(1)).tolist())))
    boot = paired_bootstrap(y, predictions)
    rows = []
    avgs = np.array([[auc(y, p) for p in pr] for pr in predictions])
    for m, ref, label in [(2, 1, 'M2-M1'), (2, 0, 'M2-M0'), (1, 0, 'M1-M0')]:
        delta = avgs[m] - avgs[ref]
        lo, hi = np.quantile(boot[:, m] - boot[:, ref], [.025, .975])
        rows.append(dict(site=name, window=window, n=len(y), contrast=label,
                         delta_auc=delta.mean(), ci_low_conditional=lo, ci_high_conditional=hi,
                         positive_repeats=int((delta > 0).sum()), repeats=repeats,
                         seed_positive=int(sum((avgs[j] - avgs[ref]).mean() > 0 for j in [3, 4, 5]))
                         if m == 2 else np.nan,
                         M0_auc=avgs[0].mean(), M1_auc=avgs[1].mean(), M2_auc=avgs[2].mean(),
                         micro_features=sum(x.shape[1] for x in Xms['raw'])))
    stem = out / f'{name}_{window}'
    for suffix, frame in [('metrics', metrics), ('oof', oof), ('selection', selections), ('contrasts', rows)]:
        pd.DataFrame(frame).to_csv(str(stem) + f'_{suffix}.tsv', sep='\t', index=False)
    print(f'{name} {window} n={len(y)} M0/M1/M2={avgs[:3].mean(1).round(3)} '
          f'delta={rows[0]["delta_auc"]:+.3f}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--out', default='results_full_microbiome')
    parser.add_argument('--representation', choices=['clr', 'two_part'], default='clr')
    args = parser.parse_args()
    out = old.HERE / args.out
    out.mkdir(exist_ok=True)
    config = dict(repeats=args.repeats, outer_folds=5, inner_folds=3, alphas=ALPHAS,
                  weights=WEIGHTS, windows=old.WINDOWS[:4], preprocessing='fold-local',
                  denoising='precomputed transductive; not refit within CV',
                  ci='2000 paired stratified subject bootstraps of fixed OOF scores; conditional only',
                  metric='mean repeat macro OVR AUC on ridge class scores',
                  primary='all6_common W3 M2-M1; all other cells exploratory',
                  seed=8400)
    if args.representation != 'clr':
        config['representation'] = args.representation
        config['primary'] = 'sensitivity analysis; abundance and occurrence equally weighted per site'
    config_path = out / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8')) != config:
        raise ValueError('Existing output uses a different configuration; choose a new --out.')
    config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    clin, labels = old.load_clinical()
    for window in old.WINDOWS[:4]:
        cache = {}
        for site in old.SITES:
            meta, mats = old.load_window(site, window)
            assert set(meta.time).issubset(set(old.WIN_DAYS[window]))
            subjects = sorted(set(meta.subject_id) & set(labels.index))
            X = {s: [old.flatten(meta, mat, subjects, old.WIN_DAYS[window])[0]]
                 for s, mat in mats.items()}
            if args.representation == 'two_part':
                presence = occurrence_matrices(site, window, meta, mats['raw'].columns)
                for s, mat in presence.items():
                    X[s].append(old.flatten(meta, mat, subjects, old.WIN_DAYS[window])[0])
            assert all(np.isfinite(a).all() for arrays in X.values() for a in arrays)
            cache[site] = (subjects, X)
        groups = {s: [s] for s in old.SITES}
        groups['all6_common'] = old.SITES
        groups['full4_common'] = old.SITES[:4]
        for name, sites in groups.items():
            if (out / f'{name}_{window}_contrasts.tsv').exists():
                print(f'skip completed {name} {window}', flush=True)
                continue
            subjects = sorted(set.intersection(*(set(cache[s][0]) for s in sites)))
            Xms = {src: [array[[cache[s][0].index(x) for x in subjects]]
                         for s in sites for array in cache[s][1][src]] for src in SOURCES}
            Xc = old.clinical_features(clin, subjects, window).to_numpy()
            evaluate(name, window, subjects, Xc, Xms, labels.loc[subjects].to_numpy(), args.repeats, out)
    for suffix in ['metrics', 'oof', 'selection', 'contrasts']:
        pd.concat([pd.read_csv(f, sep='\t') for f in sorted(out.glob(f'*_W*_{suffix}.tsv'))]) \
            .to_csv(out / f'{suffix}_all.tsv', sep='\t', index=False)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
