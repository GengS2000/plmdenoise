"""Subject-level, label-blind CLR/PCA analysis and paired stability metrics."""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

METHODS = ("naive", "m4_flex_abundance")


def clr(p, pseudocount=1e-6):
    """Same additive relative-abundance pseudocount for BOTH observed branches."""
    p = np.asarray(p, dtype=float)
    if not np.isfinite(p).all() or np.any(p < 0) or np.any(p.sum(-1) <= 0):
        raise ValueError("Invalid composition")
    p = p / p.sum(-1, keepdims=True)
    z = np.log(p + pseudocount)
    return z - z.mean(-1, keepdims=True)


def bootstrap_indices(group, exposure, seed):
    """Stratified cluster bootstrap: keep complete 12 x 500 subject blocks."""
    rng = np.random.default_rng(seed)
    indices = []
    for g, e in sorted(set(zip(group.tolist(), exposure.tolist()))):
        cell = np.flatnonzero((group == g) & (exposure == e))
        indices.extend(rng.choice(cell, size=len(cell), replace=True))
    return rng.permutation(np.asarray(indices, dtype=int))


def _ss_residual(x, design):
    residual = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    return float(np.sum(residual ** 2))


def permanova(scores, group, exposure, permutations=999, seed=0):
    """Euclidean PERMANOVA: group after exposure, labels permuted within exposure.

    For Euclidean PC scores, multivariate regression sums of squares equal
    distance-based PERMANOVA sums of squares. Permute whole subject labels, as
    in the generating design (balanced randomisation within exposure strata).
    PCA/denoising must be label blind. Bootstrap duplicates may use permutations=0
    for descriptive F/R2 only; never compute their permutation p values.
    """
    x = np.asarray(scores, float)
    g, e = np.asarray(group), np.asarray(exposure)
    if len(np.unique(g)) != 2 or len(x) != len(g) or permutations < 0:
        raise ValueError("PERMANOVA needs two groups and nonnegative permutations")
    reduced = np.ones((len(g), 1))
    if np.ptp(e):
        reduced = np.column_stack((reduced, e))
    ss0 = _ss_residual(x, reduced)
    total = float(np.sum((x - x.mean(0)) ** 2))
    if total <= np.finfo(float).eps:
        raise ValueError("No score variation")

    def statistic(labels):
        full = np.column_stack((reduced, labels))
        rank = np.linalg.matrix_rank(full)
        if rank != reduced.shape[1] + 1 or len(g) <= rank:
            raise ValueError("Group is aliased with exposure or residual df is zero")
        residual = _ss_residual(x, full)
        effect = max(0., ss0 - residual)
        return effect / max(residual / (len(g) - rank), np.finfo(float).tiny), effect

    f, effect = statistic(g)
    result = dict(permanova_f=f, group_r2=effect / total,
                  group_partial_r2=effect / max(ss0, 1e-300))
    if permutations == 0:
        return result
    rng = np.random.default_rng(seed)
    exceed = 0
    strata = [np.flatnonzero(e == level) for level in np.unique(e)]
    for _ in range(permutations):
        perm = g.copy()
        for ids in strata:
            perm[ids] = rng.permutation(g[ids])
        exceed += statistic(perm)[0] >= f - 1e-12
    return result | dict(permanova_p=(1 + exceed) / (1 + permutations))


def dispersion(scores, group):
    """Descriptive, bias-adjusted distance to group centroid (not a location test)."""
    out = {}
    for g in (0, 1):
        x = scores[group == g]
        d = np.linalg.norm(x - x.mean(0), axis=1) * np.sqrt(len(x) / (len(x) - 1))
        out[f"dispersion_group{g}"] = float(d.mean())
    out["dispersion_ratio_1_over_0"] = out["dispersion_group1"] / max(out["dispersion_group0"], 1e-300)
    return out


def geometry(x, truth, neighbors=5):
    a, b = pdist(x), pdist(truth)
    corr = float(spearmanr(a, b).statistic) if np.ptp(a) and np.ptp(b) else float("nan")
    da, db = squareform(a), squareform(b)
    np.fill_diagonal(da, np.inf)
    np.fill_diagonal(db, np.inf)
    k = min(neighbors, len(x) - 1)
    na, nb = np.argsort(da, axis=1)[:, :k], np.argsort(db, axis=1)[:, :k]
    overlap = np.mean([len(set(u) & set(v)) / k for u, v in zip(na, nb)])
    return dict(truth_distance_spearman=corr, truth_knn_overlap=float(overlap))


def align_projection(x, reference):
    """Similarity Procrustes for plot orientation only; report raw spread separately."""
    a, b = x - x.mean(0), reference - reference.mean(0)
    u, singular, vt = np.linalg.svd(a.T @ b, full_matrices=False)
    scale = singular.sum() / max(float(np.sum(a * a)), 1e-300)
    aligned = scale * a @ (u @ vt) + reference.mean(0)
    error = np.sqrt(np.sum((aligned - reference) ** 2) / max(np.sum(b * b), 1e-300))
    return aligned, float(error)


def representation(composition, truth_clr, group, exposure, *, components=5,
                   pseudocount=1e-6, seed=0, permutations=999, inference=True,
                   has_effect=True):
    # No labels enter CLR, PCA or K-means. Visit average gives one independent row/subject.
    visit_clr = clr(composition, pseudocount)
    subject = visit_clr.mean(axis=1)
    truth = truth_clr.mean(axis=1)
    if not 2 <= components < min(subject.shape):
        raise ValueError("Need 2 <= components < min(subjects, taxa)")
    pca = PCA(n_components=components, svd_solver="full").fit(subject)
    scores = pca.transform(subject)
    clusters = KMeans(n_clusters=2, n_init=20, random_state=seed).fit_predict(scores)
    metrics = dict(clr_rmse=float(np.sqrt(np.mean((visit_clr - truth_clr) ** 2))),
                   pca_variance_retained=float(pca.explained_variance_ratio_.sum()),
                   ari=adjusted_rand_score(group, clusters) if has_effect else None,
                   nmi=normalized_mutual_info_score(group, clusters) if has_effect else None,
                   silhouette=float(silhouette_score(scores, clusters)) if len(np.unique(clusters)) == 2 else None,
                   **dispersion(scores, group), **geometry(scores, truth))
    # 2-D geometry is evaluated separately from the 5-D downstream representation.
    metrics.update({"projection2_" + k: v for k, v in geometry(scores[:, :2], truth).items()})
    oracle2 = PCA(n_components=2, svd_solver="full").fit_transform(truth)
    _, metrics["projection2_procrustes_truth"] = align_projection(scores[:, :2], oracle2)
    metrics.update(permanova(scores, group, exposure, permutations if inference else 0, seed))
    arrays = dict(subject_clr=subject, scores=scores, loadings=pca.components_, clusters=clusters,
                  truth_subject_clr=truth, truth_scores2=oracle2, group=group, exposure=exposure)
    return metrics, arrays


def stability(current, reference, indices):
    """Compare common original subjects once, even if drawn repeatedly."""
    unique = np.unique(indices)
    averaged = np.array([current["scores"][indices == i].mean(0) for i in unique])
    labels = np.array([np.bincount(current["clusters"][indices == i]).argmax() for i in unique])
    base = reference["scores"][unique]
    aligned, error = align_projection(averaged[:, :2], base[:, :2])
    loading_overlap = np.sum((current["loadings"] @ reference["loadings"].T) ** 2) / len(current["loadings"])
    out = dict(stability_cluster_ari=float(adjusted_rand_score(reference["clusters"][unique], labels)),
               stability_subspace_overlap=float(loading_overlap), stability_projection2_error=error,
               stability_spread_ratio=float(np.linalg.norm(averaged - averaged.mean(0)) /
                                            max(np.linalg.norm(base - base.mean(0)), 1e-300)),
               stability_clr_rmse=float(np.sqrt(np.mean((current["subject_clr"] - reference["subject_clr"][indices]) ** 2))),
               stability_unique_subjects=int(len(unique)))
    return out, unique, aligned


def wilson(successes, n):
    if not n:
        return None, None
    z = 1.959963984540054
    p = successes / n
    center = (p + z*z / (2*n)) / (1 + z*z/n)
    half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return max(0., center-half), min(1., center+half)
