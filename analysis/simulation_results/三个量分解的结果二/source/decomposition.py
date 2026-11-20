"""Truth-aware scoring of the baseline / common / individual / event split.

Component definitions are the ones used by the real-data export
(real_denoise/event_export.py), so a verdict here transfers to that analysis:

    common     = trend + group_effect             (population spline)
    individual = shared + specific                (latent factors + residual basis)
    event      = event_plus                       (population curve x (1 + gain))
    baseline   = Z - common - individual - event  (alpha + subject intercept)

Everything is compared on the CLR scale (centred over taxa), as in
cohort_analysis.py. Three families of comparisons are reported, see
DECOMPOSITION_GUIDE.md:

  total   denoised Z versus the raw log counts, against Z_true;
  struct  each component as labelled, against the generated component;
  canon   a convention-free projection (subject mean / population visit mean /
          remainder) applied identically to truth, model and raw data.

The naive reference uses no model: log(Y + 0.5), CLR, then the same projection.
This module never feeds truth into a fit; truth enters only the scoring.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats
from scipy.interpolate import BSpline
import torch

from evaluate import _auc, _event_truth
from event_aware_denoise import event_basis

COMPONENTS = ("baseline", "common", "individual", "event")
LOG_COMPONENTS = COMPONENTS + ("total",)
EFFECT = .2      # practical CLR effect, as cohort_analysis.EFFECT
LFSR = .05       # cumulative-lfsr discovery level, as cohort_analysis.LFSR
BH_ALPHA = .05   # naive tests use the same nominal level
TIME_GRID = 60   # population trajectory grid, as event_export.AGE_GRID
CURVE_GRID = 41  # event curve grid, as event_export.CURVE_GRID
PSEUDOCOUNT = .5  # raw log(Y + 0.5), as the raw age clock in cohort_analysis
NAN = float("nan")


# ----------------------------------------------------------------- helpers
def clr(a, axis=-1):
    a = np.asarray(a, float)
    return a - a.mean(axis, keepdims=True)


def lfsr(draws):
    """Local false sign rate along axis 0."""
    return 1 - np.maximum((draws > 0).mean(0), (draws < 0).mean(0))


def cumulative_lfsr(values):
    """Average lfsr of all taxa at least as extreme (s-value), as cohort_analysis."""
    v = np.asarray(values, float)
    out = np.full(v.shape, np.nan)
    ok = np.isfinite(v)
    if ok.any():
        order = np.argsort(v[ok])
        cum = np.cumsum(v[ok][order]) / np.arange(1, ok.sum() + 1)
        tmp = np.empty(ok.sum())
        tmp[order] = cum
        out[ok] = tmp
    return out


def bh(p):
    p = np.asarray(p, float)
    q = np.ones(p.shape)
    ok = np.isfinite(p)
    if ok.any():
        v = p[ok]
        order = np.argsort(v)
        r = v[order] * ok.sum() / np.arange(1, ok.sum() + 1)
        r = np.minimum.accumulate(r[::-1])[::-1]
        tmp = np.empty_like(v)
        tmp[order] = np.minimum(r, 1)
        q[ok] = tmp
    return q


def curve_grid(span):
    """Relative-time event grid of the real-data export."""
    return np.linspace(-min(.2 * span, 90), min(.5 * span, 180), CURVE_GRID)


def spline_rows(metadata, times):
    """The fitted, centred population spline evaluated at arbitrary times."""
    knots = np.asarray(metadata["spline_knots"])
    x = (np.asarray(times, float) - metadata["time_origin"]) / metadata["time_span"]
    design = BSpline(knots, np.eye(len(knots) - 4), 3)(np.clip(x, 0, 1))
    return (design - np.asarray(metadata["spline_center"]))[..., :-1]


def event_support(times, visit, event_times, grid, span):
    """Supported post-event grid: the real-data export rule, unchanged."""
    E = event_times.shape[1]
    radius = min(.1 * span, 30.)
    support = np.zeros((E, len(grid)), int)
    eligible = np.zeros(E, int)
    for i in range(times.shape[0]):
        t = times[i, visit[i]]
        for e in range(E):
            tau = event_times[i, e]
            if not np.isfinite(tau) or (t < tau).sum() < 2:
                continue
            eligible[e] += 1
            rel = t - tau
            support[e] += np.min(np.abs(grid[:, None] - rel[None, :]), axis=1) <= radius
    return (support >= 3) & (support >= np.ceil(.5 * np.maximum(eligible[:, None], 1))) & (grid[None] >= 0)


# ----------------------------------------------------------------- truth
def truth_components(truth):
    """Generated components on the log scale, (I, T, J) each.

    One convention is imposed because the model has it by construction: the
    population spline is centred over observed visits, so the time mean of the
    generated trend is moved into the baseline. Nothing else is reassigned; in
    particular the subject time-mean of the AR(1) factor stays 'individual'.
    """
    meta = truth.get("metadata_json")
    if meta is not None and '"design": "simple_v1"' not in str(meta):
        raise ValueError("Decomposition truth is defined for simple_v1 only")
    visit = truth["mask"].any(-1)
    common = truth["trend"] + truth["group_effect"]
    offset = common[visit].mean(0)
    common = common - offset
    shape = common.shape
    baseline = np.broadcast_to(truth["alpha"] + truth["individual"][:, None, :] + offset, shape)
    parts = dict(baseline=np.array(baseline), common=common,
                 individual=truth["shared"] + truth["specific"], event=np.array(truth["event_plus"]))
    total = sum(parts.values())
    if not np.allclose(total, truth["Z_true"], atol=1e-8):
        raise ValueError("Generated components do not sum to Z_true")
    return parts | dict(total=np.array(truth["Z_true"]))


def naive_total(truth):
    y = np.where(truth["mask"], truth["Y"], 0).astype(float)
    return np.log(y + PSEUDOCOUNT) - np.log(truth["depth"])[:, :, None]


def canonical(values, visit, weights=None):
    """Convention-free split of an (I, T, J) array on a shared visit grid.

    baseline = subject mean over observed visits; common = across-subject mean
    of the remainder at each visit; individual = what is left. Linear and
    idempotent, so it can be applied identically to truth, model and raw data.
    Optional 0/1 weights (I, T, J) restrict the means to selected cells; a
    subject-taxon or visit-taxon with no selected cell falls back to all cells.
    """
    v = np.asarray(values, float)
    full = np.broadcast_to(visit[..., None], v.shape).astype(float)
    w = full if weights is None else np.asarray(weights, float) * full

    def mean(a, weight, axis):
        n = weight.sum(axis, keepdims=True)
        fallback = (a * full).sum(axis, keepdims=True) / np.maximum(full.sum(axis, keepdims=True), 1)
        return np.where(n > 0, (a * weight).sum(axis, keepdims=True) / np.maximum(n, 1e-300), fallback)

    baseline = mean(v, w, 1)
    common = mean(v - baseline, w, 0)
    shape = v.shape
    return dict(baseline=np.broadcast_to(baseline, shape).copy(),
                common=np.broadcast_to(common, shape).copy(),
                individual=v - baseline - common)


def naive_nonzero_total(truth):
    """Zero-aware raw reference: zero counts treated as missing on the log scale,
    filled with subject mean + visit mean of the nonzero cells, then CLR."""
    visit = truth["mask"].any(-1)
    log_rate = naive_total(truth)
    nonzero = truth["mask"] & (np.where(truth["mask"], truth["Y"], 0) > 0)
    parts = canonical(log_rate, visit, nonzero)
    filled = np.where(nonzero, log_rate, parts["baseline"] + parts["common"])
    return clr(filled)


def check_shared_grid(times, visit):
    filled = np.where(visit, times, np.nan)
    spread = np.nanmax(filled, 0) - np.nanmin(filled, 0)
    if np.any(spread[np.isfinite(spread)] > 1e-8 * max(np.ptp(times), 1)):
        raise ValueError("canonical projection needs a visit grid shared by subjects")


# ----------------------------------------------------------------- posterior
@torch.no_grad()
def posterior_decomposition(estimator, times, event_times, truth_clr=None, seed_offset=50000):
    """Stream posterior draws into component means, CLR coverage counts,
    posterior cross-moments, population trajectories, event curves and gains.

    Draw-for-draw the same quantities as real_denoise/event_export.py.
    """
    m, c, meta = estimator.model_, estimator.config, estimator.metadata_
    D = c.posterior_samples
    visit = m.mask.cpu().numpy().any(-1)
    I, T, J = m.y.shape
    span = meta["time_span"]
    grid = curve_grid(span)
    q_curve = torch.as_tensor(event_basis(grid / span, c.event_basis, c.event_decay,
                                          c.event_window, c.event_delay), dtype=torch.float64)
    time_grid = np.linspace(times[visit].min(), times[visit].max(), TIME_GRID)
    spline = torch.as_tensor(spline_rows(meta, time_grid), dtype=torch.float64)
    E = m.q_event.shape[-2]
    means = {k: np.zeros((I, T, J)) for k in LOG_COMPONENTS}
    clr_sum = {k: np.zeros((I, T, J)) for k in LOG_COMPONENTS}
    clr_square = {k: np.zeros((I, T, J)) for k in LOG_COMPONENTS}
    below = {k: np.zeros((I, T, J), np.int32) for k in LOG_COMPONENTS} if truth_clr is not None else None
    first = np.zeros((I, T, J, 3))
    cross = np.zeros((I, T, J, 3, 3))
    population = np.empty((D, TIME_GRID, J), np.float32)
    curves = np.empty((D, J, E, CURVE_GRID), np.float32)
    gain = np.zeros((I, J, E))
    has_gain = c.individual_event == "gain"
    torch.manual_seed(c.seed + seed_offset)
    for k in range(D):
        w = m.values(True)
        d = m.decode(w)
        if not torch.isfinite(d["Z"]).all():
            raise FloatingPointError("Nonfinite posterior draw in decomposition")
        common = (d["trend"] + d["group_effect"]).cpu().numpy()
        individual = (d["shared"] + d["specific"]).cpu().numpy()
        event = d["event_plus"].cpu().numpy()
        total = d["Z"].cpu().numpy()
        draw = dict(baseline=total - common - individual - event, common=common,
                    individual=individual, event=event, total=total)
        for name, value in draw.items():
            means[name] += value / D
            centred = clr(value)
            clr_sum[name] += centred
            clr_square[name] += centred ** 2
            if below is not None:
                below[name] += centred < truth_clr[name]
        trio = np.stack([common, individual, event], -1)
        first += trio / D
        cross += np.einsum("...a,...b->...ab", trio, trio) / D
        alpha = -math.log(J) + 2 * w["alpha"]
        population[k] = (alpha[None] + spline @ (.7 * w["G"]).T).cpu().numpy()
        curves[k] = torch.einsum("gq,jeq->jeg", q_curve, d["theta"]).cpu().numpy()
        if has_gain:
            gain += (w["d"] * torch.exp(math.log(c.gain_scale) + .5 * w["scale_d"])[None, :, None]).cpu().numpy() / D
    clr_mean = {k: v / D for k, v in clr_sum.items()}
    clr_sd = {k: np.sqrt(np.maximum(clr_square[k] / D - clr_mean[k] ** 2, 0)) for k in LOG_COMPONENTS}
    covariance = cross - first[..., :, None] * first[..., None, :]
    support = event_support(times, visit, np.asarray(event_times, float), grid, span)
    return dict(means=means, clr_mean=clr_mean, clr_sd=clr_sd, below=below, draws=D,
                covariance=covariance, population=population, time_grid=time_grid,
                curves=curves, curve_grid=grid, curve_supported=support,
                gain=gain if has_gain else None, visit=visit)


# ----------------------------------------------------------------- statistics
def _stats(t, e):
    t, e = np.ravel(t).astype(float), np.ravel(e).astype(float)
    out = dict(rmse=float(np.sqrt(np.mean((e - t) ** 2))) if t.size else NAN,
               est_sd=float(e.std()) if t.size else NAN)
    tc, ec = t - t.mean(), e - e.mean()
    ss = float(tc @ tc)
    if t.size < 3 or ss < 1e-10 * t.size:
        out.update(r2=NAN, pearson=NAN, slope=NAN)
        return out
    out["r2"] = float(1 - np.sum((e - t) ** 2) / np.sum((t - t.mean()) ** 2))
    out["slope"] = float(tc @ ec / ss)
    out["pearson"] = float(np.corrcoef(t, e)[0, 1]) if e.std() > 1e-12 else NAN
    return out


def compare(prefix, target, estimates, cells):
    """Score every estimate on the same cells; delta = m4 - naive."""
    out = {f"{prefix}_truth_sd": float(np.asarray(target)[cells].std()) if cells.any() else NAN}
    scored = {name: _stats(np.asarray(target)[cells], np.asarray(est)[cells]) for name, est in estimates.items()}
    for name, values in scored.items():
        for stat, value in values.items():
            out[f"{prefix}_{stat}_{name}"] = value
    for reference, suffix in (("naive", "delta"), ("naive_nz", "delta_nz")):
        if {"m4", reference} <= set(scored):
            for stat in ("rmse", "r2"):
                a, b = scored["m4"][stat], scored[reference][stat]
                out[f"{prefix}_{stat}_{suffix}"] = a - b if np.isfinite([a, b]).all() else NAN
    return out


def trajectory_r(target, estimate, visit, min_sd=.1):
    """Mean within (subject, taxon) correlation over visits: shape, not level."""
    w = visit[..., None].astype(float)
    n = w.sum(1)
    def centre(a):
        return (a - (a * w).sum(1, keepdims=True) / np.maximum(n[:, None], 1)) * w
    ct, ce = centre(target), centre(estimate)
    norm = np.sqrt((ct ** 2).sum(1) * (ce ** 2).sum(1))
    r = np.divide((ct * ce).sum(1), norm, out=np.zeros_like(norm), where=norm > 1e-12)
    keep = (n >= 3) & (np.sqrt((ct ** 2).sum(1) / np.maximum(n, 1)) > min_sd) & (norm > 1e-12)
    return float(r[keep].mean()) if keep.any() else NAN


def leakage(prefix, truth_parts, estimate, cells):
    """OLS of one estimated component on all generated components (+ intercept).
    The diagonal is recovery; off-diagonal terms are leakage between labels."""
    names = [k for k in COMPONENTS if truth_parts[k][cells].std() > 1e-8]
    out = {f"{prefix}_from_{k}": NAN for k in COMPONENTS}
    if not names:
        return out
    X = np.column_stack([np.ones(int(cells.sum()))] + [truth_parts[k][cells] for k in names])
    coef = np.linalg.lstsq(X, estimate[cells], rcond=None)[0]
    out.update({f"{prefix}_from_{k}": float(v) for k, v in zip(names, coef[1:])})
    return out


def variance_shares(parts, visit):
    """cohort_analysis.variance_partition on CLR components: summed-variance shares."""
    var = {k: clr(parts[k])[visit].var(0) for k in COMPONENTS}
    total = sum(v.sum() for v in var.values())
    return {k: float(var[k].sum() / max(total, 1e-12)) for k in COMPONENTS}


def discoveries(prefix, values, label, sign, estimate):
    """Cumulative-lfsr (model) or BH q (naive) <= level; truth from generation."""
    called = values <= (LFSR if prefix.endswith("_m4") else BH_ALPHA)
    tp, fp = called & label, called & ~label
    out = {f"{prefix}_n": int(called.sum()), f"{prefix}_fp": int(fp.sum()),
           f"{prefix}_fdp": float(fp.sum() / max(called.sum(), 1)),
           f"{prefix}_power": float(tp.sum() / label.sum()) if label.any() else NAN,
           f"{prefix}_sign_error": float(np.mean(np.sign(estimate[tp]) != sign[tp])) if tp.any() else NAN}
    return out


# ----------------------------------------------------------------- inference
def trajectory_inference(truth, post):
    """Population trajectory: late-minus-early CLR change, as cohort_analysis.trajectory."""
    meta = truth["meta"]
    beta = truth["gradual_beta"]
    label, sign = beta != 0, np.sign(beta)
    grid = post["time_grid"]
    x = grid / meta["horizon"]
    q = x if meta["trend_shape"] == "linear" else x ** 2
    true_curve = clr(q[:, None] * beta[None])
    true_curve = true_curve - true_curve.mean(0)
    draws = clr(post["population"].astype(float))
    change = draws[:, -1] - draws[:, 0]
    l = lfsr(change)
    out = discoveries("traj_disc_m4", cumulative_lfsr(l), label, sign, change.mean(0))
    out["traj_auc_m4"] = _auc(1 - l, label)
    mean = draws.mean(0)
    mean = mean - mean.mean(0)
    s = _stats(true_curve, mean)
    out.update(traj_curve_rmse_m4=s["rmse"], traj_curve_pearson_m4=s["pearson"], traj_curve_slope_m4=s["slope"])
    # Naive: paired t-test of last-minus-first visit CLR across subjects, BH.
    visit = truth["mask"].any(-1)
    v = clr(naive_total(truth))
    first = np.array([v[i, np.flatnonzero(visit[i])[0]] for i in range(v.shape[0])])
    last = np.array([v[i, np.flatnonzero(visit[i])[-1]] for i in range(v.shape[0])])
    diff = last - first
    p = stats.ttest_1samp(diff, 0, axis=0).pvalue
    out.update(discoveries("traj_disc_naive", bh(p), label, sign, diff.mean(0)))
    out["traj_auc_naive"] = _auc(1 - np.nan_to_num(p, nan=1.), label)
    return out


def event_inference(truth, post, est_clr_event):
    """Event curves: supported-window mean CLR effect, as cohort_analysis.event_inference;
    subject gains and response size; a naive difference-in-differences test."""
    out = {}
    amplitude = truth["event_amplitude"][:, 0]
    label, sign = amplitude != 0, np.sign(amplitude)
    support = post["curve_supported"][0]
    grid = post["curve_grid"]
    span = float(truth["times"].max() - truth["times"].min())
    exposed = np.isfinite(truth["event_times"][:, 0])
    out["evt_supported_points"] = int(support.sum())
    if not support.any() or not exposed.any():
        return out
    curves = clr(post["curves"][:, :, 0].astype(float), axis=1)      # draws x taxa x grid
    average = curves[..., support].mean(-1)
    l = lfsr(average)
    out.update(discoveries("evt_disc_m4", cumulative_lfsr(l), label, sign, average.mean(0)))
    out["evt_auc_m4"] = _auc(1 - l, label)
    true_curve = clr(_event_truth(truth, grid / span)[0][:, 0], axis=0)  # taxa x grid
    if label.any():
        s = _stats(true_curve[label][:, support], curves.mean(0)[label][:, support])
        out.update(evt_curve_rmse_m4=s["rmse"], evt_curve_pearson_m4=s["pearson"], evt_curve_slope_m4=s["slope"])
    # Naive DiD: exposed (post - pre) versus unexposed split at the median event time.
    v = clr(naive_total(truth))
    visit = truth["mask"].any(-1)
    times, tau = truth["times"], truth["event_times"][:, 0]
    hi = grid[support].max()
    cut = np.nanmedian(tau)
    change = np.full((v.shape[0], v.shape[2]), np.nan)
    for i in range(v.shape[0]):
        t0 = tau[i] if exposed[i] else cut
        pre = visit[i] & (times[i] < t0)
        post_ = visit[i] & (times[i] >= t0) & (times[i] <= t0 + hi)
        if pre.any() and post_.any():
            change[i] = v[i, post_].mean(0) - v[i, pre].mean(0)
    a, b = change[exposed & np.isfinite(change[:, 0])], change[~exposed & np.isfinite(change[:, 0])]
    if len(a) >= 2 and len(b) >= 2:
        p = stats.ttest_ind(a, b, axis=0, equal_var=False).pvalue
        out.update(discoveries("evt_disc_naive", bh(p), label, sign, a.mean(0) - b.mean(0)))
        out["evt_auc_naive"] = _auc(1 - np.nan_to_num(p, nan=1.), label)
    # Individual event gain. simple_v1 has none (true gain = 0, every exposed
    # subject has the population response), so the model's gains measure only
    # spurious individual response; cohort_analysis uses them as an outcome.
    if post["gain"] is not None and label.any():
        g_est = post["gain"][:, :, 0]
        pick = exposed[:, None] & label[None]
        w_est = average.mean(0) ** 2
        amp_est = ((1 + g_est) * w_est).sum(1) / max(w_est.sum(), 1e-12)
        out.update(gain_abs_mean_m4=float(np.abs(g_est[pick]).mean()), gain_sd_m4=float(g_est[pick].std()),
                   gain_subject_amplification_sd_m4=float(amp_est[exposed].std()))
    # Subject event response size in the supported window (cohort_analysis outcome).
    true_event = truth["clr_parts"]["event"]
    rms_t, rms_e = [], []
    for i in np.flatnonzero(exposed):
        sel = visit[i] & (times[i] >= tau[i]) & (times[i] <= tau[i] + hi)
        if sel.any():
            rms_t.append(np.sqrt(np.mean(true_event[i, sel] ** 2)))
            rms_e.append(np.sqrt(np.mean(est_clr_event[i, sel] ** 2)))
    # Undefined when every exposed subject has the same true response size (no
    # individual gain in simple_v1; e.g. a persistent step seen over equal windows).
    if len(rms_t) >= 3 and label.any() and np.ptp(rms_t) > 1e-10 and np.ptp(rms_e) > 1e-12:
        out["evt_subject_rms_spearman_m4"] = float(stats.spearmanr(rms_t, rms_e).statistic)
    return out


def individual_inference(truth, estimates, visit):
    """Individual dynamics (truth lambda_j f_it): per subject-taxon slope over centred
    time, and the subject-level individual-dynamics size used in cohort_analysis."""
    times = truth["times"]
    centred = np.where(visit, times - (times * visit).sum(1, keepdims=True) / visit.sum(1, keepdims=True), 0)
    denom = np.maximum((centred ** 2).sum(1), 1e-12)[:, None]
    def slopes(a):
        return np.einsum("it,itj->ij", centred, np.where(visit[..., None], a, 0)) / denom
    def rms(a):
        return np.sqrt((np.where(visit[..., None], a, 0) ** 2).sum((1, 2)) / (visit.sum(1) * a.shape[2]))
    target = truth["clr_parts"]["individual"]
    out = {}
    t_slope, t_rms = slopes(target), rms(target)
    for name, est in estimates.items():
        s = _stats(t_slope, slopes(est))
        out[f"indiv_slope_pearson_{name}"] = s["pearson"]
        out[f"indiv_slope_slope_{name}"] = s["slope"]
        out[f"indiv_subject_rms_spearman_{name}"] = float(stats.spearmanr(t_rms, rms(est)).statistic)
    return out


# ----------------------------------------------------------------- driver
def evaluate_decomposition(truth, post):
    """All decomposition metrics for one dataset; returns (row, arrays to save)."""
    import json
    visit = truth["mask"].any(-1)
    check_shared_grid(truth["times"], visit)
    cells = np.broadcast_to(visit[..., None], truth["Y"].shape)
    parts = truth_components(truth)
    clr_parts = {k: clr(v) for k, v in parts.items()}
    truth = dict(truth, meta=json.loads(str(truth["metadata_json"])), clr_parts=clr_parts)
    est = post["clr_mean"]
    # Two raw references: log(Y+0.5) as is, and zeros treated as missing (naive_nz).
    raw = dict(naive=clr(naive_total(truth)), naive_nz=naive_nonzero_total(truth))
    splits = {name: canonical(v, visit) | dict(event=np.zeros_like(v)) for name, v in raw.items()}
    def with_raw(m4_value, pick):
        return dict(m4=m4_value, **{name: pick(name) for name in raw})
    row = {}
    # 1. Total: does denoising approach the truth?
    y = np.where(truth["mask"], truth["Y"], 0)
    subsets = dict(all=cells, present=cells & ~truth["S_true"],
                   sampling_zero=cells & (y == 0) & ~truth["S_true"],
                   structural_zero=cells & truth["S_true"])
    for tag, pick in subsets.items():
        if pick.any():
            row.update(compare(f"total_clr_{tag}", clr_parts["total"], with_raw(est["total"], raw.get), pick))
    row["total_traj_r_m4"] = trajectory_r(clr_parts["total"], est["total"], visit)
    for name, value in raw.items():
        row[f"total_traj_r_{name}"] = trajectory_r(clr_parts["total"], value, visit)
    row.update(compare("total_log", parts["total"], dict(m4=post["means"]["total"], naive=naive_total(truth)), cells))
    # 2. Structural: the model's labels against the generated components. Also on
    # present cells only: raw log counts cannot reach Z at structural zeros.
    scopes = {"struct": cells}
    if subsets["structural_zero"].any():
        scopes["struct_present"] = subsets["present"]
    for family, pick in scopes.items():
        for k in COMPONENTS:
            row.update(compare(f"{family}_{k}", clr_parts[k], with_raw(est[k], lambda n: splits[n][k]), pick))
        for name, keys in (("dynamic", ("common", "individual")), ("background", ("baseline", "common", "individual"))):
            row.update(compare(f"{family}_{name}", sum(clr_parts[k] for k in keys),
                               with_raw(sum(est[k] for k in keys), lambda n: sum(splits[n][k] for k in keys)), pick))
    row["struct_individual_traj_r_m4"] = trajectory_r(clr_parts["individual"], est["individual"], visit)
    for name in raw:
        row[f"struct_individual_traj_r_{name}"] = trajectory_r(clr_parts["individual"], splits[name]["individual"], visit)
    for k in COMPONENTS:
        row.update(leakage(f"leak_m4_{k}", clr_parts, est[k], cells))
    # 3. Canonical: the same projection for truth, model and raw data.
    truth_bg = canonical(clr_parts["total"] - clr_parts["event"], visit)
    model_bg = canonical(est["total"] - est["event"], visit)
    truth_total = canonical(clr_parts["total"], visit)
    model_total = canonical(est["total"], visit)
    for k in ("baseline", "common", "individual"):
        row.update(compare(f"canon_bg_{k}", truth_bg[k], with_raw(model_bg[k], lambda n: splits[n][k]), cells))
        row.update(compare(f"canon_total_{k}", truth_total[k], with_raw(model_total[k], lambda n: splits[n][k]), cells))
        if "struct_present" in scopes:
            row.update(compare(f"canon_bg_present_{k}", truth_bg[k], with_raw(model_bg[k], lambda n: splits[n][k]),
                               subsets["present"]))
        # Gap: labelled split versus its own canonical projection.
        row[f"label_vs_canon_{k}_rmse_m4"] = float(np.sqrt(np.mean((est[k] - model_bg[k])[cells] ** 2)))
    # 4. Variance partition (cohort_analysis).
    shares = dict(truth=variance_shares(clr_parts, visit), m4=variance_shares(est, visit),
                  **{name: variance_shares(split, visit) for name, split in splits.items()})
    for k in COMPONENTS:
        for name, value in shares.items():
            row[f"vp_{k}_share_{name}"] = value[k]
        for name in shares:
            if name != "truth":
                row[f"vp_{k}_abs_error_{name}"] = abs(shares[name][k] - shares["truth"][k])
    # 5. Interval calibration and posterior coupling.
    if post["below"] is not None:
        D = post["draws"]
        for k in LOG_COMPONENTS:
            fraction = post["below"][k][cells] / D
            row[f"cov_{k}_m4"] = float(np.mean((fraction >= .025) & (fraction <= .975)))
            row[f"cov_{k}_sd_m4"] = float(post["clr_sd"][k][cells].mean())
    covariance = post["covariance"][cells]
    for (a, b), name in (((0, 1), "common_individual"), ((0, 2), "common_event"), ((1, 2), "individual_event")):
        denom = np.sqrt(covariance[:, a, a] * covariance[:, b, b])
        ok = denom > 1e-12
        r = covariance[ok, a, b] / denom[ok]
        row[f"postcorr_{name}_median_m4"] = float(np.median(r)) if r.size else NAN
        row[f"postcorr_{name}_below_minus07_m4"] = float(np.mean(r < -.7)) if r.size else NAN
    # 6. Inference used on real data.
    row.update(trajectory_inference(truth, post))
    row.update(event_inference(truth, post, est["event"]))
    row.update(individual_inference(truth, dict(m4=est["individual"],
                                                **{n: split["individual"] for n, split in splits.items()}), visit))
    arrays = {f"m4_{k}_clr_mean": est[k].astype(np.float32) for k in LOG_COMPONENTS}
    arrays.update({f"m4_{k}_clr_sd": post["clr_sd"][k].astype(np.float32) for k in LOG_COMPONENTS})
    arrays.update(population_log_mean=post["population"].mean(0), time_grid=post["time_grid"],
                  event_curve_mean=post["curves"].mean(0), curve_grid=post["curve_grid"],
                  curve_supported=post["curve_supported"])
    if post["gain"] is not None:
        arrays["gain_mean"] = post["gain"].astype(np.float32)
    return row, arrays
