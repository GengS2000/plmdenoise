"""Truth-aware scoring for the simulation study.

Every method is scored on two scales: latent abundance (the depth- and
batch-free X* the model targets) and relative abundance. Event fidelity,
zero-state discrimination, interval coverage and null false positives are
reported separately. This is the only module that reads simulation truth;
the denoiser never does.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

from simulate_microbiome import _event_curve

# Population event gain for group 1; mirrors (1 + .3 * group) in the simulator.
GROUP_EVENT_GAIN = 1.3
NAN = float("nan")


def _pearson(a, b):
    a, b = np.ravel(a), np.ravel(b)
    if a.size < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return NAN
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a, b):
    a, b = np.ravel(a), np.ravel(b)
    if a.size < 3:
        return NAN
    value = spearmanr(a, b).statistic
    return float(value) if np.isfinite(value) else NAN


def _slope(truth, estimate, origin=False):
    """OLS slope of estimate on truth; below one means attenuation."""
    t, e = np.ravel(truth).astype(float), np.ravel(estimate).astype(float)
    if t.size < 2:
        return NAN
    if not origin:
        t, e = t - t.mean(), e - e.mean()
    denominator = float(t @ t)
    return float(t @ e / denominator) if denominator > 1e-12 else NAN


def _auc(score, label):
    """Rank AUC with tied scores averaged; NaN when one class is empty."""
    label, score = np.ravel(label).astype(bool), np.ravel(score).astype(float)
    n1, n0 = int(label.sum()), int((~label).sum())
    if n1 == 0 or n0 == 0:
        return NAN
    rank = rankdata(score)
    return float((rank[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def bayes_fdr(null_probability, alpha):
    """Reject the largest set whose mean posterior null probability stays <= alpha."""
    p = np.ravel(np.asarray(null_probability, float))
    if not 0 < alpha < 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("FDR alpha must be in (0,1), probabilities finite and in [0,1]")
    order = np.argsort(p, kind="stable")
    running = np.cumsum(p[order]) / np.arange(1, p.size + 1)
    reject = np.zeros(p.size, bool)
    reject[order[:int(np.searchsorted(running > alpha, True))]] = True
    return reject.reshape(np.shape(null_probability))


def _close(p):
    total = p.sum(-1, keepdims=True)
    return np.divide(p, total, out=np.zeros_like(p, dtype=float), where=total > 0)


def _clr(p, delta):
    """Centred log ratio after flooring BOTH sides at the same detection limit.
    A structural zero in the truth and a negligible positive estimate are then
    scored as equal; replacing only the exact zeros would compare a pseudo-count
    against a real number and swamp the distance."""
    log_q = np.log(_close(np.maximum(p, delta)))
    return log_q - log_q.mean(-1, keepdims=True)


def _align(target, estimate, mask, floor):
    """Strip the per-sample multiplicative constant. Compositional mechanisms
    identify latent abundance only up to that constant, so read the aligned
    figure whenever metadata['target'] is clean_composition."""
    log_t = np.log(target + floor)
    log_e = np.log(np.maximum(estimate, 0) + floor)
    shift = np.where(mask, log_e - log_t, 0).sum(2, keepdims=True) / np.maximum(mask.sum(2, keepdims=True), 1)
    return np.maximum(np.exp(log_e - shift) - floor, 0)


def _accuracy(target, estimate, mask, floor, prefix):
    t = np.log(target[mask] + floor)
    e = np.log(np.maximum(estimate, 0)[mask] + floor)
    return {f"{prefix}_log_rmse": float(np.sqrt(np.mean((e - t) ** 2))),
            f"{prefix}_log_mae": float(np.mean(np.abs(e - t))),
            f"{prefix}_log_bias": float(np.mean(e - t)),
            f"{prefix}_pearson": _pearson(t, e),
            f"{prefix}_spearman": _spearman(t, e),
            f"{prefix}_slope": _slope(t, e)}


def _dynamics(target, estimate, mask, floor, prefix, min_sd=.1):
    """Within (subject, taxon) correlation across visits: trajectory shape rather
    than level. Flat truths carry no shape and are dropped."""
    log_t = np.log(target + floor)
    log_e = np.log(np.maximum(estimate, 0) + floor)
    n = mask.sum(1)
    centred_t = (log_t - ((log_t * mask).sum(1) / np.maximum(n, 1))[:, None]) * mask
    centred_e = (log_e - ((log_e * mask).sum(1) / np.maximum(n, 1))[:, None]) * mask
    sd_t = np.sqrt((centred_t ** 2).sum(1) / np.maximum(n, 1))
    norm = np.sqrt((centred_t ** 2).sum(1) * (centred_e ** 2).sum(1))
    r = np.divide((centred_t * centred_e).sum(1), norm, out=np.zeros_like(norm), where=norm > 1e-12)
    keep = (n >= 3) & (sd_t > min_sd) & (norm > 1e-12)
    return {f"{prefix}_trajectory_r": float(r[keep].mean()) if keep.any() else NAN,
            f"{prefix}_trajectory_pairs": int(keep.sum())}


def _scale_block(target, estimate, mask, floor, prefix, compositional=False):
    out = {**_accuracy(target, estimate, mask, floor, prefix),
           **_dynamics(target, estimate, mask, floor, prefix)}
    if compositional:
        visit = mask.all(axis=2)
        t, e = _close(target)[visit], _close(np.maximum(estimate, 0))[visit]
        out[f"{prefix}_bray_curtis"] = float(np.mean(.5 * np.abs(t - e).sum(-1)))
        out[f"{prefix}_aitchison"] = float(np.mean(np.sqrt(((_clr(t, floor) - _clr(e, floor)) ** 2).sum(-1))))
    return out


def _sampling_zero(truth):
    """Observed zeros at cells where the taxon is in fact present."""
    zero = truth["mask"] & (np.where(truth["mask"], truth["Y"], 0) == 0)
    return zero & ~truth["S_true"], zero & truth["S_true"]


def _estimators(truth, result, label):
    """Latent-abundance estimators on one scale, sequencing depth removed from all.

    oracle_batch additionally divides out the true technical effect. oracle_zero
    is handed the absence *label* but never the latent value: truly absent cells
    go to zero, and sampling zeros to the taxon's median observed positive rate.
    Zeroing absent cells alone would change nothing, since the naive rate is
    already zero wherever the count is zero.
    """
    rate = np.where(truth["mask"], truth["Y"], 0) / truth["depth"][:, :, None]
    corrected = rate / np.exp(truth["technical_effect"])
    sampling_zero, absent = _sampling_zero(truth)
    observed_positive = np.where(truth["mask"] & (rate > 0), corrected, np.nan)
    typical = np.zeros(rate.shape[-1])
    for j in range(rate.shape[-1]):
        column = observed_positive[..., j]
        typical[j] = np.median(column[np.isfinite(column)]) if np.isfinite(column).any() else 0.0
    oracle = np.where(sampling_zero, typical, corrected)
    estimators = {"naive": rate, "oracle_batch": corrected, "oracle_zero": np.where(absent, 0.0, oracle)}
    if result is not None:
        estimators[label] = result["X_mean"]
    return estimators


def _hard_cells(truth, estimate, floor):
    """Error restricted to the cells the zero model exists to resolve: an observed
    zero is either sampling noise on a present taxon or a genuine absence."""
    out = {}
    for tag, cell in zip(("sampling_zero", "true_absence"), _sampling_zero(truth)):
        if not cell.any():
            out[f"count_log_rmse_{tag}"] = out[f"count_log_bias_{tag}"] = NAN
            continue
        t = np.log(truth["X_true"][cell] + floor)
        e = np.log(np.maximum(estimate, 0)[cell] + floor)
        out[f"count_log_rmse_{tag}"] = float(np.sqrt(np.mean((e - t) ** 2)))
        out[f"count_log_bias_{tag}"] = float(np.mean(e - t))
    out["n_sampling_zero"] = int(_sampling_zero(truth)[0].sum())
    return out


def _zero_block(truth, result):
    """Discrimination is only a real task on the zeros; positive counts pin S=0."""
    mask, S = truth["mask"], truth["S_true"]
    rho = result["absence_probability_mean"]
    zero = mask & (np.where(mask, truth["Y"], 0) == 0)
    return {"absence_auc": _auc(rho[mask], S[mask]),
            "absence_auc_on_zeros": _auc(rho[zero], S[zero]),
            "absence_brier": float(np.mean((rho[mask] - S[mask]) ** 2)),
            "zero_fraction": float(zero.sum() / max(mask.sum(), 1)),
            "sampling_zero_share": float((zero & ~S).sum() / max(zero.sum(), 1))}


def _event_truth(truth, s):
    """Population response curves on the model's dimensionless grid."""
    if "event_truth_grid" in truth:
        span = float(truth["times"].max() - truth["times"].min())
        grid = truth["event_truth_grid"] / span
        if s.min() < grid.min() - 1e-9 or s.max() > grid.max() + 1e-9:
            raise ValueError("Requested event grid exceeds saved simulation truth support")
        def interpolate(name):
            values = truth[name]
            return np.array([[np.interp(s, grid, curve) for curve in row] for row in values])
        plus, zero = interpolate("H_plus_true"), interpolate("H_zero_true")
        return plus, plus * float(truth.get("event_group_gain", GROUP_EVENT_GAIN)), zero
    shapes = truth["event_shapes"]
    J, E = shapes.shape
    curve = np.empty((J, E, s.size))
    for j in range(J):
        for e in range(E):
            curve[j, e] = _event_curve(s, int(shapes[j, e]))
    amplitude = truth["event_amplitude"][..., None]
    return curve * amplitude, curve * amplitude * GROUP_EVENT_GAIN, curve * truth["event_amplitude_zero"][..., None]


def _event_support(truth, s, span, minimum=.5):
    """Fraction of exposed subjects whose observed relative time reaches each grid
    point. The model extrapolates outside this region, so score the two apart."""
    visit = truth["mask"].any(axis=2)
    x = np.where(visit, truth["times"] / span, np.nan)
    tau = truth["event_times"] / span
    exposed = np.isfinite(tau)
    low = np.nanmin(x, axis=1)[:, None] - tau
    high = np.nanmax(x, axis=1)[:, None] - tau
    inside = (low[..., None] <= s) & (s <= high[..., None]) & exposed[..., None]
    return inside.sum(0) / np.maximum(exposed.sum(0), 1)[:, None] >= minimum


def _event_block(truth, result, span, alpha, threshold):
    s = result["event_grid"] / span
    truth_plus, truth_group1, truth_zero = _event_truth(truth, s)
    support = result.get("event_support", _event_support(truth, s, span))
    channels = truth["event_channels"]
    responds = {"plus": np.isin(channels, [1, 3]), "zero": np.isin(channels, [2, 3])}
    out = {"n_responders_plus": int(responds["plus"].sum()),
           "n_responders_zero": int(responds["zero"].sum()),
           "n_taxon_event_pairs": int(channels.size),
           "grid_supported_fraction": float(support.mean()),
           "effect_threshold": threshold}
    for key, target, estimate in (("plus", truth_plus, result["H_plus_mean"]),
                                  ("group1", truth_group1, result["H_plus_group1_mean"]),
                                  ("zero", truth_zero, result["H_zero_mean"])):
        responder = responds["zero" if key == "zero" else "plus"]
        inside = np.broadcast_to(support[None], target.shape)
        error = (estimate - target) ** 2
        for tag, cell in (("resp", inside & responder[..., None]), ("null", inside & ~responder[..., None]),
                          ("extrap", ~inside & responder[..., None])):
            out[f"H_{key}_rmse_{tag}"] = float(np.sqrt(error[cell].mean())) if cell.any() else NAN
        weight = np.where(inside, np.abs(target), -np.inf)
        index = weight.argmax(-1)[..., None]
        peak_t = np.take_along_axis(target, index, -1)[..., 0]
        peak_e = np.take_along_axis(estimate, index, -1)[..., 0]
        pick = responder & np.broadcast_to(support.any(-1)[None], responder.shape)
        out[f"H_{key}_peak_slope"] = _slope(peak_t[pick], peak_e[pick], origin=True)
        out[f"H_{key}_sign_accuracy"] = float(np.mean(np.sign(peak_e[pick]) == np.sign(peak_t[pick]))) if pick.any() else NAN
        shape = [_pearson(target[j, e][support[e]], estimate[j, e][support[e]]) for j, e in np.argwhere(pick)]
        shape = [v for v in shape if np.isfinite(v)]
        out[f"H_{key}_shape_r"] = float(np.mean(shape)) if shape else NAN
    testable = np.broadcast_to(result.get("event_testable", np.ones(channels.shape[-1], bool)), channels.shape)
    out["n_testable_pairs"] = int(testable.sum())
    out["detection_scope"] = "supported_post_event" if "event_support" in result else "legacy_full_grid"
    for key, score, responder in (("plus", result.get("H_plus_p_any_supported", result["H_plus_p_any"]), responds["plus"]),
                                  ("zero", result.get("H_zero_p_any_supported", result["H_zero_p_any"]), responds["zero"]),
                                  ("any", 1 - result.get("event_p_joint_null_group0_supported", result["event_p_joint_null_group0"]), channels > 0)):
        score, responder = score[testable], responder[testable]
        reject = bayes_fdr(1 - score, alpha)
        true_positive = int((reject & responder).sum())
        discoveries = int(reject.sum())
        out[f"detect_{key}_auc"] = _auc(score, responder)
        out[f"detect_{key}_discoveries"] = discoveries
        out[f"detect_{key}_fdr"] = (discoveries - true_positive) / max(discoveries, 1)
        out[f"detect_{key}_false_positives"] = discoveries - true_positive
        out[f"detect_{key}_any_false_positive"] = int(discoveries > true_positive)
        out[f"detect_{key}_fpr"] = float((reject & ~responder).sum() / (~responder).sum()) if (~responder).any() else NAN
        out[f"detect_{key}_power"] = true_positive / int(responder.sum()) if responder.any() else NAN
        for level in (.5, .9):
            out[f"detect_{key}_fpr_{int(level * 100)}"] = float((score[~responder] > level).mean()) if (~responder).any() else NAN
    return out


def _component_block(truth, result, mask):
    """Only Z, the event channels and the batch term are compared directly. The
    trend / shared / specific split is not separately identifiable (factor
    rotation, trend-versus-residual trade-off), so their sum is scored instead."""
    def score(name, target, estimate):
        residual = (estimate - target)[mask]
        return {f"{name}_rmse": float(np.sqrt(np.mean(residual ** 2))),
                f"{name}_rmse_centred": float(np.sqrt(np.mean((residual - residual.mean()) ** 2))),
                f"{name}_pearson": _pearson(target[mask], estimate[mask])}
    out = {}
    out.update(score("Z", truth["Z_true"], result["Z_mean"]))
    out.update(score("background", truth["Z_true"] - truth["event_plus"],
                     result["Z_mean"] - result["event_plus_mean"]))
    out.update(score("event_plus", truth["event_plus"], result["event_plus_mean"]))
    out.update(score("event_zero", truth["event_zero"], result["event_zero_mean"]))
    out.update(score("batch", truth["batch_effect"], result["batch_effect_mean"]))
    return out


def _coverage_block(truth, result, span, floor, nominal):
    mask = truth["mask"]
    inside = (result["Z_lower"] <= truth["Z_true"]) & (truth["Z_true"] <= result["Z_upper"])
    out = {"nominal_interval": nominal, "coverage_Z": float(inside[mask].mean()),
           "width_Z": float((result["Z_upper"] - result["Z_lower"])[mask].mean())}
    inside = (result["X_lower"] <= truth["X_true"]) & (truth["X_true"] <= result["X_upper"])
    out["coverage_X"] = float(inside[mask].mean())
    out["width_X_log"] = float(np.log((result["X_upper"][mask] + floor) / (result["X_lower"][mask] + floor)).mean())
    s = result["event_grid"] / span
    support = result.get("event_support", _event_support(truth, s, span))
    curves = dict(zip(("plus", "group1", "zero"), _event_truth(truth, s)))
    for key, low, high in (("plus", "H_plus_lower", "H_plus_upper"),
                           ("group1", "H_plus_group1_lower", "H_plus_group1_upper"),
                           ("zero", "H_zero_lower", "H_zero_upper")):
        target = curves[key]
        cell = np.broadcast_to(support[None], target.shape)
        inside = (result[low] <= target) & (target <= result[high])
        out[f"coverage_H_{key}"] = float(inside[cell].mean()) if cell.any() else NAN
        out[f"width_H_{key}"] = float((result[high] - result[low])[cell].mean()) if cell.any() else NAN
        responder = np.isin(truth["event_channels"], [2, 3] if key == "zero" else [1, 3])
        for tag, active in (("active", responder), ("null", ~responder)):
            pick = cell & active[..., None]
            out[f"coverage_H_{key}_{tag}"] = float(inside[pick].mean()) if pick.any() else NAN
    return out


def _recovery_diagnostics(truth, result, mask):
    """Mask-aware within-trajectory variation; component ratios are diagnostic only."""
    t, e = truth["Z_true"], result["Z_mean"]
    n = mask.sum(1, keepdims=True)
    def centred(a):
        return (a - np.where(mask, a, 0).sum(1, keepdims=True) / np.maximum(n, 1))
    ct, ce = centred(t), centred(e)
    vt = np.where(mask, ct ** 2, 0).sum(1) / np.maximum(n[:, 0], 1)
    ve = np.where(mask, ce ** 2, 0).sum(1) / np.maximum(n[:, 0], 1)
    keep = (n[:, 0] >= 3) & (vt > 1e-8)
    out = {"Z_temporal_sd_ratio_median": float(np.median(np.sqrt(ve[keep] / vt[keep]))) if keep.any() else NAN}
    dt = np.diff(truth["times"], axis=1)[..., None]
    adjacent = mask[:, 1:] & mask[:, :-1]
    true_rough = np.mean((np.diff(t, axis=1) / dt)[adjacent] ** 2) if adjacent.any() else 0
    est_rough = np.mean((np.diff(e, axis=1) / dt)[adjacent] ** 2) if adjacent.any() else 0
    out["Z_roughness_ratio"] = float(np.sqrt(est_rough / true_rough)) if true_rough > 1e-12 else NAN
    for name in ("trend", "shared", "specific"):
        true_sd = truth[name][mask].std()
        est_sd = result[name + "_mean"][mask].std()
        out[name + "_sd_ratio"] = float(est_sd / true_sd) if true_sd > 1e-12 else NAN
        out[name + "_estimated_sd"] = float(est_sd)
    mse = float(np.mean((result["event_plus_mean"][mask] - truth["event_plus"][mask]) ** 2))
    for name, population in (("oracle_population", truth["event_population"]),
                             ("fitted_population", result.get("event_population_mean"))):
        if population is not None:
            base = float(np.mean((population[mask] - truth["event_plus"][mask]) ** 2))
            out[f"event_plus_r2_vs_{name}"] = 1 - mse / base if base > 1e-12 else NAN
            out[f"event_plus_mse_{name}"] = base
    event_times = truth.get("event_times_true", truth["event_times"])
    relative = truth["times"][..., None] - event_times[:, None, :]
    span = float(np.ptp(truth["times"]))
    near = np.any(np.isfinite(relative) & (np.abs(relative) <= .16 * span), axis=-1)
    for tag, visits in (("near_event", near), ("far_event", ~near)):
        pick = mask & visits[..., None]
        out["Z_rmse_" + tag] = float(np.sqrt(np.mean((e[pick] - t[pick]) ** 2))) if pick.any() else NAN
    return out


def evaluate(truth, result=None, label="denoised", alpha=.1, condition="base", extra=None):
    """Score one dataset. Returns one row per method; baselines need no result."""
    meta = json.loads(str(truth["metadata_json"]))
    mask = truth["mask"].astype(bool)
    floor = 1.0 / float(np.median(truth["depth"]))
    span = float(truth["times"].max() - truth["times"].min())
    fit = json.loads(str(result["metadata_json"])) if result is not None else None
    context = {"scenario": meta["scenario"], "scenario_name": meta["scenario_name"],
               "mechanism": meta["mechanism"], "target": meta["target"], "condition": condition,
               "seed": meta["seed"], "n_subjects": meta["n_subjects"], "n_visits": meta["n_visits"],
               "n_taxa": meta["n_taxa"], "median_depth": meta["median_depth"],
               "observed_fraction": float(mask.mean())}
    context.update(extra or {})
    rows = []
    for name, estimate in _estimators(truth, result, label).items():
        row = dict(context, method=name)
        row.update(_scale_block(truth["X_true"], estimate, mask, floor, "count"))
        row["count_log_rmse_aligned"] = _accuracy(
            truth["X_true"], _align(truth["X_true"], estimate, mask, floor), mask, floor, "a")["a_log_rmse"]
        row.update(_hard_cells(truth, estimate, floor))
        row.update(_scale_block(truth["clean_composition"], _close(np.maximum(estimate, 0)), mask, floor, "comp", True))
        if fit is not None and name == label:
            row.update(_component_block(truth, result, mask))
            row.update(_recovery_diagnostics(truth, result, mask))
            row.update(_coverage_block(truth, result, span, floor, fit["config"]["interval"]))
            row.update(_event_block(truth, result, span, alpha, fit["config"]["effect_threshold"]))
            if fit["config"]["zero_inflated"]:
                row.update(_zero_block(truth, result))
            row["diagnostics"] = "; ".join(fit["diagnostics"]) or "none"
            row["converged"] = fit.get("converged")
            row["vi_steps_run"] = fit.get("vi_steps_run")
            budget = fit.get("parameter_budget", {})
            row["latent_means"] = sum(b["means"] for b in budget.values()) if budget else None
            row["variational_parameters"] = sum(b["means"] + b["covariance_storage"] for b in budget.values()) if budget else None
            corr = [v for v in fit.get("event_trend_canonical_correlation", []) if v is not None]
            pre = [v for rr in fit.get("event_pre_visits", []) for v in rr if v is not None]
            row["event_trend_max_corr"] = max(corr) if corr else NAN
            row["pre_event_visits_min"] = min(pre) if pre else NAN
        rows.append(row)
    return rows


def format_table(rows, columns=None, digits=4):
    """Markdown table; missing keys render as '-' so mixed rows stay aligned."""
    if not rows:
        return "(no rows)"
    columns = list(columns) if columns else list(dict.fromkeys(k for r in rows for k in r))
    def cell(value):
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return "-"
        return f"{value:.{digits}g}" if isinstance(value, float) else str(value)
    body = [[cell(r.get(c)) for c in columns] for r in rows]
    width = [max(len(c), *(len(b[i]) for b in body)) for i, c in enumerate(columns)]
    def line(cells):
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, width)) + " |"
    return "\n".join([line(columns), "|" + "|".join("-" * (w + 2) for w in width) + "|", *(line(b) for b in body)])


SUMMARY = ("method", "count_log_rmse", "count_trajectory_r", "comp_aitchison", "comp_bray_curtis",
           "comp_trajectory_r", "absence_auc_on_zeros", "H_plus_peak_slope", "H_plus_shape_r",
           "detect_any_auc", "detect_any_fdr", "detect_any_power", "coverage_X", "coverage_H_plus",
           "Z_rmse", "coverage_Z", "width_Z", "event_plus_rmse", "detect_zero_fpr",
           "Z_temporal_sd_ratio_median", "Z_roughness_ratio", "event_plus_r2_vs_oracle_population",
           "converged", "latent_means")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("truth", type=Path)
    parser.add_argument("result", type=Path, nargs="?", help="Denoised NPZ; omit to score baselines only")
    parser.add_argument("--label", default="denoised")
    parser.add_argument("--alpha", type=float, default=.1, help="Target Bayesian FDR")
    parser.add_argument("--condition", default="base")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--full", action="store_true", help="Print every metric, not the summary set")
    args = parser.parse_args()
    with np.load(args.truth, allow_pickle=False) as loaded:
        truth = {key: loaded[key] for key in loaded.files}
    result = None
    if args.result:
        with np.load(args.result, allow_pickle=False) as loaded:
            result = {key: loaded[key] for key in loaded.files}
    rows = evaluate(truth, result, args.label, args.alpha, args.condition)
    print(format_table(rows, None if args.full else SUMMARY))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
