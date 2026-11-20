"""Truth / observation / denoised demo arrays and their server-side summary.

save_demo() is called by run_validation.py after a fit (``run --save-demo``)
and writes OUT/demo/<task_id>.npz: the simulated truth, the observed counts and
the posterior on one grid. It never changes the fit or the scored row.

``python demo_report.py --out OUT`` (run by validation_merge.sbatch) reads those
arrays and writes OUT/demo_report/: report.json (tables, exemplar trajectories,
event curves, scatter subsample), summary.md, and PNG figures when matplotlib
is importable. All numbers use evaluate.py's scale: log(X + 1/median depth).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from evaluate import _event_truth

# Relative-time windows in units of the observed time span (event shapes
# decay on a 0.08 scale, so 0.25 holds the local response).
LOCAL_WINDOW = .25
RESPONDER_CHANNELS = (1, 3)


def _f32(value):
    return np.asarray(value, dtype=np.float32)


def save_demo(path, task, truth, result, mask):
    """Compact float32 arrays for one fit; atomic write."""
    span = float(truth["times"].max() - truth["times"].min())
    h_plus, _, h_zero = _event_truth(truth, result["event_grid"] / span)
    arrays = dict(
        Y=truth["Y"].astype(np.int32), mask=mask.astype(bool), depth=truth["depth"].astype(np.int64),
        times=_f32(truth["times"]), group=truth["group"], batch=truth["batch"],
        event_times=_f32(truth["event_times"]), event_times_true=_f32(truth["event_times_true"]),
        X_true=_f32(truth["X_true"]), Z_true=_f32(truth["Z_true"]), S_true=truth["S_true"].astype(bool),
        clean_composition=_f32(truth["clean_composition"]),
        event_plus_true=_f32(truth["event_plus"]), event_channels=truth["event_channels"],
        H_plus_true=_f32(h_plus), H_zero_true=_f32(h_zero), event_grid=_f32(result["event_grid"]),
        event_support=result["event_support"],
        meta=np.array(json.dumps({k: task[k] for k in ("task_id", "scenario", "condition", "seed", "variant",
                                                      "fit_repeat", "simulation")}, ensure_ascii=False)),
    )
    for name in ("X_mean", "X_lower", "X_upper", "Z_mean", "Z_lower", "Z_upper", "event_plus_mean",
                 "absence_probability_mean", "H_plus_mean", "H_plus_lower", "H_plus_upper",
                 "H_plus_p_any_supported", "event_p_joint_null_group0_supported"):
        arrays[name] = _f32(result[name])
    x = np.maximum(np.asarray(result["X_mean"], dtype=float), 0)
    total = x.sum(axis=-1, keepdims=True)
    arrays["comp_mean"] = _f32(np.divide(x, total, out=np.zeros_like(x), where=total > 0))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + f".{os.getpid()}.tmp.npz")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _rmse(a, b, cell):
    return float(np.sqrt(np.mean((a[cell] - b[cell]) ** 2))) if cell.any() else None


def _corr(a, b, cell):
    if cell.sum() < 3:
        return None
    x, y = a[cell], b[cell]
    return float(np.corrcoef(x, y)[0, 1]) if x.std() > 1e-9 and y.std() > 1e-9 else None


def _pre_mean(log_value, visit, before):
    """Per subject/taxon mean over observed pre-event visits (NaN if none)."""
    use = (visit & before)[:, :, None]
    n = use.sum(1)
    total = np.where(use, log_value, 0).sum(1)
    return np.where(n > 0, total / np.maximum(n, 1), np.nan)


def analyse(d):
    meta = json.loads(str(d["meta"]))
    mask = d["mask"]
    visit = mask.any(-1)
    floor = 1.0 / float(np.median(d["depth"]))
    span = float(d["times"].max() - d["times"].min())
    rate = np.where(mask, d["Y"], 0) / d["depth"][:, :, None]
    log_true = np.log(d["X_true"].astype(float) + floor)
    log_obs = np.log(rate + floor)
    log_den = np.log(np.maximum(d["X_mean"], 0).astype(float) + floor)
    responder = np.isin(d["event_channels"], RESPONDER_CHANNELS).any(-1)  # abundance channel, (J,)
    null_taxon = ~(d["event_channels"] > 0).any(-1)  # no channel at all (absence-only responders excluded)
    tau = d["event_times"][:, 0].astype(float)
    exposed = np.isfinite(tau)
    rel = np.where(exposed[:, None], (d["times"] - tau[:, None]) / span, np.nan)
    windows = {"all": visit,
               "pre_event": visit & exposed[:, None] & (rel < 0),
               "post_local": visit & (rel >= 0) & (rel <= LOCAL_WINDOW),
               "post_late": visit & (rel > LOCAL_WINDOW),
               "unexposed": visit & ~exposed[:, None]}

    # 1. Level recovery: observation vs denoised, split by time window and taxon type.
    level = []
    for wname, wcell in windows.items():
        for tname, taxa in (("all_taxa", np.ones_like(responder)), ("responders", responder), ("null_taxa", null_taxon)):
            cell = wcell[:, :, None] & taxa[None, None] & mask
            if cell.any():
                level.append(dict(window=wname, taxa=tname, n_cells=int(cell.sum()),
                                  rmse_observed=_rmse(log_obs, log_true, cell),
                                  rmse_denoised=_rmse(log_den, log_true, cell)))
    zero = mask & (rate == 0)
    zero_cells = {"sampling_zero": zero & ~d["S_true"], "true_absence": zero & d["S_true"]}
    for zname, cell in zero_cells.items():
        if cell.any():
            level.append(dict(window="all", taxa=zname, n_cells=int(cell.sum()),
                              rmse_observed=_rmse(log_obs, log_true, cell),
                              rmse_denoised=_rmse(log_den, log_true, cell)))
    inside = (d["Z_lower"] <= d["Z_true"]) & (d["Z_true"] <= d["Z_upper"])

    # 2. Local event effect: change from the subject's own pre-event mean.
    before = rel < 0
    delta = {name: value - _pre_mean(value, visit, before)[:, None, :]
             for name, value in (("true", log_true), ("observed", log_obs), ("denoised", log_den))}
    post = (windows["post_local"])[:, :, None] & mask & np.isfinite(delta["true"])
    event = []
    for tname, taxa in (("responders", responder), ("null_taxa", null_taxon)):
        cell = post & taxa[None, None]
        if cell.any():
            event.append(dict(taxa=tname, n_cells=int(cell.sum()),
                              delta_rmse_observed=_rmse(delta["observed"], delta["true"], cell),
                              delta_rmse_denoised=_rmse(delta["denoised"], delta["true"], cell),
                              delta_r_observed=_corr(delta["observed"], delta["true"], cell),
                              delta_r_denoised=_corr(delta["denoised"], delta["true"], cell)))
    cell = post & responder[None, None]
    component = dict(event_plus_rmse_post_local=_rmse(d["event_plus_mean"], d["event_plus_true"], cell),
                     event_plus_r_post_local=_corr(d["event_plus_mean"], d["event_plus_true"], cell))

    # 3. Exemplars: strongest responders with enough signal, one null taxon.
    abundance = np.where(mask, log_true, np.nan)
    mean_level = np.nanmean(abundance, axis=(0, 1))
    peak = np.abs(d["H_plus_true"][:, 0]).max(-1)
    score = np.where(responder, peak * (mean_level > np.nanpercentile(mean_level, 40)), -np.inf)
    taxa = [int(j) for j in np.argsort(-score)[:3] if np.isfinite(score[j])]
    nulls = np.flatnonzero(null_taxon)
    if nulls.size:
        taxa.append(int(nulls[np.argmin(np.abs(mean_level[nulls] - np.nanmedian(mean_level[taxa] if taxa else mean_level)))]))
    n_pre = (visit & before).sum(1)
    candidates = np.flatnonzero(exposed & (n_pre >= 2))
    candidates = candidates[np.argsort(tau[candidates])]
    subjects = [int(candidates[k]) for k in np.linspace(0, len(candidates) - 1, 3).round().astype(int)] if len(candidates) >= 3 \
        else [int(i) for i in candidates]
    to_log = lambda a: np.log(np.maximum(a, 0).astype(float) + floor)
    trajectories = []
    for j in taxa:
        for i in subjects:
            t = visit[i]
            trajectories.append(dict(
                taxon=j, subject=i, responder=bool(responder[j]), channel=int(d["event_channels"][j, 0]),
                event_time=float(tau[i]), times=d["times"][i][t].tolist(),
                truth=log_true[i, t, j].tolist(), observed=log_obs[i, t, j].tolist(),
                observed_zero=(rate[i, t, j] == 0).tolist(), denoised=log_den[i, t, j].tolist(),
                lower=to_log(d["X_lower"][i, t, j]).tolist(), upper=to_log(d["X_upper"][i, t, j]).tolist(),
                truth_absent=d["S_true"][i, t, j].tolist()))
    curves = [dict(taxon=j, responder=bool(responder[j]), grid=d["event_grid"].tolist(),
                   support=d["event_support"][0].tolist(), truth=d["H_plus_true"][j, 0].tolist(),
                   mean=d["H_plus_mean"][j, 0].tolist(), lower=d["H_plus_lower"][j, 0].tolist(),
                   upper=d["H_plus_upper"][j, 0].tolist(),
                   p_any=float(d["H_plus_p_any_supported"][j, 0])) for j in taxa]

    rng = np.random.default_rng(0)
    idx = np.flatnonzero(mask.ravel())
    idx = rng.choice(idx, size=min(4000, idx.size), replace=False)
    scatter = dict(truth=log_true.ravel()[idx].round(4).tolist(), observed=log_obs.ravel()[idx].round(4).tolist(),
                   denoised=log_den.ravel()[idx].round(4).tolist())
    return dict(meta={k: meta[k] for k in ("task_id", "scenario", "condition", "seed", "variant")},
                dims=list(d["Y"].shape), floor=floor, median_depth=float(np.median(d["depth"])),
                n_exposed=int(exposed.sum()), n_responder_taxa=int(responder.sum()),
                zero_fraction=float(zero.sum() / mask.sum()),
                coverage_Z=float(inside[mask].mean()), level=level, event=event, component=component,
                exemplar_taxa=taxa, exemplar_subjects=subjects,
                trajectories=trajectories, event_curves=curves, scatter=scatter)


def _row_checks(entry, task_json):
    """Cross-check against the scored row so demo and metrics.json agree."""
    if not task_json:
        return {}
    row = task_json["row"]
    naive = next((b for b in task_json.get("baselines", []) if b["method"] == "naive"), {})
    keep = ("count_log_rmse", "count_trajectory_r", "Z_rmse", "coverage_Z", "width_Z", "event_plus_rmse",
            "H_plus_shape_r", "H_plus_peak_slope", "detect_any_auc", "detect_any_power", "detect_any_fdr",
            "converged", "fit_seconds")
    all_cells = next(r for r in entry["level"] if r["window"] == "all" and r["taxa"] == "all_taxa")
    return dict(row={k: row.get(k) for k in keep},
                naive_count_log_rmse=naive.get("count_log_rmse"),
                naive_count_trajectory_r=naive.get("count_trajectory_r"),
                consistent=bool(np.isclose(all_cells["rmse_denoised"], row["count_log_rmse"], rtol=1e-3) and
                                (not naive or np.isclose(all_cells["rmse_observed"], naive["count_log_rmse"], rtol=1e-3))))


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def _fmt(v):
    return "-" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)


def _markdown(entries):
    lines = ["# Denoising demo", "",
             "Scale: log(X + 1/median depth). observed = Y/depth; denoised = posterior mean X. "
             f"post_local = 0..{LOCAL_WINDOW} of the time span after the event.", "",
             "## Overall", "",
             "| scenario | condition | method | RMSE observed | RMSE denoised | traj r observed | traj r denoised | Z coverage | event AUC | power | FDR | consistent |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in entries:
        m, c = e["meta"], e.get("check", {})
        r = c.get("row", {})
        allc = next(x for x in e["level"] if x["window"] == "all" and x["taxa"] == "all_taxa")
        lines.append("| " + " | ".join(map(_fmt, [m["scenario"], m["condition"], m["variant"], allc["rmse_observed"],
                     allc["rmse_denoised"], c.get("naive_count_trajectory_r"), r.get("count_trajectory_r"),
                     e["coverage_Z"], r.get("detect_any_auc"), r.get("detect_any_power"), r.get("detect_any_fdr"),
                     c.get("consistent")])) + " |")
    lines += ["", "## RMSE by window and taxon type", "",
              "| scenario | condition | window | taxa | cells | observed | denoised |", "|---|---|---|---|---|---|---|"]
    for e in entries:
        for x in e["level"]:
            lines.append("| " + " | ".join(map(_fmt, [e["meta"]["scenario"], e["meta"]["condition"], x["window"],
                         x["taxa"], x["n_cells"], x["rmse_observed"], x["rmse_denoised"]])) + " |")
    lines += ["", "## Local event effect (change from own pre-event mean, post_local visits)", "",
              "| scenario | condition | taxa | cells | RMSE observed | RMSE denoised | r observed | r denoised |",
              "|---|---|---|---|---|---|---|---|"]
    for e in entries:
        for x in e["event"]:
            lines.append("| " + " | ".join(map(_fmt, [e["meta"]["scenario"], e["meta"]["condition"], x["taxa"], x["n_cells"],
                         x["delta_rmse_observed"], x["delta_rmse_denoised"], x["delta_r_observed"], x["delta_r_denoised"]])) + " |")
    return "\n".join(lines) + "\n"


def _figures(entries, folder):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "matplotlib not installed; figures skipped (report.json holds all plotted data)"
    colors = dict(truth="#222222", observed="#d08c2a", denoised="#2f6db5")
    for e in entries:
        m = e["meta"]
        tag = f"sc{m['scenario']}_{m['condition']}_{m['variant']}_s{m['seed']}"
        taxa, subjects = e["exemplar_taxa"], e["exemplar_subjects"]
        if taxa and subjects:
            fig, axes = plt.subplots(len(taxa), len(subjects), figsize=(4 * len(subjects), 2.6 * len(taxa)),
                                     squeeze=False, sharex=True)
            for tr in e["trajectories"]:
                ax = axes[taxa.index(tr["taxon"]), subjects.index(tr["subject"])]
                t = np.array(tr["times"])
                ax.fill_between(t, tr["lower"], tr["upper"], color=colors["denoised"], alpha=.18, lw=0)
                ax.plot(t, tr["truth"], color=colors["truth"], lw=2, label="truth")
                obs, z = np.array(tr["observed"]), np.array(tr["observed_zero"])
                ax.plot(t, obs, "o-", color=colors["observed"], ms=4, lw=1, label="observed")
                ax.plot(t[z], obs[z], "x", color="#b3261e", ms=6, label="zero count")
                ax.plot(t, tr["denoised"], color=colors["denoised"], lw=2, label="denoised (95% CI)")
                ax.axvline(tr["event_time"], color="#777", ls="--", lw=1)
                kind = f"responder ch{tr['channel']}" if tr["responder"] else "null"
                ax.set_title(f"taxon {tr['taxon']} ({kind}) · subject {tr['subject']}", fontsize=9)
            axes[0, 0].legend(fontsize=7, loc="best")
            fig.supylabel("log(X + floor)")
            fig.suptitle(f"Scenario {m['scenario']} · {m['condition']} · {m['variant']}")
            fig.tight_layout()
            fig.savefig(folder / f"traj_{tag}.png", dpi=130)
            plt.close(fig)
        if e["event_curves"]:
            fig, axes = plt.subplots(1, len(e["event_curves"]), figsize=(3.6 * len(e["event_curves"]), 3), squeeze=False)
            for ax, c in zip(axes[0], e["event_curves"]):
                g = np.array(c["grid"])
                ax.fill_between(g, c["lower"], c["upper"], color=colors["denoised"], alpha=.2, lw=0)
                ax.plot(g, c["truth"], color=colors["truth"], lw=2, label="truth")
                ax.plot(g, c["mean"], color=colors["denoised"], lw=2, label="estimate")
                ax.axvline(0, color="#777", ls="--", lw=1)
                ax.axhline(0, color="#bbb", lw=.8)
                ax.set_title(f"taxon {c['taxon']} ({'responder' if c['responder'] else 'null'}) P={c['p_any']:.2f}", fontsize=9)
                ax.set_xlabel("time since event")
            axes[0, 0].legend(fontsize=7)
            fig.suptitle(f"Population event response H+ · sc{m['scenario']} {m['condition']}")
            fig.tight_layout()
            fig.savefig(folder / f"event_{tag}.png", dpi=130)
            plt.close(fig)
        s = {k: np.array(v) for k, v in e["scatter"].items()}
        lo, hi = float(np.min(s["truth"])), float(np.max(s["truth"]))
        fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
        for ax, name in zip(axes, ("observed", "denoised")):
            r = float(np.sqrt(np.mean((s[name] - s["truth"]) ** 2)))
            ax.scatter(s["truth"], s[name], s=3, alpha=.3, color=colors[name])
            ax.plot([lo, hi], [lo, hi], color="#444", lw=1)
            ax.set_title(f"{name}: RMSE {r:.3f} (4000-cell sample)", fontsize=9)
            ax.set_xlabel("truth log(X + floor)")
        fig.tight_layout()
        fig.savefig(folder / f"scatter_{tag}.png", dpi=130)
        plt.close(fig)
    labels = [f"sc{e['meta']['scenario']}\n{e['meta']['condition']}" for e in entries]
    get = lambda e, w, t, k: next((x[k] for x in e["level"] if x["window"] == w and x["taxa"] == t), np.nan)
    fig, axes = plt.subplots(1, 2, figsize=(max(8, 1.6 * len(entries)), 4))
    x = np.arange(len(entries))
    for ax, (w, t, title) in zip(axes, (("all", "all_taxa", "All cells"), ("post_local", "responders", "Responders, post-event window"))):
        ax.bar(x - .2, [get(e, w, t, "rmse_observed") for e in entries], .4, color=colors["observed"], label="observed")
        ax.bar(x + .2, [get(e, w, t, "rmse_denoised") for e in entries], .4, color=colors["denoised"], label="denoised")
        ax.set_xticks(x, labels, fontsize=7)
        ax.set_title(title)
        ax.set_ylabel("log RMSE vs truth")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(folder / "rmse_overview.png", dpi=130)
    plt.close(fig)
    return "ok"


def build_report(out):
    out = Path(out)
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    entries = []
    for task in plan["tasks"]:
        path = out / "demo" / f"{task['task_id']}.npz"
        if not path.exists():
            continue
        with np.load(path) as data:
            entry = analyse({k: data[k] for k in data.files})
        task_path = out / "tasks" / f"{task['task_id']}.json"
        task_json = json.loads(task_path.read_text(encoding="utf-8")) if task_path.exists() else None
        entry["check"] = _row_checks(entry, task_json if task_json and task_json.get("status") == "complete" else None)
        entries.append(entry)
    entries.sort(key=lambda e: (e["meta"]["scenario"], e["meta"]["condition"], e["meta"]["variant"], e["meta"]["seed"]))
    folder = out / "demo_report"
    folder.mkdir(exist_ok=True)
    figures = _figures(entries, folder) if entries else "no demo arrays"
    report = dict(n_demo=len(entries), n_tasks=len(plan["tasks"]), figures=figures,
                  local_window=LOCAL_WINDOW, entries=entries)
    (folder / "report.json").write_text(json.dumps(_clean(report), ensure_ascii=False, allow_nan=False), encoding="utf-8")
    (folder / "summary.md").write_text(_markdown(entries), encoding="utf-8")
    print(f"demo report: {len(entries)}/{len(plan['tasks'])} tasks; figures: {figures}; {folder}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    build_report(p.parse_args().out)


if __name__ == "__main__":
    main()
