"""Server-side publication-ready static PCA and uncertainty figures (headless)."""
from pathlib import Path


def plot(out, rows, inference, plan):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from dr_metrics import METHODS, align_projection
    from run_dr import load_npz
    dest = Path(out) / "summary" / "figures"
    dest.mkdir(exist_ok=True)
    colors = ["#A8B6C6", "#4F587D"]
    group_colors = ["#247A96", "#CF753D"]
    labels = ["Naive + PCA", "M4_flex abundance + PCA"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    def save(fig, name):
        if plan["smoke"]:
            fig.suptitle("ENGINEERING SMOKE TEST — NOT SCIENTIFIC RESULTS", color="darkred", fontsize=11)
        fig.savefig(dest / (name + ".png"), dpi=300, bbox_inches="tight")
        fig.savefig(dest / (name + ".pdf"), bbox_inches="tight")
        plt.close(fig)

    valid = [r for r in inference if r["scope"] == "all_complete" and r["valid"]]
    if valid:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        for ax, condition, title in zip(axes, ("null", "effect"), ("Type I error", "Power")):
            for j, method in enumerate(METHODS):
                r = next((r for r in valid if r["condition"] == condition and r["method"] == method), None)
                if r is None:
                    continue
                ax.bar(j, r["rate"], color=colors[j], width=.6)
                ax.errorbar(j, r["rate"], yerr=[[max(0,r["rate"]-r["wilson95_low"])], [max(0,r["wilson95_high"]-r["rate"])]],
                            fmt="none", color="black", capsize=4)
                ax.text(j, min(1.05, r["wilson95_high"]+.03), f"n={r['valid']}", ha="center")
            ax.set(xticks=[0,1], xticklabels=labels, ylabel="Rejection proportion (Wilson 95% CI)",
                   ylim=(0,1.12), title=title)
            if condition == "null":
                ax.axhline(plan["alpha"], color="gray", ls="--", label="Nominal alpha")
                ax.legend()
        save(fig, "inference")
    specs = [("group_r2", "Group R² (fraction of total SS)"), ("ari", "ARI vs true groups"),
             ("truth_distance_spearman", "Distance Spearman vs latent truth"),
             ("dispersion_ratio_1_over_0", "Dispersion ratio (group 1 / group 0)")]
    if any(r["phase"] == "mc" and r["condition"] == "effect" for r in rows):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
        for ax, (key, label) in zip(axes.flat, specs):
            for j, method in enumerate(METHODS):
                v = [r[key] for r in rows if r["phase"] == "mc" and r["condition"] == "effect"
                     and r["method"] == method and r.get(key) is not None]
                if v:
                    ax.boxplot([v], positions=[j], widths=.45, patch_artist=True,
                               boxprops={"facecolor": colors[j]}, medianprops={"color": "black"})
            ax.set(xticks=[0,1], xticklabels=labels, ylabel=label, title=f"Effect = {plan['effect']:g}")
        save(fig, "independent_metrics")

    for condition in ("null", "effect"):
        parent = f"{condition}_mc_0000"
        if not any(r["task_id"] == parent for r in rows):
            continue
        refs = [load_npz(Path(out) / "tasks" / parent / f"{method}.npz") for method in METHODS]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
        truth = refs[0]["truth_scores2"]
        panels = [("Latent truth PCA", truth, refs[0]["group"])]
        for ref, label in zip(refs, labels):
            aligned, _ = align_projection(ref["scores"][:, :2], truth)
            panels.append((label + " (aligned)", aligned, ref["group"]))
        all_points = np.concatenate([item[1] for item in panels])
        lower, upper = all_points.min(0), all_points.max(0)
        margin = np.maximum((upper-lower)*.1, .1)
        for ax, (label, xy, group) in zip(axes, panels):
            for g in (0,1):
                ax.scatter(*xy[group==g].T, color=group_colors[g], label=f"Group {g}", s=28, alpha=.85)
            ax.set(title=label, xlabel="Aligned PC1 (truth score units)", ylabel="Aligned PC2 (truth score units)",
                   xlim=(lower[0]-margin[0],upper[0]+margin[0]), ylim=(lower[1]-margin[1],upper[1]+margin[1]))
            ax.legend()
        fig.supxlabel(f"{condition}; preselected first simulation, seed {plan['tasks'][0]['seed']}. Similarity alignment uses truth only for display.")
        save(fig, f"projection_{condition}")

        boots = [r for r in rows if r["phase"] == "bootstrap" and r["condition"] == condition]
        if not boots:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
        for ax, method, ref, label in zip(axes, METHODS, refs, labels):
            for row in boots:
                if row["method"] != method:
                    continue
                b = load_npz(Path(out) / "tasks" / row["task_id"] / f"{method}.npz")
                ids, xy = b["stability_subject_ids"], b["stability_aligned_scores2"]
                for g in (0,1):
                    ax.scatter(*xy[ref["group"][ids]==g].T, color=group_colors[g], s=5, alpha=.025, rasterized=True)
            for g in (0,1):
                ax.scatter(*ref["scores"][ref["group"]==g, :2].T, color=group_colors[g], s=25,
                           edgecolors="black", linewidths=.4, label=f"Group {g}")
            ax.set(title=label, xlabel="PC1 (reference score units)", ylabel="PC2 (reference score units)")
            ax.legend()
        fig.supxlabel("Faint points: full-refit subject bootstrap, similarity aligned to the original fit. Axes are method-specific.")
        save(fig, f"bootstrap_projection_{condition}")
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
        for ax, (key, ylabel) in zip(axes, [("stability_cluster_ari", "Cluster ARI vs original fit"),
                                            ("stability_subspace_overlap", "PCA subspace overlap"),
                                            ("stability_projection2_error", "Relative aligned projection error")]):
            for j, method in enumerate(METHODS):
                values = np.asarray([r[key] for r in boots if r["method"] == method])
                if len(values):
                    lo, median, hi = np.quantile(values, [.025,.5,.975])
                    ax.errorbar(j, median, yerr=[[median-lo],[hi-median]], fmt="o", color=colors[j],
                                capsize=5, markersize=7)
            ax.set(xticks=[0,1], xticklabels=["Naive", "M4_flex"], ylabel=ylabel)
        fig.supxlabel(f"{condition}: medians and conditional bootstrap 2.5–97.5 percentile stability ranges")
        save(fig, f"bootstrap_stability_{condition}")
