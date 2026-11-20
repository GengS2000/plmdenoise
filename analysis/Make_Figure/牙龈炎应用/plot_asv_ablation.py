from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from PIL import Image


OUT = Path(__file__).resolve().parent
BASE = OUT.parents[1] / "实际数据分析的结果" / "预测分型"
RESULTS = BASE / "results_bootstrap_top10_ablation"
WINDOWS = ("W0", "W1", "W2", "W3")
CUTOFFS = (0, 2, 5, 9)
MODES = ("all", "keep10", "drop10")
BAR_LABELS = ("All ASVs", "Top 10", "All − Top 10")
BAR_COLORS = ("#4F587D", "#C68DC0", "#C2E0EE")
BUBBLE_SCALE = 175

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 9, "axes.linewidth": .65, "xtick.major.width": .65,
    "ytick.major.width": .65, "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def taxonomy_label(asv, taxa):
    row = taxa.loc[asv]
    if not row.Genus:
        taxon = row.Family or "Unclassified"
    elif not row.Species:
        taxon = f"{row.Genus} sp."
    elif row.Species.startswith(row.Genus + " "):
        taxon = row.Species
    else:
        taxon = f"{row.Genus} {row.Species}"
    return f"{asv}  {taxon}"


def plot_bubbles(ax, panel, taxa, cmap, norm):
    y = np.arange(len(panel))
    freq = panel.selection_frequency
    ax.scatter(freq, y, s=BUBBLE_SCALE * freq, c=panel.mean_bootstrap_norm,
               cmap=cmap, norm=norm, edgecolors="#46425D", linewidths=.45, zorder=3)
    ax.set(xlim=(0, 1.06), ylim=(9.65, -.65), xticks=[0, .25, .5, .75, 1],
           yticks=y, yticklabels=[taxonomy_label(asv, taxa) for asv in panel.asv])
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.tick_params(axis="y", length=0, pad=7, labelsize=7.5)
    ax.tick_params(axis="x", length=2.5)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#E7E5EB", linewidth=.55)
    ax.spines[["top", "left", "right"]].set_visible(False)


def plot_ablation(ax, values):
    bars = ax.bar(BAR_LABELS, values, width=.57, color=BAR_COLORS,
                  edgecolor="white", linewidth=.4, zorder=3)
    ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
    ax.set(ylim=(0, 1.06), ylabel="Macro OVR AUC", yticks=[0, .25, .5, .75, 1])
    ax.axhline(.5, color="#9A97A8", linestyle=(0, (3, 2)), linewidth=.65, zorder=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=5)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color="#E7E5EB", linewidth=.55)


def main():
    panels = pd.read_csv(RESULTS / "top10_panels.tsv", sep="\t")
    ablation = pd.read_csv(RESULTS / "summary.tsv", sep="\t")
    taxa = pd.read_csv(BASE / "organized/lower_jaw_interproximal/taxa.csv",
                       keep_default_na=False).set_index("taxon_id")
    auc = ablation.pivot(index="window", columns="ablation", values="macro_ovr_auc")
    cmap = LinearSegmentedColormap.from_list("mean_norm", ["#EEE5F2", "#B46EAA", "#463457"])
    color_max = np.ceil(panels.mean_bootstrap_norm.max() * 100) / 100
    norm = Normalize(0, color_max)

    fig = plt.figure(figsize=(7.2, 9.2))
    grid = fig.add_gridspec(4, 2, left=.29, right=.97, top=.875, bottom=.06,
                            width_ratios=[1, .95], wspace=.44, hspace=.48)
    fig.text(.29, .974, "Bootstrap ASV selection", fontsize=9, weight="bold")
    fig.text(.69, .974, "Fixed-panel ablation", fontsize=9, weight="bold")
    cax = fig.add_axes([.29, .937, .31, .011])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                      orientation="horizontal")
    cb.set_ticks([0, color_max / 2, color_max],
                 labels=[f"{v:.3f}" for v in (0, color_max / 2, color_max)])
    cb.ax.tick_params(length=2, pad=2, labelsize=7)
    cb.outline.set_visible(False)
    cb.set_label("Mean bootstrap $L_2$ score", fontsize=7, labelpad=1)

    size_key = fig.add_axes([.69, .915, .27, .037])
    size_key.set(xlim=(0, 3), ylim=(0, 1))
    size_key.set_axis_off()
    size_key.set_title("Top 10 selection frequency", fontsize=7, pad=1, loc="left")
    for x, freq in enumerate((.25, .5, 1)):
        size_key.scatter(x + .16, .48, s=110 * freq, color="#DCC7E1",
                         edgecolors="#46425D", linewidths=.45, clip_on=False)
        size_key.text(x + .41, .48, f"{freq:.0%}", va="center", fontsize=7)

    for i, (window, day) in enumerate(zip(WINDOWS, CUTOFFS)):
        bubble = fig.add_subplot(grid[i, 0])
        bars = fig.add_subplot(grid[i, 1])
        panel = panels[panels.window == window].sort_values("panel_rank")
        if len(panel) != 10:
            raise ValueError(f"{window}: expected 10 ASVs")
        plot_bubbles(bubble, panel, taxa, cmap, norm)
        bubble.collections[0].set_sizes(110 * panel.selection_frequency)
        bubble.tick_params(axis="y", labelsize=6.5, pad=5)
        bubble.tick_params(axis="x", labelsize=7)
        plot_ablation(bars, auc.loc[window, list(MODES)].to_numpy())
        bars.tick_params(axis="both", labelsize=7)
        pos = bubble.get_position()
        fig.text(.025, pos.y1 + .009, "ABCD"[i], fontsize=11, weight="bold")
        fig.text(.065, pos.y1 + .009, window, fontsize=8, weight="bold")
        if i == 3:
            bubble.set_xlabel("Bootstrap Top 10 selection frequency", fontsize=7)

    stem = OUT / "Supplementary_ASV_selection_ablation"
    for extension in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix("." + extension), dpi=600, facecolor="white")
    fig.savefig(OUT / "preview.png", dpi=125, facecolor="white")
    with Image.open(stem.with_suffix(".png")) as image:
        image.save(stem.with_suffix(".tiff"), compression="tiff_lzw", dpi=(600, 600))
        print("Raster size", image.size)
    panels.assign(taxonomy=panels.asv.map(lambda asv: taxonomy_label(asv, taxa))).to_csv(
        OUT / "panel_A_source_data.csv", index=False, encoding="utf-8-sig")
    ablation.to_csv(OUT / "panel_B_source_data.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()

