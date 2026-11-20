"""Case-study figure: sensitivity on a signal-rich dataset (FARMM) versus the
dataset whose original study reported no treatment effect on soil bacteria (SOIL).

Panels
  A  FARMM - stage at which Enterobacter cloacae becomes detectable (4 methods x raw/denoised)
  B  SOIL  - nominal positive rate over all taxon-contrast tests, raw versus denoised
             (full-time, time-adjusted models: the estimand of Wang et al. 2022)
  C  SOIL  - Terrimicrobium dose response along the precipitation-exclusion gradient,
             by cultivar; the genus Wang et al. 2022 report as treatment-significant

All numbers are read from existing result tables; nothing is refitted.
Build:  python build_case.py
"""
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = ROOT / "实际数据分析的结果"
FARMM_DA = DATA / "Results/Results/FARMM/02_differential_abundance"
SOIL_DA = DATA / "Results/Results/SOIL/02_differential_abundance"          # full-time
SOIL_META = DATA / "general/generalization_analysis_20261021/SOIL"
TAXA = SOIL_META / "taxa.tsv"
SAMPLES = SOIL_META / "samples.tsv"

NAVY, PINK, PALE, PURPLE, LILAC = "#4F587D", "#C68DC0", "#C2E0EE", "#776B97", "#DBC8ED"
GREY, INK, MIST = "#8A8F9E", "#2B2F3A", "#E9EEF3"
CMAP = LinearSegmentedColormap.from_list("case", ["#FFFFFF", PALE, LILAC, PINK, PURPLE, NAVY])
METHODS = [("glmmpql", "GLMM-PQL"), ("linda", "LinDA"),
           ("maaslin2", "MaAsLin2"), ("zigmm", "ZIGMM")]
STAGES = ["depletion", "early_recovery", "late_recovery"]
STAGE_LAB = ["Depletion\nd6-8", "Early recovery\nd9-12", "Late recovery\nd13-15"]
ALPHA = 0.05
LEVELS = [25, 50, 75]
CULTS = [("Drought_tolerant", "Drought-tolerant cultivar"),
         ("Drought_sensitive", "Drought-sensitive cultivar")]
FOCUS = "OTU_13"          # the Terrimicrobium OTU carrying the FDR hit
C_METHOD = "maaslin2"     # method whose contrast-wise FDR hit the panel is built on
ITAL_T = r"$\it{Terrimicrobium}$"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7.3, "axes.titlesize": 7.8,
    "xtick.labelsize": 6.6, "ytick.labelsize": 6.6, "legend.fontsize": 6.3,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.3, "ytick.major.size": 2.3,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK,
    "figure.dpi": 150,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})


def read_da(folder, method):
    return pd.read_csv(folder / f"{method}_raw_vs_denoised.tsv", sep="\t", low_memory=False)


def nlog10(x, floor=1e-320):
    return -np.log10(np.clip(np.asarray(x, float), floor, None))


def panel_letter(ax, letter, dx, dy=1.06):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom", ha="left")


# =========================================================================== data
a_rows = []
for key, name in METHODS:
    e = read_da(FARMM_DA, key)
    e = e[e.taxon.str.contains("Enterobacter_cloacae", na=False)]
    for stage in STAGES:
        r = e[e.contrast.eq(f"FARMM__{stage}_vs_baseline__EEN_vs_Other")].iloc[0]
        for rep in ("raw", "denoised"):
            a_rows.append(dict(method=name, stage=stage, representation=rep,
                               q_bh_contrast=r[f"q_bh_contrast_{rep}"],
                               q_bh_family=r[f"q_bh_family_{rep}"],
                               pvalue=r[f"pvalue_{rep}"], estimate=r[f"estimate_{rep}"]))
A = pd.DataFrame(a_rows)
A.to_csv(OUT / "A_Ecloacae_source.csv", index=False)

soil = []
for key, name in METHODS:
    d = read_da(SOIL_DA, key).assign(method=name, method_key=key)
    d["short"] = d.contrast.str.replace("SOIL__time_adjusted_", "", regex=False)
    soil.append(d[["method", "method_key", "taxon", "contrast", "short",
                   "pvalue_raw", "pvalue_denoised",
                   "q_bh_contrast_raw", "q_bh_contrast_denoised",
                   "q_bh_family_raw", "q_bh_family_denoised",
                   "estimate_raw", "estimate_denoised", "zero_fraction_raw"]])
S = pd.concat(soil, ignore_index=True)
S.to_csv(OUT / "soil_all_tests_source.csv", index=False)

rate = []
for _, name in METHODS:
    sub = S[S.method.eq(name)]
    for rep in ("raw", "denoised"):
        p = pd.to_numeric(sub[f"pvalue_{rep}"], errors="coerce").dropna()
        q = pd.to_numeric(sub[f"q_bh_contrast_{rep}"], errors="coerce")
        qf = pd.to_numeric(sub[f"q_bh_family_{rep}"], errors="coerce")
        rate.append(dict(method=name, representation=rep, n_tested=len(p),
                         rate_p05=(p < ALPHA).mean(),
                         n_fdr_contrast=int((q < ALPHA).sum()),
                         n_fdr_family=int((qf < ALPHA).sum())))
R = pd.DataFrame(rate)
R.to_csv(OUT / "B_fulltime_source.csv", index=False)
POOLED = {rep: pd.to_numeric(S[f"pvalue_{rep}"], errors="coerce").dropna().lt(ALPHA).mean() * 100
          for rep in ("raw", "denoised")}
N_TOTAL = int(R[R.representation.eq("raw")].n_tested.sum())
N_FDR = {rep: int(R[R.representation.eq(rep)].n_fdr_contrast.sum()) for rep in ("raw", "denoised")}
N_FAM = {rep: int(R[R.representation.eq(rep)].n_fdr_family.sum()) for rep in ("raw", "denoised")}

# ---------------------------------------------- C: Terrimicrobium dose response
genus = pd.read_csv(TAXA, sep="\t").set_index("taxon_id")["name"]
S["genus"] = S.taxon.map(genus)
TERRI = sorted(S.loc[S.genus.eq("Terrimicrobium"), "taxon"].unique(),
               key=lambda t: (t != FOCUS, t))

dose_rows = []
for key, name in METHODS:
    sub = S[S.method_key.eq(key) & S.taxon.isin(TERRI)]
    for cult, _ in CULTS:
        for rep in ("raw", "denoised"):
            for t in TERRI:
                row = {"method": name, "cultivar": cult, "representation": rep, "taxon": t}
                for lv in LEVELS:
                    r = sub[sub.taxon.eq(t) & sub.short.eq(f"{cult}__{lv}_vs_0")]
                    row[f"est_{lv}"] = float(r[f"estimate_{rep}"].iloc[0]) if len(r) else np.nan
                    row[f"q_{lv}"] = float(r[f"q_bh_contrast_{rep}"].iloc[0]) if len(r) else np.nan
                    row[f"p_{lv}"] = float(r[f"pvalue_{rep}"].iloc[0]) if len(r) else np.nan
                e = np.array([row[f"est_{lv}"] for lv in LEVELS])
                row["all_negative"] = bool((e < 0).all())
                row["monotone_down"] = bool(e[0] >= e[1] >= e[2] and e[2] < 0)
                dose_rows.append(row)
D = pd.DataFrame(dose_rows)

MONO = (D.groupby(["method", "cultivar", "representation"]).monotone_down.sum()
        .astype(int).unstack("representation"))
N_OTU = len(TERRI)
C_NAME = dict(METHODS)[C_METHOD]
# Compare each exclusion level with its control within the same block and date.
sm = pd.read_csv(SAMPLES, sep="\t")
sm["cultivar"] = sm["cultivar"].str.replace(" ", "_", regex=False)
paired = sm.pivot_table(index=["cultivar", "block", "date"],
                        columns="treatment", values="moisture")
moisture_rows = []
for cult, _ in CULTS:
    by_level = paired.loc[cult]
    for level in [0] + LEVELS:
        pairs = by_level[[0, level]].dropna()
        moisture_rows.append({
            "cultivar": cult,
            "exclusion_pct": level,
            "n_matched_block_dates": len(pairs),
            "control_moisture_pct": 100 * pairs.iloc[:, 0].mean(),
            "treatment_moisture_pct": 100 * pairs.iloc[:, 1].mean(),
            "moisture_deficit_pp": 100 * (pairs.iloc[:, 0] - pairs.iloc[:, 1]).mean(),
        })
moisture = pd.DataFrame(moisture_rows)
moisture.to_csv(OUT / "C_soil_moisture_source.csv", index=False)
moisture_by_cultivar = moisture.pivot(index="cultivar", columns="exclusion_pct",
                                     values="moisture_deficit_pp")
for level in [0] + LEVELS:
    D[f"moisture_deficit_pp_{level}"] = D.cultivar.map(moisture_by_cultivar[level])
D.to_csv(OUT / "C_terrimicrobium_dose_source.csv", index=False)

# ========================================================================= figure
fig = plt.figure(figsize=(7.2, 5.85))
gs = fig.add_gridspec(2, 2, width_ratios=[1.44, 1.0], height_ratios=[1.0, 0.86],
                      left=0.170, right=0.975, top=0.925, bottom=0.10,
                      wspace=0.40, hspace=0.95)

# ---------------------------------------------------------------- A: stage grid
axA = fig.add_subplot(gs[0, 0])
order = [(name, rep) for _, name in METHODS for rep in ("raw", "denoised")]
grid = np.array([[nlog10(A[(A.method == m) & (A.representation == r) &
                           (A.stage == s)].q_bh_contrast.iloc[0]) for s in STAGES]
                 for m, r in order])
VMAX, sig_line = 16, nlog10(ALPHA)
im = axA.imshow(np.clip(grid, 0, VMAX), cmap=CMAP, vmin=0, vmax=VMAX,
                aspect="auto", origin="upper")
for i in range(8):
    first = next((j for j in range(3) if grid[i, j] >= sig_line), None)
    for j in range(3):
        sig = grid[i, j] >= sig_line
        axA.add_patch(Rectangle((j - .5, i - .5), 1, 1,
                                facecolor="none" if sig else "#F1F2F5",
                                edgecolor=INK if sig else "#A3A8B6",
                                linewidth=1.0 if sig else 0.55,
                                zorder=3 if sig else 2))
    if first is not None:
        axA.plot(first, i - 0.37, marker="v", ms=3.2, color=INK, zorder=5, clip_on=False)
axA.set_xticks(range(3))
axA.set_xticklabels(STAGE_LAB, fontsize=6.1)
axA.set_yticks(range(8))
axA.set_yticklabels(["Raw" if r == "raw" else "Denoised" for _, r in order], fontsize=6.2)
axA.set_xlim(-0.5, 2.5)
for i in range(0, 8, 2):
    axA.text(-1.30, i + 0.5, order[i][0], rotation=90, ha="center", va="center",
             fontsize=6.4, fontweight="bold", clip_on=False)
    if i:
        axA.axhline(i - 0.5, color="white", lw=2.2)
axA.tick_params(length=0)
for s in axA.spines.values():
    s.set_visible(False)
axA.set_title(r"Diet effects on $\it{E.\ cloacae}$ across stages", pad=5)
cax = axA.inset_axes([0.0, -0.265, 1.0, 0.048])
cb = fig.colorbar(im, cax=cax, orientation="horizontal")
cb.set_label("$-\\log_{10}$(BH $q$), capped at 16   ▼ first significant stage",
             fontsize=6.0, labelpad=1.5)
cb.ax.tick_params(labelsize=5.9, length=2)
cb.outline.set_linewidth(0.5)
panel_letter(axA, "A", dx=-0.285, dy=1.09)

# ------------------------------------------------------ B: nominal positive rate
axB = fig.add_subplot(gs[0, 1])
x, w = np.arange(4), 0.36
for k, (rep, color) in enumerate([("raw", NAVY), ("denoised", PINK)]):
    v = [R[(R.method == n) & (R.representation == rep)].rate_p05.iloc[0] * 100
         for _, n in METHODS]
    axB.bar(x + (k - 0.5) * w, v, w, color=color, edgecolor="white", linewidth=0.4,
            label="Raw" if rep == "raw" else "Denoised", zorder=2)
    for xi, vi in zip(x + (k - 0.5) * w, v):
        axB.text(xi, vi + (0.25 if 4.7 <= vi < 5 else 0.14), f"{vi:.2f}", ha="center", fontsize=5.6, color=INK)
axB.axhline(5, color=PURPLE, ls="--", lw=0.9, zorder=1)

axB.set_xticks(x)
axB.set_xticklabels([n for _, n in METHODS], rotation=30, ha="right", fontsize=6.3)
axB.set_ylabel("Taxon contrasts (%)")
axB.set_ylim(0, 10.6)
axB.set_yticks([0, 2, 4, 5, 6, 8, 10])
axB.legend(frameon=False, loc="upper left", handlelength=1.0, borderpad=0.1,
           bbox_to_anchor=(-0.02, 1.03))
axB.set_title("Taxon contrasts with $P$ < 0.05 before and\nafter denoising across the full season",
              fontsize=7.4, pad=5)
panel_letter(axB, "B", dx=-0.265, dy=1.09)

# ---------------------------------- C: measured moisture deficit by cultivar
gsC = gs[1, :].subgridspec(1, 2, wspace=0.16)
pool = D[D.method.eq(C_NAME) & D.representation.eq("denoised")]
axC = []
for ci, (cult, cult_lab) in enumerate(CULTS):
    ax = fig.add_subplot(gsC[0, ci], sharex=axC[0] if ci else None,
                         sharey=axC[0] if ci else None)
    axC.append(ax)
    sub = pool[pool.cultivar.eq(cult)].set_index("taxon")
    xs = moisture_by_cultivar.loc[cult, [0] + LEVELS].to_numpy()
    ax.axhline(0, color=GREY, lw=0.7, ls="--", zorder=1)
    for taxon, row in sub.iterrows():
        ys = [0.0] + [row[f"est_{level}"] for level in LEVELS]
        if taxon == FOCUS:
            style = dict(color=NAVY, lw=2.0, ms=4.1, mfc="white", mew=1.15, zorder=5)
        elif row.monotone_down:
            style = dict(color=PURPLE, lw=1.0, ms=2.7, alpha=0.85, zorder=3)
        else:
            style = dict(color="#9EA7B9", lw=0.8, ms=2.6, alpha=0.8, zorder=2)
        ax.plot(xs, ys, "-o", **style)
    ax.set(xlim=(-0.13, 3.05), ylim=(-1.25, 1.39),
           xticks=[0, 0.5, 1, 1.5, 2, 2.5, 3], yticks=[-1, -0.5, 0, 0.5, 1])
    ax.set_title(cult_lab, pad=5, fontweight="bold")
    if ci == 0:
        ax.set_ylabel("Estimated effect size (MaAsLin2)\nrelative to 0% exclusion")
    else:
        ax.tick_params(labelleft=False)
    ax.spines["bottom"].set_color(GREY)
    ax.spines["left"].set_color(GREY)

handles = [Line2D([], [], color=NAVY, lw=2, marker="o", ms=4, mfc="white"),
           Line2D([], [], color=PURPLE, lw=1),
           Line2D([], [], color="#9EA7B9", lw=0.8)]
left, right = axC[0].get_position().x0, axC[1].get_position().x1
center = (left + right) / 2
bottom = axC[0].get_position().y0
top = axC[0].get_position().y1
fig.text(left, top + .067, ITAL_T + " effect sizes along the soil moisture gradient",
         fontsize=7.4, fontweight="bold", va="bottom")
fig.text(left - .082, top + .067, "C", fontsize=10, fontweight="bold", va="bottom")
fig.text(center, bottom - .050, "Measured soil moisture deficit vs 0% exclusion (percentage points)",
         ha="center", fontsize=7.3)
fig.legend(handles, ["OTU 13", "Declining across 25–75% PE", "Other OTUs"],
           loc="center", bbox_to_anchor=(center, top + .048), ncol=3, frameon=False)

for ext in ("pdf", "svg"):
    fig.savefig(OUT / f"Case_studies.{ext}", dpi=600, facecolor="white")
fig.savefig(OUT / "Case_studies.png", dpi=600, facecolor="white")
fig.savefig(OUT / "Case_studies.tiff", dpi=600, facecolor="white",
            pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(OUT / "preview.png", dpi=170, facecolor="white")
print("written:", sorted(p.name for p in OUT.glob("Case_studies.*")))
print(f"pooled nominal rate: {POOLED['raw']:.3f}% -> {POOLED['denoised']:.3f}%; "
      f"FDR hits raw {N_FDR['raw']} / denoised {N_FDR['denoised']} of {N_TOTAL:,}; "
      f"family-wide {N_FAM['raw']} / {N_FAM['denoised']}")
print("monotone-decline counts (of %d OTUs):" % N_OTU)
print(MONO.to_string())
print(moisture.to_string(index=False))
