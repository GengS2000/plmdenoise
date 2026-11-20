"""Render the A-E scientific figure and panels from existing results only.

Run: python Make_Figure/build_figure.py
No simulation, fitting, statistical selection, or image-generation models.
Panel E faithfully redraws published vector scatter coordinates when raw scores
are unavailable. Extracted display coordinates are NEVER used for inference.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "revised"
os.environ.setdefault("MPLCONFIGDIR", str(OUT / "_qa" / "mpl_cache"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, MultipleLocator
from PIL import Image
import fitz

DA = ROOT / "run_da/multieffect_comparision"
BC = DA / "BC_effect03_minimal"
DR = ROOT / "dimension_reduction_simulation/summary"
METHODS = ["naive", "m4_flex_abundance", "embed", "microtensor", "parafac2", "tempted_gemelli"]
NAMES = dict(zip(METHODS, ["Naive", "M4_flex", "EMBED", "microTensor", "PARAFAC2", "TEMPTED"]))
NAMES["oracle_abundance"] = "Oracle"
COLORS = dict(zip(METHODS, ["#A8B6C6", "#59638B", "#C99CC7", "#B3DCE9", "#9686B4", "#D8C7EB"]))
COLORS["oracle_abundance"] = "#AFB9AF"
CLASS_COLORS = {-1: "#A6BEDA", 0: "#D6DBE1", 1: "#C6A6CB"}
GROUP_COLORS = ["#A2C4D0", "#C9AFD1"]
ENGINES = ["linda", "maaslin2", "glmmpql", "zigmm"]
ENGINE_NAMES = ["LinDA", "MaAsLin2", "glmmPQL", "ZIGMM"]
DOSES = [0., .2, .3, .4, .5, .75]


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dark(color, factor=.76):
    return tuple(np.array(matplotlib.colors.to_rgb(color)) * factor)


def style():
    plt.rcParams.update({"font.family": "Arial", "font.size": 8,
        "axes.labelsize": 8.5, "axes.titlesize": 9, "axes.titleweight": "medium",
        "axes.linewidth": .65, "axes.edgecolor": "#60646D", "text.color": "#30343B",
        "axes.labelcolor": "#30343B", "xtick.color": "#50545D", "ytick.color": "#50545D",
        "xtick.labelsize": 7, "ytick.labelsize": 7, "xtick.major.width": .6,
        "ytick.major.width": .6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
        "figure.facecolor": "white", "axes.unicode_minus": False})


def clean_axis(ax, grid=True):
    if grid:
        ax.grid(axis="y", color="#ECEEF2", linewidth=.5, zorder=0)
    ax.set_axisbelow(True)


def panel_letter(fig, spec, letter, dx=-.033, dy=.015):
    box = spec.get_position(fig)
    fig.text(box.x0 + dx, box.y1 + dy, letter, fontsize=15, fontweight="bold",
             ha="left", va="bottom", color="#20232A")


def extract_display_points(pdf):
    """Read circle centers and affine axis calibration from a known vector PDF.

    The tick labels/ranges are verified against the supplied original artwork.
    This recovers only the already plotted, aligned two-dimensional coordinates,
    to PDF precision, with no reconstruction of PCA or unseen samples.
    """
    document = fitz.open(pdf)
    page = document[0]
    drawings = page.get_drawings()
    backgrounds = [(i, d["rect"]) for i, d in enumerate(drawings)
                   if d["fill"] == (1.,1.,1.) and len(d["items"]) == 1
                   and d["items"][0][0] == "re" and 200 < d["rect"].width < 400]
    assert len(backgrounds) == 3
    words = page.get_text("words")
    original = [np.array([36,122,150])/255, np.array([207,117,61])/255]
    all_rows, output = [], []
    for panel, (start, bounds) in enumerate(backgrounds):
        end = backgrounds[panel+1][0] if panel < 2 else len(drawings)
        block = drawings[start+1:end]
        xticks, yticks = [], []
        markers = []
        for item in block:
            if len(item["items"]) == 1 and item["items"][0][0] == "l":
                _, p, q = item["items"][0]
                if abs(p.x-q.x) < 1e-4 and abs(p.y-bounds.y1) < 1e-3 and 3 < abs(p.y-q.y) < 4:
                    xticks.append(p.x)
                if abs(p.y-q.y) < 1e-4 and abs(p.x-bounds.x0) < 1e-3 and 3 < abs(p.x-q.x) < 4:
                    yticks.append(p.y)
            if item["fill"] is not None and len(item["items"]) == 8:
                color = np.asarray(item["fill"])
                matches = [np.max(np.abs(color-c)) < 1e-5 for c in original]
                if any(matches):
                    markers.append((int(matches[1]), (item["rect"].x0+item["rect"].x1)/2,
                                    (item["rect"].y0+item["rect"].y1)/2))
        # First 40 colored paths are data; last two paths are legend symbols.
        assert len(markers) == 42 and [r[0] for r in markers[:40]].count(0) == 20
        x = np.array(sorted(xticks)); y = np.array(sorted(yticks, reverse=True))
        assert len(x) == 7 and len(y) == 8
        labels = [w[4] for w in sorted(words, key=lambda w:w[0]) if w[4].isdigit()
                  and bounds.x0 < (w[0]+w[2])/2 < bounds.x1
                  and bounds.y1 <= w[1] < bounds.y1+5]
        assert labels == ["20", "15", "10", "5", "0", "5", "10"]
        xfit = np.polyfit(x, np.arange(-20,11,5), 1)
        yfit = np.polyfit(y, np.arange(-8,7,2), 1)
        assert np.max(np.abs(np.polyval(xfit,x)-np.arange(-20,11,5))) < 1e-4
        assert np.max(np.abs(np.polyval(yfit,y)-np.arange(-8,7,2))) < 1e-4
        data = np.array([[g,np.polyval(xfit,a),np.polyval(yfit,b)] for g,a,b in markers[:40]])
        output.append(data)
        for j, (g,a,b) in enumerate(data):
            all_rows.append(dict(panel=panel, displayed_point_index=j, group=int(g), aligned_x=a, aligned_y=b,
                                 provenance="original vector PDF display coordinates; not raw PC scores"))
    document.close()
    write_csv(OUT / "source_data/E_display_coordinates.csv", all_rows)
    return output


def prepare_data():
    source_paths = [DA/"da_summary.csv", BC/"plan.json", DR/"metrics.csv", DR/"paired.csv",
                    DR/"ranges.csv", DR/"status.json", DR/"figures/projection_effect.pdf"]
    data = dict(a=read(source_paths[0]), dr=read(DR/"metrics.csv"), paired=read(DR/"paired.csv"),
                bc={}, bc_all=[], validation=[])
    plan = json.loads((BC/"plan.json").read_text(encoding="utf-8"))
    for dataset in sorted((BC/"datasets").glob("seed_*_dose_s0.3")):
        seed = int(dataset.name.split("_")[1])
        truth_path = dataset/"truth.csv"
        truth = {r["taxon"]:number(r["gamma"]) for r in read(truth_path)}
        source_paths.append(truth_path)
        for method in METHODS + ["oracle_abundance"]:
            path = dataset/"da"/method/"linda"/"taxa.csv"
            status_path = path.with_name("status.json")
            status = json.loads(status_path.read_text(encoding="utf-8"))
            expected = status["artifacts"]["taxa.csv"]
            assert sha(path) == expected, f"Checksum mismatch: {path}"
            assert status["plan_id"] == plan["plan_id"]
            rows = read(path)
            assert len(rows) == 500 and len({r['taxon'] for r in rows}) == 500
            assert all(abs(number(r["gamma"])-truth[r["taxon"]])<1e-10 for r in rows)
            assert all(r["response_scale"] == "proportion_to_internal_CLR_log2" for r in rows)
            source_paths.extend([path,status_path])
            valid = 0
            for row in rows:
                record = dict(seed=seed, method=method, taxon=row["taxon"],
                              true_log2=number(row["gamma"])/np.log(2),
                              truth_class=int(np.sign(number(row["gamma"]))),
                              estimate=number(row["estimate"]), qvalue=number(row["qvalue"]),
                              pvalue=number(row["pvalue"]), fit_status=row["fit_status"])
                record["valid"] = row["fit_status"] in ("ok","warning") and all(
                    np.isfinite(record[k]) for k in ("estimate","qvalue","pvalue"))
                valid += record["valid"]
                data["bc_all"].append(record)
            data["validation"].append(dict(seed=seed, method=method, status=status["status"], valid=valid))
    assert len(data["validation"]) == 15*7
    for method in METHODS + ["oracle_abundance"]:
        data["bc"][method] = [r for r in data["bc_all"] if r["method"]==method and r["valid"]]
    write_csv(OUT/"source_data/BC_LinDA_effect03.csv", data["bc_all"])
    write_csv(OUT/"source_data/BC_coverage.csv", data["validation"])
    data["projection"] = extract_display_points(DR/"figures/projection_effect.pdf")
    data["source_hashes"] = {str(p.relative_to(ROOT)):sha(p) for p in source_paths}
    # Recalculate point-estimate confidence intervals from independent replicate IDs.
    data["d"] = []
    for key in ("truth_distance_spearman","truth_knn_overlap"):
        for method in METHODS[:2]:
            rows = sorted([r for r in data["dr"] if r["phase"]=="mc" and r["condition"]=="effect"
                           and r["method"]==method], key=lambda r:int(r["replicate"]))
            assert len(rows)==100 and len({r["replicate"] for r in rows})==100
            values=np.array([number(r[key]) for r in rows])
            # Same resampling indices for both methods: preserves paired experiment.
            indices=np.random.default_rng(20261010).integers(0,len(values),(10000,len(values)))
            samples=values[indices].mean(1)
            data["d"].append(dict(metric=key, method=method, mean=float(values.mean()),
                                   low=float(np.quantile(samples,.025)),high=float(np.quantile(samples,.975)),n=100))
    write_csv(OUT/"source_data/D_mean_CI.csv",data["d"])
    return data


def method_handles():
    return [Line2D([],[],color=COLORS[m],marker="D" if m=="m4_flex_abundance" else "o",
                   markersize=4,linewidth=1.6 if m=="m4_flex_abundance" else 1.1,label=NAMES[m]) for m in METHODS]


def draw_a(fig,spec,data):
    grid=spec.subgridspec(3,4,wspace=.27,hspace=.38)
    for r,(metric,label) in enumerate((("f1","F1 score"),("power","Sensitivity / recall"),("fdr","False discovery rate"))):
        for c,(engine,title) in enumerate(zip(ENGINES,ENGINE_NAMES)):
            ax=fig.add_subplot(grid[r,c]); clean_axis(ax)
            for method in METHODS:
                rows=sorted([x for x in data["a"] if x["input"]==method and x["method"]==engine
                             and (metric=="fdr" or number(x["s"])>0)],key=lambda x:number(x["s"]))
                x=np.array([number(v["s"]) for v in rows]); y=np.array([number(v[f"complete_{metric}_mean"]) for v in rows])
                se=np.array([number(v[f"complete_{metric}_mcse"]) for v in rows])
                ax.errorbar(x,y,yerr=se,color=COLORS[method],marker="D" if method=="m4_flex_abundance" else "o",
                            markersize=3.2 if method=="m4_flex_abundance" else 2.8,lw=1.65 if method=="m4_flex_abundance" else 1.0,
                            elinewidth=.65,capsize=1.5,zorder=5 if method=="m4_flex_abundance" else 3)
            ax.set_xticks(DOSES if r==2 else DOSES[1:])
            ax.set_xlim((-.025,.775) if r==2 else (.18,.775))
            ax.set_ylim((-.02,.58) if r==2 else (-.03,1.04))
            ax.yaxis.set_major_locator(MultipleLocator(.2))
            if r==2:
                ax.axhline(.05,color="#B999A1",lw=.7,ls=(0,(4,3)),zorder=2)
                ax.set_xlabel(r"Effect magnitude $\delta$")
            if c==0: ax.set_ylabel(label)
            if r==0: ax.set_title(title,pad=7)
    box=spec.get_position(fig)
    fig.legend(handles=method_handles(),loc="lower center",bbox_to_anchor=((box.x0+box.x1)/2,box.y1+.030),
               ncol=6,fontsize=8,columnspacing=1.4,handlelength=2)


def draw_b(fig,spec,data):
    ax=fig.add_subplot(spec); clean_axis(ax)
    offsets=np.linspace(-.32,.32,6)
    for group in (-1,0,1):
        center=group+1
        ax.hlines(group*.3/np.log(2),center-.44,center+.44,color="#767A83",linestyle=(0,(3,2)),linewidth=1.0,zorder=2)
        for j,method in enumerate(METHODS):
            v=np.array([r["estimate"] for r in data["bc"][method] if r["truth_class"]==group])
            bp=ax.boxplot([v],positions=[center+offsets[j]],widths=.09,patch_artist=True,manage_ticks=False,
                          showfliers=True,whis=1.5,
                          boxprops=dict(facecolor=COLORS[method],edgecolor=dark(COLORS[method]),linewidth=.6),
                          medianprops=dict(color="#FFFFFF" if method=="m4_flex_abundance" else dark(COLORS[method],.6),linewidth=.9),
                          whiskerprops=dict(color=dark(COLORS[method]),linewidth=.5),
                          capprops=dict(color=dark(COLORS[method]),linewidth=.5),
                          flierprops=dict(marker=".",markersize=1.2,markerfacecolor=COLORS[method],markeredgewidth=0,alpha=.32))
    ax.set(xticks=[0,1,2],xticklabels=[r"$-0.433$","0",r"$+0.433$"],xlim=(-.55,2.55),ylim=(-1.4,1.2),
           xlabel=r"True group effect (log$_2$)",ylabel=r"Estimated group effect (log$_2$)")
    ax.yaxis.set_major_locator(MultipleLocator(.5))
    ax.legend(handles=[Line2D([],[],color="#767A83",ls=(0,(3,2)),label="True effect")],loc="upper left",fontsize=7)
    box=spec.get_position(fig)
    fig.legend(handles=[Patch(facecolor=COLORS[m],edgecolor="none",label=NAMES[m]) for m in METHODS],
               loc="lower center",bbox_to_anchor=((box.x0+box.x1)/2,box.y1+.013),
               ncol=3,fontsize=7,columnspacing=1.2,handlelength=1.4,handletextpad=.4)


def draw_c(fig,spec,data,all_methods=True):
    methods=(["oracle_abundance"]+METHODS) if all_methods else ["oracle_abundance","naive","m4_flex_abundance"]
    grid=spec.subgridspec(2,4,wspace=.35,hspace=.44) if all_methods else spec.subgridspec(1,3,wspace=.34)
    representative={m:[r for r in data["bc"][m] if r["seed"]==5100] for m in methods}
    max_x=max(abs(r["estimate"]) for rows in representative.values() for r in rows)*1.08
    max_y=max(-np.log10(max(r["qvalue"],np.finfo(float).tiny)) for rows in representative.values() for r in rows)*1.08
    for j,method in enumerate(methods):
        ax=fig.add_subplot(grid[j//4,j%4] if all_methods else grid[0,j])
        for group in (-1,0,1):
            for significant in (False,True):
                rows=[r for r in representative[method] if r["truth_class"]==group and (r["qvalue"]<=.05)==significant]
                ax.scatter([r["estimate"] for r in rows],[-np.log10(max(r["qvalue"],np.finfo(float).tiny)) for r in rows],
                           s=8 if not all_methods else 10,facecolors=CLASS_COLORS[group] if significant else "none",
                           edgecolors=dark(CLASS_COLORS[group],.92),linewidths=.35,alpha=.85,zorder=3 if group else 2)
        ax.axhline(-np.log10(.05),lw=.7,ls=(0,(4,3)),color="#B999A1")
        ax.axvline(0,lw=.5,ls=(0,(2,3)),color="#D4D6DC")
        ax.set(xlim=(-max_x,max_x),ylim=(-.10,max_y),xlabel=r"Effect (log$_2$)",title=NAMES[method])
        ax.xaxis.set_major_locator(MaxNLocator(3)); ax.yaxis.set_major_locator(MaxNLocator(4))
        if j==0 or (all_methods and j==4): ax.set_ylabel(r"$-\log_{10}(q)$")
        if all_methods:
            # One extreme microTensor q value must remain visible without
            # flattening the other six distributions. Shared monotone axis.
            ax.set_yscale("symlog",linthresh=5,linscale=1)
            ax.set_ylim(-.12,max_y)
            ax.set_yticks([0,1,5,20,100])
            ax.set_yticklabels(["0","1","5","20","100"])
            if j in (0,4): ax.set_ylabel(r"$-\log_{10}(q)$ (symlog)")
    box=spec.get_position(fig)
    handles=[Line2D([],[],ls="none",marker="o",markersize=4,color=CLASS_COLORS[k],label=lab)
             for k,lab in [(-1,"True decrease"),(0,"No effect"),(1,"True increase")]]
    handles += [Line2D([],[],ls="none",marker="o",markersize=4,markerfacecolor=fill,
                       markeredgecolor="#969AA4",label=label)
                for fill,label in [("#969AA4",r"$q\leq0.05$"),("none",r"$q>0.05$")]]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=((box.x0+box.x1)/2,box.y1+.035),
               ncol=5,fontsize=6.5,columnspacing=.7,handletextpad=.3)


def draw_d(fig,spec,data):
    grid=spec.subgridspec(2,1,hspace=.72)
    for i,(key,title) in enumerate((("truth_distance_spearman","Distance correlation"),("truth_knn_overlap","Neighborhood recovery"))):
        ax=fig.add_subplot(grid[i,0])
        for j,method in enumerate(METHODS[:2]):
            r=next(r for r in data["d"] if r["method"]==method and r["metric"]==key)
            ax.errorbar(r["mean"],1-j,xerr=[[r["mean"]-r["low"]],[r["high"]-r["mean"]]],fmt="o" if j==0 else "D",
                        color=COLORS[method],ms=5,elinewidth=1.2,capsize=3,lw=0)
        ax.set(yticks=[1,0],yticklabels=["Naive","M4_flex"],ylim=(-.55,1.55),xlim=(0,1.03),title=title,
               xlabel="Spearman correlation" if i==0 else "5-nearest-neighbor overlap")
        ax.set_xticks([0,.25,.5,.75,1]); ax.grid(axis="x",color="#ECEEF2",lw=.5)
        ax.spines["left"].set_visible(False); ax.tick_params(axis="y",length=0)


def draw_e(fig,spec,data):
    grid=spec.subgridspec(1,3,wspace=.28)
    for j,(points,title) in enumerate(zip(data["projection"],["Latent truth","Naive","M4_flex"])):
        ax=fig.add_subplot(grid[0,j])
        for group in (0,1):
            xy=points[points[:,0]==group,1:]
            ax.scatter(xy[:,0],xy[:,1],s=17,marker="o" if group==0 else "^",facecolors=GROUP_COLORS[group],
                       edgecolors=dark(GROUP_COLORS[group]),linewidths=.4,alpha=.95)
        ax.set(xlim=(-22,13),ylim=(-9,8),xticks=[-20,-10,0,10],yticks=[-8,-4,0,4,8],
               title=title,xlabel="Aligned PC1")
        if j==0: ax.set_ylabel("Aligned PC2")
    box=spec.get_position(fig)
    handles=[Line2D([],[],ls="none",marker="o" if g==0 else "^",markerfacecolor=GROUP_COLORS[g],
                    markeredgecolor=dark(GROUP_COLORS[g]),markersize=5,label=f"Group {g}") for g in (0,1)]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=((box.x0+box.x1)/2,box.y1+.036),
               ncol=2,fontsize=8,handletextpad=.4)


def save(fig,name,dpi,exports):
    fig.savefig(OUT/(name+".pdf"),dpi=dpi)
    fig.savefig(OUT/(name+".png"),dpi=dpi)
    with Image.open(OUT/(name+".png")) as im:
        assert abs(im.info["dpi"][0]-dpi)<.1
        rgb=im.convert("RGB")
        rgb.save(OUT/(name+".tiff"),dpi=(dpi,dpi),compression="tiff_lzw")
        size=im.size
    with Image.open(OUT/(name+".tiff")) as im:
        assert im.mode=="RGB" and im.size==size and abs(im.info["dpi"][0]-dpi)<.1
    qa=OUT/"_qa"; qa.mkdir(exist_ok=True)
    # Render the actual PDF, not just the plotting canvas, for visual inspection.
    doc=fitz.open(OUT/(name+".pdf")); page=doc[0]
    page.get_pixmap(matrix=fitz.Matrix(1.7,1.7),alpha=False).save(qa/(name+"_pdf_preview.png"))
    text=page.get_text()
    assert text.strip(), "Expected selectable vector PDF text"
    doc.close()
    exports.append(dict(name=name,pixels=size,dpi=dpi,tiff="RGB/LZW",pdf="vector paths and embedded TrueType fonts"))
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dpi",type=int,default=600)
    p.add_argument("--preview-only",action="store_true")
    args=p.parse_args()
    OUT.mkdir(exist_ok=True); (OUT/"source_data").mkdir(exist_ok=True); (OUT/"_qa").mkdir(exist_ok=True)
    style(); data=prepare_data(); exports=[]
    fig=plt.figure(figsize=(12,16))
    main_grid=fig.add_gridspec(3,1,height_ratios=[5.4,4.0,2.6],left=.066,right=.981,bottom=.046,top=.911,hspace=.51)
    draw_a(fig,main_grid[0],data); panel_letter(fig,main_grid[0],"A",dy=.061)
    middle=main_grid[1].subgridspec(1,2,width_ratios=[.32,.68],wspace=.23)
    draw_b(fig,middle[0],data); panel_letter(fig,middle[0],"B",dy=.035)
    draw_c(fig,middle[1],data); panel_letter(fig,middle[1],"C",dy=.035)
    bottom=main_grid[2].subgridspec(1,2,width_ratios=[.29,.71],wspace=.39)
    draw_d(fig,bottom[0],data); panel_letter(fig,bottom[0],"D",dy=.037)
    draw_e(fig,bottom[1],data); panel_letter(fig,bottom[1],"E",dy=.037)
    if args.preview_only:
        fig.savefig(OUT/"_qa/Figure_ABCDE_layout.png",dpi=160)
        plt.close(fig)
        print("Layout preview ready")
        return
    save(fig,"Figure_ABCDE",args.dpi,exports)
    settings=[("A",draw_a,(11.5,7.8),.87), ("B",draw_b,(7.3,4.5),.80),
              ("C",draw_c,(11.0,6.5),.86), ("D",draw_d,(5.5,5.0),.89),
              ("E",draw_e,(10.5,3.5),.80)]
    for letter,draw,size,top in settings:
        fig=plt.figure(figsize=size)
        spec=fig.add_gridspec(1,1,left=.11 if letter=="D" else .085,right=.98,bottom=.12,top=top)[0]
        draw(fig,spec,data); panel_letter(fig,spec,letter,dx=-.055,dy=.045 if letter in ("A","C","E") else .015)
        save(fig,f"Panel_{letter}",args.dpi,exports)
    fig=plt.figure(figsize=(11.0,6.5))
    spec=fig.add_gridspec(1,1,left=.065,right=.98,bottom=.09,top=.86)[0]
    draw_c(fig,spec,data,all_methods=True)
    save(fig,"Supplement_volcano_all_methods",args.dpi,exports)
    manifest=dict(exports=exports,colors=COLORS,group_colors=GROUP_COLORS,truth_colors=CLASS_COLORS,
                  effect=.3,volcano_seed=5100,projection_seed=7100,source_hashes=data["source_hashes"],
                  bc_coverage=data["validation"],versions=dict(numpy=np.__version__,matplotlib=matplotlib.__version__),
                  notes=dict(A="Existing complete-case curves; mean +/-1 MCSE; variable valid n from source summary.",
                             B="All 15 seeds pooled; descriptive per-taxon coefficient distributions; Tukey boxes including fliers; not independent-replicate confidence intervals.",
                             C="LinDA, fixed seed 5100; BH q; colored by ground truth; filled q<=0.05, open q>0.05; raw gamma divided by ln(2). All volcano panels use a shared symlog y axis, linear through 5, to retain extreme microTensor q values without clipping.",
                             D="100 independent simulations; mean and 95% percentile CI of the mean, 10000 paired-index resamples.",
                             E="Faithful redraw from vector-PDF display coordinates, same axes and sample locations; no raw score reconstruction or new inference. Group labels are predefined, not clustering output. Bootstrap block omitted."))
    (OUT/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(f"Exported {len(exports)} figures in PDF, PNG and TIFF at {args.dpi} dpi.")


if __name__=="__main__":
    main()
