"""mbDenoise-style metric-by-scenario facets, separately for four conditions.

Run from repository root with the bundled Python and --deps tmp/publication_plot_deps.
Uses existing seed-level scores only; no fitting, rescoring, or imputation.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--deps', type=Path)
parser.add_argument('--out', type=Path, default=Path(__file__).parent / 'publication_v6_selected')
parser.add_argument('--dpi', type=int, default=600)
args = parser.parse_args()
if args.deps:
    sys.path.insert(0, str(args.deps.resolve()))
args.out.mkdir(parents=True, exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(args.out / '_qa' / 'mpl_cache')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator, LogLocator, NullFormatter
from PIL import Image

SOURCE = Path(__file__).resolve().parents[1] / 'comparison_long.csv'
METHODS = ['naive', 'm4_flex', 'embed', 'microtensor', 'parafac2', 'tempted_gemelli']
NAMES = ['Naive (no denoise)', 'Ours', 'EMBED', 'microTensor', 'PARAFAC2', 'TEMPTED']
COLORS = ['#A8B6C6', '#4F587D', '#C68DC0', '#C2E0EE', '#776B97', '#DBC8ED']
CONDS = ['base', 'depth_5000', 'depth_5000_quadratic', 'quadratic']
TITLES = ['Base | depth 30,000 | linear', 'Depth 5,000 | linear',
          'Depth 5,000 | quadratic', 'Depth 30,000 | quadratic']
METRICS = [
    ('comp_log_rmse', 'Log-RMSE', 'lower', 'relative'),
    ('comp_bray_curtis', 'Bray-Curtis', 'lower', 'relative'),
    ('comp_aitchison', 'Aitchison', 'lower', 'relative'),
    ('comp_trajectory_r', 'Trajectory r', 'higher', 'linear'),
    ('fit_seconds', 'Time (seconds)', 'lower', 'log'),
]
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.linewidth': .55,
    'xtick.labelsize': 7, 'ytick.labelsize': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.facecolor': 'white', 'text.color': '#292B35', 'axes.labelcolor': '#292B35'})
rows = list(csv.DictReader(SOURCE.open(encoding='utf-8-sig')))
raw = {}
for r in rows:
    key = (int(r['scenario']), r['condition'], int(r['seed']), r['method'])
    assert key not in raw, ('duplicate', key)
    raw[key] = r
assert {r['method'] for r in rows} == set(METHODS)

def plot_value(scenario, condition, seed, method, metric):
    value = float(raw[scenario, condition, seed, method][metric])
    if metric in ['comp_log_rmse', 'comp_bray_curtis', 'comp_aitchison']:
        baseline = float(raw[scenario, condition, seed, 'naive'][metric])
        assert value > 0 and baseline > 0
        return float(np.log2(value / baseline))
    return value

paired, coverage, plotted, summary = {}, [], [], []
for condition in CONDS:
    for scenario in range(1, 9):
        sets = [{k[2] for k in raw if k[0] == scenario and k[1] == condition and k[3] == m} for m in METHODS]
        shared = sorted(set.intersection(*sets))
        paired[scenario, condition] = shared
        coverage.append(dict(scenario=scenario, condition=condition, paired_n=len(shared),
                             **{m: len(s) for m, s in zip(METHODS, sets)}))
        for seed in shared:
            assert len({raw[scenario, condition, seed, m]['dataset_id'] for m in METHODS}) == 1
        for metric, _, _, scale in METRICS:
            for method in METHODS:
                if not shared or (metric == 'fit_seconds' and method == 'naive'):
                    continue
                vals = [plot_value(scenario, condition, seed, method, metric) for seed in shared]
                assert np.isfinite(vals).all()
                if scale == 'log':
                    assert min(vals) > 0, (metric, scenario, condition, method)
                for seed, value in zip(shared, vals):
                    plotted.append(dict(scenario=scenario, condition=condition, seed=seed,
                        dataset_id=raw[scenario, condition, seed, method]['dataset_id'],
                        metric=metric, method=method, value=value,
                        raw_value=float(raw[scenario, condition, seed, method][metric]),
                        transform='log2(method/paired naive)' if scale=='relative' else 'identity'))
                q1, med, q3 = np.percentile(vals, [25, 50, 75])
                iqr = q3-q1
                inside = [v for v in vals if q1-1.5*iqr <= v <= q3+1.5*iqr]
                summary.append(dict(scenario=scenario, condition=condition, metric=metric,
                    method=method, n=len(vals), q1=q1, median=med, q3=q3,
                    whisker_low=min(inside), whisker_high=max(inside)))
assert sum(map(len, paired.values())) == 449
assert {k for k,v in paired.items() if not v} == {(1, 'quadratic'), (1, 'depth_5000_quadratic')}

def csv_out(name, records):
    with (args.out / name).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

csv_out('plotted_seed_values.csv', plotted)
csv_out('boxplot_statistics.csv', summary)
csv_out('coverage_audit.csv', coverage)
limits = {}
for metric, _, _, scale in METRICS:
    vals = [r['value'] for r in plotted if r['metric'] == metric]
    lo, hi = min(vals), max(vals)
    if scale == 'log':
        pad = (np.log10(hi)-np.log10(lo)) * .065
        limits[metric] = (10**(np.log10(lo)-pad), 10**(np.log10(hi)+pad))
    elif scale == 'relative':
        limits[metric] = (min(-2,lo)-.18, max(0,hi)+.18)
    elif metric in ['comp_trajectory_r', 'comp_pearson', 'comp_spearman']:
        limits[metric] = (min(0, lo-.05), 1.06)
    else:
        lo, hi = min(0, lo), max(1 if metric == 'comp_slope' else 0, hi)
        pad = (hi-lo)*.09
        limits[metric] = (lo-pad, hi+pad)

for condition, title in zip(CONDS, TITLES):
    fig, axes = plt.subplots(len(METRICS), 8, figsize=(15.6, 9.4))
    fig.subplots_adjust(left=.145, right=.985, top=.855, bottom=.145, hspace=.16, wspace=.09)
    fig.text(.145, .973, 'Recovery accuracy, longitudinal fidelity and fitting time', fontsize=16, weight='bold')
    fig.text(.145, .939, title, fontsize=12, color=COLORS[1])
    fig.legend(handles=[Patch(facecolor=c, edgecolor='#595566', linewidth=.5, label=n)
        for c,n in zip(COLORS,NAMES)], loc='upper left', bbox_to_anchor=(.138,.922),
        ncol=6, frameon=False, fontsize=9, handlelength=1.5, columnspacing=1.8)
    for row, (metric, label, direction, scale) in enumerate(METRICS):
        for col, scenario in enumerate(range(1,9)):
            ax = axes[row,col]
            seeds = paired[scenario,condition]
            ax.set_yscale('linear' if scale=='relative' else scale)
            ax.set_ylim(limits[metric])
            ax.set_xlim(.35,6.65)
            ax.set_xticks(range(1,7), ['']*6)
            ax.tick_params(axis='x', length=0)
            ax.tick_params(axis='y', which='both', length=2 if col==0 else 0, labelleft=col==0, pad=3)
            ax.set_axisbelow(True)
            ax.grid(axis='y', color='#E6E5E9', lw=.5)
            ax.grid(axis='x', color='#F0EFF2', lw=.45)
            if scale == 'log':
                ax.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
                ax.yaxis.set_minor_formatter(NullFormatter())
            elif scale == 'relative':
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            else:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            for spine in ax.spines.values():
                spine.set_color('#8D8995')
            if row==0:
                ax.add_patch(Rectangle((0,1.02),1,.26,transform=ax.transAxes,
                    facecolor='#E7E5EB',edgecolor='#8D8995',linewidth=.55,clip_on=False))
                ax.text(.5,1.15,f'S{scenario}  |  n={len(seeds)}' if seeds else f'S{scenario}  |  omitted',
                        ha='center',va='center',transform=ax.transAxes,fontsize=8.5)
            if col==0:
                pos=ax.get_position()
                fig.text(.022,(pos.y0+pos.y1)/2,
                    f'{chr(65+row)}  {label}\n'+('log2(error / naive)\nlower is better' if scale=='relative' else f'{direction}; {"log axis" if scale=="log" else "linear"}'),
                    fontsize=9,va='center',linespacing=1.6)
            if not seeds:
                ax.set_facecolor('#F4F3F6')
                ax.text(.5,.5,'Same design\nas '+('base' if condition=='quadratic' else 'depth 5,000')+
                        '\n(not rerun)',transform=ax.transAxes,ha='center',va='center',
                        fontsize=7,color='#797480')
                continue
            if scale == 'relative':
                ax.axhline(0,color='#827C89',ls='--',lw=.8)
            if metric in ['comp_log_bias','comp_slope']:
                ax.axhline(0 if metric=='comp_log_bias' else 1, color='#827C89',ls='--',lw=.65)
            for x,(method,color) in enumerate(zip(METHODS,COLORS),1):
                if metric=='fit_seconds' and method=='naive':
                    ax.text(x,.08,'N/A',transform=ax.get_xaxis_transform(),ha='center',fontsize=6.5,color='#77717B')
                    continue
                values=np.array([plot_value(scenario,condition,seed,method,metric) for seed in seeds])
                ax.boxplot([values], positions=[x], widths=.82, patch_artist=True, manage_ticks=False,
                    showfliers=False, whis=1.5,
                    boxprops=dict(facecolor=color,edgecolor='#5F596A',linewidth=.7),
                    whiskerprops=dict(color='#686171',linewidth=.6),
                    capprops=dict(color='#686171',linewidth=.6),
                    medianprops=dict(color='#292532',linewidth=1.1))
                # Every seed is visible once; deterministic jitter shared across methods.
                jitter=np.random.default_rng(scenario*100+row).uniform(-.17,.17,len(values))
                ax.scatter(x+jitter, values, s=7, c=color, edgecolors='#514C59',linewidths=.18,alpha=.55,zorder=3)
    fig.text(.145,.112,'A-C: paired log2 error ratio.  -2 = one-quarter of naive error;  -1 = one-half;  0 = naive;  +1 = twice naive error.',fontsize=9)
    fig.text(.145,.088,'Left to right: Naive / Ours / EMBED / microTensor / PARAFAC2 / TEMPTED. Naive ratios equal zero by definition.',fontsize=8.5)
    fig.text(.145,.064,'Boxes: median and IQR; whiskers: 1.5 IQR; dots: matched seeds. Same metric uses identical axes across all four figures.',fontsize=8.5)
    fig.text(.145,.040,'n=15 per cell except S4-base (n=14). S1 quadratic designs were omitted. Naive time: not measured (N/A).',fontsize=8.5)
    fig.text(.145,.016,'Time is recorded elapsed fitting time; CPU allocations differ. All nine original recovery metrics remain available in the full v5 figures.',fontsize=8.5)
    dest=args.out / ('Figure_'+condition)
    fig.savefig(dest.with_suffix('.pdf'), metadata={'Title':title,'Subject':'Paired seed boxplots; metric rows and scenario columns'})
    fig.savefig(dest.with_suffix('.png'),dpi=args.dpi)
    fig.savefig(dest.with_name(dest.name+'_preview').with_suffix('.png'),dpi=150)
    plt.close(fig)
    with Image.open(dest.with_suffix('.png')) as im:
        im.convert('RGB').save(dest.with_suffix('.tiff'),compression='tiff_lzw',dpi=(args.dpi,args.dpi))
    print('Exported '+str(dest),flush=True)

manifest = dict(source=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    palette=dict(zip(METHODS,COLORS)),method_order=METHODS,conditions=CONDS,
    matched_datasets=449,metrics=METRICS,axis_limits=limits,dpi=args.dpi,
    box_definition='Quartiles and 1.5 IQR whiskers on plotted values; A-C use log2 of seed-paired naive error ratio; all paired seed points shown',
    naive_time='Not measured; N/A, never zero or imputed',
    layout='5 metric rows x 8 scenario columns; one figure per condition')
(args.out/'figure_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
