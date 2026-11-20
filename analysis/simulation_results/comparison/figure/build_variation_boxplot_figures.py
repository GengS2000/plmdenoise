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
parser.add_argument('--out', type=Path, default=Path(__file__).parent / 'publication_v7_variation')
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
    ('comp_log_rmse', 'Log-RMSE', 'lower', 'log10'),
    ('comp_bray_curtis', 'Bray-Curtis', 'lower', 'log10'),
    ('comp_aitchison', 'Aitchison', 'lower', 'log10'),
    ('comp_trajectory_r', 'Trajectory r', 'higher', 'fisher'),
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
        assert value > 0
        return float(np.log10(value))
    if metric == 'comp_trajectory_r':
        assert -1 < value < 1, 'Fisher z requires correlation strictly between -1 and 1'
        return float(np.arctanh(value))
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
                        transform='log10(raw error)' if scale=='log10' else 'atanh(r)' if scale=='fisher' else 'identity'))
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
limits, variation_limits, deviations = {}, {}, []
for metric, _, _, scale in METRICS:
    vals = [r['value'] for r in plotted if r['metric'] == metric]
    lo, hi = min(vals), max(vals)
    if scale == 'log':
        pad = (np.log10(hi)-np.log10(lo))*.06
        limits[metric] = (10**(np.log10(lo)-pad), 10**(np.log10(hi)+pad))
        continue
    pad = (hi-lo)*.06
    limits[metric] = (lo-pad, hi+pad)
    for condition in CONDS:
        for scenario in range(1,9):
            for method in METHODS:
                seeds = paired[scenario,condition]
                if not seeds:
                    continue
                values = np.array([plot_value(scenario,condition,seed,method,metric) for seed in seeds])
                median = float(np.median(values))
                for seed, value in zip(seeds,values):
                    deviations.append(dict(metric=metric,scenario=scenario,condition=condition,
                        method=method,seed=seed,transformed_value=value,group_median=median,
                        deviation=value-median))
            dev = [abs(r['deviation']) for r in deviations if r['metric']==metric and r['scenario']==scenario]
            if dev:
                # One symmetric range per metric/scenario, shared by all methods and conditions.
                variation_limits[metric,scenario] = max(dev)*1.12
csv_out('variation_seed_values.csv', deviations)
variation_lookup = {(r['metric'],r['scenario'],r['condition'],r['method'],r['seed']):r['deviation'] for r in deviations}
ROMAN = ['i','ii','iii','iv','v']

def style(ax, show_ticks=True):
    ax.set_xlim(.35,6.65)
    ax.set_xticks(range(1,7), ['']*6)
    ax.tick_params(axis='x',length=0)
    ax.tick_params(axis='y',which='both',length=2 if show_ticks else 0,labelleft=show_ticks,pad=2,labelsize=7)
    ax.set_axisbelow(True)
    ax.grid(axis='y',color='#E6E5E9',lw=.45)
    for spine in ax.spines.values():
        spine.set_color('#9B96A1')
        spine.set_linewidth(.5)

def draw_box(ax, values, x, color, scenario, row):
    ax.boxplot([values],positions=[x],widths=.86,patch_artist=True,manage_ticks=False,
        showfliers=False,whis=1.5,
        boxprops=dict(facecolor=color,edgecolor='#5F596A',linewidth=.6),
        whiskerprops=dict(color='#686171',linewidth=.6),
        capprops=dict(color='#686171',linewidth=.6),
        medianprops=dict(color='#292532',linewidth=.8))
    jitter=np.random.default_rng(scenario*100+row).uniform(-.20,.20,len(values))
    ax.scatter(x+jitter,values,s=5,c=color,edgecolors='#514C59',linewidths=.15,alpha=.6,zorder=3)

for condition in CONDS:
    fig=plt.figure(figsize=(16,11.6))
    grid=fig.add_gridspec(5,8,left=.155,right=.985,top=.895,bottom=.025,
        height_ratios=[1.4,1.4,1.4,1.4,.85],hspace=.30,wspace=.34)
    fig.text(.155,.977,'Recovery accuracy, longitudinal fidelity and fitting time',fontsize=15,weight='bold')
    fig.legend(handles=[Patch(facecolor=c,edgecolor='#595566',linewidth=.5,label=n)
        for c,n in zip(COLORS,NAMES)],loc='upper left',bbox_to_anchor=(.149,.963),
        ncol=6,frameon=False,fontsize=9.5,handlelength=1.5,columnspacing=1.8)
    for row,(metric,label,direction,scale) in enumerate(METRICS):
        for col,scenario in enumerate(range(1,9)):
            if row<4:
                inner=grid[row,col].subgridspec(2,1,height_ratios=[1.75,1],hspace=.22)
                ax=fig.add_subplot(inner[0])
                var_ax=fig.add_subplot(inner[1])
                style(var_ax)
                var_ax.set_facecolor('#F7F6F9')
                bound=variation_limits.get((metric,scenario),.01)
                var_ax.set_ylim(-bound,bound)
                var_ax.yaxis.set_major_locator(MaxNLocator(nbins=2,symmetric=True))
                var_ax.ticklabel_format(axis='y',style='plain',useOffset=False)
                var_ax.axhline(0,color='#99939E',ls='--',lw=.5)
            else:
                ax=fig.add_subplot(grid[row,col])
                var_ax=None
            style(ax,col==0)
            ax.set_ylim(limits[metric])
            if scale=='log':
                ax.set_yscale('log')
                ax.yaxis.set_major_locator(LogLocator(base=10,numticks=4))
                ax.yaxis.set_minor_formatter(NullFormatter())
            else:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            if row==0:
                ax.add_patch(Rectangle((0,1.04),1,.24,transform=ax.transAxes,
                    facecolor='#E7E5EB',edgecolor='#8D8995',linewidth=.55,clip_on=False))
                ax.text(.5,1.16,f'S{scenario}',ha='center',va='center',transform=ax.transAxes,fontsize=9)
            if col==0:
                pos=ax.get_position()
                coord='log10(error)' if scale=='log10' else 'Fisher z(r)' if scale=='fisher' else 'log axis'
                fig.text(.022,(pos.y0+pos.y1)/2,f'{ROMAN[row]}  {label}\n{coord}\n{direction} is better',
                    fontsize=10,va='center',linespacing=1.6)
                if var_ax is not None:
                    pos=var_ax.get_position()
                    fig.text(.022,(pos.y0+pos.y1)/2,'Within-method variation\n'+
                        ('Δ log10(error)' if scale=='log10' else 'Δ Fisher z(r)')+' from median',
                        fontsize=8,va='center',color='#686171',linespacing=1.5)
            seeds=paired[scenario,condition]
            if not seeds:
                for missing_ax in [ax,var_ax]:
                    if missing_ax is None:
                        continue
                    missing_ax.set_facecolor('#F4F3F6')
                    missing_ax.text(.5,.5,'Not rerun',transform=missing_ax.transAxes,ha='center',va='center',fontsize=8,color='#797480')
                    if missing_ax is var_ax:
                        missing_ax.tick_params(axis='y',labelleft=False)
                continue
            for x,(method,color) in enumerate(zip(METHODS,COLORS),1):
                if metric=='fit_seconds' and method=='naive':
                    ax.text(x,.06,'N/A',transform=ax.get_xaxis_transform(),ha='center',fontsize=7,color='#77717B')
                    continue
                values=np.array([plot_value(scenario,condition,seed,method,metric) for seed in seeds])
                draw_box(ax,values,x,color,scenario,row)
                if var_ax is not None:
                    dev=np.array([variation_lookup[metric,scenario,condition,method,seed] for seed in seeds])
                    draw_box(var_ax,dev,x,color,scenario,row)
    dest=args.out/('Figure_'+condition)
    fig.savefig(dest.with_suffix('.pdf'),metadata={'Title':condition,'Subject':'Log-scale accuracy and separate median-centered variability panels'})
    fig.savefig(dest.with_suffix('.png'),dpi=args.dpi)
    fig.savefig(dest.with_name(dest.name+'_preview').with_suffix('.png'),dpi=150)
    plt.close(fig)
    with Image.open(dest.with_suffix('.png')) as im:
        im.convert('RGB').save(dest.with_suffix('.tiff'),compression='tiff_lzw',dpi=(args.dpi,args.dpi))
    print('Exported '+str(dest),flush=True)

manifest=dict(source=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    palette=dict(zip(METHODS,COLORS)),method_order=METHODS,conditions=CONDS,matched_datasets=449,
    metrics=METRICS,axis_limits=limits,
    variation_limits={f'{m}|S{s}':[-v,v] for (m,s),v in variation_limits.items()},dpi=args.dpi,
    main_transform='i-iii: log10(raw error); iv: atanh(r)=0.5*ln((1+r)/(1-r)); v: raw seconds on log axis',
    variation_transform='Transformed value minus method/scenario/condition median; symmetric local axis per scenario/metric, shared across methods and four conditions',
    box_definition='Quartiles and 1.5 IQR whiskers on plotted values; all seed points shown; no artificial minimum box height',
    removed_annotations='No bottom notes, n labels, or condition subtitle; lower-case Roman row labels',
    naive_time='Not measured; N/A',layout='Five metric rows; i-iv have separate level and variation subpanels')
(args.out/'figure_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
