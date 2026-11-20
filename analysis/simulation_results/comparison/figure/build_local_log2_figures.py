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
parser.add_argument('--out', type=Path, default=Path(__file__).parent / 'publication_v8_local_log2')
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
    ('comp_log_rmse', 'Log-RMSE', 'lower', 'log2'),
    ('comp_bray_curtis', 'Bray-Curtis', 'lower', 'log2'),
    ('comp_aitchison', 'Aitchison', 'lower', 'log2'),
    ('comp_trajectory_r', 'Trajectory error', 'lower', 'log2loss'),
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
        return float(np.log2(value))
    if metric == 'comp_trajectory_r':
        assert -1 <= value < 1
        return float(np.log2(1-value))
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
                        transform='log2(raw error)' if scale=='log2' else 'log2(1-r)' if scale=='log2loss' else 'identity'))
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
for condition in CONDS:
    for scenario in range(1,9):
        for metric,_,_,scale in METRICS:
            values=[r['value'] for r in plotted if r['metric']==metric and r['condition']==condition and r['scenario']==scenario]
            if not values:
                continue
            lo,hi=min(values),max(values)
            if scale=='log':
                pad=(np.log2(hi)-np.log2(lo))*.04
                limits[condition,scenario,metric]=(2**(np.log2(lo)-pad),2**(np.log2(hi)+pad))
            else:
                pad=max((hi-lo)*.04,.005)
                limits[condition,scenario,metric]=(lo-pad,hi+pad)

ROMAN=['i','ii','iii','iv','v']
for condition in CONDS:
    fig,axes=plt.subplots(5,8,figsize=(16,11.2))
    fig.subplots_adjust(left=.145,right=.985,top=.89,bottom=.03,hspace=.23,wspace=.34)
    fig.text(.145,.972,'Recovery accuracy, longitudinal fidelity and fitting time',fontsize=16,weight='bold')
    fig.legend(handles=[Patch(facecolor=c,edgecolor='#595566',linewidth=.5,label=n)
        for c,n in zip(COLORS,NAMES)],loc='upper left',bbox_to_anchor=(.138,.956),
        ncol=6,frameon=False,fontsize=10,handlelength=1.5,columnspacing=1.8)
    for row,(metric,label,direction,scale) in enumerate(METRICS):
        for col,scenario in enumerate(range(1,9)):
            ax=axes[row,col]
            seeds=paired[scenario,condition]
            ax.set_xlim(.35,6.65)
            ax.set_xticks(range(1,7),['']*6)
            ax.tick_params(axis='x',length=0)
            ax.tick_params(axis='y',which='both',length=2,pad=2,labelsize=8)
            ax.set_axisbelow(True)
            ax.grid(axis='y',color='#E6E5E9',lw=.5)
            for spine in ax.spines.values():
                spine.set_color('#8D8995')
            if row==0:
                ax.add_patch(Rectangle((0,1.02),1,.19,transform=ax.transAxes,
                    facecolor='#E7E5EB',edgecolor='#8D8995',linewidth=.55,clip_on=False))
                ax.text(.5,1.115,f'S{scenario}',ha='center',va='center',transform=ax.transAxes,fontsize=10)
            if col==0:
                pos=ax.get_position()
                coord='log2(error)' if scale=='log2' else 'log2(1 - r)' if scale=='log2loss' else 'log2 axis'
                fig.text(.017,(pos.y0+pos.y1)/2,f'{ROMAN[row]}  {label}\n{coord}\nlower is better\nlocal y-scale',
                    fontsize=10,va='center',linespacing=1.6)
            if not seeds:
                ax.set_facecolor('#F4F3F6')
                ax.set_yticks([])
                ax.text(.5,.5,'Not rerun',transform=ax.transAxes,ha='center',va='center',fontsize=8,color='#797480')
                continue
            ax.set_ylim(limits[condition,scenario,metric])
            if scale=='log':
                ax.set_yscale('log',base=2)
                ax.yaxis.set_major_locator(LogLocator(base=2,numticks=4))
                ax.yaxis.set_minor_formatter(NullFormatter())
            else:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
                ax.ticklabel_format(axis='y',style='plain',useOffset=False)
            for x,(method,color) in enumerate(zip(METHODS,COLORS),1):
                if metric=='fit_seconds' and method=='naive':
                    ax.text(x,.06,'N/A',transform=ax.get_xaxis_transform(),ha='center',fontsize=7,color='#77717B')
                    continue
                values=np.array([plot_value(scenario,condition,seed,method,metric) for seed in seeds])
                ax.boxplot([values],positions=[x],widths=.86,patch_artist=True,manage_ticks=False,
                    showfliers=False,whis=1.5,
                    boxprops=dict(facecolor=color,edgecolor='#5F596A',linewidth=.6),
                    whiskerprops=dict(color='#686171',linewidth=.6),
                    capprops=dict(color='#686171',linewidth=.6),
                    medianprops=dict(color='#292532',linewidth=.85))
                jitter=np.random.default_rng(scenario*100+row).uniform(-.18,.18,len(values))
                ax.scatter(x+jitter,values,s=7,c=color,edgecolors='#514C59',linewidths=.15,alpha=.6,zorder=3)
    dest=args.out/('Figure_'+condition)
    fig.savefig(dest.with_suffix('.pdf'),metadata={'Title':condition,'Subject':'Local log2 scales; one panel per metric and scenario'})
    fig.savefig(dest.with_suffix('.png'),dpi=args.dpi)
    fig.savefig(dest.with_name(dest.name+'_preview').with_suffix('.png'),dpi=150)
    plt.close(fig)
    with Image.open(dest.with_suffix('.png')) as im:
        im.convert('RGB').save(dest.with_suffix('.tiff'),compression='tiff_lzw',dpi=(args.dpi,args.dpi))
    print('Exported '+str(dest),flush=True)

manifest=dict(source=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    palette=dict(zip(METHODS,COLORS)),method_order=METHODS,conditions=CONDS,matched_datasets=449,
    metrics=METRICS,axis_limits={'|'.join(map(str,k)):v for k,v in limits.items()},dpi=args.dpi,
    axis_policy='Each scenario-condition-metric panel uses all method and seed values plus 4% range margin. All six methods share each panel scale; scales differ between panels.',
    transform='i-iii log2(raw error); iv log2(1-r); v seconds on base-2 log axis',
    box_definition='Quartiles and 1.5 IQR whiskers on plotted values; all seed points visible; no added variability',
    removed_annotations='No variation panels, footer notes, sample sizes or condition subtitles; lower-case Roman row labels',
    naive_time='Not measured; N/A',layout='5 metric rows x 8 scenario columns')
(args.out/'figure_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
