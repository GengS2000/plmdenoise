"""Rebuild publication figures from seed-level scores; no fitting or rescoring.

2026-09-25 | publication_v4（当前版本）
按用户要求移除理论真值基准柱：误差为 0 的理想参考没有方法比较意义。
每个条件内从左到右：Naive、Ours、EMBED、microTensor、PARAFAC2、TEMPTED。
Naive 是未去噪观测相对真值的评分，不是真值本身；保留灰蓝柱及 SE。
其余配色、统计、条件、面板布局和主图 I 的改善率计算保持不变。
旧版 publication_v2/v3 保留；当前默认导出到 publication_v4。

Example: python build_publication_figures.py --deps ../../../../tmp/publication_plot_deps
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--deps', type=Path)
p.add_argument('--out', type=Path, default=Path(__file__).parent / 'publication_v4')
a = p.parse_args()
if a.deps:
    sys.path.insert(0, str(a.deps.resolve()))
a.out.mkdir(parents=True, exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(a.out / '_qa' / 'mpl_cache')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from PIL import Image, ImageDraw

SOURCE = Path(__file__).resolve().parent.parent / 'comparison_long.csv'
METHODS = ['m4_flex', 'embed', 'microtensor', 'parafac2', 'tempted_gemelli']
NAMES = ['Ours', 'EMBED', 'microTensor', 'PARAFAC2', 'TEMPTED']
COLORS = ['#4F587D', '#C68DC0', '#C2E0EE', '#776B97', '#DBC8ED']
# 六柱顺序固定；不添加理论真值柱，也不恢复 naive 虚线。
BAR_METHODS = ['naive'] + METHODS
BAR_NAMES = ['Naive (observed)'] + NAMES
BAR_COLORS = ['#A8B6C6'] + COLORS
CONDS = ['base', 'depth_5000', 'quadratic', 'depth_5000_quadratic']
CNAMES = ['Base', 'Depth 5,000', 'Quadratic', 'Depth 5,000 + quadratic']
SHORT = ['B', 'D', 'Q', 'D+Q']
METRICS = [
    ('comp_log_rmse', 'Log-RMSE', 'Lower is better'),
    ('comp_bray_curtis', 'Bray–Curtis distance', 'Lower is better'),
    ('comp_trajectory_r', 'Trajectory correlation', 'Higher is better'),
    ('comp_aitchison', 'Aitchison distance', 'Lower is better'),
    ('comp_log_mae', 'Log-MAE', 'Lower is better'),
    ('comp_log_bias', 'Mean log bias', 'Closer to 0 is better'),
    ('comp_pearson', 'Pearson correlation', 'Higher is better'),
    ('comp_spearman', 'Spearman correlation', 'Higher is better'),
    ('comp_slope', 'Calibration slope', 'Closer to 1 is better'),
]
plt.rcParams.update({'font.family': 'Arial', 'font.size': 7.5, 'axes.labelsize': 7.5,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.linewidth': .6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'axes.spines.top': False,
    'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'text.color': '#292B35', 'axes.labelcolor': '#292B35', 'savefig.facecolor': 'white'})
rows = list(csv.DictReader(SOURCE.open(encoding='utf-8-sig')))
raw = {}
datasets = {}
for r in rows:
    key = (int(r['scenario']), r['condition'], int(r['seed']), r['method'])
    assert key not in raw, ('duplicate', key)
    raw[key] = r
    cell = key[:3]
    datasets.setdefault(cell, set()).add(r['dataset_id'])
assert all(len(v) == 1 for v in datasets.values()), 'Methods must share dataset IDs'
assert set(r['method'] for r in rows) == set(METHODS + ['naive'])
assert set(r['condition'] for r in rows) == set(CONDS)
seeds = {}
stats = {}
export = []
coverage = []
for s in range(1, 9):
    for c in CONDS:
        sets = [{k[2] for k in raw if k[0] == s and k[1] == c and k[3] == m}
                for m in METHODS + ['naive']]
        shared = sorted(set.intersection(*sets))
        seeds[s, c] = shared
        coverage.append(dict(scenario=s, condition=c, paired_n=len(shared),
                             **{m: len(v) for m, v in zip(METHODS + ['naive'], sets)}))
        for metric, _, _ in METRICS:
            for m in METHODS + ['naive']:
                if not shared:
                    continue
                v = np.array([float(raw[s, c, seed, m][metric]) for seed in shared])
                assert np.isfinite(v).all(), (s, c, m, metric)
                mean, se = float(v.mean()), float(v.std(ddof=1)/np.sqrt(len(v)))
                stats[metric, s, c, m] = (mean, se)
                export.append(dict(metric=metric, scenario=s, condition=c, method=m,
                                   n=len(v), mean=mean, se=se, seeds=';'.join(map(str, shared))))
assert sum(len(v) for v in seeds.values()) == 449

def write_csv(name, records):
    with (a.out / name).open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)

write_csv('plotted_data_paired.csv', export)
write_csv('coverage_audit.csv', coverage)
manifest = dict(source=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                dpi=600, palette=dict(zip(METHODS, COLORS)), uncertainty='mean +/- 1 SE; sample SD / sqrt(n)',
                pairing='All five methods and naive share the same seeds within each cell', figures=[])
previews = []
manifest['bar_order'] = BAR_METHODS
manifest['palette'].update(naive=BAR_COLORS[0])
manifest['revision'] = 'v4: removed theoretical truth bar; naive and five methods retained; statistics unchanged'

def save(fig, name):
    folder = a.out / name.split('/')[0]
    folder.mkdir(exist_ok=True)
    dest = a.out / name
    fig.savefig(dest.with_suffix('.png'), dpi=600)
    fig.savefig(dest.with_suffix('.pdf'), dpi=600, metadata={'Title': name, 'Subject': 'Mean ± SE across paired simulation seeds'})
    with Image.open(dest.with_suffix('.png')) as im:
        assert all(abs(x-600)<.1 for x in im.info['dpi'])
        rgb = im.convert('RGB')
        rgb.save(dest.with_suffix('.tiff'), compression='tiff_lzw', dpi=(600,600))
        rgb.thumbnail((1400,1600))
        qa = a.out / '_qa'
        qa.mkdir(exist_ok=True)
        rgb.save(qa / (name.replace('/', '__') + '_preview.png'))
        previews.append((name, rgb.copy()))
    with Image.open(dest.with_suffix('.tiff')) as im:
        assert im.mode == 'RGB' and all(abs(x-600)<.1 for x in im.info['dpi'])
        size = im.size
    manifest['figures'].append(dict(name=name, pixels=size, inches=list(fig.get_size_inches())))
    plt.close(fig)
    print('Exported ' + name, flush=True)

def limits(metric, scenario=None):
    v = [x for k,x in stats.items() if k[0]==metric and (scenario is None or k[1]==scenario)]
    lo = min(m-e for m,e in v)
    hi = max(m+e for m,e in v)
    if metric == 'comp_log_bias':
        return (min(0,lo)*1.12, max(.015, hi*1.18))
    if metric in ('comp_trajectory_r', 'comp_pearson', 'comp_spearman'):
        return (min(0,lo*1.1), max(1.04, hi*1.04))
    return (min(0,lo*1.1), max(1.06 if metric=='comp_slope' else 0,hi*1.13))

def legend(fig, metric, y):
    handles = [Patch(facecolor=c, edgecolor='#565361', linewidth=.45, label=n)
               for c,n in zip(BAR_COLORS,BAR_NAMES)]
    # Matplotlib fills legend columns first; reorder to read in bar order by row.
    handles = [handles[i] for i in [0,3,1,4,2,5]]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,y), ncol=3,
               frameon=False, fontsize=7, columnspacing=1.5, handlelength=1.5)

def draw(ax, metric, cells, ticks, letter, title, ylabel=True):
    for j,(s,c) in enumerate(cells):
        if not seeds[s,c]:
            ax.axvspan(j-.43,j+.43,color='#F5F4F7',zorder=0)
            missing_label = 'Same design as B' if s == 1 and c == 'quadratic' else 'Same design as D'
            ax.text(j,.48,missing_label,rotation=90,ha='center',va='center',
                    transform=ax.get_xaxis_transform(),fontsize=6.5,color='#77747D')
            continue
        for k,m in enumerate(BAR_METHODS):
            mean,se=stats[metric,s,c,m]
            # 保留 v3 柱宽与间距；移除真值柱后将六柱整体居中。
            x = j+(k-2.5)*.112
            ax.bar(x,mean,width=.102,color=BAR_COLORS[k],edgecolor='#565361',
                   linewidth=.35,zorder=3)
            ax.errorbar(x,mean,yerr=se,fmt='none',ecolor='#303039',
                        elinewidth=.65,capsize=1.1,capthick=.65,zorder=5)
        if len(seeds[s,c]) != 15:
            ticks[j] += '*'
    ax.set_xticks(range(len(cells)), ticks)
    ax.set_xlim(-.6,len(cells)-.4)
    scenario_set = {s for s,c in cells}
    ax.set_ylim(limits(metric, next(iter(scenario_set)) if len(scenario_set)==1 else None))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.set_axisbelow(True)
    ax.grid(axis='y',color='#E9E7ED',lw=.45)
    ax.set_title(title,loc='left',fontsize=8,pad=8,weight='bold')
    ax.text(-.16,1.06,letter,transform=ax.transAxes,fontsize=10,weight='bold',color=COLORS[0])
    if ylabel:
        ax.set_ylabel(dict((m,l) for m,l,_ in METRICS)[metric])

FOOT = 'Naive (observed) and five methods: mean ± SE across paired seeds; n = 15, * n = 14.'
KEY = 'B: base   D: depth 5,000   Q: quadratic   D+Q: depth 5,000 + quadratic'

for metric,label,direction in METRICS:
    fig,axes=plt.subplots(4,2,figsize=(7.2,9.0))
    fig.subplots_adjust(left=.10,right=.98,top=.865,bottom=.09,wspace=.27,hspace=.58)
    fig.text(.10,.977,label,fontsize=12,weight='bold',color=COLORS[0],va='top')
    fig.text(.10,.947,direction,fontsize=8,color=COLORS[3],va='top')
    legend(fig,metric,.926)
    for i,ax in enumerate(axes.flat):
        draw(ax,metric,[(i+1,c) for c in CONDS],SHORT.copy(),chr(65+i),f'Simulation {i+1}',i%2==0)
    fig.text(.10,.046,KEY,fontsize=7)
    fig.text(.10,.026,FOOT,fontsize=6.5)
    fig.text(.10,.009,'Scenario-specific y-scales. S1: Q and D+Q duplicate B and D; omitted by design.',fontsize=6.3)
    save(fig,'by_scenario/'+metric)
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.8))
    fig.subplots_adjust(left=.10,right=.98,top=.79,bottom=.14,wspace=.27,hspace=.48)
    fig.text(.10,.972,label,fontsize=12,weight='bold',color=COLORS[0],va='top')
    fig.text(.10,.93,direction,fontsize=8,color=COLORS[3],va='top')
    legend(fig,metric,.90)
    for i,ax in enumerate(axes.flat):
        draw(ax,metric,[(s,CONDS[i]) for s in range(1,9)], [f'S{s}' for s in range(1,9)],chr(65+i),CNAMES[i],i%2==0)
        ax.set_xlabel('Simulation scenario')
    fig.text(.10,.045,FOOT,fontsize=6.3)
    fig.text(.10,.021,'S1: quadratic conditions duplicate the linear conditions and were omitted by design.',fontsize=6.3)
    save(fig,'by_condition/'+metric)

# Overview: aggregate within seed first; never treat 30 correlated cells as replicates.
valid_cells=[cell for cell,v in seeds.items() if v]
complete_seeds=sorted(set.intersection(*(set(seeds[cell]) for cell in valid_cells)))
reduction=[]
for m in METHODS:
    for seed in complete_seeds:
        v=[100*(1-float(raw[s,c,seed,m]['comp_log_rmse'])/float(raw[s,c,seed,'naive']['comp_log_rmse'])) for s,c in valid_cells]
        reduction.append(dict(method=m,seed=seed,reduction_percent=float(np.mean(v)),n_cells=len(v)))
write_csv('panel_I_seed_reductions.csv',reduction)
fig,axes=plt.subplots(3,3,figsize=(7.2,7.7))
fig.subplots_adjust(left=.09,right=.985,top=.835,bottom=.15,wspace=.43,hspace=.55)
fig.text(.09,.975,'Recovery of simulated microbial composition',fontsize=11,weight='bold',color=COLORS[0],va='top')
fig.text(.09,.946,'A–H: Log-RMSE (lower is better)   |   I: error reduction from the observed–truth gap',fontsize=7.2,va='top')
legend(fig,'comp_log_rmse',.922)
for i,ax in enumerate(axes.flat[:8]):
    draw(ax,'comp_log_rmse',[(i+1,c) for c in CONDS],SHORT.copy(),chr(65+i),f'Simulation {i+1}',i%3==0)
ax=axes.flat[8]
for k,m in enumerate(METHODS):
    v=np.array([r['reduction_percent'] for r in reduction if r['method']==m])
    ax.barh(k,v.mean(),height=.58,color=COLORS[k],edgecolor='#565361',linewidth=.35,
            xerr=v.std(ddof=1)/np.sqrt(len(v)),error_kw={'elinewidth':.7,'capsize':1.5},zorder=3)
ax.set_yticks(range(5),['Ours','EMBED','microTensor','PARAFAC2','TEMPTED'],fontsize=6.4)
ax.invert_yaxis()
reduction_min = min(r['reduction_percent'] for r in reduction)
ax.set_xlim(min(0, reduction_min*1.15),100)
ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
ax.axvline(0,color='#76717D',lw=.65)
ax.set_xlabel('Log-RMSE reduction (%)',fontsize=7)
ax.set_title('Paired improvement',loc='left',fontsize=8,pad=8,weight='bold')
ax.text(-.16,1.06,'I',transform=ax.transAxes,fontsize=10,weight='bold',color=COLORS[0])
ax.set_axisbelow(True)
ax.grid(axis='x',color='#E9E7ED',lw=.45)
fig.text(.09,.094,KEY,fontsize=6.7)
fig.text(.09,.071,FOOT,fontsize=6.2)
fig.text(.09,.048,f'I: equal-weight average across 30 available cells within each seed; mean ± SE across {len(complete_seeds)} complete seeds.',fontsize=6.2)
fig.text(.09,.025,'A–H: scenario-specific y-scales. S1: quadratic conditions duplicate B / D; omitted by design.',fontsize=6.3,color='#62636A')
save(fig,'article/Figure_main_A_I')
manifest['panel_I']=dict(n=len(complete_seeds),seeds=complete_seeds,cells=len(valid_cells),formula='100*(1-method_RMSE/naive_RMSE), then equal-weight cell average within seed')
(a.out/'figure_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
sheet=Image.new('RGB',(1800,((len(previews)+3)//4)*560),'white')
pen=ImageDraw.Draw(sheet)
for i,(name,im) in enumerate(previews):
    im.thumbnail((435,525))
    x,y=(i%4)*450,(i//4)*560
    sheet.paste(im,(x,y))
    pen.text((x+8,y+533),name,fill='#4F587D')
sheet.save(a.out/'_qa'/'contact_sheet.png')
print('Verified 449 paired datasets; 19 figures in PNG / RGB LZW TIFF / vector PDF.',flush=True)
