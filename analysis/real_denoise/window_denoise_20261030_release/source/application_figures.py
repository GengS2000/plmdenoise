"""One self-contained application figure per cohort; no simulation panels."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from common import digest, write_json
import re

COLORS = dict(raw='#776B97', denoised='#4F587D', common='#4F587D',
              individual='#776B97', event='#C68DC0', baseline='#888888', total='#24242B')


def taxon_label(value):
    fields=[x.strip() for x in re.split('[|;]',str(value))]
    fields=[re.sub(r'^[a-z]__','',x) for x in fields]
    fields=[x for x in fields if x and x.lower() not in ['nan','unclassified','unknown','na']]
    return fields[-1].replace('_',' ') if fields else 'Unresolved taxon'


def draw(out, cohort, dpi=600, status='conditional analysis'):
    out = Path(out)
    read = lambda n: pd.read_csv(out/n, sep='\t')
    scores = read('ordination_scores.tsv')
    contrasts = read('taxon_contrasts.tsv')
    cov = read('component_covariance.tsv')
    trajectories = read('figure_trajectory.tsv')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(3,2,figsize=(11.8,12), layout='constrained')
    fig.suptitle(f'{cohort} | differential inference, ordination and decomposition',fontsize=14)
    shared = scores[scores.basis.eq('shared_raw_basis')]
    bounds = [(shared[c].min(),shared[c].max()) for c in ['PC1','PC2']]
    for ax, rep, letter in zip(axes[0], ['raw','denoised'], ['A','B']):
        d=shared[shared.representation.eq(rep)]
        sc=ax.scatter(d.PC1,d.PC2,c=d.time,s=9,cmap='viridis',alpha=.65,rasterized=True)
        for lim,setter in zip(bounds,[ax.set_xlim,ax.set_ylim]):
            pad=max((lim[1]-lim[0])*.05,.01);setter(lim[0]-pad,lim[1]+pad)
        ax.set(xlabel='PC1 (shared raw basis)',ylabel='PC2 (shared raw basis)',title=f'{letter}  {rep.capitalize()}')
        fig.colorbar(sc,ax=ax,label='Time (days)',shrink=.8)
    keys=['contrast','taxon_id']
    merged=contrasts[contrasts.representation.eq('raw')].merge(
        contrasts[contrasts.representation.eq('denoised')],on=keys,suffixes=('_raw','_den'))
    merged.to_csv(out/'figure_concordance.tsv',sep='\t',index=False)
    ax=axes[1,0]
    if len(merged):
        positive=merged.q_bh_den.lt(.05)&merged.q_bh_raw.ge(.05)
        ax.scatter(merged.mean_effect_raw,merged.mean_effect_den,s=10,
                   c=np.where(positive,'#C68DC0','#4F587D'),alpha=.6,rasterized=True)
        extent=max(np.abs(merged[['mean_effect_raw','mean_effect_den']]).to_numpy().max(),.01)
        ax.plot([-extent,extent],[-extent,extent],color='#aaaaaa',lw=.8)
    ax.axhline(0,color='#cccccc',lw=.5);ax.axvline(0,color='#cccccc',lw=.5)
    ax.set(title='C  All prespecified taxon contrasts',xlabel='Raw effect (CLR or CLR/year)',ylabel='Denoised effect (same unit)')
    ax.text(.02,.98,'Pink: q < 0.05 only after denoising',transform=ax.transAxes,va='top',fontsize=7)
    ax=axes[1,1]
    # Selection uses raw effect magnitude, never denoised significance.
    selected=merged.assign(rank=merged.mean_effect_raw.abs()).sort_values(['rank','taxon_id'],ascending=[False,True]).head(8)
    selected.to_csv(out/'figure_selected_contrasts.tsv',sep='\t',index=False)
    for i,(_,r) in enumerate(selected.iterrows()):
        for rep,off in [('raw',-.12),('den',.12)]:
            color=COLORS['raw' if rep=='raw' else 'denoised']
            ax.plot([r['ci_low_'+rep],r['ci_high_'+rep]],[i+off]*2,color=color,lw=1.4)
            ax.plot(r['mean_effect_'+rep],i+off,'o',color=color,ms=3)
    names=[str(r.get('name_raw',r.taxon_id))[:20]+' ['+str(r.taxon_id)[-5:]+']\n'+r.contrast.replace('within_subject_slope_per_year','age slope').replace('_minus_baseline',' vs baseline')[:30] for _,r in selected.iterrows()]
    ax.set_yticks(range(len(selected)),names,fontsize=6)
    ax.axvline(0,color='#aaaaaa',lw=.8)
    ax.set(title='D  Largest raw effects (illustrative)',xlabel='Effect and conditional 95% bootstrap interval')
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0],[0],color=COLORS[r],marker='o',label=r) for r in ['raw','denoised']],fontsize=7)
    names=['baseline','common','individual','event']
    matrix=cov.pivot(index='component_a',columns='component_b',values='fraction_of_total_variance').reindex(index=names,columns=names).to_numpy()
    ax=axes[2,0];v=max(abs(matrix).max(),.01)
    im=ax.imshow(matrix,cmap='PuOr_r',vmin=-v,vmax=v)
    ax.set_xticks(range(4),names,rotation=25);ax.set_yticks(range(4),names)
    for i in range(4):
        for j in range(4):ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=8)
    ax.set_title('E  Component covariance / total variance')
    fig.colorbar(im,ax=ax,shrink=.75,label='All 16 cells sum to 1')
    ax=axes[2,1]
    for name,g in trajectories.groupby('component',sort=False):
        ax.plot(g.time,g.value,label=name,color=COLORS.get(name,'#aaaaaa'),lw=1.2)
    if len(trajectories):
        r=trajectories.iloc[0]
        ax.set_title(f'F  {str(r["name"])[:30]} | {r.subject_id}',fontsize=9)
    ax.set(xlabel='Time (days)',ylabel='CLR; baseline retained in total');ax.legend(ncol=3,fontsize=7)
    fig.get_layout_engine().set(rect=(0,.035,1,.96))
    fig.text(.02,.012,status+' | PCA is descriptive; intervals condition on fitted representations.',fontsize=7)
    stem='Figure_'+cohort+'_application'
    fig.savefig(out/(stem+'_preview.png'),dpi=130)
    fig.savefig(out/(stem+'.pdf'),dpi=dpi)
    fig.savefig(out/(stem+'.png'),dpi=dpi)
    plt.close(fig)
    with Image.open(out/(stem+'.png')) as im:
        im.convert('RGB').save(out/(stem+'.tiff'),compression='tiff_lzw',dpi=(dpi,dpi))
    write_json(out/'figure_manifest.json',dict(cohort=cohort,dpi=dpi,status=status,
        files={p.name:digest(p) for p in out.glob(stem+'*')},
        source_tables={p.name:digest(p) for p in out.glob('*.tsv')},code_sha256=digest(__file__)))
    (out/'FIGURE_CAPTION.md').write_text(
        f'# {cohort}\n\nA–B: subject-weighted PCA projected onto the same raw basis and axis limits. '
        'C: all tested contrasts; pink means conditional BH q < .05 only after denoising, not validated new discoveries. '
        'D: eight largest absolute raw effects, with subject/block bootstrap intervals; full testing family retained. '
        'E: covariance includes both off-diagonal triangles; diagonal values alone are not an additive partition. '
        'F: feature selected by raw prevalence, subject by visit count, ties broken by ID. '
        'ECAM has no defined event: its zero event component is structural, not a null-effect finding.\n\n'
        +status+'\n',encoding='utf-8')


def trajectory(s,taxa,counts,comp,out):
    prevalence=pd.DataFrame(counts>0).groupby(s.subject_id).max().mean().to_numpy()
    j=sorted(range(len(taxa)),key=lambda j:(-prevalence[j],str(taxa.taxon_id.iloc[j])))[0]
    sid=sorted(s.subject_id.unique(),key=lambda x:(-sum(s.subject_id==x),str(x)))[0]
    ix=s.index[s.subject_id.eq(sid)];rows=[]
    for name,x in comp.items():
        if name not in COLORS:continue
        x=x-x.mean(-1,keepdims=True)
        for i in ix:rows.append(dict(subject_id=sid,taxon_id=taxa.taxon_id.iloc[j],name=taxa['name'].iloc[j],time=s.time.iloc[i],component=name,value=x[i,j]))
    pd.DataFrame(rows).sort_values(['component','time']).to_csv(out/'figure_trajectory.tsv',sep='\t',index=False)
