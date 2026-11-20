"""Reproducible K=1 decomposition/application figure (A-J).

Original-server simulation components and per-seed event posterior draws are
included. Ground truth is regenerated deterministically; no model is refitted.

Usage:
  python build_figure.py --dpi 600
  python build_figure.py --preview-only
  python build_figure.py --cached --dpi 600

The source_data directory is a self-contained numeric export of every plotted
quantity. --cached rebuilds the figure without reading the original repository.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import platform
from pathlib import Path
import shutil
import sys
import time
import warnings
import numpy as np
import pandas as pd
import scipy
from scipy import stats
from scipy.cluster.hierarchy import linkage, leaves_list
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from matplotlib.transforms import blended_transform_factory
from PIL import Image

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
DATA=HERE/'source_data'
QA=HERE/'_qa'
STEM='Figure_AJ_K1_decomposition_application_v7'
COLORS={'ink':'#4F587D','pink':'#C68DC0','ice':'#C2E0EE','purple':'#776B97','lavender':'#DBC8ED',
        'raw':'#776B97','smooth':'#81AFC6','denoised':'#4F587D',
        'common':'#4F587D','individual':'#776B97','event':'#C68DC0',
        'black':'#24242B','grey':'#B8BCC5','mask':'#E7E8EC'}
METHODS=['raw','smooth','denoised']
METHOD_LABELS={'raw':'Raw','smooth':'Smooth','denoised':'K = 1'}
SCENARIOS=[3,4,6,7,8]
SCENARIO_LABELS=['NB\ndecay','NB\nstep','ZINB\ndecay','ZINB\nstep','DM\nmixed']
COMPONENTS=['baseline','common','individual','event']
SOURCE_HASHES={}

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def record(path):
    path=Path(path)
    SOURCE_HASHES[str(path.relative_to(ROOT))]=digest(path)
    return path

def read_tsv(path,**kwargs):return pd.read_csv(record(path),sep='\t',**kwargs)
def centre(x):return x-x.mean(-1,keepdims=True)
def save_tsv(name,d):d.to_csv(DATA/name,sep='\t',index=False)
def load_tsv(name,**kwargs):return pd.read_csv(DATA/name,sep='\t',**kwargs)

def short_name(s):
    fields=str(s).split('|'); last=fields[-1]
    if '__' in last:return last.split('__',1)[1].replace('_',' ')
    return last.replace('_',' ')

def raw_bundle(root,cohort):
    base=root/'real_denoise/event_server_20261018_k1/data'/cohort
    s=read_tsv(base/'samples.tsv',dtype={'sample_id':str,'subject_id':str}).reset_index(drop=True)
    t=read_tsv(base/'taxa.tsv',dtype=str)
    with np.load(record(base/'input.npz'),allow_pickle=False) as f:y=f['Y'][s.tensor_i,s.tensor_t].copy()
    with np.load(record(base/'events.npz'),allow_pickle=False) as f:events=f['event_times'].copy()
    with np.load(record(root/'实际数据分析的结果/K=1/analysis'/cohort/'ensemble_components.npz'),allow_pickle=False) as f:
        if not np.array_equal(f['sample_id'],s.sample_id.to_numpy(str)):raise ValueError(cohort+' sample ID mismatch')
        if not np.array_equal(f['taxon_id'],t.taxon_id.to_numpy(str)):raise ValueError(cohort+' taxon ID mismatch')
        comp={k:centre(f[k+'_mean'].astype(float)) for k in COMPONENTS+['total']}
    if np.max(np.abs(sum(comp[k] for k in COMPONENTS)-comp['total']))>1e-5:raise ValueError('Component sum mismatch')
    event_day=np.array([events[i,0] for i in s.tensor_i])
    detected=pd.DataFrame(y>0).groupby(s.subject_id).max().mean(0).to_numpy()
    sample_detection=(y>0).mean(0)
    return s,t,y,events,event_day,comp,detected,sample_detection

def refine_g_example(bundle):
    """Post-hoc illustrative second subject; never an accuracy/stability test."""
    s,t,y,events,ed,comp,prev,sdet=bundle
    rule=json.loads((DATA/'G_selection_rule.json').read_text(encoding='utf-8'))
    tid=rule['taxon_id'];j=int(t.index[t.taxon_id.eq(tid)][0])
    first='E020924';old_second='E016063'
    first_event=float(ed[s.index[s.subject_id.eq(first)][0]])
    rows=[]
    for sid,g in s.groupby('subject_id',sort=True):
        ix=g.index.to_numpy();tt=g.time.to_numpy();event=float(ed[ix[0]])
        c=comp['common'][ix,j];u=comp['individual'][ix,j]
        corr=float(np.corrcoef(c,u)[0,1]) if min(c.std(),u.std())>1e-8 else np.nan
        rows.append(dict(subject_id=sid,event_day=event,n_pre=int((tt<event).sum()),
            n_post=int((tt>=event).sum()),detection=float((y[ix,j]>0).mean()),
            common_individual_temporal_r=corr,abs_temporal_r=abs(corr),individual_sd=float(u.std())))
    candidates=pd.DataFrame(rows)
    candidates['eligible']=(candidates.n_pre.ge(2)&candidates.n_post.ge(5)&candidates.detection.ge(.2)&
        candidates.event_day.sub(first_event).abs().ge(30)&candidates.individual_sd.ge(.15)&
        candidates.abs_temporal_r.notna())
    ranked=candidates[candidates.eligible].sort_values(['abs_temporal_r','subject_id'])
    if ranked.empty:raise ValueError('No eligible illustrative G replacement')
    second=ranked.iloc[0].subject_id
    candidates['selected']=candidates.subject_id.isin([first,second])
    save_tsv('G_candidate_screen.tsv',candidates.sort_values(['eligible','abs_temporal_r'],ascending=[False,True]))
    selected=candidates.set_index('subject_id').loc[[first,second]].reset_index()
    grows=[]
    for _,q in selected.iterrows():
        ix=s.index[s.subject_id.eq(q.subject_id)].to_numpy()
        v=s.loc[ix,['sample_id','subject_id','time']].copy()
        v['event_day']=q.event_day;v['taxon_id']=tid;v['name']=rule['name']
        v['raw']=centre(np.log(y+.5))[ix,j]
        for k in COMPONENTS+['total']:v[k]=comp[k][ix,j]
        grows.append(v)
    save_tsv('G_subject_component_trajectories.tsv',pd.concat(grows,ignore_index=True))
    save_tsv('G_subject_selection.tsv',selected)
    rule.update(rule='First subject E020924 retained. Second subject chosen post hoc among subjects with >=2 pre-event and >=5 post-event visits, raw detection >=20%, event date >=30 days apart from first, and individual-component temporal SD >=0.15 CLR; minimise absolute Pearson correlation between common and individual posterior-mean trajectories at observed visits, breaking ties by ID.',
        purpose='Illustrative separation of shapes; selection on fitted components is not evidence of accuracy, orthogonality, posterior independence or general representativeness.',
        old_second=old_second,new_second=second,
        old_temporal_r=float(candidates.set_index('subject_id').loc[old_second,'common_individual_temporal_r']),
        new_temporal_r=float(ranked.iloc[0].common_individual_temporal_r))
    (DATA/'G_selection_rule.json').write_text(json.dumps(rule,ensure_ascii=False,indent=2),encoding='utf-8')

def prepare_data(root):
    DATA.mkdir(exist_ok=True,parents=True)
    sim=root/'simulation_results/三个量分解的结果二'
    rows=[json.loads(record(p).read_text(encoding='utf-8'))['row'] for p in sorted((sim/'tasks').glob('*.json'))]
    m=pd.DataFrame(rows)
    needed=['scenario','seed','condition']+[f'struct_{k}_r2_m4' for k in ['common','individual','event']]
    missing=[k for k in needed if k not in m]
    if missing:raise ValueError('Missing simulation fields: '+str(missing))
    long=m[m.scenario.isin(SCENARIOS)].groupby(['scenario','seed'],as_index=False)[needed[3:]].mean()
    save_tsv('B_simulation_seed_R2.tsv',long)
    heat=[]
    for scenario in [3,6]:
        for estimate in COMPONENTS:
            for truth in COMPONENTS:
                col=f'leak_m4_{estimate}_from_{truth}'
                seed=m[m.scenario.eq(scenario)].groupby('seed')[col].mean()
                heat.append(dict(scenario=scenario,estimated=estimate,truth=truth,mean=seed.mean(),se=seed.std(ddof=1)/np.sqrt(len(seed)),n_seeds=len(seed)))
    save_tsv('C_leakage_coefficients.tsv',pd.DataFrame(heat))

    real=root/'实际数据分析的结果/K=1/analysis'
    app=root/'outputs/k1_real_evaluation_20261019'
    bundles={c:raw_bundle(root,c) for c in ['DIABIMMUNE','FARMM','ECAM']}
    # D: existing subject-weighted projections, plus cumulative variance from all PCs.
    dbase=app/'DIABIMMUNE'
    scores=read_tsv(dbase/'ordination_scores.tsv',dtype={'subject_id':str})
    scores=scores[scores.basis.eq('shared_raw_basis')].copy()
    save_tsv('D_PCA_shared_raw_basis.tsv',scores)
    diag=read_tsv(dbase/'ordination_diagnostics.tsv');save_tsv('D_PCA_diagnostics.tsv',diag)
    s,t,y,events,ed,comp,prev,sdet=bundles['DIABIMMUNE']
    kept=read_tsv(dbase/'taxa.tsv',dtype=str).taxon_id
    ix=t.set_index('taxon_id').index.get_indexer(kept)
    if (ix<0).any():raise ValueError('Missing PCA taxa')
    raw=centre(np.log(y[:,ix]+.5));smooth=np.empty_like(raw)
    for _,g in s.groupby('subject_id',sort=False):
        ii=g.index.to_numpy();tt=g.time.to_numpy(float)
        w=np.exp(-.5*((tt[:,None]-tt[None,:])/30.4375)**2);w[np.abs(tt[:,None]-tt[None,:])>3*30.4375]=0
        smooth[ii]=(w/w.sum(1,keepdims=True))@raw[ii]
    reps=dict(raw=raw,smooth=smooth,denoised=centre(comp['total'][:,ix]))
    weights=1/s.groupby('subject_id').subject_id.transform('size').to_numpy(float);weights/=weights.sum()
    vv=[]
    for rep,x in reps.items():
        mu=np.average(x,axis=0,weights=weights)
        sv=np.linalg.svd((x-mu)*np.sqrt(weights[:,None]),compute_uv=False)
        frac=sv**2/(sv@sv)
        for k in range(min(30,len(frac))):vv.append(dict(representation=rep,PC=k+1,fraction=frac[k],cumulative=frac[:k+1].sum()))
    save_tsv('D_cumulative_PCA_variance.tsv',pd.DataFrame(vv))

    # E: selection by raw support and supported-window mean effect, then shape clustering.
    selection=[]
    for cohort in ['FARMM','DIABIMMUNE']:
        s,t,y,events,ed,comp,prev,sdet=bundles[cohort]
        taxa=read_tsv(real/cohort/'event_taxa.tsv')
        curves=read_tsv(real/cohort/'event_curves.tsv')
        look=pd.DataFrame(dict(taxon_id=t.taxon_id,subject_prevalence=prev,sample_detection=sdet))
        taxa=taxa.merge(look,on='taxon_id',validate='one_to_one')
        eligible=taxa[(taxa.subject_prevalence>=.30)&taxa.status.eq('tested')&taxa.mean_supported_effect.abs().ge(.15)].copy()
        eligible['abs_effect']=eligible.mean_supported_effect.abs()
        chosen=eligible.sort_values(['abs_effect','taxon_id'],ascending=[False,True]).head(16)
        if len(chosen)<6:raise ValueError('Too few event heatmap taxa: '+cohort)
        sup=curves[curves.supported].relative_day.unique()
        mat=curves[curves.taxon_id.isin(chosen.taxon_id)&curves.supported].pivot(index='taxon_id',columns='relative_day',values='clr_effect_mean').loc[chosen.taxon_id]
        z=mat.to_numpy();z=(z-z.mean(1,keepdims=True))/np.maximum(z.std(1,keepdims=True),1e-8)
        order=leaves_list(linkage(z,method='average',metric='euclidean',optimal_ordering=True))
        chosen=chosen.iloc[order].copy();chosen['row_order']=np.arange(len(chosen));chosen['cohort']=cohort
        selection.append(chosen)
        sub=curves[curves.taxon_id.isin(chosen.taxon_id)].merge(chosen[['taxon_id','row_order','name','subject_prevalence','sample_detection']],on='taxon_id',validate='many_to_one')
        save_tsv('E_'+cohort+'_heatmap.tsv',sub)
        save_tsv('E_'+cohort+'_sample_times.tsv',pd.DataFrame(dict(relative_day=s.time.to_numpy()-ed)))
    save_tsv('E_selected_event_taxa.tsv',pd.concat(selection,ignore_index=True))

    # G: same taxon across two well-observed subjects with different food-introduction dates.
    s,t,y,events,ed,comp,prev,sdet=bundles['DIABIMMUNE']
    taxa=read_tsv(real/'DIABIMMUNE/event_taxa.tsv')
    tm=t.copy();tm['j']=np.arange(len(t));tm['prevalence']=prev;tm['detection']=sdet
    tm=tm.merge(taxa[['taxon_id','name','discovery','mean_supported_effect']],on='taxon_id')
    # A priori genus preference; adequate measured detection takes precedence over fit agreement.
    preferred=['Ruminococcus','Roseburia','Alistipes','Butyrivibrio','Bacteroides']
    available=tm[tm.discovery&tm['name'].isin(preferred)&tm.prevalence.ge(.5)&tm.detection.ge(.25)].copy()
    if available.empty:available=tm[tm.discovery&tm.prevalence.ge(.5)].copy()
    available['priority']=available['name'].map({v:i for i,v in enumerate(preferred)}).fillna(99)
    available=available.sort_values(['priority','detection','taxon_id'],ascending=[True,False,True])
    chosen=available.iloc[0];j=int(chosen.j)
    subjects=[]
    for sid,g in s.groupby('subject_id',sort=True):
        idx=g.index.to_numpy();event=float(ed[idx[0]])
        pre=(g.time<event).sum();post=(g.time>=event).sum()
        if pre>=2 and post>=5 and (y[idx,j]>0).mean()>=.20:
            subjects.append(dict(subject_id=sid,event_day=event,n_pre=int(pre),n_post=int(post),detection=float((y[idx,j]>0).mean())))
    subjects=pd.DataFrame(subjects)
    if len(subjects)<2:raise ValueError('Insufficient G subjects')
    med=float(np.median(events[:,0]));low=subjects[subjects.event_day<med].sort_values(['event_day','subject_id'])
    hi=subjects[subjects.event_day>med].sort_values(['event_day','subject_id'],ascending=[False,True])
    if low.empty or hi.empty:raise ValueError('No distinct event-date strata')
    selected=pd.concat([low.head(1),hi.head(1)],ignore_index=True)
    grows=[]
    for _,q in selected.iterrows():
        ids=s.index[s.subject_id.eq(q.subject_id)].to_numpy();v=s.loc[ids,['sample_id','subject_id','time']].copy()
        v['event_day']=q.event_day;v['taxon_id']=chosen.taxon_id;v['name']=chosen['name'];v['raw']=centre(np.log(y+.5))[ids,j]
        for k in COMPONENTS+['total']:v[k]=comp[k][ids,j]
        grows.append(v)
    save_tsv('G_subject_component_trajectories.tsv',pd.concat(grows,ignore_index=True));save_tsv('G_subject_selection.tsv',selected)
    (DATA/'G_selection_rule.json').write_text(json.dumps(dict(taxon_id=chosen.taxon_id,name=chosen['name'],rule='Prespecified genus preference; raw subject prevalence >=50% and sample detection >=25%; subjects with >=2 pre and >=5 post visits, >=20% measured detection, earliest and latest eligible event-date strata. No criterion based on fit accuracy.'),ensure_ascii=False,indent=2),encoding='utf-8')

    refine_g_example(bundles['DIABIMMUNE'])

    # H: existing conditional subject-level permutation result, and subject bootstrap means.
    h=read_tsv(real/'FARMM/subject_outcomes.tsv',dtype={'subject_id':str});save_tsv('H_subject_event_RMS.tsv',h)
    tests=read_tsv(real/'FARMM/subject_associations.tsv');save_tsv('H_subject_tests.tsv',tests)

    # I/J: complete precomputed subject-level results, and raw support re-exported from counts.
    allc=[]
    for cohort in ['ECAM','FARMM']:
        c=read_tsv(app/cohort/'taxon_contrasts.tsv');c['cohort']=cohort;allc.append(c)
    contrasts=pd.concat(allc,ignore_index=True);save_tsv('IJ_taxon_contrasts.tsv',contrasts)
    selected=[];support=[]
    for cohort,name,contrast in [('ECAM','Streptococcus','within_subject_slope_per_year'),('ECAM','Phascolarctobacterium','within_subject_slope_per_year'),('FARMM','Oxalobacter_formigenes','early_recovery_minus_baseline'),('FARMM','Serratia_marcescens','depletion_minus_baseline')]:
        c=contrasts[(contrasts.cohort==cohort)&contrasts['name'].eq(name)&contrasts.contrast.eq(contrast)]
        ids=c.taxon_id.unique()
        if len(ids)!=1:raise ValueError('Candidate must have unique taxon ID: '+name)
        tid=ids[0];selected.append(dict(cohort=cohort,name=name,taxon_id=tid,contrast=contrast))
        s,t,y,events,ed,comp,prev,sdet=bundles[cohort];j=int(t.index[t.taxon_id.eq(tid)][0])
        phases=read_tsv(app/cohort/'samples.tsv',dtype={'sample_id':str,'subject_id':str})
        if not np.array_equal(phases.sample_id,s.sample_id):raise ValueError('Phase sample order mismatch')
        phases['count']=y[:,j];phases['relative']=y[:,j]/np.maximum(y.sum(1),1)
        phases['taxon_id']=tid;phases['name']=name;phases['cohort']=cohort
        support.append(phases[['cohort','name','taxon_id','subject_id','sample_id','time','phase','count','relative']])
    save_tsv('IJ_selected_taxa.tsv',pd.DataFrame(selected));save_tsv('IJ_raw_support_samples.tsv',pd.concat(support,ignore_index=True))
    provenance=dict(created=time.strftime('%Y-%m-%d %H:%M:%S'),source_hashes=SOURCE_HASHES,
                    palette=COLORS,deferred_panels=['A','F'],original_plan='A-J',
                    statistical_unit_simulation='15 independent data seeds; conditions averaged within seed',
                    real='Conditional on supplied K=1 pooled ensemble; no refitting; DIABIMMUNE has seeds 2026/2027 only',
                    selection_E='16 largest absolute supported-window mean effects among raw subject prevalence >=30%, |effect|>=0.15; cluster row-standardized supported curves for ordering only; plot absolute CLR effects',
                    source_code='build_figure.py',python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__,scipy=scipy.__version__)
    (DATA/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')

def style():
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':8.2,'axes.titlesize':8.5,'axes.labelsize':8.0,'xtick.labelsize':7.2,'ytick.labelsize':7.2,
        'axes.linewidth':.55,'lines.linewidth':1.1,'xtick.major.width':.5,'ytick.major.width':.5,
        'xtick.major.size':2.5,'ytick.major.size':2.5,'legend.fontsize':7.2,'legend.frameon':False,
        'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white','figure.facecolor':'white',
        'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':True,
        'text.color':COLORS['black'],'axes.labelcolor':COLORS['black'],'xtick.color':COLORS['black'],'ytick.color':COLORS['black']})

def title(fig,gs,letter,text):
    box=gs.get_position(fig)
    fig.text(box.x0,box.y1+.012,letter,fontsize=14,fontweight='bold',ha='left',va='bottom')
    fig.text(box.x0+.024,box.y1+.013,text,fontsize=10,fontweight='bold',ha='left',va='bottom')

def qtext(q):return f'{q:.2g}' if q>=.001 else f'{q:.1e}'.replace('e-0','e−')
def tidy(ax):ax.tick_params(pad=2);ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

def panel_b(fig,gs):
    title(fig,gs,'B','Recovery of labelled components across simulation seeds')
    sub=gs.subgridspec(1,3,wspace=.32);d=load_tsv('B_simulation_seed_R2.tsv');rng=np.random.default_rng(3241)
    cs=['common','individual','event'];labels=['Population trend','Individual dynamics','Event response']
    for k,(c,label) in enumerate(zip(cs,labels)):
        ax=fig.add_subplot(sub[k]);vals=[]
        for x,s in enumerate(SCENARIOS):
            v=d.loc[d.scenario.eq(s),f'struct_{c}_r2_m4'].dropna().to_numpy();vals.extend(v)
            ax.scatter(x+rng.uniform(-.16,.16,len(v)),v,s=12,color=COLORS[c],alpha=.58,linewidths=0,zorder=2)
            mean=v.mean();ci=stats.t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v))
            ax.errorbar(x,mean,yerr=ci,color=COLORS['black'],fmt='_',markersize=11,elinewidth=1.3,capsize=3,zorder=4)
        ax.axhline(0,color=COLORS['grey'],lw=.6,zorder=0);ax.axvspan(3.5,4.5,color=COLORS['mask'],alpha=.6,zorder=-1)
        ax.set_xticks(range(5),SCENARIO_LABELS);ax.set_xlim(-.45,4.45);ax.set_title(label,color=COLORS[c],fontweight='bold',pad=6)
        vmin=min(vals);span=max(vals)-vmin
        ax.set_ylim(min(-.04,vmin-.08*span),1.045)
        if k==0:ax.set_ylabel('Recovery $R^2$')
        if k==1:
            ax.legend(handles=[Line2D([],[],marker='o',linestyle='none',color=COLORS[c],ms=3,label='Independent seed'),Line2D([],[],marker='_',linestyle='none',color=COLORS['black'],ms=9,label='Mean ± 95% CI')],loc='lower left',fontsize=6.4,handlelength=1.2)
            ax.text(.05,.18,'n = 15 seeds / scenario',transform=ax.transAxes,fontsize=6.4,color='#555963')
        tidy(ax)

def panel_c(fig,gs):
    title(fig,gs,'C','Where each generated component is recovered')
    sub=gs.subgridspec(1,3,width_ratios=[1,1,.055],wspace=.35)
    d=load_tsv('C_leakage_coefficients.tsv');cmap=LinearSegmentedColormap.from_list('leak',[COLORS['pink'],'#FFFFFF',COLORS['ink']])
    norm=TwoSlopeNorm(vmin=-1,vcenter=0,vmax=1)
    for q,s in enumerate([3,6]):
        ax=fig.add_subplot(sub[q]);a=np.zeros((4,4))
        for i,e in enumerate(COMPONENTS):
            for j,t in enumerate(COMPONENTS):a[i,j]=d[(d.scenario==s)&(d.estimated==e)&(d.truth==t)]['mean'].iloc[0]
        im=ax.imshow(a,cmap=cmap,norm=norm,aspect='equal')
        for i in range(4):
            for j in range(4):
                value=a[i,j];label=f'{value:.3f}' if abs(value)<.01 else f'{value:.2f}'
                if abs(value)<.0005:label='0.000'
                ax.text(j,i,label,ha='center',va='center',fontsize=6.8,color='white' if abs(value)>.65 else COLORS['black'])
        ax.set_xticks(range(4),['Base.','Trend','Indiv.','Event'],rotation=40,ha='right');ax.set_yticks(range(4),['Base.','Trend','Indiv.','Event'])
        ax.set_xlabel('Generated component');ax.set_ylabel('Estimated component' if q==0 else '')
        ax.set_title('NB decay' if s==3 else 'ZINB decay',pad=7)
        ax.tick_params(length=0);ax.spines[['left','bottom']].set_visible(False)
    cax=fig.add_subplot(sub[2]);cb=fig.colorbar(im,cax=cax,ticks=[-1,0,1]);cb.set_label('Regression coefficient',fontsize=7.1,labelpad=3)

def panel_d(fig,gs):
    title(fig,gs,'D','DIABIMMUNE: age-associated structure and technical association')
    displayed=['raw','denoised']
    sub=gs.subgridspec(1,4,width_ratios=[1,1,1.10,1.10],wspace=.40)
    d=load_tsv('D_PCA_shared_raw_basis.tsv');cmap=LinearSegmentedColormap.from_list('age',[COLORS['ice'],COLORS['pink'],COLORS['ink']])
    d=d[d.representation.isin(displayed)]
    allx=d.PC1.to_numpy();ally=d.PC2.to_numpy();dx=np.ptp(allx);dy=np.ptp(ally)
    scatter_axes=[]
    for i,rep in enumerate(displayed):
        ax=fig.add_subplot(sub[i]);g=d[d.representation.eq(rep)]
        im=ax.scatter(g.PC1,g.PC2,c=g.time/365.25,cmap=cmap,vmin=0,vmax=d.time.max()/365.25,s=5,alpha=.8,linewidths=0,rasterized=True)
        ax.set_xlim(allx.min()-.04*dx,allx.max()+.04*dx);ax.set_ylim(ally.min()-.04*dy,ally.max()+.04*dy)
        ax.set_title(METHOD_LABELS[rep],fontweight='bold',pad=7);ax.set_xlabel('PC1 (raw basis)')
        if i==0:ax.set_ylabel('PC2 (raw basis)')
        else:ax.set_yticklabels([])
        ax.xaxis.set_major_locator(MaxNLocator(3));ax.yaxis.set_major_locator(MaxNLocator(3));scatter_axes.append(ax)
    # Compact age colorbar inside the first panel, without rescaling PCA coordinates.
    cax=scatter_axes[0].inset_axes([.08,.91,.55,.035]);cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[0,1,2,3])
    cb.ax.tick_params(labelsize=5.5,length=1,pad=1);cb.outline.set_linewidth(.35);cax.set_title('Age (yr)',fontsize=6,pad=1)
    ax=fig.add_subplot(sub[2]);v=load_tsv('D_cumulative_PCA_variance.tsv')
    for rep in displayed:
        g=v[v.representation.eq(rep)&v.PC.le(10)];ax.plot(g.PC,g.cumulative,color=COLORS[rep],label=METHOD_LABELS[rep],lw=1.4)
    ax.set(xlabel='Number of PCs',ylabel='Cumulative variance fraction',xlim=(1,10),ylim=(0,1));ax.set_xticks([1,2,5,10]);ax.legend(loc='lower right',fontsize=6.6);ax.set_title('Low-dimensional concentration',fontsize=8);tidy(ax)
    ax=fig.add_subplot(sub[3]);m=load_tsv('D_PCA_diagnostics.tsv')
    for metric,label,color in [('PC1_spearman_time','Age',COLORS['ink']),('PC1_spearman_depth','Depth',COLORS['pink'])]:
        values=[abs(m[(m.representation.eq(rep))&(m.metric.eq(metric))].value.iloc[0]) for rep in displayed]
        ax.plot(range(2),values,'o-',color=color,ms=3.5,lw=1.25,label=label)
        for x,y in enumerate(values):ax.annotate(f'{y:.3f}',(x,y),xytext=(0,7 if metric.endswith('time') else -11),textcoords='offset points',ha='center',fontsize=6.4,color=color)
    ax.set_xticks(range(2),['Raw','K = 1']);ax.set_xlim(-.12,1.12);ax.set_ylim(0,1.02);ax.set_ylabel('|Spearman correlation with PC1|');ax.set_title('Biological / technical association',fontsize=8);ax.legend(loc='center right',fontsize=6.8);tidy(ax)

def panel_e(fig,gs):
    title(fig,gs,'E','Event-associated taxon responses')
    sub=gs.subgridspec(1,2,wspace=.35);frames={c:load_tsv('E_'+c+'_heatmap.tsv') for c in ['FARMM','DIABIMMUNE']}
    values=np.concatenate([v.clr_effect_mean.to_numpy() for v in frames.values()]);limit=max(1.,float(np.ceil(np.quantile(np.abs(values),.98)*2)/2))
    cmap=LinearSegmentedColormap.from_list('event_effect',[COLORS['ink'],COLORS['ice'],'#FFFFFF',COLORS['lavender'],COLORS['pink']])
    prevmap=LinearSegmentedColormap.from_list('detection',['#FFFFFF',COLORS['ice'],COLORS['purple']])
    for k,(cohort,d) in enumerate(frames.items()):
        inner=sub[k].subgridspec(1,5,width_ratios=[.52,1,.045,.075,.045],wspace=.07)
        ax=fig.add_subplot(inner[1]);prevax=fig.add_subplot(inner[2]);cax=fig.add_subplot(inner[4])
        names=d[['taxon_id','name','row_order','sample_detection']].drop_duplicates().sort_values('row_order');tids=names.taxon_id
        mat=d.pivot(index='taxon_id',columns='relative_day',values='clr_effect_mean').loc[tids];grid=mat.columns.to_numpy(float)
        support=d.groupby('relative_day').supported.first().reindex(grid).to_numpy(bool)
        # Crop unsupported outer margins; retain masking for any internal gaps.
        supported_indices=np.flatnonzero(support)
        if not len(supported_indices):raise ValueError('No supported event window: '+cohort)
        window=slice(supported_indices[0],supported_indices[-1]+1)
        mat=mat.iloc[:,window];grid=grid[window];support=support[window]
        z=np.ma.array(mat.to_numpy(),mask=np.broadcast_to(~support,mat.shape));cm=cmap.copy();cm.set_bad(COLORS['mask'])
        # Cell edges preserve actual day spacing within the supported window.
        edges=np.r_[grid[0]-(grid[1]-grid[0])/2,(grid[:-1]+grid[1:])/2,grid[-1]+(grid[-1]-grid[-2])/2]
        im=ax.pcolormesh(edges,np.arange(len(tids)+1)-.5,z,cmap=cm,vmin=-limit,vmax=limit,rasterized=True,shading='flat')
        ax.set_ylim(len(tids)-.5,-.5);ax.set_yticks(np.arange(len(tids)),[str(x).replace('_',' ') for x in names['name']],fontsize=6.6)
        ax.tick_params(axis='y',length=0,pad=3);ax.set_xlabel('Days after event (supported window)')
        ax.set_title(cohort+' · '+('antibiotic / PEG' if cohort=='FARMM' else 'solid-food introduction'),fontweight='bold',pad=7)
        ax.xaxis.set_major_locator(MaxNLocator(5));ax.spines[['left','bottom']].set_visible(False)
        prevax.imshow(names.sample_detection.to_numpy()[:,None],aspect='auto',cmap=prevmap,vmin=0,vmax=1,extent=[0,1,len(tids)-.5,-.5]);prevax.set_xticks([]);prevax.set_yticks([])
        prevax.spines[['left','bottom']].set_visible(False);prevax.set_title('Raw\ndetect.\n0–1',fontsize=6.0,pad=7)
        cb=fig.colorbar(im,cax=cax,extend='both');cb.set_label('CLR event effect',fontsize=7.2);cb.ax.tick_params(labelsize=6.5)

def panel_g(fig,gs):
    title(fig,gs,'G','DIABIMMUNE: within-subject decomposition (Ruminococcus)')
    sub=gs.subgridspec(4,2,hspace=.25,wspace=.27,height_ratios=[1.25,1,1,1]);d=load_tsv('G_subject_component_trajectories.tsv',dtype={'subject_id':str})
    axes=[]
    for col,(sid,q) in enumerate(d.groupby('subject_id',sort=False)):
        q=q.sort_values('time');t=q.time.to_numpy();event=q.event_day.iloc[0]
        for row,key in enumerate(['total','common','individual','event']):
            ax=fig.add_subplot(sub[row,col]);axes.append((ax,row,key,q))
            if row==0:
                ax.scatter(t,q.raw,s=8,color=COLORS['grey'],alpha=.9,label='Raw',zorder=2)
                ax.plot(t,q.total,color=COLORS['ink'],lw=1.3,label='Total mean',zorder=3)
                ax.set_title(f'Ruminococcus · Subject {col+1}\nFood introduction: day {event:.0f}',fontsize=8,fontweight='bold',pad=5)
                if col==0:ax.legend(loc='lower right',ncol=2,fontsize=6,handlelength=1.4,columnspacing=.7)
            else:
                ax.axhline(0,color=COLORS['grey'],lw=.5);ax.plot(t,q[key],color=COLORS[key],lw=1.25)
                if key=='event':ax.axvspan(event+180,d.time.max()+15,color=COLORS['mask'],alpha=.7,zorder=-1)
            ax.axvline(event,color=COLORS['pink'],lw=.9,ls='--')
            if col==0:ax.set_ylabel({'total':'Total CLR','common':'Trend CLR','individual':'Individual CLR','event':'Event CLR'}[key],fontsize=7)
            ax.yaxis.set_major_locator(MaxNLocator(3));ax.xaxis.set_major_locator(MaxNLocator(4))
            if row==3:ax.set_xlabel('Age (days)')
            else:ax.tick_params(labelbottom=False)
            ax.set_xlim(d.time.min()-15,d.time.max()+15);ax.tick_params(labelsize=6.5,pad=1)
    # Matching y-axis scales for each component across the two subjects.
    for row in range(4):
        aa=[x[0] for x in axes if x[1]==row];lims=[a.get_ylim() for a in aa];lo=min(x[0] for x in lims);hi=max(x[1] for x in lims)
        for a in aa:a.set_ylim(lo,hi)

def panel_h(fig,gs):
    title(fig,gs,'H','FARMM event-response heterogeneity')
    ax=fig.add_subplot(gs);d=load_tsv('H_subject_event_RMS.tsv');tests=load_tsv('H_subject_tests.tsv')
    groups=['EEN','Omnivore','Vegan'];rng=np.random.default_rng(20261019);summary=[]
    for x,(group,color) in enumerate(zip(groups,[COLORS['pink'],COLORS['ink'],COLORS['purple']])):
        v=d[d.group.eq(group)].event_response_rms.to_numpy()
        means=v[rng.integers(len(v),size=(5000,len(v)))].mean(1);lo,hi=np.quantile(means,[.025,.975]);mu=v.mean()
        ax.scatter(x+rng.uniform(-.14,.14,len(v)),v,s=25,facecolor=color,edgecolor='white',lw=.4,alpha=.9,zorder=3)
        ax.errorbar(x,mu,yerr=[[mu-lo],[hi-mu]],fmt='_',color=COLORS['black'],capsize=4,ms=17,lw=1.5,zorder=5)
        summary.append(dict(group=group,n=len(v),mean=mu,lower=lo,upper=hi))
    q=tests[(tests.outcome.eq('event_response_rms'))&(tests.covariate.eq('group'))].q_bh.iloc[0]
    ax.set_xticks(range(3),[g+'\n'+f'n = {len(d[d.group.eq(g)])}' for g in groups]);ax.set_xlim(-.5,2.5)
    ax.set_ylabel('Event-response RMS (CLR)');ax.set_ylim(d.event_response_rms.min()-.03,d.event_response_rms.max()+.055)
    ax.text(.5,.96,f'Omnibus permutation BH q = {q:.4f}',ha='center',va='top',transform=ax.transAxes,fontsize=8)
    ax.text(.5,.87,'Dots: subjects\nBlack: mean ± bootstrap 95% CI',ha='center',va='top',transform=ax.transAxes,fontsize=7,color='#555963')
    ax.yaxis.set_major_locator(MaxNLocator(5));save_tsv('H_group_bootstrap_summary.tsv',pd.DataFrame(summary))

def forest(ax,table,cohort,name,contrast,methods=METHODS):
    d=table[(table.cohort.eq(cohort))&table['name'].eq(name)&table.contrast.eq(contrast)]
    if len(d)!=3:raise ValueError('Forest requires exactly one row per representation: '+name)
    lo=d.ci_low.min();hi=d.ci_high.max();span=max(hi-lo,.2)
    ax.axvline(0,color=COLORS['grey'],lw=.65)
    for k,rep in enumerate(methods):
        r=d[d.representation.eq(rep)].iloc[0];y=len(methods)-k-1
        ax.errorbar(r.mean_effect,y,xerr=[[r.mean_effect-r.ci_low],[r.ci_high-r.mean_effect]],fmt='o',color=COLORS[rep],capsize=2.5,ms=3.5,lw=1.1)
        ax.text(.985,y,'q = '+qtext(r.q_bh),transform=blended_transform_factory(ax.transAxes,ax.transData),ha='right',va='center',fontsize=6.6,color=COLORS[rep])
    ax.set_yticks(range(len(methods)),[METHOD_LABELS[r] for r in methods[::-1]],fontsize=7)
    ax.set_ylim(-.55,len(methods)-.25);ax.set_xlim(min(lo-.08*span,-.06*span),hi+.65*span)
    ax.xaxis.set_major_locator(MaxNLocator(4));ax.tick_params(axis='y',length=0);ax.set_xlabel('Age slope (CLR yr$^{-1}$)' if cohort=='ECAM' else 'Stage − baseline (CLR)',fontsize=7)
    ax.spines['left'].set_visible(False)

def panel_i(fig,gs):
    title(fig,gs,'I','Newly significant, biologically supported downstream associations')
    sub=gs.subgridspec(3,3,height_ratios=[.10,1.35,1],hspace=.95,wspace=.38)
    contrasts=load_tsv('IJ_taxon_contrasts.tsv');raw=load_tsv('IJ_raw_support_samples.tsv',dtype={'subject_id':str})
    rng=np.random.default_rng(20261020);support_export=[]
    choices=[('ECAM','Streptococcus','within_subject_slope_per_year'),('ECAM','Phascolarctobacterium','within_subject_slope_per_year'),('FARMM','Oxalobacter_formigenes','early_recovery_minus_baseline')]
    for col,(cohort,name,contrast) in enumerate(choices):
        ax=fig.add_subplot(sub[1,col]);forest(ax,contrasts,cohort,name,contrast,methods=['raw','denoised'])
        label=name.replace('_',' ')
        if name=='Oxalobacter_formigenes':label='Oxalobacter formigenes'
        ax.set_title(cohort+'\n'+label,fontsize=8,fontstyle='italic',pad=6)
        ax=fig.add_subplot(sub[2,col]);d=raw[(raw.cohort.eq(cohort))&raw['name'].eq(name)]
        phases=['0_6m','6_12m','12_24m'] if cohort=='ECAM' else ['baseline','early_recovery','late_recovery']
        labels=['0–6','6–12','12–24'] if cohort=='ECAM' else ['Base.','Early','Late']
        means=[];low=[];high=[]
        for p in phases:
            v=d[d.phase.eq(p)].groupby('subject_id').relative.mean().to_numpy()*100
            boot=v[rng.integers(len(v),size=(2000,len(v)))].mean(1);l,h=np.quantile(boot,[.025,.975]);means.append(v.mean());low.append(l);high.append(h)
            support_export.append(dict(cohort=cohort,name=name,phase=p,n_subjects=len(v),mean_relative_percent=v.mean(),ci_low=l,ci_high=h))
        ax.errorbar(range(3),means,yerr=[np.array(means)-low,np.array(high)-means],fmt='o-',color=COLORS['raw'],capsize=2.5,ms=3,lw=1)
        ax.set_xticks(range(3),labels);ax.set_xlim(-.25,2.25);ax.set_ylim(bottom=0)
        ax.set_xlabel('Age (months)' if cohort=='ECAM' else 'Study phase',fontsize=7)
        ax.set_ylabel('Raw relative abundance (%)',fontsize=7);ax.yaxis.set_major_locator(MaxNLocator(3));ax.tick_params(labelsize=6.6)
    save_tsv('I_raw_support_bootstrap.tsv',pd.DataFrame(support_export))

def panel_j(fig,gs):
    title(fig,gs,'J','A sparse reference-driven association')
    sub=gs.subgridspec(2,1,height_ratios=[1,1.35],hspace=.58)
    raw=load_tsv('IJ_raw_support_samples.tsv',dtype={'subject_id':str});d=raw[(raw.cohort.eq('FARMM'))&raw['name'].eq('Serratia_marcescens')&raw.phase.isin(['baseline','depletion'])]
    if len(d)!=224 or d['count'].sum()!=0:raise ValueError('Serratia raw-count mechanism no longer matches data')
    ax=fig.add_subplot(sub[0]);rng=np.random.default_rng(76)
    for i,p in enumerate(['baseline','depletion']):
        v=d[d.phase.eq(p)]['count'].to_numpy();ax.scatter(i+rng.uniform(-.23,.23,len(v)),v,s=8,facecolor=COLORS['raw'],alpha=.35,linewidths=0)
        ax.text(i,.38,f'{len(v)} samples\nall zero',ha='center',va='center',fontsize=7)
    ax.set_xticks([0,1],['Baseline','Depletion']);ax.set_ylabel('Observed count');ax.set_ylim(-.18,.75);ax.set_yticks([0]);ax.set_xlim(-.5,1.5)
    ax.set_title('FARMM · Serratia marcescens',fontsize=8,fontstyle='italic',pad=6)
    ax=fig.add_subplot(sub[1]);forest(ax,load_tsv('IJ_taxon_contrasts.tsv'),'FARMM','Serratia_marcescens','depletion_minus_baseline',methods=['raw','denoised'])

def make_figure(dpi,preview_only):
    import panels_af
    style();fig=plt.figure(figsize=(14.4,24.8),layout=None)
    outer=fig.add_gridspec(7,1,left=.057,right=.955,bottom=.035,top=.976,height_ratios=[3.7,2.6,2.25,3.15,3.7,3.6,3.5],hspace=.43)
    panels_af.panel_a(sys.modules[__name__],fig,outer[0])
    r0=outer[1].subgridspec(1,2,width_ratios=[1.35,1],wspace=.19)
    panel_b(fig,r0[0]);panel_c(fig,r0[1]);panel_d(fig,outer[2]);panel_e(fig,outer[3])
    panels_af.panel_f(sys.modules[__name__],fig,outer[4])
    r3=outer[5].subgridspec(1,2,width_ratios=[2.2,1],wspace=.22)
    panel_g(fig,r3[0]);panel_h(fig,r3[1])
    r4=outer[6].subgridspec(1,2,width_ratios=[2.2,1],wspace=.22)
    panel_i(fig,r4[0]);panel_j(fig,r4[1])
    fig.text(.057,.012,'K = 1 · Component recovery, event responses and real-cohort applications  |  Original-server fits; statistical definitions and selection rules in the accompanying caption.',fontsize=7,color='#555963')
    QA.mkdir(exist_ok=True)
    fig.canvas.draw();renderer=fig.canvas.get_renderer();canvas=fig.bbox
    outside=[];xlabel_overlaps=[];heading_collisions=[]
    for ax in fig.axes:
        xticks=[lab for pos,lab in zip(ax.get_xticks(),ax.get_xticklabels()) if min(ax.get_xlim())<=pos<=max(ax.get_xlim())]
        yticks=[lab for pos,lab in zip(ax.get_yticks(),ax.get_yticklabels()) if min(ax.get_ylim())<=pos<=max(ax.get_ylim())]
        for text in [ax.title,ax.xaxis.label,ax.yaxis.label]+ax.texts+xticks+yticks:
            if not text.get_visible() or not text.get_text():continue
            bb=text.get_window_extent(renderer)
            if bb.x0<canvas.x0-1 or bb.y0<canvas.y0-1 or bb.x1>canvas.x1+1 or bb.y1>canvas.y1+1:outside.append(text.get_text())
        if ax.xaxis.label.get_text():
            bb=ax.xaxis.label.get_window_extent(renderer)
            for other in fig.axes:
                if other is not ax and bb.overlaps(other.get_window_extent(renderer)):
                    xlabel_overlaps.append(dict(label=ax.xaxis.label.get_text(),other_title=other.get_title()))
    for txt in fig.texts:
        if not txt.get_visible():continue
        for ax in fig.axes:
            if ax.get_title() and txt.get_window_extent(renderer).overlaps(ax.title.get_window_extent(renderer)):
                heading_collisions.append(dict(heading=txt.get_text(),title=ax.get_title()))
    (QA/'layout_qc.json').write_text(json.dumps(dict(outside_page_text=outside,xlabel_overlaps_axes=xlabel_overlaps,heading_title_collisions=heading_collisions),ensure_ascii=False,indent=2),encoding='utf-8')
    fig.savefig(QA/'layout_preview.png',dpi=160)
    if preview_only:
        plt.close(fig);return
    print('Exporting vector PDF and 600 dpi raster files',flush=True)
    fig.savefig(HERE/(STEM+'.pdf'),dpi=dpi,metadata={'Title':'K=1 decomposition and real-cohort applications (A-J)','Author':'Research figure generated from frozen source data','Subject':'Original component recovery and downstream applications; statistical details in accompanying caption'})
    fig.savefig(HERE/(STEM+'.png'),dpi=dpi)
    plt.close(fig);gc.collect()
    Image.MAX_IMAGE_PIXELS=None
    with Image.open(HERE/(STEM+'.png')) as im:
        rgb=im.convert('RGB');rgb.save(HERE/(STEM+'.tiff'),compression='tiff_lzw',dpi=(dpi,dpi));rgb.close()
    outputs={}
    for ext in ['pdf','png','tiff']:
        p=HERE/(STEM+'.'+ext);outputs[p.name]=dict(bytes=p.stat().st_size,sha256=digest(p))
    with Image.open(HERE/(STEM+'.png')) as im:dimensions=im.size;metadata_dpi=im.info.get('dpi')
    manifest=dict(figure=STEM,panels=list('ABCDEFGHIJ'),deferred=[],
                  dpi=dpi,size_inches=[14.4,24.8],size_pixels=list(dimensions),png_dpi=metadata_dpi,
                  outputs=outputs,script_sha256=digest(__file__),panels_af_sha256=digest(HERE/'panels_af.py'),provenance=digest(DATA/'provenance.json'))
    (HERE/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2),flush=True)

def main():
    global ROOT
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--cached',action='store_true');p.add_argument('--preview-only',action='store_true');p.add_argument('--dpi',type=int,default=600)
    a=p.parse_args();ROOT=a.root.resolve()
    if not a.cached:prepare_data(ROOT)
    elif not (DATA/'provenance.json').exists():raise FileNotFoundError('Run once without --cached to create source_data')
    if not a.cached or not (DATA/'F_pooled_event_curves.tsv').exists():
        import panels_af
        panels_af.prepare(sys.modules[__name__],ROOT)
    make_figure(a.dpi,a.preview_only)

if __name__=='__main__':main()
