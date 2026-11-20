"""Seed-pooled biological analysis of the event-aware decomposition.

Per cohort, all numerically complete seeds are combined:
  description  population trajectory, event response curves, variance partition
  inference    per-taxon lfsr / cumulative-lfsr (pooled posterior draws);
               subject-level permutation tests of responses vs covariates
  age clock    subject-grouped CV on raw, denoised and event-removed CLR;
               event-attributed maturity shift = clock(total) - clock(event removed)
All log-abundance quantities are centred over taxa (CLR) before interpretation.
Posterior intervals are Gaussian-VI approximations pooled over seeds; the three
seeds are optimisation restarts, not independent replicates.
"""
from pathlib import Path
import argparse
import itertools
import json
import sys
import traceback
import numpy as np
import pandas as pd
from common import digest,write_json,write_npz

COMPONENT_KEYS=['common','individual','event','total','baseline','zero_event','removed_event_log_reconstruction']
COVARIATES={'DIABIMMUNE':dict(categorical=['group','csection','sex'],numeric=['event_day']),
            'FARMM':dict(categorical=['group'],numeric=['Age','BMI']),
            'ECAM':dict(categorical=['group','diet','sex','abx_ever'],numeric=[])}
CLOCK={'DIABIMMUNE':True,'ECAM':True,'FARMM':False}
SHIFT_BINS={'DIABIMMUNE':[-90,0,30,90,180,365,2000]}
EFFECT=.2          # practical effect on CLR / logit scale
LFSR=.05
PERMUTATIONS=4999
TOP=12


# ---------------------------------------------------------------- helpers
def centre(a,axis=-1):
    a=np.asarray(a,float);return a-a.mean(axis,keepdims=True)


def lfsr(draws):
    """Local false sign rate along axis 0."""
    return 1-np.maximum((draws>0).mean(0),(draws<0).mean(0))


def cumulative_lfsr(values):
    """Average lfsr of all taxa at least as extreme (Stephens 2017 s-value style)."""
    v=np.asarray(values,float);out=np.full(v.shape,np.nan);ok=np.isfinite(v)
    order=np.argsort(v[ok]);cum=np.cumsum(v[ok][order])/np.arange(1,ok.sum()+1)
    tmp=np.empty(ok.sum());tmp[order]=cum;out[ok]=tmp
    return out


def short_name(taxonomy):
    parts=[p for p in str(taxonomy).split('|') if p and not p.endswith('__')]
    return parts[-1].split('__')[-1] if parts else str(taxonomy)


def bh(p):
    p=np.asarray(p,float);q=np.full(p.shape,np.nan);ok=np.isfinite(p)
    if ok.any():
        v=p[ok];o=np.argsort(v);r=v[o]*ok.sum()/np.arange(1,ok.sum()+1)
        r=np.minimum.accumulate(r[::-1])[::-1];tmp=np.empty_like(v);tmp[o]=np.minimum(r,1);q[ok]=tmp
    return q


def bootstrap_mean(v,rng,draws=2000):
    v=np.asarray(v,float);v=v[np.isfinite(v)]
    if len(v)<3:return np.nan,np.nan
    return tuple(np.quantile(rng.choice(v,(draws,len(v))).mean(1),[.025,.975]))


# ---------------------------------------------------------------- loading
def load_seed(path,allow_smoke=False):
    done=json.loads((path/'done.json').read_text(encoding='utf-8'))
    smoke=done['identity'].get('smoke',False)
    if smoke and not allow_smoke:raise ValueError('Smoke fit cannot be analysed: '+str(path))
    if not done.get('usable',False) and not smoke:raise ValueError('Fit not numerically complete: '+str(path))
    for name,h in done['files'].items():
        if digest(path/name)!=h:raise ValueError('Fit checksum mismatch: '+str(path/name))
    s=pd.read_csv(path/'samples.tsv',sep='\t',dtype={'sample_id':str,'subject_id':str})
    taxa=pd.read_csv(path/'taxa.tsv',sep='\t',dtype=str)
    with np.load(path/'components.npz',allow_pickle=False) as d:
        comp={k:d[k+'_mean'].copy() for k in COMPONENT_KEYS}
        if not (np.array_equal(d['sample_id'],s.sample_id.to_numpy(str)) and np.array_equal(d['taxon_id'],taxa.taxon_id.to_numpy(str))):
            raise ValueError('Component identity mismatch: '+str(path))
    with np.load(path/'posterior_draws.npz',allow_pickle=False) as d:draws={k:d[k].copy() for k in d.files}
    with np.load(path/'posterior_observed.npz',allow_pickle=False) as d:counts=d['raw_counts'].copy()
    with np.load(path/'event_input.npz',allow_pickle=False) as d:events=d['event_times'].copy();names=d['event_names'].tolist()
    quality=json.loads((path/'quality.json').read_text(encoding='utf-8'))
    return dict(path=path,task=done['identity']['task'],samples=s,taxa=taxa,components=comp,draws=draws,
                counts=counts,events=events,event_names=names,quality=quality)


def pool(seeds):
    ref=seeds[0]
    for s in seeds[1:]:
        if not (s['samples'].sample_id.equals(ref['samples'].sample_id) and s['taxa'].taxon_id.equals(ref['taxa'].taxon_id)):
            raise ValueError('Seeds disagree on samples/taxa')
        if not np.array_equal(s['events'],ref['events'],equal_nan=True):raise ValueError('Seeds disagree on events')
    components={k:np.mean([s['components'][k] for s in seeds],0) for k in COMPONENT_KEYS}
    draws={k:np.concatenate([s['draws'][k] for s in seeds]) for k in
           ['event_curve','zero_event_curve','population_log','population_with_event_log','gain'] if k in ref['draws']}
    for k in ['age_day','relative_day','curve_supported','median_event_day','event_names','subject_id','taxon_id']:
        draws[k]=ref['draws'][k]
    return components,draws


# ---------------------------------------------------------------- description
def seed_agreement(seeds):
    rows=[]
    for a,b in itertools.combinations(seeds,2):
        row=dict(seed_a=a['task']['seed'],seed_b=b['task']['seed'])
        for k in ['common','individual','event','total']:
            x,y=centre(a['components'][k]).ravel(),centre(b['components'][k]).ravel()
            row[k+'_correlation']=float(np.corrcoef(x,y)[0,1]) if x.std()>1e-12 and y.std()>1e-12 else np.nan
            row[k+'_relative_rmse']=float(np.sqrt(np.mean((x-y)**2))/max(np.sqrt(np.mean(x**2)),1e-12))
        x=centre(a['draws']['event_curve'].mean(0),0).ravel();y=centre(b['draws']['event_curve'].mean(0),0).ravel()
        row['event_curve_correlation']=float(np.corrcoef(x,y)[0,1]) if x.std()>1e-12 and y.std()>1e-12 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def trajectory(draws,taxa,key):
    clr=centre(draws[key])            # draws x age x taxa
    tail=[.025,.975];age=draws['age_day']
    lo,hi=np.quantile(clr,tail,axis=0);mean=clr.mean(0)
    long=pd.DataFrame(dict(taxon_id=np.tile(taxa.taxon_id,len(age)),age_day=np.repeat(age,len(taxa)),
        clr_mean=mean.ravel(),clr_lower=lo.ravel(),clr_upper=hi.ravel()))
    change=clr[:,-1]-clr[:,0];l=lfsr(change)
    clo,chi=np.quantile(change,tail,axis=0)
    summary=pd.DataFrame(dict(taxon_id=taxa.taxon_id,name=taxa.taxonomy.map(short_name),
        change_last_minus_first=change.mean(0),change_lower=clo,change_upper=chi,lfsr=l,
        p_abs_change_gt_effect=(np.abs(change)>EFFECT).mean(0),cumulative_lfsr=cumulative_lfsr(l),
        amplitude=mean.max(0)-mean.min(0),peak_day=age[mean.argmax(0)],trough_day=age[mean.argmin(0)],
        mean_clr=mean.mean(0)))
    summary['discovery']=summary.cumulative_lfsr<=LFSR
    return long,summary.sort_values('lfsr')


def variance_partition(components,taxa):
    parts={k:centre(components[k]) for k in ['baseline','common','individual','event']}
    var={k:v.var(0) for k,v in parts.items()}
    total=sum(var.values());rows=pd.DataFrame(dict(taxon_id=taxa.taxon_id,name=taxa.taxonomy.map(short_name),
        total_component_variance=total,**{k+'_share':var[k]/np.maximum(total,1e-12) for k in var}))
    cohort={k+'_share_of_summed_variance':float(var[k].sum()/max(total.sum(),1e-12)) for k in var}
    cohort.update({k+'_median_taxon_share':float(np.median(rows[k+'_share'])) for k in var})
    return rows,cohort


# ---------------------------------------------------------------- event inference
def event_inference(draws,taxa):
    curves=centre(draws['event_curve'],1)          # draws x taxa x event x grid; CLR over taxa
    zero=draws['zero_event_curve'];grid=draws['relative_day'];sup=draws['curve_supported']
    long=[];summary=[]
    for e,name in enumerate(draws['event_names']):
        c=curves[:,:,e];m=c.mean(0);lo,hi=np.quantile(c,[.025,.975],axis=0);l=lfsr(c)
        long.append(pd.DataFrame(dict(event=name,taxon_id=np.repeat(taxa.taxon_id,len(grid)),
            relative_day=np.tile(grid,len(taxa)),clr_effect_mean=m.ravel(),clr_effect_lower=lo.ravel(),
            clr_effect_upper=hi.ravel(),lfsr=l.ravel(),supported=np.tile(sup[e],len(taxa)))))
        S=sup[e]
        if not S.any():
            summary.append(pd.DataFrame(dict(event=name,taxon_id=taxa.taxon_id,status='not_testable_no_supported_post_event_window')))
            continue
        avg=c[...,S].mean(-1);za=zero[:,:,e][...,S].mean(-1);la=lfsr(avg);lz=lfsr(za)
        ms=np.where(S,np.abs(m),-np.inf);peak=ms.argmax(1)
        d=pd.DataFrame(dict(event=name,taxon_id=taxa.taxon_id,name=taxa.taxonomy.map(short_name),status='tested',
            supported_days=f'{grid[S].min():.1f}..{grid[S].max():.1f}',
            mean_supported_effect=avg.mean(0),effect_lower=np.quantile(avg,.025,0),effect_upper=np.quantile(avg,.975,0),
            lfsr=la,p_abs_effect_gt_threshold=(np.abs(avg)>EFFECT).mean(0),cumulative_lfsr=cumulative_lfsr(la),
            peak_relative_day=grid[peak],peak_effect=m[np.arange(len(m)),peak],
            absence_logit_effect=za.mean(0),absence_lfsr=lz,absence_cumulative_lfsr=cumulative_lfsr(lz)))
        d['discovery']=d.cumulative_lfsr<=LFSR;d['absence_discovery']=d.absence_cumulative_lfsr<=LFSR
        summary.append(d.sort_values('lfsr'))
    return pd.concat(long,ignore_index=True),pd.concat(summary,ignore_index=True)


# ---------------------------------------------------------------- subject level
def subject_table(cohort,s,components,draws,events,effects):
    subjects=pd.Series(draws['subject_id'])
    first=s.groupby('subject_id').first()
    t=pd.DataFrame(index=subjects.values);t.index.name='subject_id'
    for c in COVARIATES[cohort]['categorical']+COVARIATES[cohort]['numeric']:
        if c in first:t[c]=first[c].reindex(t.index).values
    if cohort=='ECAM':
        t['abx_ever']=s.assign(a=pd.to_numeric(s.abx_all_sources,errors='coerce')).groupby('subject_id').a.max().reindex(t.index).astype('Int64').astype(str).values
    if cohort=='DIABIMMUNE':t['event_day']=events[:,0]
    ind=centre(components['individual']);ev=centre(components['event'])
    t['n_visits']=s.groupby('subject_id').size().reindex(t.index).values
    t['individual_dynamics_rms']=[np.sqrt(np.mean(ind[s.subject_id.eq(i).to_numpy()]**2)) for i in t.index]
    finite=np.isfinite(events).any()
    if finite and effects is not None and (effects.status=='tested').any():
        sup=draws['curve_supported'][0];grid=draws['relative_day'];hi=grid[sup].max()
        tau=pd.Series(events[:,0],index=subjects.values)
        rows=[]
        for i in t.index:
            sel=(s.subject_id.eq(i)&(s.time>=tau[i])&(s.time<=tau[i]+hi)).to_numpy()
            rows.append(np.sqrt(np.mean(ev[sel]**2)) if sel.any() else np.nan)
        t['event_response_rms']=rows
        if 'gain' in draws:
            tested=effects[effects.status.eq('tested')&effects.event.eq(draws['event_names'][0])]
            w=tested.set_index('taxon_id').mean_supported_effect.reindex(pd.Index(draws['taxon_id'])).fillna(0).to_numpy()**2
            g=1+draws['gain'][...,0].mean(0)                         # subjects x taxa
            t['event_amplification']=(g*w).sum(1)/max(w.sum(),1e-12)
    return t


def permutation_tests(y,x,kind,rng,perms=PERMUTATIONS):
    """y: subjects x outcomes. Returns statistic and permutation p per outcome."""
    ok=np.isfinite(y).all(1)&pd.notna(x)
    if kind=='categorical':ok&=~pd.Series(x).isin(['nan','<NA>','None','not provided','unknown']).to_numpy()
    y,x=y[ok],np.asarray(x)[ok];n=len(y)
    if n<8:return None
    if kind=='categorical':
        levels,codes=np.unique(x.astype(str),return_inverse=True);k=len(levels)
        if k<2 or np.bincount(codes).min()<3:return None
        def stat(c):
            o=np.eye(k)[c]                                      # (...,n,k)
            cnt=o.sum(-2);sums=np.einsum('...nk,nm->...km',o,y);mean=sums/cnt[...,None]
            grand=y.mean(0);ssb=(cnt[...,None]*(mean-grand)**2).sum(-2);sst=((y-grand)**2).sum(0)
            return (ssb/(k-1))/np.maximum((sst-ssb)/(n-k),1e-300)
        observed=stat(codes);perm=np.stack([rng.permutation(codes) for _ in range(perms)])
        null=np.concatenate([stat(perm[i:i+500]) for i in range(0,perms,500)])
        means={lv:y[codes==j].mean(0) for j,lv in enumerate(levels)}
        effect=means[levels[1]]-means[levels[0]] if k==2 else np.full(y.shape[1],np.nan)
        detail=[json.dumps({lv:round(float(means[lv][m]),4) for lv in levels}) for m in range(y.shape[1])]
        contrast=f'{levels[1]} - {levels[0]}' if k==2 else 'F test'
    else:
        xr=pd.Series(x.astype(float)).rank().to_numpy();yr=np.apply_along_axis(lambda v:pd.Series(v).rank().to_numpy(),0,y)
        xr=centre(xr,0);yr=centre(yr,0)
        def stat(xx):return np.abs(xx@yr)/np.sqrt((xx**2).sum(-1,keepdims=True)*(yr**2).sum(0))
        observed=stat(xr[None])[0];null=stat(np.stack([rng.permutation(xr) for _ in range(perms)]))
        effect=(xr@yr)/np.sqrt((xr**2).sum()*(yr**2).sum(0));detail=['']*y.shape[1];contrast='Spearman rho'
    p=(1+(null>=observed[None]-1e-12).sum(0))/(perms+1)
    return dict(n=n,statistic=observed,p=p,effect=effect,contrast=contrast,detail=detail)


def associations(cohort,subjects,effects,draws,taxa,rng):
    outcomes=[c for c in ['individual_dynamics_rms','event_response_rms','event_amplification',
                          'relative_maturity_day','event_maturity_shift_day'] if c in subjects]
    rows=[];taxon_rows=[]
    for kind in ['categorical','numeric']:
        for cov in COVARIATES[cohort][kind]:
            if cov not in subjects:continue
            for out in outcomes:
                r=permutation_tests(subjects[[out]].to_numpy(float),subjects[cov].to_numpy(),kind,rng)
                if r is None:continue
                rows.append(dict(outcome=out,covariate=cov,test=r['contrast'],n_subjects=r['n'],
                    effect=r['effect'][0],group_means=r['detail'][0],statistic=r['statistic'][0],p_permutation=r['p'][0]))
            # Per-taxon individual response multipliers for the strongest event taxa.
            if 'gain' in draws and effects is not None and (effects.status=='tested').any():
                top=effects[effects.status.eq('tested')].sort_values('lfsr').head(30)
                ix=pd.Index(draws['taxon_id']).get_indexer(top.taxon_id)
                y=1+draws['gain'][...,0].mean(0)[:,ix]
                r=permutation_tests(y,subjects[cov].to_numpy(),kind,rng)
                if r is None:continue
                for m,(tid,name) in enumerate(zip(top.taxon_id,top.name)):
                    taxon_rows.append(dict(taxon_id=tid,name=name,covariate=cov,test=r['contrast'],n_subjects=r['n'],
                        effect=r['effect'][m],group_means=r['detail'][m],p_permutation=r['p'][m]))
    a=pd.DataFrame(rows);b=pd.DataFrame(taxon_rows)
    if len(a):a['q_bh']=bh(a.p_permutation)
    if len(b):b['q_bh']=bh(b.p_permutation)
    return a,b


# ---------------------------------------------------------------- age clock
def ridge_fit(x,y,weights,alpha):
    w=weights/weights.mean();mx=np.average(x,0,weights=w);my=np.average(y,weights=w)
    a=(x-mx)*np.sqrt(w)[:,None];b=(y-my)*np.sqrt(w)
    dual=np.linalg.solve(a@a.T+alpha*np.eye(len(a)),b);coef=a.T@dual
    return lambda q:(q-mx)@coef+my


def grouped_folds(subjects,k,seed):
    u=np.array(sorted(pd.unique(subjects)));np.random.default_rng(seed).shuffle(u)
    return [np.isin(subjects,f) for f in np.array_split(u,min(k,len(u)))]


def choose_alpha(x,y,subjects,weights,seed,grid=(1.,10.,100.,1000.)):
    folds=grouped_folds(subjects,4,seed);errors=[]
    for alpha in grid:
        e=[np.average(np.abs(ridge_fit(x[~f],y[~f],weights[~f],alpha)(x[f])-y[f]),weights=weights[f]) for f in folds]
        errors.append(np.mean(e))
    return grid[int(np.argmin(errors))]


CLOCK_NOTES={('raw','raw'):'honest: raw clock, subject-held-out raw profiles',
    ('raw','denoised'):'PRIMARY: raw-trained clock read on held-out subjects denoised profiles (denoiser itself saw all subjects)',
    ('raw','event_removed'):'raw-trained clock on event-removed denoised profiles',
    ('denoised','raw'):'denoised-trained clock on raw profiles',
    ('denoised','denoised'):'CIRCULAR: the common component is an exact function of age; even an unfitted model scores well',
    ('denoised','event_removed'):'denoised-trained clock on event-removed profiles'}


def age_clock(cohort,s,components,counts,events,seed=2026):
    """Subject-grouped CV ridge clocks trained on raw (primary) and denoised (secondary) CLR.

    Each clock is applied to the held-out subjects' raw, denoised and event-removed
    profiles. Event shift = clock(denoised) - clock(event removed) under the same clock.
    """
    reps={'raw':centre(np.log(counts+.5)),'denoised':centre(components['total'])}
    has_event=bool(np.isfinite(events).any())
    if has_event:reps['event_removed']=centre(components['removed_event_log_reconstruction'])
    subj=s.subject_id.to_numpy();age=s.time.to_numpy(float)
    weights=1/s.subject_id.map(s.subject_id.value_counts()).to_numpy(float)
    pred={(c,r):np.full(len(s),np.nan) for c in ['raw','denoised'] for r in reps};alphas=[]
    for f,test in enumerate(grouped_folds(subj,5,seed)):
        train=~test
        for c in ['raw','denoised']:
            alpha=choose_alpha(reps[c][train],age[train],subj[train],weights[train],seed+f)
            model=ridge_fit(reps[c][train],age[train],weights[train],alpha)
            alphas.append(dict(fold=f,clock=c,alpha=alpha))
            for r,x in reps.items():pred[(c,r)][test]=model(x[test])
    visits=[];perf=[]
    rel=age-events[s.tensor_i.to_numpy(),0] if has_event else np.full(len(s),np.nan)
    for (c,r),p in pred.items():
        # Age-adjusted relative maturity: residual from a cubic calibration of prediction on age.
        cal=np.polyval(np.polyfit(age,p,3),age)
        shift=pred[(c,'denoised')]-pred[(c,'event_removed')] if has_event else np.full(len(s),np.nan)
        visits.append(pd.DataFrame(dict(sample_id=s.sample_id,subject_id=subj,age_day=age,clock=c,applied_to=r,
            predicted_age_day=p,residual_day=p-age,relative_maturity_day=p-cal,days_since_event=rel,
            event_maturity_shift_day=shift)))
        err=np.abs(p-age)
        perf.append(dict(clock=c,applied_to=r,mae_day=err.mean(),median_ae_day=np.median(err),
            subject_weighted_mae_day=np.average(err,weights=weights),
            spearman=pd.Series(p).corr(pd.Series(age),method='spearman'),
            r2=1-np.sum((p-age)**2)/np.sum((age-age.mean())**2),note=CLOCK_NOTES[(c,r)]))
    return pd.concat(visits,ignore_index=True),pd.DataFrame(perf),pd.DataFrame(alphas)


def shift_by_bin(cohort,visits,rng):
    rows=[]
    for c,v in visits[visits.applied_to.eq('denoised')].groupby('clock'):
        v=v.copy();v['bin']=pd.cut(v.days_since_event,SHIFT_BINS.get(cohort,[-np.inf,0,np.inf]),right=False)
        for b,g in v.groupby('bin',observed=True):
            per=g.groupby('subject_id').event_maturity_shift_day.mean();lo,hi=bootstrap_mean(per.to_numpy(),rng)
            rows.append(dict(clock=c,primary=c=='raw',days_since_event=str(b),n_subjects=len(per),n_visits=len(g),
                mean_shift_day=per.mean(),ci_low=lo,ci_high=hi,
                interpretation='model-attributed maturity contribution of the event component; subject bootstrap'))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures
def figures(out,cohort,taxa,draws,traj_summary,effects,perf,visits,shift,vp_cohort):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    age=draws['age_day'];names=taxa.taxonomy.map(short_name).to_numpy();tid=pd.Index(taxa.taxon_id)
    clr=centre(draws['population_log']);m=clr.mean(0);lo,hi=np.quantile(clr,[.025,.975],axis=0)
    top=tid.get_indexer(traj_summary.sort_values('lfsr').head(TOP).taxon_id)
    fig,axes=plt.subplots(3,4,figsize=(13,8),sharex=True)
    for ax,j in zip(axes.ravel(),top):
        ax.fill_between(age,lo[:,j],hi[:,j],color='#207D83',alpha=.25);ax.plot(age,m[:,j],color='#207D83')
        ax.set_title(names[j][:32],fontsize=8)
    fig.suptitle(f'{cohort}: population trajectory (CLR, seed-pooled 95% interval)');fig.supxlabel('Day');fig.tight_layout()
    fig.savefig(out/'population_trajectories.png',dpi=160);plt.close(fig)
    if effects is not None and (effects.status=='tested').any():
        c=centre(draws['event_curve'],1)[:,:,0];grid=draws['relative_day'];sup=draws['curve_supported'][0]
        cm=c.mean(0);cl,ch=np.quantile(c,[.025,.975],axis=0)
        top=tid.get_indexer(effects[effects.status.eq('tested')].sort_values('lfsr').head(TOP).taxon_id)
        fig,axes=plt.subplots(3,4,figsize=(13,8),sharex=True)
        for ax,j in zip(axes.ravel(),top):
            ax.fill_between(grid,cl[j],ch[j],color='#C0504D',alpha=.25);ax.plot(grid,cm[j],color='#C0504D')
            ax.axhline(0,color='grey',lw=.5);ax.axvline(0,color='grey',lw=.5,ls='--')
            for g,ok in zip(grid,sup):
                if not ok:ax.axvspan(g-.5*(grid[1]-grid[0]),g+.5*(grid[1]-grid[0]),color='#eeeeee',lw=0,zorder=0)
            ax.set_title(names[j][:32],fontsize=8)
        fig.suptitle(f'{cohort}: event response (CLR; grey = no data support)');fig.supxlabel('Days relative to event')
        fig.tight_layout();fig.savefig(out/'event_curves.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,3.5))
    keys=['baseline','common','individual','event'];vals=[vp_cohort[k+'_share_of_summed_variance'] for k in keys]
    ax.bar(keys,vals,color=['#999999','#207D83','#C28A24','#C0504D']);ax.set_ylabel('Share of summed component variance')
    ax.set_title(f'{cohort}: variance partition (CLR)');fig.tight_layout();fig.savefig(out/'variance_partition.png',dpi=160);plt.close(fig)
    if visits is not None:
        pairs=perf[['clock','applied_to']].values.tolist();fig,axes=plt.subplots(2,(len(pairs)+1)//2,figsize=(4*((len(pairs)+1)//2),7),sharey=True)
        for ax,(c,r) in zip(np.ravel(axes),pairs):
            v=visits[visits.clock.eq(c)&visits.applied_to.eq(r)];ax.scatter(v.age_day,v.predicted_age_day,s=4,alpha=.4)
            lim=[v.age_day.min(),v.age_day.max()];ax.plot(lim,lim,color='grey',lw=.6)
            mae=perf[perf.clock.eq(c)&perf.applied_to.eq(r)].mae_day.iloc[0]
            ax.set_title(f"{c} clock -> {r}\nMAE {mae:.0f} d",fontsize=9);ax.set_xlabel('Age (day)')
        fig.supylabel('Microbiota age (day)');fig.tight_layout();fig.savefig(out/'age_clock.png',dpi=160);plt.close(fig)
    if shift is not None and len(shift):
        shift=shift[shift.primary];fig,ax=plt.subplots(figsize=(6,3.6));x=np.arange(len(shift))
        ax.errorbar(x,shift.mean_shift_day,yerr=[shift.mean_shift_day-shift.ci_low,shift.ci_high-shift.mean_shift_day],fmt='o',color='#C0504D')
        ax.axhline(0,color='grey',lw=.6);ax.set_xticks(x,shift.days_since_event,rotation=30,fontsize=7)
        ax.set_ylabel('clock(total) - clock(event removed), day');ax.set_title(f'{cohort}: event-attributed maturity shift (raw-trained clock)')
        fig.tight_layout();fig.savefig(out/'event_maturity_shift.png',dpi=160);plt.close(fig)


# ---------------------------------------------------------------- driver
def analyze(cohort,fit_dirs,out,allow_smoke=False,seed=2026):
    out.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(seed)
    status=[];seeds=[]
    for path in fit_dirs:
        try:seeds.append(load_seed(path,allow_smoke));status.append(dict(fit=str(path),status='used'))
        except Exception as exc:status.append(dict(fit=str(path),status='skipped',reason=str(exc)))
    pd.DataFrame(status).to_csv(out/'seed_status.tsv',sep='\t',index=False)
    if not seeds:raise RuntimeError('No usable seed for '+cohort)
    s,taxa,events=seeds[0]['samples'],seeds[0]['taxa'],seeds[0]['events']
    components,draws=pool(seeds)
    write_npz(out/'ensemble_components.npz',sample_id=s.sample_id.to_numpy(str),taxon_id=taxa.taxon_id.to_numpy(str),
              **{k+'_mean':v for k,v in components.items()})
    pd.DataFrame([dict(seed=x['task']['seed'],**{k:v for k,v in x['quality'].items() if not isinstance(v,(list,dict))}) for x in seeds]
                 ).to_csv(out/'seed_quality.tsv',sep='\t',index=False)
    agreement=seed_agreement(seeds) if len(seeds)>1 else pd.DataFrame()
    agreement.to_csv(out/'seed_agreement.tsv',sep='\t',index=False)
    long,traj=trajectory(draws,taxa,'population_log')
    long.to_csv(out/'population_trajectory.tsv',sep='\t',index=False);traj.to_csv(out/'population_trajectory_taxa.tsv',sep='\t',index=False)
    has_event=bool(np.isfinite(events).any());effects=None
    if has_event:
        long_e,traj_e=trajectory(draws,taxa,'population_with_event_log')
        long_e.to_csv(out/'population_trajectory_with_event.tsv',sep='\t',index=False)
        traj_e.to_csv(out/'population_trajectory_with_event_taxa.tsv',sep='\t',index=False)
        curves,effects=event_inference(draws,taxa)
        curves.to_csv(out/'event_curves.tsv',sep='\t',index=False);effects.to_csv(out/'event_taxa.tsv',sep='\t',index=False)
    vp,vp_cohort=variance_partition(components,taxa);vp.to_csv(out/'variance_partition_taxa.tsv',sep='\t',index=False)
    subjects=subject_table(cohort,s,components,draws,events,effects)
    visits=perf=shift=None
    if CLOCK[cohort]:
        visits,perf,alphas=age_clock(cohort,s,components,seeds[0]['counts'],events,seed)
        visits.to_csv(out/'clock_visits.tsv',sep='\t',index=False);perf.to_csv(out/'clock_performance.tsv',sep='\t',index=False)
        alphas.to_csv(out/'clock_alphas.tsv',sep='\t',index=False)
        v=visits[visits.clock.eq('raw')&visits.applied_to.eq('denoised')]
        subjects['relative_maturity_day']=v.groupby('subject_id').relative_maturity_day.mean().reindex(subjects.index)
        if has_event:
            shift=shift_by_bin(cohort,visits,rng);shift.to_csv(out/'event_maturity_shift.tsv',sep='\t',index=False)
            early=v[(v.days_since_event>=0)&(v.days_since_event<180)]
            subjects['event_maturity_shift_day']=early.groupby('subject_id').event_maturity_shift_day.mean().reindex(subjects.index)
    subjects.to_csv(out/'subject_outcomes.tsv',sep='\t')
    assoc,taxon_assoc=associations(cohort,subjects,effects,draws,taxa,rng)
    assoc.to_csv(out/'subject_associations.tsv',sep='\t',index=False)
    if len(taxon_assoc):taxon_assoc.to_csv(out/'taxon_response_associations.tsv',sep='\t',index=False)
    figures(out,cohort,taxa,draws,traj,effects,perf,visits,shift,vp_cohort)
    key=dict(cohort=cohort,seeds_used=[x['task']['seed'] for x in seeds],n_subjects=s.subject_id.nunique(),n_samples=len(s),
        n_taxa=len(taxa),event_names=seeds[0]['event_names'] if has_event else [],
        trajectory_discoveries=int(traj.discovery.sum()),variance_partition=vp_cohort,
        min_seed_total_correlation=float(agreement.total_correlation.min()) if len(agreement) else None,
        min_seed_event_correlation=float(agreement.event_correlation.min()) if len(agreement) and has_event else None)
    if effects is not None and 'discovery' in effects:
        key.update(event_discoveries=int(effects.discovery.fillna(False).sum()),
                   absence_event_discoveries=int(effects.absence_discovery.fillna(False).sum()))
    if perf is not None:key['clock_mae_day']={f'{r.clock}_clock->{r.applied_to}':round(r.mae_day,1) for r in perf.itertuples()}
    if len(assoc):key['subject_associations_q_lt_0_1']=assoc[assoc.q_bh<.1][['outcome','covariate','effect','q_bh']].to_dict('records')
    write_json(out/'key_results.json',key)
    report(out,key,cohort,traj,effects,perf,shift,assoc)
    write_json(out/'complete.json',dict(status='complete',files={p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!='complete.json'}))
    print('Completed analysis',cohort,out,flush=True)
    return key


def report(out,key,cohort,traj,effects,perf,shift,assoc):
    fmt=lambda d:d.to_markdown(index=False,floatfmt='.3g') if hasattr(d,'to_markdown') else d.to_string(index=False)
    def table(d):
        try:return fmt(d)
        except ImportError:return '```\n'+d.to_string(index=False)+'\n```'
    lines=[f'# {cohort}: seed-pooled event-aware analysis','',
        f"Seeds: {key['seeds_used']}; subjects {key['n_subjects']}; samples {key['n_samples']}; taxa {key['n_taxa']}.",
        f"Seed agreement (min corr) total: {key['min_seed_total_correlation']}; event: {key['min_seed_event_correlation']}.",'',
        '## 1. Population trajectory',f"Taxa with credible late-vs-early change (cumulative lfsr <= {LFSR}): {key['trajectory_discoveries']}.",'',
        table(traj.head(15)[['name','change_last_minus_first','change_lower','change_upper','lfsr','peak_day']]),'',
        '## 2. Variance partition (CLR, posterior means)','',
        table(pd.DataFrame([key['variance_partition']]).T.reset_index().rename(columns={'index':'quantity',0:'value'})),'']
    if effects is not None and 'discovery' in effects:
        e=effects[effects.status.eq('tested')]
        lines+=['## 3. Event response',f"Supported window: {e.supported_days.iloc[0] if len(e) else 'none'} days. "
            f"Event taxa (cumulative lfsr <= {LFSR}): {key.get('event_discoveries')}; absence-channel: {key.get('absence_event_discoveries')}.",'',
            table(e.head(15)[['name','mean_supported_effect','effect_lower','effect_upper','lfsr','peak_relative_day','absence_logit_effect']]),'']
    if perf is not None:
        lines+=['## 4. Age clock','',table(perf),'',
            'Primary = raw-trained clock (subject-held-out training) read on denoised profiles. A clock trained on denoised profiles is circular: the common component is an exact age spline, so its MAE is not evidence of anything.','']
        if shift is not None:lines+=['Event-attributed maturity shift (same clock, event component removed):','',table(shift),'']
    if len(assoc):lines+=['## 5. Subject-level associations (permutation, BH over this table)','',table(assoc.sort_values('p_permutation')),'']
    lines+=['## Interpretation','- Components are model-based; separation of common trend and event relies on event timing variation and priors.',
        '- Event removal is a model-conditional reconstruction, not a causal counterfactual.',
        '- Intervals pool Gaussian-VI draws across optimisation seeds; they are approximate and tend to be narrow.',
        '- Subject-level tests condition on the fitted decomposition (fitting uncertainty not propagated).']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['analyze','summarize'])
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cohort',choices=list(COVARIATES))
    p.add_argument('--smoke',action='store_true',help='Analyse smoke fits (pipeline test only)')
    a=p.parse_args()
    from event_run import verify,location
    plan=verify(a.bundle.resolve());out=a.out.resolve()
    if a.command=='analyze':
        if not a.cohort:p.error('--cohort required')
        dirs=[location(out,t,a.smoke) for t in plan['tasks'] if t['dataset']==a.cohort]
        target=out/('smoke_analysis' if a.smoke else 'analysis')/a.cohort
        try:analyze(a.cohort,dirs,target,allow_smoke=a.smoke)
        except Exception:
            target.mkdir(parents=True,exist_ok=True);write_json(target/'failure.json',dict(traceback=traceback.format_exc()));raise
    else:
        rows=[]
        for data in plan['datasets']:
            t=out/'analysis'/data['dataset'];row=dict(cohort=data['dataset'])
            if (t/'complete.json').exists():row.update(status='complete',**{k:json.dumps(v) if isinstance(v,(dict,list)) else v
                for k,v in json.loads((t/'key_results.json').read_text(encoding='utf-8')).items() if k!='cohort'})
            else:row['status']='failed' if (t/'failure.json').exists() else 'missing'
            rows.append(row)
        fits=[]
        for task in plan['tasks']:
            d=location(out,task);st='missing'
            if (d/'done.json').exists():st=json.loads((d/'done.json').read_text(encoding='utf-8'))['status']
            elif (d/'failure.json').exists():st='failed'
            fits.append(dict(task,status=st))
        pd.DataFrame(fits).to_csv(out/'fit_status.tsv',sep='\t',index=False)
        pd.DataFrame(rows).to_csv(out/'analysis_status.tsv',sep='\t',index=False)
        print(pd.DataFrame(fits).to_string(index=False));print(pd.DataFrame(rows)[['cohort','status']].to_string(index=False))
        if any(r['status']!='complete' for r in rows):sys.exit(2)


if __name__=='__main__':main()
