"""Publication tables: subject-level contrasts, recovery, and descriptive ordination.

No outcome is selected for significance. All inference is conditional on fitted
representations; bootstrap intervals do NOT propagate denoiser uncertainty.
"""
from pathlib import Path
import argparse
import itertools
import json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist
from common import digest, write_json
from cohort_analysis import centre, bh, load_seed, pool, seed_agreement

MARKERS = ['Roseburia','Faecalibacterium','Lachnospira','Coprococcus',
           'Bifidobacterium','Enterobacteriaceae','Staphylococcus','Enterococcus',
           'Citrobacter','Bacteroides','Veillonella','Alistipes',
           'Eubacterium_rectale','Streptococcus_thermophilus']
REPS = ['raw','smooth','denoised']


def save(out, name, frame):
    frame.to_csv(out/name, sep='\t', index=False, float_format='%.9g')


def smooth_within_subject(x, s, bandwidth):
    result = np.empty_like(x)
    for _, g in s.groupby('subject_id', sort=False):
        ix = g.index.to_numpy(); t = g.time.to_numpy(float)
        w = np.exp(-.5*((t[:,None]-t[None,:])/bandwidth)**2)
        w[np.abs(t[:,None]-t[None,:]) > 3*bandwidth] = 0
        result[ix] = (w/w.sum(1,keepdims=True))@x[ix]
    return result


def bootstrap_columns(x, rng, draws):
    # Rows are independent subjects. Batching bounds memory for wide taxa tables.
    x = np.asarray(x, float)
    if len(x)<3:
        return np.full(x.shape[1],np.nan), np.full(x.shape[1],np.nan)
    means = []
    for start in range(0,draws,100):
        ix = rng.integers(len(x),size=(min(100,draws-start),len(x)))
        means.append(x[ix].mean(1))
    return np.quantile(np.concatenate(means), [.025,.975], axis=0)


def sign_test_matrix(x, rng, perms):
    # Symmetry-based subject sign-flip test, not a randomized intervention test.
    obs = np.abs(x.mean(0)); exceed = np.zeros(x.shape[1],int)
    for start in range(0,perms,200):
        signs = rng.choice([-1.,1.],size=(min(200,perms-start),len(x)))
        exceed += (np.abs(signs@x/len(x)) >= obs[None]-1e-12).sum(0)
    return (exceed+1)/(perms+1)


def residual_group_test(y, group, nuisance, rng, perms):
    """Freedman-Lane residual permutation, subject rows, conditional exchangeability.

    Returns per-column omnibus tests and a multivariate trace pseudo-F.
    Drops missing rows and rejects singular designs rather than silently adjusting.
    """
    y = np.asarray(y,float)
    if y.ndim==1: y=y[:,None]
    c = np.asarray(nuisance,float)
    if c.ndim==1: c=c[:,None]
    group=np.asarray(group,str)
    valid=np.isfinite(y).all(1)&np.isfinite(c).all(1)&~np.isin(group,['nan','None','unknown','<NA>'])
    y,c,group=y[valid],c[valid],group[valid]
    levels,codes=np.unique(group,return_inverse=True)
    if len(levels)<2 or np.bincount(codes).min()<3:
        return dict(status='insufficient_group_support', n=len(y), valid=valid)
    # Constant nuisance columns add no information; report retained dimensions.
    c=c[:,np.ptp(c,axis=0)>1e-10]
    if c.shape[1]: c=(c-c.mean(0))/c.std(0)
    x0=np.column_stack([np.ones(len(y)),c])
    x1=np.column_stack([x0,np.eye(len(levels))[codes,1:]])
    if np.linalg.matrix_rank(x0)<x0.shape[1] or np.linalg.matrix_rank(x1)<x1.shape[1] or len(y)-x1.shape[1]<5:
        return dict(status='rank_or_residual_df_failure',n=len(y),valid=valid)
    q0=np.linalg.qr(x0,mode='reduced')[0];q1=np.linalg.qr(x1,mode='reduced')[0]
    df1=x1.shape[1]-x0.shape[1]; df2=len(y)-x1.shape[1]
    def statistic(z):
        r0=z-q0@(q0.T@z);r1=z-q1@(q1.T@z)
        ss0=(r0*r0).sum(0);ss1=(r1*r1).sum(0)
        delta=np.maximum(ss0-ss1,0)
        f=(delta/df1)/np.maximum(ss1/df2,1e-20)
        joint=(delta.sum()/df1)/max(ss1.sum()/df2,1e-20)
        return f,joint,delta/np.maximum(ss0,1e-20)
    observed,joint,r2=statistic(y); fit=q0@(q0.T@y); residual=y-fit
    exceed=np.zeros(y.shape[1],int); joint_exceed=0
    for _ in range(perms):
        f,j,_=statistic(fit+residual[rng.permutation(len(y))])
        exceed+=f>=observed-1e-12; joint_exceed+=j>=joint-1e-12
    return dict(status='tested',n=len(y),valid=valid,statistic=observed,
                p=(exceed+1)/(perms+1),partial_r2=r2,joint_f=joint,
                joint_p=(joint_exceed+1)/(perms+1),df1=df1,df2=df2)


def phase_labels(s, cohort):
    if cohort=='FARMM':
        return pd.cut(s.time,[-np.inf,-1e-9,5,8,12,15,np.inf],
            labels=['outside_before','baseline','depletion','early_recovery','late_recovery','outside_after'],right=True).astype(str)
    return pd.cut(s.time,[-np.inf,180,365,730,np.inf],
                  labels=['0_6m','6_12m','12_24m','24m_plus']).astype(str)


def paired_contrasts(reps, s, cohort, taxa, out, rng, perms, boot):
    rows=[]; subject_rows=[]; matrices={}
    for rep,x in reps.items():
        if cohort=='FARMM':
            contrasts=[('depletion_minus_baseline','baseline','depletion'),
                       ('early_recovery_minus_baseline','baseline','early_recovery'),
                       ('late_recovery_minus_baseline','baseline','late_recovery')]
            for name,base,post in contrasts:
                vectors=[];ids=[]
                for sid,g in s.groupby('subject_id',sort=True):
                    a=g.index[g.phase.eq(base)];b=g.index[g.phase.eq(post)]
                    if len(a)>=2 and len(b)>=2:
                        vectors.append(x[b].mean(0)-x[a].mean(0));ids.append(sid)
                matrices[(rep,name)]=(ids,np.array(vectors).reshape(-1,x.shape[1]))
        else:
            vectors=[];ids=[]
            for sid,g in s.groupby('subject_id',sort=True):
                t=g.time.to_numpy(float)/365.25
                if len(g)<4 or np.ptp(t)<.5: continue
                t=t-t.mean();vectors.append(t@x[g.index]/(t@t));ids.append(sid)
            matrices[(rep,'within_subject_slope_per_year')]=(ids,np.array(vectors).reshape(-1,x.shape[1]))
    for (rep,name),(ids,x) in matrices.items():
        if len(ids)<6: continue
        lo,hi=bootstrap_columns(x,rng,boot);p=sign_test_matrix(x,rng,perms)
        for j,t in taxa.iterrows():
            rows.append(dict(representation=rep,contrast=name,taxon_id=t.taxon_id,name=t['name'],
                n_subjects=len(ids),mean_effect=x[:,j].mean(),ci_low=lo[j],ci_high=hi[j],p_signflip=p[j],
                unit='CLR/year' if cohort!='FARMM' else 'CLR',
                inference='conditional_on_representation; subject_sign_symmetry_assumption'))
        for i,sid in enumerate(ids):
            subject_rows.extend(dict(representation=rep,contrast=name,subject_id=sid,
                taxon_id=taxa.taxon_id.iloc[j],effect=v) for j,v in enumerate(x[i]))
    result=pd.DataFrame(rows)
    if len(result):
        # One family per representation, includes every predefined contrast and taxon.
        result['q_bh']=result.groupby('representation').p_signflip.transform(lambda p:bh(p.to_numpy()))
    save(out,'taxon_contrasts.tsv',result)
    save(out,'subject_taxon_contrasts.tsv.gz',pd.DataFrame(subject_rows))
    concord=[]
    if len(result):
        for name,g in result.groupby('contrast'):
            raw=g[g.representation.eq('raw')].set_index('taxon_id')
            for rep in ['smooth','denoised']:
                other=g[g.representation.eq(rep)].set_index('taxon_id'); common=raw.index.intersection(other.index)
                a=raw.loc[common];b=other.loc[common]
                for tid in common:
                    concord.append(dict(contrast=name,representation=rep,taxon_id=tid,
                        raw_effect=a.loc[tid,'mean_effect'],other_effect=b.loc[tid,'mean_effect'],
                        same_sign=np.sign(a.loc[tid,'mean_effect'])==np.sign(b.loc[tid,'mean_effect']),
                        raw_q=a.loc[tid,'q_bh'],other_q=b.loc[tid,'q_bh']))
    save(out,'contrast_concordance.tsv',pd.DataFrame(concord))
    return matrices


def recovery_tables(reps,s,out,rng,boot):
    visits=[]; endpoints=[]
    for rep,x in reps.items():
        composition=np.exp(x-x.max(1,keepdims=True));composition/=composition.sum(1,keepdims=True)
        for sid,g in s.groupby('subject_id',sort=True):
            pre=g.index[g.phase.eq('baseline')]
            if len(pre)<2: continue
            baseline=x[pre].mean(0)
            distance=np.sqrt(np.mean((x[g.index]-baseline)**2,axis=1))
            # Abundance-weighted complement to CLR distance, to expose rare/zero sensitivity.
            composition_baseline=composition[pre].mean(0)
            bray=.5*np.abs(composition[g.index]-composition_baseline).sum(1)
            for ix,d,bc in zip(g.index,distance,bray):
                visits.append(dict(representation=rep,sample_id=s.loc[ix,'sample_id'],subject_id=sid,
                    time=s.loc[ix,'time'],phase=s.loc[ix,'phase'],group=s.loc[ix,'group'],
                    rms_aitchison_distance=d,bray_curtis_distance=bc,n_baseline=len(pre)))
            for phase in ['depletion','early_recovery','late_recovery']:
                mask=g.phase.eq(phase).to_numpy()
                if mask.sum()>=2:
                    endpoints.append(dict(representation=rep,subject_id=sid,group=g.group.iloc[0],
                        endpoint=phase+'_distance',value=distance[mask].mean(),n_visits=int(mask.sum())))
                    endpoints.append(dict(representation=rep,subject_id=sid,group=g.group.iloc[0],
                        endpoint=phase+'_bray_curtis',value=bray[mask].mean(),n_visits=int(mask.sum())))
    v=pd.DataFrame(visits);e=pd.DataFrame(endpoints)
    save(out,'recovery_visits.tsv',v);save(out,'recovery_subject_endpoints.tsv',e)
    curves=[]
    if len(v):
        # First collapse duplicate subject/day records, then bootstrap subjects at each day.
        v['day_bin']=np.floor(v.time+.5).astype(int)
        for metric in ['rms_aitchison_distance','bray_curtis_distance']:
            z=v.groupby(['representation','group','day_bin','subject_id'],as_index=False)[metric].mean()
            for (rep,group,day),g in z.groupby(['representation','group','day_bin']):
                x=g[metric].to_numpy()[:,None];lo,hi=bootstrap_columns(x,rng,boot)
                curves.append(dict(representation=rep,group=group,day=day,metric=metric,n_subjects=len(x),
                    mean_distance=x.mean(),ci_low=lo[0],ci_high=hi[0]))
    save(out,'recovery_curve.tsv',pd.DataFrame(curves))
    return e


def group_inference(reps,s,taxa,matrices,endpoints,out,rng,perms):
    first=s.groupby('subject_id',sort=True).first()
    for c in ['Age','BMI']:
        if c not in first:first[c]=np.nan
        if (s.groupby('subject_id')[c].nunique(dropna=True)>1).any():
            raise ValueError('Time-varying subject covariate: '+c)
    # Raw baseline PC covariates, same for every representation. Descriptive only.
    base=[];ids=[]
    for sid,g in s.groupby('subject_id',sort=True):
        ix=g.index[g.phase.eq('baseline')]
        if len(ix)>=2:base.append(reps['raw'][ix].mean(0));ids.append(sid)
    if len(ids)<8:return
    b=np.array(base);u,d,_=np.linalg.svd(b-b.mean(0),full_matrices=False)
    pc=pd.DataFrame(u[:,:2]*d[:2],index=ids,columns=['baseline_PC1','baseline_PC2'])
    cov=first.join(pc);save(out,'subject_covariates.tsv',cov.reset_index())
    results=[];joint=[];pairs=[]
    jobs=[]
    for (rep,name),(sid,y) in matrices.items():jobs.append((rep,name,sid,y,taxa.taxon_id.to_list()))
    if len(endpoints):
        for (rep,name),g in endpoints.groupby(['representation','endpoint']):
            jobs.append((rep,name,g.subject_id.tolist(),g.value.to_numpy()[:,None],['community_distance']))
    for rep,name,sid,y,names in jobs:
        if len(sid)<8:continue
        c=cov.reindex(sid)
        for adjustment,cols in [('unadjusted',[]),('age_BMI_baseline_PC',['Age','BMI','baseline_PC1','baseline_PC2'])]:
            nuisance=c[cols].apply(pd.to_numeric,errors='coerce').to_numpy(float)
            r=residual_group_test(y,c.group.to_numpy(),nuisance,rng,perms)
            joint.append(dict(representation=rep,contrast=name,adjustment=adjustment,status=r['status'],
                n_subjects=r['n'],pseudo_f=r.get('joint_f',np.nan),p=r.get('joint_p',np.nan)))
            if r['status']!='tested':continue
            for j,tid in enumerate(names):
                if tid!='community_distance':
                    results.append(dict(representation=rep,contrast=name,adjustment=adjustment,taxon_id=tid,
                        n_subjects=r['n'],F=r['statistic'][j],partial_r2=r['partial_r2'][j],p=r['p'][j]))
        # Predefined pairwise descriptive effects and CIs, not selected after omnibus p.
        for left,right in itertools.combinations(sorted(c.group.dropna().unique()),2):
            a=y[c.group.eq(left).to_numpy()];b=y[c.group.eq(right).to_numpy()]
            if min(len(a),len(b))<3:continue
            delta=[]
            for _ in range(1000):
                delta.append(b[rng.integers(len(b),size=len(b))].mean(0)-a[rng.integers(len(a),size=len(a))].mean(0))
            lo,hi=np.quantile(delta,[.025,.975],axis=0)
            for j,tid in enumerate(names):
                pairs.append(dict(representation=rep,contrast=name,taxon_id=tid,comparison=right+' - '+left,
                    n_left=len(a),n_right=len(b),mean_difference=b[:,j].mean()-a[:,j].mean(),
                    ci_low=lo[j],ci_high=hi[j],interval='pointwise_unadjusted_subject_bootstrap'))
    for name,rows in [('group_taxon_tests.tsv',results),('group_community_tests.tsv',joint)]:
        df=pd.DataFrame(rows)
        if len(df):df['q_bh']=df.groupby(['representation','adjustment']).p.transform(lambda x:bh(x.to_numpy()))
        save(out,name,df)
    save(out,'group_pairwise_effects.tsv',pd.DataFrame(pairs))


def ordination(reps,s,taxa,out):
    weights=1/s.subject_id.map(s.subject_id.value_counts()).to_numpy(float)
    weights/=weights.sum()
    rawmean=np.average(reps['raw'],axis=0,weights=weights)
    _,d,vt=np.linalg.svd((reps['raw']-rawmean)*np.sqrt(weights[:,None]),full_matrices=False)
    scores=[];loads=[];variance=[];metrics=[]
    rawdist=pdist(reps['raw'])
    for rep,x in reps.items():
        mu=np.average(x,axis=0,weights=weights)
        _,dr,vr=np.linalg.svd((x-mu)*np.sqrt(weights[:,None]),full_matrices=False)
        for basis,mean,v,sv in [('shared_raw_basis',rawmean,vt,d),('own_basis',mu,vr,dr)]:
            k=min(5,len(v)); z=(x-mean)@v[:k].T
            frame=s.copy();frame['representation']=rep;frame['basis']=basis
            for j in range(k):frame[f'PC{j+1}']=z[:,j]
            scores.append(frame)
            if basis=='own_basis':
                for j in range(k):
                    variance.append(dict(representation=rep,axis=j+1,weighted_variance_fraction=sv[j]**2/max((sv**2).sum(),1e-20)))
                    loads.extend(dict(representation=rep,axis=j+1,taxon_id=tid,loading=val) for tid,val in zip(taxa.taxon_id,v[j]))
                    for cov in ['time','depth']:
                        if cov in s and s[cov].nunique()>1:
                            r=stats.spearmanr(z[:,j],pd.to_numeric(s[cov],errors='coerce')).statistic
                            metrics.append(dict(representation=rep,metric=f'PC{j+1}_spearman_{cov}',value=r,
                                interpretation='descriptive; no samplewise p-values'))
                    for cov in ['batch','group']:
                        if cov in s:
                            grand=np.average(z[:,j],weights=weights);between=0.
                            for level in s[cov].unique():
                                sel=s[cov].eq(level).to_numpy()
                                between+=weights[sel].sum()*(np.average(z[sel,j],weights=weights[sel])-grand)**2
                            total=np.sum(weights*(z[:,j]-grand)**2)
                            metrics.append(dict(representation=rep,metric=f'PC{j+1}_weighted_eta2_{cov}',
                                value=between/max(total,1e-20),interpretation='unadjusted descriptive categorical variance fraction'))
        metrics.append(dict(representation=rep,metric='pairwise_distance_spearman_vs_raw',
            value=stats.spearmanr(rawdist,pdist(x)).statistic,
            interpretation='structure_preservation_not_biological_accuracy'))
    save(out,'ordination_scores.tsv',pd.concat(scores,ignore_index=True));save(out,'ordination_loadings.tsv',pd.DataFrame(loads))
    save(out,'ordination_variance.tsv',pd.DataFrame(variance));save(out,'ordination_diagnostics.tsv',pd.DataFrame(metrics))


def variance_covariance(comp,out):
    names=['baseline','common','individual','event'];x=np.stack([centre(comp[k]) for k in names],axis=-1)
    x=x-x.mean(0);v=np.einsum('njk,njl->kl',x,x)/x.shape[0]
    total=float(v.sum());rows=[]
    for i,a in enumerate(names):
        for j,b in enumerate(names):
            rows.append(dict(component_a=a,component_b=b,covariance_sum_over_taxa=v[i,j],
                fraction_of_total_variance=v[i,j]/max(total,1e-20),
                note='sum of ALL cells including both off-diagonal triangles equals total variance'))
    save(out,'component_covariance.tsv',pd.DataFrame(rows))


def seed_effects(seeds,s,taxa,keep,out):
    rows=[];endpoint=[]
    for fitted in seeds:
        x=centre(fitted['components']['total'][:,keep]);seed=fitted['task']['seed']
        names=['depletion','early_recovery','late_recovery'] if fitted['task']['dataset']=='FARMM' else ['slope_per_year']
        for name in names:
            effects=[]
            for sid,g in s.groupby('subject_id'):
                if name=='slope_per_year':
                    t=g.time.to_numpy(float)/365.25
                    if len(t)<4 or np.ptp(t)<.5:continue
                    t=t-t.mean();effects.append(t@x[g.index]/(t@t))
                else:
                    a=g.index[g.phase.eq('baseline')];b=g.index[g.phase.eq(name)]
                    if min(len(a),len(b))<2:continue
                    effects.append(x[b].mean(0)-x[a].mean(0))
                    endpoint.append(dict(seed=seed,subject_id=sid,group=g.group.iloc[0],phase=name,
                        distance=np.sqrt(np.mean((x[b]-x[a].mean(0))**2,axis=1)).mean()))
            if len(effects):
                for tid,v in zip(taxa.taxon_id,np.mean(effects,axis=0)):
                    rows.append(dict(seed=seed,contrast=name,taxon_id=tid,n_subjects=len(effects),mean_effect=v))
    d=pd.DataFrame(rows)
    save(out,'seed_taxon_effects.tsv',d);save(out,'seed_recovery_endpoints.tsv',pd.DataFrame(endpoint))
    if len(d):
        stability=[]
        for (contrast,tid),g in d.groupby(['contrast','taxon_id']):
            stability.append(dict(contrast=contrast,taxon_id=tid,n_seeds=len(g),
                min_effect=g.mean_effect.min(),max_effect=g.mean_effect.max(),
                same_sign_all_seeds=g.mean_effect.gt(0).all() or g.mean_effect.lt(0).all()))
        save(out,'seed_effect_stability.tsv',pd.DataFrame(stability))


def raw_evidence(counts,reps,s,taxa,out,rng,boot,cohort,original_total):
    # Full tidy evidence includes zeros and input depth; no synthetic count rounding.
    pieces=[]
    for rep,x in reps.items():
        z=pd.DataFrame(dict(sample_id=np.repeat(s.sample_id,len(taxa)),
            subject_id=np.repeat(s.subject_id,len(taxa)),time=np.repeat(s.time,len(taxa)),
            group=np.repeat(s.group,len(taxa)),phase=np.repeat(s.phase,len(taxa)),
            taxon_id=np.tile(taxa.taxon_id,len(s)),representation=rep,clr=x.ravel()))
        if rep=='raw':
            z['count']=counts.ravel();z['detected']=(counts>0).ravel()
            z['relative_abundance']=(counts/original_total[:,None]).ravel()
        pieces.append(z)
    save(out,'abundance_long.tsv.gz',pd.concat(pieces,ignore_index=True))
    rows=[]
    for (group,phase),g in s.groupby(['group','phase']):
        for j,t in taxa.iterrows():
            rows.append(dict(group=group,phase=phase,taxon_id=t.taxon_id,n_visits=len(g),n_subjects=g.subject_id.nunique(),
                detection_fraction=(counts[g.index,j]>0).mean(),median_count=np.median(counts[g.index,j])))
    save(out,'raw_detection.tsv',pd.DataFrame(rows))
    marker_mask=taxa['name'].map(lambda n:any(m.lower() in str(n).lower() for m in MARKERS)).to_numpy()
    curve=[];bins=np.floor(s.time.to_numpy()/(1 if cohort=='FARMM' else 90)).astype(int)
    for rep,x in reps.items():
        for j in np.flatnonzero(marker_mask):
            t=s[['subject_id','group']].copy();t['bin']=bins;t['value']=x[:,j]
            t=t.groupby(['subject_id','group','bin'],as_index=False).value.mean()
            for (group,b),g in t.groupby(['group','bin']):
                v=g.value.to_numpy()[:,None];lo,hi=bootstrap_columns(v,rng,boot)
                curve.append(dict(representation=rep,taxon_id=taxa.taxon_id.iloc[j],name=taxa['name'].iloc[j],
                    group=group,bin_start_day=b*(1 if cohort=='FARMM' else 90),n_subjects=len(g),
                    mean_clr=v.mean(),ci_low=lo[0],ci_high=hi[0]))
    save(out,'marker_trajectories.tsv',pd.DataFrame(curve))


def plots(out,cohort):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    scores=pd.read_csv(out/'ordination_scores.tsv',sep='\t')
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharex=True,sharey=True,constrained_layout=True)
    for ax,rep in zip(axes,REPS):
        g=scores[scores.representation.eq(rep)&scores.basis.eq('shared_raw_basis')]
        sc=ax.scatter(g.PC1,g.PC2,c=g.time,s=9,cmap='viridis',alpha=.65)
        ax.set(title=rep,xlabel='Raw-basis PC1 (CLR)',ylabel='Raw-basis PC2 (CLR)')
    fig.colorbar(sc,ax=axes,label='Observed age / study day (days)')
    fig.suptitle(cohort+': descriptive projection; age used by denoiser')
    for ext in ['png','pdf']:fig.savefig(out/('ordination_shared.'+ext),dpi=220)
    plt.close(fig)
    if cohort=='FARMM':
        d=pd.read_csv(out/'recovery_curve.tsv',sep='\t')
        if not len(d):return
        fig,axes=plt.subplots(2,3,figsize=(13,7),sharey='row',constrained_layout=True)
        for row,metric in enumerate(['rms_aitchison_distance','bray_curtis_distance']):
            for ax,rep in zip(axes[row],REPS):
                for group,g in d[d.representation.eq(rep)&d.metric.eq(metric)].groupby('group'):
                    g=g.sort_values('day');ax.plot(g.day,g.mean_distance,label=group)
                    ax.fill_between(g.day.to_numpy(float),g.ci_low.to_numpy(float),g.ci_high.to_numpy(float),alpha=.15)
                ax.axvspan(6,8,color='grey',alpha=.12)
                ax.set(title=rep,xlabel='Study day',ylabel=metric.replace('_',' ')+' to baseline');ax.legend(fontsize=8)
        fig.suptitle('FARMM: baseline days 0–5; subject bootstrap pointwise 95% intervals')
        for ext in ['png','pdf']:fig.savefig(out/('recovery.'+ext),dpi=220)
        plt.close(fig)


def analyze(bundle,fits,out,cohort,perms=4999,boot=2000,seed=20261018,preview=None,prevalence=.10,pseudocount=.5):
    print(cohort,'loading inputs; prevalence',prevalence,'pseudocount',pseudocount,flush=True)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():raise ValueError('Use a new output directory; completed publication output is immutable')
    data=bundle/'data'/cohort
    s=pd.read_csv(data/'samples.tsv',sep='\t',dtype={'sample_id':str,'subject_id':str}).reset_index(drop=True)
    taxa=pd.read_csv(data/'taxa.tsv',sep='\t',dtype=str)
    from cohort_analysis import short_name
    taxa['name']=taxa.taxonomy.map(short_name)
    with np.load(data/'input.npz') as f:counts=f['Y'][s.tensor_i,s.tensor_t].copy()
    statuses=[];seeds=[];source_hashes={}
    if preview:
        # Explicitly labelled local exploratory export, not eligible for final inference.
        with np.load(preview/cohort/'ensemble_components.npz') as f:
            for key,expected in [('sample_id',s.sample_id.to_numpy(str)),('taxon_id',taxa.taxon_id.to_numpy(str))]:
                if not np.array_equal(f[key],expected):raise ValueError('Preview identity mismatch: '+key)
            comp={k:f[k+'_mean'].copy() for k in ['baseline','common','individual','event','total']}
        source_hashes[str(preview/cohort/'ensemble_components.npz')]=digest(preview/cohort/'ensemble_components.npz')
    else:
        plan=json.loads((bundle/'plan.json').read_text(encoding='utf-8'))
        from event_run import location
        for task in plan['tasks']:
            if task['dataset']!=cohort:continue
            path=location(fits,task)
            if not (path/'done.json').exists():
                statuses.append(dict(seed=task['seed'],status='missing',path=str(path)));continue
            # Corruption is fatal, not silently excluded.
            v=load_seed(path)
            done=json.loads((path/'done.json').read_text(encoding='utf-8'))
            if done['identity']['plan_sha256']!=digest(bundle/'plan.json') or v['task']!=task:
                raise ValueError('Fit belongs to a different frozen plan/task')
            if not v['samples'].sample_id.equals(s.sample_id) or not v['taxa'].taxon_id.equals(taxa.taxon_id):
                raise ValueError('Frozen input identities differ from fit')
            if not np.array_equal(counts,v['counts']):raise ValueError('Raw counts differ from fit')
            seeds.append(v);statuses.append(dict(seed=task['seed'],status='used',path=str(path)))
            source_hashes[str(path/'done.json')]=digest(path/'done.json')
        if not seeds:raise ValueError('No complete fits available')
        comp,_=pool(seeds)
        save(out,'seed_agreement.tsv',seed_agreement(seeds))
        save(out,'seed_quality.tsv',pd.DataFrame([dict(seed=v['task']['seed'],**v['quality']) for v in seeds]))
    save(out,'seed_status.tsv',pd.DataFrame(statuses,columns=['seed','status','path']))
    rng=np.random.default_rng(seed)
    # Identical, raw-only prevalence selection across representations. At least 10% subjects
    # must have one detected observation. Other thresholds are exported as sensitivity masks.
    detected=pd.DataFrame(counts>0).groupby(s.subject_id).max().mean(0).to_numpy()
    filt=taxa.copy();filt['subject_prevalence']=detected
    for threshold in [.05,.10,.20]:filt[f'keep_{int(threshold*100)}pct']=detected>=threshold
    save(out,'feature_filter.tsv',filt)
    keep=detected>=prevalence
    if keep.sum()<3:raise ValueError('Fewer than three taxa pass prespecified filter')
    taxa=taxa.loc[keep].reset_index(drop=True);counts=counts[:,keep]
    # Denominator for relative abundance must remain original, not filtered subcomposition.
    with np.load(data/'input.npz') as f:original_counts=f['Y'][s.tensor_i,s.tensor_t]
    raw=centre(np.log(counts+pseudocount)); bandwidth=1.5 if cohort=='FARMM' else 30.4375
    reps=dict(raw=raw,smooth=smooth_within_subject(raw,s,bandwidth),denoised=centre(comp['total'][:,keep]))
    s['phase']=phase_labels(s,cohort)
    if (s.groupby('subject_id').group.nunique(dropna=False)>1).any():raise ValueError('Group changes within subject')
    save(out,'samples.tsv',s);save(out,'taxa.tsv',taxa)
    batch=s.groupby(['group','batch'],as_index=False).agg(n_samples=('sample_id','size'),n_subjects=('subject_id','nunique'))
    save(out,'batch_group_support.tsv',batch)
    # Preserve both original and selected library totals for low-depth auditing.
    audit=s[['sample_id','subject_id','time','group','batch','depth']].copy()
    audit['original_count_sum']=original_counts.sum(1);audit['retained_count_sum']=counts.sum(1)
    audit['retained_zero_fraction']=(counts==0).mean(1)
    if 'copy_num_per_gram_feces' in s:
        audit['copy_num_per_gram_feces']=pd.to_numeric(s.copy_num_per_gram_feces,errors='coerce')
    save(out,'sample_qc.tsv',audit)
    print(cohort,'exporting raw evidence and subject contrasts',flush=True)
    raw_evidence(counts,reps,s,taxa,out,rng,boot,cohort,original_counts.sum(1))
    matrices=paired_contrasts(reps,s,cohort,taxa,out,rng,perms,boot)
    if seeds:seed_effects(seeds,s,taxa,keep,out)
    if cohort=='FARMM':
        print(cohort,'recovery and adjusted group tests',flush=True)
        endpoints=recovery_tables(reps,s,out,rng,boot)
        group_inference(reps,s,taxa,matrices,endpoints,out,rng,perms)
    print(cohort,'ordination and diagnostic figures',flush=True)
    ordination(reps,s,taxa,out);variance_covariance(comp,out);plots(out,cohort)
    write_json(out/'analysis_config.json',dict(cohort=cohort,permutations=perms,bootstrap=boot,seed=seed,
        subject_prevalence=prevalence,pseudocount=pseudocount,smoothing_bandwidth_days=bandwidth,
        preview=bool(preview),n_subjects=s.subject_id.nunique(),n_samples=len(s),n_taxa=len(taxa),
        source_hashes=source_hashes,source_code_sha256=digest(Path(__file__))))
    (out/'INTERPRETATION.md').write_text(
        '# Interpretation boundaries\n\n'
        '- Primary evidence is raw-supported subject-level recovery or age-associated change.\n'
        '- CLR is relative, not absolute abundance. All three representations share a raw-only feature filter.\n'
        '- Smooth uses all within-subject visits, bandwidth fixed in advance; it is descriptive, not forecasting.\n'
        '- Denoising used observed ages and all subjects. PCA/age association is descriptive, not external validation.\n'
        '- Bootstrap and permutation condition on the fitted representation; denoiser uncertainty is not propagated.\n'
        '- Subject sign-flips assume symmetric subject effects. Freedman–Lane assumes exchangeable residuals.\n'
        '- Diet associations are not randomized vegan-vs-other causal contrasts. Adjusted tests control Age, BMI, raw baseline PC1/2 only.\n'
        '- Pointwise bootstrap intervals and pairwise effects are not simultaneous confidence statements.\n'
        '- More discoveries or greater separation are not evidence of greater accuracy.\n'
        '- Inference requires seed and validation audits; missing seeds are explicitly reported.\n'
        + ('- PREVIEW ONLY: supplied ensemble; not verified against completed fit artifacts.\n' if preview else ''),encoding='utf-8')
    write_json(out/'complete.json',dict(status='preview' if preview else 'complete',
        files={p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!='complete.json'}))
    print(cohort,'saved',out,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,default=Path(__file__).resolve().parent.parent)
    p.add_argument('--fits',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cohort',choices=['FARMM','ECAM','DIABIMMUNE'],required=True)
    p.add_argument('--permutations',type=int,default=4999);p.add_argument('--bootstrap',type=int,default=2000)
    p.add_argument('--preview-ensemble',type=Path)
    p.add_argument('--prevalence',type=float,default=.10)
    p.add_argument('--pseudocount',type=float,default=.5)
    a=p.parse_args()
    if not a.fits and not a.preview_ensemble:p.error('--fits required unless --preview-ensemble')
    if a.permutations<99 or a.bootstrap<100:p.error('At least 99 permutations and 100 bootstrap draws')
    if not 0<a.prevalence<=1 or a.pseudocount<=0:p.error('Invalid prevalence or pseudocount')
    from event_run import verify
    verify(a.bundle.resolve())
    analyze(a.bundle.resolve(),a.fits,a.out.resolve(),a.cohort,a.permutations,a.bootstrap,
            preview=a.preview_ensemble,prevalence=a.prevalence,pseudocount=a.pseudocount)


if __name__=='__main__':main()
