"""Separate refit sensitivity tasks; never pool these with primary seeds."""
from pathlib import Path
import argparse
import copy
import json
import os
import shutil
import time
import traceback
import numpy as np
import pandas as pd
import torch
from common import digest,write_json,write_npz
from event_run import verify,task_inputs,config_for,quality,acquire,complete_valid,location
from event_aware_denoise import DenoiseConfig,EventAwareDenoiser
from event_export import export_components
from cohort_analysis import load_seed,centre,event_inference,trajectory


def tasks(plan):
    rows=[]
    for primary in plan['tasks']:
        rows.append(dict(primary,validation='extended_budget',overrides=dict(map_steps=3000,vi_steps=8000)))
        if primary['dataset']!='ECAM':
            rows.append(dict(primary,arm='noevent',validation='noevent',overrides={}))
    for cohort in ['FARMM','DIABIMMUNE']:
        for multiplier in [.5,2.]:
            rows.append(dict(dataset=cohort,arm='event',seed=2026,validation=f'prior_x{multiplier}',
                overrides=dict(event_global_scale=.05*multiplier,zero_global_scale=.03*multiplier)))
        rows.append(dict(dataset=cohort,arm='event',seed=2026,validation='flex_basis',
            overrides=dict(event_basis=['decay','pulse','step','delayed_decay','flex0','flex1','flex2','flex3'])))
    for i in range(19):
        rows.append(dict(dataset='DIABIMMUNE',arm='event',seed=2026,validation='permuted_date',
            permutation_seed=10000+i,overrides={}))
    for i,r in enumerate(rows):r['id']=i
    return rows


def target_path(out,task,smoke=False):
    return out/('validation_smoke' if smoke else 'validation_fits')/f"{task['id']:02d}_{task['dataset']}_{task['validation']}_{task['seed']}"


def permute_dates(events,s,seed):
    rng=np.random.default_rng(seed);result=events.copy()
    subjects=s.sort_values('tensor_i').drop_duplicates('tensor_i')
    if not np.array_equal(subjects.tensor_i,np.arange(len(events))):raise ValueError('Subject indexing mismatch')
    # Preserve antibiotic-group timing distributions. This is an observational
    # timing sensitivity diagnostic, not a randomized causal test.
    for group in subjects.group.unique():
        ix=subjects.loc[subjects.group.eq(group),'tensor_i'].to_numpy()
        result[ix]=events[rng.permutation(ix)]
    return result


def run(bundle,out,task,plan,smoke=False):
    target=target_path(out,task,smoke);target.mkdir(parents=True,exist_ok=True)
    cfg=config_for(task,plan,smoke);cfg.update(task['overrides'])
    if smoke:cfg.update(map_steps=2,map_starts=1,vi_steps=3,posterior_samples=6,validation_samples=2)
    identity=dict(plan_sha256=digest(bundle/'plan.json'),task=task,smoke=smoke,config=cfg,
        validation_source_sha256=digest(Path(__file__)))
    lock=target/'.lock';acquire(lock)
    try:
        if complete_valid(target,identity):print('Verified complete',target);return
        a,s,taxa,events,names=task_inputs(bundle,task)
        original_events=events.copy()
        if task['validation']=='permuted_date':events=permute_dates(events,s,task['permutation_seed'])
        mapping=s.sort_values('tensor_i').drop_duplicates('tensor_i')[['subject_id','tensor_i','group']].copy()
        mapping['original_event_day']=original_events[:,0];mapping['fitted_event_day']=events[:,0]
        mapping.to_csv(target/'event_date_mapping.tsv',sep='\t',index=False)
        write_json(target/'config.json',cfg)
        start=time.time()
        estimator=EventAwareDenoiser(DenoiseConfig(**cfg)).fit(a['Y'],a['times'],a['depth'],a['batch'],events,mask=a['mask'])
        estimator.save_checkpoint(target/'fit.pt')
        means,_=export_components(estimator,s,taxa,events,names,target)
        s.to_csv(target/'samples.tsv',sep='\t',index=False);taxa.to_csv(target/'taxa.tsv',sep='\t',index=False)
        write_npz(target/'event_input.npz',event_times=events,event_names=np.array(names,dtype=str))
        write_npz(target/'posterior_observed.npz',raw_counts=a['Y'][s.tensor_i,s.tensor_t],**means)
        q=quality(estimator.metadata_,smoke);write_json(target/'quality.json',q)
        write_json(target/'diagnostics.json',dict(estimator.metadata_,elapsed_seconds=time.time()-start))
        (target/'failure.json').unlink(missing_ok=True)
        write_json(target/'done.json',dict(identity=identity,status=q['status'],usable=q['usable'],
            files={p.name:digest(p) for p in target.iterdir() if p.is_file() and p.name!='done.json'}))
    except Exception:
        write_json(target/'failure.json',dict(traceback=traceback.format_exc()));raise
    finally:shutil.rmtree(lock,ignore_errors=True)


def event_energy(fit):
    s=fit['samples'];dates=fit['events'][s.tensor_i.to_numpy(),0]
    rel=s.time.to_numpy()-dates
    hi=180 if fit['task']['dataset']=='DIABIMMUNE' else 7.5
    values=np.mean(centre(fit['components']['event'])**2,axis=1)
    z=pd.DataFrame(dict(subject=s.subject_id,value=values,eligible=(rel>=0)&(rel<=hi)))
    per=z[z.eligible].groupby('subject').value.mean()
    return float(per.mean()),len(per)


def summarize(bundle,fits,out,plan):
    dest=out/'validation_summary';dest.mkdir(parents=True,exist_ok=True)
    (dest/'complete.json').unlink(missing_ok=True)
    primary={};status=[];summary=[];effects=[];energies=[];recovery=[];quality_rows=[]
    for task in plan['tasks']:
        path=location(fits,task)
        if (path/'done.json').exists():
            f=load_seed(path);primary[(task['dataset'],task['seed'])]=f
            energy,n=event_energy(f)
            energies.append(dict(cohort=task['dataset'],kind='primary',seed=task['seed'],energy=energy,n_subjects=n))
            if np.isfinite(f['events']).any():
                _,e=event_inference(f['draws'],f['taxa']);e['validation']='primary';e['seed']=task['seed'];e['task_id']=task['id'];e['cohort']=task['dataset'];effects.append(e)
            if task['dataset']=='FARMM':
                s=f['samples'];x=centre(f['components']['total'])
                for sid,g in s.groupby('subject_id'):
                    pre=g.index[g.time.between(0,5)];late=g.index[g.time.between(13,15)]
                    if min(len(pre),len(late))<2:continue
                    recovery.append(dict(validation='primary',seed=task['seed'],subject_id=sid,group=g.group.iloc[0],
                        late_distance=np.sqrt(np.mean((x[late]-x[pre].mean(0))**2,axis=1)).mean(),
                        feature_set='all_original_taxa; compare_validation_to_validation'))
    for task in tasks(plan):
        path=target_path(out,task)
        if not (path/'done.json').exists():
            status.append(dict(task,status='missing_or_failed',path=str(path)));continue
        f=load_seed(path);status.append(dict(task,status='complete',path=str(path)))
        q=json.loads((path/'quality.json').read_text());quality_rows.append(dict(task,**q))
        ref=primary.get((task['dataset'],task['seed']))
        row=dict(task)
        if ref:
            for k in ['common','individual','event','total']:
                x=centre(ref['components'][k]).ravel();y=centre(f['components'][k]).ravel()
                row[k+'_correlation']=np.corrcoef(x,y)[0,1] if min(x.std(),y.std())>1e-12 else np.nan
                row[k+'_relative_rmse']=np.sqrt(np.mean((x-y)**2))/max(np.sqrt(np.mean(x*x)),1e-12)
        summary.append(row)
        if np.isfinite(f['events']).any():
            energy,n=event_energy(f)
            energies.append(dict(cohort=task['dataset'],kind=task['validation'],seed=task['seed'],
                permutation_seed=task.get('permutation_seed'),energy=energy,n_subjects=n))
            _,e=event_inference(f['draws'],f['taxa']);e['validation']=task['validation'];e['seed']=task['seed'];e['task_id']=task['id'];e['cohort']=task['dataset']
            effects.append(e)
        if task['dataset']=='FARMM':
            s=f['samples'];x=centre(f['components']['total'])
            for sid,g in s.groupby('subject_id'):
                pre=g.index[g.time.between(0,5)];late=g.index[g.time.between(13,15)]
                if min(len(pre),len(late))<2:continue
                recovery.append(dict(validation=task['validation'],seed=task['seed'],subject_id=sid,group=g.group.iloc[0],
                    late_distance=np.sqrt(np.mean((x[late]-x[pre].mean(0))**2,axis=1)).mean(),
                    feature_set='all_original_taxa; compare_validation_to_validation'))
    for name,rows in [('validation_status.tsv',status),('component_stability.tsv',summary),
                      ('event_energy.tsv',energies),('recovery_stability.tsv',recovery),('quality.tsv',quality_rows)]:
        pd.DataFrame(rows).to_csv(dest/name,sep='\t',index=False)
    if effects:pd.concat(effects,ignore_index=True).to_csv(dest/'event_taxa_by_refit.tsv.gz',sep='\t',index=False)
    e=pd.DataFrame(energies);null=e[e.kind.eq('permuted_date')].energy.to_numpy() if len(e) else np.array([])
    observed=e[e.cohort.eq('DIABIMMUNE')&e.kind.eq('primary')&e.seed.eq(2026)] if len(e) else e
    complete=len(status)>0 and all(r['status']=='complete' for r in status)
    write_json(dest/'timing_negative_control.json',dict(n_permutations=len(null),expected=19,
        observed_energy=float(observed.energy.iloc[0]) if len(observed) else None,
        empirical_upper_tail=(1+int((null>=observed.energy.iloc[0]).sum()))/20 if len(null)==19 and len(observed) else None,
        interpretation='Exploratory within-antibiotic-group date permutation; smallest tail fraction 0.05. Not causal identification or FDR calibration.'))
    if complete:
        write_json(dest/'complete.json',dict(status='complete',files={p.name:digest(p) for p in dest.iterdir() if p.is_file() and p.name!='complete.json'}))
    else:raise RuntimeError('Validation incomplete: partial tables saved; inspect validation_status.tsv')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['list','run','summarize'])
    p.add_argument('--bundle',type=Path,default=Path(__file__).resolve().parent.parent)
    p.add_argument('--fits',type=Path);p.add_argument('--out',type=Path);p.add_argument('--task',type=int)
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    plan=verify(a.bundle.resolve())
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','2')))
    torch.set_num_interop_threads(1)
    if a.command=='list':
        print(json.dumps(tasks(plan),ensure_ascii=False,indent=2));return
    if a.out is None:p.error('--out required')
    if a.command=='run':
        if a.task is None or not 0<=a.task<len(tasks(plan)):p.error('Invalid --task')
        run(a.bundle.resolve(),a.out.resolve(),tasks(plan)[a.task],plan,a.smoke)
    else:
        if a.fits is None:p.error('--fits required')
        summarize(a.bundle.resolve(),a.fits.resolve(),a.out.resolve(),plan)


if __name__=='__main__':main()
