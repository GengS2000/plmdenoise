"""Fit one frozen real-data task with a FIXED budget; no convergence gate.

Each seed runs exactly map_starts x map_steps MAP steps and vi_steps VI steps,
then keeps the VI state with the best validation ELBO. Stationarity flags are
recorded as diagnostics only; every numerically finite fit goes downstream.
"""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
import pandas as pd
import torch
from common import digest,write_json,write_npz
from event_export import export_components
from event_aware_denoise import DenoiseConfig,EventAwareDenoiser

FROZEN_SOURCE=['event_run.py','event_design.py','event_export.py','event_aware_denoise.py','common.py','cohort_analysis.py']


def verify(bundle):
    p=bundle/'plan.json'
    if digest(p)!=(bundle/'PLAN.sha256').read_text().strip():raise ValueError('Plan checksum mismatch')
    plan=json.loads(p.read_text(encoding='utf-8'))
    for name,h in plan['files'].items():
        f=(bundle/name).resolve()
        if not f.is_relative_to(bundle.resolve()) or digest(f)!=h:raise ValueError('Frozen payload changed: '+name)
    for name in FROZEN_SOURCE:
        if digest(Path(__file__).with_name(name))!=plan['files']['source/'+name]:
            raise ValueError('Use the frozen source, not a different source revision')
    return plan


def location(out,task,smoke=False):
    return out/('smoke' if smoke else 'tasks')/f"{task['id']:02d}_{task['dataset']}_{task['arm']}_{task['seed']}"


def quality(metadata,smoke=False):
    """Numerical completeness only. Stationarity is reported, never used as a gate."""
    validations=np.array([r['negative_elbo_per_cell'] for r in metadata['vi_validation']])
    finite=bool(len(validations)>0 and np.isfinite(validations).all() and np.isfinite(metadata['map_best_objective']))
    best=min(metadata['map_histories'],key=lambda h:h['objective'][-1])
    usable=finite and not smoke
    return dict(usable=usable,status='smoke_only' if smoke else 'complete' if finite else 'numerical_failure',
        budget='fixed',map_steps_per_start=best['steps'],vi_steps_run=metadata['vi_steps_run'],
        vi_selected_step=metadata['vi_selected_step'],
        diagnostic_map_stationary=bool(metadata['map_stable']),diagnostic_vi_stationary=bool(metadata['vi_stable']),
        map_rejected_fraction=best['rejected_steps']/max(best['steps'],1),
        vi_validation_first_last_best=[float(validations[0]),float(validations[-1]),float(validations.min())] if finite else None,
        vi_rollbacks=len(metadata['vi_recoveries']),
        meaning='Fixed computation per seed; stationarity flags are diagnostics, not admission criteria.')


def task_inputs(bundle,task):
    path=bundle/'data'/task['dataset']
    with np.load(path/'input.npz',allow_pickle=False) as d:arrays={k:d[k].copy() for k in ['Y','times','depth','batch','mask']}
    s=pd.read_csv(path/'samples.tsv',sep='\t',dtype={'sample_id':str,'subject_id':str})
    taxa=pd.read_csv(path/'taxa.tsv',sep='\t',dtype=str)
    with np.load(path/'events.npz',allow_pickle=False) as d:events=d['event_times'].copy();names=d['event_names'].tolist()
    if task['arm']=='noevent':events[:]=np.nan
    return arrays,s,taxa,events,names


def config_for(task,plan,smoke=False):
    cfg=dict(plan['config'],**plan['dataset_config'].get(task['dataset'],{}),seed=task['seed'],verbose=True)
    if smoke:cfg.update(map_steps=2,map_starts=1,vi_steps=3,posterior_samples=6,validation_samples=2)
    return cfg


def complete_valid(target,identity):
    file=target/'done.json'
    if not file.exists():return False
    d=json.loads(file.read_text(encoding='utf-8'))
    if d['identity']!=identity:raise ValueError('Existing output belongs to another task/plan')
    for name,h in d['files'].items():
        if digest(target/name)!=h:raise ValueError('Completed output corrupted: '+name)
    return True


def acquire(lock):
    """Directory lock tagged with the Slurm job; stale locks of dead jobs are cleared."""
    job=os.environ.get('SLURM_JOB_ID','local')
    try:lock.mkdir()
    except FileExistsError:
        owner=(lock/'owner').read_text().strip() if (lock/'owner').exists() else ''
        alive=True
        if owner and owner!='local' and shutil.which('squeue'):
            alive=bool(subprocess.run(['squeue','-h','-j',owner],capture_output=True,text=True).stdout.strip())
        if alive:raise RuntimeError(f'Task locked by job {owner or "unknown"}; remove {lock} only if that job is gone')
        print(f'Removing stale lock of finished job {owner}',flush=True)
        shutil.rmtree(lock);lock.mkdir()
    (lock/'owner').write_text(job)


def fit_task(bundle,out,task,plan,smoke=False):
    target=location(out,task,smoke);target.mkdir(parents=True,exist_ok=True)
    identity=dict(plan_sha256=digest(bundle/'plan.json'),task=task,smoke=smoke)
    lock=target/'.lock';acquire(lock)
    try:
        if complete_valid(target,identity):print('Verified complete',target);return
        a,s,taxa,events,names=task_inputs(bundle,task)
        cfg=config_for(task,plan,smoke);started=time.time()
        write_json(target/'running.json',dict(identity=identity,pid=os.getpid(),job=os.environ.get('SLURM_JOB_ID'),config=cfg))
        est=EventAwareDenoiser(DenoiseConfig(**cfg)).fit(a['Y'],a['times'],a['depth'],a['batch'],events,mask=a['mask'])
        fitted=time.time()
        est.save_checkpoint(target/'fit.pt')
        qc=quality(est.metadata_,smoke)
        s.to_csv(target/'samples.tsv',sep='\t',index=False);taxa.to_csv(target/'taxa.tsv',sep='\t',index=False)
        write_npz(target/'event_input.npz',event_times=events,event_names=np.array(names,dtype=str))
        means,_=export_components(est,s,taxa,events,names,target)
        counts=a['Y'][s.tensor_i.to_numpy(),s.tensor_t.to_numpy()]
        means.update(raw_counts=counts,sample_id=s.sample_id.to_numpy(dtype=str),taxon_id=taxa.taxon_id.to_numpy(dtype=str))
        write_npz(target/'posterior_observed.npz',**means)
        metadata=dict(est.metadata_,quality=qc,fit_seconds=fitted-started,export_seconds=time.time()-fitted,smoke=smoke)
        if sys.platform!='win32':
            import resource
            metadata['peak_rss_mib_linux']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        metadata['runtime']=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,torch=torch.__version__)
        write_json(target/'diagnostics.json',metadata)
        write_json(target/'quality.json',qc)
        for f in ['running.json','failure.json']:(target/f).unlink(missing_ok=True)
        payload={p.name:digest(p) for p in target.iterdir() if p.is_file() and p.name!='done.json'}
        write_json(target/'done.json',dict(identity=identity,status=qc['status'],usable=qc['usable'],files=payload))
        print('Completed',target,qc['status'],f"fit {metadata['fit_seconds']/3600:.2f} h",flush=True)
    except Exception:
        write_json(target/'failure.json',dict(identity=identity,traceback=traceback.format_exc()));raise
    finally:shutil.rmtree(lock,ignore_errors=True)


def pilot(bundle,out,plan):
    """Short timing fits per cohort; projects the frozen full budget. Not used downstream."""
    rows=[];target=out/'pilot';target.mkdir(parents=True,exist_ok=True)
    for data in plan['datasets']:
        task=next(t for t in plan['tasks'] if t['dataset']==data['dataset'])
        a,s,taxa,events,names=task_inputs(bundle,task);full=config_for(task,plan)
        def timed(map_steps,vi_steps,draws=2):
            cfg=dict(full,map_steps=map_steps,map_starts=1,vi_steps=vi_steps,posterior_samples=draws,verbose=False)
            start=time.time()
            est=EventAwareDenoiser(DenoiseConfig(**cfg)).fit(a['Y'],a['times'],a['depth'],a['batch'],events,mask=a['mask'])
            return est,time.time()-start
        _,base=timed(1,1);_,m=timed(41,1);est,v=timed(1,81)
        per_map=max(m-base,0)/40;per_vi=max(v-base,0)/80
        start=time.time();sub=target/data['dataset'];sub.mkdir(exist_ok=True)
        est.config.posterior_samples=4;export_components(est,s,taxa,events,names,sub)
        per_draw=(time.time()-start)/4
        hours=(full['map_starts']*full['map_steps']*per_map+full['vi_steps']*per_vi+full['posterior_samples']*per_draw+base)/3600
        rows.append(dict(dataset=data['dataset'],seconds_per_map_step=per_map,seconds_per_vi_step=per_vi,
            seconds_per_export_draw=per_draw,setup_seconds=base,projected_hours_full_budget=hours,
            map_steps=full['map_steps'],map_starts=full['map_starts'],vi_steps=full['vi_steps'],
            posterior_samples=full['posterior_samples']))
        print(rows[-1],flush=True)
    if sys.platform!='win32':
        import resource
        for r in rows:r['peak_rss_mib_linux_whole_pilot']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    pd.DataFrame(rows).to_csv(target/'pilot_timing.tsv',sep='\t',index=False)
    print('Projection excludes MAP backtracking growth late in optimisation; treat as a lower bound.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['check','run','pilot'])
    p.add_argument('--bundle',type=Path,default=Path(__file__).resolve().parent.parent)
    p.add_argument('--out',type=Path)
    p.add_argument('--task',type=int,default=0)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args();a.bundle=a.bundle.resolve()
    if a.command!='check':
        if a.out is None:p.error('--out required')
        a.out=a.out.resolve()
        if a.out.is_relative_to(a.bundle) or a.bundle.is_relative_to(a.out):p.error('Separate bundle and results trees')
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','2')));torch.set_num_interop_threads(1)
    plan=verify(a.bundle)
    if a.command=='check':
        for data in plan['datasets']:
            task=next(t for t in plan['tasks'] if t['dataset']==data['dataset'])
            x,s,_,events,_=task_inputs(a.bundle,task)
            assert np.array_equal(x['times'][s.tensor_i,s.tensor_t],s.time)
            assert events.shape[0]==x['Y'].shape[0]
            print(data['dataset'],x['Y'].shape,data['event_names'],data['note'])
        print('Verified',len(plan['tasks']),'tasks')
    elif a.command=='pilot':pilot(a.bundle,a.out,plan)
    else:
        if not 0<=a.task<len(plan['tasks']):p.error('Invalid task index')
        fit_task(a.bundle,a.out,plan['tasks'][a.task],plan,a.smoke)


if __name__=='__main__':main()
