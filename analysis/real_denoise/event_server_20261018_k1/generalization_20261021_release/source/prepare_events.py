"""Freeze the real-data event workflow: 3 cohorts x 3 seeds, fixed budget."""
from pathlib import Path
import argparse
import json
import shutil
import tarfile
import numpy as np
import pandas as pd
from common import digest,write_json,write_npz
from biology_metadata import prepare as prepare_metadata
from event_design import event_design,audit_design

SOURCE=['common.py','event_design.py','prepare_events.py','event_run.py','event_export.py','cohort_analysis.py',
        'event_submit.sh','event_job.sbatch','event_after.sbatch','event_requirements.txt','EVENT_README.md',
        'test_events.py','biology_metadata.py']

# m4_flex (code_v3 run_validation: adaptive budget + joint_rank 4 + hierarchical
# observation + flexible event) with a FIXED budget instead of adaptive stopping:
# early stopping can never trigger (tol 1e-12); the best-validation VI state is kept.
CONFIG=dict(adaptive_budget=False,early_stopping=True,convergence_tol=1e-12,joint_rank=4,
    hierarchical_observation=True,flexible_event=True,device='cpu',map_steps=1500,map_starts=2,map_lr=.02,
    vi_steps=4000,vi_lr=.003,posterior_samples=200,check_every=50,validation_samples=12,vi_max_recoveries=8)
# No unbounded post-event 'ramp': with near-synchronous event dates it is a linear
# age/time trend and would absorb the common trajectory. flex0-3 are appended.
EVENT_BASIS=['decay','pulse','step','delayed_decay']
DATASET_CONFIG={c:dict(event_basis=EVENT_BASIS) for c in ['DIABIMMUNE','FARMM','ECAM']}


def prepare(args):
    if args.out.exists() or args.out.with_suffix('.tar.gz').exists():raise ValueError('Output must be new')
    old=json.loads((args.prepared/'plan.json').read_text(encoding='utf-8'))
    if digest(args.prepared/'plan.json')!=(args.prepared/'PLAN.sha256').read_text().strip():raise ValueError('Original plan checksum')
    for name,h in old['files'].items():
        if digest(args.prepared/name)!=h:raise ValueError('Original frozen input changed: '+name)
    args.out.mkdir(parents=True);src=args.out/'source';src.mkdir()
    for name in SOURCE:(src/name).write_bytes(Path(__file__).with_name(name).read_bytes().replace(b'\r\n',b'\n'))
    shutil.copyfile(args.prepared/'source/event_aware_denoise.py',src/'event_aware_denoise.py')
    prepare_metadata(args.raw,args.out/'metadata')
    tasks=[];datasets=[]
    for cohort in ['DIABIMMUNE','FARMM','ECAM']:
        path=args.out/'data'/cohort;path.mkdir(parents=True)
        for name in ['input.npz','samples.tsv','taxa.tsv']:
            shutil.copyfile(args.prepared/'data'/cohort/name,path/name)
        s=pd.read_csv(path/'samples.tsv',sep='\t',dtype={'sample_id':str,'subject_id':str})
        m=pd.read_csv(args.out/'metadata'/f'{cohort}_metadata.tsv',sep='\t',dtype={'sample_id':str,'subject_id':str})
        times,names,long,note=event_design(cohort,s,m)
        write_npz(path/'events.npz',event_times=times,event_names=np.array(names,dtype=str))
        long.to_csv(path/'events.tsv',sep='\t',index=False)
        support,overlap,distribution=audit_design(s,times,names,cohort)
        for label,frame in [('support',support),('overlap',overlap),('dates',distribution)]:
            if len(frame.columns):frame.to_csv(path/f'event_{label}.tsv',sep='\t',index=False)
        datasets.append(dict(dataset=cohort,n_subjects=s.subject_id.nunique(),n_samples=len(s),event_names=names,note=note))
        arm='noevent' if cohort=='ECAM' else 'event'
        for seed in args.seeds:tasks.append(dict(id=len(tasks),dataset=cohort,arm=arm,seed=seed))
    plan=dict(schema='real-events-v2',seeds=args.seeds,datasets=datasets,tasks=tasks,config=CONFIG,
        dataset_config=DATASET_CONFIG,resources=dict(cpus=2,mem='6G',time='48:00:00'),
        source_plan_sha256=digest(args.prepared/'plan.json'),
        reconstruction_label='模型条件下移除事件分量的重建；不代表未吃辅食或未用抗生素的因果反事实',
        files={str(p.relative_to(args.out)).replace('\\','/'):digest(p) for p in args.out.rglob('*') if p.is_file()})
    write_json(args.out/'plan.json',plan);(args.out/'PLAN.sha256').write_text(digest(args.out/'plan.json')+'\n')
    archive=args.out.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz') as tf:tf.add(args.out,arcname=args.out.name)
    Path(str(archive)+'.sha256').write_text(digest(archive)+'  '+archive.name+'\n',encoding='ascii')
    print(f'Prepared {len(tasks)} tasks');print(archive)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs=3,default=[2026,2027,2028])
    a=p.parse_args()
    if len(set(a.seeds))!=3:p.error('Three distinct seeds required')
    prepare(a)
