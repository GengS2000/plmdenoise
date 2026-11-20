"""Slurm dispatcher selecting dataset families or individual oral sites."""
from pathlib import Path
import argparse
import json
import os
import shlex
import subprocess
import sys
from common import write_json,digest


def select(plan,requested):
    known={d['dataset'] for d in plan['datasets']}
    aliases={d.get('family',d['dataset']) for d in plan['datasets']}
    unknown=set(requested)-known-aliases-{'ALL'}
    if unknown:raise ValueError('Unknown/unprepared datasets: '+', '.join(sorted(unknown)))
    result=[d['dataset'] for d in plan['datasets'] if 'ALL' in requested or d['dataset'] in requested or d.get('family',d['dataset']) in requested]
    if not result:raise ValueError('Empty selection')
    return result


def separate(*paths):
    paths=[Path(p).resolve() for p in paths]
    for i,a in enumerate(paths):
        for b in paths[i+1:]:
            if a.is_relative_to(b) or b.is_relative_to(a):raise ValueError('Use separate bundle, fit and analysis trees')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,default=Path(__file__).resolve().parent.parent)
    p.add_argument('--datasets',nargs='+',required=True)
    p.add_argument('--fits',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--stage',choices=['all','fit','analyze'],default='all')
    p.add_argument('--python',default=sys.executable);p.add_argument('--partition',default='hebhcnormal02')
    p.add_argument('--max-parallel',type=int,default=3);p.add_argument('--dry-run',action='store_true')
    p.add_argument('--dpi',type=int,default=600)
    a=p.parse_args();a.bundle=a.bundle.resolve();a.fits=a.fits.resolve();a.out=a.out.resolve()
    if a.max_parallel<1:p.error('--max-parallel must be positive')
    separate(a.bundle,a.fits,a.out)
    from event_run import verify
    plan=verify(a.bundle);cohorts=select(plan,a.datasets)
    tasks=[t['id'] for t in plan['tasks'] if t['dataset'] in cohorts]
    print(json.dumps(dict(cohorts=cohorts,fit_task_ids=tasks,stage=a.stage,dry_run=a.dry_run),indent=2))
    src=a.bundle/'source';jobs=a.out/'submission';logs=a.out/'logs'
    def command(script,*args):return shlex.join([a.python,str(src/script),*map(str,args)])
    fitcommand=command('event_run.py','run','--bundle',a.bundle,'--out',a.fits)+' --task "$SLURM_ARRAY_TASK_ID"'
    analysis=[]
    new=plan['schema']=='generalization-k1-v1'
    for c in cohorts:
        if new:cmd=command('generalization_analysis.py','--bundle',a.bundle,'--fits',a.fits,'--out',a.out/c,'--cohort',c,'--dpi',a.dpi)
        else:cmd=command('existing_application.py','--bundle',a.bundle,'--fits',a.fits,'--out',a.out/c,'--cohort',c,'--dpi',a.dpi)
        analysis.append((c,cmd))
    for c,cmd in analysis:print(c+': '+cmd)
    if a.dry_run:
        print('DRY RUN: no directories created and nothing submitted. Fit array: '+','.join(map(str,tasks))+f'%{a.max_parallel}')
        return
    if os.name=='nt':raise RuntimeError('Submit on Linux Slurm; Windows supports --dry-run')
    jobs.mkdir(parents=True,exist_ok=True);logs.mkdir(exist_ok=True)
    preamble='#!/bin/bash\nset -euo pipefail\nexport OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2\n'
    def script(name,cmd):
        path=jobs/name;path.write_text(preamble+cmd+'\n',encoding='utf-8');return str(path)
    submitted=[]
    def submit(name,path,hours,extra=()):
        argv=['sbatch','--parsable','--partition='+a.partition,'--nodes=1','--ntasks=1','--cpus-per-task=2',
              '--mem=12G','--time='+hours,'--job-name='+name,'--output='+str(logs/(name+'_%A_%a.out')),
              '--error='+str(logs/(name+'_%A_%a.err')),*extra,path]
        result=subprocess.run(argv,check=True,capture_output=True,text=True).stdout.strip().split(';')[0]
        if not result.isdigit():raise RuntimeError('Unexpected sbatch response: '+result)
        submitted.append(dict(job_id=result,name=name,command=argv))
        write_json(jobs/'submitted.json',dict(plan_sha256=digest(a.bundle/'plan.json'),datasets=cohorts,jobs=submitted))
        return result
    fit=None
    if a.stage in ['all','fit']:
        fit=submit('generalization-fit',script('fit.sh',fitcommand),'48:00:00',
            ['--array='+','.join(map(str,tasks))+f'%{a.max_parallel}'])
    if a.stage in ['all','analyze']:
        previous=None
        analysis_ids=[]
        for c,cmd in analysis:
            deps=[]
            if fit:deps.extend(fit+'_'+str(t['id']) for t in plan['tasks'] if t['dataset']==c)
            # Serialize analyses to keep aggregate memory bounded.
            dependency=[]
            if deps:dependency.append('afterok:'+':'.join(deps))
            if previous:dependency.append('afterany:'+previous)
            extra=['--dependency='+','.join(dependency)] if dependency else []
            previous=submit('analysis-'+c,script('analysis-'+c+'.sh',cmd),'12:00:00',extra)
            analysis_ids.append(previous)
        summary=command('summarize_generalization.py','--bundle',a.bundle,'--out',a.out,'--dpi',a.dpi,'--datasets',*cohorts)
        submit('generalization-summary',script('summary.sh',summary),'02:00:00',
               ['--dependency=afterany:'+':'.join(analysis_ids)])
    print('Submitted; inspect',jobs/'submitted.json')


if __name__=='__main__':main()
