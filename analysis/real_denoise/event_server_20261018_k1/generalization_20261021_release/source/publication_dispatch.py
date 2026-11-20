"""Frozen extension check, table-job dispatcher, and completion audit."""
from pathlib import Path
import argparse
import json
import shutil
import numpy as np
import pandas as pd
from common import digest,write_json


def check(bundle):
    from event_run import verify
    plan=verify(bundle)
    manifest=json.loads((bundle/'PUBLICATION_MANIFEST.json').read_text(encoding='utf-8'))
    if manifest['base_plan_sha256']!=digest(bundle/'plan.json'):raise ValueError('Base plan changed')
    for name,h in manifest['files'].items():
        path=(bundle/name).resolve()
        if not path.is_relative_to(bundle.resolve()) or digest(path)!=h:raise ValueError('Publication payload changed: '+name)
    return plan


def table_jobs():
    return [dict(cohort=c,variant=v,prevalence=p,pseudocount=z)
            for v,p,z in [('main',.10,.5),('prevalence05',.05,.5),('prevalence20',.20,.5),('pseudocount1',.10,1.)]
            for c in ['FARMM','ECAM','DIABIMMUNE']]


def check_output(done):
    if not done.exists():return False
    obj=json.loads(done.read_text(encoding='utf-8'))
    if obj['status']!='complete':return False
    for name,h in obj['files'].items():
        if digest(done.parent/name)!=h:raise ValueError('Result checksum mismatch: '+str(done.parent/name))
    return True


def audit(bundle,fits,out,validation=False):
    rows=[];summary=[]
    for task in table_jobs():
        dest=out/'tables'/task['variant']/task['cohort']
        ok=check_output(dest/'complete.json')
        rows.append(dict(task,status='complete' if ok else 'missing_or_failed',path=str(dest)))
        if ok:
            conf=json.loads((dest/'analysis_config.json').read_text(encoding='utf-8'))
            q=pd.read_csv(dest/'seed_quality.tsv',sep='\t')
            st=pd.read_csv(dest/'seed_status.tsv',sep='\t')
            summary.append(dict(task,n_subjects=conf['n_subjects'],n_samples=conf['n_samples'],n_taxa=conf['n_taxa'],
                seeds_used=int((st.status=='used').sum()),map_stationary=int(q.diagnostic_map_stationary.sum()),
                vi_stationary=int(q.diagnostic_vi_stationary.sum()),inference='conditional; not externally validated'))
    pd.DataFrame(rows).to_csv(out/'publication_status.tsv',sep='\t',index=False)
    pd.DataFrame(summary).to_csv(out/'cohort_summary.tsv',sep='\t',index=False)
    # Sensitivity results retain every tested taxon/contrast, not just discoveries.
    merged=[]
    for task in table_jobs():
        f=out/'tables'/task['variant']/task['cohort']/'taxon_contrasts.tsv'
        if f.exists():
            d=pd.read_csv(f,sep='\t');d['cohort']=task['cohort'];d['variant']=task['variant'];merged.append(d)
    if merged:pd.concat(merged,ignore_index=True).to_csv(out/'sensitivity_taxon_contrasts.tsv.gz',sep='\t',index=False)
    valid=True
    if validation:
        from publication_validation import summarize
        try:summarize(bundle,fits,out,check(bundle))
        except RuntimeError as exc:print(exc);valid=False
    valid=valid and all(r['status']=='complete' for r in rows)
    # Completion of computation is distinct from biological/statistical readiness.
    missing_seeds=any(r['seeds_used']<3 for r in summary)
    write_json(out/'publication_audit.json',dict(computations_complete=valid,missing_primary_seeds=missing_seeds,
        validation_requested=validation,interpretation='Review stability and null controls before claims; nonstationary flags are not silently waived.'))
    if not valid or missing_seeds:raise RuntimeError('Incomplete computations or missing primary seeds; inspect publication_status.tsv and seed_status.tsv')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['check','tables','summarize'])
    p.add_argument('--bundle',type=Path,default=Path(__file__).resolve().parent.parent)
    p.add_argument('--fits',type=Path);p.add_argument('--out',type=Path)
    p.add_argument('--task',type=int);p.add_argument('--validation',action='store_true')
    a=p.parse_args();bundle=a.bundle.resolve();check(bundle)
    if a.command=='check':print('Original plan and publication extension verified; 12 table jobs, 40 validation fits');return
    if a.out is None or a.fits is None:p.error('--out and --fits required')
    out=a.out.resolve();fits=a.fits.resolve()
    for other in [bundle,fits]:
        if out.is_relative_to(other) or other.is_relative_to(out):p.error('Publication output, bundle and original fits must be separate trees')
    out.mkdir(parents=True,exist_ok=True)
    if a.command=='summarize':audit(bundle,fits,out,a.validation);return
    if a.task is None or not 0<=a.task<len(table_jobs()):p.error('Invalid table task')
    task=table_jobs()[a.task];dest=out/'tables'/task['variant']/task['cohort']
    if check_output(dest/'complete.json'):print('Verified complete',dest);return
    from publication_analysis import analyze
    from event_run import acquire
    dest.mkdir(parents=True,exist_ok=True);lock=dest/'.lock';acquire(lock)
    try:
        if not check_output(dest/'complete.json'):
            analyze(bundle,fits,dest,task['cohort'],prevalence=task['prevalence'],pseudocount=task['pseudocount'])
    finally:shutil.rmtree(lock,ignore_errors=True)


if __name__=='__main__':main()
