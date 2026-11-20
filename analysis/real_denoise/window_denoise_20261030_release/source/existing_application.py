"""Reuse original cohort analysis and export a standalone application figure."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from common import digest
from application_figures import draw,trajectory,taxon_label


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['bundle','fits','out']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--cohort',choices=['DIABIMMUNE','ECAM','FARMM'],required=True)
    p.add_argument('--dpi',type=int,default=600);a=p.parse_args()
    from event_run import verify,location
    from publication_analysis import analyze
    from publication_dispatch import check_output
    from cohort_analysis import load_seed,centre
    plan=verify(a.bundle)
    if not check_output(a.out/'complete.json'):analyze(a.bundle,a.fits,a.out,a.cohort)
    tables=a.out;figures=tables/'figures';figures.mkdir(exist_ok=True)
    s=pd.read_csv(tables/'samples.tsv',sep='\t',dtype={'subject_id':str,'sample_id':str})
    t=pd.read_csv(tables/'taxa.tsv',sep='\t',dtype=str);t['name']=t.taxonomy.map(taxon_label)
    values=[]
    for task in plan['tasks']:
        if task['dataset']!=a.cohort:continue
        path=location(a.fits,task)
        if not (path/'done.json').exists():continue
        v=load_seed(path)
        done=json.loads((path/'done.json').read_text())
        if done['identity']['plan_sha256']!=digest(a.bundle/'plan.json'):raise ValueError('Wrong fit plan')
        if not v['samples'].sample_id.equals(s.sample_id):raise ValueError('Sample mismatch')
        ix=v['taxa'].set_index('taxon_id').index.get_indexer(t.taxon_id)
        if (ix<0).any():raise ValueError('Taxon mismatch')
        values.append({k:centre(v['components'][k][:,ix]) for k in ['baseline','common','individual','event','total']})
        counts=v['counts'][:,ix]
    if not values:raise ValueError('No complete original fits')
    comp={k:np.mean([v[k] for v in values],0) for k in values[0]}
    for name in ['ordination_scores.tsv','component_covariance.tsv']:
        (figures/name).write_bytes((tables/name).read_bytes())
    contrasts=pd.read_csv(tables/'taxon_contrasts.tsv',sep='\t').drop(columns='name').merge(t[['taxon_id','name']],on='taxon_id',validate='many_to_one')
    contrasts.to_csv(figures/'taxon_contrasts.tsv',sep='\t',index=False)
    trajectory(s,t,counts,comp,figures)
    draw(figures,a.cohort,a.dpi,status=f'Original K=1 fits; {len(values)} seeds; conditional inference')


if __name__=='__main__':main()
