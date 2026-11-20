"""Validate and copy existing abundance exports without fitting or sampling."""
from pathlib import Path
import argparse, json, shutil
import numpy as np
import pandas as pd
import pipeline as p

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--existing',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    p.verify(); plan=p.plan(); out=a.out.resolve(); existing=a.existing.resolve()
    if out==existing:raise ValueError('Use a separate output directory')
    if (out/'tasks.json').exists():raise ValueError('Output already prepared; choose a fresh OUT for a new submission')
    # Validate all seven exports before copying any.
    audits={}
    for d in plan['datasets']:
        name=d['dataset'];src=existing/'exports'/(name+'.npz')
        audit=json.loads(src.with_suffix('.json').read_text(encoding='utf-8'))
        if audit['output_sha256']!=p.sha(src) or audit['plan_sha256']!=plan['legacy_plan_sha256']:raise ValueError('Legacy export checksum/plan mismatch: '+name)
        if audit['representation']!='E[softmax(Z)] positive-abundance composition (all components, absence gate excluded)':raise ValueError('Wrong abundance representation')
        samples=pd.read_csv(p.PKG/'data'/name/'samples.tsv',sep='\t',dtype=str)
        taxa=pd.read_csv(p.PKG/'data'/name/'taxa.tsv',sep='\t',dtype=str)
        with np.load(src,allow_pickle=False) as z:
            assert np.array_equal(z['sample_id'],samples.sample_id.to_numpy())
            assert np.array_equal(z['taxon_id'],taxa.taxon_id.to_numpy())
            x=z['composition_mean'];counts=z['raw_counts']
            assert x.shape==counts.shape==(len(samples),len(taxa))
            assert np.isfinite(x).all() and (x>=0).all() and np.allclose(x.sum(1),1)
            assert np.isfinite(counts).all() and (counts>=0).all() and (counts.sum(1)>0).all()
        audits[name]=audit
    (out/'exports').mkdir(parents=True,exist_ok=True)
    for name,audit in audits.items():
        src=existing/'exports'/(name+'.npz');dest=out/'exports'/src.name
        shutil.copy2(src,dest)
        audit=dict(audit,source_export=str(src),source_plan_sha256=audit['plan_sha256'],plan_sha256=p.sha(p.PKG/'plan.json'),reuse_note='Identical abundance bytes; only contrast design changed')
        p.dump(dest.with_suffix('.json'),audit)
    print('Validated and reused',len(audits),'exports')

if __name__=='__main__':main()
