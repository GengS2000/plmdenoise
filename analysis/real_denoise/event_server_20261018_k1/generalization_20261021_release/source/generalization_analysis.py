"""Conditional subject/block inference for cross-environment application cohorts."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from common import digest,write_json
from cohort_analysis import centre,bh,load_seed,seed_agreement
from publication_analysis import bootstrap_columns,sign_test_matrix,ordination,variance_covariance
from application_figures import draw,trajectory,taxon_label


def contrast_matrices(reps,s,design):
    matrices={}
    for rep,x in reps.items():
        for name,before,after in design['contrasts']:
            rows=[];ids=[]
            for sid,g in s.groupby('subject_id',sort=True):
                a=g.index[g.time.isin(before)];b=g.index[g.time.isin(after)]
                # Mean within phase first; no pseudoreplication of visits.
                if len(a) and len(b):rows.append(x[b].mean(0)-x[a].mean(0));ids.append(sid)
            mat=pd.DataFrame(rows,index=ids,columns=range(x.shape[1]))
            if design['family']=='SOIL':
                meta=s.drop_duplicates('subject_id').set_index('subject_id').loc[ids]
                # Block is the independent unit for shared weather contrasts.
                blockmean=mat.groupby(meta.block).mean()
                matrices[(rep,name)]=(blockmean.index.astype(str).tolist(),blockmean.to_numpy())
                # Treatment-by-stage: pair treatment with 0 within cultivar/block,
                # then average cultivars within block; never pool 48 plots as independent weather replicates.
                for level in [25,50,75]:
                    differences=[];blocks=[]
                    for block,g in meta.groupby('block'):
                        per_cultivar=[]
                        for cultivar,h in g.groupby('cultivar'):
                            a=h.index[pd.to_numeric(h.treatment).eq(0)]
                            b=h.index[pd.to_numeric(h.treatment).eq(level)]
                            if len(a)==1 and len(b)==1:per_cultivar.append(mat.loc[b[0]].to_numpy()-mat.loc[a[0]].to_numpy())
                        if len(per_cultivar)==2:blocks.append(str(block));differences.append(np.mean(per_cultivar,axis=0))
                    matrices[(rep,name+f'__PE{level}_minus_PE0')]=(blocks,np.array(differences).reshape(-1,x.shape[1]))
            else:matrices[(rep,name)]=(ids,mat.to_numpy())
    return matrices


def infer(matrices,taxa,out,perms=4999,boot=2000):
    rng=np.random.default_rng(20261021);rows=[];long=[]
    for (rep,name),(ids,x) in matrices.items():
        if len(ids)<6:raise ValueError(f'Insufficient independent units for {name}: {len(ids)}')
        lo,hi=bootstrap_columns(x,rng,boot)
        # Exact enumeration for six soil blocks; Monte Carlo cannot increase resolution.
        if len(ids)<=12:
            import itertools
            signs=np.array(list(itertools.product([-1.,1.],repeat=len(ids))))
            p=(np.abs(signs@x/len(ids))>=np.abs(x.mean(0))[None]-1e-12).mean(0)
        else:p=sign_test_matrix(x,rng,perms)
        for j,t in taxa.iterrows():
            rows.append(dict(representation=rep,contrast=name,taxon_id=t.taxon_id,name=t['name'],
                n_subjects=len(ids),mean_effect=x[:,j].mean(),ci_low=lo[j],ci_high=hi[j],p_signflip=p[j],unit='CLR'))
        for i,unit in enumerate(ids):
            long.extend(dict(representation=rep,contrast=name,unit_id=unit,taxon_id=taxa.taxon_id.iloc[j],effect=v) for j,v in enumerate(x[i]))
    result=pd.DataFrame(rows)
    result['q_bh']=result.groupby('representation').p_signflip.transform(lambda p:bh(p.to_numpy()))
    result.to_csv(out/'taxon_contrasts.tsv',sep='\t',index=False)
    pd.DataFrame(long).to_csv(out/'unit_taxon_contrasts.tsv.gz',sep='\t',index=False)
    return result


def analyze(bundle,fits,out,cohort,dpi=600,perms=4999,boot=2000):
    from event_run import verify,location
    plan=verify(bundle)
    if out.exists() and (out/'complete.json').exists():
        done=json.loads((out/'complete.json').read_text())
        if done['plan_sha256']!=digest(bundle/'plan.json'):raise ValueError('Output belongs to another plan')
        for f,h in done['files'].items():
            if digest(out/f)!=h:raise ValueError('Corrupt analysis '+f)
        print('Verified complete',out);return
    out.mkdir(parents=True,exist_ok=True)
    entries=[d for d in plan['datasets'] if d['dataset']==cohort]
    if len(entries)!=1:raise ValueError('Unknown cohort')
    design=entries[0];seeds=[];source_hashes={}
    for task in plan['tasks']:
        if task['dataset']!=cohort:continue
        path=location(fits,task);v=load_seed(path)
        done=json.loads((path/'done.json').read_text())
        if done['identity']['plan_sha256']!=digest(bundle/'plan.json') or v['task']!=task:raise ValueError('Fit identity mismatch')
        seeds.append(v);source_hashes[str(path/'done.json')]=digest(path/'done.json')
    if len(seeds)!=len(plan['seeds']):raise ValueError('Require all planned seeds')
    s=seeds[0]['samples'];taxa=seeds[0]['taxa'];counts=seeds[0]['counts']
    for v in seeds:
        if not v['samples'].sample_id.equals(s.sample_id) or not v['taxa'].taxon_id.equals(taxa.taxon_id) or not np.array_equal(v['counts'],counts):raise ValueError('Seed alignment mismatch')
    data=bundle/'data'/cohort
    reference=pd.read_csv(data/'samples.tsv',sep='\t',dtype={'sample_id':str,'subject_id':str})
    reference_taxa=pd.read_csv(data/'taxa.tsv',sep='\t',dtype=str)
    with np.load(data/'input.npz') as f:expected=f['Y'][reference.tensor_i,reference.tensor_t]
    if not s.equals(reference) or not taxa.equals(reference_taxa) or not np.array_equal(counts,expected):raise ValueError('Fit differs from frozen inputs')
    comp={k:centre(np.mean([v['components'][k] for v in seeds],0)) for k in ['baseline','common','individual','event','total']}
    if not np.allclose(sum(comp[k] for k in ['baseline','common','individual','event']),comp['total'],atol=1e-5):raise ValueError('Nonadditive components')
    taxa=taxa.copy();taxa['name']=taxa.taxonomy.map(taxon_label)
    reps=dict(raw=centre(np.log(counts+.5)),denoised=comp['total'])
    result=infer(contrast_matrices(reps,s,design),taxa,out,perms,boot)
    seed_rows=[]
    for v in seeds:
        for (_,name),(ids,x) in contrast_matrices({'denoised':centre(v['components']['total'])},s,design).items():
            seed_rows.extend(dict(seed=v['task']['seed'],contrast=name,taxon_id=tid,mean_effect=e) for tid,e in zip(taxa.taxon_id,x.mean(0)))
    sd=pd.DataFrame(seed_rows);sd.to_csv(out/'seed_effects.tsv',sep='\t',index=False)
    stab=sd.groupby(['contrast','taxon_id']).mean_effect.agg(['min','max']).reset_index()
    stab['all_seed_same_sign']=(stab['min']>0)|(stab['max']<0)
    raw=result[result.representation.eq('raw')];den=result[result.representation.eq('denoised')]
    evidence=raw.merge(den,on=['contrast','taxon_id'],suffixes=('_raw','_den')).merge(stab,on=['contrast','taxon_id'])
    prev=pd.DataFrame(counts>0).groupby(s.subject_id).max().mean().to_numpy()
    evidence=evidence.merge(pd.DataFrame({'taxon_id':taxa.taxon_id,'raw_subject_prevalence':prev}),on='taxon_id')
    evidence['evidence_class']=np.select([
        evidence.q_bh_den.lt(.05)&evidence.q_bh_raw.lt(.05)&(evidence.mean_effect_raw*evidence.mean_effect_den>0),
        evidence.q_bh_den.lt(.05)&evidence.q_bh_raw.ge(.05)&evidence.all_seed_same_sign],
        ['raw_supported','denoised_only_exploratory'],default='not_supported_or_unstable')
    evidence['literature_status']='requires_exact_taxon_site_contrast_review'
    evidence.to_csv(out/'biological_evidence.tsv',sep='\t',index=False)
    ordination(reps,s,taxa,out);variance_covariance(comp,out);trajectory(s,taxa,counts,comp,out)
    seed_agreement(seeds).to_csv(out/'seed_agreement.tsv',sep='\t',index=False)
    pd.DataFrame([dict(seed=v['task']['seed'],**v['quality']) for v in seeds]).to_csv(out/'seed_quality.tsv',sep='\t',index=False)
    s.to_csv(out/'samples.tsv',sep='\t',index=False);taxa.to_csv(out/'taxa.tsv',sep='\t',index=False)
    pd.DataFrame(dict(taxon_id=taxa.taxon_id,subject_prevalence=prev)).to_csv(out/'raw_detection.tsv',sep='\t',index=False)
    draw(out,cohort,dpi,status='Conditional K=1 inference; resampling unit: '+design['resampling_unit'])
    write_json(out/'analysis_config.json',dict(design=design,permutations=perms,bootstrap=boot,source_hashes=source_hashes))
    (out/'REPORT.md').write_text(f'# {cohort}\n\n{len(s)} samples; {s.subject_id.nunique()} longitudinal units; {len(taxa)} features.\n\n'
        'All planned seeds verified. Numerical completion does not imply convergence; inspect seed_quality.tsv. '
        'Raw and denoised analyses use identical features. BH family: all taxa and prespecified contrasts within this site/representation. '
        'Soil resamples six blocks, including exact 64-pattern sign-flips; two-sided p resolution is 0.03125. '
        'No claim of calibrated post-denoising FDR. Events synchronized across subjects cannot be identified separately from time without assumptions. '
        'Novel findings in biological_evidence.tsv remain exploratory until exact taxonomy, literature, sensitivity and function support are checked.\n',encoding='utf-8')
    write_json(out/'functional_status.json',dict(status='awaiting_functional_mapping',
        next_step='Review taxon-level contrasts.'))
    write_json(out/'complete.json',dict(status='complete',plan_sha256=digest(bundle/'plan.json'),
        files={p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!='complete.json'}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['bundle','fits','out']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--cohort',required=True);p.add_argument('--dpi',type=int,default=600)
    a=p.parse_args();analyze(a.bundle.resolve(),a.fits.resolve(),a.out.resolve(),a.cohort,a.dpi)
