"""Prepare a NEW portable K=1 bundle, leaving the original plan untouched."""
from pathlib import Path
import argparse
import json
import shutil
import re
import numpy as np
import pandas as pd
from common import tensorize,write_npz,write_json,digest

VISIT_DAYS={1:-14,2:0,3:2,4:5,5:9,6:14,7:21}
SOIL_DAYS={'May.10':0,'May.24':14,'Jun.7':28,'Jun.21':42,'Jul.5':56,'Jul.19':70,'Aug.1':83}
SOIL_DOI='https://doi.org/10.1128/msystems.00247-22'
GING_DOI='https://doi.org/10.1038/s41522-024-00565-x'


def freeze_dataset(out,name,counts,s,taxa,max_taxa,event_times,event_names,info):
    if not counts.index.equals(s.index):raise ValueError('Sample order mismatch')
    if counts.index.duplicated().any() or counts.columns.duplicated().any():raise ValueError('Duplicate input IDs')
    if s.time.isna().any():raise ValueError('Unmapped time')
    if s.duplicated(['subject_id','time']).any():raise ValueError('Duplicate subject/time')
    a=counts.to_numpy(float)
    if not np.isfinite(a).all() or (a<0).any() or (a!=np.floor(a)).any():raise ValueError('Invalid counts')
    depth=counts.sum(axis=1)
    good=depth.gt(0)&s.subject_id.map(s[depth.gt(0)].subject_id.value_counts()).ge(2)
    excluded=s.loc[~good].copy();counts=counts.loc[good];s=s.loc[good].copy()
    s['depth']=counts.sum(axis=1);s['batch']=0
    # Preserve full-library exposure even when a fixed raw-only feature budget is used.
    prevalence=counts.gt(0).groupby(s.subject_id).max().mean()
    relative=counts.div(s.depth,axis=0).groupby(s.subject_id).mean().mean()
    annotation=taxa.set_index('taxon_id').reindex(counts.columns)
    if annotation.taxonomy.isna().any():raise ValueError('Missing taxonomy IDs')
    organelle=annotation.taxonomy.str.contains('chloroplast|mitochondri',case=False,regex=True)
    filt=pd.DataFrame({'taxon_id':counts.columns,'subject_prevalence':prevalence.values,
                       'subject_mean_relative_abundance':relative.values,'organelle':organelle.values})
    eligible=filt[filt.subject_prevalence.ge(.10)&~filt.organelle].sort_values(
        ['subject_mean_relative_abundance','taxon_id'],ascending=[False,True])
    ids=eligible.taxon_id.tolist()[:max_taxa or None]
    if len(ids)<3:raise ValueError('Insufficient retained features')
    filt['selected']=filt.taxon_id.isin(ids)
    data=out/'data'/name;data.mkdir(parents=True)
    filt.to_csv(data/'feature_filter.tsv',sep='\t',index=False)
    excluded.to_csv(data/'excluded_samples.tsv',sep='\t',index=False)
    arrays,s=tensorize(counts[ids],s)
    write_npz(data/'input.npz',**arrays)
    s.to_csv(data/'samples.tsv',sep='\t',index=False)
    taxonomy=taxa.set_index('taxon_id').loc[ids].reset_index()
    taxonomy.to_csv(data/'taxa.tsv',sep='\t',index=False)
    events=np.tile(np.array(event_times,float),(s.subject_id.nunique(),1))
    write_npz(data/'events.npz',event_times=events,event_names=np.array(event_names,dtype=str))
    if 'representative_sequence' in taxonomy:
        with (data/'representative_sequences.fasta').open('w',encoding='ascii') as f:
            for _,r in taxonomy.iterrows():
                seq=str(r.representative_sequence).upper()
                if not re.fullmatch('[ACGTN]+',seq):raise ValueError('Invalid representative sequence')
                f.write('>'+r.taxon_id+'\n'+seq+'\n')
    counts[ids].T.to_csv(data/'functional_counts.tsv',sep='\t',index_label='feature_id')
    qc=s[['sample_id','subject_id','time','depth']].copy();qc['retained_count_sum']=counts[ids].sum(axis=1).to_numpy()
    qc['retained_fraction']=qc.retained_count_sum/qc.depth
    qc.to_csv(data/'sample_qc.tsv',sep='\t',index=False)
    info.update(dataset=name,note=info['event_definition'],n_subjects=s.subject_id.nunique(),n_samples=len(s),n_taxa=len(ids),
                full_taxa=counts.shape[1],max_taxa=max_taxa,feature_rule='>=10% subject prevalence; exclude organelles; rank by subject-equal raw relative abundance',
                event_names=event_names,event_times=event_times)
    write_json(data/'DESIGN.json',info)
    return info


def prepare(raw,out,base,datasets,max_taxa=500,seeds=(2026,2027,2028)):
    if out.exists():raise ValueError('Use a new bundle directory')
    if len(set(seeds))!=len(seeds):raise ValueError('Duplicate seeds')
    out.mkdir(parents=True);(out/'source').mkdir()
    for path in (base/'source').iterdir():
        if path.is_file():shutil.copyfile(path,out/'source'/path.name)
    for name in ['GENERALIZATION_README.md','SOIL_ANALYSIS.md','GINGIVITIS_ANALYSIS.md']:
        if (base/name).exists():shutil.copyfile(base/name,out/name)
    old=json.loads((base/'plan.json').read_text(encoding='utf-8'))
    config=dict(old['config']);config['n_factors']=1
    entries=[];hashes={}
    if 'SOIL' in datasets:
        path=raw/'土壤数据'
        for file in ['16s_OTU_taxa.txt','soil water anova.txt']:hashes[str(path/file)]=digest(path/file)
        table=pd.read_csv(path/'16s_OTU_taxa.txt',sep='\t',index_col=0)
        taxa=pd.DataFrame({'taxon_id':table.index.astype(str),'taxonomy':table.taxonomy.values})
        counts=table.drop(columns='taxonomy').T
        s=pd.read_csv(path/'soil water anova.txt',sep='\t',index_col=0)
        missing=counts.index.difference(s.index)
        if len(missing):raise ValueError('Soil counts without metadata: '+str(missing.tolist()))
        s.loc[s.index.difference(counts.index)].to_csv(out/'soil_metadata_without_counts.tsv',sep='\t')
        s=s.loc[counts.index].copy();s['sample_id']=s.index.astype(str)
        s['subject_id']=s[['block','cultivar','treatment']].astype(str).agg('|'.join,axis=1)
        s['group']=s.treatment.astype(str);s['time']=s.date.map(SOIL_DAYS)
        # Rainfall onset is interval-censored: primary decomposition has NO fabricated event date.
        entries.append(freeze_dataset(out,'SOIL',counts,s,taxa,max_taxa,[np.nan],['undefined_rainfall_onset'],
            dict(family='SOIL',doi=SOIL_DOI,time_unit='days_since_2018_05_10',
                 subject_definition='block x cultivar x treatment (48 plots); never numeric sample suffix',
                 event_definition='No event channel in primary fit; exact rainfall onset not supplied. Common/individual decomposition only.',
                 contrasts=[['dry_minus_early',[0,14,28],[56]],['rewet_minus_dry',[56],[70,83]]],
                 resampling_unit='block',interpretation='Seasonal paired contrasts, not isolated causal rainfall effects')))
    if 'GINGIVITIS' in datasets:
        path=raw/'牙龈炎数据'
        for file in ['count-table.tsv','taxonomic-classification.tsv','metabolomics.tsv','metabolomics-features.tsv']:
            hashes[str(path/file)]=digest(path/file)
        table=pd.read_csv(path/'count-table.tsv',sep='\t');table=table.set_index('sample',drop=False)
        tax=pd.read_csv(path/'taxonomic-classification.tsv',sep='\t',dtype=str).rename(columns={'ASV':'taxon_id'})
        if 'taxon_id' not in tax:tax=tax.rename(columns={tax.columns[0]:'taxon_id'})
        ranks=['Kingdom','Phylum','Class','Order','Family','Genus','Species']
        tax['taxonomy']=tax[ranks].fillna('unclassified').agg('|'.join,axis=1)
        # Exactly the original author's preprocessing cohort, not test vs control treatment inference.
        s=table.iloc[:,:5].copy().rename(columns={'sample':'sample_id','subject':'subject_id'})
        s.loc[~s.group.eq('control')].to_csv(out/'gingivitis_excluded_groups.tsv',sep='\t',index=False)
        for niche in sorted(s.niche.unique()):
            sub=s[s.group.eq('control')&s.niche.eq(niche)].copy()
            sub['time']=sub.visit.map(VISIT_DAYS)
            name='GINGIVITIS_'+re.sub('[^A-Za-z0-9]+','_',niche).strip('_')
            counts=table.loc[sub.index,table.columns.str.match(r'^ASV\d+$')]
            entries.append(freeze_dataset(out,name,counts,sub,tax,max_taxa,[0.,14.],['hygiene_withdrawal','hygiene_resumption'],
                dict(family='GINGIVITIS',niche=niche,doi=GING_DOI,time_unit='days',visit_days=VISIT_DAYS,
                     group_definition='Original author code retains group=control; 41 subjects. Not low/high clinical responders.',
                     event_definition='0: oral hygiene withdrawal; 14: resumption; synchronized model-conditional components',
                     contrasts=[['challenge_minus_baseline',[0],[14]],['resolution_minus_baseline',[0],[21]],['resolution_minus_challenge',[14],[21]]],
                     resampling_unit='subject',interpretation='Per-site paired inference; no clinical responder labels available')))
        extra=out/'auxiliary';extra.mkdir()
        for file in ['metabolomics.tsv','metabolomics-features.tsv']:shutil.copyfile(path/file,extra/file)
    tasks=[dict(id=i,dataset=e['dataset'],arm='noevent' if e['family']=='SOIL' else 'event',seed=seed)
           for i,(e,seed) in enumerate((e,seed) for e in entries for seed in seeds)]
    write_json(out/'RAW_PROVENANCE.json',hashes)
    plan=dict(schema='generalization-k1-v1',config=config,seeds=list(seeds),datasets=entries,tasks=tasks,
              dataset_config={e['dataset']:dict(event_basis=['decay','pulse','step','delayed_decay']) for e in entries},
              resources=dict(cpus=2,mem='12G',time='48:00:00'),source_plan_sha256=digest(base/'plan.json'),
              files={str(p.relative_to(out)).replace('\\','/'):digest(p) for p in out.rglob('*') if p.is_file()})
    write_json(out/'plan.json',plan);(out/'PLAN.sha256').write_text(digest(out/'plan.json')+'\n',encoding='ascii')
    print(json.dumps([dict(dataset=e['dataset'],subjects=e['n_subjects'],samples=e['n_samples'],taxa=e['n_taxa']) for e in entries],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--base',type=Path,default=Path(__file__).resolve().parent.parent)
    p.add_argument('--datasets',nargs='+',choices=['SOIL','GINGIVITIS'],default=['SOIL','GINGIVITIS'])
    p.add_argument('--max-taxa',type=int,default=500,help='Raw-only feature budget; 0 keeps all eligible features')
    a=p.parse_args()
    if a.max_taxa<0:p.error('--max-taxa must be nonnegative')
    prepare(a.raw.resolve(),a.out.resolve(),a.base.resolve(),a.datasets,a.max_taxa)
