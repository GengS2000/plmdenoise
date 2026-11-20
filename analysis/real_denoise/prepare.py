"""Prepare three count cohorts and freeze a portable, verified server bundle."""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import tarfile
import numpy as np
import pandas as pd
from common import digest, tensorize, write_json, write_npz

HERE = Path(__file__).resolve().parent


def read_ecam(raw):
    p = raw / 'ECAM_data'
    x = pd.read_csv(p/'genus_count_cleaned_q2.csv', index_col=0)
    m = pd.read_csv(p/'genus_metadata_cleaned_q2.csv', index_col=0).loc[x.index]
    s = pd.DataFrame(index=x.index)
    s['subject_id'] = m.host_subject_id.astype(str)
    s['time'] = pd.to_numeric(m.day_of_life, errors='raise')
    s['group'] = m.delivery
    for col in ['diet','abx_all_sources','abx_name','sex']:
        s[col] = m[col]
    s['batch_label'] = 'unknown_single_batch'
    tax = pd.read_csv(p/'genus_taxonomy_cleaned_q2.csv',index_col=0).reindex(x.columns)
    labels = tax.fillna('').astype(str).agg('|'.join,axis=1)
    return x,s,labels,dict(time_unit='days of life', exposure='sum of supplied genus counts before new filtering',
        upstream_note='Supplied cleaned genus data; upstream filtering not reconstructed.')


def read_farmm(raw, export, rscript):
    subprocess.run([rscript,str(HERE/'export_farmm.R'),str(raw/'FARMM_data'),str(export)], check=True)
    x = pd.read_csv(export/'farmm_counts.tsv',sep='\t',index_col=0).T
    m = pd.read_csv(export/'farmm_samples.tsv',sep='\t',dtype={'SubjectID':str}).set_index('SampleID').loc[x.index]
    s = pd.DataFrame(index=x.index)
    s['subject_id'] = m.SubjectID
    s['time'] = pd.to_numeric(m.study_day, errors='raise')
    s['group'] = m.study_group
    s['batch_label'] = m.flow_cell_id
    for col in ['current_antibiotics','Age','BMI','copy_num_per_gram_feces']:
        s[col] = m[col]
    s['eligible'] = (m.Keep.astype(str).str.lower().eq('true') &
        m.isControl.astype(str).str.lower().eq('false') & m.SampleType.eq('Feces') & m.HostSpecies.eq('Human'))
    return x,s,pd.Series(x.columns,index=x.columns),dict(time_unit='study day',
        exposure='sum of supplied mat_count taxa before new filtering; conditional classified-count analysis',
        upstream_note='Integer matrix accepted as supplied; raw-read provenance not established. Non-host reads are not substituted for its count total.')


def genus_key(taxon):
    parts = [p.strip() for p in str(taxon).split(';')][:6]
    return '|'.join(parts)  # Keep full lineage; unresolved lineages stay separate.


def read_diab(raw):
    p = raw/'DIABIMMUNE_data'
    tax = pd.read_csv(p/'taxonomy.tsv',sep='\t').set_index('Feature ID')
    if not tax.index.is_unique:
        raise ValueError('Duplicate taxonomy IDs')
    keys = tax.Taxon.map(genus_key)
    summed = None
    total = None
    seen = set()
    for chunk in pd.read_csv(p/'table-matched.txt',sep='\t',index_col=0,chunksize=1000):
        a = chunk.to_numpy()
        if not chunk.index.is_unique or seen.intersection(chunk.index):
            raise ValueError('Duplicate raw feature IDs')
        seen.update(chunk.index)
        if not chunk.index.isin(keys.index).all():
            raise ValueError('Unmapped taxonomy')
        if not np.isfinite(a).all() or (a<0).any() or (a!=np.floor(a)).any():
            raise ValueError('Noninteger raw counts')
        depth = chunk.sum(0)
        total = depth if total is None else total.add(depth)
        grouped = chunk.groupby(keys.loc[chunk.index], sort=True).sum()
        summed = grouped if summed is None else summed.add(grouped,fill_value=0)
    x = summed.T.astype(np.int64)
    m = pd.read_csv(p/'11884_20190508-173103-added-month-abx.txt',sep='\t').set_index('#SampleID')
    if not m.index.is_unique or not x.index.isin(m.index).all():
        raise ValueError('Invalid DIABIMMUNE sample join')
    m = m.loc[x.index]
    s = pd.DataFrame(index=x.index)
    s['subject_id'] = m.host_subject_id.astype(str)
    s['time'] = pd.to_numeric(m.age_at_collection,errors='raise')
    s['group'] = m.abx_ever
    s['batch_label'] = 'unknown_single_batch'
    s['depth'] = total.loc[x.index]
    for col in ['csection','abx_month','number_days_abx','sex','month']:
        s[col] = m[col]
    return x,s,pd.Series(x.columns,index=x.columns),dict(time_unit='days of life',
        exposure='sum of all raw feature counts, conserved by full-lineage genus aggregation',
        upstream_note='Rebuilt from all 1098 raw samples; curated 1018-sample RData and 39-row df_subject are intentionally not used. Unresolved genus lineages retained separately.')


def finalize(name, x, s, labels, notes, target, prevalence, min_total):
    if not x.index.is_unique or not x.columns.is_unique or not s.index.equals(x.index):
        raise ValueError('Nonunique or misaligned matrix')
    if s[['subject_id','time','group','batch_label']].isna().any().any():
        raise ValueError('Missing required metadata')
    if (s.groupby('subject_id')['group'].nunique() != 1).any():
        raise ValueError('Time-varying primary group')
    if 'depth' not in s:
        s['depth'] = x.sum(1)
    reasons = pd.Series('',index=s.index)
    if 'eligible' in s:
        reasons.loc[~s.eligible] = 'failed_supplied_Keep_or_sample_type'
    reasons.loc[s.depth <= 0] = 'zero_library'
    # Unknown replicate identity: retain one sample, do not sum biological replicates.
    candidates = s.loc[reasons.eq('')].copy()
    candidates['_sid'] = candidates.index.astype(str)
    ordered = candidates.sort_values(['subject_id','time','depth','_sid'],ascending=[True,True,False,True])
    duplicates = ordered.index[ordered.duplicated(['subject_id','time'])]
    reasons.loc[duplicates] = 'duplicate_subject_time_lower_depth_or_id_tiebreak'
    valid = s.loc[reasons.eq('')]
    sparse = valid.groupby('subject_id').size().loc[lambda v:v<2].index
    reasons.loc[reasons.eq('') & s.subject_id.isin(sparse)] = 'fewer_than_two_visits'
    audit = s.copy(); audit['exclusion_reason'] = reasons
    s = s.loc[reasons.eq('')].sort_values(['subject_id','time'])
    x = x.loc[s.index]
    detected = (x>0).mean(0); total = x.sum(0)
    keep = (detected >= prevalence) & (total >= min_total)
    taxa = pd.DataFrame({'taxon_id':x.columns, 'taxonomy':labels.loc[x.columns].values,
        'prevalence':detected.values,'total_count':total.values,'retained':keep.values})
    x = x.loc[:,keep]
    if x.shape[1] < 3 or (x.sum(1)<=0).any():
        raise ValueError('Filtering leaves too few taxa or zero-composition sample; revise explicit QC plan')
    levels = sorted(s.batch_label.unique())
    s['batch'] = s.batch_label.map({v:i for i,v in enumerate(levels)})
    arrays, s = tensorize(x,s)
    target.mkdir(parents=True)
    write_npz(target/'input.npz',**arrays)
    s.index.name='sample_id'; s.to_csv(target/'samples.tsv',sep='\t')
    audit.index.name='sample_id';audit.to_csv(target/'sample_audit.tsv',sep='\t')
    taxa.to_csv(target/'taxa_audit.tsv',sep='\t',index=False)
    pd.DataFrame({'taxon_id':x.columns,'taxonomy':labels.loc[x.columns].values}).to_csv(target/'taxa.tsv',sep='\t',index=False)
    summary = dict(dataset=name, samples=len(s),subjects=s.subject_id.nunique(),taxa=x.shape[1],
        tensor_shape=list(arrays['Y'].shape),excluded_samples=int(reasons.ne('').sum()),
        groups=s.drop_duplicates('subject_id').group.value_counts().to_dict(),
        depth_min=float(s.depth.min()),depth_median=float(s.depth.median()),batch_levels=levels,
        min_prevalence=prevalence,min_total=min_total,
        padding='masked midpoints inside each observed time range; no endpoint extrapolation',
        denoise_group='blind',events='none (NaN internally); no treatment labels passed to denoiser',**notes)
    write_json(target/'summary.json',summary)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--core',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--rscript',default='Rscript')
    p.add_argument('--seeds',type=int,nargs='+',default=[2026,2027,2028])
    p.add_argument('--min-prevalence',type=float,default=0.)
    p.add_argument('--min-total',type=int,default=1)
    args=p.parse_args()
    if not 0<=args.min_prevalence<=1 or args.min_total<1 or len(set(args.seeds))!=len(args.seeds):
        p.error('Invalid filters or duplicate seeds')
    args.out=args.out.resolve();args.raw=args.raw.resolve()
    if args.out.exists():
        p.error('Output must be a new directory; prepared inputs are immutable')
    args.out.mkdir(parents=True)
    source=args.out/'source';source.mkdir()
    for f in HERE.iterdir():
        if f.suffix in {'.py','.R','.sh','.sbatch','.txt','.md'}:
            (source/f.name).write_bytes(f.read_bytes().replace(b'\r\n',b'\n'))
    (source/'event_aware_denoise.py').write_bytes(args.core.read_bytes().replace(b'\r\n',b'\n'))
    export=args.out/'preparation_export';export.mkdir()
    summaries=[]
    for name,reader in [('ECAM',lambda:read_ecam(args.raw)),
                        ('FARMM',lambda:read_farmm(args.raw,export,args.rscript)),
                        ('DIABIMMUNE',lambda:read_diab(args.raw))]:
        summaries.append(finalize(name,*reader(),args.out/'data'/name,args.min_prevalence,args.min_total))
    input_files = ['ECAM_data/genus_count_cleaned_q2.csv','ECAM_data/genus_metadata_cleaned_q2.csv',
        'ECAM_data/genus_taxonomy_cleaned_q2.csv','FARMM_data/mat_count.RData','FARMM_data/df_samples.RData',
        'DIABIMMUNE_data/table-matched.txt','DIABIMMUNE_data/taxonomy.tsv',
        'DIABIMMUNE_data/11884_20190508-173103-added-month-abx.txt']
    plan=dict(schema=1,primary_seed=2026 if 2026 in args.seeds else args.seeds[0],
        resources=dict(cpus=2,mem='6G'),datasets=summaries,
        tasks=[dict(dataset=s['dataset'],seed=seed) for s in summaries for seed in args.seeds],
        config=dict(adaptive_budget=True,joint_rank=4,hierarchical_observation=True,flexible_event=True,
            device='cpu',map_steps=600,map_starts=2,vi_steps=2000,posterior_samples=200),
        raw_sha256={n:digest(args.raw/n) for n in input_files},
        files={str(f.relative_to(args.out)).replace('\\','/'):digest(f)
               for folder in [source,args.out/'data'] for f in sorted(folder.rglob('*')) if f.is_file()})
    write_json(args.out/'plan.json',plan)
    (args.out/'PLAN.sha256').write_text(digest(args.out/'plan.json')+'\n',encoding='utf-8')
    archive=args.out.with_suffix('.tar.gz')
    if archive.exists():
        raise FileExistsError(archive)
    with tarfile.open(archive,'w:gz') as tar:
        tar.add(args.out,arcname=args.out.name)
    # Verify all frozen payload bytes in archive, not merely successful compression.
    import hashlib
    with tarfile.open(archive) as tar:
        for name,h in plan['files'].items():
            if hashlib.sha256(tar.extractfile(args.out.name+'/'+name).read()).hexdigest()!=h:
                raise ValueError('Archive checksum mismatch: '+name)
    archive.with_name(archive.name+'.sha256').write_text(digest(archive)+'  '+archive.name+'\n')
    print(json.dumps(summaries,indent=2,ensure_ascii=False))
    print('Verified archive:',archive)


if __name__=='__main__':
    main()
