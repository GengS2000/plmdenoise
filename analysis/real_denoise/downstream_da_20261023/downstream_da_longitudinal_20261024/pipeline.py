"""Read-only checkpoint replay and all-taxon differential analysis. No optimisation."""
from pathlib import Path
import argparse, hashlib, json, os, subprocess, sys
import numpy as np
import pandas as pd

PKG=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(p)
def plan():return json.loads((PKG/'plan.json').read_text(encoding='utf-8'))
def verify():
    for f,h in json.loads((PKG/'FILES_SHA256.json').read_text(encoding='utf-8')).items():
        if sha(PKG/f)!=h:raise ValueError('Package changed: '+f)
def bh(p):
    p=np.asarray(p,float);ok=np.isfinite(p);q=np.full(len(p),np.nan);v=p[ok]
    order=np.argsort(v);rank=np.minimum.accumulate((v[order]*len(v)/np.arange(1,len(v)+1))[::-1])[::-1]
    out=np.empty(len(v));out[order]=np.minimum(rank,1);q[ok]=out;return q

def r_environment(method):
    env=os.environ.copy()
    for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:env[k]='1'
    if method=='maaslin2' and os.environ.get('MAASLIN2_R_ENV_FILE',os.environ.get('R_ENV_FILE','')) != os.environ.get('R_ENV_FILE',''):
        for k in ['R_LIBS','R_LIBS_USER','R_LIBS_SITE']:env.pop(k,None)
        if env.get('MAASLIN2_R_LIBS_USER'):env['R_LIBS_USER']=env['MAASLIN2_R_LIBS_USER']
    return env

def r_launch(binary,arguments,method):
    envfile=os.environ.get('MAASLIN2_R_ENV_FILE',os.environ.get('R_ENV_FILE','')) if method=='maaslin2' else os.environ.get('R_ENV_FILE','')
    if os.name=='nt':return [binary,*arguments]
    # Arguments remain positional: no interpolation of paths or R code into shell text.
    return ['bash','-c','envfile="$1"; binary="$2"; shift 2; if [ -n "$envfile" ]; then source "$envfile" || exit $?; fi; "$binary" "$@"','da-r',envfile,binary,*arguments]

def check_r(a):
    checks=[('linda',a.r,'MicrobiomeStat,MASS,nlme,lme4,lmerTest,NBZIMM'),('maaslin2',a.maaslin2_r,'Maaslin2')]
    for method,binary,packages in checks:
        code='p<-strsplit("'+packages+'",",")[[1]]; print(.libPaths()); bad<-p[!vapply(p,requireNamespace,logical(1),quietly=TRUE)]; if(length(bad))stop(paste("Missing packages",paste(bad,collapse=",")))'
        subprocess.run(r_launch(binary,['--vanilla','-e',code],method),env=r_environment(method),check=True)

def export(a):
    import torch
    from event_aware_denoise import EventAwareDenoiser
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','2')))
    d=plan()['datasets'][a.index];name=d['dataset'];fits=Path(a.old_fits if d['root']=='old' else a.new_fits)
    samples=pd.read_csv(PKG/'data'/name/'samples.tsv',sep='\t',dtype={'sample_id':str,'subject_id':str})
    taxa=pd.read_csv(PKG/'data'/name/'taxa.tsv',sep='\t',dtype=str)
    means=[];mcvariances=[];sources=[];missing=[];raw=None
    for task in d['tasks']:
        folder=fits/'tasks'/f"{task['id']:02d}_{name}_{task['arm']}_{task['seed']}"
        if not (folder/'done.json').exists():missing.append(task['seed']);continue
        done=json.loads((folder/'done.json').read_text(encoding='utf-8'))
        if not done.get('usable'):missing.append(task['seed']);continue
        if done['identity'].get('smoke') or done['identity']['task']!=task or done['identity']['plan_sha256']!=d['plan_sha256']:raise ValueError('Wrong fit identity')
        for filename in ['fit.pt','posterior_observed.npz','samples.tsv','taxa.tsv','quality.json']:
            if sha(folder/filename)!=done['files'][filename]:raise ValueError('Corrupted '+str(folder/filename))
        # Fitting serializes TSVs again (LF/CRLF and numeric spelling can differ).
        fitted_s=pd.read_csv(folder/'samples.tsv',sep='\t',dtype={'sample_id':str,'subject_id':str})
        fitted_t=pd.read_csv(folder/'taxa.tsv',sep='\t',dtype=str)
        pd.testing.assert_frame_equal(samples,fitted_s,check_dtype=False,rtol=1e-10,atol=1e-10)
        pd.testing.assert_frame_equal(taxa,fitted_t,check_dtype=False)
        with np.load(folder/'posterior_observed.npz',allow_pickle=False) as f:
            if not np.array_equal(f['sample_id'],samples.sample_id.to_numpy(str)) or not np.array_equal(f['taxon_id'],taxa.taxon_id.to_numpy(str)):raise ValueError('Order mismatch')
            counts=f['raw_counts'].copy()
        if raw is not None and not np.array_equal(raw,counts):raise ValueError('Raw differs across seeds')
        raw=counts
        est=EventAwareDenoiser.load_checkpoint(folder/'fit.pt');m=est.model_;ix=(samples.tensor_i.to_numpy(),samples.tensor_t.to_numpy())
        if not np.array_equal(m.y[ix].cpu().numpy(),raw):raise ValueError('Checkpoint observations mismatch')
        n=max(4000,est.config.posterior_samples) if not a.draws else a.draws
        if n<100:raise ValueError('At least 100 posterior draws required')
        total=np.zeros(raw.shape,dtype=float);half=total.copy();second=total.copy();torch.manual_seed(task['seed']+20000)
        with torch.no_grad():
            for k in range(n):
                decoded=m.decode(m.values(True));value=torch.softmax(decoded['Z'][ix],dim=-1).cpu().numpy()
                total+=value/n
                second+=value*value/n
                if k<n//2:half+=value/(n//2)
                if k%100==0:print(name,task['seed'],k,'/',n,flush=True)
        if not np.isfinite(total).all() or np.max(abs(total.sum(1)-1))>1e-7:raise ValueError('Invalid posterior composition')
        means.append(total)
        mcvar=np.maximum(second-total*total,0)/(n-1);mcvariances.append(mcvar)
        sources.append(dict(seed=task['seed'],draws=n,checkpoint_sha256=sha(folder/'fit.pt'),done_sha256=sha(folder/'done.json'),
            half_vs_full_max_abs=float(np.max(abs(total-half))),mcse_max=float(np.sqrt(mcvar).max()),mcse_q95=float(np.quantile(np.sqrt(mcvar),.95)),quality=json.loads((folder/'quality.json').read_text(encoding='utf-8'))))
        del est,m
    if len(means)<d['min_seeds']:raise ValueError(f'{name}: only {len(means)} usable seeds; require {d["min_seeds"]}')
    target=a.out/'exports';target.mkdir(parents=True,exist_ok=True)
    file=target/(name+'.npz');temp=file.with_suffix('.tmp')
    with temp.open('wb') as f:np.savez_compressed(f,raw_counts=raw,composition_mean=np.mean(means,axis=0),composition_mean_mcse=np.sqrt(np.sum(mcvariances,axis=0))/len(means),composition_by_seed=np.stack(means),sample_id=samples.sample_id.to_numpy(str),taxon_id=taxa.taxon_id.to_numpy(str))
    temp.replace(file)
    dump(target/(name+'.json'),dict(dataset=name,representation='E[softmax(Z)] positive-abundance composition (all components, absence gate excluded)',
        seeds=sources,missing_seeds=missing,output_sha256=sha(file),plan_sha256=sha(PKG/'plan.json')))

def prepare(a):
    tasks=[];cache={}
    for c in plan()['contrasts']:
        if a.datasets and c['family'] not in a.datasets and c['dataset'] not in a.datasets:continue
        name=c['dataset']
        if name not in cache:
            file=a.out/'exports'/(name+'.npz');audit=json.loads(file.with_suffix('.json').read_text(encoding='utf-8'))
            if audit['output_sha256']!=sha(file) or audit['plan_sha256']!=sha(PKG/'plan.json'):raise ValueError('Export checksum/plan mismatch')
            with np.load(file,allow_pickle=False) as f:cache[name]={k:f[k].copy() for k in f.files if k!='composition_by_seed'}
        data=cache[name];meta=pd.read_csv(PKG/'contrasts'/c['id']/'metadata.csv',dtype={'sample_id':str,'participant':str})
        ix=pd.Index(data['sample_id']).get_indexer(meta.sample_id)
        if (ix<0).any():raise ValueError('Unknown sample')
        counts=data['raw_counts'][ix];raw=counts/counts.sum(1,keepdims=True);den=data['composition_mean'][ix]
        prevalence=pd.DataFrame(counts>0).groupby(meta.participant).max().mean().to_numpy()
        keep=(prevalence>=c['prevalence_threshold'])&(counts.sum(0)>0)
        if keep.sum()<10:raise ValueError('Too few features')
        audit=pd.read_csv(PKG/'data'/name/'taxa.tsv',sep='\t',dtype=str)
        audit['raw_subject_prevalence']=prevalence;audit['tested']=keep
        parent=a.out/'inputs'/c['id'];parent.mkdir(parents=True,exist_ok=True)
        audit.to_csv(parent/'feature_audit.tsv',sep='\t',index=False)
        dump(parent/'contrast.json',c)
        for rep,x in [('raw',raw),('denoised',den)]:
            target=parent/rep;target.mkdir(exist_ok=True);x=x[:,keep];x=x/x.sum(1,keepdims=True)
            table=pd.DataFrame(x,columns=data['taxon_id'][keep]);table.insert(0,'sample_id',meta.sample_id.to_numpy())
            table.to_csv(target/'proportions.csv',index=False);meta.to_csv(target/'metadata.csv',index=False)
            for method in plan()['methods']:tasks.append(dict(index=len(tasks),contrast=c['id'],dataset=name,family=c['family'],representation=rep,method=method))
    dump(a.out/'tasks.json',tasks);print(len(tasks))

def run(a):
    task=json.loads((a.out/'tasks.json').read_text(encoding='utf-8'))[a.index]
    inp=a.out/'inputs'/task['contrast']/task['representation'];dest=a.out/'runs'/task['contrast']/task['representation']/task['method'];dest.mkdir(parents=True,exist_ok=True)
    binary=a.maaslin2_r if task['method']=='maaslin2' else a.r
    # A failed retry must not silently reuse an earlier engine table.
    (dest/'engine_results.csv').unlink(missing_ok=True)
    args=[str(PKG),str(inp),task['method'],str(dest)]
    arguments=(['--vanilla',str(PKG/'run_method.R')]+args if Path(binary).stem.lower()=='rscript' else ['--vanilla','--slave','-f',str(PKG/'run_method.R'),'--args']+args)
    command=r_launch(binary,arguments,task['method'])
    with (dest/'run.log').open('w',encoding='utf-8') as log:
        proc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=r_environment(task['method']))
    features=pd.read_csv(inp.parent/'feature_audit.tsv',sep='\t',dtype={'taxon_id':str})
    result=features.rename(columns={'taxon_id':'taxon'})
    engine=dest/'engine_results.csv'
    if engine.exists():result=result.merge(pd.read_csv(engine,dtype={'taxon':str}),on='taxon',how='left',validate='one_to_one')
    else:result['fit_status']='method_error';result['pvalue']=np.nan;result['fit_message']='See run.log; engine did not produce results'
    result.loc[~result.tested,'fit_status']='filtered_raw_prevalence'
    result.loc[result.tested&result.fit_status.isna(),'fit_status']='no_result'
    result['q_bh_contrast']=bh(result.pvalue.to_numpy())
    for k,v in task.items():result[k]=v
    c=json.loads((inp.parent/'contrast.json').read_text(encoding='utf-8'))
    for k in ['endpoint','reference','positive','estimand','n_samples','n_subjects']:result[k]=c[k]
    result.to_csv(dest/'all_taxa.tsv',sep='\t',index=False)
    dump(dest/'status.json',dict(**task,returncode=proc.returncode,finite_p=int(result.pvalue.notna().sum()),total_taxa=len(result),
        result_sha256=sha(dest/'all_taxa.tsv'),input_sha256=sha(inp/'proportions.csv'),metadata_sha256=sha(inp/'metadata.csv')))
    if proc.returncode:raise SystemExit(proc.returncode)

def collect(a):
    tables=[];missing=[]
    for task in json.loads((a.out/'tasks.json').read_text(encoding='utf-8')):
        f=a.out/'runs'/task['contrast']/task['representation']/task['method']/'all_taxa.tsv'
        if f.exists():tables.append(pd.read_csv(f,sep='\t'))
        else:missing.append(task)
    dump(a.out/'missing_tasks.json',missing)
    if not tables:raise ValueError('No DA outputs')
    allrows=pd.concat(tables,ignore_index=True)
    allrows['q_bh_family']=allrows.groupby(['family','representation','method']).pvalue.transform(lambda x:bh(x.to_numpy()))
    for family,g in allrows.groupby('family'):
        dest=a.out/'Results'/family/'02_differential_abundance';dest.mkdir(parents=True,exist_ok=True)
        g.to_csv(dest/'all_taxa_all_methods.tsv.gz',sep='\t',index=False)
        for method,h in g.groupby('method'):
            h.to_csv(dest/(method+'_all_taxa.tsv'),sep='\t',index=False)
            keys=['contrast','dataset','taxon','endpoint','estimand']
            comparison=h[h.representation.eq('raw')].merge(h[h.representation.eq('denoised')],on=keys,suffixes=('_raw','_denoised'),validate='one_to_one')
            comparison['raw_nonsignificant_denoised_significant']=(comparison.q_bh_family_raw>=.05)&(comparison.q_bh_family_denoised<.05)
            if 'estimate_raw' in comparison:
                comparison['same_direction']=comparison.estimate_raw*comparison.estimate_denoised>0
            comparison.to_csv(dest/(method+'_raw_vs_denoised.tsv'),sep='\t',index=False)
    allrows.groupby(['dataset','representation','method','fit_status']).size().rename('n').to_csv(a.out/'run_status_summary.tsv',sep='\t')
    dump(a.out/'COLLECTION_STATUS.json',dict(missing_tasks=len(missing),rows=len(allrows),complete=not missing,
        tested_without_finite_p=int((allrows.tested & allrows.pvalue.isna()).sum()),
        note='Complete means files collected, not all models converged. Check run_status_summary and every fit_status. Conditional plug-in inference; no calibrated denoising uncertainty.'))
    if missing:raise SystemExit(2)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['export','prepare','run','collect','verify','check_r']);p.add_argument('--out',type=Path,default=Path('output'))
    p.add_argument('--index',type=int,default=0);p.add_argument('--old-fits');p.add_argument('--new-fits');p.add_argument('--draws',type=int,default=0)
    p.add_argument('--datasets',nargs='+');p.add_argument('--r',default=os.environ.get('R_BIN','/public/software/VersionHub/R/4.2.2/R'))
    p.add_argument('--maaslin2-r',default=os.environ.get('MAASLIN2_R_BIN',os.environ.get('R_BIN','/public/software/VersionHub/R/4.2.2/R')))
    a=p.parse_args();a.out=a.out.resolve();verify()
    if a.action!='verify':globals()[a.action](a)

if __name__=='__main__':main()
