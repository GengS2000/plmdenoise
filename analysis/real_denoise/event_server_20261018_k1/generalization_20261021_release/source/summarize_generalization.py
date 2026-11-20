"""Audit only the requested cohorts and create a cross-site oral application figure."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from common import digest,write_json
from cohort_analysis import bh
from selective_submit import select


def summarize(bundle,out,datasets,dpi=600):
    from event_run import verify
    plan=verify(bundle);cohorts=select(plan,datasets);status=[];tables=[]
    for c in cohorts:
        path=out/c;file=path/'complete.json';ok=False
        if file.exists():
            done=json.loads(file.read_text(encoding='utf-8'));ok=done.get('status')=='complete'
            if 'plan_sha256' in done and done['plan_sha256']!=digest(bundle/'plan.json'):ok=False
            for name,h in done['files'].items():
                if not (path/name).exists() or digest(path/name)!=h:ok=False
            if ok:
                t=pd.read_csv(path/'taxon_contrasts.tsv',sep='\t');t['dataset']=c;tables.append(t)
        status.append(dict(dataset=c,status='complete' if ok else 'missing_or_invalid'))
    summary=out/'summary';summary.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(status).to_csv(summary/'analysis_status.tsv',sep='\t',index=False)
    write_json(summary/'audit.json',dict(datasets=cohorts,complete=all(x['status']=='complete' for x in status)))
    if not all(x['status']=='complete' for x in status):raise RuntimeError('Selected analysis incomplete; see summary/analysis_status.tsv')
    alltests=pd.concat(tables,ignore_index=True)
    oral=alltests[alltests.dataset.str.startswith('GINGIVITIS_')].copy()
    if len(oral):
        oral['q_bh_across_selected_sites']=oral.groupby('representation').p_signflip.transform(lambda x:bh(x.to_numpy()))
        oral.to_csv(summary/'gingivitis_all_sites_contrasts.tsv',sep='\t',index=False)
        names=[c for c in cohorts if c.startswith('GINGIVITIS_')]
        fig,axes=plt.subplots(len(names),3,figsize=(13,3.1*len(names)),squeeze=False,layout='constrained')
        plt.rcParams['pdf.fonttype']=42
        for i,c in enumerate(names):
            path=out/c;sc=pd.read_csv(path/'ordination_scores.tsv',sep='\t')
            ax=axes[i,0]
            for rep,color in [('raw','#B8BCC5'),('denoised','#4F587D')]:
                d=sc[sc.basis.eq('shared_raw_basis')&sc.representation.eq(rep)]
                ax.scatter(d.PC1,d.PC2,s=5,c=color,alpha=.6,label=rep,rasterized=True)
            ax.set(title=c.replace('GINGIVITIS_','').replace('_',' '),xlabel='PC1 (raw basis)',ylabel='PC2 (raw basis)');ax.legend(fontsize=7)
            d=oral[oral.dataset.eq(c)]
            raw=d[d.representation.eq('raw')];den=d[d.representation.eq('denoised')]
            m=raw.merge(den,on=['taxon_id','contrast'],suffixes=('_raw','_den'))
            ax=axes[i,1];ax.scatter(m.mean_effect_raw,m.mean_effect_den,s=6,c=np.where(m.q_bh_across_selected_sites_den.lt(.05),'#C68DC0','#4F587D'),alpha=.6,rasterized=True)
            ax.set(title='Differential inference (all contrasts)',xlabel='Raw CLR effect',ylabel='Denoised CLR effect')
            cov=pd.read_csv(path/'component_covariance.tsv',sep='\t')
            ns=['baseline','common','individual','event'];matrix=cov.pivot(index='component_a',columns='component_b',values='fraction_of_total_variance').reindex(index=ns,columns=ns).to_numpy()
            ax=axes[i,2];v=max(abs(matrix).max(),.01);im=ax.imshow(matrix,cmap='PuOr_r',vmin=-v,vmax=v)
            ax.set_xticks(range(4),ns,rotation=20,fontsize=7);ax.set_yticks(range(4),ns,fontsize=7);ax.set_title('Component covariance / total variance')
            fig.colorbar(im,ax=ax,shrink=.75)
        fig.suptitle('GINGIVITIS | independent site models, same 41-subject source cohort',fontsize=14)
        stem=summary/'Figure_GINGIVITIS_application'
        fig.savefig(str(stem)+'_preview.png',dpi=120)
        for ext in ['png','pdf']:fig.savefig(str(stem)+'.'+ext,dpi=dpi)
        plt.close(fig)
        with Image.open(str(stem)+'.png') as im:im.convert('RGB').save(str(stem)+'.tiff',compression='tiff_lzw',dpi=(dpi,dpi))
    write_json(summary/'manifest.json',dict(datasets=cohorts,files={p.name:digest(p) for p in summary.iterdir() if p.is_file() and p.name!='manifest.json'}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--datasets',nargs='+',required=True);p.add_argument('--dpi',type=int,default=600)
    a=p.parse_args();summarize(a.bundle.resolve(),a.out.resolve(),a.datasets,a.dpi)
