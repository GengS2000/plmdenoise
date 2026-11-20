from pathlib import Path
import json, shutil, subprocess, sys
import numpy as np
import pandas as pd


from threadpoolctl import threadpool_limits
import predict_full_microbiome as p
old=p.old
OUT=old.HERE/'下颌邻面候选菌消融和功能分析'
CAND=['ASV48','ASV2','ASV35','ASV18','ASV5','ASV175','ASV131']
OUT.mkdir(exist_ok=True)
SITE='lower_jaw_interproximal'

def main():
    clin,labels=old.load_clinical()
    (OUT/'config.json').write_text(json.dumps(dict(candidates=CAND,windows=old.WINDOWS,repeats=10,script='predict_full_microbiome.py functions reused without changes',clr='fixed full-taxon CLR, column subset only',interpretation='Exploratory fixed candidates chosen from this cohort; transductive denoising; WF retrospective'),indent=2),encoding='utf-8')
    shutil.copy2(Path(p.__file__),OUT/'predict_full_microbiome_reference.py')
    for w in old.WINDOWS:
        meta,mats=old.load_window(SITE,w);subjects=sorted(set(meta.subject_id)&set(labels.index));Xc=old.clinical_features(clin,subjects,w).to_numpy();y=labels.loc[subjects].to_numpy()
        assert set(CAND)<=set(mats['raw'].columns)
        for mode in ['all','keep7','drop7']:
            target=OUT/mode;target.mkdir(exist_ok=True)
            if (target/f'{SITE}_{w}_contrasts.tsv').exists():continue
            cols=[c for c in mats['raw'].columns if mode=='all' or (c in CAND)==(mode=='keep7')]
            X={s:[old.flatten(meta,mat[cols],subjects,old.WIN_DAYS[w])[0]] for s,mat in mats.items()}
            p.evaluate(SITE,w,subjects,Xc,X,y,10,target)
    rows=[]
    for mode in ['all','keep7','drop7']:
        for f in (OUT/mode).glob('*_contrasts.tsv'):
            d=pd.read_csv(f,sep='\t');d['ablation']=mode;rows.append(d)
    pd.concat(rows).to_csv(OUT/'summary.tsv',sep='\t',index=False)
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
