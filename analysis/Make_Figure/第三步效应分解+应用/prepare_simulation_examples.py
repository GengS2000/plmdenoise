"""Reproduce two median-seed examples using the frozen simulation and fit settings.

Never edits the original 450-task results. Use --scenario 3 or 4 to run separately.
"""
from pathlib import Path
import argparse
import json
import sys
import time
import hashlib
import platform
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SIM=ROOT/'simulation_results/三个量分解的结果二'

def selected_task(scenario):
    plan=json.loads((SIM/'plan.json').read_text(encoding='utf-8'))
    rows=[json.loads(p.read_text(encoding='utf-8'))['row'] for p in sorted((SIM/'tasks').glob('*.json'))]
    d=pd.DataFrame(rows)
    g=d[d.scenario.eq(scenario)].groupby('seed').struct_event_r2_m4.mean().sort_index()
    seed=int((g-g.median()).abs().idxmin())
    task=next(t for t in plan['tasks'] if t['scenario']==scenario and t['seed']==seed and t['condition']=='base')
    return task,dict(selection='Closest seed to median scenario event R2; conditions averaged within seed. Base condition displayed.',
                     scenario=scenario,seed=seed,seed_event_r2=float(g.loc[seed]),median_seed_event_r2=float(g.median()),task_id=task['task_id'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--scenario',type=int,choices=[3,4],required=True);p.add_argument('--threads',type=int,default=2)
    a=p.parse_args();task,selection=selected_task(a.scenario)
    out=HERE/'source_data/local_simulation_refits';out.mkdir(parents=True,exist_ok=True)
    dest=out/f'scenario_{a.scenario}_example.npz'
    if dest.exists():
        print('Existing example retained:',dest,flush=True);return
    sys.path.insert(0,str(SIM/'source'))
    import torch
    from run_decomposition import run_task
    from simulate_microbiome import SimulationConfig,simulate
    from run_validation import training_mask
    from decomposition import clr,truth_components
    print('Starting exact frozen-settings local refit',selection,flush=True)
    started=time.monotonic()
    result=run_task(task,out,'figure-local-refit',threads=a.threads,save_components=True)
    truth=simulate(SimulationConfig(**task['simulation']))
    mask,_=training_mask(truth,task['holdout_fraction'],task['seed']);truth['mask']=mask
    arrays=dict(np.load(out/'components'/f"{task['task_id']}.npz",allow_pickle=False))
    arrays.update({f'true_{k}_clr':clr(v).astype(np.float32) for k,v in truth_components(truth).items()})
    arrays.update(times=truth['times'],event_times=truth['event_times'],visit_mask=mask.any(-1),
                  raw_clr=clr(np.log(truth['Y']+.5)).astype(np.float32),
                  gradual_beta=truth['gradual_beta'],event_amplitude=truth['event_amplitude'])
    np.savez_compressed(dest,**arrays)
    result.update(selection=selection,provenance='Local exact-settings rerun for representative trajectories; original 450 metrics untouched.',
                  elapsed_seconds=time.monotonic()-started,python=platform.python_version(),torch=torch.__version__)
    (out/f'scenario_{a.scenario}_refit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)),encoding='utf-8')
    print('Completed',dest,'seconds',time.monotonic()-started,flush=True)

if __name__=='__main__':main()
