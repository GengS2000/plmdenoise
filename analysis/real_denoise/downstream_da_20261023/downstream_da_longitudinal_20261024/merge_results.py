"""Replace only the two revised DA directories after validating every task."""
from pathlib import Path
import argparse, datetime, json, shutil
import pipeline as p

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--existing',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    p.verify();out=a.out.resolve();existing=a.existing.resolve()
    if out==existing:raise ValueError('Output must be isolated')
    tasks=json.loads((out/'tasks.json').read_text(encoding='utf-8'))
    expected={(c['id'],r,m) for c in p.plan()['contrasts'] for r in p.plan()['representations'] for m in p.plan()['methods']}
    assert {(t['contrast'],t['representation'],t['method']) for t in tasks}==expected
    for t in tasks:
        folder=out/'runs'/t['contrast']/t['representation']/t['method']
        s=json.loads((folder/'status.json').read_text(encoding='utf-8'))
        if s['returncode']!=0 or s['finite_p']==0 or s['result_sha256']!=p.sha(folder/'all_taxa.tsv'):raise ValueError('Failed/unverified task; merge stopped: '+str(folder))
    status=json.loads((out/'COLLECTION_STATUS.json').read_text(encoding='utf-8'))
    if not status['complete']:raise ValueError('Incomplete collection')
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    backup=existing/'revision_backups'/stamp
    for family in ['SOIL','GINGIVITIS']:
        source=out/'Results'/family/'02_differential_abundance'
        if not (source/'all_taxa_all_methods.tsv.gz').exists():raise ValueError('Missing collection '+family)
    for family in ['SOIL','GINGIVITIS']:
        source=out/'Results'/family/'02_differential_abundance'
        dest=existing/'Results'/family/'02_differential_abundance'
        staging=dest.parent/('02_differential_abundance_staging_'+stamp)
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copytree(source,staging)
        if dest.exists():
            saved=backup/family/'02_differential_abundance';saved.parent.mkdir(parents=True,exist_ok=True)
            dest.rename(saved)
        staging.rename(dest)
    report=dict(design='time_adjusted_20261024',source=str(out),backup=str(backup),families_replaced=['SOIL','GINGIVITIS'],tasks=len(tasks),collection_status=status)
    p.dump(existing/'Results'/'LONGITUDINAL_REVISION.json',report)
    shutil.copy2(out/'run_status_summary.tsv',existing/'Results'/'longitudinal_run_status_summary.tsv')
    print('Merged into',existing/'Results','; previous results backed up at',backup)

if __name__=='__main__':main()
