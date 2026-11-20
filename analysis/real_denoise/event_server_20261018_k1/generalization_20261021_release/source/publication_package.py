"""Freeze and package the additive publication extension without changing v2."""
from pathlib import Path
import argparse
import json
import tarfile
from common import digest,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,default=Path(__file__).resolve().parent.parent)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--arcname',default='event_publication_20261018',help='top-level directory inside the archive')
    a=p.parse_args();b=a.bundle.resolve()
    from event_run import verify
    from publication_validation import tasks
    from publication_dispatch import table_jobs,check
    plan=verify(b)
    write_json(b/'PUBLICATION_PLAN.json',dict(base_plan_sha256=digest(b/'plan.json'),
        table_tasks=table_jobs(),validation_tasks=tasks(plan),
        inference='conditional on fitted representations; no external age prediction claim'))
    paths=[b/'PUBLICATION_README.md',b/'PUBLICATION_PLAN.json',b/'LOCAL_VALIDATION.md']
    paths+=sorted((b/'source').glob('publication_*'))
    paths+=[b/'source'/'test_publication.py']
    write_json(b/'PUBLICATION_MANIFEST.json',dict(base_plan_sha256=digest(b/'plan.json'),
        files={f.relative_to(b).as_posix():digest(f) for f in paths if f.is_file()}))
    check(b)
    archive=a.archive.resolve()
    if archive.exists():raise ValueError('Archive exists; choose a new name')
    archive.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive,'w:gz') as tar:
        for f in sorted(b.rglob('*')):
            if not f.is_file() or '__pycache__' in f.parts or f.suffix=='.pyc':continue
            # Only frozen base payload and extension payload; omit local scratch/tests.
            rel=f.relative_to(b).as_posix()
            allowed=set(plan['files'])|{str(x.relative_to(b).as_posix()) for x in paths}|{'plan.json','PLAN.sha256','PUBLICATION_MANIFEST.json'}
            if rel in allowed:tar.add(f,arcname=a.arcname+'/'+rel)
    checksum=digest(archive)
    # Bytes, not text: a Windows text write would add CR and break sha256sum -c.
    archive.with_name(archive.name+'.sha256').write_bytes((checksum+'  '+archive.name+'\n').encode('ascii'))
    print(archive);print(checksum)


if __name__=='__main__':main()
