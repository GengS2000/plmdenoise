"""Export small, portable metadata sidecars without changing frozen fit inputs."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from common import digest, write_json


def prepare(raw, out):
    raw, out = Path(raw), Path(out)
    if out.exists():
        raise ValueError('Metadata output must be new')
    out.mkdir(parents=True)
    files = {}
    sources = {
        'DIABIMMUNE': raw/'DIABIMMUNE_data/11884_20190508-173103-added-month-abx.txt',
        'ECAM': raw/'ECAM_data/genus_metadata_cleaned_q2.csv',
        'FARMM': raw/'FARMM_data/20200619_farmm_metadata.tsv'}
    for cohort, path in sources.items():
        if cohort == 'DIABIMMUNE':
            m = pd.read_csv(path, sep='\t', dtype=str).rename(columns={'#SampleID':'sample_id'})
            m['subject_id'] = m.host_subject_id
            m['time'] = pd.to_numeric(m.age_at_collection)
        elif cohort == 'ECAM':
            m = pd.read_csv(path, index_col=0, dtype=str).rename(columns={'SampleID':'sample_id'})
            # Match prepare.read_ecam's numeric CSV inference; raw IDs include spaces.
            m['subject_id'] = pd.to_numeric(m.host_subject_id,errors='raise').astype('int64').astype(str)
            m['time'] = pd.to_numeric(m.day_of_life)
        else:
            m = pd.read_csv(path, sep='\t', dtype=str).rename(columns={'SampleID':'sample_id'})
            m['subject_id'] = m.SubjectID
            m['time'] = pd.to_numeric(m.study_day, errors='coerce')
        if m.sample_id.duplicated().any():
            raise ValueError(f'Duplicate sample ID: {cohort}')
        m.to_csv(out/f'{cohort}_metadata.tsv', sep='\t', index=False)
        events = []
        for subject, g in m.groupby('subject_id'):
            if cohort == 'DIABIMMUNE':
                columns = [('solid_food','solid_start_approx_age_months')]
                columns += [(name, f'dietary_compound_{name}_start_age_month')
                            for name in ['oat','barley','rye']]
                for name, col in columns:
                    values = pd.to_numeric(g[col], errors='coerce').dropna().unique()
                    if len(values) > 1:
                        raise ValueError(f'Inconsistent {col}: {subject}')
                    if len(values) and values[0] >= 0:
                        events.append(dict(subject_id=subject,event=name,start=values[0]*30.4375,
                                           stop=np.nan,kind='food_introduction_proxy'))
            elif cohort == 'FARMM' and g.HostSpecies.eq('Human').any():
                events.append(dict(subject_id=subject,event='recovery',start=9.,stop=np.nan,
                                   kind='antibiotics_PEG_recovery'))
        pd.DataFrame(events, columns=['subject_id','event','start','stop','kind']).to_csv(
            out/f'{cohort}_events.tsv', sep='\t', index=False)
        files[str(path.resolve())] = digest(path)
    write_json(out/'manifest.json', dict(source_sha256=files,
        months_to_days=30.4375, caveats=['Food introduction is not fiber dose.',
        'FARMM day 9 is recovery onset, not fiber start or stop.',
        'ECAM has no inferred dietary transition.', 'Original metadata values preserved.']))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    prepare(a.raw, a.out)
