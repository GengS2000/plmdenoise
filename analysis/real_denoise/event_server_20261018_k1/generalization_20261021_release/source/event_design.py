"""Event designs and support audits for the real-data event workflow."""
import numpy as np
import pandas as pd


def event_design(cohort, samples, metadata):
    subjects=samples.sort_values('tensor_i').drop_duplicates('subject_id').subject_id.tolist()
    m=metadata.set_index('sample_id').loc[samples.sample_id].copy()
    if not np.array_equal(m.subject_id.astype(str),samples.subject_id.astype(str)):
        raise ValueError('Event metadata subject IDs disagree with frozen samples')
    if not np.allclose(pd.to_numeric(m.time),samples.time):
        raise ValueError('Event metadata time differs from count time')
    m.index=samples.index
    if cohort=='DIABIMMUNE':
        col='solid_start_approx_age_months'
        ages=pd.to_numeric(m[col],errors='coerce')
        if (ages.groupby(samples.subject_id).nunique()>1).any():
            raise ValueError('Inconsistent food introduction ages')
        times=ages.groupby(samples.subject_id).first().reindex(subjects).to_numpy()[:,None]*30.4375
        names=['solid_food'];kind='food_introduction_proxy'
        note='Recorded approximate introduction month x 30.4375; not exact day or fiber dose.'
    elif cohort=='FARMM':
        # All participants received antibiotics day 6-8 (PEG day 7) after the
        # day 1-5 diet run-in. One shared channel: the three diet-specific
        # channels of the previous design summed to a population time function
        # that is collinear with the common trend. Diet differences in the
        # response are estimated from individual event gains downstream.
        times=np.full((len(subjects),1),6.)
        names=['abx_PEG'];kind='protocol_antibiotics_PEG'
        note=('Protocol antibiotics/PEG start day 6, shared by all diets; recovery is part of the '
              'post-event response curve. Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC8022197/')
    else:
        names=['unavailable'];times=np.full((len(subjects),1),np.nan);kind='none'
        note='ECAM has no verified event date; no-event model.'
    records=[]
    for i,sid in enumerate(subjects):
        for j,name in enumerate(names):
            if np.isfinite(times[i,j]):
                records.append(dict(subject_id=sid,event=name,start=times[i,j],stop=np.nan,kind=kind))
    return times,names,pd.DataFrame(records,columns=['subject_id','event','start','stop','kind']),note


def audit_design(s, events, names, cohort):
    support=[];overlap=[];distribution=[]
    window=90. if cohort=='DIABIMMUNE' else 4.
    bin_width=30.4375 if cohort=='DIABIMMUNE' else 1.
    for j,name in enumerate(names):
        dates=events[:,j];finite=dates[np.isfinite(dates)]
        if not len(finite):continue
        distribution.append(dict(event=name,n_exposed=len(finite),n_unique_dates=len(np.unique(finite)),
            min_day=finite.min(),q25_day=np.quantile(finite,.25),median_day=np.median(finite),
            q75_day=np.quantile(finite,.75),max_day=finite.max(),sd_day=np.std(finite)))
        for i,g in s.groupby('tensor_i'):
            tau=dates[int(i)]
            if not np.isfinite(tau):continue
            pre=int(((g.time<tau)&(g.time>=tau-window)).sum())
            post=int(((g.time>=tau)&(g.time<=tau+window)).sum())
            support.append(dict(subject_id=g.subject_id.iloc[0],event=name,start_day=tau,
                n_pre_window=pre,n_post_window=post,window_day=window,supported_2_each=pre>=2 and post>=2,
                last_pre_day=g.loc[g.time<tau,'time'].max(),first_post_day=g.loc[g.time>=tau,'time'].min()))
        # Visit-based overlap, count unique subjects; mixed within-bin subjects counted separately.
        temp=s.copy();temp['event_date']=dates[temp.tensor_i.to_numpy()]
        temp=temp[np.isfinite(temp.event_date)].copy()
        temp['bin']=np.floor(temp.time/bin_width).astype(int)
        temp['exposed']=temp.time>=temp.event_date
        for b,g in temp.groupby('bin'):
            states=g.groupby('subject_id').exposed.agg(['min','max'])
            n0=int((~states['max']).sum());n1=int(states['min'].sum())
            mixed=int((states['min']!=states['max']).sum())
            overlap.append(dict(event=name,age_bin_start=b*bin_width,age_bin_end=(b+1)*bin_width,
                unexposed_only_subjects=n0,exposed_only_subjects=n1,mixed_subjects=mixed,
                overlap_3_each=n0>=3 and n1>=3))
    return pd.DataFrame(support),pd.DataFrame(overlap),pd.DataFrame(distribution)
