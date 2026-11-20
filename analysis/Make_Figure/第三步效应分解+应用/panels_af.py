"""Original-server component trajectories and per-seed event posterior curves.

Called by build_figure.py; no model is refitted. Frozen simulation code is used
only to regenerate its deterministic ground truth from the stored task config.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def prepare(b, root):
    sim=root/'simulation_results/三个量分解的结果二'
    sys.path.insert(0,str(sim/'source'))
    from simulate_microbiome import SimulationConfig,simulate
    from run_validation import training_mask
    from decomposition import truth_components,clr
    for name in ['simulate_microbiome.py','decomposition.py','run_validation.py']:
        b.record(sim/'source'/name)
    plan=json.loads(b.record(sim/'plan.json').read_text(encoding='utf-8'))
    scores=b.load_tsv('B_simulation_seed_R2.tsv')
    rows=[];dense=[];selections=[]
    for scenario in [3,4]:
        ss=scores[scores.scenario.eq(scenario)].set_index('seed').struct_event_r2_m4
        seed=int((ss-ss.median()).abs().sort_index().idxmin())
        task=next(t for t in plan['tasks'] if t['scenario']==scenario and t['seed']==seed and t['condition']=='base')
        path=sim/'components'/(task['task_id']+'.npz')
        with np.load(b.record(path),allow_pickle=False) as f:
            stored=json.loads(str(f['metadata_json']))
            assert stored['simulation']==task['simulation']
            est={k:f['m4_'+k+'_clr_mean'].copy() for k in ['common','individual','event']}
            sd={k:f['m4_'+k+'_clr_sd'].copy() for k in est}
        truth=simulate(SimulationConfig(**task['simulation']))
        mask,_=training_mask(truth,task['holdout_fraction'],seed)
        truth['mask']=mask
        parts={k:clr(v) for k,v in truth_components(truth).items()}
        # Truth-only, deterministic representative choice: no fitted errors used.
        candidates=np.flatnonzero((truth['event_amplitude'][:,0]>0)&(np.abs(truth['gradual_beta'])>0))
        if not len(candidates):raise ValueError('No taxon with both generated trend and event')
        loading=np.abs(truth['loadings'][candidates,0])
        j=int(candidates[np.argmin(np.abs(loading-np.median(loading)))])
        exposed=np.flatnonzero(np.isfinite(truth['event_times'][:,0]))
        energy=np.std(parts['individual'][exposed,:,j],axis=1)
        i=int(exposed[np.argmin(np.abs(energy-np.median(energy)))])
        tau=float(truth['event_times'][i,0]);tt=truth['times'][i]
        selections.append(dict(scenario=scenario,seed=seed,task_id=task['task_id'],condition='base',
            seed_event_R2=float(ss.loc[seed]),median_event_R2=float(ss.median()),taxon_index=j,
            subject_index=i,event_day=tau,source=str(path.relative_to(root))))
        for c in est:
            for q,t in enumerate(tt):
                rows.append(dict(scenario=scenario,seed=seed,component=c,taxon_index=j,subject_index=i,
                    time=t,relative_day=t-tau,truth=parts[c][i,q,j],estimate=est[c][i,q,j],
                    lower=est[c][i,q,j]-1.96*sd[c][i,q,j],upper=est[c][i,q,j]+1.96*sd[c][i,q,j]))
        # Dense event truth comes directly from the simulator, CLR across all taxa.
        h=truth['H_plus_true'][:,0,:];h=h-h.mean(0,keepdims=True)
        grid=truth['event_truth_grid'];ok=(grid>=tt.min()-tau)&(grid<=tt.max()-tau)
        dense.extend(dict(scenario=scenario,relative_day=x,truth=y) for x,y in zip(grid[ok],h[j,ok]))
    b.save_tsv('A_component_trajectories.tsv',pd.DataFrame(rows))
    b.save_tsv('A_dense_event_truth.tsv',pd.DataFrame(dense))
    b.save_tsv('A_example_selection.tsv',pd.DataFrame(selections))
    (b.DATA/'A_selection_rule.json').write_text(json.dumps(dict(
        seed='Closest to median event R2 after averaging conditions within each seed; base condition shown.',
        taxon='Positive event amplitude and nonzero generated trend; absolute factor loading nearest candidate median.',
        subject='Exposed subject with generated individual temporal SD nearest exposed-subject median.',
        uncertainty='Original-server CLR component mean +/- 1.96 posterior SD; normal approximation, not exact quantiles.',
        truth='Deterministic regeneration using frozen simulator and original task config; original training-mask centering.',
        event='Dense generated event truth; estimated component at actual visits; no interpolation across event onset.'),indent=2),encoding='utf-8')

    seedrows=[];pooledrows=[];choices=[];checks=[]
    selected=b.load_tsv('E_selected_event_taxa.tsv')
    for cohort in ['FARMM','DIABIMMUNE']:
        eligible=selected[selected.cohort.eq(cohort)].copy()
        # Strongest negative and positive responses, plus a well-detected remaining response.
        negative=eligible[eligible.mean_supported_effect<0].sort_values('mean_supported_effect').head(1)
        positive=eligible[eligible.mean_supported_effect>0].sort_values('mean_supported_effect',ascending=False).head(1)
        picked=pd.concat([negative,positive]).drop_duplicates('taxon_id')
        remaining=eligible[~eligible.taxon_id.isin(picked.taxon_id)].sort_values(['sample_detection','abs_effect'],ascending=False)
        picked=pd.concat([picked,remaining.head(3-len(picked))]).reset_index(drop=True)
        files=sorted((root/'实际数据分析的结果/K=1/tasks').glob('*_'+cohort+'_event_*/posterior_draws.npz'))
        allcurves=[];refids=None;refgrid=None;refsup=None
        for p in files:
            seed=int(p.parent.name.rsplit('_',1)[1])
            with np.load(b.record(p),allow_pickle=False) as z:
                ids=z['taxon_id'].copy();grid=z['relative_day'].copy();sup=z['curve_supported'][0].copy()
                c=z['event_curve'].astype(np.float64);c-=c.mean(1,keepdims=True);c=c[:,:,0,:]
            if refids is not None:
                assert np.array_equal(refids,ids) and np.array_equal(refgrid,grid) and np.array_equal(refsup,sup)
            refids,refgrid,refsup=ids,grid,sup
            ix=[int(np.flatnonzero(ids==tid)[0]) for tid in picked.taxon_id]
            c=c[:,ix,:];allcurves.append(c)
            for n,row in picked.iterrows():
                for x,y,sp in zip(grid,c[:,n].mean(0),sup):
                    seedrows.append(dict(cohort=cohort,name=row['name'],taxon_id=row.taxon_id,order=n,seed=seed,relative_day=x,mean=y,supported=sp))
        if not allcurves:raise ValueError('Missing original posterior draws: '+cohort)
        # Every successful seed has exactly 200 draws: concatenation gives equal seed weight.
        assert len({len(x) for x in allcurves})==1
        pooled=np.concatenate(allcurves);mean=pooled.mean(0);lo,hi=np.quantile(pooled,[.025,.975],axis=0)
        existing=b.load_tsv('E_'+cohort+'_heatmap.tsv')
        delta=[]
        for n,row in picked.iterrows():
            old=existing[existing.taxon_id.eq(row.taxon_id)].sort_values('relative_day')
            assert np.allclose(old.relative_day,refgrid)
            delta.append(float(np.max(np.abs(mean[n]-old.clr_effect_mean.to_numpy()))))
            for q,x in enumerate(refgrid):
                pooledrows.append(dict(cohort=cohort,name=row['name'],taxon_id=row.taxon_id,order=n,
                    relative_day=x,mean=mean[n,q],lower=lo[n,q],upper=hi[n,q],supported=refsup[q],
                    n_seeds=len(files),n_draws=len(pooled),sample_detection=row.sample_detection))
        assert max(delta)<1e-5, 'Original draws do not match E pooled analysis'
        picked['selection']='Most negative, most positive, then highest raw detection among remaining E taxa'
        choices.append(picked)
        checks.append(dict(cohort=cohort,n_seeds=len(files),n_draws=len(pooled),max_abs_difference_from_E=max(delta)))
    b.save_tsv('F_seed_event_curves.tsv',pd.DataFrame(seedrows))
    b.save_tsv('F_pooled_event_curves.tsv',pd.DataFrame(pooledrows))
    b.save_tsv('F_selected_taxa.tsv',pd.concat(choices,ignore_index=True))
    (b.DATA/'AF_validation.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    provenance=json.loads((b.DATA/'provenance.json').read_text(encoding='utf-8'))
    provenance['source_hashes'].update(b.SOURCE_HASHES);provenance['deferred_panels']=[]
    provenance['added_panels']='A: original simulation components with regenerated truth; F: original posterior draws'
    provenance['AF_source_code']='panels_af.py'
    (b.DATA/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')


def panel_a(b,fig,gs):
    b.title(fig,gs,'A','Generated and estimated component trajectories')
    sub=gs.subgridspec(2,3,wspace=.25,hspace=.58)
    d=b.load_tsv('A_component_trajectories.tsv');dense=b.load_tsv('A_dense_event_truth.tsv')
    sel=b.load_tsv('A_example_selection.tsv');labels=['Population trend','Individual dynamics','Event response']
    for r,sc in enumerate([3,4]):
        s=sel[sel.scenario.eq(sc)].iloc[0]
        for k,c in enumerate(['common','individual','event']):
            ax=fig.add_subplot(sub[r,k]);v=d[(d.scenario==sc)&d.component.eq(c)]
            x=v.relative_day.to_numpy() if c=='event' else v.time.to_numpy();color=b.COLORS[c]
            if c=='event':
                tr=dense[dense.scenario.eq(sc)];ax.plot(tr.relative_day,tr.truth,'--',color=b.COLORS['black'],lw=1.15,zorder=4)
                for mask in [x<0,x>=0]:
                    ax.fill_between(x[mask],v.lower.to_numpy()[mask],v.upper.to_numpy()[mask],color=color,alpha=.22,lw=0)
                    ax.plot(x[mask],v.estimate.to_numpy()[mask],color=color,lw=1.6,marker='o',ms=2.8)
                ax.axvline(0,color=b.COLORS['grey'],ls=':',lw=.8);ax.set_xlabel('Days from event')
            else:
                ax.fill_between(x,v.lower,v.upper,color=color,alpha=.22,lw=0)
                ax.plot(x,v.estimate,color=color,lw=1.6,marker='o',ms=2.4)
                ax.plot(x,v.truth,'--',color=b.COLORS['black'],lw=1.15,zorder=4);ax.set_xlabel('Follow-up day')
            ax.axhline(0,color=b.COLORS['grey'],lw=.55,zorder=0)
            if r==0:ax.set_title(labels[k],color=color,fontweight='bold',pad=5)
            if k==0:
                ax.set_ylabel(('NB decay' if sc==3 else 'NB step')+'\nCLR component')
                ax.text(.02,.97,f'Seed {int(s.seed)} · taxon {int(s.taxon_index)+1}\nsubject {int(s.subject_index)+1}',transform=ax.transAxes,va='top',fontsize=6.5,
                    bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=1))
            b.tidy(ax)
    box=gs.get_position(fig)
    fig.legend(handles=[Line2D([],[],color=b.COLORS['black'],ls='--',label='Generated truth'),
        Line2D([],[],color=b.COLORS['ink'],marker='o',ms=2.5,label='Posterior mean at visits'),
        Patch(facecolor=b.COLORS['lavender'],alpha=.6,label='Mean ± 1.96 posterior SD')],
        loc='lower right',bbox_to_anchor=(box.x1,box.y1+.008),ncol=3,fontsize=6.8,columnspacing=1.2,handlelength=2)


def panel_f(b,fig,gs):
    b.title(fig,gs,'F','Representative event responses and stability across fitted seeds')
    sub=gs.subgridspec(2,3,wspace=.24,hspace=.66)
    d=b.load_tsv('F_pooled_event_curves.tsv');seeds=b.load_tsv('F_seed_event_curves.tsv')
    linecolors=[b.COLORS['purple'],b.COLORS['pink'],b.COLORS['ice']]
    for r,cohort in enumerate(['FARMM','DIABIMMUNE']):
        alltimes=b.load_tsv('E_'+cohort+'_sample_times.tsv').relative_day.to_numpy()
        for k in range(3):
            ax=fig.add_subplot(sub[r,k]);v=d[(d.cohort==cohort)&d.order.eq(k)].sort_values('relative_day')
            supported=v.loc[v.supported,'relative_day']
            if supported.empty:raise ValueError('No supported event curve: '+cohort)
            v=v[v.relative_day.between(supported.min(),supported.max())]
            name=v.name.iloc[0].replace('_',' ');x=v.relative_day.to_numpy();ok=v.supported.to_numpy(bool)
            ax.axhline(0,color=b.COLORS['grey'],lw=.6,zorder=0)
            ax.fill_between(x,v.lower,v.upper,where=ok,color=b.COLORS['lavender'],alpha=.6,lw=0)
            for j,(seed,s) in enumerate(seeds[(seeds.cohort==cohort)&seeds.order.eq(k)].groupby('seed')):
                s=s[s.relative_day.between(x.min(),x.max())]
                ax.plot(s.relative_day,np.where(s.supported,s['mean'],np.nan),color=linecolors[j],lw=.8,alpha=.95,zorder=4)
            ax.plot(x,np.where(ok,v['mean'],np.nan),color=b.COLORS['ink'],lw=1.9,zorder=3)
            # Display supported post-event times; omit the structural pre-event zero.
            postunsupported=(x>0)&~ok
            if postunsupported.any():ax.axvspan(x[postunsupported].min(),x[postunsupported].max(),color=b.COLORS['mask'])
            visits=alltimes[(alltimes>=x.min())&(alltimes<=x.max())]
            ax.plot(visits,np.full(len(visits),1.035),'|',transform=ax.get_xaxis_transform(),color=b.COLORS['purple'],ms=3,alpha=.12,clip_on=False)
            ax.set_title(name,fontsize=8,fontstyle='italic',pad=10)
            ax.set_xlim(x.min(),x.max());ax.set_xlabel('Days after event (supported window)')
            if k==0:ax.set_ylabel(cohort+'\nCLR event effect')
            ax.text(.98,.06,f'{int(v.n_seeds.iloc[0])} seeds · detected {v.sample_detection.iloc[0]:.0%}',transform=ax.transAxes,ha='right',fontsize=6.4,color='#555963')
            b.tidy(ax)
    box=gs.get_position(fig)
    fig.legend(handles=[Line2D([],[],color=b.COLORS['ink'],lw=1.65,label='Pooled mean'),
        Patch(facecolor=b.COLORS['lavender'],alpha=.6,label='95% posterior interval'),
        Line2D([],[],color=b.COLORS['purple'],lw=.9,label='Seed mean'),
        Line2D([],[],color=b.COLORS['purple'],marker='|',ls='none',label='Observed visits')],
        loc='lower right',bbox_to_anchor=(box.x1,box.y1+.008),ncol=4,fontsize=6.8,columnspacing=1.1,handlelength=1.8)
