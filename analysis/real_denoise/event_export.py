"""Memory-bounded component export plus compact posterior draws for seed pooling."""
from pathlib import Path
import math
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import BSpline
from common import write_npz,write_json
from event_aware_denoise import event_basis,zero_responsibility


COMPONENTS=['common','individual','event','total','baseline','zero_event']
AGE_GRID=60
CURVE_GRID=41


def spline_rows(metadata,times):
    """Re-evaluate the fitted centred population spline basis at arbitrary times."""
    knots=np.asarray(metadata['spline_knots']);n=len(knots)-4
    x=(np.asarray(times,float)-metadata['time_origin'])/metadata['time_span']
    design=BSpline(knots,np.eye(n),3)(np.clip(x,0,1))
    return (design-np.asarray(metadata['spline_center']))[...,:-1]


def curve_grid(span):
    return np.linspace(-min(.2*span,90),min(.5*span,180),CURVE_GRID)


@torch.no_grad()
def export_components(estimator,s,taxa,events,names,out,keep_draws=False):
    m,c,meta=estimator.model_,estimator.config,estimator.metadata_
    ix=(s.tensor_i.to_numpy(),s.tensor_t.to_numpy());N,J=len(s),len(taxa);D=c.posterior_samples
    I,E=m.q_event.shape[0],m.q_event.shape[-2]
    # Disk-backed exact pointwise quantiles; never allocate all draws in RAM.
    disk=Path(out)/'component_draws.npy'
    draw=np.lib.format.open_memmap(disk,mode='w+',dtype='float32',shape=(D,N,J,len(COMPONENTS)))
    span=meta['time_span']
    grid=curve_grid(span)
    basis=lambda rel:torch.as_tensor(event_basis(rel/span,c.event_basis,c.event_decay,c.event_window,c.event_delay),dtype=torch.float64)
    q=basis(grid)
    age=np.linspace(s.time.min(),s.time.max(),AGE_GRID)
    age_spline=torch.as_tensor(spline_rows(meta,age),dtype=torch.float64)
    finite=events[np.isfinite(events)]
    # Typical-subject event at the median recorded date (per channel), for the population view.
    median_event=np.array([np.nanmedian(events[:,e]) if np.isfinite(events[:,e]).any() else np.nan for e in range(E)])
    q_age=basis(age[:,None]-median_event[None,:]) if finite.size else None
    curves=np.empty((D,J,E,len(grid)),np.float32);zero_curves=np.empty_like(curves)
    population=np.empty((D,AGE_GRID,J),np.float32);population_event=np.empty_like(population)
    gains=np.empty((D,I,J,E),np.float32) if c.individual_event=='gain' else None
    means={};cross=np.zeros((N,J,3,3));first=np.zeros((N,J,3));second=np.zeros((N,J,3))
    torch.manual_seed(c.seed+20000)
    for k in range(D):
        w=m.values(True);d=m.decode(w)
        dynamic=d['shared']+d['specific']
        base=d['Z']-d['trend']-d['group_effect']-dynamic-d['event_plus']
        values=[d['trend']+d['group_effect'],dynamic,d['event_plus'],d['Z'],base,d['event_zero']]
        for j,v in enumerate(values):
            a=v[ix].cpu().numpy()
            if not np.isfinite(a).all():raise FloatingPointError('Nonfinite posterior component '+COMPONENTS[j])
            draw[k,:,:,j]=a
        v=draw[k,:,:,:3].astype(float);first+=v/D;second+=v*v/D
        cross+=np.einsum('nji,njk->njik',v,v)/D
        curves[k]=torch.einsum('gq,jeq->jeg',q,d['theta']).cpu().numpy()
        zero_curves[k]=torch.einsum('gq,jeq->jeg',q,d['theta0']).cpu().numpy()
        alpha=-math.log(J)+2*w['alpha']
        pop=alpha[None]+age_spline@(.7*w['G']).T
        population[k]=pop.cpu().numpy()
        population_event[k]=(pop+torch.einsum('geq,jeq->gj',torch.nan_to_num(q_age),d['theta'])).cpu().numpy() if q_age is not None else population[k]
        if gains is not None:
            gains[k]=(w['d']*torch.exp(math.log(c.gain_scale)+.5*w['scale_d'])[None,:,None]).cpu().numpy()
        rho=zero_responsibility(m.y,d['log_mu'],d['size'],d['eta_logit'])
        rho=torch.where(m.mask,rho,torch.sigmoid(d['eta_logit']))
        positive=torch.exp(d['Z']);signal=(1-rho)*positive
        comp=signal/signal.sum(-1,keepdim=True).clamp_min(torch.finfo(signal.dtype).tiny)
        without=d['Z']-d['event_plus'];eta_without=d['eta_logit']-d['event_zero']
        full_expected=torch.softmax(d['Z']+torch.nn.functional.logsigmoid(-d['eta_logit']),-1)
        removed_expected=torch.softmax(without+torch.nn.functional.logsigmoid(-eta_without),-1)
        values=dict(Z_mean=d['Z'],X_mean=signal,composition_mean=comp,absence_probability_mean=rho,
            model_expected_composition=full_expected,removed_event_reconstruction=removed_expected,
            removed_event_positive_composition=torch.softmax(without,-1))
        for name,v in values.items():
            a=v[ix].cpu().numpy()
            if not np.isfinite(a).all():raise FloatingPointError('Nonfinite export '+name)
            if k==0:means[name]=np.zeros_like(a)
            means[name]+=a/D
        if k%20==0:draw.flush()
    draw.flush()
    z=means['Z_mean'];v=np.exp(z-z.max(-1,keepdims=True));means['abundance_composition']=v/v.sum(-1,keepdims=True)
    results={};tail=(1-c.interval)/2
    for j,name in enumerate(COMPONENTS+['removed_event_log_reconstruction']):
        mean=np.empty((N,J));low=mean.copy();high=mean.copy()
        for start in range(0,N,32):
            stop=min(N,start+32)
            a=np.asarray(draw[:,start:stop,:,j],float) if j<len(COMPONENTS) else (
                np.asarray(draw[:,start:stop,:,3],float)-np.asarray(draw[:,start:stop,:,2],float))
            mean[start:stop]=a.mean(0);low[start:stop],high[start:stop]=np.quantile(a,[tail,1-tail],axis=0)
        results[name+'_mean']=mean;results[name+'_lower']=low;results[name+'_upper']=high
    variance=np.maximum(second-first**2,0);cov=cross-first[:,:,:,None]*first[:,:,None,:]
    correlations=[]
    for a,b in [(0,1),(0,2),(1,2)]:
        denom=np.sqrt(variance[:,:,a]*variance[:,:,b]);r=np.divide(cov[:,:,a,b],denom,out=np.full((N,J),np.nan),where=denom>1e-12)
        for j,taxon in enumerate(taxa.taxon_id):
            good=r[:,j][np.isfinite(r[:,j])]
            correlations.append(dict(taxon_id=taxon,component_a=COMPONENTS[a],component_b=COMPONENTS[b],
                median_posterior_correlation=np.median(good) if len(good) else np.nan,
                fraction_correlation_below_minus_0_7=np.mean(good<-.7) if len(good) else np.nan))
    pd.DataFrame(correlations).to_csv(Path(out)/'component_posterior_correlation.tsv',sep='\t',index=False)
    support=np.zeros((len(names),len(grid)),int);eligible=np.zeros(len(names),int)
    radius=min(.1*span,30.)
    for i,g in s.groupby('tensor_i'):
        for e,tau in enumerate(events[int(i)]):
            if not np.isfinite(tau) or (g.time<tau).sum()<2:continue
            eligible[e]+=1;rel=g.time.to_numpy()-tau
            support[e]+=np.min(np.abs(grid[:,None]-rel[None,:]),axis=1)<=radius
    supported=(support>=3)&(support>=np.ceil(.5*np.maximum(eligible[:,None],1)))&(grid[None]>=0)
    mean=curves.mean(0);low,high=np.quantile(curves,[tail,1-tail],axis=0)
    write_npz(Path(out)/'event_curves.npz',relative_day=grid,event_names=np.array(names,dtype=str),
        taxon_id=taxa.taxon_id.to_numpy(dtype=str),mean=mean,lower=low,upper=high,
        support_subjects=support,supported=supported)
    subjects=s.sort_values('tensor_i').drop_duplicates('tensor_i').subject_id.to_numpy(dtype=str)
    if len(subjects)!=I:raise ValueError('Subject order does not match the fitted tensor')
    extra=dict(gain=gains) if gains is not None else {}
    write_npz(Path(out)/'posterior_draws.npz',event_curve=curves,zero_event_curve=zero_curves,
        population_log=population,population_with_event_log=population_event,age_day=age,
        relative_day=grid,curve_supported=supported,median_event_day=median_event,
        event_names=np.array(names,dtype=str),subject_id=subjects,
        taxon_id=taxa.taxon_id.to_numpy(dtype=str),**extra)
    results.update(sample_id=s.sample_id.to_numpy(dtype=str),taxon_id=taxa.taxon_id.to_numpy(dtype=str))
    write_npz(Path(out)/'components.npz',**results)
    del draw
    if not keep_draws:disk.unlink()
    write_json(Path(out)/'component_contract.json',dict(
        identity='Z = baseline + common + individual + event (log positive-abundance scale)',
        common='population spline over time; no group term',
        individual='shared latent factors + taxon-specific individual residual; excludes baseline intercept',
        baseline='population and subject intercepts',zero_event='event contribution to absence logit',
        population_log='alpha + common spline on age_day grid (typical subject, no event)',
        population_with_event_log='population_log + population event curve at the median recorded event day',
        gain='individual multiplicative event gain: subject response = (1+gain) x population event curve',
        interval='pointwise Gaussian VI posterior intervals; approximate',
        reconstruction='模型条件下移除事件分量的重建；不是因果反事实',
        event_curve_support='At least 3 subjects and half of >=2-pre-visit subjects sampled within local radius; post-event only',
        support_radius_days=radius))
    return means,results
