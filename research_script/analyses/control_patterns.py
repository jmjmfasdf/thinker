#!/usr/bin/env python3
"""Two-clock planning analysis: computation, valuation, human switching and candidate motifs.

Read-only native replay analysis. No training, environment rollout or score evaluation.
ACE is output-relevant geometry, not a goal label. Actor readout reconstruction is
conditional on recorded features, not a forced-budget intervention. All inference
uses session-held-out fits; Thinker itself was trained on these human sessions.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time
import warnings
sys.dont_write_bytecode = True
for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
import numpy as np
import pandas as pd
from scipy import linalg
from scipy.special import expit, softmax, logsumexp
RESEARCH = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = RESEARCH.parent / 'logs/visualization/campaigns/visualization_hxln_completed_10m_20261124'
VERSION = 'control-patterns-v3.1'
GAME_IDS = {'enduro': 0, 'pong': 1, 'spaceinvaders': 2}
STAGES = ('audit', 'extract', 'representation', 'temporal', 'motifs', 'validate', 'render')
REFERENCE = 'https://arxiv.org/html/2602.22600v3'

from common.native_replay import *

DIRECT = ['V_initial', 'first_dV', 'A', 'D', 'U', 'coherence', 'voc_initial', 'voc_first', 'voc_final', 'voc_mean']
FIRST = ['V_initial', 'first_dV', 'voc_initial', 'voc_first']

def sample_runs(d, indices, count, seed):
    """Equal-run stratified sample without replacement; reproducible within folds."""
    rng = np.random.default_rng(seed)
    groups = [np.asarray(g.index, int) for _, g in d.loc[indices].groupby("run", sort=True)]
    groups = [rng.permutation(g) for g in groups]
    selected = []
    while groups and len(selected) < count:
        next_groups = []
        for group in groups:
            if len(selected) == count:
                break
            selected.append(group[0])
            if len(group) > 1:
                next_groups.append(group[1:])
        groups = next_groups
    return np.asarray(selected, int)


def balanced_weights(d, indices):
    runs = d.loc[indices, "run"]
    return (1 / runs.map(runs.value_counts()) / runs.nunique()).to_numpy(float)


def activation_factor(h, rows, weights, chunk=256):
    """Exact activation range, never pre-truncate low-variance directions."""
    dim = h.shape[1]
    mean = np.zeros(dim)
    for ids in chunks(np.arange(len(rows)), chunk):
        mean += weights[ids] @ np.asarray(h[rows[ids]], dtype=np.float64)
    if len(rows) <= dim:
        x = (np.asarray(h[rows], dtype=np.float64) - mean) * np.sqrt(weights[:, None])
        # L L^T = covariance directly. Avoid a needless wide-matrix LAPACK
        # SVD (gesdd crashes on this host for 64 x 4608); ACE's interaction SVD
        # below removes zero-sensitivity/rank-deficient directions itself.
        factor = x.T.copy()
    else:
        cov = np.zeros((dim, dim), np.float64)
        for ids in chunks(np.arange(len(rows)), chunk):
            x = (np.asarray(h[rows[ids]], dtype=np.float64) - mean) * np.sqrt(weights[ids, None])
            cov += x.T @ x
        w, u = linalg.eigh(cov, check_finite=False)
        keep = w > max(w[-1] * 1e-12, 1e-14)
        factor = u[:, keep] * np.sqrt(w[keep])
    return mean, factor


def ace_basis(factor, jacobian, energy=.99, max_rank=128):
    if factor.shape[1] == 0 or not np.any(jacobian):
        return np.empty((factor.shape[0], 0)), np.array([]), 0, 0.
    interaction = factor.T @ jacobian.T / np.sqrt(max(len(jacobian), 1))
    # This is activation-rank by Jacobian-row-count, not N by NK.
    u, s, _ = linalg.svd(interaction, full_matrices=False, check_finite=False, lapack_driver='gesvd')
    weights = s ** 2
    if not len(weights) or weights.sum() <= 1e-24:
        return np.empty((factor.shape[0], 0)), weights, 0, 0.
    r_energy = int(np.searchsorted(np.cumsum(weights) / weights.sum(), energy) + 1)
    rank = min(max_rank, factor.shape[1], int((s > s[0] * 1e-10).sum()))
    q, _ = linalg.qr(factor @ u[:, :rank], mode="economic")
    return q, weights, r_energy, float(weights[:min(r_energy, rank)].sum() / weights.sum())


def intervene(h, q, mean, mode, centered=True):
    origin = mean if centered else np.zeros_like(mean)
    delta = np.asarray(h) - origin
    projected = (delta @ q) @ q.T
    return origin + (projected if mode == "keep" else delta - projected)


def subspace_overlap(a, b):
    return float(np.linalg.norm(a.T @ b, "fro") ** 2 / min(a.shape[1], b.shape[1])) if a.shape[1] and b.shape[1] else np.nan


def fit_key(sessions):
    return 'train_'+'_'.join(map(str,sorted(sessions)))


def fit_sets(sessions):
    return [tuple(x) for k in (2,3) for x in itertools.combinations(sorted(sessions),k)]


def finite_scale(y):
    s=float(np.sqrt(np.var(y,axis=0).sum()))
    return s if s>1e-8 else 0.


def summarize_coordinates(z,d):
    """Fixed-stage and whole-computation summaries. No padding of short searches."""
    initial=z[d.start.to_numpy(int)];first=z[np.minimum(d.start.to_numpy(int)+1,d.end.to_numpy(int)-1)]
    final=z[d.end.to_numpy(int)-1]
    means=np.vstack([z[r.start:r.end].mean(0) for r in d.itertuples()])
    return np.column_stack([initial,first-initial,final-initial,means-initial]).astype('float32')


def local_head_reproduction_values(values, net, config):
    """Match saved-node reward semantics without changing raw ACE targets."""
    result = np.array(values, copy=True)
    if net == 'sr' and config['reward_clip'] > 0:
        # ModelNet._prepare_out clips SRN rewards before cenv stores cur_r.
        clip = config['reward_clip']
        result[:, 0] = np.clip(result[:, 0], -clip, clip)
    return result


def representation_game(args,game,out):
    d=read_table(out/'decisions.csv.gz');e=read_table(out/'computations.csv.gz');a=read_json(out/'audit.json')
    actor=np.load(out/'actor_features.npy',mmap_mode='r');ah=ActorHeads(a);nh=FrozenHeads(args,out)
    fields={'sr':'im_vectors','vp':'im_vp_vectors'};sessions=sorted(d.session.unique());summary=[]
    # Authenticate the full projected endpoint support, not only fit samples.
    for path,g in e.groupby('trace',sort=False):
        verify_chunks(Trace(path),list(fields.values()),g.model_event.to_numpy(int))
    for train_sessions in fit_sets(sessions):
        folder=out/'representation'/fit_key(train_sessions);folder.mkdir(parents=True,exist_ok=True)
        if resume_unit(folder,args):
            summary.extend(read_json(folder/'summary.json'));continue
        train=d.index[d.session.isin(train_sessions)].to_numpy()
        # Equal-run, equal-decision fitting; one uniformly chosen k per decision.
        rng=np.random.default_rng(args.seed);picked=sample_runs(d,train,args.basis_samples,args.seed)
        endpoint=np.array([rng.integers(d.loc[i,'start'],d.loc[i,'end']) for i in picked],int)
        weights=balanced_weights(d,picked);fold_summary=[];bank={};fixed={};coordinates={}
        held=d.index[~d.session.isin(train_sessions)].to_numpy()
        val_dec=sample_runs(d,held,min(args.jacobian_samples,len(held)),args.seed+9)
        val_ep=np.minimum(d.loc[val_dec,'start'].to_numpy(int)+1,d.loc[val_dec,'end'].to_numpy(int)-1)
        for net in ('sr','vp','actor'):
            log(f'{game} {fit_key(train_sessions)}: {net} functional spaces')
            x=load_latents(e,endpoint,fields[net],True) if net!='actor' else np.asarray(actor[endpoint,ah.learned_start:])
            mean,factor=activation_factor(x,np.arange(len(x)),weights,args.chunk_size)
            if net!='actor':
                y=nh.predict(net,x,contrast=True)
                n=min(args.jacobian_samples,len(x));rawgrad=nh.jacobian(net,x[:n],np.ones(y.shape[1])).reshape(n,y.shape[1],-1)
                targets={'sr_reward_terminal':[0,1]} if net=='sr' else {'vp_value':[0],'vp_policy':list(range(1,y.shape[1]))}
                xv=load_latents(e,val_ep,fields[net],True);yv=nh.predict(net,xv,contrast=True)
                # Check directly recorded local heads only where the evaluated node is retained.
                errors=[]
                for path,g in e.iloc[val_ep].assign(_order=np.arange(len(val_ep))).groupby('trace'):
                    t=Trace(path);ev=g.model_event.to_numpy(int);idx=g._order.to_numpy(int)
                    ok=(g.k.to_numpy()>0)&(~t.scalar('actor_reset')[ev].astype(bool))&(t.tree('cur_d',ev).ravel()==0)
                    if net=='sr':
                        ok &= t.valid('model_done_logit')[ev] & (t.scalar('model_done_target_depth')[ev]==t.scalar('current_node_depth_post')[ev])
                        ref=np.column_stack([t.tree('cur_r',ev),t.scalar('model_done_logit')[ev]])
                    else:
                        logits=t.tree('cur_policy',ev);ref=np.column_stack([t.tree('cur_v',ev),logits-logits.mean(1,keepdims=True)])
                    if ok.any():
                        reproduced=local_head_reproduction_values(yv[idx[ok]],net,a['config'])
                        error=np.max(np.abs(reproduced-ref[ok]),axis=0);tol=np.r_[.002,np.full(len(error)-1,.02)]
                        if np.any(error>tol):raise ValueError('Local head reproduction failed: '+str(error))
                        errors.append(error.tolist())
            else:
                full=np.asarray(actor[endpoint]);v,p,vo,ve=ah.outputs(full)
                logits=ah.linear(full-ah.phase[0]+ah.phase[1],'policy');logits-=logits.mean(1,keepdims=True)
                y=np.column_stack([v,logits,ve]);targets={'actor_value':[0],'actor_policy':list(range(1,1+ah.nact)),
                                                        'actor_voc':[ah.nact+1],'actor_joint':list(range(ah.nact+2))}
                w=np.vstack([ah.weights('value'),ah.weights('policy'),ah.weights('voc')])[:,ah.learned_start:]
                rawgrad=w[None];xv=np.asarray(actor[val_ep,ah.learned_start:]);yv=None;errors=[]
            for label,cols in targets.items():
                scale=np.ones(len(cols))
                if label=='sr_reward_terminal':
                    scale=np.array([finite_scale(y[:,[c]]) for c in cols])*np.sqrt(2)
                elif label=='actor_joint':
                    for block in ([0],list(range(1,1+ah.nact)),[ah.nact+1]):scale[block]=finite_scale(y[:,block])*np.sqrt(3)
                else:scale[:]=finite_scale(y[:,cols])
                grad_tensor=np.divide(rawgrad[:,cols],scale[None,:,None],out=np.zeros_like(rawgrad[:,cols]),where=scale[None,:,None]>0)
                grad=grad_tensor.reshape(-1,x.shape[1])
                ordered,spectrum,needed,retained=ace_basis(factor,grad,args.energy,args.max_rank)
                rank=min(needed,ordered.shape[1]);basis=ordered[:,:rank]
                # Sign convention fixed on training data only; never on switch labels.
                for col in range(rank):
                    if basis[np.argmax(abs(basis[:,col])),col]<0:basis[:,col]*=-1
                trainz=(x-mean)@basis
                sd=np.sqrt(np.average((trainz-np.average(trainz,axis=0,weights=weights))**2,axis=0,weights=weights));sd=np.maximum(sd,1e-8)
                np.savez_compressed(folder/(label+'_basis.npz'),mean=mean,basis=basis,sd=sd,spectrum=spectrum,
                                    train_decisions=picked,train_endpoints=endpoint,output_scale=scale)
                validation={}
                if len(xv) and rank:
                    keep=intervene(xv,basis,mean,'keep');remove=intervene(xv,basis,mean,'remove')
                    if net!='actor':
                        pred=nh.predict(net,xv,contrast=True)[:,cols]
                        pk=nh.predict(net,keep,contrast=True)[:,cols];pr=nh.predict(net,remove,contrast=True)[:,cols]
                    else:
                        pred=xv@w[cols].T;pk=keep@w[cols].T;pr=remove@w[cols].T
                    validation=dict(keep_rmse=float(np.sqrt(np.mean((pk-pred)**2))),remove_rmse=float(np.sqrt(np.mean((pr-pred)**2))),
                                    output_sd=float(np.std(pred)),scope='local head only; other actor blocks fixed')
                # Stability against independent equal-run decision sample, same covariance convention.
                stability=np.nan
                if len(grad_tensor)>2:
                    b2,_,r2,_=ace_basis(factor,grad_tensor[::2].reshape(-1,x.shape[1]),args.energy,args.max_rank)
                    stability=subspace_overlap(basis,b2[:,:min(r2,b2.shape[1])])
                fold_summary.append(dict(fit=fit_key(train_sessions),target=label,rank=rank,required_rank=needed,
                    retained_energy=retained,first_energy=float(spectrum[0]/spectrum.sum()) if spectrum.sum() else 0.,
                    sensitivity_subsample_overlap=stability,training_decisions=len(picked),equal_decision_stage_sampling=True,
                    local_head_errors=errors,**validation))
                coordinates[label]=np.lib.format.open_memmap(folder/(label+'_events.npy'),mode='w+',dtype='float32',shape=(len(e),rank))
                bank[label]=(net,mean,basis,sd)
            del x,factor
        # Stream all computation endpoints, preserving RESET/branch metadata separately.
        for net in ('sr','vp','actor'):
            labels=[k for k,v in bank.items() if v[0]==net]
            for ids in chunks(np.arange(len(e)),args.chunk_size):
                x=load_latents(e,ids,fields[net]) if net!='actor' else np.asarray(actor[ids,ah.learned_start:])
                for label in labels:
                    _,mean,basis,sd=bank[label];coordinates[label][ids]=((x-mean)@basis/sd).astype('float32')
        for label,z in coordinates.items():
            z.flush();np.save(folder/(label+'_decisions.npy'),summarize_coordinates(z,d))
            i=d.start.to_numpy(int);j=np.minimum(i+1,d.end.to_numpy(int)-1)
            np.save(folder/(label+'_first.npy'),np.column_stack([z[i],z[j]-z[i]]))
        # Actual GRU slice is available in the fused feature vector; save without claiming a full recurrent intervention.
        gru=np.asarray(actor[:,ah.gru_slice]);np.save(folder/'gru_decisions.npy',summarize_coordinates(gru,d))
        write_json(folder/'summary.json',fold_summary);complete_unit(folder);summary.extend(fold_summary)
    save_table(out/'representation_summary.csv',summary)


def window_features(x,d,w):
    """Only observations at/before endpoint; no centered smoothing or future padding."""
    x=np.asarray(x,float);answer=np.full((len(w),6*x.shape[1]),np.nan,np.float32)
    for i,r in enumerate(w.itertuples()):
        ids=np.arange(r.left,r.row+1);v=x[ids];t=d.time.to_numpy()[ids];dt=t-t.mean()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            mean=np.nanmean(v,axis=0);sd=np.nanstd(v,axis=0)
            valid=np.isfinite(v);n=valid.sum(0)
            local_t=np.divide((valid*t[:,None]).sum(0),n,out=np.zeros(x.shape[1]),where=n>0)
            centered=t[:,None]-local_t;den=(valid*centered**2).sum(0)
            slope=np.divide(np.nansum(centered*(v-mean),axis=0),den,out=np.zeros(x.shape[1]),where=den>1e-12)
            path=np.nansum(np.abs(np.diff(v,axis=0)),axis=0)
        answer[i]=np.r_[v[-1],mean,sd,slope,v[-1]-v[0],path]
    return answer


def temporal_game(args,game,out):
    d=read_table(out/'decisions.csv.gz');policy=np.load(out/'actual_policy.npy');timelines={p:ActionTimeline(p) for p in d.source.unique()}
    rows=[];action=[];raw_events=[]
    for segment,g in d.groupby('segment',sort=False):
        ids=g.index.to_numpy();times=g.time.to_numpy();tl=timelines[g.source.iloc[0]]
        changes=np.flatnonzero(tl.a[1:]!=tl.a[:-1])+1
        chosen=changes[(tl.t[changes]>=times[0])&(tl.t[changes]<=times[-1])]
        for ix in chosen:
            raw_events.append(dict(segment=segment,session=g.session.iloc[0],run=g.run.iloc[0],time=tl.t[ix],
                source_frame=ix,action_from=tl.a[ix-1],action_to=tl.a[ix]))
        for width in args.windows_seconds:
            for pos,idx in enumerate(ids):
                start=times[pos]-width
                if start<times[0] or not tl.valid(start,times[pos]):continue
                left=np.searchsorted(times,start,side='right')-1
                if left<0 or start-times[left]>.12 or pos-left<2:continue
                r=dict(row=int(idx),left=int(ids[left]),width=width,segment=segment,session=g.session.iloc[0],
                       run=g.run.iloc[0],episode=g.episode.iloc[0],time=times[pos],human_action=int(d.loc[idx,'human_action']))
                for horizon in args.horizons_seconds:
                    end=times[pos]+horizon;key=f'{horizon:g}'
                    r['switch_'+key]=float(tl.counts(times[pos],end).sum()>0) if end<=times[-1] and tl.valid(times[pos],end) else np.nan
                    dest=np.searchsorted(times,end,side='right')-1
                    r['policy_'+key]=(float(js_distance(policy[idx],policy[int(ids[dest])]))
                        if end<=times[-1] and end-times[dest]<=.12 and np.isfinite(r['switch_'+key]) else np.nan)
                rows.append(r);action.append(tl.profile(start,times[pos]))
    w=pd.DataFrame(rows);w.to_csv(out/'windows.csv.gz',index=False);pd.DataFrame(raw_events).to_csv(out/'switches.csv.gz',index=False)
    np.save(out/'action_windows.npy',np.asarray(action,np.float32))
    for name,features in [('direct',d[DIRECT].to_numpy()),('first_direct',d[FIRST].to_numpy()),
                          ('budget',np.log1p(d[['K']].to_numpy())),('tree',np.load(out/'tree_summary.npy'))]:
        np.save(out/(name+'_windows.npy'),window_features(features,d,w))
    # Fold-specific representations; pair fits are used for completely nested K selection.
    for folder in sorted((out/'representation').glob('train_*')):
        pieces=[];first=[]
        for label in ('sr_reward_terminal','vp_value','vp_policy','actor_joint'):
            pieces.append(window_features(np.load(folder/(label+'_decisions.npy')),d,w))
            first.append(window_features(np.load(folder/(label+'_first.npy')),d,w))
        np.save(folder/'ace_windows.npy',np.column_stack(pieces));np.save(folder/'first_ace_windows.npy',np.column_stack(first))
    write_json(out/'temporal.json',dict(windows=len(w),raw_switches=len(raw_events),widths=args.windows_seconds,
        horizons=args.horizons_seconds,primary_width=.5,primary_horizon=.25,max_age=.12,all_switches=True,
        first_only_sensitivity='fixed k=1 features; K=0 carries zero change and explicit recorded K',
        overlapping_windows='Only nonoverlapping training windows; block uncertainty on held-out rows'))


class BlockProjection:
    """Train-only imputation/scaling, equal block energy, bounded PCA."""
    def fit(self, blocks, ids, seed=701):
        from sklearn.decomposition import PCA
        self.stats=[];pieces=[]
        for block in blocks:
            b=np.asarray(block[ids],float)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning);mu=np.nanmean(b,axis=0)
            mu=np.nan_to_num(mu);b=np.where(np.isfinite(b),b,mu)
            sd=b.std(0);keep=sd>1e-8
            self.stats.append((mu,np.maximum(sd,1e-8),keep))
            if keep.any():pieces.append((b[:,keep]-mu[keep])/sd[keep]/np.sqrt(keep.sum()))
        self.pca=None;self.rank=0;self.retained=0.
        if not pieces:return self
        x=np.column_stack(pieces);rank=min(16,x.shape[1],len(ids)-1)
        if rank<=0:return self
        self.pca=PCA(n_components=rank,svd_solver='randomized',random_state=seed).fit(x)
        self.rank=min(rank,int(np.searchsorted(np.cumsum(self.pca.explained_variance_ratio_),.95)+1))
        self.retained=float(self.pca.explained_variance_ratio_[:self.rank].sum())
        return self

    def transform(self,blocks):
        pieces=[]
        for block,(mu,sd,keep) in zip(blocks,self.stats):
            b=np.where(np.isfinite(block),block,mu)
            if keep.any():pieces.append((b[:,keep]-mu[keep])/sd[keep]/np.sqrt(keep.sum()))
        if self.pca is None:return np.zeros((len(blocks[0]),1))
        return self.pca.transform(np.column_stack(pieces))[:,:self.rank]

    def save(self,path):
        arrays={}
        for i,(mu,sd,keep) in enumerate(self.stats):arrays.update({f'mean_{i}':mu,f'sd_{i}':sd,f'keep_{i}':keep})
        if self.pca is not None:arrays.update(pca_mean=self.pca.mean_,pca_components=self.pca.components_[:self.rank])
        np.savez_compressed(path,**arrays)


def nonoverlap(w,ids,width):
    result=[]
    for _,g in w.loc[ids].groupby('segment',sort=False):
        last=-np.inf
        for r in g.sort_values('time').itertuples():
            if r.time-last>=width-1e-9:result.append(r.Index);last=r.time
    return np.asarray(result,int)


def feature_blocks(out,fit,view,first=False):
    folder=out/'representation'/fit_key(fit)
    actions=np.load(out/'action_windows.npy',mmap_mode='r')
    if view=='action_only':return [actions]
    direct=np.load(out/('first_direct_windows.npy' if first else 'direct_windows.npy'),mmap_mode='r')
    if view=='outputs_only':return [direct]
    blocks=[direct,np.load(folder/('first_ace_windows.npy' if first else 'ace_windows.npy'),mmap_mode='r')]
    if not first:blocks.append(np.load(out/'tree_windows.npy',mmap_mode='r'))
    if view.startswith('joint'):blocks.append(actions)
    if view.endswith('_budget'):blocks.append(np.load(out/'budget_windows.npy',mmap_mode='r'))
    return blocks


def mixture_fit(x,k,args):
    from sklearn.mixture import GaussianMixture
    if len(x)<max(k*3,3):raise ValueError('Too few windows for mixture')
    with warnings.catch_warnings():
        from sklearn.exceptions import ConvergenceWarning
        warnings.simplefilter('error',ConvergenceWarning)
        model=GaussianMixture(n_components=k,covariance_type='diag',reg_covar=1e-5,
            n_init=1 if args.smoke else 5,max_iter=300,random_state=args.seed).fit(x)
    return model


def one_se_choice(tab, folds):
    eligible=tab[tab.status.eq('ok')].groupby('k').score.agg(['mean','std','count'])
    eligible=eligible[eligible['count'].eq(folds)]
    if eligible.empty:raise ValueError('No candidate converged in every inner fold')
    best=eligible['mean'].idxmax();se=eligible.loc[best,'std']/np.sqrt(eligible.loc[best,'count'])
    return int(eligible.index[eligible['mean']>=eligible.loc[best,'mean']-se].min())


def select_mixture(args,out,w,train_sessions,view,width,first=False):
    """Each inner fold re-fits ACE, scaling and PCA without its validation session."""
    scores=[]
    for held in train_sessions:
        inner=tuple(s for s in train_sessions if s!=held)
        train=nonoverlap(w,w.index[w.session.isin(inner)&w.width.eq(width)].to_numpy(),width)
        valid=nonoverlap(w,w.index[w.session.eq(held)&w.width.eq(width)].to_numpy(),width)
        if len(train)<4 or len(valid)<2:continue
        blocks=feature_blocks(out,inner,view,first);proj=BlockProjection().fit(blocks,train,args.seed)
        xt=proj.transform([b[train] for b in blocks]);xv=proj.transform([b[valid] for b in blocks])
        for k in range(1,(2 if args.smoke else 8)+1):
            if len(train)<k*3:continue
            try:
                gm=mixture_fit(xt,k,args);score=float(gm.score(xv));status='ok'
            except (ValueError,RuntimeError,Warning) as exc:score=np.nan;status=type(exc).__name__
            scores.append(dict(held=held,k=k,score=score,status=status))
    tab=pd.DataFrame(scores)
    if tab.empty:raise ValueError('No inner validation support')
    return one_se_choice(tab,len(train_sessions)),tab


def motifs_game(args,game,out):
    w=read_table(out/'windows.csv.gz');rows=[];sessions=sorted(w.session.unique())
    views=['evaluation','evaluation_budget','joint','joint_budget','action_only','outputs_only']
    for held in sessions:
        fit=tuple(s for s in sessions if s!=held)
        for width in args.windows_seconds:
            train=nonoverlap(w,w.index[w.session.isin(fit)&w.width.eq(width)].to_numpy(),width)
            test=w.index[w.session.eq(held)&w.width.eq(width)].to_numpy()
            for view in views:
                folder=out/'motifs'/f'held_{held}'/f'w{width:g}'/view;folder.mkdir(parents=True,exist_ok=True)
                if resume_unit(folder,args):continue
                k,scores=select_mixture(args,out,w,fit,view,width)
                blocks=feature_blocks(out,fit,view);prep=BlockProjection().fit(blocks,train,args.seed)
                xt=prep.transform([b[train] for b in blocks]);xv=prep.transform([b[test] for b in blocks])
                gm=mixture_fit(xt,k,args);prob=gm.predict_proba(xv);state=prob.argmax(1)
                frame=w.loc[test,['row','width','segment','session','run','episode','time','human_action']].copy()
                frame['window']=test;frame['state']=state;frame['confidence']=prob.max(1)
                for c in range(k):frame[f'p{c}']=prob[:,c]
                frame.to_csv(folder/'assignments.csv.gz',index=False);np.save(folder/'test_embedding.npy',xv);prep.save(folder/'projection.npz')
                np.savez_compressed(folder/'mixture.npz',weights=gm.weights_,means=gm.means_,covariances=gm.covariances_)
                save_table(folder/'selection.csv',scores)
                write_json(folder/'fit.json',dict(game=game,held=held,train_sessions=fit,width=width,view=view,k=k,
                    train_windows=len(train),test_windows=len(test),rank=prep.rank,pca_retained=prep.retained,
                    heldout_density=float(gm.score(xv)),cluster_labels='fold-specific; never semantically aligned by outcome',
                    smoothing=False,selection='nested session CV, one standard error, smallest K including 1'))
                # Fixed first-computation sensitivity, same selection protocol and no post-first content.
                if view in ('evaluation','joint') and width==.5:
                    k1,s1=select_mixture(args,out,w,fit,view,width,True);b1=feature_blocks(out,fit,view,True)
                    pp=BlockProjection().fit(b1,train,args.seed)
                    gg=mixture_fit(pp.transform([b[train] for b in b1]),k1,args)
                    prob1=gg.predict_proba(pp.transform([b[test] for b in b1]));sensitivity=frame[['window','row','time']].copy()
                    sensitivity['state']=prob1.argmax(1)
                    for c in range(k1):sensitivity[f'p{c}']=prob1[:,c]
                    sensitivity.to_csv(folder/'first_only_assignments.csv.gz',index=False)
                    save_table(folder/'first_only_selection.csv',s1);pp.save(folder/'first_only_projection.npz')
                    np.savez_compressed(folder/'first_only_mixture.npz',weights=gg.weights_,means=gg.means_,covariances=gg.covariances_)
                complete_unit(folder);log(f'{game} held {held}, {width:g}s {view}: K={k}')
    for folder in sorted((out/'motifs').glob('held_*/w*/*')):
        rows.append(read_json(folder/'fit.json'))
    save_table(out/'motif_summary.csv',rows)


def block_interval(frame,values,reps,seed,seconds=10):
    """Paired mean within run, then equal-run average; resample blocks within run."""
    f=frame[['run','time']].copy();f['value']=values;f=f[np.isfinite(f.value)]
    if f.empty:return dict(mean=np.nan,low=np.nan,high=np.nan,runs=0,blocks=0)
    f['block']=np.floor(f.time/seconds).astype(int)
    groups=[]
    for _,g in f.groupby('run'):
        tab=g.groupby('block').value.agg(['sum','count']);groups.append(tab.to_numpy())
    rng=np.random.default_rng(seed);res=[]
    for _ in range(reps):
        means=[]
        for b in groups:
            v=b[rng.integers(0,len(b),len(b))];means.append(v[:,0].sum()/v[:,1].sum())
        res.append(np.mean(means))
    return dict(mean=float(f.groupby('run').value.mean().mean()),low=float(np.quantile(res,.025)),
                high=float(np.quantile(res,.975)),runs=len(groups),blocks=sum(map(len,groups)))


def readout(blocks,y,train,test,seed,classification=True):
    from sklearn.linear_model import LogisticRegression,Ridge
    from sklearn.preprocessing import StandardScaler
    # Preserve the baseline columns when adding a feature family: a joint PCA
    # could displace them and confound the incremental comparison.
    selected=np.r_[train,test];local_train=np.arange(len(train));local_test=np.arange(len(train),len(selected));pieces=[]
    for block in blocks:
        prep=BlockProjection().fit([block],train,seed);pieces.append(prep.transform([block[selected]]))
    x=np.column_stack(pieces);scaler=StandardScaler().fit(x[local_train]);x=scaler.transform(x)
    if classification:
        if len(np.unique(y[train]))<2:return np.full(len(test),(y[train].sum()+.5)/(len(train)+1))
        model=LogisticRegression(C=1.,max_iter=1000,solver='liblinear',random_state=seed).fit(x[local_train],y[train])
        return model.predict_proba(x[local_test])[:,1]
    model=Ridge(alpha=10.,solver='lsqr').fit(x[local_train],y[train]);return model.predict(x[local_test])


def loss(y,p,classification=True):
    if not classification:return (y-p)**2
    p=np.clip(p,1e-7,1-1e-7);return -(y*np.log(p)+(1-y)*np.log1p(-p))


def raw_pair_changes(frame,d,timelines):
    """A same-button endpoint pair may contain an unobserved switch-return."""
    good=np.r_[False,(np.diff(frame.time)<=.12)&(frame.segment.to_numpy()[1:]==frame.segment.to_numpy()[:-1])]
    changed=np.zeros(len(frame),bool)
    sources=d.iloc[frame.row.to_numpy(int)].source.to_numpy()
    for source in np.unique(sources):
        ii=np.flatnonzero(good&(sources==source))
        if len(ii):
            if source not in timelines:timelines[source]=ActionTimeline(source)
            tl=timelines[source]
            changed[ii]=tl.counts(frame.time.to_numpy()[ii-1],frame.time.to_numpy()[ii]).sum(axis=1)>0
    return good,changed


def validate_game(args,game,out):
    d=read_table(out/'decisions.csv.gz');w=read_table(out/'windows.csv.gz');p=np.load(out/'policies.npy')
    nact=p.shape[1];di=w.row.to_numpy(int);sessions=sorted(w.session.unique())
    root=np.load(out/'screen.npy');actual=np.load(out/'actual_policy.npy')
    history=np.column_stack([np.eye(nact)[d.human_action],np.log1p(d.age),d.raw_switch,d.carry,
                              np.eye(nact+1)[d.first_action+1]])
    # The joint candidate receives these exact action-window summaries. Put
    # them in the baseline too, so extra action history is not called an
    # imagination contribution.
    base=[root[di],history[di],actual[di],np.load(out/'action_windows.npy',mmap_mode='r')]
    # Independent visual/root control (no ACE target fitting).
    e=read_table(out/'computations.csv.gz');roots=d.start.to_numpy(int)
    rootlat=[]
    for net,field in [('sr','im_vectors'),('vp','im_vp_vectors')]:
        dest=out/(net+'_root_controls.npy')
        # Stage-level resume authenticates completed output. An unmarked
        # partially written memmap is never accepted merely because it exists.
        arr=np.lib.format.open_memmap(dest,mode='w+',dtype='float32',shape=(len(d),512))
        for ids in chunks(np.arange(len(d)),args.chunk_size):arr[ids]=pool_latents(load_latents(e,roots[ids],field,True))
        arr.flush()
        rootlat.append(np.load(dest,mmap_mode='r')[di])
    base.extend(rootlat)
    predrows=[];comparison=[];continuity=[];profiles=[];examples=[];budget_predictions=[];timelines={};motif_preds={}
    for held in sessions:
        fit=tuple(s for s in sessions if s!=held)
        content=feature_blocks(out,fit,'evaluation');budget=np.load(out/'budget_windows.npy')
        # Allocation prediction uses only fixed early information, never the
        # realized length, whole-search summaries or final actor policy.
        root_base=[root,np.column_stack([np.eye(nact+1)[d.previous_action.to_numpy(int)+1],np.log1p(d.age_prior),d.carry])]
        root_base += [np.load(out/(net+'_root_controls.npy'),mmap_mode='r') for net in ('sr','vp')]
        train_d=d.index[d.session.isin(fit)].to_numpy();test_d=d.index[d.session.eq(held)].to_numpy()
        early=[d[['V_initial','first_dV','voc_initial','voc_first']].to_numpy()]
        initial=[d[['V_initial','voc_initial']].to_numpy()]
        fdir=out/'representation'/fit_key(fit)
        for label in ('sr_reward_terminal','vp_value','vp_policy','actor_joint'):
            x=np.load(fdir/(label+'_first.npy'));early.append(x);initial.append(x[:,:x.shape[1]//2])
        target=np.log1p(np.maximum(d.K.to_numpy()-1,0))
        bp=readout(root_base,target,train_d,test_d,args.seed,False)
        for name,features in [('initial',initial),('first_computation',early)]:
            pred=readout(root_base+features,target,train_d,test_d,args.seed,False)
            gain=loss(target[test_d],bp,False)-loss(target[test_d],pred,False)
            comparison.append(dict(held=held,width=0,horizon=0,target='additional_budget',model=name,parent='scene_history',
                block_seconds=10,n=len(test_d),**block_interval(d.loc[test_d],gain,args.bootstrap,args.seed)))
            for i,idx in enumerate(test_d):budget_predictions.append(dict(row=int(idx),held=held,model=name,
                target=target[idx],prediction=pred[i],baseline=bp[i]))
        for width in args.windows_seconds:
            ids=w.index[w.width.eq(width)].to_numpy()
            for horizon in args.horizons_seconds:
                for target,classification in [('switch',True),('policy',False)]:
                    y=w[f'{target}_{horizon:g}'].to_numpy();valid=ids[np.isfinite(y[ids])]
                    train=nonoverlap(w,valid[w.loc[valid,'session'].isin(fit)],width)
                    test=valid[w.loc[valid,'session'].eq(held)]
                    if len(train)<8 or not len(test):continue
                    blocks_by_name={'baseline':base,'budget':base+[budget],
                        'value':base+[budget,content[0]],'ace':base+[budget,*content],
                        'first':base+feature_blocks(out,fit,'evaluation',True)}
                    predictions={name:readout(b,y,train,test,args.seed,classification) for name,b in blocks_by_name.items()}
                    for name in ('budget','value','ace','first'):
                        parent={'budget':'baseline','value':'budget','ace':'value','first':'baseline'}[name]
                        gain=loss(y[test],predictions[parent],classification)-loss(y[test],predictions[name],classification)
                        for sec in (5,10,20):
                            ci=block_interval(w.loc[test],gain,args.bootstrap,args.seed,sec)
                            comparison.append(dict(held=held,width=width,horizon=horizon,target=target,model=name,parent=parent,block_seconds=sec,n=len(test),**ci))
                    for j,ix in enumerate(test):predrows.append(dict(window=int(ix),held=held,width=width,horizon=horizon,target=target,y=y[ix],
                        **{k:float(v[j]) for k,v in predictions.items()}))
            for view in ('evaluation','evaluation_budget','joint','joint_budget','action_only','outputs_only'):
                folder=out/'motifs'/f'held_{held}'/f'w{width:g}'/view
                f=read_table(folder/'assignments.csv.gz');k=read_json(folder/'fit.json')['k'];idx=f.row.to_numpy(int)
                attached=d.iloc[idx].reset_index(drop=True)
                good,humanchange=raw_pair_changes(f,d,timelines)
                change=np.r_[False,f.state.to_numpy()[1:]!=f.state.to_numpy()[:-1]]
                # Shift the complete state sequence within each continuous segment, preserving autocorrelation.
                rng=np.random.default_rng(args.seed);null_same=[];null_cross=[]
                for _ in range(args.null_repeats):
                    shifted=f.state.to_numpy().copy()
                    for _,g in f.groupby('segment'):
                        ii=g.index.to_numpy();shifted[ii]=np.roll(shifted[ii],int(rng.integers(max(1,len(ii)//5),max(2,4*len(ii)//5))))
                    nc=np.r_[False,np.diff(shifted)!=0]
                    if (good&~humanchange).any():null_same.append(nc[good&~humanchange].mean())
                    if (good&humanchange).any():null_cross.append((~nc[good&humanchange]).mean())
                continuity.append(dict(held=held,width=width,view=view,k=k,same_action_pairs=int((good&~humanchange).sum()),
                    cross_action_pairs=int((good&humanchange).sum()),
                    state_change_same_action=float(change[good&~humanchange].mean()) if (good&~humanchange).any() else np.nan,
                    state_persistence_cross_action=float((~change[good&humanchange]).mean()) if (good&humanchange).any() else np.nan,
                    shifted_change_same_action=np.mean(null_same) if null_same else np.nan,
                    shifted_persistence_cross_action=np.mean(null_cross) if null_cross else np.nan))
                if (folder/'first_only_assignments.csv.gz').exists():
                    first_only=read_table(folder/'first_only_assignments.csv.gz')
                    np.testing.assert_array_equal(first_only.window,f.window)
                    fc=np.r_[False,np.diff(first_only.state)!=0]
                    continuity.append(dict(held=held,width=width,view=view+'_first_only',
                        k=len([c for c in first_only if c.startswith('p') and c[1:].isdigit()]),
                        same_action_pairs=int((good&~humanchange).sum()),cross_action_pairs=int((good&humanchange).sum()),
                        state_change_same_action=float(fc[good&~humanchange].mean()) if (good&~humanchange).any() else np.nan,
                        state_persistence_cross_action=float((~fc[good&humanchange]).mean()) if (good&humanchange).any() else np.nan))
                for state in range(k):
                    take=f.state.eq(state).to_numpy()
                    for metric in ['K',*DIRECT,'policy_change']:
                        profiles.append(dict(held=held,width=width,view=view,state=state,metric=metric,n=int(take.sum()),
                            mean=float(attached.loc[take,metric].mean()),sd=float(attached.loc[take,metric].std())))
                    for action in range(nact):profiles.append(dict(held=held,width=width,view=view,state=state,metric=f'action_{action}',
                        n=int(take.sum()),mean=float((attached.loc[take,'human_action']==action).mean()),sd=np.nan))
                    # Representative by distance to fitted center, plus seeded random held-out example.
                    candidates=np.flatnonzero(take)
                    if len(candidates):
                        coords=np.load(folder/'test_embedding.npy');gm=np.load(folder/'mixture.npz')
                        dist=np.sum((coords[candidates]-gm['means'][state])**2/gm['covariances'][state],axis=1)
                        for kind,j in [('central',candidates[np.argmin(dist)]),('random',rng.choice(candidates))]:
                            examples.append(dict(held=held,width=width,view=view,state=state,kind=kind,row=int(f.row.iloc[j]),window=int(f.window.iloc[j])))
                # Conditional per-K continuity, so length-driven state separation remains visible.
                if not view.endswith('_budget'):
                    for kval in sorted(attached.K.unique()):
                        mask=good&attached.K.eq(kval).to_numpy()&np.r_[False,attached.K.to_numpy()[:-1]==kval]
                        continuity.append(dict(held=held,width=width,view=view,k=k,budget_stratum=int(kval),
                            same_action_pairs=int((mask&~humanchange).sum()),cross_action_pairs=int((mask&humanchange).sum()),
                            state_change_same_action=float(change[mask&~humanchange].mean()) if (mask&~humanchange).any() else np.nan,
                            state_persistence_cross_action=float((~change[mask&humanchange]).mean()) if (mask&humanchange).any() else np.nan))
                # Test candidate membership beyond baseline, using training responsibilities from frozen outer fit.
                if width==.5:
                    blocks=feature_blocks(out,fit,view);train=nonoverlap(w,w.index[w.session.isin(fit)&w.width.eq(width)].to_numpy(),width)
                    prep=BlockProjection().fit(blocks,train,args.seed)
                    gm=mixture_fit(prep.transform([b[train] for b in blocks]),k,args)
                    selected=np.unique(np.r_[train,f.window.to_numpy(int)]);resp=np.zeros((len(w),k))
                    resp[selected]=gm.predict_proba(prep.transform([b[selected] for b in blocks]))
                    y=w['switch_0.25'].to_numpy();tr=train[np.isfinite(y[train])];te=f.window.to_numpy(int);te=te[np.isfinite(y[te])]
                    if len(tr)>8 and len(te):
                        bp=readout(base,y,tr,te,args.seed);mp=readout(base+[resp],y,tr,te,args.seed)
                        motif_preds[(held,view)]=(te,mp)
                        gain=loss(y[te],bp)-loss(y[te],mp)
                        comparison.append(dict(held=held,width=width,horizon=.25,target='switch',model='motif_'+view,parent='baseline',
                            block_seconds=10,n=len(te),**block_interval(w.loc[te],gain,args.bootstrap,args.seed)))
    y=w['switch_0.25'].to_numpy()
    for held in sessions:
        for view in ('evaluation','joint','evaluation_budget','joint_budget'):
            for reference in ('action_only','outputs_only'):
                if (held,view) not in motif_preds or (held,reference) not in motif_preds:continue
                te,mp=motif_preds[held,view];tr,bp=motif_preds[held,reference]
                np.testing.assert_array_equal(te,tr)
                gain=loss(y[te],bp)-loss(y[te],mp)
                for sec in (5,10,20):
                    comparison.append(dict(held=held,width=.5,horizon=.25,target='switch',model='motif_'+view+'_over_'+reference,
                        parent='motif_'+reference,block_seconds=sec,n=len(te),**block_interval(w.loc[te],gain,args.bootstrap,args.seed,sec)))
    save_table(out/'predictions.csv.gz',predrows);save_table(out/'budget_predictions.csv.gz',budget_predictions);save_table(out/'comparisons.csv',comparison)
    save_table(out/'continuity.csv',continuity);save_table(out/'profiles.csv',profiles);save_table(out/'examples.csv',examples)
    # Actual voluntary continuation is a descriptive calibration check, not a causal utility test.
    ev=e.loc[e.voluntary.astype(bool)].copy();ev.to_csv(out/'voluntary_choices.csv.gz',index=False)
    write_json(out/'validation.json',dict(outer_folds=4,heldout='analysis fits only; model trained on all sessions',
        predictions=len(predrows),conditional_associations_only=True,uncertainty='within-run time block bootstrap; equal-run mean',
        continuity_null='circularly shifted state sequence per segment, preserves overlap-induced temporal dependence'))


def self_test():
    import unittest
    import tempfile
    from types import SimpleNamespace
    def bind_checkpoint_fixture(capture, cp):
        # game_inputs now authenticates a completed capture's exact files.
        # Empty directories no longer constitute a valid checkpoint fixture.
        binding = {}
        for kind in ('actor','model','config'):
            path = cp/kind
            path.write_text(str(cp)+kind)
            binding[kind] = dict(name=kind,sha256=sha(path))
        write_json(capture/'capture_plan.json',dict(semantic=dict(checkpoint=binding)))
    class Contracts(unittest.TestCase):
        def test_checkpoint_resolution_separates_participants(self):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)/'campaign'
                for subject in (1,2):
                    capture=root/'captures'/f'pong-sub{subject:03d}-all'
                    capture.mkdir(parents=True);(capture/'COMPLETED.json').write_text('{}')
                    cp=root/'checkpoints'/f'pong-s{subject:03d}-test'/'milestones/10000000';cp.mkdir(parents=True)
                    bind_checkpoint_fixture(capture,cp)
                capture,cp=game_inputs(root,'pong',2)
                self.assertEqual(capture.name,'pong-sub002-all')
                self.assertIn('pong-s002-test',str(cp));self.assertNotIn('s001',str(cp))
        def test_checkpoint_reuse_from_sibling_campaign(self):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)/'current';capture=root/'captures'/'spaceinvaders-sub002-all'
                capture.mkdir(parents=True);(capture/'COMPLETED.json').write_text('{}')
                cp=Path(tmp)/'previous/checkpoints/spaceinvaders-s002-test/milestones/10000000';cp.mkdir(parents=True)
                bind_checkpoint_fixture(capture,cp)
                self.assertEqual(game_inputs(root,'spaceinvaders',2)[1],cp)
                with self.assertRaises(ValueError):game_inputs(root,'spaceinvaders',1)
        def test_zero_and_one_computation_values(self):
            self.assertEqual(value_summary([2])['U'],0)
            self.assertTrue(np.isnan(value_summary([2])['coherence']))
            self.assertEqual(value_summary([2,1])['A'],1)
            self.assertEqual(value_summary([2,1])['D'],-1)
            self.assertEqual(value_summary([2,1])['U'],1)
        def test_net_vs_absolute_updates(self):
            r=value_summary([1,3,2]);self.assertEqual(r['A'],3);self.assertEqual(r['D'],1)
            self.assertAlmostEqual(r['coherence'],1/3)
        def test_decision_uses_pre_real_state(self):
            edges=list(decision_edges(np.array([1,0,0,0,0],bool),np.array([0,0,1,0,1],bool),
                np.array([0,1,0,1,0],bool),np.array([0,1,0,1,0]),np.zeros(5,bool)))
            self.assertEqual([(x[1],x[2]) for x in edges],[(0,2),(2,4)])
            self.assertEqual(edges[1][3].tolist(),[3])
        def test_reset_breaks_future_path(self):
            edges=list(decision_edges(np.array([1,0,0,0],bool),np.array([0,0,0,1],bool),
                np.array([0,1,1,0],bool),np.array([0,1,0,0]),np.array([0,0,1,0],bool)))
            self.assertEqual(edges[0][4],[1]);self.assertEqual(len(edges[0][3]),2)
        def test_actor_phase_reconstruction(self):
            h=object.__new__(ActorHeads);h.phase=np.array([[0,0],[3,4],[1,2]])
            x=np.array([[1,2],[4,6]])
            np.testing.assert_allclose(h.common(x,[0,1]),[[1,2],[1,2]])
        def test_exact_voc_difference_invariant_to_dueling_center(self):
            raw=np.array([[2.,1.],[-3,2]])
            for prob in (0.,.5,1.):
                q=5+raw-(raw*np.array([prob,1-prob])).sum(1,keepdims=True)+raw.mean(1,keepdims=True)
                np.testing.assert_allclose(q[:,0]-q[:,1],raw[:,0]-raw[:,1])
        def test_no_future_event_interpolation(self):
            ix=sa_indices(np.array([0,.1,.2]),np.array([.15]),np.array([0,.1]),max_age=.12)
            np.testing.assert_array_equal(ix,[[1,-1]])
        def test_action_timeline_catches_switch_return(self):
            with tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp)/'a.npz';np.savez(p,time=[0,.05,.1,.15,.2],action=np.eye(2)[[0,1,0,0,0]],
                    is_first=[1,0,0,0,0],is_terminal=[0,0,0,0,0]);tl=ActionTimeline(p)
                self.assertEqual(tl.counts(0,.15).sum(),2)
                np.testing.assert_allclose(tl.area(.15),[.1,.05])
        def test_same_action_endpoints_do_not_hide_raw_switch_return(self):
            with tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp)/'a.npz';np.savez(p,time=[0,.03,.06,.1],action=np.eye(2)[[0,1,0,0]],
                    is_first=[1,0,0,0],is_terminal=[0,0,0,0])
                d=pd.DataFrame(dict(source=[str(p),str(p)]))
                f=pd.DataFrame(dict(row=[0,1],time=[0,.1],segment=[1,1]))
                valid,changed=raw_pair_changes(f,d,{})
                np.testing.assert_array_equal(valid,[False,True]);np.testing.assert_array_equal(changed,[False,True])
        def test_gap_segmentation(self):
            d=pd.DataFrame(dict(episode=['a']*4,decision=[0,1,2,4],time=[0,.1,.5,.6]))
            np.testing.assert_array_equal(timeline_segments(d).segment,[1,1,2,3])
        def test_window_never_uses_future(self):
            d=pd.DataFrame(dict(time=np.arange(10)*.1));w=pd.DataFrame(dict(left=[0],row=[4]))
            x=np.arange(10.)[:,None];a=window_features(x,d,w);x[5:]=1e9
            np.testing.assert_array_equal(a,window_features(x,d,w))
        def test_fit_does_not_use_heldout_scaling(self):
            rng=np.random.default_rng(7);x=rng.normal(size=(40,6));tr=np.arange(25)
            p=BlockProjection().fit([x],tr);a=p.transform([x])[:25];x[25:]=1e10
            q=BlockProjection().fit([x],tr);np.testing.assert_allclose(a,q.transform([x])[:25])
        def test_missing_blocks_and_constant_features(self):
            x=np.full((30,3),np.nan);p=BlockProjection().fit([x],np.arange(20))
            self.assertEqual(p.rank,0);self.assertTrue(np.isfinite(p.transform([x])).all())
        def test_equal_decision_sampling(self):
            d=pd.DataFrame(dict(run=['a']*5+['b']*100))
            ids=sample_runs(d,d.index.to_numpy(),10,701)
            self.assertEqual((d.loc[ids,'run']=='a').sum(),5)
        def test_zero_sensitivity_not_fabricated(self):
            q,s,r,e=ace_basis(np.eye(3),np.zeros((2,3)));self.assertEqual(q.shape,(3,0));self.assertEqual(r,0)
        def test_scalar_linear_target_rank_one(self):
            q,s,r,e=ace_basis(np.eye(4),np.array([[1,2,3,4.]]));self.assertEqual(r,1)
            np.testing.assert_allclose(np.linalg.norm(q.T@q-np.eye(1)),0,atol=1e-12)
        def test_covariance_factor_exact_without_wide_svd(self):
            x=np.random.default_rng(2).normal(size=(32,100));weights=np.ones(32)/32
            mean,factor=activation_factor(x,np.arange(32),weights)
            np.testing.assert_allclose(factor@factor.T,(x-mean).T@(x-mean)/32,atol=1e-12)
        def test_heldout_labels_do_not_fit_readout(self):
            rng=np.random.default_rng(4);x=rng.normal(size=(60,8));y=(x[:,0]>0).astype(float)
            tr=np.arange(40);te=np.arange(40,60);a=readout([x],y,tr,te,701)
            y[te]=1-y[te];np.testing.assert_allclose(a,readout([x],y,tr,te,701))
        def test_one_se_prefers_single_component_when_supported(self):
            t=pd.DataFrame([dict(k=k,score=s,status='ok') for k,ss in [(1,[1.,1.,1.]),(2,[.5,1.1,1.7])] for s in ss])
            self.assertEqual(one_se_choice(t,3),1)
        def test_one_se_rejects_missing_validation_fold(self):
            t=pd.DataFrame([dict(k=1,score=1.,status='ok'),dict(k=2,score=100.,status='ok')])
            with self.assertRaises(ValueError):one_se_choice(t,3)
        def test_coordinate_summary_handles_zero_and_one(self):
            d=pd.DataFrame(dict(start=[0,1],end=[1,3]));z=np.array([[4.],[2.],[3.]])
            s=summarize_coordinates(z,d);np.testing.assert_allclose(s[0],[4,0,0,0]);np.testing.assert_allclose(s[1],[2,1,1,.5])
        def test_nonoverlap_is_segment_specific(self):
            w=pd.DataFrame(dict(segment=[1]*5+[2]*5,time=list(np.arange(5)*.1)*2))
            self.assertEqual(nonoverlap(w,w.index.to_numpy(),.25).tolist(),[0,3,5,8])
        def test_block_interval_not_event_independence(self):
            f=pd.DataFrame(dict(run=['a']*20,time=np.arange(20)/2));v=np.arange(20.)
            a=block_interval(f,v,50,701,5);b=block_interval(pd.concat([f,f]),np.r_[v,v],50,701,5)
            for k in ('mean','low','high','blocks'):self.assertAlmostEqual(a[k],b[k])
        def test_single_component_is_allowed(self):
            x=np.random.default_rng(1).normal(size=(50,2));m=mixture_fit(x,1,SimpleNamespace(smoke=True,seed=701))
            np.testing.assert_allclose(m.predict_proba(x),1)
        def test_partial_unit_resume_checksum(self):
            with tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp);(p/'x').write_text('a');complete_unit(p)
                self.assertTrue(resume_unit(p,SimpleNamespace(resume=True)))
                (p/'x').write_text('b')
                with self.assertRaises(RuntimeError):resume_unit(p,SimpleNamespace(resume=True))
        def test_session_sets_exclude_outer_test(self):
            sets=fit_sets([1,2,3,4]);self.assertEqual(len(sets),10)
            for held in [1,2,3,4]:
                outer=tuple(s for s in [1,2,3,4] if s!=held)
                self.assertIn(outer,sets)
                for inner in itertools.combinations(outer,2):self.assertIn(inner,sets);self.assertNotIn(held,inner)
        def test_probability_distance_identity(self):
            p=np.array([[.2,.8]]);self.assertEqual(js_distance(p,p)[0],0)
    return unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts)).wasSuccessful()



def complete_unit(folder):
    """Authenticate disposable nested-fit scratch; never a public result tree."""
    files = {str(p.relative_to(folder)): sha(p) for p in folder.rglob('*')
             if p.is_file() and p.name != 'unit.done.json'}
    write_json(folder/'unit.done.json', dict(version=VERSION, artifacts=files))


def resume_unit(folder, args):
    marker = folder/'unit.done.json'
    if not marker.exists():
        return False
    if not args.resume:
        raise RuntimeError('Completed unit exists; use --resume')
    old = read_json(marker)
    if old['version'] != VERSION or not all((folder/k).is_file() and sha(folder/k) == value
                                          for k, value in old['artifacts'].items()):
        raise RuntimeError('Completed unit changed: ' + str(folder))
    return True
