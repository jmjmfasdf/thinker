"""Dynamic regressors with independently cached BOLD; no fixed-length traces.

The old model regressors are never reused. Behavioral source SHA and raw frame
timestamps must match before reusing BOLD. Enduro is reported unavailable when
there is no verified ROI cache. Volume 60 / TR=1s is the inherited acquisition
alignment assumption, explicitly recorded rather than claimed measured.
"""
from pathlib import Path
import json
import os
import re
import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.spatial.distance import pdist
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from common.paths import REPO, output, tag
from common.io import sha, write_json

_relative='thinker/research_script/outputs/00_brain_cache/time_aligned_cache/v4_native_time_hrf'
_candidate=REPO.parent/_relative
BRAIN=Path(os.environ.get('THINKER_BRAIN_CACHE',str(_candidate if _candidate.exists() else Path.home()/_relative)))
FOCAL=['M','U_all','K','ema_initial','log_budget']

def run_identity(label):
    m=re.fullmatch(r'sub(\d+)_ses(\d+)_block(\d+)_game(\d+)',str(label))
    if not m: raise ValueError('Unrecognized run identity: '+str(label))
    return tuple(map(int,m.groups()))

def forward(times,values,n_tr=480,dt=.02,impulse_cols=()):
    """Persistent holds / decision-locked impulses, followed by a causal HRF."""
    times=np.asarray(times,float);values=np.asarray(values,float)
    if np.any(np.diff(times)<0) or not np.isfinite(times).all():raise ValueError('Invalid raw behavioral times')
    grid=np.arange(0,n_tr,dt);ix=np.searchsorted(times,grid,side='right')-1
    x=values[np.clip(ix,0,len(values)-1)].copy();x[ix<0]=0
    x[~np.isfinite(x)]=0
    for col in impulse_cols:
        x[:,col]=0
        positions=np.ceil(times/dt-1e-12).astype(int);good=(positions>=0)&(positions<len(grid))&np.isfinite(values[:,col])
        np.add.at(x[:,col],positions[good],values[good,col]/dt)
    t=np.arange(0,32,dt);h=stats.gamma.pdf(t,6)-stats.gamma.pdf(t,16)/6;h/=h.sum()
    convolved=signal.fftconvolve(x,h[:,None],mode='full',axes=0)[:len(grid)]
    return convolved[np.rint(np.arange(n_tr)/dt).astype(int)]

def prepare(d,cache,outputs,subject,game):
    root=output(0,outputs)/'neural'/tag(subject,game);root.mkdir(parents=True,exist_ok=True)
    brain=BRAIN/tag(subject,game)
    if not brain.exists():return [],dict(status='BLOCKED_INPUT',reason='No verified BOLD/ROI cache for this subject/game',required=str(brain))
    sources={}
    for cp in range(1,5):
        p=Path(cache)/f'checkpoint_leave{cp:02d}'/tag(subject,game)
        m=json.loads((p/'manifest.json').read_text())
        for name,entry in m['source_runs'].items():sources[run_identity(entry['run_label'])]=(name,entry['source_sha256'])
    runs=[];checks=[]
    for label,part in d.groupby('run_label'):
        identity=run_identity(label);s,se,block,g=identity
        br=brain/f'sub{s:03d}_ses{se:02d}_block{block:02d}_game{g}'
        align=list(br.glob('time_alignment*.json'))
        if len(align)!=1:return [],dict(status='BLOCKED_INPUT',reason='Missing/ambiguous BOLD run alignment',run=label)
        am=json.loads(align[0].read_text());behavior=am['source_signature']['behavior']
        source_rel,expected=sources[identity]
        if expected!=behavior['sha256']:raise ValueError('BOLD behavior source differs: '+label)
        raw=REPO/'behavioral_data_block'/source_rel
        if not raw.exists():raw=Path(behavior['path'])
        if not raw.exists() or sha(raw)!=expected:raise ValueError('Cannot validate original behavioral timing: '+str(raw))
        part=part.sort_values('real_step');frames=part.current_frame.to_numpy(int)
        with np.load(raw,allow_pickle=False) as z:
            times=z['time'][frames].astype(float)
        if times.min()<0 or times.max()>480.1:raise ValueError('Behavior outside canonical 480s acquisition window')
        controls=['log_age','recent10','prior_reward']+[c for c in part if c.startswith('previous_action_')]
        initial=['root_value','initial_q_margin']+[c for c in part if c.startswith('log_policy_')]
        visual=[c for c in part if c.startswith('visual_') and c!='visual_change']
        cols=FOCAL+controls+visual+initial
        values=part[cols].to_numpy(float)
        missing=(~np.isfinite(values)).astype(float)
        impulses=[cols.index(c) for c in ['U_all','K','log_budget']]
        x=forward(times,np.column_stack([values,missing]),impulse_cols=impulses)
        names=cols+[c+'_missing' for c in cols]
        # Drift is not HRF-convolved. No imaginary-event timestamps are invented.
        x=np.column_stack([x,np.linspace(-1,1,480),np.linspace(-1,1,480)**2]);names+=['drift','drift_squared']
        file=root/(label+'.npz')
        np.savez_compressed(file,x=x,names=np.asarray(names))
        roi=br/'roi_bold_maxall'
        if not roi.exists():return [],dict(status='BLOCKED_INPUT',reason='Full-voxel BOLD cache absent',run=label)
        runs.append(dict(run=label,session=se,x=x,names=names,roi=roi))
        checks.append(dict(run=label,behavior_sha256=expected,raw_time_range=[times.min(),times.max()],
                           native_vs_raw_time_max_difference=float(np.max(np.abs(times-part.decision_time))),
                           roi=str(roi),columns=names))
    write_json(root/'manifest.json',dict(status='COMPLETED',runs=checks,
        timebase='Raw source NPZ time[current_frame], independently SHA-verified; not synthetic 15-Hz time',
        acquisition_assumption='Behavior zero at analyzed fMRI volume 60; 480 TRs, TR 1 second; inherited assumption, no scanner triggers',
        forward='Persistent states use causal hold; U/K/budget use decision-locked impulses at 50 Hz, then canonical HRF; no imagined-event timestamps',
        confounds='Visual descriptors, preceding action, duration/history/reward, missing indicators and drift; preprocessed ROI BOLD'))
    return runs,dict(status='COMPLETED')

def standardize(x,train=None):
    train=np.ones(len(x),bool) if train is None else train
    mean=np.mean(x[train],0);sd=np.std(x[train],0);keep=sd>1e-10
    return (x[:,keep]-mean[keep])/sd[keep]

def residual(x,z):
    return x-z@np.linalg.lstsq(z,x,rcond=1e-10)[0]

def rsa(runs):
    rows=[]
    for r in runs:
        x=r['x'];names=r['names'];n=len(x)
        # Exclude near-neighbour TR pairs before computing nuisance residuals.
        i,j=np.triu_indices(n,1);keep=(j-i)>=8
        def distance(cols):
            y=standardize(x[:,[names.index(c) for c in cols]])
            return stats.rankdata(pdist(y,'euclidean')[keep]) if y.shape[1] else np.zeros(keep.sum())
        controls=[distance([c for c in names if c.startswith('visual_')]),
                  distance([c for c in names if c.startswith('previous_action_')]),
                  distance(['log_age','recent10','prior_reward']),stats.rankdata((j-i)[keep]),
                  distance([c for c in names if c.endswith('_missing')])]
        focal={k:distance([k]) for k in FOCAL}
        plans={}
        for k in FOCAL:
            # First report visual/action/history/time control. Stage 2 also controls initial support and quantity.
            for stage in ['behavior_visual_history','initial_and_quantity']:
                zcols=controls.copy()
                if stage=='initial_and_quantity':
                    zcols += [focal[c] for c in ['M','log_budget'] if c!=k]
                    zcols += [distance(['root_value','initial_q_margin']),distance([c for c in names if c.startswith('log_policy_') and not c.endswith('_missing')])]
                z=np.column_stack([np.ones(keep.sum()),*zcols]);z=np.column_stack([np.ones(len(z)),standardize(z[:,1:])])
                # Precompute an orthonormal nuisance span for every ROI in this run.
                u,s,_=np.linalg.svd(z,full_matrices=False);q=u[:,s>s[0]*1e-10]
                fx=focal[k]-q@(q.T@focal[k]);norm=np.linalg.norm(fx)
                plans[k,stage]=(q,fx,norm)
        for f in sorted(r['roi'].glob('*.npz')):
            with np.load(f,allow_pickle=False) as z:y=z['patterns'].astype(float)
            if y.shape[0]!=n or not np.isfinite(y).all():raise ValueError('Invalid BOLD shape/content: '+str(f))
            brain=stats.rankdata(pdist(y,'correlation')[keep])
            for (k,stage),(q,fx,norm) in plans.items():
                by=brain-q@(q.T@brain);den=norm*np.linalg.norm(by)
                value=float(fx@by/den) if den>1e-8 else np.nan
                rows.append(dict(run_label=r['run'],session=r['session'],roi=f.stem,stage=stage,term=k,
                                 value=value,n_pairs=int(keep.sum()),n_voxels=y.shape[1],status='ok' if np.isfinite(value) else 'constant'))
    return {'rsa_run_effects':pd.DataFrame(rows)}

def encoding(runs,decode=False):
    rows=[];qa=[];localization=[]
    names=runs[0]['names'];x=np.concatenate([r['x'] for r in runs]);session=np.concatenate([np.full(len(r['x']),r['session']) for r in runs])
    labels=np.concatenate([np.full(len(r['x']),r['run']) for r in runs]);rois=sorted(set.intersection(*[set(p.name for p in r['roi'].glob('*.npz')) for r in runs]))
    controls=[i for i,c in enumerate(names) if c not in FOCAL]
    focal=[names.index(c) for c in FOCAL]
    for roi in rois:
        yy=[];coords=None
        for r in runs:
            with np.load(r['roi']/roi,allow_pickle=False) as z:
                if coords is not None and not np.array_equal(coords,z['coords']):raise ValueError('ROI voxel order differs across runs')
                coords=z['coords'];yy.append(z['patterns'])
        y=np.concatenate(yy).astype(float)
        if not decode:
            for r,brain in zip(runs,yy):
                xx=r['x'];z=np.column_stack([np.ones(len(xx)),standardize(xx[:,controls])])
                ry=residual(brain,z)
                for fi,name in zip(focal,FOCAL):
                    rx=residual(xx[:,fi:fi+1],z).ravel()
                    # Signed partial correlation of ROI-mean BOLD; separate from held-out encoding.
                    mean_bold=ry.mean(1);den=np.linalg.norm(rx)*np.linalg.norm(mean_bold)
                    value=float(rx@mean_bold/den) if den>1e-10 else np.nan
                    localization.append(dict(run_label=r['run'],session=r['session'],roi=Path(roi).stem,
                        term=name,value=value,status='ok' if np.isfinite(value) else 'constant'))
        for held in sorted(set(session)):
            train=session!=held;test=~train
            if decode:
                # Brain PCA learned only from training sessions; all cached voxels contribute.
                pca=PCA(n_components=min(32,y.shape[1],train.sum()-1),svd_solver='randomized',random_state=20261111)
                brain_train=pca.fit_transform(y[train]);brain_test=pca.transform(y[test])
                control_x=standardize(x[:,controls],train)
                for fi,name in zip(focal,FOCAL):
                    target=x[:,fi];sd=target[train].std()
                    if sd<1e-10:continue
                    t=(target-target[train].mean())/sd
                    base=Ridge(alpha=100.).fit(control_x[train],t[train]).predict(control_x[test])
                    full=Ridge(alpha=100.).fit(np.column_stack([control_x[train],brain_train]),t[train]).predict(np.column_stack([control_x[test],brain_test]))
                    for run in np.unique(labels[test]):
                        take=labels[test]==run
                        delta=float(((t[test][take]-base[take])**2-(t[test][take]-full[take])**2).mean())
                        rows.append(dict(run_label=run,session=held,roi=Path(roi).stem,term=name,value=delta,n_voxels=y.shape[1],status='ok'))
            else:
                pred={}
                base_cols=controls+[names.index('M'),names.index('log_budget')]
                full_cols=base_cols+[names.index(c) for c in ['U_all','K','ema_initial']]
                for name,cols in [('controls',base_cols),('full',full_cols)]:
                    xx=standardize(x[:,cols],train)
                    pred[name]=Ridge(alpha=100.).fit(xx[train],y[train]).predict(xx[test])
                for run in np.unique(labels[test]):
                    take=labels[test]==run;truth=y[test][take]
                    sse0=((truth-pred['controls'][take])**2).sum(0);sse1=((truth-pred['full'][take])**2).sum(0)
                    denominator=((truth-y[train].mean(0))**2).sum(0);valid=denominator>1e-10
                    rows.append(dict(run_label=run,session=held,roi=Path(roi).stem,term='search_VOC_beyond_initial_quantity',
                        value=float(((sse0-sse1)[valid]/denominator[valid]).mean()),n_voxels=int(valid.sum()),status='ok'))
            qa.append(dict(roi=Path(roi).stem,heldout_session=held,n_train=int(train.sum()),n_test=int(test.sum()),alpha=100.))
    result={('decoding_run_effects' if decode else 'encoding_run_effects'):pd.DataFrame(rows),'neural_fit_qa':pd.DataFrame(qa)}
    if not decode:result['localization_run_effects']=pd.DataFrame(localization)
    return result

def run_neural(number,d,cache,outputs,subject,game):
    runs,status=prepare(d,cache,outputs,subject,game)
    if status['status']!='COMPLETED':return {},status
    tables=rsa(runs) if number=='07' else encoding(runs,decode=number=='06')
    return tables,status
