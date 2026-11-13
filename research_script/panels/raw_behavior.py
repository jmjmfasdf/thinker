"""Model-independent COSYNE Fig2 A--C from ALL original human observations.

Human-only panels must not inherit the replay sampling grid. Source files are
discovered from the 10M cache manifest and verified against recorded SHA256.
The same human run is loaded once, not once per model checkpoint.
"""
from pathlib import Path
import json
import shutil
import time
import numpy as np
import pandas as pd
from common.paths import REPO,output,tag
from common.native import session_of
from common.io import sha,signature,write_json
from common.reporting import save_tables
from .data import behavior_panels


def frame_from_raw(action,times,is_first,is_terminal,run):
    action=np.asarray(action); a=action.argmax(1) if action.ndim==2 else action.astype(int)
    n_actions=action.shape[1] if action.ndim==2 else int(a.max())+1
    t=np.asarray(times,float).reshape(-1)
    if not len(t) or len(t)!=len(a) or not np.isfinite(t).all():raise ValueError('Invalid human timestamps/actions')
    boundary=np.asarray(is_first,bool).reshape(-1).copy();boundary[0]=True
    boundary[1:]|=np.asarray(is_terminal,bool).reshape(-1)[:-1]|(np.diff(t)<=0)
    previous=np.r_[-1,a[:-1]].astype(int);previous[boundary]=-1
    onset=boundary.copy();onset[1:]|=a[1:]!=a[:-1]
    starts=np.maximum.accumulate(np.where(onset,np.arange(len(a)),0))
    before=np.r_[0,starts[:-1]]
    age=np.arange(len(a))-before;age_seconds=t-t[before]
    age_seconds[boundary]=np.nan
    d=pd.DataFrame(dict(run_label=run,real_step=np.arange(1,len(a)+1),
        segment=np.cumsum(boundary),human_action=a,incumbent_action=previous,
        n_actions=n_actions,session=session_of(run),scored=1,time_seconds=t,
        bout_age=age,age_seconds=age_seconds,log_age=np.log1p(age)))
    for j in range(1,n_actions):d[f'previous_action_{j}']=(previous==j).astype(float)
    return d


def run(cache,outputs,subject,game,promote=True):
    cache=Path(cache);sources={}
    for cp in range(1,5):
        p=cache/f'checkpoint_leave{cp:02d}'/tag(subject,game)/'manifest.json'
        for rel,rec in json.loads(p.read_text())['source_runs'].items():
            pair=(rec['run_label'],rec['source_sha256'])
            if rel in sources and sources[rel]!=pair:raise ValueError('Human source differs between checkpoints')
            sources[rel]=pair
    identity=dict(subject=subject,game=game,scope='ALL original human recording observations; no replay-grid downsampling',
        source_sha256=sha(__file__),estimator_sha256=sha(Path(__file__).with_name('data.py')),
        sources={rel:digest for rel,(_,digest) in sources.items()},independent_data='one copy of each human run')
    token=signature(identity);base=output(1,outputs)
    root=base/('panels' if promote else 'raw_panels_pending')/tag(subject,game)
    if (root/'manifest.json').exists():
        m=json.loads((root/'manifest.json').read_text())
        if m.get('signature')==token and m.get('status')=='COMPLETED' and all(
            (root/f).exists() and sha(root/f)==h for f,h in m.get('artifacts',{}).items()):return 'REUSED'
        archive=base/'archive'/f'before_raw_behavior_{time.time_ns()}'/tag(subject,game)
        archive.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(root),str(archive))
    root.mkdir(parents=True,exist_ok=True);write_json(root/'manifest.json',dict(identity,signature=token,status='RUNNING'))
    frames=[];audits=[]
    for rel,(run_label,digest) in sorted(sources.items()):
        path=REPO/'behavioral_data_block'/rel
        if sha(path)!=digest:raise ValueError('Original human file hash mismatch: '+str(path))
        with np.load(path,allow_pickle=False) as z:
            frame=frame_from_raw(z['action'],z['time'],z['is_first'],z['is_terminal'],run_label)
        frames.append(frame);audits.append(dict(run_label=run_label,source=str(path),sha256=digest,
            n_original_observations=len(frame),n_eligible=int(frame.incumbent_action.ge(0).sum()),
            n_boundary=int(frame.incumbent_action.lt(0).sum())))
    d=pd.concat(frames,ignore_index=True);tables=behavior_panels(d);tables['source_alignment']=pd.DataFrame(audits)
    if not tables['duration_fit_qa'].status.eq('ok').all():raise ValueError('Original-human duration fit failed')
    for table in tables.values():table['subject']=subject;table['game']=game
    save_tables(root,tables)
    write_json(root/'qa.json',dict(n_original_observations=len(d),n_runs=d.run_label.nunique(),
        uses_replay_sampling=False,duplicated_checkpoint_copies=0))
    write_json(root/'manifest.json',dict(identity,signature=token,status='COMPLETED',completed_at=time.time(),
        artifacts={str(p.relative_to(root)):sha(p) for p in (root/'tables').glob('*')}))
    return 'COMPLETED'
