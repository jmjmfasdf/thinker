"""RAM-state descriptors and observational action-conditioned change geometry."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from common.paths import REPO,tag
from common.io import sha
from .behavior import association

def run(d,cache,subject,game):
    d=d.copy();records=[];gramians=[];audit=[]
    sources={}
    for cp in range(1,5):
        p=Path(cache)/f'checkpoint_leave{cp:02d}'/tag(subject,game)
        m=json.loads((p/'manifest.json').read_text())
        for name,entry in m['source_runs'].items():sources[entry['run_label']]=(name,entry['source_sha256'])
    for label,part in d.groupby('run_label'):
        relative,expected=sources[label];path=REPO/'behavioral_data_block'/relative
        if not path.exists() or sha(path)!=expected:raise ValueError('RAM behavioral source mismatch: '+str(path))
        with np.load(path,allow_pickle=False) as z:
            if 'RAM' not in z:raise ValueError('No RAM in '+str(path))
            ram=z['RAM'][part.current_frame.to_numpy(int)].astype(np.uint8)
        norm=ram.astype(float)/255.
        entropy=[]
        for row in ram:
            p=np.bincount(row,minlength=256)/ram.shape[1];entropy.append(float(-(p[p>0]*np.log2(p[p>0])).sum()))
        change=np.mean(ram[1:]!=ram[:-1],axis=1);ok=part.segment.to_numpy()[1:]==part.segment.to_numpy()[:-1]
        change=np.r_[np.nan,np.where(ok,change,np.nan)]
        d.loc[part.index,'ram_entropy']=entropy;d.loc[part.index,'ram_change']=change
        d.loc[part.index,'ram_variance']=norm.var(1)
        delta=norm[1:]-norm[:-1];actions=part.human_action.to_numpy(int)[:-1]
        cols=[]
        for a in range(int(part.n_actions.iloc[0])):
            use=ok&(actions==a)
            if use.any():cols.append(delta[use].mean(0))
        if cols:
            b=np.stack(cols,axis=1);sv=np.linalg.svd(b,compute_uv=False);eigen=sv**2;positive=eigen[eigen>1e-12]
            prob=positive/positive.sum() if positive.size else positive
            gramians.append(dict(run_label=label,n_transitions=int(ok.sum()),observed_actions=len(cols),
                trace=float(eigen.sum()),largest_eigenvalue=float(eigen.max()),
                effective_rank=float(np.exp(-(prob*np.log(prob)).sum())) if len(prob) else 0.,
                interpretation='B=mean observed RAM change by action; B B^T is descriptive, not causal controllability'))
        audit.append(dict(run_label=label,source=str(path),sha256=expected,ram_bytes=ram.shape[1],decisions=len(part)))
    for target in ['log_budget','ema_initial']:
        for term in ['visual_change','ram_entropy','ram_change','ram_variance']:
            records+=association(d.loc[d.scored.eq(1)],target,[term],
                ['M','root_value','log_age','prior_reward','carried'],False,
                family='observed_environment_state',population=target)
    return {'effects':pd.DataFrame(records),'observational_gramian':pd.DataFrame(gramians),'ram_sources':pd.DataFrame(audit)}
