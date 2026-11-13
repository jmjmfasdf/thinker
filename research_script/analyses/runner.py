"""Scientific theme execution; input caches are read-only."""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from common.paths import output,tag,GAMES,THEMES
from common.features import prepare_unit,load_unit
from common.io import sha,signature,write_json
from common.reporting import save_tables,figures,aggregate
from . import behavior

SUPPORTED=['02','03','04','05','06','07','08','09','10','12','13']

def run_unit(number,cache,outputs,subject,game):
    number=f'{int(number):02d}';root=output(number,outputs)/tag(subject,game)
    root.mkdir(parents=True,exist_ok=True)
    source_root=Path(__file__).resolve().parents[1]
    sources={str(p.relative_to(source_root)):sha(p) for folder in ['common','analyses'] for p in (source_root/folder).rglob('*.py')}
    ident=dict(theme=number,subject=subject,game=game,cache=str(Path(cache).resolve()),
               input_manifest=sha(Path(cache)/'manifest.json'),sources=sources,heldout='checkpoint_leave_session == evaluated_session')
    if number in ['06','07','08']:
        from .neural import BRAIN
        brain_manifest=BRAIN/tag(subject,game)/'cache_manifest.json'
        ident['brain_cache']=str(BRAIN)
        ident['brain_manifest_sha256']=sha(brain_manifest) if brain_manifest.exists() else None
    token=signature(ident);mp=root/'manifest.json'
    if mp.exists():
        old=json.loads(mp.read_text())
        if old.get('status')=='COMPLETED' and old.get('signature')==token:
            if all((root/f).exists() and sha(root/f)==h for f,h in old.get('artifacts',{}).items()):return 'REUSED'
        if old.get('status')=='COMPLETED':raise ValueError(f'Completed output differs; preserve it before rerunning: {root}')
    write_json(mp,dict(ident,signature=token,status='RUNNING',started_at=time.time()))
    prepare_unit(cache,outputs,subject,game)
    d,e=load_unit(outputs,subject,game,events=number in ['02','03','04','12'])
    if number=='12':
        from .continuity import attach_arrived_mismatch
        d=attach_arrived_mismatch(d,cache,subject,game)
    tables={}
    if number=='02':tables=behavior.boundary(d,e)
    elif number=='03':tables=behavior.control(d,e)
    elif number=='04':tables=behavior.search_process(d,e)
    elif number=='09':
        from .task_state import run
        tables=run(d,cache,subject,game)
    elif number=='12':tables=behavior.temporal_associations(d,e)
    if number in ['04','05','12']:
        more={}
        for cp in range(1,5):
            full,_=load_unit(outputs,subject,game,heldout=False,events=False,checkpoint=cp)
            if number=='12':full=attach_arrived_mismatch(full,cache,subject,game)
            result=behavior.actor_readout(full) if number=='04' else behavior.human_prediction(full,temporal=number=='12')
            for k,v in result.items():more.setdefault(k,[]).append(v)
        tables.update({k:pd.concat(v,ignore_index=True) for k,v in more.items()})
    if number in ['06','07','08']:
        from .neural import run_neural
        tables,extra=run_neural(number,d,cache,outputs,subject,game)
        if extra.get('status')!='COMPLETED':
            write_json(mp,dict(ident,signature=token,**extra));return extra['status']
    if number=='13':
        from .optional import behavioral_modes
        return behavioral_modes(root,ident,token,subject,game)
    for frame in tables.values():frame['subject']=subject;frame['game']=game
    save_tables(root,tables);figures(root,tables,f'{number} · {GAMES[game]} · subject {subject:03d}')
    failed_predictions={k:v.loc[v.status.ne('ok')].to_dict('records') for k,v in tables.items()
                        if k in ['fit_qa','actor_fit_qa'] and 'status' in v and v.status.ne('ok').any()}
    qa={'decisions':len(d),'heldout_sessions':sorted(map(int,d.session.unique())),
        'runs':int(d.run_label.nunique()),'zero_budget_decisions':int(d.additional_computations.eq(0).sum()),
        'duplicate_human_keys':int(d.duplicated(['run_label','real_step','real_step_id']).sum()),
        'fit_status_counts':{k:v.status.value_counts().to_dict() for k,v in tables.items() if 'status' in v}}
    qa['failed_prediction_fits']=failed_predictions
    write_json(root/'qa.json',qa)
    artifacts={str(p.relative_to(root)):sha(p) for folder in ['tables','figures'] for p in (root/folder).glob('*') if p.is_file()}
    status='PARTIAL_FITS' if failed_predictions else 'COMPLETED'
    write_json(mp,dict(ident,signature=token,status=status,completed_at=time.time(),qa=qa,artifacts=artifacts,
        interpretation='Associational/readout analysis. Model search does not identify human planning causally.',
        limitations=['09 RAM change geometry is observational, not causal controllability or cognitive uncertainty.'] if number=='09' else []))
    return status
