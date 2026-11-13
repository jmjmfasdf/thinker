"""Regenerate only matching-dependent analyses, preserving all old results."""
from pathlib import Path
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.paths import OUTPUTS,INPUT_CACHE,tag
from common.io import sha,write_json
from common import matching
from common.outcomes import risk_set
from panels import data
from experiments import planning_bout_screen as screen
from experiments import planning_bout_trajectories as trajectory
from experiments import value_update_candidates as candidate


def save_unit(path,tables,**meta):
    path.mkdir(parents=True,exist_ok=False)
    for name,frame in tables.items():
        p=path/name;p.parent.mkdir(exist_ok=True);frame.to_csv(p,index=False)
    write_json(path/'manifest.json',dict(status='COMPLETED',**meta,
        artifacts={str(p.relative_to(path)):sha(p) for p in path.rglob('*') if p.is_file()}))


def unit(subject,game,dest):
    label=tag(subject,game);parts=[];sources={}
    columns=set(screen.BASECOLS+trajectory.BASECOLS+['K','log_age','run_progress','prior_reward','valid_update_count'])
    for cp in range(1,5):
        path=OUTPUTS/'00/derived'/label/f'leave{cp:02d}/decisions.csv.gz'
        m=json.loads((path.parent/'manifest.json').read_text());sources[str(path)]=sha(path)
        assert sources[str(path)]==m['artifacts'][path.name]
        d=pd.read_csv(path,usecols=lambda c:c in columns or c.startswith('visual_') or c.startswith('actor_action_probability_'))
        parts.append(d.loc[d.session.eq(cp)])
    d=pd.concat(parts,ignore_index=True).sort_values(['run_label','real_step']).reset_index(drop=True)
    root=OUTPUTS/'02/value_update_candidates_20261111'/label;p=root/'candidate_decisions.csv.gz'
    m=json.loads((root/'manifest.json').read_text());sources[str(p)]=sha(p);assert sources[str(p)]==m['artifacts'][p.name]
    features=pd.read_csv(p);unique=[c for c in features if c not in d]+['real_step_id']
    d=d.merge(features[unique],on='real_step_id',validate='one_to_one')
    d['log_budget']=np.log1p(d.additional_computations);d['log_depth']=np.log1p(d.max_depth)
    start=d.run_label.ne(d.run_label.shift())|d.episode_id.ne(d.episode_id.shift())|d.real_step.diff().ne(1)|d.prior_done.eq(1)
    d['segment']=start.cumsum();d=data.scaled(d)
    pairs,means=data.matched_pairs(d)
    oldpath=OUTPUTS/'02/panels'/label/'tables/matched_pairs.csv';sources[str(oldpath)]=sha(oldpath)
    old=pd.read_csv(oldpath);compare=pairs.merge(old,on=['run_label','population','switch_id'],suffixes=('_new','_old'),validate='one_to_one')
    assert len(compare)==len(pairs)==len(old)
    summary=[];visual=[c for c in d if c.startswith('visual_') and c!='visual_change'];indexed=d.set_index('real_step_id')
    for pop,r in compare.groupby('population'):
        falsezero=r.loc[r.visual_distance_old.eq(0)]
        x=indexed.loc[falsezero.switch_id,visual].to_numpy();y=indexed.loc[falsezero.persist_id_old,visual].to_numpy()
        fake=(np.linalg.norm(x-y,axis=1)>1e-10)
        summary.append(dict(subject=subject,game=game,population=pop,n_pairs=len(r),
            changed_controls=int(r.persist_id_new.ne(r.persist_id_old).sum()),
            old_zero_distance=int(r.visual_distance_old.eq(0).sum()),false_old_zero_distance=int(fake.sum()),
            new_zero_distance=int(r.visual_distance_new.eq(0).sum())))
    save_unit(OUTPUTS/'02/matching_v2'/label,{'tables/matched_pairs.csv':pairs,'tables/matched_means.csv':means},
              matching_version=matching.VERSION,matching_source=sha(matching.__file__),sources=sources)
    sr,sc=screen.contrasts(d,pairs,subject,game)
    tr,te,tc=trajectory.unit_analysis(d,pairs,subject,game)
    cr,cc=candidate.matched_analysis(d,pairs)
    for frame in (cr,cc):frame['subject']=subject;frame['game']=game
    save_unit(dest/'value_candidates'/label,{'run_contrasts.csv':cr,'coverage.csv':cc},sources=sources)
    pe=[]
    for cp in range(1,5):
        p=INPUT_CACHE/f'checkpoint_leave{cp:02d}'/label/'persistence_events.csv.gz'
        e=pd.read_csv(p);e=e.merge(d.loc[d.session.eq(cp),['real_step_id','incumbent_action','initial_alternative']],on='real_step_id',validate='many_to_one')
        e['update']=e.root_q_revision.abs().where(e.value_revision_valid.eq(1)&np.isfinite(e.root_q_revision),0.)
        e['branch_type']=np.where(e.branch.eq(-1),'reset',np.where(e.branch.eq(e.incumbent_action),'ongoing',np.where(e.branch.eq(e.initial_alternative),'strongest_alternative','other_actions')))
        pe.append(e)
    composition,compcov=data.update_composition(d,pd.concat(pe,ignore_index=True),pairs)
    for frame in (means,composition,compcov):frame['subject']=subject;frame['game']=game
    model=[];probs=d[[f'actor_action_probability_{a}' for a in range(9 if game==0 else 6)]].to_numpy()
    d['model_switch']=1-probs[np.arange(len(d)),np.maximum(d.incumbent_action.to_numpy(int),0)]
    d['model_active']=1-probs[:,0];d['model_noop']=probs[:,0]
    for pop,key in [('overall','model_switch'),('noop_to_active','model_active'),('active_to_noop','model_noop')]:
        part=risk_set(d,pop).set_index('real_step_id')
        for run,p in pairs.loc[pairs.population.eq(pop)].groupby('run_label'):
            case=part.loc[p.switch_id,key].mean();stay=part.loc[p.persist_id,key].mean()
            model.append(dict(subject=subject,game=game,population=pop,run_label=run,n=len(p),case=case,maintain=stay,difference=case-stay))
    return dict(screen_runs=sr,screen_coverage=sc,trajectory_runs=tr,trajectory_effects=te,trajectory_coverage=tc,
                matching_qa=pd.DataFrame(summary),matched_means=means,composition=composition,composition_coverage=compcov,
                model_matched=pd.DataFrame(model))


def aggregate(dest,parts):
    allframes={k:pd.concat([r[k] for r in parts],ignore_index=True) for k in parts[0]}
    for name,frame in allframes.items():frame.to_csv(dest/f'{name}.csv.gz',index=False)
    runs=allframes['screen_runs']
    subjects=runs.groupby(['subject','game','population','metric'],as_index=False).agg(
        value=('value','mean'),n_runs=('value','count'),raw_difference=('raw_difference','mean'),
        switch_mean=('switch_mean','mean'),maintain_mean=('maintain_mean','mean'),
        informative_runs=('scale',lambda x:int(x.gt(1e-10).sum())))
    subjects.loc[subjects.n_runs.lt(3),'value']=np.nan
    root=dest/'bout_screen';root.mkdir(exist_ok=True);subjects.to_csv(root/'subject_contrasts.csv',index=False)
    sg=screen.group_inference(subjects);sg.to_csv(root/'group_results.csv',index=False);screen.plot(sg,root)
    root=dest/'trajectories';root.mkdir(exist_ok=True);keys=['subject','game','population','horizon','metric']
    values=['case','maintain','difference','centered_difference','case_change','maintain_change','raw_case','raw_maintain']
    sc=allframes['trajectory_runs'].groupby(keys+['lag'],as_index=False).agg(**{m:(m,'mean') for m in values},n_runs=('difference','size'))
    sc.loc[sc.n_runs.lt(3),values]=np.nan;sc.to_csv(root/'subject_trajectories.csv',index=False)
    se=allframes['trajectory_effects'].groupby(keys+['endpoint'],as_index=False).agg(value=('value','mean'),n_runs=('value','size'),
        case_change=('case_change','mean'),maintain_change=('maintain_change','mean'),informative_runs=('informative','sum'))
    se.loc[se.n_runs.lt(3),'value']=np.nan;se.to_csv(root/'subject_effects.csv',index=False)
    tg=trajectory.infer(se);tg.to_csv(root/'group_results.csv',index=False)
    trajectory.figures(sc,root);trajectory.component_figures(sc,root)
    candidate.aggregate(dest/'value_candidates')
    model=allframes['model_matched'].groupby(['subject','game','population'])[['case','maintain','difference']].mean().reset_index()
    model.to_csv(dest/'model_matched_subjects.csv',index=False)
    model.groupby(['game','population'])[['case','maintain','difference']].agg(['mean','sem','count']).to_csv(dest/'model_matched_games.csv')
    from panels.render import render04
    for game in range(3):
        t={'update_composition':allframes['composition'].loc[allframes['composition'].game.eq(game)],
           'matched_means':allframes['matched_means'].loc[allframes['matched_means'].game.eq(game)]}
        endpoint=pd.concat([pd.read_csv(OUTPUTS/'04/panels'/tag(s,game)/'tables/endpoint_sensitivity.csv').assign(subject=s) for s in range(1,7)],ignore_index=True)
        if len(endpoint) and {'M','observed','theory'}.issubset(endpoint):t['endpoint_sensitivity']=endpoint
        t['endpoint_sensitivity_qa']=pd.concat([pd.read_csv(OUTPUTS/'04/panels'/tag(s,game)/'tables/endpoint_sensitivity_qa.csv') for s in range(1,7)],ignore_index=True)
        root=dest/'search_composition'/f'game{game}';render04(t,root,game,[],6)
    return dict(bout_corrected_significant=int(sg.p_max_t.lt(.05).sum()),trajectory_corrected_significant=int(tg.p_max_t.lt(.05).sum()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=3)
    p.add_argument('--aggregate-only',action='store_true');a=p.parse_args()
    if a.aggregate_only:
        m=json.loads((a.output/'manifest.json').read_text());assert m['status']=='RUNNING','Preserve completed outputs'
    else:a.output.mkdir(parents=True,exist_ok=False)
    trajectory.check_history_contract();parts=[]
    write_json(a.output/'manifest.json',dict(status='RUNNING',matching_version=matching.VERSION))
    if a.aggregate_only:
        keys=('screen_runs','screen_coverage','trajectory_runs','trajectory_effects','trajectory_coverage','matching_qa','matched_means','composition','composition_coverage','model_matched')
        parts=[{k:pd.read_csv(a.output/f'{k}.csv.gz') for k in keys}]
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            futures={pool.submit(unit,s,g,a.output):(s,g) for s in range(1,7) for g in range(3)}
            for f in as_completed(futures):
                parts.append(f.result());print('Corrected',futures[f],flush=True)
    result=aggregate(a.output,parts)
    source=a.output/'source';source.mkdir(exist_ok=True)
    for module in (matching,data,screen,trajectory,candidate):shutil_copy(Path(module.__file__),source/Path(module.__file__).name)
    shutil_copy(Path(__file__),source/Path(__file__).name)
    write_json(a.output/'manifest.json',dict(status='COMPLETED',matching_version=matching.VERSION,**result,
        scope='Matching-dependent analyses only. Original caches, heatmaps, regression fits and model checkpoints unchanged.',
        artifacts={str(p.relative_to(a.output)):sha(p) for p in a.output.rglob('*') if p.is_file() and p.name!='manifest.json'}))


def shutil_copy(source,destination):destination.write_bytes(source.read_bytes())


if __name__=='__main__':main()
