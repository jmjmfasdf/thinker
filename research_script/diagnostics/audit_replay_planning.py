"""Read-only model/replay/cache audit; associations are descriptive, not causal.

Use each session only with its held-out checkpoint. Native event checks may
be restricted to a subject subset, and their scope is reported separately.
"""
from pathlib import Path
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import sys
import numpy as np
import pandas as pd
from scipy.special import expit, softmax
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import OUTPUTS, INPUT_CACHE, REPO, GAMES, tag
from common.io import sha
from common.outcomes import risk_set
from common.matching import corrected_pair_root


def err(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or not np.array_equal(np.isfinite(a), np.isfinite(b)):
        raise ValueError('Finite mask or shape mismatch')
    use = np.isfinite(a)
    return float(np.max(abs(a[use]-b[use]))) if use.any() else 0.


def rho(x, y):
    z = pd.DataFrame({'x': np.asarray(x), 'y': np.asarray(y)}).dropna()
    return z.x.rank().corr(z.y.rank()) if len(z) > 2 and z.x.nunique() > 1 and z.y.nunique() > 1 else np.nan


def native_checks(d, folder, a, cp, subject, game):
    pp = [f'actor_action_probability_{j}' for j in range(a)]
    cols = ['real_step_id', 'native_event', 'event_order', 'search_computation',
            'accepted_primary_action', 'current_node_depth_pre', 'policy_type',
            'is_control_choice', 'accepted_control', 'forced_stop', 'real_transition',
            'pre_observation_source_index', 'decision_time',
            'voc_soft_continue_probability', 'voc_ema_voc', 'voc_online_voc'] + pp
    cols += [f'root_{name}_{when}_{j}' for name in ('qs_mean','ns','policy')
             for when in ('pre','post') for j in range(a)]
    ev = pd.read_csv(folder/'native_events.csv.gz', usecols=cols)
    ev = ev.loc[ev.real_step_id.isin(d.real_step_id)].sort_values(['real_step_id','event_order'])
    ix = d.set_index('real_step_id')
    controls = ev.loc[ev.is_control_choice.eq(1)]
    first = controls.groupby('real_step_id').head(1).set_index('real_step_id').reindex(ix.index)
    last = controls.groupby('real_step_id').tail(1).set_index('real_step_id').reindex(ix.index)
    real = ev.loc[ev.real_transition.eq(1)].set_index('real_step_id').reindex(ix.index)
    se = ev.loc[ev.search_computation.eq(1)].copy()
    count = se.groupby('real_step_id').size().reindex(ix.index, fill_value=0)
    assert np.array_equal(count, ix.additional_computations)
    assert np.array_equal(first.native_event, ix.first_control_native_event)
    assert np.array_equal(last.native_event, ix.last_control_native_event)
    assert np.array_equal(real.native_event, ix.real_action_native_event)
    assert real.policy_type.eq(2).all()
    indexed = ev.loc[ev.is_control_choice.eq(1) | ev.real_transition.eq(1)]
    frame_error = int((indexed.pre_observation_source_index.to_numpy() !=
                       ix.loc[indexed.real_step_id,'current_frame'].to_numpy()).sum())
    time_error = err(indexed.decision_time, ix.loc[indexed.real_step_id,'decision_time'])
    prior = softmax(first[[f'root_policy_pre_{j}' for j in range(a)]].to_numpy(),axis=1)
    prior_error = err(prior, ix[[f'model_root_p_{j}' for j in range(a)]])
    real_error = err(real[pp], ix[pp])
    gate_error = err(expit(controls.voc_ema_voc/.01), controls.voc_soft_continue_probability)
    gate = controls.assign(bin=np.minimum((controls.voc_soft_continue_probability*10).astype(int),9),
                           observed=controls.accepted_control.ne(2).astype(int))
    bins = gate.groupby('bin').agg(n=('observed','size'), actual=('observed','mean'),
                                  predicted=('voc_soft_continue_probability','mean'))
    ece = float((abs(bins.actual-bins.predicted)*bins.n).sum()/bins.n.sum())
    argmax = se[pp].to_numpy().argmax(1)
    greedy_fraction = float(np.mean(argmax == se.accepted_primary_action)) if len(se) else np.nan
    pe = pd.read_csv(folder/'persistence_events.csv.gz')
    pe = pe.loc[pe.real_step_id.isin(ix.index)]
    se['status2_position'] = se.groupby('real_step_id').cumcount()
    joined = se.merge(pe, on=['real_step_id','status2_position'], validate='one_to_one')
    assert len(joined) == len(se)
    valid_branch = joined.branch.ge(0) & joined.branch.lt(a)
    jj = joined.loc[valid_branch]
    pos = np.arange(len(jj)); branch = jj.branch.to_numpy(int)
    q0 = jj[[f'root_qs_mean_pre_{j}' for j in range(a)]].to_numpy()[pos,branch]
    q1 = jj[[f'root_qs_mean_post_{j}' for j in range(a)]].to_numpy()[pos,branch]
    n0 = jj[[f'root_ns_pre_{j}' for j in range(a)]].to_numpy()[pos,branch]
    n1 = jj[[f'root_ns_post_{j}' for j in range(a)]].to_numpy()[pos,branch]
    repeated = (n0 > 0) & (n1 > 0)
    validity_error = int(np.sum(repeated != jj.value_revision_valid.eq(1)))
    revision_error = err(q1[repeated]-q0[repeated],jj.loc[repeated,'root_q_revision'])
    total = jj.assign(recomputed=np.where(repeated,abs(q1-q0),0)).groupby('real_step_id').recomputed.sum()
    u = total.reindex(ix.index,fill_value=0)/np.maximum(ix.additional_computations,1)
    u_error = err(u,ix.U_all)
    out = dict(subject=subject, game=game, checkpoint=cp, decisions=len(d), events=len(ev),
        computations=len(se), frame_mismatches=frame_error, decision_time_error=time_error,
        prior_probability_error=prior_error, real_probability_error=real_error,
        gate_probability_error=gate_error, gate_ece=ece, search_argmax_fraction=greedy_fraction,
        revision_validity_mismatches=validity_error, revision_error=revision_error,
        U_error=u_error, first_branch_evaluations=int((~repeated).sum()),
        repeated_branch_evaluations=int(repeated.sum()),
        valid_noop_updates=int((jj.value_revision_valid.eq(1)&jj.branch.eq(0)).sum()),
        noop_branch_events=int(jj.branch.eq(0).sum()))
    assert frame_error == 0 and time_error < 1e-8
    assert prior_error < 2e-6 and real_error < 2e-6
    assert validity_error == 0 and revision_error < 2e-6 and u_error < 2e-6
    return out


def unit(subject, game, native_subjects):
    a = 9 if game == 0 else 6
    cols = ['run_label','real_step','real_step_id','episode_id','session','checkpoint','held_out',
            'incumbent_action','human_action','scored','M','additional_computations',
            'ema_initial','U_all','valid_update_count','n_actions','current_frame','previous_frame',
            'first_control_native_event','last_control_native_event','real_action_native_event',
            'decision_time','root_carried','useful_carry']
    cols += [f'{prefix}_{j}' for prefix in ('model_root_p','actor_action_probability') for j in range(a)]
    ds = []; native = []; alignment = []; sources = {}; visual_checks = []
    for cp in range(1,5):
        root = OUTPUTS/'00/derived'/tag(subject,game)/f'leave{cp:02d}'
        p = root/'decisions.csv.gz'; meta=json.loads((root/'manifest.json').read_text())
        digest = sha(p); assert digest == meta['artifacts'][p.name]
        sources[str(p)] = digest
        d = pd.read_csv(p,usecols=cols);d=d.loc[d.session.eq(cp)].copy()
        assert d.held_out.eq(True).all() and d.checkpoint.eq(cp).all()
        cache = INPUT_CACHE/f'checkpoint_leave{cp:02d}'/tag(subject,game)
        manifest = json.loads((cache/'manifest.json').read_text())
        for name, item in manifest['source_runs'].items():
            r=d.loc[d.run_label.eq(item['run_label'])]
            if not len(r):continue
            p=REPO/'behavioral_data_block'/name
            digest=sha(p);assert digest==item['source_sha256'];sources[str(p)]=digest
            with np.load(p,allow_pickle=False) as z:
                raw=z['action']; action=raw.argmax(1) if raw.ndim==2 else raw
                curr=r.current_frame.to_numpy(int);prev=r.previous_frame.to_numpy(int)
                good=prev>=0
                mismatch=int(np.sum(action[curr]!=r.human_action))
                inc_mismatch=int(np.sum(action[prev[good]]!=r.loc[good,'incumbent_action']))
                gaps=np.diff(r.loc[r.episode_id.eq(r.episode_id.shift()),'decision_time'])
            assert mismatch==0 and inc_mismatch==0
            alignment.append(dict(subject=subject,game=game,checkpoint=cp,run_label=item['run_label'],
                 decisions=len(r),current_action_mismatches=mismatch,incumbent_mismatches=inc_mismatch))
        if subject in native_subjects:
            for name in ('native_events.csv.gz','persistence_events.csv.gz'):
                binding=json.loads((cache/'checkpoint_binding.json').read_text())
                digest=sha(cache/name);assert digest==binding['artifacts'][name]['sha256'];sources[str(cache/name)]=digest
            native.append(native_checks(d,cache,a,cp,subject,game))
        ds.append(d)
    d=pd.concat(ds,ignore_index=True)
    assert not d.real_step_id.duplicated().any()
    d=d.loc[d.scored.eq(1)&d.incumbent_action.ge(0)].copy()
    prob=d[[f'actor_action_probability_{j}' for j in range(a)]].to_numpy()
    assert np.isfinite(prob).all() and (prob>=0).all() and np.allclose(prob.sum(1),1,atol=1e-5)
    d['model_switch']=1-prob[np.arange(len(d)),d.incumbent_action.to_numpy(int)]
    d['model_noop']=prob[:,0]
    d['model_active']=1-prob[:,0]
    d['human_switch']=d.human_action.ne(d.incumbent_action).astype(int)
    metrics=['additional_computations','ema_initial','U_all','M']
    correlations=[];coverage=[];matched=[]
    pairs=pd.read_csv(corrected_pair_root(OUTPUTS,tag(subject,game))/'tables/matched_pairs.csv')
    for run,r in d.groupby('run_label'):
        coverage.append(dict(subject=subject,game=game,run_label=run,n=len(r),mean_budget=r.additional_computations.mean(),
            zero_budget=r.additional_computations.eq(0).mean(),no_repeated_update=r.valid_update_count.eq(0).mean(),
            no_repeated_update_when_searched=r.loc[r.additional_computations.gt(0),'valid_update_count'].eq(0).mean(),
            zero_U=r.U_all.eq(0).mean(),useful_carry=r.useful_carry.mean()))
    for pop,endpoint in [('overall','model_switch'),('noop_to_active','model_active'),('active_to_noop','model_noop')]:
        risk=risk_set(d,pop)
        for run,r in risk.groupby('run_label'):
            for metric in metrics:
                correlations.append(dict(subject=subject,game=game,population=pop,run_label=run,metric=metric,n=len(r),
                     model_rho=rho(r[metric],r[endpoint]),human_rho=rho(r[metric],r.outcome)))
            p=pairs.loc[pairs.population.eq(pop)&pairs.run_label.eq(run)]
            z=r.set_index('real_step_id')
            if len(p):
                sw=z.loc[p.switch_id];stay=z.loc[p.persist_id]
                matched.append(dict(subject=subject,game=game,population=pop,run_label=run,n=len(p),
                    model_switch_case=sw[endpoint].mean(),model_switch_control=stay[endpoint].mean(),
                    difference=sw[endpoint].mean()-stay[endpoint].mean(),
                    zero_visual_distance=p.visual_distance.eq(0).mean()))
    return dict(coverage=coverage,correlations=correlations,matched=matched,native=native,
                alignment=alignment,sources=sources)


def main():
    p=argparse.ArgumentParser();p.add_argument('--subjects',nargs='+',type=int,default=list(range(1,7)))
    p.add_argument('--native-subjects',nargs='*',type=int,default=[1]);p.add_argument('--workers',type=int,default=3)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    all_rows={k:[] for k in ('coverage','correlations','matched','native','alignment')};sources={}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks={pool.submit(unit,s,g,args.native_subjects):(s,g) for s in args.subjects for g in GAMES}
        for future in as_completed(tasks):
            value=future.result();sources.update(value.pop('sources'))
            for k,rows in value.items():all_rows[k].extend(rows)
            print('Audited',tasks[future],flush=True)
    for k,rows in all_rows.items():
        pd.DataFrame(rows).to_csv(args.output/f'{k}.csv',index=False)
    cov=pd.DataFrame(all_rows['coverage'])
    subj=cov.groupby(['subject','game']).mean(numeric_only=True).reset_index();subj.to_csv(args.output/'coverage_subjects.csv',index=False)
    subj.groupby('game').mean(numeric_only=True).to_csv(args.output/'coverage_games.csv')
    for name,values in [('correlations',['model_rho','human_rho']),('matched',['difference','model_switch_case','model_switch_control','zero_visual_distance'])]:
        frame=pd.DataFrame(all_rows[name]);keys=['subject','game','population']+(['metric'] if name=='correlations' else [])
        sub=frame.groupby(keys)[values].mean().reset_index();sub.to_csv(args.output/f'{name}_subjects.csv',index=False)
        sub.groupby(keys[1:])[values].agg(['mean','sem','count']).to_csv(args.output/f'{name}_games.csv')
    (args.output/'source.py').write_text(Path(__file__).read_text())
    (args.output/'manifest.json').write_text(json.dumps(dict(status='COMPLETED',subjects=args.subjects,
        native_subjects=args.native_subjects,sources=sources,
        aggregation='equal runs within subjects, equal subjects; no inferential p values',
        limits='Native Zarr captures and frozen capture runtime are absent locally. Numerical checks compare delivered numeric cache to derived metrics and source actions, not fresh checkpoint inference.'),indent=2))


if __name__=='__main__':main()
