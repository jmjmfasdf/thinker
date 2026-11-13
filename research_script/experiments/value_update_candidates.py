"""Exploratory value-update screen on held-out dynamic 10M decisions.

Candidate definitions and the complete comparison family are saved before
extraction. No human target is used to select an imagined branch or define a
feature. Reuse context-matched pairs, check fixed-budget subsets, and correct
all candidate/representation/cohort/game/outcome tests jointly over subjects.
This is a discovery analysis, not an independently confirmed result.
"""
from pathlib import Path
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import json
import sys
import time
import traceback
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.paths import OUTPUTS, INPUT_CACHE, GAMES, tag
from common.io import sha, signature, write_json
from panels.data import population
from common.matching import corrected_pair_root

METRICS={
    'U_all':('Absolute update / B','sum_j |ΔQ_j| / B'),
    'U_sum':('Absolute update sum','sum_j |ΔQ_j|'),
    'signed_sum':('Signed update sum','sum_j ΔQ_j'),
    'signed_mean':('Signed update / B','sum_j ΔQ_j / B'),
    'net_branch_abs_mean':('Net branch updating / B','sum_a |sum_{j:branch=a} ΔQ_j| / B'),
    'incumbent_down_mean':('Ongoing action devaluation / B','-sum_{j:branch=incumbent} ΔQ_j / B'),
    'alternatives_up_mean':('All alternatives updating / B','sum_{j:branch!=incumbent} ΔQ_j / B'),
    'fixed_alternative_up_mean':('Initial best alternative updating / B','sum_{j:branch=initial best alternative} ΔQ_j / B'),
    'fixed_advantage_mean':('Fixed alternative minus ongoing update / B','(sum ΔQ_fixed_alternative - sum ΔQ_incumbent) / B'),
    'all_advantage_mean':('All alternatives minus ongoing update / B','(sum ΔQ_other_actions - sum ΔQ_incumbent) / B'),
    'signed_balance':('Directed update balance','(sum ΔQ_other_actions - sum ΔQ_incumbent) / sum |ΔQ|; zero if no recorded update'),
    'unlikely_abs_mean':('Initially unlikely action updating / B','sum_j (1-initial_policy[branch_j]) |ΔQ_j| / B'),
    'event_Q_sum':('Sampled Q level sum','sum of valid post-computation sampled-branch Q levels; includes first evaluations'),
    'event_Q_mean':('Sampled Q level mean','same level sum / number of valid post-computation Q observations; empty=0'),
    'terminal_Q_sum':('Final visited Q level sum','sum final Q over visited root branches; empty=0'),
    'terminal_Q_mean':('Final visited Q level mean','mean final Q over visited root branches; empty=0'),
}
LEVELS={'event_Q_sum','event_Q_mean','terminal_Q_sum','terminal_Q_mean'}
POPS=('overall','noop_to_active','active_to_noop')
COHORTS=('all_matched','same_budget_searched','both_have_valid_updates')
REPS=('within_run_sd','within_run_percentile')
KEYS=['run_label','real_step','real_step_id']
MIN_PAIRS=20
MIN_RUNS=3


def protocol():
    return dict(status='EXPLORATORY; prior U results are already known',
        models='6 subjects × Enduro/Pong/Space Invaders × 4 leave-session checkpoints, 10M, tau .01, VOC-GRU connected',
        selection='Each human session contributes only its held-out checkpoint. Existing matching is reused without feature-based rematching.',
        candidate_definitions={k:dict(label=v[0],formula=v[1],kind='Q level, not an update' if k in LEVELS else 'recorded Q update') for k,v in METRICS.items()},
        update_validity='Same finite repeated sampled-branch revisions as U; first evaluations are not zero-to-Q updates. Invalid observed revisions contribute zero recorded update.',
        zero_budget='B=0: every recorded-update total/mean is zero. Unvisited Q entries are never treated as observed Q=0.',
        endpoint='Audit fixed-alternative endpoint ΔQ only when BOTH actions have positive initial and final visits; otherwise undefined, not zero. Coverage-only, outside candidate tests.',
        grouping='Equal matched-pair weight within run, equal eligible run weight within subject, equal subject weight; subject is the inferential unit.',
        representations=list(REPS),cohorts=list(COHORTS),populations=list(POPS),
        minimum_pairs_per_run=MIN_PAIRS,minimum_runs_per_subject=MIN_RUNS,
        scaling='Within-run SD or average-rank percentile, estimated from the whole eligible transition risk set. No centering is needed for paired differences; ties stay tied. Constant features give zero contrast.',
        correction='One two-sided synchronized exact max-|t| family: 16 candidates × 2 representations × 3 cohorts × 3 games × 3 outcomes = 864 tests. All 64 subject sign flips. Incomplete tests retained as untestable; no p value is reported for them.',
        consistency='For each candidate/representation/cohort, report the minimum number of agreeing subjects over all six game×bidirectional cells, and whether all six cell means share one sign. Also report Pong/SI separately.',
        limits='Association/discovery only. Fixed-budget and valid-update subsets are selected strata, not causal interventions. Percentile representation is a prespecified robustness analysis. No new training, replay, score or neural evaluation.')


def features_from_events(d, events):
    """Pure extraction, including branch identities fixed before human action."""
    d=d.copy().reset_index(drop=True); n=len(d); a=int(d.n_actions.iloc[0])
    loc=pd.Series(np.arange(n),index=d.real_step_id)
    if loc.index.has_duplicates:raise ValueError('Repeated decision ID')
    events=events.loc[events.real_step_id.isin(loc.index)].copy()
    rows=events.real_step_id.map(loc).to_numpy(int)
    branch=events.branch.fillna(-1).to_numpy(int)
    good_branch=(branch>=0)&(branch<a)
    valid=events.value_revision_valid.eq(1).to_numpy()&np.isfinite(events.root_q_revision)&good_branch
    q=events.root_q_revision.to_numpy(float)
    signed=np.zeros((n,a)); magnitude=np.zeros((n,a)); counts=np.zeros(n,int)
    np.add.at(signed,(rows[valid],branch[valid]),q[valid])
    np.add.at(magnitude,(rows[valid],branch[valid]),abs(q[valid]))
    np.add.at(counts,rows[valid],1)
    b=d.additional_computations.to_numpy(float); divisor=np.maximum(b,1)
    np.testing.assert_allclose(magnitude.sum(1),d.U_sum,rtol=1e-6,atol=1e-10)
    np.testing.assert_allclose(signed.sum(1),d.signed_update_sum,rtol=1e-6,atol=1e-10)
    np.testing.assert_array_equal(counts,d.valid_update_count)
    assert np.all(counts<=b)
    prior=d[[f'model_root_p_{j}' for j in range(a)]].to_numpy(float)
    inc=d.incumbent_action.to_numpy(int); ix=np.arange(n); safe=np.clip(inc,0,a-1)
    altprior=prior.copy(); altprior[ix,safe]=-np.inf; alt=altprior.argmax(1)
    own=signed[ix,safe]; alternatives=signed.sum(1)-own; fixed=signed[ix,alt]
    d['signed_sum']=signed.sum(1); d['signed_mean']=signed.sum(1)/divisor
    d['net_branch_abs_mean']=abs(signed).sum(1)/divisor
    d['incumbent_down_mean']=-own/divisor
    d['alternatives_up_mean']=alternatives/divisor
    d['fixed_alternative_up_mean']=fixed/divisor
    d['fixed_advantage_mean']=(fixed-own)/divisor
    d['all_advantage_mean']=(alternatives-own)/divisor
    d['signed_balance']=np.divide(alternatives-own,magnitude.sum(1),out=np.zeros(n),where=magnitude.sum(1)>0)
    d['unlikely_abs_mean']=((1-prior)*magnitude).sum(1)/divisor
    post=events.root_q_after_sampled.to_numpy(float)
    postvalid=good_branch&np.isfinite(post)&events.root_n_after_sampled.gt(0).to_numpy()
    levels=np.bincount(rows[postvalid],weights=post[postvalid],minlength=n)
    levelcounts=np.bincount(rows[postvalid],minlength=n)
    d['event_Q_sum']=levels
    d['event_Q_mean']=np.divide(levels,levelcounts,out=np.zeros(n),where=levelcounts>0)
    q0=d[[f'root_qs_mean_initial_{j}' for j in range(a)]].to_numpy(float)
    q1=d[[f'root_qs_mean_terminal_{j}' for j in range(a)]].to_numpy(float)
    n0=d[[f'root_ns_initial_{j}' for j in range(a)]].to_numpy(float)
    n1=d[[f'root_ns_terminal_{j}' for j in range(a)]].to_numpy(float)
    visited=(n1>0)&np.isfinite(q1)
    d['terminal_Q_sum']=np.where(visited,q1,0).sum(1)
    d['terminal_Q_mean']=np.divide(d.terminal_Q_sum,visited.sum(1),out=np.zeros(n),where=visited.sum(1)>0)
    common=(n0>0)&visited&np.isfinite(q0)
    endpoint_valid=common[ix,safe]&common[ix,alt]&(inc>=0)
    d['endpoint_D']=np.where(endpoint_valid,(q1[ix,alt]-q0[ix,alt])-(q1[ix,safe]-q0[ix,safe]),np.nan)
    d['endpoint_D_valid']=endpoint_valid
    d['initial_alternative']=alt; d['terminal_visited_branches']=visited.sum(1)
    for metric in set(METRICS)-LEVELS:
        assert d.loc[d.additional_computations.eq(0),metric].eq(0).all(),metric
    return d


def load_features(outputs, cache, subject, game):
    parts=[];sources={}
    for cp in range(1,5):
        root=Path(outputs)/'00/derived'/tag(subject,game)/f'leave{cp:02d}'
        meta=json.loads((root/'manifest.json').read_text()); path=root/'decisions.csv.gz'
        if sha(path)!=meta['artifacts']['decisions.csv.gz']:raise ValueError('Changed source '+str(path))
        d=pd.read_csv(path); d=d.loc[d.session.eq(cp)].copy()
        p=Path(cache)/f'checkpoint_leave{cp:02d}'/tag(subject,game)/'persistence_events.csv.gz'
        binding=json.loads((p.parent/'checkpoint_binding.json').read_text())
        digest=sha(p)
        if digest!=binding['artifacts'][p.name]['sha256']:raise ValueError('Changed events '+str(p))
        e=pd.read_csv(p,usecols=KEYS+['status2_position','branch','root_q_revision',
            'value_revision_valid','root_q_after_sampled','root_n_after_sampled'])
        if e.duplicated(KEYS+['status2_position']).any():raise ValueError('Duplicate imagination event')
        parts.append(features_from_events(d,e)); sources[str(path)]=sha(path); sources[str(p)]=digest
    d=pd.concat(parts,ignore_index=True)
    if d.duplicated(KEYS).any():raise ValueError('Duplicate held-out decision')
    return d,sources


def matched_analysis(d,pairs):
    rows=[];coverage=[]
    for pop in POPS:
        risk=population(d,pop)
        for run,r in risk.groupby('run_label'):
            r=r.set_index('real_step_id')
            p=pairs.loc[pairs.population.eq(pop)&pairs.run_label.eq(run)]
            if not len(p):continue
            sw=r.reindex(p.switch_id); st=r.reindex(p.persist_id)
            if sw.outcome.isna().any() or st.outcome.isna().any():raise ValueError('Matching outside risk set')
            assert sw.outcome.eq(1).all() and st.outcome.eq(0).all()
            b1=sw.additional_computations.to_numpy();b0=st.additional_computations.to_numpy()
            masks={'all_matched':np.ones(len(p),bool),
                'same_budget_searched':(b1==b0)&(b1>0),
                'both_have_valid_updates':sw.valid_update_count.gt(0).to_numpy()&st.valid_update_count.gt(0).to_numpy()}
            for cohort,mask in masks.items():
                coverage.append(dict(run_label=run,population=pop,cohort=cohort,n_pairs=int(mask.sum()),
                    n_switch_unique=int(p.loc[mask,'switch_id'].nunique()),n_persist_unique=int(p.loc[mask,'persist_id'].nunique()),
                    endpoint_coverage=float(r.endpoint_D_valid.mean()),risk_n=len(r),
                    zero_U_fraction=float(r.U_all.eq(0).mean())))
            for metric in METRICS:
                sd=float(r[metric].std(ddof=0)); constant=not sd>1e-12
                rank=r[metric].rank(method='average',pct=True)
                transformed={'within_run_sd':r[metric]/sd if not constant else r[metric]*0,
                    'within_run_percentile':rank}
                for representation,x in transformed.items():
                    v1=x.reindex(p.switch_id).to_numpy();v0=x.reindex(p.persist_id).to_numpy()
                    for cohort,mask in masks.items():
                        good=mask&np.isfinite(v1)&np.isfinite(v0); n=int(good.sum())
                        rows.append(dict(run_label=run,population=pop,cohort=cohort,metric=metric,
                            representation=representation,n_pairs=n,constant_feature=constant,
                            value=float(np.mean(v1[good]-v0[good])) if n>=MIN_PAIRS else np.nan,
                            switch=float(v1[good].mean()) if n>=MIN_PAIRS else np.nan,
                            persist=float(v0[good].mean()) if n>=MIN_PAIRS else np.nan))
    return pd.DataFrame(rows),pd.DataFrame(coverage)


def run_unit(outputs,cache,dest,subject,game):
    root=Path(dest)/tag(subject,game);root.mkdir(parents=True,exist_ok=True)
    marker=root/'manifest.json'
    identity=signature(dict(script=sha(__file__),protocol=protocol(),subject=subject,game=game))
    if marker.exists():
        m=json.loads(marker.read_text())
        if m.get('signature')==identity and m.get('status')=='COMPLETED' and all(sha(p)==h for p,h in m['sources'].items()) and all(sha(root/p)==h for p,h in m['artifacts'].items()):
            return dict(subject=subject,game=game,status='REUSED')
    write_json(marker,dict(status='RUNNING',signature=identity))
    d,sources=load_features(outputs,cache,subject,game)
    p=corrected_pair_root(outputs,tag(subject,game))/'tables/matched_pairs.csv'
    pairs=pd.read_csv(p);sources[str(p)]=sha(p)
    rows,coverage=matched_analysis(d,pairs)
    rows['subject']=subject;rows['game']=game
    coverage['subject']=subject;coverage['game']=game
    rows.to_csv(root/'run_contrasts.csv',index=False);coverage.to_csv(root/'coverage.csv',index=False)
    # Small source-independent feature table supports plots and inspection.
    keep=KEYS+['session','incumbent_action','human_action','scored','M','additional_computations','valid_update_count','endpoint_D','endpoint_D_valid','initial_alternative']+list(METRICS)
    d[keep].to_csv(root/'candidate_decisions.csv.gz',index=False)
    pd.DataFrame([dict(metric=c,nonzero_fraction=float(d[c].ne(0).mean()),finite_fraction=float(np.isfinite(d[c]).mean()),
        mean=float(d[c].mean()),sd=float(d[c].std()),q01=float(d[c].quantile(.01)),q99=float(d[c].quantile(.99))) for c in METRICS]).to_csv(root/'feature_summary.csv',index=False)
    write_json(marker,dict(status='COMPLETED',signature=identity,sources=sources,
        endpoint_valid_decisions=int(d.endpoint_D_valid.sum()),decisions=len(d),
        artifacts={p.name:sha(p) for p in root.iterdir() if p.name!='manifest.json'}))
    print(f'Value-update screen: {tag(subject,game)} complete',flush=True)
    return dict(subject=subject,game=game,status='COMPLETED')


def exact_family(subjects):
    keys=['game','population','cohort','metric','representation'];records=[];vectors=[]
    for key,part in subjects.groupby(keys,sort=True):
        v=part.set_index('subject').value.reindex(range(1,7)).to_numpy(float)
        good=np.isfinite(v); n=int(good.sum()); complete=n==6
        records.append(dict(zip(keys,key),n_subjects=n,mean=float(np.mean(v[good])) if n else np.nan,
            sem=float(np.std(v[good],ddof=1)/np.sqrt(n)) if n>1 else np.nan,
            positive_subjects=int((v[good]>0).sum()),negative_subjects=int((v[good]<0).sum()),
            status='ok' if complete else 'incomplete_subjects',p=np.nan,p_max_t=np.nan))
        vectors.append(v if complete else np.zeros(6))
    matrix=np.column_stack(vectors);signs=np.array(list(itertools.product([-1,1],repeat=6)))
    perm=signs[:,:,None]*matrix[None,:,:]
    null=np.abs(perm.mean(1))/np.maximum(perm.std(1,ddof=1)/np.sqrt(6),1e-12)
    observed=np.abs(matrix.mean(0))/np.maximum(matrix.std(0,ddof=1)/np.sqrt(6),1e-12)
    for j,r in enumerate(records):
        if r['status']=='ok':
            r['p']=float(np.mean(null[:,j]>=observed[j]-1e-12))
            r['p_max_t']=float(np.mean(null.max(1)>=observed[j]-1e-12))
    return pd.DataFrame(records)


def aggregate(dest):
    dest=Path(dest); allruns=[];coverage=[]
    for game in range(3):
        for subject in range(1,7):
            p=dest/tag(subject,game)
            m=json.loads((p/'manifest.json').read_text())
            if m['status']!='COMPLETED':raise ValueError('Incomplete '+str(p))
            for f,h in m['artifacts'].items():
                if sha(p/f)!=h:raise ValueError('Changed '+str(p/f))
            allruns.append(pd.read_csv(p/'run_contrasts.csv'));coverage.append(pd.read_csv(p/'coverage.csv'))
    runs=pd.concat(allruns,ignore_index=True)
    keys=['subject','game','population','cohort','metric','representation']
    subjects=runs.groupby(keys,as_index=False).agg(value=('value','mean'),n_runs=('value','count'),
        n_pairs=('n_pairs','sum'),switch=('switch','mean'),persist=('persist','mean'))
    subjects.loc[subjects.n_runs.lt(MIN_RUNS),['value','switch','persist']]=np.nan
    inference=exact_family(subjects)
    assert len(inference)==864,len(inference)
    ranks=[]
    for scope,games in [('all_three_games',[0,1,2]),('Pong_SI',[1,2])]:
        for (cohort,metric,representation),r in inference.loc[inference.population.ne('overall')&inference.game.isin(games)].groupby(['cohort','metric','representation']):
            complete=len(r)==len(games)*2 and r.status.eq('ok').all()
            allpos=bool(r['mean'].gt(0).all()) if complete else False
            allneg=bool(r['mean'].lt(0).all()) if complete else False
            orientation='positive' if allpos else ('negative' if allneg else 'mixed_or_missing')
            agree=r.positive_subjects if allpos else r.negative_subjects if allneg else pd.Series([0])
            ranks.append(dict(scope=scope,cohort=cohort,metric=metric,label=METRICS[metric][0],
                representation=representation,kind='Q_level' if metric in LEVELS else 'Q_update',
                complete=complete,direction=orientation,min_agreeing_subjects=int(agree.min()),
                mean_agreeing_subjects=float(agree.mean()),corrected_cells=int((r.p_max_t<.05).sum()),
                worst_corrected_p=float(r.p_max_t.max()),n_cells=len(r)))
    ranking=pd.DataFrame(ranks).sort_values(['scope','cohort','min_agreeing_subjects','mean_agreeing_subjects'],ascending=[True,True,False,False])
    tables=dict(run_contrasts=runs,subject_contrasts=subjects,inference=inference,
        consistency=ranking,coverage=pd.concat(coverage,ignore_index=True))
    for name,frame in tables.items():frame.to_csv(dest/f'{name}.csv',index=False)
    render(dest,subjects,inference,ranking)
    return tables


def render(dest,subjects,inference,ranking):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=plt.colormaps['RdBu_r'].copy();colors.set_bad('#eeeeee')
    # Every candidate is shown; no selective plot of the most promising one.
    for game in range(3):
        fig,axes=plt.subplots(2,3,figsize=(15,11),layout='constrained')
        for row,rep in enumerate(REPS):
            allvalues=subjects.loc[subjects.game.eq(game)&subjects.cohort.eq('all_matched')&subjects.representation.eq(rep),'value']
            limit=max(float(abs(allvalues).quantile(.98)),.001)
            for col,pop in enumerate(POPS):
                r=subjects.loc[subjects.game.eq(game)&subjects.population.eq(pop)&subjects.cohort.eq('all_matched')&subjects.representation.eq(rep)]
                mat=r.pivot(index='metric',columns='subject',values='value').reindex(index=list(METRICS),columns=range(1,7))
                im=axes[row,col].imshow(mat,aspect='auto',cmap=colors,vmin=-limit,vmax=limit)
                axes[row,col].set(xticks=range(6),xticklabels=[f'{s:03d}' for s in range(1,7)],title=pop.replace('_',' '),
                    yticks=range(len(METRICS)),yticklabels=[METRICS[c][0] for c in METRICS] if col==0 else [])
                axes[row,col].set_xlabel('Subject')
            fig.colorbar(im,ax=axes[row,:].tolist(),label='Switch − persistence ('+('run SD' if row==0 else 'run percentile')+')',extend='both')
        fig.suptitle(GAMES[game]+' · all candidate value measures\nExisting context-matched decisions; exploratory',fontsize=15)
        for ext in ['png','pdf']:fig.savefig(dest/f'candidate_consistency_game{game}.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    import html
    text=['<!doctype html><meta charset="utf-8"><title>Value-update candidate screen</title>',
        '<style>body{font:15px/1.5 system-ui;margin:30px;max-width:1500px}img{width:100%}table{border-collapse:collapse}td,th{padding:6px;border:1px solid #ccc}</style>',
        '<h1>Exploratory value-update candidate screen</h1><p>16 definitions; 864 tests corrected jointly using 64 synchronized subject sign flips. '
        'Three cohorts: original context-matched pairs, exactly matched positive computation budgets, and pairs with valid repeated updates in both members. '
        'No feature uses the target human action to choose a branch. Q levels are separated from Q updates. '
        'Colored pixels are participant contrasts, not independent significance tests. Raw-SD and rank representations are both shown.</p>']
    for game in range(3):text.append(f'<h2>{GAMES[game]}</h2><img src="candidate_consistency_game{game}.png">')
    text.append('<h2>All candidates: consistency across games and both directions</h2>')
    text.append(ranking.to_html(index=False,float_format=lambda x:f'{x:.3g}'))
    text.append('<p><a href="inference.csv">All corrected tests</a> · <a href="subject_contrasts.csv">Subject estimates</a> · <a href="coverage.csv">Coverage</a> · <a href="protocol.json">Definitions and analysis contract</a></p>')
    (dest/'index.html').write_text(''.join(text),encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outputs-root',type=Path,default=OUTPUTS);p.add_argument('--cache-root',type=Path,default=INPUT_CACHE)
    p.add_argument('--workers',type=int,default=3);p.add_argument('--aggregate-only',action='store_true')
    a=p.parse_args();dest=a.outputs_root/'02/value_update_candidates_20261111';dest.mkdir(parents=True,exist_ok=True)
    prot=dest/'protocol.json'
    if prot.exists() and json.loads(prot.read_text())!=protocol():raise ValueError('Preserve the previous protocol before changing the candidate family')
    if not prot.exists():write_json(prot,protocol())
    import shutil
    frozen=dest/('source_'+sha(__file__)[:12]+'.py')
    if not frozen.exists():shutil.copy2(__file__,frozen)
    state=dict(status='RUNNING',started_at=time.time(),source_sha256=sha(__file__),tasks={})
    write_json(dest/'progress.json',state)
    if not a.aggregate_only:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            futures={pool.submit(run_unit,a.outputs_root,a.cache_root,dest,s,g):(s,g) for s in range(1,7) for g in range(3)}
            for f in as_completed(futures):
                s,g=futures[f]
                try:state['tasks'][tag(s,g)]=f.result()
                except Exception:state['tasks'][tag(s,g)]=dict(status='FAILED',error=traceback.format_exc())
                write_json(dest/'progress.json',state)
        if any(v['status']=='FAILED' for v in state['tasks'].values()):
            state['status']='FAILED';write_json(dest/'progress.json',state);raise RuntimeError(state)
    tables=aggregate(dest)
    state.update(status='COMPLETED',completed_at=time.time(),tests=len(tables['inference']),
        artifacts={p.name:sha(p) for p in dest.iterdir() if p.is_file() and p.name not in ['progress.json'] and not p.suffix=='.log'})
    write_json(dest/'progress.json',state)
    print(tables['consistency'].query("cohort == 'all_matched' and representation == 'within_run_percentile'").to_string(index=False),flush=True)


if __name__=='__main__':main()
