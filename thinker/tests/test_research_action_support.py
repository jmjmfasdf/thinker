"""Prospective action meaning, leakage, sampling, inference and publication."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import numpy as np
import pandas as pd
import pytest

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'research_script'))
from analyses import action_support as action


def settings(**kwargs):
    return SimpleNamespace(**dict(dict(window=.5,horizon=.5,max_gap=.25,bootstrap=40,
        seed=701,penalty=.1,minimum_train=50),**kwargs))


def fixture(mode='all', actions=None, first=None, terminal=None, **kwargs):
    t=np.arange(181)/60
    a=np.zeros(len(t),int) if actions is None else np.asarray(actions)
    if actions is None:
        a[54:60]=1  # Switch and return between some sampled observations.
        a[102:114]=2
    first=np.r_[True,np.zeros(len(t)-1,bool)] if first is None else first
    terminal=np.zeros(len(t),bool) if terminal is None else terminal
    source=action.raw_behavior(t,a,first,terminal)
    frames=np.arange(len(t)) if mode=='all' else np.arange(0,len(t),4)
    d=pd.DataFrame(dict(subject=1,game='pong',run='r',episode='e',decision=frames,
        source_frame=frames,time=t[frames],human_action=a[frames],session=1,budget=2,carry=0,
        sample_index=np.arange(len(frames)),visual_change=.2,visual_0=t[frames],
        log_age=np.log1p(t[frames]),root_value=.1,prior_reward=0.))
    v=d[action.KEYS].copy()
    v['initial_voc']=.01;v['voc_A_V']=.2;v['policy_js']=.1
    for j in range(3):
        v[f'policy_logit_pre_{j}']=float(j)*.1
        v[f'policy_logit_post_{j}']=float(j)*.1+([-.2,.4,.1][j])
    metadata=dict(episodes=[dict(metadata=dict(episode_id='e',source_id='r',
        source_first_frame=0,source_logical_observation_count=len(t)))])
    spec=action.protocol(settings(**kwargs),mode)
    return d,v,metadata,{'r':source},spec


def windows(**kwargs):
    return action.make_windows(*fixture(**kwargs))


def test_first_destination_not_endpoint_and_no_overlap():
    w,c=windows()
    first=w.iloc[0]
    assert first.time==pytest.approx(.5)
    assert first.outcome==1 and first.first_switch_time==pytest.approx(.9)
    assert first.incumbent==0
    assert w.start_time.iloc[1:].gt(w.future_time.iloc[:-1].to_numpy()).all()
    assert c.candidates.sum()==c.drop(columns=['subject','game','episode','run','protocol_id','candidates']).sum().sum()


def test_future_labels_never_define_features_or_prediction():
    d,v,m,raw,spec=fixture()
    w,_=action.make_windows(d,v,m,raw,spec)
    a=raw['r']['action'].copy(); a[54:60]=2
    d2,v2,m2,raw2,spec2=fixture(actions=a)
    w2,_=action.make_windows(d2,v2,m2,raw2,spec2)
    columns=[c for c in w if c.startswith(('pre_','delta_','context_'))]
    np.testing.assert_allclose(w.iloc[0][columns].to_numpy(float),w2.iloc[0][columns].to_numpy(float))
    assert w.iloc[0].outcome==1 and w2.iloc[0].outcome==2
    for name,p in action.forecasts(w.iloc[:1])[2].items():
        np.testing.assert_allclose(p,action.forecasts(w2.iloc[:1])[2][name])


def test_all_and_15hz_same_landmark_same_raw_outcome():
    # Same physical cutoff, including a switch-return wholly between 15Hz rows.
    a=np.zeros(181,int); a[39:40]=2
    w,_=windows(actions=a,window=.4,horizon=.4)
    sampled,_=windows(mode='15hz',actions=a,window=.4,horizon=.4)
    assert w.time.iloc[0]==pytest.approx(sampled.time.iloc[0])
    assert w.outcome.iloc[0]==sampled.outcome.iloc[0]==2
    assert w.first_switch_frame.iloc[0]==sampled.first_switch_frame.iloc[0]==39


@pytest.mark.parametrize('kind',['first','terminal'])
def test_reset_terminal_censor_future_not_persistence(kind):
    marks=np.zeros(181,bool); marks[48 if kind=='first' else 47]=True
    w,c=windows(**{kind:marks})
    assert not np.isclose(w.time,.5).any()
    assert c.future_censored.sum()>0
    assert not ((w.start_time<.8)&(w.future_time>=.8)).any()


def test_missing_selected_observation_is_a_gap():
    for mode,frame in [('all',12),('15hz',12)]:
        d,v,m,raw,spec=fixture(mode=mode)
        d=d.loc[d.source_frame.ne(frame)];v=v.loc[v.decision.ne(frame)]
        w,c=action.make_windows(d,v,m,raw,spec)
        assert c.past_gap.sum()>0
        assert not np.isclose(w.time,.5 if mode=='all' else 8/15).any()


def test_censored_tail_and_noop_destination():
    a=np.ones(181,int);a[54:]=0
    w,c=windows(actions=a)
    assert w.outcome.iloc[0]==0 and w.switched.iloc[0]
    assert w.future_time.max()<=3
    assert c.future_censored.sum()>0


def test_raw_clock_gap_splits_segments():
    t=np.array([0.,.1,.2,.8,.9,0.,.1])
    r=action.raw_behavior(t,np.zeros(7),np.zeros(7),np.zeros(7))
    np.testing.assert_array_equal(r['segment'],[0,0,0,1,1,2,2])
    assert np.all(r['next_frame']==-1)


def test_missing_logits_or_binding_fail_closed():
    d,v,m,raw,spec=fixture()
    with pytest.raises(ValueError,match='schema'):
        action.make_windows(d,v.drop(columns='policy_logit_pre_1'),m,raw,spec)
    with pytest.raises(ValueError,match='Missing native'):
        action.make_windows(d,v.iloc[1:],m,raw,spec)
    v.loc[0,'policy_logit_pre_0']=np.nan
    with pytest.raises(ValueError,match='Nonfinite policy'):
        action.make_windows(d,v,m,raw,spec)


def test_time_weight_not_sum_of_computation_or_frame_count():
    all_w,_=windows(window=.4)
    sub_w,_=windows(mode='15hz',window=.4)
    for c in ['pre_0','pre_1','pre_2','delta_0','delta_1','delta_2']:
        assert all_w[c].iloc[0]==pytest.approx(sub_w[c].iloc[0])
    assert all_w.n_past_decisions.iloc[0]>sub_w.n_past_decisions.iloc[0]


def test_common_logit_shift_and_zero_update():
    d,v,m,raw,spec=fixture()
    w,_=action.make_windows(d,v,m,raw,spec)
    shifted=v.copy()
    for phase in ('pre','post'):
        shifted[action.logit_columns(shifted,phase)]+=np.arange(len(v))[:,None]*1000+17
    ws,_=action.make_windows(d,shifted,m,raw,spec)
    for key,p in action.forecasts(w)[2].items(): np.testing.assert_allclose(p,action.forecasts(ws)[2][key],atol=1e-10)
    for j in range(3): v[f'policy_logit_post_{j}']=v[f'policy_logit_pre_{j}']
    d['budget']=0
    wz,_=action.make_windows(d,v,m,raw,spec)
    before,change,forecasts=action.forecasts(wz)
    np.testing.assert_array_equal(change,np.zeros_like(change))
    np.testing.assert_array_equal(forecasts['pre'],forecasts['post'])


def synthetic_windows(n=240, opposite=False):
    rng=np.random.default_rng(921)
    w=pd.DataFrame(dict(subject=1,game='pong',run=[f'r{i//30}' for i in range(n)],episode='e',
        window_id=np.arange(n),session=1+np.arange(n)%4,time=np.arange(n)*1.1,
        incumbent=0,outcome=1+np.arange(n)%2,switched=True,n_actions=3,
        protocol_id='synthetic',sample_mode='all',context_visual_change=rng.normal(size=n)))
    change=rng.normal(0,.1,(n,3)); change[np.arange(n),w.outcome]+=2
    if opposite: change=-change
    change-=change[:,:1]
    for j in range(3):w[f'pre_{j}']=0.;w[f'delta_{j}']=change[:,j]
    return w


def test_literal_direction_sign_and_conditional_destination():
    for opposite in (False,True):
        w=synthetic_windows(opposite=opposite)
        parts=[action.prediction_table(w,p,k) for k,p in action.forecasts(w)[2].items()]
        scores=action.score_tables(pd.concat(parts),action.protocol(settings(),'all'))['action_support_scores']
        row=scores.loc[scores.population.eq('destination') & scores.model.eq('post')].iloc[0]
        assert (row.gain>0)==(not opposite)
        assert row.reference=='pre'


def test_action_label_permutation_equivariance():
    w=synthetic_windows(); before,change,p=action.forecasts(w)
    perm=np.array([2,0,1]); inverse=np.argsort(perm)
    wp=w.copy();wp['outcome']=inverse[w.outcome];wp['incumbent']=inverse[w.incumbent]
    for j in range(3): wp[f'pre_{j}']=before[:,perm[j]];wp[f'delta_{j}']=change[:,perm[j]]
    for name,q in p.items():
        a=action.prediction_table(w,q,name)
        b=action.prediction_table(wp,action.forecasts(wp)[2][name],name)
        np.testing.assert_allclose(a.log_score,b.log_score)
        np.testing.assert_allclose(a.destination_log_score,b.destination_log_score)
        np.testing.assert_allclose(a.accuracy,b.accuracy)
        np.testing.assert_allclose(a.destination_accuracy,b.destination_accuracy)


def test_extreme_logits_do_not_clip_conditional_destination():
    w=synthetic_windows(8)
    logits=np.tile([10000.,-10000.,-9999.],(8,1))
    q=action.prediction_table(w,action.log_probabilities(logits),'pre')
    assert np.isfinite(q.destination_log_score).all()
    assert q.destination_log_score.min()>-2


def test_conditional_gain_is_not_mislabeled_as_reinforcement_over_staying():
    w=synthetic_windows(8);w['outcome']=1
    w['delta_0']=0.;w['delta_1']=-1.;w['delta_2']=-3.
    p=action.forecasts(w)[2]
    before=action.prediction_table(w,p['pre'],'pre')
    after=action.prediction_table(w,p['post'],'post')
    assert after.destination_log_score.gt(before.destination_log_score).all()
    assert after.destination_relative_update.lt(0).all()
    assert after.log_score.lt(before.log_score).all()


@pytest.mark.parametrize('signed',[False,True])
def test_multinomial_gradient(signed):
    rng=np.random.default_rng(4)
    x=rng.normal(size=(7,3));before=rng.normal(size=(7,3));change=rng.normal(size=(7,3));y=np.arange(7)%3
    beta=rng.normal(0,.2,3*3+1+int(signed))
    args=(x,before,np.abs(change),change,y,.1,signed)
    _,gradient=action._objective(beta,*args)
    numerical=[]
    for j in range(len(beta)):
        delta=np.zeros_like(beta);delta[j]=1e-6
        numerical.append((action._objective(beta+delta,*args)[0]-action._objective(beta-delta,*args)[0])/2e-6)
    np.testing.assert_allclose(gradient,numerical,atol=1e-7)


def test_heldout_pair_support_and_train_only_preprocessing(monkeypatch):
    w=synthetic_windows();before,change,_=action.forecasts(w)
    original=action.prepare_matrix;calls=[]
    def check(frame,columns,train):
        assert frame.loc[~train,'session'].nunique()==1
        assert not set(frame.loc[train,'session'])&set(frame.loc[~train,'session'])
        assert all(c.startswith('context_') for c in columns)
        calls.append(train.copy())
        return original(frame,columns,train)
    monkeypatch.setattr(action,'prepare_matrix',check)
    p,qa=action.heldout_readouts(w,before,change,action.protocol(settings(),'all'))
    assert len(calls)==4 and qa.status.eq('ok').all()
    for _,g in p.groupby('held_session'):
        assert set(g.loc[g.model.eq('signed'),'window_id'])==set(g.loc[g.model.eq('unsigned'),'window_id'])
        assert g.session.nunique()==1
    assert qa.loc[qa.model.eq('signed'),'direction_coefficient'].gt(0).all()


def test_insufficient_folds_not_zero_effect():
    w=synthetic_windows(8);before,change,_=action.forecasts(w)
    p,qa=action.heldout_readouts(w,before,change,action.protocol(settings(),'all'))
    assert p.empty and qa.status.eq('insufficient_rows').all()


def test_held_targets_do_not_change_own_fold_fit_or_predictions():
    w=synthetic_windows();before,change,_=action.forecasts(w)
    spec=action.protocol(settings(),'all')
    p,audit=action.heldout_readouts(w,before,change,spec)
    w2=w.copy();w2.loc[w2.session.eq(1),'outcome']=2
    p2,audit2=action.heldout_readouts(w2,before,change,spec)
    a=audit.loc[audit.held_session.eq(1)&audit.model.eq('signed'),'direction_coefficient'].iloc[0]
    b=audit2.loc[audit2.held_session.eq(1)&audit2.model.eq('signed'),'direction_coefficient'].iloc[0]
    assert a==b
    columns=[f'probability_{j}' for j in range(3)]
    np.testing.assert_array_equal(p.loc[p.held_session.eq(1),columns],p2.loc[p2.held_session.eq(1),columns])


def test_direction_coefficient_is_not_forced_positive():
    w=synthetic_windows(opposite=True);before,change,_=action.forecasts(w)
    _,qa=action.heldout_readouts(w,before,change,action.protocol(settings(),'all'))
    assert qa.status.eq('ok').all()
    assert qa.loc[qa.model.eq('signed'),'direction_coefficient'].lt(0).all()


def test_equal_run_session_subject_group_and_protocol_guard():
    spec=action.protocol(settings(),'all')
    protocols=pd.DataFrame([dict(spec,subject=s,game='pong') for s in (1,2)])
    rows=[]
    for sub,ses,run,gain in [(1,1,'a',0.),(1,1,'b',2.),(1,2,'c',5.),(2,1,'d',9.)]:
        rows.append(dict(subject=sub,game='pong',protocol_id=spec['protocol_id'],sample_mode='all',
            population='destination',model='post',reference='pre',session=ses,run=run,
            gain=gain,accuracy_gain=0.,log_score=0.,reference_log_score=0.,accuracy=.5,reference_accuracy=.5))
    tables=dict(action_support_protocol=protocols,action_support_run_scores=pd.DataFrame(rows))
    result=action.aggregate(tables,[1,2])
    np.testing.assert_allclose(result['action_support_participant_scores'].gain,[3.,9.])
    assert result['action_support_group_scores'].gain.iloc[0]==6.
    protocols.loc[1,'protocol_id']='other'
    with pytest.raises(ValueError,match='Mixed'):action.aggregate(tables,[1,2])
    with pytest.raises(ValueError,match='every requested'):action.aggregate(tables,[1,2,3])


def test_saved_table_render_paper_style_and_no_inputs(monkeypatch,tmp_path):
    from matplotlib.figure import Figure
    w=synthetic_windows();parts=[action.prediction_table(w,p,k) for k,p in action.forecasts(w)[2].items()]
    tables=action.score_tables(pd.concat(parts),action.protocol(settings(),'all'))
    paths=[]
    def save(fig,path,**kwargs):
        assert kwargs['dpi']==300
        assert all(not ax.get_title() for ax in fig.axes)
        assert fig._suptitle is None or not fig._suptitle.get_text()
        assert all(ax.xaxis.label.get_fontsize()==26 for ax in fig.axes)
        paths.append(path)
    monkeypatch.setattr(Figure,'savefig',save)
    monkeypatch.setattr(action,'load_raw',lambda *a:pytest.fail('render read source'))
    action.render(tmp_path,tables)
    assert len(paths)==4


def test_archive_keeps_previous_09_and_is_idempotent(tmp_path):
    from common.result_store import ResultStore
    from common.research_cli import _archive_09
    root=tmp_path/'09'/'sub001'/'pong'
    with ResultStore(root,{'sample_mode':'all'}) as store:
        with store.stage('prepare','p'):pass
        with store.stage('analyze','a') as stage:stage.write_table('old',pd.DataFrame({'value':[7]}))
        store.publish(required_stages=('prepare','analyze'))
    archived=_archive_09(root)
    assert archived.is_relative_to(tmp_path/'.archive'/'09')
    assert archived==_archive_09(root)
    assert (archived/'tables/old.csv.gz').read_bytes()==(root/'tables/old.csv.gz').read_bytes()


def test_signature_covers_new_numeric_module_and_render_separately(monkeypatch):
    from common import research_cli as cli
    from analyses import temporal_pipeline,group_results
    original=cli.code_signature
    old=[original(module) for module in (temporal_pipeline,group_results)]
    def change(module,render=False):
        if module is action and not render:return 'different-numeric-code'
        return original(module,render)
    monkeypatch.setattr(cli,'code_signature',change)
    assert all(a!=cli.code_signature(m) for a,m in zip(old,(temporal_pipeline,group_results)))


def test_temporal_integration_keeps_legacy_and_uses_unfiltered_native(monkeypatch,tmp_path):
    from analyses import temporal_pipeline as temporal
    d,v,m,raw,spec=fixture()
    prepared=dict(decisions=d,metadata=m,voc_tables={'voc_decisions':v},sample_mode='all')
    monkeypatch.setattr(temporal.accumulation,'validate_capture',lambda *a:{})
    monkeypatch.setattr(temporal.accumulation,'extract',lambda *a:(d.iloc[:2].copy(),{'actor_p':np.full((2,3),1/3)}))
    monkeypatch.setattr(temporal.accumulation,'make_windows',lambda *a:(pd.DataFrame(),{}))
    monkeypatch.setattr(temporal,'_load_raw',lambda *a:{})
    monkeypatch.setattr(temporal,'_primary_examples',lambda *a:{})
    monkeypatch.setattr(temporal,'_supplement',lambda *a:({'legacy_test':pd.DataFrame({'value':[7.]})},{}))
    def check(args,native):
        assert len(native['decisions'])==181  # Not the legacy's two supported rows.
        return {'action_support_test':pd.DataFrame({'value':[9.]})},dict(status='COMPLETED',primary_supported=True,protocol=spec)
    monkeypatch.setattr(action,'analyze',check)
    tables,qa,arrays=temporal.analyze(settings(),1,'pong',tmp_path,prepared)
    assert tables['legacy_test'].value.iloc[0]==7
    assert tables['action_support_test'].value.iloc[0]==9
    assert qa['primary_supported'] and not qa['legacy_primary_supported']
    assert qa['protocol']==spec and arrays=={}


def test_group_pipeline_reads_saved_tables_only(monkeypatch,tmp_path):
    from common.result_store import ResultStore
    from analyses import group_results
    spec=action.protocol(settings(),'all'); units=[]
    for subject in (1,2):
        root=tmp_path/'09'/f'sub{subject:03d}'/'pong'
        w=synthetic_windows();w['subject']=subject;w['protocol_id']=spec['protocol_id']
        parts=[action.prediction_table(w,p,k) for k,p in action.forecasts(w)[2].items()]
        tables=action.score_tables(pd.concat(parts),spec)
        tables['action_support_protocol']=pd.DataFrame([dict(spec,subject=subject,game='pong')])
        tables['action_support_windows']=w
        with ResultStore(root,{'sample_mode':'all'}) as store:
            with store.stage('prepare','p'):pass
            with store.stage('analyze','a') as stage:
                for name,frame in tables.items():stage.write_table(name,frame)
            store.publish(required_stages=('prepare','analyze'))
        units.append((subject,'pong',root))
    monkeypatch.setattr(action,'load_raw',lambda *a:pytest.fail('Group loaded raw sources'))
    tables,qa,_=group_results.aggregate('09',SimpleNamespace(subjects=[1,2]),units)
    assert 'action_support_windows' not in tables
    assert tables['action_support_group_scores'].n_subjects.eq(2).all()
    assert qa['action_support']['primary'].startswith('Conditional')
