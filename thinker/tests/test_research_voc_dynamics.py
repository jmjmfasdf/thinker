"""Timing, provenance, leakage and observational semantics for additive VoC."""
import os
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pytest

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'research_script'))
from common import voc_dynamics as native
from analyses import voc_relationships as relationships


def test_real_phase_logits_and_legacy_readout_contract():
    from scipy.special import softmax
    head=native.Readout.__new__(native.Readout)
    head.phase=np.array([[0.,0.],[.2,-.1]])
    head.clip=100.
    head.state={'baseline.weight':np.array([[.3,.1]]),'baseline.bias':np.zeros(1),
        'policy.weight':np.array([[.1,.2],[-.3,.4],[.2,-.1]]),'policy.bias':np.zeros(3),
        'voc_head.weight':np.array([[.1,.2],[.2,.3]]),'voc_head.bias':np.zeros(2)}
    head.ema={'weight':head.state['voc_head.weight'],'bias':np.zeros(2)}
    phases=np.array([0,1]);z=np.array([[1.,2.],[1.,2.]])+head.phase[phases]
    q=head.linear(z,'voc_head')
    fields=dict(voc_features=z,phase_pre=phases,actor_baseline=head.linear(z,'baseline'),
        voc_soft_continue_probability=np.full(2,.5),voc_state_value=np.zeros(2),voc_reconstruction_mode=np.zeros(2),
        voc_online_raw_q=q,voc_online_q=q,voc_ema_raw_q=q,voc_ema_q=q,
        actor_action_probability=softmax(head.linear(z,'policy'),axis=1))
    class Trace:
        def read(self,key,rows):return fields[key][rows]
        def valid(self,key):return np.array([False,True])
    legacy=head.evaluate(Trace(),np.arange(2))
    extended=head.evaluate(Trace(),np.arange(2),with_policy_logits=True)
    assert len(legacy)==4 and len(extended)==5
    for old,new in zip(legacy,extended[:4]):np.testing.assert_array_equal(old,new)
    np.testing.assert_allclose(extended[-1][0],extended[-1][1],atol=1e-15)
    np.testing.assert_allclose(softmax(extended[-1],axis=1),legacy[1])
    fields['actor_action_probability'][1,0]=np.nan
    with pytest.raises(ValueError,match='native REAL policy'):head.evaluate(Trace(),np.arange(2))


@pytest.mark.parametrize('n',[0,1,4])
def test_endpoint_summary_first_cost_and_zero(n):
    v=np.arange(n+1,dtype=float)
    result=native.summary(v,np.tile([.25,.75],(n+1,1)),v/10,1,[.0001]*n)
    assert result['voc_A_V']==n
    assert result['voc_U_V']==int(n>0)
    assert result['voc_D_V']==n
    assert result['computation_cost']==pytest.approx(n*.0001)
    assert result['human_logp_change']==0


def test_no_q_support_is_not_zero():
    result,count=native.supported_q_change(np.array([0.]),np.array([9.]),np.array([0]),np.array([1]))
    assert np.isnan(result) and count==0
    result,count=native.supported_q_change(np.array([9.]),np.array([9.]),np.array([1]),np.array([2]))
    assert result==0 and count==1


@pytest.mark.parametrize('cost',[[-1.],[np.nan]])
def test_invalid_cost(cost):
    with pytest.raises(ValueError):native.summary(np.array([0.,1.]),np.tile([.5,.5],(2,1)),np.zeros(2),0,cost)


def synthetic_capture(monkeypatch,n=2,capped=False,reset=False):
    right=n+1 if capped else n+2; count=right+1
    flags={k:np.zeros(count,dtype=int) for k in native.FLAGS}
    flags['initial_root'][0]=1;flags['real_transition'][[0,right]]=1
    # decision_edges treats every REAL row as an output; use initial root only.
    flags['real_transition'][0]=0
    flags['search_computation'][1:n+1]=1
    flags['current_node_depth_post'][1:n+1]=np.arange(1,n+1)
    flags['current_node_depth_pre'][1:n+1]=np.arange(n)
    flags['search_steps_pre'][1:]=np.minimum(np.arange(right),n)
    flags['phase_pre'][[0,right]]=1
    flags['pre_observation_source_index'][1:]=1
    flags['control_valid'][1:right]=1
    if not capped:flags['accepted_control'][n+1]=2
    if capped:flags['forced_stop'][n]=1
    if reset and n>1:flags['accepted_control'][2]=1;flags['actor_reset'][2]=1
    tree=np.zeros((count,4));tree[:,2:]=1;tree[1,2]=0
    post=tree.copy();post[:,2:]=1;post[:,0]=.2
    fields=dict(flags,voc_online_raw_q=np.zeros((count,2)),voc_ema_raw_q=np.zeros((count,2)),
        voc_online_q=np.column_stack([np.arange(count)*.1,np.zeros(count)]),
        voc_ema_q=np.column_stack([np.arange(count)*.1,np.zeros(count)]),
        voc_state_value=np.zeros(count),voc_soft_continue_probability=np.full(count,.8),
        voc_reconstruction_mode=np.zeros(count,dtype=int),voc_effective_think_cost=np.full(count,.001),
        voc_ema_available=np.ones(count,bool),tree_reps_pre=tree,tree_reps_post=post,
        actor_baseline=np.arange(count)[:,None]*.1,voc_features=np.zeros((count,2)),
        actor_action_probability=np.tile([.7,.3],(count,1)))
    meta={'episode_id':'e'}
    files_sha=native.hashlib.sha256(b'[]').hexdigest()
    class Trace:
        def __init__(self,path):
            self.meta=meta;self.manifest={'files_sha256':files_sha,'files':[]}
            self.tree_schema={'root_qs_mean':[0,2,None],'root_ns':[2,4,None]}
        def scalar(self,name):return fields[name]
        def read(self,name,rows=None):return fields[name] if rows is None else fields[name][rows]
        def valid(self,name):return np.ones(count,bool)
    class Head:
        def __init__(self,*args):pass
        def evaluate(self,t,rows,*,with_policy_logits=False):
            result=rows*.1,np.tile([.7,.3],(len(rows),1)),rows*.05,0.
            return (*result,np.log(result[1])) if with_policy_logits else result
    monkeypatch.setattr(native,'Trace',Trace);monkeypatch.setattr(native,'Readout',Head)
    monkeypatch.setattr(native,'verify_chunks',lambda *a:None)
    # Root j=0 in this synthetic trace; a single scored planning interval.
    row=dict(subject=1,game='pong',run='r',episode='e',decision=0,session=1,time=.1,
        M=.2,root_value=0.,age=.1,prior_reward=0.,carry=0,root_event=0,real_event=right,
        budget=n,source_frame=1,human_action=1,switch=1,V_total=n*.1,V_initial=.1,V_final=(n+1)*.1)
    prepared=dict(decisions=pd.DataFrame([row]),metadata=dict(capture='unused',episodes=[dict(path='ep',metadata=meta,files_sha256=files_sha)]))
    cfg=dict(min_search_steps=min(n,1),max_search_steps=max(n,1),think_cost=.001,
        discounting=.99,reward_clip=1.,voc_gate_q_temperature=.01,voc_gate_exact_projection=True)
    return native.extract(prepared,None,cfg,chunk_size=2)


@pytest.mark.parametrize('n,capped,reset',[(0,False,False),(1,False,False),(1,True,False),(3,False,True),(3,True,False)])
def test_native_timing_and_cap(monkeypatch,n,capped,reset):
    tables,qa=synthetic_capture(monkeypatch,n,capped,reset)
    d=tables['voc_decisions'].iloc[0];c=tables['voc_choices'];tr=tables['voc_transitions']
    assert len(tr)==n and len(tables['voc_endpoints'])==n+1
    assert d.policy_logit_pre_0==pytest.approx(np.log(.7))
    assert d.policy_logit_post_1==pytest.approx(np.log(.3))
    assert c.continued.sum()==n
    assert (c.accepted_control==2).sum()==int(not capped)
    assert d.termination==('cap' if capped else 'voluntary_stop')
    if n:
        assert np.isnan(tr.iloc[0].abs_delta_q)  # another visited action is not coverage
        assert tr.iloc[0].delta_v==pytest.approx(.1)
        if capped:
            assert not tr.iloc[-1].next_choice_available
            assert np.isnan(tr.iloc[-1].next_continue)
            assert d.last_choice_voc==pytest.approx(n*.1)
            assert d.endpoint_voc_final==pytest.approx((n+1)*.05)
        else:
            assert tr.iloc[-1].next_continue==0
        if n>1:assert tr.iloc[1].abs_delta_q==pytest.approx(.2)
    if reset:assert 1 in set(c.accepted_control)


def test_unique_keys_fail_closed():
    with pytest.raises(ValueError):native.unique(pd.DataFrame({'event':[1,1]}),['event'])


def test_trailing_excludes_current_future_and_gaps():
    d=pd.DataFrame(dict(episode=['a']*5+['b'],decision=[0,1,2,3,4,0],
        time=[0,.1,.2,.8,.9,1.],initial_voc=[1.,2.,999.,4.,5.,6.],
        endpoint_voc_change=np.zeros(6),U_V=np.ones(6)))
    r=relationships.trailing_features(d)
    assert r.loc[2,'past_voc_mean']==pytest.approx(1.5)
    assert r.loc[2,'lag1_initial_voc']==2
    assert np.isnan(r.loc[3,'lag1_initial_voc'])
    assert np.isnan(r.loc[5,'lag1_initial_voc'])
    d['prior_raw_bout_id']=[0,0,1,1,1,1]
    assert np.isnan(relationships.trailing_features(d).loc[2,'lag1_initial_voc'])


def test_window_uses_input_only_and_rejects_gap():
    d=pd.DataFrame(dict(episode=['a']*5,decision=np.arange(5),time=np.arange(5)*.1))
    v=d[['episode','decision']].assign(initial_voc=[1.,2.,3.,999.,-999.],endpoint_voc_change=0.)
    w=pd.DataFrame([dict(window_id=0,start=0,end=2,future=4)])
    result=relationships.window_features(d,w,v).iloc[0]
    assert result.voc_window_mean==pytest.approx(1.5)
    assert result.voc_window_slope==pytest.approx(10.)
    d.loc[1,'decision']=100
    assert not relationships.window_features(d,w,v).iloc[0].voc_window_supported


def test_equal_runs_not_event_weighting():
    d=pd.DataFrame(dict(subject=[1,1,2],run=['a','b','c'],bin=[0]*3,value=[0.,2.,9.],n=[10000,1,1]))
    p=relationships.equal_runs(d,['bin'],['value'])
    assert p.loc[p.subject.eq(1),'value'].iloc[0]==1
    assert p.value.mean()==5


def test_prediction_rejects_target_derived_inputs():
    with pytest.raises(ValueError):relationships.readouts(pd.DataFrame(),{'full':['human_logp_change']})


def test_matched_readouts_and_train_only_indices(monkeypatch):
    d=pd.DataFrame(dict(episode=['e']*12,run=['r1']*6+['r2']*6,decision=np.arange(12),
        session=[1]*6+[2]*6,incumbent_action=0,human_action=[0,1]*6,switch=[0,1]*6,n_actions=2,x=np.arange(12)))
    calls=[]
    def predict(frame,cols,y,train,test,**kwargs):
        assert not np.any(train&test);assert frame.loc[test,'session'].nunique()==1
        calls.append((cols,train.copy(),test.copy()))
        return np.full((test.sum(),2),.5),{'status':'ok'}
    monkeypatch.setattr(relationships,'probabilities',predict)
    tables=relationships.readouts(d,{'reference':['x'],'full':['x'],'no_voc':['x']})
    assert len(calls)==12
    assert tables['voc_readout_gains'].value.eq(0).all()
    for i in range(0,len(calls),3):
        for j in (1,2):np.testing.assert_array_equal(calls[i][2],calls[i+j][2])


def test_paper_style_and_no_source_reads(tmp_path,monkeypatch):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    seen=[]
    def capture(fig,*args,**kwargs):
        assert all(ax.get_title()=='' and ax.xaxis.label.get_fontsize()==26 for ax in fig.axes)
        assert fig._suptitle is None or not fig._suptitle.get_text()
        assert kwargs['dpi']==300
        seen.append(str(args[0]))
    monkeypatch.setattr(Figure,'savefig',capture)
    with relationships.render_style():
        fig,ax=plt.subplots();ax.set_title('remove');fig.suptitle('remove');fig.savefig(tmp_path/'x.png');plt.close(fig)
    monkeypatch.setattr(native,'prepare',lambda *a:pytest.fail('Renderer read original inputs'))
    frame=pd.DataFrame([dict(subject=1,game='pong',run='r',x_metric='voc_ema',metric='abs_delta_v',bin=0,x=.1,value=.2,n=5)])
    relationships.render(tmp_path,{'voc_relationship_curves':frame})
    assert len(seen)==3


def test_number_dispatch_preserves_legacy_and_excludes_utility(monkeypatch,tmp_path):
    from common import research_cli as cli
    legacy={'old':pd.DataFrame({'value':[7]})}
    module=SimpleNamespace(analyze=lambda *a:(dict(legacy),{'status':'COMPLETED'},{}))
    monkeypatch.setattr(native,'prepare',lambda *a:({'voc_decisions':pd.DataFrame()},{}))
    monkeypatch.setattr(relationships,'extend',lambda *a:({'voc_new':pd.DataFrame({'value':[8]})},{'status':'COMPLETED'}))
    got,qa,_=cli.analyze('03',module,None,1,'pong',{},tmp_path)
    assert got['old'].value.iloc[0]==7 and 'voc_new' in got
    got,_,_=cli.analyze('04',module,None,1,'pong',{},tmp_path)
    assert list(got)==['old']


@pytest.mark.parametrize('number',['03','05','06'])
@pytest.mark.parametrize('n',[0,1,3])
def test_additive_analysis_with_insufficient_support(monkeypatch,number,n):
    tables,_=synthetic_capture(monkeypatch,n=n,capped=n>0)
    result,qa=relationships.extend(number,SimpleNamespace(),{}, {},tables)
    assert result and qa['status'] in ('COMPLETED','PARTIAL_FITS')
    assert all(name.startswith('voc_') for name in result)
    if number=='05':
        fits=result['voc_relationship_effects']
        assert not fits.status.eq('ok').any()


def test_actual_held_session_feature_ablation():
    rng=np.random.default_rng(17);n=450
    d=pd.DataFrame(dict(episode=np.repeat(['e1','e2','e3'],150),run=np.repeat(['r1','r2','r3'],150),
        session=np.repeat([1,2,3],150),decision=np.tile(np.arange(150),3),incumbent_action=0,
        human_action=rng.integers(0,2,n),switch=rng.integers(0,2,n),n_actions=2))
    names=['log_age','log_age_sq','recent10','recent30','prior_reward','prior_done','run_progress','run_progress_sq',
        'M','root_value','V_initial','root_entropy','initial_q_margin','log_budget','searched','carried',
        *relationships.VOC_FEATURES,*relationships.REVISION_FEATURES,*relationships.POLICY_FEATURES]
    for name in names:d[name]=rng.normal(size=n)
    tables=relationships.readouts(d,relationships.prediction_blocks(d))
    assert tables['voc_readout_fit_qa'].status.eq('ok').all()
    assert len(tables['voc_readout_gains'])==24
    assert set(tables['voc_readout_gains'].reference)=={'reference','no_voc','no_revision','no_policy'}


def test_temporal_new_models_share_supported_train_and_test(monkeypatch,tmp_path):
    from analyses import evaluation_accumulation as temporal
    w=pd.DataFrame(dict(window_id=np.arange(12),session=np.repeat([1,2,3],4),run=np.repeat(['a','b','c'],4),
        episode='e',time=np.arange(12),start=np.arange(12),middle=np.arange(12),end=np.arange(12),future=np.arange(12)+1,
        nonoverlap=True,first_action_constant=True,human_switch=np.tile([0,1],6),policy_js=.1,policy_away=.2))
    features=np.ones((12,3));features[2]=np.nan
    def folded(*args):
        return {name:np.ones((12,2)) for name in ['voc_full','voc_no_voc','voc_no_direction']},np.zeros((12,5)),None,{}
    calls=[]
    def predict(x,y,train,test,*args):
        calls.append((train.copy(),test.copy()))
        return np.full(len(test),.5),'ok',np.ones(x.shape[1])
    monkeypatch.setattr(temporal,'fold_features',folded);monkeypatch.setattr(temporal,'predict_model',predict)
    monkeypatch.setattr(temporal,'block_interval',lambda *a:dict(gain=0.,ci_low=0.,ci_high=0.,blocks=1))
    args=SimpleNamespace(representations=['heads'],penalty=.1,minimum_train=1,bootstrap=20,seed=1)
    predictions,comparisons,_=temporal.fit_readouts(pd.DataFrame(),{'voc_windows':features},w,args,tmp_path)
    assert 2 not in predictions.window_id.to_numpy()
    for i in range(0,len(calls),3):
        for j in (1,2):
            np.testing.assert_array_equal(calls[i][0],calls[i+j][0]);np.testing.assert_array_equal(calls[i][1],calls[i+j][1])
    assert set(comparisons.reference)=={'voc_no_voc','voc_no_direction'}


def test_private_cache_reuse_invalidation_and_artifact_integrity(monkeypatch,tmp_path):
    from analyses import replay_themes
    cp=tmp_path/'campaign/checkpoints/run/milestones/10000000';cp.mkdir(parents=True)
    actor=cp/'actor';actor.write_bytes(b'checkpoint')
    config=cp/'config';config.write_text('{}')
    buffer=tmp_path/'campaign/runtime_source/thinker/thinker/buffer.py';buffer.parent.mkdir(parents=True);buffer.write_text('# frozen')
    monkeypatch.setattr(replay_themes,'find_checkpoint',lambda *a:(cp,actor,config))
    capture=tmp_path/'capture/ep';capture.mkdir(parents=True)
    pack=capture/'packs/p/0';pack.parent.mkdir(parents=True);pack.write_bytes(b'pack')
    manifest=dict(field_schema=[dict(name=n,storage_group='p') for n in native.FLAGS+native.VALUES],files=[dict(path='packs/p/0')])
    (capture/'manifest.json').write_text(json.dumps(manifest))
    golden=tmp_path/'golden/tables';golden.mkdir(parents=True);(golden/'decisions.csv.gz').write_bytes(b'golden')
    prepared=dict(decisions=pd.DataFrame([dict(subject=1,game='pong')]),cache=golden.parent,
        metadata=dict(capture=str(capture.parent),manifest_sha256='manifest',checkpoint={},episodes=[dict(path='ep')]))
    calls=[]
    def extract(*a):
        calls.append(1)
        return {'voc_decisions':pd.DataFrame({'value':[len(calls)]})},{'version':'test'}
    monkeypatch.setattr(native,'extract',extract)
    args=SimpleNamespace(outputs_root=tmp_path/'outputs',chunk_size=256)
    first,_=native.prepare(args,prepared);second,_=native.prepare(args,prepared)
    assert len(calls)==1
    pd.testing.assert_frame_equal(first['voc_decisions'],second['voc_decisions'])
    pack.write_bytes(b'changed pack')
    native.prepare(args,prepared)
    assert len(calls)==2 and (golden/'decisions.csv.gz').read_bytes()==b'golden'
    artifact=args.outputs_root/'.voc_cache/00/sub001/pong/tables/voc_decisions.csv.gz'
    artifact.write_bytes(b'tampered')
    with pytest.raises(Exception,match='artifact verification'):native.prepare(args,prepared)


@pytest.mark.parametrize('failure', [None, 'hash', 'runtime', 'ambiguous'])
def test_checkpoint_selection_uses_capture_owner_not_duplicate_archive(tmp_path,monkeypatch,failure):
    from analyses import replay_themes
    campaigns=tmp_path/'campaigns';owner=campaigns/'fullframes'
    capture=tmp_path/'capture'
    cp=owner/'checkpoints/run/milestones/10000000';cp.mkdir(parents=True)
    actor=cp/'actor';actor.write_bytes(b'bound actor')
    config=cp/'config';config.write_bytes(b'bound config')
    buffer=owner/'runtime_source/thinker/thinker/buffer.py'
    buffer.parent.mkdir(parents=True);buffer.write_text('# frozen')
    record={'runs':[{'capture':str(capture)}]}
    (owner/'manifest.json').write_text(json.dumps(record))
    meta={'capture':str(capture),'checkpoint':dict(logical_xpid='run',requested_step=10000000,
        actor=dict(name='actor',sha256=native.sha(actor)),config=dict(name='config',sha256=native.sha(config)))}
    def forbidden(*args):raise AssertionError('Must not select an unrelated duplicate checkpoint')
    monkeypatch.setattr(replay_themes,'find_checkpoint',forbidden)
    if failure=='hash':actor.write_bytes(b'changed')
    elif failure=='runtime':buffer.unlink()
    elif failure=='ambiguous':
        other=campaigns/'other';other.mkdir();(other/'manifest.json').write_text(json.dumps(record))
    if failure:
        with pytest.raises((ValueError,FileNotFoundError)):
            native.checkpoint_for_capture(meta,campaigns)
    else:
        assert native.checkpoint_for_capture(meta,campaigns)==(cp,actor,config)


def test_plot_only_change_does_not_invalidate_numeric_analysis(monkeypatch):
    from common import research_cli as cli
    from analyses import validation_allocation
    numerical=cli.code_signature(validation_allocation)
    drawing=cli._render_signature('03',validation_allocation)
    original=Path.read_text
    def changed(path,*a,**kw):
        content=original(path,*a,**kw)
        if path.resolve()==Path(relationships.__file__).resolve():
            content=content.replace('fig.tight_layout()','fig.tight_layout(pad=2.)')
        return content
    monkeypatch.setattr(Path,'read_text',changed)
    assert cli.code_signature(validation_allocation)==numerical
    assert cli._render_signature('03',validation_allocation)!=drawing


@pytest.mark.skipif(os.environ.get('VOC_CAPTURE_CHECK')!='1',reason='Explicit read-only native capture check')
@pytest.mark.parametrize('game',['pong','enduro','spaceinvaders'])
def test_real_capture_sample(game):
    import yaml
    from common.research_inputs import capture_metadata
    from analyses.replay_themes import find_checkpoint
    meta=capture_metadata(REPO/'logs/visualization/hxln_10m',1,game)
    # Existing 00 scalars are golden references only; new fields come from Zarr.
    d=pd.read_csv(REPO/f'research_script/outputs/00/sub001/{game}/tables/decisions.csv.gz')
    ep=d.episode.iloc[0];part=d.loc[d.episode.eq(ep)]
    chosen=pd.concat([part.head(12),part.loc[part.cap_reached.eq(1)].head(2),part.loc[part.budget.eq(1)].head(2)])
    chosen=chosen.drop_duplicates(['episode','decision'])
    _,actor,config=find_checkpoint({'semantic':{'checkpoint':meta['checkpoint']}})
    tables,qa=native.extract(dict(decisions=chosen,metadata=meta),actor,yaml.safe_load(config.read_text()))
    assert len(tables['voc_decisions'])==len(chosen)
    assert qa['episodes'][0]['readout_max_error']<2e-4
    print(game,len(chosen),len(tables['voc_transitions']),qa['episodes'])
