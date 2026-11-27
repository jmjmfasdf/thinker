from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'research_script'))
from common import sampling


def fixture_tables():
    t = np.arange(31)/60
    real = pd.DataFrame(dict(episode=['a']*31,run=['r']*31,real_event=np.arange(31)*3,
        source_frame=np.arange(31),time=t,scored=np.arange(31)>=2,sampling_origin=0.))
    d = real.iloc[2:].copy()
    d['decision'] = np.arange(2,31)
    e = d.loc[d.index.repeat(2), ['episode','decision']].copy()
    e['event'] = np.arange(len(e))
    return d,e,real


def test_all_retained_grid_causal_and_choices_remain_complete():
    d,e,real = fixture_tables()
    result = sampling.prepare_tables(d,e,real)
    assert len(result['real_steps']) == 31
    assert len(result['decisions']) == len(d)
    assert len(result['choices']) == len(e)
    chosen = result['decisions_15hz']
    np.testing.assert_array_equal(chosen.source_frame, [4,8,12,16,20,24,28])
    assert len(result['choices_15hz']) == 2*len(chosen)
    assert result['sampling_15hz'].lag_seconds.ge(0).all()
    assert not result['sampling_15hz'].iloc[0].analysis_eligible


def test_repeats_grid_retained_but_not_reweighted():
    d,e,real = fixture_tables()
    real.time = np.arange(len(real))*.11
    d.time = real.loc[d.index,'time']
    tables = sampling.prepare_tables(d,e,real)
    assert tables['sampling_15hz'].duplicated(sampling.KEYS).any()
    assert not tables['decisions_15hz'].duplicated(sampling.KEYS).any()


def test_ineligible_frame_not_replaced_and_episode_clock_restarts():
    d,e,real = fixture_tables()
    d = d.loc[d.source_frame.ne(4)]
    other = real.copy();other.episode='b';other.time += 10;other.sampling_origin=10.
    tables = sampling.prepare_tables(d,e,pd.concat([real,other],ignore_index=True))
    assert 4 not in tables['decisions_15hz'].source_frame.values
    assert tables['decisions_15hz'].sample_index.iloc[0] == 2
    assert tables['sampling_15hz'].query("episode == 'b'").grid_time.iloc[0] == 10.
    assert not tables['sampling_15hz'].query("episode == 'b'").analysis_eligible.any()


@pytest.mark.parametrize('times', [[0,0,.1], [0,.2,.1], [0,np.nan]])
def test_bad_clock_rejected(times):
    with pytest.raises(ValueError):sampling.causal_grid(times)


def test_backend_join_keeps_missing_support_gaps():
    d,e,real = fixture_tables()
    selected=sampling.prepare_tables(d,e,real)['decisions_15hz']
    prepared=dict(decisions=selected,sample_mode='15hz')
    backend=d.loc[d.source_frame.ne(8)]
    ids,z=sampling.select_backend(backend,prepared)
    np.testing.assert_array_equal(z.source_frame, backend.iloc[ids].source_frame)
    np.testing.assert_array_equal(z.sample_index, [1,3,4,5,6,7])


@pytest.mark.parametrize('number',[f'{n:02}' for n in range(14)])
def test_every_canonical_entrypoint_accepts_both_modes(number,tmp_path):
    from common.research_cli import parser,output_base
    from common.paths import OUTPUTS
    assert parser(number).parse_args([]).outputs_root==OUTPUTS
    for mode in ('all','15hz'):
        args=parser(number).parse_args(['--sample-mode',mode,'--outputs-root',str(tmp_path)])
        assert output_base(args) == tmp_path
        assert args.resume is True


def test_01_legacy_layout_is_a_compatible_noop_for_both_modes(tmp_path):
    from common.research_cli import parser,output_base
    for mode in ('all','15hz'):
        args=parser('01').parse_args(['--outputs-root',str(tmp_path),'--legacy-output-layout','--sample-mode',mode])
        assert output_base(args) == tmp_path
    with pytest.raises(SystemExit):
        parser('02').parse_args(['--legacy-output-layout'])


def test_01_legacy_rerun_replaces_existing_result_in_place(tmp_path,monkeypatch):
    from common import research_cli as cli
    from common.result_store import ResultStore
    import json
    root=tmp_path/'01/sub001/pong'
    with ResultStore(root,dict(theme='01',subject=1,game='pong',dataset='human_behavior')) as store:
        with store.stage('prepare','old'):pass
        with store.stage('analyze','old') as stage:
            stage.write_table('observations',pd.DataFrame({'old':[1]}))
        store.publish(required_stages=('prepare','analyze'))
    monkeypatch.setattr(cli,'_input_identity',lambda *a:dict(dataset='human_behavior',sources=['raw']))
    monkeypatch.setattr(cli,'analyze',lambda *a:
        ({'observations':pd.DataFrame({'new':[1,2]})},dict(status='COMPLETED'),{}))
    args=cli.parser('01').parse_args(['--outputs-root',str(tmp_path),'--legacy-output-layout',
                                    '--stage','analyze','--resume'])
    result=cli.unit('01',args,1,'pong')
    assert result['path']==str(root)
    assert pd.read_csv(root/'tables/observations.csv.gz').columns.tolist()==['new']
    assert json.loads((root/'logs/run.json').read_text())['identity']['sample_mode']=='all'
    assert not (tmp_path/'samples').exists()


def test_human_resampling_retains_original_frame_ids(tmp_path):
    from analyses.human_behavior import _source
    n=61;path=tmp_path/'human.npz'
    a=np.eye(6)[(np.arange(n)//9)%6]
    np.savez(path,action=a,time=np.arange(n)/60,is_first=np.arange(n)==0,is_terminal=np.arange(n)==n-1)
    task=dict(path=str(path),file_id=0,game='pong',subject=1,session=1,run_label='r',stage='')
    _,full,qa=_source(task)
    bouts,sampled,qa15=_source(task,'15hz')
    assert len(full)==61 and len(sampled)==16
    np.testing.assert_array_equal(sampled.frame,np.arange(0,61,4))
    assert bouts.n_samples.sum()==len(sampled)
    assert qa15['n_original_frames']==61


def test_temporal_windows_use_sample_sequence_not_raw_stride():
    from analyses.evaluation_accumulation import make_windows
    n=45;t=np.arange(n)/15
    d=pd.DataFrame(dict(episode='e',run='r',session=1,decision=np.arange(n)*4,
        sample_index=np.arange(n),time=t,raw_bout=0,human_action=1,
        next_human_switch_time=np.inf,first_action=1))
    windows,qa=make_windows(d,np.full((n,3),1/3))
    assert len(windows)>0


def test_rich_endpoint_arrays_are_remapped_together(tmp_path):
    from analyses.control_pattern_pipeline import _select_rich
    import json
    d,e,real=fixture_tables()
    d=d.reset_index(drop=True)
    d['row']=np.arange(len(d));d['start']=d.row*2;d['end']=d.start+2
    d.to_csv(tmp_path/'decisions.csv.gz',index=False)
    pd.DataFrame(dict(row=np.repeat(d.row,2),k=np.tile([0,1],len(d)))).to_csv(tmp_path/'computations.csv.gz',index=False)
    for name in ('actor_features','policies','actual_policy','screen','tree_summary','screens'):
        n=2*len(d) if name in ('actor_features','policies') else len(d)
        np.save(tmp_path/(name+'.npy'),np.arange(n)[:,None])
    (tmp_path/'extraction.json').write_text('{}')
    chosen=sampling.prepare_tables(d,e,real)['decisions_15hz']
    _select_rich(tmp_path,dict(decisions=chosen,sample_mode='15hz'))
    result=pd.read_csv(tmp_path/'decisions.csv.gz')
    ends=pd.read_csv(tmp_path/'computations.csv.gz')
    assert len(result)==len(chosen) and len(ends)==2*len(chosen)
    np.testing.assert_array_equal(ends.row,np.repeat(np.arange(len(chosen)),2))
    np.testing.assert_array_equal(np.load(tmp_path/'actual_policy.npy').ravel(),chosen.row)
    assert result.segment.nunique()==1


def test_00_retains_both_views_and_cli_modes_replace_same_output(tmp_path,monkeypatch):
    from common import research_inputs as inputs, research_cli as cli
    import json
    d,e,real=fixture_tables()
    d['budget']=1;d['V_total']=1.;d['Q_total']=1.;d['V_initial']=0.;d['V_final']=1.
    d['Q_supported']=True
    cached=tmp_path/'scalar';cached.mkdir()
    d.to_csv(cached/'decisions.csv.gz',index=False);e.to_csv(cached/'choices.csv.gz',index=False)
    meta=dict(dataset='test',manifest_sha256='a'*64,capture=str(tmp_path),episodes=[])
    old=dict(qa=dict(config={},episodes=[],sources={}),signature='scalar')
    monkeypatch.setattr(inputs,'capture_metadata',lambda *a:meta)
    monkeypatch.setattr(cli,'_input_identity',lambda *a:meta)
    monkeypatch.setattr(inputs,'verified_scalar_cache',lambda *a:(cached,old))
    monkeypatch.setattr(sampling,'inventory',lambda *a:real)
    monkeypatch.setattr(cli,'analyze',lambda number,module,args,subject,game,prepared,workdir:
        ({'observations':prepared['decisions']},dict(status='COMPLETED'),{}))
    digests=[];targets=[]
    for selected_mode in ('all','15hz','all'):
        args=cli.parser('02').parse_args(['--outputs-root',str(tmp_path/'results'),
            '--sample-mode',selected_mode,'--stage','analyze'])
        prepared=inputs.prepare_native(args,1,'pong')
        assert len(prepared['decisions']) == (29 if selected_mode=='all' else 7)
        digests.append(inputs.sha(prepared['cache']/'tables/decisions.csv.gz'))
        result=cli.unit('02',args,1,'pong')
        target=cli.output_base(args)/'02/sub001/pong';targets.append(target)
        assert result['path']==str(target)
        assert len(pd.read_csv(target/'tables/observations.csv.gz'))==len(prepared['decisions'])
        assert json.loads((target/'logs/run.json').read_text())['analysis']['sample_mode']==selected_mode
    assert len(set(digests))==1
    assert len(set(targets))==1
    assert not (tmp_path/'results/samples').exists()


def saved_result(root,selected_mode):
    from common.result_store import ResultStore
    with ResultStore(root,dict(theme='02',subject=int(root.parent.name[3:]),game='pong',sample_mode=selected_mode),resume=True) as store:
        with store.stage('prepare','fixture'):pass
        with store.stage('analyze','fixture') as stage:
            stage.write_table('observations',pd.DataFrame({'mode':[selected_mode]}))
        store.publish(required_stages=('prepare','analyze'))


def test_shared_path_render_rejects_wrong_mode_without_changing_result(tmp_path):
    from common import research_cli as cli
    from common.result_store import ResultStoreError
    root=tmp_path/'02/sub001/pong';saved_result(root,'all')
    before=(root/'logs/run.json').read_bytes()
    args=cli.parser('02').parse_args(['--outputs-root',str(tmp_path),'--sample-mode','15hz','--render-only'])
    with pytest.raises(ResultStoreError,match='sample_mode'):
        cli.unit('02',args,1,'pong')
    assert (root/'logs/run.json').read_bytes()==before


def test_shared_path_group_rejects_mixed_modes(tmp_path):
    from common import research_cli as cli
    from common.result_store import ResultStoreError
    for subject,selected_mode in ((1,'all'),(2,'15hz')):
        saved_result(tmp_path/f'02/sub{subject:03d}/pong',selected_mode)
    args=cli.parser('02').parse_args(['--outputs-root',str(tmp_path),'--subjects','1','2','--games','pong','--aggregate'])
    with pytest.raises(ResultStoreError,match='sample_mode'):
        cli.group('02',args)
    assert not (tmp_path/'02/group/pong').exists()


def test_preexisting_group_mode_is_read_from_participant_identities():
    from common import research_cli as cli
    from common.result_store import ResultStoreError
    meta={'identity':{'sources':{'1':{'identity':{'sample_mode':'all'}},'2':{'identity':{'sample_mode':'all'}}}}}
    args=cli.parser('02').parse_args([])
    cli._require_sample_mode(meta,args,'saved-group')
    args.sample_mode='15hz'
    with pytest.raises(ResultStoreError,match='sample_mode'):
        cli._require_sample_mode(meta,args,'saved-group')


def test_failed_mode_replacement_preserves_previous_result(tmp_path,monkeypatch):
    from common import research_cli as cli
    import json
    root=tmp_path/'02/sub001/pong';saved_result(root,'all')
    before={str(p.relative_to(root)):p.read_bytes() for p in (root/'tables').glob('*')}
    previous=json.loads((root/'logs/run.json').read_text())
    monkeypatch.setattr(cli,'_input_identity',lambda *a:dict(dataset='test',manifest_sha256='b'*64))
    monkeypatch.setattr(cli,'prepare_native',lambda *a:{})
    def fail(*a):raise RuntimeError('failed requested 15hz fit')
    monkeypatch.setattr(cli,'analyze',fail)
    args=cli.parser('02').parse_args(['--outputs-root',str(tmp_path),'--sample-mode','15hz'])
    with pytest.raises(RuntimeError,match='failed requested'):
        cli.unit('02',args,1,'pong')
    assert {str(p.relative_to(root)):p.read_bytes() for p in (root/'tables').glob('*')}==before
    after=json.loads((root/'logs/run.json').read_text())
    for key in ('identity','stages','status','completed_at'):
        assert after[key]==previous[key]
    assert after['latest_attempt']['status']=='FAILED'


def test_voc_history_and_windows_accept_strided_original_steps():
    from analyses.voc_relationships import trailing_features,window_features,VOC_FEATURES
    n=20
    d=pd.DataFrame(dict(episode='e',decision=np.arange(n)*4,sample_index=np.arange(n),
        time=np.arange(n)/15,initial_voc=np.arange(n)*.01,endpoint_voc_change=.01,U_V=.1))
    history=trailing_features(d)
    assert history.past_voc_mean.notna().sum()>10
    windows=pd.DataFrame([dict(window_id=0,start=0,end=8)])
    right=d[['episode','decision']].copy()
    for name in VOC_FEATURES:right[name]=.1
    left=d.drop(columns=[c for c in VOC_FEATURES if c in d])
    values=window_features(left,windows,right)
    assert values.voc_window_supported.all()
