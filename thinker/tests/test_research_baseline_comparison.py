"""Exact pairing, participant/episode/session inference, and honest figure scales."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'research_script'))
from analyses import validation_allocation as validation
from common.research_cli import _external_signature, parser


@pytest.fixture
def comparison(tmp_path):
    rows=[];native=[]
    arrays=tmp_path/'audit/aligned_arrays';arrays.mkdir(parents=True)
    (arrays.parent/'manifest.json').write_text(json.dumps(dict(subjects=[1,2],games=[1,2],models=list(validation.COMPARISON_MODELS))))
    (arrays.parent/'qa.json').write_text(json.dumps(dict(complete=True)))
    for subject in (1,2):
        names=[];run=[];frames=[];times=[];prob=[]
        for block,n,correct,dynamic_correct in ((1,5,1,2),(2,100,80,80)):
            name=f'sub{subject:03d}_ses01_block{block:02d}_game1';names.append(name)
            run.extend([block-1]*n);frames.extend(range(2,n+2));times.extend(np.arange(2,n+2)/30)
            p=np.zeros((n,6));p[:,0]=np.where(np.arange(n)<correct,.9,.1);p[:,1]=1-p[:,0];prob.extend(p)
            for model in (*validation.BASELINE_MODELS,'thinker'):
                rows.append(dict(subject=subject,game=1,model=model,seed=0,run_label=name,
                    decisions=n,action_accuracy=correct/n if model!='thinker' else .99,LL=np.log(p[:,0]).mean()))
            raw=tmp_path/f'behavior/sub-{subject:03d}/ses-01/sub{subject:03d}-ses01-block{block}-game1.npz'
            raw.parent.mkdir(parents=True,exist_ok=True)
            np.savez(raw,action=np.zeros(n+2,dtype=int),time=np.arange(n+2)/30)
            if subject==1:
                for f in range(n+2):
                    ok=2<=f<dynamic_correct+2
                    native.append(dict(run=f'sub-001/ses-01/sub001-ses01-block{block}-game1.npz',
                        source_frame=f,time=f/30,human_action=0,incumbent_action=0,model_correct=int(ok),
                        human_probability=.8 if ok else .1,log_score=np.log(.8 if ok else .1),
                        episode=name,session=1,decision=f,budget=1,switch=0))
        np.savez(arrays/f'sub{subject:03d}_game1.npz',run_names=np.array(names),run=np.array(run),
            target_index=np.array(frames),time=np.array(times),action=np.zeros(len(run),int),previous=np.zeros(len(run),int),
            **{'p_'+m:np.array(prob) for m in validation.BASELINE_MODELS})
    pd.DataFrame(rows).to_csv(tmp_path/'run_metrics.csv',index=False)
    (tmp_path/'manifest.json').write_text(json.dumps(dict(subjects=[1,2],games=[1,2],baseline_seeds=[0])))
    (tmp_path/'qa.json').write_text(json.dumps(dict(complete=True,duplicate_keys=0,
        cross_model_label_history_alignment_mismatches=0,invalid_probability_rows=0)))
    decisions=pd.DataFrame(native)
    scores=decisions.groupby('run').agg(n=('model_correct','size'),accuracy=('model_correct','mean'),log_score=('log_score','mean')).reset_index()
    scores['population']='all'
    return SimpleNamespace(baseline_root=tmp_path,baseline_predictions_root=arrays,behavior_root=tmp_path/'behavior',
                           sample_mode='all',test_decisions=decisions),scores


def compare(args,subject,game,native):
    decisions=args.test_decisions.copy()
    decisions['run']=decisions.run.str.replace('001',f'{subject:03d}',regex=False)
    return validation.baseline_comparison(args,subject,game,native,decisions)


def test_four_models_equal_runs_not_decision_weights(comparison):
    args,native=comparison
    tables,qa=compare(args,1,'pong',native)
    d=tables['baseline_comparison_subject_scores'].set_index('model')
    assert set(d.index)==set(validation.COMPARISON_MODELS)
    assert 'thinker' not in d.index
    assert d.loc['dqn','accuracy']==pytest.approx(.5)
    assert d.loc['dynamic_thinker','accuracy']==pytest.approx(.6)
    assert d.loc['dqn','n_seeds']==1 and d.loc['dynamic_thinker','n_seeds']==1
    assert set(tables['baseline_comparison_run_scores'].subject)=={1}
    assert qa['decision_matched'] is True and qa['common_run_count']==2
    assert qa['matched_decisions']==105 and qa['native_excluded']==4 and qa['baseline_excluded']==0
    assert tables['baseline_comparison_run_scores'].groupby('run').n_decisions.nunique().eq(1).all()
    assert all(len(s['sha256'])==64 for s in qa['sources'])


def test_no_enduro_baseline_does_not_fabricate_boxes(comparison):
    args,native=comparison
    tables,qa=compare(args,1,'enduro',native)
    assert qa['status']=='unavailable_game'
    assert 'baseline_comparison_subject_scores' not in tables


@pytest.mark.parametrize('change', ['duplicate','missing_model','extra_run','bad_probability','bad_count','wrong_subject'])
def test_invalid_comparison_fails_closed(comparison,change):
    args,native=comparison
    p=args.baseline_root/'run_metrics.csv';d=pd.read_csv(p)
    if change=='duplicate':d=pd.concat([d,d.iloc[[0]]],ignore_index=True)
    elif change=='missing_model':d=d.loc[d.model.ne('drc')]
    elif change=='extra_run':d.loc[0,'run_label']='sub001_ses01_block99_game1'
    elif change=='bad_probability':d.loc[0,'action_accuracy']=1.1
    elif change=='bad_count':
        d['decisions']=d.decisions.astype(float);d.loc[0,'decisions']=1.5
    elif change=='wrong_subject':d.loc[0,'run_label']='sub002_ses01_block01_game1'
    d.to_csv(p,index=False)
    with pytest.raises(ValueError):compare(args,1,'pong',native)


def test_missing_baseline_is_not_silent_success(comparison):
    args,native=comparison
    args.baseline_root=args.baseline_root/'missing'
    with pytest.raises(FileNotFoundError):compare(args,1,'pong',native)


def test_failed_baseline_qa_rejected(comparison):
    args,native=comparison
    (args.baseline_root/'qa.json').write_text(json.dumps({'complete':False}))
    with pytest.raises(ValueError):compare(args,1,'pong',native)


def test_input_changes_invalidate_analysis_signature(comparison):
    args,native=comparison
    first=_external_signature('02',args,{},1,'pong')
    p=args.baseline_root/'run_metrics.csv'
    p.write_text(p.read_text()+'\n')
    assert first!=_external_signature('02',args,{},1,'pong')


def test_render_reads_saved_tables_only(comparison,tmp_path):
    args,native=comparison
    tables,_=compare(args,1,'pong',native)
    # Original input removal cannot affect figure generation.
    (args.baseline_root/'run_metrics.csv').unlink()
    saved=[]
    import matplotlib.pyplot as plt
    def save(fig,name,comparison=False):
        assert comparison and name=='human_action_four_model_boxplot'
        assert len(fig.axes)==1
        assert all(len(ax.collections)==4 for ax in fig.axes)
        assert fig._suptitle is None or not fig._suptitle.get_text()
        assert not fig.texts
        # A one-person group panel still shows every run, not one subject mean.
        saved.append([len(points.get_offsets()) for points in fig.axes[0].collections])
        fig.savefig(tmp_path/(name+'.png'));plt.close(fig)
    validation._plot_baseline_comparison(tables,'pong sub001',save)
    assert saved[0] == [2] * 4
    tables['group_inputs']=pd.DataFrame({'subject':[1]})
    tables['baseline_comparison_accuracy_tests']=validation.comparison_accuracy_tests(tables)
    tables.update(validation.comparison_episode_tests(tables))
    validation._plot_baseline_comparison(tables,'pong group',save)
    assert saved[1] == [2] * 4


def test_baseline_argument_is_explicit_02_input():
    assert parser('02').parse_args(['--baseline-root','example']).baseline_root==Path('example')
    assert not hasattr(parser('00').parse_args([]),'baseline_root')


def test_existing_validation_tables_are_unchanged(comparison,monkeypatch):
    args,native=comparison
    prepared={'decisions':args.test_decisions.copy()}
    monkeypatch.setattr(validation,'_metadata',lambda prepared:({},{},{}))
    monkeypatch.setattr(validation,'_training',lambda *args:({},{}))
    old,oldqa,_=validation.validation(SimpleNamespace(baseline_root=None),1,'pong',prepared)
    new,newqa,_=validation.validation(args,1,'pong',prepared)
    for name,frame in old.items():pd.testing.assert_frame_equal(frame,new[name])
    assert oldqa['baseline_comparison']['status']=='not_requested'
    assert newqa['baseline_comparison']['status']=='available_decision_matched'


def test_group_has_separate_requested_subject_panels_with_run_points(comparison):
    args,native=comparison
    first,_=compare(args,1,'pong',native)
    other=native.copy();other['run']=other.run.str.replace('001','002',regex=False)
    second,_=compare(args,2,'pong',other)
    tables={name:pd.concat([first[name],second[name]],ignore_index=True) for name in first}
    tables['group_inputs']=pd.DataFrame({'subject':[1,2]})
    tables['baseline_comparison_accuracy_tests']=validation.comparison_accuracy_tests(tables)
    tables.update(validation.comparison_episode_tests(tables))
    import matplotlib.pyplot as plt
    def save(fig,name,comparison=False):
        assert fig._suptitle is None or not fig._suptitle.get_text()
        assert not fig.texts
        assert len(fig.axes)==2
        assert [t.get_text() for ax in fig.axes for t in ax.texts if t.get_gid()=='participant-label']==['sub001','sub002']
        for ax in fig.axes:
            assert len(ax.collections)==4
            assert all(len(points.get_offsets())==2 for points in ax.collections)
        plt.close(fig)
    validation._plot_baseline_comparison(tables,'group',save)


def group_comparison(n=4,minimum=.7):
    subjects=pd.DataFrame([dict(subject=s,game='pong',model=model,accuracy=minimum+.03*i+.001*s)
        for s in range(1,n+1) for i,model in enumerate(validation.COMPARISON_MODELS)])
    tables={'baseline_comparison_subject_scores':subjects,'group_inputs':pd.DataFrame({'subject':range(1,n+1)})}
    tables['baseline_comparison_accuracy_tests']=validation.comparison_accuracy_tests(tables)
    return tables


def episode_comparison(n=4,minimum=.7):
    tables=group_comparison(n,minimum)
    runs=[]
    for row in tables['baseline_comparison_subject_scores'].itertuples():
        for episode in range(11):
            session,block=episode//3+1,episode%3+1
            runs.append(dict(subject=row.subject,game=row.game,model=row.model,seed=0,
                run=f'sub{row.subject:03d}_ses{session:02d}_block{block:02d}_game1',
                accuracy=row.accuracy+.0001*episode,n_decisions=10**(episode%4)))
    tables['baseline_comparison_run_scores']=pd.DataFrame(runs)
    tables.update(validation.comparison_episode_tests(tables))
    return tables


def test_exact_four_participants_and_holm_cannot_make_stars():
    tables=group_comparison()
    tests=tables['baseline_comparison_accuracy_tests']
    assert len(tests)==6 and tests.unit.eq('participant').all()
    assert tests.n_units.eq(4).all() and tests.n_pairs.eq(4).all()
    np.testing.assert_allclose(tests.p_raw,.125)
    np.testing.assert_allclose(tests.p_holm,.75)
    assert tests.significance.eq('ns').all()
    assert not tests.decision_matched.any()


def test_individual_runs_flip_together_by_session(comparison):
    args,native=comparison
    tables,_=compare(args,1,'pong',native)
    points=[]
    for session,count in ((1,2),(2,7),(3,3)):
        for block in range(count):
            for i,model in enumerate(validation.COMPARISON_MODELS):
                points.append(dict(subject=1,model=model,seed=0,
                    run=f'sub001_ses{session:02d}_block{block:02d}_game1',accuracy=.65+.02*i+.001*session))
    tables['baseline_comparison_run_scores']=pd.DataFrame(points)
    tests=validation.comparison_accuracy_tests(tables)
    assert tests.unit.eq('session').all() and tests.n_units.eq(3).all()
    assert tests.n_pairs.eq(12).all()
    np.testing.assert_allclose(tests.p_raw,.25)
    assert tests.significance.eq('ns').all()
    assert tests.iloc[0].effect_percentage_points==pytest.approx(-2.)


def test_ties_and_insufficient_units(comparison):
    tables=group_comparison()
    tables['baseline_comparison_subject_scores']['accuracy']=.8
    tests=validation.comparison_accuracy_tests(tables)
    assert tests.p_raw.eq(1).all() and tests.p_holm.eq(1).all()
    args,native=comparison
    tables,_=compare(args,1,'pong',native)
    tests=tables['baseline_comparison_accuracy_tests']
    assert tests.status.eq('insufficient_units').all()
    assert tests.p_holm.isna().all() and tests.significance.eq('NA').all()


def test_missing_model_pair_is_rejected():
    tables=group_comparison()
    tables['baseline_comparison_subject_scores']=tables['baseline_comparison_subject_scores'].iloc[1:]
    with pytest.raises(ValueError,match='Unpaired'):validation.comparison_accuracy_tests(tables)


@pytest.mark.parametrize('minimum',[.55,.70,.01,.90])
def test_accuracy_only_zoom_does_not_hide_points(minimum):
    tables=episode_comparison(minimum=minimum)
    import matplotlib.pyplot as plt
    def save(fig,name,comparison=False):
        assert len(fig.axes)==4
        assert len({ax.get_ylim() for ax in fig.axes})==1
        for subject,ax in enumerate(fig.axes,1):
            lo,hi=ax.get_ylim()
            points=tables['baseline_comparison_run_scores'].query('subject == @subject')
            assert 0<=lo<=points.accuracy.min()
            assert points.accuracy.max()<=hi<=1
            assert ax.get_ylabel()=='Accuracy'
            for model,collection in zip(validation.COMPARISON_MODELS,ax.collections):
                np.testing.assert_allclose(collection.get_offsets()[:,1],points.loc[points.model.eq(model),'accuracy'])
            assert (lo>0)==any(line.get_gid()=='y-axis-break' for line in ax.lines)
            assert sum(line.get_gid()=='accuracy-significance-bracket' for line in ax.lines)==6
            assert [t.get_text() for t in ax.texts if t.get_gid()=='accuracy-significance']==['**']*6
            assert not fig.texts and not ax.get_title()
        plt.close(fig)
    validation._plot_baseline_comparison(tables,'No title',save)


def test_stars_use_corrected_not_raw_p_values():
    # Ten independent units can support stars; four must never be inflated to ten.
    tables=group_comparison(n=10)
    tests=tables['baseline_comparison_accuracy_tests']
    np.testing.assert_allclose(tests.p_raw,2/1024)
    np.testing.assert_allclose(tests.p_holm,12/1024)
    assert tests.significance.eq('*').all()


def test_group_recomputes_tests_instead_of_pooling_individual_pvalues(comparison,monkeypatch,tmp_path):
    from analyses import group_results
    args,native=comparison
    first,_=compare(args,1,'pong',native)
    other=native.copy();other['run']=other.run.str.replace('001','002',regex=False)
    second,_=compare(args,2,'pong',other)
    by_subject={'1':first,'2':second}
    monkeypatch.setattr(group_results,'_read_unit',lambda path:(by_subject[Path(path).name],{}))
    tables,qa,_=group_results.aggregate('02',SimpleNamespace(subjects=[1,2]),[(s,'pong',tmp_path/str(s)) for s in (1,2)])
    tests=tables['baseline_comparison_accuracy_tests']
    assert len(tests)==6 and tests.n_units.eq(2).all() and tests.unit.eq('participant').all()
    assert tests.decision_matched.all()
    assert 'accuracy_inference' in qa
    episode=tables['baseline_comparison_episode_accuracy_tests']
    assert len(episode)==12 and episode.unit.eq('episode').all() and episode.n_units.eq(2).all()
    assert set(episode.subject)=={1,2}
    sensitivity=tables['baseline_comparison_session_accuracy_tests']
    assert len(sensitivity)==12 and sensitivity.unit.eq('session').all()
    assert sensitivity.n_units.eq(1).all() and sensitivity.significance.eq('NA').all()
    assert 'within-session dependence' in qa['accuracy_figure_inference']


def test_episode_tests_are_paired_within_subject_not_pooled_frames():
    tables=episode_comparison()
    episode=tables['baseline_comparison_episode_accuracy_tests']
    assert len(episode)==24 and episode.n_units.eq(11).all() and episode.n_pairs.eq(11).all()
    np.testing.assert_allclose(episode.p_raw,2/2048)
    np.testing.assert_allclose(episode.p_holm,12/2048)
    assert episode.significance.eq('**').all()
    assert episode.groupby('subject').size().eq(6).all()
    assert episode.n_sessions.eq(4).all()
    session=tables['baseline_comparison_session_accuracy_tests']
    np.testing.assert_allclose(session.p_raw,.125)
    assert session.significance.eq('ns').all() and session.n_units.eq(4).all()
    np.testing.assert_allclose(episode.effect_a_minus_b,session.effect_a_minus_b)
    assert tables['baseline_comparison_accuracy_tests'].unit.eq('participant').all()


def test_episode_tests_average_seeds_and_are_order_invariant():
    tables=episode_comparison()
    expected=tables['baseline_comparison_episode_accuracy_tests']
    runs=tables['baseline_comparison_run_scores']
    tables['baseline_comparison_run_scores']=pd.concat([runs,runs.assign(seed=1)],ignore_index=True).sample(frac=1,random_state=9)
    actual=validation.comparison_episode_tests(tables)['baseline_comparison_episode_accuracy_tests']
    pd.testing.assert_frame_equal(expected,actual)


@pytest.mark.parametrize('problem',['missing_pair','duplicate','wrong_subject','mixed_game','nonfinite'])
def test_episode_panel_inputs_fail_closed(problem):
    tables=episode_comparison()
    runs=tables['baseline_comparison_run_scores'].copy()
    if problem=='missing_pair':runs=runs.iloc[1:]
    elif problem=='duplicate':runs=pd.concat([runs,runs.iloc[[0]]],ignore_index=True)
    elif problem=='wrong_subject':runs.loc[0,'subject']=99
    elif problem=='mixed_game':runs.loc[0,'game']='spaceinvaders'
    else:runs.loc[0,'accuracy']=np.nan
    tables['baseline_comparison_run_scores']=runs
    with pytest.raises(ValueError):validation.comparison_episode_tests(tables)


def test_episode_render_is_saved_table_only_and_rejects_stale_tests(monkeypatch):
    tables=episode_comparison()
    import matplotlib.pyplot as plt
    def no_compute(*args,**kwargs):raise AssertionError('Renderer must not recompute inference')
    monkeypatch.setattr(validation,'comparison_episode_tests',no_compute)
    monkeypatch.setattr(validation,'comparison_accuracy_tests',no_compute)
    validation._plot_baseline_comparison(tables,'',lambda fig,*args,**kwargs:plt.close(fig))
    tables.pop('baseline_comparison_episode_accuracy_tests')
    with pytest.raises(ValueError,match='rerun 02 --aggregate'):
        validation._plot_baseline_comparison(tables,'',lambda *args,**kwargs:None)


def test_participant_panel_significance_labels_do_not_touch_next_bracket():
    import matplotlib.pyplot as plt
    from common.figure_style import paper_style
    tables=episode_comparison()
    def save(fig,*args,**kwargs):
        # The export wrapper can invoke layout more than once.
        fig.tight_layout();fig.tight_layout();fig.canvas.draw()
        renderer=fig.canvas.get_renderer()
        for ax in fig.axes:
            brackets=[line for line in ax.lines if line.get_gid()=='accuracy-significance-bracket']
            labels=[text for text in ax.texts if text.get_gid()=='accuracy-significance']
            for label,upper in zip(labels[:-1],brackets[1:]):
                bottom=upper.get_transform().transform(upper.get_xydata())[:,1].min()
                assert label.get_window_extent(renderer).y1<bottom
            assert ax.yaxis.label.get_fontsize()>=26
            assert all(t.get_fontsize()>=22 for t in ax.get_xticklabels())
        plt.close(fig)
    with paper_style():validation._plot_baseline_comparison(tables,'',save)


@pytest.mark.parametrize('change',['time','human_action','incumbent_action','duplicate','missing'])
def test_alignment_mismatches_fail_closed(comparison,change):
    args,native=comparison
    d=args.test_decisions.copy();idx=d.index[d.source_frame.eq(4)][0]
    if change=='duplicate':d=pd.concat([d,d.loc[[idx]]],ignore_index=True)
    elif change=='missing':d=d.drop(idx)
    else:d.loc[idx,change]+=1
    with pytest.raises(ValueError):validation.baseline_comparison(args,1,'pong',native,d)


def test_15hz_intersection_recomputes_every_model_on_same_frames(comparison):
    args,native=comparison;args.sample_mode='15hz'
    selected=args.test_decisions.loc[args.test_decisions.source_frame.mod(2).eq(0)].copy()
    tables,qa=validation.baseline_comparison(args,1,'pong',native,selected)
    matched=tables['baseline_comparison_decisions']
    assert matched.source_frame.mod(2).eq(0).all()
    assert qa['matched_decisions']==53 and qa['baseline_excluded']==52 and qa['native_excluded']==2
    counts=tables['baseline_comparison_run_scores'].groupby('run').n_decisions
    assert counts.nunique().eq(1).all()
    for model in validation.COMPARISON_MODELS:
        expected=matched.groupby('run')[model+'_correct'].mean()
        actual=tables['baseline_comparison_run_scores'].query('model == @model').set_index('run').accuracy
        np.testing.assert_allclose(actual.reindex(expected.index),expected)


def test_prediction_file_changes_invalidate_signature(comparison):
    args,_=comparison
    first=_external_signature('02',args,{},1,'pong')
    p=args.baseline_predictions_root/'sub001_game1.npz'
    with np.load(p) as z:data={k:z[k] for k in z.files}
    data['time']=data['time']+1
    np.savez(p,**data)
    assert first!=_external_signature('02',args,{},1,'pong')


def test_no_silent_fallback_to_unmatched_aggregate_scores(comparison):
    args,native=comparison
    (args.baseline_predictions_root/'sub001_game1.npz').unlink()
    with pytest.raises(FileNotFoundError,match='baseline prediction'):compare(args,1,'pong',native)


def test_archive_must_reproduce_official_scores(comparison):
    args,native=comparison;p=args.baseline_predictions_root/'sub001_game1.npz'
    with np.load(p) as z:data={k:z[k] for k in z.files}
    data['p_dqn']=np.roll(data['p_dqn'],1,axis=1)
    np.savez(p,**data)
    with pytest.raises(ValueError,match='reproduce official'):compare(args,1,'pong',native)


def test_matching_uses_frame_keys_not_row_order_or_historical_thinker(comparison):
    args,native=comparison
    expected,_=compare(args,1,'pong',native)
    p=args.baseline_predictions_root/'sub001_game1.npz'
    with np.load(p) as z:data={k:z[k] for k in z.files}
    for k in data:
        if k!='run_names':data[k]=data[k][::-1]
    data['p_thinker']=np.full((105,6),np.nan)  # Never read historical Thinker output.
    np.savez(p,**data)
    actual,_=validation.baseline_comparison(args,1,'pong',native,args.test_decisions.iloc[::-1])
    pd.testing.assert_frame_equal(expected['baseline_comparison_run_scores'],actual['baseline_comparison_run_scores'])


def test_archived_target_must_match_original_behavior(comparison):
    args,native=comparison
    p=args.behavior_root/'sub-001/ses-01/sub001-ses01-block1-game1.npz'
    with np.load(p) as z:data={k:z[k] for k in z.files}
    data['action'][4]=1;np.savez(p,**data)
    with pytest.raises(ValueError,match='original behavior'):compare(args,1,'pong',native)
