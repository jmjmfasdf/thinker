"""Checks for scientific contrasts and independence assumptions in reports."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


PATH = Path(__file__).parents[1]/'diagnostics/summarize_adaptive_planning_evidence.py'
SPEC = importlib.util.spec_from_file_location('planning_evidence', PATH)
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def test_one_ulp_probability_roundoff_is_counted_but_real_invalid_values_fail():
    raw=np.array([0.,.5,1.,1.+np.finfo(np.float32).eps])
    corrected,qa=REPORT.checked_probabilities(raw)
    np.testing.assert_array_equal(corrected,[0.,.5,1.,1.])
    assert raw[-1] > 1.  # Never modify the raw source.
    assert qa['corrected_values']==1
    assert qa['maximum_correction']==np.finfo(np.float32).eps
    for invalid in ([1.0001],[-.0001],[np.nan],[np.inf]):
        with pytest.raises(ValueError,match='beyond float32 roundoff'):
            REPORT.checked_probabilities(invalid)


def test_choice_matched_contrast_known_rewards():
    gain, reference = REPORT.matched_choice_advantage([.1,.9],[-1.,1.],[1,1])
    assert gain == pytest.approx(.4)
    np.testing.assert_allclose(reference,[.5,.5])
    reversed_gain, _ = REPORT.matched_choice_advantage([.9,.1],[-1.,1.],[1,1])
    assert reversed_gain == pytest.approx(-.4)


def test_initial_gate_null_handles_zero_half_and_one():
    length,cap=REPORT.initial_gate_null([0.,.5,1.],minimum=1,cap=3)
    np.testing.assert_allclose(length,[1.,1.75,3.])
    np.testing.assert_allclose(cap,[0.,.25,1.])


def test_count_only_policy_has_no_state_allocation_advantage():
    gain, _ = REPORT.matched_choice_advantage([.1,.1,.9,.9],[-2.,-1.,1.,2.],[1,1,12,12])
    assert gain == 0.


def test_identical_rewards_minus_cost_do_not_create_spurious_correlation():
    utility=np.array([-0.0001,-0.0001+2e-16,-0.0001-2e-16])
    probability=np.array([.1,.5,.9])
    assert REPORT.correlation(probability,utility,rank=True) is None
    gain,_=REPORT.matched_choice_advantage(probability,utility,[1,1,1])
    assert gain == 0.


def test_cluster_count_does_not_count_repetitions_or_counts_as_independent():
    states = [dict(seed=1,stream=stream,count=count,repetitions=2,
        probability=.1+.8*stream,ema_delta=-1+2*stream,online_delta=-1+2*stream,
        utility=-1+2*stream,task_utility=-.9+2*stream,task_tail_bound=.01,
        additional_current_stage_computations=1,additional_current_stage_cost=.1,
        all_future_cost_net_utility=-1.1+2*stream,task_difference_exceeds_tail_bound=1.)
        for stream in [0,1] for count in [1,3,6,12]]
    result = REPORT.utility_report(states,bootstrap=50)
    assert result['states'] == 8
    assert result['paired_repetitions'] == 16
    assert result['independent_streams'] == 2
    assert result['choice_matched_advantage'] == pytest.approx(.4)
    assert result['choice_matched_advantage_tail_bound'] == pytest.approx(.004)
    assert result['mean_additional_current_stage_computations'] == 1
    assert result['mean_task_utility']-result['mean_additional_current_stage_cost'] == pytest.approx(result['mean_net_utility'])
    # Exact pre-intervention predictions remain exact under whole-stream resampling.
    assert result['ema_prediction_bias'] == 0.
    assert result['ema_prediction_bias_ci95'] == [0., 0.]
    assert result['ema_rmse_improvement_over_zero'] == 1.
    assert result['ema_rmse_improvement_over_zero_ci95'] == [1., 1.]


def test_variable_computation_cost_does_not_imply_task_return_prediction():
    states = [dict(seed=1, stream=i, count=1, repetitions=2,
        probability=p, ema_delta=p, online_delta=p, utility=-cost,
        task_utility=0., task_tail_bound=.00001,
        additional_current_stage_computations=int(round(cost/.1)),
        additional_current_stage_cost=cost, all_future_cost_net_utility=-cost,
        task_difference_exceeds_tail_bound=0.)
        for i, (p, cost) in enumerate(zip([.1,.3,.7,.9], [.1,.2,.3,.4]))]
    result = REPORT.utility_report(states, bootstrap=50)
    assert result['ema_utility_spearman'] == pytest.approx(-1.)
    assert result['ema_task_utility_spearman'] is None
    assert result['choice_matched_task_advantage'] == 0.
    assert result['choice_matched_computation_cost_component'] == pytest.approx(-.035)
    assert result['choice_matched_advantage'] == pytest.approx(-.035)
    assert result['states_with_any_task_difference_exceeding_tail_bound'] == 0
    assert result['choice_matched_advantage_leave_one_stream_out_range'] is not None
    states[0]['utility'] = 1.
    with pytest.raises(ValueError, match='task benefit minus computation cost'):
        REPORT.utility_report(states, bootstrap=50)


def arm_return(task, queries, all_cost):
    return dict(task_truncated=task, net_truncated=task-.1*queries,
                current_stage_cost=.1*queries, stage_computations=queries,
                cost=.1, all_future_cost_net_truncated=task-all_cost)


def test_repetitions_share_preintervention_predictions():
    pairs, arms = [], {}
    for rep in range(2):
        entry = dict(hash='state',p_continue=.7)
        arms[(8,1,rep)] = dict(
            stop=dict(prefix=128,entries=[entry],returns=[arm_return(0.,0,.2)]),
            continue_=dict(prefix=128,entries=[entry.copy()],returns=[arm_return(rep+.1,1,.4)]))
        pairs.append(dict(seed=8,stream=0,count=1,repetition=rep,entry_hash='state',
            ema_delta=.02,online_delta=.01,delta_net=rep,delta_task=rep+.1,
            delta_all_future_cost_net=rep-.1,pair_task_tail_bound=.002))
    states = REPORT.paired_states(pairs,arms)
    assert len(states) == 1
    assert states[0]['utility'] == .5
    assert states[0]['repetitions'] == 2
    assert states[0]['intervention_real_decision_index']==128
    arms[(8,1,1)]['continue_']['prefix']=896
    with pytest.raises(ValueError,match='prefixes differ'):
        REPORT.paired_states(pairs,arms)
    arms[(8,1,1)]['continue_']['prefix']=128
    for rep in range(2):
        for arm in ('stop','continue_'):
            arms[(8,1,rep)][arm]['entries'][0]['p_continue'] = 1.+np.finfo(np.float32).eps
    corrected=REPORT.paired_states(pairs,arms)[0]
    assert corrected['probability']==1.
    assert corrected['probability_roundoff_correction']==np.finfo(np.float32).eps
    # Numerical range handling must not weaken exact pre-entry reproducibility.
    for arm in ('stop','continue_'):
        arms[(8,1,1)][arm]['entries'][0]['p_continue']=1.
    with pytest.raises(ValueError,match='prediction changed'):
        REPORT.paired_states(pairs,arms)
    arms[(8,1,1)]['continue_']['entries'][0]['hash'] = 'different'
    with pytest.raises(ValueError,match='states differ'):
        REPORT.paired_states(pairs,arms)


def test_continue_accounts_for_multiple_queries_before_real_action():
    arms=dict(stop=dict(returns=[arm_return(1.,0,.2)]),
              continue_=dict(returns=[arm_return(1.5,3,.6)]))
    row=dict(stream=0,delta_task=.5,delta_net=.2,
             delta_all_future_cost_net=.1,pair_task_tail_bound=.01)
    result=REPORT.paired_return_components(row,arms)
    assert result['additional_current_stage_computations'] == 3
    assert result['additional_current_stage_cost'] == pytest.approx(.3)
    assert result['task_difference_exceeds_tail_bound']
    row['delta_net']=.4
    with pytest.raises(ValueError,match='disagrees with arm returns'):
        REPORT.paired_return_components(row,arms)
    row['delta_net']=.2
    arms['continue_']['returns'][0]['stage_computations']=1
    with pytest.raises(ValueError,match='cost differs from query count'):
        REPORT.paired_return_components(row,arms)


def test_allocation_distinguishes_stream_effect_from_within_stream_variation():
    rows = [dict(seed=1,stream=stream,decision=decision,length=1+stream,
        first_optional_probability=.1+.8*stream,root_value=decision,root_margin=.5,
        sign_changes=0) for stream in [0,1] for decision in range(4)]
    report = REPORT.allocation_report(rows)
    assert report['first_probability_sd'] == pytest.approx(.4)
    assert report['first_probability_within_stream_sd'] == pytest.approx(0.)
    assert report['observed_decision_range'] == [0,3]
    with pytest.raises(ValueError,match='duplicate'):
        REPORT.allocation_report(rows+[rows[0]])


def test_allocation_preserves_later_change_in_planning_distribution():
    rows = [dict(seed=1,stream=0,decision=decision,length=1 if decision<384 else 30,
        first_optional_probability=.01 if decision<384 else .99,
        root_value=decision,root_margin=.5,sign_changes=0)
        for decision in range(128,640)]
    report = REPORT.allocation_report(rows)
    assert report['observed_decision_range'] == [128,639]
    first,last=report['by_decision_window']
    assert first['decisions']==last['decisions']==256
    assert first['min_fraction']==1. and first['cap_fraction']==0.
    assert last['min_fraction']==0. and last['cap_fraction']==1.


def test_allocation_accepts_saved_softmax_sum_roundoff_and_reports_it():
    rows=[dict(seed=1,stream=0,decision=decision,length=30,
        first_optional_probability=1.+np.finfo(np.float32).eps,
        root_value=0.,root_margin=.5,sign_changes=0) for decision in range(3)]
    report=REPORT.allocation_report(rows)
    assert report['probability_roundoff']['corrected_values']==3
    assert report['initial_gate_constant_null']['expected_cap_fraction']==1.
    assert report['initial_gate_constant_null']['expected_mean_length']==30.


def test_shared_time_and_stream_patterns_do_not_masquerade_as_scene_association():
    rows = [dict(seed=1,stream=stream,decision=decision,length=1,
        first_optional_probability=.1+.1*stream+.05*decision,
        root_value=10*stream+decision,root_margin=.5,sign_changes=0)
        for stream in range(3) for decision in range(5)]
    report=REPORT.allocation_report(rows)
    assert report['first_probability_after_stream_and_decision_adjustment_sd'] == pytest.approx(0.,abs=1e-12)
    assert report['associations']['root_value']['stream_and_decision_adjusted_pearson_with_first_probability'] is None


def test_time_adjustment_retains_interaction_and_handles_missing_rows():
    streams=np.array([0,0,0,1,1,1,2,2])
    decisions=np.array([0,1,2,0,1,2,0,2])
    signal=np.array([1.,-1.,0.,-1.,1.,0.,0.,0.])
    values=100*streams+10*decisions+signal
    residual=REPORT.remove_stream_and_decision_effects(values,streams,decisions)
    np.testing.assert_allclose(residual,signal,atol=1e-10)
    for factor in (streams,decisions):
        for level in np.unique(factor):
            assert abs(residual[factor==level].mean()) < 1e-10
