import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

DIAGNOSTICS = Path(__file__).parents[1]/'diagnostics'
sys.path.insert(0, str(DIAGNOSTICS))
SPEC = importlib.util.spec_from_file_location('teacher_gap', DIAGNOSTICS/'audit_planning_gate_teacher_gap.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(decision, p, delta):
    return dict(seed=1, stream=0, decision=decision,
        first_optional_probability=p, first_optional_delta=delta)


def test_distinguishes_both_directions_of_disagreement():
    result, _ = MODULE.summarize([row(0,.1,.02), row(1,.9,-.02), row(2,.8,.03), row(3,.2,-.03)], .01)
    assert result['sign_mismatch_fraction'] == .5
    assert result['online_positive_gate_nonpositive_count'] == 1
    assert result['gate_positive_online_nonpositive_count'] == 1
    assert result['mean_online_probability_same_features'] == pytest.approx(.5)
    assert result['mean_gate_probability'] == pytest.approx(.5)


def test_saturation_is_finite_and_neutral_boundary_is_reported():
    result, _ = MODULE.summarize([row(0,1+1e-7,1e5),row(1,0,-1e5),row(2,.5,0)], .01)
    assert result['sign_mismatch_count'] == 0
    assert result['near_boundary_count'] == 1
    assert result['mean_absolute_probability_gap'] == 0
    np.testing.assert_array_equal(MODULE.sigmoid([-1e5,0,1e5]),[0,.5,1])


@pytest.mark.parametrize('rows', [[row(0,.5,0),row(0,.5,0)], [row(0,1.1,0)], [row(0,.5,float('nan'))]])
def test_rejects_invalid_evidence(rows):
    with pytest.raises(ValueError):
        MODULE.summarize(rows,.01)


def sample_checkpoint():
    import torch
    weight=torch.tensor([[.02,.01],[-.01,.005]])
    bias=torch.tensor([.003,-.002])
    return dict(voc_gate_policy_schema_version=4,
        flags=dict(voc_gate_exact_projection=True,voc_gate_epsilon_greedy_execution=False,
                   voc_gate_q_temperature=.01,voc_gate_temperature=1.),
        voc_ema_gate_head_state_dict=dict(weight=weight,bias=bias),
        actor_net_state_dict={'voc_head.weight':weight+.001,'voc_head.bias':bias+.001,
            'voc_gate_head.weight':100*(weight[0:1]-weight[1:2]),
            'voc_gate_head.bias':100*(bias[0:1]-bias[1:2])})


def test_checkpoint_projection_corruption_is_rejected():
    checkpoint=sample_checkpoint()
    MODULE.check_heads(checkpoint)
    checkpoint['actor_net_state_dict']['voc_gate_head.bias'] += .001
    with pytest.raises(ValueError,match='exactly'):
        MODULE.check_heads(checkpoint)


def test_feature_reconstruction_rejects_mismatched_recorded_probability():
    checkpoint=sample_checkpoint()
    x=np.array([.5,-.2])
    state=checkpoint['actor_net_state_dict']
    q=state['voc_head.weight'].numpy()@x+state['voc_head.bias'].numpy()
    gate=(state['voc_gate_head.weight'].numpy()@x+state['voc_gate_head.bias'].numpy())[0]
    entry=dict(features=x.tolist(),delta=float(q[0]-q[1]),p_continue=float(MODULE.sigmoid(gate)),count=1)
    reference=row(128,entry['p_continue'],entry['delta'])
    assert MODULE.check_entries([entry],checkpoint,[reference])['entries']==1
    with pytest.raises(ValueError,match='reconstruct'):
        MODULE.check_entries([dict(entry,p_continue=.1)],checkpoint,[reference])
