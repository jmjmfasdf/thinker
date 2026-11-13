from types import SimpleNamespace
import numpy as np
import torch
import pytest
from temperature_eval import GateIntervention, continue_probability


def test_temperature_sharpens_without_changing_sign_or_ties():
    delta = np.array([-.02, 0., .02])
    p = continue_probability(delta, .05, .02)
    low = continue_probability(delta, .01, .02)
    assert low[0] < p[0] < .5 < p[2] < low[2]
    assert p[1] == low[1] == .5
    assert np.allclose(continue_probability([-1e8, 1e8], .01, .02), [.01, .99])
    assert np.allclose(continue_probability(delta, .01, 1.), .5)


def test_hook_changes_only_readout_and_preserves_conditional_policy():
    ema = {'weight': torch.tensor([[.2, -.3], [.1, .2]]), 'bias': torch.tensor([.05, -.01])}
    head = torch.nn.Linear(2, 1)
    with torch.no_grad():
        head.weight.copy_((ema['weight'][0]-ema['weight'][1]).unsqueeze(0)/.05)
        head.bias.copy_((ema['bias'][0]-ema['bias'][1]).reshape(1)/.05)
    actor = SimpleNamespace(voc_gate_head=head, flags=SimpleNamespace(voc_gate_exact_projection=True, voc_gate_temperature=1.))
    x = torch.tensor([[.1, .3], [.7, -.2]])
    original = head(x).detach().clone(); saved = {k: v.clone() for k, v in head.state_dict().items()}
    h = GateIntervention(actor, ema, .05, .01)
    assert torch.allclose(head(x), original*5)
    assert torch.allclose(h.delta, original.flatten()*.05)
    h.close()
    assert torch.equal(head(x).detach(), original)
    assert all(torch.equal(v, saved[k]) for k, v in head.state_dict().items())
    from thinker.actor_net import compute_voc_gate_distribution
    logits = torch.tensor([[1., 2., 0.], [3., -1., .4]])
    # Check API outputs by name: scaling the gate must leave P(PROCEED|CONTINUE) unchanged.
    a = compute_voc_gate_distribution(logits, raw_gate_log_odds=original.flatten(), epsilon=.02)
    b = compute_voc_gate_distribution(logits, raw_gate_log_odds=original.flatten()*5, epsilon=.02)
    assert torch.equal(a.bout_logits, b.bout_logits)
    assert torch.allclose(torch.softmax(a.joint_logits[:, :2], -1),
                          torch.softmax(b.joint_logits[:, :2], -1))


def test_rejects_unprojected_gate():
    actor = SimpleNamespace(voc_gate_head=torch.nn.Linear(2, 1),
                            flags=SimpleNamespace(voc_gate_exact_projection=False, voc_gate_temperature=1.))
    with pytest.raises(ValueError): GateIntervention(actor, {}, .05, .01)
