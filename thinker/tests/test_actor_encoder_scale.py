from types import SimpleNamespace

import pytest
import torch

from thinker.actor_net import AFrameEncoder
from thinker.planning_protocol import planning_training_contract, validate_planning_training_checkpoint


def test_legacy_encoder_parameter_keys_and_predictions_are_preserved():
    torch.manual_seed(701)
    old = AFrameEncoder((4,5,5), SimpleNamespace())
    explicit = AFrameEncoder((4,5,5), SimpleNamespace(), output_norm='none')
    explicit.load_state_dict(old.state_dict(), strict=True)
    assert not any('output_norm' in key for key in old.state_dict())
    x = torch.randn(3,4,5,5)
    torch.testing.assert_close(old(x), explicit(x), rtol=0, atol=0)


def test_normalization_removes_encoder_output_amplification_with_finite_gradients():
    torch.manual_seed(701)
    raw = AFrameEncoder((4,5,5), SimpleNamespace())
    normalized = AFrameEncoder((4,5,5), SimpleNamespace(), output_norm='layer')
    with torch.no_grad():
        raw.fc[0].weight.mul_(10000)
        raw.fc[0].bias.mul_(10000)
    missing, unexpected = normalized.load_state_dict(raw.state_dict(), strict=False)
    assert set(missing) == {'output_norm.weight', 'output_norm.bias'} and not unexpected
    x = torch.randn(3,4,5,5)
    before, after = raw(x), normalized(x)
    assert before.square().mean().sqrt() > 100
    torch.testing.assert_close(after.square().mean(-1), torch.ones(3), rtol=1e-5, atol=1e-5)
    (after*torch.randn_like(after)).sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in normalized.parameters())
    assert normalized.fc[0].weight.grad.abs().sum() > 0


def test_encoder_architecture_cannot_change_silently_on_resume():
    old = dict(voc_gru_connected=True)
    new = dict(old, actor_h_encoder_norm='layer')
    assert planning_training_contract(old)['version'] == 1
    contract = planning_training_contract(new)
    assert contract['version'] == 2 and contract['actor_h_encoder_norm'] == 'layer'
    checkpoint = dict(flags=new, planning_training_contract=contract)
    validate_planning_training_checkpoint(checkpoint, new)
    with pytest.raises(ValueError, match='different planning'):
        validate_planning_training_checkpoint(checkpoint, old)
    with pytest.raises(ValueError, match='actor_h_encoder_norm'):
        planning_training_contract(dict(old, actor_h_encoder_norm='invalid'))
