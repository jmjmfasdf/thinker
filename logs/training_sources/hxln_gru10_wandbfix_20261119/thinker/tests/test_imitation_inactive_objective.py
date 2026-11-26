import math

import pytest
import torch
import torch.nn.functional as F

from thinker.dynamic_imitation import compute_imitation_objective


@pytest.mark.parametrize("magnitude", [3.0, 1e15, 1e20, 1e30])
def test_ce_only_ignores_pvp_graph_but_keeps_finite_diagnostic(magnitude):
    logits = torch.tensor([[magnitude, -magnitude, 0., 0., 0., 0.]],
                          requires_grad=True)
    target = torch.tensor([2])
    result = compute_imitation_objective(
        logits, target, margin_coef=0., pvp_coef=0.)
    expected = F.cross_entropy(logits, target) / math.log(6)
    torch.testing.assert_close(result["loss"], expected)
    actual_grad, = torch.autograd.grad(result["loss"], logits, retain_graph=True)
    expected_grad, = torch.autograd.grad(expected, logits)
    torch.testing.assert_close(actual_grad, expected_grad)
    assert torch.isfinite(result["pvp_loss"])
    assert result["pvp_loss"].item() > 0
    assert not result["pvp_loss"].requires_grad
    assert not result["margin_loss"].requires_grad


def test_enabled_pvp_still_contributes_to_loss_and_gradient():
    logits = torch.tensor([[3., 1., -2.]], requires_grad=True)
    result = compute_imitation_objective(
        logits, torch.tensor([1]), margin_coef=0., pvp_coef=2.)
    expected = result["normalized_ce"] + 2. * ((logits[0, 1]-1).square()
                                               + (logits[0, 0]+1).square())
    torch.testing.assert_close(result["loss"], expected)
    actual_grad, = torch.autograd.grad(result["loss"], logits, retain_graph=True)
    expected_grad, = torch.autograd.grad(expected, logits)
    torch.testing.assert_close(actual_grad, expected_grad)
    assert result["pvp_loss"].requires_grad


def test_all_disabled_returns_differentiable_zero_without_zero_times_loss():
    logits = torch.tensor([[1e30, -1e30]], requires_grad=True)
    result = compute_imitation_objective(
        logits, torch.tensor([1]), overall_coef=0.)
    result["loss"].backward()
    assert result["loss"].item() == 0.
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))
