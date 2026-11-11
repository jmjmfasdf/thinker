import pytest
import torch

from thinker.numeric_stability import masked_squared_sum, stable_clip_grad_norm_, stable_grad_norm


def test_mask_precedes_square_and_excluded_gradient_is_zero():
    value = torch.tensor([[1e20, -1e20], [float('nan'), float('inf')]], requires_grad=True)
    loss = masked_squared_sum(value, mask=torch.tensor([True, False]))
    assert torch.isfinite(loss).all()
    assert loss[1].item() == 0.
    loss.sum().backward()
    torch.testing.assert_close(value.grad[0], 2 * value.detach()[0])
    torch.testing.assert_close(value.grad[1], torch.zeros(2))


def test_squared_sum_matches_normal_loss_and_gradient():
    value = torch.arange(12., requires_grad=True).view(2,2,3)
    actual = masked_squared_sum(value, dim=(-2,-1))
    expected = value.square().sum(dim=(-2,-1))
    torch.testing.assert_close(actual, expected.double())
    a, = torch.autograd.grad(actual.sum(), value, retain_graph=True)
    b, = torch.autograd.grad(expected.sum(), value)
    torch.testing.assert_close(a,b)


@pytest.mark.parametrize('magnitude',[1e20,1e38])
def test_large_finite_gradients_are_clipped_instead_of_overflowing(magnitude):
    p = torch.nn.Parameter(torch.ones(2))
    p.grad = torch.tensor([magnitude,-magnitude])
    assert torch.isfinite(stable_grad_norm([p]))
    norm = stable_clip_grad_norm_([p], 1.)
    assert torch.isfinite(norm)
    torch.testing.assert_close(p.grad, torch.tensor([2**-.5,-2**-.5]))


def test_nonfinite_gradient_remains_an_error_without_mutation():
    p = torch.nn.Parameter(torch.ones(2))
    p.grad = torch.tensor([1.,float('inf')])
    with pytest.raises(FloatingPointError):stable_clip_grad_norm_([p], 1.)
    assert p.grad[0].item()==1.
    assert torch.isinf(p.grad[1])
