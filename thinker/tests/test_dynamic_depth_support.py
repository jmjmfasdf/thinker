import pytest
import torch

from thinker import util
from thinker.actor_net import compute_voc_gate_distribution, sample
from test_dynamic_actor import _flags, _network
from test_dynamic_cenv import FakeVectorEnv, FakeModel
from thinker.cenv import cModelWrapper


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable')
    return torch.device(request.param)


@pytest.mark.parametrize('reset_logit', [-2e9, -1e12, 2e9])
def test_depth_boundary_has_exact_reset_support_at_any_finite_scale(reset_logit, device):
    logits = torch.tensor([[50., reset_logit, 0.]], requires_grad=True, device=device)
    distribution = compute_voc_gate_distribution(
        logits, legal_control_mask=torch.tensor([[False, True, True]], device=device),
        raw_gate_log_odds=torch.tensor([0.], device=device), epsilon=.02,
    )
    assert torch.equal(torch.softmax(distribution.bout_logits, -1), torch.tensor([[0., 1.]], device=device))
    assert sample(distribution.bout_logits, greedy=False).item() == util.RESET
    joint = torch.softmax(distribution.joint_logits, -1)
    torch.testing.assert_close(joint, torch.tensor([[0., .5, .5]], device=device))
    (-distribution.joint_logits[0, 1]).backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.equal(logits.grad, torch.zeros_like(logits.grad))


@pytest.mark.parametrize('mask', [[True, False, False], [False, True, False], [False, False, True]])
@pytest.mark.parametrize('gate', [-1e12, 1e12])
def test_singleton_gate_is_deterministic_even_below_the_mask_sentinel(mask, gate, device):
    distribution = compute_voc_gate_distribution(
        torch.tensor([[-2e12, -1e12, 0.]], device=device),
        legal_control_mask=torch.tensor([mask], device=device),
        raw_gate_log_odds=torch.tensor([gate], device=device), epsilon=.02,
    )
    torch.testing.assert_close(torch.softmax(distribution.joint_logits, -1), torch.tensor([mask], device=device).float())


def test_actor_to_cenv_depth30_with_large_finite_control_logits(device):
    flags = _flags(True, max_search_steps=30, max_depth=30, rec_t=20,
                   factorized=True, voc_mode='control', voc_dedicated_gate=True)
    flags.im_cost = 0.
    env = FakeVectorEnv()
    model = FakeModel(env.num_actions)
    wrapper = cModelWrapper(env, env.env_n, flags, model, device=device)
    state, info = wrapper.reset(model)
    for _ in range(29):
        state, reward, done, truncated, info = wrapper.step((torch.zeros(3).long(), torch.zeros(3).long()), model)
    assert info['legal_control_mask'].tolist() == [[False, True, True]] * 3
    actor = _network(flags).to(device)
    with torch.no_grad():
        actor.reset.weight.zero_()
        actor.reset.bias.copy_(torch.tensor([0., -2e9, 0.]))
        actor.voc_gate_head.weight.zero_()
        actor.voc_gate_head.bias.fill_(20.)
    env_out = util.init_env_out(state, info, flags, 1, False)
    output, _ = actor(env_out, actor.initial_state(3, device=device), greedy=True)
    _, _, _, _, next_info = wrapper.step(output.action, model)
    assert output.action[1].tolist() == [util.RESET] * 3
    assert next_info['phase'].tolist() == [util.NEED_REAL_ACTION_PHASE] * 3
    assert next_info['forced_stop'].all()
    wrapper.close()
