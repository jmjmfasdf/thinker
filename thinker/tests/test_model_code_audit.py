"""Independent arithmetic and gradient checks for the current dynamic model."""
import copy

import pytest
import torch

from thinker import util
from thinker.cenv import cModelWrapper
from thinker.learn_actor import SActorLearner
from thinker.voc_gru_gradient import SelectiveVOCGradient
from tests.test_dynamic_cenv import FakeModel, FakeVectorEnv, _flags, _step
from tests.test_dynamic_learner_integration import _NullWriter, _rollout
from tests.test_dynamic_voc_learner import _exact_projection_learner_flags


class ArithmeticModel(FakeModel):
    def __init__(self, num_actions, terminal=False):
        super().__init__(num_actions)
        self.terminal = terminal
        self.single_calls = []

    def __call__(self, env_state, done, actions, state):
        out = super().__call__(env_state, done, actions, state)
        out.vs = 10 * out.state["core"].reshape(1, -1, 1)
        return out

    def forward_single(self, state, action):
        self.single_calls.append(state["core"].clone())
        out = super().forward_single(state, action)
        out.state = {"core": state["core"] + action.float() + 1}
        out.rs = action.float().reshape(1, -1, 1) + 1
        out.vs = 10 * out.state["core"].reshape(1, -1, 1)
        out.dones.fill_(self.terminal)
        return out


def test_tree_returns_match_discounted_rewards_and_refresh_carried_branch():
    env = FakeVectorEnv(env_n=1)
    model = ArithmeticModel(env.num_actions)
    flags = _flags(tree_carry=True)
    flags.discounting = 0.9
    wrapper = cModelWrapper(env, 1, flags, model, device=torch.device("cpu"))
    schema = util.slice_dynamic_tree_reps(env.num_actions)
    try:
        wrapper.reset(model)
        state, *_ = _step(wrapper, model, [1], [util.PROCEED])
        first_return = 2 + 0.9 * 20
        assert state["tree_reps"][0, schema["root_qs_mean"]][1].item() == pytest.approx(first_return)
        state, *_ = _step(wrapper, model, [2], [util.PROCEED])
        second_return = 2 + 0.9 * 3 + 0.9**2 * 50
        assert state["tree_reps"][0, schema["root_qs_mean"]][1].item() == pytest.approx((first_return + second_return) / 2)
        assert state["tree_reps"][0, schema["root_qs_max"]][1].item() == pytest.approx(second_return)
        _step(wrapper, model, [0], [util.STOP])
        # Deliberately make the observed next state disagree with imagination.
        env.values[0, 0] = 100
        state, *_, info = _step(wrapper, model, [1], [util.PROCEED])
        assert info["useful_carry"].item()
        assert state["tree_reps"][0, schema["root_v"]].item() == pytest.approx(1020)
        state, *_ = _step(wrapper, model, [2], [util.PROCEED])
        assert model.single_calls[-1].item() == 102
        refreshed_return = 3 + 0.9 * 1050
        assert state["tree_reps"][0, schema["root_qs_mean"]][2].item() == pytest.approx(refreshed_return)
    finally:
        wrapper.close()


def test_predicted_terminal_removes_bootstrap_and_future_rewards():
    env = FakeVectorEnv(env_n=1)
    model = ArithmeticModel(env.num_actions, terminal=True)
    flags = _flags()
    flags.model_done_loss_cost = 1.0
    wrapper = cModelWrapper(env, 1, flags, model, device=torch.device("cpu"))
    schema = util.slice_dynamic_tree_reps(env.num_actions)
    try:
        wrapper.reset(model)
        state, *_ = _step(wrapper, model, [1], [util.PROCEED])
        assert state["tree_reps"][0, schema["root_qs_mean"]][1].item() == 2
        state, *_ = _step(wrapper, model, [4], [util.PROCEED])
        assert len(model.single_calls) == 1
        assert state["tree_reps"][0, schema["root_qs_mean"]][1].item() == 2
    finally:
        wrapper.close()


def test_production_selective_voc_gradient_equals_live_autograd(tmp_path, monkeypatch):
    """Actual ActorNet + learner loss, with an independently attached Q head."""
    torch.manual_seed(20261113)
    flags = _exact_projection_learner_flags(tmp_path, "gradient-audit")
    flags.min_search_steps = 1
    flags.think_cost = 0.0001
    flags.voc_gate_q_temperature = 0.01
    flags.dynamic_search_state_reset = "episode"
    actor, batch, initial_state = _rollout(flags)
    monkeypatch.setattr("thinker.learn_actor.FileWriter", lambda **kwargs: _NullWriter())
    learner = SActorLearner(ray_obj=None, actor_param={}, flags=flags, actor_net=actor, device=torch.device("cpu"))
    reference = copy.deepcopy(actor)
    reference_learner = SActorLearner(ray_obj=None, actor_param={}, flags=flags, actor_net=reference, device=torch.device("cpu"))
    # A fresh zero Q head has zero input gradient by construction. Use a
    # nonzero online head while keeping the common EMA/gate starting policy.
    with torch.no_grad():
        actor.voc_head.weight.normal_(std=0.01)
    reference.load_state_dict(actor.state_dict())
    live = []
    capture = reference.policy.register_forward_pre_hook(lambda module, args: live.append(args[0]))
    connect = reference.voc_head.register_forward_pre_hook(lambda module, args: (live[-1],))
    try:
        reference_losses, _ = reference_learner.compute_losses(batch, initial_state)
        parameters = tuple(reference.tree_rep_encoder.parameters())
        expected = torch.autograd.grad(reference_losses["_voc_total_loss"], parameters, allow_unused=True)
    finally:
        capture.remove()
        connect.remove()
    adapter = SelectiveVOCGradient(learner, True)
    try:
        losses, _ = learner.compute_losses(batch, initial_state)
        assert any(g is not None and g.abs().sum() > 0 for g in adapter.pending)
        for actual, target in zip(adapter.pending, expected):
            if target is None:
                assert actual is None
            else:
                torch.testing.assert_close(actual, target, rtol=1e-5, atol=1e-7)
        # The ordinary detached Q graph reaches its head/leaf but does not
        # directly modify any other actor parameters.
        losses["_voc_total_loss"].backward()
        assert all(p.grad is None for n, p in actor.named_parameters() if not n.startswith("voc_head."))
    finally:
        adapter.close()
