from types import SimpleNamespace

import torch

from thinker import util
from thinker.actor_net import (
    compute_discrete_log_prob,
    compute_dynamic_control_entropy,
    compute_dynamic_control_log_probs,
)
from thinker.learn_actor import dynamic_factorized_policy_log_probs


def test_gate_bout_log_prob_exactly_reconstructs_three_way_joint():
    logits = torch.tensor(
        [[
            [0.3, -0.2, 0.7],
            [-0.8, 1.1, 0.2],
            [0.4, 0.5, -0.3],
            [-0.6, 0.1, 0.8],
        ]],
        dtype=torch.float64,
    )
    actions = torch.tensor(
        [[util.PROCEED, util.RESET, util.STOP, util.PROCEED]],
        dtype=torch.long,
    )
    valid = torch.tensor([[True, True, True, False]])

    parts = compute_dynamic_control_log_probs(logits, actions, valid)
    legacy_joint = compute_discrete_log_prob(logits, actions)
    legacy_joint = torch.where(valid, legacy_joint, torch.zeros_like(legacy_joint))

    torch.testing.assert_close(parts.joint, legacy_joint, rtol=1e-12, atol=1e-12)
    assert parts.bout[0, 2].item() == 0.0
    assert parts.gate[0, 2].item() != 0.0
    assert parts.bout[0, 3].item() == 0.0
    assert parts.gate[0, 3].item() == 0.0


def test_gate_policy_and_entropy_use_only_common_continue_shift():
    logits = torch.tensor(
        [[1.7, -0.6, 0.2], [-1.1, 0.8, 0.4]],
        dtype=torch.float64,
        requires_grad=True,
    )
    actions = torch.tensor([util.PROCEED, util.STOP])
    policy_gate = compute_dynamic_control_log_probs(logits, actions).gate.sum()
    policy_grad = torch.autograd.grad(
        policy_gate, logits, retain_graph=True
    )[0]
    torch.testing.assert_close(
        policy_grad[..., util.PROCEED],
        policy_grad[..., util.RESET],
        rtol=1e-12,
        atol=1e-12,
    )
    assert torch.count_nonzero(policy_grad[..., util.STOP]) > 0

    gate_entropy = compute_dynamic_control_entropy(logits).gate.sum()
    entropy_grad = torch.autograd.grad(gate_entropy, logits)[0]
    torch.testing.assert_close(
        entropy_grad[..., util.PROCEED],
        entropy_grad[..., util.RESET],
        rtol=1e-12,
        atol=1e-12,
    )
    assert torch.count_nonzero(entropy_grad[..., util.PROCEED]) > 0


def test_factorized_entropy_matches_joint_value_but_detaches_gate_mixture():
    logits = torch.tensor(
        [[0.6, -0.4, 0.2], [-0.1, 0.9, -0.7]],
        dtype=torch.float64,
        requires_grad=True,
    )
    parts = compute_dynamic_control_entropy(logits)
    joint_log_probs = torch.log_softmax(logits, dim=-1)
    joint_probs = joint_log_probs.exp()
    joint_entropy = -(joint_probs * joint_log_probs).sum(dim=-1)

    torch.testing.assert_close(
        parts.gate + parts.continue_prob * parts.bout,
        joint_entropy,
        rtol=1e-12,
        atol=1e-12,
    )

    conditional_entropy = (parts.continue_prob.detach() * parts.bout).sum()
    conditional_grad = torch.autograd.grad(conditional_entropy, logits)[0]
    # Conditional bout entropy cannot push on STOP or on the aggregate
    # CONTINUE-versus-STOP direction.
    assert torch.count_nonzero(conditional_grad[..., util.STOP]) == 0
    torch.testing.assert_close(
        conditional_grad[..., :2].sum(dim=-1),
        torch.zeros(logits.shape[:-1], dtype=logits.dtype),
        rtol=1e-12,
        atol=1e-12,
    )


def test_max_depth_mask_and_wait_rows_remain_finite():
    logits = torch.tensor(
        [[[-1e9, 0.4, -0.2], [-1e9, 0.4, -0.2], [-1e9, 0.4, -0.2]]],
        dtype=torch.float64,
    )
    actions = torch.tensor(
        [[util.RESET, util.STOP, util.PROCEED]], dtype=torch.long
    )
    valid = torch.tensor([[True, True, False]])

    parts = compute_dynamic_control_log_probs(logits, actions, valid)
    entropy = compute_dynamic_control_entropy(logits)
    expected = compute_discrete_log_prob(logits, actions)
    expected = torch.where(valid, expected, torch.zeros_like(expected))

    for value in (*parts, *entropy):
        assert torch.isfinite(value).all()
    torch.testing.assert_close(parts.joint, expected, rtol=1e-12, atol=1e-12)
    assert parts.gate[0, 2].item() == 0.0
    assert parts.bout[0, 2].item() == 0.0


def _actor_out(control_logits, primary_logits, controls, primary_actions,
               control_valid, primary_valid):
    control_joint = compute_discrete_log_prob(control_logits, controls)
    control_joint = torch.where(
        control_valid, control_joint, torch.zeros_like(control_joint)
    )
    primary_joint = compute_discrete_log_prob(primary_logits, primary_actions)
    primary_joint = torch.where(
        primary_valid, primary_joint, torch.zeros_like(primary_joint)
    )
    return SimpleNamespace(
        search_control_logits=control_logits,
        reset_logits=control_logits,
        search_control=controls,
        reset=controls,
        pri_param=primary_logits,
        pri=primary_actions,
        c_action_log_prob=control_joint + primary_joint,
    )


def test_prefix_rhos_isolate_gate_and_keep_wait_at_identity():
    controls = torch.tensor(
        [[util.PROCEED, util.RESET, util.STOP, util.PROCEED]]
    )
    control_valid = torch.tensor([[True, True, True, False]])
    primary_valid = torch.tensor([[True, True, False, False]])
    primary_actions = torch.tensor([[[0], [1], [0], [1]]])
    behavior_control = torch.zeros(1, 4, 3, dtype=torch.float64)
    target_control = behavior_control.clone()
    target_control[..., util.STOP] = 1.25
    primary_logits = torch.tensor(
        [[[[0.3, -0.2]], [[-0.1, 0.4]], [[0.0, 0.0]], [[0.2, 0.1]]]],
        dtype=torch.float64,
    )

    behavior = dynamic_factorized_policy_log_probs(
        _actor_out(
            behavior_control,
            primary_logits,
            controls,
            primary_actions,
            control_valid,
            primary_valid,
        ),
        control_valid,
        primary_valid,
        discrete_action=True,
    )
    target = dynamic_factorized_policy_log_probs(
        _actor_out(
            target_control,
            primary_logits,
            controls,
            primary_actions,
            control_valid,
            primary_valid,
        ),
        control_valid,
        primary_valid,
        discrete_action=True,
    )
    for prefix_log_probs in (behavior, target):
        torch.testing.assert_close(
            prefix_log_probs["re"], prefix_log_probs["cur"],
            rtol=1e-12, atol=1e-12,
        )
        torch.testing.assert_close(
            prefix_log_probs["re"],
            prefix_log_probs["think"] + prefix_log_probs["im"],
            rtol=1e-12,
            atol=1e-12,
        )
    rhos = {name: target[name] - behavior[name] for name in behavior}

    # Changing only the STOP logit changes the gate (and the full joint), but
    # cannot leak into conditional imaginary-action/bout likelihoods.
    torch.testing.assert_close(rhos["im"], torch.zeros_like(rhos["im"]))
    assert behavior["im"][0, 2].item() == 0.0
    assert target["im"][0, 2].item() == 0.0
    assert torch.count_nonzero(rhos["think"][:, :3]) == 3
    assert torch.count_nonzero(rhos["re"][:, :3]) == 3
    for prefix in ("re", "im", "cur", "think"):
        assert rhos[prefix][0, 3].item() == 0.0


def test_primary_only_change_is_invisible_to_think_rho():
    controls = torch.tensor([[util.PROCEED, util.RESET, util.STOP]])
    control_valid = torch.ones(1, 3, dtype=torch.bool)
    primary_valid = torch.tensor([[True, True, False]])
    primary_actions = torch.tensor([[[0], [1], [0]]])
    control_logits = torch.tensor(
        [[[0.7, -0.1, 0.2], [-0.2, 0.8, 0.1], [0.3, 0.4, -0.5]]],
        dtype=torch.float64,
    )
    behavior_primary = torch.zeros(1, 3, 1, 2, dtype=torch.float64)
    target_primary = behavior_primary.clone()
    target_primary[..., 0] = 0.9

    def prefix(primary_logits):
        return dynamic_factorized_policy_log_probs(
            _actor_out(
                control_logits,
                primary_logits,
                controls,
                primary_actions,
                control_valid,
                primary_valid,
            ),
            control_valid,
            primary_valid,
            discrete_action=True,
        )

    behavior = prefix(behavior_primary)
    target = prefix(target_primary)
    rhos = {name: target[name] - behavior[name] for name in behavior}

    assert torch.count_nonzero(rhos["think"]) == 0
    assert torch.count_nonzero(rhos["im"][:, :2]) == 2
    torch.testing.assert_close(rhos["re"][:, :2], rhos["im"][:, :2])
    assert rhos["re"][0, 2].item() == 0.0
    assert rhos["im"][0, 2].item() == 0.0


def test_continuous_primary_factorization_handles_tanh_and_exact_masks():
    control_logits = torch.tensor(
        [[[0.7, -0.2, 0.1], [-0.4, 0.2, 0.8], [0.3, -0.1, 0.4]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    controls = torch.tensor([[util.PROCEED, util.STOP, util.PROCEED]])
    control_valid = torch.tensor([[True, True, False]])
    primary_valid = torch.tensor([[True, False, False]])

    for tanh_action in (False, True):
        pri_param = torch.tensor(
            [[[[0.1, -0.4]], [[-0.2, 0.3]], [[0.4, -0.1]]]],
            dtype=torch.float64,
            requires_grad=True,
        )
        pre_tanh = torch.tensor([[[0.25], [-0.35], [0.15]]], dtype=torch.float64)
        pri = torch.tanh(pre_tanh) if tanh_action else pre_tanh
        actor_out = SimpleNamespace(
            search_control_logits=control_logits,
            reset_logits=control_logits,
            search_control=controls,
            reset=controls,
            pri_param=pri_param,
            pri=pri,
            c_action_log_prob=torch.zeros(1, 3, dtype=torch.float64),
        )
        prefix = dynamic_factorized_policy_log_probs(
            actor_out,
            control_valid,
            primary_valid,
            discrete_action=False,
            tanh_action=tanh_action,
        )

        mean = pri_param[..., 0]
        log_var = pri_param[..., 1]
        expected = torch.distributions.Normal(
            mean, torch.exp(log_var / 2)
        ).log_prob(pre_tanh)
        if tanh_action:
            expected = expected - torch.log(1.0 - pri ** 2 + 1e-6)
        expected = expected.sum(dim=-1)
        expected = torch.where(
            primary_valid, expected, torch.zeros_like(expected)
        )
        expected_bout = compute_dynamic_control_log_probs(
            control_logits, controls, control_valid
        ).bout
        torch.testing.assert_close(prefix["im"], expected + expected_bout)
        assert prefix["im"][0, 1].item() == 0.0
        assert prefix["im"][0, 2].item() == 0.0
        think_primary_grad = torch.autograd.grad(
            prefix["think"].sum(), pri_param, allow_unused=True
        )[0]
        assert think_primary_grad is None or torch.count_nonzero(
            think_primary_grad
        ) == 0
