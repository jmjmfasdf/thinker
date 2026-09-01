import copy
import math
import os

import numpy as np
import pytest
import torch
from torch import nn

from thinker import util
import thinker.learn_actor as learn_actor_module
from thinker.actor_net import compute_voc_gate_distribution
from thinker.learn_actor import (
    ActorGradientStepResult,
    SActorLearner,
    compute_dynamic_voc_loss,
    compute_dynamic_voc_gate_parameter_alignment_loss,
    compute_dynamic_voc_soft_q_gate_loss,
    compute_dynamic_voc_target,
    detach_dynamic_voc_gate_from_joint_logits,
    dynamic_voc_holdout_mask,
    dynamic_voc_observability_metrics,
    dynamic_voc_policy_log_probs,
    project_dynamic_voc_gate_head_exact_,
    resolve_dynamic_voc_learning_control_surface,
)
from tests.test_dynamic_cenv import _flags as cenv_flags
from tests.test_dynamic_learner_integration import _NullWriter, _rollout


def test_recursive_voc_target_combines_environment_return_and_think_cost():
    target = compute_dynamic_voc_target(
        task_rewards=torch.tensor([[0.0], [0.0], [2.0]]),
        think_rewards=torch.tensor([[-1.0], [-1.0], [0.0]]),
        task_discounts=torch.tensor([[1.0], [1.0], [0.0]]),
        think_discounts=torch.tensor([[1.0], [1.0], [0.0]]),
        task_vs=torch.tensor([[10.0], [20.0], [30.0]]),
        think_vs=torch.tensor([[-3.0], [-2.0], [0.0]]),
        task_bootstrap_value=torch.tensor([40.0]),
        think_bootstrap_value=torch.tensor([0.0]),
        think_cost=0.5,
    )
    torch.testing.assert_close(
        target.task, torch.tensor([[20.0], [30.0], [2.0]])
    )
    torch.testing.assert_close(
        target.think, torch.tensor([[-3.0], [-1.0], [0.0]])
    )
    torch.testing.assert_close(
        target.net, torch.tensor([[18.5], [29.5], [2.0]])
    )


def test_epsilon_greedy_behavior_logits_drive_vtrace_rho_not_soft_calibration():
    raw_logits = torch.zeros((1, 1, 3))
    soft_target = compute_voc_gate_distribution(
        raw_logits,
        epsilon=0.02,
        raw_gate_log_odds=torch.full((1, 1), 0.4),
    )
    target_execution = compute_voc_gate_distribution(
        raw_logits,
        epsilon=0.02,
        raw_gate_log_odds=torch.ones((1, 1)),
        epsilon_greedy_execution=True,
    )
    behavior = compute_voc_gate_distribution(
        raw_logits,
        epsilon=0.02,
        raw_gate_log_odds=-torch.ones((1, 1)),
        epsilon_greedy_execution=True,
    )
    result = compute_dynamic_voc_loss(
        voc_q=torch.zeros((1, 1, 2), requires_grad=True),
        target_control_logits=soft_target.joint_logits,
        target_behavior_control_logits=target_execution.joint_logits,
        behavior_control_logits=behavior.joint_logits,
        control_action=torch.tensor([[util.STOP]]),
        control_valid=torch.tensor([[True]]),
        voc_target=torch.zeros((1, 1)),
        mode="shadow",
    )

    torch.testing.assert_close(
        result.continue_probability, soft_target.continue_prob
    )
    torch.testing.assert_close(
        result.behavior_continue_probability, torch.tensor([[0.01]])
    )
    torch.testing.assert_close(
        result.gate_rho, torch.tensor([[0.01 / 0.99]])
    )


def test_execution_surface_cannot_change_soft_centered_q_or_q_loss():
    raw_logits = torch.tensor(
        [[[0.4, -0.2, 0.1], [-0.3, 0.7, -0.4]]]
    )
    soft_target = compute_voc_gate_distribution(
        raw_logits,
        epsilon=0.02,
        raw_gate_log_odds=torch.tensor([[0.35, -0.6]]),
    )
    positive_execution = compute_voc_gate_distribution(
        raw_logits,
        epsilon=0.02,
        raw_gate_log_odds=torch.ones((1, 2)),
        epsilon_greedy_execution=True,
    )
    negative_execution = compute_voc_gate_distribution(
        raw_logits,
        epsilon=0.02,
        raw_gate_log_odds=-torch.ones((1, 2)),
        epsilon_greedy_execution=True,
    )
    behavior = negative_execution
    common = dict(
        voc_q=torch.tensor([[[1.3, -0.7], [-0.4, 0.9]]]),
        target_control_logits=soft_target.joint_logits,
        behavior_control_logits=behavior.joint_logits,
        control_action=torch.tensor([[util.PROCEED, util.STOP]]),
        control_valid=torch.ones((1, 2), dtype=torch.bool),
        voc_target=torch.tensor([[0.8, -0.25]]),
        mode="shadow",
        dueling_q=True,
        voc_state_value=torch.tensor([[0.2, -0.1]]),
    )
    v11 = compute_dynamic_voc_loss(**common)
    v12_positive = compute_dynamic_voc_loss(
        **common,
        target_behavior_control_logits=positive_execution.joint_logits,
    )
    v12_negative = compute_dynamic_voc_loss(
        **common,
        target_behavior_control_logits=negative_execution.joint_logits,
    )

    for name in (
        "q_values",
        "q_loss",
        "selected_q",
        "td_error",
        "delta_q",
        "continue_probability",
    ):
        torch.testing.assert_close(
            getattr(v12_positive, name),
            getattr(v11, name),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            getattr(v12_negative, name),
            getattr(v11, name),
            rtol=0.0,
            atol=0.0,
        )
    assert not torch.equal(v12_positive.gate_rho, v12_negative.gate_rho)


def test_schema4_and_schema5_share_soft_ema_q_centering():
    raw_logits = torch.tensor(
        [[[0.2, -0.5, 0.1], [0.7, -0.1, -0.3]]]
    )
    soft = compute_voc_gate_distribution(
        raw_logits,
        epsilon=0.02,
        raw_gate_log_odds=torch.tensor([[0.45, -0.25]]),
    )
    positive_execution = compute_voc_gate_distribution(
        raw_logits,
        epsilon=0.02,
        raw_gate_log_odds=torch.ones((1, 2)),
        epsilon_greedy_execution=True,
    )
    negative_execution = compute_voc_gate_distribution(
        raw_logits,
        epsilon=0.02,
        raw_gate_log_odds=-torch.ones((1, 2)),
        epsilon_greedy_execution=True,
    )
    misc = {
        "voc_gate_soft_control_logits": soft.joint_logits,
        "voc_gate_soft_continue_probability": soft.continue_prob,
    }
    schema4_logits, schema4_probability = (
        resolve_dynamic_voc_learning_control_surface(
            execution_control_logits=soft.joint_logits,
            actor_misc=None,
            control_valid=torch.ones((1, 2), dtype=torch.bool),
            epsilon_greedy_execution=False,
        )
    )
    schema5_positive_logits, positive_probability = (
        resolve_dynamic_voc_learning_control_surface(
            execution_control_logits=positive_execution.joint_logits,
            actor_misc=misc,
            control_valid=torch.ones((1, 2), dtype=torch.bool),
            epsilon_greedy_execution=True,
        )
    )
    schema5_negative_logits, negative_probability = (
        resolve_dynamic_voc_learning_control_surface(
            execution_control_logits=negative_execution.joint_logits,
            actor_misc=misc,
            control_valid=torch.ones((1, 2), dtype=torch.bool),
            epsilon_greedy_execution=True,
        )
    )
    assert schema4_logits is soft.joint_logits
    assert schema4_probability is None
    assert schema5_positive_logits is soft.joint_logits
    assert schema5_negative_logits is soft.joint_logits
    torch.testing.assert_close(positive_probability, soft.continue_prob)
    torch.testing.assert_close(negative_probability, soft.continue_prob)

    learner = object.__new__(SActorLearner)
    learner.voc_ema_gate_weight = torch.tensor(
        [[0.4, -0.2, 0.7], [-0.3, 0.5, 0.1]], dtype=torch.float32
    )
    learner.voc_ema_gate_bias = torch.tensor(
        [0.15, -0.05], dtype=torch.float32
    )
    learner.dynamic_voc_mode = "control"
    common = dict(
        features=torch.tensor(
            [[[0.2, -0.4, 0.8], [0.5, 0.1, -0.7]]],
            dtype=torch.float32,
        ),
        valid=torch.ones((1, 2), dtype=torch.bool),
        state_value=torch.tensor([[0.6, -0.2]], dtype=torch.float32),
    )
    schema4_loss, schema4_q = learner._compute_ema_gate_loss(
        logits=schema4_logits, **common
    )
    positive_loss, positive_q = learner._compute_ema_gate_loss(
        logits=schema5_positive_logits, **common
    )
    negative_loss, negative_q = learner._compute_ema_gate_loss(
        logits=schema5_negative_logits, **common
    )
    for actual in (positive_q, negative_q):
        torch.testing.assert_close(
            actual, schema4_q, rtol=0.0, atol=0.0
        )
    for actual in (positive_loss, negative_loss):
        torch.testing.assert_close(
            actual, schema4_loss, rtol=0.0, atol=0.0
        )


@pytest.mark.parametrize(
    "corruption,expected_exception",
    [
        ("missing_logits", RuntimeError),
        ("missing_probability", RuntimeError),
        ("soft_shape", ValueError),
        ("soft_type", TypeError),
        ("soft_nonfinite", FloatingPointError),
        ("probability_shape", ValueError),
        ("probability_type", TypeError),
        ("probability_nonfinite", FloatingPointError),
        ("execution_shape", ValueError),
        ("execution_type", TypeError),
        ("execution_nonfinite", FloatingPointError),
    ],
)
def test_epsilon_execution_dual_surface_corruption_fails_closed(
    corruption, expected_exception
):
    valid = torch.ones((1, 2), dtype=torch.bool)
    raw_logits = torch.zeros((1, 2, 3))
    soft = compute_voc_gate_distribution(
        raw_logits,
        epsilon=0.02,
        raw_gate_log_odds=torch.tensor([[0.3, -0.4]]),
    )
    execution = compute_voc_gate_distribution(
        raw_logits,
        epsilon=0.02,
        raw_gate_log_odds=torch.ones((1, 2)),
        epsilon_greedy_execution=True,
    ).joint_logits.clone()
    misc = {
        "voc_gate_soft_control_logits": soft.joint_logits.clone(),
        "voc_gate_soft_continue_probability": soft.continue_prob.clone(),
    }

    if corruption == "missing_logits":
        del misc["voc_gate_soft_control_logits"]
    elif corruption == "missing_probability":
        del misc["voc_gate_soft_continue_probability"]
    elif corruption == "soft_shape":
        misc["voc_gate_soft_control_logits"] = misc[
            "voc_gate_soft_control_logits"
        ][..., :2]
    elif corruption == "soft_type":
        misc["voc_gate_soft_control_logits"] = torch.zeros(
            (1, 2, 3), dtype=torch.long
        )
    elif corruption == "soft_nonfinite":
        misc["voc_gate_soft_control_logits"][0, 0, 0] = float("nan")
    elif corruption == "probability_shape":
        misc["voc_gate_soft_continue_probability"] = misc[
            "voc_gate_soft_continue_probability"
        ][:, :1]
    elif corruption == "probability_type":
        misc["voc_gate_soft_continue_probability"] = torch.zeros(
            (1, 2), dtype=torch.long
        )
    elif corruption == "probability_nonfinite":
        misc["voc_gate_soft_continue_probability"][0, 0] = float("nan")
    elif corruption == "execution_shape":
        execution = execution[..., :2]
    elif corruption == "execution_type":
        execution = torch.zeros((1, 2, 3), dtype=torch.long)
    elif corruption == "execution_nonfinite":
        execution[0, 0, 0] = float("inf")
    else:  # pragma: no cover - parametrization is exhaustive.
        raise AssertionError(corruption)

    with pytest.raises(expected_exception):
        resolve_dynamic_voc_learning_control_surface(
            execution_control_logits=execution,
            actor_misc=misc,
            control_valid=valid,
            epsilon_greedy_execution=True,
        )


def test_gate_parameter_alignment_exactly_represents_frozen_soft_q_teacher():
    gate_weight = torch.tensor(
        [[0.5, -0.25, 0.75]], dtype=torch.float32, requires_grad=True
    )
    gate_bias = torch.tensor([0.125], dtype=torch.float32, requires_grad=True)
    ema_q_weight = torch.tensor(
        [[0.2, -0.1, 0.4], [-0.3, 0.5, 0.1]],
        dtype=torch.float32,
        requires_grad=True,
    )
    ema_q_bias = torch.tensor(
        [0.07, -0.02], dtype=torch.float32, requires_grad=True
    )
    q_temperature = 0.25
    policy_temperature = 0.5

    result = compute_dynamic_voc_gate_parameter_alignment_loss(
        gate_weight=gate_weight,
        gate_bias=gate_bias,
        ema_q_weight=ema_q_weight,
        ema_q_bias=ema_q_bias,
        q_temperature=q_temperature,
        policy_temperature=policy_temperature,
    )
    expected_weight = (
        policy_temperature
        / q_temperature
        * (ema_q_weight.detach()[0:1] - ema_q_weight.detach()[1:2])
    )
    expected_bias = (
        policy_temperature
        / q_temperature
        * (ema_q_bias.detach()[0:1] - ema_q_bias.detach()[1:2])
    )
    torch.testing.assert_close(result.target_weight, expected_weight)
    torch.testing.assert_close(result.target_bias, expected_bias)
    expected_error_sq = (
        (gate_weight - expected_weight).square().sum()
        + (gate_bias - expected_bias).square().sum()
    )
    torch.testing.assert_close(result.loss, 0.5 * expected_error_sq)
    assert result.loss.dtype == torch.float32

    features = torch.tensor(
        [[0.3, -0.8, 1.1], [-0.2, 0.4, 0.7]], dtype=torch.float32
    )
    teacher_q = torch.nn.functional.linear(
        features, ema_q_weight.detach(), ema_q_bias.detach()
    )
    teacher_log_odds = (
        teacher_q[:, 0] - teacher_q[:, 1]
    ) / q_temperature
    represented_raw_log_odds = torch.nn.functional.linear(
        features, result.target_weight, result.target_bias
    ).squeeze(-1)
    represented_log_odds = represented_raw_log_odds / policy_temperature
    torch.testing.assert_close(represented_log_odds, teacher_log_odds)
    torch.testing.assert_close(
        torch.sigmoid(represented_log_odds),
        torch.sigmoid(teacher_log_odds),
    )
    epsilon = 0.02
    behavior = compute_voc_gate_distribution(
        torch.zeros((features.shape[0], 3)),
        temperature=policy_temperature,
        epsilon=epsilon,
        raw_gate_log_odds=represented_raw_log_odds,
    )
    expected_behavior_continue = (
        (1.0 - epsilon) * torch.sigmoid(teacher_log_odds)
        + epsilon * 0.5
    )
    torch.testing.assert_close(
        behavior.continue_prob, expected_behavior_continue
    )

    # Mutating the source after construction cannot turn backward into a
    # post-Q lookahead target: the batch-start EMA difference is materialized.
    with torch.no_grad():
        ema_q_weight.fill_(100.0)
        ema_q_bias.fill_(-100.0)
    result.loss.backward()
    torch.testing.assert_close(gate_weight.grad, gate_weight - expected_weight)
    torch.testing.assert_close(gate_bias.grad, gate_bias - expected_bias)
    assert ema_q_weight.grad is None
    assert ema_q_bias.grad is None


def test_gate_parameter_alignment_preserves_exact_fresh_tie():
    gate_weight = torch.zeros((1, 4), requires_grad=True)
    gate_bias = torch.zeros(1, requires_grad=True)
    ema_q_weight = torch.zeros((2, 4), requires_grad=True)
    ema_q_bias = torch.zeros(2, requires_grad=True)

    result = compute_dynamic_voc_gate_parameter_alignment_loss(
        gate_weight=gate_weight,
        gate_bias=gate_bias,
        ema_q_weight=ema_q_weight,
        ema_q_bias=ema_q_bias,
        q_temperature=0.05,
        policy_temperature=1.0,
    )

    torch.testing.assert_close(result.loss, torch.tensor(0.0))
    torch.testing.assert_close(result.parameter_error_norm, torch.tensor(0.0))
    torch.testing.assert_close(
        result.relative_parameter_error, torch.tensor(0.0)
    )
    torch.testing.assert_close(result.relative_error_defined, torch.tensor(0.0))
    torch.testing.assert_close(result.cosine, torch.tensor(0.0))
    torch.testing.assert_close(result.cosine_defined, torch.tensor(0.0))
    result.loss.backward()
    torch.testing.assert_close(gate_weight.grad, torch.zeros_like(gate_weight))
    torch.testing.assert_close(gate_bias.grad, torch.zeros_like(gate_bias))
    assert ema_q_weight.grad is None
    assert ema_q_bias.grad is None


def test_exact_gate_projection_copies_fp32_parameter_target_bit_exactly():
    gate_weight = nn.Parameter(torch.tensor([[0.5, -0.25, 0.75]]))
    gate_bias = nn.Parameter(torch.tensor([0.125]))
    ema_q_weight = torch.tensor(
        [[0.2, -0.1, 0.4], [-0.3, 0.5, 0.1]], dtype=torch.float32
    )
    ema_q_bias = torch.tensor([0.07, -0.02], dtype=torch.float32)

    result = project_dynamic_voc_gate_head_exact_(
        gate_weight=gate_weight,
        gate_bias=gate_bias,
        ema_q_weight=ema_q_weight,
        ema_q_bias=ema_q_bias,
        q_temperature=0.25,
        policy_temperature=0.5,
    )

    expected_weight = 2.0 * (ema_q_weight[0:1] - ema_q_weight[1:2])
    expected_bias = 2.0 * (ema_q_bias[0:1] - ema_q_bias[1:2])
    assert torch.equal(gate_weight, expected_weight)
    assert torch.equal(gate_bias, expected_bias)
    assert torch.equal(result.target_weight, expected_weight)
    assert torch.equal(result.target_bias, expected_bias)
    assert result.pre_projection_error_norm.item() > 0.0
    assert result.post_projection_error_norm.item() == 0.0
    assert gate_weight.grad is None
    assert gate_bias.grad is None

    # Per-state Q reconstruction can incur independent FP32 cancellation;
    # only the stored raw head parameters are a bit-exact invariant.
    features = torch.tensor([[0.3, -0.8, 1.1]], dtype=torch.float32)
    teacher_q = torch.nn.functional.linear(
        features, ema_q_weight, ema_q_bias
    )
    teacher_probability = torch.sigmoid(
        (teacher_q[:, 0] - teacher_q[:, 1]) / 0.25
    )
    raw_gate_log_odds = torch.nn.functional.linear(
        features, gate_weight, gate_bias
    ).squeeze(-1)
    distribution = compute_voc_gate_distribution(
        torch.zeros((1, 3)),
        temperature=0.5,
        epsilon=0.02,
        raw_gate_log_odds=raw_gate_log_odds,
    )
    torch.testing.assert_close(
        distribution.continue_prob,
        0.98 * teacher_probability + 0.01,
        rtol=1e-6,
        atol=1e-7,
    )


def test_exact_gate_projection_rejects_non_fp32_head_or_ema():
    arguments = dict(
        gate_weight=torch.zeros((1, 2), dtype=torch.float32),
        gate_bias=torch.zeros(1, dtype=torch.float32),
        ema_q_weight=torch.zeros((2, 2), dtype=torch.float32),
        ema_q_bias=torch.zeros(2, dtype=torch.float32),
        q_temperature=0.05,
        policy_temperature=1.0,
    )
    for name in ("gate_weight", "gate_bias", "ema_q_weight", "ema_q_bias"):
        corrupt = dict(arguments)
        corrupt[name] = corrupt[name].double()
        with pytest.raises(TypeError, match="FP32"):
            project_dynamic_voc_gate_head_exact_(**corrupt)


def test_soft_q_bce_recovers_from_wrong_saturation_and_detaches_teacher():
    log_odds = torch.tensor([[-20.0, 20.0]], requires_grad=True)
    q_values = torch.tensor(
        [[[2.0, 0.0], [0.0, 2.0]]], requires_grad=True
    )
    result = compute_dynamic_voc_soft_q_gate_loss(
        gate_log_odds=log_odds,
        q_values=q_values,
        valid=torch.ones_like(log_odds, dtype=torch.bool),
        q_temperature=0.05,
        confidence_weighted=True,
    )

    result.loss.backward()
    assert torch.isfinite(result.loss)
    assert log_odds.grad[0, 0].item() < 0.0
    assert log_odds.grad[0, 1].item() > 0.0
    assert torch.isfinite(log_odds.grad).all()
    assert q_values.grad is None
    assert result.wrong_continue_saturation.tolist() == [[True, False]]
    assert result.wrong_stop_saturation.tolist() == [[False, True]]


@pytest.mark.parametrize("log_odds", [-20.0, -1.5, 0.0, 3.0, 20.0])
def test_confidence_weighted_soft_q_bce_is_exactly_neutral_at_equal_q(
    log_odds,
):
    student = torch.tensor([[log_odds]], requires_grad=True)
    result = compute_dynamic_voc_soft_q_gate_loss(
        gate_log_odds=student,
        q_values=torch.tensor([[[7.0, 7.0]]], requires_grad=True),
        valid=torch.tensor([[True]]),
        q_temperature=0.05,
        confidence_weighted=True,
    )

    assert result.loss.item() == 0.0
    assert result.confidence.item() == 0.0
    assert result.objective_weight.item() == 0.0
    result.loss.backward()
    torch.testing.assert_close(student.grad, torch.zeros_like(student))


@pytest.mark.parametrize("log_odds", [-20.0, -1.5, 0.0, 3.0, 20.0])
def test_unweighted_soft_q_bce_restores_neutral_policy_at_equal_q(log_odds):
    student = torch.tensor([[log_odds]], requires_grad=True)
    q_values = torch.tensor([[[7.0, 7.0]]], requires_grad=True)
    result = compute_dynamic_voc_soft_q_gate_loss(
        gate_log_odds=student,
        q_values=q_values,
        valid=torch.tensor([[True]]),
        q_temperature=0.05,
        confidence_weighted=False,
    )

    expected_probability = torch.sigmoid(student.detach())
    expected_gradient = expected_probability - 0.5
    torch.testing.assert_close(result.confidence, torch.zeros_like(student))
    torch.testing.assert_close(
        result.objective_weight, torch.ones_like(student)
    )
    result.loss.backward()
    torch.testing.assert_close(student.grad, expected_gradient)
    assert q_values.grad is None
    if log_odds != 0.0:
        assert student.grad.item() * log_odds > 0.0


def test_unweighted_soft_q_bce_first_tie_metrics_are_exact():
    student = torch.zeros((1, 1), requires_grad=True)
    result = compute_dynamic_voc_soft_q_gate_loss(
        gate_log_odds=student,
        q_values=torch.zeros((1, 1, 2), requires_grad=True),
        valid=torch.tensor([[True]]),
        q_temperature=0.05,
        confidence_weighted=False,
    )

    torch.testing.assert_close(result.confidence, torch.zeros_like(student))
    torch.testing.assert_close(
        result.objective_weight, torch.ones_like(student)
    )
    expected_bce = torch.full_like(student, torch.log(torch.tensor(2.0)))
    torch.testing.assert_close(result.bce, expected_bce)
    torch.testing.assert_close(result.loss, torch.log(torch.tensor(2.0)))
    torch.testing.assert_close(
        result.directed_logit_gradient, torch.zeros_like(student)
    )
    result.loss.backward()
    torch.testing.assert_close(student.grad, torch.zeros_like(student))


def test_unweighted_soft_q_bce_keeps_first_order_near_tie_credit():
    q_temperature = 0.05

    def directed_gradient(delta_q, confidence_weighted):
        student = torch.zeros((1, 1), requires_grad=True)
        result = compute_dynamic_voc_soft_q_gate_loss(
            gate_log_odds=student,
            q_values=torch.tensor([[[delta_q, 0.0]]]),
            valid=torch.tensor([[True]]),
            q_temperature=q_temperature,
            confidence_weighted=confidence_weighted,
        )
        result.loss.backward()
        return abs(student.grad.item())

    small = 1e-4
    unweighted_small = directed_gradient(small, False)
    unweighted_double = directed_gradient(2.0 * small, False)
    weighted_small = directed_gradient(small, True)
    weighted_double = directed_gradient(2.0 * small, True)

    assert unweighted_small > 0.0
    assert weighted_small > 0.0
    assert unweighted_double / unweighted_small == pytest.approx(2.0, rel=2e-3)
    assert weighted_double / weighted_small == pytest.approx(4.0, rel=3e-3)


def test_soft_q_bce_uses_valid_mean_and_zero_invalid_gradient():
    log_odds = torch.tensor([[0.0, 0.0, float("inf")]], requires_grad=True)
    q_values = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [float("nan"), float("nan")]]]
    )
    valid = torch.tensor([[True, True, False]])
    result = compute_dynamic_voc_soft_q_gate_loss(
        gate_log_odds=log_odds,
        q_values=q_values,
        valid=valid,
        q_temperature=0.5,
        confidence_weighted=True,
    )
    expected_teacher = torch.sigmoid(torch.tensor([2.0, -2.0]))
    expected_confidence = (2.0 * expected_teacher - 1.0).abs()
    expected = (
        expected_confidence
        * torch.nn.functional.binary_cross_entropy_with_logits(
            torch.zeros(2), expected_teacher, reduction="none"
        )
    ).mean()
    torch.testing.assert_close(result.loss, expected)

    result.loss.backward()
    assert log_odds.grad[0, 0].item() < 0.0
    assert log_odds.grad[0, 1].item() > 0.0
    assert log_odds.grad[0, 2].item() == 0.0


def test_soft_q_bce_student_matches_tempered_behavior_gate_probability():
    raw_log_odds = torch.tensor([[0.8, -0.4]], requires_grad=True)
    policy_temperature = 0.4
    distribution = compute_voc_gate_distribution(
        torch.zeros((1, 2, 3)),
        temperature=policy_temperature,
        epsilon=0.0,
        raw_gate_log_odds=raw_log_odds,
    )
    result = compute_dynamic_voc_soft_q_gate_loss(
        gate_log_odds=raw_log_odds,
        q_values=torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        valid=torch.tensor([[True, True]]),
        q_temperature=0.5,
        policy_temperature=policy_temperature,
        confidence_weighted=True,
    )

    torch.testing.assert_close(
        result.student_continue_probability,
        distribution.continue_prob,
    )
    result.loss.backward()
    torch.testing.assert_close(
        raw_log_odds.grad,
        result.directed_logit_gradient / 2.0,
    )


def test_joint_control_gate_detach_preserves_forward_and_bout_gradient():
    raw_control_logits = torch.tensor(
        [[[0.9, -0.3, 0.2]]], requires_grad=True
    )
    raw_gate_log_odds = torch.tensor([[0.7]], requires_grad=True)
    distribution = compute_voc_gate_distribution(
        raw_control_logits,
        temperature=0.6,
        epsilon=0.02,
        raw_gate_log_odds=raw_gate_log_odds,
    )
    detached = detach_dynamic_voc_gate_from_joint_logits(
        distribution.joint_logits
    )
    torch.testing.assert_close(detached, distribution.joint_logits)

    coefficients = torch.tensor([[[1.0, -0.5, 0.25]]])
    torch.sum(detached * coefficients).backward()
    assert raw_gate_log_odds.grad is None or torch.allclose(
        raw_gate_log_odds.grad,
        torch.zeros_like(raw_gate_log_odds.grad),
        atol=1e-7,
        rtol=0.0,
    )
    assert raw_control_logits.grad[..., :2].abs().sum().item() > 0.0
    assert torch.isfinite(raw_control_logits.grad).all()

def test_voc_control_keeps_exact_likelihood_but_detaches_old_gate_gradient():
    primary_and_bout = torch.tensor(1.25, requires_grad=True)
    gate = torch.tensor(-0.75, requires_grad=True)
    routes = {
        "re": primary_and_bout + gate,
        "im": primary_and_bout,
        "cur": primary_and_bout + gate,
        "think": gate,
    }

    controlled = dynamic_voc_policy_log_probs(routes, "control")
    torch.testing.assert_close(controlled["re"], routes["re"])
    torch.testing.assert_close(controlled["think"], routes["think"])
    assert controlled["im"] is routes["im"]
    assert controlled["cur"] is routes["cur"]

    (controlled["re"] + controlled["think"]).backward()
    torch.testing.assert_close(primary_and_bout.grad, torch.tensor(1.0))
    assert gate.grad is None or gate.grad.item() == 0.0


def _voc_loss(
    *, q, logits, action, target=0.0, mode="control",
    behavior_logits=None, **kwargs
):
    if behavior_logits is None:
        behavior_logits = torch.zeros_like(logits)
    return compute_dynamic_voc_loss(
        voc_q=q,
        target_control_logits=logits,
        behavior_control_logits=behavior_logits,
        control_action=torch.tensor([[action]], dtype=torch.long),
        control_valid=torch.tensor([[True]]),
        voc_target=torch.tensor([[target]], dtype=q.dtype),
        mode=mode,
        **kwargs,
    )


def _selected_error_voc_loss(errors, *, schema, q_train_valid=None):
    errors = torch.as_tensor(errors, dtype=torch.float64)
    selected_q = torch.stack((errors, torch.zeros_like(errors)), dim=-1)
    voc_q = selected_q.unsqueeze(0).clone().requires_grad_(True)
    shape = (1, errors.numel())
    logits = torch.zeros(shape + (3,), dtype=torch.float64)
    actions = torch.full(shape, util.PROCEED, dtype=torch.long)
    valid = torch.ones(shape, dtype=torch.bool)
    if q_train_valid is not None:
        q_train_valid = torch.as_tensor(
            q_train_valid, dtype=torch.bool
        ).reshape(shape)
    result = compute_dynamic_voc_loss(
        voc_q=voc_q,
        target_control_logits=logits,
        behavior_control_logits=logits,
        control_action=actions,
        control_valid=valid,
        voc_target=torch.zeros(shape, dtype=torch.float64),
        mode="shadow",
        q_train_valid=q_train_valid,
        gate_policy_schema_version=schema,
    )
    return voc_q, result


def test_schema8_half_squared_q_binds_hand_values_gradients_factor_and_sum():
    errors = torch.tensor(
        [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
        dtype=torch.float32,
    )
    voc_q, result = _selected_error_voc_loss(errors, schema=8)

    expected_rows = 0.5 * errors.square()
    assert result.q_loss.dtype == torch.float32
    torch.testing.assert_close(
        result.q_loss,
        expected_rows.sum(),
        rtol=0.0,
        atol=0.0,
    )
    gradient = torch.autograd.grad(result.q_loss, voc_q)[0]
    torch.testing.assert_close(
        gradient[0, :, 0], errors.to(dtype=gradient.dtype), rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        gradient[0, :, 1], torch.zeros_like(gradient[0, :, 1]),
        rtol=0.0, atol=0.0,
    )

    legacy_q, legacy = _selected_error_voc_loss(errors, schema=7)
    expected_huber_gradient = errors.clamp(-1.0, 1.0)
    expected_huber = torch.where(
        errors.abs() <= 1.0,
        0.5 * errors.square(),
        errors.abs() - 0.5,
    ).sum()
    torch.testing.assert_close(
        legacy.q_loss, expected_huber, rtol=0.0, atol=0.0
    )
    legacy_gradient = torch.autograd.grad(legacy.q_loss, legacy_q)[0]
    torch.testing.assert_close(
        legacy_gradient[0, :, 0],
        expected_huber_gradient.to(dtype=legacy_gradient.dtype),
        rtol=0.0,
        atol=0.0,
    )
    assert result.q_loss.item() > legacy.q_loss.item()


def test_schema8_matches_schema7_exactly_on_beta_one_quadratic_region():
    errors = [-1.0, -0.5, 0.0, 0.5, 1.0]
    schema7_q, schema7 = _selected_error_voc_loss(errors, schema=7)
    schema8_q, schema8 = _selected_error_voc_loss(errors, schema=8)

    assert torch.equal(schema8.q_loss, schema7.q_loss)
    schema7_gradient = torch.autograd.grad(schema7.q_loss, schema7_q)[0]
    schema8_gradient = torch.autograd.grad(schema8.q_loss, schema8_q)[0]
    assert torch.equal(schema8_gradient, schema7_gradient)


@pytest.mark.parametrize(
    ("error", "expected_loss", "expected_gradient"),
    [
        (0.0, 0.0, 0.0),
        (0.5, 0.125, 0.5),
        (-0.5, 0.125, -0.5),
        (1.0, 0.5, 1.0),
        (-1.0, 0.5, -1.0),
        (2.0, 1.5, 1.0),
        (-2.0, 1.5, -1.0),
    ],
)
def test_schema10_huber_binds_exact_beta_one_values_and_gradients(
    error, expected_loss, expected_gradient
):
    voc_q, result = _selected_error_voc_loss(
        [error],
        schema=util.VOC_GATE_POLICY_HUBER_COMMON_Q_SCHEMA_VERSION,
    )

    assert result.q_loss.dtype == torch.float32
    assert result.q_loss.item() == expected_loss
    gradient = torch.autograd.grad(result.q_loss, voc_q)[0]
    assert gradient[0, 0, 0].item() == expected_gradient
    assert gradient[0, 0, 1].item() == 0.0


def test_schema10_huber_mask_and_zero_support_isolate_gradients():
    schema = util.VOC_GATE_POLICY_HUBER_COMMON_Q_SCHEMA_VERSION
    voc_q, result = _selected_error_voc_loss(
        [2.0, -2.0, 0.5],
        schema=schema,
        q_train_valid=[True, False, False],
    )
    assert result.q_loss.item() == 1.5
    gradient = torch.autograd.grad(result.q_loss, voc_q)[0]
    expected = torch.zeros_like(gradient)
    expected[0, 0, 0] = 1.0
    assert torch.equal(gradient, expected)

    unsupported_q, unsupported = _selected_error_voc_loss(
        [2.0, -2.0],
        schema=schema,
        q_train_valid=[False, False],
    )
    assert unsupported.q_loss.item() == 0.0
    unsupported_gradient = torch.autograd.grad(
        unsupported.q_loss, unsupported_q
    )[0]
    assert torch.equal(
        unsupported_gradient, torch.zeros_like(unsupported_gradient)
    )


def test_schema10_huber_converts_operands_to_fp32_before_subtraction():
    selected_q = torch.tensor(
        [[[16777217.0, 0.0]]], dtype=torch.float64, requires_grad=True
    )
    target = torch.tensor([[16777216.0]], dtype=torch.float64)
    logits = torch.zeros((1, 1, 3), dtype=torch.float64)
    result = compute_dynamic_voc_loss(
        voc_q=selected_q,
        target_control_logits=logits,
        behavior_control_logits=logits,
        control_action=torch.tensor([[util.PROCEED]]),
        control_valid=torch.ones((1, 1), dtype=torch.bool),
        voc_target=target,
        mode="shadow",
        gate_policy_schema_version=(
            util.VOC_GATE_POLICY_HUBER_COMMON_Q_SCHEMA_VERSION
        ),
    )

    assert result.q_loss.dtype == torch.float32
    assert result.q_loss.item() == 0.0
    gradient = torch.autograd.grad(result.q_loss, selected_q)[0]
    assert torch.equal(gradient, torch.zeros_like(gradient))


def test_schema8_half_squared_mask_and_zero_support_isolate_gradients():
    voc_q, result = _selected_error_voc_loss(
        [2.0, -2.0, 1.0], schema=8, q_train_valid=[True, False, False]
    )
    torch.testing.assert_close(
        result.q_loss, torch.tensor(2.0), rtol=0.0, atol=0.0
    )
    gradient = torch.autograd.grad(result.q_loss, voc_q)[0]
    expected = torch.zeros_like(gradient)
    expected[0, 0, 0] = 2.0
    assert torch.equal(gradient, expected)

    unsupported_q, unsupported = _selected_error_voc_loss(
        [2.0, -2.0], schema=8, q_train_valid=[False, False]
    )
    assert unsupported.q_loss.item() == 0.0
    unsupported_gradient = torch.autograd.grad(
        unsupported.q_loss, unsupported_q
    )[0]
    assert torch.equal(
        unsupported_gradient, torch.zeros_like(unsupported_gradient)
    )


def test_schema8_dueling_gauge_and_detach_contract_are_unchanged():
    raw_logits = torch.tensor(
        [[[0.8, -0.4, 0.2], [-0.3, 0.5, -0.1]]],
        requires_grad=True,
    )
    distribution = compute_voc_gate_distribution(
        raw_logits, temperature=1.0, epsilon=0.02
    )
    raw_advantage = torch.tensor(
        [[[1.5, -0.5], [-0.2, 0.9]]], requires_grad=True
    )
    state_value = torch.tensor([[4.0, -2.0]], requires_grad=True)
    common = dict(
        target_control_logits=distribution.joint_logits,
        behavior_control_logits=distribution.joint_logits.detach(),
        control_action=torch.tensor([[util.PROCEED, util.STOP]]),
        control_valid=torch.ones((1, 2), dtype=torch.bool),
        voc_target=torch.tensor([[5.0, -1.5]]),
        mode="shadow",
        dueling_q=True,
        gate_policy_schema_version=8,
    )
    result = compute_dynamic_voc_loss(
        voc_q=raw_advantage,
        voc_state_value=state_value,
        **common,
    )
    shifted = compute_dynamic_voc_loss(
        voc_q=raw_advantage.detach() + 7.0,
        voc_state_value=state_value.detach(),
        **common,
    )

    torch.testing.assert_close(shifted.q_values, result.q_values.detach())
    torch.testing.assert_close(shifted.q_loss, result.q_loss.detach())
    result.q_loss.backward()
    assert raw_advantage.grad is not None
    assert torch.count_nonzero(raw_advantage.grad).item() > 0
    assert state_value.grad is None
    assert raw_logits.grad is None


@pytest.mark.parametrize("schema", [9, 10, 11, 12, 13])
def test_schema9_through_schema13_common_q_bind_hand_algebra_and_exact_jacobian(
    schema,
):
    raw_logits = torch.tensor(
        [[[0.8, -0.4, 0.2], [-0.3, 0.5, -0.1]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    distribution = compute_voc_gate_distribution(
        raw_logits, temperature=1.0, epsilon=0.02
    )
    raw_advantage = torch.tensor(
        [[[1.5, -0.5], [-0.2, 0.9]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    state_value = torch.tensor(
        [[4.0, -2.0]], dtype=torch.float64, requires_grad=True
    )
    target = torch.tensor(
        [[5.0, -1.5]], dtype=torch.float64, requires_grad=True
    )
    actions = torch.tensor([[util.PROCEED, util.STOP]])

    result = compute_dynamic_voc_loss(
        voc_q=raw_advantage,
        target_control_logits=distribution.joint_logits,
        behavior_control_logits=distribution.joint_logits.detach(),
        control_action=actions,
        control_valid=torch.ones_like(actions, dtype=torch.bool),
        voc_target=target,
        mode="shadow",
        dueling_q=True,
        voc_state_value=state_value,
        gate_policy_schema_version=schema,
    )
    probabilities = torch.stack(
        (distribution.continue_prob, distribution.stop_prob), dim=-1
    ).detach()
    common = raw_advantage.mean(dim=-1, keepdim=True)
    centered = raw_advantage - torch.sum(
        probabilities * raw_advantage, dim=-1, keepdim=True
    )
    expected_q = state_value.detach().unsqueeze(-1) + common + centered
    expected_selected = torch.gather(
        expected_q,
        dim=-1,
        index=torch.tensor([[[0], [1]]]),
    ).squeeze(-1)

    torch.testing.assert_close(result.q_values, expected_q)
    torch.testing.assert_close(result.selected_q, expected_selected)
    torch.testing.assert_close(
        result.delta_q,
        raw_advantage[..., 0] - raw_advantage[..., 1],
    )
    raw_jacobian, value_gradient, policy_gradient, target_gradient = (
        torch.autograd.grad(
            result.selected_q.sum(),
            (raw_advantage, state_value, raw_logits, target),
            allow_unused=True,
        )
    )
    selected_one_hot = torch.nn.functional.one_hot(
        result.gate_action, num_classes=2
    ).to(dtype=raw_advantage.dtype)
    expected_jacobian = (
        torch.full_like(probabilities, 0.5)
        + selected_one_hot
        - probabilities
    )
    torch.testing.assert_close(raw_jacobian, expected_jacobian)
    torch.testing.assert_close(
        raw_jacobian.sum(dim=-1),
        torch.ones_like(raw_jacobian[..., 0]),
    )
    assert value_gradient is None
    assert policy_gradient is None
    assert target_gradient is None


def test_schema9_rounding_safe_shift_and_fp32_large_shift_nonclaim():
    distribution = compute_voc_gate_distribution(
        torch.zeros((1, 1, 3), dtype=torch.float32),
        temperature=1.0,
        epsilon=0.0,
        raw_gate_log_odds=torch.zeros((1, 1), dtype=torch.float32),
    )
    state_value = torch.zeros((1, 1), dtype=torch.float32)
    actions = torch.tensor([[util.PROCEED]])

    def reconstruct(raw, schema):
        return compute_dynamic_voc_loss(
            voc_q=raw,
            target_control_logits=distribution.joint_logits,
            behavior_control_logits=distribution.joint_logits,
            control_action=actions,
            control_valid=torch.ones_like(actions, dtype=torch.bool),
            voc_target=torch.zeros_like(state_value),
            mode="shadow",
            dueling_q=True,
            voc_state_value=state_value,
            gate_policy_schema_version=schema,
        )

    raw8 = torch.tensor([[[1.0, -1.0]]], requires_grad=True)
    shifted8 = torch.tensor([[[9.0, 7.0]]], requires_grad=True)
    schema8 = reconstruct(
        raw8, util.VOC_GATE_POLICY_HALF_SQUARED_Q_SCHEMA_VERSION
    )
    schema8_shifted = reconstruct(
        shifted8, util.VOC_GATE_POLICY_HALF_SQUARED_Q_SCHEMA_VERSION
    )
    assert torch.equal(schema8.q_values, schema8_shifted.q_values)

    raw9 = torch.tensor([[[1.0, -1.0]]], requires_grad=True)
    shifted9 = torch.tensor([[[9.0, 7.0]]], requires_grad=True)
    schema9 = reconstruct(
        raw9, util.VOC_GATE_POLICY_COMMON_MODE_Q_SCHEMA_VERSION
    )
    schema9_shifted = reconstruct(
        shifted9, util.VOC_GATE_POLICY_COMMON_MODE_Q_SCHEMA_VERSION
    )
    assert torch.equal(
        schema9_shifted.q_values,
        schema9.q_values + torch.full_like(schema9.q_values, 8.0),
    )
    assert torch.equal(
        schema9.delta_q,
        schema9_shifted.delta_q,
    )
    schema8_gradient = torch.autograd.grad(schema8.selected_q.sum(), raw8)[0]
    schema9_gradient = torch.autograd.grad(schema9.selected_q.sum(), raw9)[0]
    assert schema8_gradient.sum().item() == 0.0
    assert schema9_gradient.sum().item() == 1.0

    raw_large = torch.tensor([1.0, 0.0], dtype=torch.float32)
    stored_shifted = raw_large + float(2**24)
    assert torch.equal(
        stored_shifted,
        torch.full_like(stored_shifted, float(2**24)),
    )
    assert (raw_large[0] - raw_large[1]).item() == 1.0
    assert (stored_shifted[0] - stored_shifted[1]).item() == 0.0


@pytest.mark.parametrize("schema", [9, 10, 11, 12, 13])
def test_schema9_through_schema13_heldout_rows_are_fully_gradient_isolated(
    schema,
):
    raw_logits = torch.zeros(
        (1, 3, 3), dtype=torch.float32, requires_grad=True
    )
    raw_advantage = torch.tensor(
        [[[2.0, -1.0], [-3.0, 1.0], [0.5, -0.5]]],
        requires_grad=True,
    )
    state_value = torch.tensor(
        [[0.25, -0.75, 1.25]], requires_grad=True
    )
    target = torch.tensor([[0.0, 4.0, -3.0]], requires_grad=True)
    q_train_valid = torch.tensor([[True, False, False]])
    result = compute_dynamic_voc_loss(
        voc_q=raw_advantage,
        target_control_logits=raw_logits,
        behavior_control_logits=raw_logits.detach(),
        control_action=torch.tensor(
            [[util.PROCEED, util.STOP, util.PROCEED]]
        ),
        control_valid=torch.ones((1, 3), dtype=torch.bool),
        voc_target=target,
        mode="shadow",
        q_train_valid=q_train_valid,
        dueling_q=True,
        voc_state_value=state_value,
        gate_policy_schema_version=schema,
    )

    result.q_loss.backward()
    assert torch.count_nonzero(raw_advantage.grad[:, :1]).item() > 0
    assert torch.count_nonzero(raw_advantage.grad[:, 1:]).item() == 0
    assert state_value.grad is None
    assert raw_logits.grad is None
    assert target.grad is None

    all_heldout_raw = raw_advantage.detach().clone().requires_grad_(True)
    all_heldout = compute_dynamic_voc_loss(
        voc_q=all_heldout_raw,
        target_control_logits=raw_logits.detach(),
        behavior_control_logits=raw_logits.detach(),
        control_action=torch.tensor(
            [[util.PROCEED, util.STOP, util.PROCEED]]
        ),
        control_valid=torch.ones((1, 3), dtype=torch.bool),
        voc_target=target.detach(),
        mode="shadow",
        q_train_valid=torch.zeros((1, 3), dtype=torch.bool),
        dueling_q=True,
        voc_state_value=state_value.detach(),
        gate_policy_schema_version=schema,
    )
    assert all_heldout.q_loss.item() == 0.0
    all_heldout.q_loss.backward()
    assert torch.count_nonzero(all_heldout_raw.grad).item() == 0


def test_schema8_schema9_half_squared_bytes_match_for_equal_selected_q():
    distribution = compute_voc_gate_distribution(
        torch.zeros((1, 2, 3)),
        raw_gate_log_odds=torch.zeros((1, 2)),
        epsilon=0.0,
    )
    raw = torch.tensor([[[2.0, -2.0], [-3.0, 3.0]]])
    common = dict(
        target_control_logits=distribution.joint_logits,
        behavior_control_logits=distribution.joint_logits,
        control_action=torch.tensor([[util.PROCEED, util.STOP]]),
        control_valid=torch.ones((1, 2), dtype=torch.bool),
        voc_target=torch.tensor([[0.5, -1.5]]),
        mode="shadow",
        dueling_q=True,
        voc_state_value=torch.tensor([[1.0, -2.0]]),
    )
    schema8 = compute_dynamic_voc_loss(
        voc_q=raw.clone().requires_grad_(True),
        gate_policy_schema_version=(
            util.VOC_GATE_POLICY_HALF_SQUARED_Q_SCHEMA_VERSION
        ),
        **common,
    )
    schema9 = compute_dynamic_voc_loss(
        voc_q=raw.clone().requires_grad_(True),
        gate_policy_schema_version=(
            util.VOC_GATE_POLICY_COMMON_MODE_Q_SCHEMA_VERSION
        ),
        **common,
    )

    assert torch.equal(schema8.q_values, schema9.q_values)
    assert torch.equal(schema8.selected_q, schema9.selected_q)
    assert torch.equal(schema8.q_loss, schema9.q_loss)


def test_schema7_through_schema13_loss_lineage_is_exact_and_differential():
    errors = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    results = {}
    gradients = {}
    for schema in (7, 8, 9, 10, 11, 12, 13):
        voc_q, results[schema] = _selected_error_voc_loss(
            errors, schema=schema
        )
        gradients[schema] = torch.autograd.grad(
            results[schema].q_loss, voc_q
        )[0]

    assert torch.equal(results[7].q_loss, results[10].q_loss)
    assert torch.equal(gradients[7], gradients[10])
    assert torch.equal(results[10].q_loss, results[11].q_loss)
    assert torch.equal(gradients[10], gradients[11])
    assert torch.equal(results[11].q_loss, results[12].q_loss)
    assert torch.equal(gradients[11], gradients[12])
    assert torch.equal(results[12].q_loss, results[13].q_loss)
    assert torch.equal(gradients[12], gradients[13])
    assert torch.equal(results[8].q_loss, results[9].q_loss)
    assert torch.equal(gradients[8], gradients[9])
    assert results[8].q_loss.item() > results[10].q_loss.item()
    assert not torch.equal(gradients[8], gradients[10])


@pytest.mark.parametrize(
    "schema",
    [
        util.VOC_GATE_POLICY_COMMON_MODE_Q_SCHEMA_VERSION,
        util.VOC_GATE_POLICY_HUBER_COMMON_Q_SCHEMA_VERSION,
        util.VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION,
        util.VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION,
        util.VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION,
    ],
)
def test_schema9_through_schema13_online_and_ema_common_q_are_identical(schema):
    learner = object.__new__(SActorLearner)
    learner.voc_gate_policy_schema_version = schema
    learner.dynamic_voc_mode = "shadow"
    learner.voc_ema_gate_weight = torch.tensor(
        [[0.5, -0.25, 1.0], [-0.75, 0.5, 0.25]], dtype=torch.float32
    )
    learner.voc_ema_gate_bias = torch.tensor(
        [0.125, -0.375], dtype=torch.float32
    )
    features = torch.tensor(
        [[[0.5, -1.0, 2.0], [-0.25, 0.75, 1.5]]],
        dtype=torch.float32,
    )
    logits = torch.tensor(
        [[[0.2, -0.4, 0.1], [-0.7, 0.3, -0.2]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    state_value = torch.tensor(
        [[1.25, -0.5]], dtype=torch.float32, requires_grad=True
    )
    valid = torch.ones((1, 2), dtype=torch.bool)

    gate_loss, ema_q = learner._compute_ema_gate_loss(
        features=features,
        logits=logits,
        valid=valid,
        state_value=state_value,
    )
    raw = torch.nn.functional.linear(
        features, learner.voc_ema_gate_weight, learner.voc_ema_gate_bias
    )
    entropy = learn_actor_module.compute_dynamic_control_entropy(
        logits, project_gate_gradient=False
    )
    probabilities = torch.stack(
        (entropy.continue_prob, entropy.stop_prob), dim=-1
    ).detach()
    expected = (
        state_value.detach().unsqueeze(-1)
        + raw.mean(dim=-1, keepdim=True)
        + raw
        - torch.sum(probabilities * raw, dim=-1, keepdim=True)
    )
    torch.testing.assert_close(ema_q, expected)
    assert not ema_q.requires_grad
    assert gate_loss.item() == 0.0

    online = compute_dynamic_voc_loss(
        voc_q=raw.clone().requires_grad_(True),
        target_control_logits=logits,
        behavior_control_logits=logits.detach(),
        control_action=torch.tensor([[util.PROCEED, util.STOP]]),
        control_valid=valid,
        voc_target=torch.zeros((1, 2)),
        mode="shadow",
        dueling_q=True,
        voc_state_value=state_value,
        gate_policy_schema_version=schema,
    )
    assert torch.equal(online.q_values, ema_q)


@pytest.mark.parametrize(
    "schema",
    [
        True,
        np.int64(8),
        8.0,
        "8",
        np.int64(9),
        9.0,
        "9",
        np.int64(10),
        10.0,
        "10",
        np.int64(11),
        11.0,
        "11",
        np.int64(12),
        12.0,
        "12",
        np.int64(13),
        13.0,
        "13",
        0,
        14,
    ],
)
def test_q_loss_rejects_noncanonical_or_unknown_schema_identity(schema):
    with pytest.raises(ValueError, match="strict integer"):
        _selected_error_voc_loss([0.5], schema=schema)


@pytest.mark.parametrize("schema", [8, 9])
def test_schema8_and_schema9_half_squared_q_overflow_fails_closed(schema):
    with pytest.raises(FloatingPointError, match="half-squared Q loss"):
        _selected_error_voc_loss([2.0e19], schema=schema)


@pytest.mark.parametrize("schema", [None, 1, 2, 3, 4, 5, 6, 7])
def test_schemas_at_most_seven_retain_exact_huber_bytes(schema):
    errors = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    reference_q, reference = _selected_error_voc_loss(errors, schema=None)
    reference_gradient = torch.autograd.grad(reference.q_loss, reference_q)[0]
    candidate_q, candidate = _selected_error_voc_loss(errors, schema=schema)
    candidate_gradient = torch.autograd.grad(candidate.q_loss, candidate_q)[0]

    assert torch.equal(candidate.q_loss, reference.q_loss)
    assert torch.equal(candidate_gradient, reference_gradient)


def test_voc_loss_reconstructs_epsilon_mixed_behavior_probability_from_logits():
    raw_behavior_log_odds = torch.tensor([[0.8, -0.4]])
    policy_temperature = 0.5
    epsilon = 0.02
    behavior = compute_voc_gate_distribution(
        torch.zeros((1, 2, 3)),
        temperature=policy_temperature,
        epsilon=epsilon,
        raw_gate_log_odds=raw_behavior_log_odds,
    )
    result = compute_dynamic_voc_loss(
        voc_q=torch.zeros((1, 2, 2)),
        target_control_logits=torch.zeros((1, 2, 3)),
        behavior_control_logits=behavior.joint_logits.detach(),
        control_action=torch.tensor([[util.PROCEED, util.STOP]]),
        control_valid=torch.ones((1, 2), dtype=torch.bool),
        voc_target=torch.zeros((1, 2)),
        mode="shadow",
    )
    expected = (
        (1.0 - epsilon)
        * torch.sigmoid(raw_behavior_log_odds / policy_temperature)
        + 0.5 * epsilon
    )

    torch.testing.assert_close(
        result.behavior_continue_probability,
        behavior.continue_prob,
        rtol=0.0,
        atol=1e-7,
    )
    torch.testing.assert_close(
        result.behavior_continue_probability,
        expected,
        rtol=0.0,
        atol=1e-7,
    )
    assert not result.behavior_continue_probability.requires_grad


@pytest.mark.parametrize(
    "action,q_values,expected_gate_action,expected_sign",
    [
        (util.PROCEED, (2.0, 0.0), 0, (-1, -1, 1)),
        (util.RESET, (2.0, 0.0), 0, (-1, -1, 1)),
        (util.STOP, (0.0, 2.0), 1, (1, 1, -1)),
    ],
)
def test_soft_voc_advantage_moves_gate_probability_without_hard_q_action(
    action, q_values, expected_gate_action, expected_sign
):
    logits = torch.zeros((1, 1, 3), requires_grad=True)
    q = torch.tensor([[q_values]], requires_grad=True)
    result = _voc_loss(q=q, logits=logits, action=action)

    assert result.gate_action.item() == expected_gate_action
    result.gate_pg_loss.backward()

    for gradient, sign in zip(logits.grad.flatten(), expected_sign):
        assert torch.sign(gradient).item() == sign
    # Gate-only VoC credit changes the shared CONTINUE shift, never the
    # conditional PROCEED/RESET preference, even for unequal bout logits.
    torch.testing.assert_close(
        logits.grad[..., util.PROCEED],
        logits.grad[..., util.RESET],
    )
    # Q only evaluates the stochastic gate; gate PG cannot train the critic.
    assert q.grad is None


def test_voc_gate_gradient_does_not_leak_into_unequal_bout_logits():
    raw_logits = torch.tensor([[[1.2, -0.4, 0.3]]], requires_grad=True)
    transformed_logits = compute_voc_gate_distribution(
        raw_logits, temperature=1.0, epsilon=0.02
    ).joint_logits
    q = torch.tensor([[[2.0, 0.0]]], requires_grad=True)
    result = _voc_loss(
        q=q,
        logits=transformed_logits,
        action=util.PROCEED,
        target=1.0,
    )

    result.gate_pg_loss.backward()

    torch.testing.assert_close(
        raw_logits.grad[..., util.PROCEED],
        raw_logits.grad[..., util.RESET],
    )


def test_equal_q_has_zero_directed_gate_gradient():
    logits = torch.tensor([[[0.4, -0.3, 0.1]]], requires_grad=True)
    q = torch.tensor([[[1.5, 1.5]]], requires_grad=True)
    result = _voc_loss(q=q, logits=logits, action=util.PROCEED)

    result.gate_pg_loss.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))
    assert q.grad is None


def test_policy_centered_dueling_q_reconstructs_and_is_gauge_invariant():
    raw_logits = torch.tensor(
        [[[0.8, -0.4, 0.2], [-0.3, 0.5, -0.1]]],
        requires_grad=True,
    )
    distribution = compute_voc_gate_distribution(
        raw_logits, temperature=1.0, epsilon=0.02
    )
    raw_advantage = torch.tensor(
        [[[1.5, -0.5], [-0.2, 0.9]]], requires_grad=True
    )
    state_value = torch.tensor([[4.0, -2.0]], requires_grad=True)
    actions = torch.tensor([[util.PROCEED, util.STOP]])
    target = torch.tensor([[5.0, -1.5]])

    result = compute_dynamic_voc_loss(
        voc_q=raw_advantage,
        target_control_logits=distribution.joint_logits,
        behavior_control_logits=distribution.joint_logits.detach(),
        control_action=actions,
        control_valid=torch.ones_like(actions, dtype=torch.bool),
        voc_target=target,
        mode="shadow",
        dueling_q=True,
        voc_state_value=state_value,
    )
    probabilities = torch.stack(
        (distribution.continue_prob, distribution.stop_prob), dim=-1
    ).detach()
    centered_mean = torch.sum(
        probabilities * result.q_values, dim=-1
    )
    torch.testing.assert_close(centered_mean, state_value.detach())
    torch.testing.assert_close(
        result.delta_q,
        raw_advantage[..., 0] - raw_advantage[..., 1],
    )

    shifted_result = compute_dynamic_voc_loss(
        voc_q=raw_advantage.detach() + 7.0,
        target_control_logits=distribution.joint_logits.detach(),
        behavior_control_logits=distribution.joint_logits.detach(),
        control_action=actions,
        control_valid=torch.ones_like(actions, dtype=torch.bool),
        voc_target=target,
        mode="shadow",
        dueling_q=True,
        voc_state_value=state_value.detach(),
    )
    torch.testing.assert_close(shifted_result.q_values, result.q_values.detach())
    torch.testing.assert_close(shifted_result.q_loss, result.q_loss.detach())

    gate_action = torch.tensor([[0, 1]])
    selected = torch.gather(
        result.q_values, -1, gate_action.unsqueeze(-1)
    ).squeeze(-1)
    expected_loss = torch.nn.functional.smooth_l1_loss(
        selected.float(), target.float(), reduction="sum"
    )
    torch.testing.assert_close(result.selected_q, selected)
    torch.testing.assert_close(result.q_loss, expected_loss)

    result.q_loss.backward()
    assert raw_advantage.grad is not None
    assert torch.count_nonzero(raw_advantage.grad).item() > 0
    assert state_value.grad is None
    assert raw_logits.grad is None


def test_policy_centered_dueling_q_requires_matching_state_value():
    q = torch.zeros((1, 2, 2))
    logits = torch.zeros((1, 2, 3))
    actions = torch.zeros((1, 2), dtype=torch.long)
    valid = torch.ones_like(actions, dtype=torch.bool)
    target = torch.zeros_like(actions, dtype=torch.float)

    with pytest.raises(ValueError, match="requires voc_state_value"):
        compute_dynamic_voc_loss(
            voc_q=q,
            target_control_logits=logits,
            behavior_control_logits=logits,
            control_action=actions,
            control_valid=valid,
            voc_target=target,
            mode="shadow",
            dueling_q=True,
        )
    with pytest.raises(ValueError, match="must match control_action"):
        compute_dynamic_voc_loss(
            voc_q=q,
            target_control_logits=logits,
            behavior_control_logits=logits,
            control_action=actions,
            control_valid=valid,
            voc_target=target,
            mode="shadow",
            dueling_q=True,
            voc_state_value=torch.zeros((1, 1)),
        )


def test_exact_expected_gate_loss_matches_enumerated_policy_gradient():
    q = torch.tensor([[[2.3, -0.7]]])

    sampled_logits = torch.tensor(
        [[[0.9, -0.2, 0.1]]], requires_grad=True
    )
    sampled_distribution = compute_voc_gate_distribution(
        sampled_logits, temperature=1.0, epsilon=0.02
    )
    sampled_losses = []
    for action in (util.PROCEED, util.STOP):
        sampled_losses.append(_voc_loss(
            q=q,
            logits=sampled_distribution.joint_logits,
            behavior_logits=sampled_distribution.joint_logits.detach(),
            action=action,
            mode="control",
        ).gate_pg_loss)
    enumerated_sample_loss = (
        sampled_distribution.continue_prob.detach() * sampled_losses[0]
        + sampled_distribution.stop_prob.detach() * sampled_losses[1]
    ).sum()
    enumerated_sample_loss.backward()
    sampled_gradient = sampled_logits.grad.detach().clone()

    exact_logits = sampled_logits.detach().clone().requires_grad_(True)
    exact_distribution = compute_voc_gate_distribution(
        exact_logits, temperature=1.0, epsilon=0.02
    )
    exact_result = _voc_loss(
        q=q,
        logits=exact_distribution.joint_logits,
        action=util.PROCEED,
        mode="control",
        expected_gate_loss=True,
    )
    literal_expected_loss = -(
        exact_distribution.continue_prob * q[..., 0]
        + exact_distribution.stop_prob * q[..., 1]
    ).sum()
    torch.testing.assert_close(
        exact_result.gate_pg_loss, literal_expected_loss
    )
    exact_result.gate_pg_loss.backward()
    torch.testing.assert_close(exact_logits.grad, sampled_gradient)
    torch.testing.assert_close(
        exact_logits.grad[..., util.PROCEED],
        exact_logits.grad[..., util.RESET],
    )
    assert exact_logits.grad[..., util.PROCEED].item() < 0.0
    assert exact_logits.grad[..., util.STOP].item() > 0.0


def test_exact_dueling_gate_gradient_ignores_base_and_q_has_no_actor_grad():
    def gradient_for_base(base):
        raw_logits = torch.tensor(
            [[[0.7, -0.6, 0.2]]], requires_grad=True
        )
        distribution = compute_voc_gate_distribution(
            raw_logits, temperature=1.0, epsilon=0.02
        )
        raw_advantage = torch.tensor(
            [[[1.2, -0.8]]], requires_grad=True
        )
        state_value = torch.tensor([[base]], requires_grad=True)
        result = _voc_loss(
            q=raw_advantage,
            logits=distribution.joint_logits,
            action=util.STOP,
            mode="control",
            dueling_q=True,
            voc_state_value=state_value,
            expected_gate_loss=True,
        )
        result.gate_pg_loss.backward()
        assert raw_advantage.grad is None
        assert state_value.grad is None
        return result, raw_logits.grad.detach().clone()

    low_base_result, low_base_gradient = gradient_for_base(-3.0)
    high_base_result, high_base_gradient = gradient_for_base(11.0)
    torch.testing.assert_close(low_base_gradient, high_base_gradient)
    torch.testing.assert_close(
        low_base_gradient[..., util.PROCEED],
        low_base_gradient[..., util.RESET],
    )
    torch.testing.assert_close(
        high_base_result.gate_pg_loss - low_base_result.gate_pg_loss,
        torch.tensor(-14.0),
    )

    tie_logits = torch.tensor(
        [[[0.5, -0.1, 0.3]]], requires_grad=True
    )
    tie_distribution = compute_voc_gate_distribution(
        tie_logits, temperature=1.0, epsilon=0.02
    )
    tie_result = _voc_loss(
        q=torch.tensor([[[2.0, 2.0]]], requires_grad=True),
        logits=tie_distribution.joint_logits,
        action=util.PROCEED,
        mode="control",
        dueling_q=True,
        voc_state_value=torch.tensor([[4.0]], requires_grad=True),
        expected_gate_loss=True,
    )
    tie_result.gate_pg_loss.backward()
    torch.testing.assert_close(
        tie_logits.grad, torch.zeros_like(tie_logits), atol=1e-7, rtol=0.0
    )


def test_exact_expected_gate_loss_is_zero_in_shadow_mode():
    logits = torch.tensor([[[0.2, -0.4, 0.1]]], requires_grad=True)
    result = _voc_loss(
        q=torch.tensor([[[3.0, -1.0]]]),
        logits=logits,
        action=util.PROCEED,
        mode="shadow",
        expected_gate_loss=True,
    )
    torch.testing.assert_close(result.gate_pg_loss, torch.tensor(0.0))
    result.gate_pg_loss.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


def _legacy_voc_observability_keys():
    keys = {
        "voc_delta_q_positive_count",
        "voc_delta_q_negative_count",
        "voc_delta_q_tie_count",
        "voc_delta_q_positive_rate",
        "voc_delta_q_negative_rate",
        "voc_delta_q_tie_rate",
        "voc_q_greedy_nontie_count",
        "voc_q_greedy_continue_rate",
        "voc_sign_gate_agreement_count",
        "voc_sign_gate_agreement_rate",
        "voc_signed_gate_margin",
        "voc_continue_probability_delta_positive",
        "voc_continue_probability_delta_negative",
    }
    prefixes = (
        "voc_depth_bin_0",
        "voc_depth_bin_1",
        "voc_depth_bin_2_3",
        "voc_depth_bin_4_7",
        "voc_depth_bin_8_15",
        "voc_depth_bin_16_plus",
        "voc_post_compute",
        "voc_post_proceed",
        "voc_post_reset",
    )
    suffixes = (
        "count",
        "delta_q",
        "delta_q_positive_rate",
        "continue_probability",
        "sampled_continue_rate",
        "sampled_stop_rate",
        "signed_gate_margin",
    )
    keys.update(
        f"{prefix}_{suffix}"
        for prefix in prefixes
        for suffix in suffixes
    )
    return keys


def test_voc_observability_exact_sign_depth_and_post_compute_slices():
    delta_q = torch.tensor([[
        2.0, -2.0, 0.0, 4.0, -4.0, 1e-7, 3.0, float("nan")
    ]], requires_grad=True)
    continue_probability = torch.tensor([[
        0.8, 0.2, 0.5, 0.7, 0.3, 0.6, 0.9, float("nan")
    ]], requires_grad=True)
    control_action = torch.tensor([[
        util.STOP,
        util.PROCEED,
        util.STOP,
        util.RESET,
        util.STOP,
        util.STOP,
        util.STOP,
        util.PROCEED,
    ]])
    gate_action = torch.where(
        control_action == util.STOP,
        torch.ones_like(control_action),
        torch.zeros_like(control_action),
    )
    control_valid = torch.tensor([[
        True, True, True, True, True, True, True, False
    ]])
    search_steps = torch.tensor([[0, 1, 1, 3, 4, 8, 16, 99]])
    # This is the accepted control in the EnvOut used to make each current
    # decision, not the current/post-step last_search_control token.
    predecision_last_control = torch.tensor([[
        util.STOP,
        util.PROCEED,
        util.PROCEED,
        util.RESET,
        util.RESET,
        util.STOP,
        util.PROCEED,
        util.RESET,
    ]])
    delta_before = delta_q.detach().clone()
    probability_before = continue_probability.detach().clone()
    torch.manual_seed(619)
    rng_before = torch.random.get_rng_state().clone()

    metrics = dynamic_voc_observability_metrics(
        delta_q=delta_q,
        continue_probability=continue_probability,
        gate_action=gate_action,
        control_valid=control_valid,
        search_steps=search_steps,
        control_action=control_action,
        predecision_last_control=predecision_last_control,
        q_temperature=1.0,
    )

    # Diagnostics neither consume RNG nor retain a gradient path or mutate
    # their model/behavior inputs.  The invalid NaN row is deliberately
    # excluded from every reported slice.
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    torch.testing.assert_close(
        delta_q.detach(), delta_before, equal_nan=True
    )
    torch.testing.assert_close(
        continue_probability.detach(), probability_before, equal_nan=True
    )
    assert all(not value.requires_grad for value in metrics.values())
    # All 76 v6 keys remain present under their exact historical names; the
    # assertions below also pin representative values from every old family.
    legacy_keys = _legacy_voc_observability_keys()
    assert len(legacy_keys) == 76
    assert legacy_keys <= metrics.keys()

    assert metrics["voc_delta_q_positive_count"].item() == 3
    assert metrics["voc_delta_q_negative_count"].item() == 2
    assert metrics["voc_delta_q_tie_count"].item() == 2
    assert metrics["voc_delta_q_positive_rate"].item() == pytest.approx(3 / 7)
    assert metrics["voc_delta_q_negative_rate"].item() == pytest.approx(2 / 7)
    assert metrics["voc_delta_q_tie_rate"].item() == pytest.approx(2 / 7)
    assert metrics["voc_q_greedy_continue_rate"].item() == pytest.approx(3 / 5)
    assert metrics["voc_sign_gate_agreement_rate"].item() == 1.0
    assert metrics["voc_signed_gate_margin"].item() == pytest.approx(0.56)
    assert metrics[
        "voc_continue_probability_delta_positive"
    ].item() == pytest.approx(0.8)
    assert metrics[
        "voc_continue_probability_delta_negative"
    ].item() == pytest.approx(0.25)

    assert metrics["voc_depth_bin_0_count"].item() == 2
    assert metrics["voc_depth_bin_0_delta_q"].item() == 0.0
    assert metrics[
        "voc_depth_bin_0_continue_probability"
    ].item() == pytest.approx(0.5)
    assert metrics[
        "voc_depth_bin_0_sampled_continue_rate"
    ].item() == pytest.approx(0.5)
    assert metrics["voc_depth_bin_1_count"].item() == 1
    assert metrics["voc_depth_bin_2_3_count"].item() == 1
    assert metrics["voc_depth_bin_4_7_count"].item() == 1
    assert metrics["voc_depth_bin_8_15_count"].item() == 1
    assert metrics["voc_depth_bin_16_plus_count"].item() == 1

    assert metrics["voc_post_compute_count"].item() == 4
    assert metrics["voc_post_proceed_count"].item() == 2
    assert metrics["voc_post_reset_count"].item() == 2
    assert metrics["voc_post_compute_delta_q"].item() == pytest.approx(0.75)
    assert metrics[
        "voc_post_compute_delta_q_positive_rate"
    ].item() == pytest.approx(0.5)
    assert metrics[
        "voc_post_compute_continue_probability"
    ].item() == pytest.approx(0.6)
    assert metrics[
        "voc_post_compute_sampled_stop_rate"
    ].item() == pytest.approx(0.75)
    assert metrics["voc_post_compute_delta_q_positive_count"].item() == 2
    assert metrics["voc_post_compute_delta_q_negative_count"].item() == 1
    assert metrics["voc_post_compute_delta_q_tie_count"].item() == 1
    assert metrics["voc_post_compute_delta_q_nontie_count"].item() == 3
    assert metrics[
        "voc_post_compute_continue_probability_delta_positive"
    ].item() == pytest.approx(0.8)
    assert metrics[
        "voc_post_compute_continue_probability_delta_negative"
    ].item() == pytest.approx(0.3)
    assert metrics[
        "voc_post_compute_sampled_continue_given_delta_positive_rate"
    ].item() == pytest.approx(0.5)
    assert metrics[
        "voc_post_compute_sampled_stop_given_delta_negative_rate"
    ].item() == 1.0
    assert metrics[
        "voc_post_compute_argmax_continue_given_delta_positive_rate"
    ].item() == 1.0
    assert metrics[
        "voc_post_compute_argmax_stop_given_delta_negative_rate"
    ].item() == 1.0
    assert metrics[
        "voc_post_compute_sign_gate_agreement_count"
    ].item() == 3
    assert metrics[
        "voc_post_compute_sign_gate_agreement_rate"
    ].item() == 1.0

    # T=1 has no immediate predecessor even though legacy post-compute labels
    # can be populated by the replay token on that first time row.
    assert metrics["voc_post_useful_compute_count"].item() == 0
    assert metrics["voc_post_useful_proceed_count"].item() == 0
    assert metrics["voc_post_useful_reset_count"].item() == 0

    # Empty depth bins/slices remain finite and are disambiguated by count.
    empty = dynamic_voc_observability_metrics(
        delta_q=torch.zeros((1, 1)),
        continue_probability=torch.full((1, 1), 0.5),
        gate_action=torch.zeros((1, 1), dtype=torch.long),
        control_valid=torch.zeros((1, 1), dtype=torch.bool),
        search_steps=torch.zeros((1, 1), dtype=torch.long),
        control_action=torch.zeros((1, 1), dtype=torch.long),
        predecision_last_control=torch.zeros((1, 1), dtype=torch.long),
        q_temperature=1.0,
    )
    assert all(torch.isfinite(value).item() for value in empty.values())
    assert empty["voc_post_compute_count"].item() == 0
    assert empty["voc_depth_bin_16_plus_count"].item() == 0
    assert empty["voc_post_useful_compute_prior_useful_count"].item() == 0
    assert empty["voc_post_useful_compute_prior_delta_q"].item() == 0
    assert (
        empty["voc_post_useful_compute_prior_useful_candidate_count"].item()
        == 0
    )
    assert empty["voc_post_useful_compute_transition_coverage_rate"].item() == 0


def test_voc_observability_adds_behavior_probability_sufficient_stats_only():
    kwargs = dict(
        delta_q=torch.tensor([[2.0, 3.0], [1.0, -1.0]]),
        continue_probability=torch.tensor([[0.6, 0.7], [0.8, 0.2]]),
        gate_action=torch.tensor([[0, 0], [0, 1]]),
        control_valid=torch.ones((2, 2), dtype=torch.bool),
        search_steps=torch.tensor([[9, 9], [9, 9]]),
        control_action=torch.tensor([
            [util.PROCEED, util.RESET],
            [util.STOP, util.STOP],
        ]),
        predecision_last_control=torch.tensor([
            [util.STOP, util.STOP],
            [util.PROCEED, util.RESET],
        ]),
        q_temperature=1.0,
    )
    legacy = dynamic_voc_observability_metrics(**kwargs)
    behavior = torch.tensor(
        [[0.11, 0.22], [0.33, 0.44]], requires_grad=True
    )
    augmented = dynamic_voc_observability_metrics(
        **kwargs, behavior_continue_probability=behavior
    )

    for key, value in legacy.items():
        assert key in augmented
        torch.testing.assert_close(augmented[key], value, rtol=0.0, atol=0.0)
    added = set(augmented) - set(legacy)
    assert added == {
        "voc_acceptance_behavior_continue_probability_delta_positive",
        "voc_acceptance_behavior_continue_probability_delta_negative",
        (
            "voc_acceptance_depth_8_plus_"
            "behavior_continue_probability_delta_positive"
        ),
        (
            "voc_acceptance_depth_8_plus_"
            "behavior_continue_probability_delta_negative"
        ),
        (
            "voc_post_useful_compute_"
            "behavior_continue_probability_delta_positive"
        ),
        (
            "voc_post_useful_compute_"
            "behavior_continue_probability_delta_negative"
        ),
    }
    for prefix in ("voc_acceptance", "voc_acceptance_depth_8_plus"):
        assert augmented[
            f"{prefix}_behavior_continue_probability_delta_positive"
        ].item() == pytest.approx(0.22)
        assert augmented[
            f"{prefix}_behavior_continue_probability_delta_negative"
        ].item() == pytest.approx(0.44)
    assert augmented[
        "voc_post_useful_compute_behavior_continue_probability_delta_positive"
    ].item() == pytest.approx(0.33)
    assert augmented[
        "voc_post_useful_compute_behavior_continue_probability_delta_negative"
    ].item() == pytest.approx(0.44)
    assert behavior.grad is None
    assert all(not value.requires_grad for value in augmented.values())


def test_voc_observability_strict_same_bout_useful_transition_slices():
    # Four strict pairs exercise both accepted controls and both signs on the
    # next decision.  The last two deliberately disagree with the sampled and
    # argmax gate so conditional rates cannot pass by counting support alone.
    # Two invalid columns contain masked NaNs and must never enter a pair.
    delta_q = torch.tensor([
        [2.0, 3.0, 4.0, 5.0, float("nan"), 8.0],
        [1.0, -1.0, -2.0, 2.0, 7.0, float("nan")],
    ], requires_grad=True)
    continue_probability = torch.tensor([
        [0.6, 0.6, 0.6, 0.6, float("nan"), 0.6],
        [0.8, 0.2, 0.7, 0.3, 0.9, float("nan")],
    ], requires_grad=True)
    control_action = torch.tensor([
        [
            util.PROCEED,
            util.RESET,
            util.PROCEED,
            util.RESET,
            util.PROCEED,
            util.RESET,
        ],
        [
            util.PROCEED,
            util.STOP,
            util.PROCEED,
            util.STOP,
            util.STOP,
            util.STOP,
        ],
    ])
    gate_action = torch.where(
        control_action == util.STOP,
        torch.ones_like(control_action),
        torch.zeros_like(control_action),
    )
    control_valid = torch.tensor([
        [True, True, True, True, False, True],
        [True, True, True, True, True, False],
    ])
    # For a selected P/R, search_steps is one larger than decision depth;
    # for STOP it already equals decision depth.  Every included column
    # therefore advances exactly 1 -> 2.
    search_steps = torch.tensor([
        [2, 2, 2, 2, 2, 2],
        [3, 2, 3, 2, 2, 2],
    ])
    predecision_last_control = torch.tensor([
        [util.STOP] * 6,
        [
            util.PROCEED,
            util.RESET,
            util.PROCEED,
            util.RESET,
            util.PROCEED,
            util.RESET,
        ],
    ])
    rng_before = torch.random.get_rng_state().clone()
    delta_before = delta_q.detach().clone()
    probability_before = continue_probability.detach().clone()

    metrics = dynamic_voc_observability_metrics(
        delta_q=delta_q,
        continue_probability=continue_probability,
        gate_action=gate_action,
        control_valid=control_valid,
        search_steps=search_steps,
        control_action=control_action,
        predecision_last_control=predecision_last_control,
        q_temperature=1.0,
    )

    assert torch.equal(torch.random.get_rng_state(), rng_before)
    torch.testing.assert_close(delta_q.detach(), delta_before, equal_nan=True)
    torch.testing.assert_close(
        continue_probability.detach(), probability_before, equal_nan=True
    )
    assert all(not value.requires_grad for value in metrics.values())
    assert all(torch.isfinite(value).item() for value in metrics.values())

    prefix = "voc_post_useful_compute"
    assert metrics[f"{prefix}_count"].item() == 4
    assert metrics[f"{prefix}_prior_useful_count"].item() == 4
    assert metrics[f"{prefix}_prior_delta_q"].item() == pytest.approx(3.5)
    assert metrics[f"{prefix}_delta_q_positive_count"].item() == 2
    assert metrics[f"{prefix}_delta_q_negative_count"].item() == 2
    assert metrics[f"{prefix}_delta_q_tie_count"].item() == 0
    assert metrics[f"{prefix}_delta_q_nontie_count"].item() == 4
    assert metrics[
        f"{prefix}_continue_probability_delta_positive"
    ].item() == pytest.approx(0.55)
    assert metrics[
        f"{prefix}_continue_probability_delta_negative"
    ].item() == pytest.approx(0.45)
    assert metrics[
        f"{prefix}_sampled_continue_given_delta_positive_rate"
    ].item() == pytest.approx(0.5)
    assert metrics[
        f"{prefix}_sampled_stop_given_delta_negative_rate"
    ].item() == pytest.approx(0.5)
    assert metrics[
        f"{prefix}_argmax_continue_given_delta_positive_rate"
    ].item() == pytest.approx(0.5)
    assert metrics[
        f"{prefix}_argmax_stop_given_delta_negative_rate"
    ].item() == pytest.approx(0.5)
    assert metrics[f"{prefix}_sign_gate_agreement_count"].item() == 4
    assert metrics[f"{prefix}_sign_gate_agreement_rate"].item() == 0.5

    proceed_prefix = "voc_post_useful_proceed"
    reset_prefix = "voc_post_useful_reset"
    assert metrics[f"{proceed_prefix}_count"].item() == 2
    assert metrics[f"{proceed_prefix}_prior_useful_count"].item() == 2
    assert metrics[f"{proceed_prefix}_prior_delta_q"].item() == 3.0
    assert metrics[f"{reset_prefix}_count"].item() == 2
    assert metrics[f"{reset_prefix}_prior_useful_count"].item() == 2
    assert metrics[f"{reset_prefix}_prior_delta_q"].item() == 4.0
    for split_prefix in (proceed_prefix, reset_prefix):
        assert metrics[f"{split_prefix}_delta_q_positive_count"].item() == 1
        assert metrics[f"{split_prefix}_delta_q_negative_count"].item() == 1
        assert metrics[f"{split_prefix}_sign_gate_agreement_count"].item() == 2
        assert metrics[f"{split_prefix}_sign_gate_agreement_rate"].item() == 0.5
    assert metrics[
        f"{proceed_prefix}_sampled_continue_given_delta_positive_rate"
    ].item() == 1.0
    assert metrics[
        f"{proceed_prefix}_sampled_stop_given_delta_negative_rate"
    ].item() == 0.0
    assert metrics[
        f"{reset_prefix}_sampled_continue_given_delta_positive_rate"
    ].item() == 0.0
    assert metrics[
        f"{reset_prefix}_sampled_stop_given_delta_negative_rate"
    ].item() == 1.0
    assert (
        metrics[f"{proceed_prefix}_count"]
        + metrics[f"{reset_prefix}_count"]
    ).item() == metrics[f"{prefix}_count"].item()

    # Candidate support includes accepted useful computations on the final
    # time row and before an invalid current row.  Neither can be a strict
    # pair, so both lower coverage rather than silently disappearing.
    assert metrics[f"{prefix}_prior_useful_candidate_count"].item() == 6
    assert metrics[f"{prefix}_transition_coverage_rate"].item() == pytest.approx(
        4 / 6
    )
    for split_prefix in (proceed_prefix, reset_prefix):
        assert (
            metrics[f"{split_prefix}_prior_useful_candidate_count"].item()
            == 3
        )
        assert metrics[
            f"{split_prefix}_transition_coverage_rate"
        ].item() == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    "prior_valid,current_valid,prior_action,current_token,"
    "prior_depth,current_depth,prior_delta",
    [
        (False, True, util.PROCEED, util.PROCEED, 1, 2, float("nan")),
        (True, False, util.PROCEED, util.PROCEED, 1, 2, 2.0),
        (True, True, util.STOP, util.STOP, 1, 2, 2.0),
        (True, True, util.PROCEED, util.RESET, 1, 2, 2.0),
        (True, True, util.RESET, util.RESET, 1, 3, 2.0),
        (True, True, util.PROCEED, util.PROCEED, 1, 2, -2.0),
        (True, True, util.RESET, util.RESET, 1, 2, 1e-6),
    ],
    ids=(
        "invalid-prior-masked-nan",
        "invalid-current",
        "prior-stop",
        "predecision-action-mismatch",
        "nonconsecutive-depth",
        "prior-negative",
        "prior-tie-boundary",
    ),
)
def test_voc_observability_strict_transition_rejects_every_mismatch(
    prior_valid,
    current_valid,
    prior_action,
    current_token,
    prior_depth,
    current_depth,
    prior_delta,
):
    current_action = util.STOP
    current_delta = 1.0 if current_valid else float("nan")
    current_probability = 0.8 if current_valid else float("nan")
    metrics = dynamic_voc_observability_metrics(
        delta_q=torch.tensor([[prior_delta], [current_delta]]),
        continue_probability=torch.tensor(
            [[0.6 if prior_valid else float("nan")], [current_probability]]
        ),
        gate_action=torch.tensor([
            [1 if prior_action == util.STOP else 0],
            [1],
        ]),
        control_valid=torch.tensor([[prior_valid], [current_valid]]),
        search_steps=torch.tensor([
            [prior_depth + int(prior_valid and prior_action != util.STOP)],
            [current_depth],
        ]),
        control_action=torch.tensor([[prior_action], [current_action]]),
        predecision_last_control=torch.tensor([
            [util.STOP],
            [current_token],
        ]),
        q_temperature=1.0,
    )

    assert all(torch.isfinite(value).item() for value in metrics.values())
    for prefix in (
        "voc_post_useful_compute",
        "voc_post_useful_proceed",
        "voc_post_useful_reset",
    ):
        assert metrics[f"{prefix}_count"].item() == 0
        assert metrics[f"{prefix}_prior_useful_count"].item() == 0
        assert metrics[f"{prefix}_prior_delta_q"].item() == 0
        assert metrics[f"{prefix}_delta_q_positive_count"].item() == 0

    is_candidate = (
        prior_valid
        and prior_action in (util.PROCEED, util.RESET)
        and prior_delta > 1e-6
    )
    assert metrics[
        "voc_post_useful_compute_prior_useful_candidate_count"
    ].item() == int(is_candidate)
    assert metrics[
        "voc_post_useful_compute_transition_coverage_rate"
    ].item() == 0


def test_voc_observability_strict_transition_excludes_first_time_row():
    metrics = dynamic_voc_observability_metrics(
        delta_q=torch.tensor([[3.0]]),
        continue_probability=torch.tensor([[0.9]]),
        gate_action=torch.tensor([[0]]),
        control_valid=torch.tensor([[True]]),
        search_steps=torch.tensor([[2]]),
        control_action=torch.tensor([[util.PROCEED]]),
        predecision_last_control=torch.tensor([[util.PROCEED]]),
        q_temperature=1.0,
    )

    assert metrics["voc_post_compute_count"].item() == 1
    assert metrics["voc_post_useful_compute_count"].item() == 0
    assert metrics["voc_post_useful_compute_prior_useful_count"].item() == 0
    assert (
        metrics["voc_post_useful_compute_prior_useful_candidate_count"].item()
        == 1
    )
    assert metrics["voc_post_useful_compute_transition_coverage_rate"].item() == 0


def test_voc_observability_argmax_tie_teacher_and_deep_acceptance_slice():
    metrics = dynamic_voc_observability_metrics(
        delta_q=torch.tensor([[1.0, -1.0]]),
        continue_probability=torch.tensor([[0.5, 0.5]]),
        gate_action=torch.tensor([[0, 1]]),
        control_valid=torch.tensor([[True, True]]),
        search_steps=torch.tensor([[8, 7]]),
        control_action=torch.tensor([[util.STOP, util.STOP]]),
        predecision_last_control=torch.tensor([[util.STOP, util.STOP]]),
        q_temperature=0.5,
    )

    # The actual [CONTINUE, STOP] argmax resolves an exact tie to index zero.
    # The legacy agreement key retains its historical strict p_continue > .5
    # definition, so one of these two signs agrees and one does not.
    assert metrics[
        "voc_acceptance_argmax_continue_given_delta_positive_rate"
    ].item() == 1.0
    assert metrics[
        "voc_acceptance_argmax_stop_given_delta_negative_rate"
    ].item() == 0.0
    assert metrics["voc_sign_gate_agreement_rate"].item() == 0.5

    assert metrics[
        "voc_acceptance_teacher_continue_probability_delta_positive"
    ].item() == pytest.approx(torch.sigmoid(torch.tensor(2.0)).item())
    assert metrics[
        "voc_acceptance_teacher_continue_probability_delta_negative"
    ].item() == pytest.approx(torch.sigmoid(torch.tensor(-2.0)).item())
    assert metrics["voc_acceptance_count"].item() == 2
    assert metrics["voc_acceptance_depth_8_plus_count"].item() == 1
    assert metrics[
        "voc_acceptance_depth_8_plus_delta_q_positive_count"
    ].item() == 1


def test_voc_observability_strict_pairs_never_cross_batch_streams():
    metrics = dynamic_voc_observability_metrics(
        delta_q=torch.tensor([[2.0, 3.0], [1.0, -1.0]]),
        continue_probability=torch.tensor([[0.7, 0.8], [0.6, 0.4]]),
        gate_action=torch.tensor([[0, 0], [0, 1]]),
        control_valid=torch.ones((2, 2), dtype=torch.bool),
        search_steps=torch.tensor([[2, 2], [2, 2]]),
        control_action=torch.tensor([
            [util.PROCEED, util.RESET],
            [util.STOP, util.STOP],
        ]),
        # Each next row matches the other batch stream only.  A flatten-based
        # predecessor implementation would admit these; strict [T,B] does not.
        predecision_last_control=torch.tensor([
            [util.STOP, util.STOP],
            [util.RESET, util.PROCEED],
        ]),
        q_temperature=1.0,
    )

    assert metrics["voc_post_useful_compute_prior_useful_count"].item() == 0
    assert (
        metrics["voc_post_useful_compute_prior_useful_candidate_count"].item()
        == 2
    )
    assert metrics["voc_post_useful_compute_transition_coverage_rate"].item() == 0


@pytest.mark.parametrize(
    "q_temperature", [True, 0.0, -1.0, float("nan"), float("inf")]
)
def test_voc_observability_rejects_invalid_q_temperature(q_temperature):
    with pytest.raises(ValueError, match="q_temperature"):
        dynamic_voc_observability_metrics(
            delta_q=torch.zeros((1, 1)),
            continue_probability=torch.full((1, 1), 0.5),
            gate_action=torch.zeros((1, 1), dtype=torch.long),
            control_valid=torch.ones((1, 1), dtype=torch.bool),
            search_steps=torch.zeros((1, 1), dtype=torch.long),
            control_action=torch.full((1, 1), util.STOP, dtype=torch.long),
            predecision_last_control=torch.full(
                (1, 1), util.STOP, dtype=torch.long
            ),
            q_temperature=q_temperature,
        )


def test_voc_observability_requires_time_batch_decisions():
    with pytest.raises(ValueError, match=r"shape \[T, B\]"):
        dynamic_voc_observability_metrics(
            delta_q=torch.zeros(2),
            continue_probability=torch.full((2,), 0.5),
            gate_action=torch.zeros(2, dtype=torch.long),
            control_valid=torch.ones(2, dtype=torch.bool),
            search_steps=torch.zeros(2, dtype=torch.long),
            control_action=torch.full((2,), util.STOP, dtype=torch.long),
            predecision_last_control=torch.full(
                (2,), util.STOP, dtype=torch.long
            ),
            q_temperature=1.0,
        )


def test_fresh_joint_three_state_mdp_learns_depth_re_evaluation():
    """Joint on-policy Q/gate learning solves easy, hard, and post-think states."""

    easy, hard, post_compute = range(3)
    computation_cost = 0.2
    q_parameter = nn.Parameter(torch.zeros(3, 2))
    gate_logits = nn.Parameter(torch.zeros(3, 3))
    q_optimizer = torch.optim.Adam([q_parameter], lr=0.04)
    gate_optimizer = torch.optim.Adam([gate_logits], lr=0.025)

    # A genuinely fresh control run starts with equal Q, so its first
    # return-directed gate gradient is exactly neutral.  Raw three-way zero
    # logits retain the legacy 2/3 CONTINUE mass; neutrality refers to the VoC
    # gradient, not to imposing an artificial 1/2 action distribution.
    initial_distribution = compute_voc_gate_distribution(
        gate_logits, temperature=1.0, epsilon=0.02
    )
    initial_result = compute_dynamic_voc_loss(
        voc_q=q_parameter.unsqueeze(0),
        target_control_logits=initial_distribution.joint_logits.unsqueeze(0),
        behavior_control_logits=(
            initial_distribution.joint_logits.detach().unsqueeze(0)
        ),
        control_action=torch.tensor([[
            util.STOP, util.PROCEED, util.RESET
        ]]),
        control_valid=torch.ones((1, 3), dtype=torch.bool),
        voc_target=torch.tensor([[1.0, 1.0, 1.0]]),
        mode="control",
    )
    torch.testing.assert_close(
        initial_result.gate_pg_loss, torch.tensor(0.0)
    )

    support = torch.zeros((3, 3), dtype=torch.long)
    state_visits = torch.zeros(3, dtype=torch.long)
    hard_continue_count = 0
    post_decision_count = 0
    torch.manual_seed(90210)

    def sample_state(state, sample_n):
        raw = gate_logits[state].expand(sample_n, 3)
        distribution = compute_voc_gate_distribution(
            raw, temperature=1.0, epsilon=0.02
        )
        action = torch.multinomial(
            torch.softmax(distribution.joint_logits.detach(), dim=-1), 1
        ).squeeze(-1)
        return action, distribution.joint_logits

    for _ in range(260):
        batch_n = 64
        easy_action, easy_logits = sample_state(easy, batch_n)
        hard_action, hard_logits = sample_state(hard, batch_n)
        hard_continue = hard_action != util.STOP
        post_n = int(hard_continue.sum().item())
        if post_n > 0:
            post_action, post_logits = sample_state(post_compute, post_n)
        else:
            post_action = torch.empty(0, dtype=torch.long)
            post_logits = torch.empty((0, 3))

        # Easy STOP returns 1.0.  Thinking cannot improve it: CONTINUE returns
        # only 0.4 and pays one computation cost.
        easy_task_reward = torch.where(
            easy_action == util.STOP,
            torch.ones(batch_n),
            torch.full((batch_n,), 0.4),
        )
        easy_think_reward = torch.where(
            easy_action == util.STOP,
            torch.zeros(batch_n),
            -torch.ones(batch_n),
        )
        easy_target = compute_dynamic_voc_target(
            task_rewards=easy_task_reward.unsqueeze(0),
            think_rewards=easy_think_reward.unsqueeze(0),
            task_discounts=torch.zeros((1, batch_n)),
            think_discounts=torch.zeros((1, batch_n)),
            task_vs=torch.zeros((1, batch_n)),
            think_vs=torch.zeros((1, batch_n)),
            task_bootstrap_value=torch.zeros(batch_n),
            think_bootstrap_value=torch.zeros(batch_n),
            think_cost=computation_cost,
        ).net[0]

        # Hard STOP returns zero.  CONTINUE pays one cost and reaches a depth-1
        # state with task return 1.5.  At that state another CONTINUE produces
        # no further task improvement and pays another cost, so the gate must
        # be invoked again and learn to STOP there.
        hard_target = torch.zeros(batch_n)
        if post_n > 0:
            post_think_reward = torch.where(
                post_action == util.STOP,
                torch.zeros(post_n),
                -torch.ones(post_n),
            )
            trajectory_target = compute_dynamic_voc_target(
                task_rewards=torch.stack((
                    torch.zeros(post_n),
                    torch.full((post_n,), 1.5),
                )),
                think_rewards=torch.stack((
                    -torch.ones(post_n), post_think_reward,
                )),
                task_discounts=torch.stack((
                    torch.ones(post_n), torch.zeros(post_n),
                )),
                think_discounts=torch.stack((
                    torch.ones(post_n), torch.zeros(post_n),
                )),
                # Exact terminal returns stand in for the recursive V-trace
                # values of the next state in this deterministic MDP.
                task_vs=torch.stack((
                    torch.zeros(post_n), torch.full((post_n,), 1.5),
                )),
                think_vs=torch.stack((
                    torch.zeros(post_n), post_think_reward,
                )),
                task_bootstrap_value=torch.zeros(post_n),
                think_bootstrap_value=torch.zeros(post_n),
                think_cost=computation_cost,
            ).net
            hard_target[hard_continue] = trajectory_target[0]
            post_target = trajectory_target[1]

        state_parts = [
            torch.full((batch_n,), easy, dtype=torch.long),
            torch.full((batch_n,), hard, dtype=torch.long),
        ]
        action_parts = [easy_action, hard_action]
        target_parts = [easy_target, hard_target]
        logit_parts = [easy_logits, hard_logits]
        if post_n > 0:
            state_parts.append(torch.full(
                (post_n,), post_compute, dtype=torch.long
            ))
            action_parts.append(post_action)
            target_parts.append(post_target)
            logit_parts.append(post_logits)

        state_index = torch.cat(state_parts)
        sampled_control = torch.cat(action_parts)
        target = torch.cat(target_parts)
        transformed_logits = torch.cat(logit_parts)
        row_n = sampled_control.numel()
        for state in (easy, hard, post_compute):
            state_mask = state_index == state
            state_visits[state] += state_mask.sum()
            for control in (util.PROCEED, util.RESET, util.STOP):
                support[state, control] += (
                    state_mask & (sampled_control == control)
                ).sum()

        result = compute_dynamic_voc_loss(
            voc_q=q_parameter[state_index].unsqueeze(0),
            target_control_logits=transformed_logits.unsqueeze(0),
            behavior_control_logits=transformed_logits.detach().unsqueeze(0),
            control_action=sampled_control.unsqueeze(0),
            control_valid=torch.ones((1, row_n), dtype=torch.bool),
            voc_target=target.unsqueeze(0),
            mode="control",
        )
        q_optimizer.zero_grad()
        gate_optimizer.zero_grad()
        ((result.q_loss + result.gate_pg_loss) / row_n).backward()
        q_optimizer.step()
        gate_optimizer.step()

        hard_continue_count += post_n
        post_decision_count += post_action.numel()

    final_distribution = compute_voc_gate_distribution(
        gate_logits.detach(), temperature=1.0, epsilon=0.0
    )
    final_continue = final_distribution.continue_prob

    assert q_parameter[easy, 1] > q_parameter[easy, 0]
    assert q_parameter[hard, 0] > q_parameter[hard, 1]
    assert q_parameter[post_compute, 1] > q_parameter[post_compute, 0]
    assert final_continue[easy].item() < 0.1
    assert final_continue[hard].item() > 0.9
    assert final_continue[post_compute].item() < 0.1
    # Every useful depth-0 CONTINUE triggers a separate depth-1 decision.
    assert hard_continue_count == post_decision_count
    assert state_visits[post_compute].item() == post_decision_count
    # On-policy training supplied both binary actions in every state, and the
    # unchanged conditional bout supplied both PROCEED and RESET samples.
    assert torch.all(support[:, util.STOP] > 0)
    assert torch.all(support[:, :2].sum(dim=-1) > 0)
    assert torch.all(support[:, util.PROCEED] > 0)
    assert torch.all(support[:, util.RESET] > 0)
    # Q only directs a soft policy gradient; finite softmax probabilities are
    # retained rather than replaced by an argmax cutoff.
    assert torch.all((final_continue > 0.0) & (final_continue < 1.0))


def test_shadow_q_loss_is_selected_action_huber_and_has_no_gate_gradient():
    logits = torch.zeros((1, 2, 3), requires_grad=True)
    q = torch.tensor(
        [[[2.0, 7.0], [3.0, -1.0]]], requires_grad=True
    )
    result = compute_dynamic_voc_loss(
        voc_q=q,
        target_control_logits=logits,
        behavior_control_logits=torch.zeros_like(logits),
        control_action=torch.tensor(
            [[util.PROCEED, util.STOP]], dtype=torch.long
        ),
        control_valid=torch.tensor([[True, True]]),
        voc_target=torch.tensor([[4.0, 1.0]]),
        mode="shadow",
    )

    # Both selected errors have magnitude two: smooth-L1 is 1.5 each.
    torch.testing.assert_close(result.q_loss, torch.tensor(3.0))
    torch.testing.assert_close(result.gate_pg_loss, torch.tensor(0.0))
    result.q_loss.backward()
    assert logits.grad is None
    torch.testing.assert_close(
        q.grad,
        torch.tensor([[[-1.0, 0.0], [0.0, -1.0]]]),
    )


def test_voc_masks_invalid_controls_from_both_losses_and_support():
    logits = torch.zeros((1, 2, 3), requires_grad=True)
    q = torch.zeros((1, 2, 2), requires_grad=True)
    result = compute_dynamic_voc_loss(
        voc_q=q,
        target_control_logits=logits,
        behavior_control_logits=torch.zeros_like(logits),
        control_action=torch.tensor(
            [[util.PROCEED, util.STOP]], dtype=torch.long
        ),
        control_valid=torch.tensor([[True, False]]),
        voc_target=torch.tensor([[2.0, 1000.0]]),
        mode="control",
    )

    # Only the valid CONTINUE row contributes (smooth-L1(0, 2) = 1.5).
    torch.testing.assert_close(result.q_loss, torch.tensor(1.5))
    assert result.valid.sum().item() == 1


def test_voc_actor_stream_holdout_is_stable_and_excluded_from_q_loss():
    valid = torch.ones((2, 3), dtype=torch.bool)
    holdout = dynamic_voc_holdout_mask(
        torch.tensor([[0, 1, 8]]), valid
    )
    torch.testing.assert_close(
        holdout,
        torch.tensor([[True, False, True], [True, False, True]]),
    )

    q = torch.zeros((1, 2, 2), requires_grad=True)
    result = compute_dynamic_voc_loss(
        voc_q=q,
        target_control_logits=torch.zeros((1, 2, 3)),
        behavior_control_logits=torch.zeros((1, 2, 3)),
        control_action=torch.tensor([[util.PROCEED, util.PROCEED]]),
        control_valid=torch.tensor([[True, True]]),
        q_train_valid=torch.tensor([[False, True]]),
        voc_target=torch.tensor([[1000.0, 2.0]]),
        mode="shadow",
    )
    # The held-out 1000-error row is measured but never optimized.
    torch.testing.assert_close(result.q_loss, torch.tensor(1.5))
    result.q_loss.backward()
    torch.testing.assert_close(q.grad[0, 0], torch.zeros(2))
    assert q.grad[0, 1, 0].item() == pytest.approx(-1.0)


def test_voc_holdout_never_leaks_from_an_all_heldout_ppo_minibatch():
    valid = torch.ones((2, 2), dtype=torch.bool)
    holdout = dynamic_voc_holdout_mask(
        torch.tensor([0, 8]), valid, total_actor_streams=16
    )
    assert torch.equal(holdout, valid)

    q = torch.zeros((2, 2, 2), requires_grad=True)
    result = compute_dynamic_voc_loss(
        voc_q=q,
        target_control_logits=torch.zeros((2, 2, 3)),
        behavior_control_logits=torch.zeros((2, 2, 3)),
        control_action=torch.full((2, 2), util.PROCEED),
        control_valid=valid,
        q_train_valid=valid & ~holdout,
        voc_target=torch.full((2, 2), 1000.0),
        mode="shadow",
    )
    torch.testing.assert_close(result.q_loss, torch.tensor(0.0))
    result.q_loss.backward()
    torch.testing.assert_close(q.grad, torch.zeros_like(q))

    # This exception is based on the full topology, never the minibatch width.
    single_stream = dynamic_voc_holdout_mask(
        torch.tensor([0]), torch.ones((2, 1), dtype=torch.bool),
        total_actor_streams=1,
    )
    assert not single_stream.any()


def test_all_heldout_voc_batch_fails_fast_on_nonfinite_q():
    q = torch.zeros((1, 1, 2))
    q[..., 1] = float("nan")
    with pytest.raises(FloatingPointError, match="VoC Q outputs"):
        compute_dynamic_voc_loss(
            voc_q=q,
            target_control_logits=torch.zeros((1, 1, 3)),
            behavior_control_logits=torch.zeros((1, 1, 3)),
            control_action=torch.tensor([[util.PROCEED]]),
            control_valid=torch.tensor([[True]]),
            q_train_valid=torch.tensor([[False]]),
            voc_target=torch.zeros((1, 1)),
            mode="shadow",
        )


def test_voc_rejects_out_of_range_valid_control_action():
    with pytest.raises(ValueError, match="outside"):
        compute_dynamic_voc_loss(
            voc_q=torch.zeros((1, 1, 2)),
            target_control_logits=torch.zeros((1, 1, 3)),
            behavior_control_logits=torch.zeros((1, 1, 3)),
            control_action=torch.tensor([[99]], dtype=torch.long),
            control_valid=torch.tensor([[True]]),
            voc_target=torch.zeros((1, 1)),
            mode="shadow",
        )


def test_nested_voc_head_parameters_are_found_without_shared_parameters():
    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = nn.Linear(3, 3)
            self.critic = nn.Module()
            self.critic.voc_head = nn.Linear(3, 2)

    network = Network()
    found = SActorLearner._find_voc_head_parameters(network)
    expected = list(network.critic.voc_head.parameters())

    assert {id(parameter) for parameter in found} == {
        id(parameter) for parameter in expected
    }
    assert {id(parameter) for parameter in found}.isdisjoint(
        {id(parameter) for parameter in network.shared.parameters()}
    )


def test_ema_gate_loss_is_fp32_frozen_and_only_updates_live_gate_logits():
    learner = object.__new__(SActorLearner)
    learner.dynamic_voc_mode = "control"
    learner.voc_ema_gate_weight = torch.tensor(
        [[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float32
    )
    learner.voc_ema_gate_bias = torch.zeros(2, dtype=torch.float32)
    features = torch.tensor([[[1.0, 0.0]]], dtype=torch.float16)
    logits = torch.zeros((1, 1, 3), requires_grad=True)
    state_value = torch.tensor([[4.0]], requires_grad=True)

    gate_loss, gate_q = learner._compute_ema_gate_loss(
        features=features,
        logits=logits,
        valid=torch.tensor([[True]]),
        state_value=state_value,
    )

    assert gate_q.dtype == torch.float32
    assert not gate_q.requires_grad
    torch.testing.assert_close(
        gate_q[..., 0] - gate_q[..., 1], torch.tensor([[2.0]])
    )
    gate_loss.backward()
    assert logits.grad[..., util.PROCEED].item() < 0.0
    assert logits.grad[..., util.RESET].item() < 0.0
    assert logits.grad[..., util.STOP].item() > 0.0
    torch.testing.assert_close(
        logits.grad[..., util.PROCEED], logits.grad[..., util.RESET]
    )
    assert state_value.grad is None
    assert learner.voc_ema_gate_weight.grad is None
    assert learner.voc_ema_gate_bias.grad is None


def test_ema_gate_polyak_update_is_atomic_and_counts_only_successes():
    learner = object.__new__(SActorLearner)
    learner.voc_online_head = nn.Linear(2, 2)
    learner.voc_gate_target_tau = 0.1
    learner.voc_ema_gate_weight = torch.zeros((2, 2), dtype=torch.float32)
    learner.voc_ema_gate_bias = torch.zeros(2, dtype=torch.float32)
    learner.voc_ema_gate_update_count = 0
    with torch.no_grad():
        learner.voc_online_head.weight.copy_(
            torch.tensor([[2.0, 4.0], [-2.0, -4.0]])
        )
        learner.voc_online_head.bias.copy_(torch.tensor([1.0, -1.0]))

    learner._update_voc_ema_gate_target()
    torch.testing.assert_close(
        learner.voc_ema_gate_weight,
        0.1 * learner.voc_online_head.weight.detach(),
    )
    torch.testing.assert_close(
        learner.voc_ema_gate_bias,
        0.1 * learner.voc_online_head.bias.detach(),
    )
    assert learner.voc_ema_gate_update_count == 1

    weight_before = learner.voc_ema_gate_weight.clone()
    bias_before = learner.voc_ema_gate_bias.clone()
    with torch.no_grad():
        learner.voc_online_head.weight[0, 0] = float("inf")
    with pytest.raises(FloatingPointError, match="post-step online VoC weight"):
        learner._update_voc_ema_gate_target()
    torch.testing.assert_close(learner.voc_ema_gate_weight, weight_before)
    torch.testing.assert_close(learner.voc_ema_gate_bias, bias_before)
    assert learner.voc_ema_gate_update_count == 1


_SCHEMA12_EMA_DEVICES = [
    pytest.param("cpu", id="cpu"),
    pytest.param(
        "cuda",
        id="cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA is unavailable"
        ),
    ),
]


def _schema12_ema_case(case, device):
    dtype = torch.float32
    if case == "normal":
        old_weight = torch.tensor(
            [[-9.0, 2.5], [7.0, -3.0]], dtype=dtype, device=device
        )
        old_bias = torch.tensor([4.0, -5.0], dtype=dtype, device=device)
        online_weight = torch.tensor(
            [[3.25, -4.5], [5.75, -6.125]], dtype=dtype, device=device
        )
        online_bias = torch.tensor([7.5, -8.25], dtype=dtype, device=device)
    elif case == "signed-zero":
        old_weight = torch.tensor(
            [[1.0, -1.0], [-2.0, 2.0]], dtype=dtype, device=device
        )
        old_bias = torch.tensor([3.0, -3.0], dtype=dtype, device=device)
        online_weight = torch.tensor(
            [[-0.0, 0.0], [0.0, -0.0]], dtype=dtype, device=device
        )
        online_bias = torch.tensor([-0.0, 0.0], dtype=dtype, device=device)
    elif case == "subnormal":
        zero = torch.tensor(0.0, dtype=dtype, device=device)
        tiny = torch.nextafter(zero, torch.tensor(1.0, dtype=dtype, device=device))
        old_weight = torch.tensor(
            [[2.0, -2.0], [-3.0, 3.0]], dtype=dtype, device=device
        )
        old_bias = torch.tensor([4.0, -4.0], dtype=dtype, device=device)
        online_weight = torch.stack((tiny, -tiny, tiny, -tiny)).reshape(2, 2)
        online_bias = torch.stack((-tiny, tiny))
    elif case == "extreme":
        limit = torch.finfo(dtype).max
        old_weight = torch.tensor(
            [[1.0, -1.0], [-2.0, 2.0]], dtype=dtype, device=device
        )
        old_bias = torch.tensor([3.0, -3.0], dtype=dtype, device=device)
        online_weight = torch.tensor(
            [[limit, -limit], [-limit, limit]], dtype=dtype, device=device
        )
        online_bias = torch.tensor([-limit, limit], dtype=dtype, device=device)
    else:
        raise AssertionError(f"unknown schema-12 EMA case {case}")
    return old_weight, old_bias, online_weight, online_bias


@pytest.mark.parametrize("device", _SCHEMA12_EMA_DEVICES)
@pytest.mark.parametrize("case", ["normal", "signed-zero", "subnormal", "extreme"])
def test_schema12_tau_one_ema_retains_inherited_fp32_arithmetic(
    device, case
):
    old_weight, old_bias, online_weight, online_bias = _schema12_ema_case(
        case, device
    )
    learner = object.__new__(SActorLearner)
    learner.voc_gate_policy_schema_version = (
        util.VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION
    )
    learner.voc_gate_target_tau = 1.0
    learner.voc_online_head = nn.Linear(2, 2, device=device, dtype=torch.float32)
    with torch.no_grad():
        learner.voc_online_head.weight.copy_(online_weight)
        learner.voc_online_head.bias.copy_(online_bias)
    learner.voc_ema_gate_weight = old_weight.clone()
    learner.voc_ema_gate_bias = old_bias.clone()
    learner.voc_ema_gate_update_count = 0

    expected_weight = (
        (1.0 - learner.voc_gate_target_tau) * old_weight
        + learner.voc_gate_target_tau * online_weight
    )
    expected_bias = (
        (1.0 - learner.voc_gate_target_tau) * old_bias
        + learner.voc_gate_target_tau * online_bias
    )
    assert torch.equal(expected_weight, online_weight)
    assert torch.equal(expected_bias, online_bias)

    learner._update_voc_ema_gate_target()

    assert learner.voc_ema_gate_update_count == 1
    assert torch.equal(learner.voc_ema_gate_weight, expected_weight)
    assert torch.equal(learner.voc_ema_gate_bias, expected_bias)
    assert torch.equal(learner.voc_ema_gate_weight, online_weight)
    assert torch.equal(learner.voc_ema_gate_bias, online_bias)
    assert learner.voc_ema_gate_weight.data_ptr() != online_weight.data_ptr()
    assert learner.voc_ema_gate_bias.data_ptr() != online_bias.data_ptr()
    if case == "signed-zero":
        assert not torch.equal(
            expected_weight.view(torch.int32), online_weight.view(torch.int32)
        )
        assert not torch.equal(
            learner.voc_ema_gate_weight.view(torch.int32),
            online_weight.view(torch.int32),
        )


@pytest.mark.parametrize("mismatch", ["weight", "bias"])
def test_schema12_tau_one_equality_failure_validates_both_before_mutation(
    mismatch,
):
    learner = object.__new__(SActorLearner)
    learner.voc_gate_policy_schema_version = (
        util.VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION
    )
    # Internal tampering supplies a finite non-one tau so the corresponding
    # candidate exercises the precommit equality failure branch directly.
    learner.voc_gate_target_tau = 0.5
    learner.voc_online_head = nn.Linear(2, 2)
    with torch.no_grad():
        learner.voc_online_head.weight.fill_(2.0)
        learner.voc_online_head.bias.fill_(3.0)
    learner.voc_ema_gate_weight = learner.voc_online_head.weight.detach().clone()
    learner.voc_ema_gate_bias = learner.voc_online_head.bias.detach().clone()
    if mismatch == "weight":
        learner.voc_ema_gate_weight.add_(4.0)
    else:
        learner.voc_ema_gate_bias.add_(4.0)
    weight_before = learner.voc_ema_gate_weight.clone()
    bias_before = learner.voc_ema_gate_bias.clone()
    learner.voc_ema_gate_update_count = 9

    with pytest.raises(RuntimeError, match=f"EMA {mismatch} candidate"):
        learner._update_voc_ema_gate_target()

    assert torch.equal(learner.voc_ema_gate_weight, weight_before)
    assert torch.equal(learner.voc_ema_gate_bias, bias_before)
    assert learner.voc_ema_gate_update_count == 9


def _learner_flags(tmp_path, xpid, *, mode="control", ppo_k=1):
    flags = cenv_flags(cap=4)
    flags.dynamic_factorized_control = True
    flags.dynamic_voc_mode = mode
    flags.think_cost = 0.0005
    flags.think_cost_anneal = False
    flags.voc_parent_checkpoint = ""
    flags.preload = ""
    flags.preload_actor = ""
    flags.float16 = False
    flags.see_real_state = False
    flags.ppo_k = ppo_k
    flags.ppo_n = ppo_k
    flags.return_norm_type = -1
    flags.parallel_actor = False
    flags.actor_batch_size = 3
    flags.env_n = 3
    flags.self_play_n = 1
    flags.total_steps = 100
    flags.ckp = False
    flags.checkpoint_interval = 0
    flags.savedir = str(tmp_path)
    flags.xpid = xpid
    flags.ckpdir = str(tmp_path / xpid)
    return flags


@pytest.mark.parametrize("value", (1, 0, "true", None))
def test_voc_gate_parameter_alignment_rejects_nonboolean_flag(
    tmp_path, value
):
    flags = _learner_flags(tmp_path, "voc-gate-param-align-invalid-flag")
    flags.voc_gate_param_align = value

    with pytest.raises(ValueError, match="voc_gate_param_align must be boolean"):
        SActorLearner(
            ray_obj=None,
            actor_param={},
            flags=flags,
            actor_net=None,
            device=None,
        )


@pytest.mark.parametrize("value", (1, 0, "true", None))
def test_epsilon_greedy_execution_rejects_nonboolean_direct_flag(
    tmp_path, value
):
    flags = _learner_flags(tmp_path, "voc-epsilon-execution-invalid-flag")
    flags.voc_gate_epsilon_greedy_execution = value

    with pytest.raises(
        ValueError, match="voc_gate_epsilon_greedy_execution must be boolean"
    ):
        SActorLearner(
            ray_obj=None,
            actor_param={},
            flags=flags,
            actor_net=None,
            device=None,
        )


@pytest.mark.parametrize(
    "overrides",
    (
        {"dynamic_voc_mode": "shadow"},
        {"voc_gate_exact_projection": False},
        {"voc_gate_param_align": True},
    ),
)
def test_epsilon_greedy_execution_direct_flags_require_v11_exact_base(
    tmp_path, overrides
):
    flags = _epsilon_execution_learner_flags(
        tmp_path, "voc-epsilon-execution-invalid-base"
    )
    for name, value in overrides.items():
        setattr(flags, name, value)

    with pytest.raises(
        ValueError,
        match="epsilon_greedy_execution|exact_projection|mutually exclusive",
    ):
        SActorLearner(
            ray_obj=None,
            actor_param={},
            flags=flags,
            actor_net=None,
            device=None,
        )


@pytest.mark.parametrize(
    "coefficient",
    (True, False, 0.0, 0.5, 2.0, float("nan"), float("inf"), "1.0"),
)
def test_voc_gate_parameter_alignment_requires_exact_unit_coefficient(
    tmp_path, coefficient
):
    flags = _learner_flags(tmp_path, "voc-gate-param-align-invalid-coef")
    flags.voc_gate_param_align = True
    flags.voc_gate_param_align_coef = coefficient

    with pytest.raises(ValueError, match="coef=1.0 exactly"):
        SActorLearner(
            ray_obj=None,
            actor_param={},
            flags=flags,
            actor_net=None,
            device=None,
        )


@pytest.mark.parametrize("value", (1, 0, "true", None))
def test_exact_gate_projection_rejects_nonboolean_flag(tmp_path, value):
    flags = _learner_flags(tmp_path, "voc-gate-projection-invalid-flag")
    flags.voc_gate_exact_projection = value

    with pytest.raises(
        ValueError, match="voc_gate_exact_projection must be boolean"
    ):
        SActorLearner(
            ray_obj=None,
            actor_param={},
            flags=flags,
            actor_net=None,
            device=None,
        )


@pytest.mark.parametrize(
    "coefficient",
    (
        True,
        False,
        0.5,
        float("nan"),
        float("inf"),
        "1.0",
        math.nextafter(1.0, 2.0),
    ),
)
def test_exact_gate_projection_direct_constructor_requires_unit_coefficient(
    tmp_path, coefficient
):
    flags = _learner_flags(tmp_path, "voc-projection-invalid-coef")
    flags.voc_gate_param_align = False
    flags.voc_gate_exact_projection = True
    flags.voc_gate_param_align_coef = coefficient

    with pytest.raises(ValueError, match="coef=1.0 exactly"):
        SActorLearner(
            ray_obj=None,
            actor_param={},
            flags=flags,
            actor_net=None,
            device=None,
        )


def test_exact_gate_projection_direct_constructor_rejects_shadow(tmp_path):
    flags = _learner_flags(
        tmp_path, "voc-projection-shadow-invalid", mode="shadow"
    )
    flags.voc_gate_exact_projection = True

    with pytest.raises(ValueError, match="requires control mode"):
        SActorLearner(
            ray_obj=None,
            actor_param={},
            flags=flags,
            actor_net=None,
            device=None,
        )


@pytest.mark.parametrize(
    "name", ("preload", "preload_actor", "voc_parent_checkpoint")
)
def test_exact_gate_projection_direct_resume_rejects_preload_surfaces(
    tmp_path, name
):
    flags = _learner_flags(tmp_path, "voc-projection-resume-parent-invalid")
    flags.voc_gate_exact_projection = True
    flags.ckp = True
    setattr(flags, name, "/forbidden-parent")

    with pytest.raises(ValueError, match=rf"{name}=''"):
        SActorLearner(
            ray_obj=None,
            actor_param={},
            flags=flags,
            actor_net=None,
            device=None,
        )


@pytest.mark.parametrize(
    "beta1",
    (True, False, -0.01, 1.0, float("nan"), float("inf"), "invalid"),
)
def test_voc_gate_adam_beta1_rejects_invalid_values(tmp_path, beta1):
    flags = _learner_flags(tmp_path, "voc-gate-invalid-beta1")
    flags.voc_gate_adam_beta1 = beta1

    with pytest.raises(
        ValueError, match=r"voc_gate_adam_beta1.*\[0, 1\)"
    ):
        SActorLearner(
            ray_obj=None,
            actor_param={},
            flags=flags,
            actor_net=None,
            device=None,
        )


def test_voc_gate_beta1_is_normalized_and_isolated_from_actor_and_q_adam(
    tmp_path, monkeypatch
):
    beta0_flags = _learner_flags(tmp_path, "voc-gate-beta0-isolation")
    beta0_flags.voc_gate_adam_beta1 = "0"
    default_flags = copy.deepcopy(beta0_flags)
    default_flags.xpid = "voc-gate-default-beta-isolation"
    default_flags.ckpdir = str(tmp_path / default_flags.xpid)
    delattr(default_flags, "voc_gate_adam_beta1")
    actor, _train_out, _initial_actor_state = _rollout(beta0_flags)
    default_actor = copy.deepcopy(actor)
    monkeypatch.setattr(
        "thinker.learn_actor.FileWriter", lambda **_kwargs: _NullWriter()
    )

    beta0 = SActorLearner(
        ray_obj=None,
        actor_param={},
        flags=beta0_flags,
        actor_net=actor,
        device=torch.device("cpu"),
    )
    default = SActorLearner(
        ray_obj=None,
        actor_param={},
        flags=default_flags,
        actor_net=default_actor,
        device=torch.device("cpu"),
    )

    assert beta0.voc_gate_adam_beta1 == 0.0
    assert beta0.flags.voc_gate_adam_beta1 == 0.0
    assert default.voc_gate_adam_beta1 == 0.9
    assert default.flags.voc_gate_adam_beta1 == 0.9
    assert isinstance(beta0.voc_gate_optimizer, torch.optim.Adam)
    assert beta0.voc_gate_optimizer.param_groups[0]["betas"] == (
        0.0,
        0.999,
    )
    assert default.voc_gate_optimizer.param_groups[0]["betas"] == (
        0.9,
        0.999,
    )

    def optimizer_parameters(optimizer):
        return [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]

    gate_ids = {id(parameter) for parameter in beta0.voc_gate_parameters}
    actor_ids = {
        id(parameter) for parameter in optimizer_parameters(beta0.optimizer)
    }
    q_ids = {
        id(parameter)
        for parameter in optimizer_parameters(beta0.voc_optimizer)
    }
    assert gate_ids == {
        id(parameter)
        for parameter in optimizer_parameters(beta0.voc_gate_optimizer)
    }
    assert actor_ids.isdisjoint(q_ids)
    assert actor_ids.isdisjoint(gate_ids)
    assert q_ids.isdisjoint(gate_ids)

    # A gate-only beta change must leave both other Adam configurations and
    # their first optimizer transition bit-exact.
    for name in ("optimizer", "voc_optimizer"):
        left_optimizer = getattr(beta0, name)
        right_optimizer = getattr(default, name)
        assert isinstance(left_optimizer, torch.optim.Adam)
        assert isinstance(right_optimizer, torch.optim.Adam)
        assert left_optimizer.param_groups[0]["betas"] == (0.9, 0.999)
        assert right_optimizer.param_groups[0]["betas"] == (0.9, 0.999)
        left_parameters = optimizer_parameters(left_optimizer)
        right_parameters = optimizer_parameters(right_optimizer)
        assert len(left_parameters) == len(right_parameters)
        for index, (left, right) in enumerate(
            zip(left_parameters, right_parameters)
        ):
            torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
            gradient = torch.full_like(left, (index % 7 + 1) * 1e-4)
            left.grad = gradient.clone()
            right.grad = gradient.clone()
        left_optimizer.step()
        right_optimizer.step()
        for left, right in zip(left_parameters, right_parameters):
            torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            left_optimizer.state_dict(),
            right_optimizer.state_dict(),
            rtol=0.0,
            atol=0.0,
        )

    actor_state_before_gate = copy.deepcopy(beta0.optimizer.state_dict())
    q_state_before_gate = copy.deepcopy(beta0.voc_optimizer.state_dict())
    gate_parameters = optimizer_parameters(beta0.voc_gate_optimizer)
    parameters_before = [parameter.detach().clone() for parameter in gate_parameters]
    gradients = []
    for index, parameter in enumerate(gate_parameters):
        gradient = torch.full_like(parameter, (index + 1) * 1e-3)
        parameter.grad = gradient.clone()
        gradients.append(gradient)

    step_result = beta0._step_voc_gate_optimizer()

    assert step_result.optimizer_stepped is True
    learning_rate = beta0.voc_gate_optimizer.param_groups[0]["lr"]
    epsilon = beta0.voc_gate_optimizer.param_groups[0]["eps"]
    for parameter, before, gradient in zip(
        gate_parameters, parameters_before, gradients
    ):
        expected = before - learning_rate * gradient / (
            gradient.abs() + epsilon
        )
        torch.testing.assert_close(parameter, expected)
        state = beta0.voc_gate_optimizer.state[parameter]
        assert state["step"].item() == 1
        torch.testing.assert_close(
            state["exp_avg"], gradient, rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            state["exp_avg_sq"],
            (1.0 - 0.999) * gradient.square(),
        )
    torch.testing.assert_close(
        beta0.optimizer.state_dict(),
        actor_state_before_gate,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        beta0.voc_optimizer.state_dict(),
        q_state_before_gate,
        rtol=0.0,
        atol=0.0,
    )


def test_voc_gate_beta1_keeps_legacy_rms_optimizer_path(
    tmp_path, monkeypatch
):
    flags = _learner_flags(tmp_path, "voc-gate-rms-beta1")
    flags.actor_use_rms = True
    flags.voc_gate_adam_beta1 = 0.0
    actor, _train_out, _initial_actor_state = _rollout(flags)
    monkeypatch.setattr(
        "thinker.learn_actor.FileWriter", lambda **_kwargs: _NullWriter()
    )

    learner = SActorLearner(
        ray_obj=None,
        actor_param={},
        flags=flags,
        actor_net=actor,
        device=torch.device("cpu"),
    )

    for optimizer in (
        learner.optimizer,
        learner.voc_optimizer,
        learner.voc_gate_optimizer,
    ):
        assert isinstance(optimizer, torch.optim.RMSprop)
        group = optimizer.param_groups[0]
        assert group["alpha"] == 0.99
        assert group["momentum"] == 0
        assert group["eps"] == 0.01
        assert group["centered"] is False


def test_voc_gate_beta0_checkpoint_restores_exact_next_optimizer_step(
    tmp_path, monkeypatch
):
    flags = _learner_flags(tmp_path, "voc-gate-beta0-resume")
    flags.voc_gate_adam_beta1 = 0.0
    (tmp_path / flags.xpid).mkdir()
    actor, train_out, initial_actor_state = _rollout(flags)
    monkeypatch.setattr(
        "thinker.learn_actor.FileWriter", lambda **_kwargs: _NullWriter()
    )
    learner = SActorLearner(
        ray_obj=None,
        actor_param={},
        flags=flags,
        actor_net=actor,
        device=torch.device("cpu"),
    )
    gate_scheduler_before = copy.deepcopy(
        learner.voc_gate_scheduler.state_dict()
    )
    learner.consume_data_single((train_out, initial_actor_state))

    # The fresh equal-Q row is a true tie.  beta1=0 changes only the Adam
    # time constant: it must not invent a first-update direction or disturb
    # the successful-step accounting around the gate.
    assert learner.voc_update_count == 1
    assert learner.voc_ema_gate_update_count == 1
    assert learner.voc_gate_update_count == 1
    assert learner.voc_gate_scaler is None
    assert all(
        torch.count_nonzero(parameter).item() == 0
        for parameter in learner.voc_gate_parameters
    )
    for parameter in learner.voc_gate_parameters:
        state = learner.voc_gate_optimizer.state[parameter]
        assert state["step"].item() == 1
        assert torch.count_nonzero(state["exp_avg"]).item() == 0
        assert torch.count_nonzero(state["exp_avg_sq"]).item() == 0
    gate_scheduler_after = learner.voc_gate_scheduler.state_dict()
    assert gate_scheduler_after["_step_count"] == (
        gate_scheduler_before["_step_count"] + 1
    )
    assert gate_scheduler_after["last_epoch"] == (
        gate_scheduler_before["last_epoch"] + 1
    )
    learner.save_checkpoint()
    checkpoint = torch.load(
        learner.ckp_path, map_location="cpu", weights_only=False
    )

    assert checkpoint["flags"]["voc_gate_adam_beta1"] == 0.0
    assert checkpoint["voc_gate_optimizer_state_dict"]["param_groups"][0][
        "betas"
    ] == (0.0, 0.999)

    resume_flags = copy.deepcopy(flags)
    resume_flags.ckp = True
    resumed = SActorLearner(
        ray_obj=None,
        actor_param={},
        flags=resume_flags,
        actor_net=copy.deepcopy(actor),
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(
        resumed.voc_gate_optimizer.state_dict(),
        learner.voc_gate_optimizer.state_dict(),
        rtol=0.0,
        atol=0.0,
    )

    learner_actor_state = copy.deepcopy(learner.optimizer.state_dict())
    learner_q_state = copy.deepcopy(learner.voc_optimizer.state_dict())
    resumed_actor_state = copy.deepcopy(resumed.optimizer.state_dict())
    resumed_q_state = copy.deepcopy(resumed.voc_optimizer.state_dict())
    for index, (original_parameter, resumed_parameter) in enumerate(
        zip(learner.voc_gate_parameters, resumed.voc_gate_parameters)
    ):
        gradient = torch.full_like(
            original_parameter, (index + 1) * 2e-3
        )
        original_parameter.grad = gradient.clone()
        resumed_parameter.grad = gradient.clone()

    original_result = learner._step_voc_gate_optimizer()
    resumed_result = resumed._step_voc_gate_optimizer()

    assert original_result.optimizer_stepped is True
    assert resumed_result.optimizer_stepped is True
    for original_parameter, resumed_parameter in zip(
        learner.voc_gate_parameters, resumed.voc_gate_parameters
    ):
        torch.testing.assert_close(
            original_parameter,
            resumed_parameter,
            rtol=0.0,
            atol=0.0,
        )
    torch.testing.assert_close(
        resumed.voc_gate_optimizer.state_dict(),
        learner.voc_gate_optimizer.state_dict(),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        learner.optimizer.state_dict(),
        learner_actor_state,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        learner.voc_optimizer.state_dict(),
        learner_q_state,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        resumed.optimizer.state_dict(),
        resumed_actor_state,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        resumed.voc_optimizer.state_dict(),
        resumed_q_state,
        rtol=0.0,
        atol=0.0,
    )


def test_learner_resumes_legacy_schema1_gate_as_canonical_beta09(
    tmp_path, monkeypatch
):
    flags = _learner_flags(tmp_path, "voc-gate-legacy-schema1-resume")
    flags.voc_gate_adam_beta1 = 0.9
    (tmp_path / flags.xpid).mkdir()
    actor, _train_out, _initial_actor_state = _rollout(flags)
    monkeypatch.setattr(
        "thinker.learn_actor.FileWriter", lambda **_kwargs: _NullWriter()
    )
    learner = SActorLearner(
        ray_obj=None,
        actor_param={},
        flags=flags,
        actor_net=actor,
        device=torch.device("cpu"),
    )
    learner.save_checkpoint()
    checkpoint = torch.load(
        learner.ckp_path, map_location="cpu", weights_only=False
    )
    checkpoint["voc_gate_policy_schema_version"] = 1
    del checkpoint["flags"]["voc_gate_adam_beta1"]
    torch.save(checkpoint, learner.ckp_path)

    resume_flags = copy.deepcopy(flags)
    resume_flags.ckp = True
    delattr(resume_flags, "voc_gate_adam_beta1")
    resumed = SActorLearner(
        ray_obj=None,
        actor_param={},
        flags=resume_flags,
        actor_net=copy.deepcopy(actor),
        device=torch.device("cpu"),
    )

    assert resumed.voc_gate_adam_beta1 == 0.9
    assert resumed.flags.voc_gate_adam_beta1 == 0.9
    assert resumed.voc_gate_optimizer.param_groups[0]["betas"] == (
        0.9,
        0.999,
    )
    for index, (original_parameter, resumed_parameter) in enumerate(
        zip(learner.voc_gate_parameters, resumed.voc_gate_parameters)
    ):
        gradient = torch.full_like(
            original_parameter, (index + 1) * 1e-3
        )
        original_parameter.grad = gradient.clone()
        resumed_parameter.grad = gradient.clone()
    learner._step_voc_gate_optimizer()
    resumed._step_voc_gate_optimizer()
    for original_parameter, resumed_parameter in zip(
        learner.voc_gate_parameters, resumed.voc_gate_parameters
    ):
        torch.testing.assert_close(
            original_parameter,
            resumed_parameter,
            rtol=0.0,
            atol=0.0,
        )
    torch.testing.assert_close(
        resumed.voc_gate_optimizer.state_dict(),
        learner.voc_gate_optimizer.state_dict(),
        rtol=0.0,
        atol=0.0,
    )


def test_unweighted_first_tie_logs_true_confidence_and_objective_weight(
    tmp_path, monkeypatch
):
    flags = _learner_flags(tmp_path, "voc-unweighted-first-tie-metrics")
    flags.voc_gate_confidence_weighted = False
    for name in ("voc_gate_param_align", "voc_gate_param_align_coef"):
        if hasattr(flags, name):
            delattr(flags, name)
    actor, train_out, initial_actor_state = _rollout(flags)
    monkeypatch.setattr(
        "thinker.learn_actor.FileWriter", lambda **_kwargs: _NullWriter()
    )
    learner = SActorLearner(
        ray_obj=None,
        actor_param={},
        flags=flags,
        actor_net=actor,
        device=torch.device("cpu"),
    )
    assert learner.voc_gate_param_align is False
    assert learner.flags.voc_gate_param_align is False
    assert learner.flags.voc_gate_param_align_coef == 1.0

    losses, _ = learner.compute_losses(train_out, initial_actor_state)
    expected_bce = torch.log(torch.tensor(2.0))
    for name in (
        "voc_gate_teacher_confidence",
        "voc_gate_bce_loss",
        "voc_gate_bce_unweighted",
        "voc_gate_directed_logit_gradient_mean",
        "voc_gate_directed_logit_gradient_abs",
        "voc_gate_directed_logit_gradient_rms",
    ):
        assert torch.isfinite(losses[name])
    torch.testing.assert_close(
        losses["voc_gate_teacher_confidence"], torch.tensor(0.0)
    )
    torch.testing.assert_close(losses["voc_gate_bce_loss"], expected_bce)
    torch.testing.assert_close(
        losses["voc_gate_bce_unweighted"], expected_bce
    )
    torch.testing.assert_close(
        losses["voc_gate_directed_logit_gradient_mean"], torch.tensor(0.0)
    )
    torch.testing.assert_close(
        losses["voc_gate_directed_logit_gradient_abs"], torch.tensor(0.0)
    )
    torch.testing.assert_close(
        losses["voc_gate_directed_logit_gradient_rms"], torch.tensor(0.0)
    )
    assert not any(
        name.startswith("voc_gate_param_") for name in losses
    )
    assert not any(
        "behavior_continue_probability" in name for name in losses
    )
    assert not any("exact_projection" in name for name in losses)


def test_parameter_alignment_first_tie_is_zero_gate_only_and_observable(
    tmp_path, monkeypatch
):
    flags = _learner_flags(tmp_path, "voc-param-align-first-tie")
    flags.voc_gate_confidence_weighted = False
    flags.voc_gate_param_align = True
    flags.voc_gate_param_align_coef = 1.0
    actor, train_out, initial_actor_state = _rollout(flags)
    monkeypatch.setattr(
        "thinker.learn_actor.FileWriter", lambda **_kwargs: _NullWriter()
    )
    learner = SActorLearner(
        ray_obj=None,
        actor_param={},
        flags=flags,
        actor_net=actor,
        device=torch.device("cpu"),
    )

    losses, _ = learner.compute_losses(train_out, initial_actor_state)
    expected_bce = torch.log(torch.tensor(2.0))
    torch.testing.assert_close(losses["voc_gate_bce_loss"], expected_bce)
    torch.testing.assert_close(
        losses["voc_gate_param_align_loss"], torch.tensor(0.0)
    )
    torch.testing.assert_close(
        losses["voc_gate_objective_loss"], expected_bce
    )
    torch.testing.assert_close(
        losses["_voc_gate_total_loss"], expected_bce
    )
    assert losses["voc_gate_param_align_applied"].item() == 1.0
    assert losses["voc_gate_param_align_train_support"].item() > 0.0
    assert losses["voc_gate_param_align_coef"].item() == 1.0
    assert losses["voc_gate_param_target_norm"].item() == 0.0
    assert losses["voc_gate_param_error_norm"].item() == 0.0
    assert losses["voc_gate_param_relative_error"].item() == 0.0
    assert losses["voc_gate_param_relative_error_defined"].item() == 0.0
    assert losses["voc_gate_param_cosine"].item() == 0.0
    assert losses["voc_gate_param_cosine_defined"].item() == 0.0

    losses["_voc_gate_total_loss"].backward()
    gate_parameter_ids = {
        id(parameter) for parameter in learner.voc_gate_parameters
    }
    for parameter in learner.voc_gate_parameters:
        torch.testing.assert_close(
            parameter.grad, torch.zeros_like(parameter)
        )
    for parameter in learner.actor_net.parameters():
        if id(parameter) not in gate_parameter_ids:
            assert parameter.grad is None


def test_off_mode_ignores_active_only_v11_defaults(tmp_path, monkeypatch):
    flags = _learner_flags(tmp_path, "voc-off-defaults", mode="off")
    for name in (
        "voc_gate_param_align",
        "voc_gate_param_align_coef",
        "voc_gate_exact_projection",
    ):
        if hasattr(flags, name):
            delattr(flags, name)
    # The shared defaults intentionally advertise the active protocol, while
    # an off-mode ActorNet does not instantiate either VoC head.
    assert flags.voc_dedicated_gate is True
    assert flags.voc_soft_q_bce_gate is True
    actor, _train_out, _initial_actor_state = _rollout(flags)
    assert not hasattr(actor, "voc_gate_head")
    monkeypatch.setattr(
        "thinker.learn_actor.FileWriter", lambda **_kwargs: _NullWriter()
    )

    learner = SActorLearner(
        ray_obj=None,
        actor_param={},
        flags=flags,
        actor_net=actor,
        device=torch.device("cpu"),
    )

    assert learner.dynamic_voc_mode == "off"
    assert learner.voc_dedicated_gate is False
    assert learner.voc_soft_q_bce_gate is False
    assert learner.voc_gate_parameters == []
    assert learner.voc_gate_optimizer is None
    assert learner.flags.voc_dedicated_gate is True
    assert learner.flags.voc_soft_q_bce_gate is True
    assert not hasattr(learner.flags, "voc_gate_param_align")
    assert not hasattr(learner.flags, "voc_gate_param_align_coef")
    assert not hasattr(learner.flags, "voc_gate_exact_projection")


def test_active_learner_resume_calls_common_gate_bundle_validator_before_load(
    tmp_path, monkeypatch
):
    flags = _learner_flags(tmp_path, "voc-gate-resume-corruption")
    (tmp_path / flags.xpid).mkdir()
    actor, _train_out, _initial_actor_state = _rollout(flags)
    monkeypatch.setattr(
        "thinker.learn_actor.FileWriter", lambda **_kwargs: _NullWriter()
    )
    learner = SActorLearner(
        ray_obj=None,
        actor_param={},
        flags=flags,
        actor_net=actor,
        device=torch.device("cpu"),
    )
    learner.save_checkpoint()
    checkpoint = torch.load(
        learner.ckp_path, map_location="cpu", weights_only=False
    )
    util.validate_voc_gate_policy_checkpoint(
        checkpoint, label="valid zero-update learner checkpoint"
    )

    corruptions = []
    corrupt = copy.deepcopy(checkpoint)
    corrupt["voc_gate_policy_schema_version"] = -1
    corruptions.append(corrupt)
    corrupt = copy.deepcopy(checkpoint)
    corrupt["actor_net_state_dict"]["voc_gate_head.bias"][0] = float("nan")
    corruptions.append(corrupt)
    corrupt = copy.deepcopy(checkpoint)
    corrupt["voc_gate_optimizer_state_dict"] = None
    corruptions.append(corrupt)
    corrupt = copy.deepcopy(checkpoint)
    corrupt["voc_gate_scheduler_state_dict"] = None
    corruptions.append(corrupt)
    corrupt = copy.deepcopy(checkpoint)
    corrupt["voc_gate_update_count"] = -1
    corruptions.append(corrupt)
    corrupt = copy.deepcopy(checkpoint)
    corrupt["voc_gate_grad_scaler_state_dict"] = {"scale": 128.0}
    corruptions.append(corrupt)
    corrupt = copy.deepcopy(checkpoint)
    corrupt["flags"]["voc_gate_adam_beta1"] = 0.0
    corrupt["voc_gate_optimizer_state_dict"]["param_groups"][0][
        "betas"
    ] = (0.0, 0.999)
    corruptions.append(corrupt)

    def forbid_optimizer_load(*_args, **_kwargs):
        pytest.fail("invalid gate bundle reached optimizer state loading")

    monkeypatch.setattr(
        "thinker.learn_actor.util.load_optimizer", forbid_optimizer_load
    )

    for corrupt in corruptions:
        monkeypatch.setattr(
            "thinker.learn_actor.torch.load",
            lambda *_args, _checkpoint=corrupt, **_kwargs: _checkpoint,
        )
        with pytest.raises(ValueError):
            learner.load_checkpoint("unused-corrupt-checkpoint")


def test_dedicated_gate_bce_is_sole_gradient_route_including_ppo_kl(
    tmp_path, monkeypatch
):
    flags = _learner_flags(
        tmp_path, "voc-gate-gradient-isolation", ppo_k=2
    )
    flags.ppo_kl_coef = 0.7
    actor, train_out, initial_actor_state = _rollout(flags)
    monkeypatch.setattr(
        "thinker.learn_actor.FileWriter", lambda **_kwargs: _NullWriter()
    )
    learner = SActorLearner(
        ray_obj=None,
        actor_param={},
        flags=flags,
        actor_net=actor,
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        learner.voc_ema_gate_weight.zero_()
        learner.voc_ema_gate_bias.copy_(torch.tensor([0.25, -0.25]))

    main_parameters = [
        parameter
        for group in learner.optimizer.param_groups
        for parameter in group["params"]
    ]
    critic_parameters = list(learner.voc_parameters)
    gate_parameters = list(learner.voc_gate_parameters)
    main_ids = {id(parameter) for parameter in main_parameters}
    critic_ids = {id(parameter) for parameter in critic_parameters}
    gate_ids = {id(parameter) for parameter in gate_parameters}
    assert main_ids.isdisjoint(critic_ids)
    assert main_ids.isdisjoint(gate_ids)
    assert critic_ids.isdisjoint(gate_ids)

    losses, _ = learner.compute_losses(train_out, initial_actor_state)
    gate_loss = losses["_voc_gate_total_loss"]
    assert gate_loss.item() > 0.0
    assert losses["kl_loss"].item() >= 0.0

    for parameter in learner.actor_net.parameters():
        parameter.grad = None
    losses["total_loss"].backward(retain_graph=True)
    # Refactorizing normalized joint probabilities can leave only roundoff
    # cancellation (~1e-7) in the raw graph.  consume_data_single clears this
    # entire actor-scaled buffer before the independently-scaled BCE backward.
    assert max(
        0.0 if parameter.grad is None else parameter.grad.abs().max().item()
        for parameter in gate_parameters
    ) < 1e-6
    assert any(parameter.grad is not None for parameter in main_parameters)

    for parameter in learner.actor_net.parameters():
        parameter.grad = None
    gate_loss.backward()
    assert any(
        parameter.grad is not None
        and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in gate_parameters
    )
    assert all(parameter.grad is None for parameter in main_parameters)
    assert all(parameter.grad is None for parameter in critic_parameters)


def test_dedicated_gate_amp_skip_reports_finite_zero_postclip_norm():
    class SkippingScaler:
        def __init__(self):
            self.scale = 128.0

        def get_scale(self):
            return self.scale

        def step(self, _optimizer):
            pass

        def update(self):
            self.scale = 64.0

    parameter = nn.Parameter(torch.tensor([1.0]))
    parameter.grad = torch.tensor([float("inf")])
    learner = object.__new__(SActorLearner)
    learner.voc_gate_parameters = [parameter]
    learner.voc_gate_optimizer = torch.optim.SGD([parameter], lr=0.1)
    learner.voc_gate_grad_norm_clipping = 1.0
    learner.actor_net = nn.Linear(1, 1)
    learner.flags = type(
        "Flags", (), {"float16": True, "actor_amp_max_consecutive_skips": 8}
    )()
    learner.voc_gate_scaler = SkippingScaler()
    learner.voc_gate_amp_skip_count = 0
    learner.voc_gate_amp_consecutive_skips = 0
    learner._logger = type("Logger", (), {"warning": lambda *_args: None})()
    learner._last_voc_gate_postclip_total_norm = float("nan")

    result = learner._step_voc_gate_optimizer()

    assert result.optimizer_stepped is False
    assert result.total_norm == 0.0
    assert learner._last_voc_gate_postclip_total_norm == 0.0
    assert torch.isfinite(
        torch.tensor(learner._last_voc_gate_postclip_total_norm)
    )
    assert learner.voc_gate_amp_skip_count == 1
    assert learner.voc_gate_amp_consecutive_skips == 1
    torch.testing.assert_close(parameter.detach(), torch.tensor([1.0]))


def _select_actor_stream(train_out, initial_actor_state, index):
    values = {}
    for field in type(train_out)._fields:
        value = getattr(train_out, field)
        values[field] = (
            None if value is None else value[:, index:index + 1]
        )
    selected_state = tuple(
        value[index:index + 1] for value in initial_actor_state
    )
    return type(train_out)(**values), selected_state


def test_heldout_only_control_batch_never_steps_dedicated_gate(
    tmp_path, monkeypatch
):
    class CapturingWriter:
        def __init__(self):
            self.records = []

        def log(self, values, *_args, **_kwargs):
            self.records.append(dict(values))

        def close(self, *_args, **_kwargs):
            pass

    flags = _learner_flags(tmp_path, "voc-control-heldout-gate")
    flags.voc_gate_param_align = True
    flags.voc_gate_param_align_coef = 1.0
    actor, train_out, initial_actor_state = _rollout(flags)
    writer = CapturingWriter()
    monkeypatch.setattr(
        "thinker.learn_actor.FileWriter", lambda **_kwargs: writer
    )
    learner = SActorLearner(
        ray_obj=None,
        actor_param={},
        flags=flags,
        actor_net=actor,
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        learner.voc_ema_gate_weight.zero_()
        learner.voc_ema_gate_bias.copy_(torch.tensor([0.25, -0.25]))
    gate_before = [
        parameter.detach().clone()
        for parameter in learner.voc_gate_parameters
    ]

    # Actor id 0 is permanently reserved by the deterministic holdout split.
    heldout_data = _select_actor_stream(
        train_out, initial_actor_state, 0
    )
    expected_holdout = int(
        heldout_data[0].control_valid[1:].sum().item()
    )
    assert expected_holdout > 0
    learner.consume_data_single(heldout_data)

    assert learner.voc_update_count == 0
    assert learner.voc_ema_gate_update_count == 0
    assert learner.voc_gate_update_count == 0
    assert learner.voc_holdout_count == expected_holdout
    assert len(writer.records) == 1
    record = writer.records[0]
    assert record["actor/voc_gate_param_align_applied"] == 0.0
    assert record["actor/voc_gate_param_align_train_support"] == 0.0
    assert record["actor/voc_gate_param_align_loss"] == 0.0
    assert record["actor/voc_gate_objective_loss"] == 0.0
    assert all(
        parameter.grad is None for parameter in learner.voc_gate_parameters
    )
    for before, after in zip(gate_before, learner.voc_gate_parameters):
        torch.testing.assert_close(after.detach(), before, rtol=0.0, atol=0.0)


def test_finite_batch_start_ema_can_step_gate_when_online_q_step_skips(
    tmp_path, monkeypatch
):
    flags = _learner_flags(tmp_path, "voc-control-q-skip-gate")
    flags.voc_gate_param_align = True
    flags.voc_gate_param_align_coef = 1.0
    actor, train_out, initial_actor_state = _rollout(flags)
    monkeypatch.setattr(
        "thinker.learn_actor.FileWriter", lambda **_kwargs: _NullWriter()
    )
    learner = SActorLearner(
        ray_obj=None,
        actor_param={},
        flags=flags,
        actor_net=actor,
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        learner.voc_ema_gate_weight.zero_()
        learner.voc_ema_gate_bias.copy_(torch.tensor([0.25, -0.25]))
    gate_head = learner.voc_gate_head_modules[0]
    alignment_before = compute_dynamic_voc_gate_parameter_alignment_loss(
        gate_weight=gate_head.weight,
        gate_bias=gate_head.bias,
        ema_q_weight=learner.voc_ema_gate_weight,
        ema_q_bias=learner.voc_ema_gate_bias,
        q_temperature=learner.voc_gate_q_temperature,
        policy_temperature=float(learner.flags.voc_gate_temperature),
    ).loss.detach().clone()
    gate_before = [
        parameter.detach().clone()
        for parameter in learner.voc_gate_parameters
    ]
    actor_parameters = [
        parameter
        for group in learner.optimizer.param_groups
        for parameter in group["params"]
    ]
    q_parameters = [
        parameter
        for group in learner.voc_optimizer.param_groups
        for parameter in group["params"]
    ]
    actor_before = [parameter.detach().clone() for parameter in actor_parameters]
    q_before = [parameter.detach().clone() for parameter in q_parameters]
    skipped = ActorGradientStepResult(
        total_norm=0.0,
        optimizer_stepped=False,
        amp_scale_before=128.0,
        amp_scale_after=64.0,
        nonfinite_gradient_names=(),
    )
    monkeypatch.setattr(
        learner, "_step_voc_optimizer", lambda _t, _b: skipped
    )
    monkeypatch.setattr(
        learner,
        "_step_actor_optimizer",
        lambda _parameters, _t, _b: skipped,
    )

    learner.consume_data_single((train_out, initial_actor_state))

    assert learner.voc_update_count == 0
    assert learner.voc_ema_gate_update_count == 0
    assert learner.voc_gate_update_count == 1
    assert learner._last_voc_gate_gradient_step.optimizer_stepped
    alignment_after = compute_dynamic_voc_gate_parameter_alignment_loss(
        gate_weight=gate_head.weight,
        gate_bias=gate_head.bias,
        ema_q_weight=learner.voc_ema_gate_weight,
        ema_q_bias=learner.voc_ema_gate_bias,
        q_temperature=learner.voc_gate_q_temperature,
        policy_temperature=float(learner.flags.voc_gate_temperature),
    ).loss.detach()
    assert alignment_after.item() < alignment_before.item()
    assert any(
        not torch.equal(before, after.detach())
        for before, after in zip(gate_before, learner.voc_gate_parameters)
    )
    for before, after in zip(actor_before, actor_parameters):
        torch.testing.assert_close(after.detach(), before, rtol=0.0, atol=0.0)
    for before, after in zip(q_before, q_parameters):
        torch.testing.assert_close(after.detach(), before, rtol=0.0, atol=0.0)

def test_shadow_learner_separates_main_and_voc_optimizer_gradients(
    tmp_path, monkeypatch
):
    torch.manual_seed(91)
    flags = cenv_flags(cap=4)
    flags.dynamic_factorized_control = True
    flags.dynamic_voc_mode = "shadow"
    flags.float16 = False
    flags.see_real_state = False
    flags.ppo_k = 1
    flags.return_norm_type = -1
    flags.parallel_actor = False
    flags.actor_batch_size = 3
    flags.env_n = 3
    flags.self_play_n = 1
    flags.total_steps = 100
    flags.ckp = False
    flags.savedir = str(tmp_path)
    flags.xpid = "voc-shadow-isolation"
    flags.ckpdir = str(tmp_path / flags.xpid)

    actor, train_out, initial_actor_state = _rollout(flags)
    monkeypatch.setattr(
        "thinker.learn_actor.FileWriter", lambda **_kwargs: _NullWriter()
    )
    learner = SActorLearner(
        ray_obj=None,
        actor_param={},
        flags=flags,
        actor_net=actor,
        device=torch.device("cpu"),
    )
    main_parameters = learner.optimizer.param_groups[0]["params"]
    assert {id(parameter) for parameter in main_parameters}.isdisjoint(
        {id(parameter) for parameter in learner.voc_parameters}
    )

    losses, _ = learner.compute_losses(train_out, initial_actor_state)
    voc_loss = losses["_voc_total_loss"]
    learner.optimizer.zero_grad()
    learner.voc_optimizer.zero_grad()
    losses["total_loss"].backward()
    assert all(parameter.grad is None for parameter in learner.voc_parameters)
    assert any(parameter.grad is not None for parameter in main_parameters)

    learner.optimizer.zero_grad()
    learner.voc_optimizer.zero_grad()
    voc_loss.backward()
    assert any(parameter.grad is not None for parameter in learner.voc_parameters)
    assert all(parameter.grad is None for parameter in main_parameters)
    assert losses["voc_gate_pg_loss"].item() == 0.0


def test_learner_wires_dueling_base_exact_gate_and_reconstructed_metrics(
    tmp_path, monkeypatch
):
    flags = cenv_flags(cap=4)
    flags.dynamic_factorized_control = True
    flags.dynamic_voc_mode = "control"
    flags.voc_dueling_q = True
    flags.voc_expected_gate_loss = True
    flags.think_cost = 0.0005
    flags.think_cost_anneal = False
    flags.voc_parent_checkpoint = ""
    flags.preload_actor = ""
    flags.float16 = False
    flags.see_real_state = False
    flags.ppo_k = 1
    flags.return_norm_type = -1
    flags.parallel_actor = False
    flags.actor_batch_size = 3
    flags.env_n = 3
    flags.self_play_n = 1
    flags.total_steps = 100
    flags.ckp = False
    flags.checkpoint_interval = 0
    flags.savedir = str(tmp_path)
    flags.xpid = "voc-v2-wiring"
    flags.ckpdir = str(tmp_path / flags.xpid)

    actor, train_out, initial_actor_state = _rollout(flags)
    monkeypatch.setattr(
        "thinker.learn_actor.FileWriter", lambda **_kwargs: _NullWriter()
    )
    captured = {}
    original_loss = compute_dynamic_voc_loss

    def capture_loss(**kwargs):
        result = original_loss(**kwargs)
        captured["kwargs"] = kwargs
        captured["result"] = result
        return result

    monkeypatch.setattr(
        "thinker.learn_actor.compute_dynamic_voc_loss", capture_loss
    )
    learner = SActorLearner(
        ray_obj=None,
        actor_param={},
        flags=flags,
        actor_net=actor,
        device=torch.device("cpu"),
    )
    losses, _ = learner.compute_losses(train_out, initial_actor_state)

    assert learner.voc_dueling_q is True
    assert learner.voc_expected_gate_loss is True
    assert captured["kwargs"]["dueling_q"] is True
    assert captured["kwargs"]["expected_gate_loss"] is True
    state_value = captured["kwargs"]["voc_state_value"].detach()
    result = captured["result"]
    valid = result.valid
    logits = captured["kwargs"]["target_control_logits"]
    probabilities = torch.softmax(logits.detach(), dim=-1)
    gate_probabilities = torch.stack(
        (
            probabilities[..., util.PROCEED]
            + probabilities[..., util.RESET],
            probabilities[..., util.STOP],
        ),
        dim=-1,
    )
    reconstructed_mean = torch.sum(
        gate_probabilities * result.q_values.detach(), dim=-1
    )
    torch.testing.assert_close(
        reconstructed_mean[valid], state_value[valid]
    )
    torch.testing.assert_close(
        losses["voc_q_continue"],
        result.q_values[..., 0].detach()[valid].float().mean(),
    )
    torch.testing.assert_close(
        losses["voc_q_stop"],
        result.q_values[..., 1].detach()[valid].float().mean(),
    )


def test_fresh_control_starts_zero_neutral_trains_and_resumes(
    tmp_path, monkeypatch
):
    """Exercise the no-parent control path through a real learner update."""

    flags = cenv_flags(cap=4)
    flags.dynamic_factorized_control = True
    flags.dynamic_voc_mode = "control"
    flags.think_cost = 0.0005
    flags.think_cost_anneal = False
    flags.voc_parent_checkpoint = ""
    flags.preload_actor = ""
    flags.float16 = False
    flags.see_real_state = False
    flags.ppo_k = 1
    flags.return_norm_type = -1
    flags.parallel_actor = False
    flags.actor_batch_size = 3
    flags.env_n = 3
    flags.self_play_n = 1
    flags.total_steps = 100
    flags.ckp = False
    flags.checkpoint_interval = 0
    flags.savedir = str(tmp_path)
    flags.xpid = "voc-fresh-control"
    flags.ckpdir = str(tmp_path / flags.xpid)
    (tmp_path / flags.xpid).mkdir()

    actor, train_out, initial_actor_state = _rollout(flags)
    monkeypatch.setattr(
        "thinker.learn_actor.FileWriter", lambda **_kwargs: _NullWriter()
    )
    learner = SActorLearner(
        ray_obj=None,
        actor_param={},
        flags=flags,
        actor_net=actor,
        device=torch.device("cpu"),
    )

    assert learner.voc_control_origin == util.VOC_CONTROL_ORIGIN_FRESH
    assert learner.voc_activation_real_step == 0
    assert learner.voc_parent_checkpoint is None
    assert learner.voc_parent_checkpoint_sha256 is None
    assert learner.voc_parent_imitation_data_signature is None
    assert all(
        torch.count_nonzero(parameter).item() == 0
        for parameter in learner.voc_parameters
    )
    assert all(
        torch.count_nonzero(parameter).item() == 0
        for parameter in learner.voc_gate_parameters
    )
    main_parameters = [
        parameter
        for group in learner.optimizer.param_groups
        for parameter in group["params"]
    ]
    assert {id(parameter) for parameter in main_parameters}.isdisjoint(
        {id(parameter) for parameter in learner.voc_parameters}
    )
    assert {id(parameter) for parameter in main_parameters}.isdisjoint(
        {id(parameter) for parameter in learner.voc_gate_parameters}
    )
    assert {id(parameter) for parameter in learner.voc_parameters}.isdisjoint(
        {id(parameter) for parameter in learner.voc_gate_parameters}
    )

    losses, _ = learner.compute_losses(train_out, initial_actor_state)
    # Equal zero Q makes the initial directed VoC gate update neutral.  The
    # stochastic gate/epsilon still supplies actual STOP/CONTINUE samples.
    torch.testing.assert_close(losses["voc_gate_pg_loss"], torch.tensor(0.0))
    assert losses["voc_gate_greedy_agreement"].item() == 0.0
    assert losses["voc_gate_greedy_agreement_count"].item() == 0.0
    assert losses["voc_gate_greedy_agreement_defined"].item() == 0.0
    assert losses["voc_gate_online_delta_sign_agreement"].item() == 0.0
    assert losses["voc_gate_online_delta_sign_agreement_count"].item() == 0.0
    assert losses["voc_gate_online_delta_sign_agreement_defined"].item() == 0.0
    assert losses["voc_gate_holdout_greedy_agreement_count"].item() == 0.0
    assert (
        losses["voc_gate_holdout_online_delta_sign_agreement_count"].item()
        == 0.0
    )
    assert losses["voc_continue_support"].item() > 0
    assert losses["voc_stop_support"].item() > 0
    shifted_valid = train_out.control_valid[1:].bool()
    shifted_control = train_out.search_control[1:]
    decision_depth = train_out.search_steps[1:].long() - (
        shifted_valid & (shifted_control != util.STOP)
    ).long()
    predecision_control = train_out.last_search_control[:-1]
    expected_post_compute = (
        shifted_valid
        & (decision_depth > 0)
        & (
            (predecision_control == util.PROCEED)
            | (predecision_control == util.RESET)
        )
    )
    assert losses["voc_post_compute_count"].item() == int(
        expected_post_compute.sum().item()
    )
    assert losses["voc_post_proceed_count"].item() == int((
        expected_post_compute & (predecision_control == util.PROCEED)
    ).sum().item())
    assert losses["voc_post_reset_count"].item() == int((
        expected_post_compute & (predecision_control == util.RESET)
    ).sum().item())
    learner.optimizer.zero_grad()
    learner.voc_optimizer.zero_grad()
    losses["total_loss"].backward()
    assert all(parameter.grad is None for parameter in learner.voc_parameters)
    assert any(parameter.grad is not None for parameter in main_parameters)
    learner.optimizer.zero_grad()
    learner.voc_optimizer.zero_grad()
    losses["_voc_total_loss"].backward()
    assert any(parameter.grad is not None for parameter in learner.voc_parameters)
    assert all(parameter.grad is None for parameter in main_parameters)

    # compute_losses deliberately leaves transactional support counters
    # pending for its caller.  Use a clean learner to exercise the complete
    # optimizer/commit/checkpoint path rather than mutating that transaction.
    learner = SActorLearner(
        ray_obj=None,
        actor_param={},
        flags=flags,
        actor_net=actor,
        device=torch.device("cpu"),
    )
    learner.consume_data_single((train_out, initial_actor_state))
    assert learner.voc_update_count == 1
    assert learner.voc_ema_gate_update_count == 1
    assert learner.voc_ema_gate_parent_update_count == 0
    assert learner.voc_gate_update_count == 1
    assert learner.voc_optimizer.state
    assert learner.voc_gate_optimizer.state
    assert any(
        torch.count_nonzero(parameter).item() > 0
        for parameter in learner.voc_parameters
    )

    learner.save_checkpoint()
    checkpoint = torch.load(
        learner.ckp_path, map_location="cpu", weights_only=False
    )
    assert checkpoint["voc_control_origin"] == "fresh"
    assert checkpoint["voc_control_origin_legacy_defaulted"] is False
    assert checkpoint["voc_parent_checkpoint_sha256"] is None
    assert checkpoint["voc_parent_checkpoint"] is None
    assert checkpoint["voc_parent_imitation_data_signature"] is None
    assert checkpoint["voc_activation_real_step"] == 0
    assert checkpoint["voc_ema_gate_schema_version"] == 1
    assert checkpoint["voc_ema_gate_update_count"] == 1
    assert checkpoint["voc_ema_gate_parent_update_count"] == 0
    assert set(checkpoint["voc_ema_gate_head_state_dict"]) == {
        "weight", "bias"
    }
    assert (
        checkpoint["voc_gate_policy_schema_version"]
        == util.VOC_GATE_POLICY_SCHEMA_VERSION
    )
    assert checkpoint["voc_gate_update_count"] == 1
    assert checkpoint["voc_gate_amp_skip_count"] == 0
    assert checkpoint["voc_gate_amp_consecutive_skips"] == 0
    assert checkpoint["voc_gate_optimizer_state_dict"]["state"]
    assert checkpoint["voc_gate_scheduler_state_dict"] is not None
    assert checkpoint["voc_gate_grad_scaler_state_dict"] is None
    util.validate_voc_gate_policy_checkpoint(
        checkpoint, label="fresh-control learner checkpoint"
    )

    resume_flags = copy.deepcopy(flags)
    resume_flags.ckp = True
    resumed = SActorLearner(
        ray_obj=None,
        actor_param={},
        flags=resume_flags,
        actor_net=copy.deepcopy(actor),
        device=torch.device("cpu"),
    )
    assert resumed.voc_control_origin == "fresh"
    assert resumed.voc_control_origin_legacy_defaulted is False
    assert resumed.voc_parent_checkpoint_sha256 is None
    assert resumed.voc_activation_real_step == 0
    assert resumed.voc_update_count == learner.voc_update_count
    assert resumed.voc_ema_gate_update_count == 1
    assert resumed.voc_ema_gate_parent_update_count == 0
    assert resumed.voc_gate_update_count == learner.voc_gate_update_count
    assert resumed.voc_gate_amp_skip_count == 0
    assert resumed.voc_gate_amp_consecutive_skips == 0
    torch.testing.assert_close(
        resumed.voc_ema_gate_weight, learner.voc_ema_gate_weight
    )
    assert resumed.voc_scheduler.state_dict() == learner.voc_scheduler.state_dict()
    assert (
        resumed.voc_gate_scheduler.state_dict()
        == learner.voc_gate_scheduler.state_dict()
    )
    for resumed_parameter, parameter in zip(
        resumed.voc_gate_parameters, learner.voc_gate_parameters
    ):
        torch.testing.assert_close(resumed_parameter, parameter)
    resumed.save_checkpoint()
    roundtrip = torch.load(
        resumed.ckp_path, map_location="cpu", weights_only=False
    )
    provenance = util.validate_voc_control_checkpoint_provenance(roundtrip)
    assert provenance["voc_control_origin"] == "fresh"
    assert all(
        roundtrip["flags"][name] == ""
        for name in ("preload", "preload_actor", "voc_parent_checkpoint")
    )


def test_fresh_control_rejects_nonzero_initial_q_head(tmp_path, monkeypatch):
    flags = cenv_flags(cap=4)
    flags.dynamic_factorized_control = True
    flags.dynamic_voc_mode = "control"
    flags.think_cost = 0.0005
    flags.voc_parent_checkpoint = ""
    flags.preload_actor = ""
    flags.parallel_actor = False
    flags.ckp = False
    flags.savedir = str(tmp_path)
    flags.xpid = "voc-fresh-control-nonzero"
    flags.ckpdir = str(tmp_path / flags.xpid)

    actor, _train_out, _initial_actor_state = _rollout(flags)
    with torch.no_grad():
        actor.voc_head.bias[0] = 0.1
    monkeypatch.setattr(
        "thinker.learn_actor.FileWriter", lambda **_kwargs: _NullWriter()
    )

    with pytest.raises(RuntimeError, match="equal zero-initialized"):
        SActorLearner(
            ray_obj=None,
            actor_param={},
            flags=flags,
            actor_net=actor,
            device=torch.device("cpu"),
        )


def test_fresh_control_second_actual_update_uses_learned_q_and_re_evaluates(
    tmp_path, monkeypatch
):
    """The first real update learns Q; the second has directed gate credit."""

    class CapturingWriter:
        def __init__(self):
            self.records = []

        def log(self, values, *_args, **_kwargs):
            self.records.append(dict(values))

        def close(self, *_args, **_kwargs):
            pass

    flags = cenv_flags(cap=4)
    flags.dynamic_factorized_control = True
    flags.dynamic_voc_mode = "control"
    flags.think_cost = 0.0005
    flags.think_cost_anneal = False
    flags.voc_parent_checkpoint = ""
    flags.preload_actor = ""
    flags.float16 = False
    flags.see_real_state = False
    flags.ppo_k = 1
    flags.return_norm_type = -1
    flags.parallel_actor = False
    flags.actor_batch_size = 3
    flags.env_n = 3
    flags.self_play_n = 1
    flags.total_steps = 100
    flags.ckp = False
    flags.checkpoint_interval = 0
    flags.savedir = str(tmp_path)
    flags.xpid = "voc-fresh-two-updates"
    flags.ckpdir = str(tmp_path / flags.xpid)

    actor, train_out, initial_actor_state = _rollout(flags)
    writer = CapturingWriter()
    monkeypatch.setattr(
        "thinker.learn_actor.FileWriter", lambda **_kwargs: writer
    )
    learner = SActorLearner(
        ray_obj=None,
        actor_param={},
        flags=flags,
        actor_net=actor,
        device=torch.device("cpu"),
    )

    learner.consume_data_single((train_out, initial_actor_state))
    learner.consume_data_single((train_out, initial_actor_state))

    assert learner.voc_control_origin == util.VOC_CONTROL_ORIGIN_FRESH
    assert learner.voc_update_count == 2
    assert len(writer.records) == 2
    first, second = writer.records
    assert first["actor/voc_gate_pg_loss"] == pytest.approx(0.0, abs=1e-12)
    assert abs(second["actor/voc_gate_pg_loss"]) > 0.0
    assert first["actor/voc_gate_target_source_update_count"] == 0
    assert first["actor/voc_ema_gate_update_count"] == 1
    assert second["actor/voc_gate_target_source_update_count"] == 1
    assert second["actor/voc_ema_gate_update_count"] == 2
    assert "actor/voc_gate_delta_q_positive_count" in second
    assert (
        second["actor/voc_delta_q_positive_count"]
        + second["actor/voc_delta_q_negative_count"]
    ) > 0
    assert second["actor/voc_post_compute_count"] > 0
    assert (
        second["actor/voc_post_proceed_count"]
        + second["actor/voc_post_reset_count"]
        == second["actor/voc_post_compute_count"]
    )
    for key in (
        "actor/voc_acceptance_count",
        "actor/voc_gate_acceptance_count",
        "actor/voc_acceptance_depth_8_plus_count",
        "actor/voc_gate_acceptance_depth_8_plus_count",
        "actor/voc_acceptance_teacher_continue_probability_delta_positive",
        "actor/voc_gate_acceptance_teacher_continue_probability_delta_positive",
        "actor/voc_post_useful_compute_prior_useful_count",
        "actor/voc_gate_post_useful_compute_prior_useful_count",
        "actor/voc_post_useful_compute_transition_coverage_rate",
        "actor/voc_gate_post_useful_compute_transition_coverage_rate",
    ):
        assert key in second
        assert torch.isfinite(torch.tensor(second[key])).item()


def test_successful_shadow_step_commits_voc_support_counters(
    tmp_path, monkeypatch
):
    flags = cenv_flags(cap=4)
    flags.dynamic_factorized_control = True
    flags.dynamic_voc_mode = "shadow"
    flags.voc_gate_param_align = True
    flags.voc_gate_param_align_coef = 1.0
    flags.float16 = False
    flags.see_real_state = False
    flags.ppo_k = 1
    flags.return_norm_type = -1
    flags.parallel_actor = False
    flags.actor_batch_size = 3
    flags.env_n = 3
    flags.self_play_n = 1
    flags.total_steps = 100
    flags.ckp = False
    flags.checkpoint_interval = 0
    flags.savedir = str(tmp_path)
    flags.xpid = "voc-shadow-step"
    flags.ckpdir = str(tmp_path / flags.xpid)
    (tmp_path / flags.xpid).mkdir()
    actor, train_out, initial_actor_state = _rollout(flags)
    monkeypatch.setattr(
        "thinker.learn_actor.FileWriter", lambda **_kwargs: _NullWriter()
    )
    learner = SActorLearner(
        ray_obj=None,
        actor_param={},
        flags=flags,
        actor_net=actor,
        device=torch.device("cpu"),
    )

    learner.consume_data_single((train_out, initial_actor_state))

    valid = train_out.control_valid[1:]
    holdout = dynamic_voc_holdout_mask(train_out.id, valid)
    expected_support = int((valid & ~holdout).sum().item())
    expected_holdout = int(holdout.sum().item())
    assert learner.voc_update_count == 1
    assert learner.voc_ema_gate_update_count == 1
    assert learner.voc_gate_param_align is False
    assert learner.flags.voc_gate_param_align is True
    assert learner.flags.voc_gate_param_align_coef == 1.0
    assert all(
        torch.count_nonzero(parameter).item() == 0
        for parameter in learner.voc_gate_parameters
    )
    assert learner.voc_continue_count + learner.voc_stop_count == expected_support
    assert learner.voc_holdout_count == expected_holdout
    assert (
        learner.voc_holdout_continue_count + learner.voc_holdout_stop_count
        == expected_holdout
    )
    assert learner._last_voc_gradient_step.optimizer_stepped
    learner.save_checkpoint()
    checkpoint = torch.load(
        learner.ckp_path, map_location="cpu", weights_only=False
    )
    assert checkpoint["dynamic_voc_mode"] == "shadow"
    assert checkpoint["voc_update_count"] == 1
    assert checkpoint["voc_ema_gate_update_count"] == 1
    assert checkpoint["voc_ema_gate_parent_update_count"] == 0
    assert checkpoint["voc_continue_count"] == learner.voc_continue_count
    assert checkpoint["voc_stop_count"] == learner.voc_stop_count
    assert checkpoint["voc_holdout_count"] == learner.voc_holdout_count
    assert checkpoint["voc_holdout_td_mae"] >= 0.0
    assert checkpoint["voc_holdout_td_rmse"] >= 0.0
    assert checkpoint["voc_parent_checkpoint_sha256"] is None
    assert checkpoint["voc_activation_real_step"] == -1
    assert checkpoint["voc_optimizer_state_dict"] is not None
    assert checkpoint["voc_scheduler_state_dict"] is not None
    assert (
        checkpoint["voc_gate_policy_schema_version"]
        == util.VOC_GATE_POLICY_SCHEMA_VERSION
    )
    assert checkpoint["voc_gate_update_count"] == 0
    assert checkpoint["flags"]["voc_gate_param_align"] is True
    assert checkpoint["flags"]["voc_gate_param_align_coef"] == 1.0
    assert checkpoint["voc_gate_optimizer_state_dict"] is not None
    assert checkpoint["voc_gate_optimizer_state_dict"]["state"] == {}
    assert checkpoint["voc_gate_scheduler_state_dict"] is not None
    assert checkpoint["voc_gate_grad_scaler_state_dict"] is None
    util.validate_voc_gate_policy_checkpoint(
        checkpoint, label="shadow learner checkpoint"
    )

    resume_flags = copy.deepcopy(flags)
    resume_flags.ckp = True
    resumed = SActorLearner(
        ray_obj=None,
        actor_param={},
        flags=resume_flags,
        actor_net=copy.deepcopy(actor),
        device=torch.device("cpu"),
    )
    assert resumed.voc_update_count == learner.voc_update_count
    assert resumed.voc_ema_gate_update_count == 1
    assert resumed.voc_gate_update_count == 0
    assert resumed.voc_continue_count == learner.voc_continue_count
    assert resumed.voc_stop_count == learner.voc_stop_count
    assert resumed.voc_holdout_count == learner.voc_holdout_count
    assert resumed.voc_holdout_td_sum == learner.voc_holdout_td_sum
    assert resumed.voc_holdout_td_abs_sum == learner.voc_holdout_td_abs_sum
    assert resumed.voc_holdout_td_sq_sum == learner.voc_holdout_td_sq_sum
    assert resumed.voc_scheduler.state_dict() == learner.voc_scheduler.state_dict()


def test_all_heldout_minibatch_commits_calibration_without_q_training(
    tmp_path, monkeypatch
):
    flags = cenv_flags(cap=4)
    flags.dynamic_factorized_control = True
    flags.dynamic_voc_mode = "shadow"
    flags.float16 = False
    flags.see_real_state = False
    flags.ppo_k = 1
    flags.return_norm_type = -1
    flags.parallel_actor = False
    flags.actor_batch_size = 3
    flags.env_n = 3
    flags.self_play_n = 1
    flags.total_steps = 100
    flags.ckp = False
    flags.checkpoint_interval = 0
    flags.savedir = str(tmp_path)
    flags.xpid = "voc-shadow-heldout-only"
    flags.ckpdir = str(tmp_path / flags.xpid)
    (tmp_path / flags.xpid).mkdir()
    actor, train_out, initial_actor_state = _rollout(flags)
    monkeypatch.setattr(
        "thinker.learn_actor.FileWriter", lambda **_kwargs: _NullWriter()
    )
    learner = SActorLearner(
        ray_obj=None,
        actor_param={},
        flags=flags,
        actor_net=actor,
        device=torch.device("cpu"),
    )

    def select_actor(index):
        values = {}
        for field in type(train_out)._fields:
            value = getattr(train_out, field)
            values[field] = None if value is None else value[:, index:index + 1]
        selected_state = tuple(
            value[index:index + 1] for value in initial_actor_state
        )
        return type(train_out)(**values), selected_state

    heldout_data = select_actor(0)  # actor id 0 is permanently reserved.
    expected_holdout = int(
        heldout_data[0].control_valid[1:].sum().item()
    )
    assert expected_holdout > 0
    learner.consume_data_single(heldout_data)
    assert learner.voc_update_count == 0
    assert learner.voc_ema_gate_update_count == 0
    assert learner.voc_continue_count == 0
    assert learner.voc_stop_count == 0
    assert learner.voc_holdout_count == expected_holdout

    training_data = select_actor(1)
    learner.consume_data_single(training_data)
    assert learner.voc_update_count == 1
    assert learner.voc_ema_gate_update_count == 1
    assert learner.voc_continue_count + learner.voc_stop_count > 0
    assert learner.voc_holdout_count == expected_holdout

    learner.save_checkpoint()
    resume_flags = copy.deepcopy(flags)
    resume_flags.ckp = True
    resumed = SActorLearner(
        ray_obj=None,
        actor_param={},
        flags=resume_flags,
        actor_net=copy.deepcopy(actor),
        device=torch.device("cpu"),
    )
    assert resumed.voc_holdout_count == expected_holdout
    assert resumed.voc_update_count == 1


def test_ema_target_follows_q_step_success_independently_of_actor_step(
    tmp_path, monkeypatch
):
    flags = cenv_flags(cap=4)
    flags.dynamic_factorized_control = True
    flags.dynamic_voc_mode = "shadow"
    flags.float16 = False
    flags.see_real_state = False
    flags.ppo_k = 1
    flags.return_norm_type = -1
    flags.parallel_actor = False
    flags.actor_batch_size = 3
    flags.env_n = 3
    flags.self_play_n = 1
    flags.total_steps = 100
    flags.ckp = False
    flags.checkpoint_interval = 0
    flags.savedir = str(tmp_path)
    flags.xpid = "voc-step-transaction"
    flags.ckpdir = str(tmp_path / flags.xpid)
    (tmp_path / flags.xpid).mkdir()
    actor, train_out, initial_actor_state = _rollout(flags)
    monkeypatch.setattr(
        "thinker.learn_actor.FileWriter", lambda **_kwargs: _NullWriter()
    )
    skipped = ActorGradientStepResult(
        total_norm=0.0,
        optimizer_stepped=False,
        amp_scale_before=128.0,
        amp_scale_after=64.0,
        nonfinite_gradient_names=(),
    )

    q_skipped = SActorLearner(
        ray_obj=None,
        actor_param={},
        flags=flags,
        actor_net=copy.deepcopy(actor),
        device=torch.device("cpu"),
    )
    monkeypatch.setattr(
        q_skipped, "_step_voc_optimizer", lambda _t, _b: skipped
    )
    q_skipped.consume_data_single((train_out, initial_actor_state))
    assert q_skipped.voc_update_count == 0
    assert q_skipped.voc_ema_gate_update_count == 0

    actor_skipped = SActorLearner(
        ray_obj=None,
        actor_param={},
        flags=flags,
        actor_net=copy.deepcopy(actor),
        device=torch.device("cpu"),
    )
    monkeypatch.setattr(
        actor_skipped,
        "_step_actor_optimizer",
        lambda _parameters, _t, _b: skipped,
    )
    actor_skipped.consume_data_single((train_out, initial_actor_state))
    assert actor_skipped.voc_update_count == 1
    assert actor_skipped.voc_ema_gate_update_count == 1


@pytest.mark.parametrize("think_cost", [-0.1, float("nan"), float("inf")])
def test_voc_target_rejects_invalid_computation_cost(think_cost):
    tensor = torch.zeros((1, 1))
    with pytest.raises(ValueError, match="think_cost"):
        compute_dynamic_voc_target(
            task_rewards=tensor,
            think_rewards=tensor,
            task_discounts=tensor,
            think_discounts=tensor,
            task_vs=tensor,
            think_vs=tensor,
            task_bootstrap_value=torch.zeros(1),
            think_bootstrap_value=torch.zeros(1),
            think_cost=think_cost,
        )


def _exact_projection_learner_flags(tmp_path, xpid):
    flags = _learner_flags(tmp_path, xpid)
    flags.voc_gate_param_align = False
    flags.voc_gate_param_align_coef = 1.0
    flags.voc_gate_exact_projection = True
    flags.voc_gate_confidence_weighted = False
    return flags


def _epsilon_execution_learner_flags(tmp_path, xpid):
    flags = _exact_projection_learner_flags(tmp_path, xpid)
    flags.voc_gate_epsilon_greedy_execution = True
    return flags


def _schema8_learner_flags(tmp_path, xpid):
    """Build the strict schema-8 identity around the existing v12 learner."""

    flags = _epsilon_execution_learner_flags(tmp_path, xpid)
    flags.voc_actor_policy_version_barrier = True
    flags.voc_actor_policy_bundle_schema_version = (
        util.VOC_ACTOR_POLICY_BUNDLE_SCHEMA_VERSION
    )
    flags.voc_actor_policy_barrier_timeout_s = (
        util.VOC_ACTOR_POLICY_BARRIER_TIMEOUT_SECONDS
    )
    flags.voc_actor_policy_ray_max_restarts = 0
    flags.voc_actor_policy_ray_max_task_retries = 0
    flags.voc_gate_policy_schema_version = (
        util.VOC_GATE_POLICY_HALF_SQUARED_Q_SCHEMA_VERSION
    )
    flags.voc_model_input_seal_schema_version = 1
    flags.voc_gate_execution_epsilon = 0.25
    flags.voc_train_epsilon = 0.02
    flags.actor_amp_init_scale = 32.0
    # Unit/integration learners are invoked directly without Ray actors.  The
    # persisted schema identity stays exact while runtime publication is
    # intentionally inactive, matching the existing private offline path.
    flags.train_actor = False
    flags.parallel_actor = False
    return flags


def _schema9_learner_flags(tmp_path, xpid):
    """Build schema 9 by changing only schema 8's reconstruction identity."""

    flags = _schema8_learner_flags(tmp_path, xpid)
    flags.voc_gate_policy_schema_version = (
        util.VOC_GATE_POLICY_COMMON_MODE_Q_SCHEMA_VERSION
    )
    return flags


def _schema10_learner_flags(tmp_path, xpid):
    """Build schema 10 by changing only schema 9's regression identity."""

    flags = _schema9_learner_flags(tmp_path, xpid)
    flags.voc_gate_policy_schema_version = (
        util.VOC_GATE_POLICY_HUBER_COMMON_Q_SCHEMA_VERSION
    )
    return flags


def _schema11_learner_flags(tmp_path, xpid):
    """Build schema 11 by changing only schema 10's optimizer coordinates."""

    flags = _schema10_learner_flags(tmp_path, xpid)
    flags.voc_gate_policy_schema_version = (
        util.VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION
    )
    return flags


def _schema12_learner_flags(tmp_path, xpid):
    """Build schema 12 by changing only schema 11's effective EMA tau."""

    flags = _schema11_learner_flags(tmp_path, xpid)
    flags.voc_gate_policy_schema_version = (
        util.VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION
    )
    flags.voc_gate_target_tau = 1.0
    return flags


@pytest.mark.parametrize(
    "tau",
    [
        pytest.param("missing", id="missing"),
        pytest.param(1, id="integer"),
        pytest.param(True, id="boolean"),
        pytest.param(np.float64(1.0), id="numpy-float"),
        pytest.param(np.int64(1), id="numpy-integer"),
        pytest.param("1.0", id="string"),
        pytest.param(0.0, id="zero"),
        pytest.param(-0.0, id="negative-zero"),
        pytest.param(0.1, id="schema11-value"),
        pytest.param(0.9999999999999999, id="subunit"),
        pytest.param(1.0000000000000002, id="above-one"),
        pytest.param(float("inf"), id="infinity"),
        pytest.param(float("nan"), id="nan"),
    ],
)
def test_schema12_learner_rejects_nonexact_tau_before_actor_construction(
    tmp_path, tau
):
    flags = _schema12_learner_flags(tmp_path, "voc-schema12-invalid-tau")
    if tau == "missing":
        delattr(flags, "voc_gate_target_tau")
    else:
        flags.voc_gate_target_tau = tau

    with pytest.raises(ValueError, match="exact built-in float"):
        SActorLearner(
            ray_obj=None,
            actor_param={},
            flags=flags,
            actor_net=None,
            device=None,
        )


@pytest.mark.parametrize("schema", [1, 2, 3, 4, 5])
@pytest.mark.parametrize(
    ("tau", "expected"),
    [
        pytest.param("missing", 0.1, id="missing-default"),
        pytest.param(0.25, 0.25, id="builtin-float"),
        pytest.param(1, 1.0, id="builtin-integer-upper-bound"),
        pytest.param(np.float64(0.75), 0.75, id="numpy-interior"),
    ],
)
def test_schema1_through_schema5_keep_historical_tau_normalization(
    tmp_path, monkeypatch, schema, tau, expected
):
    flags = _learner_flags(
        tmp_path, f"voc-schema{schema}-historical-tau-{expected}"
    )
    flags.voc_gate_policy_schema_version = schema
    if tau == "missing":
        delattr(flags, "voc_gate_target_tau")
    else:
        flags.voc_gate_target_tau = tau
    actor, _, _ = _rollout(flags)
    monkeypatch.setattr(
        "thinker.learn_actor.FileWriter", lambda **_kwargs: _NullWriter()
    )

    learner = SActorLearner(
        ray_obj=None,
        actor_param={},
        flags=flags,
        actor_net=actor,
        device=torch.device("cpu"),
    )

    assert type(learner.voc_gate_target_tau) is float
    assert learner.voc_gate_target_tau == expected


def _versioned_q_learner_flags(tmp_path, xpid, schema):
    if schema == util.VOC_GATE_POLICY_HALF_SQUARED_Q_SCHEMA_VERSION:
        return _schema8_learner_flags(tmp_path, xpid)
    if schema == util.VOC_GATE_POLICY_COMMON_MODE_Q_SCHEMA_VERSION:
        return _schema9_learner_flags(tmp_path, xpid)
    if schema == util.VOC_GATE_POLICY_HUBER_COMMON_Q_SCHEMA_VERSION:
        return _schema10_learner_flags(tmp_path, xpid)
    if schema == util.VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION:
        return _schema11_learner_flags(tmp_path, xpid)
    if schema == util.VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION:
        return _schema12_learner_flags(tmp_path, xpid)
    raise AssertionError(f"unexpected versioned Q schema {schema}")


def test_schema8_through_schema10_fresh_state_and_optimizer_are_identical(
    tmp_path, monkeypatch
):
    flags8 = _schema8_learner_flags(tmp_path, "voc-schema8-state-parity")
    flags9 = _schema9_learner_flags(tmp_path, "voc-schema9-state-parity")
    flags10 = _schema10_learner_flags(
        tmp_path, "voc-schema10-state-parity"
    )
    torch.manual_seed(9163)
    actor8, _, _ = _rollout(flags8)
    torch.manual_seed(9163)
    actor9, _, _ = _rollout(flags9)
    torch.manual_seed(9163)
    actor10, _, _ = _rollout(flags10)
    monkeypatch.setattr(
        "thinker.learn_actor.FileWriter", lambda **_kwargs: _NullWriter()
    )
    learner8 = SActorLearner(
        ray_obj=None,
        actor_param={},
        flags=flags8,
        actor_net=actor8,
        device=torch.device("cpu"),
    )
    learner9 = SActorLearner(
        ray_obj=None,
        actor_param={},
        flags=flags9,
        actor_net=actor9,
        device=torch.device("cpu"),
    )
    learner10 = SActorLearner(
        ray_obj=None,
        actor_param={},
        flags=flags10,
        actor_net=actor10,
        device=torch.device("cpu"),
    )

    assert (
        actor8.state_dict().keys()
        == actor9.state_dict().keys()
        == actor10.state_dict().keys()
    )
    for key, value in actor8.state_dict().items():
        assert torch.equal(value, actor9.state_dict()[key]), key
        assert torch.equal(value, actor10.state_dict()[key]), key
    assert (
        dict(actor8.named_parameters()).keys()
        == dict(actor9.named_parameters()).keys()
        == dict(actor10.named_parameters()).keys()
    )
    assert (
        dict(actor8.named_buffers()).keys()
        == dict(actor9.named_buffers()).keys()
        == dict(actor10.named_buffers()).keys()
    )
    assert torch.equal(
        learner8.voc_ema_gate_weight, learner9.voc_ema_gate_weight
    )
    assert torch.equal(
        learner8.voc_ema_gate_weight, learner10.voc_ema_gate_weight
    )
    assert torch.equal(
        learner8.voc_ema_gate_bias, learner9.voc_ema_gate_bias
    )
    assert torch.equal(
        learner8.voc_ema_gate_bias, learner10.voc_ema_gate_bias
    )
    for candidate in (learner9, learner10):
        assert learner8.voc_optimizer.state_dict() == (
            candidate.voc_optimizer.state_dict()
        )
        assert learner8.voc_scheduler.state_dict() == (
            candidate.voc_scheduler.state_dict()
        )
        assert learner8.voc_gate_optimizer.state_dict() == (
            candidate.voc_gate_optimizer.state_dict()
        )
        assert learner8.voc_gate_scheduler.state_dict() == (
            candidate.voc_gate_scheduler.state_dict()
        )
        assert candidate.voc_scaler is None
        assert candidate.voc_gate_scaler is None
    assert learner8.voc_scaler is None
    assert learner8.voc_gate_scaler is None


class _CaptureWriter:
    def __init__(self):
        self.records = []

    def log(self, stats):
        self.records.append(dict(stats))

    def close(self, *_args, **_kwargs):
        pass


def _select_rollout_actor_stream(train_out, initial_actor_state, index):
    values = {}
    for field in type(train_out)._fields:
        value = getattr(train_out, field)
        values[field] = None if value is None else value[:, index:index + 1]
    selected_state = tuple(
        value[index:index + 1] for value in initial_actor_state
    )
    return type(train_out)(**values), selected_state


def test_exact_projection_first_tie_is_diagnostic_and_gradient_isolated(
    tmp_path, monkeypatch
):
    flags = _exact_projection_learner_flags(tmp_path, "voc-exact-first-tie")
    actor, train_out, initial_actor_state = _rollout(flags)
    monkeypatch.setattr(
        "thinker.learn_actor.FileWriter", lambda **_kwargs: _NullWriter()
    )
    learner = SActorLearner(
        ray_obj=None,
        actor_param={},
        flags=flags,
        actor_net=actor,
        device=torch.device("cpu"),
    )

    assert learner.voc_gate_exact_projection is True
    assert learner.voc_gate_param_align is False
    assert all(
        not parameter.requires_grad and parameter.grad is None
        for parameter in learner.voc_gate_parameters
    )
    assert learner.voc_gate_optimizer.state == {}
    losses, _ = learner.compute_losses(train_out, initial_actor_state)
    expected_bce = torch.log(torch.tensor(2.0))
    torch.testing.assert_close(losses["voc_gate_bce_loss"], expected_bce)
    torch.testing.assert_close(
        losses["voc_gate_objective_loss"], expected_bce
    )
    assert "_voc_gate_total_loss" not in losses
    assert losses["voc_gate_exact_projection_enabled"].item() == 1.0
    assert losses["voc_gate_projection_batch_start_error_norm"].item() == 0.0
    for key in (
        "voc_acceptance_behavior_continue_probability_delta_positive",
        "voc_acceptance_depth_8_plus_behavior_continue_probability_delta_negative",
        "voc_post_useful_compute_behavior_continue_probability_delta_negative",
        "voc_gate_acceptance_behavior_continue_probability_delta_positive",
        "voc_gate_acceptance_depth_8_plus_behavior_continue_probability_delta_negative",
        "voc_gate_post_useful_compute_behavior_continue_probability_delta_negative",
    ):
        assert key in losses
        assert torch.isfinite(losses[key])

    learner.optimizer.zero_grad()
    learner.voc_optimizer.zero_grad()
    losses["total_loss"].backward()
    assert all(parameter.grad is None for parameter in learner.voc_parameters)
    assert all(
        parameter.grad is None for parameter in learner.voc_gate_parameters
    )

    learner.optimizer.zero_grad()
    learner.voc_optimizer.zero_grad()
    losses["_voc_total_loss"].backward()
    assert any(parameter.grad is not None for parameter in learner.voc_parameters)
    actor_parameter_ids = {
        id(parameter)
        for group in learner.optimizer.param_groups
        for parameter in group["params"]
    }
    assert all(
        parameter.grad is None
        for parameter in learner.actor_net.parameters()
        if id(parameter) in actor_parameter_ids
    )
    assert all(
        parameter.grad is None for parameter in learner.voc_gate_parameters
    )


def test_exact_projection_success_save_resume_and_boundaries(
    tmp_path, monkeypatch
):
    flags = _exact_projection_learner_flags(tmp_path, "voc-exact-success")
    (tmp_path / flags.xpid).mkdir()
    actor, train_out, initial_actor_state = _rollout(flags)
    monkeypatch.setattr(
        "thinker.learn_actor.FileWriter", lambda **_kwargs: _NullWriter()
    )
    learner = SActorLearner(
        ray_obj=None,
        actor_param={},
        flags=flags,
        actor_net=actor,
        device=torch.device("cpu"),
    )
    writer = _CaptureWriter()
    learner.plogger = writer
    gate_scheduler_before = copy.deepcopy(
        learner.voc_gate_scheduler.state_dict()
    )

    learner.consume_data((train_out, initial_actor_state))

    assert learner.voc_update_count == 1
    assert learner.voc_ema_gate_update_count == 1
    assert learner.voc_gate_update_count == 1
    assert learner.voc_ema_gate_parent_update_count == 0
    assert learner._last_voc_gate_exact_projection_applied is True
    assert learner._last_voc_gate_projection_post_error_norm == 0.0
    assert learner.voc_gate_optimizer.state == {}
    assert learner.voc_gate_scheduler.state_dict() == gate_scheduler_before
    assert all(
        not parameter.requires_grad and parameter.grad is None
        for parameter in learner.voc_gate_parameters
    )
    assert learner.voc_optimizer.state
    assert any(
        torch.count_nonzero(parameter).item() > 0
        for parameter in learner.voc_parameters
    )
    assert (
        torch.count_nonzero(learner.voc_ema_gate_weight).item()
        + torch.count_nonzero(learner.voc_ema_gate_bias).item()
    ) > 0
    gate_head = learner.voc_gate_head_modules[0]
    alignment = compute_dynamic_voc_gate_parameter_alignment_loss(
        gate_weight=gate_head.weight,
        gate_bias=gate_head.bias,
        ema_q_weight=learner.voc_ema_gate_weight,
        ema_q_bias=learner.voc_ema_gate_bias,
        q_temperature=learner.voc_gate_q_temperature,
        policy_temperature=float(learner.flags.voc_gate_temperature),
    )
    assert torch.equal(gate_head.weight, alignment.target_weight)
    assert torch.equal(gate_head.bias, alignment.target_bias)
    assert alignment.parameter_error_norm.item() == 0.0

    assert len(writer.records) == 1
    record = writer.records[0]
    assert record["actor/voc_gate_exact_projection_enabled"] == 1.0
    assert record["actor/voc_gate_exact_projection_applied"] == 1.0
    assert record["actor/voc_gate_optimizer_stepped"] == 0.0
    assert record["actor/voc_gate_update_count"] == 1.0
    assert record["actor/voc_gate_projection_post_error_norm"] == 0.0
    assert record["actor/voc_gate_bce_loss"] == pytest.approx(
        math.log(2.0)
    )
    for key in (
        "actor/voc_acceptance_behavior_continue_probability_delta_positive",
        "actor/voc_gate_acceptance_behavior_continue_probability_delta_positive",
        "actor/voc_post_useful_compute_behavior_continue_probability_delta_negative",
    ):
        assert key in record
        assert math.isfinite(record[key])

    learner._actor_weights_for_publication()
    learner.save_checkpoint()
    checkpoint = torch.load(
        learner.ckp_path, map_location="cpu", weights_only=False
    )
    assert checkpoint["voc_gate_policy_schema_version"] == 4
    assert checkpoint["flags"]["voc_gate_exact_projection"] is True
    assert checkpoint["flags"]["voc_gate_param_align"] is False
    assert checkpoint["flags"]["voc_gate_param_align_coef"] == 1.0
    assert checkpoint["voc_gate_optimizer_state_dict"]["state"] == {}
    assert checkpoint["voc_gate_scheduler_state_dict"] == gate_scheduler_before
    assert checkpoint["voc_gate_grad_scaler_state_dict"] is None
    util.validate_voc_gate_policy_checkpoint(
        checkpoint, label="exact-projection learner checkpoint"
    )

    resume_flags = copy.deepcopy(flags)
    resume_flags.ckp = True
    resumed = SActorLearner(
        ray_obj=None,
        actor_param={},
        flags=resume_flags,
        actor_net=copy.deepcopy(actor),
        device=torch.device("cpu"),
    )
    resumed._assert_voc_gate_exact_projection_invariant()
    resumed._actor_weights_for_publication()
    assert resumed.voc_gate_update_count == resumed.voc_update_count == 1
    assert resumed.voc_ema_gate_update_count == 1
    assert resumed.voc_gate_optimizer.state == {}
    assert all(
        not parameter.requires_grad and parameter.grad is None
        for parameter in resumed.voc_gate_parameters
    )

    resumed_gate = resumed.voc_gate_head_modules[0]
    saved_weight = resumed_gate.weight.detach().clone()
    with torch.no_grad():
        resumed_gate.weight[0, 0] = torch.nextafter(
            resumed_gate.weight[0, 0], torch.tensor(float("inf"))
        )
    with pytest.raises(RuntimeError, match="disagrees with EMA Q target"):
        resumed._actor_weights_for_publication()
    with pytest.raises(RuntimeError, match="disagrees with EMA Q target"):
        resumed.save_checkpoint()
    with torch.no_grad():
        resumed_gate.weight.copy_(saved_weight)
    resumed._assert_voc_gate_exact_projection_invariant()


def test_exact_projection_q_skip_leaves_ema_gate_and_metrics_unchanged(
    tmp_path, monkeypatch
):
    flags = _exact_projection_learner_flags(tmp_path, "voc-exact-q-skip")
    actor, train_out, initial_actor_state = _rollout(flags)
    monkeypatch.setattr(
        "thinker.learn_actor.FileWriter", lambda **_kwargs: _NullWriter()
    )
    learner = SActorLearner(
        ray_obj=None,
        actor_param={},
        flags=flags,
        actor_net=actor,
        device=torch.device("cpu"),
    )
    writer = _CaptureWriter()
    learner.plogger = writer
    gate_before = [
        parameter.detach().clone()
        for parameter in learner.voc_gate_parameters
    ]
    ema_weight_before = learner.voc_ema_gate_weight.clone()
    ema_bias_before = learner.voc_ema_gate_bias.clone()
    skipped = ActorGradientStepResult(
        total_norm=0.0,
        optimizer_stepped=False,
        amp_scale_before=128.0,
        amp_scale_after=64.0,
        nonfinite_gradient_names=(),
    )
    monkeypatch.setattr(
        learner, "_step_voc_optimizer", lambda _t, _b: skipped
    )
    monkeypatch.setattr(
        learner,
        "_step_actor_optimizer",
        lambda _parameters, _t, _b: skipped,
    )

    learner.consume_data((train_out, initial_actor_state))

    assert learner.voc_update_count == 0
    assert learner.voc_ema_gate_update_count == 0
    assert learner.voc_gate_update_count == 0
    assert learner._last_voc_gate_exact_projection_applied is False
    assert learner.voc_gate_optimizer.state == {}
    assert torch.equal(learner.voc_ema_gate_weight, ema_weight_before)
    assert torch.equal(learner.voc_ema_gate_bias, ema_bias_before)
    for before, after in zip(gate_before, learner.voc_gate_parameters):
        assert torch.equal(before, after)
    record = writer.records[0]
    assert record["actor/voc_gate_exact_projection_applied"] == 0.0
    assert record["actor/voc_gate_optimizer_stepped"] == 0.0
    assert record["actor/voc_gate_update_count"] == 0.0


def test_schema5_execution_success_save_resume_and_q_skip_lifecycle(
    tmp_path, monkeypatch
):
    flags = _epsilon_execution_learner_flags(
        tmp_path, "voc-epsilon-execution-success"
    )
    (tmp_path / flags.xpid).mkdir()
    actor, train_out, initial_actor_state = _rollout(flags)
    monkeypatch.setattr(
        "thinker.learn_actor.FileWriter", lambda **_kwargs: _NullWriter()
    )
    learner = SActorLearner(
        ray_obj=None,
        actor_param={},
        flags=flags,
        actor_net=actor,
        device=torch.device("cpu"),
    )

    learner.consume_data((train_out, initial_actor_state))

    assert learner.voc_gate_epsilon_greedy_execution is True
    assert learner.voc_update_count == 1
    assert learner.voc_ema_gate_update_count == 1
    assert learner.voc_gate_update_count == 1
    assert learner.voc_gate_optimizer.state == {}
    learner.save_checkpoint()
    checkpoint = torch.load(
        learner.ckp_path, map_location="cpu", weights_only=False
    )
    assert checkpoint["voc_gate_policy_schema_version"] == 5
    assert checkpoint["flags"]["voc_gate_epsilon_greedy_execution"] is True
    assert checkpoint["flags"]["voc_gate_exact_projection"] is True
    state = util.validate_voc_gate_policy_checkpoint(checkpoint)
    assert state["voc_gate_epsilon_greedy_execution"] is True

    resume_flags = copy.deepcopy(flags)
    resume_flags.ckp = True
    resumed = SActorLearner(
        ray_obj=None,
        actor_param={},
        flags=resume_flags,
        actor_net=copy.deepcopy(actor),
        device=torch.device("cpu"),
    )
    assert resumed.voc_gate_epsilon_greedy_execution is True
    assert resumed.voc_gate_update_count == resumed.voc_update_count == 1
    resumed._assert_voc_gate_exact_projection_invariant()

    skip_flags = _epsilon_execution_learner_flags(
        tmp_path, "voc-epsilon-execution-q-skip"
    )
    skip_actor, skip_train_out, skip_initial_state = _rollout(skip_flags)
    skipped_learner = SActorLearner(
        ray_obj=None,
        actor_param={},
        flags=skip_flags,
        actor_net=skip_actor,
        device=torch.device("cpu"),
    )
    skipped = ActorGradientStepResult(
        total_norm=0.0,
        optimizer_stepped=False,
        amp_scale_before=128.0,
        amp_scale_after=64.0,
        nonfinite_gradient_names=(),
    )
    monkeypatch.setattr(
        skipped_learner, "_step_voc_optimizer", lambda _t, _b: skipped
    )
    skipped_learner.consume_data((skip_train_out, skip_initial_state))
    assert skipped_learner.voc_update_count == 0
    assert skipped_learner.voc_ema_gate_update_count == 0
    assert skipped_learner.voc_gate_update_count == 0
    assert skipped_learner.voc_gate_optimizer.state == {}


def test_schema5_compute_losses_routes_soft_surface_to_online_and_ema_q(
    tmp_path, monkeypatch
):
    flags = _epsilon_execution_learner_flags(
        tmp_path, "voc-epsilon-execution-soft-q-routing"
    )
    actor, train_out, initial_actor_state = _rollout(flags)
    monkeypatch.setattr(
        "thinker.learn_actor.FileWriter", lambda **_kwargs: _NullWriter()
    )
    learner = SActorLearner(
        ray_obj=None,
        actor_param={},
        flags=flags,
        actor_net=actor,
        device=torch.device("cpu"),
    )
    # Leave the fresh exact tie before capturing: one successful update makes
    # the projected gate non-zero, so the .99/.01 execution surface is visibly
    # distinct from the preserved soft calibration surface.
    learner.consume_data((train_out, initial_actor_state))

    captured = {}
    original_forward = learner.actor_net.forward

    def capture_forward(*args, **kwargs):
        actor_out, actor_state = original_forward(*args, **kwargs)
        if kwargs.get("compute_loss", False):
            captured["forward_soft"] = actor_out.misc[
                "voc_gate_soft_control_logits"
            ][:-1].detach().clone()
            captured["forward_execution"] = actor_out.search_control_logits[
                :-1
            ].detach().clone()
        return actor_out, actor_state

    monkeypatch.setattr(learner.actor_net, "forward", capture_forward)
    original_voc_loss = learn_actor_module.compute_dynamic_voc_loss

    def capture_voc_loss(**kwargs):
        captured["online_soft"] = kwargs[
            "target_control_logits"
        ].detach().clone()
        captured["target_execution"] = kwargs[
            "target_behavior_control_logits"
        ].detach().clone()
        result = original_voc_loss(**kwargs)
        captured["q_values"] = result.q_values.detach().clone()
        captured["q_loss"] = result.q_loss.detach().clone()
        return result

    monkeypatch.setattr(
        learn_actor_module, "compute_dynamic_voc_loss", capture_voc_loss
    )
    original_ema_loss = learner._compute_ema_gate_loss

    def capture_ema_loss(**kwargs):
        captured["ema_soft"] = kwargs["logits"].detach().clone()
        result = original_ema_loss(**kwargs)
        captured["ema_q"] = result[1].detach().clone()
        return result

    monkeypatch.setattr(learner, "_compute_ema_gate_loss", capture_ema_loss)

    learner.compute_losses(train_out, initial_actor_state)

    assert torch.equal(captured["online_soft"], captured["forward_soft"])
    assert torch.equal(
        captured["target_execution"], captured["forward_execution"]
    )
    assert torch.equal(captured["online_soft"], captured["ema_soft"])
    valid = train_out.control_valid[1:].bool()
    assert torch.any(
        captured["online_soft"][valid]
        != captured["target_execution"][valid]
    )
    assert torch.isfinite(captured["q_values"]).all()
    assert torch.isfinite(captured["q_loss"])
    assert torch.isfinite(captured["ema_q"]).all()


def test_schema5_missing_soft_surface_fails_before_q_or_ema_loss(
    tmp_path, monkeypatch
):
    flags = _epsilon_execution_learner_flags(
        tmp_path, "voc-epsilon-execution-missing-soft"
    )
    actor, train_out, initial_actor_state = _rollout(flags)
    monkeypatch.setattr(
        "thinker.learn_actor.FileWriter", lambda **_kwargs: _NullWriter()
    )
    learner = SActorLearner(
        ray_obj=None,
        actor_param={},
        flags=flags,
        actor_net=actor,
        device=torch.device("cpu"),
    )
    original_forward = learner.actor_net.forward

    def corrupt_forward(*args, **kwargs):
        actor_out, actor_state = original_forward(*args, **kwargs)
        if kwargs.get("compute_loss", False):
            misc = dict(actor_out.misc)
            del misc["voc_gate_soft_control_logits"]
            actor_out = actor_out._replace(misc=misc)
        return actor_out, actor_state

    monkeypatch.setattr(learner.actor_net, "forward", corrupt_forward)
    calls = {"online": 0, "ema": 0}

    def unexpected_online_loss(**_kwargs):
        calls["online"] += 1
        raise AssertionError("online Q loss was constructed")

    def unexpected_ema_loss(**_kwargs):
        calls["ema"] += 1
        raise AssertionError("EMA Q loss was constructed")

    monkeypatch.setattr(
        learn_actor_module,
        "compute_dynamic_voc_loss",
        unexpected_online_loss,
    )
    monkeypatch.setattr(
        learner, "_compute_ema_gate_loss", unexpected_ema_loss
    )

    with pytest.raises(RuntimeError, match="separate soft gate"):
        learner.compute_losses(train_out, initial_actor_state)
    assert calls == {"online": 0, "ema": 0}


def test_exact_projection_q_success_survives_independent_actor_amp_skip(
    tmp_path, monkeypatch
):
    flags = _exact_projection_learner_flags(
        tmp_path, "voc-exact-actor-skip"
    )
    actor, train_out, initial_actor_state = _rollout(flags)
    monkeypatch.setattr(
        "thinker.learn_actor.FileWriter", lambda **_kwargs: _NullWriter()
    )
    learner = SActorLearner(
        ray_obj=None,
        actor_param={},
        flags=flags,
        actor_net=actor,
        device=torch.device("cpu"),
    )
    writer = _CaptureWriter()
    learner.plogger = writer
    skipped = ActorGradientStepResult(
        total_norm=0.0,
        optimizer_stepped=False,
        amp_scale_before=128.0,
        amp_scale_after=64.0,
        nonfinite_gradient_names=(),
    )
    monkeypatch.setattr(
        learner,
        "_step_actor_optimizer",
        lambda _parameters, _t, _b: skipped,
    )

    learner.consume_data((train_out, initial_actor_state))

    assert learner.voc_update_count == 1
    assert learner.voc_ema_gate_update_count == 1
    assert learner.voc_gate_update_count == 1
    assert learner._last_voc_gate_exact_projection_applied is True
    learner._assert_voc_gate_exact_projection_invariant()
    assert learner.voc_gate_optimizer.state == {}
    assert writer.records[0][
        "actor/voc_gate_exact_projection_applied"
    ] == 1.0
    assert writer.records[0]["actor/voc_gate_optimizer_stepped"] == 0.0


@pytest.mark.parametrize("schema", [8, 9, 10, 11, 12])
def test_schema8_through_schema12_route_q_and_update_ema_projection(
    tmp_path, monkeypatch, schema
):
    flags = _versioned_q_learner_flags(
        tmp_path, f"voc-schema{schema}-q-success", schema
    )
    actor, train_out, initial_actor_state = _rollout(flags)
    monkeypatch.setattr(
        "thinker.learn_actor.FileWriter", lambda **_kwargs: _NullWriter()
    )
    learner = SActorLearner(
        ray_obj=None,
        actor_param={},
        flags=flags,
        actor_net=actor,
        device=torch.device("cpu"),
    )
    writer = _CaptureWriter()
    learner.plogger = writer
    captured = {}
    original_voc_loss = learn_actor_module.compute_dynamic_voc_loss

    def capture_voc_loss(**kwargs):
        captured["gate_schema"] = kwargs["gate_policy_schema_version"]
        result = original_voc_loss(**kwargs)
        captured["q_loss"] = result.q_loss.detach().item()
        captured["q_support"] = int(result.q_train_valid.sum().item())
        return result

    monkeypatch.setattr(
        learn_actor_module, "compute_dynamic_voc_loss", capture_voc_loss
    )
    actor_skipped = ActorGradientStepResult(
        total_norm=0.0,
        optimizer_stepped=False,
        amp_scale_before=32.0,
        amp_scale_after=16.0,
        nonfinite_gradient_names=(),
    )
    monkeypatch.setattr(
        learner,
        "_step_actor_optimizer",
        lambda _parameters, _t, _b: actor_skipped,
    )

    learner.consume_data((train_out, initial_actor_state))

    assert captured["gate_schema"] == schema
    assert captured["q_support"] > 0
    assert learner.voc_update_count == 1
    assert learner.voc_ema_gate_update_count == 1
    assert learner.voc_gate_update_count == 1
    assert learner._last_voc_gate_exact_projection_applied is True
    learner._assert_voc_gate_exact_projection_invariant()
    if schema == util.VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION:
        assert torch.equal(
            learner.voc_ema_gate_weight,
            learner.voc_online_head.weight.detach(),
        )
        assert torch.equal(
            learner.voc_ema_gate_bias,
            learner.voc_online_head.bias.detach(),
        )
    record = writer.records[0]
    assert record["actor/voc_q_loss"] == pytest.approx(
        captured["q_loss"] / captured["q_support"]
    )
    assert record["actor/voc_optimizer_stepped"] == 1.0
    assert record["actor/voc_ema_gate_updated"] == 1.0
    assert record["actor/voc_gate_exact_projection_applied"] == 1.0
    assert record["actor/voc_gate_optimizer_stepped"] == 0.0


def test_schema12_nonzero_checkpoint_persists_tau_and_equal_raw_ema_only(
    tmp_path, monkeypatch
):
    xpid = "voc-schema12-nonzero-checkpoint-equality"
    flags = _schema12_learner_flags(tmp_path, xpid)
    (tmp_path / xpid).mkdir()
    actor, train_out, initial_actor_state = _rollout(flags)
    monkeypatch.setattr(
        "thinker.learn_actor.FileWriter", lambda **_kwargs: _NullWriter()
    )
    learner = SActorLearner(
        ray_obj=None,
        actor_param={},
        flags=flags,
        actor_net=actor,
        device=torch.device("cpu"),
    )
    learner.consume_data((train_out, initial_actor_state))
    assert learner.voc_ema_gate_update_count == 1

    learner.save_checkpoint()
    checkpoint = torch.load(
        learner.ckp_path, map_location="cpu", weights_only=False
    )

    assert checkpoint["voc_gate_policy_schema_version"] == 12
    assert type(checkpoint["voc_gate_target_tau"]) is float
    assert checkpoint["voc_gate_target_tau"] == 1.0
    assert type(checkpoint["flags"]["voc_gate_target_tau"]) is float
    assert checkpoint["flags"]["voc_gate_target_tau"] == 1.0
    assert checkpoint["voc_update_count"] == 1
    optimizer_state = checkpoint["voc_optimizer_state_dict"]
    assert set(optimizer_state) == {"state", "param_groups"}
    assert len(optimizer_state["state"]) == 2
    for state in optimizer_state["state"].values():
        assert set(state) == {"step", "exp_avg", "exp_avg_sq"}
        assert state["step"].item() == 1.0
        assert state["exp_avg"].shape[0] == 2
        assert state["exp_avg_sq"].shape == state["exp_avg"].shape
    assert torch.equal(
        checkpoint["voc_ema_gate_head_state_dict"]["weight"],
        checkpoint["actor_net_state_dict"]["voc_head.weight"],
    )
    assert torch.equal(
        checkpoint["voc_ema_gate_head_state_dict"]["bias"],
        checkpoint["actor_net_state_dict"]["voc_head.bias"],
    )
    reserved = {
        "voc_q_regression_loss",
        "voc_q_reconstruction",
        "voc_q_optimizer_coordinates",
    }
    pending = [checkpoint]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            assert reserved.isdisjoint(value)
            pending.extend(value.values())
        elif isinstance(value, (list, tuple)):
            pending.extend(value)
@pytest.mark.parametrize("schema", [8, 9, 10, 11, 12])
def test_schema8_through_schema12_q_skip_blocks_ema_projection(
    tmp_path, monkeypatch, schema
):
    flags = _versioned_q_learner_flags(
        tmp_path, f"voc-schema{schema}-q-skip", schema
    )
    actor, train_out, initial_actor_state = _rollout(flags)
    monkeypatch.setattr(
        "thinker.learn_actor.FileWriter", lambda **_kwargs: _NullWriter()
    )
    learner = SActorLearner(
        ray_obj=None,
        actor_param={},
        flags=flags,
        actor_net=actor,
        device=torch.device("cpu"),
    )
    writer = _CaptureWriter()
    learner.plogger = writer
    gate_before = [
        parameter.detach().clone()
        for parameter in learner.voc_gate_parameters
    ]
    ema_weight_before = learner.voc_ema_gate_weight.clone()
    ema_bias_before = learner.voc_ema_gate_bias.clone()
    q_skipped = ActorGradientStepResult(
        total_norm=0.0,
        optimizer_stepped=False,
        amp_scale_before=32.0,
        amp_scale_after=16.0,
        nonfinite_gradient_names=(),
    )
    monkeypatch.setattr(
        learner, "_step_voc_optimizer", lambda _t, _b: q_skipped
    )

    learner.consume_data((train_out, initial_actor_state))

    assert learner.voc_update_count == 0
    assert learner.voc_ema_gate_update_count == 0
    assert learner.voc_gate_update_count == 0
    assert learner._last_voc_gate_exact_projection_applied is False
    assert torch.equal(learner.voc_ema_gate_weight, ema_weight_before)
    assert torch.equal(learner.voc_ema_gate_bias, ema_bias_before)
    for before, after in zip(gate_before, learner.voc_gate_parameters):
        assert torch.equal(before, after)
    record = writer.records[0]
    assert record["actor/voc_optimizer_stepped"] == 0.0
    assert record["actor/voc_ema_gate_updated"] == 0.0
    assert record["actor/voc_gate_exact_projection_applied"] == 0.0


@pytest.mark.parametrize("schema", [8, 9])
def test_schema8_and_schema9_square_overflow_fails_before_update(
    tmp_path, monkeypatch, schema
):
    flags = _versioned_q_learner_flags(
        tmp_path, f"voc-schema{schema}-square-overflow", schema
    )
    actor, train_out, initial_actor_state = _rollout(flags)
    monkeypatch.setattr(
        "thinker.learn_actor.FileWriter", lambda **_kwargs: _NullWriter()
    )
    learner = SActorLearner(
        ray_obj=None,
        actor_param={},
        flags=flags,
        actor_net=actor,
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        learner.voc_online_head.weight.zero_()
        learner.voc_online_head.bias.copy_(
            torch.tensor([2.0e19, -2.0e19])
        )
    ema_weight_before = learner.voc_ema_gate_weight.clone()
    ema_bias_before = learner.voc_ema_gate_bias.clone()
    gate_before = [
        parameter.detach().clone()
        for parameter in learner.voc_gate_parameters
    ]
    calls = {"q_step": 0, "ema": 0, "projection": 0}

    def unexpected(name):
        def fail(*_args, **_kwargs):
            calls[name] += 1
            raise AssertionError(
                f"{name} ran after schema-{schema} overflow"
            )

        return fail

    monkeypatch.setattr(learner, "_step_voc_optimizer", unexpected("q_step"))
    monkeypatch.setattr(
        learner, "_update_voc_ema_gate_target", unexpected("ema")
    )
    monkeypatch.setattr(
        learner,
        "_project_voc_gate_head_to_ema_target",
        unexpected("projection"),
    )

    with pytest.raises(FloatingPointError, match="half-squared Q loss"):
        learner.consume_data((train_out, initial_actor_state))

    assert calls == {"q_step": 0, "ema": 0, "projection": 0}
    assert learner.voc_update_count == 0
    assert learner.voc_ema_gate_update_count == 0
    assert learner.voc_gate_update_count == 0
    assert torch.equal(learner.voc_ema_gate_weight, ema_weight_before)
    assert torch.equal(learner.voc_ema_gate_bias, ema_bias_before)
    for before, after in zip(gate_before, learner.voc_gate_parameters):
        assert torch.equal(before, after)


def test_schema10_nonfinite_q_fails_before_optimizer_ema_or_projection(
    tmp_path, monkeypatch
):
    schema = util.VOC_GATE_POLICY_HUBER_COMMON_Q_SCHEMA_VERSION
    flags = _versioned_q_learner_flags(
        tmp_path, "voc-schema10-nonfinite-q", schema
    )
    actor, train_out, initial_actor_state = _rollout(flags)
    monkeypatch.setattr(
        "thinker.learn_actor.FileWriter", lambda **_kwargs: _NullWriter()
    )
    learner = SActorLearner(
        ray_obj=None,
        actor_param={},
        flags=flags,
        actor_net=actor,
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        learner.voc_online_head.bias[0] = float("inf")
    ema_weight_before = learner.voc_ema_gate_weight.clone()
    ema_bias_before = learner.voc_ema_gate_bias.clone()
    gate_before = [
        parameter.detach().clone()
        for parameter in learner.voc_gate_parameters
    ]
    calls = {"q_step": 0, "ema": 0, "projection": 0}

    def unexpected(name):
        def fail(*_args, **_kwargs):
            calls[name] += 1
            raise AssertionError(f"{name} ran after schema-10 non-finite Q")

        return fail

    monkeypatch.setattr(learner, "_step_voc_optimizer", unexpected("q_step"))
    monkeypatch.setattr(
        learner, "_update_voc_ema_gate_target", unexpected("ema")
    )
    monkeypatch.setattr(
        learner,
        "_project_voc_gate_head_to_ema_target",
        unexpected("projection"),
    )

    with pytest.raises(FloatingPointError, match="VoC Q outputs"):
        learner.consume_data((train_out, initial_actor_state))

    assert calls == {"q_step": 0, "ema": 0, "projection": 0}
    assert learner.voc_update_count == 0
    assert learner.voc_ema_gate_update_count == 0
    assert learner.voc_gate_update_count == 0
    assert torch.equal(learner.voc_ema_gate_weight, ema_weight_before)
    assert torch.equal(learner.voc_ema_gate_bias, ema_bias_before)
    for before, after in zip(gate_before, learner.voc_gate_parameters):
        assert torch.equal(before, after)


@pytest.mark.parametrize("schema", [8, 9, 10])
def test_schema8_through_schema10_q_optimizer_retain_tb_scaled_clipping(schema):
    parameter = nn.Parameter(torch.tensor([1.0]))
    parameter.grad = torch.tensor([10.0])
    learner = object.__new__(SActorLearner)
    learner.voc_gate_policy_schema_version = schema
    learner.voc_parameters = [parameter]
    learner.voc_optimizer = torch.optim.SGD([parameter], lr=0.1)
    learner.actor_net = nn.Module()
    learner.actor_net.register_parameter("voc_test", parameter)
    learner.flags = type(
        "Flags",
        (),
        {
            "float16": False,
            "actor_grad_norm_clipping": 0.5,
            "actor_amp_max_consecutive_skips": 8,
        },
    )()
    learner.voc_amp_skip_count = 0
    learner.voc_amp_consecutive_skips = 0

    result = learner._step_voc_optimizer(T=2, B=3)

    assert result.optimizer_stepped is True
    assert result.total_norm == pytest.approx(10.0)
    # Existing critic clipping is coefficient * T * B = 3.0; no schema
    # introduces a new normalization or optimizer surface.  In particular,
    # bounded per-row Huber slope does not waive aggregate clipping.
    torch.testing.assert_close(parameter.detach(), torch.tensor([0.7]))
    assert learner.voc_amp_skip_count == 0
    assert learner.voc_amp_consecutive_skips == 0


@pytest.mark.parametrize("schema", [8, 9, 10])
def test_schema8_through_schema10_q_amp_skip_is_recoverable(schema):
    class SkippingScaler:
        def __init__(self):
            self.scale = 32.0

        def get_scale(self):
            return self.scale

        def step(self, _optimizer):
            pass

        def update(self):
            self.scale = 16.0

    parameter = nn.Parameter(torch.tensor([1.0]))
    parameter.grad = torch.tensor([float("inf")])
    learner = object.__new__(SActorLearner)
    learner.voc_gate_policy_schema_version = schema
    learner.voc_parameters = [parameter]
    learner.voc_optimizer = torch.optim.SGD([parameter], lr=0.1)
    learner.actor_net = nn.Module()
    learner.actor_net.register_parameter("voc_test", parameter)
    learner.flags = type(
        "Flags",
        (),
        {
            "float16": True,
            "actor_grad_norm_clipping": 0.5,
            "actor_amp_max_consecutive_skips": 8,
        },
    )()
    learner.voc_scaler = SkippingScaler()
    learner.voc_amp_skip_count = 0
    learner.voc_amp_consecutive_skips = 0
    learner._logger = type(
        "Logger", (), {"warning": lambda *_args: None}
    )()

    result = learner._step_voc_optimizer(T=2, B=3)

    assert result.optimizer_stepped is False
    assert result.total_norm == 0.0
    assert result.amp_scale_before == 32.0
    assert result.amp_scale_after == 16.0
    assert result.nonfinite_gradient_names == ("voc_test",)
    assert learner.voc_amp_skip_count == 1
    assert learner.voc_amp_consecutive_skips == 1
    torch.testing.assert_close(parameter.detach(), torch.tensor([1.0]))


@pytest.mark.parametrize(
    "schema",
    [None, 9, 10, 11, 12],
    ids=["legacy", "schema9", "schema10", "schema11", "schema12"],
)
def test_exact_projection_zero_training_support_does_not_update_gate(
    tmp_path, monkeypatch, schema
):
    if schema is None:
        flags = _exact_projection_learner_flags(
            tmp_path, "voc-exact-heldout-only"
        )
    elif schema == 9:
        flags = _schema9_learner_flags(
            tmp_path, "voc-schema9-exact-heldout-only"
        )
    elif schema == 10:
        flags = _schema10_learner_flags(
            tmp_path, "voc-schema10-exact-heldout-only"
        )
    elif schema == 11:
        flags = _schema11_learner_flags(
            tmp_path, "voc-schema11-exact-heldout-only"
        )
    else:
        flags = _schema12_learner_flags(
            tmp_path, "voc-schema12-exact-heldout-only"
        )
    actor, train_out, initial_actor_state = _rollout(flags)
    monkeypatch.setattr(
        "thinker.learn_actor.FileWriter", lambda **_kwargs: _NullWriter()
    )
    learner = SActorLearner(
        ray_obj=None,
        actor_param={},
        flags=flags,
        actor_net=actor,
        device=torch.device("cpu"),
    )
    writer = _CaptureWriter()
    learner.plogger = writer
    heldout_data = _select_rollout_actor_stream(
        train_out, initial_actor_state, 0
    )
    assert heldout_data[0].id.item() == 0

    learner.consume_data(heldout_data)

    assert learner.voc_update_count == 0
    assert learner.voc_ema_gate_update_count == 0
    assert learner.voc_gate_update_count == 0
    assert learner.voc_holdout_count > 0
    assert learner._last_voc_gate_exact_projection_applied is False
    assert learner.voc_gate_optimizer.state == {}
    assert all(
        torch.count_nonzero(parameter).item() == 0
        for parameter in learner.voc_gate_parameters
    )
    record = writer.records[0]
    assert record["actor/voc_gate_exact_projection_applied"] == 0.0
    assert record["actor/voc_gate_optimizer_stepped"] == 0.0
    assert record["actor/voc_gate_update_count"] == 0.0


def _schema11_test_optimizer(*, lr=3.0e-4, eps=1.0e-8):
    weight = nn.Parameter(
        torch.tensor(
            [[0.25, -0.5, 0.75], [-1.0, 1.25, -1.5]],
            dtype=torch.float32,
        )
    )
    bias = nn.Parameter(torch.tensor([0.125, -0.25], dtype=torch.float32))
    optimizer = learn_actor_module._VoCOrthoCDAdam(
        [weight, bias], lr=lr, eps=eps
    )
    return weight, bias, optimizer


def _schema11_scale(device="cpu"):
    return torch.tensor(
        [0x3F3504F3], dtype=torch.int32, device=device
    ).view(torch.float32)[0]


def _schema11_oracle_step(raw_parameters, raw_gradients, states, lr, eps):
    scale = _schema11_scale(raw_parameters[0].device)
    md_gradients = []
    scratch_parameters = []
    candidate_states = []
    for raw_parameter, raw_gradient, state in zip(
        raw_parameters, raw_gradients, states
    ):
        raw_c = raw_gradient[0].detach().clone()
        raw_s = raw_gradient[1].detach().clone()
        md_gradients.append(
            torch.stack(
                (
                    torch.mul(scale, torch.add(raw_c, raw_s)),
                    torch.mul(scale, torch.sub(raw_c, raw_s)),
                ),
                dim=0,
            )
        )
        scratch_parameters.append(torch.zeros_like(raw_parameter))
        if state is None:
            state = {
                "step": torch.tensor(0.0, dtype=torch.float32),
                "exp_avg": torch.zeros_like(raw_parameter),
                "exp_avg_sq": torch.zeros_like(raw_parameter),
            }
        else:
            state = {key: value.clone() for key, value in state.items()}
        candidate_states.append(state)
    learn_actor_module._torch_adam.adam(
        scratch_parameters,
        md_gradients,
        [state["exp_avg"] for state in candidate_states],
        [state["exp_avg_sq"] for state in candidate_states],
        [],
        [state["step"] for state in candidate_states],
        foreach=True,
        fused=False,
        capturable=False,
        differentiable=False,
        decoupled_weight_decay=False,
        grad_scale=None,
        found_inf=None,
        has_complex=False,
        amsgrad=False,
        maximize=False,
        weight_decay=0,
        beta1=0.9,
        beta2=0.999,
        lr=lr,
        eps=eps,
    )
    candidates = []
    for raw_parameter, coordinate_delta in zip(
        raw_parameters, scratch_parameters
    ):
        delta_m = coordinate_delta[0].detach().clone()
        delta_d = coordinate_delta[1].detach().clone()
        delta_c = torch.mul(scale, torch.add(delta_m, delta_d))
        delta_s = torch.mul(scale, torch.sub(delta_m, delta_d))
        raw_c = raw_parameter[0].detach().clone()
        raw_s = raw_parameter[1].detach().clone()
        candidates.append(
            torch.stack(
                (torch.add(raw_c, delta_c), torch.add(raw_s, delta_s)),
                dim=0,
            )
        )
    return candidates, candidate_states, scratch_parameters, md_gradients


def _assert_optimizer_state_dict_bytes_equal(actual, expected):
    assert actual["param_groups"] == expected["param_groups"]
    assert actual["state"].keys() == expected["state"].keys()
    for parameter_id in actual["state"]:
        assert actual["state"][parameter_id].keys() == (
            expected["state"][parameter_id].keys()
        )
        for key in actual["state"][parameter_id]:
            actual_value = actual["state"][parameter_id][key]
            expected_value = expected["state"][parameter_id][key]
            assert torch.equal(
                actual_value.contiguous().reshape(-1).view(torch.uint8),
                expected_value.contiguous().reshape(-1).view(torch.uint8),
            ), (parameter_id, key)


def test_schema11_scale_bits_transform_order_and_real_jacobian():
    scale = learn_actor_module._VoCOrthoCDAdam._orthocd_scale(
        torch.device("cpu")
    )
    assert scale.dtype == torch.float32
    assert scale.view(torch.int32).item() == 0x3F3504F3
    assert scale.item() == 0.7071067690849304

    raw = torch.tensor(
        [[0.0, -0.0, 3.25, -7.5], [-0.0, 0.0, -1.75, 2.25]],
        dtype=torch.float32,
    )
    raw_c = raw[0].clone()
    raw_s = raw[1].clone()
    expected = torch.stack(
        (
            torch.mul(scale, torch.add(raw_c, raw_s)),
            torch.mul(scale, torch.sub(raw_c, raw_s)),
        ),
        dim=0,
    )
    actual = learn_actor_module._VoCOrthoCDAdam._transform_raw_gradient(
        raw, "test"
    )
    assert torch.equal(
        actual.contiguous().view(torch.uint8),
        expected.contiguous().view(torch.uint8),
    )
    ideal_scale = 1.0 / math.sqrt(2.0)
    matrix = torch.tensor(
        [[ideal_scale, ideal_scale], [ideal_scale, -ideal_scale]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        matrix @ matrix.T,
        torch.eye(2, dtype=torch.float64),
        rtol=0.0,
        atol=3.0e-16,
    )


def test_schema11_adapter_matches_pinned_functional_oracle_first_and_later():
    weight, bias, optimizer = _schema11_test_optimizer()
    expected_parameters = [weight.detach().clone(), bias.detach().clone()]
    expected_states = [None, None]
    gradients = (
        (
            torch.tensor(
                [[1.0, -2.0, 0.25], [3.0, 0.5, -4.0]],
                dtype=torch.float32,
            ),
            torch.tensor([0.75, -1.25], dtype=torch.float32),
        ),
        (
            torch.tensor(
                [[-0.125, 2.5, -3.0], [1.75, -0.5, 0.25]],
                dtype=torch.float32,
            ),
            torch.tensor([-2.0, 3.5], dtype=torch.float32),
        ),
    )
    for weight_gradient, bias_gradient in gradients:
        raw_gradients = [weight_gradient, bias_gradient]
        expected_parameters, expected_states, _, _ = _schema11_oracle_step(
            expected_parameters,
            raw_gradients,
            expected_states,
            optimizer.param_groups[0]["lr"],
            optimizer.param_groups[0]["eps"],
        )
        weight.grad = weight_gradient.clone()
        bias.grad = bias_gradient.clone()
        raw_grad_clones = [weight.grad.clone(), bias.grad.clone()]
        optimizer.step()
        for actual, expected in zip(
            (weight, bias), expected_parameters
        ):
            assert torch.equal(
                actual.detach().contiguous().view(torch.uint8),
                expected.contiguous().view(torch.uint8),
            )
        for parameter, expected_state in zip(
            (weight, bias), expected_states
        ):
            assert optimizer.state[parameter].keys() == expected_state.keys()
            for key in expected_state:
                assert torch.equal(
                    optimizer.state[parameter][key]
                    .contiguous()
                    .reshape(-1)
                    .view(torch.uint8),
                    expected_state[key]
                    .contiguous()
                    .reshape(-1)
                    .view(torch.uint8),
                )
        assert torch.equal(weight.grad, raw_grad_clones[0])
        assert torch.equal(bias.grad, raw_grad_clones[1])
    assert optimizer.param_groups[0]["foreach"] is None
    assert optimizer.param_groups[0]["fused"] is None
    assert set(optimizer.state[weight]) == {"step", "exp_avg", "exp_avg_sq"}
    assert optimizer.state[weight]["step"].device.type == "cpu"
    assert optimizer.state[weight]["step"].dtype == torch.float32
    assert optimizer.state[weight]["step"].item() == 2.0


def test_schema11_adapter_calls_one_functional_adam_on_positive_zero_scratch(
    monkeypatch,
):
    weight, bias, optimizer = _schema11_test_optimizer()
    weight.grad = torch.tensor(
        [[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]], dtype=torch.float32
    )
    bias.grad = torch.tensor([7.0, -8.0], dtype=torch.float32)
    original_adam = learn_actor_module._torch_adam.adam
    calls = []

    def capture(params, grads, exp_avgs, exp_avg_sqs, max_sqs, steps, **kwargs):
        calls.append(
            {
                "params": [parameter.clone() for parameter in params],
                "requires_grad": [parameter.requires_grad for parameter in params],
                "kwargs": dict(kwargs),
                "count": len(params),
                "max_sqs": list(max_sqs),
                "steps": [step.clone() for step in steps],
            }
        )
        return original_adam(
            params,
            grads,
            exp_avgs,
            exp_avg_sqs,
            max_sqs,
            steps,
            **kwargs,
        )

    monkeypatch.setattr(learn_actor_module._torch_adam, "adam", capture)
    optimizer.step()

    assert len(calls) == 1
    call = calls[0]
    assert call["count"] == 2
    assert call["max_sqs"] == []
    assert [step.item() for step in call["steps"]] == [0.0, 0.0]
    assert call["requires_grad"] == [False, False]
    for scratch in call["params"]:
        assert torch.count_nonzero(scratch).item() == 0
        assert not torch.signbit(scratch).any().item()
    expected_keywords = {
        "foreach": True,
        "fused": False,
        "capturable": False,
        "differentiable": False,
        "decoupled_weight_decay": False,
        "grad_scale": None,
        "found_inf": None,
        "has_complex": False,
        "amsgrad": False,
        "maximize": False,
        "weight_decay": 0,
        "beta1": 0.9,
        "beta2": 0.999,
        "lr": optimizer.param_groups[0]["lr"],
        "eps": optimizer.param_groups[0]["eps"],
    }
    assert call["kwargs"] == expected_keywords


@pytest.mark.parametrize(
    ("label", "index"),
    [
        ("parameter", 0),
        ("parameter", 1),
        ("state.step", 0),
        ("state.exp_avg", 0),
        ("state.exp_avg_sq", 0),
        ("state.step", 1),
        ("state.exp_avg", 1),
        ("state.exp_avg_sq", 1),
    ],
)
def test_schema11_commit_injection_rolls_back_every_live_position(
    label, index, monkeypatch
):
    weight, bias, optimizer = _schema11_test_optimizer()
    weight.grad = torch.tensor(
        [[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]], dtype=torch.float32
    )
    bias.grad = torch.tensor([0.5, -0.75], dtype=torch.float32)
    optimizer.step()
    raw_before = [weight.detach().clone(), bias.detach().clone()]
    grads = [
        torch.tensor(
            [[-3.0, 2.0, -1.0], [0.25, 0.5, 0.75]], dtype=torch.float32
        ),
        torch.tensor([-1.25, 2.5], dtype=torch.float32),
    ]
    weight.grad, bias.grad = [gradient.clone() for gradient in grads]
    state_before = copy.deepcopy(optimizer.state_dict())
    live_state_ids = {
        parameter: {
            key: id(value) for key, value in optimizer.state[parameter].items()
        }
        for parameter in (weight, bias)
    }

    def inject(candidate_label, candidate_index):
        if (candidate_label, candidate_index) == (label, index):
            raise RuntimeError("injected commit fault")

    monkeypatch.setattr(optimizer, "_commit_injection_point", inject)
    with pytest.raises(RuntimeError, match="injected commit fault"):
        optimizer.step()

    for actual, expected in zip((weight, bias), raw_before):
        assert torch.equal(
            actual.detach().contiguous().view(torch.uint8),
            expected.contiguous().view(torch.uint8),
        )
    _assert_optimizer_state_dict_bytes_equal(
        optimizer.state_dict(), state_before
    )
    for parameter in (weight, bias):
        assert {
            key: id(value) for key, value in optimizer.state[parameter].items()
        } == live_state_ids[parameter]
    assert torch.equal(weight.grad, grads[0])
    assert torch.equal(bias.grad, grads[1])


def test_schema11_transform_functional_and_staged_failures_touch_no_live_state(
    monkeypatch,
):
    max_float = torch.finfo(torch.float32).max
    weight, bias, optimizer = _schema11_test_optimizer()
    raw_before = [weight.detach().clone(), bias.detach().clone()]
    weight.grad = torch.full_like(weight, max_float)
    bias.grad = torch.ones_like(bias)
    calls = {"functional": 0}

    def unexpected_functional(*_args, **_kwargs):
        calls["functional"] += 1
        raise AssertionError("functional Adam reached")

    monkeypatch.setattr(
        learn_actor_module._torch_adam, "adam", unexpected_functional
    )
    with pytest.raises(FloatingPointError, match="m/d gradient"):
        optimizer.step()
    assert calls["functional"] == 0
    assert optimizer.state == {}
    for actual, expected in zip((weight, bias), raw_before):
        assert torch.equal(actual, expected)

    weight.grad = torch.ones_like(weight)
    bias.grad = torch.ones_like(bias)
    with pytest.raises(AssertionError, match="functional Adam reached"):
        optimizer.step()
    assert calls["functional"] == 1
    assert optimizer.state == {}
    for actual, expected in zip((weight, bias), raw_before):
        assert torch.equal(actual, expected)


def test_schema11_staged_nonfinite_failure_and_rollback_failure_are_fatal(
    monkeypatch,
):
    weight, bias, optimizer = _schema11_test_optimizer()
    weight.grad = torch.ones_like(weight)
    bias.grad = torch.ones_like(bias)
    raw_before = [weight.detach().clone(), bias.detach().clone()]
    original_adam = learn_actor_module._torch_adam.adam

    def poison_candidate(params, *args, **kwargs):
        original_adam(params, *args, **kwargs)
        params[0].fill_(float("inf"))

    monkeypatch.setattr(
        learn_actor_module._torch_adam, "adam", poison_candidate
    )
    with pytest.raises(FloatingPointError, match="coordinate delta"):
        optimizer.step()
    assert optimizer.state == {}
    for actual, expected in zip((weight, bias), raw_before):
        assert torch.equal(actual, expected)

    monkeypatch.setattr(learn_actor_module._torch_adam, "adam", original_adam)
    monkeypatch.setattr(
        optimizer,
        "_commit_injection_point",
        lambda label, index: (_ for _ in ()).throw(
            RuntimeError("commit fault")
        )
        if (label, index) == ("parameter", 0)
        else None,
    )
    monkeypatch.setattr(
        optimizer,
        "_rollback_injection_point",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("rollback fault")),
    )
    with pytest.raises(RuntimeError, match="commit rollback failed"):
        optimizer.step()


def test_schema11_runtime_and_external_group_preflight_precede_staging(
    monkeypatch,
):
    weight, bias, optimizer = _schema11_test_optimizer()
    weight.grad = torch.ones_like(weight)
    bias.grad = torch.ones_like(bias)
    raw_before = [weight.detach().clone(), bias.detach().clone()]
    optimizer.param_groups[0]["foreach"] = True
    with pytest.raises(RuntimeError, match="foreach=None"):
        optimizer.step()
    assert optimizer.state == {}
    for actual, expected in zip((weight, bias), raw_before):
        assert torch.equal(actual, expected)

    optimizer.param_groups[0]["foreach"] = None
    monkeypatch.setattr(
        learn_actor_module,
        "_voc_orthocd_source_sha256",
        lambda _path: "0" * 64,
    )
    with pytest.raises(RuntimeError, match="source hash mismatch"):
        optimizer.step()
    assert optimizer.state == {}
    for actual, expected in zip((weight, bias), raw_before):
        assert torch.equal(actual, expected)


def test_schema11_step_keeps_raw_norm_telemetry_and_clips_exactly_once(
    monkeypatch,
):
    weight, bias, optimizer = _schema11_test_optimizer(lr=1.0e-3)
    weight.grad = torch.full_like(weight, 10.0)
    bias.grad = torch.full_like(bias, -10.0)
    learner = object.__new__(SActorLearner)
    learner.voc_gate_policy_schema_version = (
        util.VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION
    )
    learner.voc_parameters = [weight, bias]
    learner.voc_optimizer = optimizer
    learner.actor_net = nn.Module()
    learner.actor_net.register_parameter("voc_weight", weight)
    learner.actor_net.register_parameter("voc_bias", bias)
    learner.flags = type(
        "Flags",
        (),
        {
            "float16": False,
            "actor_grad_norm_clipping": 0.5,
            "actor_amp_max_consecutive_skips": 8,
        },
    )()
    learner.voc_amp_skip_count = 0
    learner.voc_amp_consecutive_skips = 0
    original_clip = torch.nn.utils.clip_grad_norm_
    calls = []

    def capture_clip(parameters, max_norm):
        result = original_clip(parameters, max_norm)
        calls.append(
            (max_norm, [parameter.grad.clone() for parameter in parameters])
        )
        return result

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", capture_clip)
    result = learner._step_voc_optimizer(T=2, B=3)

    assert result.optimizer_stepped is True
    assert result.total_norm == pytest.approx(math.sqrt(800.0))
    assert len(calls) == 1
    assert calls[0][0] == 3.0
    assert torch.equal(weight.grad, calls[0][1][0])
    assert torch.equal(bias.grad, calls[0][1][1])
    assert float(util.compute_grad_norm([weight, bias])) <= 3.000001


def test_schema11_amp_found_inf_skips_adapter_and_reports_zero_norm(
    monkeypatch,
):
    class SkippingScaler:
        def __init__(self):
            self.scale = 32.0
            self.step_count = 0
            self.update_count = 0

        def get_scale(self):
            return self.scale

        def step(self, _optimizer):
            self.step_count += 1

        def update(self):
            self.update_count += 1
            self.scale = 16.0

    weight, bias, optimizer = _schema11_test_optimizer()
    weight.grad = torch.full_like(weight, float("inf"))
    bias.grad = torch.ones_like(bias)
    learner = object.__new__(SActorLearner)
    learner.voc_gate_policy_schema_version = (
        util.VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION
    )
    learner.voc_parameters = [weight, bias]
    learner.voc_optimizer = optimizer
    learner.actor_net = nn.Module()
    learner.actor_net.register_parameter("voc_weight", weight)
    learner.actor_net.register_parameter("voc_bias", bias)
    learner.flags = type(
        "Flags",
        (),
        {
            "float16": True,
            "actor_grad_norm_clipping": 0.5,
            "actor_amp_max_consecutive_skips": 8,
        },
    )()
    learner.voc_scaler = SkippingScaler()
    learner.voc_amp_skip_count = 0
    learner.voc_amp_consecutive_skips = 0
    learner._logger = type(
        "Logger", (), {"warning": lambda *_args: None}
    )()
    monkeypatch.setattr(
        learn_actor_module._torch_adam,
        "adam",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("adapter functional call reached")
        ),
    )
    monkeypatch.setattr(
        torch.nn.utils,
        "clip_grad_norm_",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("clip reached")
        ),
    )
    result = learner._step_voc_optimizer(T=2, B=3)

    assert result.optimizer_stepped is False
    assert result.total_norm == 0.0
    assert result.amp_scale_before == 32.0
    assert result.amp_scale_after == 16.0
    assert result.nonfinite_gradient_names == ("voc_weight",)
    assert learner.voc_scaler.step_count == 1
    assert learner.voc_scaler.update_count == 1
    assert optimizer.state == {}


def test_schema11_and_schema12_fresh_learners_select_same_adapter_surface(
    tmp_path, monkeypatch
):
    flags10 = _schema10_learner_flags(tmp_path, "voc-schema10-fresh-control")
    flags11 = _schema11_learner_flags(tmp_path, "voc-schema11-fresh-control")
    flags12 = _schema12_learner_flags(tmp_path, "voc-schema12-fresh-control")
    torch.manual_seed(7319)
    actor10, _, _ = _rollout(flags10)
    torch.manual_seed(7319)
    actor11, _, _ = _rollout(flags11)
    torch.manual_seed(7319)
    actor12, _, _ = _rollout(flags12)
    monkeypatch.setattr(
        "thinker.learn_actor.FileWriter", lambda **_kwargs: _NullWriter()
    )
    learner10 = SActorLearner(
        ray_obj=None,
        actor_param={},
        flags=flags10,
        actor_net=actor10,
        device=torch.device("cpu"),
    )
    learner11 = SActorLearner(
        ray_obj=None,
        actor_param={},
        flags=flags11,
        actor_net=actor11,
        device=torch.device("cpu"),
    )
    learner12 = SActorLearner(
        ray_obj=None,
        actor_param={},
        flags=flags12,
        actor_net=actor12,
        device=torch.device("cpu"),
    )

    assert isinstance(
        learner11.voc_optimizer, learn_actor_module._VoCOrthoCDAdam
    )
    assert isinstance(
        learner12.voc_optimizer, learn_actor_module._VoCOrthoCDAdam
    )
    assert type(learner10.voc_optimizer) is torch.optim.Adam
    assert learner10.voc_optimizer.state_dict() == (
        learner11.voc_optimizer.state_dict()
    )
    assert learner11.voc_optimizer.state_dict() == (
        learner12.voc_optimizer.state_dict()
    )
    assert learner10.voc_scheduler.state_dict() == (
        learner11.voc_scheduler.state_dict()
    )
    assert learner11.voc_scheduler.state_dict() == (
        learner12.voc_scheduler.state_dict()
    )
    assert actor10.state_dict().keys() == actor11.state_dict().keys()
    for key, value in actor10.state_dict().items():
        assert torch.equal(value, actor11.state_dict()[key]), key
        assert torch.equal(value, actor12.state_dict()[key]), key
    assert torch.equal(
        learner10.voc_ema_gate_weight, learner11.voc_ema_gate_weight
    )
    assert torch.equal(
        learner10.voc_ema_gate_bias, learner11.voc_ema_gate_bias
    )
    assert torch.equal(
        learner11.voc_ema_gate_weight, learner12.voc_ema_gate_weight
    )
    assert torch.equal(
        learner11.voc_ema_gate_bias, learner12.voc_ema_gate_bias
    )
    assert learner11.voc_gate_target_tau == 0.1
    assert learner12.voc_gate_target_tau == 1.0


@pytest.mark.parametrize("schema", [6, 7, 8, 9, 10])
def test_schema10_and_earlier_voc_optimizer_route_stays_stock_adam(schema):
    learner = object.__new__(SActorLearner)
    learner.voc_gate_policy_schema_version = schema
    learner.flags = type(
        "Flags", (), {"actor_use_rms": False, "actor_adam_eps": 1.0e-8}
    )()
    parameters = [nn.Parameter(torch.zeros(2, 3)), nn.Parameter(torch.zeros(2))]
    optimizer = learner._make_voc_optimizer(parameters, 3.0e-4)
    assert type(optimizer) is torch.optim.Adam
    assert not isinstance(optimizer, learn_actor_module._VoCOrthoCDAdam)
    reference_parameters = [
        nn.Parameter(torch.zeros(2, 3)),
        nn.Parameter(torch.zeros(2)),
    ]
    reference = learner._make_optimizer(reference_parameters, 3.0e-4)
    assert optimizer.state_dict() == reference.state_dict()


@pytest.mark.parametrize("schema", [11, 12])
def test_schema11_and_schema12_reject_rms_without_constructing_optimizer(
    schema,
):
    learner = object.__new__(SActorLearner)
    learner.voc_gate_policy_schema_version = schema
    learner.flags = type(
        "Flags", (), {"actor_use_rms": True, "actor_adam_eps": 1.0e-8}
    )()
    parameters = [nn.Parameter(torch.zeros(2, 3)), nn.Parameter(torch.zeros(2))]
    with pytest.raises(ValueError, match="requires inherited Adam"):
        learner._make_voc_optimizer(parameters, 3.0e-4)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("reverse", "weight must have shape"),
        ("second_group", "exactly one group"),
        ("weight_dtype", "dense real FP32"),
        ("bias_shape", "bias must have shape"),
        ("betas", "betas"),
        ("fused", "fused=None"),
        ("capturable", "capturable=False"),
        ("differentiable", "differentiable=False"),
        ("amsgrad", "amsgrad=False"),
        ("maximize", "maximize=False"),
        ("decoupled", "decoupled_weight_decay=False"),
        ("weight_decay", "weight_decay=0"),
        ("lr", "finite valid lr"),
        ("eps", "finite valid eps"),
    ],
)
def test_schema11_preflight_attack_matrix_is_pre_staging(
    mutation, message
):
    weight, bias, optimizer = _schema11_test_optimizer()
    weight.grad = torch.ones_like(weight)
    bias.grad = torch.ones_like(bias)
    if mutation == "reverse":
        optimizer.param_groups[0]["params"].reverse()
    elif mutation == "second_group":
        extra = nn.Parameter(torch.zeros(2, 1))
        extra.grad = torch.zeros_like(extra)
        optimizer.add_param_group({"params": [extra]})
    elif mutation == "weight_dtype":
        weight.data = weight.detach().double()
        weight.grad = torch.ones_like(weight)
    elif mutation == "bias_shape":
        bias.data = torch.zeros(3, dtype=torch.float32)
        bias.grad = torch.ones_like(bias)
    elif mutation == "betas":
        optimizer.param_groups[0]["betas"] = (0.8, 0.999)
    elif mutation == "fused":
        optimizer.param_groups[0]["fused"] = False
    elif mutation == "capturable":
        optimizer.param_groups[0]["capturable"] = True
    elif mutation == "differentiable":
        optimizer.param_groups[0]["differentiable"] = True
    elif mutation == "amsgrad":
        optimizer.param_groups[0]["amsgrad"] = True
    elif mutation == "maximize":
        optimizer.param_groups[0]["maximize"] = True
    elif mutation == "decoupled":
        optimizer.param_groups[0]["decoupled_weight_decay"] = True
    elif mutation == "weight_decay":
        optimizer.param_groups[0]["weight_decay"] = 1
    elif mutation == "lr":
        optimizer.param_groups[0]["lr"] = float("nan")
    elif mutation == "eps":
        optimizer.param_groups[0]["eps"] = 0.0
    else:
        raise AssertionError(mutation)
    raw_before = [weight.detach().clone(), bias.detach().clone()]
    with pytest.raises(RuntimeError, match=message):
        optimizer.step()
    assert optimizer.state == {}
    assert torch.equal(weight, raw_before[0])
    assert torch.equal(bias, raw_before[1])


def test_schema11_first_step_commit_failure_restores_absent_lazy_state(
    monkeypatch,
):
    weight, bias, optimizer = _schema11_test_optimizer()
    weight.grad = torch.ones_like(weight)
    bias.grad = torch.ones_like(bias)
    raw_before = [weight.detach().clone(), bias.detach().clone()]

    def inject(label, index):
        if (label, index) == ("state.exp_avg", 0):
            raise RuntimeError("first-step commit fault")

    monkeypatch.setattr(optimizer, "_commit_injection_point", inject)
    with pytest.raises(RuntimeError, match="first-step commit fault"):
        optimizer.step()
    assert optimizer.state == {}
    assert torch.equal(weight, raw_before[0])
    assert torch.equal(bias, raw_before[1])


def test_schema11_staged_shape_and_step_corruption_fail_before_commit(
    monkeypatch,
):
    weight, bias, optimizer = _schema11_test_optimizer()
    weight.grad = torch.ones_like(weight)
    bias.grad = torch.ones_like(bias)
    raw_before = [weight.detach().clone(), bias.detach().clone()]
    original_adam = learn_actor_module._torch_adam.adam

    def corrupt_shape(params, *args, **kwargs):
        original_adam(params, *args, **kwargs)
        params[0].resize_(1)

    monkeypatch.setattr(
        learn_actor_module._torch_adam, "adam", corrupt_shape
    )
    with pytest.raises(RuntimeError, match="coordinate delta shape"):
        optimizer.step()
    assert optimizer.state == {}
    assert torch.equal(weight, raw_before[0])
    assert torch.equal(bias, raw_before[1])

    def corrupt_step(params, grads, avgs, avg_sqs, max_sqs, steps, **kwargs):
        original_adam(
            params, grads, avgs, avg_sqs, max_sqs, steps, **kwargs
        )
        steps[0].add_(1.0)

    monkeypatch.setattr(
        learn_actor_module._torch_adam, "adam", corrupt_step
    )
    with pytest.raises(RuntimeError, match="candidate step is invalid"):
        optimizer.step()
    assert optimizer.state == {}
    assert torch.equal(weight, raw_before[0])
    assert torch.equal(bias, raw_before[1])


def test_schema11_finite_elements_nonfinite_raw_norm_exits_before_adapter(
    monkeypatch,
):
    weight, bias, optimizer = _schema11_test_optimizer()
    weight.grad = torch.ones_like(weight)
    bias.grad = torch.ones_like(bias)
    learner = object.__new__(SActorLearner)
    learner.voc_gate_policy_schema_version = (
        util.VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION
    )
    learner.voc_parameters = [weight, bias]
    learner.voc_optimizer = optimizer
    learner.actor_net = nn.Module()
    learner.actor_net.register_parameter("voc_weight", weight)
    learner.actor_net.register_parameter("voc_bias", bias)
    learner.flags = type(
        "Flags",
        (),
        {
            "float16": True,
            "actor_grad_norm_clipping": 0.5,
            "actor_amp_max_consecutive_skips": 8,
        },
    )()
    learner.voc_scaler = type(
        "UnexpectedScaler",
        (),
        {
            "get_scale": lambda _self: 32.0,
            "step": lambda *_args: (_ for _ in ()).throw(
                AssertionError("scaler step reached")
            ),
            "update": lambda *_args: (_ for _ in ()).throw(
                AssertionError("scaler update reached")
            ),
        },
    )()
    learner.voc_amp_skip_count = 0
    learner.voc_amp_consecutive_skips = 0
    monkeypatch.setattr(
        util, "compute_grad_norm", lambda _parameters: torch.tensor(float("inf"))
    )
    monkeypatch.setattr(
        torch.nn.utils,
        "clip_grad_norm_",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("clip reached")
        ),
    )
    with pytest.raises(
        FloatingPointError,
        match="even though every gradient element is finite",
    ):
        learner._step_voc_optimizer(T=2, B=3)
    assert optimizer.state == {}


@pytest.mark.parametrize("schema", [11, 12])
def test_schema11_and_schema12_consume_transaction_keep_inherited_step_order(
    tmp_path, monkeypatch, schema
):
    flags = _versioned_q_learner_flags(
        tmp_path, f"voc-schema{schema}-transaction-order", schema
    )
    actor, train_out, initial_actor_state = _rollout(flags)
    monkeypatch.setattr(
        "thinker.learn_actor.FileWriter", lambda **_kwargs: _NullWriter()
    )
    learner = SActorLearner(
        ray_obj=None,
        actor_param={},
        flags=flags,
        actor_net=actor,
        device=torch.device("cpu"),
    )
    events = []
    for label, optimizer in (
        ("zero_actor", learner.optimizer),
        ("zero_q", learner.voc_optimizer),
        ("zero_gate", learner.voc_gate_optimizer),
    ):
        original = optimizer.zero_grad

        def capture_zero(*args, _label=label, _original=original, **kwargs):
            assert args == ()
            assert kwargs == {}
            events.append(_label)
            return _original()

        monkeypatch.setattr(optimizer, "zero_grad", capture_zero)
    original_q_step = learner._step_voc_optimizer

    def capture_q_step(t, b):
        events.append("step_q")
        return original_q_step(t, b)

    monkeypatch.setattr(learner, "_step_voc_optimizer", capture_q_step)
    actor_skipped = ActorGradientStepResult(
        total_norm=0.0,
        optimizer_stepped=False,
        amp_scale_before=32.0,
        amp_scale_after=16.0,
        nonfinite_gradient_names=(),
    )

    def capture_actor_step(_parameters, _t, _b):
        events.append("step_actor")
        return actor_skipped

    monkeypatch.setattr(
        learner, "_step_actor_optimizer", capture_actor_step
    )
    original_ema = learner._update_voc_ema_gate_target
    original_projection = learner._project_voc_gate_head_to_ema_target

    def capture_ema():
        events.append("ema")
        return original_ema()

    def capture_projection():
        events.append("projection")
        return original_projection()

    monkeypatch.setattr(learner, "_update_voc_ema_gate_target", capture_ema)
    monkeypatch.setattr(
        learner,
        "_project_voc_gate_head_to_ema_target",
        capture_projection,
    )
    original_actor_scheduler = learner.scheduler.step
    original_q_scheduler = learner.voc_scheduler.step

    def capture_actor_scheduler(*args, **kwargs):
        events.append("scheduler_actor")
        return original_actor_scheduler(*args, **kwargs)

    def capture_q_scheduler(*args, **kwargs):
        events.append("scheduler_q")
        return original_q_scheduler(*args, **kwargs)

    monkeypatch.setattr(learner.scheduler, "step", capture_actor_scheduler)
    monkeypatch.setattr(learner.voc_scheduler, "step", capture_q_scheduler)

    learner.consume_data((train_out, initial_actor_state))

    assert events == [
        "zero_actor",
        "zero_q",
        "zero_gate",
        "step_q",
        "step_actor",
        "ema",
        "projection",
        "scheduler_actor",
        "scheduler_q",
    ]
    assert learner.voc_update_count == 1
    assert learner.voc_ema_gate_update_count == 1
    assert learner._last_voc_gradient_step.optimizer_stepped is True


class _Schema11TransactionScaler:
    """CPU test double that exposes the inherited GradScaler transaction."""

    def __init__(
        self,
        label,
        events,
        *,
        call_optimizer=True,
        scale_before=32.0,
        scale_after=None,
        unscale_hook=None,
    ):
        self.label = label
        self.events = events
        self.call_optimizer = call_optimizer
        self.current_scale = float(scale_before)
        self.scale_after = (
            float(scale_before) if scale_after is None else float(scale_after)
        )
        self.unscale_hook = unscale_hook
        self.unscale_count = 0
        self.step_count = 0
        self.update_count = 0

    def scale(self, loss):
        self.events.append(f"scale_{self.label}")
        scaler = self

        class ScaledLoss:
            def backward(self, *args, **kwargs):
                scaler.events.append(f"backward_{scaler.label}")
                return loss.backward(*args, **kwargs)

        return ScaledLoss()

    def unscale_(self, optimizer):
        self.events.append(f"unscale_{self.label}")
        self.unscale_count += 1
        if self.unscale_hook is not None:
            self.unscale_hook(optimizer)

    def get_scale(self):
        return self.current_scale

    def step(self, optimizer):
        self.events.append(f"scaler_step_{self.label}")
        self.step_count += 1
        if self.call_optimizer:
            return optimizer.step()
        return None

    def update(self):
        self.events.append(f"scaler_update_{self.label}")
        self.update_count += 1
        self.current_scale = self.scale_after


def _schema11_amp_transaction_learner(
    tmp_path, monkeypatch, xpid, *, schema=11
):
    flags = _versioned_q_learner_flags(tmp_path, xpid, schema)
    actor, train_out, initial_actor_state = _rollout(flags)
    monkeypatch.setattr(
        "thinker.learn_actor.FileWriter", lambda **_kwargs: _NullWriter()
    )
    learner = SActorLearner(
        ray_obj=None,
        actor_param={},
        flags=flags,
        actor_net=actor,
        device=torch.device("cpu"),
    )
    learner.flags.float16 = True
    learner.plogger = _CaptureWriter()
    return learner, train_out, initial_actor_state


@pytest.mark.parametrize("schema", [11, 12])
def test_schema11_and_schema12_amp_success_bind_full_transaction_order(
    tmp_path, monkeypatch, schema
):
    learner, train_out, initial_actor_state = _schema11_amp_transaction_learner(
        tmp_path,
        monkeypatch,
        f"voc-schema{schema}-amp-success-order",
        schema=schema,
    )
    events = []
    actor_scaler = _Schema11TransactionScaler("actor", events)
    q_scaler = _Schema11TransactionScaler("q", events)
    learner.scaler = actor_scaler
    learner.voc_scaler = q_scaler
    q_parameter_ids = {id(parameter) for parameter in learner.voc_parameters}
    actor_parameter_ids = {
        id(parameter)
        for group in learner.optimizer.param_groups
        for parameter in group["params"]
    }
    for label, optimizer in (
        ("actor", learner.optimizer),
        ("q", learner.voc_optimizer),
        ("gate", learner.voc_gate_optimizer),
    ):
        original_zero = optimizer.zero_grad

        def capture_zero(
            *args,
            _label=label,
            _optimizer=optimizer,
            _original=original_zero,
            **kwargs,
        ):
            assert args == ()
            assert kwargs == {}
            events.append(f"zero_{_label}")
            result = _original()
            assert all(
                parameter.grad is None
                for group in _optimizer.param_groups
                for parameter in group["params"]
            )
            return result

        monkeypatch.setattr(optimizer, "zero_grad", capture_zero)
    original_finite = learn_actor_module._require_finite_tensor

    def capture_total_loss_finite(label, value):
        if label in {
            "actor total loss",
            "VoC total loss",
            "dedicated VoC gate total loss",
        }:
            events.append(
                {
                    "actor total loss": "finite_actor",
                    "VoC total loss": "finite_q",
                    "dedicated VoC gate total loss": "finite_gate",
                }[label]
            )
        return original_finite(label, value)

    monkeypatch.setattr(
        learn_actor_module, "_require_finite_tensor", capture_total_loss_finite
    )
    original_grad_norm = util.compute_grad_norm

    def capture_grad_norm(parameters):
        parameters = list(parameters)
        parameter_ids = {id(parameter) for parameter in parameters}
        if parameter_ids == q_parameter_ids:
            events.append("norm_q")
        elif parameter_ids == actor_parameter_ids:
            events.append("norm_actor")
        else:
            raise AssertionError("unexpected gradient-norm parameter set")
        return original_grad_norm(parameters)

    monkeypatch.setattr(util, "compute_grad_norm", capture_grad_norm)
    q_clipped_gradients = []
    clip_counts = {"q": 0, "actor": 0}
    original_clip = torch.nn.utils.clip_grad_norm_

    def capture_clip(parameters, max_norm):
        parameters = list(parameters)
        branch = (
            "q"
            if {id(parameter) for parameter in parameters} == q_parameter_ids
            else "actor"
        )
        events.append(f"clip_{branch}")
        clip_counts[branch] += 1
        result = original_clip(parameters, max_norm)
        if branch == "q":
            q_clipped_gradients[:] = [
                parameter.grad.clone() for parameter in parameters
            ]
        return result

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", capture_clip)

    original_actor_step = learner.optimizer.step

    def capture_actor_step(*args, **kwargs):
        events.append("optimizer_actor")
        return original_actor_step(*args, **kwargs)

    monkeypatch.setattr(learner.optimizer, "step", capture_actor_step)
    original_adapter_step = learner.voc_optimizer.step
    adapter_active = {"value": False}

    def capture_adapter_step(*args, **kwargs):
        events.append("adapter_q")
        adapter_active["value"] = True
        try:
            return original_adapter_step(*args, **kwargs)
        finally:
            adapter_active["value"] = False

    monkeypatch.setattr(learner.voc_optimizer, "step", capture_adapter_step)
    original_functional_adam = learn_actor_module._torch_adam.adam

    def capture_functional_adam(*args, **kwargs):
        if adapter_active["value"]:
            events.append("functional_adam")
        return original_functional_adam(*args, **kwargs)

    monkeypatch.setattr(
        learn_actor_module._torch_adam, "adam", capture_functional_adam
    )
    original_commit = learner.voc_optimizer._commit_candidates

    def capture_commit(*args, **kwargs):
        events.append("commit_q")
        return original_commit(*args, **kwargs)

    monkeypatch.setattr(
        learner.voc_optimizer, "_commit_candidates", capture_commit
    )
    original_ema = learner._update_voc_ema_gate_target
    original_projection = learner._project_voc_gate_head_to_ema_target

    def capture_ema():
        events.append("ema")
        return original_ema()

    def capture_projection():
        events.append("projection")
        return original_projection()

    monkeypatch.setattr(learner, "_update_voc_ema_gate_target", capture_ema)
    monkeypatch.setattr(
        learner,
        "_project_voc_gate_head_to_ema_target",
        capture_projection,
    )
    original_actor_scheduler = learner.scheduler.step
    original_q_scheduler = learner.voc_scheduler.step

    def capture_actor_scheduler(*args, **kwargs):
        events.append("scheduler_actor")
        return original_actor_scheduler(*args, **kwargs)

    def capture_q_scheduler(*args, **kwargs):
        assert learner.voc_update_count == 1
        assert learner.voc_ema_gate_update_count == 1
        assert learner._last_voc_gate_exact_projection_applied is True
        events.append("scheduler_q")
        return original_q_scheduler(*args, **kwargs)

    monkeypatch.setattr(learner.scheduler, "step", capture_actor_scheduler)
    monkeypatch.setattr(learner.voc_scheduler, "step", capture_q_scheduler)

    learner.consume_data((train_out, initial_actor_state))

    assert events == [
        "finite_actor",
        "finite_q",
        "finite_gate",
        "zero_actor",
        "zero_q",
        "zero_gate",
        "scale_actor",
        "backward_actor",
        "scale_q",
        "backward_q",
        "unscale_actor",
        "unscale_q",
        "norm_q",
        "clip_q",
        "scaler_step_q",
        "adapter_q",
        "functional_adam",
        "commit_q",
        "scaler_update_q",
        "norm_actor",
        "clip_actor",
        "scaler_step_actor",
        "optimizer_actor",
        "scaler_update_actor",
        "ema",
        "projection",
        "scheduler_actor",
        "scheduler_q",
    ]
    assert actor_scaler.unscale_count == 1
    assert actor_scaler.step_count == 1
    assert actor_scaler.update_count == 1
    assert q_scaler.unscale_count == 1
    assert q_scaler.step_count == 1
    assert q_scaler.update_count == 1
    assert clip_counts == {"q": 1, "actor": 1}
    assert len(q_clipped_gradients) == len(learner.voc_parameters)
    for parameter, clipped_gradient in zip(
        learner.voc_parameters, q_clipped_gradients
    ):
        assert torch.equal(parameter.grad, clipped_gradient)
    assert learner.voc_update_count == 1
    assert learner.voc_ema_gate_update_count == 1
    assert learner._last_voc_gradient_step.optimizer_stepped is True
    record = learner.plogger.records[0]
    assert record["actor/voc_optimizer_stepped"] == 1.0
    assert record["actor/voc_ema_gate_updated"] == 1.0
    assert record["actor/voc_update_count"] == 1.0


@pytest.mark.parametrize("schema", [11, 12])
def test_schema11_and_schema12_amp_found_inf_suppress_q_downstream(
    tmp_path, monkeypatch, schema
):
    learner, train_out, initial_actor_state = _schema11_amp_transaction_learner(
        tmp_path,
        monkeypatch,
        f"voc-schema{schema}-amp-found-inf-order",
        schema=schema,
    )
    events = []
    actor_scaler = _Schema11TransactionScaler("actor", events)

    def inject_found_inf(optimizer):
        optimizer.param_groups[0]["params"][0].grad.fill_(float("inf"))

    q_scaler = _Schema11TransactionScaler(
        "q",
        events,
        call_optimizer=False,
        scale_before=32.0,
        scale_after=16.0,
        unscale_hook=inject_found_inf,
    )
    learner.scaler = actor_scaler
    learner.voc_scaler = q_scaler
    q_parameter_ids = {id(parameter) for parameter in learner.voc_parameters}
    actor_parameter_ids = {
        id(parameter)
        for group in learner.optimizer.param_groups
        for parameter in group["params"]
    }
    for label, optimizer in (
        ("actor", learner.optimizer),
        ("q", learner.voc_optimizer),
        ("gate", learner.voc_gate_optimizer),
    ):
        original_zero = optimizer.zero_grad

        def capture_zero(
            *args,
            _label=label,
            _optimizer=optimizer,
            _original=original_zero,
            **kwargs,
        ):
            assert args == ()
            assert kwargs == {}
            events.append(f"zero_{_label}")
            result = _original()
            assert all(
                parameter.grad is None
                for group in _optimizer.param_groups
                for parameter in group["params"]
            )
            return result

        monkeypatch.setattr(optimizer, "zero_grad", capture_zero)
    original_finite = learn_actor_module._require_finite_tensor

    def capture_total_loss_finite(label, value):
        if label in {
            "actor total loss",
            "VoC total loss",
            "dedicated VoC gate total loss",
        }:
            events.append(
                {
                    "actor total loss": "finite_actor",
                    "VoC total loss": "finite_q",
                    "dedicated VoC gate total loss": "finite_gate",
                }[label]
            )
        return original_finite(label, value)

    monkeypatch.setattr(
        learn_actor_module, "_require_finite_tensor", capture_total_loss_finite
    )
    original_grad_norm = util.compute_grad_norm

    def capture_grad_norm(parameters):
        parameters = list(parameters)
        parameter_ids = {id(parameter) for parameter in parameters}
        if parameter_ids == q_parameter_ids:
            events.append("norm_q")
        elif parameter_ids == actor_parameter_ids:
            events.append("norm_actor")
        else:
            raise AssertionError("unexpected gradient-norm parameter set")
        return original_grad_norm(parameters)

    monkeypatch.setattr(util, "compute_grad_norm", capture_grad_norm)
    clip_counts = {"q": 0, "actor": 0}
    original_clip = torch.nn.utils.clip_grad_norm_

    def capture_clip(parameters, max_norm):
        parameters = list(parameters)
        branch = (
            "q"
            if {id(parameter) for parameter in parameters} == q_parameter_ids
            else "actor"
        )
        events.append(f"clip_{branch}")
        clip_counts[branch] += 1
        if branch == "q":
            raise AssertionError("raw Q clipping reached on found-inf")
        return original_clip(parameters, max_norm)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", capture_clip)
    original_actor_step = learner.optimizer.step

    def capture_actor_step(*args, **kwargs):
        events.append("optimizer_actor")
        return original_actor_step(*args, **kwargs)

    monkeypatch.setattr(learner.optimizer, "step", capture_actor_step)
    original_adapter_step = learner.voc_optimizer.step
    adapter_active = {"value": False}

    def unexpected_adapter_step(*args, **kwargs):
        adapter_active["value"] = True
        try:
            raise AssertionError("schema-11 adapter reached on found-inf")
        finally:
            adapter_active["value"] = False

    monkeypatch.setattr(
        learner.voc_optimizer, "step", unexpected_adapter_step
    )
    original_functional_adam = learn_actor_module._torch_adam.adam

    def reject_q_functional_only(*args, **kwargs):
        if adapter_active["value"]:
            raise AssertionError("functional Adam reached on found-inf")
        return original_functional_adam(*args, **kwargs)

    monkeypatch.setattr(
        learn_actor_module._torch_adam,
        "adam",
        reject_q_functional_only,
    )
    monkeypatch.setattr(
        learner.voc_optimizer,
        "_commit_candidates",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Q commit reached on found-inf")
        ),
    )
    monkeypatch.setattr(
        learner,
        "_update_voc_ema_gate_target",
        lambda: (_ for _ in ()).throw(
            AssertionError("EMA reached on found-inf")
        ),
    )
    monkeypatch.setattr(
        learner,
        "_project_voc_gate_head_to_ema_target",
        lambda: (_ for _ in ()).throw(
            AssertionError("projection reached on found-inf")
        ),
    )
    original_actor_scheduler = learner.scheduler.step

    def capture_actor_scheduler(*args, **kwargs):
        events.append("scheduler_actor")
        return original_actor_scheduler(*args, **kwargs)

    monkeypatch.setattr(learner.scheduler, "step", capture_actor_scheduler)
    monkeypatch.setattr(
        learner.voc_scheduler,
        "step",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Q scheduler reached on found-inf")
        ),
    )

    learner.consume_data((train_out, initial_actor_state))

    assert events == [
        "finite_actor",
        "finite_q",
        "finite_gate",
        "zero_actor",
        "zero_q",
        "zero_gate",
        "scale_actor",
        "backward_actor",
        "scale_q",
        "backward_q",
        "unscale_actor",
        "unscale_q",
        "norm_q",
        "scaler_step_q",
        "scaler_update_q",
        "norm_actor",
        "clip_actor",
        "scaler_step_actor",
        "optimizer_actor",
        "scaler_update_actor",
        "scheduler_actor",
    ]
    assert actor_scaler.unscale_count == 1
    assert actor_scaler.step_count == 1
    assert actor_scaler.update_count == 1
    assert q_scaler.unscale_count == 1
    assert q_scaler.step_count == 1
    assert q_scaler.update_count == 1
    assert clip_counts == {"q": 0, "actor": 1}
    assert learner.voc_update_count == 0
    assert learner.voc_ema_gate_update_count == 0
    assert learner.voc_gate_update_count == 0
    assert learner.voc_gate_optimizer.state == {}
    assert learner._last_voc_gate_exact_projection_applied is False
    assert learner._last_voc_gradient_step.optimizer_stepped is False
    assert learner._last_voc_gradient_step.total_norm == 0.0
    assert torch.isinf(learner.voc_parameters[0].grad).all().item()
    assert torch.isfinite(learner.voc_parameters[1].grad).all().item()
    assert learner.voc_holdout_count > 0
    record = learner.plogger.records[0]
    assert record["actor/voc_optimizer_stepped"] == 0.0
    assert record["actor/voc_ema_gate_updated"] == 0.0
    assert record["actor/voc_update_count"] == 0.0


@pytest.mark.parametrize(
    ("failure", "message", "functional_calls", "commit_calls"),
    [
        ("transform", "injected transform failure", 0, 0),
        ("functional", "injected functional failure", 1, 0),
        ("staged", "coordinate delta", 1, 0),
        ("commit", "injected commit failure", 1, 1),
    ],
)
def test_schema11_amp_adapter_failures_preserve_grad_and_skip_scaler_update(
    failure,
    message,
    functional_calls,
    commit_calls,
    monkeypatch,
):
    weight, bias, optimizer = _schema11_test_optimizer(lr=1.0e-3)
    weight.grad = torch.tensor(
        [[2.0, -3.0, 4.0], [-5.0, 6.0, -7.0]], dtype=torch.float32
    )
    bias.grad = torch.tensor([8.0, -9.0], dtype=torch.float32)
    learner = object.__new__(SActorLearner)
    learner.voc_gate_policy_schema_version = (
        util.VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION
    )
    learner.voc_parameters = [weight, bias]
    learner.voc_optimizer = optimizer
    learner.actor_net = nn.Module()
    learner.actor_net.register_parameter("voc_weight", weight)
    learner.actor_net.register_parameter("voc_bias", bias)
    learner.flags = type(
        "Flags",
        (),
        {
            "float16": True,
            "actor_grad_norm_clipping": 0.5,
            "actor_amp_max_consecutive_skips": 8,
        },
    )()
    learner.voc_amp_skip_count = 0
    learner.voc_amp_consecutive_skips = 0
    events = []
    learner.voc_scaler = _Schema11TransactionScaler("q", events)
    learner.voc_scaler.unscale_(optimizer)
    raw_before = [weight.detach().clone(), bias.detach().clone()]
    clipped_gradients = []
    original_clip = torch.nn.utils.clip_grad_norm_

    def capture_clip(parameters, max_norm):
        parameters = list(parameters)
        events.append("clip_q")
        result = original_clip(parameters, max_norm)
        clipped_gradients[:] = [parameter.grad.clone() for parameter in parameters]
        return result

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", capture_clip)
    calls = {"functional": 0, "commit": 0}
    original_functional = learn_actor_module._torch_adam.adam

    def injected_functional(params, *args, **kwargs):
        calls["functional"] += 1
        if failure == "functional":
            raise RuntimeError("injected functional failure")
        result = original_functional(params, *args, **kwargs)
        if failure == "staged":
            params[0].fill_(float("inf"))
        return result

    monkeypatch.setattr(
        learn_actor_module._torch_adam, "adam", injected_functional
    )
    if failure == "transform":
        monkeypatch.setattr(
            optimizer,
            "_transform_raw_gradient",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("injected transform failure")
            ),
        )
    original_commit = optimizer._commit_candidates

    def capture_commit(*args, **kwargs):
        calls["commit"] += 1
        return original_commit(*args, **kwargs)

    monkeypatch.setattr(optimizer, "_commit_candidates", capture_commit)
    if failure == "commit":
        monkeypatch.setattr(
            optimizer,
            "_commit_injection_point",
            lambda label, index: (_ for _ in ()).throw(
                RuntimeError("injected commit failure")
            )
            if (label, index) == ("parameter", 0)
            else None,
        )

    with pytest.raises((RuntimeError, FloatingPointError), match=message):
        learner._step_voc_optimizer(T=2, B=3)

    assert events == ["unscale_q", "clip_q", "scaler_step_q"]
    assert learner.voc_scaler.unscale_count == 1
    assert learner.voc_scaler.step_count == 1
    assert learner.voc_scaler.update_count == 0
    assert calls == {
        "functional": functional_calls,
        "commit": commit_calls,
    }
    assert optimizer.state == {}
    for parameter, expected_raw, expected_grad in zip(
        (weight, bias), raw_before, clipped_gradients
    ):
        assert torch.equal(parameter, expected_raw)
        assert torch.equal(parameter.grad, expected_grad)


@pytest.mark.parametrize("schema", [11, 12])
def test_schema11_and_schema12_post_q_fault_is_not_reported_as_q_rollback(
    tmp_path, monkeypatch, schema
):
    flags = _versioned_q_learner_flags(
        tmp_path, f"voc-schema{schema}-post-commit-failure", schema
    )
    actor, train_out, initial_actor_state = _rollout(flags)
    monkeypatch.setattr(
        "thinker.learn_actor.FileWriter", lambda **_kwargs: _NullWriter()
    )
    learner = SActorLearner(
        ray_obj=None,
        actor_param={},
        flags=flags,
        actor_net=actor,
        device=torch.device("cpu"),
    )
    raw_before = [parameter.detach().clone() for parameter in learner.voc_parameters]
    monkeypatch.setattr(
        learner.voc_optimizer,
        "_rollback_exact",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("committed Q update was rolled back")
        ),
    )
    monkeypatch.setattr(
        learner,
        "_update_voc_ema_gate_target",
        lambda: (_ for _ in ()).throw(RuntimeError("post-Q commit failure")),
    )

    with pytest.raises(RuntimeError, match="post-Q commit failure"):
        learner.consume_data((train_out, initial_actor_state))

    assert learner._last_voc_gradient_step.optimizer_stepped is True
    assert learner.voc_update_count == 0
    assert learner.voc_ema_gate_update_count == 0
    assert learner._last_voc_gate_exact_projection_applied is False
    assert all(parameter in learner.voc_optimizer.state for parameter in learner.voc_parameters)
    assert all(
        learner.voc_optimizer.state[parameter]["step"].item() == 1.0
        for parameter in learner.voc_parameters
    )
    assert any(
        not torch.equal(parameter, before)
        for parameter, before in zip(learner.voc_parameters, raw_before)
    )


def test_schema13_q_step_stages_actual_single_clip_and_adam_candidate(
    monkeypatch,
):
    from thinker import voc_telemetry

    # The bounded CPU test environment carries the pinned source bytes under a
    # CPU build tag; production still enforces the exact CUDA build string.
    monkeypatch.setattr(
        torch, "__version__", learn_actor_module._VOC_ORTHOCD_TORCH_VERSION
    )
    pinned_source_hashes = {
        learn_actor_module._torch_adam.__file__: (
            learn_actor_module._VOC_ORTHOCD_ADAM_SOURCE_SHA256
        ),
        learn_actor_module._torch_grad_scaler.__file__: (
            learn_actor_module._VOC_ORTHOCD_GRAD_SCALER_SOURCE_SHA256
        ),
    }
    monkeypatch.setattr(
        learn_actor_module,
        "_voc_orthocd_source_sha256",
        pinned_source_hashes.__getitem__,
    )
    weight, bias, optimizer = _schema11_test_optimizer(lr=3.0e-4)
    # Force clipping so FP32 tensor arithmetic differs observably from a
    # post-hoc Python binary64 reconstruction.
    weight.grad = torch.tensor(
        [[200.0, -400.0, 600.0], [-800.0, 1000.0, -1200.0]]
    )
    bias.grad = torch.tensor([300.0, -700.0])
    raw_gradients = (weight.grad.clone(), bias.grad.clone())
    learner = object.__new__(SActorLearner)
    learner._voc_telemetry_active = True
    learner._voc_telemetry_pending = {}
    learner.voc_parameters = [weight, bias]
    learner.voc_optimizer = optimizer
    learner.actor_net = nn.Module()
    learner.actor_net.register_parameter("voc_weight", weight)
    learner.actor_net.register_parameter("voc_bias", bias)
    learner.flags = type(
        "Flags",
        (),
        {
            "float16": False,
            "actor_grad_norm_clipping": 0.5,
            "actor_amp_max_consecutive_skips": 8,
        },
    )()
    learner.voc_amp_skip_count = 0
    learner.voc_amp_consecutive_skips = 0
    clip_calls = []
    inherited_clip = torch.nn.utils.clip_grad_norm_

    def capture_clip(parameters, limit):
        result = inherited_clip(parameters, limit)
        clip_calls.append((float(limit), result.detach().clone()))
        return result

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", capture_clip)
    result = learner._step_voc_optimizer(T=3, B=16)

    assert result.optimizer_stepped is True
    assert len(clip_calls) == 1
    assert clip_calls[0][0] == 24.0
    staged = learner._voc_telemetry_pending["q_step"]
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(staged["raw_preclip"], raw_gradients)
    )
    assert isinstance(staged["candidate"], dict)
    assert optimizer._schema13_telemetry_capture is False
    actual_clip_coefficient = torch.clamp(
        24.0 / (clip_calls[0][1] + 1.0e-6), max=1.0
    )
    assert torch.equal(staged["clip_coefficient"], actual_clip_coefficient)
    python64_reconstruction = min(
        1.0, 24.0 / (float(clip_calls[0][1]) + 1.0e-6)
    )
    assert float(staged["clip_coefficient"]).hex() != (
        python64_reconstruction.hex()
    )
    diagnostics = voc_telemetry.build_stepped_q_diagnostics(
        clip_scale=float(staged["clip_coefficient"]),
        raw_preclip=staged["raw_preclip"],
        raw_postclip=staged["raw_postclip"],
        md_postclip=staged["candidate"]["md_postclip"],
        adam_m_before=staged["candidate"]["adam_m_before"],
        adam_v_before=staged["candidate"]["adam_v_before"],
        adam_m_after=staged["candidate"]["adam_m_after"],
        adam_v_after=staged["candidate"]["adam_v_after"],
        coordinate_delta=staged["candidate"]["coordinate_delta"],
        mapped_delta=staged["candidate"]["mapped_delta"],
        q_lr_used=3.0e-4,
        adam_step_after=1,
    )
    assert set(diagnostics) == set(voc_telemetry.Q_DIAGNOSTIC_FIELDS)


def test_schema13_candidate_capture_failure_precedes_live_adam_commit(
    monkeypatch,
):
    monkeypatch.setattr(
        torch, "__version__", learn_actor_module._VOC_ORTHOCD_TORCH_VERSION
    )
    pinned_source_hashes = {
        learn_actor_module._torch_adam.__file__: (
            learn_actor_module._VOC_ORTHOCD_ADAM_SOURCE_SHA256
        ),
        learn_actor_module._torch_grad_scaler.__file__: (
            learn_actor_module._VOC_ORTHOCD_GRAD_SCALER_SOURCE_SHA256
        ),
    }
    monkeypatch.setattr(
        learn_actor_module,
        "_voc_orthocd_source_sha256",
        pinned_source_hashes.__getitem__,
    )
    weight, bias, optimizer = _schema11_test_optimizer()
    weight.grad = torch.tensor(
        [[2.0, -3.0, 4.0], [-5.0, 6.0, -7.0]], dtype=torch.float32
    )
    bias.grad = torch.tensor([8.0, -9.0], dtype=torch.float32)
    before = (weight.detach().clone(), bias.detach().clone())
    optimizer._schema13_telemetry_capture = True
    commit_calls = []
    original_commit = optimizer._commit_candidates

    def capture_commit(*args, **kwargs):
        commit_calls.append("commit")
        return original_commit(*args, **kwargs)

    monkeypatch.setattr(optimizer, "_commit_candidates", capture_commit)
    monkeypatch.setattr(
        optimizer,
        "_build_schema13_telemetry_candidate",
        lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("injected telemetry capture failure")
        ),
    )

    with pytest.raises(RuntimeError, match="telemetry capture failure"):
        optimizer.step()

    assert commit_calls == []
    assert optimizer.state == {}
    assert not hasattr(optimizer, "_schema13_telemetry_candidate")
    assert torch.equal(weight, before[0])
    assert torch.equal(bias, before[1])


def test_schema13_post_ack_no_support_builds_one_complete_transaction():
    class CaptureWriter:
        transaction_count = 0

        def append_transaction(self, **values):
            self.values = values
            return {"transaction_id": 1}

    learner = object.__new__(SActorLearner)
    learner._voc_telemetry_active = True
    learner.voc_actor_policy_version = 1
    learner.voc_actor_policy_publication_count = 1
    learner.voc_actor_policy_state_sha256 = "1" * 64
    learner.voc_actor_policy_publication_history_sha256 = "2" * 64
    learner._voc_telemetry_writer = CaptureWriter()
    learner.flags = type("Flags", (), {"actor_grad_norm_clipping": 0.5})()
    learner.voc_optimizer = type("Optimizer", (), {})()
    support = torch.ones((2, 16), dtype=torch.bool)
    empty = torch.zeros_like(support)
    gate = torch.arange(16).remainder(2).repeat(2, 1)
    learner._voc_telemetry_pending = {
        "source_policy_version": 0,
        "actor_ids": torch.arange(16).reshape(1, 16),
        "real_transition": support.clone(),
        "valid_mask": support,
        "train_mask": empty,
        "holdout_mask": support,
        "gate_action": gate,
        "control_action": torch.where(
            gate == 0, torch.zeros_like(gate), torch.full_like(gate, util.STOP)
        ),
        "search_steps": (gate == 0).long(),
        "target": torch.zeros((2, 16)),
        "online_q_values": torch.zeros((2, 16, 2)),
        "ema_q_values": torch.zeros((2, 16, 2)),
        "q_loss_sum": torch.zeros(()),
        "q_status": "no_support",
        "real_step_before": 0,
        "real_step_after": 32,
        "replay_t": 3,
        "optimized_t": 2,
        "replay_b": 16,
        "voc_update_count_before": 0,
        "voc_update_count_after": 0,
        "ema_update_count_before": 0,
        "ema_update_count_after": 0,
        "projection_count_before": 0,
        "projection_count_after": 0,
        "q_scheduler_last_epoch_before": 0,
        "q_scheduler_last_epoch_after": 0,
        "q_scheduler_step_count_before": 1,
        "q_scheduler_step_count_after": 1,
        "q_lr_before": 3.0e-4,
        "q_lr_after": 3.0e-4,
        "amp_scale_before": torch.tensor(256.0),
        "amp_scale_after": torch.tensor(256.0),
        "nonfinite_gradient_parameter_count": 0,
        "adam_step_before": (
            torch.tensor(0.0),
            torch.tensor(0.0),
        ),
        "adam_step_after": (
            torch.tensor(0.0),
            torch.tensor(0.0),
        ),
    }

    result = learner._commit_schema13_telemetry_after_ack(
        terminal=True, ack_count=1
    )

    assert result == {"transaction_id": 1}
    written = learner._voc_telemetry_writer.values
    assert len(written["td_rows"]) == 720
    assert written["replay_row"]["q_status"] == "no_support"
    assert written["replay_row"]["ack_count"] == 1
    assert written["q_row"]["q_attempted"] is False
    assert written["q_row"]["clip_scale"] == "NA"
    assert learner._voc_telemetry_pending is None


def test_schema13_failure_close_aborts_writer_and_marks_legacy_unsuccessful():
    class Writer:
        poisoned = False

        def __init__(self):
            self.abort_count = 0

        def abort(self):
            self.abort_count += 1
            self.poisoned = True

    class LegacyWriter:
        def __init__(self):
            self.successful = []

        def close(self, successful=True):
            self.successful.append(successful)

    learner = object.__new__(SActorLearner)
    learner._closed = False
    learner.bc_runner = None
    learner._voc_telemetry_active = True
    learner._voc_telemetry_writer = Writer()
    learner.voc_actor_policy_barrier_runtime = False
    learner.plogger = LegacyWriter()

    learner.close(successful=False)

    assert learner._voc_telemetry_writer.abort_count == 1
    assert learner.plogger.successful == [False]


def test_schema13_terminal_seal_binds_the_live_legacy_writer_inode(
    tmp_path, monkeypatch
):
    log_path = tmp_path / "logs.csv"
    log_path.write_bytes(b"original")
    replacement = tmp_path / "replacement.csv"
    replacement.write_bytes(b"replacement")

    class LegacyWriter:
        def __init__(self):
            self.paths = {"logs": str(log_path)}
            self._logfile = open(log_path, "a")

        def close(self, successful=True):
            self._logfile.close()

    class TelemetryWriter:
        transaction_count = 1

        def seal(self, **_kwargs):
            raise AssertionError("replacement must fail before telemetry seal")

    learner = object.__new__(SActorLearner)
    learner._voc_telemetry_active = True
    learner._voc_actor_policy_transaction_open = False
    learner._voc_telemetry_pending = None
    learner._voc_telemetry_writer = TelemetryWriter()
    learner.voc_actor_policy_version = 1
    learner.voc_actor_policy_terminal = True
    learner.voc_actor_policy_publication_count = 1
    learner.voc_actor_policy_terminal_ack_count = 1
    learner.voc_actor_policy_expected_ack_count = 1
    learner.real_step = 32
    learner.flags = type("Flags", (), {"total_steps": 32})()
    learner.plogger = LegacyWriter()
    inherited_open = os.open
    replaced = False

    def replace_at_reader_open(path, flags, *args, **kwargs):
        nonlocal replaced
        if os.fspath(path) == os.fspath(log_path) and not replaced:
            replaced = True
            os.replace(replacement, log_path)
        return inherited_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", replace_at_reader_open)
    try:
        with pytest.raises(RuntimeError, match="identity is malformed"):
            learner._seal_schema13_telemetry_before_finish()
    finally:
        if not learner.plogger._logfile.closed:
            learner.plogger._logfile.close()


def test_schema13_terminal_seal_opens_legacy_log_nonblocking(
    tmp_path, monkeypatch
):
    log_path = tmp_path / "logs.csv"
    log_path.write_bytes(b"original")

    class LegacyWriter:
        def __init__(self):
            self.paths = {"logs": str(log_path)}
            self._logfile = open(log_path, "a")

        def close(self, successful=True):
            self._logfile.close()

    class TelemetryWriter:
        transaction_count = 1

        def seal(self, **_kwargs):
            raise AssertionError("test stops at the guarded reader open")

    learner = object.__new__(SActorLearner)
    learner._voc_telemetry_active = True
    learner._voc_actor_policy_transaction_open = False
    learner._voc_telemetry_pending = None
    learner._voc_telemetry_writer = TelemetryWriter()
    learner.voc_actor_policy_version = 1
    learner.voc_actor_policy_terminal = True
    learner.voc_actor_policy_publication_count = 1
    learner.voc_actor_policy_terminal_ack_count = 1
    learner.voc_actor_policy_expected_ack_count = 1
    learner.real_step = 32
    learner.flags = type("Flags", (), {"total_steps": 32})()
    learner.plogger = LegacyWriter()
    inherited_open = os.open

    def require_nonblocking(path, flags, *args, **kwargs):
        if os.fspath(path) == os.fspath(log_path):
            assert flags & os.O_NONBLOCK
            raise RuntimeError("guarded reader open")
        return inherited_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", require_nonblocking)
    try:
        with pytest.raises(RuntimeError, match="guarded reader open"):
            learner._seal_schema13_telemetry_before_finish()
    finally:
        if not learner.plogger._logfile.closed:
            learner.plogger._logfile.close()


@pytest.mark.parametrize("failure_point", ("call", "wait"))
def test_schema13_actor_finish_failure_poison_kills_and_marks_unsuccessful(
    monkeypatch, failure_point
):
    calls = []

    class Writer:
        poisoned = False

        def abort(self):
            self.poisoned = True
            calls.append("abort")

    class FinishRemote:
        def remote(self):
            calls.append("finish")
            if failure_point == "call":
                raise RuntimeError("injected FINISH call failure")
            return object()

    class LegacyWriter:
        def close(self, successful=True):
            calls.append(("log-close", successful))

    learner = object.__new__(SActorLearner)
    learner._closed = False
    learner.bc_runner = None
    learner._voc_telemetry_active = True
    learner._voc_telemetry_writer = Writer()
    learner._voc_telemetry_log_closed = False
    learner._seal_schema13_telemetry_before_finish = lambda: calls.append("seal")
    learner.voc_actor_policy_barrier_runtime = True
    learner.voc_actor_policy_terminal = True
    learner.voc_actor_policy_terminal_ack_count = 1
    learner.voc_actor_policy_expected_ack_count = 1
    learner.voc_actor_policy_version = 1
    learner.voc_actor_policy_barrier_timeout_s = 120.0
    learner._monotonic = lambda: 0.0
    learner._logger = type(
        "Logger", (), {"error": lambda *_args, **_kwargs: None}
    )()
    learner.plogger = LegacyWriter()
    learner.actor_buffer = type(
        "ActorBuffer", (), {"set_finish": FinishRemote()}
    )()

    def wait_for_finish(_ref, **_kwargs):
        calls.append("wait")
        if failure_point == "wait":
            raise RuntimeError("injected FINISH wait failure")

    learner._barrier_ray_get = wait_for_finish
    monkeypatch.setattr(
        learn_actor_module.ray,
        "kill",
        lambda actor, no_restart: calls.append(("kill", actor, no_restart)),
    )

    with pytest.raises(RuntimeError, match=f"FINISH {failure_point} failure"):
        learner.close(successful=True)

    assert calls.count("finish") == 1
    assert any(call == "abort" for call in calls)
    assert any(isinstance(call, tuple) and call[0] == "kill" for call in calls)
    assert ("log-close", False) in calls
    assert learner._voc_telemetry_writer.poisoned is True


def test_schema13_runner_close_failure_poison_kills_and_marks_unsuccessful(
    monkeypatch,
):
    calls = []

    class Runner:
        def close(self):
            calls.append("runner-close")
            raise RuntimeError("injected runner close failure")

    class Writer:
        poisoned = False

        def abort(self):
            self.poisoned = True
            calls.append("abort")

    class Remote:
        def remote(self, *_args):
            calls.append("abort-diagnostic")
            return object()

    class LegacyWriter:
        def close(self, successful=True):
            calls.append(("log-close", successful))

    learner = object.__new__(SActorLearner)
    learner._closed = False
    learner.bc_runner = Runner()
    learner._voc_telemetry_active = True
    learner._voc_telemetry_writer = Writer()
    learner._voc_telemetry_log_closed = False
    learner.voc_actor_policy_barrier_runtime = True
    learner.voc_actor_policy_version = 1
    learner.voc_actor_policy_barrier_timeout_s = 120.0
    learner._monotonic = lambda: 0.0
    learner._logger = type(
        "Logger", (), {"error": lambda *_args, **_kwargs: None}
    )()
    learner.actor_param_buffer = type(
        "ActorParamBuffer", (), {"set_data": Remote()}
    )()
    learner.actor_buffer = object()
    learner._barrier_ray_get = lambda ref, **_kwargs: ref
    learner.plogger = LegacyWriter()
    monkeypatch.setattr(
        learn_actor_module.ray,
        "kill",
        lambda actor, no_restart: calls.append(("kill", actor, no_restart)),
    )

    with pytest.raises(RuntimeError, match="runner close failure"):
        learner.close(successful=True)

    assert calls.count("runner-close") == 1
    assert calls.count("abort") >= 1
    assert "abort-diagnostic" in calls
    assert any(isinstance(call, tuple) and call[0] == "kill" for call in calls)
    assert ("log-close", False) in calls
    assert learner._voc_telemetry_writer.poisoned is True


def test_schema12_ack_wait_preserves_none_return_value():
    class GetRemote:
        def remote(self, _key):
            return object()

    learner = object.__new__(SActorLearner)
    learner.voc_actor_policy_expected_ack_count = 1
    learner.voc_actor_policy_barrier_timeout_s = 120.0
    learner.voc_actor_policy_malformed_bundle_count = 0
    learner.voc_actor_policy_version_mismatch_count = 0
    learner.voc_actor_policy_publication_history = [{"policy_version": 0}]
    learner.voc_actor_policy_publication_history_sha256 = ""
    learner.voc_actor_policy_publication_count = 1
    learner.voc_actor_policy_state_sha256 = "0" * 64
    learner.voc_actor_policy_terminal_ack_count = 0
    learner.voc_gate_policy_schema_version = 12
    learner.actor_param_buffer = type(
        "ActorParamBuffer", (), {"get_data": GetRemote()}
    )()
    learner._monotonic = lambda: 0.0
    learner._barrier_sleep = lambda _seconds: None
    learner._barrier_ray_get = lambda _ref, **_kwargs: {
        0: util.make_actor_policy_ack(
            0, 1, terminal=False, gate_schema=12
        )
    }

    result = learner._wait_for_actor_policy_acks(
        policy_version=1, terminal=False
    )

    assert result is None
    assert len(learner.voc_actor_policy_publication_history) == 2
