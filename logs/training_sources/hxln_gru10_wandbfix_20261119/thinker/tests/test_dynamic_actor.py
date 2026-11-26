import numpy as np
import pytest
import torch
from gymnasium import spaces

from thinker import util
from thinker.actor_net import (
    ActorNet,
    ILLEGAL_CONTROL_LOGIT,
    compute_dynamic_control_log_probs,
    compute_voc_gate_distribution,
    sample,
)


def _flags(
    dynamic,
    *,
    max_search_steps=-1,
    max_depth=40,
    rec_t=40,
    factorized=False,
    voc_mode="off",
    voc_gate_temperature=1.0,
    voc_train_epsilon=0.02,
    voc_eval_stochastic=True,
    voc_dedicated_gate=False,
    voc_gate_exact_projection=False,
    voc_gate_epsilon_greedy_execution=False,
    train_actor=True,
):
    args = [
            "--dynamic_search", str(dynamic).lower(),
            "--dynamic_factorized_control", str(factorized).lower(),
            "--dynamic_voc_mode", voc_mode,
            "--voc_gate_temperature", str(voc_gate_temperature),
            "--voc_train_epsilon", str(voc_train_epsilon),
            "--voc_eval_stochastic", str(voc_eval_stochastic).lower(),
            "--voc_gate_exact_projection",
            str(voc_gate_exact_projection).lower(),
            "--voc_gate_epsilon_greedy_execution",
            str(voc_gate_epsilon_greedy_execution).lower(),
            "--train_actor", str(train_actor).lower(),
            "--wrapper_type", "0",
            "--max_search_steps", str(max_search_steps),
            "--max_depth", str(max_depth),
            "--rec_t", str(rec_t),
            "--see_real_state", "false",
            "--see_h", "false",
            "--see_x", "false",
            "--parallel", "false",
        ]
    if voc_mode != "off":
        args += ["--think_cost", "0.0005"]
    flags = util.create_setting(
        args=args,
        save_flags=False,
    )
    # Keep legacy actor tests on the explicit flag-false path; focused v4
    # cases opt in below.  This also decouples them from the public YAML
    # default while protocol wiring is landing concurrently.
    flags.voc_dedicated_gate = bool(voc_dedicated_gate)
    return flags


def _network(flags, batch_size=3, num_actions=5):
    if flags.dynamic_search:
        width = 10 * num_actions + 14
        control_n = 3
    else:
        width = 11 + 10 * num_actions + flags.rec_t
        if flags.has_action_seq:
            width += flags.max_depth * num_actions
            if flags.reset_mode == 0:
                width += num_actions
        control_n = 2
    obs_space = spaces.Dict({
        "tree_reps": spaces.Box(
            -np.inf, np.inf, shape=(batch_size, width), dtype=np.float32
        ),
        "real_states": spaces.Box(
            -np.inf, np.inf, shape=(batch_size, 1), dtype=np.float32
        ),
    })
    action_space = spaces.Tuple((
        spaces.Tuple(tuple(spaces.Discrete(num_actions) for _ in range(batch_size))),
        spaces.Tuple(tuple(spaces.Discrete(control_n) for _ in range(batch_size))),
    ))
    return ActorNet(
        obs_space=obs_space,
        action_space=action_space,
        flags=flags,
        tree_rep_meaning=util.get_tree_rep_meaning(num_actions, 1, flags),
    )


def _env_out(flags, phases, *, token_valid=None, reset_mask=None):
    batch_size = len(phases)
    num_actions = 5
    width = 10 * num_actions + 14
    if token_valid is None:
        token_valid = torch.ones(batch_size, dtype=torch.bool)
    if reset_mask is None:
        reset_mask = torch.zeros(batch_size, dtype=torch.bool)
    state = {
        "tree_reps": torch.randn(batch_size, width),
        "real_states": torch.zeros(batch_size, 1),
        "step_status": torch.ones(batch_size, dtype=torch.long),
    }
    info = {
        "phase": torch.tensor(phases),
        "legal_control_mask": torch.tensor([
            [True, True, True] if phase == util.SEARCH_PHASE
            else [False, False, False]
            for phase in phases
        ]),
        "tree_token_valid": token_valid,
        "search_state_reset": reset_mask,
        "real_transition": torch.zeros(batch_size, dtype=torch.bool),
        "stage_end": torch.zeros(batch_size, dtype=torch.bool),
        "forced_stop": torch.zeros(batch_size, dtype=torch.bool),
        "search_steps": torch.zeros(batch_size, dtype=torch.long),
        "real_done": torch.zeros(batch_size, dtype=torch.bool),
        "truncated_done": torch.zeros(batch_size, dtype=torch.bool),
    }
    return util.init_env_out(state, info, flags, dim_actions=1, tuple_action=False)


def test_dynamic_actor_routes_mixed_phases_and_joint_log_probability():
    flags = _flags(True, max_search_steps=8)
    actor = _network(flags)
    env_out = _env_out(
        flags,
        [util.SEARCH_PHASE, util.NEED_REAL_ACTION_PHASE, util.WAIT_PHASE],
        reset_mask=torch.tensor([True, True, True]),
    )
    clamp_primary = torch.tensor([[[1], [2], [3]]])
    clamp_control = torch.tensor([[util.STOP, util.PROCEED, util.PROCEED]])

    out, _ = actor(
        env_out,
        actor.initial_state(3),
        clamp_action=(clamp_primary, clamp_control),
        compute_loss=True,
    )

    assert out.primary_valid.tolist() == [[False, True, False]]
    assert out.control_valid.tolist() == [[True, False, False]]
    assert out.policy_valid.tolist() == [[True, True, False]]
    assert out.policy_type.tolist() == [[
        util.POLICY_SEARCH, util.POLICY_REAL, util.POLICY_NONE
    ]]
    torch.testing.assert_close(
        out.c_action_log_prob[0, 0], out.misc["control_log_prob"][0, 0]
    )
    torch.testing.assert_close(
        out.c_action_log_prob[0, 1], out.misc["primary_log_prob"][0, 1]
    )
    assert out.c_action_log_prob[0, 2].item() == 0
    assert out.entropy_loss[0, 2].item() == 0


def test_dynamic_actor_keeps_illegal_control_sentinel_behavior():
    flags = _flags(True, max_search_steps=8)
    actor = _network(flags)
    env_out = _env_out(flags, [util.SEARCH_PHASE])
    legal_control_mask = env_out.legal_control_mask.clone()
    legal_control_mask[..., util.PROCEED] = False
    env_out = env_out._replace(legal_control_mask=legal_control_mask)

    out, _ = actor(
        env_out,
        actor.initial_state(1),
        clamp_action=(
            torch.tensor([[[0]]]),
            torch.tensor([[util.STOP]]),
        ),
        compute_loss=True,
    )

    assert out.search_control_logits[0, 0, util.PROCEED].item() == (
        ILLEGAL_CONTROL_LOGIT
    )
    assert out.search_control[0, 0].item() == util.STOP
    assert torch.isfinite(out.c_action_log_prob).all()
    assert torch.isfinite(out.reg_loss).all()


def test_dynamic_tree_gru_freezes_non_token_calls():
    flags = _flags(True)
    actor = _network(flags)
    first = _env_out(
        flags,
        [util.SEARCH_PHASE] * 3,
        reset_mask=torch.ones(3, dtype=torch.bool),
    )
    _, state = actor(first, actor.initial_state(3))
    second = _env_out(
        flags,
        [util.NEED_REAL_ACTION_PHASE, util.WAIT_PHASE, util.WAIT_PHASE],
        token_valid=torch.zeros(3, dtype=torch.bool),
    )
    _, frozen_state = actor(second, state)

    for before, after in zip(state, frozen_state):
        assert torch.equal(before, after)


def test_initial_real_root_is_encoded_and_truncation_resets_real_rnn():
    torch.manual_seed(31)
    flags = _flags(True)
    flags.see_real_state = True
    flags.real_state_rnn = True
    actor = _network(flags)
    env_out = _env_out(
        flags,
        [util.SEARCH_PHASE] * 3,
        reset_mask=torch.ones(3, dtype=torch.bool),
    )._replace(real_states=torch.ones(1, 3, 1))

    initial_state = actor.initial_state(3)
    _, encoded_state = actor(env_out, initial_state)
    for key in ["pre_encoded_real_state", "encoded_real_state"]:
        cached = encoded_state[actor.state_idx[key]][0]
        assert torch.count_nonzero(cached) > 0

    terminal_root = env_out._replace(
        real_transition=torch.ones_like(env_out.real_transition),
        search_state_reset=torch.ones_like(env_out.search_state_reset),
        truncated_done=torch.ones_like(
            env_out.truncated_done, dtype=torch.bool
        ),
    )
    perturbed_state = tuple(
        torch.logical_not(value)
        if value.dtype == torch.bool
        else torch.randn_like(value) * 100
        for value in initial_state
    )
    _, after_zero = actor(terminal_root, initial_state)
    _, after_perturbed = actor(terminal_root, perturbed_state)
    # The observable real-state recurrent output/cache must not retain the
    # previous episode across a truncation-only auto-reset.
    for key in ["pre_encoded_real_state", "encoded_real_state"]:
        zero_cache = after_zero[actor.state_idx[key]][0]
        perturbed_cache = after_perturbed[actor.state_idx[key]][0]
        torch.testing.assert_close(zero_cache, perturbed_cache)


def test_dynamic_parameter_count_is_independent_of_caps_and_depth():
    counts = []
    state_shapes = []
    for cap, depth, rec_t in [(-1, 5, 1), (8, 40, 40), (40, 100, 100)]:
        actor = _network(_flags(
            True, max_search_steps=cap, max_depth=depth, rec_t=rec_t
        ))
        counts.append(sum(parameter.numel() for parameter in actor.parameters()))
        state_shapes.append([tuple(state.shape) for state in actor.initial_state(3)])
    assert counts[0] == counts[1] == counts[2]
    assert state_shapes[0] == state_shapes[1] == state_shapes[2]


def test_factorized_control_preserves_every_state_dict_shape():
    legacy = _network(_flags(True, factorized=False))
    factorized = _network(_flags(True, factorized=True))

    assert legacy.state_dict().keys() == factorized.state_dict().keys()
    assert {
        key: tuple(value.shape) for key, value in legacy.state_dict().items()
    } == {
        key: tuple(value.shape) for key, value in factorized.state_dict().items()
    }


def test_factorized_actor_entropy_keeps_gate_and_conditionals_isolated():
    torch.manual_seed(41)
    flags = _flags(True, factorized=True)
    actor = _network(flags)
    env_out = _env_out(flags, [util.SEARCH_PHASE] * 3)
    out, _ = actor(
        env_out,
        actor.initial_state(3),
        clamp_action=(
            torch.tensor([[[0], [1], [2]]]),
            torch.tensor([[util.PROCEED, util.RESET, util.STOP]]),
        ),
        compute_loss=True,
    )

    primary_to_control = torch.autograd.grad(
        out.misc["primary_entropy_loss"].sum(),
        actor.reset.weight,
        retain_graph=True,
        allow_unused=True,
    )[0]
    assert primary_to_control is None or torch.count_nonzero(
        primary_to_control
    ) == 0

    bout_grad = torch.autograd.grad(
        out.misc["bout_entropy_loss"].sum(),
        actor.reset.weight,
        retain_graph=True,
    )[0]
    assert torch.count_nonzero(bout_grad[util.STOP]) == 0
    torch.testing.assert_close(
        bout_grad[util.PROCEED] + bout_grad[util.RESET],
        torch.zeros_like(bout_grad[util.PROCEED]),
        rtol=1e-6,
        atol=1e-7,
    )

    gate_grad = torch.autograd.grad(
        out.misc["gate_entropy_loss"].sum(), actor.reset.weight
    )[0]
    torch.testing.assert_close(
        gate_grad[util.PROCEED],
        gate_grad[util.RESET],
        rtol=1e-6,
        atol=1e-7,
    )


def test_voc_gate_distribution_changes_only_gate_and_reports_behavior_joint():
    raw_logits = torch.tensor([[0.7, -0.2, 0.1]], requires_grad=True)
    temperature = 2.0
    epsilon = 0.2

    distribution = compute_voc_gate_distribution(
        raw_logits,
        temperature=temperature,
        epsilon=epsilon,
    )

    continue_score = torch.logsumexp(raw_logits[..., :2], dim=-1)
    base_gate = torch.softmax(
        torch.stack((continue_score, raw_logits[..., util.STOP]), dim=-1)
        / temperature,
        dim=-1,
    )
    expected_gate = (1.0 - epsilon) * base_gate + epsilon * 0.5
    expected_bout = torch.softmax(raw_logits[..., :2], dim=-1)
    expected_joint = torch.cat(
        (
            expected_gate[..., :1] * expected_bout,
            expected_gate[..., 1:],
        ),
        dim=-1,
    )

    torch.testing.assert_close(
        torch.softmax(distribution.gate_logits, dim=-1), expected_gate
    )
    torch.testing.assert_close(
        torch.softmax(distribution.bout_logits, dim=-1), expected_bout
    )
    torch.testing.assert_close(
        torch.softmax(distribution.joint_logits, dim=-1), expected_joint
    )
    # The equivalent joint keeps the original conditional P/R preference.
    actual_joint = torch.softmax(distribution.joint_logits, dim=-1)
    torch.testing.assert_close(
        actual_joint[..., :2] / actual_joint[..., :2].sum(-1, keepdim=True),
        expected_bout,
    )


def test_dedicated_voc_gate_composes_scalar_gate_with_unchanged_bout():
    raw_logits = torch.tensor(
        [[0.7, -0.2, 19.0], [-1.0, 1.5, -23.0]], requires_grad=True
    )
    raw_gate_log_odds = torch.tensor([1.4, -0.8], requires_grad=True)
    temperature = 2.0
    epsilon = 0.2

    distribution = compute_voc_gate_distribution(
        raw_logits,
        temperature=temperature,
        epsilon=epsilon,
        raw_gate_log_odds=raw_gate_log_odds,
    )

    base_gate = torch.softmax(
        torch.stack(
            (raw_gate_log_odds, torch.zeros_like(raw_gate_log_odds)), dim=-1
        ) / temperature,
        dim=-1,
    )
    expected_gate = (1.0 - epsilon) * base_gate + epsilon * 0.5
    expected_bout = torch.softmax(raw_logits[..., :2], dim=-1)
    expected_joint = torch.cat(
        (
            expected_gate[..., :1] * expected_bout,
            expected_gate[..., 1:],
        ),
        dim=-1,
    )

    torch.testing.assert_close(
        torch.softmax(distribution.gate_logits, dim=-1), expected_gate
    )
    torch.testing.assert_close(
        torch.softmax(distribution.bout_logits, dim=-1), expected_bout
    )
    torch.testing.assert_close(
        torch.softmax(distribution.joint_logits, dim=-1), expected_joint
    )
    # The old STOP row is not part of the dedicated binary gate, while the
    # first two rows retain their exact conditional P/R ordering.
    gate_grad, control_grad = torch.autograd.grad(
        distribution.gate_logits[..., 0].sum(),
        (raw_gate_log_odds, raw_logits),
        allow_unused=True,
    )
    assert torch.count_nonzero(gate_grad) == raw_gate_log_odds.numel()
    assert control_grad is None


def test_dedicated_voc_gate_respects_legal_mask_under_full_exploration():
    raw_logits = torch.tensor([[3.0, 2.0, -4.0], [-2.0, 1.0, 4.0]])
    raw_gate_log_odds = torch.tensor([100.0, -100.0])
    legal = torch.tensor([
        [False, False, True],
        [True, True, False],
    ])

    distribution = compute_voc_gate_distribution(
        raw_logits,
        temperature=1.0,
        epsilon=1.0,
        legal_control_mask=legal,
        raw_gate_log_odds=raw_gate_log_odds,
    )
    joint = torch.softmax(distribution.joint_logits, dim=-1)

    torch.testing.assert_close(joint[0], torch.tensor([0.0, 0.0, 1.0]))
    assert joint[1, util.STOP].item() == 0.0
    torch.testing.assert_close(joint[1, :2].sum(), torch.tensor(1.0))


def test_voc_gate_exploration_never_revives_an_illegal_gate():
    raw_logits = torch.tensor([[3.0, 2.0, -4.0], [-2.0, 1.0, 4.0]])
    legal = torch.tensor([
        [False, False, True],
        [True, True, False],
    ])

    distribution = compute_voc_gate_distribution(
        raw_logits,
        temperature=1.0,
        epsilon=1.0,
        legal_control_mask=legal,
    )
    joint = torch.softmax(distribution.joint_logits, dim=-1)

    torch.testing.assert_close(joint[0], torch.tensor([0.0, 0.0, 1.0]))
    assert joint[1, util.STOP].item() == 0.0
    torch.testing.assert_close(joint[1, :2].sum(), torch.tensor(1.0))


def test_epsilon_greedy_scalar_gate_has_exact_sign_tie_and_detached_gate():
    raw_logits = torch.tensor(
        [[0.4, -0.7, 3.0], [-0.3, 1.2, -5.0], [1.1, 0.2, 8.0]],
        requires_grad=True,
    )
    raw_gate_log_odds = torch.tensor(
        [2.0, -3.0, 0.0], requires_grad=True
    )

    distribution = compute_voc_gate_distribution(
        raw_logits,
        temperature=7.0,
        epsilon=0.02,
        raw_gate_log_odds=raw_gate_log_odds,
        epsilon_greedy_execution=True,
    )

    expected = torch.tensor([0.99, 0.01, 0.5])
    assert torch.equal(distribution.continue_prob, expected)
    assert torch.equal(
        distribution.stop_prob, torch.tensor([0.01, 0.99, 0.5])
    )
    assert not distribution.continue_prob.requires_grad
    assert not distribution.gate_logits.requires_grad
    assert torch.isfinite(distribution.gate_logits).all()
    assert torch.isfinite(distribution.joint_logits).all()
    # The execution transform detaches only the binary gate.  Conditional
    # PROCEED/RESET learning remains connected to the original actor head.
    distribution.joint_logits[..., util.PROCEED].sum().backward()
    assert raw_gate_log_odds.grad is None
    assert torch.count_nonzero(raw_logits.grad[..., :2]) > 0
    assert torch.count_nonzero(raw_logits.grad[..., util.STOP]) == 0


def test_epsilon_greedy_eval_is_finite_exact_zero_one_and_samples_deterministically():
    rows = 4096
    raw_logits = torch.zeros((rows * 2, 3))
    raw_gate_log_odds = torch.cat((torch.ones(rows), -torch.ones(rows)))
    distribution = compute_voc_gate_distribution(
        raw_logits,
        epsilon=0.0,
        raw_gate_log_odds=raw_gate_log_odds,
        epsilon_greedy_execution=True,
    )

    assert torch.isfinite(distribution.gate_logits).all()
    assert torch.equal(
        distribution.continue_prob,
        torch.cat((torch.ones(rows), torch.zeros(rows))),
    )
    assert torch.equal(
        torch.softmax(distribution.gate_logits, dim=-1)[..., 0],
        distribution.continue_prob,
    )
    torch.manual_seed(7301)
    first = sample(distribution.gate_logits, greedy=False)
    first_tail = torch.rand(8)
    torch.manual_seed(7301)
    second = sample(distribution.gate_logits, greedy=False)
    second_tail = torch.rand(8)
    assert torch.equal(first, second)
    assert torch.equal(first_tail, second_tail)
    assert torch.equal(
        first,
        torch.cat(
            (
                torch.zeros(rows, dtype=torch.long),
                torch.ones(rows, dtype=torch.long),
            )
        ),
    )
    assert not first.requires_grad


def test_epsilon_greedy_training_sampling_uses_declared_distribution_and_rng():
    rows = 8192
    raw_logits = torch.zeros((rows * 3, 3))
    raw_gate_log_odds = torch.cat(
        (torch.ones(rows), -torch.ones(rows), torch.zeros(rows))
    )
    distribution = compute_voc_gate_distribution(
        raw_logits,
        epsilon=0.02,
        raw_gate_log_odds=raw_gate_log_odds,
        epsilon_greedy_execution=True,
    )

    torch.manual_seed(7351)
    first = sample(distribution.gate_logits, greedy=False)
    first_tail = torch.rand(8)
    torch.manual_seed(7351)
    second = sample(distribution.gate_logits, greedy=False)
    second_tail = torch.rand(8)
    assert torch.equal(first, second)
    assert torch.equal(first_tail, second_tail)
    positive_continue = (first[:rows] == 0).float().mean().item()
    negative_continue = (first[rows:2 * rows] == 0).float().mean().item()
    tie_continue = (first[2 * rows:] == 0).float().mean().item()
    assert 0.97 < positive_continue < 1.0
    assert 0.0 < negative_continue < 0.03
    assert 0.47 < tie_continue < 0.53


def test_epsilon_greedy_exploration_is_uniform_over_legal_gate_actions():
    raw_logits = torch.tensor([[3.0, 2.0, -4.0], [-2.0, 1.0, 4.0]])
    raw_gate_log_odds = torch.tensor([-100.0, 100.0])
    legal = torch.tensor([
        [False, False, True],
        [True, True, False],
    ])

    distribution = compute_voc_gate_distribution(
        raw_logits,
        epsilon=0.02,
        legal_control_mask=legal,
        raw_gate_log_odds=raw_gate_log_odds,
        epsilon_greedy_execution=True,
    )

    assert torch.equal(
        distribution.continue_prob, torch.tensor([0.0, 1.0])
    )
    assert torch.equal(distribution.stop_prob, torch.tensor([1.0, 0.0]))


def test_epsilon_greedy_execution_requires_boolean_and_scalar_gate():
    logits = torch.zeros((1, 3))
    with pytest.raises(TypeError, match="must be boolean"):
        compute_voc_gate_distribution(
            logits, epsilon_greedy_execution=1
        )
    with pytest.raises(ValueError, match="requires raw_gate_log_odds"):
        compute_voc_gate_distribution(
            logits, epsilon_greedy_execution=True
        )


def test_epsilon_greedy_sign_is_taken_before_any_dtype_cast():
    distribution = compute_voc_gate_distribution(
        torch.zeros((2, 3), dtype=torch.float32),
        epsilon=0.02,
        raw_gate_log_odds=torch.tensor(
            [1e-300, -1e-300], dtype=torch.float64
        ),
        epsilon_greedy_execution=True,
    )

    assert torch.equal(
        distribution.continue_prob, torch.tensor([0.99, 0.01])
    )


def test_voc_gate_projection_uses_only_legal_continue_controls():
    raw_logits = torch.tensor([[2.0, 0.4, -0.2]], requires_grad=True)
    legal = torch.tensor([[False, True, True]])
    distribution = compute_voc_gate_distribution(
        raw_logits,
        temperature=1.0,
        epsilon=0.0,
        legal_control_mask=legal,
    )

    (-distribution.gate_logits[0, 0]).backward()

    assert raw_logits.grad[0, util.PROCEED].item() == 0.0
    torch.testing.assert_close(
        raw_logits.grad[0, util.RESET],
        -raw_logits.grad[0, util.STOP],
    )


def test_voc_shadow_is_behaviorally_bitwise_and_q_is_encoder_detached():
    torch.manual_seed(101)
    off_actor = _network(_flags(True, factorized=True, voc_mode="off"))
    off_rng_continuation = torch.rand(8)
    torch.manual_seed(101)
    shadow_actor = _network(_flags(
        True, factorized=True, voc_mode="shadow"
    ))
    shadow_rng_continuation = torch.rand(8)
    assert torch.equal(off_rng_continuation, shadow_rng_continuation)

    off_state_dict = off_actor.state_dict()
    shadow_state_dict = shadow_actor.state_dict()
    assert set(shadow_state_dict) - set(off_state_dict) == {
        "voc_head.weight",
        "voc_head.bias",
    }
    for key, value in off_state_dict.items():
        assert torch.equal(value, shadow_state_dict[key]), key

    flags = _flags(True, factorized=True, voc_mode="shadow")
    env_out = _env_out(
        flags,
        [util.SEARCH_PHASE, util.NEED_REAL_ACTION_PHASE, util.WAIT_PHASE],
        reset_mask=torch.ones(3, dtype=torch.bool),
    )
    clamp_action = (
        torch.tensor([[[1], [2], [3]]]),
        torch.tensor([[util.PROCEED, util.PROCEED, util.PROCEED]]),
    )
    torch.manual_seed(202)
    off_out, off_state = off_actor(
        env_out,
        off_actor.initial_state(3),
        clamp_action=clamp_action,
        compute_loss=True,
    )
    torch.manual_seed(202)
    shadow_out, shadow_state = shadow_actor(
        env_out,
        shadow_actor.initial_state(3),
        clamp_action=clamp_action,
        compute_loss=True,
    )

    assert off_out.voc_q is None
    assert shadow_out.voc_q.shape == (1, 3, 2)
    assert shadow_out.voc_features.shape[:2] == (1, 3)
    assert not shadow_out.voc_features.requires_grad
    for field in (
        "pri",
        "pri_param",
        "reset",
        "reset_logits",
        "c_action_log_prob",
        "baseline",
        "entropy_loss",
        "reg_loss",
        "search_control",
        "search_control_logits",
        "primary_valid",
        "control_valid",
        "policy_valid",
        "policy_type",
    ):
        assert torch.equal(getattr(off_out, field), getattr(shadow_out, field)), field
    for off_value, shadow_value in zip(off_state, shadow_state):
        assert torch.equal(off_value, shadow_value)

    q_bias_grad, shared_grad, gate_grad = torch.autograd.grad(
        shadow_out.voc_q.sum(),
        (
            shadow_actor.voc_head.bias,
            shadow_actor.phase_embedding.weight,
            shadow_actor.reset.weight,
        ),
        allow_unused=True,
    )
    assert torch.count_nonzero(q_bias_grad) == 2
    assert shared_grad is None
    assert gate_grad is None

    torch.manual_seed(909)
    off_behavior, _ = off_actor(
        env_out, off_actor.initial_state(3), compute_loss=False
    )
    off_rng_after = torch.rand(8)
    torch.manual_seed(909)
    shadow_behavior, _ = shadow_actor(
        env_out, shadow_actor.initial_state(3), compute_loss=False
    )
    shadow_rng_after = torch.rand(8)
    assert shadow_behavior.voc_features is None
    assert torch.equal(off_behavior.pri, shadow_behavior.pri)
    assert torch.equal(
        off_behavior.search_control, shadow_behavior.search_control
    )
    assert torch.equal(
        off_behavior.search_control_logits,
        shadow_behavior.search_control_logits,
    )
    assert torch.equal(off_rng_after, shadow_rng_after)


def test_dedicated_voc_gate_is_zero_neutral_detached_and_entropy_metric_only():
    torch.manual_seed(1001)
    flags = _flags(
        True,
        factorized=True,
        voc_mode="control",
        voc_train_epsilon=0.0,
        voc_dedicated_gate=True,
    )
    actor = _network(flags)
    env_out = _env_out(flags, [util.SEARCH_PHASE] * 3)
    out, _ = actor(
        env_out,
        actor.initial_state(3),
        clamp_action=(
            torch.tensor([[[0], [1], [2]]]),
            torch.tensor([[util.PROCEED, util.RESET, util.STOP]]),
        ),
        compute_loss=True,
    )

    assert actor.voc_gate_head.out_features == 1
    assert torch.count_nonzero(actor.voc_gate_head.weight) == 0
    assert torch.count_nonzero(actor.voc_gate_head.bias) == 0
    raw_log_odds = out.misc["voc_gate_log_odds"]
    assert raw_log_odds.shape == (1, 3)
    assert raw_log_odds.requires_grad
    joint = torch.softmax(out.search_control_logits, dim=-1)
    torch.testing.assert_close(
        joint[..., :2].sum(dim=-1), torch.full((1, 3), 0.5)
    )
    torch.testing.assert_close(joint[..., util.STOP], torch.full((1, 3), 0.5))

    raw_entropy = out.misc["voc_gate_entropy"]
    assert raw_entropy.shape == (1, 3)
    assert not raw_entropy.requires_grad
    torch.testing.assert_close(
        raw_entropy, torch.full((1, 3), np.log(2.0), dtype=raw_entropy.dtype)
    )

    head_grad, shared_grad, bout_grad = torch.autograd.grad(
        raw_log_odds.sum(),
        (
            actor.voc_gate_head.bias,
            actor.phase_embedding.weight,
            actor.reset.weight,
        ),
        retain_graph=True,
        allow_unused=True,
    )
    torch.testing.assert_close(head_grad, torch.tensor([3.0]))
    assert shared_grad is None
    assert bout_grad is None

    entropy_gate_grad = torch.autograd.grad(
        out.entropy_loss.sum(),
        actor.voc_gate_head.bias,
        allow_unused=True,
    )[0]
    assert entropy_gate_grad is None or torch.allclose(
        entropy_gate_grad, torch.zeros_like(entropy_gate_grad), atol=1e-7
    )


def test_schema5_actor_separates_soft_calibration_from_behavior_vtrace_logits():
    flags = _flags(
        True,
        factorized=True,
        voc_mode="control",
        voc_train_epsilon=0.02,
        voc_dedicated_gate=True,
        voc_gate_exact_projection=True,
        voc_gate_epsilon_greedy_execution=True,
    )
    actor = _network(flags)
    with torch.no_grad():
        actor.voc_gate_head.weight.zero_()
        actor.voc_gate_head.bias.fill_(1.0)
    env_out = _env_out(flags, [util.SEARCH_PHASE] * 3)

    torch.manual_seed(8101)
    out, _ = actor(
        env_out,
        actor.initial_state(3),
        compute_loss=True,
        greedy=False,
    )

    behavior_probability = torch.softmax(
        out.search_control_logits, dim=-1
    )[..., :2].sum(dim=-1)
    expected_behavior = torch.full_like(behavior_probability, 0.99)
    torch.testing.assert_close(behavior_probability, expected_behavior)
    assert torch.equal(
        out.misc["voc_gate_execution_continue_probability"],
        expected_behavior,
    )
    expected_soft = 0.98 * torch.sigmoid(torch.ones_like(behavior_probability))
    expected_soft = expected_soft + 0.01
    torch.testing.assert_close(
        out.misc["voc_gate_soft_continue_probability"], expected_soft
    )
    soft_from_logits = torch.softmax(
        out.misc["voc_gate_soft_control_logits"], dim=-1
    )[..., :2].sum(dim=-1)
    torch.testing.assert_close(soft_from_logits, expected_soft)
    assert not torch.equal(behavior_probability, soft_from_logits)

    behavior_parts = compute_dynamic_control_log_probs(
        out.search_control_logits,
        out.search_control,
        out.control_valid,
        project_gate_gradient=False,
    )
    torch.testing.assert_close(
        behavior_parts.gate, out.misc["gate_log_prob"]
    )
    torch.testing.assert_close(
        out.c_action_log_prob,
        out.misc["primary_log_prob"] + out.misc["control_log_prob"],
    )


def test_schema5_soft_behavior_split_propagates_through_actor_net_sep():
    flags = _flags(
        True,
        factorized=True,
        voc_mode="control",
        voc_dedicated_gate=True,
        voc_gate_exact_projection=True,
        voc_gate_epsilon_greedy_execution=True,
    )
    flags.sep_actor_critic = True
    actor = _network(flags, batch_size=1)
    with torch.no_grad():
        actor.actor.voc_gate_head.weight.zero_()
        actor.actor.voc_gate_head.bias.fill_(-1.0)
    env_out = _env_out(flags, [util.SEARCH_PHASE])

    out, _ = actor(
        env_out,
        actor.initial_state(1),
        compute_loss=True,
        greedy=False,
    )

    assert "voc_gate_soft_control_logits" in out.misc
    behavior_probability = torch.softmax(
        out.search_control_logits, dim=-1
    )[..., :2].sum(dim=-1)
    torch.testing.assert_close(
        behavior_probability, torch.full_like(behavior_probability, 0.01)
    )
    assert not torch.equal(
        behavior_probability,
        out.misc["voc_gate_soft_continue_probability"],
    )


def test_schema5_fixed_eval_surface_is_finite_deterministic_and_keeps_soft_p():
    flags = _flags(
        True,
        factorized=True,
        voc_mode="control",
        voc_dedicated_gate=True,
        voc_gate_exact_projection=True,
        voc_gate_epsilon_greedy_execution=True,
        voc_eval_stochastic=True,
        train_actor=False,
    )
    actor = _network(flags, batch_size=1)
    with torch.no_grad():
        actor.voc_gate_head.weight.zero_()
        actor.voc_gate_head.bias.fill_(1.0)
    env_out = _env_out(flags, [util.SEARCH_PHASE])

    out, _ = actor(
        env_out,
        actor.initial_state(1),
        compute_loss=True,
        greedy=False,
    )

    assert torch.isfinite(out.search_control_logits).all()
    assert out.search_control_logits.abs().max().item() <= 1000.0
    behavior_probability = torch.softmax(
        out.search_control_logits, dim=-1
    )[..., :2].sum(dim=-1)
    assert torch.equal(behavior_probability, torch.ones_like(behavior_probability))
    torch.testing.assert_close(
        out.misc["voc_gate_soft_continue_probability"],
        torch.sigmoid(torch.ones_like(behavior_probability)),
    )
    assert out.search_control.item() != util.STOP


def test_v11_execution_flag_absent_and_false_are_behavior_and_rng_identical():
    absent_flags = _flags(
        True,
        factorized=True,
        voc_mode="control",
        voc_dedicated_gate=True,
        voc_gate_exact_projection=True,
        voc_gate_epsilon_greedy_execution=False,
    )
    delattr(absent_flags, "voc_gate_epsilon_greedy_execution")
    false_flags = _flags(
        True,
        factorized=True,
        voc_mode="control",
        voc_dedicated_gate=True,
        voc_gate_exact_projection=True,
        voc_gate_epsilon_greedy_execution=False,
    )
    torch.manual_seed(8201)
    absent_actor = _network(absent_flags)
    absent_construct_tail = torch.rand(8)
    torch.manual_seed(8201)
    false_actor = _network(false_flags)
    false_construct_tail = torch.rand(8)
    assert torch.equal(absent_construct_tail, false_construct_tail)
    for key, value in absent_actor.state_dict().items():
        assert torch.equal(value, false_actor.state_dict()[key]), key

    absent_env = _env_out(absent_flags, [util.SEARCH_PHASE] * 3)
    false_env = absent_env
    torch.manual_seed(8202)
    absent_out, _ = absent_actor(
        absent_env, absent_actor.initial_state(3), compute_loss=True
    )
    absent_tail = torch.rand(8)
    torch.manual_seed(8202)
    false_out, _ = false_actor(
        false_env, false_actor.initial_state(3), compute_loss=True
    )
    false_tail = torch.rand(8)
    assert torch.equal(absent_out.search_control, false_out.search_control)
    assert torch.equal(
        absent_out.search_control_logits, false_out.search_control_logits
    )
    assert torch.equal(absent_tail, false_tail)
    for key in (
        "voc_gate_soft_control_logits",
        "voc_gate_soft_continue_probability",
        "voc_gate_execution_continue_probability",
    ):
        assert key not in absent_out.misc
        assert key not in false_out.misc


def test_schema5_changes_only_gate_execution_not_primary_action_or_rng_budget():
    common = dict(
        factorized=True,
        voc_mode="control",
        voc_dedicated_gate=True,
        voc_gate_exact_projection=True,
    )
    torch.manual_seed(8251)
    soft_actor = _network(_flags(True, **common))
    torch.manual_seed(8251)
    execution_actor = _network(_flags(
        True, voc_gate_epsilon_greedy_execution=True, **common
    ))
    for key, value in soft_actor.state_dict().items():
        assert torch.equal(value, execution_actor.state_dict()[key]), key
    with torch.no_grad():
        soft_actor.voc_gate_head.bias.fill_(0.1)
        execution_actor.voc_gate_head.bias.fill_(0.1)
    env_out = _env_out(soft_actor.flags, [util.SEARCH_PHASE] * 3)

    torch.manual_seed(8252)
    soft_out, _ = soft_actor(
        env_out, soft_actor.initial_state(3), compute_loss=True
    )
    soft_tail = torch.rand(8)
    torch.manual_seed(8252)
    execution_out, _ = execution_actor(
        env_out, execution_actor.initial_state(3), compute_loss=True
    )
    execution_tail = torch.rand(8)

    assert torch.equal(soft_out.pri, execution_out.pri)
    assert torch.equal(soft_out.pri_param, execution_out.pri_param)
    assert torch.equal(soft_out.action[0], execution_out.action[0])
    assert torch.equal(soft_tail, execution_tail)


def test_schema6_execution_entropy_uses_detached_875_policy_weight_only():
    common = dict(
        factorized=True,
        voc_mode="control",
        voc_dedicated_gate=True,
        voc_gate_exact_projection=True,
        voc_gate_epsilon_greedy_execution=True,
    )
    schema5_flags = _flags(True, **common)
    schema6_flags = _flags(True, **common)
    schema6_flags.voc_actor_policy_version_barrier = True
    schema6_flags.voc_actor_policy_bundle_schema_version = 1
    schema6_flags.voc_actor_policy_barrier_timeout_s = 120.0
    schema6_flags.voc_actor_policy_ray_max_restarts = 0
    schema6_flags.voc_actor_policy_ray_max_task_retries = 0
    schema6_flags.actor_amp_init_scale = 32.0
    schema6_flags.voc_gate_execution_epsilon = 0.25
    schema6_flags.ppo_k = 1
    schema6_flags.self_play_n = 1
    schema6_flags.env_n = 16
    schema6_flags.actor_batch_size = 16

    torch.manual_seed(8271)
    schema5_actor = _network(schema5_flags)
    torch.manual_seed(8271)
    schema6_actor = _network(schema6_flags)
    for key, value in schema5_actor.state_dict().items():
        assert torch.equal(value, schema6_actor.state_dict()[key]), key
    with torch.no_grad():
        for actor in (schema5_actor, schema6_actor):
            actor.voc_gate_head.weight.zero_()
            actor.voc_gate_head.bias.fill_(1.0)
    env_out = _env_out(schema5_flags, [util.SEARCH_PHASE] * 3)
    schema5_out, _ = schema5_actor(
        env_out,
        schema5_actor.initial_state(3),
        compute_loss=True,
        greedy=False,
    )
    schema6_out, _ = schema6_actor(
        env_out,
        schema6_actor.initial_state(3),
        compute_loss=True,
        greedy=False,
    )

    torch.testing.assert_close(
        schema5_out.misc["non_stop_prob"],
        torch.full_like(schema5_out.misc["non_stop_prob"], 0.99),
        rtol=0.0,
        atol=1e-7,
    )
    torch.testing.assert_close(
        schema6_out.misc["non_stop_prob"],
        torch.full_like(schema6_out.misc["non_stop_prob"], 0.875),
        rtol=0.0,
        atol=1e-7,
    )
    assert torch.equal(
        schema5_out.misc["voc_gate_execution_continue_probability"],
        torch.full_like(
            schema5_out.misc["voc_gate_execution_continue_probability"],
            0.99,
        ),
    )
    assert torch.equal(
        schema6_out.misc["voc_gate_execution_continue_probability"],
        torch.full_like(
            schema6_out.misc["voc_gate_execution_continue_probability"],
            0.875,
        ),
    )
    torch.testing.assert_close(
        schema6_out.misc["primary_entropy_loss"],
        schema5_out.misc["primary_entropy_loss"] * (0.875 / 0.99),
    )
    torch.testing.assert_close(
        schema6_out.misc["bout_entropy_loss"],
        schema5_out.misc["bout_entropy_loss"] * (0.875 / 0.99),
    )
    assert torch.equal(schema5_out.pri_param, schema6_out.pri_param)
    assert torch.equal(
        schema5_out.misc["voc_gate_soft_control_logits"],
        schema6_out.misc["voc_gate_soft_control_logits"],
    )
    gate_grad = torch.autograd.grad(
        schema6_out.entropy_loss.sum(),
        tuple(schema6_actor.voc_gate_head.parameters()),
        allow_unused=True,
    )
    assert all(
        gradient is None or torch.count_nonzero(gradient) == 0
        for gradient in gate_grad
    )


def _versioned_actor_flags(gate_schema, seal_schema=None):
    flags = _flags(
        True,
        factorized=True,
        voc_mode="control",
        voc_dedicated_gate=True,
        voc_gate_exact_projection=True,
        voc_gate_epsilon_greedy_execution=True,
    )
    flags.voc_actor_policy_version_barrier = True
    flags.voc_actor_policy_bundle_schema_version = 1
    flags.voc_actor_policy_barrier_timeout_s = 120.0
    flags.voc_actor_policy_ray_max_restarts = 0
    flags.voc_actor_policy_ray_max_task_retries = 0
    flags.actor_amp_init_scale = 32.0
    flags.voc_gate_execution_epsilon = 0.25
    flags.ppo_k = 1
    flags.self_play_n = 1
    flags.env_n = 16
    flags.actor_batch_size = 16
    flags.voc_gate_policy_schema_version = gate_schema
    if gate_schema in (
        util.VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION,
        util.VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION,
    ):
        flags.voc_gate_target_tau = 1.0
    if seal_schema is not None:
        flags.voc_model_input_seal_schema_version = seal_schema
    return flags


def test_schema7_through_schema13_guards_preserve_actor_state_and_rng():
    schema6_flags = _versioned_actor_flags(6)
    schema7_flags = _versioned_actor_flags(7, 1)
    schema8_flags = _versioned_actor_flags(8, 1)
    schema9_flags = _versioned_actor_flags(9, 1)
    schema10_flags = _versioned_actor_flags(10, 1)
    schema11_flags = _versioned_actor_flags(
        util.VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION, 1
    )
    schema12_flags = _versioned_actor_flags(
        util.VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION, 1
    )
    schema13_flags = _versioned_actor_flags(
        util.VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION, 1
    )
    torch.manual_seed(8291)
    schema6_actor = _network(schema6_flags)
    schema6_tail = torch.rand(8)
    torch.manual_seed(8291)
    schema7_actor = _network(schema7_flags)
    schema7_tail = torch.rand(8)
    torch.manual_seed(8291)
    schema8_actor = _network(schema8_flags)
    schema8_tail = torch.rand(8)
    torch.manual_seed(8291)
    schema9_actor = _network(schema9_flags)
    schema9_tail = torch.rand(8)
    torch.manual_seed(8291)
    schema10_actor = _network(schema10_flags)
    schema10_tail = torch.rand(8)
    torch.manual_seed(8291)
    schema11_actor = _network(schema11_flags)
    schema11_tail = torch.rand(8)
    torch.manual_seed(8291)
    schema12_actor = _network(schema12_flags)
    schema12_tail = torch.rand(8)
    torch.manual_seed(8291)
    schema13_actor = _network(schema13_flags)
    schema13_tail = torch.rand(8)

    assert (
        schema6_actor.state_dict().keys()
        == schema7_actor.state_dict().keys()
        == schema8_actor.state_dict().keys()
        == schema9_actor.state_dict().keys()
        == schema10_actor.state_dict().keys()
        == schema11_actor.state_dict().keys()
        == schema12_actor.state_dict().keys()
        == schema13_actor.state_dict().keys()
    )
    for key, value in schema6_actor.state_dict().items():
        assert torch.equal(value, schema7_actor.state_dict()[key]), key
        assert torch.equal(value, schema8_actor.state_dict()[key]), key
        assert torch.equal(value, schema9_actor.state_dict()[key]), key
        assert torch.equal(value, schema10_actor.state_dict()[key]), key
        assert torch.equal(value, schema11_actor.state_dict()[key]), key
        assert torch.equal(value, schema12_actor.state_dict()[key]), key
        assert torch.equal(value, schema13_actor.state_dict()[key]), key
    assert torch.equal(schema6_tail, schema7_tail)
    assert torch.equal(schema6_tail, schema8_tail)
    assert torch.equal(schema6_tail, schema9_tail)
    assert torch.equal(schema6_tail, schema10_tail)
    assert torch.equal(schema6_tail, schema11_tail)
    assert torch.equal(schema6_tail, schema12_tail)
    assert torch.equal(schema6_tail, schema13_tail)


@pytest.mark.parametrize(
    ("gate_schema", "seal_schema"),
    [
        (7, 0),
        (7, True),
        (7, np.int64(1)),
        (8, 0),
        (8, True),
        (8, np.int64(1)),
        (9, 0),
        (9, True),
        (9, np.int64(1)),
        (10, 0),
        (10, True),
        (10, np.int64(1)),
        (util.VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION, 0),
        (util.VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION, True),
        (
            util.VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION,
            np.int64(1),
        ),
        (util.VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION, 0),
        (util.VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION, True),
        (util.VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION, np.int64(1)),
        (util.VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION, 0),
        (util.VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION, True),
        (
            util.VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION,
            np.int64(1),
        ),
        (6, 1),
        (6, False),
    ],
)
def test_versioned_actor_rejects_wrong_model_input_seal_schema(
    gate_schema, seal_schema
):
    flags = _versioned_actor_flags(gate_schema, seal_schema)
    with pytest.raises(ValueError, match="model_input_seal_schema_version"):
        _network(flags)


@pytest.mark.parametrize(
    "gate_schema",
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
        14,
    ],
)
def test_versioned_actor_rejects_noncanonical_or_unknown_schema_identity(
    gate_schema,
):
    flags = _versioned_actor_flags(gate_schema, 1)
    with pytest.raises(ValueError, match="exact integer gate schema"):
        _network(flags)


def test_dedicated_shadow_and_explicit_false_preserve_legacy_behavior_and_rng():
    # An explicit false flag is identical to the pre-v4 absent-flag path.
    torch.manual_seed(1101)
    absent_flags = _flags(True, factorized=True, voc_mode="control")
    delattr(absent_flags, "voc_dedicated_gate")
    absent = _network(absent_flags)
    absent_rng = torch.rand(8)
    torch.manual_seed(1101)
    disabled = _network(_flags(
        True,
        factorized=True,
        voc_mode="control",
        voc_dedicated_gate=False,
    ))
    disabled_rng = torch.rand(8)
    assert absent.state_dict().keys() == disabled.state_dict().keys()
    for key, value in absent.state_dict().items():
        assert torch.equal(value, disabled.state_dict()[key]), key
    assert torch.equal(absent_rng, disabled_rng)

    # Shadow owns the zero head for checkpoint compatibility but never routes
    # actions through it and does not advance construction/sampling RNG.
    torch.manual_seed(1201)
    off = _network(_flags(
        True,
        factorized=True,
        voc_mode="off",
        voc_dedicated_gate=True,
    ))
    off_rng = torch.rand(8)
    torch.manual_seed(1201)
    shadow = _network(_flags(
        True,
        factorized=True,
        voc_mode="shadow",
        voc_dedicated_gate=True,
    ))
    shadow_rng = torch.rand(8)
    assert set(shadow.state_dict()) - set(off.state_dict()) == {
        "voc_head.weight",
        "voc_head.bias",
        "voc_gate_head.weight",
        "voc_gate_head.bias",
    }
    for key, value in off.state_dict().items():
        assert torch.equal(value, shadow.state_dict()[key]), key
    assert torch.equal(off_rng, shadow_rng)

    flags = _flags(True, factorized=True, voc_mode="off")
    env_out = _env_out(flags, [util.SEARCH_PHASE] * 3)
    torch.manual_seed(1301)
    off_out, _ = off(env_out, off.initial_state(3), compute_loss=True)
    off_after = torch.rand(8)
    torch.manual_seed(1301)
    shadow_out, _ = shadow(
        env_out, shadow.initial_state(3), compute_loss=True
    )
    shadow_after = torch.rand(8)
    for field in (
        "pri",
        "pri_param",
        "search_control",
        "search_control_logits",
        "c_action_log_prob",
        "entropy_loss",
        "reg_loss",
    ):
        assert torch.equal(getattr(off_out, field), getattr(shadow_out, field))
    assert "voc_gate_log_odds" not in shadow_out.misc
    assert "voc_gate_entropy" not in shadow_out.misc
    assert torch.equal(off_after, shadow_after)


def test_voc_control_q_values_do_not_select_or_change_policy_actions():
    torch.manual_seed(303)
    flags = _flags(
        True,
        factorized=True,
        voc_mode="control",
        voc_train_epsilon=0.0,
        voc_eval_stochastic=True,
        voc_dedicated_gate=True,
    )
    actor = _network(flags, batch_size=4)
    actor.eval()
    env_out = _env_out(flags, [util.SEARCH_PHASE] * 4)
    initial_state = actor.initial_state(4)

    with torch.no_grad():
        actor.voc_head.bias.copy_(torch.tensor([1e6, -1e6]))
    torch.manual_seed(404)
    continue_q_out, _ = actor(
        env_out, initial_state, greedy=True, compute_loss=True
    )

    with torch.no_grad():
        actor.voc_head.bias.copy_(torch.tensor([-1e6, 1e6]))
    torch.manual_seed(404)
    stop_q_out, _ = actor(
        env_out, initial_state, greedy=True, compute_loss=True
    )

    assert torch.equal(
        continue_q_out.search_control, stop_q_out.search_control
    )
    assert torch.equal(
        continue_q_out.search_control_logits,
        stop_q_out.search_control_logits,
    )
    assert torch.equal(continue_q_out.pri, stop_q_out.pri)
    assert not torch.equal(continue_q_out.voc_q, stop_q_out.voc_q)


def test_voc_training_epsilon_is_disabled_for_stochastic_self_play_eval():
    common = dict(
        dynamic=True,
        factorized=True,
        voc_mode="control",
        voc_train_epsilon=0.4,
        voc_eval_stochastic=True,
        voc_dedicated_gate=True,
    )
    torch.manual_seed(707)
    training_actor = _network(_flags(**common, train_actor=True))
    evaluation_actor = _network(_flags(**common, train_actor=False))
    no_epsilon_actor = _network(
        _flags(**{**common, "voc_train_epsilon": 0.0}, train_actor=True)
    )
    with torch.no_grad():
        # Zero log-odds is already uniform, so epsilon would be an exact
        # no-op.  Use a non-neutral learned gate to exercise its removal.
        training_actor.voc_gate_head.bias.fill_(2.0)
    evaluation_actor.load_state_dict(training_actor.state_dict())
    no_epsilon_actor.load_state_dict(training_actor.state_dict())

    env_out = _env_out(
        _flags(**common, train_actor=True), [util.SEARCH_PHASE] * 3
    )
    state = training_actor.initial_state(3)
    torch.manual_seed(808)
    training_out, _ = training_actor(env_out, state, greedy=False)
    torch.manual_seed(808)
    evaluation_out, _ = evaluation_actor(env_out, state, greedy=False)
    torch.manual_seed(808)
    no_epsilon_out, _ = no_epsilon_actor(env_out, state, greedy=False)

    # Standard SelfPlayWorker evaluation remains stochastic (`greedy=False`),
    # but its run-level train_actor=False must remove training-only epsilon.
    torch.testing.assert_close(
        evaluation_out.search_control_logits,
        no_epsilon_out.search_control_logits,
    )
    assert not torch.equal(
        training_out.search_control_logits,
        evaluation_out.search_control_logits,
    )


def test_separate_actor_critic_forwards_only_the_critic_voc_head():
    flags = _flags(True, factorized=True, voc_mode="shadow")
    flags.sep_actor_critic = True
    actor = _network(flags, batch_size=2)
    env_out = _env_out(flags, [util.SEARCH_PHASE] * 2)

    out, _ = actor(env_out, actor.initial_state(2))

    assert out.voc_q.shape == (1, 2, 2)
    assert not hasattr(actor.actor, "voc_head")
    assert hasattr(actor.critic, "voc_head")
    assert "critic.voc_head.weight" in actor.state_dict()
    assert "voc_head.weight" not in actor.state_dict()

    q_grad, shared_grad = torch.autograd.grad(
        out.voc_q.sum(),
        (actor.critic.voc_head.bias, actor.critic.phase_embedding.weight),
        allow_unused=True,
    )
    assert torch.count_nonzero(q_grad) == 2
    assert shared_grad is None


def test_legacy_preload_preserves_policy_reset_rows_and_critic_rows():
    torch.manual_seed(17)
    fixed = _network(_flags(False))
    dynamic = _network(_flags(True))

    fixed_state = fixed.state_dict()
    dynamic.set_weights(fixed_state, strict=False)
    migrated = dynamic.state_dict()

    torch.testing.assert_close(migrated["policy.weight"], fixed_state["policy.weight"])
    torch.testing.assert_close(migrated["im_policy.weight"], fixed_state["im_policy.weight"])
    torch.testing.assert_close(migrated["reset.weight"][:2], fixed_state["reset.weight"])
    assert torch.count_nonzero(migrated["reset.weight"][2]) == 0
    torch.testing.assert_close(migrated["baseline.weight"][:2], fixed_state["baseline.weight"])
    assert torch.count_nonzero(migrated["baseline.weight"][2]) == 0


def test_dynamic_resume_rejects_legacy_shapes_in_strict_mode():
    fixed = _network(_flags(False))
    dynamic = _network(_flags(True))

    with pytest.raises(RuntimeError):
        dynamic.set_weights(fixed.state_dict(), strict=True)
