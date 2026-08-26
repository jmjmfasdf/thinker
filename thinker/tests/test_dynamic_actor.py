import numpy as np
import pytest
import torch
from gymnasium import spaces

from thinker import util
from thinker.actor_net import ActorNet, ILLEGAL_CONTROL_LOGIT


def _flags(
    dynamic,
    *,
    max_search_steps=-1,
    max_depth=40,
    rec_t=40,
    factorized=False,
):
    return util.create_setting(
        args=[
            "--dynamic_search", str(dynamic).lower(),
            "--dynamic_factorized_control", str(factorized).lower(),
            "--wrapper_type", "0",
            "--max_search_steps", str(max_search_steps),
            "--max_depth", str(max_depth),
            "--rec_t", str(rec_t),
            "--see_real_state", "false",
            "--see_h", "false",
            "--see_x", "false",
            "--parallel", "false",
        ],
        save_flags=False,
    )


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
