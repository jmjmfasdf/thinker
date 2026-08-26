from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.test_dynamic_cenv import FakeModel, FakeVectorEnv
from tests.test_dynamic_cenv import _flags as cenv_flags
from thinker import util
from thinker.actor_net import (
    ActorNet,
    ILLEGAL_CONTROL_LOGIT,
    compute_dynamic_control_entropy,
)
from thinker.cenv import cModelWrapper, cPerfectWrapper
from thinker.self_play import TrainActorOut
from thinker.util import EnvOut
import thinker.learn_actor as learn_actor_module


def test_useful_carry_observability_does_not_widen_actor_replay_schema():
    telemetry = {
        "carried_descendant_visit_count",
        "carried_descendant_expanded_count",
        "useful_carry",
    }
    assert telemetry.issubset(EnvOut._fields)
    assert telemetry.isdisjoint(TrainActorOut._fields)


class _NullWriter:
    def log(self, *_args, **_kwargs):
        pass

    def close(self, *_args, **_kwargs):
        pass


def _expected_absmax_rms(value, mask=None):
    if value is None:
        return 0.0, 0.0
    selected = value.detach().double()
    if mask is not None:
        selected = selected[mask]
    if selected.numel() == 0:
        return 0.0, 0.0
    return (
        selected.abs().max().item(),
        torch.sqrt(torch.mean(selected.square())).item(),
    )


def test_dynamic_actor_observability_is_phase_masked_and_extreme_finite():
    policy_type = torch.tensor([
        [util.POLICY_REAL, util.POLICY_SEARCH, util.POLICY_NONE],
        [util.POLICY_SEARCH, util.POLICY_REAL, util.POLICY_SEARCH],
    ])
    policy_valid = policy_type != util.POLICY_NONE
    primary_valid = policy_valid.clone()
    primary_valid[1, 0] = False  # a learned STOP has no primary action
    control_valid = policy_type == util.POLICY_SEARCH

    primary_logits = torch.tensor(
        [
            [
                [[1.0e20, -1.0e20, 0.0]],
                [[-5.0e19, 2.0, 3.0]],
                [[3.0e30, 3.0e30, 3.0e30]],
            ],
            [
                [[4.0e30, 4.0e30, 4.0e30]],
                [[3.0, 4.0, 5.0]],
                [[7.0, 8.0, 9.0]],
            ],
        ],
        dtype=torch.float32,
    )
    control_logits = torch.tensor(
        [
            [
                [6.0e30, 6.0e30, 6.0e30],
                [ILLEGAL_CONTROL_LOGIT, 2.0, -3.0],
                [7.0e30] * 3,
            ],
            [[-4.0, 2.0, 1.0], [8.0e30] * 3, [7.0, 8.0, 9.0]],
        ],
        dtype=torch.float32,
    )
    actor_out = SimpleNamespace(
        hs=torch.tensor([1.0e20, -2.0e20]),
        tree_reps=torch.tensor([[3.0e20, 4.0e20]]),
        xs=torch.tensor([[-5.0e20, 0.0, 5.0e20]]),
        policy_type=policy_type,
        policy_valid=policy_valid,
        primary_valid=primary_valid,
        pri_param=primary_logits,
        control_valid=control_valid,
        search_control_logits=control_logits,
        reset_logits=None,
    )

    stats = learn_actor_module.dynamic_actor_observability_stats(
        actor_out, discrete_action=True
    )

    sources = {
        "env_hs": (actor_out.hs, None),
        "env_tree_reps": (actor_out.tree_reps, None),
        "env_xs": (actor_out.xs, None),
        "real_primary_logits": (
            primary_logits,
            policy_valid & (policy_type == util.POLICY_REAL),
        ),
        "search_primary_logits": (
            primary_logits,
            primary_valid & (policy_type == util.POLICY_SEARCH),
        ),
        "search_control_logits": (
            control_logits,
            control_valid.unsqueeze(-1)
            & control_logits.ne(ILLEGAL_CONTROL_LOGIT),
        ),
    }
    for name, (value, mask) in sources.items():
        expected_absmax, expected_rms = _expected_absmax_rms(value, mask)
        assert stats[f"actor/{name}_absmax"] == pytest.approx(
            expected_absmax
        )
        assert stats[f"actor/{name}_rms"] == pytest.approx(expected_rms)
        assert np.isfinite(stats[f"actor/{name}_absmax"])
        assert np.isfinite(stats[f"actor/{name}_rms"])

    empty = SimpleNamespace(
        hs=None,
        tree_reps=None,
        xs=None,
        policy_type=None,
        policy_valid=None,
        primary_valid=None,
        pri_param=None,
        control_valid=None,
        search_control_logits=None,
        reset_logits=None,
    )
    empty_stats = learn_actor_module.dynamic_actor_observability_stats(
        empty, discrete_action=True
    )
    assert set(empty_stats) == set(stats)
    assert all(value == 0.0 for value in empty_stats.values())

    sentinel_only = SimpleNamespace(
        **{
            **vars(actor_out),
            "control_valid": torch.ones((1, 1), dtype=torch.bool),
            "search_control_logits": torch.full(
                (1, 1, 3), ILLEGAL_CONTROL_LOGIT
            ),
        }
    )
    sentinel_stats = learn_actor_module.dynamic_actor_observability_stats(
        sentinel_only, discrete_action=True
    )
    assert sentinel_stats["actor/search_control_logits_absmax"] == 0.0
    assert sentinel_stats["actor/search_control_logits_rms"] == 0.0

    # Legacy reset logits are unmasked values, so a sentinel-like raw value
    # must not be silently discarded when search_control_logits is absent.
    legacy = SimpleNamespace(
        **{
            **vars(actor_out),
            "control_valid": torch.ones((1, 1), dtype=torch.bool),
            "search_control_logits": None,
            "reset_logits": torch.tensor(
                [[[ILLEGAL_CONTROL_LOGIT, 2.0]]]
            ),
        }
    )
    legacy_stats = learn_actor_module.dynamic_actor_observability_stats(
        legacy, discrete_action=True
    )
    assert legacy_stats["actor/search_control_logits_absmax"] == abs(
        ILLEGAL_CONTROL_LOGIT
    )


def _stack_train_rows(rows, batch_size):
    data = {}
    for field in TrainActorOut._fields:
        if field == "id":
            data[field] = torch.arange(batch_size).unsqueeze(0)
            continue
        values = [
            getattr(env_out if field in EnvOut._fields else actor_out, field)
            for env_out, actor_out in rows
        ]
        if values[0] is None:
            data[field] = None
        else:
            data[field] = torch.cat(
                [value.detach().clone() for value in values], dim=0
            )
    return TrainActorOut(**data)


def _rollout(flags):
    """Collect one overlapped learner unroll with STOP, cap and WAIT rows."""
    env = FakeVectorEnv()
    model = FakeModel(env.num_actions)
    wrapper_cls = cPerfectWrapper if flags.wrapper_type == 2 else cModelWrapper
    wrapper = wrapper_cls(
        env, env.env_n, flags, model, device=torch.device("cpu")
    )
    state, info = wrapper.reset(model)
    batch_size = wrapper.env_n
    actor = ActorNet(
        obs_space=wrapper.observation_space,
        action_space=wrapper.action_space,
        flags=flags,
        tree_rep_meaning=util.get_tree_rep_meaning(
            model.num_actions, 1, flags
        ),
    )
    actor.train(False)

    env_out = util.init_env_out(
        state, info, flags, dim_actions=1, tuple_action=False
    )
    actor_state = actor.initial_state(batch_size, device=torch.device("cpu"))
    effective_primary = torch.zeros(batch_size, dtype=torch.long)
    effective_control = torch.zeros(batch_size, dtype=torch.long)
    im_return = torch.zeros(batch_size)
    think_return = torch.zeros(batch_size)

    # Row zero is the overlap/bootstrap row and is deliberately discarded by
    # SActorLearner.  Thereafter env 0 stops first, env 1 stops second, and env
    # 2 reaches the cap.  This creates both a non-real WAIT row and a full-batch
    # barrier-release row before all environments STOP once more.
    actions = [
        ([0, 1, 2], [util.PROCEED, util.PROCEED, util.PROCEED]),
        ([1, 1, 1], [util.STOP, util.PROCEED, util.PROCEED]),
        ([4, 1, 1], [util.PROCEED, util.STOP, util.PROCEED]),
        ([0, 3, 1], [util.PROCEED, util.PROCEED, util.PROCEED]),
        ([0, 0, 2], [util.PROCEED, util.PROCEED, util.PROCEED]),
        ([1, 1, 1], [util.STOP, util.STOP, util.STOP]),
    ]

    rows = []
    initial_actor_state = None
    for row, (primary, control) in enumerate(actions):
        clamp_primary = torch.tensor(primary).view(1, batch_size, 1)
        clamp_control = torch.tensor(control).view(1, batch_size)
        with torch.no_grad():
            actor_out, actor_state = actor(
                env_out,
                actor_state,
                clamp_action=(clamp_primary, clamp_control),
            )
            state, reward, done, truncated, info = wrapper.step(
                actor_out.action, model
            )

        accepted_primary = info["accepted_primary_action"]
        accepted_primary_mask = accepted_primary >= 0
        effective_primary[accepted_primary_mask] = accepted_primary[
            accepted_primary_mask
        ]
        accepted_control = info["accepted_control"]
        accepted_control_mask = accepted_control >= 0
        effective_control[accepted_control_mask] = accepted_control[
            accepted_control_mask
        ]
        real_transition = info["real_transition"].bool()
        if torch.any(real_transition):
            effective_primary[real_transition] = info[
                "executed_primary_action"
            ][real_transition]
        info["effective_primary_action"] = effective_primary.clone()
        info["effective_search_control"] = effective_control.clone()

        im_return += info["im_reward"][:, 0]
        think_return += info["think_reward"]
        info["im_episode_return"] = im_return.clone()
        info["think_episode_return"] = think_return.clone()
        stage_end = info["stage_end"].bool()
        im_return[stage_end] = 0
        think_return[stage_end] = 0

        env_out = util.create_env_out(
            actor_out.action,
            state,
            reward,
            done,
            truncated,
            info,
            flags,
        )
        rows.append((env_out, actor_out))
        if row == 0:
            initial_actor_state = tuple(
                value.detach().clone() for value in actor_state
            )

    train_out = _stack_train_rows(rows, batch_size)
    return actor, train_out, initial_actor_state


@pytest.mark.parametrize("wrapper_type", [0, 2], ids=["learned", "perfect"])
@pytest.mark.parametrize("ppo_k", [1, 2], ids=["vtrace", "ppo"])
@pytest.mark.parametrize(
    "factorized", [False, True], ids=["legacy-control", "factorized-control"]
)
def test_dynamic_cap_rollout_backpropagates_through_stop_and_think_cost(
        tmp_path, monkeypatch, ppo_k, wrapper_type, factorized):
    torch.manual_seed(23)
    flags = cenv_flags(cap=4, wrapper_type=wrapper_type)
    flags.float16 = False
    flags.see_real_state = False
    flags.dynamic_factorized_control = factorized
    flags.ppo_k = ppo_k
    flags.ppo_kl_coef = 0.01 if ppo_k > 1 else 0.0
    flags.return_norm_type = -1
    flags.parallel_actor = False
    flags.actor_batch_size = 3
    flags.env_n = 3
    flags.self_play_n = 1
    flags.total_steps = 100
    flags.ckp = False
    flags.savedir = str(tmp_path)
    flags.xpid = "dynamic-learner-%d-%d" % (ppo_k, factorized)
    flags.ckpdir = str(Path(tmp_path) / flags.xpid)

    actor, train_out, initial_actor_state = _rollout(flags)

    # Move the target policy off the behavior policy so the V-trace rho and
    # PPO ratio paths are exercised rather than degenerating to all ones.
    with torch.no_grad():
        if factorized:
            actor.reset.bias[util.STOP].add_(0.11)
        else:
            actor.reset.bias.add_(torch.tensor([0.08, -0.05, 0.11]))

    seen_vtrace_inputs = []
    original_compute_v_trace = learn_actor_module.compute_v_trace

    def capture_vtrace(**kwargs):
        seen_vtrace_inputs.append({
            "log_rhos": kwargs["log_rhos"].detach().clone(),
            "discounts": kwargs["discounts"].detach().clone(),
        })
        return original_compute_v_trace(**kwargs)

    monkeypatch.setattr(learn_actor_module, "compute_v_trace", capture_vtrace)
    monkeypatch.setattr(
        learn_actor_module, "FileWriter", lambda **_kwargs: _NullWriter()
    )
    learner = learn_actor_module.SActorLearner(
        ray_obj=None,
        actor_param={},
        flags=flags,
        actor_net=actor,
        device=torch.device("cpu"),
    )

    imitation_observability = None
    if wrapper_type == 0 and ppo_k == 1 and factorized:
        imitation_observability = {
            "nll_max": 11.0,
            "nll_p99": 10.5,
            "target_vs_best_other_logit_gap_max": 7.0,
            "target_vs_best_other_logit_gap_p99": 6.5,
            "scored_logits_absmax": 101.0,
            "scored_logits_rms": 31.0,
        }
        detached_metrics = {
            "accuracy": 0.25,
            "sampled_accuracy": 0.125,
            "root_carried_rate": 0.5,
            **imitation_observability,
        }
        fake_imitation_result = SimpleNamespace(
            loss=torch.tensor(0.25),
            normalized_ce=torch.tensor(0.2),
            margin_loss=torch.tensor(0.1),
            pvp_loss=torch.tensor(0.0),
            nll_sum=torch.tensor(6.0),
            count=4,
            detached_metrics=lambda: detached_metrics,
        )
        learner._maybe_compute_imitation = lambda: fake_imitation_result

    losses, shifted = learner.compute_losses(
        train_out, initial_actor_state
    )
    assert torch.isfinite(losses["total_loss"])
    for value in losses.values():
        if torch.is_tensor(value):
            assert torch.isfinite(value).all()
        elif isinstance(value, (float, np.floating)):
            assert np.isfinite(value)
    if imitation_observability is not None:
        for name, expected in imitation_observability.items():
            assert losses[f"icopro_{name}"].item() == pytest.approx(expected)

    wait = shifted.policy_type == util.POLICY_NONE
    nonreal_wait = wait & ~shifted.real_transition.bool()
    assert nonreal_wait.any(), "synthetic rollout must contain a WAIT-only row"
    assert not shifted.policy_valid[wait].any()
    assert torch.count_nonzero(shifted.c_action_log_prob[wait]) == 0
    assert len(seen_vtrace_inputs) == len(util.get_reward_names(flags))
    for vtrace_input in seen_vtrace_inputs:
        assert torch.count_nonzero(vtrace_input["log_rhos"][wait]) == 0
    if factorized:
        reward_names = util.get_reward_names(flags)
        rho_by_prefix = {
            prefix: seen_vtrace_inputs[index]["log_rhos"]
            for index, prefix in enumerate(reward_names)
        }
        # A STOP-only target perturbation changes the gate/full joint but is
        # invisible to the conditional imaginary channel.
        torch.testing.assert_close(
            rho_by_prefix["im"],
            torch.zeros_like(rho_by_prefix["im"]),
            rtol=0.0,
            atol=1e-6,
        )
        search = shifted.policy_type == util.POLICY_SEARCH
        torch.testing.assert_close(
            rho_by_prefix["re"][search], rho_by_prefix["think"][search]
        )
        cap_decision = shifted.forced_stop.bool() & shifted.stage_end.bool()
        assert torch.count_nonzero(rho_by_prefix["think"][cap_decision]) > 0
        assert torch.count_nonzero(rho_by_prefix["re"][cap_decision]) > 0
        torch.testing.assert_close(
            rho_by_prefix["im"][cap_decision],
            torch.zeros_like(rho_by_prefix["im"][cap_decision]),
            rtol=0.0,
            atol=1e-6,
        )
    torch.testing.assert_close(
        seen_vtrace_inputs[0]["discounts"][nonreal_wait],
        torch.ones_like(seen_vtrace_inputs[0]["discounts"][nonreal_wait]),
    )
    barrier_release = shifted.real_transition.bool()
    torch.testing.assert_close(
        seen_vtrace_inputs[0]["discounts"][barrier_release],
        torch.full_like(
            seen_vtrace_inputs[0]["discounts"][barrier_release],
            flags.discounting,
        ),
    )

    # Main-task reward crosses STOP -> NEED_REAL -> WAIT -> barrier release.
    # The independent think channel penalizes accepted computation actions.
    main_control_grad = torch.autograd.grad(
        losses["pg_loss"], actor.reset.weight, retain_graph=True
    )[0]
    think_control_grad = torch.autograd.grad(
        losses["think_pg_loss"], actor.reset.weight, retain_graph=True
    )[0]
    assert torch.isfinite(main_control_grad).all()
    assert torch.isfinite(think_control_grad).all()
    assert torch.count_nonzero(main_control_grad[util.STOP]) > 0
    assert torch.count_nonzero(think_control_grad[util.STOP]) > 0
    if factorized:
        torch.testing.assert_close(
            think_control_grad[util.PROCEED],
            think_control_grad[util.RESET],
            rtol=1e-6,
            atol=1e-7,
        )
        im_control_grad = torch.autograd.grad(
            losses["im_pg_loss"], actor.reset.weight, retain_graph=True
        )[0]
        # Imaginary shaping learns only the conditional bout. It cannot buy
        # CONTINUE by changing the STOP row of the shared three-way head.
        assert torch.count_nonzero(im_control_grad[util.STOP]) == 0
        assert torch.count_nonzero(im_control_grad[:2]) > 0
        torch.testing.assert_close(
            im_control_grad[util.PROCEED] + im_control_grad[util.RESET],
            torch.zeros_like(im_control_grad[util.PROCEED]),
            rtol=1e-6,
            atol=1e-6,
        )
        im_primary_grad = torch.autograd.grad(
            losses["im_pg_loss"],
            actor.im_policy.weight,
            retain_graph=True,
        )[0]
        assert torch.count_nonzero(im_primary_grad) > 0
        think_primary_grad = torch.autograd.grad(
            losses["think_pg_loss"],
            actor.im_policy.weight,
            retain_graph=True,
            allow_unused=True,
        )[0]
        assert think_primary_grad is None or torch.count_nonzero(
            think_primary_grad
        ) == 0

    learner.optimizer.zero_grad()
    losses["total_loss"].backward()
    control_grad = actor.reset.weight.grad
    assert control_grad is not None
    assert torch.isfinite(control_grad).all()
    assert torch.count_nonzero(control_grad[util.STOP]) > 0
    if ppo_k > 1:
        assert len(learner.ppo_is_abs) > 0
        assert np.isfinite(np.asarray(learner.ppo_is_abs)).all()

    stat_losses = {}
    if imitation_observability is not None:
        stat_losses = {
            f"icopro_{name}": losses[f"icopro_{name}"]
            for name in imitation_observability
        }
    stats = learner.compute_stat(
        shifted, losses=stat_losses, total_norm=0.0, actor_id=train_out.id
    )
    if imitation_observability is not None:
        for name, expected in imitation_observability.items():
            assert stats[f"actor/icopro_{name}"] == pytest.approx(expected)
    ended_steps = shifted.search_steps[shifted.stage_end.bool()].float()
    assert ended_steps.tolist() == [1.0, 2.0, 4.0, 0.0, 0.0, 0.0]
    assert stats["max_budget"] == 4.0
    assert stats["mean_budget"] == pytest.approx(7.0 / 6.0)
    assert stats["search/budget_bin_0_count"] == 3
    assert stats["search/budget_bin_0_fraction"] == pytest.approx(0.5)
    assert stats["search/budget_bin_1_count"] == 1
    assert stats["search/budget_bin_1_fraction"] == pytest.approx(1.0 / 6.0)
    assert stats["search/budget_bin_2_3_count"] == 1
    assert stats["search/budget_bin_2_3_fraction"] == pytest.approx(1.0 / 6.0)
    assert stats["search/budget_bin_4_7_count"] == 1
    assert stats["search/budget_bin_4_7_fraction"] == pytest.approx(1.0 / 6.0)
    assert stats["search/budget_bin_8_15_count"] == 0
    assert stats["search/budget_bin_16_cap_count"] == 0
    assert stats["search/forced_stop_rate"] + stats[
        "search/learned_stop_rate"
    ] == pytest.approx(1.0)
    assert stats["search/forced_stop_rate"] == pytest.approx(1.0 / 6.0)
    assert stats["search/learned_stop_rate"] == pytest.approx(5.0 / 6.0)
    behavior_entropy = compute_dynamic_control_entropy(
        shifted.search_control_logits
    )
    valid_control = shifted.control_valid.bool()
    assert stats["search/mean_stop_probability"] == pytest.approx(
        behavior_entropy.stop_prob[valid_control].mean().item()
    )
    assert stats["search/mean_continue_probability"] == pytest.approx(
        behavior_entropy.continue_prob[valid_control].mean().item()
    )
    assert stats["search/mean_gate_entropy"] == pytest.approx(
        behavior_entropy.gate[valid_control].mean().item()
    )
    assert stats["search/mean_bout_entropy"] == pytest.approx(
        behavior_entropy.bout[valid_control].mean().item()
    )
    behavior_primary_log_probs = torch.log_softmax(
        shifted.pri_param, dim=-1
    )
    behavior_primary_entropy = -(
        behavior_primary_log_probs.exp() * behavior_primary_log_probs
    ).sum(dim=-1).sum(dim=-1)
    assert stats["search/mean_primary_entropy"] == pytest.approx(
        behavior_primary_entropy[valid_control].mean().item()
    )
    expected_policy_entropy = (
        behavior_entropy.gate
        + behavior_entropy.continue_prob
        * (behavior_entropy.bout + behavior_primary_entropy)
    )
    assert stats["search/mean_policy_entropy"] == pytest.approx(
        expected_policy_entropy[valid_control].mean().item()
    )
    online_observability_sources = {
        "env_hs": (shifted.hs, None),
        "env_tree_reps": (shifted.tree_reps, None),
        "env_xs": (shifted.xs, None),
        "real_primary_logits": (
            shifted.pri_param,
            shifted.policy_valid.bool()
            & (shifted.policy_type == util.POLICY_REAL),
        ),
        "search_primary_logits": (
            shifted.pri_param,
            shifted.primary_valid.bool()
            & (shifted.policy_type == util.POLICY_SEARCH),
        ),
        "search_control_logits": (
            shifted.search_control_logits,
            shifted.control_valid.bool().unsqueeze(-1)
            & shifted.search_control_logits.ne(ILLEGAL_CONTROL_LOGIT),
        ),
    }
    for name, (value, mask) in online_observability_sources.items():
        expected_absmax, expected_rms = _expected_absmax_rms(value, mask)
        assert stats[f"actor/{name}_absmax"] == pytest.approx(
            expected_absmax
        )
        assert stats[f"actor/{name}_rms"] == pytest.approx(expected_rms)
    decision_depth = shifted.search_steps.long() - (
        valid_control & (shifted.search_control != util.STOP)
    ).long()
    stop_probability = behavior_entropy.stop_prob
    depth_bins = {
        "0": decision_depth == 0,
        "1": decision_depth == 1,
        "2_3": (decision_depth >= 2) & (decision_depth <= 3),
        "4_7": (decision_depth >= 4) & (decision_depth <= 7),
        "8_15": (decision_depth >= 8) & (decision_depth <= 15),
        "16_plus": decision_depth >= 16,
    }
    for label, depth_mask in depth_bins.items():
        mask = valid_control & depth_mask
        count = int(mask.sum().item())
        assert stats[f"search/depth_bin_{label}_count"] == count
        expected_stop_probability = (
            stop_probability[mask].mean().item() if count > 0 else 0.0
        )
        assert stats[
            f"search/depth_bin_{label}_stop_probability"
        ] == pytest.approx(expected_stop_probability)
    valid_depth = decision_depth[valid_control].float()
    valid_stop_probability = stop_probability[valid_control].float()
    centered_depth = valid_depth - valid_depth.mean()
    expected_slope = (
        torch.sum(
            centered_depth
            * (valid_stop_probability - valid_stop_probability.mean())
        )
        / torch.sum(centered_depth.square())
    ).item()
    assert stats["search/depth_stop_probability_count"] == int(
        valid_control.sum().item()
    )
    assert stats["search/depth_stop_probability_slope"] == pytest.approx(
        expected_slope
    )
    assert 0.0 <= stats["search/mean_stop_probability"] <= 1.0
    assert 0.0 <= stats["search/mean_continue_probability"] <= 1.0
    assert 0.0 <= stats["search/normalized_gate_entropy"] <= 1.0 + 1e-6
    assert 0.0 <= stats["search/normalized_bout_entropy"] <= 1.0 + 1e-6
    cap_stop = shifted.forced_stop.bool() & shifted.stage_end.bool()
    assert cap_stop.any()
    assert shifted.control_valid[cap_stop].all()
    assert (shifted.search_control[cap_stop] != util.STOP).all()
    think_reward_index = util.get_reward_names(flags).index("think")
    torch.testing.assert_close(
        shifted.reward[:, :, think_reward_index][cap_stop],
        -torch.ones_like(shifted.reward[:, :, think_reward_index][cap_stop]),
    )
    assert not shifted.stage_end[barrier_release].any()
