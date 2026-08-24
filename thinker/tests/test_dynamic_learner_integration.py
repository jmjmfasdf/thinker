from pathlib import Path

import numpy as np
import pytest
import torch

from tests.test_dynamic_cenv import FakeModel, FakeVectorEnv
from tests.test_dynamic_cenv import _flags as cenv_flags
from thinker import util
from thinker.actor_net import ActorNet
from thinker.cenv import cModelWrapper, cPerfectWrapper
from thinker.self_play import TrainActorOut
from thinker.util import EnvOut
import thinker.learn_actor as learn_actor_module


class _NullWriter:
    def log(self, *_args, **_kwargs):
        pass

    def close(self, *_args, **_kwargs):
        pass


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
def test_dynamic_cap_rollout_backpropagates_through_stop_and_think_cost(
        tmp_path, monkeypatch, ppo_k, wrapper_type):
    torch.manual_seed(23)
    flags = cenv_flags(cap=4, wrapper_type=wrapper_type)
    flags.float16 = False
    flags.see_real_state = False
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
    flags.xpid = "dynamic-learner-%d" % ppo_k
    flags.ckpdir = str(Path(tmp_path) / flags.xpid)

    actor, train_out, initial_actor_state = _rollout(flags)

    # Move the target policy off the behavior policy so the V-trace rho and
    # PPO ratio paths are exercised rather than degenerating to all ones.
    with torch.no_grad():
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

    losses, shifted = learner.compute_losses(
        train_out, initial_actor_state
    )
    assert torch.isfinite(losses["total_loss"])
    for value in losses.values():
        if torch.is_tensor(value):
            assert torch.isfinite(value).all()
        elif isinstance(value, (float, np.floating)):
            assert np.isfinite(value)

    wait = shifted.policy_type == util.POLICY_NONE
    nonreal_wait = wait & ~shifted.real_transition.bool()
    assert nonreal_wait.any(), "synthetic rollout must contain a WAIT-only row"
    assert not shifted.policy_valid[wait].any()
    assert torch.count_nonzero(shifted.c_action_log_prob[wait]) == 0
    assert len(seen_vtrace_inputs) == len(util.get_reward_names(flags))
    for vtrace_input in seen_vtrace_inputs:
        assert torch.count_nonzero(vtrace_input["log_rhos"][wait]) == 0
    torch.testing.assert_close(
        seen_vtrace_inputs[0]["discounts"][nonreal_wait],
        torch.ones_like(seen_vtrace_inputs[0]["discounts"][nonreal_wait]),
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

    learner.optimizer.zero_grad()
    losses["total_loss"].backward()
    control_grad = actor.reset.weight.grad
    assert control_grad is not None
    assert torch.isfinite(control_grad).all()
    assert torch.count_nonzero(control_grad[util.STOP]) > 0
    if ppo_k > 1:
        assert len(learner.ppo_is_abs) > 0
        assert np.isfinite(np.asarray(learner.ppo_is_abs)).all()

    stats = learner.compute_stat(
        shifted, losses={}, total_norm=0.0, actor_id=train_out.id
    )
    ended_steps = shifted.search_steps[shifted.stage_end.bool()].float()
    assert ended_steps.tolist() == [1.0, 2.0, 4.0, 0.0, 0.0, 0.0]
    assert stats["max_budget"] == 4.0
    assert stats["mean_budget"] == pytest.approx(7.0 / 6.0)
    barrier_release = shifted.real_transition.bool()
    assert not shifted.stage_end[barrier_release].any()
