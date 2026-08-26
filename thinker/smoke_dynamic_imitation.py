#!/usr/bin/env python3
"""One-update smoke test for the real Dynamic imitation components.

This intentionally avoids Ray.  It obtains observation/action metadata from
the requested live EnvPool Atari environment, samples genuine behavioral
frames, constructs the production ActorNet/ModelNet/cModelWrapper stack, and
runs one differentiable teacher-forced imitation rollout plus optimizer step.
The JSON result proves that Actor weights changed while ModelNet stayed frozen.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
import random
from typing import Any, Optional, Sequence

import numpy as np
import torch
from gymnasium import spaces


def _parse_ids(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result:
        raise argparse.ArgumentTypeError("ID list cannot be empty")
    return result


def _checkpoint_state_dict(checkpoint: Any, key: str) -> Mapping[str, Any]:
    if isinstance(checkpoint, Mapping) and key in checkpoint:
        state = checkpoint[key]
    else:
        state = checkpoint
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"checkpoint does not contain a non-empty {key!r}")
    return state


def _validate_state_dict(module: torch.nn.Module, state: Mapping[str, Any], label: str):
    expected = module.state_dict()
    expected_keys = set(expected)
    incoming_keys = set(state)
    missing = sorted(expected_keys - incoming_keys)
    unexpected = sorted(incoming_keys - expected_keys)
    mismatched = []
    for key in sorted(expected_keys & incoming_keys):
        incoming_shape = tuple(np.shape(state[key]))
        expected_shape = tuple(expected[key].shape)
        if incoming_shape != expected_shape:
            mismatched.append((key, incoming_shape, expected_shape))
    if missing or unexpected or mismatched:
        details = []
        if missing:
            details.append(f"missing={missing[:8]}")
        if unexpected:
            details.append(f"unexpected={unexpected[:8]}")
        if mismatched:
            details.append(
                "shape_mismatch=["
                + "; ".join(
                    f"{key}: incoming{incoming} != expected{target}"
                    for key, incoming, target in mismatched[:8]
                )
                + "]"
            )
        raise ValueError(f"{label} state-dict is incompatible: " + ", ".join(details))


def _vector_actor_observation_space(template: spaces.Dict, batch_size: int):
    if not isinstance(template, spaces.Dict):
        raise TypeError("cModelWrapper observation_space must be spaces.Dict")
    vector_spaces = dict(template.spaces)
    for key in ("real_states", "xs"):
        space = vector_spaces.get(key)
        if space is not None:
            vector_spaces[key] = spaces.Box(
                low=np.broadcast_to(space.low, (batch_size,) + tuple(space.shape)),
                high=np.broadcast_to(space.high, (batch_size,) + tuple(space.shape)),
                dtype=space.dtype,
            )
    for key in ("tree_reps", "hs"):
        space = vector_spaces.get(key)
        if space is not None and int(space.shape[0]) != batch_size:
            raise ValueError(
                f"cModelWrapper {key} batch axis is {space.shape[0]}, "
                f"expected {batch_size}"
            )
    return spaces.Dict(vector_spaces)


def _load_flags(args: argparse.Namespace):
    from thinker import util

    config = args.config
    if args.checkpoint_dir is not None and config is None:
        candidate = args.checkpoint_dir / "config_c.yaml"
        if not candidate.is_file():
            raise FileNotFoundError(f"missing checkpoint config: {candidate}")
        config = candidate

    fresh = config is None
    overrides = {
        "config": None if config is None else str(config),
        "name": args.env_name if fresh else None,
        "dynamic_search": True if fresh else None,
        "dynamic_factorized_control": True if fresh else None,
        "envpool": True,
        "parallel": False,
        "parallel_actor": False,
        "use_wandb": False,
        "float16": False,
        "model_float16": False if fresh else None,
        "model_disable_bn": False if fresh else None,
        "model_state_projection": "clamp" if fresh else None,
        "model_state_range_loss_cost": 1.0 if fresh else None,
        "rec_t": args.rec_t if args.rec_t is not None else (20 if fresh else None),
        "max_search_steps": (
            args.max_search_steps
            if args.max_search_steps is not None
            else (20 if fresh else None)
        ),
        "max_depth": (
            args.max_depth if args.max_depth is not None else (20 if fresh else None)
        ),
        "model_unroll_len": (
            args.model_unroll_len
            if args.model_unroll_len is not None
            else (20 if fresh else None)
        ),
        "think_cost": (
            args.think_cost if args.think_cost is not None else (0.0005 if fresh else None)
        ),
        "think_cost_anneal": False if fresh else None,
        "sep_im_head": True if fresh else None,
        "model_size_nn": (
            args.model_size_nn
            if args.model_size_nn is not None
            else (2 if fresh else None)
        ),
        "frame_stack_n": args.frame_stack_n,
        "grayscale": args.grayscale,
        "tree_carry": args.tree_carry,
    }
    flags = util.create_flags(
        ["default_thinker.yaml", "default_actor.yaml"],
        save_flags=False,
        post_fn=util.process_flags_actor,
        **{key: value for key, value in overrides.items() if value is not None},
    )
    if (
        not bool(flags.dynamic_search)
        or not bool(flags.dynamic_factorized_control)
        or not bool(flags.sep_im_head)
    ):
        raise ValueError(
            "smoke requires dynamic_search=true, "
            "dynamic_factorized_control=true, and sep_im_head=true"
        )
    if str(flags.name) != args.env_name:
        raise ValueError(
            f"config environment {flags.name!r} does not match --env-name "
            f"{args.env_name!r}"
        )
    if int(flags.max_search_steps) <= 0:
        raise ValueError("smoke requires a positive max_search_steps watchdog")
    flags.batch_length = int(args.scored_length)
    flags.icopro_device = str(args.device)
    return flags


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    from thinker import util
    from thinker.actor_net import ActorNet
    from thinker.bc_loader import FrameStackedBehavioralDataLoader
    from thinker.cenv import cModelWrapper
    from thinker.dataset_env import BehaviorSequenceVectorEnv
    from thinker.dynamic_imitation import DynamicImitationRunner
    from thinker.gym_add.wrapper import create_envpool
    from thinker.main import _validate_online_env_contract
    from thinker.model_net import ModelNet

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA smoke requested, but CUDA is unavailable")
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(args.device)

    flags = _load_flags(args)
    live_env = create_envpool(args.env_name, flags, env_n=args.batch_size)
    try:
        live_obs, _ = live_env.reset()
        live_next, live_reward, _, _, _ = live_env.step(
            np.zeros(args.batch_size, dtype=np.int64)
        )
        obs_space = live_env.single_observation_space
        action_space = live_env.single_action_space
        frame_stack_n = int(live_env.frame_stack_n)
        frame_ch = _validate_online_env_contract(
            obs_space,
            action_space,
            frame_stack_n,
            expected_frame_stack_n=flags.frame_stack_n,
            require_discrete=True,
        )
        expected_live_shape = (args.batch_size,) + tuple(obs_space.shape)
        if tuple(live_obs.shape) != expected_live_shape:
            raise ValueError(f"EnvPool reset shape mismatch: {live_obs.shape}")
        if tuple(live_next.shape) != expected_live_shape:
            raise ValueError(f"EnvPool step shape mismatch: {live_next.shape}")
    finally:
        live_env.close()

    dtype = np.dtype(obs_space.dtype)
    byte_contract = (
        dtype == np.dtype(np.uint8)
        and np.all(obs_space.low == 0)
        and np.all(obs_space.high == 255)
    )
    unit_float_contract = (
        dtype == np.dtype(np.float32)
        and np.all(obs_space.low == 0.0)
        and np.all(obs_space.high == 1.0)
    )
    if not byte_contract and not unit_float_contract:
        raise ValueError(
            "behavioral preprocessing supports only uint8 [0,255] or "
            "float32 [0,1] online observations"
        )

    loader = FrameStackedBehavioralDataLoader(
        base_path=args.data_root,
        subjects=args.subjects,
        game_id=args.game_id,
        sessions=args.sessions,
        num_actions=int(action_space.n),
        scored_length=args.scored_length,
        frame_stack_n=frame_stack_n,
        target_size=tuple(int(value) for value in obs_space.shape[-2:]),
        grayscale=bool(flags.grayscale),
        normalize=unit_float_contract,
        seed=args.seed,
    )
    batch = loader.get_sequence_batch(
        batch_size=args.batch_size, sequence_length=args.scored_length
    )
    observations = np.asarray(batch["obs_seq"])
    actions = np.asarray(batch["actions_seq"], dtype=np.int64)
    if tuple(observations.shape[2:]) != tuple(obs_space.shape):
        raise ValueError(
            f"behavior/EnvPool shape mismatch: {observations.shape[2:]} "
            f"versus {obs_space.shape}"
        )
    if np.dtype(observations.dtype) != dtype:
        raise TypeError(
            f"behavior/EnvPool dtype mismatch: {observations.dtype} versus {dtype}"
        )
    if actions.min() < 0 or actions.max() >= int(action_space.n):
        raise ValueError("behavior batch contains an out-of-range action")

    model = ModelNet(
        obs_space=obs_space,
        action_space=action_space,
        flags=flags,
        frame_stack_n=frame_stack_n,
    ).to(args.device)

    behavior_env = BehaviorSequenceVectorEnv(
        obs_seq=observations,
        actions_seq=actions,
        rewards_seq=np.asarray(batch["rewards_seq"], dtype=np.float32),
        done_seq=np.asarray(batch["done_seq"], dtype=np.bool_),
        truncated_seq=np.asarray(batch["truncated_seq"], dtype=np.bool_),
        initial_prev_action=np.asarray(batch["initial_prev_action"], dtype=np.int64),
        score_mask=np.asarray(batch["score_mask"], dtype=np.bool_),
        num_actions=int(action_space.n),
    )
    template = cModelWrapper(
        env=behavior_env,
        env_n=args.batch_size,
        flags=flags,
        model_net=model,
        device=args.device,
        timing=False,
    )
    try:
        actor = ActorNet(
            obs_space=_vector_actor_observation_space(
                template.observation_space, args.batch_size
            ),
            action_space=template.action_space,
            flags=flags,
            tree_rep_meaning=util.get_tree_rep_meaning(
                int(action_space.n), 1, flags
            ),
        ).to(args.device)
    finally:
        template.close()

    if args.checkpoint_dir is not None:
        actor_path = args.checkpoint_dir / "ckp_actor.tar"
        model_path = args.checkpoint_dir / "ckp_model.tar"
        if not actor_path.is_file() or not model_path.is_file():
            raise FileNotFoundError(
                "checkpoint-dir must contain ckp_actor.tar and ckp_model.tar"
            )
        actor_state = _checkpoint_state_dict(
            torch.load(actor_path, map_location="cpu", weights_only=False),
            "actor_net_state_dict",
        )
        model_state = _checkpoint_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=False),
            "model_net_state_dict",
        )
        _validate_state_dict(actor, actor_state, "ActorNet")
        _validate_state_dict(model, model_state, "ModelNet")
        actor.set_weights(actor_state)
        model.set_weights(model_state)

    if int(actor.num_actions) != int(model.num_actions) or int(actor.num_actions) != int(
        action_space.n
    ):
        raise ValueError("EnvPool, ActorNet, and ModelNet action counts disagree")
    if tuple(actor.online_real_state_space.shape) != tuple(obs_space.shape):
        raise ValueError("Actor online observation shape was not preserved")
    if np.dtype(actor.online_real_state_space.dtype) != dtype:
        raise TypeError("Actor online observation dtype was not preserved")
    if int(model.frame_stack_n) != frame_stack_n:
        raise ValueError("ModelNet frame-stack metadata were not preserved")

    actor.train(True)
    optimizer = torch.optim.Adam(
        actor.parameters(),
        lr=(
            float(args.learning_rate)
            if args.learning_rate is not None
            else float(flags.actor_learning_rate)
        ),
        eps=float(flags.actor_adam_eps),
    )
    runner = DynamicImitationRunner(actor, model, flags, device=args.device)
    try:
        model_versions = {
            name: parameter._version for name, parameter in model.named_parameters()
        }
        if not all(not parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("DynamicImitationRunner did not freeze ModelNet")
        actor_versions = {
            name: parameter._version for name, parameter in actor.named_parameters()
        }

        optimizer.zero_grad(set_to_none=True)
        result = runner.rollout(batch, tree_carry=bool(flags.tree_carry), training=True)
        expected_count = args.batch_size * args.scored_length
        if result.count != expected_count:
            raise RuntimeError(
                f"burn-in/scored count mismatch: {result.count} versus {expected_count}"
            )
        if not torch.equal(result.all_executed.cpu(), torch.as_tensor(actions)):
            raise RuntimeError("cenv did not execute every teacher-forced human action")
        if not torch.isfinite(result.loss):
            raise RuntimeError("imitation loss is non-finite")
        result.loss.backward()

        grad_entries = [
            (name, parameter)
            for name, parameter in actor.named_parameters()
            if parameter.grad is not None
            and int(torch.count_nonzero(parameter.grad).item()) > 0
        ]
        if not grad_entries:
            raise RuntimeError("Actor received no nonzero imitation gradient")
        if not all(torch.isfinite(parameter.grad).all() for _, parameter in grad_entries):
            raise RuntimeError("Actor imitation gradient is non-finite")
        probe_name, probe = grad_entries[0]
        probe_before = probe.detach().clone()
        grad_norm = torch.sqrt(
            sum(
                parameter.grad.detach().float().square().sum()
                for _, parameter in grad_entries
            )
        )
        optimizer.step()
        probe_delta = (probe.detach() - probe_before).abs().max()
        actor_changed = sum(
            parameter._version != actor_versions[name]
            for name, parameter in actor.named_parameters()
        )
        model_unchanged = all(
            parameter._version == model_versions[name]
            for name, parameter in model.named_parameters()
        )
        model_grad_free = all(parameter.grad is None for parameter in model.parameters())
        if float(probe_delta) <= 0.0 or actor_changed == 0:
            raise RuntimeError("Actor optimizer step did not change a weight")
        if not model_unchanged or not model_grad_free:
            raise RuntimeError("frozen ModelNet changed during Actor optimization")

        result_json = {
            "env": args.env_name,
            "game_id": args.game_id,
            "device": str(args.device),
            "gpu": (
                torch.cuda.get_device_name(args.device)
                if args.device.type == "cuda"
                else None
            ),
            "A": int(action_space.n),
            "obs_shape": list(obs_space.shape),
            "obs_dtype": str(obs_space.dtype),
            "obs_low": float(np.min(obs_space.low)),
            "obs_high": float(np.max(obs_space.high)),
            "frame_stack_n": frame_stack_n,
            "frame_ch": int(frame_ch),
            "dynamic_factorized_control": bool(
                flags.dynamic_factorized_control
            ),
            "model_state_projection": str(flags.model_state_projection),
            "model_state_range_loss_cost": float(
                flags.model_state_range_loss_cost
            ),
            "behavior_batch": list(observations.shape),
            "human_actions": actions.tolist(),
            "loss": float(result.loss.detach().cpu()),
            "nll": float(result.nll_sum.cpu()) / result.count,
            "count": int(result.count),
            "augmented_steps": int(result.augmented_steps),
            "root_carried_rate": float(result.root_carried.float().mean().cpu()),
            "actor_nonzero_grad_tensors": len(grad_entries),
            "actor_grad_norm": float(grad_norm.cpu()),
            "actor_probe": probe_name,
            "actor_probe_max_abs_update": float(probe_delta.cpu()),
            "actor_changed_parameter_count": int(actor_changed),
            "model_frozen_requires_grad": all(
                not parameter.requires_grad for parameter in model.parameters()
            ),
            "model_grad_free": bool(model_grad_free),
            "model_versions_unchanged": bool(model_unchanged),
            "peak_gpu_allocated_mib": (
                torch.cuda.max_memory_allocated(args.device) / 1024**2
                if args.device.type == "cuda"
                else 0.0
            ),
            "peak_gpu_reserved_mib": (
                torch.cuda.max_memory_reserved(args.device) / 1024**2
                if args.device.type == "cuda"
                else 0.0
            ),
            "live_step_reward_mean": float(np.asarray(live_reward).mean()),
        }
    finally:
        runner.close()
    return result_json


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-name", required=True)
    parser.add_argument("--game-id", required=True, type=int)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--subjects", type=_parse_ids, default=(1,))
    parser.add_argument("--sessions", type=_parse_ids, default=(1, 2, 3))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--scored-length", type=int, default=4)
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--rec-t", type=int, default=None)
    parser.add_argument("--max-search-steps", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--model-unroll-len", type=int, default=None)
    parser.add_argument("--think-cost", type=float, default=None)
    parser.add_argument("--model-size-nn", type=int, default=None)
    parser.add_argument("--frame-stack-n", type=int, default=None)
    parser.add_argument(
        "--grayscale", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--tree-carry", action=argparse.BooleanOptionalAction, default=None
    )
    parsed = parser.parse_args(argv)
    parsed.data_root = parsed.data_root.expanduser().resolve()
    if parsed.config is not None:
        parsed.config = parsed.config.expanduser().resolve()
    if parsed.checkpoint_dir is not None:
        parsed.checkpoint_dir = parsed.checkpoint_dir.expanduser().resolve()
    if parsed.batch_size <= 0 or parsed.scored_length <= 0:
        parser.error("batch-size and scored-length must be positive")
    return parsed


def main(argv: Optional[Sequence[str]] = None) -> int:
    result = run_smoke(parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
