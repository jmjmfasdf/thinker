#!/usr/bin/env python3
"""Offline ML-IRL driver that ties behavioral data with Thinker checkpoints."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from gymnasium import spaces

from thinker import util
from thinker.actor_net import ActorNet
from thinker.bc_loader import FrameStackedBehavioralDataLoader
from thinker.model_net import ModelNet
from thinker.inverse_rl import (
    OfflineMLIRLTrainer,
    OfflinePolicyUpdater,
    RewardEstimator,
    ThinkerBehaviorDataset,
    WorldModelAdapter,
    load_offline_mlirl_config,
)
from thinker.inverse_rl.features import extract_features

if TYPE_CHECKING:
    from imitation import ThinkerPolicyAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline ML-IRL trainer for Thinker.")
    parser.add_argument("--config", required=True, help="Path to Thinker config_c.yaml (pretrained run).")
    parser.add_argument("--preload", required=True, help="Directory containing ckp_model.tar / ckp_actor.tar.")
    parser.add_argument("--irl-config", default="thinker/config/offline_mlirl.yaml", help="Offline ML-IRL YAML config.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Computation device.")
    parser.add_argument("--name", default=None, help="Override environment name for logging.")
    return parser.parse_args()


def resolve_device(pref: str) -> torch.device:
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model_components(
    flags,
    loader: FrameStackedBehavioralDataLoader,
    device: torch.device,
) -> Tuple["ModelNet", "ActorNet", "ThinkerPolicyAdapter"]:
    from imitation import ThinkerPolicyAdapter

    model_ckp = torch.load(os.path.join(flags.preload, "ckp_model.tar"), map_location=device, weights_only=False)
    pretrained_flags = model_ckp.get("flags")
    pretrained_frame_stack = getattr(pretrained_flags, "frame_stack_n", loader.frame_stack_n) if pretrained_flags else loader.frame_stack_n
    pretrained_grayscale = getattr(pretrained_flags, "grayscale", loader.grayscale) if pretrained_flags else loader.grayscale
    if loader.frame_stack_n != pretrained_frame_stack:
        print(f"[offline-mlirl] Adjusting frame_stack_n to match checkpoint: {loader.frame_stack_n} -> {pretrained_frame_stack}")
        loader.frame_stack_n = pretrained_frame_stack
    if loader.grayscale != pretrained_grayscale:
        print(f"[offline-mlirl] Adjusting grayscale flag to match checkpoint: {loader.grayscale} -> {pretrained_grayscale}")
        loader.grayscale = pretrained_grayscale

    height, width = loader.target_size
    channels_per_frame = 1 if loader.grayscale else 3
    frame_stack = loader.frame_stack_n
    obs_shape = (frame_stack * channels_per_frame, height, width)
    model_obs_space = spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
    primary_action_space = spaces.Discrete(loader.num_actions)

    state_dict = model_ckp["model_net_state_dict"]
    decoder_bias = state_dict.get("sr_net.encoder.d_conv.13.bias")
    obs_channels = frame_stack * channels_per_frame
    decoder_channels = decoder_bias.shape[0] if decoder_bias is not None else channels_per_frame
    if decoder_channels == obs_channels:
        model_frame_stack = 1
    else:
        model_frame_stack = frame_stack

    model_net = ModelNet(
        obs_space=model_obs_space,
        action_space=primary_action_space,
        flags=flags,
        frame_stack_n=model_frame_stack,
    ).to(device)
    model_net.set_weights(state_dict)
    model_net.eval()
    for param in model_net.parameters():
        param.requires_grad_(False)

    actor_obs_space = _build_actor_obs_space(
        flags,
        frame_stack,
        channels_per_frame,
        model_net,
        primary_action_space,
        height,
        width,
    )
    reset_action_space = spaces.Discrete(2)
    actor_action_space = spaces.Tuple(
        (
            spaces.Tuple((primary_action_space,)),
            spaces.Tuple((reset_action_space,)),
        )
    )
    num_actions, dim_actions, *_ = util.process_action_space(primary_action_space)
    tree_rep_meaning = util.slice_tree_reps(num_actions, dim_actions, flags.rec_t)

    actor_net = ActorNet(
        obs_space=actor_obs_space,
        action_space=actor_action_space,
        flags=flags,
        tree_rep_meaning=tree_rep_meaning,
    ).to(device)
    actor_ckp = torch.load(os.path.join(flags.preload, "ckp_actor.tar"), map_location=device, weights_only=False)
    actor_net.set_weights(actor_ckp["actor_net_state_dict"])
    actor_net.train(True)

    policy_adapter = ThinkerPolicyAdapter(actor_net, model_net, flags, device)
    policy_adapter.train(True)
    return model_net, actor_net, policy_adapter


def _build_actor_obs_space(flags, frame_stack: int, channels: int, model_net: ModelNet, action_space, height: int, width: int):
    num_actions, _, _, _, _ = util.process_action_space(action_space)
    tree_rep_size = 11 + num_actions * 10 + flags.rec_t
    if getattr(flags, "has_action_seq", False) and getattr(flags, "reset_mode", 0) == 0:
        tree_rep_size += flags.max_depth * num_actions + num_actions
    real_shape = (1, frame_stack * channels, height, width)
    xs_shape = (1,) + tuple(model_net.obs_shape)
    hs_shape = (1,) + tuple(model_net.hidden_shape)
    tree_shape = (1, tree_rep_size)
    return spaces.Dict(
        {
            "real_states": spaces.Box(low=0, high=255, shape=real_shape, dtype=np.uint8),
            "tree_reps": spaces.Box(low=-np.inf, high=np.inf, shape=tree_shape, dtype=np.float32),
            "xs": spaces.Box(low=-np.inf, high=np.inf, shape=xs_shape, dtype=np.float32),
            "hs": spaces.Box(low=-np.inf, high=np.inf, shape=hs_shape, dtype=np.float32),
        }
    )


def probe_feature_dim(
    dataset: ThinkerBehaviorDataset,
    policy_adapter: ThinkerPolicyAdapter,
    device: torch.device,
    feature_source: str,
) -> int:
    warm_batch = dataset.sample_batch(batch_size=1)
    if warm_batch is None:
        raise RuntimeError("Unable to draw warm-up batch from behavioral dataset.")
    obs_np = warm_batch.images[:, 0]
    prev_actions_np = np.zeros(obs_np.shape[0], dtype=np.int64)
    seq_flags_np = warm_batch.is_first[:, 0]
    obs = torch.from_numpy(obs_np).to(device=device, dtype=torch.float32)
    prev_actions = torch.from_numpy(prev_actions_np).to(device)
    seq_flags = torch.from_numpy(seq_flags_np).to(device)
    policy_batch = policy_adapter.forward(
        obs,
        prev_actions=prev_actions,
        sequence_starts=seq_flags,
        requires_grad=True,
    )
    feats = extract_features(policy_adapter, policy_batch, feature_source)
    feats = feats.detach()
    return feats.shape[-1]


def main() -> None:
    args = parse_args()
    irl_cfg_path = Path(args.irl_config)
    if not irl_cfg_path.exists():
        alt = REPO_ROOT / args.irl_config
        if alt.exists():
            irl_cfg_path = alt
    config_path = Path(args.config)
    if not config_path.exists():
        alt = REPO_ROOT / args.config
        if alt.exists():
            config_path = alt
    preload_path = Path(args.preload)
    if not preload_path.exists():
        alt = REPO_ROOT / args.preload
        if alt.exists():
            preload_path = alt
    if not config_path.exists():
        raise FileNotFoundError(f"Could not locate config file '{args.config}'.")
    if not preload_path.exists():
        raise FileNotFoundError(f"Could not locate preload directory '{args.preload}'.")
    cfg = load_offline_mlirl_config(irl_cfg_path)
    cfg.thinker.preload = str(preload_path)

    flags = util.create_flags(
        ["default_thinker.yaml", "default_actor.yaml"],
        save_flags=False,
        post_fn=util.process_flags_actor,
        config=str(config_path),
        preload=str(preload_path),
        preload_actor=str(preload_path),
        name=args.name or None,
    )
    cfg.behavior.frame_stack_n = flags.frame_stack_n
    cfg.behavior.grayscale = flags.grayscale
    device = resolve_device(args.device)

    loader = FrameStackedBehavioralDataLoader(
        base_path=cfg.behavior.base_path,
        subjects=cfg.behavior.subjects,
        game_id=cfg.behavior.game_id,
        frame_stack_n=cfg.behavior.frame_stack_n,
        target_size=tuple(cfg.behavior.target_size),
        grayscale=cfg.behavior.grayscale,
    )
    dataset = ThinkerBehaviorDataset(loader, cfg.behavior.sequence_length, cfg.optim.gamma)
    save_dir = Path(flags.savedir)  / (flags.xpid or "offline-mlirl")
    save_dir.mkdir(parents=True, exist_ok=True)

    model_net, actor_net, policy_adapter = build_model_components(flags, loader, device)
    feature_dim = probe_feature_dim(dataset, policy_adapter, device, cfg.thinker.reward_feature_source)
    reward_estimator = RewardEstimator(
        input_dim=feature_dim,
        hidden_dims=(256, 256),
    ).to(device)

    reward_optimizer = torch.optim.Adam(reward_estimator.parameters(), lr=cfg.optim.reward_lr)
    policy_optimizer = torch.optim.Adam(actor_net.parameters(), lr=cfg.optim.actor_lr)

    world_model = WorldModelAdapter(
        gamma=cfg.optim.gamma,
        penalty_scale=cfg.optim.penalty_scale,
        penalty_clamp=cfg.optim.penalty_clamp,
    )
    policy_updater = OfflinePolicyUpdater(
        actor_net=actor_net,
        optimizer=policy_optimizer,
        entropy_coef=cfg.optim.entropy_coef,
        max_grad_norm=cfg.optim.actor_grad_clip,
    )
    trainer = OfflineMLIRLTrainer(
        config=cfg,
        dataset=dataset,
        policy_adapter=policy_adapter,
        reward_estimator=reward_estimator,
        reward_optimizer=reward_optimizer,
        policy_updater=policy_updater,
        world_model=world_model,
        device=device,
        save_dir=save_dir,
    )

    trainer.train()
    torch.save(
        {
            "reward_state_dict": reward_estimator.state_dict(),
            "config": cfg.to_dict(),
        },
        save_dir / "reward_estimator.pt",
    )
    trainer.save_actor_checkpoint(save_dir / "ckp_actor.tar")
    print(f"Offline ML-IRL artifacts saved under {save_dir}")


if __name__ == "__main__":
    main()
