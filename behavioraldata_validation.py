#!/usr/bin/env python3
import argparse
import os
import random
import sys
from statistics import mean, stdev
from typing import List, Tuple

import cv2
import numpy as np
import torch
from gymnasium import spaces

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Ensure both the top-level repo directory and the Thinker package directory are on the path.
sys.path.insert(0, os.path.join(SCRIPT_DIR, "thinker"))
sys.path.insert(1, SCRIPT_DIR)

from thinker import util
from thinker.actor_net import ActorNet
from imitation import ThinkerPolicyAdapter
from thinker.model_net import ModelNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Thinker actions against human behavioral data samples."
    )
    parser.add_argument("--data", required=True, help="Path to behavioral npz file containing paired observations.")
    parser.add_argument("--preload", required=True, help="Checkpoint directory containing Thinker weights (ckp_model.tar / ckp_actor.tar).")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Computation device to use.",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=100,
        help="Number of random samples to evaluate per repeat.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of independent evaluation repeats.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for Thinker forward passes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print a short sample-level breakdown for each repeat.",
    )
    return parser.parse_args()


def resolve_device(device_pref: str) -> torch.device:
    if device_pref == "cpu":
        return torch.device("cpu")
    if device_pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no GPU is available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_episode_ids(is_first: np.ndarray) -> np.ndarray:
    flags = is_first.astype(bool)
    episode_ids = np.zeros_like(flags, dtype=np.int64)
    current_id = -1
    for idx, flag in enumerate(flags):
        if flag or current_id < 0:
            current_id += 1
        episode_ids[idx] = current_id
    if current_id < 0:
        # No episode boundary was indicated; treat the full sequence as a single episode.
        episode_ids[:] = 0
    return episode_ids


def preprocess_frame(image: np.ndarray, target_size: Tuple[int, int], grayscale: bool) -> np.ndarray:
    if image.ndim == 2:
        image = image[..., np.newaxis]

    if image.shape[0] != target_size[1] or image.shape[1] != target_size[0]:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        if image.ndim == 2:
            image = image[..., np.newaxis]

    if grayscale:
        if image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = image[..., np.newaxis]
    else:
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        elif image.shape[-1] == 4:
            image = image[..., :3]

    frame = image.astype(np.float32) / 255.0
    frame = np.transpose(frame, (2, 0, 1))
    return frame


def build_stack_for_index(
    images: np.ndarray,
    index: int,
    frame_stack: int,
    blank_frame: np.ndarray,
    target_size: Tuple[int, int],
    grayscale: bool,
    episode_ids: np.ndarray,
) -> np.ndarray:
    frames: List[np.ndarray] = []
    target_episode = episode_ids[index]
    for offset in range(frame_stack - 1, -1, -1):
        src_idx = index - offset
        if src_idx < 0 or episode_ids[src_idx] != target_episode:
            frames.append(blank_frame)
        else:
            frames.append(preprocess_frame(images[src_idx], target_size, grayscale))
    return np.concatenate(frames, axis=0)


def compute_stack_step_metadata(
    valid_mask: np.ndarray,
    is_first: np.ndarray,
    frame_stack: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return frame indices representing complete stacked observations."""
    is_first_bool = np.asarray(is_first, dtype=bool)
    indices: list[int] = []
    prev_indices: list[int] = []
    seq_flags: list[bool] = []

    frames_since_reset = 0
    total_frames = len(valid_mask)
    for idx in range(total_frames):
        new_episode = idx == 0 or is_first_bool[idx]
        if new_episode:
            frames_since_reset = 1
        else:
            frames_since_reset += 1

        if not valid_mask[idx]:
            continue

        if frames_since_reset >= frame_stack and frames_since_reset % frame_stack == 0:
            prev_idx = idx - frame_stack if frames_since_reset > frame_stack else -1
            indices.append(idx)
            prev_indices.append(prev_idx)
            seq_flags.append(prev_idx < 0)

    return (
        np.asarray(indices, dtype=np.int64),
        np.asarray(prev_indices, dtype=np.int64),
        np.asarray(seq_flags, dtype=np.bool_),
    )


def prepare_actor_spaces(
    flags,
    frame_stack: int,
    channels: int,
    tree_rep_size: int,
    model_net: ModelNet,
    height: int,
    width: int,
    env_n: int = 1,
):
    real_shape = (env_n, frame_stack * channels, height, width)
    xs_shape = (env_n, model_net.obs_shape[0], height, width)
    hs_shape = (env_n,) + tuple(model_net.hidden_shape)

    real_states_space = spaces.Box(low=0, high=255, shape=real_shape, dtype=np.uint8)
    tree_reps_space = spaces.Box(low=-np.inf, high=np.inf, shape=(env_n, tree_rep_size), dtype=np.float32)
    xs_space = spaces.Box(low=-np.inf, high=np.inf, shape=xs_shape, dtype=np.float32)
    hs_space = spaces.Box(low=-np.inf, high=np.inf, shape=hs_shape, dtype=np.float32)

    return spaces.Dict(
        {
            "real_states": real_states_space,
            "tree_reps": tree_reps_space,
            "xs": xs_space,
            "hs": hs_space,
        }
    )


def load_thinker_components(
    preload_dir: str,
    device: torch.device,
    *,
    target_hw: Tuple[int, int],
    raw_channels: int,
    action_dim: int,
):
    preload_dir = os.path.abspath(preload_dir)
    config_path = os.path.abspath(os.path.join(preload_dir, "config_c.yaml"))
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Could not find config_c.yaml under '{preload_dir}'.")

    flags = util.create_flags(config_path, save_flags=False)
    flags.parallel = False
    flags.env_n = 1

    if device.type != "cuda":
        flags.float16 = False

    model_ckp = os.path.join(preload_dir, "ckp_model.tar")
    model_checkpoint = torch.load(model_ckp, map_location=device) if os.path.isfile(model_ckp) else None

    if model_checkpoint is not None and "flags" in model_checkpoint:
        pretrained_flags = model_checkpoint["flags"]
        if isinstance(pretrained_flags, dict):
            flags.frame_stack_n = pretrained_flags.get("frame_stack_n", flags.frame_stack_n)
            flags.grayscale = pretrained_flags.get("grayscale", flags.grayscale)
            flags.env_n = pretrained_flags.get("env_n", flags.env_n)
        else:
            flags.frame_stack_n = getattr(pretrained_flags, "frame_stack_n", flags.frame_stack_n)
            flags.grayscale = getattr(pretrained_flags, "grayscale", flags.grayscale)
            flags.env_n = getattr(pretrained_flags, "env_n", flags.env_n)

    if flags.grayscale:
        print("[INFO] Overriding grayscale=True to False to match SR checkpoint expectations.")
        flags.grayscale = False

    frame_stack = int(flags.frame_stack_n)
    channels_per_frame = 1 if flags.grayscale else raw_channels
    total_channels = channels_per_frame * frame_stack

    model_frame_stack = frame_stack
    if model_checkpoint is not None:
        state_dict = model_checkpoint.get("model_net_state_dict", {})
        decoder_weight = state_dict.get("sr_net.encoder.d_conv.13.weight")
        if decoder_weight is None:
            for key, value in state_dict.items():
                if "sr_net.encoder.d_conv" in key and key.endswith("weight"):
                    if value.dim() == 4 and value.shape[2:] == (4, 4):
                        decoder_weight = value
                        break
        if decoder_weight is not None:
            decoder_channels = decoder_weight.shape[1]
            if decoder_channels % channels_per_frame == 0 and channels_per_frame > 0:
                inferred = max(1, (frame_stack * channels_per_frame) // decoder_channels)
                model_frame_stack = inferred

    height, width = target_hw
    model_obs_space = spaces.Box(
        low=0,
        high=255,
        shape=(total_channels, height, width),
        dtype=np.uint8,
    )
    primary_action_space = spaces.Discrete(action_dim)

    model_net = ModelNet(
        obs_space=model_obs_space,
        action_space=primary_action_space,
        flags=flags,
        frame_stack_n=model_frame_stack,
    ).to(device)

    if model_checkpoint is not None:
        model_net.set_weights(model_checkpoint["model_net_state_dict"])

    model_net.eval()
    for param in model_net.parameters():
        param.requires_grad_(False)
    if hasattr(model_net, "reward_clip"):
        try:
            model_net.reward_clip = 0.0
        except AttributeError:
            pass
    if hasattr(model_net, "value_clip"):
        try:
            model_net.value_clip = 0.0
        except AttributeError:
            pass

    num_actions, dim_actions, _, _, _ = util.process_action_space(primary_action_space)
    tree_rep_size = 11 + num_actions * 10 + flags.rec_t
    if getattr(flags, "has_action_seq", False) and getattr(flags, "reset_mode", 0) == 0:
        tree_rep_size += flags.max_depth * num_actions + num_actions

    actor_obs_space = prepare_actor_spaces(
        flags,
        frame_stack,
        channels_per_frame,
        tree_rep_size,
        model_net,
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
    tree_rep_meaning = util.slice_tree_reps(num_actions, dim_actions, flags.rec_t)

    actor_net = ActorNet(
        obs_space=actor_obs_space,
        action_space=actor_action_space,
        flags=flags,
        tree_rep_meaning=tree_rep_meaning,
    ).to(device)
    actor_net.eval()

    actor_ckp = os.path.join(preload_dir, "ckp_actor.tar")
    if os.path.isfile(actor_ckp):
        actor_checkpoint = torch.load(actor_ckp, map_location=device, weights_only=False)
        actor_net.set_weights(actor_checkpoint["actor_net_state_dict"])

    for param in actor_net.parameters():
        param.requires_grad_(False)

    policy_adapter = ThinkerPolicyAdapter(actor_net, model_net, flags, device)

    return {
        "flags": flags,
        "model_net": model_net,
        "actor_net": actor_net,
        "policy_adapter": policy_adapter,
        "frame_stack": frame_stack,
        "channels_per_frame": channels_per_frame,
    }


def load_behavioral_dataset(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Behavioral dataset '{path}' does not exist.")

    with np.load(path) as data:
        images = data["image"]
        actions = data["action"]
        rewards = data["reward"]
        is_first = data["is_first"]
        is_terminal = data["is_terminal"] if "is_terminal" in data else data["done"]

    return images, actions, rewards, is_first, is_terminal


def extract_human_actions(actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if actions.ndim == 1:
        human_actions = actions.astype(np.int64)
        valid_mask = human_actions >= 0
    elif actions.ndim == 2:
        human_actions = np.argmax(actions, axis=1).astype(np.int64)
        valid_mask = actions.sum(axis=1) > 0.0
    else:
        raise ValueError(f"Unsupported action tensor shape: {actions.shape}")
    return human_actions, valid_mask


def ensure_positive_int(value: int, name: str) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def main() -> None:
    args = parse_args()
    ensure_positive_int(args.num_samples, "num_samples")
    ensure_positive_int(args.repeat, "repeat")
    ensure_positive_int(args.batch_size, "batch_size")

    device = resolve_device(args.device)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        rng = np.random.default_rng(args.seed)
    else:
        rng = np.random.default_rng()

    images, actions, _rewards, is_first, _is_terminal = load_behavioral_dataset(args.data)
    if images.ndim != 4:
        raise ValueError(f"Expected images with shape (T, H, W, C); got {images.shape}")

    total_frames, target_h, target_w, raw_channels = images.shape
    action_dim = actions.shape[1] if actions.ndim >= 2 else int(actions.max()) + 1

    components = load_thinker_components(
        args.preload,
        device,
        target_hw=(target_h, target_w),
        raw_channels=raw_channels,
        action_dim=action_dim,
    )
    flags = components["flags"]
    policy_adapter = components["policy_adapter"]

    frame_stack = components["frame_stack"]
    channels_per_frame = components["channels_per_frame"]
    blank_frame = np.zeros((channels_per_frame, target_h, target_w), dtype=np.float32)

    episode_ids = compute_episode_ids(is_first)
    human_actions, valid_mask = extract_human_actions(actions)

    stack_frame_indices, prev_frame_indices, seq_flags_full = compute_stack_step_metadata(
        valid_mask=valid_mask,
        is_first=is_first,
        frame_stack=frame_stack,
    )
    if stack_frame_indices.size == 0:
        raise RuntimeError("No complete frame stacks found in the dataset.")

    available_step_ids = np.arange(stack_frame_indices.size, dtype=np.int64)
    replace = args.num_samples > available_step_ids.size
    if replace:
        print(
            f"[WARN] Requested {args.num_samples} samples but only {available_step_ids.size} stacked frames; "
            "sampling with replacement."
        )

    batch_size = min(args.batch_size, args.num_samples)
    results: List[float] = []
    total_matches = 0
    total_samples = 0

    try:
        for repeat_idx in range(args.repeat):
            sample_step_ids = rng.choice(available_step_ids, size=args.num_samples, replace=replace)
            match_count = 0
            sample_records: List[Tuple[int, int, int, bool]] = []

            for start in range(0, len(sample_step_ids), batch_size):
                batch_step_ids = sample_step_ids[start : start + batch_size]
                prev_actions_list = []
                seq_start_flags = []
                stacked_obs = []
                frame_indices: List[int] = []
                for step_id in batch_step_ids:
                    frame_idx = int(stack_frame_indices[step_id])
                    prev_idx = int(prev_frame_indices[step_id])
                    sequence_start = bool(seq_flags_full[step_id])

                    prev_action = 0
                    if prev_idx >= 0 and prev_idx < len(human_actions) and valid_mask[prev_idx]:
                        prev_action = int(human_actions[prev_idx])

                    prev_actions_list.append(prev_action)
                    seq_start_flags.append(sequence_start)
                    frame_indices.append(frame_idx)
                    stacked_obs.append(
                        build_stack_for_index(
                            images,
                            frame_idx,
                            frame_stack,
                            blank_frame,
                            (target_w, target_h),
                            flags.grayscale,
                            episode_ids,
                        )
                    )
                prev_actions_arr = np.asarray(prev_actions_list, dtype=np.int64)
                seq_start_arr = np.asarray(seq_start_flags, dtype=bool)
                obs_batch = np.stack(stacked_obs, axis=0).astype(np.float32)
                obs_tensor = torch.from_numpy(obs_batch).to(device=device)
                prev_action_tensor = torch.from_numpy(prev_actions_arr).to(device=device)
                seq_start_tensor = torch.from_numpy(seq_start_arr.astype(np.bool_)).to(device=device)
                teacher_tensor = torch.from_numpy(human_actions[frame_indices].astype(np.int64)).to(device=device)

                with torch.no_grad():
                    policy_batch = policy_adapter.forward(
                        obs_tensor,
                        actions=teacher_tensor,
                        prev_actions=prev_action_tensor,
                        sequence_starts=seq_start_tensor,
                        requires_grad=False,
                    )
                logits = policy_batch.logits.detach().cpu()
                predicted_actions = torch.argmax(logits, dim=-1).numpy()

                human_batch = human_actions[frame_indices]
                matches = predicted_actions == human_batch
                match_count += int(matches.sum())

                if args.verbose:
                    for idx_entry, prediction, target, is_match in zip(
                        frame_indices, predicted_actions, human_batch, matches
                    ):
                        sample_records.append((int(idx_entry), int(prediction), int(target), bool(is_match)))

            accuracy = match_count / len(sample_step_ids)
            results.append(accuracy)
            total_matches += match_count
            total_samples += len(sample_step_ids)
            print(
                f"Repeat {repeat_idx + 1}/{args.repeat}: accuracy={accuracy:.4f} "
                f"({match_count}/{len(sample_step_ids)})"
            )

            if args.verbose and sample_records:
                print("  Sample details (up to first 10 entries):")
                for entry_idx, pred, target, is_match in sample_records[:10]:
                    status = "match" if is_match else "mismatch"
                    print(f"    idx={entry_idx:5d} pred={pred} human={target} -> {status}")

        if args.repeat > 1:
            mean_acc = mean(results)
            std_acc = stdev(results) if len(results) > 1 else 0.0
            print(f"Average accuracy across repeats: {mean_acc:.4f} ± {std_acc:.4f}")

        overall_accuracy = total_matches / total_samples if total_samples > 0 else 0.0
        print(f"Overall accuracy: {overall_accuracy:.4f} ({total_matches}/{total_samples})")
    finally:
        if policy_adapter is not None:
            policy_adapter.close()


if __name__ == "__main__":
    main()
