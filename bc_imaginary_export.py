#!/usr/bin/env python3
import argparse
import os
import sys
import math

import cv2
import numpy as np
import torch
from gymnasium import spaces

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "thinker"))

from thinker import util
from thinker.actor_net import ActorNet
from thinker.model_net import ModelNet
from imitation import ThinkerPolicyAdapter


def parse_args():
    parser = argparse.ArgumentParser(description="Export Thinker rollouts from behavioral data")
    parser.add_argument("--data", required=True, help="Path to behavioral npz file")
    parser.add_argument("--preload", required=True, help="Path to checkpoint directory containing ckp_*.tar")
    parser.add_argument("--savedir", required=True, help="Directory to store the generated npy")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Computation device")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional limit on processed logical steps (each consumes frame_stack frames)",
    )
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def preprocess_frame(frame, target_size, grayscale):
    if frame.ndim == 2:
        frame = frame[..., np.newaxis]
    if frame.shape[0] != target_size[1] or frame.shape[1] != target_size[0]:
        frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        if frame.ndim == 2:
            frame = frame[..., np.newaxis]
    if grayscale:
        if frame.shape[-1] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = frame[..., np.newaxis]
    else:
        if frame.shape[-1] == 1:
            frame = np.repeat(frame, 3, axis=-1)
        elif frame.shape[-1] == 4:
            frame = frame[..., :3]
    frame = frame.astype(np.float32) / 255.0
    frame = np.transpose(frame, (2, 0, 1))
    return frame


def build_stack_for_index(
    images: np.ndarray,
    index: int,
    frame_stack: int,
    blank_frame: np.ndarray,
    target_size,
    grayscale: bool,
    episode_ids: np.ndarray,
) -> np.ndarray:
    frames = []
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
) -> tuple[list[int], list[int], list[bool], list[tuple[int, int]]]:
    """Identify frame indices that correspond to complete stacked observations."""
    is_first_bool = np.asarray(is_first, dtype=bool)
    logical_indices: list[int] = []
    prev_indices: list[int] = []
    seq_flags: list[bool] = []
    reward_ranges: list[tuple[int, int]] = []

    frames_since_reset = 0
    ep_start_ptr = -1
    total_frames = len(valid_mask)
    for idx in range(total_frames):
        new_episode = idx == 0 or is_first_bool[idx]
        if new_episode:
            frames_since_reset = 1
            ep_start_ptr = idx
        else:
            frames_since_reset += 1

        if not valid_mask[idx]:
            continue

        if frames_since_reset >= frame_stack and frames_since_reset % frame_stack == 0:
            prev_idx = idx - frame_stack if frames_since_reset > frame_stack else -1
            logical_indices.append(idx)
            prev_indices.append(prev_idx)
            seq_flags.append(prev_idx < 0)
            start = max(ep_start_ptr, idx - frame_stack + 1)
            reward_ranges.append((start, idx + 1))

    return logical_indices, prev_indices, seq_flags, reward_ranges


def convert_obs(obs_stack, model_net, device):
    obs_tensor = torch.from_numpy(obs_stack).unsqueeze(0).to(device)
    if getattr(model_net, "state_dtype_n", 0) == 0:
        return obs_tensor.to(torch.uint8)
    if obs_tensor.dtype != torch.float32:
        return (obs_tensor.float() / 255.0)
    return obs_tensor.float()




def compute_episode_ids(is_first: np.ndarray) -> np.ndarray:
    flags = is_first.astype(bool)
    episode_ids = np.zeros_like(flags, dtype=np.int64)
    current_id = -1
    for idx, flag in enumerate(flags):
        if flag or current_id < 0:
            current_id += 1
        episode_ids[idx] = current_id
    if current_id < 0:
        episode_ids[:] = 0
    return episode_ids



def extract_human_actions(actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if actions.ndim == 1:
        human_actions = actions.astype(np.int64)
        valid_mask = human_actions >= 0
    elif actions.ndim == 2:
        human_actions = np.argmax(actions, axis=1).astype(np.int64)
        valid_mask = actions.sum(axis=1) > 0.0
    else:
        raise ValueError(f"Unsupported action tensor shape: {actions.shape}")
    return human_actions, valid_mask



def decode_tree_reps_tensor(tree_reps_tensor, num_actions, dim_actions, rec_t, flags):
    enc_type = getattr(flags, "critic_enc_type", 0)
    enc_f_type = getattr(flags, "critic_enc_f_type", 0)
    decoded = util.decode_tree_reps(tree_reps_tensor.unsqueeze(0), num_actions, dim_actions, rec_t, enc_type, enc_f_type)
    return {k: v.detach().cpu().numpy() for k, v in decoded.items()}



def _squeeze_singletons(array: np.ndarray) -> np.ndarray:
    """Remove singleton dimensions beyond the batch axis while preserving data shape."""
    if array.ndim <= 1:
        return array
    squeezed = array
    # Repeatedly drop axis 1 or last axis when they are singleton to get rid of [[x]] patterns
    if squeezed.shape[1] == 1:
        squeezed = np.squeeze(squeezed, axis=1)
    if squeezed.ndim > 1 and squeezed.shape[-1] == 1:
        squeezed = np.squeeze(squeezed, axis=-1)
    return squeezed


def aggregate_tree_reps(tree_rep_entries):
    if not tree_rep_entries:
        return {}
    keys = tree_rep_entries[0].keys()
    aggregated = {}
    for key in keys:
        shapes = [entry[key].shape for entry in tree_rep_entries]
        if len({shape for shape in shapes}) != 1:
            raise ValueError(f"Mismatched shapes for key '{key}': {shapes[:5]} ... total {len(shapes)} entries")
        stacked = np.stack([entry[key] for entry in tree_rep_entries], axis=0)
        aggregated[key] = _squeeze_singletons(stacked)
    return aggregated


def convert_vector_list(vector_list):
    if not vector_list:
        return np.array([])
    if any(v is None for v in vector_list):
        return np.array(vector_list, dtype=object)
    return np.stack(vector_list).astype(np.float32)


def prepare_actor_spaces(flags, frame_stack, channels, tree_rep_size, model_net, height, width, env_n=1):
    real_shape = (env_n, frame_stack * channels, height, width)
    xs_shape = (env_n, model_net.obs_shape[0], height, width)
    hs_shape = (env_n,) + tuple(model_net.hidden_shape)

    real_states_space = spaces.Box(low=0, high=255, shape=real_shape, dtype=np.uint8)
    tree_reps_space = spaces.Box(low=-np.inf, high=np.inf, shape=(env_n, tree_rep_size), dtype=np.float32)
    xs_space = spaces.Box(low=-np.inf, high=np.inf, shape=xs_shape, dtype=np.float32)
    hs_space = spaces.Box(low=-np.inf, high=np.inf, shape=hs_shape, dtype=np.float32)

    return spaces.Dict({
        "real_states": real_states_space,
        "tree_reps": tree_reps_space,
        "xs": xs_space,
        "hs": hs_space,
    })


def main():
    args = parse_args()
    ensure_dir(args.savedir)

    config_path = os.path.abspath(os.path.join(args.preload, "config_c.yaml"))
    flags = util.create_flags(config_path, save_flags=False)
    flags.parallel = False
    flags.env_n = 1

    device = torch.device("cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu")
    if device.type != "cuda":
        flags.float16 = False

    model_ckp = os.path.join(args.preload, "ckp_model.tar")
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

    data = np.load(args.data)
    images = data["image"]
    actions = data["action"]
    if "reward" not in data:
        raise KeyError("Behavioral dataset is missing 'reward' field")
    rewards = data["reward"]
    is_first = data["is_first"]
    if "is_terminal" in data:
        is_terminal = data["is_terminal"]
    elif "done" in data:
        is_terminal = data["done"]
    else:
        raise KeyError("Behavioral dataset is missing 'is_terminal' or 'done' field")

    target_h, target_w = images.shape[1], images.shape[2]
    raw_channels = images.shape[3]
    if raw_channels == 1:
        raw_channels = 3
    action_dim = actions.shape[1]

    frame_stack = flags.frame_stack_n
    channels_per_frame = 1 if flags.grayscale else raw_channels
    total_channels = channels_per_frame * frame_stack

    model_frame_stack = frame_stack
    if model_checkpoint is not None:
        state_dict = model_checkpoint.get("model_net_state_dict", {})
        if os.environ.get("BC_DEBUG"):
            print("[DEBUG] total_channels:", total_channels)
        decoder_weight = state_dict.get("sr_net.encoder.d_conv.13.weight")
        if decoder_weight is None:
            for key in state_dict.keys():
                if "sr_net.encoder.d_conv" in key and key.endswith("weight"):
                    candidate = state_dict[key]
                    if candidate.dim() == 4 and candidate.shape[2:] == (4, 4):
                        if os.environ.get("BC_DEBUG"):
                            print("[DEBUG] candidate key", key, "shape", tuple(candidate.shape))
                        decoder_weight = candidate
                        break
        if decoder_weight is not None:
            decoder_channels = decoder_weight.shape[1]
            if os.environ.get("BC_DEBUG"):
                print("[DEBUG] decoder_channels:", decoder_channels)
            if decoder_channels % channels_per_frame == 0:
                inferred = max(1, (frame_stack * channels_per_frame) // decoder_channels)
                if os.environ.get("BC_DEBUG"):
                    print("[DEBUG] inferred frame_stack_n from weights:", inferred)
                model_frame_stack = inferred

    model_obs_space = spaces.Box(low=0, high=255, shape=(total_channels, target_h, target_w), dtype=np.uint8)
    primary_action_space = spaces.Discrete(action_dim)

    model_net = ModelNet(obs_space=model_obs_space, action_space=primary_action_space, flags=flags, frame_stack_n=model_frame_stack).to(device)
    if os.environ.get("BC_DEBUG"):
        weight = model_net.sr_net.encoder.d_conv[13].weight
        print("[DEBUG] Init sr decoder weight shape:", tuple(weight.shape))
        print("[DEBUG] frame_stack_n:", model_frame_stack, "obs_shape:", model_net.obs_shape)
    model_net.eval()

    if model_checkpoint is not None:
        model_net.set_weights(model_checkpoint["model_net_state_dict"])

    num_actions, dim_actions, dim_rep_actions, _, _ = util.process_action_space(primary_action_space)

    tree_rep_size = 11 + num_actions * 10 + flags.rec_t
    if getattr(flags, "has_action_seq", False) and getattr(flags, "reset_mode", 0) == 0:
        tree_rep_size += flags.max_depth * num_actions + num_actions

    actor_obs_space = prepare_actor_spaces(flags, frame_stack, channels_per_frame, tree_rep_size, model_net, target_h, target_w)
    reset_action_space = spaces.Discrete(2)
    actor_action_space = spaces.Tuple((spaces.Tuple((primary_action_space,)), spaces.Tuple((reset_action_space,))))
    tree_rep_meaning = util.slice_tree_reps(num_actions, dim_actions, flags.rec_t)

    actor_net = ActorNet(obs_space=actor_obs_space, action_space=actor_action_space, flags=flags, tree_rep_meaning=tree_rep_meaning).to(device)
    actor_net.eval()

    actor_ckp = os.path.join(args.preload, "ckp_actor.tar")
    if os.path.isfile(actor_ckp):
        checkpoint = torch.load(actor_ckp, map_location=device, weights_only=False)
        actor_net.set_weights(checkpoint["actor_net_state_dict"])

    policy_adapter = ThinkerPolicyAdapter(actor_net, model_net, flags, device)

    blank_frame = np.zeros((channels_per_frame, target_h, target_w), dtype=np.float32)

    video_stats = {
        "real_imgs": [],
        "im_imgs": [],
        "status": [],
        "tree_reps": [],
        "real_vectors": [],
        "im_vectors": [],
        "im_vp_vectors": [],
        "env_return": [],
        "cur_rewards": [],
        "step_times": [],
        "tree_reps_vector": [],
        "human_action": [],
        "imagined_real_action": [],
    }

    episode_ids = compute_episode_ids(is_first)
    human_actions, valid_mask = extract_human_actions(actions)

    stack_indices, prev_indices, seq_flags, reward_ranges = compute_stack_step_metadata(
        valid_mask=valid_mask,
        is_first=is_first,
        frame_stack=frame_stack,
    )
    total_logical_steps = len(stack_indices)
    if total_logical_steps == 0:
        print("[WARNING] No complete frame stacks found; nothing to export")
        policy_adapter.close()
        return

    if args.max_steps is not None:
        total_logical_steps = min(args.max_steps, total_logical_steps)

    processed_steps = 0

    for step_idx in range(total_logical_steps):
        next_idx = stack_indices[step_idx]
        prev_idx = prev_indices[step_idx]
        sequence_start_flag = seq_flags[step_idx]
        reward_start, reward_end = reward_ranges[step_idx]
        reward_slice = rewards[reward_start:reward_end]
        reward_sum = float(np.sum(reward_slice)) if reward_slice.size > 0 else 0.0
        clipped_reward = float(np.clip(reward_sum, -1.0, 1.0))
        cur_reward_vec = np.array([clipped_reward], dtype=np.float32)

        stack_input = build_stack_for_index(
            images,
            next_idx,
            frame_stack,
            blank_frame,
            (target_w, target_h),
            flags.grayscale,
            episode_ids,
        )
        obs_tensor = torch.from_numpy(stack_input).unsqueeze(0).to(device=device)

        prev_action_val = 0
        if prev_idx >= 0 and valid_mask[prev_idx]:
            prev_action_val = int(human_actions[prev_idx])
        prev_action_tensor = torch.tensor([prev_action_val], dtype=torch.long, device=device)

        sequence_start_tensor = torch.tensor([sequence_start_flag], dtype=torch.bool, device=device)

        teacher_action_val = int(human_actions[next_idx])
        teacher_tensor = torch.tensor([teacher_action_val], dtype=torch.long, device=device)

        with torch.no_grad():
            _ = policy_adapter.forward(
                obs_tensor,
                actions=teacher_tensor,
                prev_actions=prev_action_tensor,
                sequence_starts=sequence_start_tensor,
                requires_grad=False,
                record_history=True,
            )

        history = policy_adapter.last_rollout_history or [[]]
        entries = history[0]
        imagined_actions = []

        for entry_idx, entry in enumerate(entries):
            tree_rep_tensor = torch.from_numpy(entry["tree_reps"]).float()
            decoded_tree = decode_tree_reps_tensor(tree_rep_tensor, num_actions, dim_actions, flags.rec_t, flags)
            # status alignment with visual2: 0 real, 2 imagination, 1 reset, 3 force reset
            status_val = int(entry["status"])
            cur_reset_val = decoded_tree.get("cur_reset", None)
            if cur_reset_val is not None:
                cur_reset_scalar = int(np.asarray(cur_reset_val).reshape(-1)[0])
                if cur_reset_scalar == 1:
                    status_val = 1
                elif cur_reset_scalar == 3:
                    status_val = 3
            if entry_idx == 0:
                status_val_out = 0
            else:
                if status_val in (0, 3):
                    status_val_out = 0 if status_val == 0 else 3
                elif status_val in (1, 2):
                    status_val_out = 2
                else:
                    status_val_out = status_val
            env_ret_val = reward_sum if status_val_out == 0 else 0.0
            cur_reward_val = cur_reward_vec if status_val_out == 0 else np.zeros_like(cur_reward_vec)
            video_stats["real_imgs"].append(entry["real_img"].astype(np.uint8))
            video_stats["im_imgs"].append(entry["im_img"].astype(np.float32))
            video_stats["status"].append(status_val_out)
            video_stats["tree_reps"].append(decoded_tree)
            video_stats["real_vectors"].append(entry["real_vectors"] if entry["real_vectors"] is not None else None)
            video_stats["im_vectors"].append(entry["im_vectors"] if entry["im_vectors"] is not None else None)
            video_stats["im_vp_vectors"].append(entry["im_vp_vectors"] if entry.get("im_vp_vectors") is not None else None)
            video_stats["env_return"].append(env_ret_val)
            video_stats["cur_rewards"].append(cur_reward_val)
            video_stats["step_times"].append(entry.get("step_times"))
            video_stats["tree_reps_vector"].append(entry.get("tree_reps_vector"))
            video_stats["human_action"].append(int(entry["human_action"]))
            imagined_actions.append(entry["imagined_action"] if entry["imagined_action"] is not None else -1)

        video_stats["imagined_real_action"].extend(imagined_actions)

        processed_steps = step_idx + 1
        if processed_steps % 100 == 0 or processed_steps == 1:
            print(f"[INFO] Processing logical step {processed_steps}/{total_logical_steps}")

    if processed_steps == 0:
        print("[WARNING] No logical steps processed; nothing to save")
        policy_adapter.close()
        return


    video_stats["real_imgs"] = np.stack(video_stats["real_imgs"], axis=0)
    video_stats["im_imgs"] = np.stack(video_stats["im_imgs"], axis=0)
    video_stats["status"] = np.array(video_stats["status"], dtype=np.int32)
    video_stats["env_return"] = np.array(video_stats["env_return"], dtype=np.float32)
    if len(video_stats["cur_rewards"]) > 0 and video_stats["cur_rewards"][0] is not None:
        video_stats["cur_rewards"] = np.stack(video_stats["cur_rewards"], axis=0).astype(np.float32)
    else:
        video_stats["cur_rewards"] = None
    if len(video_stats["step_times"]) > 0:
        valid_step = next((st for st in video_stats["step_times"] if st is not None), None)
        if valid_step is not None:
            fill = np.full_like(np.asarray(valid_step, dtype=np.float32), np.nan, dtype=np.float32)
            video_stats["step_times"] = np.stack(
                [
                    np.asarray(st, dtype=np.float32) if st is not None else fill
                    for st in video_stats["step_times"]
                ],
                axis=0,
            )
        else:
            video_stats["step_times"] = None
    else:
        video_stats["step_times"] = None
    if len(video_stats["tree_reps_vector"]) > 0:
        valid_tree = next((tv for tv in video_stats["tree_reps_vector"] if tv is not None), None)
        if valid_tree is not None:
            fill_tree = np.full_like(np.asarray(valid_tree, dtype=np.float32), np.nan, dtype=np.float32)
            video_stats["tree_reps_vector"] = np.stack(
                [
                    np.asarray(tv, dtype=np.float32) if tv is not None else fill_tree
                    for tv in video_stats["tree_reps_vector"]
                ],
                axis=0,
            )
        else:
            video_stats["tree_reps_vector"] = None
    else:
        video_stats["tree_reps_vector"] = None
    video_stats["tree_reps"] = aggregate_tree_reps(video_stats["tree_reps"])
    video_stats["real_vectors"] = convert_vector_list(video_stats["real_vectors"])
    video_stats["im_vectors"] = convert_vector_list(video_stats["im_vectors"])
    video_stats["im_vp_vectors"] = convert_vector_list(video_stats["im_vp_vectors"])
    video_stats["human_action"] = np.array(video_stats["human_action"], dtype=np.int32)
    video_stats["imagined_real_action"] = np.array(video_stats["imagined_real_action"], dtype=np.int32)

    total_entries = video_stats["real_imgs"].shape[0]
    parts = 10
    part_size = max(1, math.ceil(total_entries / parts))
    base_name = os.path.basename(args.data)
    saved_parts = 0

    for part in range(parts):
        start = part * part_size
        end = min(total_entries, (part + 1) * part_size)
        if start >= end:
            break

        part_stats = {
            "real_imgs": video_stats["real_imgs"][start:end],
            "im_imgs": video_stats["im_imgs"][start:end],
            "status": video_stats["status"][start:end],
            "tree_reps": {k: v[start:end] for k, v in video_stats["tree_reps"].items()},
            "real_vectors": video_stats["real_vectors"][start:end],
            "im_vectors": video_stats["im_vectors"][start:end],
            "im_vp_vectors": video_stats["im_vp_vectors"][start:end],
            "env_return": video_stats["env_return"][start:end],
            "cur_rewards": video_stats["cur_rewards"][start:end] if video_stats["cur_rewards"] is not None else None,
            "step_times": video_stats["step_times"][start:end] if video_stats["step_times"] is not None else None,
            "tree_reps_vector": video_stats["tree_reps_vector"][start:end] if video_stats["tree_reps_vector"] is not None else None,
            "human_action": video_stats["human_action"][start:end],
            "imagined_real_action": video_stats["imagined_real_action"][start:end],
        }

        output_name = f"{base_name}_part{part + 1}of{parts}.npy"
        output_path = os.path.join(args.savedir, output_name)
        np.save(output_path, part_stats, allow_pickle=True)
        saved_parts += 1
        print(f"Saved {output_name} [{start}:{end}]")

    if saved_parts == 0:
        print("[WARNING] No data chunks were saved")
    else:
        print(f"Saved {saved_parts} chunk(s) to {args.savedir}")

    policy_adapter.close()


if __name__ == "__main__":
    main()
