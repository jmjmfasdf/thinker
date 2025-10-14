#!/usr/bin/env python3
import argparse
import os
import sys
from types import SimpleNamespace
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
from python_tree import TreeManager


def parse_args():
    parser = argparse.ArgumentParser(description="Export Thinker rollouts from behavioral data")
    parser.add_argument("--data", required=True, help="Path to behavioral npz file")
    parser.add_argument("--preload", required=True, help="Path to checkpoint directory containing ckp_*.tar")
    parser.add_argument("--savedir", required=True, help="Directory to store the generated npy")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Computation device")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional limit on processed steps")
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def preprocess_frame(frame, target_size, grayscale):
    if frame.shape[:2] != target_size:
        frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
    if grayscale and frame.shape[-1] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = frame[..., np.newaxis]
    frame = frame.astype(np.uint8)
    frame = np.transpose(frame, (2, 0, 1))
    return frame


def build_frame_stack(history, frame_stack, blank_frame):
    frames = []
    missing = frame_stack - len(history)
    for _ in range(missing):
        frames.append(blank_frame)
    frames.extend(list(history))
    return np.concatenate(frames, axis=0)


def clone_state(state):
    if isinstance(state, dict):
        return {k: (v.clone() if hasattr(v, "clone") else v) for k, v in state.items()}
    if hasattr(state, "clone"):
        return state.clone()
    return state


def convert_obs(obs_stack, model_net, device):
    obs_tensor = torch.from_numpy(obs_stack).unsqueeze(0).to(device)
    if getattr(model_net, "state_dtype_n", 0) == 0:
        return obs_tensor.to(torch.uint8)
    if obs_tensor.dtype != torch.float32:
        return (obs_tensor.float() / 255.0)
    return obs_tensor.float()


def extract_vectors(model_net, obs_tensor, model_state):
    real_vec = None
    im_vec = None
    if hasattr(model_net, "sr_net"):
        with torch.no_grad():
            sr = model_net.sr_net
            real_state = obs_tensor.unsqueeze(0).unsqueeze(0)
            real_state_norm = model_net.normalize(real_state)
            dummy_action = torch.zeros(1, 1, sr.dim_rep_actions, device=obs_tensor.device)
            dummy_done = torch.zeros(1, 1, dtype=torch.bool, device=obs_tensor.device)
            real_vec, _ = sr.encoder(real_state_norm, dummy_done, dummy_action, {}, flatten=True)
            real_vec = real_vec.squeeze(0).detach().cpu().numpy()
            if isinstance(model_state, dict) and "sr_h" in model_state:
                im_vec = model_state["sr_h"].detach().cpu().numpy()
    if im_vec is None and real_vec is not None:
        im_vec = np.copy(real_vec)
    return real_vec, im_vec


def decode_tree_reps_tensor(tree_reps_tensor, num_actions, dim_actions, rec_t, flags):
    enc_type = getattr(flags, "critic_enc_type", 0)
    enc_f_type = getattr(flags, "critic_enc_f_type", 0)
    decoded = util.decode_tree_reps(tree_reps_tensor.unsqueeze(0), num_actions, dim_actions, rec_t, enc_type, enc_f_type)
    return {k: v.detach().cpu().numpy() for k, v in decoded.items()}


def record_entry(entries, status, real_img, im_img, tree_reps, real_vec, im_vec, human_action):
    entry = {
        "status": int(status),
        "real_img": real_img,
        "im_img": im_img,
        "tree_reps": tree_reps,
        "real_vectors": real_vec,
        "im_vectors": im_vec,
        "human_action": int(human_action),
    }
    entries.append(entry)


def run_planning_step(
    obs_stack,
    dataset_action_idx,
    prev_action_idx,
    prev_reward_value,
    prev_done_flag,
    model_net,
    actor_net,
    flags,
    device,
    num_actions,
    dim_actions,
    copy_ch,
):
    obs_tensor = convert_obs(obs_stack, model_net, device)
    current_obs = obs_tensor.float() if obs_tensor.dtype == torch.uint8 else obs_tensor

    with torch.no_grad():
        initial_state = model_net.initial_state(batch_size=1, device=device)
        initial_out = model_net.forward(
            env_state=obs_tensor,
            actions=torch.zeros(1, 1, 1, dtype=torch.long, device=device),
            done=torch.zeros(1, dtype=torch.bool, device=device),
            state=initial_state,
        )

    model_state = clone_state(initial_out.state)
    initial_model_state = clone_state(model_state)
    initial_xs = initial_out.xs[0] if initial_out.xs is not None else None
    initial_hs = initial_out.hs[0] if initial_out.hs is not None else None

    if initial_out.policy is not None:
        init_policy = initial_out.policy[0]
        if init_policy.dim() == 3:
            init_policy = init_policy.squeeze(1)
    else:
        init_policy = torch.full((1, num_actions), 1.0 / num_actions, device=device)

    if initial_out.vs is not None:
        init_values = initial_out.vs[0]
        if init_values.dim() == 2:
            init_values = init_values.squeeze(-1)
    else:
        init_values = torch.zeros(1, device=device)

    init_rewards = None
    if hasattr(initial_out, "rs") and initial_out.rs is not None:
        init_rewards = initial_out.rs[0]

    tree_manager = TreeManager(batch_size=1, num_actions=num_actions, flags=flags, device=device)
    root_payload = {"real_states": obs_tensor[0]}
    if initial_xs is not None:
        root_payload["xs"] = initial_xs[0]
    if initial_hs is not None:
        root_payload["hs"] = initial_hs[0]
    tree_manager.expand_root(init_rewards, init_values, init_policy, [root_payload])
    tree_reps = tree_manager.compute_tree_reps()

    frame_ch = obs_stack.shape[0] // flags.frame_stack_n
    last_frame = obs_stack[-copy_ch:, :, :]
    if initial_xs is not None:
        im_root = torch.clamp(initial_xs[0], 0.0, 1.0).detach().cpu().numpy()
        im_root = im_root[-copy_ch:, :, :]
    else:
        im_root = np.zeros_like(last_frame, dtype=np.float32)

    real_vec, im_vec = extract_vectors(model_net, obs_tensor[0], initial_model_state)

    done_tensor = torch.full((1, 1), bool(prev_done_flag), dtype=torch.bool, device=device)

    entries = []
    imagined_actions = []
    decoded_root = decode_tree_reps_tensor(tree_reps, num_actions, dim_actions, flags.rec_t, flags)
    dataset_onehot = np.zeros(num_actions, dtype=np.float32)
    dataset_onehot[dataset_action_idx] = 1.0

    root_shape = decoded_root["root_action"].shape
    cur_shape = decoded_root["cur_action"].shape
    root_onehot = np.zeros(root_shape, dtype=np.float32)
    root_onehot.reshape(-1)[:] = dataset_onehot
    cur_onehot = np.zeros(cur_shape, dtype=np.float32)
    cur_onehot.reshape(-1)[:] = dataset_onehot
    decoded_root["root_action"] = root_onehot
    decoded_root["cur_action"] = cur_onehot
    record_entry(entries, 0, last_frame.copy(), im_root.copy(), decoded_root, real_vec, im_vec, dataset_action_idx)
    imagined_actions.append(99)
    root_entry_index = 0

    current_xs = initial_xs.clone() if initial_xs is not None else None
    current_hs = initial_hs.clone() if initial_hs is not None else None

    reward_dim = 1 + int(flags.im_cost > 0.0) + int(flags.cur_cost > 0.0)
    actor_state = actor_net.initial_state(batch_size=1, device=device)

    last_action = torch.tensor([prev_action_idx], dtype=torch.long, device=device)
    last_reset = torch.zeros(1, dtype=torch.long, device=device)
    final_action_idx = prev_action_idx

    for step in range(flags.rec_t - 1):
        with torch.no_grad():
            step_out = model_net.forward_single(state=model_state, action=last_action, training=False)
        xs = step_out.xs[0] if step_out.xs is not None else None
        hs = step_out.hs[0] if step_out.hs is not None else None

        if xs is not None:
            current_xs = xs.clone()
        if hs is not None:
            current_hs = hs.clone()

        model_state = clone_state(step_out.state)

        encoded_payload = [{}]
        if xs is not None:
            encoded_payload[0]["xs"] = xs[0]
        if hs is not None:
            encoded_payload[0]["hs"] = hs[0]

        logits = step_out.policy[0] if step_out.policy is not None else torch.zeros(1, num_actions, device=device)
        if logits.dim() == 3:
            logits = logits.squeeze(1)

        if hasattr(step_out, "rs") and step_out.rs is not None:
            rewards_step = step_out.rs[0]
            if rewards_step.dim() == 2:
                rewards_step = rewards_step.squeeze(1)
        else:
            rewards_step = torch.zeros(1, device=device)

        if step_out.vs is not None:
            values_step = step_out.vs[0]
            if values_step.dim() == 2:
                values_step = values_step.squeeze(1)
        else:
            values_step = torch.zeros(1, device=device)

        if hasattr(step_out, "dones") and step_out.dones is not None:
            dones_step = step_out.dones[0]
            if dones_step.dim() == 2:
                dones_step = dones_step.squeeze(1)
        else:
            dones_step = torch.zeros(1, dtype=torch.bool, device=device)

        tree_manager.expand_current(rewards_step, values_step, dones_step, logits, encoded_payload)

        step_status = 2 if step == flags.rec_t - 2 else 1
        env_out = SimpleNamespace()
        env_out.real_states = current_obs.unsqueeze(0)
        env_out.tree_reps = tree_manager.compute_tree_reps().unsqueeze(0)
        env_out.xs = current_xs.unsqueeze(0) if current_xs is not None else None
        env_out.hs = current_hs.unsqueeze(0) if current_hs is not None else None
        env_out.step_status = torch.full((1, 1), step_status, dtype=torch.long, device=device)
        env_out.done = done_tensor.clone()
        env_out.real_done = done_tensor.clone()
        env_out.last_pri = last_action.unsqueeze(0)
        env_out.last_reset = last_reset.unsqueeze(0)
        env_out.reward = torch.zeros(1, 1, reward_dim, device=device)
        if step == 0:
            env_out.reward[0, 0, 0] = float(prev_reward_value)

        with torch.no_grad():
            actor_out, actor_state = actor_net.forward(env_out=env_out, core_state=actor_state)
        action_probs = actor_out.action_prob[0]
        next_action = torch.argmax(action_probs, dim=-1)

        reset_action = torch.zeros(1, dtype=torch.bool, device=device)
        if hasattr(actor_out, "reset") and actor_out.reset is not None:
            reset_probs = actor_out.reset[0]
            reset_action = reset_probs > 0.5
        force_reset = torch.zeros(1, dtype=torch.bool, device=device)
        if getattr(flags, "max_depth", 0) > 0:
            force_reset = tree_manager.rollout_depth >= flags.max_depth
        should_reset = reset_action | force_reset

        if should_reset.any():
            tree_manager.rollout_depth[should_reset] = 0
            last_reset[should_reset] = 1
            if current_xs is not None and initial_xs is not None:
                current_xs[should_reset] = initial_xs[should_reset]
            if current_hs is not None and initial_hs is not None:
                current_hs[should_reset] = initial_hs[should_reset]
            if isinstance(model_state, dict) and isinstance(initial_model_state, dict):
                for key in model_state:
                    if key in initial_model_state and hasattr(model_state[key], "clone"):
                        model_state[key][should_reset] = initial_model_state[key][should_reset].clone()
            tree_manager.advance(next_action, resets=should_reset)
        else:
            last_reset.zero_()
            tree_manager.advance(next_action)

        tree_reps = tree_manager.compute_tree_reps(reset_flags=should_reset)

        current_im = None
        if current_xs is not None:
            current_im = torch.clamp(current_xs[0], 0.0, 1.0).detach().cpu().numpy()
            current_im = current_im[-copy_ch:, :, :]
        else:
            current_im = np.zeros_like(last_frame, dtype=np.float32)

        status_value = 2
        if force_reset.any():
            status_value = 3
        elif reset_action.any():
            status_value = 1

        real_vec_step, im_vec_step = extract_vectors(model_net, obs_tensor[0], model_state)
        decoded = decode_tree_reps_tensor(tree_reps, num_actions, dim_actions, flags.rec_t, flags)
        record_entry(entries, status_value, last_frame.copy(), current_im.copy(), decoded, real_vec_step, im_vec_step, 99)
        imagined_actions.append(99)

        final_action_idx = int(next_action.item())
        last_action = next_action

    tree_manager.reset_real_step()
    final_tree = tree_manager.compute_tree_reps()
    env_out = SimpleNamespace()
    env_out.real_states = current_obs.unsqueeze(0)
    env_out.tree_reps = final_tree.unsqueeze(0)
    env_out.xs = current_xs.unsqueeze(0) if current_xs is not None else None
    env_out.hs = current_hs.unsqueeze(0) if current_hs is not None else None
    env_out.step_status = torch.zeros((1, 1), dtype=torch.long, device=device)
    env_out.done = done_tensor.clone()
    env_out.real_done = done_tensor.clone()
    env_out.last_pri = last_action.unsqueeze(0)
    env_out.last_reset = last_reset.unsqueeze(0)
    env_out.reward = torch.zeros(1, 1, reward_dim, device=device)
    with torch.no_grad():
        _ = actor_net.forward(env_out=env_out, core_state=actor_state)

    if 0 <= final_action_idx < num_actions:
        imagined_actions[root_entry_index] = final_action_idx

    return entries, imagined_actions


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

    blank_frame = np.zeros((channels_per_frame, target_h, target_w), dtype=np.uint8)
    copy_ch = channels_per_frame

    video_stats = {
        "real_imgs": [],
        "im_imgs": [],
        "status": [],
        "tree_reps": [],
        "real_vectors": [],
        "im_vectors": [],
        "human_action": [],
        "imagined_real_action": [],
    }

    total_frames = len(images)
    limit = args.max_steps if args.max_steps is not None else total_frames

    estimated_steps = 0
    frames_in_episode_for_est = 0
    for idx in range(total_frames):
        if is_first[idx]:
            if frames_in_episode_for_est > 0:
                estimated_steps += math.ceil(frames_in_episode_for_est / frame_stack)
            frames_in_episode_for_est = 0
        frames_in_episode_for_est += 1
    if frames_in_episode_for_est > 0:
        estimated_steps += math.ceil(frames_in_episode_for_est / frame_stack)

    processed_steps = 0
    current_frames = []
    current_actions = []
    current_rewards = []
    current_dones = []
    prev_action_idx = 0
    prev_reward_value = 0.0
    prev_done_flag = False

    def emit_stack():
        nonlocal current_frames, current_actions, current_rewards, current_dones, processed_steps
        nonlocal prev_action_idx, prev_reward_value, prev_done_flag
        if not current_frames:
            return

        frames_to_stack = list(current_frames)
        stack_input = build_frame_stack(frames_to_stack, frame_stack, blank_frame)
        dataset_action = current_actions[-1] if current_actions else 0
        reward_sum = float(np.sum(current_rewards)) if current_rewards else 0.0
        done_flag = bool(current_dones[-1]) if current_dones else False

        entries, imagined_actions = run_planning_step(
            stack_input,
            dataset_action,
            prev_action_idx,
            prev_reward_value,
            prev_done_flag,
            model_net,
            actor_net,
            flags,
            device,
            num_actions,
            dim_actions,
            copy_ch,
        )
        for entry in entries:
            video_stats["real_imgs"].append(entry["real_img"].astype(np.uint8))
            video_stats["im_imgs"].append(entry["im_img"].astype(np.float32))
            video_stats["status"].append(entry["status"])
            video_stats["tree_reps"].append(entry["tree_reps"])
            video_stats["real_vectors"].append(entry["real_vectors"] if entry["real_vectors"] is not None else None)
            video_stats["im_vectors"].append(entry["im_vectors"] if entry["im_vectors"] is not None else None)
            video_stats["human_action"].append(entry["human_action"])
        video_stats["imagined_real_action"].extend(imagined_actions)

        processed_steps += 1
        if processed_steps % 100 == 0 or processed_steps == 1:
            denom = estimated_steps if estimated_steps > 0 else "?"
            print(f"[INFO] Processing logical step {processed_steps}/{denom}")

        prev_action_idx = dataset_action
        prev_reward_value = reward_sum
        prev_done_flag = done_flag
        current_frames = []
        current_actions = []
        current_rewards = []
        current_dones = []

    for idx in range(min(total_frames, limit)):
        if is_first[idx] and current_frames:
            emit_stack()
        if is_first[idx]:
            current_frames = []
            current_actions = []
            current_rewards = []
            current_dones = []
            prev_action_idx = 0
            prev_reward_value = 0.0
            prev_done_flag = False

        frame = preprocess_frame(images[idx], (target_w, target_h), flags.grayscale)
        current_frames.append(frame)
        current_actions.append(int(np.argmax(actions[idx])))
        current_rewards.append(float(rewards[idx]))
        current_dones.append(bool(is_terminal[idx]))

        if len(current_frames) == frame_stack:
            emit_stack()

    if current_frames:
        emit_stack()

    if len(video_stats["real_imgs"]) == 0:
        print("[WARNING] No logical steps processed; nothing to save")
        return

    video_stats["real_imgs"] = np.stack(video_stats["real_imgs"], axis=0)
    video_stats["im_imgs"] = np.stack(video_stats["im_imgs"], axis=0)
    video_stats["status"] = np.array(video_stats["status"], dtype=np.int32)
    video_stats["tree_reps"] = aggregate_tree_reps(video_stats["tree_reps"])
    video_stats["real_vectors"] = convert_vector_list(video_stats["real_vectors"])
    video_stats["im_vectors"] = convert_vector_list(video_stats["im_vectors"])
    video_stats["human_action"] = np.array(video_stats["human_action"], dtype=np.int32)
    video_stats["imagined_real_action"] = np.array(video_stats["imagined_real_action"], dtype=np.int32)

    total_entries = video_stats["real_imgs"].shape[0]
    parts = 5
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


if __name__ == "__main__":
    main()
