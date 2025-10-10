"""Extract per-real-step metrics from Thinker rollouts and persist them."""

import argparse
import glob
import os
from typing import Dict

import numpy as np

import fig_pong


def _ensure_1d_status(status: np.ndarray) -> np.ndarray:
    """Flatten status array safely into 1-D."""
    status_arr = np.asarray(status)
    if status_arr.ndim == 1:
        return status_arr
    return status_arr.reshape(-1)


def extract_step_metrics(data: Dict) -> Dict[str, np.ndarray]:
    """Compute per real-step metrics required for analysis export."""
    status = _ensure_1d_status(data.get("status", []))
    real_indices = np.where(status == 0)[0]
    num_real_steps = len(real_indices)

    empty_float = np.empty((0,), dtype=np.float32)
    empty_int = np.empty((0,), dtype=np.int32)
    has_cur_reward = "cur_rewards" in data and data.get("cur_rewards") is not None

    if num_real_steps == 0:
        metrics = {
            "planning_depth": empty_float,
            "action_diversity": empty_float,
            "imagination_diversity": empty_float,
            "imagination_similarity": empty_float,
            "action": empty_int,
        }
        if has_cur_reward:
            metrics["cur_reward"] = empty_float
        return metrics

    fragment_results = fig_pong.analyze_imaginary_image_similarity(
        data, similarity_mode="cosine"
    )
    imagination_diversity_dict = fig_pong.aa_analyze_imagination_diversity_by_real_step(data)

    # Accumulate per-step stats using real-index lookups for clarity.
    depth_lists = [[] for _ in range(num_real_steps)]
    similarity_lists = [[] for _ in range(num_real_steps)]
    idx_to_slot = {idx: slot for slot, idx in enumerate(real_indices)}

    for fragment in fragment_results:
        slot = idx_to_slot.get(fragment["source_real_idx"])
        if slot is None:
            continue
        depth_lists[slot].append(fragment.get("fragment_length", np.nan))
        similarity = fragment.get("similarity", np.nan)
        if np.isfinite(similarity):
            similarity_lists[slot].append(float(similarity))

    planning_depth = np.array(
        [np.mean(depth) if depth else np.nan for depth in depth_lists], dtype=np.float32
    )
    imagination_similarity = np.array(
        [np.mean(sim) if sim else np.nan for sim in similarity_lists], dtype=np.float32
    )
    imagination_diversity = np.array(
        [
            float(imagination_diversity_dict.get(int(idx), np.nan))
            for idx in real_indices
        ],
        dtype=np.float32,
    )

    action_diversity = fig_pong.calculate_action_diversities(data, real_indices)
    action_diversity = action_diversity.astype(np.float32, copy=False)

    cur_actions = (
        data.get("tree_reps", {}).get("cur_action", [])
        if isinstance(data.get("tree_reps"), dict)
        else []
    )
    actions = np.full(num_real_steps, -1, dtype=np.int32)
    for slot, idx in enumerate(real_indices):
        if len(cur_actions) > int(idx):
            one_hot = cur_actions[int(idx)]
            try:
                actions[slot] = int(np.argmax(one_hot))
            except Exception:
                actions[slot] = -1

    metrics = {
        "planning_depth": planning_depth,
        "action_diversity": action_diversity,
        "imagination_diversity": imagination_diversity,
        "imagination_similarity": imagination_similarity,
        "action": actions,
    }

    if has_cur_reward:
        cur_reward_source = data.get("cur_rewards")
        cur_reward_per_step = np.full(num_real_steps, np.nan, dtype=np.float32)
        try:
            flattened_curr = np.asarray(cur_reward_source, dtype=np.float32).reshape(-1)
        except Exception:
            flattened_curr = np.asarray(cur_reward_source).reshape(-1)
            try:
                flattened_curr = flattened_curr.astype(np.float32, copy=False)
            except Exception:
                flattened_curr = flattened_curr.astype(float, copy=False)

        for slot, idx in enumerate(real_indices):
            idx_int = int(idx)
            if 0 <= idx_int < flattened_curr.shape[0]:
                value = flattened_curr[idx_int]
                if np.isfinite(value):
                    cur_reward_per_step[slot] = np.float32(value)

        metrics["cur_reward"] = cur_reward_per_step

    return metrics


def save_metrics(metrics: Dict[str, np.ndarray], out_path: str, fmt: str) -> None:
    """Persist metrics dictionary using the requested numpy container."""
    if fmt == "npy":
        np.save(out_path, metrics, allow_pickle=True)
    else:
        np.savez(out_path, **metrics)


def process_folder(folder: str, outdir: str, fmt: str) -> None:
    """Walk through every rollout file in folder and write metrics summary."""
    os.makedirs(outdir, exist_ok=True)
    npy_files = sorted(glob.glob(os.path.join(folder, "*.npy")))

    if not npy_files:
        print(f"No .npy files found under {folder}.")
        return

    for file_path in npy_files:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        out_name = base_name + (".npy" if fmt == "npy" else ".npz")
        out_path = os.path.join(outdir, out_name)

        try:
            data = np.load(file_path, allow_pickle=True).item()
        except Exception as exc:
            print(f"Skipping {file_path}: failed to load ({exc}).")
            continue

        metrics = extract_step_metrics(data)
        save_metrics(metrics, out_path, fmt)
        print(f"Saved metrics for {base_name} -> {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export per-real-step Thinker metrics for each rollout file."
    )
    parser.add_argument("--folder", required=True, help="Input folder containing .npy rollouts")
    parser.add_argument(
        "--outdir", required=True, help="Destination folder for metric files"
    )
    parser.add_argument(
        "--format",
        choices=("npz", "npy"),
        default="npz",
        help="Output container format (default: npz)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_folder(args.folder, args.outdir, args.format)


if __name__ == "__main__":
    main()
