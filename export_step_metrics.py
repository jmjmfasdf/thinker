"""Extract per-real-step metrics from Thinker rollouts and persist them."""

import argparse
import glob
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import fig_pong


CHUNK_PATTERN = re.compile(r"^(?P<root>.+\.npz)_chunk(?P<chunk>\d+)_", re.IGNORECASE)


def _extract_chunk_metadata(file_path: str) -> Optional[Tuple[str, int]]:
    """Return (synthetic_root_path, chunk_index) when file matches chunk pattern."""
    basename = os.path.basename(file_path)
    match = CHUNK_PATTERN.match(basename)
    if not match:
        return None
    root_name = match.group("root")
    try:
        chunk_idx = int(match.group("chunk"))
    except ValueError:
        chunk_idx = 0
    synthetic_root = os.path.join(os.path.dirname(file_path), root_name)
    return synthetic_root, chunk_idx


def _chunk_sort_key(file_path: str) -> Tuple[int, str]:
    """Provide a deterministic ordering key for chunked npz files."""
    meta = _extract_chunk_metadata(file_path)
    if meta:
        return meta[1], file_path
    return 0, file_path


def _stack_chunk_arrays(chunks: Sequence[np.ndarray]) -> np.ndarray:
    """Safely concatenate chunked arrays along axis 0."""
    if not chunks:
        return np.array([])
    if len(chunks) == 1:
        return chunks[0]

    first = chunks[0]
    if isinstance(first, np.ndarray):
        if first.ndim == 0:
            return first
        try:
            return np.concatenate(chunks, axis=0)
        except Exception:
            flattened: List[Any] = []
            for chunk in chunks:
                if isinstance(chunk, np.ndarray):
                    flattened.extend(list(chunk))
                else:
                    flattened.append(chunk)
            return np.asarray(flattened, dtype=object)

    return np.asarray(chunks, dtype=object)


def _rehydrate_tree_reps(flat_data: Dict[str, Any]) -> None:
    """Reconstruct nested tree_reps dictionary from flattened keys when needed."""
    existing = flat_data.get("tree_reps")
    if isinstance(existing, dict):
        return

    prefix = "tree_reps_"
    tree_keys = [key for key in list(flat_data.keys()) if key.startswith(prefix)]
    if not tree_keys:
        return

    tree_dict: Dict[str, Any] = {}
    for key in tree_keys:
        tree_dict[key[len(prefix) :]] = flat_data.pop(key)
    flat_data["tree_reps"] = tree_dict


def _provide_vector_aliases(data: Dict[str, Any]) -> None:
    """Populate expected vector keys when only image tensors are available."""
    if "real_vectors" not in data and "real_imgs" in data:
        data["real_vectors"] = data["real_imgs"]
    if "im_vectors" not in data and "im_imgs" in data:
        data["im_vectors"] = data["im_imgs"]
    if "tree_reps" not in data or not isinstance(data["tree_reps"], dict):
        data["tree_reps"] = dict(data.get("tree_reps", {}))


def _load_npy_dict(file_path: str) -> Optional[Dict[str, Any]]:
    """Load dictionary-like data from a .npy file."""
    obj = np.load(file_path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.shape == ():
        obj = obj.item()
    elif hasattr(obj, "item") and not isinstance(obj, dict):
        try:
            obj = obj.item()
        except Exception:
            pass
    if isinstance(obj, dict):
        result = dict(obj)
        _provide_vector_aliases(result)
        return result
    return None


def _combine_npz_files(file_paths: Sequence[str]) -> Dict[str, Any]:
    """Merge one or more .npz files into a single rollout dictionary."""
    aggregated: Dict[str, List[np.ndarray]] = {}
    for path in file_paths:
        with np.load(path, allow_pickle=True) as npz_file:
            for key in npz_file.files:
                aggregated.setdefault(key, []).append(npz_file[key])

    combined: Dict[str, Any] = {}
    for key, chunks in aggregated.items():
        combined[key] = _stack_chunk_arrays(chunks)

    _rehydrate_tree_reps(combined)
    _provide_vector_aliases(combined)
    return combined


def _load_rollout_group(file_paths: Sequence[str]) -> Optional[Dict[str, Any]]:
    """Convert a logical rollout (single file or chunked) into a dictionary."""
    if not file_paths:
        return None

    first = file_paths[0]
    ext = os.path.splitext(first)[1].lower()

    if len(file_paths) == 1 and ext == ".npy":
        return _load_npy_dict(first)

    if ext == ".npz" or any(os.path.splitext(path)[1].lower() == ".npz" for path in file_paths):
        return _combine_npz_files(file_paths)

    return None


def _discover_rollout_groups(folder: str) -> List[Tuple[str, List[str]]]:
    """Identify logical rollout units, grouping chunked npz parts together."""
    npy_paths = glob.glob(os.path.join(folder, "*.npy"))
    npz_paths = glob.glob(os.path.join(folder, "*.npz"))

    chunk_groups: Dict[str, List[str]] = {}
    singles: List[str] = []

    for path in sorted(npy_paths + npz_paths):
        meta = _extract_chunk_metadata(path)
        if meta is None:
            singles.append(path)
            continue
        root_path, _ = meta
        chunk_groups.setdefault(root_path, []).append(path)

    groups: List[Tuple[str, List[str]]] = []

    for path in singles:
        groups.append((path, [path]))

    for root, paths in chunk_groups.items():
        paths.sort(key=_chunk_sort_key)
        groups.append((root, paths))

    return sorted(groups, key=lambda item: item[0])


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
    rollout_groups = _discover_rollout_groups(folder)

    if not rollout_groups:
        print(f"No .npy or .npz files found under {folder}.")
        return

    for logical_path, file_paths in rollout_groups:
        base_name = os.path.splitext(os.path.basename(logical_path))[0]
        out_name = base_name + (".npy" if fmt == "npy" else ".npz")
        out_path = os.path.join(outdir, out_name)

        try:
            data = _load_rollout_group(file_paths)
        except Exception as exc:
            print(f"Skipping {logical_path}: failed to load ({exc}).")
            continue

        if not isinstance(data, dict):
            print(f"Skipping {logical_path}: unsupported data format.")
            continue

        if "status" not in data:
            print(f"Skipping {logical_path}: missing 'status' key after loading.")
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
