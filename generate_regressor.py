"""Downsample Thinker metrics to 480 bins and export an SPM regressor set."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.io import savemat


REQUIRED_KEYS = (
    "planning_depth",
    "action_diversity",
    "imagination_diversity",
    "imagination_similarity",
    "action",
)

DEFAULT_BIN_COUNT = 480
RAW_STEP_DURATION = 1.0 / 15.0  # seconds per original sample


def load_metrics(path: Path) -> Dict[str, np.ndarray]:
    """Load the five required metric vectors from a single .npz/.npy archive."""
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            metrics = {key: np.asarray(data[key]) for key in REQUIRED_KEYS}
    elif suffix == ".npy":
        obj = np.load(path, allow_pickle=True)
        if isinstance(obj, np.ndarray) and obj.shape == ():
            obj = obj.item()
        if not isinstance(obj, dict):
            raise ValueError(f"Unexpected contents in {path}; expected dict-like metrics.")
        metrics = {key: np.asarray(obj[key]) for key in REQUIRED_KEYS}
    else:
        raise ValueError(f"Unsupported file extension for {path}")

    lengths = {key: metrics[key].shape[0] for key in REQUIRED_KEYS}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Metric lengths mismatch: {lengths}")

    return metrics


def _assign_bin_indices(num_samples: int, num_bins: int) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Return first-sample indices per bin and boolean mask for each bin."""
    sample_indices = np.arange(num_samples, dtype=np.int64)
    bin_ids = (sample_indices * num_bins) // num_samples
    bin_ids = np.clip(bin_ids, 0, num_bins - 1)

    first_indices = np.zeros(num_bins, dtype=np.int64)
    bin_masks = [None] * num_bins
    last_valid = 0
    for b in range(num_bins):
        mask = np.where(bin_ids == b)[0]
        if mask.size:
            first_indices[b] = mask[0]
            bin_masks[b] = mask
            last_valid = mask[-1]
        else:
            first_indices[b] = last_valid
            bin_masks[b] = np.array([last_valid], dtype=np.int64)
    return first_indices, bin_masks


def downsample_metrics(metrics: Dict[str, np.ndarray], num_bins: int) -> Dict[str, np.ndarray]:
    """Downsample metrics into num_bins bins."""
    num_samples = metrics["planning_depth"].shape[0]
    first_indices, bin_masks = _assign_bin_indices(num_samples, num_bins)

    binned: Dict[str, np.ndarray] = {}
    for key in REQUIRED_KEYS[:-1]:  # skip action
        values = metrics[key].reshape(-1)
        reps = values[first_indices].astype(np.float64, copy=False)
        binned[key] = reps

    actions = metrics["action"].reshape(-1)
    noop_freq = np.zeros(num_bins, dtype=np.float64)
    for b in range(num_bins):
        segment_indices = bin_masks[b]
        segment = actions[segment_indices]
        valid = segment >= 0
        total = int(valid.sum())
        if total == 0:
            noop_freq[b] = 0.0
        else:
            noop_freq[b] = float((segment[valid] == 0).sum()) / total
    binned["noop_frequency"] = noop_freq

    return binned


def build_design_matrix(binned: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Construct design matrix and names array from binned metrics."""
    columns = [
        binned["planning_depth"],
        binned["action_diversity"],
        binned["imagination_diversity"],
        binned["imagination_similarity"],
        binned["noop_frequency"],
    ]
    matrix = np.column_stack(columns).astype(np.float64, copy=False)
    names = np.array(
        [
            "planning_depth",
            "action_diversity",
            "imagination_diversity",
            "imagination_similarity",
            "noop_frequency",
        ],
        dtype=object,
    )
    return matrix, names


def drop_constant_columns(matrix: np.ndarray, names: np.ndarray, tol: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """Remove columns with near-zero variance to avoid rank deficiency."""
    stds = np.std(matrix, axis=0)
    keep = stds > tol
    if not keep.any():
        raise ValueError("All regressors were constant; nothing to save.")
    if keep.all():
        return matrix, names
    return matrix[:, keep], names[keep]


def save_spm_mat(
    matrix: np.ndarray,
    names: np.ndarray,
    output_path: Path,
    bin_duration: float,
) -> None:
    """Persist matrix in SPM-compatible multi-regressor format (R, names)."""
    num_steps = matrix.shape[0]
    step_indices = np.arange(num_steps, dtype=np.int32)
    onsets = (step_indices.astype(np.float64) * float(bin_duration)).reshape(-1, 1)
    durations = np.full((num_steps, 1), float(bin_duration), dtype=np.float64)
    names_cell = np.array(names, dtype=object).reshape(1, -1)

    payload = {
        "R": matrix,
        "names": names_cell,
        "onsets": onsets,
        "durations": durations,
        "step_indices": step_indices,
        "bin_duration": float(bin_duration),
    }
    savemat(output_path, payload, do_compression=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Downsample Thinker metrics to 480 bins and build an SPM regressor matrix."
    )
    parser.add_argument("input", type=Path, help="Path to a metrics .npz/.npy file.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination .mat file (defaults to <input>_design.mat).",
    )
    parser.add_argument(
        "--bin-count",
        type=int,
        default=DEFAULT_BIN_COUNT,
        help="Number of temporal bins to create (default: 480).",
    )
    parser.add_argument(
        "--raw-step-duration",
        type=float,
        default=RAW_STEP_DURATION,
        help="Duration (seconds) of each original sample (default: 1/15).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else input_path.with_name(f"{input_path.stem}_design.mat")
    )

    metrics = load_metrics(input_path)

    num_bins = int(args.bin_count)
    num_samples = metrics["planning_depth"].shape[0]
    total_duration = float(args.raw_step_duration) * num_samples
    bin_duration = total_duration / num_bins

    binned_metrics = downsample_metrics(metrics, num_bins)
    matrix, names = build_design_matrix(binned_metrics)
    matrix, names = drop_constant_columns(matrix, names)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_spm_mat(matrix, names, output_path, bin_duration)
    print(
        f"Wrote {matrix.shape[0]} x {matrix.shape[1]} design matrix "
        f"(bin duration {bin_duration:.4f}s) -> {output_path}"
    )


if __name__ == "__main__":
    main()
