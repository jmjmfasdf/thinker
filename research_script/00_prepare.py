#!/usr/bin/env python3
"""
Prepare RAM-based DSMs for representational similarity analysis.

This script reads every behavioral_data_block NPZ, applies a canonical HRF to
selected frame-level Atari RAM bytes, bins the convolved features to 1-second
TRs assuming an 8-minute gameplay block, and writes one RAM DSM NPZ per input
NPZ while preserving the input directory structure.

Default output:
  research_script/outputs/00_prepare/
    behavioral_data_block_ram_dsm/sub-001/ses-01/sub001-ses01-block1-game1.npz
    ram_dsm_manifest.csv

Each output NPZ contains:
  dsm                 (n_tr, n_tr) float32 Euclidean DSM over HRF features
  dsm_condensed       upper-triangle DSM vector
  ram_tr              selected RAM values averaged in 1-second TR bins
  ram_tr_z            z-scored TR-binned RAM features
  ram_tr_hrf          HRF-convolved RAM features averaged in 1-second TR bins
  ram_tr_hrf_z        z-scored HRF-convolved TR features used for DSM
  selected_ram_indices, selected_ram_labels, frame/bin metadata
"""
from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy import stats as scipy_stats

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib-cache"
os.environ["XDG_CACHE_HOME"] = "/tmp"


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_ROOT = ROOT / "behavioral_data_block"
DEFAULT_OUTPUT_BASE = Path(__file__).resolve().parent / "outputs" / "00_prepare"
DEFAULT_DSM_ROOT = DEFAULT_OUTPUT_BASE / "behavioral_data_block_ram_dsm"
DEFAULT_FMRI_ROOT = Path("/home/jeongmin/fmri/atari/derivatives/ants_mni")
DEFAULT_FMRI_IMAGE = "s5_wfiltered_func_data.nii"
DEFAULT_FMRI_DSM_ROOT = DEFAULT_OUTPUT_BASE / "fmri_roi_bold_dsm"
DEFAULT_THINKER_ROOT = ROOT / "test" / "sub001"
DEFAULT_THINKER_DSM_ROOT = DEFAULT_OUTPUT_BASE / "thinker_rep_dsm"
DEFAULT_HARVARD_OXFORD_MASK_ROOT = (
    Path(__file__).resolve().parent
    / "outputs"
    / "06_representational_mechanism"
    / "atlas"
    / "harvard_oxford"
    / "ants_mni_2p5mm_masks"
    / "masks"
)
DEFAULT_DURATION_SECONDS = 8 * 60
DEFAULT_TR_SECONDS = 1.0
DEFAULT_HRF_DURATION_SECONDS = 32.0
DEFAULT_FMRI_TRIM_VOLUMES = 60
DEFAULT_FMRI_MAX_ROI_VOXELS = 5000
DEFAULT_HPC_PFC_WINDOW_VOLUMES = 11
DEFAULT_THINKER_PCA_COMPONENTS = 128
DEFAULT_THINKER_PCA_SAMPLE_STEPS = 4096
DEFAULT_THINKER_MAX_FEATURE_GB = 0.25

EPS = 1e-12

GAME_NAMES = {
    0: "Enduro",
    1: "Pong",
    2: "SpaceInvaders",
}

FMRI_ROI_SPECS: Dict[str, Dict[str, object]] = {
    "left_hippocampus": {
        "label": "Left Hippocampus",
        "kind": "roi_pattern",
        "mask_paths": [
            DEFAULT_HARVARD_OXFORD_MASK_ROOT
            / "subcortical"
            / "roi-subcortical-010_Left-Hippocampus_mask.nii.gz",
        ],
    },
    "right_hippocampus": {
        "label": "Right Hippocampus",
        "kind": "roi_pattern",
        "mask_paths": [
            DEFAULT_HARVARD_OXFORD_MASK_ROOT
            / "subcortical"
            / "roi-subcortical-020_Right-Hippocampus_mask.nii.gz",
        ],
    },
    "pfc": {
        "label": "Prefrontal Cortex",
        "kind": "roi_pattern",
        "mask_paths": [
            DEFAULT_HARVARD_OXFORD_MASK_ROOT
            / "group"
            / "roi-HarvardOxford-PFC_mask.nii.gz",
        ],
    },
    "hippocampus_pfc_coupling": {
        "label": "Hippocampus-PFC Coupling",
        "kind": "hpc_pfc_coupling",
        "hippocampus_mask_paths": [
            DEFAULT_HARVARD_OXFORD_MASK_ROOT
            / "group"
            / "roi-HarvardOxford-Hippocampus_mask.nii.gz",
        ],
        "pfc_mask_paths": [
            DEFAULT_HARVARD_OXFORD_MASK_ROOT
            / "group"
            / "roi-HarvardOxford-PFC_mask.nii.gz",
        ],
    },
}

# Based on research_script/04_state_complexity_gramian.py. Pong and
# SpaceInvaders are expanded with the extra state bytes requested for RSA.
GAME_RAM_SPECS: Dict[int, Dict[str, object]] = {
    0: {
        "source": "04_state_complexity_gramian.py RAM_ANNOTATIONS game_0",
        "indices": [27, 28, 29, 30, 31, 32, 33, 34, 45, 46, 52, 54, 59, 106],
        "labels": {
            27: "car_slot_0_flags",
            28: "car_slot_1_flags",
            29: "car_slot_2_flags",
            30: "car_slot_3_flags",
            31: "car_slot_4_flags",
            32: "car_slot_5_flags",
            33: "car_slot_6_flags",
            34: "turn_or_road_curvature_candidate",
            45: "level_or_day_indicator",
            46: "player_x_drift_or_road_turn",
            52: "player_y_offset",
            54: "previous_player_x_or_turn_history",
            59: "car_depth_or_perspective_offset",
            106: "player_sinking_or_death_y_state",
        },
    },
    1: {
        "source": (
            "04_state_complexity_gramian.py RAM_ANNOTATIONS game_1, "
            "expanded with paddle y positions"
        ),
        "indices": [49, 50, 51, 54],
        "labels": {
            49: "ball_x",
            50: "enemy_paddle_y",
            51: "player_paddle_y",
            54: "ball_y",
        },
    },
    2: {
        "source": (
            "04_state_complexity_gramian.py RAM_ANNOTATIONS game_2, "
            "expanded with x-position and projectile bytes"
        ),
        "indices": [17, 18, 19, 20, 21, 22, 23, 26, 28, 29, 81, 82, 83, 84, 85, 86, 87, 88],
        "labels": {
            17: "number_of_alive_aliens",
            18: "alien_bitmap_row_1",
            19: "alien_bitmap_row_2",
            20: "alien_bitmap_row_3",
            21: "alien_bitmap_row_4",
            22: "alien_bitmap_row_5",
            23: "alien_bitmap_row_6",
            26: "aliens_x",
            28: "player_green_x",
            29: "player_yellow_x",
            81: "enemy_bullet_1_y",
            82: "enemy_bullet_2_y",
            83: "enemy_bullet_1_x",
            84: "enemy_bullet_2_x",
            85: "player_green_bullet_y",
            86: "player_yellow_bullet_y",
            87: "player_green_bullet_x",
            88: "player_yellow_bullet_x",
        },
    },
}


@dataclass(frozen=True)
class BlockMeta:
    subject: int
    session: int
    block: int
    game: int
    path: Path

    @property
    def relative_output_path(self) -> Path:
        return Path(f"sub-{self.subject:03d}") / f"ses-{self.session:02d}" / self.path.name

    @property
    def analysis_unit(self) -> str:
        return f"sub{self.subject:03d}_ses{self.session:02d}_block{self.block:02d}_game{self.game}"


@dataclass(frozen=True)
class TraceMeta:
    subject: int
    session: int
    block: int
    game: int
    chunk: int
    path: Path

    @property
    def analysis_unit(self) -> str:
        return f"sub{self.subject:03d}_ses{self.session:02d}_block{self.block:02d}_game{self.game}"


def parse_block_meta(path: Path) -> BlockMeta | None:
    match = re.match(r"sub(\d+)-ses(\d+)-block(\d+)-game(\d+)\.npz$", path.name)
    if match is None:
        return None
    return BlockMeta(
        subject=int(match.group(1)),
        session=int(match.group(2)),
        block=int(match.group(3)),
        game=int(match.group(4)),
        path=path,
    )


def parse_trace_meta(path: Path) -> TraceMeta | None:
    match = re.match(r"sub(\d+)-ses(\d+)-block(\d+)-game(\d+)_(\d+)\.npy$", path.name)
    if match is None:
        return None
    return TraceMeta(
        subject=int(match.group(1)),
        session=int(match.group(2)),
        block=int(match.group(3)),
        game=int(match.group(4)),
        chunk=int(match.group(5)),
        path=path,
    )


def parse_int_list(raw: str | None) -> set[int] | None:
    if raw is None or raw.strip() == "":
        return None
    out: set[int] = set()
    for item in raw.split(","):
        item = item.strip()
        if item:
            out.add(int(item))
    return out


def gather_npz_files(
    input_root: Path,
    *,
    subjects: set[int] | None,
    sessions: set[int] | None,
    games: set[int] | None,
    max_files: int | None,
) -> List[BlockMeta]:
    metas: List[BlockMeta] = []
    for path in sorted(input_root.rglob("*.npz")):
        meta = parse_block_meta(path)
        if meta is None:
            continue
        if subjects is not None and meta.subject not in subjects:
            continue
        if sessions is not None and meta.session not in sessions:
            continue
        if games is not None and meta.game not in games:
            continue
        metas.append(meta)
    metas = sorted(metas, key=lambda m: (m.subject, m.session, m.block, m.game, str(m.path)))
    if max_files is not None:
        metas = metas[:max_files]
    return metas


def gather_trace_files(thinker_root: Path, behavior_metas: Sequence[BlockMeta]) -> Dict[str, List[TraceMeta]]:
    wanted = {(m.subject, m.session, m.block, m.game) for m in behavior_metas}
    grouped: Dict[str, List[TraceMeta]] = {}
    for path in sorted(thinker_root.rglob("*.npy")):
        meta = parse_trace_meta(path)
        if meta is None:
            continue
        if (meta.subject, meta.session, meta.block, meta.game) not in wanted:
            continue
        grouped.setdefault(meta.analysis_unit, []).append(meta)
    for key in grouped:
        grouped[key] = sorted(grouped[key], key=lambda m: (m.chunk, str(m.path)))
    return dict(sorted(grouped.items()))


def selected_ram_spec(game_id: int) -> Tuple[np.ndarray, np.ndarray, str]:
    if game_id not in GAME_RAM_SPECS:
        raise ValueError(f"No RAM byte specification for game {game_id}")
    spec = GAME_RAM_SPECS[game_id]
    indices = np.asarray(spec["indices"], dtype=np.int16)
    labels_dict = spec["labels"]
    labels = np.asarray([labels_dict.get(int(idx), f"ram_{idx}") for idx in indices], dtype=str)
    return indices, labels, str(spec.get("source", ""))


def tr_bin_edges(n_frames: int, n_tr: int) -> np.ndarray:
    if n_frames <= 0:
        return np.zeros(n_tr + 1, dtype=np.int32)
    return np.rint(np.linspace(0, n_frames, n_tr + 1)).astype(np.int32)


def average_to_tr_bins(values: np.ndarray, n_tr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float32)
    edges = tr_bin_edges(len(arr), n_tr)
    counts = np.diff(edges).astype(np.int32)
    binned = np.empty((n_tr, arr.shape[1]), dtype=np.float32)
    previous = np.zeros(arr.shape[1], dtype=np.float32)
    for i in range(n_tr):
        start = int(edges[i])
        stop = int(edges[i + 1])
        if stop > start:
            previous = np.nanmean(arr[start:stop], axis=0).astype(np.float32)
        binned[i] = previous
    return binned, counts, edges[:-1].astype(np.int32), edges[1:].astype(np.int32)


def average_to_time_bins(
    values: np.ndarray,
    frame_time: np.ndarray,
    *,
    n_tr: int,
    tr_seconds: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float32)
    time = np.asarray(frame_time, dtype=np.float64).reshape(-1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got {arr.shape}")
    if len(arr) != len(time):
        raise ValueError(f"Feature/time length mismatch: {len(arr)} vs {len(time)}")
    if len(arr) == 0:
        return average_to_tr_bins(arr, n_tr)

    finite = np.isfinite(time)
    if not np.any(finite):
        return average_to_tr_bins(arr, n_tr)
    time = np.where(finite, time, np.nanmedian(time[finite]))
    time = time - float(np.nanmin(time))
    bin_index = np.floor(time / float(tr_seconds)).astype(np.int32)
    bin_index = np.clip(bin_index, 0, n_tr - 1)

    sums = np.zeros((n_tr, arr.shape[1]), dtype=np.float64)
    counts = np.zeros(n_tr, dtype=np.int32)
    frame_start = np.full(n_tr, -1, dtype=np.int32)
    frame_stop = np.full(n_tr, -1, dtype=np.int32)
    np.add.at(sums, bin_index, np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0))
    np.add.at(counts, bin_index, 1)
    for idx, bin_id in enumerate(bin_index):
        if frame_start[bin_id] < 0:
            frame_start[bin_id] = idx
        frame_stop[bin_id] = idx + 1

    binned = sums / np.maximum(counts[:, None], 1)
    previous = np.zeros(arr.shape[1], dtype=np.float64)
    for i in range(n_tr):
        if counts[i] > 0:
            previous = binned[i]
        else:
            binned[i] = previous
    return binned.astype(np.float32), counts, frame_start, frame_stop


def zscore_columns(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float32)
    mean = np.nanmean(arr, axis=0, keepdims=True)
    std = np.nanstd(arr, axis=0, keepdims=True)
    z = (arr - mean) / np.where(std <= EPS, 1.0, std)
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return z.astype(np.float32), mean.reshape(-1).astype(np.float32), std.reshape(-1).astype(np.float32)


def gamma_pdf(t: np.ndarray, shape: float, scale: float) -> np.ndarray:
    x = np.asarray(t, dtype=np.float64)
    out = np.zeros_like(x, dtype=np.float64)
    valid = x > 0
    if not np.any(valid):
        return out
    xv = x[valid]
    denom = math.gamma(shape) * (scale ** shape)
    out[valid] = (xv ** (shape - 1.0)) * np.exp(-xv / scale) / max(denom, EPS)
    return out


def canonical_hrf(
    frame_dt: float,
    *,
    duration_seconds: float,
    peak_delay: float,
    undershoot_delay: float,
    undershoot_ratio: float,
) -> Tuple[np.ndarray, np.ndarray]:
    dt = max(float(frame_dt), EPS)
    duration = max(float(duration_seconds), dt)
    t = np.arange(0.0, duration + dt, dt, dtype=np.float64)
    hrf = gamma_pdf(t, shape=peak_delay, scale=1.0)
    hrf -= gamma_pdf(t, shape=undershoot_delay, scale=1.0) / max(float(undershoot_ratio), EPS)
    if np.sum(np.abs(hrf)) <= EPS:
        hrf = np.zeros_like(t)
        hrf[0] = 1.0
    else:
        hrf = hrf / np.sum(hrf)
    return hrf.astype(np.float32), t.astype(np.float32)


def frame_time_vector(
    data: np.lib.npyio.NpzFile,
    n_frames: int,
    *,
    duration_seconds: float,
) -> np.ndarray:
    if "time" in data.files:
        time = np.asarray(data["time"], dtype=np.float64).reshape(-1)
        if len(time) >= n_frames and np.isfinite(time[:n_frames]).any():
            time = time[:n_frames]
            return (time - float(np.nanmin(time[np.isfinite(time)]))).astype(np.float32)
    return np.linspace(0.0, float(duration_seconds), n_frames, endpoint=False, dtype=np.float32)


def median_frame_dt(frame_time: np.ndarray, duration_seconds: float, n_frames: int) -> float:
    time = np.asarray(frame_time, dtype=np.float64).reshape(-1)
    diffs = np.diff(time)
    diffs = diffs[np.isfinite(diffs) & (diffs > EPS)]
    if diffs.size > 0:
        return float(np.median(diffs))
    if n_frames > 0:
        return float(duration_seconds) / float(n_frames)
    return 1.0 / 60.0


def convolve_frame_features_with_hrf(values: np.ndarray, hrf: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    kernel = np.asarray(hrf, dtype=np.float32).reshape(-1, 1)
    full = scipy_signal.fftconvolve(arr, kernel, mode="full", axes=0)
    return full[: arr.shape[0]].astype(np.float32)


def frame_level_hrf_features(
    selected_ram: np.ndarray,
    frame_time: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    frame_z, _, _ = zscore_columns(selected_ram)
    frame_dt = median_frame_dt(frame_time, args.duration_seconds, len(selected_ram))
    hrf, hrf_time = canonical_hrf(
        frame_dt,
        duration_seconds=args.hrf_duration_seconds,
        peak_delay=args.hrf_peak_delay,
        undershoot_delay=args.hrf_undershoot_delay,
        undershoot_ratio=args.hrf_undershoot_ratio,
    )
    frame_hrf = convolve_frame_features_with_hrf(frame_z, hrf)
    return frame_z, frame_hrf, hrf, hrf_time, np.asarray(frame_time, dtype=np.float32), frame_dt


def pairwise_euclidean_dsm(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    norms = np.sum(arr * arr, axis=1, keepdims=True)
    sq = norms + norms.T - 2.0 * (arr @ arr.T)
    return np.sqrt(np.maximum(sq, 0.0)).astype(np.float32)


def pairwise_correlation_dsm(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr - np.nanmean(arr, axis=1, keepdims=True)
    denom = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / np.where(denom <= EPS, 1.0, denom)
    sim = np.clip(arr @ arr.T, -1.0, 1.0)
    return (1.0 - sim).astype(np.float32)


def build_dsm(values: np.ndarray, metric: str) -> np.ndarray:
    if metric == "euclidean":
        return pairwise_euclidean_dsm(values)
    if metric == "correlation":
        return pairwise_correlation_dsm(values)
    raise ValueError(f"Unknown DSM metric: {metric}")


def condensed_upper_triangle(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        return np.empty(0, dtype=np.float32)
    tri = np.triu_indices(matrix.shape[0], k=1)
    return matrix[tri].astype(np.float32)


THINKER_REP_NAMES = ("tree_reps", "im_vectors", "im_vp_vectors")
REAL_STEP_STATUS = 0
IMAGINARY_STEP_STATUS = 2


def load_npy_dict(path: Path) -> Dict[str, object]:
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
        item = obj.item()
        if isinstance(item, dict):
            return item
    if hasattr(obj, "files"):
        return {key: obj[key] for key in obj.files}
    raise ValueError(f"Cannot parse file as dict-like npy/npz: {path}")


def vectorize_raw_value(raw: object) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float32)
    if arr.size == 0:
        return np.empty(0, dtype=np.float32)
    return np.nan_to_num(np.squeeze(arr).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def tree_key_dims(tree: Dict[str, object]) -> Dict[str, int]:
    dims: Dict[str, int] = {}
    for key in sorted(tree):
        arr = np.asarray(tree[key])
        if arr.ndim == 0 or arr.shape[0] == 0:
            continue
        dim = int(np.prod(arr.shape[1:])) if arr.ndim >= 2 else 1
        if dim > 0:
            dims[key] = dim
    return dims


def pad_or_truncate_vector(values: np.ndarray, dim: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == int(dim):
        return arr
    out = np.zeros(int(dim), dtype=np.float32)
    n = min(out.size, arr.size)
    if n > 0:
        out[:n] = arr[:n]
    return out


def tree_step_vector(tree: Dict[str, object], idx: int, key_dims: Dict[str, int]) -> np.ndarray:
    rows: List[np.ndarray] = []
    for key, dim in key_dims.items():
        if key not in tree:
            rows.append(np.zeros(dim, dtype=np.float32))
            continue
        arr = np.asarray(tree[key])
        if arr.ndim == 0 or idx >= arr.shape[0]:
            rows.append(np.zeros(dim, dtype=np.float32))
            continue
        rows.append(pad_or_truncate_vector(arr[int(idx)], dim))
    if not rows:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(rows).astype(np.float32)


def trace_step_vector(
    data: Dict[str, object],
    rep_name: str,
    idx: int,
    *,
    step_dim: int,
    key_dims: Dict[str, int] | None,
) -> np.ndarray:
    if rep_name == "tree_reps":
        tree = data.get("tree_reps")
        if not isinstance(tree, dict) or key_dims is None:
            return np.zeros(step_dim, dtype=np.float32)
        return pad_or_truncate_vector(tree_step_vector(tree, idx, key_dims), step_dim)
    values = data.get(rep_name)
    if values is None or idx >= len(values):  # type: ignore[arg-type]
        return np.zeros(step_dim, dtype=np.float32)
    return pad_or_truncate_vector(vectorize_raw_value(values[int(idx)]), step_dim)  # type: ignore[index]


def infer_trace_representation_layout(
    trace_metas: Sequence[TraceMeta],
    rep_name: str,
) -> Tuple[int, Dict[str, int], Dict[str, int]]:
    key_dims: Dict[str, int] = {}
    step_dim = 0
    seen_real = False
    interval_len = 0
    stats = {
        "n_chunks": len(trace_metas),
        "n_steps": 0,
        "n_real_steps": 0,
        "n_valid_real_intervals": 0,
        "n_imaginary_steps": 0,
        "max_imaginary_steps": 0,
    }

    for trace_meta in trace_metas:
        data = load_npy_dict(trace_meta.path)
        status = np.asarray(data["status"]).reshape(-1)
        stats["n_steps"] += int(len(status))

        if rep_name == "tree_reps" and not key_dims:
            tree = data.get("tree_reps")
            if isinstance(tree, dict):
                key_dims = tree_key_dims(tree)
                step_dim = int(sum(key_dims.values()))
        elif rep_name != "tree_reps" and step_dim == 0 and rep_name in data:
            values = data[rep_name]  # type: ignore[index]
            for idx in np.flatnonzero(status == IMAGINARY_STEP_STATUS):
                vec = vectorize_raw_value(values[int(idx)])  # type: ignore[index]
                if vec.size > 0:
                    step_dim = int(vec.size)
                    break

        for raw_status in status:
            step_status = int(raw_status)
            if step_status == IMAGINARY_STEP_STATUS:
                stats["n_imaginary_steps"] += 1
                if seen_real:
                    interval_len += 1
            elif step_status == REAL_STEP_STATUS:
                stats["n_real_steps"] += 1
                if seen_real:
                    stats["n_valid_real_intervals"] += 1
                    stats["max_imaginary_steps"] = max(stats["max_imaginary_steps"], interval_len)
                seen_real = True
                interval_len = 0

    return step_dim, key_dims, stats


def sample_imaginary_step_vectors(
    trace_metas: Sequence[TraceMeta],
    rep_name: str,
    *,
    step_dim: int,
    key_dims: Dict[str, int] | None,
    max_samples: int,
    random_seed: int,
) -> np.ndarray:
    if step_dim <= 0 or max_samples <= 0:
        return np.empty((0, 0), dtype=np.float32)
    rng = np.random.default_rng(int(random_seed))
    rows: List[np.ndarray] = []
    seen = 0
    for trace_meta in trace_metas:
        data = load_npy_dict(trace_meta.path)
        status = np.asarray(data["status"]).reshape(-1)
        for idx in np.flatnonzero(status == IMAGINARY_STEP_STATUS):
            vec = trace_step_vector(data, rep_name, int(idx), step_dim=step_dim, key_dims=key_dims)
            if vec.size == 0:
                continue
            seen += 1
            if len(rows) < max_samples:
                rows.append(vec)
            else:
                replace = int(rng.integers(0, seen))
                if replace < max_samples:
                    rows[replace] = vec
    if not rows:
        return np.empty((0, step_dim), dtype=np.float32)
    return np.vstack(rows).astype(np.float32)


def fit_step_transform(
    sample: np.ndarray,
    *,
    use_pca: bool,
    n_components: int,
    random_seed: int,
) -> Dict[str, object]:
    arr = np.asarray(sample, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("Cannot fit representation transform from an empty sample.")
    mean = np.nanmean(arr, axis=0).astype(np.float32)
    std = np.nanstd(arr, axis=0).astype(np.float32)
    std = np.where(std <= EPS, 1.0, std).astype(np.float32)
    z = np.nan_to_num((arr - mean[None, :]) / std[None, :], nan=0.0, posinf=0.0, neginf=0.0)
    pca = None
    explained = np.empty(0, dtype=np.float32)
    output_dim = int(z.shape[1])
    if use_pca:
        from sklearn.decomposition import PCA

        max_components = max(1, min(int(n_components), z.shape[0] - 1, z.shape[1]))
        solver = "randomized" if max_components < min(z.shape) else "auto"
        pca = PCA(n_components=max_components, svd_solver=solver, random_state=int(random_seed))
        pca.fit(z)
        output_dim = int(max_components)
        explained = np.asarray(pca.explained_variance_ratio_, dtype=np.float32)
    return {
        "mean": mean,
        "std": std,
        "pca": pca,
        "output_dim": output_dim,
        "explained_variance_ratio": explained,
    }


def transform_step_matrix(values: np.ndarray, transform: Dict[str, object]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        return np.empty((0, int(transform["output_dim"])), dtype=np.float32)
    mean = np.asarray(transform["mean"], dtype=np.float32)
    std = np.asarray(transform["std"], dtype=np.float32)
    z = np.nan_to_num((arr - mean[None, :]) / std[None, :], nan=0.0, posinf=0.0, neginf=0.0)
    pca = transform.get("pca")
    if pca is not None:
        return np.asarray(pca.transform(z), dtype=np.float32)
    return z.astype(np.float32)


def concat_interval_matrix(step_matrix: np.ndarray, max_steps: int, step_dim: int) -> np.ndarray:
    out = np.zeros((int(max_steps), int(step_dim)), dtype=np.float32)
    if step_matrix.size > 0:
        n = min(int(max_steps), int(step_matrix.shape[0]))
        out[:n, :] = step_matrix[:n, :step_dim]
    return out.reshape(-1).astype(np.float32)


def choose_thinker_pca_mode(n_tr: int, max_steps: int, step_dim: int, args: argparse.Namespace) -> Tuple[bool, float]:
    full_feature_dim = int(max_steps) * int(step_dim)
    estimated_gb = float(n_tr) * float(full_feature_dim) * 4.0 / (1024.0**3)
    use_pca = (
        estimated_gb > float(args.thinker_max_feature_gb)
        and int(args.thinker_pca_components) > 0
        and int(step_dim) > int(args.thinker_pca_components)
    )
    return bool(use_pca), estimated_gb


def thinker_rep_output_path(output_root: Path, meta: BlockMeta, rep_name: str) -> Path:
    return (
        output_root
        / fmri_subject_dir(meta)
        / fmri_session_dir(meta)
        / f"{meta.analysis_unit}_rep-{rep_name}_concatimag_dsm.npz"
    )


def prepare_thinker_representation_dsm(
    meta: BlockMeta,
    trace_metas: Sequence[TraceMeta],
    rep_name: str,
    args: argparse.Namespace,
) -> Dict[str, object]:
    out_path = thinker_rep_output_path(args.thinker_dsm_root, meta, rep_name)
    base_row = {
        "source_behavior_npz": str(meta.path),
        "subject": meta.subject,
        "session": meta.session,
        "block": meta.block,
        "game": meta.game,
        "game_name": GAME_NAMES.get(meta.game, f"game_{meta.game}"),
        "analysis_unit": meta.analysis_unit,
        "representation": rep_name,
        "output_npz": str(out_path),
    }
    if not trace_metas:
        return {**base_row, "status": "no_trace_files"}

    n_tr = int(round(float(args.duration_seconds) / float(args.tr_seconds)))
    step_dim, key_dims, stats = infer_trace_representation_layout(trace_metas, rep_name)
    max_steps = int(stats["max_imaginary_steps"])
    n_intervals = int(stats["n_valid_real_intervals"])
    if step_dim <= 0:
        return {**base_row, **stats, "status": "empty_representation"}
    if max_steps <= 0 or n_intervals <= 0:
        return {**base_row, **stats, "per_step_dim": step_dim, "status": "no_valid_imaginary_intervals"}

    use_pca, estimated_full_gb = choose_thinker_pca_mode(n_tr, max_steps, step_dim, args)
    sample = sample_imaginary_step_vectors(
        trace_metas,
        rep_name,
        step_dim=step_dim,
        key_dims=key_dims if rep_name == "tree_reps" else None,
        max_samples=int(args.thinker_pca_sample_steps),
        random_seed=int(args.thinker_random_seed),
    )
    transform = fit_step_transform(
        sample,
        use_pca=use_pca,
        n_components=int(args.thinker_pca_components),
        random_seed=int(args.thinker_random_seed),
    )
    transformed_step_dim = int(transform["output_dim"])
    concat_dim = int(max_steps) * transformed_step_dim
    sums = np.zeros((n_tr, concat_dim), dtype=np.float32)
    bin_counts = np.zeros(n_tr, dtype=np.int32)
    interval_lengths: List[int] = []
    seen_real = False
    interval_rows: List[np.ndarray] = []
    valid_interval_idx = 0

    for trace_meta in trace_metas:
        data = load_npy_dict(trace_meta.path)
        status = np.asarray(data["status"]).reshape(-1)
        for idx, raw_status in enumerate(status):
            step_status = int(raw_status)
            if step_status == IMAGINARY_STEP_STATUS and seen_real:
                interval_rows.append(
                    trace_step_vector(
                        data,
                        rep_name,
                        int(idx),
                        step_dim=step_dim,
                        key_dims=key_dims if rep_name == "tree_reps" else None,
                    )
                )
            elif step_status == REAL_STEP_STATUS:
                if seen_real:
                    if interval_rows:
                        interval_matrix = np.vstack(interval_rows).astype(np.float32)
                        transformed = transform_step_matrix(interval_matrix, transform)
                    else:
                        transformed = np.empty((0, transformed_step_dim), dtype=np.float32)
                    concat = concat_interval_matrix(transformed, max_steps, transformed_step_dim)
                    bin_idx = min(int(valid_interval_idx * n_tr / max(n_intervals, 1)), n_tr - 1)
                    sums[bin_idx] += concat
                    bin_counts[bin_idx] += 1
                    interval_lengths.append(len(interval_rows))
                    valid_interval_idx += 1
                seen_real = True
                interval_rows = []

    features = sums / np.maximum(bin_counts[:, None], 1)
    previous = np.zeros(features.shape[1], dtype=np.float32)
    valid_bins = bin_counts > 0
    for i in range(n_tr):
        if valid_bins[i]:
            previous = features[i].copy()
        else:
            features[i] = previous
    features_z, feature_mean, feature_std = zscore_columns(features)
    dsm = build_dsm(features_z, "correlation")
    dsm_condensed = condensed_upper_triangle(dsm)
    explained = np.asarray(transform["explained_variance_ratio"], dtype=np.float32)
    tree_keys = np.asarray(list(key_dims.keys()), dtype=str)

    if not args.dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            dsm=dsm.astype(np.float32),
            dsm_condensed=dsm_condensed.astype(np.float32),
            thinker_tr_features=features.astype(np.float32),
            thinker_tr_features_z=features_z.astype(np.float32),
            thinker_feature_mean=feature_mean.astype(np.float32),
            thinker_feature_std=feature_std.astype(np.float32),
            volume_indices=np.arange(n_tr, dtype=np.int32),
            bin_counts=bin_counts.astype(np.int32),
            real_interval_lengths=np.asarray(interval_lengths, dtype=np.int32),
            representation=np.asarray(rep_name),
            feature_model=np.asarray("previous_real_to_current_real_imaginary_sequence_concat"),
            source_trace_files=np.asarray([str(t.path) for t in trace_metas]),
            source_behavior_npz=np.asarray(str(meta.path)),
            analysis_unit=np.asarray(meta.analysis_unit),
            subject=np.asarray(meta.subject, dtype=np.int16),
            session=np.asarray(meta.session, dtype=np.int16),
            block=np.asarray(meta.block, dtype=np.int16),
            game=np.asarray(meta.game, dtype=np.int16),
            n_tr=np.asarray(n_tr, dtype=np.int16),
            per_step_dim=np.asarray(step_dim, dtype=np.int32),
            transformed_per_step_dim=np.asarray(transformed_step_dim, dtype=np.int32),
            max_imaginary_steps=np.asarray(max_steps, dtype=np.int32),
            n_valid_real_intervals=np.asarray(n_intervals, dtype=np.int32),
            n_real_steps=np.asarray(int(stats["n_real_steps"]), dtype=np.int32),
            n_imaginary_steps=np.asarray(int(stats["n_imaginary_steps"]), dtype=np.int32),
            pca_applied=np.asarray(bool(use_pca)),
            pca_components=np.asarray(transformed_step_dim if use_pca else 0, dtype=np.int16),
            pca_explained_variance_ratio=explained.astype(np.float32),
            estimated_full_feature_gb=np.asarray(estimated_full_gb, dtype=np.float32),
            tree_keys=tree_keys,
            distance_metric=np.asarray("correlation"),
        )

    return {
        **base_row,
        **stats,
        "per_step_dim": step_dim,
        "transformed_per_step_dim": transformed_step_dim,
        "concat_dim": concat_dim,
        "n_tr": n_tr,
        "n_nonempty_bins": int(np.sum(bin_counts > 0)),
        "min_real_intervals_per_bin": int(np.min(bin_counts)) if bin_counts.size else 0,
        "max_real_intervals_per_bin": int(np.max(bin_counts)) if bin_counts.size else 0,
        "pca_applied": bool(use_pca),
        "pca_components": transformed_step_dim if use_pca else 0,
        "pca_explained_variance_sum": float(np.sum(explained)) if explained.size else np.nan,
        "estimated_full_feature_gb": estimated_full_gb,
        "distance_metric": "correlation",
        "status": "dry_run" if args.dry_run else "ok",
    }


def load_dsm_from_npz(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=True) as data:
        return np.asarray(data["dsm"], dtype=np.float32)


def aligned_upper_triangles(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    arr_a = np.asarray(a, dtype=np.float32)
    arr_b = np.asarray(b, dtype=np.float32)
    if arr_a.ndim != 2 or arr_b.ndim != 2:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
    n = min(arr_a.shape[0], arr_a.shape[1], arr_b.shape[0], arr_b.shape[1])
    if n < 3:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
    tri = np.triu_indices(n, k=1)
    x = arr_a[:n, :n][tri].astype(np.float64)
    y = arr_b[:n, :n][tri].astype(np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    return x[finite].astype(np.float32), y[finite].astype(np.float32)


def dsm_vector_correlations(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, int]:
    x, y = aligned_upper_triangles(a, b)
    if len(x) < 3 or np.nanstd(x) <= EPS or np.nanstd(y) <= EPS:
        return np.nan, np.nan, int(len(x))
    spearman = scipy_stats.spearmanr(x, y).correlation
    pearson = scipy_stats.pearsonr(x, y).statistic
    return float(spearman), float(pearson), int(len(x))


def target_dsm_specs(meta: BlockMeta, args: argparse.Namespace) -> List[Tuple[str, str, List[Path]]]:
    specs: List[Tuple[str, str, List[Path]]] = [
        ("ram", "RAM HRF", [args.output_root / meta.relative_output_path]),
        (
            "left_hippocampus",
            "Left Hippocampus",
            [fmri_roi_output_path(args.fmri_dsm_root, meta, "left_hippocampus")],
        ),
        (
            "right_hippocampus",
            "Right Hippocampus",
            [fmri_roi_output_path(args.fmri_dsm_root, meta, "right_hippocampus")],
        ),
        (
            "hippocampus_mean",
            "Mean Hippocampus",
            [
                fmri_roi_output_path(args.fmri_dsm_root, meta, "left_hippocampus"),
                fmri_roi_output_path(args.fmri_dsm_root, meta, "right_hippocampus"),
            ],
        ),
        ("pfc", "Prefrontal Cortex", [fmri_roi_output_path(args.fmri_dsm_root, meta, "pfc")]),
        (
            "hippocampus_pfc_coupling",
            "Hippocampus-PFC Coupling",
            [fmri_roi_output_path(args.fmri_dsm_root, meta, "hippocampus_pfc_coupling")],
        ),
    ]
    return specs


def load_target_dsm(paths: Sequence[Path]) -> Tuple[np.ndarray | None, str]:
    if not paths:
        return None, ""
    existing = [path for path in paths if path.exists()]
    if len(existing) != len(paths):
        return None, ";".join(str(p) for p in paths)
    matrices = [load_dsm_from_npz(path) for path in existing]
    if len(matrices) == 1:
        return matrices[0], str(existing[0])
    n = min(mat.shape[0] for mat in matrices)
    stacked = np.stack([mat[:n, :n] for mat in matrices], axis=0)
    return np.nanmean(stacked, axis=0).astype(np.float32), ";".join(str(p) for p in existing)


def compare_thinker_dsms_to_targets(
    meta: BlockMeta,
    rep_names: Sequence[str],
    args: argparse.Namespace,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for rep_name in rep_names:
        thinker_path = thinker_rep_output_path(args.thinker_dsm_root, meta, rep_name)
        base = {
            "subject": meta.subject,
            "session": meta.session,
            "block": meta.block,
            "game": meta.game,
            "game_name": GAME_NAMES.get(meta.game, f"game_{meta.game}"),
            "analysis_unit": meta.analysis_unit,
            "representation": rep_name,
            "thinker_npz": str(thinker_path),
        }
        if not thinker_path.exists():
            for target_name, target_label, paths in target_dsm_specs(meta, args):
                rows.append(
                    {
                        **base,
                        "target": target_name,
                        "target_label": target_label,
                        "target_npz": ";".join(str(p) for p in paths),
                        "status": "missing_thinker_dsm",
                    }
                )
            continue
        thinker_dsm = load_dsm_from_npz(thinker_path)
        for target_name, target_label, paths in target_dsm_specs(meta, args):
            target_dsm, target_path = load_target_dsm(paths)
            if target_dsm is None:
                rows.append(
                    {
                        **base,
                        "target": target_name,
                        "target_label": target_label,
                        "target_npz": target_path,
                        "status": "missing_target_dsm",
                    }
                )
                continue
            spearman, pearson, n_pairs = dsm_vector_correlations(thinker_dsm, target_dsm)
            rows.append(
                {
                    **base,
                    "target": target_name,
                    "target_label": target_label,
                    "target_npz": target_path,
                    "spearman_rho": spearman,
                    "pearson_r": pearson,
                    "n_pairs": n_pairs,
                    "status": "ok",
                }
            )
    return rows


def clean_pattern_columns(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        n_cols = arr.shape[1] if arr.ndim == 2 else 0
        return np.empty((arr.shape[0] if arr.ndim == 2 else 0, 0), dtype=np.float32), np.zeros(n_cols, dtype=bool)
    finite = np.isfinite(arr).all(axis=0)
    keep = np.zeros(arr.shape[1], dtype=bool)
    if np.any(finite):
        std = np.nanstd(arr[:, finite], axis=0)
        keep[np.where(finite)[0]] = std > EPS
    arr = np.nan_to_num(arr[:, keep], nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size == 0:
        return arr.astype(np.float32), keep
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True)
    arr = (arr - mean) / np.where(std <= EPS, 1.0, std)
    return arr.astype(np.float32), keep


def fmri_subject_dir(meta: BlockMeta) -> str:
    return f"sub{meta.subject:03d}-{meta.session}"


def fmri_session_dir(meta: BlockMeta) -> str:
    return f"Session{meta.block}"


def discover_fmri_path(meta: BlockMeta, fmri_root: Path, image_name: str) -> Path | None:
    run_dir = fmri_root / fmri_subject_dir(meta) / fmri_session_dir(meta)
    image_path = run_dir / image_name
    if image_path.exists():
        return image_path
    if image_name == "s5_wfiltered_func_data.nii":
        fallback = run_dir / "wfiltered_func_data.nii"
        if fallback.exists():
            return fallback
    return None


def fmri_volume_indices(n_volumes: int, *, n_tr: int, trim_volumes: int) -> np.ndarray:
    n_volumes = int(n_volumes)
    n_tr = int(n_tr)
    trim = max(0, int(trim_volumes))
    if n_volumes >= n_tr + 2 * trim:
        start = trim
    elif n_volumes >= n_tr:
        start = 0
    else:
        start = 0
        n_tr = n_volumes
    return np.arange(start, start + n_tr, dtype=np.int32)


def load_union_mask(mask_paths: Sequence[Path], ref_img: object, threshold: float) -> Tuple[np.ndarray, int]:
    try:
        import nibabel as nib
    except Exception as exc:
        raise RuntimeError("nibabel is required for fMRI ROI DSM preparation.") from exc

    ref_shape = tuple(int(x) for x in ref_img.shape[:3])
    union = np.zeros(ref_shape, dtype=bool)
    raw_voxels = 0
    for mask_path in mask_paths:
        if not mask_path.exists():
            raise FileNotFoundError(f"ROI mask not found: {mask_path}")
        mask_img = nib.load(str(mask_path))
        mask_data = np.asarray(mask_img.get_fdata(dtype=np.float32))
        mask_data = np.squeeze(mask_data)
        if mask_data.ndim == 4:
            mask_data = mask_data[..., 0]
        if tuple(mask_data.shape) != ref_shape:
            raise ValueError(
                f"ROI mask shape {mask_data.shape} does not match fMRI shape {ref_shape}: {mask_path}"
            )
        mask = np.isfinite(mask_data) & (mask_data > float(threshold))
        raw_voxels += int(mask.sum())
        union |= mask
    return union, raw_voxels


def extract_fmri_roi_patterns(
    fmri_path: Path,
    *,
    mask_paths: Sequence[Path],
    volume_indices: np.ndarray,
    roi_threshold: float,
    max_roi_voxels: int,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    try:
        import nibabel as nib
    except Exception as exc:
        raise RuntimeError("nibabel is required for fMRI ROI DSM preparation.") from exc

    img = nib.load(str(fmri_path))
    mask, raw_voxels = load_union_mask(mask_paths, img, roi_threshold)
    coords = np.column_stack(np.where(mask)).astype(np.int64)
    if coords.size == 0:
        return np.empty((len(volume_indices), 0), dtype=np.float32), coords, raw_voxels, 0

    if max_roi_voxels > 0 and len(coords) > int(max_roi_voxels):
        keep = np.linspace(0, len(coords) - 1, int(max_roi_voxels), dtype=int)
        coords = coords[keep]

    volume_indices = np.asarray(volume_indices, dtype=np.int32).reshape(-1)
    patterns = np.empty((len(volume_indices), len(coords)), dtype=np.float32)
    proxy = img.dataobj
    x = coords[:, 0].astype(int)
    y = coords[:, 1].astype(int)
    z = coords[:, 2].astype(int)
    for row, volume in enumerate(volume_indices):
        vol = np.asarray(proxy[:, :, :, int(volume)], dtype=np.float32)
        patterns[row] = vol[x, y, z]

    patterns, keep_cols = clean_pattern_columns(patterns)
    coords = coords[keep_cols]
    return patterns, coords.astype(np.int16), raw_voxels, int(patterns.shape[1])


def extract_dual_roi_mean_signals(
    fmri_path: Path,
    *,
    hippocampus_mask_paths: Sequence[Path],
    pfc_mask_paths: Sequence[Path],
    volume_indices: np.ndarray,
    roi_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, int, int, int, int]:
    try:
        import nibabel as nib
    except Exception as exc:
        raise RuntimeError("nibabel is required for fMRI ROI DSM preparation.") from exc

    img = nib.load(str(fmri_path))
    hpc_mask, hpc_raw_voxels = load_union_mask(hippocampus_mask_paths, img, roi_threshold)
    pfc_mask, pfc_raw_voxels = load_union_mask(pfc_mask_paths, img, roi_threshold)
    hpc_coords = np.column_stack(np.where(hpc_mask)).astype(np.int64)
    pfc_coords = np.column_stack(np.where(pfc_mask)).astype(np.int64)
    if hpc_coords.size == 0 or pfc_coords.size == 0:
        return (
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
            hpc_raw_voxels,
            pfc_raw_voxels,
            int(len(hpc_coords)),
            int(len(pfc_coords)),
        )

    volume_indices = np.asarray(volume_indices, dtype=np.int32).reshape(-1)
    hpc_signal = np.empty(len(volume_indices), dtype=np.float32)
    pfc_signal = np.empty(len(volume_indices), dtype=np.float32)
    proxy = img.dataobj
    hx, hy, hz = (hpc_coords[:, i].astype(int) for i in range(3))
    px, py, pz = (pfc_coords[:, i].astype(int) for i in range(3))
    for row, volume in enumerate(volume_indices):
        vol = np.asarray(proxy[:, :, :, int(volume)], dtype=np.float32)
        hpc_signal[row] = float(np.nanmean(vol[hx, hy, hz]))
        pfc_signal[row] = float(np.nanmean(vol[px, py, pz]))

    hpc_signal = np.nan_to_num(hpc_signal, nan=0.0, posinf=0.0, neginf=0.0)
    pfc_signal = np.nan_to_num(pfc_signal, nan=0.0, posinf=0.0, neginf=0.0)
    return (
        hpc_signal.astype(np.float32),
        pfc_signal.astype(np.float32),
        hpc_raw_voxels,
        pfc_raw_voxels,
        int(len(hpc_coords)),
        int(len(pfc_coords)),
    )


def zscore_vector(values: np.ndarray) -> Tuple[np.ndarray, float, float]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return arr, np.nan, np.nan
    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr))
    if not np.isfinite(std) or std <= EPS:
        z = np.zeros_like(arr, dtype=np.float32)
    else:
        z = (arr - mean) / std
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return z.astype(np.float32), mean, std


def effective_odd_window(window_volumes: int, n_timepoints: int) -> int:
    window = max(3, int(window_volumes))
    if window % 2 == 0:
        window += 1
    if n_timepoints < window:
        window = n_timepoints if n_timepoints % 2 == 1 else n_timepoints - 1
    return max(3, int(window))


def sliding_window_correlation(
    x: np.ndarray,
    y: np.ndarray,
    window_volumes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    x_arr = np.asarray(x, dtype=np.float32).reshape(-1)
    y_arr = np.asarray(y, dtype=np.float32).reshape(-1)
    if len(x_arr) != len(y_arr):
        raise ValueError(f"Signal length mismatch: {len(x_arr)} vs {len(y_arr)}")
    n_timepoints = len(x_arr)
    if n_timepoints < 3:
        raise ValueError(f"Need at least 3 timepoints for coupling correlation, got {n_timepoints}")
    window = effective_odd_window(window_volumes, n_timepoints)
    half = window // 2
    corr = np.empty(n_timepoints, dtype=np.float32)
    starts = np.empty(n_timepoints, dtype=np.int32)
    stops = np.empty(n_timepoints, dtype=np.int32)
    for i in range(n_timepoints):
        start = min(max(i - half, 0), n_timepoints - window)
        stop = start + window
        starts[i] = start
        stops[i] = stop
        xw = x_arr[start:stop] - float(np.mean(x_arr[start:stop]))
        yw = y_arr[start:stop] - float(np.mean(y_arr[start:stop]))
        denom = float(np.linalg.norm(xw) * np.linalg.norm(yw))
        if denom <= EPS:
            corr[i] = 0.0
        else:
            corr[i] = float(np.clip(np.dot(xw, yw) / denom, -1.0, 1.0))
    return corr, starts, stops, window


def pairwise_absolute_difference(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    return np.abs(arr[:, None] - arr[None, :]).astype(np.float32)


def fmri_roi_output_path(output_root: Path, meta: BlockMeta, roi_name: str) -> Path:
    return (
        output_root
        / fmri_subject_dir(meta)
        / fmri_session_dir(meta)
        / f"{meta.analysis_unit}_roi-{roi_name}_bold_dsm.npz"
    )


def prepare_fmri_roi_dsms(
    meta: BlockMeta,
    args: argparse.Namespace,
    roi_names: Sequence[str],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    try:
        import nibabel as nib
    except Exception as exc:
        base = {
            "source_behavior_npz": str(meta.path),
            "subject": meta.subject,
            "session": meta.session,
            "block": meta.block,
            "game": meta.game,
            "status": f"error: nibabel_missing: {exc}",
        }
        return [base]

    fmri_path = discover_fmri_path(meta, args.fmri_root, args.fmri_image)
    base_row = {
        "source_behavior_npz": str(meta.path),
        "subject": meta.subject,
        "session": meta.session,
        "block": meta.block,
        "game": meta.game,
        "game_name": GAME_NAMES.get(meta.game, f"game_{meta.game}"),
        "fmri_subject": fmri_subject_dir(meta),
        "fmri_session": fmri_session_dir(meta),
        "analysis_unit": meta.analysis_unit,
        "fmri_path": str(fmri_path) if fmri_path is not None else "",
        "distance_metric": args.fmri_distance_metric,
    }
    if fmri_path is None:
        return [{**base_row, "roi": "", "status": "no_matching_fmri_run"}]

    img = nib.load(str(fmri_path))
    n_volumes = int(img.shape[3]) if len(img.shape) >= 4 else 1
    n_tr = int(round(float(args.duration_seconds) / float(args.tr_seconds)))
    volume_indices = fmri_volume_indices(
        n_volumes,
        n_tr=n_tr,
        trim_volumes=args.fmri_trim_volumes,
    )
    if len(volume_indices) < 3:
        return [{**base_row, "roi": "", "n_volumes": n_volumes, "status": "too_few_fmri_volumes"}]

    for roi_name in roi_names:
        if roi_name not in FMRI_ROI_SPECS:
            rows.append({**base_row, "roi": roi_name, "status": "unknown_roi"})
            continue
        spec = FMRI_ROI_SPECS[roi_name]
        out_path = fmri_roi_output_path(args.fmri_dsm_root, meta, roi_name)
        try:
            kind = str(spec.get("kind", "roi_pattern"))
            if kind == "hpc_pfc_coupling":
                hippocampus_mask_paths = [Path(p) for p in spec["hippocampus_mask_paths"]]
                pfc_mask_paths = [Path(p) for p in spec["pfc_mask_paths"]]
                (
                    hpc_signal,
                    pfc_signal,
                    hpc_raw_voxels,
                    pfc_raw_voxels,
                    hpc_used_voxels,
                    pfc_used_voxels,
                ) = extract_dual_roi_mean_signals(
                    fmri_path,
                    hippocampus_mask_paths=hippocampus_mask_paths,
                    pfc_mask_paths=pfc_mask_paths,
                    volume_indices=volume_indices,
                    roi_threshold=args.fmri_roi_threshold,
                )
                if len(hpc_signal) < 3 or len(pfc_signal) < 3:
                    rows.append(
                        {
                            **base_row,
                            "roi": roi_name,
                            "roi_label": str(spec["label"]),
                            "n_hippocampus_voxels_raw": hpc_raw_voxels,
                            "n_pfc_voxels_raw": pfc_raw_voxels,
                            "n_hippocampus_voxels_used": hpc_used_voxels,
                            "n_pfc_voxels_used": pfc_used_voxels,
                            "n_volumes": n_volumes,
                            "n_dsm_volumes": len(volume_indices),
                            "status": "empty_hpc_pfc_signal",
                        }
                    )
                    continue

                hpc_z, hpc_mean, hpc_std = zscore_vector(hpc_signal)
                pfc_z, pfc_mean, pfc_std = zscore_vector(pfc_signal)
                window_corr, window_start, window_stop, effective_window = sliding_window_correlation(
                    hpc_z,
                    pfc_z,
                    args.hpc_pfc_window_volumes,
                )
                clipped_corr = np.clip(window_corr, -0.999999, 0.999999)
                window_corr_fisher_z = np.arctanh(clipped_corr).astype(np.float32)
                dsm = pairwise_absolute_difference(window_corr)
                dsm_condensed = condensed_upper_triangle(dsm)
                distance_metric = "absolute_window_corr_difference"
                if not args.dry_run:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        out_path,
                        dsm=dsm.astype(np.float32),
                        dsm_condensed=dsm_condensed.astype(np.float32),
                        volume_indices=volume_indices.astype(np.int32),
                        hippocampus_mean_bold=hpc_signal.astype(np.float32),
                        pfc_mean_bold=pfc_signal.astype(np.float32),
                        hippocampus_mean_z=hpc_z.astype(np.float32),
                        pfc_mean_z=pfc_z.astype(np.float32),
                        hpc_pfc_coactivation=(hpc_z * pfc_z).astype(np.float32),
                        hpc_pfc_window_corr=window_corr.astype(np.float32),
                        hpc_pfc_window_corr_fisher_z=window_corr_fisher_z.astype(np.float32),
                        hpc_pfc_window_start=window_start.astype(np.int32),
                        hpc_pfc_window_stop=window_stop.astype(np.int32),
                        source_fmri=np.asarray(str(fmri_path)),
                        source_behavior_npz=np.asarray(str(meta.path)),
                        analysis_unit=np.asarray(meta.analysis_unit),
                        roi=np.asarray(roi_name),
                        roi_label=np.asarray(str(spec["label"])),
                        hippocampus_mask_paths=np.asarray([str(p) for p in hippocampus_mask_paths]),
                        pfc_mask_paths=np.asarray([str(p) for p in pfc_mask_paths]),
                        n_hippocampus_voxels_raw=np.asarray(hpc_raw_voxels, dtype=np.int32),
                        n_pfc_voxels_raw=np.asarray(pfc_raw_voxels, dtype=np.int32),
                        n_hippocampus_voxels_used=np.asarray(hpc_used_voxels, dtype=np.int32),
                        n_pfc_voxels_used=np.asarray(pfc_used_voxels, dtype=np.int32),
                        n_roi_voxels_raw=np.asarray(hpc_raw_voxels + pfc_raw_voxels, dtype=np.int32),
                        n_roi_voxels_used=np.asarray(hpc_used_voxels + pfc_used_voxels, dtype=np.int32),
                        n_volumes=np.asarray(n_volumes, dtype=np.int32),
                        fmri_trim_volumes=np.asarray(int(args.fmri_trim_volumes), dtype=np.int16),
                        hpc_pfc_window_volumes=np.asarray(effective_window, dtype=np.int16),
                        requested_hpc_pfc_window_volumes=np.asarray(
                            int(args.hpc_pfc_window_volumes), dtype=np.int16
                        ),
                        hippocampus_mean_bold_mean=np.asarray(hpc_mean, dtype=np.float32),
                        hippocampus_mean_bold_std=np.asarray(hpc_std, dtype=np.float32),
                        pfc_mean_bold_mean=np.asarray(pfc_mean, dtype=np.float32),
                        pfc_mean_bold_std=np.asarray(pfc_std, dtype=np.float32),
                        distance_metric=np.asarray(distance_metric),
                        feature_model=np.asarray("roi_mean_sliding_window_hpc_pfc_correlation"),
                    )
                rows.append(
                    {
                        **base_row,
                        "roi": roi_name,
                        "roi_label": str(spec["label"]),
                        "output_npz": str(out_path),
                        "n_volumes": n_volumes,
                        "n_dsm_volumes": len(volume_indices),
                        "volume_start": int(volume_indices[0]),
                        "volume_stop": int(volume_indices[-1] + 1),
                        "n_hippocampus_voxels_raw": hpc_raw_voxels,
                        "n_pfc_voxels_raw": pfc_raw_voxels,
                        "n_hippocampus_voxels_used": hpc_used_voxels,
                        "n_pfc_voxels_used": pfc_used_voxels,
                        "n_roi_voxels_raw": hpc_raw_voxels + pfc_raw_voxels,
                        "n_roi_voxels_used": hpc_used_voxels + pfc_used_voxels,
                        "hpc_pfc_window_volumes": effective_window,
                        "requested_hpc_pfc_window_volumes": int(args.hpc_pfc_window_volumes),
                        "distance_metric": distance_metric,
                        "feature_model": "roi_mean_sliding_window_hpc_pfc_correlation",
                        "roi_mask_paths": ";".join(
                            [str(p) for p in hippocampus_mask_paths] + [str(p) for p in pfc_mask_paths]
                        ),
                        "status": "dry_run" if args.dry_run else "ok",
                    }
                )
                continue

            mask_paths = [Path(p) for p in spec["mask_paths"]]
            patterns, coords, raw_voxels, used_voxels = extract_fmri_roi_patterns(
                fmri_path,
                mask_paths=mask_paths,
                volume_indices=volume_indices,
                roi_threshold=args.fmri_roi_threshold,
                max_roi_voxels=args.fmri_max_roi_voxels,
            )
            if patterns.shape[1] == 0:
                rows.append(
                    {
                        **base_row,
                        "roi": roi_name,
                        "roi_label": str(spec["label"]),
                        "n_roi_voxels_raw": raw_voxels,
                        "n_roi_voxels_used": used_voxels,
                        "n_volumes": n_volumes,
                        "n_dsm_volumes": len(volume_indices),
                        "status": "empty_roi_patterns",
                    }
                )
                continue
            dsm = build_dsm(patterns, args.fmri_distance_metric)
            dsm_condensed = condensed_upper_triangle(dsm)
            if not args.dry_run:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                save_kwargs = {}
                if args.save_fmri_patterns:
                    save_kwargs["patterns_z"] = patterns.astype(np.float32)
                np.savez_compressed(
                    out_path,
                    dsm=dsm.astype(np.float32),
                    dsm_condensed=dsm_condensed.astype(np.float32),
                    roi_coords_ijk=coords.astype(np.int16),
                    volume_indices=volume_indices.astype(np.int32),
                    source_fmri=np.asarray(str(fmri_path)),
                    source_behavior_npz=np.asarray(str(meta.path)),
                    analysis_unit=np.asarray(meta.analysis_unit),
                    roi=np.asarray(roi_name),
                    roi_label=np.asarray(str(spec["label"])),
                    roi_mask_paths=np.asarray([str(p) for p in mask_paths]),
                    n_roi_voxels_raw=np.asarray(raw_voxels, dtype=np.int32),
                    n_roi_voxels_used=np.asarray(used_voxels, dtype=np.int32),
                    n_volumes=np.asarray(n_volumes, dtype=np.int32),
                    fmri_trim_volumes=np.asarray(int(args.fmri_trim_volumes), dtype=np.int16),
                    distance_metric=np.asarray(args.fmri_distance_metric),
                    **save_kwargs,
                )
            rows.append(
                {
                    **base_row,
                    "roi": roi_name,
                    "roi_label": str(spec["label"]),
                    "output_npz": str(out_path),
                    "n_volumes": n_volumes,
                    "n_dsm_volumes": len(volume_indices),
                    "volume_start": int(volume_indices[0]),
                    "volume_stop": int(volume_indices[-1] + 1),
                    "n_roi_voxels_raw": raw_voxels,
                    "n_roi_voxels_used": used_voxels,
                    "roi_mask_paths": ";".join(str(p) for p in mask_paths),
                    "status": "dry_run" if args.dry_run else "ok",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    **base_row,
                    "roi": roi_name,
                    "roi_label": str(spec["label"]),
                    "output_npz": str(out_path),
                    "status": f"error: {exc}",
                }
            )
    return rows


def safe_time_summary(frame_time: np.ndarray) -> Tuple[float, float]:
    time = np.asarray(frame_time, dtype=np.float32).reshape(-1)
    if time.size == 0:
        return np.nan, np.nan
    return float(np.nanmin(time)), float(np.nanmax(time))


def prepare_one(meta: BlockMeta, args: argparse.Namespace) -> Dict[str, object]:
    out_path = args.output_root / meta.relative_output_path
    indices, labels, spec_source = selected_ram_spec(meta.game)

    with np.load(meta.path, allow_pickle=True) as data:
        if "RAM" not in data.files:
            raise KeyError(f"Missing RAM key in {meta.path}")
        ram = np.asarray(data["RAM"])
        if ram.ndim != 2:
            raise ValueError(f"Expected RAM to be 2D, got shape {ram.shape}: {meta.path}")
        if ram.shape[1] <= int(indices.max()):
            raise ValueError(
                f"RAM has {ram.shape[1]} bytes, cannot select byte {int(indices.max())}: {meta.path}"
            )
        n_frames = int(ram.shape[0])
        frame_time = frame_time_vector(data, n_frames, duration_seconds=args.duration_seconds)
        time_min, time_max = safe_time_summary(frame_time)
        selected_ram = ram[:, indices].astype(np.float32)

    n_tr = int(round(float(args.duration_seconds) / float(args.tr_seconds)))
    ram_tr, frame_counts, frame_start, frame_stop = average_to_time_bins(
        selected_ram,
        frame_time,
        n_tr=n_tr,
        tr_seconds=args.tr_seconds,
    )
    ram_tr_z, raw_z_mean, raw_z_std = zscore_columns(ram_tr)
    frame_z, frame_hrf, hrf, hrf_time, frame_time, frame_dt = frame_level_hrf_features(
        selected_ram,
        frame_time,
        args,
    )
    ram_tr_hrf, hrf_frame_counts, _, _ = average_to_time_bins(
        frame_hrf,
        frame_time,
        n_tr=n_tr,
        tr_seconds=args.tr_seconds,
    )
    ram_tr_hrf_z, hrf_z_mean, hrf_z_std = zscore_columns(ram_tr_hrf)
    ram_tr_model = ram_tr_hrf_z
    valid_model_mask = hrf_frame_counts > 0
    dsm = build_dsm(ram_tr_model, args.distance_metric)
    dsm_condensed = condensed_upper_triangle(dsm)

    if not args.dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            dsm=dsm.astype(np.float32),
            dsm_condensed=dsm_condensed.astype(np.float32),
            ram_tr=ram_tr.astype(np.float32),
            ram_tr_z=ram_tr_z.astype(np.float32),
            ram_tr_hrf=ram_tr_hrf.astype(np.float32),
            ram_tr_hrf_z=ram_tr_hrf_z.astype(np.float32),
            ram_tr_model=ram_tr_model.astype(np.float32),
            ram_tr_delayed=ram_tr_model.astype(np.float32),
            valid_model_mask=valid_model_mask.astype(bool),
            valid_delay_mask=valid_model_mask.astype(bool),
            selected_ram_indices=indices.astype(np.int16),
            selected_ram_labels=labels,
            selected_ram_source=np.asarray(spec_source),
            frame_counts_per_tr=frame_counts.astype(np.int32),
            hrf_frame_counts_per_tr=hrf_frame_counts.astype(np.int32),
            frame_start=frame_start.astype(np.int32),
            frame_stop=frame_stop.astype(np.int32),
            ram_z_mean=raw_z_mean.astype(np.float32),
            ram_z_std=raw_z_std.astype(np.float32),
            ram_hrf_z_mean=hrf_z_mean.astype(np.float32),
            ram_hrf_z_std=hrf_z_std.astype(np.float32),
            hrf=hrf.astype(np.float32),
            hrf_time=hrf_time.astype(np.float32),
            frame_dt=np.asarray(frame_dt, dtype=np.float32),
            source_npz=np.asarray(str(meta.path)),
            analysis_unit=np.asarray(meta.analysis_unit),
            subject=np.asarray(meta.subject, dtype=np.int16),
            session=np.asarray(meta.session, dtype=np.int16),
            block=np.asarray(meta.block, dtype=np.int16),
            game=np.asarray(meta.game, dtype=np.int16),
            game_name=np.asarray(GAME_NAMES.get(meta.game, f"game_{meta.game}")),
            n_frames=np.asarray(n_frames, dtype=np.int32),
            n_tr=np.asarray(n_tr, dtype=np.int16),
            duration_seconds=np.asarray(float(args.duration_seconds), dtype=np.float32),
            tr_seconds=np.asarray(float(args.tr_seconds), dtype=np.float32),
            hrf_duration_seconds=np.asarray(float(args.hrf_duration_seconds), dtype=np.float32),
            hrf_peak_delay=np.asarray(float(args.hrf_peak_delay), dtype=np.float32),
            hrf_undershoot_delay=np.asarray(float(args.hrf_undershoot_delay), dtype=np.float32),
            hrf_undershoot_ratio=np.asarray(float(args.hrf_undershoot_ratio), dtype=np.float32),
            distance_metric=np.asarray(args.distance_metric),
            feature_model=np.asarray("frame_zscore_canonical_hrf_tr_mean_tr_zscore"),
        )

    return {
        "source_npz": str(meta.path),
        "output_npz": str(out_path),
        "subject": meta.subject,
        "session": meta.session,
        "block": meta.block,
        "game": meta.game,
        "game_name": GAME_NAMES.get(meta.game, f"game_{meta.game}"),
        "n_frames": n_frames,
        "time_min": time_min,
        "time_max": time_max,
        "assumed_duration_seconds": float(args.duration_seconds),
        "tr_seconds": float(args.tr_seconds),
        "n_tr": n_tr,
        "mean_frames_per_tr": float(np.mean(frame_counts)),
        "min_frames_per_tr": int(np.min(frame_counts)),
        "max_frames_per_tr": int(np.max(frame_counts)),
        "n_selected_ram_bytes": int(len(indices)),
        "selected_ram_indices": ",".join(str(int(i)) for i in indices),
        "selected_ram_labels": ",".join(str(x) for x in labels),
        "feature_model": "frame_zscore_canonical_hrf_tr_mean_tr_zscore",
        "frame_dt": frame_dt,
        "hrf_duration_seconds": float(args.hrf_duration_seconds),
        "n_hrf_samples": int(len(hrf)),
        "distance_metric": args.distance_metric,
        "status": "dry_run" if args.dry_run else "ok",
    }


def write_summary(manifest: pd.DataFrame, out_base: Path) -> None:
    lines: List[str] = []
    lines.append("RAM DSM preparation summary")
    lines.append("")
    lines.append(f"files: {len(manifest):,}")
    if not manifest.empty:
        lines.append("frame counts by game:")
        by_game = manifest.groupby(["game", "game_name"])["n_frames"].agg(["count", "min", "median", "max"])
        lines.append(by_game.to_string())
        lines.append("")
        lines.append("selected RAM bytes by game:")
        for _, row in manifest.sort_values(["game"]).drop_duplicates("game").iterrows():
            lines.append(
                f"game {int(row['game'])} {row['game_name']}: "
                f"{row['selected_ram_indices']} ({int(row['n_selected_ram_bytes'])} bytes)"
            )
    (out_base / "summary.txt").write_text("\n".join(lines).rstrip() + "\n")


def write_fmri_summary(manifest: pd.DataFrame, out_base: Path) -> None:
    lines: List[str] = []
    lines.append("fMRI ROI BOLD DSM preparation summary")
    lines.append("")
    lines.append(f"rows: {len(manifest):,}")
    if not manifest.empty and "status" in manifest:
        lines.append(manifest["status"].value_counts(dropna=False).to_string())
    if not manifest.empty and {"roi", "n_roi_voxels_used"}.issubset(manifest.columns):
        ok = manifest[manifest["status"] == "ok"].copy()
        if not ok.empty:
            lines.append("")
            lines.append("ROI voxel counts used:")
            roi_stats = ok.groupby(["roi", "roi_label"])["n_roi_voxels_used"].agg(["count", "min", "median", "max"])
            lines.append(roi_stats.to_string())
            if "hpc_pfc_window_volumes" in ok.columns:
                coupling = ok[ok["roi"] == "hippocampus_pfc_coupling"]
                if not coupling.empty:
                    lines.append("")
                    lines.append("Hippocampus-PFC coupling window volumes:")
                    lines.append(coupling["hpc_pfc_window_volumes"].value_counts(dropna=False).to_string())
    (out_base / "fmri_roi_dsm_summary.txt").write_text("\n".join(lines).rstrip() + "\n")


def write_thinker_summary(manifest: pd.DataFrame, comparison: pd.DataFrame, out_base: Path) -> None:
    lines: List[str] = []
    lines.append("Thinker imaginary-interval DSM preparation summary")
    lines.append("")
    lines.append(f"DSM rows: {len(manifest):,}")
    if not manifest.empty and "status" in manifest:
        lines.append(manifest["status"].value_counts(dropna=False).to_string())
    ok = manifest[manifest["status"] == "ok"].copy() if not manifest.empty and "status" in manifest else pd.DataFrame()
    if not ok.empty:
        lines.append("")
        lines.append("Feature dimensions:")
        dim_cols = [
            "representation",
            "per_step_dim",
            "transformed_per_step_dim",
            "max_imaginary_steps",
            "concat_dim",
            "pca_applied",
            "pca_explained_variance_sum",
        ]
        lines.append(ok[[c for c in dim_cols if c in ok.columns]].to_string(index=False))
    lines.append("")
    lines.append(f"Comparison rows: {len(comparison):,}")
    if not comparison.empty and "status" in comparison:
        lines.append(comparison["status"].value_counts(dropna=False).to_string())
    comp_ok = comparison[comparison["status"] == "ok"].copy() if not comparison.empty and "status" in comparison else pd.DataFrame()
    if not comp_ok.empty:
        lines.append("")
        lines.append("Top Spearman matches:")
        top = comp_ok.sort_values("spearman_rho", ascending=False).head(20)
        lines.append(
            top[
                [
                    "analysis_unit",
                    "representation",
                    "target",
                    "spearman_rho",
                    "pearson_r",
                    "n_pairs",
                ]
            ].to_string(index=False)
        )
    (out_base / "thinker_dsm_summary.txt").write_text("\n".join(lines).rstrip() + "\n")


def default_example_metas(input_root: Path) -> List[BlockMeta]:
    example_paths = [
        input_root / "sub-001" / "ses-01" / "sub001-ses01-block1-game1.npz",
        input_root / "sub-001" / "ses-01" / "sub001-ses01-block3-game2.npz",
    ]
    metas: List[BlockMeta] = []
    for path in example_paths:
        meta = parse_block_meta(path)
        if meta is not None and path.exists():
            metas.append(meta)
    return metas


def plot_example_thinker_comparisons(args: argparse.Namespace, rep_names: Sequence[str]) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    comparison_path = args.output_base / "thinker_dsm_comparison_manifest.csv"
    if not comparison_path.exists():
        return None
    comparison = pd.read_csv(comparison_path)
    metas = default_example_metas(args.input_root)
    if len(metas) == 0:
        return None
    fig_dir = args.output_base / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    n_rows = len(metas)
    n_cols = len(rep_names) + 1
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.0 * n_cols, 3.8 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )
    preferred_targets = [
        "ram",
        "left_hippocampus",
        "right_hippocampus",
        "hippocampus_mean",
        "pfc",
        "hippocampus_pfc_coupling",
    ]
    target_labels = {
        "ram": "RAM",
        "left_hippocampus": "LH",
        "right_hippocampus": "RH",
        "hippocampus_mean": "HPC",
        "pfc": "PFC",
        "hippocampus_pfc_coupling": "HPC-PFC",
    }
    for row, meta in enumerate(metas):
        for col, rep_name in enumerate(rep_names):
            ax = axes[row, col]
            path = thinker_rep_output_path(args.thinker_dsm_root, meta, rep_name)
            if not path.exists():
                ax.text(0.5, 0.5, "missing DSM", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue
            data = np.load(path)
            dsm = np.asarray(data["dsm"], dtype=float)
            vmax = float(np.nanpercentile(dsm, 99)) if np.isfinite(dsm).any() else 1.0
            im = ax.imshow(dsm, origin="lower", interpolation="nearest", cmap="viridis", vmin=0, vmax=vmax)
            title = rep_name.replace("_", " ")
            if bool(data["pca_applied"]):
                title += f"\nPCA {int(data['pca_components'])}"
            ax.set_title(f"{GAME_NAMES.get(meta.game, f'game {meta.game}')} | {title}", fontsize=8)
            ax.set_xlabel("TR bin")
            ax.set_ylabel("TR bin")
            fig.colorbar(im, ax=ax, shrink=0.72, label="1 - corr")

        ax = axes[row, -1]
        sub = comparison[(comparison["analysis_unit"] == meta.analysis_unit) & (comparison["status"] == "ok")]
        matrix = np.full((len(rep_names), len(preferred_targets)), np.nan, dtype=float)
        for r_i, rep_name in enumerate(rep_names):
            for t_i, target in enumerate(preferred_targets):
                hit = sub[(sub["representation"] == rep_name) & (sub["target"] == target)]
                if not hit.empty:
                    matrix[r_i, t_i] = float(hit.iloc[0]["spearman_rho"])
        im = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        ax.set_title(f"{GAME_NAMES.get(meta.game, f'game {meta.game}')} | DSM Spearman rho", fontsize=8)
        ax.set_xticks(np.arange(len(preferred_targets)))
        ax.set_xticklabels([target_labels[t] for t in preferred_targets], rotation=45, ha="right", fontsize=7)
        ax.set_yticks(np.arange(len(rep_names)))
        ax.set_yticklabels([r.replace("_", " ") for r in rep_names], fontsize=7)
        for r_i in range(matrix.shape[0]):
            for t_i in range(matrix.shape[1]):
                if np.isfinite(matrix[r_i, t_i]):
                    ax.text(t_i, r_i, f"{matrix[r_i, t_i]:.2f}", ha="center", va="center", fontsize=6)
        fig.colorbar(im, ax=ax, shrink=0.72, label="Spearman rho")
    out = fig_dir / "thinker_vs_target_dsm_game1_game2_examples.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_example_ram_dsms(args: argparse.Namespace) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    metas = default_example_metas(args.input_root)
    if len(metas) == 0:
        return None
    fig_dir = args.output_base / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(metas), figsize=(5.5 * len(metas), 4.8), constrained_layout=True)
    axes_arr = np.asarray(axes).reshape(-1)
    for ax, meta in zip(axes_arr, metas):
        path = args.output_root / meta.relative_output_path
        if not path.exists():
            ax.text(0.5, 0.5, "missing DSM", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        data = np.load(path)
        dsm = np.asarray(data["dsm"], dtype=float)
        vmax = float(np.nanpercentile(dsm, 99)) if np.isfinite(dsm).any() else 1.0
        im = ax.imshow(dsm, origin="lower", interpolation="nearest", cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(
            f"{GAME_NAMES.get(meta.game, f'game {meta.game}')}\n"
            f"HRF RAM DSM\n{meta.path.name}",
            fontsize=9,
        )
        ax.set_xlabel("TR bin (1 s)")
        ax.set_ylabel("TR bin (1 s)")
        fig.colorbar(im, ax=ax, shrink=0.82, label="Euclidean DSM")
    out = fig_dir / "ram_dsm_game1_game2_example_hrf.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_example_fmri_roi_dsms(args: argparse.Namespace, roi_names: Sequence[str]) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    metas = default_example_metas(args.input_root)
    if len(metas) == 0:
        return None
    fig_dir = args.output_base / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    n_rows = len(metas)
    n_cols = len(roi_names)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.0 * n_cols, 3.7 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )
    for row, meta in enumerate(metas):
        for col, roi_name in enumerate(roi_names):
            ax = axes[row, col]
            path = fmri_roi_output_path(args.fmri_dsm_root, meta, roi_name)
            if not path.exists():
                ax.text(0.5, 0.5, "missing DSM", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue
            data = np.load(path)
            dsm = np.asarray(data["dsm"], dtype=float)
            vmax = float(np.nanpercentile(dsm, 99)) if np.isfinite(dsm).any() else 1.0
            im = ax.imshow(dsm, origin="lower", interpolation="nearest", cmap="magma", vmin=0, vmax=vmax)
            roi_label = str(data["roi_label"])
            ax.set_title(
                f"{GAME_NAMES.get(meta.game, f'game {meta.game}')} | {fmri_session_dir(meta)}\n"
                f"{roi_label}",
                fontsize=8,
            )
            ax.set_xlabel("fMRI volume")
            ax.set_ylabel("fMRI volume")
            metric_label = str(data["distance_metric"])
            if metric_label == "correlation":
                metric_label = "correlation distance"
            elif metric_label == "absolute_window_corr_difference":
                metric_label = "|Delta window r|"
            fig.colorbar(im, ax=ax, shrink=0.75, label=metric_label)
    out = fig_dir / "fmri_roi_bold_dsm_game1_game2_examples.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare Cross-style RAM DSM NPZ files.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_DSM_ROOT)
    parser.add_argument("--skip-ram-dsms", action="store_true")
    parser.add_argument("--subjects", default=None, help="Comma-separated subject numbers, e.g. 1,2,3.")
    parser.add_argument("--sessions", default=None, help="Comma-separated session numbers, e.g. 1,2,3,4.")
    parser.add_argument("--games", default=None, help="Comma-separated game ids. Default: all games found.")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--duration-seconds", type=float, default=float(DEFAULT_DURATION_SECONDS))
    parser.add_argument("--tr-seconds", type=float, default=float(DEFAULT_TR_SECONDS))
    parser.add_argument("--hrf-duration-seconds", type=float, default=float(DEFAULT_HRF_DURATION_SECONDS))
    parser.add_argument("--hrf-peak-delay", type=float, default=6.0)
    parser.add_argument("--hrf-undershoot-delay", type=float, default=16.0)
    parser.add_argument("--hrf-undershoot-ratio", type=float, default=6.0)
    parser.add_argument(
        "--distance-metric",
        choices=["euclidean", "correlation"],
        default="euclidean",
        help="DSM metric over prepared RAM model features. Cross-style HDF default: euclidean.",
    )
    parser.add_argument("--prepare-fmri-dsms", action="store_true")
    parser.add_argument("--fmri-root", type=Path, default=DEFAULT_FMRI_ROOT)
    parser.add_argument("--fmri-image", default=DEFAULT_FMRI_IMAGE)
    parser.add_argument("--fmri-dsm-root", type=Path, default=DEFAULT_FMRI_DSM_ROOT)
    parser.add_argument(
        "--fmri-rois",
        nargs="+",
        default=list(FMRI_ROI_SPECS),
        choices=sorted(FMRI_ROI_SPECS),
    )
    parser.add_argument("--fmri-trim-volumes", type=int, default=DEFAULT_FMRI_TRIM_VOLUMES)
    parser.add_argument("--fmri-roi-threshold", type=float, default=0.0)
    parser.add_argument("--fmri-max-roi-voxels", type=int, default=DEFAULT_FMRI_MAX_ROI_VOXELS)
    parser.add_argument(
        "--hpc-pfc-window-volumes",
        type=int,
        default=DEFAULT_HPC_PFC_WINDOW_VOLUMES,
        help="Centered sliding-window length, in fMRI volumes, for the hippocampus-PFC coupling DSM.",
    )
    parser.add_argument(
        "--fmri-distance-metric",
        choices=["correlation", "euclidean"],
        default="correlation",
    )
    parser.add_argument("--prepare-thinker-dsms", action="store_true")
    parser.add_argument("--thinker-root", type=Path, default=DEFAULT_THINKER_ROOT)
    parser.add_argument("--thinker-dsm-root", type=Path, default=DEFAULT_THINKER_DSM_ROOT)
    parser.add_argument(
        "--thinker-reps",
        nargs="+",
        default=list(THINKER_REP_NAMES),
        choices=sorted(THINKER_REP_NAMES),
    )
    parser.add_argument("--thinker-pca-components", type=int, default=DEFAULT_THINKER_PCA_COMPONENTS)
    parser.add_argument("--thinker-pca-sample-steps", type=int, default=DEFAULT_THINKER_PCA_SAMPLE_STEPS)
    parser.add_argument("--thinker-max-feature-gb", type=float, default=DEFAULT_THINKER_MAX_FEATURE_GB)
    parser.add_argument("--thinker-random-seed", type=int, default=2026)
    parser.add_argument("--save-fmri-patterns", action="store_true")
    parser.add_argument("--make-example-pngs", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Only inspect files and write manifest/summary.")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    args.output_base.mkdir(parents=True, exist_ok=True)

    metas = gather_npz_files(
        args.input_root,
        subjects=parse_int_list(args.subjects),
        sessions=parse_int_list(args.sessions),
        games=parse_int_list(args.games),
        max_files=args.max_files,
    )
    if not metas:
        raise FileNotFoundError(f"No behavioral NPZ files found under {args.input_root}")

    print(f"[input] {len(metas)} NPZ files")

    if not args.skip_ram_dsms:
        print(f"[output] RAM DSM root: {args.output_root}")
        rows: List[Dict[str, object]] = []
        for i, meta in enumerate(metas, start=1):
            print(f"[ram {i}/{len(metas)}] {meta.path.relative_to(args.input_root)}", flush=True)
            try:
                rows.append(prepare_one(meta, args))
            except Exception as exc:
                rows.append(
                    {
                        "source_npz": str(meta.path),
                        "output_npz": str(args.output_root / meta.relative_output_path),
                        "subject": meta.subject,
                        "session": meta.session,
                        "block": meta.block,
                        "game": meta.game,
                        "game_name": GAME_NAMES.get(meta.game, f"game_{meta.game}"),
                        "status": f"error: {exc}",
                    }
                )
                print(f"  [error] {exc}", flush=True)

        manifest = pd.DataFrame(rows)
        manifest_path = args.output_base / "ram_dsm_manifest.csv"
        manifest.to_csv(manifest_path, index=False)
        write_summary(manifest, args.output_base)
        print(f"[done] wrote RAM manifest: {manifest_path}")
    else:
        print("[ram] skipped")

    if args.prepare_fmri_dsms:
        print(f"[output] fMRI ROI DSM root: {args.fmri_dsm_root}")
        fmri_rows: List[Dict[str, object]] = []
        for i, meta in enumerate(metas, start=1):
            print(f"[fmri {i}/{len(metas)}] {meta.path.relative_to(args.input_root)}", flush=True)
            fmri_rows.extend(prepare_fmri_roi_dsms(meta, args, args.fmri_rois))
        fmri_manifest = pd.DataFrame(fmri_rows)
        fmri_manifest_path = args.output_base / "fmri_roi_dsm_manifest.csv"
        fmri_manifest.to_csv(fmri_manifest_path, index=False)
        write_fmri_summary(fmri_manifest, args.output_base)
        print(f"[done] wrote fMRI manifest: {fmri_manifest_path}")

    if args.prepare_thinker_dsms:
        print(f"[output] Thinker DSM root: {args.thinker_dsm_root}")
        trace_groups = gather_trace_files(args.thinker_root, metas)
        thinker_rows: List[Dict[str, object]] = []
        comparison_rows: List[Dict[str, object]] = []
        for i, meta in enumerate(metas, start=1):
            trace_metas = trace_groups.get(meta.analysis_unit, [])
            print(
                f"[thinker {i}/{len(metas)}] {meta.path.relative_to(args.input_root)} "
                f"({len(trace_metas)} trace chunks)",
                flush=True,
            )
            for rep_name in args.thinker_reps:
                try:
                    thinker_rows.append(prepare_thinker_representation_dsm(meta, trace_metas, rep_name, args))
                except Exception as exc:
                    thinker_rows.append(
                        {
                            "source_behavior_npz": str(meta.path),
                            "subject": meta.subject,
                            "session": meta.session,
                            "block": meta.block,
                            "game": meta.game,
                            "game_name": GAME_NAMES.get(meta.game, f"game_{meta.game}"),
                            "analysis_unit": meta.analysis_unit,
                            "representation": rep_name,
                            "output_npz": str(thinker_rep_output_path(args.thinker_dsm_root, meta, rep_name)),
                            "status": f"error: {exc}",
                        }
                    )
                    print(f"  [error {rep_name}] {exc}", flush=True)
            comparison_rows.extend(compare_thinker_dsms_to_targets(meta, args.thinker_reps, args))
        thinker_manifest = pd.DataFrame(thinker_rows)
        thinker_manifest_path = args.output_base / "thinker_dsm_manifest.csv"
        thinker_manifest.to_csv(thinker_manifest_path, index=False)
        comparison_manifest = pd.DataFrame(comparison_rows)
        comparison_manifest_path = args.output_base / "thinker_dsm_comparison_manifest.csv"
        comparison_manifest.to_csv(comparison_manifest_path, index=False)
        write_thinker_summary(thinker_manifest, comparison_manifest, args.output_base)
        print(f"[done] wrote Thinker manifest: {thinker_manifest_path}")
        print(f"[done] wrote Thinker comparison manifest: {comparison_manifest_path}")

    if args.make_example_pngs:
        ram_fig = plot_example_ram_dsms(args)
        if ram_fig is not None:
            print(f"[figure] {ram_fig}")
        if args.prepare_fmri_dsms:
            fmri_fig = plot_example_fmri_roi_dsms(args, args.fmri_rois)
            if fmri_fig is not None:
                print(f"[figure] {fmri_fig}")
        if args.prepare_thinker_dsms:
            thinker_fig = plot_example_thinker_comparisons(args, args.thinker_reps)
            if thinker_fig is not None:
                print(f"[figure] {thinker_fig}")


if __name__ == "__main__":
    main()
