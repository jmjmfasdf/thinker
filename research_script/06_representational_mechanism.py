#!/usr/bin/env python3
"""
Section 6 representational mechanism analysis.

This script builds real-step aligned representational summaries from filtered
Thinker traces and prepares fMRI-aligned ROI-RSA inputs for:

1. Spectral geometry of imagined tree representations.
2. Drift/diffusion proxy analyses over tree centroid trajectories.
3. DMDc-style intrinsic vs input-driven dynamics.
4. ROI voxel representation similarity analysis against Thinker latent RDMs.

Input layout
------------
The loader supports both trace layouts found in this repository:

  test/sub001/ses-04/sub001-ses04-block5-game1_000.npy
  test/sub001-ses04-block8-game2/video_stat_000.npy

If both layouts contain the same subject/session/block/game/chunk, the nested
subXXX/ses-0X file is preferred to avoid double-counting.

fMRI alignment
--------------
Trace session and block numbers define the fMRI run, while game id only
filters the trace files to analyze. If no `--sessions` argument is provided,
all matching trace sessions are analyzed. For example, subject 1/game 2 with
`--sessions 3` reads `test/sub001/ses-03/*game2*.npy`; block 2, 4, and 8 then
map to:

  /home/jeongmin/fmri/atari/derivatives/ants_mni/sub001-3/Session2/
  /home/jeongmin/fmri/atari/derivatives/ants_mni/sub001-3/Session4/
  /home/jeongmin/fmri/atari/derivatives/ants_mni/sub001-3/Session8/

Each block is concatenated across chunk files and analyzed as one fMRI-aligned
analysis unit. By default, the first and last 60 fMRI volumes are excluded
from analysis (600-volume runs become 480-volume analysis windows).

For brain RSA, provide a binary or probabilistic ROI mask in the same MNI grid
as the fMRI data via `--roi-mask`. The script extracts voxel patterns from the
trimmed fMRI volumes, builds a brain RDM across volumes, resamples the selected
Thinker latent trajectory to the same number of volumes, builds a model RDM,
and correlates the two condensed RDMs.

Outputs
-------
research_script/outputs/06_representational_mechanism/
  results/
    real_step_geometry_metrics.csv
    dmdc_analysis_unit_summary.csv
    fmri_alignment_manifest.csv
    roi_rsa_summary.csv
    roi_rsa_<subXXX-ses>_Session<block>_game<game>_<space>.npz
    rsa_samples_<subXXX-ses>_Session<block>_game<game>.npz
    latent_vectors.npz
    summary.txt
  figures/
    fig_metric_overview.png
"""
from __future__ import annotations

import argparse
import gc
import math
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib-cache"
os.environ["XDG_CACHE_HOME"] = "/tmp"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore", category=RuntimeWarning)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_ROOT = ROOT / "test"
DEFAULT_FMRI_ROOT = Path("/home/jeongmin/fmri/atari/derivatives/ants_mni")
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "outputs" / "06_representational_mechanism"
DEFAULT_FMRI_TRIM_VOLUMES = 60

EPS = 1e-12
NOOP_ACTION = 0
FRAGMENT_STATUSES = (1, 2)

GAME_LABELS = {
    1: "pong",
    2: "spaceinvaders",
}

GAME_TITLES = {
    1: "Pong",
    2: "Space Invaders",
}

@dataclass(frozen=True)
class TraceMeta:
    subject: int
    session: int
    block: int
    game: int
    chunk: int
    path: Path
    layout: str

    @property
    def key(self) -> Tuple[int, int, int, int, int]:
        return (self.subject, self.session, self.block, self.game, self.chunk)

    @property
    def source_file_name(self) -> str:
        return self.path.name

    @property
    def block_key(self) -> str:
        return (
            f"sub{self.subject:03d}_ses{self.session:02d}_"
            f"block{self.block:02d}_game{self.game}"
        )

    @property
    def subject_game(self) -> str:
        return f"sub{self.subject:03d}_game{self.game}"

    @property
    def fmri_subject(self) -> str:
        return f"sub{self.subject:03d}-{self.session}"

    @property
    def fmri_session(self) -> int:
        return self.block

    @property
    def analysis_unit(self) -> str:
        return f"{self.fmri_subject}_Session{self.block}_game{self.game}"


def parse_int_list(raw: str | None) -> set[int] | None:
    if raw is None or raw.strip() == "":
        return None
    out: set[int] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        out.add(int(item))
    return out


def parse_trace_meta(path: Path) -> TraceMeta | None:
    nested = re.match(
        r"sub(\d+)-ses(\d+)-block(\d+)-game(\d+)_(\d+)\.npy$",
        path.name,
    )
    if nested is not None:
        return TraceMeta(
            subject=int(nested.group(1)),
            session=int(nested.group(2)),
            block=int(nested.group(3)),
            game=int(nested.group(4)),
            chunk=int(nested.group(5)),
            path=path,
            layout="nested",
        )

    legacy_file = re.match(r"video_stat_(\d+)\.npy$", path.name)
    legacy_parent = re.match(
        r"sub(\d+)-ses(\d+)-block(\d+)-game(\d+)$",
        path.parent.name,
    )
    if legacy_file is not None and legacy_parent is not None:
        return TraceMeta(
            subject=int(legacy_parent.group(1)),
            session=int(legacy_parent.group(2)),
            block=int(legacy_parent.group(3)),
            game=int(legacy_parent.group(4)),
            chunk=int(legacy_file.group(1)),
            path=path,
            layout="legacy_video_stat",
        )

    return None


def gather_trace_files(
    input_root: Path,
    *,
    subjects: set[int] | None,
    sessions: set[int] | None,
    games: set[int] | None,
    game_session_offset: int | None,
    max_files: int | None,
) -> List[TraceMeta]:
    by_key: Dict[Tuple[int, int, int, int, int], TraceMeta] = {}
    for path in sorted(input_root.rglob("*.npy")):
        meta = parse_trace_meta(path)
        if meta is None:
            continue
        if subjects is not None and meta.subject not in subjects:
            continue
        if sessions is not None and meta.session not in sessions:
            continue
        if games is not None and meta.game not in games:
            continue
        if sessions is None and games is not None and game_session_offset is not None:
            if meta.session != meta.game + game_session_offset:
                continue

        prev = by_key.get(meta.key)
        if prev is None or (prev.layout != "nested" and meta.layout == "nested"):
            by_key[meta.key] = meta

    metas = sorted(
        by_key.values(),
        key=lambda m: (m.subject, m.session, m.game, m.block, m.chunk, str(m.path)),
    )
    if max_files is not None:
        metas = metas[:max_files]
    return metas


def load_npy_dict(path: Path) -> Dict[str, object]:
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
        item = obj.item()
        if isinstance(item, dict):
            return item
    if hasattr(obj, "files"):
        return {key: obj[key] for key in obj.files}
    raise ValueError(f"Cannot parse file as dict-like npy/npz: {path}")


def ensure_2d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        return arr[:, None]
    return arr.reshape(arr.shape[0], -1)


def to_action_ids(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim == 1:
        return arr.astype(int).reshape(-1)
    return np.argmax(arr.reshape(arr.shape[0], -1), axis=1).astype(int)


def softmax_rows(x: np.ndarray) -> np.ndarray:
    arr = ensure_2d(x)
    arr = arr - np.nanmax(arr, axis=1, keepdims=True)
    ex = np.exp(np.clip(arr, -60.0, 60.0))
    denom = np.nansum(ex, axis=1, keepdims=True)
    return ex / np.where(denom <= EPS, 1.0, denom)


def normalize_policy_rows(x: np.ndarray) -> np.ndarray:
    arr = ensure_2d(x)
    clipped = np.clip(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    sums = clipped.sum(axis=1, keepdims=True)
    direct = clipped / np.where(sums <= EPS, 1.0, sums)
    fallback = softmax_rows(arr)
    return np.where(sums > EPS, direct, fallback)


def entropy_rows(x: np.ndarray) -> np.ndarray:
    prob = normalize_policy_rows(x)
    return -np.sum(prob * np.log(prob + EPS), axis=1)


def top2_gap_rows(x: np.ndarray) -> np.ndarray:
    arr = ensure_2d(x)
    if arr.shape[1] < 2:
        return np.zeros(arr.shape[0], dtype=float)
    part = np.partition(arr, -2, axis=1)[:, -2:]
    return part[:, 1] - part[:, 0]


def chosen_values(x: np.ndarray, actions: np.ndarray) -> np.ndarray:
    arr = ensure_2d(x)
    actions = np.asarray(actions, dtype=int).reshape(-1)
    n = min(len(arr), len(actions))
    out = np.full(n, np.nan, dtype=float)
    rows = np.arange(n)
    valid = (actions[:n] >= 0) & (actions[:n] < arr.shape[1])
    out[valid] = arr[rows[valid], actions[:n][valid]]
    return out


def js_divergence_rows(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    pp = normalize_policy_rows(p)
    qq = normalize_policy_rows(q)
    n = min(len(pp), len(qq))
    pp = pp[:n]
    qq = qq[:n]
    m = 0.5 * (pp + qq)
    kl_pm = np.sum(pp * (np.log(pp + EPS) - np.log(m + EPS)), axis=1)
    kl_qm = np.sum(qq * (np.log(qq + EPS) - np.log(m + EPS)), axis=1)
    return 0.5 * (kl_pm + kl_qm)


def prev_index_of_status(status_indices: np.ndarray, ref_indices: np.ndarray) -> np.ndarray:
    if len(status_indices) == 0:
        return np.full(len(ref_indices), -1, dtype=int)
    pos = np.searchsorted(status_indices, ref_indices) - 1
    return np.where(pos >= 0, status_indices[pos], -1).astype(int)


def pool_vector(raw: object) -> np.ndarray:
    if raw is None:
        return np.array([], dtype=np.float32)
    arr = np.asarray(raw, dtype=np.float32)
    if arr.size == 0:
        return np.array([], dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim == 3:
        return arr.reshape(arr.shape[0], -1).mean(axis=1).astype(np.float32)
    if arr.ndim == 2 and arr.shape[0] == 128:
        return arr.mean(axis=1).astype(np.float32)
    return arr.reshape(-1).astype(np.float32)


def pool_indices(vector_list: Sequence[object], indices: np.ndarray) -> np.ndarray:
    rows: List[np.ndarray] = []
    for idx in indices:
        if idx < 0 or idx >= len(vector_list):
            continue
        vec = pool_vector(vector_list[int(idx)])
        if vec.size > 0:
            rows.append(vec)
    if not rows:
        return np.empty((0, 0), dtype=np.float32)
    min_dim = min(row.size for row in rows)
    return np.vstack([row[:min_dim] for row in rows]).astype(np.float32)


def sample_rows(x: np.ndarray, max_rows: int) -> np.ndarray:
    if len(x) <= max_rows:
        return x
    idx = np.linspace(0, len(x) - 1, max_rows, dtype=int)
    return x[idx]


def pairwise_sq_dists(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    norms = np.sum(arr * arr, axis=1, keepdims=True)
    sq = norms + norms.T - 2.0 * (arr @ arr.T)
    return np.maximum(sq, 0.0)


def pairwise_distance_summary(nodes: np.ndarray, max_nodes: int) -> Dict[str, float]:
    out = {
        "n_pairwise_nodes_used": 0,
        "pairwise_dist_mean": np.nan,
        "pairwise_dist_std": np.nan,
        "pairwise_dist_max": np.nan,
    }
    if nodes.ndim != 2 or len(nodes) < 2:
        return out
    used = sample_rows(nodes, max_nodes)
    sq = pairwise_sq_dists(used)
    tri = np.triu_indices(len(used), k=1)
    vals = np.sqrt(sq[tri])
    out.update(
        {
            "n_pairwise_nodes_used": int(len(used)),
            "pairwise_dist_mean": float(np.mean(vals)),
            "pairwise_dist_std": float(np.std(vals)),
            "pairwise_dist_max": float(np.max(vals)),
        }
    )
    return out


def participation_ratio_from_eigs(eigs: np.ndarray) -> float:
    vals = np.asarray(eigs, dtype=np.float64)
    vals = vals[np.isfinite(vals) & (vals > EPS)]
    if vals.size == 0:
        return np.nan
    return float((np.sum(vals) ** 2) / np.sum(vals ** 2))


def local_effective_dimensionality(nodes: np.ndarray) -> float:
    if nodes.ndim != 2 or len(nodes) < 3:
        return np.nan
    x = np.asarray(nodes, dtype=np.float64)
    x = x - np.mean(x, axis=0, keepdims=True)
    if not np.isfinite(x).all() or np.linalg.norm(x) <= EPS:
        return np.nan
    _, s, _ = np.linalg.svd(x, full_matrices=False)
    eigs = (s ** 2) / max(len(x) - 1, 1)
    return participation_ratio_from_eigs(eigs)


def standardize_node_cloud(x: np.ndarray, mode: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if mode == "none":
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if mode != "local_zscore":
        raise ValueError(f"Unknown node standardization: {mode}")
    mean = np.nanmean(arr, axis=0, keepdims=True)
    std = np.nanstd(arr, axis=0, keepdims=True)
    return np.nan_to_num((arr - mean) / np.where(std <= EPS, 1.0, std))


def diffusion_geometry(
    root_vec: np.ndarray,
    imag_nodes: np.ndarray,
    *,
    max_nodes: int,
    epsilon_quantile: float,
    powers: Sequence[int],
    standardization: str,
) -> Dict[str, float]:
    result: Dict[str, float] = {
        "n_spectral_nodes_used": 0,
        "epsilon": np.nan,
        "lambda1": np.nan,
        "lambda2": np.nan,
        "lambda3": np.nan,
        "lambda4": np.nan,
        "lambda5": np.nan,
        "lambda2_ratio": np.nan,
        "spectral_gap": np.nan,
        "diffusion_entropy": np.nan,
    }
    for power in powers:
        result[f"diffusion_eff_rank_t{power}"] = np.nan
        result[f"diffdist_t{power}_mean"] = np.nan
        result[f"diffdist_t{power}_std"] = np.nan
        result[f"diffdist_t{power}_max"] = np.nan

    if root_vec.size == 0 or imag_nodes.ndim != 2 or len(imag_nodes) < 2:
        return result

    dim = min(root_vec.size, imag_nodes.shape[1])
    root = root_vec[:dim][None, :]
    nodes = imag_nodes[:, :dim]
    nodes = sample_rows(nodes, max(2, max_nodes - 1))
    x_raw = np.vstack([root, nodes])
    x = standardize_node_cloud(x_raw, standardization)

    sq = pairwise_sq_dists(x)
    tri = sq[np.triu_indices(len(x), k=1)]
    tri = tri[np.isfinite(tri) & (tri > EPS)]
    if tri.size == 0:
        return result

    epsilon = float(np.quantile(tri, epsilon_quantile))
    epsilon = max(epsilon, EPS)
    kernel = np.exp(-sq / epsilon)
    degrees = np.sum(kernel, axis=1)
    valid = degrees > EPS
    if np.sum(valid) < 2:
        return result
    kernel = kernel[valid][:, valid]
    degrees = degrees[valid]

    inv_sqrt_d = 1.0 / np.sqrt(np.maximum(degrees, EPS))
    sym = kernel * inv_sqrt_d[:, None] * inv_sqrt_d[None, :]
    vals, vecs = np.linalg.eigh(sym)
    order = np.argsort(vals)[::-1]
    vals = np.clip(np.real(vals[order]), 0.0, 1.0)
    vecs = np.real(vecs[:, order])

    n_report = min(5, len(vals))
    for i in range(n_report):
        result[f"lambda{i + 1}"] = float(vals[i])

    lam1 = vals[0] if len(vals) > 0 else np.nan
    lam2 = vals[1] if len(vals) > 1 else np.nan
    result["n_spectral_nodes_used"] = int(len(vals))
    result["epsilon"] = epsilon
    result["lambda2_ratio"] = float(lam2 / lam1) if lam1 > EPS and np.isfinite(lam2) else np.nan
    result["spectral_gap"] = float(lam1 - lam2) if np.isfinite(lam1) and np.isfinite(lam2) else np.nan

    pos = vals[1:][vals[1:] > EPS]
    if pos.size > 0:
        prob = pos / np.sum(pos)
        result["diffusion_entropy"] = float(-np.sum(prob * np.log(prob + EPS)))

    if not valid[0]:
        return result

    psi = vecs * inv_sqrt_d[:, None]
    root_idx = 0
    other = np.arange(len(psi))
    other = other[other != root_idx]
    nontrivial = np.arange(1, len(vals))
    for power in powers:
        weights = vals[nontrivial] ** (2 * power)
        result[f"diffusion_eff_rank_t{power}"] = participation_ratio_from_eigs(weights)
        if other.size > 0 and nontrivial.size > 0:
            diffs = psi[other][:, nontrivial] - psi[root_idx, nontrivial][None, :]
            dist2 = np.sum((diffs ** 2) * weights[None, :], axis=1)
            dist = np.sqrt(np.maximum(dist2, 0.0))
            result[f"diffdist_t{power}_mean"] = float(np.mean(dist))
            result[f"diffdist_t{power}_std"] = float(np.std(dist))
            result[f"diffdist_t{power}_max"] = float(np.max(dist))

    return result


def path_length(nodes: np.ndarray) -> float:
    if nodes.ndim != 2 or len(nodes) < 2:
        return np.nan
    diffs = np.diff(nodes.astype(np.float64), axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def segment_slope(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2 or np.nanstd(arr) <= EPS:
        return 0.0
    x = np.linspace(0.0, 1.0, arr.size)
    return float(scipy_stats.linregress(x, arr).slope)


def extract_file_real_steps(
    meta: TraceMeta,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    data = load_npy_dict(meta.path)
    tree = data["tree_reps"]
    status = np.asarray(data["status"]).reshape(-1)
    im_vp_vectors = data["im_vp_vectors"]

    human_action = to_action_ids(np.asarray(data["human_action"]))
    thinker_action = to_action_ids(np.asarray(data["thinker_action"]))
    actor_logits = np.asarray(data["actor_policy"]).reshape(len(status), -1)
    env_return = np.asarray(data["env_return"], dtype=float).reshape(-1)

    root_policy_raw = ensure_2d(np.asarray(tree["root_policy"], dtype=float))
    cur_policy_raw = ensure_2d(np.asarray(tree["cur_policy"], dtype=float))
    root_qs = ensure_2d(np.asarray(tree["root_qs_mean"], dtype=float))
    cur_qs = ensure_2d(np.asarray(tree["cur_qs_mean"], dtype=float))
    root_qs_max = ensure_2d(np.asarray(tree.get("root_qs_max", tree["root_qs_mean"]), dtype=float))
    root_ns = ensure_2d(np.asarray(tree.get("root_ns", np.zeros_like(root_policy_raw)), dtype=float))
    rollout_return = np.asarray(tree["rollout_return"], dtype=float).reshape(-1)
    max_rollout_return = np.asarray(tree["max_rollout_return"], dtype=float).reshape(-1)
    root_v = np.asarray(tree["root_v"], dtype=float).reshape(-1)
    cur_v = np.asarray(tree["cur_v"], dtype=float).reshape(-1)

    t = min(
        len(status),
        len(im_vp_vectors),
        len(human_action),
        len(thinker_action),
        len(actor_logits),
        len(env_return),
        len(root_policy_raw),
        len(cur_policy_raw),
        len(root_qs),
        len(cur_qs),
        len(root_qs_max),
        len(root_ns),
        len(rollout_return),
        len(max_rollout_return),
        len(root_v),
        len(cur_v),
    )
    status = status[:t]
    human_action = human_action[:t]
    thinker_action = thinker_action[:t]
    actor_logits = actor_logits[:t]
    env_return = env_return[:t]
    root_policy_raw = root_policy_raw[:t]
    cur_policy_raw = cur_policy_raw[:t]
    root_qs = root_qs[:t]
    cur_qs = cur_qs[:t]
    root_qs_max = root_qs_max[:t]
    root_ns = root_ns[:t]
    rollout_return = rollout_return[:t]
    max_rollout_return = max_rollout_return[:t]
    root_v = root_v[:t]
    cur_v = cur_v[:t]

    actor_probs = softmax_rows(actor_logits)
    root_policy = normalize_policy_rows(root_policy_raw)
    cur_policy = normalize_policy_rows(cur_policy_raw)

    entropy_actor = entropy_rows(actor_probs)
    entropy_root = entropy_rows(root_policy)
    entropy_cur = entropy_rows(cur_policy)
    q_gap = top2_gap_rows(root_qs)
    cur_q_gap = top2_gap_rows(cur_qs)
    q_gap_max = top2_gap_rows(root_qs_max)
    actor_policy_gap = top2_gap_rows(actor_probs)
    search_jsd_actor_root = js_divergence_rows(actor_probs, root_policy)
    search_jsd_actor_cur = js_divergence_rows(actor_probs, cur_policy)
    search_jsd_root_cur = js_divergence_rows(root_policy, cur_policy)
    tree_width = np.sum(root_ns > EPS, axis=1).astype(float)
    rollout_spread = np.abs(max_rollout_return - rollout_return)

    real_idx = np.flatnonzero(status == 0)
    if args.max_real_steps_per_file is not None:
        real_idx = real_idx[: args.max_real_steps_per_file]
    if real_idx.size == 0:
        return pd.DataFrame(), np.empty((0, 0)), np.empty((0, 0)), np.empty((0, 0))

    status2_idx = np.flatnonzero(status == 2)
    prev_s2_all = prev_index_of_status(status2_idx, real_idx)
    prev_s2_all = np.where(prev_s2_all >= 0, prev_s2_all, np.maximum(real_idx - 1, 0)).astype(int)

    rows: List[Dict[str, object]] = []
    centroids: List[np.ndarray] = []
    roots: List[np.ndarray] = []
    finals: List[np.ndarray] = []
    episode_in_file = 0
    episode_step = 0

    for real_pos, idx_global in enumerate(real_idx):
        prev_real_global = int(real_idx[real_pos - 1]) if real_pos > 0 else -1
        between = np.arange(prev_real_global + 1, int(idx_global), dtype=int)
        if real_pos > 0 and between.size > 0 and np.isin(status[between], [1, 3]).any():
            episode_in_file += 1
            episode_step = 0

        imag_idx = between[status[between] == 2] if between.size else np.array([], dtype=int)
        prev_s2 = int(prev_s2_all[real_pos])
        used_fallback = 0
        if imag_idx.size == 0:
            fallback = prev_s2 if 0 <= prev_s2 < t else int(idx_global)
            imag_idx = np.array([fallback], dtype=int)
            used_fallback = 1

        root_vec = pool_vector(im_vp_vectors[int(idx_global)])
        imag_nodes = pool_indices(im_vp_vectors, imag_idx)
        if imag_nodes.size == 0 and root_vec.size > 0:
            imag_nodes = root_vec[None, :]
            used_fallback = 1

        if root_vec.size == 0 and imag_nodes.size > 0:
            root_vec = imag_nodes[0].copy()
            used_fallback = 1

        dim = min(root_vec.size, imag_nodes.shape[1] if imag_nodes.ndim == 2 else 0)
        if dim == 0:
            root_vec = np.zeros(128, dtype=np.float32)
            imag_nodes = root_vec[None, :]
            dim = 128
            used_fallback = 1

        root_vec = root_vec[:dim].astype(np.float32)
        imag_nodes = imag_nodes[:, :dim].astype(np.float32)
        centroid = np.mean(imag_nodes, axis=0).astype(np.float32)
        final_vec = imag_nodes[-1].astype(np.float32)

        centered_dist = np.linalg.norm(imag_nodes - centroid[None, :], axis=1)
        root_dist = np.linalg.norm(imag_nodes - root_vec[None, :], axis=1)
        direct = float(np.linalg.norm(final_vec - imag_nodes[0])) if len(imag_nodes) >= 2 else np.nan
        plen = path_length(imag_nodes)
        tortuosity = float(plen / max(direct, EPS)) if np.isfinite(plen) and np.isfinite(direct) else np.nan

        geom = diffusion_geometry(
            root_vec=root_vec,
            imag_nodes=imag_nodes,
            max_nodes=args.max_spectral_nodes,
            epsilon_quantile=args.epsilon_quantile,
            powers=args.diffusion_powers,
            standardization=args.node_standardization,
        )
        geom.update(pairwise_distance_summary(imag_nodes, args.max_pairwise_nodes))

        pre_reset_returns: List[float] = []
        if between.size > 0:
            reset_in_between = between[np.isin(status[between], [1, 3])]
            for reset_idx in reset_in_between:
                s2_before = between[(between < reset_idx) & (status[between] == 2)]
                if len(s2_before) > 0:
                    pre_reset_returns.append(float(rollout_return[s2_before[-1]]))
        var_rollout_return = float(np.var(pre_reset_returns)) if len(pre_reset_returns) >= 2 else np.nan

        imag_valid = imag_idx[(imag_idx >= 0) & (imag_idx < t)]
        q_gap_imag = q_gap[imag_valid] if imag_valid.size else np.array([], dtype=float)
        spread_imag = rollout_spread[imag_valid] if imag_valid.size else np.array([], dtype=float)

        row: Dict[str, object] = {
            "file": str(meta.path),
            "source_file_name": meta.source_file_name,
            "layout": meta.layout,
            "subject": meta.subject,
            "session": meta.session,
            "block": meta.block,
            "game": meta.game,
            "game_name": GAME_TITLES.get(meta.game, f"Game {meta.game}"),
            "chunk": meta.chunk,
            "subject_game": meta.subject_game,
            "fmri_subject": meta.fmri_subject,
            "fmri_session": meta.fmri_session,
            "analysis_unit": meta.analysis_unit,
            "block_key": meta.block_key,
            "episode_in_file": episode_in_file,
            "episode_step": episode_step,
            "real_pos_file": real_pos,
            "global_idx": int(idx_global),
            "prev_s2_idx": int(prev_s2),
            "human_action": int(human_action[idx_global]),
            "thinker_action": int(thinker_action[idx_global]),
            "is_human_noop": int(human_action[idx_global] == NOOP_ACTION),
            "is_thinker_noop": int(thinker_action[idx_global] == NOOP_ACTION),
            "env_return": float(env_return[idx_global]),
            "n_imag_nodes": int(len(imag_nodes)),
            "n_status_1_since_prev_real": int(np.sum(status[between] == 1)) if between.size else 0,
            "n_status_2_since_prev_real": int(np.sum(status[between] == 2)) if between.size else 0,
            "n_status_3_since_prev_real": int(np.sum(status[between] == 3)) if between.size else 0,
            "used_fallback": int(used_fallback),
            "entropy_actor": float(entropy_actor[idx_global]),
            "entropy_root_policy": float(entropy_root[idx_global]),
            "entropy_cur_policy": float(entropy_cur[idx_global]),
            "actor_policy_gap": float(actor_policy_gap[idx_global]),
            "q_gap": float(q_gap[idx_global]),
            "q_gap_prev_s2": float(q_gap[prev_s2]) if 0 <= prev_s2 < t else np.nan,
            "cur_q_gap": float(cur_q_gap[idx_global]),
            "q_gap_max": float(q_gap_max[idx_global]),
            "var_rollout_return": var_rollout_return,
            "rollout_spread_prev_s2": float(rollout_spread[prev_s2]) if 0 <= prev_s2 < t else np.nan,
            "search_jsd_actor_root": float(search_jsd_actor_root[prev_s2]) if 0 <= prev_s2 < len(search_jsd_actor_root) else np.nan,
            "search_jsd_actor_cur": float(search_jsd_actor_cur[idx_global]),
            "search_jsd_root_cur": float(search_jsd_root_cur[idx_global]),
            "tree_width_prev_s2": float(tree_width[prev_s2]) if 0 <= prev_s2 < t else np.nan,
            "root_value": float(root_v[idx_global]),
            "cur_value": float(cur_v[idx_global]),
            "centroid_norm": float(np.linalg.norm(centroid)),
            "root_norm": float(np.linalg.norm(root_vec)),
            "final_imag_norm": float(np.linalg.norm(final_vec)),
            "centroid_to_root_dist": float(np.linalg.norm(centroid - root_vec)),
            "final_to_root_dist": float(np.linalg.norm(final_vec - root_vec)),
            "imag_dispersion_mean": float(np.mean(centered_dist)),
            "imag_dispersion_std": float(np.std(centered_dist)),
            "imag_dispersion_max": float(np.max(centered_dist)),
            "root_distance_mean": float(np.mean(root_dist)),
            "root_distance_std": float(np.std(root_dist)),
            "root_distance_max": float(np.max(root_dist)),
            "trajectory_path_length": plen,
            "trajectory_direct_dist": direct,
            "trajectory_tortuosity": tortuosity,
            "diffusion_proxy": float(np.mean(centered_dist)),
            "local_effective_dim": local_effective_dimensionality(imag_nodes),
            "q_gap_imag_mean": float(np.nanmean(q_gap_imag)) if q_gap_imag.size else np.nan,
            "q_gap_imag_final": float(q_gap_imag[-1]) if q_gap_imag.size else np.nan,
            "q_gap_imag_slope": segment_slope(q_gap_imag),
            "rollout_spread_imag_mean": float(np.nanmean(spread_imag)) if spread_imag.size else np.nan,
            "rollout_spread_imag_final": float(spread_imag[-1]) if spread_imag.size else np.nan,
            "rollout_spread_imag_slope": segment_slope(spread_imag),
        }
        row.update(geom)
        rows.append(row)
        centroids.append(centroid)
        roots.append(root_vec)
        finals.append(final_vec)
        episode_step += 1

    del data
    gc.collect()

    min_dim = min(vec.size for vec in centroids)
    centroid_arr = np.vstack([vec[:min_dim] for vec in centroids]).astype(np.float32)
    root_arr = np.vstack([vec[:min_dim] for vec in roots]).astype(np.float32)
    final_arr = np.vstack([vec[:min_dim] for vec in finals]).astype(np.float32)
    return pd.DataFrame(rows), centroid_arr, root_arr, final_arr


def add_bout_features(df: pd.DataFrame, target: str, prefix: str) -> pd.DataFrame:
    out = df.copy()
    out[f"{prefix}_noop_lag1"] = np.nan
    out[f"{prefix}_noop_onset"] = 0
    out[f"{prefix}_commit"] = 0
    out[f"{prefix}_bout_id"] = -1
    out[f"{prefix}_bout_age"] = 0
    out[f"{prefix}_bout_length"] = 0

    for _, idx in out.groupby("block_key", sort=False).groups.items():
        loc = np.asarray(list(idx), dtype=int)
        vals = out.loc[loc, target].to_numpy(dtype=int)
        prev = np.r_[0, vals[:-1]]
        onset = (vals == 1) & (prev == 0)
        commit = (vals == 0) & (prev == 1)
        out.loc[loc, f"{prefix}_noop_lag1"] = prev
        out.loc[loc, f"{prefix}_noop_onset"] = onset.astype(int)
        out.loc[loc, f"{prefix}_commit"] = commit.astype(int)

        run_id = -1
        i = 0
        while i < len(vals):
            if vals[i] != 1:
                i += 1
                continue
            j = i
            while j < len(vals) and vals[j] == 1:
                j += 1
            run_id += 1
            run_len = j - i
            rows = loc[i:j]
            out.loc[rows, f"{prefix}_bout_id"] = run_id
            out.loc[rows, f"{prefix}_bout_age"] = np.arange(1, run_len + 1)
            out.loc[rows, f"{prefix}_bout_length"] = run_len
            i = j
    return out


def add_drift_features(df: pd.DataFrame, centroid: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    out["drift_norm"] = np.nan
    out["drift_cosine_to_root_delta"] = np.nan
    out["drift_diffusion_ratio"] = np.nan
    out["centroid_step_index_within_block"] = 0

    for _, idx in out.groupby("block_key", sort=False).groups.items():
        loc = np.asarray(list(idx), dtype=int)
        out.loc[loc, "centroid_step_index_within_block"] = np.arange(len(loc))
        if len(loc) < 2:
            continue
        diffs = np.diff(centroid[loc].astype(np.float64), axis=0)
        drift = np.linalg.norm(diffs, axis=1)
        out.loc[loc[1:], "drift_norm"] = drift
        diffusion = out.loc[loc[1:], "diffusion_proxy"].to_numpy(dtype=float)
        out.loc[loc[1:], "drift_diffusion_ratio"] = drift / np.maximum(diffusion, EPS)
    return out


def fit_dmdc(
    df: pd.DataFrame,
    centroid: np.ndarray,
    root: np.ndarray,
    *,
    dmd_dim: int,
    ridge_alpha: float,
    min_pairs: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    metric_cols = [
        "dmdc_intrinsic_norm",
        "dmdc_input_norm",
        "dmdc_intrinsic_input_ratio",
        "dmdc_pred_norm",
        "dmdc_step_norm",
        "dmdc_residual_norm",
    ]
    for col in metric_cols:
        out[col] = np.nan

    summaries: List[Dict[str, object]] = []

    for analysis_unit, sg in out.groupby("analysis_unit", sort=False):
        pair_left: List[int] = []
        pair_right: List[int] = []
        for _, block_idx in sg.groupby("block_key", sort=False).groups.items():
            loc = np.asarray(list(block_idx), dtype=int)
            if len(loc) >= 2:
                pair_left.extend(loc[:-1].tolist())
                pair_right.extend(loc[1:].tolist())

        if len(pair_left) < min_pairs:
            summaries.append(
                {
                    "analysis_unit": analysis_unit,
                    "subject_game": str(sg["subject_game"].iloc[0]),
                    "fmri_subject": str(sg["fmri_subject"].iloc[0]),
                    "fmri_session": int(sg["fmri_session"].iloc[0]),
                    "n_pairs": len(pair_left),
                    "status": "too_few_pairs",
                }
            )
            continue

        left = np.asarray(pair_left, dtype=int)
        right = np.asarray(pair_right, dtype=int)
        z_all = centroid[sg.index.to_numpy()].astype(np.float64)
        u_all = root[sg.index.to_numpy()].astype(np.float64)
        n_components_z = min(dmd_dim, z_all.shape[1], max(1, len(z_all) - 1))
        n_components_u = min(dmd_dim, u_all.shape[1], max(1, len(u_all) - 1))
        if n_components_z < 1 or n_components_u < 1:
            continue

        z_scaler = StandardScaler()
        u_scaler = StandardScaler()
        z_pca = PCA(n_components=n_components_z, random_state=0)
        u_pca = PCA(n_components=n_components_u, random_state=0)

        z_pca.fit(z_scaler.fit_transform(z_all))
        u_pca.fit(u_scaler.fit_transform(u_all))

        z_left = z_pca.transform(z_scaler.transform(centroid[left]))
        z_right = z_pca.transform(z_scaler.transform(centroid[right]))
        u_left = u_pca.transform(u_scaler.transform(root[left]))

        design = np.hstack([z_left, u_left])
        reg = ridge_alpha * np.eye(design.shape[1])
        coef = np.linalg.solve(design.T @ design + reg, design.T @ z_right)

        coef_z = coef[:n_components_z]
        coef_u = coef[n_components_z:]
        intrinsic = z_left @ coef_z
        input_part = u_left @ coef_u
        pred = intrinsic + input_part
        residual = z_right - pred
        step = z_right - z_left

        intrinsic_norm = np.linalg.norm(intrinsic, axis=1)
        input_norm = np.linalg.norm(input_part, axis=1)
        pred_norm = np.linalg.norm(pred, axis=1)
        step_norm = np.linalg.norm(step, axis=1)
        residual_norm = np.linalg.norm(residual, axis=1)
        ratio = intrinsic_norm / np.maximum(input_norm, EPS)

        out.loc[left, "dmdc_intrinsic_norm"] = intrinsic_norm
        out.loc[left, "dmdc_input_norm"] = input_norm
        out.loc[left, "dmdc_intrinsic_input_ratio"] = ratio
        out.loc[left, "dmdc_pred_norm"] = pred_norm
        out.loc[left, "dmdc_step_norm"] = step_norm
        out.loc[left, "dmdc_residual_norm"] = residual_norm

        a_matrix = coef_z.T
        b_matrix = coef_u.T
        try:
            a_radius = float(np.max(np.abs(np.linalg.eigvals(a_matrix))))
        except np.linalg.LinAlgError:
            a_radius = np.nan
        try:
            b_svals = np.linalg.svd(b_matrix, compute_uv=False)
            b_spectral = float(b_svals[0])
        except np.linalg.LinAlgError:
            b_spectral = np.nan

        summaries.append(
            {
                "analysis_unit": analysis_unit,
                "subject_game": str(sg["subject_game"].iloc[0]),
                "fmri_subject": str(sg["fmri_subject"].iloc[0]),
                "fmri_session": int(sg["fmri_session"].iloc[0]),
                "subject": int(sg["subject"].iloc[0]),
                "trace_session": int(sg["session"].iloc[0]),
                "block": int(sg["block"].iloc[0]),
                "game": int(sg["game"].iloc[0]),
                "n_pairs": int(len(left)),
                "state_dim": int(n_components_z),
                "input_dim": int(n_components_u),
                "a_spectral_radius": a_radius,
                "b_spectral_norm": b_spectral,
                "b_fro_norm": float(np.linalg.norm(b_matrix)),
                "mean_intrinsic_norm": float(np.nanmean(intrinsic_norm)),
                "mean_input_norm": float(np.nanmean(input_norm)),
                "mean_intrinsic_input_ratio": float(np.nanmean(ratio)),
                "mean_residual_norm": float(np.nanmean(residual_norm)),
                "status": "ok",
            }
        )

    return out, pd.DataFrame(summaries)


def read_nifti_shape(path: Path) -> Tuple[str, int, float]:
    try:
        import nibabel as nib
    except Exception:
        return "", 0, np.nan
    try:
        img = nib.load(str(path))
        shape = tuple(int(x) for x in img.shape)
        zooms = img.header.get_zooms()
        n_volumes = shape[3] if len(shape) >= 4 else 1
        tr = float(zooms[3]) if len(zooms) >= 4 else np.nan
        return "x".join(str(x) for x in shape), int(n_volumes), tr
    except Exception:
        return "", 0, np.nan


def discover_fmri_run(
    fmri_root: Path,
    fmri_subject: str,
    fmri_session: int,
    image_name: str,
) -> Dict[str, object] | None:
    session_dir = fmri_root / fmri_subject / f"Session{fmri_session}"
    if not session_dir.exists():
        return None
    image_path = session_dir / image_name
    if not image_path.exists() and image_name == "s5_wfiltered_func_data.nii":
        image_path = session_dir / "wfiltered_func_data.nii"
    if not image_path.exists():
        return None
    shape, n_volumes, tr = read_nifti_shape(image_path)
    return {
        "fmri_subject": fmri_subject,
        "fmri_session": int(fmri_session),
        "fmri_session_dir": str(session_dir),
        "fmri_path": str(image_path),
        "fmri_shape": shape,
        "n_volumes": n_volumes,
        "tr": tr,
    }


def resample_matrix_rows(values: np.ndarray, n_out: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if n_out <= 0:
        return np.empty((0, arr.shape[1] if arr.ndim == 2 else 0), dtype=np.float32)
    if arr.size == 0 or len(arr) == 0:
        return np.zeros((n_out, 0), dtype=np.float32)
    arr = np.nan_to_num(arr.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if len(arr) == 1:
        return np.repeat(arr.astype(np.float32), n_out, axis=0)
    x_old = np.linspace(0.0, 1.0, len(arr))
    x_new = np.linspace(0.0, 1.0, n_out)
    out = np.empty((n_out, arr.shape[1]), dtype=np.float32)
    for col in range(arr.shape[1]):
        out[:, col] = np.interp(x_new, x_old, arr[:, col]).astype(np.float32)
    return out


def analysis_volume_window(n_volumes: int, trim_volumes: int) -> Tuple[int, int, int, str]:
    n_volumes = max(0, int(n_volumes))
    trim = max(0, int(trim_volumes))
    if n_volumes > 2 * trim:
        start = trim
        stop = n_volumes - trim
        return start, stop, stop - start, "block_trace_to_matching_fmri_session"
    return 0, 0, 0, "fmri_run_too_short_after_trim"


def build_fmri_alignment_manifest(
    df: pd.DataFrame,
    *,
    fmri_root: Path,
    image_name: str,
    fmri_trim_volumes: int,
) -> pd.DataFrame:
    manifest_rows: List[Dict[str, object]] = []

    for analysis_unit, sg in df.groupby("analysis_unit", sort=True):
        sg_sorted = sg.sort_values(["chunk", "real_pos_file"]).reset_index(drop=True)
        first = sg_sorted.iloc[0]
        fmri_subject = str(first["fmri_subject"])
        fmri_session = int(first["fmri_session"])
        fmri_run = discover_fmri_run(fmri_root, fmri_subject, fmri_session, image_name)

        trace_rows = len(sg_sorted)
        trace_files = int(sg_sorted["file"].nunique())
        trace_chunks = int(sg_sorted["chunk"].nunique())
        source_files = ";".join(sorted(sg_sorted["file"].unique().tolist()))

        base_manifest = {
            "analysis_unit": analysis_unit,
            "subject_game": str(first["subject_game"]),
            "subject": int(first["subject"]),
            "trace_session": int(first["session"]),
            "block": int(first["block"]),
            "game": int(first["game"]),
            "fmri_subject": fmri_subject,
            "fmri_session": fmri_session,
            "trace_rows_analysis_unit": trace_rows,
            "trace_files_analysis_unit": trace_files,
            "trace_chunks_analysis_unit": trace_chunks,
            "source_files": source_files,
        }

        if fmri_run is None:
            manifest_rows.append(
                {
                    **base_manifest,
                    "alignment_mode": "no_matching_fmri_run_found",
                }
            )
            continue

        n_volumes = int(fmri_run.get("n_volumes", 0))
        trim = max(0, int(fmri_trim_volumes))
        analysis_volume_start, analysis_volume_stop, n_analysis_volumes, alignment_mode = (
            analysis_volume_window(n_volumes, trim)
        )

        manifest_rows.append(
            {
                **base_manifest,
                **fmri_run,
                "raw_volume_start": 0,
                "raw_volume_stop": n_volumes,
                "trim_front_volumes": trim,
                "trim_back_volumes": trim,
                "volume_start": analysis_volume_start,
                "volume_stop": analysis_volume_stop,
                "n_analysis_volumes": n_analysis_volumes,
                "alignment_mode": alignment_mode,
            }
        )

    return pd.DataFrame(manifest_rows)


def clean_pattern_columns(x: np.ndarray) -> Tuple[np.ndarray, int, np.ndarray]:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        n_cols = arr.shape[1] if arr.ndim == 2 else 0
        return np.empty((arr.shape[0] if arr.ndim == 2 else 0, 0), dtype=np.float32), 0, np.zeros(n_cols, dtype=bool)
    finite = np.isfinite(arr).all(axis=0)
    std = np.nanstd(arr[:, finite], axis=0) if np.any(finite) else np.array([], dtype=float)
    keep = np.zeros(arr.shape[1], dtype=bool)
    keep[np.where(finite)[0]] = std > EPS
    arr = np.nan_to_num(arr[:, keep], nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size == 0:
        return arr.astype(np.float32), 0, keep
    mean = arr.mean(axis=0, keepdims=True)
    scale = arr.std(axis=0, keepdims=True)
    arr = (arr - mean) / np.where(scale <= EPS, 1.0, scale)
    return arr.astype(np.float32), int(arr.shape[1]), keep


def load_roi_mask(mask_path: Path, ref_img: object, threshold: float) -> Tuple[np.ndarray, int]:
    try:
        import nibabel as nib
    except Exception as exc:
        raise RuntimeError("nibabel is required for ROI-RSA fMRI loading.") from exc

    mask_img = nib.load(str(mask_path))
    mask_data = np.asarray(mask_img.get_fdata(dtype=np.float32))
    mask_data = np.squeeze(mask_data)
    if mask_data.ndim == 4:
        mask_data = mask_data[..., 0]
    if mask_data.ndim != 3:
        raise ValueError(f"ROI mask must be 3D after squeezing, got shape {mask_data.shape}: {mask_path}")
    ref_shape = tuple(int(x) for x in ref_img.shape[:3])
    if tuple(mask_data.shape) != ref_shape:
        raise ValueError(
            f"ROI mask shape {mask_data.shape} does not match fMRI spatial shape {ref_shape}: {mask_path}"
        )
    mask = np.isfinite(mask_data) & (mask_data > float(threshold))
    return mask, int(mask.sum())


def extract_roi_patterns(
    fmri_path: Path,
    *,
    roi_mask: Path,
    roi_threshold: float,
    volume_indices: np.ndarray,
    max_roi_voxels: int,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    try:
        import nibabel as nib
    except Exception as exc:
        raise RuntimeError("nibabel is required for ROI-RSA fMRI loading.") from exc

    img = nib.load(str(fmri_path))
    mask, raw_voxels = load_roi_mask(roi_mask, img, roi_threshold)
    coords = np.column_stack(np.where(mask)).astype(np.int64)
    if coords.size == 0:
        return np.empty((0, 0), dtype=np.float32), coords, raw_voxels, 0

    if max_roi_voxels > 0 and len(coords) > max_roi_voxels:
        keep = np.linspace(0, len(coords) - 1, max_roi_voxels, dtype=int)
        coords = coords[keep]

    volume_indices = np.asarray(volume_indices, dtype=int).reshape(-1)
    n_volumes = len(volume_indices)
    patterns = np.empty((n_volumes, len(coords)), dtype=np.float32)
    proxy = img.dataobj
    x = coords[:, 0].astype(int)
    y = coords[:, 1].astype(int)
    z = coords[:, 2].astype(int)
    for row, volume in enumerate(volume_indices):
        vol = np.asarray(proxy[:, :, :, int(volume)], dtype=np.float32)
        patterns[row] = vol[x, y, z]

    patterns, used_voxels, keep_cols = clean_pattern_columns(patterns)
    coords = coords[keep_cols]
    return patterns, coords.astype(np.int16), raw_voxels, used_voxels


def rdm_condensed(patterns: np.ndarray, metric: str) -> np.ndarray:
    arr, n_features, _ = clean_pattern_columns(patterns)
    if arr.shape[0] < 2 or n_features < 1:
        return np.empty(0, dtype=np.float32)
    if metric == "correlation" and n_features < 2:
        return np.empty(0, dtype=np.float32)
    dist = pdist(arr.astype(np.float64), metric=metric)
    return np.nan_to_num(dist, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def correlate_rdms(a: np.ndarray, b: np.ndarray, method: str) -> Tuple[float, float, int]:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    n = min(len(aa), len(bb))
    if n == 0:
        return np.nan, np.nan, 0
    aa = aa[:n]
    bb = bb[:n]
    valid = np.isfinite(aa) & np.isfinite(bb)
    aa = aa[valid]
    bb = bb[valid]
    if len(aa) < 3 or np.nanstd(aa) <= EPS or np.nanstd(bb) <= EPS:
        return np.nan, np.nan, int(len(aa))
    if method == "pearson":
        r, p = scipy_stats.pearsonr(aa, bb)
    else:
        r, p = scipy_stats.spearmanr(aa, bb)
    return float(r), float(p), int(len(aa))


def choose_volume_rows(n_rows: int, max_rows: int) -> np.ndarray:
    if max_rows <= 0 or n_rows <= max_rows:
        return np.arange(n_rows, dtype=int)
    return np.linspace(0, n_rows - 1, max_rows, dtype=int)


def run_roi_rsa(
    df: pd.DataFrame,
    latent_spaces: Mapping[str, np.ndarray],
    manifest: pd.DataFrame,
    *,
    roi_mask: Path,
    roi_name: str,
    rsa_spaces: Sequence[str],
    rsa_distance: str,
    rsa_correlation: str,
    rsa_max_volumes: int,
    roi_threshold: float,
    max_roi_voxels: int,
    result_dir: Path,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if manifest.empty:
        return pd.DataFrame(rows)

    roi_label = roi_name if roi_name else roi_mask.stem.replace(".nii", "")
    for _, fmri_row in manifest.sort_values("analysis_unit").iterrows():
        analysis_unit = str(fmri_row["analysis_unit"])
        base = {
            "analysis_unit": analysis_unit,
            "roi_name": roi_label,
            "roi_mask": str(roi_mask),
            "subject_game": fmri_row.get("subject_game", ""),
            "subject": fmri_row.get("subject", np.nan),
            "trace_session": fmri_row.get("trace_session", np.nan),
            "block": fmri_row.get("block", np.nan),
            "game": fmri_row.get("game", np.nan),
            "fmri_subject": fmri_row.get("fmri_subject", ""),
            "fmri_session": fmri_row.get("fmri_session", np.nan),
            "fmri_path": fmri_row.get("fmri_path", ""),
            "volume_start": fmri_row.get("volume_start", np.nan),
            "volume_stop": fmri_row.get("volume_stop", np.nan),
            "n_analysis_volumes_raw": fmri_row.get("n_analysis_volumes", 0),
            "rsa_distance": rsa_distance,
            "rsa_correlation": rsa_correlation,
        }

        if fmri_row.get("alignment_mode") != "block_trace_to_matching_fmri_session":
            rows.append({**base, "model_space": "", "status": str(fmri_row.get("alignment_mode", "no_fmri"))})
            continue

        sg = df[df["analysis_unit"] == analysis_unit].sort_values(["chunk", "real_pos_file"])
        if sg.empty:
            rows.append({**base, "model_space": "", "status": "no_trace_rows"})
            continue

        volume_start = int(fmri_row["volume_start"])
        volume_stop = int(fmri_row["volume_stop"])
        n_analysis = int(fmri_row["n_analysis_volumes"])
        if n_analysis < 3:
            rows.append({**base, "model_space": "", "status": "too_few_fmri_volumes"})
            continue

        try:
            keep_volumes = choose_volume_rows(n_analysis, rsa_max_volumes)
            volume_in_session = np.arange(volume_start, volume_stop, dtype=int)[keep_volumes]
            brain_patterns, coords, raw_voxels, used_voxels = extract_roi_patterns(
                Path(str(fmri_row["fmri_path"])),
                roi_mask=roi_mask,
                roi_threshold=roi_threshold,
                volume_indices=volume_in_session,
                max_roi_voxels=max_roi_voxels,
            )
        except Exception as exc:
            rows.append({**base, "model_space": "", "status": f"roi_load_error: {exc}"})
            continue

        brain_rdm = rdm_condensed(brain_patterns, rsa_distance)
        if brain_rdm.size == 0:
            rows.append(
                {
                    **base,
                    "model_space": "",
                    "n_roi_voxels_raw": raw_voxels,
                    "n_roi_voxels_used": used_voxels,
                    "n_rsa_volumes": len(keep_volumes),
                    "status": "empty_brain_rdm",
                }
            )
            continue

        trace_idx = sg.index.to_numpy(dtype=int)
        source_trace_row_float = np.linspace(0, max(len(trace_idx) - 1, 0), n_analysis)[keep_volumes]
        for space in rsa_spaces:
            if space not in latent_spaces:
                rows.append({**base, "model_space": space, "status": "unknown_model_space"})
                continue
            model_full = resample_matrix_rows(latent_spaces[space][trace_idx], n_analysis)
            model_patterns = model_full[keep_volumes]
            model_rdm = rdm_condensed(model_patterns, rsa_distance)
            r, p, n_pairs = correlate_rdms(brain_rdm, model_rdm, rsa_correlation)
            status = "ok" if np.isfinite(r) else "invalid_rdm_correlation"
            out_path = result_dir / f"roi_rsa_{analysis_unit}_{space}.npz"
            np.savez_compressed(
                out_path,
                brain_rdm=brain_rdm.astype(np.float32),
                model_rdm=model_rdm.astype(np.float32),
                volume_in_session=volume_in_session.astype(np.int32),
                source_trace_row_float=source_trace_row_float.astype(np.float32),
                roi_coords_ijk=coords.astype(np.int16),
                trace_row_index=trace_idx.astype(np.int64),
            )
            rows.append(
                {
                    **base,
                    "model_space": space,
                    "n_trace_rows": int(len(trace_idx)),
                    "n_roi_voxels_raw": raw_voxels,
                    "n_roi_voxels_used": used_voxels,
                    "n_rsa_volumes": int(len(keep_volumes)),
                    "n_rdm_pairs": n_pairs,
                    "rsa_r": r,
                    "rsa_p": p,
                    "output_npz": str(out_path),
                    "status": status,
                }
            )

    return pd.DataFrame(rows)


def write_rsa_samples(
    df: pd.DataFrame,
    centroid: np.ndarray,
    root: np.ndarray,
    out_dir: Path,
    *,
    max_steps: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for analysis_unit, sg in df.groupby("analysis_unit", sort=True):
        idx = sg.index.to_numpy(dtype=int)
        if idx.size < 2:
            continue
        if idx.size > max_steps:
            keep = np.linspace(0, idx.size - 1, max_steps, dtype=int)
            idx = idx[keep]
        centroid_rdm = np.sqrt(pairwise_sq_dists(centroid[idx]))
        root_rdm = np.sqrt(pairwise_sq_dists(root[idx]))
        meta = df.loc[idx, ["subject", "session", "block", "game", "chunk", "real_pos_file", "is_human_noop"]]
        np.savez_compressed(
            out_dir / f"rsa_samples_{analysis_unit}.npz",
            row_index=idx.astype(np.int64),
            centroid_rdm=centroid_rdm.astype(np.float32),
            root_rdm=root_rdm.astype(np.float32),
            meta=meta.to_records(index=False),
        )


def plot_metric_overview(df: pd.DataFrame, out_path: Path) -> None:
    metrics = [
        "lambda2_ratio",
        "diffusion_proxy",
        "local_effective_dim",
        "drift_diffusion_ratio",
        "dmdc_intrinsic_input_ratio",
    ]
    metrics = [m for m in metrics if m in df.columns]
    if not metrics or df.empty:
        return
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.3 * len(metrics), 4.0), squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        vals0 = df.loc[df["is_human_noop"] == 0, metric].replace([np.inf, -np.inf], np.nan).dropna()
        vals1 = df.loc[df["is_human_noop"] == 1, metric].replace([np.inf, -np.inf], np.nan).dropna()
        if vals0.empty or vals1.empty:
            ax.text(0.5, 0.5, "insufficient data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["action", "NOOP"])
            ax.set_title(metric)
            continue
        parts = ax.violinplot([vals0.to_numpy(), vals1.to_numpy()], positions=[0, 1], showmeans=True)
        for body in parts["bodies"]:
            body.set_facecolor("#93c5fd")
            body.set_edgecolor("#1f2937")
            body.set_alpha(0.75)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["action", "NOOP"])
        ax.set_title(metric)
        ax.grid(axis="y", color="#d1d5db", lw=0.8, alpha=0.7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def ensure_dirs(base: Path) -> Tuple[Path, Path]:
    result_dir = base / "results"
    figure_dir = base / "figures"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    return result_dir, figure_dir


def summarize(
    df: pd.DataFrame,
    dmd_summary: pd.DataFrame,
    manifest: pd.DataFrame,
    roi_summary: pd.DataFrame,
) -> str:
    lines: List[str] = []
    lines.append("Section 6 representational mechanism summary")
    lines.append("")
    lines.append(f"real-step rows: {len(df):,}")
    lines.append(f"subjects: {df['subject'].nunique() if not df.empty else 0}")
    lines.append(f"subject-game groups: {df['subject_game'].nunique() if not df.empty else 0}")
    lines.append(f"fMRI analysis units: {df['analysis_unit'].nunique() if not df.empty else 0}")
    lines.append(f"fMRI subject folders: {df['fmri_subject'].nunique() if not df.empty else 0}")
    lines.append(f"trace files: {df['file'].nunique() if not df.empty else 0}")
    lines.append(f"trace blocks: {df['block_key'].nunique() if not df.empty else 0}")
    if not df.empty:
        lines.append(f"mean imagined nodes per real step: {df['n_imag_nodes'].mean():.2f}")
        lines.append(f"fallback rate: {df['used_fallback'].mean():.4f}")
        noop = df[df["is_human_noop"] == 1]
        action = df[df["is_human_noop"] == 0]
        for metric in ["lambda2_ratio", "diffusion_proxy", "local_effective_dim", "dmdc_intrinsic_input_ratio"]:
            if metric in df.columns:
                lines.append(
                    f"{metric}: NOOP mean={noop[metric].mean():.4g}, "
                    f"action mean={action[metric].mean():.4g}"
                )
    lines.append("")
    lines.append(f"DMDc groups: {len(dmd_summary):,}")
    if not dmd_summary.empty and "status" in dmd_summary.columns:
        lines.append(dmd_summary["status"].value_counts(dropna=False).to_string())
    lines.append("")
    if not manifest.empty:
        n_fmri = int(manifest.get("fmri_path", pd.Series(dtype=object)).dropna().nunique())
        n_vol = int(manifest.get("n_volumes", pd.Series(dtype=float)).fillna(0).sum())
        n_analysis_vol = int(manifest.get("n_analysis_volumes", pd.Series(dtype=float)).fillna(0).sum())
        lines.append(f"matched fMRI runs: {n_fmri}")
        lines.append(f"matched fMRI volumes raw: {n_vol:,}")
        lines.append(f"matched fMRI volumes after trim: {n_analysis_vol:,}")
    else:
        lines.append("matched fMRI runs: 0")
    lines.append("")
    if roi_summary.empty:
        lines.append("ROI RSA: skipped or no valid ROI comparisons")
    else:
        ok = roi_summary[roi_summary.get("status", pd.Series(dtype=object)) == "ok"]
        lines.append(f"ROI RSA comparisons: {len(ok):,}/{len(roi_summary):,} ok")
        if not ok.empty and "model_space" in ok.columns:
            means = ok.groupby("model_space")["rsa_r"].mean().sort_index()
            lines.append("mean ROI RSA r by model space:")
            lines.append(means.to_string())
    return "\n".join(lines).rstrip() + "\n"


def run_analysis(args: argparse.Namespace) -> None:
    result_dir, figure_dir = ensure_dirs(args.output_dir)
    metas = gather_trace_files(
        args.input_root,
        subjects=parse_int_list(args.subjects),
        sessions=parse_int_list(args.sessions),
        games=parse_int_list(args.games),
        game_session_offset=args.game_session_offset,
        max_files=args.max_files,
    )
    if not metas:
        raise FileNotFoundError(f"No trace .npy files found under {args.input_root}")

    print(f"[input] using {len(metas)} trace files")
    frames: List[pd.DataFrame] = []
    centroid_blocks: List[np.ndarray] = []
    root_blocks: List[np.ndarray] = []
    final_blocks: List[np.ndarray] = []

    for i, meta in enumerate(metas, start=1):
        print(
            f"[{i}/{len(metas)}] sub{meta.subject:03d} ses{meta.session:02d} "
            f"block{meta.block} game{meta.game} chunk{meta.chunk}: {meta.path}"
        )
        frame, centroid, root, final = extract_file_real_steps(meta, args)
        if frame.empty:
            continue
        frames.append(frame)
        centroid_blocks.append(centroid)
        root_blocks.append(root)
        final_blocks.append(final)

    if not frames:
        raise RuntimeError("No real-step rows were produced.")

    df = pd.concat(frames, ignore_index=True)
    df["_latent_row"] = np.arange(len(df), dtype=int)
    min_dim = min(block.shape[1] for block in centroid_blocks if block.size)
    centroid_arr = np.vstack([block[:, :min_dim] for block in centroid_blocks]).astype(np.float32)
    root_arr = np.vstack([block[:, :min_dim] for block in root_blocks]).astype(np.float32)
    final_arr = np.vstack([block[:, :min_dim] for block in final_blocks]).astype(np.float32)

    df = df.sort_values(["subject", "session", "game", "block", "chunk", "real_pos_file"]).reset_index(drop=True)
    sort_order = df["_latent_row"].to_numpy(dtype=int)
    if len(sort_order) != len(centroid_arr):
        raise RuntimeError("Latent vector length mismatch after table construction.")
    centroid_arr = centroid_arr[sort_order]
    root_arr = root_arr[sort_order]
    final_arr = final_arr[sort_order]
    df = df.drop(columns=["_latent_row"])

    df = add_bout_features(df, "is_human_noop", "human")
    df = add_bout_features(df, "is_thinker_noop", "thinker")
    df = add_drift_features(df, centroid_arr)
    df, dmd_summary = fit_dmdc(
        df,
        centroid_arr,
        root_arr,
        dmd_dim=args.dmd_dim,
        ridge_alpha=args.dmd_ridge_alpha,
        min_pairs=args.dmd_min_pairs,
    )

    manifest = build_fmri_alignment_manifest(
        df,
        fmri_root=args.fmri_root,
        image_name=args.fmri_image,
        fmri_trim_volumes=args.fmri_trim_volumes,
    )

    df.to_csv(result_dir / "real_step_geometry_metrics.csv", index=False)
    dmd_summary.to_csv(result_dir / "dmdc_analysis_unit_summary.csv", index=False)
    manifest.to_csv(result_dir / "fmri_alignment_manifest.csv", index=False)
    np.savez_compressed(
        result_dir / "latent_vectors.npz",
        centroid=centroid_arr,
        root=root_arr,
        final_imag=final_arr,
        delta_centroid_root=(centroid_arr - root_arr).astype(np.float32),
        row_index=np.arange(len(df), dtype=np.int64),
    )
    latent_spaces = {
        "centroid": centroid_arr,
        "root": root_arr,
        "final": final_arr,
        "delta": (centroid_arr - root_arr).astype(np.float32),
    }
    if args.roi_mask is not None:
        roi_summary = run_roi_rsa(
            df,
            latent_spaces,
            manifest,
            roi_mask=args.roi_mask,
            roi_name=args.roi_name,
            rsa_spaces=args.rsa_space,
            rsa_distance=args.rsa_distance,
            rsa_correlation=args.rsa_correlation,
            rsa_max_volumes=args.rsa_max_volumes,
            roi_threshold=args.roi_threshold,
            max_roi_voxels=args.max_roi_voxels,
            result_dir=result_dir,
        )
    else:
        roi_summary = pd.DataFrame()
    roi_summary.to_csv(result_dir / "roi_rsa_summary.csv", index=False)
    write_rsa_samples(df, centroid_arr, root_arr, result_dir, max_steps=args.rsa_max_steps)
    plot_metric_overview(df, figure_dir / "fig_metric_overview.png")

    summary = summarize(df, dmd_summary, manifest, roi_summary)
    (result_dir / "summary.txt").write_text(summary)
    print(summary)
    print(f"[done] wrote outputs to {args.output_dir}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--fmri-root", type=Path, default=DEFAULT_FMRI_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--subjects", default=None, help="Comma-separated subject ids, e.g. 1,2,3")
    parser.add_argument("--sessions", default=None, help="Comma-separated trace session ids, e.g. 1,4")
    parser.add_argument("--games", default=None, help="Comma-separated game ids, e.g. 1,2")
    parser.add_argument(
        "--game-session-offset",
        type=int,
        default=None,
        help=(
            "Optional legacy filter: when --games is set and --sessions is not set, "
            "keep only traces where trace session == game id + offset. Default is "
            "disabled, so all sessions are analyzed."
        ),
    )
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-real-steps-per-file", type=int, default=None)
    parser.add_argument("--max-spectral-nodes", type=int, default=96)
    parser.add_argument("--max-pairwise-nodes", type=int, default=256)
    parser.add_argument("--epsilon-quantile", type=float, default=0.5)
    parser.add_argument(
        "--diffusion-powers",
        type=int,
        nargs="+",
        default=[8, 64, 1024],
        help="Diffusion times used for multiscale distances.",
    )
    parser.add_argument(
        "--node-standardization",
        choices=["local_zscore", "none"],
        default="local_zscore",
    )
    parser.add_argument("--dmd-dim", type=int, default=16)
    parser.add_argument("--dmd-ridge-alpha", type=float, default=1e-3)
    parser.add_argument("--dmd-min-pairs", type=int, default=30)
    parser.add_argument("--rsa-max-steps", type=int, default=1500)
    parser.add_argument("--fmri-image", default="s5_wfiltered_func_data.nii")
    parser.add_argument(
        "--fmri-trim-volumes",
        type=int,
        default=DEFAULT_FMRI_TRIM_VOLUMES,
        help=(
            "Number of volumes to exclude from both the beginning and end of each "
            "fMRI run before ROI-RSA. Default: 60."
        ),
    )
    parser.add_argument(
        "--roi-mask",
        type=Path,
        default=None,
        help="3D ROI mask NIfTI in the same MNI grid as the fMRI images. Enables brain ROI-RSA.",
    )
    parser.add_argument("--roi-name", default="", help="Optional short ROI label for output summaries.")
    parser.add_argument("--roi-threshold", type=float, default=0.0)
    parser.add_argument(
        "--max-roi-voxels",
        type=int,
        default=5000,
        help="Deterministically subsample ROI voxels above this count before loading fMRI time series.",
    )
    parser.add_argument(
        "--rsa-space",
        nargs="+",
        choices=["centroid", "root", "final", "delta"],
        default=["centroid"],
        help="Thinker latent spaces to compare against the ROI voxel RDM.",
    )
    parser.add_argument(
        "--rsa-distance",
        choices=["correlation", "euclidean", "cosine"],
        default="correlation",
        help="Distance metric used to build brain/model RDMs.",
    )
    parser.add_argument(
        "--rsa-correlation",
        choices=["spearman", "pearson"],
        default="spearman",
        help="Correlation used between condensed brain and model RDM vectors.",
    )
    parser.add_argument(
        "--rsa-max-volumes",
        type=int,
        default=480,
        help="Maximum trimmed fMRI volumes per run used in ROI-RSA. Use 0 for all volumes.",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
