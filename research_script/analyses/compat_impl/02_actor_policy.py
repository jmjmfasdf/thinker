#!/usr/bin/env python3
"""Actor-policy coordinate replication of the 02 real-step analyses.

Each observation is one real step ``t`` (public ``status == 0``).  Its model
features are built from every imaginary/reset record with public status 1, 2,
or 3 strictly between the preceding real step ``t-1`` and ``t``.  Node order is
preserved by temporal concatenation.  Windows have variable lengths because
status 1/3 reset duplicates vary, so shorter concatenated vectors are padded
only after their final node; no observed node is discarded.

Outputs
-------
1. vectorized tree-representation t-SNE, colored by human action;
2. raw-flattened im_vectors t-SNE, colored by human action;
3. raw-flattened im_vp_vectors t-SNE, colored by human action;
4. a 10_tree_topology_analysis-style topology t-SNE, colored by human action;
5. the pre-search Policy Slack-Demand/search-geometry map;
6. action-transition matrix and start-aligned A3a-d profiles;
7. an integrated same-action-bout analysis, separately for NOOP-to-active,
   active-to-NOOP, and active-to-different-active transitions;
8. for each transition, active-action-faceted 50x50 endpoint probability-mass
   and signed density-difference figures comparing the pre-search root Policy
   Slack-Demand space with the delta-first normalized final-cur-minus-root
   policy contrast;
9. for each transition, a continuous bout-duration regression figure retaining
   the duration distribution plus adjusted diversity, depth, and deep-allocation
   curves; and
10. realized/pre-search policy state-space action and transition views S5-S9.

This is an analysis variant of ``02_TSNE.py``.  The original script is not
modified.  Search-geometry and behavioral definitions are unchanged, while
the two policy coordinates are reconstructed from the source-state
``actor_policy`` stored in the trace.  Because ``actor_policy`` is evaluated
before ``env.step`` but written on the destination trace row, each completed
actor-policy search window is assigned to the preceding real-step/action row:

* the start coordinate is the actor policy that generated the first status-2
  transition after that real step;
* the terminal coordinate is the actor policy that generated the current
  status-0 transition at the end of that search window.

Status-1/3 duplicate reset markers do not vote in the coordinate.  All figures
are written as PNG only to a separate ``02_actor_policy`` output tree.

The non-tSNE figures use every comparable search root from completed,
known-onset same-action bouts, not the stratified t-SNE sample.  The search
that enters a bout is excluded because its previous action differs from the
bout action.  Search-geometry outcomes use only status-2 events:
``cur_reset`` divides them into root-started bouts, mean search depth is the
mean terminal depth across bouts, and allocation fractions/diversity give each
status-2 search event one vote.  Cross-chunk boundary windows are excluded.

Examples
--------
python research_script/02_actor_policy.py --subject 3 --game 2
python research_script/02_actor_policy.py --subject sub003 --game 2 --sessions 1,2
python research_script/00_cache.py --subject 3 --game 2
python research_script/02_actor_policy.py --subject 3 --game 2 --non-tsne-analysis-only
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import resource
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from common.shared_analysis_cache import (
    DEFAULT_SHARED_CACHE_ROOT,
    cache_paths as shared_cache_paths,
    load_actor_metrics,
    load_representations,
    read_manifest as read_shared_manifest,
    validate_manifest as validate_shared_manifest,
)


from common.paths import REPO as ROOT
from common.paths import SCRIPTS as SCRIPT_DIR
DEFAULT_TRACE_ROOT = ROOT / "test"
BASE_TSNE_OUTPUT_ROOT = SCRIPT_DIR / "outputs" / "02" / "legacy" / "02_TSNE"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs" / "02" / "legacy" / "02_actor_policy"
DEFAULT_CROSS2_SUMMARY = (
    SCRIPT_DIR / "outputs" / "01_behavioral_analysis"
    / "1-7_bout_short_long_summary.csv"
)

REAL_STATUS = 0
WINDOW_STATUSES = (1, 2, 3)
EPS = 1e-12
CACHE_VERSION = 2
ANALYSIS_CACHE_VERSION = 6
COORDINATE_VARIANT = "source_state_actor_policy_shifted_one_real_step_v2"

ACTOR_COORDINATE_COLUMNS = (
    "actor_status2_count",
    "actor_slack_start",
    "actor_demand_start",
    "actor_slack_terminal",
    "actor_demand_terminal",
    "actor_demand_search",
    "actor_demand_within",
    "actor_demand_between",
    "actor_demand_auc",
    "actor_demand_slope",
)

TREE_REP_KEYS = (
    "root_action", "root_r", "root_d", "root_v", "root_policy",
    "root_qs_mean", "root_qs_max", "root_ns", "root_trail_r",
    "rollout_return", "max_rollout_return", "root_raw_action",
    "cur_action", "cur_r", "cur_d", "cur_v", "cur_policy",
    "cur_qs_mean", "cur_qs_max", "cur_ns", "cur_raw_action",
    "cur_reset", "k", "deprec", "action_seq",
)

ACTION_LABELS = {
    0: "NOOP",
    1: "FIRE",
    2: "RIGHT",
    3: "LEFT",
    4: "RIGHTFIRE",
    5: "LEFTFIRE",
}
ACTION_COLORS = {
    0: "#4E79A7",
    1: "#E15759",
    2: "#59A14F",
    3: "#F28E2B",
    4: "#B07AA1",
    5: "#76B7B2",
}

# Plot text is configurable so sibling analyses can reuse the exact same
# samples, transition definitions, regressions, inference, and layouts while
# changing only the scientific coordinate system.  The default preserves the
# actor-policy Slack--Demand figures generated by this script.
GEOMETRY_DISPLAY = {
    "x_term": "Slack",
    "y_term": "Demand",
    "x_short": "S",
    "y_short": "D",
    "start_x_label": "Normalized Start Actor Policy Slack",
    "start_y_label": "Start Actor Policy Demand",
    "start_title": "Search-Start Actor Policy State",
    "main1_title": "Search-Start Actor Policy State and Search Geometry",
    "main3_title": "Search-Start Actor Policy State Predicts Search Geometry",
    "trajectory_x_label": "Actor Policy Slack",
    "trajectory_y_label": "Actor Policy Demand",
    "trajectory_coordinate_label": "start actor coordinate",
    "trajectory_title_prefix": "Start-aligned Actor Policy trajectories",
    "model_name": "Actor Policy",
}


def geometry_display(key: str) -> str:
    """Return the active coordinate label for reusable non-tSNE plots."""
    return str(GEOMETRY_DISPLAY[key])

ACTION_TRANSITION_CLASS_LABELS = {
    "noop_to_active": "NOOP → active",
    "active_to_noop": "Active → NOOP",
    "active_to_active": "Active → different active",
}
ACTION_TRANSITION_CLASS_COLORS = {
    "noop_to_active": "#4E79A7",
    "active_to_noop": "#F28E2B",
    "active_to_active": "#7B61A8",
}

T0 = time.time()


def log(message: str) -> None:
    elapsed = time.time() - T0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1_000_000.0
    print(f"[{elapsed:8.1f}s rss={rss:6.2f}GB] {message}", flush=True)


@dataclass(frozen=True)
class TraceMeta:
    subject: int
    session: int
    block: int
    game: int
    chunk: int
    path: Path

    @property
    def block_id(self) -> Tuple[int, int, int, int]:
        return self.subject, self.session, self.block, self.game

    @property
    def run_label(self) -> str:
        return (
            f"sub{self.subject:03d}_ses{self.session:02d}_"
            f"block{self.block:02d}_game{self.game}"
        )


@dataclass
class NodeBatch:
    status: np.ndarray
    tree_reps: np.ndarray
    im_vectors: np.ndarray
    im_vp_vectors: np.ndarray
    paths: List[Tuple[int, ...]]
    cur_qs_mean: np.ndarray
    cur_policy: np.ndarray

    def __len__(self) -> int:
        return int(len(self.status))


@dataclass
class SampleRecord:
    metadata: Dict[str, object]
    tree_reps: np.ndarray
    im_vectors: np.ndarray
    im_vp_vectors: np.ndarray
    tree_topology: np.ndarray


@dataclass
class AnalysisRoot:
    """Information available at one real row before the following search."""

    root_q: np.ndarray
    root_policy: np.ndarray
    human_action: int
    thinker_action: int
    reward: float
    done: float
    noop_run_length: int
    source_chunk: int


@dataclass
class AnalysisBatch:
    """Compact status-2 content plus status QA counts for one search window."""

    root_q: np.ndarray
    root_n: np.ndarray
    root_policy: np.ndarray
    all_root_policy: np.ndarray
    cur_q: np.ndarray
    cur_policy: np.ndarray
    rollout_return: np.ndarray
    cur_reset: np.ndarray
    paths: np.ndarray
    path_valid: np.ndarray
    depth: np.ndarray
    n_status1: int
    n_status2: int
    n_status3: int

    def __len__(self) -> int:
        return int(self.n_status1 + self.n_status2 + self.n_status3)


def parse_subject(value: object) -> int:
    match = re.search(r"\d+", str(value))
    if not match:
        raise argparse.ArgumentTypeError(f"Cannot parse subject id: {value}")
    return int(match.group(0))


def parse_sessions(value: Optional[str]) -> Optional[set[int]]:
    if value is None or not value.strip():
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def parse_trace_meta(path: Path) -> Optional[TraceMeta]:
    match = re.match(
        r"sub(\d+)-ses(\d+)-block(\d+)-game(\d+)_(\d+)\.npy$",
        path.name,
    )
    if match:
        return TraceMeta(
            subject=int(match.group(1)),
            session=int(match.group(2)),
            block=int(match.group(3)),
            game=int(match.group(4)),
            chunk=int(match.group(5)),
            path=path,
        )
    file_match = re.match(r"video_stat_(\d+)\.npy$", path.name)
    parent_match = re.match(
        r"sub(\d+)-ses(\d+)-block(\d+)-game(\d+)$", path.parent.name
    )
    if file_match and parent_match:
        return TraceMeta(
            subject=int(parent_match.group(1)),
            session=int(parent_match.group(2)),
            block=int(parent_match.group(3)),
            game=int(parent_match.group(4)),
            chunk=int(file_match.group(1)),
            path=path,
        )
    return None


def gather_trace_blocks(
    trace_root: Path,
    subject: int,
    game: int,
    sessions: Optional[set[int]],
) -> Dict[Tuple[int, int, int, int], List[TraceMeta]]:
    subject_dir = trace_root / f"sub{subject:03d}"
    flat_files = subject_dir.glob(f"ses-*/*game{game}_*.npy") if subject_dir.exists() else []
    legacy_files = (
        path
        for directory in trace_root.glob(f"sub{subject:03d}-ses*-block*-game{game}")
        for path in directory.glob("video_stat_*.npy")
    )
    blocks: Dict[Tuple[int, int, int, int], List[TraceMeta]] = {}
    for path in sorted(list(flat_files) + list(legacy_files)):
        meta = parse_trace_meta(path)
        if meta is None or meta.subject != subject or meta.game != game:
            continue
        if sessions is not None and meta.session not in sessions:
            continue
        blocks.setdefault(meta.block_id, []).append(meta)
    for metas in blocks.values():
        metas.sort(key=lambda item: item.chunk)
    return dict(sorted(blocks.items()))


def load_npy_dict(path: Path) -> Dict[str, object]:
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
        item = obj.item()
        if isinstance(item, dict):
            return item
    if hasattr(obj, "files"):
        return {key: obj[key] for key in obj.files}
    raise ValueError(f"Cannot parse trace dictionary: {path}")


def decode_action(array: object, index: int) -> int:
    if array is None:
        return -1
    try:
        value = np.asarray(array[index], dtype=float).reshape(-1)
    except (IndexError, TypeError, ValueError):
        return -1
    if value.size == 0 or not np.any(np.isfinite(value)):
        return -1
    if value.size == 1:
        action = int(value[0])
        return action if action >= 0 else -1
    finite = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    if np.max(np.abs(finite)) <= EPS:
        return -1
    return int(np.argmax(finite))


def infer_num_actions(tree: Mapping[str, object]) -> int:
    for key in ("root_action", "root_policy", "root_qs_mean", "cur_action"):
        if key not in tree:
            continue
        arr = np.asarray(tree[key])
        if arr.ndim >= 2 and arr.shape[-1] > 1:
            return int(arr.shape[-1])
    raise ValueError("Could not infer action count from tree_reps")


def decode_action_path(row: object, num_actions: int) -> Tuple[Tuple[int, ...], bool]:
    """Decode the current path below root, matching script 10."""
    flat = np.asarray(row, dtype=float).reshape(-1)
    if num_actions <= 0 or flat.size == 0 or flat.size % num_actions != 0:
        return tuple(), False
    slots = flat.reshape(-1, num_actions)
    occupied = np.sum(np.abs(slots), axis=1) > 0.5
    occupied_idx = np.flatnonzero(occupied)
    if occupied_idx.size == 0 or occupied_idx[0] != 0:
        return tuple(), False
    last = int(occupied_idx[-1])
    if not np.all(occupied[: last + 1]):
        return tuple(), False
    full_path = tuple(int(np.argmax(slots[depth])) for depth in range(last + 1))
    below_root = full_path[1:]
    return below_root, len(below_root) > 0


class TreeLayout:
    """Stable per-node vectorization of all tree_reps fields."""

    def __init__(self) -> None:
        self.keys: List[str] = []
        self.dims: Dict[str, int] = {}

    def initialize_or_validate(self, tree: Mapping[str, object]) -> None:
        available: Dict[str, int] = {}
        for key, value in tree.items():
            if str(key).startswith("__"):
                continue
            arr = np.asarray(value)
            if arr.ndim < 1 or len(arr) == 0:
                continue
            dim = int(np.prod(arr.shape[1:])) if arr.ndim > 1 else 1
            if dim > 0:
                available[str(key)] = dim
        keys = [key for key in TREE_REP_KEYS if key in available]
        keys.extend(sorted(set(available) - set(keys)))
        dims = {key: available[key] for key in keys}
        if not self.keys:
            self.keys = keys
            self.dims = dims
            log(
                f"Tree layout: {len(keys)} fields, "
                f"{sum(dims.values()):,} values/node"
            )
            return
        if keys != self.keys or dims != self.dims:
            raise ValueError(
                "tree_reps layout changed across trace chunks: "
                f"expected={self.dims}, observed={dims}"
            )

    def vectorize(self, tree: Mapping[str, object], indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices, dtype=int)
        total_dim = int(sum(self.dims.values()))
        out = np.zeros((len(indices), total_dim), dtype=np.float32)
        max_index = int(np.max(indices)) if len(indices) else -1
        cursor = 0
        for key in self.keys:
            dim = self.dims[key]
            value = tree.get(key)
            if value is None:
                raise ValueError(f"Missing or short tree field: {key}")
            arr = np.asarray(value)
            if arr.ndim < 1 or len(arr) <= max_index:
                raise ValueError(f"Missing or short tree field: {key}")
            # Slice first: converting the complete trace field to float32 here
            # would copy hundreds of thousands of unselected rows per window.
            values = np.asarray(arr[indices], dtype=np.float32).reshape(len(indices), -1)
            np.nan_to_num(values, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            if values.shape[1] != dim:
                raise ValueError(
                    f"tree_reps[{key}] width changed: {values.shape[1]} vs {dim}"
                )
            out[:, cursor : cursor + dim] = values
            cursor += dim
        return out


def flatten_image_nodes(value: object, indices: np.ndarray) -> np.ndarray:
    """Flatten every value of each selected image-representation node."""
    arr = np.asarray(value)
    indices = np.asarray(indices, dtype=int)
    if len(indices) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    if arr.ndim < 2 or len(arr) <= int(indices.max()):
        raise ValueError(f"Invalid image-representation shape: {arr.shape}")
    # Preserve the complete (for example 1 x 128 x 6 x 6) representation.
    # Only the leading node/time axis is kept separate for temporal concat.
    selected = np.asarray(arr[indices], dtype=np.float32)
    flattened = selected.reshape(len(indices), -1)
    return np.nan_to_num(
        flattened, nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32, copy=False)


def action_matrix(tree: Mapping[str, object], key: str, indices: np.ndarray, n_actions: int) -> np.ndarray:
    value = tree.get(key)
    if value is None:
        raise ValueError(f"tree_reps['{key}'] must have final width {n_actions}")
    arr = np.asarray(value)
    if arr.ndim < 2 or arr.shape[-1] != n_actions:
        raise ValueError(f"tree_reps['{key}'] must have final width {n_actions}")
    indices = np.asarray(indices, dtype=int)
    max_index = int(np.max(indices)) if len(indices) else -1
    if len(arr) <= max_index:
        raise ValueError(f"tree_reps['{key}'] is shorter than status")
    selected = np.asarray(arr[indices], dtype=np.float32).reshape(len(indices), n_actions)
    return np.nan_to_num(
        selected, copy=False, nan=0.0, posinf=0.0, neginf=0.0
    )


def build_node_batch(
    data: Mapping[str, object],
    tree: Mapping[str, object],
    status: np.ndarray,
    indices: np.ndarray,
    layout: TreeLayout,
    n_actions: int,
) -> Optional[NodeBatch]:
    indices = np.asarray(indices, dtype=int)
    if len(indices) == 0:
        return None
    indices = indices[np.isin(status[indices], WINDOW_STATUSES)]
    if len(indices) == 0:
        return None
    if "im_vectors" not in data or "im_vp_vectors" not in data:
        raise ValueError("Trace must contain im_vectors and im_vp_vectors")
    action_seq = tree.get("action_seq")
    if action_seq is None:
        raise ValueError("tree_reps['action_seq'] is required")
    paths: List[Tuple[int, ...]] = []
    for index in indices:
        path, valid = decode_action_path(action_seq[int(index)], n_actions)
        paths.append(path if valid else tuple())
    return NodeBatch(
        status=np.asarray(status[indices], dtype=np.int8),
        tree_reps=layout.vectorize(tree, indices),
        im_vectors=flatten_image_nodes(data["im_vectors"], indices),
        im_vp_vectors=flatten_image_nodes(data["im_vp_vectors"], indices),
        paths=paths,
        cur_qs_mean=action_matrix(tree, "cur_qs_mean", indices, n_actions),
        cur_policy=action_matrix(tree, "cur_policy", indices, n_actions),
    )


def concatenate_node_batches(
    first: Optional[NodeBatch], second: Optional[NodeBatch]
) -> Optional[NodeBatch]:
    if first is None:
        return second
    if second is None:
        return first
    for name in ("tree_reps", "im_vectors", "im_vp_vectors", "cur_qs_mean", "cur_policy"):
        left = getattr(first, name)
        right = getattr(second, name)
        if left.shape[1:] != right.shape[1:]:
            raise ValueError(f"Node feature width changed for {name}: {left.shape} vs {right.shape}")
    return NodeBatch(
        status=np.concatenate([first.status, second.status]),
        tree_reps=np.vstack([first.tree_reps, second.tree_reps]),
        im_vectors=np.vstack([first.im_vectors, second.im_vectors]),
        im_vp_vectors=np.vstack([first.im_vp_vectors, second.im_vp_vectors]),
        paths=[*first.paths, *second.paths],
        cur_qs_mean=np.vstack([first.cur_qs_mean, second.cur_qs_mean]),
        cur_policy=np.vstack([first.cur_policy, second.cur_policy]),
    )


def normalized_entropy_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    total = float(np.sum(counts))
    if total <= 0 or counts.size <= 1:
        return 0.0
    probabilities = counts[counts > 0] / total
    return float(-np.sum(probabilities * np.log(probabilities)) / np.log(counts.size))


def topology_vector(batch: NodeBatch, n_actions: int) -> np.ndarray:
    """Fixed topology summary based on script 10, without human-label leakage."""
    statuses = np.asarray(batch.status, dtype=int)
    status_counts = np.asarray(
        [np.sum(statuses == code) for code in WINDOW_STATUSES], dtype=np.float32
    )
    valid_positions = [index for index, path in enumerate(batch.paths) if path]
    paths = [batch.paths[index] for index in valid_positions]
    if not paths:
        return np.concatenate(
            [status_counts, np.asarray([len(statuses), 0, len(statuses)], dtype=np.float32),
             np.zeros(10 + n_actions * 16, dtype=np.float32)]
        )

    valid_status = statuses[np.asarray(valid_positions, dtype=int)]
    branches = np.asarray([path[0] for path in paths], dtype=int)
    depths = np.asarray([len(path) for path in paths], dtype=float)
    branch_counts = np.bincount(branches, minlength=n_actions).astype(float)
    total = float(len(paths))
    unique_paths = len(set(paths))
    global_features = np.asarray(
        [
            float(len(statuses)),
            total,
            float(len(statuses) - total),
            float(unique_paths),
            float(unique_paths / total),
            float(1.0 - unique_paths / total),
            normalized_entropy_from_counts(branch_counts),
            float(np.sum(np.square(branch_counts / total))),
            float(np.mean(depths)),
            float(np.std(depths)),
            float(np.max(depths)),
            float(len({path[:-1] for path in paths})),
            float(len({path for path in paths if len(path) == 1})),
        ],
        dtype=np.float32,
    )

    branch_features: List[float] = []
    for action in range(n_actions):
        mask = branches == action
        count = int(np.sum(mask))
        if count == 0:
            branch_features.extend([0.0] * 16)
            continue
        selected_paths = [paths[index] for index in np.flatnonzero(mask)]
        selected_depths = depths[mask]
        selected_status = valid_status[mask]
        unique_count = len(set(selected_paths))
        revisits = count - unique_count
        branch_features.extend(
            [
                float(count),
                float(count / total),
                float(unique_count),
                float(revisits),
                float(revisits / count),
                float(np.mean(selected_depths)),
                float(np.std(selected_depths)),
                float(np.max(selected_depths)),
                float(np.mean(selected_depths == 1)),
                float(np.mean(selected_depths == 2)),
                float(np.mean(selected_depths >= 3)),
                float(np.mean(selected_status == 1)),
                float(np.mean(selected_status == 2)),
                float(np.mean(selected_status == 3)),
                float(len({path[:-1] for path in selected_paths})),
                normalized_entropy_from_counts(
                    np.bincount(selected_depths.astype(int), minlength=int(np.max(depths)) + 1)
                ),
            ]
        )
    return np.concatenate(
        [status_counts, global_features, np.asarray(branch_features, dtype=np.float32)]
    ).astype(np.float32, copy=False)


def softmax_rows(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    shifted = x - np.nanmax(x, axis=1, keepdims=True)
    exp_values = np.exp(np.clip(shifted, -50.0, 50.0))
    return exp_values / np.where(np.sum(exp_values, axis=1, keepdims=True) > 0,
                                 np.sum(exp_values, axis=1, keepdims=True), 1.0)


def policy_to_logits(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    finite = np.all(np.isfinite(x), axis=1)
    if np.any(finite):
        xf = x[finite]
        probability_like = np.mean(
            (np.min(xf, axis=1) >= -1e-6)
            & (np.max(xf, axis=1) <= 1.0 + 1e-6)
            & (np.abs(np.sum(xf, axis=1) - 1.0) < 1e-3)
        ) > 0.9
        if probability_like:
            return np.log(np.clip(x, 1e-8, 1.0))
    return x


def effective_branching_rows(probabilities: np.ndarray) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float)
    entropy = -np.sum(np.where(p > 0, p * np.log(p + EPS), 0.0), axis=1)
    return np.exp(entropy) / max(p.shape[1], 1)


def rescale_effective_branching(values: np.ndarray, n_actions: int) -> np.ndarray:
    floor = 1.0 / max(n_actions, 1)
    if floor >= 1.0:
        return np.zeros_like(np.asarray(values, dtype=float))
    return np.clip((np.asarray(values, dtype=float) - floor) / (1.0 - floor), 0.0, 1.0)


def action_margin_rows(matrix: np.ndarray, action: int) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if not 0 <= action < values.shape[1]:
        raise ValueError(f"Passive action {action} outside 0..{values.shape[1] - 1}")
    others = np.delete(values, action, axis=1)
    return values[:, action] - np.nanmax(others, axis=1)


def sigmoid(values: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(values, dtype=float), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def window_opportunity(
    batch: NodeBatch,
    passive_action: int,
    q_temperature: float,
) -> Tuple[float, float, float, float]:
    q_values = np.asarray(batch.cur_qs_mean, dtype=float)
    policy_values = policy_to_logits(batch.cur_policy)
    valid = np.all(np.isfinite(q_values), axis=1) & np.all(np.isfinite(policy_values), axis=1)
    q_values = q_values[valid]
    policy_values = policy_values[valid]
    if len(q_values) == 0:
        return np.nan, np.nan, np.nan, np.nan

    q_median = np.median(q_values, axis=1, keepdims=True)
    q_mad = np.median(np.abs(q_values - q_median), axis=1)
    q_sd = np.std(q_values, axis=1)
    q_scale_raw = np.fmax(q_mad, q_sd)
    usable = np.isfinite(q_scale_raw) & (q_scale_raw > EPS)
    fallback = float(np.median(q_scale_raw[usable])) if np.any(usable) else 1.0
    q_scale = np.where(usable, q_scale_raw, fallback)

    q_slack = action_margin_rows(q_values, passive_action) / q_scale
    policy_slack = action_margin_rows(policy_values, passive_action)
    q_centered = (q_values - q_median) / q_scale[:, None]
    q_prob = softmax_rows(q_centered / max(q_temperature, EPS))
    policy_prob = softmax_rows(policy_values)
    value_branching = rescale_effective_branching(
        effective_branching_rows(q_prob), q_values.shape[1]
    )
    policy_branching = rescale_effective_branching(
        effective_branching_rows(policy_prob), policy_values.shape[1]
    )
    value = sigmoid(q_slack) * value_branching
    policy = sigmoid(policy_slack) * policy_branching
    return (
        float(np.mean(value)),
        float(np.mean(policy)),
        float(np.std(value)),
        float(np.std(policy)),
    )


def make_sample_record(
    batch: NodeBatch,
    metadata: Dict[str, object],
    n_actions: int,
    passive_action: int,
    q_temperature: float,
) -> SampleRecord:
    value_opp, policy_opp, value_sd, policy_sd = window_opportunity(
        batch, passive_action=passive_action, q_temperature=q_temperature
    )
    metadata = dict(metadata)
    metadata.update(
        {
            "n_window_nodes": len(batch),
            "n_status1": int(np.sum(batch.status == 1)),
            "n_status2": int(np.sum(batch.status == 2)),
            "n_status3": int(np.sum(batch.status == 3)),
            "opportunity_value": value_opp,
            "opportunity_policy": policy_opp,
            "opportunity_value_node_sd": value_sd,
            "opportunity_policy_node_sd": policy_sd,
        }
    )
    return SampleRecord(
        metadata=metadata,
        tree_reps=batch.tree_reps.reshape(-1).astype(np.float32, copy=False),
        im_vectors=batch.im_vectors.reshape(-1).astype(np.float32, copy=False),
        im_vp_vectors=batch.im_vp_vectors.reshape(-1).astype(np.float32, copy=False),
        tree_topology=topology_vector(batch, n_actions),
    )


class StratifiedReservoir:
    def __init__(self, max_points: int, n_actions: int, random_state: int) -> None:
        if max_points < n_actions:
            raise ValueError("--max-points must be at least the number of actions")
        base, remainder = divmod(max_points, n_actions)
        self.capacities = {
            action: base + int(action < remainder) for action in range(n_actions)
        }
        self.records: Dict[int, List[SampleRecord]] = {
            action: [] for action in range(n_actions)
        }
        self.seen = {action: 0 for action in range(n_actions)}
        self.rng = np.random.default_rng(random_state)

    def position(self, action: int) -> Optional[int]:
        if action not in self.records:
            return None
        self.seen[action] += 1
        current = self.records[action]
        capacity = self.capacities[action]
        if len(current) < capacity:
            return len(current)
        candidate = int(self.rng.integers(0, self.seen[action]))
        return candidate if candidate < capacity else None

    def store(self, action: int, position: int, record: SampleRecord) -> None:
        current = self.records[action]
        if position == len(current):
            current.append(record)
        else:
            current[position] = record

    def combined(self) -> List[SampleRecord]:
        output = [record for action in sorted(self.records) for record in self.records[action]]
        output.sort(
            key=lambda record: (
                int(record.metadata["session"]),
                int(record.metadata["block"]),
                int(record.metadata["real_step"]),
            )
        )
        return output


def process_traces(
    blocks: Mapping[Tuple[int, int, int, int], Sequence[TraceMeta]],
    max_files: Optional[int],
    max_points: int,
    passive_action: int,
    q_temperature: float,
    random_state: int,
) -> Tuple[List[SampleRecord], Dict[str, object]]:
    layout = TreeLayout()
    reservoir: Optional[StratifiedReservoir] = None
    n_actions: Optional[int] = None
    total_real_windows = 0
    total_nonempty_windows = 0
    invalid_human_actions = 0
    loaded_files = 0
    observed_window_counts: List[int] = []

    for block_number, metas_all in enumerate(blocks.values(), start=1):
        metas = list(metas_all[:max_files] if max_files is not None else metas_all)
        have_previous_real = False
        carry: Optional[NodeBatch] = None
        real_step = 0
        log(f"Block {block_number}/{len(blocks)}: {len(metas)} chunks")

        for file_number, meta in enumerate(metas, start=1):
            log(
                f"Load {meta.path.name} ({meta.path.stat().st_size / 1024**3:.1f} GiB; "
                f"chunk {file_number}/{len(metas)})"
            )
            data = load_npy_dict(meta.path)
            loaded_files += 1
            status = np.asarray(data.get("status", []), dtype=int).reshape(-1)
            tree = data.get("tree_reps")
            if status.size == 0 or not isinstance(tree, Mapping):
                raise ValueError(f"Missing status/tree_reps in {meta.path}")
            if n_actions is None:
                n_actions = infer_num_actions(tree)
                if not 0 <= passive_action < n_actions:
                    raise ValueError(
                        f"--passive-action {passive_action} outside 0..{n_actions - 1}"
                    )
                reservoir = StratifiedReservoir(max_points, n_actions, random_state)
                log(f"Detected {n_actions} actions; stratified sample cap={max_points:,}")
            elif infer_num_actions(tree) != n_actions:
                raise ValueError(f"Action count changed in {meta.path}")
            assert reservoir is not None
            layout.initialize_or_validate(tree)

            real_indices = np.flatnonzero(status == REAL_STATUS)
            previous_local_real: Optional[int] = None
            for current_real in real_indices:
                current_real = int(current_real)
                real_step += 1
                if previous_local_real is None:
                    between = np.arange(0, current_real, dtype=int)
                else:
                    between = np.arange(previous_local_real + 1, current_real, dtype=int)
                node_indices = between[np.isin(status[between], WINDOW_STATUSES)]
                carry_count = len(carry) if previous_local_real is None and carry is not None else 0
                window_node_count = carry_count + len(node_indices)

                if have_previous_real:
                    total_real_windows += 1
                    if window_node_count > 0:
                        total_nonempty_windows += 1
                        observed_window_counts.append(window_node_count)
                        human_action = decode_action(data.get("human_action"), current_real)
                        if 0 <= human_action < n_actions:
                            position = reservoir.position(human_action)
                            if position is not None:
                                # Reservoir selection precedes expensive image/tree
                                # extraction. Only retained real steps are vectorized.
                                current_nodes = build_node_batch(
                                    data, tree, status, node_indices, layout, n_actions
                                )
                                batch = concatenate_node_batches(
                                    carry if previous_local_real is None else None,
                                    current_nodes,
                                )
                                if batch is None or len(batch) != window_node_count:
                                    raise RuntimeError(
                                        "Internal window construction mismatch: "
                                        f"expected {window_node_count}, got "
                                        f"{0 if batch is None else len(batch)}"
                                    )
                                metadata = {
                                    "subject": meta.subject,
                                    "session": meta.session,
                                    "block": meta.block,
                                    "game": meta.game,
                                    "run_label": meta.run_label,
                                    "chunk": meta.chunk,
                                    "real_step": real_step,
                                    "real_step_id": f"{meta.run_label}_real{real_step:06d}",
                                    "human_action": human_action,
                                }
                                record = make_sample_record(
                                    batch,
                                    metadata,
                                    n_actions=n_actions,
                                    passive_action=passive_action,
                                    q_temperature=q_temperature,
                                )
                                reservoir.store(human_action, position, record)
                        else:
                            invalid_human_actions += 1

                have_previous_real = True
                carry = None
                previous_local_real = current_real

            if have_previous_real:
                tail_start = (
                    int(previous_local_real) + 1 if previous_local_real is not None else 0
                )
                tail_indices = np.arange(tail_start, len(status), dtype=int)
                tail = build_node_batch(data, tree, status, tail_indices, layout, n_actions)
                carry = concatenate_node_batches(carry, tail)

            del data, tree, status
            gc.collect()
            log(
                f"Processed {meta.path.name}; eligible windows={total_nonempty_windows:,}; "
                f"retained={sum(len(rows) for rows in reservoir.records.values()):,}"
            )

    if reservoir is None or n_actions is None:
        raise RuntimeError("No usable trace data")
    records = reservoir.combined()
    if len(records) < 4:
        raise RuntimeError(f"Only {len(records)} sampled real steps; cannot run t-SNE")
    qa = {
        "cache_version": CACHE_VERSION,
        "loaded_files": loaded_files,
        "n_actions": n_actions,
        "total_real_windows": total_real_windows,
        "total_nonempty_windows": total_nonempty_windows,
        "invalid_human_actions": invalid_human_actions,
        "sampled_points": len(records),
        "seen_by_action": reservoir.seen,
        "retained_by_action": {
            action: len(rows) for action, rows in reservoir.records.items()
        },
        "window_nodes_min": int(min(observed_window_counts)) if observed_window_counts else 0,
        "window_nodes_median": float(np.median(observed_window_counts)) if observed_window_counts else 0.0,
        "window_nodes_max": int(max(observed_window_counts)) if observed_window_counts else 0,
        "tree_values_per_node": int(sum(layout.dims.values())),
        "tree_keys": layout.keys,
    }
    return records, qa


def _scalar_at(value: object, index: int) -> float:
    if value is None:
        return np.nan
    try:
        row = np.asarray(value[index], dtype=float).reshape(-1)
    except (IndexError, TypeError, ValueError):
        return np.nan
    finite = row[np.isfinite(row)]
    return float(finite[0]) if finite.size else np.nan


def _empty_analysis_batch(n_actions: int, path_width: int = 0) -> AnalysisBatch:
    return AnalysisBatch(
        root_q=np.zeros((0, n_actions), dtype=np.float32),
        root_n=np.zeros((0, n_actions), dtype=np.float32),
        root_policy=np.zeros((0, n_actions), dtype=np.float32),
        all_root_policy=np.zeros((0, n_actions), dtype=np.float32),
        cur_q=np.zeros((0, n_actions), dtype=np.float32),
        cur_policy=np.zeros((0, n_actions), dtype=np.float32),
        rollout_return=np.zeros(0, dtype=np.float32),
        cur_reset=np.zeros(0, dtype=np.int8),
        paths=np.zeros((0, path_width), dtype=np.int16),
        path_valid=np.zeros(0, dtype=bool),
        depth=np.zeros(0, dtype=np.int16),
        n_status1=0,
        n_status2=0,
        n_status3=0,
    )


def build_analysis_batch(
    tree: Mapping[str, object],
    status: np.ndarray,
    indices: np.ndarray,
    n_actions: int,
) -> AnalysisBatch:
    """Extract only status-2 statistics; never materialize im/im_vp vectors."""
    indices = np.asarray(indices, dtype=int)
    if indices.size:
        indices = indices[np.isin(status[indices], WINDOW_STATUSES)]
    counts = {
        code: int(np.sum(status[indices] == code)) if indices.size else 0
        for code in WINDOW_STATUSES
    }
    selected = indices[status[indices] == 2] if indices.size else np.zeros(0, dtype=int)
    action_seq = np.asarray(tree.get("action_seq", []))
    path_width = (
        int(action_seq.shape[-1] // n_actions - 1)
        if action_seq.ndim >= 2 and action_seq.shape[-1] % n_actions == 0
        else 0
    )
    if selected.size == 0:
        batch = _empty_analysis_batch(n_actions, path_width)
        if indices.size:
            batch.all_root_policy = action_matrix(
                tree, "root_policy", indices, n_actions
            )
        batch.n_status1 = counts[1]
        batch.n_status2 = counts[2]
        batch.n_status3 = counts[3]
        return batch
    if action_seq.ndim < 2 or action_seq.shape[-1] % n_actions != 0:
        raise ValueError("tree_reps['action_seq'] has an invalid action-slot layout")

    slots = np.asarray(action_seq[selected], dtype=np.float32).reshape(
        len(selected), -1, n_actions
    )
    occupied = np.sum(np.abs(slots), axis=2) > 0.5
    consecutive = np.cumprod(occupied.astype(np.int8), axis=1).astype(bool)
    occupied_count = np.sum(occupied, axis=1)
    path_valid = (
        occupied[:, 0]
        & (occupied_count >= 2)
        & (np.sum(consecutive, axis=1) == occupied_count)
    )
    depth = np.where(path_valid, occupied_count - 1, 0).astype(np.int16)
    decoded = np.argmax(slots[:, 1:, :], axis=2).astype(np.int16)
    positions = np.arange(decoded.shape[1])[None, :]
    decoded[positions >= depth[:, None]] = -1
    decoded[~path_valid] = -1

    rollout = np.asarray(tree.get("rollout_return", []))
    if rollout.ndim < 1 or len(rollout) <= int(selected.max()):
        rollout_values = np.full(len(selected), np.nan, dtype=np.float32)
    else:
        rollout_values = np.asarray(rollout[selected], dtype=np.float32).reshape(
            len(selected), -1
        )[:, 0]

    reset_source = tree.get("cur_reset")
    if reset_source is None:
        reset_values = np.full(len(selected), -1, dtype=np.int8)
    else:
        reset_array = np.asarray(reset_source)
        if reset_array.ndim < 1 or len(reset_array) <= int(selected.max()):
            reset_values = np.full(len(selected), -1, dtype=np.int8)
        else:
            reset_float = np.asarray(
                reset_array[selected], dtype=float
            ).reshape(len(selected), -1)[:, 0]
            reset_values = np.where(
                np.isfinite(reset_float), (reset_float > 0.5).astype(np.int8), -1
            ).astype(np.int8)

    return AnalysisBatch(
        root_q=action_matrix(tree, "root_qs_mean", selected, n_actions),
        root_n=action_matrix(tree, "root_ns", selected, n_actions),
        root_policy=action_matrix(tree, "root_policy", selected, n_actions),
        all_root_policy=action_matrix(tree, "root_policy", indices, n_actions),
        cur_q=action_matrix(tree, "cur_qs_mean", selected, n_actions),
        cur_policy=action_matrix(tree, "cur_policy", selected, n_actions),
        rollout_return=np.nan_to_num(
            rollout_values, nan=np.nan, posinf=np.nan, neginf=np.nan
        ),
        cur_reset=reset_values,
        paths=decoded,
        path_valid=path_valid,
        depth=depth,
        n_status1=counts[1],
        n_status2=counts[2],
        n_status3=counts[3],
    )


def concatenate_analysis_batches(
    first: Optional[AnalysisBatch], second: Optional[AnalysisBatch]
) -> Optional[AnalysisBatch]:
    if first is None:
        return second
    if second is None:
        return first
    path_width = max(first.paths.shape[1], second.paths.shape[1])

    def pad_paths(batch: AnalysisBatch) -> np.ndarray:
        if batch.paths.shape[1] == path_width:
            return batch.paths
        out = np.full((batch.n_status2, path_width), -1, dtype=np.int16)
        out[:, : batch.paths.shape[1]] = batch.paths
        return out

    return AnalysisBatch(
        root_q=np.vstack([first.root_q, second.root_q]),
        root_n=np.vstack([first.root_n, second.root_n]),
        root_policy=np.vstack([first.root_policy, second.root_policy]),
        all_root_policy=np.vstack(
            [first.all_root_policy, second.all_root_policy]
        ),
        cur_q=np.vstack([first.cur_q, second.cur_q]),
        cur_policy=np.vstack([first.cur_policy, second.cur_policy]),
        rollout_return=np.concatenate([first.rollout_return, second.rollout_return]),
        cur_reset=np.concatenate([first.cur_reset, second.cur_reset]),
        paths=np.vstack([pad_paths(first), pad_paths(second)]),
        path_valid=np.concatenate([first.path_valid, second.path_valid]),
        depth=np.concatenate([first.depth, second.depth]),
        n_status1=first.n_status1 + second.n_status1,
        n_status2=first.n_status2 + second.n_status2,
        n_status3=first.n_status3 + second.n_status3,
    )


def search_bout_geometry(batch: AnalysisBatch) -> Dict[str, object]:
    """Summarize reset-delimited status-2 search bouts.

    ``cur_reset == 1`` marks the status-2 event after which the next search
    event restarts from depth 1.  The final unterminated segment in a real-step
    window is also a bout.  If ``cur_reset`` is unavailable, depth returning to
    1 is used as an explicit fallback and reported in ``reset_source_exact``.
    """
    n_events = int(batch.n_status2)
    empty = {
        "reset_source_exact": 0,
        "reset_count": 0,
        "bout_count": 0,
        "valid_bout_count": 0,
        "mean_bout_terminal_depth": np.nan,
        "max_bout_terminal_depth": np.nan,
        "deep_bout_fraction": np.nan,
        "single_step_bout_fraction": np.nan,
        "bout_root_branches": np.zeros(0, dtype=np.int16),
        "reset_transition_count": 0,
        "reset_transition_violation_count": 0,
        "continuation_transition_count": 0,
        "continuation_transition_violation_count": 0,
    }
    if n_events == 0:
        return empty

    reset_known = (
        len(batch.cur_reset) == n_events
        and np.all(np.isin(batch.cur_reset, (0, 1)))
    )
    if reset_known:
        reset_flags = batch.cur_reset.astype(np.int8, copy=False)
        ends = np.flatnonzero(reset_flags == 1).tolist()
        if not ends or ends[-1] != n_events - 1:
            ends.append(n_events - 1)
    else:
        reset_flags = np.zeros(n_events, dtype=np.int8)
        valid = batch.path_valid
        for index in range(n_events - 1):
            if (
                valid[index]
                and valid[index + 1]
                and batch.depth[index + 1] <= batch.depth[index]
            ):
                reset_flags[index] = 1
        ends = np.flatnonzero(reset_flags == 1).tolist()
        if not ends or ends[-1] != n_events - 1:
            ends.append(n_events - 1)

    starts = [0] + [int(end) + 1 for end in ends[:-1]]
    terminal_depths: List[int] = []
    bout_branches: List[int] = []
    for start, end in zip(starts, ends):
        positions = np.flatnonzero(batch.path_valid[start : end + 1]) + start
        if positions.size == 0:
            continue
        first = int(positions[0])
        terminal = int(positions[-1])
        terminal_depths.append(int(batch.depth[terminal]))
        bout_branches.append(int(batch.paths[first, 0]))

    terminal_array = np.asarray(terminal_depths, dtype=float)
    branch_array = np.asarray(bout_branches, dtype=np.int16)
    pair_valid = batch.path_valid[:-1] & batch.path_valid[1:]
    reset_pairs = pair_valid & (reset_flags[:-1] == 1)
    continuation_pairs = pair_valid & (reset_flags[:-1] == 0)
    return {
        "reset_source_exact": int(reset_known),
        "reset_count": int(np.sum(reset_flags == 1)),
        "bout_count": int(len(ends)),
        "valid_bout_count": int(len(terminal_array)),
        "mean_bout_terminal_depth": (
            float(np.mean(terminal_array)) if terminal_array.size else np.nan
        ),
        "max_bout_terminal_depth": (
            float(np.max(terminal_array)) if terminal_array.size else np.nan
        ),
        "deep_bout_fraction": (
            float(np.mean(terminal_array >= 3)) if terminal_array.size else np.nan
        ),
        "single_step_bout_fraction": (
            float(np.mean(terminal_array == 1)) if terminal_array.size else np.nan
        ),
        "bout_root_branches": branch_array,
        "reset_transition_count": int(np.sum(reset_pairs)),
        "reset_transition_violation_count": int(
            np.sum(reset_pairs & (batch.depth[1:] != 1))
        ),
        "continuation_transition_count": int(np.sum(continuation_pairs)),
        "continuation_transition_violation_count": int(
            np.sum(
                continuation_pairs
                & (batch.depth[1:] != batch.depth[:-1] + 1)
            )
        ),
    }


def make_analysis_root(
    data: Mapping[str, object],
    tree: Mapping[str, object],
    index: int,
    n_actions: int,
    previous: Optional[AnalysisRoot],
    source_chunk: int,
) -> AnalysisRoot:
    human_action = decode_action(data.get("human_action"), index)
    thinker_action = decode_action(data.get("thinker_action"), index)
    if thinker_action < 0:
        thinker_action = decode_action(tree.get("root_action"), index)
    noop_run_length = (
        previous.noop_run_length + 1
        if previous is not None and human_action == 0 and previous.human_action == 0
        else int(human_action == 0)
    )
    return AnalysisRoot(
        root_q=action_matrix(tree, "root_qs_mean", np.asarray([index]), n_actions)[0],
        root_policy=action_matrix(tree, "root_policy", np.asarray([index]), n_actions)[0],
        human_action=human_action,
        thinker_action=thinker_action,
        reward=_scalar_at(data.get("cur_rewards"), index),
        done=_scalar_at(tree.get("root_d"), index),
        noop_run_length=noop_run_length,
        source_chunk=source_chunk,
    )


def normalized_entropy_rows(probabilities: np.ndarray) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[1] <= 1:
        return np.zeros(len(p), dtype=float)
    entropy = -np.sum(np.where(p > 0, p * np.log(p + EPS), 0.0), axis=1)
    return entropy / math.log(p.shape[1])


def row_q_scale(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    q = np.asarray(values, dtype=float)
    median = np.nanmedian(q, axis=1, keepdims=True)
    mad = np.nanmedian(np.abs(q - median), axis=1)
    sd = np.nanstd(q, axis=1)
    scale = np.fmax(mad, sd)
    valid = np.all(np.isfinite(q), axis=1) & np.any(np.abs(q) > EPS, axis=1)
    scale = np.where(np.isfinite(scale) & (scale > EPS), scale, 1.0)
    return scale, valid


def q_active_probabilities(
    values: np.ndarray, passive_action: int, temperature: float
) -> Tuple[np.ndarray, np.ndarray]:
    q = np.asarray(values, dtype=float)
    active = np.delete(q, passive_action, axis=1)
    scale, valid = row_q_scale(q)
    centered = active - np.nanmedian(active, axis=1, keepdims=True)
    probabilities = softmax_rows(centered / scale[:, None] / max(temperature, EPS))
    probabilities[~valid] = np.nan
    return probabilities, valid


def policy_active_probabilities(
    values: np.ndarray, passive_action: int, temperature: float
) -> np.ndarray:
    logits = policy_to_logits(np.asarray(values, dtype=float))
    active = np.delete(logits, passive_action, axis=1)
    probabilities = softmax_rows(active / max(temperature, EPS))
    finite = np.all(np.isfinite(active), axis=1)
    probabilities[~finite] = np.nan
    return probabilities


def _nan_geometric_mean(first: float, second: float) -> float:
    if not np.isfinite(first) or not np.isfinite(second):
        return np.nan
    return float(np.sqrt(max(first, 0.0) * max(second, 0.0)))


def _trajectory_slope(values: np.ndarray) -> float:
    y = np.asarray(values, dtype=float)
    valid = np.isfinite(y)
    if np.sum(valid) < 2:
        return np.nan
    x = np.linspace(0.0, 1.0, len(y))
    return float(np.polyfit(x[valid], y[valid], 1)[0])


def analysis_window_metrics(
    initial: AnalysisRoot,
    batch: AnalysisBatch,
    current_human_action: int,
    current_thinker_action: int,
    passive_action: int,
    q_temperature: float,
    policy_temperature: float,
) -> Dict[str, object]:
    n_actions = int(initial.root_q.size)
    active_actions = [action for action in range(n_actions) if action != passive_action]
    q0 = np.asarray(initial.root_q, dtype=float)[None, :]
    p0_saved = np.asarray(initial.root_policy, dtype=float)[None, :]
    p0_logits = policy_to_logits(p0_saved)
    q_scale, q0_valid_arr = row_q_scale(q0)
    q0_valid = bool(q0_valid_arr[0])

    q_slack = np.nan
    if q0_valid:
        q_slack = float(
            (q0[0, passive_action] - np.nanmax(q0[0, active_actions])) / q_scale[0]
        )
    policy_slack = float(
        (p0_logits[0, passive_action] - np.nanmax(p0_logits[0, active_actions]))
        / max(policy_temperature, EPS)
    )
    q_gate = float(np.exp(-max(0.0, -q_slack))) if np.isfinite(q_slack) else np.nan
    policy_gate = float(np.exp(-max(0.0, -policy_slack)))
    slack_gate = min(q_gate, policy_gate) if np.isfinite(q_gate) else np.nan

    q0_prob, _ = q_active_probabilities(q0, passive_action, q_temperature)
    p0_prob = policy_active_probabilities(p0_saved, passive_action, policy_temperature)
    d_q_start = float(normalized_entropy_rows(q0_prob)[0]) if q0_valid else np.nan
    d_policy_start = float(normalized_entropy_rows(p0_prob)[0])
    d_pre = _nan_geometric_mean(d_q_start, d_policy_start)

    q_trace_prob, q_trace_valid = q_active_probabilities(
        batch.root_q, passive_action, q_temperature
    ) if batch.n_status2 else (np.zeros((0, len(active_actions))), np.zeros(0, dtype=bool))
    q_trace_entropy = normalized_entropy_rows(q_trace_prob)
    q_trace_entropy[~q_trace_valid] = np.nan
    d_q_auc = float(np.nanmean(q_trace_entropy)) if np.any(np.isfinite(q_trace_entropy)) else np.nan
    d_q_final = float(q_trace_entropy[-1]) if batch.n_status2 else np.nan
    q_resolution = d_q_start - d_q_final if np.isfinite(d_q_start + d_q_final) else np.nan
    q_resolution_robust = np.nan
    if np.any(np.isfinite(q_trace_entropy)):
        width = max(1, int(math.ceil(batch.n_status2 / 4)))
        early = float(np.nanmean(q_trace_entropy[:width]))
        late = float(np.nanmean(q_trace_entropy[-width:]))
        q_resolution_robust = early - late

    root_policy_prob = policy_active_probabilities(
        batch.root_policy, passive_action, policy_temperature
    ) if batch.n_status2 else np.zeros((0, len(active_actions)))
    root_policy_entropy = normalized_entropy_rows(root_policy_prob)
    d_policy_root_auc = (
        float(np.nanmean(root_policy_entropy))
        if np.any(np.isfinite(root_policy_entropy)) else np.nan
    )
    d_policy_root_final = float(root_policy_entropy[-1]) if batch.n_status2 else np.nan
    policy_resolution = (
        d_policy_start - d_policy_root_final
        if np.isfinite(d_policy_start + d_policy_root_final) else np.nan
    )

    def policy_constancy(
        values: np.ndarray,
    ) -> Tuple[float, float]:
        """Return within-window range and deviation from the real-step root."""
        rows = np.asarray(values, dtype=float)
        if rows.ndim != 2 or len(rows) == 0:
            return np.nan, np.nan
        within_range = float(
            np.nanmax(np.nanmax(rows, axis=0) - np.nanmin(rows, axis=0))
        )
        from_real_root = float(
            np.nanmax(np.abs(rows - p0_saved[0][None, :]))
        )
        return within_range, from_real_root

    (
        root_policy_status2_max_range,
        root_policy_status2_max_from_real_root,
    ) = policy_constancy(batch.root_policy)
    (
        root_policy_all_status_max_range,
        root_policy_all_status_max_from_real_root,
    ) = policy_constancy(batch.all_root_policy)

    cur_policy_prob = policy_active_probabilities(
        batch.cur_policy, passive_action, policy_temperature
    ) if batch.n_status2 else np.zeros((0, len(active_actions)))
    cur_policy_entropy = normalized_entropy_rows(cur_policy_prob)
    d_policy_within = (
        float(np.nanmean(cur_policy_entropy))
        if np.any(np.isfinite(cur_policy_entropy)) else np.nan
    )
    mean_policy = np.nanmean(cur_policy_prob, axis=0, keepdims=True) if batch.n_status2 else np.full((1, len(active_actions)), np.nan)
    d_policy_search = (
        float(normalized_entropy_rows(mean_policy)[0])
        if np.all(np.isfinite(mean_policy)) else np.nan
    )
    d_policy_between = (
        d_policy_search - d_policy_within
        if np.isfinite(d_policy_search + d_policy_within) else np.nan
    )
    root_policy_slack_normalized = float(sigmoid(np.asarray([policy_slack]))[0])
    final_cur_policy_slack_logit = np.nan
    final_cur_policy_slack = np.nan
    final_cur_policy_demand = np.nan
    if batch.n_status2:
        final_cur_logits = policy_to_logits(batch.cur_policy[-1:])
        final_cur_policy_slack_logit = float(
            action_margin_rows(final_cur_logits, passive_action)[0]
            / max(policy_temperature, EPS)
        )
        final_cur_policy_slack = float(
            sigmoid(np.asarray([final_cur_policy_slack_logit]))[0]
        )
        final_cur_policy_demand = float(cur_policy_entropy[-1])
    final_minus_root_policy_slack = (
        final_cur_policy_slack - root_policy_slack_normalized
        if np.isfinite(final_cur_policy_slack) else np.nan
    )
    final_minus_root_policy_demand = (
        final_cur_policy_demand - d_policy_start
        if np.isfinite(final_cur_policy_demand + d_policy_start) else np.nan
    )

    valid_paths = batch.paths[batch.path_valid]
    unique_indices = np.zeros(0, dtype=int)
    if len(valid_paths):
        _, first_indices = np.unique(valid_paths, axis=0, return_index=True)
        valid_positions = np.flatnonzero(batch.path_valid)
        unique_indices = valid_positions[np.sort(first_indices)]
    d_policy_search_unique = np.nan
    if unique_indices.size:
        unique_mean = np.nanmean(cur_policy_prob[unique_indices], axis=0, keepdims=True)
        if np.all(np.isfinite(unique_mean)):
            d_policy_search_unique = float(normalized_entropy_rows(unique_mean)[0])

    d_search = _nan_geometric_mean(d_q_auc, d_policy_search)
    opportunity_pre = slack_gate * d_pre if np.isfinite(slack_gate + d_pre) else np.nan
    opportunity_realized = (
        slack_gate * d_search if np.isfinite(slack_gate + d_search) else np.nan
    )

    valid_path_count = int(np.sum(batch.path_valid))
    unique_path_count = int(len(unique_indices))
    unique_path_fraction = (
        unique_path_count / valid_path_count if valid_path_count else np.nan
    )
    revisit_fraction = (
        1.0 - unique_path_fraction if np.isfinite(unique_path_fraction) else np.nan
    )
    # Reset-aware search geometry.  Each bout begins at the root and ends at a
    # status-2 event with cur_reset=1, or at the final status-2 event in the
    # real-step window.  Mean terminal depth gives each bout one vote.
    bout = search_bout_geometry(batch)
    bout_branches = np.asarray(bout["bout_root_branches"], dtype=int)
    bout_effective_branches = np.nan
    bout_branching_normalized = np.nan
    if bout_branches.size:
        counts = np.bincount(bout_branches, minlength=n_actions).astype(float)
        probabilities = counts[counts > 0] / np.sum(counts)
        bout_effective_branches = float(
            np.exp(-np.sum(probabilities * np.log(probabilities)))
        )
        bout_branching_normalized = float(
            (bout_effective_branches - 1.0) / max(n_actions - 1, 1)
        )

    # Root-branch allocation diversity gives each valid status-2 search event
    # one vote.  A long bout therefore contributes more computation to its
    # root branch, which is intentional for the broad-vs-deep allocation view.
    effective_branches = np.nan
    branching_normalized = np.nan
    searched_branch_count = 0
    if valid_path_count:
        root_branches = valid_paths[:, 0]
        visit_counts = np.bincount(
            root_branches, minlength=n_actions
        ).astype(float)
        visit_probabilities = (
            visit_counts[visit_counts > 0] / np.sum(visit_counts)
        )
        effective_branches = float(
            np.exp(-np.sum(visit_probabilities * np.log(visit_probabilities)))
        )
        branching_normalized = float(
            (effective_branches - 1.0) / max(n_actions - 1, 1)
        )
        searched_branch_count = int(np.sum(visit_counts > 0))

    value_surprise = np.full(batch.n_status2, np.nan, dtype=float)
    if batch.n_status2 and q0_valid:
        q_before = np.vstack([q0, batch.root_q[:-1]])
        branches = np.where(batch.path_valid, batch.paths[:, 0], -1)
        valid_surprise = (
            batch.path_valid
            & (branches >= 0)
            & np.isfinite(batch.rollout_return)
            & np.all(np.isfinite(q_before), axis=1)
        )
        rows = np.flatnonzero(valid_surprise)
        if rows.size:
            value_surprise[rows] = np.abs(
                batch.rollout_return[rows] - q_before[rows, branches[rows]]
            ) / q_scale[0]

    q_final = batch.root_q[-1] if batch.n_status2 else np.full(n_actions, np.nan)
    q_root_update_norm = (
        float(np.linalg.norm(q_final - q0[0]) / (q_scale[0] * np.sqrt(n_actions)))
        if q0_valid and np.all(np.isfinite(q_final)) else np.nan
    )
    q_noop_margin_update = np.nan
    if q0_valid and np.all(np.isfinite(q_final)):
        q_margin_start = q0[0, passive_action] - np.nanmax(
            q0[0, active_actions]
        )
        q_margin_final = q_final[passive_action] - np.nanmax(
            q_final[active_actions]
        )
        # Positive values move the root-Q evidence toward retaining NOOP;
        # negative values move it toward the best active action.  The initial
        # root scale is held fixed so this is a signed change, not a change in
        # normalization.
        q_noop_margin_update = float(
            (q_margin_final - q_margin_start) / q_scale[0]
        )
    q_rank_changed = np.nan
    if q0_valid and np.all(np.isfinite(q_final)):
        q_rank_changed = float(
            active_actions[int(np.argmax(q0[0, active_actions]))]
            != active_actions[int(np.argmax(q_final[active_actions]))]
        )
    cur_q_scale, cur_q_valid = row_q_scale(batch.cur_q) if batch.n_status2 else (np.zeros(0), np.zeros(0, dtype=bool))
    del cur_q_scale

    return {
        "previous_human_action": int(initial.human_action),
        "human_action": int(current_human_action),
        "previous_thinker_action": int(initial.thinker_action),
        "thinker_action": int(current_thinker_action),
        "commit_from_noop": int(
            initial.human_action == passive_action
            and current_human_action >= 0
            and current_human_action != passive_action
        ),
        "previous_reward": float(initial.reward),
        "previous_done": float(initial.done),
        "noop_run_length_prev": int(initial.noop_run_length),
        "n_status1": int(batch.n_status1),
        "n_status2": int(batch.n_status2),
        "n_status3": int(batch.n_status3),
        "q0_valid": int(q0_valid),
        "cur_q_valid_fraction": float(np.mean(cur_q_valid)) if batch.n_status2 else np.nan,
        "slack_q": q_slack,
        "slack_policy": policy_slack,
        "slack_policy_normalized": root_policy_slack_normalized,
        "slack_gate_q": q_gate,
        "slack_gate_policy": policy_gate,
        "slack_gate_consensus": slack_gate,
        "slack_channel_disagreement": abs(q_gate - policy_gate) if np.isfinite(q_gate) else np.nan,
        "d_q_start": d_q_start,
        "d_policy_root_start": d_policy_start,
        "root_policy_status2_max_range": root_policy_status2_max_range,
        "root_policy_status2_max_from_real_root": (
            root_policy_status2_max_from_real_root
        ),
        "root_policy_all_status_max_range": root_policy_all_status_max_range,
        "root_policy_all_status_max_from_real_root": (
            root_policy_all_status_max_from_real_root
        ),
        "final_cur_policy_slack_logit": final_cur_policy_slack_logit,
        "final_cur_policy_slack": final_cur_policy_slack,
        "final_cur_policy_demand": final_cur_policy_demand,
        "final_minus_root_policy_slack": final_minus_root_policy_slack,
        "final_minus_root_policy_demand": final_minus_root_policy_demand,
        "d_pre": d_pre,
        "d_q_auc": d_q_auc,
        "d_policy_root_auc": d_policy_root_auc,
        "d_policy_search": d_policy_search,
        "d_policy_within": d_policy_within,
        "d_policy_between": d_policy_between,
        "d_policy_search_unique": d_policy_search_unique,
        "d_search": d_search,
        "d_channel_disagreement": abs(d_q_auc - d_policy_search) if np.isfinite(d_q_auc + d_policy_search) else np.nan,
        "opportunity_pre_descriptive": opportunity_pre,
        "opportunity_realized_descriptive": opportunity_realized,
        "q_resolution": q_resolution,
        "q_resolution_robust": q_resolution_robust,
        "q_demand_slope": _trajectory_slope(q_trace_entropy),
        "policy_resolution": policy_resolution,
        "policy_demand_slope": _trajectory_slope(root_policy_entropy),
        "q_rank_changed": q_rank_changed,
        "q_root_update_norm": q_root_update_norm,
        "q_noop_margin_update": q_noop_margin_update,
        "value_surprise_mean": float(np.nanmean(value_surprise)) if np.any(np.isfinite(value_surprise)) else np.nan,
        "value_surprise_max": float(np.nanmax(value_surprise)) if np.any(np.isfinite(value_surprise)) else np.nan,
        "valid_path_fraction": valid_path_count / batch.n_status2 if batch.n_status2 else np.nan,
        "unique_path_count": unique_path_count,
        "unique_path_fraction": unique_path_fraction,
        "revisit_event_count": valid_path_count - unique_path_count,
        "revisit_fraction": revisit_fraction,
        "effective_branches": effective_branches,
        "topology_branching_normalized": branching_normalized,
        "visit_effective_branches": effective_branches,
        "visit_topology_branching_normalized": branching_normalized,
        "bout_effective_branches": bout_effective_branches,
        "bout_topology_branching_normalized": bout_branching_normalized,
        "searched_branch_count": searched_branch_count,
        "reset_source_exact": int(bout["reset_source_exact"]),
        "reset_count": int(bout["reset_count"]),
        "bout_count": int(bout["bout_count"]),
        "valid_bout_count": int(bout["valid_bout_count"]),
        "mean_depth": float(bout["mean_bout_terminal_depth"]),
        "mean_bout_terminal_depth": float(bout["mean_bout_terminal_depth"]),
        "max_bout_terminal_depth": float(bout["max_bout_terminal_depth"]),
        "deep_fraction": (
            float(np.mean(batch.depth[batch.path_valid] >= 3))
            if valid_path_count else np.nan
        ),
        "deep_bout_fraction": float(bout["deep_bout_fraction"]),
        "single_step_bout_fraction": float(bout["single_step_bout_fraction"]),
        "mean_node_depth": (
            float(np.mean(batch.depth[batch.path_valid]))
            if valid_path_count else np.nan
        ),
        "deep_event_fraction": (
            float(np.mean(batch.depth[batch.path_valid] >= 3))
            if valid_path_count else np.nan
        ),
        "reset_transition_count": int(bout["reset_transition_count"]),
        "reset_transition_violation_count": int(
            bout["reset_transition_violation_count"]
        ),
        "continuation_transition_count": int(
            bout["continuation_transition_count"]
        ),
        "continuation_transition_violation_count": int(
            bout["continuation_transition_violation_count"]
        ),
    }


def process_analysis_traces(
    blocks: Mapping[Tuple[int, int, int, int], Sequence[TraceMeta]],
    max_files: Optional[int],
    passive_action: int,
    q_temperature: float,
    policy_temperature: float,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    loaded_files = 0
    n_actions_global: Optional[int] = None
    invalid_current_actions = 0
    cross_chunk_windows = 0

    for block_number, metas_all in enumerate(blocks.values(), start=1):
        metas = list(metas_all[:max_files] if max_files is not None else metas_all)
        previous_root: Optional[AnalysisRoot] = None
        carry: Optional[AnalysisBatch] = None
        real_step = 0
        log(f"Analysis block {block_number}/{len(blocks)}: {len(metas)} chunks")
        for file_number, meta in enumerate(metas, start=1):
            log(
                f"Analysis load {meta.path.name} "
                f"({meta.path.stat().st_size / 1024**3:.1f} GiB; "
                f"chunk {file_number}/{len(metas)})"
            )
            data = load_npy_dict(meta.path)
            loaded_files += 1
            status = np.asarray(data.get("status", []), dtype=int).reshape(-1)
            tree = data.get("tree_reps")
            if status.size == 0 or not isinstance(tree, Mapping):
                raise ValueError(f"Missing status/tree_reps in {meta.path}")
            n_actions = infer_num_actions(tree)
            if n_actions_global is None:
                n_actions_global = n_actions
            elif n_actions != n_actions_global:
                raise ValueError(f"Action count changed in {meta.path}")
            if not 0 <= passive_action < n_actions:
                raise ValueError(f"Passive action {passive_action} outside 0..{n_actions - 1}")

            real_indices = np.flatnonzero(status == REAL_STATUS)
            previous_local_real: Optional[int] = None
            for current_real_raw in real_indices:
                current_real = int(current_real_raw)
                real_step += 1
                start = 0 if previous_local_real is None else previous_local_real + 1
                between = np.arange(start, current_real, dtype=int)
                current_batch = build_analysis_batch(
                    tree, status, between, n_actions
                )
                crossed_chunk = (
                    previous_root is not None
                    and previous_local_real is None
                    and previous_root.source_chunk != meta.chunk
                )
                batch = concatenate_analysis_batches(
                    carry if previous_local_real is None else None,
                    current_batch,
                )
                if previous_root is not None and batch is not None:
                    current_human_action = decode_action(data.get("human_action"), current_real)
                    current_thinker_action = decode_action(data.get("thinker_action"), current_real)
                    if current_thinker_action < 0:
                        current_thinker_action = decode_action(tree.get("root_action"), current_real)
                    if current_human_action < 0:
                        invalid_current_actions += 1
                    metrics = analysis_window_metrics(
                        previous_root,
                        batch,
                        current_human_action=current_human_action,
                        current_thinker_action=current_thinker_action,
                        passive_action=passive_action,
                        q_temperature=q_temperature,
                        policy_temperature=policy_temperature,
                    )
                    rows.append(
                        {
                            "subject": meta.subject,
                            "session": meta.session,
                            "block": meta.block,
                            "game": meta.game,
                            "run_label": meta.run_label,
                            "chunk": meta.chunk,
                            "real_step": real_step,
                            "real_step_id": f"{meta.run_label}_real{real_step:06d}",
                            "cross_chunk_boundary": int(crossed_chunk),
                            **metrics,
                        }
                    )
                    cross_chunk_windows += int(crossed_chunk)

                next_root = make_analysis_root(
                    data,
                    tree,
                    current_real,
                    n_actions,
                    previous=previous_root,
                    source_chunk=meta.chunk,
                )
                previous_root = next_root
                carry = None
                previous_local_real = current_real

            tail_start = int(previous_local_real) + 1 if previous_local_real is not None else 0
            tail_indices = np.arange(tail_start, len(status), dtype=int)
            tail = build_analysis_batch(tree, status, tail_indices, n_actions)
            carry = concatenate_analysis_batches(carry, tail)
            del data, tree, status
            gc.collect()
            log(f"Analysis processed {meta.path.name}; real-step rows={len(rows):,}")

    if not rows or n_actions_global is None:
        raise RuntimeError("No real-step analysis rows were extracted")
    metrics = pd.DataFrame(rows).sort_values(["run_label", "real_step"]).reset_index(drop=True)
    qa = {
        "analysis_cache_version": ANALYSIS_CACHE_VERSION,
        "loaded_files": loaded_files,
        "n_actions": n_actions_global,
        "n_real_step_windows": int(len(metrics)),
        "n_runs": int(metrics["run_label"].nunique()),
        "invalid_current_actions": invalid_current_actions,
        "cross_chunk_windows": cross_chunk_windows,
        "status2_min": int(metrics["n_status2"].min()),
        "status2_median": float(metrics["n_status2"].median()),
        "status2_max": int(metrics["n_status2"].max()),
        "q0_invalid_fraction": float(1.0 - metrics["q0_valid"].mean()),
        "cur_q_invalid_fraction": float(1.0 - metrics["cur_q_valid_fraction"].mean()),
        "reset_source_exact_fraction": float(metrics["reset_source_exact"].mean()),
        "reset_transition_violations": int(
            metrics["reset_transition_violation_count"].sum()
        ),
        "continuation_transition_violations": int(
            metrics["continuation_transition_violation_count"].sum()
        ),
        "mean_reset_count": float(metrics["reset_count"].mean()),
        "mean_bout_count": float(metrics["bout_count"].mean()),
        "root_policy_constancy_tolerance": 1e-6,
        "root_policy_status2_constant_fraction": float(
            np.mean(
                metrics["root_policy_status2_max_range"].to_numpy(dtype=float)
                <= 1e-6
            )
        ),
        "root_policy_all_status_constant_fraction": float(
            np.mean(
                metrics["root_policy_all_status_max_range"].to_numpy(dtype=float)
                <= 1e-6
            )
        ),
        "root_policy_status2_matches_real_root_fraction": float(
            np.mean(
                metrics[
                    "root_policy_status2_max_from_real_root"
                ].to_numpy(dtype=float)
                <= 1e-6
            )
        ),
        "root_policy_all_status_matches_real_root_fraction": float(
            np.mean(
                metrics[
                    "root_policy_all_status_max_from_real_root"
                ].to_numpy(dtype=float)
                <= 1e-6
            )
        ),
        "root_policy_status2_max_observed_range": float(
            np.nanmax(metrics["root_policy_status2_max_range"])
        ),
        "root_policy_all_status_max_observed_range": float(
            np.nanmax(metrics["root_policy_all_status_max_range"])
        ),
    }
    return metrics, qa


def process_analysis_traces_parallel(
    blocks: Mapping[Tuple[int, int, int, int], Sequence[TraceMeta]],
    max_files: Optional[int],
    passive_action: int,
    q_temperature: float,
    policy_temperature: float,
    workers: int,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Extract independent runs concurrently while keeping each run ordered."""
    block_items = list(blocks.items())
    effective_workers = min(max(1, int(workers)), len(block_items))
    if effective_workers == 1:
        return process_analysis_traces(
            blocks,
            max_files=max_files,
            passive_action=passive_action,
            q_temperature=q_temperature,
            policy_temperature=policy_temperature,
        )

    frames: List[pd.DataFrame] = []
    qa_parts: List[Dict[str, object]] = []
    log(
        f"Parallel analysis extraction: {len(block_items)} runs, "
        f"{effective_workers} workers"
    )
    with ProcessPoolExecutor(max_workers=effective_workers) as executor:
        futures = {
            executor.submit(
                process_analysis_traces,
                {block_id: metas},
                max_files,
                passive_action,
                q_temperature,
                policy_temperature,
            ): block_id
            for block_id, metas in block_items
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            block_id = futures[future]
            frame, part_qa = future.result()
            frames.append(frame)
            qa_parts.append(part_qa)
            log(
                f"Parallel run complete {completed}/{len(futures)}: "
                f"{block_id}; rows={len(frame):,}"
            )

    metrics = pd.concat(frames, ignore_index=True).sort_values(
        ["run_label", "real_step"]
    ).reset_index(drop=True)
    n_actions = {int(part["n_actions"]) for part in qa_parts}
    if len(n_actions) != 1:
        raise ValueError(f"Action count differs across runs: {sorted(n_actions)}")
    qa = {
        "analysis_cache_version": ANALYSIS_CACHE_VERSION,
        "analysis_workers": effective_workers,
        "loaded_files": int(sum(int(part["loaded_files"]) for part in qa_parts)),
        "n_actions": int(next(iter(n_actions))),
        "n_real_step_windows": int(len(metrics)),
        "n_runs": int(metrics["run_label"].nunique()),
        "invalid_current_actions": int(
            sum(int(part["invalid_current_actions"]) for part in qa_parts)
        ),
        "cross_chunk_windows": int(
            sum(int(part["cross_chunk_windows"]) for part in qa_parts)
        ),
        "status2_min": int(metrics["n_status2"].min()),
        "status2_median": float(metrics["n_status2"].median()),
        "status2_max": int(metrics["n_status2"].max()),
        "q0_invalid_fraction": float(1.0 - metrics["q0_valid"].mean()),
        "cur_q_invalid_fraction": float(
            1.0 - metrics["cur_q_valid_fraction"].mean()
        ),
        "reset_source_exact_fraction": float(metrics["reset_source_exact"].mean()),
        "reset_transition_violations": int(
            metrics["reset_transition_violation_count"].sum()
        ),
        "continuation_transition_violations": int(
            metrics["continuation_transition_violation_count"].sum()
        ),
        "mean_reset_count": float(metrics["reset_count"].mean()),
        "mean_bout_count": float(metrics["bout_count"].mean()),
        "root_policy_constancy_tolerance": 1e-6,
        "root_policy_status2_constant_fraction": float(
            np.mean(
                metrics["root_policy_status2_max_range"].to_numpy(dtype=float)
                <= 1e-6
            )
        ),
        "root_policy_all_status_constant_fraction": float(
            np.mean(
                metrics["root_policy_all_status_max_range"].to_numpy(dtype=float)
                <= 1e-6
            )
        ),
        "root_policy_status2_matches_real_root_fraction": float(
            np.mean(
                metrics[
                    "root_policy_status2_max_from_real_root"
                ].to_numpy(dtype=float)
                <= 1e-6
            )
        ),
        "root_policy_all_status_matches_real_root_fraction": float(
            np.mean(
                metrics[
                    "root_policy_all_status_max_from_real_root"
                ].to_numpy(dtype=float)
                <= 1e-6
            )
        ),
        "root_policy_status2_max_observed_range": float(
            np.nanmax(metrics["root_policy_status2_max_range"])
        ),
        "root_policy_all_status_max_observed_range": float(
            np.nanmax(metrics["root_policy_all_status_max_range"])
        ),
    }
    return metrics, qa


def stack_padded(rows: Sequence[np.ndarray], label: str) -> np.ndarray:
    widths = np.asarray([np.asarray(row).size for row in rows], dtype=int)
    max_width = int(np.max(widths))
    matrix = np.zeros((len(rows), max_width), dtype=np.float32)
    for index, row in enumerate(rows):
        values = np.asarray(row, dtype=np.float32).reshape(-1)
        matrix[index, : len(values)] = values
    log(
        f"Stack {label}: rows={len(rows):,}, width={max_width:,}, "
        f"size={matrix.nbytes / 1024**3:.2f} GiB"
    )
    return matrix


def _actor_policy_matrix(
    data: Mapping[str, object],
    indices: np.ndarray,
    n_actions: int,
) -> np.ndarray:
    value = data.get("actor_policy")
    if value is None:
        raise ValueError("Trace is missing top-level actor_policy")
    arr = np.asarray(value)
    indices = np.asarray(indices, dtype=int)
    if arr.ndim < 2 or arr.shape[-1] != n_actions:
        raise ValueError(
            f"actor_policy must have final width {n_actions}; observed {arr.shape}"
        )
    if len(indices) and len(arr) <= int(np.max(indices)):
        raise ValueError("actor_policy is shorter than status")
    selected = np.asarray(arr[indices], dtype=float).reshape(len(indices), n_actions)
    selected[~np.isfinite(selected)] = np.nan
    return selected


def _actor_coordinate_summary(
    search_policy: np.ndarray,
    terminal_policy: np.ndarray,
    passive_action: int,
    policy_temperature: float,
) -> Dict[str, float]:
    """Summarize source-state actor policies in one real-step search window."""
    search_policy = np.asarray(search_policy, dtype=float)
    terminal_policy = np.asarray(terminal_policy, dtype=float).reshape(1, -1)
    if search_policy.ndim != 2 or len(search_policy) == 0:
        return {
            name: np.nan
            for name in (
                "actor_slack_start",
                "actor_demand_start",
                "actor_slack_terminal",
                "actor_demand_terminal",
                "actor_demand_search",
                "actor_demand_within",
                "actor_demand_between",
                "actor_demand_auc",
                "actor_demand_slope",
            )
        }
    start_policy = search_policy[:1]
    start_logits = policy_to_logits(start_policy)
    terminal_logits = policy_to_logits(terminal_policy)
    start_slack = float(
        action_margin_rows(start_logits, passive_action)[0]
        / max(policy_temperature, EPS)
    )
    terminal_slack = float(
        action_margin_rows(terminal_logits, passive_action)[0]
        / max(policy_temperature, EPS)
    )
    search_prob = policy_active_probabilities(
        search_policy, passive_action, policy_temperature
    )
    start_prob = policy_active_probabilities(
        start_policy, passive_action, policy_temperature
    )
    terminal_prob = policy_active_probabilities(
        terminal_policy, passive_action, policy_temperature
    )
    search_entropy = normalized_entropy_rows(search_prob)
    ensemble = np.nanmean(search_prob, axis=0, keepdims=True)
    ensemble_entropy = (
        float(normalized_entropy_rows(ensemble)[0])
        if np.all(np.isfinite(ensemble))
        else np.nan
    )
    within_entropy = (
        float(np.nanmean(search_entropy))
        if np.any(np.isfinite(search_entropy))
        else np.nan
    )
    return {
        "actor_slack_start": start_slack,
        "actor_demand_start": float(normalized_entropy_rows(start_prob)[0]),
        "actor_slack_terminal": terminal_slack,
        "actor_demand_terminal": float(
            normalized_entropy_rows(terminal_prob)[0]
        ),
        "actor_demand_search": ensemble_entropy,
        "actor_demand_within": within_entropy,
        "actor_demand_between": (
            ensemble_entropy - within_entropy
            if np.isfinite(ensemble_entropy + within_entropy)
            else np.nan
        ),
        "actor_demand_auc": within_entropy,
        "actor_demand_slope": _trajectory_slope(search_entropy),
    }


def process_actor_policy_coordinates(
    blocks: Mapping[Tuple[int, int, int, int], Sequence[TraceMeta]],
    max_files: Optional[int],
    passive_action: int,
    policy_temperature: float,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Extract actor coordinates and attach them to their source real step.

    A row written with logged real-step index ``k`` contains an actor policy
    evaluated on the source side of the transition.  The complete search
    window ending at ``k`` therefore belongs to behavioral real step ``k-1``.
    Re-keying within each run prevents cross-run leakage and naturally leaves
    the final behavioral step without a future completed actor window.
    """
    rows: List[Dict[str, object]] = []
    loaded_files = 0
    n_actions_global: Optional[int] = None
    for block_number, metas_all in enumerate(blocks.values(), start=1):
        metas = list(metas_all[:max_files] if max_files is not None else metas_all)
        have_previous_real = False
        carry = np.zeros((0, 0), dtype=float)
        real_step = 0
        log(
            f"Actor-coordinate block {block_number}/{len(blocks)}: "
            f"{len(metas)} chunks"
        )
        for meta in metas:
            data = load_npy_dict(meta.path)
            loaded_files += 1
            status = np.asarray(data.get("status", []), dtype=int).reshape(-1)
            tree = data.get("tree_reps")
            if status.size == 0 or not isinstance(tree, Mapping):
                raise ValueError(f"Missing status/tree_reps in {meta.path}")
            n_actions = infer_num_actions(tree)
            if n_actions_global is None:
                n_actions_global = n_actions
                carry = np.zeros((0, n_actions), dtype=float)
            elif n_actions != n_actions_global:
                raise ValueError(f"Action count changed in {meta.path}")
            real_indices = np.flatnonzero(status == REAL_STATUS)
            previous_local_real: Optional[int] = None
            for current_real_raw in real_indices:
                current_real = int(current_real_raw)
                real_step += 1
                start = 0 if previous_local_real is None else previous_local_real + 1
                between = np.arange(start, current_real, dtype=int)
                status2 = between[status[between] == 2]
                local_search = _actor_policy_matrix(data, status2, n_actions)
                search_policy = (
                    np.vstack([carry, local_search])
                    if previous_local_real is None and len(carry)
                    else local_search
                )
                if have_previous_real:
                    actor_policy_source_real_step = real_step
                    target_real_step = actor_policy_source_real_step - 1
                    terminal_policy = _actor_policy_matrix(
                        data, np.asarray([current_real]), n_actions
                    )[0]
                    rows.append(
                        {
                            "run_label": meta.run_label,
                            "real_step": target_real_step,
                            "real_step_id": (
                                f"{meta.run_label}_real{target_real_step:06d}"
                            ),
                            "actor_policy_source_real_step": (
                                actor_policy_source_real_step
                            ),
                            "actor_status2_count": int(len(search_policy)),
                            **_actor_coordinate_summary(
                                search_policy,
                                terminal_policy,
                                passive_action=passive_action,
                                policy_temperature=policy_temperature,
                            ),
                        }
                    )
                have_previous_real = True
                carry = np.zeros((0, n_actions), dtype=float)
                previous_local_real = current_real
            tail_start = (
                int(previous_local_real) + 1
                if previous_local_real is not None
                else 0
            )
            tail = np.arange(tail_start, len(status), dtype=int)
            tail_status2 = tail[status[tail] == 2]
            tail_policy = _actor_policy_matrix(data, tail_status2, n_actions)
            carry = (
                np.vstack([carry, tail_policy])
                if previous_local_real is None and len(carry)
                else tail_policy
            )
            del data, tree, status
            gc.collect()
    if not rows or n_actions_global is None:
        raise RuntimeError("No actor-policy coordinate rows were extracted")
    frame = pd.DataFrame(rows).sort_values(
        ["run_label", "real_step"]
    ).reset_index(drop=True)
    finite = np.isfinite(
        frame[
            [
                "actor_slack_start",
                "actor_demand_start",
                "actor_slack_terminal",
                "actor_demand_terminal",
            ]
        ].to_numpy(dtype=float)
    ).all(axis=1)
    qa = {
        "coordinate_variant": COORDINATE_VARIANT,
        "loaded_files": int(loaded_files),
        "n_actions": int(n_actions_global),
        "n_actor_coordinate_rows": int(len(frame)),
        "finite_coordinate_fraction": float(np.mean(finite)),
        "actor_status2_min": int(frame["actor_status2_count"].min()),
        "actor_status2_median": float(frame["actor_status2_count"].median()),
        "actor_status2_max": int(frame["actor_status2_count"].max()),
        "alignment": (
            "source-state actor-policy window ending at logged real step k "
            "is attached to behavioral real step k-1 within the same run"
        ),
    }
    return frame, qa


def process_actor_policy_coordinates_parallel(
    blocks: Mapping[Tuple[int, int, int, int], Sequence[TraceMeta]],
    max_files: Optional[int],
    passive_action: int,
    policy_temperature: float,
    workers: int,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    block_items = list(blocks.items())
    effective_workers = min(max(1, int(workers)), len(block_items))
    if effective_workers == 1:
        return process_actor_policy_coordinates(
            blocks, max_files, passive_action, policy_temperature
        )
    frames: List[pd.DataFrame] = []
    qa_parts: List[Dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=effective_workers) as executor:
        futures = {
            executor.submit(
                process_actor_policy_coordinates,
                {block_id: metas},
                max_files,
                passive_action,
                policy_temperature,
            ): block_id
            for block_id, metas in block_items
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            frame, qa = future.result()
            frames.append(frame)
            qa_parts.append(qa)
            log(
                f"Actor-coordinate run complete {completed}/{len(futures)}: "
                f"{futures[future]}; rows={len(frame):,}"
            )
    result = pd.concat(frames, ignore_index=True).sort_values(
        ["run_label", "real_step"]
    ).reset_index(drop=True)
    qa = {
        "coordinate_variant": COORDINATE_VARIANT,
        "analysis_workers": effective_workers,
        "loaded_files": int(sum(part["loaded_files"] for part in qa_parts)),
        "n_actions": int(qa_parts[0]["n_actions"]),
        "n_actor_coordinate_rows": int(len(result)),
        "finite_coordinate_fraction": float(
            np.mean(
                np.isfinite(
                    result[
                        [
                            "actor_slack_start",
                            "actor_demand_start",
                            "actor_slack_terminal",
                            "actor_demand_terminal",
                        ]
                    ].to_numpy(dtype=float)
                ).all(axis=1)
            )
        ),
        "alignment": qa_parts[0]["alignment"],
    }
    return result, qa


def apply_actor_policy_coordinates(
    base_metrics: pd.DataFrame,
    actor_metrics: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    keys = ["run_label", "real_step", "real_step_id"]
    merged = base_metrics.merge(
        actor_metrics, on=keys, how="left", validate="one_to_one"
    )
    matched = merged["actor_slack_start"].notna()
    if not np.any(matched):
        raise ValueError("Actor-policy coordinates did not align with base 02 rows")

    start_slack = merged["actor_slack_start"].to_numpy(dtype=float)
    start_demand = merged["actor_demand_start"].to_numpy(dtype=float)
    terminal_slack = merged["actor_slack_terminal"].to_numpy(dtype=float)
    terminal_demand = merged["actor_demand_terminal"].to_numpy(dtype=float)
    search_demand = merged["actor_demand_search"].to_numpy(dtype=float)
    within_demand = merged["actor_demand_within"].to_numpy(dtype=float)
    actor_gate = np.exp(-np.maximum(0.0, -start_slack))

    merged["slack_policy"] = start_slack
    merged["slack_policy_normalized"] = sigmoid(start_slack)
    merged["slack_gate_policy"] = actor_gate
    if "slack_gate_q" in merged:
        q_gate = merged["slack_gate_q"].to_numpy(dtype=float)
        merged["slack_gate_consensus"] = np.fmin(q_gate, actor_gate)
        merged["slack_channel_disagreement"] = np.abs(q_gate - actor_gate)
    merged["d_policy_root_start"] = start_demand
    merged["d_policy_root_auc"] = merged["actor_demand_auc"]
    merged["d_policy_search"] = search_demand
    merged["d_policy_within"] = within_demand
    merged["d_policy_between"] = merged["actor_demand_between"]
    merged["d_policy_search_unique"] = search_demand
    merged["final_cur_policy_slack_logit"] = terminal_slack
    merged["final_cur_policy_slack"] = sigmoid(terminal_slack)
    merged["final_cur_policy_demand"] = terminal_demand
    merged["final_minus_root_policy_slack"] = (
        merged["final_cur_policy_slack"]
        - merged["slack_policy_normalized"]
    )
    merged["final_minus_root_policy_demand"] = terminal_demand - start_demand
    merged["policy_resolution"] = start_demand - terminal_demand
    merged["policy_demand_slope"] = merged["actor_demand_slope"]
    if "d_q_start" in merged:
        merged["d_pre"] = np.sqrt(
            np.clip(merged["d_q_start"].to_numpy(dtype=float), 0.0, None)
            * np.clip(start_demand, 0.0, None)
        )
    if "d_q_auc" in merged:
        merged["d_search"] = np.sqrt(
            np.clip(merged["d_q_auc"].to_numpy(dtype=float), 0.0, None)
            * np.clip(search_demand, 0.0, None)
        )
        merged["d_channel_disagreement"] = np.abs(
            merged["d_q_auc"].to_numpy(dtype=float) - search_demand
        )
    if "opportunity_pre_descriptive" in merged:
        merged["opportunity_pre_descriptive"] = actor_gate * start_demand
    if "opportunity_realized_descriptive" in merged:
        merged["opportunity_realized_descriptive"] = actor_gate * search_demand

    qa = {
        "coordinate_variant": COORDINATE_VARIANT,
        "base_rows": int(len(base_metrics)),
        "matched_actor_rows": int(np.sum(matched)),
        "matched_actor_fraction": float(np.mean(matched)),
        "actor_source_is_next_real_step_fraction": float(
            np.mean(
                merged.loc[
                    matched, "actor_policy_source_real_step"
                ].to_numpy(dtype=int)
                == merged.loc[matched, "real_step"].to_numpy(dtype=int) + 1
            )
        ),
    }
    return merged, qa


def reuse_original_tsne_embeddings(
    args: argparse.Namespace,
    metadata: pd.DataFrame,
    figure_dir: Path,
    result_dir: Path,
) -> bool:
    """Replot unchanged representation embeddings without refitting t-SNE."""
    base_result = (
        BASE_TSNE_OUTPUT_ROOT
        / f"sub{args.subject:03d}_game{args.game}"
        / "results"
        / "sampled_real_steps_with_embeddings.csv"
    )
    if not base_result.exists() or args.max_files is not None or args.sessions:
        return False
    combined = pd.read_csv(base_result, low_memory=False)
    if len(combined) != len(metadata):
        return False
    titles = {
        "tree_reps": "Vectorized tree representations",
        "im_vectors": "Imagined-state representations",
        "im_vp_vectors": "Imagined value-policy representations",
        "tree_topology": "Search-tree topology",
    }
    actions = combined["human_action"].to_numpy(dtype=int)
    for name, title in titles.items():
        columns = [f"{name}_tsne_1", f"{name}_tsne_2"]
        if not set(columns).issubset(combined.columns):
            return False
        plot_embedding(
            combined[columns].to_numpy(dtype=float),
            actions,
            title,
            figure_dir / f"fig_tsne_{name}_by_human_action.png",
        )
    combined.to_csv(
        result_dir / "sampled_real_steps_with_embeddings.csv", index=False
    )
    source_diag = base_result.parent / "tsne_diagnostics.csv"
    if source_diag.exists():
        diagnostics = pd.read_csv(source_diag)
        diagnostics["embedding_source"] = str(base_result)
        diagnostics.to_csv(result_dir / "tsne_diagnostics.csv", index=False)
    log(f"Reused unchanged t-SNE embeddings from {base_result}")
    return True


def save_static_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log(f"Saved {output_path}")


def plot_planning_state_space(
    metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot every eligible real step in the primary policy Slack–Demand plane."""
    required = [
        "cross_chunk_boundary",
        "n_status2",
        "slack_policy",
        "d_policy_search",
    ]
    data = metrics.loc[
        metrics["cross_chunk_boundary"].eq(0) & metrics["n_status2"].gt(0),
        required,
    ].replace([np.inf, -np.inf], np.nan).dropna(
        subset=["slack_policy", "d_policy_search"]
    )
    if data.empty:
        raise ValueError("No finite eligible policy Slack–Demand rows")

    raw_slack = data["slack_policy"].to_numpy(dtype=float)
    slack = sigmoid(raw_slack)
    demand = data["d_policy_search"].to_numpy(dtype=float)
    demand_boundary = 0.5
    slack_limits = (0.0, 1.0)
    demand_limits = (0.0, 1.0)

    density, x_edges, y_edges = np.histogram2d(
        slack,
        demand,
        bins=(180, 140),
        range=(slack_limits, demand_limits),
    )
    density = gaussian_filter(density, sigma=2.0)
    positive = density[density > 0]
    contour_levels = (
        np.quantile(positive, [0.72, 0.86, 0.94, 0.985])
        if positive.size else np.asarray([])
    )
    contour_levels = np.unique(contour_levels)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    fig = plt.figure(figsize=(10.4, 8.4), facecolor="white")
    grid = fig.add_gridspec(
        4, 4,
        width_ratios=[1.0, 1.0, 1.0, 0.42],
        height_ratios=[0.42, 1.0, 1.0, 1.0],
        hspace=0.06,
        wspace=0.06,
    )
    ax_top = fig.add_subplot(grid[0, :3])
    ax = fig.add_subplot(grid[1:, :3], sharex=ax_top)
    ax_right = fig.add_subplot(grid[1:, 3], sharey=ax)

    ax.scatter(
        slack,
        demand,
        s=1.0,
        alpha=0.035,
        color="#3D78A8",
        edgecolors="none",
        rasterized=True,
        zorder=1,
    )
    if contour_levels.size >= 2:
        ax.contour(
            x_centers,
            y_centers,
            density.T,
            levels=contour_levels,
            colors=["#9BBEDB", "#6C9DC5", "#3D78A8", "#174D75"][-len(contour_levels):],
            linewidths=[0.8, 1.0, 1.2, 1.5][-len(contour_levels):],
            zorder=2,
        )
    ax.axvline(0.5, color="#343A40", lw=1.0, ls="--", zorder=3)
    ax.axhline(demand_boundary, color="#6B7280", lw=1.0, ls=":", zorder=3)
    ax.set_xlim(slack_limits)
    ax.set_ylim(demand_limits)
    ax.set_xlabel("Normalized policy Slack at the preceding real step")
    ax.set_ylabel("Realized policy Demand across all status-2 search nodes")
    ax.grid(True, color="#E5E7EB", lw=0.55, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    label_box = dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.5)
    x_left = slack_limits[0] + 0.035 * (slack_limits[1] - slack_limits[0])
    x_right = slack_limits[1] - 0.035 * (slack_limits[1] - slack_limits[0])
    y_low = demand_limits[0] + 0.045 * (demand_limits[1] - demand_limits[0])
    y_high = demand_limits[1] - 0.04 * (demand_limits[1] - demand_limits[0])
    ax.text(x_right, y_high, "Planning opportunity", ha="right", va="top", fontsize=9, bbox=label_box)
    ax.text(x_right, y_low, "Safe · low need", ha="right", va="bottom", fontsize=9, bbox=label_box)
    ax.text(x_left, y_high, "Conflict · low slack", ha="left", va="top", fontsize=9, bbox=label_box)
    ax.text(x_left, y_low, "Routine control", ha="left", va="bottom", fontsize=9, bbox=label_box)

    ax_top.hist(slack, bins=100, range=(0, 1), color="#4E87B7", alpha=0.85, edgecolor="none")
    ax_top.axvline(0.5, color="#343A40", lw=1.0, ls="--")
    ax_top.set_ylabel("Count")
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.grid(axis="y", color="#E5E7EB", lw=0.55)
    for spine in ("top", "right"):
        ax_top.spines[spine].set_visible(False)

    ax_right.hist(
        demand,
        bins=80,
        range=(0, 1),
        orientation="horizontal",
        color="#4E87B7",
        alpha=0.85,
        edgecolor="none",
    )
    ax_right.axhline(demand_boundary, color="#6B7280", lw=1.0, ls=":")
    ax_right.set_xlabel("Count")
    ax_right.tick_params(axis="y", labelleft=False)
    ax_right.grid(axis="x", color="#E5E7EB", lw=0.55)
    for spine in ("top", "right"):
        ax_right.spines[spine].set_visible(False)

    high_high = float(np.mean((slack >= 0.5) & (demand >= demand_boundary)))
    fig.suptitle("Policy Slack–Demand state space", fontsize=15, y=0.985)
    fig.text(
        0.5,
        0.945,
        f"All eligible real steps (N={len(data):,}); Slack = sigmoid(NOOP logit margin) · "
        f"dashed/dotted lines = 0.5 · high-slack/high-demand = {high_high:.1%}",
        ha="center",
        va="top",
        fontsize=9.2,
        color="#5B6470",
    )
    fig.subplots_adjust(left=0.095, right=0.965, bottom=0.09, top=0.89)
    save_static_figure(fig, output_path)


def plot_planning_state_space_by_action(
    metrics: pd.DataFrame,
    action_column: str,
    action_source: str,
    output_path: Path,
    slack_column: str = "slack_policy",
    slack_is_normalized: bool = False,
    slack_axis_label: str = "Normalized policy Slack at the preceding real step",
    demand_column: str = "d_policy_search",
    demand_axis_label: str = "Policy Demand across all status-2 search nodes",
    figure_title: str = "Policy Slack–Demand state space",
    show_opportunity_labels: bool = True,
) -> None:
    """Plot every eligible real step in normalized Slack–Demand space by action."""
    required = [
        "cross_chunk_boundary",
        "n_status2",
        slack_column,
        demand_column,
        action_column,
    ]
    data = metrics.loc[
        metrics["cross_chunk_boundary"].eq(0) & metrics["n_status2"].gt(0),
        required,
    ].replace([np.inf, -np.inf], np.nan).dropna(
        subset=[slack_column, demand_column, action_column]
    )
    data = data.loc[data[action_column].ge(0)].copy()
    if data.empty:
        raise ValueError(f"No finite eligible rows for {action_column}")

    slack_values = data[slack_column].to_numpy(dtype=float)
    slack = (
        np.clip(slack_values, 0.0, 1.0)
        if slack_is_normalized
        else sigmoid(slack_values)
    )
    demand = data[demand_column].to_numpy(dtype=float)
    actions = data[action_column].to_numpy(dtype=int)

    density, x_edges, y_edges = np.histogram2d(
        slack, demand, bins=(160, 160), range=((0, 1), (0, 1))
    )
    density = gaussian_filter(density, sigma=2.0)
    positive = density[density > 0]
    levels = (
        np.unique(np.quantile(positive, [0.78, 0.90, 0.97]))
        if positive.size else np.asarray([])
    )
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    # Randomized draw order prevents a frequent or last-drawn action from
    # systematically covering the other labels in dense parts of the plane.
    order = np.random.default_rng(2026).permutation(len(data))
    point_colors = np.empty((len(data), 4), dtype=float)
    for action in np.unique(actions):
        point_colors[actions == action] = matplotlib.colors.to_rgba(
            ACTION_COLORS.get(int(action), "#6B7280"), alpha=0.12
        )

    unique_actions = sorted(np.unique(actions))
    action_colors = [
        ACTION_COLORS.get(int(action), "#6B7280") for action in unique_actions
    ]
    fig = plt.figure(figsize=(12.2, 8.5), facecolor="white")
    grid = fig.add_gridspec(
        4,
        5,
        width_ratios=[1.0, 1.0, 1.0, 0.42, 0.92],
        height_ratios=[0.42, 1.0, 1.0, 1.0],
        hspace=0.06,
        wspace=0.06,
    )
    ax_top = fig.add_subplot(grid[0, :3])
    ax = fig.add_subplot(grid[1:, :3], sharex=ax_top)
    ax_right = fig.add_subplot(grid[1:, 3], sharey=ax)
    ax_legend = fig.add_subplot(grid[1:, 4])
    ax_legend.axis("off")
    ax.scatter(
        slack[order],
        demand[order],
        s=1.6,
        c=point_colors[order],
        edgecolors="none",
        rasterized=True,
        zorder=1,
    )
    if levels.size >= 2:
        ax.contour(
            x_centers,
            y_centers,
            density.T,
            levels=levels,
            colors="#374151",
            linewidths=(0.7, 1.0, 1.3)[-len(levels):],
            alpha=0.65,
            zorder=2,
        )
    ax.axvline(0.5, color="#20252B", lw=1.1, ls="--", zorder=3)
    ax.axhline(0.5, color="#20252B", lw=1.1, ls="--", zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(slack_axis_label)
    ax.set_ylabel(demand_axis_label)
    ax.grid(True, color="#E5E7EB", lw=0.55, alpha=0.75)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    if show_opportunity_labels:
        label_box = dict(
            facecolor="white", edgecolor="none", alpha=0.76, pad=1.5
        )
        ax.text(
            0.97, 0.97, "Planning opportunity",
            ha="right", va="top", fontsize=9, bbox=label_box,
        )
        ax.text(
            0.97, 0.03, "Safe · low need",
            ha="right", va="bottom", fontsize=9, bbox=label_box,
        )
        ax.text(
            0.03, 0.97, "Conflict · low slack",
            ha="left", va="top", fontsize=9, bbox=label_box,
        )
        ax.text(
            0.03, 0.03, "Routine control",
            ha="left", va="bottom", fontsize=9, bbox=label_box,
        )

    slack_groups = [slack[actions == action] for action in unique_actions]
    demand_groups = [demand[actions == action] for action in unique_actions]
    ax_top.hist(
        slack_groups,
        bins=80,
        range=(0, 1),
        stacked=True,
        color=action_colors,
        alpha=0.88,
        edgecolor="none",
    )
    ax_top.axvline(0.5, color="#20252B", lw=1.1, ls="--")
    ax_top.set_ylabel("Count")
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.grid(axis="y", color="#E5E7EB", lw=0.55)
    for spine in ("top", "right"):
        ax_top.spines[spine].set_visible(False)

    ax_right.hist(
        demand_groups,
        bins=70,
        range=(0, 1),
        orientation="horizontal",
        stacked=True,
        color=action_colors,
        alpha=0.88,
        edgecolor="none",
    )
    ax_right.axhline(0.5, color="#20252B", lw=1.1, ls="--")
    ax_right.set_xlabel("Count")
    ax_right.tick_params(axis="y", labelleft=False)
    ax_right.grid(axis="x", color="#E5E7EB", lw=0.55)
    for spine in ("top", "right"):
        ax_right.spines[spine].set_visible(False)

    handles = []
    for action in unique_actions:
        count = int(np.sum(actions == action))
        handles.append(
            Line2D(
                [0], [0], marker="o", linestyle="none", markersize=6,
                markerfacecolor=ACTION_COLORS.get(int(action), "#6B7280"),
                markeredgecolor="none",
                label=f"{action_label(int(action))} (n={count:,})",
            )
        )
    ax_legend.legend(
        handles=handles,
        title=f"{action_source} action at real step t",
        frameon=False,
        loc="upper left",
        borderaxespad=0.0,
        fontsize=8.5,
        title_fontsize=9,
    )
    fig.suptitle(
        f"{figure_title} · {action_source} action",
        fontsize=15,
        y=0.985,
    )
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.09, top=0.92)
    save_static_figure(fig, output_path)


def plot_cross_game_policy_margin_by_action(
    game_metrics: Sequence[Tuple[str, pd.DataFrame]],
    output_path: Path,
    coordinate_mode: str = "change",
) -> None:
    """Compare root or root-to-terminal actor-policy maps across games.

    Each central state-space panel is square and retains the S9 point cloud,
    density contours, and action-stacked marginal distributions.  For the
    root coordinate, the 0.5 Slack boundary denotes equal NOOP and
    best-active preference, while the 0.5 Demand boundary splits lower and
    higher active-action ambiguity.  For change/margin coordinates, 0.5
    denotes no terminal-versus-root difference on either axis.
    """
    coordinate_specs = {
        "root": {
            "x_column": "state_x",
            "y_column": "state_y",
            "x_label": "Root Actor Policy Slack",
            "y_label": "Root Actor Policy Demand",
            "title": "Root Actor Policy Slack–Demand by Human Action",
            "quadrants": (
                "Active Action Preferred\nHigh Ambiguity",
                "Waiting Preferred\nHigh Ambiguity",
                "Active Action Preferred\nLow Ambiguity",
                "Waiting Preferred\nLow Ambiguity",
            ),
            "empty_error": "root actor-policy state",
        },
        "change": {
            "x_column": "delta_policy_slack_normalized",
            "y_column": "delta_policy_demand_normalized",
            "x_label": "Normalized Actor Policy Slack Change",
            "y_label": "Normalized Actor Policy Demand Change",
            "title": (
                "Normalized Actor Policy Slack–Demand Changes "
                "by Human Action"
            ),
            "quadrants": (
                "Slack Decreased\nDemand Increased",
                "Slack Increased\nDemand Increased",
                "Slack Decreased\nDemand Decreased",
                "Slack Increased\nDemand Decreased",
            ),
            "empty_error": "actor-policy change",
        },
        "margin": {
            "x_column": "delta_policy_slack_normalized",
            "y_column": "delta_policy_demand_normalized",
            "x_label": "Normalized Actor Policy Slack Margin",
            "y_label": "Normalized Actor Policy Demand Margin",
            "title": (
                "Normalized Actor Policy Slack–Demand Margins "
                "by Human Action"
            ),
            "quadrants": (
                "Negative Slack Margin\nPositive Demand Margin",
                "Positive Slack Margin\nPositive Demand Margin",
                "Negative Slack Margin\nNegative Demand Margin",
                "Positive Slack Margin\nNegative Demand Margin",
            ),
            "empty_error": "actor-policy margin",
        },
    }
    if coordinate_mode not in coordinate_specs:
        raise ValueError(
            "coordinate_mode must be one of "
            + ", ".join(sorted(coordinate_specs))
        )
    spec = coordinate_specs[coordinate_mode]

    prepared: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for game_label, metrics in game_metrics:
        data = _primary_state_data(metrics)
        required = [
            "cross_chunk_boundary",
            "n_status2",
            str(spec["x_column"]),
            str(spec["y_column"]),
            "human_action",
        ]
        data = data.loc[
            data["cross_chunk_boundary"].eq(0)
            & data["n_status2"].gt(0),
            required,
        ].replace([np.inf, -np.inf], np.nan).dropna()
        data = data.loc[data["human_action"].between(0, 5)].copy()
        if data.empty:
            raise ValueError(
                f"No eligible {spec['empty_error']} rows for {game_label}"
            )
        prepared.append(
            (
                game_label,
                np.clip(
                    data[str(spec["x_column"])].to_numpy(dtype=float),
                    0.0,
                    1.0,
                ),
                np.clip(
                    data[str(spec["y_column"])].to_numpy(dtype=float),
                    0.0,
                    1.0,
                ),
                data["human_action"].to_numpy(dtype=int),
            )
        )

    if len(prepared) != 2:
        raise ValueError(
            "Cross-game S9 requires exactly two game datasets"
        )

    fig = plt.figure(figsize=(16.4, 8.8), facecolor="white")
    outer = fig.add_gridspec(
        1,
        2,
        left=0.055,
        right=0.965,
        bottom=0.135,
        top=0.89,
        wspace=0.24,
    )
    rng = np.random.default_rng(2026)
    for panel_index, (game_label, slack, demand, actions) in enumerate(
        prepared
    ):
        inner = outer[0, panel_index].subgridspec(
            2,
            2,
            height_ratios=[0.24, 1.0],
            width_ratios=[1.0, 0.24],
            hspace=0.04,
            wspace=0.04,
        )
        ax_top = fig.add_subplot(inner[0, 0])
        ax = fig.add_subplot(inner[1, 0], sharex=ax_top)
        ax_right = fig.add_subplot(inner[1, 1], sharey=ax)
        ax_corner = fig.add_subplot(inner[0, 1])
        ax_corner.axis("off")

        unique_actions = sorted(np.unique(actions))
        action_colors = [
            ACTION_COLORS.get(int(action), "#6B7280")
            for action in unique_actions
        ]
        point_colors = np.empty((len(actions), 4), dtype=float)
        for action in unique_actions:
            point_colors[actions == action] = matplotlib.colors.to_rgba(
                ACTION_COLORS.get(int(action), "#6B7280"),
                alpha=0.105,
            )
        order = rng.permutation(len(actions))
        ax.scatter(
            slack[order],
            demand[order],
            s=1.35,
            c=point_colors[order],
            edgecolors="none",
            rasterized=True,
            zorder=1,
        )

        density, x_edges, y_edges = np.histogram2d(
            slack,
            demand,
            bins=(160, 160),
            range=((0, 1), (0, 1)),
        )
        density = gaussian_filter(density, sigma=2.0)
        positive = density[density > 0]
        levels = (
            np.unique(np.quantile(positive, [0.78, 0.90, 0.97]))
            if positive.size
            else np.asarray([])
        )
        if levels.size >= 2:
            x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
            y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
            ax.contour(
                x_centers,
                y_centers,
                density.T,
                levels=levels,
                colors="#374151",
                linewidths=(0.8, 1.1, 1.45)[-len(levels):],
                alpha=0.72,
                zorder=2,
            )

        ax.axvline(0.5, color="#20252B", lw=1.1, ls="--", zorder=3)
        ax.axhline(0.5, color="#20252B", lw=1.1, ls="--", zorder=3)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_box_aspect(1)
        ax.set_xlabel(str(spec["x_label"]), fontsize=11)
        ax.set_ylabel(str(spec["y_label"]), fontsize=11)
        ax.grid(True, color="#E5E7EB", lw=0.55, alpha=0.75)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        quadrant_box = dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.78,
            pad=1.8,
        )
        quadrant_style = dict(
            fontsize=11.5,
            fontweight="normal",
            color="#20252B",
            bbox=quadrant_box,
            zorder=5,
        )
        (
            upper_left,
            upper_right,
            lower_left,
            lower_right,
        ) = spec["quadrants"]
        ax.text(
            0.035,
            0.965,
            str(upper_left),
            ha="left",
            va="top",
            **quadrant_style,
        )
        ax.text(
            0.965,
            0.965,
            str(upper_right),
            ha="right",
            va="top",
            **quadrant_style,
        )
        ax.text(
            0.035,
            0.035,
            str(lower_left),
            ha="left",
            va="bottom",
            **quadrant_style,
        )
        ax.text(
            0.965,
            0.035,
            str(lower_right),
            ha="right",
            va="bottom",
            **quadrant_style,
        )

        slack_groups = [
            slack[actions == action] for action in unique_actions
        ]
        demand_groups = [
            demand[actions == action] for action in unique_actions
        ]
        ax_top.hist(
            slack_groups,
            bins=80,
            range=(0, 1),
            stacked=True,
            color=action_colors,
            alpha=0.88,
            edgecolor="none",
        )
        ax_top.axvline(0.5, color="#20252B", lw=1.1, ls="--")
        ax_top.set_ylabel("Count", fontsize=9.5)
        ax_top.tick_params(axis="x", labelbottom=False)
        ax_top.grid(axis="y", color="#E5E7EB", lw=0.55)
        ax_top.set_title(
            f"{game_label} (N={len(actions):,})",
            fontsize=14,
            fontweight="normal",
            pad=10,
        )
        for spine in ("top", "right"):
            ax_top.spines[spine].set_visible(False)

        ax_right.hist(
            demand_groups,
            bins=70,
            range=(0, 1),
            orientation="horizontal",
            stacked=True,
            color=action_colors,
            alpha=0.88,
            edgecolor="none",
        )
        ax_right.axhline(0.5, color="#20252B", lw=1.1, ls="--")
        ax_right.set_xlabel("Count", fontsize=9.5)
        ax_right.tick_params(axis="y", labelleft=False)
        ax_right.grid(axis="x", color="#E5E7EB", lw=0.55)
        for spine in ("top", "right"):
            ax_right.spines[spine].set_visible(False)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=7,
            markerfacecolor=ACTION_COLORS[action],
            markeredgecolor="none",
            label=action_label(action),
        )
        for action in range(6)
    ]
    fig.legend(
        handles=legend_handles,
        title="Human action at real step t",
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=6,
        fontsize=10,
        title_fontsize=10.5,
    )
    fig.suptitle(
        str(spec["title"]),
        fontsize=16,
        fontweight="normal",
        y=0.982,
    )
    save_static_figure(fig, output_path)


def generate_cross_game_actor_policy_s9(
    subject: int,
    output_root: Path,
    shared_cache_root: Path = DEFAULT_SHARED_CACHE_ROOT,
) -> Optional[Path]:
    """Build cross-game root, change, and margin actor-policy figures."""
    game_metrics: List[Tuple[str, pd.DataFrame]] = []
    for game, game_label in ((1, "Pong"), (2, "Space Invaders")):
        paths = shared_cache_paths(shared_cache_root, subject, game)
        if not paths.actor_metrics.exists() or not paths.manifest.exists():
            log(
                "Skip cross-game actor S9; missing shifted cache for "
                f"sub{subject:03d}, game {game}: {paths.root}"
            )
            return None
        manifest = read_shared_manifest(paths)
        variant = (
            manifest.get("components", {})
            .get("actor_policy", {})
            .get("coordinate_variant")
        )
        if variant != COORDINATE_VARIANT:
            log(
                "Skip cross-game actor S9; cache coordinate variant is not "
                f"current for game {game}: {variant!r}"
            )
            return None
        metrics = load_actor_metrics(paths)
        game_metrics.append((game_label, metrics))

    output_dir = (
        output_root
        / f"sub{subject:03d}_cross_game"
        / "figures"
        / "supplementary"
    )
    change_path = (
        output_dir
        / (
            "supplementary_figureS9_"
            "normalized_actor_policy_change_by_human_action.png"
        )
    )
    root_path = (
        output_dir
        / (
            "supplementary_figureS7_"
            "root_actor_policy_state_by_human_action.png"
        )
    )
    margin_path = (
        output_dir
        / (
            "supplementary_figureS9_"
            "normalized_actor_policy_margin_by_human_action.png"
        )
    )
    plot_cross_game_policy_margin_by_action(
        game_metrics,
        change_path,
        coordinate_mode="change",
    )
    plot_cross_game_policy_margin_by_action(
        game_metrics,
        root_path,
        coordinate_mode="root",
    )
    plot_cross_game_policy_margin_by_action(
        game_metrics,
        margin_path,
        coordinate_mode="margin",
    )
    return change_path


def _planning_panel_contours(
    ax: plt.Axes,
    slack: np.ndarray,
    demand: np.ndarray,
) -> None:
    """Add within-panel density contours without changing the point sample."""
    if len(slack) < 2:
        return
    density, x_edges, y_edges = np.histogram2d(
        slack, demand, bins=(90, 90), range=((0, 1), (0, 1))
    )
    density = gaussian_filter(density, sigma=1.6)
    positive = density[density > 0]
    if positive.size == 0:
        return
    levels = np.unique(np.quantile(positive, [0.80, 0.92, 0.975]))
    if levels.size < 2:
        return
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    ax.contour(
        x_centers,
        y_centers,
        density.T,
        levels=levels,
        colors="#374151",
        linewidths=(0.65, 0.9, 1.15)[-len(levels):],
        alpha=0.68,
        zorder=2,
    )


def _format_conditioned_state_axis(ax: plt.Axes) -> None:
    ax.axvline(0.5, color="#20252B", lw=0.9, ls="--", zorder=3)
    ax.axhline(0.5, color="#20252B", lw=0.9, ls="--", zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, color="#E5E7EB", lw=0.5, alpha=0.72)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_state_space_conditioned_on_previous_action(
    metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Facet the current human action by the preceding human action."""
    required = [
        "cross_chunk_boundary",
        "n_status2",
        "slack_policy",
        "d_policy_search",
        "previous_human_action",
        "human_action",
    ]
    data = metrics.loc[
        metrics["cross_chunk_boundary"].eq(0) & metrics["n_status2"].gt(0),
        required,
    ].replace([np.inf, -np.inf], np.nan).dropna(
        subset=[
            "slack_policy",
            "d_policy_search",
            "previous_human_action",
            "human_action",
        ]
    )
    data = data.loc[
        data["previous_human_action"].between(0, 5)
        & data["human_action"].between(0, 5)
    ].copy()
    if data.empty:
        raise ValueError("No eligible rows for previous-human-action facets")
    data["slack_normalized"] = sigmoid(
        data["slack_policy"].to_numpy(dtype=float)
    )

    fig, axes = plt.subplots(
        2, 3, figsize=(14.4, 9.1), sharex=True, sharey=True, facecolor="white"
    )
    rng = np.random.default_rng(2026)
    for previous_action, ax in enumerate(axes.ravel()):
        panel = data.loc[data["previous_human_action"].eq(previous_action)]
        slack = panel["slack_normalized"].to_numpy(dtype=float)
        demand = panel["d_policy_search"].to_numpy(dtype=float)
        current = panel["human_action"].to_numpy(dtype=int)
        order = rng.permutation(len(panel))
        point_colors = np.empty((len(panel), 4), dtype=float)
        for action in np.unique(current):
            point_colors[current == action] = matplotlib.colors.to_rgba(
                ACTION_COLORS.get(int(action), "#6B7280"), alpha=0.11
            )
        ax.scatter(
            slack[order],
            demand[order],
            s=1.25,
            c=point_colors[order],
            edgecolors="none",
            rasterized=True,
            zorder=1,
        )
        _planning_panel_contours(ax, slack, demand)
        _format_conditioned_state_axis(ax)
        ax.set_title(
            f"Prev = {action_label(previous_action)} · N={len(panel):,}",
            fontsize=10.5,
        )

    for ax in axes[:, 0]:
        ax.set_ylabel("Policy Demand")
    for ax in axes[-1, :]:
        ax.set_xlabel("Normalized policy Slack")
    handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=6,
            markerfacecolor=ACTION_COLORS[action], markeredgecolor="none",
            label=action_label(action),
        )
        for action in range(6)
    ]
    fig.legend(
        handles=handles,
        title="Current human action at real step t",
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=6,
        fontsize=9,
        title_fontsize=9.5,
    )
    fig.suptitle(
        "Policy Slack–Demand state space conditioned on previous human action",
        fontsize=15,
        y=0.992,
    )
    fig.text(
        0.5,
        0.955,
        f"Eligible real steps with valid previous/current action (N={len(data):,}); "
        "point color = current human action; "
        "contours are computed within each panel",
        ha="center",
        va="top",
        fontsize=9.2,
        color="#5B6470",
    )
    fig.subplots_adjust(
        left=0.07, right=0.985, bottom=0.13, top=0.88, hspace=0.28, wspace=0.10
    )
    save_static_figure(fig, output_path)


def plot_state_space_human_transitions(
    metrics: pd.DataFrame,
    output_path: Path,
    slack_axis_label: str = "Normalized policy Slack",
    demand_column: str = "d_policy_search",
    demand_axis_label: str = "Policy Demand",
    figure_title: str = (
        "Human action transitions in the conditional Slack–Demand state space"
    ),
    coordinate_note: str = "",
) -> None:
    """Facet behavioral persistence versus transition by preceding action."""
    required = [
        "cross_chunk_boundary",
        "n_status2",
        "slack_policy",
        demand_column,
        "previous_human_action",
        "human_action",
    ]
    data = metrics.loc[
        metrics["cross_chunk_boundary"].eq(0) & metrics["n_status2"].gt(0),
        required,
    ].replace([np.inf, -np.inf], np.nan).dropna(
        subset=[
            "slack_policy",
            demand_column,
            "previous_human_action",
            "human_action",
        ]
    )
    data = data.loc[
        data["previous_human_action"].between(0, 5)
        & data["human_action"].between(0, 5)
    ].copy()
    if data.empty:
        raise ValueError("No eligible rows for human-transition facets")
    data["slack_normalized"] = sigmoid(
        data["slack_policy"].to_numpy(dtype=float)
    )
    data["transition"] = (
        data["human_action"].astype(int)
        != data["previous_human_action"].astype(int)
    ).astype(int)
    transition_colors = {0: "#4E79A7", 1: "#E15759"}

    fig, axes = plt.subplots(
        2, 3, figsize=(14.4, 9.1), sharex=True, sharey=True, facecolor="white"
    )
    for previous_action, ax in enumerate(axes.ravel()):
        panel = data.loc[data["previous_human_action"].eq(previous_action)]
        slack = panel["slack_normalized"].to_numpy(dtype=float)
        demand = panel[demand_column].to_numpy(dtype=float)
        transition = panel["transition"].to_numpy(dtype=int)
        stable = transition == 0
        changed = transition == 1
        ax.scatter(
            slack[stable],
            demand[stable],
            s=1.1,
            color=transition_colors[0],
            alpha=0.065,
            edgecolors="none",
            rasterized=True,
            zorder=1,
        )
        ax.scatter(
            slack[changed],
            demand[changed],
            s=2.0,
            color=transition_colors[1],
            alpha=0.30,
            edgecolors="none",
            rasterized=True,
            zorder=2,
        )
        _planning_panel_contours(ax, slack, demand)
        _format_conditioned_state_axis(ax)
        rate = float(np.mean(transition)) if len(panel) else np.nan
        event_name = "Commit" if previous_action == 0 else "Switch"
        ax.set_title(
            f"Prev = {action_label(previous_action)} · N={len(panel):,} · "
            f"{event_name}={rate:.1%}",
            fontsize=10.5,
        )

    for ax in axes[:, 0]:
        ax.set_ylabel(demand_axis_label)
    for ax in axes[-1, :]:
        ax.set_xlabel(slack_axis_label)
    handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=6,
            markerfacecolor=transition_colors[0], markeredgecolor="none",
            label="Continue NOOP / stay",
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=6,
            markerfacecolor=transition_colors[1], markeredgecolor="none",
            label="Commit / switch",
        ),
    ]
    fig.legend(
        handles=handles,
        title="Human action transition at real step t",
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=2,
        fontsize=9,
        title_fontsize=9.5,
    )
    fig.suptitle(
        figure_title,
        fontsize=15,
        y=0.992,
    )
    fig.subplots_adjust(
        left=0.07, right=0.985, bottom=0.13, top=0.92, hspace=0.28, wspace=0.10
    )
    save_static_figure(fig, output_path)


STATE_BINS = 25
PRIMARY_MIN_BIN = 50
CONDITIONAL_MIN_BIN = 20
TRANSITION_STATE_BINS = 50


def _add_delta_first_policy_coordinates(data: pd.DataFrame) -> pd.DataFrame:
    """Add terminal-minus-start actor-policy coordinates.

    Slack is differenced on the raw NOOP-versus-best-active logit-margin
    scale and only then mapped to (0, 1) with a sigmoid.  Demand is already a
    normalized entropy in [0, 1], so its raw difference lies in [-1, 1] and is
    mapped affinely to [0, 1].  In both axes, 0.5 denotes no final-cur versus
    root difference.

    This is a contrast between source-state actor policies evaluated at two
    different states.  It must not be described as an update of one policy.
    """
    required = {
        "slack_policy",
        "d_policy_root_start",
        "final_cur_policy_slack_logit",
        "final_cur_policy_demand",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(
            "Delta-first policy coordinate missing columns: "
            + ", ".join(missing)
        )
    result = data.copy()
    result["delta_policy_slack_raw"] = (
        result["final_cur_policy_slack_logit"].to_numpy(dtype=float)
        - result["slack_policy"].to_numpy(dtype=float)
    )
    result["delta_policy_demand_raw"] = (
        result["final_cur_policy_demand"].to_numpy(dtype=float)
        - result["d_policy_root_start"].to_numpy(dtype=float)
    )
    result["delta_policy_slack_normalized"] = sigmoid(
        result["delta_policy_slack_raw"].to_numpy(dtype=float)
    )
    result["delta_policy_demand_normalized"] = np.clip(
        (
            result["delta_policy_demand_raw"].to_numpy(dtype=float)
            + 1.0
        )
        / 2.0,
        0.0,
        1.0,
    )
    return result


def _primary_state_data(metrics: pd.DataFrame) -> pd.DataFrame:
    data = metrics.loc[
        metrics["cross_chunk_boundary"].eq(0) & metrics["n_status2"].gt(0)
    ].replace([np.inf, -np.inf], np.nan).copy()
    # Both axes use the actor policy attached to this behavioral real step
    # after correcting the logger's one-real-step source/destination offset.
    # Status-2 rows are retained for realized-search outcomes, but no
    # search-derived quantity defines the start coordinate itself.
    data = data.dropna(subset=["slack_policy", "d_policy_root_start"])
    data["state_x"] = sigmoid(data["slack_policy"].to_numpy(dtype=float))
    data["state_y"] = data["d_policy_root_start"].to_numpy(dtype=float)
    return _add_delta_first_policy_coordinates(data)


def _bin_indices(
    x: np.ndarray,
    y: np.ndarray,
    bins: int = STATE_BINS,
) -> Tuple[np.ndarray, np.ndarray]:
    ix = np.clip((np.asarray(x, dtype=float) * bins).astype(int), 0, bins - 1)
    iy = np.clip((np.asarray(y, dtype=float) * bins).astype(int), 0, bins - 1)
    return ix, iy


def _binned_count(
    x: np.ndarray,
    y: np.ndarray,
    bins: int = STATE_BINS,
) -> np.ndarray:
    ix, iy = _bin_indices(x, y, bins)
    counts = np.zeros((bins, bins), dtype=int)
    np.add.at(counts, (iy, ix), 1)
    return counts


def _binned_mean(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    min_count: int = PRIMARY_MIN_BIN,
    bins: int = STATE_BINS,
    log1p_geometric: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)
    x, y, values = x[valid], y[valid], values[valid]
    if log1p_geometric:
        values = np.log1p(np.clip(values, 0.0, None))
    ix, iy = _bin_indices(x, y, bins)
    counts = np.zeros((bins, bins), dtype=int)
    totals = np.zeros((bins, bins), dtype=float)
    np.add.at(counts, (iy, ix), 1)
    np.add.at(totals, (iy, ix), values)
    grid = np.full((bins, bins), np.nan, dtype=float)
    shown = counts >= min_count
    grid[shown] = totals[shown] / counts[shown]
    if log1p_geometric:
        grid[shown] = np.expm1(grid[shown])
    return grid, counts


def _binned_binary_probability(
    x: np.ndarray,
    y: np.ndarray,
    event: np.ndarray,
    min_count: int = CONDITIONAL_MIN_BIN,
    min_each_class: int = 5,
    bins: int = STATE_BINS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    event = np.asarray(event, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(event)
    x, y, event = x[valid], y[valid], event[valid]
    ix, iy = _bin_indices(x, y, bins)
    counts = np.zeros((bins, bins), dtype=int)
    events = np.zeros((bins, bins), dtype=int)
    np.add.at(counts, (iy, ix), 1)
    np.add.at(events, (iy, ix), event.astype(int))
    nonevents = counts - events
    shown = (
        (counts >= min_count)
        & (events >= min_each_class)
        & (nonevents >= min_each_class)
    )
    grid = np.full((bins, bins), np.nan, dtype=float)
    grid[shown] = events[shown] / counts[shown]
    return grid, counts, events


def _binned_standardized_group_difference(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    event: np.ndarray,
    min_each_group: int = 5,
    bins: int = STATE_BINS,
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    values = np.asarray(values, dtype=float)
    event = np.asarray(event, dtype=int)
    valid = (
        np.isfinite(x) & np.isfinite(y) & np.isfinite(values)
        & np.isin(event, [0, 1])
    )
    x, y, values, event = x[valid], y[valid], values[valid], event[valid]
    scale = float(np.std(values))
    if not np.isfinite(scale) or scale <= EPS:
        return np.full((bins, bins), np.nan), np.zeros((bins, bins), dtype=int)
    values = (values - float(np.mean(values))) / scale
    ix, iy = _bin_indices(x, y, bins)
    count0 = np.zeros((bins, bins), dtype=int)
    count1 = np.zeros((bins, bins), dtype=int)
    sum0 = np.zeros((bins, bins), dtype=float)
    sum1 = np.zeros((bins, bins), dtype=float)
    stable = event == 0
    changed = event == 1
    np.add.at(count0, (iy[stable], ix[stable]), 1)
    np.add.at(count1, (iy[changed], ix[changed]), 1)
    np.add.at(sum0, (iy[stable], ix[stable]), values[stable])
    np.add.at(sum1, (iy[changed], ix[changed]), values[changed])
    shown = (count0 >= min_each_group) & (count1 >= min_each_group)
    grid = np.full((bins, bins), np.nan, dtype=float)
    grid[shown] = sum1[shown] / count1[shown] - sum0[shown] / count0[shown]
    return grid, count0 + count1


def _finite_limits(
    grids: Sequence[np.ndarray],
    diverging: bool = False,
    lower_quantile: float = 0.02,
    upper_quantile: float = 0.98,
) -> Tuple[float, float]:
    values = np.concatenate(
        [grid[np.isfinite(grid)] for grid in grids if np.any(np.isfinite(grid))]
    )
    if values.size == 0:
        return (-1.0, 1.0) if diverging else (0.0, 1.0)
    if diverging:
        bound = float(np.quantile(np.abs(values), upper_quantile))
        return -max(bound, EPS), max(bound, EPS)
    low, high = np.quantile(values, [lower_quantile, upper_quantile])
    if high - low <= EPS:
        high = low + 1.0
    return float(low), float(high)


def _plot_state_heatmap(
    fig: plt.Figure,
    ax: plt.Axes,
    grid: np.ndarray,
    title: str,
    color_label: str,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    contour_xy: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> None:
    cmap_object = plt.get_cmap(cmap).copy()
    cmap_object.set_bad("#F1F3F5")
    image_artist = ax.imshow(
        np.ma.masked_invalid(grid),
        origin="lower",
        extent=(0, 1, 0, 1),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap_object,
        vmin=vmin,
        vmax=vmax,
        zorder=1,
    )
    if contour_xy is not None:
        _planning_panel_contours(ax, contour_xy[0], contour_xy[1])
    _format_conditioned_state_axis(ax)
    ax.set_title(title, fontsize=10.2)
    colorbar = fig.colorbar(image_artist, ax=ax, fraction=0.046, pad=0.025)
    colorbar.set_label(color_label, fontsize=8.5)
    colorbar.ax.tick_params(labelsize=7.5)


def _finish_state_grid(
    fig: plt.Figure,
    axes: np.ndarray,
    title: str,
    subtitle: str,
    top: float = 0.88,
) -> None:
    axes = np.asarray(axes)
    for ax in axes[:, 0]:
        ax.set_ylabel("Policy Demand")
    for ax in axes[-1, :]:
        ax.set_xlabel("Normalized policy Slack")
    fig.suptitle(title, fontsize=15, y=0.992)
    fig.text(
        0.5, 0.955, subtitle,
        ha="center", va="top", fontsize=9.2, color="#5B6470",
    )
    fig.subplots_adjust(
        left=0.065, right=0.985, bottom=0.075, top=top,
        hspace=0.30, wspace=0.25,
    )


def plot_fig6_search_geometry(metrics: pd.DataFrame, output_path: Path) -> None:
    data = _primary_state_data(metrics)
    x = data["state_x"].to_numpy(dtype=float)
    y = data["state_y"].to_numpy(dtype=float)
    specifications = [
        ("topology_branching_normalized", "Root-branch allocation diversity", "Mean"),
        ("mean_depth", "Mean bout terminal depth", "Bout-terminal depth"),
        ("deep_fraction", "Deep-node allocation fraction", "Mean fraction"),
        ("revisit_fraction", "Observed path-revisit fraction", "Mean fraction"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 9.0), sharex=True, sharey=True)
    for ax, (column, title, label) in zip(axes.ravel(), specifications):
        grid, counts = _binned_mean(x, y, data[column].to_numpy(dtype=float))
        vmin, vmax = _finite_limits([grid])
        _plot_state_heatmap(
            fig, ax, grid,
            f"{title}\nvalid N={int(np.isfinite(data[column]).sum()):,}",
            label, vmin=vmin, vmax=vmax, contour_xy=(x, y),
        )
    _finish_state_grid(
        fig, axes,
        "Fig 6 · Search geometry across the Policy Slack–Demand state space",
        f"Same 25×25 bins in every panel; bins with n<{PRIMARY_MIN_BIN} are masked; "
        f"all status-2 search windows contain 29 nodes (N={len(data):,})",
    )
    save_static_figure(fig, output_path)


def plot_fig7_search_evidence(metrics: pd.DataFrame, output_path: Path) -> None:
    data = _primary_state_data(metrics)
    x = data["state_x"].to_numpy(dtype=float)
    y = data["state_y"].to_numpy(dtype=float)
    between_share = (
        data["d_policy_between"].to_numpy(dtype=float)
        / np.maximum(data["d_policy_search"].to_numpy(dtype=float), EPS)
    )
    specifications = [
        (data["d_policy_within"].to_numpy(dtype=float), "Within-node policy Demand", "Mean", False, False),
        (data["d_policy_between"].to_numpy(dtype=float), "Between-node policy Demand", "Mean", False, False),
        (between_share, "Between-node share", "Mean share", False, False),
        (data["q0_valid"].to_numpy(dtype=float), "Valid initial root-Q fraction", "Fraction", False, False),
        (data["q_resolution"].to_numpy(dtype=float), "Strict Q resolution", "Start − final", True, False),
        (data["q_resolution_robust"].to_numpy(dtype=float), "Q early–late resolution", "Early − late", True, False),
        (data["q_root_update_norm"].to_numpy(dtype=float), "Root-Q update", "Geometric mean", False, True),
        (data["value_surprise_mean"].to_numpy(dtype=float), "Value surprise", "Geometric mean", False, True),
    ]
    grids = []
    for values, _, _, _, geometric in specifications:
        grid, _ = _binned_mean(x, y, values, log1p_geometric=geometric)
        grids.append(grid)
    fig, axes = plt.subplots(2, 4, figsize=(16.8, 8.4), sharex=True, sharey=True)
    for ax, specification, grid in zip(axes.ravel(), specifications, grids):
        values, title, label, diverging, _ = specification
        vmin, vmax = _finite_limits([grid], diverging=diverging)
        if title == "Valid initial root-Q fraction":
            vmin, vmax = 0.0, 1.0
        _plot_state_heatmap(
            fig, ax, grid,
            f"{title}\nvalid N={int(np.isfinite(values).sum()):,}",
            label,
            cmap="RdBu_r" if diverging else "viridis",
            vmin=vmin, vmax=vmax, contour_xy=(x, y),
        )
    _finish_state_grid(
        fig, axes,
        "Supplementary Figure S1 · Construct and evidence-update checks",
        "Root-defined Policy Slack×Demand coordinates; within/between search components are algebraically linked; "
        "strict Q metrics use only valid initial root-Q rows; heavy-tailed metrics use log1p geometric means",
        top=0.86,
    )
    save_static_figure(fig, output_path)


def plot_fig8_transition_probability(metrics: pd.DataFrame, output_path: Path) -> None:
    data = _primary_state_data(metrics)
    data = data.loc[
        data["previous_human_action"].between(0, 5)
        & data["human_action"].between(0, 5)
    ].copy()
    panels = []
    for previous_action in range(6):
        panel = data.loc[data["previous_human_action"].eq(previous_action)]
        event = panel["human_action"].ne(previous_action).to_numpy(dtype=int)
        grid, counts, events = _binned_binary_probability(
            panel["state_x"], panel["state_y"], event
        )
        panels.append((previous_action, panel, event, grid, counts, events))
    finite = np.concatenate([item[3][np.isfinite(item[3])] for item in panels])
    vmax = min(1.0, max(0.20, float(np.quantile(finite, 0.98)))) if finite.size else 1.0
    fig, axes = plt.subplots(2, 3, figsize=(14.4, 9.0), sharex=True, sharey=True)
    for ax, (previous_action, panel, event, grid, _, _) in zip(axes.ravel(), panels):
        event_name = "Commit" if previous_action == 0 else "Switch"
        _plot_state_heatmap(
            fig, ax, grid,
            f"Prev = {action_label(previous_action)} · N={len(panel):,}\n"
            f"overall {event_name}={float(np.mean(event)):.1%}",
            f"P({event_name})", cmap="magma", vmin=0.0, vmax=vmax,
            contour_xy=(panel["state_x"].to_numpy(), panel["state_y"].to_numpy()),
        )
    _finish_state_grid(
        fig, axes,
        "Supplementary Figure S2 · Human transitions conditioned on previous action",
        f"Observed bin probabilities; bins require n≥{CONDITIONAL_MIN_BIN} and at least 5 events/non-events; "
        "all panels use the canonical Policy Slack–Demand coordinates",
    )
    save_static_figure(fig, output_path)


def plot_fig9_action_probability(metrics: pd.DataFrame, output_path: Path) -> None:
    data = _primary_state_data(metrics)
    data = data.loc[data["human_action"].between(0, 5)].copy()
    fig, axes = plt.subplots(2, 3, figsize=(14.4, 9.0), sharex=True, sharey=True)
    for action, ax in enumerate(axes.ravel()):
        event = data["human_action"].eq(action).to_numpy(dtype=int)
        grid, _, _ = _binned_binary_probability(
            data["state_x"], data["state_y"], event,
            min_count=PRIMARY_MIN_BIN,
        )
        action_rows = data.loc[data["human_action"].eq(action)]
        _plot_state_heatmap(
            fig, ax, grid,
            f"{action_label(action)} · n={int(np.sum(event)):,}",
            f"P({action_label(action)})", cmap="magma", vmin=0.0, vmax=1.0,
            contour_xy=(
                action_rows["state_x"].to_numpy(dtype=float),
                action_rows["state_y"].to_numpy(dtype=float),
            ),
        )
    _finish_state_grid(
        fig, axes,
        "Supplementary Figure S3 · Human-action probability surfaces",
        "Each panel shows the observed conditional probability of one current action; "
        "contours show that action's density; previous action is not controlled in this descriptive figure",
    )
    save_static_figure(fig, output_path)


def plot_fig10_commit_spline(metrics: pd.DataFrame, output_path: Path) -> None:
    import patsy
    import statsmodels.api as sm

    data = _primary_state_data(metrics)
    data = data.loc[
        data["previous_human_action"].eq(0)
        & data["human_action"].between(0, 5)
    ].copy()
    data["commit"] = data["human_action"].ne(0).astype(int)
    for column in ("noop_run_length_prev", "previous_reward", "previous_done"):
        median = float(data[column].median()) if np.any(data[column].notna()) else 0.0
        data[column] = data[column].fillna(median)
    formula = (
        "commit ~ bs(state_x, df=5, degree=3, lower_bound=0, upper_bound=1) "
        "* bs(state_y, df=5, degree=3, lower_bound=0, upper_bound=1) "
        "+ bs(noop_run_length_prev, df=4, degree=3) "
        "+ previous_reward + C(run_label)"
    )
    log(f"Fig 10: fit single tensor-spline binomial GLM on N={len(data):,}")
    model = sm.GLM.from_formula(formula, data=data, family=sm.families.Binomial())
    result = model.fit(
        maxiter=100,
        cov_type="cluster",
        cov_kwds={"groups": data["run_label"]},
    )

    centers = (np.arange(STATE_BINS) + 0.5) / STATE_BINS
    grid_x, grid_y = np.meshgrid(centers, centers)
    support_counts = _binned_count(
        data["state_x"].to_numpy(dtype=float),
        data["state_y"].to_numpy(dtype=float),
    )
    support = support_counts >= CONDITIONAL_MIN_BIN
    reference_run = str(data["run_label"].mode().iloc[0])
    prediction_data = pd.DataFrame(
        {
            "state_x": grid_x[support],
            "state_y": grid_y[support],
            "noop_run_length_prev": float(data["noop_run_length_prev"].median()),
            "previous_reward": float(data["previous_reward"].median()),
            "previous_done": float(data["previous_done"].median()),
            "run_label": reference_run,
        }
    )
    prediction = result.get_prediction(prediction_data).summary_frame()
    fitted = np.full((STATE_BINS, STATE_BINS), np.nan, dtype=float)
    fitted[support] = prediction["mean"].to_numpy()
    ci_width = np.full((STATE_BINS, STATE_BINS), np.nan, dtype=float)
    ci_width[support] = (
        prediction["mean_ci_upper"].to_numpy()
        - prediction["mean_ci_lower"].to_numpy()
    )
    observed, counts, _ = _binned_binary_probability(
        data["state_x"], data["state_y"], data["commit"]
    )
    residual = observed - fitted

    x = data["state_x"].to_numpy(dtype=float)
    y = data["state_y"].to_numpy(dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 9.0), sharex=True, sharey=True)
    panels = [
        (observed, "Observed commit probability", "Probability", "magma", 0.0, 0.5),
        (fitted, "Tensor-spline fitted probability", "Probability", "magma", 0.0, 0.5),
        (residual, "Observed − fitted", "Probability difference", "RdBu_r", None, None),
        (ci_width, "95% CI width", "Probability width", "viridis", 0.0, None),
    ]
    for ax, (grid, title, label, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        if vmin is None or vmax is None:
            auto_min, auto_max = _finite_limits([grid], diverging=(title == "Observed − fitted"))
            vmin = auto_min if vmin is None else vmin
            vmax = auto_max if vmax is None else vmax
        _plot_state_heatmap(
            fig, ax, grid, title, label, cmap=cmap,
            vmin=vmin, vmax=vmax, contour_xy=(x, y),
        )
    _finish_state_grid(
        fig, axes,
        "Fig 10 · Controlled nonlinear association with NOOP-to-commit behavior",
        f"Single tensor B-spline binomial GLM (N={len(data):,}); controls: NOOP duration, reward, run; "
        "cluster-robust uncertainty by run; predictions shown at median controls",
    )
    save_static_figure(fig, output_path)


def plot_fig11_search_behavior_difference(metrics: pd.DataFrame, output_path: Path) -> None:
    data = _primary_state_data(metrics)
    data = data.loc[
        data["previous_human_action"].eq(0)
        & data["human_action"].between(0, 5)
    ].copy()
    event = data["human_action"].ne(0).to_numpy(dtype=int)
    between_share = (
        data["d_policy_between"].to_numpy(dtype=float)
        / np.maximum(data["d_policy_search"].to_numpy(dtype=float), EPS)
    )
    specifications = [
        (data["topology_branching_normalized"].to_numpy(dtype=float), "Root-branch allocation diversity"),
        (data["mean_depth"].to_numpy(dtype=float), "Mean bout terminal depth"),
        (data["deep_fraction"].to_numpy(dtype=float), "Deep-node allocation fraction"),
        (data["revisit_fraction"].to_numpy(dtype=float), "Observed path-revisit fraction"),
        (between_share, "Between-node Demand share"),
        (data["q_resolution_robust"].to_numpy(dtype=float), "Q early–late resolution"),
        (np.log1p(np.clip(data["q_root_update_norm"].to_numpy(dtype=float), 0, None)), "log1p Root-Q update"),
        (np.log1p(np.clip(data["value_surprise_mean"].to_numpy(dtype=float), 0, None)), "log1p Value surprise"),
    ]
    grids = [
        _binned_standardized_group_difference(
            data["state_x"], data["state_y"], values, event
        )[0]
        for values, _ in specifications
    ]
    vmin, vmax = _finite_limits(grids, diverging=True)
    fig, axes = plt.subplots(2, 4, figsize=(16.8, 8.4), sharex=True, sharey=True)
    x = data["state_x"].to_numpy(dtype=float)
    y = data["state_y"].to_numpy(dtype=float)
    for ax, (values, title), grid in zip(axes.ravel(), specifications, grids):
        _plot_state_heatmap(
            fig, ax, grid, title,
            "Commit − continue (SD units)", cmap="RdBu_r",
            vmin=vmin, vmax=vmax, contour_xy=(x, y),
        )
    _finish_state_grid(
        fig, axes,
        "Fig 11 · Search evidence associated with commit within the same state bins",
        "Each cell is the standardized mean difference (commit − continue NOOP) within an identical "
        "Policy Slack–Demand bin; cells require at least 5 observations in each behavior group",
        top=0.86,
    )
    save_static_figure(fig, output_path)


def _q_state_data(metrics: pd.DataFrame) -> pd.DataFrame:
    data = metrics.loc[
        metrics["cross_chunk_boundary"].eq(0)
        & metrics["n_status2"].gt(0)
        & metrics["q0_valid"].eq(1)
    ].replace([np.inf, -np.inf], np.nan).copy()
    data = data.dropna(subset=["slack_q", "d_q_start", "q_resolution"])
    data["q_x"] = sigmoid(data["slack_q"].to_numpy(dtype=float))
    data["q_y_start"] = data["d_q_start"].to_numpy(dtype=float)
    data["q_y_final"] = np.clip(
        data["d_q_start"].to_numpy(dtype=float)
        - data["q_resolution"].to_numpy(dtype=float),
        0.0, 1.0,
    )
    return data


def plot_fig12_q_flow(metrics: pd.DataFrame, output_path: Path) -> None:
    data = _q_state_data(metrics)
    x = data["q_x"].to_numpy(dtype=float)
    y_start = data["q_y_start"].to_numpy(dtype=float)
    y_final = data["q_y_final"].to_numpy(dtype=float)
    delta = y_final - y_start
    start_count = _binned_count(x, y_start)
    final_count = _binned_count(x, y_final)
    start_grid = np.where(start_count >= 1, np.log10(start_count + 1), np.nan)
    final_grid = np.where(final_count >= 1, np.log10(final_count + 1), np.nan)
    delta_grid, delta_count = _binned_mean(x, y_start, delta)
    vmin, vmax = _finite_limits([delta_grid], diverging=True)

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 9.0), sharex=True, sharey=True)
    _plot_state_heatmap(fig, axes[0, 0], start_grid, "Start-state density", "log10(n+1)", contour_xy=(x, y_start))
    _plot_state_heatmap(fig, axes[0, 1], final_grid, "Final-state density", "log10(n+1)", contour_xy=(x, y_final))
    _plot_state_heatmap(
        fig, axes[1, 0], delta_grid, "Mean Q-Demand change", "Final − start",
        cmap="RdBu_r", vmin=vmin, vmax=vmax, contour_xy=(x, y_start),
    )
    ax = axes[1, 1]
    _format_conditioned_state_axis(ax)
    centers = (np.arange(STATE_BINS) + 0.5) / STATE_BINS
    grid_x, grid_y = np.meshgrid(centers, centers)
    shown = np.isfinite(delta_grid) & (delta_count >= PRIMARY_MIN_BIN)
    ax.quiver(
        grid_x[shown], grid_y[shown], np.zeros(np.sum(shown)), delta_grid[shown],
        delta_grid[shown], cmap="RdBu_r", angles="xy", scale_units="xy", scale=1.0,
        width=0.006, headwidth=3.5, headlength=4.5,
    )
    _planning_panel_contours(ax, x, y_start)
    ax.set_title("Vertical flow field\nQ Demand start → final", fontsize=10.2)
    for axis in axes[:, 0]:
        axis.set_ylabel("Q Demand")
    for axis in axes[-1, :]:
        axis.set_xlabel("Normalized Q Slack")
    fig.suptitle("Fig 12 · Q-channel search flow diagnostic", fontsize=15, y=0.992)
    fig.text(
        0.5, 0.955,
        f"Separate Q coordinate system; valid initial root-Q subset N={len(data):,}; "
        "arrow x-position is held fixed and vertical displacement is final − start Q Demand",
        ha="center", va="top", fontsize=9.2, color="#5B6470",
    )
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.075, top=0.88, hspace=0.30, wspace=0.22)
    save_static_figure(fig, output_path)


def plot_fig13_q_sensitivity(metrics: pd.DataFrame, output_path: Path) -> None:
    data = _q_state_data(metrics)
    x = data["q_x"].to_numpy(dtype=float)
    y = data["q_y_start"].to_numpy(dtype=float)
    count = _binned_count(x, y)
    density_grid = np.where(count >= 1, np.log10(count + 1), np.nan)
    q_resolution, _ = _binned_mean(x, y, data["q_resolution"])
    q_update, _ = _binned_mean(x, y, data["q_root_update_norm"], log1p_geometric=True)
    surprise, _ = _binned_mean(x, y, data["value_surprise_mean"], log1p_geometric=True)
    noop_grid, _, _ = _binned_binary_probability(
        x, y, data["human_action"].eq(0).to_numpy(dtype=int),
        min_count=PRIMARY_MIN_BIN,
    )
    risk = data.loc[
        data["previous_human_action"].eq(0)
        & data["human_action"].between(0, 5)
    ]
    commit_grid, _, _ = _binned_binary_probability(
        risk["q_x"], risk["q_y_start"], risk["human_action"].ne(0).to_numpy(dtype=int)
    )
    fig, axes = plt.subplots(2, 3, figsize=(14.6, 8.8), sharex=True, sharey=True)
    specifications = [
        (density_grid, "Q-state occupancy", "log10(n+1)", "viridis", None, None, (x, y)),
        (q_resolution, "Strict Q resolution", "Start − final", "RdBu_r", None, None, (x, y)),
        (q_update, "Root-Q update", "Geometric mean", "viridis", None, None, (x, y)),
        (surprise, "Value surprise", "Geometric mean", "viridis", None, None, (x, y)),
        (noop_grid, "Current human NOOP", "P(NOOP)", "magma", 0.0, 1.0, (x, y)),
        (commit_grid, "Commit given previous NOOP", "P(Commit)", "magma", 0.0, 0.5,
         (risk["q_x"].to_numpy(dtype=float), risk["q_y_start"].to_numpy(dtype=float))),
    ]
    for ax, (grid, title, label, cmap, vmin, vmax, contour) in zip(axes.ravel(), specifications):
        if vmin is None or vmax is None:
            auto_min, auto_max = _finite_limits([grid], diverging=(title == "Strict Q resolution"))
            vmin = auto_min if vmin is None else vmin
            vmax = auto_max if vmax is None else vmax
        _plot_state_heatmap(
            fig, ax, grid, title, label, cmap=cmap,
            vmin=vmin, vmax=vmax, contour_xy=contour,
        )
    for ax in axes[:, 0]:
        ax.set_ylabel("Q Demand at search start")
    for ax in axes[-1, :]:
        ax.set_xlabel("Normalized Q Slack")
    fig.suptitle("Supplementary Figure S4 · Q-channel sensitivity analysis", fontsize=15, y=0.992)
    fig.text(
        0.5, 0.955,
        f"Separate Q Slack×Q Demand coordinates; valid initial root-Q subset N={len(data):,}; "
        f"metric bins require n≥{PRIMARY_MIN_BIN}",
        ha="center", va="top", fontsize=9.2, color="#5B6470",
    )
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.08, top=0.88, hspace=0.30, wspace=0.24)
    save_static_figure(fig, output_path)


def _zscore_finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    scale = float(np.nanstd(values))
    if not np.isfinite(scale) or scale <= EPS:
        return np.zeros_like(values, dtype=float)
    return (values - float(np.nanmean(values))) / scale


def _exact_signflip_p(values: np.ndarray, center: float = 0.0) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)] - float(center)
    n = len(values)
    if n == 0:
        return np.nan
    observed = abs(float(np.mean(values)))
    if n <= 16:
        codes = np.arange(1 << n, dtype=np.uint32)[:, None]
        bits = (codes >> np.arange(n, dtype=np.uint32)[None, :]) & 1
        signs = 1.0 - 2.0 * bits
        null = np.abs(np.mean(signs * values[None, :], axis=1))
        return float(np.mean(null >= observed - 1e-15))
    rng = np.random.default_rng(2026)
    null = np.empty(20_000, dtype=float)
    for index in range(len(null)):
        null[index] = abs(float(np.mean(
            values * rng.choice([-1.0, 1.0], size=n)
        )))
    return float((1 + np.sum(null >= observed)) / (len(null) + 1))


def _bh_qvalues(pvalues: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    q = np.full_like(p, np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    if finite.size == 0:
        return q
    order = finite[np.argsort(p[finite])]
    ranked = p[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[order] = np.clip(ranked, 0.0, 1.0)
    return q


def _summarize_run_effects(
    run_effects: pd.DataFrame,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    grouper: object = group_columns[0] if len(group_columns) == 1 else list(group_columns)
    for keys, group in run_effects.groupby(grouper, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = group["coefficient"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        n = len(values)
        mean = float(np.mean(values)) if n else np.nan
        sem = float(stats.sem(values)) if n >= 2 else np.nan
        critical = float(stats.t.ppf(0.975, n - 1)) if n >= 2 else np.nan
        row = {column: value for column, value in zip(group_columns, keys)}
        row.update(
            {
                "n_runs": n,
                "mean_coefficient": mean,
                "ci_low": mean - critical * sem if n >= 2 else np.nan,
                "ci_high": mean + critical * sem if n >= 2 else np.nan,
                "signflip_p": _exact_signflip_p(values),
                "positive_run_fraction": float(np.mean(values > 0)) if n else np.nan,
            }
        )
        rows.append(row)
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["bh_q"] = _bh_qvalues(summary["signflip_p"])
    return summary


def _run_level_geometry_effects(
    data: pd.DataFrame,
    specifications: Sequence[Tuple[str, str]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    records: List[Dict[str, object]] = []
    for column, label in specifications:
        for run_label, run in data.groupby("run_label", sort=False):
            valid = run[["state_x", "state_y", column]].replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            if len(valid) < 100:
                continue
            z_slack = _zscore_finite(valid["state_x"].to_numpy(dtype=float))
            z_demand = _zscore_finite(valid["state_y"].to_numpy(dtype=float))
            z_outcome = _zscore_finite(valid[column].to_numpy(dtype=float))
            design = np.column_stack(
                [np.ones(len(valid)), z_slack, z_demand, z_slack * z_demand]
            )
            coefficients = np.linalg.lstsq(design, z_outcome, rcond=None)[0]
            for term, coefficient in zip(
                ("Slack", "Demand", "Slack × Demand"), coefficients[1:]
            ):
                records.append(
                    {
                        "outcome": label,
                        "term": term,
                        "run_label": run_label,
                        "n": len(valid),
                        "coefficient": float(coefficient),
                    }
                )
    effects = pd.DataFrame(records)
    return effects, _summarize_run_effects(effects, ["outcome", "term"])


def _prepare_commit_risk(data: pd.DataFrame) -> pd.DataFrame:
    risk = data.loc[
        data["previous_human_action"].eq(0)
        & data["human_action"].between(0, 5)
    ].copy()
    risk["commit"] = risk["human_action"].ne(0).astype(int)
    risk["log_noop_duration"] = np.log1p(
        np.clip(risk["noop_run_length_prev"].to_numpy(dtype=float), 0, None)
    )
    return risk


def _run_level_commit_state_effects(
    risk: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    import statsmodels.api as sm

    records: List[Dict[str, object]] = []
    for run_label, run in risk.groupby("run_label", sort=False):
        columns = [
            "commit", "state_x", "state_y", "log_noop_duration",
            "previous_reward", "previous_done",
        ]
        valid = run[columns].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) < 100 or valid["commit"].nunique() < 2:
            continue
        z_slack = _zscore_finite(valid["state_x"])
        z_demand = _zscore_finite(valid["state_y"])
        design = pd.DataFrame(
            {
                "Slack": z_slack,
                "Demand": z_demand,
                "Slack × Demand": z_slack * z_demand,
                "NOOP duration": _zscore_finite(valid["log_noop_duration"]),
                "Reward": _zscore_finite(valid["previous_reward"]),
                "Done": _zscore_finite(valid["previous_done"]),
            },
            index=valid.index,
        )
        design = sm.add_constant(design, has_constant="add")
        try:
            result = sm.GLM(
                valid["commit"], design, family=sm.families.Binomial()
            ).fit(maxiter=100, disp=0)
        except Exception as error:
            log(f"Commit run model skipped {run_label}: {error}")
            continue
        for term in ("Slack", "Demand", "Slack × Demand"):
            records.append(
                {
                    "term": term,
                    "run_label": run_label,
                    "n": len(valid),
                    "coefficient": float(result.params[term]),
                }
            )
    effects = pd.DataFrame(records)
    return effects, _summarize_run_effects(effects, ["term"])


def _run_level_search_commit_effects(
    risk: pd.DataFrame,
    specifications: Sequence[Tuple[str, str]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    import statsmodels.api as sm

    records: List[Dict[str, object]] = []
    base_columns = [
        "commit", "state_x", "state_y", "log_noop_duration",
        "previous_reward", "previous_done",
    ]
    for column, label in specifications:
        if column not in risk.columns:
            continue
        for run_label, run in risk.groupby("run_label", sort=False):
            valid = run[base_columns + [column]].replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            if (
                len(valid) < 100
                or valid["commit"].nunique() < 2
                or float(np.std(valid[column])) <= EPS
            ):
                continue
            z_slack = _zscore_finite(valid["state_x"])
            z_demand = _zscore_finite(valid["state_y"])
            design = pd.DataFrame(
                {
                    "Search metric": _zscore_finite(valid[column]),
                    "Slack": z_slack,
                    "Demand": z_demand,
                    "Slack × Demand": z_slack * z_demand,
                    "NOOP duration": _zscore_finite(valid["log_noop_duration"]),
                    "Reward": _zscore_finite(valid["previous_reward"]),
                    "Done": _zscore_finite(valid["previous_done"]),
                },
                index=valid.index,
            )
            design = sm.add_constant(design, has_constant="add")
            try:
                result = sm.GLM(
                    valid["commit"], design, family=sm.families.Binomial()
                ).fit(maxiter=100, disp=0)
            except Exception as error:
                log(f"Search run model skipped {label}/{run_label}: {error}")
                continue
            records.append(
                {
                    "metric": label,
                    "column": column,
                    "run_label": run_label,
                    "n": len(valid),
                    "coefficient": float(result.params["Search metric"]),
                }
            )
    effects = pd.DataFrame(records)
    return effects, _summarize_run_effects(effects, ["metric"])


def _partial_commit_circular_null(
    risk: pd.DataFrame,
    permutations: int,
    random_state: int,
) -> Tuple[float, np.ndarray, float]:
    rng = np.random.default_rng(random_state)
    correlations: List[np.ndarray] = []
    valid_lags: List[np.ndarray] = []
    weights: List[float] = []
    for _, run in risk.sort_values(["run_label", "real_step"]).groupby(
        "run_label", sort=False
    ):
        columns = [
            "commit", "state_x", "state_y", "log_noop_duration",
            "previous_reward", "previous_done",
        ]
        valid = run[columns].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) < 100 or valid["commit"].nunique() < 2:
            continue
        controls = np.column_stack(
            [
                np.ones(len(valid)),
                _zscore_finite(valid["state_y"]),
                _zscore_finite(valid["state_y"]) ** 2,
                _zscore_finite(valid["log_noop_duration"]),
                _zscore_finite(valid["previous_reward"]),
                _zscore_finite(valid["previous_done"]),
            ]
        )
        slack = _zscore_finite(valid["state_x"])
        commit = valid["commit"].to_numpy(dtype=float)
        slack_residual = slack - controls @ np.linalg.lstsq(
            controls, slack, rcond=None
        )[0]
        commit_residual = commit - controls @ np.linalg.lstsq(
            controls, commit, rcond=None
        )[0]
        slack_residual -= np.mean(slack_residual)
        commit_residual -= np.mean(commit_residual)
        denominator = float(
            np.sqrt(np.sum(slack_residual ** 2) * np.sum(commit_residual ** 2))
        )
        if denominator <= EPS:
            continue
        n = len(valid)
        cross = np.fft.irfft(
            np.conj(np.fft.rfft(slack_residual))
            * np.fft.rfft(commit_residual),
            n=n,
        ) / denominator
        correlations.append(np.clip(cross, -0.999999, 0.999999))
        exclusion = max(20, int(round(n * 0.01)))
        lags = np.arange(exclusion, max(exclusion + 1, n - exclusion))
        valid_lags.append(lags)
        weights.append(float(max(n - 3, 1)))
    if not correlations:
        return np.nan, np.full(permutations, np.nan), np.nan
    weights_array = np.asarray(weights, dtype=float)
    observed_z = np.average(
        [np.arctanh(values[0]) for values in correlations],
        weights=weights_array,
    )
    null_z = np.empty(permutations, dtype=float)
    for permutation in range(permutations):
        sampled = [
            np.arctanh(values[int(rng.choice(lags))])
            for values, lags in zip(correlations, valid_lags)
        ]
        null_z[permutation] = np.average(sampled, weights=weights_array)
    observed = float(np.tanh(observed_z))
    null = np.tanh(null_z)
    p = float(
        (1 + np.sum(np.abs(null) >= abs(observed))) / (permutations + 1)
    )
    return observed, null, p


def _write_inference_tables(
    inference_dir: Path,
    stem: str,
    effects: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    inference_dir.mkdir(parents=True, exist_ok=True)
    effects.to_csv(inference_dir / f"{stem}_run_effects.csv", index=False)
    summary.to_csv(inference_dir / f"{stem}_summary.csv", index=False)


def _plot_effect_summary(
    ax: plt.Axes,
    summary: pd.DataFrame,
    label_column: str,
    title: str,
    odds_ratio: bool = False,
) -> None:
    shown = summary.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(shown))
    mean = shown["mean_coefficient"].to_numpy(dtype=float)
    low = shown["ci_low"].to_numpy(dtype=float)
    high = shown["ci_high"].to_numpy(dtype=float)
    if odds_ratio:
        mean, low, high = np.exp(mean), np.exp(low), np.exp(high)
        reference = 1.0
        ax.set_xscale("log")
        xlabel = "Adjusted odds ratio per 1 SD"
    else:
        reference = 0.0
        xlabel = "Across-run mean standardized coefficient"
    ax.axvline(reference, color="#343A40", lw=1.0, ls="--")
    ax.errorbar(
        mean, y,
        xerr=np.vstack([mean - low, high - mean]),
        fmt="o", color="#2F6690", ecolor="#8AB0CB",
        capsize=3, markersize=5, lw=1.4,
    )
    labels = []
    for _, row in shown.iterrows():
        q = float(row.get("bh_q", np.nan))
        marker = "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else ""
        labels.append(f"{row[label_column]} {marker}".rstrip())
    ax.set_yticks(y, labels)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", color="#E5E7EB", lw=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_main_figure1_search_geometry(
    metrics: pd.DataFrame,
    output_path: Path,
    inference_dir: Path,
) -> None:
    data = _primary_state_data(metrics)
    x = data["state_x"].to_numpy(dtype=float)
    y = data["state_y"].to_numpy(dtype=float)
    specifications = [
        ("topology_branching_normalized", "Root-branch allocation diversity"),
        ("mean_depth", "Mean bout terminal depth"),
        ("deep_fraction", "Deep-node allocation fraction"),
        ("revisit_fraction", "Observed path-revisit fraction"),
    ]
    effects, summary = _run_level_geometry_effects(data, specifications)
    _write_inference_tables(inference_dir, "main_figure1_geometry", effects, summary)

    count = _binned_count(x, y)
    density = np.where(count >= 1, np.log10(count + 1), np.nan)
    fig, axes = plt.subplots(2, 3, figsize=(16.8, 9.1), facecolor="white")
    _plot_state_heatmap(
        fig, axes[0, 0], density, "Search-start actor-state occupancy",
        "log10(n+1)", contour_xy=(x, y),
    )
    for ax, (column, label) in zip(
        (axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]), specifications
    ):
        grid, _ = _binned_mean(x, y, data[column].to_numpy(dtype=float))
        vmin, vmax = _finite_limits([grid])
        _plot_state_heatmap(
            fig, ax, grid, label, "Bin mean", vmin=vmin, vmax=vmax,
            contour_xy=(x, y),
        )
    coefficient_panel = summary.loc[
        summary["term"].isin(["Slack", "Demand"])
    ].copy()
    short_outcomes = {
        "Root-branch allocation diversity": "Branch",
        "Mean bout terminal depth": "Depth",
        "Deep-node allocation fraction": "Deep",
        "Observed path-revisit fraction": "Revisit",
    }
    short_terms = {
        "Slack": geometry_display("x_short"),
        "Demand": geometry_display("y_short"),
    }
    coefficient_panel["label"] = [
        f"{short_outcomes.get(outcome, outcome)} · {short_terms.get(term, term)}"
        for outcome, term in zip(
            coefficient_panel["outcome"], coefficient_panel["term"]
        )
    ]
    _plot_effect_summary(
        axes[1, 2], coefficient_panel, "label",
        (
            "Run-level inference · "
            f"{geometry_display('x_short')}={geometry_display('x_term')}, "
            f"{geometry_display('y_short')}={geometry_display('y_term')}\n"
            "(exact sign-flip BH-FDR)"
        ),
    )
    for ax in axes[:, 0]:
        ax.set_ylabel(geometry_display("start_y_label"))
    for ax in axes[-1, :2]:
        ax.set_xlabel(geometry_display("start_x_label"))
    fig.suptitle(
        geometry_display("main1_title"),
        fontsize=15, y=0.992,
    )
    fig.subplots_adjust(
        left=0.07, right=0.985, bottom=0.08, top=0.92,
        hspace=0.30, wspace=0.46,
    )
    save_static_figure(fig, output_path)


def _run_adjusted_binned_geometry_curves(
    data: pd.DataFrame,
    x_column: str,
    term: str,
    outcome_column: str,
    bins: int = 20,
    minimum_count: int = 50,
) -> pd.DataFrame:
    """Compute partial-residual bin means separately within each run.

    The nuisance Policy coordinate and Slack×Demand term are removed with the
    same standardized linear model used for the run-level coefficient.  The
    resulting curve and the annotated beta therefore answer the same adjusted
    association question.
    """
    if term not in {"Slack", "Demand"}:
        raise ValueError(f"Unsupported Policy term: {term}")
    edges = np.linspace(0.0, 1.0, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    records: List[Dict[str, object]] = []
    for run_label, run in data.groupby("run_label", sort=False):
        valid = run[["state_x", "state_y", outcome_column]].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        if len(valid) < 100:
            continue
        slack = valid["state_x"].to_numpy(dtype=float)
        demand = valid["state_y"].to_numpy(dtype=float)
        z_slack = _zscore_finite(slack)
        z_demand = _zscore_finite(demand)
        z_outcome = _zscore_finite(
            valid[outcome_column].to_numpy(dtype=float)
        )
        interaction = z_slack * z_demand
        design = np.column_stack(
            [np.ones(len(valid)), z_slack, z_demand, interaction]
        )
        coefficients = np.linalg.lstsq(
            design, z_outcome, rcond=None
        )[0]
        if term == "Slack":
            x = slack
            adjusted_outcome = z_outcome - (
                coefficients[0]
                + coefficients[2] * z_demand
                + coefficients[3] * interaction
            )
        else:
            x = demand
            adjusted_outcome = z_outcome - (
                coefficients[0]
                + coefficients[1] * z_slack
                + coefficients[3] * interaction
            )
        indices = np.clip(np.digitize(x, edges[1:-1]), 0, bins - 1)
        for bin_index, center in enumerate(centers):
            selected = indices == bin_index
            count = int(np.sum(selected))
            if count < minimum_count:
                continue
            records.append(
                {
                    "run_label": str(run_label),
                    "x_column": x_column,
                    "outcome_column": outcome_column,
                    "bin_index": int(bin_index),
                    "bin_center": float(center),
                    "n": count,
                    "mean_outcome": float(
                        np.mean(adjusted_outcome[selected])
                    ),
                }
            )
    return pd.DataFrame(records)


def _plot_run_binned_geometry_relationship(
    ax: plt.Axes,
    curves: pd.DataFrame,
    summary_row: pd.Series,
    color: str,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    """Plot run-wise binned means plus the run-equalized mean and CI."""
    for _, run in curves.groupby("run_label", sort=False):
        ordered = run.sort_values("bin_index")
        ax.plot(
            ordered["bin_center"],
            ordered["mean_outcome"],
            color=color,
            alpha=0.18,
            lw=0.8,
            marker="o",
            markersize=1.8,
            zorder=1,
        )

    aggregate = (
        curves.groupby(["bin_index", "bin_center"], as_index=False)
        .agg(
            mean_outcome=("mean_outcome", "mean"),
            sem_outcome=("mean_outcome", "sem"),
            n_runs=("run_label", "nunique"),
        )
        .sort_values("bin_index")
    )
    aggregate = aggregate.loc[aggregate["n_runs"].ge(3)].copy()
    if not aggregate.empty:
        critical = np.asarray(
            [
                stats.t.ppf(0.975, max(int(n_runs) - 1, 1))
                for n_runs in aggregate["n_runs"]
            ],
            dtype=float,
        )
        mean = aggregate["mean_outcome"].to_numpy(dtype=float)
        interval = (
            critical * aggregate["sem_outcome"].to_numpy(dtype=float)
        )
        center = aggregate["bin_center"].to_numpy(dtype=float)
        ax.fill_between(
            center,
            mean - interval,
            mean + interval,
            color=color,
            alpha=0.16,
            linewidth=0,
            zorder=2,
        )
        ax.plot(
            center,
            mean,
            color=color,
            lw=2.4,
            marker="o",
            markersize=3.2,
            zorder=3,
        )

    coefficient = float(summary_row["mean_coefficient"])
    ci_low = float(summary_row["ci_low"])
    ci_high = float(summary_row["ci_high"])
    q_value = float(summary_row["bh_q"])
    q_text = "<0.001" if q_value < 0.001 else f"={q_value:.3f}"
    ax.text(
        0.04,
        0.96,
        (
            f"Adjusted standardized β={coefficient:+.3f}\n"
            f"95% CI [{ci_low:+.3f}, {ci_high:+.3f}]\n"
            f"BH-FDR q{q_text}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.7,
        bbox=dict(
            boxstyle="round,pad=0.28",
            facecolor="white",
            edgecolor="#D1D5DB",
            alpha=0.90,
        ),
        zorder=5,
    )
    ax.axvline(0.5, color="#4B5563", lw=0.9, ls="--", alpha=0.75)
    ax.set_xlim(0.0, 1.0)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, color="#E5E7EB", lw=0.55, alpha=0.80)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_main_figure3_policy_state_search_geometry(
    metrics: pd.DataFrame,
    output_path: Path,
    inference_dir: Path,
) -> None:
    """Directly show start actor-policy coordinates versus search geometry."""
    data = _primary_state_data(metrics)
    specifications = [
        ("topology_branching_normalized", "Root-branch allocation diversity"),
        ("mean_depth", "Mean bout terminal depth"),
    ]
    effects, summary = _run_level_geometry_effects(data, specifications)
    _write_inference_tables(
        inference_dir,
        "main_figure3_policy_state_search_geometry",
        effects,
        summary,
    )

    panel_specs = [
        (
            "state_x",
            "Slack",
            geometry_display("start_x_label"),
            specifications[0][0],
            specifications[0][1],
            f"{geometry_display('x_term')} → Branch diversity",
            "#2F6690",
        ),
        (
            "state_y",
            "Demand",
            geometry_display("start_y_label"),
            specifications[0][0],
            specifications[0][1],
            f"{geometry_display('y_term')} → Branch diversity",
            "#D17C00",
        ),
        (
            "state_x",
            "Slack",
            geometry_display("start_x_label"),
            specifications[1][0],
            specifications[1][1],
            f"{geometry_display('x_term')} → Bout terminal depth",
            "#2F6690",
        ),
        (
            "state_y",
            "Demand",
            geometry_display("start_y_label"),
            specifications[1][0],
            specifications[1][1],
            f"{geometry_display('y_term')} → Bout terminal depth",
            "#D17C00",
        ),
    ]
    curve_parts: List[pd.DataFrame] = []
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.8), facecolor="white")
    for ax, (
        x_column,
        term,
        x_label,
        outcome_column,
        outcome_label,
        panel_title,
        color,
    ) in zip(axes.ravel(), panel_specs):
        curves = _run_adjusted_binned_geometry_curves(
            data,
            x_column=x_column,
            term=term,
            outcome_column=outcome_column,
        )
        curves["term"] = term
        curves["outcome"] = outcome_label
        curve_parts.append(curves)
        matching = summary.loc[
            summary["outcome"].eq(outcome_label)
            & summary["term"].eq(term)
        ]
        if matching.empty:
            raise ValueError(
                f"Missing geometry inference row for {outcome_label}/{term}"
            )
        _plot_run_binned_geometry_relationship(
            ax,
            curves=curves,
            summary_row=matching.iloc[0],
            color=color,
            title=panel_title,
            x_label=x_label,
            y_label=f"Adjusted standardized {outcome_label.lower()}",
        )

    pd.concat(curve_parts, ignore_index=True).to_csv(
        inference_dir
        / "main_figure3_policy_state_search_geometry_binned_curves.csv",
        index=False,
    )
    fig.suptitle(
        geometry_display("main3_title"),
        fontsize=15,
        y=0.985,
    )
    fig.text(
        0.5,
        0.015,
        (
            "Curves are within-run partial-residual bin means after removing "
            f"the other {geometry_display('model_name')} coordinate and "
            f"{geometry_display('x_term')}×{geometry_display('y_term')} term. "
            "Thin lines: "
            "individual runs; solid line and shading: run-equalized mean ± "
            "95% CI. β values use the same standardized model; "
            "exact sign-flip tests with BH-FDR."
        ),
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#30343B",
    )
    fig.subplots_adjust(
        left=0.09,
        right=0.98,
        bottom=0.11,
        top=0.91,
        hspace=0.34,
        wspace=0.27,
    )
    save_static_figure(fig, output_path)


def _adjusted_policy_geometry_effects(
    metrics: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Estimate pre-search Policy effects after pre-search/run controls.

    Current action is intentionally not a primary control because it follows
    the search window.  Previous action, elapsed same-action/NOOP duration,
    previous reward/done, and within-run progress are available before search.
    """
    data = _primary_state_data(metrics).copy()
    data = data.loc[
        data["previous_human_action"].between(0, 5)
    ].copy()
    data["log_duration"] = np.log1p(
        np.maximum(
            pd.to_numeric(
                data["noop_run_length_prev"], errors="coerce"
            ).fillna(0.0),
            0.0,
        )
    )
    data["run_progress"] = data.groupby(
        "run_label", sort=False
    ).cumcount()
    maximum = data.groupby("run_label", sort=False)["run_progress"].transform(
        "max"
    )
    data["run_progress"] = (
        data["run_progress"] / maximum.replace(0, np.nan)
    ).fillna(0.0)
    specifications = [
        ("topology_branching_normalized", "Root-branch allocation diversity"),
        ("mean_depth", "Mean bout terminal depth"),
        ("deep_fraction", "Deep-node allocation fraction"),
        ("revisit_fraction", "Observed path-revisit fraction"),
    ]
    records: List[Dict[str, object]] = []
    adjusted_parts: List[pd.DataFrame] = []
    for run_label, run in data.groupby("run_label", sort=False):
        action_dummies = pd.get_dummies(
            run["previous_human_action"].astype(int),
            prefix="prev",
            dtype=float,
        ).reindex(columns=[f"prev_{action}" for action in range(1, 6)], fill_value=0.0)
        base = pd.DataFrame(
            {
                "Duration": _zscore_finite(run["log_duration"]),
                "Reward": _zscore_finite(
                    pd.to_numeric(
                        run["previous_reward"], errors="coerce"
                    ).fillna(0.0)
                ),
                "Done": _zscore_finite(
                    pd.to_numeric(
                        run["previous_done"], errors="coerce"
                    ).fillna(0.0)
                ),
                "Run progress": _zscore_finite(run["run_progress"]),
            },
            index=run.index,
        )
        base = pd.concat([base, action_dummies.set_axis(run.index)], axis=1)
        base_values = np.column_stack(
            (np.ones(len(run), dtype=float), base.to_numpy(dtype=float))
        )
        slack = _zscore_finite(run["state_x"].to_numpy(dtype=float))
        demand = _zscore_finite(run["state_y"].to_numpy(dtype=float))
        full_values = np.column_stack(
            (base_values, slack, demand, slack * demand)
        )
        for column, label in specifications:
            valid = np.isfinite(
                pd.to_numeric(run[column], errors="coerce").to_numpy(dtype=float)
            )
            if np.sum(valid) < 100:
                continue
            outcome = _zscore_finite(
                pd.to_numeric(
                    run.loc[valid, column], errors="coerce"
                ).to_numpy(dtype=float)
            )
            reduced_design = base_values[valid]
            full_design = full_values[valid]
            reduced_beta = np.linalg.lstsq(
                reduced_design, outcome, rcond=None
            )[0]
            full_beta = np.linalg.lstsq(
                full_design, outcome, rcond=None
            )[0]
            reduced_prediction = reduced_design @ reduced_beta
            full_prediction = full_design @ full_beta
            denominator = float(np.sum((outcome - np.mean(outcome)) ** 2))
            reduced_r2 = (
                1.0
                - float(np.sum((outcome - reduced_prediction) ** 2))
                / denominator
                if denominator > EPS
                else np.nan
            )
            full_r2 = (
                1.0
                - float(np.sum((outcome - full_prediction) ** 2))
                / denominator
                if denominator > EPS
                else np.nan
            )
            for term, coefficient in zip(
                ("Slack", "Demand", "Slack × Demand"),
                full_beta[-3:],
            ):
                records.append(
                    {
                        "outcome": label,
                        "term": term,
                        "run_label": str(run_label),
                        "n": int(np.sum(valid)),
                        "coefficient": float(coefficient),
                        "controls_r2": reduced_r2,
                        "full_r2": full_r2,
                        "delta_r2": full_r2 - reduced_r2,
                    }
                )
            residual = outcome - reduced_prediction
            residual_scale = float(np.std(residual))
            if residual_scale > EPS:
                residual = residual / residual_scale
            adjusted_parts.append(
                pd.DataFrame(
                    {
                        "run_label": str(run_label),
                        "outcome": label,
                        "state_x": run.loc[valid, "state_x"].to_numpy(dtype=float),
                        "state_y": run.loc[valid, "state_y"].to_numpy(dtype=float),
                        "adjusted_outcome": residual,
                    }
                )
            )
    effects = pd.DataFrame(records)
    summary = _summarize_run_effects(effects, ["outcome", "term"])
    adjusted = (
        pd.concat(adjusted_parts, ignore_index=True)
        if adjusted_parts
        else pd.DataFrame()
    )
    return effects, summary, adjusted


def _fit_commit_surface(
    risk: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    import statsmodels.api as sm

    formula = (
        "commit ~ bs(state_x, df=5, degree=3, lower_bound=0, upper_bound=1) "
        "* bs(state_y, df=5, degree=3, lower_bound=0, upper_bound=1) "
        "+ bs(log_noop_duration, df=4, degree=3) "
        "+ previous_reward + previous_done + C(run_label)"
    )
    result = sm.GLM.from_formula(
        formula, data=risk, family=sm.families.Binomial()
    ).fit(maxiter=100, disp=0)
    centers = (np.arange(STATE_BINS) + 0.5) / STATE_BINS
    grid_x, grid_y = np.meshgrid(centers, centers)
    counts = _binned_count(risk["state_x"], risk["state_y"])
    support = counts >= CONDITIONAL_MIN_BIN
    prediction_data = pd.DataFrame(
        {
            "state_x": grid_x[support],
            "state_y": grid_y[support],
            "log_noop_duration": float(risk["log_noop_duration"].median()),
            "previous_reward": float(risk["previous_reward"].median()),
            "previous_done": float(risk["previous_done"].median()),
            "run_label": str(risk["run_label"].mode().iloc[0]),
        }
    )
    prediction = result.get_prediction(prediction_data).summary_frame()
    fitted = np.full((STATE_BINS, STATE_BINS), np.nan, dtype=float)
    fitted[support] = prediction["mean"].to_numpy(dtype=float)
    ci_width = np.full((STATE_BINS, STATE_BINS), np.nan, dtype=float)
    ci_width[support] = (
        prediction["mean_ci_upper"].to_numpy(dtype=float)
        - prediction["mean_ci_lower"].to_numpy(dtype=float)
    )
    observed, _, _ = _binned_binary_probability(
        risk["state_x"], risk["state_y"], risk["commit"]
    )
    return observed, fitted, ci_width


def plot_main_figure2_human_commit(
    metrics: pd.DataFrame,
    output_path: Path,
    inference_dir: Path,
    permutations: int,
    random_state: int,
) -> None:
    data = _primary_state_data(metrics)
    risk = _prepare_commit_risk(data)
    effects, summary = _run_level_commit_state_effects(risk)
    _write_inference_tables(inference_dir, "main_figure2_commit_state", effects, summary)
    observed, fitted, _ = _fit_commit_surface(risk)
    observed_r, null_r, shift_p = _partial_commit_circular_null(
        risk, permutations=permutations, random_state=random_state
    )
    pd.DataFrame({"null_partial_r": null_r}).to_csv(
        inference_dir / "main_figure2_circular_shift_null.csv", index=False
    )
    pd.DataFrame(
        [{"observed_partial_r": observed_r, "two_sided_p": shift_p,
          "permutations": permutations}]
    ).to_csv(inference_dir / "main_figure2_circular_shift_summary.csv", index=False)

    x = risk["state_x"].to_numpy(dtype=float)
    y = risk["state_y"].to_numpy(dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.2), facecolor="white")
    _plot_state_heatmap(
        fig, axes[0, 0], observed, "Observed NOOP-to-commit probability",
        "Probability", cmap="magma", vmin=0.0, vmax=0.5,
        contour_xy=(x, y),
    )
    _plot_state_heatmap(
        fig, axes[0, 1], fitted,
        "Adjusted tensor-spline probability\n(median controls)",
        "Probability", cmap="magma", vmin=0.0, vmax=0.5,
        contour_xy=(x, y),
    )
    _plot_effect_summary(
        axes[1, 0], summary, "term",
        "Adjusted state effects across runs\n(exact sign-flip BH-FDR)",
        odds_ratio=True,
    )
    ax = axes[1, 1]
    ax.hist(null_r[np.isfinite(null_r)], bins=45, color="#9BBEDB", edgecolor="white")
    ax.axvline(observed_r, color="#B23A48", lw=2.0,
               label=f"Observed partial r={observed_r:.3f}")
    ax.axvline(0.0, color="#343A40", lw=1.0, ls="--")
    ax.set_xlabel("Run-weighted partial correlation")
    ax.set_ylabel("Circular-shift count")
    ax.set_title(
        f"Temporal circular-shift null\ntwo-sided p={shift_p:.4g}"
    )
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(axis="y", color="#E5E7EB", lw=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for axis in axes[0, :]:
        axis.set_xlabel("Normalized root Policy Slack before search")
    axes[0, 0].set_ylabel("Root Policy Demand before search")
    axes[0, 1].set_ylabel("Root Policy Demand before search")
    fig.suptitle(
        "Main Figure 2 · Pre-search opportunity and human commitment",
        fontsize=15, y=0.992,
    )
    fig.text(
        0.5, 0.957,
        f"Previous-human-NOOP risk set N={len(risk):,}; controls: NOOP duration, reward, done, and run; "
        f"run-level coefficients use {risk['run_label'].nunique()} independent run estimates; "
        "null circularly shifts commit residuals within run",
        ha="center", va="top", fontsize=9.1, color="#5B6470",
    )
    fig.subplots_adjust(
        left=0.095, right=0.985, bottom=0.08, top=0.87,
        hspace=0.32, wspace=0.28,
    )
    save_static_figure(fig, output_path)


def plot_main_figure3_search_commit(
    metrics: pd.DataFrame,
    output_path: Path,
    inference_dir: Path,
) -> None:
    data = _primary_state_data(metrics)
    risk = _prepare_commit_risk(data)
    risk["policy_demand_change"] = (
        risk["d_policy_search"] - risk["d_policy_root_start"]
    )
    risk["log_root_q_update"] = np.log1p(
        np.clip(risk["q_root_update_norm"].to_numpy(dtype=float), 0, None)
    )
    risk["log_value_surprise"] = np.log1p(
        np.clip(risk["value_surprise_mean"].to_numpy(dtype=float), 0, None)
    )
    risk["asinh_q_noop_margin_update"] = np.arcsinh(
        risk["q_noop_margin_update"].to_numpy(dtype=float)
    )
    specifications = [
        ("topology_branching_normalized", "Root-branch allocation diversity"),
        ("mean_depth", "Mean bout terminal depth"),
        ("deep_fraction", "Deep-node allocation fraction"),
        ("revisit_fraction", "Observed path-revisit fraction"),
        ("policy_demand_change", "Policy Demand change"),
        ("q_resolution_robust", "Q early–late resolution"),
        ("asinh_q_noop_margin_update", "Signed Q NOOP-margin update"),
        ("log_root_q_update", "Root-Q update magnitude"),
        ("log_value_surprise", "Value surprise"),
    ]
    effects, summary = _run_level_search_commit_effects(risk, specifications)
    _write_inference_tables(inference_dir, "main_figure3_search_commit", effects, summary)

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 10.0), facecolor="white")
    _plot_effect_summary(
        axes[0, 0], summary, "metric",
        "Adjusted search effects on commit\n(separate theory-defined tests; BH-FDR)",
        odds_ratio=True,
    )
    metric_order = summary["metric"].tolist()
    run_order = list(dict.fromkeys(effects["run_label"].tolist()))
    effect_grid = effects.pivot(
        index="metric", columns="run_label", values="coefficient"
    ).reindex(index=metric_order, columns=run_order)
    bound = float(np.nanquantile(np.abs(effect_grid.to_numpy()), 0.97))
    image_artist = axes[0, 1].imshow(
        effect_grid.to_numpy(), aspect="auto", interpolation="nearest",
        cmap="RdBu_r", vmin=-max(bound, EPS), vmax=max(bound, EPS),
    )
    short_metric_labels = {
        "Policy Demand change": "Policy ΔDemand",
        "Q early–late resolution": "Q resolution",
        "Signed Q NOOP-margin update": "Signed Q margin",
        "Root-Q update magnitude": "Q-update magnitude",
    }
    axes[0, 1].set_yticks(
        np.arange(len(metric_order)),
        [short_metric_labels.get(label, label) for label in metric_order],
    )
    axes[0, 1].set_xticks(
        np.arange(len(run_order)), [f"R{i+1}" for i in range(len(run_order))],
        rotation=45, ha="right",
    )
    axes[0, 1].set_title("Run-level adjusted log-odds coefficients")
    fig.colorbar(image_artist, ax=axes[0, 1], fraction=0.046, pad=0.025,
                 label="Coefficient")

    event = risk["commit"].to_numpy(dtype=int)
    x = risk["state_x"].to_numpy(dtype=float)
    y = risk["state_y"].to_numpy(dtype=float)
    depth_grid, _ = _binned_standardized_group_difference(
        x, y, risk["mean_depth"].to_numpy(dtype=float), event
    )
    q_margin_values = (
        risk["asinh_q_noop_margin_update"].to_numpy(dtype=float)
        if "asinh_q_noop_margin_update" in risk else np.full(len(risk), np.nan)
    )
    margin_grid, _ = _binned_standardized_group_difference(
        x, y, q_margin_values, event
    )
    vmin, vmax = _finite_limits([depth_grid, margin_grid], diverging=True)
    _plot_state_heatmap(
        fig, axes[1, 0], depth_grid,
        "Mean bout terminal depth: commit − continue within root-state bin",
        "Standardized difference", cmap="RdBu_r", vmin=vmin, vmax=vmax,
        contour_xy=(x, y),
    )
    _plot_state_heatmap(
        fig, axes[1, 1], margin_grid,
        "Signed Q NOOP-margin update (asinh): commit − continue",
        "Standardized difference", cmap="RdBu_r", vmin=vmin, vmax=vmax,
        contour_xy=(x, y),
    )
    for ax in axes[1, :]:
        ax.set_xlabel("Normalized root Policy Slack before search")
        ax.set_ylabel("Root Policy Demand before search")
    fig.suptitle(
        "Main Figure 3 · Realized search evidence and human commitment",
        fontsize=15, y=0.992,
    )
    fig.text(
        0.5, 0.958,
        f"Previous-human-NOOP risk set N={len(risk):,}; each search metric is tested separately after "
        "root Slack×Demand, their interaction, NOOP duration, reward, and done; signed Q margin uses asinh; "
        "inference is across runs",
        ha="center", va="top", fontsize=9.1, color="#5B6470",
    )
    fig.subplots_adjust(
        left=0.18, right=0.985, bottom=0.075, top=0.87,
        hspace=0.35, wspace=0.38,
    )
    save_static_figure(fig, output_path)


def plot_main_figure4_q_ambiguity(
    metrics: pd.DataFrame,
    output_path: Path,
    inference_dir: Path,
) -> None:
    data = _q_state_data(metrics)
    x = data["q_x"].to_numpy(dtype=float)
    start = data["q_y_start"].to_numpy(dtype=float)
    final = data["q_y_final"].to_numpy(dtype=float)
    delta = final - start
    delta_grid, delta_count = _binned_mean(x, start, delta)
    vmin, vmax = _finite_limits([delta_grid], diverging=True)
    run_rows = []
    for run_label, run in data.groupby("run_label", sort=False):
        run_delta = run["q_y_final"].to_numpy() - run["q_y_start"].to_numpy()
        run_rows.append(
            {
                "run_label": run_label,
                "n": len(run),
                "mean_final_minus_start": float(np.mean(run_delta)),
                "median_final_minus_start": float(np.median(run_delta)),
                "fraction_increased": float(np.mean(run_delta > 0)),
            }
        )
    run_summary = pd.DataFrame(run_rows)
    mean_values = run_summary["mean_final_minus_start"].to_numpy(dtype=float)
    n_runs = len(mean_values)
    mean_delta = float(np.mean(mean_values))
    sem = float(stats.sem(mean_values))
    critical = float(stats.t.ppf(0.975, n_runs - 1))
    overall = pd.DataFrame(
        [{
            "n_q_valid": len(data),
            "n_runs": n_runs,
            "step_weighted_mean_delta": float(np.mean(delta)),
            "run_mean_delta": mean_delta,
            "run_mean_ci_low": mean_delta - critical * sem,
            "run_mean_ci_high": mean_delta + critical * sem,
            "fraction_increased": float(np.mean(delta > 0)),
            "exact_signflip_p": _exact_signflip_p(mean_values),
        }]
    )
    inference_dir.mkdir(parents=True, exist_ok=True)
    run_summary.to_csv(inference_dir / "main_figure4_q_ambiguity_runs.csv", index=False)
    overall.to_csv(inference_dir / "main_figure4_q_ambiguity_summary.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.4), facecolor="white")
    hex_artist = axes[0, 0].hexbin(
        start, final, gridsize=45, extent=(0, 1, 0, 1), mincnt=1,
        bins="log", cmap="viridis",
    )
    axes[0, 0].plot([0, 1], [0, 1], color="#343A40", lw=1.2, ls="--",
                    label="No change")
    axes[0, 0].set_xlim(0, 1)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_xlabel("Q Demand at search start")
    axes[0, 0].set_ylabel("Q Demand at search end")
    axes[0, 0].set_title("Start-to-final Q Demand")
    axes[0, 0].legend(frameon=False, fontsize=8.5)
    fig.colorbar(hex_artist, ax=axes[0, 0], fraction=0.046, pad=0.025,
                 label="log10 count")
    _plot_state_heatmap(
        fig, axes[0, 1], delta_grid, "Mean Q Demand change",
        "Final − start", cmap="RdBu_r", vmin=vmin, vmax=vmax,
        contour_xy=(x, start),
    )
    axes[0, 1].set_xlabel("Normalized Q Slack at search start")
    axes[0, 1].set_ylabel("Q Demand at search start")

    ax = axes[1, 0]
    _format_conditioned_state_axis(ax)
    centers = (np.arange(STATE_BINS) + 0.5) / STATE_BINS
    grid_x, grid_y = np.meshgrid(centers, centers)
    shown = np.isfinite(delta_grid) & (delta_count >= PRIMARY_MIN_BIN)
    ax.quiver(
        grid_x[shown], grid_y[shown], np.zeros(np.sum(shown)), delta_grid[shown],
        delta_grid[shown], cmap="RdBu_r", angles="xy", scale_units="xy", scale=1.0,
        width=0.006, headwidth=3.5, headlength=4.5,
    )
    _planning_panel_contours(ax, x, start)
    ax.set_xlabel("Normalized Q Slack at search start")
    ax.set_ylabel("Q Demand at search start")
    ax.set_title("Vertical Q-Demand flow field")

    ax = axes[1, 1]
    positions = np.arange(len(run_summary))
    ax.scatter(
        run_summary["mean_final_minus_start"], positions,
        color="#2F6690", s=34, zorder=3,
    )
    ax.axvline(0, color="#343A40", lw=1.0, ls="--")
    ax.axvspan(
        mean_delta - critical * sem, mean_delta + critical * sem,
        color="#9BBEDB", alpha=0.35, zorder=1,
    )
    ax.axvline(mean_delta, color="#B23A48", lw=1.8, zorder=2)
    ax.set_yticks(positions, [f"Run {i+1}" for i in positions])
    ax.set_xlabel("Mean final − start Q Demand")
    ax.set_title(
        "Run-level replication\n"
        f"sign-flip p={float(overall['exact_signflip_p'].iloc[0]):.4g}"
    )
    ax.grid(axis="x", color="#E5E7EB", lw=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    eligible = _primary_state_data(metrics)
    coverage = len(data) / len(eligible)
    fig.suptitle(
        "Main Figure 4 · Search-related change in Q ambiguity",
        fontsize=15, y=0.992,
    )
    fig.text(
        0.5, 0.957,
        f"Valid initial root-Q subset N={len(data):,} ({coverage:.1%} of eligible rows); "
        f"Q Demand increased in {np.mean(delta > 0):.1%}; run-level intervals treat runs, not steps, as replicates",
        ha="center", va="top", fontsize=9.1, color="#5B6470",
    )
    fig.subplots_adjust(
        left=0.09, right=0.985, bottom=0.08, top=0.87,
        hspace=0.31, wspace=0.27,
    )
    save_static_figure(fig, output_path)


# Figures 5-7 contain only positive behavioral bouts (L>=1).  Within this
# already-filtered population, phase 0 is the search window producing the
# first NOOP and phase 1 is the window producing the terminal active commit.
# This phase origin is distinct from the L=0 active-to-active category used
# only as a reference in the Figure 8 two-part length model.
BOUT_PHASE_GRID = np.linspace(0.0, 1.0, 11)
BOUT_DYNAMIC_SPECIFICATIONS = [
    ("state_x", "Policy Slack"),
    ("state_y", "Policy Demand"),
    ("topology_branching_normalized", "Branch diversity"),
    ("mean_depth", "Terminal depth"),
    ("deep_fraction", "Deep allocation"),
    ("revisit_fraction", "Path revisit"),
]
BOUT_PROFILE_SPECIFICATIONS = BOUT_DYNAMIC_SPECIFICATIONS[2:]


def _estimate_cross2_threshold(lengths: Sequence[int]) -> int:
    values = np.asarray(lengths, dtype=int)
    values = values[values > 0]
    if values.size == 0:
        raise ValueError("Cannot estimate Cross2 without positive bout lengths")
    if values.size < 5:
        return max(1, int(np.percentile(values, 95)))
    times = np.arange(1, int(np.max(values)) + 1)
    survival = np.asarray([np.mean(values >= time) for time in times])
    exponential = np.exp(-(1.0 / np.mean(values)) * times)
    above = survival > exponential
    if above[0]:
        below = np.flatnonzero(~above)
        cross1 = int(below[0]) if below.size else len(times) - 1
    else:
        cross1 = 0
    later = np.flatnonzero(above[cross1 + 1 :])
    if later.size:
        return int(times[cross1 + 1 + int(later[0])])
    return max(1, int(np.percentile(values, 95)))


def _load_cross2_threshold(
    bouts: pd.DataFrame,
    cross2_summary: Path,
) -> Tuple[int, str]:
    subject = int(bouts["subject"].iloc[0])
    game = int(bouts["game"].iloc[0])
    if cross2_summary.exists():
        table = pd.read_csv(cross2_summary)
        required = {"subject", "game", "cross2"}
        if required.issubset(table.columns):
            subject_ids = np.asarray([parse_subject(value) for value in table["subject"]])
            game_ids = pd.to_numeric(table["game"], errors="coerce").to_numpy()
            shown = table.loc[(subject_ids == subject) & (game_ids == game)]
            if not shown.empty:
                return int(shown["cross2"].iloc[0]), str(cross2_summary.resolve())
    return (
        _estimate_cross2_threshold(bouts["length_noop_steps"]),
        "within-analysis Kaplan-Meier/exponential fallback",
    )


def build_behavioral_bout_dynamics(
    metrics: pd.DataFrame,
    cross2_summary: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """Build complete human NOOP-to-active bouts from cached real-step rows.

    A behavioral bout contains one or more consecutive current-human NOOP rows
    followed by the contiguous active-action row.  Cross2 length counts only
    the NOOP rows; the terminal active row is retained at normalized phase 1.
    """
    data = _primary_state_data(metrics).copy()
    required = {
        "subject", "session", "block", "game", "run_label", "real_step",
        "real_step_id", "previous_human_action", "human_action",
        "previous_done", "topology_branching_normalized", "mean_depth",
        "deep_fraction", "revisit_fraction", "state_x", "state_y",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Behavioral bout dynamics missing columns: {missing}")
    data = data.loc[
        data["previous_human_action"].between(0, 5)
        & data["human_action"].between(0, 5)
    ].sort_values(["run_label", "real_step"]).reset_index(drop=True)

    bout_rows: List[Dict[str, object]] = []
    step_parts: List[pd.DataFrame] = []
    censored = 0
    done_crossing = 0
    immediate_active_transitions = 0
    for run_label, run in data.groupby("run_label", sort=False):
        g = run.sort_values("real_step").reset_index(drop=True)
        actions = g["human_action"].to_numpy(dtype=int)
        real_steps = g["real_step"].to_numpy(dtype=int)
        index = 0
        bout_number = 0
        while index < len(g):
            if actions[index] != 0:
                if int(g.loc[index, "previous_human_action"]) != 0:
                    immediate_active_transitions += 1
                index += 1
                continue
            start = index
            end = index
            while (
                end + 1 < len(g)
                and actions[end + 1] == 0
                and real_steps[end + 1] == real_steps[end] + 1
            ):
                end += 1
            commit = end + 1
            onset_known = (
                (start > 0 and real_steps[start] == real_steps[start - 1] + 1)
                or (start == 0 and int(g.loc[start, "previous_human_action"]) != 0)
            )
            completed = (
                onset_known
                and commit < len(g)
                and real_steps[commit] == real_steps[end] + 1
                and actions[commit] != 0
            )
            if not completed:
                censored += 1
                index = end + 1
                continue
            segment = g.iloc[start : commit + 1].copy()
            if np.any(segment["previous_done"].fillna(0).to_numpy(dtype=float) > 0.5):
                done_crossing += 1
                index = commit + 1
                continue

            length = int(end - start + 1)
            bout_id = f"{run_label}::bout{bout_number:05d}"
            bout_number += 1
            segment["behavioral_bout_id"] = bout_id
            segment["length_noop_steps"] = length
            # A row is an action-producing decision window: the root at the
            # preceding real step, all intervening search events, and the
            # ensuing human action at the current real step.  Preserve both
            # ends explicitly so bout phase cannot be mistaken for a state
            # sampled after the ensuing action.
            segment["search_root_human_action"] = segment[
                "previous_human_action"
            ].astype(int)
            segment["search_outcome_human_action"] = segment[
                "human_action"
            ].astype(int)
            segment["bout_position"] = np.arange(length + 1, dtype=int)
            segment["bout_phase"] = segment["bout_position"] / float(length)
            segment["is_bout_onset"] = segment["bout_position"].eq(0).astype(int)
            segment["is_bout_commit"] = segment["bout_position"].eq(length).astype(int)
            step_parts.append(segment)
            bout_rows.append(
                {
                    "behavioral_bout_id": bout_id,
                    "subject": int(segment["subject"].iloc[0]),
                    "session": int(segment["session"].iloc[0]),
                    "block": int(segment["block"].iloc[0]),
                    "game": int(segment["game"].iloc[0]),
                    "run_label": str(run_label),
                    "onset_real_step": int(segment["real_step"].iloc[0]),
                    "commit_real_step": int(segment["real_step"].iloc[-1]),
                    "length_noop_steps": length,
                    "commit_action": int(segment["human_action"].iloc[-1]),
                }
            )
            index = commit + 1

    if not bout_rows or not step_parts:
        raise ValueError("No complete NOOP-to-active behavioral bouts were reconstructed")
    bouts = pd.DataFrame(bout_rows)
    steps = pd.concat(step_parts, ignore_index=True)
    cross2, cross2_source = _load_cross2_threshold(bouts, cross2_summary)
    bouts["cross2"] = int(cross2)
    bouts["bout_class"] = np.where(
        bouts["length_noop_steps"].ge(cross2), "Long", "Short"
    )
    class_map = bouts.set_index("behavioral_bout_id")["bout_class"]
    steps["cross2"] = int(cross2)
    steps["bout_class"] = steps["behavioral_bout_id"].map(class_map)
    onset_alignment_errors = int(
        np.sum(
            steps["is_bout_onset"].eq(1)
            & (
                steps["search_outcome_human_action"].ne(0)
                | steps["search_root_human_action"].eq(0)
            )
        )
    )
    commit_alignment_errors = int(
        np.sum(
            steps["is_bout_commit"].eq(1)
            & (
                steps["search_outcome_human_action"].eq(0)
                | steps["search_root_human_action"].ne(0)
            )
        )
    )
    if onset_alignment_errors or commit_alignment_errors:
        raise ValueError(
            "Behavioral-bout search/action alignment failed: "
            f"onset={onset_alignment_errors}, commit={commit_alignment_errors}"
        )
    qa = {
        "n_complete_bouts": int(len(bouts)),
        "n_bout_rows_including_commit": int(len(steps)),
        "n_short_bouts": int(bouts["bout_class"].eq("Short").sum()),
        "n_long_bouts": int(bouts["bout_class"].eq("Long").sum()),
        "cross2": int(cross2),
        "cross2_source": cross2_source,
        "censored_or_boundary_bouts_excluded": int(censored),
        "done_crossing_bouts_excluded": int(done_crossing),
        "immediate_active_transitions_not_bouts": int(immediate_active_transitions),
        "onset_search_action_alignment_errors": onset_alignment_errors,
        "commit_search_action_alignment_errors": commit_alignment_errors,
        "row_definition": (
            "root at preceding real step + intervening search -> ensuing "
            "human action at current real step"
        ),
        "phase_grid": BOUT_PHASE_GRID.tolist(),
    }
    return bouts, steps, qa


def _phase_normalized_bout_profiles(steps: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [column for column, _ in BOUT_DYNAMIC_SPECIFICATIONS]
    interpolated: List[pd.DataFrame] = []
    for bout_id, bout in steps.groupby("behavioral_bout_id", sort=False):
        ordered = bout.sort_values("bout_phase")
        phase = ordered["bout_phase"].to_numpy(dtype=float)
        record: Dict[str, object] = {
            "behavioral_bout_id": str(bout_id),
            "run_label": str(ordered["run_label"].iloc[0]),
            "bout_class": str(ordered["bout_class"].iloc[0]),
            "length_noop_steps": int(ordered["length_noop_steps"].iloc[0]),
        }
        frame = pd.DataFrame({"bout_phase": BOUT_PHASE_GRID})
        for column in metric_columns:
            values = ordered[column].to_numpy(dtype=float)
            finite = np.isfinite(phase) & np.isfinite(values)
            frame[column] = (
                np.interp(BOUT_PHASE_GRID, phase[finite], values[finite])
                if np.sum(finite) >= 2 else np.nan
            )
        for key, value in record.items():
            frame[key] = value
        interpolated.append(frame)
    return pd.concat(interpolated, ignore_index=True)


def _bout_run_scales(steps: pd.DataFrame) -> Dict[Tuple[str, str], float]:
    scales: Dict[Tuple[str, str], float] = {}
    for run_label, run in steps.groupby("run_label", sort=False):
        for column, _ in BOUT_DYNAMIC_SPECIFICATIONS:
            value = float(np.nanstd(run[column].to_numpy(dtype=float)))
            scales[(str(run_label), column)] = value if value > EPS else np.nan
    return scales


def _bout_endpoint_effects(
    steps: pd.DataFrame,
    scales: Mapping[Tuple[str, str], float],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    records: List[Dict[str, object]] = []
    for run_label, run in steps.groupby("run_label", sort=False):
        for column, label in BOUT_DYNAMIC_SPECIFICATIONS:
            deltas: List[float] = []
            scale = float(scales.get((str(run_label), column), np.nan))
            if not np.isfinite(scale) or scale <= EPS:
                continue
            for _, bout in run.groupby("behavioral_bout_id", sort=False):
                ordered = bout.sort_values("bout_phase")
                onset = float(ordered[column].iloc[0])
                commit = float(ordered[column].iloc[-1])
                if np.isfinite(onset + commit):
                    deltas.append((commit - onset) / scale)
            if deltas:
                records.append(
                    {
                        "metric": label,
                        "run_label": str(run_label),
                        "n_bouts": int(len(deltas)),
                        "coefficient": float(np.mean(deltas)),
                    }
                )
    effects = pd.DataFrame(records)
    return effects, _summarize_run_effects(effects, ["metric"])


def _short_long_endpoint_contrast(
    short_effects: pd.DataFrame,
    long_effects: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    left = long_effects.rename(columns={"coefficient": "long_change"})
    right = short_effects.rename(columns={"coefficient": "short_change"})
    joined = left.merge(right, on=["metric", "run_label"], how="inner")
    joined["coefficient"] = joined["long_change"] - joined["short_change"]
    effects = joined[["metric", "run_label", "coefficient"]].copy()
    return effects, _summarize_run_effects(effects, ["metric"])


def _bout_profile_limits(profiles: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    limits: Dict[str, Tuple[float, float]] = {}
    for column, _ in BOUT_PROFILE_SPECIFICATIONS:
        values = profiles[column].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if not values.size:
            limits[column] = (0.0, 1.0)
            continue
        low, high = float(np.min(values)), float(np.max(values))
        padding = max((high - low) * 0.08, 0.01 if high <= 1.0 else 0.2)
        limits[column] = (low - padding, high + padding)
    return limits


def _draw_density_contours(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    linestyle: str,
) -> None:
    finite = np.isfinite(x) & np.isfinite(y)
    if np.sum(finite) < 20:
        return
    hist, y_edges, x_edges = np.histogram2d(
        y[finite], x[finite], bins=STATE_BINS, range=[[0, 1], [0, 1]]
    )
    smooth = gaussian_filter(hist.astype(float), sigma=1.0)
    positive = smooth[smooth > 0]
    if not positive.size:
        return
    levels = np.unique(np.quantile(positive, [0.65, 0.85]))
    if levels.size:
        centers_x = (x_edges[:-1] + x_edges[1:]) / 2
        centers_y = (y_edges[:-1] + y_edges[1:]) / 2
        ax.contour(
            centers_x, centers_y, smooth,
            levels=levels, colors=[color], linewidths=1.2,
            linestyles=linestyle, alpha=0.9,
        )


def _plot_bout_state_arrows(
    fig: plt.Figure,
    ax: plt.Axes,
    steps: pd.DataFrame,
    profiles: pd.DataFrame,
) -> None:
    count = _binned_count(steps["state_x"], steps["state_y"])
    density = np.where(count > 0, np.log10(count + 1), np.nan)
    image = ax.imshow(
        density, origin="lower", extent=(0, 1, 0, 1), aspect="auto",
        cmap="Greys", interpolation="nearest", alpha=0.72,
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.025)
    colorbar.set_label("log10 count", fontsize=8.0, labelpad=3)
    onset = steps.loc[steps["is_bout_onset"].eq(1)]
    commit = steps.loc[steps["is_bout_commit"].eq(1)]
    _draw_density_contours(
        ax, onset["state_x"].to_numpy(float), onset["state_y"].to_numpy(float),
        "#2F6690", "solid",
    )
    _draw_density_contours(
        ax, commit["state_x"].to_numpy(float), commit["state_y"].to_numpy(float),
        "#D17C00", "dashed",
    )
    for _, run in profiles.groupby("run_label", sort=False):
        ordered = run.sort_values("bout_phase")
        ax.plot(
            ordered["state_x"], ordered["state_y"],
            color="#6B7280", alpha=0.26, lw=0.8, zorder=3,
        )
    grand = profiles.groupby("bout_phase", as_index=False)[["state_x", "state_y"]].mean()
    cmap = plt.get_cmap("cividis")
    for index in range(len(grand) - 1):
        start = grand.iloc[index]
        end = grand.iloc[index + 1]
        ax.annotate(
            "", xy=(end.state_x, end.state_y), xytext=(start.state_x, start.state_y),
            arrowprops={
                "arrowstyle": "-|>", "color": cmap(index / max(len(grand) - 2, 1)),
                "lw": 2.0, "mutation_scale": 11, "shrinkA": 0, "shrinkB": 0,
            },
            zorder=5,
        )
    start = grand.iloc[0]
    middle = grand.iloc[len(grand) // 2]
    end = grand.iloc[-1]
    ax.scatter(start.state_x, start.state_y, s=55, color="#2F6690", zorder=6)
    ax.scatter(end.state_x, end.state_y, s=70, marker="*", color="#D17C00", zorder=6)
    annotations = (
        (start, "Search→first NOOP", (5, 5), "left"),
        (middle, "50%", (5, -14), "left"),
        (end, "Search→active commit", (-5, 10), "right"),
    )
    for point, label, offset, horizontal_alignment in annotations:
        ax.annotate(
            label, (point.state_x, point.state_y), xytext=offset,
            textcoords="offset points", fontsize=8.5, color="#1F2933",
            ha=horizontal_alignment,
        )
    ax.axvline(0.5, color="#343A40", ls="--", lw=0.8)
    ax.axhline(0.5, color="#343A40", ls="--", lw=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Policy Slack at the search root")
    ax.set_ylabel("Policy Demand at the search root")
    ax.set_title("Search-root trajectory: first NOOP → active commit")


def _plot_bout_metric_profile(
    ax: plt.Axes,
    profiles: pd.DataFrame,
    column: str,
    title: str,
    limits: Tuple[float, float],
) -> None:
    for _, run in profiles.groupby("run_label", sort=False):
        ordered = run.sort_values("bout_phase")
        ax.plot(
            ordered["bout_phase"], ordered[column],
            color="#8B949E", alpha=0.28, lw=0.8,
        )
    grouped = profiles.groupby("bout_phase")[column]
    mean = grouped.mean()
    sem = grouped.sem()
    count = grouped.count()
    critical = stats.t.ppf(0.975, np.maximum(count.to_numpy() - 1, 1))
    low = mean.to_numpy() - critical * sem.to_numpy()
    high = mean.to_numpy() + critical * sem.to_numpy()
    phase = mean.index.to_numpy(dtype=float)
    ax.fill_between(phase, low, high, color="#9BBEDB", alpha=0.38, linewidth=0)
    ax.plot(phase, mean, color="#2F6690", lw=2.2)
    ax.scatter(phase[[0, -1]], mean.to_numpy()[[0, -1]], color=["#2F6690", "#D17C00"], s=28, zorder=4)
    ax.set_xlim(0, 1)
    ax.set_ylim(*limits)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel(
        "Normalized bout phase (0 = first NOOP; 1 = active commit)"
    )
    ax.set_ylabel("Run-equalized mean")
    ax.set_title(title)
    ax.grid(color="#E5E7EB", lw=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_main_bout_dynamics(
    steps: pd.DataFrame,
    profiles: pd.DataFrame,
    effects: pd.DataFrame,
    summary: pd.DataFrame,
    output_path: Path,
    title: str,
    subtitle_scope: str,
    profile_limits: Mapping[str, Tuple[float, float]],
    effect_limits: Tuple[float, float],
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17.0, 9.2), facecolor="white")
    _plot_bout_state_arrows(fig, axes[0, 0], steps, profiles)
    for ax, (column, label) in zip(
        (axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]),
        BOUT_PROFILE_SPECIFICATIONS,
    ):
        _plot_bout_metric_profile(ax, profiles, column, label, profile_limits[column])
    _plot_effect_summary(
        axes[1, 2], summary, "metric",
        "Active-commit search − first-NOOP search\n(exact sign-flip BH-FDR)",
    )
    axes[1, 2].set_xlabel(
        "Active-commit search − first-NOOP search (within-run SD)"
    )
    axes[1, 2].set_xlim(*effect_limits)
    n_bouts = int(steps["behavioral_bout_id"].nunique())
    n_runs = int(steps["run_label"].nunique())
    fig.suptitle(title, fontsize=15, y=0.992)
    fig.text(
        0.5, 0.957,
        f"{subtitle_scope}; N={n_bouts:,} complete bouts; {n_runs} runs; each row: "
        "preceding root + intervening search → ensuing action; circle = first-NOOP "
        "search, star = active-commit search; thin lines are run means",
        ha="center", va="top", fontsize=9.0, color="#5B6470",
    )
    fig.subplots_adjust(
        left=0.065, right=0.985, bottom=0.075, top=0.875,
        hspace=0.34, wspace=0.38,
    )
    save_static_figure(fig, output_path)


ACTIVE_BOUT_LENGTH_TERMS = (
    "Any NOOP delay (L≥1 vs 0)",
    "Positive-bout log length",
)


def build_active_to_active_bouts(
    metrics: pd.DataFrame,
    cross2_summary: Path,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Build consecutive-active-anchor units, retaining zero-NOOP bouts.

    Each unit starts at the search window producing one active human action
    and ends at the search window producing the next active human action.
    Its length is the number of intervening current-human NOOP actions.  Thus
    L=0 is a genuine active-to-active pair with two distinct search windows.
    """
    data = _primary_state_data(metrics).copy()
    metric_columns = [column for column, _ in BOUT_PROFILE_SPECIFICATIONS]
    required = {
        "subject", "session", "block", "game", "run_label", "real_step",
        "real_step_id", "human_action", "previous_done", "state_x", "state_y",
        *metric_columns,
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Active-to-active bout analysis missing columns: {missing}")
    data = data.loc[data["human_action"].between(0, 5)].sort_values(
        ["run_label", "real_step"]
    )

    parts: List[pd.DataFrame] = []
    candidate_pairs = 0
    gap_excluded = 0
    done_excluded = 0
    for run_label, run in data.groupby("run_label", sort=False):
        g = run.sort_values("real_step").reset_index(drop=True)
        actions = g["human_action"].to_numpy(dtype=int)
        active_positions = np.flatnonzero(actions != 0)
        if active_positions.size < 2:
            continue
        starts = active_positions[:-1]
        ends = active_positions[1:]
        candidate_pairs += int(len(starts))
        real_steps = g["real_step"].to_numpy(dtype=int)
        contiguous = (
            real_steps[ends] - real_steps[starts]
            == ends - starts
        )
        gap_excluded += int(np.sum(~contiguous))
        done = g["previous_done"].fillna(0).to_numpy(dtype=float) > 0.5
        done_prefix = np.concatenate([[0], np.cumsum(done.astype(int))])
        crosses_done = (
            done_prefix[ends + 1] - done_prefix[starts]
        ) > 0
        done_excluded += int(np.sum(contiguous & crosses_done))
        valid = contiguous & ~crosses_done
        starts = starts[valid]
        ends = ends[valid]
        if not len(starts):
            continue

        frame = pd.DataFrame(
            {
                "subject": g.loc[starts, "subject"].to_numpy(dtype=int),
                "session": g.loc[starts, "session"].to_numpy(dtype=int),
                "block": g.loc[starts, "block"].to_numpy(dtype=int),
                "game": g.loc[starts, "game"].to_numpy(dtype=int),
                "run_label": str(run_label),
                "start_real_step": real_steps[starts],
                "commit_real_step": real_steps[ends],
                "length_noop_steps": (ends - starts - 1).astype(int),
                "start_action": actions[starts],
                "commit_action": actions[ends],
                "start_slack": g.loc[starts, "state_x"].to_numpy(dtype=float),
                "start_demand": g.loc[starts, "state_y"].to_numpy(dtype=float),
            }
        )
        frame["active_bout_id"] = [
            f"{run_label}::activepair{index:06d}"
            for index in range(len(frame))
        ]
        for column in metric_columns:
            frame[f"{column}_start"] = g.loc[starts, column].to_numpy(dtype=float)
            frame[f"{column}_commit"] = g.loc[ends, column].to_numpy(dtype=float)
            frame[f"{column}_delta"] = (
                frame[f"{column}_commit"] - frame[f"{column}_start"]
            )
        parts.append(frame)
    if not parts:
        raise ValueError("No valid consecutive-active-anchor bouts were reconstructed")
    bouts = pd.concat(parts, ignore_index=True)
    cross2, cross2_source = _load_cross2_threshold(bouts, cross2_summary)
    bouts["cross2"] = int(cross2)
    bouts["has_noop_delay"] = bouts["length_noop_steps"].gt(0).astype(int)
    bouts["log_positive_length"] = np.where(
        bouts["length_noop_steps"].gt(0),
        np.log(bouts["length_noop_steps"].clip(lower=1)),
        0.0,
    )
    bouts["bout_class"] = np.select(
        [
            bouts["length_noop_steps"].eq(0),
            bouts["length_noop_steps"].lt(cross2),
        ],
        ["Zero", "Short"],
        default="Long",
    )
    qa = {
        "unit_definition": "active -> zero or more NOOPs -> next active",
        "endpoint_definition": (
            "search producing starting active action -> search producing next active action"
        ),
        "n_units": int(len(bouts)),
        "n_zero_length": int(bouts["length_noop_steps"].eq(0).sum()),
        "n_positive_length": int(bouts["length_noop_steps"].gt(0).sum()),
        "n_short_positive": int(
            bouts["length_noop_steps"].between(1, cross2 - 1).sum()
        ),
        "n_long": int(bouts["length_noop_steps"].ge(cross2).sum()),
        "n_runs": int(bouts["run_label"].nunique()),
        "maximum_length": int(bouts["length_noop_steps"].max()),
        "candidate_active_pairs": int(candidate_pairs),
        "gap_pairs_excluded": int(gap_excluded),
        "done_crossing_pairs_excluded": int(done_excluded),
        "cross2": int(cross2),
        "cross2_source": cross2_source,
    }
    return bouts, qa


def _fit_active_bout_length_effects(
    bouts: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records: List[Dict[str, object]] = []
    curve_records: List[Dict[str, object]] = []
    specifications = list(BOUT_PROFILE_SPECIFICATIONS)
    maximum_length = int(bouts["length_noop_steps"].max())
    positive_grid = np.unique(
        np.rint(np.geomspace(1, max(maximum_length, 1), 80)).astype(int)
    )
    # L=0 remains in the fitted two-part model as the separate active-to-active
    # reference category.  The displayed continuous-length curves begin at
    # L=1 because positive NOOP duration is a distinct regime.
    length_grid = positive_grid
    for run_label, run in bouts.groupby("run_label", sort=False):
        any_delay = run["has_noop_delay"].to_numpy(dtype=float)
        log_length = run["log_positive_length"].to_numpy(dtype=float)
        slack = _zscore_finite(run["start_slack"].to_numpy(dtype=float))
        demand = _zscore_finite(run["start_demand"].to_numpy(dtype=float))
        controls = [
            slack, demand, slack ** 2, demand ** 2, slack * demand,
        ]
        for action in range(2, 6):
            controls.append(run["start_action"].eq(action).to_numpy(dtype=float))
            controls.append(run["commit_action"].eq(action).to_numpy(dtype=float))
        control_matrix = np.column_stack(controls)
        keep_controls = np.nanstd(control_matrix, axis=0) > EPS
        control_matrix = control_matrix[:, keep_controls]
        base_design = np.column_stack(
            [np.ones(len(run)), any_delay, log_length, control_matrix]
        )
        for column, label in specifications:
            start = run[f"{column}_start"].to_numpy(dtype=float)
            commit = run[f"{column}_commit"].to_numpy(dtype=float)
            scale = float(np.nanstd(np.concatenate([start, commit])))
            if not np.isfinite(scale) or scale <= EPS:
                continue
            outcome = (commit - start) / scale
            finite = np.isfinite(outcome) & np.all(np.isfinite(base_design), axis=1)
            if np.sum(finite) < 100:
                continue
            design = base_design[finite]
            if np.linalg.matrix_rank(design) < 3:
                continue
            coefficients = np.linalg.lstsq(design, outcome[finite], rcond=None)[0]
            for term, coefficient in zip(ACTIVE_BOUT_LENGTH_TERMS, coefficients[1:3]):
                records.append(
                    {
                        "outcome": label,
                        "term": term,
                        "run_label": str(run_label),
                        "n_bouts": int(np.sum(finite)),
                        "coefficient": float(coefficient),
                    }
                )
            run_minimum_length = int(
                run.loc[finite, "bout_length_actions"].min()
            )
            run_maximum_length = int(
                run.loc[finite, "bout_length_actions"].max()
            )
            supported_grid = length_grid[
                (length_grid >= run_minimum_length)
                & (length_grid <= run_maximum_length)
            ]
            for length in supported_grid:
                contribution = float(
                    coefficients[1] + coefficients[2] * np.log(length)
                )
                curve_records.append(
                    {
                        "outcome": label,
                        "run_label": str(run_label),
                        "length_noop_steps": int(length),
                        "log1p_length": float(np.log1p(length)),
                        "adjusted_length_contribution": contribution,
                    }
                )
    effects = pd.DataFrame(records)
    if effects.empty:
        raise ValueError("No run-level continuous bout-length effects were estimable")
    summary = _summarize_run_effects(effects, ["outcome", "term"])
    outcome_labels = {
        "Branch diversity": "Branch",
        "Terminal depth": "Depth",
        "Deep allocation": "Deep",
        "Path revisit": "Revisit",
    }
    term_labels = {
        ACTIVE_BOUT_LENGTH_TERMS[0]: r"$I(L>0)$",
        ACTIVE_BOUT_LENGTH_TERMS[1]: r"$\log L\mid L>0$",
    }
    summary["effect_label"] = (
        summary["outcome"].map(outcome_labels)
        + " · "
        + summary["term"].map(term_labels)
    )
    return effects, summary, pd.DataFrame(curve_records)


def _plot_active_bout_length_distribution(
    ax: plt.Axes,
    bouts: pd.DataFrame,
    cross2: int,
) -> None:
    all_length = bouts["length_noop_steps"].to_numpy(dtype=int)
    if cross2 < 1:
        raise ValueError(f"2nd cross point must be positive, got {cross2}")
    if np.any(all_length < 0):
        raise ValueError("Intervening NOOP count cannot be negative")
    length = all_length[all_length >= 1]

    # Preserve the interpretable base ranges while inserting the subject/game-
    # specific 2nd cross point as an additional boundary.  This keeps every
    # bin mutually exclusive even when Cross2 is larger than a fixed boundary
    # (for example, Cross2=110 in sub003 Pong).
    boundaries = sorted(
        set((1, 2, 3, 5, 8, 12, 18, 36, 60, int(cross2)))
    )
    bins = []
    membership = np.zeros(len(length), dtype=np.int8)
    for index, lower in enumerate(boundaries):
        upper = boundaries[index + 1] if index + 1 < len(boundaries) else None
        if upper is None:
            mask = length >= lower
            label = f"{lower}+"
        else:
            mask = (length >= lower) & (length < upper)
            label = str(lower) if upper == lower + 1 else f"{lower}–{upper - 1}"
        membership += mask.astype(np.int8)
        if lower < cross2:
            color = "#4E79A7"
        else:
            color = "#D17C00"
        bins.append((label, mask, color))

    if len(length) and not np.all(membership == 1):
        raise RuntimeError(
            "Dynamic bout-length bins must partition every observation exactly once"
        )
    labels = [item[0] for item in bins]
    counts = np.asarray([np.sum(item[1]) for item in bins], dtype=int)
    colors = [item[2] for item in bins]
    positions = np.arange(len(labels))
    ax.bar(positions, counts, color=colors, width=0.82, edgecolor="white", lw=0.5)
    ax.set_yscale("log")
    ax.set_xticks(positions, labels, rotation=35, ha="right")
    ax.set_ylabel("Number of active-to-active units (log scale)")
    ax.set_xlabel("Intervening NOOP count")
    ax.set_title("Positive bout-length distribution (L≥1)")
    ax.grid(axis="y", color="#E5E7EB", lw=0.6)
    for index, count in enumerate(counts):
        if count:
            ax.text(index, count * 1.12, f"{count:,}", ha="center", va="bottom", fontsize=7.5)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="s", color="none", markerfacecolor="#4E79A7", label=f"1≤L<{cross2}", markersize=8),
            Line2D([0], [0], marker="s", color="none", markerfacecolor="#D17C00", label=f"L≥{cross2}", markersize=8),
        ],
        frameon=False, fontsize=8, loc="upper right",
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _plot_active_bout_adjusted_curve(
    ax: plt.Axes,
    curves: pd.DataFrame,
    outcome: str,
    cross2: int,
    y_limits: Tuple[float, float],
) -> None:
    shown = curves.loc[curves["outcome"].eq(outcome)].copy()
    for _, run in shown.groupby("run_label", sort=False):
        ordered = run.sort_values("length_noop_steps")
        ax.plot(
            ordered["log1p_length"], ordered["adjusted_length_contribution"],
            color="#8B949E", alpha=0.28, lw=0.8,
        )
    grouped = shown.groupby("length_noop_steps", sort=True)
    mean = grouped["adjusted_length_contribution"].mean()
    sem = grouped["adjusted_length_contribution"].sem()
    count = grouped["adjusted_length_contribution"].count()
    critical = stats.t.ppf(0.975, np.maximum(count.to_numpy() - 1, 1))
    lengths = mean.index.to_numpy(dtype=int)
    x = np.log1p(lengths)
    low = mean.to_numpy() - critical * sem.to_numpy()
    high = mean.to_numpy() + critical * sem.to_numpy()
    ax.fill_between(x, low, high, color="#9BBEDB", alpha=0.38, linewidth=0)
    ax.plot(x, mean, color="#2F6690", lw=2.2)
    ax.scatter(x[0], mean.iloc[0], color="#2F6690", s=28, zorder=4)
    ax.axhline(0, color="#343A40", ls="--", lw=0.8)
    ax.axvline(np.log1p(cross2), color="#D17C00", ls=":", lw=1.2)
    maximum = int(lengths.max())
    base_ticks = [
        value for value in (1, 2, 4, 8, 16, 50, 100, 200)
        if value <= maximum
    ]
    if cross2 <= maximum:
        # Always show the subject/game-specific boundary.  Remove a nearby
        # generic log-axis tick when both labels would overlap (e.g. 100 and
        # Cross2=110 in sub003 Pong).
        cross2_x = float(np.log1p(cross2))
        base_ticks = [
            value
            for value in base_ticks
            if abs(float(np.log1p(value)) - cross2_x) >= 0.13
        ]
        base_ticks.append(int(cross2))
    ticks = sorted(set(base_ticks))
    ax.set_xticks(np.log1p(ticks), [str(value) for value in ticks])
    ax.set_xlim(float(np.log1p(1)), float(np.log1p(maximum)))
    ax.set_ylim(*y_limits)
    ax.set_xlabel("Intervening NOOP count (log-spaced)")
    ax.set_ylabel("Adjusted length contribution (within-run SD)")
    ax.set_title(outcome)
    ax.grid(color="#E5E7EB", lw=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_main_figure8_continuous_bout_length(
    bouts: pd.DataFrame,
    effects: pd.DataFrame,
    summary: pd.DataFrame,
    curves: pd.DataFrame,
    output_path: Path,
    cross2: int,
) -> None:
    specifications = list(BOUT_PROFILE_SPECIFICATIONS)
    curve_values = curves["adjusted_length_contribution"].to_numpy(dtype=float)
    finite_curve = curve_values[np.isfinite(curve_values)]
    low = min(0.0, float(np.min(finite_curve)))
    high = max(0.0, float(np.max(finite_curve)))
    padding = max((high - low) * 0.08, 0.05)
    y_limits = (low - padding, high + padding)
    fig, axes = plt.subplots(2, 3, figsize=(17.0, 9.2), facecolor="white")
    _plot_active_bout_length_distribution(axes[0, 0], bouts, cross2)
    for ax, (_, label) in zip(
        (axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]), specifications
    ):
        _plot_active_bout_adjusted_curve(ax, curves, label, cross2, y_limits)
    _plot_effect_summary(
        axes[1, 2], summary, "effect_label",
        "Controlled length effects across runs\n(exact sign-flip, BH-FDR)",
    )
    axes[1, 2].set_xlabel("Across-run mean coefficient (within-run SD)")
    n_units = int(len(bouts))
    n_runs = int(bouts["run_label"].nunique())
    fig.suptitle(
        "Main Figure 8 · Continuous bout length and active-to-active search change",
        fontsize=15, y=0.992,
    )
    fig.text(
        0.5, 0.957,
        f"Each unit: active → NOOP^L → active; N={n_units:,}; {n_runs} runs; "
        "L=0 is fitted as a separate reference but count/curve axes begin at "
        "L=1; two-part length = I(L>0) + log L within L>0; controls = starting "
        "Slack, Demand (quadratic + interaction), starting action, and commit action",
        ha="center", va="top", fontsize=9.0, color="#5B6470",
    )
    fig.subplots_adjust(
        left=0.065, right=0.985, bottom=0.085, top=0.875,
        hspace=0.37, wspace=0.35,
    )
    save_static_figure(fig, output_path)


def _action_transition_class(origin_action: int, destination_action: int) -> str:
    """Return the prespecified transition family for a completed action bout."""
    origin = int(origin_action)
    destination = int(destination_action)
    if origin == destination:
        return "stay"
    if origin == 0 and destination != 0:
        return "noop_to_active"
    if origin != 0 and destination == 0:
        return "active_to_noop"
    return "active_to_active"


def build_same_action_transition_risk(
    metrics: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """Reconstruct comparable same-action risk episodes.

    Every retained row has the fixed temporal interpretation

        previous human action == bout action
            -> intervening search
            -> current human action (stay or transition).

    The search window that *enters* the bout is deliberately excluded because
    its root belongs to a different previous-action condition.  A one-action
    bout therefore contributes one transition-risk row but no within-bout
    trajectory.  Run/chunk gaps and terminal episodes are right censored.
    """
    primary = _primary_state_data(metrics)
    required = {
        "subject",
        "session",
        "block",
        "game",
        "run_label",
        "real_step",
        "real_step_id",
        "previous_human_action",
        "human_action",
        "previous_reward",
        "previous_done",
        "noop_run_length_prev",
        "state_x",
        "state_y",
        "final_cur_policy_slack",
        "final_cur_policy_demand",
        "final_minus_root_policy_slack",
        "final_minus_root_policy_demand",
        "delta_policy_slack_raw",
        "delta_policy_demand_raw",
        "delta_policy_slack_normalized",
        "delta_policy_demand_normalized",
        "topology_branching_normalized",
        "mean_depth",
        "deep_fraction",
        "revisit_fraction",
    }
    missing = sorted(required.difference(primary.columns))
    if missing:
        raise ValueError(
            "Generalized action-bout analysis missing columns: "
            + ", ".join(missing)
        )
    # Keep only fields consumed by the generalized analysis before reconstructing
    # episodes.  The analysis cache contains many wide vector columns, and copying
    # one such frame per bout can otherwise multiply peak memory use.
    retained_columns = [
        "subject",
        "session",
        "block",
        "game",
        "run_label",
        "real_step",
        "real_step_id",
        "previous_human_action",
        "human_action",
        "previous_reward",
        "previous_done",
        "noop_run_length_prev",
        "state_x",
        "state_y",
        "final_cur_policy_slack",
        "final_cur_policy_demand",
        "final_minus_root_policy_slack",
        "final_minus_root_policy_demand",
        "delta_policy_slack_raw",
        "delta_policy_demand_raw",
        "delta_policy_slack_normalized",
        "delta_policy_demand_normalized",
        "topology_branching_normalized",
        "mean_depth",
        "deep_fraction",
        "revisit_fraction",
    ]
    eligible = (
        primary["previous_human_action"].between(0, 5)
        & primary["human_action"].between(0, 5)
    )
    data = (
        primary.loc[eligible, retained_columns]
        .sort_values(["run_label", "real_step"], kind="stable")
        .reset_index(drop=True)
    )
    if data.empty:
        raise ValueError("No eligible rows for generalized action bouts")

    n_rows = len(data)
    action_bout_id = np.full(n_rows, -1, dtype=np.int64)
    bout_action = np.full(n_rows, -1, dtype=np.int8)
    destination_action = np.full(n_rows, -1, dtype=np.int8)
    transition_class = np.full(n_rows, "unassigned", dtype=object)
    root_position = np.full(n_rows, -1, dtype=np.int32)
    duration_at_root = np.full(n_rows, np.nan, dtype=np.float32)
    is_transition = np.zeros(n_rows, dtype=np.int8)
    is_complete_bout = np.zeros(n_rows, dtype=np.int8)
    onset_known_row = np.zeros(n_rows, dtype=np.int8)
    steps_to_transition = np.full(n_rows, np.nan, dtype=np.float32)

    episode_records: List[Dict[str, object]] = []
    continuity_errors = 0
    left_censored = 0
    right_censored = 0
    completed = 0
    episode_counter = 0

    for run_label, run in data.groupby("run_label", sort=False):
        g = run.sort_values("real_step", kind="stable")
        global_positions = g.index.to_numpy(dtype=np.int64)
        previous = g["previous_human_action"].to_numpy(dtype=int)
        current = g["human_action"].to_numpy(dtype=int)
        real_steps = g["real_step"].to_numpy(dtype=int)
        if len(g) > 1:
            consecutive = real_steps[1:] == real_steps[:-1] + 1
            continuity_errors += int(
                np.sum(consecutive & (previous[1:] != current[:-1]))
            )

        index = 0
        while index < len(g):
            origin = int(previous[index])
            start = index
            end = index
            destination = -1
            is_complete = False
            while end < len(g):
                if int(previous[end]) != origin:
                    break
                if end > start and real_steps[end] != real_steps[end - 1] + 1:
                    break
                if int(current[end]) != origin:
                    destination = int(current[end])
                    is_complete = True
                    break
                next_index = end + 1
                if (
                    next_index >= len(g)
                    or real_steps[next_index] != real_steps[end] + 1
                    or int(previous[next_index]) != origin
                ):
                    break
                end = next_index

            if not is_complete:
                # Include every observed stay-risk row up to the discontinuity
                # and mark the episode as right censored.
                while (
                    end + 1 < len(g)
                    and real_steps[end + 1] == real_steps[end] + 1
                    and int(previous[end + 1]) == origin
                    and int(current[end]) == origin
                ):
                    end += 1
                right_censored += 1
            else:
                completed += 1

            onset_known = bool(
                start > 0
                and real_steps[start] == real_steps[start - 1] + 1
                and int(current[start - 1]) == origin
                and int(previous[start - 1]) != origin
            )
            if not onset_known:
                left_censored += 1

            local_positions = np.arange(start, end + 1, dtype=np.int64)
            row_positions = global_positions[local_positions]
            episode_id = int(episode_counter)
            episode_counter += 1
            class_name = (
                _action_transition_class(origin, destination)
                if is_complete
                else "censored"
            )
            segment_length = int(len(row_positions))
            action_bout_id[row_positions] = episode_id
            bout_action[row_positions] = origin
            destination_action[row_positions] = destination
            transition_class[row_positions] = class_name
            root_position[row_positions] = np.arange(
                segment_length, dtype=np.int32
            )
            if onset_known:
                duration_at_root[row_positions] = np.arange(
                    1, segment_length + 1, dtype=np.float32
                )
            is_transition[row_positions] = (
                current[local_positions] != origin
            ).astype(np.int8)
            is_complete_bout[row_positions] = int(is_complete)
            onset_known_row[row_positions] = int(onset_known)
            if is_complete:
                steps_to_transition[row_positions] = (
                    np.arange(segment_length, dtype=np.float32)
                    - float(segment_length - 1)
                )
            first_row = g.iloc[start]
            last_row = g.iloc[end]
            episode_records.append(
                {
                    "action_bout_id": episode_id,
                    "subject": int(first_row["subject"]),
                    "session": int(first_row["session"]),
                    "block": int(first_row["block"]),
                    "game": int(first_row["game"]),
                    "run_label": str(run_label),
                    "bout_action": origin,
                    "destination_action": destination,
                    "transition_class": class_name,
                    "n_risk_rows": segment_length,
                    "onset_known": int(onset_known),
                    "completed": int(is_complete),
                    "first_risk_real_step": int(first_row["real_step"]),
                    "last_risk_real_step": int(last_row["real_step"]),
                }
            )
            index = end + 1

    unassigned = int(np.sum(action_bout_id < 0))
    if unassigned:
        raise AssertionError(
            f"Generalized action-bout reconstruction left {unassigned} rows unassigned"
        )
    # Materialize the retained risk table once, after all episode metadata have
    # been assigned.  This preserves the row-level output while keeping peak RAM
    # approximately linear in the number of real steps.
    risk = data.copy()
    risk["action_bout_id"] = action_bout_id
    risk["bout_action"] = bout_action
    risk["destination_action"] = destination_action
    risk["transition_class"] = transition_class
    risk["root_position"] = root_position
    risk["duration_at_root"] = duration_at_root
    risk["is_transition"] = is_transition
    risk["is_complete_bout"] = is_complete_bout
    risk["onset_known"] = onset_known_row
    risk["steps_to_transition"] = steps_to_transition
    bouts = pd.DataFrame(episode_records)
    transition_rows = risk.loc[risk["is_transition"].eq(1)]
    if len(transition_rows) != int(bouts["completed"].sum()):
        raise AssertionError(
            "Each completed same-action bout must contain exactly one transition row"
        )
    comparable_errors = int(
        np.sum(
            risk["previous_human_action"].astype(int)
            != risk["bout_action"].astype(int)
        )
    )
    if comparable_errors:
        raise ValueError(
            f"Generalized action-bout root condition failed for {comparable_errors} rows"
        )
    qa = {
        "row_definition": (
            "root after the bout action + intervening search -> stay in or exit "
            "the same-action bout"
        ),
        "n_eligible_risk_rows": int(len(risk)),
        "n_action_bouts": int(len(bouts)),
        "n_completed_bouts": int(completed),
        "n_left_censored_bouts": int(left_censored),
        "n_right_censored_bouts": int(right_censored),
        "n_one_row_completed_bouts": int(
            np.sum(bouts["completed"].eq(1) & bouts["n_risk_rows"].eq(1))
        ),
        "action_continuity_errors": int(continuity_errors),
        "root_condition_errors": comparable_errors,
        "transition_class_counts": {
            str(key): int(value)
            for key, value in bouts.loc[bouts["completed"].eq(1), "transition_class"]
            .value_counts()
            .items()
        },
        "selection_note": (
            "A1 retains every eligible stay/exit risk row. Start-aligned A3 "
            "retains every observed root from completed, known-onset bouts "
            "without interpolation or fixed-length truncation."
        ),
    }
    return bouts, risk, qa


def _plot_action_transition_matrix(
    risk: pd.DataFrame,
    output_path: Path,
) -> None:
    observed_actions = sorted(
        set(risk["previous_human_action"].astype(int))
        | set(risk["human_action"].astype(int))
    )
    counts = pd.crosstab(
        risk["previous_human_action"].astype(int),
        risk["human_action"].astype(int),
    ).reindex(
        index=observed_actions,
        columns=observed_actions,
        fill_value=0,
    )
    probabilities = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.3), facecolor="white")
    specifications = (
        (counts.to_numpy(dtype=float), "Risk-row counts", "Count", "Blues"),
        (
            probabilities.to_numpy(dtype=float),
            "Conditional action probability",
            "Probability",
            "cividis",
        ),
    )
    labels = [action_label(action) for action in observed_actions]
    for ax, (values, title, colorbar_label, cmap) in zip(axes, specifications):
        image = ax.imshow(values, cmap=cmap, aspect="equal")
        ax.set_xticks(range(len(observed_actions)), labels, rotation=35, ha="right")
        ax.set_yticks(range(len(observed_actions)), labels)
        ax.set_xlabel("Current action after search")
        ax.set_ylabel("Previous / bout action at root")
        ax.set_title(title)
        threshold = float(np.nanmedian(values[np.isfinite(values)]))
        for row in range(len(observed_actions)):
            for column in range(len(observed_actions)):
                shown = (
                    f"{int(values[row, column]):,}"
                    if colorbar_label == "Count"
                    else f"{values[row, column]:.2f}"
                )
                ax.text(
                    column,
                    row,
                    shown,
                    ha="center",
                    va="center",
                    fontsize=7.2,
                    color=(
                        "white"
                        if np.isfinite(values[row, column])
                        and values[row, column] > threshold
                        else "#20252B"
                    ),
                )
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
        colorbar.set_label(colorbar_label)
    fig.suptitle(
        "Generalized same-action risk and transition matrix",
        fontsize=15,
        y=0.995,
    )
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.16, top=0.90, wspace=0.28)
    save_static_figure(fig, output_path)


def _start_aligned_transition_profiles(
    risk: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize all observed roots after aligning completed bouts at onset.

    ``root_position == 0`` is the first comparable root whose previous action
    equals the action maintained within the bout.  No phase interpolation,
    terminal alignment, or fixed-duration truncation is applied.  Later
    positions therefore describe the subset of bouts that remained in the
    same action for at least that many observed roots.
    """
    complete = risk.loc[
        risk["is_complete_bout"].eq(1)
        & risk["onset_known"].eq(1)
        & risk["transition_class"].isin(ACTION_TRANSITION_CLASS_LABELS)
    ].copy()
    if complete.empty:
        return pd.DataFrame(), pd.DataFrame()
    keys = [
        "run_label",
        "transition_class",
        "bout_action",
        "destination_action",
        "action_bout_id",
        "root_position",
    ]
    bout_profiles = (
        complete.groupby(keys, as_index=False)[["state_x", "state_y"]]
        .mean()
    )
    run_profiles = (
        bout_profiles.groupby(
            [
                "run_label",
                "transition_class",
                "bout_action",
                "destination_action",
                "root_position",
            ],
            as_index=False,
        )
        .agg(
            state_x=("state_x", "mean"),
            state_y=("state_y", "mean"),
            n_bouts=("action_bout_id", "nunique"),
        )
    )
    summary = (
        run_profiles.groupby(
            [
                "transition_class",
                "bout_action",
                "destination_action",
                "root_position",
            ],
            as_index=False,
        )
        .agg(
            state_x=("state_x", "mean"),
            state_x_sem=("state_x", "sem"),
            state_y=("state_y", "mean"),
            state_y_sem=("state_y", "sem"),
            n_runs=("run_label", "nunique"),
            n_bouts=("n_bouts", "sum"),
        )
        .sort_values(
            [
                "transition_class",
                "bout_action",
                "destination_action",
                "root_position",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    return run_profiles, summary


def _action_switch_endpoint_tables(
    risk: pd.DataFrame,
    x_column: str = "state_x",
    y_column: str = "state_y",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Construct paired start-to-switch displacement tables.

    Every completed, known-onset bout contributes its first comparable root
    and the final root that produced the ensuing action switch.  This is a
    paired descriptive summary; it does not interpolate the intervening roots
    or compare predictive models.
    """
    complete = risk.loc[
        risk["is_complete_bout"].eq(1)
        & risk["onset_known"].eq(1)
        & risk["transition_class"].isin(ACTION_TRANSITION_CLASS_LABELS)
    ].sort_values(
        ["action_bout_id", "root_position"],
        kind="stable",
    )
    if complete.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if x_column not in complete.columns or y_column not in complete.columns:
        raise ValueError(
            f"Missing policy-coordinate columns: {x_column}, {y_column}"
        )
    group_columns = [
        "run_label",
        "transition_class",
        "bout_action",
        "destination_action",
        "action_bout_id",
    ]
    bout_endpoints = (
        complete.groupby(group_columns, as_index=False)
        .agg(
            start_state_x=(x_column, "first"),
            start_state_y=(y_column, "first"),
            switch_state_x=(x_column, "last"),
            switch_state_y=(y_column, "last"),
            n_observed_roots=("root_position", "size"),
        )
    )
    bout_endpoints["delta_state_x"] = (
        bout_endpoints["switch_state_x"]
        - bout_endpoints["start_state_x"]
    )
    bout_endpoints["delta_state_y"] = (
        bout_endpoints["switch_state_y"]
        - bout_endpoints["start_state_y"]
    )
    bout_endpoints["displacement"] = np.hypot(
        bout_endpoints["delta_state_x"],
        bout_endpoints["delta_state_y"],
    )
    pair_columns = [
        "run_label",
        "transition_class",
        "bout_action",
        "destination_action",
    ]
    run_endpoints = (
        bout_endpoints.groupby(pair_columns, as_index=False)
        .agg(
            start_state_x=("start_state_x", "mean"),
            start_state_y=("start_state_y", "mean"),
            switch_state_x=("switch_state_x", "mean"),
            switch_state_y=("switch_state_y", "mean"),
            n_bouts=("action_bout_id", "nunique"),
        )
    )
    run_endpoints["delta_state_x"] = (
        run_endpoints["switch_state_x"]
        - run_endpoints["start_state_x"]
    )
    run_endpoints["delta_state_y"] = (
        run_endpoints["switch_state_y"]
        - run_endpoints["start_state_y"]
    )
    run_endpoints["displacement"] = np.hypot(
        run_endpoints["delta_state_x"],
        run_endpoints["delta_state_y"],
    )
    summary = (
        run_endpoints.groupby(
            [
                "transition_class",
                "bout_action",
                "destination_action",
            ],
            as_index=False,
        )
        .agg(
            start_state_x=("start_state_x", "mean"),
            start_state_x_sem=("start_state_x", "sem"),
            start_state_y=("start_state_y", "mean"),
            start_state_y_sem=("start_state_y", "sem"),
            switch_state_x=("switch_state_x", "mean"),
            switch_state_x_sem=("switch_state_x", "sem"),
            switch_state_y=("switch_state_y", "mean"),
            switch_state_y_sem=("switch_state_y", "sem"),
            delta_state_x=("delta_state_x", "mean"),
            delta_state_x_sem=("delta_state_x", "sem"),
            delta_state_y=("delta_state_y", "mean"),
            delta_state_y_sem=("delta_state_y", "sem"),
            mean_run_displacement=("displacement", "mean"),
            n_runs=("run_label", "nunique"),
            n_bouts=("n_bouts", "sum"),
        )
    )
    summary["net_displacement"] = np.hypot(
        summary["delta_state_x"],
        summary["delta_state_y"],
    )
    return bout_endpoints, run_endpoints, summary


def _draw_action_switch_displacement_panel(
    ax: plt.Axes,
    global_density: np.ndarray,
    pair_bouts: pd.DataFrame,
    pair_runs: pd.DataFrame,
    pair_summary: pd.Series,
) -> plt.AxesImage:
    image = ax.imshow(
        global_density,
        origin="lower",
        extent=(0, 1, 0, 1),
        aspect="equal",
        cmap="Greys",
        interpolation="nearest",
        alpha=0.68,
        vmin=0,
        vmax=float(np.nanmax(global_density)),
    )
    _draw_density_contours(
        ax,
        pair_bouts["start_state_x"].to_numpy(dtype=float),
        pair_bouts["start_state_y"].to_numpy(dtype=float),
        "#2F6690",
        "solid",
    )
    _draw_density_contours(
        ax,
        pair_bouts["switch_state_x"].to_numpy(dtype=float),
        pair_bouts["switch_state_y"].to_numpy(dtype=float),
        "#D17C00",
        "dashed",
    )
    ax.quiver(
        pair_runs["start_state_x"].to_numpy(dtype=float),
        pair_runs["start_state_y"].to_numpy(dtype=float),
        pair_runs["delta_state_x"].to_numpy(dtype=float),
        pair_runs["delta_state_y"].to_numpy(dtype=float),
        angles="xy",
        scale_units="xy",
        scale=1,
        color="#6B7280",
        alpha=0.34,
        width=0.005,
        headwidth=4.0,
        headlength=5.0,
        headaxislength=4.3,
        zorder=4,
    )
    start_x = float(pair_summary["start_state_x"])
    start_y = float(pair_summary["start_state_y"])
    switch_x = float(pair_summary["switch_state_x"])
    switch_y = float(pair_summary["switch_state_y"])
    ax.annotate(
        "",
        xy=(switch_x, switch_y),
        xytext=(start_x, start_y),
        arrowprops={
            "arrowstyle": "-|>",
            "color": "#1F2937",
            "lw": 2.3,
            "mutation_scale": 12,
            "shrinkA": 1.0,
            "shrinkB": 1.0,
        },
        zorder=6,
    )
    ax.scatter(
        start_x,
        start_y,
        s=48,
        color="#2F6690",
        edgecolor="white",
        linewidth=0.65,
        zorder=7,
    )
    ax.scatter(
        switch_x,
        switch_y,
        s=76,
        marker="*",
        color="#D17C00",
        edgecolor="white",
        linewidth=0.55,
        zorder=7,
    )
    ax.text(
        0.035,
        0.035,
        (
            f"|Δ|={float(pair_summary['net_displacement']):.3f}\n"
            f"runs={int(pair_summary['n_runs'])}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.2,
        color="#1F2937",
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": "#D1D5DB",
            "alpha": 0.86,
        },
        zorder=8,
    )
    ax.axvline(0.5, color="#343A40", ls="--", lw=0.65, alpha=0.78)
    ax.axhline(0.5, color="#343A40", ls="--", lw=0.65, alpha=0.78)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(labelsize=7.2)
    return image


def _plot_action_switch_displacement_facets(
    risk: pd.DataFrame,
    bout_endpoints: pd.DataFrame,
    run_endpoints: pd.DataFrame,
    summary: pd.DataFrame,
    transition_class: str,
    output_path: Path,
) -> None:
    selected_summary = (
        summary.loc[summary["transition_class"].eq(transition_class)]
        .sort_values(["bout_action", "destination_action"], kind="stable")
        .reset_index(drop=True)
    )
    if selected_summary.empty:
        log(f"Switch displacement figure skipped: {transition_class}")
        return
    n_panels = len(selected_summary)
    maximum_columns = 4 if transition_class == "active_to_active" else 5
    n_columns = min(maximum_columns, n_panels)
    n_rows = int(np.ceil(n_panels / n_columns))
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(3.05 * n_columns + 0.9, 2.85 * n_rows + 1.75),
        facecolor="white",
        squeeze=False,
    )
    global_count = _binned_count(risk["state_x"], risk["state_y"])
    global_density = np.where(
        global_count > 0,
        np.log10(global_count + 1),
        0.0,
    )
    used_axes: List[plt.Axes] = []
    image: Optional[plt.AxesImage] = None
    for panel_index, (_, row) in enumerate(selected_summary.iterrows()):
        axis_row, axis_column = divmod(panel_index, n_columns)
        ax = axes[axis_row, axis_column]
        used_axes.append(ax)
        origin = int(row["bout_action"])
        destination = int(row["destination_action"])
        pair_selector = (
            bout_endpoints["transition_class"].eq(transition_class)
            & bout_endpoints["bout_action"].eq(origin)
            & bout_endpoints["destination_action"].eq(destination)
        )
        run_selector = (
            run_endpoints["transition_class"].eq(transition_class)
            & run_endpoints["bout_action"].eq(origin)
            & run_endpoints["destination_action"].eq(destination)
        )
        pair_bouts = bout_endpoints.loc[pair_selector]
        pair_runs = run_endpoints.loc[run_selector]
        image = _draw_action_switch_displacement_panel(
            ax,
            global_density,
            pair_bouts,
            pair_runs,
            row,
        )
        ax.set_title(
            (
                f"{action_label(origin)} → {action_label(destination)}"
                f"\nN={int(row['n_bouts']):,}"
            ),
            fontsize=8.7,
        )
        if axis_row == n_rows - 1:
            ax.set_xlabel("Policy Slack", fontsize=8.0)
        else:
            ax.tick_params(labelbottom=False)
        if axis_column == 0:
            ax.set_ylabel("Policy Demand", fontsize=8.0)
        else:
            ax.tick_params(labelleft=False)
    for panel_index in range(n_panels, n_rows * n_columns):
        axis_row, axis_column = divmod(panel_index, n_columns)
        axes[axis_row, axis_column].axis("off")
    handles = [
        Line2D(
            [0],
            [0],
            color="#2F6690",
            marker="o",
            lw=1.2,
            label="Bout onset distribution / mean",
        ),
        Line2D(
            [0],
            [0],
            color="#D17C00",
            marker="*",
            ls="--",
            lw=1.2,
            label="Switch-root distribution / mean",
        ),
        Line2D(
            [0],
            [0],
            color="#6B7280",
            lw=1.2,
            label="Run-level displacement",
        ),
        Line2D(
            [0],
            [0],
            color="#1F2937",
            lw=2.3,
            label="Run-equalized mean displacement",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.89),
        ncol=4,
        frameon=False,
        fontsize=7.6,
    )
    titles = {
        "noop_to_active": "NOOP → active switches by destination action",
        "active_to_noop": "Active → NOOP switches by origin action",
        "active_to_active": (
            "Active → different-active switches by action pair"
        ),
    }
    fig.suptitle(
        f"Policy Slack–Demand displacement · {titles[transition_class]}",
        fontsize=14.5,
        y=0.992,
    )
    fig.text(
        0.5,
        0.953,
        (
            "Paired first comparable root → switch-producing root; all completed, "
            "known-onset bouts; gray background is the common risk-root state "
            "space; |Δ| is Euclidean net displacement"
        ),
        ha="center",
        va="top",
        fontsize=8.5,
        color="#5B6470",
    )
    fig.subplots_adjust(
        left=0.07,
        right=0.91,
        bottom=0.075,
        top=0.80,
        hspace=0.35,
        wspace=0.20,
    )
    if image is not None:
        colorbar_axis = fig.add_axes([0.93, 0.12, 0.012, 0.62])
        colorbar = fig.colorbar(image, cax=colorbar_axis)
        colorbar.set_label("Global log10 risk-root count", fontsize=7.8)
        colorbar.ax.tick_params(labelsize=7)
    save_static_figure(fig, output_path)


def _draw_ranged_density_contours(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    linestyle: str,
    x_limits: Tuple[float, float],
    y_limits: Tuple[float, float],
) -> None:
    finite = np.isfinite(x) & np.isfinite(y)
    if np.sum(finite) < 20:
        return
    hist, y_edges, x_edges = np.histogram2d(
        y[finite],
        x[finite],
        bins=STATE_BINS,
        range=[list(y_limits), list(x_limits)],
    )
    smooth = gaussian_filter(hist.astype(float), sigma=1.0)
    positive = smooth[smooth > 0]
    if not positive.size:
        return
    levels = np.unique(np.quantile(positive, [0.65, 0.85]))
    if not levels.size:
        return
    centers_x = (x_edges[:-1] + x_edges[1:]) / 2
    centers_y = (y_edges[:-1] + y_edges[1:]) / 2
    ax.contour(
        centers_x,
        centers_y,
        smooth,
        levels=levels,
        colors=[color],
        linewidths=1.2,
        linestyles=linestyle,
        alpha=0.9,
    )


def _coordinate_density(
    x: np.ndarray,
    y: np.ndarray,
    x_limits: Tuple[float, float],
    y_limits: Tuple[float, float],
) -> np.ndarray:
    finite = np.isfinite(x) & np.isfinite(y)
    hist, _, _ = np.histogram2d(
        y[finite],
        x[finite],
        bins=STATE_BINS,
        range=[list(y_limits), list(x_limits)],
    )
    return np.where(hist > 0, np.log10(hist + 1), 0.0)


def _draw_coordinate_switch_panel(
    ax: plt.Axes,
    global_density: np.ndarray,
    pair_bouts: pd.DataFrame,
    pair_runs: pd.DataFrame,
    pair_summary: pd.Series,
    x_limits: Tuple[float, float],
    y_limits: Tuple[float, float],
    reference: Tuple[float, float],
) -> plt.AxesImage:
    image = ax.imshow(
        global_density,
        origin="lower",
        extent=(*x_limits, *y_limits),
        aspect="equal",
        cmap="Greys",
        interpolation="nearest",
        alpha=0.68,
        vmin=0,
        vmax=float(np.nanmax(global_density)),
    )
    _draw_ranged_density_contours(
        ax,
        pair_bouts["start_state_x"].to_numpy(dtype=float),
        pair_bouts["start_state_y"].to_numpy(dtype=float),
        "#2F6690",
        "solid",
        x_limits,
        y_limits,
    )
    _draw_ranged_density_contours(
        ax,
        pair_bouts["switch_state_x"].to_numpy(dtype=float),
        pair_bouts["switch_state_y"].to_numpy(dtype=float),
        "#D17C00",
        "dashed",
        x_limits,
        y_limits,
    )
    ax.quiver(
        pair_runs["start_state_x"].to_numpy(dtype=float),
        pair_runs["start_state_y"].to_numpy(dtype=float),
        pair_runs["delta_state_x"].to_numpy(dtype=float),
        pair_runs["delta_state_y"].to_numpy(dtype=float),
        angles="xy",
        scale_units="xy",
        scale=1,
        color="#6B7280",
        alpha=0.34,
        width=0.005,
        headwidth=4.0,
        headlength=5.0,
        headaxislength=4.3,
        zorder=4,
    )
    start_x = float(pair_summary["start_state_x"])
    start_y = float(pair_summary["start_state_y"])
    switch_x = float(pair_summary["switch_state_x"])
    switch_y = float(pair_summary["switch_state_y"])
    ax.annotate(
        "",
        xy=(switch_x, switch_y),
        xytext=(start_x, start_y),
        arrowprops={
            "arrowstyle": "-|>",
            "color": "#1F2937",
            "lw": 2.3,
            "mutation_scale": 12,
            "shrinkA": 1.0,
            "shrinkB": 1.0,
        },
        zorder=6,
    )
    ax.scatter(
        start_x,
        start_y,
        s=48,
        color="#2F6690",
        edgecolor="white",
        linewidth=0.65,
        zorder=7,
    )
    ax.scatter(
        switch_x,
        switch_y,
        s=76,
        marker="*",
        color="#D17C00",
        edgecolor="white",
        linewidth=0.55,
        zorder=7,
    )
    ax.text(
        0.035,
        0.035,
        (
            f"|Δ|={float(pair_summary['net_displacement']):.3f}\n"
            f"runs={int(pair_summary['n_runs'])}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.2,
        color="#1F2937",
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": "#D1D5DB",
            "alpha": 0.86,
        },
        zorder=8,
    )
    ax.axvline(reference[0], color="#343A40", ls="--", lw=0.65, alpha=0.78)
    ax.axhline(reference[1], color="#343A40", ls="--", lw=0.65, alpha=0.78)
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=7.2)
    return image


def _plot_coordinate_switch_displacement_facets(
    risk: pd.DataFrame,
    bout_endpoints: pd.DataFrame,
    run_endpoints: pd.DataFrame,
    summary: pd.DataFrame,
    transition_class: str,
    output_path: Path,
    coordinate_title: str,
    x_label: str,
    y_label: str,
    x_column: str,
    y_column: str,
    x_limits: Tuple[float, float],
    y_limits: Tuple[float, float],
    reference: Tuple[float, float],
) -> None:
    selected_summary = (
        summary.loc[summary["transition_class"].eq(transition_class)]
        .sort_values(["bout_action", "destination_action"], kind="stable")
        .reset_index(drop=True)
    )
    if selected_summary.empty:
        log(
            f"Coordinate switch figure skipped: "
            f"{coordinate_title}, {transition_class}"
        )
        return
    n_panels = len(selected_summary)
    maximum_columns = 4 if transition_class == "active_to_active" else 5
    n_columns = min(maximum_columns, n_panels)
    n_rows = int(np.ceil(n_panels / n_columns))
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(3.05 * n_columns + 0.9, 2.85 * n_rows + 1.75),
        facecolor="white",
        squeeze=False,
    )
    global_density = _coordinate_density(
        risk[x_column].to_numpy(dtype=float),
        risk[y_column].to_numpy(dtype=float),
        x_limits,
        y_limits,
    )
    image: Optional[plt.AxesImage] = None
    for panel_index, (_, row) in enumerate(selected_summary.iterrows()):
        axis_row, axis_column = divmod(panel_index, n_columns)
        ax = axes[axis_row, axis_column]
        origin = int(row["bout_action"])
        destination = int(row["destination_action"])
        pair_selector = (
            bout_endpoints["transition_class"].eq(transition_class)
            & bout_endpoints["bout_action"].eq(origin)
            & bout_endpoints["destination_action"].eq(destination)
        )
        run_selector = (
            run_endpoints["transition_class"].eq(transition_class)
            & run_endpoints["bout_action"].eq(origin)
            & run_endpoints["destination_action"].eq(destination)
        )
        pair_bouts = bout_endpoints.loc[pair_selector]
        pair_runs = run_endpoints.loc[run_selector]
        image = _draw_coordinate_switch_panel(
            ax,
            global_density,
            pair_bouts,
            pair_runs,
            row,
            x_limits,
            y_limits,
            reference,
        )
        ax.set_title(
            (
                f"{action_label(origin)} → {action_label(destination)}"
                f"\nN={int(row['n_bouts']):,}"
            ),
            fontsize=8.7,
        )
        if axis_row == n_rows - 1:
            ax.set_xlabel(x_label, fontsize=8.0)
        else:
            ax.tick_params(labelbottom=False)
        if axis_column == 0:
            ax.set_ylabel(y_label, fontsize=8.0)
        else:
            ax.tick_params(labelleft=False)
    for panel_index in range(n_panels, n_rows * n_columns):
        axis_row, axis_column = divmod(panel_index, n_columns)
        axes[axis_row, axis_column].axis("off")
    titles = {
        "noop_to_active": "NOOP → active switches by destination",
        "active_to_noop": "Active → NOOP switches by origin",
        "active_to_active": "Active → different-active switches by pair",
    }
    fig.suptitle(
        f"{coordinate_title} · {titles[transition_class]}",
        fontsize=14.5,
        y=0.992,
    )
    fig.text(
        0.5,
        0.953,
        (
            "Same completed, known-onset bouts as A4; paired first comparable "
            "root → switch-producing root; thin vectors are run means and the "
            "thick vector is the equally weighted across-run mean"
        ),
        ha="center",
        va="top",
        fontsize=8.5,
        color="#5B6470",
    )
    handles = [
        Line2D(
            [0], [0], color="#2F6690", marker="o", lw=1.2,
            label="Bout onset distribution / mean",
        ),
        Line2D(
            [0], [0], color="#D17C00", marker="*", ls="--", lw=1.2,
            label="Switch-root distribution / mean",
        ),
        Line2D(
            [0], [0], color="#6B7280", lw=1.2,
            label="Run-level displacement",
        ),
        Line2D(
            [0], [0], color="#1F2937", lw=2.3,
            label="Run-equalized mean displacement",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.89),
        ncol=4,
        frameon=False,
        fontsize=7.6,
    )
    fig.subplots_adjust(
        left=0.07,
        right=0.91,
        bottom=0.075,
        top=0.80,
        hspace=0.35,
        wspace=0.20,
    )
    if image is not None:
        colorbar_axis = fig.add_axes([0.93, 0.12, 0.012, 0.62])
        colorbar = fig.colorbar(image, cax=colorbar_axis)
        colorbar.set_label("Global log10 risk-root count", fontsize=7.8)
        colorbar.ax.tick_params(labelsize=7)
    save_static_figure(fig, output_path)


def _plot_policy_coordinate_overview(
    metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    data = _primary_state_data(metrics).dropna(
        subset=[
            "final_cur_policy_slack",
            "final_cur_policy_demand",
            "final_minus_root_policy_slack",
            "final_minus_root_policy_demand",
        ]
    )
    data = data.loc[data["human_action"].between(0, 5)].copy()
    specifications = [
        (
            "state_x",
            "state_y",
            "Real-step root policy",
            "Root Policy Slack",
            "Root Policy Demand",
            (0.0, 1.0),
            (0.0, 1.0),
            (0.5, 0.5),
        ),
        (
            "final_cur_policy_slack",
            "final_cur_policy_demand",
            "Final status-2 cur policy",
            "Final cur Policy Slack",
            "Final cur Policy Demand",
            (0.0, 1.0),
            (0.0, 1.0),
            (0.5, 0.5),
        ),
        (
            "final_minus_root_policy_slack",
            "final_minus_root_policy_demand",
            "Final imagined node − root\n(different states)",
            "Δ Policy Slack",
            "Δ Policy Demand",
            (-1.0, 1.0),
            (-1.0, 1.0),
            (0.0, 0.0),
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.8), facecolor="white")
    rng = np.random.default_rng(2026)
    sample_size = min(60_000, len(data))
    sample = data.iloc[
        rng.choice(len(data), size=sample_size, replace=False)
    ].copy()
    for ax, specification in zip(axes, specifications):
        (
            x_column,
            y_column,
            title,
            x_label,
            y_label,
            x_limits,
            y_limits,
            reference,
        ) = specification
        for action in sorted(sample["human_action"].astype(int).unique()):
            rows = sample.loc[sample["human_action"].eq(action)]
            ax.scatter(
                rows[x_column],
                rows[y_column],
                s=2.0,
                alpha=0.11,
                color=ACTION_COLORS.get(action, "#6B7280"),
                edgecolors="none",
                rasterized=True,
            )
            all_action = data.loc[data["human_action"].eq(action)]
            _draw_ranged_density_contours(
                ax,
                all_action[x_column].to_numpy(dtype=float),
                all_action[y_column].to_numpy(dtype=float),
                ACTION_COLORS.get(action, "#6B7280"),
                "solid",
                x_limits,
                y_limits,
            )
        ax.axvline(reference[0], color="#343A40", ls="--", lw=0.75)
        ax.axhline(reference[1], color="#343A40", ls="--", lw=0.75)
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(color="#ECEFF1", lw=0.45)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=ACTION_COLORS.get(action, "#6B7280"),
            markeredgecolor="none",
            label=f"{action_label(action)}",
            markersize=6,
        )
        for action in sorted(data["human_action"].astype(int).unique())
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=len(handles),
        frameon=False,
    )
    fig.suptitle(
        "Policy-coordinate diagnostic across one real-step search window",
        fontsize=15,
        y=0.995,
    )
    fig.text(
        0.5,
        0.955,
        (
            f"Eligible real-step windows N={len(data):,}; points show a fixed "
            f"random sample of {sample_size:,}; contours use all rows and are "
            "colored by the ensuing human action"
        ),
        ha="center",
        va="top",
        fontsize=8.8,
        color="#5B6470",
    )
    fig.subplots_adjust(
        left=0.055,
        right=0.985,
        bottom=0.105,
        top=0.82,
        wspace=0.27,
    )
    save_static_figure(fig, output_path)


def _set_elapsed_root_axis(ax: plt.Axes, maximum: int) -> None:
    maximum = max(int(maximum), 1)
    if maximum > 20:
        ax.set_xscale("symlog", linthresh=5, linscale=1.0, base=10)
    ticks = [
        value
        for value in (0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1_000, 3_000)
        if value <= maximum
    ]
    if maximum not in ticks:
        if (
            maximum > 20
            and ticks
            and ticks[-1] >= 20
            and maximum / ticks[-1] < 1.5
        ):
            ticks[-1] = maximum
        else:
            ticks.append(maximum)
    ticks = sorted(set(ticks))
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [f"{value:,}" for value in ticks],
        rotation=32,
        ha="right",
        rotation_mode="anchor",
        fontsize=7.2,
    )
    ax.set_xlim(0, maximum)


def _plot_start_aligned_single_action_facets(
    summary: pd.DataFrame,
    transition_class: str,
    action_role: str,
    output_path: Path,
) -> None:
    selected = summary.loc[
        summary["transition_class"].eq(transition_class)
    ].copy()
    action_column = (
        "destination_action" if action_role == "destination" else "bout_action"
    )
    actions = sorted(selected[action_column].astype(int).unique())
    if not actions:
        log(f"Start-aligned {transition_class} figure skipped: no actions")
        return
    n_columns = len(actions)
    fig = plt.figure(
        figsize=(max(7.0, 3.2 * n_columns), 7.1),
        facecolor="white",
    )
    grid = fig.add_gridspec(
        2,
        n_columns,
        height_ratios=(3.2, 1.0),
        hspace=0.10,
        wspace=0.28,
    )
    state_axes: List[plt.Axes] = []
    count_axes: List[plt.Axes] = []
    for column, action in enumerate(actions):
        ax = fig.add_subplot(grid[0, column])
        count_ax = fig.add_subplot(grid[1, column], sharex=ax)
        state_axes.append(ax)
        count_axes.append(count_ax)
        panel = selected.loc[selected[action_column].eq(action)].copy()
        panel = (
            panel.groupby("root_position", as_index=False)
            .agg(
                state_x=("state_x", "mean"),
                state_x_sem=("state_x_sem", "mean"),
                state_y=("state_y", "mean"),
                state_y_sem=("state_y_sem", "mean"),
                n_runs=("n_runs", "max"),
                n_bouts=("n_bouts", "sum"),
            )
            .sort_values("root_position")
        )
        x = panel["root_position"].to_numpy(dtype=float)
        slack = panel["state_x"].to_numpy(dtype=float)
        demand = panel["state_y"].to_numpy(dtype=float)
        slack_sem = panel["state_x_sem"].fillna(0).to_numpy(dtype=float)
        demand_sem = panel["state_y_sem"].fillna(0).to_numpy(dtype=float)
        ax.plot(
            x,
            slack,
            color="#4E79A7",
            lw=1.8,
            label=geometry_display("trajectory_x_label"),
        )
        ax.fill_between(
            x,
            slack - 1.96 * slack_sem,
            slack + 1.96 * slack_sem,
            color="#4E79A7",
            alpha=0.14,
            linewidth=0,
        )
        ax.plot(
            x,
            demand,
            color="#F28E2B",
            lw=1.8,
            ls="--",
            label=geometry_display("trajectory_y_label"),
        )
        ax.fill_between(
            x,
            demand - 1.96 * demand_sem,
            demand + 1.96 * demand_sem,
            color="#F28E2B",
            alpha=0.12,
            linewidth=0,
        )
        ax.scatter(
            [x[0]],
            [slack[0]],
            s=32,
            color="#4E79A7",
            edgecolor="white",
            linewidth=0.5,
            zorder=4,
        )
        ax.scatter(
            [x[0]],
            [demand[0]],
            s=32,
            color="#F28E2B",
            edgecolor="white",
            linewidth=0.5,
            zorder=4,
        )
        ax.set_ylim(0, 1)
        ax.grid(color="#E5E7EB", lw=0.55)
        ax.set_title(
            (
                f"{action_label(action)}\n"
                f"N={int(panel.iloc[0]['n_bouts']):,}; "
                f"max={int(panel['root_position'].max()) + 1:,} roots"
            ),
            fontsize=9.5,
        )
        if column == 0:
            ax.set_ylabel(
                "Run-equalized "
                + geometry_display("trajectory_coordinate_label")
            )
        else:
            ax.tick_params(labelleft=False)
        plt.setp(ax.get_xticklabels(), visible=False)

        count_ax.plot(
            x,
            panel["n_bouts"],
            color="#6B7280",
            lw=1.4,
        )
        use_log_count_axis = float(panel["n_bouts"].max()) > 10
        count_baseline = (
            np.ones_like(x, dtype=float)
            if use_log_count_axis
            else np.zeros_like(x, dtype=float)
        )
        count_ax.fill_between(
            x,
            count_baseline,
            panel["n_bouts"].to_numpy(dtype=float),
            color="#D1D5DB",
            alpha=0.75,
            linewidth=0,
        )
        if use_log_count_axis:
            count_ax.set_yscale("log")
        count_ax.grid(color="#ECEFF1", lw=0.45)
        if column == 0:
            count_ax.set_ylabel("Bouts at risk", fontsize=8)
        else:
            count_ax.tick_params(labelleft=False)
        _set_elapsed_root_axis(count_ax, int(panel["root_position"].max()))

    handles = [
        Line2D(
            [0],
            [0],
            color="#4E79A7",
            lw=2,
            label=geometry_display("trajectory_x_label"),
        ),
        Line2D(
            [0],
            [0],
            color="#F28E2B",
            lw=2,
            ls="--",
            label=geometry_display("trajectory_y_label"),
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=2,
        frameon=False,
    )
    title = (
        "NOOP → active transitions by destination action"
        if transition_class == "noop_to_active"
        else "Active → NOOP transitions by origin action"
    )
    fig.suptitle(
        f"{geometry_display('trajectory_title_prefix')} · {title}",
        fontsize=15,
        y=0.995,
    )
    fig.supxlabel(
        "Observed roots since comparable bout onset",
        x=0.52,
        y=0.025,
        fontsize=9.4,
    )
    fig.subplots_adjust(
        left=0.065,
        right=0.985,
        bottom=0.115,
        top=0.79,
    )
    save_static_figure(fig, output_path)


def _plot_start_aligned_active_switch_facets(
    summary: pd.DataFrame,
    metric: str,
    output_path: Path,
) -> None:
    selected = summary.loc[
        summary["transition_class"].eq("active_to_active")
    ].copy()
    origins = sorted(selected["bout_action"].astype(int).unique())
    if not origins:
        log("Start-aligned active-switch figure skipped: no active switches")
        return
    metric_label = (
        geometry_display("trajectory_x_label")
        if metric == "state_x"
        else geometry_display("trajectory_y_label")
    )
    n_columns = len(origins)
    fig = plt.figure(
        figsize=(max(7.0, 3.2 * n_columns), 7.1),
        facecolor="white",
    )
    grid = fig.add_gridspec(
        2,
        n_columns,
        height_ratios=(3.2, 1.0),
        hspace=0.10,
        wspace=0.28,
    )
    for column, origin in enumerate(origins):
        ax = fig.add_subplot(grid[0, column])
        count_ax = fig.add_subplot(grid[1, column], sharex=ax)
        origin_data = selected.loc[selected["bout_action"].eq(origin)]
        maximum = int(origin_data["root_position"].max())
        for destination in sorted(
            origin_data["destination_action"].astype(int).unique()
        ):
            panel = (
                origin_data.loc[
                    origin_data["destination_action"].eq(destination)
                ]
                .sort_values("root_position")
                .copy()
            )
            x = panel["root_position"].to_numpy(dtype=float)
            y = panel[metric].to_numpy(dtype=float)
            sem = panel[f"{metric}_sem"].fillna(0).to_numpy(dtype=float)
            color = ACTION_COLORS.get(destination, "#6B7280")
            label = (
                f"→ {action_label(destination)} "
                f"(N={int(panel.iloc[0]['n_bouts']):,})"
            )
            ax.plot(x, y, color=color, lw=1.7, label=label)
            ax.fill_between(
                x,
                y - 1.96 * sem,
                y + 1.96 * sem,
                color=color,
                alpha=0.10,
                linewidth=0,
            )
            ax.scatter(
                [x[0]],
                [y[0]],
                s=24,
                color=color,
                edgecolor="white",
                linewidth=0.45,
                zorder=4,
            )
            count_ax.plot(
                x,
                panel["n_bouts"],
                color=color,
                lw=1.2,
            )
        ax.set_ylim(0, 1)
        ax.grid(color="#E5E7EB", lw=0.55)
        ax.set_title(
            f"Origin = {action_label(origin)}",
            fontsize=9.8,
        )
        if column == 0:
            ax.set_ylabel(f"Run-equalized {metric_label}")
        else:
            ax.tick_params(labelleft=False)
        plt.setp(ax.get_xticklabels(), visible=False)
        if float(origin_data["n_bouts"].max()) > 10:
            count_ax.set_yscale("log")
        count_ax.grid(color="#ECEFF1", lw=0.45)
        if column == 0:
            count_ax.set_ylabel("Bouts at risk", fontsize=8)
        else:
            count_ax.tick_params(labelleft=False)
        _set_elapsed_root_axis(count_ax, maximum)
        ax.legend(
            frameon=False,
            fontsize=6.5,
            loc="best",
            handlelength=1.8,
        )

    fig.suptitle(
        (
            "Start-aligned active → different-active trajectories · "
            f"{metric_label}"
        ),
        fontsize=15,
        y=0.995,
    )
    fig.supxlabel(
        "Observed roots since comparable bout onset",
        x=0.52,
        y=0.025,
        fontsize=9.4,
    )
    fig.subplots_adjust(
        left=0.065,
        right=0.985,
        bottom=0.115,
        top=0.90,
    )
    save_static_figure(fig, output_path)


def _clear_action_transition_namespace(*directories: Path) -> None:
    """Delete only deterministic artifacts in dedicated transition folders."""
    for directory in directories:
        resolved = directory.resolve()
        if (
            resolved.name != "action_transitions"
            or resolved.parent.name not in {"figures", "results"}
        ):
            raise ValueError(
                f"Refusing to clear non-transition output directory: {resolved}"
            )
        if not resolved.exists():
            continue
        for child in resolved.iterdir():
            if child.is_file() or child.is_symlink():
                child.unlink()
            else:
                raise ValueError(
                    "Unexpected nested directory in transition namespace: "
                    f"{child}"
                )
        log(f"Cleared previous action-transition artifacts: {resolved}")


def generate_generalized_action_transition_figures(
    metrics: pd.DataFrame,
    figure_dir: Path,
    result_dir: Path,
) -> Dict[str, object]:
    """Refresh the dedicated action-transition analysis namespace."""
    action_figure_dir = figure_dir / "action_transitions"
    action_result_dir = result_dir / "action_transitions"
    _clear_action_transition_namespace(
        action_figure_dir,
        action_result_dir,
    )
    action_figure_dir.mkdir(parents=True, exist_ok=True)
    action_result_dir.mkdir(parents=True, exist_ok=True)
    bouts, risk, qa = build_same_action_transition_risk(metrics)
    run_profiles, start_summary = _start_aligned_transition_profiles(risk)
    qa.update(
        {
            "a3_alignment": (
                "root_position=0 is the first comparable root after the action "
                "bout began; all subsequently observed roots are retained at "
                "their integer elapsed position without interpolation, terminal "
                "alignment, or fixed-length truncation"
            ),
            "a3_sample_scope": (
                "completed, known-onset bouts are required to identify the "
                "transition class and active-action origin/destination; all "
                "eligible rows, including censored bouts, remain in A1"
            ),
            "a3_observed_roots": int(
                risk.loc[
                    risk["is_complete_bout"].eq(1)
                    & risk["onset_known"].eq(1)
                ].shape[0]
            ),
            "a3_complete_known_onset_bouts": int(
                bouts.loc[
                    bouts["completed"].eq(1) & bouts["onset_known"].eq(1)
                ].shape[0]
            ),
            "a3_outputs": [
                "main_figureA3a_noop_to_active_start_aligned",
                "main_figureA3b_active_to_noop_start_aligned",
                "main_figureA3c_active_switch_slack_start_aligned",
                "main_figureA3d_active_switch_demand_start_aligned",
            ],
        }
    )
    bouts.to_csv(
        action_result_dir / "same_action_bouts.csv.gz",
        index=False,
        compression="gzip",
    )
    risk.to_csv(
        action_result_dir / "same_action_transition_risk_rows.csv.gz",
        index=False,
        compression="gzip",
    )
    run_profiles.to_csv(
        action_result_dir / "start_aligned_run_profiles.csv.gz",
        index=False,
        compression="gzip",
    )
    start_summary.to_csv(
        action_result_dir / "start_aligned_summary.csv.gz",
        index=False,
        compression="gzip",
    )
    (action_result_dir / "same_action_bout_qa.json").write_text(
        json.dumps(qa, indent=2), encoding="utf-8"
    )
    _plot_action_transition_matrix(
        risk,
        action_figure_dir / "main_figureA1_action_transition_matrix.png",
    )
    _plot_start_aligned_single_action_facets(
        start_summary,
        "noop_to_active",
        "destination",
        action_figure_dir
        / "main_figureA3a_noop_to_active_start_aligned.png",
    )
    _plot_start_aligned_single_action_facets(
        start_summary,
        "active_to_noop",
        "origin",
        action_figure_dir
        / "main_figureA3b_active_to_noop_start_aligned.png",
    )
    _plot_start_aligned_active_switch_facets(
        start_summary,
        "state_x",
        action_figure_dir
        / "main_figureA3c_active_switch_slack_start_aligned.png",
    )
    _plot_start_aligned_active_switch_facets(
        start_summary,
        "state_y",
        action_figure_dir
        / "main_figureA3d_active_switch_demand_start_aligned.png",
    )
    return qa


INTEGRATED_TRANSITION_METRICS = (
    (
        "topology_branching_normalized",
        "Root-branch allocation diversity",
        "Branch diversity",
    ),
    ("mean_depth", "Mean bout terminal depth", "Terminal depth"),
    ("deep_fraction", "Deep-node allocation fraction", "Deep allocation"),
    ("revisit_fraction", "Observed path-revisit fraction", "Path revisit"),
)

INTEGRATED_COORDINATES = (
    (
        "actor_start",
        "state_x",
        "state_y",
        "Search-start Actor Policy Slack–Demand",
        "Normalized Start Actor Policy Slack",
        "Start Actor Policy Demand",
    ),
    (
        "actor_change",
        "delta_policy_slack_normalized",
        "delta_policy_demand_normalized",
        "Normalized Terminal-minus-Start Actor Policy Change",
        "Normalized Actor Policy Slack Change",
        "Normalized Actor Policy Demand Change",
    ),
)

TRANSITION_FIGURE_LABELS = {
    "noop_to_active": "NOOP → Active Action",
    "active_to_noop": "Active Action → NOOP",
    "active_to_active": "Active Action → Different Active Action",
}

COMPACT_TRANSITION_LABELS = {
    "noop_to_active": "Delay-to-Active",
    "active_to_noop": "Active-to-Delay",
    "active_to_active": "Active-to-Active",
}

COMPACT_COORDINATE_LABELS = {
    "actor_start": "Search-Start Actor-Policy State",
    "actor_change": "Normalized Actor-Policy Margins",
}


def _compact_transition_figure_title(
    records: Sequence[Mapping[str, object]],
    transition_class: str,
    metric_label: str,
) -> str:
    coordinate_name = str(records[0]["coordinate_name"])
    coordinate_label = COMPACT_COORDINATE_LABELS.get(
        coordinate_name,
        str(records[0]["coordinate_title"]),
    )
    return (
        f"{metric_label} for "
        f"{COMPACT_TRANSITION_LABELS[transition_class]} — "
        f"{coordinate_label}"
    )


def _complete_known_transition_rows(
    risk: pd.DataFrame,
    transition_class: str,
) -> pd.DataFrame:
    """Return all comparable roots in completed, known-onset bouts.

    The row that enters a same-action bout is not in ``risk`` by construction:
    every retained row has ``previous_human_action == bout_action``.  For
    example, 3,0,0,0,0,1 contributes the three 0->0 search windows and the
    terminal 0->1 search window, but not the entering 3->0 window.
    """
    selected = risk.loc[
        risk["is_complete_bout"].eq(1)
        & risk["onset_known"].eq(1)
        & risk["transition_class"].eq(transition_class)
    ].sort_values(
        ["run_label", "action_bout_id", "root_position"],
        kind="stable",
    )
    return selected.copy()


def _class_coordinate_endpoint_tables(
    selected: pd.DataFrame,
    x_column: str,
    y_column: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    group_columns = [
        "run_label",
        "transition_class",
        "bout_action",
        "destination_action",
        "action_bout_id",
    ]
    endpoints = (
        selected.groupby(group_columns, as_index=False)
        .agg(
            start_x=(x_column, "first"),
            start_y=(y_column, "first"),
            switch_x=(x_column, "last"),
            switch_y=(y_column, "last"),
            n_observed_roots=("root_position", "size"),
        )
    )
    endpoints["delta_x"] = endpoints["switch_x"] - endpoints["start_x"]
    endpoints["delta_y"] = endpoints["switch_y"] - endpoints["start_y"]
    run_endpoints = (
        endpoints.groupby("run_label", as_index=False)
        .agg(
            start_x=("start_x", "mean"),
            start_y=("start_y", "mean"),
            switch_x=("switch_x", "mean"),
            switch_y=("switch_y", "mean"),
            n_bouts=("action_bout_id", "nunique"),
        )
    )
    run_endpoints["delta_x"] = (
        run_endpoints["switch_x"] - run_endpoints["start_x"]
    )
    run_endpoints["delta_y"] = (
        run_endpoints["switch_y"] - run_endpoints["start_y"]
    )
    summary = {
        "start_x": float(run_endpoints["start_x"].mean()),
        "start_y": float(run_endpoints["start_y"].mean()),
        "switch_x": float(run_endpoints["switch_x"].mean()),
        "switch_y": float(run_endpoints["switch_y"].mean()),
        "delta_x": float(run_endpoints["delta_x"].mean()),
        "delta_y": float(run_endpoints["delta_y"].mean()),
        "n_runs": int(run_endpoints["run_label"].nunique()),
        "n_bouts": int(len(endpoints)),
    }
    return endpoints, run_endpoints, summary


def _class_search_endpoint_tables(
    selected: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    group_columns = [
        "run_label",
        "transition_class",
        "bout_action",
        "destination_action",
        "action_bout_id",
    ]
    records: List[pd.DataFrame] = []
    run_records: List[Dict[str, object]] = []
    for column, label, short_label in INTEGRATED_TRANSITION_METRICS:
        endpoints = (
            selected.groupby(group_columns, as_index=False)
            .agg(
                start=(column, "first"),
                switch=(column, "last"),
                n_observed_roots=("root_position", "size"),
            )
        )
        endpoints["delta"] = endpoints["switch"] - endpoints["start"]
        endpoints["outcome"] = label
        endpoints["short_outcome"] = short_label
        records.append(endpoints)
        for run_label, run_endpoints in endpoints.groupby(
            "run_label", sort=False
        ):
            run_rows = selected.loc[selected["run_label"].eq(run_label), column]
            scale = float(np.nanstd(run_rows.to_numpy(dtype=float)))
            valid = run_endpoints[["start", "switch", "delta"]].replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            if valid.empty:
                continue
            standardized_delta = (
                float(valid["delta"].mean() / scale)
                if np.isfinite(scale) and scale > EPS
                else np.nan
            )
            run_records.append(
                {
                    "run_label": str(run_label),
                    "outcome": label,
                    "short_outcome": short_label,
                    "n_bouts": int(len(valid)),
                    "start": float(valid["start"].mean()),
                    "switch": float(valid["switch"].mean()),
                    "coefficient": standardized_delta,
                }
            )
    bout_endpoints = pd.concat(records, ignore_index=True)
    run_endpoints = pd.DataFrame(run_records)
    inference = _summarize_run_effects(run_endpoints, ["outcome"])
    short_lookup = {
        label: short for _, label, short in INTEGRATED_TRANSITION_METRICS
    }
    inference["short_outcome"] = inference["outcome"].map(short_lookup)
    return bout_endpoints, run_endpoints, inference


def _transition_density(
    x: np.ndarray,
    y: np.ndarray,
    bins: int = TRANSITION_STATE_BINS,
) -> np.ndarray:
    finite = np.isfinite(x) & np.isfinite(y)
    histogram, _, _ = np.histogram2d(
        y[finite],
        x[finite],
        bins=bins,
        range=((0.0, 1.0), (0.0, 1.0)),
    )
    return np.where(histogram > 0, np.log10(histogram + 1.0), np.nan)


def _draw_integrated_coordinate_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    selected: pd.DataFrame,
    endpoints: pd.DataFrame,
    run_endpoints: pd.DataFrame,
    summary: Mapping[str, float],
    x_column: str,
    y_column: str,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    density = _transition_density(
        selected[x_column].to_numpy(dtype=float),
        selected[y_column].to_numpy(dtype=float),
    )
    image = ax.imshow(
        density,
        origin="lower",
        extent=(0, 1, 0, 1),
        aspect="equal",
        cmap="viridis",
        interpolation="nearest",
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.025)
    colorbar.set_label("log10 risk-root count", fontsize=7.3)
    colorbar.ax.tick_params(labelsize=6.8)
    _draw_ranged_density_contours(
        ax,
        endpoints["start_x"].to_numpy(dtype=float),
        endpoints["start_y"].to_numpy(dtype=float),
        "#2F6690",
        "solid",
        (0.0, 1.0),
        (0.0, 1.0),
    )
    _draw_ranged_density_contours(
        ax,
        endpoints["switch_x"].to_numpy(dtype=float),
        endpoints["switch_y"].to_numpy(dtype=float),
        "#D17C00",
        "dashed",
        (0.0, 1.0),
        (0.0, 1.0),
    )
    ax.annotate(
        "",
        xy=(float(summary["switch_x"]), float(summary["switch_y"])),
        xytext=(float(summary["start_x"]), float(summary["start_y"])),
        arrowprops={
            "arrowstyle": "-|>",
            "color": "#202020",
            "lw": 2.5,
            "mutation_scale": 13,
        },
        zorder=7,
    )
    ax.scatter(
        float(summary["start_x"]),
        float(summary["start_y"]),
        s=48,
        color="#2F6690",
        edgecolor="white",
        linewidth=0.7,
        zorder=8,
    )
    ax.scatter(
        float(summary["switch_x"]),
        float(summary["switch_y"]),
        s=82,
        marker="*",
        color="#D17C00",
        edgecolor="white",
        linewidth=0.6,
        zorder=8,
    )
    ax.axvline(0.5, color="#F8FAFC", ls="--", lw=0.8, alpha=0.9)
    ax.axhline(0.5, color="#F8FAFC", ls="--", lw=0.8, alpha=0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(x_label, fontsize=7.8)
    ax.set_ylabel(y_label, fontsize=7.8)
    ax.set_title(
        (
            f"{title}\n"
            f"N={int(summary['n_bouts']):,} bouts; "
            f"Δ=({float(summary['delta_x']):+.3f}, "
            f"{float(summary['delta_y']):+.3f})"
        ),
        fontsize=8.9,
    )
    ax.tick_params(labelsize=7.2)


def _draw_integrated_metric_endpoint_panel(
    ax: plt.Axes,
    run_endpoints: pd.DataFrame,
    inference_row: pd.Series,
) -> None:
    outcome = str(inference_row["outcome"])
    selected = run_endpoints.loc[
        run_endpoints["outcome"].eq(outcome)
    ].dropna(subset=["start", "switch"])
    if (
        selected.empty
        or not np.isfinite(float(inference_row.get("mean_coefficient", np.nan)))
    ):
        ax.text(
            0.5,
            0.53,
            "Not estimable",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#404040",
        )
        ax.text(
            0.5,
            0.43,
            "Insufficient run-level support",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=7.8,
            color="#5B6470",
        )
        ax.set_title(
            str(inference_row.get("short_outcome", outcome)),
            fontsize=8.9,
        )
        ax.set_axis_off()
        return
    for _, row in selected.iterrows():
        ax.plot(
            [0, 1],
            [row["start"], row["switch"]],
            color="#9AA4AE",
            alpha=0.34,
            lw=0.8,
            zorder=1,
        )
    means = selected[["start", "switch"]].mean().to_numpy(dtype=float)
    sems = selected[["start", "switch"]].sem().to_numpy(dtype=float)
    n_runs = len(selected)
    critical = (
        float(stats.t.ppf(0.975, n_runs - 1)) if n_runs >= 2 else np.nan
    )
    errors = critical * sems if n_runs >= 2 else np.zeros(2, dtype=float)
    ax.errorbar(
        [0, 1],
        means,
        yerr=errors,
        fmt="none",
        ecolor="#404040",
        elinewidth=1.2,
        capsize=3,
        zorder=3,
    )
    ax.plot([0, 1], means, color="#404040", lw=1.4, zorder=3)
    ax.scatter(
        [0, 1],
        means,
        s=[42, 58],
        marker="o",
        color=["#2F6690", "#D17C00"],
        edgecolor="white",
        linewidth=0.6,
        zorder=4,
    )
    q_value = float(inference_row.get("bh_q", np.nan))
    marker = (
        "***" if q_value < 0.001
        else "**" if q_value < 0.01
        else "*" if q_value < 0.05
        else ""
    )
    coefficient = float(inference_row["mean_coefficient"])
    ax.set_xticks([0, 1], ["First comparable\nsearch", "Transition-producing\nsearch"])
    ax.set_title(
        (
            f"{inference_row['short_outcome']}\n"
            f"standardized Δ={coefficient:+.3f}{marker}"
        ),
        fontsize=8.9,
    )
    ax.set_ylabel("Run-equalized mean")
    ax.set_xlim(-0.25, 1.25)
    finite_values = selected[["start", "switch"]].to_numpy(dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size:
        low = float(np.min(finite_values))
        high = float(np.max(finite_values))
        padding = max((high - low) * 0.10, 0.01)
        ax.set_ylim(low - padding, high + padding)
    ax.grid(axis="y", color="#E5E7EB", lw=0.55)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(labelsize=7.3)


def _transition_action_facets(
    selected: pd.DataFrame,
    transition_class: str,
) -> Tuple[str, str, List[int]]:
    if transition_class == "noop_to_active":
        action_column = "destination_action"
        action_role = "Destination"
    else:
        action_column = "bout_action"
        action_role = "Origin"
    actions = sorted(
        int(action)
        for action in selected[action_column].dropna().unique()
        if 1 <= int(action) <= 5
    )
    return action_column, action_role, actions


def _run_weighted_endpoint_densities(
    endpoints: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    columns = [
        "run_label",
        "start_x",
        "start_y",
        "switch_x",
        "switch_y",
    ]
    valid = endpoints[columns].replace(
        [np.inf, -np.inf], np.nan
    ).dropna().copy()
    if valid.empty:
        empty = np.zeros(
            (TRANSITION_STATE_BINS, TRANSITION_STATE_BINS),
            dtype=float,
        )
        return empty.copy(), empty
    run_bout_count = valid.groupby("run_label")["run_label"].transform("size")
    weights = 1.0 / run_bout_count.to_numpy(dtype=float)
    weights /= np.sum(weights)
    histogram_arguments = {
        "bins": TRANSITION_STATE_BINS,
        "range": ((0.0, 1.0), (0.0, 1.0)),
        "weights": weights,
    }
    start, _, _ = np.histogram2d(
        valid["start_y"].to_numpy(dtype=float),
        valid["start_x"].to_numpy(dtype=float),
        **histogram_arguments,
    )
    switch, _, _ = np.histogram2d(
        valid["switch_y"].to_numpy(dtype=float),
        valid["switch_x"].to_numpy(dtype=float),
        **histogram_arguments,
    )
    start = gaussian_filter(start, sigma=1.0, mode="constant")
    switch = gaussian_filter(switch, sigma=1.0, mode="constant")
    if np.sum(start) > 0:
        start /= np.sum(start)
    if np.sum(switch) > 0:
        switch /= np.sum(switch)
    return start, switch


def _highest_density_thresholds(
    density: np.ndarray,
    masses: Sequence[float] = (0.80, 0.50),
) -> np.ndarray:
    values = np.asarray(density, dtype=float)
    positive = values[np.isfinite(values) & (values > 0)]
    if positive.size == 0:
        return np.asarray([], dtype=float)
    ordered = np.sort(positive)[::-1]
    cumulative = np.cumsum(ordered)
    cumulative /= cumulative[-1]
    thresholds = []
    for mass in masses:
        index = min(
            int(np.searchsorted(cumulative, float(mass), side="left")),
            len(ordered) - 1,
        )
        thresholds.append(float(ordered[index]))
    return np.unique(np.sort(np.asarray(thresholds, dtype=float)))


def _draw_probability_mass_contours(
    ax: plt.Axes,
    density: np.ndarray,
    color: str,
    linestyle: str,
) -> None:
    levels = _highest_density_thresholds(density)
    if levels.size == 0:
        return
    centers = (
        np.arange(TRANSITION_STATE_BINS, dtype=float) + 0.5
    ) / TRANSITION_STATE_BINS
    linewidths = np.linspace(0.9, 1.35, len(levels))
    ax.contour(
        centers,
        centers,
        density,
        levels=levels,
        colors=[color],
        linestyles=linestyle,
        linewidths=linewidths,
        alpha=0.95,
        zorder=4,
    )


def _transition_density_facet_records(
    selected: pd.DataFrame,
    transition_class: str,
) -> Tuple[List[Dict[str, object]], str, List[int]]:
    action_column, action_role, actions = _transition_action_facets(
        selected,
        transition_class,
    )
    records: List[Dict[str, object]] = []
    for row_index, specification in enumerate(INTEGRATED_COORDINATES):
        (
            coordinate_name,
            x_column,
            y_column,
            coordinate_title,
            x_label,
            y_label,
        ) = specification
        for column_index, action in enumerate(actions):
            action_rows = selected.loc[
                selected[action_column].eq(action)
            ].copy()
            endpoints, run_endpoints, summary = (
                _class_coordinate_endpoint_tables(
                    action_rows,
                    x_column,
                    y_column,
                )
            )
            start_density, switch_density = (
                _run_weighted_endpoint_densities(endpoints)
            )
            records.append(
                {
                    "row_index": row_index,
                    "column_index": column_index,
                    "coordinate_name": coordinate_name,
                    "coordinate_title": coordinate_title,
                    "x_label": x_label,
                    "y_label": y_label,
                    "active_action": action,
                    "action_role": action_role,
                    "endpoints": endpoints,
                    "run_endpoints": run_endpoints,
                    "summary": summary,
                    "start_density": start_density,
                    "switch_density": switch_density,
                    "pooled_density": (
                        0.5 * (start_density + switch_density)
                    ),
                    "difference_density": (
                        switch_density - start_density
                    ),
                }
            )
    return records, action_role, actions


def _format_transition_density_axis(
    ax: plt.Axes,
    record: Mapping[str, object],
    show_endpoint_markers: bool,
) -> None:
    summary = record["summary"]
    if show_endpoint_markers:
        start = (
            float(summary["start_x"]),
            float(summary["start_y"]),
        )
        switch = (
            float(summary["switch_x"]),
            float(summary["switch_y"]),
        )
        ax.annotate(
            "",
            xy=switch,
            xytext=start,
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#202020",
                "lw": 2.0,
                "mutation_scale": 11,
            },
            zorder=7,
        )
        ax.scatter(
            start[0],
            start[1],
            s=36,
            color="#2F6690",
            edgecolor="white",
            linewidth=0.6,
            zorder=8,
        )
        ax.scatter(
            switch[0],
            switch[1],
            s=64,
            marker="*",
            color="#D17C00",
            edgecolor="white",
            linewidth=0.55,
            zorder=8,
        )
    ax.axvline(0.5, color="#404040", ls="--", lw=0.65, alpha=0.65)
    ax.axhline(0.5, color="#404040", ls="--", lw=0.65, alpha=0.65)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6.6)
    if int(record["column_index"]) == 0:
        ax.set_ylabel(str(record["y_label"]), fontsize=7.2)
    else:
        ax.tick_params(labelleft=False)
    ax.set_xlabel(str(record["x_label"]), fontsize=7.0)
    n_bouts = int(summary["n_bouts"])
    support_note = " · low n" if n_bouts < 20 else ""
    ax.set_title(
        (
            f"{record['action_role']} = "
            f"{action_label(int(record['active_action']))}{support_note}\n"
            f"N={n_bouts:,}; "
            f"Δ=({float(summary['delta_x']):+.3f}, "
            f"{float(summary['delta_y']):+.3f})"
        ),
        fontsize=7.8,
    )


def _plot_transition_probability_mass_facets(
    records: Sequence[Mapping[str, object]],
    actions: Sequence[int],
    transition_class: str,
    action_role: str,
    output_path: Path,
    title_suffix: str = "",
) -> None:
    n_columns = len(actions)
    figure_width = max(9.2, 3.45 * n_columns + 1.0)
    fig, axes = plt.subplots(
        1,
        n_columns,
        figsize=(figure_width, 4.9),
        facecolor="white",
        squeeze=False,
    )
    supported_records = [
        record
        for record in records
        if int(record["summary"]["n_bouts"]) >= 20
    ]
    scale_records = supported_records or list(records)
    maximum = max(
        float(np.nanmax(record["pooled_density"]))
        for record in scale_records
    )
    image = None
    for record in records:
        ax = axes[
            0,
            int(record["column_index"]),
        ]
        if int(record["summary"]["n_bouts"]) < 20:
            ax.set_facecolor("#F5F6F7")
            _format_transition_density_axis(
                ax,
                record,
                show_endpoint_markers=False,
            )
            ax.text(
                0.5,
                0.5,
                "Insufficient support\nDensity not shown",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=8.0,
                color="#5B6470",
            )
            continue
        image = ax.imshow(
            100.0 * np.asarray(record["pooled_density"]),
            origin="lower",
            extent=(0, 1, 0, 1),
            aspect="equal",
            cmap="viridis",
            interpolation="nearest",
            vmin=0.0,
            vmax=max(100.0 * maximum, EPS),
        )
        _draw_probability_mass_contours(
            ax,
            np.asarray(record["start_density"]),
            "#2F6690",
            "solid",
        )
        _draw_probability_mass_contours(
            ax,
            np.asarray(record["switch_density"]),
            "#D17C00",
            "dashed",
        )
        _format_transition_density_axis(
            ax,
            record,
            show_endpoint_markers=True,
        )
    if image is not None:
        colorbar_axis = fig.add_axes([0.94, 0.16, 0.011, 0.60])
        colorbar = fig.colorbar(
            image,
            cax=colorbar_axis,
        )
        colorbar.set_label(
            "Pooled endpoint probability mass per 50×50 bin (%)",
            fontsize=8,
        )
        colorbar.ax.tick_params(labelsize=7)
    handles = [
        Line2D(
            [0], [0], color="#2F6690", lw=1.4,
            label="Start 50% / 80% mass contours",
        ),
        Line2D(
            [0], [0], color="#D17C00", lw=1.4, ls="--",
            label="Transition 50% / 80% mass contours",
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none",
            markerfacecolor="#2F6690", markeredgecolor="white",
            label="Run-equalized start",
        ),
        Line2D(
            [0], [0], marker="*", linestyle="none", markersize=9,
            markerfacecolor="#D17C00", markeredgecolor="white",
            label="Run-equalized transition",
        ),
        Line2D(
            [0], [0], color="#202020", lw=2.0,
            label="Mean displacement",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=5,
        frameon=False,
        fontsize=7.5,
    )
    fig.suptitle(
        _compact_transition_figure_title(
            records,
            transition_class,
            "Pooled Endpoint Probability Mass",
        )
        + title_suffix,
        fontsize=14.5,
        x=0.5,
        y=0.992,
        ha="center",
    )
    fig.subplots_adjust(
        left=0.055,
        right=0.92,
        bottom=0.12,
        top=0.78,
        wspace=0.20,
    )
    save_static_figure(fig, output_path)


def _plot_transition_density_difference_facets(
    records: Sequence[Mapping[str, object]],
    actions: Sequence[int],
    transition_class: str,
    action_role: str,
    output_path: Path,
) -> None:
    n_columns = len(actions)
    figure_width = max(9.2, 3.45 * n_columns + 1.0)
    fig, axes = plt.subplots(
        1,
        n_columns,
        figsize=(figure_width, 4.6),
        facecolor="white",
        squeeze=False,
    )
    supported_records = [
        record
        for record in records
        if int(record["summary"]["n_bouts"]) >= 20
    ]
    displayed_differences: Dict[Tuple[int, int], np.ndarray] = {}
    scale_values: List[np.ndarray] = []
    for record in records:
        pooled = np.asarray(record["pooled_density"], dtype=float)
        difference = 100.0 * np.asarray(
            record["difference_density"],
            dtype=float,
        )
        support_thresholds = _highest_density_thresholds(
            pooled,
            masses=(0.99,),
        )
        support_threshold = (
            float(support_thresholds[0])
            if support_thresholds.size
            else np.inf
        )
        shown = np.where(
            pooled >= support_threshold,
            difference,
            np.nan,
        )
        key = (
            int(record["row_index"]),
            int(record["column_index"]),
        )
        displayed_differences[key] = shown
        if (
            int(record["summary"]["n_bouts"]) >= 20
            or not supported_records
        ):
            finite = np.abs(shown[np.isfinite(shown)])
            if finite.size:
                scale_values.append(finite)
    combined_scale = (
        np.concatenate(scale_values)
        if scale_values
        else np.asarray([EPS], dtype=float)
    )
    limit = max(
        float(np.quantile(combined_scale, 0.99)),
        EPS,
    )
    colormap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "endpoint_mass_change",
        ["#2F6690", "#F7F7F7", "#D17C00"],
    )
    image = None
    for record in records:
        ax = axes[
            0,
            int(record["column_index"]),
        ]
        if int(record["summary"]["n_bouts"]) < 20:
            ax.set_facecolor("#F5F6F7")
            _format_transition_density_axis(
                ax,
                record,
                show_endpoint_markers=False,
            )
            ax.text(
                0.5,
                0.5,
                "Insufficient support\nDensity not shown",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=8.0,
                color="#5B6470",
            )
            continue
        difference = displayed_differences[
            (
                int(record["row_index"]),
                int(record["column_index"]),
            )
        ]
        image = ax.imshow(
            difference,
            origin="lower",
            extent=(0, 1, 0, 1),
            aspect="equal",
            cmap=colormap,
            interpolation="nearest",
            vmin=-limit,
            vmax=limit,
        )
        _format_transition_density_axis(
            ax,
            record,
            show_endpoint_markers=False,
        )
    if image is not None:
        colorbar_axis = fig.add_axes([0.94, 0.14, 0.011, 0.67])
        colorbar = fig.colorbar(
            image,
            cax=colorbar_axis,
            extend="both",
        )
        colorbar.set_label(
            "Transition − start mass (percentage points)",
            fontsize=8,
        )
        colorbar.ax.tick_params(labelsize=7)
    fig.suptitle(
        _compact_transition_figure_title(
            records,
            transition_class,
            (
                "Probability-Mass Difference "
                "(Transition − Start)"
            ),
        ),
        fontsize=14.5,
        x=0.5,
        y=0.992,
        ha="center",
    )
    fig.subplots_adjust(
        left=0.055,
        right=0.92,
        bottom=0.12,
        top=0.86,
        wspace=0.20,
    )
    save_static_figure(fig, output_path)


def _plot_integrated_transition_geometry(
    selected: pd.DataFrame,
    transition_class: str,
    probability_output_paths: Mapping[str, Path],
    density_difference_output_paths: Mapping[str, Path],
    reduced_margin_probability_output_path: Optional[Path] = None,
    reduced_margin_density_difference_output_path: Optional[Path] = None,
) -> pd.DataFrame:
    records, action_role, actions = _transition_density_facet_records(
        selected,
        transition_class,
    )
    if not records or not actions:
        raise ValueError(
            f"No active-action facets for {transition_class}"
        )
    for (
        coordinate_name,
        _,
        _,
        _,
        _,
        _,
    ) in INTEGRATED_COORDINATES:
        coordinate_records = [
            record
            for record in records
            if record["coordinate_name"] == coordinate_name
        ]
        if not coordinate_records:
            continue
        _plot_transition_probability_mass_facets(
            coordinate_records,
            actions,
            transition_class,
            action_role,
            probability_output_paths[coordinate_name],
        )
        _plot_transition_density_difference_facets(
            coordinate_records,
            actions,
            transition_class,
            action_role,
            density_difference_output_paths[coordinate_name],
        )
        if (
            coordinate_name == "actor_change"
            and (
                reduced_margin_probability_output_path is not None
                or reduced_margin_density_difference_output_path is not None
            )
        ):
            reduced_actions = [
                action for action in actions if action not in (4, 5)
            ]
            if reduced_actions and len(reduced_actions) < len(actions):
                reduced_records: List[Dict[str, object]] = []
                for column_index, action in enumerate(reduced_actions):
                    for record in coordinate_records:
                        if int(record["active_action"]) != action:
                            continue
                        reduced_record = dict(record)
                        reduced_record["column_index"] = column_index
                        reduced_records.append(reduced_record)
                if reduced_margin_probability_output_path is not None:
                    _plot_transition_probability_mass_facets(
                        reduced_records,
                        reduced_actions,
                        transition_class,
                        action_role,
                        reduced_margin_probability_output_path,
                    )
                if reduced_margin_density_difference_output_path is not None:
                    _plot_transition_density_difference_facets(
                        reduced_records,
                        reduced_actions,
                        transition_class,
                        action_role,
                        reduced_margin_density_difference_output_path,
                    )
    summary_records = []
    for record in records:
        summary = record["summary"]
        summary_records.append(
            {
                "transition_class": transition_class,
                "action_role": action_role,
                "active_action": int(record["active_action"]),
                "active_action_label": action_label(
                    int(record["active_action"])
                ),
                "coordinate_name": str(record["coordinate_name"]),
                **summary,
            }
        )
    return pd.DataFrame(summary_records)


def _build_transition_duration_bouts(
    selected: pd.DataFrame,
) -> pd.DataFrame:
    group_columns = [
        "run_label",
        "transition_class",
        "bout_action",
        "destination_action",
        "action_bout_id",
    ]
    aggregation: Dict[str, Tuple[str, str]] = {
        "subject": ("subject", "first"),
        "session": ("session", "first"),
        "block": ("block", "first"),
        "game": ("game", "first"),
        "first_real_step": ("real_step", "first"),
        "last_real_step": ("real_step", "last"),
        "bout_length_actions": ("root_position", "size"),
        "start_slack": ("state_x", "first"),
        "start_demand": ("state_y", "first"),
        "previous_reward": ("previous_reward", "first"),
        "previous_done": ("previous_done", "first"),
    }
    for column, _, short_label in INTEGRATED_TRANSITION_METRICS:
        aggregation[f"{short_label}_start"] = (column, "first")
        aggregation[f"{short_label}_switch"] = (column, "last")
    bouts = selected.groupby(group_columns, as_index=False).agg(**aggregation)
    bouts["log_bout_length"] = np.log(
        bouts["bout_length_actions"].to_numpy(dtype=float)
    )
    for _, _, short_label in INTEGRATED_TRANSITION_METRICS:
        bouts[f"{short_label}_delta"] = (
            bouts[f"{short_label}_switch"]
            - bouts[f"{short_label}_start"]
        )
    minimum_by_run = bouts.groupby("run_label")["first_real_step"].transform(
        "min"
    )
    maximum_by_run = bouts.groupby("run_label")["first_real_step"].transform(
        "max"
    )
    denominator = (maximum_by_run - minimum_by_run).replace(0, 1)
    bouts["run_progress"] = (
        bouts["first_real_step"] - minimum_by_run
    ) / denominator
    return bouts


def _fit_transition_duration_regression(
    bouts: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    effect_records: List[Dict[str, object]] = []
    curve_records: List[Dict[str, object]] = []
    maximum_length = max(int(bouts["bout_length_actions"].max()), 1)
    length_grid = np.unique(
        np.rint(
            np.geomspace(1, maximum_length, min(80, maximum_length))
        ).astype(int)
    )
    for run_label, run in bouts.groupby("run_label", sort=False):
        z_length = _zscore_finite(
            run["log_bout_length"].to_numpy(dtype=float)
        )
        start_slack = _zscore_finite(
            run["start_slack"].to_numpy(dtype=float)
        )
        start_demand = _zscore_finite(
            run["start_demand"].to_numpy(dtype=float)
        )
        controls = [
            start_slack,
            start_demand,
            start_slack * start_demand,
            run["previous_reward"].fillna(0).to_numpy(dtype=float),
            run["previous_done"].fillna(0).to_numpy(dtype=float),
            run["run_progress"].to_numpy(dtype=float),
        ]
        for action in range(1, 6):
            controls.append(
                run["bout_action"].eq(action).to_numpy(dtype=float)
            )
            controls.append(
                run["destination_action"].eq(action).to_numpy(dtype=float)
            )
        control_matrix = np.column_stack(controls)
        keep_controls = np.nanstd(control_matrix, axis=0) > EPS
        control_matrix = control_matrix[:, keep_controls]
        for _, _, short_label in INTEGRATED_TRANSITION_METRICS:
            start = run[f"{short_label}_start"].to_numpy(dtype=float)
            switch = run[f"{short_label}_switch"].to_numpy(dtype=float)
            scale = float(np.nanstd(np.concatenate([start, switch])))
            if not np.isfinite(scale) or scale <= EPS:
                continue
            outcome = (switch - start) / scale
            design = np.column_stack(
                [
                    np.ones(len(run)),
                    z_length,
                    control_matrix,
                ]
            )
            finite = np.isfinite(outcome) & np.all(
                np.isfinite(design), axis=1
            )
            fitted_design = design[finite]
            if (
                np.sum(finite) < max(12, fitted_design.shape[1] + 3)
                or np.linalg.matrix_rank(fitted_design) < 2
            ):
                continue
            coefficients = np.linalg.lstsq(
                fitted_design,
                outcome[finite],
                rcond=None,
            )[0]
            length_coefficient = float(coefficients[1])
            effect_records.append(
                {
                    "run_label": str(run_label),
                    "outcome": short_label,
                    "n_bouts": int(np.sum(finite)),
                    "coefficient": length_coefficient,
                }
            )
            run_log_mean = float(
                np.mean(run.loc[finite, "log_bout_length"])
            )
            run_log_sd = float(
                np.std(run.loc[finite, "log_bout_length"])
            )
            if not np.isfinite(run_log_sd) or run_log_sd <= EPS:
                continue
            for length in length_grid:
                standardized_length = (
                    np.log(float(length)) - run_log_mean
                ) / run_log_sd
                curve_records.append(
                    {
                        "run_label": str(run_label),
                        "outcome": short_label,
                        "bout_length_actions": int(length),
                        "adjusted_length_contribution": (
                            length_coefficient * standardized_length
                        ),
                    }
                )
    effects = pd.DataFrame(effect_records)
    if effects.empty:
        summary = pd.DataFrame(
            columns=[
                "outcome",
                "n_runs",
                "mean_coefficient",
                "ci_low",
                "ci_high",
                "signflip_p",
                "positive_run_fraction",
                "bh_q",
            ]
        )
    else:
        summary = _summarize_run_effects(effects, ["outcome"])
    return effects, summary, pd.DataFrame(curve_records)


def _plot_transition_duration_distribution(
    ax: plt.Axes,
    bouts: pd.DataFrame,
) -> None:
    length = bouts["bout_length_actions"].to_numpy(dtype=int)
    maximum = max(int(np.max(length)), 1)
    boundaries = sorted(
        set(
            value
            for value in (1, 2, 3, 5, 8, 12, 18, 36, 60, 100, 200)
            if value <= maximum
        )
    )
    if not boundaries:
        boundaries = [1]
    labels: List[str] = []
    counts: List[int] = []
    for index, lower in enumerate(boundaries):
        upper = boundaries[index + 1] if index + 1 < len(boundaries) else None
        if upper is None:
            mask = length >= lower
            label = f"{lower}+"
        else:
            mask = (length >= lower) & (length < upper)
            label = str(lower) if upper == lower + 1 else f"{lower}–{upper - 1}"
        labels.append(label)
        counts.append(int(np.sum(mask)))
    positions = np.arange(len(labels))
    ax.bar(
        positions,
        counts,
        color="#2F6690",
        edgecolor="white",
        linewidth=0.5,
    )
    if max(counts, default=0) > 50:
        ax.set_yscale("log")
    ax.set_xticks(positions, labels, rotation=35, ha="right")
    ax.set_xlabel("Bout action count")
    ax.set_ylabel("Completed bouts")
    ax.set_title("Bout-length distribution")
    ax.grid(axis="y", color="#E5E7EB", lw=0.55)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _plot_transition_duration_curve(
    ax: plt.Axes,
    curves: pd.DataFrame,
    summary: pd.DataFrame,
    outcome: str,
) -> None:
    if "outcome" not in curves.columns or "outcome" not in summary.columns:
        ax.set_axis_off()
        return
    selected = curves.loc[curves["outcome"].eq(outcome)]
    summary_row = summary.loc[summary["outcome"].eq(outcome)]
    estimable_runs = (
        int(summary_row.iloc[0]["n_runs"]) if not summary_row.empty else 0
    )
    if selected.empty or summary_row.empty or estimable_runs < 5:
        ax.text(
            0.5,
            0.52,
            "Not estimable",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#404040",
        )
        ax.text(
            0.5,
            0.42,
            "Insufficient run-level support",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=7.8,
            color="#5B6470",
        )
        ax.set_title(outcome)
        ax.set_axis_off()
        return
    for _, run in selected.groupby("run_label", sort=False):
        ordered = run.sort_values("bout_length_actions")
        ax.plot(
            np.log1p(ordered["bout_length_actions"]),
            ordered["adjusted_length_contribution"],
            color="#9AA4AE",
            alpha=0.28,
            lw=0.8,
        )
    grouped = selected.groupby("bout_length_actions", sort=True)[
        "adjusted_length_contribution"
    ]
    mean = grouped.mean()
    sem = grouped.sem()
    n = grouped.count()
    adequately_supported = n >= 5
    mean = mean.loc[adequately_supported]
    sem = sem.loc[adequately_supported]
    n = n.loc[adequately_supported]
    if mean.empty:
        ax.text(
            0.5,
            0.52,
            "Not estimable",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#404040",
        )
        ax.text(
            0.5,
            0.42,
            "Fewer than five runs overlap in length",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=7.8,
            color="#5B6470",
        )
        ax.set_title(outcome)
        ax.set_axis_off()
        return
    critical = stats.t.ppf(0.975, np.maximum(n.to_numpy() - 1, 1))
    x = np.log1p(mean.index.to_numpy(dtype=float))
    low = mean.to_numpy() - critical * sem.to_numpy()
    high = mean.to_numpy() + critical * sem.to_numpy()
    ax.fill_between(x, low, high, color="#9BBEDB", alpha=0.34, linewidth=0)
    ax.plot(x, mean, color="#2F6690", lw=2.0)
    ax.axhline(0, color="#404040", ls="--", lw=0.75)
    maximum = int(mean.index.max())
    ticks = [
        value
        for value in (1, 2, 4, 8, 16, 32, 64, 128, 256)
        if value <= maximum
    ]
    if maximum not in ticks:
        ticks.append(maximum)
    ticks = sorted(set(ticks))
    ax.set_xticks(
        np.log1p(ticks),
        [str(value) for value in ticks],
        rotation=28,
        ha="right",
    )
    ax.set_xlabel("Bout action count (log-spaced)")
    ax.set_ylabel("Adjusted endpoint change (within-run SD)")
    row = summary_row.iloc[0]
    q_value = float(row.get("bh_q", np.nan))
    marker = (
        "***" if q_value < 0.001
        else "**" if q_value < 0.01
        else "*" if q_value < 0.05
        else ""
    )
    ax.set_title(
        f"{outcome}\nlength b={float(row['mean_coefficient']):+.3f}{marker}"
    )
    ax.grid(color="#E5E7EB", lw=0.55)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _plot_transition_duration_regression(
    bouts: pd.DataFrame,
    summary: pd.DataFrame,
    curves: pd.DataFrame,
    transition_class: str,
    output_path: Path,
) -> None:
    displayed_outcomes: List[str] = []
    if "outcome" in curves.columns and "outcome" in summary.columns:
        for _, _, outcome in INTEGRATED_TRANSITION_METRICS[:3]:
            outcome_curves = curves.loc[curves["outcome"].eq(outcome)]
            outcome_summary = summary.loc[summary["outcome"].eq(outcome)]
            estimable_runs = (
                int(outcome_summary.iloc[0]["n_runs"])
                if not outcome_summary.empty
                else 0
            )
            if not outcome_curves.empty and estimable_runs >= 5:
                displayed_outcomes.append(outcome)

    n_columns = 1 + len(displayed_outcomes)
    fig, axes = plt.subplots(
        1,
        n_columns,
        figsize=(4.2 * n_columns, 4.9),
        facecolor="white",
        squeeze=False,
    )
    axes = axes.ravel()
    _plot_transition_duration_distribution(axes[0], bouts)
    for ax, outcome in zip(axes[1:], displayed_outcomes):
        _plot_transition_duration_curve(
            ax,
            curves,
            summary,
            outcome,
        )
    fig.suptitle(
        (
            "Continuous same-action bout length and endpoint search change · "
            f"{ACTION_TRANSITION_CLASS_LABELS[transition_class]}"
        ),
        fontsize=14.2,
        y=0.995,
    )
    fig.subplots_adjust(
        left=0.055,
        right=0.99,
        bottom=0.18,
        top=0.88,
        wspace=0.38,
    )
    save_static_figure(fig, output_path)


def generate_integrated_transition_geometry_analysis(
    metrics: pd.DataFrame,
    main_dir: Path,
    inference_dir: Path,
) -> Dict[str, object]:
    """Generate the retained non-tSNE 02 analysis requested by the user."""
    bouts, risk, qa = build_same_action_transition_risk(metrics)
    output_dir = inference_dir / "integrated_transition_geometry"
    output_dir.mkdir(parents=True, exist_ok=True)
    risk.to_csv(
        output_dir / "same_action_transition_risk_rows.csv.gz",
        index=False,
        compression="gzip",
    )
    bouts.to_csv(
        output_dir / "same_action_bouts.csv.gz",
        index=False,
        compression="gzip",
    )
    class_file_tags = {
        "noop_to_active": "1a_noop_to_active",
        "active_to_noop": "1b_active_to_noop",
        "active_to_active": "1c_active_to_active",
    }
    duration_file_tags = {
        "noop_to_active": "2a_noop_to_active",
        "active_to_noop": "2b_active_to_noop",
        "active_to_active": "2c_active_to_active",
    }
    class_qa: Dict[str, object] = {}
    for transition_class in (
        "noop_to_active",
        "active_to_noop",
        "active_to_active",
    ):
        selected = _complete_known_transition_rows(
            risk,
            transition_class,
        )
        if selected.empty:
            log(f"Integrated transition analysis skipped: {transition_class}")
            continue
        class_output = output_dir / transition_class
        class_output.mkdir(parents=True, exist_ok=True)
        selected.to_csv(
            class_output / "comparable_search_rows.csv.gz",
            index=False,
            compression="gzip",
        )
        coordinate_tables = {}
        for (
            coordinate_name,
            x_column,
            y_column,
            _,
            _,
            _,
        ) in INTEGRATED_COORDINATES:
            coordinate_tables[coordinate_name] = (
                _class_coordinate_endpoint_tables(
                    selected,
                    x_column,
                    y_column,
                )
            )
            endpoint_table, run_table, summary = coordinate_tables[
                coordinate_name
            ]
            endpoint_table.to_csv(
                class_output / f"{coordinate_name}_bout_endpoints.csv.gz",
                index=False,
                compression="gzip",
            )
            run_table.to_csv(
                class_output / f"{coordinate_name}_run_endpoints.csv",
                index=False,
            )
            pd.DataFrame([summary]).to_csv(
                class_output / f"{coordinate_name}_summary.csv",
                index=False,
            )
        (
            metric_bout_endpoints,
            metric_run_endpoints,
            metric_inference,
        ) = _class_search_endpoint_tables(selected)
        metric_bout_endpoints.to_csv(
            class_output / "search_metric_bout_endpoints.csv.gz",
            index=False,
            compression="gzip",
        )
        metric_run_endpoints.to_csv(
            class_output / "search_metric_run_endpoints.csv",
            index=False,
        )
        metric_inference.to_csv(
            class_output / "search_metric_endpoint_inference.csv",
            index=False,
        )
        action_facet_summary = _plot_integrated_transition_geometry(
            selected,
            transition_class,
            {
                "actor_start": main_dir
                / (
                    "main_figure"
                    f"{class_file_tags[transition_class]}_"
                    "actor_policy_start_geometry.png"
                ),
                "actor_change": main_dir
                / (
                    "main_figure"
                    f"{class_file_tags[transition_class]}_"
                    "normalized_actor_policy_change_geometry.png"
                ),
            },
            {
                "actor_start": main_dir
                / (
                    "main_figure"
                    f"{class_file_tags[transition_class]}_"
                    "actor_policy_start_density_difference.png"
                ),
                "actor_change": main_dir
                / (
                    "main_figure"
                    f"{class_file_tags[transition_class]}_"
                    "normalized_actor_policy_change_density_difference.png"
                ),
            },
            (
                main_dir
                / (
                    "main_figure"
                    f"{class_file_tags[transition_class]}_"
                    "normalized_actor_policy_change_geometry_"
                    "without_rightfire_leftfire.png"
                )
                if transition_class in {
                    "noop_to_active",
                    "active_to_noop",
                }
                else None
            ),
            (
                main_dir
                / (
                    "main_figure"
                    f"{class_file_tags[transition_class]}_"
                    "normalized_actor_policy_change_density_difference_"
                    "without_rightfire_leftfire.png"
                )
                if transition_class in {
                    "noop_to_active",
                    "active_to_noop",
                }
                else None
            ),
        )
        action_facet_summary.to_csv(
            class_output / "action_facet_coordinate_summary.csv",
            index=False,
        )
        duration_bouts = _build_transition_duration_bouts(selected)
        duration_effects, duration_summary, duration_curves = (
            _fit_transition_duration_regression(duration_bouts)
        )
        duration_bouts.to_csv(
            class_output / "duration_regression_bouts.csv.gz",
            index=False,
            compression="gzip",
        )
        duration_effects.to_csv(
            class_output / "duration_regression_run_effects.csv",
            index=False,
        )
        duration_summary.to_csv(
            class_output / "duration_regression_summary.csv",
            index=False,
        )
        duration_curves.to_csv(
            class_output / "duration_regression_adjusted_curves.csv",
            index=False,
        )
        _plot_transition_duration_regression(
            duration_bouts,
            duration_summary,
            duration_curves,
            transition_class,
            main_dir
            / (
                "main_figure"
                f"{duration_file_tags[transition_class]}_"
                "continuous_duration_regression.png"
            ),
        )
        class_qa[transition_class] = {
            "n_comparable_search_rows": int(len(selected)),
            "n_bouts": int(selected["action_bout_id"].nunique()),
            "n_runs": int(selected["run_label"].nunique()),
            "minimum_bout_action_count": int(
                duration_bouts["bout_length_actions"].min()
            ),
            "maximum_bout_action_count": int(
                duration_bouts["bout_length_actions"].max()
            ),
            "duration_regression_estimable_runs": {
                str(outcome): int(count)
                for outcome, count in duration_effects.groupby(
                    "outcome"
                )["run_label"].nunique().items()
            }
            if not duration_effects.empty
            else {},
        }
    manifest = {
        "analysis": "02 integrated transition geometry",
        "row_definition": (
            "previous human action equals the same-action bout action -> "
            "intervening search -> stay or transition; the entering row whose "
            "previous action differs is excluded"
        ),
        "transition_classes": {
            "noop_to_active": "NOOP stays followed by an active action",
            "active_to_noop": (
                "same-active stays followed by a transition to NOOP"
            ),
            "active_to_active": (
                "same-active stays followed by a different active action"
            ),
        },
        "coordinate_spaces": {
            "actor_start": {
                "x": (
                    "sigmoid(first status-2 source actor-policy raw "
                    "NOOP-minus-best-active logit margin)"
                ),
                "y": "start actor-policy normalized active-action entropy",
                "temporal_role": "source state of first status-2 transition",
            },
            "actor_change": {
                "raw_x": (
                    "terminal status-0 source actor margin minus search-start "
                    "actor margin"
                ),
                "x": "sigmoid(raw_x)",
                "raw_y": "terminal actor Demand minus start actor Demand",
                "y": "(raw_y + 1) / 2",
                "zero_change": "0.5 on both axes",
                "temporal_role": (
                    "within-search contrast between source states, not an "
                    "update of a single actor policy"
                ),
            },
        },
        "density": {
            "bins": TRANSITION_STATE_BINS,
            "weighting": (
                "each run contributes equal total endpoint mass within "
                "active-action facet"
            ),
            "smoothing": "Gaussian sigma=1 grid cell",
            "probability_mass": (
                "pooled mean of normalized start and transition densities"
            ),
            "probability_mass_contours": "50% and 80% highest-density regions",
            "density_difference": "transition density minus start density",
            "colormaps": {
                "probability_mass": "viridis",
                "density_difference": "blue-white-orange",
            },
        },
        "search_outcomes": [
            short for _, _, short in INTEGRATED_TRANSITION_METRICS
        ],
        "figure_display": {
            "transition_geometry": (
                "search-start actor-policy coordinates and normalized "
                "terminal-minus-start Actor Policy changes are saved as "
                "separate figures; each is "
                "faceted by destination action for NOOP-to-active and by "
                "origin action otherwise; probability mass/mean displacement "
                "and signed density-difference are also saved separately; "
                "endpoint search-metric inference remains in CSV tables"
            ),
            "duration_regression": [
                "Bout-length distribution",
                "Branch diversity",
                "Terminal depth",
                "Deep allocation",
            ],
            "path_revisit": (
                "computed and retained in inference tables but omitted from "
                "the duration-regression figures"
            ),
        },
        "endpoint_inference": (
            "bout-paired first comparable versus transition-producing search; "
            "run-equalized standardized changes; exact sign-flip with BH-FDR "
            "across the four outcomes within transition class"
        ),
        "duration_regression": (
            "within-run OLS of standardized endpoint search change on log bout "
            "action count; controls start Actor Policy Slack, Demand, interaction, "
            "origin/destination action, previous reward/done, and run progress"
        ),
        "retired_from_generation": [
            "old Main Figures 2 and 3 commit surfaces",
            "old policy_geometry namespace",
            "old Main Figure 4 Q ambiguity",
            "old interpolated Main Figures 5-7",
            "old Main Figure 8 coefficient-summary sixth panel",
        ],
        "class_qa": class_qa,
        "source_bout_qa": qa,
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest


def generate_planning_state_space_figures(
    metrics: pd.DataFrame,
    figure_dir: Path,
    result_dir: Path,
    permutations: int,
    random_state: int,
    cross2_summary: Path,
) -> None:
    del permutations, random_state, cross2_summary
    main_dir = figure_dir / "main"
    supplementary_dir = figure_dir / "supplementary"
    inference_dir = result_dir / "inference"
    main_dir.mkdir(parents=True, exist_ok=True)
    supplementary_dir.mkdir(parents=True, exist_ok=True)
    inference_dir.mkdir(parents=True, exist_ok=True)

    plot_main_figure1_search_geometry(
        metrics,
        main_dir / "main_figure1_actor_policy_search_geometry.png",
        inference_dir,
    )
    plot_main_figure3_policy_state_search_geometry(
        metrics,
        main_dir
        / "main_figure3_policy_state_search_geometry_associations.png",
        inference_dir,
    )
    manifest = generate_integrated_transition_geometry_analysis(
        metrics,
        main_dir,
        inference_dir,
    )
    action_transition_qa = generate_generalized_action_transition_figures(
        metrics,
        figure_dir,
        result_dir,
    )
    plot_planning_state_space_by_action(
        metrics,
        action_column="human_action",
        action_source="Human",
        output_path=(
            supplementary_dir
            / "supplementary_figureS5_actor_search_ensemble_by_human_action.png"
        ),
        slack_axis_label="Normalized Start Actor Policy Slack",
        demand_axis_label="Actor Policy Demand across status-2 source states",
        figure_title="Actor Policy Slack–Demand during Search",
    )
    plot_state_space_human_transitions(
        metrics,
        supplementary_dir
        / "supplementary_figureS6_actor_search_previous_action_transitions.png",
        slack_axis_label="Normalized Start Actor Policy Slack",
        demand_axis_label="Actor Policy Demand across status-2 source states",
        figure_title=(
            "Human action transitions in Actor Policy Slack–Demand space"
        ),
    )
    plot_planning_state_space_by_action(
        metrics,
        action_column="human_action",
        action_source="Human",
        output_path=(
            supplementary_dir
            / "supplementary_figureS7_actor_start_by_human_action.png"
        ),
        demand_column="d_policy_root_start",
        slack_axis_label="Normalized Start Actor Policy Slack",
        demand_axis_label="Start Actor Policy Demand",
        figure_title="Search-Start Actor Policy Slack–Demand",
    )
    plot_state_space_human_transitions(
        metrics,
        supplementary_dir
        / "supplementary_figureS8_actor_start_previous_action_transitions.png",
        slack_axis_label="Normalized Start Actor Policy Slack",
        demand_column="d_policy_root_start",
        demand_axis_label="Start Actor Policy Demand",
        figure_title=(
            "Human action transitions in search-start Actor Policy "
            "Slack–Demand space"
        ),
        coordinate_note=(
            "start actor policy is the source state of the first status-2 "
            "transition"
        ),
    )
    margin_metrics = _primary_state_data(metrics)
    plot_planning_state_space_by_action(
        margin_metrics,
        action_column="human_action",
        action_source="Human",
        output_path=(
            supplementary_dir
            / "supplementary_figureS9_normalized_actor_policy_change_by_human_action.png"
        ),
        slack_column="delta_policy_slack_normalized",
        slack_is_normalized=True,
        slack_axis_label="Normalized Actor Policy Slack Change",
        demand_column="delta_policy_demand_normalized",
        demand_axis_label="Normalized Actor Policy Demand Change",
        figure_title="Normalized Terminal-minus-Start Actor Policy Change",
        show_opportunity_labels=False,
    )
    manifest["execution_scope"] = (
        "canonical non-tSNE PNG set only; t-SNE generation remains available "
        "in the full execution path; PDF generation is disabled"
    )
    manifest["action_transition_qa"] = action_transition_qa
    manifest["retained_non_tsne_pngs"] = [
        "action_transitions/main_figureA1_action_transition_matrix.png",
        "action_transitions/main_figureA3a_noop_to_active_start_aligned.png",
        "action_transitions/main_figureA3b_active_to_noop_start_aligned.png",
        "action_transitions/main_figureA3c_active_switch_slack_start_aligned.png",
        "action_transitions/main_figureA3d_active_switch_demand_start_aligned.png",
        "main/main_figure1_actor_policy_search_geometry.png",
        "main/main_figure3_policy_state_search_geometry_associations.png",
        "main/main_figure1a_noop_to_active_actor_policy_start_geometry.png",
        "main/main_figure1a_noop_to_active_normalized_actor_policy_change_geometry.png",
        "main/main_figure1a_noop_to_active_actor_policy_start_density_difference.png",
        "main/main_figure1a_noop_to_active_normalized_actor_policy_change_density_difference.png",
        "main/main_figure1b_active_to_noop_actor_policy_start_geometry.png",
        "main/main_figure1b_active_to_noop_normalized_actor_policy_change_geometry.png",
        "main/main_figure1b_active_to_noop_actor_policy_start_density_difference.png",
        "main/main_figure1b_active_to_noop_normalized_actor_policy_change_density_difference.png",
        "main/main_figure1c_active_to_active_actor_policy_start_geometry.png",
        "main/main_figure1c_active_to_active_normalized_actor_policy_change_geometry.png",
        "main/main_figure1c_active_to_active_actor_policy_start_density_difference.png",
        "main/main_figure1c_active_to_active_normalized_actor_policy_change_density_difference.png",
        "main/main_figure2a_noop_to_active_continuous_duration_regression.png",
        "main/main_figure2b_active_to_noop_continuous_duration_regression.png",
        "main/main_figure2c_active_to_active_continuous_duration_regression.png",
        "supplementary/supplementary_figureS5_actor_search_ensemble_by_human_action.png",
        "supplementary/supplementary_figureS6_actor_search_previous_action_transitions.png",
        "supplementary/supplementary_figureS7_actor_start_by_human_action.png",
        "supplementary/supplementary_figureS8_actor_start_previous_action_transitions.png",
        "supplementary/supplementary_figureS9_normalized_actor_policy_change_by_human_action.png",
    ]
    optional_reduced_margin_figures = [
        (
            "main/main_figure1a_noop_to_active_"
            "normalized_actor_policy_change_geometry_without_rightfire_leftfire.png"
        ),
        (
            "main/main_figure1b_active_to_noop_"
            "normalized_actor_policy_change_geometry_without_rightfire_leftfire.png"
        ),
        (
            "main/main_figure1a_noop_to_active_"
            "normalized_actor_policy_change_density_difference_"
            "without_rightfire_leftfire.png"
        ),
        (
            "main/main_figure1b_active_to_noop_"
            "normalized_actor_policy_change_density_difference_"
            "without_rightfire_leftfire.png"
        ),
    ]
    manifest["retained_non_tsne_pngs"].extend(
        relative_path
        for relative_path in optional_reduced_margin_figures
        if (figure_dir / relative_path).exists()
    )
    (inference_dir / "inference_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def fit_tsne(
    matrix: np.ndarray,
    label: str,
    pca_dim: int,
    perplexity: float,
    max_iter: int,
    random_state: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    x = np.asarray(matrix, dtype=np.float32).copy()
    finite = np.all(np.isfinite(x), axis=0)
    variance = np.var(x, axis=0)
    keep = finite & (variance > EPS)
    x = x[:, keep]
    if x.shape[1] == 0:
        raise ValueError(f"{label}: all feature columns are constant or non-finite")
    x = StandardScaler(copy=False).fit_transform(x).astype(np.float32, copy=False)
    n_components = min(int(pca_dim), x.shape[0] - 1, x.shape[1])
    explained = np.nan
    if n_components >= 2 and x.shape[1] > n_components:
        log(f"{label}: PCA {x.shape[1]:,} -> {n_components}")
        pca = PCA(
            n_components=n_components,
            svd_solver="randomized",
            random_state=random_state,
        )
        x = pca.fit_transform(x).astype(np.float32, copy=False)
        explained = float(np.sum(pca.explained_variance_ratio_))
    effective_perplexity = min(float(perplexity), max(2.0, (len(x) - 1) / 3.0))
    if effective_perplexity >= len(x):
        effective_perplexity = max(1.0, len(x) - 1.0)
    log(
        f"{label}: t-SNE n={len(x):,}, dim={x.shape[1]}, "
        f"perplexity={effective_perplexity:g}"
    )
    embedding = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        learning_rate="auto",
        init="pca",
        max_iter=max_iter,
        random_state=random_state,
        method="barnes_hut",
        angle=0.5,
    ).fit_transform(x).astype(np.float32, copy=False)
    diagnostics = {
        "model": label,
        "n_points": int(len(embedding)),
        "raw_width": int(matrix.shape[1]),
        "nonconstant_width": int(np.sum(keep)),
        "pca_width": int(x.shape[1]),
        "pca_explained_variance": explained,
        "perplexity": float(effective_perplexity),
        "max_iter": int(max_iter),
    }
    del x
    gc.collect()
    return embedding, diagnostics


def action_label(action: int) -> str:
    return ACTION_LABELS.get(int(action), f"Action {int(action)}")


def plot_embedding(
    embedding: np.ndarray,
    actions: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 7.0), facecolor="white")
    for action in sorted(np.unique(actions)):
        mask = actions == action
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=10,
            alpha=0.55,
            color=ACTION_COLORS.get(int(action), "#6B7280"),
            edgecolors="none",
            label=f"{action_label(int(action))} (n={int(np.sum(mask)):,})",
        )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(title, loc="center", fontsize=14, pad=12)
    ax.grid(True, color="#E5E7EB", lw=0.6, alpha=0.75)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="best", fontsize=8, markerscale=1.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log(f"Saved {output_path}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--subject", type=parse_subject, required=True, help="Subject, e.g. 3 or sub003")
    parser.add_argument("--game", type=int, choices=[1, 2], required=True)
    parser.add_argument("--sessions", default=None, help="Comma-separated session ids, e.g. 1,2")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--shared-cache-root",
        type=Path,
        default=DEFAULT_SHARED_CACHE_ROOT,
        help="Read-only cache produced by 00_cache.py.",
    )
    parser.add_argument(
        "--cross2-summary",
        type=Path,
        default=DEFAULT_CROSS2_SUMMARY,
        help=(
            "CSV containing subject, game, and Cross2 for Short/Long behavioral "
            "bouts (default: 01_behavioral_analysis summary)"
        ),
    )
    parser.add_argument("--max-files", type=int, default=None, help="Diagnostic: cap chunks per block")
    parser.add_argument("--pca-dim", type=int, default=50)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-iterations", type=int, default=1_000)
    parser.add_argument("--random-state", type=int, default=2026)
    parser.add_argument(
        "--state-space-only",
        "--non-tsne-analysis-only",
        dest="state_space_only",
        action="store_true",
        help=(
            "Read the shared compact cache and generate the canonical "
            "non-tSNE PNG figure set; do not run t-SNE"
        ),
    )
    parser.add_argument(
        "--action-transition-only",
        action="store_true",
        help=(
            "Read the shared analysis cache and refresh only the "
            "dedicated generalized action-transition outputs"
        ),
    )
    parser.add_argument(
        "--cross-game-s9-only",
        action="store_true",
        help=(
            "Generate only the combined Pong/Space Invaders shifted actor-"
            "policy S9 from existing game 1 and game 2 analysis caches"
        ),
    )
    parser.add_argument(
        "--inference-permutations",
        type=int,
        default=2_000,
        help="Within-run circular shifts for Main Figure 2 (default: 2000)",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.pca_dim < 2:
        raise ValueError("--pca-dim must be >= 2")
    if args.perplexity <= 0:
        raise ValueError("--perplexity must be positive")
    if args.tsne_iterations < 250:
        raise ValueError("--tsne-iterations must be >= 250")
    if args.inference_permutations < 100:
        raise ValueError("--inference-permutations must be >= 100")
    figure_only_modes = (
        args.state_space_only,
        args.action_transition_only,
        args.cross_game_s9_only,
    )
    if sum(bool(mode) for mode in figure_only_modes) > 1:
        raise ValueError(
            "Figure-only analysis modes are mutually exclusive"
        )

    sessions = parse_sessions(args.sessions)
    if args.cross_game_s9_only:
        if sessions is not None or args.max_files is not None:
            raise ValueError(
                "--cross-game-s9-only requires full-session caches and cannot "
                "be combined with --sessions or --max-files"
            )
        output_path = generate_cross_game_actor_policy_s9(
            args.subject, args.output_root, args.shared_cache_root
        )
        if output_path is None:
            raise FileNotFoundError(
                "Both current shifted actor-policy analysis caches are "
                "required for cross-game S9"
            )
        log(f"Cross-game actor-policy S9 complete -> {output_path}")
        return

    output_dir = args.output_root / f"sub{args.subject:03d}_game{args.game}"
    figure_dir = output_dir / "figures"
    result_dir = output_dir / "results"
    figure_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    metadata: Optional[pd.DataFrame] = None
    matrices: Optional[Dict[str, np.ndarray]] = None
    qa: Dict[str, object] = {}
    shared_paths = shared_cache_paths(
        args.shared_cache_root, args.subject, args.game
    )
    shared_manifest = validate_shared_manifest(
        shared_paths,
        subject=args.subject,
        game=args.game,
        sessions=sessions,
        max_files=args.max_files,
    )
    analysis_metrics = load_actor_metrics(shared_paths)
    if sessions is not None:
        analysis_metrics = analysis_metrics[
            pd.to_numeric(analysis_metrics["session"], errors="coerce").isin(sessions)
        ].copy()
    analysis_qa = (
        shared_manifest.get("components", {}).get("actor_policy", {})
    )
    if not (
        args.state_space_only
        or args.action_transition_only
    ):
        metadata, matrices, representation_manifest = load_representations(
            shared_paths
        )
        if sessions is not None:
            raise ValueError(
                "Session-subset t-SNE is not supported by the stratified shared "
                "sample; build a session-specific 00 cache instead."
            )
        qa = representation_manifest.get("qa", representation_manifest)
        (result_dir / "qa_summary.json").write_text(
            json.dumps(qa, indent=2), encoding="utf-8"
        )
        metadata.to_csv(result_dir / "sampled_real_steps.csv", index=False)
    if args.action_transition_only:
        generate_generalized_action_transition_figures(
            analysis_metrics,
            figure_dir,
            result_dir,
        )
        log(
            "Generalized action-transition figures complete -> "
            f"{figure_dir / 'action_transitions'}"
        )
        return
    generate_planning_state_space_figures(
        analysis_metrics,
        figure_dir,
        result_dir,
        permutations=args.inference_permutations,
        random_state=args.random_state,
        cross2_summary=args.cross2_summary,
    )
    if sessions is None and args.max_files is None:
        generate_cross_game_actor_policy_s9(
            args.subject, args.output_root, args.shared_cache_root
        )
    if args.state_space_only:
        log(f"State-space figure complete -> {figure_dir}")
        return

    assert metadata is not None and matrices is not None
    if reuse_original_tsne_embeddings(
        args, metadata, figure_dir, result_dir
    ):
        log(f"Complete -> {output_dir}")
        return
    actions = metadata["human_action"].to_numpy(dtype=int)
    titles = {
        "tree_reps": "Vectorized tree representations",
        "im_vectors": "Imagined-state representations",
        "im_vp_vectors": "Imagined value-policy representations",
        "tree_topology": "Search-tree topology",
    }
    diagnostics: List[Dict[str, object]] = []
    combined = metadata.copy()
    for model_index, name in enumerate(titles):
        embedding, diagnostic = fit_tsne(
            matrices[name],
            label=name,
            pca_dim=args.pca_dim,
            perplexity=args.perplexity,
            max_iter=args.tsne_iterations,
            random_state=args.random_state + model_index,
        )
        diagnostics.append(diagnostic)
        combined[f"{name}_tsne_1"] = embedding[:, 0]
        combined[f"{name}_tsne_2"] = embedding[:, 1]
        plot_embedding(
            embedding,
            actions,
            titles[name],
            figure_dir / f"fig_tsne_{name}_by_human_action.png",
        )
        del embedding
        gc.collect()

    combined.to_csv(result_dir / "sampled_real_steps_with_embeddings.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(result_dir / "tsne_diagnostics.csv", index=False)
    log(f"Complete -> {output_dir}")


if __name__ == "__main__":
    main()
