#!/usr/bin/env python3
"""Action-aligned control state and fixed-budget search allocation.

This script implements the analysis frozen in
``research_plan/05_persistence_uncertainty_allocation.md``.  It deliberately
keeps three policies/actions separate:

* human incumbent action: the previous human action;
* root model policy: the value-policy network output before search;
* initial imaginary actor policy: the first status-2 actor output.

The script reads the compact one-row-per-real-step table produced by
``00_cache.py`` and performs within-participant/game nested leave-one-run-out
prediction.  It never reads the raw trace or creates an analysis-specific
cache.  PNG figures are descriptive; held-out increments are the primary
result.

Terminology
-----------
``persistence_evidence`` is a stay-versus-switch log odds.  A larger value
means that the root model policy assigns more probability mass to continuing
the current human action.

``switch_target_uncertainty`` is normalized policy entropy after conditioning
on switching away from the incumbent.  It is choice uncertainty among switch
targets, not epistemic uncertainty about the environment or model.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import re
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import helmert
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from shared_analysis_cache import (
    DEFAULT_SHARED_CACHE_ROOT,
    cache_paths as shared_cache_paths,
    load_model_control,
    validate_manifest as validate_shared_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE_ROOT = ROOT / "test"
DEFAULT_OUTPUT_ROOT = ROOT / "research_script" / "outputs" / "05_persistence_uncertainty_allocation"
ACTOR_SCRIPT = ROOT / "research_script" / "02_actor_policy.py"
EPS = 1e-8
REAL_STATUS = 0
WINDOW_STATUSES = (1, 2, 3)
PRIMARY_STATUS = 2
CACHE_SCHEMA_VERSION = 2


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def parse_subject(value: object) -> int:
    match = re.search(r"\d+", str(value))
    if not match:
        raise argparse.ArgumentTypeError(f"Cannot parse subject id: {value}")
    return int(match.group(0))


def parse_sessions(value: Optional[str]) -> Optional[set[int]]:
    if value is None or not value.strip():
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def _load_trace_helpers():
    """Load stable trace readers from 02 without importing a numeric module name."""
    spec = importlib.util.spec_from_file_location("actor_policy_analysis_02", ACTOR_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load trace helpers from {ACTOR_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TRACE = _load_trace_helpers()


def scalar_at(value: object, index: int) -> float:
    if value is None:
        return np.nan
    try:
        row = np.asarray(value[index], dtype=float).reshape(-1)
    except (IndexError, TypeError, ValueError):
        return np.nan
    finite = row[np.isfinite(row)]
    return float(finite[0]) if finite.size else np.nan


def policy_probabilities(row: object, n_actions: int) -> np.ndarray:
    """Convert a stored policy row (probabilities or logits) to probabilities."""
    try:
        x = np.asarray(row, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.full(n_actions, np.nan)
    if x.size != n_actions or not np.all(np.isfinite(x)):
        return np.full(n_actions, np.nan)
    probability_like = (
        np.min(x) >= -1e-6
        and np.max(x) <= 1.0 + 1e-6
        and abs(float(np.sum(x)) - 1.0) < 1e-3
    )
    if probability_like:
        p = np.clip(x, EPS, None)
        return p / np.sum(p)
    shifted = x - np.max(x)
    exp_x = np.exp(np.clip(shifted, -50.0, 50.0))
    return exp_x / np.sum(exp_x)


def action_row(container: Mapping[str, object], key: str, index: int, n_actions: int) -> np.ndarray:
    if key not in container:
        return np.full(n_actions, np.nan)
    try:
        value = np.asarray(container[key][index], dtype=float).reshape(-1)
    except (IndexError, TypeError, ValueError):
        return np.full(n_actions, np.nan)
    if value.size != n_actions:
        return np.full(n_actions, np.nan)
    return value


def ilr_coordinates(probabilities: np.ndarray) -> np.ndarray:
    """Return K-1 orthonormal log-ratio coordinates for a policy composition."""
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    if p.size < 2 or not np.all(np.isfinite(p)):
        return np.full(max(p.size - 1, 0), np.nan)
    p = np.clip(p, EPS, None)
    p /= np.sum(p)
    return helmert(p.size, full=False) @ np.log(p)


def normalized_entropy(probabilities: np.ndarray) -> float:
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    if p.size <= 1:
        return 0.0
    if not np.all(np.isfinite(p)) or np.sum(p) <= 0:
        return np.nan
    p = np.clip(p, EPS, None)
    p /= np.sum(p)
    return float(-np.sum(p * np.log(p)) / math.log(p.size))


def persistence_coordinates(model_policy: np.ndarray, incumbent: int) -> Dict[str, float]:
    p = np.asarray(model_policy, dtype=float)
    if not (p.ndim == 1 and 0 <= incumbent < p.size and np.all(np.isfinite(p))):
        return {
            "persistence_evidence": np.nan,
            "switch_target_uncertainty": np.nan,
            "strongest_competitor_margin": np.nan,
        }
    p = np.clip(p, EPS, None)
    p /= np.sum(p)
    alternatives = np.delete(p, incumbent)
    alt_mass = float(np.sum(alternatives))
    q = alternatives / max(alt_mass, EPS)
    return {
        "persistence_evidence": float(np.log(p[incumbent]) - np.log(alt_mass)),
        "switch_target_uncertainty": normalized_entropy(q),
        "strongest_competitor_margin": float(
            np.log(p[incumbent]) - np.max(np.log(alternatives))
        ),
    }


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != q.shape or not np.all(np.isfinite(p)) or not np.all(np.isfinite(q)):
        return np.nan
    p = np.clip(p, EPS, None); p /= np.sum(p)
    q = np.clip(q, EPS, None); q /= np.sum(q)
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def add_actor_guidance_coordinates(frame: pd.DataFrame) -> pd.DataFrame:
    """Add action-general coordinates for the actor that starts each search.

    The model-policy coordinates describe the evaluator's pre-search state.
    These actor coordinates describe the policy that actually samples the
    first imaginary action.  They are derived from cached probabilities, so
    this operation does not require re-reading the raw trace.
    """
    actor_columns = sorted(
        [column for column in frame if column.startswith("actor_initial_p_")],
        key=lambda column: int(column.rsplit("_", 1)[1]),
    )
    if not actor_columns:
        raise ValueError("Cached actor_initial_p_* columns are required")
    actor = frame[actor_columns].to_numpy(float)
    incumbents = frame["incumbent_action"].to_numpy(int)
    evidence = np.full(len(frame), np.nan, dtype=float)
    uncertainty = np.full(len(frame), np.nan, dtype=float)
    for row_index, incumbent in enumerate(incumbents):
        coordinates = persistence_coordinates(actor[row_index], int(incumbent))
        evidence[row_index] = coordinates["persistence_evidence"]
        uncertainty[row_index] = coordinates["switch_target_uncertainty"]
    frame = frame.copy()
    frame["actor_persistence_evidence"] = evidence
    frame["actor_switch_target_uncertainty"] = uncertainty
    frame["actor_persistence_x_uncertainty"] = evidence * uncertainty
    frame["model_to_actor_persistence_shift"] = evidence - frame["persistence_evidence"].to_numpy(float)
    frame["model_to_actor_uncertainty_shift"] = uncertainty - frame["switch_target_uncertainty"].to_numpy(float)
    return frame


@dataclass
class SearchBatch:
    actor_policy: np.ndarray
    paths: np.ndarray
    path_valid: np.ndarray
    depth: np.ndarray
    reset_code: np.ndarray
    leading_reset_code: int
    leading_marker_count: int
    reset_marker_count: int
    reset_marker_collision_count: int
    rollout_return: np.ndarray
    n_status1: int
    n_status2: int
    n_status3: int


@dataclass
class RootState:
    model_policy: np.ndarray
    root_value: float
    human_action: int
    reward: float
    done: float
    bout_age: int
    preceding_action: int
    source_chunk: int


def empty_batch(n_actions: int, path_width: int = 0) -> SearchBatch:
    return SearchBatch(
        actor_policy=np.zeros((0, n_actions), dtype=np.float32),
        paths=np.zeros((0, path_width), dtype=np.int16),
        path_valid=np.zeros(0, dtype=bool),
        depth=np.zeros(0, dtype=np.int16),
        reset_code=np.zeros(0, dtype=np.int16),
        leading_reset_code=0,
        leading_marker_count=0,
        reset_marker_count=0,
        reset_marker_collision_count=0,
        rollout_return=np.zeros(0, dtype=np.float32),
        n_status1=0,
        n_status2=0,
        n_status3=0,
    )


def build_search_batch(
    data: Mapping[str, object],
    tree: Mapping[str, object],
    status: np.ndarray,
    indices: np.ndarray,
    n_actions: int,
) -> SearchBatch:
    indices = np.asarray(indices, dtype=int)
    if indices.size:
        indices = indices[np.isin(status[indices], WINDOW_STATUSES)]
    counts = {code: int(np.sum(status[indices] == code)) for code in WINDOW_STATUSES}
    selected = indices[status[indices] == PRIMARY_STATUS]
    marker_indices = indices[np.isin(status[indices], (1, 3))]
    reset = np.zeros(len(selected), dtype=np.int16)
    leading_reset_code = 0
    leading_marker_count = 0
    reset_marker_collision_count = 0
    for marker_index in marker_indices:
        marker_code = int(status[marker_index])
        preceding_position = int(np.searchsorted(selected, marker_index, side="left") - 1)
        if preceding_position >= 0:
            if reset[preceding_position] > 0:
                reset_marker_collision_count += 1
            reset[preceding_position] = (
                3 if marker_code == 3 or reset[preceding_position] == 3 else 1
            )
        else:
            leading_marker_count += 1
            if leading_reset_code > 0:
                reset_marker_collision_count += 1
            leading_reset_code = (
                3 if marker_code == 3 or leading_reset_code == 3 else 1
            )
    action_seq = np.asarray(tree.get("action_seq", []))
    path_width = (
        int(action_seq.shape[-1] // n_actions - 1)
        if action_seq.ndim >= 2 and action_seq.shape[-1] % n_actions == 0
        else 0
    )
    if selected.size == 0:
        out = empty_batch(n_actions, path_width)
        out.n_status1, out.n_status2, out.n_status3 = counts[1], counts[2], counts[3]
        out.leading_reset_code = leading_reset_code
        out.leading_marker_count = leading_marker_count
        out.reset_marker_count = int(len(marker_indices))
        out.reset_marker_collision_count = reset_marker_collision_count
        return out
    if action_seq.ndim < 2 or action_seq.shape[-1] % n_actions:
        raise ValueError("tree_reps['action_seq'] has an invalid action-slot layout")

    slots = np.asarray(action_seq[selected], dtype=np.float32).reshape(len(selected), -1, n_actions)
    occupied = np.sum(np.abs(slots), axis=2) > 0.5
    consecutive = np.cumprod(occupied.astype(np.int8), axis=1).astype(bool)
    occupied_count = np.sum(occupied, axis=1)
    valid = occupied[:, 0] & (occupied_count >= 2) & (np.sum(consecutive, axis=1) == occupied_count)
    depth = np.where(valid, occupied_count - 1, 0).astype(np.int16)
    paths = np.argmax(slots[:, 1:, :], axis=2).astype(np.int16)
    positions = np.arange(paths.shape[1])[None, :]
    paths[positions >= depth[:, None]] = -1
    paths[~valid] = -1

    actor = np.full((len(selected), n_actions), np.nan, dtype=np.float32)
    actor_source = data.get("actor_policy")
    if actor_source is not None:
        actor_array = np.asarray(actor_source)
        if actor_array.ndim >= 2 and len(actor_array) > int(selected.max()):
            raw = np.asarray(actor_array[selected], dtype=np.float32).reshape(len(selected), -1)
            if raw.shape[1] == n_actions:
                actor = raw

    rollout = np.full(len(selected), np.nan, dtype=np.float32)
    rollout_source = tree.get("rollout_return")
    if rollout_source is not None:
        rollout_array = np.asarray(rollout_source)
        if rollout_array.ndim >= 1 and len(rollout_array) > int(selected.max()):
            rollout = np.asarray(rollout_array[selected], dtype=np.float32).reshape(len(selected), -1)[:, 0]
            rollout[~np.isfinite(rollout)] = np.nan

    return SearchBatch(
        actor_policy=actor,
        paths=paths,
        path_valid=valid,
        depth=depth,
        reset_code=reset,
        leading_reset_code=leading_reset_code,
        leading_marker_count=leading_marker_count,
        reset_marker_count=int(len(marker_indices)),
        reset_marker_collision_count=reset_marker_collision_count,
        rollout_return=rollout,
        n_status1=counts[1],
        n_status2=counts[2],
        n_status3=counts[3],
    )


def concatenate_batches(first: Optional[SearchBatch], second: Optional[SearchBatch]) -> Optional[SearchBatch]:
    if first is None:
        return second
    if second is None:
        return first
    width = max(first.paths.shape[1], second.paths.shape[1])

    def pad(batch: SearchBatch) -> np.ndarray:
        if batch.paths.shape[1] == width:
            return batch.paths
        result = np.full((batch.n_status2, width), -1, dtype=np.int16)
        result[:, : batch.paths.shape[1]] = batch.paths
        return result

    first_reset = first.reset_code.copy()
    leading_reset_code = first.leading_reset_code
    leading_marker_count = first.leading_marker_count + second.leading_marker_count
    reset_marker_collision_count = (
        first.reset_marker_collision_count + second.reset_marker_collision_count
    )
    if second.leading_reset_code > 0 and first.n_status2 > 0:
        # The first marker of a following chunk is the boundary after the
        # preceding chunk's final status-2 event, so it is not unresolved.
        leading_marker_count = max(0, leading_marker_count - 1)
        if first_reset[-1] > 0:
            reset_marker_collision_count += 1
        first_reset[-1] = (
            3
            if second.leading_reset_code == 3 or first_reset[-1] == 3
            else 1
        )
    elif second.leading_reset_code > 0:
        if leading_reset_code > 0:
            reset_marker_collision_count += 1
        leading_reset_code = (
            3
            if second.leading_reset_code == 3 or leading_reset_code == 3
            else 1
        )

    return SearchBatch(
        actor_policy=np.vstack([first.actor_policy, second.actor_policy]),
        paths=np.vstack([pad(first), pad(second)]),
        path_valid=np.concatenate([first.path_valid, second.path_valid]),
        depth=np.concatenate([first.depth, second.depth]),
        reset_code=np.concatenate([first_reset, second.reset_code]),
        leading_reset_code=leading_reset_code,
        leading_marker_count=leading_marker_count,
        reset_marker_count=first.reset_marker_count + second.reset_marker_count,
        reset_marker_collision_count=reset_marker_collision_count,
        rollout_return=np.concatenate([first.rollout_return, second.rollout_return]),
        n_status1=first.n_status1 + second.n_status1,
        n_status2=first.n_status2 + second.n_status2,
        n_status3=first.n_status3 + second.n_status3,
    )


def make_root(
    data: Mapping[str, object], tree: Mapping[str, object], index: int,
    n_actions: int, previous: Optional[RootState], source_chunk: int,
) -> RootState:
    human_action = TRACE.decode_action(data.get("human_action"), index)
    bout_age = (
        previous.bout_age + 1
        if previous is not None and human_action == previous.human_action
        else 1
    )
    model_policy = policy_probabilities(action_row(tree, "root_policy", index, n_actions), n_actions)
    root_value = scalar_at(tree.get("root_v"), index)
    if not np.isfinite(root_value):
        root_value = scalar_at(tree.get("root_vs"), index)
    return RootState(
        model_policy=model_policy,
        root_value=root_value,
        human_action=human_action,
        reward=scalar_at(data.get("cur_rewards"), index),
        done=scalar_at(tree.get("root_d"), index),
        bout_age=bout_age,
        preceding_action=(previous.human_action if previous is not None else -1),
        source_chunk=source_chunk,
    )


def search_bouts(batch: SearchBatch) -> List[Dict[str, float]]:
    """Return bouts delimited by public status-1/status-3 reset markers."""
    n = batch.n_status2
    if n == 0:
        return []
    known = (
        len(batch.reset_code) == n
        and batch.reset_marker_collision_count == 0
        and int(np.sum(batch.reset_code > 0))
        == batch.reset_marker_count - batch.leading_marker_count
    )
    reset = batch.reset_code.copy()
    ends = np.flatnonzero(reset > 0).tolist()
    if not ends or ends[-1] != n - 1:
        ends.append(n - 1)
    starts = [0] + [end + 1 for end in ends[:-1]]
    bouts: List[Dict[str, float]] = []
    for start, end in zip(starts, ends):
        positions = np.flatnonzero(batch.path_valid[start : end + 1]) + start
        if not positions.size:
            continue
        first, terminal = int(positions[0]), int(positions[-1])
        bouts.append({
            "start": first,
            "end": terminal,
            "branch": int(batch.paths[first, 0]),
            "terminal_depth": int(batch.depth[terminal]),
            "terminal_return": float(batch.rollout_return[terminal]),
            "reset_code": int(reset[end]) if end < len(reset) else 0,
            "reset_source_exact": int(known),
        })
    return bouts


def allocation_metrics(
    batch: SearchBatch,
    incumbent: int,
    n_actions: int,
    voluntary_reset_code: int,
    forced_reset_code: int,
) -> Dict[str, float]:
    valid_positions = np.flatnonzero(batch.path_valid)
    valid_branches = batch.paths[valid_positions, 0] if valid_positions.size else np.zeros(0, dtype=int)
    bouts = search_bouts(batch)
    bout_branches = np.asarray([item["branch"] for item in bouts], dtype=int)
    terminal_depths = np.asarray([item["terminal_depth"] for item in bouts], dtype=float)

    node_inc = int(np.sum(valid_branches == incumbent))
    node_alt = int(np.sum(valid_branches != incumbent))
    start_inc = int(np.sum(bout_branches == incumbent))
    start_alt = int(np.sum(bout_branches != incumbent))
    start_logratio = float(np.log((start_inc + 0.5) / (start_alt + 0.5)))
    node_logratio = float(np.log((node_inc + 0.5) / (node_alt + 0.5)))

    counts = np.bincount(valid_branches, minlength=n_actions).astype(float)
    alt_counts = np.delete(counts, incumbent) if incumbent < len(counts) else np.zeros(0)
    alt_branch_entropy = normalized_entropy(alt_counts) if np.sum(alt_counts) > 0 else np.nan

    inc_depth = terminal_depths[bout_branches == incumbent]
    alt_depth = terminal_depths[bout_branches != incumbent]
    both_visited = bool(inc_depth.size and alt_depth.size)

    all_reset_codes = batch.reset_code[batch.reset_code > 0]
    voluntary = int(np.sum(all_reset_codes == voluntary_reset_code))
    forced = int(np.sum(all_reset_codes == forced_reset_code))
    voluntary_pairs = 0
    voluntary_changes = 0
    all_pairs = 0
    all_changes = 0
    for index in range(len(bouts) - 1):
        previous, following = bouts[index], bouts[index + 1]
        code = int(previous["reset_code"])
        if code <= 0:
            continue
        all_pairs += 1
        all_changes += int(previous["branch"] != following["branch"])
        if code == voluntary_reset_code:
            voluntary_pairs += 1
            voluntary_changes += int(previous["branch"] != following["branch"])

    inc_returns = np.asarray([
        item["terminal_return"] for item in bouts
        if item["branch"] == incumbent and np.isfinite(item["terminal_return"])
    ])
    alt_returns = np.asarray([
        item["terminal_return"] for item in bouts
        if item["branch"] != incumbent and np.isfinite(item["terminal_return"])
    ])

    return {
        "n_status1": batch.n_status1,
        "n_status2": batch.n_status2,
        "n_status3": batch.n_status3,
        "reset_marker_count": batch.reset_marker_count,
        "leading_marker_count": batch.leading_marker_count,
        "mapped_marker_count": int(batch.reset_marker_count - batch.leading_marker_count),
        "reset_boundary_count": int(np.sum(batch.reset_code > 0)),
        "reset_marker_collision_count": batch.reset_marker_collision_count,
        "unresolved_leading_reset": int(batch.leading_reset_code > 0),
        "reset_marker_coverage": (
            float(
                np.sum(batch.reset_code > 0)
                / (batch.reset_marker_count - batch.leading_marker_count)
            )
            if batch.reset_marker_count > batch.leading_marker_count else 1.0
        ),
        "n_valid_expansions": int(valid_positions.size),
        "valid_expansion_fraction": float(valid_positions.size / batch.n_status2) if batch.n_status2 else np.nan,
        "n_search_bouts": int(len(bouts)),
        "incumbent_start_count": start_inc,
        "alternative_start_count": start_alt,
        "incumbent_node_count": node_inc,
        "alternative_node_count": node_alt,
        "incumbent_start_share": float(start_inc / len(bouts)) if bouts else np.nan,
        "incumbent_node_share": float(node_inc / valid_positions.size) if valid_positions.size else np.nan,
        "incumbent_start_logratio": start_logratio,
        "incumbent_node_logratio": node_logratio,
        "continuation_allocation": node_logratio - start_logratio,
        "alternative_branch_entropy": alt_branch_entropy,
        "both_visited": int(both_visited),
        "conditional_terminal_depth_contrast": (
            float(np.mean(inc_depth) - np.mean(alt_depth)) if both_visited else np.nan
        ),
        "mean_bout_terminal_depth": float(np.mean(terminal_depths)) if terminal_depths.size else np.nan,
        "voluntary_reset_count": voluntary,
        "forced_reset_count": forced,
        "voluntary_reset_fraction": float(voluntary / batch.n_status2) if batch.n_status2 else np.nan,
        "forced_reset_fraction": float(forced / batch.n_status2) if batch.n_status2 else np.nan,
        "reset_reorientation": float(all_changes / all_pairs) if all_pairs else np.nan,
        "voluntary_reset_reorientation": float(voluntary_changes / voluntary_pairs) if voluntary_pairs else np.nan,
        "reset_code_exact": int(all(item["reset_source_exact"] for item in bouts)) if bouts else 0,
        "root_branch_return_contrast": (
            float(np.mean(inc_returns) - np.mean(alt_returns))
            if inc_returns.size and alt_returns.size else np.nan
        ),
        "root_branch_return_coverage": int(bool(inc_returns.size and alt_returns.size)),
    }


def make_window_record(
    meta: object,
    real_step: int,
    previous_root: RootState,
    current_action: int,
    current_actor_policy: np.ndarray,
    batch: SearchBatch,
    n_actions: int,
    voluntary_reset_code: int,
    forced_reset_code: int,
) -> Dict[str, object]:
    incumbent = previous_root.human_action
    model_policy = previous_root.model_policy
    actor_initial = (
        policy_probabilities(batch.actor_policy[0], n_actions)
        if batch.n_status2 and len(batch.actor_policy) else np.full(n_actions, np.nan)
    )
    actor_terminal = policy_probabilities(current_actor_policy, n_actions)
    coordinates = persistence_coordinates(model_policy, incumbent)
    allocation = allocation_metrics(
        batch, incumbent, n_actions, voluntary_reset_code, forced_reset_code
    )
    model_candidate = int(np.argmax(model_policy)) if np.all(np.isfinite(model_policy)) else -1
    actor_candidate = int(np.argmax(actor_initial)) if np.all(np.isfinite(actor_initial)) else -1

    record: Dict[str, object] = {
        "subject": int(meta.subject),
        "session": int(meta.session),
        "block": int(meta.block),
        "game": int(meta.game),
        "run_label": str(meta.run_label),
        "source_chunk": int(meta.chunk),
        "real_step": int(real_step),
        "real_step_id": f"{meta.run_label}_real{real_step:06d}",
        "incumbent_action": int(incumbent),
        "action_before_incumbent": int(previous_root.preceding_action),
        "human_action": int(current_action),
        "switch": int(current_action != incumbent),
        "bout_age": int(previous_root.bout_age),
        "prior_reward": previous_root.reward,
        "prior_done": previous_root.done,
        "root_value": previous_root.root_value,
        "model_candidate": model_candidate,
        "actor_initial_candidate": actor_candidate,
        **coordinates,
        **allocation,
        "persistence_x_uncertainty": (
            coordinates["persistence_evidence"] * coordinates["switch_target_uncertainty"]
        ),
        "actor_model_candidate_support": (
            float(actor_initial[model_candidate]) if model_candidate >= 0 and np.all(np.isfinite(actor_initial)) else np.nan
        ),
        "actor_model_js": jensen_shannon(model_policy, actor_initial),
        "initial_actor_policy_complete": int(np.all(np.isfinite(actor_initial))),
        "terminal_actor_policy_complete": int(np.all(np.isfinite(actor_terminal))),
    }
    model_ilr = ilr_coordinates(model_policy)
    actor_ilr = ilr_coordinates(actor_initial)
    terminal_ilr = ilr_coordinates(actor_terminal)
    for action in range(n_actions):
        record[f"model_root_p_{action}"] = float(model_policy[action])
        record[f"actor_initial_p_{action}"] = float(actor_initial[action])
        record[f"actor_terminal_p_{action}"] = float(actor_terminal[action])
    for index in range(n_actions - 1):
        record[f"model_root_ilr_{index}"] = float(model_ilr[index])
        record[f"actor_initial_ilr_{index}"] = float(actor_ilr[index])
        record[f"actor_terminal_ilr_{index}"] = float(terminal_ilr[index])
    return record


def trace_signature(blocks: Mapping[object, Sequence[object]]) -> List[Dict[str, object]]:
    signature: List[Dict[str, object]] = []
    for metas in blocks.values():
        for meta in metas:
            stat = meta.path.stat()
            signature.append({
                "path": str(meta.path.resolve()),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            })
    return signature


def extract_model_control_windows(
    blocks: Mapping[object, Sequence[object]],
    max_files: Optional[int],
    voluntary_reset_code: int,
    forced_reset_code: int,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    records: List[Dict[str, object]] = []
    n_actions: Optional[int] = None
    loaded_files = 0
    invalid_actions = 0
    empty_windows = 0

    for block_number, metas_all in enumerate(blocks.values(), start=1):
        metas = list(metas_all[:max_files] if max_files is not None else metas_all)
        previous_root: Optional[RootState] = None
        carry: Optional[SearchBatch] = None
        real_step = 0
        log(f"Block {block_number}/{len(blocks)}: {len(metas)} chunks")

        for file_number, meta in enumerate(metas, start=1):
            size_gib = meta.path.stat().st_size / 1024**3
            log(f"Load {meta.path.name} ({size_gib:.1f} GiB; {file_number}/{len(metas)})")
            data = TRACE.load_npy_dict(meta.path)
            loaded_files += 1
            status = np.asarray(data.get("status", []), dtype=int).reshape(-1)
            tree = data.get("tree_reps")
            if status.size == 0 or not isinstance(tree, Mapping):
                raise ValueError(f"Missing status/tree_reps in {meta.path}")
            detected_actions = TRACE.infer_num_actions(tree)
            if n_actions is None:
                n_actions = detected_actions
                log(f"Detected {n_actions} actions")
            elif detected_actions != n_actions:
                raise ValueError(f"Action count changed in {meta.path}")

            real_indices = np.flatnonzero(status == REAL_STATUS)
            previous_local_real: Optional[int] = None
            for current_real_value in real_indices:
                current_real = int(current_real_value)
                real_step += 1
                between_start = 0 if previous_local_real is None else previous_local_real + 1
                between = np.arange(between_start, current_real, dtype=int)
                current_batch = build_search_batch(data, tree, status, between, n_actions)
                batch = concatenate_batches(
                    carry if previous_local_real is None else None,
                    current_batch,
                )
                if batch is None:
                    batch = empty_batch(n_actions)

                current_action = TRACE.decode_action(data.get("human_action"), current_real)
                if previous_root is not None:
                    if 0 <= previous_root.human_action < n_actions and 0 <= current_action < n_actions:
                        current_actor = action_row(data, "actor_policy", current_real, n_actions)
                        records.append(make_window_record(
                            meta=meta,
                            real_step=real_step,
                            previous_root=previous_root,
                            current_action=current_action,
                            current_actor_policy=current_actor,
                            batch=batch,
                            n_actions=n_actions,
                            voluntary_reset_code=voluntary_reset_code,
                            forced_reset_code=forced_reset_code,
                        ))
                        empty_windows += int(batch.n_status2 == 0)
                    else:
                        invalid_actions += 1

                previous_root = make_root(
                    data, tree, current_real, n_actions,
                    previous=previous_root, source_chunk=int(meta.chunk),
                )
                previous_local_real = current_real
                carry = None

            tail_start = previous_local_real + 1 if previous_local_real is not None else 0
            tail_indices = np.arange(tail_start, len(status), dtype=int)
            tail = build_search_batch(data, tree, status, tail_indices, n_actions)
            carry = concatenate_batches(carry, tail)

            del data, tree, status
            gc.collect()
            log(f"Processed {meta.path.name}; cached windows={len(records):,}")

    if not records or n_actions is None:
        raise RuntimeError("No eligible real-step search windows were extracted")
    frame = pd.DataFrame.from_records(records)
    frame["run_progress"] = frame.groupby("run_label", sort=False).cumcount()
    run_sizes = frame.groupby("run_label")["run_label"].transform("size")
    frame["run_progress"] = frame["run_progress"] / np.maximum(run_sizes - 1, 1)
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "n_actions": n_actions,
        "n_rows": int(len(frame)),
        "n_runs": int(frame["run_label"].nunique()),
        "loaded_files": loaded_files,
        "max_files_per_block": max_files,
        "invalid_action_windows": invalid_actions,
        "empty_status2_windows": empty_windows,
        "voluntary_reset_code": voluntary_reset_code,
        "forced_reset_code": forced_reset_code,
        "trace_signature": trace_signature(blocks),
        "policy_roles": {
            "model_root": "tree_reps.root_policy at the preceding real step",
            "actor_initial": "first status-2 actor_policy in the search window",
            "actor_terminal": "status-0 actor_policy at the next real step; sensitivity only",
        },
    }
    return frame, manifest


def save_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def merge_task_state(
    frame: pd.DataFrame,
    task_state_csv: Optional[Path],
    requested_columns: Sequence[str],
) -> Tuple[pd.DataFrame, List[str]]:
    if task_state_csv is None:
        return frame, []
    state = pd.read_csv(task_state_csv)
    keys = [key for key in ("real_step_id", "run_label", "real_step") if key in frame and key in state]
    if not keys:
        raise ValueError("Task-state CSV needs real_step_id or run_label+real_step keys")
    if "real_step_id" in keys:
        keys = ["real_step_id"]
    elif set(("run_label", "real_step")).issubset(keys):
        keys = ["run_label", "real_step"]
    columns = list(requested_columns)
    if not columns:
        columns = [
            column for column in state.columns
            if column not in keys
            and pd.api.types.is_numeric_dtype(state[column])
            and (column.startswith("state_") or column.startswith("phase_") or column.startswith("visual_"))
        ]
    missing = sorted(set(columns) - set(state.columns))
    if missing:
        raise ValueError(f"Task-state columns absent from CSV: {missing}")
    merged = frame.merge(state[keys + columns], on=keys, how="left", validate="many_to_one")
    return merged, columns


def qa_summary(frame: pd.DataFrame, manifest: Mapping[str, object], task_columns: Sequence[str]) -> Dict[str, object]:
    action_table = (
        frame.groupby("incumbent_action")["switch"]
        .agg(["size", "sum", "mean"])
        .rename(columns={"size": "opportunities", "sum": "switches", "mean": "switch_rate"})
        .reset_index()
        .to_dict(orient="records")
    )
    status2 = frame["n_status2"].to_numpy(float)
    valid = frame["n_valid_expansions"].to_numpy(float)
    return {
        "subject": int(frame["subject"].iloc[0]),
        "game": int(frame["game"].iloc[0]),
        "n_rows": int(len(frame)),
        "n_runs": int(frame["run_label"].nunique()),
        "switch_count": int(frame["switch"].sum()),
        "switch_prevalence": float(frame["switch"].mean()),
        "action_coverage": action_table,
        "status2_count": {
            "min": float(np.nanmin(status2)),
            "median": float(np.nanmedian(status2)),
            "max": float(np.nanmax(status2)),
            "fraction_nominal_29": float(np.mean(status2 == 29)),
        },
        "valid_expansions": {
            "median": float(np.nanmedian(valid)),
            "fraction_nonzero": float(np.mean(valid > 0)),
            "mean_valid_fraction": float(frame["valid_expansion_fraction"].mean()),
        },
        "initial_actor_policy_coverage": float(frame["initial_actor_policy_complete"].mean()),
        "terminal_actor_policy_coverage": float(frame["terminal_actor_policy_complete"].mean()),
        "both_visited_fraction": float(frame["both_visited"].mean()),
        "return_contrast_coverage": float(frame["root_branch_return_coverage"].mean()),
        "voluntary_reset_total": int(frame["voluntary_reset_count"].sum()),
        "forced_reset_total": int(frame["forced_reset_count"].sum()),
        "exact_reset_code_fraction": float(frame["reset_code_exact"].mean()),
        "reset_markers": {
            "total": int(frame["reset_marker_count"].sum()),
            "leading_excluded": int(frame["leading_marker_count"].sum()),
            "eligible": int(
                (frame["reset_marker_count"] - frame["leading_marker_count"]).sum()
            ),
            "mapped_boundaries": int(frame["reset_boundary_count"].sum()),
            "collision_count": int(frame["reset_marker_collision_count"].sum()),
            "unresolved_leading_windows": int(frame["unresolved_leading_reset"].sum()),
            "mean_coverage": float(frame["reset_marker_coverage"].mean()),
        },
        "task_state_columns": list(task_columns),
        "baseline_scope": "task-state augmented" if task_columns else "minimal behavioral baseline",
        "timing": (
            "All primary predictors use the preceding real-step root and intervening status-2 "
            "search records. The current human action is used only as the outcome."
        ),
        "cache_manifest": dict(manifest),
    }


def make_preprocessor(
    numeric_columns: Sequence[str], categorical_columns: Sequence[str]
) -> ColumnTransformer:
    numeric_columns = list(dict.fromkeys(numeric_columns))
    categorical_columns = list(dict.fromkeys(categorical_columns))
    transformers = [
        (
            "age",
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("spline", SplineTransformer(n_knots=4, degree=3, include_bias=False)),
                ("scale", StandardScaler()),
            ]),
            ["bout_age"],
        ),
        (
            "numeric",
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]),
            [column for column in numeric_columns if column != "bout_age"],
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns,
        ),
    ]
    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_classifier(
    numeric_columns: Sequence[str], categorical_columns: Sequence[str], penalty_c: float
) -> Pipeline:
    return Pipeline([
        ("features", make_preprocessor(numeric_columns, categorical_columns)),
        ("classifier", LogisticRegression(C=penalty_c, penalty="l2", solver="lbfgs", max_iter=3000)),
    ])


def inner_select_penalty(
    frame: pd.DataFrame,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    penalties: Sequence[float],
    inner_folds: int,
) -> float:
    groups = frame["run_label"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return float(penalties[len(penalties) // 2])
    splitter = GroupKFold(n_splits=min(inner_folds, len(unique_groups)))
    y = frame["switch"].to_numpy(int)
    losses: Dict[float, List[Tuple[float, int]]] = {float(value): [] for value in penalties}
    for train_index, valid_index in splitter.split(frame, y, groups):
        y_train, y_valid = y[train_index], y[valid_index]
        if np.unique(y_train).size < 2:
            continue
        for penalty in penalties:
            model = make_classifier(numeric_columns, categorical_columns, float(penalty))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(frame.iloc[train_index], y_train)
            prediction = model.predict_proba(frame.iloc[valid_index])[:, 1]
            losses[float(penalty)].append((float(log_loss(y_valid, prediction, labels=[0, 1])), len(valid_index)))
    scored = []
    for penalty, values in losses.items():
        if values:
            scored.append((sum(loss * n for loss, n in values) / sum(n for _, n in values), penalty))
    return min(scored)[1] if scored else float(penalties[len(penalties) // 2])


def nested_loro_pair(
    frame: pd.DataFrame,
    test_name: str,
    base_numeric: Sequence[str],
    augmented_numeric: Sequence[str],
    categorical: Sequence[str],
    penalties: Sequence[float],
    inner_folds: int,
    require_valid_search: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    required = ["switch", "run_label", *base_numeric, *augmented_numeric, *categorical]
    required = list(dict.fromkeys(required))
    data = frame.copy()
    if require_valid_search:
        data = data.loc[data["n_valid_expansions"] > 0]
    data = data.loc[data["switch"].isin([0, 1])]
    # Policy/allocation variables define eligibility; nuisance variables can be imputed.
    core = [column for column in augmented_numeric if column.startswith(("persistence_", "switch_target_", "model_root_ilr_", "actor_initial_ilr_", "actor_terminal_ilr_", "incumbent_"))]
    if core:
        data = data.loc[np.all(np.isfinite(data[core].to_numpy(float)), axis=1)]
    if len(data) == 0:
        return pd.DataFrame(), pd.DataFrame()

    score_rows: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []
    for test_run in sorted(data["run_label"].unique()):
        train = data.loc[data["run_label"] != test_run]
        test = data.loc[data["run_label"] == test_run]
        if len(train) == 0 or len(test) == 0 or train["switch"].nunique() < 2:
            continue
        c_base = inner_select_penalty(train, base_numeric, categorical, penalties, inner_folds)
        c_aug = inner_select_penalty(train, augmented_numeric, categorical, penalties, inner_folds)
        base_model = make_classifier(base_numeric, categorical, c_base)
        aug_model = make_classifier(augmented_numeric, categorical, c_aug)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base_model.fit(train, train["switch"].to_numpy(int))
            aug_model.fit(train, train["switch"].to_numpy(int))
        base_prediction = base_model.predict_proba(test)[:, 1]
        aug_prediction = aug_model.predict_proba(test)[:, 1]
        y_test = test["switch"].to_numpy(int)
        base_loss = float(log_loss(y_test, base_prediction, labels=[0, 1]))
        aug_loss = float(log_loss(y_test, aug_prediction, labels=[0, 1]))
        score_rows.append({
            "test": test_name,
            "test_run": test_run,
            "n_test": len(test),
            "switch_prevalence": float(np.mean(y_test)),
            "base_log_loss": base_loss,
            "augmented_log_loss": aug_loss,
            "delta_log_loss": base_loss - aug_loss,
            "base_penalty_c": c_base,
            "augmented_penalty_c": c_aug,
        })
        for row_index, observed, base_p, aug_p in zip(test.index, y_test, base_prediction, aug_prediction):
            prediction_rows.append({
                "test": test_name,
                "row_index": int(row_index),
                "test_run": test_run,
                "observed": int(observed),
                "base_probability": float(base_p),
                "augmented_probability": float(aug_p),
            })
    return pd.DataFrame(score_rows), pd.DataFrame(prediction_rows)


def exact_signflip_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    observed = float(np.mean(values))
    if values.size > 20:
        rng = np.random.default_rng(20260806)
        signs = rng.choice((-1.0, 1.0), size=(100_000, values.size))
    else:
        codes = np.arange(2 ** values.size, dtype=np.uint64)[:, None]
        bits = (codes >> np.arange(values.size, dtype=np.uint64)) & 1
        signs = np.where(bits, 1.0, -1.0)
    null = np.mean(signs * values[None, :], axis=1)
    return float((1 + np.sum(null >= observed - 1e-15)) / (len(null) + 1))


def summarize_nested_scores(scores: pd.DataFrame, subject: int, game: int) -> pd.DataFrame:
    rows = []
    for test_name, group in scores.groupby("test"):
        delta = group["delta_log_loss"].to_numpy(float)
        rows.append({
            "subject": subject,
            "game": game,
            "test": test_name,
            "n_runs": len(group),
            "mean_delta_log_loss_nats_per_opportunity": float(np.mean(delta)),
            "median_delta_log_loss": float(np.median(delta)),
            "positive_run_fraction": float(np.mean(delta > 0)),
            "development_run_signflip_p_descriptive": exact_signflip_p(delta),
            "inference_note": "Run-level descriptive development evidence; participant is the population unit.",
        })
    return pd.DataFrame(rows)


def action_generalization_gate(
    frame: pd.DataFrame,
    base_numeric: Sequence[str],
    augmented_numeric: Sequence[str],
    categorical: Sequence[str],
    penalties: Sequence[float],
    inner_folds: int,
    min_outcomes: int,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    data = frame.copy()
    defining_columns = [
        column for column in augmented_numeric
        if column.startswith(("persistence_", "switch_target_"))
    ]
    if defining_columns:
        data = data.loc[
            np.all(np.isfinite(data[defining_columns].to_numpy(float)), axis=1)
        ]
    for held_action in sorted(data["incumbent_action"].unique()):
        test = data.loc[data["incumbent_action"] == held_action]
        train = data.loc[data["incumbent_action"] != held_action]
        n_switch = int(test["switch"].sum())
        n_persist = int(len(test) - n_switch)
        if min(n_switch, n_persist) < min_outcomes or train["switch"].nunique() < 2:
            rows.append({
                "heldout_incumbent_action": int(held_action),
                "eligible": 0,
                "n_test": len(test),
                "n_switch": n_switch,
                "n_persist": n_persist,
            })
            continue
        c_base = inner_select_penalty(train, base_numeric, categorical, penalties, inner_folds)
        c_aug = inner_select_penalty(train, augmented_numeric, categorical, penalties, inner_folds)
        base = make_classifier(base_numeric, categorical, c_base).fit(train, train["switch"])
        aug = make_classifier(augmented_numeric, categorical, c_aug).fit(train, train["switch"])
        y = test["switch"].to_numpy(int)
        loss_base = log_loss(y, base.predict_proba(test)[:, 1], labels=[0, 1])
        loss_aug = log_loss(y, aug.predict_proba(test)[:, 1], labels=[0, 1])
        rows.append({
            "heldout_incumbent_action": int(held_action),
            "eligible": 1,
            "n_test": len(test),
            "n_switch": n_switch,
            "n_persist": n_persist,
            "base_log_loss": float(loss_base),
            "augmented_log_loss": float(loss_aug),
            "delta_log_loss": float(loss_base - loss_aug),
        })
    return pd.DataFrame(rows)


def binned_surface(
    frame: pd.DataFrame,
    value_column: Optional[str],
    bins: int,
    minimum_count: int,
    x_column: str = "persistence_evidence",
    y_column: str = "switch_target_uncertainty",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = frame[x_column].to_numpy(float)
    y = frame[y_column].to_numpy(float)
    finite = np.isfinite(x) & np.isfinite(y)
    if value_column is not None:
        values = frame[value_column].to_numpy(float)
        finite &= np.isfinite(values)
    else:
        values = np.ones(len(frame), dtype=float)
    x, y, values = x[finite], y[finite], values[finite]
    if len(x) == 0:
        raise ValueError("No finite Persistence Evidence × Switch-Target Uncertainty rows")
    x_low, x_high = np.quantile(x, [0.01, 0.99])
    if not np.isfinite(x_low) or not np.isfinite(x_high) or x_low >= x_high:
        x_low, x_high = float(np.min(x)), float(np.max(x) + EPS)
    x_edges = np.linspace(x_low, x_high, bins + 1)
    y_edges = np.linspace(0.0, 1.0, bins + 1)
    x_bin = np.digitize(x, x_edges) - 1
    y_bin = np.digitize(y, y_edges) - 1
    valid_bin = (x_bin >= 0) & (x_bin < bins) & (y_bin >= 0) & (y_bin < bins)
    count = np.zeros((bins, bins), dtype=int)
    total = np.zeros((bins, bins), dtype=float)
    for xi, yi, value in zip(x_bin[valid_bin], y_bin[valid_bin], values[valid_bin]):
        count[yi, xi] += 1
        total[yi, xi] += value
    if value_column is None:
        surface = np.where(count >= minimum_count, np.log10(count + 1.0), np.nan)
    else:
        with np.errstate(invalid="ignore", divide="ignore"):
            surface = total / count
        surface[count < minimum_count] = np.nan
    return surface, x_edges, y_edges


def draw_surface(
    ax: plt.Axes,
    frame: pd.DataFrame,
    value_column: Optional[str],
    title: str,
    colorbar_label: str,
    bins: int,
    minimum_count: int,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    x_column: str = "persistence_evidence",
    y_column: str = "switch_target_uncertainty",
    x_label: str = "Model evidence for continuing the current action",
    y_label: str = "Model uncertainty among other actions",
) -> None:
    surface, x_edges, y_edges = binned_surface(
        frame, value_column, bins, minimum_count, x_column=x_column, y_column=y_column
    )
    mesh = ax.pcolormesh(x_edges, y_edges, surface, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)
    colorbar = ax.figure.colorbar(mesh, ax=ax, fraction=0.047, pad=0.03)
    colorbar.set_label(colorbar_label)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_ylim(0, 1)


def plot_actor_guidance_surfaces(
    frame: pd.DataFrame,
    output: Path,
    bins: int,
    minimum_count: int,
    dpi: int,
) -> None:
    panels = [
        ("actor_persistence_evidence", "Actor evidence for continuing", "Actor log evidence", "coolwarm", None, None),
        ("actor_switch_target_uncertainty", "Actor uncertainty among other actions", "Normalized entropy", "viridis", 0, 1),
        ("actor_model_candidate_support", "Actor support for model's preferred action", "Actor probability", "viridis", 0, 1),
        ("actor_model_js", "Model–actor policy divergence", "Jensen–Shannon divergence", "magma", 0, None),
        ("model_to_actor_persistence_shift", "Model-to-actor shift in continuation evidence", "Actor minus model", "RdBu_r", None, None),
        ("model_to_actor_uncertainty_shift", "Model-to-actor shift in uncertainty", "Actor minus model", "RdBu_r", None, None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for ax, (column, title, label, cmap, vmin, vmax) in zip(axes.flat, panels):
        if vmin is None and cmap == "RdBu_r":
            limit = float(np.nanquantile(np.abs(frame[column]), 0.98))
            vmin, vmax = -limit, limit
        draw_surface(
            ax, frame, column, title, label, bins, minimum_count,
            cmap=cmap, vmin=vmin, vmax=vmax,
        )
    fig.suptitle("Initial actor guidance within the model-defined control state", fontsize=17, x=0.5, ha="center")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_actor_allocation_surfaces(
    frame: pd.DataFrame,
    output: Path,
    bins: int,
    minimum_count: int,
    dpi: int,
) -> None:
    panels = [
        ("incumbent_start_share", "Search-bout starts on the current action", "Mean share", "viridis", 0, 1),
        ("incumbent_node_share", "Node budget assigned to the current action", "Mean share", "viridis", 0, 1),
        ("continuation_allocation", "Within-bout continuation allocation", "Node minus start log ratio", "RdBu_r", None, None),
        ("mean_bout_terminal_depth", "Mean bout terminal depth", "Mean depth", "viridis", None, None),
        ("alternative_branch_entropy", "Allocation breadth among other actions", "Normalized entropy", "viridis", 0, 1),
        ("reset_reorientation", "Branch change after an observed reset", "Probability", "viridis", 0, 1),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for ax, (column, title, label, cmap, vmin, vmax) in zip(axes.flat, panels):
        if cmap == "RdBu_r":
            limit = float(np.nanquantile(np.abs(frame[column]), 0.98))
            vmin, vmax = -limit, limit
        draw_surface(
            ax, frame, column, title, label, bins, minimum_count,
            cmap=cmap, vmin=vmin, vmax=vmax,
            x_column="actor_persistence_evidence",
            y_column="actor_switch_target_uncertainty",
            x_label="Actor evidence for continuing the current action",
            y_label="Actor uncertainty among other actions",
        )
    fig.suptitle("Fixed-budget allocation within the initial actor-guidance state", fontsize=17, x=0.5, ha="center")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_control_state(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    output: Path,
    bins: int,
    minimum_count: int,
    dpi: int,
) -> None:
    prediction = predictions.loc[predictions["test"] == "presearch_control"].copy()
    oof = frame.copy()
    oof["oof_augmented_probability"] = np.nan
    if len(prediction):
        mapped = prediction.set_index("row_index")["augmented_probability"]
        common = oof.index.intersection(mapped.index)
        oof.loc[common, "oof_augmented_probability"] = mapped.loc[common]
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.5), constrained_layout=True)
    draw_surface(axes[0], frame, None, "State occupancy", r"$\log_{10}(n+1)$", bins, minimum_count)
    draw_surface(axes[1], frame, "switch", "Observed switch probability", "Probability", bins, minimum_count, cmap="magma", vmin=0, vmax=1)
    draw_surface(axes[2], oof, "oof_augmented_probability", "Held-out predicted switch probability", "Probability", bins, minimum_count, cmap="magma", vmin=0, vmax=1)
    fig.suptitle("Pre-search action-control state and human switching", fontsize=17, x=0.5, ha="center")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _quantile_run_summary(
    frame: pd.DataFrame,
    x_column: str,
    value_column: str,
    bins: int = 5,
) -> pd.DataFrame:
    """Summarize a marginal relationship at the run level.

    The raw cache has one row per real step, but runs—not real steps—are the
    descriptive replication units in this analysis.  We therefore average
    within run and x-quantile first, then compute the across-run mean and CI.
    """
    required = {x_column, value_column, "run_label"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    data = frame.loc[:, [x_column, value_column, "run_label"]].copy()
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < bins:
        return pd.DataFrame()
    try:
        data["quantile"] = pd.qcut(data[x_column], q=bins, labels=False, duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    run_means = data.groupby(["run_label", "quantile"], observed=True)[value_column].mean().reset_index()
    summary = run_means.groupby("quantile", observed=True)[value_column].agg(["mean", "std", "count"]).reset_index()
    summary["se"] = summary["std"].fillna(0.0) / np.sqrt(summary["count"].clip(lower=1))
    summary["low"] = summary["mean"] - 1.96 * summary["se"]
    summary["high"] = summary["mean"] + 1.96 * summary["se"]
    return summary


def _draw_marginal_line(
    ax: plt.Axes,
    frame: pd.DataFrame,
    x_column: str,
    value_column: str,
    color: str,
    label: str,
    ylabel: str,
    bins: int = 5,
) -> None:
    summary = _quantile_run_summary(frame, x_column, value_column, bins)
    if summary.empty:
        ax.text(0.5, 0.5, "Insufficient finite data", ha="center", va="center", transform=ax.transAxes)
        return
    x = summary["quantile"].to_numpy(float) + 1.0
    y = summary["mean"].to_numpy(float)
    low = summary["low"].to_numpy(float)
    high = summary["high"].to_numpy(float)
    ax.plot(x, y, color=color, marker="o", lw=2.2, label=label)
    ax.fill_between(x, low, high, color=color, alpha=0.16)
    ax.set_xticks(np.arange(1, bins + 1))
    ax.set_xlabel("State-axis quintile (low → high)")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.22)


def _finish_linear_figure(fig: plt.Figure, output: Path, dpi: int, title: str) -> None:
    fig.suptitle(title, fontsize=17, x=0.5, ha="center")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_control_linear(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    output: Path,
    dpi: int,
) -> None:
    """Marginal linear view of pre-search state and switching."""
    plot_frame = frame.copy()
    plot_frame["heldout_switch"] = np.nan
    if len(predictions) and "row_index" in predictions and "augmented_probability" in predictions:
        # The prediction table contains one row per nested-CV test for the
        # same real step.  Restrict to the pre-search-control test before
        # indexing; otherwise row_index is intentionally duplicated across
        # the nested tests and pandas cannot align the assignment.
        if "test" in predictions.columns:
            prediction = predictions.loc[
                predictions["test"].eq("presearch_control")
            ].copy()
        else:
            prediction = predictions.copy()
        prediction = prediction.drop_duplicates("row_index", keep="last")
        mapped = prediction.set_index("row_index")["augmented_probability"]
        common = plot_frame.index.intersection(mapped.index)
        plot_frame.loc[common, "heldout_switch"] = mapped.loc[common]
    panels = [
        ("persistence_evidence", "switch", "Observed switch probability", "#4E79A7"),
        ("switch_target_uncertainty", "switch", "Observed switch probability", "#4E79A7"),
        ("persistence_evidence", "heldout_switch", "Held-out switch probability", "#F28E2B"),
        ("switch_target_uncertainty", "heldout_switch", "Held-out switch probability", "#F28E2B"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, (x_column, value_column, ylabel, color) in zip(axes.flat, panels):
        _draw_marginal_line(ax, plot_frame, x_column, value_column, color, value_column, ylabel)
        ax.set_title(f"{x_column.replace('_', ' ').title()} → {ylabel}")
    _finish_linear_figure(fig, output, dpi, "Marginal pre-search control and action switching")


def plot_allocation_linear(
    frame: pd.DataFrame,
    output: Path,
    dpi: int,
) -> None:
    panels = [
        ("incumbent_start_share", "Bout-start share"),
        ("incumbent_node_share", "Node-budget share"),
        ("continuation_allocation", "Continuation allocation"),
    ]
    predictors = [
        ("actor_persistence_evidence", "Actor persistence evidence", "#4E79A7"),
        ("actor_switch_target_uncertainty", "Actor switch-target uncertainty", "#F28E2B"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for row, (x_column, x_label, color) in enumerate(predictors):
        for ax, (value_column, title) in zip(axes[row], panels):
            _draw_marginal_line(ax, frame, x_column, value_column, color, x_label, title)
            ax.set_title(title)
    _finish_linear_figure(fig, output, dpi, "Marginal fixed-budget allocation across actor state axes")


def _plot_metric_linear_grid(
    frame: pd.DataFrame,
    output: Path,
    dpi: int,
    metrics: Sequence[Tuple[str, str]],
    title: str,
    x_columns: Sequence[Tuple[str, str, str]],
) -> None:
    fig, axes = plt.subplots(len(x_columns), len(metrics), figsize=(5.0 * len(metrics), 4.0 * len(x_columns)), constrained_layout=True, squeeze=False)
    for row, (x_column, x_label, color) in enumerate(x_columns):
        for col, (value_column, value_label) in enumerate(metrics):
            ax = axes[row, col]
            _draw_marginal_line(ax, frame, x_column, value_column, color, x_label, value_label)
            ax.set_title(value_label)
    _finish_linear_figure(fig, output, dpi, title)


def plot_mechanistic_linear(frame: pd.DataFrame, output: Path, dpi: int) -> None:
    _plot_metric_linear_grid(
        frame, output, dpi,
        [("actor_model_candidate_support", "Model-candidate support"),
         ("actor_model_js", "Actor–model JS divergence"),
         ("voluntary_reset_fraction", "Voluntary reset fraction"),
         ("voluntary_reset_reorientation", "Reset reorientation")],
        "Marginal mechanistic diagnostics",
        [("persistence_evidence", "Model persistence evidence", "#4E79A7"),
         ("switch_target_uncertainty", "Model switch-target uncertainty", "#F28E2B")],
    )


def plot_actor_guidance_linear(frame: pd.DataFrame, output: Path, dpi: int) -> None:
    _plot_metric_linear_grid(
        frame, output, dpi,
        [("actor_persistence_evidence", "Actor persistence evidence"),
         ("actor_switch_target_uncertainty", "Actor uncertainty"),
         ("actor_model_candidate_support", "Model-candidate support"),
         ("actor_model_js", "Actor–model JS divergence"),
         ("model_to_actor_persistence_shift", "Persistence shift"),
         ("model_to_actor_uncertainty_shift", "Uncertainty shift")],
        "Marginal model-to-actor guidance",
        [("persistence_evidence", "Model persistence evidence", "#4E79A7"),
         ("switch_target_uncertainty", "Model switch-target uncertainty", "#F28E2B")],
    )


def plot_actor_allocation_linear(frame: pd.DataFrame, output: Path, dpi: int) -> None:
    _plot_metric_linear_grid(
        frame, output, dpi,
        [("incumbent_start_share", "Bout-start share"),
         ("incumbent_node_share", "Node-budget share"),
         ("continuation_allocation", "Continuation allocation"),
         ("mean_bout_terminal_depth", "Mean terminal depth"),
         ("alternative_branch_entropy", "Alternative-branch entropy"),
         ("reset_reorientation", "Reset reorientation")],
        "Marginal actor-guided search allocation",
        [("actor_persistence_evidence", "Actor persistence evidence", "#4E79A7"),
         ("actor_switch_target_uncertainty", "Actor uncertainty", "#F28E2B")],
    )


def plot_allocation_surfaces(
    frame: pd.DataFrame,
    output: Path,
    bins: int,
    minimum_count: int,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.5), constrained_layout=True)
    draw_surface(axes[0], frame, "incumbent_start_share", "Incumbent bout-start share", "Mean share", bins, minimum_count, cmap="coolwarm", vmin=0, vmax=1)
    draw_surface(axes[1], frame, "incumbent_node_share", "Incumbent node-budget share", "Mean share", bins, minimum_count, cmap="coolwarm", vmin=0, vmax=1)
    absolute = np.nanquantile(np.abs(frame["continuation_allocation"]), 0.98)
    draw_surface(axes[2], frame, "continuation_allocation", "Relative continuation allocation", "Node minus start log ratio", bins, minimum_count, cmap="RdBu_r", vmin=-absolute, vmax=absolute)
    fig.suptitle("Fixed-budget allocation across the pre-search control state", fontsize=17, x=0.5, ha="center")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_nested_increments(scores: pd.DataFrame, output: Path, dpi: int) -> None:
    tests = [
        ("presearch_control", "Pre-search control increment", "M0", "M1"),
        ("model_evaluation_increment", "Add model evaluation", "Nuisance", "+ Model"),
        ("actor_guidance_increment", "Add initial actor guidance", "+ Model", "+ Actor"),
        ("allocation_increment", "Add realized search allocation", "+ Actor", "+ Allocation"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    for ax, (test_name, title, base_label, augmented_label) in zip(axes.flat, tests):
        group = scores.loc[scores["test"] == test_name].reset_index(drop=True)
        x_base, x_aug = 0.0, 1.0
        for index, row in group.iterrows():
            color = plt.cm.viridis(index / max(len(group) - 1, 1))
            ax.plot([x_base, x_aug], [row["base_log_loss"], row["augmented_log_loss"]], color=color, alpha=0.55, linewidth=1.2)
            ax.scatter([x_base, x_aug], [row["base_log_loss"], row["augmented_log_loss"]], color=color, s=25, zorder=3)
        if len(group):
            means = [group["base_log_loss"].mean(), group["augmented_log_loss"].mean()]
            ax.plot([x_base, x_aug], means, color="black", linewidth=3.0, marker="o", markersize=7, zorder=4)
            delta = group["delta_log_loss"].mean()
            ax.text(0.5, 0.98, f"Mean improvement = {delta:.4f} nats/opportunity", transform=ax.transAxes, ha="center", va="top")
        ax.set_xticks([0, 1], [base_label, augmented_label])
        ax.set_ylabel("Held-out log loss")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Nested held-out prediction across runs", fontsize=17, x=0.5, ha="center")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_mechanistic_surfaces(
    frame: pd.DataFrame,
    output: Path,
    bins: int,
    minimum_count: int,
    dpi: int,
) -> None:
    panels = [
        ("actor_model_candidate_support", "Actor support for model candidate", "Mean probability", "viridis", 0, 1),
        ("actor_model_js", "Actor–model policy divergence", "Jensen–Shannon divergence", "magma", 0, None),
        ("voluntary_reset_fraction", "Actor-selected reset fraction", "Resets per status-2 expansion", "cividis", 0, 1),
        ("voluntary_reset_reorientation", "Reset-conditioned reorientation", "Mean probability", "cividis", 0, 1),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 9.0), constrained_layout=True)
    for ax, (column, title, label, cmap, vmin, vmax) in zip(axes.flat, panels):
        draw_surface(ax, frame, column, title, label, bins, minimum_count, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.suptitle("Mechanistic search diagnostics", fontsize=17, x=0.5, ha="center")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def switch_destination_summary(frame: pd.DataFrame, quantiles: int = 5) -> pd.DataFrame:
    switched = frame.loc[frame["switch"].eq(1) & np.isfinite(frame["switch_target_uncertainty"])].copy()
    if len(switched) < max(quantiles * 5, 20):
        return pd.DataFrame()
    try:
        switched["uncertainty_bin"] = pd.qcut(
            switched["switch_target_uncertainty"], q=quantiles,
            labels=False, duplicates="drop",
        )
    except ValueError:
        return pd.DataFrame()
    switched["matches_model_candidate"] = (
        switched["human_action"] == switched["model_candidate"]
    ).astype(float)
    switched["matches_actor_candidate"] = np.where(
        switched["actor_initial_candidate"] >= 0,
        (switched["human_action"] == switched["actor_initial_candidate"]).astype(float),
        np.nan,
    )
    rows: List[Dict[str, object]] = []
    for bin_value, group in switched.groupby("uncertainty_bin", observed=True):
        destination_counts = np.bincount(
            group["human_action"].to_numpy(int),
            minlength=int(frame["human_action"].max()) + 1,
        )
        row: Dict[str, object] = {
            "uncertainty_bin": int(bin_value),
            "n_switches": len(group),
            "mean_switch_target_uncertainty": float(group["switch_target_uncertainty"].mean()),
            "human_destination_entropy": normalized_entropy(destination_counts),
        }
        for column in ("matches_model_candidate", "matches_actor_candidate"):
            values = group[column].dropna().to_numpy(float)
            row[f"{column}_n"] = len(values)
            row[f"{column}_proportion"] = float(np.mean(values)) if len(values) else np.nan
            if len(values):
                p = float(np.mean(values)); n = len(values); z = 1.96
                denominator = 1 + z * z / n
                center = (p + z * z / (2 * n)) / denominator
                half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
                row[f"{column}_ci_low"] = center - half
                row[f"{column}_ci_high"] = center + half
            else:
                row[f"{column}_ci_low"] = np.nan
                row[f"{column}_ci_high"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def plot_switch_destination(summary: pd.DataFrame, output: Path, dpi: int) -> None:
    if len(summary) == 0:
        return
    x = summary["mean_switch_target_uncertainty"].to_numpy(float)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    for column, label, color in (
        ("matches_model_candidate", "Model-policy candidate", "#3b6ea8"),
        ("matches_actor_candidate", "Initial actor candidate", "#d2691e"),
    ):
        y = summary[f"{column}_proportion"].to_numpy(float)
        low = summary[f"{column}_ci_low"].to_numpy(float)
        high = summary[f"{column}_ci_high"].to_numpy(float)
        axes[0].errorbar(x, y, yerr=np.vstack([y - low, high - y]), marker="o", linewidth=2, capsize=3, label=label, color=color)
    axes[0].set_xlabel("Switch-target uncertainty")
    axes[0].set_ylabel("Destination–candidate match probability")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Candidate consistency on switch trials")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)
    axes[1].plot(x, summary["human_destination_entropy"], marker="o", color="#5b8c5a", linewidth=2)
    axes[1].set_xlabel("Switch-target uncertainty")
    axes[1].set_ylabel("Human destination entropy")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Breadth of human switch destinations")
    axes[1].grid(alpha=0.25)
    fig.suptitle("Switch-target uncertainty and destination choice", fontsize=17, x=0.5, ha="center")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def run_analysis(
    frame: pd.DataFrame,
    subject: int,
    game: int,
    task_columns: Sequence[str],
    args: argparse.Namespace,
    output_dir: Path,
) -> Dict[str, object]:
    frame = add_actor_guidance_coordinates(frame)
    results_dir = output_dir / "results"
    figures_dir = output_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    model_ilr = sorted([column for column in frame if column.startswith("model_root_ilr_")], key=lambda x: int(x.rsplit("_", 1)[1]))
    actor_ilr = sorted([column for column in frame if column.startswith("actor_initial_ilr_")], key=lambda x: int(x.rsplit("_", 1)[1]))
    terminal_actor_ilr = sorted([column for column in frame if column.startswith("actor_terminal_ilr_")], key=lambda x: int(x.rsplit("_", 1)[1]))
    base_numeric = ["bout_age", "prior_reward", "prior_done", "run_progress", *task_columns]
    categorical = ["incumbent_action", "action_before_incumbent"]
    control_numeric = [
        *base_numeric,
        "persistence_evidence",
        "switch_target_uncertainty",
        "persistence_x_uncertainty",
    ]
    model_numeric = [*base_numeric, "root_value", *model_ilr]
    actor_numeric = [*model_numeric, *actor_ilr, "actor_model_js"]
    allocation_numeric = [*actor_numeric, "incumbent_start_logratio", "incumbent_node_logratio"]
    terminal_presearch_numeric = [*actor_numeric, *terminal_actor_ilr]
    terminal_allocation_numeric = [*allocation_numeric, *terminal_actor_ilr]

    all_scores: List[pd.DataFrame] = []
    all_predictions: List[pd.DataFrame] = []
    for name, base, augmented, require_valid_search in (
        ("presearch_control", base_numeric, control_numeric, False),
        ("model_evaluation_increment", base_numeric, model_numeric, False),
        ("actor_guidance_increment", model_numeric, actor_numeric, True),
        ("allocation_increment", actor_numeric, allocation_numeric, True),
        (
            "allocation_increment_terminal_policy_sensitivity",
            terminal_presearch_numeric,
            terminal_allocation_numeric,
            True,
        ),
    ):
        scores, predictions = nested_loro_pair(
            frame=frame,
            test_name=name,
            base_numeric=base,
            augmented_numeric=augmented,
            categorical=categorical,
            penalties=args.penalty_grid,
            inner_folds=args.inner_folds,
            require_valid_search=require_valid_search,
        )
        all_scores.append(scores)
        all_predictions.append(predictions)
    scores = pd.concat(all_scores, ignore_index=True) if all_scores else pd.DataFrame()
    predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    if len(scores) == 0:
        raise RuntimeError("Nested cross-validation produced no valid outer folds")

    summary = summarize_nested_scores(scores, subject, game)
    action_gate = action_generalization_gate(
        frame=frame,
        base_numeric=base_numeric,
        augmented_numeric=control_numeric,
        categorical=categorical,
        penalties=args.penalty_grid,
        inner_folds=args.inner_folds,
        min_outcomes=args.min_action_outcomes,
    )
    scores.insert(0, "game", game); scores.insert(0, "subject", subject)
    predictions.insert(0, "game", game); predictions.insert(0, "subject", subject)
    action_gate.insert(0, "game", game); action_gate.insert(0, "subject", subject)
    destination = switch_destination_summary(frame)
    if len(destination):
        destination.insert(0, "game", game); destination.insert(0, "subject", subject)
    scores.to_csv(results_dir / "nested_cv_outer_scores.csv", index=False)
    predictions.to_csv(results_dir / "nested_cv_outer_predictions.csv.gz", index=False, compression="gzip")
    summary.to_csv(results_dir / "nested_cv_summary.csv", index=False)
    action_gate.to_csv(results_dir / "leave_one_action_out_gate.csv", index=False)
    destination.to_csv(results_dir / "switch_destination_summary.csv", index=False)
    guidance_columns = [
        "subject", "session", "block", "game", "run_label", "real_step_id",
        "incumbent_action", "human_action", "switch",
        "persistence_evidence", "switch_target_uncertainty",
        "actor_persistence_evidence", "actor_switch_target_uncertainty",
        "model_to_actor_persistence_shift", "model_to_actor_uncertainty_shift",
        "actor_model_js", "actor_model_candidate_support",
    ]
    frame[guidance_columns].to_csv(
        results_dir / "model_actor_guidance_coordinates.csv.gz", index=False, compression="gzip"
    )

    plot_control_state(frame, predictions, figures_dir / "figure1_presearch_control_switch.png", args.figure_bins, args.min_bin_count, args.dpi)
    plot_control_linear(frame, predictions, figures_dir / "figure1_linear_control_switch.png", args.dpi)
    plot_allocation_surfaces(frame, figures_dir / "figure2_fixed_budget_allocation.png", args.figure_bins, args.min_bin_count, args.dpi)
    plot_allocation_linear(frame, figures_dir / "figure2_linear_fixed_budget_allocation.png", args.dpi)
    # The nested-increment analysis remains in CSV form for auditability, but
    # its former figure is intentionally omitted from the primary output.
    plot_mechanistic_surfaces(frame, figures_dir / "figure4_mechanistic_diagnostics.png", args.figure_bins, args.min_bin_count, args.dpi)
    plot_mechanistic_linear(frame, figures_dir / "figure4_linear_mechanistic_diagnostics.png", args.dpi)
    plot_switch_destination(destination, figures_dir / "figure5_switch_destination_uncertainty.png", args.dpi)
    plot_actor_guidance_surfaces(frame, figures_dir / "figure6_model_to_actor_guidance.png", args.figure_bins, args.min_bin_count, args.dpi)
    plot_actor_guidance_linear(frame, figures_dir / "figure6_linear_model_to_actor_guidance.png", args.dpi)
    plot_actor_allocation_surfaces(frame, figures_dir / "figure7_actor_guidance_allocation.png", args.figure_bins, args.min_bin_count, args.dpi)
    plot_actor_allocation_linear(frame, figures_dir / "figure7_linear_actor_guidance_allocation.png", args.dpi)

    analysis_manifest = {
        "subject": subject,
        "game": game,
        "development_status": args.sample_role,
        "task_state_columns": list(task_columns),
        "baseline_scope": "task-state augmented" if task_columns else "minimal behavioral baseline",
        "primary_tests": {
            "M0_to_M1": ["persistence_evidence", "switch_target_uncertainty", "persistence_x_uncertainty"],
            "nuisance_to_model": ["root_value", "model-policy ILR coordinates"],
            "model_to_actor": ["initial actor-policy ILR coordinates", "actor_model_js"],
            "actor_to_allocation": ["incumbent_start_logratio", "incumbent_node_logratio"],
        },
        "sensitivity_test": {
            "terminal_status0_actor_policy": (
                "added symmetrically to M2 and M3; downstream readout, not a primary baseline"
            )
        },
        "policy_coordinates": "isometric log-ratio (K-1 coordinates)",
        "outer_cv": "leave one run out",
        "inner_cv": f"grouped {args.inner_folds}-fold maximum",
        "penalty_grid": list(args.penalty_grid),
        "primary_score": "base minus augmented held-out log loss, nats/opportunity",
        "inference_unit": "participant",
        "independent_of_metric_development": args.sample_role == "prospective-replication",
    }
    save_json(results_dir / "analysis_manifest.json", analysis_manifest)
    return analysis_manifest


def run_self_test() -> None:
    p = np.asarray([0.7, 0.2, 0.1])
    coordinates = persistence_coordinates(p, 0)
    assert np.isclose(coordinates["persistence_evidence"], np.log(0.7 / 0.3))
    expected_entropy = -np.sum(np.asarray([2 / 3, 1 / 3]) * np.log([2 / 3, 1 / 3])) / np.log(2)
    assert np.isclose(coordinates["switch_target_uncertainty"], expected_entropy)
    batch = SearchBatch(
        actor_policy=np.tile(p, (4, 1)),
        paths=np.asarray([[0, -1], [0, 1], [1, -1], [1, 2]], dtype=np.int16),
        path_valid=np.ones(4, dtype=bool),
        depth=np.asarray([1, 2, 1, 2], dtype=np.int16),
        reset_code=np.asarray([0, 1, 0, 3], dtype=np.int16),
        leading_reset_code=0,
        leading_marker_count=0,
        reset_marker_count=2,
        reset_marker_collision_count=0,
        rollout_return=np.asarray([0, 1, 0, 2], dtype=float),
        n_status1=0, n_status2=4, n_status3=0,
    )
    allocation = allocation_metrics(batch, incumbent=0, n_actions=3, voluntary_reset_code=1, forced_reset_code=3)
    assert allocation["incumbent_start_count"] == 1
    assert allocation["alternative_start_count"] == 1
    assert allocation["incumbent_node_count"] == 2
    assert allocation["alternative_node_count"] == 2
    assert np.isclose(allocation["incumbent_start_logratio"], 0)
    assert np.isclose(allocation["incumbent_node_logratio"], 0)
    assert allocation["voluntary_reset_count"] == 1
    assert allocation["forced_reset_count"] == 1
    assert allocation["reset_boundary_count"] == 2
    # Public reset markers terminate the immediately preceding status-2 event.
    marker_status = np.asarray([2, 1, 2, 2, 3, 0], dtype=int)
    marker_indices = np.arange(5, dtype=int)
    marker_tree = {
        "action_seq": np.asarray([
            [1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0],
            [1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ], dtype=float),
        "rollout_return": np.zeros((5, 1), dtype=float),
    }
    marker_data = {"actor_policy": np.tile(p, (5, 1))}
    marker_batch = build_search_batch(
        marker_data, marker_tree, marker_status, marker_indices, n_actions=3
    )
    assert marker_batch.reset_code.tolist() == [1, 0, 3]
    assert marker_batch.reset_marker_count == 2
    first_chunk = SearchBatch(
        actor_policy=np.tile(p, (1, 1)),
        paths=np.asarray([[0]], dtype=np.int16),
        path_valid=np.ones(1, dtype=bool),
        depth=np.ones(1, dtype=np.int16),
        reset_code=np.zeros(1, dtype=np.int16),
        leading_reset_code=0,
        leading_marker_count=0,
        reset_marker_count=0,
        reset_marker_collision_count=0,
        rollout_return=np.zeros(1),
        n_status1=0, n_status2=1, n_status3=0,
    )
    second_tree = {
        "action_seq": np.asarray([
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0],
        ], dtype=float),
        "rollout_return": np.zeros((2, 1), dtype=float),
    }
    second_data = {"actor_policy": np.tile(p, (2, 1))}
    second_chunk = build_search_batch(
        second_data, second_tree, np.asarray([3, 2]), np.arange(2), n_actions=3
    )
    combined = concatenate_batches(first_chunk, second_chunk)
    assert combined is not None
    assert combined.reset_code.tolist() == [3, 0]
    assert combined.leading_reset_code == 0
    assert combined.reset_marker_count == 1
    synthetic = pd.DataFrame({
        "bout_age": np.arange(40) % 6 + 1,
        "prior_reward": 0.0,
        "prior_done": 0.0,
        "run_progress": np.linspace(0, 1, 40),
        "incumbent_action": np.arange(40) % 3,
        "action_before_incumbent": (np.arange(40) + 1) % 3,
        "switch": (np.arange(40) % 4 == 0).astype(int),
    })
    model = make_classifier(["bout_age", "prior_reward", "prior_done", "run_progress"], ["incumbent_action", "action_before_incumbent"], 1.0)
    model.fit(synthetic, synthetic["switch"])
    assert model.predict_proba(synthetic).shape == (40, 2)
    print("Self-test passed: policy coordinates, allocation, and CV preprocessing.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=parse_subject)
    parser.add_argument("--game", type=int)
    parser.add_argument("--sessions", type=str, default=None, help="Comma-separated session ids")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--shared-cache-root",
        type=Path,
        default=DEFAULT_SHARED_CACHE_ROOT,
        help="Read-only cache produced by 00_cache.py.",
    )
    parser.add_argument("--max-files", type=int, default=None, help="Debug: maximum chunks per block")
    parser.add_argument("--task-state-csv", type=Path, default=None)
    parser.add_argument("--task-state-columns", type=str, default="", help="Comma-separated numeric task-state columns")
    parser.add_argument("--voluntary-reset-code", type=int, default=1)
    parser.add_argument("--forced-reset-code", type=int, default=3)
    parser.add_argument("--penalty-grid", type=str, default="0.01,0.1,1,10,100")
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--min-action-outcomes", type=int, default=25)
    parser.add_argument("--figure-bins", type=int, default=35)
    parser.add_argument("--min-bin-count", type=int, default=40)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--sample-role", choices=("development", "prospective-replication"), default="development")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.subject is None or args.game is None:
        raise SystemExit("--subject and --game are required unless --self-test is used")
    try:
        args.penalty_grid = [float(value) for value in args.penalty_grid.split(",") if value.strip()]
    except ValueError as exc:
        raise SystemExit(f"Invalid --penalty-grid: {exc}") from exc
    if not args.penalty_grid or any(value <= 0 for value in args.penalty_grid):
        raise SystemExit("--penalty-grid must contain positive values")
    task_columns_requested = [value.strip() for value in args.task_state_columns.split(",") if value.strip()]
    sessions = parse_sessions(args.sessions)
    output_dir = args.output_root / f"sub{args.subject:03d}_game{args.game}"
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
    frame, cache_manifest = load_model_control(shared_paths)
    if sessions is not None:
        frame = frame[
            pd.to_numeric(frame["session"], errors="coerce").isin(sessions)
        ].copy()
    cache_manifest = {
        **cache_manifest,
        "shared_cache_path": str(shared_paths.root),
        "shared_cache_version": shared_manifest["shared_cache_version"],
    }
    frame, task_columns = merge_task_state(frame, args.task_state_csv, task_columns_requested)
    qa = qa_summary(frame, cache_manifest, task_columns)
    save_json(output_dir / "results" / "qa_summary.json", qa)
    analysis_manifest = run_analysis(
        frame=frame,
        subject=args.subject,
        game=args.game,
        task_columns=task_columns,
        args=args,
        output_dir=output_dir,
    )
    log(
        f"Complete: {output_dir} | baseline={analysis_manifest['baseline_scope']} | "
        f"sample role={args.sample_role}"
    )


if __name__ == "__main__":
    main()
