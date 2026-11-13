#!/usr/bin/env python3
"""Rollout value and allocation within an action-general control geometry.

This analysis keeps the temporal roles separate:

1. the preceding real-step root model policy defines pre-search control state;
2. status-2 rows record raw discounted/bootstrapped rollout returns and paths;
3. the following status-2 operation and the next human action are outcomes.

The nominal search budget is fixed (normally 29 status-2 rows per real-step
window), so the analysis concerns allocation, not total computation.  Raw
rollout return is primary.  Within-current-depth rank is a sensitivity check;
return is never residualized by final search depth.

Window and prefix-event tables are read from the shared cache produced by
``00_cache.py``; this script does not create a private cache.
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
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

from shared_analysis_cache import (
    DEFAULT_SHARED_CACHE_ROOT,
    cache_paths as shared_cache_paths,
    load_rollout_value,
    validate_manifest as validate_shared_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = ROOT / "research_script" / "05_persistence_uncertainty_allocation.py"
DEFAULT_TRACE_ROOT = ROOT / "test"
DEFAULT_OUTPUT_ROOT = ROOT / "research_script" / "outputs" / "05_rollout_value_allocation"
REAL_STATUS = 0
SEARCH_STATUS = 2
WINDOW_STATUSES = (1, 2, 3)
EPS = 1e-8
CACHE_SCHEMA_VERSION = 2
BLUE = "#4E79A7"
ORANGE = "#F28E2B"
GOLD = "#EDC948"
PURPLE = "#7B61A8"
CHARCOAL = "#404040"


def _load_base():
    spec = importlib.util.spec_from_file_location("persistence_allocation_05", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_base()
TRACE = BASE.TRACE


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def parse_subject(value: object) -> int:
    match = re.search(r"\d+", str(value))
    if not match:
        raise argparse.ArgumentTypeError(f"Cannot parse subject: {value}")
    return int(match.group(0))


def parse_sessions(value: Optional[str]) -> Optional[set[int]]:
    if value is None or not value.strip():
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def scalar_at(value: object, index: int) -> float:
    if value is None:
        return np.nan
    try:
        row = np.asarray(value[index], dtype=float).reshape(-1)
    except (IndexError, TypeError, ValueError):
        return np.nan
    finite = row[np.isfinite(row)]
    return float(finite[0]) if finite.size else np.nan


def action_row(container: Mapping[str, object], key: str, index: int, n_actions: int) -> np.ndarray:
    if key not in container:
        return np.full(n_actions, np.nan)
    try:
        row = np.asarray(container[key][index], dtype=float).reshape(-1)
    except (IndexError, TypeError, ValueError):
        return np.full(n_actions, np.nan)
    return row if row.size == n_actions else np.full(n_actions, np.nan)


@dataclass
class RootState:
    model_policy: np.ndarray
    root_value: float
    human_action: int
    preceding_action: int
    reward: float
    done: float
    bout_age: int


@dataclass
class EventBatch:
    paths: np.ndarray
    path_valid: np.ndarray
    depth: np.ndarray
    rollout_return: np.ndarray
    max_rollout_return: np.ndarray
    actor_policy: np.ndarray
    model_policy: np.ndarray
    current_action: np.ndarray
    current_value: np.ndarray
    reset_after: np.ndarray
    leading_reset_code: int
    leading_marker_count: int
    marker_count: int
    marker_collision_count: int
    n_status1: int
    n_status2: int
    n_status3: int


def empty_batch(n_actions: int, width: int = 0) -> EventBatch:
    return EventBatch(
        paths=np.full((0, width), -1, dtype=np.int8),
        path_valid=np.zeros(0, dtype=bool),
        depth=np.zeros(0, dtype=np.int8),
        rollout_return=np.zeros(0, dtype=np.float32),
        max_rollout_return=np.zeros(0, dtype=np.float32),
        actor_policy=np.full((0, n_actions), np.nan, dtype=np.float32),
        model_policy=np.full((0, n_actions), np.nan, dtype=np.float32),
        current_action=np.full(0, -1, dtype=np.int8),
        current_value=np.zeros(0, dtype=np.float32),
        reset_after=np.zeros(0, dtype=np.int8),
        leading_reset_code=0,
        leading_marker_count=0,
        marker_count=0,
        marker_collision_count=0,
        n_status1=0,
        n_status2=0,
        n_status3=0,
    )


def build_event_batch(
    data: Mapping[str, object], tree: Mapping[str, object], status: np.ndarray,
    indices: np.ndarray, n_actions: int,
) -> EventBatch:
    indices = np.asarray(indices, dtype=int)
    if indices.size:
        indices = indices[np.isin(status[indices], WINDOW_STATUSES)]
    counts = {code: int(np.sum(status[indices] == code)) for code in WINDOW_STATUSES}
    selected = indices[status[indices] == SEARCH_STATUS]
    markers = indices[np.isin(status[indices], (1, 3))]
    reset_after = np.zeros(len(selected), dtype=np.int8)
    leading = 0
    leading_count = 0
    collisions = 0
    for marker in markers:
        code = int(status[marker])
        previous = int(np.searchsorted(selected, marker, side="left") - 1)
        if previous >= 0:
            collisions += int(reset_after[previous] > 0)
            reset_after[previous] = 3 if code == 3 or reset_after[previous] == 3 else 1
        else:
            leading_count += 1
            collisions += int(leading > 0)
            leading = 3 if code == 3 or leading == 3 else 1

    action_seq = np.asarray(tree.get("action_seq", []))
    width = (
        int(action_seq.shape[-1] // n_actions - 1)
        if action_seq.ndim >= 2 and action_seq.shape[-1] % n_actions == 0
        else 0
    )
    if selected.size == 0:
        out = empty_batch(n_actions, width)
        out.leading_reset_code = leading
        out.leading_marker_count = leading_count
        out.marker_count = int(len(markers))
        out.marker_collision_count = collisions
        out.n_status1, out.n_status2, out.n_status3 = counts[1], counts[2], counts[3]
        return out
    if action_seq.ndim < 2 or action_seq.shape[-1] % n_actions:
        raise ValueError("tree_reps.action_seq has an invalid action-slot layout")

    slots = np.asarray(action_seq[selected], dtype=np.float32).reshape(len(selected), -1, n_actions)
    occupied = np.sum(np.abs(slots), axis=2) > 0.5
    consecutive = np.cumprod(occupied.astype(np.int8), axis=1).astype(bool)
    occupied_count = np.sum(occupied, axis=1)
    valid = occupied[:, 0] & (occupied_count >= 2) & (np.sum(consecutive, axis=1) == occupied_count)
    depth = np.where(valid, occupied_count - 1, 0).astype(np.int8)
    paths = np.argmax(slots[:, 1:, :], axis=2).astype(np.int8)
    positions = np.arange(paths.shape[1])[None, :]
    paths[positions >= depth[:, None]] = -1
    paths[~valid] = -1

    def selected_scalar(source: object) -> np.ndarray:
        out = np.full(len(selected), np.nan, dtype=np.float32)
        if source is None:
            return out
        arr = np.asarray(source)
        if arr.ndim >= 1 and len(arr) > int(selected.max()):
            raw = np.asarray(arr[selected], dtype=np.float32).reshape(len(selected), -1)
            out = raw[:, 0]
            out[~np.isfinite(out)] = np.nan
        return out

    def selected_policy(source: object) -> np.ndarray:
        output = np.full((len(selected), n_actions), np.nan, dtype=np.float32)
        if source is not None:
            arr = np.asarray(source)
            if arr.ndim >= 2 and len(arr) > int(selected.max()):
                raw = np.asarray(arr[selected], dtype=np.float32).reshape(len(selected), -1)
                if raw.shape[1] == n_actions:
                    output = raw
        return output

    actor = selected_policy(data.get("actor_policy"))
    model = selected_policy(tree.get("cur_policy"))
    current_action = np.full(len(selected), -1, dtype=np.int8)
    source = tree.get("cur_action")
    if source is not None:
        arr = np.asarray(source)
        if arr.ndim >= 2 and len(arr) > int(selected.max()):
            raw = np.asarray(arr[selected], dtype=np.float32).reshape(len(selected), -1)
            if raw.shape[1] == n_actions:
                valid_action = np.sum(np.abs(raw), axis=1) > 0.5
                current_action[valid_action] = np.argmax(raw[valid_action], axis=1).astype(np.int8)
    fallback = valid & (depth > 0) & (current_action < 0)
    if np.any(fallback):
        rows = np.flatnonzero(fallback)
        current_action[rows] = paths[rows, depth[rows] - 1]

    return EventBatch(
        paths=paths,
        path_valid=valid,
        depth=depth,
        rollout_return=selected_scalar(tree.get("rollout_return")),
        max_rollout_return=selected_scalar(tree.get("max_rollout_return")),
        actor_policy=actor,
        model_policy=model,
        current_action=current_action,
        current_value=selected_scalar(tree.get("cur_v")),
        reset_after=reset_after,
        leading_reset_code=leading,
        leading_marker_count=leading_count,
        marker_count=int(len(markers)),
        marker_collision_count=collisions,
        n_status1=counts[1],
        n_status2=counts[2],
        n_status3=counts[3],
    )


def concatenate_batches(first: Optional[EventBatch], second: Optional[EventBatch]) -> Optional[EventBatch]:
    if first is None:
        return second
    if second is None:
        return first
    width = max(first.paths.shape[1], second.paths.shape[1])

    def pad(batch: EventBatch) -> np.ndarray:
        if batch.paths.shape[1] == width:
            return batch.paths
        out = np.full((batch.n_status2, width), -1, dtype=np.int8)
        out[:, : batch.paths.shape[1]] = batch.paths
        return out

    first_reset = first.reset_after.copy()
    leading = first.leading_reset_code
    leading_count = first.leading_marker_count + second.leading_marker_count
    collisions = first.marker_collision_count + second.marker_collision_count
    if second.leading_reset_code > 0 and first.n_status2 > 0:
        # The first marker in the next chunk closes the preceding chunk's
        # final status-2 event and is therefore mapped across the boundary.
        leading_count = max(0, leading_count - 1)
        collisions += int(first_reset[-1] > 0)
        first_reset[-1] = 3 if second.leading_reset_code == 3 or first_reset[-1] == 3 else 1
    elif second.leading_reset_code > 0:
        collisions += int(leading > 0)
        leading = 3 if second.leading_reset_code == 3 or leading == 3 else 1
    return EventBatch(
        paths=np.vstack([pad(first), pad(second)]),
        path_valid=np.concatenate([first.path_valid, second.path_valid]),
        depth=np.concatenate([first.depth, second.depth]),
        rollout_return=np.concatenate([first.rollout_return, second.rollout_return]),
        max_rollout_return=np.concatenate([first.max_rollout_return, second.max_rollout_return]),
        actor_policy=np.vstack([first.actor_policy, second.actor_policy]),
        model_policy=np.vstack([first.model_policy, second.model_policy]),
        current_action=np.concatenate([first.current_action, second.current_action]),
        current_value=np.concatenate([first.current_value, second.current_value]),
        reset_after=np.concatenate([first_reset, second.reset_after]),
        leading_reset_code=leading,
        leading_marker_count=leading_count,
        marker_count=first.marker_count + second.marker_count,
        marker_collision_count=collisions,
        n_status1=first.n_status1 + second.n_status1,
        n_status2=first.n_status2 + second.n_status2,
        n_status3=first.n_status3 + second.n_status3,
    )


def make_root(
    data: Mapping[str, object], tree: Mapping[str, object], index: int,
    n_actions: int, previous: Optional[RootState],
) -> RootState:
    action = TRACE.decode_action(data.get("human_action"), index)
    age = previous.bout_age + 1 if previous is not None and action == previous.human_action else 1
    raw_policy = action_row(tree, "root_policy", index, n_actions)
    policy = BASE.policy_probabilities(raw_policy, n_actions)
    root_value = scalar_at(tree.get("root_v"), index)
    if not np.isfinite(root_value):
        root_value = scalar_at(tree.get("root_vs"), index)
    return RootState(
        model_policy=policy,
        root_value=root_value,
        human_action=action,
        preceding_action=(previous.human_action if previous is not None else -1),
        reward=scalar_at(data.get("cur_rewards"), index),
        done=scalar_at(tree.get("root_d"), index),
        bout_age=age,
    )


def normalized_effective_branch_diversity(branches: np.ndarray, n_actions: int) -> float:
    branches = np.asarray(branches, dtype=int)
    branches = branches[(branches >= 0) & (branches < n_actions)]
    if branches.size == 0 or n_actions <= 1:
        return np.nan
    counts = np.bincount(branches, minlength=n_actions).astype(float)
    p = counts[counts > 0] / counts.sum()
    effective = float(np.exp(-np.sum(p * np.log(p))))
    return (effective - 1.0) / (n_actions - 1.0)


def rank_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if np.sum(finite) < 2:
        return np.nan
    rx, ry = rankdata(x[finite]), rankdata(y[finite])
    if np.std(rx) <= EPS or np.std(ry) <= EPS:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def path_extends(current: np.ndarray, current_depth: int, following: np.ndarray, following_depth: int) -> bool:
    if current_depth <= 0 or following_depth <= current_depth:
        return False
    return bool(np.array_equal(current[:current_depth], following[:current_depth]))


EVENT_KEYS = (
    "window_index", "event_index", "root_branch", "current_depth", "return_advantage",
    "running_max_gain", "branch_relative_return", "event_progress", "pair_valid",
    "same_branch_next", "branch_switch_next", "same_path_deepen_next", "reset_after",
    "next_return_change", "next_depth_change",
    "next_actor_model_js", "next_actor_chosen_probability",
    "model_probability_for_next_action", "next_action_matches_model_recommendation",
    "next_path_first_observation", "next_path_revisit",
)


def policy_matrix_probabilities(values: np.ndarray, n_actions: int) -> np.ndarray:
    """Convert a matrix of stored logits/probabilities row-wise."""
    values = np.asarray(values, dtype=float).reshape(-1, n_actions)
    output = np.full_like(values, np.nan, dtype=float)
    finite = np.all(np.isfinite(values), axis=1)
    if not np.any(finite):
        return output
    x = values[finite]
    probability_like = (
        (np.min(x, axis=1) >= -1e-6)
        & (np.max(x, axis=1) <= 1.0 + 1e-6)
        & (np.abs(np.sum(x, axis=1) - 1.0) < 1e-3)
    )
    converted = np.empty_like(x)
    if np.any(probability_like):
        p = np.clip(x[probability_like], EPS, None)
        converted[probability_like] = p / np.sum(p, axis=1, keepdims=True)
    if np.any(~probability_like):
        logits = x[~probability_like] - np.max(x[~probability_like], axis=1, keepdims=True)
        exp_logits = np.exp(np.clip(logits, -50, 50))
        converted[~probability_like] = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    output[finite] = converted
    return output


def window_and_events(
    meta: object, real_step: int, root: RootState, next_action: int,
    batch: EventBatch, n_actions: int, window_index: int,
) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    n = batch.n_status2
    branches = np.where(batch.path_valid & (batch.depth > 0), batch.paths[:, 0], -1).astype(np.int8)
    returns = batch.rollout_return.astype(float)
    finite_return = np.isfinite(returns)
    valid = batch.path_valid & finite_return & (branches >= 0)
    coordinates = BASE.persistence_coordinates(root.model_policy, root.human_action)
    actor_probabilities = policy_matrix_probabilities(batch.actor_policy, n_actions)
    model_probabilities = policy_matrix_probabilities(batch.model_policy, n_actions)

    # A path is labelled by whether it has already appeared earlier in the
    # same real-step search window.  This is an observed-path distinction, not
    # a claim that the underlying cModel node was newly expanded.
    first_observation = np.zeros(n, dtype=np.int8)
    seen_paths = set()
    for k in range(n):
        if not batch.path_valid[k] or batch.depth[k] <= 0:
            continue
        key = tuple(int(value) for value in batch.paths[k, : int(batch.depth[k])])
        first_observation[k] = int(key not in seen_paths)
        seen_paths.add(key)

    # Prefix variables use information available at event k to predict k+1.
    m = max(n - 1, 0)
    branch_relative = np.full(n, np.nan, dtype=float)
    sums = np.zeros(n_actions, dtype=float); counts = np.zeros(n_actions, dtype=int)
    for k in range(n):
        branch = int(branches[k])
        if branch < 0 or not np.isfinite(returns[k]):
            continue
        other = counts > 0
        other[branch] = False
        if np.any(other):
            branch_relative[k] = returns[k] - float(np.mean(sums[other] / counts[other]))
        sums[branch] += returns[k]; counts[branch] += 1

    pair_valid = np.zeros(m, dtype=np.int8)
    same_branch = np.zeros(m, dtype=np.int8)
    branch_switch = np.zeros(m, dtype=np.int8)
    deepen = np.zeros(m, dtype=np.int8)
    next_return_change = np.full(m, np.nan, dtype=np.float32)
    next_depth_change = np.full(m, np.nan, dtype=np.float32)
    next_actor_model_js = np.full(m, np.nan, dtype=np.float32)
    next_actor_chosen_probability = np.full(m, np.nan, dtype=np.float32)
    model_probability_for_next_action = np.full(m, np.nan, dtype=np.float32)
    next_action_matches_model = np.zeros(m, dtype=np.int8)
    for k in range(m):
        if batch.path_valid[k] and batch.path_valid[k + 1]:
            pair_valid[k] = 1
            same_branch[k] = int(branches[k] == branches[k + 1])
            branch_switch[k] = int(branches[k] != branches[k + 1])
            deepen[k] = int(
                batch.reset_after[k] == 0
                and path_extends(batch.paths[k], int(batch.depth[k]), batch.paths[k + 1], int(batch.depth[k + 1]))
            )
            next_depth_change[k] = float(batch.depth[k + 1] - batch.depth[k])
        if np.isfinite(returns[k]) and np.isfinite(returns[k + 1]):
            next_return_change[k] = float(returns[k + 1] - returns[k])
        next_action = int(batch.current_action[k + 1])
        if (
            0 <= next_action < n_actions
            and np.all(np.isfinite(model_probabilities[k]))
            and np.all(np.isfinite(actor_probabilities[k + 1]))
        ):
            next_actor_model_js[k] = BASE.jensen_shannon(
                model_probabilities[k], actor_probabilities[k + 1]
            )
            next_actor_chosen_probability[k] = actor_probabilities[k + 1, next_action]
            model_probability_for_next_action[k] = model_probabilities[k, next_action]
            next_action_matches_model[k] = int(next_action == int(np.argmax(model_probabilities[k])))

    running_max = np.maximum.accumulate(np.where(finite_return, returns, -np.inf))
    running_max[~np.isfinite(running_max)] = np.nan
    root_value = root.root_value
    return_advantage = returns - root_value if np.isfinite(root_value) else np.full(n, np.nan)
    running_max_gain = running_max - root_value if np.isfinite(root_value) else np.full(n, np.nan)

    event_arrays = {
        "window_index": np.full(m, window_index, dtype=np.int32),
        "event_index": np.arange(m, dtype=np.int8),
        "root_branch": branches[:m].astype(np.int8),
        "current_depth": batch.depth[:m].astype(np.int8),
        "return_advantage": return_advantage[:m].astype(np.float32),
        "running_max_gain": running_max_gain[:m].astype(np.float32),
        "branch_relative_return": branch_relative[:m].astype(np.float32),
        "event_progress": (np.arange(m) / max(m - 1, 1)).astype(np.float32),
        "pair_valid": pair_valid,
        "same_branch_next": same_branch,
        "branch_switch_next": branch_switch,
        "same_path_deepen_next": deepen,
        "reset_after": (batch.reset_after[:m] > 0).astype(np.int8),
        "next_return_change": next_return_change,
        "next_depth_change": next_depth_change,
        "next_actor_model_js": next_actor_model_js,
        "next_actor_chosen_probability": next_actor_chosen_probability,
        "model_probability_for_next_action": model_probability_for_next_action,
        "next_action_matches_model_recommendation": next_action_matches_model,
        "next_path_first_observation": first_observation[1:].astype(np.int8),
        "next_path_revisit": (1 - first_observation[1:]).astype(np.int8),
    }

    branch_means = np.full(n_actions, np.nan)
    branch_visits = np.zeros(n_actions, dtype=int)
    for action in range(n_actions):
        mask = valid & (branches == action)
        branch_visits[action] = int(np.sum(mask))
        if np.any(mask):
            branch_means[action] = float(np.mean(returns[mask]))
    observed = np.isfinite(branch_means)
    n_observed = int(np.sum(observed))
    best_branch = int(np.nanargmax(branch_means)) if n_observed else -1
    other_actions = np.arange(n_actions) != root.human_action
    other_values = branch_means[other_actions & observed]
    current_value = (
        float(branch_means[root.human_action])
        if 0 <= root.human_action < n_actions and np.isfinite(branch_means[root.human_action])
        else np.nan
    )
    best_other = float(np.max(other_values)) if other_values.size else np.nan

    ends = np.flatnonzero(batch.reset_after > 0).tolist()
    if n and (not ends or ends[-1] != n - 1):
        ends.append(n - 1)
    terminal_depths = batch.depth[ends] if ends else np.asarray([], dtype=float)
    actor_initial = (
        actor_probabilities[0]
        if n else np.full(n_actions, np.nan)
    )
    actor_candidate = int(np.argmax(actor_initial)) if np.all(np.isfinite(actor_initial)) else -1
    model_candidate = int(np.argmax(root.model_policy)) if np.all(np.isfinite(root.model_policy)) else -1

    record: Dict[str, object] = {
        "subject": int(meta.subject), "session": int(meta.session), "block": int(meta.block),
        "game": int(meta.game), "run_label": str(meta.run_label), "real_step": int(real_step),
        "real_step_id": f"{meta.run_label}_real{real_step:06d}",
        "maintained_action": int(root.human_action), "action_before_maintained": int(root.preceding_action),
        "human_action": int(next_action), "switch": int(next_action != root.human_action),
        "bout_age": int(root.bout_age), "prior_reward": root.reward, "prior_done": root.done,
        "root_value": root_value, "persistence_evidence": coordinates["persistence_evidence"],
        "switch_target_uncertainty": coordinates["switch_target_uncertainty"],
        "persistence_x_uncertainty": coordinates["persistence_evidence"] * coordinates["switch_target_uncertainty"],
        "n_status1": batch.n_status1, "n_status2": batch.n_status2, "n_status3": batch.n_status3,
        "reset_boundary_count": int(np.sum(batch.reset_after > 0)),
        "leading_marker": int(batch.leading_reset_code > 0),
        "leading_marker_count": batch.leading_marker_count,
        "mapped_marker_count": int(batch.marker_count - batch.leading_marker_count),
        "marker_count": batch.marker_count,
        "marker_collision_count": batch.marker_collision_count,
        "n_valid_expansions": int(np.sum(batch.path_valid)),
        "branch_diversity": normalized_effective_branch_diversity(branches[batch.path_valid], n_actions),
        "mean_bout_terminal_depth": float(np.mean(terminal_depths)) if terminal_depths.size else np.nan,
        "deep_node_fraction": float(np.mean(batch.depth[batch.path_valid] >= 3)) if np.any(batch.path_valid) else np.nan,
        "branch_switch_fraction": float(np.mean(branch_switch[pair_valid > 0])) if np.any(pair_valid) else np.nan,
        "reset_fraction": float(np.mean(batch.reset_after > 0)) if n else np.nan,
        "rollout_return_mean": float(np.mean(returns[finite_return])) if np.any(finite_return) else np.nan,
        "rollout_return_sd": float(np.std(returns[finite_return])) if np.any(finite_return) else np.nan,
        "rollout_return_first": float(returns[np.flatnonzero(finite_return)[0]]) if np.any(finite_return) else np.nan,
        "rollout_return_final": float(returns[np.flatnonzero(finite_return)[-1]]) if np.any(finite_return) else np.nan,
        "search_return_change": (
            float(returns[np.flatnonzero(finite_return)[-1]] - returns[np.flatnonzero(finite_return)[0]])
            if np.sum(finite_return) >= 2 else np.nan
        ),
        "observed_max_return": float(np.max(returns[finite_return])) if np.any(finite_return) else np.nan,
        "final_running_max_return": (
            float(batch.max_rollout_return[np.flatnonzero(np.isfinite(batch.max_rollout_return))[-1]])
            if np.any(np.isfinite(batch.max_rollout_return)) else np.nan
        ),
        "max_value_gain": (
            float(np.max(returns[finite_return]) - root_value)
            if np.any(finite_return) and np.isfinite(root_value) else np.nan
        ),
        "mean_absolute_value_discrepancy": (
            float(np.mean(np.abs(returns[finite_return] - root_value)))
            if np.any(finite_return) and np.isfinite(root_value) else np.nan
        ),
        "n_return_branches": n_observed,
        "branch_return_spread": float(np.std(branch_means[observed])) if n_observed >= 2 else np.nan,
        "maintained_branch_return": current_value, "best_other_branch_return": best_other,
        "maintained_minus_best_other_return": (
            current_value - best_other if np.isfinite(current_value) and np.isfinite(best_other) else np.nan
        ),
        "best_return_branch": best_branch,
        "best_branch_visit_share": float(branch_visits[best_branch] / np.sum(branch_visits)) if best_branch >= 0 else np.nan,
        "value_allocation_alignment": rank_correlation(branch_means[observed], branch_visits[observed]),
        "model_candidate": model_candidate, "actor_initial_candidate": actor_candidate,
        "actor_model_js": BASE.jensen_shannon(root.model_policy, actor_initial),
        "value_metric_coverage": int(n_observed >= 2 and np.isfinite(root_value)),
    }
    model_ilr = BASE.ilr_coordinates(root.model_policy)
    actor_ilr = BASE.ilr_coordinates(actor_initial)
    for index in range(n_actions - 1):
        record[f"model_root_ilr_{index}"] = float(model_ilr[index])
        record[f"actor_initial_ilr_{index}"] = float(actor_ilr[index])
    return record, event_arrays


def trace_signature(blocks: Mapping[object, Sequence[object]]) -> List[Dict[str, object]]:
    rows = []
    for metas in blocks.values():
        for meta in metas:
            stat = meta.path.stat()
            rows.append({"path": str(meta.path.resolve()), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)})
    return rows


def save_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def extract_rollout_tables(
    blocks: Mapping[object, Sequence[object]], max_files: Optional[int],
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, object]]:
    """Extract rollout tables in memory.

    Cache persistence is owned exclusively by ``00_cache.py``.  Keeping this
    function side-effect free also makes it usable by the shared-cache builder
    without creating a second analysis-specific copy.
    """
    windows: List[Dict[str, object]] = []
    event_lists: Dict[str, List[np.ndarray]] = {key: [] for key in EVENT_KEYS}
    n_actions: Optional[int] = None
    loaded_files = 0

    for block_number, metas_all in enumerate(blocks.values(), start=1):
        metas = list(metas_all[:max_files] if max_files is not None else metas_all)
        previous_root: Optional[RootState] = None
        carry: Optional[EventBatch] = None
        real_step = 0
        log(f"Block {block_number}/{len(blocks)}: {len(metas)} chunks")
        for chunk_number, meta in enumerate(metas, start=1):
            size_gib = meta.path.stat().st_size / 1024**3
            log(f"Load {meta.path.name} ({size_gib:.1f} GiB; {chunk_number}/{len(metas)})")
            data = TRACE.load_npy_dict(meta.path); loaded_files += 1
            status = np.asarray(data.get("status", []), dtype=int).reshape(-1)
            tree = data.get("tree_reps")
            if status.size == 0 or not isinstance(tree, Mapping):
                raise ValueError(f"Missing status/tree_reps in {meta.path}")
            detected = TRACE.infer_num_actions(tree)
            if n_actions is None:
                n_actions = detected; log(f"Detected {n_actions} actions")
            elif detected != n_actions:
                raise ValueError(f"Action count changed in {meta.path}")

            real_indices = np.flatnonzero(status == REAL_STATUS)
            previous_local_real: Optional[int] = None
            for current_real_value in real_indices:
                current_real = int(current_real_value); real_step += 1
                start = 0 if previous_local_real is None else previous_local_real + 1
                current_batch = build_event_batch(data, tree, status, np.arange(start, current_real), n_actions)
                batch = concatenate_batches(carry if previous_local_real is None else None, current_batch)
                if batch is None:
                    batch = empty_batch(n_actions)
                current_action = TRACE.decode_action(data.get("human_action"), current_real)
                if previous_root is not None and 0 <= previous_root.human_action < n_actions and 0 <= current_action < n_actions:
                    record, events = window_and_events(
                        meta, real_step, previous_root, current_action, batch, n_actions, len(windows)
                    )
                    windows.append(record)
                    for key in EVENT_KEYS:
                        event_lists[key].append(events[key])
                previous_root = make_root(data, tree, current_real, n_actions, previous_root)
                previous_local_real = current_real
                carry = None

            tail_start = previous_local_real + 1 if previous_local_real is not None else 0
            tail = build_event_batch(data, tree, status, np.arange(tail_start, len(status)), n_actions)
            carry = concatenate_batches(carry, tail)
            del data, tree, status
            gc.collect()
            log(f"Processed {meta.path.name}; windows={len(windows):,}")

    if not windows or n_actions is None:
        raise RuntimeError("No eligible windows extracted")
    frame = pd.DataFrame.from_records(windows)
    frame["run_progress"] = frame.groupby("run_label", sort=False).cumcount()
    sizes = frame.groupby("run_label")["run_label"].transform("size")
    frame["run_progress"] = frame["run_progress"] / np.maximum(sizes - 1, 1)
    events = {
        key: np.concatenate(values) if values else np.asarray([], dtype=float)
        for key, values in event_lists.items()
    }
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION, "n_actions": n_actions,
        "n_windows": int(len(frame)), "n_prefix_pairs": int(len(events["window_index"])),
        "n_runs": int(frame["run_label"].nunique()), "loaded_files": loaded_files,
        "max_files_per_block": max_files, "trace_signature": trace_signature(blocks),
        "temporal_definition": "raw rollout return at status-2 k predicts operation at status-2 k+1",
        "actor_alignment": "cur_policy[k] evaluation -> actor_policy[k+1] -> cur_action[k+1]; primary actor analyses exclude observed reset boundaries",
        "path_novelty": "first observation versus revisit of a complete action path within the same real-step search window",
        "depth_policy": "raw primary; within-current-depth rank sensitivity; no final-depth residualization",
    }
    return frame, events, manifest


def event_frame(windows: pd.DataFrame, events: Mapping[str, np.ndarray]) -> pd.DataFrame:
    frame = pd.DataFrame({key: events[key] for key in EVENT_KEYS})
    index = frame["window_index"].to_numpy(int)
    for column in (
        "run_label", "maintained_action", "bout_age", "persistence_evidence",
        "switch_target_uncertainty", "root_value", "switch",
    ):
        frame[column] = windows[column].to_numpy()[index]
    frame["persistence_x_uncertainty"] = (
        frame["persistence_evidence"] * frame["switch_target_uncertainty"]
    )
    frame["within_depth_return_rank"] = (
        frame.groupby(["run_label", "current_depth"], sort=False)["return_advantage"]
        .rank(method="average", pct=True)
    )
    return frame


def exact_signflip_p(values: Sequence[float]) -> float:
    return BASE.exact_signflip_p(np.asarray(values, dtype=float))


def exact_signflip_two_sided_p(values: Sequence[float]) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    observed = abs(float(np.mean(values)))
    if values.size > 20:
        rng = np.random.default_rng(20260807)
        signs = rng.choice((-1.0, 1.0), size=(100_000, values.size))
    else:
        codes = np.arange(2 ** values.size, dtype=np.uint64)[:, None]
        bits = (codes >> np.arange(values.size, dtype=np.uint64)) & 1
        signs = np.where(bits, 1.0, -1.0)
    null = np.abs(np.mean(signs * values[None, :], axis=1))
    return float((1 + np.sum(null >= observed - 1e-15)) / (len(null) + 1))


def temporal_models(events: pd.DataFrame, subject: int, game: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, object]] = []
    outcomes = ["same_branch_next", "same_path_deepen_next", "branch_switch_next", "reset_after"]
    for run, run_data in events.groupby("run_label", sort=True):
        run_data = run_data.loc[run_data["pair_valid"].eq(1)].copy()
        run_data = run_data.replace([np.inf, -np.inf], np.nan)
        for predictor, analysis_type in (
            ("return_advantage", "raw_return_primary"),
            ("within_depth_return_rank", "within_current_depth_sensitivity"),
        ):
            numeric = [
                predictor, "current_depth", "event_progress", "persistence_evidence",
                "switch_target_uncertainty", "persistence_x_uncertainty", "bout_age",
            ]
            categorical = ["maintained_action", "root_branch"]
            usable = run_data.dropna(subset=numeric)
            if len(usable) < 100:
                continue
            x_numeric = StandardScaler().fit_transform(usable[numeric].to_numpy(float))
            x_category = pd.get_dummies(
                usable[categorical].astype(int).astype(str), drop_first=True, dtype=float
            ).to_numpy(float)
            x = np.column_stack([x_numeric, x_category])
            for outcome in outcomes:
                y = usable[outcome].to_numpy(int)
                if np.unique(y).size < 2 or min(np.sum(y), np.sum(1 - y)) < 20:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000).fit(x, y)
                rows.append({
                    "subject": subject, "game": game, "run_label": run,
                    "analysis": analysis_type, "predictor": predictor, "outcome": outcome,
                    "standardized_log_odds_coefficient": float(model.coef_[0, 0]),
                    "n_pairs": len(usable), "outcome_rate": float(np.mean(y)),
                })

        # Reverse direction: does choosing to deepen predict the next return change?
        cols = [
            "next_return_change", "same_path_deepen_next", "return_advantage", "current_depth",
            "event_progress", "persistence_evidence", "switch_target_uncertainty",
            "persistence_x_uncertainty", "bout_age",
        ]
        usable = run_data.dropna(subset=cols)
        if len(usable) >= 100 and usable["same_path_deepen_next"].nunique() > 1:
            x = StandardScaler().fit_transform(usable[cols[1:]].to_numpy(float))
            y = StandardScaler().fit_transform(usable[["next_return_change"]].to_numpy(float)).ravel()
            model = LinearRegression().fit(x, y)
            rows.append({
                "subject": subject, "game": game, "run_label": run,
                "analysis": "deepening_to_next_return", "predictor": "same_path_deepen_next",
                "outcome": "next_return_change",
                "standardized_log_odds_coefficient": float(model.coef_[0]),
                "n_pairs": len(usable), "outcome_rate": np.nan,
            })
    coefficients = pd.DataFrame(rows)
    summary_rows = []
    for keys, group in coefficients.groupby(["analysis", "predictor", "outcome"], sort=False):
        values = group["standardized_log_odds_coefficient"].to_numpy(float)
        summary_rows.append({
            "subject": subject, "game": game, "analysis": keys[0], "predictor": keys[1],
            "outcome": keys[2], "n_runs": len(values), "mean_run_coefficient": float(np.mean(values)),
            "median_run_coefficient": float(np.median(values)),
            "positive_run_fraction": float(np.mean(values > 0)),
            "development_run_signflip_p_descriptive": exact_signflip_p(values),
        })
    return coefficients, pd.DataFrame(summary_rows)


def actor_cycle_models(
    events: pd.DataFrame, subject: int, game: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate the ordered evaluator -> actor -> sampled-action sequence.

    Rows crossing an observed status-1/status-3 reset boundary are excluded:
    the trace contains the primary actor policy but not the reset policy, so an
    evaluator-to-next-actor claim is only aligned within an uninterrupted bout.
    """
    rows: List[Dict[str, object]] = []
    for run, run_data in events.groupby("run_label", sort=True):
        data = run_data.loc[
            run_data["pair_valid"].eq(1) & run_data["reset_after"].eq(0)
        ].replace([np.inf, -np.inf], np.nan).copy()
        controls = [
            "return_advantage", "current_depth", "event_progress",
            "persistence_evidence", "switch_target_uncertainty",
            "persistence_x_uncertainty", "bout_age",
        ]
        categorical = ["maintained_action", "root_branch"]
        complete = data.dropna(subset=[
            *controls, "next_actor_model_js", "next_actor_chosen_probability",
            "model_probability_for_next_action",
        ])
        if len(complete) < 100:
            continue
        x_numeric = StandardScaler().fit_transform(complete[controls].to_numpy(float))
        x_category = pd.get_dummies(
            complete[categorical].astype(int).astype(str), drop_first=True, dtype=float
        ).to_numpy(float)
        x = np.column_stack([x_numeric, x_category])
        for outcome in ("next_actor_model_js", "next_actor_chosen_probability"):
            y = StandardScaler().fit_transform(complete[[outcome]].to_numpy(float)).ravel()
            model = LinearRegression().fit(x, y)
            rows.append({
                "subject": subject, "game": game, "run_label": run,
                "analysis": "evaluation_to_next_actor", "predictor": "return_advantage",
                "outcome": outcome, "standardized_coefficient": float(model.coef_[0]),
                "n_pairs": len(complete), "outcome_rate": np.nan,
            })
        for outcome in (
            "next_action_matches_model_recommendation", "next_path_first_observation",
        ):
            y = complete[outcome].to_numpy(int)
            if np.unique(y).size < 2 or min(np.sum(y), np.sum(1 - y)) < 20:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000).fit(x, y)
            rows.append({
                "subject": subject, "game": game, "run_label": run,
                "analysis": "evaluation_to_next_actor", "predictor": "return_advantage",
                "outcome": outcome, "standardized_coefficient": float(model.coef_[0, 0]),
                "n_pairs": len(complete), "outcome_rate": float(np.mean(y)),
            })

        # Descriptive attenuation, not a causal mediation estimate.  It asks
        # whether the return/deepening association is reduced after adding the
        # aligned next-actor policy readouts.
        y = complete["same_path_deepen_next"].to_numpy(int)
        if np.unique(y).size > 1 and min(np.sum(y), np.sum(1 - y)) >= 20:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                total = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000).fit(x, y)
            adjusted_columns = [
                *controls, "next_actor_model_js", "next_actor_chosen_probability",
                "model_probability_for_next_action",
            ]
            x_adjusted_numeric = StandardScaler().fit_transform(
                complete[adjusted_columns].to_numpy(float)
            )
            x_adjusted = np.column_stack([x_adjusted_numeric, x_category])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                adjusted = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000).fit(x_adjusted, y)
            for label, coefficient in (
                ("total", float(total.coef_[0, 0])),
                ("actor_adjusted", float(adjusted.coef_[0, 0])),
            ):
                rows.append({
                    "subject": subject, "game": game, "run_label": run,
                    "analysis": "return_to_deepening_attenuation", "predictor": label,
                    "outcome": "same_path_deepen_next",
                    "standardized_coefficient": coefficient,
                    "n_pairs": len(complete), "outcome_rate": float(np.mean(y)),
                })
    coefficients = pd.DataFrame(rows)
    summary_rows: List[Dict[str, object]] = []
    if len(coefficients):
        for keys, group in coefficients.groupby(["analysis", "predictor", "outcome"], sort=False):
            values = group["standardized_coefficient"].to_numpy(float)
            summary_rows.append({
                "subject": subject, "game": game, "analysis": keys[0],
                "predictor": keys[1], "outcome": keys[2], "n_runs": len(values),
                "mean_run_coefficient": float(np.mean(values)),
                "median_run_coefficient": float(np.median(values)),
                "positive_run_fraction": float(np.mean(values > 0)),
                "development_run_signflip_p_two_sided_descriptive": exact_signflip_two_sided_p(values),
            })
    return coefficients, pd.DataFrame(summary_rows)


def actor_cycle_bins(events: pd.DataFrame, subject: int, game: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    eligible = events.loc[
        events["pair_valid"].eq(1) & events["reset_after"].eq(0)
    ].replace([np.inf, -np.inf], np.nan)
    outcomes = [
        "next_actor_model_js", "next_actor_chosen_probability",
        "next_action_matches_model_recommendation", "next_return_change",
    ]
    for run, group in eligible.groupby("run_label", sort=True):
        data = group.dropna(subset=["return_advantage"]).copy()
        if len(data) < 50:
            continue
        try:
            data["return_bin"] = pd.qcut(
                data["return_advantage"], 5, labels=False, duplicates="drop"
            )
        except ValueError:
            continue
        status_masks = {
            "all": np.ones(len(data), dtype=bool),
            "first-observed path": data["next_path_first_observation"].eq(1).to_numpy(),
            "revisited path": data["next_path_revisit"].eq(1).to_numpy(),
        }
        for status_label, status_mask in status_masks.items():
            subset = data.loc[status_mask]
            for value_bin, cell in subset.groupby("return_bin"):
                row: Dict[str, object] = {
                    "subject": subject, "game": game, "run_label": run,
                    "path_status": status_label, "return_bin": int(value_bin),
                    "n_pairs": len(cell), "return_advantage_mean": float(cell["return_advantage"].mean()),
                }
                for outcome in outcomes:
                    row[outcome] = float(cell[outcome].mean())
                rows.append(row)
    return pd.DataFrame(rows)


def prefix_bins(events: pd.DataFrame, subject: int, game: int) -> pd.DataFrame:
    rows = []
    outcomes = ["same_branch_next", "same_path_deepen_next", "branch_switch_next", "reset_after"]
    for run, group in events.loc[events["pair_valid"].eq(1)].groupby("run_label", sort=True):
        group = group.copy()
        for predictor, analysis in (
            ("return_advantage", "raw_return_primary"),
            ("within_depth_return_rank", "within_current_depth_sensitivity"),
        ):
            finite = np.isfinite(group[predictor].to_numpy(float))
            data = group.loc[finite].copy()
            if len(data) < 20:
                continue
            try:
                data["value_bin"] = pd.qcut(data[predictor], 5, labels=False, duplicates="drop")
            except ValueError:
                continue
            for value_bin, cell in data.groupby("value_bin"):
                row = {
                    "subject": subject, "game": game, "run_label": run,
                    "analysis": analysis, "predictor": predictor, "value_bin": int(value_bin),
                    "n_pairs": len(cell), "predictor_mean": float(cell[predictor].mean()),
                }
                for outcome in outcomes:
                    row[outcome] = float(cell[outcome].mean())
                rows.append(row)
    return pd.DataFrame(rows)


def behavior_nested(windows: pd.DataFrame, subject: int, game: int, args: argparse.Namespace):
    data = windows.loc[windows["value_metric_coverage"].eq(1)].copy()
    n_actions = int(max(data["maintained_action"].max(), data["human_action"].max()) + 1)
    model_ilr = [f"model_root_ilr_{i}" for i in range(n_actions - 1)]
    actor_ilr = [f"actor_initial_ilr_{i}" for i in range(n_actions - 1)]
    base = ["bout_age", "prior_reward", "prior_done", "run_progress"]
    geometry = [
        *base, "root_value", *model_ilr, *actor_ilr, "persistence_evidence",
        "switch_target_uncertainty", "persistence_x_uncertainty", "actor_model_js",
    ]
    value = [
        *geometry, "max_value_gain", "search_return_change", "branch_return_spread",
        "maintained_minus_best_other_return", "mean_absolute_value_discrepancy",
    ]
    allocation = [
        *value, "value_allocation_alignment", "best_branch_visit_share", "branch_diversity",
        "mean_bout_terminal_depth", "deep_node_fraction", "branch_switch_fraction", "reset_fraction",
    ]
    categorical = ["maintained_action", "action_before_maintained"]
    tests = [
        ("rollout_value_increment", geometry, value),
        ("value_aligned_allocation_increment", value, allocation),
    ]
    score_frames = []; prediction_frames = []
    for name, base_columns, augmented_columns in tests:
        complete = data.dropna(subset=list(dict.fromkeys(augmented_columns)))
        scores, predictions = BASE.nested_loro_pair(
            complete, name, base_columns, augmented_columns, categorical,
            args.penalty_grid, args.inner_folds, require_valid_search=False,
        )
        score_frames.append(scores); prediction_frames.append(predictions)
    scores = pd.concat(score_frames, ignore_index=True) if score_frames else pd.DataFrame()
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    summary = BASE.summarize_nested_scores(scores, subject, game) if len(scores) else pd.DataFrame()
    return scores, predictions, summary


def binned_surface(
    frame: pd.DataFrame, value: Optional[str], bins: int, minimum: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = frame["persistence_evidence"].to_numpy(float)
    y = frame["switch_target_uncertainty"].to_numpy(float)
    z = np.ones(len(frame)) if value is None else frame[value].to_numpy(float)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[finite], y[finite], z[finite]
    lo, hi = np.quantile(x, [0.01, 0.99])
    x_edges = np.linspace(lo, hi, bins + 1); y_edges = np.linspace(0, 1, bins + 1)
    count, _, _ = np.histogram2d(y, x, bins=[y_edges, x_edges])
    if value is None:
        surface = np.log10(count + 1); surface[count < minimum] = np.nan
    else:
        total, _, _ = np.histogram2d(y, x, bins=[y_edges, x_edges], weights=z)
        with np.errstate(invalid="ignore", divide="ignore"):
            surface = total / count
        surface[count < minimum] = np.nan
    return surface, x_edges, y_edges


def draw_surface(
    ax: plt.Axes, frame: pd.DataFrame, value: Optional[str], title: str, label: str,
    bins: int, minimum: int, cmap: str = "viridis", symmetric: bool = False,
) -> None:
    surface, x_edges, y_edges = binned_surface(frame, value, bins, minimum)
    kwargs = {}
    if symmetric and np.any(np.isfinite(surface)):
        limit = float(np.nanquantile(np.abs(surface), 0.98)); kwargs = {"vmin": -limit, "vmax": limit}
    mesh = ax.pcolormesh(x_edges, y_edges, surface, shading="auto", cmap=cmap, **kwargs)
    cb = ax.figure.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03); cb.set_label(label)
    ax.set_title(title); ax.set_xlabel("Evidence for continuing the current action")
    ax.set_ylabel("Uncertainty among other actions"); ax.set_ylim(0, 1)


def save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig); log(f"Saved {path}")


def _quantile_run_summary(
    frame: pd.DataFrame,
    x_column: str,
    value_column: str,
    bins: int = 5,
) -> pd.DataFrame:
    """Return run-level means and 95% CIs across x-axis quintiles."""
    required = {x_column, value_column, "run_label"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    data = frame.loc[:, [x_column, value_column, "run_label"]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
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
    x_label: str,
    y_label: str,
    bins: int = 5,
) -> None:
    summary = _quantile_run_summary(frame, x_column, value_column, bins)
    if summary.empty:
        ax.text(0.5, 0.5, "Insufficient finite data", ha="center", va="center", transform=ax.transAxes)
        return
    x = summary["quantile"].to_numpy(float) + 1.0
    y = summary["mean"].to_numpy(float)
    ax.plot(x, y, color=color, marker="o", lw=2.2)
    ax.fill_between(x, summary["low"], summary["high"], color=color, alpha=0.16)
    ax.set_xticks(np.arange(1, bins + 1))
    ax.set_xlabel(f"{x_label} quintile (low → high)")
    ax.set_ylabel(y_label)
    ax.grid(axis="y", alpha=0.22)


def _finish_linear_figure(fig: plt.Figure, path: Path, dpi: int, title: str) -> None:
    fig.suptitle(title, fontsize=18, x=0.5, ha="center")
    save_figure(fig, path, dpi)


def _plot_metric_linear_grid(
    frame: pd.DataFrame,
    path: Path,
    args: argparse.Namespace,
    metrics: Sequence[Tuple[str, str]],
    title: str,
) -> None:
    predictors = [
        ("persistence_evidence", "Persistence evidence", BLUE),
        ("switch_target_uncertainty", "Switch-target uncertainty", ORANGE),
    ]
    fig, axes = plt.subplots(
        len(predictors), len(metrics),
        figsize=(4.7 * len(metrics), 3.9 * len(predictors)),
        constrained_layout=True,
        squeeze=False,
    )
    for row, (x_column, x_label, color) in enumerate(predictors):
        for col, (value_column, y_label) in enumerate(metrics):
            ax = axes[row, col]
            _draw_marginal_line(ax, frame, x_column, value_column, color, x_label, y_label)
            ax.set_title(y_label)
    _finish_linear_figure(fig, path, args.dpi, title)


def plot_value_geometry(frame: pd.DataFrame, path: Path, args: argparse.Namespace) -> None:
    panels = [
        (None, "Search-window occupancy", r"$\log_{10}(n+1)$", "viridis", False),
        ("max_value_gain", "Best discovered return − root value", "Raw value difference", "coolwarm", True),
        ("maintained_minus_best_other_return", "Current-action return − best other return", "Raw value difference", "coolwarm", True),
        ("branch_return_spread", "Return differences across action branches", "Across-branch SD", "magma", False),
        ("search_return_change", "Final − first rollout return", "Raw return change", "coolwarm", True),
        ("mean_absolute_value_discrepancy", "Rollout–root value discrepancy", "Mean absolute difference", "magma", False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for ax, panel in zip(axes.flat, panels):
        draw_surface(ax, frame, panel[0], panel[1], panel[2], args.figure_bins, args.min_bin_count, panel[3], panel[4])
    fig.suptitle("Rollout value across the pre-search control geometry", fontsize=18, x=0.5, ha="center")
    save_figure(fig, path, args.dpi)


def plot_allocation_geometry(frame: pd.DataFrame, path: Path, args: argparse.Namespace) -> None:
    panels = [
        ("branch_diversity", "Branch allocation diversity", "Mean", "viridis", False),
        ("mean_bout_terminal_depth", "Mean bout terminal depth", "Mean depth", "viridis", False),
        ("deep_node_fraction", "Deep-node allocation", "Fraction", "viridis", False),
        ("value_allocation_alignment", "Return–allocation alignment", "Rank correlation", "coolwarm", True),
        ("best_branch_visit_share", "Search share on best-return branch", "Fraction", "viridis", False),
        ("branch_switch_fraction", "Ordered branch-switch fraction", "Fraction", "viridis", False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for ax, panel in zip(axes.flat, panels):
        draw_surface(ax, frame, panel[0], panel[1], panel[2], args.figure_bins, args.min_bin_count, panel[3], panel[4])
    fig.suptitle("Value and fixed-budget search allocation", fontsize=18, x=0.5, ha="center")
    save_figure(fig, path, args.dpi)


def plot_value_geometry_linear(frame: pd.DataFrame, path: Path, args: argparse.Namespace) -> None:
    _plot_metric_linear_grid(
        frame,
        path,
        args,
        [
            ("max_value_gain", "Best return − root value"),
            ("maintained_minus_best_other_return", "Current − best other return"),
            ("branch_return_spread", "Across-branch return SD"),
            ("search_return_change", "Final − first return"),
            ("mean_absolute_value_discrepancy", "Rollout–root discrepancy"),
        ],
        "Marginal rollout-value measures across the pre-search control state",
    )


def plot_allocation_geometry_linear(frame: pd.DataFrame, path: Path, args: argparse.Namespace) -> None:
    _plot_metric_linear_grid(
        frame,
        path,
        args,
        [
            ("branch_diversity", "Branch allocation diversity"),
            ("mean_bout_terminal_depth", "Mean bout terminal depth"),
            ("deep_node_fraction", "Deep-node allocation"),
            ("value_allocation_alignment", "Return–allocation alignment"),
            ("best_branch_visit_share", "Best-branch visit share"),
            ("branch_switch_fraction", "Branch-switch fraction"),
        ],
        "Marginal value-sensitive search allocation across the pre-search control state",
    )


def plot_prefix_outcomes(bins: pd.DataFrame, path: Path, args: argparse.Namespace, sensitivity: bool) -> None:
    analysis = "within_current_depth_sensitivity" if sensitivity else "raw_return_primary"
    data = bins.loc[bins["analysis"].eq(analysis)]
    outcomes = [
        ("same_branch_next", "Continue the same action branch"),
        ("same_path_deepen_next", "Deepen the same path"),
        ("branch_switch_next", "Move to another action branch"),
        ("reset_after", "Reset after the current search event"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, (column, title) in zip(axes.flat, outcomes):
        run_table = data.pivot_table(index="run_label", columns="value_bin", values=column, aggfunc="mean")
        x = np.arange(1, 6)
        mean = run_table.reindex(columns=range(5)).mean(axis=0).to_numpy(float)
        se = run_table.reindex(columns=range(5)).sem(axis=0).to_numpy(float)
        ax.plot(x, mean, color=BLUE, marker="o", lw=2)
        ax.fill_between(x, mean - 1.96 * se, mean + 1.96 * se, color=BLUE, alpha=0.18)
        ax.set_title(title); ax.set_xlabel("Return quintile (low → high)"); ax.set_ylabel("Probability")
        ax.set_xticks(x); ax.set_ylim(bottom=0); ax.grid(alpha=0.2)
    title = (
        "Next search operation by return rank within current depth"
        if sensitivity else "Next search operation by raw rollout-return advantage"
    )
    fig.suptitle(title, fontsize=17, x=0.5, ha="center")
    save_figure(fig, path, args.dpi)


def plot_temporal_coefficients(coefficients: pd.DataFrame, path: Path, args: argparse.Namespace) -> None:
    labels = {
        "same_branch_next": "Continue branch", "same_path_deepen_next": "Deepen path",
        "branch_switch_next": "Change branch", "reset_after": "Reset",
        "next_return_change": "Next return change",
    }
    order = ["same_branch_next", "same_path_deepen_next", "branch_switch_next", "reset_after", "next_return_change"]
    # Keep the primary forward temporal test.  The reverse deepening test is
    # retained in the CSV but is not displayed because it is not estimable in
    # all runs and was not part of the retained figure specification.
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 5), constrained_layout=True)
    data = coefficients.loc[coefficients["analysis"].eq("raw_return_primary")]
    active = [item for item in order if item in set(data["outcome"])]
    for position, outcome in enumerate(active):
        values = data.loc[data["outcome"].eq(outcome), "standardized_log_odds_coefficient"].to_numpy(float)
        jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) else np.asarray([])
        ax.scatter(np.full(len(values), position) + jitter, values, color=BLUE, alpha=0.7, s=30)
        if len(values):
            ax.plot([position - 0.18, position + 0.18], [np.mean(values)] * 2, color=CHARCOAL, lw=3)
    ax.axhline(0, color=CHARCOAL, ls="--", lw=1)
    ax.set_xticks(range(len(active)), [labels[item] for item in active], rotation=20, ha="right")
    ax.set_ylabel("Within-run standardized coefficient")
    ax.set_title("Raw return predicting the next search operation")
    ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Temporal direction of rollout value and search allocation", fontsize=17, x=0.5, ha="center")
    save_figure(fig, path, args.dpi)


def plot_actor_cycle_bins(bins: pd.DataFrame, path: Path, args: argparse.Namespace) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)
    panels = [
        ("next_actor_model_js", "Next actor–model policy divergence", "Jensen–Shannon divergence"),
        ("next_actor_chosen_probability", "Next actor confidence in its sampled action", "Actor probability"),
        ("next_action_matches_model_recommendation", "Sample follows model recommendation", "Probability"),
    ]
    all_paths = bins.loc[bins["path_status"].eq("all")]
    for ax, (column, title, ylabel) in zip(axes.flat[:3], panels):
        table = all_paths.pivot_table(index="run_label", columns="return_bin", values=column, aggfunc="mean")
        mean = table.reindex(columns=range(5)).mean(axis=0).to_numpy(float)
        se = table.reindex(columns=range(5)).sem(axis=0).to_numpy(float)
        x = np.arange(1, 6)
        ax.plot(x, mean, color=BLUE, marker="o", lw=2)
        ax.fill_between(x, mean - 1.96 * se, mean + 1.96 * se, color=BLUE, alpha=0.18)
        ax.set_title(title); ax.set_xlabel("Current rollout-return quintile (low → high)")
        ax.set_ylabel(ylabel); ax.set_xticks(x); ax.grid(alpha=0.2)

    ax = axes.flat[3]
    for status, color in (("first-observed path", ORANGE), ("revisited path", PURPLE)):
        subset = bins.loc[bins["path_status"].eq(status)]
        table = subset.pivot_table(index="run_label", columns="return_bin", values="next_return_change", aggfunc="mean")
        mean = table.reindex(columns=range(5)).mean(axis=0).to_numpy(float)
        se = table.reindex(columns=range(5)).sem(axis=0).to_numpy(float)
        x = np.arange(1, 6)
        ax.plot(x, mean, color=color, marker="o", lw=2, label=status.capitalize())
        ax.fill_between(x, mean - 1.96 * se, mean + 1.96 * se, color=color, alpha=0.15)
    ax.axhline(0, color=CHARCOAL, ls="--", lw=1)
    ax.set_title("Next return change by observed path history")
    ax.set_xlabel("Current rollout-return quintile (low → high)")
    ax.set_ylabel("Next minus current rollout return"); ax.set_xticks(np.arange(1, 6))
    ax.legend(frameon=False); ax.grid(alpha=0.2)
    fig.suptitle("Model evaluation, next actor choice, and path outcome", fontsize=17, x=0.5, ha="center")
    save_figure(fig, path, args.dpi)


def plot_actor_cycle_coefficients(
    coefficients: pd.DataFrame, path: Path, args: argparse.Namespace,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 5), constrained_layout=True)
    data = coefficients.loc[coefficients["analysis"].eq("evaluation_to_next_actor")]
    labels = {
        "next_actor_model_js": "Actor–model divergence",
        "next_action_matches_model_recommendation": "Follow model recommendation",
    }
    active = [name for name in labels if name in set(data["outcome"])]
    for position, outcome in enumerate(active):
        values = data.loc[data["outcome"].eq(outcome), "standardized_coefficient"].to_numpy(float)
        jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) else np.asarray([])
        ax.scatter(np.full(len(values), position) + jitter, values, color=BLUE, alpha=0.7, s=30)
        if len(values):
            ax.plot([position - 0.18, position + 0.18], [np.mean(values)] * 2, color=CHARCOAL, lw=3)
    ax.axhline(0, color=CHARCOAL, ls="--", lw=1)
    ax.set_xticks(range(len(active)), [labels[name] for name in active], rotation=20, ha="right")
    ax.set_ylabel("Within-run standardized coefficient")
    ax.set_title("Current return predicting the next actor/action")
    ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Ordered evaluator-to-actor analysis within uninterrupted search bouts", fontsize=17, x=0.5, ha="center")
    save_figure(fig, path, args.dpi)


def plot_behavior_increment(scores: pd.DataFrame, path: Path, args: argparse.Namespace) -> None:
    tests = [
        ("rollout_value_increment", "Add rollout-value evidence"),
        ("value_aligned_allocation_increment", "Add value-aligned allocation"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for ax, (test, title) in zip(axes, tests):
        data = (
            scores.loc[scores["test"].eq(test)]
            if "test" in scores.columns else pd.DataFrame()
        )
        for _, row in data.iterrows():
            ax.plot([0, 1], [row["base_log_loss"], row["augmented_log_loss"]], color=BLUE, alpha=0.35)
            ax.scatter([0, 1], [row["base_log_loss"], row["augmented_log_loss"]], color=[ORANGE, BLUE], s=30)
        ax.set_xticks([0, 1], ["Before", "After"]); ax.set_ylabel("Held-out log loss")
        delta = (data["base_log_loss"] - data["augmented_log_loss"]).mean() if len(data) else np.nan
        ax.set_title(f"{title}\nmean improvement = {delta:.4g} nats/opportunity")
        ax.grid(axis="y", alpha=0.2)
        if data.empty:
            ax.text(0.5, 0.5, "Requires at least two runs", ha="center", va="center", transform=ax.transAxes)
    fig.suptitle("Held-out human action-switch prediction", fontsize=17, x=0.5, ha="center")
    save_figure(fig, path, args.dpi)


def qa_summary(windows: pd.DataFrame, events: Mapping[str, np.ndarray], manifest: Mapping[str, object]) -> Dict[str, object]:
    return {
        "n_windows": int(len(windows)), "n_runs": int(windows["run_label"].nunique()),
        "n_prefix_pairs": int(len(events["window_index"])),
        "status2": {
            "min": int(windows["n_status2"].min()), "median": float(windows["n_status2"].median()),
            "max": int(windows["n_status2"].max()), "nominal_29_fraction": float(np.mean(windows["n_status2"] == 29)),
        },
        "finite_root_value_fraction": float(np.mean(np.isfinite(windows["root_value"]))),
        "finite_rollout_mean_fraction": float(np.mean(np.isfinite(windows["rollout_return_mean"]))),
        "two_branch_value_coverage": float(windows["value_metric_coverage"].mean()),
        "mean_branch_diversity": float(windows["branch_diversity"].mean()),
        "mean_terminal_depth": float(windows["mean_bout_terminal_depth"].mean()),
        "mean_deep_node_fraction": float(windows["deep_node_fraction"].mean()),
        "marker_accounting": {
            "total": int(windows["marker_count"].sum()),
            "leading_excluded": int(windows["leading_marker_count"].sum()),
            "eligible": int(
                (windows["marker_count"] - windows["leading_marker_count"]).sum()
            ),
            "coverage": float(
                np.mean(
                    np.where(
                        windows["mapped_marker_count"].to_numpy() > 0,
                        windows["reset_boundary_count"].to_numpy()
                        / np.maximum(windows["mapped_marker_count"].to_numpy(), 1),
                        1.0,
                    )
                )
            ),
            "leading": int(windows["leading_marker"].sum()),
            "internal_boundaries": int(windows["reset_boundary_count"].sum()),
            "collisions": int(windows["marker_collision_count"].sum()),
        },
        "cache_manifest": dict(manifest),
        "inference_note": "Run-level effects are descriptive within one participant; participant is the population unit.",
    }


def run_analysis(
    windows: pd.DataFrame, event_arrays: Mapping[str, np.ndarray], subject: int, game: int,
    output_dir: Path, args: argparse.Namespace,
) -> None:
    result_dir = output_dir / "results"; figure_dir = output_dir / "figures"
    result_dir.mkdir(parents=True, exist_ok=True); figure_dir.mkdir(parents=True, exist_ok=True)
    events = event_frame(windows, event_arrays)
    log(f"Analyze {len(events):,} ordered search-event pairs")
    coefficients, temporal_summary = temporal_models(events, subject, game)
    bins = prefix_bins(events, subject, game)
    actor_coefficients, actor_summary = actor_cycle_models(events, subject, game)
    actor_bins = actor_cycle_bins(events, subject, game)
    scores, predictions, nested_summary = behavior_nested(windows, subject, game, args)

    windows.to_csv(result_dir / "window_value_allocation_metrics.csv.gz", index=False, compression="gzip")
    coefficients.to_csv(result_dir / "temporal_run_coefficients.csv", index=False)
    temporal_summary.to_csv(result_dir / "temporal_summary.csv", index=False)
    bins.to_csv(result_dir / "prefix_return_quintiles.csv", index=False)
    actor_coefficients.to_csv(result_dir / "actor_cycle_run_coefficients.csv", index=False)
    actor_summary.to_csv(result_dir / "actor_cycle_summary.csv", index=False)
    actor_bins.to_csv(result_dir / "actor_cycle_return_quintiles.csv", index=False)
    scores.to_csv(result_dir / "nested_cv_outer_scores.csv", index=False)
    predictions.to_csv(result_dir / "nested_cv_outer_predictions.csv.gz", index=False, compression="gzip")
    nested_summary.to_csv(result_dir / "nested_cv_summary.csv", index=False)

    plot_value_geometry(windows, figure_dir / "figure1_rollout_value_geometry.png", args)
    plot_value_geometry_linear(windows, figure_dir / "figure1_linear_rollout_value_geometry.png", args)
    plot_allocation_geometry(windows, figure_dir / "figure2_value_allocation_geometry.png", args)
    plot_allocation_geometry_linear(windows, figure_dir / "figure2_linear_value_allocation_geometry.png", args)
    plot_prefix_outcomes(bins, figure_dir / "figure3_raw_return_next_search_operation.png", args, False)
    # The within-depth sensitivity is retained in the CSV as a diagnostic but
    # its supplementary figure is intentionally omitted.
    plot_temporal_coefficients(coefficients, figure_dir / "figure4_temporal_direction.png", args)
    # Nested held-out behavior scores remain available in results; the former
    # Figure 5 is not part of the retained visual set.
    plot_actor_cycle_bins(actor_bins, figure_dir / "figure6_model_actor_event_cycle.png", args)
    plot_actor_cycle_coefficients(actor_coefficients, figure_dir / "figure7_actor_aligned_temporal_effects.png", args)
    save_json(result_dir / "analysis_manifest.json", {
        "subject": subject, "game": game, "primary_return": "raw discounted/bootstrapped rollout return",
        "primary_temporal_test": "return at status-2 k -> search operation at status-2 k+1",
        "reverse_temporal_test": "same-path deepening at k+1 -> next rollout-return change",
        "ordered_actor_test": "cur_policy[k] and rollout return[k] -> actor_policy[k+1] -> sampled cur_action[k+1]",
        "ordered_actor_scope": "pairs not crossing observed status-1/status-3 reset boundaries",
        "path_history_definition": "first observation versus revisit within the same real-step search window",
        "sensitivity": "within-run and current-depth percentile rank",
        "behavior_validation": "nested leave-one-run-out regularized logistic prediction",
        "sample_role": args.sample_role,
    })
    del events
    gc.collect()


def run_self_test() -> None:
    n_actions = 3
    status = np.asarray([3, 2, 2, 1, 2, 0])
    action_seq = np.asarray([
        np.zeros(9),
        [1, 0, 0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 1, 0],
        np.zeros(9),
        [1, 0, 0, 0, 1, 0, 0, 0, 0],
        np.zeros(9),
    ], dtype=float)
    tree = {
        "action_seq": action_seq, "rollout_return": np.arange(6)[:, None].astype(float),
        "max_rollout_return": np.arange(6)[:, None].astype(float), "cur_v": np.arange(6)[:, None].astype(float),
    }
    data = {"actor_policy": np.tile([0.6, 0.3, 0.1], (6, 1))}
    batch = build_event_batch(data, tree, status, np.arange(5), n_actions)
    assert batch.n_status2 == 3 and batch.leading_reset_code == 3
    assert batch.reset_after.tolist() == [0, 1, 0]
    assert path_extends(batch.paths[0], int(batch.depth[0]), batch.paths[1], int(batch.depth[1]))
    assert not path_extends(batch.paths[1], int(batch.depth[1]), batch.paths[2], int(batch.depth[2]))
    assert np.isclose(normalized_effective_branch_diversity(np.asarray([0, 1]), 3), 0.5)
    print("Self-test passed: reset boundaries, path extension, and branch diversity.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=parse_subject)
    parser.add_argument("--game", type=int)
    parser.add_argument("--sessions", type=str, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--shared-cache-root",
        type=Path,
        default=DEFAULT_SHARED_CACHE_ROOT,
        help="Read-only cache produced by 00_cache.py.",
    )
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--penalty-grid", type=str, default="0.1,1,10")
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--figure-bins", type=int, default=40)
    parser.add_argument("--min-bin-count", type=int, default=40)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--sample-role", choices=("development", "prospective-replication"), default="development")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.self_test:
        run_self_test(); return
    if args.subject is None or args.game is None:
        raise SystemExit("--subject and --game are required")
    try:
        args.penalty_grid = [float(value) for value in args.penalty_grid.split(",") if value.strip()]
    except ValueError as exc:
        raise SystemExit(f"Invalid --penalty-grid: {exc}") from exc
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
    windows, events, manifest = load_rollout_value(shared_paths)
    if sessions is not None:
        keep = pd.to_numeric(windows["session"], errors="coerce").isin(sessions)
        old_to_new = np.full(len(windows), -1, dtype=np.int64)
        old_to_new[np.flatnonzero(keep.to_numpy())] = np.arange(int(keep.sum()))
        event_keep = keep.to_numpy()[events["window_index"].astype(int)]
        events = {
            key: np.asarray(values)[event_keep]
            for key, values in events.items()
        }
        events["window_index"] = old_to_new[
            events["window_index"].astype(int)
        ]
        windows = windows.loc[keep].reset_index(drop=True)
    manifest = {
        **manifest,
        "shared_cache_path": str(shared_paths.root),
        "shared_cache_version": shared_manifest["shared_cache_version"],
    }
    save_json(output_dir / "results" / "qa_summary.json", qa_summary(windows, events, manifest))
    run_analysis(windows, events, args.subject, args.game, output_dir, args)
    log(f"Complete: {output_dir}")


if __name__ == "__main__":
    main()
