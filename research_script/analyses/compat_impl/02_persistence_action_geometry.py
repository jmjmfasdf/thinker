#!/usr/bin/env python3
"""Current-action margin--uncertainty geometry and value revision.

This is a non-tSNE sibling of ``02_actor_policy.py``.  It preserves that
script's behavioral rows, transition classes, search-geometry outcomes,
duration regressions, run-level inference, and figure layouts, while changing
the scientific coordinates in three ways:

* the x coordinate is the actor-policy margin of the maintained human action
  over its strongest competitor, mapped to zero--one with a sigmoid;
* the y coordinate is conditional entropy over all alternative actions;
* status-2 records are summarized within reset-delimited imagination bouts;
* the primary z coordinate is a bounded directional value-revision balance:
  signed event-wise root-Q revisions are divided by the corresponding total
  absolute revision magnitude, so positive values favor alternatives and
  negative values favor maintaining the current action;
* the former destination-minus-current branch-return contrast is retained as
  an explicitly retrospective supplementary sensitivity analysis;
* a separate future-blind analysis tests class-specific branch-value evidence
  for NOOP-to-active, active-to-NOOP, and active-to-active transition risk.

The primary point is that a root policy is a pre-search state, whereas a
status-2 ``cur_policy`` is evaluated at an imagined state.  Their difference
is therefore retained only as a within-window descriptive contrast.  It is
not called a policy update or a realization score.

No representation extraction, PCA, or t-SNE is performed here.  Behavioral
transition figures contain one three-dimensional endpoint trajectory and
three pairwise two-dimensional probability-mass-change maps.

Examples
--------
python research_script/02_persistence_action_geometry.py --subject 3 --game 2
python research_script/02_persistence_action_geometry.py --subject 3 --game 2 \
    --root-reference rollout_return
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter
from scipy import stats

from common.shared_analysis_cache import (
    DEFAULT_SHARED_CACHE_ROOT,
    cache_paths as shared_cache_paths,
    load_actor_metrics,
    load_persistence_geometry,
    validate_manifest as validate_shared_manifest,
)


from common.paths import REPO as ROOT
from common.paths import SCRIPTS as SCRIPT_DIR
DEFAULT_TRACE_ROOT = ROOT / "test"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs" / "02" / "legacy" / "02_persistence_action_geometry"
DEFAULT_ACTOR_OUTPUT_ROOT = SCRIPT_DIR / "outputs" / "02" / "legacy" / "02_actor_policy"
DEFAULT_BASE_OUTPUT_ROOT = SCRIPT_DIR / "outputs" / "02" / "legacy" / "02_TSNE"
ACTOR_SCRIPT = Path(__file__).with_name("02_actor_policy.py")

REAL_STATUS = 0
WINDOW_STATUSES = (1, 2, 3)
PRIMARY_STATUS = 2
EPS = 1e-8
SCHEMA_VERSION = 4


def load_actor_module():
    """Load the numeric-named 02 module without importing it as a package."""
    spec = importlib.util.spec_from_file_location(
        "actor_policy_analysis_02_for_persistence_geometry", ACTOR_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load trace helpers from {ACTOR_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TRACE = load_actor_module()


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


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
    """Convert a trace policy row, whether probabilities or logits, to p."""
    try:
        values = np.asarray(row, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.full(n_actions, np.nan)
    if values.size != n_actions or not np.all(np.isfinite(values)):
        return np.full(n_actions, np.nan)
    probability_like = (
        np.min(values) >= -1e-6
        and np.max(values) <= 1.0 + 1e-6
        and abs(float(np.sum(values)) - 1.0) < 1e-3
    )
    if probability_like:
        probabilities = np.clip(values, EPS, None)
        return probabilities / np.sum(probabilities)
    shifted = values - np.max(values)
    exp_values = np.exp(np.clip(shifted, -50.0, 50.0))
    return exp_values / np.sum(exp_values)


def normalized_entropy(probabilities: np.ndarray) -> float:
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    if p.size <= 1 or not np.all(np.isfinite(p)) or np.sum(p) <= 0:
        return np.nan if p.size > 1 else 0.0
    p = np.clip(p, EPS, None)
    p /= np.sum(p)
    return float(-np.sum(p * np.log(p)) / math.log(p.size))


def persistence_coordinates(
    policy: object,
    incumbent: int,
    n_actions: int,
    policy_temperature: float = 1.0,
) -> Dict[str, float]:
    """Return current-action support and alternative uncertainty.

    The primary support coordinate is the signed logit margin between the
    currently maintained human action and its strongest actor-policy
    competitor.  The historical stay-versus-all-switch log odds and current
    action probability are retained as sensitivity variables, but they no
    longer define the x coordinate.
    """
    p = policy_probabilities(policy, n_actions)
    if not (0 <= incumbent < n_actions and np.all(np.isfinite(p))):
        return {
            "persistence_evidence": np.nan,
            "switch_target_uncertainty": np.nan,
            "strongest_competitor_margin": np.nan,
            "persistence_mass_logodds": np.nan,
            "current_action_probability": np.nan,
            "persistence_x": np.nan,
            "persistence_y": np.nan,
        }
    if not np.isfinite(policy_temperature) or policy_temperature <= 0:
        raise ValueError("policy_temperature must be positive")
    tempered_logits = np.log(np.clip(p, EPS, None)) / policy_temperature
    tempered_logits -= np.max(tempered_logits)
    p = np.exp(np.clip(tempered_logits, -50.0, 50.0))
    p /= np.sum(p)
    alternatives = np.delete(p, incumbent)
    alternative_mass = float(np.sum(alternatives))
    conditional = alternatives / max(alternative_mass, EPS)
    mass_logodds = float(
        np.log(p[incumbent]) - np.log(max(alternative_mass, EPS))
    )
    uncertainty = normalized_entropy(conditional)
    competitor_margin = float(
        np.log(p[incumbent]) - np.max(np.log(np.clip(alternatives, EPS, None)))
    )
    return {
        "persistence_evidence": competitor_margin,
        "switch_target_uncertainty": uncertainty,
        "strongest_competitor_margin": competitor_margin,
        "persistence_mass_logodds": mass_logodds,
        "current_action_probability": float(p[incumbent]),
        "persistence_x": float(
            1.0
            / (1.0 + np.exp(-np.clip(competitor_margin, -50.0, 50.0)))
        ),
        "persistence_y": uncertainty,
    }


def jensen_shannon(p: object, q: object, n_actions: int) -> float:
    first = policy_probabilities(p, n_actions)
    second = policy_probabilities(q, n_actions)
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        return np.nan
    first = np.clip(first, EPS, None); first /= np.sum(first)
    second = np.clip(second, EPS, None); second /= np.sum(second)
    mean = 0.5 * (first + second)
    return float(
        0.5 * np.sum(first * np.log(first / mean))
        + 0.5 * np.sum(second * np.log(second / mean))
    )


@dataclass
class RootState:
    model_policy: np.ndarray
    root_q: np.ndarray
    root_n: np.ndarray
    root_value: float
    root_rollout_return: float
    root_max_rollout_return: float
    human_action: int
    reward: float
    done: float
    source_chunk: int


def make_root(
    data: Mapping[str, object],
    tree: Mapping[str, object],
    index: int,
    n_actions: int,
    source_chunk: int,
) -> RootState:
    model_policy = TRACE.action_matrix(
        tree, "root_policy", np.asarray([index]), n_actions
    )[0]
    root_value = scalar_at(tree.get("root_v"), index)
    if not np.isfinite(root_value):
        root_value = scalar_at(tree.get("root_vs"), index)
    return RootState(
        model_policy=policy_probabilities(model_policy, n_actions),
        root_q=TRACE.action_matrix(
            tree, "root_qs_mean", np.asarray([index]), n_actions
        )[0],
        root_n=TRACE.action_matrix(
            tree, "root_ns", np.asarray([index]), n_actions
        )[0],
        root_value=root_value,
        root_rollout_return=scalar_at(tree.get("rollout_return"), index),
        root_max_rollout_return=scalar_at(tree.get("max_rollout_return"), index),
        human_action=TRACE.decode_action(data.get("human_action"), index),
        reward=scalar_at(data.get("cur_rewards"), index),
        done=scalar_at(tree.get("root_d"), index),
        source_chunk=source_chunk,
    )


def extract_actor_rows(
    data: Mapping[str, object],
    status: np.ndarray,
    indices: np.ndarray,
    n_actions: int,
) -> np.ndarray:
    selected = np.asarray(indices, dtype=int)
    if selected.size:
        selected = selected[status[selected] == PRIMARY_STATUS]
    source = data.get("actor_policy")
    if source is None:
        return np.full((len(selected), n_actions), np.nan)
    array = np.asarray(source)
    if array.ndim < 2 or array.shape[-1] != n_actions:
        raise ValueError(
            f"actor_policy must have final width {n_actions}; observed {array.shape}"
        )
    if selected.size and len(array) <= int(selected.max()):
        raise ValueError("actor_policy is shorter than status")
    return np.asarray(array[selected], dtype=float).reshape(len(selected), n_actions)


def concatenate_policy_rows(
    first: Optional[np.ndarray], second: Optional[np.ndarray]
) -> np.ndarray:
    if first is None or len(first) == 0:
        return np.asarray(second if second is not None else np.zeros((0, 0)))
    if second is None or len(second) == 0:
        return np.asarray(first)
    return np.vstack([first, second])


def split_bouts(batch: object) -> List[Dict[str, object]]:
    """Split one real-step window into reset-delimited status-2 bouts."""
    n_events = int(batch.n_status2)
    if n_events == 0:
        return []
    valid = np.asarray(batch.path_valid, dtype=bool)
    reset_values = np.asarray(batch.cur_reset, dtype=int)
    reset_known = len(reset_values) == n_events and np.all(
        np.isin(reset_values, (0, 1))
    )
    if reset_known:
        reset_flags = reset_values.astype(np.int8, copy=False)
    else:
        reset_flags = np.zeros(n_events, dtype=np.int8)
        for index in range(n_events - 1):
            if (
                valid[index]
                and valid[index + 1]
                and int(batch.depth[index + 1]) <= int(batch.depth[index])
            ):
                reset_flags[index] = 1
    ends = np.flatnonzero(reset_flags == 1).tolist()
    if not ends or ends[-1] != n_events - 1:
        ends.append(n_events - 1)
    starts = [0] + [int(end) + 1 for end in ends[:-1]]
    bouts: List[Dict[str, object]] = []
    for bout_index, (start, end) in enumerate(zip(starts, ends)):
        positions = np.flatnonzero(valid[start : end + 1]) + start
        if positions.size == 0:
            continue
        first = int(positions[0])
        terminal = int(positions[-1])
        path = tuple(
            int(value)
            for value in np.asarray(batch.paths[first], dtype=int)
            if int(value) >= 0
        )
        bouts.append(
            {
                "bout_index": int(bout_index),
                "first": first,
                "terminal": terminal,
                "positions": positions.astype(int),
                "branch": int(batch.paths[first, 0]),
                "terminal_depth": int(batch.depth[terminal]),
                "path": path,
                "reset_known": int(reset_known),
            }
        )
    return bouts


def effective_branching(branches: np.ndarray, n_actions: int) -> float:
    if branches.size == 0:
        return np.nan
    counts = np.bincount(branches.astype(int), minlength=n_actions).astype(float)
    counts = counts[counts > 0]
    probabilities = counts / np.sum(counts)
    effective = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    return float((effective - 1.0) / max(n_actions - 1, 1))


def root_reference(root: RootState, requested: str) -> Tuple[float, str]:
    candidates = {
        "root_v": root.root_value,
        "rollout_return": root.root_rollout_return,
        "max_rollout_return": root.root_max_rollout_return,
    }
    order = [requested] + [name for name in candidates if name != requested]
    for name in order:
        if np.isfinite(candidates[name]):
            return float(candidates[name]), name
    return np.nan, "unavailable"


def summarize_window(
    root: RootState,
    batch: object,
    policy_rows: np.ndarray,
    current_action: int,
    policy_source: str,
    requested_root_reference: str,
    n_actions: int,
    policy_temperature: float,
) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]]]:
    incumbent = int(root.human_action)
    if not 0 <= incumbent < n_actions:
        raise ValueError("Invalid incumbent action")
    if policy_source == "actor":
        if len(policy_rows) != int(batch.n_status2):
            raise ValueError("Actor policy rows do not align with status-2 events")
        presearch_policy = policy_rows[0]
        policy_trajectory = policy_rows
    else:
        presearch_policy = root.model_policy
        policy_trajectory = np.asarray(batch.cur_policy, dtype=float)
    pre_coords = persistence_coordinates(
        presearch_policy,
        incumbent,
        n_actions,
        policy_temperature=policy_temperature,
    )
    root_ref, root_ref_name = root_reference(root, requested_root_reference)
    bouts = split_bouts(batch)
    bout_rows: List[Dict[str, object]] = []
    event_rows: List[Dict[str, object]] = []

    # Each status-2 row stores the root-Q state after the corresponding
    # expansion.  Pair it with the previous root-Q state to recover the
    # sampled branch's actual update.  root_ns guards against treating an
    # unvisited branch's placeholder Q as an estimate.
    root_q_after = np.asarray(batch.root_q, dtype=float)
    root_n_after = np.asarray(batch.root_n, dtype=float)
    root_q_before = np.vstack([root.root_q[None, :], root_q_after[:-1]])
    root_n_before = np.vstack([root.root_n[None, :], root_n_after[:-1]])

    for bout in bouts:
        first = int(bout["first"])
        terminal = int(bout["terminal"])
        first_return = float(batch.rollout_return[first])
        terminal_return = float(batch.rollout_return[terminal])
        first_coords = persistence_coordinates(
            policy_trajectory[first],
            incumbent,
            n_actions,
            policy_temperature=policy_temperature,
        )
        terminal_coords = persistence_coordinates(
            policy_trajectory[terminal],
            incumbent,
            n_actions,
            policy_temperature=policy_temperature,
        )
        gain_root = (
            terminal_return - root_ref
            if np.isfinite(terminal_return + root_ref)
            else np.nan
        )
        gain_within = (
            terminal_return - first_return
            if np.isfinite(terminal_return + first_return)
            else np.nan
        )
        bout_record = {
            "bout_index": int(bout["bout_index"]),
            "first_status2_position": first,
            "terminal_status2_position": terminal,
            # Preserve the reset-boundary provenance from split_bouts so the
            # window-level QA summary can quantify exact versus inferred
            # bout boundaries after the rows are flattened.
            "reset_known": int(bout["reset_known"]),
            "branch": int(bout["branch"]),
            "terminal_depth": int(bout["terminal_depth"]),
            "path": "/".join(map(str, bout["path"])),
            "first_rollout_return": first_return,
            "terminal_rollout_return": terminal_return,
            "return_gain_from_root": gain_root,
            "return_gain_within_bout": gain_within,
            "root_reference": root_ref,
            "root_reference_name": root_ref_name,
            "incumbent_branch": int(int(bout["branch"]) == incumbent),
            "presearch_candidate_branch": int(
                int(bout["branch"]) == int(np.argmax(policy_probabilities(presearch_policy, n_actions)))
                if np.all(np.isfinite(policy_probabilities(presearch_policy, n_actions)))
                else -1
            ),
            "start_persistence_evidence": first_coords["persistence_evidence"],
            "start_switch_target_uncertainty": first_coords[
                "switch_target_uncertainty"
            ],
            "terminal_persistence_evidence": terminal_coords[
                "persistence_evidence"
            ],
            "terminal_switch_target_uncertainty": terminal_coords[
                "switch_target_uncertainty"
            ],
            "terminal_persistence_x": terminal_coords["persistence_x"],
            "terminal_persistence_y": terminal_coords["persistence_y"],
            "terminal_delta_persistence_evidence": (
                terminal_coords["persistence_evidence"]
                - pre_coords["persistence_evidence"]
                if np.isfinite(
                    terminal_coords["persistence_evidence"]
                    + pre_coords["persistence_evidence"]
                )
                else np.nan
            ),
            "terminal_delta_switch_target_uncertainty": (
                terminal_coords["switch_target_uncertainty"]
                - pre_coords["switch_target_uncertainty"]
                if np.isfinite(
                    terminal_coords["switch_target_uncertainty"]
                    + pre_coords["switch_target_uncertainty"]
                )
                else np.nan
            ),
            "within_bout_policy_js": jensen_shannon(
                policy_trajectory[first], policy_trajectory[terminal], n_actions
            ),
        }
        bout_rows.append(bout_record)
        for event_position in np.asarray(bout["positions"], dtype=int):
            event_policy_coords = persistence_coordinates(
                policy_trajectory[event_position],
                incumbent,
                n_actions,
                policy_temperature=policy_temperature,
            )
            event_return = float(batch.rollout_return[event_position])
            branch = int(batch.paths[event_position, 0])
            branch_valid = bool(
                batch.path_valid[event_position]
                and 0 <= branch < n_actions
            )
            q_before = (
                float(root_q_before[event_position, branch])
                if branch_valid
                else np.nan
            )
            q_after = (
                float(root_q_after[event_position, branch])
                if branch_valid
                else np.nan
            )
            n_before = (
                float(root_n_before[event_position, branch])
                if branch_valid
                else np.nan
            )
            n_after = (
                float(root_n_after[event_position, branch])
                if branch_valid
                else np.nan
            )
            revision_valid = bool(
                branch_valid
                and np.isfinite(q_before)
                and np.isfinite(q_after)
                and np.isfinite(n_before)
                and np.isfinite(n_after)
                and n_before > EPS
                and n_after > EPS
            )
            q_revision = q_after - q_before if revision_valid else np.nan
            prediction_error = (
                event_return - q_before
                if revision_valid and np.isfinite(event_return)
                else np.nan
            )
            revision_direction = (
                -1.0 if branch == incumbent else 1.0
            ) if branch_valid else np.nan
            event_rows.append(
                {
                    "bout_index": int(bout["bout_index"]),
                    "status2_position": int(event_position),
                    "depth": int(batch.depth[event_position]),
                    "branch": branch,
                    "rollout_return": event_return,
                    "return_gain_from_root": (
                        event_return - root_ref
                        if np.isfinite(event_return + root_ref)
                        else np.nan
                    ),
                    "persistence_x": event_policy_coords["persistence_x"],
                    "persistence_y": event_policy_coords["persistence_y"],
                    "persistence_evidence": event_policy_coords[
                        "persistence_evidence"
                    ],
                    "switch_target_uncertainty": event_policy_coords[
                        "switch_target_uncertainty"
                    ],
                    "root_q_before_sampled": q_before,
                    "root_q_after_sampled": q_after,
                    "root_n_before_sampled": n_before,
                    "root_n_after_sampled": n_after,
                    "root_q_revision": q_revision,
                    "rollout_prediction_error": prediction_error,
                    "revision_direction": revision_direction,
                    "directed_value_revision": (
                        revision_direction * q_revision
                        if revision_valid
                        else np.nan
                    ),
                    "directed_prediction_error": (
                        revision_direction * prediction_error
                        if np.isfinite(revision_direction + prediction_error)
                        else np.nan
                    ),
                    "value_revision_valid": int(revision_valid),
                }
            )

    valid_positions = np.flatnonzero(np.asarray(batch.path_valid, dtype=bool))
    valid_branches = (
        np.asarray(batch.paths[valid_positions, 0], dtype=int)
        if valid_positions.size
        else np.zeros(0, dtype=int)
    )
    bout_branches = np.asarray([int(row["branch"]) for row in bout_rows], dtype=int)
    terminal_depths = np.asarray(
        [float(row["terminal_depth"]) for row in bout_rows], dtype=float
    )
    terminal_returns = np.asarray(
        [float(row["terminal_rollout_return"]) for row in bout_rows], dtype=float
    )
    first_returns = np.asarray(
        [float(row["first_rollout_return"]) for row in bout_rows], dtype=float
    )
    finite_terminal = terminal_returns[np.isfinite(terminal_returns)]
    finite_first = first_returns[np.isfinite(first_returns)]
    finite_gain_root = np.asarray(
        [float(row["return_gain_from_root"]) for row in bout_rows]
    )
    finite_gain_root = finite_gain_root[np.isfinite(finite_gain_root)]
    finite_gain_within = terminal_returns - first_returns
    finite_gain_within = finite_gain_within[np.isfinite(finite_gain_within)]
    counts = np.bincount(valid_branches, minlength=n_actions).astype(float)
    start_counts = np.bincount(bout_branches, minlength=n_actions).astype(float)
    alternatives = np.delete(counts, incumbent)
    alternative_entropy = (
        normalized_entropy(alternatives) if np.sum(alternatives) > 0 else np.nan
    )
    unique_paths = {
        tuple(int(value) for value in path if int(value) >= 0)
        for path in np.asarray(batch.paths[valid_positions], dtype=int)
    }
    unique_path_fraction = (
        len(unique_paths) / len(valid_positions) if valid_positions.size else np.nan
    )
    directed_revisions = np.asarray(
        [row["directed_value_revision"] for row in event_rows], dtype=float
    )
    valid_directed_revisions = directed_revisions[
        np.isfinite(directed_revisions)
    ]
    incumbent_revisions = np.asarray(
        [
            row["root_q_revision"]
            for row in event_rows
            if row["branch"] == incumbent
        ],
        dtype=float,
    )
    incumbent_revisions = incumbent_revisions[np.isfinite(incumbent_revisions)]
    alternative_revisions = np.asarray(
        [
            row["root_q_revision"]
            for row in event_rows
            if row["branch"] != incumbent
        ],
        dtype=float,
    )
    alternative_revisions = alternative_revisions[np.isfinite(alternative_revisions)]
    return_record: Dict[str, object] = {
        "previous_human_action": incumbent,
        "human_action": int(current_action),
        "switch": int(current_action != incumbent),
        "incumbent_action": incumbent,
        "root_value": root.root_value,
        "root_rollout_return": root.root_rollout_return,
        "root_max_rollout_return": root.root_max_rollout_return,
        "root_reference": root_ref,
        "root_reference_name": root_ref_name,
        "presearch_candidate": (
            int(np.argmax(policy_probabilities(presearch_policy, n_actions)))
            if np.all(np.isfinite(policy_probabilities(presearch_policy, n_actions)))
            else -1
        ),
        **pre_coords,
        "n_status1": int(batch.n_status1),
        "n_status2": int(batch.n_status2),
        "n_status3": int(batch.n_status3),
        "n_valid_expansions": int(valid_positions.size),
        "valid_expansion_fraction": (
            float(valid_positions.size / batch.n_status2)
            if batch.n_status2
            else np.nan
        ),
        "n_valid_value_revision_events": int(valid_directed_revisions.size),
        "value_revision_event_fraction": (
            float(valid_directed_revisions.size / batch.n_status2)
            if batch.n_status2
            else np.nan
        ),
        "net_value_revision_toward_alternatives_raw": (
            float(np.mean(valid_directed_revisions))
            if valid_directed_revisions.size
            else np.nan
        ),
        "mean_incumbent_root_q_revision": (
            float(np.mean(incumbent_revisions))
            if incumbent_revisions.size
            else np.nan
        ),
        "mean_alternative_root_q_revision": (
            float(np.mean(alternative_revisions))
            if alternative_revisions.size
            else np.nan
        ),
        "reset_count": int(sum(int(batch.cur_reset[pos]) == 1 for pos in range(len(batch.cur_reset))))
        if len(batch.cur_reset) == batch.n_status2
        else np.nan,
        "n_bouts": int(len(bout_rows)),
        "incumbent_start_share": (
            float(np.mean(bout_branches == incumbent)) if bout_branches.size else np.nan
        ),
        "incumbent_node_share": (
            float(np.mean(valid_branches == incumbent))
            if valid_branches.size
            else np.nan
        ),
        "incumbent_start_logratio": (
            float(np.log((start_counts[incumbent] + 0.5) / (np.sum(start_counts) - start_counts[incumbent] + 0.5)))
            if bout_branches.size
            else np.nan
        ),
        "incumbent_node_logratio": (
            float(np.log((counts[incumbent] + 0.5) / (np.sum(counts) - counts[incumbent] + 0.5)))
            if valid_branches.size
            else np.nan
        ),
        "continuation_allocation": np.nan,
        "alternative_branch_entropy": alternative_entropy,
        "root_branch_allocation_diversity": effective_branching(valid_branches, n_actions),
        "root_bout_branch_diversity": effective_branching(bout_branches, n_actions),
        "mean_bout_terminal_depth": (
            float(np.mean(terminal_depths)) if terminal_depths.size else np.nan
        ),
        "deep_bout_fraction": (
            float(np.mean(terminal_depths >= 3)) if terminal_depths.size else np.nan
        ),
        "deep_event_fraction": (
            float(np.mean(np.asarray(batch.depth[valid_positions]) >= 3))
            if valid_positions.size
            else np.nan
        ),
        "path_revisit_fraction": (
            1.0 - unique_path_fraction
            if np.isfinite(unique_path_fraction)
            else np.nan
        ),
        "reset_reorientation": (
            float(np.mean(bout_branches[1:] != bout_branches[:-1]))
            if bout_branches.size >= 2
            else np.nan
        ),
        "terminal_return_mean": (
            float(np.mean(finite_terminal)) if finite_terminal.size else np.nan
        ),
        "terminal_return_max": (
            float(np.max(finite_terminal)) if finite_terminal.size else np.nan
        ),
        "terminal_return_sd": (
            float(np.std(finite_terminal)) if finite_terminal.size else np.nan
        ),
        "return_gain_from_root": (
            float(np.mean(finite_gain_root)) if finite_gain_root.size else np.nan
        ),
        "return_gain_within_bout": (
            float(np.mean(finite_gain_within)) if finite_gain_within.size else np.nan
        ),
        "incumbent_terminal_return": (
            float(np.mean([
                row["terminal_rollout_return"]
                for row in bout_rows
                if row["incumbent_branch"] == 1
                and np.isfinite(row["terminal_rollout_return"])
            ]))
            if any(
                row["incumbent_branch"] == 1
                and np.isfinite(row["terminal_rollout_return"])
                for row in bout_rows
            )
            else np.nan
        ),
        "alternative_terminal_return": (
            float(np.mean([
                row["terminal_rollout_return"]
                for row in bout_rows
                if row["incumbent_branch"] == 0
                and np.isfinite(row["terminal_rollout_return"])
            ]))
            if any(
                row["incumbent_branch"] == 0
                and np.isfinite(row["terminal_rollout_return"])
                for row in bout_rows
            )
            else np.nan
        ),
        "presearch_candidate_terminal_share": (
            float(np.mean([
                row["presearch_candidate_branch"] == 1 for row in bout_rows
                if row["presearch_candidate_branch"] >= 0
            ]))
            if any(row["presearch_candidate_branch"] >= 0 for row in bout_rows)
            else np.nan
        ),
    }
    final_root_q = root_q_after[-1] if len(root_q_after) else root.root_q
    final_root_n = root_n_after[-1] if len(root_n_after) else root.root_n
    for action in range(n_actions):
        return_record[f"root_q_initial_{action}"] = float(root.root_q[action])
        return_record[f"root_q_final_{action}"] = float(final_root_q[action])
        return_record[f"root_n_initial_{action}"] = float(root.root_n[action])
        return_record[f"root_n_final_{action}"] = float(final_root_n[action])
    return_record["continuation_allocation"] = (
        return_record["incumbent_node_logratio"]
        - return_record["incumbent_start_logratio"]
        if np.isfinite(
            return_record["incumbent_node_logratio"]
            + return_record["incumbent_start_logratio"]
        )
        else np.nan
    )
    alt_returns = [
        row["terminal_rollout_return"]
        for row in bout_rows
        if row["incumbent_branch"] == 0
        and np.isfinite(row["terminal_rollout_return"])
    ]
    inc_returns = [
        row["terminal_rollout_return"]
        for row in bout_rows
        if row["incumbent_branch"] == 1
        and np.isfinite(row["terminal_rollout_return"])
    ]
    return_record["incumbent_minus_alternative_return"] = (
        float(np.mean(inc_returns) - np.mean(alt_returns))
        if inc_returns and alt_returns
        else np.nan
    )
    return return_record, bout_rows, event_rows


def trace_signature(blocks: Mapping[object, Sequence[object]]) -> List[Dict[str, object]]:
    signature: List[Dict[str, object]] = []
    for metas in blocks.values():
        for meta in metas:
            stat = meta.path.stat()
            signature.append(
                {
                    "path": str(meta.path.resolve()),
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                }
            )
    return signature


def extract_analysis(
    blocks: Mapping[Tuple[int, int, int, int], Sequence[object]],
    max_files: Optional[int],
    policy_source: str,
    root_reference_name: str,
    policy_temperature: float = 1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    window_rows: List[Dict[str, object]] = []
    bout_rows: List[Dict[str, object]] = []
    event_rows: List[Dict[str, object]] = []
    loaded_files = 0
    n_actions_global: Optional[int] = None
    cross_chunk_windows = 0

    for block_number, metas_all in enumerate(blocks.values(), start=1):
        metas = list(metas_all[:max_files] if max_files is not None else metas_all)
        previous_root: Optional[RootState] = None
        carry = None
        actor_carry: Optional[np.ndarray] = None
        real_step = 0
        log(f"Block {block_number}/{len(blocks)}: {len(metas)} chunks")
        for meta in metas:
            data = TRACE.load_npy_dict(meta.path)
            loaded_files += 1
            status = np.asarray(data.get("status", []), dtype=int).reshape(-1)
            tree = data.get("tree_reps")
            if status.size == 0 or not isinstance(tree, Mapping):
                raise ValueError(f"Missing status/tree_reps in {meta.path}")
            n_actions = TRACE.infer_num_actions(tree)
            if n_actions_global is None:
                n_actions_global = n_actions
            elif n_actions != n_actions_global:
                raise ValueError(f"Action count changed in {meta.path}")
            if policy_source == "actor" and data.get("actor_policy") is None:
                raise ValueError(f"Actor policy is missing in {meta.path}")

            real_indices = np.flatnonzero(status == REAL_STATUS)
            previous_local_real: Optional[int] = None
            for current_raw in real_indices:
                current_real = int(current_raw)
                real_step += 1
                start = (
                    0
                    if previous_local_real is None
                    else previous_local_real + 1
                )
                between = np.arange(start, current_real, dtype=int)
                current_batch = TRACE.build_analysis_batch(
                    tree, status, between, n_actions
                )
                batch = TRACE.concatenate_analysis_batches(
                    carry if previous_local_real is None else None,
                    current_batch,
                )
                local_actor = extract_actor_rows(data, status, between, n_actions)
                actor_rows = concatenate_policy_rows(
                    actor_carry if previous_local_real is None else None,
                    local_actor,
                )
                crossed_chunk = int(
                    previous_root is not None
                    and previous_local_real is None
                    and previous_root.source_chunk != meta.chunk
                )
                if (
                    previous_root is not None
                    and batch is not None
                    and int(batch.n_status2) > 0
                ):
                    current_action = TRACE.decode_action(
                        data.get("human_action"), current_real
                    )
                    if 0 <= current_action < n_actions and 0 <= previous_root.human_action < n_actions:
                        policy_rows = (
                            actor_rows
                            if policy_source == "actor"
                            else np.asarray(batch.cur_policy, dtype=float)
                        )
                        metrics, local_bouts, local_events = summarize_window(
                            previous_root,
                            batch,
                            policy_rows,
                            current_action,
                            policy_source,
                            root_reference_name,
                            n_actions,
                            policy_temperature,
                        )
                        metadata = {
                            "subject": int(meta.subject),
                            "session": int(meta.session),
                            "block": int(meta.block),
                            "game": int(meta.game),
                            "run_label": str(meta.run_label),
                            "chunk": int(meta.chunk),
                            "real_step": int(real_step),
                            "real_step_id": f"{meta.run_label}_real{real_step:06d}",
                            "cross_chunk_boundary": crossed_chunk,
                            "policy_source": policy_source,
                        }
                        window_rows.append({**metadata, **metrics})
                        for row in local_bouts:
                            bout_rows.append({**metadata, **row})
                        for row in local_events:
                            event_rows.append({**metadata, **row})
                        cross_chunk_windows += crossed_chunk
                previous_root = make_root(
                    data, tree, current_real, n_actions, int(meta.chunk)
                )
                carry = None
                actor_carry = None
                previous_local_real = current_real

            tail_start = (
                int(previous_local_real) + 1
                if previous_local_real is not None
                else 0
            )
            tail_indices = np.arange(tail_start, len(status), dtype=int)
            tail = TRACE.build_analysis_batch(tree, status, tail_indices, n_actions)
            carry = TRACE.concatenate_analysis_batches(carry, tail)
            tail_actor = extract_actor_rows(data, status, tail_indices, n_actions)
            actor_carry = concatenate_policy_rows(actor_carry, tail_actor)
            del data, tree, status
            gc.collect()

    if not window_rows or n_actions_global is None:
        raise RuntimeError("No real-step windows with status-2 imagination were extracted")
    windows = pd.DataFrame(window_rows).sort_values(
        ["run_label", "real_step"]
    ).reset_index(drop=True)
    bouts = pd.DataFrame(bout_rows)
    events = pd.DataFrame(event_rows)
    if not bouts.empty:
        bouts = bouts.sort_values(
            ["run_label", "real_step", "bout_index"]
        ).reset_index(drop=True)
    if not events.empty:
        events = events.sort_values(
            ["run_label", "real_step", "bout_index", "status2_position"]
        ).reset_index(drop=True)
    qa = {
        "schema_version": SCHEMA_VERSION,
        "loaded_files": int(loaded_files),
        "n_actions": int(n_actions_global),
        "n_real_step_windows": int(len(windows)),
        "n_bouts": int(len(bouts)),
        "n_events": int(len(events)),
        "n_runs": int(windows["run_label"].nunique()),
        "cross_chunk_windows": int(cross_chunk_windows),
        "policy_source": policy_source,
        "policy_temperature": float(policy_temperature),
        "root_reference_requested": root_reference_name,
        "root_reference_used": windows["root_reference_name"].value_counts(dropna=False).to_dict(),
        "status2_median": float(windows["n_status2"].median()),
        "valid_expansion_fraction_mean": float(windows["valid_expansion_fraction"].mean()),
        "return_gain_coverage": float(windows["return_gain_from_root"].notna().mean()),
        "value_revision_window_coverage": float(
            windows["net_value_revision_toward_alternatives_raw"].notna().mean()
        ),
        "value_revision_event_fraction_mean": float(
            windows["value_revision_event_fraction"].mean()
        ),
        "reset_exact_fraction": float(
            bouts["reset_known"].mean() if not bouts.empty else np.nan
        ),
    }
    return windows, bouts, events, qa


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log(f"Saved {path}")


def finite_bout_data(bouts: pd.DataFrame) -> pd.DataFrame:
    if bouts.empty:
        return bouts.copy()
    required = [
        "cross_chunk_boundary",
        "root_persistence_x",
        "root_persistence_y",
        "terminal_persistence_x",
        "terminal_persistence_y",
        "return_gain_from_root",
    ]
    data = bouts.loc[
        bouts["cross_chunk_boundary"].eq(0)
        & bouts[required[1:]].notna().all(axis=1)
    ].copy()
    return data


def add_root_coordinates(windows: pd.DataFrame, bouts: pd.DataFrame) -> pd.DataFrame:
    if bouts.empty:
        return bouts
    root = windows[
        ["real_step_id", "persistence_x", "persistence_y"]
    ].rename(
        columns={
            "persistence_x": "root_persistence_x",
            "persistence_y": "root_persistence_y",
        }
    )
    return bouts.drop(
        columns=["root_persistence_x", "root_persistence_y"], errors="ignore"
    ).merge(root, on="real_step_id", how="left", validate="many_to_one")


def _load_canonical_base_metrics(
    subject: int,
    game: int,
    sessions: Optional[set[int]],
    shared_cache_root: Path,
) -> Tuple[pd.DataFrame, Path]:
    """Load canonical 02 rows from the read-only shared cache."""
    paths = shared_cache_paths(shared_cache_root, subject, game)
    base = load_actor_metrics(paths)
    source = paths.actor_metrics
    if sessions is not None:
        base = base.loc[base["session"].isin(sorted(sessions))].copy()
    if base.empty:
        raise RuntimeError(f"No canonical 02 rows remain after filtering {source}")
    return base, source


def attach_branch_relative_value_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach future-blind and observed-action branch-return contrasts.

    Every branch value is the mean terminal raw rollout return across
    reset-delimited imaginary bouts beginning with that depth-1 action in the
    same real-step window. Missing branches remain missing rather than being
    replaced by zero.
    """
    result = frame.copy()
    value_columns = [
        f"branch_terminal_return_mean_{action}" for action in range(6)
    ]
    values = result[value_columns].to_numpy(dtype=float)
    incumbent = pd.to_numeric(
        result["previous_human_action"], errors="coerce"
    ).to_numpy(dtype=float)
    observed = pd.to_numeric(
        result["human_action"], errors="coerce"
    ).to_numpy(dtype=float)
    n_rows = len(result)
    incumbent_value = np.full(n_rows, np.nan, dtype=float)
    observed_value = np.full(n_rows, np.nan, dtype=float)
    valid_incumbent = np.isfinite(incumbent) & (incumbent >= 0) & (incumbent < 6)
    valid_observed = np.isfinite(observed) & (observed >= 0) & (observed < 6)
    row_index = np.arange(n_rows)
    incumbent_integer = np.full(n_rows, -1, dtype=int)
    observed_integer = np.full(n_rows, -1, dtype=int)
    incumbent_integer[valid_incumbent] = incumbent[valid_incumbent].astype(int)
    observed_integer[valid_observed] = observed[valid_observed].astype(int)
    incumbent_value[valid_incumbent] = values[
        row_index[valid_incumbent], incumbent_integer[valid_incumbent]
    ]
    observed_value[valid_observed] = values[
        row_index[valid_observed], observed_integer[valid_observed]
    ]

    alternatives = values.copy()
    alternatives[row_index[valid_incumbent], incumbent_integer[valid_incumbent]] = np.nan
    alternative_count = np.sum(np.isfinite(alternatives), axis=1).astype(int)
    best_alternative_action = np.full(n_rows, -1, dtype=int)
    best_alternative_value = np.full(n_rows, np.nan, dtype=float)
    has_alternative = alternative_count > 0
    safe_alternatives = np.where(np.isfinite(alternatives), alternatives, -np.inf)
    best_alternative_action[has_alternative] = np.argmax(
        safe_alternatives[has_alternative], axis=1
    )
    best_alternative_value[has_alternative] = safe_alternatives[
        row_index[has_alternative], best_alternative_action[has_alternative]
    ]

    active_values = values[:, 1:]
    best_active_value = np.full(n_rows, np.nan, dtype=float)
    has_active = np.any(np.isfinite(active_values), axis=1)
    best_active_value[has_active] = np.max(
        np.where(np.isfinite(active_values[has_active]), active_values[has_active], -np.inf),
        axis=1,
    )
    other_active_values = active_values.copy()
    active_incumbent = valid_incumbent & (incumbent_integer >= 1)
    other_active_values[
        row_index[active_incumbent], incumbent_integer[active_incumbent] - 1
    ] = np.nan
    best_other_active_value = np.full(n_rows, np.nan, dtype=float)
    has_other_active = np.any(np.isfinite(other_active_values), axis=1)
    best_other_active_value[has_other_active] = np.max(
        np.where(
            np.isfinite(other_active_values[has_other_active]),
            other_active_values[has_other_active],
            -np.inf,
        ),
        axis=1,
    )

    result["incumbent_branch_return"] = incumbent_value
    result["observed_action_branch_return"] = observed_value
    result["observed_action_value_advantage"] = observed_value - incumbent_value
    result["best_alternative_action"] = best_alternative_action
    result["best_alternative_branch_return"] = best_alternative_value
    result["best_alternative_value_advantage"] = (
        best_alternative_value - incumbent_value
    )
    result["n_observed_alternative_branches"] = alternative_count
    result["best_active_value_advantage_over_noop"] = (
        best_active_value - values[:, 0]
    )
    result["noop_value_advantage_over_incumbent"] = values[:, 0] - incumbent_value
    result["best_other_active_value_advantage_over_incumbent"] = (
        best_other_active_value - incumbent_value
    )
    is_switch = (
        valid_incumbent
        & valid_observed
        & (incumbent_integer != observed_integer)
    )
    match = np.full(n_rows, np.nan, dtype=float)
    match[is_switch & has_alternative] = (
        best_alternative_action[is_switch & has_alternative]
        == observed_integer[is_switch & has_alternative]
    ).astype(float)
    result["best_alternative_matches_observed_switch"] = match
    return result


def add_directional_value_revision_balance(
    windows: pd.DataFrame,
    events: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Attach a bounded, event-weighted directional Q-revision balance.

    For every real-step search window, valid sampled-branch root-Q revisions
    are signed positively for alternative branches and negatively for the
    incumbent branch.  Their signed sum is divided by the total absolute
    revision magnitude:

        B = sum(d * delta_Q) / (sum(abs(delta_Q)) + EPS)

    The result lies in [-1, 1], preserves zero and direction, and no longer
    depends on the empirical dispersion of other windows in the same run.
    The absolute revision magnitude is retained separately so a strongly
    directional but numerically tiny update can be identified downstream.
    """
    keys = ["run_label", "real_step", "real_step_id"]
    required = keys + [
        "root_q_revision",
        "directed_value_revision",
        "value_revision_valid",
    ]
    missing = [column for column in required if column not in events.columns]
    if missing:
        raise KeyError(
            "Event table is missing value-revision columns: "
            + ", ".join(missing)
        )

    event_values = events[required].copy()
    event_values["root_q_revision"] = pd.to_numeric(
        event_values["root_q_revision"], errors="coerce"
    )
    event_values["directed_value_revision"] = pd.to_numeric(
        event_values["directed_value_revision"], errors="coerce"
    )
    event_values["value_revision_valid"] = pd.to_numeric(
        event_values["value_revision_valid"], errors="coerce"
    )
    valid = (
        event_values["value_revision_valid"].eq(1)
        & np.isfinite(event_values["root_q_revision"])
        & np.isfinite(event_values["directed_value_revision"])
    )
    event_values = event_values.loc[valid].copy()
    event_values["absolute_root_q_revision"] = event_values[
        "root_q_revision"
    ].abs()

    if event_values.empty:
        raise RuntimeError("No valid event-level root-Q revisions were found")

    summary = (
        event_values.groupby(keys, as_index=False)
        .agg(
            value_revision_balance_numerator=(
                "directed_value_revision",
                "sum",
            ),
            value_revision_balance_denominator=(
                "absolute_root_q_revision",
                "sum",
            ),
            value_revision_magnitude_mean=(
                "absolute_root_q_revision",
                "mean",
            ),
            value_revision_balance_event_count=(
                "directed_value_revision",
                "count",
            ),
        )
    )
    summary["net_value_revision_toward_alternatives"] = (
        summary["value_revision_balance_numerator"]
        / (summary["value_revision_balance_denominator"] + EPS)
    )
    balance = summary["net_value_revision_toward_alternatives"].to_numpy(
        dtype=float
    )
    if np.nanmax(np.abs(balance)) > 1.0 + 1e-7:
        raise RuntimeError("Directional value-revision balance exceeded [-1, 1]")

    result = windows.copy()
    if "net_value_revision_toward_alternatives_raw" in result.columns:
        result["legacy_mean_signed_value_revision"] = result[
            "net_value_revision_toward_alternatives_raw"
        ]
    added_columns = [
        "value_revision_balance_numerator",
        "value_revision_balance_denominator",
        "value_revision_magnitude_mean",
        "value_revision_balance_event_count",
        "net_value_revision_toward_alternatives",
    ]
    result = result.drop(columns=added_columns, errors="ignore").merge(
        summary,
        on=keys,
        how="left",
        validate="one_to_one",
    )
    finite_balance = pd.to_numeric(
        result["net_value_revision_toward_alternatives"], errors="coerce"
    )
    diagnostics = {
        "definition": (
            "sum(directed sampled-branch root-Q revision) divided by "
            "sum(absolute sampled-branch root-Q revision) plus EPS"
        ),
        "range": "[-1, 1]",
        "zero_and_sign_preserved": True,
        "run_distribution_scaling": False,
        "window_coverage": float(finite_balance.notna().mean()),
        "finite_min": float(finite_balance.min()),
        "finite_max": float(finite_balance.max()),
        "mean_valid_events_per_covered_window": float(
            pd.to_numeric(
                result["value_revision_balance_event_count"], errors="coerce"
            ).mean()
        ),
    }
    return result, diagnostics


def apply_shifted_actor_current_action_coordinates(
    windows: pd.DataFrame,
    actor_metrics: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Replace same-index actor coordinates with run-safe shifted ones.

    ``02_actor_policy.py`` establishes that an actor-policy window ending at
    logged real step k belongs to behavioral step k-1.  The shared actor cache
    therefore shifts the full policy window first and only then selects the
    target row's previous human action.  Tree/Q events remain on their original
    behavioral row; only the actor-policy coordinate columns are replaced.
    """
    keys = ["run_label", "real_step", "real_step_id"]
    source_columns = keys + [
        "actor_policy_source_real_step",
        "actor_policy_start_candidate",
        "actor_current_action_margin_start",
        "actor_current_action_uncertainty_start",
        "actor_current_action_probability_start",
        "actor_current_action_mass_logodds_start",
        "actor_current_action_margin_terminal",
        "actor_current_action_uncertainty_terminal",
        "actor_current_action_margin_search",
        "actor_current_action_uncertainty_search",
        "actor_current_action_coordinate_valid",
    ]
    missing = [column for column in source_columns if column not in actor_metrics]
    if missing:
        raise KeyError(
            "Shared actor cache lacks shifted incumbent coordinates: "
            + ", ".join(missing)
        )
    aligned = actor_metrics[source_columns].copy()
    result = windows.drop(
        columns=[column for column in source_columns if column not in keys],
        errors="ignore",
    ).merge(aligned, on=keys, how="left", validate="one_to_one")

    start_margin = pd.to_numeric(
        result["actor_current_action_margin_start"], errors="coerce"
    )
    start_uncertainty = pd.to_numeric(
        result["actor_current_action_uncertainty_start"], errors="coerce"
    )
    terminal_margin = pd.to_numeric(
        result["actor_current_action_margin_terminal"], errors="coerce"
    )
    terminal_uncertainty = pd.to_numeric(
        result["actor_current_action_uncertainty_terminal"], errors="coerce"
    )
    result["persistence_evidence"] = start_margin
    result["strongest_competitor_margin"] = start_margin
    result["switch_target_uncertainty"] = start_uncertainty
    result["persistence_mass_logodds"] = pd.to_numeric(
        result["actor_current_action_mass_logodds_start"], errors="coerce"
    )
    result["current_action_probability"] = pd.to_numeric(
        result["actor_current_action_probability_start"], errors="coerce"
    )
    result["persistence_x"] = 1.0 / (
        1.0 + np.exp(-np.clip(start_margin.to_numpy(dtype=float), -50.0, 50.0))
    )
    result["persistence_y"] = start_uncertainty
    result["terminal_persistence_evidence"] = terminal_margin
    result["terminal_switch_target_uncertainty"] = terminal_uncertainty
    result["terminal_persistence_x"] = 1.0 / (
        1.0
        + np.exp(
            -np.clip(terminal_margin.to_numpy(dtype=float), -50.0, 50.0)
        )
    )
    result["terminal_persistence_y"] = terminal_uncertainty
    result["search_persistence_evidence"] = pd.to_numeric(
        result["actor_current_action_margin_search"], errors="coerce"
    )
    result["search_switch_target_uncertainty"] = pd.to_numeric(
        result["actor_current_action_uncertainty_search"], errors="coerce"
    )
    result["presearch_candidate"] = pd.to_numeric(
        result["actor_policy_start_candidate"], errors="coerce"
    )

    finite = result[
        [
            "persistence_x",
            "switch_target_uncertainty",
            "terminal_persistence_x",
            "terminal_switch_target_uncertainty",
        ]
    ].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    source_step = pd.to_numeric(
        result["actor_policy_source_real_step"], errors="coerce"
    )
    target_step = pd.to_numeric(result["real_step"], errors="coerce")
    source_relation = source_step.eq(target_step + 1)
    diagnostics = {
        "definition": (
            "full actor-policy window ending at logged real step k is attached "
            "within run to behavioral step k-1; current-action coordinates are "
            "computed only after this shift"
        ),
        "matched_fraction": float(finite.mean()),
        "source_is_next_real_step_fraction_among_matched": float(
            source_relation.loc[finite].mean() if finite.any() else np.nan
        ),
        "unmatched_rows": int((~finite).sum()),
        "expected_unmatched_structure": (
            "at most the final behavioral row of each run lacks a future "
            "completed actor-policy window"
        ),
    }
    return result, diagnostics


def build_canonical_persistence_metrics(
    windows: pd.DataFrame,
    bouts: pd.DataFrame,
    events: pd.DataFrame,
    subject: int,
    game: int,
    sessions: Optional[set[int]],
    shared_cache_root: Path,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Attach incumbent-relative coordinates to canonical 02 analysis rows."""
    keys = ["run_label", "real_step", "real_step_id"]
    coordinate_columns = keys + [
        "persistence_evidence",
        "switch_target_uncertainty",
        "strongest_competitor_margin",
        "persistence_mass_logodds",
        "current_action_probability",
        "persistence_x",
        "persistence_y",
        "terminal_persistence_evidence",
        "terminal_switch_target_uncertainty",
        "terminal_persistence_x",
        "terminal_persistence_y",
        "search_persistence_evidence",
        "search_switch_target_uncertainty",
        "terminal_return_mean",
        "n_valid_value_revision_events",
        "value_revision_event_fraction",
        "net_value_revision_toward_alternatives_raw",
        "legacy_mean_signed_value_revision",
        "value_revision_balance_numerator",
        "value_revision_balance_denominator",
        "value_revision_magnitude_mean",
        "value_revision_balance_event_count",
        "net_value_revision_toward_alternatives",
        "mean_incumbent_root_q_revision",
        "mean_alternative_root_q_revision",
    ]
    coordinate = windows[coordinate_columns].copy()
    if bouts.empty:
        raise RuntimeError("Bout rows are required for branch-relative value evidence")
    branch_summary = (
        bouts.assign(
            branch=pd.to_numeric(bouts["branch"], errors="coerce"),
            terminal_rollout_return=pd.to_numeric(
                bouts["terminal_rollout_return"], errors="coerce"
            ),
        )
        .loc[lambda frame: frame["branch"].between(0, 5)]
        .groupby(keys + ["branch"], as_index=False)
        .agg(
            branch_terminal_return_mean=("terminal_rollout_return", "mean"),
            branch_terminal_return_count=("terminal_rollout_return", "count"),
        )
    )
    branch_means = branch_summary.pivot(
        index=keys,
        columns="branch",
        values="branch_terminal_return_mean",
    ).reindex(columns=range(6))
    branch_counts = branch_summary.pivot(
        index=keys,
        columns="branch",
        values="branch_terminal_return_count",
    ).reindex(columns=range(6))
    branch_means.columns = [
        f"branch_terminal_return_mean_{action}" for action in range(6)
    ]
    branch_counts.columns = [
        f"branch_terminal_return_count_{action}" for action in range(6)
    ]
    branch_values = branch_means.join(branch_counts).reset_index()
    count_columns = [
        f"branch_terminal_return_count_{action}" for action in range(6)
    ]
    branch_values[count_columns] = branch_values[count_columns].fillna(0).astype(int)
    branch_values["n_observed_return_branches"] = (
        branch_values[count_columns].gt(0).sum(axis=1).astype(int)
    )
    coordinate = coordinate.merge(
        branch_values,
        on=keys,
        how="left",
        validate="one_to_one",
    )

    base, source = _load_canonical_base_metrics(
        subject,
        game,
        sessions,
        shared_cache_root,
    )
    overlap = sorted(set(base.columns).intersection(coordinate.columns) - set(keys))
    if overlap:
        base = base.drop(columns=overlap)
    merged = base.merge(
        coordinate,
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise RuntimeError("Persistence windows did not align with canonical 02 rows")

    # Canonical plotting/statistical functions consume these aliases.  Only
    # their coordinate meaning changes; the behavior, outcomes, controls,
    # transition definitions, and run-level inference remain untouched.
    merged["slack_policy"] = merged["persistence_evidence"]
    merged["slack_policy_normalized"] = merged["persistence_x"]
    merged["d_policy_root_start"] = merged["switch_target_uncertainty"]
    merged["d_policy_search"] = merged["search_switch_target_uncertainty"]
    merged["d_policy_within"] = merged["search_switch_target_uncertainty"]
    merged["d_policy_between"] = np.nan
    merged["d_policy_search_unique"] = merged[
        "search_switch_target_uncertainty"
    ]
    merged["d_policy_root_auc"] = merged["search_switch_target_uncertainty"]
    merged["final_cur_policy_slack_logit"] = merged[
        "terminal_persistence_evidence"
    ]
    merged["final_cur_policy_slack"] = merged["terminal_persistence_x"]
    merged["final_cur_policy_demand"] = merged[
        "terminal_switch_target_uncertainty"
    ]
    merged["final_minus_root_policy_slack"] = (
        merged["terminal_persistence_x"] - merged["persistence_x"]
    )
    merged["final_minus_root_policy_demand"] = (
        merged["terminal_switch_target_uncertainty"]
        - merged["switch_target_uncertainty"]
    )
    merged["policy_resolution"] = -merged["final_minus_root_policy_demand"]
    merged["actor_slack_start"] = merged["persistence_evidence"]
    merged["actor_demand_start"] = merged["switch_target_uncertainty"]
    merged["actor_slack_terminal"] = merged[
        "terminal_persistence_evidence"
    ]
    merged["actor_demand_terminal"] = merged[
        "terminal_switch_target_uncertainty"
    ]
    merged["actor_demand_search"] = merged[
        "search_switch_target_uncertainty"
    ]
    merged["actor_demand_within"] = merged[
        "search_switch_target_uncertainty"
    ]
    merged["actor_demand_between"] = np.nan
    merged["actor_demand_auc"] = merged["search_switch_target_uncertainty"]
    merged = attach_branch_relative_value_metrics(merged)

    finite = merged[
        [
            "persistence_x",
            "switch_target_uncertainty",
            "terminal_persistence_x",
            "terminal_switch_target_uncertainty",
        ]
    ].notna().all(axis=1)
    qa = {
        "canonical_base": str(source),
        "canonical_base_rows": int(len(base)),
        "persistence_window_rows": int(len(windows)),
        "merged_rows": int(len(merged)),
        "window_merge_fraction": float(len(merged) / max(len(windows), 1)),
        "canonical_merge_fraction": float(len(merged) / max(len(base), 1)),
        "finite_coordinate_fraction": float(finite.mean()),
        "value_revision_window_coverage": float(
            merged["net_value_revision_toward_alternatives"].notna().mean()
        ),
        "value_revision_event_fraction_mean": float(
            merged["value_revision_event_fraction"].mean()
        ),
        "value_revision_scaling": (
            "bounded directional balance: sum(directed root-Q revision) "
            "divided by sum(absolute root-Q revision) plus EPS; no run-level "
            "distribution scaling; zero and sign are preserved"
        ),
        "coordinate_replacement_only": False,
        "value_revision_added": True,
    }
    return merged, qa


def configure_canonical_persistence_display() -> None:
    """Configure reusable 02 plots for incumbent-relative coordinates."""
    TRACE.GEOMETRY_DISPLAY.update(
        {
            "x_term": "Current-Action Margin",
            "y_term": "Alternative Uncertainty",
            "x_short": "S",
            "y_short": "U",
            "start_x_label": "Normalized Start Current-Action Margin",
            "start_y_label": "Start Alternative-Action Uncertainty",
            "start_title": "Search-Start Margin--Uncertainty State",
            "main1_title": (
                "Search-Start Margin--Uncertainty State and Search Geometry"
            ),
            "main3_title": (
                "Search-Start Margin--Uncertainty State Predicts Search Geometry"
            ),
            "trajectory_x_label": "Current-Action Margin",
            "trajectory_y_label": "Alternative-Action Uncertainty",
            "trajectory_coordinate_label": "margin--uncertainty coordinate",
            "trajectory_title_prefix": (
                "Start-aligned Margin--Uncertainty trajectories"
            ),
            "model_name": "margin--uncertainty",
        }
    )
    TRACE.INTEGRATED_COORDINATES = (
        (
            "actor_start",
            "state_x",
            "state_y",
            "Search-start Margin--Uncertainty State",
            "Normalized Start Current-Action Margin",
            "Start Alternative-Action Uncertainty",
        ),
    )
    TRACE.COMPACT_COORDINATE_LABELS = {
        "actor_start": "Search-Start Margin--Uncertainty State",
    }


def _transition_3d_endpoint_tables(
    selected: pd.DataFrame,
    z_column: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """Build exact first/last 3D endpoints and run-equalized summaries."""
    group_columns = [
        "run_label",
        "transition_class",
        "bout_action",
        "destination_action",
        "action_bout_id",
    ]
    ordered = selected.sort_values(
        group_columns + ["root_position"], kind="stable"
    ).copy()
    if z_column is not None:
        if z_column not in ordered:
            raise KeyError(f"Missing requested z coordinate: {z_column}")
        ordered["_z_coordinate"] = pd.to_numeric(
            ordered[z_column], errors="coerce"
        )
    else:
        destination_advantage = np.full(len(ordered), np.nan, dtype=float)
        incumbent = pd.to_numeric(
            ordered["bout_action"], errors="coerce"
        ).to_numpy(dtype=float)
        destination = pd.to_numeric(
            ordered["destination_action"], errors="coerce"
        ).to_numpy(dtype=float)
        incumbent_integer = np.full(len(ordered), -1, dtype=int)
        destination_integer = np.full(len(ordered), -1, dtype=int)
        incumbent_valid = (
            np.isfinite(incumbent) & (incumbent >= 0) & (incumbent < 6)
        )
        destination_valid = (
            np.isfinite(destination) & (destination >= 0) & (destination < 6)
        )
        incumbent_integer[incumbent_valid] = incumbent[incumbent_valid].astype(int)
        destination_integer[destination_valid] = destination[destination_valid].astype(int)
        for incumbent_action in range(6):
            incumbent_column = f"branch_terminal_return_mean_{incumbent_action}"
            for destination_action in range(6):
                destination_column = f"branch_terminal_return_mean_{destination_action}"
                mask = (
                    (incumbent_integer == incumbent_action)
                    & (destination_integer == destination_action)
                )
                if not np.any(mask):
                    continue
                destination_advantage[mask] = (
                    pd.to_numeric(
                        ordered.loc[mask, destination_column], errors="coerce"
                    ).to_numpy(dtype=float)
                    - pd.to_numeric(
                        ordered.loc[mask, incumbent_column], errors="coerce"
                    ).to_numpy(dtype=float)
                )
        ordered["_z_coordinate"] = destination_advantage

    starts = (
        ordered.groupby(group_columns, sort=False, as_index=False)
        .head(1)[
            group_columns
            + ["state_x", "state_y", "_z_coordinate"]
        ]
        .rename(
            columns={
                "state_x": "start_x",
                "state_y": "start_y",
                "_z_coordinate": "start_z",
            }
        )
    )
    switches = (
        ordered.groupby(group_columns, sort=False, as_index=False)
        .tail(1)[
            group_columns
            + ["state_x", "state_y", "_z_coordinate"]
        ]
        .rename(
            columns={
                "state_x": "switch_x",
                "state_y": "switch_y",
                "_z_coordinate": "switch_z",
            }
        )
    )
    counts = (
        ordered.groupby(group_columns, as_index=False)
        .size()
        .rename(columns={"size": "n_observed_roots"})
    )
    endpoints = starts.merge(
        switches, on=group_columns, validate="one_to_one"
    ).merge(counts, on=group_columns, validate="one_to_one")
    endpoints["delta_x"] = endpoints["switch_x"] - endpoints["start_x"]
    endpoints["delta_y"] = endpoints["switch_y"] - endpoints["start_y"]
    endpoints["delta_z"] = endpoints["switch_z"] - endpoints["start_z"]
    finite_columns = [
        "start_x",
        "start_y",
        "start_z",
        "switch_x",
        "switch_y",
        "switch_z",
    ]
    finite = endpoints[finite_columns].replace(
        [np.inf, -np.inf], np.nan
    ).notna().all(axis=1)
    endpoints["complete_3d"] = finite.astype(int)
    valid = endpoints.loc[finite].copy()
    if valid.empty:
        return endpoints, pd.DataFrame(), {
            "n_bouts_total": int(len(endpoints)),
            "n_bouts_3d": 0,
            "coverage": 0.0,
            "n_runs": 0,
        }
    run_endpoints = (
        valid.groupby("run_label", as_index=False)
        .agg(
            start_x=("start_x", "mean"),
            start_y=("start_y", "mean"),
            start_z=("start_z", "mean"),
            switch_x=("switch_x", "mean"),
            switch_y=("switch_y", "mean"),
            switch_z=("switch_z", "mean"),
            n_bouts=("action_bout_id", "nunique"),
        )
    )
    for dimension in "xyz":
        run_endpoints[f"delta_{dimension}"] = (
            run_endpoints[f"switch_{dimension}"]
            - run_endpoints[f"start_{dimension}"]
        )
    summary: Dict[str, float] = {
        f"{endpoint}_{dimension}": float(
            run_endpoints[f"{endpoint}_{dimension}"].mean()
        )
        for endpoint in ("start", "switch")
        for dimension in "xyz"
    }
    summary.update(
        {
            f"delta_{dimension}": float(
                run_endpoints[f"delta_{dimension}"].mean()
            )
            for dimension in "xyz"
        }
    )
    summary.update(
        {
            "n_bouts_total": int(len(endpoints)),
            "n_bouts_3d": int(len(valid)),
            "coverage": float(len(valid) / max(len(endpoints), 1)),
            "n_runs": int(run_endpoints["run_label"].nunique()),
        }
    )
    return endpoints, run_endpoints, summary


def _pair_endpoint_densities(
    endpoints: pd.DataFrame,
    horizontal: str,
    vertical: str,
    horizontal_range: Tuple[float, float],
    vertical_range: Tuple[float, float],
    bins: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate run-equalized start/transition endpoint mass on one plane."""
    columns = [
        "run_label",
        "complete_3d",
        f"start_{horizontal}",
        f"start_{vertical}",
        f"switch_{horizontal}",
        f"switch_{vertical}",
    ]
    valid = endpoints[columns].replace(
        [np.inf, -np.inf], np.nan
    ).dropna().copy()
    valid = valid.loc[valid["complete_3d"].eq(1)].copy()
    empty = np.zeros((bins, bins), dtype=float)
    if valid.empty:
        return empty.copy(), empty
    per_run = valid.groupby("run_label")["run_label"].transform("size")
    weights = 1.0 / per_run.to_numpy(dtype=float)
    weights /= np.sum(weights)
    arguments = {
        "bins": bins,
        "range": (vertical_range, horizontal_range),
        "weights": weights,
    }
    start, _, _ = np.histogram2d(
        valid[f"start_{vertical}"].to_numpy(dtype=float),
        valid[f"start_{horizontal}"].to_numpy(dtype=float),
        **arguments,
    )
    switch, _, _ = np.histogram2d(
        valid[f"switch_{vertical}"].to_numpy(dtype=float),
        valid[f"switch_{horizontal}"].to_numpy(dtype=float),
        **arguments,
    )
    start = gaussian_filter(start, sigma=1.0, mode="constant")
    switch = gaussian_filter(switch, sigma=1.0, mode="constant")
    if np.sum(start) > 0:
        start /= np.sum(start)
    if np.sum(switch) > 0:
        switch /= np.sum(switch)
    return start, switch


def _density_mass_thresholds(
    density: np.ndarray,
    masses: Sequence[float] = (0.80, 0.50),
) -> np.ndarray:
    positive = np.asarray(density, dtype=float)
    positive = positive[np.isfinite(positive) & (positive > 0)]
    if positive.size == 0:
        return np.asarray([], dtype=float)
    descending = np.sort(positive)[::-1]
    cumulative = np.cumsum(descending) / np.sum(descending)
    thresholds = []
    for mass in masses:
        index = min(
            int(np.searchsorted(cumulative, float(mass), side="left")),
            len(descending) - 1,
        )
        thresholds.append(float(descending[index]))
    return np.unique(np.sort(np.asarray(thresholds, dtype=float)))


def _draw_pair_mass_contours(
    ax: plt.Axes,
    density: np.ndarray,
    horizontal_range: Tuple[float, float],
    vertical_range: Tuple[float, float],
    color: str,
    linestyle: str,
) -> None:
    levels = _density_mass_thresholds(density)
    if levels.size == 0:
        return
    n_vertical, n_horizontal = density.shape
    horizontal_centers = np.linspace(
        horizontal_range[0], horizontal_range[1], n_horizontal,
        endpoint=False,
    ) + (horizontal_range[1] - horizontal_range[0]) / (2 * n_horizontal)
    vertical_centers = np.linspace(
        vertical_range[0], vertical_range[1], n_vertical,
        endpoint=False,
    ) + (vertical_range[1] - vertical_range[0]) / (2 * n_vertical)
    ax.contour(
        horizontal_centers,
        vertical_centers,
        density,
        levels=levels,
        colors=[color],
        linestyles=linestyle,
        linewidths=np.linspace(0.95, 1.35, len(levels)),
        alpha=0.95,
        zorder=4,
    )


def _plot_pairwise_probability_mass_change(
    transition_class: str,
    transition_title: str,
    action_role: str,
    panel_records: Sequence[Mapping[str, object]],
    pair: Tuple[str, str],
    axis_ranges: Mapping[str, Tuple[float, float]],
    axis_labels: Mapping[str, str],
    output_path: Path,
) -> None:
    """Plot transition-minus-start mass and mean movement on one 2D plane."""
    horizontal, vertical = pair
    horizontal_range = axis_ranges[horizontal]
    vertical_range = axis_ranges[vertical]
    prepared: List[Dict[str, object]] = []
    for record in panel_records:
        endpoints = record["endpoints"]
        start, switch = _pair_endpoint_densities(
            endpoints,
            horizontal,
            vertical,
            horizontal_range,
            vertical_range,
        )
        prepared.append(
            {
                **dict(record),
                "start_density": start,
                "switch_density": switch,
                "difference": switch - start,
            }
        )
    if not prepared:
        return
    supported = [
        record for record in prepared
        if int(record["summary"].get("n_bouts_3d", 0)) >= 8
        and int(record["summary"].get("n_runs", 0)) >= 3
    ]
    maximum = (
        max(
            float(np.nanmax(np.abs(record["difference"])))
            for record in supported
        )
        if supported else EPS
    )
    maximum = max(100.0 * maximum, EPS)
    fig, axes = plt.subplots(
        1,
        len(prepared),
        figsize=(max(7.4, 4.15 * len(prepared) + 0.8), 5.25),
        facecolor="white",
        squeeze=False,
    )
    image = None
    for column, record in enumerate(prepared):
        ax = axes[0, column]
        summary = record["summary"]
        run_endpoints = record["run_endpoints"]
        is_supported = (
            int(summary.get("n_bouts_3d", 0)) >= 8
            and int(summary.get("n_runs", 0)) >= 3
        )
        if not is_supported:
            ax.set_facecolor("#F5F6F7")
            ax.text(
                0.5, 0.5,
                "Insufficient joint\nbranch coverage",
                transform=ax.transAxes,
                ha="center", va="center",
                fontsize=9.0, color="#5B6470",
            )
            ax.set_xlim(*horizontal_range)
            ax.set_ylim(*vertical_range)
            ax.set_xlabel(axis_labels[horizontal], fontsize=8.5)
            if column == 0:
                ax.set_ylabel(axis_labels[vertical], fontsize=8.5)
            else:
                ax.tick_params(labelleft=False)
            action = int(record["action"])
            ax.set_title(
                (
                    f"{action_role} = {TRACE.ACTION_LABELS.get(action, str(action))}\n"
                    f"N={int(summary.get('n_bouts_3d', 0)):,}; "
                    f"runs={int(summary.get('n_runs', 0))}"
                ),
                fontsize=9.0,
            )
            continue
        image = ax.imshow(
            100.0 * np.asarray(record["difference"]),
            origin="lower",
            extent=(
                horizontal_range[0], horizontal_range[1],
                vertical_range[0], vertical_range[1],
            ),
            aspect="auto",
            cmap="PuOr_r",
            vmin=-maximum,
            vmax=maximum,
            interpolation="nearest",
        )
        _draw_pair_mass_contours(
            ax,
            np.asarray(record["start_density"]),
            horizontal_range,
            vertical_range,
            "#2F6690",
            "solid",
        )
        _draw_pair_mass_contours(
            ax,
            np.asarray(record["switch_density"]),
            horizontal_range,
            vertical_range,
            "#D17C00",
            "dashed",
        )
        for _, row in run_endpoints.iterrows():
            ax.plot(
                [row[f"start_{horizontal}"], row[f"switch_{horizontal}"]],
                [row[f"start_{vertical}"], row[f"switch_{vertical}"]],
                color="#6B7280",
                lw=0.7,
                alpha=0.34,
                zorder=5,
            )
        start = (
            float(summary[f"start_{horizontal}"]),
            float(summary[f"start_{vertical}"]),
        )
        switch = (
            float(summary[f"switch_{horizontal}"]),
            float(summary[f"switch_{vertical}"]),
        )
        ax.annotate(
            "",
            xy=switch,
            xytext=start,
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#202020",
                "lw": 2.2,
                "mutation_scale": 12,
            },
            zorder=7,
        )
        ax.scatter(
            start[0], start[1], marker="o", s=58,
            facecolor="white", edgecolor="#2F6690",
            linewidth=1.5, zorder=8,
        )
        ax.scatter(
            switch[0], switch[1], marker="*", s=105,
            facecolor="#D17C00", edgecolor="white",
            linewidth=0.65, zorder=8,
        )
        if horizontal in ("x", "y"):
            ax.axvline(0.5, color="#404040", ls="--", lw=0.65, alpha=0.55)
        if vertical in ("x", "y"):
            ax.axhline(0.5, color="#404040", ls="--", lw=0.65, alpha=0.55)
        ax.set_xlim(*horizontal_range)
        ax.set_ylim(*vertical_range)
        ax.set_xlabel(axis_labels[horizontal], fontsize=8.5)
        if column == 0:
            ax.set_ylabel(axis_labels[vertical], fontsize=8.5)
        else:
            ax.tick_params(labelleft=False)
        ax.tick_params(labelsize=7.3)
        action = int(record["action"])
        ax.set_title(
            (
                f"{action_role} = {TRACE.ACTION_LABELS.get(action, str(action))}\n"
                f"N={int(summary.get('n_bouts_3d', 0)):,}; "
                f"runs={int(summary.get('n_runs', 0))}"
            ),
            fontsize=9.0,
        )
    if image is not None:
        colorbar_axis = fig.add_axes([0.935, 0.16, 0.012, 0.60])
        colorbar = fig.colorbar(image, cax=colorbar_axis)
        colorbar.set_label(
            "Transition − start probability mass per 50×50 bin (%)",
            fontsize=8,
        )
        colorbar.ax.tick_params(labelsize=7)
    handles = [
        Line2D([0], [0], color="#2F6690", lw=1.4,
               label="Start 50% / 80% mass contours"),
        Line2D([0], [0], color="#D17C00", lw=1.4, ls="--",
               label="Transition 50% / 80% mass contours"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=6,
               markerfacecolor="white", markeredgecolor="#2F6690",
               label="Run-equalized start"),
        Line2D([0], [0], marker="*", linestyle="none", markersize=8,
               markerfacecolor="#D17C00", markeredgecolor="#D17C00",
               label="Run-equalized transition"),
    ]
    fig.suptitle(
        f"{transition_title}: Probability-Mass Change in {axis_labels[horizontal]}–{axis_labels[vertical]} Space",
        fontsize=14,
        x=0.5,
        y=0.985,
        ha="center",
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.915),
        ncol=4,
        frameon=False,
        fontsize=8.0,
    )
    fig.subplots_adjust(
        left=0.06, right=0.91, bottom=0.12, top=0.78, wspace=0.18,
    )
    save_figure(fig, output_path)


def plot_transition_value_geometry(
    metrics: pd.DataFrame,
    main_dir: Path,
    inference_dir: Path,
    *,
    z_mode: str,
) -> Dict[str, object]:
    """Draw transition trajectories in current-action margin--U--value space."""
    if z_mode not in {"net_revision", "destination_return"}:
        raise ValueError(f"Unknown z_mode: {z_mode}")
    primary_revision = z_mode == "net_revision"
    _, risk, risk_qa = TRACE.build_same_action_transition_risk(metrics)
    keys = ["run_label", "real_step", "real_step_id"]
    if primary_revision:
        z_column = "net_value_revision_toward_alternatives"
        z_values = metrics[keys + [z_column]].copy()
        risk = risk.drop(columns=[z_column], errors="ignore").merge(
            z_values, on=keys, how="left", validate="one_to_one"
        )
        output_root = inference_dir / "transition_value_revision_geometry"
        z_axis_label = "Directional Value-Revision Balance"
        z_title = "Value-Revision Balance"
        z_file_term = "value_revision"
        support_message = "Insufficient value-revision\ncoverage"
        manifest_z = (
            "sum of signed sampled-branch root-Q revisions divided by their "
            "total absolute magnitude plus EPS; alternative branch updates "
            "enter positively and current-action branch updates negatively; "
            "bounded to [-1, 1] without run-distribution scaling"
        )
        interpretation = (
            "positive values indicate that search revised root-Q evidence "
            "toward alternatives; negative values indicate revision toward "
            "maintaining the current action"
        )
        future_action_conditioning = False
    else:
        z_column = None
        branch_columns = [
            f"branch_terminal_return_mean_{action}" for action in range(6)
        ]
        branch_values = metrics[keys + branch_columns].copy()
        risk = risk.drop(columns=branch_columns, errors="ignore").merge(
            branch_values, on=keys, how="left", validate="one_to_one"
        )
        output_root = inference_dir / "retrospective_destination_value_geometry"
        z_axis_label = "Destination − Incumbent Return"
        z_title = "Destination-Value"
        z_file_term = "destination_value"
        support_message = "Insufficient joint\nbranch coverage"
        manifest_z = (
            "mean terminal rollout return for the eventual destination-action "
            "root branch minus the corresponding mean for the incumbent-action "
            "root branch"
        )
        interpretation = (
            "retrospective destination-aligned sensitivity analysis; not used "
            "as a prospective transition predictor"
        )
        future_action_conditioning = True

    file_tags = {
        "noop_to_active": "1a_noop_to_active",
        "active_to_noop": "1b_active_to_noop",
        "active_to_active": "1c_active_to_active",
    }
    transition_titles = {
        "noop_to_active": "Delay-to-Active",
        "active_to_noop": "Active-to-Delay",
        "active_to_active": "Active-to-Active Switch",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, object] = {
        "mode": z_mode,
        "x": "search-start normalized current-action margin",
        "y": "search-start alternative-action uncertainty",
        "z": manifest_z,
        "future_action_conditioning": future_action_conditioning,
        "interpretation": interpretation,
        "depth_adjustment": False,
        "endpoint_definition": (
            "first comparable search window to transition-producing search "
            "window within completed known-onset same-action bouts"
        ),
        "aggregation": "bout endpoints averaged within run, then runs equally weighted",
        "risk_qa": risk_qa,
        "transitions": {},
    }
    legend_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=6,
            markerfacecolor="white", markeredgecolor="#2F6690",
            label="First comparable search",
        ),
        Line2D(
            [0], [0], marker="*", linestyle="none", markersize=9,
            markerfacecolor="#D17C00", markeredgecolor="#D17C00",
            label="Transition-producing search",
        ),
        Line2D(
            [0], [0], color="#202020", lw=2.2,
            label="Run-equalized mean movement",
        ),
    ]

    for transition_class in (
        "noop_to_active",
        "active_to_noop",
        "active_to_active",
    ):
        selected = TRACE._complete_known_transition_rows(
            risk, transition_class
        )
        if selected.empty:
            continue
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
        if not actions:
            continue

        panel_records: List[Dict[str, object]] = []
        endpoint_tables: List[pd.DataFrame] = []
        run_tables: List[pd.DataFrame] = []
        summaries: List[Dict[str, object]] = []
        for action in actions:
            action_rows = selected.loc[selected[action_column].eq(action)]
            endpoints, run_endpoints, summary = _transition_3d_endpoint_tables(
                action_rows,
                z_column=z_column,
            )
            endpoints["facet_action"] = action
            endpoints["facet_action_role"] = action_role
            endpoint_tables.append(endpoints)
            if not run_endpoints.empty:
                run_endpoints["facet_action"] = action
                run_endpoints["facet_action_role"] = action_role
                run_tables.append(run_endpoints)
            summary_record = {
                "transition_class": transition_class,
                "facet_action": action,
                "facet_action_label": TRACE.ACTION_LABELS.get(
                    action, str(action)
                ),
                "facet_action_role": action_role,
                **summary,
            }
            summaries.append(summary_record)
            panel_records.append(
                {
                    "action": action,
                    "endpoints": endpoints,
                    "run_endpoints": run_endpoints,
                    "summary": summary,
                }
            )

        # Keep every action facet so sparse action classes remain visible, but
        # estimate movements only when all three coordinates have sufficient
        # bout- and run-level support.
        supported_panel_records = [
            record
            for record in panel_records
            if int(record["summary"].get("n_bouts_3d", 0)) >= 8
            and int(record["summary"].get("n_runs", 0)) >= 3
        ]
        displayed_panel_records = panel_records

        endpoint_table = pd.concat(endpoint_tables, ignore_index=True)
        run_table = (
            pd.concat(run_tables, ignore_index=True)
            if run_tables
            else pd.DataFrame()
        )
        summary_table = pd.DataFrame(summaries)
        class_output = output_root / transition_class
        class_output.mkdir(parents=True, exist_ok=True)
        endpoint_table.to_csv(
            class_output / "bout_endpoints.csv.gz",
            index=False,
            compression="gzip",
        )
        run_table.to_csv(class_output / "run_endpoints.csv", index=False)
        summary_table.to_csv(class_output / "summary.csv", index=False)

        z_values = []
        run_z_values = []
        for record in displayed_panel_records:
            endpoints = record["endpoints"]
            if endpoints.empty:
                continue
            z_values.extend(endpoints["start_z"].tolist())
            z_values.extend(endpoints["switch_z"].tolist())
            run_endpoints = record["run_endpoints"]
            if not run_endpoints.empty:
                run_z_values.extend(run_endpoints["start_z"].tolist())
                run_z_values.extend(run_endpoints["switch_z"].tolist())
        finite_z = np.asarray(z_values, dtype=float)
        finite_z = finite_z[np.isfinite(finite_z)]
        if finite_z.size:
            density_z_low = float(np.min(finite_z))
            density_z_high = float(np.max(finite_z))
            density_z_padding = max(
                (density_z_high - density_z_low) * 0.04, 1e-4
            )
        else:
            density_z_low, density_z_high, density_z_padding = -1.0, 1.0, 0.1
        finite_run_z = np.asarray(run_z_values, dtype=float)
        finite_run_z = finite_run_z[np.isfinite(finite_run_z)]
        if finite_run_z.size:
            plot_z_low = float(np.min(finite_run_z))
            plot_z_high = float(np.max(finite_run_z))
            plot_z_padding = max(
                (plot_z_high - plot_z_low) * 0.12, 1e-4
            )
        else:
            plot_z_low, plot_z_high, plot_z_padding = -1.0, 1.0, 0.1

        fig = plt.figure(
            figsize=(
                max(7.0, 5.1 * max(len(displayed_panel_records), 1)),
                6.4,
            ),
            facecolor="white",
        )
        for panel_index, record in enumerate(displayed_panel_records, start=1):
            ax = fig.add_subplot(
                1,
                max(len(displayed_panel_records), 1),
                panel_index,
                projection="3d",
            )
            run_endpoints = record["run_endpoints"]
            summary = record["summary"]
            action = int(record["action"])
            is_supported = (
                int(summary.get("n_bouts_3d", 0)) >= 8
                and int(summary.get("n_runs", 0)) >= 3
            )
            if not is_supported:
                ax.text2D(
                    0.5, 0.5, support_message,
                    transform=ax.transAxes, ha="center", va="center",
                )
            else:
                for _, row in run_endpoints.iterrows():
                    ax.plot(
                        [row["start_x"], row["switch_x"]],
                        [row["start_y"], row["switch_y"]],
                        [row["start_z"], row["switch_z"]],
                        color="#9AA4AE", lw=0.9, alpha=0.45,
                    )
                    ax.scatter(
                        row["start_x"], row["start_y"], row["start_z"],
                        marker="o", s=22, facecolor="white",
                        edgecolor="#2F6690", linewidth=0.7, alpha=0.75,
                    )
                    ax.scatter(
                        row["switch_x"], row["switch_y"], row["switch_z"],
                        marker="*", s=38, color="#D17C00", alpha=0.72,
                    )
                ax.quiver(
                    summary["start_x"], summary["start_y"], summary["start_z"],
                    summary["delta_x"], summary["delta_y"], summary["delta_z"],
                    color="#202020", linewidth=2.4, arrow_length_ratio=0.14,
                    normalize=False,
                )
                ax.scatter(
                    summary["start_x"], summary["start_y"], summary["start_z"],
                    marker="o", s=72, facecolor="white",
                    edgecolor="#2F6690", linewidth=1.8, depthshade=False,
                )
                ax.scatter(
                    summary["switch_x"], summary["switch_y"], summary["switch_z"],
                    marker="*", s=145, color="#D17C00",
                    edgecolor="white", linewidth=0.7, depthshade=False,
                )
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.set_zlim(
                plot_z_low - plot_z_padding,
                plot_z_high + plot_z_padding,
            )
            ax.set_xlabel("Current-action margin", labelpad=7)
            ax.set_ylabel("Alternative uncertainty", labelpad=7)
            ax.set_zlabel(z_axis_label, labelpad=7)
            ax.view_init(elev=24, azim=-58)
            ax.set_box_aspect((1.0, 1.0, 0.82))
            ax.grid(True)
            for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
                pane.set_facecolor((0.97, 0.98, 0.99, 1.0))
                pane.set_edgecolor((0.84, 0.86, 0.88, 1.0))
            ax.set_title(
                (
                    f"{action_role} = "
                    f"{TRACE.ACTION_LABELS.get(action, str(action))}\n"
                    f"N={int(summary.get('n_bouts_3d', 0)):,}/"
                    f"{int(summary.get('n_bouts_total', 0)):,} bouts; "
                    f"runs={int(summary.get('n_runs', 0))}\n"
                    f"Δ=({summary.get('delta_x', np.nan):+.3f}, "
                    f"{summary.get('delta_y', np.nan):+.3f}, "
                    f"{summary.get('delta_z', np.nan):+.3f})"
                ),
                fontsize=9.3,
                pad=12,
            )
            ax.tick_params(labelsize=7.2, pad=1)

        fig.suptitle(
            (
                f"{transition_titles[transition_class]} Search Movement in "
                f"Margin–Uncertainty–{z_title} Space"
            ),
            fontsize=15,
            x=0.5,
            y=0.985,
            ha="center",
        )
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.915),
            ncol=3,
            frameon=False,
            fontsize=8.7,
        )
        fig.subplots_adjust(
            left=0.025, right=0.985, bottom=0.05, top=0.76,
            wspace=0.08,
        )
        output_path = main_dir / (
            "main_figure"
            f"{file_tags[transition_class]}_actor_policy_start_geometry.png"
        )
        save_figure(fig, output_path)

        axis_ranges = {
            "x": (0.0, 1.0),
            "y": (0.0, 1.0),
            "z": (
                density_z_low - density_z_padding,
                density_z_high + density_z_padding,
            ),
        }
        axis_labels = {
            "x": "Current-Action Margin",
            "y": "Alternative Uncertainty",
            "z": z_axis_label,
        }
        pair_output_names = {
            ("x", "y"): (
                "main_figure"
                f"{file_tags[transition_class]}_actor_policy_start_density_difference.png"
            ),
            ("x", "z"): (
                "main_figure"
                f"{file_tags[transition_class]}_margin_{z_file_term}_density_difference.png"
            ),
            ("y", "z"): (
                "main_figure"
                f"{file_tags[transition_class]}_uncertainty_{z_file_term}_density_difference.png"
            ),
        }
        pair_figures: Dict[str, str] = {}
        for pair, output_name in pair_output_names.items():
            pair_path = main_dir / output_name
            _plot_pairwise_probability_mass_change(
                transition_class,
                transition_titles[transition_class],
                action_role,
                displayed_panel_records,
                pair,
                axis_ranges,
                axis_labels,
                pair_path,
            )
            pair_figures["-".join(pair)] = str(pair_path)
        manifest["transitions"][transition_class] = {
            "figure_3d": str(output_path),
            "figures_2d_probability_mass_change": pair_figures,
            "available_actions": actions,
            "displayed_actions": [
                int(record["action"])
                for record in displayed_panel_records
            ],
            "supported_actions": [
                int(record["action"])
                for record in supported_panel_records
            ],
            "minimum_estimation_support": (
                "8 complete 3D bouts in at least 3 runs; unsupported facets "
                "are retained as explicit blank panels"
            ),
            "n_bouts_total": int(summary_table["n_bouts_total"].sum()),
            "n_bouts_3d": int(summary_table["n_bouts_3d"].sum()),
            "coverage": float(
                summary_table["n_bouts_3d"].sum()
                / max(summary_table["n_bouts_total"].sum(), 1)
            ),
        }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    return manifest


def _prospective_value_specifications() -> Dict[str, Dict[str, str]]:
    return {
        "noop_to_active": {
            "label": "NOOP → Active",
            "value_column": "best_active_value_advantage_over_noop",
            "definition": "best observed active branch minus NOOP branch",
        },
        "active_to_noop": {
            "label": "Active → NOOP",
            "value_column": "noop_value_advantage_over_incumbent",
            "definition": "NOOP branch minus incumbent active branch",
        },
        "active_to_active": {
            "label": "Active → Active",
            "value_column": "best_other_active_value_advantage_over_incumbent",
            "definition": "best observed other-active branch minus incumbent active branch",
        },
    }


def _build_prospective_transition_risk(
    metrics: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """Build future-blind competing-risk rows for the three transition classes."""
    bouts, risk, risk_qa = TRACE.build_same_action_transition_risk(metrics)
    keys = ["run_label", "real_step", "real_step_id"]
    value_columns = [
        "incumbent_branch_return",
        "observed_action_branch_return",
        "observed_action_value_advantage",
        "best_alternative_action",
        "best_alternative_branch_return",
        "best_alternative_value_advantage",
        "n_observed_alternative_branches",
        "best_active_value_advantage_over_noop",
        "noop_value_advantage_over_incumbent",
        "best_other_active_value_advantage_over_incumbent",
        "best_alternative_matches_observed_switch",
    ]
    values = metrics[keys + value_columns].copy()
    risk = risk.drop(columns=value_columns, errors="ignore").merge(
        values,
        on=keys,
        how="left",
        validate="one_to_one",
    )
    risk["observed_duration"] = pd.to_numeric(
        risk["root_position"], errors="coerce"
    ) + 1.0
    risk["log_observed_duration"] = np.log1p(
        np.clip(risk["observed_duration"].to_numpy(dtype=float), 0, None)
    )
    minimum = risk.groupby("run_label")["real_step"].transform("min")
    maximum = risk.groupby("run_label")["real_step"].transform("max")
    denominator = (maximum - minimum).replace(0, 1)
    risk["run_progress"] = (risk["real_step"] - minimum) / denominator

    specifications = _prospective_value_specifications()
    class_rows: List[pd.DataFrame] = []
    previous = pd.to_numeric(risk["previous_human_action"], errors="coerce")
    current = pd.to_numeric(risk["human_action"], errors="coerce")
    for transition_class, specification in specifications.items():
        if transition_class == "noop_to_active":
            eligible = previous.eq(0) & current.between(0, 5)
            event = current.ne(0)
        elif transition_class == "active_to_noop":
            eligible = previous.between(1, 5) & (
                current.eq(previous) | current.eq(0)
            )
            event = current.eq(0)
        else:
            eligible = previous.between(1, 5) & current.between(1, 5)
            event = current.ne(previous)
        selected = risk.loc[eligible].copy()
        selected["prospective_transition_class"] = transition_class
        selected["prospective_transition"] = event.loc[selected.index].astype(int)
        selected["prospective_value_advantage"] = pd.to_numeric(
            selected[specification["value_column"]], errors="coerce"
        )
        selected["prospective_value_definition"] = specification["definition"]
        class_rows.append(selected)
    combined = pd.concat(class_rows, ignore_index=True)
    qa = {
        "source_risk_qa": risk_qa,
        "row_definition": (
            "discrete-time same-action transition risk; competing terminal "
            "transition rows are excluded from the other active transition class"
        ),
        "future_action_conditioning": False,
        "missing_branch_policy": "complete-case; unobserved branches are never zero-filled",
    }
    return combined, risk, qa


def _fit_prospective_transition_models(
    prospective: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    effect_records: List[Dict[str, object]] = []
    curve_records: List[Dict[str, object]] = []
    coverage_records: List[Dict[str, object]] = []
    grid = np.linspace(-2.5, 2.5, 101)
    for transition_class, class_data in prospective.groupby(
        "prospective_transition_class", sort=False
    ):
        for run_label, run in class_data.groupby("run_label", sort=False):
            required = [
                "prospective_transition",
                "prospective_value_advantage",
                "state_x",
                "state_y",
                "log_observed_duration",
                "previous_reward",
                "previous_done",
                "run_progress",
                "previous_human_action",
            ]
            all_rows = run[required].replace([np.inf, -np.inf], np.nan)
            valid = all_rows.dropna().copy()
            n_total = int(len(run))
            n_valid = int(len(valid))
            n_events_total = int(run["prospective_transition"].sum())
            n_events_valid = int(valid["prospective_transition"].sum())
            coverage_records.append(
                {
                    "transition_class": transition_class,
                    "run_label": str(run_label),
                    "n_risk_rows": n_total,
                    "n_complete_value_rows": n_valid,
                    "coverage": n_valid / max(n_total, 1),
                    "n_transition_events": n_events_total,
                    "n_complete_value_events": n_events_valid,
                    "event_coverage": n_events_valid / max(n_events_total, 1),
                }
            )
            if (
                n_valid < 100
                or n_events_valid < 5
                or (n_valid - n_events_valid) < 5
                or float(np.nanstd(valid["prospective_value_advantage"])) <= EPS
            ):
                continue
            value_z = TRACE._zscore_finite(valid["prospective_value_advantage"])
            persistence_z = TRACE._zscore_finite(valid["state_x"])
            uncertainty_z = TRACE._zscore_finite(valid["state_y"])
            design = pd.DataFrame(
                {
                    "Value evidence": value_z,
                    "Persistence": persistence_z,
                    "Uncertainty": uncertainty_z,
                    "Persistence × Uncertainty": persistence_z * uncertainty_z,
                    "Duration": TRACE._zscore_finite(valid["log_observed_duration"]),
                    "Reward": TRACE._zscore_finite(valid["previous_reward"]),
                    "Done": TRACE._zscore_finite(valid["previous_done"]),
                    "Run progress": TRACE._zscore_finite(valid["run_progress"]),
                },
                index=valid.index,
            )
            action_dummies = pd.get_dummies(
                valid["previous_human_action"].astype(int),
                prefix="Incumbent",
                drop_first=True,
                dtype=float,
            )
            design = pd.concat([design, action_dummies], axis=1)
            design = design.loc[:, design.std(axis=0).fillna(0).gt(EPS)]
            design = sm.add_constant(design, has_constant="add")
            try:
                result = sm.GLM(
                    valid["prospective_transition"].astype(float),
                    design,
                    family=sm.families.Binomial(),
                ).fit(maxiter=100, disp=0)
            except Exception as error:
                log(
                    "Prospective value model skipped "
                    f"{transition_class}/{run_label}: {error}"
                )
                continue
            coefficient = float(result.params["Value evidence"])
            effect_records.append(
                {
                    "transition_class": transition_class,
                    "run_label": str(run_label),
                    "n": n_valid,
                    "n_events": n_events_valid,
                    "coefficient": coefficient,
                }
            )
            for grid_value in grid:
                prediction_design = design.copy()
                prediction_design["Value evidence"] = float(grid_value)
                probability = np.asarray(
                    result.predict(prediction_design), dtype=float
                )
                curve_records.append(
                    {
                        "transition_class": transition_class,
                        "run_label": str(run_label),
                        "value_advantage_z": float(grid_value),
                        "predicted_probability": float(np.mean(probability)),
                    }
                )
    effects = pd.DataFrame(effect_records)
    summary = (
        TRACE._summarize_run_effects(effects, ["transition_class"])
        if not effects.empty
        else pd.DataFrame()
    )
    return effects, summary, pd.DataFrame(curve_records), pd.DataFrame(coverage_records)


def _destination_match_tables(
    risk: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    events = risk.loc[
        risk["is_transition"].eq(1)
        & risk["transition_class"].isin(
            ["noop_to_active", "active_to_noop", "active_to_active"]
        )
        & pd.to_numeric(
            risk["n_observed_alternative_branches"], errors="coerce"
        ).ge(2)
        & pd.to_numeric(
            risk["observed_action_branch_return"], errors="coerce"
        ).notna()
        & pd.to_numeric(
            risk["incumbent_branch_return"], errors="coerce"
        ).notna()
    ].copy()
    events["chance_match"] = 1.0 / pd.to_numeric(
        events["n_observed_alternative_branches"], errors="coerce"
    )
    events["observed_match"] = pd.to_numeric(
        events["best_alternative_matches_observed_switch"], errors="coerce"
    )
    run_match = (
        events.groupby(["transition_class", "run_label"], as_index=False)
        .agg(
            n_transitions=("real_step_id", "size"),
            observed_match=("observed_match", "mean"),
            chance_match=("chance_match", "mean"),
            mean_destination_advantage=("observed_action_value_advantage", "mean"),
        )
    )
    run_match["excess_match"] = (
        run_match["observed_match"] - run_match["chance_match"]
    )
    effects = run_match.rename(columns={"excess_match": "coefficient"})
    summary = (
        TRACE._summarize_run_effects(effects, ["transition_class"])
        if not effects.empty
        else pd.DataFrame()
    )
    return events, run_match, summary


def _plot_prospective_value_curves(
    curves: pd.DataFrame,
    summary: pd.DataFrame,
    coverage: pd.DataFrame,
    output_path: Path,
) -> None:
    specifications = _prospective_value_specifications()
    if "transition_class" not in curves.columns:
        curves = pd.DataFrame(columns=["transition_class"])
    if "transition_class" not in summary.columns:
        summary = pd.DataFrame(columns=["transition_class"])
    if "transition_class" not in coverage.columns:
        coverage = pd.DataFrame(columns=["transition_class"])
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.9), facecolor="white")
    for ax, (transition_class, specification) in zip(axes, specifications.items()):
        selected = curves.loc[curves["transition_class"].eq(transition_class)]
        n_runs = int(selected["run_label"].nunique()) if not selected.empty else 0
        if selected.empty or n_runs < 3:
            ax.text(
                0.5,
                0.5,
                f"Insufficient complete branch coverage\n({n_runs} estimable runs)",
                ha="center",
                va="center",
            )
            ax.set_axis_off()
            continue
        for _, run in selected.groupby("run_label", sort=False):
            ax.plot(
                run["value_advantage_z"], run["predicted_probability"],
                color="#9BBEDB", lw=0.8, alpha=0.42,
            )
        group = selected.groupby("value_advantage_z")["predicted_probability"]
        mean = group.mean()
        count = group.count()
        sem = group.std(ddof=1) / np.sqrt(count)
        critical = stats.t.ppf(0.975, np.maximum(count - 1, 1))
        low = np.clip(mean - critical * sem, 0, 1)
        high = np.clip(mean + critical * sem, 0, 1)
        x = mean.index.to_numpy(dtype=float)
        ax.fill_between(x, low.to_numpy(), high.to_numpy(), color="#2F6690", alpha=0.16)
        ax.plot(x, mean.to_numpy(), color="#2F6690", lw=2.3)
        row = summary.loc[summary["transition_class"].eq(transition_class)]
        coverage_rows = coverage.loc[coverage["transition_class"].eq(transition_class)]
        coverage_value = (
            float(
                coverage_rows["n_complete_value_rows"].sum()
                / max(coverage_rows["n_risk_rows"].sum(), 1)
            )
            if not coverage_rows.empty else np.nan
        )
        annotation = f"runs={n_runs}; coverage={coverage_value:.1%}"
        if not row.empty:
            annotation = (
                f"b={float(row['mean_coefficient'].iloc[0]):+.3f}; "
                f"q={float(row['bh_q'].iloc[0]):.3g}; " + annotation
            )
        ax.text(0.03, 0.97, annotation, transform=ax.transAxes, ha="left", va="top", fontsize=8.4)
        ax.axvline(0, color="#50555A", ls="--", lw=0.8)
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(bottom=0)
        ax.set_title(specification["label"], fontsize=11)
        ax.set_xlabel("Prospective value advantage (within-run SD)")
        ax.grid(color="#E5E7EB", lw=0.55)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("Adjusted transition probability")
    fig.suptitle("Prospective Branch-Value Evidence and Human Action Transitions", fontsize=14.5, y=0.985)
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.16, top=0.84, wspace=0.25)
    save_figure(fig, output_path)


def _plot_destination_match(
    run_match: pd.DataFrame,
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    specifications = _prospective_value_specifications()
    if "transition_class" not in run_match.columns:
        run_match = pd.DataFrame(columns=["transition_class"])
    if "transition_class" not in summary.columns:
        summary = pd.DataFrame(columns=["transition_class"])
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.9), facecolor="white")
    for ax, (transition_class, specification) in zip(axes, specifications.items()):
        selected = run_match.loc[run_match["transition_class"].eq(transition_class)]
        n_runs = int(selected["run_label"].nunique()) if not selected.empty else 0
        if selected.empty or n_runs < 3:
            ax.text(
                0.5,
                0.5,
                f"Insufficient observed alternatives\n({n_runs} estimable runs)",
                ha="center",
                va="center",
            )
            ax.set_axis_off()
            continue
        for _, row in selected.iterrows():
            ax.plot(
                [0, 1], [row["chance_match"], row["observed_match"]],
                color="#A9AFB5", lw=0.8, alpha=0.55,
            )
            ax.scatter(0, row["chance_match"], s=18, facecolor="white", edgecolor="#6B7280", zorder=3)
            ax.scatter(1, row["observed_match"], s=20, color="#D17C00", zorder=3)
        chance_mean = float(selected["chance_match"].mean())
        observed_mean = float(selected["observed_match"].mean())
        ax.plot([0, 1], [chance_mean, observed_mean], color="#202020", lw=2.3, zorder=4)
        ax.scatter(0, chance_mean, s=68, facecolor="white", edgecolor="#202020", linewidth=1.5, zorder=5)
        ax.scatter(1, observed_mean, s=72, color="#D17C00", edgecolor="white", linewidth=0.6, zorder=5)
        row = summary.loc[summary["transition_class"].eq(transition_class)]
        annotation = (
            f"runs={n_runs}; "
            f"N={int(selected['n_transitions'].sum()):,} transitions"
        )
        if not row.empty:
            annotation = (
                f"excess={float(row['mean_coefficient'].iloc[0]):+.3f}; "
                f"q={float(row['bh_q'].iloc[0]):.3g}\n" + annotation
            )
        ax.text(0.03, 0.97, annotation, transform=ax.transAxes, ha="left", va="top", fontsize=8.4)
        ax.set_xlim(-0.35, 1.35)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 1], ["Coverage-matched\nchance", "Highest-return branch\nmatches destination"])
        ax.set_title(specification["label"], fontsize=11)
        ax.grid(axis="y", color="#E5E7EB", lw=0.55)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("Destination-match fraction")
    fig.suptitle("Does the Highest-Return Imagined Branch Match the Human Transition?", fontsize=14.5, y=0.985)
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.19, top=0.84, wspace=0.24)
    save_figure(fig, output_path)


def analyze_prospective_branch_value(
    metrics: pd.DataFrame,
    main_dir: Path,
    inference_dir: Path,
) -> Dict[str, object]:
    prospective, risk, qa = _build_prospective_transition_risk(metrics)
    effects, summary, curves, coverage = _fit_prospective_transition_models(prospective)
    events, run_match, match_summary = _destination_match_tables(risk)
    output_dir = inference_dir / "prospective_branch_value"
    output_dir.mkdir(parents=True, exist_ok=True)
    prospective.to_csv(output_dir / "transition_risk_rows.csv.gz", index=False, compression="gzip")
    effects.to_csv(output_dir / "run_effects.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    curves.to_csv(output_dir / "adjusted_probability_curves.csv", index=False)
    coverage.to_csv(output_dir / "coverage_by_run.csv", index=False)
    events.to_csv(output_dir / "destination_match_events.csv.gz", index=False, compression="gzip")
    run_match.to_csv(output_dir / "destination_match_by_run.csv", index=False)
    match_summary.to_csv(output_dir / "destination_match_summary.csv", index=False)
    _plot_prospective_value_curves(
        curves,
        summary,
        coverage,
        main_dir / "main_figure4_prospective_branch_value_transitions.png",
    )
    _plot_destination_match(
        run_match,
        match_summary,
        main_dir / "main_figure5_destination_branch_match.png",
    )
    manifest = {
        **qa,
        "prospective_value_definitions": _prospective_value_specifications(),
        "model": (
            "separate within-run binomial GLM per transition class; value "
            "advantage plus Persistence, Uncertainty, interaction, observed "
            "same-action duration, reward, done, run progress, and incumbent-action controls"
        ),
        "inference": "run coefficients -> exact sign-flip; BH-FDR across three transition classes",
        "destination_match": (
            "actual transition rows with incumbent, destination, and at least "
            "two observed alternative branches; observed match minus 1/n-observed-alternatives"
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    return manifest


def generate_canonical_persistence_figures(
    metrics: pd.DataFrame,
    figure_dir: Path,
    result_dir: Path,
) -> Dict[str, object]:
    """Generate the complete canonical non-tSNE 02 figure family."""
    configure_canonical_persistence_display()
    main_dir = figure_dir / "main"
    supplementary_dir = figure_dir / "supplementary"
    inference_dir = result_dir / "inference"
    main_dir.mkdir(parents=True, exist_ok=True)
    supplementary_dir.mkdir(parents=True, exist_ok=True)
    inference_dir.mkdir(parents=True, exist_ok=True)

    TRACE.plot_main_figure1_search_geometry(
        metrics,
        main_dir / "main_figure1_current_action_margin_uncertainty_search_geometry.png",
        inference_dir,
    )
    TRACE.plot_main_figure3_policy_state_search_geometry(
        metrics,
        main_dir
        / "main_figure3_current_action_margin_uncertainty_search_geometry_associations.png",
        inference_dir,
    )
    integrated = TRACE.generate_integrated_transition_geometry_analysis(
        metrics,
        main_dir,
        inference_dir,
    )
    # Replace the actor-policy wording in the shared manifest.  The numerical
    # analysis above is unchanged; only the coordinate interpretation differs.
    integrated["analysis"] = "02 current-action margin transition geometry"
    integrated["coordinate_spaces"] = {
        "actor_start": {
            "x": (
                "sigmoid of the actor-policy logit margin between the current "
                "action and its strongest alternative"
            ),
            "y": (
                "normalized entropy of the actor policy conditional on all "
                "actions other than the incumbent"
            ),
            "incumbent": "previous human action at the current real step",
        },
    }
    integrated["duration_regression"] = (
        "within-run OLS of standardized endpoint search change on log bout "
        "action count; controls start Current-Action Margin, Alternative "
        "Uncertainty, "
        "their interaction, origin/destination action, previous reward/done, "
        "and run progress"
    )
    integrated_manifest_path = (
        inference_dir
        / "integrated_transition_geometry"
        / "analysis_manifest.json"
    )
    integrated_manifest_path.write_text(
        json.dumps(integrated, indent=2), encoding="utf-8"
    )
    transition_qa = TRACE.generate_generalized_action_transition_figures(
        metrics,
        figure_dir,
        result_dir,
    )
    TRACE.plot_planning_state_space_by_action(
        metrics,
        action_column="human_action",
        action_source="Human",
        output_path=(
            supplementary_dir
            / "supplementary_figureS5_persistence_search_by_human_action.png"
        ),
        slack_axis_label="Normalized Start Current-Action Margin",
        demand_axis_label=(
            "Mean Alternative-Action Uncertainty during Tree Search"
        ),
        figure_title="Current-Action Margin--Uncertainty during Tree Search",
    )
    TRACE.plot_state_space_human_transitions(
        metrics,
        supplementary_dir
        / "supplementary_figureS6_persistence_search_transitions.png",
        slack_axis_label="Normalized Start Current-Action Margin",
        demand_axis_label=(
            "Mean Alternative-Action Uncertainty during Tree Search"
        ),
        figure_title=(
            "Human Action Transitions in Margin--Uncertainty Space"
        ),
    )
    TRACE.plot_planning_state_space_by_action(
        metrics,
        action_column="human_action",
        action_source="Human",
        output_path=(
            supplementary_dir
            / "supplementary_figureS7_persistence_start_by_human_action.png"
        ),
        demand_column="d_policy_root_start",
        slack_axis_label="Normalized Start Current-Action Margin",
        demand_axis_label="Start Alternative-Action Uncertainty",
        figure_title="Search-Start Margin--Uncertainty State",
    )
    TRACE.plot_state_space_human_transitions(
        metrics,
        supplementary_dir
        / "supplementary_figureS8_persistence_start_transitions.png",
        slack_axis_label="Normalized Start Current-Action Margin",
        demand_column="d_policy_root_start",
        demand_axis_label="Start Alternative-Action Uncertainty",
        figure_title=(
            "Human Action Transitions in Search-Start "
            "Margin--Uncertainty Space"
        ),
        coordinate_note=(
            "the incumbent is the previous human action at each real step"
        ),
    )
    return {
        "integrated_transition_geometry": integrated,
        "action_transition_qa": transition_qa,
    }


def plot_circle_star_3d(
    windows: pd.DataFrame,
    bouts: pd.DataFrame,
    output_path: Path,
    max_points: int,
) -> None:
    data = finite_bout_data(bouts)
    if data.empty:
        log("Skip circle-star figure: no finite bout endpoints")
        return
    if len(data) > max_points:
        data = data.sample(max_points, random_state=2026)
    x0 = data["root_persistence_x"].to_numpy(float)
    y0 = data["root_persistence_y"].to_numpy(float)
    x1 = data["terminal_persistence_x"].to_numpy(float)
    y1 = data["terminal_persistence_y"].to_numpy(float)
    z1 = data["return_gain_from_root"].to_numpy(float)
    z_low = float(np.nanquantile(z1, 0.02))
    z_high = float(np.nanquantile(z1, 0.98))
    if not z_high > z_low:
        pad = max(1e-6, abs(z_low) * 1e-3)
        z_low -= pad
        z_high += pad
    norm = Normalize(vmin=z_low, vmax=z_high)
    cmap = plt.get_cmap("coolwarm")
    colors = cmap(norm(z1))
    fig = plt.figure(figsize=(16.5, 12.0), facecolor="white")
    ax3 = fig.add_subplot(2, 2, 1, projection="3d")
    ax_xy = fig.add_subplot(2, 2, 2)
    ax_xz = fig.add_subplot(2, 2, 3)
    ax_yz = fig.add_subplot(2, 2, 4)
    for index in range(len(data)):
        line_color = (*colors[index, :3], 0.18)
        ax3.plot([x0[index], x1[index]], [y0[index], y1[index]], [0.0, z1[index]], color=line_color, lw=0.6)
        ax_xy.plot([x0[index], x1[index]], [y0[index], y1[index]], color=line_color, lw=0.6)
        ax_xz.plot([x0[index], x1[index]], [0.0, z1[index]], color=line_color, lw=0.6)
        ax_yz.plot([y0[index], y1[index]], [0.0, z1[index]], color=line_color, lw=0.6)
    ax3.scatter(x0, y0, np.zeros(len(data)), marker="o", facecolors="none", edgecolors="#20252B", s=24, alpha=0.40, label="Pre-search root")
    ax3.scatter(x1, y1, z1, marker="*", c=z1, cmap=cmap, norm=norm, s=46, alpha=0.80, edgecolors="none", label="Bout terminal")
    ax_xy.scatter(x0, y0, marker="o", facecolors="none", edgecolors="#20252B", s=20, alpha=0.35)
    ax_xy.scatter(x1, y1, marker="*", c=z1, cmap=cmap, norm=norm, s=42, alpha=0.80, edgecolors="none")
    ax_xz.scatter(x0, np.zeros(len(data)), marker="o", facecolors="none", edgecolors="#20252B", s=20, alpha=0.35)
    ax_xz.scatter(x1, z1, marker="*", c=z1, cmap=cmap, norm=norm, s=42, alpha=0.80, edgecolors="none")
    ax_yz.scatter(y0, np.zeros(len(data)), marker="o", facecolors="none", edgecolors="#20252B", s=20, alpha=0.35)
    ax_yz.scatter(y1, z1, marker="*", c=z1, cmap=cmap, norm=norm, s=42, alpha=0.80, edgecolors="none")
    ax3.set_xlabel("Root persistence evidence")
    ax3.set_ylabel("Root switch-target uncertainty")
    ax3.set_zlabel("Terminal rollout gain from root")
    ax3.set_title("Three-dimensional root → imagination-bout endpoint")
    ax3.legend(frameon=False, loc="best", fontsize=8)
    ax_xy.set_xlabel("Persistence evidence → sigmoid")
    ax_xy.set_ylabel("Switch-target uncertainty")
    ax_xy.set_title("Policy geometry projection")
    ax_xz.set_xlabel("Persistence evidence → sigmoid")
    ax_xz.set_ylabel("Terminal rollout gain from root")
    ax_xz.set_title("Persistence evidence × value gain")
    ax_yz.set_xlabel("Switch-target uncertainty")
    ax_yz.set_ylabel("Terminal rollout gain from root")
    ax_yz.set_title("Uncertainty × value gain")
    for axis in (ax_xy, ax_xz, ax_yz):
        axis.grid(True, color="#E5E7EB", lw=0.55)
        axis.set_axisbelow(True)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(z1)
    fig.colorbar(sm, ax=[ax_xy, ax_xz, ax_yz], shrink=0.72, pad=0.04, label="Terminal rollout gain from root")
    fig.suptitle("Pre-search persistence geometry and realized rollout value", fontsize=16, y=0.985)
    fig.subplots_adjust(left=0.05, right=0.94, bottom=0.07, top=0.92, hspace=0.25, wspace=0.22)
    save_figure(fig, output_path)


def binned_mean(
    x: np.ndarray, y: np.ndarray, values: np.ndarray, bins: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    grid = np.full((bins, bins), np.nan, dtype=float)
    counts = np.zeros((bins, bins), dtype=int)
    ix = np.clip((x * bins).astype(int), 0, bins - 1)
    iy = np.clip((y * bins).astype(int), 0, bins - 1)
    for i in range(bins):
        for j in range(bins):
            selected = (ix == i) & (iy == j) & np.isfinite(values)
            counts[j, i] = int(np.sum(selected))
            if counts[j, i]:
                grid[j, i] = float(np.mean(values[selected]))
    return grid, counts


def plot_value_geometry(
    windows: pd.DataFrame, output_path: Path
) -> None:
    data = windows.loc[
        windows["cross_chunk_boundary"].eq(0)
        & windows[["persistence_x", "persistence_y"]].notna().all(axis=1)
    ].copy()
    if data.empty:
        return
    x = data["persistence_x"].to_numpy(float)
    y = data["persistence_y"].to_numpy(float)
    specifications = [
        ("return_gain_from_root", "Terminal rollout gain from root", "coolwarm"),
        ("return_gain_within_bout", "Within-bout terminal − first return", "coolwarm"),
        ("incumbent_minus_alternative_return", "Incumbent − alternative terminal return", "coolwarm"),
        ("mean_bout_terminal_depth", "Mean bout terminal depth", "viridis"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 10.0), facecolor="white", sharex=True, sharey=True)
    for axis, (column, title, cmap) in zip(axes.ravel(), specifications):
        values = data[column].to_numpy(float)
        grid, counts = binned_mean(x, y, values)
        finite = grid[np.isfinite(grid)]
        if finite.size:
            limit = float(np.nanquantile(np.abs(finite), 0.98))
            vmin, vmax = (-limit, limit) if "coolwarm" in cmap else (float(np.nanquantile(finite, 0.02)), float(np.nanquantile(finite, 0.98)))
        else:
            vmin, vmax = 0.0, 1.0
        image = axis.imshow(grid, origin="lower", extent=(0, 1, 0, 1), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        axis.scatter(x, y, s=2, color="#20252B", alpha=0.08, edgecolors="none")
        axis.set_title(f"{title}\nN={int(np.isfinite(values).sum()):,}")
        axis.set_xlabel("Persistence evidence → sigmoid")
        axis.set_ylabel("Switch-target uncertainty")
        axis.grid(True, color="white", lw=0.5, alpha=0.45)
        fig.colorbar(image, ax=axis, shrink=0.82, pad=0.02, label="Binned mean")
    fig.suptitle("Three-axis value realization over incumbent-aligned geometry", fontsize=15, y=0.99)
    fig.subplots_adjust(left=0.07, right=0.96, bottom=0.07, top=0.92, hspace=0.28, wspace=0.32)
    save_figure(fig, output_path)


def plot_return_trajectory(events: pd.DataFrame, output_path: Path) -> None:
    if events.empty:
        return
    data = events.loc[
        events["cross_chunk_boundary"].eq(0)
        & events[["depth", "return_gain_from_root"]].notna().all(axis=1)
    ].copy()
    if data.empty:
        return
    fig, axis = plt.subplots(figsize=(10.5, 7.2), facecolor="white")
    for _, bout in data.groupby(["real_step_id", "bout_index"], sort=False):
        ordered = bout.sort_values("status2_position")
        axis.plot(
            ordered["depth"],
            ordered["return_gain_from_root"],
            color="#6B7280",
            alpha=0.08,
            lw=0.7,
        )
    summary = (
        data.groupby("depth", as_index=False)
        .agg(mean_gain=("return_gain_from_root", "mean"), sem_gain=("return_gain_from_root", "sem"), n=("return_gain_from_root", "size"))
        .sort_values("depth")
    )
    summary = summary.loc[summary["n"].ge(5)]
    if not summary.empty:
        x = summary["depth"].to_numpy(float)
        y = summary["mean_gain"].to_numpy(float)
        error = 1.96 * summary["sem_gain"].fillna(0).to_numpy(float)
        axis.fill_between(x, y - error, y + error, color="#4E79A7", alpha=0.18, linewidth=0)
        axis.plot(x, y, color="#2F6690", marker="o", lw=2.2, label="Mean ± 95% SEM")
    axis.axhline(0.0, color="#20252B", lw=0.9, ls="--")
    axis.set_xlabel("Actual imagined path depth")
    axis.set_ylabel("Rollout return − root reference")
    axis.set_title("Within-imagination rollout-value trajectory")
    axis.grid(True, color="#E5E7EB", lw=0.55)
    axis.set_axisbelow(True)
    if not summary.empty:
        axis.legend(frameon=False)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    save_figure(fig, output_path)


def plot_transition_matrix(windows: pd.DataFrame, output_path: Path) -> None:
    data = windows.loc[
        windows["previous_human_action"].between(0, 5)
        & windows["human_action"].between(0, 5)
    ]
    if data.empty:
        return
    matrix = pd.crosstab(
        data["previous_human_action"], data["human_action"]
    ).reindex(index=range(6), columns=range(6), fill_value=0)
    row_sum = matrix.sum(axis=1).replace(0, np.nan)
    probability = matrix.div(row_sum, axis=0)
    fig, axis = plt.subplots(figsize=(8.2, 7.2), facecolor="white")
    image = axis.imshow(probability.to_numpy(float), origin="upper", vmin=0, vmax=1, cmap="magma")
    axis.set_xticks(range(6), [TRACE.ACTION_LABELS.get(i, str(i)) for i in range(6)], rotation=35, ha="right")
    axis.set_yticks(range(6), [TRACE.ACTION_LABELS.get(i, str(i)) for i in range(6)])
    axis.set_xlabel("Current human action")
    axis.set_ylabel("Incumbent human action")
    axis.set_title("Action transitions in the incumbent-aligned analysis")
    fig.colorbar(image, ax=axis, label="P(current action | incumbent)")
    save_figure(fig, output_path)


def write_outputs(
    output_dir: Path,
    windows: pd.DataFrame,
    bouts: pd.DataFrame,
    events: pd.DataFrame,
    canonical_metrics: pd.DataFrame,
    qa: Mapping[str, object],
) -> Dict[str, object]:
    result_dir = output_dir / "results"
    figure_dir = output_dir / "figures"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    windows.to_csv(result_dir / "window_metrics.csv", index=False)
    bouts.to_csv(result_dir / "bout_metrics.csv", index=False)
    events.to_csv(result_dir / "event_metrics.csv", index=False)
    canonical_metrics.to_csv(
        result_dir / "canonical_persistence_analysis_metrics.csv.gz",
        index=False,
        compression="gzip",
    )
    (result_dir / "qa_summary.json").write_text(
        json.dumps(dict(qa), indent=2, default=str), encoding="utf-8"
    )
    canonical_outputs = generate_canonical_persistence_figures(
        canonical_metrics,
        figure_dir,
        result_dir,
    )
    # The primary 3D space is future-blind: current-action margin,
    # alternative uncertainty, and the signed event-wise root-Q revision.
    canonical_outputs["transition_value_revision_geometry"] = (
        plot_transition_value_geometry(
            canonical_metrics,
            figure_dir / "main",
            result_dir / "inference",
            z_mode="net_revision",
        )
    )
    # Preserve the former destination-conditioned value contrast as an
    # explicitly retrospective supplementary sensitivity analysis.
    canonical_outputs["retrospective_destination_value_geometry"] = (
        plot_transition_value_geometry(
            canonical_metrics,
            figure_dir / "supplementary" / "retrospective_destination_value",
            result_dir / "inference",
            z_mode="destination_return",
        )
    )
    canonical_outputs["prospective_branch_value"] = (
        analyze_prospective_branch_value(
            canonical_metrics,
            figure_dir / "main",
            result_dir / "inference",
        )
    )
    return canonical_outputs


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--subject", type=TRACE.parse_subject, required=True)
    parser.add_argument("--game", type=int, choices=[1, 2], required=True)
    parser.add_argument("--sessions", default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--shared-cache-root",
        type=Path,
        default=DEFAULT_SHARED_CACHE_ROOT,
        help="Read-only cache produced by 00_cache.py.",
    )
    parser.add_argument("--max-files", type=int, default=None)
    parser.set_defaults(policy_source="actor")
    parser.add_argument(
        "--root-reference",
        choices=("root_v", "rollout_return", "max_rollout_return"),
        default="root_v",
        help="Root baseline for rollout-value gain; unavailable values fall back safely.",
    )
    parser.add_argument(
        "--policy-temperature",
        type=float,
        default=1.0,
        help="Temperature used for current-action margin and uncertainty.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.max_files is not None and args.max_files <= 0:
        raise ValueError("--max-files must be positive")
    if args.policy_temperature <= 0:
        raise ValueError("--policy-temperature must be positive")
    sessions = TRACE.parse_sessions(args.sessions)
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
    built_parameters = shared_manifest.get("parameters", {})
    if built_parameters.get("persistence_policy_source") != args.policy_source:
        raise RuntimeError(
            "Requested policy source differs from the shared cache. Rebuild "
            "00_cache.py with the required shared policy definition."
        )
    if built_parameters.get("root_reference") != args.root_reference:
        raise RuntimeError(
            "Requested root reference differs from the shared cache. Rebuild "
            "00_cache.py with --root-reference matching this analysis."
        )
    cached_temperature = float(built_parameters.get("policy_temperature", np.nan))
    if not np.isfinite(cached_temperature) or not np.isclose(
        cached_temperature,
        args.policy_temperature,
    ):
        raise RuntimeError(
            "Requested policy temperature differs from the shared cache. "
            "Rebuild 00_cache.py with --policy-temperature matching this analysis."
        )
    windows, bouts, events, qa = load_persistence_geometry(shared_paths)
    if sessions is not None:
        for frame in (windows, bouts, events):
            if "session" not in frame.columns:
                raise KeyError("Shared persistence table is missing session")
        windows = windows[
            pd.to_numeric(windows["session"], errors="coerce").isin(sessions)
        ].copy()
        bouts = bouts[
            pd.to_numeric(bouts["session"], errors="coerce").isin(sessions)
        ].copy()
        events = events[
            pd.to_numeric(events["session"], errors="coerce").isin(sessions)
        ].copy()
    shifted_actor_metrics = load_actor_metrics(shared_paths)
    if sessions is not None:
        shifted_actor_metrics = shifted_actor_metrics.loc[
            pd.to_numeric(
                shifted_actor_metrics["session"], errors="coerce"
            ).isin(sessions)
        ].copy()
    windows, actor_alignment_qa = apply_shifted_actor_current_action_coordinates(
        windows,
        shifted_actor_metrics,
    )
    bouts = add_root_coordinates(windows, bouts)
    windows, value_revision_qa = add_directional_value_revision_balance(
        windows,
        events,
    )
    qa = dict(qa)
    qa["actor_policy_real_step_alignment"] = actor_alignment_qa
    qa["directional_value_revision_balance"] = value_revision_qa
    canonical_metrics, canonical_qa = build_canonical_persistence_metrics(
        windows,
        bouts,
        events,
        subject=args.subject,
        game=args.game,
        sessions=sessions,
        shared_cache_root=args.shared_cache_root,
    )
    canonical_outputs = write_outputs(
        output_dir,
        windows,
        bouts,
        events,
        canonical_metrics,
        qa,
    )
    manifest = {
        "script": str(Path(__file__).resolve()),
        "schema_version": SCHEMA_VERSION,
        "policy_source": args.policy_source,
        "policy_temperature": float(args.policy_temperature),
        "root_reference_requested": args.root_reference,
        "presearch_geometry": (
            "one-real-step-shifted source-state actor policy, followed by "
            "current-action-versus-strongest-competitor margin and conditional "
            "entropy over alternative actions"
        ),
        "realized_search": "reset-delimited status-2 bouts and branch-relative terminal returns",
        "behavioral_transition_3d": (
            "first comparable search to transition-producing search in "
            "current-action-margin--alternative-uncertainty--directional-"
            "value-revision-balance space; value revision is future-blind, "
            "bounded to [-1, 1], and not scaled by the run distribution; "
            "bout endpoints are averaged within run before equal weighting "
            "across runs"
        ),
        "retrospective_sensitivity": (
            "destination-minus-incumbent terminal return is retained under "
            "supplementary/retrospective_destination_value"
        ),
        "prospective_value": (
            "future-blind class-specific alternative-minus-incumbent return "
            "advantage predicts transition risk with run-level inference"
        ),
        "execution_scope": (
            "canonical non-tSNE 02 analysis plus transition-specific 3D and "
            "pairwise probability-mass-change figures; "
            "representation extraction, PCA, and t-SNE are not part of this script"
        ),
        "canonical_coordinate_merge": canonical_qa,
        "canonical_outputs": canonical_outputs,
        "qa": qa,
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    log(f"Complete -> {output_dir}")


if __name__ == "__main__":
    main()
