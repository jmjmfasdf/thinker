#!/usr/bin/env python3
"""
Depth-1 allocation analysis of Thinker search trees.

For every real step, the script reconstructs the local action-path topology and
assigns every deeper imaginary node to its first action below the current root.
It then computes:

1. local tree topology from ``tree_reps['action_seq']``;
2. expansion/revisit labels (exact when an internal-status field is present,
   otherwise explicitly labelled within-real-step first-occurrence proxies);
3. depth-1 visit-allocation geometry;
4. expansion/revisit geometry;
5. branch-conditional depth summaries;
6. stepwise and cumulative root-evidence updates;
7. relations with the Thinker/human committed actions; and
8. cost sensitivity under c_R=1 and c_E=1+delta, delta>0.

The saved trace format uses status=0 for real frames, status=2 for imaginary
frames, and inserts status=1/3 duplicate reset frames.  Only status=2 records
are counted as imaginary search events.  In ``action_seq``, slot zero describes
the root node itself; slot one is therefore the depth-1 action branch.

Examples
--------
python research_script/10_tree_topology_analysis.py --subject 2 --game 2
python research_script/10_tree_topology_analysis.py --subject sub002 --game 2 --sessions 1,2
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import resource
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TRACE_ROOT = ROOT / "test"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs" / "10_tree_topology_analysis"

EPS = 1e-12
REAL_STATUS = 0
IMAGINARY_STATUS = 2
RESET_DUPLICATE_STATUSES = (1, 3)

# Root features whose final dimension has one value per action.
ACTION_EVIDENCE_KEYS = (
    "root_policy",
    "root_qs_mean",
    "root_qs_max",
    "root_ns",
)

# Scalar root/search features.  Their update is assigned to the branch visited
# by the event, but they are not themselves action-specific.
SCALAR_EVIDENCE_KEYS = (
    "root_r",
    "root_v",
    "root_trail_r",
    "rollout_return",
    "max_rollout_return",
)

EXACT_EVENT_KEY_CANDIDATES = (
    "search_event",
    "internal_search_status",
    "internal_status",
    "node_event",
    "cenv_status",
)


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
class SearchEpisode:
    meta: TraceMeta
    real_step: int
    real_step_id: str
    start_global_index: int
    initial_snapshot: Dict[str, np.ndarray]
    previous_snapshot: Dict[str, np.ndarray]
    events: List[Dict[str, object]] = field(default_factory=list)
    seen_paths: set[Tuple[int, ...]] = field(default_factory=set)
    duplicate_reset_records: int = 0
    invalid_path_records: int = 0


def parse_subject(value: object) -> int:
    match = re.search(r"\d+", str(value))
    if not match:
        raise argparse.ArgumentTypeError(f"Cannot parse subject id: {value}")
    return int(match.group(0))


def parse_sessions(value: Optional[str]) -> Optional[set[int]]:
    if value is None or not value.strip():
        return None
    return {int(x.strip()) for x in value.split(",") if x.strip()}


def parse_cost_deltas(value: str) -> List[float]:
    deltas = sorted({float(x.strip()) for x in value.split(",") if x.strip()})
    if not deltas or any((not np.isfinite(x)) or x <= 0 for x in deltas):
        raise argparse.ArgumentTypeError("All cost deltas must be finite and > 0")
    return deltas


def parse_trace_meta(path: Path) -> Optional[TraceMeta]:
    match = re.match(
        r"sub(\d+)-ses(\d+)-block(\d+)-game(\d+)_(\d+)\.npy$", path.name
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
    blocks: Dict[Tuple[int, int, int, int], List[TraceMeta]] = {}
    # The trace root also contains frame-render directories with millions of
    # PNG files.  Prefer the canonical per-subject trace directory so Path does
    # not recursively scan those unrelated trees before finding the npy files.
    subject_dir = trace_root / f"sub{subject:03d}"
    if subject_dir.is_dir():
        candidates = subject_dir.rglob("*.npy")
    else:
        flat_files = trace_root.glob(
            f"sub{subject:03d}-ses*-block*-game{game}_*.npy"
        )
        legacy_files = (
            path
            for directory in trace_root.glob(
                f"sub{subject:03d}-ses*-block*-game{game}"
            )
            for path in directory.glob("video_stat_*.npy")
        )
        candidates = list(flat_files) + list(legacy_files)

    for path in sorted(candidates):
        meta = parse_trace_meta(path)
        if meta is None or meta.subject != subject or meta.game != game:
            continue
        if sessions is not None and meta.session not in sessions:
            continue
        blocks.setdefault(meta.block_id, []).append(meta)
    for metas in blocks.values():
        metas.sort(key=lambda x: x.chunk)
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


def scalar_at(array: object, index: int, default: float = np.nan) -> float:
    if array is None:
        return default
    try:
        value = np.asarray(array[index]).reshape(-1)
    except (IndexError, TypeError, ValueError):
        return default
    return float(value[0]) if value.size else default


def action_at(array: object, index: int) -> int:
    if array is None:
        return -1
    try:
        value = np.asarray(array[index]).reshape(-1)
    except (IndexError, TypeError, ValueError):
        return -1
    if value.size == 0 or not np.any(np.isfinite(value)):
        return -1
    if value.size == 1:
        candidate = int(value[0])
        return candidate if candidate >= 0 else -1
    finite = np.nan_to_num(value.astype(float), nan=0.0)
    if np.max(np.abs(finite)) <= EPS:
        return -1
    return int(np.argmax(finite))


def number_of_actions(tree: Mapping[str, object]) -> int:
    for key in ("root_action", "root_policy", "root_qs_mean", "cur_action"):
        if key not in tree:
            continue
        arr = np.asarray(tree[key])
        if arr.ndim >= 2 and arr.shape[-1] > 1:
            return int(arr.shape[-1])
    raise ValueError("Could not infer number of actions from tree_reps")


def tree_snapshot(
    tree: Mapping[str, object], index: int, num_actions: int
) -> Dict[str, np.ndarray]:
    snapshot: Dict[str, np.ndarray] = {}
    for key in ACTION_EVIDENCE_KEYS:
        if key not in tree:
            continue
        value = np.asarray(tree[key][index], dtype=float).reshape(-1)
        if value.size == num_actions:
            snapshot[key] = value.copy()
    for key in SCALAR_EVIDENCE_KEYS:
        if key not in tree:
            continue
        value = np.asarray(tree[key][index], dtype=float).reshape(-1)
        if value.size:
            snapshot[key] = np.array([value[0]], dtype=float)
    return snapshot


def decode_action_path(
    action_seq_row: object, num_actions: int
) -> Tuple[Tuple[int, ...], int, bool]:
    """Return the path below root, depth, and path-valid flag.

    cenv stores the current root in action_seq slot 0.  The returned path drops
    that slot, making path[0] the depth-1 candidate relevant to this analysis.
    """
    flat = np.asarray(action_seq_row, dtype=float).reshape(-1)
    if flat.size == 0 or flat.size % num_actions != 0:
        return tuple(), 0, False
    slots = flat.reshape(-1, num_actions)
    occupied = np.sum(np.abs(slots), axis=1) > 0.5
    occupied_indices = np.flatnonzero(occupied)
    if occupied_indices.size == 0 or occupied_indices[0] != 0:
        return tuple(), 0, False
    last = int(occupied_indices[-1])
    if not np.all(occupied[: last + 1]):
        return tuple(), 0, False
    full_path = tuple(int(np.argmax(slots[d])) for d in range(last + 1))
    below_root = full_path[1:]
    return below_root, len(below_root), len(below_root) > 0


def exact_event_array(
    data: Mapping[str, object], requested_key: Optional[str]
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    candidates: Iterable[str]
    if requested_key:
        candidates = (requested_key,)
    else:
        candidates = EXACT_EVENT_KEY_CANDIDATES
    for key in candidates:
        if key not in data:
            continue
        arr = np.asarray(data[key]).reshape(-1)
        if arr.size:
            return arr, key
    return None, None


def classify_event(
    exact_status: Optional[int],
    path: Tuple[int, ...],
    seen_paths: set[Tuple[int, ...]],
) -> Tuple[str, str]:
    if exact_status is not None:
        mapping = {
            2: "revisit",
            3: "terminal",
            4: "expansion",
            5: "reset",
        }
        return mapping.get(int(exact_status), "unclassified"), "exact_internal_status"
    if path in seen_paths:
        return "observed_revisit", "within_real_step_path_proxy"
    return "observed_new", "within_real_step_path_proxy"


def normalized_entropy(values: Sequence[float]) -> float:
    x = np.asarray(values, dtype=float)
    total = np.nansum(x)
    if x.size <= 1 or total <= 0:
        return 0.0 if total > 0 else np.nan
    p = x / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)) / np.log(x.size))


def hhi(values: Sequence[float]) -> float:
    x = np.asarray(values, dtype=float)
    total = np.nansum(x)
    if total <= 0:
        return np.nan
    return float(np.sum((x / total) ** 2))


def safe_share(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else np.nan


def evidence_margin(vector: Optional[np.ndarray], action: int) -> float:
    if vector is None or not (0 <= action < len(vector)):
        return np.nan
    others = np.delete(np.asarray(vector, dtype=float), action)
    if others.size == 0 or not np.any(np.isfinite(others)):
        return np.nan
    return float(vector[action] - np.nanmax(others))


def cost_label(delta: float) -> str:
    return f"delta_{delta:g}".replace(".", "p").replace("-", "m")


def build_event(
    episode: SearchEpisode,
    tree: Mapping[str, object],
    data: Mapping[str, object],
    row_index: int,
    global_index: int,
    num_actions: int,
    exact_status: Optional[int],
) -> Optional[Dict[str, object]]:
    path, depth, valid = decode_action_path(tree["action_seq"][row_index], num_actions)
    if not valid:
        episode.invalid_path_records += 1
        return None

    event_type, event_source = classify_event(exact_status, path, episode.seen_paths)
    episode.seen_paths.add(path)
    current = tree_snapshot(tree, row_index, num_actions)
    branch = int(path[0])
    event: Dict[str, object] = {
        "subject": episode.meta.subject,
        "session": episode.meta.session,
        "block": episode.meta.block,
        "game": episode.meta.game,
        "run_label": episode.meta.run_label,
        "real_step": episode.real_step,
        "real_step_id": episode.real_step_id,
        "event_index": len(episode.events),
        "global_trace_index": global_index,
        "root_branch": branch,
        "depth": depth,
        "action_path": ">".join(str(x) for x in path),
        "parent_path": ">".join(str(x) for x in path[:-1]),
        "event_type": event_type,
        "event_label_source": event_source,
        "exact_internal_status": exact_status if exact_status is not None else np.nan,
        "reset_after_event": int(round(scalar_at(tree.get("cur_reset"), row_index, 0.0))),
        "current_node_done": int(round(scalar_at(tree.get("cur_d"), row_index, 0.0))),
    }

    if "k" in tree:
        k_row = np.asarray(tree["k"][row_index]).reshape(-1)
        event["search_slot"] = int(np.argmax(k_row)) if k_row.size else -1
    else:
        event["search_slot"] = len(episode.events) + 1

    step_times = data.get("step_times")
    event["step_time"] = np.nan
    if step_times is not None:
        timing = np.asarray(step_times[row_index], dtype=float).reshape(-1)
        slot = int(event["search_slot"])
        if 0 <= slot < timing.size:
            event["step_time"] = float(timing[slot])

    previous = episode.previous_snapshot
    for key in ACTION_EVIDENCE_KEYS:
        if key not in previous or key not in current:
            continue
        before = float(previous[key][branch])
        after = float(current[key][branch])
        event[f"{key}_before"] = before
        event[f"{key}_after"] = after
        event[f"{key}_delta"] = after - before
    for key in SCALAR_EVIDENCE_KEYS:
        if key not in previous or key not in current:
            continue
        before = float(previous[key][0])
        after = float(current[key][0])
        event[f"{key}_before"] = before
        event[f"{key}_after"] = after
        event[f"{key}_delta"] = after - before

    episode.previous_snapshot = current
    return event


def topology_rows_for_episode(
    episode: SearchEpisode,
    committed_thinker: int,
    committed_human: int,
) -> List[Dict[str, object]]:
    stats: Dict[Tuple[int, ...], Dict[str, object]] = {
        tuple(): {
            "direct_visit_count": 0,
            "descendant_event_count": len(episode.events),
            "first_event_index": -1,
            "first_event_type": "root",
        }
    }
    for event in episode.events:
        path = tuple(int(x) for x in str(event["action_path"]).split(">") if x != "")
        for depth in range(1, len(path) + 1):
            prefix = path[:depth]
            node = stats.setdefault(
                prefix,
                {
                    "direct_visit_count": 0,
                    "descendant_event_count": 0,
                    "first_event_index": int(event["event_index"]),
                    "first_event_type": event["event_type"] if depth == len(path) else "prefix",
                },
            )
            node["descendant_event_count"] = int(node["descendant_event_count"]) + 1
            if depth == len(path):
                node["direct_visit_count"] = int(node["direct_visit_count"]) + 1

    paths = set(stats)
    rows: List[Dict[str, object]] = []
    for path, node in sorted(stats.items(), key=lambda item: (len(item[0]), item[0])):
        children = sum(1 for other in paths if len(other) == len(path) + 1 and other[:-1] == path)
        root_branch = path[0] if path else -1
        rows.append(
            {
                "subject": episode.meta.subject,
                "session": episode.meta.session,
                "block": episode.meta.block,
                "game": episode.meta.game,
                "run_label": episode.meta.run_label,
                "real_step": episode.real_step,
                "real_step_id": episode.real_step_id,
                "node_path": ">".join(str(x) for x in path),
                "parent_path": ">".join(str(x) for x in path[:-1]),
                "root_branch": root_branch,
                "depth": len(path),
                "direct_visit_count": node["direct_visit_count"],
                "descendant_event_count": node["descendant_event_count"],
                "observed_child_count": children,
                "is_observed_leaf": int(children == 0),
                "first_event_index": node["first_event_index"],
                "first_event_type": node["first_event_type"],
                "is_thinker_committed_branch": int(root_branch == committed_thinker),
                "is_human_committed_branch": int(root_branch == committed_human),
            }
        )
    return rows


def finalize_episode(
    episode: SearchEpisode,
    num_actions: int,
    committed_thinker: int,
    committed_human: int,
    complete: bool,
    cost_deltas: Sequence[float],
) -> Tuple[
    List[Dict[str, object]],
    List[Dict[str, object]],
    Dict[str, object],
    List[Dict[str, object]],
    List[Dict[str, object]],
]:
    events = episode.events
    for event in events:
        branch = int(event["root_branch"])
        event["thinker_committed_action"] = committed_thinker
        event["human_committed_action"] = committed_human
        event["is_thinker_committed_branch"] = int(branch == committed_thinker)
        event["is_human_committed_branch"] = int(branch == committed_human)
        event["real_step_complete"] = int(complete)

    final_snapshot = episode.previous_snapshot
    branch_rows: List[Dict[str, object]] = []
    visit_counts = np.zeros(num_actions, dtype=int)
    expansion_counts = np.zeros(num_actions, dtype=int)
    revisit_counts = np.zeros(num_actions, dtype=int)
    terminal_counts = np.zeros(num_actions, dtype=int)
    reset_counts = np.zeros(num_actions, dtype=int)

    for action in range(num_actions):
        branch_events = [e for e in events if int(e["root_branch"]) == action]
        visits = len(branch_events)
        expansions = sum(e["event_type"] in ("expansion", "observed_new") for e in branch_events)
        revisits = sum(e["event_type"] in ("revisit", "observed_revisit") for e in branch_events)
        terminals = sum(e["event_type"] == "terminal" for e in branch_events)
        resets = sum(e["event_type"] == "reset" for e in branch_events)
        depths = np.asarray([e["depth"] for e in branch_events], dtype=float)
        visit_counts[action] = visits
        expansion_counts[action] = expansions
        revisit_counts[action] = revisits
        terminal_counts[action] = terminals
        reset_counts[action] = resets

        row: Dict[str, object] = {
            "subject": episode.meta.subject,
            "session": episode.meta.session,
            "block": episode.meta.block,
            "game": episode.meta.game,
            "run_label": episode.meta.run_label,
            "real_step": episode.real_step,
            "real_step_id": episode.real_step_id,
            "action": action,
            "thinker_committed_action": committed_thinker,
            "human_committed_action": committed_human,
            "is_thinker_committed": int(action == committed_thinker),
            "is_human_committed": int(action == committed_human),
            "visit_count": visits,
            "expansion_count": expansions,
            "revisit_count": revisits,
            "terminal_count": terminals,
            "reset_count": resets,
            "operation_count": expansions + revisits,
            "expansion_fraction": safe_share(expansions, expansions + revisits),
            "mean_depth": float(np.mean(depths)) if depths.size else np.nan,
            "max_depth": float(np.max(depths)) if depths.size else np.nan,
            "depth1_fraction": safe_share(float(np.sum(depths == 1)), visits),
            "depth2_fraction": safe_share(float(np.sum(depths == 2)), visits),
            "depth3plus_fraction": safe_share(float(np.sum(depths >= 3)), visits),
            "real_step_complete": int(complete),
        }

        for key in ACTION_EVIDENCE_KEYS:
            initial = episode.initial_snapshot.get(key)
            final = final_snapshot.get(key)
            if initial is None or final is None:
                continue
            row[f"{key}_initial"] = float(initial[action])
            row[f"{key}_final"] = float(final[action])
            row[f"{key}_net"] = float(final[action] - initial[action])
            deltas = np.asarray(
                [e.get(f"{key}_delta", np.nan) for e in branch_events], dtype=float
            )
            row[f"{key}_assigned_update"] = float(np.nansum(deltas)) if deltas.size else 0.0
            row[f"{key}_assigned_abs_update"] = (
                float(np.nansum(np.abs(deltas))) if deltas.size else 0.0
            )

        for key in SCALAR_EVIDENCE_KEYS:
            initial = episode.initial_snapshot.get(key)
            final = final_snapshot.get(key)
            if initial is None or final is None:
                continue
            deltas = np.asarray(
                [e.get(f"{key}_delta", np.nan) for e in branch_events], dtype=float
            )
            row[f"{key}_assigned_update"] = float(np.nansum(deltas)) if deltas.size else 0.0
            row[f"{key}_assigned_abs_update"] = (
                float(np.nansum(np.abs(deltas))) if deltas.size else 0.0
            )
        branch_rows.append(row)

    total_visits = int(np.sum(visit_counts))
    total_expansions = int(np.sum(expansion_counts))
    total_revisits = int(np.sum(revisit_counts))
    total_operations = total_expansions + total_revisits
    for row in branch_rows:
        action = int(row["action"])
        row["visit_share"] = safe_share(visit_counts[action], total_visits)
        row["expansion_share"] = safe_share(expansion_counts[action], total_expansions)
        row["revisit_share"] = safe_share(revisit_counts[action], total_revisits)

    real_row: Dict[str, object] = {
        "subject": episode.meta.subject,
        "session": episode.meta.session,
        "block": episode.meta.block,
        "game": episode.meta.game,
        "run_label": episode.meta.run_label,
        "real_step": episode.real_step,
        "real_step_id": episode.real_step_id,
        "thinker_committed_action": committed_thinker,
        "human_committed_action": committed_human,
        "real_step_complete": int(complete),
        "imaginary_event_count": total_visits,
        "operation_count": total_operations,
        "expansion_count": total_expansions,
        "revisit_count": total_revisits,
        "terminal_count": int(np.sum(terminal_counts)),
        "reset_event_count": int(np.sum(reset_counts)),
        "duplicate_reset_record_count": episode.duplicate_reset_records,
        "invalid_path_record_count": episode.invalid_path_records,
        "effective_search_fraction": safe_share(total_operations, total_visits),
        "expansion_fraction": safe_share(total_expansions, total_operations),
        "visit_entropy": normalized_entropy(visit_counts),
        "visit_hhi": hhi(visit_counts),
        "visit_top_share": safe_share(float(np.max(visit_counts)), total_visits),
        "visit_top_action": int(np.argmax(visit_counts)) if total_visits else -1,
        "searched_branch_count": int(np.sum(visit_counts > 0)),
        "thinker_chosen_visit_share": (
            safe_share(visit_counts[committed_thinker], total_visits)
            if 0 <= committed_thinker < num_actions else np.nan
        ),
        "human_chosen_visit_share": (
            safe_share(visit_counts[committed_human], total_visits)
            if 0 <= committed_human < num_actions else np.nan
        ),
        "thinker_chosen_expansion_share": (
            safe_share(expansion_counts[committed_thinker], total_expansions)
            if 0 <= committed_thinker < num_actions else np.nan
        ),
        "human_chosen_expansion_share": (
            safe_share(expansion_counts[committed_human], total_expansions)
            if 0 <= committed_human < num_actions else np.nan
        ),
    }

    for key in ACTION_EVIDENCE_KEYS:
        initial = episode.initial_snapshot.get(key)
        final = final_snapshot.get(key)
        if initial is None or final is None:
            continue
        real_row[f"{key}_initial_best_action"] = int(np.nanargmax(initial))
        real_row[f"{key}_final_best_action"] = int(np.nanargmax(final))
        real_row[f"{key}_ranking_changed"] = int(np.nanargmax(initial) != np.nanargmax(final))
        for prefix, action in (
            ("thinker", committed_thinker), ("human", committed_human)
        ):
            initial_margin = evidence_margin(initial, action)
            final_margin = evidence_margin(final, action)
            real_row[f"{prefix}_{key}_initial_margin"] = initial_margin
            real_row[f"{prefix}_{key}_final_margin"] = final_margin
            real_row[f"{prefix}_{key}_margin_change"] = final_margin - initial_margin

    cost_rows: List[Dict[str, object]] = []
    for delta in cost_deltas:
        costs = revisit_counts.astype(float) + (1.0 + delta) * expansion_counts
        total_cost = float(np.sum(costs))
        label = cost_label(delta)
        chosen_cost_share = (
            safe_share(costs[committed_thinker], total_cost)
            if 0 <= committed_thinker < num_actions else np.nan
        )
        real_row[f"cost_entropy_{label}"] = normalized_entropy(costs)
        real_row[f"cost_hhi_{label}"] = hhi(costs)
        real_row[f"thinker_chosen_cost_share_{label}"] = chosen_cost_share
        for row in branch_rows:
            action = int(row["action"])
            row[f"cost_{label}"] = float(costs[action])
            row[f"cost_share_{label}"] = safe_share(costs[action], total_cost)
        cost_rows.append(
            {
                "subject": episode.meta.subject,
                "session": episode.meta.session,
                "block": episode.meta.block,
                "game": episode.meta.game,
                "run_label": episode.meta.run_label,
                "real_step": episode.real_step,
                "real_step_id": episode.real_step_id,
                "delta": delta,
                "c_revisit": 1.0,
                "c_expansion": 1.0 + delta,
                "total_cost": total_cost,
                "cost_entropy": normalized_entropy(costs),
                "cost_hhi": hhi(costs),
                "cost_top_share": safe_share(float(np.max(costs)), total_cost),
                "cost_top_action": int(np.argmax(costs)) if total_cost else -1,
                "thinker_chosen_cost_share": chosen_cost_share,
                "human_chosen_cost_share": (
                    safe_share(costs[committed_human], total_cost)
                    if 0 <= committed_human < num_actions else np.nan
                ),
                "cost_top_matches_thinker": int(
                    total_cost > 0 and int(np.argmax(costs)) == committed_thinker
                ),
            }
        )

    topology_rows = topology_rows_for_episode(
        episode, committed_thinker=committed_thinker, committed_human=committed_human
    )
    return events, branch_rows, real_row, cost_rows, topology_rows


def process_block(
    metas: Sequence[TraceMeta],
    cost_deltas: Sequence[float],
    requested_event_key: Optional[str],
    max_files: Optional[int],
) -> Tuple[
    List[Dict[str, object]],
    List[Dict[str, object]],
    List[Dict[str, object]],
    List[Dict[str, object]],
    List[Dict[str, object]],
    Dict[str, object],
]:
    event_rows: List[Dict[str, object]] = []
    branch_rows: List[Dict[str, object]] = []
    real_rows: List[Dict[str, object]] = []
    cost_rows: List[Dict[str, object]] = []
    topology_rows: List[Dict[str, object]] = []
    active: Optional[SearchEpisode] = None
    num_actions: Optional[int] = None
    global_index = 0
    real_step = 0
    exact_keys: set[str] = set()
    event_label_sources: set[str] = set()
    missing_active_imaginary = 0
    loaded_files = 0

    selected_metas = list(metas)
    if max_files is not None:
        selected_metas = selected_metas[:max_files]

    def close_active(thinker_action: int, human_action: int, complete: bool) -> None:
        nonlocal active
        if active is None or num_actions is None:
            return
        # visual_behav seeds a new status=0 frame after an environment episode
        # and may end the chunk before any search follows.  It is not a search
        # episode and should not become a zero-budget real-step row.
        if not active.events:
            active = None
            return
        events, branches, real, costs, nodes = finalize_episode(
            active,
            num_actions=num_actions,
            committed_thinker=thinker_action,
            committed_human=human_action,
            complete=complete,
            cost_deltas=cost_deltas,
        )
        event_rows.extend(events)
        branch_rows.extend(branches)
        real_rows.append(real)
        cost_rows.extend(costs)
        topology_rows.extend(nodes)
        active = None

    for meta in selected_metas:
        log(f"Loading {meta.path} ({meta.path.stat().st_size / 1024**3:.1f} GiB)")
        data = load_npy_dict(meta.path)
        loaded_files += 1
        if "tree_reps" not in data or "status" not in data:
            raise ValueError(f"Missing tree_reps/status in {meta.path}")
        tree = data["tree_reps"]
        if not isinstance(tree, Mapping) or "action_seq" not in tree:
            raise ValueError(f"Missing tree_reps['action_seq'] in {meta.path}")
        status = np.asarray(data["status"]).reshape(-1)
        if num_actions is None:
            num_actions = number_of_actions(tree)
            log(f"Detected {num_actions} actions")
        elif number_of_actions(tree) != num_actions:
            raise ValueError(f"Action count changed within block: {meta.path}")

        exact_events, exact_key = exact_event_array(data, requested_event_key)
        if requested_event_key and exact_key is None:
            raise KeyError(f"Requested event-label key '{requested_event_key}' missing in {meta.path}")
        if exact_key:
            exact_keys.add(exact_key)
            if len(exact_events) != len(status):
                raise ValueError(f"{exact_key} length does not match status in {meta.path}")

        for row_index, public_status in enumerate(status):
            public_status = int(public_status)
            if public_status == REAL_STATUS:
                if active is not None:
                    close_active(
                        thinker_action=action_at(data.get("thinker_action"), row_index),
                        human_action=action_at(data.get("human_action"), row_index),
                        complete=True,
                    )
                    real_step += 1
                initial = tree_snapshot(tree, row_index, num_actions)
                real_step_id = f"{meta.run_label}_real{real_step:06d}"
                active = SearchEpisode(
                    meta=meta,
                    real_step=real_step,
                    real_step_id=real_step_id,
                    start_global_index=global_index,
                    initial_snapshot=initial,
                    previous_snapshot={k: v.copy() for k, v in initial.items()},
                )
            elif public_status == IMAGINARY_STATUS:
                if active is None:
                    missing_active_imaginary += 1
                else:
                    internal = int(exact_events[row_index]) if exact_events is not None else None
                    event = build_event(
                        active,
                        tree=tree,
                        data=data,
                        row_index=row_index,
                        global_index=global_index,
                        num_actions=num_actions,
                        exact_status=internal,
                    )
                    if event is not None:
                        event_label_sources.add(str(event["event_label_source"]))
                        active.events.append(event)
            elif public_status in RESET_DUPLICATE_STATUSES:
                if active is not None:
                    active.duplicate_reset_records += 1
            global_index += 1

        del tree, data
        gc.collect()
        log(f"Processed {meta.path.name}")

    if active is not None:
        close_active(thinker_action=-1, human_action=-1, complete=False)

    qa = {
        "loaded_files": loaded_files,
        "num_actions": num_actions,
        "exact_event_keys": sorted(exact_keys),
        "event_label_mode": (
            "exact"
            if event_label_sources == {"exact_internal_status"}
            else "proxy"
            if event_label_sources == {"within_real_step_path_proxy"}
            else "mixed"
            if event_label_sources
            else "unknown"
        ),
        "imaginary_records_without_active_real_step": missing_active_imaginary,
    }
    return event_rows, branch_rows, real_rows, cost_rows, topology_rows, qa


def mean_or_nan(values: object) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else np.nan


def save_summary_figure(
    branch_df: pd.DataFrame,
    real_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    output_path: Path,
    commit_source: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    actions = sorted(branch_df["action"].unique())
    grouped = branch_df.groupby("action", observed=True)
    means_e = grouped["expansion_count"].mean().reindex(actions).fillna(0)
    means_r = grouped["revisit_count"].mean().reindex(actions).fillna(0)
    axes[0, 0].bar(actions, means_e, label="Expansion / observed-new", color="#4c78a8")
    axes[0, 0].bar(actions, means_r, bottom=means_e, label="Revisit", color="#f58518")
    axes[0, 0].set(title="Mean branch allocation", xlabel="Depth-1 action", ylabel="Events / real step")
    axes[0, 0].legend(frameon=False, fontsize=8)

    chosen_col = f"is_{commit_source}_committed"
    chosen = branch_df.loc[branch_df[chosen_col] == 1, "visit_share"].dropna()
    unchosen = branch_df.loc[branch_df[chosen_col] == 0, "visit_share"].dropna()
    axes[0, 1].boxplot(
        [chosen, unchosen], tick_labels=["Committed", "Other"], showfliers=False
    )
    axes[0, 1].set(title="Visit allocation and commitment", ylabel="Visit share")

    entropy = real_df["visit_entropy"].dropna()
    axes[1, 0].hist(entropy, bins=25, color="#54a24b", alpha=0.85)
    axes[1, 0].axvline(mean_or_nan(entropy), color="black", linestyle="--", linewidth=1)
    axes[1, 0].set(title="Allocation concentration", xlabel="Normalized visit entropy", ylabel="Real steps")

    if not cost_df.empty:
        cost_summary = cost_df.groupby("delta", observed=True)[f"{commit_source}_chosen_cost_share"].mean()
        axes[1, 1].plot(cost_summary.index, cost_summary.values, marker="o", color="#e45756")
        axes[1, 1].set_xscale("log")
        axes[1, 1].set(title=r"Cost sensitivity ($c_E=1+\delta$, $c_R=1$)", xlabel=r"$\delta$", ylabel="Committed-action cost share")
    else:
        axes[1, 1].text(0.5, 0.5, "No cost rows", ha="center", va="center")
        axes[1, 1].set_axis_off()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_evidence_figure(
    real_df: pd.DataFrame,
    output_path: Path,
    commit_source: str,
) -> None:
    margin_initial = f"{commit_source}_root_qs_mean_initial_margin"
    margin_final = f"{commit_source}_root_qs_mean_final_margin"
    margin_change = f"{commit_source}_root_qs_mean_margin_change"
    share = f"{commit_source}_chosen_visit_share"
    if not {margin_initial, margin_final, margin_change, share}.issubset(real_df.columns):
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    valid = real_df[[margin_initial, margin_final]].dropna()
    axes[0].scatter(valid[margin_initial], valid[margin_final], s=10, alpha=0.35)
    if not valid.empty:
        lo = float(np.nanmin(valid.to_numpy()))
        hi = float(np.nanmax(valid.to_numpy()))
        axes[0].plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1)
    axes[0].set(title="Committed-action Q evidence", xlabel="Initial Q margin", ylabel="Final Q margin")

    valid = real_df[[share, margin_change]].dropna()
    axes[1].scatter(valid[share], valid[margin_change], s=10, alpha=0.35, color="#4c78a8")
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set(title="Allocation versus evidence update", xlabel="Committed-action visit share", ylabel="Q-margin change")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_evidence_by_human_action_figure(
    real_df: pd.DataFrame,
    output_path: Path,
    commit_source: str,
) -> None:
    """Label the evidence-update geometry by the human action at each real step.

    The scatter panels use a deterministic, action-stratified subsample to keep
    overplotting manageable.  The two summary panels use every valid real step.
    The Q margins remain defined relative to ``commit_source``; human action is
    a label, not a redefinition of the committed-action evidence measure.
    """
    margin_initial = f"{commit_source}_root_qs_mean_initial_margin"
    margin_final = f"{commit_source}_root_qs_mean_final_margin"
    margin_change = f"{commit_source}_root_qs_mean_margin_change"
    share = f"{commit_source}_chosen_visit_share"
    required = {
        "human_committed_action",
        margin_initial,
        margin_final,
        margin_change,
        share,
    }
    if not required.issubset(real_df.columns):
        return

    valid = real_df[list(required)].dropna().copy()
    valid["human_committed_action"] = valid["human_committed_action"].astype(int)
    valid = valid.loc[valid["human_committed_action"] >= 0]
    if valid.empty:
        return

    actions = sorted(valid["human_committed_action"].unique())
    cmap = plt.get_cmap("tab10")
    colors = {action: cmap(i % 10) for i, action in enumerate(actions)}
    sampled_parts = []
    for action in actions:
        group = valid.loc[valid["human_committed_action"] == action]
        sampled_parts.append(
            group.sample(n=min(len(group), 12_000), random_state=10_002 + action)
        )
    sampled = pd.concat(sampled_parts, ignore_index=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    lo = float(np.nanmin(valid[[margin_initial, margin_final]].to_numpy()))
    hi = float(np.nanmax(valid[[margin_initial, margin_final]].to_numpy()))
    for action in actions:
        group = sampled.loc[sampled["human_committed_action"] == action]
        axes[0, 0].scatter(
            group[margin_initial],
            group[margin_final],
            s=5,
            alpha=0.12,
            color=colors[action],
            label=f"Human action {action}",
            rasterized=True,
        )
    axes[0, 0].plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1)
    axes[0, 0].set(
        title=f"{commit_source.title()}-action Q evidence, labeled by human action",
        xlabel="Initial Q margin",
        ylabel="Final Q margin",
        xlim=(lo, hi),
        ylim=(lo, hi),
    )
    axes[0, 0].legend(frameon=False, fontsize=8, markerscale=2)

    above = valid[margin_final] > valid[margin_initial]
    above_by_action = above.groupby(valid["human_committed_action"]).mean().reindex(actions)
    counts = valid.groupby("human_committed_action", observed=True).size().reindex(actions)
    bars = axes[0, 1].bar(
        actions,
        above_by_action.values,
        color=[colors[action] for action in actions],
    )
    axes[0, 1].axhline(float(above.mean()), color="black", linestyle="--", linewidth=1)
    axes[0, 1].set(
        title="Fraction above the identity line",
        xlabel="Human action",
        ylabel=r"$P(\mathrm{final\ margin} > \mathrm{initial\ margin})$",
        ylim=(0, 0.72),
        xticks=actions,
    )
    for bar, action in zip(bars, actions):
        axes[0, 1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f"{bar.get_height():.3f}\nn={int(counts.loc[action]):,}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    for action in actions:
        group = sampled.loc[sampled["human_committed_action"] == action]
        axes[1, 0].scatter(
            group[share],
            group[margin_change],
            s=5,
            alpha=0.12,
            color=colors[action],
            label=f"Human action {action}",
            rasterized=True,
        )
    axes[1, 0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1, 0].set(
        title="Allocation versus evidence update, labeled by human action",
        xlabel=f"{commit_source.title()}-action visit share",
        ylabel="Q-margin change",
    )

    grouped = valid.groupby("human_committed_action", observed=True)[margin_change]
    means = grouped.mean().reindex(actions)
    medians = grouped.median().reindex(actions)
    bars = axes[1, 1].bar(
        actions,
        means.values,
        color=[colors[action] for action in actions],
        alpha=0.85,
        label="Mean",
    )
    axes[1, 1].scatter(actions, medians.values, color="black", marker="D", s=28, label="Median")
    axes[1, 1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1, 1].set(
        title="Q-margin change by human action",
        xlabel="Human action",
        ylabel="Q-margin change",
        xticks=actions,
    )
    axes[1, 1].legend(frameon=False, fontsize=8)
    for bar in bars:
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.035,
            f"{bar.get_height():.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.suptitle(
        "Human-action labeling of real-step evidence updates\n"
        "Scatter panels are action-stratified samples; summaries use all valid steps",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_readme(
    path: Path,
    args: argparse.Namespace,
    event_df: pd.DataFrame,
    branch_df: pd.DataFrame,
    real_df: pd.DataFrame,
    qa: Mapping[str, object],
) -> None:
    complete = real_df[real_df["real_step_complete"] == 1]
    budgets = complete["imaginary_event_count"].dropna()
    label_mode = qa.get("event_label_mode", "unknown")
    if label_mode == "exact":
        label_note = "Expansion/revisit labels use the saved internal cenv status."
    elif label_mode == "mixed":
        label_note = (
            "Some blocks use saved internal cenv statuses and others use within-real-step "
            "path proxies. Inspect qa_summary.json before combining their operation labels."
        )
    else:
        label_note = (
            "Expansion/revisit labels are within-real-step path proxies: the first observed "
            "path is observed_new and later occurrences are observed_revisit. With tree carry, "
            "observed_new is not guaranteed to be a true expansion."
        )
    lines = [
        "# Tree topology analysis summary",
        "",
        f"- Subject: sub{args.subject:03d}",
        f"- Game: {args.game}",
        f"- Sessions: {args.sessions or 'all discovered'}",
        f"- Event-label mode: {label_mode}",
        f"- Complete real steps: {len(complete)}",
        f"- Imaginary events: {len(event_df)}",
        f"- Branch rows: {len(branch_df)}",
        f"- Mean imaginary budget: {mean_or_nan(budgets):.3f}",
        f"- Distinct complete-step budgets: {sorted(int(x) for x in budgets.unique()) if len(budgets) else []}",
        f"- Mean visit entropy: {mean_or_nan(complete['visit_entropy']):.4f}",
        f"- Mean expansion fraction: {mean_or_nan(complete['expansion_fraction']):.4f}",
        "",
        "## Interpretation guardrail",
        "",
        label_note,
        "",
        "Cost sensitivity uses c_R=1 and c_E=1+delta. The primary reusable variables "
        "remain visit_count and expansion_count rather than any single assumed cost ratio.",
        "",
        "## Outputs",
        "",
        "- imaginary_events.csv: one row per valid imaginary event",
        "- node_topology.csv: one row per observed local path node",
        "- branch_search_shape.csv: one row per real-step × depth-1 action",
        "- real_step_search_shape.csv: one row per real step",
        "- cost_sensitivity.csv: one row per real-step × delta",
        "- qa_summary.json: data and reconstruction checks",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=parse_subject, required=True, help="Subject, e.g. 2 or sub002")
    parser.add_argument("--game", type=int, choices=[1, 2], required=True)
    parser.add_argument("--sessions", default=None, help="Comma-separated session ids, e.g. 1,2")
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--cost-deltas",
        type=parse_cost_deltas,
        default=parse_cost_deltas("0.1,0.25,0.5,1,2,4,8"),
        help="Positive delta values for c_R=1, c_E=1+delta",
    )
    parser.add_argument(
        "--event-label-key",
        default=None,
        help="Optional top-level trace key containing exact internal cenv statuses",
    )
    parser.add_argument(
        "--commit-source",
        choices=["thinker", "human"],
        default="thinker",
        help="Committed action used in summary figures",
    )
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Diagnostic only: cap chunks loaded per block",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    sessions = parse_sessions(args.sessions)
    blocks = gather_trace_blocks(args.trace_root, args.subject, args.game, sessions)
    if not blocks:
        raise FileNotFoundError(
            f"No traces for subject=sub{args.subject:03d}, game={args.game}, "
            f"sessions={args.sessions or 'all'} under {args.trace_root}"
        )

    output_dir = args.output_root / f"sub{args.subject:03d}_game{args.game}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Found {len(blocks)} blocks; output -> {output_dir}")

    all_events: List[Dict[str, object]] = []
    all_branches: List[Dict[str, object]] = []
    all_real: List[Dict[str, object]] = []
    all_cost: List[Dict[str, object]] = []
    all_topology: List[Dict[str, object]] = []
    block_qa: List[Dict[str, object]] = []

    for block_id, metas in blocks.items():
        log(f"Processing block {block_id} with {len(metas)} chunk(s)")
        events, branches, real, costs, topology, qa = process_block(
            metas,
            cost_deltas=args.cost_deltas,
            requested_event_key=args.event_label_key,
            max_files=args.max_files,
        )
        qa = {"block_id": list(block_id), **qa}
        block_qa.append(qa)
        all_events.extend(events)
        all_branches.extend(branches)
        all_real.extend(real)
        all_cost.extend(costs)
        all_topology.extend(topology)

    event_df = pd.DataFrame(all_events)
    branch_df = pd.DataFrame(all_branches)
    real_df = pd.DataFrame(all_real)
    cost_df = pd.DataFrame(all_cost)
    topology_df = pd.DataFrame(all_topology)
    if real_df.empty:
        raise RuntimeError("No real-step search episodes were reconstructed")

    event_df.to_csv(output_dir / "imaginary_events.csv", index=False)
    topology_df.to_csv(output_dir / "node_topology.csv", index=False)
    branch_df.to_csv(output_dir / "branch_search_shape.csv", index=False)
    real_df.to_csv(output_dir / "real_step_search_shape.csv", index=False)
    cost_df.to_csv(output_dir / "cost_sensitivity.csv", index=False)

    block_modes = {str(q["event_label_mode"]) for q in block_qa}
    qa_summary: Dict[str, object] = {
        "subject": args.subject,
        "game": args.game,
        "sessions": sorted(sessions) if sessions is not None else None,
        "blocks": block_qa,
        "event_label_mode": (
            next(iter(block_modes))
            if len(block_modes) == 1 and next(iter(block_modes)) in {"exact", "proxy"}
            else "mixed"
        ),
        "cost_deltas": args.cost_deltas,
        "real_steps": len(real_df),
        "complete_real_steps": int(real_df["real_step_complete"].sum()),
        "imaginary_events": len(event_df),
        "branch_rows": len(branch_df),
        "topology_nodes": len(topology_df),
        "complete_step_budget_values": sorted(
            int(x)
            for x in real_df.loc[
                real_df["real_step_complete"] == 1, "imaginary_event_count"
            ].unique()
        ),
    }
    (output_dir / "qa_summary.json").write_text(
        json.dumps(qa_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_readme(
        output_dir / "README.md",
        args=args,
        event_df=event_df,
        branch_df=branch_df,
        real_df=real_df,
        qa=qa_summary,
    )

    if not args.skip_plots and not branch_df.empty:
        save_summary_figure(
            branch_df,
            real_df,
            cost_df,
            output_dir / "fig_10_1_search_shape_summary.png",
            commit_source=args.commit_source,
        )
        save_evidence_figure(
            real_df,
            output_dir / "fig_10_2_evidence_updates.png",
            commit_source=args.commit_source,
        )
        save_evidence_by_human_action_figure(
            real_df,
            output_dir / "fig_10_2a_evidence_updates_by_human_action.png",
            commit_source=args.commit_source,
        )

    log(
        f"Done: {len(real_df)} real steps, {len(event_df)} imaginary events, "
        f"label_mode={qa_summary['event_label_mode']}"
    )


if __name__ == "__main__":
    main()
