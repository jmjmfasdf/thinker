#!/usr/bin/env python3
"""Held-out behavioral validation of the complete 29-event Thinker trace.

This analysis is the behavioral bridge between the real/imaginary-step analyses
(02--04) and the planning-trace neural analyses (06--08).  It asks whether a
raw decision-level planning trace improves held-out human-action prediction
beyond pre-outcome behavioral/visual history and the pre-search model state.

The script reads only:

* ``00_brain_cache/.../core_source/real_step_core_features.csv.gz``;
* ``00_brain_cache/.../core_source/imaginary_event_core_features.csv.gz``;
* the actor-row visual descriptors produced by 04 v2.

It never reads the HRF/TR planning arrays.  The 29 x 12 raw token is rebuilt
with the exact contract in :mod:`planning_trace_common`; all 11 pre-HRF hashes
must match the planning-trace *metadata only* before analysis can proceed.

Primary models are deliberately parallel after the policy anchor::

    A1 controls
    A2 A1 + pre-search policy ILR + Q0 incumbent margin/availability
    A3 A2 + interpretable Q-revision/evidence/geometry summaries
    A4 A2 + mean of the complete 29-event token sequence
    A5 A2 + event-position-preserving 29 x 12 concatenation

``A5 - A4`` is saved as a descriptive predictive contrast only.  Because the
representations differ in dimension and A5 contains pooled content, it is not
an order-specific test.

Outcomes are exact six-way human action, NOOP -> active, and active -> NOOP.
All preprocessing and regularization selection are nested inside outer
leave-one-run-out (LORO) folds.  Training uses inverse-run-size weights and
both inner selection and outer effects use equal-run loss averages.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import helmert
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

import planning_trace_common as TRACE


SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_VERSION = "v4_raw29_nested_loro_final_grid_converged"
N_ACTIONS = 6
EPS = 1e-12
SEED = 20260817

DEFAULT_CORE_ROOT = (
    SCRIPT_DIR / "outputs" / "00_brain_cache" / "time_aligned_cache"
    / "v4_native_time_hrf"
)
DEFAULT_VISUAL_ROOT = (
    SCRIPT_DIR / "outputs" / "04_search_evidence_geometry_pathway_v2"
)
DEFAULT_TRACE_METADATA_ROOT = (
    SCRIPT_DIR / "outputs" / "00_brain_cache" / "planning_trace_cache"
    / TRACE.TRACE_REPRESENTATION_VERSION
)
DEFAULT_OUTPUT_ROOT = (
    SCRIPT_DIR / "outputs" / "05_planning_trace_behavioral_validation"
)

OUTCOMES: Tuple[str, ...] = (
    "exact_action",
    "noop_to_active",
    "active_to_noop",
)
MODEL_ORDER: Tuple[str, ...] = (
    "A1_controls",
    "A2_policy",
    "A3_summary",
    "A4_mean29",
    "A5_concat29",
)
MODEL_LABELS = {
    "A1_controls": "A1\ncontrols",
    "A2_policy": "A2\npolicy/Q0",
    "A3_summary": "A3\nsummary",
    "A4_mean29": "A4\nmean29",
    "A5_concat29": "A5\nconcat29",
}
OUTCOME_LABELS = {
    "exact_action": "Exact human action",
    "noop_to_active": "NOOP -> active",
    "active_to_noop": "Active -> NOOP",
}

POLICY_EFFECTS: Tuple[str, ...] = tuple(
    f"{outcome}::A1_to_A2" for outcome in OUTCOMES
)
PLANNING_EFFECTS: Tuple[str, ...] = tuple(
    f"{outcome}::A2_to_{model.split('_', 1)[0]}"
    for outcome in OUTCOMES
    for model in ("A3_summary", "A4_mean29", "A5_concat29")
)
PRIMARY_EFFECTS: Tuple[str, ...] = POLICY_EFFECTS + PLANNING_EFFECTS

# Every candidate is refit in each inner group fold. The rejected pre-inference
# three-point grid truncated A5 at its low endpoint and compact models at its
# high endpoint. This final common grid fixes a broad low-C range and a long
# high-C guard tail before inference, including rare model/outcome cells whose
# inner loss can remain nearly flat at weak regularization. A pre-inference
# exact-split diagnostic established that the difficult C=300 and C=1000 fits
# converge from a cold start below 1,000 iterations; 2,000 is the fixed guard.
DEFAULT_C_GRID: Tuple[float, ...] = (
    0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3,
    1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0,
)
INNER_GROUP_FOLDS = 2
LOGISTIC_MAX_ITER = 2000
LOGISTIC_TOL = 1e-4

COLORS = {
    "A1_controls": "#7f8c8d",
    "A2_policy": "#34495e",
    "A3_summary": "#d55e00",
    "A4_mean29": "#009e73",
    "A5_concat29": "#0072b2",
}
INK = "#24323d"
GRID = "#dfe6e9"


REAL_COLUMNS: Tuple[str, ...] = (
    "actor_row", "subject", "session", "block", "game", "run_label",
    "real_step", "real_step_id", "source_chunk", "incumbent_action",
    "action_before_incumbent", "previous_human_action", "human_action",
    "prior_reward", "prior_done", "bout_age", "run_progress",
    "noop_run_length_prev", "model_current_action_margin_M",
    "initial_root_q_incumbent_margin", "event_window_complete",
    "R_log", "revision_abs_mean", "signed_q_revision_mean",
    "directional_q_revision_mean", "rollout_evidence_slope",
    "rollout_evidence_all29_mean", "rollout_evidence_all29_max",
    "rollout_evidence_all29_positive_fraction", "search_breadth",
    "search_mean_bout_terminal_depth", "search_deep_event_fraction",
    "path_revisit_fraction", "reset_count", "n_bouts",
    *(f"model_root_p_{action}" for action in range(N_ACTIONS)),
)

SUMMARY_COLUMNS: Tuple[str, ...] = (
    "R_log_filled0", "revision_available", "signed_q_revision_mean_filled0",
    "directional_q_revision_mean_filled0", "rollout_evidence_slope_filled0",
    "rollout_evidence_all29_mean", "rollout_evidence_all29_max",
    "rollout_evidence_all29_positive_fraction", "search_breadth",
    "search_mean_bout_terminal_depth", "search_deep_event_fraction",
    "path_revisit_fraction", "reset_count", "n_bouts",
)

CONTROL_FEATURE_NAMES: Tuple[str, ...] = (
    "prior_reward", "prior_done", "log1p_bout_age", "log1p_bout_age_sq",
    "run_progress", "run_progress_sq", "log1p_noop_run_length_prev",
    *(f"incumbent_{action}" for action in range(N_ACTIONS)),
    *(f"pre_incumbent_{action}" for action in range(N_ACTIONS)),
    *(f"visual_state_{index:02d}" for index in range(40)),
)
POLICY_FEATURE_NAMES: Tuple[str, ...] = (
    *(f"policy_ilr_{index + 1}" for index in range(N_ACTIONS - 1)),
    "Q0_incumbent_margin_filled0", "Q0_margin_available",
)
MODEL_BLOCK_ORDER: Dict[str, Tuple[str, ...]] = {
    "A1_controls": ("controls",),
    "A2_policy": ("controls", "policy"),
    "A3_summary": ("controls", "policy", "summary"),
    "A4_mean29": ("controls", "policy", "mean29"),
    "A5_concat29": ("controls", "policy", "concat29"),
}


def canonical_block_feature_names() -> Dict[str, List[str]]:
    return {
        "controls": list(CONTROL_FEATURE_NAMES),
        "policy": list(POLICY_FEATURE_NAMES),
        "summary": list(SUMMARY_COLUMNS),
        "mean29": list(TRACE.MEAN_COMPONENT_NAMES),
        "concat29": list(TRACE.CONCAT_COMPONENT_NAMES),
    }


def canonical_model_feature_contract() -> Dict[str, List[str]]:
    blocks = canonical_block_feature_names()
    return {
        model: [feature for block in MODEL_BLOCK_ORDER[model] for feature in blocks[block]]
        for model in MODEL_ORDER
    }


def json_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def analysis_contract(C_grid: Sequence[float]) -> Dict[str, object]:
    grid = [float(value) for value in C_grid]
    if not grid or any(not np.isfinite(value) or value <= 0 for value in grid):
        raise ValueError(f"Invalid C grid: {grid}")
    if len(set(grid)) != len(grid) or any(
        right <= left for left, right in zip(grid[:-1], grid[1:])
    ):
        raise ValueError(f"C grid must be strictly increasing and unique: {grid}")
    feature_contract = canonical_model_feature_contract()
    payload: Dict[str, object] = {
        "analysis_version": ANALYSIS_VERSION,
        "outcomes": list(OUTCOMES), "models": list(MODEL_ORDER),
        "model_feature_contract_sha256": json_sha256(feature_contract),
        "trace_representation_version": TRACE.TRACE_REPRESENTATION_VERSION,
        "C_grid": grid,
        "C_grid_guard_endpoints": [grid[0], grid[-1]],
        "C_tie_policy": "minimum inner NLL, then smaller C",
        "inner_group_folds": INNER_GROUP_FOLDS,
        "estimator": "LogisticRegression(lbfgs,L2,cold_start,max_iter=2000,tol=1e-4)",
        "convergence_policy": "cold start for every fit; fail closed if n_iter reaches 2000; record all warnings",
        "primary_subject_estimand": "equal mean of 11 held-out-run Delta NLL",
        "class_weight": None,
    }
    payload["contract_sha256"] = json_sha256(payload)
    return payload


def assert_script_unchanged(snapshot: Mapping[str, object]) -> None:
    current = file_signature(Path(__file__))
    if any(current.get(key) != snapshot.get(key) for key in ("size", "mtime_ns", "sha256")):
        raise RuntimeError("05 script changed during analysis; refusing to write a complete manifest")


def verify_recorded_signature(recorded: Mapping[str, object], label: str) -> Dict[str, object]:
    path = Path(str(recorded.get("path", "")))
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    current = file_signature(path)
    if any(current.get(key) != recorded.get(key) for key in ("size", "mtime_ns", "sha256")):
        raise RuntimeError(f"Stale or changed {label}: {path}")
    return current


@dataclass
class SubjectDesign:
    subject: int
    game: int
    metadata: pd.DataFrame
    blocks: Dict[str, np.ndarray]
    feature_names: Dict[str, List[str]]
    source_audit: Dict[str, Any]


@dataclass
class Preprocessor:
    median: np.ndarray
    center: np.ndarray
    scale: np.ndarray


def log(message: str) -> None:
    print(f"[05] {message}", flush=True)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_signature(path: Path) -> Dict[str, object]:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256_file(resolved),
    }


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def stable_seed(*parts: object) -> int:
    token = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:4], "little")


def actor_key_hash(frame: pd.DataFrame) -> str:
    columns = [
        "run_label", "real_step", "real_step_id", "source_chunk", "session",
        "block", "incumbent_action", "human_action",
    ]
    hashed = pd.util.hash_pandas_object(frame[columns], index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _raw_trace_hash(
    actor_rows: np.ndarray,
    timestamps: np.ndarray,
    concat29: np.ndarray,
    mean29: np.ndarray,
) -> str:
    # Exact implementation used by planning_trace_common row_provenance.
    # Actor rows and decision times have their own hashes in that contract.
    del actor_rows, timestamps
    return TRACE._array_sha256(concat29, mean29)


def load_raw_trace(
    event_path: Path,
    *,
    trace_metadata_dir: Optional[Path],
    chunk_rows: int = 580_000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Stream the event CSV and rebuild complete raw 29-event windows."""
    if trace_metadata_dir is None or not Path(trace_metadata_dir).is_dir():
        raise FileNotFoundError(
            "Canonical 05 completion requires the per-run planning-trace metadata directory: "
            f"{trace_metadata_dir}"
        )
    current_event_signature = file_signature(event_path)
    current_trace_common_sha256 = sha256_file(Path(TRACE.__file__).resolve())
    pieces: Dict[str, List[np.ndarray]] = {
        "label": [], "actor": [], "time": [], "concat": [], "mean": [],
    }
    carry: Optional[pd.DataFrame] = None
    source_rows = 0
    for chunk in pd.read_csv(
        event_path,
        usecols=list(TRACE.SOURCE_COLUMNS),
        dtype=TRACE.SOURCE_DTYPES,
        chunksize=chunk_rows,
        low_memory=False,
    ):
        source_rows += len(chunk)
        if carry is not None:
            chunk = pd.concat((carry, chunk), ignore_index=True)
        last_actor = int(chunk["actor_row"].iloc[-1])
        complete = chunk["actor_row"].to_numpy(dtype=np.int64) != last_actor
        ready = chunk.loc[complete]
        carry = chunk.loc[~complete].copy()
        if not ready.empty:
            batch = TRACE._rows_to_window_batch(ready)
            pieces["label"].append(batch.run_labels)
            pieces["actor"].append(batch.actor_rows)
            pieces["time"].append(batch.timestamps_s)
            pieces["concat"].append(batch.concat29)
            pieces["mean"].append(batch.mean29)
    if carry is None or carry.empty:
        raise RuntimeError(f"No event rows in {event_path}")
    batch = TRACE._rows_to_window_batch(carry)
    pieces["label"].append(batch.run_labels)
    pieces["actor"].append(batch.actor_rows)
    pieces["time"].append(batch.timestamps_s)
    pieces["concat"].append(batch.concat29)
    pieces["mean"].append(batch.mean29)

    labels = np.concatenate(pieces["label"]).astype(str)
    actor = np.concatenate(pieces["actor"]).astype(np.int64)
    timestamp = np.concatenate(pieces["time"]).astype(np.float64)
    concat29 = np.concatenate(pieces["concat"]).astype(np.float32)
    mean29 = np.concatenate(pieces["mean"]).astype(np.float32)
    if np.any(np.diff(actor) <= 0) or len(actor) * TRACE.N_EVENTS != source_rows:
        raise RuntimeError("Raw event stream did not form one ordered 29-event grid")

    hash_checks: List[Dict[str, object]] = []
    source_contract_tokens: List[str] = []
    if trace_metadata_dir is not None and trace_metadata_dir.is_dir():
        for run_label in pd.unique(labels):
            metadata_path = trace_metadata_dir / str(run_label) / TRACE.TRACE_METADATA_FILENAME
            if not metadata_path.is_file():
                raise FileNotFoundError(metadata_path)
            recorded = _read_json(metadata_path)
            if (
                int(recorded.get("schema_version", -1)) != TRACE.TRACE_SCHEMA_VERSION
                or str(recorded.get("representation_version", ""))
                != TRACE.TRACE_REPRESENTATION_VERSION
                or int(recorded.get("n_events", -1)) != TRACE.N_EVENTS
                or list(recorded.get("token_components", [])) != list(TRACE.TOKEN_COMPONENTS)
            ):
                raise RuntimeError(f"Trace representation contract mismatch: {metadata_path}")
            if str(recorded.get("planning_trace_common_sha256", "")) != current_trace_common_sha256:
                raise RuntimeError(f"planning_trace_common dependency mismatch: {metadata_path}")
            source_contract = recorded.get("source_contract", {})
            event_contract = (
                source_contract.get("imaginary_event_core", {})
                if isinstance(source_contract, Mapping) else {}
            )
            if str(event_contract.get("sha256", "")) != current_event_signature["sha256"]:
                raise RuntimeError(f"Trace metadata event-source SHA mismatch: {metadata_path}")
            source_contract_token = str(recorded.get("source_contract_sha256", ""))
            if not source_contract_token:
                raise RuntimeError(f"Trace metadata lacks source-contract token: {metadata_path}")
            source_contract_tokens.append(source_contract_token)
            mask = labels == run_label
            observed_hash = _raw_trace_hash(
                actor[mask], timestamp[mask], concat29[mask], mean29[mask]
            )
            row_provenance = recorded.get("row_provenance", {})
            if not isinstance(row_provenance, Mapping):
                raise RuntimeError(f"Trace metadata lacks row_provenance: {metadata_path}")
            expected_hash = str(row_provenance.get("raw_representation_sha256", ""))
            observed_actor_hash = TRACE._array_sha256(actor[mask])
            observed_time_hash = TRACE._array_sha256(timestamp[mask])
            if observed_actor_hash != str(row_provenance.get("actor_row_sha256", "")):
                raise RuntimeError(f"Actor-row hash mismatch for {run_label}")
            if observed_time_hash != str(row_provenance.get("decision_time_sha256", "")):
                raise RuntimeError(f"Decision-time hash mismatch for {run_label}")
            if observed_hash != expected_hash:
                raise RuntimeError(
                    f"Raw trace hash mismatch for {run_label}: "
                    f"{observed_hash} != {expected_hash}"
                )
            hash_checks.append(
                {
                    "run_label": str(run_label),
                    "metadata": file_signature(metadata_path),
                    "raw_representation_sha256": observed_hash,
                    "actor_row_sha256": observed_actor_hash,
                    "decision_time_sha256": observed_time_hash,
                    "planning_trace_common_sha256": current_trace_common_sha256,
                    "source_contract_sha256": source_contract_token,
                    "complete_windows": int(np.sum(mask)),
                    "HRF_TR_array_read": False,
                }
            )
        if len(set(source_contract_tokens)) != 1:
            raise RuntimeError("The 11 trace metadata files do not share one source contract")
    if len(hash_checks) != 11 or len({item["run_label"] for item in hash_checks}) != 11:
        raise RuntimeError(
            f"Canonical 05 requires exactly 11 per-run trace metadata checks; got {len(hash_checks)}"
        )
    audit = {
        "event_source_rows": int(source_rows),
        "complete_windows": int(len(actor)),
        "n_events_per_window": TRACE.N_EVENTS,
        "token_components": list(TRACE.TOKEN_COMPONENTS),
        "concat_dimension": int(concat29.shape[1]),
        "mean_dimension": int(mean29.shape[1]),
        "raw_hash_checks": hash_checks,
        "planning_trace_common": file_signature(Path(TRACE.__file__).resolve()),
        "event_source": current_event_signature,
        "source_contract_sha256": source_contract_tokens[0] if source_contract_tokens else None,
        "planning_trace_npz_read": False,
        "HRF_TR_trace_used": False,
    }
    return actor, mean29, concat29, audit


def policy_ilr(probabilities: np.ndarray) -> np.ndarray:
    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != N_ACTIONS:
        raise ValueError(f"Invalid policy shape: {matrix.shape}")
    total = np.sum(matrix, axis=1)
    if (
        not np.all(np.isfinite(matrix))
        or np.any(matrix < 0)
        or np.any(total <= EPS)
    ):
        raise ValueError("Pre-search policy contains invalid probability rows")
    normalized = matrix / total[:, None]
    logp = np.log(np.clip(normalized, EPS, 1.0))
    clr = logp - np.mean(logp, axis=1, keepdims=True)
    basis = helmert(N_ACTIONS, full=False)  # 5 x 6 orthonormal contrasts
    return np.asarray(clr @ basis.T, dtype=np.float32)


def action_one_hot(values: np.ndarray, prefix: str) -> Tuple[np.ndarray, List[str]]:
    action = np.asarray(values, dtype=int)
    result = np.zeros((len(action), N_ACTIONS), dtype=np.float32)
    valid = (action >= 0) & (action < N_ACTIONS)
    result[np.flatnonzero(valid), action[valid]] = 1.0
    return result, [f"{prefix}_{index}" for index in range(N_ACTIONS)]


def load_subject_design(
    subject: int,
    game: int,
    *,
    core_root: Path,
    visual_root: Path,
    trace_metadata_root: Optional[Path],
) -> SubjectDesign:
    tag = f"sub{subject:03d}_game{game}"
    source_dir = Path(core_root) / tag / "core_source"
    real_path = source_dir / "real_step_core_features.csv.gz"
    event_path = source_dir / "imaginary_event_core_features.csv.gz"
    source_manifest_path = source_dir / "source_manifest.json"
    visual_dir = Path(visual_root) / tag
    visual_path = visual_dir / "cache" / "visual_state_controls_v2.npz"
    visual_qa_path = visual_dir / "qa" / "visual_state_controls.json"
    visual_alignment_path = visual_dir / "tables" / "behavior_frame_alignment.csv.gz"
    for path in (
        real_path, event_path, source_manifest_path, visual_path,
        visual_qa_path, visual_alignment_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    source_manifest = _read_json(source_manifest_path)
    if (
        int(source_manifest.get("subject", -1)) != subject
        or int(source_manifest.get("game", -1)) != game
        or int(source_manifest.get("schema_version", -1)) != 4
    ):
        raise RuntimeError(f"Invalid core source manifest: {source_manifest_path}")
    real = pd.read_csv(real_path, usecols=list(REAL_COLUMNS), low_memory=False)
    if len(real) != int(source_manifest.get("full_actor_rows", -1)):
        raise RuntimeError("Real-step row count disagrees with source manifest")
    actor_full = pd.to_numeric(real["actor_row"], errors="raise").to_numpy(dtype=int)
    if not np.array_equal(actor_full, np.arange(len(real), dtype=int)):
        raise RuntimeError("Real-step actor spine is not actor_row=0..N-1")

    with np.load(visual_path, allow_pickle=False) as archive:
        visual = np.asarray(archive["descriptor"], dtype=np.float32)
        stored_actor_hash = str(np.asarray(archive["actor_key_hash"]).item())
        visual_formula_hash = str(np.asarray(archive["descriptor_formula_hash"]).item())
    observed_actor_hash = actor_key_hash(real)
    if visual.shape != (len(real), 40) or stored_actor_hash != observed_actor_hash:
        raise RuntimeError("Visual descriptor does not match the real-step actor spine")
    visual_qa = _read_json(visual_qa_path)
    if (
        str(visual_qa.get("descriptor_formula_hash", "")) != visual_formula_hash
        or float(visual_qa.get("action_pair_match_fraction", 0.0)) < 0.999999
    ):
        raise RuntimeError("Visual descriptor QA contract failed")
    alignment = pd.read_csv(
        visual_alignment_path,
        usecols=["actor_row", "run_label", "real_step", "real_step_id",
                 "previous_action", "current_action", "action_pair_match"],
        low_memory=False,
    )
    if alignment["actor_row"].duplicated().any() or not alignment["action_pair_match"].eq(1).all():
        raise RuntimeError("Visual alignment is not one-to-one/action-exact")

    metadata_dir = None
    if trace_metadata_root is not None:
        metadata_dir = Path(trace_metadata_root) / tag
    trace_actor, mean29, concat29, trace_audit = load_raw_trace(
        event_path, trace_metadata_dir=metadata_dir
    )
    expected_trace_actor = real.loc[
        pd.to_numeric(real["event_window_complete"], errors="coerce").eq(1),
        "actor_row",
    ].to_numpy(dtype=int)
    if not np.array_equal(trace_actor, expected_trace_actor):
        raise RuntimeError("Raw trace actor rows differ from complete real-step windows")

    selected = real.set_index("actor_row", drop=False).loc[trace_actor].reset_index(drop=True)
    aligned_actor = alignment["actor_row"].to_numpy(dtype=int)
    if not np.all(np.isin(trace_actor, aligned_actor)):
        raise RuntimeError("Complete trace rows are not covered by visual alignment")
    visual_selected = visual[trace_actor]
    if not np.all(np.isfinite(visual_selected)):
        raise RuntimeError("Complete trace rows contain non-finite visual controls")

    incumbent = pd.to_numeric(selected["incumbent_action"], errors="coerce").fillna(-1).to_numpy(dtype=int)
    previous = pd.to_numeric(selected["action_before_incumbent"], errors="coerce").fillna(-1).to_numpy(dtype=int)
    incumbent_onehot, incumbent_names = action_one_hot(incumbent, "incumbent")
    previous_onehot, previous_names = action_one_hot(previous, "pre_incumbent")
    numeric_b0 = np.column_stack(
        (
            pd.to_numeric(selected["prior_reward"], errors="coerce").to_numpy(float),
            pd.to_numeric(selected["prior_done"], errors="coerce").to_numpy(float),
            np.log1p(pd.to_numeric(selected["bout_age"], errors="coerce").clip(lower=0).to_numpy(float)),
            np.log1p(pd.to_numeric(selected["bout_age"], errors="coerce").clip(lower=0).to_numpy(float)) ** 2,
            pd.to_numeric(selected["run_progress"], errors="coerce").to_numpy(float),
            pd.to_numeric(selected["run_progress"], errors="coerce").to_numpy(float) ** 2,
            np.log1p(pd.to_numeric(selected["noop_run_length_prev"], errors="coerce").clip(lower=0).to_numpy(float)),
        )
    ).astype(np.float32)
    b0_names = [
        "prior_reward", "prior_done", "log1p_bout_age", "log1p_bout_age_sq",
        "run_progress", "run_progress_sq", "log1p_noop_run_length_prev",
        *incumbent_names, *previous_names,
        *(f"visual_state_{index:02d}" for index in range(visual_selected.shape[1])),
    ]
    b0 = np.column_stack((numeric_b0, incumbent_onehot, previous_onehot, visual_selected)).astype(np.float32)

    root_policy = selected[
        [f"model_root_p_{action}" for action in range(N_ACTIONS)]
    ].to_numpy(dtype=float)
    ilr = policy_ilr(root_policy)
    q0_margin = pd.to_numeric(
        selected["initial_root_q_incumbent_margin"], errors="coerce"
    ).to_numpy(dtype=float)
    q0_available = np.isfinite(q0_margin).astype(np.float32)
    q0_margin = np.where(np.isfinite(q0_margin), q0_margin, 0.0).astype(np.float32)
    policy_block = np.column_stack((ilr, q0_margin, q0_available)).astype(np.float32)
    policy_names = [*(f"policy_ilr_{i + 1}" for i in range(N_ACTIONS - 1)),
                    "Q0_incumbent_margin_filled0", "Q0_margin_available"]

    summary = selected.copy()
    revision_available = np.isfinite(
        pd.to_numeric(summary["revision_abs_mean"], errors="coerce")
    )
    summary["revision_available"] = revision_available.astype(float)
    for source, destination in (
        ("R_log", "R_log_filled0"),
        ("signed_q_revision_mean", "signed_q_revision_mean_filled0"),
        ("directional_q_revision_mean", "directional_q_revision_mean_filled0"),
        ("rollout_evidence_slope", "rollout_evidence_slope_filled0"),
    ):
        value = pd.to_numeric(summary[source], errors="coerce")
        summary[destination] = value.where(np.isfinite(value), 0.0)
    summary_block = summary[list(SUMMARY_COLUMNS)].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float32)

    metadata = selected[
        ["actor_row", "subject", "game", "run_label", "session", "block",
         "real_step", "real_step_id", "incumbent_action", "human_action",
         "model_current_action_margin_M"]
    ].copy()
    metadata["trace_row"] = np.arange(len(metadata), dtype=int)
    valid_action = (
        pd.to_numeric(metadata["incumbent_action"], errors="coerce").between(0, N_ACTIONS - 1)
        & pd.to_numeric(metadata["human_action"], errors="coerce").between(0, N_ACTIONS - 1)
    )
    if not valid_action.all():
        raise RuntimeError("Complete trace contains invalid incumbent/human action")
    if metadata["run_label"].nunique() != int(source_manifest.get("n_runs", -1)):
        raise RuntimeError("Complete trace does not cover every recorded run")

    blocks = {
        "controls": b0,
        "policy": policy_block,
        "summary": summary_block,
        "mean29": mean29,
        "concat29": concat29,
    }
    feature_names = {
        "controls": b0_names,
        "policy": policy_names,
        "summary": list(SUMMARY_COLUMNS),
        "mean29": list(TRACE.MEAN_COMPONENT_NAMES),
        "concat29": list(TRACE.CONCAT_COMPONENT_NAMES),
    }
    expected_feature_names = canonical_block_feature_names()
    if feature_names != expected_feature_names:
        raise RuntimeError("Constructed feature names differ from the frozen analysis contract")
    for name, values in blocks.items():
        if values.shape[0] != len(metadata):
            raise RuntimeError(f"Block {name} row mismatch")
        if values.shape[1] != len(feature_names[name]):
            raise RuntimeError(f"Block {name} column mismatch")

    source_audit = {
        "subject": subject,
        "game": game,
        "core_manifest": file_signature(source_manifest_path),
        "core_manifest_source_token": source_manifest.get("source_token"),
        "real_step_core": file_signature(real_path),
        "imaginary_event_core": file_signature(event_path),
        "visual_descriptor": file_signature(visual_path),
        "visual_qa": file_signature(visual_qa_path),
        "visual_alignment": file_signature(visual_alignment_path),
        "visual_actor_key_hash": observed_actor_hash,
        "visual_descriptor_formula_hash": visual_formula_hash,
        "raw_trace": trace_audit,
        "full_actor_rows": int(len(real)),
        "analysis_rows": int(len(metadata)),
        "runs": int(metadata["run_label"].nunique()),
    }
    return SubjectDesign(subject, game, metadata, blocks, feature_names, source_audit)


def model_matrix(design: SubjectDesign, model: str) -> Tuple[np.ndarray, List[str]]:
    if model not in MODEL_BLOCK_ORDER:
        raise KeyError(model)
    values = np.column_stack(
        [design.blocks[name] for name in MODEL_BLOCK_ORDER[model]]
    ).astype(np.float32)
    names = [
        feature for name in MODEL_BLOCK_ORDER[model]
        for feature in design.feature_names[name]
    ]
    if names != canonical_model_feature_contract()[model]:
        raise RuntimeError(f"Model {model} violates the frozen feature contract")
    return values, names


def outcome_definition(
    metadata: pd.DataFrame, outcome: str
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    incumbent = pd.to_numeric(metadata["incumbent_action"], errors="raise").to_numpy(dtype=int)
    human = pd.to_numeric(metadata["human_action"], errors="raise").to_numpy(dtype=int)
    valid = (
        (incumbent >= 0) & (incumbent < N_ACTIONS)
        & (human >= 0) & (human < N_ACTIONS)
    )
    if outcome == "exact_action":
        risk = valid
        target = human
        details = {"risk_definition": "all valid complete-trace decisions", "classes": list(range(N_ACTIONS))}
    elif outcome == "noop_to_active":
        risk = valid & (incumbent == 0)
        target = (human > 0).astype(int)
        details = {"risk_definition": "incumbent=NOOP; target human_action>0", "classes": [0, 1]}
    elif outcome == "active_to_noop":
        competing = valid & (incumbent > 0) & (human > 0) & (human != incumbent)
        risk = valid & (incumbent > 0) & ~competing
        target = (human == 0).astype(int)
        details = {
            "risk_definition": "incumbent active; compare stay-same-active vs NOOP",
            "classes": [0, 1],
            "competing_active_to_other_active_excluded": int(np.sum(competing)),
        }
    else:
        raise KeyError(outcome)
    details.update({"risk_rows": int(np.sum(risk)), "event_rows": int(np.sum(target[risk] != 0))})
    return risk, target, details


def equal_run_weights(groups: np.ndarray) -> np.ndarray:
    labels, inverse, counts = np.unique(np.asarray(groups).astype(str), return_inverse=True, return_counts=True)
    if len(labels) < 2:
        raise ValueError("Equal-run weighting requires at least two runs")
    weight = 1.0 / counts[inverse].astype(float)
    weight *= len(weight) / np.sum(weight)
    totals = np.asarray([np.sum(weight[inverse == i]) for i in range(len(labels))])
    if not np.allclose(totals, totals[0], rtol=1e-10, atol=1e-10):
        raise AssertionError("Inverse-run weights do not equalize run totals")
    return weight


def fit_preprocessor(x: np.ndarray, groups: np.ndarray) -> Preprocessor:
    values = np.asarray(x, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(values, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    filled = np.where(np.isfinite(values), values, median[None, :])
    weight = equal_run_weights(groups)
    center = np.average(filled, axis=0, weights=weight)
    variance = np.average((filled - center[None, :]) ** 2, axis=0, weights=weight)
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale = np.where(scale > 1e-8, scale, 1.0)
    return Preprocessor(median.astype(np.float64), center.astype(np.float64), scale.astype(np.float64))


def transform_preprocessor(preprocessor: Preprocessor, x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64)
    filled = np.where(np.isfinite(values), values, preprocessor.median[None, :])
    transformed = (filled - preprocessor.center[None, :]) / preprocessor.scale[None, :]
    if not np.all(np.isfinite(transformed)):
        raise RuntimeError("Non-finite value after train-only preprocessing")
    return np.asarray(transformed, dtype=np.float32)


def fit_logistic_l2(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    C: float,
    seed: int,
) -> Tuple[LogisticRegression, Dict[str, object]]:
    observed_classes = np.unique(np.asarray(y, dtype=int))
    if len(observed_classes) < 2:
        raise RuntimeError(f"Training fold has fewer than two classes: {observed_classes.tolist()}")
    # random_state is recorded for a stable contract although deterministic
    # dense L-BFGS itself does not use stochastic updates.
    model = LogisticRegression(
        penalty="l2",
        C=float(C),
        fit_intercept=True,
        solver="lbfgs",
        max_iter=LOGISTIC_MAX_ITER,
        tol=LOGISTIC_TOL,
        random_state=int(seed),
    )
    weight = equal_run_weights(groups)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        model.fit(x, y, sample_weight=weight)
    elapsed = time.perf_counter() - started
    n_iter = int(np.max(np.atleast_1d(model.n_iter_)))
    converged = n_iter < LOGISTIC_MAX_ITER
    if not converged:
        raise RuntimeError(
            f"L-BFGS did not converge within {LOGISTIC_MAX_ITER} iterations (C={C})"
        )
    return model, {
        "n_iter": n_iter,
        "converged": converged,
        "elapsed_s": float(elapsed),
        "warnings": [str(item.message) for item in captured],
    }


def aligned_probability(
    model: Any, x: np.ndarray, target_classes: Sequence[int]
) -> np.ndarray:
    raw = np.asarray(model.predict_proba(x), dtype=float)
    classes = np.asarray(model.classes_, dtype=int)
    target = np.asarray(target_classes, dtype=int)
    output = np.full((len(x), len(target)), EPS, dtype=float)
    for source_index, value in enumerate(classes):
        matches = np.flatnonzero(target == value)
        if len(matches) == 1:
            output[:, matches[0]] = raw[:, source_index]
    output = np.clip(output, EPS, 1.0)
    output /= np.sum(output, axis=1, keepdims=True)
    return output


def row_metrics(
    y: np.ndarray, probability: np.ndarray, target_classes: Sequence[int]
) -> Tuple[np.ndarray, np.ndarray]:
    target = np.asarray(target_classes, dtype=int)
    outcome = np.asarray(y, dtype=int)
    lookup = {int(value): index for index, value in enumerate(target)}
    index = np.asarray([lookup[int(value)] for value in outcome], dtype=int)
    p = np.clip(probability, EPS, 1.0)
    nll = -np.log(p[np.arange(len(outcome)), index])
    truth = np.eye(len(target), dtype=float)[index]
    brier = np.sum((p - truth) ** 2, axis=1)
    return nll, brier


def equal_run_mean(values: np.ndarray, groups: np.ndarray) -> float:
    frame = pd.DataFrame({"value": np.asarray(values, dtype=float), "group": np.asarray(groups).astype(str)})
    return float(frame.groupby("group", sort=False, observed=True)["value"].mean().mean())


def tune_C(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    target_classes: Sequence[int],
    *,
    C_grid: Sequence[float],
    seed_parts: Sequence[object],
) -> Tuple[float, List[Dict[str, object]]]:
    unique_groups = np.unique(groups.astype(str))
    n_splits = min(INNER_GROUP_FOLDS, len(unique_groups))
    if n_splits < 2:
        raise RuntimeError("Nested tuning requires at least two inner runs")
    splitter = GroupKFold(n_splits=n_splits)
    records: List[Dict[str, object]] = []
    scores: Dict[float, List[float]] = {float(C): [] for C in C_grid}
    for split_index, (train_index, validation_index) in enumerate(
        splitter.split(x, y, groups=groups)
    ):
        train_groups = groups[train_index]
        validation_groups = groups[validation_index]
        preprocessor = fit_preprocessor(x[train_index], train_groups)
        x_train = transform_preprocessor(preprocessor, x[train_index])
        x_validation = transform_preprocessor(preprocessor, x[validation_index])
        for C in C_grid:
            model, diagnostics = fit_logistic_l2(
                x_train, y[train_index], train_groups,
                C=float(C),
                seed=stable_seed(SEED, *seed_parts, "inner", split_index),
            )
            probability = aligned_probability(model, x_validation, target_classes)
            loss, _ = row_metrics(y[validation_index], probability, target_classes)
            score = equal_run_mean(loss, validation_groups)
            scores[float(C)].append(score)
            records.append(
                {
                    "inner_fold": split_index,
                    "C": float(C),
                    "validation_run_equal_nll": score,
                    "n_train": int(len(train_index)),
                    "n_validation": int(len(validation_index)),
                    "n_train_runs": int(len(np.unique(train_groups))),
                    "n_validation_runs": int(len(np.unique(validation_groups))),
                    **diagnostics,
                }
            )
    mean_score = {C: float(np.mean(value)) for C, value in scores.items()}
    # Prefer stronger regularization (smaller C) on an exact score tie.
    selected = min(mean_score, key=lambda C: (mean_score[C], C))
    for record in records:
        record["inner_mean_nll_for_C"] = mean_score[float(record["C"])]
        record["selected_C"] = float(selected)
    return float(selected), records


def calibration_bins(
    outcome: str,
    y: np.ndarray,
    probability: np.ndarray,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    if outcome == "exact_action":
        confidence = np.max(probability, axis=1)
        observed = (np.argmax(probability, axis=1) == np.asarray(y, dtype=int)).astype(float)
        kind = "top_class_confidence_vs_accuracy"
    else:
        confidence = probability[:, 1]
        observed = np.asarray(y, dtype=float)
        kind = "event_probability_vs_event_rate"
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    index = np.clip(np.digitize(confidence, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for bin_index in range(n_bins):
        mask = index == bin_index
        if not np.any(mask):
            continue
        rows.append(
            {
                "calibration_kind": kind,
                "bin": bin_index,
                "bin_left": float(edges[bin_index]),
                "bin_right": float(edges[bin_index + 1]),
                "predicted_mean": float(np.mean(confidence[mask])),
                "observed_mean": float(np.mean(observed[mask])),
                "n": int(np.sum(mask)),
            }
        )
    return pd.DataFrame(rows)


def fit_subject_loro(
    design: SubjectDesign,
    *,
    C_grid: Sequence[float],
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.DataFrame, Dict[str, object],
]:
    metadata = design.metadata
    run_all = metadata["run_label"].astype(str).to_numpy()
    model_cache: Dict[str, Tuple[np.ndarray, List[str]]] = {
        model: model_matrix(design, model) for model in MODEL_ORDER
    }
    fold_metrics: List[Dict[str, object]] = []
    tuning_rows: List[Dict[str, object]] = []
    calibration_rows: List[pd.DataFrame] = []
    outcome_qa: Dict[str, object] = {}
    for outcome in OUTCOMES:
        risk, target_all, definition = outcome_definition(metadata, outcome)
        rows = np.flatnonzero(risk)
        y = target_all[rows]
        groups = run_all[rows]
        runs = pd.unique(groups).tolist()
        if len(runs) != 11:
            raise RuntimeError(f"{outcome}: expected 11 LORO runs, found {len(runs)}")
        target_classes = list(range(N_ACTIONS)) if outcome == "exact_action" else [0, 1]
        outcome_qa[outcome] = {
            **definition,
            "runs": len(runs),
            "target_counts": {
                str(value): int(np.sum(y == value)) for value in target_classes
            },
            "identical_rows_across_models": True,
        }
        log(f"sub{design.subject:03d} game{design.game} {outcome}: {len(rows):,} rows")
        for held_out in runs:
            test_local = groups == held_out
            train_local = ~test_local
            if not np.any(test_local) or len(np.unique(groups[train_local])) != 10:
                raise RuntimeError(f"Invalid outer fold {outcome}/{held_out}")
            train_support = sorted(map(int, np.unique(y[train_local])))
            test_support = sorted(map(int, np.unique(y[test_local])))
            unseen_test_support = sorted(set(test_support).difference(train_support))
            train_counts = {
                str(value): int(np.sum(y[train_local] == value)) for value in target_classes
            }
            test_counts = {
                str(value): int(np.sum(y[test_local] == value)) for value in target_classes
            }
            for model_name in MODEL_ORDER:
                full_x, feature_names = model_cache[model_name]
                x = full_x[rows]
                selected_C, tuning = tune_C(
                    x[train_local], y[train_local], groups[train_local], target_classes,
                    C_grid=C_grid,
                    seed_parts=(design.subject, design.game, outcome, model_name, held_out),
                )
                for record in tuning:
                    tuning_rows.append(
                        {
                            "subject": design.subject, "game": design.game,
                            "outcome": outcome, "model": model_name,
                            "held_out_run": held_out, **record,
                        }
                    )
                preprocessor = fit_preprocessor(x[train_local], groups[train_local])
                x_train = transform_preprocessor(preprocessor, x[train_local])
                x_test = transform_preprocessor(preprocessor, x[test_local])
                fitted, diagnostics = fit_logistic_l2(
                    x_train, y[train_local], groups[train_local],
                    C=selected_C,
                    seed=stable_seed(
                        SEED, design.subject, design.game, outcome,
                        model_name, held_out, "outer",
                    ),
                )
                probability = aligned_probability(fitted, x_test, target_classes)
                nll, brier = row_metrics(y[test_local], probability, target_classes)
                fold_metrics.append(
                    {
                        "subject": design.subject,
                        "game": design.game,
                        "outcome": outcome,
                        "model": model_name,
                        "held_out_run": held_out,
                        "n_test": int(np.sum(test_local)),
                        "n_train": int(np.sum(train_local)),
                        "test_events": int(np.sum(y[test_local] != 0)),
                        "fixed_output_support": json.dumps(target_classes),
                        "train_class_support": json.dumps(train_support),
                        "test_class_support": json.dumps(test_support),
                        "unseen_test_classes": json.dumps(unseen_test_support),
                        "train_class_counts": json.dumps(train_counts, sort_keys=True),
                        "test_class_counts": json.dumps(test_counts, sort_keys=True),
                        "test_rows_dropped_for_class_support": 0,
                        "test_nll": float(np.mean(nll)),
                        "test_brier": float(np.mean(brier)),
                        "selected_C": selected_C,
                        "n_features": int(x.shape[1]),
                        **diagnostics,
                    }
                )
                calibration = calibration_bins(outcome, y[test_local], probability)
                calibration.insert(0, "held_out_run", held_out)
                calibration.insert(0, "model", model_name)
                calibration.insert(0, "outcome", outcome)
                calibration.insert(0, "game", design.game)
                calibration.insert(0, "subject", design.subject)
                calibration_rows.append(calibration)
                del x_train, x_test, fitted, probability

    metrics = pd.DataFrame(fold_metrics)
    expected_rows = len(OUTCOMES) * 11 * len(MODEL_ORDER)
    if len(metrics) != expected_rows or not np.all(np.isfinite(metrics["test_nll"])):
        raise RuntimeError(f"Incomplete/non-finite fold metrics: {len(metrics)} != {expected_rows}")
    for outcome in OUTCOMES:
        expected = int(outcome_qa[outcome]["risk_rows"])
        coverage = metrics.loc[metrics["outcome"].eq(outcome)].groupby(
            "model", sort=False, observed=True
        )["n_test"].sum()
        if set(coverage.index) != set(MODEL_ORDER) or not np.all(coverage.to_numpy() == expected):
            raise RuntimeError(f"{outcome}: OOF row coverage differs across models")
        outcome_qa[outcome]["oof_rows_per_model"] = {
            str(model): int(value) for model, value in coverage.items()
        }
        outcome_qa[outcome]["test_rows_dropped_for_class_support"] = 0
        outcome_qa[outcome]["fixed_output_support"] = (
            list(range(N_ACTIONS)) if outcome == "exact_action" else [0, 1]
        )
        unseen = metrics.loc[
            metrics["outcome"].eq(outcome),
            ["held_out_run", "unseen_test_classes"],
        ].drop_duplicates()
        outcome_qa[outcome]["folds_with_unseen_test_classes"] = [
            {"held_out_run": str(row.held_out_run), "unseen_test_classes": json.loads(row.unseen_test_classes)}
            for row in unseen.itertuples(index=False)
            if json.loads(row.unseen_test_classes)
        ]
    wide = metrics.pivot(
        index=["subject", "game", "outcome", "held_out_run"],
        columns="model", values="test_nll",
    ).reset_index()
    effect_rows: List[Dict[str, object]] = []
    for _, row in wide.iterrows():
        base = {name: row[name] for name in ("subject", "game", "outcome", "held_out_run")}
        effects = {
            "A1_to_A2": float(row["A1_controls"] - row["A2_policy"]),
            "A2_to_A3": float(row["A2_policy"] - row["A3_summary"]),
            "A2_to_A4": float(row["A2_policy"] - row["A4_mean29"]),
            "A2_to_A5": float(row["A2_policy"] - row["A5_concat29"]),
            "A4_to_A5_descriptive": float(row["A4_mean29"] - row["A5_concat29"]),
        }
        for comparison, value in effects.items():
            primary = comparison != "A4_to_A5_descriptive"
            family = "policy_anchor" if comparison == "A1_to_A2" else (
                "planning_primary" if primary else "descriptive_not_order"
            )
            effect_rows.append(
                {
                    **base,
                    "comparison": comparison,
                    "effect_id": f"{row['outcome']}::{comparison}",
                    "delta_nll_positive_is_improvement": value,
                    "family": family,
                    "primary": int(primary),
                }
            )
    effects = pd.DataFrame(effect_rows)
    subject_effects = effects.groupby(
        ["subject", "game", "outcome", "comparison", "effect_id", "family", "primary"],
        sort=False, observed=True, as_index=False,
    )["delta_nll_positive_is_improvement"].mean()
    # Sensitivity estimand: pool every OOF row rather than giving each run the
    # same weight. This is saved beside, but never replaces, the prespecified
    # equal-mean-of-11-runs subject effect above.
    pooled = metrics.assign(
        weighted_nll=metrics["test_nll"] * metrics["n_test"]
    ).groupby(
        ["subject", "game", "outcome", "model"],
        sort=False, observed=True, as_index=False,
    ).agg(weighted_nll=("weighted_nll", "sum"), n_rows=("n_test", "sum"))
    pooled["row_pooled_oof_nll"] = pooled["weighted_nll"] / pooled["n_rows"]
    pooled_wide = pooled.pivot(
        index=["subject", "game", "outcome"],
        columns="model", values="row_pooled_oof_nll",
    ).reset_index()
    pooled_effect_rows: List[Dict[str, object]] = []
    for _, row in pooled_wide.iterrows():
        comparisons = {
            "A1_to_A2": float(row["A1_controls"] - row["A2_policy"]),
            "A2_to_A3": float(row["A2_policy"] - row["A3_summary"]),
            "A2_to_A4": float(row["A2_policy"] - row["A4_mean29"]),
            "A2_to_A5": float(row["A2_policy"] - row["A5_concat29"]),
            "A4_to_A5_descriptive": float(row["A4_mean29"] - row["A5_concat29"]),
        }
        for comparison, value in comparisons.items():
            pooled_effect_rows.append(
                {
                    "subject": row["subject"], "game": row["game"],
                    "outcome": row["outcome"], "comparison": comparison,
                    "effect_id": f"{row['outcome']}::{comparison}",
                    "row_pooled_oof_delta_nll_sensitivity": value,
                }
            )
    subject_effects = subject_effects.merge(
        pd.DataFrame(pooled_effect_rows),
        on=["subject", "game", "outcome", "comparison", "effect_id"],
        how="left", validate="one_to_one",
    )
    calibration = pd.concat(calibration_rows, ignore_index=True)
    tuning = pd.DataFrame(tuning_rows)
    selected = tuning[
        ["outcome", "model", "held_out_run", "selected_C"]
    ].drop_duplicates()
    expected_selections = len(OUTCOMES) * len(MODEL_ORDER) * 11
    if (
        len(selected) != expected_selections
        or selected.groupby(
            ["outcome", "model", "held_out_run"], observed=True
        ).size().ne(1).any()
    ):
        raise RuntimeError(
            "Nested regularization selections are not unique and complete: "
            f"{len(selected)} != {expected_selections}"
        )
    grid = [float(value) for value in C_grid]
    selection_counts = {
        format(value, ".12g"): int(np.sum(selected["selected_C"].eq(value)))
        for value in grid
    }
    lower_count = int(np.sum(selected["selected_C"].eq(grid[0])))
    upper_count = int(np.sum(selected["selected_C"].eq(grid[-1])))
    qa = {
        "outcomes": outcome_qa,
        "outer_cv": "leave one complete run out",
        "inner_cv": f"GroupKFold({INNER_GROUP_FOLDS}) on outer-training runs",
        "inner_selection_metric": "equal-validation-run mean negative log loss",
        "training_weights": "inverse training-run row count; equal total weight per run",
        "preprocessing": "median imputation and equal-run weighted centering/scaling, fit on training rows only",
        "estimator": "dense LogisticRegression with L-BFGS and L2 penalty",
        "C_grid": list(map(float, C_grid)),
        "regularization_selection": {
            "selection_counts_by_C": selection_counts,
            "lower_endpoint": grid[0],
            "lower_endpoint_count": lower_count,
            "lower_endpoint_fraction": lower_count / expected_selections,
            "upper_endpoint": grid[-1],
            "upper_endpoint_count": upper_count,
            "upper_endpoint_fraction": upper_count / expected_selections,
            "total_selections": expected_selections,
        },
        "class_weight": None,
        "class_prevalence_policy": "natural observed prevalence; no outcome rebalancing",
        "same_rows_across_A1_A5_within_outcome": True,
        "primary_subject_estimand": "unweighted mean of the 11 held-out-run Delta NLL values",
        "sensitivity_subject_estimand": "row-pooled OOF Delta NLL; saved but not used for primary inference",
        "A3_A4_A5_are_parallel_extensions_of_A2": True,
        "A5_minus_A4_is_order_test": False,
        "model_features": {
            model: model_cache[model][1] for model in MODEL_ORDER
        },
        "M_in_A2": False,
        "M_policy_relationship_policy": "M is a deterministic focal contrast of P; retained only for descriptive sensitivity",
    }
    return metrics, effects, subject_effects, calibration, tuning, qa


def build_m_descriptive(design: SubjectDesign) -> pd.DataFrame:
    metadata = design.metadata.copy()
    margin = pd.to_numeric(metadata["model_current_action_margin_M"], errors="coerce")
    incumbent = pd.to_numeric(metadata["incumbent_action"], errors="coerce").to_numpy(dtype=int)
    human = pd.to_numeric(metadata["human_action"], errors="coerce").to_numpy(dtype=int)
    records: List[Dict[str, object]] = []
    definitions = {
        "exact_switch": ((incumbent >= 0) & (human >= 0), (human != incumbent).astype(int)),
        "noop_to_active": (incumbent == 0, (human > 0).astype(int)),
        "active_to_noop": (
            (incumbent > 0) & ((human == 0) | (human == incumbent)),
            (human == 0).astype(int),
        ),
    }
    edges = np.asarray([-np.inf, -1.0, -0.5, 0.0, 0.5, 1.0, np.inf])
    for run_label, run_index in metadata.groupby("run_label", sort=False).groups.items():
        index = np.asarray(list(run_index), dtype=int)
        run_margin = margin.iloc[index].to_numpy(dtype=float)
        finite = np.isfinite(run_margin)
        center = float(np.mean(run_margin[finite]))
        scale = float(np.std(run_margin[finite]))
        z = np.full(len(index), np.nan)
        z[finite] = (run_margin[finite] - center) / max(scale, EPS)
        bin_index = np.digitize(z, edges[1:-1], right=False)
        for outcome, (risk_all, target_all) in definitions.items():
            risk = risk_all[index] & finite
            for value in range(len(edges) - 1):
                chosen = risk & (bin_index == value)
                if np.sum(chosen) < 10:
                    continue
                records.append(
                    {
                        "subject": design.subject, "game": design.game,
                        "run_label": str(run_label), "outcome": outcome,
                        "M_z_bin": value,
                        "M_z_mean": float(np.mean(z[chosen])),
                        "observed_rate": float(np.mean(target_all[index][chosen])),
                        "n": int(np.sum(chosen)),
                    }
                )
    return pd.DataFrame(records)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans", "font.size": 9,
            "axes.edgecolor": INK, "axes.labelcolor": INK,
            "axes.titlecolor": INK, "xtick.color": INK, "ytick.color": INK,
            "axes.grid": True, "grid.color": GRID, "grid.alpha": 0.65,
            "axes.spines.top": False, "axes.spines.right": False,
            "figure.facecolor": "white", "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def mean_ci(values: np.ndarray) -> Tuple[float, float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    mean = float(np.mean(x)) if len(x) else np.nan
    if len(x) < 2:
        return mean, np.nan, np.nan
    half = float(stats.t.ppf(0.975, len(x) - 1) * np.std(x, ddof=1) / math.sqrt(len(x)))
    return mean, mean - half, mean + half


def plot_subject_effects(effects: pd.DataFrame, output: Path, subject: int, game: int) -> None:
    comparisons = ("A1_to_A2", "A2_to_A3", "A2_to_A4", "A2_to_A5")
    labels = ("Policy\nA1->A2", "Summary\nA2->A3", "Mean29\nA2->A4", "Concat29\nA2->A5")
    colors = (COLORS["A2_policy"], COLORS["A3_summary"], COLORS["A4_mean29"], COLORS["A5_concat29"])
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharey=False)
    for axis, outcome in zip(axes, OUTCOMES):
        data = effects.loc[effects["outcome"].eq(outcome)]
        for position, (comparison, color) in enumerate(zip(comparisons, colors)):
            value = data.loc[data["comparison"].eq(comparison), "delta_nll_positive_is_improvement"].to_numpy(float)
            if len(value):
                violin = axis.violinplot(value, positions=[position], widths=0.72, showextrema=False)
                for body in violin["bodies"]:
                    body.set_facecolor(color); body.set_alpha(0.17); body.set_edgecolor(color)
                jitter = np.linspace(-0.17, 0.17, len(value))
                axis.scatter(position + jitter, value, s=18, color=color, alpha=0.58, zorder=3)
                mean, lower, upper = mean_ci(value)
                axis.errorbar(position, mean, yerr=[[mean - lower], [upper - mean]], fmt="o", color=color, ms=6, lw=1.6, zorder=4)
        axis.axhline(0, color=INK, lw=1, ls="--")
        axis.set_xticks(range(len(labels)), labels)
        axis.set_title(OUTCOME_LABELS[outcome])
        axis.set_ylabel("Held-out run Delta NLL\n(positive = improvement)")
    fig.suptitle(f"Subject {subject:03d}, Game {game}: run-level held-out improvements", y=1.02)
    fig.tight_layout()
    save_figure(fig, output)


def plot_calibration(calibration: pd.DataFrame, output: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), sharex=True, sharey=True)
    for axis, outcome in zip(axes, OUTCOMES):
        data = calibration.loc[calibration["outcome"].eq(outcome)]
        for model in MODEL_ORDER:
            model_data = data.loc[data["model"].eq(model)]
            if model_data.empty:
                continue
            curve = model_data.groupby("bin", sort=True, observed=True).apply(
                lambda frame: pd.Series(
                    {
                        "predicted": np.average(frame["predicted_mean"], weights=frame["n"]),
                        "observed": np.average(frame["observed_mean"], weights=frame["n"]),
                    }
                ), include_groups=False,
            ).reset_index()
            axis.plot(curve["predicted"], curve["observed"], marker="o", ms=3.2, lw=1.3,
                      color=COLORS[model], alpha=0.88, label=MODEL_LABELS[model].replace("\n", " "))
        axis.plot([0, 1], [0, 1], color=INK, ls="--", lw=1)
        axis.set_xlim(0, 1); axis.set_ylim(0, 1)
        axis.set_title(OUTCOME_LABELS[outcome])
        axis.set_xlabel("Predicted")
    axes[0].set_ylabel("Observed")
    axes[-1].legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    save_figure(fig, output)


def plot_m_descriptive(frame: pd.DataFrame, output: Path, title: str) -> None:
    order = ("exact_switch", "noop_to_active", "active_to_noop")
    labels = ("Exact-action switch", "NOOP -> active", "Active -> NOOP")
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), sharex=True)
    for axis, outcome, label in zip(axes, order, labels):
        data = frame.loc[frame["outcome"].eq(outcome)]
        for _, run in data.groupby(["subject", "game", "run_label"], sort=False):
            axis.plot(run["M_z_mean"], run["observed_rate"], color="#95a5a6", alpha=0.18, lw=0.8)
        curve = data.groupby("M_z_bin", sort=True, observed=True).agg(
            M_z_mean=("M_z_mean", "mean"), observed_rate=("observed_rate", "mean")
        ).reset_index()
        axis.plot(curve["M_z_mean"], curve["observed_rate"], color="#c0392b", marker="o", lw=2)
        axis.set_title(label); axis.set_xlabel("Within-run pre-search M (z)")
        axis.set_ylabel("Observed probability")
    fig.suptitle(title + "\nM is descriptive only; A2 uses the nonredundant policy ILR", y=1.06)
    fig.tight_layout()
    save_figure(fig, output)


def exact_sign_patterns(n_units: int) -> np.ndarray:
    if n_units <= 1 or n_units > 20:
        raise ValueError(n_units)
    return np.asarray(list(itertools.product((-1.0, 1.0), repeat=n_units)))


def studentized_mean(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    mean = float(np.mean(x)); sd = float(np.std(x, ddof=1))
    if sd <= EPS:
        return 0.0 if abs(mean) <= EPS else math.copysign(np.inf, mean)
    return mean / (sd / math.sqrt(len(x)))


def synchronized_sign_flip(matrix: np.ndarray) -> pd.DataFrame:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != 6 or not np.all(np.isfinite(values)):
        raise ValueError(f"Group inference requires a finite 6 x K matrix, got {values.shape}")
    patterns = exact_sign_patterns(values.shape[0])
    observed = np.asarray([studentized_mean(values[:, j]) for j in range(values.shape[1])])
    null = np.empty((len(patterns), values.shape[1]))
    for i, sign in enumerate(patterns):
        null[i] = [studentized_mean(values[:, j] * sign) for j in range(values.shape[1])]
    max_abs = np.max(np.abs(null), axis=1)
    rows = []
    for j, statistic in enumerate(observed):
        threshold = abs(statistic)
        rows.append(
            {
                "mean_delta_nll": float(np.mean(values[:, j])),
                "studentized_t": float(statistic),
                "exact_sign_flip_p": float(np.mean(np.abs(null[:, j]) >= threshold - 1e-12)),
                "synchronized_max_t_p": float(np.mean(max_abs >= threshold - 1e-12)),
                "n_positive": int(np.sum(values[:, j] > 0)),
                "n_subjects": 6,
                "n_sign_patterns": 64,
            }
        )
    return pd.DataFrame(rows)


def infer_effect_family(subject_effects: pd.DataFrame, family: str) -> pd.DataFrame:
    effect_ids = list(POLICY_EFFECTS if family == "policy_anchor" else PLANNING_EFFECTS)
    matrix = subject_effects.pivot(index="subject", columns="effect_id", values="delta_nll_positive_is_improvement").reindex(
        index=range(1, 7), columns=effect_ids
    )
    if matrix.isna().any().any():
        raise RuntimeError(f"Incomplete subject x effect matrix for {family}")
    inference = synchronized_sign_flip(matrix.to_numpy(dtype=float))
    inference.insert(0, "effect_id", effect_ids)
    inference.insert(0, "family", family)
    return inference


def significance_stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return "ns"


def plot_group_effects(
    subject_effects: pd.DataFrame,
    inference: pd.DataFrame,
    output: Path,
    title: str,
) -> None:
    comparisons = ("A1_to_A2", "A2_to_A3", "A2_to_A4", "A2_to_A5")
    labels = ("Policy", "Summary", "Mean29", "Concat29")
    colors = (COLORS["A2_policy"], COLORS["A3_summary"], COLORS["A4_mean29"], COLORS["A5_concat29"])
    p_lookup = dict(zip(inference["effect_id"], inference["synchronized_max_t_p"]))
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2))
    for axis, outcome in zip(axes, OUTCOMES):
        data = subject_effects.loc[subject_effects["outcome"].eq(outcome)]
        for position, (comparison, color) in enumerate(zip(comparisons, colors)):
            value = data.loc[data["comparison"].eq(comparison)].sort_values("subject")
            y = value["delta_nll_positive_is_improvement"].to_numpy(float)
            violin = axis.violinplot(y, positions=[position], widths=0.72, showextrema=False)
            for body in violin["bodies"]:
                body.set_facecolor(color); body.set_alpha(0.18); body.set_edgecolor(color)
            jitter = np.linspace(-0.15, 0.15, len(y))
            axis.scatter(position + jitter, y, color=color, edgecolor="white", linewidth=0.5, s=34, zorder=3)
            mean, lower, upper = mean_ci(y)
            axis.errorbar(position, mean, yerr=[[mean - lower], [upper - mean]], fmt="o", color=INK, ms=5, lw=1.5, zorder=4)
            effect_id = f"{outcome}::{comparison}"
            star = significance_stars(float(p_lookup.get(effect_id, 1.0)))
            upper_y = float(np.max(y))
            span = max(float(np.ptp(y)), abs(mean), 1e-4)
            axis.text(position, upper_y + 0.12 * span, star, ha="center", va="bottom", color=INK, fontsize=9)
        axis.axhline(0, color=INK, ls="--", lw=1)
        axis.set_xticks(range(4), labels)
        axis.set_title(OUTCOME_LABELS[outcome])
        axis.set_ylabel("Subject mean Delta NLL\n(equal weight per held-out run)")
    fig.suptitle(title + "\nStars: family-wise synchronized maxT exact sign-flip", y=1.04)
    fig.tight_layout()
    save_figure(fig, output)


def subject_dir(output_root: Path, subject: int, game: int) -> Path:
    return Path(output_root) / f"sub{subject:03d}_game{game}"


def validate_subject_output(
    directory: Path,
    *,
    subject: int,
    game: int,
    C_grid: Sequence[float],
    script_snapshot: Mapping[str, object],
    verify_sources: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, object]]:
    manifest_path = directory / "manifest.json"
    manifest = _read_json(manifest_path)
    expected_contract = analysis_contract(C_grid)
    if (
        manifest.get("status") != "complete"
        or int(manifest.get("subject", -1)) != subject
        or int(manifest.get("game", -1)) != game
        or manifest.get("analysis_contract") != expected_contract
    ):
        raise RuntimeError(f"Stale/incompatible subject manifest: {manifest_path}")
    recorded_script = manifest.get("script", {})
    if not isinstance(recorded_script, Mapping) or any(
        recorded_script.get(key) != script_snapshot.get(key)
        for key in ("size", "mtime_ns", "sha256")
    ):
        raise RuntimeError(f"Subject output was made by a different script: {manifest_path}")
    recorded_tables = manifest.get("tables", {})
    if not isinstance(recorded_tables, Mapping):
        raise RuntimeError(f"Subject manifest lacks table signatures: {manifest_path}")
    required_tables = (
        "fold_metrics", "fold_effects", "subject_effects", "calibration_bins",
        "nested_tuning", "M_descriptive",
    )
    table_signatures = {
        name: verify_recorded_signature(recorded_tables.get(name, {}), f"{directory.name}/{name}")
        for name in required_tables
    }
    verify_recorded_signature(manifest.get("qa", {}), f"{directory.name}/qa")
    recorded_figures = manifest.get("figures", {})
    if not isinstance(recorded_figures, Mapping) or len(recorded_figures) != 3:
        raise RuntimeError(f"Subject manifest lacks the three figure signatures: {manifest_path}")
    for name, signature in recorded_figures.items():
        verify_recorded_signature(signature, f"{directory.name}/{name}")
    if verify_sources:
        source = manifest.get("source_audit", {})
        if not isinstance(source, Mapping):
            raise RuntimeError(f"Subject manifest lacks source audit: {manifest_path}")
        for name in (
            "core_manifest", "real_step_core", "imaginary_event_core",
            "visual_descriptor", "visual_qa", "visual_alignment",
        ):
            verify_recorded_signature(source.get(name, {}), f"{directory.name}/source/{name}")
        raw = source.get("raw_trace", {})
        checks = raw.get("raw_hash_checks", []) if isinstance(raw, Mapping) else []
        if len(checks) != 11:
            raise RuntimeError(f"Subject raw-trace metadata audit is incomplete: {directory}")
        for item in checks:
            verify_recorded_signature(item.get("metadata", {}), f"{directory.name}/trace_metadata")
    provenance = {
        "subject": subject, "game": game,
        "manifest": file_signature(manifest_path),
        "tables": table_signatures,
        "analysis_contract_sha256": expected_contract["contract_sha256"],
    }
    return manifest, provenance


def run_subject(
    subject: int,
    game: int,
    *,
    core_root: Path,
    visual_root: Path,
    trace_metadata_root: Optional[Path],
    output_root: Path,
    C_grid: Sequence[float],
    overwrite: bool,
) -> Path:
    script_snapshot = file_signature(Path(__file__))
    contract = analysis_contract(C_grid)
    output = subject_dir(output_root, subject, game)
    manifest_path = output / "manifest.json"
    configured_roots = {
        "core_root": str(Path(core_root).resolve()),
        "visual_root": str(Path(visual_root).resolve()),
        "trace_metadata_root": (
            str(Path(trace_metadata_root).resolve())
            if trace_metadata_root is not None else None
        ),
    }
    if manifest_path.exists() and not overwrite:
        existing, _ = validate_subject_output(
            output, subject=subject, game=game, C_grid=C_grid,
            script_snapshot=script_snapshot, verify_sources=True,
        )
        if existing.get("configured_roots") != configured_roots:
            raise RuntimeError(f"Subject output uses different configured source roots: {output}")
        log(f"strictly validated and reused {output}")
        return output
    output.mkdir(parents=True, exist_ok=True)
    (output / "tables").mkdir(exist_ok=True)
    (output / "figures" / "main").mkdir(parents=True, exist_ok=True)
    started = time.time()
    log(f"loading raw decision data sub{subject:03d} game{game}")
    design = load_subject_design(
        subject, game, core_root=core_root, visual_root=visual_root,
        trace_metadata_root=trace_metadata_root,
    )
    metrics, effects, subject_effects, calibration, tuning, qa = fit_subject_loro(
        design, C_grid=C_grid
    )
    m_descriptive = build_m_descriptive(design)
    tables = {
        "fold_metrics": output / "tables" / "fold_metrics.csv.gz",
        "fold_effects": output / "tables" / "fold_effects.csv.gz",
        "subject_effects": output / "tables" / "subject_effects.csv",
        "calibration_bins": output / "tables" / "calibration_bins.csv.gz",
        "nested_tuning": output / "tables" / "nested_tuning.csv.gz",
        "M_descriptive": output / "tables" / "M_descriptive.csv.gz",
    }
    metrics.to_csv(tables["fold_metrics"], index=False, compression="gzip")
    effects.to_csv(tables["fold_effects"], index=False, compression="gzip")
    subject_effects.to_csv(tables["subject_effects"], index=False)
    calibration.to_csv(tables["calibration_bins"], index=False, compression="gzip")
    tuning.to_csv(tables["nested_tuning"], index=False, compression="gzip")
    m_descriptive.to_csv(tables["M_descriptive"], index=False, compression="gzip")
    figures = {
        "model_improvement": output / "figures" / "main" / "main_figure1_run_level_model_improvement.png",
        "calibration": output / "figures" / "main" / "main_figure2_oof_calibration.png",
        "M_descriptive": output / "figures" / "main" / "main_figure3_presearch_M_descriptive.png",
    }
    plot_subject_effects(
        effects, figures["model_improvement"],
        subject, game,
    )
    plot_calibration(
        calibration, figures["calibration"],
        f"Subject {subject:03d}, Game {game}: held-out calibration",
    )
    plot_m_descriptive(
        m_descriptive, figures["M_descriptive"],
        f"Subject {subject:03d}, Game {game}: pre-search margin sensitivity",
    )
    qa_path = output / "qa.json"
    qa["analysis_contract"] = contract
    atomic_json(qa_path, qa)
    assert_script_unchanged(script_snapshot)
    manifest = {
        "status": "complete",
        "analysis_version": ANALYSIS_VERSION,
        "script": script_snapshot,
        "analysis_contract": contract,
        "subject": subject, "game": game,
        "configured_roots": configured_roots,
        "elapsed_s": float(time.time() - started),
        "source_audit": design.source_audit,
        "representation_contract": TRACE._representation_contract(),
        "models": {
            "A1": "pre-outcome behavior/visual/history controls",
            "A2": "A1 + five-dimensional orthonormal log-ratio P + Q0 incumbent margin/availability",
            "A3": "A2 + Q-revision/evidence/geometry summaries",
            "A4": "A2 + arithmetic mean of the raw 29 x 12 trace",
            "A5": "A2 + event-major raw 29 x 12 concatenation",
        },
        "primary_families": {
            "policy_anchor": list(POLICY_EFFECTS),
            "planning_primary": list(PLANNING_EFFECTS),
        },
        "training_provenance_caveat": {
            "imitation_checkpoint_or_training_split_in_consumed_cache": "not_available",
            "claim_limit": (
                "Outer LORO validates the behavioral readout, not necessarily the upstream "
                "Thinker checkpoint. Do not call this fully out-of-sample Thinker validation "
                "unless separate checkpoint provenance proves the tested gameplay was excluded "
                "from imitation training."
            ),
        },
        "replay_conditioning_caveat": {
            "human_gameplay_replayed_through_thinker": True,
            "teacher_forcing_or_observed_trajectory_conditioning": True,
            "claim_limit": (
                "The recovered trace is conditional on the observed human gameplay/state-action "
                "trajectory. It is not evidence that a free-running Thinker independently generated "
                "the same trajectory or that the participant executed the literal 29 latent events."
            ),
        },
        "HRF_TR_trace_used": False,
        "imaginary_event_specific_timing_claimed": False,
        "tables": {name: file_signature(path) for name, path in tables.items()},
        "figures": {name: file_signature(path) for name, path in figures.items()},
        "qa": file_signature(qa_path),
    }
    assert_script_unchanged(script_snapshot)
    atomic_json(manifest_path, manifest)
    log(f"completed {output} in {manifest['elapsed_s']:.1f}s")
    return output


def load_group_subject_tables(
    output_root: Path,
    game: int,
    *,
    C_grid: Sequence[float],
    script_snapshot: Mapping[str, object],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[Dict[str, object]]]:
    effects: List[pd.DataFrame] = []
    calibration: List[pd.DataFrame] = []
    margin: List[pd.DataFrame] = []
    inputs: List[Dict[str, object]] = []
    for subject in range(1, 7):
        directory = subject_dir(output_root, subject, game)
        _, provenance = validate_subject_output(
            directory, subject=subject, game=game, C_grid=C_grid,
            script_snapshot=script_snapshot, verify_sources=True,
        )
        subject_effect = pd.read_csv(directory / "tables" / "subject_effects.csv")
        subject_calibration = pd.read_csv(directory / "tables" / "calibration_bins.csv.gz")
        subject_margin = pd.read_csv(directory / "tables" / "M_descriptive.csv.gz")
        for label, frame in (
            ("effects", subject_effect), ("calibration", subject_calibration),
            ("margin", subject_margin),
        ):
            if (
                set(pd.to_numeric(frame["subject"], errors="raise").astype(int)) != {subject}
                or set(pd.to_numeric(frame["game"], errors="raise").astype(int)) != {game}
            ):
                raise RuntimeError(f"{directory.name} {label} identity mismatch")
        if (
            len(subject_effect) != len(OUTCOMES) * 5
            or subject_effect.duplicated(["outcome", "comparison"]).any()
            or set(subject_effect.loc[subject_effect["primary"].eq(1), "effect_id"])
            != set(PRIMARY_EFFECTS)
        ):
            raise RuntimeError(f"Malformed subject-effect grid: {directory}")
        effects.append(subject_effect)
        calibration.append(subject_calibration)
        margin.append(subject_margin)
        inputs.append(provenance)
    combined_effects = pd.concat(effects, ignore_index=True)
    combined_calibration = pd.concat(calibration, ignore_index=True)
    combined_margin = pd.concat(margin, ignore_index=True)
    expected_subjects = set(range(1, 7))
    for label, frame in (
        ("effects", combined_effects), ("calibration", combined_calibration),
        ("margin", combined_margin),
    ):
        observed = set(pd.to_numeric(frame["subject"], errors="raise").astype(int))
        if observed != expected_subjects:
            raise RuntimeError(f"Game {game} {label} does not contain exactly subjects 1..6")
    if len(inputs) != 6 or len({item["subject"] for item in inputs}) != 6:
        raise RuntimeError(f"Game {game} does not have exactly six unique subject manifests")
    return combined_effects, combined_calibration, combined_margin, inputs


def run_group_game(output_root: Path, game: int, C_grid: Sequence[float]) -> Path:
    script_snapshot = file_signature(Path(__file__))
    contract = analysis_contract(C_grid)
    output = Path(output_root) / f"group_game{game}"
    (output / "tables").mkdir(parents=True, exist_ok=True)
    (output / "figures" / "main").mkdir(parents=True, exist_ok=True)
    effects, calibration, margin, inputs = load_group_subject_tables(
        output_root, game, C_grid=C_grid, script_snapshot=script_snapshot
    )
    primary = effects.loc[effects["primary"].eq(1)].copy()
    if set(primary["effect_id"]) != set(PRIMARY_EFFECTS):
        raise RuntimeError(f"Game {game} primary effect family is incomplete")
    inference = pd.concat(
        [infer_effect_family(primary, "policy_anchor"), infer_effect_family(primary, "planning_primary")],
        ignore_index=True,
    )
    tables = {
        "subject_effects": output / "tables" / "subject_effects.csv",
        "exact_inference": output / "tables" / "exact_inference.csv",
    }
    figures = {
        "behavioral_validation": output / "figures" / "main" / "main_figure1_cross_subject_behavioral_validation.png",
        "calibration": output / "figures" / "main" / "main_figure2_cross_subject_calibration.png",
        "M_descriptive": output / "figures" / "main" / "main_figure3_cross_subject_presearch_M.png",
    }
    effects.to_csv(tables["subject_effects"], index=False)
    inference.to_csv(tables["exact_inference"], index=False)
    plot_group_effects(
        primary, inference, figures["behavioral_validation"],
        f"Game {game}: full-trace behavioral validation",
    )
    plot_calibration(
        calibration, figures["calibration"],
        f"Game {game}: cross-subject held-out calibration",
    )
    plot_m_descriptive(
        margin, figures["M_descriptive"],
        f"Game {game}: descriptive pre-search M relationship",
    )
    assert_script_unchanged(script_snapshot)
    atomic_json(
        output / "manifest.json",
        {
            "status": "complete", "analysis_version": ANALYSIS_VERSION,
            "analysis_contract": contract, "game": game,
            "subjects": list(range(1, 7)), "n_subjects": 6,
            "inference": "two-sided exact subject sign-flip; synchronized max-|studentized t| separately within policy_anchor and planning_primary",
            "n_sign_patterns": 64, "run_equal_subject_estimand": True,
            "script": script_snapshot, "input_subject_artifacts": inputs,
            "output_tables": {name: file_signature(path) for name, path in tables.items()},
            "output_figures": {name: file_signature(path) for name, path in figures.items()},
        },
    )
    return output


def run_paired_group(output_root: Path, C_grid: Sequence[float]) -> Path:
    script_snapshot = file_signature(Path(__file__))
    contract = analysis_contract(C_grid)
    game1, calibration1, margin1, inputs1 = load_group_subject_tables(
        output_root, 1, C_grid=C_grid, script_snapshot=script_snapshot
    )
    game2, calibration2, margin2, inputs2 = load_group_subject_tables(
        output_root, 2, C_grid=C_grid, script_snapshot=script_snapshot
    )
    inputs = inputs1 + inputs2
    if len(inputs) != 12 or len({(item["subject"], item["game"]) for item in inputs}) != 12:
        raise RuntimeError("Paired analysis requires exactly 12 unique subject-game inputs")
    key = ["subject", "outcome", "comparison", "effect_id", "family", "primary"]
    paired = game1.merge(game2, on=key, suffixes=("_game1", "_game2"), validate="one_to_one")
    paired["game"] = "across_game_mean"
    paired["delta_nll_positive_is_improvement"] = (
        paired["delta_nll_positive_is_improvement_game1"]
        + paired["delta_nll_positive_is_improvement_game2"]
    ) / 2.0
    common = paired[key + ["game", "delta_nll_positive_is_improvement"]].copy()
    common["row_pooled_oof_delta_nll_sensitivity"] = (
        paired["row_pooled_oof_delta_nll_sensitivity_game1"]
        + paired["row_pooled_oof_delta_nll_sensitivity_game2"]
    ) / 2.0
    difference = common.copy()
    difference["game"] = "game2_minus_game1"
    difference["delta_nll_positive_is_improvement"] = (
        paired["delta_nll_positive_is_improvement_game2"]
        - paired["delta_nll_positive_is_improvement_game1"]
    )
    difference["row_pooled_oof_delta_nll_sensitivity"] = (
        paired["row_pooled_oof_delta_nll_sensitivity_game2"]
        - paired["row_pooled_oof_delta_nll_sensitivity_game1"]
    )
    inference_parts = []
    for estimand, frame in (("across_game_mean", common), ("game2_minus_game1", difference)):
        primary = frame.loc[frame["primary"].eq(1)]
        for family in ("policy_anchor", "planning_primary"):
            result = infer_effect_family(primary, family)
            result.insert(0, "estimand", estimand)
            inference_parts.append(result)
    inference = pd.concat(inference_parts, ignore_index=True)
    output = Path(output_root) / "group_paired_games"
    (output / "tables").mkdir(parents=True, exist_ok=True)
    (output / "figures" / "main").mkdir(parents=True, exist_ok=True)
    tables = {
        "paired_subject_effects": output / "tables" / "paired_subject_effects.csv",
        "across_game_mean_subject_effects": output / "tables" / "across_game_mean_subject_effects.csv",
        "game2_minus_game1_subject_effects": output / "tables" / "game2_minus_game1_subject_effects.csv",
        "paired_exact_inference": output / "tables" / "paired_exact_inference.csv",
    }
    figures = {
        "common_validation": output / "figures" / "main" / "main_figure1_across_game_common_validation.png",
        "game_difference": output / "figures" / "main" / "main_figure2_game2_minus_game1_validation.png",
        "calibration": output / "figures" / "main" / "main_figure3_across_game_calibration.png",
        "M_descriptive": output / "figures" / "main" / "main_figure4_across_game_presearch_M.png",
    }
    paired.to_csv(tables["paired_subject_effects"], index=False)
    common.to_csv(tables["across_game_mean_subject_effects"], index=False)
    difference.to_csv(tables["game2_minus_game1_subject_effects"], index=False)
    inference.to_csv(tables["paired_exact_inference"], index=False)
    plot_group_effects(
        common.loc[common["primary"].eq(1)],
        inference.loc[inference["estimand"].eq("across_game_mean")],
        figures["common_validation"],
        "Across-game mean: full-trace behavioral validation",
    )
    plot_group_effects(
        difference.loc[difference["primary"].eq(1)],
        inference.loc[inference["estimand"].eq("game2_minus_game1")],
        figures["game_difference"],
        "Game 2 minus Game 1: behavioral-validation effect difference",
    )
    plot_calibration(
        pd.concat((calibration1, calibration2), ignore_index=True),
        figures["calibration"],
        "Across games: held-out calibration",
    )
    plot_m_descriptive(
        pd.concat((margin1, margin2), ignore_index=True),
        figures["M_descriptive"],
        "Across games: descriptive pre-search M relationship",
    )
    assert_script_unchanged(script_snapshot)
    atomic_json(
        output / "manifest.json",
        {
            "status": "complete", "analysis_version": ANALYSIS_VERSION,
            "analysis_contract": contract,
            "subjects": list(range(1, 7)), "games": [1, 2],
            "n_subjects": 6, "n_subject_game_inputs": 12,
            "estimands": ["within-subject across-game mean", "within-subject game2-game1 difference"],
            "inference": "two-sided exact paired sign-flip; synchronized maxT separately by estimand and effect family",
            "n_sign_patterns": 64, "script": script_snapshot,
            "input_subject_artifacts": inputs,
            "output_tables": {name: file_signature(path) for name, path in tables.items()},
            "output_figures": {name: file_signature(path) for name, path in figures.items()},
        },
    )
    return output


def benchmark_estimators(
    design: SubjectDesign,
    output: Optional[Path] = None,
    *,
    C_grid: Sequence[float] = DEFAULT_C_GRID,
) -> Dict[str, object]:
    """Acceptance benchmark for one real exact-action/A5/outer-run cell."""
    risk, target, _ = outcome_definition(design.metadata, "exact_action")
    rows = np.flatnonzero(risk)
    groups = design.metadata.loc[risk, "run_label"].astype(str).to_numpy()
    x, _ = model_matrix(design, "A5_concat29")
    x = x[rows]; y = target[rows]
    held_out = pd.unique(groups)[-1]
    train = groups != held_out; test = ~train
    preprocessor = fit_preprocessor(x[train], groups[train])
    x_train = transform_preprocessor(preprocessor, x[train])
    x_test = transform_preprocessor(preprocessor, x[test])
    seed = stable_seed(SEED, "benchmark", design.subject, design.game)
    fit1, diag1 = fit_logistic_l2(x_train, y[train], groups[train], C=0.1, seed=seed)
    p1 = aligned_probability(fit1, x_test, range(N_ACTIONS))
    nll1, _ = row_metrics(y[test], p1, range(N_ACTIONS))
    fit2, diag2 = fit_logistic_l2(x_train, y[train], groups[train], C=0.1, seed=seed)
    p2 = aligned_probability(fit2, x_test, range(N_ACTIONS))
    deterministic_difference = float(np.max(np.abs(p1 - p2)))
    payload = {
        "script": file_signature(Path(__file__)),
        "analysis_contract": analysis_contract(C_grid),
        "subject": design.subject, "game": design.game,
        "outcome": "exact_action", "model": "A5_concat29",
        "held_out_run": held_out, "n_train": int(np.sum(train)), "n_test": int(np.sum(test)),
        "n_features": int(x.shape[1]),
        "accepted_LBFGS_C_0.1": {
            **diag1, "test_nll": float(np.mean(nll1)),
            "repeat_max_probability_difference": deterministic_difference,
            "repeat_elapsed_s": diag2["elapsed_s"],
        },
        "rejected_prior_benchmark": {
            "averaged_SGD": "converged/deterministic but severe held-out underfit",
            "SAGA": "approximately ten times slower than converged L-BFGS and did not converge",
        },
    }
    if deterministic_difference != 0.0:
        raise RuntimeError("Repeated L-BFGS fit is not deterministic")
    if output is not None:
        atomic_json(output, payload)
    return payload


def run_self_test() -> Dict[str, object]:
    rng = np.random.default_rng(17)
    p = rng.dirichlet(np.ones(N_ACTIONS), size=50)
    ilr = policy_ilr(p)
    if ilr.shape != (50, 5) or not np.all(np.isfinite(ilr)):
        raise AssertionError("Policy ILR failed")
    basis = helmert(N_ACTIONS, full=False)
    if not np.allclose(basis @ basis.T, np.eye(N_ACTIONS - 1), atol=1e-12):
        raise AssertionError("Helmert ILR basis is not orthonormal")
    groups = np.repeat(np.asarray(["a", "b", "c"]), [5, 10, 20])
    weight = equal_run_weights(groups)
    totals = [np.sum(weight[groups == label]) for label in ("a", "b", "c")]
    if not np.allclose(totals, totals[0]):
        raise AssertionError("Equal-run weighting failed")
    matrix = np.arange(1, 7, dtype=float)[:, None] * np.ones((1, 3))
    exact = synchronized_sign_flip(matrix)
    if not np.allclose(exact["exact_sign_flip_p"], 0.03125):
        raise AssertionError("n=6 exact two-sided floor failed")
    if not np.all(exact["synchronized_max_t_p"] >= exact["exact_sign_flip_p"]):
        raise AssertionError("maxT p must be >= unadjusted p")
    # Minimal exact raw-token construction gate.
    rows = []
    for actor in (10, 11):
        for position in range(TRACE.N_EVENTS):
            rows.append(
                {
                    "run_label": "r1", "actor_row": actor,
                    "status2_position": position, "bout_index": position // 4,
                    "branch": position % N_ACTIONS, "depth": 1 + position % 3,
                    "return_gain_from_root": float(position - 14),
                    "root_q_revision": np.nan if position == 0 else position / 100,
                    "valid_repeated_q_revision": 0 if position == 0 else 1,
                    "root_n_before_sampled": position / 30,
                    "current_time_s": float(actor),
                }
            )
    batch = TRACE._rows_to_window_batch(pd.DataFrame(rows))
    if batch.concat29.shape != (2, 348) or batch.mean29.shape != (2, 12):
        raise AssertionError("Raw 29-event representation contract failed")
    hierarchy = {
        "A1_controls": ("controls",),
        "A2_policy": ("controls", "policy"),
        "A3_summary": ("controls", "policy", "summary"),
        "A4_mean29": ("controls", "policy", "mean29"),
        "A5_concat29": ("controls", "policy", "concat29"),
    }
    if any("M" in token for token in hierarchy["A2_policy"]):
        raise AssertionError("M must not duplicate P in A2")
    expected_C_grid = (
        0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3,
        1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0,
    )
    if DEFAULT_C_GRID != expected_C_grid:
        raise AssertionError("Default C grid differs from the final fixed common grid")
    if len(set(DEFAULT_C_GRID)) != len(DEFAULT_C_GRID) or any(
        right <= left
        for left, right in zip(DEFAULT_C_GRID[:-1], DEFAULT_C_GRID[1:])
    ):
        raise AssertionError("Default C grid must be strictly increasing and unique")
    contract = analysis_contract(DEFAULT_C_GRID)
    if contract["C_grid_guard_endpoints"] != [0.0003, 1000.0]:
        raise AssertionError("C-grid guard endpoints are absent from the contract")
    if LOGISTIC_MAX_ITER != 2000:
        raise AssertionError("The frozen L-BFGS iteration guard must be 2000")
    accepted_estimator = (
        "LogisticRegression(lbfgs,L2,cold_start,max_iter=2000,tol=1e-4)"
    )
    if contract["estimator"] != accepted_estimator:
        raise AssertionError("The accepted estimator text differs from the contract")
    payload = {
        "status": "PASS", "analysis_version": ANALYSIS_VERSION,
        "policy_ilr_dimension": 5, "policy_ilr_orthonormal": True,
        "equal_run_weights": True, "nested_group_cv": True,
        "accepted_estimator": accepted_estimator,
        "analysis_contract_sha256": contract["contract_sha256"],
        "raw_trace_shape": [29, 12], "raw_concat_dimension": 348,
        "HRF_TR_trace_used": False, "M_in_A2": False,
        "outcomes": list(OUTCOMES), "models": hierarchy,
        "n6_exact_two_sided_floor": 0.03125,
        "primary_families": {
            "policy_anchor": list(POLICY_EFFECTS),
            "planning_primary": list(PLANNING_EFFECTS),
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=int, choices=range(1, 7))
    parser.add_argument("--game", type=int, choices=(1, 2))
    parser.add_argument("--run-all", action="store_true", help="run all 12 cells, both game groups, and paired games")
    parser.add_argument("--group-game", type=int, choices=(1, 2))
    parser.add_argument("--paired-group", action="store_true")
    parser.add_argument("--benchmark-models", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--core-root", type=Path, default=DEFAULT_CORE_ROOT)
    parser.add_argument("--visual-root", type=Path, default=DEFAULT_VISUAL_ROOT)
    parser.add_argument("--trace-metadata-root", type=Path, default=DEFAULT_TRACE_METADATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--C-grid", nargs="+", type=float, default=list(DEFAULT_C_GRID))
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    configure_style()
    args = build_argparser().parse_args()
    if args.self_test:
        run_self_test(); return
    if args.benchmark_models:
        if args.subject is None or args.game is None:
            raise ValueError("--benchmark-models requires --subject and --game")
        design = load_subject_design(
            args.subject, args.game, core_root=args.core_root,
            visual_root=args.visual_root, trace_metadata_root=args.trace_metadata_root,
        )
        payload = benchmark_estimators(
            design,
            args.output_root / "qa" / f"benchmark_sub{args.subject:03d}_game{args.game}.json",
            C_grid=args.C_grid,
        )
        print(json.dumps(payload, indent=2, sort_keys=True)); return
    if args.run_all:
        for game in (1, 2):
            for subject in range(1, 7):
                run_subject(
                    subject, game, core_root=args.core_root, visual_root=args.visual_root,
                    trace_metadata_root=args.trace_metadata_root,
                    output_root=args.output_root, C_grid=args.C_grid,
                    overwrite=args.overwrite,
                )
            run_group_game(args.output_root, game, args.C_grid)
        run_paired_group(args.output_root, args.C_grid); return
    if args.subject is not None:
        if args.game is None:
            raise ValueError("--subject requires --game")
        run_subject(
            args.subject, args.game, core_root=args.core_root, visual_root=args.visual_root,
            trace_metadata_root=args.trace_metadata_root,
            output_root=args.output_root, C_grid=args.C_grid,
            overwrite=args.overwrite,
        ); return
    if args.group_game is not None:
        run_group_game(args.output_root, args.group_game, args.C_grid); return
    if args.paired_group:
        run_paired_group(args.output_root, args.C_grid); return
    raise ValueError("Choose --subject/--game, --run-all, --group-game, --paired-group, --benchmark-models, or --self-test")


if __name__ == "__main__":
    main()
