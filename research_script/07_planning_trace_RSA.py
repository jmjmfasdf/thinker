#!/usr/bin/env python3
"""Controlled RSA for the full 29-event Thinker planning trace.

This analysis asks whether ROI multivoxel geometry tracks either of two
predeclared summaries of the same planning episode:

``planning_concat29``
    The ordered 29-event trace, retaining event identity/order.
``planning_mean29``
    The event-mean trace, retaining episode content but discarding order.

Both are tested separately (they are alternative representations, not mutual
nuisances) beyond low-level visual state, generic decision-event HRF density,
current and previous actions, task history, drift, temporal lag, and pre-search
P/M/Q0.  The primary run statistic
is signed conditional partial Spearman correlation computed with an exact FWL
sufficient-statistic implementation.  Runs are averaged equally within each
subject.  Population inference uses all 64 sign patterns of the six subjects
and synchronized max-|t| correction across the single 2 trace x 110 ROI family.

The optional within-run temporal block permutation is deliberately diagnostic;
population inference is the primary inferential tier.  This module is the
maintained computation engine used by ``07_RSA.py``; the unified entry point
owns canonical output routing and MNI brain-map figures.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t


SCRIPT_DIR = Path(__file__).resolve().parent
COMMON_PATH = SCRIPT_DIR / "planning_brain_common.py"
TRACE_COMMON_PATH = SCRIPT_DIR / "planning_trace_common.py"


def _load_local_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


COMMON = _load_local_module("planning_brain_common", COMMON_PATH)

ANALYSIS_NAME = "07_planning_trace_RSA"
ANALYSIS_VERSION = "2026-08-16-controlled-trace-rsa-v2-decision-event"
ESTIMATOR_VERSION = "rank-fwl-sufficient-statistics-v1"
TRACE_NAMES = ("planning_concat29", "planning_mean29")
TRACE_WIDTHS = {"planning_concat29": 348, "planning_mean29": 12}
TRACE_LABELS = {
    "planning_concat29": "Ordered concat29",
    "planning_mean29": "Mean29",
}
CONTROL_NAMES = (
    "N_visual",
    "N_decision_event",
    "N_current_action",
    "N_previous_action",
    "N_task_history",
    "N_drift",
    "N_temporal_lag",
    "P_model_prior",
    "M_current_action_margin",
    "Q0_initial_value_margin",
)
EXPECTED_RUNS = 11
EXPECTED_ROIS = 110
EXPECTED_SUBJECTS = tuple(range(1, 7))
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs" / "07_RSA"
DEFAULT_PLANNING_TRACE_ROOT = (
    SCRIPT_DIR / "outputs" / "00_brain_cache" / "planning_trace_cache"
    / "v1_concat29_mean29"
)
SIGNFLIP_RTOL = 128.0 * np.finfo(np.float64).eps
SIGNFLIP_ATOL = 128.0 * np.finfo(np.float64).eps
T0 = time.time()

BLUE = "#31688E"
ORANGE = "#D17C24"
INK = "#30343B"
MUTED = "#737B87"
GRID = "#D7DCE3"


def log(message: str) -> None:
    print(f"[{ANALYSIS_NAME} +{time.time() - T0:8.1f}s] {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(COMMON, "atomic_write_json"):
        COMMON.atomic_write_json(path, dict(payload))
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(COMMON, "atomic_save_npz"):
        COMMON.atomic_save_npz(path, **arrays)
        return
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def save_figure(fig: plt.Figure, png_path: Path) -> None:
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(
        png_path.with_suffix(".svg"), bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)


def clean_roi_label(value: str) -> str:
    label = str(value)
    for prefix in ("ho_cortical_", "ho_subcortical_"):
        if label.startswith(prefix):
            label = label[len(prefix) :]
    return label.replace("_", " ").strip().title()


class TraceLoaderAdapter:
    """Thin compatibility boundary around the evolving strict trace API.

    No trace data are discovered or reconstructed here.  This adapter accepts
    only a small set of explicitly named strict-loader call signatures; all
    scientific validation remains in ``planning_trace_common`` and is audited
    again below for run/TR/key identity.
    """

    def __init__(self, module_path: Path = TRACE_COMMON_PATH):
        self.module_path = Path(module_path)
        self._module = None

    @property
    def module(self):
        if self._module is None:
            if not self.module_path.exists():
                raise FileNotFoundError(
                    f"Missing strict planning-trace loader: {self.module_path}"
                )
            self._module = _load_local_module(
                "planning_trace_common", self.module_path
            )
        return self._module

    def prepare(self, args: argparse.Namespace):
        module = self.module
        candidates = (
            "prepare_trace_runs_from_args",
            "prepare_planning_trace_runs_from_args",
        )
        function = next(
            (getattr(module, name) for name in candidates if hasattr(module, name)),
            None,
        )
        if function is None:
            raise RuntimeError(
                "planning_trace_common lacks a recognized strict prepare API: "
                f"{candidates}"
            )
        result = function(args)
        if not isinstance(result, tuple):
            raise TypeError("Strict trace prepare API must return a tuple")
        if len(result) == 3:
            run_specs, cache_audit, trace_audit = result
        elif len(result) == 2:
            run_specs, combined_audit = result
            if not isinstance(combined_audit, Mapping):
                raise TypeError("Two-item trace prepare result needs an audit mapping")
            cache_audit = combined_audit.get("brain_cache", combined_audit)
            trace_audit = combined_audit.get("planning_trace", combined_audit)
        else:
            raise TypeError(
                "Strict trace prepare API must return (runs, cache_audit, "
                "trace_audit) or (runs, combined_audit)"
            )
        runs = list(run_specs)
        if not runs:
            raise RuntimeError("Strict trace loader returned no runs")
        if not isinstance(cache_audit, Mapping) or not isinstance(
            trace_audit, Mapping
        ):
            raise TypeError("Strict trace prepare audits must be mappings")
        return runs, dict(cache_audit), dict(trace_audit)

    def load_run(self, run: Any, planning_trace_root: Path):
        module = self.module
        candidates = ("load_run_trace", "load_planning_trace_run")
        function = next(
            (getattr(module, name) for name in candidates if hasattr(module, name)),
            None,
        )
        if function is None:
            raise RuntimeError(
                "planning_trace_common lacks a recognized strict run loader: "
                f"{candidates}"
            )
        try:
            result = function(run, Path(planning_trace_root))
        except TypeError as first_error:
            try:
                result = function(run=run, planning_trace_root=Path(planning_trace_root))
            except TypeError:
                raise first_error
        if isinstance(result, tuple) and len(result) == 2:
            blocks, metadata = result
        elif isinstance(result, Mapping) and "blocks" in result:
            blocks = result["blocks"]
            metadata = result.get("metadata", {})
        else:
            raise TypeError(
                "Strict run loader must return (blocks, metadata) or a mapping "
                "with a 'blocks' entry"
            )
        if not isinstance(blocks, Mapping) or not isinstance(metadata, Mapping):
            raise TypeError("Trace blocks and metadata must be mappings")
        return dict(blocks), dict(metadata)

    @property
    def source_sha256(self) -> str:
        return sha256_file(self.module_path)


TRACE_ADAPTER = TraceLoaderAdapter()


def _finite_matrix(values: Any, *, name: str, n_rows: int | None = None) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a nonempty 2-D matrix; got {matrix.shape}")
    if n_rows is not None and matrix.shape[0] != int(n_rows):
        raise ValueError(
            f"{name} TR mismatch: {matrix.shape[0]} != expected {int(n_rows)}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def _feature_rdm(values: np.ndarray) -> np.ndarray:
    matrix = _finite_matrix(values, name="feature_rdm")
    spread = np.std(matrix, axis=0)
    keep = np.isfinite(spread) & (spread > 1e-12)
    if not np.any(keep):
        n_rows = len(matrix)
        return np.zeros(n_rows * (n_rows - 1) // 2, dtype=np.float32)
    return np.asarray(COMMON.euclidean_rdm(matrix[:, keep]), dtype=np.float32)


def _pattern_rdm(values: np.ndarray) -> np.ndarray:
    matrix = _finite_matrix(values, name="pattern_rdm")
    spread = np.std(matrix, axis=0)
    keep = np.isfinite(spread) & (spread > 1e-12)
    if int(np.sum(keep)) < 2:
        n_rows = len(matrix)
        return np.zeros(n_rows * (n_rows - 1) // 2, dtype=np.float32)
    return np.asarray(COMMON.correlation_rdm(matrix[:, keep]), dtype=np.float32)


def load_run_blocks(
    run: Any,
    *,
    planning_trace_root: Path,
    adapter: TraceLoaderAdapter = TRACE_ADAPTER,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load controls from the brain cache and traces from the strict loader."""
    n_tr = int(run.n_tr)
    process = COMMON.load_process_blocks(run)
    visual = _finite_matrix(COMMON.load_visual(run), name="visual", n_rows=n_tr)
    trace_blocks, trace_metadata = adapter.load_run(run, planning_trace_root)
    required_process = (
        "decision_event",
        "current_action",
        "previous_action",
        "task_history",
        "drift",
        "model_prior_policy",
        "model_margin",
        "initial_q_state",
    )
    missing_process = sorted(set(required_process).difference(process))
    if missing_process:
        raise ValueError(
            f"Canonical process cache lacks controls: {missing_process}"
        )
    blocks: dict[str, np.ndarray] = {"visual": visual}
    for name in required_process:
        blocks[name] = _finite_matrix(
            process[name], name=name, n_rows=n_tr
        )
    for name in TRACE_NAMES:
        if name not in trace_blocks:
            raise ValueError(f"Strict trace loader lacks required block {name}")
        blocks[name] = _finite_matrix(
            trace_blocks[name], name=name, n_rows=n_tr
        )
        if blocks[name].shape[1] != TRACE_WIDTHS[name]:
            raise ValueError(
                f"{name} width mismatch: {blocks[name].shape[1]} "
                f"!= canonical {TRACE_WIDTHS[name]}"
            )
    metadata_run = trace_metadata.get("run_label")
    if metadata_run is not None and str(metadata_run) != str(run.run_label):
        raise ValueError(
            f"Trace metadata run mismatch: {metadata_run} != {run.run_label}"
        )
    metadata_n_tr = trace_metadata.get("n_tr")
    if metadata_n_tr is not None and int(metadata_n_tr) != n_tr:
        raise ValueError(
            f"Trace metadata TR mismatch: {metadata_n_tr} != {n_tr}"
        )
    audit = {
        "run_label": str(run.run_label),
        "n_tr": n_tr,
        "trace_shapes": {
            name: list(blocks[name].shape) for name in TRACE_NAMES
        },
        "control_shapes": {
            name: list(blocks[name].shape)
            for name in ("visual", *required_process)
        },
        "trace_metadata": trace_metadata,
    }
    return blocks, audit


def build_rdms(blocks: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    n_tr = int(_finite_matrix(blocks["visual"], name="visual").shape[0])
    rdms = {
        "N_visual": _pattern_rdm(blocks["visual"]),
        "N_decision_event": _feature_rdm(blocks["decision_event"]),
        "N_current_action": _feature_rdm(blocks["current_action"]),
        "N_previous_action": _feature_rdm(blocks["previous_action"]),
        "N_task_history": _feature_rdm(blocks["task_history"]),
        "N_drift": _feature_rdm(blocks["drift"]),
        "N_temporal_lag": np.asarray(
            COMMON.temporal_lag_rdm(n_tr), dtype=np.float32
        ),
        "P_model_prior": _feature_rdm(blocks["model_prior_policy"]),
        "M_current_action_margin": _feature_rdm(blocks["model_margin"]),
        "Q0_initial_value_margin": _feature_rdm(blocks["initial_q_state"]),
        "planning_concat29": _pattern_rdm(blocks["planning_concat29"]),
        "planning_mean29": _pattern_rdm(blocks["planning_mean29"]),
    }
    expected_pairs = n_tr * (n_tr - 1) // 2
    bad = {
        name: int(len(values))
        for name, values in rdms.items()
        if len(values) != expected_pairs or not np.all(np.isfinite(values))
    }
    if bad:
        raise ValueError(
            f"RDM contract failed ({expected_pairs} finite pairs expected): {bad}"
        )
    return rdms


@dataclass(frozen=True)
class FWLDesign:
    reduced_q: np.ndarray
    target_residuals: np.ndarray
    target_residual_ss: np.ndarray
    target_ranked: np.ndarray
    kept_controls: tuple[str, ...]
    dropped_controls: tuple[str, ...]
    target_residual_ratio: tuple[float, ...]
    n_pairs_total: int
    n_pairs_retained: int


def prepare_fwl_design(
    rdms: Mapping[str, np.ndarray],
    pair_mask: np.ndarray,
    *,
    tolerance: float = 1e-10,
) -> FWLDesign:
    """Build one shared reduced space and two separately residualized targets."""
    mask = np.asarray(pair_mask, dtype=bool).reshape(-1)
    if int(np.sum(mask)) < 8:
        raise ValueError("Controlled trace RSA needs at least eight retained pairs")
    n_total = len(mask)
    basis: list[np.ndarray] = []
    intercept = np.ones(int(np.sum(mask)), dtype=np.float64)
    intercept /= np.linalg.norm(intercept)
    basis.append(intercept)
    kept: list[str] = []
    dropped: list[str] = []

    def residualize(column: np.ndarray) -> np.ndarray:
        residual = np.asarray(column, dtype=np.float64).copy()
        for _ in range(2):
            for vector in basis:
                residual -= vector * float(np.dot(vector, residual))
        return residual

    for name in CONTROL_NAMES:
        if name not in rdms or len(rdms[name]) != n_total:
            raise KeyError(f"Missing or malformed control RDM: {name}")
        column = np.asarray(COMMON.rank_zscore(rdms[name]), dtype=float)[mask]
        residual = residualize(column)
        scale = max(float(np.linalg.norm(column)), 1.0)
        norm = float(np.linalg.norm(residual))
        if norm <= tolerance * scale:
            dropped.append(name)
            continue
        kept.append(name)
        basis.append(residual / norm)

    target_ranked: list[np.ndarray] = []
    target_residuals: list[np.ndarray] = []
    target_ss: list[float] = []
    target_ratios: list[float] = []
    for name in TRACE_NAMES:
        if name not in rdms or len(rdms[name]) != n_total:
            raise KeyError(f"Missing or malformed trace RDM: {name}")
        target = np.asarray(COMMON.rank_zscore(rdms[name]), dtype=float)[mask]
        residual = residualize(target)
        scale = max(float(np.linalg.norm(target)), 1.0)
        norm = float(np.linalg.norm(residual))
        if norm <= tolerance * scale:
            raise ValueError(
                f"Trace RDM {name} is not estimable beyond controls {kept}; "
                f"dropped aliases={dropped}"
            )
        target_ranked.append(target)
        target_residuals.append(residual)
        target_ss.append(float(np.dot(residual, residual)))
        target_ratios.append(norm / scale)
    return FWLDesign(
        reduced_q=np.column_stack(basis),
        target_residuals=np.column_stack(target_residuals),
        target_residual_ss=np.asarray(target_ss, dtype=float),
        target_ranked=np.column_stack(target_ranked),
        kept_controls=tuple(kept),
        dropped_controls=tuple(dropped),
        target_residual_ratio=tuple(target_ratios),
        n_pairs_total=int(n_total),
        n_pairs_retained=int(np.sum(mask)),
    )


def score_ranked_brain(
    ranked_brain: np.ndarray, design: FWLDesign
) -> dict[str, np.ndarray]:
    y = np.asarray(ranked_brain, dtype=float).reshape(-1)
    if len(y) != design.n_pairs_retained or not np.all(np.isfinite(y)):
        raise ValueError("Ranked brain RDM does not match the FWL design")
    centered_ss = float(np.sum((y - np.mean(y)) ** 2))
    if centered_ss <= 1e-12:
        return {
            name: np.full(len(TRACE_NAMES), np.nan, dtype=float)
            for name in ("conditional_partial_r", "added_beta", "delta_r2")
        }
    projection = design.reduced_q.T @ y
    reduced_ss = max(
        float(np.dot(y, y) - np.dot(projection, projection)), 0.0
    )
    target_dot = design.target_residuals.T @ y
    partial_denominator = np.sqrt(
        np.maximum(design.target_residual_ss * reduced_ss, 0.0)
    )
    partial = np.divide(
        target_dot,
        partial_denominator,
        out=np.zeros_like(target_dot),
        where=partial_denominator > 0,
    )
    improvement = target_dot * target_dot / design.target_residual_ss
    return {
        "conditional_partial_r": partial,
        "added_beta": target_dot / design.target_residual_ss,
        "delta_r2": improvement / centered_ss,
    }


def score_ranked_brain_batch(
    ranked_brain_batch: np.ndarray, design: FWLDesign
) -> np.ndarray:
    """Return null conditional partial-r for draw x trace."""
    y = np.asarray(ranked_brain_batch, dtype=float)
    if y.ndim == 1:
        y = y[None, :]
    if y.shape[1] != design.n_pairs_retained or not np.all(np.isfinite(y)):
        raise ValueError("Null brain batch does not match the FWL design")
    y_ss = np.einsum("ij,ij->i", y, y)
    projection = y @ design.reduced_q
    reduced_ss = np.maximum(
        y_ss - np.einsum("ij,ij->i", projection, projection), 0.0
    )
    target_dot = y @ design.target_residuals
    denominator = np.sqrt(
        reduced_ss[:, None] * design.target_residual_ss[None, :]
    )
    return np.divide(
        target_dot,
        denominator,
        out=np.zeros_like(target_dot),
        where=denominator > 0,
    )


def synchronized_block_orders(
    n_tr: int,
    n_permutations: int,
    block_size: int,
    seed: int,
) -> np.ndarray:
    if n_permutations < 0 or n_permutations > 200:
        raise ValueError("Diagnostic temporal permutations must be in 0..200")
    if block_size < 2:
        raise ValueError("Temporal block size must be at least 2 TR")
    if n_permutations == 0:
        return np.empty((0, int(n_tr)), dtype=np.int32)
    rng = np.random.default_rng(int(seed))
    return np.vstack(
        [
            COMMON.block_permutation_indices(int(n_tr), int(block_size), rng)
            for _ in range(int(n_permutations))
        ]
    ).astype(np.int32)


def analyze_roi_run(
    run: Any,
    roi_name: str,
    design: FWLDesign,
    pair_mask: np.ndarray,
    orders: np.ndarray,
    *,
    permutation_batch_size: int,
    return_exemplar: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, np.ndarray]]:
    patterns, _ = COMMON.load_roi(run, roi_name)
    patterns = _finite_matrix(
        patterns, name=f"ROI {roi_name}", n_rows=int(run.n_tr)
    )
    brain_rdm = np.asarray(COMMON.correlation_rdm(patterns), dtype=float)
    globally_ranked = np.asarray(COMMON.rank_zscore(brain_rdm), dtype=float)
    ranked_brain = globally_ranked[np.asarray(pair_mask, dtype=bool)]
    observed = score_ranked_brain(ranked_brain, design)
    null = np.empty((len(orders), len(TRACE_NAMES)), dtype=np.float32)
    if len(orders):
        square = np.asarray(
            COMMON.square_from_condensed(globally_ranked, int(run.n_tr)),
            dtype=float,
        )
        all_row, all_column = np.triu_indices(int(run.n_tr), k=1)
        mask = np.asarray(pair_mask, dtype=bool)
        pair_row = all_row[mask]
        pair_column = all_column[mask]
        batch_size = max(1, int(permutation_batch_size))
        for start in range(0, len(orders), batch_size):
            stop = min(start + batch_size, len(orders))
            batch_orders = np.asarray(orders[start:stop], dtype=int)
            permuted = square[
                batch_orders[:, pair_row], batch_orders[:, pair_column]
            ]
            null[start:stop] = score_ranked_brain_batch(permuted, design)
    rows: list[dict[str, Any]] = []
    stored: dict[str, np.ndarray] = {}
    for trace_index, trace in enumerate(TRACE_NAMES):
        value = float(observed["conditional_partial_r"][trace_index])
        trace_null = np.asarray(null[:, trace_index], dtype=np.float32)
        p_block = (
            float(
                (1 + np.sum(np.abs(trace_null) >= abs(value)))
                / (1 + len(trace_null))
            )
            if len(trace_null)
            else np.nan
        )
        rows.append(
            {
                "run_label": str(run.run_label),
                "subject": int(run.subject),
                "session": int(run.session),
                "block": int(run.block),
                "game": int(run.game),
                "roi": str(roi_name),
                "trace": trace,
                "trace_label": TRACE_LABELS[trace],
                "n_tr": int(run.n_tr),
                "n_voxels": int(patterns.shape[1]),
                "n_pairs_total": int(design.n_pairs_total),
                "n_pairs_retained": int(design.n_pairs_retained),
                "kept_controls": "|".join(design.kept_controls),
                "dropped_alias_controls": "|".join(
                    design.dropped_controls
                ),
                "target_residual_ratio": float(
                    design.target_residual_ratio[trace_index]
                ),
                "conditional_partial_r": value,
                "added_beta": float(observed["added_beta"][trace_index]),
                "delta_r2_descriptive": float(
                    observed["delta_r2"][trace_index]
                ),
                "p_temporal_block_diagnostic": p_block,
            }
        )
        stored[f"{run.run_label}__{roi_name}__{trace}"] = trace_null
    exemplar = {}
    if return_exemplar:
        exemplar = {
            "ranked_brain": ranked_brain,
            "brain_control_residual": (
                ranked_brain
                - design.reduced_q @ (design.reduced_q.T @ ranked_brain)
            ),
        }
    return rows, stored, exemplar


def add_within_run_temporal_fwer(
    run_effects: pd.DataFrame,
    null_store: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    result = run_effects.copy()
    result["p_temporal_block_maxT_diagnostic"] = np.nan
    for run_label, group in result.groupby("run_label", sort=False):
        arrays: list[np.ndarray] = []
        indices: list[int] = []
        for row_index, row in group.iterrows():
            key = f"{run_label}__{row['roi']}__{row['trace']}"
            if key not in null_store:
                raise KeyError(f"Missing synchronized temporal null: {key}")
            arrays.append(np.asarray(null_store[key], dtype=float))
            indices.append(int(row_index))
        lengths = {len(values) for values in arrays}
        if len(lengths) != 1:
            raise ValueError(f"Unsynchronized temporal nulls in {run_label}")
        n_draws = next(iter(lengths))
        if n_draws == 0:
            continue
        matrix = np.vstack(arrays)
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"Non-finite temporal null in {run_label}")
        max_abs = np.max(np.abs(matrix), axis=0)
        for row_index in indices:
            observed = abs(float(result.at[row_index, "conditional_partial_r"]))
            result.at[row_index, "p_temporal_block_maxT_diagnostic"] = float(
                (1 + np.sum(max_abs >= observed)) / (1 + n_draws)
            )
    return result


def _mean_ci(values: Sequence[float], confidence: float = 0.95) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.size < 2 or not np.all(np.isfinite(array)):
        return (np.nan, np.nan)
    mean = float(np.mean(array))
    se = float(np.std(array, ddof=1) / math.sqrt(array.size))
    critical = float(student_t.ppf(0.5 + confidence / 2.0, array.size - 1))
    return mean - critical * se, mean + critical * se


def validate_run_effect_contract(
    run_effects: pd.DataFrame,
    *,
    expected_runs: int | None = EXPECTED_RUNS,
    expected_rois: int | None = EXPECTED_ROIS,
) -> tuple[list[str], list[str]]:
    required = {
        "run_label", "subject", "game", "roi", "trace",
        "conditional_partial_r", "added_beta", "delta_r2_descriptive",
    }
    missing = sorted(required.difference(run_effects.columns))
    if missing:
        raise ValueError(f"Run-effect table lacks columns: {missing}")
    if run_effects.empty:
        raise ValueError("Run-effect table is empty")
    duplicate = run_effects.duplicated(["run_label", "roi", "trace"], keep=False)
    if np.any(duplicate):
        raise ValueError("Duplicate run/ROI/trace rows")
    traces = list(run_effects["trace"].astype(str).unique())
    if set(traces) != set(TRACE_NAMES):
        raise ValueError(f"Trace contract mismatch: {sorted(traces)}")
    runs = sorted(run_effects["run_label"].astype(str).unique())
    rois = sorted(run_effects["roi"].astype(str).unique())
    if expected_runs is not None and len(runs) != int(expected_runs):
        raise ValueError(f"Expected {expected_runs} runs, found {len(runs)}")
    if expected_rois is not None and len(rois) != int(expected_rois):
        raise ValueError(f"Expected {expected_rois} ROIs, found {len(rois)}")
    expected_rows = len(runs) * len(rois) * len(TRACE_NAMES)
    if len(run_effects) != expected_rows:
        raise ValueError(f"Run row contract failed: {len(run_effects)} != {expected_rows}")
    expected_grid = pd.MultiIndex.from_product(
        [runs, rois, TRACE_NAMES], names=["run_label", "roi", "trace"]
    )
    observed_grid = pd.MultiIndex.from_frame(
        run_effects[["run_label", "roi", "trace"]].astype(str)
    )
    if set(observed_grid) != set(expected_grid):
        raise ValueError("Run-effect table is not a complete run x ROI x trace grid")
    numeric = run_effects[
        ["conditional_partial_r", "added_beta", "delta_r2_descriptive"]
    ].to_numpy(dtype=float)
    if not np.all(np.isfinite(numeric)):
        raise ValueError("Primary run effects contain non-finite values")
    return runs, rois


def summarize_subject(
    run_effects: pd.DataFrame,
    *,
    expected_runs: int | None = EXPECTED_RUNS,
    expected_rois: int | None = EXPECTED_ROIS,
) -> pd.DataFrame:
    runs, rois = validate_run_effect_contract(
        run_effects, expected_runs=expected_runs, expected_rois=expected_rois
    )
    subjects = pd.to_numeric(run_effects["subject"], errors="raise").astype(int)
    games = pd.to_numeric(run_effects["game"], errors="raise").astype(int)
    if subjects.nunique() != 1 or games.nunique() != 1:
        raise ValueError("Subject summary must contain one subject-game cell")
    rows: list[dict[str, Any]] = []
    for roi in rois:
        for trace in TRACE_NAMES:
            selected = run_effects[
                run_effects["roi"].astype(str).eq(roi)
                & run_effects["trace"].astype(str).eq(trace)
            ].sort_values("run_label")
            if list(selected["run_label"].astype(str)) != runs:
                raise ValueError(f"Unequal run contribution for {roi}/{trace}")
            primary = selected["conditional_partial_r"].to_numpy(dtype=float)
            ci_low, ci_high = _mean_ci(primary)
            rows.append({
                "subject": int(subjects.iloc[0]), "game": int(games.iloc[0]),
                "roi": roi, "trace": trace, "trace_label": TRACE_LABELS[trace],
                "n_runs": int(len(runs)),
                "mean_conditional_partial_r": float(np.mean(primary)),
                "sd_across_runs": float(np.std(primary, ddof=1)),
                "se_across_runs": float(np.std(primary, ddof=1) / math.sqrt(len(primary))),
                "ci95_low_across_runs": ci_low, "ci95_high_across_runs": ci_high,
                "positive_run_fraction": float(np.mean(primary > 0)),
                "mean_added_beta": float(selected["added_beta"].mean()),
                "mean_delta_r2_descriptive": float(selected["delta_r2_descriptive"].mean()),
                "median_p_temporal_block_diagnostic": (
                    float(selected["p_temporal_block_diagnostic"].median())
                    if selected["p_temporal_block_diagnostic"].notna().any() else np.nan
                ),
                "median_p_temporal_block_maxT_diagnostic": (
                    float(selected["p_temporal_block_maxT_diagnostic"].median())
                    if selected["p_temporal_block_maxT_diagnostic"].notna().any() else np.nan
                ),
            })
    result = pd.DataFrame(rows)
    if len(result) != len(rois) * len(TRACE_NAMES):
        raise AssertionError("Subject summary row contract failed")
    return result


def _one_sample_t_columns(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.shape[0] < 2:
        raise ValueError("Population t statistic needs at least two subjects")
    mean = np.mean(matrix, axis=0)
    denominator = np.std(matrix, axis=0, ddof=1) / math.sqrt(matrix.shape[0])
    result = np.zeros_like(mean)
    finite_scale = denominator > 1e-15
    result[finite_scale] = mean[finite_scale] / denominator[finite_scale]
    result[(~finite_scale) & (mean > 1e-15)] = np.inf
    result[(~finite_scale) & (mean < -1e-15)] = -np.inf
    return result


def _ge_with_ties(null: np.ndarray, threshold: float) -> np.ndarray:
    return (null > threshold) | np.isclose(
        null, threshold, rtol=SIGNFLIP_RTOL, atol=SIGNFLIP_ATOL, equal_nan=False
    )


def exact_synchronized_signflip(
    values: np.ndarray, effect_names: Sequence[str]
) -> pd.DataFrame:
    """Exact two-sided raw and synchronized max-|t| population inference."""
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(effect_names):
        raise ValueError("Sign-flip effect matrix/name mismatch")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Exact sign-flip requires a complete finite matrix")
    n_subjects = matrix.shape[0]
    if n_subjects != len(EXPECTED_SUBJECTS):
        raise ValueError(f"Confirmatory inference requires 6 subjects, got {n_subjects}")
    signs = np.asarray(
        list(itertools.product((-1.0, 1.0), repeat=n_subjects)), dtype=float
    )
    observed_t = _one_sample_t_columns(matrix)
    null_t = np.empty((len(signs), matrix.shape[1]), dtype=float)
    for index, sign in enumerate(signs):
        null_t[index] = _one_sample_t_columns(matrix * sign[:, None])
    absolute_null = np.abs(null_t)
    max_null = np.max(absolute_null, axis=1)
    rows: list[dict[str, Any]] = []
    for column, name in enumerate(effect_names):
        threshold = abs(float(observed_t[column]))
        ci_low, ci_high = _mean_ci(matrix[:, column])
        rows.append({
            "effect_key": str(name), "n_subjects": int(n_subjects),
            "mean_effect": float(np.mean(matrix[:, column])),
            "sd_subject": float(np.std(matrix[:, column], ddof=1)),
            "ci95_low": ci_low, "ci95_high": ci_high,
            "t_stat": float(observed_t[column]),
            "positive_fraction": float(np.mean(matrix[:, column] > 0)),
            "p_exact_two_sided": float(np.mean(_ge_with_ties(absolute_null[:, column], threshold))),
            "p_max_abs_t_fwer": float(np.mean(_ge_with_ties(max_null, threshold))),
            "n_sign_patterns": int(len(signs)),
        })
    return pd.DataFrame(rows)


def fdr_bh(values: Sequence[float]) -> np.ndarray:
    if hasattr(COMMON, "fdr_bh"):
        return np.asarray(COMMON.fdr_bh(values), dtype=float)
    p = np.asarray(values, dtype=float)
    result = np.full_like(p, np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    if finite.size == 0:
        return result
    order = finite[np.argsort(p[finite])]
    adjusted = p[order] * len(order) / np.arange(1, len(order) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result[order] = np.minimum(adjusted, 1.0)
    return result


def validate_subject_effect_contract(
    subject_effects: pd.DataFrame,
    *, expected_subjects: Sequence[int] = EXPECTED_SUBJECTS,
    expected_rois: int = EXPECTED_ROIS,
) -> tuple[list[int], list[str]]:
    required = {"subject", "game", "roi", "trace", "n_runs", "mean_conditional_partial_r"}
    missing = sorted(required.difference(subject_effects.columns))
    if missing:
        raise ValueError(f"Subject-effect table lacks columns: {missing}")
    subjects = sorted(pd.to_numeric(subject_effects["subject"], errors="raise").astype(int).unique())
    if subjects != list(expected_subjects):
        raise ValueError(f"Expected subjects {list(expected_subjects)}, found {subjects}")
    rois = sorted(subject_effects["roi"].astype(str).unique())
    if len(rois) != int(expected_rois):
        raise ValueError(f"Expected {expected_rois} ROIs, found {len(rois)}")
    if set(subject_effects["trace"].astype(str)) != set(TRACE_NAMES):
        raise ValueError("Subject trace-name contract failed")
    if np.any(subject_effects.duplicated(["subject", "roi", "trace"], keep=False)):
        raise ValueError("Duplicate subject/ROI/trace effects")
    expected_rows = len(subjects) * len(rois) * len(TRACE_NAMES)
    if len(subject_effects) != expected_rows:
        raise ValueError(f"Subject row contract failed: {len(subject_effects)} != {expected_rows}")
    if not pd.to_numeric(subject_effects["n_runs"], errors="raise").eq(EXPECTED_RUNS).all():
        raise ValueError("Every population input must average exactly 11 runs")
    values = pd.to_numeric(subject_effects["mean_conditional_partial_r"], errors="raise").to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Population inputs contain non-finite effects")
    return subjects, rois


def group_inference(subject_effects: pd.DataFrame) -> pd.DataFrame:
    subjects, rois = validate_subject_effect_contract(subject_effects)
    columns = [(roi, trace) for roi in rois for trace in TRACE_NAMES]
    pivot = subject_effects.pivot(
        index="subject", columns=["roi", "trace"], values="mean_conditional_partial_r"
    ).reindex(index=subjects, columns=pd.MultiIndex.from_tuples(columns))
    if pivot.isna().any().any():
        raise ValueError("Population effect matrix is incomplete")
    names = [f"{roi}\t{trace}" for roi, trace in columns]
    result = exact_synchronized_signflip(pivot.to_numpy(dtype=float), names)
    lookup = {f"{roi}\t{trace}": (roi, trace) for roi, trace in columns}
    result["roi"] = result["effect_key"].map(lambda key: lookup[str(key)][0])
    result["trace"] = result["effect_key"].map(lambda key: lookup[str(key)][1])
    result["trace_label"] = result["trace"].map(TRACE_LABELS)
    result["q_bh_family"] = fdr_bh(result["p_exact_two_sided"])
    result["inference_family"] = "planning_trace_2x110"
    result["primary_estimand"] = "conditional_partial_spearman_r"
    return result.drop(columns=["effect_key"])


def build_paired_effects(game1: pd.DataFrame, game2: pd.DataFrame) -> pd.DataFrame:
    validate_subject_effect_contract(game1)
    validate_subject_effect_contract(game2)
    keys = ["subject", "roi", "trace"]
    columns = [*keys, "mean_conditional_partial_r"]
    merged = game1[columns].merge(
        game2[columns], on=keys, how="inner", validate="one_to_one",
        suffixes=("_game1", "_game2")
    )
    expected_rows = len(EXPECTED_SUBJECTS) * EXPECTED_ROIS * len(TRACE_NAMES)
    if len(merged) != expected_rows:
        raise ValueError(f"Paired game row contract failed: {len(merged)}")
    merged["across_game_common"] = 0.5 * (
        merged["mean_conditional_partial_r_game1"] + merged["mean_conditional_partial_r_game2"]
    )
    merged["game2_minus_game1"] = (
        merged["mean_conditional_partial_r_game2"] - merged["mean_conditional_partial_r_game1"]
    )
    return merged


def paired_inference(paired: pd.DataFrame) -> pd.DataFrame:
    rois = sorted(paired["roi"].astype(str).unique())
    if len(rois) != EXPECTED_ROIS:
        raise ValueError("Paired inference requires 110 synchronized ROIs")
    columns = [(roi, trace) for roi in rois for trace in TRACE_NAMES]
    frames: list[pd.DataFrame] = []
    for contrast in ("across_game_common", "game2_minus_game1"):
        pivot = paired.pivot(index="subject", columns=["roi", "trace"], values=contrast).reindex(
            index=list(EXPECTED_SUBJECTS), columns=pd.MultiIndex.from_tuples(columns)
        )
        if pivot.isna().any().any():
            raise ValueError(f"Incomplete paired contrast matrix: {contrast}")
        names = [f"{roi}\t{trace}" for roi, trace in columns]
        inference = exact_synchronized_signflip(pivot.to_numpy(dtype=float), names)
        lookup = {f"{roi}\t{trace}": (roi, trace) for roi, trace in columns}
        inference["roi"] = inference["effect_key"].map(lambda key: lookup[str(key)][0])
        inference["trace"] = inference["effect_key"].map(lambda key: lookup[str(key)][1])
        inference["trace_label"] = inference["trace"].map(TRACE_LABELS)
        inference["contrast"] = contrast
        inference["q_bh_family"] = fdr_bh(inference["p_exact_two_sided"])
        inference["inference_family"] = "planning_trace_2x110"
        inference["primary_estimand"] = "conditional_partial_spearman_r"
        frames.append(inference.drop(columns=["effect_key"]))
    return pd.concat(frames, ignore_index=True)


def plot_residual_rdm_scatter(
    exemplar: Mapping[str, Mapping[str, np.ndarray]],
    output_path: Path,
    *,
    title: str,
    random_state: int,
) -> None:
    """Descriptive pair-distance view; pair dots are not inferential units."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharex=True, sharey=True)
    rng = np.random.default_rng(int(random_state))
    for axis, trace, color in zip(axes, TRACE_NAMES, (BLUE, ORANGE)):
        x = np.asarray(exemplar[trace]["trace_residual"], dtype=float)
        y = np.asarray(exemplar[trace]["brain_residual"], dtype=float)
        if len(x) != len(y) or len(x) < 8:
            raise ValueError("Exemplar residual-distance arrays are malformed")
        n_show = min(3500, len(x))
        selected = np.sort(rng.choice(len(x), n_show, replace=False))
        axis.scatter(
            x[selected], y[selected], s=5, alpha=0.08, color=color,
            edgecolors="none", rasterized=True,
        )
        quantile_edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, 17)))
        centers: list[float] = []
        medians: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        for lower, upper in zip(quantile_edges[:-1], quantile_edges[1:]):
            mask = (x >= lower) & (
                x <= upper if upper == quantile_edges[-1] else x < upper
            )
            if int(np.sum(mask)) < 2:
                continue
            centers.append(float(np.median(x[mask])))
            medians.append(float(np.median(y[mask])))
            lows.append(float(np.quantile(y[mask], 0.25)))
            highs.append(float(np.quantile(y[mask], 0.75)))
        center_array = np.asarray(centers)
        median_array = np.asarray(medians)
        axis.plot(center_array, median_array, color=INK, linewidth=1.8, zorder=4)
        axis.fill_between(
            center_array, np.asarray(lows), np.asarray(highs),
            color=color, alpha=0.18, linewidth=0,
        )
        partial_r = float(np.corrcoef(x, y)[0, 1])
        axis.axhline(0.0, color=GRID, linewidth=0.8)
        axis.axvline(0.0, color=GRID, linewidth=0.8)
        axis.set_title(f"{TRACE_LABELS[trace]}  ·  partial r={partial_r:.3f}")
        axis.set_xlabel("Trace RDM rank residual")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Neural RDM rank residual")
    fig.suptitle(title, fontsize=12.5)
    fig.text(
        0.5, -0.01,
        "Dots and quantile medians/IQR are descriptive RDM pairs; population inference uses subjects.",
        ha="center", va="top", fontsize=8.5, color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    save_figure(fig, output_path)


def plot_ranked_lollipop(
    inference: pd.DataFrame,
    subject_inputs: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    value_column: str = "mean_conditional_partial_r",
) -> None:
    """All-ROI effects, retaining six subject points and corrected markers."""
    required = {
        "roi", "trace", "mean_effect", "ci95_low", "ci95_high",
        "p_max_abs_t_fwer",
    }
    if required.difference(inference.columns):
        raise ValueError("Inference table lacks lollipop columns")
    pivot = inference.pivot(index="roi", columns="trace", values="mean_effect")
    order = (
        pivot.reindex(columns=list(TRACE_NAMES)).abs().max(axis=1)
        .sort_values(ascending=True).index.tolist()
    )
    height = max(9.0, 0.205 * len(order) + 2.2)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, height), sharey=True)
    all_effects = inference["mean_effect"].to_numpy(dtype=float)
    all_ci = inference[["ci95_low", "ci95_high"]].to_numpy(dtype=float).reshape(-1)
    subject_values = subject_inputs[value_column].to_numpy(dtype=float)
    finite = np.concatenate((all_effects, all_ci, subject_values))
    finite = finite[np.isfinite(finite)]
    limit = max(float(np.max(np.abs(finite))) * 1.08 if finite.size else 0.1, 0.025)
    y = np.arange(len(order), dtype=float)
    subject_ids = sorted(
        pd.to_numeric(subject_inputs["subject"], errors="raise").astype(int).unique()
    )
    offsets = dict(zip(subject_ids, np.linspace(-0.11, 0.11, len(subject_ids))))
    for axis, trace, color, marker in zip(
        axes, TRACE_NAMES, (BLUE, ORANGE), ("o", "s")
    ):
        trace_inference = (
            inference[inference["trace"].astype(str).eq(trace)]
            .set_index("roi").loc[order]
        )
        means = trace_inference["mean_effect"].to_numpy(dtype=float)
        lower = trace_inference["ci95_low"].to_numpy(dtype=float)
        upper = trace_inference["ci95_high"].to_numpy(dtype=float)
        axis.hlines(y, 0.0, means, color=color, alpha=0.45, linewidth=1.0)
        axis.errorbar(
            means, y, xerr=np.vstack((means - lower, upper - means)),
            fmt="none", ecolor=INK, elinewidth=0.8, capsize=1.5, zorder=3,
        )
        for subject in subject_ids:
            dots = subject_inputs[
                pd.to_numeric(subject_inputs["subject"], errors="coerce").eq(subject)
                & subject_inputs["trace"].astype(str).eq(trace)
            ].set_index("roi").loc[order]
            axis.scatter(
                dots[value_column].to_numpy(dtype=float),
                y + offsets[subject], s=8, color=MUTED, alpha=0.45,
                edgecolors="none", zorder=2,
            )
        significant = (
            trace_inference["p_max_abs_t_fwer"].to_numpy(dtype=float) < 0.05
        )
        axis.scatter(
            means, y, s=30, color=color, marker=marker, edgecolors="white",
            linewidths=0.5, zorder=4,
        )
        if np.any(significant):
            axis.scatter(
                means[significant], y[significant], s=90, marker="*",
                facecolors="none", edgecolors=INK, linewidths=0.9, zorder=5,
                label="family-maxT < .05",
            )
        axis.axvline(0.0, color=INK, linewidth=0.8)
        axis.set_xlim(-limit, limit)
        axis.set_ylim(-0.8, len(order) - 0.2)
        axis.set_title(TRACE_LABELS[trace])
        axis.set_xlabel("Mean conditional partial Spearman r")
        axis.grid(axis="x", color=GRID, linewidth=0.5, alpha=0.7)
        axis.spines[["top", "right", "left"]].set_visible(False)
        if np.any(significant):
            axis.legend(loc="lower right", frameon=False, fontsize=8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(
        [clean_roi_label(roi) for roi in order], fontsize=6.3
    )
    axes[1].tick_params(axis="y", labelleft=False)
    fig.suptitle(title, fontsize=13)
    fig.text(
        0.5, 0.006,
        "Grey dots: six subject means; bars: descriptive 95% t CI; star outline: synchronized 2×110 max-|t| FWER < .05.",
        ha="center", fontsize=8.5, color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.018, 1, 0.975))
    save_figure(fig, output_path)


def plot_concat_vs_mean(
    inference: pd.DataFrame, output_path: Path, *, title: str
) -> None:
    pivot = inference.pivot(index="roi", columns="trace", values="mean_effect")
    pmax = inference.pivot(
        index="roi", columns="trace", values="p_max_abs_t_fwer"
    )
    if not set(TRACE_NAMES).issubset(pivot.columns):
        raise ValueError("Concat-versus-mean plot lacks a trace")
    x = pivot["planning_mean29"].to_numpy(dtype=float)
    y = pivot["planning_concat29"].to_numpy(dtype=float)
    significant = (
        pmax.reindex(pivot.index)[list(TRACE_NAMES)] < 0.05
    ).any(axis=1).to_numpy()
    finite = np.concatenate((x, y))
    limit = max(float(np.max(np.abs(finite))) * 1.08, 0.025)
    fig, axis = plt.subplots(figsize=(6.3, 6.0))
    axis.scatter(
        x[~significant], y[~significant], s=28, color=BLUE, alpha=0.68,
        edgecolors="white", linewidths=0.35, label="ROI",
    )
    if np.any(significant):
        axis.scatter(
            x[significant], y[significant], s=95, marker="*", color=ORANGE,
            edgecolors=INK, linewidths=0.5, label="maxT < .05 in either trace",
        )
        for roi, x_value, y_value in zip(
            pivot.index[significant], x[significant], y[significant]
        ):
            axis.annotate(
                clean_roi_label(roi), (x_value, y_value), xytext=(4, 4),
                textcoords="offset points", fontsize=7,
            )
    axis.plot(
        [-limit, limit], [-limit, limit], color=MUTED, linestyle="--",
        linewidth=1.0, label="identity",
    )
    axis.axhline(0.0, color=INK, linewidth=0.7)
    axis.axvline(0.0, color=INK, linewidth=0.7)
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Mean29 conditional partial Spearman r")
    axis.set_ylabel("Ordered concat29 conditional partial Spearman r")
    axis.set_title(title)
    axis.grid(color=GRID, linewidth=0.45, alpha=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    save_figure(fig, output_path)


def analysis_settings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "analysis_version": ANALYSIS_VERSION,
        "estimator_version": ESTIMATOR_VERSION,
        "trace_names": list(TRACE_NAMES),
        "trace_widths": dict(TRACE_WIDTHS),
        "trace_rdm": "correlation_distance",
        "neural_rdm": "correlation_distance",
        "rank_transform": "global_average_ranks_then_lag_mask",
        "controls": list(CONTROL_NAMES),
        "decision_event_control": (
            "canonical process decision_event: generic decision-anchor HRF "
            "mass, independent of planning-trace content"
        ),
        "targets_tested_separately": True,
        "min_pair_lag_tr": int(args.min_pair_lag),
        "n_temporal_block_permutations": int(args.n_permutations),
        "temporal_block_size_tr": int(args.permutation_block_size),
        "random_state": int(args.random_state),
        "temporal_null_role": "diagnostic_not_population_inference",
        "primary_estimand": "conditional_partial_spearman_r",
        "subject_aggregation": "equal_mean_of_exactly_11_runs",
        "population_unit": "subject",
        "n_population_units": 6,
        "population_null": "all_64_subject_sign_patterns",
        "multiple_comparison": "synchronized_max_abs_t_across_2_traces_x_110_rois",
        "expected_runs": EXPECTED_RUNS,
        "expected_rois": EXPECTED_ROIS,
        "roi_set": str(args.roi_set),
        "roi_include": (
            str(args.roi_include) if args.roi_include is not None else None
        ),
        "max_roi_voxels": int(args.max_roi_voxels),
        "required_status2": int(args.required_status2),
    }


def current_source_contract(adapter: TraceLoaderAdapter = TRACE_ADAPTER) -> dict[str, Any]:
    return {
        "analysis_script_sha256": sha256_file(Path(__file__)),
        "planning_brain_common_sha256": sha256_file(COMMON_PATH),
        "planning_trace_common_sha256": adapter.source_sha256,
    }


def canonical_run_contract(
    run_specs: Sequence[Any],
    roi_names: Sequence[str],
    *,
    allow_noncanonical: bool,
) -> dict[str, Any]:
    labels = [str(run.run_label) for run in run_specs]
    if len(labels) != len(set(labels)):
        raise ValueError("Strict trace preparation returned duplicate run labels")
    if not labels:
        raise ValueError("No runs available")
    if any(int(run.n_tr) != 480 for run in run_specs):
        raise ValueError("Planning-trace RSA requires exactly 480 aligned TRs/run")
    if len(roi_names) != len(set(roi_names)):
        raise ValueError("ROI names are not unique")
    if not allow_noncanonical:
        if len(run_specs) != EXPECTED_RUNS:
            raise ValueError(
                f"Canonical subject analysis requires {EXPECTED_RUNS} runs, "
                f"found {len(run_specs)}"
            )
        if len(roi_names) != EXPECTED_ROIS:
            raise ValueError(
                f"Canonical subject analysis requires {EXPECTED_ROIS} ROIs, "
                f"found {len(roi_names)}"
            )
    return {
        "run_labels": labels,
        "n_runs": int(len(labels)),
        "roi_names": list(roi_names),
        "n_rois": int(len(roi_names)),
        "n_tr_per_run": [int(run.n_tr) for run in run_specs],
        "canonical_full_v1": bool(
            len(labels) == EXPECTED_RUNS
            and len(roi_names) == EXPECTED_ROIS
            and all(int(run.n_tr) == 480 for run in run_specs)
        ),
    }


def choose_exemplar_roi(roi_names: Sequence[str], token: str) -> str:
    matches = [
        roi for roi in roi_names
        if str(token).lower().replace(" ", "_") in str(roi).lower()
    ]
    return sorted(matches)[0] if matches else sorted(roi_names)[0]


def subject_output_root(args: argparse.Namespace) -> Path:
    return (
        Path(args.output_root)
        / f"sub{int(args.subject):03d}_game{int(args.game)}"
    )


def run_subject_analysis(args: argparse.Namespace) -> None:
    run_specs, cache_audit, trace_audit = TRACE_ADAPTER.prepare(args)
    roi_names = list(COMMON.common_roi_names(run_specs))
    run_contract = canonical_run_contract(
        run_specs, roi_names,
        allow_noncanonical=bool(args.diagnostic_noncanonical),
    )
    output_dir = subject_output_root(args)
    result_dir = output_dir / "results"
    figure_dir = output_dir / "figures"
    rdm_dir = output_dir / "rdms"
    for directory in (result_dir, figure_dir, rdm_dir):
        directory.mkdir(parents=True, exist_ok=True)

    settings = analysis_settings(args)
    settings_hash = stable_hash(settings)
    source_contract = current_source_contract()
    exemplar_roi = choose_exemplar_roi(roi_names, args.exemplar_roi)
    all_rows: list[dict[str, Any]] = []
    null_store: dict[str, np.ndarray] = {}
    rdm_rows: list[dict[str, Any]] = []
    design_rows: list[dict[str, Any]] = []
    trace_run_audits: list[dict[str, Any]] = []
    exemplar_payload: dict[str, dict[str, np.ndarray]] | None = None
    exemplar_run_label: str | None = None

    for run_index, run in enumerate(run_specs, start=1):
        log(f"[run {run_index}/{len(run_specs)}] {run.run_label}")
        blocks, block_audit = load_run_blocks(
            run, planning_trace_root=args.planning_trace_root
        )
        trace_run_audits.append(block_audit)
        rdms = build_rdms(blocks)
        pair_mask = np.asarray(
            COMMON.pair_mask_min_lag(int(run.n_tr), int(args.min_pair_lag)),
            dtype=bool,
        )
        design = prepare_fwl_design(rdms, pair_mask)
        seed = (
            int(args.random_state)
            + 1_000_003 * int(args.game)
            + 1009 * int(run.session)
            + 100_003 * int(run.block)
        )
        orders = synchronized_block_orders(
            int(run.n_tr), int(args.n_permutations),
            int(args.permutation_block_size), seed,
        )
        rdm_archive = (
            rdm_dir / f"planning_trace_rdms_{run.run_label}.npz"
        )
        atomic_save_npz(
            rdm_archive,
            **{name: np.asarray(values, dtype=np.float32) for name, values in rdms.items()},
        )
        rdm_archive_sha256 = sha256_file(rdm_archive)
        for name, values in rdms.items():
            rdm_rows.append({
                "run_label": str(run.run_label), "rdm": name,
                "n_tr": int(run.n_tr), "n_pairs": int(len(values)),
                "rdm_archive_path": str(rdm_archive.resolve()),
                "rdm_archive_sha256": rdm_archive_sha256,
                "rdm_role": (
                    "target" if name in TRACE_NAMES else
                    "nuisance_or_presearch_control"
                ),
            })
        for trace_index, trace in enumerate(TRACE_NAMES):
            design_rows.append({
                "run_label": str(run.run_label), "trace": trace,
                "kept_controls": "|".join(design.kept_controls),
                "dropped_alias_controls": "|".join(design.dropped_controls),
                "target_residual_ratio": float(design.target_residual_ratio[trace_index]),
                "n_pairs_total": int(design.n_pairs_total),
                "n_pairs_retained": int(design.n_pairs_retained),
                "reduced_rank": int(design.reduced_q.shape[1]),
                "permutation_seed": int(seed),
                "n_temporal_permutations": int(len(orders)),
            })

        def analyze_one(roi_name: str):
            return analyze_roi_run(
                run, roi_name, design, pair_mask, orders,
                permutation_batch_size=int(args.permutation_batch_size),
                return_exemplar=(
                    run_index == 1 and str(roi_name) == str(exemplar_roi)
                ),
            )

        if int(args.n_jobs) == 1:
            roi_results = []
            for roi_index, roi_name in enumerate(roi_names, start=1):
                if roi_index == 1 or roi_index % 10 == 0 or roi_index == len(roi_names):
                    log(f"  [ROI {roi_index}/{len(roi_names)}] {roi_name}")
                roi_results.append(analyze_one(roi_name))
        else:
            results_by_index: dict[int, Any] = {}
            with ThreadPoolExecutor(
                max_workers=min(int(args.n_jobs), len(roi_names))
            ) as executor:
                futures = {
                    executor.submit(analyze_one, roi_name): index
                    for index, roi_name in enumerate(roi_names)
                }
                for completed, future in enumerate(as_completed(futures), start=1):
                    results_by_index[futures[future]] = future.result()
                    if completed % 10 == 0 or completed == len(roi_names):
                        log(
                            f"  [ROI {completed}/{len(roi_names)}] "
                            f"workers={min(int(args.n_jobs), len(roi_names))}"
                        )
            roi_results = [
                results_by_index[index] for index in range(len(roi_names))
            ]
        for roi_name, (rows, stored, exemplar) in zip(roi_names, roi_results):
            all_rows.extend(rows)
            null_store.update(stored)
            if exemplar:
                brain_residual = np.asarray(
                    exemplar["brain_control_residual"], dtype=float
                )
                exemplar_payload = {
                    trace: {
                        "trace_residual": np.asarray(
                            design.target_residuals[:, trace_index], dtype=float
                        ),
                        "brain_residual": brain_residual,
                    }
                    for trace_index, trace in enumerate(TRACE_NAMES)
                }
                exemplar_run_label = str(run.run_label)

    run_effects = add_within_run_temporal_fwer(
        pd.DataFrame(all_rows), null_store
    )
    expected_runs = None if args.diagnostic_noncanonical else EXPECTED_RUNS
    expected_rois = None if args.diagnostic_noncanonical else EXPECTED_ROIS
    subject_effects = summarize_subject(
        run_effects, expected_runs=expected_runs, expected_rois=expected_rois
    )
    run_path = result_dir / "planning_trace_run_effects.csv"
    subject_path = result_dir / "subject_effects.csv"
    rdm_manifest_path = result_dir / "rdm_manifest.csv"
    design_path = result_dir / "design_rank_audit.csv"
    trace_audit_path = result_dir / "trace_run_audit.json"
    atomic_write_csv(run_path, run_effects)
    atomic_write_csv(subject_path, subject_effects)
    atomic_write_csv(rdm_manifest_path, pd.DataFrame(rdm_rows))
    atomic_write_csv(design_path, pd.DataFrame(design_rows))
    atomic_write_json(trace_audit_path, {"runs": trace_run_audits})
    null_path = result_dir / "temporal_block_nulls_diagnostic.npz"
    if null_store:
        atomic_save_npz(
            null_path, **null_store
        )
    brain_maps_only = bool(getattr(args, "brain_maps_only", False))
    if not brain_maps_only:
        if exemplar_payload is None or exemplar_run_label is None:
            raise RuntimeError("Prespecified exemplar ROI/run was not analyzed")
        plot_residual_rdm_scatter(
            exemplar_payload,
            figure_dir / "residual_rdm_distance_scatter.png",
            title=(
                f"Residual RDM distances · sub{int(args.subject):03d} "
                f"game {int(args.game)} · {clean_roi_label(exemplar_roi)} · "
                f"{exemplar_run_label}"
            ),
            random_state=int(args.random_state),
        )
    figure_png = figure_dir / "residual_rdm_distance_scatter.png"
    figure_svg = figure_dir / "residual_rdm_distance_scatter.svg"

    artifacts = {
        "planning_trace_run_effects.csv": {
            "sha256": sha256_file(run_path), "rows": int(len(run_effects))
        },
        "subject_effects.csv": {
            "sha256": sha256_file(subject_path), "rows": int(len(subject_effects))
        },
        "rdm_manifest.csv": {
            "sha256": sha256_file(rdm_manifest_path), "rows": int(len(rdm_rows))
        },
        "design_rank_audit.csv": {
            "sha256": sha256_file(design_path), "rows": int(len(design_rows))
        },
        "trace_run_audit.json": {"sha256": sha256_file(trace_audit_path)},
    }
    if null_path.is_file():
        artifacts["temporal_block_nulls_diagnostic.npz"] = {
            "sha256": sha256_file(null_path),
            "arrays": int(len(null_store)),
            "draws_per_array": int(args.n_permutations),
        }
    if figure_png.is_file() and figure_svg.is_file():
        artifacts["residual_rdm_distance_scatter.png"] = {
            "sha256": sha256_file(figure_png)
        }
        artifacts["residual_rdm_distance_scatter.svg"] = {
            "sha256": sha256_file(figure_svg)
        }
    manifest = {
        "analysis": ANALYSIS_NAME,
        "analysis_version": ANALYSIS_VERSION,
        "schema_version": 1,
        "subject": int(args.subject),
        "game": int(args.game),
        "settings": settings,
        "settings_sha256": settings_hash,
        "source_contract": source_contract,
        "run_contract": run_contract,
        "trace_definition": {
            "planning_concat29": (
                "29 event-major 12-component vectors concatenated; "
                "order/event identity retained"
            ),
            "planning_mean29": (
                "mean of the same 29 12-component event vectors; "
                "order/event identity discarded"
            ),
            "alignment": (
                "decision-anchor impulse at native 50 Hz, canonical HRF, "
                "then sampled at 480 scanner TRs"
            ),
        },
        "controls": list(CONTROL_NAMES),
        "decision_event_control": (
            "canonical generic decision-density HRF RDM; prevents the "
            "parametric planning trace from inheriting decision-anchor mass"
        ),
        "primary_inference": (
            "population only: six subject means; exact 64 sign flips; "
            "synchronized max-|t| across 2 traces x 110 ROIs"
        ),
        "within_run_temporal_null": (
            f"{int(args.n_permutations)} synchronized "
            f"{int(args.permutation_block_size)}-TR block permutations; "
            "diagnostic only"
        ),
        "figure_contract": (
            "MNI brain maps are generated by the unified 07_RSA.py entry point"
            if brain_maps_only
            else "legacy descriptive residual-distance scatter"
        ),
        "brain_cache_audit": cache_audit,
        "planning_trace_audit": trace_audit,
        "artifacts": artifacts,
    }
    atomic_write_json(result_dir / "analysis_manifest.json", manifest)
    (output_dir / "summary.md").write_text(
        "\n".join([
            f"# sub{int(args.subject):03d} game{int(args.game)} planning-trace RSA",
            "",
            f"- Runs: {len(run_specs)} (equal weight); ROIs: {len(roi_names)}",
            "- Targets: ordered concat29 and mean29, tested separately.",
            (
                "- Controls: visual, generic decision-event HRF density, "
                "current/previous action, task history, drift, temporal lag, "
                "and pre-search P/M/Q0."
            ),
            (
                "- Primary estimand: run-wise conditional partial Spearman r; "
                "subject effect is the equal mean over runs."
            ),
            (
                "- Temporal block permutations are diagnostic. Population "
                "claims require six-subject exact max-|t| inference."
            ),
        ]) + "\n",
        encoding="utf-8",
    )
    log(f"Subject analysis complete -> {output_dir}")


def read_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def load_population_inputs(
    output_root: Path, game: int
) -> tuple[pd.DataFrame, list[dict[str, Any]], str, dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    input_rows: list[dict[str, Any]] = []
    settings_hash: str | None = None
    settings: dict[str, Any] | None = None
    current_sources = current_source_contract()
    expected_key_order: list[tuple[str, str]] | None = None
    for subject in EXPECTED_SUBJECTS:
        result_dir = (
            Path(output_root) / f"sub{subject:03d}_game{int(game)}" / "results"
        )
        effect_path = result_dir / "subject_effects.csv"
        manifest_path = result_dir / "analysis_manifest.json"
        if not effect_path.exists() or not manifest_path.exists():
            raise FileNotFoundError(
                f"Missing canonical subject input for sub{subject:03d} game{game}"
            )
        manifest = read_json(manifest_path)
        if manifest.get("analysis") != ANALYSIS_NAME:
            raise ValueError(f"Wrong analysis manifest: {manifest_path}")
        if int(manifest.get("schema_version", -1)) != 1:
            raise ValueError(f"Unsupported subject schema: {manifest_path}")
        if int(manifest.get("subject", -1)) != subject or int(
            manifest.get("game", -1)
        ) != int(game):
            raise ValueError(f"Subject/game manifest mismatch: {manifest_path}")
        if manifest.get("source_contract") != current_sources:
            raise ValueError(
                f"Frozen source contract mismatch in {manifest_path}; "
                "rerun the subject under the current source"
            )
        run_contract = manifest.get("run_contract", {})
        if not bool(run_contract.get("canonical_full_v1", False)):
            raise ValueError(f"Noncanonical subject input: {manifest_path}")
        if int(run_contract.get("n_runs", -1)) != EXPECTED_RUNS or int(
            run_contract.get("n_rois", -1)
        ) != EXPECTED_ROIS:
            raise ValueError(f"Run/ROI contract mismatch: {manifest_path}")
        observed_settings_hash = str(manifest.get("settings_sha256", ""))
        observed_settings = manifest.get("settings")
        if not isinstance(observed_settings, dict):
            raise ValueError(f"Manifest lacks settings: {manifest_path}")
        if stable_hash(observed_settings) != observed_settings_hash:
            raise ValueError(f"Settings hash mismatch: {manifest_path}")
        if settings_hash is None:
            settings_hash = observed_settings_hash
            settings = observed_settings
        elif observed_settings_hash != settings_hash or observed_settings != settings:
            raise ValueError("Subject analysis settings differ within the group")
        declared = (
            manifest.get("artifacts", {})
            .get("subject_effects.csv", {})
            .get("sha256")
        )
        if str(declared) != sha256_file(effect_path):
            raise ValueError(f"Subject effect artifact hash mismatch: {effect_path}")
        frame = pd.read_csv(effect_path)
        if set(pd.to_numeric(frame["subject"], errors="raise").astype(int)) != {subject}:
            raise ValueError(f"Subject column mismatch: {effect_path}")
        if set(pd.to_numeric(frame["game"], errors="raise").astype(int)) != {int(game)}:
            raise ValueError(f"Game column mismatch: {effect_path}")
        if len(frame) != EXPECTED_ROIS * len(TRACE_NAMES):
            raise ValueError(f"Subject row count mismatch: {effect_path}")
        key_order = list(
            frame[["roi", "trace"]].astype(str).itertuples(index=False, name=None)
        )
        if expected_key_order is None:
            expected_key_order = key_order
        elif key_order != expected_key_order:
            raise ValueError("ROI/trace row order is not synchronized across subjects")
        frames.append(frame)
        input_rows.append({
            "subject": subject, "game": int(game),
            "subject_effects_path": str(effect_path.resolve()),
            "subject_effects_sha256": sha256_file(effect_path),
            "subject_manifest_path": str(manifest_path.resolve()),
            "subject_manifest_sha256": sha256_file(manifest_path),
            "settings_sha256": observed_settings_hash,
            "n_runs": int(run_contract["n_runs"]),
            "n_rois": int(run_contract["n_rois"]),
        })
    combined = pd.concat(frames, ignore_index=True)
    validate_subject_effect_contract(combined)
    if settings_hash is None or settings is None:
        raise AssertionError("Population input settings were not initialized")
    return combined, input_rows, settings_hash, settings


def run_group_analysis(args: argparse.Namespace) -> None:
    subject_inputs, input_rows, settings_hash, settings = load_population_inputs(
        args.output_root, int(args.game)
    )
    inference = group_inference(subject_inputs)
    if len(inference) != EXPECTED_ROIS * len(TRACE_NAMES):
        raise AssertionError("Group inference row contract failed")
    output_dir = Path(args.output_root) / f"group_game{int(args.game)}"
    result_dir = output_dir / "results"
    figure_dir = output_dir / "figures"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    subject_path = result_dir / "subject_inputs.csv"
    inference_path = result_dir / "group_inference.csv"
    inputs_path = result_dir / "input_manifest.csv"
    atomic_write_csv(subject_path, subject_inputs)
    atomic_write_csv(inference_path, inference)
    atomic_write_csv(inputs_path, pd.DataFrame(input_rows))
    plot_ranked_lollipop(
        inference, subject_inputs,
        figure_dir / "ranked_trace_rsa_lollipop.png",
        title=(
            f"Planning-trace RSA · Game {int(args.game)} · "
            "all Harvard–Oxford ROIs"
        ),
    )
    plot_concat_vs_mean(
        inference,
        figure_dir / "concat_vs_mean_roi_scatter.png",
        title=f"Ordered trace versus event mean · Game {int(args.game)}",
    )
    manifest = {
        "analysis": ANALYSIS_NAME,
        "analysis_version": ANALYSIS_VERSION,
        "schema_version": 1,
        "aggregation": "group",
        "game": int(args.game),
        "source_contract": current_source_contract(),
        "settings": settings,
        "settings_sha256": settings_hash,
        "n_subjects": len(EXPECTED_SUBJECTS),
        "n_rois": EXPECTED_ROIS,
        "n_traces": len(TRACE_NAMES),
        "n_tests_in_synchronized_family": EXPECTED_ROIS * len(TRACE_NAMES),
        "n_sign_patterns": 64,
        "correction": "two-sided exact synchronized max-|t| across 2x110",
        "artifacts": {
            "subject_inputs.csv": {
                "sha256": sha256_file(subject_path), "rows": int(len(subject_inputs))
            },
            "group_inference.csv": {
                "sha256": sha256_file(inference_path), "rows": int(len(inference))
            },
            "input_manifest.csv": {
                "sha256": sha256_file(inputs_path), "rows": int(len(input_rows))
            },
        },
    }
    atomic_write_json(result_dir / "analysis_manifest.json", manifest)
    log(f"Group analysis complete -> {output_dir}")


def run_paired_analysis(args: argparse.Namespace) -> None:
    game1, rows1, settings_hash1, settings1 = load_population_inputs(
        args.output_root, 1
    )
    game2, rows2, settings_hash2, settings2 = load_population_inputs(
        args.output_root, 2
    )
    if settings_hash1 != settings_hash2 or settings1 != settings2:
        raise ValueError("Game 1 and Game 2 subject analyses use different settings")
    paired = build_paired_effects(game1, game2)
    inference = paired_inference(paired)
    if len(inference) != 2 * EXPECTED_ROIS * len(TRACE_NAMES):
        raise AssertionError("Paired inference row contract failed")
    output_dir = Path(args.output_root) / "group_paired_games"
    result_dir = output_dir / "results"
    figure_dir = output_dir / "figures"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    paired_path = result_dir / "paired_subject_effects.csv"
    inference_path = result_dir / "paired_game_inference.csv"
    inputs_path = result_dir / "input_manifest.csv"
    atomic_write_csv(paired_path, paired)
    atomic_write_csv(inference_path, inference)
    atomic_write_csv(inputs_path, pd.DataFrame([*rows1, *rows2]))
    for contrast, label in (
        ("across_game_common", "Across-game common effect"),
        ("game2_minus_game1", "Game 2 minus Game 1"),
    ):
        selected_inference = inference[
            inference["contrast"].astype(str).eq(contrast)
        ].copy()
        selected_subjects = paired[
            ["subject", "roi", "trace", contrast]
        ].rename(columns={contrast: "contrast_value"})
        plot_ranked_lollipop(
            selected_inference, selected_subjects,
            figure_dir / f"ranked_trace_rsa_lollipop_{contrast}.png",
            title=f"Planning-trace RSA · {label}",
            value_column="contrast_value",
        )
        plot_concat_vs_mean(
            selected_inference,
            figure_dir / f"concat_vs_mean_roi_scatter_{contrast}.png",
            title=f"Ordered trace versus event mean · {label}",
        )
    manifest = {
        "analysis": ANALYSIS_NAME,
        "analysis_version": ANALYSIS_VERSION,
        "schema_version": 1,
        "aggregation": "paired_games",
        "source_contract": current_source_contract(),
        "settings": settings1,
        "settings_sha256": settings_hash1,
        "contrasts": {
            "across_game_common": "(Game1 + Game2) / 2 within subject",
            "game2_minus_game1": "Game2 - Game1 within subject",
        },
        "n_subjects": len(EXPECTED_SUBJECTS),
        "n_rois": EXPECTED_ROIS,
        "n_traces": len(TRACE_NAMES),
        "n_tests_per_contrast_family": EXPECTED_ROIS * len(TRACE_NAMES),
        "n_sign_patterns_per_contrast": 64,
        "correction": (
            "separate two-sided exact synchronized max-|t| family for each "
            "paired contrast, each spanning 2 traces x 110 ROIs"
        ),
        "artifacts": {
            "paired_subject_effects.csv": {
                "sha256": sha256_file(paired_path), "rows": int(len(paired))
            },
            "paired_game_inference.csv": {
                "sha256": sha256_file(inference_path), "rows": int(len(inference))
            },
            "input_manifest.csv": {
                "sha256": sha256_file(inputs_path), "rows": int(len(rows1) + len(rows2))
            },
        },
    }
    atomic_write_json(result_dir / "analysis_manifest.json", manifest)
    log(f"Paired-game analysis complete -> {output_dir}")


def self_test() -> dict[str, Any]:
    rng = np.random.default_rng(7029)
    n_tr = 42
    blocks = {
        "visual": rng.normal(size=(n_tr, 18)),
        "decision_event": np.convolve(
            (np.arange(n_tr) % 5 == 0).astype(float),
            np.asarray([0.2, 0.7, 1.0, 0.7, 0.2]),
            mode="same",
        )[:, None],
        "current_action": np.eye(6)[rng.integers(0, 6, size=n_tr)],
        "previous_action": np.eye(6)[rng.integers(0, 6, size=n_tr)],
        "task_history": rng.normal(size=(n_tr, 8)),
        "drift": np.column_stack((
            np.linspace(-1, 1, n_tr), np.linspace(-1, 1, n_tr) ** 2
        )),
        "model_prior_policy": rng.normal(size=(n_tr, 5)),
        "model_margin": rng.normal(size=(n_tr, 1)),
        "initial_q_state": rng.normal(size=(n_tr, 1)),
        "planning_concat29": rng.normal(size=(n_tr, TRACE_WIDTHS["planning_concat29"])),
        "planning_mean29": rng.normal(size=(n_tr, TRACE_WIDTHS["planning_mean29"])),
    }
    rdms = build_rdms(blocks)
    pair_mask = np.asarray(COMMON.pair_mask_min_lag(n_tr, 10), dtype=bool)
    design = prepare_fwl_design(rdms, pair_mask)
    synthetic_brain = (
        0.8 * design.target_residuals[:, 0]
        + 0.2 * design.target_residuals[:, 1]
        + rng.normal(scale=0.15, size=design.n_pairs_retained)
    )
    score = score_ranked_brain(synthetic_brain, design)
    y_residual = (
        synthetic_brain
        - design.reduced_q @ (design.reduced_q.T @ synthetic_brain)
    )
    brute = np.asarray([
        np.dot(y_residual, design.target_residuals[:, index])
        / math.sqrt(
            np.dot(y_residual, y_residual)
            * design.target_residual_ss[index]
        )
        for index in range(len(TRACE_NAMES))
    ])
    fwl_error = float(
        np.max(np.abs(brute - score["conditional_partial_r"]))
    )
    if fwl_error > 1e-12 or float(score["conditional_partial_r"][0]) < 0.8:
        raise AssertionError(
            f"Synthetic FWL equivalence/injection failed: error={fwl_error}, "
            f"score={score}"
        )
    orders = synchronized_block_orders(n_tr, 7, 10, 91)
    if orders.shape != (7, n_tr) or any(
        not np.array_equal(np.sort(order), np.arange(n_tr)) for order in orders
    ):
        raise AssertionError("Synchronized temporal permutation contract failed")

    small_rows: list[dict[str, Any]] = []
    for run_index in range(3):
        for roi_index in range(4):
            for trace_index, trace in enumerate(TRACE_NAMES):
                value = float(
                    0.03 + 0.002 * run_index - 0.001 * roi_index
                    + 0.004 * trace_index
                )
                small_rows.append({
                    "run_label": f"run{run_index:02d}", "subject": 1, "game": 1,
                    "roi": f"roi_{roi_index:03d}", "trace": trace,
                    "conditional_partial_r": value, "added_beta": value * 0.8,
                    "delta_r2_descriptive": value * value,
                    "p_temporal_block_diagnostic": 0.5,
                    "p_temporal_block_maxT_diagnostic": 0.8,
                })
    small_summary = summarize_subject(
        pd.DataFrame(small_rows), expected_runs=3, expected_rois=4
    )
    if len(small_summary) != 8 or not small_summary["n_runs"].eq(3).all():
        raise AssertionError("Run-equal subject summary failed")

    population_rows: list[dict[str, Any]] = []
    for subject in EXPECTED_SUBJECTS:
        subject_noise = rng.normal(scale=0.006)
        for roi_index in range(EXPECTED_ROIS):
            for trace_index, trace in enumerate(TRACE_NAMES):
                effect = (
                    0.012 * (roi_index == 0)
                    + 0.004 * trace_index
                    + subject_noise
                    + rng.normal(scale=0.005)
                )
                population_rows.append({
                    "subject": subject, "game": 1,
                    "roi": f"roi_{roi_index:03d}", "trace": trace,
                    "trace_label": TRACE_LABELS[trace],
                    "n_runs": EXPECTED_RUNS,
                    "mean_conditional_partial_r": float(effect),
                })
    population = pd.DataFrame(population_rows)
    inference = group_inference(population)
    if (
        len(inference) != EXPECTED_ROIS * len(TRACE_NAMES)
        or not inference["n_sign_patterns"].eq(64).all()
        or np.any(
            inference["p_max_abs_t_fwer"].to_numpy(dtype=float)
            + 1e-15
            < inference["p_exact_two_sided"].to_numpy(dtype=float)
        )
    ):
        raise AssertionError("Exact 2x110 population inference failed")
    game2 = population.copy()
    game2["game"] = 2
    game2["mean_conditional_partial_r"] += rng.normal(
        loc=0.001, scale=0.003, size=len(game2)
    )
    paired = build_paired_effects(population, game2)
    paired_result = paired_inference(paired)
    if (
        len(paired_result) != 2 * EXPECTED_ROIS * len(TRACE_NAMES)
        or set(paired_result["contrast"]) != {
            "across_game_common", "game2_minus_game1"
        }
    ):
        raise AssertionError("Paired common/difference inference failed")

    with tempfile.TemporaryDirectory(prefix="planning_trace_rsa_selftest_") as tmp:
        root = Path(tmp)
        exemplar = {
            trace: {
                "trace_residual": design.target_residuals[:, index],
                "brain_residual": y_residual,
            }
            for index, trace in enumerate(TRACE_NAMES)
        }
        plot_residual_rdm_scatter(
            exemplar, root / "residual.png",
            title="Synthetic residual RDM distances", random_state=7,
        )
        selected_rois = [f"roi_{index:03d}" for index in range(6)]
        selected_inference = inference[
            inference["roi"].isin(selected_rois)
        ].copy()
        selected_population = population[
            population["roi"].isin(selected_rois)
        ].copy()
        plot_ranked_lollipop(
            selected_inference, selected_population,
            root / "lollipop.png", title="Synthetic all-subject effects",
        )
        plot_concat_vs_mean(
            selected_inference, root / "trace_scatter.png",
            title="Synthetic trace comparison",
        )
        expected_figures = [
            root / f"{stem}.{suffix}"
            for stem in ("residual", "lollipop", "trace_scatter")
            for suffix in ("png", "svg")
        ]
        if any(not path.is_file() or path.stat().st_size == 0 for path in expected_figures):
            raise AssertionError("Synthetic figure rendering failed")

    return {
        "analysis": ANALYSIS_NAME,
        "fwl_max_abs_error": fwl_error,
        "injected_concat_partial_r": float(score["conditional_partial_r"][0]),
        "n_pairs_retained_lag10": int(design.n_pairs_retained),
        "temporal_orders": int(len(orders)),
        "subject_summary_rows": int(len(small_summary)),
        "group_rows": int(len(inference)),
        "paired_rows": int(len(paired_result)),
        "exact_sign_patterns": int(inference["n_sign_patterns"].iloc[0]),
        "figure_files_checked": 6,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    COMMON.add_shared_cli_arguments(parser)
    TRACE_ADAPTER.module.add_trace_cli_arguments(parser)
    for action in parser._actions:
        if action.dest in {"subject", "game"}:
            action.required = False
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--aggregate-subjects", action="store_true",
        help="Aggregate subjects 1..6 for one game using exact population inference",
    )
    parser.add_argument(
        "--aggregate-games", action="store_true",
        help="Pair Game 1/Game 2 within subjects and test common and difference effects",
    )
    parser.add_argument(
        "--n-permutations", type=int, default=0,
        help=(
            "Diagnostic synchronized temporal block-null draws (0..200; "
            "population sign-flip inference remains primary)"
        ),
    )
    parser.add_argument("--permutation-block-size", type=int, default=40)
    parser.add_argument("--permutation-batch-size", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--min-pair-lag", type=int, default=10,
        help="Retain only RDM pairs separated by at least 10 TRs",
    )
    parser.add_argument(
        "--exemplar-roi", default="right_caudate",
        help="Prespecified token for the descriptive residual-distance figure",
    )
    parser.add_argument(
        "--diagnostic-noncanonical", action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)
    if args.self_test:
        print(json.dumps(self_test(), indent=2, sort_keys=True))
        return
    parser.error(
        "07_planning_trace_RSA.py is now an importable computation engine. "
        "Run 07_RSA.py --rsa-family planning-trace so results are written "
        "only under outputs/07_RSA."
    )
    if args.aggregate_subjects and args.aggregate_games:
        parser.error("Choose one aggregation mode")
    if not 0 <= int(args.n_permutations) <= 200:
        parser.error("--n-permutations must be in 0..200")
    if int(args.permutation_block_size) < 2:
        parser.error("--permutation-block-size must be >=2")
    if int(args.permutation_batch_size) < 1:
        parser.error("--permutation-batch-size must be >=1")
    if int(args.n_jobs) < 1:
        parser.error("--n-jobs must be >=1")
    if int(args.min_pair_lag) < 10:
        parser.error("--min-pair-lag must be >=10")
    if args.aggregate_subjects:
        if args.game is None:
            parser.error("--game is required with --aggregate-subjects")
        if args.subject is not None:
            parser.error("Do not pass --subject with --aggregate-subjects")
        run_group_analysis(args)
        return
    if args.aggregate_games:
        if args.subject is not None or args.game is not None:
            parser.error("Do not pass --subject/--game with --aggregate-games")
        run_paired_analysis(args)
        return
    if args.subject is None or args.game is None:
        parser.error("--subject and --game are required in subject mode")
    run_subject_analysis(args)


if __name__ == "__main__":
    main()
