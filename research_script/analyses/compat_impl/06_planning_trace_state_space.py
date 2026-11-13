#!/usr/bin/env python3
"""Decode observed 29-event planning traces from ROI population activity.

This analysis is intentionally separate from ``06_neural_state_space.py``.
It compares two representations built from the same observed event tokens:

``planning_concat29``
    Event-major concatenation of all 29 tokens (29 x 12 dimensions).

``planning_mean29``
    The component-wise mean of the same 29 tokens (12 dimensions).

The target representation never contains DQN features, Thinker hidden states,
terminal Actor output, human action, or a next-event label.  In particular,
event 28 is retained using contemporaneous ``return_gain_from_root``; it is not
silently dropped because no event-29 outcome exists.

Inference uses leave-one-run-out (LORO) prediction.  Pre-search P/M/Q0, decision-event timing, visual PCA, action/history/drift controls, nuisance
residualization, target scaling, neural scaling, and neural PCA are learned on
the ten outer-training runs only.  The primary fold estimand is a pooled
held-out multivariate correlation after per-target-dimension training-fold
standardization.  Thus every target component enters on the same scale and the
348- versus 12-dimensional comparison is not driven by raw units.  A
dimension-equal CV-R2 and per-TR pattern correlations are secondary metrics.

Raw concat29 contains mean29 information, so different decoding scores do not
constitute a unique temporal-order test.  That requires a separately declared
within-window-centred/order-residual representation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

import neural_roi_figures as NROI


SCRIPT_DIR = Path(__file__).resolve().parent
TRACE_COMMON_PATH = SCRIPT_DIR / "planning_trace_common.py"
BRAIN_COMMON_PATH = SCRIPT_DIR / "planning_brain_common.py"

ANALYSIS_NAME = "06_planning_trace_state_space"
ANALYSIS_VERSION = "2026-08-20-participant-permutation-predefined-roi-v2"

# The canonical subject outputs were produced before anatomy-only CLI and
# display changes were added.  Those changes do not alter decoding inputs,
# folds, estimands, or inference, so the frozen inferential revisions remain
# valid group inputs alongside the live source hashes.
COMPATIBLE_SUBJECT_SCRIPT_SHA256 = frozenset(
    {"4431ca97c7166b35e88c7e4065ff3aa73098ee7d25a89b98751727e6030a3d57"}
)
COMPATIBLE_SUBJECT_NEURAL_FIGURE_SHA256 = frozenset(
    {"a2751f791091da26bf4bde17bc663bda9296bd9f481ca82a934dd6847505905c"}
)
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs" / ANALYSIS_NAME
DEFAULT_TRACE_ROOT = (
    SCRIPT_DIR
    / "outputs"
    / "00_brain_cache"
    / "planning_trace_cache"
    / "v1_concat29_mean29"
)

N_EVENTS = 29
EXPECTED_TOKEN_COMPONENTS = (
    "branch_action_0",
    "branch_action_1",
    "branch_action_2",
    "branch_action_3",
    "branch_action_4",
    "branch_action_5",
    "depth",
    "return_gain_from_root",
    "root_q_revision_filled0",
    "q_revision_valid",
    "log1p_root_n_before_count",
    "reset_before",
)
REPRESENTATIONS = ("concat29", "mean29")
TRACE_BLOCK_NAMES = {
    "concat29": "planning_concat29",
    "mean29": "planning_mean29",
}
EXPECTED_ROIS = 110
EXPECTED_RUNS = 11
EXPECTED_SUBJECTS = tuple(range(1, 7))

VISUAL_PCA_COMPONENTS = 20
NEURAL_PCA_COMPONENTS = 64
RIDGE_ALPHA = 100.0
NUISANCE_ALPHA = 1.0
EPS = 1e-12
SIGNFLIP_RTOL = 128.0 * np.finfo(np.float64).eps
SIGNFLIP_ATOL = 128.0 * np.finfo(np.float64).eps
T95_DF5 = 2.570581835636314
PARTICIPANT_PERMUTATIONS = 9999
PARTICIPANT_BLOCK_SIZE_TR = 40

INK = "#33363A"
GRID = "#D9DDE3"
BLUE = "#386D9C"
ORANGE = "#CF7B35"
REPRESENTATION_COLORS = {"concat29": BLUE, "mean29": ORANGE}


def log(message: str) -> None:
    print(message, flush=True)


def _load_local_module(path: Path, name: str) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Required shared module is missing: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(int(chunk_size))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_signature(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256_file(resolved),
    }


def analysis_source_signatures() -> dict[str, dict[str, Any]]:
    return {
        "script": file_signature(Path(__file__)),
        "planning_trace_common": file_signature(TRACE_COMMON_PATH),
        "planning_brain_common": file_signature(BRAIN_COMMON_PATH),
        "neural_roi_figures": file_signature(Path(NROI.__file__)),
    }


def verify_source_signatures(expected: Mapping[str, Mapping[str, Any]]) -> None:
    observed = analysis_source_signatures()
    if observed != dict(expected):
        raise RuntimeError("Analysis source changed while this job was running")


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def atomic_write_csv(frame: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    compression: str | None = "gzip" if path.name.endswith(".gz") else None
    frame.to_csv(temporary, index=index, compression=compression)
    temporary.replace(path)


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _as_finite_2d(
    values: Any,
    *,
    name: str,
    n_rows: int | None = None,
) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim == 1:
        result = result[:, None]
    if result.ndim != 2 or result.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2-D matrix, got {result.shape}")
    if n_rows is not None and result.shape[0] != int(n_rows):
        raise ValueError(f"{name} row mismatch: {result.shape[0]} != {int(n_rows)}")
    if not np.all(np.isfinite(result)):
        bad = int(np.sum(~np.isfinite(result)))
        raise ValueError(f"{name} contains {bad} non-finite values")
    return result


def _resolve_callable(module: Any, names: Sequence[str], role: str) -> Callable[..., Any]:
    found = [name for name in names if callable(getattr(module, name, None))]
    if len(found) != 1:
        raise RuntimeError(
            f"planning_trace_common must expose exactly one supported {role} "
            f"callable; candidates={tuple(names)}, found={found}"
        )
    return getattr(module, found[0])


def _mapping_from_metadata(metadata: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = metadata.get(key, {})
    return value if isinstance(value, Mapping) else {}


class StrictTraceAdapter:
    """Isolate the evolving trace-cache API behind a fail-closed contract."""

    PREPARE_NAMES = (
        "prepare_trace_runs_from_args",
        "prepare_planning_trace_runs_from_args",
        "shared_prepare_from_args",
    )
    LOAD_NAMES = (
        "load_run_trace",
        "load_planning_trace",
        "load_trace_blocks",
    )

    def __init__(self, module: Any):
        self.module = module
        self.prepare_callable = _resolve_callable(
            module, self.PREPARE_NAMES, "run preparation"
        )
        self.load_callable = _resolve_callable(module, self.LOAD_NAMES, "trace loading")

    def prepare(self, args: argparse.Namespace) -> tuple[list[Any], dict[str, Any], dict[str, Any]]:
        result = self.prepare_callable(args)
        if not isinstance(result, tuple) or len(result) != 3:
            raise RuntimeError(
                "Trace preparation must return (run_specs, cache_audit, trace_audit)"
            )
        run_specs, cache_audit, trace_audit = result
        runs = list(run_specs)
        if not runs:
            raise ValueError("Trace preparation returned no runs")
        if not isinstance(cache_audit, Mapping) or not isinstance(trace_audit, Mapping):
            raise TypeError("Trace preparation audits must both be mappings")
        labels = [str(getattr(run, "run_label", "")) for run in runs]
        if any(not label for label in labels) or len(set(labels)) != len(labels):
            raise ValueError(f"Invalid or duplicate run labels: {labels}")
        return runs, dict(cache_audit), dict(trace_audit)

    def load(
        self, run: Any, trace_root: Path
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        try:
            result = self.load_callable(run, Path(trace_root))
        except TypeError as first_error:
            try:
                result = self.load_callable(run, trace_root=Path(trace_root))
            except TypeError:
                raise first_error
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError("Trace loader must return (blocks, metadata)")
        blocks_raw, metadata_raw = result
        if not isinstance(blocks_raw, Mapping) or not isinstance(metadata_raw, Mapping):
            raise TypeError("Trace blocks and metadata must be mappings")
        metadata = dict(metadata_raw)
        n_tr = int(getattr(run, "n_tr"))
        blocks = {
            name: _as_finite_2d(
                blocks_raw[name], name=f"{run.run_label}:{name}", n_rows=n_tr
            )
            for name in TRACE_BLOCK_NAMES.values()
            if name in blocks_raw
        }
        missing = sorted(set(TRACE_BLOCK_NAMES.values()).difference(blocks))
        if missing:
            raise KeyError(f"{run.run_label} trace cache lacks {missing}")
        self._validate_contract(run, blocks, metadata)
        return blocks, metadata

    @staticmethod
    def _validate_contract(
        run: Any,
        blocks: Mapping[str, np.ndarray],
        metadata: Mapping[str, Any],
    ) -> None:
        n_events = metadata.get("n_events")
        if n_events is None:
            n_events = _mapping_from_metadata(metadata, "representation_contract").get(
                "n_events"
            )
        if int(n_events if n_events is not None else -1) != N_EVENTS:
            raise ValueError(f"{run.run_label}: trace metadata does not declare 29 events")

        tokens: Any = metadata.get("token_components")
        if tokens is None:
            tokens = _mapping_from_metadata(metadata, "representation_contract").get(
                "token_components"
            )
        if tuple(str(value) for value in (tokens or ())) != EXPECTED_TOKEN_COMPONENTS:
            raise ValueError(
                f"{run.run_label}: token order changed: {tokens} != "
                f"{EXPECTED_TOKEN_COMPONENTS}"
            )

        layout = metadata.get("concat_layout")
        if layout is None:
            layout = _mapping_from_metadata(metadata, "representation_contract").get(
                "concat_layout"
            )
        if str(layout) not in {"event_major", "event-major"}:
            raise ValueError(f"{run.run_label}: concat29 is not event-major: {layout}")

        centering = metadata.get("concat_centering")
        if centering is None:
            centering = _mapping_from_metadata(metadata, "representation_contract").get(
                "concat_centering"
            )
        if str(centering).lower() not in {"none", "raw", "uncentered"}:
            raise ValueError(
                f"{run.run_label}: this estimand requires raw concat29, got {centering}"
            )

        mean = blocks[TRACE_BLOCK_NAMES["mean29"]]
        concat = blocks[TRACE_BLOCK_NAMES["concat29"]]
        if mean.shape[1] != len(EXPECTED_TOKEN_COMPONENTS):
            raise ValueError(f"{run.run_label}: mean29 width is {mean.shape[1]}, expected 12")
        if concat.shape[1] != N_EVENTS * len(EXPECTED_TOKEN_COMPONENTS):
            raise ValueError(
                f"{run.run_label}: concat29 width is {concat.shape[1]}, expected 348"
            )

        components = _mapping_from_metadata(metadata, "component_names")
        if components:
            for block_name, expected_width in (
                (TRACE_BLOCK_NAMES["concat29"], concat.shape[1]),
                (TRACE_BLOCK_NAMES["mean29"], mean.shape[1]),
            ):
                names = components.get(block_name)
                if not isinstance(names, Sequence) or isinstance(names, (str, bytes)):
                    raise ValueError(
                        f"{run.run_label}: missing component names for {block_name}"
                    )
                if len(names) != expected_width:
                    raise ValueError(
                        f"{run.run_label}:{block_name} component width mismatch"
                    )

        forbidden = json.dumps(metadata, sort_keys=True).lower()
        declared_sources = metadata.get("target_source_fields")
        if declared_sources is not None:
            fields = {str(value) for value in declared_sources}
            prohibited = {
                name
                for name in fields
                if name.startswith("next_")
                or "human" in name.lower()
                or "actor_terminal" in name.lower()
                or "hidden" in name.lower()
                or "dqn" in name.lower()
            }
            if prohibited:
                raise ValueError(
                    f"{run.run_label}: prohibited trace source fields: {sorted(prohibited)}"
                )
        # The metadata must positively state that the last event is observed,
        # not a fabricated next-event target.  Accept either top-level or nested
        # audit spelling but do not infer this silently from array width.
        last_policy = metadata.get("last_event_policy")
        if last_policy is None:
            last_policy = _mapping_from_metadata(metadata, "representation_contract").get(
                "last_event_policy"
            )
        if last_policy is not None and "next" in str(last_policy).lower() and "no" not in str(last_policy).lower():
            raise ValueError(
                f"{run.run_label}: last-event policy may contain a next-outcome label"
            )
        if any(token in forbidden for token in ('"dqn_used": true', '"hidden_activations_used": true')):
            raise ValueError(f"{run.run_label}: metadata declares prohibited neural-model features")


@dataclass
class TraceRun:
    run: Any
    run_label: str
    subject: int
    game: int
    targets: dict[str, np.ndarray]
    target_component_names: dict[str, tuple[str, ...]]
    base_controls: np.ndarray
    visual: np.ndarray
    metadata: dict[str, Any]


@dataclass
class PreparedFold:
    held_out: int
    train_indices: tuple[int, ...]
    control_train: np.ndarray
    control_test: np.ndarray
    target_train: dict[str, np.ndarray]
    target_test: dict[str, np.ndarray]
    target_component_names: dict[str, tuple[str, ...]]
    visual_components: int


def _component_names(
    metadata: Mapping[str, Any], block_name: str, width: int
) -> tuple[str, ...]:
    mapping = _mapping_from_metadata(metadata, "component_names")
    names = mapping.get(block_name)
    if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        result = tuple(str(value) for value in names)
        if len(result) == int(width):
            return result
    if block_name == TRACE_BLOCK_NAMES["mean29"]:
        return tuple(f"mean29::{name}" for name in EXPECTED_TOKEN_COMPONENTS)
    return tuple(
        f"event{event:02d}::{name}"
        for event in range(N_EVENTS)
        for name in EXPECTED_TOKEN_COMPONENTS
    )


def load_trace_runs(
    run_specs: Sequence[Any],
    *,
    trace_adapter: StrictTraceAdapter,
    brain_common: Any,
    trace_root: Path,
) -> tuple[list[TraceRun], list[dict[str, Any]]]:
    process_components = getattr(brain_common, "PROCESS_BLOCK_COMPONENTS", None)
    if not isinstance(process_components, Mapping):
        raise RuntimeError("planning_brain_common lacks PROCESS_BLOCK_COMPONENTS")
    required_controls = (
        "model_prior_policy",
        "model_margin",
        "initial_q_state",
        "decision_event",
        "current_action",
        "previous_action",
        "task_history",
        "drift",
    )
    runs: list[TraceRun] = []
    audits: list[dict[str, Any]] = []
    for run in run_specs:
        blocks, metadata = trace_adapter.load(run, trace_root)
        process = brain_common.load_process_blocks(run)
        n_tr = int(run.n_tr)
        control_arrays: list[np.ndarray] = []
        control_widths: dict[str, int] = {}
        for name in required_controls:
            if name not in process:
                raise KeyError(f"{run.run_label}: process cache lacks control {name}")
            values = _as_finite_2d(
                process[name], name=f"{run.run_label}:{name}", n_rows=n_tr
            )
            expected = tuple(process_components.get(name, ()))
            if values.shape[1] != len(expected):
                raise ValueError(
                    f"{run.run_label}:{name} width {values.shape[1]} != "
                    f"component contract {expected}"
                )
            control_arrays.append(values)
            control_widths[name] = int(values.shape[1])
        visual = _as_finite_2d(
            brain_common.load_visual(run),
            name=f"{run.run_label}:visual",
            n_rows=n_tr,
        )
        targets = {
            representation: np.asarray(
                blocks[TRACE_BLOCK_NAMES[representation]], dtype=np.float64
            )
            for representation in REPRESENTATIONS
        }
        names = {
            representation: _component_names(
                metadata,
                TRACE_BLOCK_NAMES[representation],
                targets[representation].shape[1],
            )
            for representation in REPRESENTATIONS
        }
        runs.append(
            TraceRun(
                run=run,
                run_label=str(run.run_label),
                subject=int(run.subject),
                game=int(run.game),
                targets=targets,
                target_component_names=names,
                base_controls=np.column_stack(control_arrays),
                visual=visual,
                metadata=metadata,
            )
        )
        audits.append(
            {
                "run_label": str(run.run_label),
                "n_tr": n_tr,
                "concat29_width": int(targets["concat29"].shape[1]),
                "mean29_width": int(targets["mean29"].shape[1]),
                "visual_width": int(visual.shape[1]),
                "control_widths": control_widths,
                "trace_metadata": metadata,
            }
        )
    if len(runs) < 3:
        raise ValueError("LORO decoding requires at least three runs so every held-out fold contains at least two independent training runs")
    if len({run.subject for run in runs}) != 1 or len({run.game for run in runs}) != 1:
        raise ValueError("A subject analysis must contain one subject-game only")
    mean_names = {run.target_component_names["mean29"] for run in runs}
    concat_names = {run.target_component_names["concat29"] for run in runs}
    if len(mean_names) != 1 or len(concat_names) != 1:
        raise ValueError("Trace component order differs between runs")
    return runs, audits


def _standardize_train_test(
    train: np.ndarray,
    test: np.ndarray,
    *,
    drop_constant: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_train = _as_finite_2d(train, name="training matrix")
    x_test = _as_finite_2d(test, name="test matrix")
    if x_train.shape[1] != x_test.shape[1]:
        raise ValueError("Training/test feature widths differ")
    mean = np.mean(x_train, axis=0)
    scale = np.std(x_train, axis=0, ddof=0)
    keep = np.isfinite(scale) & (scale > EPS)
    if drop_constant:
        if not np.any(keep):
            raise ValueError("Every training feature is constant")
        mean = mean[keep]
        scale = scale[keep]
        return (
            (x_train[:, keep] - mean) / scale,
            (x_test[:, keep] - mean) / scale,
            keep,
        )
    safe_scale = np.where(keep, scale, 1.0)
    return (x_train - mean) / safe_scale, (x_test - mean) / safe_scale, keep


def _ridge_residualize(
    train_data: np.ndarray,
    test_data: np.ndarray,
    control_train: np.ndarray,
    control_test: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    y_train = _as_finite_2d(train_data, name="nuisance training outcome")
    y_test = _as_finite_2d(test_data, name="nuisance test outcome")
    c_train = _as_finite_2d(control_train, name="training controls")
    c_test = _as_finite_2d(control_test, name="test controls")
    if len(y_train) != len(c_train) or len(y_test) != len(c_test):
        raise ValueError("Nuisance outcome/control row mismatch")
    design_train = np.column_stack((np.ones(len(c_train)), c_train))
    design_test = np.column_stack((np.ones(len(c_test)), c_test))
    gram = design_train.T @ design_train
    penalty = np.eye(gram.shape[0], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    right = design_train.T @ y_train
    try:
        coefficients = np.linalg.solve(gram + penalty, right)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(gram + penalty, right, rcond=None)[0]
    return (
        y_train - design_train @ coefficients,
        y_test - design_test @ coefficients,
    )


def _prepare_visual_controls(
    visual_train: np.ndarray,
    visual_test: np.ndarray,
    *,
    n_components: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    train_z, test_z, _ = _standardize_train_test(
        visual_train, visual_test, drop_constant=True
    )
    maximum = min(int(n_components), train_z.shape[0] - 1, train_z.shape[1])
    if maximum <= 0:
        raise ValueError("Visual control PCA has no estimable component")
    solver = "randomized" if maximum < min(train_z.shape) else "full"
    pca = PCA(
        n_components=maximum,
        svd_solver=solver,
        random_state=int(random_state),
    )
    return pca.fit_transform(train_z), pca.transform(test_z), int(maximum)


def precompute_loro_folds(
    runs: Sequence[TraceRun], *, random_state: int
) -> list[PreparedFold]:
    folds: list[PreparedFold] = []
    for held_out in range(len(runs)):
        train_indices = tuple(index for index in range(len(runs)) if index != held_out)
        base_train = np.vstack([runs[index].base_controls for index in train_indices])
        base_test = runs[held_out].base_controls
        base_train_z, base_test_z, _ = _standardize_train_test(
            base_train, base_test, drop_constant=False
        )
        visual_train = np.vstack([runs[index].visual for index in train_indices])
        visual_test = runs[held_out].visual
        visual_train_pc, visual_test_pc, n_visual = _prepare_visual_controls(
            visual_train,
            visual_test,
            n_components=VISUAL_PCA_COMPONENTS,
            random_state=int(random_state) + held_out * 101,
        )
        control_train = np.column_stack((base_train_z, visual_train_pc))
        control_test = np.column_stack((base_test_z, visual_test_pc))
        control_train, control_test, _ = _standardize_train_test(
            control_train, control_test, drop_constant=False
        )

        target_train: dict[str, np.ndarray] = {}
        target_test: dict[str, np.ndarray] = {}
        component_names: dict[str, tuple[str, ...]] = {}
        for representation in REPRESENTATIONS:
            raw_train = np.vstack(
                [runs[index].targets[representation] for index in train_indices]
            )
            raw_test = runs[held_out].targets[representation]
            residual_train, residual_test = _ridge_residualize(
                raw_train,
                raw_test,
                control_train,
                control_test,
                alpha=NUISANCE_ALPHA,
            )
            train_z, test_z, keep = _standardize_train_test(
                residual_train, residual_test, drop_constant=True
            )
            names = runs[0].target_component_names[representation]
            component_names[representation] = tuple(
                name for name, retained in zip(names, keep) if bool(retained)
            )
            target_train[representation] = np.asarray(train_z, dtype=np.float32)
            target_test[representation] = np.asarray(test_z, dtype=np.float32)
        folds.append(
            PreparedFold(
                held_out=held_out,
                train_indices=train_indices,
                control_train=np.asarray(control_train, dtype=np.float32),
                control_test=np.asarray(control_test, dtype=np.float32),
                target_train=target_train,
                target_test=target_test,
                target_component_names=component_names,
                visual_components=n_visual,
            )
        )
    return folds


def pooled_multivariate_correlation(actual: np.ndarray, predicted: np.ndarray) -> float:
    first = np.asarray(actual, dtype=np.float64).reshape(-1)
    second = np.asarray(predicted, dtype=np.float64).reshape(-1)
    valid = np.isfinite(first) & np.isfinite(second)
    if int(np.sum(valid)) < 3:
        return float("nan")
    first = first[valid] - float(np.mean(first[valid]))
    second = second[valid] - float(np.mean(second[valid]))
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= EPS:
        return float("nan")
    return float(np.dot(first, second) / denominator)


def dimension_equal_cv_r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Out-of-fold Q2 against the outer-training mean, equal-weighting columns."""
    y = _as_finite_2d(actual, name="CV-R2 actual")
    prediction = _as_finite_2d(predicted, name="CV-R2 prediction", n_rows=len(y))
    if y.shape != prediction.shape:
        raise ValueError("CV-R2 actual/prediction shapes differ")
    mse = np.mean((y - prediction) ** 2, axis=0)
    baseline_mse = np.mean(y**2, axis=0)
    denominator = float(np.mean(baseline_mse))
    if denominator <= EPS:
        return float("nan")
    return float(1.0 - float(np.mean(mse)) / denominator)


def per_tr_pattern_correlation(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    y = _as_finite_2d(actual, name="pattern actual")
    prediction = _as_finite_2d(
        predicted, name="pattern prediction", n_rows=len(y)
    )
    if y.shape != prediction.shape:
        raise ValueError("Pattern actual/prediction shapes differ")
    y_centered = y - np.mean(y, axis=1, keepdims=True)
    p_centered = prediction - np.mean(prediction, axis=1, keepdims=True)
    numerator = np.sum(y_centered * p_centered, axis=1)
    denominator = np.sqrt(
        np.sum(y_centered**2, axis=1) * np.sum(p_centered**2, axis=1)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.full(len(y), np.nan, dtype=np.float64),
        where=denominator > EPS,
    )


def _prepare_neural_fold(
    patterns_by_run: Sequence[np.ndarray],
    fold: PreparedFold,
    *,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    raw_train = np.vstack([patterns_by_run[index] for index in fold.train_indices])
    raw_test = patterns_by_run[fold.held_out]
    n_raw = int(raw_train.shape[1])
    residual_train, residual_test = _ridge_residualize(
        raw_train,
        raw_test,
        fold.control_train,
        fold.control_test,
        alpha=NUISANCE_ALPHA,
    )
    train_z, test_z, _ = _standardize_train_test(
        residual_train, residual_test, drop_constant=True
    )
    maximum = min(
        NEURAL_PCA_COMPONENTS,
        train_z.shape[0] - 1,
        train_z.shape[1],
    )
    if maximum <= 0:
        raise ValueError("Neural PCA has no estimable component")
    solver = "randomized" if maximum < min(train_z.shape) else "full"
    pca = PCA(
        n_components=maximum,
        svd_solver=solver,
        random_state=int(random_state),
    )
    return (
        np.asarray(pca.fit_transform(train_z), dtype=np.float32),
        np.asarray(pca.transform(test_z), dtype=np.float32),
        n_raw,
        int(maximum),
    )


def synchronized_block_orders(
    runs: Sequence[TraceRun],
    *,
    n_permutations: int,
    block_size: int,
    random_state: int,
) -> list[np.ndarray]:
    """Fixed within-run block orders shared by every ROI/representation."""
    if int(n_permutations) < 1 or int(block_size) < 1:
        raise ValueError("Participant permutation count and block size must be positive")
    result: list[np.ndarray] = []
    for run_index, run in enumerate(runs):
        n_tr = int(run.run.n_tr)
        if n_tr % int(block_size) != 0:
            raise ValueError(
                f"{run.run_label}: n_tr={n_tr} is not divisible by block size {block_size}"
            )
        n_blocks = n_tr // int(block_size)
        rng = np.random.default_rng(
            int(random_state)
            + 1_000_003 * int(run.run.game)
            + 10_007 * int(run.run.subject)
            + 1_009 * int(run_index)
        )
        orders = np.vstack(
            [rng.permutation(n_blocks) for _ in range(int(n_permutations))]
        ).astype(np.int16)
        result.append(orders)
    return result


def block_permuted_pooled_correlations(
    actual: np.ndarray,
    predicted: np.ndarray,
    orders: np.ndarray,
    *,
    block_size: int,
) -> np.ndarray:
    """Correlation null from block-shuffling held-out targets, without refitting."""
    y = _as_finite_2d(actual, name="block-null actual")
    prediction = _as_finite_2d(
        predicted, name="block-null prediction", n_rows=y.shape[0]
    )
    if y.shape != prediction.shape:
        raise ValueError(f"Block-null target/prediction mismatch: {y.shape} != {prediction.shape}")
    if y.shape[0] % int(block_size) != 0:
        raise ValueError("Block-null rows must be divisible by block size")
    n_blocks = y.shape[0] // int(block_size)
    order_matrix = np.asarray(orders, dtype=np.int64)
    if order_matrix.ndim != 2 or order_matrix.shape[1] != n_blocks:
        raise ValueError("Block-order shape does not match held-out run")
    y_blocks = y.reshape(n_blocks, int(block_size), y.shape[1])
    p_blocks = prediction.reshape(n_blocks, int(block_size), prediction.shape[1])
    cross = np.einsum("ikd,jkd->ij", y_blocks, p_blocks, optimize=True)
    positions = np.arange(n_blocks, dtype=np.int64)[None, :]
    raw_numerator = cross[order_matrix, positions].sum(axis=1)
    n_values = float(y.size)
    centered_numerator = raw_numerator - float(y.sum() * prediction.sum()) / n_values
    y_ss = float(np.sum(y * y) - y.sum() ** 2 / n_values)
    p_ss = float(
        np.sum(prediction * prediction) - prediction.sum() ** 2 / n_values
    )
    denominator = math.sqrt(max(y_ss, 0.0) * max(p_ss, 0.0))
    if denominator <= EPS:
        return np.zeros(order_matrix.shape[0], dtype=np.float64)
    null = centered_numerator / denominator
    if not np.all(np.isfinite(null)):
        raise ValueError("Non-finite block-permutation correlation")
    identity = np.arange(n_blocks, dtype=np.int64)[None, :]
    identity_value = float(
        block_permuted_pooled_correlations(y, prediction, identity, block_size=block_size)[0]
    ) if order_matrix.shape[0] != 1 or not np.array_equal(order_matrix[0], identity[0]) else None
    if identity_value is not None:
        observed = pooled_multivariate_correlation(y, prediction)
        if not np.isclose(identity_value, observed, rtol=1e-10, atol=1e-12):
            raise AssertionError("Block-null identity does not reproduce pooled correlation")
    return null


def decode_roi_loro(
    patterns_by_run: Sequence[np.ndarray],
    runs: Sequence[TraceRun],
    folds: Sequence[PreparedFold],
    *,
    roi_name: str,
    random_state: int,
    block_orders_by_run: Sequence[np.ndarray] | None = None,
    permutation_block_size: int = PARTICIPANT_BLOCK_SIZE_TR,
    example_limit: int = 4000,
) -> tuple[
    pd.DataFrame,
    dict[str, np.ndarray],
    dict[str, dict[str, np.ndarray]],
    dict[str, np.ndarray],
]:
    if len(patterns_by_run) != len(runs) or len(folds) != len(runs):
        raise ValueError("ROI/run/fold counts differ")
    rows: list[dict[str, Any]] = []
    pattern_by_rep: dict[str, list[np.ndarray]] = {
        representation: [] for representation in REPRESENTATIONS
    }
    examples: dict[str, dict[str, np.ndarray]] = {}
    null_by_rep: dict[str, list[np.ndarray]] = {
        representation: [] for representation in REPRESENTATIONS
    }
    for fold_index, fold in enumerate(folds):
        x_train, x_test, n_raw, n_components = _prepare_neural_fold(
            patterns_by_run,
            fold,
            random_state=int(random_state) + fold_index * 1009,
        )
        for representation in REPRESENTATIONS:
            y_train = np.asarray(fold.target_train[representation], dtype=np.float64)
            y_test = np.asarray(fold.target_test[representation], dtype=np.float64)
            model = Ridge(alpha=RIDGE_ALPHA, solver="lsqr")
            model.fit(x_train, y_train)
            prediction = np.asarray(model.predict(x_test), dtype=np.float64)
            correlation = pooled_multivariate_correlation(y_test, prediction)
            if block_orders_by_run is not None:
                null_by_rep[representation].append(
                    block_permuted_pooled_correlations(
                        y_test,
                        prediction,
                        block_orders_by_run[fold.held_out],
                        block_size=int(permutation_block_size),
                    )
                )
            q2 = dimension_equal_cv_r2(y_test, prediction)
            pattern_r = per_tr_pattern_correlation(y_test, prediction)
            pattern_by_rep[representation].append(pattern_r.astype(np.float32))
            finite_pattern = pattern_r[np.isfinite(pattern_r)]
            run = runs[fold.held_out]
            rows.append(
                {
                    "subject": int(run.subject),
                    "game": int(run.game),
                    "roi": str(roi_name),
                    "representation": representation,
                    "effect_id": f"trace_decode::{representation}",
                    "held_out_run": str(run.run_label),
                    "train_runs": json.dumps(
                        [runs[index].run_label for index in fold.train_indices]
                    ),
                    "beta": correlation,
                    "multivariate_r": correlation,
                    "dimension_equal_cv_r2": q2,
                    "mean_tr_pattern_r": (
                        float(np.mean(finite_pattern))
                        if finite_pattern.size
                        else float("nan")
                    ),
                    "median_tr_pattern_r": (
                        float(np.median(finite_pattern))
                        if finite_pattern.size
                        else float("nan")
                    ),
                    "n_test": int(len(y_test)),
                    "n_target_dimensions": int(y_test.shape[1]),
                    "n_neural_features_raw": n_raw,
                    "n_neural_components": n_components,
                    "n_control_features": int(fold.control_train.shape[1]),
                    "n_visual_components": int(fold.visual_components),
                    "ridge_alpha": RIDGE_ALPHA,
                    "nuisance_alpha": NUISANCE_ALPHA,
                    "primary_metric": "heldout_dimension_standardized_pooled_multivariate_r",
                }
            )
            if fold_index == 0:
                actual_flat = y_test.reshape(-1)
                predicted_flat = prediction.reshape(-1)
                if len(actual_flat) > int(example_limit):
                    indices = np.linspace(
                        0, len(actual_flat) - 1, int(example_limit), dtype=int
                    )
                else:
                    indices = np.arange(len(actual_flat), dtype=int)
                examples[representation] = {
                    "element_index": indices.astype(np.int64),
                    "actual": actual_flat[indices].astype(np.float32),
                    "predicted": predicted_flat[indices].astype(np.float32),
                    "held_out_run": np.asarray([run.run_label]),
                }
    pattern_arrays = {
        representation: np.vstack(pattern_by_rep[representation]).astype(np.float32)
        for representation in REPRESENTATIONS
    }
    participant_nulls: dict[str, np.ndarray] = {}
    if block_orders_by_run is not None:
        for representation in REPRESENTATIONS:
            matrix = np.vstack(null_by_rep[representation])
            if matrix.shape[0] != len(runs):
                raise AssertionError("Participant null is missing a held-out run")
            participant_nulls[representation] = np.mean(matrix, axis=0)
    return pd.DataFrame(rows), pattern_arrays, examples, participant_nulls


def subject_effects_from_folds(
    folds: pd.DataFrame, *, subject: int, game: int, expected_runs: int
) -> pd.DataFrame:
    required = {
        "roi",
        "representation",
        "effect_id",
        "held_out_run",
        "beta",
        "dimension_equal_cv_r2",
        "mean_tr_pattern_r",
    }
    missing = required.difference(folds.columns)
    if missing:
        raise ValueError(f"Fold table lacks {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    grouped = folds.groupby(
        ["roi", "representation", "effect_id"], sort=True, dropna=False
    )
    for (roi, representation, effect_id), frame in grouped:
        run_frame = frame.sort_values("held_out_run", kind="stable")
        if run_frame["held_out_run"].nunique() != int(expected_runs):
            raise ValueError(
                f"{roi}/{representation} has {run_frame['held_out_run'].nunique()} "
                f"folds, expected {expected_runs}"
            )
        if len(run_frame) != int(expected_runs):
            raise ValueError(f"Duplicate fold row for {roi}/{representation}")
        numeric = run_frame[
            ["beta", "dimension_equal_cv_r2", "mean_tr_pattern_r"]
        ].apply(pd.to_numeric, errors="coerce")
        if not np.all(np.isfinite(numeric.to_numpy(float))):
            raise ValueError(f"Non-finite subject fold metric for {roi}/{representation}")
        rows.append(
            {
                "subject": int(subject),
                "game": int(game),
                "roi": str(roi),
                "representation": str(representation),
                "effect_id": str(effect_id),
                "beta": float(numeric["beta"].mean()),
                "mean_multivariate_r": float(numeric["beta"].mean()),
                "mean_dimension_equal_cv_r2": float(
                    numeric["dimension_equal_cv_r2"].mean()
                ),
                "mean_tr_pattern_r": float(numeric["mean_tr_pattern_r"].mean()),
                "n_runs": int(expected_runs),
                "run_weighting": f"equal_mean_across_{expected_runs}_heldout_runs",
                "primary_metric": "heldout_dimension_standardized_pooled_multivariate_r",
            }
        )
    result = pd.DataFrame(rows)
    expected = len(result["roi"].unique()) * len(REPRESENTATIONS)
    if len(result) != expected:
        raise ValueError(f"Subject row contract failed: {len(result)} != {expected}")
    return result.sort_values(["representation", "roi"], kind="stable").reset_index(
        drop=True
    )


def _permutation_ge_or_tie(null: np.ndarray, observed: float) -> np.ndarray:
    return (null > observed) | np.isclose(
        null,
        observed,
        rtol=SIGNFLIP_RTOL,
        atol=SIGNFLIP_ATOL,
        equal_nan=False,
    )


def _t_statistic(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    mean = np.mean(array, axis=0)
    sd = np.std(array, axis=0, ddof=1)
    denominator = sd / math.sqrt(array.shape[0])
    result = np.divide(
        mean, denominator, out=np.zeros_like(mean), where=denominator > EPS
    )
    degenerate = (denominator <= EPS) & (np.abs(mean) > EPS)
    result[degenerate] = np.sign(mean[degenerate]) * np.inf
    return result


def fdr_bh(values: Sequence[float]) -> np.ndarray:
    p = np.asarray(values, dtype=np.float64)
    result = np.full_like(p, np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    if finite.size == 0:
        return result
    order = finite[np.argsort(p[finite])]
    adjusted = p[order] * float(len(order)) / np.arange(1, len(order) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result[order] = np.minimum(adjusted, 1.0)
    return result


def exact_group_inference(
    subject_effects: pd.DataFrame,
    *,
    family_name: str,
) -> pd.DataFrame:
    required = {"subject", "roi", "representation", "effect_id", "beta"}
    missing = required.difference(subject_effects.columns)
    if missing:
        raise ValueError(f"Group input lacks {sorted(missing)}")
    data = subject_effects.loc[:, sorted(required)].copy()
    data["subject"] = pd.to_numeric(data["subject"], errors="raise").astype(int)
    subjects = tuple(sorted(data["subject"].unique().tolist()))
    if subjects != EXPECTED_SUBJECTS:
        raise ValueError(f"Group inference requires subjects 1..6, observed={subjects}")
    data["key"] = (
        data["representation"].astype(str) + "@@" + data["roi"].astype(str)
    )
    if data.duplicated(["subject", "key"], keep=False).any():
        raise ValueError("Duplicate subject/representation/ROI group row")
    matrix = data.pivot(index="subject", columns="key", values="beta").reindex(
        EXPECTED_SUBJECTS
    )
    if matrix.isna().any().any():
        raise ValueError("Incomplete subject x representation x ROI matrix")
    if matrix.shape[1] != len(REPRESENTATIONS) * EXPECTED_ROIS:
        raise ValueError(
            f"Official family must contain 2 x 110 tests, got {matrix.shape[1]}"
        )
    values = matrix.to_numpy(dtype=np.float64)
    observed_t = _t_statistic(values)
    signs = np.asarray(
        list(itertools.product((-1.0, 1.0), repeat=len(EXPECTED_SUBJECTS))),
        dtype=np.float64,
    )
    null_t = np.vstack([_t_statistic(values * sign[:, None]) for sign in signs])
    max_abs_t = np.max(np.abs(null_t), axis=1)
    means = np.mean(values, axis=0)
    rows: list[dict[str, Any]] = []
    for column, key in enumerate(matrix.columns):
        representation, roi = str(key).split("@@", 1)
        raw_p = float(
            np.mean(
                _permutation_ge_or_tie(
                    np.abs(null_t[:, column]), abs(float(observed_t[column]))
                )
            )
        )
        max_p = float(
            np.mean(
                _permutation_ge_or_tie(
                    max_abs_t, abs(float(observed_t[column]))
                )
            )
        )
        rows.append(
            {
                "inference_family": str(family_name),
                "representation": representation,
                "effect_id": f"trace_decode::{representation}",
                "roi": roi,
                "group_beta": float(means[column]),
                "t_stat": float(observed_t[column]),
                "n_subjects": len(EXPECTED_SUBJECTS),
                "p_exact_two_sided": raw_p,
                "p_max_abs_t_fwer": max_p,
                "p_maxT_fwer_family": max_p,
                "signflip_patterns": int(len(signs)),
            }
        )
    result = pd.DataFrame(rows)
    result["q_bh_family"] = fdr_bh(result["p_exact_two_sided"].to_numpy(float))
    result["family_fwer_05"] = result["p_max_abs_t_fwer"].le(0.05)
    return result.sort_values(["representation", "roi"], kind="stable").reset_index(
        drop=True
    )


def participant_block_inference(
    subject_effects: pd.DataFrame,
    *,
    rois: Sequence[str],
    null_cube: np.ndarray,
) -> pd.DataFrame:
    """Cross-style within-participant temporal-null inference.

    The held-out target rows are block-shuffled within each run.  The same
    block orders are used for every ROI and representation, so the maximum-null
    p value corrects the full 2 x 110 participant map.
    """
    null = np.asarray(null_cube, dtype=np.float64)
    if null.ndim != 3 or null.shape[:2] != (len(rois), len(REPRESENTATIONS)):
        raise ValueError(f"Participant null cube has invalid shape: {null.shape}")
    if not np.all(np.isfinite(null)):
        raise ValueError("Participant null cube contains non-finite values")
    indexed = subject_effects.set_index(["roi", "representation"], verify_integrity=True)
    family_max = np.max(null, axis=(0, 1))
    rows: list[dict[str, Any]] = []
    for roi_index, roi in enumerate(rois):
        for representation_index, representation in enumerate(REPRESENTATIONS):
            source = indexed.loc[(str(roi), str(representation))]
            observed = float(source["beta"])
            cell_null = null[roi_index, representation_index]
            denominator = 1.0 + float(cell_null.size)
            rows.append(
                {
                    "subject": int(source["subject"]),
                    "game": int(source["game"]),
                    "roi": str(roi),
                    "representation": str(representation),
                    "effect_id": f"trace_decode::{representation}",
                    "beta": observed,
                    "p_block_one_sided": float(
                        (1.0 + np.sum(cell_null >= observed)) / denominator
                    ),
                    "p_block_maxT_fwer": float(
                        (1.0 + np.sum(family_max >= observed)) / denominator
                    ),
                    "n_permutations": int(cell_null.size),
                    "permutation_tail": "greater",
                    "correction_family": (
                        f"participant_2_representations_x_{len(rois)}_parcels"
                    ),
                }
            )
    result = pd.DataFrame(rows)
    result["q_bh_participant_family"] = fdr_bh(
        result["p_block_one_sided"].to_numpy(float)
    )
    result["participant_fwer_05"] = result["p_block_maxT_fwer"].le(0.05)
    return result


def predefined_roi_subject_effects(subject_effects: pd.DataFrame) -> pd.DataFrame:
    return NROI.aggregate_predefined_rois(
        subject_effects,
        value_columns=("beta",),
        grouping_columns=("subject", "game", "representation", "effect_id"),
    )


def exact_predefined_roi_inference(predefined: pd.DataFrame) -> pd.DataFrame:
    required = {"subject", "representation", "roi_key", "roi_label", "beta"}
    missing = sorted(required - set(predefined.columns))
    if missing:
        raise ValueError(f"Predefined ROI group input lacks {missing}")
    data = predefined.copy()
    subjects = tuple(sorted(pd.to_numeric(data["subject"], errors="raise").astype(int).unique()))
    if subjects != EXPECTED_SUBJECTS:
        raise ValueError(f"Predefined ROI inference requires subjects 1..6: {subjects}")
    data["key"] = data["representation"].astype(str) + "@@" + data["roi_key"].astype(str)
    matrix = data.pivot(index="subject", columns="key", values="beta").reindex(EXPECTED_SUBJECTS)
    expected_columns = {
        f"{representation}@@{roi_key}"
        for representation in REPRESENTATIONS
        for roi_key in NROI.ROI_ORDER
    }
    if set(matrix.columns) != expected_columns or matrix.isna().any().any():
        raise ValueError("Incomplete 6 x 2 x 8 predefined ROI matrix")
    matrix = matrix.reindex(
        columns=[
            f"{representation}@@{roi_key}"
            for representation in REPRESENTATIONS
            for roi_key in NROI.ROI_ORDER
        ]
    )
    values = matrix.to_numpy(float)
    observed_t = _t_statistic(values)
    signs = np.asarray(
        list(itertools.product((-1.0, 1.0), repeat=len(EXPECTED_SUBJECTS))),
        dtype=float,
    )
    null_t = np.vstack([_t_statistic(values * sign[:, None]) for sign in signs])
    max_abs_16 = np.max(np.abs(null_t), axis=1)
    max_positive_16 = np.max(null_t, axis=1)
    rows: list[dict[str, Any]] = []
    for column, key in enumerate(matrix.columns):
        representation, roi_key = str(key).split("@@", 1)
        representation_indices = [
            index for index, name in enumerate(matrix.columns)
            if str(name).startswith(f"{representation}@@")
        ]
        max_abs_8 = np.max(np.abs(null_t[:, representation_indices]), axis=1)
        max_positive_8 = np.max(null_t[:, representation_indices], axis=1)
        observed = float(observed_t[column])
        rows.append(
            {
                "representation": representation,
                "effect_id": f"trace_decode::{representation}",
                "roi_key": roi_key,
                "roi_label": NROI.ROI_LABELS[roi_key],
                "group_beta": float(np.mean(values[:, column])),
                "t_stat": observed,
                "n_subjects": len(EXPECTED_SUBJECTS),
                "p_exact_two_sided": float(
                    np.mean(_permutation_ge_or_tie(np.abs(null_t[:, column]), abs(observed)))
                ),
                "p_exact_greater": float(np.mean(null_t[:, column] >= observed - SIGNFLIP_ATOL)),
                "p_maxT_2x8_two_sided": float(
                    np.mean(_permutation_ge_or_tie(max_abs_16, abs(observed)))
                ),
                "p_maxT_2x8_greater": float(np.mean(max_positive_16 >= observed - SIGNFLIP_ATOL)),
                "p_maxT_within_representation_8_two_sided": float(
                    np.mean(_permutation_ge_or_tie(max_abs_8, abs(observed)))
                ),
                "p_maxT_within_representation_8_greater": float(
                    np.mean(max_positive_8 >= observed - SIGNFLIP_ATOL)
                ),
                "n_sign_patterns": int(len(signs)),
                "roi_analysis_role": "literature_defined_targeted_secondary",
                "mapping_version": NROI.ROI_MAPPING_VERSION,
            }
        )
    result = pd.DataFrame(rows)
    result["q_bh_2x8_two_sided"] = fdr_bh(result["p_exact_two_sided"].to_numpy(float))
    result["family_fwer_2x8_05"] = result["p_maxT_2x8_two_sided"].le(0.05)
    return result


def plot_predefined_roi_effects_06(
    subject_effects: pd.DataFrame,
    inference: pd.DataFrame,
    output_path: Path,
    *,
    game: int,
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(6.0, 5.0), sharex=True, layout="constrained")
    rng = np.random.default_rng(20260820 + int(game))
    colors = {"concat29": BLUE, "mean29": ORANGE}
    for axis, representation in zip(axes, REPRESENTATIONS):
        selected = subject_effects[subject_effects["representation"].eq(representation)]
        tests = inference[inference["representation"].eq(representation)].set_index("roi_key")
        values_by_roi = [
            selected.loc[selected["roi_key"].eq(roi_key), "beta"].to_numpy(float)
            for roi_key in NROI.ROI_ORDER
        ]
        axis.boxplot(
            values_by_roi,
            positions=np.arange(len(NROI.ROI_ORDER)),
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            boxprops={"facecolor": colors[representation], "alpha": 0.18, "edgecolor": colors[representation]},
            medianprops={"color": INK, "linewidth": 1.0},
            whiskerprops={"color": colors[representation], "linewidth": 0.8},
            capprops={"color": colors[representation], "linewidth": 0.8},
        )
        for x, values in enumerate(values_by_roi):
            jitter = rng.uniform(-0.12, 0.12, size=len(values))
            axis.scatter(x + jitter, values, s=15, color=colors[representation], edgecolor="white", linewidth=0.35, zorder=3)
            row = tests.loc[NROI.ROI_ORDER[x]]
            marker = "†" if float(row["p_maxT_2x8_greater"]) < 0.05 else ("*" if float(row["p_exact_greater"]) < 0.05 else "")
            if marker:
                top = max(float(np.max(values)), 0.0)
                axis.annotate(marker, (x, top), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8)
        axis.axhline(0, color="#777B80", linewidth=0.7, linestyle="--")
        axis.set_ylabel("Held-out prediction r", fontsize=7)
        axis.set_title(
            "Full observable trace (concat29)" if representation == "concat29" else "Mean-pooled trace (mean29)",
            fontsize=7.4,
        )
        axis.tick_params(labelsize=6.2)
        axis.grid(axis="y", color=GRID, linewidth=0.45)
    axes[-1].set_xticks(np.arange(len(NROI.ROI_ORDER)), [NROI.ROI_LABELS[key] for key in NROI.ROI_ORDER], rotation=25, ha="right")
    figure.suptitle(f"Game {int(game)} · literature-defined ROI trace prediction", fontsize=8.2)
    figure.text(0.5, 0.005, "* one-sided exact p<.05; † one-sided 2×8 maxT-FWER p<.05", ha="center", fontsize=5.8, color="#6C737A")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, facecolor="white")
    plt.close(figure)


def save_group_brain_maps_06(
    inference: pd.DataFrame,
    output_path: Path,
    *,
    game: int,
) -> None:
    panels = []
    for representation, title in (
        ("concat29", "Full observable trace (concat29)"),
        ("mean29", "Mean-pooled trace (mean29)"),
    ):
        selected = inference[
            inference["representation"].eq(representation)
            & inference["p_max_abs_t_fwer"].lt(0.05)
        ][["roi", "group_beta"]].copy()
        panels.append((title, selected))
    NROI.save_effect_brain_maps(
        output_path,
        panels=panels,
        value_column="group_beta",
        title=f"Game {int(game)} · planning-trace prediction",
        subtitle="Six-participant exact maxT-FWER corrected parcels",
        metric_label="Held-out prediction r",
        signed=True,
        n_columns=2,
    )


def build_paired_game_effects(subject_game: pd.DataFrame) -> pd.DataFrame:
    required = {"subject", "game", "roi", "representation", "effect_id", "beta"}
    missing = required.difference(subject_game.columns)
    if missing:
        raise ValueError(f"Paired-game input lacks {sorted(missing)}")
    data = subject_game.loc[:, sorted(required)].copy()
    if data.duplicated(
        ["subject", "game", "roi", "representation"], keep=False
    ).any():
        raise ValueError("Paired-game input contains duplicate rows")
    wide = data.pivot(
        index=["subject", "roi", "representation", "effect_id"],
        columns="game",
        values="beta",
    )
    if set(wide.columns) != {1, 2} or wide.isna().any().any():
        raise ValueError("Every paired effect must be complete in games 1 and 2")
    rows: list[pd.DataFrame] = []
    for estimand, values in (
        ("across_game_mean", (wide[1] + wide[2]) / 2.0),
        ("game2_minus_game1", wide[2] - wide[1]),
    ):
        frame = values.rename("beta").reset_index()
        frame["estimand"] = estimand
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _short_roi(name: str, maximum: int = 48) -> str:
    text = str(name).replace("ho_cortical_", "").replace("ho_subcortical_", "")
    return text if len(text) <= maximum else text[: maximum - 1] + "…"


def _star(p_value: float) -> str:
    if not np.isfinite(p_value):
        return ""
    if p_value <= 0.001:
        return "***"
    if p_value <= 0.01:
        return "**"
    if p_value <= 0.05:
        return "*"
    return ""


def plot_subject_run_raincloud(
    fold_effects: pd.DataFrame,
    subject_effects: pd.DataFrame,
    path: Path,
) -> None:
    pivot = subject_effects.pivot(index="roi", columns="representation", values="beta")
    top_roi = str(pivot.mean(axis=1).sort_values(ascending=False).index[0])
    selected = fold_effects[fold_effects["roi"].astype(str).eq(top_roi)].copy()
    wide = selected.pivot(
        index="held_out_run", columns="representation", values="beta"
    ).reindex(columns=REPRESENTATIONS)
    figure, axis = plt.subplots(figsize=(7.8, 5.2), constrained_layout=True)
    data = [wide[name].to_numpy(float) for name in REPRESENTATIONS]
    violins = axis.violinplot(data, positions=[0, 1], widths=0.72, showextrema=False)
    for body, representation in zip(violins["bodies"], REPRESENTATIONS):
        body.set_facecolor(REPRESENTATION_COLORS[representation])
        body.set_edgecolor("none")
        body.set_alpha(0.18)
    for row_index, (_, row) in enumerate(wide.iterrows()):
        values = row.to_numpy(float)
        axis.plot([0, 1], values, color=GRID, linewidth=0.8, alpha=0.8, zorder=1)
        jitter = (row_index - (len(wide) - 1) / 2.0) * 0.012
        for index, representation in enumerate(REPRESENTATIONS):
            axis.scatter(
                index + jitter,
                values[index],
                s=32,
                color=REPRESENTATION_COLORS[representation],
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
    axis.axhline(0.0, color=INK, linewidth=0.8)
    axis.set_xticks([0, 1], ["concat29", "mean29"])
    axis.set_ylabel("Held-out multivariate r")
    axis.set_title(f"Run-level paired decoding | {_short_roi(top_roi)}")
    axis.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.7)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=190)
    plt.close(figure)


def plot_concat_mean_scatter(
    effects: pd.DataFrame,
    path: Path,
    *,
    value_column: str,
    title: str,
) -> None:
    pivot = effects.pivot_table(
        index="roi", columns="representation", values=value_column, aggfunc="first"
    ).reindex(columns=("mean29", "concat29"))
    pivot = pivot.dropna()
    figure, axis = plt.subplots(figsize=(6.2, 5.8), constrained_layout=True)
    axis.scatter(
        pivot["mean29"], pivot["concat29"], s=28, color=BLUE, alpha=0.66
    )
    lower = float(np.nanmin(pivot.to_numpy(float)))
    upper = float(np.nanmax(pivot.to_numpy(float)))
    padding = max((upper - lower) * 0.06, 0.01)
    axis.plot(
        [lower - padding, upper + padding],
        [lower - padding, upper + padding],
        color=INK,
        linestyle="--",
        linewidth=0.9,
    )
    axis.set_xlabel("mean29 held-out multivariate r")
    axis.set_ylabel("concat29 held-out multivariate r")
    axis.set_title(title + "\n(descriptive; raw concat contains the mean)")
    axis.grid(color=GRID, linewidth=0.6, alpha=0.65)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=190)
    plt.close(figure)


def plot_representative_predictions(
    example_frame: pd.DataFrame,
    path: Path,
    *,
    roi: str,
    held_out_run: str,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.7), constrained_layout=True)
    for axis, representation in zip(axes, REPRESENTATIONS):
        frame = example_frame[example_frame["representation"].eq(representation)]
        x = frame["actual_z"].to_numpy(float)
        y = frame["predicted_z"].to_numpy(float)
        axis.scatter(
            x,
            y,
            s=7,
            color=REPRESENTATION_COLORS[representation],
            alpha=0.24,
            linewidths=0,
        )
        lower = float(np.nanpercentile(np.concatenate((x, y)), 1))
        upper = float(np.nanpercentile(np.concatenate((x, y)), 99))
        axis.plot([lower, upper], [lower, upper], color=INK, linestyle="--", linewidth=0.8)
        axis.set_xlabel("Observed target (outer-train z)")
        axis.set_ylabel("Predicted target")
        axis.set_title(representation)
        axis.grid(color=GRID, linewidth=0.5, alpha=0.55)
    figure.suptitle(
        f"Representative held-out prediction | {_short_roi(roi)} | {held_out_run}"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=190)
    plt.close(figure)


def plot_group_lollipop(
    inference: pd.DataFrame,
    subject_effects: pd.DataFrame,
    path: Path,
    *,
    title: str,
    top_n: int = 12,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(15.2, 8.0), constrained_layout=True)
    for axis, representation in zip(axes, REPRESENTATIONS):
        frame = inference[inference["representation"].eq(representation)].copy()
        frame = frame.sort_values("group_beta", ascending=False).head(int(top_n))
        frame = frame.sort_values("group_beta", ascending=True).reset_index(drop=True)
        y_positions = np.arange(len(frame), dtype=float)
        for y, row in zip(y_positions, frame.itertuples(index=False)):
            values = subject_effects[
                subject_effects["representation"].eq(representation)
                & subject_effects["roi"].astype(str).eq(str(row.roi))
            ].sort_values("subject")["beta"].to_numpy(float)
            mean = float(row.group_beta)
            sem = float(np.std(values, ddof=1) / math.sqrt(len(values)))
            ci = T95_DF5 * sem
            axis.plot([0.0, mean], [y, y], color=GRID, linewidth=1.2, zorder=1)
            axis.plot([mean - ci, mean + ci], [y, y], color=INK, linewidth=1.4, zorder=2)
            jitter = np.linspace(-0.16, 0.16, len(values))
            axis.scatter(
                values,
                y + jitter,
                s=23,
                color=REPRESENTATION_COLORS[representation],
                alpha=0.55,
                edgecolor="white",
                linewidth=0.35,
                zorder=3,
            )
            axis.scatter(
                [mean], [y], s=58, color=REPRESENTATION_COLORS[representation], zorder=4
            )
            stars = _star(float(row.p_max_abs_t_fwer))
            if stars:
                axis.text(mean + ci + 0.004, y, stars, va="center", fontsize=10)
        axis.axvline(0.0, color=INK, linewidth=0.8)
        axis.set_yticks(y_positions, [_short_roi(name) for name in frame["roi"]])
        axis.set_xlabel("Group mean held-out multivariate r")
        axis.set_title(representation)
        axis.grid(axis="x", color=GRID, linewidth=0.6, alpha=0.7)
    figure.suptitle(title + " | dots: 6 subjects; bars: 95% t CI; stars: family max-T")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=190)
    plt.close(figure)


def _prepare_output_root(path: Path, *, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory already contains files: {path}. Use --overwrite explicitly."
        )
    path.mkdir(parents=True, exist_ok=True)


def run_subject_analysis(
    args: argparse.Namespace,
    *,
    trace_module: Any,
    brain_common: Any,
) -> None:
    source_signatures = analysis_source_signatures()
    adapter = StrictTraceAdapter(trace_module)
    run_specs, cache_audit, trace_audit = adapter.prepare(args)
    official_eligible = (
        len(run_specs) == EXPECTED_RUNS
        and str(args.roi_set) == "all"
        and not bool(args.roi_include)
        and int(args.max_roi_voxels) == 0
        and int(args.required_status2) == N_EVENTS
        and int(args.random_state) == 2026
        and int(args.participant_permutations) == PARTICIPANT_PERMUTATIONS
        and int(args.participant_block_size) == PARTICIPANT_BLOCK_SIZE_TR
    )
    if args.max_runs is None and len(run_specs) != EXPECTED_RUNS:
        raise ValueError(
            f"Official subject analysis requires exactly 11 runs, got {len(run_specs)}"
        )
    runs, run_audits = load_trace_runs(
        run_specs,
        trace_adapter=adapter,
        brain_common=brain_common,
        trace_root=Path(args.planning_trace_root),
    )
    rois = list(brain_common.common_roi_names(run_specs))
    if args.roi_include:
        tokens = tuple(
            part.strip().lower() for part in str(args.roi_include).split(",") if part.strip()
        )
        rois = [roi for roi in rois if any(token in roi.lower() for token in tokens)]
    if not rois:
        raise ValueError("No common ROI remains after selection")
    if official_eligible and len(rois) != EXPECTED_ROIS:
        raise ValueError(f"Official analysis requires 110 common ROIs, got {len(rois)}")

    output_dir = (
        Path(args.output_root)
        / f"sub{int(args.subject):03d}_game{int(args.game)}"
    )
    _prepare_output_root(output_dir, overwrite=bool(args.overwrite))
    result_dir = output_dir / "results"
    figure_dir = output_dir / "figures"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    log(f"Precomputing {len(runs)} outer-fold controls and targets")
    prepared_folds = precompute_loro_folds(runs, random_state=int(args.random_state))
    block_orders = synchronized_block_orders(
        runs,
        n_permutations=int(args.participant_permutations),
        block_size=int(args.participant_block_size),
        random_state=int(args.random_state),
    )
    fold_frames: list[pd.DataFrame] = []
    n_tr_max = max(int(run.run.n_tr) for run in runs)
    pattern_cube = np.full(
        (len(rois), len(runs), len(REPRESENTATIONS), n_tr_max),
        np.nan,
        dtype=np.float32,
    )
    example_by_roi: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    participant_null_cube = np.full(
        (len(rois), len(REPRESENTATIONS), int(args.participant_permutations)),
        np.nan,
        dtype=np.float32,
    )
    for roi_index, roi in enumerate(rois, start=1):
        log(f"ROI {roi_index}/{len(rois)}: {roi}")
        patterns: list[np.ndarray] = []
        for run in run_specs:
            values, _ = brain_common.load_roi(run, roi)
            patterns.append(
                _as_finite_2d(
                    values, name=f"{run.run_label}:{roi}", n_rows=int(run.n_tr)
                )
            )
        result, pattern_by_rep, examples, participant_nulls = decode_roi_loro(
            patterns,
            runs,
            prepared_folds,
            roi_name=roi,
            random_state=int(args.random_state) + roi_index * 10007,
            block_orders_by_run=block_orders,
            permutation_block_size=int(args.participant_block_size),
        )
        result["official_eligible"] = bool(official_eligible)
        fold_frames.append(result)
        for representation_index, representation in enumerate(REPRESENTATIONS):
            values = pattern_by_rep[representation]
            pattern_cube[roi_index - 1, :, representation_index, : values.shape[1]] = values
        example_by_roi[roi] = examples
        for representation_index, representation in enumerate(REPRESENTATIONS):
            participant_null_cube[roi_index - 1, representation_index] = participant_nulls[representation]

    fold_effects = pd.concat(fold_frames, ignore_index=True)
    expected_fold_rows = len(rois) * len(runs) * len(REPRESENTATIONS)
    if len(fold_effects) != expected_fold_rows:
        raise AssertionError(
            f"Fold row contract failed: {len(fold_effects)} != {expected_fold_rows}"
        )
    subject_effects = subject_effects_from_folds(
        fold_effects,
        subject=int(args.subject),
        game=int(args.game),
        expected_runs=len(runs),
    )
    subject_effects["official_eligible"] = bool(official_eligible)
    participant_inference = participant_block_inference(
        subject_effects,
        rois=rois,
        null_cube=participant_null_cube,
    )
    participant_inference["official_eligible"] = bool(official_eligible)

    atomic_write_csv(fold_effects, result_dir / "loro_fold_effects.csv.gz")
    atomic_write_csv(subject_effects, result_dir / "subject_effects.csv")
    atomic_write_csv(
        participant_inference,
        result_dir / "participant_block_permutation_inference.csv",
    )
    atomic_savez(
        result_dir / "participant_block_permutation_nulls.npz",
        null_correlations=participant_null_cube,
        family_max_null=np.max(participant_null_cube, axis=(0, 1)),
        roi_names=np.asarray(rois),
        representations=np.asarray(REPRESENTATIONS),
        n_permutations=np.asarray([int(args.participant_permutations)], dtype=np.int32),
        block_size_tr=np.asarray([int(args.participant_block_size)], dtype=np.int32),
    )
    atomic_savez(
        result_dir / "per_tr_pattern_correlation.npz",
        pattern_correlation=pattern_cube,
        roi_names=np.asarray(rois),
        run_labels=np.asarray([run.run_label for run in runs]),
        representations=np.asarray(REPRESENTATIONS),
        n_tr_by_run=np.asarray([int(run.run.n_tr) for run in runs], dtype=np.int32),
    )

    pivot = subject_effects.pivot(index="roi", columns="representation", values="beta")
    representative_roi = str(pivot.mean(axis=1).sort_values(ascending=False).index[0])
    example_rows: list[dict[str, Any]] = []
    held_out_run = runs[0].run_label
    for representation in REPRESENTATIONS:
        example = example_by_roi[representative_roi][representation]
        held_out_run = str(example["held_out_run"][0])
        for index, actual, predicted in zip(
            example["element_index"], example["actual"], example["predicted"]
        ):
            example_rows.append(
                {
                    "roi": representative_roi,
                    "representation": representation,
                    "held_out_run": held_out_run,
                    "element_index": int(index),
                    "actual_z": float(actual),
                    "predicted_z": float(predicted),
                }
            )
    example_frame = pd.DataFrame(example_rows)
    atomic_write_csv(example_frame, result_dir / "representative_prediction_points.csv.gz")

    plot_subject_run_raincloud(
        fold_effects,
        subject_effects,
        figure_dir / "figure1_run_raincloud_paired_dots.png",
    )
    plot_concat_mean_scatter(
        subject_effects,
        figure_dir / "figure2_concat29_vs_mean29_scatter.png",
        value_column="beta",
        title=f"Subject {int(args.subject):03d} | Game {int(args.game)} | ROI decoding",
    )
    plot_representative_predictions(
        example_frame,
        figure_dir / "figure3_representative_prediction.png",
        roi=representative_roi,
        held_out_run=held_out_run,
    )
    participant_panels = []
    for representation, label in (
        ("concat29", "Full observable trace (concat29)"),
        ("mean29", "Mean-pooled trace (mean29)"),
    ):
        selected = participant_inference[
            participant_inference["representation"].eq(representation)
            & participant_inference["p_block_maxT_fwer"].lt(0.05)
            & participant_inference["beta"].gt(0)
        ][["roi", "beta"]].copy()
        participant_panels.append((label, selected))
    NROI.save_effect_brain_maps(
        figure_dir / "figure4_participant_corrected_brain_maps.pdf",
        panels=participant_panels,
        value_column="beta",
        title=(
            f"Subject {int(args.subject):03d} · Game {int(args.game)} · "
            "planning-trace prediction"
        ),
        subtitle="Within-participant 40-TR block-permutation maxT-FWER corrected parcels",
        metric_label="Held-out prediction r",
        signed=False,
        n_columns=2,
    )

    verify_source_signatures(source_signatures)
    manifest = {
        "analysis_name": ANALYSIS_NAME,
        "analysis_version": ANALYSIS_VERSION,
        "analysis_source_signatures": source_signatures,
        "script_signature": file_signature(Path(__file__)),
        "planning_trace_common_signature": file_signature(TRACE_COMMON_PATH),
        "planning_brain_common_signature": file_signature(BRAIN_COMMON_PATH),
        "neural_roi_figures_signature": file_signature(Path(NROI.__file__)),
        "subject": int(args.subject),
        "game": int(args.game),
        "official_eligible": bool(official_eligible),
        "run_labels": [run.run_label for run in runs],
        "n_runs": len(runs),
        "roi_names": rois,
        "n_rois": len(rois),
        "representations": list(REPRESENTATIONS),
        "target_contract": {
            "n_events": N_EVENTS,
            "token_components": list(EXPECTED_TOKEN_COMPONENTS),
            "concat_layout": "event-major raw 29 x 12",
            "concat_contains_mean": True,
            "last_event": (
                "event 28 retained with contemporaneous return_gain_from_root; "
                "no next-event outcome"
            ),
            "structural_q_missing": "q_revision=0 plus q_revision_valid mask",
        },
        "controls": [
            "pre-search model-prior policy P",
            "pre-search current-action margin M",
            "pre-search initial Q state Q0",
            "decision-event density/timing",
            "visual outer-training PCA20",
            "current action",
            "previous action",
            "task history",
            "linear/quadratic drift",
        ],
        "preprocessing": (
            "outer-training-only control scaling/visual PCA, ridge nuisance "
            "residualization of neural and target matrices, target-column "
            "scaling, neural PCA<=64, fixed multi-output ridge"
        ),
        "primary_estimand": (
            "pooled held-out multivariate Pearson r after outer-training-only "
            "per-target-dimension scaling; equal mean across 11 held-out runs"
        ),
        "participant_inference": (
            f"{int(args.participant_permutations)} synchronized within-run "
            f"{int(args.participant_block_size)}-TR target-block permutations; "
            "one-sided greater-than null; max statistic across 2 x 110 cells"
        ),
        "secondary_estimands": [
            "dimension-equal held-out CV-R2 against outer-training mean",
            "per-TR target-pattern correlation",
        ],
        "config": {
            "visual_pca_components": VISUAL_PCA_COMPONENTS,
            "neural_pca_components": NEURAL_PCA_COMPONENTS,
            "ridge_alpha": RIDGE_ALPHA,
            "nuisance_alpha": NUISANCE_ALPHA,
            "random_state": int(args.random_state),
            "participant_permutations": int(args.participant_permutations),
            "participant_block_size_tr": int(args.participant_block_size),
        },
        "row_contracts": {
            "loro_fold_effects": expected_fold_rows,
            "subject_effects": len(rois) * len(REPRESENTATIONS),
            "participant_block_permutation_inference": len(rois)
            * len(REPRESENTATIONS),
            "per_tr_pattern_correlation_shape": list(pattern_cube.shape),
        },
        "shared_cache_audit": cache_audit,
        "shared_trace_audit": trace_audit,
        "run_trace_audits": run_audits,
        "outputs": {
            "fold_effects": file_signature(result_dir / "loro_fold_effects.csv.gz"),
            "subject_effects": file_signature(result_dir / "subject_effects.csv"),
            "participant_block_permutation_inference": file_signature(
                result_dir / "participant_block_permutation_inference.csv"
            ),
            "participant_block_permutation_nulls": file_signature(
                result_dir / "participant_block_permutation_nulls.npz"
            ),
            "per_tr_pattern_correlation": file_signature(
                result_dir / "per_tr_pattern_correlation.npz"
            ),
            "representative_prediction_points": file_signature(
                result_dir / "representative_prediction_points.csv.gz"
            ),
        },
    }
    atomic_write_json(result_dir / "analysis_manifest.json", manifest)
    log(f"Wrote {output_dir}")


def _load_subject_effect_files(
    output_root: Path, games: Sequence[int]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    signatures: list[dict[str, Any]] = []
    for subject in EXPECTED_SUBJECTS:
        for game in games:
            path = (
                Path(output_root)
                / f"sub{subject:03d}_game{int(game)}"
                / "results"
                / "subject_effects.csv"
            )
            if not path.is_file():
                raise FileNotFoundError(f"Missing official subject effects: {path}")
            manifest_path = path.with_name("analysis_manifest.json")
            if not manifest_path.is_file():
                raise FileNotFoundError(f"Missing subject analysis manifest: {manifest_path}")
            manifest_signature_before = file_signature(manifest_path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_signature_after = file_signature(manifest_path)
            if manifest_signature_before != manifest_signature_after:
                raise RuntimeError(f"Manifest changed while being read: {manifest_path}")
            if manifest.get("analysis_name") != ANALYSIS_NAME or manifest.get("analysis_version") != ANALYSIS_VERSION:
                raise ValueError(f"Subject analysis identity/version mismatch: {manifest_path}")
            if manifest.get("official_eligible") is not True or int(manifest.get("n_runs", -1)) != EXPECTED_RUNS or int(manifest.get("n_rois", -1)) != EXPECTED_ROIS:
                raise ValueError(f"Subject manifest is not official 11-run/110-ROI output: {manifest_path}")
            if tuple(manifest.get("representations", ())) != REPRESENTATIONS:
                raise ValueError(f"Subject representation manifest changed: {manifest_path}")
            subject_script_sha256 = _mapping_from_metadata(
                manifest, "script_signature"
            ).get("sha256")
            if subject_script_sha256 not in (
                {sha256_file(Path(__file__))} | COMPATIBLE_SUBJECT_SCRIPT_SHA256
            ):
                raise ValueError(f"Subject output was produced by another script revision: {manifest_path}")
            if _mapping_from_metadata(manifest, "planning_trace_common_signature").get("sha256") != sha256_file(TRACE_COMMON_PATH):
                raise ValueError(f"Subject output uses another trace-common revision: {manifest_path}")
            if _mapping_from_metadata(manifest, "planning_brain_common_signature").get("sha256") != sha256_file(BRAIN_COMMON_PATH):
                raise ValueError(f"Subject output uses another brain-common revision: {manifest_path}")
            subject_figure_sha256 = _mapping_from_metadata(
                manifest, "neural_roi_figures_signature"
            ).get("sha256")
            if subject_figure_sha256 not in (
                {sha256_file(Path(NROI.__file__))}
                | COMPATIBLE_SUBJECT_NEURAL_FIGURE_SHA256
            ):
                raise ValueError(f"Subject output uses another neural-figure revision: {manifest_path}")
            signature_before = file_signature(path)
            frame = pd.read_csv(path)
            signature_after = file_signature(path)
            if signature_before != signature_after:
                raise RuntimeError(f"Source changed while being read: {path}")
            recorded_subject_effects = _mapping_from_metadata(_mapping_from_metadata(manifest, "outputs"), "subject_effects")
            if recorded_subject_effects.get("sha256") != signature_before.get("sha256") or int(recorded_subject_effects.get("size", -1)) != int(signature_before.get("size", -2)):
                raise ValueError(f"Subject effects artifact no longer matches its manifest: {path}")
            if "official_eligible" not in frame or not frame["official_eligible"].astype(str).str.lower().eq("true").all():
                raise ValueError(f"Non-official subject effects cannot enter group analysis: {path}")
            if set(pd.to_numeric(frame["subject"], errors="raise").astype(int)) != {subject}:
                raise ValueError(f"Subject identity mismatch: {path}")
            if set(pd.to_numeric(frame["game"], errors="raise").astype(int)) != {int(game)}:
                raise ValueError(f"Game identity mismatch: {path}")
            required = {"subject", "game", "roi", "representation", "effect_id", "beta", "n_runs", "primary_metric", "official_eligible"}
            missing = required.difference(frame.columns)
            if missing:
                raise ValueError(f"Subject effect schema lacks {sorted(missing)}: {path}")
            if len(frame) != len(REPRESENTATIONS) * EXPECTED_ROIS:
                raise ValueError(f"Official subject row count is {len(frame)}, expected 220: {path}")
            if set(frame["representation"].astype(str)) != set(REPRESENTATIONS):
                raise ValueError(f"Representation contract mismatch: {path}")
            if set(frame["primary_metric"].astype(str)) != {"heldout_dimension_standardized_pooled_multivariate_r"}:
                raise ValueError(f"Primary estimand changed: {path}")
            if not pd.to_numeric(frame["n_runs"], errors="raise").eq(EXPECTED_RUNS).all():
                raise ValueError(f"Official subject effects do not use 11 equal LORO folds: {path}")
            expected_ids = frame["representation"].map(lambda name: f"trace_decode::{name}")
            if not frame["effect_id"].astype(str).eq(expected_ids).all():
                raise ValueError(f"Effect-id/representation mismatch: {path}")
            roi_sets = [set(frame.loc[frame["representation"].eq(name), "roi"].astype(str)) for name in REPRESENTATIONS]
            if any(len(values) != EXPECTED_ROIS for values in roi_sets) or roi_sets[0] != roi_sets[1]:
                raise ValueError(f"Representation ROI contracts differ or are not 110-wide: {path}")
            if not np.all(np.isfinite(pd.to_numeric(frame["beta"], errors="coerce").to_numpy(float))):
                raise ValueError(f"Non-finite official subject beta: {path}")
            frames.append(frame)
            signatures.append({"subject_effects": signature_before, "analysis_manifest": manifest_signature_before})
    return pd.concat(frames, ignore_index=True), signatures


def run_group_game(args: argparse.Namespace, *, game: int) -> None:
    source_signatures = analysis_source_signatures()
    subject_effects, sources = _load_subject_effect_files(Path(args.output_root), [game])
    inference = exact_group_inference(
        subject_effects,
        family_name=f"trace_representations_2x110_game{game}",
    )
    predefined_subject = predefined_roi_subject_effects(subject_effects)
    predefined_inference = exact_predefined_roi_inference(predefined_subject)
    predefined_inference.insert(0, "game", int(game))
    output_dir = Path(args.output_root) / f"group_game{game}"
    _prepare_output_root(output_dir, overwrite=bool(args.overwrite))
    result_dir = output_dir / "results"
    figure_dir = output_dir / "figures"
    atomic_write_csv(subject_effects, result_dir / "subject_effects_input.csv")
    atomic_write_csv(inference, result_dir / "group_inference.csv")
    atomic_write_csv(
        NROI.predefined_roi_mapping_frame(),
        result_dir / "predefined_roi_mapping.csv",
    )
    atomic_write_csv(
        predefined_subject,
        result_dir / "predefined_roi_subject_effects.csv",
    )
    atomic_write_csv(
        predefined_inference,
        result_dir / "predefined_roi_inference.csv",
    )
    plot_group_lollipop(
        inference,
        subject_effects,
        figure_dir / "figure1_top_roi_lollipop_subject_points.png",
        title=f"Game {game} planning-trace decoding",
        top_n=int(args.top_rois),
    )
    plot_concat_mean_scatter(
        inference,
        figure_dir / "figure2_concat29_vs_mean29_scatter.png",
        value_column="group_beta",
        title=f"Game {game} group ROI decoding",
    )
    save_group_brain_maps_06(
        inference,
        figure_dir / "figure3_group_corrected_brain_maps.pdf",
        game=int(game),
    )
    NROI.save_predefined_roi_anatomy(
        figure_dir / "figure4_predefined_roi_anatomy.pdf"
    )
    plot_predefined_roi_effects_06(
        predefined_subject,
        predefined_inference,
        figure_dir / "figure5_predefined_roi_effects.pdf",
        game=int(game),
    )
    verify_source_signatures(source_signatures)
    manifest = {
        "analysis_name": ANALYSIS_NAME,
        "analysis_version": ANALYSIS_VERSION,
        "analysis_source_signatures": source_signatures,
        "mode": "group_game",
        "game": int(game),
        "script_signature": file_signature(Path(__file__)),
        "source_signatures": sources,
        "subjects": list(EXPECTED_SUBJECTS),
        "inference": (
            "64 exhaustive synchronized subject sign patterns; two-sided exact p; "
            "max-|t| across concat29/mean29 x 110 ROIs"
        ),
        "correction_family_size": len(REPRESENTATIONS) * EXPECTED_ROIS,
        "predefined_roi_analysis": {
            "role": "literature_defined_targeted_secondary_not_preregistered",
            "mapping_version": NROI.ROI_MAPPING_VERSION,
            "mapping_source": NROI.ROI_MAPPING_SOURCE,
            "hemisphere_weighting": "equal_left_right_mean",
            "planned_tests": "2 representations x 8 bilateral ROIs",
            "reported_inference": (
                "two-sided exact p, one-sided decoding p, 2x8 maxT, "
                "within-representation 8-ROI maxT, and BH across 2x8"
            ),
        },
        "row_contracts": {
            "subject_effects_input": len(EXPECTED_SUBJECTS)
            * len(REPRESENTATIONS)
            * EXPECTED_ROIS,
            "group_inference": len(REPRESENTATIONS) * EXPECTED_ROIS,
            "predefined_roi_subject_effects": len(EXPECTED_SUBJECTS)
            * len(REPRESENTATIONS)
            * len(NROI.ROI_ORDER),
            "predefined_roi_inference": len(REPRESENTATIONS)
            * len(NROI.ROI_ORDER),
        },
        "outputs": {
            "subject_effects_input": file_signature(
                result_dir / "subject_effects_input.csv"
            ),
            "group_inference": file_signature(result_dir / "group_inference.csv"),
            "predefined_roi_mapping": file_signature(
                result_dir / "predefined_roi_mapping.csv"
            ),
            "predefined_roi_subject_effects": file_signature(
                result_dir / "predefined_roi_subject_effects.csv"
            ),
            "predefined_roi_inference": file_signature(
                result_dir / "predefined_roi_inference.csv"
            ),
        },
    }
    atomic_write_json(result_dir / "analysis_manifest.json", manifest)
    log(f"Wrote {output_dir}")


def run_paired_games(args: argparse.Namespace) -> None:
    source_signatures = analysis_source_signatures()
    subject_game, sources = _load_subject_effect_files(Path(args.output_root), [1, 2])
    paired = build_paired_game_effects(subject_game)
    inference_frames: list[pd.DataFrame] = []
    for estimand, frame in paired.groupby("estimand", sort=False):
        inference = exact_group_inference(
            frame,
            family_name=f"trace_representations_2x110_{estimand}",
        )
        inference.insert(0, "estimand", str(estimand))
        inference_frames.append(inference)
    all_inference = pd.concat(inference_frames, ignore_index=True)
    output_dir = Path(args.output_root) / "group_paired_games"
    _prepare_output_root(output_dir, overwrite=bool(args.overwrite))
    result_dir = output_dir / "results"
    figure_dir = output_dir / "figures"
    atomic_write_csv(paired, result_dir / "paired_subject_effects.csv")
    atomic_write_csv(all_inference, result_dir / "paired_group_inference.csv")
    for estimand, title in (
        (
            "across_game_mean",
            "Equal-weight mean of two observed games · planning-trace decoding",
        ),
        ("game2_minus_game1", "Game 2 minus Game 1 planning-trace decoding"),
    ):
        plot_group_lollipop(
            all_inference[all_inference["estimand"].eq(estimand)],
            paired[paired["estimand"].eq(estimand)],
            figure_dir / f"figure_{estimand}_top_roi_lollipop.png",
            title=title,
            top_n=int(args.top_rois),
        )
        plot_concat_mean_scatter(
            all_inference[all_inference["estimand"].eq(estimand)],
            figure_dir / f"figure_{estimand}_concat29_vs_mean29.png",
            value_column="group_beta",
            title=title,
        )
    verify_source_signatures(source_signatures)
    manifest = {
        "analysis_name": ANALYSIS_NAME,
        "analysis_version": ANALYSIS_VERSION,
        "analysis_source_signatures": source_signatures,
        "mode": "paired_games",
        "script_signature": file_signature(Path(__file__)),
        "source_signatures": sources,
        "subjects": list(EXPECTED_SUBJECTS),
        "estimands": {
            "across_game_mean": "within-subject (Game1 + Game2) / 2",
            "game2_minus_game1": "within-subject Game2 - Game1",
        },
        "inference": (
            "64 exhaustive synchronized subject sign patterns; separate "
            "representation x ROI max-|t| family within each paired estimand"
        ),
        "correction_family_size_per_estimand": len(REPRESENTATIONS) * EXPECTED_ROIS,
        "row_contracts": {
            "paired_subject_effects": len(EXPECTED_SUBJECTS)
            * len(REPRESENTATIONS)
            * EXPECTED_ROIS
            * 2,
            "paired_group_inference": len(REPRESENTATIONS) * EXPECTED_ROIS * 2,
        },
        "outputs": {
            "paired_subject_effects": file_signature(
                result_dir / "paired_subject_effects.csv"
            ),
            "paired_group_inference": file_signature(
                result_dir / "paired_group_inference.csv"
            ),
        },
    }
    atomic_write_json(result_dir / "analysis_manifest.json", manifest)
    log(f"Wrote {output_dir}")


def self_test(random_state: int = 2026) -> dict[str, Any]:
    rng = np.random.default_rng(int(random_state))
    identity = rng.normal(size=(24, 7))
    if not np.isclose(pooled_multivariate_correlation(identity, identity), 1.0):
        raise AssertionError("Perfect multivariate correlation is not one")
    if not np.isclose(dimension_equal_cv_r2(identity, identity), 1.0):
        raise AssertionError("Perfect dimension-equal CV-R2 is not one")
    pattern = per_tr_pattern_correlation(identity, identity)
    if not np.allclose(pattern, 1.0):
        raise AssertionError("Perfect per-TR pattern correlations are not one")

    n_runs = 4
    n_tr = 72
    mean_dim = 4
    concat_dim = 12
    synthetic_runs: list[TraceRun] = []
    neural_by_run: list[np.ndarray] = []
    mean_weights = rng.normal(size=(mean_dim, 18))
    concat_weights = rng.normal(size=(concat_dim, 18))
    for run_index in range(n_runs):
        controls = rng.normal(size=(n_tr, 5))
        visual = rng.normal(size=(n_tr, 24))
        latent = rng.normal(size=(n_tr, 5))
        mean_target = latent @ rng.normal(size=(5, mean_dim)) + 0.08 * rng.normal(
            size=(n_tr, mean_dim)
        )
        concat_target = latent @ rng.normal(size=(5, concat_dim)) + 0.08 * rng.normal(
            size=(n_tr, concat_dim)
        )
        neural = (
            mean_target @ mean_weights
            + concat_target @ concat_weights
            + controls @ rng.normal(size=(5, 18))
            + 0.12 * rng.normal(size=(n_tr, 18))
        )
        stub = argparse.Namespace(
            run_label=f"run{run_index + 1}",
            subject=1,
            game=1,
            n_tr=n_tr,
        )
        synthetic_runs.append(
            TraceRun(
                run=stub,
                run_label=stub.run_label,
                subject=1,
                game=1,
                targets={"concat29": concat_target, "mean29": mean_target},
                target_component_names={
                    "concat29": tuple(f"c{index}" for index in range(concat_dim)),
                    "mean29": tuple(f"m{index}" for index in range(mean_dim)),
                },
                base_controls=controls,
                visual=visual,
                metadata={},
            )
        )
        neural_by_run.append(neural)
    prepared = precompute_loro_folds(synthetic_runs, random_state=int(random_state))
    decoded, patterns, _, _ = decode_roi_loro(
        neural_by_run,
        synthetic_runs,
        prepared,
        roi_name="synthetic_roi",
        random_state=int(random_state),
        example_limit=100,
    )
    if len(decoded) != n_runs * len(REPRESENTATIONS):
        raise AssertionError("Synthetic LORO row count failed")
    if float(decoded["multivariate_r"].mean()) < 0.60:
        raise AssertionError("Synthetic signal was not recovered")
    if any(values.shape != (n_runs, n_tr) for values in patterns.values()):
        raise AssertionError("Synthetic per-TR pattern output shape failed")
    simple = rng.normal(size=(72, 5))
    simple_prediction = simple + 0.05 * rng.normal(size=simple.shape)
    simple_orders = np.vstack([np.arange(6), np.arange(6)[::-1]])
    simple_null = block_permuted_pooled_correlations(
        simple, simple_prediction, simple_orders, block_size=12
    )
    if not np.isclose(
        simple_null[0], pooled_multivariate_correlation(simple, simple_prediction)
    ):
        raise AssertionError("Participant block-null identity failed")

    rows: list[dict[str, Any]] = []
    for subject in EXPECTED_SUBJECTS:
        for roi_index in range(EXPECTED_ROIS):
            for representation in REPRESENTATIONS:
                rows.append(
                    {
                        "subject": subject,
                        "roi": f"roi{roi_index:03d}",
                        "representation": representation,
                        "effect_id": f"trace_decode::{representation}",
                        "beta": 0.08
                        + 0.002 * subject
                        + rng.normal(scale=0.004),
                    }
                )
    inference = exact_group_inference(
        pd.DataFrame(rows), family_name="synthetic_2x110"
    )
    if len(inference) != len(REPRESENTATIONS) * EXPECTED_ROIS:
        raise AssertionError("Synthetic group row contract failed")
    if np.any(
        inference["p_max_abs_t_fwer"].to_numpy(float)
        + EPS
        < inference["p_exact_two_sided"].to_numpy(float)
    ):
        raise AssertionError("max-T p value is smaller than its raw p value")
    return {
        "status": "PASS",
        "analysis_version": ANALYSIS_VERSION,
        "synthetic_fold_rows": int(len(decoded)),
        "synthetic_mean_multivariate_r": float(decoded["multivariate_r"].mean()),
        "synthetic_group_rows": int(len(inference)),
        "signflip_patterns": 64,
    }


def build_parser(*, trace_module: Any | None = None, brain_common: Any | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    if brain_common is not None:
        brain_common.add_shared_cli_arguments(parser, require_subject_game=False)
    if trace_module is not None and callable(
        getattr(trace_module, "add_trace_cli_arguments", None)
    ):
        add_arguments = getattr(trace_module, "add_trace_cli_arguments")
        try:
            add_arguments(parser, require_subject_game=False)
        except TypeError:
            add_arguments(parser)
    elif brain_common is not None:
        brain_common.add_shared_cli_arguments(parser, require_subject_game=False)
        parser.add_argument(
            "--planning-trace-root", type=Path, default=DEFAULT_TRACE_ROOT
        )
    else:
        parser.add_argument("--random-state", type=int, default=2026)
    if not any(action.dest == "planning_trace_root" for action in parser._actions):
        parser.add_argument("--planning-trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--render-predefined-roi-anatomy",
        action="store_true",
        help=(
            "Render only the literature-defined bilateral ROI anatomy PDF "
            "for --game, without decoding or group inference"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--aggregate-group", action="store_true")
    parser.add_argument("--aggregate-paired", action="store_true")
    parser.add_argument("--top-rois", type=int, default=12)
    parser.add_argument(
        "--participant-permutations",
        type=int,
        default=PARTICIPANT_PERMUTATIONS,
        help="Within-participant held-out target block permutations.",
    )
    parser.add_argument(
        "--participant-block-size",
        type=int,
        default=PARTICIPANT_BLOCK_SIZE_TR,
        help="Temporal block size (TR) for participant permutation inference.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    supplied = list(sys.argv[1:] if argv is None else argv)
    if "--self-test" in supplied:
        parser = build_parser()
        args = parser.parse_args(supplied)
        print(json.dumps(self_test(int(args.random_state)), indent=2, sort_keys=True))
        return

    trace_module = _load_local_module(TRACE_COMMON_PATH, "planning_trace_common_06")
    brain_common = _load_local_module(BRAIN_COMMON_PATH, "planning_brain_common_06_trace")
    parser = build_parser(trace_module=trace_module, brain_common=brain_common)
    args = parser.parse_args(supplied)
    if bool(args.render_predefined_roi_anatomy):
        if args.game not in (1, 2):
            parser.error(
                "--render-predefined-roi-anatomy requires --game 1 or --game 2"
            )
        anatomy_path = (
            Path(args.output_root)
            / f"group_game{int(args.game)}"
            / "figures"
            / "figure4_predefined_roi_anatomy.pdf"
        )
        NROI.save_predefined_roi_anatomy(anatomy_path)
        print(f"Predefined ROI anatomy -> {anatomy_path}")
        return
    if bool(args.aggregate_group) and bool(args.aggregate_paired):
        parser.error("Choose at most one of --aggregate-group and --aggregate-paired")
    if int(args.top_rois) <= 0:
        parser.error("--top-rois must be positive")
    if args.aggregate_paired:
        run_paired_games(args)
        return
    if args.aggregate_group:
        if args.game not in (1, 2):
            parser.error("--aggregate-group requires --game 1 or --game 2")
        run_group_game(args, game=int(args.game))
        return
    if args.subject is None or args.game not in (1, 2):
        parser.error("Subject mode requires --subject and --game")
    run_subject_analysis(args, trace_module=trace_module, brain_common=brain_common)


if __name__ == "__main__":
    main()
