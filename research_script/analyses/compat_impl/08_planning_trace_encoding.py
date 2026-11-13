#!/usr/bin/env python3
"""Cross-validated voxel encoding of observed 29-event planning traces.

This additive analysis leaves the frozen 08 localization analysis and all of
its outputs untouched.  It compares two transparent representations derived
only from the observed Thinker search log:

* ``concat29`` preserves the position of every one of the 29 status-2 events;
* ``mean29`` averages the same event features across the complete window.

No DQN activation, Thinker hidden state, im-vector, early/middle/late bin, or
event-specific fMRI timestamp is used.  Each completed trace is a decision-
locked state regressor.  Leave-one-run-out encoding asks whether each trace
representation predicts held-out voxel activity beyond visual, action,
task-history, drift, generic decision-density, and pre-search P/M/Q0 controls.
The concat-versus-mean display is descriptive, not a pure order contrast.
Population inference uses six subjects and all 64 exact sign patterns with
synchronized max-|t| correction across both representations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from scipy import stats
from sklearn.decomposition import PCA

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import planning_brain_common as COMMON
import planning_trace_common as TRACE


ANALYSIS_VERSION = "planning-trace-encoding-v2-nested-loro"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs" / "08_planning_trace_encoding"
REPRESENTATIONS = ("planning_concat29", "planning_mean29")
REPRESENTATION_LABELS = {
    "planning_concat29": "Concat-29 (position-preserving)",
    "planning_mean29": "Mean-29 (pooled)",
}
N_SUBJECTS = 6
N_RUNS = 11
N_ROIS = 110
N_TR = 480
VISUAL_PCS = 20
TRACE_PCS = 32
NUISANCE_ALPHA = 1.0
TRACE_ALPHA_GRID = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e7, 1e12)
ALPHA_SELECTION_VOXELS = 256
EPS = 1e-12
COMMON_PATH = Path(COMMON.__file__).resolve()
TRACE_COMMON_PATH = Path(TRACE.__file__).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def current_source_hashes() -> Dict[str, str]:
    return {
        "analysis_script_sha256": sha256_file(SCRIPT_PATH),
        "planning_brain_common_sha256": sha256_file(COMMON_PATH),
        "planning_trace_common_sha256": sha256_file(TRACE_COMMON_PATH),
    }


def trace_artifact_sha256(metadata: Mapping[str, Any]) -> str:
    artifact = metadata.get("artifact")
    if not isinstance(artifact, Mapping):
        raise ValueError("Planning-trace metadata lacks artifact signature")
    token = str(artifact.get("sha256", ""))
    if len(token) != 64 or any(character not in "0123456789abcdef" for character in token):
        raise ValueError("Planning-trace artifact SHA256 is malformed")
    return token


def coordinate_hash_selection(
    coordinates: np.ndarray, limit: int = ALPHA_SELECTION_VOXELS
) -> np.ndarray:
    """Outcome-independent, atlas-wide deterministic tuning subset."""
    coords = np.asarray(coordinates, dtype=np.int32)
    if coords.ndim != 2 or coords.shape[1] != 3 or len(coords) == 0:
        raise ValueError("Atlas coordinates must be a non-empty n x 3 array")
    ranked = sorted(
        range(len(coords)),
        key=lambda index: hashlib.sha256(
            np.ascontiguousarray(coords[index]).tobytes()
        ).digest(),
    )
    return np.asarray(sorted(ranked[: min(int(limit), len(coords))]), dtype=np.int32)


def scientific_contract(random_state: int) -> Dict[str, Any]:
    return {
        "trace_summary": (
            "complete concat29 and complete mean29 evaluated separately; no bins"
        ),
        "neural_model": "outer LORO residualized voxel encoding",
        "controls": [
            "visual_PCA20",
            "current_action",
            "previous_action",
            "task_history",
            "drift",
            "generic_decision_event_HRF",
            "presearch_P_policy",
            "presearch_M_margin",
            "presearch_Q0_margin",
        ],
        "trace_sidecar_normalization": (
            "fixed per-run, label-free preprocessing before outer CV"
        ),
        "visual_pca_fit": "outer training runs only",
        "trace_outer_scaling_pca_fit": "outer training runs only",
        "trace_residual_score_scaling": (
            "B-residualized trace scores standardized on outer training rows; "
            "residual constants dropped"
        ),
        "trace_pca_components_max": TRACE_PCS,
        "nuisance_ridge_alpha": NUISANCE_ALPHA,
        "trace_ridge_alpha_grid": list(TRACE_ALPHA_GRID),
        "alpha_grid_near_null_endpoint": (
            "1e7 is a finite near-null ridge endpoint; for unit-scale outer Z "
            "with 4800 rows and p<=32 its effective df is about 0.016 or less"
        ),
        "alpha_grid_practical_null_endpoint": (
            "1e12 is a finite practical-null ridge endpoint with effectively zero "
            "degrees of freedom while preserving finite JSON and linear solves"
        ),
        "practical_null_interpretation": (
            "selection of 1e12 means effective-null/no detectable incremental "
            "trace signal, not positive evidence for a planning representation"
        ),
        "zero_predictor_reference": (
            "partial delta CV-R2=0 is the audited/shown no-increment reference; "
            "it is not an alpha candidate or an additional formal representation"
        ),
        "trace_ridge_alpha_selection": (
            "representation-specific nested LORO; every inner split refits visual "
            "PCA, control/trace scaling, trace PCA, trace and neural nuisance "
            "residualization, and residual-score scaling using inner-train only"
        ),
        "alpha_selection_score": (
            "equal-inner-run mean voxelwise held-out partial delta CV-R2"
        ),
        "alpha_selection_voxels": ALPHA_SELECTION_VOXELS,
        "alpha_selection_voxel_rule": (
            "lowest coordinate-byte SHA256 ranks over the full sorted unique atlas; "
            "outcome-independent and shared by representations/folds"
        ),
        "alpha_tie_break": (
            "largest alpha among scores equal within rtol=atol=1e-12"
        ),
        "lt2_outer_train_fallback": (
            "effective-df nearest 0.5*p, smoke-only; canonical group requires 11 runs"
        ),
        "primary_estimand": "equal-run mean voxelwise partial delta CV-R2",
        "formal_roi_inference": (
            "each representation versus zero; synchronized exact 64-sign-pattern "
            "max-|t| family over 2 representations x 110 ROIs"
        ),
        "formal_voxel_inference": (
            "each representation versus zero; synchronized exact 64-sign-pattern "
            "max-|t| family over 2 representations x all unique atlas voxels"
        ),
        "representation_comparison": (
            "concat-versus-mean plots are descriptive only; no concat-minus-mean "
            "or order-specific formal claim"
        ),
        "position_claim_caveat": (
            "concat greater than mean would not be a pure order contrast because "
            "position-specific content and model capacity also differ"
        ),
        "required_order_test": (
            "a mean-residual or order-shuffle comparator is required for an order claim"
        ),
        "random_state": int(random_state),
        "DQN_or_model_activation_used": False,
        "event_specific_fMRI_timestamps_claimed": False,
    }


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def safe_t(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    mean = np.mean(values, axis=0)
    sd = np.std(values, axis=0, ddof=1)
    denominator = sd / math.sqrt(values.shape[0])
    result = np.divide(
        mean,
        denominator,
        out=np.zeros_like(mean),
        where=denominator > EPS,
    )
    result[(denominator <= EPS) & (mean > 0)] = np.inf
    result[(denominator <= EPS) & (mean < 0)] = -np.inf
    return result


def exact_signflip_inference(matrix: np.ndarray) -> Dict[str, np.ndarray]:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != N_SUBJECTS:
        raise ValueError(f"Exact group inference requires 6 subjects, got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("Exact group matrix contains non-finite values")
    signs = np.asarray(
        [
            [1.0 if (pattern >> index) & 1 else -1.0 for index in range(N_SUBJECTS)]
            for pattern in range(1 << N_SUBJECTS)
        ],
        dtype=float,
    )
    observed_t = safe_t(values)
    null_t = np.vstack([safe_t(values * sign[:, None]) for sign in signs])
    maximum = np.max(np.abs(null_t), axis=1)
    raw = np.mean(
        (np.abs(null_t) > np.abs(observed_t)[None, :])
        | np.isclose(
            np.abs(null_t),
            np.abs(observed_t)[None, :],
            rtol=1e-12,
            atol=128.0 * np.finfo(float).eps,
        ),
        axis=0,
    )
    corrected = np.mean(
        (maximum[:, None] > np.abs(observed_t)[None, :])
        | np.isclose(
            maximum[:, None],
            np.abs(observed_t)[None, :],
            rtol=1e-12,
            atol=128.0 * np.finfo(float).eps,
        ),
        axis=0,
    )
    sem = stats.sem(values, axis=0)
    critical = float(stats.t.ppf(0.975, df=N_SUBJECTS - 1))
    return {
        "beta": np.mean(values, axis=0),
        "t_stat": observed_t,
        "p_exact_two_sided": raw,
        "p_maxT_fwer_family": corrected,
        "ci95_low": np.mean(values, axis=0) - critical * sem,
        "ci95_high": np.mean(values, axis=0) + critical * sem,
        "positive_fraction": np.mean(values > 0, axis=0),
        "n_subjects": np.full(values.shape[1], N_SUBJECTS, dtype=int),
        "n_sign_patterns": np.full(values.shape[1], 1 << N_SUBJECTS, dtype=int),
    }


def significance_stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def compact_roi_label(name: str, width: int = 38) -> str:
    """Readable compact Harvard-Oxford label retaining side and anatomy."""
    text = str(name)
    text = text.replace("ho_cortical_", "").replace("ho_subcortical_", "")
    if text.startswith("left_"):
        text = "L " + text[len("left_") :]
    elif text.startswith("right_"):
        text = "R " + text[len("right_") :]
    replacements = {
        "cingulate_gyrus": "cingulate",
        "lateral_occipital_cortex": "lat_occipital",
        "inferior_temporal_gyrus": "inf_temporal",
        "middle_temporal_gyrus": "mid_temporal",
        "superior_temporal_gyrus": "sup_temporal",
        "temporal_fusiform_cortex": "temporal_fusiform",
        "juxtapositional_lobule_cortex_formerly_supplementary_motor_cortex": "SMA",
        "heschl_s_gyrus_includes_h1_and_h2": "Heschl_H1-H2",
        "anterior_division": "ant",
        "posterior_division": "post",
        "inferior_division": "inf",
        "superior_division": "sup",
        "temporooccipital_part": "temp-occ",
        "pars_opercularis": "pars_operc",
        "pars_triangularis": "pars_triang",
        "cortex": "ctx",
        "gyrus": "gyr",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = text.replace("_", " ")
    text = " ".join(text.split())
    if len(text) > int(width):
        text = text[: max(1, int(width) - 1)].rstrip() + "…"
    return text


def concise_numeric_axis(axis: plt.Axes, include_y: bool = False) -> None:
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.xaxis.set_major_formatter(formatter)
    if include_y:
        y_formatter = ScalarFormatter(useMathText=True)
        y_formatter.set_powerlimits((-2, 2))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.yaxis.set_major_formatter(y_formatter)


def standardize_train(
    train: np.ndarray, tests: Sequence[np.ndarray]
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    source = np.asarray(train, dtype=np.float64)
    mean = np.mean(source, axis=0)
    scale = np.std(source, axis=0)
    keep = np.isfinite(scale) & (scale > 1e-8)
    if not np.any(keep):
        raise ValueError("No varying columns remain after train-only scaling")
    scaled_train = (source[:, keep] - mean[keep]) / scale[keep]
    scaled_tests = [
        (np.asarray(test, dtype=np.float64)[:, keep] - mean[keep]) / scale[keep]
        for test in tests
    ]
    return scaled_train, scaled_tests, keep


def ridge_left_operator(design: np.ndarray, alpha: float) -> np.ndarray:
    x = np.asarray(design, dtype=np.float64)
    penalty = np.eye(x.shape[1], dtype=np.float64) * float(alpha)
    if x.shape[1] and np.allclose(x[:, 0], 1.0):
        penalty[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + penalty, x.T)


def ridge_effective_df(design: np.ndarray, alpha: float) -> float:
    singular = np.linalg.svd(np.asarray(design, dtype=np.float64), compute_uv=False)
    squared = singular**2
    return float(np.sum(squared / (squared + float(alpha))))


def column_correlation(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    y = np.asarray(observed, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    yc = y - np.mean(y, axis=0, keepdims=True)
    pc = p - np.mean(p, axis=0, keepdims=True)
    numerator = np.sum(yc * pc, axis=0)
    denominator = np.sqrt(np.sum(yc**2, axis=0) * np.sum(pc**2, axis=0))
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > EPS,
    )


def _control_matrix(blocks: Mapping[str, np.ndarray]) -> np.ndarray:
    # Pre-search P/M/Q0 are explicit scalar/policy controls, not hidden model
    # activations. Including them asks whether the completed trace explains
    # neural activity beyond the state already available before search.
    # decision_event adds the generic decision-anchor density HRF nuisance so
    # trace gain cannot be attributed to shared impulse timing or mass.
    names = (
        "current_action",
        "previous_action",
        "task_history",
        "drift",
        "decision_event",
        "model_prior_policy",
        "model_margin",
        "initial_q_state",
    )
    arrays = [np.asarray(blocks[name], dtype=np.float32) for name in names]
    if {array.shape[0] for array in arrays} != {N_TR}:
        raise ValueError("Canonical control blocks are not all 480 TR")
    return np.column_stack(arrays).astype(np.float32)


def _prepare_single_fold(
    visuals: Sequence[np.ndarray],
    controls: Sequence[np.ndarray],
    traces: Sequence[Mapping[str, np.ndarray]],
    train_indices: Sequence[int],
    test_index: int,
    random_state: int,
) -> Dict[str, Any]:
    """Fit every preprocessing step on ``train_indices`` and apply to one test run."""
    train_indices = [int(index) for index in train_indices]
    if not train_indices or int(test_index) in train_indices:
        raise ValueError("Fold requires non-empty disjoint train and test runs")
    visual_train = np.vstack([visuals[index] for index in train_indices])
    visual_test = np.asarray(visuals[int(test_index)])
    n_visual = min(VISUAL_PCS, visual_train.shape[1], visual_train.shape[0] - 1)
    if n_visual < 1:
        raise ValueError("Fold has insufficient visual rows/components")
    visual_pca = PCA(
        n_components=n_visual,
        svd_solver="randomized",
        random_state=int(random_state),
    )
    visual_train_pc = visual_pca.fit_transform(visual_train)
    visual_test_pc = visual_pca.transform(visual_test)
    control_train_raw = np.vstack([controls[index] for index in train_indices])
    control_test_raw = np.asarray(controls[int(test_index)])
    control_train, [control_test], _ = standardize_train(
        control_train_raw, [control_test_raw]
    )
    visual_train_scaled, [visual_test_scaled], _ = standardize_train(
        visual_train_pc, [visual_test_pc]
    )
    b_train = np.column_stack(
        (np.ones(len(control_train)), control_train, visual_train_scaled)
    )
    b_test = np.column_stack(
        (np.ones(len(control_test)), control_test, visual_test_scaled)
    )
    b_operator = ridge_left_operator(b_train, NUISANCE_ALPHA)
    representation_rows: Dict[str, Any] = {}
    for representation_index, representation in enumerate(REPRESENTATIONS):
        z_train_raw = np.vstack(
            [np.asarray(traces[index][representation]) for index in train_indices]
        )
        z_test_raw = np.asarray(traces[int(test_index)][representation])
        z_train_scaled, [z_test_scaled], keep = standardize_train(
            z_train_raw, [z_test_raw]
        )
        limit = min(TRACE_PCS, z_train_scaled.shape[1], z_train_scaled.shape[0] - 1)
        if limit < 1:
            raise ValueError(f"No trace components remain for {representation}")
        if z_train_scaled.shape[1] > limit:
            trace_pca = PCA(
                n_components=limit,
                svd_solver="randomized",
                random_state=int(random_state + 100 + representation_index),
            )
            z_train_reduced = trace_pca.fit_transform(z_train_scaled)
            z_test_reduced = trace_pca.transform(z_test_scaled)
            explained = float(np.sum(trace_pca.explained_variance_ratio_))
        else:
            z_train_reduced = z_train_scaled
            z_test_reduced = z_test_scaled
            explained = 1.0
        nuisance_coefficients = b_operator @ z_train_reduced
        z_train_residual_raw = z_train_reduced - b_train @ nuisance_coefficients
        z_test_residual_raw = z_test_reduced - b_test @ nuisance_coefficients
        # Put every residual-score component on an outer/inner-train-only unit
        # scale before applying the common ridge grid. Residual constants are
        # dropped without consulting the held-out run.
        z_train_residual, [z_test_residual], residual_keep = standardize_train(
            z_train_residual_raw, [z_test_residual_raw]
        )
        run_slices = []
        cursor = 0
        for index in train_indices:
            rows = int(len(traces[index][representation]))
            run_slices.append((cursor, cursor + rows))
            cursor += rows
        if cursor != len(z_train_residual):
            raise AssertionError("Trace training slices do not cover the fold")
        representation_rows[representation] = {
            "z_train_residual": z_train_residual,
            "z_test_residual": z_test_residual,
            "train_run_slices": run_slices,
            "n_raw_components": int(z_train_raw.shape[1]),
            "n_varying_components": int(np.sum(keep)),
            "n_pre_residual_components": int(z_train_residual_raw.shape[1]),
            "n_model_components": int(z_train_residual.shape[1]),
            "n_residual_constant_components_dropped": int(
                len(residual_keep) - np.sum(residual_keep)
            ),
            "pca_explained_variance": explained,
        }
    return {
        "test_index": int(test_index),
        "train_indices": train_indices,
        "b_train": b_train,
        "b_test": b_test,
        "representations": representation_rows,
    }


def prepare_fold_operators(
    visuals: Sequence[np.ndarray],
    controls: Sequence[np.ndarray],
    traces: Sequence[Mapping[str, np.ndarray]],
    random_state: int,
) -> list[Dict[str, Any]]:
    if not (len(visuals) == len(controls) == len(traces)):
        raise ValueError("Visual, control, and trace run grids differ")
    folds: list[Dict[str, Any]] = []
    for test_index in range(len(traces)):
        train_indices = [index for index in range(len(traces)) if index != test_index]
        folds.append(
            _prepare_single_fold(
                visuals,
                controls,
                traces,
                train_indices,
                test_index,
                int(random_state + 10_000 * test_index),
            )
        )
    return folds


def _prepare_neural_fold(
    neural_runs: Sequence[np.ndarray],
    train_indices: Sequence[int],
    test_index: int,
    b_train: np.ndarray,
    b_test: np.ndarray,
    voxel_indices: np.ndarray | None = None,
    return_train_residuals: bool = False,
) -> Dict[str, Any]:
    """Fit target scaling and nuisance regression on training runs only."""
    train_indices = [int(index) for index in train_indices]
    selected = None if voxel_indices is None else np.asarray(voxel_indices, dtype=int)
    if selected is not None and (selected.ndim != 1 or len(np.unique(selected)) != len(selected)):
        raise ValueError("Neural selection indices must be a unique vector")

    def take(index: int) -> np.ndarray:
        values = np.asarray(neural_runs[int(index)], dtype=np.float64)
        return values if selected is None else values[:, selected]

    n_targets = int(take(train_indices[0]).shape[1])
    n_train = sum(int(len(neural_runs[index])) for index in train_indices)
    sum_y = np.zeros(n_targets, dtype=np.float64)
    sum_y2 = np.zeros(n_targets, dtype=np.float64)
    for index in train_indices:
        values = take(index)
        sum_y += np.sum(values, axis=0, dtype=np.float64)
        sum_y2 += np.sum(values**2, axis=0, dtype=np.float64)
    mean_y = sum_y / float(n_train)
    variance_y = np.maximum(sum_y2 / float(n_train) - mean_y**2, 0.0)
    scale_y = np.sqrt(variance_y)
    valid_y = np.isfinite(scale_y) & (scale_y > 1e-8)
    scale_y[~valid_y] = 1.0
    b_train = np.asarray(b_train, dtype=np.float64)
    b_test = np.asarray(b_test, dtype=np.float64)
    b_cross_y = np.zeros((b_train.shape[1], n_targets), dtype=np.float64)
    cursor = 0
    for index in train_indices:
        values = take(index)
        rows = int(len(values))
        y_scaled = (values - mean_y) / scale_y
        b_cross_y += b_train[cursor : cursor + rows].T @ y_scaled
        cursor += rows
    if cursor != len(b_train):
        raise ValueError("Neural and nuisance training rows differ")
    penalty = np.diag([0.0] + [NUISANCE_ALPHA] * (b_train.shape[1] - 1))
    nuisance_beta = np.linalg.solve(b_train.T @ b_train + penalty, b_cross_y)
    y_test_scaled = (take(int(test_index)) - mean_y) / scale_y
    y_test_residual = y_test_scaled - b_test @ nuisance_beta
    result: Dict[str, Any] = {
        "mean_y": mean_y,
        "scale_y": scale_y,
        "valid_y": valid_y,
        "nuisance_beta": nuisance_beta,
        "y_test_residual": y_test_residual,
    }
    if return_train_residuals:
        train_residuals = []
        cursor = 0
        for index in train_indices:
            values = take(index)
            rows = int(len(values))
            y_scaled = (values - mean_y) / scale_y
            train_residuals.append(
                y_scaled - b_train[cursor : cursor + rows] @ nuisance_beta
            )
            cursor += rows
        result["y_train_residual_runs"] = train_residuals
    return result


def _heldout_partial_delta_r2(observed: np.ndarray, predicted: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    denominator = np.sum(observed**2, axis=0)
    error = np.sum((observed - predicted) ** 2, axis=0)
    valid = np.isfinite(denominator) & np.isfinite(error) & (denominator > EPS)
    if not np.any(valid):
        raise ValueError("Alpha-selection validation run has no varying targets")
    delta = (denominator[valid] - error[valid]) / denominator[valid]
    if not np.all(np.isfinite(delta)):
        raise ValueError("Alpha-selection score is non-finite")
    return float(np.mean(delta))


def _choose_largest_alpha_on_tie(
    scores: Mapping[float, float]
) -> tuple[float, float]:
    if not scores or not all(np.isfinite(value) for value in scores.values()):
        raise ValueError("Alpha scores must be finite and non-empty")
    best_score = float(max(scores.values()))
    tied = [
        float(alpha)
        for alpha, score in scores.items()
        if np.isclose(score, best_score, rtol=1e-12, atol=1e-12)
    ]
    return float(max(tied)), best_score


def select_trace_alphas_nested(
    outer_fold: Mapping[str, Any],
    visuals: Sequence[np.ndarray],
    controls: Sequence[np.ndarray],
    traces: Sequence[Mapping[str, np.ndarray]],
    neural_runs: Sequence[np.ndarray],
    selection_indices: np.ndarray,
    selection_coordinate_sha256: str,
    run_labels: Sequence[str],
    trace_tokens: Sequence[str],
    random_state: int,
) -> Dict[str, Dict[str, Any]]:
    """Select one alpha per representation without consulting the outer test run.

    Each inner LORO split starts from raw run arrays and refits visual PCA,
    control/trace scaling, trace PCA, B-to-trace residualization, residual-score
    scaling, neural scaling, and B-to-neural nuisance regression. The six ridge
    candidates then reuse that one fully inner-fitted split.
    """
    alpha_grid = tuple(float(value) for value in TRACE_ALPHA_GRID)
    if (
        any(value <= 0 for value in alpha_grid)
        or alpha_grid != tuple(sorted(set(alpha_grid)))
    ):
        raise ValueError("TRACE_ALPHA_GRID must be unique, positive, and increasing")
    outer_test_index = int(outer_fold["test_index"])
    outer_train_indices = [int(index) for index in outer_fold["train_indices"]]
    if not (
        len(visuals)
        == len(controls)
        == len(traces)
        == len(neural_runs)
        == len(run_labels)
        == len(trace_tokens)
    ):
        raise ValueError("Nested selector run grids differ")
    scores_by_rep = {
        representation: {alpha: [] for alpha in alpha_grid}
        for representation in REPRESENTATIONS
    }
    validation_labels: list[str] = []

    if len(outer_train_indices) >= 2:
        for inner_order, validation_index in enumerate(outer_train_indices):
            inner_train_indices = [
                index for index in outer_train_indices if index != validation_index
            ]
            inner_fold = _prepare_single_fold(
                visuals,
                controls,
                traces,
                inner_train_indices,
                validation_index,
                int(
                    random_state
                    + 1_000_000
                    + 100_000 * outer_test_index
                    + 1_000 * validation_index
                    + inner_order
                ),
            )
            inner_target = _prepare_neural_fold(
                neural_runs,
                inner_train_indices,
                validation_index,
                np.asarray(inner_fold["b_train"]),
                np.asarray(inner_fold["b_test"]),
                voxel_indices=np.asarray(selection_indices, dtype=int),
                return_train_residuals=True,
            )
            y_train = np.vstack(inner_target["y_train_residual_runs"])
            y_validation = np.asarray(inner_target["y_test_residual"])
            validation_labels.append(str(run_labels[validation_index]))
            for representation in REPRESENTATIONS:
                rep = inner_fold["representations"][representation]
                z_train = np.asarray(rep["z_train_residual"], dtype=np.float64)
                z_validation = np.asarray(rep["z_test_residual"], dtype=np.float64)
                cross = z_train.T @ y_train
                gram = z_train.T @ z_train
                identity = np.eye(z_train.shape[1], dtype=np.float64)
                for alpha in alpha_grid:
                    beta = np.linalg.solve(gram + alpha * identity, cross)
                    score = _heldout_partial_delta_r2(
                        y_validation, z_validation @ beta
                    )
                    scores_by_rep[representation][alpha].append(score)

    output: Dict[str, Dict[str, Any]] = {}
    for representation in REPRESENTATIONS:
        outer_z = np.asarray(
            outer_fold["representations"][representation]["z_train_residual"],
            dtype=np.float64,
        )
        if len(outer_train_indices) >= 2:
            equal_run_scores = {
                alpha: float(np.mean(scores_by_rep[representation][alpha]))
                for alpha in alpha_grid
            }
            selected_alpha, selection_score = _choose_largest_alpha_on_tie(
                equal_run_scores
            )
            mode = "nested_loro_full_inner_refit"
            metric = "equal-inner-run mean voxelwise held-out partial delta CV-R2"
            n_inner_folds = len(outer_train_indices)
        else:
            target_edf = 0.5 * float(outer_z.shape[1])
            equal_run_scores = {
                alpha: -abs(ridge_effective_df(outer_z, alpha) - target_edf)
                for alpha in alpha_grid
            }
            selected_alpha, selection_score = _choose_largest_alpha_on_tie(
                equal_run_scores
            )
            mode = "effective_df_fallback_lt2_outer_train_runs"
            metric = "negative absolute distance from effective df to 0.5*p"
            n_inner_folds = 0
        effective_df = ridge_effective_df(outer_z, selected_alpha)
        practical_null = bool(np.isclose(selected_alpha, 1e12, rtol=0.0, atol=0.0))
        alpha_interpretation = (
            "effective-null/no detectable incremental trace signal"
            if practical_null
            else "finite non-null ridge fit; evidence still requires held-out and group tests"
        )
        token_payload = {
            "algorithm": "nested-loro-full-inner-refit-v1",
            "outer_heldout_run": str(run_labels[outer_test_index]),
            "outer_training_trace_tokens": [
                str(trace_tokens[index]) for index in outer_train_indices
            ],
            "representation": representation,
            "alpha_grid": list(alpha_grid),
            "equal_run_scores": [equal_run_scores[alpha] for alpha in alpha_grid],
            "selected_alpha": selected_alpha,
            "selected_alpha_is_practical_null": practical_null,
            "selection_coordinate_sha256": selection_coordinate_sha256,
            "inner_validation_runs": validation_labels,
            "tie_break": "largest alpha among scores equal within rtol=atol=1e-12",
        }
        output[representation] = {
            "selected_alpha": selected_alpha,
            "selection_score": selection_score,
            "selection_mode": mode,
            "selection_metric": metric,
            "n_inner_folds": int(n_inner_folds),
            "n_selection_voxels": int(len(selection_indices)),
            "effective_df": effective_df,
            "selected_alpha_is_practical_null": practical_null,
            "selected_alpha_interpretation": alpha_interpretation,
            "scores": equal_run_scores,
            "selection_token": json_sha256(token_payload),
            "inner_validation_runs": validation_labels,
            "tie_break": token_payload["tie_break"],
            "outer_test_used": False,
        }
    return output


def load_atlas_patterns(
    run_specs: Sequence[COMMON.BrainRunSpec], roi_names: Sequence[str]
) -> tuple[np.ndarray, list[np.ndarray], Dict[str, np.ndarray]]:
    coordinate_blocks: Dict[str, np.ndarray] = {}
    all_coordinates: list[tuple[int, int, int]] = []
    for roi in roi_names:
        _, coords = COMMON.load_roi(run_specs[0], roi)
        coords = np.asarray(coords, dtype=np.int32)
        coordinate_blocks[roi] = coords
        all_coordinates.extend(map(tuple, coords.tolist()))
    unique_coordinates = np.asarray(sorted(set(all_coordinates)), dtype=np.int32)
    lookup = {tuple(row): index for index, row in enumerate(unique_coordinates.tolist())}
    roi_indices = {
        roi: np.asarray([lookup[tuple(row)] for row in coords.tolist()], dtype=np.int32)
        for roi, coords in coordinate_blocks.items()
    }
    patterns_by_run: list[np.ndarray] = []
    for run in run_specs:
        atlas = np.full((int(run.n_tr), len(unique_coordinates)), np.nan, dtype=np.float32)
        filled = np.zeros(len(unique_coordinates), dtype=bool)
        for roi in roi_names:
            patterns, coords = COMMON.load_roi(run, roi)
            if not np.array_equal(np.asarray(coords, dtype=np.int32), coordinate_blocks[roi]):
                raise ValueError(f"ROI coordinates changed across runs for {roi}")
            indices = roi_indices[roi]
            overlap = filled[indices]
            if np.any(overlap):
                previous = atlas[:, indices[overlap]]
                current = np.asarray(patterns[:, overlap], dtype=np.float32)
                if not np.allclose(previous, current, atol=2e-5, rtol=2e-5):
                    raise ValueError(f"Overlapping atlas voxels disagree for {roi}")
            atlas[:, indices] = np.asarray(patterns, dtype=np.float32)
            filled[indices] = True
        if not filled.all() or not np.all(np.isfinite(atlas)):
            raise ValueError(f"Atlas pattern assembly failed for {run.run_label}")
        patterns_by_run.append(atlas)
    return unique_coordinates, patterns_by_run, roi_indices


def fit_subject_encoding(
    run_specs: Sequence[COMMON.BrainRunSpec],
    traces: Sequence[Mapping[str, np.ndarray]],
    roi_names: Sequence[str],
    random_state: int,
    trace_tokens: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray], list[Dict[str, Any]]]:
    coordinates, neural_runs, roi_indices = load_atlas_patterns(run_specs, roi_names)
    visuals = [COMMON.load_visual(run) for run in run_specs]
    controls = [_control_matrix(COMMON.load_process_blocks(run)) for run in run_specs]
    if len(trace_tokens) != len(run_specs):
        raise ValueError("Trace artifact token grid differs from run grid")
    folds = prepare_fold_operators(visuals, controls, traces, random_state)
    selection_indices = coordinate_hash_selection(
        coordinates, ALPHA_SELECTION_VOXELS
    )
    if len(selection_indices) == 0:
        raise ValueError("No atlas voxels available for alpha selection")
    selection_coordinate_sha256 = array_sha256(coordinates[selection_indices])
    run_labels = [str(run.run_label) for run in run_specs]
    fold_delta = {
        representation: np.zeros((len(run_specs), len(coordinates)), dtype=np.float32)
        for representation in REPRESENTATIONS
    }
    fold_r = {
        representation: np.zeros((len(run_specs), len(coordinates)), dtype=np.float32)
        for representation in REPRESENTATIONS
    }
    fold_audit: list[Dict[str, Any]] = []
    for fold in folds:
        test_index = int(fold["test_index"])
        train_indices = list(fold["train_indices"])
        b_train = np.asarray(fold["b_train"], dtype=np.float64)
        outer_target = _prepare_neural_fold(
            neural_runs,
            train_indices,
            test_index,
            b_train,
            np.asarray(fold["b_test"], dtype=np.float64),
        )
        mean_y = np.asarray(outer_target["mean_y"])
        scale_y = np.asarray(outer_target["scale_y"])
        valid_y = np.asarray(outer_target["valid_y"], dtype=bool)
        nuisance_beta = np.asarray(outer_target["nuisance_beta"])
        y_test_residual = np.asarray(outer_target["y_test_residual"])
        alpha_selection = select_trace_alphas_nested(
            fold,
            visuals,
            controls,
            traces,
            neural_runs,
            selection_indices,
            selection_coordinate_sha256,
            run_labels,
            trace_tokens,
            random_state,
        )
        for representation in REPRESENTATIONS:
            rep = fold["representations"][representation]
            selection = alpha_selection[representation]
            selected_alpha = float(selection["selected_alpha"])
            z_train = np.asarray(rep["z_train_residual"], dtype=np.float64)
            cross = np.zeros((z_train.shape[1], len(coordinates)), dtype=np.float64)
            cursor = 0
            for index in train_indices:
                rows = neural_runs[index].shape[0]
                y_scaled = (neural_runs[index] - mean_y) / scale_y
                b_rows = b_train[cursor : cursor + rows]
                y_residual = y_scaled - b_rows @ nuisance_beta
                cross += z_train[cursor : cursor + rows].T @ y_residual
                cursor += rows
            trace_beta = np.linalg.solve(
                z_train.T @ z_train
                + selected_alpha * np.eye(z_train.shape[1]),
                cross,
            )
            prediction = np.asarray(rep["z_test_residual"]) @ trace_beta
            denominator = np.sum(y_test_residual**2, axis=0)
            error = np.sum((y_test_residual - prediction) ** 2, axis=0)
            delta = np.divide(
                denominator - error,
                denominator,
                out=np.zeros_like(denominator),
                where=denominator > EPS,
            )
            correlation = column_correlation(y_test_residual, prediction)
            delta[~valid_y] = 0.0
            correlation[~valid_y] = 0.0
            fold_delta[representation][test_index] = delta.astype(np.float32)
            fold_r[representation][test_index] = correlation.astype(np.float32)
            fold_audit.append(
                {
                    "heldout_run": run_specs[test_index].run_label,
                    "representation": representation,
                    "n_train_runs": len(train_indices),
                    "n_test_tr": int(len(y_test_residual)),
                    "n_voxels": int(len(coordinates)),
                    "n_raw_components": rep["n_raw_components"],
                    "n_varying_components": rep["n_varying_components"],
                    "n_pre_residual_components": rep["n_pre_residual_components"],
                    "n_model_components": rep["n_model_components"],
                    "n_residual_constant_components_dropped": rep[
                        "n_residual_constant_components_dropped"
                    ],
                    "pca_explained_variance": rep["pca_explained_variance"],
                    "selected_alpha": selected_alpha,
                    "alpha_grid_json": json.dumps(list(TRACE_ALPHA_GRID)),
                    "alpha_equal_inner_run_scores_json": json.dumps(
                        {
                            f"{alpha:g}": float(selection["scores"][alpha])
                            for alpha in TRACE_ALPHA_GRID
                        },
                        sort_keys=True,
                        allow_nan=False,
                    ),
                    "alpha_selection_score": selection["selection_score"],
                    "alpha_selection_mode": selection["selection_mode"],
                    "alpha_selection_metric": selection["selection_metric"],
                    "alpha_selection_inner_folds": selection["n_inner_folds"],
                    "alpha_selection_inner_validation_runs_json": json.dumps(
                        selection["inner_validation_runs"]
                    ),
                    "alpha_selection_voxels": selection["n_selection_voxels"],
                    "alpha_selection_coordinate_sha256": selection_coordinate_sha256,
                    "alpha_selection_tie_break": selection["tie_break"],
                    "alpha_selection_token": selection["selection_token"],
                    "alpha_selection_outer_test_used": selection["outer_test_used"],
                    "alpha_selection_full_inner_refit": True,
                    "trace_effective_df": selection["effective_df"],
                    "selected_alpha_is_practical_null": selection[
                        "selected_alpha_is_practical_null"
                    ],
                    "selected_alpha_interpretation": selection[
                        "selected_alpha_interpretation"
                    ],
                    "zero_predictor_delta_cv_r2_reference": 0.0,
                    "random_state": int(random_state),
                    "finite_fraction": float(np.mean(np.isfinite(delta))),
                }
            )
    run_rows: list[Dict[str, Any]] = []
    subject_rows: list[Dict[str, Any]] = []
    for representation in REPRESENTATIONS:
        for roi in roi_names:
            indices = roi_indices[roi]
            run_values = np.mean(fold_delta[representation][:, indices], axis=1)
            run_correlations = np.mean(fold_r[representation][:, indices], axis=1)
            for index, run in enumerate(run_specs):
                run_rows.append(
                    {
                        "subject": int(run.subject),
                        "game": int(run.game),
                        "run_label": run.run_label,
                        "roi": roi,
                        "representation": representation,
                        "delta_cv_r2": float(run_values[index]),
                        "residual_prediction_r": float(run_correlations[index]),
                        "n_voxels": int(len(indices)),
                    }
                )
            subject_rows.append(
                {
                    "subject": int(run_specs[0].subject),
                    "game": int(run_specs[0].game),
                    "roi": roi,
                    "representation": representation,
                    "beta": float(np.mean(run_values)),
                    "residual_prediction_r": float(np.mean(run_correlations)),
                    "n_runs": int(len(run_specs)),
                    "n_voxels": int(len(indices)),
                }
            )
    voxel_payload = {"coords": coordinates.astype(np.int32)}
    for representation in REPRESENTATIONS:
        short = "concat29" if representation.endswith("concat29") else "mean29"
        voxel_payload[f"{short}_delta_cv_r2"] = np.mean(
            fold_delta[representation], axis=0
        ).astype(np.float32)
        voxel_payload[f"{short}_residual_r"] = np.mean(
            fold_r[representation], axis=0
        ).astype(np.float32)
    return (
        pd.DataFrame(run_rows),
        pd.DataFrame(subject_rows),
        voxel_payload,
        fold_audit,
    )


def plot_subject_figures(
    run_frame: pd.DataFrame,
    subject_frame: pd.DataFrame,
    fold_audit: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {"planning_concat29": "#2b6cb0", "planning_mean29": "#dd6b20"}
    n_practical_null = sum(
        bool(row["selected_alpha_is_practical_null"]) for row in fold_audit
    )
    null_note = (
        f"practical-null α=1e12: {n_practical_null}/{len(fold_audit)} folds "
        "(effective-null/no detectable incremental trace signal)"
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    for position, representation in enumerate(REPRESENTATIONS):
        values = subject_frame.loc[
            subject_frame.representation.eq(representation), "beta"
        ].to_numpy()
        parts = axes[0].violinplot(values, positions=[position], widths=0.72, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(colors[representation]); body.set_alpha(0.28)
        jitter = np.linspace(-0.18, 0.18, len(values))
        axes[0].scatter(position + jitter, values, s=18, alpha=0.65, color=colors[representation])
    axes[0].axhline(0, color="black", lw=0.8, ls="--")
    axes[0].set_xticks([0, 1], [REPRESENTATION_LABELS[x] for x in REPRESENTATIONS])
    axes[0].tick_params(axis="x", labelsize=9)
    axes[0].set_ylabel(r"held-out partial $\Delta CV\!R^2$")
    axes[0].set_title("ROI distribution (descriptive)")
    pivot = subject_frame.pivot(index="roi", columns="representation", values="beta")
    axes[1].scatter(
        pivot["planning_mean29"], pivot["planning_concat29"],
        s=24, alpha=0.7, color="#4a5568",
    )
    bounds = np.asarray([pivot.min().min(), pivot.max().max()])
    axes[1].plot(bounds, bounds, color="black", lw=0.8, ls="--")
    axes[1].set_xlabel("Mean-29 (pooled) ΔCVR²")
    axes[1].set_ylabel("Concat-29 (position-preserving) ΔCVR²")
    axes[1].set_title(
        "Position-preserving concat vs pooled mean\n(descriptive; not a contrast)",
        fontsize=10,
    )
    concise_numeric_axis(axes[1], include_y=True)
    fig.suptitle(null_note, fontsize=9)
    fig.savefig(output_dir / "figure1_representation_comparison.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(
        1, 2, figsize=(16, 7.5), sharex=True, constrained_layout=True
    )
    for axis, representation in zip(axes, REPRESENTATIONS):
        subset = subject_frame[subject_frame.representation.eq(representation)].copy()
        subset = subset.reindex(subset.beta.abs().sort_values(ascending=False).index).head(15)
        y = np.arange(len(subset))
        axis.hlines(y, 0, subset.beta, color=colors[representation], alpha=0.75)
        axis.scatter(subset.beta, y, color=colors[representation], s=45, zorder=3)
        for yi, row in enumerate(subset.itertuples()):
            runs = run_frame[
                run_frame.roi.eq(row.roi) & run_frame.representation.eq(representation)
            ].delta_cv_r2.to_numpy()
            axis.scatter(runs, np.full(len(runs), yi), s=9, color="black", alpha=0.25)
        axis.axvline(0, color="black", lw=0.8, ls="--")
        axis.set_yticks(y, [compact_roi_label(name) for name in subset.roi])
        axis.invert_yaxis()
        axis.set_xlabel("run-equal ΔCVR²")
        axis.set_title(REPRESENTATION_LABELS[representation], fontsize=10, pad=8)
        concise_numeric_axis(axis)
    fig.suptitle(
        "Per-representation held-out encoding (descriptive ranking)\n" + null_note,
        fontsize=10,
    )
    fig.savefig(output_dir / "figure2_ranked_roi_lollipops.png", dpi=220)
    plt.close(fig)


def _subject_output(output_root: Path, subject: int, game: int) -> Path:
    return Path(output_root) / f"sub{subject:03d}_game{game}"


def run_subject(args: argparse.Namespace) -> None:
    started_sources = current_source_hashes()
    run_specs, cache_audit, trace_audit = TRACE.prepare_trace_runs_from_args(args)
    roi_names = COMMON.common_roi_names(run_specs)
    if not roi_names:
        raise ValueError("No common ROIs")
    traces = []
    trace_metadata = []
    for run in run_specs:
        blocks, metadata = TRACE.load_run_trace(run, Path(args.planning_trace_root))
        traces.append(blocks)
        trace_metadata.append(metadata)
    trace_tokens = [trace_artifact_sha256(row) for row in trace_metadata]
    run_frame, subject_frame, voxel_payload, fold_audit = fit_subject_encoding(
        run_specs, traces, roi_names, int(args.random_state), trace_tokens
    )
    output = _subject_output(args.output_root, args.subject, args.game)
    table_dir = output / "tables"
    figure_dir = output / "figures"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {output}")
    table_dir.mkdir(parents=True, exist_ok=True)
    run_frame.to_csv(table_dir / "run_roi_effects.csv.gz", index=False, compression="gzip")
    subject_frame.to_csv(table_dir / "subject_roi_effects.csv", index=False)
    pd.DataFrame(fold_audit).to_csv(table_dir / "fold_audit.csv", index=False)
    atomic_npz(table_dir / "subject_voxel_effects.npz", **voxel_payload)
    plot_subject_figures(run_frame, subject_frame, fold_audit, figure_dir)
    if current_source_hashes() != started_sources:
        raise RuntimeError("Analysis or shared source changed during execution")
    contract = scientific_contract(int(args.random_state))
    selection_hashes = {
        str(row["alpha_selection_coordinate_sha256"]) for row in fold_audit
    }
    if len(selection_hashes) != 1:
        raise AssertionError("Alpha selection coordinates changed across folds")
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        **started_sources,
        "settings_sha256": json_sha256(contract),
        "subject": int(args.subject),
        "game": int(args.game),
        "random_state": int(args.random_state),
        "n_runs": len(run_specs),
        "n_rois": len(roi_names),
        "n_unique_atlas_voxels": int(len(voxel_payload["coords"])),
        "run_labels": [run.run_label for run in run_specs],
        "roi_names": list(roi_names),
        "representations": list(REPRESENTATIONS),
        "scientific_contract": contract,
        "alpha_selection_coordinate_sha256": next(iter(selection_hashes)),
        "alpha_selection_voxel_algorithm": contract["alpha_selection_voxel_rule"],
        "base_cache": {
            "cache_root": str(cache_audit.get("cache_root", "")),
            "cache_version": cache_audit.get("cache_version"),
            "process_feature_version": cache_audit.get("process_feature_version"),
            "neural_feature_schema_version": cache_audit.get("neural_feature_schema_version"),
            "feature_source_token": cache_audit.get("feature_source_token"),
        },
        "planning_trace_audit": trace_audit,
        "planning_trace_run_tokens": trace_tokens,
        "tables": {
            path.name: sha256_file(path) for path in sorted(table_dir.iterdir()) if path.is_file()
        },
        "figures": {
            path.name: sha256_file(path)
            for path in sorted(figure_dir.iterdir()) if path.is_file()
        },
    }
    atomic_json(output / "analysis_manifest.json", manifest)
    print(f"08 planning-trace subject complete: {output}", flush=True)


def _load_subjects(output_root: Path, game: int) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    frames = []
    voxels = []
    manifests = []
    reference_contract = None
    reference_selection_hash = None
    reference_coords = None
    expected_sources = current_source_hashes()
    for subject in range(1, N_SUBJECTS + 1):
        root = _subject_output(output_root, subject, game)
        manifest = json.loads((root / "analysis_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("analysis_version") != ANALYSIS_VERSION or manifest.get("subject") != subject or manifest.get("game") != game:
            raise ValueError(f"Subject manifest mismatch: {root}")
        for name, expected in expected_sources.items():
            if manifest.get(name) != expected:
                raise ValueError(f"Subject source provenance mismatch ({name}): {root}")
        if int(manifest.get("n_runs", -1)) != N_RUNS or int(manifest.get("n_rois", -1)) != N_ROIS:
            raise ValueError(f"Non-canonical subject output: {root}")
        contract = manifest.get("scientific_contract")
        random_state = int(manifest.get("random_state", -1))
        if contract != scientific_contract(random_state):
            raise ValueError(f"Scientific contract is not canonical: {root}")
        if manifest.get("settings_sha256") != json_sha256(contract):
            raise ValueError(f"Settings hash mismatch: {root}")
        if reference_contract is None:
            reference_contract = contract
        elif contract != reference_contract:
            raise ValueError("Scientific contracts differ across subjects")
        selection_hash = str(manifest.get("alpha_selection_coordinate_sha256", ""))
        if len(selection_hash) != 64:
            raise ValueError(f"Alpha-selection coordinate hash missing: {root}")
        if reference_selection_hash is None:
            reference_selection_hash = selection_hash
        elif selection_hash != reference_selection_hash:
            raise ValueError("Alpha-selection coordinates differ across subjects")
        frame = pd.read_csv(root / "tables" / "subject_roi_effects.csv")
        expected_grid = {
            (representation, roi)
            for representation in REPRESENTATIONS
            for roi in manifest.get("roi_names", [])
        }
        observed_grid = set(zip(frame.representation.astype(str), frame.roi.astype(str)))
        if (
            len(frame) != len(REPRESENTATIONS) * N_ROIS
            or observed_grid != expected_grid
            or frame.duplicated(["representation", "roi"]).any()
            or set(frame.subject.astype(int)) != {subject}
            or set(frame.game.astype(int)) != {game}
            or set(frame.n_runs.astype(int)) != {N_RUNS}
            or not np.all(np.isfinite(frame[["beta", "residual_prediction_r"]]))
        ):
            raise ValueError(f"Subject ROI grid invalid: {root}")
        run_frame = pd.read_csv(root / "tables" / "run_roi_effects.csv.gz")
        if (
            len(run_frame) != N_RUNS * len(REPRESENTATIONS) * N_ROIS
            or run_frame.duplicated(["run_label", "representation", "roi"]).any()
            or set(run_frame.subject.astype(int)) != {subject}
            or set(run_frame.game.astype(int)) != {game}
            or set(run_frame.run_label.astype(str)) != set(manifest["run_labels"])
            or not np.all(
                np.isfinite(run_frame[["delta_cv_r2", "residual_prediction_r"]])
            )
        ):
            raise ValueError(f"Subject run ROI grid invalid: {root}")
        fold_frame = pd.read_csv(root / "tables" / "fold_audit.csv")
        if (
            len(fold_frame) != N_RUNS * len(REPRESENTATIONS)
            or fold_frame.duplicated(["heldout_run", "representation"]).any()
            or set(fold_frame.representation.astype(str)) != set(REPRESENTATIONS)
            or set(fold_frame.heldout_run.astype(str)) != set(manifest["run_labels"])
            or set(fold_frame.n_train_runs.astype(int)) != {N_RUNS - 1}
            or set(fold_frame.n_test_tr.astype(int)) != {N_TR}
            or set(fold_frame.n_voxels.astype(int))
            != {int(manifest["n_unique_atlas_voxels"])}
            or set(fold_frame.alpha_selection_mode.astype(str))
            != {"nested_loro_full_inner_refit"}
            or set(fold_frame.alpha_selection_inner_folds.astype(int)) != {N_RUNS - 1}
            or not set(fold_frame.selected_alpha.astype(float)).issubset(
                set(TRACE_ALPHA_GRID)
            )
            or set(fold_frame.alpha_selection_coordinate_sha256.astype(str))
            != {selection_hash}
            or not np.all(np.isfinite(fold_frame[[
                "alpha_selection_score", "trace_effective_df", "finite_fraction"
            ]]))
            or not np.all(fold_frame.alpha_selection_full_inner_refit.astype(bool))
            or np.any(fold_frame.alpha_selection_outer_test_used.astype(bool))
        ):
            raise ValueError(f"Subject nested-fold audit invalid: {root}")
        trace_tokens = manifest.get("planning_trace_run_tokens")
        if (
            not isinstance(trace_tokens, list)
            or len(trace_tokens) != N_RUNS
            or any(not isinstance(token, str) or len(token) != 64 for token in trace_tokens)
        ):
            raise ValueError(f"Subject trace artifact token grid invalid: {root}")
        recorded_tables = manifest.get("tables", {})
        for filename in (
            "run_roi_effects.csv.gz",
            "subject_roi_effects.csv",
            "fold_audit.csv",
            "subject_voxel_effects.npz",
        ):
            path = root / "tables" / filename
            if recorded_tables.get(filename) != sha256_file(path):
                raise ValueError(f"Subject table provenance mismatch: {path}")
        recorded_figures = manifest.get("figures", {})
        for filename in (
            "figure1_representation_comparison.png",
            "figure2_ranked_roi_lollipops.png",
        ):
            path = root / "figures" / filename
            if recorded_figures.get(filename) != sha256_file(path):
                raise ValueError(f"Subject figure provenance mismatch: {path}")
        frames.append(frame)
        with np.load(root / "tables" / "subject_voxel_effects.npz", allow_pickle=False) as archive:
            expected_voxel_keys = {
                "coords",
                "concat29_delta_cv_r2",
                "concat29_residual_r",
                "mean29_delta_cv_r2",
                "mean29_residual_r",
            }
            if set(archive.files) != expected_voxel_keys:
                raise ValueError(f"Subject voxel keys invalid: {root}")
            payload = {name: np.asarray(archive[name]) for name in archive.files}
        coords = np.asarray(payload["coords"])
        n_voxels = int(manifest["n_unique_atlas_voxels"])
        if (
            coords.shape != (n_voxels, 3)
            or len(np.unique(coords, axis=0)) != n_voxels
            or not np.issubdtype(coords.dtype, np.integer)
            or any(
                payload[name].shape != (n_voxels,)
                for name in expected_voxel_keys if name != "coords"
            )
            or not all(
                np.all(np.isfinite(values))
                for name, values in payload.items() if name != "coords"
            )
        ):
            raise ValueError(f"Subject voxel effects contain non-finite values: {root}")
        if reference_coords is None:
            reference_coords = np.array(coords, copy=True)
        elif not np.array_equal(coords, reference_coords):
            raise ValueError("Subject atlas voxel coordinates/order differ")
        voxels.append(payload)
        manifests.append(manifest)
    return pd.concat(frames, ignore_index=True), voxels, manifests


def _infer_roi(frame: pd.DataFrame, estimand: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = [(representation, roi) for representation in REPRESENTATIONS for roi in sorted(frame.roi.unique())]
    matrix = np.column_stack([
        frame[(frame.representation == representation) & (frame.roi == roi)]
        .sort_values("subject").beta.to_numpy()
        for representation, roi in labels
    ])
    result = exact_signflip_inference(matrix)
    rows = []
    for index, (representation, roi) in enumerate(labels):
        rows.append({
            "estimand": estimand,
            "representation": representation,
            "roi": roi,
            **{name: float(values[index]) if name not in ("n_subjects", "n_sign_patterns") else int(values[index]) for name, values in result.items()},
            "significance": significance_stars(float(result["p_maxT_fwer_family"][index])),
        })
    subject_matrix = pd.DataFrame(matrix, columns=[f"{r}::{roi}" for r, roi in labels])
    subject_matrix.insert(0, "subject", np.arange(1, N_SUBJECTS + 1))
    subject_matrix.insert(1, "estimand", estimand)
    return pd.DataFrame(rows), subject_matrix


def _infer_voxels(voxels: Sequence[Mapping[str, np.ndarray]], estimand: str) -> Dict[str, np.ndarray]:
    coordinates = np.asarray(voxels[0]["coords"], dtype=np.int32)
    for payload in voxels[1:]:
        if not np.array_equal(coordinates, np.asarray(payload["coords"], dtype=np.int32)):
            raise ValueError("Voxel coordinates differ across subjects")
    matrices = []
    for short in ("concat29", "mean29"):
        matrices.append(np.vstack([np.asarray(payload[f"{short}_delta_cv_r2"], dtype=float) for payload in voxels]))
    combined = np.column_stack(matrices)
    result = exact_signflip_inference(combined)
    n_voxels = len(coordinates)
    output: Dict[str, np.ndarray] = {"coords": coordinates}
    for representation_index, short in enumerate(("concat29", "mean29")):
        region = slice(representation_index * n_voxels, (representation_index + 1) * n_voxels)
        for name, values in result.items():
            output[f"{short}_{name}"] = np.asarray(values[region])
    output["estimand"] = np.asarray([estimand])
    return output


def plot_group_figures(
    inference: pd.DataFrame,
    subject_frame: pd.DataFrame,
    voxel_payload: Mapping[str, np.ndarray],
    output_dir: Path,
    title: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {"planning_concat29": "#2b6cb0", "planning_mean29": "#dd6b20"}
    fig, axes = plt.subplots(
        1, 2, figsize=(17, 7.5), sharex=True, constrained_layout=True
    )
    for axis, representation in zip(axes, REPRESENTATIONS):
        subset = inference[inference.representation.eq(representation)].copy()
        subset = subset.reindex(subset.t_stat.abs().sort_values(ascending=False).index).head(15)
        y = np.arange(len(subset))
        axis.hlines(y, subset.ci95_low, subset.ci95_high, color=colors[representation], lw=2)
        axis.scatter(subset.beta, y, color=colors[representation], s=48, zorder=3)
        for yi, row in enumerate(subset.itertuples()):
            points = subject_frame[(subject_frame.roi == row.roi) & (subject_frame.representation == representation)].sort_values("subject").beta
            axis.scatter(points, np.full(len(points), yi), color="black", s=14, alpha=0.55)
            if row.significance:
                axis.text(max(row.ci95_high, row.beta), yi, f"  {row.significance}", va="center", fontsize=10)
        axis.axvline(0, color="black", ls="--", lw=0.8)
        axis.set_yticks(y, [compact_roi_label(name) for name in subset.roi])
        axis.invert_yaxis(); axis.set_xlabel("partial ΔCVR² (95% subject CI)")
        axis.set_title(REPRESENTATION_LABELS[representation], fontsize=10, pad=8)
        concise_numeric_axis(axis)
    fig.suptitle(
        f"{title}: per-representation ranked ROI encoding", fontsize=14
    )
    fig.savefig(output_dir / "figure1_ranked_roi_lollipops.png", dpi=220)
    plt.close(fig)

    pivot = inference.pivot(index="roi", columns="representation", values="beta")
    pmax = inference.pivot(index="roi", columns="representation", values="p_maxT_fwer_family")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    supported = (pmax < 0.05).any(axis=1)
    axes[0].scatter(pivot.planning_mean29[~supported], pivot.planning_concat29[~supported], s=25, color="#718096", alpha=0.65)
    axes[0].scatter(pivot.planning_mean29[supported], pivot.planning_concat29[supported], s=55, color="#c53030", edgecolor="black")
    bounds = np.asarray([pivot.min().min(), pivot.max().max()])
    axes[0].plot(bounds, bounds, color="black", ls="--", lw=0.8)
    axes[0].set_xlabel("Mean-29 (pooled) group ΔCVR²")
    axes[0].set_ylabel("Concat-29 (position-preserving) group ΔCVR²")
    axes[0].set_title("Representation correspondence (descriptive; not a contrast)")
    concise_numeric_axis(axes[0], include_y=True)
    axes[1].scatter(inference.beta, -np.log10(inference.p_maxT_fwer_family), c=[colors[x] for x in inference.representation], s=24, alpha=0.72)
    axes[1].axhline(-math.log10(0.05), color="black", ls="--", lw=0.8)
    axes[1].axvline(0, color="black", lw=0.6)
    axes[1].set_xlabel("group partial ΔCVR²"); axes[1].set_ylabel(r"$-\log_{10}$(family FWER p)")
    axes[1].set_title("Effect–evidence plot")
    concise_numeric_axis(axes[1])
    fig.suptitle(title)
    fig.savefig(output_dir / "figure2_representation_scatter_and_evidence.png", dpi=220)
    plt.close(fig)

    coords = np.asarray(voxel_payload["coords"])
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for row, short in enumerate(("concat29", "mean29")):
        beta = np.asarray(voxel_payload[f"{short}_beta"])
        p = np.asarray(voxel_payload[f"{short}_p_maxT_fwer_family"])
        threshold = p < 0.05
        vmax = float(np.quantile(np.abs(beta), 0.995)) + EPS
        for column, (x_index, y_index, labels) in enumerate(((0,1,("i","j")),(0,2,("i","k")),(1,2,("j","k")))):
            axis = axes[row, column]
            axis.scatter(coords[:, x_index], coords[:, y_index], c=beta, s=2.0, cmap="coolwarm", vmin=-vmax, vmax=vmax, alpha=0.22, rasterized=True)
            if np.any(threshold):
                axis.scatter(coords[threshold, x_index], coords[threshold, y_index], c=beta[threshold], s=13, cmap="coolwarm", vmin=-vmax, vmax=vmax, edgecolor="black", linewidth=0.2)
            axis.set_xlabel(labels[0]); axis.set_ylabel(labels[1]); axis.set_aspect("equal")
            if column == 0:
                axis.set_title(f"{short}: corrected voxels {int(np.sum(threshold))}")
    fig.suptitle(f"{title}: atlas-voxel projections (descriptive color; filled outline = maxT<.05)")
    fig.savefig(output_dir / "figure3_voxel_projection.png", dpi=220)
    plt.close(fig)


def run_group(args: argparse.Namespace) -> None:
    started_sources = current_source_hashes()
    frame, voxels, manifests = _load_subjects(args.output_root, int(args.game))
    inference, subject_matrix = _infer_roi(frame, f"game{args.game}")
    voxel_inference = _infer_voxels(voxels, f"game{args.game}")
    output = Path(args.output_root) / f"group_game{args.game}"
    tables = output / "tables"; figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    frame.to_csv(tables / "subject_roi_effects_long.csv", index=False)
    subject_matrix.to_csv(tables / "subject_effect_matrix.csv", index=False)
    inference.to_csv(tables / "group_exact_inference.csv", index=False)
    atomic_npz(tables / "voxel_exact_inference.npz", **voxel_inference)
    plot_group_figures(inference, frame, voxel_inference, figures, f"Game {args.game}")
    if current_source_hashes() != started_sources:
        raise RuntimeError("Analysis or shared source changed during group execution")
    input_artifacts = []
    for subject, manifest in enumerate(manifests, start=1):
        subject_root = _subject_output(args.output_root, subject, int(args.game))
        input_artifacts.append({
            "subject": subject,
            "game": int(args.game),
            "manifest_sha256": sha256_file(subject_root / "analysis_manifest.json"),
            "subject_roi_effects_sha256": manifest["tables"]["subject_roi_effects.csv"],
            "subject_voxel_effects_sha256": manifest["tables"]["subject_voxel_effects.npz"],
        })
    atomic_json(output / "analysis_manifest.json", {
        "analysis_version": ANALYSIS_VERSION,
        **started_sources,
        "settings_sha256": manifests[0]["settings_sha256"],
        "game": int(args.game), "subjects": list(range(1,7)),
        "n_sign_patterns": 64,
        "family": (
            "one a-priori game family: 2 representations x 110 ROIs; voxel "
            "family separately 2 x all unique atlas voxels"
        ),
        "representation_comparison": "descriptive only; no concat-minus-mean test",
        "input_subject_artifacts": input_artifacts,
        "tables": {
            path.name: sha256_file(path)
            for path in sorted(tables.iterdir()) if path.is_file()
        },
        "figures": {
            str(path.relative_to(figures)): sha256_file(path)
            for path in sorted(figures.rglob("*")) if path.is_file()
        },
    })
    print(f"08 planning-trace group complete: {output}", flush=True)


def run_paired(args: argparse.Namespace) -> None:
    started_sources = current_source_hashes()
    frame1, voxels1, manifests1 = _load_subjects(args.output_root, 1)
    frame2, voxels2, manifests2 = _load_subjects(args.output_root, 2)
    if manifests1[0]["scientific_contract"] != manifests2[0]["scientific_contract"]:
        raise ValueError("Scientific contracts differ across games")
    if manifests1[0]["settings_sha256"] != manifests2[0]["settings_sha256"]:
        raise ValueError("Settings hashes differ across games")
    key = ["subject", "roi", "representation"]
    paired = frame1[key + ["beta"]].merge(frame2[key + ["beta"]], on=key, suffixes=("_g1", "_g2"), validate="one_to_one")
    outputs = []
    matrices = []
    voxel_outputs: Dict[str, np.ndarray] = {}
    for estimand, expression in (
        ("across_game_mean", 0.5 * (paired.beta_g1 + paired.beta_g2)),
        ("game2_minus_game1", paired.beta_g2 - paired.beta_g1),
    ):
        current = paired[key].copy(); current["beta"] = expression
        inference, matrix = _infer_roi(current, estimand)
        outputs.append(inference); matrices.append(matrix)
        transformed = []
        for left, right in zip(voxels1, voxels2):
            if not np.array_equal(left["coords"], right["coords"]):
                raise ValueError("Game voxel coordinates differ")
            payload = {"coords": left["coords"]}
            for short in ("concat29", "mean29"):
                a = np.asarray(left[f"{short}_delta_cv_r2"])
                b = np.asarray(right[f"{short}_delta_cv_r2"])
                payload[f"{short}_delta_cv_r2"] = 0.5 * (a+b) if estimand == "across_game_mean" else b-a
            transformed.append(payload)
        voxel_result = _infer_voxels(transformed, estimand)
        for name, values in voxel_result.items():
            if name == "coords":
                voxel_outputs.setdefault("coords", values)
            elif name != "estimand":
                voxel_outputs[f"{estimand}_{name}"] = values
    inference = pd.concat(outputs, ignore_index=True)
    subject_matrix = pd.concat(matrices, ignore_index=True)
    output = Path(args.output_root) / "group_paired_games"
    tables = output / "tables"; figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    paired.to_csv(tables / "paired_subject_effects.csv", index=False)
    subject_matrix.to_csv(tables / "paired_subject_effect_matrix.csv", index=False)
    inference.to_csv(tables / "paired_exact_inference.csv", index=False)
    atomic_npz(tables / "paired_voxel_exact_inference.npz", **voxel_outputs)
    for estimand in ("across_game_mean", "game2_minus_game1"):
        roi_subset = inference[inference.estimand.eq(estimand)]
        subject_subset = paired[key].copy()
        subject_subset["beta"] = 0.5*(paired.beta_g1+paired.beta_g2) if estimand == "across_game_mean" else paired.beta_g2-paired.beta_g1
        voxel_subset = {"coords": voxel_outputs["coords"]}
        for short in ("concat29", "mean29"):
            for statistic in ("beta","t_stat","p_exact_two_sided","p_maxT_fwer_family","ci95_low","ci95_high","positive_fraction","n_subjects","n_sign_patterns"):
                voxel_subset[f"{short}_{statistic}"] = voxel_outputs[f"{estimand}_{short}_{statistic}"]
        plot_group_figures(roi_subset, subject_subset, voxel_subset, figures / estimand, estimand.replace("_", " "))
    if current_source_hashes() != started_sources:
        raise RuntimeError("Analysis or shared source changed during paired execution")
    input_artifacts = []
    for game, game_manifests in ((1, manifests1), (2, manifests2)):
        for subject, manifest in enumerate(game_manifests, start=1):
            subject_root = _subject_output(args.output_root, subject, game)
            input_artifacts.append({
                "subject": subject,
                "game": game,
                "manifest_sha256": sha256_file(subject_root / "analysis_manifest.json"),
                "subject_roi_effects_sha256": manifest["tables"]["subject_roi_effects.csv"],
                "subject_voxel_effects_sha256": manifest["tables"]["subject_voxel_effects.npz"],
            })
    atomic_json(output / "analysis_manifest.json", {
        "analysis_version": ANALYSIS_VERSION,
        **started_sources,
        "settings_sha256": manifests1[0]["settings_sha256"],
        "subjects": list(range(1,7)), "games": [1,2], "n_sign_patterns":64,
        "pairing": "within subject before inference; never 12 independent cells",
        "a_priori_separate_families": {
            "across_game_mean": (
                "2 representations x 110 ROIs; voxel family separately 2 x atlas"
            ),
            "game2_minus_game1": (
                "2 representations x 110 ROIs; voxel family separately 2 x atlas"
            ),
        },
        "representation_comparison": "descriptive only; no concat-minus-mean test",
        "input_subject_artifacts": input_artifacts,
        "tables": {
            path.name: sha256_file(path)
            for path in sorted(tables.iterdir()) if path.is_file()
        },
        "figures": {
            str(path.relative_to(figures)): sha256_file(path)
            for path in sorted(figures.rglob("*")) if path.is_file()
        },
    })
    print(f"08 planning-trace paired complete: {output}", flush=True)


def self_test() -> Dict[str, Any]:
    rng = np.random.default_rng(82)
    n = 240
    nuisance = np.column_stack((np.ones(n), rng.normal(size=(n, 4))))
    trace = rng.normal(size=(n, 12))
    signal = trace @ rng.normal(size=(12, 7))
    neural = nuisance @ rng.normal(size=(5, 7)) + 0.6 * signal + rng.normal(scale=0.3, size=(n, 7))
    split = 180
    b_op = ridge_left_operator(nuisance[:split], 1.0)
    z_coef = b_op @ trace[:split]
    z_train = trace[:split] - nuisance[:split] @ z_coef
    z_test = trace[split:] - nuisance[split:] @ z_coef
    y_coef = b_op @ neural[:split]
    y_train = neural[:split] - nuisance[:split] @ y_coef
    y_test = neural[split:] - nuisance[split:] @ y_coef
    beta = ridge_left_operator(z_train, 10.0) @ y_train
    prediction = z_test @ beta
    delta = 1.0 - np.sum((y_test-prediction)**2) / np.sum(y_test**2)
    if not delta > 0.4:
        raise AssertionError(f"Injected planning encoding was not recovered: {delta}")
    matrix = np.tile(np.linspace(0.01, 0.06, 6)[:,None], (1,3))
    exact = exact_signflip_inference(matrix)
    if not np.allclose(exact["p_exact_two_sided"], 0.03125):
        raise AssertionError("Six-subject exact sign test failed")

    # End-to-end selector sanity: three outer-training run blocks produce
    # three inner folds, and changing every outer-test array cannot alter the
    # chosen alpha or any inner score because the helper never consumes it.
    selector_rng = np.random.default_rng(8202)
    n_selector_runs, rows = 4, 48
    selector_visuals = [
        selector_rng.normal(size=(rows, 7)) for _ in range(n_selector_runs)
    ]
    selector_controls = [
        selector_rng.normal(size=(rows, 5)) for _ in range(n_selector_runs)
    ]
    selector_traces = [
        {
            "planning_concat29": selector_rng.normal(size=(rows, 9)),
            "planning_mean29": selector_rng.normal(size=(rows, 4)),
        }
        for _ in range(n_selector_runs)
    ]
    concat_weight = selector_rng.normal(size=(9, 11))
    mean_weight = selector_rng.normal(size=(4, 11))
    nuisance_weight = selector_rng.normal(size=(5, 11))
    selector_neural = [
        selector_controls[index] @ nuisance_weight
        + 0.35 * selector_traces[index]["planning_concat29"] @ concat_weight
        + 0.25 * selector_traces[index]["planning_mean29"] @ mean_weight
        + selector_rng.normal(scale=0.5, size=(rows, 11))
        for index in range(n_selector_runs)
    ]
    selector_coords = np.column_stack(
        (np.arange(11), np.arange(11) + 20, np.arange(11) + 40)
    ).astype(np.int32)
    selector_indices = coordinate_hash_selection(selector_coords)
    selector_coordinate_sha = array_sha256(selector_coords[selector_indices])
    selector_labels = [f"synthetic_run_{index}" for index in range(n_selector_runs)]
    selector_tokens = [
        hashlib.sha256(label.encode("utf-8")).hexdigest()
        for label in selector_labels
    ]
    outer_fold = _prepare_single_fold(
        selector_visuals,
        selector_controls,
        selector_traces,
        [0, 1, 2],
        3,
        82,
    )
    selected = select_trace_alphas_nested(
        outer_fold,
        selector_visuals,
        selector_controls,
        selector_traces,
        selector_neural,
        selector_indices,
        selector_coordinate_sha,
        selector_labels,
        selector_tokens,
        82,
    )
    for representation in REPRESENTATIONS:
        audit = selected[representation]
        if audit["selected_alpha"] not in TRACE_ALPHA_GRID:
            raise AssertionError("Nested selector returned alpha outside the grid")
        if audit["n_inner_folds"] != 3:
            raise AssertionError("Nested selector did not use all three run blocks")
        if not (
            np.isfinite(audit["selection_score"])
            and np.isfinite(audit["effective_df"])
            and all(np.isfinite(value) for value in audit["scores"].values())
        ):
            raise AssertionError("Nested selector audit contains non-finite values")
        if audit["outer_test_used"]:
            raise AssertionError("Nested selector reports outer-test use")
    tied_alpha, _ = _choose_largest_alpha_on_tie(
        {alpha: 0.5 for alpha in TRACE_ALPHA_GRID}
    )
    if tied_alpha != 1e12 or TRACE_ALPHA_GRID[-1] != 1e12:
        raise AssertionError("Alpha tie-break is not deterministic toward larger alpha")
    if ridge_effective_df(z_train, 1e12) >= 1e-6:
        raise AssertionError("The 1e12 endpoint is not a practical-null ridge fit")
    if "effective-null/no detectable" not in scientific_contract(2026)[
        "practical_null_interpretation"
    ]:
        raise AssertionError("Practical-null interpretation is absent from contract")

    changed_visuals = [np.array(value, copy=True) for value in selector_visuals]
    changed_controls = [np.array(value, copy=True) for value in selector_controls]
    changed_traces = [
        {name: np.array(value, copy=True) for name, value in run.items()}
        for run in selector_traces
    ]
    changed_neural = [np.array(value, copy=True) for value in selector_neural]
    changed_visuals[3] = selector_rng.normal(loc=500.0, size=changed_visuals[3].shape)
    changed_controls[3] = selector_rng.normal(loc=-500.0, size=changed_controls[3].shape)
    changed_traces[3] = {
        name: selector_rng.normal(loc=1000.0, size=value.shape)
        for name, value in changed_traces[3].items()
    }
    changed_neural[3] = selector_rng.normal(loc=-1000.0, size=changed_neural[3].shape)
    changed_outer_fold = _prepare_single_fold(
        changed_visuals,
        changed_controls,
        changed_traces,
        [0, 1, 2],
        3,
        82,
    )
    selected_after_outer_change = select_trace_alphas_nested(
        changed_outer_fold,
        changed_visuals,
        changed_controls,
        changed_traces,
        changed_neural,
        selector_indices,
        selector_coordinate_sha,
        selector_labels,
        selector_tokens,
        82,
    )
    for representation in REPRESENTATIONS:
        before = selected[representation]
        after = selected_after_outer_change[representation]
        if (
            before["selected_alpha"] != after["selected_alpha"]
            or before["selection_token"] != after["selection_token"]
            or not np.allclose(
                list(before["scores"].values()),
                list(after["scores"].values()),
                rtol=0.0,
                atol=0.0,
            )
        ):
            raise AssertionError("Outer-test mutation changed nested alpha selection")
    return {
        "injected_delta_cv_r2": float(delta),
        "exact_p": float(exact["p_exact_two_sided"][0]),
        "selector_inner_folds": 3,
        "selector_outer_test_invariant": True,
        "selector_selected_alpha": {
            representation: float(selected[representation]["selected_alpha"])
            for representation in REPRESENTATIONS
        },
        "selector_effective_df": {
            representation: float(selected[representation]["effective_df"])
            for representation in REPRESENTATIONS
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    COMMON.add_shared_cli_arguments(parser, require_subject_game=False)
    TRACE.add_trace_cli_arguments(parser)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--aggregate-subjects", nargs="+", type=int)
    parser.add_argument("--aggregate-games", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2, sort_keys=True)); return
    if args.aggregate_games:
        if args.subject is not None or args.game is not None or args.aggregate_subjects:
            raise ValueError("--aggregate-games is standalone")
        run_paired(args); return
    if args.aggregate_subjects:
        if sorted(set(args.aggregate_subjects)) != list(range(1,7)) or args.game not in (1,2):
            raise ValueError("Group inference requires --aggregate-subjects 1 2 3 4 5 6 and --game")
        run_group(args); return
    if args.subject is None or args.game is None:
        raise ValueError("--subject and --game are required")
    run_subject(args)


if __name__ == "__main__":
    main()
