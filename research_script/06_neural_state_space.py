#!/usr/bin/env python3
"""Switch/stay cross-condition generalization in ROI population activity.

This analysis asks one deliberately narrow MVPA question: does a linear decoder
trained to distinguish switch from stay in one action-origin context generalize
to the other origin context?  The two cross-directions are fit separately and
their equal-weight mean is the primary CCGP estimand, following the analysis
logic used by Sung et al.  Direct N->A versus A->N decoding, four-way decoding,
continuous encoding, RSA, and pooled neural PCA are intentionally not included.

All neural data come from the canonical native-time v4/process-v8/schema-v4
``00_brain_cache``.  One invocation analyzes one subject x game, sequentially
over ROIs.  Nuisance residualization, scaling, and the linear SVM are learned on
the ten outer training runs only.  Participant inference uses synchronized
within-run 40-TR block permutations of held-out predictions against fixed
transition labels; it never treats the overlapping LORO folds as independent.
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
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

import neural_roi_figures as NROI


SCRIPT_DIR = Path(__file__).resolve().parent
COMMON_PATH = SCRIPT_DIR / "planning_brain_common.py"
COMMON_SPEC = importlib.util.spec_from_file_location(
    "planning_brain_common_06_ccgp", COMMON_PATH
)
if COMMON_SPEC is None or COMMON_SPEC.loader is None:
    raise RuntimeError(f"Could not load shared helpers from {COMMON_PATH}")
COMMON = importlib.util.module_from_spec(COMMON_SPEC)
sys.modules[COMMON_SPEC.name] = COMMON
COMMON_SPEC.loader.exec_module(COMMON)


ANALYSIS_NAME = "06_neural_state_space"
ANALYSIS_VERSION = "2026-08-21-ccgp-only-linear-svm-v4"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs" / ANALYSIS_NAME
EXPECTED_SUBJECTS = tuple(range(1, 7))
EXPECTED_RUNS = 11
EXPECTED_ROIS = 110
EXPECTED_CACHE_VERSION = 4
EXPECTED_PROCESS_FEATURE_VERSION = 8
EXPECTED_NEURAL_SCHEMA_VERSION = 4
TRANSITION_NAMES = ("N_to_N", "N_to_A", "A_to_N", "A_to_A")
DIRECTIONS = (
    (
        "train_N_origin_test_A_origin",
        ("N_to_N", "N_to_A"),
        ("A_to_A", "A_to_N"),
    ),
    (
        "train_A_origin_test_N_origin",
        ("A_to_A", "A_to_N"),
        ("N_to_N", "N_to_A"),
    ),
)
DIRECTION_NAMES = tuple(row[0] for row in DIRECTIONS)
DIRECTION_LABELS = {
    DIRECTION_NAMES[0]: "Train hold / test act",
    DIRECTION_NAMES[1]: "Train act / test hold",
}

PARTICIPANT_PERMUTATIONS = 9_999
PARTICIPANT_BLOCK_SIZE_TR = 40
CONDITION_FRACTION = 0.08
CONDITION_MIN_TRS = 12
NUISANCE_ALPHA = 1.0
RAM_PCA_COMPONENTS = 20
VISUAL_PCA_COMPONENTS = 20
SVM_C = 1.0
SVM_MAX_ITER = 20_000
CHANCE = 0.5
SIGNFLIP_RTOL = 128.0 * np.finfo(np.float64).eps
SIGNFLIP_ATOL = 128.0 * np.finfo(np.float64).eps

INK = "#33363A"
GRAY = "#757B82"
GRID = "#D9DDE3"
BLUE = "#386D9C"
ORANGE = "#CF7B35"


def log(message: str) -> None:
    print(message, flush=True)


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


def source_signatures() -> dict[str, dict[str, Any]]:
    return {
        "script": file_signature(Path(__file__)),
        "planning_brain_common": file_signature(COMMON_PATH),
        "neural_roi_figures": file_signature(Path(NROI.__file__)),
    }


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    compression = "gzip" if path.name.endswith(".gz") else None
    frame.to_csv(temporary, index=False, compression=compression)
    temporary.replace(path)


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def fdr_bh(values: Sequence[float]) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    result = np.full_like(p, np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    if finite.size == 0:
        return result
    order = finite[np.argsort(p[finite])]
    ranked = p[order] * float(order.size) / np.arange(1, order.size + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result[order] = np.minimum(ranked, 1.0)
    return result


def permutation_ge_or_tie(
    null_statistics: Sequence[float], observed_statistic: float
) -> np.ndarray:
    null = np.asarray(null_statistics, dtype=float)
    observed = float(observed_statistic)
    return (null > observed) | np.isclose(
        null,
        observed,
        rtol=SIGNFLIP_RTOL,
        atol=SIGNFLIP_ATOL,
        equal_nan=False,
    )


def _as_finite_2d(
    values: Any, *, name: str, n_rows: int | None = None
) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim == 1:
        result = result[:, None]
    if result.ndim != 2 or result.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2-D matrix, got {result.shape}")
    if n_rows is not None and result.shape[0] != int(n_rows):
        raise ValueError(f"{name} row mismatch: {result.shape[0]} != {int(n_rows)}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _t_statistic(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[0] < 2:
        raise ValueError("t statistic requires a subject x cell matrix")
    mean = np.mean(array, axis=0)
    standard_error = np.std(array, axis=0, ddof=1) / math.sqrt(array.shape[0])
    result = np.divide(
        mean,
        standard_error,
        out=np.zeros_like(mean),
        where=standard_error > 1e-12,
    )
    degenerate = (standard_error <= 1e-12) & (np.abs(mean) > 1e-12)
    result[degenerate] = np.sign(mean[degenerate]) * np.inf
    return result


@dataclass(frozen=True)
class ConditionSelection:
    indices: dict[str, np.ndarray]
    dominance_scores: dict[str, np.ndarray]
    n_per_condition: int


@dataclass
class RunData:
    run_spec: Any
    run_label: str
    transitions: np.ndarray
    transition_controls: np.ndarray
    ram: np.ndarray
    visual: np.ndarray
    selection: ConditionSelection


def select_fourway_condition_trs(
    transition_events: np.ndarray,
    *,
    fraction_per_condition: float = CONDITION_FRACTION,
    min_per_condition: int = CONDITION_MIN_TRS,
    min_total_event: float = 1e-8,
) -> ConditionSelection:
    """Select balanced, disjoint, HRF-dominant TRs without using BOLD."""
    events = _as_finite_2d(transition_events, name="transition_events")
    if events.shape[1] != len(TRANSITION_NAMES):
        raise ValueError("transition_events must contain NN, NA, AN, AA in that order")
    if not 0.0 < float(fraction_per_condition) <= 0.25:
        raise ValueError("fraction_per_condition must lie in (0, .25]")
    winner = np.argmax(events, axis=1)
    ordered = np.sort(events, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    total = np.sum(np.maximum(events, 0.0), axis=1)
    valid = np.isfinite(margin) & (total > float(min_total_event))
    candidates = {
        name: np.flatnonzero(valid & (winner == index))
        for index, name in enumerate(TRANSITION_NAMES)
    }
    counts = {name: int(indices.size) for name, indices in candidates.items()}
    smallest = min(counts.values())
    if smallest < int(min_per_condition):
        raise ValueError(
            f"Too few condition-dominant TRs: {counts}; need {min_per_condition}"
        )
    desired = max(
        int(min_per_condition), int(math.ceil(events.shape[0] * fraction_per_condition))
    )
    n_each = min(smallest, desired)
    selected: dict[str, np.ndarray] = {}
    scores: dict[str, np.ndarray] = {}
    for name, indices in candidates.items():
        order = np.lexsort((indices, -margin[indices]))
        chosen = np.sort(indices[order[:n_each]]).astype(int)
        selected[name] = chosen
        scores[name] = margin[chosen].astype(float)
    concatenated = np.concatenate(list(selected.values()))
    if np.unique(concatenated).size != concatenated.size:
        raise AssertionError("Condition selections overlap")
    return ConditionSelection(selected, scores, int(n_each))


def _stack_selected(
    patterns_by_run: Sequence[np.ndarray],
    selections: Sequence[ConditionSelection],
    run_indices: Sequence[int],
    condition_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    samples: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for run_index in run_indices:
        for label, condition in enumerate(condition_names):
            indices = selections[run_index].indices[condition]
            samples.append(np.asarray(patterns_by_run[run_index])[indices])
            labels.append(np.full(indices.size, label, dtype=np.int8))
    return np.vstack(samples), np.concatenate(labels)


def balanced_accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    truth = np.asarray(y_true, dtype=int)
    prediction = np.asarray(y_pred, dtype=int)
    recalls = []
    for label in (0, 1):
        selected = truth == label
        if not np.any(selected):
            return float("nan")
        recalls.append(float(np.mean(prediction[selected] == label)))
    return float(np.mean(recalls))


def fit_linear_svm_predict_all(
    train_x: np.ndarray,
    train_y: np.ndarray,
    heldout_all_x: np.ndarray,
    *,
    svm_c: float,
    random_state: int,
) -> np.ndarray:
    """Train-only voxel scaling plus a fixed-C binary linear SVM."""
    x_train = _as_finite_2d(train_x, name="svm_train_x")
    x_test = _as_finite_2d(
        heldout_all_x, name="svm_heldout_all_x", n_rows=heldout_all_x.shape[0]
    )
    scaler = StandardScaler().fit(x_train)
    train_z = scaler.transform(x_train)
    test_z = scaler.transform(x_test)
    model = LinearSVC(
        C=float(svm_c),
        dual="auto",
        class_weight=None,
        max_iter=SVM_MAX_ITER,
        random_state=int(random_state),
    )
    model.fit(train_z, np.asarray(train_y, dtype=int))
    if int(model.n_iter_) >= SVM_MAX_ITER:
        raise RuntimeError("LinearSVC did not converge; do not use this fold")
    return np.asarray(model.predict(test_z), dtype=np.int8)


def _pca_train_test(
    train: np.ndarray,
    test: np.ndarray,
    n_components: int,
    *,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(train)
    train_z = scaler.transform(train)
    test_z = scaler.transform(test)
    maximum = min(int(n_components), train_z.shape[0] - 1, train_z.shape[1])
    if maximum <= 0:
        return np.zeros((train_z.shape[0], 0)), np.zeros((test_z.shape[0], 0))
    solver = "randomized" if maximum < min(train_z.shape) else "full"
    pca = PCA(n_components=maximum, svd_solver=solver, random_state=random_state)
    return pca.fit_transform(train_z), pca.transform(test_z)


def residualize_train_test(
    train_data: np.ndarray,
    test_data: np.ndarray,
    train_controls: np.ndarray,
    test_controls: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(train_controls)
    control_train = np.column_stack([np.ones(len(train_controls)), scaler.transform(train_controls)])
    control_test = np.column_stack([np.ones(len(test_controls)), scaler.transform(test_controls)])
    gram = control_train.T @ control_train
    gram.flat[:: gram.shape[0] + 1] += float(alpha)
    right = control_train.T @ train_data
    try:
        coefficients = np.linalg.solve(gram, right)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(gram, right, rcond=None)[0]
    return train_data - control_train @ coefficients, test_data - control_test @ coefficients


def precompute_fold_controls(
    runs: Sequence[RunData],
    *,
    ram_components: int,
    visual_components: int,
    include_ram: bool,
    random_state: int,
) -> list[tuple[np.ndarray, np.ndarray, list[int]]]:
    """Fit neural-independent nuisance PC transforms once per outer fold."""
    result: list[tuple[np.ndarray, np.ndarray, list[int]]] = []
    for held_out in range(len(runs)):
        train_runs = [index for index in range(len(runs)) if index != held_out]
        train_lengths = [runs[index].transition_controls.shape[0] for index in train_runs]
        train_parts = [np.vstack([runs[index].transition_controls for index in train_runs])]
        test_parts = [runs[held_out].transition_controls]
        if include_ram and int(ram_components) > 0:
            train_ram, test_ram = _pca_train_test(
                np.vstack([runs[index].ram for index in train_runs]),
                runs[held_out].ram,
                int(ram_components),
                random_state=int(random_state) + 101 * held_out + 13,
            )
            train_parts.append(train_ram)
            test_parts.append(test_ram)
        if int(visual_components) > 0:
            train_visual, test_visual = _pca_train_test(
                np.vstack([runs[index].visual for index in train_runs]),
                runs[held_out].visual,
                int(visual_components),
                random_state=int(random_state) + 101 * held_out + 29,
            )
            train_parts.append(train_visual)
            test_parts.append(test_visual)
        result.append((np.hstack(train_parts), np.hstack(test_parts), train_lengths))
    return result


def transition_residual_pattern_fold(
    patterns_by_run: Sequence[np.ndarray],
    held_out: int,
    *,
    nuisance_alpha: float,
    control_fold: tuple[np.ndarray, np.ndarray, list[int]],
) -> list[np.ndarray]:
    train_runs = [index for index in range(len(patterns_by_run)) if index != held_out]
    train_x = np.vstack([patterns_by_run[index] for index in train_runs])
    test_x = np.asarray(patterns_by_run[held_out], dtype=float)
    train_controls, test_controls, train_lengths = control_fold
    train_residual, test_residual = residualize_train_test(
        train_x, test_x, train_controls, test_controls, alpha=float(nuisance_alpha)
    )
    result: list[np.ndarray | None] = [None] * len(patterns_by_run)
    start = 0
    for run_index, length in zip(train_runs, train_lengths):
        result[run_index] = np.asarray(train_residual[start : start + length], dtype=np.float32)
        start += length
    result[held_out] = np.asarray(test_residual, dtype=np.float32)
    if start != train_residual.shape[0] or any(item is None for item in result):
        raise AssertionError("Residualized fold could not be split back into runs")
    return [np.asarray(item) for item in result]


def generate_synchronized_block_orders(
    run_lengths: Sequence[int],
    *,
    n_permutations: int,
    block_size: int,
    random_state: int,
) -> list[np.ndarray]:
    """One fixed permutation schedule per run, shared by every ROI/direction."""
    schedules: list[np.ndarray] = []
    for run_index, length in enumerate(run_lengths):
        if int(length) % int(block_size) != 0:
            raise ValueError(
                f"Run length {length} is not divisible by {block_size}-TR blocks"
            )
        n_blocks = int(length) // int(block_size)
        rng = np.random.default_rng(int(random_state) + 10_007 * run_index)
        schedules.append(
            np.vstack([rng.permutation(n_blocks) for _ in range(int(n_permutations))]).astype(np.int16)
        )
    return schedules


def _selected_test_labels(
    selection: ConditionSelection, conditions: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.concatenate([selection.indices[name] for name in conditions])
    labels = np.concatenate(
        [
            np.full(selection.indices[name].size, label, dtype=np.int8)
            for label, name in enumerate(conditions)
        ]
    )
    order = np.argsort(indices)
    return indices[order], labels[order]


def block_permuted_balanced_accuracy(
    prediction_all: np.ndarray,
    selected_indices: np.ndarray,
    selected_labels: np.ndarray,
    block_orders: np.ndarray,
    *,
    block_size: int,
) -> np.ndarray:
    """Evaluate block permutations from a small destination/source score table."""
    prediction = np.asarray(prediction_all, dtype=int)
    indices = np.asarray(selected_indices, dtype=int)
    labels = np.asarray(selected_labels, dtype=int)
    orders = np.asarray(block_orders, dtype=int)
    n_blocks = prediction.size // int(block_size)
    if prediction.size % int(block_size) or orders.shape[1] != n_blocks:
        raise ValueError("Prediction length and block schedule are incompatible")
    class_counts = np.asarray([np.sum(labels == label) for label in (0, 1)], dtype=float)
    if np.any(class_counts == 0):
        raise ValueError("Balanced accuracy requires both held-out classes")
    contribution = np.zeros((n_blocks, n_blocks), dtype=float)
    for destination in range(n_blocks):
        mask = indices // int(block_size) == destination
        destination_indices = indices[mask]
        destination_labels = labels[mask]
        offsets = destination_indices % int(block_size)
        for source in range(n_blocks):
            source_predictions = prediction[source * int(block_size) + offsets]
            weights = np.where(
                destination_labels == 0,
                0.5 / class_counts[0],
                0.5 / class_counts[1],
            )
            contribution[destination, source] = float(
                np.sum(weights * (source_predictions == destination_labels))
            )
    destinations = np.arange(n_blocks, dtype=int)[None, :]
    return np.sum(contribution[destinations, orders], axis=1)


def ccgp_loro_roi(
    patterns_by_run: Sequence[np.ndarray],
    runs: Sequence[RunData],
    *,
    nuisance_alpha: float,
    svm_c: float,
    control_folds: Sequence[tuple[np.ndarray, np.ndarray, list[int]]],
    block_orders: Sequence[np.ndarray],
    block_size: int,
    random_state: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return direction-by-run folds and direction-by-permutation null betas."""
    selections = [run.selection for run in runs]
    rows: list[dict[str, Any]] = []
    null_accuracy = np.empty(
        (len(DIRECTIONS), len(runs), block_orders[0].shape[0]), dtype=np.float64
    )
    for held_out in range(len(runs)):
        residual = transition_residual_pattern_fold(
            patterns_by_run,
            held_out,
            nuisance_alpha=float(nuisance_alpha),
            control_fold=control_folds[held_out],
        )
        train_runs = [index for index in range(len(runs)) if index != held_out]
        for direction_index, (direction, train_conditions, test_conditions) in enumerate(DIRECTIONS):
            train_x, train_y = _stack_selected(
                residual, selections, train_runs, train_conditions
            )
            prediction_all = fit_linear_svm_predict_all(
                train_x,
                train_y,
                residual[held_out],
                svm_c=float(svm_c),
                random_state=int(random_state) + 97 * held_out + direction_index,
            )
            test_indices, test_y = _selected_test_labels(
                selections[held_out], test_conditions
            )
            score = balanced_accuracy(test_y, prediction_all[test_indices])
            rows.append(
                {
                    "held_out_run": runs[held_out].run_label,
                    "direction": direction,
                    "balanced_accuracy": score,
                    "beta": score - CHANCE,
                    "chance": CHANCE,
                    "n_test": int(test_indices.size),
                    "n_test_per_class": int(test_indices.size // 2),
                    "classifier": "LinearSVC",
                    "svm_c": float(svm_c),
                }
            )
            null_accuracy[direction_index, held_out] = block_permuted_balanced_accuracy(
                prediction_all,
                test_indices,
                test_y,
                block_orders[held_out],
                block_size=int(block_size),
            )
    null_beta = np.mean(null_accuracy, axis=1) - CHANCE
    return pd.DataFrame(rows), np.asarray(null_beta, dtype=np.float64)


def _required_process_block(
    process: Mapping[str, Any], name: str, *, n_rows: int | None = None
) -> np.ndarray:
    if name not in process:
        raise KeyError(f"Canonical neural cache lacks process block {name!r}")
    return _as_finite_2d(process[name], name=name, n_rows=n_rows)


def load_run_designs(
    run_specs: Sequence[Any],
    *,
    fraction_per_condition: float,
    min_per_condition: int,
) -> tuple[list[RunData], pd.DataFrame]:
    runs: list[RunData] = []
    audit_rows: list[dict[str, Any]] = []
    for run in run_specs:
        if str(run.cache_schema) != "canonical_00brain_nativehrf_schema4_v8":
            raise ValueError(f"Noncanonical run cache schema: {run.cache_schema}")
        process = COMMON.load_process_blocks(run)
        transitions = _required_process_block(process, "transition_events")[:, :4]
        n_tr = transitions.shape[0]
        history = _required_process_block(process, "task_history", n_rows=n_tr)
        decision = _required_process_block(process, "decision_event", n_rows=n_tr)
        drift = _required_process_block(process, "drift", n_rows=n_tr)
        # Current and previous action one-hots algebraically define the decoded
        # transition endpoints.  They are audited in the canonical cache but are
        # not nuisance-regressed from the CCGP patterns.
        _required_process_block(process, "current_action", n_rows=n_tr)
        _required_process_block(process, "previous_action", n_rows=n_tr)
        controls = np.column_stack([history, decision, drift]).astype(np.float32)
        selection = select_fourway_condition_trs(
            transitions,
            fraction_per_condition=float(fraction_per_condition),
            min_per_condition=int(min_per_condition),
        )
        ram = _as_finite_2d(
            np.load(run.ram_path, allow_pickle=False), name=f"{run.run_label}:ram", n_rows=n_tr
        ).astype(np.float32)
        visual = _as_finite_2d(
            COMMON.load_visual(run), name=f"{run.run_label}:visual", n_rows=n_tr
        ).astype(np.float32)
        runs.append(
            RunData(
                run_spec=run,
                run_label=str(run.run_label),
                transitions=np.asarray(transitions, dtype=np.float32),
                transition_controls=controls,
                ram=ram,
                visual=visual,
                selection=selection,
            )
        )
        for condition in TRANSITION_NAMES:
            scores = selection.dominance_scores[condition]
            audit_rows.append(
                {
                    "run": str(run.run_label),
                    "condition": condition,
                    "n_selected": int(selection.indices[condition].size),
                    "dominance_mean": float(np.mean(scores)),
                    "dominance_min": float(np.min(scores)),
                    "dominance_max": float(np.max(scores)),
                }
            )
    if len(runs) < 2:
        raise ValueError("CCGP LORO requires at least two runs")
    return runs, pd.DataFrame(audit_rows)


def participant_inference(
    subject_effects: pd.DataFrame,
    *,
    roi_names: Sequence[str],
    null_cube: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Participant primary 110-cell and secondary 2x110 maxT inference."""
    rois = tuple(str(name) for name in roi_names)
    null = np.asarray(null_cube, dtype=float)
    if null.shape[:2] != (len(rois), len(DIRECTIONS)):
        raise ValueError(f"Participant null cube shape mismatch: {null.shape}")
    indexed = subject_effects.set_index("roi", verify_integrity=True).reindex(rois)
    if indexed.isna().any().any():
        raise ValueError("Participant effects are incomplete")
    primary_null = np.mean(null, axis=1)
    primary_family_max = np.max(primary_null, axis=0)
    direction_family_max = np.max(null, axis=(0, 1))
    denominator = float(null.shape[-1] + 1)
    primary_rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    for roi_index, roi in enumerate(rois):
        source = indexed.loc[roi]
        observed = float(source["beta"])
        p_raw = float(
            (1.0 + np.sum(permutation_ge_or_tie(primary_null[roi_index], observed)))
            / denominator
        )
        p_max = float(
            (1.0 + np.sum(permutation_ge_or_tie(primary_family_max, observed)))
            / denominator
        )
        both_positive = bool(
            float(source[f"beta_{DIRECTION_NAMES[0]}"]) > 0.0
            and float(source[f"beta_{DIRECTION_NAMES[1]}"]) > 0.0
        )
        primary_rows.append(
            {
                "subject": int(source["subject"]),
                "game": int(source["game"]),
                "roi": roi,
                "balanced_accuracy": float(source["balanced_accuracy"]),
                "beta": observed,
                f"accuracy_{DIRECTION_NAMES[0]}": float(
                    source[f"accuracy_{DIRECTION_NAMES[0]}"]
                ),
                f"accuracy_{DIRECTION_NAMES[1]}": float(
                    source[f"accuracy_{DIRECTION_NAMES[1]}"]
                ),
                "both_directions_above_chance": both_positive,
                "p_block_one_sided": p_raw,
                "p_block_maxT_fwer_110": p_max,
                "participant_ccgp_supported": bool(p_max <= 0.05 and observed > 0 and both_positive),
                "correction_family": f"participant_primary_mean_ccgp_{len(rois)}_parcels",
                "n_permutations": int(null.shape[-1]),
            }
        )
        for direction_index, direction in enumerate(DIRECTION_NAMES):
            direction_observed = float(source[f"beta_{direction}"])
            cell_null = null[roi_index, direction_index]
            direction_rows.append(
                {
                    "subject": int(source["subject"]),
                    "game": int(source["game"]),
                    "roi": roi,
                    "direction": direction,
                    "balanced_accuracy": direction_observed + CHANCE,
                    "beta": direction_observed,
                    "p_block_one_sided": float(
                        (1.0 + np.sum(permutation_ge_or_tie(cell_null, direction_observed)))
                        / denominator
                    ),
                    "p_block_maxT_fwer_2x110": float(
                        (
                            1.0
                            + np.sum(
                                permutation_ge_or_tie(
                                    direction_family_max, direction_observed
                                )
                            )
                        )
                        / denominator
                    ),
                    "correction_family": f"participant_directional_ccgp_2_x_{len(rois)}_parcels",
                    "n_permutations": int(null.shape[-1]),
                }
            )
    primary = pd.DataFrame(primary_rows)
    primary["q_bh_participant_110"] = fdr_bh(
        primary["p_block_one_sided"].to_numpy(float)
    )
    directions = pd.DataFrame(direction_rows)
    directions["q_bh_participant_2x110"] = fdr_bh(
        directions["p_block_one_sided"].to_numpy(float)
    )
    return primary, directions


def predefined_participant_inference(
    primary_inference: pd.DataFrame,
    *,
    roi_names: Sequence[str],
    null_cube: np.ndarray,
) -> pd.DataFrame:
    """Separate targeted family of eight equal-L/R parcel averages."""
    mapping = NROI.predefined_roi_mapping_frame()
    roi_index = {str(roi): index for index, roi in enumerate(roi_names)}
    required_components = set(mapping["component_roi"].astype(str))
    if not required_components.issubset(roi_index):
        return pd.DataFrame()
    observed = NROI.aggregate_predefined_rois(
        primary_inference,
        value_columns=(
            "beta",
            "balanced_accuracy",
            f"accuracy_{DIRECTION_NAMES[0]}",
            f"accuracy_{DIRECTION_NAMES[1]}",
        ),
        grouping_columns=("subject", "game"),
    )
    null = np.asarray(null_cube, dtype=float)
    targeted_null = np.empty((len(NROI.ROI_ORDER), null.shape[-1]), dtype=float)
    for target_index, roi_key in enumerate(NROI.ROI_ORDER):
        components = mapping.loc[mapping["roi_key"].eq(roi_key), "component_roi"].astype(str)
        component_indices = [roi_index[name] for name in components]
        targeted_null[target_index] = np.mean(
            np.mean(null[component_indices], axis=1), axis=0
        )
    family_max = np.max(targeted_null, axis=0)
    denominator = float(targeted_null.shape[-1] + 1)
    rows: list[dict[str, Any]] = []
    indexed = observed.set_index("roi_key", verify_integrity=True)
    for target_index, roi_key in enumerate(NROI.ROI_ORDER):
        source = indexed.loc[roi_key]
        beta = float(source["beta"])
        p_raw = float(
            (1.0 + np.sum(permutation_ge_or_tie(targeted_null[target_index], beta)))
            / denominator
        )
        p_max = float(
            (1.0 + np.sum(permutation_ge_or_tie(family_max, beta))) / denominator
        )
        direction_accuracies = [
            float(source[f"accuracy_{direction}"]) for direction in DIRECTION_NAMES
        ]
        both_positive = bool(all(value > CHANCE for value in direction_accuracies))
        rows.append(
            {
                "subject": int(source["subject"]),
                "game": int(source["game"]),
                "roi_key": roi_key,
                "roi_label": str(source["roi_label"]),
                "anatomical_label": str(source["anatomical_label"]),
                "balanced_accuracy": float(source["balanced_accuracy"]),
                "beta": beta,
                f"accuracy_{DIRECTION_NAMES[0]}": direction_accuracies[0],
                f"accuracy_{DIRECTION_NAMES[1]}": direction_accuracies[1],
                "both_directions_above_chance": both_positive,
                "p_block_one_sided": p_raw,
                "p_block_maxT_fwer_8": p_max,
                "participant_ccgp_supported": bool(p_max <= 0.05 and beta > 0 and both_positive),
                "n_component_parcels": 2,
                "hemisphere_weighting": "equal_left_right_mean_of_unilateral_decoders",
                "mapping_version": NROI.ROI_MAPPING_VERSION,
                "roi_analysis_role": "literature_defined_targeted_secondary",
            }
        )
    result = pd.DataFrame(rows)
    result["q_bh_participant_8"] = fdr_bh(
        result["p_block_one_sided"].to_numpy(float)
    )
    return result


def exact_group_maxT(
    values: np.ndarray,
    *,
    cell_names: Sequence[str],
    p_column: str,
) -> pd.DataFrame:
    """Six-subject exact synchronized sign flips and two-sided max-|t|."""
    array = np.asarray(values, dtype=float)
    if array.shape[0] != len(EXPECTED_SUBJECTS) or array.shape[1] != len(cell_names):
        raise ValueError("Group matrix does not match subjects 1..6 and cell names")
    if not np.all(np.isfinite(array)):
        raise ValueError("Group matrix contains non-finite effects")
    signs = np.asarray(
        list(itertools.product((-1.0, 1.0), repeat=len(EXPECTED_SUBJECTS))),
        dtype=float,
    )
    observed_t = _t_statistic(array)
    null_t = np.vstack([_t_statistic(array * sign[:, None]) for sign in signs])
    max_abs = np.max(np.abs(null_t), axis=1)
    rows = []
    for column, cell in enumerate(cell_names):
        rows.append(
            {
                "cell": str(cell),
                "group_beta": float(np.mean(array[:, column])),
                "t_stat": float(observed_t[column]),
                "p_exact_two_sided": float(
                    np.mean(
                        permutation_ge_or_tie(
                            np.abs(null_t[:, column]), abs(observed_t[column])
                        )
                    )
                ),
                p_column: float(
                    np.mean(permutation_ge_or_tie(max_abs, abs(observed_t[column])))
                ),
                "n_subjects": len(EXPECTED_SUBJECTS),
                "n_sign_patterns": len(signs),
            }
        )
    result = pd.DataFrame(rows)
    result["q_bh"] = fdr_bh(result["p_exact_two_sided"].to_numpy(float))
    return result


def group_inference(
    subject_effects: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subjects = tuple(
        sorted(pd.to_numeric(subject_effects["subject"], errors="raise").astype(int).unique())
    )
    if subjects != EXPECTED_SUBJECTS:
        raise ValueError(f"Group inference requires subjects 1..6, got {subjects}")
    primary_matrix = subject_effects.pivot(index="subject", columns="roi", values="beta").reindex(EXPECTED_SUBJECTS)
    if primary_matrix.shape[1] != EXPECTED_ROIS or primary_matrix.isna().any().any():
        raise ValueError("Group primary matrix must be complete 6 x 110")
    primary = exact_group_maxT(
        primary_matrix.to_numpy(float),
        cell_names=primary_matrix.columns.astype(str),
        p_column="p_maxT_fwer_110",
    ).rename(columns={"cell": "roi"})
    primary["group_balanced_accuracy"] = primary["group_beta"] + CHANCE
    direction_frames = []
    for direction in DIRECTION_NAMES:
        frame = subject_effects.pivot(
            index="subject", columns="roi", values=f"beta_{direction}"
        ).reindex(EXPECTED_SUBJECTS)
        frame.columns = [f"{direction}@@{roi}" for roi in frame.columns]
        direction_frames.append(frame)
    direction_matrix = pd.concat(direction_frames, axis=1)
    direction = exact_group_maxT(
        direction_matrix.to_numpy(float),
        cell_names=direction_matrix.columns.astype(str),
        p_column="p_maxT_fwer_2x110",
    )
    split = direction["cell"].str.split("@@", n=1, expand=True)
    direction["direction"] = split[0]
    direction["roi"] = split[1]
    direction = direction.drop(columns="cell")
    direction["group_balanced_accuracy"] = direction["group_beta"] + CHANCE
    direction_lookup = direction.set_index(["roi", "direction"])
    primary[f"group_accuracy_{DIRECTION_NAMES[0]}"] = [
        float(direction_lookup.loc[(roi, DIRECTION_NAMES[0]), "group_beta"] + CHANCE)
        for roi in primary["roi"]
    ]
    primary[f"group_accuracy_{DIRECTION_NAMES[1]}"] = [
        float(direction_lookup.loc[(roi, DIRECTION_NAMES[1]), "group_beta"] + CHANCE)
        for roi in primary["roi"]
    ]
    primary["both_direction_group_means_above_chance"] = (
        primary[f"group_accuracy_{DIRECTION_NAMES[0]}"].gt(CHANCE)
        & primary[f"group_accuracy_{DIRECTION_NAMES[1]}"].gt(CHANCE)
    )
    primary["group_ccgp_supported"] = (
        primary["p_maxT_fwer_110"].le(0.05)
        & primary["group_beta"].gt(0)
        & primary["both_direction_group_means_above_chance"]
    )
    return primary, direction


def predefined_group_inference(
    subject_effects: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predefined_subject = NROI.aggregate_predefined_rois(
        subject_effects,
        value_columns=(
            "beta",
            "balanced_accuracy",
            f"accuracy_{DIRECTION_NAMES[0]}",
            f"accuracy_{DIRECTION_NAMES[1]}",
        ),
        grouping_columns=("subject", "game"),
    )
    matrix = predefined_subject.pivot(
        index="subject", columns="roi_key", values="beta"
    ).reindex(index=EXPECTED_SUBJECTS, columns=NROI.ROI_ORDER)
    if matrix.isna().any().any():
        raise ValueError("Targeted group matrix must be complete 6 x 8")
    inference = exact_group_maxT(
        matrix.to_numpy(float),
        cell_names=NROI.ROI_ORDER,
        p_column="p_maxT_fwer_8",
    ).rename(columns={"cell": "roi_key"})
    inference["group_balanced_accuracy"] = inference["group_beta"] + CHANCE
    inference["roi_label"] = inference["roi_key"].map(NROI.ROI_LABELS)
    for direction in DIRECTION_NAMES:
        direction_mean = (
            predefined_subject.groupby("roi_key", sort=False)[f"accuracy_{direction}"]
            .mean()
            .reindex(NROI.ROI_ORDER)
        )
        inference[f"group_accuracy_{direction}"] = inference["roi_key"].map(direction_mean)
    inference["both_direction_group_means_above_chance"] = (
        inference[f"group_accuracy_{DIRECTION_NAMES[0]}"].gt(CHANCE)
        & inference[f"group_accuracy_{DIRECTION_NAMES[1]}"].gt(CHANCE)
    )
    inference["group_ccgp_supported"] = (
        inference["p_maxT_fwer_8"].le(0.05)
        & inference["group_beta"].gt(0)
        & inference["both_direction_group_means_above_chance"]
    )
    inference["hemisphere_weighting"] = "equal_left_right_mean_of_unilateral_decoders"
    inference["mapping_version"] = NROI.ROI_MAPPING_VERSION
    inference["roi_analysis_role"] = "literature_defined_targeted_secondary"
    return predefined_subject, inference


def _figure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.titlesize": 7.6,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_subject_predefined_ccgp(
    effects: pd.DataFrame, output_path: Path, *, subject: int, game: int
) -> None:
    """Sung-style participant CCGP summary; LORO folds are not error bars."""
    if effects.empty:
        return
    _figure_style()
    indexed = effects.set_index("roi_key", verify_integrity=True).reindex(NROI.ROI_ORDER)
    x = np.arange(len(NROI.ROI_ORDER), dtype=float)
    figure, axis = plt.subplots(figsize=(6.0, 2.65), layout="constrained")
    axis.axhline(CHANCE, color=GRAY, linewidth=0.8, linestyle="--", zorder=0)
    axis.scatter(
        x - 0.16,
        indexed[f"accuracy_{DIRECTION_NAMES[0]}"],
        s=24,
        marker="o",
        facecolor=BLUE,
        edgecolor=BLUE,
        linewidth=0.7,
        label=DIRECTION_LABELS[DIRECTION_NAMES[0]],
        zorder=3,
    )
    axis.scatter(
        x + 0.16,
        indexed[f"accuracy_{DIRECTION_NAMES[1]}"],
        s=27,
        marker="s",
        facecolor="white",
        edgecolor=ORANGE,
        linewidth=1.0,
        label=DIRECTION_LABELS[DIRECTION_NAMES[1]],
        zorder=3,
    )
    axis.scatter(
        x,
        indexed["balanced_accuracy"],
        s=29,
        marker="D",
        facecolor=INK,
        edgecolor="white",
        linewidth=0.45,
        label="Bidirectional mean CCGP",
        zorder=4,
    )
    values = indexed[
        [
            "balanced_accuracy",
            f"accuracy_{DIRECTION_NAMES[0]}",
            f"accuracy_{DIRECTION_NAMES[1]}",
        ]
    ].to_numpy(float)
    lower = min(0.44, float(np.min(values)) - 0.035)
    upper = max(0.58, float(np.max(values)) + 0.055)
    for position, supported in zip(x, indexed["participant_ccgp_supported"].astype(bool)):
        if supported:
            axis.plot([position - 0.16, position + 0.16], [upper - 0.014] * 2, color=INK, linewidth=1.2)
    axis.plot([], [], color=INK, linewidth=1.2, label="Participant maxT-corrected")
    axis.set_ylim(lower, upper)
    axis.set_xticks(x, [NROI.ROI_LABELS[key] for key in NROI.ROI_ORDER])
    axis.set_ylabel("Switch/stay CCGP accuracy")
    axis.set_title(
        f"Subject {int(subject):03d} · Game {int(game)} · switch/stay CCGP across origin states",
        loc="center",
    )
    axis.grid(axis="y", color=GRID, linewidth=0.45)
    axis.legend(frameon=False, fontsize=5.5, ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.19))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, facecolor="white", metadata=NROI.PDF_METADATA)
    plt.close(figure)


def plot_group_predefined_ccgp(
    subject_effects: pd.DataFrame,
    inference: pd.DataFrame,
    output_path: Path,
    *,
    game: int,
) -> None:
    """All six participant estimates plus the two crossed directions."""
    _figure_style()
    tests = inference.set_index("roi_key", verify_integrity=True).reindex(NROI.ROI_ORDER)
    x = np.arange(len(NROI.ROI_ORDER), dtype=float)
    figure, axis = plt.subplots(figsize=(6.0, 2.85), layout="constrained")
    rng = np.random.default_rng(20_260_821 + int(game))
    accuracy_by_roi = [
        subject_effects.loc[
            subject_effects["roi_key"].eq(roi_key), "balanced_accuracy"
        ].to_numpy(float)
        for roi_key in NROI.ROI_ORDER
    ]
    axis.boxplot(
        accuracy_by_roi,
        positions=x,
        widths=0.50,
        patch_artist=True,
        showfliers=False,
        boxprops={"facecolor": "#ECEFF2", "edgecolor": "#A9AFB5", "linewidth": 0.7},
        medianprops={"color": INK, "linewidth": 0.9},
        whiskerprops={"color": "#A9AFB5", "linewidth": 0.65},
        capprops={"color": "#A9AFB5", "linewidth": 0.65},
    )
    for position, values in zip(x, accuracy_by_roi):
        if len(values) != len(EXPECTED_SUBJECTS):
            raise ValueError("Every group ROI figure must contain all six participant dots")
        axis.scatter(
            position + rng.uniform(-0.10, 0.10, size=len(values)),
            values,
            s=13,
            color="#8B9299",
            edgecolor="white",
            linewidth=0.3,
            zorder=3,
        )
    axis.scatter(
        x - 0.18,
        tests[f"group_accuracy_{DIRECTION_NAMES[0]}"],
        s=24,
        marker="o",
        facecolor=BLUE,
        edgecolor=BLUE,
        linewidth=0.7,
        label=DIRECTION_LABELS[DIRECTION_NAMES[0]],
        zorder=4,
    )
    axis.scatter(
        x + 0.18,
        tests[f"group_accuracy_{DIRECTION_NAMES[1]}"],
        s=27,
        marker="s",
        facecolor="white",
        edgecolor=ORANGE,
        linewidth=1.0,
        label=DIRECTION_LABELS[DIRECTION_NAMES[1]],
        zorder=4,
    )
    axis.scatter(
        x,
        tests["group_beta"] + CHANCE,
        s=30,
        marker="D",
        facecolor=INK,
        edgecolor="white",
        linewidth=0.45,
        label="Group bidirectional mean",
        zorder=5,
    )
    axis.axhline(CHANCE, color=GRAY, linewidth=0.8, linestyle="--", zorder=0)
    all_values = np.concatenate(
        [
            subject_effects["balanced_accuracy"].to_numpy(float),
            tests[f"group_accuracy_{DIRECTION_NAMES[0]}"].to_numpy(float),
            tests[f"group_accuracy_{DIRECTION_NAMES[1]}"].to_numpy(float),
        ]
    )
    lower = min(0.44, float(np.min(all_values)) - 0.035)
    upper = max(0.58, float(np.max(all_values)) + 0.06)
    for position, supported in zip(x, tests["group_ccgp_supported"].astype(bool)):
        if supported:
            axis.plot([position - 0.16, position + 0.16], [upper - 0.014] * 2, color=INK, linewidth=1.2)
    axis.plot([], [], color=INK, linewidth=1.2, label="Group maxT-corrected")
    axis.set_ylim(lower, upper)
    axis.set_xticks(x, [NROI.ROI_LABELS[key] for key in NROI.ROI_ORDER])
    axis.set_ylabel("Switch/stay CCGP accuracy")
    axis.set_title(
        f"Game {int(game)} · switch/stay CCGP across origin states",
        loc="center",
    )
    axis.grid(axis="y", color=GRID, linewidth=0.45)
    axis.legend(frameon=False, fontsize=5.5, ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.19))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, facecolor="white", metadata=NROI.PDF_METADATA)
    plt.close(figure)


def save_subject_brain_map(
    inference: pd.DataFrame, output_path: Path, *, subject: int, game: int
) -> None:
    selected = inference.loc[
        inference["participant_ccgp_supported"].astype(bool), ["roi", "beta"]
    ].copy()
    NROI.save_effect_brain_maps(
        output_path,
        panels=[("Bidirectional switch/stay CCGP", selected)],
        value_column="beta",
        title=f"Subject {int(subject):03d} · Game {int(game)} · switch/stay CCGP across origin states",
        subtitle="Within-participant 40-TR block-permutation maxT-FWER corrected parcels",
        metric_label="CCGP accuracy - .50",
        signed=False,
        n_columns=1,
    )


def save_group_brain_map(
    inference: pd.DataFrame, output_path: Path, *, game: int
) -> None:
    selected = inference.loc[
        inference["group_ccgp_supported"].astype(bool), ["roi", "group_beta"]
    ].copy()
    NROI.save_effect_brain_maps(
        output_path,
        panels=[("Bidirectional switch/stay CCGP", selected)],
        value_column="group_beta",
        title=f"Game {int(game)} · switch/stay CCGP across origin states",
        subtitle="Six-participant exact sign-flip maxT-FWER corrected parcels",
        metric_label="CCGP accuracy - .50",
        signed=False,
        n_columns=1,
    )


def _official_cli_contract(args: argparse.Namespace) -> bool:
    return bool(
        Path(args.cache_root).resolve() == Path(COMMON.DEFAULT_SHARED_CACHE_ROOT).resolve()
        and Path(args.output_root).resolve() == DEFAULT_OUTPUT_ROOT.resolve()
        and args.sessions is None
        and args.max_runs is None
        and args.roi_set == "all"
        and args.roi_include is None
        and int(args.max_roi_voxels) == 0
        and int(args.required_status2) == 29
        and not bool(args.allow_smoothed_fallback)
        and not bool(args.include_topology_cache)
        and not bool(args.legacy_process_cache)
        and abs(float(args.condition_fraction) - CONDITION_FRACTION) <= 1e-12
        and int(args.condition_min_trs) == CONDITION_MIN_TRS
        and abs(float(args.nuisance_alpha) - NUISANCE_ALPHA) <= 1e-12
        and int(args.ram_pca_components) == RAM_PCA_COMPONENTS
        and int(args.visual_pca_components) == VISUAL_PCA_COMPONENTS
        and not bool(args.exclude_ram_control)
        and abs(float(args.svm_c) - SVM_C) <= 1e-12
        and int(args.participant_permutations) == PARTICIPANT_PERMUTATIONS
        and int(args.participant_block_size) == PARTICIPANT_BLOCK_SIZE_TR
        and int(args.random_state) == 2026
    )


def validate_prepared_cache(
    run_specs: Sequence[Any],
    cache_audit: Mapping[str, Any],
    roi_names: Sequence[str],
    *,
    official: bool,
) -> None:
    expected = {
        "cache_version": EXPECTED_CACHE_VERSION,
        "process_feature_version": EXPECTED_PROCESS_FEATURE_VERSION,
        "neural_feature_schema_version": EXPECTED_NEURAL_SCHEMA_VERSION,
    }
    observed = {name: int(cache_audit.get(name, -1)) for name in expected}
    if observed != expected:
        raise ValueError(f"CCGP requires canonical v4/process8/schema4 cache: {observed}")
    if any(
        str(run.cache_schema) != "canonical_00brain_nativehrf_schema4_v8"
        for run in run_specs
    ):
        raise ValueError("All runs must use canonical native-time schema4/v8 cache")
    if official:
        if len(run_specs) != EXPECTED_RUNS or len(roi_names) != EXPECTED_ROIS:
            raise ValueError(
                f"Official CCGP requires {EXPECTED_RUNS} runs and {EXPECTED_ROIS} ROIs; "
                f"got {len(run_specs)} and {len(roi_names)}"
            )
        if any(int(run.n_tr) != 480 for run in run_specs):
            raise ValueError("Official CCGP requires 480 analyzed TRs in every run")


def prepare_output_directory(path: Path, *, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace artifacts: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _artifacts_manifest(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    return {name: file_signature(path) for name, path in paths.items()}


def run_subject_analysis(args: argparse.Namespace) -> None:
    official = not bool(args.development)
    if official and not _official_cli_contract(args):
        raise ValueError(
            "Official mode is fixed to the canonical cache, 11 runs, all 110 full-voxel "
            "ROIs, default estimator/null settings, and outputs/06_neural_state_space. "
            "Use --development for a reduced smoke run."
        )
    run_specs, _, cache_audit = COMMON.shared_prepare_from_args(args)
    roi_names = list(COMMON.common_roi_names(run_specs))
    validate_prepared_cache(
        run_specs, cache_audit, roi_names, official=official
    )
    output_root = Path(args.output_root)
    if args.development:
        output_root = output_root / "_development"
    output_dir = output_root / f"sub{int(args.subject):03d}_game{int(args.game)}"
    prepare_output_directory(output_dir, overwrite=bool(args.overwrite))
    result_dir = output_dir / "results"
    figure_dir = output_dir / "figures"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    runs, selection_audit = load_run_designs(
        run_specs,
        fraction_per_condition=float(args.condition_fraction),
        min_per_condition=int(args.condition_min_trs),
    )
    control_folds = precompute_fold_controls(
        runs,
        ram_components=int(args.ram_pca_components),
        visual_components=int(args.visual_pca_components),
        include_ram=not bool(args.exclude_ram_control),
        random_state=int(args.random_state),
    )
    block_orders = generate_synchronized_block_orders(
        [run.transitions.shape[0] for run in runs],
        n_permutations=int(args.participant_permutations),
        block_size=int(args.participant_block_size),
        random_state=int(args.random_state) + 60_000,
    )

    fold_frames: list[pd.DataFrame] = []
    subject_rows: list[dict[str, Any]] = []
    null_parts: list[np.ndarray] = []
    roi_audit_rows: list[dict[str, Any]] = []
    for roi_index, roi_name in enumerate(roi_names):
        patterns_by_run: list[np.ndarray] = []
        reference_coordinates: np.ndarray | None = None
        for run in runs:
            patterns, coordinates = COMMON.load_roi(run.run_spec, roi_name)
            patterns = _as_finite_2d(
                patterns,
                name=f"{run.run_label}:{roi_name}",
                n_rows=run.transitions.shape[0],
            ).astype(np.float32)
            coordinates = np.asarray(coordinates, dtype=np.int32)
            if reference_coordinates is None:
                reference_coordinates = coordinates.copy()
            elif not np.array_equal(reference_coordinates, coordinates):
                raise ValueError(
                    f"ROI voxel coordinates differ across runs for {roi_name}"
                )
            patterns_by_run.append(patterns)
        folds, roi_null = ccgp_loro_roi(
            patterns_by_run,
            runs,
            nuisance_alpha=float(args.nuisance_alpha),
            svm_c=float(args.svm_c),
            control_folds=control_folds,
            block_orders=block_orders,
            block_size=int(args.participant_block_size),
            random_state=int(args.random_state) + 1_003 * roi_index,
        )
        folds.insert(0, "roi", roi_name)
        folds.insert(0, "game", int(args.game))
        folds.insert(0, "subject", int(args.subject))
        fold_frames.append(folds)
        means = folds.groupby("direction", sort=False)["balanced_accuracy"].mean()
        if set(means.index.astype(str)) != set(DIRECTION_NAMES):
            raise ValueError(f"Missing CCGP direction for {roi_name}")
        direction_accuracy = {
            direction: float(means.loc[direction]) for direction in DIRECTION_NAMES
        }
        primary_accuracy = float(np.mean(list(direction_accuracy.values())))
        subject_rows.append(
            {
                "subject": int(args.subject),
                "game": int(args.game),
                "roi": roi_name,
                "balanced_accuracy": primary_accuracy,
                "beta": primary_accuracy - CHANCE,
                f"accuracy_{DIRECTION_NAMES[0]}": direction_accuracy[DIRECTION_NAMES[0]],
                f"accuracy_{DIRECTION_NAMES[1]}": direction_accuracy[DIRECTION_NAMES[1]],
                f"beta_{DIRECTION_NAMES[0]}": direction_accuracy[DIRECTION_NAMES[0]] - CHANCE,
                f"beta_{DIRECTION_NAMES[1]}": direction_accuracy[DIRECTION_NAMES[1]] - CHANCE,
                "n_runs": len(runs),
                "effect_id": "ccgp::switch_stay_across_origin_states_bidirectional_mean",
                "classifier": "LinearSVC",
                "svm_c": float(args.svm_c),
                "official_eligible": bool(official),
            }
        )
        null_parts.append(roi_null)
        roi_audit_rows.append(
            {
                "roi": roi_name,
                "n_voxels": int(patterns_by_run[0].shape[1]),
                "coordinates_identical_across_runs": True,
            }
        )
        log(f"ROI {roi_index + 1}/{len(roi_names)}: {roi_name}")

    fold_effects = pd.concat(fold_frames, ignore_index=True)
    subject_effects = pd.DataFrame(subject_rows)
    participant_null = np.stack(null_parts, axis=0).astype(np.float64)
    primary_inference, direction_inference = participant_inference(
        subject_effects, roi_names=roi_names, null_cube=participant_null
    )
    predefined = predefined_participant_inference(
        primary_inference, roi_names=roi_names, null_cube=participant_null
    )
    expected_fold_rows = len(roi_names) * len(runs) * len(DIRECTIONS)
    if (
        len(fold_effects) != expected_fold_rows
        or fold_effects.duplicated(["roi", "held_out_run", "direction"]).any()
        or len(subject_effects) != len(roi_names)
        or len(primary_inference) != len(roi_names)
        or len(direction_inference) != len(roi_names) * len(DIRECTIONS)
        or participant_null.shape
        != (len(roi_names), len(DIRECTIONS), int(args.participant_permutations))
    ):
        raise AssertionError("Subject CCGP output row/array contract failed")
    if official and len(predefined) != len(NROI.ROI_ORDER):
        raise AssertionError("Official analysis must contain all eight predefined ROIs")

    paths: dict[str, Path] = {
        "transition_selection": result_dir / "transition_selection.csv",
        "roi_alignment_audit": result_dir / "roi_alignment_audit.csv",
        "loro_folds": result_dir / "switch_stay_ccgp_loro_folds.csv.gz",
        "subject_effects": result_dir / "subject_ccgp_effects.csv",
        "participant_primary_inference": result_dir / "participant_ccgp_inference.csv",
        "participant_direction_inference": result_dir / "participant_ccgp_direction_inference.csv",
        "participant_nulls": result_dir / "participant_ccgp_block_permutation_nulls.npz",
    }
    atomic_write_csv(selection_audit, paths["transition_selection"])
    atomic_write_csv(pd.DataFrame(roi_audit_rows), paths["roi_alignment_audit"])
    atomic_write_csv(fold_effects, paths["loro_folds"])
    atomic_write_csv(subject_effects, paths["subject_effects"])
    atomic_write_csv(primary_inference, paths["participant_primary_inference"])
    atomic_write_csv(direction_inference, paths["participant_direction_inference"])
    atomic_savez(
        paths["participant_nulls"],
        null_beta=participant_null,
        primary_mean_null_beta=np.mean(participant_null, axis=1),
        roi_names=np.asarray(roi_names),
        direction_names=np.asarray(DIRECTION_NAMES),
        n_permutations=np.asarray([int(args.participant_permutations)], dtype=np.int32),
        block_size_tr=np.asarray([int(args.participant_block_size)], dtype=np.int32),
    )
    if not predefined.empty:
        paths["predefined_roi_mapping"] = result_dir / "predefined_roi_mapping.csv"
        paths["predefined_roi_inference"] = result_dir / "predefined_roi_ccgp_inference.csv"
        atomic_write_csv(NROI.predefined_roi_mapping_frame(), paths["predefined_roi_mapping"])
        atomic_write_csv(predefined, paths["predefined_roi_inference"])

    paths["participant_brain_map"] = figure_dir / "figure1_participant_ccgp_corrected_brain_map.pdf"
    save_subject_brain_map(
        primary_inference,
        paths["participant_brain_map"],
        subject=int(args.subject),
        game=int(args.game),
    )
    if not predefined.empty:
        paths["predefined_roi_figure"] = figure_dir / "figure2_predefined_roi_ccgp.pdf"
        plot_subject_predefined_ccgp(
            predefined,
            paths["predefined_roi_figure"],
            subject=int(args.subject),
            game=int(args.game),
        )

    manifest = {
        "analysis_name": ANALYSIS_NAME,
        "analysis_version": ANALYSIS_VERSION,
        "mode": "subject_game",
        "subject": int(args.subject),
        "game": int(args.game),
        "official_eligible": bool(official),
        "source_signatures": source_signatures(),
        "run_labels": [run.run_label for run in runs],
        "n_runs": len(runs),
        "roi_names": roi_names,
        "n_rois": len(roi_names),
        "cache_contract": {
            "cache_version": EXPECTED_CACHE_VERSION,
            "process_feature_version": EXPECTED_PROCESS_FEATURE_VERSION,
            "neural_feature_schema_version": EXPECTED_NEURAL_SCHEMA_VERSION,
            "shared_cache_audit": cache_audit,
        },
        "estimand": {
            "name": "switch/stay CCGP across origin states",
            "directions": list(DIRECTION_NAMES),
            "primary": "equal mean of the two crossed-direction held-out balanced accuracies",
            "chance": CHANCE,
            "transition_selection": (
                "balanced disjoint winner-minus-runner-up HRF transition-event TRs; "
                "selection is independent of neural data"
            ),
        },
        "classifier": {
            "estimator": "sklearn LinearSVC",
            "C": float(args.svm_c),
            "max_iter": SVM_MAX_ITER,
            "class_weight": None,
            "voxel_scaling": "outer-training selected TRs only",
            "cross_validation": "leave one run out",
        },
        "controls": {
            "included": [
                "task history",
                "decision-event timing",
                "linear/quadratic drift",
                f"visual outer-training PCA{int(args.visual_pca_components)}",
                (
                    f"RAM outer-training PCA{int(args.ram_pca_components)}"
                    if not args.exclude_ram_control
                    else "RAM excluded"
                ),
            ],
            "current_previous_action": (
                "not residualized because these one-hots algebraically define the "
                "transition endpoints; crossed-origin CCGP is the endpoint-generalization test"
            ),
            "neural_residualization": "outer-training-only fixed ridge with explicit intercept",
            "nuisance_alpha": float(args.nuisance_alpha),
        },
        "participant_inference": {
            "null": (
                f"{int(args.participant_permutations)} synchronized within-run "
                f"{int(args.participant_block_size)}-TR permutations of held-out "
                "prediction time series against fixed transition labels"
            ),
            "prediction_refit_per_null": False,
            "interpretation": "conditional held-out temporal-alignment null for the fitted LORO decoder",
            "tail": "one-sided greater",
            "primary_family_size": len(roi_names),
            "primary_correction": "max statistic over bidirectional mean CCGP parcels",
            "secondary_direction_family_size": 2 * len(roi_names),
            "secondary_correction": "max statistic over direction x parcel cells",
            "claim_gate": "primary corrected positive effect and both observed directions above chance",
            "loro_folds_as_independent_units": False,
        },
        "predefined_roi_analysis": {
            "available": not predefined.empty,
            "role": "literature_defined_targeted_secondary",
            "family_size": len(NROI.ROI_ORDER) if not predefined.empty else 0,
            "mapping_version": NROI.ROI_MAPPING_VERSION,
            "parcel_aggregation": (
                "equal mean of separate left/right unilateral parcel decoder effects; "
                "not a pooled bilateral-voxel decoder"
            ),
        },
        "cpu": {
            "subject_games_per_invocation": 1,
            "roi_execution": "sequential",
            "thread_limit": int(args.cpu_threads),
        },
        "row_contracts": {
            "loro_folds": expected_fold_rows,
            "subject_effects": len(roi_names),
            "participant_primary_inference": len(roi_names),
            "participant_direction_inference": len(roi_names) * len(DIRECTIONS),
            "participant_null_shape": list(participant_null.shape),
            "predefined_roi_inference": len(predefined),
        },
        "artifacts": _artifacts_manifest(paths),
    }
    atomic_write_json(result_dir / "analysis_manifest.json", manifest)
    log(f"Complete -> {output_dir}")


def load_official_subject_effects(
    output_root: Path, *, game: int, subjects: Sequence[int]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    sources: list[dict[str, Any]] = []
    current_sources = source_signatures()
    current_source_hashes = {
        name: str(signature["sha256"])
        for name, signature in current_sources.items()
    }
    for subject in subjects:
        result_dir = Path(output_root) / f"sub{int(subject):03d}_game{int(game)}" / "results"
        effect_path = result_dir / "subject_ccgp_effects.csv"
        manifest_path = result_dir / "analysis_manifest.json"
        if not effect_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"Missing official subject output: {result_dir}")
        manifest_signature = file_signature(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("analysis_name") != ANALYSIS_NAME
            or manifest.get("analysis_version") != ANALYSIS_VERSION
            or manifest.get("official_eligible") is not True
            or int(manifest.get("n_runs", -1)) != EXPECTED_RUNS
            or int(manifest.get("n_rois", -1)) != EXPECTED_ROIS
        ):
            raise ValueError(f"Incompatible subject manifest: {manifest_path}")
        recorded_sources = dict(manifest.get("source_signatures", {}))
        recorded_source_hashes = {
            name: str(dict(recorded_sources.get(name, {})).get("sha256", ""))
            for name in current_source_hashes
        }
        if recorded_source_hashes != current_source_hashes:
            raise ValueError(
                "Subject output was not generated by the current CCGP code and "
                f"helpers: {manifest_path}"
            )
        cache_contract = dict(manifest.get("cache_contract", {}))
        recorded_cache_versions = {
            "cache_version": int(cache_contract.get("cache_version", -1)),
            "process_feature_version": int(
                cache_contract.get("process_feature_version", -1)
            ),
            "neural_feature_schema_version": int(
                cache_contract.get("neural_feature_schema_version", -1)
            ),
        }
        expected_cache_versions = {
            "cache_version": EXPECTED_CACHE_VERSION,
            "process_feature_version": EXPECTED_PROCESS_FEATURE_VERSION,
            "neural_feature_schema_version": EXPECTED_NEURAL_SCHEMA_VERSION,
        }
        if recorded_cache_versions != expected_cache_versions:
            raise ValueError(
                "Subject output does not use the canonical v4/process8/schema4 "
                f"cache contract: {manifest_path}"
            )
        recorded = dict(manifest.get("artifacts", {})).get("subject_effects", {})
        observed = file_signature(effect_path)
        if recorded.get("sha256") != observed["sha256"] or int(recorded.get("size", -1)) != observed["size"]:
            raise ValueError(f"Subject effect artifact changed after manifest: {effect_path}")
        frame = pd.read_csv(effect_path)
        required = {
            "subject",
            "game",
            "roi",
            "balanced_accuracy",
            "beta",
            f"accuracy_{DIRECTION_NAMES[0]}",
            f"accuracy_{DIRECTION_NAMES[1]}",
            f"beta_{DIRECTION_NAMES[0]}",
            f"beta_{DIRECTION_NAMES[1]}",
            "n_runs",
            "official_eligible",
        }
        if required.difference(frame.columns) or len(frame) != EXPECTED_ROIS:
            raise ValueError(f"Subject effect row/schema contract failed: {effect_path}")
        if frame["roi"].astype(str).nunique() != EXPECTED_ROIS:
            raise ValueError(f"Subject effect ROI rows are not unique: {effect_path}")
        if set(pd.to_numeric(frame["subject"], errors="raise").astype(int)) != {int(subject)}:
            raise ValueError(f"Subject identity mismatch: {effect_path}")
        if set(pd.to_numeric(frame["game"], errors="raise").astype(int)) != {int(game)}:
            raise ValueError(f"Game identity mismatch: {effect_path}")
        if not frame["official_eligible"].astype(str).str.lower().eq("true").all():
            raise ValueError(f"Development result cannot enter group inference: {effect_path}")
        if not pd.to_numeric(frame["n_runs"], errors="raise").eq(EXPECTED_RUNS).all():
            raise ValueError(f"Subject effects do not use 11 equal LORO folds: {effect_path}")
        numeric_columns = [
            "subject",
            "game",
            "balanced_accuracy",
            "beta",
            f"accuracy_{DIRECTION_NAMES[0]}",
            f"accuracy_{DIRECTION_NAMES[1]}",
            f"beta_{DIRECTION_NAMES[0]}",
            f"beta_{DIRECTION_NAMES[1]}",
            "n_runs",
        ]
        numeric = frame[numeric_columns].apply(pd.to_numeric, errors="raise")
        if not np.all(np.isfinite(numeric.to_numpy(float))):
            raise ValueError(f"Subject effect contains non-finite values: {effect_path}")
        first_accuracy = numeric[f"accuracy_{DIRECTION_NAMES[0]}"].to_numpy(float)
        second_accuracy = numeric[f"accuracy_{DIRECTION_NAMES[1]}"].to_numpy(float)
        if not (
            np.allclose(numeric["balanced_accuracy"], (first_accuracy + second_accuracy) / 2.0)
            and np.allclose(numeric["beta"], numeric["balanced_accuracy"] - CHANCE)
            and np.allclose(
                numeric[f"beta_{DIRECTION_NAMES[0]}"], first_accuracy - CHANCE
            )
            and np.allclose(
                numeric[f"beta_{DIRECTION_NAMES[1]}"], second_accuracy - CHANCE
            )
        ):
            raise ValueError(f"Subject CCGP estimand identities failed: {effect_path}")
        frames.append(frame)
        sources.append(
            {
                "subject": int(subject),
                "subject_effects": observed,
                "analysis_manifest": manifest_signature,
            }
        )
    result = pd.concat(frames, ignore_index=True)
    roi_sets = [set(result.loc[result["subject"].eq(subject), "roi"].astype(str)) for subject in subjects]
    if any(len(rois) != EXPECTED_ROIS for rois in roi_sets) or any(rois != roi_sets[0] for rois in roi_sets[1:]):
        raise ValueError("The six subject ROI contracts differ")
    return result, sources


def run_group_analysis(args: argparse.Namespace) -> None:
    subjects = tuple(sorted(set(int(value) for value in args.aggregate_subjects)))
    if subjects != EXPECTED_SUBJECTS:
        raise ValueError("Group mode requires --aggregate-subjects 1 2 3 4 5 6")
    if args.development:
        raise ValueError("Development results cannot enter group inference")
    if Path(args.output_root).resolve() != DEFAULT_OUTPUT_ROOT.resolve():
        raise ValueError("Official group output root is fixed to outputs/06_neural_state_space")
    subject_effects, source_artifacts = load_official_subject_effects(
        Path(args.output_root), game=int(args.game), subjects=subjects
    )
    primary, directions = group_inference(subject_effects)
    predefined_subject, predefined = predefined_group_inference(subject_effects)
    output_dir = Path(args.output_root) / f"group_game{int(args.game)}"
    prepare_output_directory(output_dir, overwrite=bool(args.overwrite))
    result_dir = output_dir / "results"
    figure_dir = output_dir / "figures"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {
        "subject_effects_input": result_dir / "subject_ccgp_effects_input.csv",
        "group_primary_inference": result_dir / "group_ccgp_inference.csv",
        "group_direction_inference": result_dir / "group_ccgp_direction_inference.csv",
        "predefined_roi_mapping": result_dir / "predefined_roi_mapping.csv",
        "predefined_roi_subject_effects": result_dir / "predefined_roi_subject_effects.csv",
        "predefined_roi_inference": result_dir / "predefined_roi_ccgp_inference.csv",
        "group_brain_map": figure_dir / "figure1_group_ccgp_corrected_brain_map.pdf",
        "predefined_roi_figure": figure_dir / "figure2_predefined_roi_ccgp.pdf",
        "predefined_roi_anatomy": figure_dir / "figure3_predefined_roi_anatomy.pdf",
    }
    atomic_write_csv(subject_effects, paths["subject_effects_input"])
    atomic_write_csv(primary, paths["group_primary_inference"])
    atomic_write_csv(directions, paths["group_direction_inference"])
    atomic_write_csv(NROI.predefined_roi_mapping_frame(), paths["predefined_roi_mapping"])
    atomic_write_csv(predefined_subject, paths["predefined_roi_subject_effects"])
    atomic_write_csv(predefined, paths["predefined_roi_inference"])
    save_group_brain_map(
        primary, paths["group_brain_map"], game=int(args.game)
    )
    plot_group_predefined_ccgp(
        predefined_subject,
        predefined,
        paths["predefined_roi_figure"],
        game=int(args.game),
    )
    NROI.save_predefined_roi_anatomy(paths["predefined_roi_anatomy"])
    manifest = {
        "analysis_name": ANALYSIS_NAME,
        "analysis_version": ANALYSIS_VERSION,
        "mode": "group_game",
        "game": int(args.game),
        "subjects": list(subjects),
        "source_signatures": source_signatures(),
        "source_artifacts": source_artifacts,
        "primary_inference": {
            "unit": "participant",
            "estimand": "bidirectional mean switch/stay CCGP accuracy minus .50",
            "sign_patterns": 64,
            "tail": "two-sided",
            "correction": "exact synchronized max-|t|",
            "family_size": EXPECTED_ROIS,
            "claim_gate": "corrected positive primary effect and both direction group means above chance",
        },
        "secondary_direction_inference": {
            "sign_patterns": 64,
            "tail": "two-sided",
            "correction": "exact synchronized max-|t|",
            "family_size": len(DIRECTION_NAMES) * EXPECTED_ROIS,
        },
        "predefined_roi_analysis": {
            "role": "literature_defined_targeted_secondary",
            "sign_patterns": 64,
            "tail": "two-sided",
            "correction": "exact synchronized max-|t|",
            "family_size": len(NROI.ROI_ORDER),
            "mapping_version": NROI.ROI_MAPPING_VERSION,
            "parcel_aggregation": (
                "equal mean of separate left/right unilateral parcel decoder effects; "
                "not a pooled bilateral-voxel decoder"
            ),
        },
        "no_across_game_average": True,
        "row_contracts": {
            "subject_effects_input": len(EXPECTED_SUBJECTS) * EXPECTED_ROIS,
            "group_primary_inference": EXPECTED_ROIS,
            "group_direction_inference": len(DIRECTION_NAMES) * EXPECTED_ROIS,
            "predefined_roi_subject_effects": len(EXPECTED_SUBJECTS) * len(NROI.ROI_ORDER),
            "predefined_roi_inference": len(NROI.ROI_ORDER),
        },
        "artifacts": _artifacts_manifest(paths),
    }
    atomic_write_json(result_dir / "analysis_manifest.json", manifest)
    log(f"Complete -> {output_dir}")


def _synthetic_selection(n_per_condition: int) -> ConditionSelection:
    indices = {
        name: np.arange(index * n_per_condition, (index + 1) * n_per_condition)
        for index, name in enumerate(TRANSITION_NAMES)
    }
    scores = {name: np.ones(n_per_condition) for name in TRANSITION_NAMES}
    return ConditionSelection(indices, scores, n_per_condition)


def _synthetic_ccgp_patterns(
    *,
    abstract_switch: bool,
    random_state: int,
    n_runs: int = 5,
    n_per_condition: int = 24,
) -> tuple[list[np.ndarray], list[ConditionSelection], list[str]]:
    rng = np.random.default_rng(random_state)
    selections = [_synthetic_selection(n_per_condition) for _ in range(n_runs)]
    patterns: list[np.ndarray] = []
    for _ in range(n_runs):
        x = rng.normal(0, 0.35, size=(4 * n_per_condition, 8))
        for condition_index, condition in enumerate(TRANSITION_NAMES):
            selected = selections[0].indices[condition]
            is_switch = condition in {"N_to_A", "A_to_N"}
            origin_active = condition in {"A_to_N", "A_to_A"}
            if abstract_switch:
                x[selected, 0] += 2.5 if is_switch else -2.5
                x[selected, 1] += 0.25 if origin_active else -0.25
            else:
                # Origin-specific coding reverses the switch axis across contexts,
                # so within-context classification is easy but CCGP fails.
                signed = 2.5 if is_switch else -2.5
                x[selected, 0] += signed if not origin_active else -signed
        patterns.append(x)
    return patterns, selections, [f"run{index + 1}" for index in range(n_runs)]


def _simple_ccgp_accuracy(
    patterns: Sequence[np.ndarray],
    selections: Sequence[ConditionSelection],
    *,
    random_state: int,
) -> float:
    scores = []
    for held_out in range(len(patterns)):
        train_runs = [index for index in range(len(patterns)) if index != held_out]
        for direction_index, (_, train_conditions, test_conditions) in enumerate(DIRECTIONS):
            train_x, train_y = _stack_selected(
                patterns, selections, train_runs, train_conditions
            )
            prediction = fit_linear_svm_predict_all(
                train_x,
                train_y,
                patterns[held_out],
                svm_c=SVM_C,
                random_state=random_state + held_out * 3 + direction_index,
            )
            indices, labels = _selected_test_labels(
                selections[held_out], test_conditions
            )
            scores.append(balanced_accuracy(labels, prediction[indices]))
    return float(np.mean(scores))


def run_self_test(random_state: int = 2026) -> dict[str, Any]:
    events = np.zeros((160, 4), dtype=float)
    for index in range(4):
        events[index * 40 : (index + 1) * 40, index] = 1.0
    selection = select_fourway_condition_trs(
        events, fraction_per_condition=0.08, min_per_condition=8
    )
    assert selection.n_per_condition == 13
    abstract = _synthetic_ccgp_patterns(
        abstract_switch=True, random_state=random_state
    )
    contextual = _synthetic_ccgp_patterns(
        abstract_switch=False, random_state=random_state + 1
    )
    abstract_accuracy = _simple_ccgp_accuracy(
        abstract[0], abstract[1], random_state=random_state
    )
    contextual_accuracy = _simple_ccgp_accuracy(
        contextual[0], contextual[1], random_state=random_state
    )
    if abstract_accuracy < 0.95 or contextual_accuracy > 0.15:
        raise AssertionError(
            f"Synthetic CCGP separation failed: {abstract_accuracy}, {contextual_accuracy}"
        )

    rng = np.random.default_rng(random_state + 77)
    prediction = rng.integers(0, 2, size=120)
    indices = np.sort(rng.choice(120, size=40, replace=False))
    labels = np.tile((0, 1), 20)
    orders = generate_synchronized_block_orders(
        [120], n_permutations=29, block_size=20, random_state=random_state
    )[0]
    optimized = block_permuted_balanced_accuracy(
        prediction, indices, labels, orders, block_size=20
    )
    brute = []
    blocks = prediction.reshape(6, 20)
    for order in orders:
        permuted = blocks[order].reshape(-1)
        brute.append(balanced_accuracy(labels, permuted[indices]))
    if not np.allclose(optimized, brute):
        raise AssertionError("Block-permutation score-table optimization is incorrect")

    group_values = rng.normal(0.04, 0.02, size=(6, 8))
    group = exact_group_maxT(
        group_values,
        cell_names=[f"roi{index}" for index in range(8)],
        p_column="p_maxT",
    )
    if len(group) != 8 or not group["p_maxT"].between(0, 1).all():
        raise AssertionError("Exact group inference contract failed")
    result = {
        "selection_per_condition": selection.n_per_condition,
        "abstract_switch_ccgp_accuracy": abstract_accuracy,
        "origin_specific_ccgp_accuracy": contextual_accuracy,
        "block_permutation_equivalence": True,
        "exact_group_rows": len(group),
        "classifier": "LinearSVC",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    COMMON.add_shared_cli_arguments(parser, require_subject_game=False)
    parser.set_defaults(roi_set="all")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--aggregate-subjects", nargs="+", type=int, default=None)
    parser.add_argument("--condition-fraction", type=float, default=CONDITION_FRACTION)
    parser.add_argument("--condition-min-trs", type=int, default=CONDITION_MIN_TRS)
    parser.add_argument("--nuisance-alpha", type=float, default=NUISANCE_ALPHA)
    parser.add_argument("--ram-pca-components", type=int, default=RAM_PCA_COMPONENTS)
    parser.add_argument("--visual-pca-components", type=int, default=VISUAL_PCA_COMPONENTS)
    parser.add_argument("--exclude-ram-control", action="store_true")
    parser.add_argument("--svm-c", type=float, default=SVM_C)
    parser.add_argument(
        "--participant-permutations", type=int, default=PARTICIPANT_PERMUTATIONS
    )
    parser.add_argument(
        "--participant-block-size", type=int, default=PARTICIPANT_BLOCK_SIZE_TR
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=1,
        help="Maximum BLAS/OpenMP threads; ROIs and subject-games remain sequential.",
    )
    parser.add_argument(
        "--development",
        action="store_true",
        help="Allow reduced runs/ROIs/permutations and write under _development.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--render-predefined-roi-anatomy",
        action="store_true",
        help="Render only the fixed eight-ROI anatomy PDF for --game.",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser


def validate_cli(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if int(args.cpu_threads) < 1:
        parser.error("--cpu-threads must be >=1")
    if not 0.0 < float(args.condition_fraction) <= 0.25:
        parser.error("--condition-fraction must lie in (0,.25]")
    if int(args.condition_min_trs) < 4:
        parser.error("--condition-min-trs must be >=4")
    if float(args.nuisance_alpha) <= 0 or float(args.svm_c) <= 0:
        parser.error("--nuisance-alpha and --svm-c must be positive")
    if int(args.ram_pca_components) < 0 or int(args.visual_pca_components) < 0:
        parser.error("PCA component counts cannot be negative")
    minimum_permutations = 19 if args.development else PARTICIPANT_PERMUTATIONS
    if int(args.participant_permutations) < minimum_permutations:
        parser.error(f"--participant-permutations must be >= {minimum_permutations}")
    if int(args.participant_block_size) <= 0:
        parser.error("--participant-block-size must be positive")
    if args.aggregate_subjects is not None:
        if args.subject is not None or args.game not in (1, 2):
            parser.error("Group mode requires --game and omits --subject")
    elif not args.render_predefined_roi_anatomy and not args.self_test:
        if args.subject is None or args.game not in (1, 2):
            parser.error("Subject mode requires exactly one --subject and --game")


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    validate_cli(args, parser)
    if args.self_test:
        run_self_test(int(args.random_state))
        return
    if args.render_predefined_roi_anatomy:
        if args.game not in (1, 2):
            parser.error("--render-predefined-roi-anatomy requires --game")
        path = (
            Path(args.output_root)
            / f"group_game{int(args.game)}"
            / "figures"
            / "figure3_predefined_roi_anatomy.pdf"
        )
        NROI.save_predefined_roi_anatomy(path)
        log(f"Predefined ROI anatomy -> {path}")
        return
    with threadpool_limits(limits=int(args.cpu_threads)):
        if args.aggregate_subjects is not None:
            run_group_analysis(args)
        else:
            run_subject_analysis(args)


if __name__ == "__main__":
    main()
