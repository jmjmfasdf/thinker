#!/usr/bin/env python3
"""Pure helpers for controlled run-wise neural localization GLMs.

The module deliberately knows nothing about filesystem/cache layout.  It
accepts already HRF-aligned process blocks and returns one target-contrast
weight vector per run.  Multiplying a weight vector by a TR x voxel BOLD matrix
gives the run-wise beta map.  This keeps preprocessing identical across ROIs
and makes the subject/group aggregation auditable.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.decomposition import PCA


EPS = 1e-12


@dataclass(frozen=True)
class LocalizationTarget:
    name: str
    target_block: str
    target_column: int
    upstream_blocks: Tuple[str, ...]
    interpretation: str


CONTINUOUS_TARGETS: Tuple[LocalizationTarget, ...] = (
    LocalizationTarget(
        "model_margin", "model_margin", 0, ("model_prior_policy",),
        "pre-search current-action model margin",
    ),
    LocalizationTarget(
        "initial_q_margin", "initial_q_state", 0,
        ("model_prior_policy", "model_margin"),
        "initial root-Q incumbent margin",
    ),
    LocalizationTarget(
        "revision_magnitude", "revision_primary", 0,
        ("model_prior_policy", "model_margin", "initial_q_state"),
        "window mean repeated-visit absolute Q revision",
    ),
    LocalizationTarget(
        "rollout_evidence", "evidence_primary", 0,
        ("model_prior_policy", "model_margin", "initial_q_state", "revision_primary"),
        "window mean rollout evidence",
    ),
    LocalizationTarget(
        "search_breadth", "geometry_primary", 0,
        ("model_prior_policy", "model_margin", "initial_q_state", "revision_primary", "evidence_primary"),
        "root-branch breadth",
    ),
    LocalizationTarget(
        "search_depth", "geometry_primary", 1,
        ("model_prior_policy", "model_margin", "initial_q_state", "revision_primary", "evidence_primary"),
        "mean bout-terminal depth",
    ),
    LocalizationTarget(
        "actor_residual", "actor_readout", 1,
        (
            "model_prior_policy", "model_margin", "initial_q_state",
            "revision_primary", "evidence_primary", "geometry_primary",
        ),
        "terminal Actor margin residual beyond model prior",
    ),
)


def _as_2d(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError(f"Block {name!r} must be a non-empty 2D array")
    return array


def _finite_standardize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    mean = np.nanmean(array, axis=0)
    array = np.where(np.isfinite(array), array, mean[None, :])
    mean = np.mean(array, axis=0)
    scale = np.std(array, axis=0)
    scale = np.where(scale > EPS, scale, 1.0)
    return (array - mean) / scale


def _pca_control(values: np.ndarray, n_components: int) -> np.ndarray:
    array = _finite_standardize(values)
    n_take = min(int(n_components), array.shape[0] - 1, array.shape[1])
    if n_take <= 0:
        return np.empty((len(array), 0), dtype=float)
    return PCA(n_components=n_take, svd_solver="full").fit_transform(array)


def _append_block(
    columns: List[np.ndarray], names: List[str], blocks: Mapping[str, np.ndarray], block: str
) -> None:
    if block not in blocks:
        raise KeyError(f"Required HRF block is absent: {block}")
    array = _as_2d(blocks[block], block)
    for index in range(array.shape[1]):
        columns.append(array[:, index])
        names.append(f"{block}[{index}]")


def _ordered_estimable_design(
    columns: Sequence[np.ndarray], names: Sequence[str], target_name: str,
    relative_tolerance: float = 1e-10,
) -> Tuple[np.ndarray, List[str], int, List[str]]:
    """Retain target first and remove only later exactly redundant nuisance."""
    if len(columns) != len(names) or not columns:
        raise ValueError("Design columns/names are empty or mismatched")
    raw = np.column_stack(columns).astype(float)
    if not np.all(np.isfinite(raw)):
        raise ValueError("Localization design contains non-finite values")
    target_index = list(names).index(target_name)

    # Target-first pruning alone can hide non-identifiability by dropping a
    # nuisance column when the target lies exactly in the joint nuisance span.
    # Test that full span first with the same deterministic two-pass MGS rule.
    nuisance_basis: List[np.ndarray] = []
    for index in range(len(names)):
        if index == target_index:
            continue
        vector = raw[:, index]
        norm = float(np.linalg.norm(vector))
        if norm <= EPS:
            continue
        residual = vector / norm
        for _ in range(2):
            for direction in nuisance_basis:
                residual -= direction * float(np.dot(direction, residual))
        residual_norm = float(np.linalg.norm(residual))
        if residual_norm > relative_tolerance:
            nuisance_basis.append(residual / residual_norm)
    target = raw[:, target_index]
    target_norm = float(np.linalg.norm(target))
    if target_norm <= EPS:
        raise ValueError(f"Target has zero variance: {target_name}")
    target_residual = target / target_norm
    for _ in range(2):
        for direction in nuisance_basis:
            target_residual -= direction * float(np.dot(direction, target_residual))
    if float(np.linalg.norm(target_residual)) <= relative_tolerance:
        raise ValueError(
            f"Target is aliased with the full nuisance span: {target_name}"
        )

    order = [target_index] + [index for index in range(len(names)) if index != target_index]
    basis: List[np.ndarray] = []
    retained_indices: List[int] = []
    dropped: List[str] = []
    for index in order:
        vector = raw[:, index]
        norm = float(np.linalg.norm(vector))
        if norm <= EPS:
            if index == target_index:
                raise ValueError(f"Target has zero variance: {target_name}")
            dropped.append(str(names[index]))
            continue
        residual = vector / norm
        for _ in range(2):
            for direction in basis:
                residual -= direction * float(np.dot(direction, residual))
        residual_norm = float(np.linalg.norm(residual))
        if residual_norm <= relative_tolerance:
            if index == target_index:
                raise ValueError(f"Target is not estimable: {target_name}")
            dropped.append(str(names[index]))
            continue
        basis.append(residual / residual_norm)
        retained_indices.append(index)
    retained_indices.sort()
    design = raw[:, retained_indices]
    retained_names = [str(names[index]) for index in retained_indices]
    return design, retained_names, retained_names.index(target_name), dropped


def _contrast_weight(
    columns: Sequence[np.ndarray], names: Sequence[str], target_name: str
) -> Tuple[np.ndarray, Dict[str, object]]:
    design, retained, target_index, dropped = _ordered_estimable_design(
        columns, names, target_name
    )
    inverse = np.linalg.pinv(design, rcond=1e-10)
    weight = inverse[target_index]
    return weight, {
        "n_tr": int(design.shape[0]),
        "design_columns": int(design.shape[1]),
        "target_design_column": int(target_index),
        "retained_columns": retained,
        "dropped_redundant_columns": dropped,
        "target_weight_norm": float(np.linalg.norm(weight)),
    }


def _base_controls(
    blocks: Mapping[str, np.ndarray], visual: np.ndarray, ram: np.ndarray,
    *, visual_pca_dim: int, ram_pca_dim: int, include_ram: bool,
    include_action: bool, include_eligibility: bool,
) -> Tuple[List[np.ndarray], List[str]]:
    visual_scores = _pca_control(visual, visual_pca_dim)
    columns: List[np.ndarray] = [np.ones(len(visual_scores), dtype=float)]
    names: List[str] = ["intercept"]
    for index in range(visual_scores.shape[1]):
        columns.append(visual_scores[:, index]); names.append(f"visual_pc[{index}]")
    if include_ram:
        ram_scores = _pca_control(ram, ram_pca_dim)
        for index in range(ram_scores.shape[1]):
            columns.append(ram_scores[:, index]); names.append(f"ram_pc[{index}]")
    for block in ("task_history", "drift"):
        if block in blocks:
            _append_block(columns, names, blocks, block)
    if include_eligibility and "eligibility" in blocks:
        # Columns 0:8 are true availability/completeness indicators in the
        # canonical v7 contract. Columns 8:12 are semantic NOOP/active risk
        # sets and must never enter a localization nuisance model.
        eligibility = _as_2d(blocks["eligibility"], "eligibility")
        for index in range(min(8, eligibility.shape[1])):
            columns.append(eligibility[:, index])
            names.append(f"eligibility[{index}]")
    if include_action:
        if "decision_event" in blocks:
            _append_block(columns, names, blocks, "decision_event")
        for block in ("previous_action", "current_action"):
            _append_block(columns, names, blocks, block)
    return columns, names


def build_localization_weights(
    blocks: Mapping[str, np.ndarray], visual: np.ndarray, ram: np.ndarray,
    *, visual_pca_dim: int = 20, ram_pca_dim: int = 20,
    include_ram: bool = False, tiers: Sequence[str] = ("marginal", "partial"),
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, object]]]:
    """Build run-wise target beta weights for all prespecified localization models."""
    n_rows = len(_as_2d(visual, "visual"))
    if len(_as_2d(ram, "ram")) != n_rows:
        raise ValueError("RAM/visual row mismatch")
    for name, values in blocks.items():
        if len(_as_2d(values, name)) != n_rows:
            raise ValueError(f"TR-row mismatch for block {name}")
    # PCA controls are a run-level nuisance representation, not a
    # target-specific fit.  Compute them once so every target uses exactly the
    # same sensory/state basis and avoid repeating the expensive decomposition.
    base_action_columns, base_action_names = _base_controls(
        blocks, visual, ram, visual_pca_dim=visual_pca_dim,
        ram_pca_dim=ram_pca_dim, include_ram=include_ram,
        include_action=True, include_eligibility=True,
    )
    base_transition_columns, base_transition_names = _base_controls(
        blocks, visual, ram, visual_pca_dim=visual_pca_dim,
        ram_pca_dim=ram_pca_dim, include_ram=include_ram,
        # Eligibility contains transition-risk flags tied to NOOP/active
        # status on an un-HRF time base. Including those in a transition GLM
        # would partly condition on the target with mismatched timing.
        include_action=False, include_eligibility=False,
    )
    weights: Dict[str, np.ndarray] = {}
    audit: Dict[str, Dict[str, object]] = {}
    for spec in CONTINUOUS_TARGETS:
        target_block = _as_2d(blocks[spec.target_block], spec.target_block)
        if spec.target_column >= target_block.shape[1]:
            raise ValueError(f"Target column missing: {spec.name}")
        target_values = _finite_standardize(target_block[:, [spec.target_column]])[:, 0]
        for tier in tiers:
            if tier not in {"marginal", "partial"}:
                raise ValueError(f"Unknown localization tier: {tier}")
            columns = list(base_action_columns)
            names = list(base_action_names)
            if tier == "partial":
                for block in spec.upstream_blocks:
                    _append_block(columns, names, blocks, block)
            target_name = f"target::{spec.name}"
            columns.insert(0, target_values); names.insert(0, target_name)
            model_name = f"{tier}::{spec.name}"
            weights[model_name], qa = _contrast_weight(columns, names, target_name)
            qa.update(
                {
                    "tier": tier, "target": spec.name,
                    "interpretation": spec.interpretation,
                    "include_ram": bool(include_ram),
                }
            )
            audit[model_name] = qa

    # Transition status-switch interaction after origin/destination main
    # effects and total HRF decision-event mass. The latter separates the 2x2
    # condition contrast from irregular decision-event density across TRs.
    events = _as_2d(blocks["transition_events"], "transition_events")
    if events.shape[1] < 4:
        raise ValueError("transition_events requires N_N,N_A,A_N,A_A")
    transition = _as_2d(blocks["transition_factors"], "transition_factors")
    if transition.shape[1] < 3:
        raise ValueError("transition_factors requires origin,destination,switch")
    columns = list(base_transition_columns)
    names = list(base_transition_names)
    columns.append(np.sum(events[:, :4], axis=1))
    names.append("decision_event_mass")
    for index, label in enumerate(("origin_active", "destination_active")):
        columns.append(transition[:, index]); names.append(label)
    switch_name = "target::switch_status_interaction"
    columns.insert(0, transition[:, 2]); names.insert(0, switch_name)
    weights["partial::switch_status_interaction"], qa = _contrast_weight(
        columns, names, switch_name
    )
    qa.update(
        {
            "tier": "partial", "target": "switch_status_interaction",
            "interpretation": (
                "NOOP/active status-switch interaction beyond origin and "
                "destination main effects; invariant cross-context coding is "
                "tested separately by CCGP in 06"
            ),
            "include_ram": bool(include_ram),
        }
    )
    audit["partial::switch_status_interaction"] = qa

    # Direct N->A versus A->N endpoint contrast, explicitly composite/exploratory.
    direction = events[:, 1] - events[:, 2]
    columns = list(base_transition_columns)
    names = list(base_transition_names)
    for values, label in (
        (events[:, 1] + events[:, 2], "transition_event_mass"),
        (events[:, 0], "N_to_N"), (events[:, 3], "A_to_A"),
    ):
        columns.append(values); names.append(label)
    direction_name = "target::transition_direction"
    columns.insert(0, direction); names.insert(0, direction_name)
    weights["exploratory::transition_direction"], qa = _contrast_weight(
        columns, names, direction_name
    )
    qa.update(
        {
            "tier": "exploratory", "target": "transition_direction",
            "interpretation": "N->A minus A->N ordered endpoint/motor composite",
            "include_ram": bool(include_ram),
        }
    )
    audit["exploratory::transition_direction"] = qa
    return weights, audit


def apply_localization_weight(weight: np.ndarray, bold: np.ndarray) -> np.ndarray:
    values = _as_2d(bold, "bold")
    vector = np.asarray(weight, dtype=float)
    if len(vector) != len(values):
        raise ValueError("Localization weight/BOLD TR mismatch")
    return (vector @ values).astype(np.float32)


def exact_synchronized_sign_flip(
    matrix: np.ndarray, *, chunk_columns: int = 20_000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != 6 or not np.all(np.isfinite(values)):
        raise ValueError("Group localization requires a finite 6-subject matrix")
    def studentized_from_moments(mean: np.ndarray, sum_squares: np.ndarray) -> np.ndarray:
        variance = np.maximum((sum_squares - 6.0 * mean**2) / 5.0, 0.0)
        sd = np.sqrt(variance)
        denominator = sd / np.sqrt(6)
        statistic = np.zeros_like(mean, dtype=float)
        regular = denominator > EPS
        statistic[regular] = mean[regular] / denominator[regular]
        degenerate = (~regular) & (np.abs(mean) > EPS)
        statistic[degenerate] = np.sign(mean[degenerate]) * np.inf
        return statistic

    observed_mean = np.mean(values, axis=0)
    sum_squares = np.sum(values**2, axis=0)
    observed_t = studentized_from_moments(observed_mean, sum_squares)
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=6)), dtype=float)
    raw = np.empty(values.shape[1], dtype=float)
    maximum = np.zeros(len(signs), dtype=float)
    take = max(1, int(chunk_columns))
    for start in range(0, values.shape[1], take):
        stop = min(start + take, values.shape[1])
        permuted_mean = (signs @ values[:, start:stop]) / 6.0
        statistic = studentized_from_moments(
            permuted_mean, sum_squares[None, start:stop]
        )
        absolute = np.abs(statistic)
        raw[start:stop] = np.mean(
            absolute >= np.abs(observed_t[None, start:stop]), axis=0
        )
        maximum = np.maximum(maximum, np.max(absolute, axis=1))
    max_t = np.mean(maximum[:, None] >= np.abs(observed_t)[None, :], axis=0)
    return observed_t, raw, max_t


def self_test() -> Dict[str, float]:
    rng = np.random.default_rng(808)
    n_tr, n_voxels = 240, 8
    blocks = {
        "model_prior_policy": rng.normal(size=(n_tr, 5)),
        "model_margin": rng.normal(size=(n_tr, 1)),
        "initial_q_state": rng.normal(size=(n_tr, 1)),
        "revision_primary": rng.normal(size=(n_tr, 1)),
        "evidence_primary": rng.normal(size=(n_tr, 4)),
        "geometry_primary": rng.normal(size=(n_tr, 2)),
        "actor_readout": rng.normal(size=(n_tr, 3)),
        "transition_factors": rng.normal(size=(n_tr, 3)),
        "transition_events": rng.normal(size=(n_tr, 4)),
        "decision_event": rng.normal(size=(n_tr, 1)),
        "previous_action": rng.normal(size=(n_tr, 6)),
        "current_action": rng.normal(size=(n_tr, 6)),
        "task_history": rng.normal(size=(n_tr, 5)),
        "drift": np.column_stack((np.linspace(-1, 1, n_tr), np.linspace(-1, 1, n_tr) ** 2)),
        "eligibility": np.ones((n_tr, 1)),
    }
    visual = rng.normal(size=(n_tr, 30)); ram = rng.normal(size=(n_tr, 20))
    weights, qa = build_localization_weights(
        blocks, visual, ram, visual_pca_dim=5, ram_pca_dim=4, include_ram=True
    )
    switch_audit = qa["partial::switch_status_interaction"]
    if "decision_event_mass" not in switch_audit["retained_columns"]:
        raise AssertionError("Transition interaction lacks decision-event mass")
    if any(
        str(name).startswith("eligibility[")
        for name in switch_audit["retained_columns"]
    ):
        raise AssertionError("Transition interaction retained semantic eligibility")
    target = _finite_standardize(blocks["revision_primary"])[:, 0]
    bold = target[:, None] * np.linspace(.2, 1.0, n_voxels)[None, :] + rng.normal(
        scale=.25, size=(n_tr, n_voxels)
    )
    beta = apply_localization_weight(weights["marginal::revision_magnitude"], bold)
    if not np.all(beta > 0.1):
        raise AssertionError("Injected localization beta was not recovered")
    confound = _finite_standardize(blocks["model_margin"])[:, 0]
    blocks_confounded = dict(blocks)
    blocks_confounded["revision_primary"] = confound[:, None]
    try:
        build_localization_weights(
            blocks_confounded, visual, ram, visual_pca_dim=5, include_ram=False,
            tiers=("partial",),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Aliased partial target was not rejected")
    matrix = np.abs(rng.normal(size=(6, 5))) + .1
    _, raw, corrected = exact_synchronized_sign_flip(matrix)
    if not np.all(corrected >= raw) or not np.all(raw >= 1 / 64):
        raise AssertionError("Exact synchronized group inference failed")
    return {
        "n_models": float(len(weights)),
        "minimum_injected_beta": float(np.min(beta)),
        "minimum_raw_p": float(np.min(raw)),
        "minimum_max_t_p": float(np.min(corrected)),
        "mean_design_columns": float(np.mean([row["design_columns"] for row in qa.values()])),
    }
