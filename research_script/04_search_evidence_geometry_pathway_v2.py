#!/usr/bin/env python3
"""Transition-aligned 02 -> 03 -> 04 pathway analysis.

This script consumes the canonical shared analysis cache directly.  It asks a
more tightly ordered question:

    pre-search M/P -> event E/signed dQ -> destination-aligned dG
        -> terminal Actor residual -> human N<->A transition.

Subject mode estimates three stages.  Stage 1 uses strict k>j suffixes and
within-window fixed effects.  Stage 2 constructs early-to-late, action-aligned
window summaries without using the eventual human destination.  Stage 3 uses
leave-one-run-out nested probabilistic models B0..B3 and held-out log loss.
Group mode uses exactly six subjects and exact synchronized sign flips, with
separate max-|t| families for the three scientific stages.

By default, Actor OOF predictions are refit from the shared cache and visual
controls are rebuilt/cached locally from the behavioral NPZ files.  No neural
cache and no accepted-04 output artifact is required.  The accepted 04 output
can be selected explicitly only as a numerical compatibility sensitivity.
New outputs are written under ``outputs/04_search_evidence_geometry_pathway_v2``
so the frozen 04 SHA referenced by existing analyses 06--08 is never altered.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from shared_analysis_cache import (
    DEFAULT_SHARED_CACHE_ROOT,
    cache_paths,
    validate_manifest,
    validate_one_to_one_real_steps,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
EXPECTED_ACCEPTED_04_SHA256 = (
    "4537caa171d060c07fa16ee8c8292437f20b9970a996c7413f49f53530a88e29"
)
DEFAULT_BEHAV_ROOT = ROOT / "behavioral_data_block"
DEFAULT_BASE_OUTPUT_ROOT = (
    SCRIPT_DIR / "outputs" / "04_search_evidence_geometry_pathway"
)
DEFAULT_OUTPUT_ROOT = (
    SCRIPT_DIR / "outputs" / "04_search_evidence_geometry_pathway_v2"
)
SCHEMA_VERSION = 2
ANALYSIS_VERSION = "2026-09-14-transition-aligned-pathway-v2-dynamic"
DERIVED_ACTOR_CACHE_VERSION = 2
ACTOR_SOFTMAX_MAXITER = 1000
STAGE3_MIN_TRAIN_ROWS = 400
DERIVED_ACTOR_IMPLEMENTATION_SPEC = (
    "v2-local action-long terminal Actor LORO: P/A1/A2/A3/A4 soft-target "
    "multinomial models; train-fold-only imputation/scaling; fixed L2; 11 runs; "
    f"L-BFGS-B maxiter={ACTOR_SOFTMAX_MAXITER}"
)
DERIVED_ACTOR_IMPLEMENTATION_HASH = hashlib.sha256(
    DERIVED_ACTOR_IMPLEMENTATION_SPEC.encode("utf-8")
).hexdigest()
KEYS = ["run_label", "real_step", "real_step_id"]
N_ACTIONS = 6
N_EVENTS = 29
MIN_FUTURE = 1
EPS = 1e-12
RANDOM_STATE = 2026
VISUAL_DESCRIPTOR_DEFINITION = (
    "current pre-action target-frame 4x4 pooled grayscale (16); target-minus-source "
    "absolute 4x4 pooled grayscale motion (16); target RGB mean/std (6); target "
    "horizontal/vertical edge energies (2); target+1 and reward[target] prohibited"
)
VISUAL_DESCRIPTOR_HASH = hashlib.sha256(
    VISUAL_DESCRIPTOR_DEFINITION.encode("utf-8")
).hexdigest()

INK = "#33363A"
GRID = "#D9DDE3"
BLUE = "#386D9C"
BLUE_LIGHT = "#BFD2E3"
ORANGE = "#CF7B35"
ORANGE_LIGHT = "#F0CEAF"
PURPLE = "#7B5EA7"
PURPLE_LIGHT = "#D8CCE8"
GREY = "#8A8F96"
OPEN = "#F7F8FA"
TRANSITION_LABELS = {
    "noop_to_active": "NOOP -> Active",
    "active_to_noop": "Active -> NOOP",
}
OUTCOME_LABELS = {
    "future_destination_share_contrast": "Future destination-share contrast",
    "future_destination_depth_score": "Future destination-weighted depth",
}
PREDICTOR_LABELS = {
    "aligned_E": "Aligned rollout evidence E",
    "aligned_signed_q": "Aligned signed Q update",
    "directed_q": "03 directed Q update (exact-action)",
}
STAGE3_LABELS = {
    "B0": "B0\nM/R + state",
    "B1": "B1\n+ E/dQ",
    "B2": "B2\n+ dG",
    "B3": "B3\n+ Actor",
}

STAGE1_PRIMARY_IDS = tuple(
    f"stage1::{transition}::{predictor}::{outcome}"
    for transition in ("noop_to_active", "active_to_noop")
    for predictor in ("aligned_E", "aligned_signed_q")
    for outcome in (
        "future_destination_share_contrast",
        "future_destination_depth_score",
    )
)
STAGE2_PRIMARY_IDS = tuple(
    f"stage2::{transition}::{metric}"
    for transition in ("noop_to_active", "active_to_noop")
    for metric in ("destination_share_shift", "destination_depth_shift")
)
STAGE1_MODERATION_IDS = tuple(
    f"stage1_moderation::{transition}::{interaction}::{outcome}"
    for transition in ("noop_to_active", "active_to_noop")
    for interaction in ("M_x_E", "M_x_signed_q", "M_x_directed_q")
    for outcome in (
        "future_destination_share_contrast",
        "future_destination_depth_score",
    )
)
STAGE1_SENSITIVITY_IDS = tuple(
    f"stage1_sensitivity::{transition}::directed_q::{outcome}"
    for transition in ("noop_to_active", "active_to_noop")
    for outcome in (
        "future_destination_share_contrast",
        "future_destination_depth_score",
    )
)
STAGE3_PRIMARY_IDS = tuple(
    f"stage3::{transition}::{increment}"
    for transition in ("noop_to_active", "active_to_noop")
    for increment in ("B0_to_B1", "B1_to_B2", "B2_to_B3")
)

CHART_CONTRACTS = {
    "stage1_micro_dynamics": {
        "question": "How do aligned E, signed/directed dQ, and their interaction with M relate to strictly future destination geometry?",
        "chart": "small-multiple quantile relationship lines plus low/high-M moderation curves",
        "grain": "run lines in subject figures; subject lines in group figures",
        "reason": "The scientific object is a continuous within-window relationship, not coefficient ranking.",
    },
    "stage2_reallocation": {
        "question": "Does search allocation move toward the transition destination from early to late search?",
        "chart": "paired-change slope plot",
        "grain": "one line per run or subject",
        "reason": "The estimand is a within-unit early-to-late change.",
    },
    "stage3_nested_prediction": {
        "question": "Which ordered feature block lowers held-out transition log loss?",
        "chart": "staged held-out loss trajectory",
        "grain": "one line per held-out run or subject",
        "reason": "The model stages are ordered and evaluated on identical held-out rows.",
    },
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)

def read_columns(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    available = pd.read_csv(path, nrows=0).columns.tolist()
    missing = sorted(set(columns).difference(available))
    if missing:
        raise KeyError(f"{path} lacks required columns: {missing}")
    return pd.read_csv(path, usecols=list(columns), low_memory=False)


def standardize(values: pd.Series, *, constant_zero: bool = True) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce").astype(float)
    result = pd.Series(np.nan, index=x.index, dtype=float)
    finite = np.isfinite(x)
    if not finite.any():
        return result
    scale = float(x.loc[finite].std(ddof=0))
    if not np.isfinite(scale) or scale <= EPS:
        if constant_zero:
            result.loc[finite] = 0.0
        return result
    result.loc[finite] = (x.loc[finite] - float(x.loc[finite].mean())) / scale
    return result


def behavior_path(
    root: Path, subject: int, session: int, block: int, game: int
) -> Path:
    path = (
        Path(root)
        / f"sub-{subject:03d}"
        / f"ses-{session:02d}"
        / f"sub{subject:03d}-ses{session:02d}-block{block}-game{game}.npz"
    )
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def probabilities_to_margin(probabilities: np.ndarray, choices: np.ndarray) -> np.ndarray:
    output = np.full(len(choices), np.nan, dtype=float)
    for index, choice in enumerate(np.asarray(choices, dtype=int)):
        if not 0 <= choice < probabilities.shape[1]:
            continue
        row = probabilities[index]
        if not np.all(np.isfinite(row)):
            continue
        other = float(np.max(np.delete(row, choice)))
        output[index] = math.log(max(float(row[choice]), EPS)) - math.log(max(other, EPS))
    return output


def load_subject_cache(
    subject: int, game: int, cache_root: Path
) -> Tuple[object, Dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray], Dict[str, object]]:
    """Load only shared-v2 tables; preserve actor rows as the authoritative spine."""
    global N_ACTIONS
    paths = cache_paths(cache_root, subject, game)
    manifest = validate_manifest(paths, subject=subject, game=game)
    N_ACTIONS = int(manifest.get("num_actions", manifest.get("n_actions", 6)))
    actor_columns = KEYS + [
        "subject", "session", "block", "game", "chunk",
        "previous_human_action", "human_action", "previous_reward", "previous_done",
        "noop_run_length_prev", "actor_policy_source_real_step", "thinker_action",
        "actor_current_action_margin_terminal", "actor_current_action_coordinate_valid",
    ]
    native_columns = ["episode_id", "episode_start"] if manifest.get("source_format") == "native_zarr_v2" else []
    actor = read_columns(paths.actor_metrics, actor_columns + native_columns)
    model_columns = KEYS + [
        "source_chunk", "incumbent_action", "action_before_incumbent", "human_action", "switch",
        "bout_age", "prior_reward", "prior_done", "run_progress", "root_value",
        "model_candidate", "strongest_competitor_margin", "switch_target_uncertainty",
        "n_status2", "initial_actor_policy_complete", "terminal_actor_policy_complete",
    ] + [
        name
        for action in range(N_ACTIONS)
        for name in (f"model_root_p_{action}", f"actor_terminal_p_{action}")
    ]
    model = read_columns(paths.model_control_windows, model_columns)
    geometry_columns = KEYS + [
        "root_branch_allocation_diversity", "mean_bout_terminal_depth",
        "deep_event_fraction", "deep_bout_fraction", "path_revisit_fraction",
        "root_bout_branch_diversity", "reset_count", "n_bouts",
        "value_revision_event_fraction",
    ] + [
        name
        for action in range(N_ACTIONS)
        for name in (
            f"root_q_initial_{action}", f"root_q_final_{action}",
            f"root_n_initial_{action}", f"root_n_final_{action}",
        )
    ]
    geometry = read_columns(paths.persistence_windows, geometry_columns)
    event_columns = KEYS + [
        "status2_position", "bout_index", "branch", "depth", "rollout_return",
        "root_q_before_sampled", "root_q_after_sampled", "root_n_before_sampled",
        "root_n_after_sampled", "root_q_revision", "rollout_prediction_error",
        "directed_value_revision", "value_revision_valid", "return_gain_from_root",
    ]
    events = read_columns(paths.persistence_events, event_columns)
    rollout_keys = read_columns(paths.rollout_windows, KEYS).reset_index(names="window_index")
    prefix_names = ("window_index", "event_index", "return_advantage", "pair_valid")
    with np.load(paths.rollout_prefix_events, allow_pickle=False) as archive:
        missing = sorted(set(prefix_names).difference(archive.files))
        if missing:
            raise KeyError(f"prefix cache lacks {missing}")
        prefix = {name: archive[name] for name in prefix_names}

    audit = validate_one_to_one_real_steps(
        {"actor_backbone": actor, "model": model, "geometry": geometry, "rollout": rollout_keys},
        KEYS,
    )
    if actor.duplicated(KEYS).any():
        raise RuntimeError("Actor backbone is not one row per real step")
    actor = actor.reset_index(drop=True).reset_index(names="actor_row")
    windows = actor.merge(model, on=KEYS, how="left", validate="one_to_one", suffixes=("_actor", ""))
    # Actor is the full-row authority; model has no row for each run's initial spine row.
    for column in ("subject", "session", "block", "game", "human_action"):
        actor_column = f"{column}_actor"
        if actor_column not in windows:
            continue
        left = pd.to_numeric(windows[actor_column], errors="coerce")
        right = pd.to_numeric(windows[column], errors="coerce")
        overlap = left.notna() & right.notna()
        if not np.allclose(left[overlap], right[overlap], equal_nan=True):
            raise RuntimeError(f"Actor/model {column} disagree on overlapping rows")
        windows[column] = left.where(left.notna(), right)
        windows = windows.drop(columns=actor_column)
    windows = windows.merge(geometry, on=KEYS, how="left", validate="one_to_one")
    if manifest.get("source_format") == "native_zarr_v2":
        windows.attrs["native_visual_controls"] = str(paths.visual_controls)
        windows.attrs["native_visual_alignment"] = str(paths.visual_alignment)
    if len(windows) != len(actor) or not windows["actor_row"].eq(np.arange(len(actor))).all():
        raise RuntimeError("Full actor backbone row count/order was not preserved")
    return paths, manifest, windows, events, rollout_keys, actor, prefix, audit


def attach_event_evidence(
    events: pd.DataFrame,
    rollout_keys: pd.DataFrame,
    prefix: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    """Align cached event evidence to persistence events without rebuilding traces."""
    evidence = pd.DataFrame({name: np.asarray(prefix[name]) for name in
        ("window_index", "event_index", "return_advantage", "pair_valid")}).rename(
            columns={"event_index": "status2_position", "pair_valid": "rollout_pair_valid"})
    frame = events.merge(rollout_keys, on=KEYS, how="left", validate="many_to_one")
    if frame["window_index"].isna().any():
        raise RuntimeError("Persistence events do not align to rollout windows")
    position = pd.to_numeric(frame["status2_position"], errors="coerce").to_numpy(dtype=int)
    if np.any(position < 0):
        raise RuntimeError("Negative status-2 position")
    frame = frame.merge(evidence, on=["window_index", "status2_position"], how="left", validate="one_to_one")
    q = pd.to_numeric(frame["root_q_revision"], errors="coerce")
    repeated = (
        pd.to_numeric(frame["value_revision_valid"], errors="coerce").eq(1)
        & np.isfinite(q)
        & pd.to_numeric(frame["root_n_before_sampled"], errors="coerce").gt(0)
    )
    frame["absolute_q_revision"] = np.where(repeated, q.abs(), np.nan)
    frame["log_absolute_q_revision"] = np.where(
        repeated, np.log(np.maximum(q.abs(), EPS)), np.nan
    )
    frame["valid_repeated_q_revision"] = repeated.astype(int)
    length = frame.groupby(KEYS)["status2_position"].transform("max") + 1
    frame["event_progress"] = frame["status2_position"] / (length - 1).clip(lower=1)
    return frame


def _residualize_window_fixed_effects(
    data: pd.DataFrame, columns: Sequence[str]
) -> pd.DataFrame:
    result = pd.DataFrame(index=data.index)
    groups = data["window_id"]
    for column in columns:
        numeric = pd.to_numeric(data[column], errors="coerce")
        result[column] = numeric - numeric.groupby(groups, sort=False).transform("mean")
    return result


def _ordered_estimable_event_design(
    design: pd.DataFrame,
    focal: str,
    *,
    relative_tolerance: float = 1e-10,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Preserve focal and design span; remove only later aliased nuisance columns.

    Priority is focal, position fixed effects, branch fixed effects, then remaining
    nuisance in its prespecified order.  Each L2-normalized candidate is twice
    re-orthogonalized against the retained basis.  A non-estimable focal is a hard
    error; only later nuisance can be removed.
    """
    if focal not in design.columns:
        raise RuntimeError(f"Protected focal is absent from event design: {focal}")
    original = list(design.columns)
    priority = (
        [focal]
        + [column for column in original if column.startswith("position_")]
        + [column for column in original if column.startswith("branch_")]
        + [
            column for column in original
            if column != focal
            and not column.startswith("position_")
            and not column.startswith("branch_")
        ]
    )

    def independent_subset(columns: Sequence[str]) -> Tuple[List[str], List[str], np.ndarray]:
        retained: List[str] = []
        dropped: List[str] = []
        basis = np.empty((len(design), 0), dtype=float)
        for column in columns:
            vector = design[column].to_numpy(dtype=float)
            if not np.all(np.isfinite(vector)):
                raise RuntimeError(f"Non-finite event-design column: {column}")
            norm = float(np.linalg.norm(vector))
            if not np.isfinite(norm) or norm <= EPS:
                if column == focal:
                    raise RuntimeError("Protected focal has zero within-window variance")
                dropped.append(column)
                continue
            residual = vector / norm
            for _ in range(2):
                if basis.shape[1]:
                    residual = residual - basis @ (basis.T @ residual)
            relative_residual = float(np.linalg.norm(residual))
            if relative_residual <= relative_tolerance:
                if column == focal:
                    raise RuntimeError("Protected focal is not estimable")
                dropped.append(column)
                continue
            basis = np.column_stack((basis, residual / relative_residual))
            retained.append(column)
        return retained, dropped, basis

    retained, dropped, _ = independent_subset(priority)
    nuisance_priority = [column for column in priority if column != focal]
    _, _, nuisance_basis = independent_subset(nuisance_priority)
    focal_vector = design[focal].to_numpy(dtype=float)
    focal_unit = focal_vector / float(np.linalg.norm(focal_vector))
    focal_residual = focal_unit.copy()
    for _ in range(2):
        if nuisance_basis.shape[1]:
            focal_residual = focal_residual - nuisance_basis @ (
                nuisance_basis.T @ focal_residual
            )
    focal_residual_ratio = float(np.linalg.norm(focal_residual))
    focal_estimable = bool(focal_residual_ratio > relative_tolerance)
    if not focal_estimable:
        raise RuntimeError(
            "Protected focal is aliased with the full nuisance span; adjusted effect is not estimable"
        )
    diagnostics: Dict[str, object] = {
        "design_columns_before": int(len(original)),
        "design_columns_after": int(len(retained)),
        "design_rank_diagnostic": int(len(retained)),
        "n_dropped_collinear_nuisance": int(len(dropped)),
        "dropped_collinear_nuisance": "|".join(dropped),
        "focal_estimable_full_nuisance_span": focal_estimable,
        "focal_residual_ratio_after_nuisance": focal_residual_ratio,
        "nuisance_span_policy": (
            "focal_then_position_FE_then_branch_FE_then_prespecified_nuisance; "
            "two-pass_normalized_MGS_rtol_1e-10"
        ),
    }
    return design.loc[:, retained], diagnostics


def normalize_policy_rows(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(matrix, dtype=float)
    total = np.sum(values, axis=1)
    valid = (
        np.all(np.isfinite(values), axis=1)
        & np.all(values >= 0, axis=1)
        & (total > EPS)
    )
    normalized = np.full_like(values, np.nan)
    normalized[valid] = values[valid] / total[valid, None]
    return normalized, valid


def terminal_action_features(
    windows: pd.DataFrame, event_frame: pd.DataFrame
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """Action-long P, value V, and geometry G; no future human-action features."""
    base = windows[KEYS + ["actor_row"]].rename(columns={"actor_row": "window_index"})
    event = event_frame.drop(columns=["window_index"], errors="ignore").merge(
        base, on=KEYS, how="left", validate="many_to_one"
    )
    event = event.loc[event["window_index"].notna() & event["branch"].between(0, N_ACTIONS-1)].copy()
    event["depth_numeric"] = pd.to_numeric(event["depth"], errors="coerce")
    event["rollout_numeric"] = pd.to_numeric(event["rollout_return"], errors="coerce")
    grouped = (
        event.groupby(["window_index", "branch"], sort=False, observed=True)
        .agg(
            event_count=("status2_position", "size"),
            mean_rollout=("rollout_numeric", "mean"),
            max_rollout=("rollout_numeric", "max"),
            deep_fraction=("depth_numeric", lambda value: float(np.mean(value >= 3))),
        )
        .reset_index()
    )
    terminal = (
        event.sort_values(["window_index", "bout_index", "status2_position"], kind="stable")
        .groupby(["window_index", "bout_index"], sort=False, observed=True)
        .tail(1)
        .groupby(["window_index", "branch"], sort=False, observed=True)["depth_numeric"]
        .mean()
        .rename("branch_mean_bout_terminal_depth")
        .reset_index()
    )
    grouped = grouped.merge(
        terminal, on=["window_index", "branch"], how="left", validate="one_to_one"
    )
    shape = (len(base), N_ACTIONS)
    matrices = {
        name: np.zeros(shape, dtype=float)
        for name in (
            "terminal_mean_rollout_gain", "terminal_max_rollout_gain",
            "branch_event_share", "branch_mean_bout_terminal_depth",
            "branch_deep_event_fraction", "branch_visited",
        )
    }
    wi = grouped["window_index"].to_numpy(dtype=int)
    branch = grouped["branch"].to_numpy(dtype=int)
    root = pd.to_numeric(windows["root_value"], errors="coerce").to_numpy(dtype=float)
    matrices["terminal_mean_rollout_gain"][wi, branch] = (
        grouped["mean_rollout"].to_numpy(dtype=float) - root[wi]
    )
    matrices["terminal_max_rollout_gain"][wi, branch] = (
        grouped["max_rollout"].to_numpy(dtype=float) - root[wi]
    )
    matrices["branch_event_share"][wi, branch] = (
        grouped["event_count"].to_numpy(dtype=float)
        / windows["n_status2"].to_numpy(dtype=float)[grouped["window_index"].to_numpy(dtype=int)].clip(min=1)
    )
    matrices["branch_mean_bout_terminal_depth"][wi, branch] = grouped[
        "branch_mean_bout_terminal_depth"
    ].fillna(0).to_numpy(dtype=float)
    matrices["branch_deep_event_fraction"][wi, branch] = grouped[
        "deep_fraction"
    ].fillna(0).to_numpy(dtype=float)
    matrices["branch_visited"][wi, branch] = 1.0
    terminal_q = windows[
        [f"root_q_final_{action}" for action in range(N_ACTIONS)]
    ].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    prior_raw = windows[
        [f"model_root_p_{action}" for action in range(N_ACTIONS)]
    ].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    target_raw = windows[
        [f"actor_terminal_p_{action}" for action in range(N_ACTIONS)]
    ].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    prior, prior_valid = normalize_policy_rows(prior_raw)
    target, target_valid = normalize_policy_rows(target_raw)
    features = {
        "log_model_prior": np.log(np.clip(prior, EPS, 1.0)),
        "terminal_root_q": np.nan_to_num(terminal_q, nan=0.0),
        **matrices,
    }
    complete = prior_valid & target_valid & windows["source_chunk"].notna().to_numpy()
    qa = {
        "rows": int(len(windows)),
        "eligible_rows": int(np.sum(complete)),
        "prior_policy_coverage": float(prior_valid.mean()),
        "terminal_actor_policy_coverage": float(target_valid.mean()),
        "geometry_action_specific": True,
        "geometry_primary": ["branch_event_share", "branch_mean_bout_terminal_depth"],
        "geometry_secondary": ["branch_deep_event_fraction"],
        "missing_branch_encoding": (
            "zero-valued summaries plus explicit branch_visited; "
            "branch_event_share is the prespecified normalized visit count"
        ),
        "value_block": [
            "terminal_root_q", "terminal_mean_rollout_gain", "terminal_max_rollout_gain"
        ],
    }
    return features, prior, target, complete, qa


def fit_softmax_soft_targets(
    features: Mapping[str, np.ndarray],
    target: np.ndarray,
    train_mask: np.ndarray,
    *,
    max_train: int = 60_000,
) -> Optional[Dict[str, object]]:
    """Train-fold-only scale/impute and penalized action-long softmax fit."""
    names = list(features)
    raw = np.stack([np.asarray(features[name], dtype=float) for name in names], axis=2)
    valid = np.asarray(train_mask, dtype=bool) & np.all(np.isfinite(target), axis=1)
    index = np.flatnonzero(valid)
    if len(index) < 500:
        return None
    if len(index) > max_train:
        index = index[np.linspace(0, len(index) - 1, max_train, dtype=int)]
    x = raw[index].copy()
    y = target[index]
    impute = np.zeros(x.shape[2], dtype=float)
    for column in range(x.shape[2]):
        finite = x[:, :, column][np.isfinite(x[:, :, column])]
        impute[column] = float(np.median(finite)) if len(finite) else 0.0
        x[:, :, column] = np.where(
            np.isfinite(x[:, :, column]), x[:, :, column], impute[column]
        )
    center = np.mean(x, axis=(0, 1))
    scale = np.std(x, axis=(0, 1))
    scale = np.where(scale > EPS, scale, 1.0)
    x = (x - center[None, None, :]) / scale[None, None, :]
    action_fixed = np.broadcast_to(
        np.eye(N_ACTIONS)[None, :, 1:], (len(index), N_ACTIONS, N_ACTIONS - 1)
    )
    design = np.concatenate([x, action_fixed], axis=2)

    def objective(beta: np.ndarray) -> Tuple[float, np.ndarray]:
        eta = np.einsum("nap,p->na", design, beta)
        log_probability = eta - logsumexp(eta, axis=1, keepdims=True)
        probability = np.exp(log_probability)
        penalty = 5e-5 * float(np.sum(beta[: len(names)] ** 2))
        loss = -float(np.sum(y * log_probability)) + penalty
        gradient = -np.sum(
            np.einsum("na,nap->nap", y - probability, design), axis=(0, 1)
        )
        gradient[: len(names)] += 1e-4 * beta[: len(names)]
        return loss, gradient

    fit = minimize(
        objective, np.zeros(design.shape[2]), jac=True, method="L-BFGS-B",
        options={"maxiter": ACTOR_SOFTMAX_MAXITER},
    )
    if not fit.success or not np.all(np.isfinite(fit.x)):
        log(f"Actor optimizer failed: maxiter={ACTOR_SOFTMAX_MAXITER} "
            f"nit={fit.nit} status={fit.status} message={fit.message} "
            f"finite_coefficients={bool(np.all(np.isfinite(fit.x)))} features={names}")
        return None
    return {
        "names": names, "beta": fit.x, "center": center, "scale": scale,
        "impute": impute, "n_train_available": int(np.sum(valid)),
        "n_train_used": int(len(index)), "objective": float(fit.fun),
    }


def predict_softmax(
    fit: Mapping[str, object],
    features: Mapping[str, np.ndarray],
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    names = list(fit["names"])
    raw = np.stack([np.asarray(features[name], dtype=float) for name in names], axis=2)
    index = np.flatnonzero(mask)
    values = raw[index].copy()
    impute = np.asarray(fit["impute"], dtype=float)
    for column in range(values.shape[2]):
        values[:, :, column] = np.where(
            np.isfinite(values[:, :, column]), values[:, :, column], impute[column]
        )
    center = np.asarray(fit["center"], dtype=float)
    scale = np.asarray(fit["scale"], dtype=float)
    values = (values - center[None, None, :]) / scale[None, None, :]
    action_fixed = np.broadcast_to(
        np.eye(N_ACTIONS)[None, :, 1:], (len(index), N_ACTIONS, N_ACTIONS - 1)
    )
    design = np.concatenate([values, action_fixed], axis=2)
    eta = np.einsum("nap,p->na", design, np.asarray(fit["beta"], dtype=float))
    return index, np.exp(eta - logsumexp(eta, axis=1, keepdims=True))


def soft_cross_entropy(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return -np.sum(target * np.log(np.clip(prediction, EPS, 1.0)), axis=1)


def fit_actor_geometry_loro(
    windows: pd.DataFrame, event_frame: pd.DataFrame, subject: int, game: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    features, prior, target, complete, qa = terminal_action_features(windows, event_frame)
    feature_sets = {
        "P": ["log_model_prior"],
        "A1_P_plus_G": [
            "log_model_prior", "branch_event_share",
            "branch_mean_bout_terminal_depth", "branch_visited",
        ],
        "A2_P_plus_V": [
            "log_model_prior", "terminal_root_q", "terminal_mean_rollout_gain",
            "terminal_max_rollout_gain", "branch_visited",
        ],
        "A3_P_plus_V_plus_G": [
            "log_model_prior", "terminal_root_q", "terminal_mean_rollout_gain",
            "terminal_max_rollout_gain", "branch_event_share",
            "branch_mean_bout_terminal_depth", "branch_visited",
        ],
        "A4_secondary_plus_deep": [
            "log_model_prior", "terminal_root_q", "terminal_mean_rollout_gain",
            "terminal_max_rollout_gain", "branch_event_share",
            "branch_mean_bout_terminal_depth", "branch_deep_event_fraction",
            "branch_visited",
        ],
    }
    run_array = windows["run_label"].astype(str).to_numpy()
    incumbent = pd.to_numeric(windows["incumbent_action"], errors="coerce").fillna(-1).to_numpy(dtype=int)
    predictions = pd.DataFrame(
        {
            "actor_row": windows["actor_row"].to_numpy(dtype=int),
            "actor_loro_eligible": complete.astype(int),
            "actor_loro_oof_complete": 0,
            "actor_actual_incumbent_margin": probabilities_to_margin(target, incumbent),
            "actor_p_only_predicted_incumbent_margin": np.nan,
            "actor_residual_beyond_model_prior": np.nan,
            "actor_geometry_predicted_margin_increment": np.nan,
        }
    )
    for model_code in ("P", "A2", "A3"):
        for action in range(N_ACTIONS):
            predictions[f"actor_{model_code}_oof_p_{action}"] = np.nan
    fold_rows: List[Dict[str, object]] = []
    coefficient_rows: List[Dict[str, object]] = []
    for run_label in windows["run_label"].drop_duplicates():
        test = (run_array == str(run_label)) & complete
        train = (run_array != str(run_label)) & complete
        fits: Dict[str, Mapping[str, object]] = {}
        outputs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for label, names in feature_sets.items():
            subset = {name: features[name] for name in names}
            fit = fit_softmax_soft_targets(subset, target, train)
            if fit is None:
                log(f"Actor LORO failed: subject={subject:03d} game={game} "
                    f"held_out_run={run_label} model={label}")
                break
            fits[label] = fit
            outputs[label] = predict_softmax(fit, subset, test)
        if len(fits) != len(feature_sets):
            continue
        indices = [output[0] for output in outputs.values()]
        if not all(np.array_equal(indices[0], value) for value in indices[1:]):
            raise RuntimeError("Actor LORO model test rows differ")
        index = indices[0]
        if len(index) == 0:
            continue
        ce = {
            label: soft_cross_entropy(target[index], output[1])
            for label, output in outputs.items()
        }
        p_margin = probabilities_to_margin(outputs["P"][1], incumbent[index])
        a2_margin = probabilities_to_margin(outputs["A2_P_plus_V"][1], incumbent[index])
        a3_margin = probabilities_to_margin(outputs["A3_P_plus_V_plus_G"][1], incumbent[index])
        predictions.loc[index, "actor_p_only_predicted_incumbent_margin"] = p_margin
        predictions.loc[index, "actor_residual_beyond_model_prior"] = (
            predictions.loc[index, "actor_actual_incumbent_margin"].to_numpy(dtype=float) - p_margin
        )
        predictions.loc[index, "actor_geometry_predicted_margin_increment"] = a3_margin - a2_margin
        for action in range(N_ACTIONS):
            predictions.loc[index, f"actor_P_oof_p_{action}"] = outputs["P"][1][:, action]
            predictions.loc[index, f"actor_A2_oof_p_{action}"] = outputs[
                "A2_P_plus_V"
            ][1][:, action]
            predictions.loc[index, f"actor_A3_oof_p_{action}"] = outputs[
                "A3_P_plus_V_plus_G"
            ][1][:, action]
        predictions.loc[index, "actor_loro_oof_complete"] = 1
        for model_name, fit in fits.items():
            names = list(fit["names"])
            beta = np.asarray(fit["beta"], dtype=float)
            center = np.asarray(fit["center"], dtype=float)
            scale = np.asarray(fit["scale"], dtype=float)
            impute = np.asarray(fit["impute"], dtype=float)
            for feature_index, feature in enumerate(names):
                coefficient_rows.append(
                    {
                        "subject": subject, "game": game, "run_label": str(run_label),
                        "model": model_name, "feature": feature,
                        "standardized_beta": float(beta[feature_index]),
                        "train_center": float(center[feature_index]),
                        "train_scale": float(scale[feature_index]),
                        "train_impute": float(impute[feature_index]),
                    }
                )
            for action in range(1, N_ACTIONS):
                coefficient_rows.append(
                    {
                        "subject": subject, "game": game, "run_label": str(run_label),
                        "model": model_name, "feature": f"action_intercept_{action}",
                        "standardized_beta": float(beta[len(names) + action - 1]),
                        "train_center": np.nan, "train_scale": np.nan,
                        "train_impute": np.nan,
                    }
                )
        fold_rows.append(
            {
                "subject": subject, "game": game, "run_label": str(run_label),
                "effect_id": "actor::geometry_delta_ce",
                "beta": float(np.mean(ce["A2_P_plus_V"] - ce["A3_P_plus_V_plus_G"])),
                "actor_P_ce": float(np.mean(ce["P"])),
                "actor_A1_ce": float(np.mean(ce["A1_P_plus_G"])),
                "actor_A2_ce": float(np.mean(ce["A2_P_plus_V"])),
                "actor_A3_ce": float(np.mean(ce["A3_P_plus_V_plus_G"])),
                "actor_A4_ce": float(np.mean(ce["A4_secondary_plus_deep"])),
                "secondary_deep_delta_ce": float(
                    np.mean(ce["A3_P_plus_V_plus_G"] - ce["A4_secondary_plus_deep"])
                ),
                "secondary_P_to_PG_delta_ce": float(
                    np.mean(ce["P"] - ce["A1_P_plus_G"])
                ),
                "n_test": int(len(index)),
                "n_train_available": int(fits["A3_P_plus_V_plus_G"]["n_train_available"]),
                "n_train_used": int(fits["A3_P_plus_V_plus_G"]["n_train_used"]),
                "primary": 1,
            }
        )
    folds = pd.DataFrame(fold_rows)
    expected = int(windows.loc[complete, "run_label"].nunique())
    if len(folds) != expected:
        raise RuntimeError(f"Actor LORO produced {len(folds)} of {expected} folds")
    if not predictions.loc[complete, "actor_loro_oof_complete"].eq(1).all():
        raise RuntimeError("Actor OOF predictions do not cover eligible rows exactly once")
    qa["loro_folds"] = int(len(folds))
    qa["actor_softmax_maxiter"] = ACTOR_SOFTMAX_MAXITER
    qa["train_fold_only_preprocessing"] = True
    qa["actor_target_is_terminal_real_head"] = True
    return folds, predictions, pd.DataFrame(coefficient_rows), qa


def _action_indices(action: np.ndarray) -> np.ndarray:
    values = np.asarray(action)
    if values.ndim == 1:
        return values.astype(int)
    flat = values.reshape(values.shape[0], -1)
    valid = np.any(np.isfinite(flat), axis=1) & (
        np.max(np.abs(np.nan_to_num(flat, nan=0.0)), axis=1) > EPS
    )
    result = np.full(len(flat), -1, dtype=int)
    result[valid] = np.argmax(np.nan_to_num(flat[valid], nan=0.0), axis=1)
    return result


def _pooled_4x4(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"Expected N x H x W grayscale frames, found {array.shape}")
    height = array.shape[1] - array.shape[1] % 4
    width = array.shape[2] - array.shape[2] % 4
    array = array[:, :height, :width]
    return array.reshape(
        len(array), 4, height // 4, 4, width // 4
    ).mean(axis=(2, 4)).reshape(len(array), 16)


def low_level_visual_descriptor(
    current_images: np.ndarray, past_images: np.ndarray
) -> np.ndarray:
    """Forty current-pre-action/past controls; no future/outcome pixels are used."""
    current = np.asarray(current_images, dtype=np.float32)[..., :3] / 255.0
    past = np.asarray(past_images, dtype=np.float32)[..., :3] / 255.0
    if current.shape != past.shape or current.ndim != 4 or current.shape[-1] != 3:
        raise ValueError(f"Unexpected image batches: {current.shape}, {past.shape}")
    weight = np.asarray([0.2989, 0.5870, 0.1140], dtype=np.float32)
    gray = np.einsum("nhwc,c->nhw", current, weight)
    gray_past = np.einsum("nhwc,c->nhw", past, weight)
    pooled_gray = _pooled_4x4(gray)
    pooled_motion = _pooled_4x4(np.abs(gray - gray_past))
    rgb_mean = current.mean(axis=(1, 2))
    rgb_sd = current.std(axis=(1, 2))
    horizontal_edge = np.abs(np.diff(gray, axis=2)).mean(axis=(1, 2))
    vertical_edge = np.abs(np.diff(gray, axis=1)).mean(axis=(1, 2))
    descriptor = np.column_stack(
        [
            pooled_gray, pooled_motion, rgb_mean, rgb_sd,
            horizontal_edge, vertical_edge,
        ]
    ).astype(np.float32)
    if descriptor.shape[1] != 40 or not np.all(np.isfinite(descriptor)):
        raise RuntimeError(f"Invalid visual descriptor matrix: {descriptor.shape}")
    return descriptor


def _actor_key_hash(windows: pd.DataFrame) -> str:
    dependency_columns = KEYS + [
        "source_chunk", "session", "block", "incumbent_action", "human_action"
    ]
    hashed = pd.util.hash_pandas_object(
        windows[dependency_columns], index=False
    ).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def _behavior_source_signature(
    windows: pd.DataFrame, behavior_root: Path, subject: int, game: int
) -> Tuple[str, List[Dict[str, object]]]:
    entries: List[Dict[str, object]] = []
    for run_label, run in windows.groupby("run_label", sort=False):
        eligible = run.loc[run["source_chunk"].notna()]
        if eligible.empty:
            continue
        path = behavior_path(
            behavior_root, subject, int(eligible["session"].iloc[0]),
            int(eligible["block"].iloc[0]), game,
        )
        stat = path.stat()
        entries.append(
            {
                "run_label": str(run_label), "path": str(path.resolve()),
                "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    serialized = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest(), entries


def build_visual_state_controls(
    windows: pd.DataFrame,
    *,
    subject: int,
    game: int,
    behavior_root: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, object]]:
    """Map cached windows to current pre-action behavioral images exactly.

    ``target=real_step+1+3*source_chunk``. Dataset semantics establish
    image[target] as the current observation before executing action[target].
    Descriptor current=target and past=source=target-1; target+1 is prohibited.
    """
    if "native_visual_controls" in windows.attrs:
        with np.load(windows.attrs["native_visual_controls"], allow_pickle=False) as archive:
            descriptor = np.asarray(archive["descriptor"], dtype=np.float32)
        alignment = pd.read_csv(windows.attrs["native_visual_alignment"])
        if descriptor.shape != (len(windows), 40) or len(alignment) != len(windows):
            raise ValueError("Native visual controls do not match actor spine")
        if not alignment[KEYS].equals(windows[KEYS].reset_index(drop=True)):
            raise ValueError("Native visual-control keys differ from actor spine")
        eligible = windows["source_chunk"].notna().to_numpy()
        if not np.isfinite(descriptor[eligible]).all():
            raise ValueError("Missing native visual controls on eligible rows")
        qa = {"source": "native_zarr_real_anchors", "descriptor_features": 40,
              "current_is_pre_action": True, "past_is_previous_real_observation": True,
              "episode_boundaries_crossed": 0, "behavior_npz_image_used": False,
              "eligible_rows": int(eligible.sum()), "mapped_rows": int(eligible.sum()),
              "mapping_coverage": 1.0, "cache_reused": True}
        (output_dir / "qa").mkdir(parents=True, exist_ok=True)
        atomic_json(output_dir / "qa" / "visual_state_controls.json", qa)
        return descriptor, alignment.loc[eligible].reset_index(drop=True), qa
    cache_dir = output_dir / "cache"
    table_dir = output_dir / "tables"
    qa_dir = output_dir / "qa"
    cache_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "visual_state_controls_v2.npz"
    alignment_path = table_dir / "behavior_frame_alignment.csv.gz"
    qa_path = qa_dir / "visual_state_controls.json"
    key_hash = _actor_key_hash(windows)
    resolved_root = str(Path(behavior_root).resolve())
    behavior_signature, behavior_sources = _behavior_source_signature(
        windows, behavior_root, subject, game
    )
    if cache_path.exists() and alignment_path.exists() and qa_path.exists() and not overwrite:
        with np.load(cache_path, allow_pickle=False) as archive:
            descriptor = np.asarray(archive["descriptor"], dtype=np.float32)
            stored_hash = str(np.asarray(archive["actor_key_hash"]).item())
            stored_root = str(np.asarray(archive["behavior_root"]).item())
            stored_behavior_signature = str(
                np.asarray(archive["behavior_source_signature"]).item()
            ) if "behavior_source_signature" in archive.files else ""
            stored_formula_hash = str(
                np.asarray(archive["descriptor_formula_hash"]).item()
            ) if "descriptor_formula_hash" in archive.files else ""
        if (
            descriptor.shape == (len(windows), 40)
            and stored_hash == key_hash
            and stored_root == resolved_root
            and stored_behavior_signature == behavior_signature
            and stored_formula_hash == VISUAL_DESCRIPTOR_HASH
        ):
            alignment = pd.read_csv(alignment_path, low_memory=False)
            qa = json.loads(qa_path.read_text(encoding="utf-8"))
            eligible_rows = windows.loc[windows["source_chunk"].notna(), "actor_row"].to_numpy(dtype=int)
            aligned_rows = pd.to_numeric(alignment["actor_row"], errors="raise").to_numpy(dtype=int)
            if (
                int(qa.get("descriptor_version", -1)) != 2
                or str(qa.get("actor_key_hash", "")) != key_hash
                or alignment["actor_row"].duplicated().any()
                or not np.array_equal(np.sort(aligned_rows), np.sort(eligible_rows))
            ):
                raise RuntimeError(f"Invalid cached visual alignment metadata at {cache_path}")
            qa["cache_reused"] = True
            return descriptor, alignment, qa
        raise RuntimeError(
            f"Stale visual-control cache at {cache_path}; rerun with --overwrite-visual-cache"
        )

    descriptor = np.full((len(windows), 40), np.nan, dtype=np.float32)
    alignment_parts: List[pd.DataFrame] = []
    run_audits: List[Dict[str, object]] = []
    eligible_total = 0
    matched_total = 0
    for run_label, run in windows.groupby("run_label", sort=False):
        eligible = run.loc[run["source_chunk"].notna()].copy()
        if eligible.empty:
            continue
        session = int(eligible["session"].iloc[0])
        block = int(eligible["block"].iloc[0])
        path = behavior_path(behavior_root, subject, session, block, game)
        source_chunk = pd.to_numeric(eligible["source_chunk"], errors="raise").to_numpy(dtype=int)
        real_step = pd.to_numeric(eligible["real_step"], errors="raise").to_numpy(dtype=int)
        target_frame = real_step + 1 + 3 * source_chunk
        source_frame = target_frame - 1
        past_frame = source_frame
        with np.load(path, allow_pickle=False) as archive:
            required = {"image", "action", "time"}
            missing = sorted(required.difference(archive.files))
            if missing:
                raise ValueError(f"Behavior file lacks {missing}: {path}")
            actions = _action_indices(archive["action"])
            times = np.asarray(archive["time"], dtype=float).reshape(-1)
            n_behavior = len(times)
            if (
                np.any(source_frame < 0)
                or np.any(target_frame >= n_behavior)
                or np.any(np.diff(target_frame) <= 0)
            ):
                raise RuntimeError(f"Invalid behavioral frame mapping for {run_label}")
            source_action = actions[source_frame]
            target_action = actions[target_frame]
            incumbent = pd.to_numeric(
                eligible["incumbent_action"], errors="coerce"
            ).fillna(-1).to_numpy(dtype=int)
            human = pd.to_numeric(
                eligible["human_action"], errors="coerce"
            ).fillna(-1).to_numpy(dtype=int)
            source_match = source_action == incumbent
            target_match = target_action == human
            pair_match = source_match & target_match
            images = archive["image"]
            actor_rows = eligible["actor_row"].to_numpy(dtype=int)
            for start in range(0, len(eligible), 512):
                stop = min(start + 512, len(eligible))
                descriptor[actor_rows[start:stop]] = low_level_visual_descriptor(
                    images[target_frame[start:stop]], images[past_frame[start:stop]]
                )
        match_fraction = float(np.mean(pair_match))
        if match_fraction < 0.98:
            raise RuntimeError(
                f"Low source/target action-pair alignment for {run_label}: {match_fraction:.4f}"
            )
        eligible_total += len(eligible)
        matched_total += int(np.sum(pair_match))
        n_chunks = int(source_chunk.max()) + 1
        expected_difference = int(4 + 3 * (n_chunks - 1))
        observed_difference = int(n_behavior - len(run))
        if observed_difference != expected_difference:
            raise RuntimeError(
                f"Behavior/actor length invariant failed for {run_label}: "
                f"observed {observed_difference}, expected {expected_difference}"
            )
        run_audits.append(
            {
                "run_label": str(run_label), "behavior_path": str(path),
                "n_eligible": int(len(eligible)), "n_behavior_frames": int(n_behavior),
                "n_source_chunks": n_chunks,
                "expected_behavior_minus_actor_rows": expected_difference,
                "observed_behavior_minus_actor_rows": observed_difference,
                "pair_action_match_fraction": match_fraction,
                "previous_time_start_s": float(times[source_frame[0]]),
                "previous_time_end_s": float(times[source_frame[-1]]),
                "current_time_start_s": float(times[target_frame[0]]),
                "current_time_end_s": float(times[target_frame[-1]]),
                "target_time_strictly_increasing": bool(np.all(np.diff(times[target_frame]) > 0)),
            }
        )
        alignment_parts.append(
            pd.DataFrame(
                {
                    "actor_row": eligible["actor_row"].to_numpy(dtype=int),
                    "run_label": str(run_label),
                    "real_step": real_step, "real_step_id": eligible["real_step_id"].astype(str),
                    "source_chunk": source_chunk, "previous_frame": source_frame,
                    "current_frame": target_frame,
                    "previous_time_s": times[source_frame], "current_time_s": times[target_frame],
                    "previous_action": source_action, "current_action": target_action,
                    "source_action_match": source_match.astype(int),
                    "target_action_match": target_match.astype(int),
                    "action_pair_match": pair_match.astype(int),
                }
            )
        )
    alignment = pd.concat(alignment_parts, ignore_index=True)
    if len(alignment) != eligible_total or alignment["actor_row"].duplicated().any():
        raise RuntimeError("Visual alignment does not cover eligible actor rows one-to-one")
    coverage = float(np.isfinite(descriptor).all(axis=1).sum() / max(eligible_total, 1))
    if coverage != 1.0:
        raise RuntimeError(f"Visual descriptor coverage is {coverage:.6f}, expected 1")
    temporary_cache = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temporary_cache.open("wb") as handle:
        np.savez_compressed(
            handle, descriptor=descriptor, actor_key_hash=np.asarray(key_hash),
            behavior_root=np.asarray(resolved_root),
            behavior_source_signature=np.asarray(behavior_signature),
            descriptor_formula_hash=np.asarray(VISUAL_DESCRIPTOR_HASH),
        )
    temporary_cache.replace(cache_path)
    alignment.to_csv(alignment_path, index=False, compression="gzip")
    qa = {
        "descriptor_version": 2, "descriptor_features": 40,
        "descriptor_definition": VISUAL_DESCRIPTOR_DEFINITION,
        "descriptor_formula_hash": VISUAL_DESCRIPTOR_HASH,
        "temporal_policy": (
            "current=image[target] before action[target]; past=image[target-1]; "
            "target+1 pixels and reward[target] are never used"
        ),
        "mapping_formula": "target_frame=real_step+1+3*source_chunk; source_frame=target_frame-1",
        "im_vectors_used": False, "behavior_png_used": False,
        "behavior_npz_image_used": True, "actor_rows": int(len(windows)),
        "eligible_rows": int(eligible_total), "mapped_rows": int(len(alignment)),
        "mapping_coverage": float(len(alignment) / max(eligible_total, 1)),
        "action_pair_match_fraction": float(matched_total / max(eligible_total, 1)),
        "actor_key_hash": key_hash, "behavior_root": resolved_root,
        "behavior_source_signature": behavior_signature,
        "behavior_sources": behavior_sources,
        "cache_path": str(cache_path), "alignment_path": str(alignment_path),
        "cache_reused": False, "runs": run_audits,
    }
    atomic_json(qa_path, qa)
    return descriptor, alignment, qa


def _fit_binary_l2(
    x: np.ndarray, y: np.ndarray
) -> Optional[Dict[str, object]]:
    values = np.asarray(x, dtype=float).copy()
    outcome = np.asarray(y, dtype=int)
    if len(values) < STAGE3_MIN_TRAIN_ROWS or len(np.unique(outcome)) < 2:
        return None
    impute = np.nanmedian(values, axis=0)
    impute = np.where(np.isfinite(impute), impute, 0.0)
    values = np.where(np.isfinite(values), values, impute[None, :])
    scaler = StandardScaler().fit(values)
    scaled = scaler.transform(values)
    model = LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs", max_iter=300,
        class_weight=None, random_state=2026,
    ).fit(scaled, outcome)
    return {"impute": impute, "scaler": scaler, "model": model}


def _predict_binary(fit: Mapping[str, object], x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=float).copy()
    impute = np.asarray(fit["impute"], dtype=float)
    values = np.where(np.isfinite(values), values, impute[None, :])
    scaler = fit["scaler"]
    model = fit["model"]
    return model.predict_proba(scaler.transform(values))[:, 1]


def binary_log_loss(y: np.ndarray, probability: np.ndarray) -> np.ndarray:
    outcome = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(probability, dtype=float), EPS, 1.0 - EPS)
    return -(outcome * np.log(p) + (1.0 - outcome) * np.log(1.0 - p))




def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "grid.alpha": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def subject_output_dir(root: Path, subject: int, game: int) -> Path:
    return root / f"sub{subject:03d}_game{game}"


def standardize_array(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    output = np.full_like(array, np.nan, dtype=float)
    if not np.any(finite):
        return output
    center = float(np.mean(array[finite]))
    scale = float(np.std(array[finite]))
    if not np.isfinite(scale) or scale <= EPS:
        output[finite] = 0.0
    else:
        output[finite] = (array[finite] - center) / scale
    return output


def exact_sign_patterns(n_units: int) -> np.ndarray:
    if n_units <= 0 or n_units > 20:
        raise ValueError(f"Invalid exact sign-pattern size: {n_units}")
    return np.asarray(list(itertools.product((-1.0, 1.0), repeat=n_units)))


def studentized_mean(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    if len(x) < 2 or not np.all(np.isfinite(x)):
        return np.nan
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    if sd <= EPS:
        if abs(mean) <= EPS:
            return 0.0
        return math.copysign(np.inf, mean)
    return mean / (sd / math.sqrt(len(x)))


def synchronized_sign_flip(matrix: np.ndarray) -> pd.DataFrame:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or not np.all(np.isfinite(values)):
        raise ValueError(f"Invalid exact-inference matrix: {values.shape}")
    patterns = exact_sign_patterns(values.shape[0])
    observed_t = np.asarray(
        [studentized_mean(values[:, column]) for column in range(values.shape[1])]
    )
    null_t = np.empty((len(patterns), values.shape[1]), dtype=float)
    for index, sign in enumerate(patterns):
        signed = values * sign[:, None]
        null_t[index] = [
            studentized_mean(signed[:, column])
            for column in range(values.shape[1])
        ]
    max_abs = np.max(np.abs(null_t), axis=1)
    rows = []
    for column in range(values.shape[1]):
        threshold = abs(observed_t[column])
        rows.append(
            {
                "mean_beta": float(np.mean(values[:, column])),
                "studentized_t": float(observed_t[column]),
                "exact_sign_flip_p": float(
                    np.mean(np.abs(null_t[:, column]) >= threshold - 1e-12)
                ),
                "synchronized_max_t_p": float(
                    np.mean(max_abs >= threshold - 1e-12)
                ),
                "n_positive": int(np.sum(values[:, column] > 0)),
                "n_units": int(values.shape[0]),
                "n_sign_patterns": int(len(patterns)),
            }
        )
    return pd.DataFrame(rows)


def significance_stars(p_value: float) -> str:
    if not np.isfinite(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def transition_margin(
    prior: np.ndarray, incumbent: np.ndarray
) -> np.ndarray:
    probabilities = np.asarray(prior, dtype=float)
    action = np.asarray(incumbent, dtype=int)
    result = np.full(len(action), np.nan, dtype=float)
    for index, current in enumerate(action):
        row = probabilities[index]
        if not np.all(np.isfinite(row)) or np.any(row < 0) or np.sum(row) <= 0:
            continue
        row = row / np.sum(row)
        if current == 0:
            result[index] = math.log(max(row[0], EPS)) - math.log(
                max(float(np.max(row[1:])), EPS)
            )
        elif 1 <= current < N_ACTIONS:
            result[index] = math.log(max(row[current], EPS)) - math.log(
                max(row[0], EPS)
            )
    return result


def branch_alignment_sign(branch: np.ndarray, incumbent: np.ndarray) -> np.ndarray:
    """Destination-positive branch sign without using the eventual human action."""
    b = np.asarray(branch, dtype=int)
    inc = np.asarray(incumbent, dtype=int)
    sign = np.zeros(np.broadcast_shapes(b.shape, inc.shape), dtype=float)
    b, inc = np.broadcast_arrays(b, inc)
    noop_risk = inc == 0
    sign[noop_risk & (b == 0)] = -1.0
    sign[noop_risk & (b >= 1) & (b < N_ACTIONS)] = 1.0
    active_risk = (inc >= 1) & (inc < N_ACTIONS)
    sign[active_risk & (b == 0)] = 1.0
    sign[active_risk & (b == inc)] = -1.0
    return sign


def transition_name(incumbent: np.ndarray) -> np.ndarray:
    inc = np.asarray(incumbent, dtype=int)
    result = np.full(len(inc), "invalid", dtype=object)
    result[inc == 0] = "noop_to_active"
    result[(inc >= 1) & (inc < N_ACTIONS)] = "active_to_noop"
    return result


def load_legacy_actor_and_visual(
    subject: int,
    game: int,
    windows: pd.DataFrame,
    legacy_root: Path,
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, object]]:
    directory = subject_output_dir(legacy_root, subject, game)
    manifest_path = directory / "analysis_manifest.json"
    actor_path = directory / "tables" / "actor_geometry_loro_predictions.csv.gz"
    visual_path = directory / "cache" / "visual_state_controls_v2.npz"
    for path in (manifest_path, actor_path, visual_path):
        if not path.exists():
            raise FileNotFoundError(f"Required frozen-04 artifact is missing: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("script_sha256") != EXPECTED_ACCEPTED_04_SHA256:
        raise RuntimeError(f"Frozen-04 manifest SHA mismatch: {manifest_path}")
    if int(manifest.get("schema_version", -1)) != 3:
        raise RuntimeError(f"Frozen-04 manifest schema mismatch: {manifest_path}")
    actor = pd.read_csv(actor_path)
    if len(actor) != len(windows) or actor["actor_row"].duplicated().any():
        raise RuntimeError("Frozen Actor OOF prediction does not match actor spine")
    actor = actor.sort_values("actor_row", kind="stable").reset_index(drop=True)
    if not actor["actor_row"].eq(windows["actor_row"].to_numpy()).all():
        raise RuntimeError("Frozen Actor OOF row order differs from current actor spine")
    with np.load(visual_path, allow_pickle=False) as archive:
        visual = np.asarray(archive["descriptor"], dtype=float)
        actor_key_hash = str(archive["actor_key_hash"].item())
    expected_key_hash = _actor_key_hash(windows)
    if actor_key_hash != expected_key_hash or visual.shape != (len(windows), 40):
        raise RuntimeError("Frozen visual descriptor provenance/shape mismatch")
    qa = {
        "legacy_manifest": str(manifest_path),
        "legacy_manifest_sha256": sha256_path(manifest_path),
        "legacy_actor_prediction": str(actor_path),
        "legacy_actor_prediction_sha256": sha256_path(actor_path),
        "legacy_visual_cache": str(visual_path),
        "legacy_visual_cache_sha256": sha256_path(visual_path),
        "actor_rows": int(len(actor)),
        "visual_shape": [int(value) for value in visual.shape],
        "actor_key_hash": actor_key_hash,
    }
    return actor, visual, qa


def _json_signature(payload: Mapping[str, object]) -> str:
    serialized = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _validate_actor_prediction(
    actor: pd.DataFrame, windows: pd.DataFrame
) -> None:
    required = ["actor_row", "actor_loro_eligible", "actor_loro_oof_complete"] + [
        f"actor_{model}_oof_p_{action}"
        for model in ("P", "A2", "A3")
        for action in range(N_ACTIONS)
    ]
    missing = sorted(set(required).difference(actor.columns))
    if missing:
        raise RuntimeError(f"Actor prediction cache lacks columns: {missing}")
    if len(actor) != len(windows) or actor["actor_row"].duplicated().any():
        raise RuntimeError("Actor prediction cache does not match actor spine")
    actor_rows = pd.to_numeric(actor["actor_row"], errors="raise").to_numpy(dtype=int)
    expected = pd.to_numeric(
        windows["actor_row"], errors="raise"
    ).to_numpy(dtype=int)
    if not np.array_equal(actor_rows, expected):
        raise RuntimeError("Actor prediction row order differs from actor spine")
    eligible = pd.to_numeric(
        actor["actor_loro_eligible"], errors="raise"
    ).eq(1).to_numpy()
    complete = pd.to_numeric(
        actor["actor_loro_oof_complete"], errors="raise"
    ).eq(1).to_numpy()
    if not np.array_equal(eligible, complete):
        raise RuntimeError("Actor eligible and OOF-complete masks differ")
    probability_columns = [
        f"actor_{model}_oof_p_{action}"
        for model in ("P", "A2", "A3")
        for action in range(N_ACTIONS)
    ]
    probability = actor.loc[eligible, probability_columns].to_numpy(dtype=float)
    if not np.all(np.isfinite(probability)):
        raise RuntimeError("Actor OOF probabilities contain non-finite values")
    probability = probability.reshape(-1, 3, N_ACTIONS)
    if not np.allclose(probability.sum(axis=2), 1.0, atol=1e-8):
        raise RuntimeError("Actor OOF probability rows do not sum to one")


def prepare_actor_and_visual_from_basic_cache(
    *,
    subject: int,
    game: int,
    windows: pd.DataFrame,
    event_frame: pd.DataFrame,
    shared_manifest: Mapping[str, object],
    behavior_root: Path,
    output: Path,
    rebuild: bool,
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, object]]:
    """Derive Stage-3 controls from the canonical cache, never neural cache."""
    cache_root = output / "cache"
    qa_root = output / "qa"
    table_root = output / "tables"
    cache_root.mkdir(parents=True, exist_ok=True)
    qa_root.mkdir(parents=True, exist_ok=True)
    table_root.mkdir(parents=True, exist_ok=True)
    prediction_path = cache_root / "actor_loro_predictions_from_basic_cache.csv.gz"
    fold_path = table_root / "actor_loro_folds_from_basic_cache.csv"
    coefficient_path = cache_root / "actor_loro_coefficients_from_basic_cache.csv.gz"
    sidecar_path = qa_root / "actor_loro_from_basic_cache.json"
    actor_key_hash = _actor_key_hash(windows)
    shared_manifest_sha = _json_signature(shared_manifest)
    dependency = {
        "cache_version": DERIVED_ACTOR_CACHE_VERSION,
        "actor_implementation_sha256": DERIVED_ACTOR_IMPLEMENTATION_HASH,
        "actor_key_hash": actor_key_hash,
        "shared_manifest_sha256": shared_manifest_sha,
        "subject": int(subject),
        "game": int(game),
        "n_rows": int(len(windows)),
    }
    reused = False
    actor_qa: Dict[str, object]
    if prediction_path.exists() and sidecar_path.exists() and not rebuild:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if sidecar.get("dependency") != dependency:
            raise RuntimeError(
                f"Stale basic-cache Actor derivative at {prediction_path}; "
                "rerun with --rebuild-derived-features"
            )
        if sidecar.get("prediction_sha256") != sha256_path(prediction_path):
            raise RuntimeError(f"Actor derivative content hash mismatch: {prediction_path}")
        actor_prediction = pd.read_csv(prediction_path)
        _validate_actor_prediction(actor_prediction, windows)
        actor_qa = dict(sidecar.get("actor_qa", {}))
        reused = True
    else:
        actor_folds, actor_prediction, actor_coefficients, actor_qa = (
            fit_actor_geometry_loro(windows, event_frame, subject, game)
        )
        _validate_actor_prediction(actor_prediction, windows)
        actor_folds.to_csv(fold_path, index=False)
        actor_prediction.to_csv(
            prediction_path, index=False, compression="gzip"
        )
        actor_coefficients.to_csv(
            coefficient_path, index=False, compression="gzip"
        )
        atomic_json(
            sidecar_path,
            {
                "dependency": dependency,
                "prediction_path": str(prediction_path),
                "prediction_sha256": sha256_path(prediction_path),
                "fold_path": str(fold_path),
                "fold_sha256": sha256_path(fold_path),
                "coefficient_path": str(coefficient_path),
                "coefficient_sha256": sha256_path(coefficient_path),
                "actor_qa": actor_qa,
            },
        )
    visual, alignment, visual_qa = build_visual_state_controls(
        windows,
        subject=subject,
        game=game,
        behavior_root=behavior_root,
        output_dir=output,
        overwrite=rebuild,
    )
    expected_visual_rows = pd.to_numeric(
        windows.loc[windows["source_chunk"].notna(), "actor_row"],
        errors="raise",
    ).to_numpy(dtype=int)
    aligned_visual_rows = pd.to_numeric(
        alignment["actor_row"], errors="raise"
    ).to_numpy(dtype=int)
    if visual.shape != (len(windows), 40) or not np.array_equal(
        np.sort(aligned_visual_rows), np.sort(expected_visual_rows)
    ):
        raise RuntimeError("Basic-cache visual derivative does not match actor spine")
    qa = {
        "source": "native_zarr_cache" if "native_visual_controls" in windows.attrs else "canonical_shared_cache_plus_behavioral_npz",
        "neural_cache_used": False,
        "accepted_04_output_used": False,
        "actor_prediction_reused": bool(reused),
        "actor_dependency": dependency,
        "actor_prediction_path": str(prediction_path),
        "actor_prediction_sha256": sha256_path(prediction_path),
        "actor_qa": actor_qa,
        "visual_qa": visual_qa,
    }
    return actor_prediction, visual, qa


def actor_transition_log_odds(
    probabilities: np.ndarray, incumbent: np.ndarray
) -> np.ndarray:
    matrix = np.asarray(probabilities, dtype=float)
    inc = np.asarray(incumbent, dtype=int)
    result = np.full(len(inc), np.nan, dtype=float)
    for index, current in enumerate(inc):
        row = matrix[index]
        if not np.all(np.isfinite(row)) or np.any(row < 0) or np.sum(row) <= 0:
            continue
        row = row / np.sum(row)
        if current == 0:
            result[index] = math.log(max(float(np.sum(row[1:])), EPS)) - math.log(
                max(row[0], EPS)
            )
        elif 1 <= current < N_ACTIONS:
            result[index] = math.log(max(row[0], EPS)) - math.log(
                max(row[current], EPS)
            )
    return result


def _event_length_groups(events, windows):
    """Group complete windows by run and observed length, without padding."""
    counts = events.groupby(KEYS, sort=False).agg(
        event_count=("status2_position", "size"), maximum=("status2_position", "max"),
        minimum=("status2_position", "min"), unique=("status2_position", "nunique"),
        branch_min=("branch", "min"), branch_max=("branch", "max"))
    if "n_status2" in windows:
        counts = counts.join(windows.set_index(KEYS)["n_status2"], how="left")
        if not counts.event_count.eq(counts.n_status2).all():
            raise ValueError("Event count differs from recorded planning budget")
    valid = ((counts.event_count >= 2) & counts.minimum.eq(0)
             & counts.maximum.eq(counts.event_count-1) & counts.unique.eq(counts.event_count)
             & counts.branch_min.ge(0) & counts.branch_max.lt(N_ACTIONS))
    sizes = counts.loc[valid, ["event_count"]].reset_index()
    selected = events.merge(sizes, on=KEYS, how="inner", validate="many_to_one")
    for _, group in selected.groupby(["run_label", "event_count"], sort=False):
        yield group


def _complete_run_event_arrays(
    run_events: pd.DataFrame, windows: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Return rectangular arrays for a single observed-length group."""
    if run_events["run_label"].nunique() != 1:
        raise ValueError("Event-array construction is run-local")
    ordered = run_events.sort_values(KEYS + ["status2_position"], kind="stable")
    sizes = ordered.groupby(KEYS, sort=False, observed=True).size()
    if sizes.empty or sizes.nunique() != 1:
        raise ValueError("Event arrays require one observed-length group")
    N_EVENTS = int(sizes.iloc[0])
    complete = sizes.index[sizes.eq(N_EVENTS)]
    if len(complete) == 0:
        raise RuntimeError("No complete imagination window in run")
    key_frame = pd.DataFrame(list(complete), columns=KEYS)
    ordered = ordered.merge(key_frame, on=KEYS, how="inner", validate="many_to_one")
    ordered = ordered.sort_values(KEYS + ["status2_position"], kind="stable")
    first_keys = ordered[KEYS].iloc[::N_EVENTS].reset_index(drop=True)
    metadata_columns = KEYS + [
        "actor_row", "subject", "game", "incumbent_action", "human_action",
        "action_before_incumbent", "strongest_competitor_margin", "bout_age",
        "prior_reward", "prior_done", "run_progress",
        "value_revision_event_fraction",
    ] + [f"model_root_p_{action}" for action in range(N_ACTIONS)] + [
        f"actor_terminal_p_{action}" for action in range(N_ACTIONS)
    ] + [f"root_q_initial_{action}" for action in range(N_ACTIONS)] + [
        f"root_n_initial_{action}" for action in range(N_ACTIONS)
    ]
    metadata = first_keys.merge(
        windows[metadata_columns], on=KEYS, how="left", validate="one_to_one"
    )
    if metadata["actor_row"].isna().any():
        raise RuntimeError("Complete event window lacks actor-spine metadata")
    position = ordered["status2_position"].to_numpy(dtype=int).reshape(-1, N_EVENTS)
    if not np.array_equal(position, np.broadcast_to(np.arange(N_EVENTS), position.shape)):
        raise RuntimeError("Event positions must be exactly 0 through observed L-1")
    arrays = {
        "branch": ordered["branch"].to_numpy(dtype=int).reshape(-1, N_EVENTS),
        "depth": pd.to_numeric(ordered["depth"], errors="coerce").to_numpy(dtype=float).reshape(-1, N_EVENTS),
        "E": pd.to_numeric(ordered["return_advantage"], errors="coerce").to_numpy(dtype=float).reshape(-1, N_EVENTS),
        "q": pd.to_numeric(ordered["root_q_revision"], errors="coerce").to_numpy(dtype=float).reshape(-1, N_EVENTS),
        "directed_q": pd.to_numeric(ordered["directed_value_revision"], errors="coerce").to_numpy(dtype=float).reshape(-1, N_EVENTS),
        "q_valid": pd.to_numeric(ordered["valid_repeated_q_revision"], errors="coerce").fillna(0).to_numpy(dtype=int).reshape(-1, N_EVENTS),
        "root_n_before": pd.to_numeric(ordered["root_n_before_sampled"], errors="coerce").to_numpy(dtype=float).reshape(-1, N_EVENTS),
    }
    if np.any((arrays["branch"] < 0) | (arrays["branch"] >= N_ACTIONS)):
        raise RuntimeError("Invalid event branch")
    if not np.all(np.isfinite(arrays["depth"])):
        raise RuntimeError("Non-finite event depth")
    return metadata, arrays


def _build_transition_suffix_length_group(
    run_events: pd.DataFrame, windows: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Strict j -> k>j table in origin/destination transition coordinates."""
    metadata, arrays = _complete_run_event_arrays(run_events, windows)
    N_EVENTS = arrays["branch"].shape[1]
    n_window = len(metadata)
    incumbent = pd.to_numeric(metadata["incumbent_action"], errors="coerce").fillna(-1).to_numpy(dtype=int)
    transitions = transition_name(incumbent)
    prior = metadata[[f"model_root_p_{a}" for a in range(N_ACTIONS)]].to_numpy(dtype=float)
    initial_q = metadata[[f"root_q_initial_{a}" for a in range(N_ACTIONS)]].to_numpy(dtype=float)
    initial_n = metadata[[f"root_n_initial_{a}" for a in range(N_ACTIONS)]].to_numpy(dtype=float)
    margin = transition_margin(prior, incumbent)
    margin_z = standardize_array(margin)
    pieces: List[pd.DataFrame] = []
    for event_index in range(N_EVENTS - MIN_FUTURE):
        branch = arrays["branch"][:, event_index]
        sign = branch_alignment_sign(branch, incumbent)
        future_sign = branch_alignment_sign(
            arrays["branch"][:, event_index + 1 :], incumbent[:, None]
        )
        future_depth = arrays["depth"][:, event_index + 1 :]
        if event_index:
            prefix_sign = branch_alignment_sign(
                arrays["branch"][:, :event_index], incumbent[:, None]
            )
            prefix_share = np.mean(prefix_sign, axis=1)
            prefix_depth = np.mean(
                prefix_sign * arrays["depth"][:, :event_index], axis=1
            )
        else:
            prefix_share = np.zeros(n_window, dtype=float)
            prefix_depth = np.zeros(n_window, dtype=float)
        row_index = np.arange(n_window)
        sampled_prior = prior[row_index, branch]
        sampled_initial_q = initial_q[row_index, branch]
        sampled_visited = (initial_n[row_index, branch] > 0).astype(float)
        other_q = initial_q.copy()
        other_q[row_index, branch] = np.nan
        initial_q_contrast = sampled_initial_q - np.nanmax(other_q, axis=1)
        current_e = sign * arrays["E"][:, event_index]
        current_q = np.where(
            arrays["q_valid"][:, event_index].astype(bool),
            sign * arrays["q"][:, event_index],
            np.nan,
        )
        directed_q = np.where(
            arrays["q_valid"][:, event_index].astype(bool),
            arrays["directed_q"][:, event_index],
            np.nan,
        )
        piece = metadata[KEYS + ["actor_row", "subject", "game"]].copy()
        piece["window_id"] = (
            piece["run_label"].astype(str) + "|" + piece["real_step"].astype(str)
            + "|" + piece["real_step_id"].astype(str)
        )
        piece["transition_type"] = transitions
        piece["event_index"] = int(event_index)
        piece["n_future_events"] = int(N_EVENTS - 1 - event_index)
        piece["current_branch"] = branch
        piece["current_depth"] = arrays["depth"][:, event_index]
        piece["transition_margin_M"] = margin
        piece["transition_margin_M_z"] = margin_z
        piece["aligned_E"] = current_e
        piece["aligned_signed_q"] = current_q
        piece["directed_q"] = directed_q
        piece["M_x_E"] = margin_z * current_e
        piece["M_x_signed_q"] = margin_z * current_q
        piece["M_x_directed_q"] = margin_z * directed_q
        piece["sampled_model_prior"] = np.log(np.clip(sampled_prior, EPS, 1.0))
        piece["sampled_initial_root_q"] = sampled_initial_q
        piece["sampled_initial_root_q_visited"] = sampled_visited
        piece["initial_root_q_contrast"] = initial_q_contrast
        piece["log_prior_visits"] = np.log1p(
            np.maximum(arrays["root_n_before"][:, event_index], 0.0)
        )
        piece["prefix_destination_share_contrast"] = prefix_share
        piece["prefix_destination_depth_score"] = prefix_depth
        piece["future_destination_share_contrast"] = np.mean(future_sign, axis=1)
        piece["future_destination_depth_score"] = np.mean(
            future_sign * future_depth, axis=1
        )
        pieces.append(piece)
    suffix = pd.concat(pieces, ignore_index=True)
    suffix = suffix.loc[suffix["transition_type"].isin(TRANSITION_LABELS)].copy()
    counts = suffix.groupby("window_id", sort=False)["event_index"].nunique()
    if not counts.eq(N_EVENTS - MIN_FUTURE).all():
        raise RuntimeError("Strict suffix row-count invariant failed")
    qa = {
        "run_label": str(run_events["run_label"].iloc[0]),
        "complete_windows": int(len(counts)),
        "rows": int(len(suffix)),
        "rows_per_window": int(N_EVENTS - MIN_FUTURE),
        "minimum_future_events": int(MIN_FUTURE),
        "maximum_included_event": int(N_EVENTS - MIN_FUTURE - 1),
        "human_action_used_for_alignment": False,
        "alignment_definition": (
            "N-origin: active +1, NOOP -1; A-origin: NOOP +1, "
            "incumbent active -1, other active 0"
        ),
    }
    return suffix, qa


def build_transition_suffix_run(run_events, windows):
    pieces, audits = [], []
    for group in _event_length_groups(run_events, windows):
        suffix, qa = _build_transition_suffix_length_group(group, windows)
        pieces.append(suffix)
        audits.append(qa)
    if not pieces:
        raise RuntimeError("No windows with at least two branch-bearing computations")
    result = pd.concat(pieces, ignore_index=True)
    # Standardize M once across eligible windows in the run, not within length.
    margin = result.drop_duplicates("window_id").set_index("window_id")["transition_margin_M"]
    margin_z = standardize(margin)
    result["transition_margin_M_z"] = result["window_id"].map(margin_z)
    for source, target in (("aligned_E", "M_x_E"), ("aligned_signed_q", "M_x_signed_q"), ("directed_q", "M_x_directed_q")):
        result[target] = result["transition_margin_M_z"] * result[source]
    return result, {"run_label": str(run_events.run_label.iloc[0]), "minimum_future_events": MIN_FUTURE,
                    "human_action_used_for_alignment": False,
                    "rows": len(result), "windows": int(result.window_id.nunique()), "length_groups": audits,
                    "excluded_geometry_windows": int(run_events[KEYS].drop_duplicates().shape[0]-result.window_id.nunique())}


def _event_design_fit(
    clean: pd.DataFrame,
    *,
    focal: str,
    interaction: str,
    outcome: str,
) -> Tuple[Mapping[str, float], Dict[str, object]]:
    controls = [
        "current_depth", "sampled_model_prior", "sampled_initial_root_q",
        "sampled_initial_root_q_visited", "initial_root_q_contrast",
        "log_prior_visits", "prefix_destination_share_contrast",
        "prefix_destination_depth_score",
    ]
    extra = ["aligned_E"] if focal in ("aligned_signed_q", "directed_q") else []
    predictors = list(dict.fromkeys([focal, interaction] + extra + controls))
    columns = list(dict.fromkeys(predictors + [outcome]))
    data = clean[
        columns + ["window_id", "event_index", "current_branch"]
    ].replace([np.inf, -np.inf], np.nan).dropna()
    # A singleton row vanishes under window fixed effects and carries no information.
    data = data.loc[data.groupby("window_id")["window_id"].transform("size") >= 2].copy()
    if data["window_id"].nunique() < 40:
        raise RuntimeError(
            f"Insufficient event rows for {focal}/{outcome}: "
            f"rows={len(data)}, windows={data['window_id'].nunique()}"
        )
    dummies = pd.concat(
        [
            pd.get_dummies(data["event_index"].astype(int), prefix="position", drop_first=True, dtype=float),
            pd.get_dummies(data["current_branch"].astype(int), prefix="branch", drop_first=True, dtype=float),
        ],
        axis=1,
    )
    augmented = pd.concat([data[columns + ["window_id"]], dummies], axis=1)
    residual = _residualize_window_fixed_effects(
        augmented, columns + dummies.columns.tolist()
    )
    for column in columns:
        residual[column] = standardize_array(residual[column].to_numpy(dtype=float))
    design = residual[predictors + dummies.columns.tolist()].copy()
    design = design.loc[:, design.std(ddof=0).fillna(0).gt(EPS)]
    design, diagnostics = _ordered_estimable_event_design(design, focal)
    if len(design) <= design.shape[1] or np.linalg.matrix_rank(design.to_numpy()) != design.shape[1]:
        raise RuntimeError("Insufficient residual degrees of freedom or rank-deficient design")
    count = data.groupby("window_id", sort=False)["window_id"].transform("size")
    weights = 1.0 / count.to_numpy(dtype=float)
    fit = sm.WLS(
        residual[outcome].astype(float), design.astype(float), weights=weights
    ).fit()
    weight_sums = pd.Series(weights).groupby(data["window_id"].reset_index(drop=True)).sum()
    result = {
        "focal_beta": float(fit.params[focal]),
        "interaction_beta": float(fit.params.get(interaction, np.nan)),
        "focal_se": float(fit.bse[focal]),
        "interaction_se": float(fit.bse.get(interaction, np.nan)),
        "n_rows": int(len(data)),
        "n_windows": int(data["window_id"].nunique()),
        "window_weight_sum_min": float(weight_sums.min()),
        "window_weight_sum_max": float(weight_sums.max()),
    }
    return result, diagnostics


def fit_stage1_run(
    suffix: pd.DataFrame, subject: int, game: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    records: List[Dict[str, object]] = []
    curve_rows: List[Dict[str, object]] = []
    run_label = str(suffix["run_label"].iloc[0])
    for transition in TRANSITION_LABELS:
        transition_data = suffix.loc[suffix["transition_type"].eq(transition)].copy()
        for outcome in OUTCOME_LABELS:
            for focal, interaction, is_primary in (
                ("aligned_E", "M_x_E", True),
                ("aligned_signed_q", "M_x_signed_q", True),
                ("directed_q", "M_x_directed_q", False),
            ):
                try:
                    result, diagnostics = _event_design_fit(
                        transition_data, focal=focal, interaction=interaction, outcome=outcome
                    )
                    diagnostics["estimable"] = True
                    diagnostics["not_estimable_reason"] = ""
                except RuntimeError as error:
                    result = {"focal_beta": np.nan, "interaction_beta": np.nan,
                              "n_rows": len(transition_data), "n_windows": transition_data.window_id.nunique()}
                    diagnostics = {"estimable": False, "not_estimable_reason": str(error)}
                common = {
                    "subject": subject,
                    "game": game,
                    "run_label": run_label,
                    "transition_type": transition,
                    "outcome": outcome,
                    "predictor": focal,
                    "minimum_future_events": MIN_FUTURE,
                    **{key: value for key, value in result.items() if key not in ("focal_beta", "interaction_beta")},
                    **diagnostics,
                }
                records.append(
                    {
                        **common,
                        "effect_id": (
                            f"stage1::{transition}::{focal}::{outcome}"
                            if is_primary else
                            f"stage1_sensitivity::{transition}::{focal}::{outcome}"
                        ),
                        "effect_role": (
                            "primary_aligned_event_effect" if is_primary
                            else "sensitivity_03_exact_action_directed_q"
                        ),
                        "beta": result["focal_beta"],
                        "primary": int(is_primary),
                    }
                )
                records.append(
                    {
                        **common,
                        "effect_id": f"stage1_moderation::{transition}::{interaction}::{outcome}",
                        "effect_role": "margin_moderation",
                        "beta": result["interaction_beta"],
                        "estimable": bool(np.isfinite(result["interaction_beta"])),
                        "not_estimable_reason": (
                            "" if np.isfinite(result["interaction_beta"]) else
                            diagnostics["not_estimable_reason"] or
                            f"Interaction {interaction} has no independent within-window variation in the full design"
                        ),
                        "primary": 0,
                    }
                )
                plot_data = transition_data[
                    ["window_id", "transition_margin_M_z", focal, outcome]
                ].replace([np.inf, -np.inf], np.nan).dropna()
                plot_data["x"] = plot_data[focal] - plot_data.groupby(
                    "window_id", sort=False
                )[focal].transform("mean")
                plot_data["y"] = plot_data[outcome] - plot_data.groupby(
                    "window_id", sort=False
                )[outcome].transform("mean")
                plot_data["x"] = standardize_array(plot_data["x"].to_numpy())
                plot_data["margin_group"] = np.where(
                    plot_data["transition_margin_M_z"] < 0, "low_M", "high_M"
                )
                for margin_group, group in [("all", plot_data)] + list(
                    plot_data.groupby("margin_group", sort=False)
                ):
                    group = group.copy()
                    if group["x"].nunique() < 5:
                        continue
                    group["bin"] = pd.qcut(
                        group["x"], q=5, labels=False, duplicates="drop"
                    )
                    curve = group.groupby("bin", observed=True).agg(
                        predictor_z=("x", "mean"),
                        future_geometry_within_window=("y", "mean"),
                        n_rows=("y", "size"),
                    ).reset_index()
                    for row in curve.to_dict("records"):
                        curve_rows.append({
                            "subject": subject, "game": game,
                            "run_label": run_label,
                            "transition_type": transition, "predictor": focal,
                            "outcome": outcome, "margin_group": margin_group,
                            **row,
                        })
    return pd.DataFrame(records), pd.DataFrame(curve_rows, columns=["subject", "game", "run_label",
        "transition_type", "predictor", "outcome", "margin_group", "bin", "predictor_z",
        "future_geometry_within_window", "n_rows"])


def _linear_slope(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    if np.sum(finite) < 3:
        return np.nan
    x = np.arange(len(array), dtype=float)[finite]
    y = array[finite]
    x = x - np.mean(x)
    denominator = float(np.dot(x, x))
    return float(np.dot(x, y - np.mean(y)) / denominator) if denominator > EPS else np.nan


def build_window_pathway_summary(
    event_frame: pd.DataFrame,
    windows: pd.DataFrame,
    actor_prediction: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """One actor-spine row with aligned event dynamics, dG, and Actor terms."""
    summaries: List[pd.DataFrame] = []
    for run_events in _event_length_groups(event_frame, windows):
        metadata, arrays = _complete_run_event_arrays(run_events, windows)
        length = arrays["branch"].shape[1]
        incumbent = pd.to_numeric(metadata["incumbent_action"], errors="coerce").fillna(-1).to_numpy(dtype=int)
        sign = branch_alignment_sign(arrays["branch"], incumbent[:, None])
        aligned_e = sign * arrays["E"]
        aligned_q = np.where(arrays["q_valid"].astype(bool), sign * arrays["q"], np.nan)
        directed_q = np.where(
            arrays["q_valid"].astype(bool), arrays["directed_q"], np.nan
        )
        share = sign
        depth_score = sign * arrays["depth"]
        early = slice(0, length // 2)
        late = slice(length // 2, length)
        frame = metadata[KEYS + ["actor_row", "subject", "game"]].copy()
        frame["n_status2"] = length
        frame["transition_type"] = transition_name(incumbent)
        frame["transition_margin_M"] = transition_margin(
            metadata[[f"model_root_p_{a}" for a in range(N_ACTIONS)]].to_numpy(dtype=float), incumbent
        )
        frame["aligned_E_mean"] = np.nanmean(aligned_e, axis=1)
        frame["aligned_E_slope"] = np.asarray([_linear_slope(row) for row in aligned_e])
        frame["aligned_signed_q_mean"] = np.nanmean(aligned_q, axis=1)
        frame["aligned_signed_q_slope"] = np.asarray([_linear_slope(row) for row in aligned_q])
        frame["directed_q_mean"] = np.nanmean(directed_q, axis=1)
        frame["directed_q_slope"] = np.asarray([_linear_slope(row) for row in directed_q])
        frame["signed_q_available"] = np.any(np.isfinite(aligned_q), axis=1).astype(float)
        frame["signed_q_event_count"] = np.sum(np.isfinite(aligned_q), axis=1)
        repeated_abs_q = np.where(arrays["q_valid"].astype(bool), np.abs(arrays["q"]), np.nan)
        frame["R_log"] = np.log(np.maximum(np.nanmean(repeated_abs_q, axis=1), EPS))
        frame["early_destination_share"] = np.mean(share[:, early], axis=1)
        frame["late_destination_share"] = np.mean(share[:, late], axis=1)
        frame["destination_share_shift"] = frame["late_destination_share"] - frame["early_destination_share"]
        frame["early_destination_depth"] = np.mean(depth_score[:, early], axis=1)
        frame["late_destination_depth"] = np.mean(depth_score[:, late], axis=1)
        frame["destination_depth_shift"] = frame["late_destination_depth"] - frame["early_destination_depth"]
        summaries.append(frame)
    summary = pd.concat(summaries, ignore_index=True).sort_values("actor_row", kind="stable").reset_index(drop=True)
    summary = summary.merge(
        windows[[
            "actor_row", "incumbent_action", "human_action", "action_before_incumbent",
            "bout_age", "prior_reward", "prior_done", "run_progress",
            "value_revision_event_fraction",
        ] + [f"actor_terminal_p_{a}" for a in range(N_ACTIONS)]],
        on="actor_row", how="left", validate="one_to_one",
    ).merge(actor_prediction, on="actor_row", how="left", validate="one_to_one")
    # Temporal controls use the full real-step spine, including budget 0/1.
    # Episode boundaries must not carry an active action over from another game.
    history = windows[["actor_row", "run_label", "incumbent_action"] +
                      (["episode_id"] if "episode_id" in windows else [])].copy()
    history = history.sort_values("actor_row", kind="stable")
    active = pd.to_numeric(history["incumbent_action"], errors="coerce")
    active = active.where(active.between(1, N_ACTIONS-1))
    groups = [history["run_label"]] + ([history["episode_id"]] if "episode_id" in history else [])
    history["pre_bout_active_action"] = active.groupby(groups, sort=False).ffill()
    summary = summary.merge(history[["actor_row", "pre_bout_active_action"]],
                            on="actor_row", how="left", validate="one_to_one")
    incumbent = pd.to_numeric(summary["incumbent_action"], errors="coerce").fillna(-1).to_numpy(dtype=int)
    actual = summary[[f"actor_terminal_p_{a}" for a in range(N_ACTIONS)]].to_numpy(dtype=float)
    p_only = summary[[f"actor_P_oof_p_{a}" for a in range(N_ACTIONS)]].to_numpy(dtype=float)
    a2 = summary[[f"actor_A2_oof_p_{a}" for a in range(N_ACTIONS)]].to_numpy(dtype=float)
    a3 = summary[[f"actor_A3_oof_p_{a}" for a in range(N_ACTIONS)]].to_numpy(dtype=float)
    summary["actor_actual_transition_log_odds"] = actor_transition_log_odds(actual, incumbent)
    summary["actor_P_transition_log_odds"] = actor_transition_log_odds(p_only, incumbent)
    summary["actor_A2_transition_log_odds"] = actor_transition_log_odds(a2, incumbent)
    summary["actor_A3_transition_log_odds"] = actor_transition_log_odds(a3, incumbent)
    summary["actor_residual_beyond_P"] = summary["actor_actual_transition_log_odds"] - summary["actor_P_transition_log_odds"]
    summary["actor_geometry_transition_increment"] = summary["actor_A3_transition_log_odds"] - summary["actor_A2_transition_log_odds"]
    qa = {
        "complete_window_rows": int(len(summary)),
        "unique_actor_rows": int(summary["actor_row"].nunique()),
        "transition_counts": {str(k): int(v) for k, v in summary["transition_type"].value_counts().items()},
        "alignment_uses_human_destination": False,
        "early_positions": "0 .. L//2-1",
        "late_positions": "L//2 .. L-1",
        "actor_residual_definition": "actual terminal transition log-odds minus P-only OOF transition log-odds",
        "actor_geometry_increment_definition": "A3(P+V+G) OOF transition log-odds minus A2(P+V) OOF transition log-odds",
    }
    return summary, qa


def summarize_stage2_runs(
    windows: pd.DataFrame, subject: int, game: int
) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for (run_label, transition), data in windows.groupby(["run_label", "transition_type"], sort=False):
        if transition not in TRANSITION_LABELS or len(data) < 20:
            continue
        for metric, early_name, late_name in (
            ("destination_share_shift", "early_destination_share", "late_destination_share"),
            ("destination_depth_shift", "early_destination_depth", "late_destination_depth"),
        ):
            early = pd.to_numeric(data[early_name], errors="coerce")
            late = pd.to_numeric(data[late_name], errors="coerce")
            valid = early.notna() & late.notna()
            if int(valid.sum()) < 20:
                continue
            early_mean = float(early[valid].mean())
            late_mean = float(late[valid].mean())
            records.append({
                "subject": subject, "game": game, "run_label": str(run_label),
                "transition_type": str(transition), "metric": metric,
                "effect_id": f"stage2::{transition}::{metric}",
                "early_mean": early_mean, "late_mean": late_mean,
                "beta": late_mean - early_mean,
                "n_windows": int(valid.sum()), "primary": 1,
            })
    result = pd.DataFrame(records)
    expected = windows["run_label"].nunique() * len(STAGE2_PRIMARY_IDS)
    if len(result) != expected:
        raise RuntimeError(f"Stage-2 run grid incomplete: {len(result)} != {expected}")
    return result


def _stage3_action_controls(frame: pd.DataFrame) -> Dict[str, List[str]]:
    incumbent = pd.to_numeric(frame["incumbent_action"], errors="coerce").fillna(-1).to_numpy(dtype=int)
    active = pd.to_numeric(frame["incumbent_action"], errors="coerce").where(
        pd.to_numeric(frame["incumbent_action"], errors="coerce").between(1, N_ACTIONS - 1)
    )
    if "pre_bout_active_action" not in frame:
        frame["pre_bout_active_action"] = active.groupby(frame["run_label"], sort=False).ffill()
    previous = pd.to_numeric(frame["action_before_incumbent"], errors="coerce")
    for action in range(1, N_ACTIONS):
        frame[f"incumbent_is_{action}"] = (incumbent == action).astype(float)
        frame[f"previous_incumbent_is_{action}"] = (previous == action).astype(float)
        frame[f"pre_bout_active_is_{action}"] = (frame["pre_bout_active_action"] == action).astype(float)
    return {
        "noop_to_active": [
            f"pre_bout_active_is_{action}" for action in range(2, N_ACTIONS)
        ] + [f"previous_incumbent_is_{action}" for action in range(1, N_ACTIONS)],
        "active_to_noop": [
            f"incumbent_is_{action}" for action in range(2, N_ACTIONS)
        ] + [f"previous_incumbent_is_{action}" for action in range(1, N_ACTIONS)],
    }


def fit_stage3_loro(
    summary: pd.DataFrame,
    visual: np.ndarray,
    subject: int,
    game: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """Transition-specific LORO B0 -> B3 on one identical held-out row set."""
    frame = summary.copy().reset_index(drop=True)
    actor_rows = pd.to_numeric(frame["actor_row"], errors="raise").to_numpy(dtype=int)
    visual_names = [f"visual_state_{index:02d}" for index in range(visual.shape[1])]
    visual_frame = pd.DataFrame(visual[actor_rows], columns=visual_names, index=frame.index)
    frame = pd.concat([frame, visual_frame], axis=1)
    frame["log_bout_age"] = np.log1p(pd.to_numeric(frame["bout_age"], errors="coerce").clip(lower=0))
    frame["log_bout_age_sq"] = frame["log_bout_age"] ** 2
    frame["run_progress_sq"] = pd.to_numeric(frame["run_progress"], errors="coerce") ** 2
    frame["revision_available"] = np.isfinite(pd.to_numeric(frame["R_log"], errors="coerce")).astype(float)
    action_controls = _stage3_action_controls(frame)
    common_b0 = [
        "transition_margin_M", "R_log", "revision_available", "prior_reward",
        "prior_done", "log_bout_age", "log_bout_age_sq", "run_progress",
        "run_progress_sq", "value_revision_event_fraction",
    ] + visual_names
    block1 = [
        "aligned_E_mean", "aligned_E_slope", "aligned_signed_q_mean",
        "aligned_signed_q_slope", "directed_q_mean", "directed_q_slope",
        "signed_q_available", "signed_q_event_count",
    ]
    block2 = ["destination_share_shift", "destination_depth_shift"]
    block3 = ["actor_residual_beyond_P", "actor_geometry_transition_increment"]
    incumbent = pd.to_numeric(frame["incumbent_action"], errors="coerce").fillna(-1).to_numpy(dtype=int)
    human = pd.to_numeric(frame["human_action"], errors="coerce").fillna(-1).to_numpy(dtype=int)
    run_array = frame["run_label"].astype(str).to_numpy()
    fold_rows: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []
    coefficient_rows: List[Dict[str, object]] = []
    qa: Dict[str, object] = {
        "preprocessing": "train-run median imputation and z-scaling only",
        "regularization": "fixed a-priori L2 logistic C=1.0; no held-out tuning",
        "identical_rows_across_B0_B3": True,
        "minimum_train_rows": STAGE3_MIN_TRAIN_ROWS,
        "risk_sets": {},
        "features_by_transition": {},
    }
    for transition in TRANSITION_LABELS:
        if transition == "noop_to_active":
            risk = (incumbent == 0) & (human >= 0) & (human < N_ACTIONS)
            outcome = (human != 0).astype(int)
            competing = np.zeros(len(frame), dtype=bool)
        else:
            valid_active = (incumbent >= 1) & (incumbent < N_ACTIONS)
            valid_human = (human >= 0) & (human < N_ACTIONS)
            competing = valid_active & valid_human & (human != 0) & (human != incumbent)
            risk = valid_active & valid_human & ~competing
            outcome = (human == 0).astype(int)
        b0 = common_b0 + action_controls[transition]
        feature_sets = {
            "B0": b0,
            "B1": b0 + block1,
            "B2": b0 + block1 + block2,
            "B3": b0 + block1 + block2 + block3,
        }
        qa["features_by_transition"][transition] = feature_sets
        imputed = {
            "R_log", "aligned_signed_q_mean", "aligned_signed_q_slope",
            "directed_q_mean", "directed_q_slope",
        }
        required = [name for name in feature_sets["B3"] if name not in imputed]
        finite_required = frame[required].replace([np.inf, -np.inf], np.nan).notna().all(axis=1).to_numpy()
        complete = risk & finite_required
        runs = frame.loc[complete, "run_label"].astype(str).drop_duplicates().tolist()
        if len(runs) != 11:
            raise RuntimeError(f"{transition} expected 11 LORO folds, observed {len(runs)}")
        oof_count = np.zeros(len(frame), dtype=np.int8)
        qa["risk_sets"][transition] = {
            "risk_rows": int(np.sum(risk)),
            "complete_rows": int(np.sum(complete)),
            "event_rows": int(np.sum(outcome[complete])),
            "competing_active_to_active_excluded": int(np.sum(competing)),
            "R_missing_complete_rows": int(np.sum(complete & ~np.isfinite(frame["R_log"]))),
            "signed_q_missing_complete_rows": int(np.sum(complete & ~np.isfinite(frame["aligned_signed_q_mean"]))),
        }
        for held_out in runs:
            test = complete & (run_array == held_out)
            train = complete & (run_array != held_out)
            if np.sum(test) == 0 or np.sum(train) < STAGE3_MIN_TRAIN_ROWS or len(np.unique(outcome[train])) != 2:
                raise RuntimeError(
                    f"Invalid {transition} fold {held_out}: test={int(np.sum(test))}, "
                    f"train={int(np.sum(train))}, minimum_train={STAGE3_MIN_TRAIN_ROWS}, "
                    f"train_classes={np.unique(outcome[train]).tolist()}"
                )
            probabilities: Dict[str, np.ndarray] = {}
            losses: Dict[str, np.ndarray] = {}
            for stage, names in feature_sets.items():
                fit = _fit_binary_l2(
                    frame.loc[train, names].to_numpy(dtype=float), outcome[train]
                )
                if fit is None:
                    raise RuntimeError(f"Failed {transition}/{held_out}/{stage} L2 fit")
                probabilities[stage] = _predict_binary(
                    fit, frame.loc[test, names].to_numpy(dtype=float)
                )
                losses[stage] = binary_log_loss(outcome[test], probabilities[stage])
                model = fit["model"]
                scaler = fit["scaler"]
                for feature_index, feature in enumerate(names):
                    coefficient_rows.append({
                        "subject": subject, "game": game, "run_label": held_out,
                        "transition_type": transition, "stage": stage,
                        "feature": feature,
                        "standardized_beta": float(model.coef_[0, feature_index]),
                        "train_mean": float(scaler.mean_[feature_index]),
                        "train_scale": float(scaler.scale_[feature_index]),
                        "train_impute": float(fit["impute"][feature_index]),
                    })
            mean_losses = {stage: float(np.mean(value)) for stage, value in losses.items()}
            for start, end in (("B0", "B1"), ("B1", "B2"), ("B2", "B3")):
                fold_rows.append({
                    "subject": subject, "game": game, "run_label": held_out,
                    "transition_type": transition,
                    "increment": f"{start}_to_{end}",
                    "effect_id": f"stage3::{transition}::{start}_to_{end}",
                    "beta": mean_losses[start] - mean_losses[end],
                    "start_log_loss": mean_losses[start],
                    "end_log_loss": mean_losses[end],
                    **{f"{stage}_log_loss": value for stage, value in mean_losses.items()},
                    "n_test": int(np.sum(test)), "test_events": int(np.sum(outcome[test])),
                    "n_train": int(np.sum(train)), "train_events": int(np.sum(outcome[train])),
                    "primary": 1,
                })
            test_indices = np.flatnonzero(test)
            for local_index, row_index in enumerate(test_indices):
                prediction_rows.append({
                    "actor_row": int(frame.loc[row_index, "actor_row"]),
                    "subject": subject, "game": game, "run_label": held_out,
                    "transition_type": transition, "outcome": int(outcome[row_index]),
                    **{f"{stage}_probability": float(values[local_index]) for stage, values in probabilities.items()},
                })
            oof_count[test] += 1
        if not np.all(oof_count[complete] == 1) or np.any(oof_count[~complete] != 0):
            raise RuntimeError(f"{transition} OOF coverage is not exactly once")
        qa["risk_sets"][transition]["oof_rows"] = int(np.sum(oof_count == 1))
        qa["risk_sets"][transition]["observed_folds"] = int(len(runs))
    folds = pd.DataFrame(fold_rows)
    expected = 11 * len(STAGE3_PRIMARY_IDS)
    if len(folds) != expected or not np.all(np.isfinite(folds["beta"])):
        raise RuntimeError(f"Stage-3 fold grid/finite invariant failed: {len(folds)} != {expected}")
    return folds, pd.DataFrame(prediction_rows), pd.DataFrame(coefficient_rows), qa


def exact_unit_inference(
    effects: pd.DataFrame,
    effect_ids: Sequence[str],
    unit_column: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    means = effects.groupby([unit_column, "effect_id"], sort=False, observed=True)["beta"].mean().unstack("effect_id")
    means = means.reindex(columns=list(effect_ids))
    if means.isna().any().any() or len(means) < 2:
        # Keep the prespecified synchronized family intact. Missing effects are
        # not zero effects, and do not justify changing the family or its units.
        inference = pd.DataFrame({"mean_beta": means.mean().to_numpy(),
            "studentized_t": np.nan, "exact_sign_flip_p": np.nan, "synchronized_max_t_p": np.nan,
            "n_positive": (means > 0).sum().to_numpy(), "n_units": means.notna().sum().to_numpy(),
            "n_sign_patterns": 0, "not_estimable_reason": "incomplete prespecified effect family"})
    else:
        inference = synchronized_sign_flip(means.to_numpy(dtype=float))
    inference.insert(0, "effect_id", list(effect_ids))
    long = means.reset_index().melt(id_vars=unit_column, var_name="effect_id", value_name="beta")
    return inference, long


def _mean_ci(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=float)
    mean = np.full(array.shape[1], np.nan, dtype=float)
    half = np.full(array.shape[1], np.nan, dtype=float)
    for column in range(array.shape[1]):
        finite = array[np.isfinite(array[:, column]), column]
        if len(finite):
            mean[column] = float(np.mean(finite))
        if len(finite) > 1:
            half[column] = float(
                stats.t.ppf(0.975, len(finite) - 1)
                * np.std(finite, ddof=1) / math.sqrt(len(finite))
            )
    return mean, mean - half, mean + half


def _inference_lookup(inference: Optional[pd.DataFrame]) -> Dict[str, float]:
    if inference is None or inference.empty:
        return {}
    return dict(
        zip(
            inference["effect_id"].astype(str),
            pd.to_numeric(inference["synchronized_max_t_p"], errors="coerce"),
        )
    )


def plot_stage1_relationships(
    curves: pd.DataFrame,
    output: Path,
    *,
    unit_column: str,
    title: str,
    inference: Optional[pd.DataFrame] = None,
) -> None:
    """Small-multiple response curves; not a coefficient-ranking forest plot."""
    configure_style()
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 9.2), sharex=True)
    p_values = _inference_lookup(inference)
    colors = {"aligned_E": BLUE, "aligned_signed_q": ORANGE}
    light = {"aligned_E": BLUE_LIGHT, "aligned_signed_q": ORANGE_LIGHT}
    colors["directed_q"] = GREY
    light["directed_q"] = "#D6D8DB"
    for row_index, transition in enumerate(TRANSITION_LABELS):
        for column_index, outcome in enumerate(OUTCOME_LABELS):
            axis = axes[row_index, column_index]
            panel = curves.loc[
                curves["transition_type"].eq(transition)
                & curves["outcome"].eq(outcome)
                & curves["margin_group"].eq("all")
            ].copy()
            for predictor in PREDICTOR_LABELS:
                predictor_data = panel.loc[panel["predictor"].eq(predictor)]
                unit_curves: List[np.ndarray] = []
                for _, unit in predictor_data.groupby(unit_column, sort=False):
                    points = unit.groupby("bin", observed=True).agg(
                        x=("predictor_z", "mean"),
                        y=("future_geometry_within_window", "mean"),
                    ).reindex(range(5))
                    if points["y"].notna().all():
                        axis.plot(
                            np.arange(5, dtype=float), points["y"], color=light[predictor],
                            alpha=0.45, linewidth=0.9, zorder=1,
                        )
                        unit_curves.append(points["y"].to_numpy(dtype=float))
                if unit_curves:
                    matrix = np.asarray(unit_curves)
                    mean, low, high = _mean_ci(matrix)
                    x_axis = np.arange(5, dtype=float)
                    axis.fill_between(x_axis, low, high, color=colors[predictor], alpha=0.14, linewidth=0)
                    label = PREDICTOR_LABELS[predictor]
                    if p_values:
                        prefix = "stage1" if predictor != "directed_q" else "stage1_sensitivity"
                        effect_id = f"{prefix}::{transition}::{predictor}::{outcome}"
                        label += f"  {significance_stars(p_values.get(effect_id, np.nan))}"
                    axis.plot(x_axis, mean, color=colors[predictor], linewidth=2.6, marker="o", markersize=4.5, label=label, zorder=3)
            axis.axhline(0, color=GREY, linestyle="--", linewidth=0.9)
            axis.set_title(f"{TRANSITION_LABELS[transition]} | {OUTCOME_LABELS[outcome]}")
            axis.set_xticks(range(5), ["low", "", "mid", "", "high"])
            axis.set_xlabel("Within-window aligned predictor quintile")
            axis.set_ylabel("Future geometry\n(within-window deviation)")
            if row_index == 0 and column_index == 0:
                axis.legend(frameon=False, loc="best", fontsize=8)
    figure.suptitle(title, fontsize=15, fontweight="bold", y=0.995)
    figure.text(
        0.5, 0.01,
        "Thin lines: runs/subjects. Bold line and shading: mean +/- 95% CI. "
        "Inferential betas additionally control event position, branch, P/Q history, depth, and prefix geometry.",
        ha="center", va="bottom", fontsize=8.5, color=INK,
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_stage1_margin_moderation(
    curves: pd.DataFrame,
    output: Path,
    *,
    unit_column: str,
    title: str,
    inference: Optional[pd.DataFrame] = None,
) -> None:
    """Low/high pre-search M response curves for the event-level interactions."""
    configure_style()
    figure, axes = plt.subplots(2, 3, figsize=(15.0, 8.8), sharex=True)
    p_values = _inference_lookup(inference)
    margin_colors = {"low_M": ORANGE, "high_M": BLUE}
    outcome_styles = {
        "future_destination_share_contrast": "-",
        "future_destination_depth_score": "--",
    }
    interaction_names = {
        "aligned_E": "M_x_E", "aligned_signed_q": "M_x_signed_q",
        "directed_q": "M_x_directed_q",
    }
    for row_index, transition in enumerate(TRANSITION_LABELS):
        for column_index, predictor in enumerate(PREDICTOR_LABELS):
            axis = axes[row_index, column_index]
            panel = curves.loc[
                curves["transition_type"].eq(transition)
                & curves["predictor"].eq(predictor)
                & curves["margin_group"].isin(["low_M", "high_M"])
            ]
            for margin_group in ("low_M", "high_M"):
                for outcome in OUTCOME_LABELS:
                    data = panel.loc[
                        panel["margin_group"].eq(margin_group)
                        & panel["outcome"].eq(outcome)
                    ]
                    unit_curves: List[np.ndarray] = []
                    for _, unit in data.groupby(unit_column, sort=False):
                        points = unit.groupby("bin", observed=True)[
                            "future_geometry_within_window"
                        ].mean().reindex(range(5))
                        if points.notna().all():
                            unit_curves.append(points.to_numpy(dtype=float))
                            axis.plot(
                                range(5), points, color=margin_colors[margin_group],
                                linestyle=outcome_styles[outcome], alpha=0.12,
                                linewidth=0.75,
                            )
                    if unit_curves:
                        matrix = np.asarray(unit_curves)
                        mean, low, high = _mean_ci(matrix)
                        axis.fill_between(
                            range(5), low, high, color=margin_colors[margin_group],
                            alpha=0.08, linewidth=0,
                        )
                        outcome_short = "share" if "share" in outcome else "depth"
                        margin_short = "Low M" if margin_group == "low_M" else "High M"
                        axis.plot(
                            range(5), mean, color=margin_colors[margin_group],
                            linestyle=outcome_styles[outcome], linewidth=2.0,
                            label=f"{margin_short} | {outcome_short}",
                        )
            if p_values:
                interaction = interaction_names[predictor]
                share_id = f"stage1_moderation::{transition}::{interaction}::future_destination_share_contrast"
                depth_id = f"stage1_moderation::{transition}::{interaction}::future_destination_depth_score"
                axis.text(
                    0.03, 0.97,
                    f"M x predictor: share {significance_stars(p_values.get(share_id, np.nan))}; "
                    f"depth {significance_stars(p_values.get(depth_id, np.nan))}",
                    transform=axis.transAxes, ha="left", va="top", fontsize=8,
                    bbox={"facecolor": "white", "edgecolor": GRID, "alpha": 0.85, "pad": 3},
                )
            axis.axhline(0, color=GREY, linestyle=":", linewidth=0.8)
            axis.set_xticks(range(5), ["low", "", "mid", "", "high"])
            axis.set_xlabel("Aligned predictor quintile")
            axis.set_ylabel("Future geometry deviation")
            axis.set_title(f"{TRANSITION_LABELS[transition]} | {PREDICTOR_LABELS[predictor]}")
            if row_index == 0 and column_index == 0:
                axis.legend(frameon=False, fontsize=7, ncol=2, loc="lower left")
    figure.suptitle(title, fontsize=15, fontweight="bold", y=0.995)
    figure.text(
        0.5, 0.012,
        "Orange/blue separate below/above-run-mean pre-search M. Solid=destination share; dashed=destination-weighted depth. Group stars test M x event predictor within the moderation family.",
        ha="center", fontsize=8.5,
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_stage2_reallocation(
    run_effects: pd.DataFrame,
    output: Path,
    *,
    unit_column: str,
    title: str,
    inference: Optional[pd.DataFrame] = None,
) -> None:
    configure_style()
    figure, axes = plt.subplots(2, 2, figsize=(11.8, 9.0))
    p_values = _inference_lookup(inference)
    metrics = (
        ("destination_share_shift", "Destination-share contrast"),
        ("destination_depth_shift", "Destination-weighted depth"),
    )
    for row_index, transition in enumerate(TRANSITION_LABELS):
        for column_index, (metric, label) in enumerate(metrics):
            axis = axes[row_index, column_index]
            panel = run_effects.loc[
                run_effects["transition_type"].eq(transition)
                & run_effects["metric"].eq(metric)
            ].copy()
            unit = panel.groupby(unit_column, sort=False).agg(
                early=("early_mean", "mean"), late=("late_mean", "mean")
            )
            for _, values in unit.iterrows():
                axis.plot([0, 1], [values["early"], values["late"]], color=PURPLE_LIGHT, linewidth=1.2, alpha=0.7)
                axis.scatter([0, 1], [values["early"], values["late"]], color=PURPLE, s=18, alpha=0.65, zorder=2)
            matrix = unit[["early", "late"]].to_numpy(dtype=float)
            mean, low, high = _mean_ci(matrix)
            axis.plot([0, 1], mean, color=PURPLE, linewidth=3.0, marker="o", markersize=7, zorder=4)
            axis.errorbar([0, 1], mean, yerr=[mean - low, high - mean], fmt="none", color=PURPLE, capsize=4, linewidth=1.5, zorder=4)
            effect_id = f"stage2::{transition}::{metric}"
            star = significance_stars(p_values.get(effect_id, np.nan)) if p_values else ""
            if star:
                upper = float(np.nanmax(matrix))
                lower = float(np.nanmin(matrix))
                y = upper + max((upper - lower) * 0.10, 0.02)
                axis.plot([0, 0, 1, 1], [y, y + 0.01, y + 0.01, y], color=INK, linewidth=1)
                axis.text(0.5, y + 0.012, star, ha="center", va="bottom", fontweight="bold")
            axis.axhline(0, color=GREY, linestyle="--", linewidth=0.9)
            axis.set_xticks([0, 1], ["early\nfirst half", "late\nsecond half"])
            axis.set_xlim(-0.25, 1.25)
            axis.set_title(f"{TRANSITION_LABELS[transition]} | {label}")
            axis.set_ylabel("Destination-aligned geometry")
    figure.suptitle(title, fontsize=15, fontweight="bold", y=0.995)
    figure.text(0.5, 0.012, "Each thin paired line is one run/subject; bold line is the equal-unit mean with 95% CI. Stars use stage-family exact synchronized max-|t| only.", ha="center", fontsize=8.5)
    figure.tight_layout(rect=(0, 0.04, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_stage3_nested_loss(
    folds: pd.DataFrame,
    output: Path,
    *,
    unit_column: str,
    title: str,
    inference: Optional[pd.DataFrame] = None,
) -> None:
    configure_style()
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 5.5), sharey=False)
    p_values = _inference_lookup(inference)
    stages = ["B0", "B1", "B2", "B3"]
    for axis, transition in zip(axes, TRANSITION_LABELS):
        panel = folds.loc[folds["transition_type"].eq(transition)].drop_duplicates([unit_column, "run_label"] if unit_column != "run_label" and "run_label" in folds else [unit_column])
        if unit_column != "run_label":
            unit = panel.groupby(unit_column, sort=False)[[f"{stage}_log_loss" for stage in stages]].mean()
        else:
            unit = panel.set_index(unit_column)[[f"{stage}_log_loss" for stage in stages]]
        matrix = unit.to_numpy(dtype=float)
        for values in matrix:
            axis.plot(range(4), values, color=BLUE_LIGHT, linewidth=1.0, alpha=0.65)
            axis.scatter(range(4), values, color=BLUE, s=15, alpha=0.55, zorder=2)
        mean, low, high = _mean_ci(matrix)
        axis.fill_between(range(4), low, high, color=BLUE, alpha=0.14, linewidth=0)
        axis.plot(range(4), mean, color=BLUE, linewidth=3.0, marker="o", markersize=6, zorder=4)
        axis.set_xticks(range(4), [STAGE3_LABELS[stage] for stage in stages])
        axis.set_ylabel("Held-out binary log loss (lower is better)")
        axis.set_title(TRANSITION_LABELS[transition])
        if p_values:
            span = max(float(np.nanmax(matrix) - np.nanmin(matrix)), 0.001)
            for index, increment in enumerate(("B0_to_B1", "B1_to_B2", "B2_to_B3")):
                effect_id = f"stage3::{transition}::{increment}"
                star = significance_stars(p_values.get(effect_id, np.nan))
                y = max(mean[index], mean[index + 1]) + span * (0.08 + 0.045 * (index % 2))
                axis.plot([index, index + 1], [y, y], color=INK, linewidth=1)
                axis.text(index + 0.5, y + span * 0.015, star, ha="center", va="bottom", fontsize=9, fontweight="bold")
    figure.suptitle(title, fontsize=15, fontweight="bold")
    figure.text(0.5, 0.015, "Thin trajectories: held-out runs/subjects. Every B0-B3 point uses the identical eligible rows. Group stars use the six-subject stage-family exact max-|t| test.", ha="center", fontsize=8.5)
    figure.tight_layout(rect=(0, 0.06, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _subject_manifest(
    *,
    subject: int,
    game: int,
    source_sha: str,
    shared_manifest: Mapping[str, object],
    derived_qa: Mapping[str, object],
    suffix_qa: Sequence[Mapping[str, object]],
    window_qa: Mapping[str, object],
    stage3_qa: Mapping[str, object],
    tables: Mapping[str, Path],
) -> Dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_version": ANALYSIS_VERSION,
        "actor_softmax_maxiter": ACTOR_SOFTMAX_MAXITER,
        "stage3_min_train_rows": STAGE3_MIN_TRAIN_ROWS,
        "script": str(Path(__file__).resolve()),
        "script_sha256": source_sha,
        "self_contained_v2": True,
        "v1_source_required": False,
        "subject": int(subject),
        "game": int(game),
        "n_runs": 11,
        "primary_effect_families": {
            "stage1": list(STAGE1_PRIMARY_IDS),
            "stage1_moderation": list(STAGE1_MODERATION_IDS),
            "stage1_sensitivity": list(STAGE1_SENSITIVITY_IDS),
            "stage2": list(STAGE2_PRIMARY_IDS),
            "stage3": list(STAGE3_PRIMARY_IDS),
        },
        "chart_contracts": CHART_CONTRACTS,
        "group_inference": (
            "exact synchronized six-subject sign flips; separate stage-wise "
            "max-|studentized t| FWER families"
        ),
        "subject_run_inference_role": "diagnostic only; subject is the population unit",
        "significance_star_policy": {
            "group_only": True,
            "thresholds": {"*": "p<.05", "**": "p<.01", "***": "p<.001"},
            "n6_exact_two_sided_floor": 0.03125,
        },
        "alignment_policy": (
            "incumbent-defined origin/destination status contrast; eventual human "
            "action is outcome only and never used to orient E, dQ, or dG"
        ),
        "dynamic_window_policy": {"minimum_future_events": MIN_FUTURE,
            "early_late_split": "L//2", "minimum_informative_windows": 40,
            "nonestimable_effects": "NaN with reason; no zero filling or changed inference families"},
        "causal_claim": False,
        "shared_cache_manifest": dict(shared_manifest),
        "derived_features": dict(derived_qa),
        "suffix_run_qa": list(suffix_qa),
        "window_summary_qa": dict(window_qa),
        "stage3_qa": dict(stage3_qa),
        "tables": {
            name: {"path": str(path), "sha256": sha256_path(path)}
            for name, path in tables.items()
        },
    }


def run_subject_analysis(args: argparse.Namespace) -> Path:
    subject = int(args.subject)
    game = int(args.game)
    output = subject_output_dir(args.output_root, subject, game)
    manifest_path = output / "analysis_manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"Subject output already exists; use --overwrite: {output}")
    source_sha = sha256_path(Path(__file__).resolve())
    log(f"load canonical cache: subject={subject:03d}, game={game}")
    _, shared_manifest, windows, events, rollout_keys, _, prefix, cache_audit = load_subject_cache(
        subject, game, args.shared_cache_root
    )
    run_labels = windows["run_label"].astype(str).drop_duplicates().tolist()
    if len(run_labels) != 11:
        raise RuntimeError(f"Expected 11 runs, observed {len(run_labels)}")
    event_frame = attach_event_evidence(events, rollout_keys, prefix)
    del events, rollout_keys, prefix
    if args.derived_feature_source == "basic-cache":
        log("derive Actor OOF and visual controls from canonical cache (native Zarr when available)")
        actor_prediction, visual, derived_qa = (
            prepare_actor_and_visual_from_basic_cache(
                subject=subject,
                game=game,
                windows=windows,
                event_frame=event_frame,
                shared_manifest=shared_manifest,
                behavior_root=args.behavior_root,
                output=output,
                rebuild=args.rebuild_derived_features,
            )
        )
    else:
        log("compatibility mode: load accepted-04 Actor/visual derivatives")
        actor_prediction, visual, legacy_qa = load_legacy_actor_and_visual(
            subject, game, windows, args.legacy_04_output_root
        )
        derived_qa = {
            "source": "accepted_04_output_compatibility_sensitivity",
            "neural_cache_used": False,
            "accepted_04_output_used": True,
            **legacy_qa,
        }
    stage1_parts: List[pd.DataFrame] = []
    curve_parts: List[pd.DataFrame] = []
    suffix_qa: List[Mapping[str, object]] = []
    for index, run_label in enumerate(run_labels, 1):
        log(f"stage 1 event->future geometry: {index}/11 {run_label}")
        run_events = event_frame.loc[event_frame["run_label"].astype(str).eq(run_label)]
        suffix, run_qa = build_transition_suffix_run(run_events, windows)
        effects, curves = fit_stage1_run(suffix, subject, game)
        stage1_parts.append(effects)
        curve_parts.append(curves)
        suffix_qa.append(run_qa)
        del suffix
    stage1 = pd.concat(stage1_parts, ignore_index=True)
    curves = pd.concat(curve_parts, ignore_index=True)
    expected_stage1 = 11 * (
        len(STAGE1_PRIMARY_IDS)
        + len(STAGE1_MODERATION_IDS)
        + len(STAGE1_SENSITIVITY_IDS)
    )
    if len(stage1) != expected_stage1:
        raise RuntimeError(f"Stage-1 grid/finite invariant failed: {len(stage1)} != {expected_stage1}")
    log("stage 2 aligned early->late window summaries")
    window_summary, window_qa = build_window_pathway_summary(
        event_frame, windows, actor_prediction
    )
    stage2 = summarize_stage2_runs(window_summary, subject, game)
    del event_frame
    log("stage 3 transition-specific B0->B3 LORO")
    stage3, transition_predictions, coefficients, stage3_qa = fit_stage3_loro(
        window_summary, visual, subject, game
    )
    primary_stage1 = stage1.loc[stage1["primary"].eq(1)].copy()
    subject_stage1, stage1_long = exact_unit_inference(primary_stage1, STAGE1_PRIMARY_IDS, "run_label")
    subject_stage1_moderation, stage1_moderation_long = exact_unit_inference(
        stage1.loc[stage1["effect_id"].isin(STAGE1_MODERATION_IDS)],
        STAGE1_MODERATION_IDS, "run_label",
    )
    subject_stage1_sensitivity, stage1_sensitivity_long = exact_unit_inference(
        stage1.loc[stage1["effect_id"].isin(STAGE1_SENSITIVITY_IDS)],
        STAGE1_SENSITIVITY_IDS, "run_label",
    )
    subject_stage2, stage2_long = exact_unit_inference(stage2, STAGE2_PRIMARY_IDS, "run_label")
    subject_stage3, stage3_long = exact_unit_inference(stage3, STAGE3_PRIMARY_IDS, "run_label")
    subject_stage1["stage"] = "stage1"
    subject_stage1_moderation["stage"] = "stage1_moderation"
    subject_stage1_sensitivity["stage"] = "stage1_sensitivity"
    subject_stage2["stage"] = "stage2"
    subject_stage3["stage"] = "stage3"
    subject_inference = pd.concat(
        [subject_stage1, subject_stage1_moderation, subject_stage1_sensitivity, subject_stage2, subject_stage3],
        ignore_index=True,
    )
    output.mkdir(parents=True, exist_ok=True)
    table_root = output / "tables"
    tables = {
        "stage1_run_effects": table_root / "stage1_event_to_future_geometry_run_effects.csv",
        "stage1_curves": table_root / "stage1_quantile_relationship_curves.csv.gz",
        "stage2_window_summary": table_root / "stage2_window_pathway_summary.csv.gz",
        "stage2_run_effects": table_root / "stage2_early_late_run_effects.csv",
        "stage3_fold_effects": table_root / "stage3_nested_loro_fold_effects.csv",
        "stage3_predictions": table_root / "stage3_nested_loro_predictions.csv.gz",
        "stage3_coefficients": table_root / "stage3_nested_loro_coefficients.csv.gz",
        "subject_diagnostic_inference": table_root / "subject_run_signflip_diagnostic.csv",
        "subject_effect_matrix": table_root / "subject_run_effect_matrix_long.csv",
    }
    _write_table(stage1, tables["stage1_run_effects"])
    (output / "qa").mkdir(exist_ok=True)
    _write_table(stage1.loc[~stage1["estimable"].fillna(False)], output / "qa" / "stage1_not_estimable.csv")
    _write_table(curves, tables["stage1_curves"])
    _write_table(window_summary, tables["stage2_window_summary"])
    _write_table(stage2, tables["stage2_run_effects"])
    _write_table(stage3, tables["stage3_fold_effects"])
    _write_table(transition_predictions, tables["stage3_predictions"])
    _write_table(coefficients, tables["stage3_coefficients"])
    _write_table(subject_inference, tables["subject_diagnostic_inference"])
    effect_long = pd.concat(
        [
            stage1_long.assign(stage="stage1"),
            stage1_moderation_long.assign(stage="stage1_moderation"),
            stage1_sensitivity_long.assign(stage="stage1_sensitivity"),
            stage2_long.assign(stage="stage2"), stage3_long.assign(stage="stage3"),
        ],
        ignore_index=True,
    )
    _write_table(effect_long, tables["subject_effect_matrix"])
    figure_root = output / "figures"
    plot_stage1_relationships(
        curves, figure_root / "figure1_event_evidence_to_future_geometry.png",
        unit_column="run_label",
        title=f"Subject {subject:03d} | Game {game} | Stage 1: event evidence -> future geometry",
    )
    plot_stage1_margin_moderation(
        curves, figure_root / "figure1b_presearch_margin_moderation.png",
        unit_column="run_label",
        title=f"Subject {subject:03d} | Game {game} | Stage 1b: pre-search M moderation",
    )
    plot_stage2_reallocation(
        stage2, figure_root / "figure2_early_to_late_reallocation.png",
        unit_column="run_label",
        title=f"Subject {subject:03d} | Game {game} | Stage 2: transition-aligned reallocation",
    )
    plot_stage3_nested_loss(
        stage3, figure_root / "figure3_nested_human_transition_prediction.png",
        unit_column="run_label",
        title=f"Subject {subject:03d} | Game {game} | Stage 3: nested held-out transition prediction",
    )
    if sha256_path(Path(__file__).resolve()) != source_sha:
        raise RuntimeError("v2 source changed while subject analysis was running")
    manifest = _subject_manifest(
        subject=subject, game=game, source_sha=source_sha,
        shared_manifest={"manifest": shared_manifest, "load_audit": cache_audit},
        derived_qa=derived_qa, suffix_qa=suffix_qa, window_qa=window_qa,
        stage3_qa=stage3_qa, tables=tables,
    )
    atomic_json(manifest_path, manifest)
    log(f"subject complete: {output}")
    return output


def run_group_analysis(args: argparse.Namespace) -> Path:
    subjects = [int(value) for value in args.aggregate_subjects]
    if subjects != [1, 2, 3, 4, 5, 6]:
        raise ValueError("Group analysis requires exactly --aggregate-subjects 1 2 3 4 5 6")
    game = int(args.game)
    source_sha = sha256_path(Path(__file__).resolve())
    group_output = args.output_root / f"group_game{game}"
    manifest_path = group_output / "analysis_manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"Group output already exists; use --overwrite: {group_output}")
    all_stage1: List[pd.DataFrame] = []
    all_curves: List[pd.DataFrame] = []
    all_stage2: List[pd.DataFrame] = []
    all_stage3: List[pd.DataFrame] = []
    input_manifest_rows: List[Dict[str, object]] = []
    derived_sources: List[str] = []
    for subject in subjects:
        directory = subject_output_dir(args.output_root, subject, game)
        subject_manifest_path = directory / "analysis_manifest.json"
        manifest = json.loads(subject_manifest_path.read_text(encoding="utf-8"))
        if (
            int(manifest.get("subject", -1)) != subject
            or int(manifest.get("game", -1)) != game
            or manifest.get("script_sha256") != source_sha
            or manifest.get("self_contained_v2") is not True
            or manifest.get("v1_source_required") is not False
            or int(manifest.get("n_runs", -1)) != 11
        ):
            raise RuntimeError(f"Incompatible v2 subject manifest: {subject_manifest_path}")
        derived = manifest.get("derived_features")
        if not isinstance(derived, Mapping):
            raise RuntimeError(f"Missing derived-feature provenance: {subject_manifest_path}")
        derived_source = str(derived.get("source", ""))
        if derived_source not in {
            "native_zarr_cache",
            "canonical_shared_cache_plus_behavioral_npz",
            "accepted_04_output_compatibility_sensitivity",
        }:
            raise RuntimeError(f"Unknown derived-feature source: {subject_manifest_path}")
        derived_sources.append(derived_source)
        table_root = directory / "tables"
        all_stage1.append(pd.read_csv(table_root / "stage1_event_to_future_geometry_run_effects.csv"))
        all_curves.append(pd.read_csv(table_root / "stage1_quantile_relationship_curves.csv.gz"))
        all_stage2.append(pd.read_csv(table_root / "stage2_early_late_run_effects.csv"))
        all_stage3.append(pd.read_csv(table_root / "stage3_nested_loro_fold_effects.csv"))
        input_manifest_rows.append({
            "subject": subject, "path": str(subject_manifest_path),
            "sha256": sha256_path(subject_manifest_path),
            "derived_feature_source": derived_source,
        })
    if len(set(derived_sources)) != 1:
        raise RuntimeError(
            "Refusing to aggregate mixed basic-cache and accepted-04 derivatives"
        )
    stage1 = pd.concat(all_stage1, ignore_index=True)
    curves = pd.concat(all_curves, ignore_index=True)
    stage2 = pd.concat(all_stage2, ignore_index=True)
    stage3 = pd.concat(all_stage3, ignore_index=True)
    stage1_inference, stage1_matrix = exact_unit_inference(
        stage1.loc[stage1["primary"].eq(1)], STAGE1_PRIMARY_IDS, "subject"
    )
    stage1_moderation_inference, stage1_moderation_matrix = exact_unit_inference(
        stage1.loc[stage1["effect_id"].isin(STAGE1_MODERATION_IDS)],
        STAGE1_MODERATION_IDS, "subject",
    )
    stage1_sensitivity_inference, stage1_sensitivity_matrix = exact_unit_inference(
        stage1.loc[stage1["effect_id"].isin(STAGE1_SENSITIVITY_IDS)],
        STAGE1_SENSITIVITY_IDS, "subject",
    )
    stage2_inference, stage2_matrix = exact_unit_inference(stage2, STAGE2_PRIMARY_IDS, "subject")
    stage3_inference, stage3_matrix = exact_unit_inference(stage3, STAGE3_PRIMARY_IDS, "subject")
    stage1_inference["stage"] = "stage1"
    stage1_moderation_inference["stage"] = "stage1_moderation"
    stage1_sensitivity_inference["stage"] = "stage1_sensitivity"
    stage2_inference["stage"] = "stage2"
    stage3_inference["stage"] = "stage3"
    inference = pd.concat(
        [stage1_inference, stage1_moderation_inference, stage1_sensitivity_inference, stage2_inference, stage3_inference],
        ignore_index=True,
    )
    matrices = pd.concat(
        [
            stage1_matrix.assign(stage="stage1"),
            stage1_moderation_matrix.assign(stage="stage1_moderation"),
            stage1_sensitivity_matrix.assign(stage="stage1_sensitivity"),
            stage2_matrix.assign(stage="stage2"), stage3_matrix.assign(stage="stage3"),
        ],
        ignore_index=True,
    )
    group_output.mkdir(parents=True, exist_ok=True)
    table_root = group_output / "tables"
    tables = {
        "group_exact_inference": table_root / "group_stagewise_exact_inference.csv",
        "group_subject_effect_matrix": table_root / "group_subject_effect_matrix_long.csv",
    }
    _write_table(inference, tables["group_exact_inference"])
    _write_table(matrices, tables["group_subject_effect_matrix"])
    figure_root = group_output / "figures"
    plot_stage1_relationships(
        curves, figure_root / "figure1_group_event_evidence_to_future_geometry.png",
        unit_column="subject", title=f"Game {game} | Stage 1: event evidence -> future geometry",
        inference=pd.concat([stage1_inference, stage1_sensitivity_inference], ignore_index=True),
    )
    plot_stage1_margin_moderation(
        curves, figure_root / "figure1b_group_presearch_margin_moderation.png",
        unit_column="subject", title=f"Game {game} | Stage 1b: pre-search M moderation",
        inference=stage1_moderation_inference,
    )
    plot_stage2_reallocation(
        stage2, figure_root / "figure2_group_early_to_late_reallocation.png",
        unit_column="subject", title=f"Game {game} | Stage 2: transition-aligned reallocation",
        inference=stage2_inference,
    )
    plot_stage3_nested_loss(
        stage3, figure_root / "figure3_group_nested_human_transition_prediction.png",
        unit_column="subject", title=f"Game {game} | Stage 3: nested held-out transition prediction",
        inference=stage3_inference,
    )
    if sha256_path(Path(__file__).resolve()) != source_sha:
        raise RuntimeError("v2 source changed during group aggregation")
    atomic_json(
        manifest_path,
        {
            "schema_version": SCHEMA_VERSION,
            "analysis_version": ANALYSIS_VERSION,
            "actor_softmax_maxiter": ACTOR_SOFTMAX_MAXITER,
            "stage3_min_train_rows": STAGE3_MIN_TRAIN_ROWS,
            "script": str(Path(__file__).resolve()),
            "script_sha256": source_sha,
            "self_contained_v2": True,
            "v1_source_required": False,
            "derived_feature_source": derived_sources[0],
            "neural_cache_used": False,
            "game": game,
            "subjects": subjects,
            "n_subjects": 6,
            "n_sign_patterns": 64,
            "family_control": "separate synchronized max-|studentized t| family per stage",
            "primary_effect_families": {
                "stage1": list(STAGE1_PRIMARY_IDS),
                "stage1_moderation": list(STAGE1_MODERATION_IDS),
                "stage1_sensitivity": list(STAGE1_SENSITIVITY_IDS),
                "stage2": list(STAGE2_PRIMARY_IDS),
                "stage3": list(STAGE3_PRIMARY_IDS),
            },
            "chart_contracts": CHART_CONTRACTS,
            "input_subject_manifests": input_manifest_rows,
            "tables": {name: {"path": str(path), "sha256": sha256_path(path)} for name, path in tables.items()},
        },
    )
    log(f"group complete: {group_output}")
    return group_output


def run_self_test() -> Dict[str, object]:
    n_sign = branch_alignment_sign(
        np.asarray([0, 1, 5]), np.asarray([0, 0, 0])
    )
    a_sign = branch_alignment_sign(
        np.asarray([0, 2, 4]), np.asarray([2, 2, 2])
    )
    if not np.array_equal(n_sign, np.asarray([-1.0, 1.0, 1.0])):
        raise AssertionError("N-origin branch alignment failed")
    if not np.array_equal(a_sign, np.asarray([1.0, -1.0, 0.0])):
        raise AssertionError("A-origin branch alignment failed")
    synthetic_windows = pd.DataFrame(
        {
            "run_label": ["synthetic", "synthetic"],
            "real_step": [1, 2], "real_step_id": [101, 102],
            "actor_row": [0, 1], "subject": [1, 1], "game": [1, 1],
            "incumbent_action": [0, 2], "human_action": [1, 0],
            "action_before_incumbent": [2, 2],
            "strongest_competitor_margin": [0.2, 0.3],
            "bout_age": [1.0, 2.0], "prior_reward": [0.0, 0.0],
            "prior_done": [0.0, 0.0], "run_progress": [0.1, 0.2],
            "value_revision_event_fraction": [0.5, 0.5],
        }
    )
    for action in range(N_ACTIONS):
        synthetic_windows[f"model_root_p_{action}"] = [1 / N_ACTIONS, 1 / N_ACTIONS]
        synthetic_windows[f"actor_terminal_p_{action}"] = [1 / N_ACTIONS, 1 / N_ACTIONS]
        synthetic_windows[f"root_q_initial_{action}"] = [float(action), float(action)]
        synthetic_windows[f"root_n_initial_{action}"] = [1.0, 1.0]
    event_rows: List[Dict[str, object]] = []
    for real_step, real_step_id in ((1, 101), (2, 102)):
        for position in range(N_EVENTS):
            event_rows.append(
                {
                    "run_label": "synthetic", "real_step": real_step,
                    "real_step_id": real_step_id, "status2_position": position,
                    "branch": position % N_ACTIONS, "depth": 1 + position % 4,
                    "return_advantage": float(position - 14),
                    "root_q_revision": float(position - 10) / 10,
                    "directed_value_revision": float(position - 10) / 10,
                    "valid_repeated_q_revision": 1,
                    "root_n_before_sampled": float(position + 1),
                }
            )
    synthetic_events = pd.DataFrame(event_rows)
    suffix_a, suffix_qa = build_transition_suffix_run(
        synthetic_events, synthetic_windows
    )
    synthetic_windows.loc[:, "human_action"] = [5, 4]
    suffix_b, _ = build_transition_suffix_run(synthetic_events, synthetic_windows)
    compared = [
        "aligned_E", "aligned_signed_q", "directed_q", "future_destination_share_contrast",
        "future_destination_depth_score",
    ]
    if len(suffix_a) != 2 * (N_EVENTS - MIN_FUTURE):
        raise AssertionError("Strict suffix synthetic row count failed")
    if not suffix_a[compared].equals(suffix_b[compared]):
        raise AssertionError("Human destination leaked into event/geometry alignment")
    if not suffix_qa["human_action_used_for_alignment"] is False:
        raise AssertionError("Alignment provenance flag failed")
    probability = np.asarray([[0.7, 0.1, 0.1, 0.05, 0.03, 0.02], [0.1, 0.05, 0.6, 0.1, 0.1, 0.05]])
    odds = actor_transition_log_odds(probability, np.asarray([0, 2]))
    expected_odds = np.asarray([math.log(0.3) - math.log(0.7), math.log(0.1) - math.log(0.6)])
    if not np.allclose(odds, expected_odds):
        raise AssertionError("Actor transition log-odds failed")
    matrix = np.arange(1, 7, dtype=float)[:, None] * np.ones((1, 3))
    exact = synchronized_sign_flip(matrix)
    if not np.allclose(exact["exact_sign_flip_p"], 0.03125):
        raise AssertionError("Six-subject exact p-value floor failed")
    if not np.all(exact["synchronized_max_t_p"] >= exact["exact_sign_flip_p"]):
        raise AssertionError("maxT must be no smaller than raw exact p")
    if significance_stars(0.03125) != "*" or significance_stars(0.0625) != "ns":
        raise AssertionError("Significance-star mapping failed")
    source_text = Path(__file__).read_text(encoding="utf-8")
    forbidden_dependencies = (
        "B" + "ASE.",
        "load_frozen" + "_base",
        "importlib" + ".util",
        "BASE" + "_SCRIPT",
    )
    if any(token in source_text for token in forbidden_dependencies):
        raise AssertionError("v2 still contains a v1 source dependency")
    synthetic_actor = pd.DataFrame(
        {
            "actor_row": [0, 1],
            "actor_loro_eligible": [1, 1],
            "actor_loro_oof_complete": [1, 1],
        }
    )
    for model in ("P", "A2", "A3"):
        for action in range(N_ACTIONS):
            synthetic_actor[f"actor_{model}_oof_p_{action}"] = 1.0 / N_ACTIONS
    _validate_actor_prediction(synthetic_actor, synthetic_windows)
    payload = {
        "status": "PASS",
        "self_contained_v2": True,
        "v1_source_required": False,
        "default_derived_feature_source": "basic-cache",
        "neural_cache_required": False,
        "accepted_04_output_required": False,
        "actor_prediction_contract_gate": True,
        "stage1_primary_effects": len(STAGE1_PRIMARY_IDS),
        "stage1_moderation_effects": len(STAGE1_MODERATION_IDS),
        "stage1_sensitivity_effects": len(STAGE1_SENSITIVITY_IDS),
        "stage2_primary_effects": len(STAGE2_PRIMARY_IDS),
        "stage3_primary_effects": len(STAGE3_PRIMARY_IDS),
        "n6_exact_two_sided_floor": 0.03125,
        "subject_figures_use_significance_stars": False,
        "group_stars_use_stagewise_synchronized_max_t": True,
        "strict_future_rows_per_window": N_EVENTS - MIN_FUTURE,
        "human_destination_leakage_gate": True,
        "chart_contracts": CHART_CONTRACTS,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--subject", type=int, choices=range(1, 7))
    parser.add_argument("--game", type=int, choices=(0, 1, 2))
    parser.add_argument("--aggregate-subjects", nargs="+", type=int)
    parser.add_argument("--shared-cache-root", type=Path, default=DEFAULT_SHARED_CACHE_ROOT)
    parser.add_argument("--behavior-root", type=Path, default=DEFAULT_BEHAV_ROOT)
    parser.add_argument(
        "--derived-feature-source",
        choices=("basic-cache", "accepted-04"),
        default="basic-cache",
        help=(
            "basic-cache refits Actor OOF and builds visual controls locally "
            "(default); accepted-04 is a compatibility sensitivity only"
        ),
    )
    parser.add_argument("--legacy-04-output-root", type=Path, default=DEFAULT_BASE_OUTPUT_ROOT)
    parser.add_argument(
        "--rebuild-derived-features",
        action="store_true",
        help="rebuild v2-local Actor OOF and visual controls from source inputs",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.subject is not None and args.aggregate_subjects:
        raise ValueError("--subject and --aggregate-subjects are mutually exclusive")
    if args.game is None:
        raise ValueError("--game is required")
    if args.subject is not None:
        run_subject_analysis(args)
        return
    if args.aggregate_subjects:
        run_group_analysis(args)
        return
    raise ValueError("Provide --subject or --aggregate-subjects")


if __name__ == "__main__":
    main()
