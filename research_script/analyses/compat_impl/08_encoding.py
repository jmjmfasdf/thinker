#!/usr/bin/env python3
"""Controlled localization and optional predictive encoding.

The default analysis is a classical run-wise localization GLM.  It estimates
standardized beta maps for M, initial Q margin, revision magnitude, rollout
evidence, search breadth/depth, the terminal Actor residual, and transition
contrasts.  Marginal models add one target to sensory/motor/history controls;
partial models additionally include only prespecified upstream computational
blocks.  Runs are equalized within subject, and final group inference uses the
six subjects as the exchangeability units with exact synchronized sign flips.

This default answers *where a quantity covaries with BOLD*.  It does not require
delta CV-R2.  The previous nested LORO ridge engine remains available through
``--analysis-mode predictive`` (or ``--predictive-encoding``) for the distinct
question of whether a feature block adds held-out predictive information.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

import planning_brain_common as COMMON
import encoding_localization_pipeline as LOCALIZATION


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs" / "08_encoding"
T0 = time.time()
EPS = 1e-12
INK = "#20252B"
MUTED = "#59616A"
CROSS_RED_YELLOW = matplotlib.colors.LinearSegmentedColormap.from_list(
    "cross_red_yellow",
    ("#C8191E", "#ED3B24", "#F57C20", "#FDBE32", "#FFE45C"),
)


MODEL_LABELS = {
    "controls": "Controls",
    "root": "+ Pre-search Policy State",
    "geometry": "+ Search geometry",
    "evidence_global": "+ Final ensemble uncertainty",
    "evidence_decomposed": "+ Within/between-state uncertainty",
}
FEATURE_ONLY_MODEL_LABELS = {
    "feature_actor_slack_change": "Actor-policy Slack change",
    "feature_actor_demand_change": "Actor-policy Demand change",
    "feature_policy_slack": "Policy Slack",
    "feature_policy_demand": "Policy Demand",
    "feature_branch_diversity": "Root-branch diversity",
    "feature_terminal_depth": "Mean bout terminal depth",
    "feature_deep_allocation": "Deep-node allocation",
    "feature_path_revisit": "Path-revisit fraction",
    "feature_evidence_global": "Final ensemble uncertainty",
    "feature_evidence_within": "Within-state uncertainty",
    "feature_evidence_between": "Between-state disagreement",
    "feature_outcome_mean": "Outcome mean shift",
    "feature_outcome_dispersion": "Outcome dispersion",
    "feature_sr_reward": "SR reward trail",
    "feature_vp_bootstrap": "VP bootstrap revision",
}
CONTROLLED_UNIQUE_MODEL_LABELS = {
    "unique_root_full": "Controls + full Policy State",
    "unique_root_without_slack": "Controls + Demand terms",
    "unique_root_without_demand": "Controls + Slack terms",
    "unique_geometry_full": "Controls + Policy State + primary geometry",
    "unique_geometry_depth_only": "Controls + Policy State + depth",
    "unique_geometry_branch_only": "Controls + Policy State + diversity",
    "unique_geometry_all": "Controls + Policy State + all geometry",
    "unique_geometry_without_deep": "All except deep allocation",
    "unique_geometry_without_revisit": "All except path revisit",
    "unique_evidence_global_full": "All geometry + final uncertainty",
    "unique_evidence_decomposed_full": "All geometry + within/between",
    "unique_evidence_without_within": "All geometry + between-state",
    "unique_evidence_without_between": "All geometry + within-state",
    "unique_value_outcome_full": "All geometry + outcome mean/dispersion",
    "unique_value_without_mean": "All geometry + outcome dispersion",
    "unique_value_without_dispersion": "All geometry + outcome mean",
    "unique_value_mechanism_full": "All geometry + SR/VP",
    "unique_value_without_sr": "All geometry + VP bootstrap",
    "unique_value_without_vp": "All geometry + SR reward trail",
}
CONTRASTS = {
    "root_over_controls": ("controls", "root"),
    "geometry_over_root": ("root", "geometry"),
    "evidence_global_over_geometry": ("geometry", "evidence_global"),
    "evidence_decomposed_over_geometry": ("geometry", "evidence_decomposed"),
}
CONTROLLED_UNIQUE_CONTRASTS = {
    "policy_slack_unique": ("unique_root_without_slack", "unique_root_full"),
    "policy_demand_unique": ("unique_root_without_demand", "unique_root_full"),
    "branch_diversity_unique": (
        "unique_geometry_depth_only",
        "unique_geometry_full",
    ),
    "terminal_depth_unique": (
        "unique_geometry_branch_only",
        "unique_geometry_full",
    ),
    "deep_allocation_unique": (
        "unique_geometry_without_deep",
        "unique_geometry_all",
    ),
    "path_revisit_unique": (
        "unique_geometry_without_revisit",
        "unique_geometry_all",
    ),
    "evidence_global_unique": (
        "unique_geometry_all",
        "unique_evidence_global_full",
    ),
    "evidence_within_unique": (
        "unique_evidence_without_within",
        "unique_evidence_decomposed_full",
    ),
    "evidence_between_unique": (
        "unique_evidence_without_between",
        "unique_evidence_decomposed_full",
    ),
}
VALUE_CONTROLLED_UNIQUE_CONTRASTS = {
    "outcome_mean_unique": (
        "unique_value_without_mean",
        "unique_value_outcome_full",
    ),
    "outcome_dispersion_unique": (
        "unique_value_without_dispersion",
        "unique_value_outcome_full",
    ),
    "sr_reward_unique": (
        "unique_value_without_sr",
        "unique_value_mechanism_full",
    ),
    "vp_bootstrap_unique": (
        "unique_value_without_vp",
        "unique_value_mechanism_full",
    ),
}
EXPANDED_CONTROLLED_UNIQUE_CONTRASTS = {
    **CONTROLLED_UNIQUE_CONTRASTS,
    **VALUE_CONTROLLED_UNIQUE_CONTRASTS,
}
CONTROLLED_UNIQUE_CONTRAST_LABELS = {
    "policy_slack_unique": "Policy Slack",
    "policy_demand_unique": "Policy Demand",
    "branch_diversity_unique": "Root-branch diversity",
    "terminal_depth_unique": "Mean bout terminal depth",
    "deep_allocation_unique": "Deep-node allocation",
    "path_revisit_unique": "Path-revisit fraction",
    "evidence_global_unique": "Final ensemble uncertainty",
    "evidence_within_unique": "Within-state uncertainty",
    "evidence_between_unique": "Between-state disagreement",
    "outcome_mean_unique": "Outcome mean shift",
    "outcome_dispersion_unique": "Outcome dispersion",
    "sr_reward_unique": "SR reward trail",
    "vp_bootstrap_unique": "VP bootstrap revision",
}


def log(message: str) -> None:
    print(f"[08_encoding +{time.time() - T0:8.1f}s] {message}", flush=True)


def zscore_fit(array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(array, dtype=np.float64)
    mean = np.mean(values, axis=0)
    scale = np.std(values, axis=0)
    scale[scale < EPS] = 1.0
    return mean, scale


@dataclass
class FeatureTransformer:
    ram_mean: np.ndarray
    ram_scale: np.ndarray
    ram_pca: Optional[PCA]
    visual_mean: np.ndarray
    visual_scale: np.ndarray
    visual_pca: Optional[PCA]
    process_mean: np.ndarray
    process_scale: np.ndarray
    process_names: Tuple[str, ...]
    include_ram: bool = True
    include_visual: bool = True

    def transform(self, blocks: Mapping[str, np.ndarray]) -> np.ndarray:
        arrays: List[np.ndarray] = []
        if self.include_ram:
            ram = (
                np.asarray(blocks["ram"], dtype=float) - self.ram_mean
            ) / self.ram_scale
            if self.ram_pca is not None:
                ram = self.ram_pca.transform(ram)
            arrays.append(ram)
        if self.include_visual:
            visual = (
                np.asarray(blocks["visual"], dtype=float) - self.visual_mean
            ) / self.visual_scale
            if self.visual_pca is not None:
                visual = self.visual_pca.transform(visual)
            arrays.append(visual)
        process = COMMON.concatenate_named_blocks(blocks, self.process_names)
        process = (process - self.process_mean) / self.process_scale
        arrays.append(process)
        return np.column_stack(arrays).astype(np.float32)


def model_process_names(
    include_secondary_geometry: bool,
) -> Dict[str, Tuple[str, ...]]:
    controls = (
        "q_state",
        "nuisance_continuous",
        "current_action",
        "previous_action",
        "drift",
    )
    geometry_names = ("geometry_primary",)
    if include_secondary_geometry:
        geometry_names = ("geometry_primary", "geometry_secondary")
    return {
        "controls": controls,
        "root": (*controls, "root"),
        "geometry": (*controls, "root", *geometry_names),
        "evidence_global": (
            *controls,
            "root",
            *geometry_names,
            "evidence_global",
        ),
        "evidence_decomposed": (
            *controls,
            "root",
            *geometry_names,
            "evidence_decomposed",
        ),
    }


def feature_only_process_names() -> Dict[str, Tuple[str, ...]]:
    """One HRF-aligned planning variable per Cross-style encoding model."""
    return {
        "feature_policy_slack": ("policy_slack",),
        "feature_policy_demand": ("policy_demand",),
        "feature_branch_diversity": ("branch_diversity",),
        "feature_terminal_depth": ("terminal_depth",),
        "feature_deep_allocation": ("deep_allocation",),
        "feature_path_revisit": ("path_revisit",),
        "feature_evidence_global": ("evidence_global",),
        "feature_evidence_within": ("evidence_within",),
        "feature_evidence_between": ("evidence_between",),
    }


def actor_planning_feature_only_process_names() -> Dict[str, Tuple[str, ...]]:
    """Final Section-3 features, each fit as a separate pure encoding model."""
    return {
        "feature_actor_slack_change": ("actor_slack_change",),
        "feature_actor_demand_change": ("actor_demand_change",),
        "feature_branch_diversity": ("branch_diversity",),
        "feature_terminal_depth": ("terminal_depth",),
        "feature_deep_allocation": ("deep_allocation",),
    }


def expanded_feature_only_process_names() -> Dict[str, Tuple[str, ...]]:
    """05 value additions to the already-complete feature-only tier.

    The original nine feature-only models live in ``feature_only/`` and are
    bitwise compatible with the v3 cache.  Re-running them would not add
    information, so the expanded namespace computes only the four new models
    and records the legacy namespace as its companion in the manifest.
    """
    return {
        "feature_outcome_mean": ("outcome_mean",),
        "feature_outcome_dispersion": ("outcome_dispersion",),
        "feature_sr_reward": ("sr_reward",),
        "feature_vp_bootstrap": ("vp_bootstrap",),
    }


def controlled_unique_process_names() -> Dict[str, Tuple[str, ...]]:
    """Full/reduced models for feature-family unique held-out variance."""
    controls = (
        "q_state",
        "nuisance_continuous",
        "current_action",
        "previous_action",
        "drift",
    )
    root = (*controls, "root")
    primary = (*root, "geometry_primary")
    all_geometry = (*primary, "geometry_secondary")
    return {
        "unique_root_full": root,
        "unique_root_without_slack": (*controls, "root_without_slack"),
        "unique_root_without_demand": (*controls, "root_without_demand"),
        "unique_geometry_full": primary,
        "unique_geometry_depth_only": (*root, "terminal_depth"),
        "unique_geometry_branch_only": (*root, "branch_diversity"),
        "unique_geometry_all": all_geometry,
        "unique_geometry_without_deep": (*primary, "path_revisit"),
        "unique_geometry_without_revisit": (*primary, "deep_allocation"),
        "unique_evidence_global_full": (*all_geometry, "evidence_global"),
        "unique_evidence_decomposed_full": (
            *all_geometry,
            "evidence_decomposed",
        ),
        "unique_evidence_without_within": (
            *all_geometry,
            "evidence_between",
        ),
        "unique_evidence_without_between": (
            *all_geometry,
            "evidence_within",
        ),
    }


def expanded_controlled_unique_process_names() -> Dict[str, Tuple[str, ...]]:
    """05 value full/reduced pairs complementing ``controlled_unique/``."""
    controls = (
        "q_state",
        "nuisance_continuous",
        "current_action",
        "previous_action",
        "drift",
    )
    root = (*controls, "root")
    all_geometry = (*root, "geometry_primary", "geometry_secondary")
    return {
        "unique_value_outcome_full": (
            *all_geometry,
            "value_outcome",
        ),
        "unique_value_without_mean": (
            *all_geometry,
            "outcome_dispersion",
        ),
        "unique_value_without_dispersion": (
            *all_geometry,
            "outcome_mean",
        ),
        # This is an alternative decomposition of outcome mean.  It never
        # appears in the same model as value_outcome.
        "unique_value_mechanism_full": (
            *all_geometry,
            "value_mechanism",
        ),
        "unique_value_without_sr": (
            *all_geometry,
            "vp_bootstrap",
        ),
        "unique_value_without_vp": (
            *all_geometry,
            "sr_reward",
        ),
    }


def add_individual_feature_blocks(blocks: Dict[str, np.ndarray]) -> None:
    """Expose named one-column views without changing the shared time cache."""
    required_widths = {
        "root": 5,
        "actor_policy_change": 2,
        "geometry_primary": 2,
        "geometry_secondary": 2,
        "evidence_global": 1,
        "evidence_decomposed": 2,
    }
    for name, width in required_widths.items():
        if name not in blocks or np.asarray(blocks[name]).shape[1] < width:
            raise ValueError(
                f"Cached process block {name!r} does not have {width} columns"
            )
    blocks.update(
        {
            "policy_slack": np.asarray(blocks["root"])[:, 0:1],
            "policy_demand": np.asarray(blocks["root"])[:, 1:2],
            "actor_slack_change": np.asarray(
                blocks["actor_policy_change"]
            )[:, 0:1],
            "actor_demand_change": np.asarray(
                blocks["actor_policy_change"]
            )[:, 1:2],
            # The interaction belongs to both feature-family tests: each
            # reduced model removes the named variable, its square, and S×D.
            "root_without_slack": np.asarray(blocks["root"])[:, [1, 3]],
            "root_without_demand": np.asarray(blocks["root"])[:, [0, 2]],
            "branch_diversity": np.asarray(
                blocks["geometry_primary"]
            )[:, 0:1],
            "terminal_depth": np.asarray(
                blocks["geometry_primary"]
            )[:, 1:2],
            "deep_allocation": np.asarray(
                blocks["geometry_secondary"]
            )[:, 0:1],
            "path_revisit": np.asarray(
                blocks["geometry_secondary"]
            )[:, 1:2],
            "evidence_within": np.asarray(
                blocks["evidence_decomposed"]
            )[:, 0:1],
            "evidence_between": np.asarray(
                blocks["evidence_decomposed"]
            )[:, 1:2],
        }
    )


def load_run_feature_blocks(
    run_specs: Sequence[COMMON.BrainRunSpec],
) -> List[Dict[str, np.ndarray]]:
    result: List[Dict[str, np.ndarray]] = []
    for run in run_specs:
        blocks = COMMON.load_process_blocks(run)
        add_individual_feature_blocks(blocks)
        blocks["ram"] = np.asarray(
            np.load(run.ram_path, allow_pickle=False), dtype=np.float32
        )
        blocks["visual"] = COMMON.load_visual(run)
        result.append(blocks)
    return result


def fit_transformer(
    run_blocks: Sequence[Mapping[str, np.ndarray]],
    train_indices: Sequence[int],
    process_names: Sequence[str],
    ram_pca_dim: int,
    visual_pca_dim: int,
    random_state: int,
    include_ram: bool = True,
    include_visual: bool = True,
) -> FeatureTransformer:
    first = run_blocks[int(train_indices[0])]
    ram_width = int(np.asarray(first["ram"]).shape[1])
    ram_mean = np.zeros(ram_width, dtype=float)
    ram_scale = np.ones(ram_width, dtype=float)
    ram_pca: Optional[PCA] = None
    if include_ram:
        ram_train = np.concatenate(
            [
                np.asarray(run_blocks[index]["ram"], dtype=float)
                for index in train_indices
            ],
            axis=0,
        )
        ram_mean, ram_scale = zscore_fit(ram_train)
        ram_standardized = (ram_train - ram_mean) / ram_scale
        output_dim = min(
            int(ram_pca_dim),
            ram_standardized.shape[1],
            max(1, ram_standardized.shape[0] - 1),
        )
        if 0 < output_dim < ram_standardized.shape[1]:
            ram_pca = PCA(
                n_components=output_dim,
                svd_solver="randomized",
                random_state=int(random_state),
            )
            ram_pca.fit(ram_standardized)

    visual_width = int(np.asarray(first["visual"]).shape[1])
    visual_mean = np.zeros(visual_width, dtype=float)
    visual_scale = np.ones(visual_width, dtype=float)
    visual_pca: Optional[PCA] = None
    if include_visual:
        visual_train = np.concatenate(
            [
                np.asarray(run_blocks[index]["visual"], dtype=float)
                for index in train_indices
            ],
            axis=0,
        )
        visual_mean, visual_scale = zscore_fit(visual_train)
        visual_standardized = (visual_train - visual_mean) / visual_scale
        visual_output_dim = min(
            int(visual_pca_dim),
            visual_standardized.shape[1],
            max(1, visual_standardized.shape[0] - 1),
        )
        if 0 < visual_output_dim < visual_standardized.shape[1]:
            visual_pca = PCA(
                n_components=visual_output_dim,
                svd_solver="randomized",
                random_state=int(random_state) + 17,
            )
            visual_pca.fit(visual_standardized)

    process_train = np.concatenate(
        [
            COMMON.concatenate_named_blocks(run_blocks[index], process_names)
            for index in train_indices
        ],
        axis=0,
    ).astype(float)
    process_mean, process_scale = zscore_fit(process_train)
    return FeatureTransformer(
        ram_mean=ram_mean,
        ram_scale=ram_scale,
        ram_pca=ram_pca,
        visual_mean=visual_mean,
        visual_scale=visual_scale,
        visual_pca=visual_pca,
        process_mean=process_mean,
        process_scale=process_scale,
        process_names=tuple(process_names),
        include_ram=bool(include_ram),
        include_visual=bool(include_visual),
    )


def pearson_columns(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    y = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    y = y - np.mean(y, axis=0, keepdims=True)
    p = p - np.mean(p, axis=0, keepdims=True)
    numerator = np.sum(y * p, axis=0)
    denominator = np.sqrt(np.sum(y * y, axis=0) * np.sum(p * p, axis=0))
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator > EPS,
    )


def cv_r2_columns(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    y = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    denominator = np.sum((y - np.mean(y, axis=0, keepdims=True)) ** 2, axis=0)
    numerator = np.sum((y - p) ** 2, axis=0)
    return np.divide(
        denominator - numerator,
        denominator,
        out=np.full_like(denominator, np.nan, dtype=float),
        where=denominator > EPS,
    )


def exact_sign_flip_p(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return np.nan
    observed = float(np.mean(array))
    if array.size <= 20:
        signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=array.size)))
        null = np.mean(signs * array[None, :], axis=1)
    else:
        rng = np.random.default_rng(2026)
        null = np.mean(
            rng.choice((-1.0, 1.0), size=(100_000, array.size))
            * array[None, :],
            axis=1,
        )
    return float((1 + np.sum(null >= observed)) / (1 + len(null)))


def build_alpha_selection_targets(
    run_specs: Sequence[COMMON.BrainRunSpec],
    roi_names: Sequence[str],
    max_voxels: int,
) -> List[np.ndarray]:
    """Use the same deterministic voxel sample in every run.

    The target sample is used only inside training folds to select one alpha per
    model/outer fold.  The held-out outer run never contributes to selection.
    """
    per_roi = max(1, int(math.ceil(max_voxels / max(1, len(roi_names)))))
    target_runs: List[List[np.ndarray]] = [[] for _ in run_specs]
    used = 0
    for roi_name in roi_names:
        first_patterns, first_coords = COMMON.load_roi(run_specs[0], roi_name)
        take = min(per_roi, first_patterns.shape[1], max_voxels - used)
        if take <= 0:
            break
        indices = np.linspace(0, first_patterns.shape[1] - 1, take, dtype=int)
        for run_index, run in enumerate(run_specs):
            patterns, coords = COMMON.load_roi(run, roi_name)
            if patterns.shape[1] != first_patterns.shape[1] or not np.array_equal(
                coords, first_coords
            ):
                raise ValueError(
                    f"Voxel identity differs across runs for {roi_name}; "
                    "delete the shared planning-brain cache and rebuild it."
                )
            target_runs[run_index].append(patterns[:, indices])
        used += take
        if used >= max_voxels:
            break
    if used == 0:
        raise ValueError("No alpha-selection voxels were available")
    return [np.column_stack(parts).astype(np.float32) for parts in target_runs]


def select_outer_alpha(
    *,
    run_blocks: Sequence[Mapping[str, np.ndarray]],
    selection_y: Sequence[np.ndarray],
    outer_test: int,
    process_names: Sequence[str],
    alphas: np.ndarray,
    ram_pca_dim: int,
    visual_pca_dim: int,
    random_state: int,
    include_ram: bool = True,
    include_visual: bool = True,
) -> Tuple[float, np.ndarray]:
    outer_train = [index for index in range(len(run_blocks)) if index != outer_test]
    if len(outer_train) < 2:
        raise ValueError("Nested alpha selection requires at least three runs")
    scores = np.full((len(alphas), len(outer_train)), np.nan, dtype=float)
    for inner_column, inner_validation in enumerate(outer_train):
        inner_train = [index for index in outer_train if index != inner_validation]
        transformer = fit_transformer(
            run_blocks,
            inner_train,
            process_names,
            ram_pca_dim,
            visual_pca_dim,
            random_state + 97 * outer_test + inner_column,
            include_ram=include_ram,
            include_visual=include_visual,
        )
        x_train = np.concatenate(
            [transformer.transform(run_blocks[index]) for index in inner_train],
            axis=0,
        )
        x_validation = transformer.transform(run_blocks[inner_validation])
        y_train = np.concatenate([selection_y[index] for index in inner_train], axis=0)
        y_validation = selection_y[inner_validation]
        for alpha_index, alpha in enumerate(alphas):
            prediction = Ridge(alpha=float(alpha), fit_intercept=True).fit(
                x_train, y_train
            ).predict(x_validation)
            scores[alpha_index, inner_column] = float(
                np.nanmean(pearson_columns(y_validation, prediction))
            )
    mean_scores = np.nanmean(scores, axis=1)
    best_index = int(np.nanargmax(mean_scores))
    return float(alphas[best_index]), scores


def prepare_outer_designs(
    *,
    model_name: str,
    process_names: Sequence[str],
    run_specs: Sequence[COMMON.BrainRunSpec],
    run_blocks: Sequence[Mapping[str, np.ndarray]],
    selection_y: Sequence[np.ndarray],
    alphas: np.ndarray,
    ram_pca_dim: int,
    visual_pca_dim: int,
    random_state: int,
    design_dir: Path,
    include_ram: bool = True,
    include_visual: bool = True,
    reuse_existing: bool = True,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    model_dir = design_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    for outer_test, run in enumerate(run_specs):
        design_path = model_dir / f"outer_{outer_test:02d}.npz"
        if reuse_existing and design_path.exists():
            with np.load(design_path, allow_pickle=False) as archive:
                saved_process_names = tuple(
                    str(value) for value in archive.get("process_names", [])
                )
                saved_include_ram = bool(
                    np.asarray(
                        archive.get("include_ram", [1]), dtype=np.uint8
                    ).reshape(-1)[0]
                )
                saved_include_visual = bool(
                    np.asarray(
                        archive.get("include_visual", [1]), dtype=np.uint8
                    ).reshape(-1)[0]
                )
                compatible = (
                    saved_process_names == tuple(process_names)
                    and saved_include_ram == bool(include_ram)
                    and saved_include_visual == bool(include_visual)
                    and int(np.asarray(archive["test_index"]).reshape(-1)[0])
                    == outer_test
                )
                if compatible:
                    best_alpha = float(
                        np.asarray(archive["best_alpha"]).reshape(-1)[0]
                    )
                    n_features = int(np.asarray(archive["x_train"]).shape[1])
                    cached_inner_scores = np.asarray(
                        archive.get("inner_scores", []), dtype=float
                    )
                else:
                    best_alpha = np.nan
                    n_features = -1
                    cached_inner_scores = np.array([], dtype=float)
            if compatible:
                cached_mean = (
                    np.nanmean(cached_inner_scores, axis=1)
                    if cached_inner_scores.ndim == 2
                    and cached_inner_scores.size
                    else np.array([], dtype=float)
                )
                cached_best = (
                    int(np.nanargmax(cached_mean))
                    if cached_mean.size
                    else -1
                )
                rows.append(
                    {
                        "model": model_name,
                        "outer_test": outer_test,
                        "held_out_run": run.run_label,
                        "best_alpha": best_alpha,
                        "inner_mean_r": (
                            float(cached_mean[cached_best])
                            if cached_best >= 0
                            else np.nan
                        ),
                        "inner_min_r": (
                            float(
                                np.nanmin(
                                    cached_inner_scores[cached_best]
                                )
                            )
                            if cached_best >= 0
                            else np.nan
                        ),
                        "inner_max_r": (
                            float(
                                np.nanmax(
                                    cached_inner_scores[cached_best]
                                )
                            )
                            if cached_best >= 0
                            else np.nan
                        ),
                        "n_features": n_features,
                        "include_ram": bool(include_ram),
                        "include_visual": bool(include_visual),
                        "process_names": "|".join(process_names),
                        "design_cache_reused": True,
                    }
                )
                log(
                    f"[design:{model_name}] outer "
                    f"{outer_test + 1}/{len(run_specs)} reused "
                    f"alpha={best_alpha:g} p={n_features}"
                )
                continue
        outer_train = [index for index in range(len(run_specs)) if index != outer_test]
        best_alpha, inner_scores = select_outer_alpha(
            run_blocks=run_blocks,
            selection_y=selection_y,
            outer_test=outer_test,
            process_names=process_names,
            alphas=alphas,
            ram_pca_dim=ram_pca_dim,
            visual_pca_dim=visual_pca_dim,
            random_state=random_state,
            include_ram=include_ram,
            include_visual=include_visual,
        )
        transformer = fit_transformer(
            run_blocks,
            outer_train,
            process_names,
            ram_pca_dim,
            visual_pca_dim,
            random_state + 10_007 + outer_test,
            include_ram=include_ram,
            include_visual=include_visual,
        )
        x_train = np.concatenate(
            [transformer.transform(run_blocks[index]) for index in outer_train],
            axis=0,
        )
        x_test = transformer.transform(run_blocks[outer_test])
        COMMON.atomic_save_npz(
            model_dir / f"outer_{outer_test:02d}.npz",
            x_train=x_train.astype(np.float32),
            x_test=x_test.astype(np.float32),
            train_indices=np.asarray(outer_train, dtype=np.int16),
            test_index=np.asarray([outer_test], dtype=np.int16),
            best_alpha=np.asarray([best_alpha], dtype=np.float64),
            inner_scores=np.asarray(inner_scores, dtype=np.float32),
            include_ram=np.asarray([include_ram], dtype=np.uint8),
            include_visual=np.asarray([include_visual], dtype=np.uint8),
            process_names=np.asarray(tuple(process_names), dtype="U64"),
        )
        rows.append(
            {
                "model": model_name,
                "outer_test": outer_test,
                "held_out_run": run.run_label,
                "best_alpha": best_alpha,
                "inner_mean_r": float(
                    np.nanmax(np.nanmean(inner_scores, axis=1))
                ),
                "inner_min_r": float(
                    np.nanmin(inner_scores[np.nanargmax(np.nanmean(inner_scores, axis=1))])
                ),
                "inner_max_r": float(
                    np.nanmax(inner_scores[np.nanargmax(np.nanmean(inner_scores, axis=1))])
                ),
                "n_features": int(x_train.shape[1]),
                "include_ram": bool(include_ram),
                "include_visual": bool(include_visual),
                "process_names": "|".join(process_names),
                "design_cache_reused": False,
            }
        )
        log(
            f"[design:{model_name}] outer {outer_test + 1}/{len(run_specs)} "
            f"alpha={best_alpha:g} p={x_train.shape[1]}"
        )
    return pd.DataFrame(rows)


def predict_roi_model(
    *,
    model_name: str,
    run_specs: Sequence[COMMON.BrainRunSpec],
    y_runs: Sequence[np.ndarray],
    design_dir: Path,
) -> Tuple[List[np.ndarray], List[Dict[str, object]]]:
    predictions: List[Optional[np.ndarray]] = [None] * len(run_specs)
    fold_rows: List[Dict[str, object]] = []
    for outer_test, run in enumerate(run_specs):
        with np.load(
            design_dir / model_name / f"outer_{outer_test:02d}.npz",
            allow_pickle=False,
        ) as archive:
            x_train = np.asarray(archive["x_train"], dtype=np.float32)
            x_test = np.asarray(archive["x_test"], dtype=np.float32)
            train_indices = np.asarray(archive["train_indices"], dtype=int)
            alpha = float(np.asarray(archive["best_alpha"]).reshape(-1)[0])
        y_train = np.concatenate([y_runs[index] for index in train_indices], axis=0)
        prediction = Ridge(alpha=alpha, fit_intercept=True).fit(
            x_train, y_train
        ).predict(x_test)
        predictions[outer_test] = np.asarray(prediction, dtype=np.float32)
        fold_rows.append(
            {
                "model": model_name,
                "held_out_run": run.run_label,
                "outer_test": outer_test,
                "best_alpha": alpha,
                "mean_r": float(
                    np.nanmean(pearson_columns(y_runs[outer_test], prediction))
                ),
                "mean_cv_r2": float(
                    np.nanmean(cv_r2_columns(y_runs[outer_test], prediction))
                ),
            }
        )
    return [np.asarray(value) for value in predictions], fold_rows


def concatenate_run_scores(
    y_runs: Sequence[np.ndarray],
    predictions: Sequence[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.concatenate(y_runs, axis=0)
    predicted = np.concatenate(predictions, axis=0)
    return pearson_columns(y, predicted), cv_r2_columns(y, predicted), predicted


def block_permutation_roi_delta(
    *,
    y_runs: Sequence[np.ndarray],
    reduced_predictions: Sequence[np.ndarray],
    full_predictions: Sequence[np.ndarray],
    n_permutations: int,
    block_size: int,
    max_voxels: int,
    random_state: int,
) -> np.ndarray:
    """Conditional prediction-space residual block permutation.

    The reduced-model prediction is kept in its observed temporal position.
    Only its held-out residual is shuffled, so the null preserves variance
    already explained by RAM, visual, action, Q-state, and other reduced-model
    controls while destroying alignment to the added feature block.  Block
    dot products are precomputed and all selected voxels receive the same
    block order, preserving the synchronized ROI × contrast max-FWER null.
    """
    if n_permutations <= 0:
        return np.array([], dtype=np.float32)
    n_voxels = y_runs[0].shape[1]
    take = min(max_voxels, n_voxels)
    selected = np.linspace(0, n_voxels - 1, take, dtype=int)
    n_total = int(sum(len(values) for values in y_runs))
    total_sum = np.zeros(take, dtype=np.float64)
    constant_sum_squares = np.zeros(take, dtype=np.float64)
    delta_prediction_squares = np.zeros(take, dtype=np.float64)
    cross_tables: List[Tuple[np.ndarray, np.ndarray]] = []
    for values, reduced_values, full_values in zip(
        y_runs, reduced_predictions, full_predictions
    ):
        reduced = np.asarray(reduced_values[:, selected], dtype=np.float64)
        full = np.asarray(full_values[:, selected], dtype=np.float64)
        observed = np.asarray(values[:, selected], dtype=np.float64)
        if not (
            observed.shape == reduced.shape == full.shape
            and len(observed) % block_size == 0
        ):
            raise ValueError(
                "Fast ROI block null requires matching arrays and n_TR "
                f"divisible by block size; got {observed.shape}, "
                f"{reduced.shape}, {full.shape}, block={block_size}"
            )
        residual = observed - reduced
        delta_prediction = full - reduced
        total_sum += np.sum(reduced + residual, axis=0)
        constant_sum_squares += np.sum(
            reduced**2 + residual**2, axis=0
        )
        delta_prediction_squares += np.sum(
            delta_prediction**2, axis=0
        )
        n_blocks = len(observed) // block_size
        residual_blocks = residual.reshape(n_blocks, block_size, take)
        delta_blocks = delta_prediction.reshape(
            n_blocks, block_size, take
        )
        reduced_blocks = reduced.reshape(n_blocks, block_size, take)
        cross_tables.append(
            (
                np.einsum(
                    "slv,dlv->sdv",
                    residual_blocks,
                    delta_blocks,
                    optimize=True,
                ),
                np.einsum(
                    "slv,dlv->sdv",
                    residual_blocks,
                    reduced_blocks,
                    optimize=True,
                ),
            )
        )

    mean_correction = total_sum**2 / max(n_total, 1)
    rng = np.random.default_rng(random_state)
    null = np.empty(n_permutations, dtype=np.float32)
    batch_size = min(10_000, n_permutations)
    for start in range(0, n_permutations, batch_size):
        stop = min(start + batch_size, n_permutations)
        batch = stop - start
        cross_delta_total = np.zeros((batch, take), dtype=np.float64)
        cross_reduced_total = np.zeros((batch, take), dtype=np.float64)
        for cross_delta, cross_reduced in cross_tables:
            n_blocks = cross_delta.shape[0]
            orders = np.argsort(
                rng.random((batch, n_blocks)), axis=1
            )
            destinations = np.arange(n_blocks, dtype=int)[None, :]
            cross_delta_total += np.sum(
                cross_delta[orders, destinations, :], axis=1
            )
            cross_reduced_total += np.sum(
                cross_reduced[orders, destinations, :], axis=1
            )
        numerator = (
            2.0 * cross_delta_total
            - delta_prediction_squares[None, :]
        )
        denominator = (
            constant_sum_squares[None, :]
            + 2.0 * cross_reduced_total
            - mean_correction[None, :]
        )
        delta_r2 = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan),
            where=denominator > EPS,
        )
        null[start:stop] = np.nanmean(delta_r2, axis=1).astype(np.float32)
    return null


def add_global_block_max_fwer(
    contrast_df: pd.DataFrame,
    null_store: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    """FWER correction across all tested ROI × contrast combinations."""
    result = contrast_df.copy()
    result["p_block_permutation_fwer_one_sided"] = np.nan
    if result.empty or not null_store:
        return result
    ordered_nulls: List[np.ndarray] = []
    row_indices: List[int] = []
    for row_index, row in result.iterrows():
        key = f"{COMMON.safe_name(str(row['roi']))}__{row['contrast']}"
        if key not in null_store:
            continue
        ordered_nulls.append(np.asarray(null_store[key], dtype=float))
        row_indices.append(int(row_index))
    if not ordered_nulls:
        return result
    lengths = {len(values) for values in ordered_nulls}
    if len(lengths) != 1:
        raise ValueError(f"Permutation length mismatch across encoding tests: {lengths}")
    n_permutations = next(iter(lengths))
    if n_permutations == 0:
        return result
    max_null = np.max(np.vstack(ordered_nulls), axis=0)
    for row_index in row_indices:
        observed = float(
            result.at[row_index, "permutation_subset_mean_delta_cv_r2"]
        )
        result.at[row_index, "p_block_permutation_fwer_one_sided"] = (
            1 + np.sum(max_null >= observed)
        ) / (1 + n_permutations)
    return result


def pooled_conditional_voxel_null(
    *,
    y_runs: Sequence[np.ndarray],
    reduced_runs: Sequence[np.ndarray],
    full_runs: Sequence[np.ndarray],
    n_permutations: int,
    block_size: int,
    random_state: int,
    batch_size: int = 10_000,
) -> np.ndarray:
    """Efficient Cross-style pooled null for a small selected voxel set.

    For 480-TR runs and 40-TR blocks, all source/destination block dot
    products can be precomputed.  Each permutation then only re-indexes a
    12×12 matrix instead of materializing the full shuffled time course.
    """
    if n_permutations <= 0:
        return np.array([], dtype=np.float32)
    if not (len(y_runs) == len(reduced_runs) == len(full_runs)):
        raise ValueError("Selected-voxel run lists have different lengths")
    n_voxels = int(y_runs[0].shape[1])
    n_total = int(sum(len(values) for values in y_runs))
    total_sum = np.zeros(n_voxels, dtype=np.float64)
    constant_sum_squares = np.zeros(n_voxels, dtype=np.float64)
    delta_prediction_squares = np.zeros(n_voxels, dtype=np.float64)
    cross_tables: List[Tuple[np.ndarray, np.ndarray]] = []
    for y, reduced, full in zip(y_runs, reduced_runs, full_runs):
        observed = np.asarray(y, dtype=np.float64)
        reduced_prediction = np.asarray(reduced, dtype=np.float64)
        full_prediction = np.asarray(full, dtype=np.float64)
        if observed.shape != reduced_prediction.shape or observed.shape != full_prediction.shape:
            raise ValueError("Selected-voxel prediction shape mismatch")
        if len(observed) % block_size != 0:
            raise ValueError(
                "Fast pooled null requires n_TR divisible by block size; "
                f"got {len(observed)} and {block_size}"
            )
        residual = observed - reduced_prediction
        delta_prediction = full_prediction - reduced_prediction
        total_sum += np.sum(reduced_prediction + residual, axis=0)
        constant_sum_squares += np.sum(
            reduced_prediction**2 + residual**2, axis=0
        )
        delta_prediction_squares += np.sum(delta_prediction**2, axis=0)
        n_blocks = len(observed) // block_size
        residual_blocks = residual.reshape(n_blocks, block_size, n_voxels)
        delta_blocks = delta_prediction.reshape(n_blocks, block_size, n_voxels)
        reduced_blocks = reduced_prediction.reshape(
            n_blocks, block_size, n_voxels
        )
        cross_delta = np.einsum(
            "slv,dlv->sdv", residual_blocks, delta_blocks, optimize=True
        )
        cross_reduced = np.einsum(
            "slv,dlv->sdv", residual_blocks, reduced_blocks, optimize=True
        )
        cross_tables.append((cross_delta, cross_reduced))

    mean_correction = total_sum**2 / max(n_total, 1)
    rng = np.random.default_rng(random_state)
    pooled = np.empty((n_permutations, n_voxels), dtype=np.float32)
    for start in range(0, n_permutations, batch_size):
        stop = min(start + batch_size, n_permutations)
        take = stop - start
        cross_delta_total = np.zeros((take, n_voxels), dtype=np.float64)
        cross_reduced_total = np.zeros((take, n_voxels), dtype=np.float64)
        for cross_delta, cross_reduced in cross_tables:
            n_blocks = cross_delta.shape[0]
            orders = np.argsort(
                rng.random((take, n_voxels, n_blocks)), axis=2
            )
            destinations = np.arange(n_blocks, dtype=int)[None, None, :]
            voxels = np.arange(n_voxels, dtype=int)[None, :, None]
            cross_delta_total += np.sum(
                cross_delta[orders, destinations, voxels], axis=2
            )
            cross_reduced_total += np.sum(
                cross_reduced[orders, destinations, voxels], axis=2
            )
        numerator = 2.0 * cross_delta_total - delta_prediction_squares[None, :]
        denominator = (
            constant_sum_squares[None, :]
            + 2.0 * cross_reduced_total
            - mean_correction[None, :]
        )
        pooled[start:stop] = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > EPS,
        ).astype(np.float32)
    return pooled.reshape(-1)


def pooled_encoding_r_voxel_null(
    *,
    y_runs: Sequence[np.ndarray],
    prediction_runs: Sequence[np.ndarray],
    n_permutations: int,
    block_size: int,
    random_state: int,
    batch_size: int = 10_000,
) -> np.ndarray:
    """Cross-style pooled null for held-out voxelwise encoding correlations.

    Held-out predictions remain fixed.  For every selected voxel, observed
    BOLD is independently shuffled in contiguous blocks within each run.  The
    block dot products are precomputed, so 100,000 permutations per voxel do
    not require materializing full permuted time courses.
    """
    if n_permutations <= 0:
        return np.array([], dtype=np.float32)
    if len(y_runs) != len(prediction_runs):
        raise ValueError("Selected-voxel observation/prediction run lists differ")
    if not y_runs:
        raise ValueError("No selected-voxel runs were supplied")
    n_voxels = int(y_runs[0].shape[1])
    n_total = int(sum(len(values) for values in y_runs))
    sum_y = np.zeros(n_voxels, dtype=np.float64)
    sum_prediction = np.zeros(n_voxels, dtype=np.float64)
    sum_y_squared = np.zeros(n_voxels, dtype=np.float64)
    sum_prediction_squared = np.zeros(n_voxels, dtype=np.float64)
    cross_tables: List[np.ndarray] = []
    for y, prediction in zip(y_runs, prediction_runs):
        observed = np.asarray(y, dtype=np.float64)
        predicted = np.asarray(prediction, dtype=np.float64)
        if observed.shape != predicted.shape:
            raise ValueError("Selected-voxel encoding prediction shape mismatch")
        if observed.ndim != 2 or observed.shape[1] != n_voxels:
            raise ValueError("Selected-voxel encoding arrays must be TR × voxel")
        if len(observed) % block_size != 0:
            raise ValueError(
                "Fast pooled encoding-r null requires n_TR divisible by block "
                f"size; got {len(observed)} and {block_size}"
            )
        sum_y += np.sum(observed, axis=0)
        sum_prediction += np.sum(predicted, axis=0)
        sum_y_squared += np.sum(observed**2, axis=0)
        sum_prediction_squared += np.sum(predicted**2, axis=0)
        n_blocks = len(observed) // block_size
        y_blocks = observed.reshape(n_blocks, block_size, n_voxels)
        prediction_blocks = predicted.reshape(n_blocks, block_size, n_voxels)
        cross_tables.append(
            np.einsum(
                "slv,dlv->sdv", y_blocks, prediction_blocks, optimize=True
            )
        )

    covariance_correction = sum_y * sum_prediction / max(n_total, 1)
    y_ss = sum_y_squared - sum_y**2 / max(n_total, 1)
    prediction_ss = (
        sum_prediction_squared
        - sum_prediction**2 / max(n_total, 1)
    )
    denominator = np.sqrt(np.maximum(y_ss, 0.0) * np.maximum(prediction_ss, 0.0))
    rng = np.random.default_rng(random_state)
    pooled = np.empty((n_permutations, n_voxels), dtype=np.float32)
    for start in range(0, n_permutations, batch_size):
        stop = min(start + batch_size, n_permutations)
        take = stop - start
        dot_product = np.zeros((take, n_voxels), dtype=np.float64)
        for cross_table in cross_tables:
            n_blocks = cross_table.shape[0]
            orders = np.argsort(
                rng.random((take, n_voxels, n_blocks)), axis=2
            )
            destinations = np.arange(n_blocks, dtype=int)[None, None, :]
            voxels = np.arange(n_voxels, dtype=int)[None, :, None]
            dot_product += np.sum(
                cross_table[orders, destinations, voxels], axis=2
            )
        numerator = dot_product - covariance_correction[None, :]
        pooled[start:stop] = np.divide(
            numerator,
            denominator[None, :],
            out=np.zeros_like(numerator),
            where=denominator[None, :] > EPS,
        ).astype(np.float32)
    return pooled.reshape(-1)


def run_cross_style_voxel_inference(
    *,
    run_specs: Sequence[COMMON.BrainRunSpec],
    voxel_stats: Mapping[str, Mapping[str, np.ndarray]],
    design_dir: Path,
    output_dir: Path,
    n_permutations: int,
    block_size: int,
    top_fraction: float,
    selected_voxels: int,
    fdr_threshold: float,
    random_state: int,
    contrast_definitions: Mapping[str, Tuple[str, str]],
    contrast_labels: Mapping[str, str],
    figure_dir: Path,
    brain_template: Optional[Path] = None,
    display_smoothing_fwhm: float = 5.0,
) -> None:
    """Cross-style 14-voxel pooled null and atlas-wide voxel BH-FDR."""
    import nibabel as nib

    output_dir.mkdir(parents=True, exist_ok=True)
    reference = nib.load(str(run_specs[0].fmri_path))
    shape = tuple(int(value) for value in reference.shape[:3])
    anatomical, _ = load_encoding_anatomical_template(
        reference,
        run_specs[0].fmri_path,
        brain_template,
    )
    null_store: Dict[str, np.ndarray] = {}
    selection_rows: List[Dict[str, object]] = []
    fraction_rows: List[Dict[str, object]] = []
    contrast_rows: List[Dict[str, object]] = []

    for contrast_index, (contrast, (reduced_name, full_name)) in enumerate(
        contrast_definitions.items()
    ):
        # De-duplicate atlas overlaps by MNI voxel coordinate.
        unique: Dict[
            Tuple[int, int, int], Tuple[str, int, float, float]
        ] = {}
        for roi_name, statistics in voxel_stats.items():
            _, coords = COMMON.load_roi(run_specs[0], roi_name)
            full_r = np.asarray(
                statistics[f"encoding_r_{full_name}"], dtype=float
            )
            delta = np.asarray(
                statistics[f"delta_cv_r2_{contrast}"], dtype=float
            )
            for voxel_index, coordinate in enumerate(coords):
                key = tuple(int(value) for value in coordinate)
                if key not in unique:
                    unique[key] = (
                        roi_name,
                        int(voxel_index),
                        float(full_r[voxel_index]),
                        float(delta[voxel_index]),
                    )
        if len(unique) < selected_voxels:
            raise ValueError(
                f"{contrast}: only {len(unique)} unique voxels; "
                f"need at least {selected_voxels}"
            )
        coordinates = np.asarray(list(unique), dtype=np.int32)
        records = list(unique.values())
        encoding_scores = np.asarray([record[2] for record in records], dtype=float)
        observed_delta = np.asarray([record[3] for record in records], dtype=float)
        cutoff = float(np.nanquantile(encoding_scores, 1.0 - top_fraction))
        candidates = np.flatnonzero(
            np.isfinite(encoding_scores) & (encoding_scores >= cutoff)
        )
        if len(candidates) < selected_voxels:
            raise ValueError(
                f"{contrast}: top-fraction candidate set has only {len(candidates)} voxels"
            )
        rng = np.random.default_rng(
            random_state + 100_003 * contrast_index
        )
        selected_global = np.sort(
            rng.choice(candidates, selected_voxels, replace=False)
        )

        selected_y: List[List[np.ndarray]] = [[] for _ in run_specs]
        selected_reduced: List[List[np.ndarray]] = [[] for _ in run_specs]
        selected_full: List[List[np.ndarray]] = [[] for _ in run_specs]
        by_roi: Dict[str, List[Tuple[int, int]]] = {}
        for selected_order, global_index in enumerate(selected_global):
            roi_name, voxel_index, score, delta = records[int(global_index)]
            by_roi.setdefault(roi_name, []).append(
                (selected_order, int(voxel_index))
            )
            x, y, z = coordinates[int(global_index)]
            selection_rows.append(
                {
                    "contrast": contrast,
                    "selected_order": int(selected_order),
                    "roi": roi_name,
                    "roi_voxel_index": int(voxel_index),
                    "x": int(x),
                    "y": int(y),
                    "z": int(z),
                    "full_model_encoding_r": float(score),
                    "observed_delta_cv_r2": float(delta),
                    "top_fraction": float(top_fraction),
                    "candidate_cutoff_r": cutoff,
                    "random_seed": int(
                        random_state + 100_003 * contrast_index
                    ),
                }
            )

        # Refit only ROIs containing the selected 14 voxels.  Outer designs and
        # selected alphas are reused, so no hyperparameter search is repeated.
        run_slot_y: List[List[Optional[np.ndarray]]] = [
            [None] * selected_voxels for _ in run_specs
        ]
        run_slot_reduced: List[List[Optional[np.ndarray]]] = [
            [None] * selected_voxels for _ in run_specs
        ]
        run_slot_full: List[List[Optional[np.ndarray]]] = [
            [None] * selected_voxels for _ in run_specs
        ]
        for roi_name, selected_pairs in by_roi.items():
            y_runs: List[np.ndarray] = []
            for run in run_specs:
                patterns, _ = COMMON.load_roi(run, roi_name)
                y_runs.append(patterns)
            reduced_predictions, _ = predict_roi_model(
                model_name=reduced_name,
                run_specs=run_specs,
                y_runs=y_runs,
                design_dir=design_dir,
            )
            full_predictions, _ = predict_roi_model(
                model_name=full_name,
                run_specs=run_specs,
                y_runs=y_runs,
                design_dir=design_dir,
            )
            for selected_order, voxel_index in selected_pairs:
                for run_index in range(len(run_specs)):
                    run_slot_y[run_index][selected_order] = y_runs[run_index][
                        :, voxel_index
                    ]
                    run_slot_reduced[run_index][selected_order] = (
                        reduced_predictions[run_index][:, voxel_index]
                    )
                    run_slot_full[run_index][selected_order] = (
                        full_predictions[run_index][:, voxel_index]
                    )
        for run_index in range(len(run_specs)):
            if any(value is None for value in run_slot_y[run_index]):
                raise RuntimeError(f"Incomplete selected-voxel assembly: {contrast}")
            selected_y[run_index] = [
                np.asarray(value) for value in run_slot_y[run_index]
            ]
            selected_reduced[run_index] = [
                np.asarray(value) for value in run_slot_reduced[run_index]
            ]
            selected_full[run_index] = [
                np.asarray(value) for value in run_slot_full[run_index]
            ]
        y_matrix = [np.column_stack(values) for values in selected_y]
        reduced_matrix = [
            np.column_stack(values) for values in selected_reduced
        ]
        full_matrix = [np.column_stack(values) for values in selected_full]
        pooled_null = pooled_conditional_voxel_null(
            y_runs=y_matrix,
            reduced_runs=reduced_matrix,
            full_runs=full_matrix,
            n_permutations=n_permutations,
            block_size=block_size,
            random_state=random_state + 1_000_003 * contrast_index,
        )
        null_store[contrast] = pooled_null.astype(np.float32)
        sorted_null = np.sort(np.asarray(pooled_null, dtype=float))
        exceedances = len(sorted_null) - np.searchsorted(
            sorted_null, observed_delta, side="left"
        )
        p_values = (1.0 + exceedances) / (1.0 + len(sorted_null))
        q_values = COMMON.fdr_bh(p_values)
        COMMON.atomic_save_npz(
            output_dir / f"voxel_stats_{contrast}.npz",
            coords=coordinates.astype(np.int32),
            observed_delta_cv_r2=observed_delta.astype(np.float32),
            p_one_sided=np.asarray(p_values, dtype=np.float32),
            q_bh=np.asarray(q_values, dtype=np.float32),
        )
        thresholded = np.zeros(shape, dtype=np.float32)
        significant = (
            np.isfinite(q_values)
            & (q_values < fdr_threshold)
            & (observed_delta > 0)
        )
        if np.any(significant):
            x, y, z = coordinates[significant].T
            thresholded[x, y, z] = observed_delta[significant].astype(np.float32)
        nib.save(
            nib.Nifti1Image(thresholded, reference.affine, reference.header),
            str(
                output_dir
                / f"thresholded_delta_cv_r2_{contrast}_q{fdr_threshold:g}.nii.gz"
            ),
        )
        display_volume = display_smooth_thresholded_encoding(
            thresholded,
            thresholded > 0,
            reference,
            display_smoothing_fwhm,
        )
        plot_thresholded_encoding_brain(
            display_volume,
            anatomical,
            thresholded > 0,
            reference,
            f"Unique held-out prediction · {contrast_labels[contrast]}",
            figure_dir / f"delta_cv_r2_thresholded_brain_{contrast}.png",
            fdr_threshold=fdr_threshold,
            smoothing_fwhm=display_smoothing_fwhm,
            statistic_label="Held-out ΔCV R²",
        )

        coordinate_to_index = {
            tuple(int(value) for value in coordinate): index
            for index, coordinate in enumerate(coordinates)
        }
        for roi_name in voxel_stats:
            _, roi_coords = COMMON.load_roi(run_specs[0], roi_name)
            global_indices = np.asarray(
                [
                    coordinate_to_index[
                        tuple(int(value) for value in coordinate)
                    ]
                    for coordinate in roi_coords
                ],
                dtype=int,
            )
            tested = np.isfinite(q_values[global_indices])
            significant_roi = significant[global_indices] & tested
            n_tested = int(np.sum(tested))
            n_significant = int(np.sum(significant_roi))
            fraction_rows.append(
                {
                    "contrast": contrast,
                    "contrast_label": contrast_labels[contrast],
                    "roi": roi_name,
                    "n_roi_voxels": int(len(roi_coords)),
                    "n_tested_voxels": n_tested,
                    "n_significant_voxels": n_significant,
                    "significant_voxel_fraction": (
                        n_significant / n_tested if n_tested else np.nan
                    ),
                    "mean_delta_cv_r2": float(
                        np.nanmean(observed_delta[global_indices])
                    ),
                    "voxel_fdr_threshold": float(fdr_threshold),
                }
            )
        contrast_rows.append(
            {
                "contrast": contrast,
                "contrast_label": contrast_labels[contrast],
                "reduced_model": reduced_name,
                "full_model": full_name,
                "n_unique_atlas_voxels": int(len(coordinates)),
                "n_significant_voxels": int(np.sum(significant)),
                "significant_voxel_fraction": float(np.mean(significant)),
                "mean_delta_cv_r2": float(np.nanmean(observed_delta)),
                "median_delta_cv_r2": float(np.nanmedian(observed_delta)),
                "n_selected_voxels": int(selected_voxels),
                "permutations_per_selected_voxel": int(n_permutations),
                "voxel_fdr_threshold": float(fdr_threshold),
            }
        )
        log(
            f"[voxel inference:{contrast}] unique={len(unique):,} "
            f"selected={selected_voxels} pooled-null={len(pooled_null):,} "
            f"q<{fdr_threshold:g}: {int(np.sum(significant)):,}"
        )

    pd.DataFrame(selection_rows).to_csv(
        output_dir / "selected_voxel_manifest.csv", index=False
    )
    fraction_df = pd.DataFrame(fraction_rows)
    fraction_df.to_csv(
        output_dir / "delta_cv_r2_significant_voxel_fraction_by_roi.csv",
        index=False,
    )
    pd.DataFrame(contrast_rows).to_csv(
        output_dir / "delta_cv_r2_contrast_summary.csv",
        index=False,
    )
    COMMON.atomic_save_npz(
        output_dir / "cross_style_pooled_nulls.npz", **null_store
    )
    plot_delta_roi_fraction_heatmap(
        fraction_df,
        figure_dir / "delta_cv_r2_significant_voxel_fraction_all_rois.png",
        contrast_definitions=contrast_definitions,
        contrast_labels=contrast_labels,
        fdr_threshold=fdr_threshold,
        n_permutations=n_permutations,
        selected_voxels=selected_voxels,
    )


def clean_roi_label(roi_name: str) -> str:
    return (
        str(roi_name)
        .replace("ho_cortical_", "")
        .replace("ho_subcortical_", "")
        .replace("_", " ")
    )


def load_encoding_anatomical_template(
    reference_img,
    reference_fmri_path: Path,
    requested_path: Optional[Path],
) -> Tuple[np.ndarray, str]:
    """Load a clean anatomical underlay and resample it to the analysis grid."""
    import nibabel as nib
    from nibabel.processing import resample_from_to

    candidates: List[Path] = []
    if requested_path is not None:
        requested = Path(requested_path)
        if not requested.exists():
            raise FileNotFoundError(
                f"Requested --brain-template does not exist: {requested}"
            )
        candidates.append(requested)
    candidates.extend(
        [
            reference_fmri_path.parent
            / "MNI152_T1_2.5x2.5x2.5mm_brain.nii",
            reference_fmri_path.parent
            / "MNI152_T1_2.5x2.5x2.5mm_brain.nii.gz",
        ]
    )
    template_path = next((path for path in candidates if path.exists()), None)
    if template_path is None:
        log(
            "[visualization] no anatomical template found; using temporal-mean "
            "fMRI as the underlay"
        )
        fmri = nib.load(str(reference_fmri_path))
        data = np.asarray(fmri.dataobj, dtype=np.float32)
        if data.ndim == 4:
            data = np.nanmean(data, axis=3)
        data[~np.isfinite(data)] = 0.0
        return data, f"{reference_fmri_path} (temporal mean fallback)"

    anatomical_img = nib.load(str(template_path))
    if anatomical_img.ndim > 3:
        anatomical_img = nib.Nifti1Image(
            np.asarray(anatomical_img.dataobj[..., 0], dtype=np.float32),
            anatomical_img.affine,
        )
    target = (
        tuple(int(value) for value in reference_img.shape[:3]),
        reference_img.affine,
    )
    if (
        anatomical_img.shape[:3] != reference_img.shape[:3]
        or not np.allclose(anatomical_img.affine, reference_img.affine)
    ):
        anatomical_img = resample_from_to(anatomical_img, target, order=1)
        source = f"{template_path} (resampled to analysis grid)"
    else:
        source = str(template_path)
    anatomical = np.asarray(anatomical_img.dataobj, dtype=np.float32)
    anatomical[~np.isfinite(anatomical)] = 0.0
    log(f"[visualization] anatomical underlay={source}")
    return anatomical, source


def display_smooth_thresholded_encoding(
    values: np.ndarray,
    significant: np.ndarray,
    reference_img,
    fwhm_mm: float,
) -> np.ndarray:
    """Smooth only after inference; the saved statistical mask stays raw."""
    if fwhm_mm <= 0 or not np.any(significant):
        return np.where(significant, values, 0.0).astype(np.float32)
    from scipy.ndimage import gaussian_filter

    voxel_sizes = np.asarray(reference_img.header.get_zooms()[:3], dtype=float)
    sigma = float(fwhm_mm) / (2.354820045 * voxel_sizes)
    thresholded = np.where(significant, values, 0.0).astype(np.float32)
    smoothed = gaussian_filter(thresholded, sigma=sigma, mode="constant")
    support = gaussian_filter(
        significant.astype(np.float32), sigma=sigma, mode="constant"
    )
    return np.where(support >= 0.01, smoothed, 0.0).astype(np.float32)


def world_axis_to_voxel(
    coordinate_mm: float,
    axis: int,
    reference_img,
) -> int:
    import nibabel as nib

    world = np.zeros(3, dtype=float)
    world[axis] = coordinate_mm
    voxel = nib.affines.apply_affine(
        np.linalg.inv(reference_img.affine), world
    )
    return int(
        np.clip(
            np.rint(voxel[axis]),
            0,
            reference_img.shape[axis] - 1,
        )
    )


def oriented_encoding_slice(
    volume: np.ndarray,
    axis: int,
    index: int,
) -> np.ndarray:
    if axis == 0:
        # Transpose preserves superior-at-top with origin="lower".
        return np.asarray(volume[index, :, :]).T
    if axis == 1:
        plane = volume[:, index, :]
    else:
        plane = volume[:, :, index]
    return np.rot90(np.asarray(plane))


def plot_thresholded_encoding_brain(
    stat_volume: np.ndarray,
    anatomical_volume: np.ndarray,
    significant_mask: np.ndarray,
    reference_img,
    title: str,
    output_path: Path,
    *,
    fdr_threshold: float,
    smoothing_fwhm: float,
    statistic_label: str = "Held-out r",
) -> None:
    """Cross Figure-3-like anatomical sagittal/axial summary."""
    import nibabel as nib
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    stat = np.asarray(stat_volume, dtype=np.float32)
    anatomical = np.asarray(anatomical_volume, dtype=np.float32)
    significant = np.asarray(significant_mask, dtype=bool)
    finite_values = stat[np.isfinite(stat) & (stat > 0)]
    if finite_values.size:
        sagittal_index = int(np.argmax(np.sum(np.abs(stat), axis=(1, 2))))
        axial_index = int(np.argmax(np.sum(np.abs(stat), axis=(0, 1))))
        vmin = float(np.min(finite_values))
        vmax = float(np.max(finite_values))
        if vmax <= vmin:
            vmax = vmin + 0.01
    else:
        sagittal_index = world_axis_to_voxel(0.0, 0, reference_img)
        axial_index = world_axis_to_voxel(40.0, 2, reference_img)
        vmin, vmax = 0.0, 0.01
    sagittal_mm = float(
        nib.affines.apply_affine(
            reference_img.affine, [sagittal_index, 0, 0]
        )[0]
    )
    axial_mm = float(
        nib.affines.apply_affine(
            reference_img.affine, [0, 0, axial_index]
        )[2]
    )
    positive_anatomy = anatomical[
        np.isfinite(anatomical) & (anatomical > 0)
    ]
    if positive_anatomy.size:
        anatomy_vmin, anatomy_vmax = np.quantile(
            positive_anatomy, [0.02, 0.995]
        )
    else:
        anatomy_vmin, anatomy_vmax = 0.0, 1.0

    panels = (
        (0, sagittal_index, f"Sagittal  x={sagittal_mm:+g} mm"),
        (2, axial_index, f"Axial  z={axial_mm:+g} mm"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.6), facecolor="white")
    for ax, (axis, index, coordinate_label) in zip(axes, panels):
        anatomy_slice = oriented_encoding_slice(anatomical, axis, index)
        stat_slice = oriented_encoding_slice(stat, axis, index)
        support = anatomy_slice > 0
        if np.any(support):
            rows, columns = np.where(support)
            padding = 3
            row_start = max(0, int(rows.min()) - padding)
            row_stop = min(
                anatomy_slice.shape[0], int(rows.max()) + padding + 1
            )
            column_start = max(0, int(columns.min()) - padding)
            column_stop = min(
                anatomy_slice.shape[1], int(columns.max()) + padding + 1
            )
            anatomy_slice = anatomy_slice[
                row_start:row_stop, column_start:column_stop
            ]
            stat_slice = stat_slice[
                row_start:row_stop, column_start:column_stop
            ]
        ax.imshow(
            np.ma.masked_where(anatomy_slice <= 0, anatomy_slice),
            cmap="gray",
            origin="lower",
            interpolation="bilinear",
            vmin=anatomy_vmin,
            vmax=anatomy_vmax,
        )
        ax.imshow(
            np.ma.masked_where(stat_slice <= 0, stat_slice),
            cmap=CROSS_RED_YELLOW,
            origin="lower",
            interpolation="bilinear",
            vmin=vmin,
            vmax=vmax,
            alpha=0.92,
        )
        ax.text(
            0.5,
            -0.055,
            coordinate_label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
            color=INK,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    colorbar_axis = fig.add_axes([0.32, 0.105, 0.36, 0.035])
    colorbar = fig.colorbar(
        ScalarMappable(
            norm=Normalize(vmin=vmin, vmax=vmax),
            cmap=CROSS_RED_YELLOW,
        ),
        cax=colorbar_axis,
        orientation="horizontal",
    )
    colorbar.set_ticks([vmin, vmax])
    colorbar.set_ticklabels([f"{vmin:.2f}", f"{vmax:.2f}"])
    colorbar.set_label(statistic_label, fontsize=9, labelpad=-1)
    colorbar.ax.tick_params(labelsize=8, length=0, pad=3)
    colorbar.outline.set_visible(False)
    fig.suptitle(
        title,
        x=0.5,
        y=0.975,
        ha="center",
        fontsize=12.5,
        color=INK,
    )
    fig.text(
        0.5,
        0.915,
        (
            "Cross pooled null · unsmoothed voxelwise inference · "
            f"{smoothing_fwhm:g} mm display smoothing"
        ),
        ha="center",
        va="top",
        fontsize=8.2,
        color=MUTED,
    )
    fig.text(
        0.5,
        0.175,
        f"BH-FDR q < {fdr_threshold:g}",
        ha="center",
        va="center",
        fontsize=9.5,
        color=INK,
    )
    if not np.any(significant):
        fig.text(
            0.5,
            0.49,
            "No voxels passed the corrected threshold",
            ha="center",
            va="center",
            fontsize=9,
            color=MUTED,
            bbox={
                "facecolor": "white",
                "edgecolor": "#D8DDE3",
                "boxstyle": "round,pad=0.4",
                "alpha": 0.95,
            },
        )
    fig.subplots_adjust(
        left=0.04,
        right=0.96,
        top=0.84,
        bottom=0.29,
        wspace=0.06,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_encoding_r_roi_fraction_heatmap(
    fraction_df: pd.DataFrame,
    output_path: Path,
    *,
    model_labels: Mapping[str, str],
    analysis_label: str,
    fdr_threshold: float,
    n_permutations: int,
    selected_voxels: int,
) -> None:
    """Plot every tested atlas ROI; do not silently retain only a top subset."""
    if fraction_df.empty:
        return
    pivot = fraction_df.pivot(
        index="roi", columns="model", values="significant_voxel_fraction"
    )
    model_order = [name for name in model_labels if name in pivot.columns]
    pivot = pivot[model_order]
    ordering = (
        pivot.max(axis=1)
        .sort_values(ascending=False, kind="mergesort")
        .index
    )
    pivot = pivot.loc[ordering]
    values = pivot.to_numpy(dtype=float)
    figure_height = max(9.0, 2.8 + 0.205 * len(pivot))
    figure_width = max(11.5, 5.8 + 0.85 * len(model_order))
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    image = ax.imshow(
        values,
        aspect="auto",
        interpolation="nearest",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
    )
    if not np.any(np.isfinite(values) & (values > 0)):
        ax.text(
            0.5,
            0.5,
            (
                "No positive voxels passed "
                f"voxelwise BH-FDR q < {fdr_threshold:g}"
            ),
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color=MUTED,
            bbox={
                "facecolor": "white",
                "edgecolor": "#D8DDE3",
                "boxstyle": "round,pad=0.5",
                "alpha": 0.95,
            },
        )
    ax.set_xticks(np.arange(len(model_order)))
    ax.set_xticklabels(
        [model_labels[name] for name in model_order],
        rotation=28,
        ha="right",
    )
    ax.set_yticks(np.arange(len(pivot)))
    ax.set_yticklabels(
        [clean_roi_label(name) for name in pivot.index],
        fontsize=6.2,
    )
    ax.set_xlabel(analysis_label)
    ax.set_ylabel("Harvard–Oxford ROI")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label("Proportion of significant voxels within ROI")
    fig.suptitle(
        "Cross-style significant encoding voxels by atlas ROI",
        x=0.5,
        y=0.995,
        ha="center",
        fontsize=15,
    )
    fig.text(
        0.5,
        0.972,
        (
            f"Held-out Pearson r; {selected_voxels} selected voxels × "
            f"{n_permutations:,} 40-TR permutations; voxelwise BH-FDR "
            f"q < {fdr_threshold:g}; denominator = tested ROI voxels"
        ),
        ha="center",
        va="top",
        fontsize=9,
        color="#59636E",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_delta_roi_fraction_heatmap(
    fraction_df: pd.DataFrame,
    output_path: Path,
    *,
    contrast_definitions: Mapping[str, Tuple[str, str]],
    contrast_labels: Mapping[str, str],
    fdr_threshold: float,
    n_permutations: int,
    selected_voxels: int,
) -> None:
    """ROI fractions for positive feature-unique ΔCV R² voxels."""
    if fraction_df.empty:
        return
    pivot = fraction_df.pivot(
        index="roi",
        columns="contrast",
        values="significant_voxel_fraction",
    )
    contrast_order = [
        name for name in contrast_definitions if name in pivot.columns
    ]
    pivot = pivot[contrast_order]
    ordering = (
        pivot.max(axis=1)
        .sort_values(ascending=False, kind="mergesort")
        .index
    )
    pivot = pivot.loc[ordering]
    values = pivot.to_numpy(dtype=float)
    figure_height = max(9.0, 2.8 + 0.205 * len(pivot))
    figure_width = max(11.5, 5.8 + 0.85 * len(contrast_order))
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    image = ax.imshow(
        values,
        aspect="auto",
        interpolation="nearest",
        cmap="Purples",
        vmin=0.0,
        vmax=1.0,
    )
    if not np.any(np.isfinite(values) & (values > 0)):
        ax.text(
            0.5,
            0.5,
            (
                "No positive voxels passed "
                f"voxelwise BH-FDR q < {fdr_threshold:g}"
            ),
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color=MUTED,
            bbox={
                "facecolor": "white",
                "edgecolor": "#D8DDE3",
                "boxstyle": "round,pad=0.5",
                "alpha": 0.95,
            },
        )
    ax.set_xticks(np.arange(len(contrast_order)))
    ax.set_xticklabels(
        [contrast_labels[name] for name in contrast_order],
        rotation=28,
        ha="right",
    )
    ax.set_yticks(np.arange(len(pivot)))
    ax.set_yticklabels(
        [clean_roi_label(name) for name in pivot.index],
        fontsize=6.2,
    )
    ax.set_xlabel("Controlled feature-unique contrast")
    ax.set_ylabel("Harvard–Oxford ROI")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label("Proportion of significant positive voxels within ROI")
    fig.suptitle(
        "Controlled feature-unique encoding voxels by atlas ROI",
        x=0.5,
        y=0.995,
        ha="center",
        fontsize=15,
    )
    fig.text(
        0.5,
        0.972,
        (
            f"Held-out ΔCV R²; {selected_voxels} selected voxels × "
            f"{n_permutations:,} block permutations; voxelwise BH-FDR "
            f"q < {fdr_threshold:g}"
        ),
        ha="center",
        va="top",
        fontsize=9,
        color="#59636E",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_encoding_r_null_distributions(
    null_store: Mapping[str, np.ndarray],
    observed_store: Mapping[str, np.ndarray],
    q_store: Mapping[str, np.ndarray],
    output_path: Path,
    *,
    model_labels: Mapping[str, str],
    analysis_label: str,
    fdr_threshold: float,
    n_permutations: int,
    selected_voxels: int,
    block_size: int,
) -> None:
    if not null_store:
        return
    model_order = [name for name in model_labels if name in null_store]
    combined = np.concatenate(
        [
            np.asarray(null_store[name], dtype=float)
            for name in model_order
        ]
        + [
            np.asarray(observed_store[name], dtype=float)
            for name in model_order
        ]
    )
    combined = combined[np.isfinite(combined)]
    if combined.size:
        x_min, x_max = np.quantile(combined, [0.0005, 0.9995])
        padding = max(0.01, 0.05 * float(x_max - x_min))
        x_limits = (float(x_min - padding), float(x_max + padding))
    else:
        x_limits = (-0.2, 0.2)

    n_columns = min(3, max(1, len(model_order)))
    n_rows = int(math.ceil(len(model_order) / n_columns))
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(4.1 * n_columns, 3.25 * n_rows + 1.5),
        sharex=True,
        squeeze=False,
    )
    axes_flat = axes.ravel()
    for axis_index, model_name in enumerate(model_order):
        ax = axes_flat[axis_index]
        null = np.asarray(null_store[model_name], dtype=float)
        null = null[np.isfinite(null)]
        observed = np.asarray(observed_store[model_name], dtype=float)
        observed = observed[np.isfinite(observed)]
        ax.hist(
            null,
            bins=90,
            range=x_limits,
            density=True,
            color="#4C78A8",
            alpha=0.72,
            linewidth=0,
            label="Pooled block null",
        )
        ax.hist(
            observed,
            bins=70,
            range=x_limits,
            density=True,
            histtype="step",
            color="#404040",
            linewidth=1.4,
            label="Observed atlas voxels",
        )
        raw_tail = float(np.quantile(null, 1.0 - fdr_threshold))
        ax.axvline(
            raw_tail,
            color="#E3A008",
            linewidth=1.3,
            linestyle="--",
            label=f"Null {100 * (1 - fdr_threshold):.2f}th percentile",
        )
        q_values = np.asarray(q_store[model_name], dtype=float)
        observed_all = np.asarray(observed_store[model_name], dtype=float)
        significant = np.isfinite(q_values) & (q_values < fdr_threshold)
        if np.any(significant):
            fdr_cutoff = float(np.nanmin(observed_all[significant]))
            ax.axvline(
                fdr_cutoff,
                color="#B24C63",
                linewidth=1.4,
                linestyle="-",
                label="Smallest FDR-significant r",
            )
        ax.set_title(model_labels[model_name], fontsize=10.5)
        ax.set_xlim(*x_limits)
        ax.set_ylabel("Density")
        ax.grid(axis="y", color="#D8DDE3", linewidth=0.6, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes_flat[len(model_order) :]:
        ax.axis("off")
    for ax in axes[-1]:
        if ax.get_visible():
            ax.set_xlabel("Held-out voxelwise Pearson r")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=2,
        frameon=False,
        fontsize=8.5,
    )
    fig.suptitle(
        f"Cross-style encoding-r pooled nulls · {analysis_label}",
        x=0.5,
        y=0.995,
        ha="center",
        fontsize=15,
    )
    fig.text(
        0.5,
        0.968,
        (
            f"{selected_voxels} selected voxels × {n_permutations:,} "
            f"independent {block_size}-TR block permutations per model; "
            "observed distributions include unique atlas coordinates"
        ),
        ha="center",
        va="top",
        fontsize=9,
        color="#59636E",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_cross_style_encoding_r_inference(
    *,
    run_specs: Sequence[COMMON.BrainRunSpec],
    voxel_stats: Mapping[str, Mapping[str, np.ndarray]],
    design_dir: Path,
    output_dir: Path,
    figure_dir: Path,
    model_labels: Mapping[str, str],
    analysis_label: str,
    n_permutations: int,
    block_size: int,
    top_fraction: float,
    selected_voxels: int,
    fdr_threshold: float,
    random_state: int,
    brain_template: Optional[Path] = None,
    display_smoothing_fwhm: float = 5.0,
    threshold_maps_only: bool = False,
) -> None:
    """Cross Figure-3 analogue based on held-out encoding correlations."""
    import nibabel as nib

    output_dir.mkdir(parents=True, exist_ok=True)
    reference = nib.load(str(run_specs[0].fmri_path))
    shape = tuple(int(value) for value in reference.shape[:3])
    anatomical, anatomical_source = load_encoding_anatomical_template(
        reference,
        run_specs[0].fmri_path,
        brain_template,
    )
    null_store: Dict[str, np.ndarray] = {}
    observed_store: Dict[str, np.ndarray] = {}
    q_store: Dict[str, np.ndarray] = {}
    selection_rows: List[Dict[str, object]] = []
    fraction_rows: List[Dict[str, object]] = []
    model_rows: List[Dict[str, object]] = []

    for model_index, model_name in enumerate(model_labels):
        unique: Dict[Tuple[int, int, int], Tuple[str, int, float]] = {}
        for roi_name, statistics in voxel_stats.items():
            _, coords = COMMON.load_roi(run_specs[0], roi_name)
            observed_r = np.asarray(
                statistics[f"encoding_r_{model_name}"], dtype=float
            )
            if len(observed_r) != len(coords):
                raise ValueError(
                    f"Encoding-r coordinate mismatch: {roi_name}:{model_name}"
                )
            for voxel_index, coordinate in enumerate(coords):
                key = tuple(int(value) for value in coordinate)
                if key not in unique:
                    unique[key] = (
                        roi_name,
                        int(voxel_index),
                        float(observed_r[voxel_index]),
                    )
        if len(unique) < selected_voxels:
            raise ValueError(
                f"{model_name}: only {len(unique)} unique atlas voxels; "
                f"need at least {selected_voxels}"
            )
        coordinates = np.asarray(list(unique), dtype=np.int32)
        records = list(unique.values())
        observed_r = np.asarray([record[2] for record in records], dtype=float)
        finite_scores = observed_r[np.isfinite(observed_r)]
        if not finite_scores.size:
            raise ValueError(f"{model_name}: no finite encoding-r values")
        cutoff = float(
            np.quantile(finite_scores, 1.0 - top_fraction)
        )
        candidates = np.flatnonzero(
            np.isfinite(observed_r) & (observed_r >= cutoff)
        )
        if len(candidates) < selected_voxels:
            raise ValueError(
                f"{model_name}: top-fraction pool has only "
                f"{len(candidates)} voxels"
            )
        selection_seed = int(random_state + 2_000_003 * model_index)
        rng = np.random.default_rng(selection_seed)
        selected_global = np.sort(
            rng.choice(candidates, selected_voxels, replace=False)
        )
        by_roi: Dict[str, List[Tuple[int, int]]] = {}
        for selected_order, global_index in enumerate(selected_global):
            roi_name, voxel_index, score = records[int(global_index)]
            by_roi.setdefault(roi_name, []).append(
                (selected_order, int(voxel_index))
            )
            x, y, z = coordinates[int(global_index)]
            selection_rows.append(
                {
                    "model": model_name,
                    "model_label": model_labels[model_name],
                    "selected_order": int(selected_order),
                    "roi": roi_name,
                    "roi_voxel_index": int(voxel_index),
                    "x": int(x),
                    "y": int(y),
                    "z": int(z),
                    "observed_encoding_r": float(score),
                    "top_fraction": float(top_fraction),
                    "candidate_cutoff_r": cutoff,
                    "selection_seed": selection_seed,
                    "permutation_seed": int(
                        random_state + 3_000_017 * model_index
                    ),
                }
            )

        run_slot_y: List[List[Optional[np.ndarray]]] = [
            [None] * selected_voxels for _ in run_specs
        ]
        run_slot_prediction: List[List[Optional[np.ndarray]]] = [
            [None] * selected_voxels for _ in run_specs
        ]
        for roi_name, selected_pairs in by_roi.items():
            y_runs: List[np.ndarray] = []
            for run in run_specs:
                patterns, _ = COMMON.load_roi(run, roi_name)
                y_runs.append(patterns)
            predictions, _ = predict_roi_model(
                model_name=model_name,
                run_specs=run_specs,
                y_runs=y_runs,
                design_dir=design_dir,
            )
            for selected_order, voxel_index in selected_pairs:
                for run_index in range(len(run_specs)):
                    run_slot_y[run_index][selected_order] = y_runs[run_index][
                        :, voxel_index
                    ]
                    run_slot_prediction[run_index][selected_order] = predictions[
                        run_index
                    ][:, voxel_index]
        y_matrix: List[np.ndarray] = []
        prediction_matrix: List[np.ndarray] = []
        for run_index in range(len(run_specs)):
            if any(value is None for value in run_slot_y[run_index]) or any(
                value is None
                for value in run_slot_prediction[run_index]
            ):
                raise RuntimeError(
                    f"Incomplete selected encoding-r voxel assembly: {model_name}"
                )
            y_matrix.append(
                np.column_stack(
                    [np.asarray(value) for value in run_slot_y[run_index]]
                )
            )
            prediction_matrix.append(
                np.column_stack(
                    [
                        np.asarray(value)
                        for value in run_slot_prediction[run_index]
                    ]
                )
            )

        permutation_seed = int(random_state + 3_000_017 * model_index)
        pooled_null = pooled_encoding_r_voxel_null(
            y_runs=y_matrix,
            prediction_runs=prediction_matrix,
            n_permutations=n_permutations,
            block_size=block_size,
            random_state=permutation_seed,
        )
        finite_null = np.asarray(pooled_null, dtype=float)
        finite_null = finite_null[np.isfinite(finite_null)]
        if not finite_null.size:
            raise RuntimeError(f"{model_name}: encoding-r pooled null is empty")
        sorted_null = np.sort(finite_null)
        p_values = np.full(observed_r.shape, np.nan, dtype=float)
        finite_observed = np.isfinite(observed_r)
        exceedances = len(sorted_null) - np.searchsorted(
            sorted_null, observed_r[finite_observed], side="left"
        )
        p_values[finite_observed] = (
            1.0 + exceedances
        ) / (1.0 + len(sorted_null))
        q_values = COMMON.fdr_bh(p_values)
        significant = (
            np.isfinite(q_values)
            & (q_values < fdr_threshold)
            & (observed_r > 0)
        )

        COMMON.atomic_save_npz(
            output_dir / f"encoding_r_voxel_stats_{model_name}.npz",
            coords=coordinates.astype(np.int32),
            observed_encoding_r=observed_r.astype(np.float32),
            p_one_sided=np.asarray(p_values, dtype=np.float32),
            q_bh=np.asarray(q_values, dtype=np.float32),
            significant_q_threshold=significant.astype(np.uint8),
        )
        thresholded = np.zeros(shape, dtype=np.float32)
        if np.any(significant):
            x, y, z = coordinates[significant].T
            thresholded[x, y, z] = observed_r[significant].astype(np.float32)
        nib.save(
            nib.Nifti1Image(thresholded, reference.affine, reference.header),
            str(
                output_dir
                / f"thresholded_encoding_r_{model_name}_q{fdr_threshold:g}.nii.gz"
            ),
        )
        display_volume = display_smooth_thresholded_encoding(
            thresholded,
            thresholded > 0,
            reference,
            display_smoothing_fwhm,
        )
        plot_thresholded_encoding_brain(
            display_volume,
            anatomical,
            thresholded > 0,
            reference,
            f"Held-out encoding r · {model_labels[model_name]}",
            # Labels identify the single feature in feature-only mode and the
            # cumulative block in the controlled mode.
            figure_dir / f"encoding_r_thresholded_brain_{model_name}.png",
            fdr_threshold=fdr_threshold,
            smoothing_fwhm=display_smoothing_fwhm,
        )

        coordinate_to_index = {
            tuple(int(value) for value in coordinate): index
            for index, coordinate in enumerate(coordinates)
        }
        for roi_name in voxel_stats:
            _, roi_coords = COMMON.load_roi(run_specs[0], roi_name)
            global_indices = np.asarray(
                [
                    coordinate_to_index[
                        tuple(int(value) for value in coordinate)
                    ]
                    for coordinate in roi_coords
                ],
                dtype=int,
            )
            tested = np.isfinite(q_values[global_indices])
            significant_roi = significant[global_indices] & tested
            n_tested = int(np.sum(tested))
            n_significant = int(np.sum(significant_roi))
            fraction_rows.append(
                {
                    "model": model_name,
                    "model_label": model_labels[model_name],
                    "roi": roi_name,
                    "n_roi_voxels": int(len(roi_coords)),
                    "n_tested_voxels": n_tested,
                    "n_significant_voxels": n_significant,
                    "significant_voxel_fraction": (
                        n_significant / n_tested if n_tested else np.nan
                    ),
                    "significant_voxel_percent": (
                        100.0 * n_significant / n_tested
                        if n_tested
                        else np.nan
                    ),
                    "mean_encoding_r": float(
                        np.nanmean(observed_r[global_indices])
                    ),
                    "median_encoding_r": float(
                        np.nanmedian(observed_r[global_indices])
                    ),
                    "voxel_fdr_threshold": float(fdr_threshold),
                    "denominator_definition": (
                        "finite-q voxels in this atlas ROI"
                    ),
                    "overlap_policy": (
                        "global coordinate-level q reused in each containing ROI"
                    ),
                }
            )

        model_rows.append(
            {
                "model": model_name,
                "model_label": model_labels[model_name],
                "n_unique_atlas_voxels": int(len(coordinates)),
                "n_finite_tested_voxels": int(np.sum(finite_observed)),
                "n_significant_voxels": int(np.sum(significant)),
                "significant_voxel_fraction": float(
                    np.mean(significant[finite_observed])
                ),
                "mean_encoding_r": float(np.nanmean(observed_r)),
                "median_encoding_r": float(np.nanmedian(observed_r)),
                "selection_cutoff_r": cutoff,
                "n_selection_candidates": int(len(candidates)),
                "n_selected_voxels": int(selected_voxels),
                "null_values": int(len(finite_null)),
                "permutations_per_selected_voxel": int(n_permutations),
                "block_size_tr": int(block_size),
                "voxel_fdr_threshold": float(fdr_threshold),
                "anatomical_underlay": anatomical_source,
                "display_smoothing_fwhm_mm": float(
                    display_smoothing_fwhm
                ),
            }
        )
        null_store[model_name] = np.asarray(pooled_null, dtype=np.float32)
        observed_store[model_name] = observed_r.astype(np.float32)
        q_store[model_name] = np.asarray(q_values, dtype=np.float32)
        log(
            f"[encoding-r inference:{model_name}] unique={len(unique):,} "
            f"selected={selected_voxels} pooled-null={len(finite_null):,} "
            f"q<{fdr_threshold:g}: {int(np.sum(significant)):,}"
        )

    fraction_df = pd.DataFrame(fraction_rows)
    pd.DataFrame(selection_rows).to_csv(
        output_dir / "encoding_r_selected_voxel_manifest.csv", index=False
    )
    pd.DataFrame(model_rows).to_csv(
        output_dir / "encoding_r_model_summary.csv", index=False
    )
    fraction_df.to_csv(
        output_dir / "encoding_r_significant_voxel_fraction_by_roi.csv",
        index=False,
    )
    COMMON.atomic_save_npz(
        output_dir / "cross_style_encoding_r_pooled_nulls.npz",
        **null_store,
    )
    if not threshold_maps_only:
        plot_encoding_r_roi_fraction_heatmap(
            fraction_df,
            figure_dir / "encoding_r_significant_voxel_fraction_all_rois.png",
            model_labels=model_labels,
            analysis_label=analysis_label,
            fdr_threshold=fdr_threshold,
            n_permutations=n_permutations,
            selected_voxels=selected_voxels,
        )
        plot_encoding_r_null_distributions(
            null_store,
            observed_store,
            q_store,
            figure_dir / "encoding_r_pooled_null_distributions.png",
            model_labels=model_labels,
            analysis_label=analysis_label,
            fdr_threshold=fdr_threshold,
            n_permutations=n_permutations,
            selected_voxels=selected_voxels,
            block_size=block_size,
        )


def build_unthresholded_maps(
    *,
    run_specs: Sequence[COMMON.BrainRunSpec],
    voxel_stats: Mapping[str, Mapping[str, np.ndarray]],
    output_dir: Path,
) -> None:
    import nibabel as nib

    reference = nib.load(str(run_specs[0].fmri_path))
    shape = tuple(int(v) for v in reference.shape[:3])
    output_dir.mkdir(parents=True, exist_ok=True)
    map_names = sorted(
        {
            statistic
            for roi_statistics in voxel_stats.values()
            for statistic in roi_statistics
        }
    )
    for statistic in map_names:
        total = np.zeros(shape, dtype=np.float64)
        count = np.zeros(shape, dtype=np.int16)
        for roi_name, roi_statistics in voxel_stats.items():
            if statistic not in roi_statistics:
                continue
            _, coords = COMMON.load_roi(run_specs[0], roi_name)
            values = np.asarray(roi_statistics[statistic], dtype=float)
            if len(values) != len(coords):
                raise ValueError(f"Map length mismatch: {roi_name}:{statistic}")
            x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
            np.add.at(total, (x, y, z), values)
            np.add.at(count, (x, y, z), 1)
        data = np.zeros(shape, dtype=np.float32)
        nonzero = count > 0
        data[nonzero] = (total[nonzero] / count[nonzero]).astype(np.float32)
        nib.save(
            nib.Nifti1Image(data, reference.affine, reference.header),
            str(output_dir / f"{COMMON.safe_name(statistic)}.nii.gz"),
        )


def plot_roi_delta_heatmap(
    summary: pd.DataFrame,
    output_path: Path,
    *,
    contrast_definitions: Mapping[str, Tuple[str, str]] = CONTRASTS,
    contrast_labels: Optional[Mapping[str, str]] = None,
) -> None:
    if summary.empty:
        return
    pivot = summary.pivot(index="roi", columns="contrast", values="mean_delta_cv_r2")
    contrast_order = [
        name for name in contrast_definitions if name in pivot.columns
    ]
    if not contrast_order:
        return
    pivot = pivot[contrast_order]
    ordering = pivot.abs().max(axis=1).sort_values(ascending=False).index[:30]
    pivot = pivot.loc[ordering]
    values = pivot.to_numpy(dtype=float)
    finite = np.abs(values[np.isfinite(values)])
    vmax = max(float(np.quantile(finite, 0.98)) if finite.size else 0.0, 1e-5)
    fig, ax = plt.subplots(figsize=(11.5, max(6.0, len(pivot) * 0.28 + 2.0)))
    image = ax.imshow(values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(contrast_order)))
    ax.set_xticklabels(
        [
            (
                contrast_labels[name]
                if contrast_labels is not None and name in contrast_labels
                else name.replace("_", " ")
            )
            for name in contrast_order
        ],
        rotation=25,
        ha="right",
    )
    ax.set_yticks(np.arange(len(pivot)))
    ax.set_yticklabels(
        [
            name.replace("ho_cortical_", "").replace("ho_subcortical_", "")
            for name in pivot.index
        ],
        fontsize=7,
    )
    ax.set_title("Nested LORO planning-process encoding")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label("Across-run mean ΔCV R²")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    COMMON.add_shared_cli_arguments(parser, require_subject_game=False)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--analysis-mode",
        choices=("localization", "predictive"),
        default="localization",
        help=(
            "Default localization fits controlled run-wise beta maps.  The "
            "legacy nested LORO delta-CV-R2 engine is retained as predictive."
        ),
    )
    parser.add_argument(
        "--predictive-encoding",
        action="store_true",
        help="Compatibility alias for --analysis-mode predictive",
    )
    parser.add_argument(
        "--aggregate-subjects",
        nargs="+",
        type=int,
        default=None,
        help="Aggregate completed localization outputs; final inference requires 1 2 3 4 5 6",
    )
    parser.add_argument(
        "--aggregate-games",
        action="store_true",
        help=(
            "After both games are complete, form within-subject across-game "
            "means and Game2-Game1 contrasts before exact group inference"
        ),
    )
    parser.add_argument(
        "--localization-tier",
        choices=("marginal", "partial", "both"),
        default="both",
        help="Fit controls+target, staged upstream+target, or both localization tiers",
    )
    parser.add_argument(
        "--state-control",
        choices=("visual", "visual+ram"),
        default="visual",
        help="Primary sensory control or RAM-added sensitivity control",
    )
    parser.add_argument("--ram-pca-dim", type=int, default=20)
    parser.add_argument("--visual-pca-dim", type=int, default=20)
    parser.add_argument("--alphas-log", nargs=2, type=float, default=[-2, 5])
    parser.add_argument("--n-alphas", type=int, default=12)
    parser.add_argument("--alpha-selection-voxels", type=int, default=512)
    parser.add_argument("--include-secondary-geometry", action="store_true")
    parser.add_argument(
        "--actor-planning-only",
        action="store_true",
        help=(
            "Fit the five final Section-3 actor-policy-change/search-geometry "
            "features separately with no controls (default analysis)"
        ),
    )
    parser.add_argument(
        "--legacy-cumulative",
        action="store_true",
        help="Run the former no-flag cumulative controlled analysis",
    )
    parser.add_argument(
        "--feature-only",
        action="store_true",
        help=(
            "Fit separate one-variable models without RAM, visual, action, "
            "Q-state, or nuisance controls (Cross Figure-3 localization tier)"
        ),
    )
    parser.add_argument(
        "--controlled-unique",
        action="store_true",
        help=(
            "Fit full/reduced controlled models for unique ΔCV R² of each "
            "planning feature family"
        ),
    )
    parser.add_argument(
        "--expanded-feature-only",
        action="store_true",
        help=(
            "Run the existing individual localization models plus outcome "
            "mean/dispersion and SR/VP in feature_only_expanded/"
        ),
    )
    parser.add_argument(
        "--expanded-controlled-unique",
        action="store_true",
        help=(
            "Run existing feature-unique contrasts plus non-redundant 05 "
            "outcome and SR/VP contrasts in controlled_unique_expanded/"
        ),
    )
    parser.add_argument(
        "--rebuild-design-cache",
        action="store_true",
        help="Ignore compatible nested design files and refit every outer fold",
    )
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--permutation-block-size", type=int, default=40)
    parser.add_argument("--permutation-voxels", type=int, default=64)
    parser.add_argument(
        "--cross-voxel-permutations",
        type=int,
        default=0,
        help=(
            "Optional Cross-style pooled-null stage; use 100000 for final "
            "atlas-coverage voxel inference"
        ),
    )
    parser.add_argument("--cross-selected-voxels", type=int, default=14)
    parser.add_argument("--cross-top-fraction", type=float, default=0.10)
    parser.add_argument("--voxel-fdr-threshold", type=float, default=0.0001)
    parser.add_argument(
        "--brain-template",
        type=Path,
        default=None,
        help=(
            "Optional anatomical NIfTI underlay for thresholded encoding-r "
            "figures; defaults to the run-local MNI152 template"
        ),
    )
    parser.add_argument(
        "--display-smoothing-fwhm",
        type=float,
        default=5.0,
        help="Display-only smoothing for thresholded brain figures in mm",
    )
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument(
        "--overwrite-localization",
        action="store_true",
        help=(
            "Explicitly replace an incompatible/incomplete localization "
            "configuration namespace; never implied by cache overwrite flags."
        ),
    )
    parser.add_argument("--self-test", action="store_true")
    return parser


def run_self_test() -> None:
    common = COMMON.self_test()
    localization = LOCALIZATION.self_test()
    rng = np.random.default_rng(11)
    x = rng.normal(size=(300, 8))
    weights = rng.normal(size=(8, 12))
    y = x @ weights + rng.normal(scale=0.2, size=(300, 12))
    prediction = Ridge(alpha=1.0).fit(x[:200], y[:200]).predict(x[200:])
    score = float(np.mean(pearson_columns(y[200:], prediction)))
    if score < 0.8:
        raise AssertionError(f"Synthetic encoding failed: r={score}")
    y_runs = [rng.normal(size=(80, 3)) for _ in range(2)]
    reduced_runs = [
        0.3 * values + rng.normal(scale=0.5, size=values.shape)
        for values in y_runs
    ]
    full_runs = [
        reduced + 0.2 * values
        for values, reduced in zip(y_runs, reduced_runs)
    ]
    roi_null = block_permutation_roi_delta(
        y_runs=y_runs,
        reduced_predictions=reduced_runs,
        full_predictions=full_runs,
        n_permutations=5,
        block_size=20,
        max_voxels=3,
        random_state=19,
    )
    roi_rng = np.random.default_rng(19)
    roi_orders = [
        np.argsort(
            roi_rng.random((5, len(values) // 20)), axis=1
        )
        for values in y_runs
    ]
    roi_brute = np.empty(5, dtype=float)
    for permutation in range(5):
        permuted_runs: List[np.ndarray] = []
        for run_index, (values, reduced) in enumerate(
            zip(y_runs, reduced_runs)
        ):
            residual_blocks = (values - reduced).reshape(-1, 20, 3)
            permuted_runs.append(
                reduced
                + residual_blocks[
                    roi_orders[run_index][permutation]
                ].reshape(len(values), 3)
            )
        permuted_y = np.concatenate(permuted_runs, axis=0)
        reduced_y = np.concatenate(reduced_runs, axis=0)
        full_y = np.concatenate(full_runs, axis=0)
        roi_brute[permutation] = float(
            np.nanmean(
                cv_r2_columns(permuted_y, full_y)
                - cv_r2_columns(permuted_y, reduced_y)
            )
        )
    roi_null_error = float(np.max(np.abs(roi_null - roi_brute)))
    if roi_null_error > 1e-6:
        raise AssertionError(
            f"Fast ROI block null mismatch: {roi_null_error}"
        )
    pooled = pooled_conditional_voxel_null(
        y_runs=y_runs,
        reduced_runs=reduced_runs,
        full_runs=full_runs,
        n_permutations=5,
        block_size=20,
        random_state=13,
        batch_size=5,
    )
    if pooled.shape != (15,) or not np.all(np.isfinite(pooled)):
        raise AssertionError("Synthetic pooled voxel null failed")
    encoding_null = pooled_encoding_r_voxel_null(
        y_runs=y_runs,
        prediction_runs=full_runs,
        n_permutations=5,
        block_size=20,
        random_state=17,
        batch_size=5,
    )
    if encoding_null.shape != (15,) or not np.all(np.isfinite(encoding_null)):
        raise AssertionError("Synthetic pooled encoding-r null failed")

    # Independently reconstruct the same permutations by materializing the
    # shuffled BOLD blocks.  This checks the optimized block-dot-product path.
    brute_rng = np.random.default_rng(17)
    orders_by_run: List[np.ndarray] = []
    for values in y_runs:
        n_blocks = len(values) // 20
        orders_by_run.append(
            np.argsort(brute_rng.random((5, 3, n_blocks)), axis=2)
        )
    brute = np.empty((5, 3), dtype=float)
    for permutation in range(5):
        permuted_runs: List[np.ndarray] = []
        for run_index, values in enumerate(y_runs):
            blocks = values.reshape(-1, 20, 3)
            permuted = np.empty_like(blocks)
            for voxel in range(3):
                permuted[:, :, voxel] = blocks[
                    orders_by_run[run_index][permutation, voxel], :, voxel
                ]
            permuted_runs.append(permuted.reshape(len(values), 3))
        brute[permutation] = pearson_columns(
            np.concatenate(permuted_runs, axis=0),
            np.concatenate(full_runs, axis=0),
        )
    max_null_error = float(
        np.max(np.abs(encoding_null.reshape(5, 3) - brute))
    )
    if max_null_error > 1e-6:
        raise AssertionError(
            f"Fast pooled encoding-r null mismatch: {max_null_error}"
        )
    print(
        json.dumps(
            {
                **common,
                **{f"localization_{key}": value for key, value in localization.items()},
                "encoding_mean_r": score,
                "roi_block_null_max_error": roi_null_error,
                "pooled_voxel_null_values": int(len(pooled)),
                "pooled_encoding_r_null_values": int(len(encoding_null)),
                "pooled_encoding_r_max_error": max_null_error,
            },
            indent=2,
        )
    )


def main() -> None:
    args = build_argparser().parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.predictive_encoding:
        args.analysis_mode = "predictive"
    if args.analysis_mode == "localization":
        if not args.roi_include:
            # Localization asks where a variable is represented.  Use the full
            # atlas and every cached voxel by default; restricted ROI runs are
            # explicit sensitivities rather than the main analysis.
            args.roi_set = "all"
            args.max_roi_voxels = 0
        if args.aggregate_games:
            if args.aggregate_subjects or args.subject is not None or args.game is not None:
                raise ValueError("--aggregate-games is standalone; omit subject/game arguments")
            LOCALIZATION.run_paired_games(args)
        elif args.aggregate_subjects:
            LOCALIZATION.run_group(args)
        else:
            if args.subject is None or args.game is None:
                raise ValueError("Localization subject mode requires --subject and --game")
            LOCALIZATION.run_subject(args)
        return
    if args.ram_pca_dim < 1:
        raise ValueError("--ram-pca-dim must be positive")
    if args.visual_pca_dim < 1:
        raise ValueError("--visual-pca-dim must be positive")
    if args.n_alphas < 2:
        raise ValueError("--n-alphas must be >= 2")
    if args.alpha_selection_voxels < 1:
        raise ValueError("--alpha-selection-voxels must be positive")
    if args.n_permutations < 0:
        raise ValueError("--n-permutations must be non-negative")
    if args.cross_voxel_permutations < 0:
        raise ValueError("--cross-voxel-permutations must be non-negative")
    if args.cross_selected_voxels < 1:
        raise ValueError("--cross-selected-voxels must be positive")
    if not 0.0 < args.cross_top_fraction <= 1.0:
        raise ValueError("--cross-top-fraction must be in (0, 1]")
    if not 0.0 < args.voxel_fdr_threshold < 1.0:
        raise ValueError("--voxel-fdr-threshold must be in (0, 1)")
    if args.display_smoothing_fwhm < 0:
        raise ValueError("--display-smoothing-fwhm must be non-negative")
    requested_tiers = [
        args.actor_planning_only,
        args.feature_only,
        args.controlled_unique,
        args.expanded_feature_only,
        args.expanded_controlled_unique,
    ]
    if sum(bool(value) for value in requested_tiers) > 1:
        raise ValueError(
            "Choose at most one of --actor-planning-only, --feature-only, "
            "--controlled-unique, --expanded-feature-only, and "
            "--expanded-controlled-unique"
        )
    if args.legacy_cumulative and any(bool(value) for value in requested_tiers):
        raise ValueError(
            "--legacy-cumulative cannot be combined with a feature tier"
        )
    # The finalized behavioral story uses actor-policy change and three
    # realized geometry variables.  Preserve the former cumulative path only
    # behind an explicit compatibility flag.
    if not args.legacy_cumulative and not any(
        bool(value) for value in requested_tiers
    ):
        args.actor_planning_only = True
    actor_planning_only = bool(args.actor_planning_only)
    expanded_feature_only = bool(args.expanded_feature_only)
    expanded_controlled_unique = bool(args.expanded_controlled_unique)
    if expanded_feature_only or expanded_controlled_unique:
        raise ValueError(
            "The expanded 05_value_plan_analysis encoding tiers were retired. "
            "Use --actor-planning-only, --feature-only, or --controlled-unique."
        )
    # Reuse the mature feature-only/controlled code paths while retaining the
    # requested expanded tier for namespace and model selection.
    args.feature_only = bool(
        args.feature_only
        or expanded_feature_only
        or actor_planning_only
    )
    args.controlled_unique = bool(
        args.controlled_unique or expanded_controlled_unique
    )
    if actor_planning_only and not args.roi_include:
        # Thresholded whole-brain atlas-coverage maps require every ROI and
        # every available voxel.  This does not alter the BOLD data; it only
        # selects the complete atlas cache.
        args.roi_set = "all"
        args.max_roi_voxels = 0
    if args.cross_voxel_permutations > 0 and (
        args.roi_set != "all"
        or args.roi_include
        or args.max_roi_voxels != 0
    ):
        raise ValueError(
            "Cross-style voxel inference requires --roi-set all, "
            "--max-roi-voxels 0, and no --roi-include restriction"
        )

    run_specs, _, cache_audit = COMMON.shared_prepare_from_args(args)
    output_dir = args.output_root / f"sub{args.subject:03d}_game{args.game}"
    if actor_planning_only:
        analysis_dir = output_dir / "actor_planning_feature_only"
    elif expanded_feature_only:
        analysis_dir = output_dir / "feature_only_expanded"
    elif expanded_controlled_unique:
        analysis_dir = output_dir / "controlled_unique_expanded"
    elif args.feature_only:
        analysis_dir = output_dir / "feature_only"
    elif args.controlled_unique:
        analysis_dir = output_dir / "controlled_unique"
    else:
        analysis_dir = output_dir
    result_dir = analysis_dir / "results"
    figure_dir = analysis_dir / "figures"
    design_signature = {
        "metrics_source_token": cache_audit.get("metrics_source_token"),
        "runs": [run.run_label for run in run_specs],
        "alphas_log": list(args.alphas_log),
        "n_alphas": int(args.n_alphas),
        "ram_pca_dim": int(args.ram_pca_dim),
        "visual_pca_dim": int(args.visual_pca_dim),
        "random_state": int(args.random_state),
        "actor_planning_only": actor_planning_only,
        "feature_only": bool(args.feature_only),
        "controlled_unique": bool(args.controlled_unique),
    }
    design_token = hashlib.sha256(
        json.dumps(design_signature, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    design_dir = analysis_dir / "nested_design_cache" / f"src_{design_token}"
    voxel_dir = analysis_dir / "voxel_stats"
    map_dir = analysis_dir / "unthresholded_roi_maps"
    voxel_inference_dir = analysis_dir / "voxelwise_inference"
    for path in (
        result_dir,
        figure_dir,
        design_dir,
        voxel_dir,
        map_dir,
        voxel_inference_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
    if args.cache_only:
        log(f"Cache-only complete: {args.cache_root}")
        return

    if len(run_specs) < 3:
        raise RuntimeError("Nested LORO encoding requires at least three runs")
    roi_names = COMMON.common_roi_names(run_specs)
    run_blocks = load_run_feature_blocks(run_specs)
    selection_y = build_alpha_selection_targets(
        run_specs, roi_names, args.alpha_selection_voxels
    )
    alphas = np.logspace(args.alphas_log[0], args.alphas_log[1], args.n_alphas)
    if args.feature_only:
        if actor_planning_only:
            definitions = actor_planning_feature_only_process_names()
        elif expanded_feature_only:
            definitions = expanded_feature_only_process_names()
        else:
            definitions = feature_only_process_names()
        model_labels = {
            name: FEATURE_ONLY_MODEL_LABELS[name]
            for name in definitions
        }
        contrasts: Mapping[str, Tuple[str, str]] = {}
        analysis_label = (
            "Actor-policy change and search geometry (separate pure models)"
            if actor_planning_only
            else (
                "05 value additions to individual feature-only encoding (no controls)"
                if expanded_feature_only
                else "Individual feature-only models (no controls)"
            )
        )
        analysis_tier = (
            "actor_planning_feature_only"
            if actor_planning_only
            else (
                "feature_only_expanded"
                if expanded_feature_only
                else "feature_only"
            )
        )
    elif args.controlled_unique:
        definitions = (
            expanded_controlled_unique_process_names()
            if expanded_controlled_unique
            else controlled_unique_process_names()
        )
        model_labels = {
            name: CONTROLLED_UNIQUE_MODEL_LABELS[name]
            for name in definitions
        }
        contrasts = (
            VALUE_CONTROLLED_UNIQUE_CONTRASTS
            if expanded_controlled_unique
            else CONTROLLED_UNIQUE_CONTRASTS
        )
        analysis_label = (
            "05 value additions to controlled feature-unique encoding"
            if expanded_controlled_unique
            else "Controlled feature-unique contrasts"
        )
        analysis_tier = (
            "controlled_unique_expanded"
            if expanded_controlled_unique
            else "controlled_unique"
        )
    else:
        definitions = model_process_names(args.include_secondary_geometry)
        model_labels = MODEL_LABELS
        contrasts = CONTRASTS
        analysis_label = "Cumulative controlled models"
        analysis_tier = "controlled"

    alpha_frames: List[pd.DataFrame] = []
    for model_name, process_names in definitions.items():
        log(f"[nested design] {model_name}")
        alpha_frames.append(
            prepare_outer_designs(
                model_name=model_name,
                process_names=process_names,
                run_specs=run_specs,
                run_blocks=run_blocks,
                selection_y=selection_y,
                alphas=alphas,
                ram_pca_dim=args.ram_pca_dim,
                visual_pca_dim=args.visual_pca_dim,
                random_state=args.random_state,
                design_dir=design_dir,
                include_ram=not args.feature_only,
                include_visual=not args.feature_only,
                reuse_existing=not args.rebuild_design_cache,
            )
        )
    alpha_manifest = pd.concat(alpha_frames, ignore_index=True)
    alpha_manifest.to_csv(result_dir / "encoding_nested_alpha_manifest.csv", index=False)

    model_rows: List[Dict[str, object]] = []
    fold_rows: List[Dict[str, object]] = []
    contrast_rows: List[Dict[str, object]] = []
    run_contrast_rows: List[Dict[str, object]] = []
    null_store: Dict[str, np.ndarray] = {}
    voxel_map_stats: Dict[str, Dict[str, np.ndarray]] = {}

    for roi_index, roi_name in enumerate(roi_names, start=1):
        log(f"[ROI {roi_index}/{len(roi_names)}] {roi_name}")
        y_runs: List[np.ndarray] = []
        reference_coords: Optional[np.ndarray] = None
        for run in run_specs:
            patterns, coords = COMMON.load_roi(run, roi_name)
            if reference_coords is None:
                reference_coords = coords
            elif not np.array_equal(reference_coords, coords):
                raise ValueError(f"Voxel coordinates differ across runs for {roi_name}")
            y_runs.append(patterns)

        predictions_by_model: Dict[str, List[np.ndarray]] = {}
        r_by_model: Dict[str, np.ndarray] = {}
        r2_by_model: Dict[str, np.ndarray] = {}
        for model_name in definitions:
            predictions, folds = predict_roi_model(
                model_name=model_name,
                run_specs=run_specs,
                y_runs=y_runs,
                design_dir=design_dir,
            )
            for row in folds:
                row["roi"] = roi_name
                row["n_voxels"] = int(y_runs[0].shape[1])
            fold_rows.extend(folds)
            observed_r, observed_r2, _ = concatenate_run_scores(y_runs, predictions)
            predictions_by_model[model_name] = predictions
            r_by_model[model_name] = observed_r
            r2_by_model[model_name] = observed_r2
            model_rows.append(
                {
                    "roi": roi_name,
                    "model": model_name,
                    "n_runs": len(run_specs),
                    "n_voxels": int(len(observed_r)),
                    "mean_r": float(np.nanmean(observed_r)),
                    "median_r": float(np.nanmedian(observed_r)),
                    "mean_cv_r2": float(np.nanmean(observed_r2)),
                    "median_cv_r2": float(np.nanmedian(observed_r2)),
                    "positive_r_fraction": float(np.nanmean(observed_r > 0)),
                    "analysis_tier": (
                        analysis_tier
                    ),
                }
            )

        roi_voxel_stats: Dict[str, np.ndarray] = {}
        for model_name in definitions:
            roi_voxel_stats[f"encoding_r_{model_name}"] = r_by_model[model_name].astype(
                np.float32
            )
            roi_voxel_stats[f"encoding_cv_r2_{model_name}"] = r2_by_model[
                model_name
            ].astype(np.float32)

        for contrast, (reduced_name, full_name) in contrasts.items():
            voxel_delta = r2_by_model[full_name] - r2_by_model[reduced_name]
            roi_voxel_stats[f"delta_cv_r2_{contrast}"] = voxel_delta.astype(np.float32)
            run_means: List[float] = []
            for run_index, run in enumerate(run_specs):
                reduced_run_r2 = cv_r2_columns(
                    y_runs[run_index], predictions_by_model[reduced_name][run_index]
                )
                full_run_r2 = cv_r2_columns(
                    y_runs[run_index], predictions_by_model[full_name][run_index]
                )
                run_mean = float(np.nanmean(full_run_r2 - reduced_run_r2))
                run_means.append(run_mean)
                run_contrast_rows.append(
                    {
                        "roi": roi_name,
                        "contrast": contrast,
                        "run_label": run.run_label,
                        "delta_cv_r2": run_mean,
                    }
                )
            null = block_permutation_roi_delta(
                y_runs=y_runs,
                reduced_predictions=predictions_by_model[reduced_name],
                full_predictions=predictions_by_model[full_name],
                n_permutations=args.n_permutations,
                block_size=args.permutation_block_size,
                max_voxels=args.permutation_voxels,
                # Identical block orders across ROI × contrast permit a valid
                # family-wise max-statistic correction.
                random_state=args.random_state,
            )
            observed_mean = float(np.nanmean(voxel_delta))
            permutation_take = min(args.permutation_voxels, len(voxel_delta))
            permutation_indices = np.linspace(
                0, len(voxel_delta) - 1, permutation_take, dtype=int
            )
            observed_permutation_subset = float(
                np.nanmean(voxel_delta[permutation_indices])
            )
            p_block = (
                float(
                    (1 + np.sum(null >= observed_permutation_subset))
                    / (1 + len(null))
                )
                if len(null)
                else np.nan
            )
            contrast_rows.append(
                {
                    "roi": roi_name,
                    "contrast": contrast,
                    "reduced_model": reduced_name,
                    "full_model": full_name,
                    "n_runs": len(run_specs),
                    "n_voxels": int(len(voxel_delta)),
                    "mean_delta_cv_r2": observed_mean,
                    "median_delta_cv_r2": float(np.nanmedian(voxel_delta)),
                    "positive_voxel_fraction": float(np.nanmean(voxel_delta > 0)),
                    "positive_run_fraction": float(np.mean(np.asarray(run_means) > 0)),
                    "p_run_signflip_one_sided": exact_sign_flip_p(run_means),
                    "permutation_subset_mean_delta_cv_r2": (
                        observed_permutation_subset
                    ),
                    "p_block_permutation_one_sided": p_block,
                    "permutation_voxels": permutation_take,
                }
            )
            if len(null):
                null_store[
                    f"{COMMON.safe_name(roi_name)}__{contrast}"
                ] = null.astype(np.float32)

        assert reference_coords is not None
        COMMON.atomic_save_npz(
            voxel_dir / f"{COMMON.safe_name(roi_name)}.npz",
            coords=reference_coords.astype(np.int32),
            **roi_voxel_stats,
        )
        voxel_map_stats[roi_name] = roi_voxel_stats
        del predictions_by_model, r_by_model, r2_by_model, y_runs

    model_df = pd.DataFrame(model_rows)
    fold_df = pd.DataFrame(fold_rows)
    contrast_df = add_global_block_max_fwer(
        pd.DataFrame(contrast_rows), null_store
    )
    run_contrast_df = pd.DataFrame(run_contrast_rows)
    if not contrast_df.empty:
        contrast_df["q_global_fdr_run_signflip"] = COMMON.fdr_bh(
            contrast_df["p_run_signflip_one_sided"]
        )
        contrast_df["q_global_fdr_block_permutation"] = COMMON.fdr_bh(
            contrast_df["p_block_permutation_one_sided"]
        )
    model_df.to_csv(result_dir / "encoding_roi_model_summary.csv", index=False)
    fold_df.to_csv(result_dir / "encoding_loro_fold_manifest.csv", index=False)
    contrast_df.to_csv(
        result_dir / "encoding_incremental_roi_summary.csv", index=False
    )
    run_contrast_df.to_csv(
        result_dir / "encoding_incremental_run_effects.csv", index=False
    )
    if null_store:
        COMMON.atomic_save_npz(
            result_dir / "encoding_incremental_block_nulls.npz", **null_store
        )

    build_unthresholded_maps(
        run_specs=run_specs,
        voxel_stats=voxel_map_stats,
        output_dir=map_dir,
    )
    if args.cross_voxel_permutations > 0 and not args.feature_only:
        run_cross_style_voxel_inference(
            run_specs=run_specs,
            voxel_stats=voxel_map_stats,
            design_dir=design_dir,
            output_dir=voxel_inference_dir,
            n_permutations=args.cross_voxel_permutations,
            block_size=args.permutation_block_size,
            top_fraction=args.cross_top_fraction,
            selected_voxels=args.cross_selected_voxels,
            fdr_threshold=args.voxel_fdr_threshold,
            random_state=args.random_state,
            contrast_definitions=contrasts,
            contrast_labels=(
                CONTROLLED_UNIQUE_CONTRAST_LABELS
                if args.controlled_unique
                else {
                    name: name.replace("_", " ")
                    for name in contrasts
                }
            ),
            figure_dir=figure_dir,
            brain_template=args.brain_template,
            display_smoothing_fwhm=args.display_smoothing_fwhm,
        )
    if args.cross_voxel_permutations > 0 and not args.controlled_unique:
        run_cross_style_encoding_r_inference(
            run_specs=run_specs,
            voxel_stats=voxel_map_stats,
            design_dir=design_dir,
            output_dir=voxel_inference_dir,
            figure_dir=figure_dir,
            model_labels=model_labels,
            analysis_label=analysis_label,
            n_permutations=args.cross_voxel_permutations,
            block_size=args.permutation_block_size,
            top_fraction=args.cross_top_fraction,
            selected_voxels=args.cross_selected_voxels,
            fdr_threshold=args.voxel_fdr_threshold,
            random_state=args.random_state,
            brain_template=args.brain_template,
            display_smoothing_fwhm=args.display_smoothing_fwhm,
            threshold_maps_only=actor_planning_only,
        )
    if not actor_planning_only:
        plot_roi_delta_heatmap(
            contrast_df,
            figure_dir / "encoding_incremental_delta_cv_r2_heatmap.png",
            contrast_definitions=contrasts,
            contrast_labels=(
                CONTROLLED_UNIQUE_CONTRAST_LABELS
                if args.controlled_unique
                else None
            ),
        )

    manifest = {
        "analysis": (
            "individual feature-only LORO planning-process encoding"
            if args.feature_only
            else (
                "controlled feature-unique LORO planning-process encoding"
                if args.controlled_unique
                else "nested LORO planning-process encoding"
            )
        ),
        "analysis_tier": analysis_tier,
        "actor_planning_only": actor_planning_only,
        "subject": args.subject,
        "game": args.game,
        "n_runs": len(run_specs),
        "n_rois": len(roi_names),
        "models": model_labels,
        "contrasts": contrasts,
        "feature_only_no_controls": bool(args.feature_only),
        "controlled_unique_contrasts": bool(args.controlled_unique),
        "expanded_feature_only": expanded_feature_only,
        "expanded_controlled_unique": expanded_controlled_unique,
        "nested_design_cache": {
            "path": str(design_dir),
            "source_token": design_token,
            "signature": design_signature,
        },
        "non_destructive_output_namespace": analysis_dir.name,
        "legacy_companion_namespace": (
            "feature_only"
            if expanded_feature_only
            else (
                "controlled_unique"
                if expanded_controlled_unique
                else None
            )
        ),
        "combined_interpretation": (
            "Read this value-addition namespace together with its legacy "
            "companion; unchanged models were not recomputed because v2/v3 "
            "common process, RAM, and visual blocks are bitwise identical."
            if (expanded_feature_only or expanded_controlled_unique)
            else None
        ),
        "ram_pca_dim": args.ram_pca_dim,
        "ram_pca_scope": (
            "not included"
            if args.feature_only
            else "fit inside every inner/outer training fold"
        ),
        "visual_pca_dim": args.visual_pca_dim,
        "visual_pca_scope": (
            "not included"
            if args.feature_only
            else "fit inside every inner/outer training fold"
        ),
        "q_state_control": (
            []
            if args.feature_only
            else ["normalized Q Slack", "root Q Demand", "Q validity"]
        ),
        "temporal_alignment": (
            "behavioral timestamps; exact 1-second averaging; canonical HRF"
        ),
        "alpha_selection": (
            "one alpha per model/outer fold using a fixed voxel sample from "
            "inner training/validation runs only"
        ),
        "n_permutations": args.n_permutations,
        "block_size_tr": args.permutation_block_size,
        "permutation_voxels_per_roi": args.permutation_voxels,
        "permutation_null": (
            "not applicable to the feature-only tier"
            if args.feature_only
            else "prediction-space reduced-model residual block permutation"
        ),
        "within_subject_family_control": (
            "not applicable to the feature-only tier"
            if args.feature_only
            else "synchronized max-statistic FWER across ROI x contrast"
        ),
        "cross_style_voxel_inference": {
            "enabled": bool(args.cross_voxel_permutations > 0),
            "permutations_per_selected_voxel": int(
                args.cross_voxel_permutations
            ),
            "selected_voxels": int(args.cross_selected_voxels),
            "selection_pool": (
                f"top {100 * args.cross_top_fraction:g}% full-model encoding r"
            ),
            "voxel_fdr_threshold": float(args.voxel_fdr_threshold),
            "coverage": "unique Harvard-Oxford atlas voxels",
            "outputs": {
                "primary_incremental": (
                "not applicable to the feature-only tier"
                if args.feature_only
                    else "conditional pooled-null inference on delta CV R2"
                ),
                "cross_figure3_analogue": (
                    "not generated in the controlled-unique tier"
                    if args.controlled_unique
                    else (
                        "model-wise held-out encoding r with independent "
                        "within-run BOLD block permutations"
                    )
                ),
                "figure_policy": (
                    "thresholded brain PNG only; numerical nulls, q-values, "
                    "CSV summaries, and NIfTI maps retained"
                    if actor_planning_only
                    else "legacy full figure set"
                ),
                "roi_fraction_denominator": (
                    (
                        "finite-q tested voxels for each unique delta contrast"
                        if args.controlled_unique
                        else "finite-q tested voxels in each Harvard-Oxford ROI"
                    )
                ),
                "atlas_overlap_policy": (
                    "test each coordinate once globally; reuse its q-value "
                    "within every containing ROI"
                ),
                "brain_figure_smoothing": (
                    f"{args.display_smoothing_fwhm:g} mm FWHM after "
                    "thresholding, for display only"
                ),
            },
        },
        "roi_maps": (
            "unthresholded atlas-ROI coverage only; overlapping ROI values are averaged"
        ),
        "population_inference": False,
        "cache_manifest": cache_audit,
    }
    COMMON.atomic_write_json(result_dir / "analysis_manifest.json", manifest)
    summary_lines = [
        (
            f"# sub{args.subject:03d} game{args.game} "
            + (
                "feature-only planning-process encoding"
                if args.feature_only
                else (
                    "controlled feature-unique planning-process encoding"
                    if args.controlled_unique
                    else "planning-process encoding"
                )
            )
        ),
        "",
        f"- Runs: {len(run_specs)}",
        f"- ROIs: {len(roi_names)} ({args.roi_set})",
        "- Validation: outer LORO; ridge alpha chosen by inner LORO.",
        (
                "- Models: one HRF-aligned planning variable per model; no RAM, "
                "visual, action, Q-state, or nuisance controls."
            if args.feature_only
            else "- RAM/visual PCA and scaling: fitted within each training fold."
        ),
        (
            "- Interpretation: Cross-style descriptive localization, not "
            "feature-unique variance."
            if args.feature_only
            else (
                "- Full/reduced pairs isolate each feature family conditional "
                "on RAM, visual, action, Q-state, nuisance, and related planning terms."
                if args.controlled_unique
                else "- Controls include pre-search Q state and low-level RGB/motion features."
            )
        ),
        (
            "- Primary statistic: held-out voxelwise Pearson r."
            if args.feature_only
            else "- Primary statistic: held-out ΔCV R² for each full/reduced block comparison."
        ),
        *(
            []
            if args.feature_only
            else [
                (
                    f"- Temporal null: {args.n_permutations:,} synchronized "
                    f"{args.permutation_block_size}-TR block permutations per "
                    f"ROI/contrast using up to {args.permutation_voxels} voxels."
                ),
                (
                    "- Conditional null: reduced-model held-out prediction is "
                    "preserved; its residual is block-shuffled."
                ),
                "- Family control: synchronized max-statistic FWER across ROI × contrast.",
            ]
        ),
        (
            "- NIfTI maps are unthresholded atlas-ROI coverage maps, not "
            "whole-brain voxelwise inference."
        ),
        *(
            [
                "- Primary features: raw terminal-minus-root actor-policy "
                "Slack change, actor-policy Demand change, root-branch "
                "diversity, mean bout-terminal depth, and deep-node allocation.",
                "- Figure output: thresholded brain PNGs only; no PDF, ROI-"
                "fraction figure, or null-distribution figure.",
            ]
            if actor_planning_only
            else []
        ),
        (
            f"- Cross-style voxel stage: {'enabled' if args.cross_voxel_permutations > 0 else 'disabled'}; "
            + (
                "when enabled, outputs feature-wise encoding-r inference."
                if args.feature_only
                else (
                    "when enabled, outputs conditional feature-unique ΔCV R² inference."
                    if args.controlled_unique
                    else "when enabled, outputs both conditional ΔCV R² inference and Cross-Figure-3-style model-wise encoding-r inference."
                )
            )
        ),
        (
            (
                "- Unique ΔCV R² ROI fractions use positive significant/tested "
                "voxels after "
                if args.controlled_unique
                else "- Encoding-r ROI fractions use significant/tested voxels after "
            )
            + f"voxelwise BH-FDR q < {args.voxel_fdr_threshold:g}; atlas-overlap "
            + "coordinates are tested once globally."
        ),
        "- Population claim requires subject-level replication and group inference.",
        *(
            [
                "- Expanded value models add outcome mean shift, outcome "
                "dispersion, SR reward trail, and VP bootstrap revision.",
                "- Outcome mean/dispersion and SR/VP are fit as alternative "
                "families to avoid deterministic redundancy.",
            ]
            if (expanded_feature_only or expanded_controlled_unique)
            else []
        ),
        "",
    ]
    (analysis_dir / "summary.md").write_text(
        "\n".join(summary_lines), encoding="utf-8"
    )
    log(f"Complete -> {analysis_dir}")


if __name__ == "__main__":
    main()
