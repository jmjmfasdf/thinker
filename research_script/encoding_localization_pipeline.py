#!/usr/bin/env python3
"""Subject and six-subject orchestration for controlled neural localization.

This module is intentionally separate from the older predictive-encoding
engine.  Its primary estimand is a run-wise OLS beta map for a prespecified
computational target after sensory, motor, history, and staged computational
controls.  Runs are averaged within subject; subjects, not subject-game cells,
are the population-inference unit.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import encoding_localization_core as CORE
import planning_brain_common as COMMON


EPS = 1e-12
INK = "#20252B"
LOCALIZATION_SCHEMA_VERSION = 2
EXPECTED_FULL_RUNS = 11
CANONICAL_CONFIG_ID = "canonical_full_v1"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes() -> Dict[str, str]:
    pipeline_path = Path(__file__).resolve()
    core_path = Path(CORE.__file__).resolve()
    entrypoint_path = pipeline_path.with_name("08_encoding.py")
    return {
        "entrypoint_sha256": _file_sha256(entrypoint_path),
        "pipeline_sha256": _file_sha256(pipeline_path),
        "core_sha256": _file_sha256(core_path),
    }


def _safe_model(model: str) -> str:
    return COMMON.safe_name(model.replace("::", "__"))


def _localization_configuration(args) -> Dict[str, object]:
    roi_include = sorted(
        part.strip()
        for part in str(args.roi_include or "").split(",")
        if part.strip()
    )
    return {
        "localization_tier": str(args.localization_tier),
        "state_control": str(args.state_control),
        "visual_pca_dim": int(args.visual_pca_dim),
        "ram_pca_dim": int(args.ram_pca_dim),
        "sessions": str(args.sessions) if args.sessions is not None else None,
        "max_runs": int(args.max_runs) if args.max_runs is not None else None,
        "roi_set": str(args.roi_set),
        "roi_include": roi_include,
        "max_roi_voxels": int(args.max_roi_voxels),
        "required_status2": int(args.required_status2),
        "allow_smoothed_fallback": bool(args.allow_smoothed_fallback),
        "feature_root": str(Path(args.feature_root).resolve()),
    }


def _configuration_id(configuration: Mapping[str, object]) -> str:
    canonical = (
        configuration["localization_tier"] == "both"
        and configuration["state_control"] == "visual"
        and int(configuration["visual_pca_dim"]) == 20
        and int(configuration["ram_pca_dim"]) == 20
        and configuration["sessions"] is None
        and configuration["max_runs"] is None
        and configuration["roi_set"] == "all"
        and not configuration["roi_include"]
        and int(configuration["max_roi_voxels"]) == 0
        and int(configuration["required_status2"]) == 29
        and not bool(configuration["allow_smoothed_fallback"])
    )
    if canonical:
        return CANONICAL_CONFIG_ID
    payload = json.dumps(
        dict(configuration), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "sensitivity_" + hashlib.sha256(payload).hexdigest()[:16]


def _family(model: str) -> str:
    if model == "partial::switch_status_interaction":
        return "primary_transition_interaction"
    if model.startswith("partial::") and "transition_direction" not in model:
        return "primary_computational_partial"
    if model.startswith("marginal::"):
        return "supplementary_marginal"
    return "exploratory_transition_direction"


def _load_ram(run: COMMON.BrainRunSpec) -> np.ndarray:
    values = np.asarray(np.load(run.ram_path, allow_pickle=False), dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or len(values) != run.n_tr:
        raise ValueError(f"RAM cache shape mismatch for {run.run_label}: {values.shape}")
    return values


def _exact_run_sign_flip(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0 or len(array) > 16:
        return np.nan
    observed = abs(float(np.mean(array)))
    signs = np.asarray(
        list(itertools.product((-1.0, 1.0), repeat=len(array))), dtype=float
    )
    null = np.abs(np.mean(signs * array[None, :], axis=1))
    return float(np.mean(null >= observed - 1e-15))


def _collapse_coordinate_entries(
    values: np.ndarray,
    inverse_coordinates: np.ndarray,
    n_unique: int,
    *,
    agreement_tolerance: float = 1e-6,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Collapse ROI-overlap entries in O(subjects x atlas entries)."""
    array = np.asarray(values, dtype=float)
    was_vector = array.ndim == 1
    if was_vector:
        array = array[None, :]
    inverse = np.asarray(inverse_coordinates, dtype=np.int64).reshape(-1)
    if (
        array.ndim != 2
        or array.shape[1] != len(inverse)
        or len(inverse) == 0
        or np.any(inverse < 0)
        or np.any(inverse >= int(n_unique))
        or not np.all(np.isfinite(array))
    ):
        raise ValueError("Invalid coordinate-collapse inputs")
    counts = np.bincount(inverse, minlength=int(n_unique)).astype(float)
    if np.any(counts <= 0):
        raise ValueError("Unique-coordinate table contains an empty coordinate")
    collapsed = np.zeros((array.shape[0], int(n_unique)), dtype=float)
    minimum = np.full_like(collapsed, np.inf)
    maximum = np.full_like(collapsed, -np.inf)
    for row in range(array.shape[0]):
        np.add.at(collapsed[row], inverse, array[row])
        np.minimum.at(minimum[row], inverse, array[row])
        np.maximum.at(maximum[row], inverse, array[row])
    collapsed /= counts[None, :]
    overlap = counts > 1
    disagreement = (
        float(np.max((maximum - minimum)[:, overlap]))
        if np.any(overlap)
        else 0.0
    )
    if disagreement > float(agreement_tolerance):
        raise ValueError(
            "Overlapping atlas ROIs disagree at an identical voxel: "
            f"max range={disagreement:.6g}"
        )
    result = collapsed[0] if was_vector else collapsed
    return result, {
        "roi_entry_count": int(len(inverse)),
        "unique_coordinate_count": int(n_unique),
        "overlap_entry_count": int(len(inverse) - int(n_unique)),
        "coordinates_in_multiple_rois": int(np.sum(overlap)),
        "maximum_overlap_value_range": disagreement,
        "agreement_tolerance": float(agreement_tolerance),
    }


def _write_maps(
    reference_path: Path,
    roi_coordinates: Mapping[str, np.ndarray],
    voxel_stats: Mapping[str, Mapping[str, np.ndarray]],
    output_dir: Path,
) -> None:
    import nibabel as nib

    reference = nib.load(str(reference_path))
    shape = tuple(int(value) for value in reference.shape[:3])
    output_dir.mkdir(parents=True, exist_ok=True)
    statistics = sorted(
        {
            statistic
            for values in voxel_stats.values()
            for statistic in values
        }
    )
    for statistic in statistics:
        coordinate_parts: List[np.ndarray] = []
        value_parts: List[np.ndarray] = []
        for roi, values_by_statistic in voxel_stats.items():
            if statistic not in values_by_statistic:
                continue
            coords = np.asarray(roi_coordinates[roi], dtype=np.int32)
            values = np.asarray(values_by_statistic[statistic], dtype=float)
            if (
                coords.ndim != 2
                or coords.shape[1] != 3
                or len(values) != len(coords)
                or np.any(coords < 0)
                or np.any(coords >= np.asarray(shape, dtype=int)[None, :])
            ):
                raise ValueError(f"Map length mismatch: {roi}:{statistic}")
            coordinate_parts.append(coords)
            value_parts.append(values)
        coordinates = np.concatenate(coordinate_parts, axis=0)
        unique_coordinates, inverse = np.unique(
            coordinates, axis=0, return_inverse=True
        )
        collapsed, _ = _collapse_coordinate_entries(
            np.concatenate(value_parts), inverse, len(unique_coordinates)
        )
        # A p-value of zero outside the tested atlas would look maximally
        # significant to downstream viewers.  Use one outside coverage for
        # p maps; signed/statistic maps retain a neutral zero background.
        background = 1.0 if statistic.startswith("p_") else 0.0
        data = np.full(shape, background, dtype=np.float32)
        x, y, z = (
            unique_coordinates[:, 0],
            unique_coordinates[:, 1],
            unique_coordinates[:, 2],
        )
        data[x, y, z] = np.asarray(collapsed, dtype=np.float32)
        nib.save(
            nib.Nifti1Image(data, reference.affine, reference.header),
            str(output_dir / f"{COMMON.safe_name(statistic)}.nii.gz"),
        )


def _plot_heatmap(
    frame: pd.DataFrame,
    *,
    value: str,
    output_path: Path,
    title: str,
    significance: Optional[str] = None,
) -> None:
    if frame.empty:
        return
    models = list(dict.fromkeys(frame["model"].astype(str)))
    pivot = frame.pivot(index="roi", columns="model", values=value).reindex(
        columns=models
    )
    values = pivot.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
    if vmax < EPS:
        vmax = 1.0
    height = max(6.5, min(30.0, 2.4 + 0.22 * len(pivot)))
    width = max(11.0, 2.7 + 0.92 * len(models))
    fig, axis = plt.subplots(figsize=(width, height), facecolor="white")
    image = axis.imshow(
        np.ma.masked_invalid(values), aspect="auto", cmap="RdBu_r",
        vmin=-vmax, vmax=vmax,
    )
    axis.set_yticks(np.arange(len(pivot)))
    axis.set_yticklabels(
        [
            str(name).replace("ho_cortical_", "").replace("ho_subcortical_", "")
            for name in pivot.index
        ],
        fontsize=7,
    )
    axis.set_xticks(np.arange(len(models)))
    axis.set_xticklabels(
        [name.replace("::", "\n") for name in models],
        rotation=42, ha="right", fontsize=8,
    )
    if significance and significance in frame.columns:
        sig = frame.pivot(index="roi", columns="model", values=significance).reindex(
            index=pivot.index, columns=pivot.columns
        )
        for row, column in zip(*np.where(sig.to_numpy(dtype=float) < .05)):
            axis.text(column, row, "•", ha="center", va="center", color="black", fontsize=9)
    axis.set_title(title)
    colorbar = fig.colorbar(image, ax=axis, fraction=.025, pad=.02)
    colorbar.set_label("Raw standardized beta")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _subject_paths(
    output_root: Path,
    subject: int,
    game: int,
    configuration_id: str = CANONICAL_CONFIG_ID,
) -> Dict[str, Path]:
    base = Path(output_root) / f"sub{subject:03d}_game{game}" / "localization"
    if configuration_id != CANONICAL_CONFIG_ID:
        base = base / "configurations" / str(configuration_id)
    return {
        "base": base,
        "results": base / "results",
        "figures": base / "figures",
        "voxels": base / "voxel_stats",
        "maps": base / "unthresholded_maps",
    }


def _prepare_subject_namespace(
    paths: Mapping[str, Path],
    requested_contract: Mapping[str, object],
    *,
    overwrite: bool,
) -> None:
    base = Path(paths["base"])
    manifest_path = Path(paths["results"]) / "analysis_manifest.json"
    if not base.exists() or not any(base.iterdir()):
        return
    matches = False
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        observed = {
            "schema_version": existing.get("schema_version"),
            "configuration_id": existing.get("configuration_id"),
            "configuration": existing.get("configuration"),
            "feature_source_token": existing.get("cache_manifest", {}).get(
                "feature_source_token"
            ),
            "source": existing.get("source"),
        }
        matches = observed == dict(requested_contract)
    if matches:
        return
    if not overwrite:
        raise FileExistsError(
            "Localization output namespace already contains an incompatible "
            f"or incomplete run: {base}. Use a new --output-root or explicit "
            "--overwrite-localization."
        )
    shutil.rmtree(base)


def run_subject(args) -> None:
    configuration = _localization_configuration(args)
    configuration_id = _configuration_id(configuration)
    run_specs, _, cache_audit = COMMON.shared_prepare_from_args(args)
    if len(run_specs) < 2:
        raise RuntimeError("Localization requires at least two runs")
    tiers = (
        ("marginal", "partial")
        if args.localization_tier == "both"
        else (args.localization_tier,)
    )
    include_ram = args.state_control == "visual+ram"
    paths = _subject_paths(
        args.output_root, args.subject, args.game, configuration_id
    )
    source_hashes = _source_hashes()
    requested_contract = {
        "schema_version": LOCALIZATION_SCHEMA_VERSION,
        "configuration_id": configuration_id,
        "configuration": configuration,
        "feature_source_token": cache_audit.get("feature_source_token"),
        "source": source_hashes,
    }
    _prepare_subject_namespace(
        paths,
        requested_contract,
        overwrite=bool(getattr(args, "overwrite_localization", False)),
    )
    for key in ("results", "figures", "voxels", "maps"):
        paths[key].mkdir(parents=True, exist_ok=True)

    weights_by_run: Dict[str, Dict[str, np.ndarray]] = {}
    audit_rows: List[Dict[str, object]] = []
    model_order: Optional[List[str]] = None
    for run in run_specs:
        blocks = COMMON.load_process_blocks(run)
        visual = COMMON.load_visual(run)
        ram = _load_ram(run)
        weights, audit = CORE.build_localization_weights(
            blocks, visual, ram,
            visual_pca_dim=args.visual_pca_dim,
            ram_pca_dim=args.ram_pca_dim,
            include_ram=include_ram,
            tiers=tiers,
        )
        if model_order is None:
            model_order = list(weights)
        elif list(weights) != model_order:
            raise ValueError("Localization model order differs across runs")
        weights_by_run[run.run_label] = weights
        for model, row in audit.items():
            audit_rows.append(
                {
                    "run_label": run.run_label,
                    "subject": int(args.subject),
                    "game": int(args.game),
                    "model": model,
                    "family": _family(model),
                    **{
                        key: (
                            json.dumps(value, sort_keys=True)
                            if isinstance(value, (list, tuple, dict))
                            else value
                        )
                        for key, value in row.items()
                    },
                }
            )
    assert model_order is not None

    roi_names = COMMON.common_roi_names(run_specs)
    if not roi_names:
        raise RuntimeError("No ROI is shared by all runs")
    run_rows: List[Dict[str, object]] = []
    subject_rows: List[Dict[str, object]] = []
    voxel_stats: Dict[str, Dict[str, np.ndarray]] = {}
    roi_coordinates: Dict[str, np.ndarray] = {}
    for roi_index, roi in enumerate(roi_names, start=1):
        print(f"[localization ROI {roi_index}/{len(roi_names)}] {roi}", flush=True)
        reference_coords: Optional[np.ndarray] = None
        beta_by_model: Dict[str, List[np.ndarray]] = {model: [] for model in model_order}
        for run in run_specs:
            bold, coords = COMMON.load_roi(run, roi)
            if reference_coords is None:
                reference_coords = coords
            elif not np.array_equal(reference_coords, coords):
                raise ValueError(f"Voxel coordinates differ across runs: {roi}")
            for model in model_order:
                beta = CORE.apply_localization_weight(
                    weights_by_run[run.run_label][model], bold
                )
                beta_by_model[model].append(beta)
                run_rows.append(
                    {
                        "subject": int(args.subject),
                        "game": int(args.game),
                        "run_label": run.run_label,
                        "roi": roi,
                        "model": model,
                        "family": _family(model),
                        "n_voxels": int(len(beta)),
                        "mean_beta": float(np.mean(beta)),
                        "median_beta": float(np.median(beta)),
                        "positive_voxel_fraction": float(np.mean(beta > 0)),
                    }
                )
        assert reference_coords is not None
        roi_coordinates[roi] = reference_coords
        stored: Dict[str, np.ndarray] = {}
        for model in model_order:
            run_matrix = np.stack(beta_by_model[model], axis=0)
            mean_map = np.mean(run_matrix, axis=0).astype(np.float32)
            statistic = f"beta__{_safe_model(model)}"
            stored[statistic] = mean_map
            run_means = np.mean(run_matrix, axis=1)
            subject_rows.append(
                {
                    "subject": int(args.subject),
                    "game": int(args.game),
                    "roi": roi,
                    "model": model,
                    "family": _family(model),
                    "n_runs": int(len(run_specs)),
                    "n_voxels": int(run_matrix.shape[1]),
                    "mean_beta": float(np.mean(run_means)),
                    "median_run_beta": float(np.median(run_means)),
                    "n_positive_runs": int(np.sum(run_means > 0)),
                    "n_negative_runs": int(np.sum(run_means < 0)),
                    "exact_run_sign_flip_p": _exact_run_sign_flip(run_means),
                }
            )
        COMMON.atomic_save_npz(
            paths["voxels"] / f"{COMMON.safe_name(roi)}.npz",
            coords=reference_coords.astype(np.int32),
            **stored,
        )
        voxel_stats[roi] = stored

    run_frame = pd.DataFrame(run_rows)
    subject_frame = pd.DataFrame(subject_rows)
    audit_frame = pd.DataFrame(audit_rows)
    run_frame.to_csv(paths["results"] / "localization_run_roi_effects.csv", index=False)
    subject_frame.to_csv(
        paths["results"] / "localization_subject_roi_effects.csv", index=False
    )
    audit_frame.to_csv(paths["results"] / "localization_design_audit.csv", index=False)
    _plot_heatmap(
        subject_frame, value="mean_beta",
        output_path=paths["figures"] / "figure1_subject_localization_beta_heatmap.png",
        title=(
            f"Controlled neural localization — sub{args.subject:03d}, "
            f"game {args.game}\nrun-equalized raw beta"
        ),
    )
    _write_maps(
        run_specs[0].fmri_path, roi_coordinates, voxel_stats, paths["maps"]
    )

    manifest = {
        "schema_version": LOCALIZATION_SCHEMA_VERSION,
        "analysis": "run-wise controlled OLS neural localization",
        "subject": int(args.subject),
        "game": int(args.game),
        "n_runs": int(len(run_specs)),
        "n_rois": int(len(roi_names)),
        "roi_names": list(roi_names),
        "n_models": int(len(model_order)),
        "models": model_order,
        "families": {model: _family(model) for model in model_order},
        "localization_tier": args.localization_tier,
        "state_control": args.state_control,
        "visual_pca_dim": int(args.visual_pca_dim),
        "ram_pca_dim": int(args.ram_pca_dim),
        "configuration_id": configuration_id,
        "configuration": configuration,
        "canonical_full_configuration": (
            configuration_id == CANONICAL_CONFIG_ID
        ),
        "full_run_contract_complete": int(len(run_specs)) == EXPECTED_FULL_RUNS,
        "estimand": (
            "one standardized target beta map per run; controls include visual "
            "state, action/history/drift, and for partial models prespecified "
            "upstream computational blocks"
        ),
        "continuous_control_contract": (
            "visual state, previous/current action, task history, drift, and "
            "availability-only eligibility columns 0:8; semantic transition-risk "
            "eligibility columns 8:12 are excluded"
        ),
        "transition_control_contract": (
            "visual state, task history, drift, HRF decision-event mass, and "
            "origin/destination main effects; action one-hot and all eligibility/"
            "risk columns are excluded"
        ),
        "aggregation": "equal runs within subject",
        "population_inference": False,
        "maps": "descriptive unthresholded subject-mean beta maps",
        "reference_fmri": str(run_specs[0].fmri_path),
        "source": source_hashes,
        "cache_manifest": cache_audit,
    }
    COMMON.atomic_write_json(paths["results"] / "analysis_manifest.json", manifest)
    print(f"Localization complete -> {paths['base']}", flush=True)


def _group_paths(output_root: Path, game: int) -> Dict[str, Path]:
    base = Path(output_root) / f"group_game{game}" / "localization"
    return {
        "base": base,
        "results": base / "results",
        "figures": base / "figures",
        "voxels": base / "voxel_stats",
        "maps": base / "unthresholded_maps",
    }


def _validate_full_subject_artifact(
    frame: pd.DataFrame,
    manifest: Mapping[str, object],
    *,
    subject: int,
    game: int,
) -> Dict[str, object]:
    if int(manifest.get("schema_version", -1)) != LOCALIZATION_SCHEMA_VERSION:
        raise ValueError(f"Localization schema mismatch for sub{subject:03d} game{game}")
    if (
        int(manifest.get("subject", -1)) != int(subject)
        or int(manifest.get("game", -1)) != int(game)
    ):
        raise ValueError(f"Manifest subject/game mismatch for sub{subject:03d} game{game}")
    if (
        manifest.get("configuration_id") != CANONICAL_CONFIG_ID
        or not bool(manifest.get("canonical_full_configuration"))
        or not bool(manifest.get("full_run_contract_complete"))
        or int(manifest.get("n_runs", -1)) != EXPECTED_FULL_RUNS
    ):
        raise ValueError(
            f"Population inference requires the canonical full {EXPECTED_FULL_RUNS}-run "
            f"artifact: sub{subject:03d} game{game}"
        )
    configuration = manifest.get("configuration")
    if not isinstance(configuration, dict) or _configuration_id(configuration) != CANONICAL_CONFIG_ID:
        raise ValueError(f"Invalid canonical configuration: sub{subject:03d} game{game}")
    if manifest.get("source") != _source_hashes():
        raise ValueError(
            f"Localization source hash is stale for sub{subject:03d} game{game}"
        )
    cache = manifest.get("cache_manifest")
    if not isinstance(cache, dict):
        raise ValueError(f"Cache manifest missing for sub{subject:03d} game{game}")
    feature = cache.get("feature_source")
    if not isinstance(feature, dict):
        raise ValueError(f"04 feature provenance missing for sub{subject:03d} game{game}")
    if (
        int(cache.get("process_feature_version", -1))
        != int(COMMON.PROCESS_FEATURE_VERSION)
        or int(cache.get("neural_feature_schema_version", -1))
        != int(COMMON.NEURAL_FEATURE_SCHEMA_VERSION)
        or int(cache.get("subject", -1)) != int(subject)
        or int(cache.get("game", -1)) != int(game)
        or bool(cache.get("include_topology"))
        or bool(cache.get("im_vectors_used", True))
        or cache.get("roi_set") != "all"
        or int(cache.get("max_roi_voxels", -1)) != 0
        or len(cache.get("runs", [])) != EXPECTED_FULL_RUNS
    ):
        raise ValueError(f"Canonical brain-cache contract failed: sub{subject:03d} game{game}")
    models = [str(value) for value in manifest.get("models", [])]
    families = {
        str(key): str(value)
        for key, value in dict(manifest.get("families", {})).items()
    }
    rois = [str(value) for value in manifest.get("roi_names", [])]
    if (
        not models
        or len(models) != int(manifest.get("n_models", -1))
        or set(families) != set(models)
        or not rois
        or len(rois) != int(manifest.get("n_rois", -1))
        or sorted(rois) != sorted(str(value) for value in cache.get("roi_names", []))
    ):
        raise ValueError(f"Manifest model/ROI contract failed: sub{subject:03d} game{game}")
    required = {
        "subject", "game", "roi", "model", "family", "n_runs",
        "n_voxels", "mean_beta", "exact_run_sign_flip_p",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Subject effect table lacks columns {missing}")
    expected_rows = len(rois) * len(models)
    if (
        len(frame) != expected_rows
        or frame.duplicated(["roi", "model"]).any()
        or set(frame["subject"].astype(int)) != {int(subject)}
        or set(frame["game"].astype(int)) != {int(game)}
        or set(frame["n_runs"].astype(int)) != {EXPECTED_FULL_RUNS}
        or set(frame["roi"].astype(str)) != set(rois)
        or set(frame["model"].astype(str)) != set(models)
    ):
        raise ValueError(f"Subject effect grid is incomplete: sub{subject:03d} game{game}")
    expected_families = frame["model"].astype(str).map(families)
    if not np.array_equal(expected_families.to_numpy(), frame["family"].astype(str).to_numpy()):
        raise ValueError(f"Subject effect family mismatch: sub{subject:03d} game{game}")
    numeric = frame[["n_voxels", "mean_beta", "exact_run_sign_flip_p"]].to_numpy(dtype=float)
    if not np.all(np.isfinite(numeric)) or np.any(numeric[:, 0] <= 0):
        raise ValueError(f"Subject effect table is non-finite: sub{subject:03d} game{game}")
    import nibabel as nib

    reference_path = Path(str(manifest.get("reference_fmri", "")))
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference fMRI is missing: {reference_path}")
    reference_image = nib.load(str(reference_path))
    reference_geometry = {
        "shape": [int(value) for value in reference_image.shape[:3]],
        "affine_rounded_8dp": np.round(reference_image.affine, 8).tolist(),
    }
    return {
        "configuration": configuration,
        "models": models,
        "families": families,
        "roi_names": sorted(rois),
        "source": manifest["source"],
        "feature_root": str(cache.get("feature_root")),
        "process_feature_version": int(cache["process_feature_version"]),
        "neural_feature_schema_version": int(cache["neural_feature_schema_version"]),
        "04_script_sha256": str(feature.get("04_script_sha256")),
        "reference_geometry": reference_geometry,
    }


def _assert_matching_contracts(
    contracts: Mapping[Tuple[int, int], Mapping[str, object]]
) -> Dict[str, object]:
    if not contracts:
        raise ValueError("No localization contracts to compare")
    first_key = next(iter(contracts))
    reference = dict(contracts[first_key])
    for key, contract in contracts.items():
        if dict(contract) != reference:
            raise ValueError(
                f"Localization settings/source/ROI mismatch: {key} != {first_key}"
            )
    return reference


def run_group(args) -> None:
    subjects = sorted({int(value) for value in args.aggregate_subjects})
    if subjects != [1, 2, 3, 4, 5, 6]:
        raise ValueError("Population localization requires subjects 1,2,3,4,5,6")
    if _configuration_id(_localization_configuration(args)) != CANONICAL_CONFIG_ID:
        raise ValueError("Population localization accepts only the canonical full configuration")
    frames: List[pd.DataFrame] = []
    manifests: Dict[int, Dict[str, object]] = {}
    contracts: Dict[Tuple[int, int], Dict[str, object]] = {}
    for subject in subjects:
        paths = _subject_paths(args.output_root, subject, args.game)
        effect_path = paths["results"] / "localization_subject_roi_effects.csv"
        manifest_path = paths["results"] / "analysis_manifest.json"
        if not effect_path.exists() or not manifest_path.exists():
            raise FileNotFoundError(
                f"Missing subject localization output: sub{subject:03d}, game {args.game}"
            )
        frame = pd.read_csv(effect_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        contracts[(subject, int(args.game))] = _validate_full_subject_artifact(
            frame, manifest, subject=subject, game=int(args.game)
        )
        frames.append(frame)
        manifests[subject] = manifest
    shared_contract = _assert_matching_contracts(contracts)
    all_subjects = pd.concat(frames, ignore_index=True)
    duplicate = all_subjects.duplicated(["subject", "roi", "model"])
    if duplicate.any():
        raise ValueError("Duplicate subject/ROI/model localization rows")
    paths = _group_paths(args.output_root, args.game)
    for key in ("results", "figures", "voxels", "maps"):
        paths[key].mkdir(parents=True, exist_ok=True)
    all_subjects.to_csv(
        paths["results"] / "localization_subject_effects.csv", index=False
    )

    inference_rows: List[Dict[str, object]] = []
    for family, source in all_subjects.groupby("family", sort=False):
        keys = sorted(
            {(str(row.roi), str(row.model)) for row in source.itertuples()},
            key=lambda item: (item[1], item[0]),
        )
        matrix = np.empty((6, len(keys)), dtype=float)
        for subject_index, subject in enumerate(subjects):
            lookup = source.loc[source["subject"].eq(subject)].set_index(
                ["roi", "model"]
            )["mean_beta"]
            if len(lookup) != len(keys):
                raise ValueError(f"Incomplete localization family for sub{subject:03d}: {family}")
            matrix[subject_index] = [float(lookup.loc[key]) for key in keys]
        statistic, raw_p, max_t_p = CORE.exact_synchronized_sign_flip(matrix)
        for column, (roi, model) in enumerate(keys):
            values = matrix[:, column]
            inference_rows.append(
                {
                    "game": int(args.game),
                    "family": family,
                    "roi": roi,
                    "model": model,
                    "n_subjects": 6,
                    "mean_subject_beta": float(np.mean(values)),
                    "median_subject_beta": float(np.median(values)),
                    "studentized_t": float(statistic[column]),
                    "exact_sign_flip_p": float(raw_p[column]),
                    "synchronized_max_t_p": float(max_t_p[column]),
                    "n_positive_subjects": int(np.sum(values > 0)),
                    "n_negative_subjects": int(np.sum(values < 0)),
                }
            )
    inference = pd.DataFrame(inference_rows)
    inference.to_csv(
        paths["results"] / "localization_group_inference.csv", index=False
    )
    for family, frame in inference.groupby("family", sort=False):
        _plot_heatmap(
            frame.rename(columns={"mean_subject_beta": "plot_beta"}),
            value="plot_beta",
            significance="synchronized_max_t_p",
            output_path=paths["figures"] / f"figure_{COMMON.safe_name(family)}.png",
            title=(
                f"Neural localization — game {args.game}, six subjects\n"
                f"{family.replace('_', ' ')}; dot = synchronized max-|t| p < .05"
            ),
        )

    # Voxelwise population inference uses the same six subject-level maps and
    # synchronizes every sign pattern across all unique atlas coordinates and
    # all targets in a prespecified family.  Overlapping atlas parcels are
    # collapsed before testing and mapped back afterward.
    first_manifest = manifests[subjects[0]]
    reference_fmri = Path(str(first_manifest["reference_fmri"]))
    reference_shape = np.asarray(
        shared_contract["reference_geometry"]["shape"], dtype=int
    )
    roi_coordinates: Dict[str, np.ndarray] = {}
    group_voxel_stats: Dict[str, Dict[str, np.ndarray]] = {}
    subject_voxel_stats: Dict[str, Dict[str, np.ndarray]] = {}
    rois = sorted(all_subjects["roi"].astype(str).unique())
    models = list(dict.fromkeys(all_subjects["model"].astype(str)))
    for roi in rois:
        subject_archives = []
        reference_coords: Optional[np.ndarray] = None
        for subject in subjects:
            archive_path = (
                _subject_paths(args.output_root, subject, args.game)["voxels"]
                / f"{COMMON.safe_name(roi)}.npz"
            )
            archive = np.load(archive_path, allow_pickle=False)
            coords = np.asarray(archive["coords"], dtype=np.int32)
            if (
                coords.ndim != 2
                or coords.shape[1] != 3
                or np.any(coords < 0)
                or np.any(coords >= reference_shape[None, :])
            ):
                archive.close()
                raise ValueError(f"Invalid voxel coordinates: {archive_path}")
            if reference_coords is None:
                reference_coords = coords
            elif not np.array_equal(reference_coords, coords):
                raise ValueError(f"Group voxel coordinates differ: {roi}")
            subject_archives.append(archive)
        assert reference_coords is not None
        subject_voxel_stats[roi] = {}
        group_voxel_stats[roi] = {}
        for model in models:
            statistic = f"beta__{_safe_model(model)}"
            values = np.stack(
                [np.asarray(archive[statistic], dtype=float) for archive in subject_archives],
                axis=0,
            )
            if values.shape[0] != 6 or not np.all(np.isfinite(values)):
                raise ValueError(f"Invalid six-subject voxel matrix: {roi}:{model}")
            subject_voxel_stats[roi][model] = values.astype(np.float32)
        for archive in subject_archives:
            archive.close()
        roi_coordinates[roi] = reference_coords
    all_coordinates = np.concatenate(
        [roi_coordinates[roi] for roi in rois], axis=0
    ).astype(np.int32)
    unique_coordinates, inverse_coordinates = np.unique(
        all_coordinates, axis=0, return_inverse=True
    )
    roi_slices: Dict[str, slice] = {}
    cursor = 0
    for roi in rois:
        size = len(roi_coordinates[roi])
        roi_slices[roi] = slice(cursor, cursor + size)
        cursor += size
    voxel_rows: List[Dict[str, object]] = []
    family_models: Dict[str, List[str]] = {}
    for model in models:
        family_models.setdefault(_family(model), []).append(model)
    coordinate_collapse_audits: List[Dict[str, object]] = []
    for family, members in family_models.items():
        arrays: List[np.ndarray] = []
        for model in members:
            concatenated = np.concatenate(
                [subject_voxel_stats[roi][model] for roi in rois], axis=1
            )
            unique_values, collapse_audit = _collapse_coordinate_entries(
                concatenated, inverse_coordinates, len(unique_coordinates)
            )
            coordinate_collapse_audits.append(
                {"family": family, "model": model, **collapse_audit}
            )
            arrays.append(unique_values)
        family_matrix = np.concatenate(arrays, axis=1).astype(float)
        statistic, raw_p, max_t_p = CORE.exact_synchronized_sign_flip(family_matrix)
        family_mean = np.mean(family_matrix, axis=0)
        for model_index, model in enumerate(members):
            take = slice(
                model_index * len(unique_coordinates),
                (model_index + 1) * len(unique_coordinates),
            )
            mean_unique = family_mean[take]
            t_unique = statistic[take]
            raw_unique = raw_p[take]
            corrected_unique = max_t_p[take]
            safe = _safe_model(model)
            for roi in rois:
                coordinate_indices = inverse_coordinates[roi_slices[roi]]
                mean_values = mean_unique[coordinate_indices]
                t_values = t_unique[coordinate_indices]
                raw_values = raw_unique[coordinate_indices]
                corrected_values = corrected_unique[coordinate_indices]
                group_voxel_stats[roi][f"beta__{safe}"] = mean_values.astype(np.float32)
                group_voxel_stats[roi][f"t__{safe}"] = t_values.astype(np.float32)
                group_voxel_stats[roi][f"p_exact__{safe}"] = raw_values.astype(np.float32)
                group_voxel_stats[roi][f"p_family_fwer__{safe}"] = corrected_values.astype(np.float32)
                group_voxel_stats[roi][f"beta_fwer05__{safe}"] = np.where(
                    corrected_values < .05, mean_values, 0.0
                ).astype(np.float32)
                significant = corrected_values < .05
                voxel_rows.append(
                    {
                        "game": int(args.game),
                        "family": family,
                        "roi": roi,
                        "model": model,
                        "n_subjects": 6,
                        "n_tested_voxels": int(len(mean_values)),
                        "n_family_fwer_significant_voxels": int(np.sum(significant)),
                        "family_fwer_significant_fraction": float(np.mean(significant)),
                        "positive_fraction_among_significant": (
                            float(np.mean(mean_values[significant] > 0))
                            if np.any(significant) else np.nan
                        ),
                        "mean_voxel_beta": float(np.mean(mean_values)),
                        "minimum_family_fwer_p": float(np.min(corrected_values)),
                    }
                )
    for roi in rois:
        COMMON.atomic_save_npz(
            paths["voxels"] / f"{COMMON.safe_name(roi)}.npz",
            coords=roi_coordinates[roi],
            **group_voxel_stats[roi],
        )
    pd.DataFrame(voxel_rows).to_csv(
        paths["results"] / "localization_voxelwise_family_fwer_by_roi.csv",
        index=False,
    )
    _write_maps(reference_fmri, roi_coordinates, group_voxel_stats, paths["maps"])

    manifest = {
        "schema_version": LOCALIZATION_SCHEMA_VERSION,
        "analysis": "six-subject exact population inference for neural localization",
        "game": int(args.game),
        "subjects": subjects,
        "n_subjects": 6,
        "families": sorted(inference["family"].unique()),
        "inference": (
            "run means equalized within subject; exact two-sided subject sign "
            "flip; synchronized max-|t| separately within prespecified primary "
            "computational-partial, transition-interaction, supplementary "
            "marginal, and exploratory direction families"
        ),
        "voxel_inference": (
            "exact six-subject two-sided sign flips; synchronized max-|t| FWER "
            "across every atlas voxel and target within each prespecified family"
        ),
        "atlas_roi_inference": (
            "exact six-subject sign flips with family max-|t| across ROI x target"
        ),
        "subject_manifests": manifests,
        "validated_shared_contract": shared_contract,
        "coordinate_overlap_validation": {
            "roi_entry_count": int(len(all_coordinates)),
            "unique_coordinate_count": int(len(unique_coordinates)),
            "overlap_entry_count": int(
                len(all_coordinates) - len(unique_coordinates)
            ),
            "maximum_overlap_value_range": float(
                max(
                    row["maximum_overlap_value_range"]
                    for row in coordinate_collapse_audits
                )
            ),
            "all_model_audits": coordinate_collapse_audits,
        },
    }
    COMMON.atomic_write_json(paths["results"] / "analysis_manifest.json", manifest)
    print(f"Group localization complete -> {paths['base']}", flush=True)


def run_paired_games(args) -> None:
    """Test the across-game mean and Game2-Game1 within subject."""
    if _configuration_id(_localization_configuration(args)) != CANONICAL_CONFIG_ID:
        raise ValueError("Paired-game localization accepts only the canonical full configuration")
    subjects = list(range(1, 7))
    frames: List[pd.DataFrame] = []
    inputs: List[str] = []
    contracts: Dict[Tuple[int, int], Dict[str, object]] = {}
    for subject in subjects:
        for game in (1, 2):
            path = (
                _subject_paths(args.output_root, subject, game)["results"]
                / "localization_subject_roi_effects.csv"
            )
            if not path.exists():
                raise FileNotFoundError(f"Missing paired-game localization input: {path}")
            manifest_path = path.parent / "analysis_manifest.json"
            if not manifest_path.exists():
                raise FileNotFoundError(
                    f"Missing paired-game localization manifest: {manifest_path}"
                )
            frame = pd.read_csv(path)
            if set(frame["subject"].astype(int)) != {subject} or set(frame["game"].astype(int)) != {game}:
                raise ValueError(f"Subject/game mismatch in {path}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            contracts[(subject, game)] = _validate_full_subject_artifact(
                frame, manifest, subject=subject, game=game
            )
            frames.append(frame)
            inputs.extend((str(path), str(manifest_path)))
    shared_contract = _assert_matching_contracts(contracts)
    data = pd.concat(frames, ignore_index=True)
    if data.duplicated(["subject", "game", "roi", "model"]).any():
        raise ValueError("Duplicate paired-game localization rows")
    wide = data.pivot(
        index=["subject", "roi", "model", "family"],
        columns="game", values="mean_beta",
    )
    if set(wide.columns) != {1, 2} or wide.isna().any().any():
        raise ValueError("Every paired localization effect must exist in both games")
    paired_frames: List[pd.DataFrame] = []
    for estimand, values in (
        ("across_game_mean", (wide[1] + wide[2]) / 2.0),
        ("game2_minus_game1", wide[2] - wide[1]),
    ):
        frame = values.rename("mean_beta").reset_index()
        frame["estimand"] = estimand
        paired_frames.append(frame)
    paired = pd.concat(paired_frames, ignore_index=True)
    inference_rows: List[Dict[str, object]] = []
    for (estimand, family), source in paired.groupby(["estimand", "family"], sort=False):
        keys = sorted(
            {(str(row.roi), str(row.model)) for row in source.itertuples()},
            key=lambda item: (item[1], item[0]),
        )
        matrix = np.empty((6, len(keys)), dtype=float)
        for index, subject in enumerate(subjects):
            lookup = source.loc[source["subject"].eq(subject)].set_index(
                ["roi", "model"]
            )["mean_beta"]
            if len(lookup) != len(keys):
                raise ValueError(f"Incomplete paired-game localization for sub{subject:03d}")
            matrix[index] = [float(lookup.loc[key]) for key in keys]
        statistic, raw, corrected = CORE.exact_synchronized_sign_flip(matrix)
        for column, (roi, model) in enumerate(keys):
            values = matrix[:, column]
            inference_rows.append(
                {
                    "estimand": estimand,
                    "family": family,
                    "roi": roi,
                    "model": model,
                    "n_subjects": 6,
                    "mean_subject_beta": float(np.mean(values)),
                    "studentized_t": float(statistic[column]),
                    "exact_sign_flip_p": float(raw[column]),
                    "synchronized_max_t_p": float(corrected[column]),
                    "n_positive_subjects": int(np.sum(values > 0)),
                    "n_negative_subjects": int(np.sum(values < 0)),
                }
            )
    inference = pd.DataFrame(inference_rows)
    base = Path(args.output_root) / "group_paired_games" / "localization"
    result_dir = base / "results"
    figure_dir = base / "figures"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(result_dir / "paired_subject_effects.csv", index=False)
    inference.to_csv(result_dir / "paired_game_inference.csv", index=False)
    for estimand, frame in inference.groupby("estimand", sort=False):
        _plot_heatmap(
            frame.rename(columns={"mean_subject_beta": "plot_beta"}),
            value="plot_beta", significance="synchronized_max_t_p",
            output_path=figure_dir / f"figure_{estimand}.png",
            title=(
                "Within-subject across-game mean neural localization"
                if estimand == "across_game_mean"
                else "Paired Game 2 minus Game 1 neural localization"
            ),
        )
    COMMON.atomic_write_json(
        result_dir / "analysis_manifest.json",
        {
            "schema_version": LOCALIZATION_SCHEMA_VERSION,
            "analysis": "paired-game controlled neural localization",
            "subjects": subjects,
            "games": [1, 2],
            "estimands": {
                "across_game_mean": "within-subject (Game1 + Game2) / 2",
                "game2_minus_game1": "within-subject Game2 - Game1",
            },
            "population_unit": "subject",
            "twelve_subject_game_cells_pooled": False,
            "inference": "64 exact synchronized subject sign patterns within estimand and family",
            "validated_shared_contract": shared_contract,
            "inputs": inputs,
        },
    )
    print(f"Paired-game localization complete -> {base}", flush=True)


def _synthetic_group_voxel_test() -> Dict[str, float]:
    import nibabel as nib

    with tempfile.TemporaryDirectory(prefix="localization_group_selftest_") as directory:
        root = Path(directory)
        reference_path = root / "reference.nii.gz"
        nib.save(
            nib.Nifti1Image(
                np.zeros((4, 4, 4, 8), dtype=np.float32), np.eye(4)
            ),
            str(reference_path),
        )
        args = SimpleNamespace(
            output_root=root,
            game=1,
            aggregate_subjects=list(range(1, 7)),
            localization_tier="both",
            state_control="visual",
            visual_pca_dim=20,
            ram_pca_dim=20,
            sessions=None,
            max_runs=None,
            roi_set="all",
            roi_include=None,
            max_roi_voxels=0,
            required_status2=29,
            allow_smoothed_fallback=False,
            feature_root=COMMON.DEFAULT_FEATURE_ROOT,
        )
        configuration = _localization_configuration(args)
        if _configuration_id(configuration) != CANONICAL_CONFIG_ID:
            raise AssertionError("Synthetic canonical configuration routing failed")
        model = "partial::model_margin"
        family = _family(model)
        rois = ("synthetic_roi_a", "synthetic_roi_b")
        coordinates = {
            rois[0]: np.asarray([[0, 0, 0], [1, 1, 1]], dtype=np.int32),
            rois[1]: np.asarray([[1, 1, 1], [2, 2, 2]], dtype=np.int32),
        }
        statistic = f"beta__{_safe_model(model)}"
        source = _source_hashes()
        for subject in range(1, 7):
            paths = _subject_paths(root, subject, 1)
            paths["results"].mkdir(parents=True, exist_ok=True)
            paths["voxels"].mkdir(parents=True, exist_ok=True)
            scale = float(subject) / 10.0
            values = {
                rois[0]: np.asarray([scale, 2.0 * scale], dtype=np.float32),
                rois[1]: np.asarray([2.0 * scale, 3.0 * scale], dtype=np.float32),
            }
            rows = []
            for roi in rois:
                rows.append(
                    {
                        "subject": subject,
                        "game": 1,
                        "roi": roi,
                        "model": model,
                        "family": family,
                        "n_runs": EXPECTED_FULL_RUNS,
                        "n_voxels": 2,
                        "mean_beta": float(np.mean(values[roi])),
                        "median_run_beta": float(np.mean(values[roi])),
                        "n_positive_runs": EXPECTED_FULL_RUNS,
                        "n_negative_runs": 0,
                        "exact_run_sign_flip_p": 2.0 ** (1 - EXPECTED_FULL_RUNS),
                    }
                )
                COMMON.atomic_save_npz(
                    paths["voxels"] / f"{COMMON.safe_name(roi)}.npz",
                    coords=coordinates[roi],
                    **{statistic: values[roi]},
                )
            pd.DataFrame(rows).to_csv(
                paths["results"] / "localization_subject_roi_effects.csv",
                index=False,
            )
            cache_manifest = {
                "process_feature_version": int(COMMON.PROCESS_FEATURE_VERSION),
                "neural_feature_schema_version": int(
                    COMMON.NEURAL_FEATURE_SCHEMA_VERSION
                ),
                "subject": subject,
                "game": 1,
                "feature_root": str(Path(COMMON.DEFAULT_FEATURE_ROOT).resolve()),
                "feature_source": {"04_script_sha256": "a" * 64},
                "runs": [
                    {"run_label": f"synthetic_sub{subject:03d}_run{run:02d}"}
                    for run in range(EXPECTED_FULL_RUNS)
                ],
                "include_topology": False,
                "im_vectors_used": False,
                "roi_set": "all",
                "max_roi_voxels": 0,
                "roi_names": list(rois),
            }
            COMMON.atomic_write_json(
                paths["results"] / "analysis_manifest.json",
                {
                    "schema_version": LOCALIZATION_SCHEMA_VERSION,
                    "analysis": "synthetic localization",
                    "subject": subject,
                    "game": 1,
                    "n_runs": EXPECTED_FULL_RUNS,
                    "n_rois": len(rois),
                    "roi_names": list(rois),
                    "n_models": 1,
                    "models": [model],
                    "families": {model: family},
                    "configuration_id": CANONICAL_CONFIG_ID,
                    "configuration": configuration,
                    "canonical_full_configuration": True,
                    "full_run_contract_complete": True,
                    "reference_fmri": str(reference_path),
                    "source": source,
                    "cache_manifest": cache_manifest,
                },
            )
        run_group(args)
        group_paths = _group_paths(root, 1)
        inference = pd.read_csv(
            group_paths["results"] / "localization_group_inference.csv"
        )
        voxel = pd.read_csv(
            group_paths["results"]
            / "localization_voxelwise_family_fwer_by_roi.csv"
        )
        manifest = json.loads(
            (group_paths["results"] / "analysis_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        overlap = manifest["coordinate_overlap_validation"]
        if (
            len(inference) != 2
            or len(voxel) != 2
            or int(overlap["overlap_entry_count"]) != 1
            or float(overlap["maximum_overlap_value_range"]) > EPS
        ):
            raise AssertionError("Synthetic group voxel inference contract failed")
        return {
            "synthetic_group_roi_rows": float(len(inference)),
            "synthetic_group_voxel_rows": float(len(voxel)),
            "synthetic_group_overlap_entries": float(
                overlap["overlap_entry_count"]
            ),
            "synthetic_group_min_fwer_p": float(
                voxel["minimum_family_fwer_p"].min()
            ),
        }


def self_test() -> Dict[str, float]:
    result = CORE.self_test()
    if _family("partial::model_margin") != "primary_computational_partial":
        raise AssertionError("Localization family routing failed")
    if _family("partial::switch_status_interaction") != "primary_transition_interaction":
        raise AssertionError("Transition localization family routing failed")
    if _family("marginal::model_margin") != "supplementary_marginal":
        raise AssertionError("Localization family routing failed")
    if abs(_exact_run_sign_flip(np.ones(6)) - .03125) > EPS:
        raise AssertionError("Run sign-flip boundary failed")
    rng = np.random.default_rng(809)
    original = rng.normal(size=(6, 4))
    # Entries 1 and 2 represent the same atlas coordinate in two ROIs.
    entries = original[:, [0, 1, 1, 2, 3]]
    inverse = np.asarray([0, 1, 1, 2, 3], dtype=int)
    collapsed, collapse_audit = _collapse_coordinate_entries(
        entries, inverse, 4
    )
    if not np.allclose(collapsed, original, atol=1e-12):
        raise AssertionError("Vectorized unique-coordinate collapse is incorrect")
    inconsistent = entries.copy()
    inconsistent[0, 2] += 1e-3
    try:
        _collapse_coordinate_entries(inconsistent, inverse, 4)
    except ValueError:
        pass
    else:
        raise AssertionError("Coordinate-overlap disagreement was not rejected")
    result.update(
        {
            "coordinate_collapse_overlap_entries": float(
                collapse_audit["overlap_entry_count"]
            ),
            "coordinate_collapse_max_error": float(
                np.max(np.abs(collapsed - original))
            ),
            "coordinate_overlap_disagreement_rejected": 1.0,
        }
    )
    result.update(_synthetic_group_voxel_test())
    return result
