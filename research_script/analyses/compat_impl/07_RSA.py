#!/usr/bin/env python3
"""Canonical native-state and full-planning-trace RSA entry point.

The native-state family tests distinct P, M, Q0, R, E, joint search geometry,
terminal Actor, and switch/stay representations with staged upstream models
and visual, action, task-history, drift/time, and optional RAM nuisances.  The
full-trace family tests cached concat29 and mean29 planning representations.
Both families use signed conditional beta/partial Spearman statistics and
exact synchronized run- and subject-level sign-flip inference.  This script
only reads canonical v4/v1 caches and writes all results below outputs/07_RSA.

Direct transition decoding and MVPA belong to 06_neural_state_space.py.  The
former uncontrolled transition/topology, composite, and feature-resolved
analyses remain available only through explicit legacy flags.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import planning_brain_common as COMMON
import neural_roi_figures as NROI
from scipy.stats import t as student_t


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs" / "07_RSA"
DEFAULT_PLANNING_TRACE_ROOT = (
    SCRIPT_DIR / "outputs" / "00_brain_cache" / "planning_trace_cache"
    / "v1_concat29_mean29"
)
PLANNING_TRACE_SCRIPT = SCRIPT_DIR / "07_planning_trace_RSA.py"
T0 = time.time()

_PLANNING_TRACE_MODULE: Optional[Any] = None


def _load_planning_trace_module() -> Any:
    """Load the maintained full-trace implementation without importing by digit."""
    global _PLANNING_TRACE_MODULE
    if _PLANNING_TRACE_MODULE is not None:
        return _PLANNING_TRACE_MODULE
    if not PLANNING_TRACE_SCRIPT.is_file():
        raise FileNotFoundError(
            f"Missing planning-trace implementation: {PLANNING_TRACE_SCRIPT}"
        )
    module_name = "_thinker_07_planning_trace_RSA"
    spec = importlib.util.spec_from_file_location(
        module_name, PLANNING_TRACE_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load planning-trace implementation: {PLANNING_TRACE_SCRIPT}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    required = (
        "run_subject_analysis", "run_group_analysis", "run_paired_analysis",
        "TRACE_NAMES", "TRACE_LABELS",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise RuntimeError(
            f"Planning-trace module lacks required API: {missing}"
        )
    _PLANNING_TRACE_MODULE = module
    return module

TRANSITION_CONDITIONS = (
    "noop_to_active",
    "active_to_noop",
)
TRANSITION_LABELS = {
    "noop_to_active": "NOOP → active",
    "active_to_noop": "Active → NOOP",
}
TRANSITION_COLORS = {
    "noop_to_active": "#2F6690",
    "active_to_noop": "#E07A2D",
}


# ---------------------------------------------------------------------------
# Controlled RSA: default 02--04 neural bridge analysis
# ---------------------------------------------------------------------------

CONTROLLED_RSA_SCHEMA_VERSION = 1
CONTROLLED_RSA_ESTIMATOR_VERSION = 2
CONTROLLED_RSA_ESTIMATOR_SPEC = (
    "controlled-rsa-fwl-sufficient-statistics-v2: reduced-space "
    "orthonormal projection; one residualized focal regressor; no "
    "materialized reduced/full residual arrays"
)
CONTROLLED_RSA_ESTIMATOR_HASH = hashlib.sha256(
    CONTROLLED_RSA_ESTIMATOR_SPEC.encode("utf-8")
).hexdigest()
V4_COMPATIBLE_CONTROLLED_SOURCES = (
    "55c350383881e7c22ae008d7f762804a183e5c404d9aecd6aacb2f9e1920cc6a",
)
V4_COMPATIBLE_CONTROLLED_SETTINGS_SHA256 = (
    "d5738897a2f5f148b168ce8fb9a06cbe12d04daef68e3c3b5908b10f57860714",
)

def controlled_estimator_provenance() -> Dict[str, object]:
    return {
        "controlled_estimator_version": CONTROLLED_RSA_ESTIMATOR_VERSION,
        "controlled_estimator_sha256": CONTROLLED_RSA_ESTIMATOR_HASH,
        "source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }


def validate_controlled_estimator_manifest(
    manifest: Mapping[str, object],
    path: Path,
    *,
    allow_compatible_v4_source: bool = False,
) -> None:
    expected = controlled_estimator_provenance()
    for key, value in expected.items():
        if key == "source_sha256" and allow_compatible_v4_source:
            observed = str(manifest.get(key, ""))
            if observed in V4_COMPATIBLE_CONTROLLED_SOURCES:
                continue
        if manifest.get(key) != value:
            raise ValueError(
                f"Controlled RSA estimator provenance mismatch in {path}: "
                f"{key}={manifest.get(key)!r} != {value!r}"
            )


CONTROLLED_EXPECTED_04_SCRIPT_SHA256 = (
    "2f9027b4fb1e978d55f9d26b096bace6d2619a6cd406c9b783f9c3523e00d69c"
)
CONTROLLED_CANONICAL_FIXED_SETTINGS = {
    "analysis_mode": "controlled_02_04_pathway",
    "n_runs": 11,
    "n_rois": 110,
    "n_tr_values": [480],
    "roi_set": "all",
    "roi_include": None,
    "max_roi_voxels": 0,
    "sessions": None,
    "max_runs": None,
    "required_status2": 29,
    "allow_smoothed_fallback": False,
    "include_ram_control": False,
    "min_pair_lag_tr": 10,
    "n_permutations": 1000,
    "permutation_block_size_tr": 40,
    "random_state": 2026,
    "cache_version": 4,
    "process_feature_version": 8,
    "neural_feature_schema_version": 4,
    "feature_source_schema_version": 4,
    "feature_source_04_script_sha256": (
        CONTROLLED_EXPECTED_04_SCRIPT_SHA256
    ),
    "full_actor_spine_preserved_in_04_source": True,
    "process_rows": "exact aligned model_geometry_eligible subset only",
    "im_vectors_used": False,
    "include_topology": False,
    "run_cache_schemas": ["canonical_00brain_nativehrf_schema4_v8"],
}


def _controlled_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _controlled_roi_names_sha256(names: Sequence[str]) -> str:
    payload = "\n".join(sorted(str(name) for name in names)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _controlled_cache_source(cache: Mapping[str, object]) -> Dict[str, object]:
    feature = cache.get("feature_source")
    if not isinstance(feature, Mapping) or not isinstance(
        feature.get("source_signature"), Mapping
    ):
        raise ValueError("Controlled RSA cache provenance is incomplete")
    return {
        "cache_version": int(cache.get("cache_version", -1)),
        "process_feature_version": int(
            cache.get("process_feature_version", -1)
        ),
        "neural_feature_schema_version": int(
            cache.get("neural_feature_schema_version", -1)
        ),
        "feature_source_token": str(cache.get("feature_source_token", "")),
        "feature_source_schema_version": int(
            feature.get("schema_version", -1)
        ),
        "feature_source_04_script_sha256": str(
            feature.get("04_script_sha256", "")
        ),
        "feature_source_signature_sha256": _controlled_json_sha256(
            feature["source_signature"]
        ),
    }


def controlled_subject_provenance(
    args: argparse.Namespace,
    run_specs: Sequence["COMMON.BrainRunSpec"],
    cache: Mapping[str, object],
    roi_names: Sequence[str],
) -> Dict[str, object]:
    feature = cache.get("feature_source", {})
    if not isinstance(feature, Mapping):
        raise ValueError("Controlled RSA feature-source provenance is missing")
    settings = {
        "analysis_mode": "controlled_02_04_pathway",
        "n_runs": len(run_specs),
        "n_rois": len(roi_names),
        "n_tr_values": sorted({int(run.n_tr) for run in run_specs}),
        "roi_set": str(args.roi_set),
        "roi_include": (
            None if args.roi_include is None else str(args.roi_include)
        ),
        "max_roi_voxels": int(args.max_roi_voxels),
        "sessions": None if args.sessions is None else str(args.sessions),
        "max_runs": None if args.max_runs is None else int(args.max_runs),
        "required_status2": int(args.required_status2),
        "allow_smoothed_fallback": bool(args.allow_smoothed_fallback),
        "include_ram_control": bool(args.include_ram_control),
        "min_pair_lag_tr": int(args.min_pair_lag),
        "n_permutations": int(args.n_permutations),
        "permutation_block_size_tr": int(args.permutation_block_size),
        "random_state": int(args.random_state),
        "cache_version": int(cache.get("cache_version", -1)),
        "process_feature_version": int(
            cache.get("process_feature_version", -1)
        ),
        "neural_feature_schema_version": int(
            cache.get("neural_feature_schema_version", -1)
        ),
        "feature_source_schema_version": int(
            feature.get("schema_version", -1)
        ),
        "feature_source_04_script_sha256": str(
            feature.get("04_script_sha256", "")
        ),
        "full_actor_spine_preserved_in_04_source": bool(
            cache.get("full_actor_spine_preserved_in_04_source", False)
        ),
        "process_rows": str(cache.get("process_rows", "")),
        "im_vectors_used": bool(cache.get("im_vectors_used", True)),
        "include_topology": bool(cache.get("include_topology", True)),
        "run_cache_schemas": sorted(
            {str(run.cache_schema) for run in run_specs}
        ),
        "roi_names_sha256": _controlled_roi_names_sha256(roi_names),
    }
    source = _controlled_cache_source(cache)
    fixed_ok = all(
        settings.get(key) == expected
        for key, expected in CONTROLLED_CANONICAL_FIXED_SETTINGS.items()
    )
    if not fixed_ok:
        bad = {
            key: (settings.get(key), expected)
            for key, expected in CONTROLLED_CANONICAL_FIXED_SETTINGS.items()
            if settings.get(key) != expected
        }
        raise ValueError(
            f"Controlled RSA subject mode requires the canonical full "
            f"contract; mismatches={bad}"
        )

    subject = int(args.subject)
    game = int(args.game)
    ordered_labels = [str(run.run_label) for run in run_specs]
    if len(ordered_labels) != 11 or len(set(ordered_labels)) != 11:
        raise ValueError("Controlled RSA requires 11 unique ordered runs")
    roi_list = [str(name) for name in roi_names]
    if len(roi_list) != 110 or len(set(roi_list)) != 110:
        raise ValueError("Controlled RSA requires 110 unique common ROIs")
    roi_digest = _controlled_roi_names_sha256(roi_list)
    top_token = str(cache.get("feature_source_token", ""))
    feature_token = str(feature.get("source_token", ""))
    recomputed_token = COMMON.signature_token(feature["source_signature"])
    if not top_token or not (
        top_token == feature_token == recomputed_token
    ):
        raise ValueError("Controlled RSA feature-source token mismatch")
    if (
        int(cache.get("subject", -1)) != subject
        or int(cache.get("game", -1)) != game
        or str(cache.get("roi_set", "")) != "all"
        or int(cache.get("max_roi_voxels", -1)) != 0
        or sorted(str(name) for name in cache.get("roi_names", []))
        != sorted(roi_list)
        or int(feature.get("subject", -1)) != subject
        or int(feature.get("game", -1)) != game
        or int(feature.get("n_runs", -1)) != 11
        or bool(feature.get("im_vectors_used", True))
        or bool(feature.get("raw_topology_used", True))
    ):
        raise ValueError("Controlled RSA cache/feature subject contract mismatch")

    cache_runs = cache.get("runs")
    feature_runs = feature.get("runs")
    if not isinstance(cache_runs, list) or not isinstance(feature_runs, list):
        raise ValueError("Controlled RSA cache lacks ordered run manifests")
    cache_labels = [str(row.get("run_label", "")) for row in cache_runs]
    feature_labels = [str(row.get("run_label", "")) for row in feature_runs]
    if cache_labels != ordered_labels or feature_labels != ordered_labels:
        raise ValueError("Controlled RSA run order differs across cache sources")

    run_contract: List[Dict[str, object]] = []
    for run, cache_run in zip(run_specs, cache_runs):
        if (
            int(run.subject) != subject
            or int(run.game) != game
            or int(run.n_tr) != 480
            or str(run.cache_schema)
            != "canonical_00brain_nativehrf_schema4_v8"
            or str(run.feature_source_token) != top_token
            or sorted(str(name) for name in run.roi_names)
            != sorted(roi_list)
            or int(cache_run.get("n_tr", -1)) != 480
            or sorted(str(name) for name in cache_run.get("roi_names", []))
            != sorted(roi_list)
        ):
            raise ValueError(
                f"Controlled RSA run contract mismatch: {run.run_label}"
            )
        run_contract.append(
            {
                "run_label": str(run.run_label),
                "subject": int(run.subject),
                "session": int(run.session),
                "block": int(run.block),
                "game": int(run.game),
                "n_tr": int(run.n_tr),
                "cache_schema": str(run.cache_schema),
                "feature_source_token": str(run.feature_source_token),
                "roi_names_sha256": roi_digest,
            }
        )
    return {
        "controlled_canonical_full_analysis": bool(fixed_ok),
        "controlled_settings": settings,
        "controlled_settings_sha256": _controlled_json_sha256(settings),
        "controlled_cache_source": source,
        "controlled_cache_source_sha256": _controlled_json_sha256(source),
        "controlled_run_contract": run_contract,
        "controlled_run_contract_sha256": _controlled_json_sha256(
            run_contract
        ),
        "controlled_execution_settings": {
            "n_jobs": int(args.n_jobs),
            "permutation_batch_size": int(args.permutation_batch_size),
        },
    }


def validate_controlled_subject_provenance(
    manifest: Mapping[str, object],
    path: Path,
    frame: Optional[pd.DataFrame] = None,
    *,
    allow_compatible_v4_source: bool = False,
) -> Tuple[str, Dict[str, object], str]:
    validate_controlled_estimator_manifest(
        manifest, path,
        allow_compatible_v4_source=allow_compatible_v4_source,
    )
    settings = manifest.get("controlled_settings")
    if not isinstance(settings, Mapping):
        raise ValueError(f"Missing controlled settings in {path}")
    digest = _controlled_json_sha256(settings)
    if manifest.get("controlled_settings_sha256") != digest:
        raise ValueError(f"Controlled settings hash mismatch in {path}")
    observed_source = str(manifest.get("source_sha256", ""))
    if (
        allow_compatible_v4_source
        and observed_source in V4_COMPATIBLE_CONTROLLED_SOURCES
        and digest not in V4_COMPATIBLE_CONTROLLED_SETTINGS_SHA256
    ):
        raise ValueError(
            f"Compatible v4 settings hash mismatch in {path}: {digest}"
        )
    bad = {
        key: (settings.get(key), expected)
        for key, expected in CONTROLLED_CANONICAL_FIXED_SETTINGS.items()
        if settings.get(key) != expected
    }
    if (
        bad
        or manifest.get("controlled_canonical_full_analysis") is not True
    ):
        raise ValueError(
            f"Non-canonical controlled RSA subject output in {path}: {bad}"
        )
    if (
        int(manifest.get("n_runs", -1)) != 11
        or int(manifest.get("n_rois", -1)) != 110
        or int(settings["n_runs"]) != 11
        or int(settings["n_rois"]) != 110
    ):
        raise ValueError(f"Incomplete controlled RSA subject output in {path}")
    roi_digest = str(settings.get("roi_names_sha256", ""))
    if len(roi_digest) != 64:
        raise ValueError(f"Invalid ROI-set digest in {path}")
    cache = manifest.get("cache_manifest")
    if not isinstance(cache, Mapping):
        raise ValueError(f"Missing cache manifest in {path}")
    source = _controlled_cache_source(cache)
    if (
        manifest.get("controlled_cache_source") != source
        or manifest.get("controlled_cache_source_sha256")
        != _controlled_json_sha256(source)
    ):
        raise ValueError(
            f"Controlled cache-source provenance mismatch in {path}"
        )
    subject = int(manifest.get("subject", -1))
    game = int(manifest.get("game", -1))
    feature = cache.get("feature_source")
    if not isinstance(feature, Mapping):
        raise ValueError(f"Missing feature-source manifest in {path}")
    top_token = str(cache.get("feature_source_token", ""))
    feature_token = str(feature.get("source_token", ""))
    recomputed_token = COMMON.signature_token(feature["source_signature"])
    if not top_token or not (
        top_token == feature_token == recomputed_token
    ):
        raise ValueError(f"Feature-source token mismatch in {path}")
    if (
        int(cache.get("subject", -1)) != subject
        or int(cache.get("game", -1)) != game
        or str(cache.get("roi_set", "")) != "all"
        or int(cache.get("max_roi_voxels", -1)) != 0
        or bool(cache.get("im_vectors_used", True))
        or bool(cache.get("include_topology", True))
        or int(feature.get("subject", -1)) != subject
        or int(feature.get("game", -1)) != game
        or int(feature.get("n_runs", -1)) != 11
        or bool(feature.get("im_vectors_used", True))
        or bool(feature.get("raw_topology_used", True))
    ):
        raise ValueError(f"Cache subject/source contract mismatch in {path}")

    run_contract = manifest.get("controlled_run_contract")
    if not isinstance(run_contract, list) or len(run_contract) != 11:
        raise ValueError(f"Missing complete run contract in {path}")
    if manifest.get("controlled_run_contract_sha256") != _controlled_json_sha256(
        run_contract
    ):
        raise ValueError(f"Run-contract hash mismatch in {path}")
    contract_labels = [str(row.get("run_label", "")) for row in run_contract]
    if len(set(contract_labels)) != 11:
        raise ValueError(f"Duplicate run contract in {path}")
    cache_runs = cache.get("runs")
    feature_runs = feature.get("runs")
    if not isinstance(cache_runs, list) or not isinstance(feature_runs, list):
        raise ValueError(f"Missing nested run manifests in {path}")
    cache_labels = [str(row.get("run_label", "")) for row in cache_runs]
    feature_labels = [str(row.get("run_label", "")) for row in feature_runs]
    if contract_labels != cache_labels or contract_labels != feature_labels:
        raise ValueError(f"Run order mismatch in {path}")
    if _controlled_roi_names_sha256(cache.get("roi_names", [])) != roi_digest:
        raise ValueError(f"Cache ROI grid mismatch in {path}")
    for contract, cache_run in zip(run_contract, cache_runs):
        expected_cache_schema = "canonical_00brain_nativehrf_schema4_v8"
        if (
            int(contract.get("subject", -1)) != subject
            or int(contract.get("game", -1)) != game
            or int(contract.get("n_tr", -1)) != 480
            or str(contract.get("cache_schema", ""))
            != expected_cache_schema
            or str(contract.get("feature_source_token", "")) != top_token
            or str(contract.get("roi_names_sha256", "")) != roi_digest
            or int(cache_run.get("n_tr", -1)) != 480
            or _controlled_roi_names_sha256(cache_run.get("roi_names", []))
            != roi_digest
        ):
            raise ValueError(f"Nested run contract mismatch in {path}")
    if frame is not None:
        required = {
            "roi",
            "effect",
            "statistic",
            "mean_effect",
            "n_units",
            "n_sign_patterns",
            "n_runs_descriptive",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Subject RSA input lacks {missing}: {path}")
        numeric_columns = (
            "mean_effect",
            "t_stat",
            "positive_fraction",
            "p_exact_two_sided",
            "p_max_abs_t_fwer",
            "q_bh_family",
            "mean_delta_r2",
            "median_delta_r2",
        )
        numeric_finite = all(
            column in frame
            and np.all(np.isfinite(frame[column].to_numpy(dtype=float)))
            for column in numeric_columns
        )
        probability_columns = (
            "positive_fraction",
            "p_exact_two_sided",
            "p_max_abs_t_fwer",
            "q_bh_family",
        )
        probability_valid = all(
            np.all(
                (frame[column].to_numpy(dtype=float) >= 0.0)
                & (frame[column].to_numpy(dtype=float) <= 1.0)
            )
            for column in probability_columns
        )
        expected_families = {
            effect: controlled_effect_family(effect)
            for effect in CONTROLLED_EFFECT_ORDER
        }
        family_valid = all(
            set(
                frame.loc[
                    frame["effect"].astype(str).eq(effect),
                    "inference_family",
                ].astype(str)
            )
            == {family}
            for effect, family in expected_families.items()
        )
        label_valid = all(
            set(
                frame.loc[
                    frame["effect"].astype(str).eq(effect),
                    "effect_label",
                ].astype(str)
            )
            == {CONTROLLED_EFFECT_LABELS[effect]}
            for effect in CONTROLLED_EFFECT_ORDER
        )
        if (
            len(frame) != 2420
            or frame[["roi", "effect", "statistic"]].duplicated().any()
            or frame["roi"].nunique() != 110
            or set(frame["effect"].astype(str)) != set(CONTROLLED_EFFECT_ORDER)
            or set(frame["statistic"].astype(str))
            != {"added_beta", "conditional_partial_r"}
            or not numeric_finite
            or not probability_valid
            or not family_valid
            or not label_valid
            or not np.all(frame["n_units"].to_numpy(int) == 11)
            or not np.all(frame["n_sign_patterns"].to_numpy(int) == 2048)
            or not np.all(frame["n_runs_descriptive"].to_numpy(int) == 11)
        ):
            raise ValueError(f"Non-canonical subject RSA result grid: {path}")
        if _controlled_roi_names_sha256(
            frame["roi"].astype(str).unique()
        ) != roi_digest:
            raise ValueError(f"Subject RSA ROI grid mismatch in {path}")
    return digest, dict(settings), str(
        manifest["controlled_cache_source_sha256"]
    )
CONTROLLED_PRIMARY_ORDER = (
    "P",
    "M",
    "Q0",
    "R",
    "E",
    "G_joint",
    "A",
)
CONTROLLED_TRANSITION_ORDER = (
    "switch_status_interaction",
)
CONTROLLED_SENSITIVITY_ORDER = (
    "E_all29",
    "G_breadth",
    "G_depth",
)
CONTROLLED_EFFECT_ORDER = (
    "P",
    "M",
    "Q0",
    "R",
    "E",
    "E_all29",
    "G_joint",
    "G_breadth",
    "G_depth",
    "A",
    *CONTROLLED_TRANSITION_ORDER,
)
CONTROLLED_EFFECT_LABELS = {
    "P": "Pre-search model-prior policy",
    "M": "Pre-search current-action model margin",
    "Q0": "Initial incumbent Q state",
    "R": "Within-search Q revision",
    "E": "Mean rollout evidence",
    "E_all29": "All-29-event mean rollout evidence",
    "G_joint": "Search geometry (breadth + depth)",
    "G_breadth": "Search breadth",
    "G_depth": "Search depth",
    "A": "Terminal Actor residual beyond model prior",
    "switch_status_interaction": (
        "Active-status switch/stay interaction geometry"
    ),
}
CONTROLLED_BASE_NUISANCE_ORDER = (
    "N_visual",
    "N_RAM",
    "N_action",
    "N_decision_density",
    "N_history",
    "N_drift",
    "N_time",
)
CONTROLLED_RDM_ORDER = (
    *CONTROLLED_BASE_NUISANCE_ORDER,
    "N_origin_active",
    "N_destination_active",
    *CONTROLLED_EFFECT_ORDER,
)


def _as_finite_feature_matrix(
    values: np.ndarray,
    *,
    name: str,
    n_rows: Optional[int] = None,
) -> np.ndarray:
    """Validate one TR-aligned feature block without reordering columns."""
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2-D feature block")
    if n_rows is not None and len(matrix) != int(n_rows):
        raise ValueError(
            f"TR row mismatch for {name}: {len(matrix)} != {int(n_rows)}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"Non-finite values in controlled RSA block {name}")
    return matrix


def _required_controlled_block(
    blocks: Mapping[str, np.ndarray],
    name: str,
    *,
    n_rows: Optional[int] = None,
) -> np.ndarray:
    if name not in blocks:
        raise KeyError(
            f"Controlled RSA requires canonical TR block '{name}'. "
            "Rebuild the shared neural cache with the 02--04 bridge schema."
        )
    return _as_finite_feature_matrix(blocks[name], name=name, n_rows=n_rows)


def _optional_controlled_block(
    blocks: Mapping[str, np.ndarray],
    name: str,
    *,
    n_rows: int,
) -> np.ndarray:
    if name not in blocks:
        return np.zeros((int(n_rows), 1), dtype=float)
    return _as_finite_feature_matrix(blocks[name], name=name, n_rows=n_rows)


def _feature_rdm(values: np.ndarray) -> np.ndarray:
    """Euclidean RDM after removing exactly constant feature columns."""
    matrix = _as_finite_feature_matrix(values, name="feature_rdm")
    spread = np.std(matrix, axis=0)
    keep = np.isfinite(spread) & (spread > 1e-12)
    if not np.any(keep):
        n_rows = len(matrix)
        return np.zeros(n_rows * (n_rows - 1) // 2, dtype=np.float32)
    return COMMON.euclidean_rdm(matrix[:, keep])


def _pattern_rdm(values: np.ndarray) -> np.ndarray:
    """Correlation-distance RDM for high-dimensional control patterns."""
    matrix = _as_finite_feature_matrix(values, name="pattern_rdm")
    spread = np.std(matrix, axis=0)
    keep = np.isfinite(spread) & (spread > 1e-12)
    if np.sum(keep) < 2:
        n_rows = len(matrix)
        return np.zeros(n_rows * (n_rows - 1) // 2, dtype=np.float32)
    return COMMON.correlation_rdm(matrix[:, keep])


def _categorical_mismatch_rdm(labels: np.ndarray) -> np.ndarray:
    values = np.asarray(labels).reshape(-1)
    row, column = np.triu_indices(len(values), k=1)
    return (values[row] != values[column]).astype(np.float32)


def build_controlled_rdms(
    blocks: Mapping[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Build canonical controlled-RSA RDMs from HRF-aligned TR blocks.

    Canonical process variables are never replaced by legacy proxies.  A
    missing P, M, Q0, R, E, G, or A block therefore fails loudly instead of
    producing an apparently valid analysis with a changed estimand.
    """
    prior = _required_controlled_block(blocks, "model_prior_policy")
    n_tr = len(prior)
    margin = _required_controlled_block(blocks, "model_margin", n_rows=n_tr)
    initial_q = _required_controlled_block(
        blocks, "initial_q_state", n_rows=n_tr
    )
    revision = _required_controlled_block(
        blocks, "revision_primary", n_rows=n_tr
    )
    evidence = _required_controlled_block(
        blocks, "evidence_primary", n_rows=n_tr
    )
    evidence_all29 = _required_controlled_block(
        blocks, "evidence_all29", n_rows=n_tr
    )
    geometry = _required_controlled_block(
        blocks, "geometry_primary", n_rows=n_tr
    )
    if geometry.shape[1] < 2:
        raise ValueError(
            "geometry_primary must contain breadth then depth components"
        )
    actor = _required_controlled_block(blocks, "actor_readout", n_rows=n_tr)
    if actor.shape[1] < 2:
        raise ValueError(
            "actor_readout must contain actual margin then residual beyond prior"
        )
    transition = _required_controlled_block(
        blocks, "transition_factors", n_rows=n_tr
    )
    if transition.shape[1] < 2:
        raise ValueError(
            "transition_factors must begin with origin_active and "
            "destination_active"
        )
    # These are canonical-HRF-aligned continuous regressors.  Thresholding
    # them back into labels would discard the temporal response shape and
    # create arbitrary category boundaries at overlapping events.
    origin = transition[:, [0]]
    destination = transition[:, [1]]
    if transition.shape[1] < 3:
        raise ValueError("transition_factors lacks the switch component")
    switch = transition[:, [2]]

    visual = _required_controlled_block(blocks, "visual", n_rows=n_tr)
    ram = _optional_controlled_block(blocks, "ram", n_rows=n_tr)
    current_action = _required_controlled_block(
        blocks, "current_action", n_rows=n_tr
    )
    previous_action = _required_controlled_block(
        blocks, "previous_action", n_rows=n_tr
    )
    decision_event = _required_controlled_block(
        blocks, "decision_event", n_rows=n_tr
    )
    history = _required_controlled_block(
        blocks, "task_history", n_rows=n_tr
    )
    drift = _required_controlled_block(blocks, "drift", n_rows=n_tr)

    rdms: Dict[str, np.ndarray] = {
        "N_visual": _pattern_rdm(visual),
        "N_RAM": _pattern_rdm(ram),
        "N_action": _feature_rdm(
            np.column_stack((current_action, previous_action))
        ),
        "N_decision_density": _feature_rdm(decision_event),
        "N_history": _feature_rdm(history),
        "N_drift": _feature_rdm(drift),
        "N_time": COMMON.temporal_lag_rdm(n_tr),
        "N_origin_active": _feature_rdm(origin),
        "N_destination_active": _feature_rdm(destination),
        "P": _feature_rdm(prior),
        "M": _feature_rdm(margin),
        "Q0": _feature_rdm(initial_q),
        "R": _feature_rdm(revision[:, [0]]),
        "E": _feature_rdm(evidence[:, [0]]),
        "E_all29": _feature_rdm(evidence_all29[:, [0]]),
        "G_joint": _feature_rdm(
            _optional_controlled_block(
                blocks, "geometry_joint_loro", n_rows=n_tr
            )[:, :2]
            if "geometry_joint_loro" in blocks
            else geometry[:, :2]
        ),
        "G_breadth": _feature_rdm(geometry[:, [0]]),
        "G_depth": _feature_rdm(geometry[:, [1]]),
        "A": _feature_rdm(actor[:, [1]]),
        "switch_status_interaction": _feature_rdm(switch),
    }
    if tuple(rdms) != CONTROLLED_RDM_ORDER:
        raise AssertionError(f"Controlled RDM order changed: {tuple(rdms)}")
    expected_pairs = n_tr * (n_tr - 1) // 2
    bad = {
        name: len(values)
        for name, values in rdms.items()
        if len(values) != expected_pairs
    }
    if bad:
        raise AssertionError(
            f"Controlled RDM pair-count mismatch "
            f"({expected_pairs} expected): {bad}"
        )
    return rdms


def controlled_stage_controls() -> Dict[str, Tuple[str, ...]]:
    """Theory-ordered reduced models for primary and sensitivity RDMs."""
    nuisance = CONTROLLED_BASE_NUISANCE_ORDER
    through_p = (*nuisance, "P")
    through_m = (*through_p, "M")
    through_q0 = (*through_m, "Q0")
    through_r = (*through_q0, "R")
    through_e = (*through_r, "E")
    return {
        "P": nuisance,
        # M is the current-action contrast beyond the complete model-prior
        # policy geometry, keeping the two pre-search representations distinct.
        "M": through_p,
        "Q0": through_m,
        "R": through_q0,
        "E": through_r,
        # All-29 evidence replaces, and is never co-entered with, primary E.
        "E_all29": through_r,
        "G_joint": through_e,
        # Individual axes are descriptive sensitivity tests against the same
        # upstream model; joint G remains the primary geometry estimand.
        "G_breadth": through_e,
        "G_depth": through_e,
        "A": (*through_e, "G_joint"),
        # Status-switch interaction is tested only beyond endpoint main
        # effects and the full visual/RAM/action/history/time nuisance set.
        "switch_status_interaction": (
            *nuisance,
            "N_origin_active",
            "N_destination_active",
        ),
    }


def _ranked_design_with_deterministic_pruning(
    rdms: Mapping[str, np.ndarray],
    control_names: Sequence[str],
    target_name: str,
    pair_mask: np.ndarray,
    *,
    tolerance: float = 1e-10,
) -> Dict[str, Any]:
    """Factor one conditional RSA design while pruning nuisance aliases only.

    Control priority follows the explicit stage order.  The focal RDM is
    always retained or reported as non-estimable; it is never silently
    dropped in order to obtain a full-rank fit.
    """
    mask = np.asarray(pair_mask, dtype=bool)
    if int(np.sum(mask)) < 8:
        raise ValueError("Controlled RSA needs at least eight retained pairs")
    n_total_pairs = len(mask)
    kept_names: List[str] = []
    dropped_names: List[str] = []
    kept_columns: List[np.ndarray] = []
    basis: List[np.ndarray] = []
    intercept = np.ones(int(np.sum(mask)), dtype=float)
    intercept /= np.linalg.norm(intercept)
    basis.append(intercept)

    def residual_against_basis(column: np.ndarray) -> np.ndarray:
        residual = np.asarray(column, dtype=float).copy()
        # Re-orthogonalization stabilizes the exact-alias decision.
        for _ in range(2):
            for vector in basis:
                residual -= vector * float(np.dot(vector, residual))
        return residual

    for name in control_names:
        if name not in rdms:
            raise KeyError(f"Missing controlled RSA RDM: {name}")
        raw = np.asarray(rdms[name], dtype=float)
        if len(raw) != n_total_pairs:
            raise ValueError(f"RDM length mismatch for {name}")
        column = COMMON.rank_zscore(raw)[mask]
        residual = residual_against_basis(column)
        scale = max(float(np.linalg.norm(column)), 1.0)
        residual_norm = float(np.linalg.norm(residual))
        if residual_norm <= tolerance * scale:
            dropped_names.append(name)
            continue
        kept_names.append(name)
        kept_columns.append(column)
        basis.append(residual / residual_norm)

    if target_name not in rdms:
        raise KeyError(f"Missing focal controlled RSA RDM: {target_name}")
    raw_target = np.asarray(rdms[target_name], dtype=float)
    if len(raw_target) != n_total_pairs:
        raise ValueError(f"RDM length mismatch for focal {target_name}")
    target = COMMON.rank_zscore(raw_target)[mask]
    target_residual = residual_against_basis(target)
    target_scale = max(float(np.linalg.norm(target)), 1.0)
    target_residual_norm = float(np.linalg.norm(target_residual))
    if target_residual_norm <= tolerance * target_scale:
        raise ValueError(
            f"Focal RDM {target_name} is not estimable beyond controls "
            f"{kept_names}; dropped nuisance aliases={dropped_names}"
        )

    reduced = np.column_stack((np.ones(len(target)), *kept_columns))
    full = np.column_stack((reduced, target))
    reduced_rcond = np.finfo(float).eps * max(reduced.shape)
    full_rcond = np.finfo(float).eps * max(full.shape)
    reduced_q = np.column_stack(basis)
    target_residual_ss = float(np.dot(target_residual, target_residual))
    return {
        "target": target,
        "reduced_design": reduced,
        "full_design": full,
        "reduced_pinv": np.linalg.pinv(reduced, rcond=reduced_rcond),
        "full_pinv": np.linalg.pinv(full, rcond=full_rcond),
        "reduced_q": reduced_q,
        "target_residual": target_residual,
        "target_residual_ss": target_residual_ss,
        "kept_controls": tuple(kept_names),
        "dropped_controls": tuple(dropped_names),
        "target_residual_ratio": target_residual_norm / target_scale,
        "n_pairs_total": int(n_total_pairs),
        "n_pairs_retained": int(np.sum(mask)),
    }


def prepare_controlled_comparisons(
    rdms: Mapping[str, np.ndarray],
    pair_mask: np.ndarray,
) -> Dict[str, Dict[str, Any]]:
    stages = controlled_stage_controls()
    if tuple(stages) != CONTROLLED_EFFECT_ORDER:
        raise AssertionError(f"Controlled stage order changed: {tuple(stages)}")
    return {
        target: _ranked_design_with_deterministic_pruning(
            rdms,
            stages[target],
            target,
            pair_mask,
        )
        for target in CONTROLLED_EFFECT_ORDER
    }


def score_controlled_design(
    ranked_brain: np.ndarray,
    design: Mapping[str, Any],
) -> Dict[str, float]:
    """Score a ranked neural RDM against an already factored fixed design."""
    y = np.asarray(ranked_brain, dtype=float).reshape(-1)
    reduced_q = np.asarray(design["reduced_q"], dtype=float)
    if len(y) != len(reduced_q):
        raise ValueError(
            f"Neural/design pair mismatch: {len(y)} != {len(reduced_q)}"
        )
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    if denominator <= 1e-12:
        return {
            "reduced_r2": np.nan,
            "full_r2": np.nan,
            "delta_r2": np.nan,
            "added_beta": np.nan,
            "conditional_partial_r": np.nan,
        }
    projection = reduced_q.T @ y
    reduced_ss = max(
        float(np.dot(y, y) - np.dot(projection, projection)), 0.0
    )
    target_residual = np.asarray(design["target_residual"], dtype=float)
    target_residual_ss = float(design["target_residual_ss"])
    target_dot = float(np.dot(y, target_residual))
    improvement_ss = target_dot * target_dot / target_residual_ss
    full_ss = max(reduced_ss - improvement_ss, 0.0)
    partial_denominator = math.sqrt(target_residual_ss * reduced_ss)
    return {
        "reduced_r2": float(1.0 - reduced_ss / denominator),
        "full_r2": float(1.0 - full_ss / denominator),
        "delta_r2": float(improvement_ss / denominator),
        "added_beta": float(target_dot / target_residual_ss),
        "conditional_partial_r": (
            float(target_dot / partial_denominator)
            if partial_denominator > 0
            else 0.0
        ),
    }


def score_controlled_design_batch(
    ranked_brain_batch: np.ndarray,
    design: Mapping[str, Any],
) -> Dict[str, np.ndarray]:
    """Vectorized counterpart used for synchronized temporal null draws."""
    targets = np.asarray(ranked_brain_batch, dtype=float)
    if targets.ndim == 1:
        targets = targets[None, :]
    reduced_q = np.asarray(design["reduced_q"], dtype=float)
    if targets.shape[1] != len(reduced_q):
        raise ValueError("Null target/design pair mismatch")
    y_ss = np.einsum("ij,ij->i", targets, targets)
    y_sum = np.sum(targets, axis=1)
    denominator = y_ss - y_sum * y_sum / float(targets.shape[1])
    projection = targets @ reduced_q
    reduced_ss = np.maximum(
        y_ss - np.einsum("ij,ij->i", projection, projection), 0.0
    )
    target_residual = np.asarray(design["target_residual"], dtype=float)
    target_residual_ss = float(design["target_residual_ss"])
    target_dot = targets @ target_residual
    improvement_ss = target_dot * target_dot / target_residual_ss
    delta_r2 = np.divide(
        improvement_ss,
        denominator,
        out=np.full(len(targets), np.nan, dtype=float),
        where=denominator > 1e-12,
    )
    partial_denominator = np.sqrt(target_residual_ss * reduced_ss)
    partial = np.divide(
        target_dot,
        partial_denominator,
        out=np.zeros(len(targets), dtype=float),
        where=partial_denominator > 0,
    )
    return {
        "delta_r2": delta_r2,
        "added_beta": target_dot / target_residual_ss,
        "conditional_partial_r": partial,
    }


def controlled_effect_family(effect: str) -> str:
    if effect in CONTROLLED_PRIMARY_ORDER:
        return "primary_planning"
    if effect in CONTROLLED_TRANSITION_ORDER:
        return "primary_transition"
    if effect in CONTROLLED_SENSITIVITY_ORDER:
        return "sensitivity"
    raise KeyError(f"Unknown controlled RSA effect: {effect}")


def _controlled_null_key(
    run_label: str,
    roi: str,
    effect: str,
    statistic: str,
) -> str:
    return "__".join(
        (
            COMMON.safe_name(run_label),
            COMMON.safe_name(roi),
            COMMON.safe_name(effect),
            COMMON.safe_name(statistic),
        )
    )


def load_controlled_run_blocks(
    run: COMMON.BrainRunSpec,
    *,
    include_ram_control: bool,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """Strict read-only adapter for the native-time schema-4 cache."""
    process_version = int(
        getattr(COMMON, "PROCESS_FEATURE_VERSION", -1)
    )
    if process_version < 8:
        raise RuntimeError(
            "Controlled RSA requires canonical active-status-switch process "
            f"features (version >=8); found {process_version}"
        )
    if int(getattr(COMMON, "NEURAL_FEATURE_SCHEMA_VERSION", -1)) != 4:
        raise RuntimeError(
            "Controlled RSA requires neural feature schema version 4"
        )
    process = COMMON.load_process_blocks(run)
    expected_components = getattr(COMMON, "PROCESS_BLOCK_COMPONENTS", None)
    if not isinstance(expected_components, Mapping):
        raise RuntimeError(
            "planning_brain_common lacks PROCESS_BLOCK_COMPONENTS; "
            "the canonical schema-4 native-time cache API is required"
        )
    if tuple(expected_components.get("transition_events", ())) != (
        "N_to_N",
        "N_to_A",
        "A_to_N",
        "A_to_A",
    ):
        raise RuntimeError(
            "Controlled RSA requires canonical N_N/N_A/A_N/A_A event order"
        )
    transition_factor_names = tuple(
        expected_components.get("transition_factors", ())
    )
    if (
        transition_factor_names[:2]
        != ("origin_active", "destination_active")
        or len(transition_factor_names) != 3
        or transition_factor_names[2] not in {"status_switch", "switch"}
    ):
        raise RuntimeError(
            "Controlled RSA requires active-status switch, not exact-action "
            "change, in transition_factors"
        )
    required = (
        "model_prior_policy",
        "model_margin",
        "initial_q_state",
        "revision_primary",
        "evidence_primary",
        "evidence_all29",
        "geometry_primary",
        "actor_readout",
        "decision_event",
        "transition_events",
        "transition_factors",
        "current_action",
        "previous_action",
        "task_history",
        "drift",
    )
    blocks: Dict[str, np.ndarray] = {}
    component_audit: Dict[str, List[str]] = {}
    for name in required:
        if name not in process:
            raise KeyError(
                f"{run.run_label} process cache lacks canonical block {name}; "
                "rebuild it with 00_brain_cache.py"
            )
        expected = tuple(str(value) for value in expected_components[name])
        values = _as_finite_feature_matrix(
            process[name], name=name, n_rows=run.n_tr
        )
        if values.shape[1] != len(expected):
            raise ValueError(
                f"{run.run_label}:{name} width {values.shape[1]} does not "
                f"match canonical components {expected}"
            )
        blocks[name] = values
        component_audit[name] = list(expected)
    blocks["visual"] = _as_finite_feature_matrix(
        COMMON.load_visual(run), name="visual", n_rows=run.n_tr
    )
    transition_events = blocks["transition_events"]
    if transition_events.shape[1] < 3:
        raise ValueError(
            f"{run.run_label}:transition_events lacks crossed directions"
        )
    transition_cell_sd = {
        name: float(np.std(transition_events[:, index]))
        for index, name in enumerate(("N_to_N", "N_to_A", "A_to_N", "A_to_A"))
    }
    unavailable = [
        name
        for name, spread in transition_cell_sd.items()
        if not np.isfinite(spread) or spread <= 1e-8
    ]
    if unavailable:
        raise ValueError(
            f"{run.run_label}: status-switch interaction RSA requires all "
            f"four origin x destination cells; unavailable={unavailable}"
        )
    if include_ram_control:
        blocks["ram"] = _as_finite_feature_matrix(
            np.load(run.ram_path, allow_pickle=False),
            name="ram",
            n_rows=run.n_tr,
        )
        behavior_sidecar = json.loads(
            Path(run.behavior_metadata_path).read_text(encoding="utf-8")
        )
        expected_shape = behavior_sidecar.get(
            "block_shapes", {}
        ).get("controls_ram")
        if expected_shape != list(blocks["ram"].shape):
            raise ValueError(
                f"RAM cache/sidecar mismatch for {run.run_label}: "
                f"{blocks['ram'].shape} != {expected_shape}"
            )
    metadata = (
        COMMON.load_process_metadata(run)
        if hasattr(COMMON, "load_process_metadata")
        else {}
    )
    audit = {
        "run_label": run.run_label,
        "n_tr": int(run.n_tr),
        "process_path": str(run.process_path),
        "process_feature_version": int(
            getattr(COMMON, "PROCESS_FEATURE_VERSION", -1)
        ),
        "neural_feature_schema_version": int(
            getattr(COMMON, "NEURAL_FEATURE_SCHEMA_VERSION", -1)
        ),
        "component_names": component_audit,
        "source_signature": metadata.get("source_signature", {}),
        "transition_cell_signal_sd": transition_cell_sd,
        "include_ram_control": bool(include_ram_control),
        "visual_features": int(blocks["visual"].shape[1]),
        "ram_features": (
            int(blocks["ram"].shape[1])
            if "ram" in blocks
            else 0
        ),
    }
    return blocks, audit


def add_loro_orthonormal_geometry(
    run_blocks: Mapping[str, Mapping[str, np.ndarray]],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Fit the joint-G whitening transform on other runs only."""
    labels = list(run_blocks)
    if len(labels) < 2:
        raise ValueError("LORO geometry orthonormalization needs >=2 runs")
    result: Dict[str, Dict[str, np.ndarray]] = {}
    for held_out in labels:
        training = np.vstack(
            [
                _required_controlled_block(
                    run_blocks[label], "geometry_primary"
                )[:, :2]
                for label in labels
                if label != held_out
            ]
        ).astype(float)
        mean = np.mean(training, axis=0)
        centered = training - mean
        covariance = centered.T @ centered / max(len(centered) - 1, 1)
        eigenvalue, eigenvector = np.linalg.eigh(covariance)
        floor = max(float(np.max(eigenvalue)) * 1e-6, 1e-8)
        eigenvalue = np.maximum(eigenvalue, floor)
        whitener = eigenvector @ np.diag(1.0 / np.sqrt(eigenvalue)) @ eigenvector.T
        held = dict(run_blocks[held_out])
        held_geometry = _required_controlled_block(
            held, "geometry_primary"
        )[:, :2]
        held["geometry_joint_loro"] = (
            (held_geometry - mean) @ whitener
        ).astype(np.float32)
        result[held_out] = held
    return result


def analyze_controlled_roi_run(
    run: COMMON.BrainRunSpec,
    roi_name: str,
    designs: Mapping[str, Mapping[str, Any]],
    pair_mask: np.ndarray,
    orders: np.ndarray,
    *,
    permutation_batch_size: int,
) -> Tuple[List[Dict[str, object]], Dict[str, np.ndarray]]:
    """Estimate all fixed controlled comparisons in one ROI/run."""
    patterns, _ = COMMON.load_roi(run, roi_name)
    brain_rdm = COMMON.correlation_rdm(patterns)
    ranked_brain = COMMON.rank_zscore(brain_rdm)[pair_mask]
    observed = {
        effect: score_controlled_design(ranked_brain, design)
        for effect, design in designs.items()
    }
    nulls: Dict[str, Dict[str, np.ndarray]] = {
        effect: {
            statistic: np.empty(len(orders), dtype=np.float32)
            for statistic in (
                "delta_r2",
                "added_beta",
                "conditional_partial_r",
            )
        }
        for effect in designs
    }
    if len(orders):
        ranked_square = COMMON.square_from_condensed(
            COMMON.rank_zscore(brain_rdm), run.n_tr
        )
        all_row, all_column = np.triu_indices(run.n_tr, k=1)
        pair_row = all_row[np.asarray(pair_mask, dtype=bool)]
        pair_column = all_column[np.asarray(pair_mask, dtype=bool)]
        batch_size = max(1, int(permutation_batch_size))
        for start in range(0, len(orders), batch_size):
            stop = min(start + batch_size, len(orders))
            batch_orders = np.asarray(orders[start:stop], dtype=int)
            permuted = ranked_square[
                batch_orders[:, pair_row],
                batch_orders[:, pair_column],
            ]
            for effect, design in designs.items():
                scored = score_controlled_design_batch(permuted, design)
                for statistic, values in scored.items():
                    nulls[effect][statistic][start:stop] = values

    rows: List[Dict[str, object]] = []
    stored: Dict[str, np.ndarray] = {}
    for effect in CONTROLLED_EFFECT_ORDER:
        values = observed[effect]
        design = designs[effect]
        row: Dict[str, object] = {
            "run_label": run.run_label,
            "subject": int(run.subject),
            "session": int(run.session),
            "block": int(run.block),
            "game": int(run.game),
            "roi": roi_name,
            "effect": effect,
            "effect_label": CONTROLLED_EFFECT_LABELS[effect],
            "inference_family": controlled_effect_family(effect),
            "n_tr": int(run.n_tr),
            "n_voxels": int(patterns.shape[1]),
            "n_pairs_retained": int(design["n_pairs_retained"]),
            "kept_controls": "|".join(design["kept_controls"]),
            "dropped_alias_controls": "|".join(
                design["dropped_controls"]
            ),
            "target_residual_ratio": float(
                design["target_residual_ratio"]
            ),
            **values,
        }
        for statistic in (
            "delta_r2",
            "added_beta",
            "conditional_partial_r",
        ):
            null = nulls[effect][statistic]
            stored[
                _controlled_null_key(
                    run.run_label, roi_name, effect, statistic
                )
            ] = null
            observed_value = float(values[statistic])
            if len(null):
                if statistic == "delta_r2":
                    exceedance = null >= observed_value
                else:
                    exceedance = np.abs(null) >= abs(observed_value)
                row[f"p_block_{statistic}"] = float(
                    (1 + np.sum(exceedance)) / (1 + len(null))
                )
            else:
                row[f"p_block_{statistic}"] = np.nan
        rows.append(row)
    return rows, stored


def add_controlled_within_run_fwer(
    run_effects: pd.DataFrame,
    null_store: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    """Synchronized family-wise temporal-null correction within each run."""
    result = run_effects.copy()
    statistics = (
        "delta_r2",
        "added_beta",
        "conditional_partial_r",
    )
    for statistic in statistics:
        result[f"p_block_fwer_{statistic}"] = np.nan
    for (run_label, family), group in result.groupby(
        ["run_label", "inference_family"], sort=False
    ):
        for statistic in statistics:
            arrays = []
            indices = []
            for row_index, row in group.iterrows():
                key = _controlled_null_key(
                    str(run_label),
                    str(row["roi"]),
                    str(row["effect"]),
                    statistic,
                )
                if key not in null_store:
                    continue
                arrays.append(np.asarray(null_store[key], dtype=float))
                indices.append(int(row_index))
            if not arrays:
                continue
            lengths = {len(values) for values in arrays}
            if len(lengths) != 1:
                raise ValueError(
                    f"Unsynchronized null lengths in {run_label}/{family}: "
                    f"{lengths}"
                )
            n_draws = next(iter(lengths))
            if n_draws == 0:
                continue
            matrix = np.vstack(arrays)
            if statistic == "delta_r2":
                max_null = np.max(matrix, axis=0)
            else:
                max_null = np.max(np.abs(matrix), axis=0)
            for row_index in indices:
                observed = float(result.at[row_index, statistic])
                threshold = (
                    observed if statistic == "delta_r2" else abs(observed)
                )
                result.at[row_index, f"p_block_fwer_{statistic}"] = (
                    1 + np.sum(max_null >= threshold)
                ) / (1 + n_draws)
    return result


def controlled_exact_summary(
    unit_effects: pd.DataFrame,
    *,
    unit_column: str,
    value_column: str,
    statistic_name: str,
) -> pd.DataFrame:
    """Equal-unit exact inference, synchronized across every ROI x effect."""
    required = {unit_column, "roi", "effect", "inference_family", value_column}
    missing = required.difference(unit_effects.columns)
    if missing:
        raise ValueError(f"Exact summary lacks columns: {sorted(missing)}")
    output: List[pd.DataFrame] = []
    family_orders = {
        "primary_planning": CONTROLLED_PRIMARY_ORDER,
        "primary_transition": CONTROLLED_TRANSITION_ORDER,
        "sensitivity": CONTROLLED_SENSITIVITY_ORDER,
    }
    for family, effect_order in family_orders.items():
        selected = unit_effects[
            unit_effects["inference_family"].astype(str).eq(family)
        ].copy()
        if selected.empty:
            continue
        duplicates = selected.duplicated(
            [unit_column, "roi", "effect"], keep=False
        )
        if np.any(duplicates):
            raise ValueError(
                "Duplicate unit/ROI/effect rows in exact inference: "
                f"{selected.loc[duplicates, [unit_column, 'roi', 'effect']].head().to_dict('records')}"
            )
        rois = sorted(selected["roi"].astype(str).unique())
        columns = [
            (roi, effect)
            for roi in rois
            for effect in effect_order
        ]
        pivot = selected.pivot(
            index=unit_column,
            columns=["roi", "effect"],
            values=value_column,
        )
        missing_columns = [
            key for key in columns if key not in pivot.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Incomplete {family} exact-inference grid: "
                f"{missing_columns[:10]}"
            )
        pivot = pivot.reindex(columns=pd.MultiIndex.from_tuples(columns))
        if pivot.isna().any().any():
            raise ValueError(
                f"Non-finite/missing {family} unit effects for {value_column}"
            )
        names = [f"{roi}\t{effect}" for roi, effect in columns]
        inference = exact_synchronized_signflip(
            pivot.to_numpy(dtype=float), names
        )
        lookup = {
            f"{roi}\t{effect}": (roi, effect)
            for roi, effect in columns
        }
        inference["roi"] = inference["effect"].map(
            lambda name: lookup[str(name)][0]
        )
        inference["model_effect"] = inference["effect"].map(
            lambda name: lookup[str(name)][1]
        )
        inference = inference.drop(columns=["effect"]).rename(
            columns={"model_effect": "effect"}
        )
        inference["effect_label"] = inference["effect"].map(
            CONTROLLED_EFFECT_LABELS
        )
        inference["inference_family"] = family
        inference["statistic"] = statistic_name
        inference["q_bh_family"] = COMMON.fdr_bh(
            inference["p_exact_two_sided"].to_numpy(dtype=float)
        )
        output.append(inference)
    return pd.concat(output, ignore_index=True) if output else pd.DataFrame()


def summarize_controlled_subject(
    run_effects: pd.DataFrame,
) -> pd.DataFrame:
    """Run-equal subject inference; runs are not treated as population units."""
    summaries = []
    for statistic in ("added_beta", "conditional_partial_r"):
        summaries.append(
            controlled_exact_summary(
                run_effects,
                unit_column="run_label",
                value_column=statistic,
                statistic_name=statistic,
            )
        )
    result = pd.concat(summaries, ignore_index=True)
    descriptive = (
        run_effects.groupby(
            ["roi", "effect", "inference_family"], sort=False
        )
        .agg(
            mean_delta_r2=("delta_r2", "mean"),
            median_delta_r2=("delta_r2", "median"),
            n_runs_descriptive=("run_label", "nunique"),
        )
        .reset_index()
    )
    return result.merge(
        descriptive,
        on=["roi", "effect", "inference_family"],
        how="left",
        validate="many_to_one",
    )


def summarize_controlled_group(
    subject_effects: pd.DataFrame,
) -> pd.DataFrame:
    """Subject-equal population inference from subject mean effects only."""
    required = {
        "subject",
        "roi",
        "effect",
        "inference_family",
        "statistic",
        "mean_effect",
    }
    missing = required.difference(subject_effects.columns)
    if missing:
        raise ValueError(f"Group input lacks columns: {sorted(missing)}")
    summaries = []
    for statistic, selected in subject_effects.groupby(
        "statistic", sort=False
    ):
        summaries.append(
            controlled_exact_summary(
                selected,
                unit_column="subject",
                value_column="mean_effect",
                statistic_name=str(statistic),
            )
        )
    return pd.concat(summaries, ignore_index=True)


def plot_controlled_rsa_heatmap(
    summary: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
) -> None:
    selected = summary[
        summary["statistic"].astype(str).eq("conditional_partial_r")
        & summary["inference_family"].astype(str).eq("primary_planning")
    ].copy()
    if selected.empty:
        return
    pivot = selected.pivot(
        index="roi", columns="effect", values="mean_effect"
    )
    columns = [
        effect for effect in CONTROLLED_PRIMARY_ORDER
        if effect in pivot.columns
    ]
    pivot = pivot[columns]
    order = pivot.abs().max(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[order[: min(40, len(order))]]
    values = pivot.to_numpy(dtype=float)
    finite = np.abs(values[np.isfinite(values)])
    vmax = max(
        float(np.quantile(finite, 0.98)) if finite.size else 0.1,
        1e-4,
    )
    fig, ax = plt.subplots(
        figsize=(11.0, max(6.0, 0.24 * len(pivot) + 2.0))
    )
    image = ax.imshow(
        values,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels(
        [CONTROLLED_EFFECT_LABELS[name] for name in columns],
        rotation=28,
        ha="right",
    )
    ax.set_yticks(np.arange(len(pivot)))
    ax.set_yticklabels(
        [clean_roi_label(value) for value in pivot.index],
        fontsize=6.5,
    )
    ax.set_title(title, fontsize=13)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label("Mean conditional partial Spearman r")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path, dpi=240, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)


def synchronized_block_orders(
    n_tr: int,
    n_permutations: int,
    block_size: int,
    seed: int,
) -> np.ndarray:
    """Generate one temporal-null schedule shared by every ROI and effect."""
    if n_permutations < 0:
        raise ValueError("n_permutations must be non-negative")
    if block_size < 2:
        raise ValueError("block_size must be at least two TRs")
    if n_permutations == 0:
        return np.empty((0, int(n_tr)), dtype=np.int32)
    rng = np.random.default_rng(int(seed))
    return np.vstack(
        [
            COMMON.block_permutation_indices(n_tr, block_size, rng)
            for _ in range(int(n_permutations))
        ]
    ).astype(np.int32)


def ranked_permuted_brain_batch(
    brain_rdm: np.ndarray,
    n_tr: int,
    pair_mask: np.ndarray,
    orders: np.ndarray,
) -> np.ndarray:
    """Apply synchronized row/column orders to a globally ranked brain RDM."""
    ranked_square = COMMON.square_from_condensed(
        COMMON.rank_zscore(brain_rdm), n_tr
    )
    all_row, all_column = np.triu_indices(n_tr, k=1)
    mask = np.asarray(pair_mask, dtype=bool)
    pair_row = all_row[mask]
    pair_column = all_column[mask]
    order_array = np.asarray(orders, dtype=int)
    if order_array.ndim != 2 or order_array.shape[1] != n_tr:
        raise ValueError("Permutation order matrix has the wrong TR dimension")
    return ranked_square[
        order_array[:, pair_row], order_array[:, pair_column]
    ]


def _one_sample_t_columns(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    n_units = matrix.shape[0]
    mean = np.mean(matrix, axis=0)
    if n_units < 2:
        result = np.zeros_like(mean)
        result[mean > 0] = np.inf
        result[mean < 0] = -np.inf
        return result
    sd = np.std(matrix, axis=0, ddof=1)
    denominator = sd / np.sqrt(float(n_units))
    result = np.zeros_like(mean)
    finite_scale = denominator > 1e-15
    result[finite_scale] = mean[finite_scale] / denominator[finite_scale]
    result[(~finite_scale) & (mean > 1e-15)] = np.inf
    result[(~finite_scale) & (mean < -1e-15)] = -np.inf
    return result


def exact_synchronized_signflip(
    values: np.ndarray,
    effect_names: Sequence[str],
) -> pd.DataFrame:
    """Exact two-sided raw and synchronized max-absolute-t inference."""
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("Sign-flip input must be unit x effect")
    if matrix.shape[1] != len(effect_names):
        raise ValueError("Sign-flip effect-name count mismatch")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Exact sign-flip requires a complete finite matrix")
    n_units = matrix.shape[0]
    if n_units < 2 or n_units > 20:
        raise ValueError("Exact sign-flip supports 2..20 equal-weight units")
    signs = np.asarray(
        list(itertools.product((-1.0, 1.0), repeat=n_units)),
        dtype=float,
    )
    observed_t = _one_sample_t_columns(matrix)
    null_t = np.empty((len(signs), matrix.shape[1]), dtype=float)
    for index, sign in enumerate(signs):
        null_t[index] = _one_sample_t_columns(matrix * sign[:, None])
    absolute_null = np.abs(null_t)
    max_null = np.max(absolute_null, axis=1)
    rows: List[Dict[str, object]] = []
    for effect_index, name in enumerate(effect_names):
        threshold = abs(float(observed_t[effect_index]))
        rows.append(
            {
                "effect": str(name),
                "n_units": int(n_units),
                "mean_effect": float(np.mean(matrix[:, effect_index])),
                "t_stat": float(observed_t[effect_index]),
                "positive_fraction": float(
                    np.mean(matrix[:, effect_index] > 0)
                ),
                "p_exact_two_sided": float(
                    np.mean(absolute_null[:, effect_index] >= threshold)
                ),
                "p_max_abs_t_fwer": float(
                    np.mean(max_null >= threshold)
                ),
                "n_sign_patterns": int(len(signs)),
            }
        )
    return pd.DataFrame(rows)


COMPARISON_LABELS = {
    "root": "Pre-search Policy State",
    "geometry": "Realized search geometry",
    "evidence_global": "Final ensemble uncertainty",
    "evidence_decomposed": "Within/between-state uncertainty",
    "policy_slack_individual": "Policy Slack",
    "policy_demand_individual": "Policy Demand",
    "branch_diversity_individual": "Root-branch diversity",
    "terminal_depth_individual": "Mean bout terminal depth",
    "deep_allocation_individual": "Deep-node allocation",
    "path_revisit_individual": "Path-revisit fraction",
    "outcome_mean_individual": "Outcome mean shift",
    "outcome_dispersion_individual": "Outcome dispersion",
    "sr_reward_individual": "SR reward trail",
    "vp_bootstrap_individual": "VP bootstrap revision",
    "uncertainty_within_individual": "Within-state uncertainty",
    "uncertainty_between_individual": "Between-state disagreement",
}


def log(message: str) -> None:
    print(f"[07_RSA +{time.time() - T0:8.1f}s] {message}", flush=True)


def build_model_rdms(
    run: COMMON.BrainRunSpec,
    include_secondary_geometry: bool,
    include_individual: bool = False,
) -> Dict[str, np.ndarray]:
    process = COMMON.load_process_blocks(run)
    ram = np.load(run.ram_path, allow_pickle=False)
    visual = COMMON.load_visual(run)
    action = np.column_stack((process["current_action"], process["previous_action"]))
    nuisance = np.column_stack((process["nuisance_continuous"], process["drift"]))
    geometry = process["geometry_primary"]
    if include_secondary_geometry:
        geometry = np.column_stack((geometry, process["geometry_secondary"]))
    rdms = {
        "ram": COMMON.correlation_rdm(ram),
        "visual": COMMON.correlation_rdm(visual),
        "q_state": COMMON.euclidean_rdm(process["q_state"]),
        "action": COMMON.euclidean_rdm(action),
        "nuisance": COMMON.euclidean_rdm(nuisance),
        "temporal_lag": COMMON.temporal_lag_rdm(run.n_tr),
        # The root RDM is defined by the two interpretable coordinates.  The
        # quadratic and interaction columns remain available to encoding.
        "root": COMMON.euclidean_rdm(process["root"][:, :2]),
        "geometry": COMMON.euclidean_rdm(geometry),
        "evidence_global": COMMON.euclidean_rdm(process["evidence_global"]),
        "evidence_decomposed": COMMON.euclidean_rdm(
            process["evidence_decomposed"]
        ),
    }
    if include_individual:
        rdms.update(
            {
                "policy_slack": COMMON.euclidean_rdm(
                    process["root"][:, [0]]
                ),
                "policy_demand": COMMON.euclidean_rdm(
                    process["root"][:, [1]]
                ),
                "branch_diversity": COMMON.euclidean_rdm(
                    process["geometry_primary"][:, [0]]
                ),
                "terminal_depth": COMMON.euclidean_rdm(
                    process["geometry_primary"][:, [1]]
                ),
                "deep_allocation": COMMON.euclidean_rdm(
                    process["geometry_secondary"][:, [0]]
                ),
                "path_revisit": COMMON.euclidean_rdm(
                    process["geometry_secondary"][:, [1]]
                ),
                "uncertainty_within": COMMON.euclidean_rdm(
                    process["evidence_decomposed"][:, [0]]
                ),
                "uncertainty_between": COMMON.euclidean_rdm(
                    process["evidence_decomposed"][:, [1]]
                ),
            }
        )
    return rdms


def comparison_predictors(
    rdms: Mapping[str, np.ndarray],
    individual: bool = False,
) -> Dict[str, Tuple[List[np.ndarray], np.ndarray]]:
    controls = [
        rdms["ram"],
        rdms["visual"],
        rdms["q_state"],
        rdms["action"],
        rdms["nuisance"],
        rdms["temporal_lag"],
    ]
    if individual:
        policy = [rdms["policy_slack"], rdms["policy_demand"]]
        primary_geometry = [
            rdms["branch_diversity"],
            rdms["terminal_depth"],
        ]
        all_geometry = [
            *primary_geometry,
            rdms["deep_allocation"],
            rdms["path_revisit"],
        ]
        policy_adjusted = [*controls, *policy]
        primary_adjusted = [*policy_adjusted, *primary_geometry]
        all_geometry_adjusted = [*policy_adjusted, *all_geometry]
        return {
            "policy_slack_individual": (
                [*controls, rdms["policy_demand"]],
                rdms["policy_slack"],
            ),
            "policy_demand_individual": (
                [*controls, rdms["policy_slack"]],
                rdms["policy_demand"],
            ),
            "branch_diversity_individual": (
                [*policy_adjusted, rdms["terminal_depth"]],
                rdms["branch_diversity"],
            ),
            "terminal_depth_individual": (
                [*policy_adjusted, rdms["branch_diversity"]],
                rdms["terminal_depth"],
            ),
            "deep_allocation_individual": (
                [
                    *primary_adjusted,
                    rdms["path_revisit"],
                ],
                rdms["deep_allocation"],
            ),
            "path_revisit_individual": (
                [
                    *primary_adjusted,
                    rdms["deep_allocation"],
                ],
                rdms["path_revisit"],
            ),
            "uncertainty_within_individual": (
                [*all_geometry_adjusted, rdms["uncertainty_between"]],
                rdms["uncertainty_within"],
            ),
            "uncertainty_between_individual": (
                [*all_geometry_adjusted, rdms["uncertainty_within"]],
                rdms["uncertainty_between"],
            ),
        }
    root_adjusted = [*controls, rdms["root"]]
    geometry_adjusted = [*root_adjusted, rdms["geometry"]]
    return {
        "root": (controls, rdms["root"]),
        "geometry": (root_adjusted, rdms["geometry"]),
        "evidence_global": (geometry_adjusted, rdms["evidence_global"]),
        "evidence_decomposed": (
            geometry_adjusted,
            rdms["evidence_decomposed"],
        ),
    }


def exact_sign_flip_p(values: Sequence[float], alternative: str = "greater") -> float:
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
        signs = rng.choice((-1.0, 1.0), size=(100_000, array.size))
        null = np.mean(signs * array[None, :], axis=1)
    if alternative == "greater":
        return float((1 + np.sum(null >= observed)) / (1 + len(null)))
    return float((1 + np.sum(np.abs(null) >= abs(observed))) / (1 + len(null)))


def residualize_ranked_rdm(
    target: np.ndarray,
    predictors: Sequence[np.ndarray],
) -> np.ndarray:
    y = COMMON.rank_zscore(target)
    columns = [COMMON.rank_zscore(values) for values in predictors]
    design = np.column_stack([np.ones(len(y), dtype=float), *columns])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    return y - design @ beta


def safe_correlation(first: np.ndarray, second: np.ndarray) -> float:
    x = np.asarray(first, dtype=float)
    y = np.asarray(second, dtype=float)
    x = x - np.mean(x)
    y = y - np.mean(y)
    denominator = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    return float(np.sum(x * y) / denominator) if denominator > 0 else 0.0


def prepare_permutation_designs(
    comparisons: Mapping[str, Tuple[Sequence[np.ndarray], np.ndarray]],
    pair_mask: np.ndarray,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Pre-rank and factor the fixed RSA predictor designs once per run.

    The legacy implementation rebuilt and re-ranked the same reduced/full
    designs for every permutation and every ROI.  These designs depend only
    on the model RDMs and the fixed pair mask, so a precomputed pseudoinverse
    is algebraically equivalent and can be reused throughout the run.
    """
    mask = np.asarray(pair_mask, dtype=bool)
    prepared: Dict[str, Dict[str, np.ndarray]] = {}
    for comparison, (reduced_predictors, added_predictor) in comparisons.items():
        reduced_columns = [
            COMMON.rank_zscore(np.asarray(values)[mask])
            for values in reduced_predictors
        ]
        added_column = COMMON.rank_zscore(
            np.asarray(added_predictor)[mask]
        )
        reduced_design = np.column_stack(
            [
                np.ones(int(np.sum(mask)), dtype=float),
                *reduced_columns,
            ]
        )
        full_design = np.column_stack((reduced_design, added_column))
        # Match np.linalg.lstsq(..., rcond=None), whose cutoff is machine
        # precision times the larger matrix dimension.
        rcond = np.finfo(float).eps * max(reduced_design.shape)
        prepared[comparison] = {
            "reduced_design": reduced_design,
            "reduced_pinv": np.linalg.pinv(reduced_design, rcond=rcond),
            "full_design": full_design,
            "full_pinv": np.linalg.pinv(full_design, rcond=rcond),
        }
    return prepared


def fixed_design_r2_batch(
    targets: np.ndarray,
    design: np.ndarray,
    pseudoinverse: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Score multiple permuted targets against one already-factored design."""
    values = np.asarray(targets, dtype=float)
    beta = values @ np.asarray(pseudoinverse, dtype=float).T
    fitted = beta @ np.asarray(design, dtype=float).T
    residual = values - fitted
    denominator = np.sum(
        (values - np.mean(values, axis=1, keepdims=True)) ** 2,
        axis=1,
    )
    numerator = np.sum(residual**2, axis=1)
    r2 = np.divide(
        denominator - numerator,
        denominator,
        out=np.full(len(values), np.nan, dtype=float),
        where=denominator > 1e-12,
    )
    return r2, beta


def safe_correlation_batch(
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    """Row-wise counterpart of ``safe_correlation``."""
    x = np.asarray(first, dtype=float)
    y = np.asarray(second, dtype=float).reshape(-1)
    x = x - np.mean(x, axis=1, keepdims=True)
    y = y - np.mean(y)
    numerator = x @ y
    denominator = np.sqrt(np.sum(x * x, axis=1) * np.sum(y * y))
    return np.divide(
        numerator,
        denominator,
        out=np.zeros(len(x), dtype=float),
        where=denominator > 0,
    )


def analyze_roi_run(
    run: COMMON.BrainRunSpec,
    roi_name: str,
    rdms: Mapping[str, np.ndarray],
    min_pair_lag: int,
    n_permutations: int,
    block_size: int,
    random_state: int,
    permutation_designs: Mapping[str, Mapping[str, np.ndarray]] | None = None,
    comparison_definitions: Mapping[
        str, Tuple[List[np.ndarray], np.ndarray]
    ]
    | None = None,
    permutation_batch_size: int = 20,
) -> Tuple[List[Dict[str, object]], Dict[str, np.ndarray]]:
    patterns, _ = COMMON.load_roi(run, roi_name)
    brain_rdm = COMMON.correlation_rdm(patterns)
    pair_mask = COMMON.pair_mask_min_lag(run.n_tr, min_pair_lag)
    comparisons = (
        dict(comparison_definitions)
        if comparison_definitions is not None
        else comparison_predictors(rdms)
    )
    if permutation_designs is None:
        permutation_designs = prepare_permutation_designs(
            comparisons, pair_mask
        )
    observed: Dict[str, Tuple[float, float, float]] = {}
    partial_observed: Dict[str, float] = {}
    partial_components: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for comparison, (reduced_predictors, added) in comparisons.items():
        observed[comparison] = COMMON.incremental_rdm_fit(
            brain_rdm,
            reduced_predictors,
            added,
            pair_mask=pair_mask,
        )
        neural_residual = residualize_ranked_rdm(
            brain_rdm, reduced_predictors
        )
        added_residual = residualize_ranked_rdm(added, reduced_predictors)
        partial_observed[comparison] = safe_correlation(
            neural_residual[pair_mask], added_residual[pair_mask]
        )
        partial_components[comparison] = (
            COMMON.square_from_condensed(neural_residual, run.n_tr),
            added_residual,
        )

    nulls = {
        comparison: np.empty(n_permutations, dtype=np.float32)
        for comparison in comparisons
    }
    beta_nulls = {
        comparison: np.empty(n_permutations, dtype=np.float32)
        for comparison in comparisons
    }
    partial_nulls = {
        comparison: np.empty(n_permutations, dtype=np.float32)
        for comparison in comparisons
    }
    if n_permutations > 0:
        seed = (
            int(random_state)
            + 101 * int(run.session)
            + 1009 * int(run.block)
        )
        rng = np.random.default_rng(seed)
        all_rows, all_columns = np.triu_indices(run.n_tr, k=1)
        pair_rows = all_rows[pair_mask]
        pair_columns = all_columns[pair_mask]
        # Ranking is invariant to permutation.  Rank the complete neural RDM
        # once, then permute its entries before applying the pair mask.  This
        # exactly matches rank_zscore(permuted_brain)[pair_mask] in the
        # original implementation.
        ranked_brain_square = COMMON.square_from_condensed(
            COMMON.rank_zscore(brain_rdm), run.n_tr
        )
        batch_size = max(1, int(permutation_batch_size))
        for batch_start in range(0, n_permutations, batch_size):
            batch_stop = min(batch_start + batch_size, n_permutations)
            orders = np.vstack(
                [
                    COMMON.block_permutation_indices(
                        run.n_tr, block_size, rng
                    )
                    for _ in range(batch_stop - batch_start)
                ]
            )
            permuted_brain_batch = ranked_brain_square[
                orders[:, pair_rows],
                orders[:, pair_columns],
            ]
            batch_slice = slice(batch_start, batch_stop)
            for comparison in comparisons:
                design = permutation_designs[comparison]
                reduced_r2, _ = fixed_design_r2_batch(
                    permuted_brain_batch,
                    design["reduced_design"],
                    design["reduced_pinv"],
                )
                full_r2, full_beta = fixed_design_r2_batch(
                    permuted_brain_batch,
                    design["full_design"],
                    design["full_pinv"],
                )
                nulls[comparison][batch_slice] = full_r2 - reduced_r2
                beta_nulls[comparison][batch_slice] = full_beta[:, -1]
                residual_square, added_residual = partial_components[comparison]
                permuted_residual_batch = residual_square[
                    orders[:, pair_rows],
                    orders[:, pair_columns],
                ]
                partial_nulls[comparison][batch_slice] = (
                    safe_correlation_batch(
                        permuted_residual_batch,
                    added_residual[pair_mask],
                )
                )

    rows: List[Dict[str, object]] = []
    stored: Dict[str, np.ndarray] = {}
    for comparison, (reduced_r2, full_r2, beta) in observed.items():
        delta = float(full_r2 - reduced_r2)
        delta_null = nulls[comparison]
        beta_null = beta_nulls[comparison]
        partial_r = float(partial_observed[comparison])
        partial_null = partial_nulls[comparison]
        if n_permutations > 0:
            p_delta = float(
                (1 + np.sum(delta_null >= delta)) / (1 + n_permutations)
            )
            p_beta = float(
                (1 + np.sum(beta_null >= beta)) / (1 + n_permutations)
            )
            p_partial = float(
                (1 + np.sum(partial_null >= partial_r))
                / (1 + n_permutations)
            )
        else:
            p_delta = np.nan
            p_beta = np.nan
            p_partial = np.nan
        rows.append(
            {
                "run_label": run.run_label,
                "subject": run.subject,
                "session": run.session,
                "block": run.block,
                "game": run.game,
                "roi": roi_name,
                "comparison": comparison,
                "n_tr": run.n_tr,
                "n_voxels": int(patterns.shape[1]),
                "min_pair_lag_tr": int(min_pair_lag),
                "reduced_r2": reduced_r2,
                "full_r2": full_r2,
                "delta_r2": delta,
                "added_beta": beta,
                "conditional_partial_r": partial_r,
                "p_block_delta_one_sided": p_delta,
                "p_block_beta_one_sided": p_beta,
                "p_block_conditional_partial_one_sided": p_partial,
            }
        )
        key = f"{COMMON.safe_name(run.run_label)}__{COMMON.safe_name(roi_name)}__{comparison}"
        stored[f"{key}__delta_r2"] = delta_null
        stored[f"{key}__beta"] = beta_null
        stored[f"{key}__conditional_partial_r"] = partial_null
    return rows, stored


def add_within_run_max_fwer(
    run_effects: pd.DataFrame,
    null_store: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    """Apply synchronized max-statistic FWER across ROI × comparison."""
    result = run_effects.copy()
    result["p_fwer_delta_one_sided"] = np.nan
    result["p_fwer_beta_one_sided"] = np.nan
    result["p_fwer_conditional_partial_one_sided"] = np.nan
    for run_label, group in result.groupby("run_label", sort=False):
        delta_nulls: List[np.ndarray] = []
        beta_nulls: List[np.ndarray] = []
        partial_nulls: List[np.ndarray] = []
        row_indices: List[int] = []
        for row_index, row in group.iterrows():
            key = (
                f"{COMMON.safe_name(run_label)}__"
                f"{COMMON.safe_name(str(row['roi']))}__{row['comparison']}"
            )
            delta_key = f"{key}__delta_r2"
            beta_key = f"{key}__beta"
            partial_key = f"{key}__conditional_partial_r"
            if (
                delta_key not in null_store
                or beta_key not in null_store
                or partial_key not in null_store
            ):
                continue
            delta_nulls.append(np.asarray(null_store[delta_key], dtype=float))
            beta_nulls.append(np.asarray(null_store[beta_key], dtype=float))
            partial_nulls.append(
                np.asarray(null_store[partial_key], dtype=float)
            )
            row_indices.append(int(row_index))
        if not row_indices:
            continue
        lengths = {
            len(values)
            for values in [*delta_nulls, *beta_nulls, *partial_nulls]
        }
        if len(lengths) != 1:
            raise ValueError(f"Permutation length mismatch within {run_label}: {lengths}")
        n_permutations = next(iter(lengths))
        if n_permutations == 0:
            continue
        max_delta = np.max(np.vstack(delta_nulls), axis=0)
        max_beta = np.max(np.vstack(beta_nulls), axis=0)
        max_partial = np.max(np.vstack(partial_nulls), axis=0)
        for position, row_index in enumerate(row_indices):
            observed_delta = float(result.at[row_index, "delta_r2"])
            observed_beta = float(result.at[row_index, "added_beta"])
            observed_partial = float(
                result.at[row_index, "conditional_partial_r"]
            )
            result.at[row_index, "p_fwer_delta_one_sided"] = (
                1 + np.sum(max_delta >= observed_delta)
            ) / (1 + n_permutations)
            result.at[row_index, "p_fwer_beta_one_sided"] = (
                1 + np.sum(max_beta >= observed_beta)
            ) / (1 + n_permutations)
            result.at[
                row_index, "p_fwer_conditional_partial_one_sided"
            ] = (
                1 + np.sum(max_partial >= observed_partial)
            ) / (1 + n_permutations)
    return result


def aggregate_run_effects(run_effects: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (roi, comparison), group in run_effects.groupby(
        ["roi", "comparison"], sort=False
    ):
        delta = group["delta_r2"].to_numpy(dtype=float)
        beta = group["added_beta"].to_numpy(dtype=float)
        partial = group["conditional_partial_r"].to_numpy(dtype=float)
        rows.append(
            {
                "roi": roi,
                "comparison": comparison,
                "n_runs": int(group["run_label"].nunique()),
                "mean_delta_r2": float(np.nanmean(delta)),
                "median_delta_r2": float(np.nanmedian(delta)),
                # In-sample nested-model ΔR² is non-negative by construction.
                # Its null is the within-run block permutation, not a sign-flip.
                "median_p_block_delta_one_sided": float(
                    np.nanmedian(
                        group["p_block_delta_one_sided"].to_numpy(dtype=float)
                    )
                ),
                "median_p_fwer_delta_one_sided": float(
                    np.nanmedian(
                        group["p_fwer_delta_one_sided"].to_numpy(dtype=float)
                    )
                ),
                "mean_added_beta": float(np.nanmean(beta)),
                "positive_run_fraction_beta": float(np.nanmean(beta > 0)),
                "p_run_signflip_beta_one_sided": exact_sign_flip_p(beta),
                "median_p_fwer_beta_one_sided": float(
                    np.nanmedian(
                        group["p_fwer_beta_one_sided"].to_numpy(dtype=float)
                    )
                ),
                "mean_conditional_partial_r": float(np.nanmean(partial)),
                "positive_run_fraction_conditional_partial": float(
                    np.nanmean(partial > 0)
                ),
                "p_run_signflip_conditional_partial_one_sided": (
                    exact_sign_flip_p(partial)
                ),
                "median_p_fwer_conditional_partial_one_sided": float(
                    np.nanmedian(
                        group[
                            "p_fwer_conditional_partial_one_sided"
                        ].to_numpy(dtype=float)
                    )
                ),
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["q_global_fdr_beta"] = COMMON.fdr_bh(
            result["p_run_signflip_beta_one_sided"]
        )
        result["q_global_fdr_conditional_partial"] = COMMON.fdr_bh(
            result["p_run_signflip_conditional_partial_one_sided"]
        )
    return result


def plot_delta_heatmap(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        return
    pivot = summary.pivot(index="roi", columns="comparison", values="mean_delta_r2")
    order = [name for name in COMPARISON_LABELS if name in pivot.columns]
    pivot = pivot[order]
    score = pivot.abs().max(axis=1)
    pivot = pivot.loc[score.sort_values(ascending=False).index[:30]]
    values = pivot.to_numpy(dtype=float)
    finite = np.abs(values[np.isfinite(values)])
    vmax = float(np.quantile(finite, 0.98)) if finite.size else 1.0
    vmax = max(vmax, 1e-6)
    fig_height = max(6.0, 0.28 * len(pivot) + 2.2)
    fig_width = max(10.5, 3.0 + 0.95 * len(order))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([COMPARISON_LABELS[name] for name in order], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(pivot)))
    ax.set_yticklabels([name.replace("ho_cortical_", "").replace("ho_subcortical_", "") for name in pivot.index], fontsize=7)
    ax.set_title("Planning-process RSA: incremental representational fit", fontsize=13)
    ax.set_xlabel("Full model minus theory-matched reduced model")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label("Across-run mean ΔR²")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_run_consistency(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        return
    selected = (
        summary.sort_values(
            ["q_global_fdr_conditional_partial", "mean_conditional_partial_r"],
            ascending=[True, False],
        )
        .groupby("comparison", sort=False)
        .head(5)
        .copy()
    )
    selected["label"] = (
        selected["comparison"].map(COMPARISON_LABELS)
        + " · "
        + selected["roi"].str.replace(r"^ho_(?:cortical|subcortical)_", "", regex=True)
    )
    selected = selected.sort_values("mean_conditional_partial_r")
    fig, ax = plt.subplots(figsize=(10.5, max(5.5, 0.34 * len(selected) + 1.8)))
    y = np.arange(len(selected))
    ax.scatter(
        selected["mean_conditional_partial_r"], y, color="#31688E", s=30
    )
    ax.axvline(0, color="#333333", linewidth=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(selected["label"], fontsize=7.5)
    ax.set_xlabel("Across-run mean conditional partial Spearman r")
    ax.set_title("Top a priori ROI effects by conditional RSA")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def select_transition_dominant_trs(
    transition_events: np.ndarray,
    *,
    fraction: float,
    min_trs: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Select equal-sized HRF-dominant TR sets for the two transitions."""
    events = np.asarray(transition_events, dtype=float)
    if events.ndim != 2 or events.shape[1] < 2:
        raise ValueError(
            "transition_events must contain NOOP→active and active→NOOP columns"
        )
    n_tr = int(len(events))
    if n_tr < 8:
        raise ValueError("Too few TRs for transition selection")
    score = events[:, 0] - events[:, 1]
    if not np.any(np.isfinite(score)) or float(np.nanstd(score)) <= 1e-12:
        raise ValueError("Transition contrast is constant after HRF alignment")
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    requested = max(int(min_trs), int(round(float(fraction) * n_tr)))
    n_selected = min(requested, n_tr // 2)
    if n_selected < 4:
        raise ValueError("Transition selection retained fewer than four TRs")
    order = np.argsort(score, kind="stable")
    active_to_noop = np.sort(order[:n_selected])
    noop_to_active = np.sort(order[-n_selected:])
    overlap = np.intersect1d(active_to_noop, noop_to_active)
    if overlap.size:
        raise AssertionError("Transition-dominant TR sets overlap")
    selections = {
        "noop_to_active": noop_to_active.astype(np.int32),
        "active_to_noop": active_to_noop.astype(np.int32),
    }
    audit = {
        "n_tr": n_tr,
        "n_selected_per_condition": int(n_selected),
        "selection_fraction_per_condition": float(n_selected / n_tr),
        "noop_to_active_score_mean": float(np.mean(score[noop_to_active])),
        "active_to_noop_score_mean": float(np.mean(score[active_to_noop])),
        "contrast_score_sd": float(np.std(score)),
    }
    return selections, audit


def selected_pair_indices(
    selected_trs: np.ndarray,
    *,
    min_pair_lag: int,
) -> Tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(selected_trs, dtype=int)
    local_row, local_column = np.triu_indices(len(selected), k=1)
    global_row = selected[local_row]
    global_column = selected[local_column]
    keep = (
        np.abs(global_column - global_row) >= max(0, int(min_pair_lag))
    )
    return global_row[keep], global_column[keep]


def prepare_transition_topology_run(
    run: COMMON.BrainRunSpec,
    *,
    transition_fraction: float,
    min_transition_trs: int,
    min_pair_lag: int,
    n_permutations: int,
    permutation_block_size: int,
    random_state: int,
) -> Dict[str, object]:
    """Prepare shared topology RDM vectors and temporal nulls once per run."""
    process = COMMON.load_process_blocks(run)
    if "topology_matrix" not in process:
        raise ValueError(
            f"{run.run_label} lacks topology_matrix; rebuild with "
            "00_brain_cache.py --include-topology-cache --overwrite"
        )
    selections, selection_audit = select_transition_dominant_trs(
        process["transition_events"],
        fraction=transition_fraction,
        min_trs=min_transition_trs,
    )
    topology = np.asarray(process["topology_matrix"], dtype=np.float32)
    topology_rdm = COMMON.correlation_rdm(topology)
    topology_square = COMMON.square_from_condensed(topology_rdm, run.n_tr)
    seed = (
        int(random_state)
        + 1009 * int(run.session)
        + 100_003 * int(run.block)
    )
    rng = np.random.default_rng(seed)
    orders = [
        COMMON.block_permutation_indices(
            run.n_tr, permutation_block_size, rng
        )
        for _ in range(int(n_permutations))
    ]
    conditions: Dict[str, Dict[str, np.ndarray | int | float]] = {}
    for condition in TRANSITION_CONDITIONS:
        selected = selections[condition]
        pair_row, pair_column = selected_pair_indices(
            selected, min_pair_lag=min_pair_lag
        )
        if len(pair_row) < 20:
            raise ValueError(
                f"{run.run_label}:{condition} retained only "
                f"{len(pair_row)} topology pairs"
            )
        observed = topology_square[pair_row, pair_column]
        observed_rank = COMMON.rank_zscore(observed).astype(np.float32)
        null_rank = np.empty(
            (int(n_permutations), len(pair_row)), dtype=np.float32
        )
        for permutation, order in enumerate(orders):
            values = topology_square[
                order[pair_row], order[pair_column]
            ]
            null_rank[permutation] = COMMON.rank_zscore(values)
        conditions[condition] = {
            "selected_trs": selected.astype(np.int32),
            "pair_row": pair_row.astype(np.int32),
            "pair_column": pair_column.astype(np.int32),
            "topology_rank": observed_rank,
            "topology_null_rank": null_rank,
            "n_pairs": int(len(pair_row)),
        }
    return {
        "run": run,
        "conditions": conditions,
        "selection_audit": selection_audit,
        "topology_features": int(topology.shape[1]),
        "permutation_seed": int(seed),
    }


def balanced_accuracy_score(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> float:
    truth_values = np.asarray(truth, dtype=int)
    predicted = np.asarray(prediction, dtype=int)
    recalls = []
    for label in (0, 1):
        mask = truth_values == label
        if np.any(mask):
            recalls.append(float(np.mean(predicted[mask] == label)))
    return float(np.mean(recalls)) if len(recalls) == 2 else np.nan


def nearest_centroid_transition_accuracy(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> float:
    """Parameter-free standardized nearest-centroid MVPA fold."""
    prediction = nearest_centroid_transition_prediction(
        train_x, train_y, test_x
    )
    return balanced_accuracy_score(test_y, prediction)


def nearest_centroid_transition_prediction(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
) -> np.ndarray:
    """Return held-out predictions from the standardized nearest-centroid model."""
    x_train = np.asarray(train_x, dtype=float)
    x_test = np.asarray(test_x, dtype=float)
    y_train = np.asarray(train_y, dtype=int)
    mean = np.mean(x_train, axis=0)
    scale = np.std(x_train, axis=0)
    scale[~np.isfinite(scale) | (scale <= 1e-8)] = 1.0
    x_train = np.nan_to_num((x_train - mean) / scale)
    x_test = np.nan_to_num((x_test - mean) / scale)
    centroids = np.vstack(
        [np.mean(x_train[y_train == label], axis=0) for label in (0, 1)]
    )
    distance = np.column_stack(
        [
            np.mean((x_test - centroid[None, :]) ** 2, axis=1)
            for centroid in centroids
        ]
    )
    return np.argmin(distance, axis=1).astype(np.int8)


def transition_confusion_counts(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> Dict[str, float | int]:
    """Return named 2x2 counts and class recalls for one held-out run."""
    truth_values = np.asarray(truth, dtype=int)
    predicted = np.asarray(prediction, dtype=int)
    counts = np.zeros((2, 2), dtype=int)
    for true_label in (0, 1):
        for predicted_label in (0, 1):
            counts[true_label, predicted_label] = int(
                np.sum(
                    (truth_values == true_label)
                    & (predicted == predicted_label)
                )
            )
    class_total = counts.sum(axis=1)
    recalls = np.divide(
        np.diag(counts),
        class_total,
        out=np.full(2, np.nan, dtype=float),
        where=class_total > 0,
    )
    return {
        "true_active_to_noop_pred_active_to_noop": int(counts[0, 0]),
        "true_active_to_noop_pred_noop_to_active": int(counts[0, 1]),
        "true_noop_to_active_pred_active_to_noop": int(counts[1, 0]),
        "true_noop_to_active_pred_noop_to_active": int(counts[1, 1]),
        "recall_active_to_noop": float(recalls[0]),
        "recall_noop_to_active": float(recalls[1]),
    }


def summarize_transition_topology_rsa(
    run_effects: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: List[Dict[str, object]] = []
    difference_rows: List[Dict[str, object]] = []
    for (roi, condition), group in run_effects.groupby(
        ["roi", "condition"], sort=False
    ):
        values = group["topology_r"].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        sem = (
            float(np.std(finite, ddof=1) / np.sqrt(len(finite)))
            if len(finite) > 1
            else np.nan
        )
        summary_rows.append(
            {
                "roi": roi,
                "condition": condition,
                "n_runs": int(group["run_label"].nunique()),
                "mean_topology_r": float(np.mean(finite)),
                "ci_low": float(np.mean(finite) - 1.96 * sem),
                "ci_high": float(np.mean(finite) + 1.96 * sem),
                "positive_run_fraction": float(np.mean(finite > 0)),
                "p_run_signflip_one_sided": exact_sign_flip_p(finite),
                "median_temporal_permutation_p": float(
                    np.nanmedian(
                        group["p_temporal_permutation_one_sided"].to_numpy(
                            dtype=float
                        )
                    )
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary["q_bh_across_roi_condition"] = COMMON.fdr_bh(
            summary["p_run_signflip_one_sided"].to_numpy(dtype=float)
        )
    pivot = run_effects.pivot_table(
        index=["run_label", "roi"],
        columns="condition",
        values="topology_r",
        aggfunc="first",
    ).reset_index()
    if set(TRANSITION_CONDITIONS).issubset(pivot.columns):
        pivot["difference"] = (
            pivot["noop_to_active"] - pivot["active_to_noop"]
        )
        for roi, group in pivot.groupby("roi", sort=False):
            values = group["difference"].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            sem = (
                float(np.std(finite, ddof=1) / np.sqrt(len(finite)))
                if len(finite) > 1
                else np.nan
            )
            difference_rows.append(
                {
                    "roi": roi,
                    "n_runs": int(group["run_label"].nunique()),
                    "mean_difference": float(np.mean(finite)),
                    "ci_low": float(np.mean(finite) - 1.96 * sem),
                    "ci_high": float(np.mean(finite) + 1.96 * sem),
                    "positive_run_fraction": float(np.mean(finite > 0)),
                    "p_run_signflip_two_sided": exact_sign_flip_p(
                        finite, alternative="two-sided"
                    ),
                }
            )
    differences = pd.DataFrame(difference_rows)
    if not differences.empty:
        differences["q_bh_across_rois"] = COMMON.fdr_bh(
            differences["p_run_signflip_two_sided"].to_numpy(dtype=float)
        )
    return summary, differences


def summarize_transition_mvpa(
    fold_effects: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for roi, group in fold_effects.groupby("roi", sort=False):
        accuracy = group["balanced_accuracy"].to_numpy(dtype=float)
        centered = accuracy - 0.5
        finite = accuracy[np.isfinite(accuracy)]
        sem = (
            float(np.std(finite, ddof=1) / np.sqrt(len(finite)))
            if len(finite) > 1
            else np.nan
        )
        rows.append(
            {
                "roi": roi,
                "n_runs": int(group["held_out_run"].nunique()),
                "mean_balanced_accuracy": float(np.mean(finite)),
                "ci_low": float(np.mean(finite) - 1.96 * sem),
                "ci_high": float(np.mean(finite) + 1.96 * sem),
                "above_chance_run_fraction": float(np.mean(finite > 0.5)),
                "p_run_signflip_above_chance": exact_sign_flip_p(centered),
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["q_bh_across_rois"] = COMMON.fdr_bh(
            result["p_run_signflip_above_chance"].to_numpy(dtype=float)
        )
    return result


def summarize_transition_mvpa_class_recall(
    fold_effects: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize class-specific recall and correct across ROI x transition."""
    condition_columns = {
        "noop_to_active": "recall_noop_to_active",
        "active_to_noop": "recall_active_to_noop",
    }
    rows: List[Dict[str, object]] = []
    for roi, group in fold_effects.groupby("roi", sort=False):
        for condition in TRANSITION_CONDITIONS:
            values = pd.to_numeric(
                group[condition_columns[condition]], errors="coerce"
            ).to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                continue
            sem = (
                float(np.std(finite, ddof=1) / np.sqrt(len(finite)))
                if len(finite) > 1
                else np.nan
            )
            rows.append(
                {
                    "roi": roi,
                    "condition": condition,
                    "condition_label": TRANSITION_LABELS[condition],
                    "n_runs": int(group["held_out_run"].nunique()),
                    "mean_recall": float(np.mean(finite)),
                    "ci_low": float(np.mean(finite) - 1.96 * sem),
                    "ci_high": float(np.mean(finite) + 1.96 * sem),
                    "above_chance_run_fraction": float(
                        np.mean(finite > 0.5)
                    ),
                    "p_run_signflip_above_chance": exact_sign_flip_p(
                        finite - 0.5
                    ),
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["q_bh_across_roi_condition"] = COMMON.fdr_bh(
            result["p_run_signflip_above_chance"].to_numpy(dtype=float)
        )
    return result


def summarize_transition_mvpa_recall_difference(
    fold_effects: pd.DataFrame,
) -> pd.DataFrame:
    """Test paired class-recall differences across held-out runs per ROI."""
    rows: List[Dict[str, object]] = []
    for roi, group in fold_effects.groupby("roi", sort=False):
        noop_to_active = pd.to_numeric(
            group["recall_noop_to_active"], errors="coerce"
        ).to_numpy(dtype=float)
        active_to_noop = pd.to_numeric(
            group["recall_active_to_noop"], errors="coerce"
        ).to_numpy(dtype=float)
        finite = np.isfinite(noop_to_active) & np.isfinite(active_to_noop)
        difference = noop_to_active[finite] - active_to_noop[finite]
        if difference.size == 0:
            continue
        sem = (
            float(
                np.std(difference, ddof=1) / np.sqrt(len(difference))
            )
            if len(difference) > 1
            else np.nan
        )
        rows.append(
            {
                "roi": roi,
                "n_paired_runs": int(len(difference)),
                "mean_recall_noop_to_active": float(
                    np.mean(noop_to_active[finite])
                ),
                "mean_recall_active_to_noop": float(
                    np.mean(active_to_noop[finite])
                ),
                "mean_paired_difference": float(np.mean(difference)),
                "ci_low": float(np.mean(difference) - 1.96 * sem),
                "ci_high": float(np.mean(difference) + 1.96 * sem),
                "p_run_signflip_two_sided": exact_sign_flip_p(
                    difference, alternative="two-sided"
                ),
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["q_bh_across_rois"] = COMMON.fdr_bh(
            result["p_run_signflip_two_sided"].to_numpy(dtype=float)
        )
    return result


def transition_confusion_summary(
    fold_effects: pd.DataFrame,
    mvpa_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build run- and ROI-equalized normalized confusion matrices."""
    significant_rois = set(
        mvpa_summary.loc[
            pd.to_numeric(
                mvpa_summary["q_bh_across_rois"], errors="coerce"
            )
            < 0.05,
            "roi",
        ].astype(str)
    )
    roi_groups = {
        "all_rois": set(fold_effects["roi"].astype(str)),
        "fdr_significant_rois": significant_rois,
    }
    count_columns = {
        (0, 0): "true_active_to_noop_pred_active_to_noop",
        (0, 1): "true_active_to_noop_pred_noop_to_active",
        (1, 0): "true_noop_to_active_pred_active_to_noop",
        (1, 1): "true_noop_to_active_pred_noop_to_active",
    }
    condition_by_label = {
        0: "active_to_noop",
        1: "noop_to_active",
    }
    rows: List[Dict[str, object]] = []
    for group_name, rois in roi_groups.items():
        if not rois:
            continue
        selected = fold_effects[
            fold_effects["roi"].astype(str).isin(rois)
        ]
        cell_values: Dict[Tuple[int, int], List[float]] = {
            key: [] for key in count_columns
        }
        for _, fold in selected.iterrows():
            matrix = np.asarray(
                [
                    [
                        fold[count_columns[(0, 0)]],
                        fold[count_columns[(0, 1)]],
                    ],
                    [
                        fold[count_columns[(1, 0)]],
                        fold[count_columns[(1, 1)]],
                    ],
                ],
                dtype=float,
            )
            totals = matrix.sum(axis=1, keepdims=True)
            normalized = np.divide(
                matrix,
                totals,
                out=np.full_like(matrix, np.nan),
                where=totals > 0,
            )
            for key in count_columns:
                cell_values[key].append(float(normalized[key]))
        for (true_label, predicted_label), values in cell_values.items():
            finite = np.asarray(values, dtype=float)
            finite = finite[np.isfinite(finite)]
            rows.append(
                {
                    "roi_group": group_name,
                    "n_rois": int(len(rois)),
                    "n_runs": int(
                        selected["held_out_run"].nunique()
                    ),
                    "true_condition": condition_by_label[true_label],
                    "true_condition_label": TRANSITION_LABELS[
                        condition_by_label[true_label]
                    ],
                    "predicted_condition": condition_by_label[
                        predicted_label
                    ],
                    "predicted_condition_label": TRANSITION_LABELS[
                        condition_by_label[predicted_label]
                    ],
                    "mean_row_normalized_fraction": float(
                        np.mean(finite)
                    ),
                    "n_roi_run_cells": int(len(finite)),
                }
            )
    return pd.DataFrame(rows)


def clean_roi_label(value: str) -> str:
    return (
        str(value)
        .replace("ho_cortical_", "")
        .replace("ho_subcortical_", "")
    )


def compact_roi_label(value: str) -> str:
    label = clean_roi_label(value).replace("_", " ")
    label = label.replace("left ", "L · ", 1).replace("right ", "R · ", 1)
    label = label.replace("formerly supplementary motor cortex", "SMA")
    return "\n".join(textwrap.wrap(label, width=31))


def significance_stars(q_value: float) -> str:
    if not np.isfinite(q_value):
        return ""
    if q_value < 0.001:
        return "***"
    if q_value < 0.01:
        return "**"
    if q_value < 0.05:
        return "*"
    return ""


def plot_transition_topology_rsa(
    summary: pd.DataFrame,
    differences: pd.DataFrame,
    output_path: Path,
) -> None:
    if summary.empty:
        return
    effect = summary.pivot(
        index="roi", columns="condition", values="mean_topology_r"
    )
    q_values = summary.pivot(
        index="roi",
        columns="condition",
        values="q_bh_across_roi_condition",
    )
    difference = differences.set_index("roi") if not differences.empty else pd.DataFrame()
    for condition in TRANSITION_CONDITIONS:
        if condition not in effect:
            effect[condition] = np.nan
            q_values[condition] = np.nan
    effect["difference"] = (
        difference["mean_difference"].reindex(effect.index)
        if not difference.empty
        else np.nan
    )
    q_values["difference"] = (
        difference["q_bh_across_rois"].reindex(effect.index)
        if not difference.empty
        else np.nan
    )
    columns = [*TRANSITION_CONDITIONS, "difference"]
    effect = effect[columns]
    q_values = q_values[columns]
    order = effect.abs().max(axis=1).sort_values(ascending=False).index
    effect = effect.loc[order]
    q_values = q_values.loc[order]
    values = effect.to_numpy(dtype=float)
    finite = np.abs(values[np.isfinite(values)])
    vmax = float(np.quantile(finite, 0.98)) if finite.size else 0.1
    vmax = max(vmax, 1e-4)
    fig, ax = plt.subplots(
        figsize=(8.2, max(8.0, 0.19 * len(effect) + 2.0))
    )
    image = ax.imshow(
        values,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(
        [
            "NOOP → active",
            "Active → NOOP",
            "Difference",
        ],
        rotation=15,
        ha="right",
    )
    ax.set_yticks(np.arange(len(effect)))
    ax.set_yticklabels(
        [clean_roi_label(name) for name in effect.index], fontsize=6
    )
    for row in range(len(effect)):
        for column in range(3):
            if np.isfinite(q_values.iat[row, column]) and (
                q_values.iat[row, column] < 0.05
            ):
                ax.text(
                    column,
                    row,
                    "*",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black",
                )
    ax.set_title(
        "Search-topology representational similarity by transition",
        loc="center",
        fontsize=13,
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.025)
    colorbar.set_label("Across-run mean Spearman r")
    ax.text(
        1.0,
        -0.045,
        "* BH-FDR q < 0.05",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        color="#59616A",
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_transition_mvpa(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    if summary.empty:
        return
    selected = summary.sort_values(
        "mean_balanced_accuracy", ascending=True
    ).reset_index(drop=True)
    fig, ax = plt.subplots(
        figsize=(8.2, max(8.0, 0.19 * len(selected) + 2.0))
    )
    y = np.arange(len(selected))
    x = selected["mean_balanced_accuracy"].to_numpy(dtype=float)
    low = selected["ci_low"].to_numpy(dtype=float)
    high = selected["ci_high"].to_numpy(dtype=float)
    ax.errorbar(
        x,
        y,
        xerr=np.vstack((x - low, high - x)),
        fmt="o",
        markersize=3.5,
        color="#2F6690",
        ecolor="#9AA4AE",
        elinewidth=0.8,
        capsize=1.5,
    )
    significant = (
        selected["q_bh_across_rois"].to_numpy(dtype=float) < 0.05
    )
    ax.scatter(
        x[significant],
        y[significant],
        s=28,
        facecolor="#E07A2D",
        edgecolor="#20252B",
        linewidth=0.4,
        zorder=3,
    )
    ax.axvline(0.5, color="#30343B", linestyle="--", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [clean_roi_label(name) for name in selected["roi"]], fontsize=6
    )
    ax.set_xlabel("LORO balanced accuracy")
    ax.set_title(
        "NOOP→active versus active→NOOP transition decoding",
        loc="center",
        fontsize=13,
    )
    ax.grid(axis="x", color="#E4E7EB", linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_transition_confusion_matrix(
    confusion: pd.DataFrame,
    output_path: Path,
    *,
    game: int,
) -> None:
    """Plot all-parcel and significant-parcel normalized confusion matrices."""
    if confusion.empty:
        return
    condition_order = ["noop_to_active", "active_to_noop"]
    group_specs = [
        ("all_rois", "All atlas parcels"),
        ("fdr_significant_rois", "FDR-significant parcels"),
    ]
    available = [
        spec
        for spec in group_specs
        if np.any(confusion["roi_group"].astype(str).eq(spec[0]))
    ]
    if not available:
        return
    game_label = "Pong" if int(game) == 1 else "Space Invaders"
    fig, axes = plt.subplots(
        1,
        len(available),
        figsize=(5.2 * len(available), 4.9),
        squeeze=False,
        facecolor="white",
    )
    image = None
    for ax, (group_name, panel_title) in zip(axes[0], available):
        selected = confusion[
            confusion["roi_group"].astype(str).eq(group_name)
        ]
        matrix = (
            selected.pivot(
                index="true_condition",
                columns="predicted_condition",
                values="mean_row_normalized_fraction",
            )
            .reindex(index=condition_order, columns=condition_order)
            .to_numpy(dtype=float)
        )
        image = ax.imshow(
            matrix,
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        for row in range(2):
            for column in range(2):
                value = matrix[row, column]
                ax.text(
                    column,
                    row,
                    f"{100 * value:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=15,
                    color="white" if value >= 0.58 else "#20252B",
                )
        n_rois = int(selected["n_rois"].iloc[0])
        n_runs = int(selected["n_runs"].iloc[0])
        ax.set_title(
            f"{panel_title}\n{n_rois} ROIs · {n_runs} held-out runs",
            fontsize=11,
            pad=10,
        )
        labels = ["NOOP → active", "Active → NOOP"]
        ax.set_xticks([0, 1], labels=labels, rotation=18, ha="right")
        ax.set_yticks([0, 1], labels=labels)
        ax.set_xlabel("Predicted transition")
        ax.set_ylabel("Observed transition")
        for spine in ax.spines.values():
            spine.set_visible(False)
    if image is not None:
        colorbar = fig.colorbar(
            image, ax=axes.ravel().tolist(), fraction=0.032, pad=0.04
        )
        colorbar.set_label("Row-normalized prediction fraction")
    fig.suptitle(
        f"Transition-decoding confusion matrix · {game_label}",
        fontsize=15,
        y=0.98,
    )
    fig.subplots_adjust(
        left=0.10, right=0.90, top=0.82, bottom=0.18, wspace=0.34
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path, dpi=260, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)


def plot_top_transition_roi_recalls(
    fold_effects: pd.DataFrame,
    mvpa_summary: pd.DataFrame,
    recall_difference: pd.DataFrame,
    output_path: Path,
    *,
    game: int,
    n_top_rois: int,
) -> None:
    """Plot transition-specific run recalls in top significant decoding ROIs."""
    if fold_effects.empty or mvpa_summary.empty or recall_difference.empty:
        return
    significant = mvpa_summary[
        pd.to_numeric(
            mvpa_summary["q_bh_across_rois"], errors="coerce"
        )
        < 0.05
    ]
    candidates = significant if not significant.empty else mvpa_summary
    selected_rois = (
        candidates.sort_values(
            "mean_balanced_accuracy", ascending=False
        )
        .head(max(1, int(n_top_rois)))["roi"]
        .astype(str)
        .tolist()
    )
    if not selected_rois:
        return
    recall_columns = {
        "noop_to_active": "recall_noop_to_active",
        "active_to_noop": "recall_active_to_noop",
    }
    conditions = ["noop_to_active", "active_to_noop"]
    selected_folds = fold_effects[
        fold_effects["roi"].astype(str).isin(selected_rois)
    ]
    all_recall = np.concatenate(
        [
            pd.to_numeric(
                selected_folds[column], errors="coerce"
            ).to_numpy(dtype=float)
            for column in recall_columns.values()
        ]
    )
    finite_recall = all_recall[np.isfinite(all_recall)]
    y_min = max(
        0.0,
        min(0.40, float(np.min(finite_recall)) - 0.05),
    )
    y_max = min(
        1.0,
        max(0.72, float(np.max(finite_recall)) + 0.12),
    )
    n_columns = min(4, len(selected_rois))
    n_rows = int(math.ceil(len(selected_rois) / n_columns))
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(3.25 * n_columns, 3.25 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
        facecolor="white",
    )
    difference_lookup = recall_difference.set_index("roi")
    for roi_index, (ax, roi_name) in enumerate(
        zip(axes.ravel(), selected_rois)
    ):
        group = selected_folds[
            selected_folds["roi"].astype(str).eq(roi_name)
        ]
        paired_noop_to_active = pd.to_numeric(
            group["recall_noop_to_active"], errors="coerce"
        ).to_numpy(dtype=float)
        paired_active_to_noop = pd.to_numeric(
            group["recall_active_to_noop"], errors="coerce"
        ).to_numpy(dtype=float)
        paired_finite = np.isfinite(paired_noop_to_active) & np.isfinite(
            paired_active_to_noop
        )
        paired_noop_to_active = paired_noop_to_active[paired_finite]
        paired_active_to_noop = paired_active_to_noop[paired_finite]
        paired_jitter = (
            np.linspace(-0.09, 0.09, len(paired_noop_to_active))
            if len(paired_noop_to_active) > 1
            else np.zeros(len(paired_noop_to_active))
        )
        for run_index in range(len(paired_noop_to_active)):
            ax.plot(
                [
                    paired_jitter[run_index],
                    1 + paired_jitter[run_index],
                ],
                [
                    paired_noop_to_active[run_index],
                    paired_active_to_noop[run_index],
                ],
                color="#C7CCD1",
                linewidth=0.65,
                alpha=0.62,
                zorder=2,
            )
        for x_position, condition in enumerate(conditions):
            values = pd.to_numeric(
                group[recall_columns[condition]], errors="coerce"
            ).to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            mean = float(np.mean(values))
            sem = (
                float(np.std(values, ddof=1) / np.sqrt(len(values)))
                if len(values) > 1
                else np.nan
            )
            ci = 1.96 * sem if np.isfinite(sem) else 0.0
            color = TRANSITION_COLORS[condition]
            ax.bar(
                x_position,
                mean - 0.5,
                bottom=0.5,
                width=0.55,
                color=color,
                alpha=0.18,
                edgecolor=color,
                linewidth=1.0,
                zorder=1,
            )
            jitter = (
                np.linspace(-0.09, 0.09, len(values))
                if len(values) > 1
                else np.zeros(len(values))
            )
            ax.scatter(
                x_position + jitter,
                values,
                s=21,
                facecolor=color,
                edgecolor="white",
                linewidth=0.45,
                alpha=0.80,
                zorder=3,
            )
            ax.errorbar(
                [x_position],
                [mean],
                yerr=[[ci], [ci]],
                fmt="o",
                color="#20252B",
                markerfacecolor="white",
                markersize=5.0,
                elinewidth=1.25,
                capsize=3.0,
                zorder=4,
            )
        try:
            q_value = float(
                difference_lookup.loc[roi_name, "q_bh_across_rois"]
            )
        except (KeyError, TypeError, ValueError):
            q_value = np.nan
        bracket_y = y_max - 0.055
        bracket_height = 0.014
        ax.plot(
            [0, 0, 1, 1],
            [
                bracket_y - bracket_height,
                bracket_y,
                bracket_y,
                bracket_y - bracket_height,
            ],
            color="#20252B",
            linewidth=1.0,
            clip_on=False,
        )
        ax.text(
            0.5,
            bracket_y + 0.008,
            significance_stars(q_value) or "n.s.",
            ha="center",
            va="bottom",
            fontsize=10.5,
            color="#20252B",
        )
        overall = mvpa_summary[
            mvpa_summary["roi"].astype(str).eq(roi_name)
        ].iloc[0]
        overall_stars = significance_stars(
            float(overall["q_bh_across_rois"])
        )
        ax.set_title(
            (
                f"{compact_roi_label(roi_name)}\n"
                f"balanced accuracy = "
                f"{float(overall['mean_balanced_accuracy']):.3f}"
                f"{overall_stars}"
            ),
            fontsize=9.2,
            pad=6,
        )
        ax.axhline(
            0.5, color="#30343B", linestyle="--", linewidth=0.9, zorder=0
        )
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(
            [0, 1],
            labels=["NOOP →\nactive", "Active →\nNOOP"],
        )
        ax.grid(axis="y", color="#E4E7EB", linewidth=0.6)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        if roi_index % n_columns == 0:
            ax.set_ylabel("Held-out transition recall")
    for ax in axes.ravel()[len(selected_rois) :]:
        ax.axis("off")
    game_label = "Pong" if int(game) == 1 else "Space Invaders"
    fig.suptitle(
        f"Transition-specific decoding in top ROIs · {game_label}",
        fontsize=15,
        y=0.995,
    )
    fig.text(
        0.5,
        0.012,
        (
            "Dots = held-out runs; bars = deviation from 50% chance; "
            "mean ± 95% CI; title stars = balanced accuracy above chance; "
            "bracket = paired recall difference; * q<.05, ** q<.01, "
            "*** q<.001 (BH-FDR across 110 ROIs)"
        ),
        ha="center",
        va="bottom",
        fontsize=8.2,
        color="#59616A",
    )
    fig.tight_layout(rect=(0.02, 0.055, 0.98, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path, dpi=260, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)


def plot_transition_mvpa_parcel_map(
    summary: pd.DataFrame,
    atlas_root: Path,
    roi_mask_paths: Mapping[str, str],
    figure_path: Path,
    map_dir: Path,
    *,
    game: int,
) -> None:
    """Create a thresholded atlas-parcel decoding map and NIfTI volumes."""
    if summary.empty:
        return
    import nibabel as nib

    summary_lookup = summary.set_index("roi")
    reference_img = None
    shape: Optional[Tuple[int, int, int]] = None
    effect_sum = None
    effect_count = None
    significant_mask = None
    brain_mask = None
    for roi_name, relative_path in roi_mask_paths.items():
        if roi_name not in summary_lookup.index:
            continue
        mask_path = Path(atlas_root) / relative_path
        if not mask_path.exists():
            continue
        image = nib.load(str(mask_path))
        if reference_img is None:
            reference_img = image
            shape = tuple(int(value) for value in image.shape[:3])
            effect_sum = np.zeros(shape, dtype=np.float32)
            effect_count = np.zeros(shape, dtype=np.int16)
            significant_mask = np.zeros(shape, dtype=bool)
            brain_mask = np.zeros(shape, dtype=bool)
        mask = np.asarray(image.dataobj) > 0
        if mask.shape != shape:
            raise ValueError(
                f"Atlas mask shape mismatch for {roi_name}: "
                f"{mask.shape} != {shape}"
            )
        row = summary_lookup.loc[roi_name]
        effect = float(row["mean_balanced_accuracy"]) - 0.5
        q_value = float(row["q_bh_across_rois"])
        brain_mask |= mask
        if np.isfinite(effect):
            effect_sum[mask] += effect
            effect_count[mask] += 1
        if np.isfinite(q_value) and q_value < 0.05 and effect > 0:
            significant_mask[mask] = True
    if (
        reference_img is None
        or effect_sum is None
        or effect_count is None
        or significant_mask is None
        or brain_mask is None
    ):
        return
    effect_map = np.zeros_like(effect_sum, dtype=np.float32)
    valid = effect_count > 0
    effect_map[valid] = effect_sum[valid] / effect_count[valid]
    map_dir.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(
            effect_map, reference_img.affine, reference_img.header
        ),
        str(map_dir / "transition_mvpa_parcel_accuracy_above_chance.nii.gz"),
    )
    nib.save(
        nib.Nifti1Image(
            significant_mask.astype(np.uint8),
            reference_img.affine,
            reference_img.header,
        ),
        str(map_dir / "transition_mvpa_parcel_q05_mask.nii.gz"),
    )

    shown = np.where(significant_mask, effect_map, 0.0)
    finite = shown[np.isfinite(shown) & (shown > 0)]
    vmax = max(
        0.15,
        float(math.ceil(20 * float(np.max(finite))) / 20)
        if finite.size
        else 0.15,
    )
    views = ((0, "Sagittal projection"), (1, "Coronal projection"), (2, "Axial projection"))
    fig, axes = plt.subplots(
        1, 3, figsize=(12.8, 4.5), facecolor="white"
    )
    image = None
    for ax, (axis, view_label) in zip(axes, views):
        base = np.any(brain_mask, axis=axis).T
        projection = np.max(shown, axis=axis).T * 100.0
        ax.imshow(
            base,
            cmap="Greys",
            origin="lower",
            interpolation="nearest",
            alpha=0.24,
        )
        image = ax.imshow(
            np.ma.masked_where(projection <= 0, projection),
            cmap="magma",
            origin="lower",
            interpolation="nearest",
            vmin=0.0,
            vmax=100.0 * vmax,
        )
        ax.set_title(view_label, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    if image is not None:
        colorbar = fig.colorbar(
            image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.025
        )
        colorbar.set_label(
            "Balanced accuracy above chance (percentage points)"
        )
    n_significant = int(
        np.sum(
            pd.to_numeric(
                summary["q_bh_across_rois"], errors="coerce"
            )
            < 0.05
        )
    )
    game_label = "Pong" if int(game) == 1 else "Space Invaders"
    fig.suptitle(
        f"Parcel-wise transition decoding · {game_label}",
        fontsize=15,
        y=0.98,
    )
    fig.text(
        0.5,
        0.025,
        (
            f"Colored parcels: {n_significant}/{len(summary)} with "
            "BH-FDR q<.05; grey: non-significant atlas support"
        ),
        ha="center",
        va="bottom",
        fontsize=9,
        color="#59616A",
    )
    fig.subplots_adjust(
        left=0.02, right=0.92, top=0.84, bottom=0.10, wspace=0.03
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        figure_path, dpi=260, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)


def transition_contrast_permutation_weights(
    transition_events: np.ndarray,
    *,
    fraction: float,
    min_trs: int,
    n_permutations: int,
    block_size: int,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Build observed and block-permuted transition-contrast weights.

    Row zero is the observed NOOP-to-active minus active-to-NOOP contrast.
    Subsequent rows move the complete HRF-aligned transition-label stream in
    40-TR blocks.  This preserves the number of selected TRs and the temporal
    clustering within each block while breaking its alignment to BOLD.
    """
    events = np.asarray(transition_events, dtype=float)
    selections, audit = select_transition_dominant_trs(
        events,
        fraction=fraction,
        min_trs=min_trs,
    )
    noop_to_active = np.asarray(
        selections["noop_to_active"], dtype=int
    )
    active_to_noop = np.asarray(
        selections["active_to_noop"], dtype=int
    )
    n_tr = int(len(events))
    n_selected = int(len(noop_to_active))
    if n_selected != len(active_to_noop):
        raise AssertionError("Transition contrast must use balanced conditions")
    weights = np.zeros(
        (int(n_permutations) + 1, n_tr), dtype=np.float32
    )
    scale = np.float32(1.0 / n_selected)
    weights[0, noop_to_active] = scale
    weights[0, active_to_noop] = -scale

    rng = np.random.default_rng(int(seed))
    identity = np.arange(n_tr, dtype=int)
    for permutation in range(int(n_permutations)):
        order = COMMON.block_permutation_indices(n_tr, block_size, rng)
        inverse = np.empty(n_tr, dtype=int)
        inverse[order] = identity
        weights[permutation + 1, inverse[noop_to_active]] = scale
        weights[permutation + 1, inverse[active_to_noop]] = -scale
    audit = {
        **audit,
        "permutation_seed": int(seed),
        "permutation_block_size_tr": int(block_size),
        "n_permutations": int(n_permutations),
    }
    return weights, audit


def common_union_roi_layout(
    run_specs: Sequence[COMMON.BrainRunSpec],
    roi_names: Sequence[str],
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Create one union-voxel layout and ROI index lookup.

    ROI coordinates are required to be identical across runs so that the
    cross-run dot product compares the same voxel with itself.
    """
    if not run_specs:
        raise ValueError("No runs available for union ROI layout")
    first_run = run_specs[0]
    coords_by_roi: Dict[str, np.ndarray] = {}
    coordinate_blocks: List[np.ndarray] = []
    for roi_name in roi_names:
        _, coords = COMMON.load_roi(first_run, roi_name)
        coords = np.asarray(coords, dtype=np.int32)
        coords_by_roi[roi_name] = coords
        coordinate_blocks.append(coords)
    union_coords = np.unique(np.vstack(coordinate_blocks), axis=0)
    coordinate_lookup = {
        tuple(int(value) for value in coordinate): index
        for index, coordinate in enumerate(union_coords)
    }
    roi_indices: Dict[str, np.ndarray] = {}
    for roi_name, coords in coords_by_roi.items():
        roi_indices[roi_name] = np.asarray(
            [
                coordinate_lookup[tuple(int(value) for value in coordinate)]
                for coordinate in coords
            ],
            dtype=np.int32,
        )
    return union_coords, roi_indices, coords_by_roi


def load_union_roi_patterns(
    run: COMMON.BrainRunSpec,
    roi_names: Sequence[str],
    union_coords: np.ndarray,
    roi_indices: Mapping[str, np.ndarray],
    expected_coords: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Assemble cached ROI patterns into one non-duplicated voxel matrix."""
    patterns = np.empty(
        (run.n_tr, len(union_coords)), dtype=np.float32
    )
    assigned = np.zeros(len(union_coords), dtype=bool)
    for roi_name in roi_names:
        roi_patterns, coords = COMMON.load_roi(run, roi_name)
        coords = np.asarray(coords, dtype=np.int32)
        if not np.array_equal(coords, expected_coords[roi_name]):
            raise ValueError(
                f"Voxel-coordinate mismatch across runs: "
                f"{run.run_label}:{roi_name}"
            )
        indices = np.asarray(roi_indices[roi_name], dtype=int)
        overlap = assigned[indices]
        if np.any(overlap):
            if not np.allclose(
                patterns[:, indices[overlap]],
                roi_patterns[:, overlap],
                atol=1e-6,
                rtol=1e-6,
            ):
                raise ValueError(
                    f"Overlapping ROI voxels disagree: "
                    f"{run.run_label}:{roi_name}"
                )
        new = ~overlap
        patterns[:, indices[new]] = roi_patterns[:, new]
        assigned[indices[new]] = True
    if not np.all(assigned):
        raise AssertionError("Union ROI pattern matrix has unassigned voxels")
    return patterns


def summarize_transition_pattern_distance(
    roi_names: Sequence[str],
    roi_indices: Mapping[str, np.ndarray],
    voxel_distances: np.ndarray,
    *,
    n_runs: int,
    n_permutations: int,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Summarize cross-validated pattern distances and permutation nulls."""
    distances = np.empty(
        (int(n_permutations) + 1, len(roi_names)), dtype=np.float64
    )
    for roi_index, roi_name in enumerate(roi_names):
        indices = np.asarray(roi_indices[roi_name], dtype=int)
        distances[:, roi_index] = np.mean(
            voxel_distances[:, indices], axis=1
        )
    observed = distances[0]
    null = distances[1:]
    max_null = np.max(null, axis=1)
    p_uncorrected = (
        1.0 + np.sum(null >= observed[None, :], axis=0)
    ) / (1.0 + int(n_permutations))
    p_fwer = (
        1.0 + np.sum(max_null[:, None] >= observed[None, :], axis=0)
    ) / (1.0 + int(n_permutations))
    q_bh = COMMON.fdr_bh(p_uncorrected)
    rows: List[Dict[str, object]] = []
    for index, roi_name in enumerate(roi_names):
        null_values = null[:, index]
        null_sd = float(np.std(null_values, ddof=1))
        rows.append(
            {
                "roi": roi_name,
                "n_voxels": int(len(roi_indices[roi_name])),
                "n_runs": int(n_runs),
                "n_run_pairs": int(n_runs * (n_runs - 1) // 2),
                "n_permutations": int(n_permutations),
                "observed_crossvalidated_distance": float(observed[index]),
                "null_mean": float(np.mean(null_values)),
                "null_sd": null_sd,
                "null_q025": float(np.quantile(null_values, 0.025)),
                "null_q975": float(np.quantile(null_values, 0.975)),
                "z_against_null": (
                    float(
                        (observed[index] - np.mean(null_values)) / null_sd
                    )
                    if null_sd > 0
                    else np.nan
                ),
                "p_block_permutation_one_sided": float(
                    p_uncorrected[index]
                ),
                "q_bh_across_rois": float(q_bh[index]),
                "p_fwer_max_stat": float(p_fwer[index]),
            }
        )
    return pd.DataFrame(rows), null.astype(np.float32), max_null.astype(
        np.float32
    )


def plot_transition_pattern_distance(
    summary: pd.DataFrame,
    output_path: Path,
    *,
    game: int,
) -> None:
    """Plot observed ROI distances against each ROI's permutation interval."""
    if summary.empty:
        return
    shown = summary.sort_values(
        "observed_crossvalidated_distance", kind="stable"
    ).reset_index(drop=True)
    y = np.arange(len(shown))
    significant = pd.to_numeric(
        shown["q_bh_across_rois"], errors="coerce"
    ).to_numpy(dtype=float) < 0.05
    fwer = pd.to_numeric(
        shown["p_fwer_max_stat"], errors="coerce"
    ).to_numpy(dtype=float) < 0.05
    colors = np.where(fwer, "#B2182B", np.where(significant, "#D6604D", "#31688E"))
    fig, ax = plt.subplots(
        figsize=(11.2, max(14.0, 0.23 * len(shown) + 2.6)),
        facecolor="white",
    )
    ax.hlines(
        y,
        shown["null_q025"],
        shown["null_q975"],
        color="#B8C1C9",
        linewidth=1.2,
        zorder=1,
    )
    ax.scatter(
        shown["observed_crossvalidated_distance"],
        y,
        c=colors,
        s=24,
        zorder=2,
    )
    ax.axvline(0.0, color="#333333", linewidth=0.9, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(
        [clean_roi_label(value) for value in shown["roi"]],
        fontsize=7.3,
    )
    ax.set_xlabel(
        "Cross-validated NOOP→active versus active→NOOP pattern distance"
    )
    ax.grid(axis="x", alpha=0.18)
    game_label = "Pong" if int(game) == 1 else "Space Invaders"
    ax.set_title(
        f"Transition-related multivoxel pattern separation · {game_label}",
        fontsize=15,
        pad=14,
        loc="center",
    )
    ax.text(
        0.5,
        1.002,
        (
            "Lines: ROI-specific 95% block-permutation null interval; "
            "orange: BH-FDR q<.05; red: max-stat FWER p<.05"
        ),
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        color="#59616A",
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path, dpi=240, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)


def plot_transition_pattern_nulls(
    summary: pd.DataFrame,
    null: np.ndarray,
    max_null: np.ndarray,
    output_path: Path,
    *,
    game: int,
    n_top: int = 8,
) -> None:
    """Show top ROI nulls plus the synchronized max-statistic null."""
    if summary.empty or null.size == 0:
        return
    ranked = summary.sort_values(
        [
            "p_block_permutation_one_sided",
            "observed_crossvalidated_distance",
        ],
        ascending=[True, False],
        kind="stable",
    ).head(max(1, int(n_top)))
    index_lookup = {
        roi_name: index
        for index, roi_name in enumerate(summary["roi"].astype(str))
    }
    fig, axes = plt.subplots(
        3, 3, figsize=(13.4, 10.7), facecolor="white"
    )
    flat_axes = axes.ravel()
    for ax, (_, row) in zip(flat_axes, ranked.iterrows()):
        roi_name = str(row["roi"])
        values = null[:, index_lookup[roi_name]]
        ax.hist(
            values,
            bins=38,
            color="#8FB3C9",
            edgecolor="white",
            linewidth=0.35,
        )
        ax.axvline(
            float(row["observed_crossvalidated_distance"]),
            color="#C53B2C",
            linewidth=2.0,
        )
        ax.axvline(0.0, color="#333333", linewidth=0.8, linestyle="--")
        ax.set_title(compact_roi_label(roi_name), fontsize=9.5)
        ax.text(
            0.98,
            0.94,
            (
                f"p={float(row['p_block_permutation_one_sided']):.4f}\n"
                f"q={float(row['q_bh_across_rois']):.4f}"
            ),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )
        ax.grid(axis="y", alpha=0.15)
    max_ax = flat_axes[-1]
    max_ax.hist(
        max_null,
        bins=38,
        color="#B7B7B7",
        edgecolor="white",
        linewidth=0.35,
    )
    max_observed = float(
        summary["observed_crossvalidated_distance"].max()
    )
    max_ax.axvline(max_observed, color="#7F0000", linewidth=2.0)
    max_ax.set_title("Across-ROI max-statistic null", fontsize=9.5)
    max_ax.text(
        0.98,
        0.94,
        (
            "red = largest observed ROI distance\n"
            "null = maximum across all 110 ROIs"
        ),
        transform=max_ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )
    for ax in flat_axes[len(ranked) : -1]:
        ax.axis("off")
    for ax in flat_axes:
        if ax.axison:
            ax.set_xlabel("Cross-validated pattern distance", fontsize=8)
            ax.set_ylabel("Permutation count", fontsize=8)
    game_label = "Pong" if int(game) == 1 else "Space Invaders"
    fig.suptitle(
        f"Block-permutation null diagnostics · {game_label}",
        fontsize=15,
        y=0.99,
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.965))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path, dpi=240, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)


def transition_discriminant_projection_scores(
    roi_names: Sequence[str],
    roi_indices: Mapping[str, np.ndarray],
    condition_means: Mapping[str, Sequence[np.ndarray]],
    run_specs: Sequence[COMMON.BrainRunSpec],
) -> pd.DataFrame:
    """Project held-out condition patterns onto a training-run contrast axis.

    The discriminant direction and midpoint are estimated without the held-out
    run.  The resulting scores therefore provide a readable two-bar summary of
    the multivoxel contrast without using in-sample condition means.
    """
    noop_to_active_all = np.stack(
        condition_means["noop_to_active"], axis=0
    ).astype(np.float64)
    active_to_noop_all = np.stack(
        condition_means["active_to_noop"], axis=0
    ).astype(np.float64)
    if not (
        len(noop_to_active_all)
        == len(active_to_noop_all)
        == len(run_specs)
    ):
        raise ValueError("Condition-mean/run count mismatch")

    rows: List[Dict[str, object]] = []
    all_runs = np.arange(len(run_specs), dtype=int)
    for roi_name in roi_names:
        indices = np.asarray(roi_indices[roi_name], dtype=int)
        noop_to_active = noop_to_active_all[:, indices]
        active_to_noop = active_to_noop_all[:, indices]
        for held_out in all_runs:
            train = all_runs != held_out
            train_noop_to_active = np.mean(
                noop_to_active[train], axis=0
            )
            train_active_to_noop = np.mean(
                active_to_noop[train], axis=0
            )
            direction = train_noop_to_active - train_active_to_noop
            direction_scale = float(np.sqrt(np.mean(direction**2)))
            if not np.isfinite(direction_scale) or direction_scale <= 1e-12:
                continue
            direction /= direction_scale
            midpoint = (
                train_noop_to_active + train_active_to_noop
            ) / 2.0
            for condition, held_pattern in (
                ("noop_to_active", noop_to_active[held_out]),
                ("active_to_noop", active_to_noop[held_out]),
            ):
                score = float(
                    np.mean((held_pattern - midpoint) * direction)
                )
                rows.append(
                    {
                        "roi": roi_name,
                        "held_out_run": run_specs[held_out].run_label,
                        "condition": condition,
                        "condition_label": TRANSITION_LABELS[condition],
                        "crossvalidated_discriminant_score": score,
                    }
                )
    return pd.DataFrame(rows)


def plot_transition_discriminant_bars(
    scores: pd.DataFrame,
    summary: pd.DataFrame,
    output_path: Path,
    *,
    game: int,
    roi_order: Sequence[str] | None = None,
    significance_column: str = "p_fwer_max_stat",
    significance_display: str = "FWER p",
    correction_description: str = "max-stat FWER",
    figure_title: str | None = None,
    n_columns_override: int | None = None,
) -> None:
    """Bar small multiples for selected transition-pattern ROIs."""
    if scores.empty or summary.empty:
        return
    if significance_column not in summary:
        raise ValueError(
            f"Missing significance column: {significance_column}"
        )
    if roi_order is None:
        significant = summary[
            (
                pd.to_numeric(
                    summary[significance_column], errors="coerce"
                )
                < 0.05
            )
            & (
                pd.to_numeric(
                    summary["observed_crossvalidated_distance"],
                    errors="coerce",
                )
                > 0
            )
        ].sort_values(
            "observed_crossvalidated_distance",
            ascending=False,
            kind="stable",
        )
    else:
        lookup = summary.set_index("roi", drop=False)
        missing = [
            roi_name for roi_name in roi_order
            if roi_name not in lookup.index
        ]
        if missing:
            raise ValueError(
                f"Requested ROI rows are missing: {missing}"
            )
        significant = lookup.loc[list(roi_order)].reset_index(drop=True)
    if significant.empty:
        return

    n_panels = len(significant)
    n_columns = min(
        int(n_columns_override) if n_columns_override else 5,
        n_panels,
    )
    n_rows = int(math.ceil(n_panels / n_columns))
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(3.25 * n_columns, 3.15 * n_rows + 0.65),
        squeeze=False,
        facecolor="white",
    )
    condition_order = ("noop_to_active", "active_to_noop")
    colors = ("#2F6690", "#E07A2D")
    rng = np.random.default_rng(20260731 + int(game))

    for ax, (_, effect_row) in zip(axes.ravel(), significant.iterrows()):
        roi_name = str(effect_row["roi"])
        roi_scores = scores[
            scores["roi"].astype(str).eq(roi_name)
        ].copy()
        pivot = roi_scores.pivot(
            index="held_out_run",
            columns="condition",
            values="crossvalidated_discriminant_score",
        ).dropna(subset=list(condition_order))
        values = [
            pivot[condition].to_numpy(dtype=float)
            for condition in condition_order
        ]
        means = np.asarray(
            [float(np.mean(value)) for value in values], dtype=float
        )
        errors = []
        for value in values:
            if len(value) > 1:
                critical = float(
                    student_t.ppf(0.975, df=len(value) - 1)
                )
                errors.append(
                    critical
                    * float(np.std(value, ddof=1))
                    / math.sqrt(len(value))
                )
            else:
                errors.append(np.nan)
        x = np.arange(2, dtype=float)
        ax.bar(
            x,
            means,
            width=0.62,
            color=colors,
            alpha=0.78,
            edgecolor="#404040",
            linewidth=0.8,
            zorder=2,
        )
        ax.errorbar(
            x,
            means,
            yerr=np.asarray(errors),
            fmt="none",
            ecolor="#202020",
            elinewidth=1.2,
            capsize=4,
            capthick=1.2,
            zorder=4,
        )
        jitter = rng.uniform(-0.075, 0.075, size=len(pivot))
        for run_index in range(len(pivot)):
            paired_x = x + jitter[run_index]
            paired_y = np.asarray(
                [values[0][run_index], values[1][run_index]], dtype=float
            )
            ax.plot(
                paired_x,
                paired_y,
                color="#8B8B8B",
                linewidth=0.65,
                alpha=0.50,
                zorder=1,
            )
            ax.scatter(
                paired_x,
                paired_y,
                s=17,
                facecolor="white",
                edgecolor="#3D3D3D",
                linewidth=0.65,
                zorder=5,
            )
        finite_values = np.concatenate(values)
        value_min = float(np.min(finite_values))
        value_max = float(np.max(finite_values))
        span = max(value_max - value_min, 0.05)
        bracket_y = value_max + 0.13 * span
        bracket_height = 0.045 * span
        ax.plot(
            [0, 0, 1, 1],
            [
                bracket_y,
                bracket_y + bracket_height,
                bracket_y + bracket_height,
                bracket_y,
            ],
            color="#202020",
            linewidth=1.0,
            clip_on=False,
        )
        significance_value = float(effect_row[significance_column])
        ax.text(
            0.5,
            bracket_y + 1.35 * bracket_height,
            significance_stars(significance_value),
            ha="center",
            va="bottom",
            fontsize=14,
        )
        lower = min(value_min, 0.0) - 0.12 * span
        upper = bracket_y + 0.32 * span
        ax.set_ylim(lower, upper)
        ax.axhline(0.0, color="#777777", linewidth=0.7, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(
            ["NOOP→active", "Active→NOOP"], fontsize=8.3
        )
        ax.set_title(compact_roi_label(roi_name), fontsize=9.4, pad=7)
        ax.text(
            0.98,
            0.03,
            f"{significance_display}={significance_value:.4f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.4,
            color="#555555",
        )
        ax.grid(axis="y", alpha=0.16)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes.ravel()[n_panels:]:
        ax.axis("off")
    for row_index in range(n_rows):
        axes[row_index, 0].set_ylabel(
            "Held-out discriminant score", fontsize=9
        )
    game_label = "Pong" if int(game) == 1 else "Space Invaders"
    fig.suptitle(
        figure_title
        or (
            "Cross-validated neural patterns for NOOP→active and "
            f"active→NOOP · {game_label}"
        ),
        fontsize=15,
        y=0.995,
    )
    fig.text(
        0.5,
        0.012,
        (
            "Bars: run-equalized mean; error bars: 95% CI across held-out "
            f"runs; paired points: runs; stars: {correction_description} "
            f"(* {'q' if 'q' in significance_display.lower() else 'p'}<.05, "
            f"** {'q' if 'q' in significance_display.lower() else 'p'}<.01, "
            f"*** {'q' if 'q' in significance_display.lower() else 'p'}<.001)"
        ),
        ha="center",
        va="bottom",
        fontsize=8.7,
        color="#555555",
    )
    fig.tight_layout(rect=(0.02, 0.045, 0.98, 0.955))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path, dpi=260, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)


def generate_cross_game_common_fdr_bars(
    output_root: Path,
    subject: int,
) -> List[str]:
    """Plot the same BH-FDR-significant ROI conjunction for both games."""
    game_data: Dict[int, Tuple[pd.DataFrame, pd.DataFrame]] = {}
    for game in (1, 2):
        result_dir = (
            Path(output_root)
            / f"sub{int(subject):03d}_game{game}"
            / "transition_pattern_permutation"
            / "results"
        )
        summary_path = (
            result_dir / "transition_pattern_distance_summary.csv"
        )
        scores_path = (
            result_dir / "transition_pattern_discriminant_scores.csv"
        )
        if not summary_path.exists() or not scores_path.exists():
            return []
        game_data[game] = (
            pd.read_csv(summary_path),
            pd.read_csv(scores_path),
        )

    significant_sets = []
    for game in (1, 2):
        summary = game_data[game][0]
        significant_sets.append(
            set(
                summary.loc[
                    (
                        pd.to_numeric(
                            summary["q_bh_across_rois"],
                            errors="coerce",
                        )
                        < 0.05
                    )
                    & (
                        pd.to_numeric(
                            summary[
                                "observed_crossvalidated_distance"
                            ],
                            errors="coerce",
                        )
                        > 0
                    ),
                    "roi",
                ].astype(str)
            )
        )
    common = significant_sets[0].intersection(significant_sets[1])
    if not common:
        return []

    joint_rows: List[Dict[str, object]] = []
    lookups = {
        game: game_data[game][0].set_index("roi") for game in (1, 2)
    }
    for roi_name in common:
        row = {"roi": roi_name}
        for game in (1, 2):
            row[f"game{game}_distance"] = float(
                lookups[game].loc[
                    roi_name, "observed_crossvalidated_distance"
                ]
            )
            row[f"game{game}_q_bh"] = float(
                lookups[game].loc[roi_name, "q_bh_across_rois"]
            )
        row["weaker_game_q"] = max(
            float(row["game1_q_bh"]),
            float(row["game2_q_bh"]),
        )
        row["mean_distance"] = np.mean(
            [row["game1_distance"], row["game2_distance"]]
        )
        joint_rows.append(row)
    joint = pd.DataFrame(joint_rows).sort_values(
        ["weaker_game_q", "mean_distance"],
        ascending=[True, False],
        kind="stable",
    )
    roi_order = joint["roi"].astype(str).tolist()

    cross_game_result_dir = (
        Path(output_root)
        / f"sub{int(subject):03d}_cross_game"
        / "transition_pattern_permutation"
        / "results"
    )
    cross_game_result_dir.mkdir(parents=True, exist_ok=True)
    joint.to_csv(
        cross_game_result_dir / "common_bh_fdr_rois.csv",
        index=False,
    )

    outputs = []
    for game in (1, 2):
        summary, scores = game_data[game]
        game_label = "Pong" if game == 1 else "Space Invaders"
        figure_path = (
            Path(output_root)
            / f"sub{int(subject):03d}_game{game}"
            / "transition_pattern_permutation"
            / "figures"
            / "transition_pattern_common_bh_fdr_roi_bars.png"
        )
        plot_transition_discriminant_bars(
            scores,
            summary,
            figure_path,
            game=game,
            roi_order=roi_order,
            significance_column="q_bh_across_rois",
            significance_display="BH-FDR q",
            correction_description=(
                "within-game BH-FDR; panels selected at q<.05 in both games"
            ),
            figure_title=(
                "Transition-related neural patterns in ROIs significant "
                f"in both games · {game_label}"
            ),
            n_columns_override=4,
        )
        outputs.append(str(figure_path))
    return outputs


def run_transition_pattern_permutation_pilot(
    args: argparse.Namespace,
    run_specs: Sequence[COMMON.BrainRunSpec],
    cache_audit: Mapping[str, object],
) -> None:
    """Directly test reproducible transition-pattern separation in every ROI."""
    output_dir = (
        args.output_root
        / f"sub{args.subject:03d}_game{args.game}"
        / "transition_pattern_permutation"
    )
    result_dir = output_dir / "results"
    figure_dir = output_dir / "figures"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    roi_names = COMMON.common_roi_names(run_specs)
    if not roi_names:
        raise RuntimeError("No common ROIs for transition-pattern pilot")
    union_coords, roi_indices, coords_by_roi = common_union_roi_layout(
        run_specs, roi_names
    )
    log(
        f"[pattern pilot] {len(run_specs)} runs, {len(roi_names)} ROIs, "
        f"{len(union_coords):,} unique voxels"
    )
    n_draws = int(args.n_permutations) + 1
    sum_contrast = np.zeros(
        (n_draws, len(union_coords)), dtype=np.float64
    )
    sum_square_contrast = np.zeros_like(sum_contrast)
    observed_run_contrasts: List[np.ndarray] = []
    observed_condition_means: Dict[str, List[np.ndarray]] = {
        condition: [] for condition in TRANSITION_CONDITIONS
    }
    selection_rows: List[Dict[str, object]] = []

    for run_index, run in enumerate(run_specs, start=1):
        process = COMMON.load_process_blocks(run)
        seed = (
            int(args.random_state)
            + 1_000_003 * int(args.game)
            + 1009 * int(run.session)
            + 100_003 * int(run.block)
        )
        weights, audit = transition_contrast_permutation_weights(
            process["transition_events"],
            fraction=args.transition_fraction,
            min_trs=args.min_transition_trs,
            n_permutations=args.n_permutations,
            block_size=args.permutation_block_size,
            seed=seed,
        )
        log(
            f"[pattern pilot {run_index}/{len(run_specs)}] "
            f"{run.run_label}: loading union BOLD"
        )
        patterns = load_union_roi_patterns(
            run,
            roi_names,
            union_coords,
            roi_indices,
            coords_by_roi,
        )
        log(
            f"[pattern pilot {run_index}/{len(run_specs)}] "
            f"{run.run_label}: {args.n_permutations:,} contrasts"
        )
        observed_condition_means["noop_to_active"].append(
            np.mean(patterns[weights[0] > 0], axis=0).astype(np.float32)
        )
        observed_condition_means["active_to_noop"].append(
            np.mean(patterns[weights[0] < 0], axis=0).astype(np.float32)
        )
        contrasts = np.asarray(weights @ patterns, dtype=np.float32)
        observed_run_contrasts.append(contrasts[0].copy())
        sum_contrast += contrasts
        sum_square_contrast += contrasts.astype(np.float64) ** 2
        selection_rows.append(
            {
                "run_label": run.run_label,
                "subject": int(run.subject),
                "session": int(run.session),
                "block": int(run.block),
                "game": int(run.game),
                **audit,
            }
        )
        del patterns, contrasts, weights

    n_runs = int(len(run_specs))
    if n_runs < 2:
        raise RuntimeError("Cross-validated distance needs at least two runs")
    voxel_distances = (
        sum_contrast**2 - sum_square_contrast
    ) / float(n_runs * (n_runs - 1))
    summary, null, max_null = summarize_transition_pattern_distance(
        roi_names,
        roi_indices,
        voxel_distances,
        n_runs=n_runs,
        n_permutations=args.n_permutations,
    )
    discriminant_scores = transition_discriminant_projection_scores(
        roi_names,
        roi_indices,
        observed_condition_means,
        run_specs,
    )

    pair_rows: List[Dict[str, object]] = []
    for first in range(n_runs):
        for second in range(first + 1, n_runs):
            voxel_product = (
                observed_run_contrasts[first].astype(np.float64)
                * observed_run_contrasts[second].astype(np.float64)
            )
            for roi_name in roi_names:
                pair_rows.append(
                    {
                        "roi": roi_name,
                        "run_a": run_specs[first].run_label,
                        "run_b": run_specs[second].run_label,
                        "crossvalidated_pair_distance": float(
                            np.mean(voxel_product[roi_indices[roi_name]])
                        ),
                    }
                )

    summary.to_csv(
        result_dir / "transition_pattern_distance_summary.csv",
        index=False,
    )
    pd.DataFrame(pair_rows).to_csv(
        result_dir / "transition_pattern_distance_run_pairs.csv",
        index=False,
    )
    pd.DataFrame(selection_rows).to_csv(
        result_dir / "transition_pattern_permutation_manifest.csv",
        index=False,
    )
    discriminant_scores.to_csv(
        result_dir / "transition_pattern_discriminant_scores.csv",
        index=False,
    )
    COMMON.atomic_save_npz(
        result_dir / "transition_pattern_distance_nulls.npz",
        roi_names=np.asarray(roi_names, dtype="U"),
        null_distances=null,
        max_stat_null=max_null,
    )
    plot_transition_pattern_distance(
        summary,
        figure_dir / "transition_pattern_distance_all_rois.png",
        game=args.game,
    )
    plot_transition_pattern_nulls(
        summary,
        null,
        max_null,
        figure_dir / "transition_pattern_distance_null_diagnostics.png",
        game=args.game,
        n_top=8,
    )
    plot_transition_discriminant_bars(
        discriminant_scores,
        summary,
        figure_dir / "transition_pattern_significant_roi_bars.png",
        game=args.game,
    )

    pooled_null = null.reshape(-1).astype(float)
    manifest = {
        "analysis": (
            "direct NOOP-to-active versus active-to-NOOP multivoxel "
            "pattern-separation permutation pilot"
        ),
        "subject": int(args.subject),
        "game": int(args.game),
        "n_runs": n_runs,
        "n_rois": int(len(roi_names)),
        "n_unique_voxels": int(len(union_coords)),
        "n_permutations": int(args.n_permutations),
        "distance": (
            "cross-run cross-validated squared Euclidean pattern distance, "
            "averaged per ROI voxel"
        ),
        "condition_contrast": "NOOP-to-active minus active-to-NOOP",
        "null": (
            f"synchronized {args.permutation_block_size}-TR block "
            "permutation of the HRF-aligned transition-label stream"
        ),
        "inference": (
            "one-sided permutation p; BH-FDR across ROIs; synchronized "
            "across-ROI max-statistic FWER"
        ),
        "pooled_null_mean": float(np.mean(pooled_null)),
        "pooled_null_sd": float(np.std(pooled_null, ddof=1)),
        "max_null_mean": float(np.mean(max_null)),
        "minimum_resolvable_p": float(1.0 / (args.n_permutations + 1)),
        "controls": [],
        "population_inference": False,
        "claim_boundary": (
            "A positive distance indicates a transition-related multivoxel "
            "contrast reproducible across runs in this subject. It does not "
            "identify a planning-unique representation because motor/action, "
            "visual, and task-state controls are not included."
        ),
        "cache_manifest": cache_audit,
    }
    COMMON.atomic_write_json(
        result_dir / "analysis_manifest.json", manifest
    )
    n_bh = int(
        np.sum(
            pd.to_numeric(
                summary["q_bh_across_rois"], errors="coerce"
            )
            < 0.05
        )
    )
    n_fwer = int(
        np.sum(
            pd.to_numeric(
                summary["p_fwer_max_stat"], errors="coerce"
            )
            < 0.05
        )
    )
    (output_dir / "summary.md").write_text(
        "\n".join(
            [
                (
                    f"# sub{args.subject:03d} game{args.game} "
                    "transition-pattern permutation pilot"
                ),
                "",
                f"- Runs: {n_runs}",
                f"- ROIs: {len(roi_names)}",
                f"- Permutations: {args.n_permutations:,}",
                (
                    "- Direct statistic: cross-run cross-validated pattern "
                    "distance (not class-recall difference)."
                ),
                f"- BH-FDR q<.05 ROIs: {n_bh}",
                f"- Max-stat FWER p<.05 ROIs: {n_fwer}",
                (
                    f"- Pooled null mean: {float(np.mean(pooled_null)):.6g}"
                ),
                (
                    "- Caveat: pure-effect pilot without motor, visual, "
                    "task-state, or planning controls."
                ),
            ]
        ),
        encoding="utf-8",
    )
    cross_game_outputs = generate_cross_game_common_fdr_bars(
        args.output_root, args.subject
    )
    if cross_game_outputs:
        log(
            "Cross-game common-FDR ROI bars complete -> "
            + ", ".join(cross_game_outputs)
        )
    log(f"Transition-pattern pilot complete -> {output_dir}")


def run_transition_topology_analysis(
    args: argparse.Namespace,
    run_specs: Sequence[COMMON.BrainRunSpec],
    cache_audit: Mapping[str, object],
) -> None:
    """Pure transition MVPA and condition-specific topology RSA in all ROIs."""
    output_dir = (
        args.output_root
        / f"sub{args.subject:03d}_game{args.game}"
        / "transition_topology"
    )
    result_dir = output_dir / "results"
    figure_dir = output_dir / "figures"
    rdm_dir = output_dir / "rdms"
    map_dir = output_dir / "maps"
    for path in (result_dir, figure_dir, rdm_dir, map_dir):
        path.mkdir(parents=True, exist_ok=True)
    if args.cache_only:
        log(f"Transition-topology cache complete: {args.cache_root}")
        return

    prepared_runs: List[Dict[str, object]] = []
    selection_rows: List[Dict[str, object]] = []
    for run_index, run in enumerate(run_specs, start=1):
        log(f"[transition model {run_index}/{len(run_specs)}] {run.run_label}")
        prepared = prepare_transition_topology_run(
            run,
            transition_fraction=args.transition_fraction,
            min_transition_trs=args.min_transition_trs,
            min_pair_lag=args.min_pair_lag,
            n_permutations=(
                0 if args.mvpa_figures_only else args.n_permutations
            ),
            permutation_block_size=args.permutation_block_size,
            random_state=args.random_state,
        )
        prepared_runs.append(prepared)
        audit = prepared["selection_audit"]
        selection_rows.append(
            {
                "run_label": run.run_label,
                **audit,
                "topology_features": int(prepared["topology_features"]),
                "permutation_seed": int(prepared["permutation_seed"]),
            }
        )
        save_arrays: Dict[str, np.ndarray] = {}
        for condition in TRANSITION_CONDITIONS:
            condition_data = prepared["conditions"][condition]
            save_arrays[f"{condition}_selected_trs"] = np.asarray(
                condition_data["selected_trs"], dtype=np.int32
            )
            save_arrays[f"{condition}_topology_rdm_rank"] = np.asarray(
                condition_data["topology_rank"], dtype=np.float32
            )
        COMMON.atomic_save_npz(
            rdm_dir / f"topology_rdms_{run.run_label}.npz",
            **save_arrays,
        )

    roi_names = COMMON.common_roi_names(run_specs)
    rsa_rows: List[Dict[str, object]] = []
    mvpa_rows: List[Dict[str, object]] = []
    for roi_index, roi_name in enumerate(roi_names, start=1):
        if roi_index == 1 or roi_index % 10 == 0 or roi_index == len(roi_names):
            log(f"[transition ROI {roi_index}/{len(roi_names)}] {roi_name}")
        patterns_by_run: List[np.ndarray] = []
        for run in run_specs:
            patterns, _ = COMMON.load_roi(run, roi_name)
            patterns_by_run.append(patterns)

        if not args.mvpa_figures_only:
            for prepared, patterns in zip(prepared_runs, patterns_by_run):
                run = prepared["run"]
                brain_square = COMMON.square_from_condensed(
                    COMMON.correlation_rdm(patterns), run.n_tr
                )
                for condition in TRANSITION_CONDITIONS:
                    condition_data = prepared["conditions"][condition]
                    pair_row = np.asarray(
                        condition_data["pair_row"], dtype=int
                    )
                    pair_column = np.asarray(
                        condition_data["pair_column"], dtype=int
                    )
                    neural_rank = COMMON.rank_zscore(
                        brain_square[pair_row, pair_column]
                    )
                    topology_rank = np.asarray(
                        condition_data["topology_rank"], dtype=float
                    )
                    observed = safe_correlation(neural_rank, topology_rank)
                    topology_null_rank = np.asarray(
                        condition_data["topology_null_rank"], dtype=float
                    )
                    if len(topology_null_rank):
                        null = safe_correlation_batch(
                            topology_null_rank, neural_rank
                        )
                        p_temporal = float(
                            (1 + np.sum(null >= observed))
                            / (1 + len(null))
                        )
                    else:
                        p_temporal = np.nan
                    rsa_rows.append(
                        {
                            "run_label": run.run_label,
                            "subject": run.subject,
                            "session": run.session,
                            "block": run.block,
                            "game": run.game,
                            "roi": roi_name,
                            "condition": condition,
                            "condition_label": TRANSITION_LABELS[condition],
                            "n_voxels": int(patterns.shape[1]),
                            "n_selected_trs": int(
                                len(condition_data["selected_trs"])
                            ),
                            "n_pairs": int(condition_data["n_pairs"]),
                            "topology_r": float(observed),
                            "p_temporal_permutation_one_sided": p_temporal,
                        }
                    )

        # Direct transition MVPA.  Each held-out fold is an entire run.
        for held_out_index, prepared in enumerate(prepared_runs):
            train_x: List[np.ndarray] = []
            train_y: List[np.ndarray] = []
            for run_index, (other, patterns) in enumerate(
                zip(prepared_runs, patterns_by_run)
            ):
                if run_index == held_out_index:
                    continue
                for label, condition in enumerate(
                    ("active_to_noop", "noop_to_active")
                ):
                    indices = np.asarray(
                        other["conditions"][condition]["selected_trs"],
                        dtype=int,
                    )
                    train_x.append(patterns[indices])
                    train_y.append(
                        np.full(len(indices), label, dtype=np.int8)
                    )
            test_x: List[np.ndarray] = []
            test_y: List[np.ndarray] = []
            held_patterns = patterns_by_run[held_out_index]
            for label, condition in enumerate(
                ("active_to_noop", "noop_to_active")
            ):
                indices = np.asarray(
                    prepared["conditions"][condition]["selected_trs"],
                    dtype=int,
                )
                test_x.append(held_patterns[indices])
                test_y.append(np.full(len(indices), label, dtype=np.int8))
            test_truth = np.concatenate(test_y)
            prediction = nearest_centroid_transition_prediction(
                np.vstack(train_x),
                np.concatenate(train_y),
                np.vstack(test_x),
            )
            accuracy = balanced_accuracy_score(test_truth, prediction)
            confusion = transition_confusion_counts(
                test_truth, prediction
            )
            held_run = run_specs[held_out_index]
            mvpa_rows.append(
                {
                    "roi": roi_name,
                    "held_out_run": held_run.run_label,
                    "subject": held_run.subject,
                    "game": held_run.game,
                    "n_voxels": int(held_patterns.shape[1]),
                    "n_test_samples": int(
                        sum(len(values) for values in test_y)
                    ),
                    "balanced_accuracy": float(accuracy),
                    **confusion,
                }
            )

    rsa_run = pd.DataFrame(rsa_rows)
    if rsa_run.empty:
        rsa_summary = pd.DataFrame()
        rsa_difference = pd.DataFrame()
    else:
        rsa_summary, rsa_difference = (
            summarize_transition_topology_rsa(rsa_run)
        )
    mvpa_fold = pd.DataFrame(mvpa_rows)
    mvpa_summary = summarize_transition_mvpa(mvpa_fold)
    mvpa_class_summary = summarize_transition_mvpa_class_recall(mvpa_fold)
    mvpa_recall_difference = (
        summarize_transition_mvpa_recall_difference(mvpa_fold)
    )
    confusion_summary = transition_confusion_summary(
        mvpa_fold, mvpa_summary
    )
    pd.DataFrame(selection_rows).to_csv(
        result_dir / "transition_tr_selection_manifest.csv", index=False
    )
    if not args.mvpa_figures_only:
        rsa_run.to_csv(
            result_dir / "topology_rsa_run_effects.csv", index=False
        )
        rsa_summary.to_csv(
            result_dir / "topology_rsa_across_run_summary.csv", index=False
        )
        rsa_difference.to_csv(
            result_dir / "topology_rsa_condition_difference.csv", index=False
        )
    mvpa_fold.to_csv(
        result_dir / "transition_mvpa_loro_folds.csv", index=False
    )
    mvpa_summary.to_csv(
        result_dir / "transition_mvpa_across_run_summary.csv", index=False
    )
    mvpa_class_summary.to_csv(
        result_dir / "transition_mvpa_class_recall_summary.csv",
        index=False,
    )
    mvpa_recall_difference.to_csv(
        result_dir / "transition_mvpa_recall_difference_summary.csv",
        index=False,
    )
    confusion_summary.to_csv(
        result_dir / "transition_mvpa_confusion_matrix_summary.csv",
        index=False,
    )
    if not args.mvpa_figures_only:
        plot_transition_topology_rsa(
            rsa_summary,
            rsa_difference,
            figure_dir / "transition_topology_rsa_all_rois.png",
        )
    plot_transition_mvpa(
        mvpa_summary,
        figure_dir / "transition_mvpa_all_rois.png",
    )
    plot_transition_confusion_matrix(
        confusion_summary,
        figure_dir / "transition_mvpa_confusion_matrix.png",
        game=args.game,
    )
    plot_top_transition_roi_recalls(
        mvpa_fold,
        mvpa_summary,
        mvpa_recall_difference,
        figure_dir / "transition_mvpa_top_roi_class_recall.png",
        game=args.game,
        n_top_rois=args.top_mvpa_rois,
    )
    roi_mask_paths = COMMON.select_roi_masks(
        args.atlas_root, args.roi_set, args.roi_include
    )
    plot_transition_mvpa_parcel_map(
        mvpa_summary,
        args.atlas_root,
        roi_mask_paths,
        figure_dir / "transition_mvpa_parcel_map.png",
        map_dir,
        game=args.game,
    )
    manifest = {
        "analysis": (
            "uncontrolled transition MVPA and condition-specific "
            "search-topology RSA"
        ),
        "subject": int(args.subject),
        "game": int(args.game),
        "n_runs": int(len(run_specs)),
        "n_rois": int(len(roi_names)),
        "roi_set": str(args.roi_set),
        "max_roi_voxels": int(args.max_roi_voxels),
        "transition_conditions": list(TRANSITION_CONDITIONS),
        "transition_definition": (
            "real-step previous/current human action; exact-time 1-Hz "
            "aggregation; canonical HRF; equal-sized top/bottom transition-"
            "contrast TR sets"
        ),
        "transition_fraction": float(args.transition_fraction),
        "min_transition_trs": int(args.min_transition_trs),
        "topology_representation": (
            "status-2 action-prefix allocation matrix: root branch x depth x "
            "{visit, unique-prefix, repeated-prefix}"
        ),
        "topology_max_depth": int(args.topology_max_depth),
        "topology_rdm": "correlation distance",
        "neural_rdm": "ROI multivoxel correlation distance",
        "rsa_statistic": "uncontrolled Spearman correlation",
        "mvpa": (
            "standardized nearest-centroid classifier with leave-one-run-out "
            "validation, balanced accuracy, class-specific recall, and "
            "run-/ROI-equalized confusion matrices"
        ),
        "mvpa_class_recall_inference": (
            "paired NOOP-to-active minus active-to-NOOP recall; two-sided "
            "exact sign-flip across held-out runs; BH-FDR across ROIs"
        ),
        "mvpa_confusion_matrix": (
            "row-normalized within each ROI/run, then equal-weight averaged "
            "across runs and ROIs"
        ),
        "mvpa_top_roi_plot_count": int(args.top_mvpa_rois),
        "mvpa_figures_only": bool(args.mvpa_figures_only),
        "controls": [],
        "min_pair_lag_tr": int(args.min_pair_lag),
        "temporal_null": (
            f"{args.n_permutations} synchronized "
            f"{args.permutation_block_size}-TR topology block permutations"
        ),
        "across_run_inference": (
            "exact sign-flip; BH-FDR across all Harvard-Oxford ROIs"
        ),
        "claim_boundary": (
            "Pure effects only. Direct transition MVPA includes current and "
            "previous action/motor differences; topology RSA may include "
            "action-labelled branch structure. Neither result is unique to "
            "planning until controlled sensitivity analyses are added."
        ),
        "population_inference": False,
        "cache_manifest": cache_audit,
    }
    COMMON.atomic_write_json(
        result_dir / "analysis_manifest.json", manifest
    )
    summary_lines = [
        f"# sub{args.subject:03d} game{args.game} transition/topology RSA",
        "",
        f"- Runs: {len(run_specs)}",
        f"- ROIs: {len(roi_names)} (all available atlas ROIs)",
        "- Controls: none; this is the requested pure-effect tier.",
        (
            "- RSA: neural-pattern similarity versus canonical search-"
            "topology similarity, estimated separately in transition-"
            "dominant TRs."
        ),
        (
            "- MVPA: direct NOOP→active versus active→NOOP decoding with "
            "leave-one-run-out validation."
        ),
        (
            "- MVPA figures: parcel-wise balanced-accuracy map, normalized "
            "confusion matrices, and class-specific recall in top ROIs."
        ),
        (
            "- Caveat: transition decoding includes action/motor history and "
            "cannot by itself establish planning-specific information."
        ),
        "- Figures: PNG only.",
        "",
    ]
    (output_dir / "summary.md").write_text(
        "\n".join(summary_lines), encoding="utf-8"
    )
    log(f"Transition/topology analysis complete -> {output_dir}")


def run_controlled_subject_analysis(
    args: argparse.Namespace,
    run_specs: Sequence[COMMON.BrainRunSpec],
    cache_audit: Mapping[str, object],
) -> None:
    """Default controlled RSA for one subject/game."""
    output_dir = (
        Path(args.output_root)
        / f"sub{int(args.subject):03d}_game{int(args.game)}"
    )
    result_dir = output_dir / "results"
    figure_dir = output_dir / "figures"
    rdm_dir = output_dir / "rdms"
    for path in (result_dir, figure_dir, rdm_dir):
        path.mkdir(parents=True, exist_ok=True)
    if args.cache_only:
        log(f"Read-only shared-cache validation complete: {args.cache_root}")
        return

    roi_names = COMMON.common_roi_names(run_specs)
    if not roi_names:
        raise RuntimeError("No common ROI cache across runs")
    subject_provenance = controlled_subject_provenance(
        args, run_specs, cache_audit, roi_names
    )

    raw_blocks: Dict[str, Dict[str, np.ndarray]] = {}
    block_audits: List[Dict[str, object]] = []
    for run in run_specs:
        blocks, audit = load_controlled_run_blocks(
            run, include_ram_control=args.include_ram_control
        )
        raw_blocks[run.run_label] = blocks
        block_audits.append(audit)
    blocks_by_run = add_loro_orthonormal_geometry(raw_blocks)

    all_rows: List[Dict[str, object]] = []
    null_store: Dict[str, np.ndarray] = {}
    rdm_manifest: List[Dict[str, object]] = []
    design_audit_rows: List[Dict[str, object]] = []
    correlation_audits: List[Dict[str, object]] = []

    for run_index, run in enumerate(run_specs, start=1):
        log(
            f"[controlled run {run_index}/{len(run_specs)}] "
            f"{run.run_label}"
        )
        blocks = blocks_by_run[run.run_label]
        rdms = build_controlled_rdms(blocks)
        pair_mask = COMMON.pair_mask_min_lag(
            run.n_tr, args.min_pair_lag
        )
        designs = prepare_controlled_comparisons(rdms, pair_mask)
        seed = (
            int(args.random_state)
            + 1_000_003 * int(args.game)
            + 1009 * int(run.session)
            + 100_003 * int(run.block)
        )
        orders = synchronized_block_orders(
            run.n_tr,
            args.n_permutations,
            args.permutation_block_size,
            seed,
        )
        COMMON.atomic_save_npz(
            rdm_dir / f"controlled_model_rdms_{run.run_label}.npz",
            **{
                name: np.asarray(values, dtype=np.float32)
                for name, values in rdms.items()
            },
        )
        for name, values in rdms.items():
            rdm_manifest.append(
                {
                    "run_label": run.run_label,
                    "rdm": name,
                    "n_tr": int(run.n_tr),
                    "n_pairs": int(len(values)),
                    "rdm_kind": (
                        "nuisance" if name.startswith("N_") else "model"
                    ),
                    "inference_family": (
                        controlled_effect_family(name)
                        if name in CONTROLLED_EFFECT_ORDER
                        else "nuisance"
                    ),
                }
            )
        for effect, design in designs.items():
            design_audit_rows.append(
                {
                    "run_label": run.run_label,
                    "effect": effect,
                    "kept_controls": "|".join(
                        design["kept_controls"]
                    ),
                    "dropped_alias_controls": "|".join(
                        design["dropped_controls"]
                    ),
                    "target_residual_ratio": float(
                        design["target_residual_ratio"]
                    ),
                    "reduced_rank": int(
                        np.linalg.matrix_rank(
                            design["reduced_design"]
                        )
                    ),
                    "reduced_columns": int(
                        np.asarray(design["reduced_design"]).shape[1]
                    ),
                    "full_rank": int(
                        np.linalg.matrix_rank(design["full_design"])
                    ),
                    "full_columns": int(
                        np.asarray(design["full_design"]).shape[1]
                    ),
                }
            )
        correlation_audits.append(
            {
                "run_label": run.run_label,
                "evidence_primary_vs_all29_r": safe_correlation(
                    np.asarray(blocks["evidence_primary"])[:, 0],
                    np.asarray(blocks["evidence_all29"])[:, 0],
                ),
                "breadth_vs_depth_r": safe_correlation(
                    np.asarray(blocks["geometry_primary"])[:, 0],
                    np.asarray(blocks["geometry_primary"])[:, 1],
                ),
                "permutation_seed": int(seed),
                "n_permutations": int(len(orders)),
            }
        )

        def analyze_one(roi_name: str):
            return analyze_controlled_roi_run(
                run,
                roi_name,
                designs,
                pair_mask,
                orders,
                permutation_batch_size=args.permutation_batch_size,
            )

        if args.n_jobs == 1:
            roi_results = []
            for roi_index, roi_name in enumerate(roi_names, start=1):
                if (
                    roi_index == 1
                    or roi_index % 10 == 0
                    or roi_index == len(roi_names)
                ):
                    log(
                        f"  [ROI {roi_index}/{len(roi_names)}] "
                        f"{roi_name}"
                    )
                roi_results.append(analyze_one(roi_name))
        else:
            results_by_index: Dict[int, Tuple[List[Dict[str, object]], Dict[str, np.ndarray]]] = {}
            with ThreadPoolExecutor(
                max_workers=min(int(args.n_jobs), len(roi_names))
            ) as executor:
                future_to_index = {
                    executor.submit(analyze_one, roi_name): index
                    for index, roi_name in enumerate(roi_names)
                }
                for completed, future in enumerate(
                    as_completed(future_to_index), start=1
                ):
                    results_by_index[future_to_index[future]] = future.result()
                    if completed % 10 == 0 or completed == len(roi_names):
                        log(
                            f"  [ROI {completed}/{len(roi_names)}] "
                            f"workers={min(int(args.n_jobs), len(roi_names))}"
                        )
            roi_results = [
                results_by_index[index] for index in range(len(roi_names))
            ]
        for rows, stored in roi_results:
            all_rows.extend(rows)
            null_store.update(stored)

    run_effects = add_controlled_within_run_fwer(
        pd.DataFrame(all_rows), null_store
    )
    for statistic in ("added_beta", "conditional_partial_r"):
        p_column = f"p_block_{statistic}"
        q_column = f"q_block_family_{statistic}"
        run_effects[q_column] = (
            run_effects.groupby(
                ["run_label", "inference_family"], group_keys=False
            )[p_column].transform(
                lambda values: COMMON.fdr_bh(
                    values.to_numpy(dtype=float)
                )
            )
        )
    subject_effects = summarize_controlled_subject(run_effects)
    subject_effects.insert(0, "game", int(args.game))
    subject_effects.insert(0, "subject", int(args.subject))

    run_effects.to_csv(
        result_dir / "controlled_rsa_run_effects.csv", index=False
    )
    subject_effects.to_csv(
        result_dir / "subject_effects.csv", index=False
    )
    pd.DataFrame(rdm_manifest).to_csv(
        result_dir / "rdm_manifest.csv", index=False
    )
    pd.DataFrame(design_audit_rows).to_csv(
        result_dir / "design_rank_audit.csv", index=False
    )
    pd.DataFrame(correlation_audits).to_csv(
        result_dir / "feature_correlation_audit.csv", index=False
    )
    pd.DataFrame(block_audits).to_json(
        result_dir / "canonical_block_audit.json",
        orient="records",
        indent=2,
    )
    if null_store:
        COMMON.atomic_save_npz(
            result_dir / "controlled_block_permutation_nulls.npz",
            **null_store,
        )
    plot_controlled_rsa_heatmap(
        subject_effects,
        figure_dir / "controlled_rsa_subject_partial_r.png",
        title=(
            f"Controlled planning RSA · sub{int(args.subject):03d} "
            f"game {int(args.game)}"
        ),
    )

    subject_map_source = subject_effects[
        subject_effects["statistic"].astype(str).eq("conditional_partial_r")
        & subject_effects["effect"].astype(str).isin(NEURAL_MAP_EFFECTS)
    ].copy()
    subject_corrected = subject_map_source[
        pd.to_numeric(
            subject_map_source["p_max_abs_t_fwer"], errors="raise"
        ) < 0.05
    ].copy()
    subject_brain_path = (
        figure_dir / "controlled_rsa_corrected_brain_maps.pdf"
    )
    NROI.save_effect_brain_maps(
        subject_brain_path,
        panels=_effect_panels(
            subject_corrected, value_column="mean_effect"
        ),
        value_column="mean_effect",
        title=(
            f"{NEURAL_GAME_LABELS[int(args.game)]} · participant "
            f"{int(args.subject)} process-resolved RSA"
        ),
        subtitle=(
            "11-run exact sign flip; max-|t| within the prespecified "
            "planning, transition, and sensitivity families"
        ),
        metric_label="Conditional partial Spearman r",
        signed=True,
        n_columns=4,
    )
    manifest = {
        "analysis": "controlled 02--04 planning-pathway RSA",
        "schema_version": CONTROLLED_RSA_SCHEMA_VERSION,
        **controlled_estimator_provenance(),
        **subject_provenance,
        "subject": int(args.subject),
        "game": int(args.game),
        "n_runs": int(len(run_specs)),
        "n_rois": int(len(roi_names)),
        "primary_effects": list(CONTROLLED_PRIMARY_ORDER),
        "primary_transition_effects": list(CONTROLLED_TRANSITION_ORDER),
        "sensitivity_effects": list(CONTROLLED_SENSITIVITY_ORDER),
        "staged_controls": {
            effect: list(controls)
            for effect, controls in controlled_stage_controls().items()
        },
        "nuisance_controls": list(CONTROLLED_BASE_NUISANCE_ORDER),
        "include_ram_control": bool(args.include_ram_control),
        "geometry_joint_scaling": (
            "two-dimensional whitening fit on other runs only"
        ),
        "primary_signed_statistics": [
            "added_beta",
            "conditional_partial_r",
        ],
        "delta_r2_role": "in-sample descriptive only",
        "within_run_temporal_null": (
            f"{int(args.n_permutations)} synchronized "
            f"{int(args.permutation_block_size)}-TR block permutations; "
            "same draws for every ROI and effect"
        ),
        "subject_inference": (
            "run-equal exhaustive sign flip, two-sided raw p and "
            "family-wise synchronized max-|t|; BH within predeclared family"
        ),
        "population_inference": False,
        "transition_claim_boundary": (
            "active-status switch/stay interaction geometry beyond origin, "
            "destination, and motor/action RDMs; this is not CCGP, not an "
            "invariant crossed-direction conjunction, and not direct "
            "NOOP-to-active versus active-to-NOOP decoding"
        ),
        "mvpa_location": "06_neural_state_space.py",
        "cache_manifest": cache_audit,
    }
    COMMON.atomic_write_json(
        result_dir / "analysis_manifest.json", manifest
    )
    (output_dir / "summary.md").write_text(
        "\n".join(
            [
                (
                    f"# sub{int(args.subject):03d} game{int(args.game)} "
                    "controlled RSA"
                ),
                "",
                f"- Runs: {len(run_specs)}; ROIs: {len(roi_names)}",
                (
                    "- Default analysis: staged P/M/Q0/R/E/G/A and "
                    "status-switch interaction RSA with visual, action, history, "
                    "drift/time, and optional RAM controls."
                ),
                (
                    "- Primary signed statistics: conditional beta and "
                    "partial Spearman r. In-sample delta R2 is descriptive."
                ),
                (
                    "- Subject inference: exact run sign flip; synchronized "
                    "max-|t| within planning, transition, and sensitivity "
                    "families."
                ),
                (
                    "- MVPA and direct transition decoding belong to "
                    "06_neural_state_space.py."
                ),
            ]
        ),
        encoding="utf-8",
    )
    log(f"Controlled subject RSA complete -> {output_dir}")


def parse_subject_list(value: str) -> List[int]:
    subjects = sorted(
        {
            COMMON.parse_subject(part.strip())
            for part in str(value).split(",")
            if part.strip()
        }
    )
    expected = list(range(1, 7))
    if subjects != expected:
        raise argparse.ArgumentTypeError(
            "Confirmatory group RSA requires exactly subjects 1,2,3,4,5,6; "
            f"received {subjects}"
        )
    return subjects


def run_controlled_group_analysis(args: argparse.Namespace) -> None:
    """Aggregate already-computed subject means; never pool runs."""
    subjects = parse_subject_list(args.subjects)
    frames = []
    input_rows = []
    common_settings_hash: Optional[str] = None
    common_settings: Optional[Dict[str, object]] = None
    cache_source_hashes: Dict[str, str] = {}
    for subject in subjects:
        path = (
            Path(args.output_root)
            / f"sub{subject:03d}_game{int(args.game)}"
            / "results"
            / "subject_effects.csv"
        )
        if not path.exists():
            raise FileNotFoundError(
                f"Missing subject RSA input: {path}"
            )
        frame = pd.read_csv(path)
        if "subject" not in frame:
            frame.insert(0, "subject", int(subject))
        observed_subjects = set(
            pd.to_numeric(frame["subject"], errors="raise").astype(int)
        )
        if observed_subjects != {int(subject)}:
            raise ValueError(
                f"Subject label mismatch in {path}: {observed_subjects}"
            )
        if "game" not in frame:
            raise ValueError(f"Subject RSA input lacks game label: {path}")
        observed_games = set(
            pd.to_numeric(frame["game"], errors="raise").astype(int)
        )
        if observed_games != {int(args.game)}:
            raise ValueError(
                f"Game label mismatch in {path}: {observed_games} "
                f"!= {{{int(args.game)}}}"
            )
        subject_manifest_path = path.parent / "analysis_manifest.json"
        if not subject_manifest_path.exists():
            raise FileNotFoundError(
                f"Missing subject analysis manifest: {subject_manifest_path}"
            )
        subject_manifest = json.loads(
            subject_manifest_path.read_text(encoding="utf-8")
        )
        if (
            int(subject_manifest.get("subject", -1)) != int(subject)
            or int(subject_manifest.get("game", -1)) != int(args.game)
            or int(subject_manifest.get("schema_version", -1))
            != CONTROLLED_RSA_SCHEMA_VERSION
        ):
            raise ValueError(
                f"Subject manifest mismatch: {subject_manifest_path}"
            )
        settings_hash, settings, cache_hash = (
            validate_controlled_subject_provenance(
                subject_manifest, subject_manifest_path, frame,
                allow_compatible_v4_source=True,
            )
        )
        if common_settings_hash is None:
            common_settings_hash = settings_hash
            common_settings = settings
        elif settings_hash != common_settings_hash or settings != common_settings:
            raise ValueError("Controlled RSA subject settings are incompatible")
        cache_source_hashes[f"sub{int(subject):03d}"] = cache_hash
        frames.append(frame)
        input_rows.append(
            {
                "subject": int(subject),
                "path": str(path),
                "manifest_path": str(subject_manifest_path),
                "n_rows": int(len(frame)),
                "controlled_settings_sha256": settings_hash,
                "controlled_cache_source_sha256": cache_hash,
            }
        )
    subject_effects = pd.concat(frames, ignore_index=True)
    group_effects = summarize_controlled_group(subject_effects)
    group_effects.insert(0, "game", int(args.game))

    output_dir = (
        Path(args.output_root) / f"group_game{int(args.game)}"
    )
    result_dir = output_dir / "results"
    figure_dir = output_dir / "figures"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    subject_effects.to_csv(
        result_dir / "group_subject_inputs.csv", index=False
    )
    group_effects.to_csv(
        result_dir / "group_effects.csv", index=False
    )
    pd.DataFrame(input_rows).to_csv(
        result_dir / "group_input_manifest.csv", index=False
    )
    plot_controlled_rsa_heatmap(
        group_effects,
        figure_dir / "controlled_rsa_group_partial_r.png",
        title=f"Controlled planning RSA · group game {int(args.game)}",
    )
    group_map_source = group_effects[
        group_effects["statistic"].astype(str).eq("conditional_partial_r")
        & group_effects["effect"].astype(str).isin(NEURAL_MAP_EFFECTS)
    ].copy()
    group_corrected = group_map_source[
        pd.to_numeric(
            group_map_source["p_max_abs_t_fwer"], errors="raise"
        ) < 0.05
    ].copy()
    group_brain_path = figure_dir / "controlled_rsa_corrected_brain_maps.pdf"
    NROI.save_effect_brain_maps(
        group_brain_path,
        panels=_effect_panels(
            group_corrected, value_column="mean_effect"
        ),
        value_column="mean_effect",
        title=f"{NEURAL_GAME_LABELS[int(args.game)]} · process-resolved RSA",
        subtitle=(
            "Six-subject exact sign flip; max-|t| within the prespecified "
            "planning, transition, and sensitivity families"
        ),
        metric_label="Mean conditional partial Spearman r",
        signed=True,
        n_columns=4,
    )
    primary_subjects = subject_effects[
        subject_effects["statistic"].astype(str).eq("conditional_partial_r")
        & subject_effects["effect"].astype(str).isin(NEURAL_MAP_EFFECTS)
    ].copy()
    predefined_subjects = NROI.aggregate_predefined_rois(
        primary_subjects,
        value_columns=["mean_effect"],
        grouping_columns=[
            "subject", "game", "effect", "effect_label",
            "inference_family", "statistic",
        ],
    )
    predefined_inference = _predefined_roi_exact_inference_07(
        predefined_subjects
    )
    predefined_mapping_path = result_dir / "predefined_roi_mapping.csv"
    predefined_subject_path = result_dir / "predefined_roi_subject_effects.csv"
    predefined_inference_path = result_dir / "predefined_roi_inference.csv"
    NROI.predefined_roi_mapping_frame().to_csv(
        predefined_mapping_path, index=False
    )
    predefined_subjects.to_csv(predefined_subject_path, index=False)
    predefined_inference.to_csv(predefined_inference_path, index=False)
    predefined_figure_path = figure_dir / "predefined_roi_effects.pdf"
    _save_predefined_roi_effect_figure_07(
        predefined_subjects, predefined_inference, predefined_figure_path,
        game=int(args.game),
    )
    anatomy_path = Path(args.output_root) / "figures" / "predefined_roi_anatomy.pdf"
    NROI.save_predefined_roi_anatomy(anatomy_path)

    COMMON.atomic_write_json(
        result_dir / "analysis_manifest.json",
        {
            "analysis": "controlled 02--04 planning-pathway RSA group",
            "schema_version": CONTROLLED_RSA_SCHEMA_VERSION,
            **controlled_estimator_provenance(),
            "controlled_subject_settings_sha256": common_settings_hash,
            "controlled_subject_settings": common_settings,
            "input_cache_source_sha256": cache_source_hashes,
            "game": int(args.game),
            "subjects": [int(value) for value in subjects],
            "n_subjects": int(len(subjects)),
            "population_unit": "subject mean across runs",
            "run_pooling": False,
            "inference": (
                "exhaustive synchronized subject sign flip; two-sided raw "
                "p, max-|t| FWER within predeclared planning, transition, "
                "and sensitivity families, and BH q within family"
            ),
            "predefined_roi_inference": (
                "equal left/right parcel mean; exhaustive 64-pattern subject "
                "sign flip; max-|t| across eight literature-defined ROIs "
                "separately for each effect and game"
            ),
            "predefined_roi_status": (
                "literature-defined secondary, not preregistered"
            ),
            "predefined_roi_artifacts": {
                "mapping": str(predefined_mapping_path.resolve()),
                "subject_effects": str(predefined_subject_path.resolve()),
                "inference": str(predefined_inference_path.resolve()),
                "effect_figure": str(predefined_figure_path.resolve()),
                "anatomy_figure": str(anatomy_path.resolve()),
            },
            "n_sign_patterns": int(2 ** len(subjects)),
        },
    )
    (output_dir / "summary.md").write_text(
        "\n".join(
            [
                f"# Controlled RSA group · game {int(args.game)}",
                "",
                f"- Subjects: {', '.join(f'sub{x:03d}' for x in subjects)}",
                "- Population unit: subject; runs were not pooled.",
                f"- Exact synchronized sign patterns: {2 ** len(subjects)}",
            ]
        ),
        encoding="utf-8",
    )
    log(f"Controlled group RSA complete -> {output_dir}")


def run_controlled_cross_game_analysis(args: argparse.Namespace) -> None:
    """Exact paired-game synthesis after pairing games within each subject."""
    subjects = parse_subject_list(args.subjects)
    frames = []
    input_rows = []
    common_settings_hash: Optional[str] = None
    common_settings: Optional[Dict[str, object]] = None
    cache_source_hashes: Dict[str, str] = {}
    for subject in subjects:
        for game in (1, 2):
            path = (
                Path(args.output_root)
                / f"sub{subject:03d}_game{game}"
                / "results"
                / "subject_effects.csv"
            )
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing paired-game subject RSA input: {path}"
                )
            frame = pd.read_csv(path)
            if "subject" not in frame or "game" not in frame:
                raise ValueError(
                    f"Paired-game input lacks subject/game labels: {path}"
                )
            observed_subjects = set(
                pd.to_numeric(
                    frame["subject"], errors="raise"
                ).astype(int)
            )
            observed_games = set(
                pd.to_numeric(frame["game"], errors="raise").astype(int)
            )
            if observed_subjects != {int(subject)} or observed_games != {
                int(game)
            }:
                raise ValueError(
                    f"Paired-game label mismatch in {path}: "
                    f"subjects={observed_subjects}, games={observed_games}"
                )
            manifest_path = path.parent / "analysis_manifest.json"
            if not manifest_path.exists():
                raise FileNotFoundError(
                    f"Missing paired-game subject manifest: {manifest_path}"
                )
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            if (
                int(manifest.get("subject", -1)) != int(subject)
                or int(manifest.get("game", -1)) != int(game)
                or int(manifest.get("schema_version", -1))
                != CONTROLLED_RSA_SCHEMA_VERSION
            ):
                raise ValueError(
                    f"Paired-game manifest mismatch: {manifest_path}"
                )
            settings_hash, settings, cache_hash = (
                validate_controlled_subject_provenance(
                    manifest, manifest_path, frame,
                    allow_compatible_v4_source=True,
                )
            )
            if common_settings_hash is None:
                common_settings_hash = settings_hash
                common_settings = settings
            elif (
                settings_hash != common_settings_hash
                or settings != common_settings
            ):
                raise ValueError(
                    "Paired-game controlled RSA settings are incompatible"
                )
            cache_source_hashes[
                f"sub{int(subject):03d}_game{int(game)}"
            ] = cache_hash
            frames.append(frame)
            input_rows.append(
                {
                    "subject": int(subject),
                    "game": int(game),
                    "path": str(path),
                    "manifest_path": str(manifest_path),
                    "n_rows": int(len(frame)),
                    "controlled_settings_sha256": settings_hash,
                    "controlled_cache_source_sha256": cache_hash,
                }
            )
    inputs = pd.concat(frames, ignore_index=True)
    paired, cross_game = summarize_controlled_cross_game(inputs)
    if not np.all(cross_game["n_units"].to_numpy(dtype=int) == 6):
        raise AssertionError(
            "Cross-game inference did not retain exactly six paired subjects"
        )
    if not np.all(
        cross_game["n_sign_patterns"].to_numpy(dtype=int) == 64
    ):
        raise AssertionError(
            "Cross-game inference did not enumerate exactly 64 sign patterns"
        )

    output_dir = Path(args.output_root) / "group_cross_game"
    result_dir = output_dir / "results"
    figure_dir = output_dir / "figures"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    inputs.to_csv(
        result_dir / "paired_game_subject_inputs.csv", index=False
    )
    paired.to_csv(
        result_dir / "paired_game_subject_contrasts.csv", index=False
    )
    cross_game.to_csv(
        result_dir / "cross_game_effects.csv", index=False
    )
    pd.DataFrame(input_rows).to_csv(
        result_dir / "cross_game_input_manifest.csv", index=False
    )
    for contrast, title in (
        (
            "common_mean",
            "Controlled RSA · equal-weight mean of two observed games",
        ),
        (
            "game2_minus_game1",
            "Controlled RSA · Game 2 minus Game 1",
        ),
    ):
        plot_controlled_rsa_heatmap(
            cross_game[
                cross_game["cross_game_contrast"].astype(str).eq(contrast)
            ],
            figure_dir / f"controlled_rsa_{contrast}_partial_r.png",
            title=title,
        )
    COMMON.atomic_write_json(
        result_dir / "analysis_manifest.json",
        {
            "analysis": "paired-game controlled 02--04 pathway RSA",
            "schema_version": CONTROLLED_RSA_SCHEMA_VERSION,
            **controlled_estimator_provenance(),
            "controlled_subject_settings_sha256": common_settings_hash,
            "controlled_subject_settings": common_settings,
            "input_cache_source_sha256": cache_source_hashes,
            "subjects": [int(value) for value in subjects],
            "n_subjects": 6,
            "games": [1, 2],
            "contrasts": {
                "common_mean": "(Game1 + Game2) / 2 within subject",
                "game2_minus_game1": "Game2 - Game1 within subject",
            },
            "population_unit": "paired subject contrast",
            "subject_game_cell_pooling": False,
            "inference": (
                "64-pattern exhaustive synchronized subject sign flip; "
                "two-sided raw p, max-|t| FWER within planning, transition, "
                "and sensitivity families, BH q within family"
            ),
        },
    )
    (output_dir / "summary.md").write_text(
        "\n".join(
            [
                "# Controlled RSA paired-game synthesis",
                "",
                "- Subjects: sub001--sub006, paired within subject.",
                (
                    "- Contrasts: within-subject equal-weight two-game mean and "
                    "Game 2 minus Game 1."
                ),
                "- Subject-game cells were not pooled.",
                "- Exact synchronized sign patterns: 64.",
            ]
        ),
        encoding="utf-8",
    )
    log(f"Controlled paired-game RSA complete -> {output_dir}")


def summarize_controlled_cross_game(
    inputs: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pair Game1/Game2 within subject, then run exact subject inference."""
    keys = [
        "subject",
        "roi",
        "effect",
        "effect_label",
        "inference_family",
        "statistic",
    ]
    duplicates = inputs.duplicated([*keys, "game"], keep=False)
    if np.any(duplicates):
        raise ValueError(
            "Duplicate paired-game subject/effect rows: "
            f"{inputs.loc[duplicates, [*keys, 'game']].head().to_dict('records')}"
        )
    pivot = inputs.pivot(
        index=keys,
        columns="game",
        values="mean_effect",
    )
    if set(pivot.columns) != {1, 2} or pivot[[1, 2]].isna().any().any():
        raise ValueError(
            "Paired-game RSA requires exactly one finite Game1 and Game2 "
            "subject mean for every ROI/effect/statistic"
        )
    paired = pivot.reset_index()
    paired["common_mean"] = 0.5 * (
        paired[1].to_numpy(dtype=float)
        + paired[2].to_numpy(dtype=float)
    )
    paired["game2_minus_game1"] = (
        paired[2].to_numpy(dtype=float)
        - paired[1].to_numpy(dtype=float)
    )
    output_frames: List[pd.DataFrame] = []
    for statistic, statistic_rows in paired.groupby(
        "statistic", sort=False
    ):
        for contrast in ("common_mean", "game2_minus_game1"):
            inference = controlled_exact_summary(
                statistic_rows,
                unit_column="subject",
                value_column=contrast,
                statistic_name=str(statistic),
            )
            inference["cross_game_contrast"] = contrast
            output_frames.append(inference)
    cross_game = pd.concat(output_frames, ignore_index=True)
    return paired, cross_game


NEURAL_MAP_EFFECTS = (
    "P",
    "M",
    "Q0",
    "R",
    "E",
    "G_joint",
    "A",
    "switch_status_interaction",
)
NEURAL_MAP_TITLES = {
    "P": "Pre-search policy\n$\\pi_t^0$",
    "M": "Incumbent margin\n$M_t$",
    "Q0": "Initial Q state\n$Q_t^0$",
    "R": "Value-update magnitude\n$\\log U_t$",
    "E": "Rollout advantage\n$\\mathcal{A}^{roll}_t$",
    "G_joint": "Search allocation\n$\\mathcal{G}_t$",
    "A": "Post-search Actor residual\n$A_t^{res}$",
    "switch_status_interaction": "Hold/act state change\n$Y_t^{status}$",
}
NEURAL_GAME_LABELS = {1: "Pong", 2: "Space Invaders"}


def _validate_unified_v4_cache_roots(
    args: argparse.Namespace,
    *,
    require_planning_trace: bool,
) -> None:
    """Fail closed: analysis code is a read-only consumer of canonical caches."""
    observed_neural = Path(args.cache_root).expanduser().resolve()
    expected_neural = Path(COMMON.DEFAULT_SHARED_CACHE_ROOT).resolve()
    if observed_neural != expected_neural:
        raise ValueError(
            "Unified 07_RSA accepts only the canonical v4 neural cache: "
            f"{expected_neural}; received {observed_neural}"
        )
    if not expected_neural.is_dir():
        raise FileNotFoundError(
            f"Canonical v4 neural cache is missing: {expected_neural}. "
            "Build it with 00_brain_cache.py."
        )
    output_root = Path(args.output_root).expanduser().resolve()
    if (
        output_root == expected_neural
        or output_root in expected_neural.parents
        or expected_neural in output_root.parents
    ):
        raise ValueError("Analysis output may not overlap the v4 neural cache")
    if not require_planning_trace:
        return
    observed_trace = Path(args.planning_trace_root).expanduser().resolve()
    expected_trace = DEFAULT_PLANNING_TRACE_ROOT.resolve()
    if observed_trace != expected_trace:
        raise ValueError(
            "Planning-trace RSA accepts only the persistent v1 cache: "
            f"{expected_trace}; received {observed_trace}"
        )
    if not expected_trace.is_dir():
        raise FileNotFoundError(
            f"Persistent planning-trace cache is missing: {expected_trace}. "
            "Build it with 00_brain_cache.py --planning-trace-only."
        )
    output_root = Path(args.output_root).expanduser().resolve()
    if (
        output_root == expected_trace
        or output_root in expected_trace.parents
        or expected_trace in output_root.parents
    ):
        raise ValueError("Analysis output may not overlap the planning-trace cache")


def _planning_trace_namespace(
    output_root: Path,
    *,
    subject: Optional[int] = None,
    game: Optional[int] = None,
    paired: bool = False,
) -> Path:
    if paired:
        return Path(output_root) / "group_paired_games" / "planning_trace"
    if subject is not None and game is not None:
        return (
            Path(output_root)
            / f"sub{int(subject):03d}_game{int(game)}"
            / "planning_trace"
        )
    if game is not None:
        return Path(output_root) / f"group_game{int(game)}" / "planning_trace"
    raise ValueError("Planning-trace namespace requires subject/game or paired=True")


def _planning_trace_args(args: argparse.Namespace) -> argparse.Namespace:
    trace_args = argparse.Namespace(**vars(args))
    trace_args.planning_trace_root = DEFAULT_PLANNING_TRACE_ROOT
    trace_args.n_permutations = int(args.trace_temporal_permutations)
    trace_args.permutation_block_size = int(args.trace_permutation_block_size)
    trace_args.permutation_batch_size = int(args.trace_permutation_batch_size)
    trace_args.n_jobs = int(args.trace_n_jobs)
    trace_args.exemplar_roi = str(args.trace_exemplar_roi)
    trace_args.diagnostic_noncanonical = False
    trace_args.brain_maps_only = True
    return trace_args


def _planning_trace_subject_exact_inference(
    run_effects: pd.DataFrame,
    trace_module: Any,
) -> pd.DataFrame:
    required = {
        "run_label", "roi", "trace", "conditional_partial_r",
    }
    missing = sorted(required.difference(run_effects.columns))
    if missing:
        raise ValueError(f"Planning-trace run table lacks columns: {missing}")
    runs = sorted(run_effects["run_label"].astype(str).unique())
    rois = sorted(run_effects["roi"].astype(str).unique())
    traces = tuple(str(value) for value in trace_module.TRACE_NAMES)
    if len(runs) != 11 or len(rois) != 110:
        raise ValueError(
            "Participant planning-trace inference requires 11 runs x 110 ROIs"
        )
    columns = [(roi, trace) for roi in rois for trace in traces]
    pivot = run_effects.pivot(
        index="run_label",
        columns=["roi", "trace"],
        values="conditional_partial_r",
    ).reindex(
        index=runs,
        columns=pd.MultiIndex.from_tuples(columns),
    )
    if pivot.isna().any().any():
        raise ValueError("Incomplete run x ROI x trace matrix")
    names = [f"{roi}\t{trace}" for roi, trace in columns]
    result = exact_synchronized_signflip(
        pivot.to_numpy(dtype=float), names
    )
    lookup = {
        f"{roi}\t{trace}": (roi, trace) for roi, trace in columns
    }
    result["roi"] = result["effect"].map(
        lambda key: lookup[str(key)][0]
    )
    result["trace"] = result["effect"].map(
        lambda key: lookup[str(key)][1]
    )
    result = result.drop(columns=["effect"])
    result["trace_label"] = result["trace"].map(trace_module.TRACE_LABELS)
    result["q_bh_family"] = COMMON.fdr_bh(
        result["p_exact_two_sided"].to_numpy(dtype=float)
    )
    result["inference_family"] = "planning_trace_2x110"
    result["population_unit"] = "run"
    return result


def _planning_trace_map_panels(
    frame: pd.DataFrame,
    trace_module: Any,
) -> List[Tuple[str, pd.DataFrame]]:
    return [
        (
            str(trace_module.TRACE_LABELS[trace]),
            frame[frame["trace"].astype(str).eq(str(trace))].copy(),
        )
        for trace in trace_module.TRACE_NAMES
    ]


def _save_planning_trace_brain_map(
    frame: pd.DataFrame,
    path: Path,
    *,
    value_column: str,
    title: str,
    subtitle: str,
    trace_module: Any,
) -> None:
    corrected = frame[
        pd.to_numeric(frame["p_max_abs_t_fwer"], errors="raise") < 0.05
    ].copy()
    NROI.save_effect_brain_maps(
        Path(path),
        panels=_planning_trace_map_panels(corrected, trace_module),
        value_column=value_column,
        title=title,
        subtitle=subtitle,
        metric_label="Conditional partial Spearman r",
        signed=True,
        n_columns=2,
    )


def run_planning_trace_subject_analysis(args: argparse.Namespace) -> None:
    """Run full-trace RSA from persistent caches and publish corrected maps."""
    _validate_unified_v4_cache_roots(args, require_planning_trace=True)
    if bool(args.cache_only):
        raise ValueError(
            "Planning-trace cache construction belongs to 00_brain_cache.py; "
            "07_RSA.py is read-only and does not implement --cache-only."
        )
    trace_module = _load_planning_trace_module()
    trace_args = _planning_trace_args(args)
    output_dir = _planning_trace_namespace(
        Path(args.output_root),
        subject=int(args.subject),
        game=int(args.game),
    )
    original_subject_output_root = trace_module.subject_output_root
    trace_module.subject_output_root = lambda _unused: output_dir
    try:
        trace_module.run_subject_analysis(trace_args)
    finally:
        trace_module.subject_output_root = original_subject_output_root

    result_dir = output_dir / "results"
    run_path = result_dir / "planning_trace_run_effects.csv"
    exact = _planning_trace_subject_exact_inference(
        pd.read_csv(run_path), trace_module
    )
    exact.insert(0, "game", int(args.game))
    exact.insert(0, "subject", int(args.subject))
    exact_path = result_dir / "subject_run_exact_inference.csv"
    exact.to_csv(exact_path, index=False)

    direct_figure_dir = (
        Path(args.output_root)
        / f"sub{int(args.subject):03d}_game{int(args.game)}"
        / "figures"
    )
    brain_path = direct_figure_dir / "planning_trace_corrected_brain_maps.pdf"
    _save_planning_trace_brain_map(
        exact,
        brain_path,
        value_column="mean_effect",
        title=(
            f"{NEURAL_GAME_LABELS[int(args.game)]} · participant "
            f"{int(args.subject)} full planning trace"
        ),
        subtitle=(
            "11-run exact sign flip; max-|t| across "
            "2 trace representations x 110 parcels"
        ),
        trace_module=trace_module,
    )

    for name in (
        "residual_rdm_distance_scatter.png",
        "residual_rdm_distance_scatter.svg",
    ):
        path = output_dir / "figures" / name
        if path.exists():
            path.unlink()
    manifest_path = result_dir / "analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = dict(manifest.get("artifacts", {}))
    artifacts.pop("residual_rdm_distance_scatter.png", None)
    artifacts.pop("residual_rdm_distance_scatter.svg", None)
    artifacts["subject_run_exact_inference.csv"] = {
        "sha256": trace_module.sha256_file(exact_path),
        "rows": int(len(exact)),
        "n_sign_patterns": 2048,
    }
    artifacts["planning_trace_corrected_brain_maps.pdf"] = {
        "path": str(brain_path.resolve()),
        "sha256": trace_module.sha256_file(brain_path),
    }
    manifest["artifacts"] = artifacts
    manifest.pop("exemplar_figure", None)
    manifest["participant_inference"] = (
        "11 run means; all 2048 exact sign patterns; synchronized "
        "max-|t| across 2 trace representations x 110 parcels"
    )
    manifest["participant_figure"] = (
        "corrected Harvard-Oxford parcel effects on a 1-mm MNI template"
    )
    manifest["unified_entrypoint"] = str(Path(__file__).resolve())
    trace_module.atomic_write_json(manifest_path, manifest)
    log(f"Planning-trace subject RSA complete -> {output_dir}")


def _load_unified_planning_trace_subjects(
    output_root: Path,
    game: int,
    trace_module: Any,
) -> Tuple[pd.DataFrame, List[Dict[str, object]], str, Dict[str, object]]:
    frames: List[pd.DataFrame] = []
    input_rows: List[Dict[str, object]] = []
    settings_hash: Optional[str] = None
    settings: Optional[Dict[str, object]] = None
    source_contract = trace_module.current_source_contract()
    for subject in range(1, 7):
        result_dir = _planning_trace_namespace(
            Path(output_root), subject=subject, game=int(game)
        ) / "results"
        effect_path = result_dir / "subject_effects.csv"
        manifest_path = result_dir / "analysis_manifest.json"
        if not effect_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing unified planning-trace input: {result_dir}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("analysis") != trace_module.ANALYSIS_NAME
            or int(manifest.get("schema_version", -1)) != 1
            or int(manifest.get("subject", -1)) != int(subject)
            or int(manifest.get("game", -1)) != int(game)
        ):
            raise ValueError(
                f"Planning-trace manifest identity mismatch: {manifest_path}"
            )
        brain_audit = manifest.get("brain_cache_audit")
        if (
            not isinstance(brain_audit, Mapping)
            or int(brain_audit.get("cache_version", -1)) != 4
            or int(brain_audit.get("process_feature_version", -1)) != 8
            or int(brain_audit.get("neural_feature_schema_version", -1)) != 4
        ):
            raise ValueError(
                f"Planning-trace input is not based on canonical cache 4/8/4: "
                f"{manifest_path}"
            )
        trace_audit = manifest.get("planning_trace_audit")
        trace_source = (
            trace_audit.get("source_contract", {})
            if isinstance(trace_audit, Mapping)
            else {}
        )
        if (
            not isinstance(trace_audit, Mapping)
            or int(trace_audit.get("schema_version", -1)) != 1
            or str(trace_audit.get("representation_version", ""))
            != "v1_concat29_mean29"
            or Path(str(trace_audit.get("trace_root", ""))).resolve()
            != DEFAULT_PLANNING_TRACE_ROOT.resolve()
            or int(trace_source.get("cache_version", -1)) != 4
            or int(trace_source.get("process_feature_version", -1)) != 8
            or int(trace_source.get("neural_feature_schema_version", -1)) != 4
        ):
            raise ValueError(
                f"Planning-trace v1 source contract mismatch: {manifest_path}"
            )
        declared_effect_hash = (
            manifest.get("artifacts", {})
            .get("subject_effects.csv", {})
            .get("sha256")
        )
        if str(declared_effect_hash) != trace_module.sha256_file(effect_path):
            raise ValueError(
                f"Planning-trace subject artifact hash mismatch: {effect_path}"
            )
        if manifest.get("source_contract") != source_contract:
            raise ValueError(
                f"Planning-trace source contract mismatch: {manifest_path}"
            )
        run_contract = manifest.get("run_contract", {})
        if (
            not bool(run_contract.get("canonical_full_v1", False))
            or int(run_contract.get("n_runs", -1)) != 11
            or int(run_contract.get("n_rois", -1)) != 110
        ):
            raise ValueError(f"Noncanonical planning-trace input: {manifest_path}")
        observed_hash = str(manifest.get("settings_sha256", ""))
        observed_settings = manifest.get("settings")
        if (
            not isinstance(observed_settings, dict)
            or trace_module.stable_hash(observed_settings) != observed_hash
        ):
            raise ValueError(f"Planning-trace settings hash mismatch: {manifest_path}")
        if settings_hash is None:
            settings_hash = observed_hash
            settings = observed_settings
        elif settings_hash != observed_hash or settings != observed_settings:
            raise ValueError("Planning-trace subject settings differ within game")
        frame = pd.read_csv(effect_path)
        if (
            set(pd.to_numeric(frame["subject"], errors="raise").astype(int))
            != {int(subject)}
            or set(pd.to_numeric(frame["game"], errors="raise").astype(int))
            != {int(game)}
        ):
            raise ValueError(
                f"Planning-trace subject/game table mismatch: {effect_path}"
            )
        frames.append(frame)
        input_rows.append(
            {
                "subject": subject,
                "game": int(game),
                "subject_effects_path": str(effect_path.resolve()),
                "subject_effects_sha256": trace_module.sha256_file(effect_path),
                "manifest_path": str(manifest_path.resolve()),
                "manifest_sha256": trace_module.sha256_file(manifest_path),
            }
        )
    combined = pd.concat(frames, ignore_index=True)
    trace_module.validate_subject_effect_contract(combined)
    if settings_hash is None or settings is None:
        raise AssertionError("Planning-trace settings were not initialized")
    return combined, input_rows, settings_hash, settings


def _planning_trace_predefined_inference(
    subject_effects: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    predefined = NROI.aggregate_predefined_rois(
        subject_effects,
        value_columns=["mean_conditional_partial_r"],
        grouping_columns=["subject", "game", "trace", "trace_label"],
    )
    rows: List[pd.DataFrame] = []
    for (game, trace, trace_label), selected in predefined.groupby(
        ["game", "trace", "trace_label"], sort=False
    ):
        pivot = selected.pivot(
            index="subject",
            columns="roi_key",
            values="mean_conditional_partial_r",
        ).reindex(index=range(1, 7), columns=NROI.ROI_ORDER)
        if pivot.isna().any().any():
            raise ValueError(
                f"Incomplete trace predefined ROI matrix: game={game}, trace={trace}"
            )
        inference = exact_synchronized_signflip(
            pivot.to_numpy(dtype=float), list(NROI.ROI_ORDER)
        ).rename(columns={"effect": "roi_key"})
        inference["q_bh_8roi"] = COMMON.fdr_bh(
            inference["p_exact_two_sided"].to_numpy(dtype=float)
        )
        inference.insert(0, "trace_label", str(trace_label))
        inference.insert(0, "trace", str(trace))
        inference.insert(0, "game", int(game))
        rows.append(inference)
    result = pd.concat(rows, ignore_index=True).merge(
        NROI.predefined_roi_mapping_frame()[
            ["roi_key", "roi_label", "anatomical_label"]
        ].drop_duplicates(),
        on="roi_key",
        how="left",
        validate="many_to_one",
    )
    result["roi_family"] = "eight_literature_defined_bilateral_rois"
    result["selection_timing"] = "literature_defined_secondary_not_preregistered"
    return predefined, result


def _save_planning_trace_predefined_figure(
    subjects: pd.DataFrame,
    inference: pd.DataFrame,
    path: Path,
    *,
    game: int,
    trace: str,
    trace_module: Any,
) -> None:
    if str(trace) not in set(trace_module.TRACE_NAMES):
        raise ValueError(f"Unknown planning trace figure target: {trace}")
    figure, axis = plt.subplots(
        1, 1, figsize=(6.0, 2.8), constrained_layout=False
    )
    colors = plt.cm.tab10(np.linspace(0.05, 0.85, len(NROI.ROI_ORDER)))
    x = np.arange(len(NROI.ROI_ORDER), dtype=float)
    selected = subjects[
        subjects["game"].astype(int).eq(int(game))
        & subjects["trace"].astype(str).eq(str(trace))
    ]
    tested = inference[
        inference["game"].astype(int).eq(int(game))
        & inference["trace"].astype(str).eq(str(trace))
    ]
    data = [
        selected[selected["roi_key"].eq(key)][
            "mean_conditional_partial_r"
        ].to_numpy(dtype=float)
        for key in NROI.ROI_ORDER
    ]
    axis.boxplot(
        data, positions=x, widths=0.58, patch_artist=True, showfliers=False,
        medianprops={"color": "#222426", "linewidth": 0.8},
        whiskerprops={"color": "#6C737A", "linewidth": 0.6},
        capprops={"color": "#6C737A", "linewidth": 0.6},
        boxprops={
            "facecolor": "#E9EDF1", "edgecolor": "#6C737A", "linewidth": 0.6
        },
    )
    for roi_index, roi_key in enumerate(NROI.ROI_ORDER):
        values = data[roi_index]
        offsets = np.linspace(-0.16, 0.16, len(values))
        axis.scatter(
            roi_index + offsets, values, s=12, color=colors[roi_index],
            edgecolor="white", linewidth=0.25, zorder=3,
        )
        row = tested[tested["roi_key"].eq(roi_key)]
        if len(row) != 1:
            raise ValueError(f"Missing predefined trace row: {trace}/{roi_key}")
        if float(row.iloc[0]["p_max_abs_t_fwer"]) < 0.05:
            span = max(float(np.ptp(values)), 0.02)
            axis.text(
                roi_index, max(float(np.max(values)), 0.0) + 0.08 * span,
                "†", ha="center", va="bottom", fontsize=7.0,
            )
    axis.axhline(0, color="#8B9094", linewidth=0.55, linestyle="--")
    data_min = min(0.0, min(float(np.min(values)) for values in data))
    data_max = max(0.0, max(float(np.max(values)) for values in data))
    data_span = max(data_max - data_min, 0.02)
    axis.set_ylim(
        data_min - 0.12 * data_span,
        data_max + 0.24 * data_span,
    )
    axis.set_title(str(trace_module.TRACE_LABELS[trace]), fontsize=7.2)
    axis.grid(axis="y", color="#E1E4E7", linewidth=0.45)
    axis.tick_params(labelsize=5.8)
    axis.set_xticks(x)
    axis.set_xticklabels(
        [NROI.ROI_LABELS[key] for key in NROI.ROI_ORDER],
        rotation=30, ha="right",
    )
    figure.suptitle(
        f"{NEURAL_GAME_LABELS[int(game)]} · full-trace predefined ROIs",
        fontsize=8.2, fontweight="normal", y=0.97,
    )
    figure.subplots_adjust(
        left=0.075, right=0.99, top=0.72, bottom=0.25,
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, facecolor="white")
    plt.close(figure)


def run_planning_trace_group_analysis(args: argparse.Namespace) -> None:
    _validate_unified_v4_cache_roots(args, require_planning_trace=True)
    trace_module = _load_planning_trace_module()
    subject_effects, input_rows, settings_hash, settings = (
        _load_unified_planning_trace_subjects(
            Path(args.output_root), int(args.game), trace_module
        )
    )
    inference = trace_module.group_inference(subject_effects)
    output_dir = _planning_trace_namespace(
        Path(args.output_root), game=int(args.game)
    )
    result_dir = output_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    subject_path = result_dir / "subject_inputs.csv"
    inference_path = result_dir / "group_inference.csv"
    input_path = result_dir / "input_manifest.csv"
    subject_effects.to_csv(subject_path, index=False)
    inference.to_csv(inference_path, index=False)
    pd.DataFrame(input_rows).to_csv(input_path, index=False)

    direct_figure_dir = (
        Path(args.output_root) / f"group_game{int(args.game)}" / "figures"
    )
    brain_path = direct_figure_dir / "planning_trace_corrected_brain_maps.pdf"
    _save_planning_trace_brain_map(
        inference,
        brain_path,
        value_column="mean_effect",
        title=f"{NEURAL_GAME_LABELS[int(args.game)]} · full planning trace",
        subtitle=(
            "Six-subject exact sign flip; max-|t| across "
            "2 trace representations x 110 parcels"
        ),
        trace_module=trace_module,
    )

    predefined_subjects, predefined_inference = (
        _planning_trace_predefined_inference(subject_effects)
    )
    predefined_subject_path = (
        result_dir / "predefined_roi_subject_effects.csv"
    )
    predefined_inference_path = (
        result_dir / "predefined_roi_inference.csv"
    )
    predefined_subjects.to_csv(predefined_subject_path, index=False)
    predefined_inference.to_csv(predefined_inference_path, index=False)
    predefined_figure_specs = (
        (
            "planning_concat29",
            "planning_trace_predefined_roi_effects_concat29.pdf",
        ),
        (
            "planning_mean29",
            "planning_trace_predefined_roi_effects_mean29.pdf",
        ),
    )
    legacy_predefined_figure = (
        direct_figure_dir / "planning_trace_predefined_roi_effects.pdf"
    )
    legacy_predefined_figure.unlink(missing_ok=True)
    predefined_figure_paths: Dict[str, Path] = {}
    for trace, filename in predefined_figure_specs:
        figure_path = direct_figure_dir / filename
        _save_planning_trace_predefined_figure(
            predefined_subjects,
            predefined_inference,
            figure_path,
            game=int(args.game),
            trace=trace,
            trace_module=trace_module,
        )
        predefined_figure_paths[filename] = figure_path
    anatomy_path = Path(args.output_root) / "figures" / "predefined_roi_anatomy.pdf"
    NROI.save_predefined_roi_anatomy(anatomy_path)

    manifest = {
        "analysis": "07_RSA unified planning-trace RSA group",
        "analysis_version": trace_module.ANALYSIS_VERSION,
        "schema_version": 1,
        "aggregation": "group",
        "game": int(args.game),
        "source_contract": trace_module.current_source_contract(),
        "unified_entrypoint_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "settings": settings,
        "settings_sha256": settings_hash,
        "n_subjects": 6,
        "n_rois": 110,
        "n_traces": 2,
        "n_sign_patterns": 64,
        "correction": "two-sided exact synchronized max-|t| across 2x110",
        "predefined_roi_inference": (
            "equal left/right parcel mean; exact 64 subject sign flips; "
            "max-|t| across eight ROIs separately for each trace"
        ),
        "artifacts": {
            "subject_inputs.csv": {
                "sha256": trace_module.sha256_file(subject_path),
                "rows": int(len(subject_effects)),
            },
            "group_inference.csv": {
                "sha256": trace_module.sha256_file(inference_path),
                "rows": int(len(inference)),
            },
            "input_manifest.csv": {
                "sha256": trace_module.sha256_file(input_path),
                "rows": int(len(input_rows)),
            },
            "planning_trace_corrected_brain_maps.pdf": {
                "path": str(brain_path.resolve()),
                "sha256": trace_module.sha256_file(brain_path),
            },
            "predefined_roi_subject_effects.csv": {
                "sha256": trace_module.sha256_file(predefined_subject_path),
                "rows": int(len(predefined_subjects)),
            },
            "predefined_roi_inference.csv": {
                "sha256": trace_module.sha256_file(predefined_inference_path),
                "rows": int(len(predefined_inference)),
            },
            **{
                filename: {
                    "path": str(figure_path.resolve()),
                    "sha256": trace_module.sha256_file(figure_path),
                }
                for filename, figure_path in predefined_figure_paths.items()
            },
            "predefined_roi_anatomy.pdf": {
                "path": str(anatomy_path.resolve()),
                "sha256": trace_module.sha256_file(anatomy_path),
            },
        },
    }
    trace_module.atomic_write_json(
        result_dir / "analysis_manifest.json", manifest
    )
    log(f"Planning-trace group RSA complete -> {output_dir}")


def run_planning_trace_paired_analysis(args: argparse.Namespace) -> None:
    _validate_unified_v4_cache_roots(args, require_planning_trace=True)
    trace_module = _load_planning_trace_module()
    game1, rows1, settings_hash1, settings1 = (
        _load_unified_planning_trace_subjects(
            Path(args.output_root), 1, trace_module
        )
    )
    game2, rows2, settings_hash2, settings2 = (
        _load_unified_planning_trace_subjects(
            Path(args.output_root), 2, trace_module
        )
    )
    if settings_hash1 != settings_hash2 or settings1 != settings2:
        raise ValueError("Planning-trace settings differ between games")
    paired = trace_module.build_paired_effects(game1, game2)
    inference = trace_module.paired_inference(paired)
    output_dir = _planning_trace_namespace(
        Path(args.output_root), paired=True
    )
    result_dir = output_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    paired_path = result_dir / "paired_subject_effects.csv"
    inference_path = result_dir / "paired_game_inference.csv"
    input_path = result_dir / "input_manifest.csv"
    paired.to_csv(paired_path, index=False)
    inference.to_csv(inference_path, index=False)
    pd.DataFrame([*rows1, *rows2]).to_csv(input_path, index=False)
    trace_module.atomic_write_json(
        result_dir / "analysis_manifest.json",
        {
            "analysis": "07_RSA unified paired-game planning-trace RSA",
            "analysis_version": trace_module.ANALYSIS_VERSION,
            "source_contract": trace_module.current_source_contract(),
            "settings": settings1,
            "settings_sha256": settings_hash1,
            "contrasts": {
                "across_game_common": "(Game1 + Game2) / 2 within subject",
                "game2_minus_game1": "Game2 - Game1 within subject",
            },
            "n_subjects": 6,
            "n_sign_patterns_per_contrast": 64,
            "correction": (
                "separate exact synchronized max-|t| family per contrast, "
                "each across 2 traces x 110 parcels"
            ),
        },
    )
    log(f"Planning-trace paired-game RSA complete -> {output_dir}")


def _predefined_roi_exact_inference_07(
    subject_values: pd.DataFrame,
) -> pd.DataFrame:
    """Exact subject inference, maxT over the eight ROIs within each effect."""
    rows: List[pd.DataFrame] = []
    for (game, effect, family, statistic), selected in subject_values.groupby(
        ["game", "effect", "inference_family", "statistic"], sort=False
    ):
        pivot = selected.pivot(
            index="subject", columns="roi_key", values="mean_effect"
        ).reindex(index=range(1, 7), columns=NROI.ROI_ORDER)
        if pivot.isna().any().any():
            raise ValueError(
                f"Incomplete predefined ROI matrix: game={game}, effect={effect}"
            )
        inference = exact_synchronized_signflip(
            pivot.to_numpy(dtype=float), list(NROI.ROI_ORDER)
        ).rename(columns={"effect": "roi_key"})
        inference["q_bh_8roi"] = COMMON.fdr_bh(
            inference["p_exact_two_sided"].to_numpy(float)
        )
        inference.insert(0, "statistic", str(statistic))
        inference.insert(0, "inference_family", str(family))
        inference.insert(0, "effect_label", CONTROLLED_EFFECT_LABELS[str(effect)])
        inference.insert(0, "effect", str(effect))
        inference.insert(0, "game", int(game))
        rows.append(inference)
    result = pd.concat(rows, ignore_index=True)
    result = result.merge(
        NROI.predefined_roi_mapping_frame()[["roi_key", "roi_label", "anatomical_label"]]
        .drop_duplicates(),
        on="roi_key",
        how="left",
        validate="many_to_one",
    )
    result["roi_family"] = "eight_literature_defined_bilateral_rois"
    result["selection_timing"] = "literature_defined_secondary_not_preregistered"
    return result


def _save_predefined_roi_effect_figure_07(
    subject_values: pd.DataFrame,
    inference: pd.DataFrame,
    path: Path,
    *,
    game: int,
) -> None:
    """Participant distributions for the eight targeted bilateral ROIs."""
    selected_values = subject_values[
        subject_values["game"].astype(int).eq(int(game))
        & subject_values["statistic"].astype(str).eq("conditional_partial_r")
        & subject_values["effect"].astype(str).isin(NEURAL_MAP_EFFECTS)
    ].copy()
    selected_inference = inference[
        inference["game"].astype(int).eq(int(game))
        & inference["statistic"].astype(str).eq("conditional_partial_r")
        & inference["effect"].astype(str).isin(NEURAL_MAP_EFFECTS)
    ].copy()
    figure, axes = plt.subplots(
        2, 4, figsize=(12.8, 5.0), sharex=True, constrained_layout=False
    )
    colors = plt.cm.tab10(np.linspace(0.05, 0.85, 8))
    x = np.arange(len(NROI.ROI_ORDER), dtype=float)
    for ax, effect in zip(axes.ravel(), NEURAL_MAP_EFFECTS):
        effect_rows = selected_values[selected_values["effect"].eq(effect)]
        effect_test = selected_inference[selected_inference["effect"].eq(effect)]
        data = [
            effect_rows[effect_rows["roi_key"].eq(key)]["mean_effect"].to_numpy(float)
            for key in NROI.ROI_ORDER
        ]
        data_min = min(0.0, min(float(np.min(values)) for values in data))
        data_max = max(0.0, max(float(np.max(values)) for values in data))
        data_span = max(data_max - data_min, 0.02)
        marker_y = data_max + 0.08 * data_span
        ax.boxplot(
            data,
            positions=x,
            widths=0.58,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#222426", "linewidth": 0.8},
            whiskerprops={"color": "#6C737A", "linewidth": 0.6},
            capprops={"color": "#6C737A", "linewidth": 0.6},
            boxprops={"facecolor": "#E9EDF1", "edgecolor": "#6C737A", "linewidth": 0.6},
        )
        for roi_index, roi_key in enumerate(NROI.ROI_ORDER):
            values = data[roi_index]
            offsets = np.linspace(-0.16, 0.16, len(values))
            ax.scatter(
                roi_index + offsets,
                values,
                s=11,
                color=colors[roi_index],
                edgecolor="white",
                linewidth=0.25,
                zorder=3,
            )
            test_row = effect_test[effect_test["roi_key"].eq(roi_key)]
            if len(test_row) != 1:
                raise ValueError(f"Missing targeted inference row: {effect}/{roi_key}")
            p_max = float(test_row.iloc[0]["p_max_abs_t_fwer"])
            p_raw = float(test_row.iloc[0]["p_exact_two_sided"])
            marker = "†" if p_max < 0.05 else ("*" if p_raw < 0.05 else "")
            if marker:
                ax.text(
                    roi_index,
                    marker_y,
                    marker,
                    ha="center",
                    va="bottom",
                    fontsize=7.0,
                )
        ax.set_ylim(
            data_min - 0.12 * data_span,
            data_max + 0.30 * data_span,
        )
        ax.axhline(0, color="#8B9094", linewidth=0.55, linestyle="--")
        ax.set_title(NEURAL_MAP_TITLES[effect], fontsize=7.2, fontweight="normal")
        ax.tick_params(labelsize=5.8)
        ax.grid(axis="y", color="#E1E4E7", linewidth=0.45)
    for ax in axes[-1]:
        ax.set_xticks(x)
        ax.set_xticklabels(
            [NROI.ROI_LABELS[key] for key in NROI.ROI_ORDER],
            rotation=30,
            ha="right",
        )
    figure.subplots_adjust(
        left=0.045, right=0.995, top=0.82, bottom=0.23,
        hspace=0.62, wspace=0.22,
    )
    figure.suptitle(
        f"{NEURAL_GAME_LABELS[int(game)]} · literature-defined ROI effects",
        fontsize=8.2,
        fontweight="normal", y=0.985,
    )
    figure.text(
        0.5,
        0.018,
        "points: participants  |  * raw exact p<.05  |  † max-|t| corrected across 8 ROIs",
        ha="center",
        fontsize=5.7,
        color=NROI.GRAY,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, facecolor="white")
    plt.close(figure)


def _effect_panels(
    frame: pd.DataFrame,
    *,
    value_column: str,
) -> List[Tuple[str, pd.DataFrame]]:
    return [
        (
            NEURAL_MAP_TITLES[effect],
            frame[frame["effect"].astype(str).eq(effect)].copy(),
        )
        for effect in NEURAL_MAP_EFFECTS
    ]


def run_neural_figure_postprocessing(args: argparse.Namespace) -> None:
    """Retired compatibility hook; canonical runs emit maps in-place."""
    del args
    raise RuntimeError(
        "Central neural_figures postprocessing is retired; run canonical "
        "subject/group modes so figures are written in their result folders."
    )

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    COMMON.add_shared_cli_arguments(parser)
    # Group aggregation reads completed subject outputs and has no single
    # subject argument. Subject mode is checked explicitly in main.
    for action in parser._actions:
        if action.dest in {"subject", "game"}:
            action.required = False
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--render-predefined-roi-anatomy",
        action="store_true",
        help=(
            "Render only the shared literature-defined bilateral ROI anatomy "
            "PDF, without fitting RSA or recomputing inference"
        ),
    )
    parser.add_argument(
        "--rsa-family",
        choices=("native", "planning-trace", "both"),
        default="native",
        help=(
            "RSA representation family: process-resolved native variables, "
            "full concat29/mean29 planning trace, or both"
        ),
    )
    parser.add_argument(
        "--analysis-level",
        choices=("auto", "subject", "group", "both"),
        default="auto",
        help=(
            "Execution level. 'auto' preserves legacy subject/aggregation "
            "flags; 'both' runs the requested subject cell and then its game group."
        ),
    )
    parser.add_argument(
        "--planning-trace-root",
        type=Path,
        default=DEFAULT_PLANNING_TRACE_ROOT,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--trace-temporal-permutations",
        type=int,
        default=0,
        help=(
            "Optional diagnostic within-run block permutations for full-trace "
            "RSA (0..200). Exact 2048 run and 64 subject sign flips are primary."
        ),
    )
    parser.add_argument(
        "--trace-permutation-block-size", type=int, default=40
    )
    parser.add_argument(
        "--trace-permutation-batch-size", type=int, default=20
    )
    parser.add_argument(
        "--trace-n-jobs",
        type=int,
        default=1,
        help="Low-concurrency ROI workers for planning-trace RSA",
    )
    parser.add_argument(
        "--trace-exemplar-roi",
        default="right_caudate",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--aggregate-subjects",
        action="store_true",
        help=(
            "Aggregate finished subject_effects.csv files with exact "
            "subject-level inference; runs are never pooled"
        ),
    )
    parser.add_argument(
        "--aggregate-games",
        action="store_true",
        help=(
            "Pair Game1/Game2 within each of subjects 1--6, then test the "
            "common mean and Game2-minus-Game1 with exact subject inference"
        ),
    )
    parser.add_argument(
        "--postprocess-neural-figures",
        action="store_true",
        help=(
            "Deprecated: corrected brain maps and predefined-ROI results are "
            "written directly by canonical subject/group runs; "
            "no central neural_figures directory is created"
        ),
    )
    parser.add_argument(
        "--subjects",
        default="1,2,3,4,5,6",
        help="Comma-separated subject ids for --aggregate-subjects",
    )
    parser.add_argument(
        "--include-ram-control",
        action="store_true",
        help=(
            "Add the high-dimensional RAM RDM; visual, action, history, "
            "drift, and temporal-lag nuisances remain mandatory"
        ),
    )
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--permutation-block-size", type=int, default=40)
    parser.add_argument(
        "--min-pair-lag",
        type=int,
        default=10,
        help="Exclude RDM pairs separated by fewer than this many TRs",
    )
    parser.add_argument("--include-secondary-geometry", action="store_true")
    parser.add_argument(
        "--legacy-composite",
        action="store_true",
        help=(
            "Run the older controlled composite planning-process RSA instead "
            "of the default 02--04 controlled RSA"
        ),
    )
    parser.add_argument(
        "--individual-rdms",
        "--legacy-individual-rdms",
        action="store_true",
        help=(
            "Run the older feature-resolved RSA (explicit legacy mode)"
        ),
    )
    parser.add_argument(
        "--legacy-transition-topology",
        action="store_true",
        help=(
            "Run the former uncontrolled transition/topology RSA and MVPA; "
            "new direct transition decoding belongs to "
            "06_neural_state_space.py"
        ),
    )
    parser.add_argument(
        "--transition-fraction",
        type=float,
        default=0.15,
        help=(
            "Fraction of TRs retained at each tail of the HRF-aligned "
            "NOOP-to-active minus active-to-NOOP contrast"
        ),
    )
    parser.add_argument(
        "--min-transition-trs",
        type=int,
        default=24,
        help="Minimum number of transition-dominant TRs retained per condition",
    )
    parser.add_argument(
        "--top-mvpa-rois",
        type=int,
        default=8,
        help=(
            "Number of highest-balanced-accuracy significant ROIs shown in "
            "the transition-specific recall small multiples"
        ),
    )
    parser.add_argument(
        "--mvpa-figures-only",
        action="store_true",
        help=(
            "Legacy transition/topology only: recompute old MVPA figures; "
            "new MVPA belongs to 06_neural_state_space.py"
        ),
    )
    parser.add_argument(
        "--transition-pattern-permutation-pilot",
        action="store_true",
        help=(
            "Explicit legacy pilot for direct transition-pattern separation; "
            "new direct decoding belongs to 06_neural_state_space.py"
        ),
    )
    parser.add_argument(
        "--cross-game-fdr-bars-only",
        action="store_true",
        help=(
            "Regenerate only the separate Pong and Space Invaders bar "
            "figures for ROIs with BH-FDR q<.05 in both games"
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help=(
            "Number of ROIs processed concurrently within each run. "
            "Permutation seeds and synchronized null draws are unchanged."
        ),
    )
    parser.add_argument(
        "--permutation-batch-size",
        type=int,
        default=100,
        help=(
            "Number of synchronized block permutations evaluated per matrix "
            "batch. This changes memory/speed only, not seeds or statistics."
        ),
    )
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.set_defaults(roi_set="all")
    return parser


def _synthetic_controlled_blocks(
    n_tr: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Synthetic canonical bridge used only by deterministic self-tests."""
    time_axis = np.linspace(-1.0, 1.0, n_tr)
    origin = ((np.arange(n_tr) // 2) % 2).astype(float)
    destination = (np.arange(n_tr) % 2).astype(float)
    switch = (origin != destination).astype(float)
    current_action = np.zeros((n_tr, 6), dtype=float)
    previous_action = np.zeros((n_tr, 6), dtype=float)
    active_identity = 1 + (np.arange(n_tr) % 5)
    lagged_identity = 1 + ((np.arange(n_tr) + 2) % 5)
    current_code = np.where(destination > 0, active_identity, 0)
    previous_code = np.where(origin > 0, lagged_identity, 0)
    current_action[np.arange(n_tr), current_code.astype(int)] = 1.0
    previous_action[np.arange(n_tr), previous_code.astype(int)] = 1.0
    breadth = rng.normal(size=n_tr)
    depth = -0.55 * breadth + rng.normal(scale=0.85, size=n_tr)
    evidence = rng.normal(size=n_tr)
    all29 = evidence + rng.normal(scale=0.03, size=n_tr)
    return {
        "model_prior_policy": rng.normal(size=(n_tr, 5)),
        "model_margin": rng.normal(size=(n_tr, 1)),
        "initial_q_state": rng.normal(size=(n_tr, 1)),
        "revision_primary": rng.normal(size=(n_tr, 1)),
        "evidence_primary": np.column_stack(
            (
                evidence,
                rng.normal(size=(n_tr, 3)),
            )
        ),
        "evidence_all29": np.column_stack(
            (
                all29,
                rng.normal(size=(n_tr, 3)),
            )
        ),
        "geometry_primary": np.column_stack((breadth, depth)),
        "actor_readout": np.column_stack(
            (rng.normal(size=n_tr), rng.normal(size=n_tr), rng.normal(size=n_tr))
        ),
        "decision_event": rng.normal(size=(n_tr, 1)),
        "transition_events": np.column_stack(
            (
                (origin == 0) & (destination == 0),
                (origin == 0) & (destination == 1),
                (origin == 1) & (destination == 0),
                (origin == 1) & (destination == 1),
            )
        ).astype(float),
        "transition_factors": np.column_stack(
            (origin, destination, switch)
        ),
        "visual": rng.normal(size=(n_tr, 8)),
        "ram": rng.normal(size=(n_tr, 7)),
        "current_action": current_action,
        "previous_action": previous_action,
        "task_history": rng.normal(size=(n_tr, 8)),
        "drift": np.column_stack(
            (time_axis, time_axis**2, np.sin(np.pi * time_axis))
        ),
    }


def run_controlled_rsa_self_tests() -> Dict[str, object]:
    """Four acceptance gates for naming/rank, nulls, geometry, and switch."""
    rng = np.random.default_rng(20260814)

    # 1. Stable RDM registry and estimable fixed designs.
    n_tr = 48
    blocks = _synthetic_controlled_blocks(n_tr, rng)
    rdms = build_controlled_rdms(blocks)
    if tuple(rdms) != CONTROLLED_RDM_ORDER:
        raise AssertionError("Controlled RDM registry order mismatch")
    expected_pairs = n_tr * (n_tr - 1) // 2
    if any(len(values) != expected_pairs for values in rdms.values()):
        raise AssertionError("Controlled RDM pair-count mismatch")
    pair_mask = COMMON.pair_mask_min_lag(n_tr, 3)
    designs = prepare_controlled_comparisons(rdms, pair_mask)
    rank_failures = {}
    for effect, design in designs.items():
        full = np.asarray(design["full_design"], dtype=float)
        rank = int(np.linalg.matrix_rank(full))
        if rank != full.shape[1]:
            rank_failures[effect] = (rank, int(full.shape[1]))
    if rank_failures:
        raise AssertionError(f"Controlled design rank failures: {rank_failures}")

    # Fixed old-vs-FWL equivalence gate for observed and null-array scores.
    equivalence_targets = rng.normal(
        size=(13, int(np.sum(pair_mask)))
    )
    equivalence_error = 0.0
    for effect, design in designs.items():
        reduced = np.asarray(design["reduced_design"], dtype=float)
        full = np.asarray(design["full_design"], dtype=float)
        reduced_beta = equivalence_targets @ np.asarray(
            design["reduced_pinv"], dtype=float
        ).T
        full_beta = equivalence_targets @ np.asarray(
            design["full_pinv"], dtype=float
        ).T
        reduced_residual = equivalence_targets - reduced_beta @ reduced.T
        full_residual = equivalence_targets - full_beta @ full.T
        denominator = np.sum(
            (
                equivalence_targets
                - np.mean(equivalence_targets, axis=1, keepdims=True)
            )
            ** 2,
            axis=1,
        )
        old_reduced_r2 = 1.0 - np.sum(
            reduced_residual**2, axis=1
        ) / denominator
        old_full_r2 = 1.0 - np.sum(
            full_residual**2, axis=1
        ) / denominator
        target = np.asarray(design["target"], dtype=float)
        old_target_residual = target - reduced @ (
            np.asarray(design["reduced_pinv"], dtype=float) @ target
        )
        old = {
            "reduced_r2": old_reduced_r2,
            "full_r2": old_full_r2,
            "delta_r2": old_full_r2 - old_reduced_r2,
            "added_beta": full_beta[:, -1],
            "conditional_partial_r": safe_correlation_batch(
                reduced_residual, old_target_residual
            ),
        }
        new_observed = score_controlled_design(
            equivalence_targets[0], design
        )
        for statistic in (
            "reduced_r2",
            "full_r2",
            "delta_r2",
            "added_beta",
            "conditional_partial_r",
        ):
            equivalence_error = max(
                equivalence_error,
                abs(float(new_observed[statistic]) - float(old[statistic][0])),
            )
        new_null = score_controlled_design_batch(
            equivalence_targets[1:], design
        )
        for statistic in (
            "delta_r2",
            "added_beta",
            "conditional_partial_r",
        ):
            equivalence_error = max(
                equivalence_error,
                float(
                    np.max(
                        np.abs(
                            np.asarray(new_null[statistic])
                            - np.asarray(old[statistic][1:])
                        )
                    )
                ),
            )
    if equivalence_error > 5e-11:
        raise AssertionError(
            f"FWL sufficient-statistic mismatch: {equivalence_error}"
        )

    # 2. Fast synchronized block permutation equals literal square reindexing.
    permutation_n_tr = 32
    permutation_pairs = permutation_n_tr * (permutation_n_tr - 1) // 2
    brain_rdm = rng.normal(size=permutation_pairs)
    permutation_mask = COMMON.pair_mask_min_lag(permutation_n_tr, 4)
    orders = synchronized_block_orders(
        permutation_n_tr, 12, 7, seed=991
    )
    fast = ranked_permuted_brain_batch(
        brain_rdm, permutation_n_tr, permutation_mask, orders
    )
    raw_square = COMMON.square_from_condensed(
        brain_rdm, permutation_n_tr
    )
    brute = np.vstack(
        [
            COMMON.rank_zscore(
                COMMON.permuted_condensed(raw_square, order)
            )[permutation_mask]
            for order in orders
        ]
    )
    permutation_error = float(np.max(np.abs(fast - brute)))
    if permutation_error > 1e-12:
        raise AssertionError(
            f"Block-permutation implementation mismatch: {permutation_error}"
        )

    # 3. Injected geometry remains detectable after the full upstream model.
    geometry_design = designs["G_joint"]
    geometry_target = np.asarray(rdms["G_joint"], dtype=float)
    nuisance_component = np.asarray(rdms["E"], dtype=float)
    injected_brain = (
        1.25 * COMMON.rank_zscore(geometry_target)
        + 0.40 * COMMON.rank_zscore(nuisance_component)
        + rng.normal(scale=0.20, size=expected_pairs)
    )
    injected_ranked = COMMON.rank_zscore(injected_brain)[pair_mask]
    injected_score = score_controlled_design(
        injected_ranked, geometry_design
    )
    if (
        injected_score["added_beta"] <= 0
        or injected_score["conditional_partial_r"] < 0.75
        or injected_score["delta_r2"] <= 0
    ):
        raise AssertionError(
            f"Injected conditional geometry was not recovered: {injected_score}"
        )

    # 4. All four transition cells exist; status-switch interaction has
    # variance beyond endpoints and an injected neural geometry is recovered.
    transition_events = np.asarray(blocks["transition_events"], dtype=float)
    if np.any(np.std(transition_events, axis=0) <= 1e-8):
        raise AssertionError(
            "Synthetic status-switch test lacks one of four transition cells"
        )
    switch_target = COMMON.rank_zscore(
        rdms["switch_status_interaction"]
    )[pair_mask]
    endpoint_design = np.column_stack(
        (
            np.ones(int(np.sum(pair_mask))),
            COMMON.rank_zscore(rdms["N_origin_active"])[pair_mask],
            COMMON.rank_zscore(rdms["N_destination_active"])[pair_mask],
        )
    )
    switch_residual = switch_target - endpoint_design @ (
        np.linalg.pinv(endpoint_design) @ switch_target
    )
    switch_residual_ratio = float(
        np.linalg.norm(switch_residual)
        / max(np.linalg.norm(switch_target), 1e-12)
    )
    if switch_residual_ratio < 0.20:
        raise AssertionError(
            "Status-switch interaction has insufficient variance beyond "
            "endpoints: "
            f"{switch_residual_ratio}"
        )
    switch_design = designs["switch_status_interaction"]
    switch_brain = (
        1.3 * COMMON.rank_zscore(rdms["switch_status_interaction"])
        + 0.3 * COMMON.rank_zscore(rdms["N_origin_active"])
        + 0.3 * COMMON.rank_zscore(rdms["N_destination_active"])
        + rng.normal(scale=0.20, size=expected_pairs)
    )
    switch_score = score_controlled_design(
        COMMON.rank_zscore(switch_brain)[pair_mask],
        switch_design,
    )
    if (
        switch_score["added_beta"] <= 0
        or switch_score["conditional_partial_r"] < 0.75
    ):
        raise AssertionError(
            f"Injected status-switch interaction not recovered: {switch_score}"
        )

    # Exact synchronized max-|t| sanity: exhaustive 2^6, no Monte Carlo.
    signflip = exact_synchronized_signflip(
        np.column_stack(
            (
                np.linspace(0.2, 0.7, 6),
                np.asarray((-0.4, 0.2, -0.1, 0.3, -0.2, 0.1)),
            )
        ),
        ("injected_positive", "mixed"),
    )
    if not np.all(signflip["n_sign_patterns"].to_numpy() == 64):
        raise AssertionError("Exact sign-flip did not enumerate 2^6 patterns")

    # Subject and group layers must retain equal run/subject weighting.
    synthetic_rows = []
    for run_index in range(6):
        for roi_index, roi in enumerate(("roi_a", "roi_b")):
            for effect_index, effect in enumerate(CONTROLLED_EFFECT_ORDER):
                value = (
                    0.08
                    + 0.01 * run_index
                    + 0.002 * roi_index
                    + 0.001 * effect_index
                )
                synthetic_rows.append(
                    {
                        "run_label": f"run_{run_index}",
                        "roi": roi,
                        "effect": effect,
                        "inference_family": controlled_effect_family(effect),
                        "added_beta": value,
                        "conditional_partial_r": 0.8 * value,
                        "delta_r2": value**2,
                    }
                )
    synthetic_run_effects = pd.DataFrame(synthetic_rows)
    synthetic_subject = summarize_controlled_subject(
        synthetic_run_effects
    )
    expected_subject_rows = 2 * 2 * len(CONTROLLED_EFFECT_ORDER)
    if (
        len(synthetic_subject) != expected_subject_rows
        or not np.all(synthetic_subject["n_units"].to_numpy() == 6)
        or not np.all(
            synthetic_subject["n_sign_patterns"].to_numpy() == 64
        )
    ):
        raise AssertionError("Subject exact-inference grid is not run-equal")
    group_inputs = []
    for subject in range(1, 7):
        frame = synthetic_subject.copy()
        frame.insert(0, "subject", subject)
        frame["mean_effect"] *= 1.0 + 0.02 * subject
        group_inputs.append(frame)
    synthetic_group = summarize_controlled_group(
        pd.concat(group_inputs, ignore_index=True)
    )
    if (
        len(synthetic_group) != expected_subject_rows
        or not np.all(synthetic_group["n_units"].to_numpy() == 6)
        or not np.all(
            synthetic_group["n_sign_patterns"].to_numpy() == 64
        )
    ):
        raise AssertionError("Group exact-inference grid is not subject-equal")
    paired_inputs = []
    for subject in range(1, 7):
        for game in (1, 2):
            frame = synthetic_subject.copy()
            frame.insert(0, "game", game)
            frame.insert(0, "subject", subject)
            frame["mean_effect"] = (
                frame["mean_effect"]
                + 0.01 * subject
                + (0.04 if game == 2 else 0.0)
            )
            paired_inputs.append(frame)
    synthetic_paired, synthetic_cross_game = (
        summarize_controlled_cross_game(
            pd.concat(paired_inputs, ignore_index=True)
        )
    )
    if (
        len(synthetic_paired)
        != 6 * 2 * len(CONTROLLED_EFFECT_ORDER) * 2
        or len(synthetic_cross_game) != 2 * expected_subject_rows
        or not np.all(
            synthetic_cross_game["n_units"].to_numpy(dtype=int) == 6
        )
        or not np.all(
            synthetic_cross_game["n_sign_patterns"].to_numpy(dtype=int)
            == 64
        )
    ):
        raise AssertionError(
            "Paired-game inference did not preserve six subject pairs"
        )
    observed_difference = synthetic_cross_game[
        synthetic_cross_game["cross_game_contrast"]
        .astype(str)
        .eq("game2_minus_game1")
    ]["mean_effect"].to_numpy(dtype=float)
    if not np.allclose(observed_difference, 0.04, atol=1e-12):
        raise AssertionError(
            "Paired-game difference was not computed within subject"
        )

    # Canonical aggregation must accept only the complete 11-run x 110-ROI
    # contract and reject an otherwise well-formed bounded smoke output.
    synthetic_roi_names = [f"synthetic_roi_{index:03d}" for index in range(110)]
    synthetic_roi_digest = _controlled_roi_names_sha256(synthetic_roi_names)
    synthetic_labels = [f"synthetic_run_{index:02d}" for index in range(11)]
    synthetic_source_signature = {"synthetic": True, "schema": 3}
    synthetic_token = COMMON.signature_token(synthetic_source_signature)
    synthetic_cache_runs = [
        {
            "run_label": label,
            "n_tr": 480,
            "roi_names": synthetic_roi_names,
        }
        for label in synthetic_labels
    ]
    synthetic_feature_runs = [
        {"run_label": label} for label in synthetic_labels
    ]
    synthetic_run_contract = [
        {
            "run_label": label,
            "subject": 1,
            "session": index // 3 + 1,
            "block": index + 1,
            "game": 1,
            "n_tr": 480,
            "cache_schema": "canonical_00brain_nativehrf_schema4_v8",
            "feature_source_token": synthetic_token,
            "roi_names_sha256": synthetic_roi_digest,
        }
        for index, label in enumerate(synthetic_labels)
    ]
    synthetic_cache = {
        "cache_version": 4,
        "process_feature_version": 8,
        "neural_feature_schema_version": 4,
        "subject": 1,
        "game": 1,
        "feature_source_token": synthetic_token,
        "roi_set": "all",
        "max_roi_voxels": 0,
        "roi_names": synthetic_roi_names,
        "im_vectors_used": False,
        "include_topology": False,
        "runs": synthetic_cache_runs,
        "feature_source": {
            "subject": 1,
            "game": 1,
            "n_runs": 11,
            "schema_version": 4,
            "04_script_sha256": CONTROLLED_EXPECTED_04_SCRIPT_SHA256,
            "source_signature": synthetic_source_signature,
            "source_token": synthetic_token,
            "im_vectors_used": False,
            "raw_topology_used": False,
            "runs": synthetic_feature_runs,
        },
    }
    canonical_settings = {
        **CONTROLLED_CANONICAL_FIXED_SETTINGS,
        "roi_names_sha256": synthetic_roi_digest,
    }
    canonical_source = _controlled_cache_source(synthetic_cache)
    canonical_manifest = {
        **controlled_estimator_provenance(),
        "subject": 1,
        "game": 1,
        "n_runs": 11,
        "n_rois": 110,
        "controlled_canonical_full_analysis": True,
        "controlled_settings": canonical_settings,
        "controlled_settings_sha256": _controlled_json_sha256(
            canonical_settings
        ),
        "controlled_cache_source": canonical_source,
        "controlled_cache_source_sha256": _controlled_json_sha256(
            canonical_source
        ),
        "controlled_run_contract": synthetic_run_contract,
        "controlled_run_contract_sha256": _controlled_json_sha256(
            synthetic_run_contract
        ),
        "cache_manifest": synthetic_cache,
    }
    validate_controlled_subject_provenance(
        canonical_manifest, Path("<synthetic-canonical>")
    )
    smoke_manifest = json.loads(json.dumps(canonical_manifest))
    smoke_manifest["n_runs"] = 2
    smoke_manifest["controlled_settings"]["n_runs"] = 2
    smoke_manifest["controlled_settings_sha256"] = _controlled_json_sha256(
        smoke_manifest["controlled_settings"]
    )
    try:
        validate_controlled_subject_provenance(
            smoke_manifest, Path("<synthetic-smoke>")
        )
    except ValueError:
        smoke_rejected = True
    else:
        raise AssertionError(
            "Two-run smoke output passed canonical aggregation gate"
        )
    token_manifest = json.loads(json.dumps(canonical_manifest))
    token_manifest["cache_manifest"]["feature_source_token"] = "wrong-token"
    try:
        validate_controlled_subject_provenance(
            token_manifest, Path("<synthetic-token-mismatch>")
        )
    except ValueError:
        token_rejected = True
    else:
        raise AssertionError("Source-token mismatch passed aggregation gate")
    duplicate_manifest = json.loads(json.dumps(canonical_manifest))
    duplicate_label = duplicate_manifest["controlled_run_contract"][0][
        "run_label"
    ]
    duplicate_manifest["controlled_run_contract"][1][
        "run_label"
    ] = duplicate_label
    duplicate_manifest["controlled_run_contract_sha256"] = (
        _controlled_json_sha256(
            duplicate_manifest["controlled_run_contract"]
        )
    )
    try:
        validate_controlled_subject_provenance(
            duplicate_manifest, Path("<synthetic-duplicate-run>")
        )
    except ValueError:
        duplicate_run_rejected = True
    else:
        raise AssertionError("Duplicate run passed aggregation gate")
    return {
        "controlled_rdm_names": list(rdms),
        "controlled_effect_names": list(designs),
        "controlled_full_rank_designs": int(len(designs)),
        "controlled_fwl_max_abs_error": equivalence_error,
        "controlled_estimator_version": CONTROLLED_RSA_ESTIMATOR_VERSION,
        "controlled_estimator_sha256": CONTROLLED_RSA_ESTIMATOR_HASH,
        "controlled_block_permutation_max_abs_error": permutation_error,
        "controlled_injected_geometry_partial_r": float(
            injected_score["conditional_partial_r"]
        ),
        "controlled_switch_residual_ratio": switch_residual_ratio,
        "controlled_injected_switch_partial_r": float(
            switch_score["conditional_partial_r"]
        ),
        "controlled_group_sign_patterns": 64,
        "controlled_subject_summary_rows": int(len(synthetic_subject)),
        "controlled_group_summary_rows": int(len(synthetic_group)),
        "controlled_cross_game_summary_rows": int(
            len(synthetic_cross_game)
        ),
        "controlled_noncanonical_smoke_rejected": bool(smoke_rejected),
        "controlled_source_token_mismatch_rejected": bool(token_rejected),
        "controlled_duplicate_run_rejected": bool(duplicate_run_rejected),
    }


def run_self_test() -> None:
    result = COMMON.self_test()
    if result["rsa_delta_r2"] <= 0:
        raise AssertionError(result)
    rng = np.random.default_rng(17)
    n_tr = 32
    n_pairs = n_tr * (n_tr - 1) // 2
    brain = rng.normal(size=n_pairs)
    reduced = [rng.normal(size=n_pairs) for _ in range(3)]
    added = rng.normal(size=n_pairs)
    comparisons = {"test": (reduced, added)}
    pair_mask = COMMON.pair_mask_min_lag(n_tr, 4)
    prepared = prepare_permutation_designs(comparisons, pair_mask)["test"]
    orders = np.vstack(
        [
            COMMON.block_permutation_indices(n_tr, 7, rng)
            for _ in range(8)
        ]
    )
    raw_square = COMMON.square_from_condensed(brain, n_tr)
    ranked_square = COMMON.square_from_condensed(
        COMMON.rank_zscore(brain), n_tr
    )
    all_rows, all_columns = np.triu_indices(n_tr, k=1)
    pair_rows = all_rows[pair_mask]
    pair_columns = all_columns[pair_mask]
    target_batch = ranked_square[
        orders[:, pair_rows], orders[:, pair_columns]
    ]
    reduced_fast, _ = fixed_design_r2_batch(
        target_batch,
        prepared["reduced_design"],
        prepared["reduced_pinv"],
    )
    full_fast, beta_fast = fixed_design_r2_batch(
        target_batch,
        prepared["full_design"],
        prepared["full_pinv"],
    )
    legacy = np.asarray(
        [
            COMMON.incremental_rdm_fit(
                COMMON.permuted_condensed(raw_square, order),
                reduced,
                added,
                pair_mask=pair_mask,
            )
            for order in orders
        ]
    )
    fit_error = float(
        np.max(
            np.abs(
                np.column_stack(
                    (reduced_fast, full_fast, beta_fast[:, -1])
                )
                - legacy
            )
        )
    )
    neural_residual = residualize_ranked_rdm(brain, reduced)
    added_residual = residualize_ranked_rdm(added, reduced)
    residual_square = COMMON.square_from_condensed(
        neural_residual, n_tr
    )
    partial_fast = safe_correlation_batch(
        residual_square[
            orders[:, pair_rows], orders[:, pair_columns]
        ],
        added_residual[pair_mask],
    )
    partial_legacy = np.asarray(
        [
            safe_correlation(
                COMMON.permuted_condensed(residual_square, order)[
                    pair_mask
                ],
                added_residual[pair_mask],
            )
            for order in orders
        ]
    )
    partial_error = float(np.max(np.abs(partial_fast - partial_legacy)))
    result["rsa_fast_permutation_fit_max_abs_error"] = fit_error
    result["rsa_fast_partial_max_abs_error"] = partial_error
    if fit_error > 1e-10 or partial_error > 1e-12:
        raise AssertionError(result)

    transition = np.zeros((40, 4), dtype=float)
    transition[:10, 0] = np.linspace(0.5, 1.0, 10)
    transition[-10:, 1] = np.linspace(0.5, 1.0, 10)
    selected, selection_audit = select_transition_dominant_trs(
        transition,
        fraction=0.2,
        min_trs=6,
    )
    if (
        len(selected["noop_to_active"]) != 8
        or len(selected["active_to_noop"]) != 8
        or np.intersect1d(
            selected["noop_to_active"], selected["active_to_noop"]
        ).size
    ):
        raise AssertionError(selection_audit)

    class SyntheticBatch:
        paths = np.asarray(
            [
                [0, 1, -1, -1],
                [0, 1, 2, -1],
                [3, 2, -1, -1],
            ],
            dtype=np.int16,
        )
        path_valid = np.asarray([True, True, True])
        depth = np.asarray([2, 3, 2])

    topology = COMMON.canonical_topology_matrix(
        SyntheticBatch(), n_actions=6, max_depth=4
    )
    if topology.shape != (6 * 4 * 3,) or not np.all(np.isfinite(topology)):
        raise AssertionError(
            f"Unexpected canonical topology shape/content: {topology.shape}"
        )

    train_y = np.repeat([0, 1], 20)
    test_y = np.repeat([0, 1], 10)
    train_x = rng.normal(scale=0.2, size=(40, 6))
    test_x = rng.normal(scale=0.2, size=(20, 6))
    train_x[train_y == 1, 0] += 2.0
    test_x[test_y == 1, 0] += 2.0
    decoding_accuracy = nearest_centroid_transition_accuracy(
        train_x, train_y, test_x, test_y
    )
    if decoding_accuracy < 0.9:
        raise AssertionError(
            f"Synthetic transition decoding failed: {decoding_accuracy}"
        )
    result["transition_selected_trs_per_condition"] = int(
        selection_audit["n_selected_per_condition"]
    )
    result["topology_feature_count_test"] = int(topology.size)
    result["transition_decoding_accuracy_test"] = float(decoding_accuracy)
    result.update(run_controlled_rsa_self_tests())
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.render_predefined_roi_anatomy:
        anatomy_path = (
            Path(args.output_root) / "figures" / "predefined_roi_anatomy.pdf"
        )
        NROI.save_predefined_roi_anatomy(anatomy_path)
        anatomy_sha256 = NROI.sha256_file(anatomy_path)
        for game in (1, 2):
            manifest_path = (
                Path(args.output_root)
                / f"group_game{game}"
                / "planning_trace"
                / "results"
                / "analysis_manifest.json"
            )
            if not manifest_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            artifact = manifest.get("artifacts", {}).get(
                "predefined_roi_anatomy.pdf"
            )
            if isinstance(artifact, dict):
                artifact["path"] = str(anatomy_path.resolve())
                artifact["sha256"] = str(anatomy_sha256)
                COMMON.atomic_write_json(manifest_path, manifest)
        log(f"Predefined ROI anatomy -> {anatomy_path}")
        return
    run_native = args.rsa_family in {"native", "both"}
    run_trace = args.rsa_family in {"planning-trace", "both"}
    run_group_after_subject = False
    if args.analysis_level == "subject":
        if args.aggregate_subjects or args.aggregate_games:
            parser.error(
                "--analysis-level subject cannot be combined with aggregation flags"
            )
    elif args.analysis_level == "group":
        if args.aggregate_games:
            parser.error(
                "--analysis-level group cannot be combined with --aggregate-games"
            )
        if args.subject is not None:
            parser.error(
                "--analysis-level group does not accept --subject"
            )
        args.aggregate_subjects = True
    elif args.analysis_level == "both":
        if args.aggregate_subjects or args.aggregate_games:
            parser.error(
                "--analysis-level both cannot be combined with aggregation flags"
            )
        run_group_after_subject = True

    if not 0 <= int(args.trace_temporal_permutations) <= 200:
        parser.error("--trace-temporal-permutations must be in 0..200")
    if int(args.trace_permutation_block_size) < 2:
        parser.error("--trace-permutation-block-size must be >=2")
    if int(args.trace_permutation_batch_size) < 1:
        parser.error("--trace-permutation-batch-size must be >=1")
    if int(args.trace_n_jobs) < 1:
        parser.error("--trace-n-jobs must be >=1")
    legacy_transition_mode = bool(
        args.legacy_transition_topology
        or args.transition_pattern_permutation_pilot
    )
    legacy_modes = (
        int(bool(args.legacy_composite))
        + int(bool(args.individual_rdms))
        + int(legacy_transition_mode)
    )
    if legacy_modes and args.rsa_family != "native":
        raise ValueError(
            "Legacy RSA modes are native-only and cannot be combined with "
            "--rsa-family planning-trace/both"
        )
    if args.cross_game_fdr_bars_only and args.rsa_family != "native":
        raise ValueError(
            "--cross-game-fdr-bars-only is a legacy native-only option"
        )
    if args.postprocess_neural_figures:
        parser.error(
            "--postprocess-neural-figures is retired. Canonical subject and "
            "group runs now write corrected brain maps directly to their "
            "figures directories."
        )
    if args.aggregate_subjects:
        if args.aggregate_games:
            raise ValueError(
                "Choose one of --aggregate-subjects or --aggregate-games"
            )
        if legacy_modes or args.cross_game_fdr_bars_only:
            raise ValueError(
                "--aggregate-subjects cannot be combined with legacy modes"
            )
        if args.game is None:
            parser.error("--game is required with --aggregate-subjects")
        if run_native:
            _validate_unified_v4_cache_roots(
                args, require_planning_trace=False
            )
            run_controlled_group_analysis(args)
        if run_trace:
            run_planning_trace_group_analysis(args)
        return
    if args.aggregate_games:
        if legacy_modes or args.cross_game_fdr_bars_only:
            raise ValueError(
                "--aggregate-games cannot be combined with legacy modes"
            )
        if args.game is not None:
            raise ValueError(
                "--aggregate-games pairs both games; do not pass --game"
            )
        if run_native:
            _validate_unified_v4_cache_roots(
                args, require_planning_trace=False
            )
            run_controlled_cross_game_analysis(args)
        if run_trace:
            run_planning_trace_paired_analysis(args)
        return
    if args.subject is None:
        parser.error(
            "--subject is required unless a group aggregation mode is used"
        )
    if args.game is None:
        parser.error(
            "--game is required unless --aggregate-games is used"
        )
    if args.cross_game_fdr_bars_only:
        outputs = generate_cross_game_common_fdr_bars(
            args.output_root, args.subject
        )
        if not outputs:
            raise FileNotFoundError(
                "Both games' transition-pattern summaries and "
                "discriminant scores are required"
            )
        for output in outputs:
            log(f"Cross-game common-FDR bar figure -> {output}")
        return
    if args.n_permutations < 0:
        raise ValueError("--n-permutations must be non-negative")
    if args.permutation_block_size < 2:
        raise ValueError("--permutation-block-size must be >= 2")
    if args.min_pair_lag < 0:
        raise ValueError("--min-pair-lag must be non-negative")
    if args.n_jobs < 1:
        raise ValueError("--n-jobs must be >= 1")
    if args.permutation_batch_size < 1:
        raise ValueError("--permutation-batch-size must be >= 1")
    if not 0 < args.transition_fraction <= 0.5:
        raise ValueError("--transition-fraction must be in (0, 0.5]")
    if args.min_transition_trs < 4:
        raise ValueError("--min-transition-trs must be >= 4")
    if args.top_mvpa_rois < 1:
        raise ValueError("--top-mvpa-rois must be >= 1")
    if legacy_modes > 1:
        raise ValueError(
            "Choose only one legacy analysis family"
        )
    if args.mvpa_figures_only and not args.legacy_transition_topology:
        raise ValueError(
            "--mvpa-figures-only is a legacy transition/topology option. "
            "For current MVPA use 06_neural_state_space.py."
        )

    if legacy_transition_mode:
        args.include_topology_cache = True
        if not args.roi_include:
            args.roi_set = "all"
        run_specs, _, cache_audit = COMMON.shared_prepare_from_args(args)
        if args.transition_pattern_permutation_pilot:
            run_transition_pattern_permutation_pilot(
                args, run_specs, cache_audit
            )
            return
        run_transition_topology_analysis(args, run_specs, cache_audit)
        return

    if not (args.legacy_composite or args.individual_rdms):
        if run_native:
            _validate_unified_v4_cache_roots(
                args, require_planning_trace=False
            )
            args.include_topology_cache = False
            run_specs, _, cache_audit = COMMON.shared_prepare_from_args(args)
            run_controlled_subject_analysis(
                args, run_specs, cache_audit
            )
        if run_trace:
            run_planning_trace_subject_analysis(args)
        if run_group_after_subject:
            if run_native:
                run_controlled_group_analysis(args)
            if run_trace:
                run_planning_trace_group_analysis(args)
        return

    args.legacy_process_cache = True
    run_specs, _, cache_audit = COMMON.shared_prepare_from_args(args)
    base_output_dir = (
        args.output_root / f"sub{args.subject:03d}_game{args.game}"
    )
    output_dir = (
        base_output_dir / "legacy_individual"
        if args.individual_rdms
        else base_output_dir / "legacy_composite"
    )
    result_dir = output_dir / "results"
    figure_dir = output_dir / "figures"
    rdm_dir = output_dir / "rdms"
    for path in (result_dir, figure_dir, rdm_dir):
        path.mkdir(parents=True, exist_ok=True)
    if args.cache_only:
        log(f"Cache-only complete: {args.cache_root}")
        return

    roi_names = COMMON.common_roi_names(run_specs)
    rows: List[Dict[str, object]] = []
    null_store: Dict[str, np.ndarray] = {}
    rdm_manifest: List[Dict[str, object]] = []
    for run_index, run in enumerate(run_specs, start=1):
        log(f"[run {run_index}/{len(run_specs)}] {run.run_label}")
        rdms = build_model_rdms(
            run,
            args.include_secondary_geometry,
            include_individual=args.individual_rdms,
        )
        pair_mask = COMMON.pair_mask_min_lag(
            run.n_tr, args.min_pair_lag
        )
        comparison_definitions = comparison_predictors(
            rdms,
            individual=args.individual_rdms,
        )
        permutation_designs = prepare_permutation_designs(
            comparison_definitions, pair_mask
        )
        COMMON.atomic_save_npz(
            rdm_dir / f"model_rdms_{run.run_label}.npz",
            **{name: np.asarray(values, dtype=np.float32) for name, values in rdms.items()},
        )
        for name, values in rdms.items():
            rdm_manifest.append(
                {
                    "run_label": run.run_label,
                    "rdm": name,
                    "n_pairs": int(len(values)),
                    "n_tr": int(run.n_tr),
                }
            )
        def analyze_one_roi(roi_name: str):
            return analyze_roi_run(
                run,
                roi_name,
                rdms,
                min_pair_lag=args.min_pair_lag,
                n_permutations=args.n_permutations,
                block_size=args.permutation_block_size,
                random_state=args.random_state,
                permutation_designs=permutation_designs,
                comparison_definitions=comparison_definitions,
                permutation_batch_size=args.permutation_batch_size,
            )

        if args.n_jobs == 1:
            roi_results = [
                analyze_one_roi(roi_name) for roi_name in roi_names
            ]
        else:
            roi_results_by_index = {}
            completed_count = 0
            with ThreadPoolExecutor(
                max_workers=min(args.n_jobs, len(roi_names))
            ) as executor:
                future_to_index = {
                    executor.submit(analyze_one_roi, roi_name): roi_index
                    for roi_index, roi_name in enumerate(roi_names)
                }
                for future in as_completed(future_to_index):
                    roi_index = future_to_index[future]
                    roi_results_by_index[roi_index] = future.result()
                    completed_count += 1
                    if (
                        completed_count % 10 == 0
                        or completed_count == len(roi_names)
                    ):
                        log(
                            f"  ROI {completed_count}/{len(roi_names)} "
                            f"(parallel workers={min(args.n_jobs, len(roi_names))})"
                        )
            roi_results = [
                roi_results_by_index[index]
                for index in range(len(roi_names))
            ]
        if args.n_jobs == 1:
            log(f"  ROI {len(roi_names)}/{len(roi_names)}")
        for roi_rows, roi_nulls in roi_results:
            rows.extend(roi_rows)
            null_store.update(roi_nulls)

    run_effects = add_within_run_max_fwer(pd.DataFrame(rows), null_store)
    if not run_effects.empty:
        run_effects["q_within_run_delta"] = (
            run_effects.groupby("run_label", group_keys=False)[
                "p_block_delta_one_sided"
            ].transform(lambda values: COMMON.fdr_bh(values.to_numpy(dtype=float)))
        )
        run_effects["q_within_run_beta"] = (
            run_effects.groupby("run_label", group_keys=False)[
                "p_block_beta_one_sided"
            ].transform(lambda values: COMMON.fdr_bh(values.to_numpy(dtype=float)))
        )
        run_effects["q_within_run_conditional_partial"] = (
            run_effects.groupby("run_label", group_keys=False)[
                "p_block_conditional_partial_one_sided"
            ].transform(lambda values: COMMON.fdr_bh(values.to_numpy(dtype=float)))
        )
    summary = aggregate_run_effects(run_effects)
    run_effects.to_csv(result_dir / "rsa_run_effects.csv", index=False)
    summary.to_csv(result_dir / "rsa_across_run_summary.csv", index=False)
    pd.DataFrame(rdm_manifest).to_csv(result_dir / "rdm_manifest.csv", index=False)
    if null_store:
        COMMON.atomic_save_npz(result_dir / "rsa_block_permutation_nulls.npz", **null_store)
    plot_delta_heatmap(summary, figure_dir / "rsa_incremental_delta_r2_heatmap.png")
    plot_run_consistency(summary, figure_dir / "rsa_top_roi_run_consistency.png")

    manifest = {
        "analysis": (
            "individual planning-feature multiple-regression RSA"
            if args.individual_rdms
            else "planning-process multiple-regression RSA"
        ),
        "subject": args.subject,
        "game": args.game,
        "n_runs": len(run_specs),
        "n_rois": len(roi_names),
        "comparisons": {
            name: COMPARISON_LABELS[name]
            for name in comparison_definitions
        },
        "controls": [
            "RAM",
            "low-level visual descriptor",
            "pre-search Q state",
            "current/previous action",
            "nuisance",
            "temporal lag",
        ],
        "n_permutations": args.n_permutations,
        "n_jobs": args.n_jobs,
        "permutation_batch_size": args.permutation_batch_size,
        "block_size_tr": args.permutation_block_size,
        "min_pair_lag_tr": args.min_pair_lag,
        "include_secondary_geometry": bool(args.include_secondary_geometry),
        "individual_rdms": bool(args.individual_rdms),
        "non_destructive_output_namespace": (
            "legacy_individual"
            if args.individual_rdms
            else "legacy_composite"
        ),
        "within_run_multiple_comparison": (
            "synchronized permutation max statistic across ROI x comparison"
        ),
        "conditional_rsa_statistic": (
            "partial correlation after rank-residualizing neural and added RDMs "
            "against the theory-matched reduced model"
        ),
        "temporal_alignment": (
            "behavioral timestamps; exact 1-second averaging; canonical HRF"
        ),
        "population_inference": False,
        "cache_manifest": cache_audit,
    }
    COMMON.atomic_write_json(result_dir / "analysis_manifest.json", manifest)
    summary_lines = [
        (
            f"# sub{args.subject:03d} game{args.game} "
            + (
                "individual planning-feature RSA"
                if args.individual_rdms
                else "planning-process RSA"
            )
        ),
        "",
        f"- Runs: {len(run_specs)}",
        f"- ROIs: {len(roi_names)} ({args.roi_set})",
        f"- Block permutations/run/ROI: {args.n_permutations:,}",
        f"- Excluded near-diagonal RDM pairs: lag < {args.min_pair_lag} TR",
        (
            "- Inference: block-permutation of incremental R² within run; "
            "exact sign-flip of the signed added-RDM beta across runs."
        ),
        (
            "- Primary conditional statistic: residualized partial r with "
            "within-run block null and across-run exact sign-flip."
        ),
        "- Within-run family control: synchronized max-statistic FWER across ROI × contrast.",
        "- Controls include exact-time RAM, low-level visual features, and pre-search Q state.",
        (
            "- Individual mode: Policy axes are mutually adjusted; diversity "
            "and depth are mutually adjusted after Policy State; outcome mean "
            "and dispersion are mutually adjusted after Policy/Search; SR and "
            "VP are tested as an alternative decomposition."
            if args.individual_rdms
            else "- Composite mode: each stage is tested as one joint RDM."
        ),
        "- Population claim: not available until subject-level effects are combined across subjects.",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    log(f"Complete -> {output_dir}")


if __name__ == "__main__":
    main()
