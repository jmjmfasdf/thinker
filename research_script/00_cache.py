#!/usr/bin/env python3
"""Build or incrementally complete the shared non-neural analysis cache.

The raw Thinker trace is large and the historical analysis scripts each
reconstructed overlapping real-step windows.  This script centralizes that
work under ``outputs/00_cache/subXXX_gameY`` and records one source signature
for every component.  Downstream scripts are read-only cache consumers.  By
default, valid components are reused and only missing, stale, or
schema-incompatible components are generated.  ``--overwrite`` is reserved
for an explicit full rebuild.

This cache is intentionally separate from the fMRI cache.  TR alignment, HRF
convolution, visual/RAM features, voxel patterns, and ROI coordinates remain
under ``outputs/00_brain_cache/time_aligned_cache`` and are built only by
``00_brain_cache.py``.

Examples
--------
python research_script/00_cache.py --subject 3 --game 1
python research_script/00_cache.py --subject 3 --game 2 --include-tsne-vectors
python research_script/00_cache.py --subject 3 --game 1 --include-root-state
python research_script/00_cache.py --subject 3 --game 1 --max-files 1 --overwrite
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from shared_analysis_cache import (
    DEFAULT_SHARED_CACHE_ROOT,
    SHARED_CACHE_VERSION,
    cache_paths,
    validate_one_to_one_real_steps,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TRACE_ROOT = ROOT / "test"

COMPONENT_VERSIONS = {
    "actor_policy": 2,
    "persistence_action_geometry": 2,
    "model_control": 1,
    "rollout_value": 1,
    "root_state": 1,
    "representations": 1,
}

ACTOR_POLICY_COLUMNS_V2 = {
    "actor_policy_source_real_step",
    "actor_current_action_margin_start",
    "actor_current_action_uncertainty_start",
    "actor_current_action_probability_start",
    "actor_current_action_mass_logodds_start",
    "actor_current_action_margin_terminal",
    "actor_current_action_uncertainty_terminal",
    "actor_current_action_margin_search",
    "actor_current_action_uncertainty_search",
    "actor_current_action_coordinate_valid",
}

PERSISTENCE_WINDOW_COLUMNS_V2 = {
    "net_value_revision_toward_alternatives_raw",
    "value_revision_event_fraction",
}
PERSISTENCE_EVENT_COLUMNS_V2 = {
    "root_q_before_sampled",
    "root_q_after_sampled",
    "root_n_before_sampled",
    "root_n_after_sampled",
    "directed_value_revision",
    "value_revision_valid",
}

CORE_COMPONENTS = (
    "actor_policy",
    "persistence_action_geometry",
    "model_control",
    "rollout_value",
)
CORE_KEY_COLUMNS = {"run_label", "real_step", "real_step_id"}
ROLLOUT_PREFIX_COLUMNS = {
    "window_index",
    "event_index",
    "return_advantage",
    "event_progress",
    "same_branch_next",
    "branch_switch_next",
    "same_path_deepen_next",
    "reset_after",
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_script(module_name: str, filename: str):
    path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import cache extractor from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, compression="gzip")
    os.replace(temporary, path)


def save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def selected_trace_signature(
    blocks: Mapping[object, Sequence[object]], max_files: Optional[int]
) -> list[dict[str, object]]:
    signature: list[dict[str, object]] = []
    for metas_all in blocks.values():
        metas = list(metas_all[:max_files] if max_files is not None else metas_all)
        for meta in metas:
            stat = meta.path.stat()
            signature.append(
                {
                    "path": str(meta.path.resolve()),
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                }
            )
    return signature


def read_json(path: Path, default: Optional[dict[str, object]] = None) -> dict[str, object]:
    if not path.exists():
        return {} if default is None else dict(default)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def csv_has_columns(path: Path, required: set[str]) -> bool:
    if not path.exists():
        return False
    try:
        columns = set(pd.read_csv(path, nrows=0).columns)
    except (OSError, EOFError, ValueError):
        return False
    return required.issubset(columns)


def parameters_match(
    observed: Mapping[str, object],
    expected: Mapping[str, object],
) -> bool:
    for key, value in expected.items():
        current = observed.get(key)
        if isinstance(value, float):
            try:
                if not np.isclose(float(current), value):
                    return False
            except (TypeError, ValueError):
                return False
        elif current != value:
            return False
    return True


def root_state_files_complete(paths) -> bool:
    if not paths.root_state_index.exists():
        return False
    try:
        index = pd.read_csv(paths.root_state_index)
    except (OSError, EOFError, ValueError):
        return False
    if index.empty or "vector_file" not in index:
        return False
    return all(
        (paths.root_state_dir / str(filename)).exists()
        for filename in index["vector_file"].dropna().unique()
    )


def representation_files_complete(paths) -> bool:
    required = [
        paths.representation_dir / "manifest.json",
        paths.representation_dir / "sample_metadata.csv",
        paths.representation_dir / "tree_reps.npy",
        paths.representation_dir / "im_vectors.npy",
        paths.representation_dir / "im_vp_vectors.npy",
        paths.representation_dir / "tree_topology.npy",
    ]
    return all(path.exists() for path in required)


def recover_manifestless_core(
    paths,
    *,
    args,
    sessions: Optional[set[int]],
    source_signature: Sequence[Mapping[str, object]],
) -> Optional[dict[str, object]]:
    """Adopt a complete core cache left without a top-level manifest.

    Older runs wrote the manifest only after the optional root-state cache.
    This strict recovery path prevents that optional failure from forcing an
    unnecessary raw-trace rebuild.
    """
    if paths.manifest.exists() or not paths.root.exists():
        return None

    required_files = (
        paths.root_policy_metrics,
        paths.actor_metrics,
        paths.actor_qa,
        paths.persistence_windows,
        paths.persistence_bouts,
        paths.persistence_events,
        paths.persistence_qa,
        paths.model_control_windows,
        paths.model_control_qa,
        paths.rollout_windows,
        paths.rollout_prefix_events,
        paths.rollout_qa,
    )
    if not all(path.is_file() and path.stat().st_size > 0 for path in required_files):
        return None

    if not all(
        (
            csv_has_columns(paths.root_policy_metrics, CORE_KEY_COLUMNS),
            csv_has_columns(
                paths.actor_metrics,
                CORE_KEY_COLUMNS | ACTOR_POLICY_COLUMNS_V2,
            ),
            csv_has_columns(
                paths.persistence_windows,
                CORE_KEY_COLUMNS | PERSISTENCE_WINDOW_COLUMNS_V2,
            ),
            csv_has_columns(
                paths.persistence_events,
                CORE_KEY_COLUMNS | PERSISTENCE_EVENT_COLUMNS_V2,
            ),
            csv_has_columns(paths.persistence_bouts, CORE_KEY_COLUMNS),
            csv_has_columns(paths.model_control_windows, CORE_KEY_COLUMNS),
            csv_has_columns(paths.rollout_windows, CORE_KEY_COLUMNS),
        )
    ):
        return None

    actor_qa = read_json(paths.actor_qa)
    persistence_qa = read_json(paths.persistence_qa)
    model_qa = read_json(paths.model_control_qa)
    rollout_qa = read_json(paths.rollout_qa)
    expected_qa_versions = (
        (actor_qa, "analysis_cache_version", 6),
        (persistence_qa, "schema_version", 4),
        (model_qa, "schema_version", 2),
        (rollout_qa, "schema_version", 2),
    )
    try:
        versions_match = all(
            int(qa.get(field, -1)) == expected
            for qa, field, expected in expected_qa_versions
        )
    except (TypeError, ValueError):
        versions_match = False
    if not versions_match:
        return None
    # These two legacy actor-extraction parameters were not stored in QA.
    # Only adopt their known default configuration; non-default requests must
    # be rebuilt from raw traces rather than guessed.
    if int(args.passive_action) != 0 or not np.isclose(float(args.q_temperature), 1.0):
        return None
    expected_signature = list(source_signature)
    if model_qa.get("trace_signature") != expected_signature:
        return None
    if rollout_qa.get("trace_signature") != expected_signature:
        return None
    if not parameters_match(
        persistence_qa,
        {
            "policy_source": "actor",
            "policy_temperature": float(args.policy_temperature),
            "root_reference_requested": args.root_reference,
        },
    ):
        return None
    if not parameters_match(
        model_qa,
        {
            "voluntary_reset_code": int(args.voluntary_reset_code),
            "forced_reset_code": int(args.forced_reset_code),
        },
    ):
        return None
    if actor_qa.get("coordinate_variant") != (
        "source_state_actor_policy_shifted_one_real_step_v3"
    ):
        return None

    try:
        with np.load(paths.rollout_prefix_events, allow_pickle=False) as archive:
            if not ROLLOUT_PREFIX_COLUMNS.issubset(archive.files):
                return None
            lengths = {int(archive[name].shape[0]) for name in ROLLOUT_PREFIX_COLUMNS}
    except (OSError, EOFError, ValueError, KeyError):
        return None
    if len(lengths) != 1:
        return None
    expected_pairs = int(rollout_qa.get("n_prefix_pairs", -1))
    if expected_pairs < 0 or lengths != {expected_pairs}:
        return None

    log(
        "Validated a manifest-less core cache against its schemas, QA "
        "parameters, prefix arrays, and current trace signatures; adopting "
        "it without raw-trace recomputation"
    )
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    return {
        "shared_cache_version": SHARED_CACHE_VERSION,
        "component_versions": {
            name: COMPONENT_VERSIONS[name] for name in CORE_COMPONENTS
        },
        "subject": int(args.subject),
        "game": int(args.game),
        "sessions": sorted(sessions) if sessions is not None else None,
        "max_files_per_run": args.max_files,
        "trace_root": str(args.trace_root.resolve()),
        "source_files": expected_signature,
        "parameters": {
            "passive_action": int(args.passive_action),
            "q_temperature": float(args.q_temperature),
            "policy_temperature": float(args.policy_temperature),
            "voluntary_reset_code": int(args.voluntary_reset_code),
            "forced_reset_code": int(args.forced_reset_code),
            "persistence_policy_source": "actor",
            "root_reference": args.root_reference,
            "tsne_vectors_included": False,
            "root_state_included": False,
            "representation_max_points": None,
            "random_state": int(args.random_state),
        },
        "components": {
            "actor_policy": actor_qa,
            "persistence_action_geometry": persistence_qa,
            "model_control": model_qa,
            "rollout_value": rollout_qa,
            "root_state": None,
            "representations": None,
        },
        "created_at": now,
        "updated_at": now,
        "recovered_manifestless_core": True,
    }


def cache_identity_matches(
    manifest: Mapping[str, object],
    *,
    args,
    sessions: Optional[set[int]],
    source_signature: Sequence[Mapping[str, object]],
) -> bool:
    expected_sessions = sorted(sessions) if sessions is not None else None
    return bool(manifest) and all(
        (
            manifest.get("subject") == int(args.subject),
            manifest.get("game") == int(args.game),
            manifest.get("sessions") == expected_sessions,
            manifest.get("max_files_per_run") == args.max_files,
            manifest.get("trace_root") == str(args.trace_root.resolve()),
            manifest.get("source_files") == list(source_signature),
        )
    )


def component_validity(
    paths,
    manifest: Mapping[str, object],
    *,
    identity_matches: bool,
    args,
) -> dict[str, bool]:
    """Return whether each independently reproducible component is reusable."""
    parameters = manifest.get("parameters", {})
    if not isinstance(parameters, Mapping):
        parameters = {}
    versions = manifest.get("component_versions", {})
    if not isinstance(versions, Mapping):
        versions = {}
    legacy_cache = int(manifest.get("shared_cache_version", 0)) == 1

    def version_ok(name: str) -> bool:
        observed = versions.get(name)
        if observed is None and legacy_cache:
            observed = 1
        try:
            return int(observed) == COMPONENT_VERSIONS[name]
        except (TypeError, ValueError):
            return False

    actor_files = all(
        path.exists()
        for path in (
            paths.root_policy_metrics,
            paths.actor_metrics,
            paths.actor_qa,
        )
    )
    persistence_files = all(
        path.exists()
        for path in (
            paths.persistence_windows,
            paths.persistence_bouts,
            paths.persistence_events,
            paths.persistence_qa,
        )
    )
    model_files = paths.model_control_windows.exists() and paths.model_control_qa.exists()
    rollout_files = all(
        path.exists()
        for path in (
            paths.rollout_windows,
            paths.rollout_prefix_events,
            paths.rollout_qa,
        )
    )
    representation_files = representation_files_complete(paths)

    validity = {
        "actor_policy": bool(
            identity_matches
            and version_ok("actor_policy")
            and actor_files
            and csv_has_columns(paths.actor_metrics, ACTOR_POLICY_COLUMNS_V2)
            and parameters_match(
                parameters,
                {
                    "passive_action": int(args.passive_action),
                    "q_temperature": float(args.q_temperature),
                    "policy_temperature": float(args.policy_temperature),
                },
            )
        ),
        "persistence_action_geometry": bool(
            identity_matches
            and version_ok("persistence_action_geometry")
            and persistence_files
            and csv_has_columns(
                paths.persistence_windows,
                PERSISTENCE_WINDOW_COLUMNS_V2,
            )
            and csv_has_columns(
                paths.persistence_events,
                PERSISTENCE_EVENT_COLUMNS_V2,
            )
            and parameters_match(
                parameters,
                {
                    "policy_temperature": float(args.policy_temperature),
                    "persistence_policy_source": "actor",
                    "root_reference": args.root_reference,
                },
            )
        ),
        "model_control": bool(
            identity_matches
            and version_ok("model_control")
            and model_files
            and parameters_match(
                parameters,
                {
                    "voluntary_reset_code": int(args.voluntary_reset_code),
                    "forced_reset_code": int(args.forced_reset_code),
                },
            )
        ),
        "rollout_value": bool(
            identity_matches
            and version_ok("rollout_value")
            and rollout_files
        ),
        "root_state": bool(
            identity_matches
            and version_ok("root_state")
            and root_state_files_complete(paths)
        ),
        "representations": bool(
            identity_matches
            and version_ok("representations")
            and representation_files
            and parameters_match(
                parameters,
                {
                    "passive_action": int(args.passive_action),
                    "q_temperature": float(args.q_temperature),
                    "representation_max_points": int(args.max_points),
                    "random_state": int(args.random_state),
                },
            )
        ),
    }
    return validity


def build_actor_metrics(actor, actor_cache, blocks, args):
    log("[1/6] Extract canonical real-step search geometry")
    base_metrics, base_qa = actor.process_analysis_traces_parallel(
        blocks,
        max_files=args.max_files,
        passive_action=args.passive_action,
        q_temperature=args.q_temperature,
        policy_temperature=args.policy_temperature,
        workers=args.workers,
    )
    log("[2/6] Extract one-real-step-shifted actor-policy coordinates")
    actor_coordinates, actor_qa = (
        actor_cache.process_actor_policy_coordinates_parallel(
            blocks,
            max_files=args.max_files,
            passive_action=args.passive_action,
            policy_temperature=args.policy_temperature,
            workers=args.workers,
        )
    )
    metrics, merge_qa = actor.apply_actor_policy_coordinates(
        base_metrics, actor_coordinates
    )
    metrics, incumbent_qa = actor_cache.select_target_incumbent_coordinates(
        metrics
    )
    qa = {
        **base_qa,
        "coordinate_variant": actor_cache.COORDINATE_VARIANT,
        "actor_coordinate_extraction": actor_qa,
        "actor_coordinate_merge": merge_qa,
        "incumbent_coordinate_selection": incumbent_qa,
    }
    return base_metrics, metrics, qa


def build_representation_sample(actor, blocks, args, directory: Path) -> dict[str, object]:
    log("[optional] Extract stratified representation sample for t-SNE")
    records, qa = actor.process_traces(
        blocks,
        max_files=args.max_files,
        max_points=args.max_points,
        passive_action=args.passive_action,
        q_temperature=args.q_temperature,
        random_state=args.random_state,
    )
    directory.mkdir(parents=True, exist_ok=True)
    metadata = pd.DataFrame([record.metadata for record in records])
    metadata.to_csv(directory / "sample_metadata.csv", index=False)
    matrices = {
        "tree_reps": actor.stack_padded(
            [record.tree_reps for record in records], "tree_reps"
        ),
        "im_vectors": actor.stack_padded(
            [record.im_vectors for record in records], "im_vectors"
        ),
        "im_vp_vectors": actor.stack_padded(
            [record.im_vp_vectors for record in records], "im_vp_vectors"
        ),
        "tree_topology": actor.stack_padded(
            [record.tree_topology for record in records], "tree_topology"
        ),
    }
    for name, matrix in matrices.items():
        np.save(directory / f"{name}.npy", matrix, allow_pickle=False)
    component = {
        "rows": int(len(metadata)),
        "max_points": int(args.max_points),
        "random_state": int(args.random_state),
        "image_vectorization": "raw_node_flatten_then_temporal_concat",
        "qa": qa,
    }
    atomic_json(directory / "manifest.json", component)
    del records, matrices
    gc.collect()
    return component


def build_root_state_cache(actor, blocks, args, root_state_dir: Path) -> dict[str, object]:
    """Cache real-root im-vectors per run for target-specific 03 drift metrics."""
    log("[6/6] Extract real-root hidden-state vectors for 03")
    root_state_dir.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, object]] = []
    dimensions: set[int] = set()
    for run_number, metas_all in enumerate(blocks.values(), start=1):
        metas = list(
            metas_all[: args.max_files]
            if args.max_files is not None
            else metas_all
        )
        if not metas:
            continue
        run_label = str(metas[0].run_label)
        run_vectors: list[np.ndarray] = []
        real_step = 0
        for meta in metas:
            data = actor.load_npy_dict(meta.path)
            status = np.asarray(data.get("status", []), dtype=int).reshape(-1)
            im_vectors = data.get("im_vectors")
            if status.size == 0 or im_vectors is None:
                raise ValueError(f"Missing status/im_vectors in {meta.path}")
            im_array = np.asarray(im_vectors)
            for raw_index in np.flatnonzero(status == actor.REAL_STATUS):
                real_step += 1
                if int(raw_index) >= len(im_array):
                    raise IndexError(
                        f"im_vectors shorter than status in {meta.path}: {raw_index}"
                    )
                vector = np.asarray(
                    im_array[int(raw_index)], dtype=np.float32
                ).reshape(-1)
                vector = np.nan_to_num(
                    vector, nan=0.0, posinf=0.0, neginf=0.0
                )
                dimensions.add(int(vector.size))
                run_vectors.append(vector)
            del data, status, im_vectors, im_array
            gc.collect()
        if not run_vectors:
            continue
        width = {int(vector.size) for vector in run_vectors}
        if len(width) != 1:
            raise RuntimeError(
                f"Root im-vector width changes within {run_label}: {sorted(width)}"
            )
        matrix = np.vstack(run_vectors).astype(np.float32, copy=False)
        safe_run = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in run_label
        )
        filename = f"{safe_run}.npy"
        np.save(root_state_dir / filename, matrix, allow_pickle=False)
        for vector_row in range(len(matrix)):
            index_rows.append(
                {
                    "run_label": run_label,
                    "real_step": int(vector_row + 1),
                    "vector_row": int(vector_row),
                    "vector_file": filename,
                    "dimension": int(matrix.shape[1]),
                }
            )
        log(
            f"Root-state run {run_number}/{len(blocks)}: {run_label}, "
            f"rows={len(matrix):,}, dim={matrix.shape[1]:,}"
        )
        del run_vectors, matrix
        gc.collect()
    index = pd.DataFrame.from_records(index_rows)
    if index.empty or index.duplicated(["run_label", "real_step"]).any():
        raise RuntimeError("Root-state cache has no rows or duplicated real-step keys")
    save_csv(index, root_state_dir / "index.csv.gz")
    return {
        "rows": int(len(index)),
        "runs": int(index["run_label"].nunique()),
        "dimensions": sorted(dimensions),
        "dtype": "float32",
        "alignment": "status-0 im_vector indexed by within-run real_step starting at 1",
    }


def parse_subject(value: object) -> int:
    text = "".join(character for character in str(value) if character.isdigit())
    if not text:
        raise argparse.ArgumentTypeError(f"Cannot parse subject: {value}")
    return int(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=parse_subject, required=True)
    parser.add_argument("--game", type=int, choices=(0, 1, 2), required=True)
    capture_input = parser.add_mutually_exclusive_group()
    capture_input.add_argument("--capture-dir", type=Path, help="Completed native visualization capture")
    capture_input.add_argument("--capture-index", type=Path, help="Completed held-out subject/game collection index")
    parser.add_argument("--include-prediction-summaries", action="store_true",
        help="Native captures: add aligned imagined/XS-vs-real image distances and static descriptors; no raw image copies")
    parser.add_argument("--include-latent-summaries", action="store_true",
        help="Native captures: add im_vectors channel mean/std and aligned latent distances; no global PCA or raw vector copies")
    parser.add_argument("--native-summary-batch-size", type=int, default=128,
        help="Maximum image/latent rows read per summary batch (default: 128)")
    parser.add_argument("--sessions", default=None)
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_SHARED_CACHE_ROOT)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--passive-action", type=int, default=0)
    parser.add_argument("--q-temperature", type=float, default=1.0)
    parser.add_argument("--policy-temperature", type=float, default=1.0)
    parser.add_argument("--voluntary-reset-code", type=int, default=1)
    parser.add_argument("--forced-reset-code", type=int, default=3)
    parser.add_argument("--root-reference", default="root_v", choices=(
        "root_v", "rollout_return", "max_rollout_return"
    ))
    parser.add_argument("--max-points", type=int, default=5_000)
    parser.add_argument("--random-state", type=int, default=2026)
    representation_group = parser.add_mutually_exclusive_group()
    representation_group.add_argument(
        "--include-tsne-vectors",
        "--include-representations",
        dest="include_tsne_vectors",
        action="store_true",
        help=(
            "Also extract the sampled tree_reps, im_vectors, im_vp_vectors, "
            "and tree_topology arrays required only for t-SNE figures. "
            "These expensive arrays are omitted by default."
        ),
    )
    representation_group.add_argument(
        "--skip-representations",
        dest="include_tsne_vectors",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(include_tsne_vectors=False)
    root_state_group = parser.add_mutually_exclusive_group()
    root_state_group.add_argument(
        "--include-root-state",
        dest="include_root_state",
        action="store_true",
        help=(
            "Also cache status-0 im_vectors for the optional root-state-drift "
            "analysis. These vectors are not required by the core 02/03/05 "
            "behavioral and search analyses."
        ),
    )
    root_state_group.add_argument(
        "--skip-root-state",
        dest="include_root_state",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(include_root_state=False)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Move the existing cache to a timestamped backup and rebuild all "
            "requested components. The default only fills missing or stale "
            "components."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.capture_dir is not None or args.capture_index is not None:
        from dynamic_analysis_cache import build_capture_cache
        build_capture_cache(args)
        return
    if args.include_prediction_summaries or args.include_latent_summaries:
        raise ValueError("Prediction/latent summaries require --capture-dir or --capture-index")
    if args.max_files is not None and args.max_files <= 0:
        raise ValueError("--max-files must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.q_temperature <= 0 or args.policy_temperature <= 0:
        raise ValueError("Policy/Q temperatures must be positive")

    actor = load_script("shared_cache_actor_02", "02_actor_policy.py")
    actor_cache = load_script(
        "shared_actor_policy_cache", "shared_actor_policy_cache.py"
    )
    persistence = load_script(
        "shared_cache_persistence_02", "02_persistence_action_geometry.py"
    )
    model_control = load_script(
        "shared_cache_model_control_05",
        "05_persistence_uncertainty_allocation.py",
    )
    rollout = load_script(
        "shared_cache_rollout_05", "05_rollout_value_allocation.py"
    )

    sessions = actor.parse_sessions(args.sessions)
    blocks = actor.gather_trace_blocks(
        args.trace_root, args.subject, args.game, sessions
    )
    if not blocks:
        raise FileNotFoundError(
            f"No traces for sub{args.subject:03d} game{args.game} under "
            f"{args.trace_root}"
        )
    paths = cache_paths(args.cache_root, args.subject, args.game)
    existing_manifest = read_json(paths.manifest)
    if paths.root.exists() and args.overwrite:
        backup = paths.root.with_name(
            paths.root.name + f".backup_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        shutil.move(str(paths.root), str(backup))
        log(f"Moved previous shared cache to recoverable backup: {backup}")
        existing_manifest = {}
    paths.root.mkdir(parents=True, exist_ok=True)

    source_signature = selected_trace_signature(blocks, args.max_files)
    recovered_manifestless_core = False
    if not existing_manifest and not args.overwrite:
        recovered_manifest = recover_manifestless_core(
            paths,
            args=args,
            sessions=sessions,
            source_signature=source_signature,
        )
        if recovered_manifest is not None:
            existing_manifest = recovered_manifest
            recovered_manifestless_core = True
    identity_matches = cache_identity_matches(
        existing_manifest,
        args=args,
        sessions=sessions,
        source_signature=source_signature,
    )
    if existing_manifest and not identity_matches and not args.overwrite:
        raise RuntimeError(
            "The existing cache belongs to a different trace source or data "
            "selection. Incremental filling is only safe for the same source, "
            "subject/game, sessions, and max-files scope. Use a different "
            "--cache-root or request an explicit --overwrite rebuild."
        )
    validity = component_validity(
        paths,
        existing_manifest,
        identity_matches=identity_matches,
        args=args,
    )
    if args.overwrite:
        validity = {name: False for name in validity}
    requested_components = list(CORE_COMPONENTS)
    if args.include_root_state:
        requested_components.append("root_state")
    if args.include_tsne_vectors:
        requested_components.append("representations")
    plan = {
        name: ("reuse" if validity[name] else "build")
        for name in requested_components
    }
    log(
        "Incremental cache plan: "
        + ", ".join(f"{name}={action}" for name, action in plan.items())
    )
    if (
        all(validity[name] for name in requested_components)
        and not recovered_manifestless_core
    ):
        log(f"Shared cache is already complete and current -> {paths.root}")
        return

    if validity["actor_policy"]:
        log("[reuse] Actor/root-policy real-step metrics")
        root_policy_metrics = pd.read_csv(paths.root_policy_metrics)
        actor_metrics = pd.read_csv(paths.actor_metrics)
        actor_qa = read_json(paths.actor_qa)
    else:
        if identity_matches and paths.root_policy_metrics.exists():
            log(
                "[1/6 reuse] Canonical root/search metrics; rebuilding only "
                "shifted actor-policy coordinates"
            )
            root_policy_metrics = pd.read_csv(paths.root_policy_metrics)
            shifted_actor, shifted_qa = (
                actor_cache.process_actor_policy_coordinates_parallel(
                    blocks,
                    max_files=args.max_files,
                    passive_action=args.passive_action,
                    policy_temperature=args.policy_temperature,
                    workers=args.workers,
                )
            )
            actor_metrics, merge_qa = actor.apply_actor_policy_coordinates(
                root_policy_metrics, shifted_actor
            )
            actor_metrics, incumbent_qa = (
                actor_cache.select_target_incumbent_coordinates(actor_metrics)
            )
            previous_qa = read_json(paths.actor_qa)
            actor_qa = {
                **previous_qa,
                "coordinate_variant": actor_cache.COORDINATE_VARIANT,
                "actor_coordinate_extraction": shifted_qa,
                "actor_coordinate_merge": merge_qa,
                "incumbent_coordinate_selection": incumbent_qa,
                "incremental_rebuild": "shifted_actor_coordinates_only",
            }
        else:
            root_policy_metrics, actor_metrics, actor_qa = build_actor_metrics(
                actor, actor_cache, blocks, args
            )
        save_csv(root_policy_metrics, paths.root_policy_metrics)
        save_csv(actor_metrics, paths.actor_metrics)
        atomic_json(paths.actor_qa, actor_qa)

    representation_manifest: Optional[dict[str, object]] = None
    if args.include_tsne_vectors and not validity["representations"]:
        representation_manifest = build_representation_sample(
            actor, blocks, args, paths.representation_dir
        )
    elif validity["representations"]:
        log("[reuse] Optional t-SNE representation sample")
        representation_manifest = read_json(
            paths.representation_dir / "manifest.json"
        )

    if validity["persistence_action_geometry"]:
        log("[reuse] Persistence/action-geometry windows")
        persistence_windows = pd.read_csv(paths.persistence_windows)
        persistence_qa = read_json(paths.persistence_qa)
        persistence_bout_count = int(persistence_qa.get("n_bouts", 0))
        persistence_event_count = int(persistence_qa.get("n_events", 0))
    else:
        log("[3/6] Extract actor-aligned persistence windows, bouts, and events")
        persistence_windows, persistence_bouts, persistence_events, persistence_qa = (
            persistence.extract_analysis(
                blocks,
                max_files=args.max_files,
                policy_source="actor",
                root_reference_name=args.root_reference,
                policy_temperature=args.policy_temperature,
            )
        )
        persistence_bouts = persistence.add_root_coordinates(
            persistence_windows, persistence_bouts
        )
        save_csv(persistence_windows, paths.persistence_windows)
        save_csv(persistence_bouts, paths.persistence_bouts)
        save_csv(persistence_events, paths.persistence_events)
        atomic_json(paths.persistence_qa, persistence_qa)
        persistence_bout_count = int(len(persistence_bouts))
        persistence_event_count = int(len(persistence_events))

    if validity["model_control"]:
        log("[reuse] Model-policy control-state windows")
        model_windows = pd.read_csv(paths.model_control_windows)
        model_qa = read_json(paths.model_control_qa)
    else:
        log("[4/6] Extract model-policy control-state windows")
        model_windows, model_qa = model_control.extract_model_control_windows(
            blocks=blocks,
            max_files=args.max_files,
            voluntary_reset_code=args.voluntary_reset_code,
            forced_reset_code=args.forced_reset_code,
        )
        save_csv(model_windows, paths.model_control_windows)
        atomic_json(paths.model_control_qa, model_qa)

    if validity["rollout_value"]:
        log("[reuse] Rollout-value windows and prefix events")
        rollout_windows = pd.read_csv(paths.rollout_windows)
        rollout_qa = read_json(paths.rollout_qa)
        prefix_pair_count = int(rollout_qa.get("n_prefix_pairs", 0))
    else:
        log("[5/6] Extract rollout-value windows and adjacent prefix events")
        rollout_windows, prefix_events, rollout_qa = rollout.extract_rollout_tables(
            blocks,
            args.max_files,
        )
        save_csv(rollout_windows, paths.rollout_windows)
        save_npz(paths.rollout_prefix_events, prefix_events)
        atomic_json(paths.rollout_qa, rollout_qa)
        prefix_pair_count = int(len(prefix_events.get("window_index", [])))

    root_state_qa: Optional[dict[str, object]] = None
    if validity["root_state"]:
        log("[reuse] Real-root hidden-state vectors")
        existing_components = existing_manifest.get("components", {})
        if not isinstance(existing_components, Mapping):
            existing_components = {}
        existing_root_state = existing_components.get("root_state", {})
        root_state_qa = (
            dict(existing_root_state)
            if isinstance(existing_root_state, Mapping)
            else {}
        )
        if not root_state_qa:
            root_index = pd.read_csv(paths.root_state_index)
            root_state_qa = {
                "rows": int(len(root_index)),
                "runs": int(root_index["run_label"].nunique()),
                "dimensions": sorted(
                    int(value) for value in root_index["dimension"].unique()
                ),
                "dtype": "float32",
                "alignment": (
                    "status-0 im_vector indexed by within-run real_step starting at 1"
                ),
            }
    elif args.include_root_state:
        root_state_qa = build_root_state_cache(
            actor, blocks, args, paths.root_state_dir
        )
    else:
        log("[skip] Optional real-root hidden-state vectors")

    integrity = validate_one_to_one_real_steps(
        {
            "actor_policy": actor_metrics,
            "root_policy": root_policy_metrics,
            "persistence_action_geometry": persistence_windows,
            "model_control": model_windows,
            "rollout_value": rollout_windows,
        }
    )
    manifest = {
        "shared_cache_version": SHARED_CACHE_VERSION,
        "component_versions": {
            name: version
            for name, version in COMPONENT_VERSIONS.items()
            if (
                (name != "representations" or representation_manifest is not None)
                and (name != "root_state" or root_state_qa is not None)
            )
        },
        "subject": int(args.subject),
        "game": int(args.game),
        "sessions": sorted(sessions) if sessions is not None else None,
        "max_files_per_run": args.max_files,
        "trace_root": str(args.trace_root.resolve()),
        "source_files": source_signature,
        "parameters": {
            "passive_action": int(args.passive_action),
            "q_temperature": float(args.q_temperature),
            "policy_temperature": float(args.policy_temperature),
            "voluntary_reset_code": int(args.voluntary_reset_code),
            "forced_reset_code": int(args.forced_reset_code),
            "persistence_policy_source": "actor",
            "root_reference": args.root_reference,
            "tsne_vectors_included": representation_manifest is not None,
            "root_state_included": root_state_qa is not None,
            "representation_max_points": (
                int(representation_manifest.get("max_points", args.max_points))
                if representation_manifest is not None
                else None
            ),
            "random_state": (
                int(representation_manifest.get("random_state", args.random_state))
                if representation_manifest is not None
                else int(args.random_state)
            ),
        },
        "components": {
            "actor_policy": actor_qa,
            "root_policy": {
                "rows": int(len(root_policy_metrics)),
                "definition": "unchanged 02 root model-policy coordinates",
            },
            "representations": representation_manifest,
            "persistence_action_geometry": persistence_qa,
            "model_control": model_qa,
            "rollout_value": rollout_qa,
            "root_state": root_state_qa,
        },
        "integrity": integrity,
        "brain_cache": {
            "included": False,
            "owner": "00_brain_cache.py / planning_brain_common.py",
            "default_root": str(
                SCRIPT_DIR
                / "outputs"
                / "00_brain_cache"
                / "time_aligned_cache"
            ),
        },
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "incremental_update": {
            "identity_reused": bool(identity_matches),
            "manifestless_core_recovered": bool(recovered_manifestless_core),
            "component_plan": plan,
            "full_overwrite": bool(args.overwrite),
        },
    }
    if existing_manifest and identity_matches and not args.overwrite:
        manifest["created_at"] = existing_manifest.get(
            "created_at", manifest["created_at"]
        )
    atomic_json(paths.manifest, manifest)
    log(
        f"Shared cache complete -> {paths.root} | "
        f"actor rows={len(actor_metrics):,}, bouts={persistence_bout_count:,}, "
        f"status-2 events={persistence_event_count:,}, "
        f"prefix pairs={prefix_pair_count:,}"
    )


if __name__ == "__main__":
    main()
