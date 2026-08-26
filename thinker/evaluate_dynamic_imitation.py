#!/usr/bin/env python3
"""Paired held-out-session evaluation for teacher-forced Dynamic Thinker.

The canonical evaluation tiles each recorded episode with non-overlapping
``L=4`` scored targets.  Every window first executes one unscored human
burn-in action, then compares the same actor/model checkpoint and recorded
inputs with tree carry disabled and enabled.  This script reports descriptive
NLL differences only; it deliberately performs no significance testing.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import random
import sys
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml


EVALUATION_SCHEMA_VERSION = 2


DYNAMIC_PROTOCOL = {
    "dynamic_factorized_control": True,
    "rec_t": 20,
    "max_search_steps": 20,
    "max_depth": 20,
    "model_unroll_len": 20,
    "think_cost": 0.0005,
    "think_cost_anneal": False,
    "tree_carry": True,
    "sep_im_head": True,
    "frame_stack_n": 4,
}
IMITATION_PROTOCOL = {
    "batch_length": 4,
    "icopro_action_diff_coef": 1.0,
    "icopro_margin": 1.0,
    "icopro_margin_coef": 1.0,
    "icopro_pvp_coef": 0.0,
    "icopro_coef": 1.0,
    "icopro_supervised_freq": 1,
    "icopro_batch_size": 16,
    "action_prior_weight": 1.0,
    "action_prior_ema": 0.05,
}
RUNTIME_SEMANTIC_FIELDS = (
    "wrapper_type",
    "discounting",
    "reward_clip",
    "require_prob",
    "reset_mode",
    "return_h",
    "return_x",
    "has_action_seq",
    "im_enable",
    "stat_mask_type",
    "model_size_nn",
    "model_done_loss_cost",
    "model_enc_type",
    "model_enc_f_type",
    "model_mem_unroll_len",
    "model_has_memory",
    "model_ordinal",
    "model_decoder_depth",
    "model_downscale_c",
    "model_downscale_c_vp",
    "model_disable_bn",
    "model_state_projection",
    "model_state_range_loss_cost",
    "model_zero_init",
    "dual_net",
    "noise_enable",
    "see_real_state",
    "see_tree_rep",
    "see_h",
    "see_x",
    "dynamic_search_hidden_dim",
    "tree_rep_rnn",
    "x_rnn",
    "h_rnn",
    "real_state_rnn",
    "real_state_ch",
    "critic_enc_type",
    "critic_enc_f_type",
    "actor_ordinal",
    "float16",
    "model_float16",
    "schedule_total_steps",
    "actor_amp_max_consecutive_skips",
)


@dataclass(frozen=True)
class EvaluationSpec:
    """Checkpoint-derived identity and live Atari environment contract."""

    subjects: Tuple[int, ...]
    train_sessions: Tuple[int, ...]
    holdout_sessions: Tuple[int, ...]
    game_id: int
    env_name: str
    num_actions: int
    scored_length: int
    frame_stack_n: int
    grayscale: bool
    observation_shape: Tuple[int, ...]
    observation_dtype: str
    target_size: Tuple[int, int]
    observation_low: float = 0.0
    observation_high: float = 255.0

    @property
    def subject(self) -> int:
        return self.subjects[0]

    @property
    def holdout_session(self) -> int:
        return self.holdout_sessions[0]

    @property
    def score_mask(self) -> List[bool]:
        return [False] + [True] * self.scored_length


def required_imitation_protocol(spec: EvaluationSpec) -> Dict[str, Any]:
    return {**IMITATION_PROTOCOL, "icopro_game_id": spec.game_id}

CSV_FIELDS = (
    "window_id",
    "subject",
    "session",
    "block",
    "game",
    "source_file",
    "episode_index",
    "window_start",
    "scored_position",
    "edge_position",
    "decision_time",
    "observation_source_index",
    "human_action",
    "nll_no_carry",
    "nll_carry",
    "delta_nll",
    "root_carried_no_carry",
    "root_carried_carry",
    "carried_descendant_visit_count_no_carry",
    "carried_descendant_visit_count_carry",
    "carried_descendant_expanded_count_no_carry",
    "carried_descendant_expanded_count_carry",
    "useful_carry_no_carry",
    "useful_carry_carry",
    "argmax_no_carry",
    "argmax_carry",
    "proposal_no_carry",
    "proposal_carry",
)


def _array(value: Any, dtype: Optional[np.dtype] = None) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    return result.astype(dtype, copy=False) if dtype is not None else result


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 hash for an auditable input or output."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


REQUIRED_CHECKPOINT_FILES = (
    "config_c.yaml",
    "ckp_actor.tar",
    "ckp_model.tar",
    "finish",
)


def validate_completion_marker(checkpoint_dir: str | Path) -> Dict[str, Any]:
    """Verify that ``finish`` binds the final checkpoints and training source."""

    root = Path(checkpoint_dir).expanduser().resolve()
    marker_path = root / "finish"
    try:
        with marker_path.open("r", encoding="utf-8") as handle:
            marker = json.load(handle)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid completion marker JSON: {marker_path}") from error
    if marker.get("schema_version") != 1 or marker.get("status") != "complete":
        raise ValueError("checkpoint completion marker is not a completed v1 bundle")

    recorded_files = marker.get("checkpoint_files")
    if not isinstance(recorded_files, Mapping):
        raise ValueError("completion marker lacks checkpoint file hashes")
    for name in REQUIRED_CHECKPOINT_FILES[:-1]:
        path = root / name
        record = recorded_files.get(name)
        if not isinstance(record, Mapping):
            raise ValueError(f"completion marker lacks {name}")
        actual_hash = sha256_file(path)
        if record.get("sha256") != actual_hash:
            raise ValueError(
                f"completion marker does not match final {name}: "
                f"{record.get('sha256')!r} != {actual_hash!r}"
            )
        if int(record.get("size", -1)) != path.stat().st_size:
            raise ValueError(f"completion marker size does not match final {name}")

    recorded_sources = marker.get("implementation_sources")
    if not isinstance(recorded_sources, Mapping) or not recorded_sources:
        raise ValueError("completion marker lacks training implementation hashes")
    package_root = Path(__file__).resolve().parent
    for relative, record in recorded_sources.items():
        relative_path = Path(str(relative))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"unsafe implementation path in completion marker: {relative}")
        path = (package_root / relative_path).resolve()
        if package_root not in path.parents and path != package_root:
            raise ValueError(f"implementation path escapes package root: {relative}")
        if not path.is_file():
            raise FileNotFoundError(
                f"training implementation source is unavailable: {path}"
            )
        if not isinstance(record, Mapping) or record.get("sha256") != sha256_file(path):
            raise ValueError(
                f"evaluation source differs from training implementation: {relative}"
            )

    recorded_extensions = marker.get("loaded_extensions")
    if not isinstance(recorded_extensions, Mapping) or not recorded_extensions:
        raise ValueError("completion marker lacks the loaded Cython extension hash")
    for relative, record in recorded_extensions.items():
        relative_path = Path(str(relative))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"unsafe extension path in completion marker: {relative}")
        path = (package_root / relative_path).resolve()
        if package_root not in path.parents or not path.is_file():
            raise FileNotFoundError(
                f"training Cython extension is unavailable: {path}"
            )
        if not isinstance(record, Mapping) or record.get("sha256") != sha256_file(path):
            raise ValueError(
                f"loaded Cython extension differs from training: {relative}"
            )
    return marker


def checkpoint_hashes(checkpoint_dir: str | Path) -> Dict[str, str]:
    """Hash the completed checkpoint bundle used for an evaluation."""

    root = Path(checkpoint_dir).expanduser().resolve()
    paths = tuple(root / name for name in REQUIRED_CHECKPOINT_FILES)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("checkpoint is incomplete: " + ", ".join(missing))
    validate_completion_marker(root)
    return {path.name: sha256_file(path) for path in paths}


def _metadata_value(batch: Mapping[str, Any], key: str, row: int) -> Any:
    value = batch[key]
    array = _array(value)
    if array.ndim == 0:
        return array.item()
    item = array[row]
    return item.item() if np.asarray(item).ndim == 0 else item


def validate_holdout_batch(
    batch: Mapping[str, Any], spec: EvaluationSpec
) -> int:
    """Enforce the checkpoint-derived holdout and burn-in contract."""

    required = (
        "obs_seq",
        "actions_seq",
        "initial_prev_action",
        "score_mask",
        "source_file",
        "subject",
        "session",
        "block",
        "game",
        "episode_index",
        "window_start",
        "decision_times",
        "observation_source_index",
    )
    missing = [key for key in required if key not in batch]
    if missing:
        raise KeyError("evaluation batch is missing: " + ", ".join(missing))

    actions = _array(batch["actions_seq"], np.int64)
    observations = _array(batch["obs_seq"])
    previous = _array(batch["initial_prev_action"], np.int64)
    if actions.ndim != 2 or actions.shape[1] != spec.scored_length + 1:
        raise ValueError(
            f"actions_seq must be [B,{spec.scored_length + 1}], got {actions.shape}"
        )
    batch_size = int(actions.shape[0])
    if observations.shape[:2] != (batch_size, spec.scored_length + 2):
        raise ValueError(
            f"obs_seq must begin [B,{spec.scored_length + 2}], got {observations.shape}"
        )
    if tuple(observations.shape[2:]) != spec.observation_shape:
        raise ValueError(
            "obs_seq observation shape does not match the live environment: "
            f"{tuple(observations.shape[2:])} versus {spec.observation_shape}"
        )
    if np.dtype(observations.dtype).name != spec.observation_dtype:
        raise ValueError(
            "obs_seq dtype does not match the live environment: "
            f"{np.dtype(observations.dtype).name} versus {spec.observation_dtype}"
        )
    if np.any(observations < spec.observation_low) or np.any(
        observations > spec.observation_high
    ):
        raise ValueError(
            "obs_seq values do not match the live observation bounds "
            f"[{spec.observation_low},{spec.observation_high}]"
        )
    if previous.shape != (batch_size,):
        raise ValueError("initial_prev_action must have shape [B]")

    mask = _array(batch["score_mask"], np.bool_)
    expected_mask = np.asarray(spec.score_mask, dtype=np.bool_)
    if mask.shape == (batch_size, spec.scored_length + 1):
        if not np.all(mask == expected_mask[None, :]):
            raise ValueError("every score_mask row must exclude only burn-in")
    elif mask.shape != expected_mask.shape or not np.array_equal(mask, expected_mask):
        raise ValueError(f"score_mask must be {spec.score_mask}")

    entities = {
        "subject": spec.subjects,
        "session": spec.holdout_sessions,
        "game": (spec.game_id,),
    }
    for key, allowed in entities.items():
        values = _array(batch[key], np.int64).reshape(-1)
        if values.shape != (batch_size,) or not np.all(np.isin(values, allowed)):
            raise ValueError(
                f"holdout evaluation requires {key} in {allowed}, got "
                f"{np.unique(values).tolist()}"
            )
    sources = _array(batch["source_file"]).reshape(-1)
    if sources.shape != (batch_size,):
        raise ValueError("source_file must have one entry per window")
    # Metadata is authoritative, while this path check catches accidental use
    # of an archive whose metadata was relabelled upstream.
    session_tokens = {
        f"ses-{session:02d}" for session in spec.holdout_sessions
    } | {f"ses-{session}" for session in spec.holdout_sessions} | {
        f"day-{session:02d}" for session in spec.holdout_sessions
    } | {f"day-{session}" for session in spec.holdout_sessions}
    for source in sources:
        normalized = str(source).replace("_", "-").lower()
        if not any(token in normalized for token in session_tokens):
            raise ValueError(
                "source outside checkpoint holdout sessions reached evaluation: "
                f"{source}"
            )
    return batch_size


def _window_uid(source: str, episode: int, start: int) -> str:
    payload = f"{Path(source).resolve()}|{episode}|{start}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def build_paired_rows(
    batch: Mapping[str, Any],
    no_carry: Any,
    carry: Any,
    spec: EvaluationSpec,
) -> List[Dict[str, Any]]:
    """Flatten two paired rollout results into auditable per-step records."""

    batch_size = validate_holdout_batch(batch, spec)
    actions = _array(batch["actions_seq"], np.int64)
    scored_actions = actions[:, 1:]

    def result_array(result: Any, name: str, dtype: np.dtype) -> np.ndarray:
        value = _array(getattr(result, name), dtype)
        expected = (batch_size, spec.scored_length)
        if value.shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {value.shape}")
        return value

    nll_no = result_array(no_carry, "per_stage_nll", np.float64)
    nll_yes = result_array(carry, "per_stage_nll", np.float64)
    roots_no = result_array(no_carry, "root_carried", np.bool_)
    roots_yes = result_array(carry, "root_carried", np.bool_)
    if np.any(roots_no):
        raise RuntimeError("root_carried was true while tree_carry=false")
    visits_no = result_array(
        no_carry, "carried_descendant_visit_count", np.int64
    )
    visits_yes = result_array(
        carry, "carried_descendant_visit_count", np.int64
    )
    expanded_no = result_array(
        no_carry, "carried_descendant_expanded_count", np.int64
    )
    expanded_yes = result_array(
        carry, "carried_descendant_expanded_count", np.int64
    )
    useful_no = result_array(no_carry, "useful_carry", np.bool_)
    useful_yes = result_array(carry, "useful_carry", np.bool_)
    for condition, roots, visits, expanded, useful in (
        ("no-carry", roots_no, visits_no, expanded_no, useful_no),
        ("carry", roots_yes, visits_yes, expanded_yes, useful_yes),
    ):
        if np.any(visits < 0) or np.any(expanded < 0):
            raise RuntimeError(
                f"{condition} descendant carry counts must be nonnegative"
            )
        if np.any(visits[~roots] != 0) or np.any(expanded[~roots] != 0):
            raise RuntimeError(
                f"{condition} descendant carry counts were nonzero while "
                "root_carried=false"
            )
        if np.any(expanded > visits) or not np.array_equal(
            expanded > 0, visits > 0
        ):
            raise RuntimeError(
                f"{condition} descendant expanded/visit counts violate "
                "the cenv tree contract"
            )
        expected_useful = roots & (visits > 0)
        if not np.array_equal(useful, expected_useful):
            raise RuntimeError(
                f"{condition} useful_carry must equal root_carried and "
                "carried_descendant_visit_count > 0"
            )

    argmax_no = result_array(no_carry, "argmax", np.int64)
    argmax_yes = result_array(carry, "argmax", np.int64)
    proposal_no = result_array(no_carry, "proposal", np.int64)
    proposal_yes = result_array(carry, "proposal", np.int64)
    executed_no = result_array(no_carry, "executed", np.int64)
    executed_yes = result_array(carry, "executed", np.int64)
    if not np.array_equal(executed_no, scored_actions) or not np.array_equal(
        executed_yes, scored_actions
    ):
        raise RuntimeError("a paired rollout did not execute the human targets")
    burnin = actions[:, 0]
    if not np.array_equal(_array(no_carry.burnin_executed, np.int64), burnin):
        raise RuntimeError("no-carry rollout did not execute the burn-in action")
    if not np.array_equal(_array(carry.burnin_executed, np.int64), burnin):
        raise RuntimeError("carry rollout did not execute the burn-in action")
    expected_count = batch_size * spec.scored_length
    if int(no_carry.count) != expected_count or int(carry.count) != expected_count:
        raise RuntimeError("burn-in was scored or a scored action was omitted")

    decision_times = _array(batch["decision_times"], np.float64)
    source_indices = _array(batch["observation_source_index"], np.int64)
    if decision_times.shape != (batch_size, spec.scored_length + 2):
        raise ValueError("decision_times must align with obs_seq")
    if source_indices.shape != decision_times.shape:
        raise ValueError("observation_source_index must align with obs_seq")

    rows: List[Dict[str, Any]] = []
    for b in range(batch_size):
        source = str(_metadata_value(batch, "source_file", b))
        episode = int(_metadata_value(batch, "episode_index", b))
        start = int(_metadata_value(batch, "window_start", b))
        uid = _window_uid(source, episode, start)
        for position in range(spec.scored_length):
            no_value = float(nll_no[b, position])
            yes_value = float(nll_yes[b, position])
            rows.append(
                {
                    "window_id": uid,
                    "subject": int(_metadata_value(batch, "subject", b)),
                    "session": int(_metadata_value(batch, "session", b)),
                    "block": int(_metadata_value(batch, "block", b)),
                    "game": int(_metadata_value(batch, "game", b)),
                    "source_file": source,
                    "episode_index": episode,
                    "window_start": start,
                    "scored_position": position,
                    # Edge zero is burn-in, so scored targets occupy 1..4.
                    "edge_position": position + 1,
                    "decision_time": float(decision_times[b, position + 1]),
                    "observation_source_index": int(
                        source_indices[b, position + 1]
                    ),
                    "human_action": int(scored_actions[b, position]),
                    "nll_no_carry": no_value,
                    "nll_carry": yes_value,
                    "delta_nll": no_value - yes_value,
                    "root_carried_no_carry": bool(roots_no[b, position]),
                    "root_carried_carry": bool(roots_yes[b, position]),
                    "carried_descendant_visit_count_no_carry": int(
                        visits_no[b, position]
                    ),
                    "carried_descendant_visit_count_carry": int(
                        visits_yes[b, position]
                    ),
                    "carried_descendant_expanded_count_no_carry": int(
                        expanded_no[b, position]
                    ),
                    "carried_descendant_expanded_count_carry": int(
                        expanded_yes[b, position]
                    ),
                    "useful_carry_no_carry": bool(useful_no[b, position]),
                    "useful_carry_carry": bool(useful_yes[b, position]),
                    "argmax_no_carry": int(argmax_no[b, position]),
                    "argmax_carry": int(argmax_yes[b, position]),
                    "proposal_no_carry": int(proposal_no[b, position]),
                    "proposal_carry": int(proposal_yes[b, position]),
                }
            )
    return rows


def _metric_block(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "nll_no_carry": None,
            "nll_carry": None,
            "delta_nll_no_minus_carry": None,
            "argmax_accuracy_no_carry": None,
            "argmax_accuracy_carry": None,
        }
    count = len(rows)
    no = sum(float(row["nll_no_carry"]) for row in rows) / count
    yes = sum(float(row["nll_carry"]) for row in rows) / count
    return {
        "count": count,
        "nll_no_carry": no,
        "nll_carry": yes,
        "delta_nll_no_minus_carry": no - yes,
        "argmax_accuracy_no_carry": sum(
            int(row["argmax_no_carry"] == row["human_action"]) for row in rows
        )
        / count,
        "argmax_accuracy_carry": sum(
            int(row["argmax_carry"] == row["human_action"]) for row in rows
        )
        / count,
    }


def summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    coverage: Mapping[str, Any],
    stride: int,
    spec: EvaluationSpec,
) -> Dict[str, Any]:
    """Aggregate overall, promoted-root, and informative-carry results."""

    carried = [row for row in rows if bool(row["root_carried_carry"])]
    useful = [row for row in rows if bool(row["useful_carry_carry"])]
    windows = {str(row["window_id"]) for row in rows}
    sources = sorted({str(row["source_file"]) for row in rows})
    sessions = sorted({int(row["session"]) for row in rows})
    blocks = sorted({int(row["block"]) for row in rows})
    total = len(rows)
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "metric_definition": {
            "nll": "negative log likelihood of the recorded human action",
            "delta": "NLL_no_carry - NLL_carry; positive values favor carry",
            "conditional_subset": (
                "scored roots for which the carry-enabled cenv reported "
                "root_carried=true"
            ),
            "useful_carry": (
                "root_carried=true and carried_descendant_visit_count>0; "
                "the count is captured immediately before the real-state "
                "override and sums rollout_n over the promoted root's "
                "immediate children"
            ),
            "useful_conditional_subset": (
                "scored roots with useful_carry=true"
            ),
        },
        "protocol": {
            "subject": spec.subject,
            "session": spec.holdout_session,
            "game": spec.game_id,
            "env_name": spec.env_name,
            "num_actions": spec.num_actions,
            "scored_length": spec.scored_length,
            "score_mask": spec.score_mask,
            "window_stride": int(stride),
            "overlapping_scored_targets": bool(stride < spec.scored_length),
            "paired_identical_inputs": True,
            "greedy_actor": True,
            "inferential_statistics": "none",
        },
        "overall": _metric_block(rows),
        "root_carried_true": _metric_block(carried),
        "root_carry_coverage": {
            "count": len(carried),
            "total_scored_rows": total,
            "fraction": float(len(carried) / total) if total else 0.0,
        },
        "useful_carry_true": _metric_block(useful),
        "useful_carry_coverage": {
            "count": len(useful),
            "total_scored_rows": total,
            "fraction": float(len(useful) / total) if total else 0.0,
            "support_status": "has_support" if useful else "no_support",
            "descendant_visit_count_sum": sum(
                int(row["carried_descendant_visit_count_carry"])
                for row in rows
            ),
            "descendant_expanded_count_sum": sum(
                int(row["carried_descendant_expanded_count_carry"])
                for row in rows
            ),
        },
        "data_coverage": {
            **dict(coverage),
            "windows_evaluated": len(windows),
            "scored_rows_evaluated": total,
            "source_file_count": len(sources),
            "source_files": sources,
            "sessions": sessions,
            "blocks": blocks,
        },
    }


def write_rows_csv(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    path = Path(path)
    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def write_json(value: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def checkpoint_protocol(flags: Any) -> Dict[str, Any]:
    """Return the actual checkpoint settings relevant to this evaluation."""

    names = tuple(DYNAMIC_PROTOCOL) + tuple(IMITATION_PROTOCOL) + (
        "icopro_data_path",
        "icopro_subjects",
        "icopro_train_sessions",
        "icopro_holdout_sessions",
        "grayscale",
        "envpool",
        "name",
        "dynamic_search",
    ) + RUNTIME_SEMANTIC_FIELDS
    return {name: getattr(flags, name, None) for name in names}


def make_manifest(
    *,
    checkpoint_dir: str | Path,
    source_files: Iterable[str | Path],
    training_source_files: Iterable[str | Path] = (),
    output_files: Iterable[str | Path],
    args: Mapping[str, Any],
    spec: EvaluationSpec,
    flags: Optional[Any] = None,
    expected_checkpoint_hashes: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Hash all source, checkpoint and already-written result artifacts."""

    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    checkpoints = [checkpoint_dir / name for name in REQUIRED_CHECKPOINT_FILES]
    missing = [str(path) for path in checkpoints if not path.is_file()]
    if missing:
        raise FileNotFoundError("checkpoint is incomplete: " + ", ".join(missing))
    completion_state = validate_completion_marker(checkpoint_dir)
    current_checkpoint_hashes = {path.name: sha256_file(path) for path in checkpoints}
    if (
        expected_checkpoint_hashes is not None
        and dict(expected_checkpoint_hashes) != current_checkpoint_hashes
    ):
        raise RuntimeError(
            "checkpoint files changed during evaluation; evaluate an immutable snapshot"
        )

    source_paths = sorted({Path(path).expanduser().resolve() for path in source_files})
    training_source_paths = sorted(
        {Path(path).expanduser().resolve() for path in training_source_files}
    )
    output_paths = sorted({Path(path).expanduser().resolve() for path in output_files})
    package_dir = Path(__file__).resolve().parent / "thinker"
    implementation_paths = [
        Path(__file__).resolve(),
        package_dir / "bc_loader.py",
        package_dir / "dataset_env.py",
        package_dir / "dynamic_imitation.py",
        package_dir / "cenv.pyx",
        package_dir / "actor_net.py",
        package_dir / "model_net.py",
        package_dir / "learn_actor.py",
        package_dir / "gym_add" / "wrapper.py",
    ]
    loaded_cenv = sys.modules.get("thinker.cenv")
    loaded_cenv_path = getattr(loaded_cenv, "__file__", None)
    if loaded_cenv_path:
        implementation_paths.append(Path(loaded_cenv_path).resolve())
    implementation_paths = [path for path in implementation_paths if path.is_file()]
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": list(sys.argv),
        "arguments": dict(args),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": str(args.get("device", "unknown")),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "checkpoint": {
            "directory": str(checkpoint_dir),
            "files": {
                path.name: {
                    "path": str(path),
                    "sha256": current_checkpoint_hashes[path.name],
                }
                for path in checkpoints
            },
            "checkpoint_git_revision": getattr(
                flags, "checkpoint_git_revision", None
            ),
            "required_dynamic_protocol": dict(DYNAMIC_PROTOCOL),
            "required_imitation_protocol": required_imitation_protocol(spec),
            "checkpoint_protocol": (
                checkpoint_protocol(flags) if flags is not None else None
            ),
            "actor_imitation_state": (
                getattr(flags, "actor_checkpoint_imitation_state", None)
                if flags is not None else None
            ),
            "model_validation_state": (
                getattr(flags, "model_checkpoint_validation_state", None)
                if flags is not None else None
            ),
            "evaluation_spec": asdict(spec),
            "completion_marker": completion_state,
        },
        "behavioral_sources": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "subject": spec.subject,
                "session": spec.holdout_session,
                "game": spec.game_id,
                "env_name": spec.env_name,
                "num_actions": spec.num_actions,
            }
            for path in source_paths
        ],
        "behavioral_training_sources": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "subject": spec.subject,
                "sessions": list(spec.train_sessions),
                "game": spec.game_id,
                "env_name": spec.env_name,
                "num_actions": spec.num_actions,
            }
            for path in training_source_paths
        ],
        "implementation_sources": {
            str(path): {"sha256": sha256_file(path)}
            for path in implementation_paths
        },
        "outputs": {
            path.name: {"path": str(path), "sha256": sha256_file(path)}
            for path in output_paths
        },
    }


def _set_pair_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _parse_id_setting(value: Any, name: str) -> Tuple[int, ...]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, (tuple, list, np.ndarray)):
        parts = list(value)
    elif value is None:
        parts = []
    else:
        parts = [value]
    try:
        return tuple(sorted({int(part) for part in parts}))
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid checkpoint {name}={value!r}") from error


def _protocol_value_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        try:
            return abs(float(actual) - expected) <= 1e-12
        except (TypeError, ValueError):
            return False
    return actual == expected


def _make_live_environment(flags: Any) -> Any:
    """Construct the same vector-environment frontend used for online training."""

    from thinker.gym_add import wrapper

    if bool(getattr(flags, "envpool", False)):
        return wrapper.create_envpool(str(flags.name), flags, env_n=1)

    from thinker.gym_add.asyn_vector_env import AsyncVectorEnv

    env_fn = wrapper.create_env_fn(str(flags.name), flags)
    return AsyncVectorEnv([env_fn])


def resolve_evaluation_spec(
    flags: Any,
    *,
    expected_env_name: Optional[str] = None,
    expected_game_id: Optional[int] = None,
) -> EvaluationSpec:
    """Resolve identity from checkpoint flags and dimensions from a live env.

    The optional expected values are assertions only.  They never select a
    dataset or override checkpoint configuration.
    """

    subjects = _parse_id_setting(getattr(flags, "icopro_subjects", None), "icopro_subjects")
    train_sessions = _parse_id_setting(
        getattr(flags, "icopro_train_sessions", None), "icopro_train_sessions"
    )
    holdout_sessions = _parse_id_setting(
        getattr(flags, "icopro_holdout_sessions", None), "icopro_holdout_sessions"
    )
    if len(subjects) != 1:
        raise ValueError(
            "paired evaluation currently requires exactly one checkpoint "
            f"subject, got {subjects}"
        )
    if not train_sessions:
        raise ValueError(
            "checkpoint icopro_train_sessions must contain at least one session"
        )
    if len(holdout_sessions) != 1:
        raise ValueError(
            "paired evaluation currently requires exactly one checkpoint "
            f"holdout session, got {holdout_sessions}"
        )
    overlap = set(train_sessions) & set(holdout_sessions)
    if overlap:
        raise ValueError(
            "checkpoint training and holdout sessions overlap: "
            f"{sorted(overlap)}"
        )

    try:
        game_id = int(getattr(flags, "icopro_game_id"))
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("checkpoint has no valid icopro_game_id") from error
    env_name = str(getattr(flags, "name", "")).strip()
    if not env_name:
        raise ValueError("checkpoint has an empty environment name")
    if expected_game_id is not None and int(expected_game_id) != game_id:
        raise ValueError(
            f"expected game id {int(expected_game_id)}, but checkpoint contains {game_id}"
        )
    if expected_env_name is not None and str(expected_env_name) != env_name:
        raise ValueError(
            f"expected environment {expected_env_name!r}, but checkpoint contains {env_name!r}"
        )

    scored_length = int(getattr(flags, "batch_length", -1))
    frame_stack_n = int(getattr(flags, "frame_stack_n", -1))
    grayscale = bool(getattr(flags, "grayscale", False))
    if scored_length < 1:
        raise ValueError("checkpoint batch_length must be positive")
    if frame_stack_n < 1:
        raise ValueError("checkpoint frame_stack_n must be positive")

    live_env = _make_live_environment(flags)
    try:
        action_space = getattr(live_env, "single_action_space", None)
        observation_space = getattr(live_env, "single_observation_space", None)
        from gymnasium import spaces

        if not isinstance(action_space, spaces.Discrete):
            raise TypeError(
                "dynamic imitation evaluation requires a live Discrete "
                f"single_action_space, got {action_space!r}"
            )
        if not isinstance(observation_space, spaces.Box):
            raise TypeError(
                "dynamic imitation evaluation requires a live Box "
                f"single_observation_space, got {observation_space!r}"
            )
        observation_shape = tuple(int(value) for value in observation_space.shape)
        if len(observation_shape) != 3:
            raise ValueError(
                "live Atari observation must be CHW, got "
                f"shape {observation_shape}"
            )
        channels, height, width = observation_shape
        expected_channels = frame_stack_n * (1 if grayscale else 3)
        if channels != expected_channels:
            raise ValueError(
                "live observation channels disagree with checkpoint stacking: "
                f"shape={observation_shape}, frame_stack_n={frame_stack_n}, "
                f"grayscale={grayscale}, expected_channels={expected_channels}"
            )
        if height < 1 or width < 1:
            raise ValueError(f"invalid live observation shape {observation_shape}")
        num_actions = int(action_space.n)
        if num_actions < 1:
            raise ValueError("live Discrete action space is empty")
        action_start = int(getattr(action_space, "start", 0))
        if action_start != 0:
            raise ValueError(
                "behavioral action indices require a zero-based live Discrete "
                f"space, got start={action_start}"
            )
        probe_action = 0
        if not action_space.contains(probe_action):
            raise ValueError(
                f"live action space rejected legal probe action {probe_action}"
            )
        observation_dtype = np.dtype(observation_space.dtype).name
        observation_low_array = np.asarray(observation_space.low)
        observation_high_array = np.asarray(observation_space.high)
        byte_contract = (
            observation_dtype == "uint8"
            and np.all(observation_low_array == 0)
            and np.all(observation_high_array == 255)
        )
        unit_float_contract = (
            observation_dtype == "float32"
            and np.all(observation_low_array == 0.0)
            and np.all(observation_high_array == 1.0)
        )
        if not byte_contract and not unit_float_contract:
            raise ValueError(
                "behavioral evaluation supports live uint8 [0,255] or "
                "float32 [0,1] observations"
            )
        observation_low = 0.0
        observation_high = 255.0 if byte_contract else 1.0

        def validate_live_observation(value: Any, stage: str) -> None:
            array = np.asarray(value)
            expected_shape = (1,) + observation_shape
            if tuple(array.shape) != expected_shape:
                raise ValueError(
                    f"live environment {stage} observation has shape "
                    f"{tuple(array.shape)}, expected {expected_shape}"
                )
            if np.dtype(array.dtype).name != observation_dtype:
                raise ValueError(
                    f"live environment {stage} observation has dtype "
                    f"{np.dtype(array.dtype).name}, expected {observation_dtype}"
                )
            if np.any(array < observation_low) or np.any(array > observation_high):
                raise ValueError(
                    f"live environment {stage} observation lies outside "
                    f"[{observation_low},{observation_high}]"
                )

        reset_result = live_env.reset()
        if not isinstance(reset_result, tuple) or len(reset_result) != 2:
            raise TypeError(
                "live vector environment reset must return (observation, info)"
            )
        validate_live_observation(reset_result[0], "reset")
        step_result = live_env.step(
            np.asarray([probe_action], dtype=action_space.dtype)
        )
        if not isinstance(step_result, tuple) or len(step_result) != 5:
            raise TypeError(
                "live vector environment step must return the Gymnasium 5-tuple"
            )
        validate_live_observation(step_result[0], "step")
    finally:
        live_env.close()

    return EvaluationSpec(
        subjects=subjects,
        train_sessions=train_sessions,
        holdout_sessions=holdout_sessions,
        game_id=game_id,
        env_name=env_name,
        num_actions=num_actions,
        scored_length=scored_length,
        frame_stack_n=frame_stack_n,
        grayscale=grayscale,
        observation_shape=observation_shape,
        observation_dtype=observation_dtype,
        target_size=(height, width),
        observation_low=observation_low,
        observation_high=observation_high,
    )


def _validate_embedded_checkpoint_flags(
    checkpoint: Mapping[str, Any],
    config_flags: Any,
    spec: EvaluationSpec,
    *,
    label: str,
) -> Mapping[str, Any]:
    embedded = checkpoint.get("flags")
    if not isinstance(embedded, Mapping):
        raise ValueError(f"{label} checkpoint lacks embedded training flags")

    protocol = {**DYNAMIC_PROTOCOL, **required_imitation_protocol(spec)}
    for key, expected in protocol.items():
        actual = embedded.get(key)
        if not _protocol_value_matches(actual, expected):
            raise ValueError(
                f"{label} checkpoint embedded protocol mismatch: "
                f"{key}={actual!r}, expected {expected!r}"
            )
        configured = getattr(config_flags, key, None)
        if not _protocol_value_matches(actual, configured):
            raise ValueError(
                f"{label} checkpoint and config_c.yaml disagree on {key}: "
                f"{actual!r} versus {configured!r}"
            )

    for key in RUNTIME_SEMANTIC_FIELDS:
        actual = embedded.get(key)
        if key == "model_float16" and actual is None:
            actual = embedded.get("float16")
        if key == "model_state_projection" and actual is None:
            actual = "none"
        if key == "model_state_range_loss_cost" and actual is None:
            actual = 0.0
        if key == "schedule_total_steps" and actual is None:
            actual = embedded.get("total_steps")
        configured = getattr(config_flags, key, None)
        if key == "model_state_projection" and configured is None:
            configured = "none"
        if key == "model_state_range_loss_cost" and configured is None:
            configured = 0.0
        if not _protocol_value_matches(actual, configured):
            raise ValueError(
                f"{label} checkpoint and config_c.yaml disagree on runtime "
                f"semantic {key}: {actual!r} versus {configured!r}"
            )

    identity = {
        "name": spec.env_name,
        "icopro_game_id": spec.game_id,
        "frame_stack_n": spec.frame_stack_n,
        "grayscale": spec.grayscale,
        "envpool": bool(getattr(config_flags, "envpool", False)),
    }
    configured_xpid = str(getattr(config_flags, "xpid", "")).strip()
    if configured_xpid:
        identity["xpid"] = configured_xpid
    for key, expected in identity.items():
        actual = embedded.get(key)
        if not _protocol_value_matches(actual, expected):
            raise ValueError(
                f"{label} checkpoint embedded identity mismatch: "
                f"{key}={actual!r}, expected {expected!r}"
            )
        configured = getattr(config_flags, key, None)
        if not _protocol_value_matches(actual, configured):
            raise ValueError(
                f"{label} checkpoint and config_c.yaml disagree on {key}: "
                f"{actual!r} versus {configured!r}"
            )

    for name, expected in {
        "icopro_subjects": spec.subjects,
        "icopro_train_sessions": spec.train_sessions,
        "icopro_holdout_sessions": spec.holdout_sessions,
    }.items():
        actual = _parse_id_setting(embedded.get(name), name)
        if actual != expected:
            raise ValueError(
                f"{label} checkpoint embedded {name}={actual}, expected {expected}"
            )
    if not str(embedded.get("icopro_data_path", "")).strip():
        raise ValueError(f"{label} checkpoint embedded icopro_data_path is empty")
    return embedded


def validate_actor_imitation_checkpoint(
    checkpoint: Mapping[str, Any], config_flags: Any, spec: EvaluationSpec
) -> Dict[str, Any]:
    """Prove that the actor weights contain completed imitation updates.

    ``config_c.yaml`` describes an intended run, but it can coexist with an
    actor snapshot saved before behavioral learning began.  The evaluator is
    deliberately fail-closed: it requires the state written by the imitation
    learner and verifies that the checkpoint-embedded flags agree with the
    external configuration used to reconstruct the networks.
    """

    try:
        update_count = int(checkpoint.get("imitation_update_count", 0))
        schedule_step = int(checkpoint.get("imitation_schedule_step", 0))
    except (TypeError, ValueError) as error:
        raise ValueError("actor checkpoint has invalid imitation counters") from error
    if update_count <= 0:
        raise ValueError(
            "actor checkpoint contains no completed Dynamic imitation updates"
        )
    if schedule_step < update_count:
        raise ValueError(
            "actor checkpoint imitation schedule precedes its update count"
        )
    try:
        actor_step = int(checkpoint.get("step", 0))
        actor_real_step = int(checkpoint.get("real_step", 0))
    except (TypeError, ValueError) as error:
        raise ValueError("actor checkpoint has invalid progress counters") from error
    if actor_step <= 0 or actor_real_step <= 0:
        raise ValueError(
            "actor checkpoint was saved before any training progress "
            f"(step={actor_step}, real_step={actor_real_step})"
        )

    signature = checkpoint.get("imitation_data_signature")
    if not isinstance(signature, str) or len(signature) != 64:
        raise ValueError(
            "actor checkpoint lacks a valid behavioral-data signature"
        )
    try:
        int(signature, 16)
    except ValueError as error:
        raise ValueError(
            "actor checkpoint behavioral-data signature is not SHA-256"
        ) from error

    embedded = _validate_embedded_checkpoint_flags(
        checkpoint, config_flags, spec, label="actor"
    )

    prior_state = checkpoint.get("action_prior_ema")
    if float(embedded.get("action_prior_weight", 0.0)) > 0.0:
        if prior_state is None:
            raise ValueError(
                "actor checkpoint has action_prior_weight>0 but no saved prior EMA"
            )
        prior_array = _array(prior_state, np.float64).reshape(-1)
        if (
            prior_array.shape != (spec.num_actions,)
            or not np.all(np.isfinite(prior_array))
            or float(prior_array.sum()) <= 0.0
        ):
            raise ValueError("actor checkpoint contains an invalid action-prior EMA")

    return {
        "imitation_update_count": update_count,
        "imitation_schedule_step": schedule_step,
        "step": actor_step,
        "real_step": actor_real_step,
        "imitation_data_signature": signature,
        "action_prior_ema_saved": prior_state is not None,
        "embedded_protocol_verified": True,
    }


def verify_actor_behavioral_data_signature(
    checkpoint: Mapping[str, Any], computed_signature: str
) -> Dict[str, Any]:
    """Require the evaluation training files to equal the checkpoint corpus."""

    saved_signature = checkpoint.get("imitation_data_signature")
    if saved_signature != computed_signature:
        raise ValueError(
            "actor checkpoint behavioral-data signature does not match the "
            "selected training sessions"
        )
    return {
        "imitation_data_signature": saved_signature,
        "training_data_signature_recomputed": True,
    }


def validate_model_checkpoint(
    checkpoint: Mapping[str, Any], config_flags: Any, spec: EvaluationSpec
) -> Dict[str, Any]:
    """Reject model weights paired with another environment or protocol."""

    embedded = _validate_embedded_checkpoint_flags(
        checkpoint, config_flags, spec, label="model"
    )
    if not bool(embedded.get("train_model", False)):
        raise ValueError("model checkpoint was not produced by train_model=True")
    if not bool(getattr(config_flags, "train_model", False)):
        raise ValueError("config_c.yaml does not describe a trained ModelNet run")
    try:
        step = int(checkpoint.get("step", 0))
        real_step = int(checkpoint.get("real_step", 0))
    except (TypeError, ValueError) as error:
        raise ValueError("model checkpoint has invalid progress counters") from error
    if step <= 0 or real_step <= 0:
        raise ValueError(
            "model checkpoint was saved before a completed model update "
            f"(step={step}, real_step={real_step})"
        )
    _state_dict(checkpoint, "model_net_state_dict")
    return {
        "embedded_protocol_verified": True,
        "train_model": True,
        "step": step,
        "real_step": real_step,
    }


def _load_flags(checkpoint_dir: Path) -> Any:
    from thinker import util

    config_path = checkpoint_dir / "config_c.yaml"
    if config_path.is_file():
        with config_path.open("r", encoding="utf-8") as handle:
            saved_config = yaml.safe_load(handle) or {}
    else:
        # Unit-level callers may replace create_flags with an in-memory
        # namespace.  Production evaluation still fails in create_flags when
        # the required config file is absent.
        saved_config = {}
    flags = util.create_flags(
        ["default_thinker.yaml", "default_actor.yaml"],
        save_flags=False,
        post_fn=util.process_flags_actor,
        config=str(config_path),
        ckp=False,
    )
    flags.ckpdir = str(checkpoint_dir)
    flags.checkpoint_git_revision = saved_config.get("git_revision")
    flags.parallel = False
    flags.parallel_actor = False
    if not bool(getattr(flags, "dynamic_search", False)):
        raise ValueError("checkpoint is not a Dynamic Thinker checkpoint")
    for key, expected in {**DYNAMIC_PROTOCOL, **IMITATION_PROTOCOL}.items():
        actual = getattr(flags, key, None)
        if not _protocol_value_matches(actual, expected):
            raise ValueError(
                f"checkpoint protocol mismatch: {key}={actual!r}, expected {expected!r}"
            )
    imitation_path = str(getattr(flags, "icopro_data_path", "")).strip()
    if not imitation_path:
        raise ValueError(
            "checkpoint protocol mismatch: icopro_data_path is empty; this is "
            "not an imitation-trained checkpoint"
        )
    return flags


def _state_dict(checkpoint: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    if key in checkpoint:
        return checkpoint[key]
    # A plain state_dict remains useful for manually exported checkpoints.
    if checkpoint and all(isinstance(name, str) for name in checkpoint):
        if all(torch.is_tensor(value) or isinstance(value, np.ndarray) for value in checkpoint.values()):
            return checkpoint
    raise KeyError(f"checkpoint does not contain {key!r}")


def _vector_actor_observation_space(template_space: Any, batch_size: int) -> Any:
    """Restore the leading vector dimension omitted by the replay env.

    Online ``VectorWrap`` exposes real observations as ``[B,C,H,W]`` before
    ``cModelWrapper`` builds the actor observation space.  The behavioral
    replay environment deliberately exposes its single-observation space
    ``[C,H,W]`` so ModelNet can be reconstructed correctly.  Real-state and
    predicted-state spaces therefore need one explicit batch dimension for
    ActorNet, whose constructor removes that leading dimension.
    """
    from gymnasium import spaces

    if not isinstance(template_space, spaces.Dict):
        raise TypeError("cModelWrapper observation_space must be spaces.Dict")
    vector_spaces = dict(template_space.spaces)
    for name in ("real_states", "xs"):
        space = vector_spaces.get(name)
        if space is None:
            continue
        low = np.broadcast_to(space.low, (batch_size,) + tuple(space.shape))
        high = np.broadcast_to(space.high, (batch_size,) + tuple(space.shape))
        vector_spaces[name] = spaces.Box(
            low=low, high=high, dtype=space.dtype
        )
    for name in ("tree_reps", "hs"):
        space = vector_spaces.get(name)
        if space is not None and space.shape[0] != batch_size:
            raise ValueError(
                f"template {name} already must begin with batch size "
                f"{batch_size}, got {space.shape}"
            )
    return spaces.Dict(vector_spaces)


def build_networks(
    first_batch: Mapping[str, Any],
    device: torch.device,
    flags: Any,
    spec: EvaluationSpec,
    actor_checkpoint: Mapping[str, Any],
    model_checkpoint: Mapping[str, Any],
) -> Tuple[torch.nn.Module, torch.nn.Module, Any]:
    """Reconstruct ActorNet/ModelNet from one behavior batch and config."""

    from thinker.actor_net import ActorNet
    from thinker.cenv import cModelWrapper
    from thinker.dataset_env import BehaviorSequenceVectorEnv
    from thinker.model_net import ModelNet
    from thinker import util
    from gymnasium import spaces

    obs = _array(first_batch["obs_seq"])
    actions = _array(first_batch["actions_seq"], np.int64)
    validate_holdout_batch(first_batch, spec)
    base_env = BehaviorSequenceVectorEnv(
        obs_seq=obs,
        actions_seq=actions,
        rewards_seq=_array(first_batch["rewards_seq"], np.float32),
        done_seq=_array(first_batch["done_seq"], np.bool_),
        truncated_seq=_array(first_batch["truncated_seq"], np.bool_),
        initial_prev_action=_array(first_batch["initial_prev_action"], np.int64),
        score_mask=_array(first_batch["score_mask"], np.bool_),
        num_actions=spec.num_actions,
        observation_space=spaces.Box(
            low=spec.observation_low,
            high=spec.observation_high,
            shape=spec.observation_shape,
            dtype=np.dtype(spec.observation_dtype),
        ),
    )
    primary_action_space = base_env.action_space[0]
    if int(primary_action_space.n) != spec.num_actions:
        raise RuntimeError(
            "behavior replay action space disagrees with the live environment: "
            f"{primary_action_space.n} versus {spec.num_actions}"
        )
    model_net = ModelNet(
        obs_space=base_env.observation_space,
        action_space=primary_action_space,
        flags=flags,
        frame_stack_n=int(getattr(flags, "frame_stack_n", 4)),
    ).to(device)
    model_net.set_weights(_state_dict(model_checkpoint, "model_net_state_dict"))
    model_net.eval()

    template = cModelWrapper(
        env=base_env,
        env_n=actions.shape[0],
        flags=flags,
        model_net=model_net,
        device=device,
        timing=False,
    )
    try:
        actor_observation_space = _vector_actor_observation_space(
            template.observation_space, actions.shape[0]
        )
        actor_net = ActorNet(
            obs_space=actor_observation_space,
            action_space=template.action_space,
            flags=flags,
            tree_rep_meaning=util.get_tree_rep_meaning(
                spec.num_actions, 1, flags
            ),
        ).to(device)
    finally:
        template.close()
    if int(actor_net.num_actions) != spec.num_actions:
        raise RuntimeError(
            "actor action dimension disagrees with the live environment: "
            f"{actor_net.num_actions} versus {spec.num_actions}"
        )
    actor_net.set_weights(_state_dict(actor_checkpoint, "actor_net_state_dict"))
    actor_net.eval()
    return actor_net, model_net, flags


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def default_output_dir(
    checkpoint_dir: str | Path, spec: EvaluationSpec
) -> Path:
    return (
        Path(checkpoint_dir).expanduser().resolve()
        / f"dynamic_imitation_session{spec.holdout_session}_eval"
    )


def evaluate(args: argparse.Namespace) -> Dict[str, Path]:
    from thinker.bc_loader import (
        FrameStackedBehavioralDataLoader,
        behavioral_data_signature,
    )
    from thinker.dynamic_imitation import DynamicImitationRunner

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    _set_pair_seed(int(args.seed))
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    loaded_checkpoint_hashes = checkpoint_hashes(checkpoint_dir)
    completion_state = validate_completion_marker(checkpoint_dir)
    flags = _load_flags(checkpoint_dir)
    flags.checkpoint_completion_state = completion_state
    spec = resolve_evaluation_spec(
        flags,
        expected_env_name=args.expected_env_name,
        expected_game_id=args.expected_game_id,
    )
    stride = spec.scored_length if args.stride is None else int(args.stride)
    if stride not in {1, spec.scored_length}:
        raise ValueError(
            f"--stride must be 1 or checkpoint batch_length={spec.scored_length}"
        )
    actor_checkpoint = torch.load(
        checkpoint_dir / "ckp_actor.tar",
        map_location="cpu",
        weights_only=False,
    )
    model_checkpoint = torch.load(
        checkpoint_dir / "ckp_model.tar",
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint_hashes(checkpoint_dir) != loaded_checkpoint_hashes:
        raise RuntimeError(
            "checkpoint files changed while they were being loaded; retry with "
            "an immutable snapshot"
        )
    flags.actor_checkpoint_imitation_state = validate_actor_imitation_checkpoint(
        actor_checkpoint, flags, spec
    )
    flags.model_checkpoint_validation_state = validate_model_checkpoint(
        model_checkpoint, flags, spec
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else default_output_dir(checkpoint_dir, spec)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "csv": output_dir / "paired_steps.csv",
        "summary": output_dir / "summary.json",
        "manifest": output_dir / "manifest.json",
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "evaluation outputs already exist; pass --overwrite to replace: "
            + ", ".join(str(path) for path in existing)
        )

    device = _resolve_device(args.device)
    loader_kwargs = dict(
        base_path=args.data_root,
        subjects=spec.subjects,
        game_id=spec.game_id,
        split=None,
        scored_length=spec.scored_length,
        frame_stack_n=spec.frame_stack_n,
        target_size=spec.target_size,
        grayscale=spec.grayscale,
        normalize=spec.observation_dtype == "float32",
        decision_hz=15.0,
        num_actions=spec.num_actions,
    )
    train_loader = FrameStackedBehavioralDataLoader(
        **loader_kwargs,
        sessions=spec.train_sessions,
        seed=args.seed,
    )
    computed_training_signature = behavioral_data_signature(
        train_loader, args.data_root
    )
    signature_state = verify_actor_behavioral_data_signature(
        actor_checkpoint, computed_training_signature
    )
    flags.actor_checkpoint_imitation_state = {
        **flags.actor_checkpoint_imitation_state,
        **signature_state,
    }
    loader = FrameStackedBehavioralDataLoader(
        **loader_kwargs,
        sessions=spec.holdout_sessions,
        seed=args.seed,
    )
    if loader.subjects != spec.subjects or loader.sessions != spec.holdout_sessions:
        raise RuntimeError("loader did not preserve the fixed holdout filter")
    if int(loader.num_actions) != spec.num_actions:
        raise RuntimeError(
            "behavior loader action dimension disagrees with the live environment: "
            f"{loader.num_actions} versus {spec.num_actions}"
        )

    iterator = loader.iter_batches(
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        stride=stride,
        sequence_length=spec.scored_length,
    )
    try:
        first_batch = next(iterator)
    except StopIteration as error:
        raise RuntimeError(
            f"holdout session {spec.holdout_session} produced no evaluation windows"
        ) from error
    validate_holdout_batch(first_batch, spec)

    actor_net, model_net, flags = build_networks(
        first_batch,
        device,
        flags,
        spec,
        actor_checkpoint,
        model_checkpoint,
    )
    runner = DynamicImitationRunner(actor_net, model_net, flags, device=device)
    rows: List[Dict[str, Any]] = []
    try:
        for batch_index, batch in enumerate(itertools.chain([first_batch], iterator)):
            validate_holdout_batch(batch, spec)
            pair_seed = int(args.seed) + batch_index
            _set_pair_seed(pair_seed)
            with torch.no_grad():
                no_carry = runner.rollout(
                    batch, tree_carry=False, training=False
                )
            _set_pair_seed(pair_seed)
            with torch.no_grad():
                carry = runner.rollout(batch, tree_carry=True, training=False)
            rows.extend(build_paired_rows(batch, no_carry, carry, spec))
            if (batch_index + 1) % 10 == 0:
                print(
                    f"evaluated {len(rows) // spec.scored_length} windows "
                    f"({len(rows)} scored rows)",
                    flush=True,
                )
    finally:
        runner.close()

    coverage = loader.evaluation_coverage(
        sequence_length=spec.scored_length, stride=stride
    )
    summary = summarize_rows(rows, coverage, stride, spec)
    summary["seed"] = int(args.seed)
    summary["device"] = str(device)
    summary["checkpoint_protocol"] = checkpoint_protocol(flags)
    summary["actor_checkpoint_imitation_state"] = (
        flags.actor_checkpoint_imitation_state
    )
    summary["model_checkpoint_validation_state"] = (
        flags.model_checkpoint_validation_state
    )
    summary["evaluation_spec"] = asdict(spec)
    summary["required_dynamic_protocol"] = dict(DYNAMIC_PROTOCOL)
    summary["required_imitation_protocol"] = required_imitation_protocol(spec)
    write_rows_csv(rows, output_paths["csv"])
    write_json(summary, output_paths["summary"])

    manifest_args = {
        "checkpoint_dir": str(checkpoint_dir),
        "data_root": str(Path(args.data_root).expanduser().resolve()),
        "output_dir": str(output_dir),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "stride": int(stride),
        "device": str(device),
        "subject": spec.subject,
        "session": spec.holdout_session,
        "game": spec.game_id,
        "env_name": spec.env_name,
        "num_actions": spec.num_actions,
        "expected_env_name": args.expected_env_name,
        "expected_game_id": args.expected_game_id,
    }
    manifest = make_manifest(
        checkpoint_dir=checkpoint_dir,
        source_files=loader.data_files,
        training_source_files=train_loader.data_files,
        output_files=(output_paths["csv"], output_paths["summary"]),
        args=manifest_args,
        spec=spec,
        flags=flags,
        expected_checkpoint_hashes=loaded_checkpoint_hashes,
    )
    manifest["data_coverage"] = summary["data_coverage"]
    write_json(manifest, output_paths["manifest"])
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    print(f"wrote evaluation artifacts to {output_dir}")
    return output_paths


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Directory containing config_c.yaml, ckp_actor.tar and ckp_model.tar",
    )
    parser.add_argument(
        "--data-root",
        default=str(repo_root / "behavioral_data_block"),
        help="behavioral_data_block root; identity is read from the checkpoint",
    )
    parser.add_argument(
        "--expected-env-name",
        "--expected-env",
        dest="expected_env_name",
        default=None,
        help=(
            "Assertion only: fail unless config_c.yaml names this environment; "
            "never overrides checkpoint configuration"
        ),
    )
    parser.add_argument(
        "--expected-game-id",
        type=int,
        default=None,
        help=(
            "Assertion only: fail unless the checkpoint has this behavioral "
            "game id; never selects a dataset"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: CHECKPOINT_DIR/dynamic_imitation_session{holdout}_eval",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, or a CUDA device"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help=(
            "Default: checkpoint batch_length for canonical non-overlapping "
            "targets; 1 requests an overlapping-context sensitivity analysis"
        ),
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace the three named outputs"
    )
    parsed = parser.parse_args(argv)
    if parsed.batch_size < 1:
        parser.error("--batch-size must be positive")
    return parsed


def main(argv: Optional[Sequence[str]] = None) -> int:
    evaluate(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
