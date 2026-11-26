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
import copy
import csv
from dataclasses import asdict, dataclass
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import platform
import random
import re
import stat
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
VOC_PROTOCOL_DEFAULTS = {
    "dynamic_voc_mode": "off",
    "voc_loss_cost": 1.0,
    "voc_gate_temperature": 1.0,
    "voc_train_epsilon": 0.02,
    "voc_eval_stochastic": True,
    "voc_dueling_q": True,
    "voc_expected_gate_loss": True,
    "voc_ema_gate_target": True,
    "voc_gate_target_tau": 0.1,
    "voc_dedicated_gate": True,
    "voc_soft_q_bce_gate": True,
    "voc_gate_q_temperature": 0.05,
    "voc_gate_confidence_weighted": True,
    "voc_gate_adam_beta1": 0.9,
    "voc_gate_param_align": False,
    "voc_gate_param_align_coef": 1.0,
    "voc_gate_exact_projection": False,
    "voc_gate_epsilon_greedy_execution": False,
    "voc_gate_execution_epsilon": 0.02,
    "voc_actor_policy_version_barrier": False,
    "voc_actor_policy_bundle_schema_version": 1,
    "voc_actor_policy_barrier_timeout_s": 120.0,
    "voc_actor_policy_ray_max_restarts": 0,
    "voc_actor_policy_ray_max_task_retries": 0,
    "actor_amp_init_scale": 256.0,
    "voc_model_input_seal_schema_version": 0,
    "voc_gate_learning_rate": 0.0003,
    "voc_gate_grad_norm_clipping": 1.0,
    "entropy_r_cost": 0.0,
}
VOC_ACTIVE_ONLY_PROTOCOL_FIELDS = frozenset(
    (
        "entropy_r_cost",
        "voc_ema_gate_target",
        "voc_gate_target_tau",
        "voc_dedicated_gate",
        "voc_soft_q_bce_gate",
        "voc_gate_q_temperature",
        "voc_gate_confidence_weighted",
        "voc_gate_adam_beta1",
        "voc_gate_param_align",
        "voc_gate_param_align_coef",
        "voc_gate_exact_projection",
        "voc_gate_epsilon_greedy_execution",
        "voc_gate_execution_epsilon",
        "voc_actor_policy_version_barrier",
        "voc_actor_policy_bundle_schema_version",
        "voc_actor_policy_barrier_timeout_s",
        "voc_actor_policy_ray_max_restarts",
        "voc_actor_policy_ray_max_task_retries",
        "actor_amp_init_scale",
        "voc_model_input_seal_schema_version",
        "voc_gate_learning_rate",
        "voc_gate_grad_norm_clipping",
    )
)
VOC_GATE_POLICY_LEGACY_SCHEMA_VERSION = 1
VOC_GATE_POLICY_INTERMEDIATE_SCHEMA_VERSION = 2
VOC_GATE_POLICY_SCHEMA_VERSION = 3
VOC_GATE_POLICY_EXACT_PROJECTION_SCHEMA_VERSION = 4
VOC_GATE_POLICY_EPSILON_GREEDY_EXECUTION_SCHEMA_VERSION = 5
VOC_GATE_POLICY_VERSION_BARRIER_SCHEMA_VERSION = 6
VOC_GATE_POLICY_MODEL_INPUT_SEAL_SCHEMA_VERSION = 7
VOC_GATE_POLICY_HALF_SQUARED_Q_SCHEMA_VERSION = 8
VOC_GATE_POLICY_COMMON_MODE_Q_SCHEMA_VERSION = 9
VOC_GATE_POLICY_HUBER_COMMON_Q_SCHEMA_VERSION = 10
VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION = 11
VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION = 12
VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION = 13
VOC_TELEMETRY_SCHEMA_VERSION = 1
ATOMIC_ACTOR_POLICY_EVIDENCE_FIELDS = frozenset(
    {
        "actor_amp_consecutive_skips",
        "actor_amp_growth_tracker",
        "actor_amp_init_scale",
        "actor_amp_scale",
        "actor_amp_skip_count",
        "voc_actor_policy_barrier_timeout_count",
        "voc_actor_policy_bundle_summary",
        "voc_actor_policy_expected_ack_count",
        "voc_actor_policy_final_publication_event",
        "voc_actor_policy_malformed_bundle_count",
        "voc_actor_policy_publication_count",
        "voc_actor_policy_publication_event_count",
        "voc_actor_policy_publication_history",
        "voc_actor_policy_publication_history_sha256",
        "voc_actor_policy_state_sha256",
        "voc_actor_policy_terminal",
        "voc_actor_policy_terminal_ack_count",
        "voc_actor_policy_version",
        "voc_actor_policy_version_mismatch_count",
    }
)
VOC_GATE_POLICY_SCHEMA7_COMPLETE_IDENTITY_KEY_COUNT = 229
VOC_GATE_POLICY_SCHEMA7_V12_PROJECTION_KEY_COUNT = 209
VOC_GATE_POLICY_SCHEMA7_V12_PROJECTION_SHA256 = (
    "bd386ad1c78f030409562dea5c34d8866cc128cf99b7f39938f6a090f5987407"
)
VOC_GATE_POLICY_SCHEMA8_COMPLETE_IDENTITY_KEY_COUNT = 229
VOC_GATE_POLICY_SCHEMA8_V12_PROJECTION_KEY_COUNT = 209
VOC_GATE_POLICY_SCHEMA8_V12_PROJECTION_SHA256 = (
    VOC_GATE_POLICY_SCHEMA7_V12_PROJECTION_SHA256
)
VOC_GATE_POLICY_SCHEMA8_Q_REGRESSION_LOSS = "half_squared_td"
VOC_GATE_POLICY_SCHEMA8_STAGE_PROFILES = (
    (
        "enduro-voc-v15-halfsq-eps25-sentinel-wire1200",
        1,
        1_200,
        512,
        41,
        False,
    ),
    (
        "enduro-voc-v15-halfsq-eps25-seed1-qual-fresh-100k",
        1,
        100_000,
        10_000,
        201,
        True,
    ),
    (
        "enduro-voc-v15-halfsq-eps25-seed5-strict-fresh-300k",
        5,
        300_000,
        10_000,
        201,
        True,
    ),
)
VOC_GATE_POLICY_SCHEMA9_COMPLETE_IDENTITY_KEY_COUNT = 229
VOC_GATE_POLICY_SCHEMA9_V12_PROJECTION_KEY_COUNT = 209
VOC_GATE_POLICY_SCHEMA9_V12_PROJECTION_SHA256 = (
    VOC_GATE_POLICY_SCHEMA8_V12_PROJECTION_SHA256
)
VOC_GATE_POLICY_SCHEMA9_Q_REGRESSION_LOSS = "half_squared_td"
VOC_GATE_POLICY_SCHEMA9_Q_RECONSTRUCTION = (
    "detached_value_plus_raw_head_mean_plus_policy_centered_raw_head"
)
VOC_GATE_POLICY_SCHEMA9_STAGE_PROFILES = (
    (
        "enduro-voc-v16-commonmode-eps25-sentinel-wire1200",
        1,
        1_200,
        512,
        41,
        False,
    ),
    (
        "enduro-voc-v16-commonmode-eps25-seed1-qual-fresh-100k",
        1,
        100_000,
        10_000,
        201,
        True,
    ),
    (
        "enduro-voc-v16-commonmode-eps25-seed5-strict-fresh-300k",
        5,
        300_000,
        10_000,
        201,
        True,
    ),
)
VOC_GATE_POLICY_SCHEMA10_COMPLETE_IDENTITY_KEY_COUNT = 229
VOC_GATE_POLICY_SCHEMA10_V12_PROJECTION_KEY_COUNT = 209
VOC_GATE_POLICY_SCHEMA10_V12_PROJECTION_SHA256 = (
    VOC_GATE_POLICY_SCHEMA9_V12_PROJECTION_SHA256
)
VOC_GATE_POLICY_SCHEMA10_Q_REGRESSION_LOSS = "smooth_l1_beta1"
VOC_GATE_POLICY_SCHEMA10_Q_RECONSTRUCTION = (
    VOC_GATE_POLICY_SCHEMA9_Q_RECONSTRUCTION
)
VOC_GATE_POLICY_SCHEMA10_STAGE_PROFILES = (
    (
        "enduro-voc-v17-huber-common-eps25-sentinel-wire1200",
        1,
        1_200,
        512,
        41,
        False,
    ),
    (
        "enduro-voc-v17-huber-common-eps25-seed1-qual-fresh-100k",
        1,
        100_000,
        10_000,
        201,
        True,
    ),
    (
        "enduro-voc-v17-huber-common-eps25-seed5-strict-fresh-300k",
        5,
        300_000,
        10_000,
        201,
        True,
    ),
)
VOC_GATE_POLICY_SCHEMA11_COMPLETE_IDENTITY_KEY_COUNT = 229
VOC_GATE_POLICY_SCHEMA11_V12_PROJECTION_KEY_COUNT = 209
VOC_GATE_POLICY_SCHEMA11_V12_PROJECTION_SHA256 = (
    VOC_GATE_POLICY_SCHEMA10_V12_PROJECTION_SHA256
)
VOC_GATE_POLICY_SCHEMA11_Q_REGRESSION_LOSS = (
    VOC_GATE_POLICY_SCHEMA10_Q_REGRESSION_LOSS
)
VOC_GATE_POLICY_SCHEMA11_Q_RECONSTRUCTION = (
    VOC_GATE_POLICY_SCHEMA10_Q_RECONSTRUCTION
)
VOC_GATE_POLICY_SCHEMA11_Q_OPTIMIZER_COORDINATES = (
    "orthonormal_common_difference_adam"
)
VOC_GATE_POLICY_SCHEMA11_STAGE_PROFILES = (
    (
        "enduro-voc-v18-orthocd-adam-eps25-sentinel-wire1200",
        1,
        1_200,
        512,
        41,
        False,
    ),
    (
        "enduro-voc-v18-orthocd-adam-eps25-seed1-qual-fresh-100k",
        1,
        100_000,
        10_000,
        201,
        True,
    ),
    (
        "enduro-voc-v18-orthocd-adam-eps25-seed5-strict-fresh-300k",
        5,
        300_000,
        10_000,
        201,
        True,
    ),
)
VOC_GATE_POLICY_SCHEMA12_COMPLETE_IDENTITY_KEY_COUNT = 229
VOC_GATE_POLICY_SCHEMA12_V12_PROJECTION_KEY_COUNT = 209
VOC_GATE_POLICY_SCHEMA12_V12_PROJECTION_SHA256 = (
    "ad22b91fdd06a30ac7f53c0135b32fac2530687c3c36dad5dccf06d700f83f82"
)
VOC_GATE_POLICY_SCHEMA12_Q_REGRESSION_LOSS = (
    VOC_GATE_POLICY_SCHEMA11_Q_REGRESSION_LOSS
)
VOC_GATE_POLICY_SCHEMA12_Q_RECONSTRUCTION = (
    VOC_GATE_POLICY_SCHEMA11_Q_RECONSTRUCTION
)
VOC_GATE_POLICY_SCHEMA12_Q_OPTIMIZER_COORDINATES = (
    VOC_GATE_POLICY_SCHEMA11_Q_OPTIMIZER_COORDINATES
)
VOC_GATE_POLICY_SCHEMA12_STAGE_PROFILES = (
    (
        "enduro-voc-v19-tau1-orthocd-adam-eps25-sentinel-wire1200",
        1,
        1_200,
        512,
        41,
        False,
    ),
    (
        "enduro-voc-v19-tau1-orthocd-adam-eps25-seed1-qual-fresh-100k",
        1,
        100_000,
        10_000,
        201,
        True,
    ),
    (
        "enduro-voc-v19-tau1-orthocd-adam-eps25-seed5-strict-fresh-300k",
        5,
        300_000,
        10_000,
        201,
        True,
    ),
)
VOC_GATE_POLICY_SCHEMA13_COMPLETE_IDENTITY_KEY_COUNT = 229
VOC_GATE_POLICY_SCHEMA13_V12_PROJECTION_KEY_COUNT = 209
VOC_GATE_POLICY_SCHEMA13_V12_PROJECTION_SHA256 = (
    VOC_GATE_POLICY_SCHEMA12_V12_PROJECTION_SHA256
)
VOC_GATE_POLICY_SCHEMA13_Q_REGRESSION_LOSS = (
    VOC_GATE_POLICY_SCHEMA12_Q_REGRESSION_LOSS
)
VOC_GATE_POLICY_SCHEMA13_Q_RECONSTRUCTION = (
    VOC_GATE_POLICY_SCHEMA12_Q_RECONSTRUCTION
)
VOC_GATE_POLICY_SCHEMA13_Q_OPTIMIZER_COORDINATES = (
    VOC_GATE_POLICY_SCHEMA12_Q_OPTIMIZER_COORDINATES
)
VOC_GATE_POLICY_SCHEMA13_STAGE_PROFILES = (
    (
        "enduro-voc-v20-telemetry-tau1-orthocd-adam-eps25-sentinel-wire1200",
        1,
        1_200,
        512,
        41,
        False,
    ),
    (
        "enduro-voc-v20-telemetry-tau1-orthocd-adam-eps25-seed1-qual-fresh-100k",
        1,
        100_000,
        10_000,
        201,
        True,
    ),
    (
        "enduro-voc-v20-telemetry-tau1-orthocd-adam-eps25-seed5-strict-fresh-300k",
        5,
        300_000,
        10_000,
        201,
        True,
    ),
)
VOC_MODEL_INPUT_SEAL_EVIDENCE_FIELDS = frozenset(
    {
        "voc_model_input_seal_schema_version",
        "voc_model_input_sealed",
        "voc_model_input_seal_count",
        "voc_model_terminal_processed_n",
        "voc_model_terminal_drain_update_count",
        "voc_model_terminal_drain_pre_real_step",
        "voc_model_terminal_drain_pre_grad_step_count_m",
        "voc_model_terminal_drain_pre_grad_step_count_p",
        "voc_model_input_late_write_count",
        "voc_model_input_abort_count",
    }
)
VOC_GATE_ADAM_BETA1_LEGACY_DEFAULT = 0.9
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
    *VOC_PROTOCOL_DEFAULTS,
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
SCHEMA13_COMPLETION_CHECKPOINT_FILES = (
    "config_c.yaml",
    "ckp_actor.tar",
    "ckp_model.tar",
    "voc_telemetry_manifest.json",
)
SCHEMA13_BOUND_RUN_FILES = (
    "config_c.yaml",
    "ckp_actor.tar",
    "ckp_model.tar",
    "voc_telemetry_manifest.json",
    "voc_td_cells.csv",
    "voc_replay_events.csv",
    "voc_q_transactions.csv",
    "voc_telemetry_commits.csv",
    "logs.csv",
    "finish",
)
SCHEMA13_IMPLEMENTATION_SOURCES = frozenset(
    {
        "train.py",
        "thinker/actor_net.py",
        "thinker/bc_loader.py",
        "thinker/cenv.pyx",
        "thinker/dataset_env.py",
        "thinker/dynamic_imitation.py",
        "thinker/gym_add/wrapper.py",
        "thinker/learn_actor.py",
        "thinker/learn_model.py",
        "thinker/logger.py",
        "thinker/main.py",
        "thinker/model_net.py",
        "thinker/self_play.py",
        "thinker/util.py",
        "thinker/voc_telemetry.py",
    }
)


def _completion_json_pairs(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    """Build a JSON object while rejecting duplicate names at every depth."""

    result: Dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(
                f"completion marker contains duplicate JSON key {name!r}"
            )
        result[name] = value
    return result


def _reject_completion_json_constant(value: str) -> None:
    raise ValueError(
        f"completion marker contains non-finite JSON constant {value!r}"
    )


def _completion_file_identity(value: os.stat_result, *, label: str) -> Tuple[int, ...]:
    if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
        raise ValueError(f"{label} must be a regular single-link file")
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_ctime_ns),
        int(value.st_size),
    )


def _read_completion_marker_json(path: Path) -> Dict[str, Any]:
    """Read ``finish`` through one stable, non-symlinked file generation."""

    label = "checkpoint completion marker"
    before = os.lstat(path)
    before_identity = _completion_file_identity(before, label=label)
    descriptor: Optional[int] = None
    try:
        open_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        open_flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, open_flags)
        opened_identity = _completion_file_identity(
            os.fstat(descriptor), label=label
        )
        if opened_identity != before_identity:
            raise RuntimeError(
                "checkpoint completion marker identity changed before read"
            )
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            descriptor = None
            text = handle.read()
            read_identity = _completion_file_identity(
                os.fstat(handle.fileno()), label=label
            )
    finally:
        if descriptor is not None:
            os.close(descriptor)
    after_identity = _completion_file_identity(os.lstat(path), label=label)
    if not (
        before_identity == opened_identity == read_identity == after_identity
    ):
        raise RuntimeError("checkpoint completion marker changed during read")
    try:
        marker = json.loads(
            text,
            object_pairs_hook=_completion_json_pairs,
            parse_constant=_reject_completion_json_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid completion marker JSON: {path}") from error
    if not isinstance(marker, dict):
        raise ValueError("checkpoint completion marker must be a JSON object")
    return marker


def _read_stable_single_link_bytes(path: str | Path, *, label: str) -> bytes:
    """Read one regular single-link generation without following symlinks."""

    path = Path(path)
    before_identity = _completion_file_identity(os.lstat(path), label=label)
    descriptor: Optional[int] = None
    try:
        open_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        open_flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, open_flags)
        opened_identity = _completion_file_identity(
            os.fstat(descriptor), label=label
        )
        if opened_identity != before_identity:
            raise RuntimeError(f"{label} identity changed before read")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        read_identity = _completion_file_identity(
            os.fstat(descriptor), label=label
        )
    finally:
        if descriptor is not None:
            os.close(descriptor)
    after_identity = _completion_file_identity(os.lstat(path), label=label)
    if not (
        before_identity == opened_identity == read_identity == after_identity
    ):
        raise RuntimeError(f"{label} changed during read")
    payload = b"".join(chunks)
    if len(payload) != before_identity[-1]:
        raise RuntimeError(f"{label} size changed during read")
    return payload


def _validate_sha256_record(
    record: Any,
    *,
    label: str,
    require_size: bool,
) -> Tuple[str, Optional[int]]:
    expected_keys = {"sha256", "size"} if require_size else {"sha256"}
    if not isinstance(record, Mapping) or set(record) != expected_keys:
        raise ValueError(f"{label} has invalid fields")
    digest = record["sha256"]
    if (
        type(digest) is not str
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{label} has invalid sha256")
    if not require_size:
        return digest, None
    size = record["size"]
    if type(size) is not int or size <= 0:
        raise ValueError(f"{label} has invalid size")
    return digest, size


def _load_checkpoint_from_completion_bytes(
    checkpoint_dir: str | Path,
    filename: str,
    completion_state: Mapping[str, Any],
    *,
    label: str,
) -> Mapping[str, Any]:
    """Load exactly the stable checkpoint bytes bound by ``finish``."""

    checkpoint_files = completion_state.get("checkpoint_files")
    if not isinstance(checkpoint_files, Mapping) or filename not in checkpoint_files:
        raise ValueError(f"{label} is not bound by the completion marker")
    expected_digest, expected_size = _validate_sha256_record(
        checkpoint_files[filename],
        label=f"completion checkpoint_files[{filename!r}]",
        require_size=True,
    )
    payload = _read_stable_single_link_bytes(
        Path(checkpoint_dir) / filename,
        label=label,
    )
    if len(payload) != expected_size:
        raise RuntimeError(f"{label} size disagrees with the completion marker")
    if hashlib.sha256(payload).hexdigest() != expected_digest:
        raise RuntimeError(f"{label} digest disagrees with the completion marker")
    checkpoint = torch.load(
        io.BytesIO(payload), map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"{label} must deserialize to a mapping")
    return checkpoint


def _load_runtime_checkpoint(
    checkpoint_dir: str | Path,
    filename: str,
    completion_state: Mapping[str, Any],
    *,
    schema13: bool,
    label: str,
) -> Any:
    """Keep legacy pathname loading while binding schema-13 to one generation."""

    if schema13:
        return _load_checkpoint_from_completion_bytes(
            checkpoint_dir,
            filename,
            completion_state,
            label=label,
        )
    return torch.load(
        Path(checkpoint_dir) / filename,
        map_location="cpu",
        weights_only=False,
    )


def validate_completion_marker(checkpoint_dir: str | Path) -> Dict[str, Any]:
    """Verify that ``finish`` binds the final checkpoints and training source."""

    root = Path(checkpoint_dir).expanduser().resolve()
    marker_path = root / "finish"
    marker = _read_completion_marker_json(marker_path)
    base_keys = {
        "schema_version",
        "status",
        "completed_unix",
        "checkpoint_files",
        "implementation_sources",
        "loaded_extensions",
    }
    if set(marker) not in (
        base_keys,
        base_keys | {"voc_actor_policy_logger_completion"},
    ):
        raise ValueError("checkpoint completion marker has invalid fields")
    if type(marker.get("schema_version")) is not int or marker["schema_version"] != 1:
        raise ValueError("checkpoint completion marker is not a completed v1 bundle")
    if type(marker.get("status")) is not str or marker["status"] != "complete":
        raise ValueError("checkpoint completion marker is not a completed v1 bundle")
    completed_unix = marker.get("completed_unix")
    if (
        type(completed_unix) not in (int, float)
        or not np.isfinite(completed_unix)
        or completed_unix <= 0
    ):
        raise ValueError("checkpoint completion marker has invalid completion time")

    recorded_files = marker.get("checkpoint_files")
    expected_checkpoint_names = set(REQUIRED_CHECKPOINT_FILES[:-1])
    if (
        not isinstance(recorded_files, Mapping)
        or set(recorded_files) != expected_checkpoint_names
    ):
        raise ValueError("completion marker has invalid checkpoint file hashes")
    for name in REQUIRED_CHECKPOINT_FILES[:-1]:
        path = root / name
        record = recorded_files.get(name)
        recorded_hash, recorded_size = _validate_sha256_record(
            record,
            label=f"completion marker {name}",
            require_size=True,
        )
        actual_hash = sha256_file(path)
        if recorded_hash != actual_hash:
            raise ValueError(
                f"completion marker does not match final {name}: "
                f"{recorded_hash!r} != {actual_hash!r}"
            )
        if recorded_size != path.stat().st_size:
            raise ValueError(f"completion marker size does not match final {name}")

    recorded_sources = marker.get("implementation_sources")
    if not isinstance(recorded_sources, Mapping) or not recorded_sources:
        raise ValueError("completion marker lacks training implementation hashes")
    package_root = Path(__file__).resolve().parent
    for relative, record in recorded_sources.items():
        if type(relative) is not str or not relative:
            raise ValueError("completion marker has invalid implementation path")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"unsafe implementation path in completion marker: {relative}")
        path = (package_root / relative_path).resolve()
        if package_root not in path.parents and path != package_root:
            raise ValueError(f"implementation path escapes package root: {relative}")
        if not path.is_file():
            raise FileNotFoundError(
                f"training implementation source is unavailable: {path}"
            )
        recorded_hash, _ = _validate_sha256_record(
            record,
            label=f"completion marker implementation {relative}",
            require_size=False,
        )
        if recorded_hash != sha256_file(path):
            raise ValueError(
                f"evaluation source differs from training implementation: {relative}"
            )

    recorded_extensions = marker.get("loaded_extensions")
    if not isinstance(recorded_extensions, Mapping) or not recorded_extensions:
        raise ValueError("completion marker lacks the loaded Cython extension hash")
    for relative, record in recorded_extensions.items():
        if type(relative) is not str or not relative:
            raise ValueError("completion marker has invalid extension path")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"unsafe extension path in completion marker: {relative}")
        path = (package_root / relative_path).resolve()
        if package_root not in path.parents or not path.is_file():
            raise FileNotFoundError(
                f"training Cython extension is unavailable: {path}"
            )
        recorded_hash, _ = _validate_sha256_record(
            record,
            label=f"completion marker extension {relative}",
            require_size=False,
        )
        if recorded_hash != sha256_file(path):
            raise ValueError(
                f"loaded Cython extension differs from training: {relative}"
            )
    return marker


def _validate_schema13_completion_marker_state(
    checkpoint_dir: str | Path, marker: Any
) -> Dict[str, Any]:
    """Validate the schema-13-only completion-v2 surface and bound files."""

    root = Path(checkpoint_dir).expanduser().resolve()
    expected_outer = {
        "schema_version",
        "status",
        "completed_unix",
        "checkpoint_files",
        "implementation_sources",
        "loaded_extensions",
        "voc_actor_policy_logger_completion",
    }
    if type(marker) is not dict or set(marker) != expected_outer:
        raise ValueError("schema-13 completion marker has invalid exact fields")
    if type(marker.get("schema_version")) is not int or marker["schema_version"] != 2:
        raise ValueError("schema-13 completion marker requires exact integer schema 2")
    if type(marker.get("status")) is not str or marker["status"] != "complete":
        raise ValueError("schema-13 checkpoint is not a completed schema-v2 bundle")
    completed_unix = marker.get("completed_unix")
    if (
        type(completed_unix) not in (int, float)
        or isinstance(completed_unix, bool)
        or not np.isfinite(completed_unix)
        or completed_unix <= 0
    ):
        raise ValueError("schema-13 completion marker has invalid completion time")

    recorded_files = marker.get("checkpoint_files")
    if type(recorded_files) is not dict or set(recorded_files) != set(
        SCHEMA13_COMPLETION_CHECKPOINT_FILES
    ):
        raise ValueError(
            "schema-13 completion marker requires exact four checkpoint files"
        )
    for name in SCHEMA13_COMPLETION_CHECKPOINT_FILES:
        expected_hash, expected_size = _validate_sha256_record(
            recorded_files[name],
            label=f"schema-13 completion marker {name}",
            require_size=True,
        )
        payload = _read_stable_single_link_bytes(
            root / name, label=f"schema-13 completion marker {name}"
        )
        if len(payload) != expected_size or hashlib.sha256(payload).hexdigest() != expected_hash:
            raise ValueError(
                f"schema-13 completion marker disagrees with final {name}"
            )

    recorded_sources = marker.get("implementation_sources")
    if type(recorded_sources) is not dict or set(recorded_sources) != set(
        SCHEMA13_IMPLEMENTATION_SOURCES
    ):
        raise ValueError(
            "schema-13 completion marker requires exact 15 implementation sources"
        )
    package_root = Path(__file__).resolve().parent
    for relative in sorted(recorded_sources):
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(
                f"unsafe schema-13 implementation path: {relative}"
            )
        path = (package_root / relative_path).resolve()
        if package_root not in path.parents and path != package_root:
            raise ValueError(
                f"schema-13 implementation path escapes package root: {relative}"
            )
        expected_hash, _ = _validate_sha256_record(
            recorded_sources[relative],
            label=f"schema-13 completion implementation {relative}",
            require_size=False,
        )
        payload = _read_stable_single_link_bytes(
            path, label=f"schema-13 implementation {relative}"
        )
        if hashlib.sha256(payload).hexdigest() != expected_hash:
            raise ValueError(
                f"evaluation source differs from schema-13 training: {relative}"
            )

    recorded_extensions = marker.get("loaded_extensions")
    if type(recorded_extensions) is not dict or not recorded_extensions:
        raise ValueError("schema-13 completion marker lacks loaded extensions")
    for relative, record in recorded_extensions.items():
        if type(relative) is not str or not relative:
            raise ValueError("schema-13 completion marker has invalid extension path")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"unsafe schema-13 extension path: {relative}")
        path = (package_root / relative_path).resolve()
        if package_root not in path.parents:
            raise ValueError(f"schema-13 extension path escapes package: {relative}")
        expected_hash, _ = _validate_sha256_record(
            record,
            label=f"schema-13 completion extension {relative}",
            require_size=False,
        )
        payload = _read_stable_single_link_bytes(
            path, label=f"schema-13 loaded extension {relative}"
        )
        if hashlib.sha256(payload).hexdigest() != expected_hash:
            raise ValueError(
                f"loaded extension differs from schema-13 training: {relative}"
            )
    return copy.deepcopy(marker)


def validate_schema13_completion_marker(
    checkpoint_dir: str | Path,
) -> Dict[str, Any]:
    """Stable-read and validate the dedicated schema-13 completion marker."""

    root = Path(checkpoint_dir).expanduser().resolve()
    marker = _read_completion_marker_json(root / "finish")
    return _validate_schema13_completion_marker_state(root, marker)


def _schema13_checkpoint_hashes(
    checkpoint_dir: str | Path,
    *,
    completion_state: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    """Hash the inherited checkpoint quartet under completion schema 2."""

    root = Path(checkpoint_dir).expanduser().resolve()
    marker = (
        _validate_schema13_completion_marker_state(root, dict(completion_state))
        if completion_state is not None
        else validate_schema13_completion_marker(root)
    )
    result: Dict[str, str] = {}
    for name in SCHEMA13_BOUND_RUN_FILES:
        payload = _read_stable_single_link_bytes(
            root / name, label=f"schema-13 checkpoint {name}"
        )
        result[name] = hashlib.sha256(payload).hexdigest()
    if marker != validate_schema13_completion_marker(root):
        raise RuntimeError("schema-13 completion marker changed during checkpoint hash")
    return result


def validate_schema6_completed_bundle(
    checkpoint_dir: str | Path,
    *,
    completion_state: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Validate and expose a completed schema-6 bundle without tensor leakage.

    Legacy schemas keep their historical public-evaluation path.  Schema 6 is
    fail-closed: its immutable three-surface identity, complete actor/model
    training state, terminal publication history, public logger attestation,
    and private-marker cleanup are all validated before an evaluation-only
    flags copy may disable the live barrier.
    """

    from thinker import util

    root = Path(checkpoint_dir).expanduser().resolve()
    config_path = root / "config_c.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"checkpoint is missing config_c.yaml: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint config_c.yaml must contain a mapping")
    raw_schema = config.get("voc_gate_policy_schema_version")
    if raw_schema != VOC_GATE_POLICY_VERSION_BARRIER_SCHEMA_VERSION:
        return None
    if isinstance(raw_schema, (bool, np.bool_)) or not isinstance(
        raw_schema, (int, np.integer)
    ):
        raise ValueError("schema-6 config requires a non-boolean integer schema")

    marker = (
        dict(completion_state)
        if completion_state is not None
        else validate_completion_marker(root)
    )
    full = util.validate_schema6_final_bundle(
        root, label="public schema-6 completed bundle"
    )
    post_marker = validate_completion_marker(root)
    if marker != post_marker:
        raise RuntimeError("schema-6 completion marker changed during validation")

    completion_evidence = full.get("completion_evidence")
    marker_evidence = {
        name: marker.get(name)
        for name in (
            "checkpoint_files",
            "implementation_sources",
            "loaded_extensions",
        )
    }
    if completion_evidence != marker_evidence:
        raise ValueError(
            "schema-6 public finish evidence disagrees with the validated bundle"
        )

    logger_completion = util.validate_actor_policy_logger_completion(
        marker.get("voc_actor_policy_logger_completion")
    )
    actor_policy = full.get("actor_policy")
    if not isinstance(actor_policy, Mapping):
        raise ValueError("schema-6 final validation lacks actor-policy evidence")
    for completion_name, actor_name in (
        ("policy_version", "voc_actor_policy_version"),
        ("state_sha256", "voc_actor_policy_state_sha256"),
        (
            "publication_history_sha256",
            "voc_actor_policy_publication_history_sha256",
        ),
    ):
        if logger_completion[completion_name] != actor_policy.get(actor_name):
            raise ValueError(
                "schema-6 logger completion disagrees with terminal actor "
                f"evidence: {completion_name}"
            )
    if logger_completion["checkpoint_files"] != marker_evidence["checkpoint_files"]:
        raise ValueError(
            "schema-6 logger completion disagrees with public checkpoint hashes"
        )
    if logger_completion["use_wandb"] != full.get("config_use_wandb"):
        raise ValueError(
            "schema-6 logger completion disagrees with immutable use_wandb"
        )

    private_markers = (
        util.VOC_ACTOR_POLICY_LOGGER_FINISH_REQUEST_FILE,
        util.VOC_ACTOR_POLICY_LOGGER_FINISH_ACK_FILE,
    )
    remaining = [name for name in private_markers if os.path.lexists(root / name)]
    if remaining:
        raise ValueError(
            "schema-6 completed bundle retains private logger marker(s): "
            + ", ".join(remaining)
        )

    resolved_identity = full.get("resolved_identity")
    stored_surfaces = {
        name: copy.deepcopy(resolved_identity)
        for name in ("config", "actor_checkpoint", "model_checkpoint")
    }
    record = {
        **copy.deepcopy(full),
        "stored_surface_identity": stored_surfaces,
        "logger_completion": logger_completion,
        "private_logger_markers_absent": True,
        "public_finish_verified": True,
    }
    try:
        json.dumps(
            record,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "schema-6 public validation record is not strict JSON-safe"
        ) from error
    return record


def _require_schema7_public_model_input_seal(
    full: Mapping[str, Any],
) -> Dict[str, Any]:
    """Independently bind the JSON-safe schema-7 ModelNet seal summary."""

    expected_full_fields = {
        "actor_policy",
        "resolved_identity",
        "actor_training_state",
        "model_step",
        "model_real_step",
        "model_state_tensor_count",
        "model_optimizer_state",
        "model_scheduler_state",
        "model_scaler_state",
        "config_use_wandb",
        "completion_evidence",
        "model_input_seal",
    }
    if set(full) != expected_full_fields:
        raise ValueError("schema-7 authoritative bundle has invalid fields")
    resolved = full.get("resolved_identity")
    if not isinstance(resolved, Mapping):
        raise ValueError("schema-7 authoritative bundle lacks resolved identity")
    for name, expected in (
        ("gate_schema", VOC_GATE_POLICY_MODEL_INPUT_SEAL_SCHEMA_VERSION),
        (
            "voc_gate_policy_schema_version",
            VOC_GATE_POLICY_MODEL_INPUT_SEAL_SCHEMA_VERSION,
        ),
        ("voc_model_input_seal_schema_version", 1),
        ("key_count", VOC_GATE_POLICY_SCHEMA7_COMPLETE_IDENTITY_KEY_COUNT),
        (
            "v12_projection_key_count",
            VOC_GATE_POLICY_SCHEMA7_V12_PROJECTION_KEY_COUNT,
        ),
    ):
        value = resolved.get(name)
        if type(value) is not int or value != expected:
            raise ValueError(
                f"schema-7 authoritative identity requires {name}={expected}"
            )
    projection_digest = resolved.get("v12_projection_sha256")
    if projection_digest != VOC_GATE_POLICY_SCHEMA7_V12_PROJECTION_SHA256:
        raise ValueError("schema-7 v12 projection digest disagrees")
    complete_digest = resolved.get("complete_surface_sha256")
    if (
        type(complete_digest) is not str
        or len(complete_digest) != 64
        or any(character not in "0123456789abcdef" for character in complete_digest)
    ):
        raise ValueError("schema-7 complete-surface digest is invalid")
    stage = resolved.get("stage")
    if not isinstance(stage, (list, tuple)) or len(stage) != 6:
        raise ValueError("schema-7 authoritative identity lacks a closed stage")
    total_steps = stage[2]
    if type(total_steps) is not int or total_steps <= 0:
        raise ValueError("schema-7 stage total_steps is invalid")

    evidence = full.get("model_input_seal")
    if not isinstance(evidence, Mapping) or set(evidence) != set(
        VOC_MODEL_INPUT_SEAL_EVIDENCE_FIELDS
    ):
        raise ValueError("schema-7 final bundle lacks exact ModelNet seal evidence")
    if evidence.get("voc_model_input_sealed") is not True:
        raise ValueError("schema-7 ModelNet input must be sealed")
    integer_fields = VOC_MODEL_INPUT_SEAL_EVIDENCE_FIELDS - {
        "voc_model_input_sealed"
    }
    for name in integer_fields:
        if type(evidence.get(name)) is not int:
            raise ValueError(f"schema-7 ModelNet seal {name} must be Python int")
    if evidence["voc_model_input_seal_schema_version"] != 1:
        raise ValueError("schema-7 ModelNet seal schema must equal one")
    if evidence["voc_model_input_seal_count"] != 1:
        raise ValueError("schema-7 ModelNet must be sealed exactly once")
    if evidence["voc_model_input_late_write_count"] != 0:
        raise ValueError("schema-7 ModelNet has a late input write")
    if evidence["voc_model_input_abort_count"] != 0:
        raise ValueError("schema-7 ModelNet input was aborted")
    drain = evidence["voc_model_terminal_drain_update_count"]
    if drain not in (0, 1):
        raise ValueError("schema-7 ModelNet drain count must be zero or one")
    terminal = evidence["voc_model_terminal_processed_n"]
    model_real_step = full.get("model_real_step")
    if (
        type(model_real_step) is not int
        or terminal != model_real_step
        or terminal < total_steps
    ):
        raise ValueError("schema-7 ModelNet terminal progress disagrees")
    pre_real = evidence["voc_model_terminal_drain_pre_real_step"]
    pre_m = evidence["voc_model_terminal_drain_pre_grad_step_count_m"]
    pre_p = evidence["voc_model_terminal_drain_pre_grad_step_count_p"]
    if pre_real < 0 or pre_m < 0 or pre_p < 0:
        raise ValueError("schema-7 ModelNet pre-drain counters are negative")
    if drain == 0 and pre_real != terminal:
        raise ValueError("schema-7 zero-drain progress disagrees")
    if drain == 1 and pre_real >= terminal:
        raise ValueError("schema-7 one-drain progress did not advance")

    optimizers = full.get("model_optimizer_state")
    schedulers = full.get("model_scheduler_state")
    if (
        not isinstance(optimizers, Mapping)
        or set(optimizers) != {"m", "p"}
        or not isinstance(schedulers, Mapping)
        or set(schedulers) != {"m", "p"}
    ):
        raise ValueError("schema-7 ModelNet optimizer/scheduler summary is incomplete")
    for component, pre_count in (("m", pre_m), ("p", pre_p)):
        optimizer = optimizers[component]
        scheduler = schedulers[component]
        if (
            not isinstance(optimizer, Mapping)
            or type(optimizer.get("expected_step")) is not int
            or optimizer["expected_step"] != pre_count + drain
            or optimizer["expected_step"] <= 0
            or not isinstance(scheduler, Mapping)
            or type(scheduler.get("last_epoch")) is not int
            or scheduler["last_epoch"] != terminal
            or type(scheduler.get("step_count")) is not int
            or scheduler["step_count"] != optimizer["expected_step"] + 1
        ):
            raise ValueError(
                f"schema-7 ModelNet {component} terminal state disagrees"
            )
    if optimizers["m"]["expected_step"] != optimizers["p"]["expected_step"]:
        raise ValueError("schema-7 ModelNet m/p update counts are not lockstep")
    if full.get("model_scaler_state") != {}:
        raise ValueError("schema-7 FP32 ModelNet unexpectedly has scaler state")
    return copy.deepcopy(dict(evidence))


def _require_schema89_public_model_input_seal(
    full: Mapping[str, Any],
    *,
    schema_version: int,
    complete_identity_key_count: int,
    projection_key_count: int,
    projection_sha256: str,
    stage_profiles: Sequence[Tuple[Any, ...]],
    q_regression_loss: str,
    q_reconstruction: Optional[str],
    q_optimizer_coordinates: Optional[str] = None,
    q_regression_error: str = "voc_q_regression_loss='half_squared_td'",
) -> Dict[str, Any]:
    """Independently bind a JSON-safe schema-8/9 ModelNet seal summary."""

    schema_label = f"schema-{schema_version}"

    expected_full_fields = {
        "actor_policy",
        "resolved_identity",
        "actor_training_state",
        "model_step",
        "model_real_step",
        "model_state_tensor_count",
        "model_optimizer_state",
        "model_scheduler_state",
        "model_scaler_state",
        "config_use_wandb",
        "completion_evidence",
        "model_input_seal",
    }
    if schema_version == VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION:
        expected_full_fields.add("telemetry")
    if set(full) != expected_full_fields:
        raise ValueError(f"{schema_label} authoritative bundle has invalid fields")
    resolved = full.get("resolved_identity")
    expected_resolved_fields = {
        "gate_schema",
        "voc_gate_policy_schema_version",
        "voc_model_input_seal_schema_version",
        "voc_q_regression_loss",
        "key_count",
        "v12_projection_key_count",
        "v12_projection_sha256",
        "complete_surface_sha256",
        "stage",
        "paths",
    }
    if q_reconstruction is not None:
        expected_resolved_fields.add("voc_q_reconstruction")
    if q_optimizer_coordinates is not None:
        expected_resolved_fields.add("voc_q_optimizer_coordinates")
    if not isinstance(resolved, Mapping) or set(resolved) != expected_resolved_fields:
        raise ValueError(
            f"{schema_label} authoritative bundle lacks exact resolved identity"
        )
    for name, expected in (
        ("gate_schema", schema_version),
        (
            "voc_gate_policy_schema_version",
            schema_version,
        ),
        ("voc_model_input_seal_schema_version", 1),
        ("key_count", complete_identity_key_count),
        ("v12_projection_key_count", projection_key_count),
    ):
        value = resolved.get(name)
        if type(value) is not int or value != expected:
            raise ValueError(
                f"{schema_label} authoritative identity requires {name}={expected}"
            )
    if resolved.get("voc_q_regression_loss") != q_regression_loss:
        raise ValueError(
            f"{schema_label} authoritative identity requires "
            f"{q_regression_error}"
        )
    if (
        q_reconstruction is not None
        and resolved.get("voc_q_reconstruction") != q_reconstruction
    ):
        raise ValueError(
            f"{schema_label} authoritative identity requires exact "
            "voc_q_reconstruction"
        )
    if (
        q_optimizer_coordinates is not None
        and resolved.get("voc_q_optimizer_coordinates")
        != q_optimizer_coordinates
    ):
        raise ValueError(
            f"{schema_label} authoritative identity requires exact "
            "voc_q_optimizer_coordinates"
        )
    if resolved.get("v12_projection_sha256") != projection_sha256:
        raise ValueError(f"{schema_label} v12 projection digest disagrees")
    complete_digest = resolved.get("complete_surface_sha256")
    if (
        type(complete_digest) is not str
        or len(complete_digest) != 64
        or any(character not in "0123456789abcdef" for character in complete_digest)
    ):
        raise ValueError(f"{schema_label} complete-surface digest is invalid")
    stage = resolved.get("stage")
    if (
        type(stage) not in (list, tuple)
        or len(stage) != 6
        or type(stage[0]) is not str
        or any(type(value) is not int for value in stage[1:5])
        or type(stage[5]) is not bool
        or tuple(stage) not in stage_profiles
    ):
        raise ValueError(f"{schema_label} authoritative identity lacks a closed stage")
    paths = resolved.get("paths")
    if type(paths) is not dict or set(paths) != {
        "savedir",
        "ckpdir",
        "cmd",
        "icopro_data_path",
    }:
        raise ValueError(f"{schema_label} authoritative identity lacks exact paths")
    if any(type(value) is not str or not value for value in paths.values()):
        raise ValueError(
            f"{schema_label} authoritative identity has invalid path values"
        )
    for name in ("savedir", "ckpdir", "icopro_data_path"):
        value = paths[name]
        if (
            not os.path.isabs(value)
            or os.path.normpath(value) != value
            or os.path.realpath(value) != value
        ):
            raise ValueError(
                f"{schema_label} authoritative identity has invalid {name} path"
            )
    xpid = stage[0]
    if (
        paths["ckpdir"] != os.path.join(paths["savedir"], xpid)
        or os.path.basename(paths["ckpdir"]) != xpid
        or paths["icopro_data_path"]
        != os.path.join(
            os.path.dirname(paths["savedir"]),
            "data",
            "behavioral_data_block",
        )
    ):
        raise ValueError(
            f"{schema_label} authoritative identity path relationships disagree"
        )

    evidence = full.get("model_input_seal")
    if not isinstance(evidence, Mapping) or set(evidence) != set(
        VOC_MODEL_INPUT_SEAL_EVIDENCE_FIELDS
    ):
        raise ValueError(
            f"{schema_label} final bundle lacks exact ModelNet seal evidence"
        )
    if evidence.get("voc_model_input_sealed") is not True:
        raise ValueError(f"{schema_label} ModelNet input must be sealed")
    for name in VOC_MODEL_INPUT_SEAL_EVIDENCE_FIELDS - {
        "voc_model_input_sealed"
    }:
        if type(evidence.get(name)) is not int:
            raise ValueError(
                f"{schema_label} ModelNet seal {name} must be Python int"
            )
    for name, expected in (
        ("voc_model_input_seal_schema_version", 1),
        ("voc_model_input_seal_count", 1),
        ("voc_model_input_late_write_count", 0),
        ("voc_model_input_abort_count", 0),
    ):
        if evidence[name] != expected:
            raise ValueError(
                f"{schema_label} ModelNet seal requires {name}={expected}"
            )
    drain = evidence["voc_model_terminal_drain_update_count"]
    terminal = evidence["voc_model_terminal_processed_n"]
    pre_real = evidence["voc_model_terminal_drain_pre_real_step"]
    pre_m = evidence["voc_model_terminal_drain_pre_grad_step_count_m"]
    pre_p = evidence["voc_model_terminal_drain_pre_grad_step_count_p"]
    if drain not in (0, 1) or min(terminal, pre_real, pre_m, pre_p) < 0:
        raise ValueError(f"{schema_label} ModelNet drain evidence is invalid")
    total_steps = tuple(stage)[2]
    model_real_step = full.get("model_real_step")
    if (
        type(model_real_step) is not int
        or terminal != model_real_step
        or terminal < total_steps
    ):
        raise ValueError(f"{schema_label} ModelNet terminal progress disagrees")
    if drain == 0 and pre_real != terminal:
        raise ValueError(f"{schema_label} zero-drain progress disagrees")
    if drain == 1 and pre_real >= terminal:
        raise ValueError(f"{schema_label} one-drain progress did not advance")

    optimizers = full.get("model_optimizer_state")
    schedulers = full.get("model_scheduler_state")
    if (
        not isinstance(optimizers, Mapping)
        or set(optimizers) != {"m", "p"}
        or not isinstance(schedulers, Mapping)
        or set(schedulers) != {"m", "p"}
    ):
        raise ValueError(
            f"{schema_label} ModelNet optimizer/scheduler summary is incomplete"
        )
    for component, pre_count in (("m", pre_m), ("p", pre_p)):
        optimizer = optimizers[component]
        scheduler = schedulers[component]
        if (
            not isinstance(optimizer, Mapping)
            or type(optimizer.get("expected_step")) is not int
            or optimizer["expected_step"] != pre_count + drain
            or optimizer["expected_step"] <= 0
            or not isinstance(scheduler, Mapping)
            or type(scheduler.get("last_epoch")) is not int
            or scheduler["last_epoch"] != terminal
            or type(scheduler.get("step_count")) is not int
            or scheduler["step_count"] != optimizer["expected_step"] + 1
        ):
            raise ValueError(
                f"{schema_label} ModelNet {component} terminal state disagrees"
            )
    if optimizers["m"]["expected_step"] != optimizers["p"]["expected_step"]:
        raise ValueError(
            f"{schema_label} ModelNet m/p update counts are not lockstep"
        )
    if full.get("model_scaler_state") != {}:
        raise ValueError(
            f"{schema_label} FP32 ModelNet unexpectedly has scaler state"
        )
    return copy.deepcopy(dict(evidence))


def _require_schema8_public_model_input_seal(
    full: Mapping[str, Any],
) -> Dict[str, Any]:
    """Independently bind the JSON-safe schema-8 ModelNet seal summary."""

    return _require_schema89_public_model_input_seal(
        full,
        schema_version=VOC_GATE_POLICY_HALF_SQUARED_Q_SCHEMA_VERSION,
        complete_identity_key_count=(
            VOC_GATE_POLICY_SCHEMA8_COMPLETE_IDENTITY_KEY_COUNT
        ),
        projection_key_count=VOC_GATE_POLICY_SCHEMA8_V12_PROJECTION_KEY_COUNT,
        projection_sha256=VOC_GATE_POLICY_SCHEMA8_V12_PROJECTION_SHA256,
        stage_profiles=VOC_GATE_POLICY_SCHEMA8_STAGE_PROFILES,
        q_regression_loss=VOC_GATE_POLICY_SCHEMA8_Q_REGRESSION_LOSS,
        q_reconstruction=None,
    )


def _require_schema9_public_model_input_seal(
    full: Mapping[str, Any],
) -> Dict[str, Any]:
    """Independently bind the JSON-safe schema-9 ModelNet seal summary."""

    return _require_schema89_public_model_input_seal(
        full,
        schema_version=VOC_GATE_POLICY_COMMON_MODE_Q_SCHEMA_VERSION,
        complete_identity_key_count=(
            VOC_GATE_POLICY_SCHEMA9_COMPLETE_IDENTITY_KEY_COUNT
        ),
        projection_key_count=VOC_GATE_POLICY_SCHEMA9_V12_PROJECTION_KEY_COUNT,
        projection_sha256=VOC_GATE_POLICY_SCHEMA9_V12_PROJECTION_SHA256,
        stage_profiles=VOC_GATE_POLICY_SCHEMA9_STAGE_PROFILES,
        q_regression_loss=VOC_GATE_POLICY_SCHEMA9_Q_REGRESSION_LOSS,
        q_reconstruction=VOC_GATE_POLICY_SCHEMA9_Q_RECONSTRUCTION,
    )


def _require_schema10_public_model_input_seal(
    full: Mapping[str, Any],
) -> Dict[str, Any]:
    """Independently bind the JSON-safe schema-10 ModelNet seal summary."""

    return _require_schema89_public_model_input_seal(
        full,
        schema_version=VOC_GATE_POLICY_HUBER_COMMON_Q_SCHEMA_VERSION,
        complete_identity_key_count=(
            VOC_GATE_POLICY_SCHEMA10_COMPLETE_IDENTITY_KEY_COUNT
        ),
        projection_key_count=VOC_GATE_POLICY_SCHEMA10_V12_PROJECTION_KEY_COUNT,
        projection_sha256=VOC_GATE_POLICY_SCHEMA10_V12_PROJECTION_SHA256,
        stage_profiles=VOC_GATE_POLICY_SCHEMA10_STAGE_PROFILES,
        q_regression_loss=VOC_GATE_POLICY_SCHEMA10_Q_REGRESSION_LOSS,
        q_reconstruction=VOC_GATE_POLICY_SCHEMA10_Q_RECONSTRUCTION,
        q_regression_error="voc_q_regression_loss='smooth_l1_beta1'",
    )


def _require_schema11_public_model_input_seal(
    full: Mapping[str, Any],
) -> Dict[str, Any]:
    """Independently bind the JSON-safe schema-11 ModelNet seal summary."""

    return _require_schema89_public_model_input_seal(
        full,
        schema_version=VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION,
        complete_identity_key_count=(
            VOC_GATE_POLICY_SCHEMA11_COMPLETE_IDENTITY_KEY_COUNT
        ),
        projection_key_count=VOC_GATE_POLICY_SCHEMA11_V12_PROJECTION_KEY_COUNT,
        projection_sha256=VOC_GATE_POLICY_SCHEMA11_V12_PROJECTION_SHA256,
        stage_profiles=VOC_GATE_POLICY_SCHEMA11_STAGE_PROFILES,
        q_regression_loss=VOC_GATE_POLICY_SCHEMA11_Q_REGRESSION_LOSS,
        q_reconstruction=VOC_GATE_POLICY_SCHEMA11_Q_RECONSTRUCTION,
        q_optimizer_coordinates=(
            VOC_GATE_POLICY_SCHEMA11_Q_OPTIMIZER_COORDINATES
        ),
        q_regression_error="voc_q_regression_loss='smooth_l1_beta1'",
    )


def _require_schema12_public_model_input_seal(
    full: Mapping[str, Any],
) -> Dict[str, Any]:
    """Independently bind the JSON-safe schema-12 ModelNet seal summary."""

    return _require_schema89_public_model_input_seal(
        full,
        schema_version=VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION,
        complete_identity_key_count=(
            VOC_GATE_POLICY_SCHEMA12_COMPLETE_IDENTITY_KEY_COUNT
        ),
        projection_key_count=VOC_GATE_POLICY_SCHEMA12_V12_PROJECTION_KEY_COUNT,
        projection_sha256=VOC_GATE_POLICY_SCHEMA12_V12_PROJECTION_SHA256,
        stage_profiles=VOC_GATE_POLICY_SCHEMA12_STAGE_PROFILES,
        q_regression_loss=VOC_GATE_POLICY_SCHEMA12_Q_REGRESSION_LOSS,
        q_reconstruction=VOC_GATE_POLICY_SCHEMA12_Q_RECONSTRUCTION,
        q_optimizer_coordinates=(
            VOC_GATE_POLICY_SCHEMA12_Q_OPTIMIZER_COORDINATES
        ),
        q_regression_error="voc_q_regression_loss='smooth_l1_beta1'",
    )


def _require_schema13_public_model_input_seal(
    full: Mapping[str, Any],
) -> Dict[str, Any]:
    """Independently bind the JSON-safe schema-13 ModelNet seal summary."""

    return _require_schema89_public_model_input_seal(
        full,
        schema_version=VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION,
        complete_identity_key_count=(
            VOC_GATE_POLICY_SCHEMA13_COMPLETE_IDENTITY_KEY_COUNT
        ),
        projection_key_count=VOC_GATE_POLICY_SCHEMA13_V12_PROJECTION_KEY_COUNT,
        projection_sha256=VOC_GATE_POLICY_SCHEMA13_V12_PROJECTION_SHA256,
        stage_profiles=VOC_GATE_POLICY_SCHEMA13_STAGE_PROFILES,
        q_regression_loss=VOC_GATE_POLICY_SCHEMA13_Q_REGRESSION_LOSS,
        q_reconstruction=VOC_GATE_POLICY_SCHEMA13_Q_RECONSTRUCTION,
        q_optimizer_coordinates=(
            VOC_GATE_POLICY_SCHEMA13_Q_OPTIMIZER_COORDINATES
        ),
        q_regression_error="voc_q_regression_loss='smooth_l1_beta1'",
    )


def _require_schema12_public_ema_online_equality(
    root: Path,
    completion_marker: Mapping[str, Any],
) -> None:
    """Independently compare bound schema-12 raw EMA and online Q tensors."""

    checkpoint_files = completion_marker.get("checkpoint_files")
    if not isinstance(checkpoint_files, Mapping):
        raise ValueError("schema-12 completion marker lacks checkpoint files")
    expected_digest, expected_size = _validate_sha256_record(
        checkpoint_files.get("ckp_actor.tar"),
        label="schema-12 public actor checkpoint",
        require_size=True,
    )
    payload = _read_stable_single_link_bytes(
        root / "ckp_actor.tar",
        label="schema-12 public actor checkpoint",
    )
    if (
        len(payload) != expected_size
        or hashlib.sha256(payload).hexdigest() != expected_digest
    ):
        raise ValueError(
            "schema-12 public actor checkpoint disagrees with completion evidence"
        )
    checkpoint = torch.load(
        io.BytesIO(payload),
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError("schema-12 public actor checkpoint must be a mapping")
    update_count = checkpoint.get("voc_ema_gate_update_count")
    if type(update_count) is not int or update_count < 0:
        raise ValueError(
            "schema-12 public actor checkpoint requires an exact non-negative "
            "Python integer EMA update count"
        )
    if update_count == 0:
        return
    ema_state = checkpoint.get("voc_ema_gate_head_state_dict")
    online_state = checkpoint.get("actor_net_state_dict")
    if not isinstance(ema_state, Mapping) or set(ema_state) != {"weight", "bias"}:
        raise ValueError("schema-12 public actor checkpoint lacks raw EMA state")
    if not isinstance(online_state, Mapping):
        raise ValueError("schema-12 public actor checkpoint lacks online Q state")
    online_pair = next(
        (
            pair
            for pair in (
                ("voc_head.weight", "voc_head.bias"),
                ("critic.voc_head.weight", "critic.voc_head.bias"),
            )
            if all(name in online_state for name in pair)
        ),
        None,
    )
    if online_pair is None:
        raise ValueError(
            "schema-12 public actor checkpoint lacks online raw Q weight/bias"
        )
    for ema_name, online_name in zip(("weight", "bias"), online_pair):
        ema_tensor = ema_state[ema_name]
        online_tensor = online_state[online_name]
        if (
            not isinstance(ema_tensor, torch.Tensor)
            or not isinstance(online_tensor, torch.Tensor)
            or not torch.equal(ema_tensor, online_tensor)
        ):
            raise ValueError(
                f"schema-12 public raw EMA {ema_name} disagrees with stored "
                "online raw Q"
            )


def _require_schema13_public_ema_online_equality(
    root: Path,
    completion_marker: Mapping[str, Any],
) -> None:
    """Independently compare bound schema-13 raw EMA and online Q tensors."""

    checkpoint_files = completion_marker.get("checkpoint_files")
    if not isinstance(checkpoint_files, Mapping):
        raise ValueError("schema-13 completion marker lacks checkpoint files")
    expected_digest, expected_size = _validate_sha256_record(
        checkpoint_files.get("ckp_actor.tar"),
        label="schema-13 public actor checkpoint",
        require_size=True,
    )
    payload = _read_stable_single_link_bytes(
        root / "ckp_actor.tar",
        label="schema-13 public actor checkpoint",
    )
    if (
        len(payload) != expected_size
        or hashlib.sha256(payload).hexdigest() != expected_digest
    ):
        raise ValueError(
            "schema-13 public actor checkpoint disagrees with completion evidence"
        )
    checkpoint = torch.load(
        io.BytesIO(payload),
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError("schema-13 public actor checkpoint must be a mapping")
    update_count = checkpoint.get("voc_ema_gate_update_count")
    if type(update_count) is not int or update_count < 0:
        raise ValueError(
            "schema-13 public actor checkpoint requires an exact non-negative "
            "Python integer EMA update count"
        )
    if update_count == 0:
        return
    ema_state = checkpoint.get("voc_ema_gate_head_state_dict")
    online_state = checkpoint.get("actor_net_state_dict")
    if not isinstance(ema_state, Mapping) or set(ema_state) != {"weight", "bias"}:
        raise ValueError("schema-13 public actor checkpoint lacks raw EMA state")
    if not isinstance(online_state, Mapping):
        raise ValueError("schema-13 public actor checkpoint lacks online Q state")
    online_pair = next(
        (
            pair
            for pair in (
                ("voc_head.weight", "voc_head.bias"),
                ("critic.voc_head.weight", "critic.voc_head.bias"),
            )
            if all(name in online_state for name in pair)
        ),
        None,
    )
    if online_pair is None:
        raise ValueError(
            "schema-13 public actor checkpoint lacks online raw Q weight/bias"
        )
    for ema_name, online_name in zip(("weight", "bias"), online_pair):
        ema_tensor = ema_state[ema_name]
        online_tensor = online_state[online_name]
        if (
            not isinstance(ema_tensor, torch.Tensor)
            or not isinstance(online_tensor, torch.Tensor)
            or not torch.equal(ema_tensor, online_tensor)
        ):
            raise ValueError(
                f"schema-13 public raw EMA {ema_name} disagrees with stored "
                "online raw Q"
            )


def validate_schema7_completed_bundle(
    checkpoint_dir: str | Path,
    *,
    completion_state: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Validate a completed schema-7 bundle before any private eval copy."""

    from thinker import util

    root = Path(checkpoint_dir).expanduser().resolve()
    config_path = root / "config_c.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"checkpoint is missing config_c.yaml: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint config_c.yaml must contain a mapping")
    raw_schema = config.get("voc_gate_policy_schema_version")
    if raw_schema != VOC_GATE_POLICY_MODEL_INPUT_SEAL_SCHEMA_VERSION:
        return None
    if type(raw_schema) is not int:
        raise ValueError("schema-7 config requires a Python integer schema")

    marker = (
        dict(completion_state)
        if completion_state is not None
        else validate_completion_marker(root)
    )
    full = util.validate_schema7_final_bundle(
        root, label="public schema-7 completed bundle"
    )
    model_input_seal = _require_schema7_public_model_input_seal(full)
    post_marker = validate_completion_marker(root)
    if marker != post_marker:
        raise RuntimeError("schema-7 completion marker changed during validation")

    completion_evidence = full.get("completion_evidence")
    marker_evidence = {
        name: marker.get(name)
        for name in (
            "checkpoint_files",
            "implementation_sources",
            "loaded_extensions",
        )
    }
    if completion_evidence != marker_evidence:
        raise ValueError(
            "schema-7 public finish evidence disagrees with the validated bundle"
        )

    logger_completion = util.validate_actor_policy_logger_completion(
        marker.get("voc_actor_policy_logger_completion")
    )
    actor_policy = full.get("actor_policy")
    if not isinstance(actor_policy, Mapping):
        raise ValueError("schema-7 final validation lacks actor-policy evidence")
    bundle_summary = actor_policy.get("voc_actor_policy_bundle_summary")
    if (
        not isinstance(bundle_summary, Mapping)
        or bundle_summary.get("gate_schema")
        != VOC_GATE_POLICY_MODEL_INPUT_SEAL_SCHEMA_VERSION
    ):
        raise ValueError("schema-7 final actor bundle has the wrong gate schema")
    for completion_name, actor_name in (
        ("policy_version", "voc_actor_policy_version"),
        ("state_sha256", "voc_actor_policy_state_sha256"),
        (
            "publication_history_sha256",
            "voc_actor_policy_publication_history_sha256",
        ),
    ):
        if logger_completion[completion_name] != actor_policy.get(actor_name):
            raise ValueError(
                "schema-7 logger completion disagrees with terminal actor "
                f"evidence: {completion_name}"
            )
    if logger_completion["checkpoint_files"] != marker_evidence["checkpoint_files"]:
        raise ValueError(
            "schema-7 logger completion disagrees with public checkpoint hashes"
        )
    if logger_completion["use_wandb"] != full.get("config_use_wandb"):
        raise ValueError(
            "schema-7 logger completion disagrees with immutable use_wandb"
        )

    private_markers = (
        util.VOC_ACTOR_POLICY_LOGGER_FINISH_REQUEST_FILE,
        util.VOC_ACTOR_POLICY_LOGGER_FINISH_ACK_FILE,
    )
    remaining = [name for name in private_markers if os.path.lexists(root / name)]
    if remaining:
        raise ValueError(
            "schema-7 completed bundle retains private logger marker(s): "
            + ", ".join(remaining)
        )

    resolved_identity = full.get("resolved_identity")
    stored_surfaces = {
        name: copy.deepcopy(resolved_identity)
        for name in ("config", "actor_checkpoint", "model_checkpoint")
    }
    record = {
        "authoritative_validator": "thinker.util.validate_schema7_final_bundle",
        **copy.deepcopy(full),
        "model_input_seal": model_input_seal,
        "stored_surface_identity": stored_surfaces,
        "logger_completion": logger_completion,
        "private_logger_markers_absent": True,
        "public_finish_verified": True,
    }
    try:
        json.dumps(
            record,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "schema-7 public validation record is not strict JSON-safe"
        ) from error
    return record


def _validate_schema89_completed_bundle(
    checkpoint_dir: str | Path,
    *,
    completion_state: Optional[Mapping[str, Any]] = None,
    config_payload: Optional[bytes] = None,
    expected_config_sha256: Optional[str] = None,
    schema_version: int,
    util_validator_name: str,
    model_seal_validator: Any,
) -> Optional[Dict[str, Any]]:
    """Validate a completed schema-8/9 bundle before downstream eval load."""

    from thinker import util

    schema_label = f"schema-{schema_version}"
    root = Path(checkpoint_dir).expanduser().resolve()
    config_path = root / "config_c.yaml"
    if config_payload is None:
        if not config_path.is_file():
            raise FileNotFoundError(
                f"checkpoint is missing config_c.yaml: {config_path}"
            )
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    else:
        if type(config_payload) is not bytes:
            raise TypeError(
                f"{schema_label} validation config payload must be exact bytes"
            )
        if (
            type(expected_config_sha256) is not str
            or re.fullmatch(r"[0-9a-f]{64}", expected_config_sha256) is None
            or hashlib.sha256(config_payload).hexdigest()
            != expected_config_sha256
        ):
            raise ValueError(f"{schema_label} validation config digest disagrees")
        try:
            config = yaml.safe_load(config_payload.decode("utf-8"))
        except (UnicodeDecodeError, yaml.YAMLError) as error:
            raise ValueError(
                f"{schema_label} validation config is not UTF-8 YAML"
            ) from error
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint config_c.yaml must contain a mapping")
    raw_schema = config.get("voc_gate_policy_schema_version")
    if type(raw_schema) is not int or raw_schema != schema_version:
        raise ValueError(
            f"dedicated {schema_label} validation requires exact Python integer "
            f"voc_gate_policy_schema_version={schema_version}"
        )

    if schema_version == VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION:
        marker = (
            _validate_schema13_completion_marker_state(root, dict(completion_state))
            if completion_state is not None
            else validate_schema13_completion_marker(root)
        )
    else:
        marker = (
            dict(completion_state)
            if completion_state is not None
            else validate_completion_marker(root)
        )
    util_validator = getattr(util, util_validator_name)
    full = util_validator(root, label=f"public {schema_label} completed bundle")
    if schema_version == VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION:
        _require_schema12_public_ema_online_equality(root, marker)
    elif schema_version == VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION:
        _require_schema13_public_ema_online_equality(root, marker)
        telemetry = full.get("telemetry")
        actor_policy_for_telemetry = full.get("actor_policy")
        resolved_for_telemetry = full.get("resolved_identity")
        if (
            type(telemetry) is not dict
            or not isinstance(actor_policy_for_telemetry, Mapping)
            or not isinstance(resolved_for_telemetry, Mapping)
        ):
            raise ValueError(
                "schema-13 authoritative bundle lacks exact telemetry evidence"
            )
        stage_for_telemetry = resolved_for_telemetry.get("stage")
        if type(stage_for_telemetry) not in (list, tuple) or len(stage_for_telemetry) != 6:
            raise ValueError("schema-13 telemetry stage evidence is malformed")
        manifest_record = marker["checkpoint_files"][
            "voc_telemetry_manifest.json"
        ]
        direct_telemetry = util.validate_schema13_telemetry_manifest(
            root,
            expected_xpid=stage_for_telemetry[0],
            expected_terminal_policy_version=actor_policy_for_telemetry.get(
                "voc_actor_policy_version"
            ),
            expected_terminal_real_step=telemetry.get("terminal_real_step"),
            expected_actor_state_sha256=actor_policy_for_telemetry.get(
                "voc_actor_policy_state_sha256"
            ),
            expected_publication_history_sha256=actor_policy_for_telemetry.get(
                "voc_actor_policy_publication_history_sha256"
            ),
            expected_stage_total_steps=stage_for_telemetry[2],
            expected_actor_unroll_len=stage_for_telemetry[4],
            expected_terminal_ack_count=1,
            expected_manifest_sha256=manifest_record["sha256"],
            expected_manifest_size=manifest_record["size"],
        )
        if direct_telemetry != telemetry:
            raise ValueError(
                "schema-13 public telemetry disagrees with authoritative manifest"
            )
    model_input_seal = model_seal_validator(full)
    resolved_paths = full["resolved_identity"]["paths"]
    if Path(resolved_paths["ckpdir"]).expanduser().resolve() != root:
        raise ValueError(f"{schema_label} authoritative checkpoint path disagrees")
    if config_payload is not None:
        current_payload = _read_stable_single_link_bytes(
            config_path, label=f"{schema_label} checkpoint config"
        )
        if (
            current_payload != config_payload
            or hashlib.sha256(current_payload).hexdigest()
            != expected_config_sha256
        ):
            raise RuntimeError(
                f"{schema_label} checkpoint config changed during validation"
            )
    post_marker = (
        validate_schema13_completion_marker(root)
        if schema_version == VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION
        else validate_completion_marker(root)
    )
    if marker != post_marker:
        raise RuntimeError(
            f"{schema_label} completion marker changed during validation"
        )

    completion_evidence = full.get("completion_evidence")
    marker_evidence = {
        name: marker.get(name)
        for name in (
            "checkpoint_files",
            "implementation_sources",
            "loaded_extensions",
        )
    }
    if completion_evidence != marker_evidence:
        raise ValueError(
            f"{schema_label} public finish evidence disagrees with the validated bundle"
        )

    logger_completion = util.validate_actor_policy_logger_completion(
        marker.get("voc_actor_policy_logger_completion")
    )
    actor_policy = full.get("actor_policy")
    if not isinstance(actor_policy, Mapping):
        raise ValueError(f"{schema_label} final validation lacks actor-policy evidence")
    if schema_version in (
        VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION,
    ) and set(actor_policy) != ATOMIC_ACTOR_POLICY_EVIDENCE_FIELDS:
        raise ValueError(
            f"{schema_label} actor-policy evidence must preserve the exact "
            "schema-10 lifecycle keyset"
        )
    bundle_summary = actor_policy.get("voc_actor_policy_bundle_summary")
    bundle_summary_fields = {
        "bundle_schema_version",
        "policy_version",
        "terminal",
        "gate_schema",
        "actor_state_dict_sha256",
        "actor_state_dict_key_count",
        "actor_state_dict_keys",
        "actor_state_dict_metadata",
    }
    version = actor_policy.get("voc_actor_policy_version")
    state_digest = actor_policy.get("voc_actor_policy_state_sha256")
    if (
        type(bundle_summary) is not dict
        or set(bundle_summary) != bundle_summary_fields
        or type(bundle_summary.get("gate_schema")) is not int
        or bundle_summary["gate_schema"]
        != schema_version
        or type(bundle_summary.get("bundle_schema_version")) is not int
        or bundle_summary["bundle_schema_version"] != 1
        or bundle_summary.get("terminal") is not True
        or type(version) is not int
        or version < 1
        or type(bundle_summary.get("policy_version")) is not int
        or bundle_summary["policy_version"] != version
        or type(state_digest) is not str
        or re.fullmatch(r"[0-9a-f]{64}", state_digest) is None
        or type(bundle_summary.get("actor_state_dict_sha256")) is not str
        or re.fullmatch(
            r"[0-9a-f]{64}", bundle_summary["actor_state_dict_sha256"]
        )
        is None
        or bundle_summary["actor_state_dict_sha256"] != state_digest
        or type(bundle_summary.get("actor_state_dict_key_count")) is not int
        or bundle_summary["actor_state_dict_key_count"] <= 0
        or not isinstance(bundle_summary.get("actor_state_dict_keys"), list)
        or len(bundle_summary["actor_state_dict_keys"])
        != bundle_summary["actor_state_dict_key_count"]
        or any(
            type(key) is not str or not key
            for key in bundle_summary["actor_state_dict_keys"]
        )
        or len(set(bundle_summary["actor_state_dict_keys"]))
        != bundle_summary["actor_state_dict_key_count"]
        or not isinstance(bundle_summary.get("actor_state_dict_metadata"), list)
        or len(bundle_summary["actor_state_dict_metadata"])
        != bundle_summary["actor_state_dict_key_count"]
    ):
        raise ValueError(f"{schema_label} final actor bundle has the wrong identity")
    for key, metadata in zip(
        bundle_summary["actor_state_dict_keys"],
        bundle_summary["actor_state_dict_metadata"],
    ):
        if (
            type(metadata) is not dict
            or set(metadata) != {"key", "dtype", "shape", "numel"}
            or type(metadata.get("key")) is not str
            or metadata["key"] != key
            or type(metadata.get("dtype")) is not str
            or not metadata["dtype"]
            or type(metadata.get("shape")) is not list
            or any(type(size) is not int or size < 0 for size in metadata["shape"])
            or type(metadata.get("numel")) is not int
            or metadata["numel"] < 0
        ):
            raise ValueError(
                f"{schema_label} final actor bundle metadata is malformed"
            )
        expected_numel = 1
        for size in metadata["shape"]:
            expected_numel *= size
        if metadata["numel"] != expected_numel:
            raise ValueError(
                f"{schema_label} final actor bundle metadata numel disagrees"
            )
    for name, expected in (
        ("voc_actor_policy_terminal", True),
        ("voc_actor_policy_publication_count", version),
        ("voc_actor_policy_expected_ack_count", 1),
        ("voc_actor_policy_terminal_ack_count", 1),
        ("voc_actor_policy_version_mismatch_count", 0),
        ("voc_actor_policy_malformed_bundle_count", 0),
        ("voc_actor_policy_barrier_timeout_count", 0),
    ):
        value = actor_policy.get(name)
        if type(value) is not type(expected) or value != expected:
            raise ValueError(
                f"{schema_label} final actor evidence requires {name}={expected!r}"
            )
    history = actor_policy.get("voc_actor_policy_publication_history")
    event_fields = {
        "predecessor_version",
        "policy_version",
        "publication_count",
        "terminal",
        "ack_ranks",
        "expected_ack_count",
        "state_sha256",
    }
    if not isinstance(history, (list, tuple)) or len(history) != version + 1:
        raise ValueError(
            f"{schema_label} final actor publication history is incomplete"
        )
    for index, event in enumerate(history):
        if (
            not isinstance(event, Mapping)
            or set(event) != event_fields
            or event.get("predecessor_version") != index - 1
            or type(event.get("predecessor_version")) is not int
            or event.get("policy_version") != index
            or type(event.get("policy_version")) is not int
            or event.get("publication_count") != index
            or type(event.get("publication_count")) is not int
            or event.get("terminal") is not (index == version)
            or event.get("ack_ranks") != [0]
            or type(event.get("ack_ranks")) is not list
            or type(event["ack_ranks"][0]) is not int
            or event.get("expected_ack_count") != 1
            or type(event.get("expected_ack_count")) is not int
            or type(event.get("state_sha256")) is not str
            or re.fullmatch(r"[0-9a-f]{64}", event["state_sha256"])
            is None
        ):
            raise ValueError(
                f"{schema_label} final actor publication history is malformed"
            )
    try:
        canonical_history = json.dumps(
            list(history),
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{schema_label} final actor publication history is not JSON-safe"
        ) from error
    history_digest = actor_policy.get(
        "voc_actor_policy_publication_history_sha256"
    )
    if (
        type(history_digest) is not str
        or re.fullmatch(r"[0-9a-f]{64}", history_digest) is None
        or hashlib.sha256(canonical_history).hexdigest() != history_digest
    ):
        raise ValueError(
            f"{schema_label} final actor publication history digest disagrees"
        )
    if history[-1].get("state_sha256") != state_digest:
        raise ValueError(f"{schema_label} final publication state digest disagrees")
    for completion_name, actor_name in (
        ("policy_version", "voc_actor_policy_version"),
        ("state_sha256", "voc_actor_policy_state_sha256"),
        (
            "publication_history_sha256",
            "voc_actor_policy_publication_history_sha256",
        ),
    ):
        if logger_completion[completion_name] != actor_policy.get(actor_name):
            raise ValueError(
                f"{schema_label} logger completion disagrees with terminal actor "
                f"evidence: {completion_name}"
            )
    if logger_completion["checkpoint_files"] != marker_evidence["checkpoint_files"]:
        raise ValueError(
            f"{schema_label} logger completion disagrees with public checkpoint hashes"
        )
    if logger_completion["use_wandb"] != full.get("config_use_wandb"):
        raise ValueError(
            f"{schema_label} logger completion disagrees with immutable use_wandb"
        )

    private_markers = (
        util.VOC_ACTOR_POLICY_LOGGER_FINISH_REQUEST_FILE,
        util.VOC_ACTOR_POLICY_LOGGER_FINISH_ACK_FILE,
    )
    remaining = [name for name in private_markers if os.path.lexists(root / name)]
    if remaining:
        raise ValueError(
            f"{schema_label} completed bundle retains private logger marker(s): "
            + ", ".join(remaining)
        )

    resolved_identity = full.get("resolved_identity")
    stored_surfaces = {
        name: copy.deepcopy(resolved_identity)
        for name in ("config", "actor_checkpoint", "model_checkpoint")
    }
    record = {
        "authoritative_validator": f"thinker.util.{util_validator_name}",
        **copy.deepcopy(full),
        "model_input_seal": model_input_seal,
        "stored_surface_identity": stored_surfaces,
        "logger_completion": logger_completion,
        "private_logger_markers_absent": True,
        "public_finish_verified": True,
    }
    expected_record_fields = {
        "authoritative_validator",
        "actor_policy",
        "resolved_identity",
        "actor_training_state",
        "model_step",
        "model_real_step",
        "model_state_tensor_count",
        "model_optimizer_state",
        "model_scheduler_state",
        "model_scaler_state",
        "config_use_wandb",
        "completion_evidence",
        "model_input_seal",
        "stored_surface_identity",
        "logger_completion",
        "private_logger_markers_absent",
        "public_finish_verified",
    }
    if schema_version == VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION:
        expected_record_fields.add("telemetry")
    if set(record) != expected_record_fields:
        raise ValueError(f"{schema_label} public validation record has the wrong shape")
    try:
        json.dumps(
            record,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{schema_label} public validation record is not strict JSON-safe"
        ) from error
    return record


def validate_schema8_completed_bundle(
    checkpoint_dir: str | Path,
    *,
    completion_state: Optional[Mapping[str, Any]] = None,
    config_payload: Optional[bytes] = None,
    expected_config_sha256: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Validate a completed schema-8 bundle before any downstream eval load."""

    return _validate_schema89_completed_bundle(
        checkpoint_dir,
        completion_state=completion_state,
        config_payload=config_payload,
        expected_config_sha256=expected_config_sha256,
        schema_version=VOC_GATE_POLICY_HALF_SQUARED_Q_SCHEMA_VERSION,
        util_validator_name="validate_schema8_final_bundle",
        model_seal_validator=_require_schema8_public_model_input_seal,
    )


def validate_schema9_completed_bundle(
    checkpoint_dir: str | Path,
    *,
    completion_state: Optional[Mapping[str, Any]] = None,
    config_payload: Optional[bytes] = None,
    expected_config_sha256: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Validate a completed schema-9 bundle before any downstream eval load."""

    return _validate_schema89_completed_bundle(
        checkpoint_dir,
        completion_state=completion_state,
        config_payload=config_payload,
        expected_config_sha256=expected_config_sha256,
        schema_version=VOC_GATE_POLICY_COMMON_MODE_Q_SCHEMA_VERSION,
        util_validator_name="validate_schema9_final_bundle",
        model_seal_validator=_require_schema9_public_model_input_seal,
    )


def validate_schema10_completed_bundle(
    checkpoint_dir: str | Path,
    *,
    completion_state: Optional[Mapping[str, Any]] = None,
    config_payload: Optional[bytes] = None,
    expected_config_sha256: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Validate a completed schema-10 bundle before downstream eval load."""

    return _validate_schema89_completed_bundle(
        checkpoint_dir,
        completion_state=completion_state,
        config_payload=config_payload,
        expected_config_sha256=expected_config_sha256,
        schema_version=VOC_GATE_POLICY_HUBER_COMMON_Q_SCHEMA_VERSION,
        util_validator_name="validate_schema10_final_bundle",
        model_seal_validator=_require_schema10_public_model_input_seal,
    )


def validate_schema11_completed_bundle(
    checkpoint_dir: str | Path,
    *,
    completion_state: Optional[Mapping[str, Any]] = None,
    config_payload: Optional[bytes] = None,
    expected_config_sha256: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Validate a completed schema-11 bundle before downstream eval load."""

    return _validate_schema89_completed_bundle(
        checkpoint_dir,
        completion_state=completion_state,
        config_payload=config_payload,
        expected_config_sha256=expected_config_sha256,
        schema_version=VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION,
        util_validator_name="validate_schema11_final_bundle",
        model_seal_validator=_require_schema11_public_model_input_seal,
    )


def validate_schema12_completed_bundle(
    checkpoint_dir: str | Path,
    *,
    completion_state: Optional[Mapping[str, Any]] = None,
    config_payload: Optional[bytes] = None,
    expected_config_sha256: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Validate a completed schema-12 bundle before downstream eval load."""

    return _validate_schema89_completed_bundle(
        checkpoint_dir,
        completion_state=completion_state,
        config_payload=config_payload,
        expected_config_sha256=expected_config_sha256,
        schema_version=VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION,
        util_validator_name="validate_schema12_final_bundle",
        model_seal_validator=_require_schema12_public_model_input_seal,
    )


def validate_schema13_completed_bundle(
    checkpoint_dir: str | Path,
    *,
    completion_state: Optional[Mapping[str, Any]] = None,
    config_payload: Optional[bytes] = None,
    expected_config_sha256: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Validate a completed schema-13 telemetry bundle before downstream use."""

    return _validate_schema89_completed_bundle(
        checkpoint_dir,
        completion_state=completion_state,
        config_payload=config_payload,
        expected_config_sha256=expected_config_sha256,
        schema_version=VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION,
        util_validator_name="validate_schema13_final_bundle",
        model_seal_validator=_require_schema13_public_model_input_seal,
    )


def dispatch_schema8_completed_bundle(
    checkpoint_dir: str | Path,
    *,
    completion_state: Optional[Mapping[str, Any]] = None,
    config_payload: Optional[bytes] = None,
    expected_config_sha256: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Route only a schema-8/v15 claim to the strict dedicated validator."""

    root = Path(checkpoint_dir).expanduser().resolve()
    if config_payload is None:
        config_path = root / "config_c.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"checkpoint is missing config_c.yaml: {config_path}"
            )
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    else:
        if type(config_payload) is not bytes:
            raise TypeError("schema-8 dispatch config payload must be exact bytes")
        if (
            type(expected_config_sha256) is not str
            or re.fullmatch(r"[0-9a-f]{64}", expected_config_sha256) is None
            or hashlib.sha256(config_payload).hexdigest()
            != expected_config_sha256
        ):
            raise ValueError("schema-8 dispatch config digest disagrees")
        try:
            config = yaml.safe_load(config_payload.decode("utf-8"))
        except (UnicodeDecodeError, yaml.YAMLError) as error:
            raise ValueError("schema-8 dispatch config is not UTF-8 YAML") from error
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint config_c.yaml must contain a mapping")
    raw_schema = config.get("voc_gate_policy_schema_version")
    raw_xpid = config.get("xpid")
    claimed = (
        raw_schema == VOC_GATE_POLICY_HALF_SQUARED_Q_SCHEMA_VERSION
        or (
            type(raw_xpid) is str
            and raw_xpid
            in {stage[0] for stage in VOC_GATE_POLICY_SCHEMA8_STAGE_PROFILES}
        )
    )
    if not claimed:
        return None
    return validate_schema8_completed_bundle(
        root, completion_state=completion_state
    )


def dispatch_schema9_completed_bundle(
    checkpoint_dir: str | Path,
    *,
    completion_state: Optional[Mapping[str, Any]] = None,
    config_payload: Optional[bytes] = None,
    expected_config_sha256: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Route only a schema-9/v16 claim to the strict dedicated validator."""

    root = Path(checkpoint_dir).expanduser().resolve()
    if config_payload is None:
        config_path = root / "config_c.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"checkpoint is missing config_c.yaml: {config_path}"
            )
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    else:
        if type(config_payload) is not bytes:
            raise TypeError("schema-9 dispatch config payload must be exact bytes")
        if (
            type(expected_config_sha256) is not str
            or re.fullmatch(r"[0-9a-f]{64}", expected_config_sha256) is None
            or hashlib.sha256(config_payload).hexdigest()
            != expected_config_sha256
        ):
            raise ValueError("schema-9 dispatch config digest disagrees")
        try:
            config = yaml.safe_load(config_payload.decode("utf-8"))
        except (UnicodeDecodeError, yaml.YAMLError) as error:
            raise ValueError("schema-9 dispatch config is not UTF-8 YAML") from error
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint config_c.yaml must contain a mapping")
    raw_schema = config.get("voc_gate_policy_schema_version")
    raw_xpid = config.get("xpid")
    claimed = (
        raw_schema == VOC_GATE_POLICY_COMMON_MODE_Q_SCHEMA_VERSION
        or (
            type(raw_xpid) is str
            and raw_xpid
            in {stage[0] for stage in VOC_GATE_POLICY_SCHEMA9_STAGE_PROFILES}
        )
    )
    if not claimed:
        return None
    return validate_schema9_completed_bundle(
        root,
        completion_state=completion_state,
        config_payload=config_payload,
        expected_config_sha256=expected_config_sha256,
    )


def dispatch_schema10_completed_bundle(
    checkpoint_dir: str | Path,
    *,
    completion_state: Optional[Mapping[str, Any]] = None,
    config_payload: Optional[bytes] = None,
    expected_config_sha256: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Route only a schema-10/v17 claim to the strict dedicated validator."""

    root = Path(checkpoint_dir).expanduser().resolve()
    if config_payload is None:
        config_path = root / "config_c.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"checkpoint is missing config_c.yaml: {config_path}"
            )
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    else:
        if type(config_payload) is not bytes:
            raise TypeError("schema-10 dispatch config payload must be exact bytes")
        if (
            type(expected_config_sha256) is not str
            or re.fullmatch(r"[0-9a-f]{64}", expected_config_sha256) is None
            or hashlib.sha256(config_payload).hexdigest()
            != expected_config_sha256
        ):
            raise ValueError("schema-10 dispatch config digest disagrees")
        try:
            config = yaml.safe_load(config_payload.decode("utf-8"))
        except (UnicodeDecodeError, yaml.YAMLError) as error:
            raise ValueError("schema-10 dispatch config is not UTF-8 YAML") from error
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint config_c.yaml must contain a mapping")
    raw_schema = config.get("voc_gate_policy_schema_version")
    raw_xpid = config.get("xpid")
    claimed = (
        raw_schema == VOC_GATE_POLICY_HUBER_COMMON_Q_SCHEMA_VERSION
        or (
            type(raw_xpid) is str
            and raw_xpid
            in {stage[0] for stage in VOC_GATE_POLICY_SCHEMA10_STAGE_PROFILES}
        )
    )
    if not claimed:
        return None
    return validate_schema10_completed_bundle(
        root,
        completion_state=completion_state,
        config_payload=config_payload,
        expected_config_sha256=expected_config_sha256,
    )


def dispatch_schema11_completed_bundle(
    checkpoint_dir: str | Path,
    *,
    completion_state: Optional[Mapping[str, Any]] = None,
    config_payload: Optional[bytes] = None,
    expected_config_sha256: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Route only a schema-11/v18 claim to the strict dedicated validator."""

    root = Path(checkpoint_dir).expanduser().resolve()
    if config_payload is None:
        config_path = root / "config_c.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"checkpoint is missing config_c.yaml: {config_path}"
            )
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    else:
        if type(config_payload) is not bytes:
            raise TypeError("schema-11 dispatch config payload must be exact bytes")
        if (
            type(expected_config_sha256) is not str
            or re.fullmatch(r"[0-9a-f]{64}", expected_config_sha256) is None
            or hashlib.sha256(config_payload).hexdigest()
            != expected_config_sha256
        ):
            raise ValueError("schema-11 dispatch config digest disagrees")
        try:
            config = yaml.safe_load(config_payload.decode("utf-8"))
        except (UnicodeDecodeError, yaml.YAMLError) as error:
            raise ValueError("schema-11 dispatch config is not UTF-8 YAML") from error
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint config_c.yaml must contain a mapping")
    raw_schema = config.get("voc_gate_policy_schema_version")
    raw_xpid = config.get("xpid")
    claimed = (
        raw_schema == VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION
        or (
            type(raw_xpid) is str
            and raw_xpid.startswith(
                "enduro-voc-v18-orthocd-adam-eps25-"
            )
        )
    )
    if not claimed:
        return None
    return validate_schema11_completed_bundle(
        root,
        completion_state=completion_state,
        config_payload=config_payload,
        expected_config_sha256=expected_config_sha256,
    )


def _schema12_xpid_claims_intent(value: Any) -> bool:
    """Classify V19 lexical intent without normalizing it into validity."""

    try:
        lexical_value = os.fspath(value) if isinstance(value, os.PathLike) else value
        if isinstance(
            lexical_value,
            (bytes, bytearray, memoryview, np.bytes_),
        ):
            lexical_value = bytes(lexical_value).decode("utf-8")
        else:
            lexical_value = str(lexical_value)
    except (TypeError, UnicodeError) as error:
        raise ValueError(
            "schema-12 xpid intent could not be classified before downstream I/O"
        ) from error
    return lexical_value.strip().startswith(
        "enduro-voc-v19-tau1-orthocd-adam-eps25-"
    )


def dispatch_schema12_completed_bundle(
    checkpoint_dir: str | Path,
    *,
    completion_state: Optional[Mapping[str, Any]] = None,
    config_payload: Optional[bytes] = None,
    expected_config_sha256: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Route every schema-12/V19 claim to the strict dedicated validator."""

    root = Path(checkpoint_dir).expanduser().resolve()
    if config_payload is None:
        config_path = root / "config_c.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"checkpoint is missing config_c.yaml: {config_path}"
            )
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    else:
        if type(config_payload) is not bytes:
            raise TypeError("schema-12 dispatch config payload must be exact bytes")
        if (
            type(expected_config_sha256) is not str
            or re.fullmatch(r"[0-9a-f]{64}", expected_config_sha256) is None
            or hashlib.sha256(config_payload).hexdigest()
            != expected_config_sha256
        ):
            raise ValueError("schema-12 dispatch config digest disagrees")
        try:
            config = yaml.safe_load(config_payload.decode("utf-8"))
        except (UnicodeDecodeError, yaml.YAMLError) as error:
            raise ValueError("schema-12 dispatch config is not UTF-8 YAML") from error
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint config_c.yaml must contain a mapping")
    raw_schema = config.get("voc_gate_policy_schema_version")
    raw_xpid = config.get("xpid")
    claimed = (
        raw_schema == VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION
        or _schema12_xpid_claims_intent(raw_xpid)
    )
    if not claimed:
        return None
    return validate_schema12_completed_bundle(
        root,
        completion_state=completion_state,
        config_payload=config_payload,
        expected_config_sha256=expected_config_sha256,
    )


def _schema13_xpid_claims_intent(value: Any) -> bool:
    """Classify V20 lexical intent without normalizing it into validity."""

    try:
        lexical_value = os.fspath(value) if isinstance(value, os.PathLike) else value
        if isinstance(
            lexical_value,
            (bytes, bytearray, memoryview, np.bytes_),
        ):
            lexical_value = bytes(lexical_value).decode("utf-8")
        else:
            lexical_value = str(lexical_value)
    except (TypeError, UnicodeError) as error:
        raise ValueError(
            "schema-13 xpid intent could not be classified before downstream I/O"
        ) from error
    return lexical_value.strip().startswith(
        "enduro-voc-v20-telemetry-tau1-orthocd-adam-eps25-"
    )


def dispatch_schema13_completed_bundle(
    checkpoint_dir: str | Path,
    *,
    completion_state: Optional[Mapping[str, Any]] = None,
    config_payload: Optional[bytes] = None,
    expected_config_sha256: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Route every schema-13/V20 claim to the strict dedicated validator."""

    root = Path(checkpoint_dir).expanduser().resolve()
    if config_payload is None:
        config_payload = _read_stable_single_link_bytes(
            root / "config_c.yaml", label="schema-13 dispatch config"
        )
        expected_config_sha256 = hashlib.sha256(config_payload).hexdigest()
    elif type(config_payload) is not bytes:
        raise TypeError("schema-13 dispatch config payload must be exact bytes")
    if (
        type(expected_config_sha256) is not str
        or re.fullmatch(r"[0-9a-f]{64}", expected_config_sha256) is None
        or hashlib.sha256(config_payload).hexdigest() != expected_config_sha256
    ):
        raise ValueError("schema-13 dispatch config digest disagrees")
    try:
        config = yaml.safe_load(config_payload.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError("schema-13 dispatch config is not UTF-8 YAML") from error
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint config_c.yaml must contain a mapping")
    raw_schema = config.get("voc_gate_policy_schema_version")
    raw_xpid = config.get("xpid")
    claimed = (
        raw_schema == VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION
        or _schema13_xpid_claims_intent(raw_xpid)
    )
    if not claimed:
        return None
    return validate_schema13_completed_bundle(
        root,
        completion_state=completion_state,
        config_payload=config_payload,
        expected_config_sha256=expected_config_sha256,
    )


def evaluation_runtime_flags(
    training_flags: Any,
) -> Tuple[Any, Optional[Dict[str, Any]]]:
    """Create a private non-distributed evaluation copy after attestation."""

    runtime_flags = copy.deepcopy(training_flags)
    runtime_flags.parallel = False
    runtime_flags.parallel_actor = False
    runtime_flags.use_wandb = False
    raw_schema = getattr(training_flags, "voc_gate_policy_schema_version", None)
    if raw_schema not in (
        VOC_GATE_POLICY_VERSION_BARRIER_SCHEMA_VERSION,
        VOC_GATE_POLICY_MODEL_INPUT_SEAL_SCHEMA_VERSION,
        VOC_GATE_POLICY_HALF_SQUARED_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_COMMON_MODE_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_HUBER_COMMON_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION,
    ):
        return runtime_flags, None
    runtime_flags.train_actor = False
    runtime_flags.voc_actor_policy_barrier_runtime = False
    sealed_schema = raw_schema in (
        VOC_GATE_POLICY_MODEL_INPUT_SEAL_SCHEMA_VERSION,
        VOC_GATE_POLICY_HALF_SQUARED_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_COMMON_MODE_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_HUBER_COMMON_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION,
    )
    if sealed_schema:
        if getattr(training_flags, "train_model", None) is not True:
            raise ValueError(
                f"schema-{raw_schema} immutable training flags require "
                "train_model=true"
            )
        if (
            type(
                getattr(
                    training_flags,
                    "voc_model_input_seal_schema_version",
                    None,
                )
            )
            is not int
            or training_flags.voc_model_input_seal_schema_version != 1
        ):
            raise ValueError(
                f"schema-{raw_schema} immutable training flags require seal schema 1"
            )
        runtime_flags.train_model = False
    record = {
        "immutable_training": {
            "train_actor": getattr(training_flags, "train_actor", None),
            "parallel_actor": getattr(training_flags, "parallel_actor", None),
            "voc_actor_policy_barrier_runtime": getattr(
                training_flags, "voc_actor_policy_barrier_runtime", None
            ),
            "voc_train_epsilon": getattr(training_flags, "voc_train_epsilon", None),
            "voc_gate_execution_epsilon": getattr(
                training_flags, "voc_gate_execution_epsilon", None
            ),
        },
        "evaluation_copy": {
            "train_actor": False,
            "parallel_actor": False,
            "voc_actor_policy_barrier_runtime": False,
            "effective_soft_gate_epsilon": 0.0,
            "effective_execution_gate_epsilon": 0.0,
            "use_wandb": False,
        },
        "persisted_surfaces_unchanged": True,
    }
    if sealed_schema:
        record["immutable_training"].update(
            {
                "train_model": True,
                "voc_model_input_seal_schema_version": 1,
            }
        )
        record["evaluation_copy"].update(
            {
                "train_model": False,
                "effective_model_input_seal_coordination": False,
            }
        )
    json.dumps(record, sort_keys=True, allow_nan=False)
    return runtime_flags, record


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
    protocol = {name: getattr(flags, name, None) for name in names}
    gate_schema = getattr(
        flags,
        "voc_gate_policy_schema_version",
        0,
    )
    sealed_schemas = (
        VOC_GATE_POLICY_MODEL_INPUT_SEAL_SCHEMA_VERSION,
        VOC_GATE_POLICY_HALF_SQUARED_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_COMMON_MODE_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_HUBER_COMMON_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION,
    )
    if not (type(gate_schema) is int and gate_schema in sealed_schemas):
        protocol.pop("voc_model_input_seal_schema_version", None)
    for name, default in VOC_PROTOCOL_DEFAULTS.items():
        # The seal identity was introduced with schema 7.  Keep the historical
        # schema <= 6 protocol/summary/manifest record shape byte-for-byte
        # compatible even though VOC_PROTOCOL_DEFAULTS must expose the runtime
        # legacy default for parser/contract parity.
        if (
            name == "voc_model_input_seal_schema_version"
            and not (
                type(gate_schema) is int and gate_schema in sealed_schemas
            )
        ):
            continue
        protocol[name] = getattr(flags, name, default)
    return protocol


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
    gate_schema = (
        getattr(flags, "voc_gate_policy_schema_version", None)
        if flags is not None
        else None
    )
    schema13 = type(gate_schema) is int and gate_schema == (
        VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION
    )
    if schema13:
        checkpoints = [
            checkpoint_dir / name for name in SCHEMA13_BOUND_RUN_FILES
        ]
        completion_state = validate_schema13_completion_marker(checkpoint_dir)
        current_checkpoint_hashes = _schema13_checkpoint_hashes(
            checkpoint_dir,
            completion_state=completion_state,
        )
    else:
        checkpoints = [checkpoint_dir / name for name in REQUIRED_CHECKPOINT_FILES]
        missing = [str(path) for path in checkpoints if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "checkpoint is incomplete: " + ", ".join(missing)
            )
        completion_state = validate_completion_marker(checkpoint_dir)
        current_checkpoint_hashes = {
            path.name: sha256_file(path) for path in checkpoints
        }
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


def _require_environment_return_only_voc(value: Any, *, label: str) -> None:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.number))
        or not np.isfinite(value)
        or float(value) != 0.0
    ):
        raise ValueError(
            f"{label} requires entropy_r_cost=0 for environment-return-only "
            f"VoC targets; got {value!r}"
        )


def _require_voc_ema_gate_protocol(
    enabled: Any, tau: Any, *, label: str
) -> None:
    if not isinstance(enabled, (bool, np.bool_)) or not bool(enabled):
        raise ValueError(f"{label} requires voc_ema_gate_target=true")
    if (
        isinstance(tau, (bool, np.bool_))
        or not isinstance(tau, (int, float, np.number))
        or not np.isfinite(tau)
        or not 0.0 < float(tau) <= 1.0
    ):
        raise ValueError(
            f"{label} requires 0 < voc_gate_target_tau <= 1; got {tau!r}"
        )


def _require_voc_gate_policy_protocol(
    values: Mapping[str, Any], *, label: str
) -> Tuple[
    bool, bool, float, bool, float, float, float, bool, float, bool, bool
]:
    for name in (
        "voc_dedicated_gate",
        "voc_soft_q_bce_gate",
    ):
        value = values.get(name)
        if not isinstance(value, (bool, np.bool_)) or not bool(value):
            raise ValueError(f"{label} requires {name}=true")
    confidence_weighted = values.get("voc_gate_confidence_weighted")
    if not isinstance(confidence_weighted, (bool, np.bool_)):
        raise ValueError(
            f"{label} requires voc_gate_confidence_weighted to be boolean"
        )
    param_align = values.get("voc_gate_param_align")
    if not isinstance(param_align, (bool, np.bool_)):
        raise ValueError(
            f"{label} requires voc_gate_param_align to be boolean"
        )
    param_align_coef = values.get("voc_gate_param_align_coef")
    if (
        isinstance(param_align_coef, (bool, np.bool_))
        or not isinstance(param_align_coef, (int, float, np.number))
        or not np.isfinite(param_align_coef)
        or float(param_align_coef) != 1.0
    ):
        raise ValueError(
            f"{label} requires voc_gate_param_align_coef=1.0 exactly; "
            f"got {param_align_coef!r}"
        )
    exact_projection = values.get("voc_gate_exact_projection")
    if not isinstance(exact_projection, (bool, np.bool_)):
        raise ValueError(
            f"{label} requires voc_gate_exact_projection to be boolean"
        )
    if bool(exact_projection) and bool(param_align):
        raise ValueError(
            f"{label} requires voc_gate_exact_projection and "
            "voc_gate_param_align to be mutually exclusive"
        )
    epsilon_greedy_execution = values.get(
        "voc_gate_epsilon_greedy_execution"
    )
    if not isinstance(epsilon_greedy_execution, (bool, np.bool_)):
        raise ValueError(
            f"{label} requires voc_gate_epsilon_greedy_execution to be "
            "boolean"
        )
    if bool(epsilon_greedy_execution) and not bool(exact_projection):
        raise ValueError(
            f"{label} requires voc_gate_epsilon_greedy_execution only with "
            "voc_gate_exact_projection=true"
        )
    adam_beta1 = values.get("voc_gate_adam_beta1")
    if (
        isinstance(adam_beta1, (bool, np.bool_))
        or not isinstance(adam_beta1, (int, float, np.number))
        or not np.isfinite(adam_beta1)
        or not 0.0 <= float(adam_beta1) < 1.0
    ):
        raise ValueError(
            f"{label} requires 0 <= voc_gate_adam_beta1 < 1; "
            f"got {adam_beta1!r}"
        )
    normalized = []
    for name in (
        "voc_gate_q_temperature",
        "voc_gate_learning_rate",
        "voc_gate_grad_norm_clipping",
    ):
        value = values.get(name)
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.number))
            or not np.isfinite(value)
            or float(value) <= 0.0
        ):
            raise ValueError(f"{label} requires {name}>0; got {value!r}")
        normalized.append(float(value))
    return (
        True,
        True,
        normalized[0],
        bool(confidence_weighted),
        float(adam_beta1),
        normalized[1],
        normalized[2],
        bool(param_align),
        1.0,
        bool(exact_projection),
        bool(epsilon_greedy_execution),
    )


def _validate_voc_gate_policy_schema(
    checkpoint: Mapping[str, Any],
    embedded: Mapping[str, Any],
    *,
    label: str,
) -> Dict[str, Any]:
    """Resolve the actor schema or infer the schema-less ModelNet generation.

    Actor checkpoints publish the gate-policy schema at top level.  ModelNet
    checkpoints have historically stored only the same embedded training flags,
    so their generation is inferred without weakening the field-level contract.
    """

    schema = checkpoint.get("voc_gate_policy_schema_version")
    embedded_schema = embedded.get("voc_gate_policy_schema_version")
    atomic_schemas = (
        VOC_GATE_POLICY_VERSION_BARRIER_SCHEMA_VERSION,
        VOC_GATE_POLICY_MODEL_INPUT_SEAL_SCHEMA_VERSION,
        VOC_GATE_POLICY_HALF_SQUARED_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_COMMON_MODE_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_HUBER_COMMON_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION,
    )
    if schema in atomic_schemas or embedded_schema in atomic_schemas:
        from thinker import util

        if schema is not None and schema != embedded_schema:
            raise ValueError(
                f"{label} atomic top-level and embedded schema disagree"
            )
        resolved_atomic_schema = (
            embedded_schema if schema is None else schema
        )
        if type(resolved_atomic_schema) is not int:
            raise ValueError(
                f"{label} atomic gate-policy schema must be a Python integer"
            )
        atomic_checkpoint = dict(checkpoint)
        atomic_checkpoint["voc_gate_policy_schema_version"] = (
            resolved_atomic_schema
        )
        atomic_checkpoint["flags"] = embedded
        return util.validate_voc_gate_policy_schema(
            atomic_checkpoint, label=f"{label} active VoC checkpoint"
        )
    align_present = "voc_gate_param_align" in embedded
    coefficient_present = "voc_gate_param_align_coef" in embedded
    projection_present = "voc_gate_exact_projection" in embedded
    execution_present = "voc_gate_epsilon_greedy_execution" in embedded
    beta1_present = "voc_gate_adam_beta1" in embedded
    if schema is None:
        # ModelNet has no top-level gate schema.  The full actor-state validator
        # separately rejects a missing one after this embedded/config audit.
        schema = (
            VOC_GATE_POLICY_LEGACY_SCHEMA_VERSION
            if not beta1_present
            else (
                VOC_GATE_POLICY_EPSILON_GREEDY_EXECUTION_SCHEMA_VERSION
                if (
                    execution_present
                    and isinstance(
                        embedded["voc_gate_epsilon_greedy_execution"],
                        (bool, np.bool_),
                    )
                    and bool(
                        embedded["voc_gate_epsilon_greedy_execution"]
                    )
                )
                else (
                    VOC_GATE_POLICY_EXACT_PROJECTION_SCHEMA_VERSION
                    if (
                        projection_present
                        and isinstance(
                            embedded["voc_gate_exact_projection"],
                            (bool, np.bool_),
                        )
                        and bool(embedded["voc_gate_exact_projection"])
                    )
                    else (
                        VOC_GATE_POLICY_SCHEMA_VERSION
                        if align_present or coefficient_present
                        else VOC_GATE_POLICY_INTERMEDIATE_SCHEMA_VERSION
                    )
                )
            )
        )
    if (
        isinstance(schema, (bool, np.bool_))
        or not isinstance(schema, (int, np.integer))
        or int(schema)
        not in (
            VOC_GATE_POLICY_LEGACY_SCHEMA_VERSION,
            VOC_GATE_POLICY_INTERMEDIATE_SCHEMA_VERSION,
            VOC_GATE_POLICY_SCHEMA_VERSION,
            VOC_GATE_POLICY_EXACT_PROJECTION_SCHEMA_VERSION,
            VOC_GATE_POLICY_EPSILON_GREEDY_EXECUTION_SCHEMA_VERSION,
        )
    ):
        raise ValueError(
            f"{label} active VoC checkpoint has unsupported "
            f"voc_gate_policy_schema_version={schema!r}"
        )
    schema = int(schema)

    beta1_legacy_defaulted = not beta1_present
    if beta1_legacy_defaulted:
        if schema != VOC_GATE_POLICY_LEGACY_SCHEMA_VERSION:
            raise ValueError(
                f"{label} active VoC checkpoint schema {schema} lacks "
                "embedded voc_gate_adam_beta1"
            )
        beta1 = VOC_GATE_ADAM_BETA1_LEGACY_DEFAULT
    else:
        beta1 = embedded["voc_gate_adam_beta1"]
        if (
            isinstance(beta1, (bool, np.bool_))
            or not isinstance(beta1, (int, float, np.number))
            or not np.isfinite(beta1)
            or not 0.0 <= float(beta1) < 1.0
        ):
            raise ValueError(
                f"{label} active VoC checkpoint requires "
                f"0 <= voc_gate_adam_beta1 < 1; got {beta1!r}"
            )
        beta1 = float(beta1)
        if (
            schema == VOC_GATE_POLICY_LEGACY_SCHEMA_VERSION
            and beta1 != VOC_GATE_ADAM_BETA1_LEGACY_DEFAULT
        ):
            raise ValueError(
                f"{label} active VoC checkpoint schema 1 requires legacy "
                f"voc_gate_adam_beta1={VOC_GATE_ADAM_BETA1_LEGACY_DEFAULT}; "
                f"got {beta1!r}"
            )

    if schema >= VOC_GATE_POLICY_SCHEMA_VERSION:
        for name, present in (
            ("voc_gate_param_align", align_present),
            ("voc_gate_param_align_coef", coefficient_present),
        ):
            if not present:
                raise ValueError(
                    f"{label} active VoC checkpoint schema {schema} lacks "
                    f"embedded {name}"
                )
    elif align_present != coefficient_present:
        missing = (
            "voc_gate_param_align_coef"
            if align_present
            else "voc_gate_param_align"
        )
        raise ValueError(
            f"{label} active VoC checkpoint schema {schema} has partial "
            f"legacy alignment metadata; lacks embedded {missing}"
        )

    align_legacy_defaulted = not align_present
    if align_legacy_defaulted:
        param_align = False
        param_align_coef = 1.0
    else:
        param_align = embedded["voc_gate_param_align"]
        param_align_coef = embedded["voc_gate_param_align_coef"]
        if not isinstance(param_align, (bool, np.bool_)):
            raise ValueError(
                f"{label} active VoC checkpoint requires "
                f"voc_gate_param_align to be boolean; got {param_align!r}"
            )
        param_align = bool(param_align)
        if (
            isinstance(param_align_coef, (bool, np.bool_))
            or not isinstance(param_align_coef, (int, float, np.number))
            or not np.isfinite(param_align_coef)
            or float(param_align_coef) != 1.0
        ):
            raise ValueError(
                f"{label} active VoC checkpoint requires "
                "voc_gate_param_align_coef=1.0 exactly; "
                f"got {param_align_coef!r}"
            )
        param_align_coef = 1.0
        if schema < VOC_GATE_POLICY_SCHEMA_VERSION and param_align:
            raise ValueError(
                f"{label} active VoC checkpoint schema {schema} predates "
                "parameter alignment and requires voc_gate_param_align=false"
            )

    if (
        schema >= VOC_GATE_POLICY_EXACT_PROJECTION_SCHEMA_VERSION
        and not projection_present
    ):
        raise ValueError(
            f"{label} active VoC checkpoint schema {schema} lacks embedded "
            "voc_gate_exact_projection"
        )
    projection_legacy_defaulted = not projection_present
    if projection_legacy_defaulted:
        exact_projection = False
    else:
        exact_projection = embedded["voc_gate_exact_projection"]
        if not isinstance(exact_projection, (bool, np.bool_)):
            raise ValueError(
                f"{label} active VoC checkpoint requires "
                "voc_gate_exact_projection to be boolean; "
                f"got {exact_projection!r}"
            )
        exact_projection = bool(exact_projection)
        if (
            schema < VOC_GATE_POLICY_EXACT_PROJECTION_SCHEMA_VERSION
            and exact_projection
        ):
            raise ValueError(
                f"{label} active VoC checkpoint schema {schema} predates "
                "exact projection and requires "
                "voc_gate_exact_projection=false"
            )
    if (
        schema >= VOC_GATE_POLICY_EXACT_PROJECTION_SCHEMA_VERSION
        and not exact_projection
    ):
        raise ValueError(
            f"{label} active VoC checkpoint schema {schema} requires "
            "voc_gate_exact_projection=true"
        )
    if exact_projection and param_align:
        raise ValueError(
            f"{label} active VoC checkpoint requires "
            "voc_gate_exact_projection and voc_gate_param_align to be "
            "mutually exclusive"
        )
    if (
        schema == VOC_GATE_POLICY_EPSILON_GREEDY_EXECUTION_SCHEMA_VERSION
        and not execution_present
    ):
        raise ValueError(
            f"{label} active VoC checkpoint schema {schema} lacks embedded "
            "voc_gate_epsilon_greedy_execution"
        )
    execution_legacy_defaulted = not execution_present
    if execution_legacy_defaulted:
        epsilon_greedy_execution = False
    else:
        epsilon_greedy_execution = embedded[
            "voc_gate_epsilon_greedy_execution"
        ]
        if not isinstance(epsilon_greedy_execution, (bool, np.bool_)):
            raise ValueError(
                f"{label} active VoC checkpoint requires "
                "voc_gate_epsilon_greedy_execution to be boolean; "
                f"got {epsilon_greedy_execution!r}"
            )
        epsilon_greedy_execution = bool(epsilon_greedy_execution)
        if (
            schema
            < VOC_GATE_POLICY_EPSILON_GREEDY_EXECUTION_SCHEMA_VERSION
            and epsilon_greedy_execution
        ):
            raise ValueError(
                f"{label} active VoC checkpoint schema {schema} predates "
                "epsilon-greedy execution and requires "
                "voc_gate_epsilon_greedy_execution=false"
            )
    if (
        schema == VOC_GATE_POLICY_EPSILON_GREEDY_EXECUTION_SCHEMA_VERSION
        and not epsilon_greedy_execution
    ):
        raise ValueError(
            f"{label} active VoC checkpoint schema {schema} requires "
            "voc_gate_epsilon_greedy_execution=true"
        )
    if epsilon_greedy_execution and not exact_projection:
        raise ValueError(
            f"{label} active VoC checkpoint epsilon-greedy execution "
            "requires exact projection"
        )
    return {
        "voc_gate_policy_schema_version": schema,
        "voc_gate_adam_beta1": float(beta1),
        "voc_gate_adam_beta1_legacy_defaulted": beta1_legacy_defaulted,
        "voc_gate_param_align": bool(param_align),
        "voc_gate_param_align_coef": float(param_align_coef),
        "voc_gate_param_align_legacy_defaulted": align_legacy_defaulted,
        "voc_gate_exact_projection": bool(exact_projection),
        "voc_gate_exact_projection_legacy_defaulted": (
            projection_legacy_defaulted
        ),
        "voc_gate_epsilon_greedy_execution": bool(
            epsilon_greedy_execution
        ),
        "voc_gate_epsilon_greedy_execution_legacy_defaulted": (
            execution_legacy_defaulted
        ),
    }


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

    embedded_voc_mode = embedded.get(
        "dynamic_voc_mode", VOC_PROTOCOL_DEFAULTS["dynamic_voc_mode"]
    )
    gate_schema_state = None
    if embedded_voc_mode != "off":
        gate_schema_state = _validate_voc_gate_policy_schema(
            checkpoint, embedded, label=label
        )
        if "entropy_r_cost" not in embedded:
            raise ValueError(
                f"{label} active VoC checkpoint lacks embedded entropy_r_cost"
            )
        _require_environment_return_only_voc(
            embedded["entropy_r_cost"], label=f"{label} active VoC checkpoint"
        )
        _require_environment_return_only_voc(
            getattr(
                config_flags,
                "entropy_r_cost",
                VOC_PROTOCOL_DEFAULTS["entropy_r_cost"],
            ),
            label=f"{label} active VoC config_c.yaml",
        )
        for name in ("voc_ema_gate_target", "voc_gate_target_tau"):
            if name not in embedded:
                raise ValueError(
                    f"{label} active VoC checkpoint lacks embedded {name}"
                )
        _require_voc_ema_gate_protocol(
            embedded["voc_ema_gate_target"],
            embedded["voc_gate_target_tau"],
            label=f"{label} active VoC checkpoint",
        )
        _require_voc_ema_gate_protocol(
            getattr(
                config_flags,
                "voc_ema_gate_target",
                VOC_PROTOCOL_DEFAULTS["voc_ema_gate_target"],
            ),
            getattr(
                config_flags,
                "voc_gate_target_tau",
                VOC_PROTOCOL_DEFAULTS["voc_gate_target_tau"],
            ),
            label=f"{label} active VoC config_c.yaml",
        )
        gate_protocol_names = (
            "voc_dedicated_gate",
            "voc_soft_q_bce_gate",
            "voc_gate_q_temperature",
            "voc_gate_confidence_weighted",
            "voc_gate_adam_beta1",
            "voc_gate_learning_rate",
            "voc_gate_grad_norm_clipping",
            "voc_gate_param_align",
            "voc_gate_param_align_coef",
            "voc_gate_exact_projection",
            "voc_gate_epsilon_greedy_execution",
            "voc_gate_execution_epsilon",
            "voc_actor_policy_version_barrier",
            "voc_actor_policy_bundle_schema_version",
            "voc_actor_policy_barrier_timeout_s",
            "voc_actor_policy_ray_max_restarts",
            "voc_actor_policy_ray_max_task_retries",
            "actor_amp_init_scale",
            "voc_model_input_seal_schema_version",
        )
        for name in gate_protocol_names:
            if name not in embedded and name not in (
                "voc_gate_adam_beta1",
                "voc_gate_param_align",
                "voc_gate_param_align_coef",
                "voc_gate_exact_projection",
                "voc_gate_epsilon_greedy_execution",
                "voc_gate_execution_epsilon",
                "voc_actor_policy_version_barrier",
                "voc_actor_policy_bundle_schema_version",
                "voc_actor_policy_barrier_timeout_s",
                "voc_actor_policy_ray_max_restarts",
                "voc_actor_policy_ray_max_task_retries",
                "actor_amp_init_scale",
                "voc_model_input_seal_schema_version",
            ):
                raise ValueError(
                    f"{label} active VoC checkpoint lacks embedded {name}"
                )
        gate_protocol = dict(embedded)
        for name in (
            "voc_gate_adam_beta1",
            "voc_gate_param_align",
            "voc_gate_param_align_coef",
            "voc_gate_exact_projection",
            "voc_gate_epsilon_greedy_execution",
            "voc_gate_execution_epsilon",
            "voc_actor_policy_version_barrier",
            "voc_actor_policy_bundle_schema_version",
            "voc_actor_policy_barrier_timeout_s",
            "voc_actor_policy_ray_max_restarts",
            "voc_actor_policy_ray_max_task_retries",
            "actor_amp_init_scale",
            "voc_model_input_seal_schema_version",
        ):
            gate_protocol[name] = gate_schema_state.get(
                name, VOC_PROTOCOL_DEFAULTS[name]
            )
        _require_voc_gate_policy_protocol(
            gate_protocol, label=f"{label} active VoC checkpoint"
        )
        _require_voc_gate_policy_protocol(
            {
                name: getattr(config_flags, name, VOC_PROTOCOL_DEFAULTS[name])
                for name in gate_protocol_names
            },
            label=f"{label} active VoC config_c.yaml",
        )
    for key in RUNTIME_SEMANTIC_FIELDS:
        if (
            embedded_voc_mode == "off"
            and key in VOC_ACTIVE_ONLY_PROTOCOL_FIELDS
        ):
            # Entropy reward predates VoC and is training-only in off mode.
            # Do not make legacy-off evaluation depend on new VoC metadata.
            continue
        actual = embedded.get(key)
        if (
            key in (
                "voc_gate_adam_beta1",
                "voc_gate_param_align",
                "voc_gate_param_align_coef",
                "voc_gate_exact_projection",
                "voc_gate_epsilon_greedy_execution",
                "voc_gate_execution_epsilon",
                "voc_actor_policy_version_barrier",
                "voc_actor_policy_bundle_schema_version",
                "voc_actor_policy_barrier_timeout_s",
                "voc_actor_policy_ray_max_restarts",
                "voc_actor_policy_ray_max_task_retries",
                "actor_amp_init_scale",
                "voc_model_input_seal_schema_version",
            )
            and embedded_voc_mode != "off"
            and gate_schema_state is not None
        ):
            actual = gate_schema_state.get(key, VOC_PROTOCOL_DEFAULTS[key])
        if key in VOC_PROTOCOL_DEFAULTS and actual is None:
            if embedded_voc_mode != "off":
                raise ValueError(
                    f"{label} active VoC checkpoint lacks embedded {key}"
                )
            actual = VOC_PROTOCOL_DEFAULTS[key]
        if key == "model_float16" and actual is None:
            actual = embedded.get("float16")
        if key == "model_state_projection" and actual is None:
            actual = "none"
        if key == "model_state_range_loss_cost" and actual is None:
            actual = 0.0
        if key == "schedule_total_steps" and actual is None:
            actual = embedded.get("total_steps")
        configured = getattr(
            config_flags, key, VOC_PROTOCOL_DEFAULTS.get(key)
        )
        if key == "model_state_projection" and configured is None:
            configured = "none"
        if key == "model_state_range_loss_cost" and configured is None:
            configured = 0.0
        matches_config = (
            float(actual) == float(configured)
            if key in (
                "voc_gate_adam_beta1",
                "voc_gate_param_align_coef",
            )
            else _protocol_value_matches(actual, configured)
        )
        if not matches_config:
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


def validate_actor_voc_provenance(
    checkpoint: Mapping[str, Any],
    embedded_flags: Mapping[str, Any],
    config_flags: Any = None,
) -> Dict[str, Any]:
    """Validate VoC mode/provenance without rejecting legacy-off snapshots."""

    mode = embedded_flags.get(
        "dynamic_voc_mode", VOC_PROTOCOL_DEFAULTS["dynamic_voc_mode"]
    )
    saved_mode = checkpoint.get("dynamic_voc_mode")
    if mode == "off" and saved_mode is None:
        return {
            "dynamic_voc_mode": "off",
            "legacy_voc_metadata_defaulted": True,
            "voc_update_count": 0,
            "voc_continue_count": 0,
            "voc_stop_count": 0,
            "voc_control_origin": None,
            "voc_control_origin_legacy_defaulted": False,
            "voc_parent_checkpoint_sha256": None,
            "voc_parent_checkpoint": None,
            "voc_parent_imitation_data_signature": None,
            "voc_activation_real_step": -1,
        }
    if saved_mode != mode:
        raise ValueError(
            "actor checkpoint VoC mode disagrees with embedded flags: "
            f"{saved_mode!r} versus {mode!r}"
        )
    if mode not in {"off", "shadow", "control"}:
        raise ValueError(f"actor checkpoint has invalid dynamic_voc_mode={mode!r}")

    counters = {}
    for key in ("voc_update_count", "voc_continue_count", "voc_stop_count"):
        value = checkpoint.get(key)
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
        ):
            raise ValueError(f"actor checkpoint has invalid {key}")
        value = int(value)
        if value < 0:
            raise ValueError(f"actor checkpoint has negative {key}")
        counters[key] = value

    from thinker import util as thinker_util

    if mode == "control":
        control_provenance = (
            thinker_util.validate_voc_control_checkpoint_provenance(
                checkpoint, label="control actor checkpoint"
            )
        )
        activation_real_step = control_provenance[
            "voc_activation_real_step"
        ]
        parent_sha256 = control_provenance[
            "voc_parent_checkpoint_sha256"
        ]
    elif mode == "shadow":
        control_provenance = (
            thinker_util.validate_voc_shadow_checkpoint_provenance(
                checkpoint, label="shadow actor checkpoint"
            )
        )
        activation_real_step = control_provenance[
            "voc_activation_real_step"
        ]
        parent_sha256 = None
    else:
        activation_real_step = -1
        parent_sha256 = None
        control_provenance = {
            "voc_control_origin": None,
            "voc_control_origin_legacy_defaulted": False,
            "voc_parent_checkpoint_sha256": None,
            "voc_parent_checkpoint": None,
            "voc_parent_imitation_data_signature": None,
            "voc_activation_real_step": -1,
        }

    if mode != "off":
        for key, value in counters.items():
            if value <= 0:
                raise ValueError(
                    f"{mode} checkpoint must have positive {key}; got {value}"
                )
        holdout_calibration = thinker_util.validate_voc_holdout_calibration(
            checkpoint,
            label=f"{mode} actor checkpoint",
            require_positive_support=True,
        )
        holdout_split = thinker_util.validate_voc_holdout_split(
            checkpoint,
            flags=config_flags,
            label=f"{mode} actor checkpoint",
        )
        matched_pair = thinker_util.validate_voc_checkpoint_components(
            checkpoint,
            flags=config_flags,
            label=f"{mode} actor checkpoint",
        )
        ema_gate_state = thinker_util.validate_voc_ema_gate_checkpoint(
            checkpoint, label=f"{mode} actor checkpoint"
        )
        amp_state = thinker_util.validate_voc_amp_checkpoint(
            checkpoint, label=f"{mode} actor checkpoint"
        )
        gate_policy_state = (
            thinker_util.validate_voc_gate_policy_checkpoint(
                checkpoint,
                flags=config_flags,
                label=f"{mode} actor checkpoint",
            )
        )
        if (
            mode == "control"
            and gate_policy_state["voc_gate_update_count"] <= 0
        ):
            raise ValueError(
                "control checkpoint must have positive "
                "voc_gate_update_count"
            )
    return {
        "dynamic_voc_mode": mode,
        "legacy_voc_metadata_defaulted": False,
        **counters,
        **(holdout_calibration if mode != "off" else {}),
        **(holdout_split if mode != "off" else {}),
        **(amp_state if mode != "off" else {}),
        **(gate_policy_state if mode != "off" else {}),
        **(
            {
                "voc_ema_gate_target": ema_gate_state[
                    "voc_ema_gate_target"
                ],
                "voc_gate_target_tau": ema_gate_state[
                    "voc_gate_target_tau"
                ],
                "voc_ema_gate_schema_version": ema_gate_state[
                    "voc_ema_gate_schema_version"
                ],
                "voc_ema_gate_update_count": ema_gate_state[
                    "voc_ema_gate_update_count"
                ],
                "voc_ema_gate_parent_update_count": ema_gate_state[
                    "voc_ema_gate_parent_update_count"
                ],
                "voc_ema_gate_head_state_saved": True,
            }
            if mode != "off" else {}
        ),
        **control_provenance,
        "voc_head_keys": list(matched_pair) if mode != "off" else None,
        "voc_optimizer_state_saved": mode != "off",
    }


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
    voc_state = validate_actor_voc_provenance(
        checkpoint, embedded, config_flags
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
        "voc": voc_state,
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
    if saved_config.get("voc_gate_policy_schema_version") not in (
        VOC_GATE_POLICY_VERSION_BARRIER_SCHEMA_VERSION,
        VOC_GATE_POLICY_MODEL_INPUT_SEAL_SCHEMA_VERSION,
        VOC_GATE_POLICY_HALF_SQUARED_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_COMMON_MODE_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_HUBER_COMMON_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION,
    ):
        flags.checkpoint_git_revision = saved_config.get("git_revision")
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


def _load_flags_from_validated_config_bytes(
    checkpoint_dir: str | Path,
    config_payload: bytes,
    expected_sha256: str,
) -> Any:
    """Load flags from already-bound bytes without reopening the checkpoint."""

    from thinker import util

    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    if type(config_payload) is not bytes:
        raise TypeError("validated checkpoint config payload must be exact bytes")
    if (
        type(expected_sha256) is not str
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
        or hashlib.sha256(config_payload).hexdigest() != expected_sha256
    ):
        raise ValueError("validated checkpoint config digest disagrees")
    try:
        saved_config = yaml.safe_load(config_payload.decode("utf-8")) or {}
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError("validated checkpoint config is not UTF-8 YAML") from error
    if type(saved_config) is not dict:
        raise ValueError("validated checkpoint config must contain a mapping")

    gate_schema = saved_config.get("voc_gate_policy_schema_version")
    schema13_intent = (
        type(gate_schema) is int
        and gate_schema == VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION
    ) or _schema13_xpid_claims_intent(saved_config.get("xpid"))
    if schema13_intent:
        util.validate_voc_gate_policy_schema(
            {
                "voc_gate_policy_schema_version": gate_schema,
                "flags": saved_config,
            },
            label="validated schema-13 evaluation config",
        )
        flags = argparse.Namespace(**copy.deepcopy(saved_config))
        flags = util.process_flags(flags)
        flags = util.process_flags_actor(flags)
        if vars(flags) != saved_config:
            raise RuntimeError(
                "schema-13 validated-byte flag reconstruction changed the "
                "frozen config surface"
            )
        util.validate_voc_gate_policy_schema(
            {
                "voc_gate_policy_schema_version": gate_schema,
                "flags": vars(flags),
            },
            label="reconstructed schema-13 evaluation flags",
        )
    else:
        with tempfile.TemporaryDirectory(prefix="thinker-eval-config-") as temp_dir:
            private_config = Path(temp_dir) / "config_c.yaml"
            descriptor = os.open(
                private_config,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    descriptor = -1
                    handle.write(config_payload)
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
            flags = util.create_flags(
                ["default_thinker.yaml", "default_actor.yaml"],
                save_flags=False,
                post_fn=util.process_flags_actor,
                config=str(private_config),
                ckp=False,
            )
    flags.ckpdir = str(checkpoint_dir)
    if saved_config.get("voc_gate_policy_schema_version") not in (
        VOC_GATE_POLICY_VERSION_BARRIER_SCHEMA_VERSION,
        VOC_GATE_POLICY_MODEL_INPUT_SEAL_SCHEMA_VERSION,
        VOC_GATE_POLICY_HALF_SQUARED_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_COMMON_MODE_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_HUBER_COMMON_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION,
    ):
        flags.checkpoint_git_revision = saved_config.get("git_revision")
    if not bool(getattr(flags, "dynamic_search", False)):
        raise ValueError("checkpoint is not a Dynamic Thinker checkpoint")
    for key, expected in {**DYNAMIC_PROTOCOL, **IMITATION_PROTOCOL}.items():
        actual = getattr(flags, key, None)
        if not _protocol_value_matches(actual, expected):
            raise ValueError(
                f"checkpoint protocol mismatch: {key}={actual!r}, "
                f"expected {expected!r}"
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
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    schema13_probe_payload = _read_stable_single_link_bytes(
        checkpoint_dir / "config_c.yaml", label="checkpoint schema claim"
    )
    try:
        schema13_probe = yaml.safe_load(schema13_probe_payload.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError("checkpoint config is not UTF-8 YAML") from error
    if not isinstance(schema13_probe, Mapping):
        raise ValueError("checkpoint config_c.yaml must contain a mapping")
    schema13_claimed = (
        schema13_probe.get("voc_gate_policy_schema_version")
        == VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION
        or _schema13_xpid_claims_intent(schema13_probe.get("xpid"))
    )
    if schema13_claimed:
        completion_state = validate_schema13_completion_marker(checkpoint_dir)
        loaded_checkpoint_hashes = _schema13_checkpoint_hashes(
            checkpoint_dir, completion_state=completion_state
        )
    else:
        loaded_checkpoint_hashes = checkpoint_hashes(checkpoint_dir)
        completion_state = validate_completion_marker(checkpoint_dir)
    validated_config_digest = completion_state["checkpoint_files"][
        "config_c.yaml"
    ]["sha256"]
    validated_config_payload = (
        schema13_probe_payload
        if schema13_claimed
        else _read_stable_single_link_bytes(
            checkpoint_dir / "config_c.yaml",
            label="checkpoint config",
        )
    )
    if (
        hashlib.sha256(validated_config_payload).hexdigest()
        != validated_config_digest
        or loaded_checkpoint_hashes.get("config_c.yaml")
        != validated_config_digest
    ):
        raise RuntimeError(
            "checkpoint config changed after completion validation"
        )
    schema13_bundle_validation = dispatch_schema13_completed_bundle(
        checkpoint_dir,
        completion_state=completion_state,
        config_payload=validated_config_payload,
        expected_config_sha256=validated_config_digest,
    )
    schema8_bundle_validation = dispatch_schema8_completed_bundle(
        checkpoint_dir,
        completion_state=completion_state,
        config_payload=validated_config_payload,
        expected_config_sha256=validated_config_digest,
    )
    schema9_bundle_validation = dispatch_schema9_completed_bundle(
        checkpoint_dir,
        completion_state=completion_state,
        config_payload=validated_config_payload,
        expected_config_sha256=validated_config_digest,
    )
    schema10_bundle_validation = dispatch_schema10_completed_bundle(
        checkpoint_dir,
        completion_state=completion_state,
        config_payload=validated_config_payload,
        expected_config_sha256=validated_config_digest,
    )
    schema11_bundle_validation = dispatch_schema11_completed_bundle(
        checkpoint_dir,
        completion_state=completion_state,
        config_payload=validated_config_payload,
        expected_config_sha256=validated_config_digest,
    )
    schema12_bundle_validation = dispatch_schema12_completed_bundle(
        checkpoint_dir,
        completion_state=completion_state,
        config_payload=validated_config_payload,
        expected_config_sha256=validated_config_digest,
    )
    try:
        raw_training_config = (
            yaml.safe_load(validated_config_payload.decode("utf-8")) or {}
        )
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError("checkpoint config is not UTF-8 YAML") from error
    if not isinstance(raw_training_config, Mapping):
        raise ValueError("checkpoint config_c.yaml must contain a mapping")
    training_flags = _load_flags_from_validated_config_bytes(
        checkpoint_dir,
        validated_config_payload,
        validated_config_digest,
    )
    current_checkpoint_hashes = (
        _schema13_checkpoint_hashes(
            checkpoint_dir, completion_state=completion_state
        )
        if schema13_bundle_validation is not None
        else checkpoint_hashes(checkpoint_dir)
    )
    if current_checkpoint_hashes != loaded_checkpoint_hashes:
        raise RuntimeError(
            "checkpoint changed before evaluation flag resolution"
        )
    schema6_bundle_validation = validate_schema6_completed_bundle(
        checkpoint_dir, completion_state=completion_state
    )
    schema7_bundle_validation = validate_schema7_completed_bundle(
        checkpoint_dir, completion_state=completion_state
    )
    resolved_versioned_bundles = sum(
        validation is not None
        for validation in (
            schema6_bundle_validation,
            schema7_bundle_validation,
            schema8_bundle_validation,
            schema9_bundle_validation,
            schema10_bundle_validation,
            schema11_bundle_validation,
            schema12_bundle_validation,
            schema13_bundle_validation,
        )
    )
    if resolved_versioned_bundles > 1:
        raise RuntimeError("checkpoint resolved to multiple versioned schemas")
    flags, atomic_runtime_state = evaluation_runtime_flags(training_flags)
    if (
        schema6_bundle_validation is not None
        or schema7_bundle_validation is not None
        or schema8_bundle_validation is not None
        or schema9_bundle_validation is not None
        or schema10_bundle_validation is not None
        or schema11_bundle_validation is not None
        or schema12_bundle_validation is not None
        or schema13_bundle_validation is not None
    ):
        flags.checkpoint_git_revision = raw_training_config.get("git_revision")
    flags.checkpoint_completion_state = completion_state
    flags.schema6_final_bundle_validation = schema6_bundle_validation
    flags.schema6_runtime_state = (
        atomic_runtime_state if schema6_bundle_validation is not None else None
    )
    flags.schema6_training_protocol = (
        checkpoint_protocol(training_flags)
        if schema6_bundle_validation is not None
        else None
    )
    flags.schema7_final_bundle_validation = schema7_bundle_validation
    flags.schema7_runtime_state = (
        atomic_runtime_state if schema7_bundle_validation is not None else None
    )
    flags.schema7_training_protocol = (
        checkpoint_protocol(training_flags)
        if schema7_bundle_validation is not None
        else None
    )
    flags.schema8_final_bundle_validation = schema8_bundle_validation
    flags.schema8_runtime_state = (
        atomic_runtime_state if schema8_bundle_validation is not None else None
    )
    flags.schema8_training_protocol = (
        checkpoint_protocol(training_flags)
        if schema8_bundle_validation is not None
        else None
    )
    flags.schema9_final_bundle_validation = schema9_bundle_validation
    flags.schema9_runtime_state = (
        atomic_runtime_state if schema9_bundle_validation is not None else None
    )
    flags.schema9_training_protocol = (
        checkpoint_protocol(training_flags)
        if schema9_bundle_validation is not None
        else None
    )
    flags.schema10_final_bundle_validation = schema10_bundle_validation
    flags.schema10_runtime_state = (
        atomic_runtime_state if schema10_bundle_validation is not None else None
    )
    flags.schema10_training_protocol = (
        checkpoint_protocol(training_flags)
        if schema10_bundle_validation is not None
        else None
    )
    flags.schema11_final_bundle_validation = schema11_bundle_validation
    flags.schema11_runtime_state = (
        atomic_runtime_state if schema11_bundle_validation is not None else None
    )
    flags.schema11_training_protocol = (
        checkpoint_protocol(training_flags)
        if schema11_bundle_validation is not None
        else None
    )
    flags.schema12_final_bundle_validation = schema12_bundle_validation
    flags.schema12_runtime_state = (
        atomic_runtime_state if schema12_bundle_validation is not None else None
    )
    flags.schema12_training_protocol = (
        checkpoint_protocol(training_flags)
        if schema12_bundle_validation is not None
        else None
    )
    flags.schema13_final_bundle_validation = schema13_bundle_validation
    flags.schema13_runtime_state = (
        atomic_runtime_state if schema13_bundle_validation is not None else None
    )
    flags.schema13_training_protocol = (
        checkpoint_protocol(training_flags)
        if schema13_bundle_validation is not None
        else None
    )
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
    actor_checkpoint = _load_runtime_checkpoint(
        checkpoint_dir,
        "ckp_actor.tar",
        completion_state,
        schema13=schema13_bundle_validation is not None,
        label="actor checkpoint",
    )
    model_checkpoint = _load_runtime_checkpoint(
        checkpoint_dir,
        "ckp_model.tar",
        completion_state,
        schema13=schema13_bundle_validation is not None,
        label="model checkpoint",
    )
    current_checkpoint_hashes = (
        _schema13_checkpoint_hashes(
            checkpoint_dir, completion_state=completion_state
        )
        if schema13_bundle_validation is not None
        else checkpoint_hashes(checkpoint_dir)
    )
    if current_checkpoint_hashes != loaded_checkpoint_hashes:
        raise RuntimeError(
            "checkpoint files changed while they were being loaded; retry with "
            "an immutable snapshot"
        )
    flags.actor_checkpoint_imitation_state = validate_actor_imitation_checkpoint(
        actor_checkpoint, training_flags, spec
    )
    flags.model_checkpoint_validation_state = validate_model_checkpoint(
        model_checkpoint, training_flags, spec
    )
    from thinker.bc_loader import (
        FrameStackedBehavioralDataLoader,
        behavioral_data_signature,
    )
    from thinker.dynamic_imitation import DynamicImitationRunner

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    _set_pair_seed(int(args.seed))
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
    if schema6_bundle_validation is not None:
        final_schema6_validation = validate_schema6_completed_bundle(
            checkpoint_dir, completion_state=completion_state
        )
        if final_schema6_validation != schema6_bundle_validation:
            raise RuntimeError(
                "schema-6 completed-bundle evidence changed during evaluation"
            )
    if schema7_bundle_validation is not None:
        final_schema7_validation = validate_schema7_completed_bundle(
            checkpoint_dir, completion_state=completion_state
        )
        if final_schema7_validation != schema7_bundle_validation:
            raise RuntimeError(
                "schema-7 completed-bundle evidence changed during evaluation"
            )
    if schema8_bundle_validation is not None:
        final_schema8_validation = validate_schema8_completed_bundle(
            checkpoint_dir, completion_state=completion_state
        )
        if final_schema8_validation != schema8_bundle_validation:
            raise RuntimeError(
                "schema-8 completed-bundle evidence changed during evaluation"
            )
    if schema9_bundle_validation is not None:
        final_schema9_validation = validate_schema9_completed_bundle(
            checkpoint_dir,
            completion_state=completion_state,
            config_payload=validated_config_payload,
            expected_config_sha256=validated_config_digest,
        )
        if final_schema9_validation != schema9_bundle_validation:
            raise RuntimeError(
                "schema-9 completed-bundle evidence changed during evaluation"
            )
    if schema10_bundle_validation is not None:
        final_schema10_validation = validate_schema10_completed_bundle(
            checkpoint_dir,
            completion_state=completion_state,
            config_payload=validated_config_payload,
            expected_config_sha256=validated_config_digest,
        )
        if final_schema10_validation != schema10_bundle_validation:
            raise RuntimeError(
                "schema-10 completed-bundle evidence changed during evaluation"
            )
    if schema11_bundle_validation is not None:
        final_schema11_validation = validate_schema11_completed_bundle(
            checkpoint_dir,
            completion_state=completion_state,
            config_payload=validated_config_payload,
            expected_config_sha256=validated_config_digest,
        )
        if final_schema11_validation != schema11_bundle_validation:
            raise RuntimeError(
                "schema-11 completed-bundle evidence changed during evaluation"
            )
    if schema12_bundle_validation is not None:
        final_schema12_validation = validate_schema12_completed_bundle(
            checkpoint_dir,
            completion_state=completion_state,
            config_payload=validated_config_payload,
            expected_config_sha256=validated_config_digest,
        )
        if final_schema12_validation != schema12_bundle_validation:
            raise RuntimeError(
                "schema-12 completed-bundle evidence changed during evaluation"
            )
    if schema13_bundle_validation is not None:
        final_schema13_validation = validate_schema13_completed_bundle(
            checkpoint_dir,
            completion_state=completion_state,
            config_payload=validated_config_payload,
            expected_config_sha256=validated_config_digest,
        )
        if final_schema13_validation != schema13_bundle_validation:
            raise RuntimeError(
                "schema-13 completed-bundle evidence changed during evaluation"
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
    summary["schema6_final_bundle_validation"] = schema6_bundle_validation
    summary["schema6_training_protocol"] = flags.schema6_training_protocol
    summary["schema6_runtime_state"] = flags.schema6_runtime_state
    if schema7_bundle_validation is not None:
        summary["schema7_final_bundle_validation"] = schema7_bundle_validation
        summary["schema7_training_protocol"] = flags.schema7_training_protocol
        summary["schema7_runtime_state"] = flags.schema7_runtime_state
    if schema8_bundle_validation is not None:
        summary["schema8_final_bundle_validation"] = schema8_bundle_validation
        summary["schema8_training_protocol"] = flags.schema8_training_protocol
        summary["schema8_runtime_state"] = flags.schema8_runtime_state
    if schema9_bundle_validation is not None:
        summary["schema9_final_bundle_validation"] = schema9_bundle_validation
        summary["schema9_training_protocol"] = flags.schema9_training_protocol
        summary["schema9_runtime_state"] = flags.schema9_runtime_state
    if schema10_bundle_validation is not None:
        summary["schema10_final_bundle_validation"] = schema10_bundle_validation
        summary["schema10_training_protocol"] = flags.schema10_training_protocol
        summary["schema10_runtime_state"] = flags.schema10_runtime_state
    if schema11_bundle_validation is not None:
        summary["schema11_final_bundle_validation"] = schema11_bundle_validation
        summary["schema11_training_protocol"] = flags.schema11_training_protocol
        summary["schema11_runtime_state"] = flags.schema11_runtime_state
    if schema12_bundle_validation is not None:
        summary["schema12_final_bundle_validation"] = schema12_bundle_validation
        summary["schema12_training_protocol"] = flags.schema12_training_protocol
        summary["schema12_runtime_state"] = flags.schema12_runtime_state
    if schema13_bundle_validation is not None:
        summary["schema13_final_bundle_validation"] = schema13_bundle_validation
        summary["schema13_training_protocol"] = flags.schema13_training_protocol
        summary["schema13_runtime_state"] = flags.schema13_runtime_state
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
    manifest["schema6_final_bundle_validation"] = schema6_bundle_validation
    manifest["schema6_training_protocol"] = flags.schema6_training_protocol
    manifest["schema6_runtime_state"] = flags.schema6_runtime_state
    if schema7_bundle_validation is not None:
        manifest["schema7_final_bundle_validation"] = schema7_bundle_validation
        manifest["schema7_training_protocol"] = flags.schema7_training_protocol
        manifest["schema7_runtime_state"] = flags.schema7_runtime_state
    if schema8_bundle_validation is not None:
        manifest["schema8_final_bundle_validation"] = schema8_bundle_validation
        manifest["schema8_training_protocol"] = flags.schema8_training_protocol
        manifest["schema8_runtime_state"] = flags.schema8_runtime_state
    if schema9_bundle_validation is not None:
        manifest["schema9_final_bundle_validation"] = schema9_bundle_validation
        manifest["schema9_training_protocol"] = flags.schema9_training_protocol
        manifest["schema9_runtime_state"] = flags.schema9_runtime_state
    if schema10_bundle_validation is not None:
        manifest["schema10_final_bundle_validation"] = schema10_bundle_validation
        manifest["schema10_training_protocol"] = flags.schema10_training_protocol
        manifest["schema10_runtime_state"] = flags.schema10_runtime_state
    if schema11_bundle_validation is not None:
        manifest["schema11_final_bundle_validation"] = schema11_bundle_validation
        manifest["schema11_training_protocol"] = flags.schema11_training_protocol
        manifest["schema11_runtime_state"] = flags.schema11_runtime_state
    if schema12_bundle_validation is not None:
        manifest["schema12_final_bundle_validation"] = schema12_bundle_validation
        manifest["schema12_training_protocol"] = flags.schema12_training_protocol
        manifest["schema12_runtime_state"] = flags.schema12_runtime_state
    if schema13_bundle_validation is not None:
        manifest["schema13_final_bundle_validation"] = schema13_bundle_validation
        manifest["schema13_training_protocol"] = flags.schema13_training_protocol
        manifest["schema13_runtime_state"] = flags.schema13_runtime_state
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
