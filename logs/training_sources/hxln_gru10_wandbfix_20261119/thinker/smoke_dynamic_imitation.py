#!/usr/bin/env python3
"""One-update smoke test for the real Dynamic imitation components.

This intentionally avoids Ray.  It obtains observation/action metadata from
the requested live EnvPool Atari environment, samples genuine behavioral
frames, constructs the production ActorNet/ModelNet/cModelWrapper stack, and
runs one differentiable teacher-forced imitation rollout plus optimizer step.
The JSON result proves that Actor weights changed while ModelNet stayed frozen.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import random
import re
import tempfile
from typing import Any, Optional, Sequence

import numpy as np
import torch
import yaml
from gymnasium import spaces


def _smoke_schema13_xpid_claims_intent(value: Any) -> bool:
    """Classify forward V20 intent without depending on a frozen evaluator."""

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
            "smoke schema-13 xpid intent could not be classified before "
            "downstream I/O"
        ) from error
    return lexical_value.strip().startswith(
        "enduro-voc-v20-telemetry-tau1-orthocd-adam-eps25-"
    )


_SCHEMA10_FINAL_ACTOR_EVIDENCE_FIELDS = frozenset(
    {
        "actor_amp_consecutive_skips",
        "actor_amp_growth_tracker",
        "actor_amp_init_scale",
        "actor_amp_init_scale_legacy_defaulted",
        "actor_amp_scale",
        "actor_amp_skip_count",
        "actor_batch_size",
        "dynamic_voc_mode",
        "env_n",
        "ppo_k",
        "self_play_n",
        "voc_activation_real_step",
        "voc_actor_policy_barrier_timeout_count",
        "voc_actor_policy_barrier_timeout_s",
        "voc_actor_policy_barrier_timeout_s_legacy_defaulted",
        "voc_actor_policy_bundle_schema_version",
        "voc_actor_policy_bundle_schema_version_legacy_defaulted",
        "voc_actor_policy_bundle_summary",
        "voc_actor_policy_expected_ack_count",
        "voc_actor_policy_final_publication_event",
        "voc_actor_policy_malformed_bundle_count",
        "voc_actor_policy_publication_count",
        "voc_actor_policy_publication_event_count",
        "voc_actor_policy_publication_history",
        "voc_actor_policy_publication_history_sha256",
        "voc_actor_policy_ray_max_restarts",
        "voc_actor_policy_ray_max_restarts_legacy_defaulted",
        "voc_actor_policy_ray_max_task_retries",
        "voc_actor_policy_ray_max_task_retries_legacy_defaulted",
        "voc_actor_policy_state_sha256",
        "voc_actor_policy_terminal",
        "voc_actor_policy_terminal_ack_count",
        "voc_actor_policy_version",
        "voc_actor_policy_version_barrier",
        "voc_actor_policy_version_barrier_legacy_defaulted",
        "voc_actor_policy_version_mismatch_count",
        "voc_amp_consecutive_skips",
        "voc_amp_skip_count",
        "voc_continue_count",
        "voc_control_origin",
        "voc_control_origin_legacy_defaulted",
        "voc_dedicated_gate",
        "voc_ema_gate_head_state_dict",
        "voc_ema_gate_parent_update_count",
        "voc_ema_gate_schema_version",
        "voc_ema_gate_target",
        "voc_ema_gate_update_count",
        "voc_float16",
        "voc_gate_adam_beta1",
        "voc_gate_adam_beta1_legacy_defaulted",
        "voc_gate_amp_consecutive_skips",
        "voc_gate_amp_skip_count",
        "voc_gate_confidence_weighted",
        "voc_gate_epsilon_greedy_execution",
        "voc_gate_epsilon_greedy_execution_legacy_defaulted",
        "voc_gate_exact_projection",
        "voc_gate_exact_projection_legacy_defaulted",
        "voc_gate_execution_epsilon",
        "voc_gate_execution_epsilon_legacy_defaulted",
        "voc_gate_grad_norm_clipping",
        "voc_gate_grad_scaler_state_saved",
        "voc_gate_head_keys",
        "voc_gate_learning_rate",
        "voc_gate_optimizer_state_saved",
        "voc_gate_param_align",
        "voc_gate_param_align_coef",
        "voc_gate_param_align_legacy_defaulted",
        "voc_gate_policy_schema_version",
        "voc_gate_q_temperature",
        "voc_gate_scheduler_state_saved",
        "voc_gate_target_tau",
        "voc_gate_update_count",
        "voc_grad_scaler_state_saved",
        "voc_holdout_actor_modulus",
        "voc_holdout_actor_streams",
        "voc_holdout_continue_count",
        "voc_holdout_count",
        "voc_holdout_env_n",
        "voc_holdout_self_play_n",
        "voc_holdout_split_version",
        "voc_holdout_stop_count",
        "voc_holdout_td_abs_sum",
        "voc_holdout_td_bias",
        "voc_holdout_td_mae",
        "voc_holdout_td_rmse",
        "voc_holdout_td_sq_sum",
        "voc_holdout_td_sum",
        "voc_model_input_seal_schema_version",
        "voc_model_input_seal_schema_version_legacy_defaulted",
        "voc_parent_checkpoint",
        "voc_parent_checkpoint_sha256",
        "voc_parent_imitation_data_signature",
        "voc_protocol",
        "voc_q_reconstruction",
        "voc_q_regression_loss",
        "voc_soft_q_bce_gate",
        "voc_stop_count",
        "voc_update_count",
    }
)
_SCHEMA11_FINAL_ACTOR_EVIDENCE_FIELDS = (
    _SCHEMA10_FINAL_ACTOR_EVIDENCE_FIELDS
    | {"voc_q_optimizer_coordinates"}
)
_SCHEMA11_RESOLVED_IDENTITY_FIELDS = frozenset(
    {
        "key_count",
        "v12_projection_key_count",
        "v12_projection_sha256",
        "complete_surface_sha256",
        "stage",
        "paths",
        "gate_schema",
        "voc_gate_policy_schema_version",
        "voc_model_input_seal_schema_version",
        "voc_q_regression_loss",
        "voc_q_reconstruction",
        "voc_q_optimizer_coordinates",
    }
)
_SCHEMA11_COMPLETED_BUNDLE_FIELDS = frozenset(
    {
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
)
_SCHEMA11_DERIVED_IDENTITY = {
    "voc_q_regression_loss": "smooth_l1_beta1",
    "voc_q_reconstruction": (
        "detached_value_plus_raw_head_mean_plus_policy_centered_raw_head"
    ),
    "voc_q_optimizer_coordinates": "orthonormal_common_difference_adam",
}
_SCHEMA12_FINAL_ACTOR_EVIDENCE_FIELDS = _SCHEMA11_FINAL_ACTOR_EVIDENCE_FIELDS
_SCHEMA12_RESOLVED_IDENTITY_FIELDS = _SCHEMA11_RESOLVED_IDENTITY_FIELDS
_SCHEMA12_COMPLETED_BUNDLE_FIELDS = _SCHEMA11_COMPLETED_BUNDLE_FIELDS
_SCHEMA12_DERIVED_IDENTITY = _SCHEMA11_DERIVED_IDENTITY
_SCHEMA13_FINAL_ACTOR_EVIDENCE_FIELDS = _SCHEMA12_FINAL_ACTOR_EVIDENCE_FIELDS
_SCHEMA13_RESOLVED_IDENTITY_FIELDS = _SCHEMA12_RESOLVED_IDENTITY_FIELDS
_SCHEMA13_COMPLETED_BUNDLE_FIELDS = (
    _SCHEMA12_COMPLETED_BUNDLE_FIELDS | {"telemetry"}
)
_SCHEMA13_DERIVED_IDENTITY = _SCHEMA12_DERIVED_IDENTITY
_SCHEMA13_TELEMETRY_EVIDENCE_FIELDS = frozenset(
    {
        "telemetry_schema_version",
        "gate_schema",
        "manifest_name",
        "manifest_sha256",
        "manifest_size",
        "transaction_count",
        "terminal_policy_version",
        "terminal_real_step",
        "actor_state_sha256",
        "publication_history_sha256",
    }
)


def _parse_ids(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result:
        raise argparse.ArgumentTypeError("ID list cannot be empty")
    return result


def _checkpoint_state_dict(checkpoint: Any, key: str) -> Mapping[str, Any]:
    if isinstance(checkpoint, Mapping) and key in checkpoint:
        state = checkpoint[key]
    else:
        state = checkpoint
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"checkpoint does not contain a non-empty {key!r}")
    return state


def _validate_state_dict(module: torch.nn.Module, state: Mapping[str, Any], label: str):
    expected = module.state_dict()
    expected_keys = set(expected)
    incoming_keys = set(state)
    missing = sorted(expected_keys - incoming_keys)
    unexpected = sorted(incoming_keys - expected_keys)
    mismatched = []
    for key in sorted(expected_keys & incoming_keys):
        incoming_shape = tuple(np.shape(state[key]))
        expected_shape = tuple(expected[key].shape)
        if incoming_shape != expected_shape:
            mismatched.append((key, incoming_shape, expected_shape))
    if missing or unexpected or mismatched:
        details = []
        if missing:
            details.append(f"missing={missing[:8]}")
        if unexpected:
            details.append(f"unexpected={unexpected[:8]}")
        if mismatched:
            details.append(
                "shape_mismatch=["
                + "; ".join(
                    f"{key}: incoming{incoming} != expected{target}"
                    for key, incoming, target in mismatched[:8]
                )
                + "]"
            )
        raise ValueError(f"{label} state-dict is incompatible: " + ", ".join(details))


def _checkpoint_evaluation_spec(
    args: argparse.Namespace,
    flags: Any,
    obs_space: spaces.Box,
    action_space: spaces.Discrete,
    frame_stack_n: int,
):
    """Bind public checkpoint validation to this smoke's live/data identity."""

    import evaluate_dynamic_imitation as checkpoint_eval

    return checkpoint_eval.EvaluationSpec(
        subjects=checkpoint_eval._parse_id_setting(
            args.subjects, "icopro_subjects"
        ),
        train_sessions=checkpoint_eval._parse_id_setting(
            args.sessions, "icopro_train_sessions"
        ),
        holdout_sessions=checkpoint_eval._parse_id_setting(
            getattr(flags, "icopro_holdout_sessions", None),
            "icopro_holdout_sessions",
        ),
        game_id=int(args.game_id),
        env_name=str(args.env_name),
        num_actions=int(action_space.n),
        scored_length=int(args.scored_length),
        frame_stack_n=int(frame_stack_n),
        grayscale=bool(flags.grayscale),
        observation_shape=tuple(int(value) for value in obs_space.shape),
        observation_dtype=np.dtype(obs_space.dtype).name,
        target_size=tuple(int(value) for value in obs_space.shape[-2:]),
        observation_low=float(np.min(obs_space.low)),
        observation_high=float(np.max(obs_space.high)),
    )


def _schema5_identity_value(source: Any, name: str, kind: str, *, label: str):
    if isinstance(source, Mapping):
        if name not in source:
            raise ValueError(f"{label} lacks explicit {name}")
        value = source[name]
    else:
        if not hasattr(source, name):
            raise ValueError(f"{label} lacks explicit {name}")
        value = getattr(source, name)

    if kind == "bool":
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{label} has non-boolean {name}={value!r}")
        return bool(value)
    if kind == "int":
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
        ):
            raise ValueError(f"{label} has non-integer {name}={value!r}")
        normalized = int(value)
        minimum = 0 if name == "base_seed" else 1
        if normalized < minimum:
            raise ValueError(f"{label} has invalid {name}={value!r}")
        return normalized
    if kind == "float":
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.number))
            or not np.isfinite(value)
        ):
            raise ValueError(
                f"{label} has non-finite/non-numeric {name}={value!r}"
            )
        return float(value)
    if kind == "str":
        if not isinstance(value, str):
            raise ValueError(f"{label} has non-string {name}={value!r}")
        return value
    raise AssertionError(f"unsupported identity kind {kind!r}")


def _schema5_checkpoint_identity(source: Any, *, label: str) -> dict[str, Any]:
    kinds = {
        "base_seed": "int",
        "total_steps": "int",
        "schedule_total_steps": "int",
        "dynamic_voc_mode": "str",
        "voc_gate_adam_beta1": "float",
        "voc_gate_param_align": "bool",
        "voc_gate_param_align_coef": "float",
        "voc_gate_exact_projection": "bool",
        "voc_gate_epsilon_greedy_execution": "bool",
        "voc_train_epsilon": "float",
        "voc_eval_stochastic": "bool",
        "float16": "bool",
        "model_float16": "bool",
        "ckp": "bool",
        "preload": "str",
        "preload_actor": "str",
        "voc_parent_checkpoint": "str",
    }
    identity = {
        name: _schema5_identity_value(source, name, kind, label=label)
        for name, kind in kinds.items()
    }
    required = {
        "dynamic_voc_mode": "control",
        "voc_gate_adam_beta1": 0.0,
        "voc_gate_param_align": False,
        "voc_gate_param_align_coef": 1.0,
        "voc_gate_exact_projection": True,
        "voc_gate_epsilon_greedy_execution": True,
        "voc_train_epsilon": 0.02,
        "voc_eval_stochastic": True,
        "float16": True,
        "model_float16": False,
        "ckp": False,
        "preload": "",
        "preload_actor": "",
        "voc_parent_checkpoint": "",
    }
    for name, expected in required.items():
        if identity[name] != expected:
            raise ValueError(
                f"{label} schema-5 identity requires {name}={expected!r}; "
                f"got {identity[name]!r}"
            )
    return identity


def _validate_schema11_smoke_resolved_identity(
    identity: Any, *, label: str
) -> dict[str, Any]:
    """Require the authoritative exact-12 schema-11 derived identity."""

    from thinker import util

    if type(identity) is not dict or set(identity) != (
        _SCHEMA11_RESOLVED_IDENTITY_FIELDS
    ):
        raise ValueError(f"{label} must be the exact 12-key schema-11 identity")
    for name, expected in (
        ("key_count", 229),
        ("v12_projection_key_count", 209),
        ("gate_schema", 11),
        ("voc_gate_policy_schema_version", 11),
        ("voc_model_input_seal_schema_version", 1),
    ):
        value = identity.get(name)
        if type(value) is not int or value != expected:
            raise ValueError(f"{label} requires exact integer {name}={expected}")
    projection_digest = identity.get("v12_projection_sha256")
    expected_projection_digest = (
        util.VOC_GATE_POLICY_SCHEMA6_V12_BASELINE_SHA256
    )
    if (
        type(projection_digest) is not str
        or projection_digest != expected_projection_digest
    ):
        raise ValueError(f"{label} has the wrong v12 projection digest")
    complete_digest = identity.get("complete_surface_sha256")
    if (
        type(complete_digest) is not str
        or len(complete_digest) != 64
        or any(character not in "0123456789abcdef" for character in complete_digest)
    ):
        raise ValueError(f"{label} has an invalid complete-surface digest")
    stage = identity.get("stage")
    if (
        type(stage) not in (list, tuple)
        or len(stage) != 6
        or type(stage[0]) is not str
        or any(type(value) is not int for value in stage[1:5])
        or type(stage[5]) is not bool
        or tuple(stage) not in util.VOC_GATE_POLICY_SCHEMA11_STAGE_PROFILES
    ):
        raise ValueError(f"{label} lacks an exact closed schema-11 stage")
    paths = identity.get("paths")
    expected_path_fields = {
        "savedir",
        "ckpdir",
        "cmd",
        "icopro_data_path",
    }
    if type(paths) is not dict or set(paths) != expected_path_fields:
        raise ValueError(f"{label} lacks exact schema-11 path identity")
    if any(type(value) is not str or not value for value in paths.values()):
        raise ValueError(f"{label} path identity requires exact nonempty strings")
    for name in ("savedir", "ckpdir", "icopro_data_path"):
        value = paths[name]
        if (
            not os.path.isabs(value)
            or os.path.normpath(value) != value
            or os.path.realpath(value) != value
        ):
            raise ValueError(f"{label} has invalid normalized {name}")
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
        raise ValueError(f"{label} schema-11 path relationships disagree")
    for name, expected in _SCHEMA11_DERIVED_IDENTITY.items():
        value = identity.get(name)
        if type(value) is not str or value != expected:
            raise ValueError(f"{label} requires exact {name}={expected!r}")
    return copy.deepcopy(identity)


def _validate_schema11_smoke_active_state(
    active_state: Any, *, resolved_identity: Mapping[str, Any]
) -> dict[str, str]:
    """Bind the dedicated actor-only result to its exact schema-10 delta."""

    if type(active_state) is not dict or set(active_state) != (
        _SCHEMA11_FINAL_ACTOR_EVIDENCE_FIELDS
    ):
        raise ValueError(
            "Smoke schema-11 final actor evidence must equal the exact "
            "schema-10 actor-only keyset plus optimizer coordinates"
        )
    schema = active_state.get("voc_gate_policy_schema_version")
    if type(schema) is not int or schema != 11:
        raise ValueError(
            "Smoke schema-11 final actor evidence requires exact Python "
            "integer voc_gate_policy_schema_version=11"
        )
    seal_schema = active_state.get("voc_model_input_seal_schema_version")
    if type(seal_schema) is not int or seal_schema != 1:
        raise ValueError(
            "Smoke schema-11 final actor evidence requires exact Python "
            "integer voc_model_input_seal_schema_version=1"
        )
    for name, expected in _SCHEMA11_DERIVED_IDENTITY.items():
        value = active_state.get(name)
        if (
            type(value) is not str
            or value != expected
            or value != resolved_identity.get(name)
        ):
            raise ValueError(
                "Smoke schema-11 final actor evidence disagrees with the "
                f"authoritative {name}"
            )
    return {
        name: active_state[name]
        for name in _SCHEMA11_DERIVED_IDENTITY
    }


def _validate_schema12_smoke_resolved_identity(
    identity: Any, *, label: str
) -> dict[str, Any]:
    """Require the authoritative exact-12 schema-12 derived identity."""

    from thinker import util

    if type(identity) is not dict or set(identity) != (
        _SCHEMA12_RESOLVED_IDENTITY_FIELDS
    ):
        raise ValueError(f"{label} must be the exact 12-key schema-12 identity")
    for name, expected in (
        ("key_count", 229),
        ("v12_projection_key_count", 209),
        ("gate_schema", 12),
        ("voc_gate_policy_schema_version", 12),
        ("voc_model_input_seal_schema_version", 1),
    ):
        value = identity.get(name)
        if type(value) is not int or value != expected:
            raise ValueError(f"{label} requires exact integer {name}={expected}")
    projection_digest = identity.get("v12_projection_sha256")
    if (
        type(projection_digest) is not str
        or projection_digest
        != "ad22b91fdd06a30ac7f53c0135b32fac2530687c3c36dad5dccf06d700f83f82"
    ):
        raise ValueError(f"{label} has the wrong v12 projection digest")
    complete_digest = identity.get("complete_surface_sha256")
    if (
        type(complete_digest) is not str
        or len(complete_digest) != 64
        or any(character not in "0123456789abcdef" for character in complete_digest)
    ):
        raise ValueError(f"{label} has an invalid complete-surface digest")
    stage = identity.get("stage")
    if (
        type(stage) not in (list, tuple)
        or len(stage) != 6
        or type(stage[0]) is not str
        or any(type(value) is not int for value in stage[1:5])
        or type(stage[5]) is not bool
        or tuple(stage) not in util.VOC_GATE_POLICY_SCHEMA12_STAGE_PROFILES
    ):
        raise ValueError(f"{label} lacks an exact closed schema-12 stage")
    paths = identity.get("paths")
    expected_path_fields = {"savedir", "ckpdir", "cmd", "icopro_data_path"}
    if type(paths) is not dict or set(paths) != expected_path_fields:
        raise ValueError(f"{label} lacks exact schema-12 path identity")
    if any(type(value) is not str or not value for value in paths.values()):
        raise ValueError(f"{label} path identity requires exact nonempty strings")
    for name in ("savedir", "ckpdir", "icopro_data_path"):
        value = paths[name]
        if (
            not os.path.isabs(value)
            or os.path.normpath(value) != value
            or os.path.realpath(value) != value
        ):
            raise ValueError(f"{label} has invalid normalized {name}")
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
        raise ValueError(f"{label} schema-12 path relationships disagree")
    for name, expected in _SCHEMA12_DERIVED_IDENTITY.items():
        value = identity.get(name)
        if type(value) is not str or value != expected:
            raise ValueError(f"{label} requires exact {name}={expected!r}")
    return copy.deepcopy(identity)


def _validate_schema12_smoke_active_state(
    active_state: Any, *, resolved_identity: Mapping[str, Any]
) -> dict[str, str]:
    """Bind schema-12 actor-only evidence to the inherited exact keyset."""

    if type(active_state) is not dict or set(active_state) != (
        _SCHEMA12_FINAL_ACTOR_EVIDENCE_FIELDS
    ):
        raise ValueError(
            "Smoke schema-12 final actor evidence must preserve the exact "
            "schema-11 actor-only keyset"
        )
    schema = active_state.get("voc_gate_policy_schema_version")
    if type(schema) is not int or schema != 12:
        raise ValueError(
            "Smoke schema-12 final actor evidence requires exact Python "
            "integer voc_gate_policy_schema_version=12"
        )
    seal_schema = active_state.get("voc_model_input_seal_schema_version")
    if type(seal_schema) is not int or seal_schema != 1:
        raise ValueError(
            "Smoke schema-12 final actor evidence requires exact Python "
            "integer voc_model_input_seal_schema_version=1"
        )
    tau = active_state.get("voc_gate_target_tau")
    if type(tau) is not float or tau != 1.0:
        raise ValueError(
            "Smoke schema-12 final actor evidence requires exact tau 1.0"
        )
    for name, expected in _SCHEMA12_DERIVED_IDENTITY.items():
        value = active_state.get(name)
        if (
            type(value) is not str
            or value != expected
            or value != resolved_identity.get(name)
        ):
            raise ValueError(
                "Smoke schema-12 final actor evidence disagrees with the "
                f"authoritative {name}"
            )
    return {name: active_state[name] for name in _SCHEMA12_DERIVED_IDENTITY}


def _require_schema12_smoke_ema_online_equality(
    checkpoint_dir: Path,
    completion_state: Mapping[str, Any],
) -> None:
    """Independently bind tau-one raw EMA equality before smoke downstream use."""

    import evaluate_dynamic_imitation as checkpoint_eval

    checkpoint_files = completion_state.get("checkpoint_files")
    actor_record = (
        checkpoint_files.get("ckp_actor.tar")
        if isinstance(checkpoint_files, Mapping)
        else None
    )
    if not isinstance(actor_record, Mapping):
        raise ValueError("Smoke schema-12 completion lacks actor checkpoint evidence")
    expected_sha = actor_record.get("sha256")
    expected_size = actor_record.get("size")
    if type(expected_sha) is not str or type(expected_size) is not int:
        raise ValueError("Smoke schema-12 actor checkpoint evidence is malformed")
    payload = checkpoint_eval._read_stable_single_link_bytes(
        checkpoint_dir / "ckp_actor.tar",
        label="schema-12 smoke actor checkpoint",
    )
    if len(payload) != expected_size or hashlib.sha256(payload).hexdigest() != expected_sha:
        raise RuntimeError("schema-12 smoke actor checkpoint disagrees with completion")
    checkpoint = torch.load(
        io.BytesIO(payload), map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Smoke schema-12 actor checkpoint must be a mapping")
    tau = checkpoint.get("voc_gate_target_tau")
    if type(tau) is not float or tau != 1.0:
        raise ValueError("Smoke schema-12 actor checkpoint requires exact tau 1.0")
    update_count = checkpoint.get("voc_ema_gate_update_count")
    if type(update_count) is not int or update_count < 0:
        raise ValueError("Smoke schema-12 gate update count must be an integer >= 0")
    if update_count == 0:
        return
    ema = checkpoint.get("voc_ema_gate_head_state_dict")
    online = checkpoint.get("actor_net_state_dict")
    if not isinstance(ema, Mapping) or set(ema) != {"weight", "bias"}:
        raise ValueError("Smoke schema-12 EMA head must contain exact weight and bias")
    if not isinstance(online, Mapping):
        raise ValueError("Smoke schema-12 online actor state must be a mapping")
    prefix = "voc_head" if "voc_head.weight" in online else "critic.voc_head"
    for name in ("weight", "bias"):
        ema_tensor = ema.get(name)
        online_tensor = online.get(f"{prefix}.{name}")
        if not torch.is_tensor(ema_tensor) or not torch.is_tensor(online_tensor):
            raise ValueError("Smoke schema-12 EMA/online equality requires tensors")
        if not torch.equal(ema_tensor, online_tensor):
            raise ValueError(
                f"Smoke schema-12 raw EMA {name} must equal online raw Q {name}"
            )


def _validate_schema13_smoke_resolved_identity(
    identity: Any, *, label: str
) -> dict[str, Any]:
    """Require the authoritative exact-12 schema-13 derived identity."""

    from thinker import util

    if type(identity) is not dict or set(identity) != (
        _SCHEMA13_RESOLVED_IDENTITY_FIELDS
    ):
        raise ValueError(f"{label} must be the exact 12-key schema-13 identity")
    for name, expected in (
        ("key_count", 229),
        ("v12_projection_key_count", 209),
        ("gate_schema", 13),
        ("voc_gate_policy_schema_version", 13),
        ("voc_model_input_seal_schema_version", 1),
    ):
        value = identity.get(name)
        if type(value) is not int or value != expected:
            raise ValueError(f"{label} requires exact integer {name}={expected}")
    if identity.get("v12_projection_sha256") != (
        "ad22b91fdd06a30ac7f53c0135b32fac2530687c3c36dad5dccf06d700f83f82"
    ) or type(identity.get("v12_projection_sha256")) is not str:
        raise ValueError(f"{label} has the wrong v12 projection digest")
    complete_digest = identity.get("complete_surface_sha256")
    if (
        type(complete_digest) is not str
        or len(complete_digest) != 64
        or any(character not in "0123456789abcdef" for character in complete_digest)
    ):
        raise ValueError(f"{label} has an invalid complete-surface digest")
    stage = identity.get("stage")
    if (
        type(stage) not in (list, tuple)
        or len(stage) != 6
        or type(stage[0]) is not str
        or any(type(value) is not int for value in stage[1:5])
        or type(stage[5]) is not bool
        or tuple(stage) not in util.VOC_GATE_POLICY_SCHEMA13_STAGE_PROFILES
    ):
        raise ValueError(f"{label} lacks an exact closed schema-13 stage")
    paths = identity.get("paths")
    expected_path_fields = {"savedir", "ckpdir", "cmd", "icopro_data_path"}
    if type(paths) is not dict or set(paths) != expected_path_fields:
        raise ValueError(f"{label} lacks exact schema-13 path identity")
    if any(type(value) is not str or not value for value in paths.values()):
        raise ValueError(f"{label} path identity requires exact nonempty strings")
    for name in ("savedir", "ckpdir", "icopro_data_path"):
        value = paths[name]
        if (
            not os.path.isabs(value)
            or os.path.normpath(value) != value
            or os.path.realpath(value) != value
        ):
            raise ValueError(f"{label} has invalid normalized {name}")
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
        raise ValueError(f"{label} schema-13 path relationships disagree")
    for name, expected in _SCHEMA13_DERIVED_IDENTITY.items():
        value = identity.get(name)
        if type(value) is not str or value != expected:
            raise ValueError(f"{label} requires exact {name}={expected!r}")
    return copy.deepcopy(identity)


def _validate_schema13_smoke_active_state(
    active_state: Any, *, resolved_identity: Mapping[str, Any]
) -> dict[str, str]:
    """Bind schema-13 actor-only evidence to the inherited exact keyset."""

    if type(active_state) is not dict or set(active_state) != (
        _SCHEMA13_FINAL_ACTOR_EVIDENCE_FIELDS
    ):
        raise ValueError(
            "Smoke schema-13 final actor evidence must preserve the exact "
            "schema-12 actor-only keyset"
        )
    schema = active_state.get("voc_gate_policy_schema_version")
    if type(schema) is not int or schema != 13:
        raise ValueError(
            "Smoke schema-13 final actor evidence requires exact Python "
            "integer voc_gate_policy_schema_version=13"
        )
    seal_schema = active_state.get("voc_model_input_seal_schema_version")
    if type(seal_schema) is not int or seal_schema != 1:
        raise ValueError(
            "Smoke schema-13 final actor evidence requires exact Python "
            "integer voc_model_input_seal_schema_version=1"
        )
    tau = active_state.get("voc_gate_target_tau")
    if type(tau) is not float or tau != 1.0:
        raise ValueError("Smoke schema-13 final actor evidence requires exact tau 1.0")
    for name, expected in _SCHEMA13_DERIVED_IDENTITY.items():
        value = active_state.get(name)
        if (
            type(value) is not str
            or value != expected
            or value != resolved_identity.get(name)
        ):
            raise ValueError(
                "Smoke schema-13 final actor evidence disagrees with the "
                f"authoritative {name}"
            )
    return {name: active_state[name] for name in _SCHEMA13_DERIVED_IDENTITY}


def _require_schema13_smoke_ema_online_equality(
    checkpoint_dir: Path,
    completion_state: Mapping[str, Any],
) -> None:
    """Independently bind tau-one raw EMA equality for schema-13 smoke."""

    import evaluate_dynamic_imitation as checkpoint_eval

    checkpoint_files = completion_state.get("checkpoint_files")
    actor_record = (
        checkpoint_files.get("ckp_actor.tar")
        if isinstance(checkpoint_files, Mapping)
        else None
    )
    if not isinstance(actor_record, Mapping):
        raise ValueError("Smoke schema-13 completion lacks actor checkpoint evidence")
    expected_sha = actor_record.get("sha256")
    expected_size = actor_record.get("size")
    if type(expected_sha) is not str or type(expected_size) is not int:
        raise ValueError("Smoke schema-13 actor checkpoint evidence is malformed")
    payload = checkpoint_eval._read_stable_single_link_bytes(
        checkpoint_dir / "ckp_actor.tar",
        label="schema-13 smoke actor checkpoint",
    )
    if len(payload) != expected_size or hashlib.sha256(payload).hexdigest() != expected_sha:
        raise RuntimeError("schema-13 smoke actor checkpoint disagrees with completion")
    checkpoint = torch.load(
        io.BytesIO(payload), map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Smoke schema-13 actor checkpoint must be a mapping")
    if type(checkpoint.get("voc_gate_target_tau")) is not float or (
        checkpoint["voc_gate_target_tau"] != 1.0
    ):
        raise ValueError("Smoke schema-13 actor checkpoint requires exact tau 1.0")
    update_count = checkpoint.get("voc_ema_gate_update_count")
    if type(update_count) is not int or update_count < 0:
        raise ValueError("Smoke schema-13 gate update count must be an integer >= 0")
    if update_count == 0:
        return
    ema = checkpoint.get("voc_ema_gate_head_state_dict")
    online = checkpoint.get("actor_net_state_dict")
    if not isinstance(ema, Mapping) or set(ema) != {"weight", "bias"}:
        raise ValueError("Smoke schema-13 EMA head must contain exact weight and bias")
    if not isinstance(online, Mapping):
        raise ValueError("Smoke schema-13 online actor state must be a mapping")
    prefix = "voc_head" if "voc_head.weight" in online else "critic.voc_head"
    for name in ("weight", "bias"):
        ema_tensor = ema.get(name)
        online_tensor = online.get(f"{prefix}.{name}")
        if not torch.is_tensor(ema_tensor) or not torch.is_tensor(online_tensor):
            raise ValueError("Smoke schema-13 EMA/online equality requires tensors")
        if not torch.equal(ema_tensor, online_tensor):
            raise ValueError(
                f"Smoke schema-13 raw EMA {name} must equal online raw Q {name}"
            )


def _validate_smoke_checkpoint_metadata(
    actor_checkpoint: Mapping[str, Any],
    model_checkpoint: Mapping[str, Any],
    flags: Any,
    spec: Any,
    active_state: Mapping[str, Any],
    schema6_bundle_validation: Optional[Mapping[str, Any]] = None,
    schema7_bundle_validation: Optional[Mapping[str, Any]] = None,
    schema8_bundle_validation: Optional[Mapping[str, Any]] = None,
    schema9_bundle_validation: Optional[Mapping[str, Any]] = None,
    schema10_bundle_validation: Optional[Mapping[str, Any]] = None,
    schema11_bundle_validation: Optional[Mapping[str, Any]] = None,
    schema12_bundle_validation: Optional[Mapping[str, Any]] = None,
    schema13_bundle_validation: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Validate and resolve actor/ModelNet metadata before smoke execution."""

    import evaluate_dynamic_imitation as checkpoint_eval
    from thinker import util

    actor_public = checkpoint_eval.validate_actor_imitation_checkpoint(
        actor_checkpoint, flags, spec
    )
    model_public = checkpoint_eval.validate_model_checkpoint(
        model_checkpoint, flags, spec
    )
    actor_embedded = actor_checkpoint.get("flags")
    model_embedded = model_checkpoint.get("flags")
    if not isinstance(actor_embedded, Mapping):
        raise ValueError("Smoke actor checkpoint lacks embedded training flags")
    if not isinstance(model_embedded, Mapping):
        raise ValueError("Smoke ModelNet checkpoint lacks embedded training flags")

    mode = active_state.get("dynamic_voc_mode")
    model_gate_state = None
    resolved_identity = None
    if mode != "off":
        actor_schema = active_state.get("voc_gate_policy_schema_version")
        schema13_version = getattr(
            util, "VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION", 13
        )
        if actor_schema == schema13_version and type(actor_schema) is not int:
            raise ValueError(
                "Smoke schema-13 actor evidence requires exact built-in int 13"
            )
        model_gate_state = checkpoint_eval._validate_voc_gate_policy_schema(
            model_checkpoint,
            model_embedded,
            label="Smoke ModelNet checkpoint",
        )
        for name in (
            "voc_gate_policy_schema_version",
            "voc_gate_adam_beta1",
            "voc_gate_param_align",
            "voc_gate_param_align_coef",
            "voc_gate_exact_projection",
            "voc_gate_epsilon_greedy_execution",
            "voc_model_input_seal_schema_version",
        ):
            default = 0 if name == "voc_model_input_seal_schema_version" else None
            model_value = model_gate_state.get(name, default)
            actor_value = active_state.get(name, default)
            if model_value != actor_value:
                raise ValueError(
                    "Smoke actor and ModelNet checkpoint metadata disagree on "
                    f"{name}: {actor_value!r} versus {model_value!r}"
                )

        if (
            int(actor_schema)
            == util.VOC_GATE_POLICY_EPSILON_GREEDY_EXECUTION_SCHEMA_VERSION
        ):
            resolved_identity = {
                "config": _schema5_checkpoint_identity(
                    flags, label="Smoke config_c.yaml"
                ),
                "actor_checkpoint": _schema5_checkpoint_identity(
                    actor_embedded,
                    label="Smoke actor checkpoint embedded flags",
                ),
                "model_checkpoint": _schema5_checkpoint_identity(
                    model_embedded,
                    label="Smoke ModelNet checkpoint embedded flags",
                ),
            }
            for values in resolved_identity.values():
                values["voc_gate_policy_schema_version"] = int(actor_schema)
            configured = resolved_identity["config"]
            for source in ("actor_checkpoint", "model_checkpoint"):
                for name, expected in configured.items():
                    actual = resolved_identity[source][name]
                    if actual != expected:
                        raise ValueError(
                            f"Smoke {source.replace('_', ' ')} schema-5 "
                            f"identity disagrees with config_c.yaml on {name}: "
                            f"{actual!r} versus {expected!r}"
                        )
        elif (
            int(actor_schema)
            == util.VOC_GATE_POLICY_VERSION_BARRIER_SCHEMA_VERSION
        ):
            if not isinstance(schema6_bundle_validation, Mapping):
                raise ValueError(
                    "Smoke schema-6 checkpoint lacks completed-bundle validation"
                )
            stored_identity = schema6_bundle_validation.get(
                "stored_surface_identity"
            )
            if (
                not isinstance(stored_identity, Mapping)
                or set(stored_identity)
                != {"config", "actor_checkpoint", "model_checkpoint"}
            ):
                raise ValueError(
                    "Smoke schema-6 completed bundle lacks three-surface identity"
            )
            resolved_identity = copy.deepcopy(dict(stored_identity))
        elif (
            int(actor_schema)
            == util.VOC_GATE_POLICY_MODEL_INPUT_SEAL_SCHEMA_VERSION
        ):
            if not isinstance(schema7_bundle_validation, Mapping):
                raise ValueError(
                    "Smoke schema-7 checkpoint lacks completed-bundle validation"
                )
            stored_identity = schema7_bundle_validation.get(
                "stored_surface_identity"
            )
            if (
                not isinstance(stored_identity, Mapping)
                or set(stored_identity)
                != {"config", "actor_checkpoint", "model_checkpoint"}
            ):
                raise ValueError(
                    "Smoke schema-7 completed bundle lacks three-surface identity"
                )
            resolved_identity = copy.deepcopy(dict(stored_identity))
        elif (
            int(actor_schema)
            == util.VOC_GATE_POLICY_HALF_SQUARED_Q_SCHEMA_VERSION
        ):
            if not isinstance(schema8_bundle_validation, Mapping):
                raise ValueError(
                    "Smoke schema-8 checkpoint lacks completed-bundle validation"
                )
            stored_identity = schema8_bundle_validation.get(
                "stored_surface_identity"
            )
            if (
                not isinstance(stored_identity, Mapping)
                or set(stored_identity)
                != {"config", "actor_checkpoint", "model_checkpoint"}
                or any(
                    not isinstance(identity, Mapping)
                    or identity.get("voc_q_regression_loss")
                    != "half_squared_td"
                    for identity in stored_identity.values()
                )
            ):
                raise ValueError(
                    "Smoke schema-8 completed bundle lacks exact three-surface "
                    "half-squared identity"
                )
            resolved_identity = copy.deepcopy(dict(stored_identity))
        elif (
            int(actor_schema)
            == util.VOC_GATE_POLICY_COMMON_MODE_Q_SCHEMA_VERSION
        ):
            if not isinstance(schema9_bundle_validation, Mapping):
                raise ValueError(
                    "Smoke schema-9 checkpoint lacks completed-bundle validation"
                )
            stored_identity = schema9_bundle_validation.get(
                "stored_surface_identity"
            )
            if (
                not isinstance(stored_identity, Mapping)
                or set(stored_identity)
                != {"config", "actor_checkpoint", "model_checkpoint"}
                or any(
                    not isinstance(identity, Mapping)
                    or set(identity)
                    != {
                        "gate_schema",
                        "voc_gate_policy_schema_version",
                        "voc_model_input_seal_schema_version",
                        "voc_q_regression_loss",
                        "voc_q_reconstruction",
                        "key_count",
                        "v12_projection_key_count",
                        "v12_projection_sha256",
                        "complete_surface_sha256",
                        "stage",
                        "paths",
                    }
                    or identity.get("voc_q_regression_loss")
                    != "half_squared_td"
                    or identity.get("voc_q_reconstruction")
                    != (
                        "detached_value_plus_raw_head_mean_plus_"
                        "policy_centered_raw_head"
                    )
                    for identity in stored_identity.values()
                )
            ):
                raise ValueError(
                    "Smoke schema-9 completed bundle lacks exact three-surface "
                    "common-mode identity"
                )
            resolved_identity = copy.deepcopy(dict(stored_identity))
        elif int(actor_schema) == getattr(
            util, "VOC_GATE_POLICY_HUBER_COMMON_Q_SCHEMA_VERSION", 10
        ):
            if not isinstance(schema10_bundle_validation, Mapping):
                raise ValueError(
                    "Smoke schema-10 checkpoint lacks completed-bundle validation"
                )
            stored_identity = schema10_bundle_validation.get(
                "stored_surface_identity"
            )
            if (
                not isinstance(stored_identity, Mapping)
                or set(stored_identity)
                != {"config", "actor_checkpoint", "model_checkpoint"}
                or any(
                    not isinstance(identity, Mapping)
                    or set(identity)
                    != {
                        "gate_schema",
                        "voc_gate_policy_schema_version",
                        "voc_model_input_seal_schema_version",
                        "voc_q_regression_loss",
                        "voc_q_reconstruction",
                        "key_count",
                        "v12_projection_key_count",
                        "v12_projection_sha256",
                        "complete_surface_sha256",
                        "stage",
                        "paths",
                    }
                    or identity.get("voc_q_regression_loss")
                    != "smooth_l1_beta1"
                    or identity.get("voc_q_reconstruction")
                    != (
                        "detached_value_plus_raw_head_mean_plus_"
                        "policy_centered_raw_head"
                    )
                    for identity in stored_identity.values()
                )
            ):
                raise ValueError(
                    "Smoke schema-10 completed bundle lacks exact three-surface "
                    "Huber-common identity"
                )
            resolved_identity = copy.deepcopy(dict(stored_identity))
        elif int(actor_schema) == getattr(
            util, "VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION", 11
        ):
            if (
                type(schema11_bundle_validation) is not dict
                or set(schema11_bundle_validation)
                != _SCHEMA11_COMPLETED_BUNDLE_FIELDS
            ):
                raise ValueError(
                    "Smoke schema-11 checkpoint requires the exact completed-"
                    "bundle container"
                )
            if schema11_bundle_validation.get("authoritative_validator") != (
                "thinker.util.validate_schema11_final_bundle"
            ):
                raise ValueError(
                    "Smoke schema-11 completed-bundle validator identity "
                    "disagrees"
                )
            lifecycle_actor_policy = schema11_bundle_validation.get(
                "actor_policy"
            )
            expected_actor_policy_fields = getattr(
                checkpoint_eval,
                "ATOMIC_ACTOR_POLICY_EVIDENCE_FIELDS",
                None,
            )
            if (
                type(lifecycle_actor_policy) is not dict
                or not isinstance(expected_actor_policy_fields, frozenset)
                or set(lifecycle_actor_policy)
                != expected_actor_policy_fields
            ):
                raise ValueError(
                    "Smoke schema-11 lifecycle actor-policy evidence must "
                    "preserve the exact schema-10 keyset"
                )
            authoritative_identity = _validate_schema11_smoke_resolved_identity(
                schema11_bundle_validation.get("resolved_identity"),
                label="Smoke schema-11 authoritative resolved identity",
            )
            _validate_schema11_smoke_active_state(
                active_state,
                resolved_identity=authoritative_identity,
            )
            for name, expected in (
                ("voc_gate_policy_schema_version", 11),
                ("voc_model_input_seal_schema_version", 1),
            ):
                model_value = model_gate_state.get(name)
                if type(model_value) is not int or model_value != expected:
                    raise ValueError(
                        "Smoke schema-11 ModelNet evidence requires exact "
                        f"Python integer {name}={expected}"
                    )
            stored_identity = schema11_bundle_validation.get(
                "stored_surface_identity"
            )
            if (
                type(stored_identity) is not dict
                or set(stored_identity)
                != {"config", "actor_checkpoint", "model_checkpoint"}
                or any(
                    type(identity) is not dict
                    or identity != authoritative_identity
                    for identity in stored_identity.values()
                )
            ):
                raise ValueError(
                    "Smoke schema-11 completed bundle stored-surface identity "
                    "must be three exact copies of authoritative resolved identity"
                )
            resolved_identity = copy.deepcopy(dict(stored_identity))
        elif type(actor_schema) is int and actor_schema == getattr(
            util, "VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION", 12
        ):
            if (
                type(schema12_bundle_validation) is not dict
                or set(schema12_bundle_validation)
                != _SCHEMA12_COMPLETED_BUNDLE_FIELDS
            ):
                raise ValueError(
                    "Smoke schema-12 checkpoint requires the exact completed-"
                    "bundle container"
                )
            if schema12_bundle_validation.get("authoritative_validator") != (
                "thinker.util.validate_schema12_final_bundle"
            ):
                raise ValueError(
                    "Smoke schema-12 completed-bundle validator identity disagrees"
                )
            lifecycle_actor_policy = schema12_bundle_validation.get(
                "actor_policy"
            )
            expected_actor_policy_fields = getattr(
                checkpoint_eval,
                "ATOMIC_ACTOR_POLICY_EVIDENCE_FIELDS",
                None,
            )
            if (
                type(lifecycle_actor_policy) is not dict
                or not isinstance(expected_actor_policy_fields, frozenset)
                or set(lifecycle_actor_policy) != expected_actor_policy_fields
            ):
                raise ValueError(
                    "Smoke schema-12 lifecycle actor-policy evidence must "
                    "preserve the exact schema-11 keyset"
                )
            authoritative_identity = _validate_schema12_smoke_resolved_identity(
                schema12_bundle_validation.get("resolved_identity"),
                label="Smoke schema-12 authoritative resolved identity",
            )
            _validate_schema12_smoke_active_state(
                active_state,
                resolved_identity=authoritative_identity,
            )
            for name, expected in (
                ("voc_gate_policy_schema_version", 12),
                ("voc_model_input_seal_schema_version", 1),
            ):
                model_value = model_gate_state.get(name)
                if type(model_value) is not int or model_value != expected:
                    raise ValueError(
                        "Smoke schema-12 ModelNet evidence requires exact "
                        f"Python integer {name}={expected}"
                    )
            model_tau = model_gate_state.get("voc_gate_target_tau")
            if type(model_tau) is not float or model_tau != 1.0:
                raise ValueError(
                    "Smoke schema-12 ModelNet evidence requires exact tau 1.0"
                )
            stored_identity = schema12_bundle_validation.get(
                "stored_surface_identity"
            )
            if (
                type(stored_identity) is not dict
                or set(stored_identity)
                != {"config", "actor_checkpoint", "model_checkpoint"}
                or any(
                    type(identity) is not dict
                    or identity != authoritative_identity
                    for identity in stored_identity.values()
                )
            ):
                raise ValueError(
                    "Smoke schema-12 completed bundle stored-surface identity "
                    "must be three exact copies of authoritative resolved identity"
                )
            resolved_identity = copy.deepcopy(dict(stored_identity))
        elif type(actor_schema) is int and actor_schema == getattr(
            util, "VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION", 13
        ):
            if (
                type(schema13_bundle_validation) is not dict
                or set(schema13_bundle_validation)
                != _SCHEMA13_COMPLETED_BUNDLE_FIELDS
            ):
                raise ValueError(
                    "Smoke schema-13 checkpoint requires the exact completed-"
                    "bundle container"
                )
            if schema13_bundle_validation.get("authoritative_validator") != (
                "thinker.util.validate_schema13_final_bundle"
            ):
                raise ValueError(
                    "Smoke schema-13 completed-bundle validator identity disagrees"
                )
            lifecycle_actor_policy = schema13_bundle_validation.get("actor_policy")
            expected_actor_policy_fields = getattr(
                checkpoint_eval, "ATOMIC_ACTOR_POLICY_EVIDENCE_FIELDS", None
            )
            if (
                type(lifecycle_actor_policy) is not dict
                or not isinstance(expected_actor_policy_fields, frozenset)
                or set(lifecycle_actor_policy) != expected_actor_policy_fields
            ):
                raise ValueError(
                    "Smoke schema-13 lifecycle actor-policy evidence must "
                    "preserve the exact schema-12 keyset"
                )
            authoritative_identity = _validate_schema13_smoke_resolved_identity(
                schema13_bundle_validation.get("resolved_identity"),
                label="Smoke schema-13 authoritative resolved identity",
            )
            _validate_schema13_smoke_active_state(
                active_state,
                resolved_identity=authoritative_identity,
            )
            for name, expected in (
                ("voc_gate_policy_schema_version", 13),
                ("voc_model_input_seal_schema_version", 1),
            ):
                model_value = model_gate_state.get(name)
                if type(model_value) is not int or model_value != expected:
                    raise ValueError(
                        "Smoke schema-13 ModelNet evidence requires exact "
                        f"Python integer {name}={expected}"
                    )
            model_tau = model_gate_state.get("voc_gate_target_tau")
            if type(model_tau) is not float or model_tau != 1.0:
                raise ValueError(
                    "Smoke schema-13 ModelNet evidence requires exact tau 1.0"
                )
            telemetry = schema13_bundle_validation.get("telemetry")
            if (
                type(telemetry) is not dict
                or set(telemetry) != _SCHEMA13_TELEMETRY_EVIDENCE_FIELDS
                or type(telemetry.get("telemetry_schema_version")) is not int
                or telemetry["telemetry_schema_version"] != 1
                or type(telemetry.get("gate_schema")) is not int
                or telemetry["gate_schema"] != 13
                or type(telemetry.get("manifest_name")) is not str
                or telemetry["manifest_name"] != "voc_telemetry_manifest.json"
                or type(telemetry.get("manifest_size")) is not int
                or telemetry["manifest_size"] <= 0
                or type(telemetry.get("transaction_count")) is not int
                or telemetry["transaction_count"] <= 0
                or type(telemetry.get("terminal_policy_version")) is not int
                or telemetry["terminal_policy_version"]
                != lifecycle_actor_policy.get("voc_actor_policy_version")
                or type(telemetry.get("terminal_real_step")) is not int
                or telemetry["terminal_real_step"] < authoritative_identity["stage"][2]
            ):
                raise ValueError("Smoke schema-13 telemetry evidence is malformed")
            for name in (
                "manifest_sha256",
                "actor_state_sha256",
                "publication_history_sha256",
            ):
                value = telemetry.get(name)
                if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                    raise ValueError(
                        f"Smoke schema-13 telemetry {name} is malformed"
                    )
            if (
                telemetry["actor_state_sha256"]
                != lifecycle_actor_policy.get("voc_actor_policy_state_sha256")
                or telemetry["publication_history_sha256"]
                != lifecycle_actor_policy.get(
                    "voc_actor_policy_publication_history_sha256"
                )
            ):
                raise ValueError(
                    "Smoke schema-13 telemetry disagrees with actor evidence"
                )
            completion_evidence = schema13_bundle_validation.get(
                "completion_evidence"
            )
            checkpoint_files = (
                completion_evidence.get("checkpoint_files")
                if type(completion_evidence) is dict
                else None
            )
            manifest_record = (
                checkpoint_files.get("voc_telemetry_manifest.json")
                if type(checkpoint_files) is dict
                else None
            )
            if (
                type(manifest_record) is not dict
                or set(manifest_record) != {"sha256", "size"}
                or telemetry["manifest_sha256"] != manifest_record["sha256"]
                or telemetry["manifest_size"] != manifest_record["size"]
                or telemetry["transaction_count"]
                != active_state.get("voc_update_count")
            ):
                raise ValueError(
                    "Smoke schema-13 telemetry disagrees with completion/actor state"
                )
            stored_identity = schema13_bundle_validation.get(
                "stored_surface_identity"
            )
            if (
                type(stored_identity) is not dict
                or set(stored_identity)
                != {"config", "actor_checkpoint", "model_checkpoint"}
                or any(
                    type(identity) is not dict
                    or identity != authoritative_identity
                    for identity in stored_identity.values()
                )
            ):
                raise ValueError(
                    "Smoke schema-13 completed bundle stored-surface identity "
                    "must be three exact copies of authoritative resolved identity"
                )
            resolved_identity = copy.deepcopy(dict(stored_identity))

    return {
        "actor_public_validation": actor_public,
        "model_public_validation": model_public,
        "model_gate_policy": model_gate_state,
        "resolved_identity": resolved_identity,
        "schema6_completed_bundle_validation": (
            copy.deepcopy(schema6_bundle_validation)
            if schema6_bundle_validation is not None
            else None
        ),
        **(
            {
                "schema7_completed_bundle_validation": copy.deepcopy(
                    schema7_bundle_validation
                )
            }
            if schema7_bundle_validation is not None
            else {}
        ),
        **(
            {
                "schema8_completed_bundle_validation": copy.deepcopy(
                    schema8_bundle_validation
                )
            }
            if schema8_bundle_validation is not None
            else {}
        ),
        **(
            {
                "schema9_completed_bundle_validation": copy.deepcopy(
                    schema9_bundle_validation
                )
            }
            if schema9_bundle_validation is not None
            else {}
        ),
        **(
            {
                "schema10_completed_bundle_validation": copy.deepcopy(
                    schema10_bundle_validation
                )
            }
            if schema10_bundle_validation is not None
            else {}
        ),
        **(
            {
                "schema11_completed_bundle_validation": copy.deepcopy(
                    schema11_bundle_validation
                )
            }
            if schema11_bundle_validation is not None
            else {}
        ),
        **(
            {
                "schema12_completed_bundle_validation": copy.deepcopy(
                    schema12_bundle_validation
                )
            }
            if schema12_bundle_validation is not None
            else {}
        ),
        **(
            {
                "schema13_completed_bundle_validation": copy.deepcopy(
                    schema13_bundle_validation
                )
            }
            if schema13_bundle_validation is not None
            else {}
        ),
    }


def _vector_actor_observation_space(template: spaces.Dict, batch_size: int):
    if not isinstance(template, spaces.Dict):
        raise TypeError("cModelWrapper observation_space must be spaces.Dict")
    vector_spaces = dict(template.spaces)
    for key in ("real_states", "xs"):
        space = vector_spaces.get(key)
        if space is not None:
            vector_spaces[key] = spaces.Box(
                low=np.broadcast_to(space.low, (batch_size,) + tuple(space.shape)),
                high=np.broadcast_to(space.high, (batch_size,) + tuple(space.shape)),
                dtype=space.dtype,
            )
    for key in ("tree_reps", "hs"):
        space = vector_spaces.get(key)
        if space is not None and int(space.shape[0]) != batch_size:
            raise ValueError(
                f"cModelWrapper {key} batch axis is {space.shape[0]}, "
                f"expected {batch_size}"
            )
    return spaces.Dict(vector_spaces)


def _load_flags(
    args: argparse.Namespace,
    *,
    config_payload: Optional[bytes] = None,
    expected_sha256: Optional[str] = None,
    schema13_bound: bool = False,
):
    from thinker import util

    if type(schema13_bound) is not bool:
        raise TypeError("schema13_bound must be an exact Python bool")
    private_config_dir = None
    flags = None
    if config_payload is not None:
        if type(config_payload) is not bytes:
            raise TypeError("validated smoke config payload must be exact bytes")
        if (
            type(expected_sha256) is not str
            or len(expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha256)
            or hashlib.sha256(config_payload).hexdigest() != expected_sha256
        ):
            raise ValueError("validated smoke config digest disagrees")
        if schema13_bound:
            if args.checkpoint_dir is None:
                raise ValueError(
                    "schema-13 bound smoke flags require a checkpoint directory"
                )
            if getattr(args, "config", None) is not None:
                raise ValueError(
                    "schema-13 forbids explicit user --config indirection"
                )
            import evaluate_dynamic_imitation as checkpoint_eval

            flags = checkpoint_eval._load_flags_from_validated_config_bytes(
                args.checkpoint_dir,
                config_payload,
                expected_sha256,
            )
            config = None
        else:
            private_config_dir = tempfile.TemporaryDirectory(
                prefix="thinker-smoke-config-"
            )
            config = Path(private_config_dir.name) / "config_c.yaml"
            descriptor = os.open(
                config,
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
    else:
        if schema13_bound:
            raise ValueError("schema-13 bound smoke flags require config bytes")
        config = args.config
        if args.checkpoint_dir is not None and config is None:
            candidate = args.checkpoint_dir / "config_c.yaml"
            if not candidate.is_file():
                raise FileNotFoundError(f"missing checkpoint config: {candidate}")
            config = candidate

    fresh = False if schema13_bound else config is None
    overrides = {
        "config": None if config is None else str(config),
        "name": args.env_name if fresh else None,
        "dynamic_search": True if fresh else None,
        "dynamic_factorized_control": True if fresh else None,
        "dynamic_voc_mode": (
            getattr(args, "dynamic_voc_mode", None)
            if getattr(args, "dynamic_voc_mode", None) is not None
            else ("off" if fresh else None)
        ),
        "voc_loss_cost": (
            getattr(args, "voc_loss_cost", None)
            if getattr(args, "voc_loss_cost", None) is not None
            else (1.0 if fresh else None)
        ),
        "voc_gate_temperature": (
            getattr(args, "voc_gate_temperature", None)
            if getattr(args, "voc_gate_temperature", None) is not None
            else (1.0 if fresh else None)
        ),
        "voc_train_epsilon": (
            getattr(args, "voc_train_epsilon", None)
            if getattr(args, "voc_train_epsilon", None) is not None
            else (0.02 if fresh else None)
        ),
        "voc_eval_stochastic": (
            getattr(args, "voc_eval_stochastic", None)
            if getattr(args, "voc_eval_stochastic", None) is not None
            else (True if fresh else None)
        ),
        "voc_dueling_q": (
            getattr(args, "voc_dueling_q", None)
            if getattr(args, "voc_dueling_q", None) is not None
            else (True if fresh else None)
        ),
        "voc_expected_gate_loss": (
            getattr(args, "voc_expected_gate_loss", None)
            if getattr(args, "voc_expected_gate_loss", None) is not None
            else (True if fresh else None)
        ),
        "voc_ema_gate_target": (
            getattr(args, "voc_ema_gate_target", None)
            if getattr(args, "voc_ema_gate_target", None) is not None
            else (True if fresh else None)
        ),
        "voc_gate_target_tau": (
            getattr(args, "voc_gate_target_tau", None)
            if getattr(args, "voc_gate_target_tau", None) is not None
            else (0.1 if fresh else None)
        ),
        "voc_dedicated_gate": (
            getattr(args, "voc_dedicated_gate", None)
            if getattr(args, "voc_dedicated_gate", None) is not None
            else (True if fresh else None)
        ),
        "voc_soft_q_bce_gate": (
            getattr(args, "voc_soft_q_bce_gate", None)
            if getattr(args, "voc_soft_q_bce_gate", None) is not None
            else (True if fresh else None)
        ),
        "voc_gate_q_temperature": (
            getattr(args, "voc_gate_q_temperature", None)
            if getattr(args, "voc_gate_q_temperature", None) is not None
            else (0.05 if fresh else None)
        ),
        "voc_gate_confidence_weighted": (
            getattr(args, "voc_gate_confidence_weighted", None)
            if getattr(args, "voc_gate_confidence_weighted", None) is not None
            else (True if fresh else None)
        ),
        "voc_gate_adam_beta1": (
            getattr(args, "voc_gate_adam_beta1", None)
            if getattr(args, "voc_gate_adam_beta1", None) is not None
            else (0.9 if fresh else None)
        ),
        "voc_gate_param_align": (
            getattr(args, "voc_gate_param_align", None)
            if getattr(args, "voc_gate_param_align", None) is not None
            else (False if fresh else None)
        ),
        "voc_gate_param_align_coef": (
            getattr(args, "voc_gate_param_align_coef", None)
            if getattr(args, "voc_gate_param_align_coef", None) is not None
            else (1.0 if fresh else None)
        ),
        "voc_gate_exact_projection": (
            getattr(args, "voc_gate_exact_projection", None)
            if getattr(args, "voc_gate_exact_projection", None) is not None
            else (False if fresh else None)
        ),
        "voc_gate_epsilon_greedy_execution": (
            getattr(args, "voc_gate_epsilon_greedy_execution", None)
            if getattr(args, "voc_gate_epsilon_greedy_execution", None)
            is not None
            else (False if fresh else None)
        ),
        "voc_gate_execution_epsilon": (
            getattr(args, "voc_gate_execution_epsilon", None)
            if getattr(args, "voc_gate_execution_epsilon", None) is not None
            else (0.02 if fresh else None)
        ),
        "voc_actor_policy_version_barrier": (
            getattr(args, "voc_actor_policy_version_barrier", None)
            if getattr(args, "voc_actor_policy_version_barrier", None) is not None
            else (False if fresh else None)
        ),
        "voc_actor_policy_bundle_schema_version": (
            getattr(args, "voc_actor_policy_bundle_schema_version", None)
            if getattr(args, "voc_actor_policy_bundle_schema_version", None)
            is not None
            else (1 if fresh else None)
        ),
        "voc_actor_policy_barrier_timeout_s": (
            getattr(args, "voc_actor_policy_barrier_timeout_s", None)
            if getattr(args, "voc_actor_policy_barrier_timeout_s", None) is not None
            else (120.0 if fresh else None)
        ),
        "voc_actor_policy_ray_max_restarts": (
            getattr(args, "voc_actor_policy_ray_max_restarts", None)
            if getattr(args, "voc_actor_policy_ray_max_restarts", None) is not None
            else (0 if fresh else None)
        ),
        "voc_actor_policy_ray_max_task_retries": (
            getattr(args, "voc_actor_policy_ray_max_task_retries", None)
            if getattr(args, "voc_actor_policy_ray_max_task_retries", None)
            is not None
            else (0 if fresh else None)
        ),
        "actor_amp_init_scale": (
            getattr(args, "actor_amp_init_scale", None)
            if getattr(args, "actor_amp_init_scale", None) is not None
            else (256.0 if fresh else None)
        ),
        "voc_model_input_seal_schema_version": (
            getattr(args, "voc_model_input_seal_schema_version", None)
            if getattr(args, "voc_model_input_seal_schema_version", None)
            is not None
            else (0 if fresh else None)
        ),
        "voc_gate_learning_rate": (
            getattr(args, "voc_gate_learning_rate", None)
            if getattr(args, "voc_gate_learning_rate", None) is not None
            else (0.0003 if fresh else None)
        ),
        "voc_gate_grad_norm_clipping": (
            getattr(args, "voc_gate_grad_norm_clipping", None)
            if getattr(args, "voc_gate_grad_norm_clipping", None) is not None
            else (1.0 if fresh else None)
        ),
        "envpool": True if fresh else None,
        "parallel": False if fresh else None,
        "parallel_actor": False if fresh else None,
        "use_wandb": False if fresh else None,
        "float16": False if fresh else None,
        "model_float16": False if fresh else None,
        "model_disable_bn": False if fresh else None,
        "model_state_projection": "clamp" if fresh else None,
        "model_state_range_loss_cost": 1.0 if fresh else None,
        "rec_t": args.rec_t if args.rec_t is not None else (20 if fresh else None),
        "max_search_steps": (
            args.max_search_steps
            if args.max_search_steps is not None
            else (20 if fresh else None)
        ),
        "max_depth": (
            args.max_depth if args.max_depth is not None else (20 if fresh else None)
        ),
        "model_unroll_len": (
            args.model_unroll_len
            if args.model_unroll_len is not None
            else (20 if fresh else None)
        ),
        "think_cost": (
            args.think_cost if args.think_cost is not None else (0.0005 if fresh else None)
        ),
        "think_cost_anneal": False if fresh else None,
        "sep_im_head": True if fresh else None,
        "model_size_nn": (
            args.model_size_nn
            if args.model_size_nn is not None
            else (2 if fresh else None)
        ),
        "frame_stack_n": args.frame_stack_n,
        "grayscale": args.grayscale,
        "tree_carry": args.tree_carry,
    }
    if flags is None:
        try:
            flags = util.create_flags(
                ["default_thinker.yaml", "default_actor.yaml"],
                save_flags=False,
                post_fn=util.process_flags_actor,
                **{
                    key: value
                    for key, value in overrides.items()
                    if value is not None
                },
            )
        finally:
            if private_config_dir is not None:
                private_config_dir.cleanup()
    if (
        not bool(flags.dynamic_search)
        or not bool(flags.dynamic_factorized_control)
        or not bool(flags.sep_im_head)
    ):
        raise ValueError(
            "smoke requires dynamic_search=true, "
            "dynamic_factorized_control=true, and sep_im_head=true"
        )
    if str(flags.name) != args.env_name:
        raise ValueError(
            f"config environment {flags.name!r} does not match --env-name "
            f"{args.env_name!r}"
        )
    if int(flags.max_search_steps) <= 0:
        raise ValueError("smoke requires a positive max_search_steps watchdog")
    if fresh:
        flags.batch_length = int(args.scored_length)
        flags.icopro_device = str(args.device)
    elif int(flags.batch_length) != int(args.scored_length):
        raise ValueError(
            "checkpoint batch_length disagrees with --scored-length: "
            f"{flags.batch_length!r} versus {args.scored_length!r}"
        )
    return flags


def _load_stable_smoke_checkpoint(
    checkpoint_eval: Any,
    checkpoint_dir: Path,
    filename: str,
    *,
    completion_state: Optional[Mapping[str, Any]],
    label: str,
) -> Mapping[str, Any]:
    """Deserialize one stable generation, binding it to ``finish`` when present."""

    payload = checkpoint_eval._read_stable_single_link_bytes(
        Path(checkpoint_dir) / filename,
        label=label,
    )
    if completion_state is not None:
        checkpoint_files = completion_state.get("checkpoint_files")
        if not isinstance(checkpoint_files, Mapping) or filename not in checkpoint_files:
            raise ValueError(f"{label} is not bound by the completion marker")
        expected_digest, expected_size = checkpoint_eval._validate_sha256_record(
            checkpoint_files[filename],
            label=f"completion checkpoint_files[{filename!r}]",
            require_size=True,
        )
        if len(payload) != expected_size:
            raise RuntimeError(
                f"{label} size disagrees with the completion marker"
            )
        if hashlib.sha256(payload).hexdigest() != expected_digest:
            raise RuntimeError(
                f"{label} digest disagrees with the completion marker"
            )
    checkpoint = torch.load(
        io.BytesIO(payload), map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"{label} must deserialize to a mapping")
    return checkpoint


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    import evaluate_dynamic_imitation as checkpoint_eval

    schema6_bundle_validation = None
    schema7_bundle_validation = None
    schema8_bundle_validation = None
    schema9_bundle_validation = None
    schema10_bundle_validation = None
    schema11_bundle_validation = None
    schema12_bundle_validation = None
    schema13_bundle_validation = None
    schema6_runtime_state = None
    schema7_runtime_state = None
    schema8_runtime_state = None
    schema9_runtime_state = None
    schema10_runtime_state = None
    schema11_runtime_state = None
    schema12_runtime_state = None
    schema13_runtime_state = None
    atomic_loaded_hashes = None
    validated_config_payload = None
    validated_config_digest = None
    completion_state = None
    schema9_dispatch = getattr(
        checkpoint_eval, "dispatch_schema9_completed_bundle", None
    )
    schema9_stage_profiles = getattr(
        checkpoint_eval, "VOC_GATE_POLICY_SCHEMA9_STAGE_PROFILES", ()
    )
    schema9_version = getattr(
        checkpoint_eval, "VOC_GATE_POLICY_COMMON_MODE_Q_SCHEMA_VERSION", 9
    )
    schema10_dispatch = getattr(
        checkpoint_eval, "dispatch_schema10_completed_bundle", None
    )
    schema10_stage_profiles = getattr(
        checkpoint_eval, "VOC_GATE_POLICY_SCHEMA10_STAGE_PROFILES", ()
    )
    schema10_version = getattr(
        checkpoint_eval, "VOC_GATE_POLICY_HUBER_COMMON_Q_SCHEMA_VERSION", 10
    )
    schema11_dispatch = getattr(
        checkpoint_eval, "dispatch_schema11_completed_bundle", None
    )
    schema11_stage_profiles = getattr(
        checkpoint_eval, "VOC_GATE_POLICY_SCHEMA11_STAGE_PROFILES", ()
    )
    schema11_version = getattr(
        checkpoint_eval, "VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION", 11
    )
    schema12_dispatch = getattr(
        checkpoint_eval, "dispatch_schema12_completed_bundle", None
    )
    schema12_stage_profiles = getattr(
        checkpoint_eval, "VOC_GATE_POLICY_SCHEMA12_STAGE_PROFILES", ()
    )
    schema12_version = getattr(
        checkpoint_eval, "VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION", 12
    )
    schema12_claims_intent = getattr(
        checkpoint_eval, "_schema12_xpid_claims_intent", None
    )
    schema13_dispatch = getattr(
        checkpoint_eval, "dispatch_schema13_completed_bundle", None
    )
    schema13_stage_profiles = getattr(
        checkpoint_eval, "VOC_GATE_POLICY_SCHEMA13_STAGE_PROFILES", ()
    )
    schema13_version = getattr(
        checkpoint_eval, "VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION", 13
    )
    schema13_claims_intent = getattr(
        checkpoint_eval, "_schema13_xpid_claims_intent", None
    )
    if args.checkpoint_dir is not None:
        validated_config_payload = (
            checkpoint_eval._read_stable_single_link_bytes(
                args.checkpoint_dir / "config_c.yaml",
                label="smoke checkpoint config",
            )
        )
        validated_config_digest = hashlib.sha256(
            validated_config_payload
        ).hexdigest()
        try:
            config_claim = yaml.safe_load(validated_config_payload.decode("utf-8"))
        except (UnicodeDecodeError, yaml.YAMLError) as error:
            raise ValueError(
                "smoke checkpoint config is not strict UTF-8 YAML"
            ) from error
        if not isinstance(config_claim, Mapping):
            raise ValueError("smoke checkpoint config must be a mapping")
        raw_schema = config_claim.get("voc_gate_policy_schema_version")
        raw_xpid = config_claim.get("xpid")
        local_schema13_intent = _smoke_schema13_xpid_claims_intent(raw_xpid)
        if callable(schema13_claims_intent):
            external_schema13_intent = schema13_claims_intent(raw_xpid)
            if (
                type(external_schema13_intent) is not bool
                or external_schema13_intent != local_schema13_intent
            ):
                raise RuntimeError(
                    "smoke/public schema-13 lexical classifiers disagree"
                )
        raw_schema13_intent = (
            raw_schema == schema13_version or local_schema13_intent
        )
        if raw_schema13_intent and (
            schema13_dispatch is None or not callable(schema13_claims_intent)
        ):
            raise RuntimeError(
                "schema-13 smoke validator lacks schema-13 dispatch or "
                "lexical classifier"
            )
        versioned_completion_claim = (
            type(raw_schema) is int and raw_schema in (6, 7, 8, 9, 10, 11, 12, 13)
        ) or (
            type(raw_xpid) is str
            and raw_xpid
            in {
                stage[0]
                for stage in checkpoint_eval.VOC_GATE_POLICY_SCHEMA8_STAGE_PROFILES
            }
            | {
                stage[0]
                for stage in schema9_stage_profiles
            }
            | {
                stage[0]
                for stage in schema10_stage_profiles
            }
            | {
                stage[0]
                for stage in schema11_stage_profiles
            }
            | {
                stage[0]
                for stage in schema12_stage_profiles
            }
            | {
                stage[0]
                for stage in schema13_stage_profiles
            }
            or (
                type(raw_xpid) is str
                and raw_xpid.startswith("enduro-voc-v17-huber-common-eps25-")
            )
            or (
                type(raw_xpid) is str
                and raw_xpid.startswith("enduro-voc-v18-orthocd-adam-eps25-")
            )
            or (
                type(raw_xpid) is str
                and raw_xpid.startswith(
                    "enduro-voc-v19-tau1-orthocd-adam-eps25-"
                )
            )
            or (
                type(raw_xpid) is str
                and raw_xpid.startswith(
                    "enduro-voc-v20-telemetry-tau1-orthocd-adam-eps25-"
                )
            )
            or raw_schema13_intent
        )
        if versioned_completion_claim:
            if raw_schema13_intent:
                completion_state = checkpoint_eval.validate_schema13_completion_marker(
                    args.checkpoint_dir
                )
                atomic_loaded_hashes = checkpoint_eval._schema13_checkpoint_hashes(
                    args.checkpoint_dir, completion_state=completion_state
                )
            else:
                atomic_loaded_hashes = checkpoint_eval.checkpoint_hashes(
                    args.checkpoint_dir
                )
                completion_state = checkpoint_eval.validate_completion_marker(
                    args.checkpoint_dir
                )
            validated_config_digest = completion_state["checkpoint_files"][
                "config_c.yaml"
            ]["sha256"]
        if (
            hashlib.sha256(validated_config_payload).hexdigest()
            != validated_config_digest
            or (
                atomic_loaded_hashes is not None
                and atomic_loaded_hashes.get("config_c.yaml")
                != validated_config_digest
            )
        ):
            raise RuntimeError(
                "checkpoint config changed after smoke completion validation"
            )
        # The schema-8 completed bundle is authoritative before any private
        # flag load/copy or live environment/data/tensor path.
        schema8_bundle_validation = (
            checkpoint_eval.dispatch_schema8_completed_bundle(
                args.checkpoint_dir,
                completion_state=completion_state,
                config_payload=validated_config_payload,
                expected_config_sha256=validated_config_digest,
            )
        )
        if schema9_dispatch is None and (
            raw_schema == schema9_version
            or (
                type(raw_xpid) is str
                and raw_xpid.startswith("enduro-voc-v16-commonmode-eps25-")
            )
        ):
            raise RuntimeError("schema-9 smoke validator lacks schema-9 dispatch")
        schema9_bundle_validation = (
            schema9_dispatch(
                args.checkpoint_dir,
                completion_state=completion_state,
                config_payload=validated_config_payload,
                expected_config_sha256=validated_config_digest,
            )
            if schema9_dispatch is not None
            else None
        )
        if schema10_dispatch is None and (
            raw_schema == schema10_version
            or (
                type(raw_xpid) is str
                and raw_xpid.startswith("enduro-voc-v17-huber-common-eps25-")
            )
        ):
            raise RuntimeError(
                "schema-10 smoke validator lacks schema-10 dispatch"
            )
        schema10_bundle_validation = (
            schema10_dispatch(
                args.checkpoint_dir,
                completion_state=completion_state,
                config_payload=validated_config_payload,
                expected_config_sha256=validated_config_digest,
            )
            if schema10_dispatch is not None
            else None
        )
        if schema11_dispatch is None and (
            raw_schema == schema11_version
            or (
                type(raw_xpid) is str
                and raw_xpid.startswith("enduro-voc-v18-orthocd-adam-eps25-")
            )
        ):
            raise RuntimeError(
                "schema-11 smoke validator lacks schema-11 dispatch"
            )
        schema11_bundle_validation = (
            schema11_dispatch(
                args.checkpoint_dir,
                completion_state=completion_state,
                config_payload=validated_config_payload,
                expected_config_sha256=validated_config_digest,
            )
            if schema11_dispatch is not None
            else None
        )
        raw_schema12_intent = (
            raw_schema == schema12_version
            or (
                type(raw_xpid) is str
                and raw_xpid.startswith(
                    "enduro-voc-v19-tau1-orthocd-adam-eps25-"
                )
            )
        )
        if raw_schema12_intent and (
            schema12_dispatch is None or not callable(schema12_claims_intent)
        ):
            raise RuntimeError(
                "schema-12 smoke validator lacks schema-12 dispatch"
            )
        schema12_bundle_validation = (
            schema12_dispatch(
                args.checkpoint_dir,
                completion_state=completion_state,
                config_payload=validated_config_payload,
                expected_config_sha256=validated_config_digest,
            )
            if schema12_dispatch is not None
            else None
        )
        schema13_bundle_validation = (
            schema13_dispatch(
                args.checkpoint_dir,
                completion_state=completion_state,
                config_payload=validated_config_payload,
                expected_config_sha256=validated_config_digest,
            )
            if schema13_dispatch is not None
            else None
        )
        if schema8_bundle_validation is not None:
            authoritative_config_digest = schema8_bundle_validation[
                "completion_evidence"
            ]["checkpoint_files"]["config_c.yaml"]["sha256"]
            if authoritative_config_digest != validated_config_digest:
                raise RuntimeError(
                    "schema-8 authoritative config digest disagrees in smoke"
                )
        if schema9_bundle_validation is not None:
            authoritative_config_digest = schema9_bundle_validation[
                "completion_evidence"
            ]["checkpoint_files"]["config_c.yaml"]["sha256"]
            if authoritative_config_digest != validated_config_digest:
                raise RuntimeError(
                    "schema-9 authoritative config digest disagrees in smoke"
                )
        if schema10_bundle_validation is not None:
            authoritative_config_digest = schema10_bundle_validation[
                "completion_evidence"
            ]["checkpoint_files"]["config_c.yaml"]["sha256"]
            if authoritative_config_digest != validated_config_digest:
                raise RuntimeError(
                    "schema-10 authoritative config digest disagrees in smoke"
                )
        if schema11_bundle_validation is not None:
            authoritative_config_digest = schema11_bundle_validation[
                "completion_evidence"
            ]["checkpoint_files"]["config_c.yaml"]["sha256"]
            if authoritative_config_digest != validated_config_digest:
                raise RuntimeError(
                    "schema-11 authoritative config digest disagrees in smoke"
                )
        if schema12_bundle_validation is not None:
            authoritative_config_digest = schema12_bundle_validation[
                "completion_evidence"
            ]["checkpoint_files"]["config_c.yaml"]["sha256"]
            if authoritative_config_digest != validated_config_digest:
                raise RuntimeError(
                    "schema-12 authoritative config digest disagrees in smoke"
                )
            _require_schema12_smoke_ema_online_equality(
                args.checkpoint_dir,
                completion_state,
            )
        if schema13_bundle_validation is not None:
            authoritative_config_digest = schema13_bundle_validation[
                "completion_evidence"
            ]["checkpoint_files"]["config_c.yaml"]["sha256"]
            if authoritative_config_digest != validated_config_digest:
                raise RuntimeError(
                    "schema-13 authoritative config digest disagrees in smoke"
                )
            _require_schema13_smoke_ema_online_equality(
                args.checkpoint_dir,
                completion_state,
            )
    if validated_config_payload is not None:
        runtime_config_payload = validated_config_payload
        runtime_config_digest = validated_config_digest
        if getattr(args, "config", None) is not None:
            runtime_config_payload = checkpoint_eval._read_stable_single_link_bytes(
                args.config,
                label="explicit smoke config",
            )
            runtime_config_digest = hashlib.sha256(
                runtime_config_payload
            ).hexdigest()
            selected_schema8_validation = (
                checkpoint_eval.dispatch_schema8_completed_bundle(
                    args.checkpoint_dir,
                    completion_state=completion_state,
                    config_payload=runtime_config_payload,
                    expected_config_sha256=runtime_config_digest,
                )
            )
            selected_schema9_validation = (
                schema9_dispatch(
                    args.checkpoint_dir,
                    completion_state=completion_state,
                    config_payload=runtime_config_payload,
                    expected_config_sha256=runtime_config_digest,
                )
                if schema9_dispatch is not None
                else None
            )
            selected_schema10_validation = (
                schema10_dispatch(
                    args.checkpoint_dir,
                    completion_state=completion_state,
                    config_payload=runtime_config_payload,
                    expected_config_sha256=runtime_config_digest,
                )
                if schema10_dispatch is not None
                else None
            )
            selected_schema11_validation = (
                schema11_dispatch(
                    args.checkpoint_dir,
                    completion_state=completion_state,
                    config_payload=runtime_config_payload,
                    expected_config_sha256=runtime_config_digest,
                )
                if schema11_dispatch is not None
                else None
            )
            selected_schema12_validation = (
                schema12_dispatch(
                    args.checkpoint_dir,
                    completion_state=completion_state,
                    config_payload=runtime_config_payload,
                    expected_config_sha256=runtime_config_digest,
                )
                if schema12_dispatch is not None
                else None
            )
            selected_schema13_validation = (
                schema13_dispatch(
                    args.checkpoint_dir,
                    completion_state=completion_state,
                    config_payload=runtime_config_payload,
                    expected_config_sha256=runtime_config_digest,
                )
                if schema13_dispatch is not None
                else None
            )
            try:
                runtime_claim = yaml.safe_load(
                    runtime_config_payload.decode("utf-8")
                )
            except (UnicodeDecodeError, yaml.YAMLError) as error:
                raise ValueError(
                    "explicit smoke config is not strict UTF-8 YAML"
                ) from error
            if not isinstance(runtime_claim, Mapping):
                raise ValueError("explicit smoke config must be a mapping")
            runtime_schema = runtime_claim.get("voc_gate_policy_schema_version")
            runtime_xpid = runtime_claim.get("xpid")
            runtime_claims_schema8 = (
                runtime_schema
                == checkpoint_eval.VOC_GATE_POLICY_HALF_SQUARED_Q_SCHEMA_VERSION
                or (
                    type(runtime_xpid) is str
                    and runtime_xpid
                    in {
                        stage[0]
                        for stage in checkpoint_eval.VOC_GATE_POLICY_SCHEMA8_STAGE_PROFILES
                    }
                )
            )
            runtime_claims_schema9 = (
                runtime_schema
                == schema9_version
                or (
                    type(runtime_xpid) is str
                    and runtime_xpid
                    in {
                        stage[0]
                        for stage in schema9_stage_profiles
                    }
                )
            )
            runtime_claims_schema10 = (
                runtime_schema
                == schema10_version
                or (
                    type(runtime_xpid) is str
                    and (
                        runtime_xpid
                        in {
                            stage[0]
                            for stage in schema10_stage_profiles
                        }
                        or runtime_xpid.startswith(
                            "enduro-voc-v17-huber-common-eps25-"
                        )
                    )
                )
            )
            runtime_claims_schema11 = (
                runtime_schema
                == schema11_version
                or (
                    type(runtime_xpid) is str
                    and (
                        runtime_xpid
                        in {
                            stage[0]
                            for stage in schema11_stage_profiles
                        }
                        or runtime_xpid.startswith(
                            "enduro-voc-v18-orthocd-adam-eps25-"
                        )
                    )
                )
            )
            runtime_claims_schema12 = (
                runtime_schema == schema12_version
                or (
                    callable(schema12_claims_intent)
                    and schema12_claims_intent(runtime_xpid)
                )
                or (
                    type(runtime_xpid) is str
                    and runtime_xpid.startswith(
                        "enduro-voc-v19-tau1-orthocd-adam-eps25-"
                    )
                )
                or (
                    type(runtime_xpid) is str
                    and runtime_xpid
                    in {stage[0] for stage in schema12_stage_profiles}
                )
            )
            local_runtime_schema13_intent = (
                _smoke_schema13_xpid_claims_intent(runtime_xpid)
            )
            if callable(schema13_claims_intent):
                external_runtime_schema13_intent = schema13_claims_intent(
                    runtime_xpid
                )
                if (
                    type(external_runtime_schema13_intent) is not bool
                    or external_runtime_schema13_intent
                    != local_runtime_schema13_intent
                ):
                    raise RuntimeError(
                        "smoke/public schema-13 lexical classifiers disagree"
                    )
            runtime_claims_schema13 = (
                runtime_schema == schema13_version
                or local_runtime_schema13_intent
            )
            if runtime_claims_schema13 and not callable(
                schema13_claims_intent
            ):
                raise RuntimeError(
                    "schema-13 smoke validator lacks schema-13 lexical "
                    "classifier"
                )
            if schema8_bundle_validation is not None and (
                runtime_config_payload != validated_config_payload
                or selected_schema8_validation != schema8_bundle_validation
            ):
                raise ValueError(
                    "schema-8 smoke requires its authoritative checkpoint "
                    "config payload"
                )
            if runtime_claims_schema8 and schema8_bundle_validation is None:
                raise ValueError(
                    "schema-8 smoke requires its authoritative checkpoint "
                    "config payload"
                )
            if schema9_bundle_validation is not None and (
                runtime_config_payload != validated_config_payload
                or selected_schema9_validation != schema9_bundle_validation
            ):
                raise ValueError(
                    "schema-9 smoke requires its authoritative checkpoint "
                    "config payload"
                )
            if runtime_claims_schema9 and schema9_bundle_validation is None:
                if schema9_dispatch is None:
                    raise RuntimeError(
                        "schema-9 smoke validator lacks schema-9 dispatch"
                    )
                raise ValueError(
                    "schema-9 smoke requires its authoritative checkpoint "
                    "config payload"
                )
            if schema10_bundle_validation is not None and (
                runtime_config_payload != validated_config_payload
                or selected_schema10_validation != schema10_bundle_validation
            ):
                raise ValueError(
                    "schema-10 smoke requires its authoritative checkpoint "
                    "config payload"
                )
            if runtime_claims_schema10 and schema10_bundle_validation is None:
                if schema10_dispatch is None:
                    raise RuntimeError(
                        "schema-10 smoke validator lacks schema-10 dispatch"
                    )
                raise ValueError(
                    "schema-10 smoke requires its authoritative checkpoint "
                    "config payload"
                )
            if schema11_bundle_validation is not None and (
                runtime_config_payload != validated_config_payload
                or selected_schema11_validation != schema11_bundle_validation
            ):
                raise ValueError(
                    "schema-11 smoke requires its authoritative checkpoint "
                    "config payload"
                )
            if runtime_claims_schema11 and schema11_bundle_validation is None:
                if schema11_dispatch is None:
                    raise RuntimeError(
                        "schema-11 smoke validator lacks schema-11 dispatch"
                    )
                raise ValueError(
                    "schema-11 smoke requires its authoritative checkpoint "
                    "config payload"
                )
            if schema12_bundle_validation is not None and (
                runtime_config_payload != validated_config_payload
                or selected_schema12_validation != schema12_bundle_validation
            ):
                raise ValueError(
                    "schema-12 smoke requires its authoritative checkpoint "
                    "config payload"
                )
            if runtime_claims_schema12 and schema12_bundle_validation is None:
                if schema12_dispatch is None:
                    raise RuntimeError(
                        "schema-12 smoke validator lacks schema-12 dispatch"
                    )
                raise ValueError(
                    "schema-12 smoke requires its authoritative checkpoint "
                    "config payload"
                )
            if schema13_bundle_validation is not None and (
                runtime_config_payload != validated_config_payload
                or selected_schema13_validation != schema13_bundle_validation
            ):
                raise ValueError(
                    "schema-13 smoke requires its authoritative checkpoint "
                    "config payload"
                )
            if runtime_claims_schema13 and schema13_bundle_validation is None:
                if schema13_dispatch is None:
                    raise RuntimeError(
                        "schema-13 smoke validator lacks schema-13 dispatch"
                    )
                raise ValueError(
                    "schema-13 smoke requires its authoritative checkpoint "
                    "config payload"
                )
        load_flag_kwargs = {
            "config_payload": runtime_config_payload,
            "expected_sha256": runtime_config_digest,
        }
        if schema13_bundle_validation is not None:
            load_flag_kwargs["schema13_bound"] = True
        training_flags = _load_flags(args, **load_flag_kwargs)
        if str(training_flags.name) != args.env_name:
            raise ValueError(
                f"config environment {training_flags.name!r} does not match "
                f"--env-name {args.env_name!r}"
            )
        if int(training_flags.max_search_steps) <= 0:
            raise ValueError(
                "smoke requires a positive max_search_steps watchdog"
            )
        if int(training_flags.batch_length) != int(args.scored_length):
            raise ValueError(
                "checkpoint batch_length disagrees with --scored-length: "
                f"{training_flags.batch_length!r} versus "
                f"{args.scored_length!r}"
            )
    else:
        training_flags = _load_flags(args)
    if args.checkpoint_dir is not None:
        schema6_bundle_validation = (
            checkpoint_eval.validate_schema6_completed_bundle(
                args.checkpoint_dir
            )
        )
        schema7_bundle_validation = (
            checkpoint_eval.validate_schema7_completed_bundle(
                args.checkpoint_dir
            )
        )
        if sum(
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
        ) > 1:
            raise RuntimeError("Smoke checkpoint resolved to multiple schemas")
        if atomic_loaded_hashes is None and (
            schema6_bundle_validation is not None
            or schema7_bundle_validation is not None
        ):
            atomic_loaded_hashes = checkpoint_eval.checkpoint_hashes(
                args.checkpoint_dir
            )
    flags, atomic_runtime_state = checkpoint_eval.evaluation_runtime_flags(
        training_flags
    )
    if schema6_bundle_validation is not None:
        schema6_runtime_state = atomic_runtime_state
    if schema7_bundle_validation is not None:
        schema7_runtime_state = atomic_runtime_state
    if schema8_bundle_validation is not None:
        schema8_runtime_state = atomic_runtime_state
        post_copy_schema8_validation = (
            checkpoint_eval.validate_schema8_completed_bundle(
                args.checkpoint_dir
            )
        )
        if post_copy_schema8_validation != schema8_bundle_validation:
            raise RuntimeError(
                "schema-8 completed-bundle evidence changed across private copy"
            )
        if (
            checkpoint_eval.checkpoint_hashes(args.checkpoint_dir)
            != atomic_loaded_hashes
        ):
            raise RuntimeError(
                "schema-8 checkpoint changed across private evaluation copy"
            )
    if schema9_bundle_validation is not None:
        schema9_runtime_state = atomic_runtime_state
        post_copy_schema9_validation = (
            checkpoint_eval.validate_schema9_completed_bundle(
                args.checkpoint_dir,
                completion_state=completion_state,
                config_payload=validated_config_payload,
                expected_config_sha256=validated_config_digest,
            )
        )
        if post_copy_schema9_validation != schema9_bundle_validation:
            raise RuntimeError(
                "schema-9 completed-bundle evidence changed across private copy"
            )
        if (
            checkpoint_eval.checkpoint_hashes(args.checkpoint_dir)
            != atomic_loaded_hashes
        ):
            raise RuntimeError(
                "schema-9 checkpoint changed across private evaluation copy"
            )
    if schema10_bundle_validation is not None:
        schema10_runtime_state = atomic_runtime_state
        post_copy_schema10_validation = (
            checkpoint_eval.validate_schema10_completed_bundle(
                args.checkpoint_dir,
                completion_state=completion_state,
                config_payload=validated_config_payload,
                expected_config_sha256=validated_config_digest,
            )
        )
        if post_copy_schema10_validation != schema10_bundle_validation:
            raise RuntimeError(
                "schema-10 completed-bundle evidence changed across private copy"
            )
        if (
            checkpoint_eval.checkpoint_hashes(args.checkpoint_dir)
            != atomic_loaded_hashes
        ):
            raise RuntimeError(
                "schema-10 checkpoint changed across private evaluation copy"
            )
    if schema11_bundle_validation is not None:
        schema11_runtime_state = atomic_runtime_state
        post_copy_schema11_validation = (
            checkpoint_eval.validate_schema11_completed_bundle(
                args.checkpoint_dir,
                completion_state=completion_state,
                config_payload=validated_config_payload,
                expected_config_sha256=validated_config_digest,
            )
        )
        if post_copy_schema11_validation != schema11_bundle_validation:
            raise RuntimeError(
                "schema-11 completed-bundle evidence changed across private copy"
            )
        if (
            checkpoint_eval.checkpoint_hashes(args.checkpoint_dir)
            != atomic_loaded_hashes
        ):
            raise RuntimeError(
                "schema-11 checkpoint changed across private evaluation copy"
            )
    if schema12_bundle_validation is not None:
        schema12_runtime_state = atomic_runtime_state
        post_copy_schema12_validation = (
            checkpoint_eval.validate_schema12_completed_bundle(
                args.checkpoint_dir,
                completion_state=completion_state,
                config_payload=validated_config_payload,
                expected_config_sha256=validated_config_digest,
            )
        )
        if post_copy_schema12_validation != schema12_bundle_validation:
            raise RuntimeError(
                "schema-12 completed-bundle evidence changed across private copy"
            )
        _require_schema12_smoke_ema_online_equality(
            args.checkpoint_dir,
            completion_state,
        )
        if (
            checkpoint_eval.checkpoint_hashes(args.checkpoint_dir)
            != atomic_loaded_hashes
        ):
            raise RuntimeError(
                "schema-12 checkpoint changed across private evaluation copy"
            )
    if schema13_bundle_validation is not None:
        schema13_runtime_state = atomic_runtime_state
        post_copy_schema13_validation = (
            checkpoint_eval.validate_schema13_completed_bundle(
                args.checkpoint_dir,
                completion_state=completion_state,
                config_payload=validated_config_payload,
                expected_config_sha256=validated_config_digest,
            )
        )
        if post_copy_schema13_validation != schema13_bundle_validation:
            raise RuntimeError(
                "schema-13 completed-bundle evidence changed across private copy"
            )
        _require_schema13_smoke_ema_online_equality(
            args.checkpoint_dir,
            completion_state,
        )
        if (
            checkpoint_eval._schema13_checkpoint_hashes(
                args.checkpoint_dir, completion_state=completion_state
            )
            != atomic_loaded_hashes
        ):
            raise RuntimeError(
                "schema-13 checkpoint changed across private evaluation copy"
            )
    if validated_config_payload is not None:
        current_config_payload = checkpoint_eval._read_stable_single_link_bytes(
            args.checkpoint_dir / "config_c.yaml",
            label="smoke checkpoint config before live environment",
        )
        if (
            hashlib.sha256(current_config_payload).hexdigest()
            != validated_config_digest
        ):
            raise RuntimeError(
                "checkpoint config changed before smoke live environment"
            )
    current_atomic_hashes = (
        checkpoint_eval._schema13_checkpoint_hashes(
            args.checkpoint_dir, completion_state=completion_state
        )
        if schema13_bundle_validation is not None
        else checkpoint_eval.checkpoint_hashes(args.checkpoint_dir)
    ) if atomic_loaded_hashes is not None else None
    if atomic_loaded_hashes is not None and current_atomic_hashes != atomic_loaded_hashes:
        raise RuntimeError("checkpoint changed before smoke live environment")

    actor_checkpoint = None
    model_checkpoint = None
    if args.checkpoint_dir is not None and schema13_bundle_validation is not None:
        actor_checkpoint = _load_stable_smoke_checkpoint(
            checkpoint_eval,
            args.checkpoint_dir,
            "ckp_actor.tar",
            completion_state=completion_state,
            label="smoke actor checkpoint",
        )
        model_checkpoint = _load_stable_smoke_checkpoint(
            checkpoint_eval,
            args.checkpoint_dir,
            "ckp_model.tar",
            completion_state=completion_state,
            label="smoke model checkpoint",
        )

    from thinker.actor_net import ActorNet
    from thinker.bc_loader import FrameStackedBehavioralDataLoader
    from thinker.cenv import cModelWrapper
    from thinker.dataset_env import BehaviorSequenceVectorEnv
    from thinker.dynamic_imitation import DynamicImitationRunner
    from thinker.gym_add.wrapper import create_envpool
    from thinker.main import _validate_online_env_contract
    from thinker.model_net import ModelNet
    from thinker import util

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA smoke requested, but CUDA is unavailable")
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(args.device)

    flags.batch_length = int(args.scored_length)
    flags.icopro_device = str(args.device)
    live_env = create_envpool(args.env_name, flags, env_n=args.batch_size)
    try:
        live_obs, _ = live_env.reset()
        live_next, live_reward, _, _, _ = live_env.step(
            np.zeros(args.batch_size, dtype=np.int64)
        )
        obs_space = live_env.single_observation_space
        action_space = live_env.single_action_space
        frame_stack_n = int(live_env.frame_stack_n)
        frame_ch = _validate_online_env_contract(
            obs_space,
            action_space,
            frame_stack_n,
            expected_frame_stack_n=flags.frame_stack_n,
            require_discrete=True,
        )
        expected_live_shape = (args.batch_size,) + tuple(obs_space.shape)
        if tuple(live_obs.shape) != expected_live_shape:
            raise ValueError(f"EnvPool reset shape mismatch: {live_obs.shape}")
        if tuple(live_next.shape) != expected_live_shape:
            raise ValueError(f"EnvPool step shape mismatch: {live_next.shape}")
    finally:
        live_env.close()

    dtype = np.dtype(obs_space.dtype)
    byte_contract = (
        dtype == np.dtype(np.uint8)
        and np.all(obs_space.low == 0)
        and np.all(obs_space.high == 255)
    )
    unit_float_contract = (
        dtype == np.dtype(np.float32)
        and np.all(obs_space.low == 0.0)
        and np.all(obs_space.high == 1.0)
    )
    if not byte_contract and not unit_float_contract:
        raise ValueError(
            "behavioral preprocessing supports only uint8 [0,255] or "
            "float32 [0,1] online observations"
        )

    loader = FrameStackedBehavioralDataLoader(
        base_path=args.data_root,
        subjects=args.subjects,
        game_id=args.game_id,
        sessions=args.sessions,
        num_actions=int(action_space.n),
        scored_length=args.scored_length,
        frame_stack_n=frame_stack_n,
        target_size=tuple(int(value) for value in obs_space.shape[-2:]),
        grayscale=bool(flags.grayscale),
        normalize=unit_float_contract,
        seed=args.seed,
    )
    batch = loader.get_sequence_batch(
        batch_size=args.batch_size, sequence_length=args.scored_length
    )
    observations = np.asarray(batch["obs_seq"])
    actions = np.asarray(batch["actions_seq"], dtype=np.int64)
    if tuple(observations.shape[2:]) != tuple(obs_space.shape):
        raise ValueError(
            f"behavior/EnvPool shape mismatch: {observations.shape[2:]} "
            f"versus {obs_space.shape}"
        )
    if np.dtype(observations.dtype) != dtype:
        raise TypeError(
            f"behavior/EnvPool dtype mismatch: {observations.dtype} versus {dtype}"
        )
    if actions.min() < 0 or actions.max() >= int(action_space.n):
        raise ValueError("behavior batch contains an out-of-range action")

    model = ModelNet(
        obs_space=obs_space,
        action_space=action_space,
        flags=flags,
        frame_stack_n=frame_stack_n,
    ).to(args.device)

    behavior_env = BehaviorSequenceVectorEnv(
        obs_seq=observations,
        actions_seq=actions,
        rewards_seq=np.asarray(batch["rewards_seq"], dtype=np.float32),
        done_seq=np.asarray(batch["done_seq"], dtype=np.bool_),
        truncated_seq=np.asarray(batch["truncated_seq"], dtype=np.bool_),
        initial_prev_action=np.asarray(batch["initial_prev_action"], dtype=np.int64),
        score_mask=np.asarray(batch["score_mask"], dtype=np.bool_),
        num_actions=int(action_space.n),
    )
    template = cModelWrapper(
        env=behavior_env,
        env_n=args.batch_size,
        flags=flags,
        model_net=model,
        device=args.device,
        timing=False,
    )
    try:
        actor = ActorNet(
            obs_space=_vector_actor_observation_space(
                template.observation_space, args.batch_size
            ),
            action_space=template.action_space,
            flags=flags,
            tree_rep_meaning=util.get_tree_rep_meaning(
                int(action_space.n), 1, flags
            ),
        ).to(args.device)
    finally:
        template.close()

    validated_voc_protocol = util.get_voc_protocol(flags)
    voc_checkpoint_validation = None
    schema11_actor_identity: dict[str, str] = {}
    schema12_actor_identity: dict[str, str] = {}
    schema13_actor_identity: dict[str, str] = {}
    actor_checkpoint_public_validation = None
    model_checkpoint_public_validation = None
    voc_checkpoint_resolved_identity = None
    if args.checkpoint_dir is not None:
        if schema13_bundle_validation is None:
            actor_path = args.checkpoint_dir / "ckp_actor.tar"
            model_path = args.checkpoint_dir / "ckp_model.tar"
            if not actor_path.is_file() or not model_path.is_file():
                raise FileNotFoundError(
                    "checkpoint-dir must contain ckp_actor.tar and ckp_model.tar"
                )
            actor_checkpoint = torch.load(
                actor_path, map_location="cpu", weights_only=False
            )
            actor_state = _checkpoint_state_dict(
                actor_checkpoint, "actor_net_state_dict"
            )
            model_checkpoint = torch.load(
                model_path, map_location="cpu", weights_only=False
            )
            if not isinstance(model_checkpoint, Mapping):
                raise ValueError("Smoke ModelNet checkpoint must be a mapping")
        else:
            actor_state = _checkpoint_state_dict(
                actor_checkpoint, "actor_net_state_dict"
            )
        model_state = _checkpoint_state_dict(
            model_checkpoint, "model_net_state_dict"
        )
        current_atomic_hashes = (
            checkpoint_eval._schema13_checkpoint_hashes(
                args.checkpoint_dir, completion_state=completion_state
            )
            if schema13_bundle_validation is not None
            else checkpoint_eval.checkpoint_hashes(args.checkpoint_dir)
        ) if atomic_loaded_hashes is not None else None
        if atomic_loaded_hashes is not None and current_atomic_hashes != atomic_loaded_hashes:
            raise RuntimeError(
                "atomic checkpoint changed while smoke loaded its bundle"
            )
        _validate_state_dict(actor, actor_state, "ActorNet")
        _validate_state_dict(model, model_state, "ModelNet")
        if schema13_bundle_validation is not None:
            active_state = util.validate_voc_schema13_final_actor_checkpoint(
                actor_checkpoint,
                training_flags,
                label="Smoke schema-13 final actor checkpoint",
            )
        elif schema12_bundle_validation is not None:
            active_state = util.validate_voc_schema12_final_actor_checkpoint(
                actor_checkpoint,
                training_flags,
                label="Smoke schema-12 final actor checkpoint",
            )
        elif schema11_bundle_validation is not None:
            active_state = util.validate_voc_schema11_final_actor_checkpoint(
                actor_checkpoint,
                training_flags,
                label="Smoke schema-11 final actor checkpoint",
            )
        elif schema10_bundle_validation is not None:
            active_state = util.validate_voc_schema10_final_actor_checkpoint(
                actor_checkpoint,
                training_flags,
                label="Smoke schema-10 final actor checkpoint",
            )
        elif schema9_bundle_validation is not None:
            active_state = util.validate_voc_schema9_final_actor_checkpoint(
                actor_checkpoint,
                training_flags,
                label="Smoke schema-9 final actor checkpoint",
            )
        elif schema8_bundle_validation is not None:
            active_state = util.validate_voc_schema8_final_actor_checkpoint(
                actor_checkpoint,
                training_flags,
                label="Smoke schema-8 final actor checkpoint",
            )
        else:
            active_state = util.validate_voc_active_resume_checkpoint(
                actor_checkpoint, training_flags, label="Smoke actor checkpoint"
            )
        checkpoint_metadata = _validate_smoke_checkpoint_metadata(
            actor_checkpoint,
            model_checkpoint,
            training_flags,
            _checkpoint_evaluation_spec(
                args, flags, obs_space, action_space, frame_stack_n
            ),
            active_state,
            schema6_bundle_validation,
            schema7_bundle_validation,
            schema8_bundle_validation,
            schema9_bundle_validation,
            schema10_bundle_validation,
            schema11_bundle_validation,
            schema12_bundle_validation,
            schema13_bundle_validation,
        )
        actor_checkpoint_public_validation = checkpoint_metadata[
            "actor_public_validation"
        ]
        model_checkpoint_public_validation = checkpoint_metadata[
            "model_public_validation"
        ]
        voc_checkpoint_resolved_identity = checkpoint_metadata[
            "resolved_identity"
        ]
        if schema11_bundle_validation is not None:
            schema11_actor_identity = _validate_schema11_smoke_active_state(
                active_state,
                resolved_identity=voc_checkpoint_resolved_identity["config"],
            )
        if schema12_bundle_validation is not None:
            schema12_actor_identity = _validate_schema12_smoke_active_state(
                active_state,
                resolved_identity=voc_checkpoint_resolved_identity["config"],
            )
        if schema13_bundle_validation is not None:
            schema13_actor_identity = _validate_schema13_smoke_active_state(
                active_state,
                resolved_identity=voc_checkpoint_resolved_identity["config"],
            )
        validated_voc_protocol = active_state["voc_protocol"]
        checkpoint_voc_mode = active_state["dynamic_voc_mode"]
        if checkpoint_voc_mode != "off":
            promotion_qualified = all(
                int(active_state[name]) > 0
                for name in (
                    "voc_update_count",
                    "voc_continue_count",
                    "voc_stop_count",
                    "voc_holdout_count",
                    "voc_holdout_continue_count",
                    "voc_holdout_stop_count",
                )
            )
            if checkpoint_voc_mode == "control":
                promotion_qualified = (
                    promotion_qualified
                    and int(active_state["voc_gate_update_count"]) > 0
                )
            if promotion_qualified:
                certification_status = "promotion_qualified"
            elif int(active_state["voc_update_count"]) == 0:
                certification_status = "wiring_only"
            else:
                certification_status = "learned_unqualified"
            voc_checkpoint_validation = {
                "dynamic_voc_mode": checkpoint_voc_mode,
                "status": certification_status,
                "promotion_qualified": promotion_qualified,
                "voc_ema_gate_schema_version": active_state[
                    "voc_ema_gate_schema_version"
                ],
                "voc_ema_gate_update_count": active_state[
                    "voc_ema_gate_update_count"
                ],
                "voc_ema_gate_parent_update_count": active_state[
                    "voc_ema_gate_parent_update_count"
                ],
                "voc_gate_target_tau": active_state[
                    "voc_gate_target_tau"
                ],
                "voc_gate_policy_schema_version": active_state[
                    "voc_gate_policy_schema_version"
                ],
                "voc_gate_adam_beta1": active_state[
                    "voc_gate_adam_beta1"
                ],
                "voc_gate_adam_beta1_legacy_defaulted": active_state[
                    "voc_gate_adam_beta1_legacy_defaulted"
                ],
                "voc_gate_param_align": active_state[
                    "voc_gate_param_align"
                ],
                "voc_gate_param_align_coef": active_state[
                    "voc_gate_param_align_coef"
                ],
                "voc_gate_param_align_legacy_defaulted": active_state[
                    "voc_gate_param_align_legacy_defaulted"
                ],
                "voc_gate_exact_projection": active_state[
                    "voc_gate_exact_projection"
                ],
                "voc_gate_exact_projection_legacy_defaulted": active_state[
                    "voc_gate_exact_projection_legacy_defaulted"
                ],
                "voc_gate_epsilon_greedy_execution": active_state[
                    "voc_gate_epsilon_greedy_execution"
                ],
                "voc_gate_epsilon_greedy_execution_legacy_defaulted": (
                    active_state[
                        "voc_gate_epsilon_greedy_execution_legacy_defaulted"
                    ]
                ),
                "voc_gate_update_count": active_state[
                    "voc_gate_update_count"
                ],
                "voc_gate_optimizer_state_saved": active_state[
                    "voc_gate_optimizer_state_saved"
                ],
                "voc_gate_grad_scaler_state_saved": active_state[
                    "voc_gate_grad_scaler_state_saved"
                ],
                "voc_float16": active_state["voc_float16"],
                "voc_control_origin": active_state.get(
                    "voc_control_origin"
                ),
                **(
                    schema11_actor_identity
                    if schema11_bundle_validation is not None
                    else {}
                ),
                **(
                    schema12_actor_identity
                    if schema12_bundle_validation is not None
                    else {}
                ),
                **(
                    schema13_actor_identity
                    if schema13_bundle_validation is not None
                    else {}
                ),
            }
        actor.set_weights(actor_state)
        model.set_weights(model_state)

    if int(actor.num_actions) != int(model.num_actions) or int(actor.num_actions) != int(
        action_space.n
    ):
        raise ValueError("EnvPool, ActorNet, and ModelNet action counts disagree")
    if tuple(actor.online_real_state_space.shape) != tuple(obs_space.shape):
        raise ValueError("Actor online observation shape was not preserved")
    if np.dtype(actor.online_real_state_space.dtype) != dtype:
        raise TypeError("Actor online observation dtype was not preserved")
    if int(model.frame_stack_n) != frame_stack_n:
        raise ValueError("ModelNet frame-stack metadata were not preserved")

    actor.train(True)
    optimizer = torch.optim.Adam(
        actor.parameters(),
        lr=(
            float(args.learning_rate)
            if args.learning_rate is not None
            else float(flags.actor_learning_rate)
        ),
        eps=float(flags.actor_adam_eps),
    )
    runner = DynamicImitationRunner(actor, model, flags, device=args.device)
    try:
        model_versions = {
            name: parameter._version for name, parameter in model.named_parameters()
        }
        if not all(not parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("DynamicImitationRunner did not freeze ModelNet")
        actor_versions = {
            name: parameter._version for name, parameter in actor.named_parameters()
        }

        optimizer.zero_grad(set_to_none=True)
        result = runner.rollout(batch, tree_carry=bool(flags.tree_carry), training=True)
        expected_count = args.batch_size * args.scored_length
        if result.count != expected_count:
            raise RuntimeError(
                f"burn-in/scored count mismatch: {result.count} versus {expected_count}"
            )
        if not torch.equal(result.all_executed.cpu(), torch.as_tensor(actions)):
            raise RuntimeError("cenv did not execute every teacher-forced human action")
        if not torch.isfinite(result.loss):
            raise RuntimeError("imitation loss is non-finite")
        result.loss.backward()

        grad_entries = [
            (name, parameter)
            for name, parameter in actor.named_parameters()
            if parameter.grad is not None
            and int(torch.count_nonzero(parameter.grad).item()) > 0
        ]
        if not grad_entries:
            raise RuntimeError("Actor received no nonzero imitation gradient")
        if not all(torch.isfinite(parameter.grad).all() for _, parameter in grad_entries):
            raise RuntimeError("Actor imitation gradient is non-finite")
        probe_name, probe = grad_entries[0]
        probe_before = probe.detach().clone()
        grad_norm = torch.sqrt(
            sum(
                parameter.grad.detach().float().square().sum()
                for _, parameter in grad_entries
            )
        )
        optimizer.step()
        probe_delta = (probe.detach() - probe_before).abs().max()
        actor_changed = sum(
            parameter._version != actor_versions[name]
            for name, parameter in actor.named_parameters()
        )
        model_unchanged = all(
            parameter._version == model_versions[name]
            for name, parameter in model.named_parameters()
        )
        model_grad_free = all(parameter.grad is None for parameter in model.parameters())
        if float(probe_delta) <= 0.0 or actor_changed == 0:
            raise RuntimeError("Actor optimizer step did not change a weight")
        if not model_unchanged or not model_grad_free:
            raise RuntimeError("frozen ModelNet changed during Actor optimization")

        result_json = {
            "env": args.env_name,
            "game_id": args.game_id,
            "device": str(args.device),
            "gpu": (
                torch.cuda.get_device_name(args.device)
                if args.device.type == "cuda"
                else None
            ),
            "A": int(action_space.n),
            "obs_shape": list(obs_space.shape),
            "obs_dtype": str(obs_space.dtype),
            "obs_low": float(np.min(obs_space.low)),
            "obs_high": float(np.max(obs_space.high)),
            "frame_stack_n": frame_stack_n,
            "frame_ch": int(frame_ch),
            "dynamic_factorized_control": bool(
                flags.dynamic_factorized_control
            ),
            "voc_protocol": validated_voc_protocol,
            "voc_checkpoint_validation": voc_checkpoint_validation,
            "actor_checkpoint_public_validation": (
                actor_checkpoint_public_validation
            ),
            "model_checkpoint_public_validation": (
                model_checkpoint_public_validation
            ),
            "voc_checkpoint_resolved_identity": (
                voc_checkpoint_resolved_identity
            ),
            "schema6_final_bundle_validation": schema6_bundle_validation,
            "schema6_runtime_state": schema6_runtime_state,
            **(
                {
                    "schema7_final_bundle_validation": (
                        schema7_bundle_validation
                    ),
                    "schema7_runtime_state": schema7_runtime_state,
                }
                if schema7_bundle_validation is not None
                else {}
            ),
            **(
                {
                    "schema8_final_bundle_validation": (
                        schema8_bundle_validation
                    ),
                    "schema8_runtime_state": schema8_runtime_state,
                }
                if schema8_bundle_validation is not None
                else {}
            ),
            **(
                {
                    "schema9_final_bundle_validation": (
                        schema9_bundle_validation
                    ),
                    "schema9_runtime_state": schema9_runtime_state,
                }
                if schema9_bundle_validation is not None
                else {}
            ),
            **(
                {
                    "schema10_final_bundle_validation": (
                        schema10_bundle_validation
                    ),
                    "schema10_runtime_state": schema10_runtime_state,
                }
                if schema10_bundle_validation is not None
                else {}
            ),
            **(
                {
                    "schema11_final_bundle_validation": (
                        schema11_bundle_validation
                    ),
                    "schema11_runtime_state": schema11_runtime_state,
                }
                if schema11_bundle_validation is not None
                else {}
            ),
            **(
                {
                    "schema12_final_bundle_validation": (
                        schema12_bundle_validation
                    ),
                    "schema12_runtime_state": schema12_runtime_state,
                }
                if schema12_bundle_validation is not None
                else {}
            ),
            **(
                {
                    "schema13_final_bundle_validation": (
                        schema13_bundle_validation
                    ),
                    "schema13_runtime_state": schema13_runtime_state,
                    "schema13_telemetry": copy.deepcopy(
                        schema13_bundle_validation["telemetry"]
                    ),
                }
                if schema13_bundle_validation is not None
                else {}
            ),
            "model_state_projection": str(flags.model_state_projection),
            "model_state_range_loss_cost": float(
                flags.model_state_range_loss_cost
            ),
            "behavior_batch": list(observations.shape),
            "human_actions": actions.tolist(),
            "loss": float(result.loss.detach().cpu()),
            "nll": float(result.nll_sum.cpu()) / result.count,
            "count": int(result.count),
            "augmented_steps": int(result.augmented_steps),
            "root_carried_rate": float(result.root_carried.float().mean().cpu()),
            "actor_nonzero_grad_tensors": len(grad_entries),
            "actor_grad_norm": float(grad_norm.cpu()),
            "actor_probe": probe_name,
            "actor_probe_max_abs_update": float(probe_delta.cpu()),
            "actor_changed_parameter_count": int(actor_changed),
            "model_frozen_requires_grad": all(
                not parameter.requires_grad for parameter in model.parameters()
            ),
            "model_grad_free": bool(model_grad_free),
            "model_versions_unchanged": bool(model_unchanged),
            "peak_gpu_allocated_mib": (
                torch.cuda.max_memory_allocated(args.device) / 1024**2
                if args.device.type == "cuda"
                else 0.0
            ),
            "peak_gpu_reserved_mib": (
                torch.cuda.max_memory_reserved(args.device) / 1024**2
                if args.device.type == "cuda"
                else 0.0
            ),
            "live_step_reward_mean": float(np.asarray(live_reward).mean()),
        }
    finally:
        runner.close()
    if schema6_bundle_validation is not None:
        final_schema6_validation = (
            checkpoint_eval.validate_schema6_completed_bundle(
                args.checkpoint_dir
            )
        )
        if final_schema6_validation != schema6_bundle_validation:
            raise RuntimeError(
                "schema-6 completed-bundle evidence changed during smoke"
            )
        if (
            checkpoint_eval.checkpoint_hashes(args.checkpoint_dir)
            != atomic_loaded_hashes
        ):
            raise RuntimeError("schema-6 checkpoint changed during smoke")
    if schema7_bundle_validation is not None:
        final_schema7_validation = (
            checkpoint_eval.validate_schema7_completed_bundle(
                args.checkpoint_dir
            )
        )
        if final_schema7_validation != schema7_bundle_validation:
            raise RuntimeError(
                "schema-7 completed-bundle evidence changed during smoke"
            )
        if (
            checkpoint_eval.checkpoint_hashes(args.checkpoint_dir)
            != atomic_loaded_hashes
        ):
            raise RuntimeError("schema-7 checkpoint changed during smoke")
    if schema8_bundle_validation is not None:
        if (
            checkpoint_eval.checkpoint_hashes(args.checkpoint_dir)
            != atomic_loaded_hashes
        ):
            raise RuntimeError("schema-8 checkpoint changed during smoke")
    if schema9_bundle_validation is not None:
        final_schema9_validation = (
            checkpoint_eval.validate_schema9_completed_bundle(
                args.checkpoint_dir,
                completion_state=completion_state,
                config_payload=validated_config_payload,
                expected_config_sha256=validated_config_digest,
            )
        )
        if final_schema9_validation != schema9_bundle_validation:
            raise RuntimeError(
                "schema-9 completed-bundle evidence changed during smoke"
            )
        if (
            checkpoint_eval.checkpoint_hashes(args.checkpoint_dir)
            != atomic_loaded_hashes
        ):
            raise RuntimeError("schema-9 checkpoint changed during smoke")
    if schema10_bundle_validation is not None:
        final_schema10_validation = (
            checkpoint_eval.validate_schema10_completed_bundle(
                args.checkpoint_dir,
                completion_state=completion_state,
                config_payload=validated_config_payload,
                expected_config_sha256=validated_config_digest,
            )
        )
        if final_schema10_validation != schema10_bundle_validation:
            raise RuntimeError(
                "schema-10 completed-bundle evidence changed during smoke"
            )
        if (
            checkpoint_eval.checkpoint_hashes(args.checkpoint_dir)
            != atomic_loaded_hashes
        ):
            raise RuntimeError("schema-10 checkpoint changed during smoke")
    if schema11_bundle_validation is not None:
        final_schema11_validation = (
            checkpoint_eval.validate_schema11_completed_bundle(
                args.checkpoint_dir,
                completion_state=completion_state,
                config_payload=validated_config_payload,
                expected_config_sha256=validated_config_digest,
            )
        )
        if final_schema11_validation != schema11_bundle_validation:
            raise RuntimeError(
                "schema-11 completed-bundle evidence changed during smoke"
            )
        if (
            checkpoint_eval.checkpoint_hashes(args.checkpoint_dir)
            != atomic_loaded_hashes
        ):
            raise RuntimeError("schema-11 checkpoint changed during smoke")
    if schema12_bundle_validation is not None:
        final_schema12_validation = (
            checkpoint_eval.validate_schema12_completed_bundle(
                args.checkpoint_dir,
                completion_state=completion_state,
                config_payload=validated_config_payload,
                expected_config_sha256=validated_config_digest,
            )
        )
        if final_schema12_validation != schema12_bundle_validation:
            raise RuntimeError(
                "schema-12 completed-bundle evidence changed during smoke"
            )
        _require_schema12_smoke_ema_online_equality(
            args.checkpoint_dir,
            completion_state,
        )
        if (
            checkpoint_eval.checkpoint_hashes(args.checkpoint_dir)
            != atomic_loaded_hashes
        ):
            raise RuntimeError("schema-12 checkpoint changed during smoke")
    if schema13_bundle_validation is not None:
        final_schema13_validation = (
            checkpoint_eval.validate_schema13_completed_bundle(
                args.checkpoint_dir,
                completion_state=completion_state,
                config_payload=validated_config_payload,
                expected_config_sha256=validated_config_digest,
            )
        )
        if final_schema13_validation != schema13_bundle_validation:
            raise RuntimeError(
                "schema-13 completed-bundle evidence changed during smoke"
            )
        _require_schema13_smoke_ema_online_equality(
            args.checkpoint_dir,
            completion_state,
        )
        if (
            checkpoint_eval._schema13_checkpoint_hashes(
                args.checkpoint_dir, completion_state=completion_state
            )
            != atomic_loaded_hashes
        ):
            raise RuntimeError("schema-13 checkpoint changed during smoke")
    return result_json


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-name", required=True)
    parser.add_argument("--game-id", required=True, type=int)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--subjects", type=_parse_ids, default=(1,))
    parser.add_argument("--sessions", type=_parse_ids, default=(1, 2, 3))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--scored-length", type=int, default=4)
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--rec-t", type=int, default=None)
    parser.add_argument("--max-search-steps", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--model-unroll-len", type=int, default=None)
    parser.add_argument("--think-cost", type=float, default=None)
    parser.add_argument(
        "--dynamic-voc-mode",
        choices=("off", "shadow", "control"),
        default=None,
        help="Default for a fresh smoke is off; pass shadow/control explicitly",
    )
    parser.add_argument("--voc-loss-cost", type=float, default=None)
    parser.add_argument("--voc-gate-temperature", type=float, default=None)
    parser.add_argument("--voc-train-epsilon", type=float, default=None)
    parser.add_argument(
        "--voc-eval-stochastic",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--voc-dueling-q",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--voc-expected-gate-loss",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--voc-ema-gate-target",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--voc-gate-target-tau", type=float, default=None)
    parser.add_argument(
        "--voc-dedicated-gate",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--voc-soft-q-bce-gate",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--voc-gate-q-temperature", type=float, default=None)
    parser.add_argument(
        "--voc-gate-confidence-weighted",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--voc-gate-adam-beta1", type=float, default=None)
    parser.add_argument(
        "--voc-gate-param-align",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--voc-gate-param-align-coef", type=float, default=None)
    parser.add_argument(
        "--voc-gate-exact-projection",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--voc-gate-epsilon-greedy-execution",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--voc-gate-execution-epsilon", type=float, default=None)
    parser.add_argument(
        "--voc-actor-policy-version-barrier",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--voc-actor-policy-bundle-schema-version", type=int, default=None
    )
    parser.add_argument(
        "--voc-actor-policy-barrier-timeout-s", type=float, default=None
    )
    parser.add_argument(
        "--voc-actor-policy-ray-max-restarts", type=int, default=None
    )
    parser.add_argument(
        "--voc-actor-policy-ray-max-task-retries", type=int, default=None
    )
    parser.add_argument("--actor-amp-init-scale", type=float, default=None)
    parser.add_argument(
        "--voc-model-input-seal-schema-version", type=int, default=None
    )
    parser.add_argument("--voc-gate-learning-rate", type=float, default=None)
    parser.add_argument(
        "--voc-gate-grad-norm-clipping", type=float, default=None
    )
    parser.add_argument("--model-size-nn", type=int, default=None)
    parser.add_argument("--frame-stack-n", type=int, default=None)
    parser.add_argument(
        "--grayscale", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--tree-carry", action=argparse.BooleanOptionalAction, default=None
    )
    parsed = parser.parse_args(argv)
    parsed.data_root = parsed.data_root.expanduser().resolve()
    if parsed.config is not None:
        parsed.config = parsed.config.expanduser().resolve()
    if parsed.checkpoint_dir is not None:
        parsed.checkpoint_dir = parsed.checkpoint_dir.expanduser().resolve()
    if parsed.batch_size <= 0 or parsed.scored_length <= 0:
        parser.error("batch-size and scored-length must be positive")
    return parsed


def main(argv: Optional[Sequence[str]] = None) -> int:
    result = run_smoke(parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
