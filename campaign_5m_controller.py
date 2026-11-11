#!/usr/bin/env python3
"""Run one 100M-config campaign cell and publish its first verified 5M pair.

The training process intentionally keeps ``total_steps=100_000_000``.  This
controller watches the scratch run directory, selects the first Actor and
Model checkpoints in the half-open interval ``[5_000_000, 5_250_000)``, and
validates each file twice.  It then terminates the complete training process
group and atomically publishes a small canonical artifact directory.

This is deliberately *not* a normal Thinker completion.  No ``finish`` marker
is created or copied, and the milestone record calls the termination expected.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import dataclasses
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from typing import Any, Callable


MILESTONE_MIN_STEP = 5_000_000
MILESTONE_MAX_STEP_EXCLUSIVE = 5_250_000
MILESTONE_SCHEMA_VERSION = 1

TOTAL_STEPS = 100_000_000
SCHEDULE_TOTAL_STEPS = 100_000_000
CHECKPOINT_INTERVAL = 250_000
MAX_SEARCH_STEPS = 30
MAX_DEPTH = 30
MODEL_UNROLL_LEN = 30
REC_T = 20
BASE_SEED = 1

REFERENCE_FLAGS_KEY_COUNT = 227
REFERENCE_FLAGS_KEYSET_SHA256 = (
    "b06f2efbf43666c2878be3e27a963b62e6fa7c74aff58723e509f79afbd961e3"
)
REFERENCE_STATIC_FLAGS_SHA256 = (
    "611b8243a44cac75822b3ce32804364ae7a465bdbd7e6e4514785e6ca2104fd8"
)
SOURCE_MANIFEST_SHA256 = (
    "f32e83ba07d4b91bd454dcc84706e45bd1be2e33a66273ca3eeaa1ca5679b97f"
)
BEHAVIOR_MANIFEST_SHA256 = (
    "d976ee71a94fa7bbe17ca53e6fdaf42c9fc64923266c1ff2469565c866e4c65f"
)
VARIABLE_FLAG_FIELDS = frozenset(
    {
        "xpid",
        "ckpdir",
        "cmd",
        "name",
        "savedir",
        "icopro_data_path",
        "icopro_game_id",
        "icopro_subjects",
        "icopro_train_sessions",
        "icopro_holdout_sessions",
    }
)

_CANONICAL_XPID_RE = re.compile(
    r"^dynamic-imitation-(enduro|pong|spaceinvaders)-voc-"
    r"sub([1-6])-leavesession([1-4])$"
)
_JOB_KEY_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class CampaignError(RuntimeError):
    """A fail-closed campaign validation or lifecycle error."""


@dataclasses.dataclass(frozen=True)
class ExpectedRun:
    xpid: str
    game_label: str
    env_name: str
    game_id: int
    subject: int
    train_sessions: tuple[int, ...]
    holdout_session: int

    def __post_init__(self) -> None:
        match = _CANONICAL_XPID_RE.fullmatch(self.xpid)
        expected_xpid = (
            f"dynamic-imitation-{self.game_label}-voc-"
            f"sub{self.subject}-leavesession{self.holdout_session}"
        )
        if match is None or self.xpid != expected_xpid:
            raise CampaignError(
                f"non-canonical xpid {self.xpid!r}; expected {expected_xpid!r}"
            )
        game_contract = {
            "enduro": (0, "Enduro-v5"),
            "pong": (1, "Pong-v5"),
            "spaceinvaders": (2, "SpaceInvaders-v5"),
        }
        if self.game_label not in game_contract:
            raise CampaignError(f"unknown game label: {self.game_label!r}")
        if game_contract[self.game_label] != (self.game_id, self.env_name):
            raise CampaignError(
                "game label/id/environment disagree: "
                f"{self.game_label!r}/{self.game_id!r}/{self.env_name!r}"
            )
        if not 1 <= self.subject <= 6:
            raise CampaignError("subject must be in [1, 6]")
        if not 1 <= self.holdout_session <= 4:
            raise CampaignError("holdout session must be in [1, 4]")
        expected_train = tuple(
            session
            for session in range(1, 5)
            if session != self.holdout_session
        )
        if self.train_sessions != expected_train:
            raise CampaignError(
                f"train sessions {self.train_sessions!r} do not equal "
                f"leave-one-out complement {expected_train!r}"
            )

    @property
    def task_id(self) -> str:
        return (
            f"g{self.game_id}-{self.game_label}-s{self.subject:02d}-"
            f"h{self.holdout_session}"
        )


def parse_sessions(value: str | Sequence[int]) -> tuple[int, ...]:
    """Return a strict, ordered session tuple."""

    if isinstance(value, str):
        raw = value.split(",")
    else:
        raw = list(value)
    try:
        sessions = tuple(int(item) for item in raw)
    except (TypeError, ValueError) as error:
        raise CampaignError(f"invalid session list: {value!r}") from error
    if not sessions or any(session not in range(1, 5) for session in sessions):
        raise CampaignError(f"sessions must lie in [1, 4]: {sessions!r}")
    if tuple(sorted(set(sessions))) != sessions:
        raise CampaignError(
            f"sessions must be unique and ascending: {sessions!r}"
        )
    return sessions


def _checkpoint_pattern(role: str) -> re.Pattern[str]:
    if role not in {"actor", "model"}:
        raise CampaignError(f"unsupported checkpoint role: {role!r}")
    return re.compile(rf"^ckp_{role}\.tar_step_([0-9]+)$")


def checkpoint_step(path: Path, role: str) -> int:
    match = _checkpoint_pattern(role).fullmatch(path.name)
    if match is None:
        raise CampaignError(
            f"malformed {role} checkpoint filename: {path.name!r}"
        )
    return int(match.group(1))


def list_checkpoint_steps(run_dir: Path, role: str) -> list[tuple[int, Path]]:
    pattern = _checkpoint_pattern(role)
    found: list[tuple[int, Path]] = []
    if not run_dir.is_dir():
        return found
    for path in run_dir.iterdir():
        match = pattern.fullmatch(path.name)
        if match is not None and path.is_file():
            found.append((int(match.group(1)), path))
    return sorted(found, key=lambda item: item[0])


def first_checkpoint_in_window(
    run_dir: Path,
    role: str,
    minimum: int = MILESTONE_MIN_STEP,
    maximum_exclusive: int = MILESTONE_MAX_STEP_EXCLUSIVE,
) -> Path | None:
    if minimum < 0 or maximum_exclusive <= minimum:
        raise CampaignError("invalid checkpoint selection window")
    candidates = [
        path
        for step, path in list_checkpoint_steps(run_dir, role)
        if minimum <= step < maximum_exclusive
    ]
    return candidates[0] if candidates else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CampaignError("value is not strict canonical JSON") from error
    return hashlib.sha256(encoded).hexdigest()


def _stat_identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    if not path.is_file() or stat.st_size <= 0:
        raise CampaignError(f"checkpoint is not a non-empty regular file: {path}")
    return {
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CampaignError(f"{label} must be a mapping")
    return value


def _nonempty_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    result = _mapping(value, label=label)
    if not result:
        raise CampaignError(f"{label} must not be empty")
    return result


def _strict_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CampaignError(f"{label} must be an integer, got {value!r}")
    return int(value)


def _strict_bool(value: Any, *, label: str) -> bool:
    if type(value) is not bool:
        raise CampaignError(f"{label} must be a bool, got {value!r}")
    return value


def _finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CampaignError(f"{label} must be numeric, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise CampaignError(f"{label} must be finite, got {value!r}")
    return result


def _flag_sessions(value: Any, *, label: str) -> tuple[int, ...]:
    if isinstance(value, str):
        return parse_sessions(value)
    if isinstance(value, (list, tuple)):
        return parse_sessions(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return parse_sessions((value,))
    raise CampaignError(f"{label} has invalid session encoding: {value!r}")


def _resolve_flag_path(value: Any, *, cwd: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise CampaignError(f"{label} must be a non-empty path string")
    path = Path(value)
    if not path.is_absolute():
        path = cwd / path
    return path.resolve()


def _validate_run_flags(
    flags: Mapping[str, Any],
    expected: ExpectedRun,
    *,
    role: str,
    run_dir: Path,
    staged_behavior_root: Path,
    expected_train_cmd: str,
    command_cwd: Path,
) -> dict[str, Any]:
    key_count = len(flags)
    keyset_sha256 = _canonical_json_sha256(sorted(flags))
    if key_count != REFERENCE_FLAGS_KEY_COUNT:
        raise CampaignError(
            f"{role} flags have {key_count} keys, expected "
            f"{REFERENCE_FLAGS_KEY_COUNT}"
        )
    if keyset_sha256 != REFERENCE_FLAGS_KEYSET_SHA256:
        raise CampaignError(
            f"{role} flag keyset digest {keyset_sha256} does not match reference"
        )
    if not VARIABLE_FLAG_FIELDS.issubset(flags):
        missing = sorted(VARIABLE_FLAG_FIELDS.difference(flags))
        raise CampaignError(f"{role} flags lack variable fields: {missing}")
    static_flags = {
        name: value
        for name, value in flags.items()
        if name not in VARIABLE_FLAG_FIELDS
    }
    static_sha256 = _canonical_json_sha256(static_flags)
    if static_sha256 != REFERENCE_STATIC_FLAGS_SHA256:
        raise CampaignError(
            f"{role} static flag digest {static_sha256} does not match "
            f"reference {REFERENCE_STATIC_FLAGS_SHA256}"
        )
    exact_values: dict[str, Any] = {
        "xpid": expected.xpid,
        "name": expected.env_name,
        "dynamic_voc_mode": "control",
        "model_state_projection": "clamp",
    }
    integer_values = {
        "icopro_game_id": expected.game_id,
        "total_steps": TOTAL_STEPS,
        "schedule_total_steps": SCHEDULE_TOTAL_STEPS,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "max_search_steps": MAX_SEARCH_STEPS,
        "max_depth": MAX_DEPTH,
        "model_unroll_len": MODEL_UNROLL_LEN,
        "rec_t": REC_T,
        "base_seed": BASE_SEED,
    }
    float_values = {
        "icopro_margin_coef": 0.0,
        "icopro_pvp_coef": 0.0,
        "action_prior_weight": 0.0,
    }
    for name, wanted in exact_values.items():
        if flags.get(name) != wanted:
            raise CampaignError(
                f"{role} checkpoint flag {name}={flags.get(name)!r}, "
                f"expected {wanted!r}"
            )
    for name, wanted in integer_values.items():
        actual = _strict_int(flags.get(name), label=f"{role} flag {name}")
        if actual != wanted:
            raise CampaignError(
                f"{role} checkpoint flag {name}={actual}, expected {wanted}"
            )
    for name, wanted in float_values.items():
        actual = _finite_float(flags.get(name), label=f"{role} flag {name}")
        if not math.isclose(actual, wanted, rel_tol=0.0, abs_tol=1e-12):
            raise CampaignError(
                f"{role} checkpoint flag {name}={actual}, expected {wanted}"
            )
    if _flag_sessions(
        flags.get("icopro_subjects"), label=f"{role} flag icopro_subjects"
    ) != (expected.subject,):
        raise CampaignError(f"{role} checkpoint subject does not match campaign")
    if _flag_sessions(
        flags.get("icopro_train_sessions"),
        label=f"{role} flag icopro_train_sessions",
    ) != expected.train_sessions:
        raise CampaignError(
            f"{role} checkpoint train sessions do not match campaign"
        )
    if _flag_sessions(
        flags.get("icopro_holdout_sessions"),
        label=f"{role} flag icopro_holdout_sessions",
    ) != (expected.holdout_session,):
        raise CampaignError(
            f"{role} checkpoint holdout session does not match campaign"
        )
    if _strict_bool(flags.get("use_wandb"), label=f"{role} flag use_wandb") is not True:
        raise CampaignError(f"{role} checkpoint must have W&B enabled")
    path_contract = {
        "ckpdir": run_dir,
        "savedir": run_dir.parent,
        "icopro_data_path": staged_behavior_root,
    }
    for name, wanted in path_contract.items():
        actual = _resolve_flag_path(
            flags.get(name), cwd=command_cwd, label=f"{role} flag {name}"
        )
        if actual != wanted.resolve():
            raise CampaignError(
                f"{role} checkpoint flag {name} resolves to {actual}, "
                f"expected {wanted.resolve()}"
            )
    if flags.get("cmd") != expected_train_cmd:
        raise CampaignError(f"{role} checkpoint cmd does not match launched command")
    return {
        **exact_values,
        **integer_values,
        **float_values,
        "icopro_subjects": [expected.subject],
        "icopro_train_sessions": list(expected.train_sessions),
        "icopro_holdout_sessions": [expected.holdout_session],
        "use_wandb": True,
        "flag_key_count": key_count,
        "flag_keyset_sha256": keyset_sha256,
        "static_flags_sha256": static_sha256,
        "full_flags_sha256": _canonical_json_sha256(flags),
    }


def snapshot_checkpoint(
    path: Path,
    role: str,
    expected: ExpectedRun,
    *,
    run_dir: Path,
    staged_behavior_root: Path,
    expected_train_cmd: str,
    command_cwd: Path,
    expected_imitation_data_signature: str,
) -> dict[str, Any]:
    """Hash and deserialize one checkpoint without accepting a racing file."""

    step = checkpoint_step(path, role)
    if not MILESTONE_MIN_STEP <= step < MILESTONE_MAX_STEP_EXCLUSIVE:
        raise CampaignError(
            f"{role} checkpoint step {step} lies outside campaign window"
        )
    before = _stat_identity(path)
    sha256 = _sha256_file(path)
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise CampaignError(f"cannot deserialize {role} checkpoint {path}") from error
    after = _stat_identity(path)
    if before != after:
        raise CampaignError(f"{role} checkpoint changed while being validated: {path}")
    payload = _mapping(payload, label=f"{role} checkpoint payload")
    real_step = _strict_int(
        payload.get("real_step"), label=f"{role} checkpoint real_step"
    )
    if real_step != step:
        raise CampaignError(
            f"{role} filename step {step} disagrees with payload {real_step}"
        )
    internal_step = _strict_int(
        payload.get("step"), label=f"{role} checkpoint internal step"
    )
    if internal_step <= 0:
        raise CampaignError(f"{role} checkpoint internal step must be positive")
    flags = _mapping(payload.get("flags"), label=f"{role} checkpoint flags")
    validated_flags = _validate_run_flags(
        flags,
        expected,
        role=role,
        run_dir=run_dir,
        staged_behavior_root=staged_behavior_root,
        expected_train_cmd=expected_train_cmd,
        command_cwd=command_cwd,
    )
    if role == "actor":
        actor_values = {
            "dynamic_search": True,
            "dynamic_factorized_control": True,
            "dynamic_voc_mode": "control",
            "voc_gate_policy_schema_version": 3,
            "voc_actor_policy_terminal": False,
        }
        for name, wanted in actor_values.items():
            if payload.get(name) != wanted:
                raise CampaignError(
                    f"actor checkpoint {name}={payload.get(name)!r}, "
                    f"expected {wanted!r}"
                )
        imitation_root = _resolve_flag_path(
            payload.get("imitation_data_root"),
            cwd=command_cwd,
            label="actor imitation_data_root",
        )
        if imitation_root != staged_behavior_root.resolve():
            raise CampaignError("actor imitation_data_root does not match staged data")
        if payload.get("imitation_data_signature") != expected_imitation_data_signature:
            raise CampaignError(
                "actor imitation_data_signature does not match independently "
                "recomputed staged training data"
            )
        _require_sha256(
            str(payload.get("imitation_data_signature")),
            label="actor imitation_data_signature",
        )
        imitation_updates = _strict_int(
            payload.get("imitation_update_count"),
            label="actor imitation_update_count",
        )
        if imitation_updates <= 0:
            raise CampaignError("actor imitation_update_count must be positive")
        for name in (
            "actor_net_state_dict",
            "actor_net_optimizer_state_dict",
            "voc_optimizer_state_dict",
            "voc_gate_optimizer_state_dict",
            "voc_ema_gate_head_state_dict",
        ):
            _nonempty_mapping(payload.get(name), label=f"actor {name}")
        scheduler_names = (
            "actor_net_scheduler_state_dict",
            "voc_scheduler_state_dict",
            "voc_gate_scheduler_state_dict",
        )
    else:
        for name in (
            "model_net_state_dict",
            "model_net_optimizer_p_state_dict",
            "model_net_optimizer_m_state_dict",
        ):
            _nonempty_mapping(payload.get(name), label=f"model {name}")
        scheduler_names = (
            "model_net_scheduler_p_state_dict",
            "model_net_scheduler_m_state_dict",
        )
    for name in scheduler_names:
        scheduler = _nonempty_mapping(payload.get(name), label=f"{role} {name}")
        last_epoch = _strict_int(
            scheduler.get("last_epoch"), label=f"{role} {name}.last_epoch"
        )
        if last_epoch != real_step:
            raise CampaignError(
                f"{role} {name}.last_epoch={last_epoch}, expected {real_step}"
            )
    return {
        "role": role,
        "path": str(path.resolve()),
        "filename": path.name,
        "step": step,
        "internal_step": internal_step,
        "size": before["size"],
        "mtime_ns": before["mtime_ns"],
        "device": before["device"],
        "inode": before["inode"],
        "sha256": sha256,
        "validated_flags": validated_flags,
        "imitation_data_signature": (
            expected_imitation_data_signature if role == "actor" else None
        ),
    }


def snapshot_config(
    path: Path,
    expected: ExpectedRun,
    *,
    run_dir: Path,
    staged_behavior_root: Path,
    expected_train_cmd: str,
    command_cwd: Path,
) -> dict[str, Any]:
    before = _stat_identity(path)
    sha256 = _sha256_file(path)
    try:
        import yaml

        with path.open("r", encoding="utf-8") as stream:
            flags = yaml.safe_load(stream)
    except Exception as error:
        raise CampaignError(f"cannot parse config YAML {path}") from error
    after = _stat_identity(path)
    if before != after:
        raise CampaignError(f"config changed while being validated: {path}")
    flags = _mapping(flags, label="config YAML")
    validated_flags = _validate_run_flags(
        flags,
        expected,
        role="config",
        run_dir=run_dir,
        staged_behavior_root=staged_behavior_root,
        expected_train_cmd=expected_train_cmd,
        command_cwd=command_cwd,
    )
    return {
        "path": str(path.resolve()),
        "filename": path.name,
        "size": before["size"],
        "mtime_ns": before["mtime_ns"],
        "device": before["device"],
        "inode": before["inode"],
        "sha256": sha256,
        "validated_flags": validated_flags,
    }


def double_validate_checkpoint_pair(
    run_dir: Path,
    expected: ExpectedRun,
    *,
    stable_seconds: float,
    staged_behavior_root: Path,
    expected_train_cmd: str,
    command_cwd: Path,
    expected_imitation_data_signature: str,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    if not math.isfinite(stable_seconds) or stable_seconds < 0.0:
        raise CampaignError("stable_seconds must be finite and non-negative")
    selected: dict[str, Path] = {}
    for role in ("actor", "model"):
        path = first_checkpoint_in_window(run_dir, role)
        if path is None:
            raise CampaignError(f"no {role} checkpoint in milestone window")
        selected[role] = path
    config_path = run_dir / "config_c.yaml"
    if not config_path.is_file():
        raise CampaignError("scratch run lacks config_c.yaml")
    checkpoint_kwargs = {
        "run_dir": run_dir,
        "staged_behavior_root": staged_behavior_root,
        "expected_train_cmd": expected_train_cmd,
        "command_cwd": command_cwd,
        "expected_imitation_data_signature": expected_imitation_data_signature,
    }
    config_kwargs = {
        "run_dir": run_dir,
        "staged_behavior_root": staged_behavior_root,
        "expected_train_cmd": expected_train_cmd,
        "command_cwd": command_cwd,
    }
    first_time = time.time_ns()
    first = {
        role: snapshot_checkpoint(path, role, expected, **checkpoint_kwargs)
        for role, path in selected.items()
    }
    first_config = snapshot_config(config_path, expected, **config_kwargs)
    sleeper(stable_seconds)
    for role, path in selected.items():
        if first_checkpoint_in_window(run_dir, role) != path:
            raise CampaignError(
                f"an earlier {role} checkpoint appeared during stability wait"
            )
    second_time = time.time_ns()
    second = {
        role: snapshot_checkpoint(path, role, expected, **checkpoint_kwargs)
        for role, path in selected.items()
    }
    second_config = snapshot_config(config_path, expected, **config_kwargs)
    stable_keys = (
        "filename",
        "step",
        "internal_step",
        "size",
        "mtime_ns",
        "device",
        "inode",
        "sha256",
        "validated_flags",
        "imitation_data_signature",
    )
    for role in ("actor", "model"):
        for key in stable_keys:
            if first[role][key] != second[role][key]:
                raise CampaignError(
                    f"{role} checkpoint was not stable across validation: {key}"
                )
    if (
        second["actor"]["validated_flags"]["full_flags_sha256"]
        != second["model"]["validated_flags"]["full_flags_sha256"]
    ):
        raise CampaignError("actor and model embedded flags are not exactly equal")
    for key in stable_keys:
        if key == "step":
            continue
        if first_config.get(key) != second_config.get(key):
            raise CampaignError(
                f"config was not stable across validation: {key}"
            )
    return {
        "first_validation_unix_ns": first_time,
        "second_validation_unix_ns": second_time,
        "stable_seconds": stable_seconds,
        "artifacts": second,
        "config": second_config,
    }


def _session_members(session_id: int) -> list[int]:
    """Return current-UID processes that still belong to one Linux session."""

    members: list[int] = []
    current_uid = os.getuid()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != current_uid:
                continue
            raw = (entry / "stat").read_text(encoding="utf-8")
            tail = raw[raw.rfind(")") + 2 :].split()
            if len(tail) >= 4 and int(tail[3]) == session_id:
                members.append(int(entry.name))
        except (FileNotFoundError, PermissionError, ValueError):
            continue
    return sorted(members)


def _signal_session_members(session_id: int, sig: signal.Signals) -> None:
    for pid in _session_members(session_id):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def _terminate_process_group(
    process: subprocess.Popen[Any], *, grace_seconds: float
) -> dict[str, Any]:
    if not math.isfinite(grace_seconds) or grace_seconds <= 0.0:
        raise CampaignError("termination grace must be finite and positive")
    if process.poll() is not None:
        raise CampaignError(
            f"training child exited before expected milestone stop: {process.returncode}"
        )
    sent_signal = "SIGTERM"
    forced_kill = False
    session_id = process.pid
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    _signal_session_members(session_id, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None and not _session_members(session_id):
            break
        time.sleep(0.1)
    survivors_before_kill = _session_members(session_id)
    if survivors_before_kill:
        forced_kill = True
        sent_signal = "SIGTERM+SIGKILL"
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        _signal_session_members(session_id, signal.SIGKILL)
    try:
        returncode = process.wait(timeout=max(grace_seconds, 1.0))
    except subprocess.TimeoutExpired as error:
        raise CampaignError("training session leader survived SIGKILL") from error
    survivor_deadline = time.monotonic() + max(grace_seconds, 1.0)
    survivors_after_kill = _session_members(session_id)
    while survivors_after_kill and time.monotonic() < survivor_deadline:
        _signal_session_members(session_id, signal.SIGKILL)
        time.sleep(0.1)
        survivors_after_kill = _session_members(session_id)
    if survivors_after_kill:
        raise CampaignError(
            f"training session still has survivors after termination: "
            f"{survivors_after_kill}"
        )
    # A second empty scan makes the no-survivor publish gate explicit.
    time.sleep(0.1)
    late_survivors = _session_members(session_id)
    if late_survivors:
        raise CampaignError(
            f"training session repopulated after termination: {late_survivors}"
        )
    return {
        "expected": True,
        "session_id": session_id,
        "signal": sent_signal,
        "forced_kill": forced_kill,
        "survivors_before_kill": survivors_before_kill,
        "survivors_after_kill": late_survivors,
        "child_returncode": int(returncode),
    }


def _kill_process_group_best_effort(process: subprocess.Popen[Any] | None) -> None:
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    _signal_session_members(process.pid, signal.SIGKILL)
    try:
        process.wait(timeout=10.0)
    except Exception:
        pass


def validate_interaction_metrics(run_dir: Path, *, checkpoint_step: int) -> dict[str, Any]:
    logs_path = run_dir / "logs.csv"
    if not logs_path.is_file():
        raise CampaignError("scratch run lacks actor logs.csv")
    try:
        with logs_path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
    except (OSError, StopIteration, csv.Error) as error:
        raise CampaignError("cannot read actor logs.csv header") from error
    required = (
        "interaction/noop_action_index",
        "interaction/noop_supported",
        "interaction/real_action_count",
        "interaction/noop_count",
        "interaction/noop_frequency",
    )
    header = list(rows[0]) if rows else []
    missing = [name for name in ("real_step", *required) if name not in header]
    if missing:
        raise CampaignError(
            f"actor logs.csv lacks required interaction metrics: {missing}"
        )
    try:
        real_steps = [int(row["real_step"]) for row in rows]
    except (TypeError, ValueError) as error:
        raise CampaignError("actor logs.csv has a non-integer real_step") from error
    if any(right <= left for left, right in zip(real_steps, real_steps[1:])):
        raise CampaignError("actor logs.csv real_step is not strictly increasing")
    selected = [
        row for row, real_step in zip(rows, real_steps) if real_step == checkpoint_step
    ]
    if len(selected) != 1:
        raise CampaignError(
            f"actor logs.csv has {len(selected)} rows for checkpoint step "
            f"{checkpoint_step}, expected exactly one"
        )
    row = selected[0]
    try:
        noop_index = int(row["interaction/noop_action_index"])
        noop_supported = int(row["interaction/noop_supported"])
        real_action_count = int(row["interaction/real_action_count"])
        noop_count = int(row["interaction/noop_count"])
        noop_frequency = float(row["interaction/noop_frequency"])
    except (TypeError, ValueError) as error:
        raise CampaignError("actor logs.csv has malformed NOOP metrics") from error
    if noop_index != 0 or noop_supported != 1:
        raise CampaignError("runtime NOOP contract must be supported at action index 0")
    if real_action_count <= 0 or not 0 <= noop_count <= real_action_count:
        raise CampaignError("runtime NOOP counts are outside their valid range")
    expected_frequency = noop_count / real_action_count
    if not math.isfinite(noop_frequency) or not math.isclose(
        noop_frequency, expected_frequency, rel_tol=0.0, abs_tol=1e-12
    ):
        raise CampaignError("runtime NOOP frequency disagrees with its counts")
    return {
        "filename": logs_path.name,
        "sha256": _sha256_file(logs_path),
        "size": logs_path.stat().st_size,
        "required_columns": list(required),
        "checkpoint_row": {
            "real_step": checkpoint_step,
            "noop_action_index": noop_index,
            "noop_supported": noop_supported,
            "real_action_count": real_action_count,
            "noop_count": noop_count,
            "noop_frequency": noop_frequency,
        },
    }


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _rename_noreplace(source: Path, target: Path) -> None:
    """Atomically publish a directory without replacing an existing target."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        if target.exists() or target.is_symlink():
            raise CampaignError(f"canonical publish target already exists: {target}")
        os.rename(source, target)
        return
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    result = renameat2(
        at_fdcwd,
        os.fsencode(source),
        at_fdcwd,
        os.fsencode(target),
        rename_noreplace,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in (errno.EEXIST, errno.ENOTEMPTY):
            raise CampaignError(
                f"canonical publish target already exists: {target}"
            )
        raise OSError(error_number, os.strerror(error_number), str(target))


def _copy_if_present(source_dir: Path, staging_dir: Path, name: str) -> bool:
    source = source_dir / name
    if not source.is_file():
        return False
    destination = staging_dir / name
    shutil.copy2(source, destination)
    _fsync_file(destination)
    return True


def acquire_reservation(
    campaign_root: Path,
    publish_root: Path,
    expected: ExpectedRun,
    *,
    job_key: str,
) -> Path:
    if _JOB_KEY_RE.fullmatch(job_key) is None:
        raise CampaignError(f"invalid job key: {job_key!r}")
    campaign_root.mkdir(parents=True, exist_ok=True)
    publish_root.mkdir(parents=True, exist_ok=True)
    target = publish_root / expected.xpid
    if target.exists() or target.is_symlink():
        raise CampaignError(f"canonical run already exists: {target}")
    reservation_root = campaign_root / "reservations"
    reservation_root.mkdir(exist_ok=True)
    reservation = reservation_root / expected.xpid
    try:
        reservation.mkdir()
    except FileExistsError as error:
        raise CampaignError(
            f"canonical run is already reserved: {expected.xpid}"
        ) from error
    _write_json_atomic(
        reservation / "owner.json",
        {
            "job_key": job_key,
            "pid": os.getpid(),
            "task_id": expected.task_id,
            "xpid": expected.xpid,
        },
    )
    return reservation


def release_reservation(reservation: Path) -> None:
    owner = reservation / "owner.json"
    try:
        owner.unlink()
    except FileNotFoundError:
        pass
    try:
        reservation.rmdir()
    except FileNotFoundError:
        pass


def publish_canonical_run(
    *,
    run_dir: Path,
    campaign_root: Path,
    publish_root: Path,
    expected: ExpectedRun,
    validation: Mapping[str, Any],
    termination: Mapping[str, Any],
    source_root: Path,
    source_manifest: Path,
    behavior_root: Path,
    behavior_manifest: Path,
    launcher_path: Path,
    controller_path: Path,
    interaction_metrics: Mapping[str, Any],
    job_key: str,
    array_job_id: str,
    array_task_id: int,
) -> tuple[Path, dict[str, Any]]:
    if (run_dir / "finish").exists():
        raise CampaignError("100M-config milestone run unexpectedly has finish")
    target = publish_root / expected.xpid
    if target.exists() or target.is_symlink():
        raise CampaignError(f"canonical publish target already exists: {target}")
    milestones_root = campaign_root / "milestones"
    milestones_root.mkdir(parents=True, exist_ok=True)
    external_milestone = milestones_root / f"{expected.task_id}.json"
    if external_milestone.exists() or external_milestone.is_symlink():
        raise CampaignError(f"external milestone already exists: {external_milestone}")
    staging_root = campaign_root / "publish-staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    if staging_root.stat().st_dev != publish_root.stat().st_dev:
        raise CampaignError(
            "campaign staging and canonical publish root must share a filesystem"
        )
    staging = staging_root / f"{expected.xpid}.{job_key}"
    try:
        staging.mkdir()
    except FileExistsError as error:
        raise CampaignError(f"publish staging path already exists: {staging}") from error
    published = False
    try:
        artifacts = _mapping(validation.get("artifacts"), label="artifacts")
        published_artifacts: dict[str, dict[str, Any]] = {}
        for role in ("actor", "model"):
            artifact = _mapping(artifacts.get(role), label=f"{role} artifact")
            source = Path(str(artifact["path"]))
            destination = staging / str(artifact["filename"])
            shutil.copy2(source, destination)
            _fsync_file(destination)
            copied_sha = _sha256_file(destination)
            if copied_sha != artifact["sha256"]:
                raise CampaignError(
                    f"published {role} checkpoint hash changed during copy"
                )
            published_artifacts[role] = {
                "filename": destination.name,
                "step": int(artifact["step"]),
                "internal_step": int(artifact["internal_step"]),
                "size": int(artifact["size"]),
                "sha256": copied_sha,
                "validated_flags": artifact["validated_flags"],
            }
        config_source = Path(str(validation["config"]["path"]))
        config_destination = staging / "config_c.yaml"
        shutil.copy2(config_source, config_destination)
        _fsync_file(config_destination)
        config_sha256 = _sha256_file(config_destination)
        if config_sha256 != validation["config"]["sha256"]:
            raise CampaignError("published config hash changed during copy")
        copied_metadata = []
        for name in (
            "meta.json",
            "meta_model.json",
            "logs.csv",
            "logs_model.csv",
            "out.log",
            "out_model.log",
        ):
            if _copy_if_present(run_dir, staging, name):
                copied_metadata.append(name)
        source_manifest_copy = staging / "source.sha256"
        behavior_manifest_copy = staging / "behavior.sha256"
        shutil.copy2(source_manifest, source_manifest_copy)
        shutil.copy2(behavior_manifest, behavior_manifest_copy)
        _fsync_file(source_manifest_copy)
        _fsync_file(behavior_manifest_copy)
        milestone = {
            "schema_version": MILESTONE_SCHEMA_VERSION,
            "status": "VERIFIED_5M_STOPPED",
            "task_id": expected.task_id,
            "array_job_id": str(array_job_id),
            "array_task_id": int(array_task_id),
            "job_key": job_key,
            "logical": {
                "xpid": expected.xpid,
                "game_id": expected.game_id,
                "game_label": expected.game_label,
                "env_name": expected.env_name,
                "subject": expected.subject,
                "leave_session": expected.holdout_session,
                "train_sessions": list(expected.train_sessions),
            },
            "protocol": {
                "total_steps": TOTAL_STEPS,
                "schedule_total_steps": SCHEDULE_TOTAL_STEPS,
                "checkpoint_interval": CHECKPOINT_INTERVAL,
                "milestone_min_step": MILESTONE_MIN_STEP,
                "milestone_max_step_exclusive": MILESTONE_MAX_STEP_EXCLUSIVE,
                "max_search_steps": MAX_SEARCH_STEPS,
                "max_depth": MAX_DEPTH,
                "model_unroll_len": MODEL_UNROLL_LEN,
                "rec_t": REC_T,
                "dynamic_voc_mode": "control",
                "icopro_margin_coef": 0.0,
                "icopro_pvp_coef": 0.0,
                "action_prior_weight": 0.0,
                "base_seed": BASE_SEED,
                "reference_flags_key_count": REFERENCE_FLAGS_KEY_COUNT,
                "reference_flags_keyset_sha256": REFERENCE_FLAGS_KEYSET_SHA256,
                "reference_static_flags_sha256": REFERENCE_STATIC_FLAGS_SHA256,
            },
            "source": {
                "root": str(source_root.resolve()),
                "manifest": str(source_manifest.resolve()),
                "manifest_sha256": _sha256_file(source_manifest),
                "published_manifest": source_manifest_copy.name,
            },
            "behavior": {
                "root": str(behavior_root.resolve()),
                "manifest": str(behavior_manifest.resolve()),
                "manifest_sha256": _sha256_file(behavior_manifest),
                "published_manifest": behavior_manifest_copy.name,
                "imitation_data_signature": artifacts["actor"][
                    "imitation_data_signature"
                ],
            },
            "orchestration": {
                "launcher": str(launcher_path.resolve()),
                "launcher_sha256": _sha256_file(launcher_path),
                "controller": str(controller_path.resolve()),
                "controller_sha256": _sha256_file(controller_path),
            },
            "validation": {
                "first_validation_unix_ns": int(
                    validation["first_validation_unix_ns"]
                ),
                "second_validation_unix_ns": int(
                    validation["second_validation_unix_ns"]
                ),
                "stable_seconds": float(validation["stable_seconds"]),
            },
            "artifacts": published_artifacts,
            "config": {
                "filename": config_destination.name,
                "size": config_destination.stat().st_size,
                "sha256": config_sha256,
                "validated_flags": validation["config"]["validated_flags"],
            },
            "interaction_metrics": dict(interaction_metrics),
            "copied_metadata": copied_metadata,
            "termination": dict(termination),
            "finish_present": False,
            "published_at_unix_ns": time.time_ns(),
        }
        _write_json_atomic(staging / "milestone_5000000.json", milestone)
        for path in staging.iterdir():
            if path.is_file():
                _fsync_file(path)
        _fsync_dir(staging)
        _rename_noreplace(staging, target)
        published = True
        _fsync_dir(publish_root)
        try:
            _write_json_atomic(
                external_milestone,
                {
                    **milestone,
                    "canonical_path": str(target.resolve()),
                    "canonical_milestone_sha256": _sha256_file(
                        target / "milestone_5000000.json"
                    ),
                },
            )
            _fsync_dir(milestones_root)
        except (OSError, CampaignError) as error:
            # The canonical directory and its milestone are authoritative and
            # already durable.  A ledger repair can reconstruct this index.
            print(
                f"[campaign-warning] external milestone index needs repair: {error}",
                file=sys.stderr,
                flush=True,
            )
        return target, milestone
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def _manifest_digest(path: Path, *, label: str) -> str:
    if not path.is_absolute() or not path.is_file():
        raise CampaignError(f"{label} must be an absolute readable file: {path}")
    return _sha256_file(path)


def _require_sha256(value: str, *, label: str) -> str:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise CampaignError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _existing_checkpoint_missed_window(run_dir: Path, role: str) -> bool:
    steps = [step for step, _ in list_checkpoint_steps(run_dir, role)]
    return bool(steps) and max(steps) >= MILESTONE_MAX_STEP_EXCLUSIVE and not any(
        MILESTONE_MIN_STEP <= step < MILESTONE_MAX_STEP_EXCLUSIVE
        for step in steps
    )


def run_controller(args: argparse.Namespace) -> dict[str, Any]:
    expected = ExpectedRun(
        xpid=args.xpid,
        game_label=args.game_label,
        env_name=args.env_name,
        game_id=args.game_id,
        subject=args.subject,
        train_sessions=parse_sessions(args.train_sessions),
        holdout_session=args.holdout_session,
    )
    source_root = Path(args.source_root).resolve()
    behavior_root = Path(args.behavior_root).resolve()
    staged_behavior_root = Path(args.staged_behavior_root).resolve()
    source_manifest = Path(args.source_manifest).resolve()
    behavior_manifest = Path(args.behavior_manifest).resolve()
    if (
        not source_root.is_dir()
        or not behavior_root.is_dir()
        or not staged_behavior_root.is_dir()
    ):
        raise CampaignError(
            "source, behavior, and staged behavior roots must be directories"
        )
    source_manifest_digest = _manifest_digest(
        source_manifest, label="source manifest"
    )
    behavior_manifest_digest = _manifest_digest(
        behavior_manifest, label="behavior manifest"
    )
    if source_manifest_digest != SOURCE_MANIFEST_SHA256:
        raise CampaignError("source manifest digest does not match frozen campaign")
    if behavior_manifest_digest != BEHAVIOR_MANIFEST_SHA256:
        raise CampaignError("behavior manifest digest does not match frozen campaign")
    expected_imitation_data_signature = _require_sha256(
        args.expected_imitation_data_signature,
        label="expected imitation data signature",
    )
    launcher_path = Path(args.launcher_path).resolve()
    controller_path = Path(args.controller_path).resolve()
    if not launcher_path.is_file() or not controller_path.is_file():
        raise CampaignError("frozen launcher/controller paths must be regular files")
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    plan = {
        "task_id": expected.task_id,
        "xpid": expected.xpid,
        "run_dir": str(Path(args.run_dir).resolve()),
        "publish_dir": str((Path(args.publish_root).resolve() / expected.xpid)),
        "campaign_root": str(Path(args.campaign_root).resolve()),
        "source_manifest_sha256": source_manifest_digest,
        "behavior_manifest_sha256": behavior_manifest_digest,
        "command": command,
    }
    if args.dry_run:
        return {"dry_run": True, **plan}
    if not command:
        raise CampaignError("training command is required after --")
    run_dir = Path(args.run_dir).resolve()
    command_cwd = Path.cwd().resolve()
    publish_root = Path(args.publish_root).resolve()
    campaign_root = Path(args.campaign_root).resolve()
    expected_run_dir = (
        command_cwd.parent / "logs" / "thinker" / expected.xpid
    ).resolve()
    if run_dir != expected_run_dir:
        raise CampaignError(
            f"run dir {run_dir} does not match staged source layout {expected_run_dir}"
        )
    if len(command) < 2 or Path(command[0]).resolve() != Path(sys.executable).resolve():
        raise CampaignError("training command must start with this controller's Python")
    if command[1] != "train.py":
        raise CampaignError("training command must invoke staged train.py directly")
    expected_train_cmd = " ".join(command[1:])
    if shlex.split(expected_train_cmd) != command[1:]:
        raise CampaignError("training command contains unsupported shell-like tokens")
    reservation = acquire_reservation(
        campaign_root, publish_root, expected, job_key=args.job_key
    )
    process: subprocess.Popen[Any] | None = None
    published = False
    try:
        process = subprocess.Popen(command, start_new_session=True)
        while True:
            returncode = process.poll()
            if returncode is not None:
                raise CampaignError(
                    "training child exited before a verified milestone pair: "
                    f"{returncode}"
                )
            actor_path = first_checkpoint_in_window(run_dir, "actor")
            model_path = first_checkpoint_in_window(run_dir, "model")
            if actor_path is not None and model_path is not None:
                break
            for role in ("actor", "model"):
                if _existing_checkpoint_missed_window(run_dir, role):
                    raise CampaignError(
                        f"{role} checkpoint stream passed the selection window"
                    )
            time.sleep(args.poll_seconds)
        validation = double_validate_checkpoint_pair(
            run_dir,
            expected,
            stable_seconds=args.stable_seconds,
            staged_behavior_root=staged_behavior_root,
            expected_train_cmd=expected_train_cmd,
            command_cwd=command_cwd,
            expected_imitation_data_signature=expected_imitation_data_signature,
        )
        if process.poll() is not None:
            raise CampaignError(
                "training child exited during milestone stability validation"
            )
        termination = _terminate_process_group(
            process, grace_seconds=args.term_grace_seconds
        )
        if (run_dir / "finish").exists():
            raise CampaignError("externally stopped campaign cell created finish")
        # A final post-termination read proves the selected scratch artifacts
        # survived process-group teardown unchanged.
        for role in ("actor", "model"):
            final = snapshot_checkpoint(
                Path(validation["artifacts"][role]["path"]),
                role,
                expected,
                run_dir=run_dir,
                staged_behavior_root=staged_behavior_root,
                expected_train_cmd=expected_train_cmd,
                command_cwd=command_cwd,
                expected_imitation_data_signature=expected_imitation_data_signature,
            )
            if final["sha256"] != validation["artifacts"][role]["sha256"]:
                raise CampaignError(
                    f"{role} checkpoint changed during expected termination"
                )
        interaction_metrics = validate_interaction_metrics(
            run_dir,
            checkpoint_step=int(validation["artifacts"]["actor"]["step"]),
        )
        target, milestone = publish_canonical_run(
            run_dir=run_dir,
            campaign_root=campaign_root,
            publish_root=publish_root,
            expected=expected,
            validation=validation,
            termination=termination,
            source_root=source_root,
            source_manifest=source_manifest,
            behavior_root=behavior_root,
            behavior_manifest=behavior_manifest,
            launcher_path=launcher_path,
            controller_path=controller_path,
            interaction_metrics=interaction_metrics,
            job_key=args.job_key,
            array_job_id=args.array_job_id,
            array_task_id=args.array_task_id,
        )
        published = True
        return {
            "dry_run": False,
            "published": True,
            "target": str(target),
            "milestone": milestone,
        }
    finally:
        if not published:
            _kill_process_group_best_effort(process)
        release_reservation(reservation)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--publish-root", required=True)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--xpid", required=True)
    parser.add_argument("--game-label", required=True)
    parser.add_argument("--env-name", required=True)
    parser.add_argument("--game-id", type=int, required=True)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--train-sessions", required=True)
    parser.add_argument("--holdout-session", type=int, required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--behavior-root", required=True)
    parser.add_argument("--behavior-manifest", required=True)
    parser.add_argument("--staged-behavior-root", required=True)
    parser.add_argument("--expected-imitation-data-signature", required=True)
    parser.add_argument("--launcher-path", required=True)
    parser.add_argument("--controller-path", required=True)
    parser.add_argument("--job-key", required=True)
    parser.add_argument("--array-job-id", required=True)
    parser.add_argument("--array-task-id", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--stable-seconds", type=float, default=10.0)
    parser.add_argument("--term-grace-seconds", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ("poll_seconds", "stable_seconds", "term_grace_seconds"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")
    if args.poll_seconds <= 0.0 or args.term_grace_seconds <= 0.0:
        parser.error("poll and termination grace must be positive")
    try:
        result = run_controller(args)
    except CampaignError as error:
        print(f"[campaign-error] {error}", file=sys.stderr, flush=True)
        return 2
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
