#!/usr/bin/env python3
"""Stream Dynamic Thinker behavior traces without retaining a run in RAM.

Replay belongs to ``DynamicImitationRunner`` and storage belongs to
``thinker.behavior_trace``.  Heavy optional imports stay out of module import
and out of ``capture --dry-run``.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import contextlib
import copy
from dataclasses import asdict, dataclass
import hashlib
import importlib
import inspect
import io
import json
import multiprocessing as multiprocessing
import os
from pathlib import Path
import queue
import random
import re
import shutil
import sys
import threading
import time
import traceback
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence


CAPTURE_SCHEMA_VERSION = 2
TRACE_EVENT_SCHEMA_VERSION = 3
DETERMINISM_BACKEND_SCHEMA_VERSION = 1
SEED_PROTOCOL = "sha256/base-seed+source-sha256+episode-index/v2"
DEFAULT_SOURCE_PATTERN = "*.npz"
DEFAULT_QUEUE_SIZE = 2
DEFAULT_CHUNK_EVENTS = 256
MAX_INFLIGHT_WRITER_BYTES = 64 * 1024 * 1024
ZARR_INSTALL_HINT = "python -m pip install 'zarr==2.18.3' 'numcodecs==0.13.1'"
CPU_MODEL_MEMORY_MULTIPLIER = 3
CPU_WORKER_OVERHEAD_BYTES = 512 * 1024 * 1024
CPU_MEMORY_UTILISATION = 0.75
TIMING_MODE = "none"
CAPTURE_POLICY_MODES = ("checkpoint", "greedy", "sampled")


class VisualBehaviorError(RuntimeError):
    """A user-facing capture or trace error."""


class RuntimeDependencyError(VisualBehaviorError):
    """An optional runtime dependency is unavailable."""


@dataclass(frozen=True)
class StableFile:
    path: str
    size: int
    sha256: str
    device: int
    inode: int
    mtime_ns: int


@dataclass(frozen=True)
class CheckpointBundle:
    directory: str
    requested_step: Optional[int]
    config: StableFile
    actor: StableFile
    model: StableFile
    bundle_sha256: str
    milestone: Optional[StableFile] = None
    milestone_status: Optional[str] = None
    logical_xpid: Optional[str] = None
    source_xpid: Optional[str] = None
    actor_real_step: Optional[int] = None
    model_real_step: Optional[int] = None


@dataclass(frozen=True)
class SourceBundle:
    source_index: int
    source_id: str
    path: str
    size: int
    sha256: str
    device: int
    inode: int
    mtime_ns: int


@dataclass(frozen=True)
class WorkerSpec:
    rank: int
    device: str


@dataclass(frozen=True)
class ReplayPreprocessSpec:
    frame_stack_n: int
    target_size: tuple[int, int]
    grayscale: bool
    normalize: bool


@dataclass(frozen=True)
class SourceTask:
    source: SourceBundle
    episodes: tuple[Any, ...]
    preprocess: ReplayPreprocessSpec


@dataclass(frozen=True)
class _BehaviorFileView:
    path: Path
    subject: int
    session: int
    block: int
    game: int


@dataclass
class _EpisodeView:
    file_index: int
    episode_index: int
    observation_source_index: Any
    decision_times: Any
    actions: Any
    rewards: Any
    done: Any
    truncated: Any


@dataclass(frozen=True)
class CaptureConfig:
    checkpoint: CheckpointBundle
    data_root: str
    output_root: str
    scratch_root: Optional[str]
    base_seed: int
    queue_size: int
    chunk_events: int
    images: str
    tree_carry: bool
    policy_mode: str
    determinism_backend_fingerprint: Mapping[str, Any]
    resume: bool
    torch_threads_per_worker: int
    source_hash_workers: int
    episode_indices: tuple[int, ...]
    compression_level: int = 3
    store_predicted_xs: bool = True

    def __post_init__(self) -> None:
        if type(self.compression_level) is not int or not 0 <= self.compression_level <= 9:
            raise VisualBehaviorError("compression_level must be an integer in 0..9")
        if type(self.store_predicted_xs) is not bool:
            raise VisualBehaviorError("store_predicted_xs must be boolean")
        if not self.store_predicted_xs and self.images not in ("imagined", "all"):
            raise VisualBehaviorError("Omitting predicted_xs requires retaining imagined images")
        if self.policy_mode not in CAPTURE_POLICY_MODES:
            raise VisualBehaviorError(
                "policy_mode must be one of " + ", ".join(CAPTURE_POLICY_MODES)
            )
        _validate_determinism_backend_fingerprint(
            self.determinism_backend_fingerprint
        )


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _identity(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    if not path.is_file():
        raise VisualBehaviorError(f"expected a regular file: {path}")
    return (int(stat.st_dev), int(stat.st_ino), int(stat.st_size), int(stat.st_mtime_ns))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_bytes(
    record: StableFile,
    *,
    label: str,
) -> bytes:
    """Read the already-bound file generation once without a pathname reopen."""

    path = Path(record.path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise VisualBehaviorError(f"cannot open bound {label}: {path}: {error}") from error
    try:
        before = os.fstat(descriptor)
        identity = (
            int(before.st_dev), int(before.st_ino), int(before.st_size), int(before.st_mtime_ns)
        )
        expected = (record.device, record.inode, record.size, record.mtime_ns)
        if identity != expected:
            raise VisualBehaviorError(f"bound {label} identity changed: {path}")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 8 * 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        after_identity = (
            int(after.st_dev), int(after.st_ino), int(after.st_size), int(after.st_mtime_ns)
        )
        if after_identity != identity:
            raise VisualBehaviorError(f"bound {label} changed while being read: {path}")
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    if len(payload) != record.size or hashlib.sha256(payload).hexdigest() != record.sha256:
        raise VisualBehaviorError(f"bound {label} bytes disagree with their digest: {path}")
    return payload


def _flags_from_stable_config(record: StableFile, checkpoint_dir: str | Path) -> Any:
    """Reconstruct resolved flags exclusively from the bound config bytes."""

    try:
        yaml = importlib.import_module("yaml")
    except ModuleNotFoundError as error:
        raise RuntimeDependencyError("checkpoint loading requires PyYAML") from error
    payload = _stable_bytes(record, label="checkpoint config")
    try:
        mapping = yaml.safe_load(payload.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise VisualBehaviorError("checkpoint config is not valid UTF-8 YAML") from error
    if not isinstance(mapping, Mapping):
        raise VisualBehaviorError("checkpoint config must contain a mapping")
    from thinker import util

    flags = argparse.Namespace(**copy.deepcopy(dict(mapping)))
    flags = util.process_flags(flags)
    flags = util.process_flags_actor(flags)
    flags.ckpdir = str(Path(checkpoint_dir).expanduser().resolve())
    if not util.dynamic_search_enabled(flags):
        raise VisualBehaviorError("checkpoint is not a Dynamic Thinker checkpoint")
    return flags


def _raw_config_mapping(record: StableFile) -> Mapping[str, Any]:
    try:
        yaml = importlib.import_module("yaml")
    except ModuleNotFoundError as error:
        raise RuntimeDependencyError("checkpoint loading requires PyYAML") from error
    try:
        value = yaml.safe_load(
            _stable_bytes(record, label="checkpoint config").decode("utf-8")
        )
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise VisualBehaviorError("checkpoint config is not valid UTF-8 YAML") from error
    if not isinstance(value, Mapping):
        raise VisualBehaviorError("checkpoint config must contain a mapping")
    return value


def _torch_load_stable(record: StableFile, *, label: str, torch: Any) -> Any:
    """Load exactly the generation whose bytes were hashed during preflight."""

    payload = _stable_bytes(record, label=label)
    stream = io.BytesIO(payload)
    incidental_stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(incidental_stdout):
            return torch.load(stream, map_location="cpu", weights_only=False)
    except TypeError:
        # PyTorch versions predating ``weights_only`` still retain the same
        # stable-BytesIO safety property.
        stream.seek(0)
        with contextlib.redirect_stdout(incidental_stdout):
            return torch.load(stream, map_location="cpu")


def _checkpoint_flags_mapping(checkpoint: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    embedded = checkpoint.get("flags")
    if isinstance(embedded, Mapping):
        return embedded
    namespace = getattr(embedded, "__dict__", None)
    if isinstance(namespace, Mapping):
        return namespace
    raise VisualBehaviorError(f"{label} checkpoint lacks embedded source-run flags")


def _validate_loaded_checkpoint_payloads(
    actor: Any,
    model: Any,
    bundle: CheckpointBundle,
    *,
    config_xpid: Optional[str],
    config_mapping: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    requested = bundle.requested_step
    embedded_xpids: dict[str, str] = {}
    embedded_flags: dict[str, Mapping[str, Any]] = {}
    real_steps: dict[str, int] = {}
    for label, checkpoint in (("actor", actor), ("model", model)):
        if not isinstance(checkpoint, Mapping):
            raise VisualBehaviorError(f"{label} checkpoint payload is not a mapping")
        state_key = f"{label}_net_state_dict"
        state = checkpoint.get(state_key)
        if not isinstance(state, Mapping) or not state:
            raise VisualBehaviorError(
                f"{label} checkpoint lacks nonempty {state_key}"
            )
        try:
            step = int(checkpoint.get("real_step", -1))
        except (TypeError, ValueError) as error:
            raise VisualBehaviorError(
                f"{label} checkpoint real_step is invalid"
            ) from error
        if step <= 0:
            raise VisualBehaviorError(
                f"{label} checkpoint real_step must be positive, got {step}"
            )
        expected_step = getattr(bundle, f"{label}_real_step")
        if expected_step is None:
            expected_step = requested
        if expected_step is not None and step != expected_step:
            raise VisualBehaviorError(
                f"{label} checkpoint real_step={step}, expected {expected_step}"
            )
        real_steps[label] = step
        flags = _checkpoint_flags_mapping(checkpoint, label)
        embedded_flags[label] = flags
        xpid = str(flags.get("xpid", "")).strip()
        if not xpid:
            raise VisualBehaviorError(
                f"{label} checkpoint lacks embedded source-run xpid"
            )
        embedded_xpids[label] = xpid
    if embedded_xpids["actor"] != embedded_xpids["model"]:
        raise VisualBehaviorError("actor/model embedded source-run xpids disagree")
    if (bundle.actor_real_step is None or bundle.model_real_step is None) and real_steps["actor"] != real_steps["model"]:
        raise VisualBehaviorError(
            "actor/model checkpoint real_steps disagree: "
            f"{real_steps['actor']} != {real_steps['model']}"
        )
    if config_xpid is not None and config_xpid != embedded_xpids["actor"]:
        raise VisualBehaviorError(
            "config and checkpoint embedded source-run xpids disagree"
        )
    if bundle.source_xpid is not None and bundle.source_xpid != embedded_xpids["actor"]:
        raise VisualBehaviorError(
            "milestone source_run.xpid disagrees with embedded checkpoint identity"
        )
    actor_flags = dict(embedded_flags["actor"])
    model_flags = dict(embedded_flags["model"])
    if actor_flags != model_flags:
        raise VisualBehaviorError(
            "actor/model embedded flags differ outside their shared source-run identity"
        )
    if config_mapping is not None:
        relocation_only = {"ckpdir", "savedir"}
        frozen_config = {
            key: value
            for key, value in dict(config_mapping).items()
            if key not in relocation_only
        }
        frozen_embedded = {
            key: value
            for key, value in actor_flags.items()
            if key not in relocation_only
        }
        if frozen_config != frozen_embedded:
            missing = sorted(set(frozen_embedded) - set(frozen_config))
            extra = sorted(set(frozen_config) - set(frozen_embedded))
            changed = sorted(
                key
                for key in set(frozen_config) & set(frozen_embedded)
                if frozen_config[key] != frozen_embedded[key]
            )
            raise VisualBehaviorError(
                "config and embedded checkpoint flags differ outside relocation fields: "
                f"missing={missing}, extra={extra}, changed={changed}"
            )
    identity_keys = (
        "name",
        "icopro_game_id",
        "dynamic_search",
        "dynamic_factorized_control",
        "max_search_steps",
        "max_depth",
        "model_unroll_len",
        "rec_t",
        "frame_stack_n",
    )
    for key in identity_keys:
        actor_value = embedded_flags["actor"].get(key)
        model_value = embedded_flags["model"].get(key)
        if actor_value != model_value:
            raise VisualBehaviorError(
                f"actor/model embedded {key} settings disagree"
            )
        if config_mapping is not None and key in config_mapping:
            if actor_value != config_mapping[key]:
                raise VisualBehaviorError(
                    f"config and checkpoint embedded {key} settings disagree"
                )
    voc_mode = embedded_flags["actor"].get("dynamic_voc_mode", "off")
    if voc_mode != "off":
        ema = actor.get("voc_ema_gate_head_state_dict")
        if not isinstance(ema, Mapping) or not ema:
            raise VisualBehaviorError(
                "active VoC actor checkpoint lacks a nonempty EMA gate head"
            )
    return {
        "actor_real_step": real_steps["actor"],
        "model_real_step": real_steps["model"],
        "embedded_source_xpid": embedded_xpids["actor"],
        "game_id": int(embedded_flags["actor"].get("icopro_game_id", -1)),
        "environment_name": str(embedded_flags["actor"].get("name", "")),
    }


def validate_checkpoint_payload_bundle(bundle: CheckpointBundle) -> dict[str, Any]:
    """CPU-only stable payload validation used by both dry-run and capture."""

    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError as error:
        raise RuntimeDependencyError(
            "checkpoint payload validation requires PyTorch"
        ) from error
    config = _raw_config_mapping(bundle.config)
    config_xpid_value = config.get("xpid")
    config_xpid = (
        str(config_xpid_value).strip()
        if config_xpid_value is not None
        else None
    )
    actor = _torch_load_stable(bundle.actor, label="actor checkpoint", torch=torch)
    model = _torch_load_stable(bundle.model, label="model checkpoint", torch=torch)
    try:
        return _validate_loaded_checkpoint_payloads(
            actor,
            model,
            bundle,
            config_xpid=config_xpid,
            config_mapping=config,
        )
    finally:
        del actor, model


def stable_file(
    path: str | Path,
    *,
    polls: int = 2,
    interval_seconds: float = 0.05,
) -> StableFile:
    """Hash a file only when identity and content stay unchanged across reads."""

    path = Path(path).expanduser().resolve(strict=True)
    if polls < 1:
        raise ValueError("polls must be positive")
    previous: Optional[tuple[tuple[int, int, int, int], str]] = None
    for poll in range(polls):
        before = _identity(path)
        digest = _sha256_file(path)
        after = _identity(path)
        if before != after:
            raise VisualBehaviorError(f"file changed while it was read: {path}")
        current = (after, digest)
        if previous is not None and previous != current:
            raise VisualBehaviorError(f"file was not stable across reads: {path}")
        previous = current
        if poll + 1 < polls and interval_seconds > 0:
            time.sleep(interval_seconds)
    assert previous is not None
    (device, inode, size, mtime_ns), digest = previous
    return StableFile(
        path=str(path),
        size=size,
        sha256=digest,
        device=device,
        inode=inode,
        mtime_ns=mtime_ns,
    )


def _artifact_filename(marker: Mapping[str, Any], role: str) -> Optional[str]:
    artifacts = marker.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return None
    direct = artifacts.get(role)
    if isinstance(direct, Mapping) and isinstance(direct.get("filename"), str):
        return str(direct["filename"])
    prefix = f"ckp_{role}.tar_step_"
    matching = sorted(
        str(name)
        for name in artifacts
        if isinstance(name, str) and name.startswith(prefix)
    )
    return matching[-1] if matching else None


def _resolve_checkpoint_paths(
    checkpoint_dir: Path,
    requested_step: Optional[int],
    marker: Mapping[str, Any],
) -> tuple[Path, Path]:
    if requested_step is not None:
        actor_name = f"ckp_actor.tar_step_{requested_step}"
        model_name = f"ckp_model.tar_step_{requested_step}"
    else:
        actor_name = "ckp_actor.tar"
        model_name = "ckp_model.tar"

    actor = checkpoint_dir / actor_name
    model = checkpoint_dir / model_name
    if not actor.is_file() and marker:
        marker_name = _artifact_filename(marker, "actor")
        if marker_name:
            actor = checkpoint_dir / marker_name
    if not model.is_file() and marker:
        marker_name = _artifact_filename(marker, "model")
        if marker_name:
            model = checkpoint_dir / marker_name
    if not actor.is_file() or not model.is_file():
        raise VisualBehaviorError(
            "checkpoint bundle is incomplete; expected actor/model files "
            f"under {checkpoint_dir}"
        )
    return actor, model


def _milestone_artifact_record(
    marker: Mapping[str, Any], role: str, filename: str
) -> Optional[tuple[int, str]]:
    artifacts = marker.get("artifacts")
    record: Any = None
    if isinstance(artifacts, Mapping):
        record = artifacts.get(role)
        if not isinstance(record, Mapping):
            record = artifacts.get(filename)
    if not isinstance(record, Mapping) and role == "config":
        record = marker.get("config")
    if not isinstance(record, Mapping):
        return None
    declared_name = record.get("filename")
    if declared_name is not None and declared_name != filename:
        raise VisualBehaviorError(
            f"milestone {role} filename mismatch: {declared_name!r} != {filename!r}"
        )
    size = record.get("size", record.get("size_bytes"))
    digest = record.get("sha256")
    if type(size) is not int or size <= 0:
        raise VisualBehaviorError(f"milestone {role} size is invalid")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise VisualBehaviorError(f"milestone {role} SHA-256 is invalid")
    return int(size), digest


def _resolve_10m_bundle(root: Path, *, polls: int, interval: float) -> CheckpointBundle:
    """Use the published asynchronous 10M pair, never guess by newest filename."""
    marker = stable_file(root / "bundle.json", polls=polls, interval_seconds=interval)
    data = json.loads(_stable_bytes(marker, label="10M bundle").decode("utf-8"))
    if data.get("nominal_milestone") != 10_000_000:
        raise VisualBehaviorError("bundle.json must bind the 10M milestone")
    steps = {}
    records = {}
    for role in ("config", "actor", "model"):
        artifact = data.get("files", {}).get(role, {})
        name = Path(str(artifact.get("path", ""))).name
        if role == "config":
            expected_name = "config_c.yaml"
        else:
            step = data.get(role + "_real_step")
            if type(step) is not int or not 10_000_000 <= step < 10_100_000:
                raise VisualBehaviorError(f"Invalid published 10M {role} step")
            steps[role] = step
            expected_name = f"ckp_{role}.tar_step_{step}"
        if name != expected_name:
            raise VisualBehaviorError(f"10M {role} filename/step mismatch")
        records[role] = stable_file(root / name, polls=polls, interval_seconds=interval)
        if records[role].sha256 != artifact.get("sha256"):
            raise VisualBehaviorError(f"10M {role} SHA-256 mismatch")
    if data.get("real_step") != steps["actor"] or data.get("pair_step_difference") != steps["model"] - steps["actor"]:
        raise VisualBehaviorError("10M asynchronous pairing metadata disagrees")
    logical_root = root.parent.parent if root.parent.name == "milestones" else root
    source_xpid = str(_raw_config_mapping(records["config"]).get("xpid", "")).strip()
    if not source_xpid:
        raise VisualBehaviorError("10M config has no source-run xpid")
    payload = dict(nominal_milestone=10_000_000, marker_sha256=marker.sha256,
                   files={k: v.sha256 for k, v in records.items()}, steps=steps,
                   logical_xpid=logical_root.name, source_xpid=source_xpid)
    return CheckpointBundle(
        directory=str(root), requested_step=10_000_000,
        config=records["config"], actor=records["actor"], model=records["model"],
        bundle_sha256=hashlib.sha256(_canonical_json(payload)).hexdigest(),
        milestone=marker, milestone_status="HASH_VERIFIED_10M_ASYNC_PAIR",
        logical_xpid=logical_root.name, source_xpid=source_xpid,
        actor_real_step=steps["actor"], model_real_step=steps["model"],
    )


def resolve_checkpoint_bundle(
    checkpoint_dir: str | Path,
    *,
    requested_step: Optional[int] = None,
    stable_polls: int = 2,
    stable_interval_seconds: float = 0.05,
) -> CheckpointBundle:
    root = Path(checkpoint_dir).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise VisualBehaviorError(f"checkpoint path is not a directory: {root}")
    if requested_step == 10_000_000 and (root / "milestones/10000000/bundle.json").is_file():
        root = root / "milestones/10000000"
    if (root / "bundle.json").is_file():
        if requested_step not in (None, 10_000_000):
            raise VisualBehaviorError("10M bundle disagrees with requested step")
        return _resolve_10m_bundle(root, polls=stable_polls, interval=stable_interval_seconds)
    milestone_path = root / "milestone_5000000.json"
    milestone_record: Optional[StableFile] = None
    marker: Mapping[str, Any] = {}
    published_steps: dict[str, int] = {}
    if milestone_path.is_file():
        milestone_record = stable_file(
            milestone_path,
            polls=stable_polls,
            interval_seconds=stable_interval_seconds,
        )
        try:
            decoded = json.loads(
                _stable_bytes(milestone_record, label="milestone marker").decode("utf-8")
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise VisualBehaviorError(f"invalid milestone marker {milestone_path}: {error}") from error
        if not isinstance(decoded, Mapping):
            raise VisualBehaviorError("milestone marker must contain one JSON object")
        marker = decoded
        marker_step = marker.get("target_real_step")
        # Campaign publication binds asynchronously saved actor/model steps,
        # each in the declared 5M milestone interval, by filename and SHA256.
        # Historical exact-step canonical bundles retain their strict rule.
        if marker_step is None and marker.get("status") == "VERIFIED_5M_STOPPED":
            protocol = marker.get("protocol", {})
            if (protocol.get("milestone_min_step"), protocol.get("milestone_max_step_exclusive")) != (5_000_000, 5_250_000):
                raise VisualBehaviorError("Unrecognized campaign milestone interval")
            marker_step = 5_000_000
            for role in ("actor", "model"):
                artifact = marker.get("artifacts", {}).get(role, {})
                step = artifact.get("step")
                if type(step) is not int or not 5_000_000 <= step < 5_250_000:
                    raise VisualBehaviorError(f"Invalid published {role} milestone step")
                if artifact.get("filename") != f"ckp_{role}.tar_step_{step}":
                    raise VisualBehaviorError(f"Published {role} filename/step mismatch")
                published_steps[role] = step
        if type(marker_step) is not int or marker_step <= 0:
            raise VisualBehaviorError("milestone target_real_step is invalid")
        if requested_step is None:
            requested_step = marker_step
        elif marker_step != requested_step:
            raise VisualBehaviorError(
                f"milestone step {marker_step} does not match requested {requested_step}"
            )
        marker_status = marker.get("status")
        marker_kind = marker.get("kind")
        if marker_step != 5_000_000:
            raise VisualBehaviorError(
                "milestone_5000000.json must bind target_real_step=5000000"
            )
        if marker_status is not None and marker_status != "VERIFIED_5M_STOPPED":
            raise VisualBehaviorError("milestone marker status is contradictory")
        if (
            marker_kind is not None
            and marker_kind != "dynamic-imitation-canonical-milestone"
        ):
            raise VisualBehaviorError("milestone marker kind is contradictory")
        if marker_status is None and marker_kind is None:
            raise VisualBehaviorError(
                "milestone marker has no recognized verification discriminator"
            )

    actor_path, model_path = _resolve_checkpoint_paths(root, requested_step, marker)
    files = [
        stable_file(
            path, polls=stable_polls, interval_seconds=stable_interval_seconds
        )
        for path in (root / "config_c.yaml", actor_path, model_path)
    ]
    if marker:
        for role, record in (("config", files[0]), ("actor", files[1]), ("model", files[2])):
            expected = _milestone_artifact_record(marker, role, Path(record.path).name)
            if expected is None:
                raise VisualBehaviorError(f"milestone does not bind its {role} checkpoint")
            if expected != (record.size, record.sha256):
                raise VisualBehaviorError(
                    f"milestone {role} size/SHA-256 disagrees with stable checkpoint bytes"
                )
    logical = marker.get("logical", {}) if marker else {}
    source_run = marker.get("source_run", {}) if marker else {}
    logical_xpid = logical.get("xpid") if isinstance(logical, Mapping) else None
    source_xpid = source_run.get("xpid") if isinstance(source_run, Mapping) else None
    if logical_xpid is not None and logical_xpid != root.name:
        raise VisualBehaviorError(
            f"milestone logical xpid {logical_xpid!r} does not match directory {root.name!r}"
        )
    payload = {
        "requested_step": requested_step,
        "config": {"name": Path(files[0].path).name, "sha256": files[0].sha256},
        "actor": {"name": Path(files[1].path).name, "sha256": files[1].sha256},
        "model": {"name": Path(files[2].path).name, "sha256": files[2].sha256},
        "milestone_sha256": milestone_record.sha256 if milestone_record else None,
        "logical_xpid": logical_xpid,
        "source_xpid": source_xpid,
    }
    return CheckpointBundle(
        directory=str(root),
        requested_step=requested_step,
        config=files[0],
        actor=files[1],
        model=files[2],
        bundle_sha256=hashlib.sha256(_canonical_json(payload)).hexdigest(),
        milestone=milestone_record,
        milestone_status=(
            str(marker.get("status", marker.get("kind"))) if marker else None
        ),
        logical_xpid=str(logical_xpid) if logical_xpid is not None else None,
        source_xpid=str(source_xpid) if source_xpid is not None else None,
        actor_real_step=published_steps.get("actor"),
        model_real_step=published_steps.get("model"),
    )


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _selector_paths(root: Path, selector: str, pattern: str) -> Iterator[Path]:
    value = Path(selector).expanduser()
    if value.is_absolute():
        candidates = [value]
    elif any(character in selector for character in "*?["):
        candidates = list(root.glob(selector))
    else:
        candidates = [root / value]
    for candidate in candidates:
        candidate = candidate.resolve(strict=True)
        if not _is_within(candidate, root):
            raise VisualBehaviorError(f"source selector escaped data root: {candidate}")
        if candidate.is_dir():
            yield from candidate.rglob(pattern)
        elif candidate.is_file():
            yield candidate


def _path_game_id(path: Path) -> Optional[int]:
    match = re.search(
        r"(?:^|[/_-])game[-_]?0*(\d+)(?=$|[/_.-])",
        path.as_posix().lower(),
    )
    return None if match is None else int(match.group(1))


def enumerate_source_bundles(
    data_root: str | Path,
    *,
    selectors: Sequence[str] = (),
    pattern: str = DEFAULT_SOURCE_PATTERN,
    stable_polls: int = 2,
    stable_interval_seconds: float = 0.05,
    hash_workers: str | int = "auto",
    game_id: Optional[int] = None,
) -> list[SourceBundle]:
    root = Path(data_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise VisualBehaviorError(f"data root is not a directory: {root}")
    candidates: Iterable[Path]
    if selectors:
        selected: list[Path] = []
        for selector in selectors:
            selected.extend(_selector_paths(root, selector, pattern))
        candidates = selected
    else:
        candidates = root.rglob(pattern)
    by_path: dict[Path, str] = {}
    for candidate in candidates:
        path = candidate.expanduser().resolve(strict=True)
        if not path.is_file() or path.suffix.lower() != ".npz":
            continue
        if game_id is not None and _path_game_id(path) != int(game_id):
            continue
        if not _is_within(path, root):
            raise VisualBehaviorError(f"source is outside data root: {path}")
        by_path[path] = path.relative_to(root).as_posix()
    ordered = sorted(by_path.items(), key=lambda item: item[1])
    if not ordered:
        raise VisualBehaviorError(f"no NPZ sources matched under {root}")

    worker_count = resolve_hash_workers(hash_workers, len(ordered))

    def bind(item: tuple[Path, str]) -> StableFile:
        return stable_file(
            item[0],
            polls=stable_polls,
            interval_seconds=stable_interval_seconds,
        )

    # executor.map preserves input order, so scheduling never changes source
    # indices even though independent files are hashed concurrently.
    with ThreadPoolExecutor(
        max_workers=worker_count, thread_name_prefix="behavior-source-hash"
    ) as executor:
        records = list(executor.map(bind, ordered))

    bundles: list[SourceBundle] = []
    for source_index, ((path, source_id), record) in enumerate(zip(ordered, records)):
        bundles.append(
            SourceBundle(
                source_index=source_index,
                source_id=source_id,
                path=record.path,
                size=record.size,
                sha256=record.sha256,
                device=record.device,
                inode=record.inode,
                mtime_ns=record.mtime_ns,
            )
        )
    by_digest: dict[str, list[str]] = {}
    for source in bundles:
        by_digest.setdefault(source.sha256, []).append(source.source_id)
    duplicate_digests = {
        digest: ids for digest, ids in by_digest.items() if len(ids) > 1
    }
    if duplicate_digests:
        raise VisualBehaviorError(
            "selected source files contain byte-identical SHA-256 identities: "
            f"{duplicate_digests}"
        )
    return bundles


def validate_source_snapshots(
    sources: Sequence[SourceBundle], *, hash_workers: int
) -> None:
    """Re-hash every replay input before publishing a completion marker."""

    worker_count = max(1, min(int(hash_workers), len(sources)))

    def bind(source: SourceBundle) -> StableFile:
        return stable_file(source.path, polls=2, interval_seconds=0.05)

    with ThreadPoolExecutor(
        max_workers=worker_count, thread_name_prefix="behavior-source-final-hash"
    ) as executor:
        rebound = list(executor.map(bind, sources))
    for source, observed in zip(sources, rebound):
        expected = (
            source.device,
            source.inode,
            source.size,
            source.mtime_ns,
            source.sha256,
        )
        actual = (
            observed.device,
            observed.inode,
            observed.size,
            observed.mtime_ns,
            observed.sha256,
        )
        if actual != expected:
            raise VisualBehaviorError(
                f"behavior source bytes changed during capture: {source.path}"
            )


def resolve_hash_workers(value: str | int, source_count: int) -> int:
    if source_count < 1:
        raise ValueError("source_count must be positive")
    if str(value) == "auto":
        return max(1, min(source_count, max(1, _available_cpu_count() // 2), 8))
    try:
        count = int(value)
    except (TypeError, ValueError) as error:
        raise VisualBehaviorError(
            "--cpu-workers must be auto or a positive integer"
        ) from error
    if count < 1:
        raise VisualBehaviorError("--cpu-workers must be positive")
    return min(source_count, count)


def _episode_seed_from_identity(
    base_seed: int, source_sha256: str, episode_index: int
) -> int:
    if episode_index < 0:
        raise ValueError("episode_index must be nonnegative")
    if not re.fullmatch(r"[0-9a-f]{64}", str(source_sha256)):
        raise ValueError("source_sha256 must be a lowercase SHA-256 digest")
    payload = {
        "protocol": SEED_PROTOCOL,
        "base_seed": int(base_seed),
        "source_sha256": str(source_sha256),
        "episode_index": int(episode_index),
    }
    return int.from_bytes(hashlib.sha256(_canonical_json(payload)).digest()[:8], "big") & (
        2**63 - 1
    )


def episode_seed(base_seed: int, source: SourceBundle, episode_index: int) -> int:
    return _episode_seed_from_identity(base_seed, source.sha256, episode_index)


def episode_store_id(source: SourceBundle, episode_index: int) -> str:
    if episode_index < 0:
        raise ValueError("episode_index must be nonnegative")
    return f"{source.sha256}-e{int(episode_index):05d}"


def _available_cpu_count() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def _physical_core_count_from_cpuinfo(
    text: str, affinity: set[int]
) -> Optional[int]:
    records: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in text.splitlines() + [""]:
        if not line.strip():
            if current:
                records.append(current)
                current = {}
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            current[key.strip()] = value.strip()
    cores: set[tuple[str, str]] = set()
    for record in records:
        try:
            processor = int(record["processor"])
            physical = record["physical id"]
            core = record["core id"]
        except (KeyError, ValueError):
            continue
        if processor in affinity:
            cores.add((physical, core))
    return len(cores) or None


def _available_physical_core_count() -> int:
    try:
        affinity = set(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = set(range(max(1, os.cpu_count() or 1)))
    try:
        parsed = _physical_core_count_from_cpuinfo(
            Path("/proc/cpuinfo").read_text(encoding="ascii"), affinity
        )
        if parsed is not None:
            return parsed
    except OSError:
        pass
    # Conservative SMT fallback when topology is unavailable.
    return max(1, len(affinity) // 2)


def _available_memory_bytes() -> int:
    """Return an OS-level available-memory estimate without optional packages."""

    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        # This is only a last-resort sizing fallback.  One worker remains safe
        # and deterministic when the host cannot expose an availability value.
        return CPU_WORKER_OVERHEAD_BYTES


def cpu_worker_sizing(
    source_count: int,
    *,
    cpu_count: Optional[int] = None,
    available_memory_bytes: Optional[int] = None,
    checkpoint_bytes: int = 0,
) -> dict[str, int | float]:
    if source_count < 1:
        raise ValueError("source_count must be positive")
    cpus = (
        _available_physical_core_count()
        if cpu_count is None
        else max(1, int(cpu_count))
    )
    available = (
        _available_memory_bytes()
        if available_memory_bytes is None
        else max(1, int(available_memory_bytes))
    )
    checkpoint_bytes = max(0, int(checkpoint_bytes))
    estimated_per_worker = (
        checkpoint_bytes * CPU_MODEL_MEMORY_MULTIPLIER + CPU_WORKER_OVERHEAD_BYTES
    )
    memory_budget = max(1, int(available * CPU_MEMORY_UTILISATION))
    cpu_limit = max(1, cpus // 4)
    memory_limit = max(1, memory_budget // max(1, estimated_per_worker))
    selected = max(1, min(source_count, cpu_limit, memory_limit))
    return {
        "available_physical_core_count": cpus,
        "available_memory_bytes": available,
        "memory_utilisation": CPU_MEMORY_UTILISATION,
        "checkpoint_bytes": checkpoint_bytes,
        "model_memory_multiplier": CPU_MODEL_MEMORY_MULTIPLIER,
        "fixed_worker_overhead_bytes": CPU_WORKER_OVERHEAD_BYTES,
        "estimated_bytes_per_worker": estimated_per_worker,
        "cpu_limit": cpu_limit,
        "memory_limit": memory_limit,
        "selected_workers": selected,
    }


def auto_cpu_workers(
    source_count: int,
    cpu_count: Optional[int] = None,
    *,
    available_memory_bytes: Optional[int] = None,
    checkpoint_bytes: int = 0,
) -> int:
    return int(
        cpu_worker_sizing(
            source_count,
            cpu_count=cpu_count,
            available_memory_bytes=available_memory_bytes,
            checkpoint_bytes=checkpoint_bytes,
        )["selected_workers"]
    )


def resolve_torch_threads(value: str, worker_count: int) -> int:
    if value == "auto":
        return max(1, min(4, _available_cpu_count() // max(1, worker_count)))
    try:
        count = int(value)
    except ValueError as error:
        raise VisualBehaviorError(
            "--torch-threads-per-worker must be auto or a positive integer"
        ) from error
    if count < 1:
        raise VisualBehaviorError("--torch-threads-per-worker must be positive")
    return count


def _visible_cuda_indices() -> Optional[list[int]]:
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is None:
        return None
    value = value.strip()
    if not value or value == "-1":
        return []
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    return list(range(len(tokens)))


def _seal_determinism_backend_fingerprint(
    payload: Mapping[str, Any]
) -> dict[str, Any]:
    sealed = dict(payload)
    sealed["fingerprint_sha256"] = hashlib.sha256(
        _canonical_json(sealed)
    ).hexdigest()
    return sealed


def _validate_determinism_backend_fingerprint(
    fingerprint: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(fingerprint, Mapping):
        raise VisualBehaviorError("determinism backend fingerprint must be a mapping")
    value = dict(fingerprint)
    claimed_sha = value.pop("fingerprint_sha256", None)
    backend = value.get("backend")
    if backend == "cpu":
        required = {"schema_version", "backend"}
    elif backend == "cuda":
        required = {
            "schema_version",
            "backend",
            "cuda_device_name",
            "cuda_compute_capability",
        }
        name = value.get("cuda_device_name")
        capability = value.get("cuda_compute_capability")
        if not isinstance(name, str) or not name.strip():
            raise VisualBehaviorError(
                "CUDA determinism fingerprint lacks a device model"
            )
        if (
            not isinstance(capability, list)
            or len(capability) != 2
            or any(type(component) is not int or component < 0 for component in capability)
        ):
            raise VisualBehaviorError(
                "CUDA determinism fingerprint has an invalid compute capability"
            )
    else:
        raise VisualBehaviorError(
            "determinism backend fingerprint must select cpu or cuda"
        )
    if set(value) != required:
        raise VisualBehaviorError(
            "determinism backend fingerprint has unexpected or missing fields"
        )
    if (
        type(value.get("schema_version")) is not int
        or value["schema_version"] != DETERMINISM_BACKEND_SCHEMA_VERSION
    ):
        raise VisualBehaviorError("determinism backend fingerprint schema is invalid")
    actual_sha = hashlib.sha256(_canonical_json(value)).hexdigest()
    if claimed_sha != actual_sha:
        raise VisualBehaviorError("determinism backend fingerprint SHA-256 is invalid")
    return {**value, "fingerprint_sha256": actual_sha}


def resolve_determinism_backend_fingerprint(
    workers: Sequence[WorkerSpec], *, dry_run: bool = False, torch_module: Any = None
) -> dict[str, Any]:
    """Bind numeric replay semantics without binding worker count or indices."""

    if not workers:
        raise VisualBehaviorError("cannot fingerprint an empty worker topology")
    cpu_workers = [worker for worker in workers if worker.device == "cpu"]
    cuda_workers = [
        worker for worker in workers if re.fullmatch(r"cuda:[0-9]+", worker.device)
    ]
    if len(cpu_workers) == len(workers):
        return _seal_determinism_backend_fingerprint(
            {
                "schema_version": DETERMINISM_BACKEND_SCHEMA_VERSION,
                "backend": "cpu",
            }
        )
    if len(cuda_workers) != len(workers):
        raise VisualBehaviorError(
            "worker topology mixes CPU, CUDA, or malformed device identifiers"
        )
    if torch_module is None:
        try:
            torch_module = importlib.import_module("torch")
        except (ImportError, ModuleNotFoundError) as error:
            raise RuntimeDependencyError(
                ("CUDA --dry-run" if dry_run else "CUDA capture")
                + " requires PyTorch to bind the device model and compute capability"
            ) from error
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not callable(getattr(cuda, "is_available", None)):
        raise RuntimeDependencyError(
            "PyTorch lacks CUDA introspection required for the determinism fingerprint"
        )
    try:
        available = bool(cuda.is_available())
    except BaseException as error:
        raise RuntimeDependencyError(
            "CUDA availability could not be checked for the determinism fingerprint"
        ) from error
    if not available:
        raise RuntimeDependencyError(
            "CUDA workers were requested but CUDA is unavailable; refusing an "
            "unbound determinism fingerprint"
        )
    profiles: set[tuple[str, int, int]] = set()
    for worker in cuda_workers:
        index = int(worker.device.split(":", 1)[1])
        try:
            properties = cuda.get_device_properties(index)
            name = str(properties.name).strip()
            major = int(properties.major)
            minor = int(properties.minor)
        except BaseException as error:
            raise RuntimeDependencyError(
                f"cannot inspect {worker.device} for the determinism fingerprint"
            ) from error
        if not name or major < 0 or minor < 0:
            raise RuntimeDependencyError(
                f"{worker.device} returned an invalid CUDA architecture"
            )
        profiles.add((name, major, minor))
    if len(profiles) != 1:
        raise VisualBehaviorError(
            "selected CUDA workers have heterogeneous device models or compute "
            f"capabilities: {sorted(profiles)!r}"
        )
    name, major, minor = next(iter(profiles))
    return _seal_determinism_backend_fingerprint(
        {
            "schema_version": DETERMINISM_BACKEND_SCHEMA_VERSION,
            "backend": "cuda",
            "cuda_device_name": name,
            "cuda_compute_capability": [major, minor],
        }
    )


def _validate_workers_match_determinism_backend(
    workers: Sequence[WorkerSpec], fingerprint: Mapping[str, Any]
) -> None:
    validated = _validate_determinism_backend_fingerprint(fingerprint)
    worker_backends = {
        "cpu" if worker.device == "cpu" else "cuda"
        if re.fullmatch(r"cuda:[0-9]+", worker.device)
        else "invalid"
        for worker in workers
    }
    if worker_backends != {validated["backend"]}:
        raise VisualBehaviorError(
            "worker topology disagrees with the determinism backend fingerprint"
        )


def resolve_worker_specs(
    device_spec: str,
    *,
    source_count: int,
    cpu_workers: str,
    dry_run: bool,
    checkpoint_bytes: int = 0,
) -> list[WorkerSpec]:
    value = str(device_spec).strip().lower()
    cuda_indices: list[int] = []
    if value == "cpu":
        pass
    elif value == "auto":
        visible = _visible_cuda_indices()
        if visible is not None:
            cuda_indices = visible
        else:
            try:
                torch = importlib.import_module("torch")
                cuda_indices = list(range(int(torch.cuda.device_count())))
            except (ImportError, ModuleNotFoundError) as error:
                raise RuntimeDependencyError(
                    "--devices auto requires PyTorch to resolve the capture backend; "
                    "use --devices cpu for a CUDA-independent dry-run"
                ) from error
    else:
        try:
            cuda_indices = [int(token) for token in value.split(",")]
        except ValueError as error:
            raise VisualBehaviorError(
                "--devices must be auto, cpu, or comma-separated CUDA indices"
            ) from error
        if not cuda_indices or min(cuda_indices) < 0 or len(set(cuda_indices)) != len(
            cuda_indices
        ):
            raise VisualBehaviorError("CUDA device indices must be unique and nonnegative")

    if cuda_indices:
        # Scheduling is source-bundle-granular.  Extra device owners would only
        # duplicate checkpoint memory and sit idle when fewer sources exist.
        cuda_indices = cuda_indices[:source_count]
        return [
            WorkerSpec(rank=rank, device=f"cuda:{index}")
            for rank, index in enumerate(cuda_indices)
        ]
    if cpu_workers == "auto":
        count = auto_cpu_workers(
            source_count, checkpoint_bytes=checkpoint_bytes
        )
    else:
        try:
            count = int(cpu_workers)
        except ValueError as error:
            raise VisualBehaviorError("--cpu-workers must be auto or a positive integer") from error
        if count < 1:
            raise VisualBehaviorError("--cpu-workers must be positive")
        count = min(count, source_count)
    return [WorkerSpec(rank=rank, device="cpu") for rank in range(count)]


def build_capture_plan(
    config: CaptureConfig,
    sources: Sequence[SourceBundle],
    workers: Sequence[WorkerSpec],
    source_tasks: Optional[Sequence[SourceTask]] = None,
) -> dict[str, Any]:
    _validate_workers_match_determinism_backend(
        workers, config.determinism_backend_fingerprint
    )
    checkpoint_semantics = {
        "requested_step": config.checkpoint.requested_step,
        "bundle_sha256": config.checkpoint.bundle_sha256,
        "logical_xpid": config.checkpoint.logical_xpid,
        "source_xpid": config.checkpoint.source_xpid,
        "config": {
            "name": Path(config.checkpoint.config.path).name,
            "size": config.checkpoint.config.size,
            "sha256": config.checkpoint.config.sha256,
        },
        "actor": {
            "name": Path(config.checkpoint.actor.path).name,
            "size": config.checkpoint.actor.size,
            "sha256": config.checkpoint.actor.sha256,
        },
        "model": {
            "name": Path(config.checkpoint.model.path).name,
            "size": config.checkpoint.model.size,
            "sha256": config.checkpoint.model.sha256,
        },
        "milestone_sha256": (
            config.checkpoint.milestone.sha256
            if config.checkpoint.milestone is not None
            else None
        ),
    }
    semantic: dict[str, Any] = {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "kind": "dynamic-thinker-behavior-capture-semantics",
        "checkpoint": checkpoint_semantics,
        "seed_protocol": SEED_PROTOCOL,
        "base_seed": config.base_seed,
        "chunk_events": config.chunk_events,
        "images": config.images,
        "compression_level": config.compression_level,
        "store_predicted_xs": config.store_predicted_xs,
        "tree_carry": config.tree_carry,
        "policy_mode": config.policy_mode,
        "determinism_backend_fingerprint": dict(
            config.determinism_backend_fingerprint
        ),
        "timing_mode": TIMING_MODE,
        "trace_event_schema_version": TRACE_EVENT_SCHEMA_VERSION,
        "episode_indices": list(config.episode_indices),
        "source_count": len(sources),
        "sources": [
            {
                "source_index": source.source_index,
                "source_id": source.source_id,
                "size": source.size,
                "sha256": source.sha256,
            }
            for source in sources
        ],
        "preprocessing": (
            {
                "frame_stack_n": source_tasks[0].preprocess.frame_stack_n,
                "target_size": list(source_tasks[0].preprocess.target_size),
                "grayscale": source_tasks[0].preprocess.grayscale,
                "normalize": source_tasks[0].preprocess.normalize,
            }
            if source_tasks
            else "deferred-until-live-preflight"
        ),
        "episode_counts": (
            [len(task.episodes) for task in source_tasks]
            if source_tasks is not None
            else "deferred-until-live-preflight"
        ),
    }
    semantic_sha256 = hashlib.sha256(_canonical_json(semantic)).hexdigest()
    cpu_sizing = cpu_worker_sizing(
        len(sources),
        checkpoint_bytes=config.checkpoint.actor.size + config.checkpoint.model.size,
    )
    execution: dict[str, Any] = {
        "data_root": config.data_root,
        "output_root": config.output_root,
        "scratch_root": config.scratch_root,
        "max_inflight_chunks": config.queue_size,
        "source_hash_workers": config.source_hash_workers,
        "torch_threads_per_worker": config.torch_threads_per_worker,
        "worker_count": len(workers),
        "workers": [asdict(worker) for worker in workers],
        "cpu_worker_sizing": cpu_sizing,
        "scheduling": {
            "unit": "source_bundle",
            "episode_split": False,
            "start_method": "spawn",
            "one_process_per_cuda_device": True,
            "writer_queue_max_chunks": config.queue_size,
            "writer_queue_max_events": config.queue_size * config.chunk_events,
            "writer_queue_max_bytes": MAX_INFLIGHT_WRITER_BYTES,
            "cpu_overlap": {
                "source_hash_pool_workers": config.source_hash_workers,
                "frame_prefetch_threads_per_worker": 1,
                "max_prefetched_frames_per_worker": 1,
                "npz_image_extraction": "streamed-in-frame-prefetch-thread",
                "zarr_writer_threads_per_worker": 1,
            },
        },
    }
    payload: dict[str, Any] = {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "kind": "dynamic-thinker-behavior-capture-plan",
        "semantic_sha256": semantic_sha256,
        "semantic": semantic,
        "execution": execution,
    }
    payload["plan_sha256"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return payload


def _load_behavior_trace() -> Any:
    try:
        # Importing the historical Thinker package may trigger pygame's
        # support banner.  Keep inspect/dry-run JSON stdout machine-readable.
        with contextlib.redirect_stdout(io.StringIO()):
            module = importlib.import_module("thinker.behavior_trace")
    except ModuleNotFoundError as error:
        if error.name in {"zarr", "numcodecs", "asciitree", "fasteners"}:
            raise RuntimeDependencyError(
                f"behavior trace storage requires Zarr; install it with: {ZARR_INSTALL_HINT}"
            ) from error
        raise RuntimeDependencyError(
            "thinker.behavior_trace is unavailable; install/build the current Thinker source"
        ) from error
    required = (
        "BehaviorEpisodeSource",
        "EpisodeZarrWriter",
        "open_video_stat",
        "export_legacy",
        "validate_episode_store",
    )
    missing = [name for name in required if not callable(getattr(module, name, None))]
    if missing:
        raise RuntimeDependencyError(
            "thinker.behavior_trace lacks required public APIs: " + ", ".join(missing)
        )
    return module


def _call_supported(function: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(*args, **kwargs)
    accepts_var_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if not accepts_var_kwargs:
        kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return function(*args, **kwargs)


def _freeze_event(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _freeze_event(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_freeze_event(item) for item in value)
    if isinstance(value, list):
        return [_freeze_event(item) for item in value]
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
        numpy = getattr(value, "numpy", None)
        if callable(numpy):
            value = numpy()
    copy = getattr(value, "copy", None)
    if callable(copy) and not isinstance(value, (str, bytes, bytearray)):
        try:
            return copy()
        except TypeError:
            pass
    return value


def _retained_nbytes(value: Any) -> int:
    """Conservatively estimate bytes retained by one frozen queue payload."""

    seen: set[int] = set()

    def visit(item: Any) -> int:
        identity = id(item)
        if identity in seen:
            return 0
        seen.add(identity)
        if isinstance(item, Mapping):
            return sys.getsizeof(item) + sum(
                sys.getsizeof(key) + visit(member) for key, member in item.items()
            )
        if isinstance(item, (tuple, list)):
            return sys.getsizeof(item) + sum(visit(member) for member in item)
        if isinstance(item, (bytes, bytearray, memoryview)):
            return sys.getsizeof(item)
        nbytes = getattr(item, "nbytes", None)
        if nbytes is not None:
            try:
                return sys.getsizeof(item) + int(nbytes)
            except (TypeError, ValueError):
                pass
        return sys.getsizeof(item)

    return max(1, visit(value))


class _BoundedWriter:
    """A one-owner writer thread with bounded producer backpressure."""

    def __init__(
        self,
        trace_api: Any,
        *,
        partial_path: Path,
        final_path: Path,
        metadata: Mapping[str, Any],
        num_actions: int,
        tree_rep_schema: Mapping[str, Any],
        chunk_events: int,
        images: str,
        queue_size: int,
        max_inflight_bytes: int = MAX_INFLIGHT_WRITER_BYTES,
        compression_level: int = 3,
    ) -> None:
        self._api = trace_api
        self._partial_path = partial_path
        self._final_path = final_path
        self._metadata = dict(metadata)
        self._num_actions = int(num_actions)
        self._tree_rep_schema = dict(tree_rep_schema)
        self._chunk_events = int(chunk_events)
        self._images = str(images)
        self._compression_level = int(compression_level)
        self._max_inflight_chunks = int(queue_size)
        self._max_inflight_bytes = int(max_inflight_bytes)
        if self._max_inflight_bytes < 1:
            raise ValueError("max_inflight_bytes must be positive")
        self._queue: queue.Queue[tuple[str, Any, int]] = queue.Queue(
            maxsize=max(1, self._max_inflight_chunks * self._chunk_events)
        )
        self._byte_condition = threading.Condition()
        self._pending_bytes = 0
        self._peak_pending_bytes = 0
        self._outcome: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)
        self._thread = threading.Thread(
            target=self._run,
            name=f"zarr-writer-{final_path.stem}",
            daemon=True,
        )
        self._closed = False
        self._thread.start()

    def _run(self) -> None:
        sink = None
        try:
            sink = _call_supported(
                self._api.EpisodeZarrWriter,
                self._partial_path,
                metadata=self._metadata,
                num_actions=self._num_actions,
                tree_rep_schema=self._tree_rep_schema,
                chunk_events=self._chunk_events,
                images=self._images,
                compression_level=self._compression_level,
            )
            while True:
                operation, payload, retained_bytes = self._queue.get()
                try:
                    if operation == "event":
                        sink.write_event(payload)
                    elif operation == "finalize":
                        self._outcome.put((True, sink.finalize(self._final_path)))
                        return
                    elif operation == "abort":
                        sink.abort()
                        self._outcome.put((True, None))
                        return
                    else:
                        raise RuntimeError(f"unknown writer operation {operation!r}")
                finally:
                    if retained_bytes:
                        self._release_bytes(retained_bytes)
        except BaseException as error:
            if sink is not None:
                try:
                    sink.abort()
                except BaseException:
                    pass
            self._outcome.put((False, (error, traceback.format_exc())))

    def _reserve_bytes(self, retained_bytes: int) -> None:
        if retained_bytes > self._max_inflight_bytes:
            raise VisualBehaviorError(
                "one frozen trace event exceeds the configured writer queue byte limit: "
                f"{retained_bytes} > {self._max_inflight_bytes} bytes"
            )
        with self._byte_condition:
            while self._pending_bytes + retained_bytes > self._max_inflight_bytes:
                if not self._thread.is_alive():
                    self._raise_outcome()
                self._byte_condition.wait(timeout=0.25)
            self._pending_bytes += retained_bytes
            self._peak_pending_bytes = max(
                self._peak_pending_bytes, self._pending_bytes
            )

    def _release_bytes(self, retained_bytes: int) -> None:
        with self._byte_condition:
            self._pending_bytes -= retained_bytes
            self._byte_condition.notify_all()

    def _put(self, item: tuple[str, Any], *, retained_bytes: int = 0) -> None:
        if retained_bytes:
            self._reserve_bytes(retained_bytes)
        enqueued = False
        try:
            while True:
                if not self._thread.is_alive():
                    break
                try:
                    self._queue.put((*item, retained_bytes), timeout=0.25)
                    enqueued = True
                    return
                except queue.Full:
                    continue
        finally:
            if retained_bytes and not enqueued:
                self._release_bytes(retained_bytes)
        self._raise_outcome()

    def _raise_outcome(self) -> None:
        try:
            ok, value = self._outcome.get_nowait()
        except queue.Empty as error:
            raise VisualBehaviorError("writer stopped without an outcome") from error
        if ok:
            raise VisualBehaviorError("writer closed before capture completed")
        original, formatted = value
        raise VisualBehaviorError(f"writer failed: {original}\n{formatted}") from original

    def write_event(self, event: Mapping[str, Any]) -> None:
        if self._closed:
            raise VisualBehaviorError("cannot write to a closed episode writer")
        frozen = _freeze_event(event)
        self._put(
            ("event", frozen), retained_bytes=_retained_nbytes(frozen)
        )

    def finalize(self) -> Any:
        if self._closed:
            raise VisualBehaviorError("episode writer is already closed")
        self._put(("finalize", None))
        self._closed = True
        self._thread.join()
        ok, value = self._outcome.get()
        if not ok:
            original, formatted = value
            raise VisualBehaviorError(f"writer failed: {original}\n{formatted}") from original
        return value

    def abort(self) -> None:
        if self._closed:
            return
        if self._thread.is_alive():
            self._put(("abort", None))
            self._thread.join()
        self._closed = True
        if not self._outcome.empty():
            self._outcome.get_nowait()


class _CanonicalEventSink:
    """Fail closed unless the runner emits a fixed numeric trace contract."""

    _REQUIRED = {
        "initial_root",
        "forced_root_reset",
        "phase_pre",
        "search_computation",
        "real_transition",
        "human_action",
        "human_action_valid",
        "actor_policy",
        "actor_policy_valid",
        "tree_reps_raw",
        "tree_reps_raw_valid",
        "predicted_xs",
        "predicted_xs_valid",
        "step_times",
        "step_times_valid",
    }

    _REAL_IMAGE_FIELDS = {
        "pre_real_states", "pre_real_states_valid",
        "post_real_states", "post_real_states_valid",
    }
    _IMAGINED_IMAGE_FIELDS = {
        "pre_xs", "pre_xs_valid", "post_xs", "post_xs_valid"
    }

    def __init__(
        self,
        output: _BoundedWriter,
        *,
        images: str,
        source_frame_indices: Any,
        decision_times: Any,
        grayscale: bool,
        observation_provider: Any,
        store_predicted_xs: bool = True,
    ) -> None:
        self.output = output
        self.images = images
        self.store_predicted_xs = bool(store_predicted_xs)
        self.source_frame_indices = source_frame_indices
        self.decision_times = decision_times
        self.grayscale = bool(grayscale)
        self.observation_provider = observation_provider
        self.event_count = 0
        self._schema: Optional[dict[str, tuple[str, tuple[int, ...]]]] = None
        self._real_image_template: Any = None
        self._imagined_image_template: Any = None

    @staticmethod
    def _numeric_schema(event: Mapping[str, Any]) -> dict[str, tuple[str, tuple[int, ...]]]:
        numpy = importlib.import_module("numpy")
        schema: dict[str, tuple[str, tuple[int, ...]]] = {}

        def visit(mapping: Mapping[str, Any], prefix: str = "") -> None:
            for key in sorted(mapping):
                if not isinstance(key, str) or not key:
                    raise VisualBehaviorError("trace event keys must be nonempty strings")
                name = f"{prefix}/{key}" if prefix else key
                value = mapping[key]
                if isinstance(value, Mapping):
                    visit(value, name)
                    continue
                if value is None or isinstance(value, (str, bytes)):
                    raise VisualBehaviorError(
                        f"runner emitted nonnumeric trace field {name!r}; use sentinel+valid"
                    )
                frozen = _freeze_event(value)
                array = numpy.asarray(frozen)
                if array.dtype.kind not in "biuf" or array.dtype.hasobject:
                    raise VisualBehaviorError(
                        f"runner emitted nonnumeric trace field {name!r}: {array.dtype}"
                    )
                schema[name] = (array.dtype.str, tuple(int(item) for item in array.shape))

        visit(event)
        return schema

    def _filter_images(self, event: Mapping[str, Any], cursor: int) -> dict[str, Any]:
        output = dict(event)
        initial = self._scalar_bool(event.get("initial_root"), "initial_root")
        real_transition = self._scalar_bool(
            event.get("real_transition"), "real_transition"
        )
        raw_real = (
            output.get("pre_real_states")
            if initial
            else output.get("post_real_states")
        )
        raw_imagined = output.get("predicted_xs")
        real_valid = output.get(
            "pre_real_states_valid" if initial else "post_real_states_valid",
            True,
        )
        imagined_valid = output.get("predicted_xs_valid", False)
        for name in (
            self._REAL_IMAGE_FIELDS
            | self._IMAGINED_IMAGE_FIELDS
            | {"real_image", "real_image_valid", "imagined_image", "imagined_image_valid"}
        ):
            output.pop(name, None)
        numpy = importlib.import_module("numpy")
        if raw_imagined is None:
            raise VisualBehaviorError(
                "runner omitted always-on predicted_xs; refusing to fabricate a model state"
            )
        predicted_xs = numpy.asarray(_freeze_event(raw_imagined))
        if predicted_xs.dtype != numpy.dtype("float32"):
            raise VisualBehaviorError(
                f"predicted_xs must be float32, got {predicted_xs.dtype}"
            )
        output["predicted_xs"] = numpy.ascontiguousarray(predicted_xs)
        output["predicted_xs_valid"] = numpy.bool_(
            self._scalar_bool(imagined_valid, "predicted_xs_valid")
        )
        if self.images in {"real", "all"}:
            # Store one real frame per real edge (plus the initial root), not
            # one duplicate frame for every augmented search computation.
            edge_has_image = initial or real_transition
            if edge_has_image:
                if initial:
                    image_cursor = cursor
                else:
                    after = numpy.asarray(_freeze_event(event.get("cursor_after")))
                    if after.size != 1:
                        raise VisualBehaviorError(
                            "runner cursor_after must be one integer"
                        )
                    image_cursor = int(after.reshape(-1)[0])
                if not 0 <= image_cursor < len(self.observation_provider):
                    raise VisualBehaviorError(
                        f"real image cursor {image_cursor} is outside episode observations"
                    )
                real_value = (
                    raw_real
                    if raw_real is not None
                    else self.observation_provider[image_cursor]
                )
                image = self._display_image(real_value)
                if self._real_image_template is None:
                    self._real_image_template = numpy.zeros_like(image)
                elif image.shape != self._real_image_template.shape:
                    raise VisualBehaviorError("real image shape changed during replay")
                valid = (
                    self._scalar_bool(real_valid, "real image validity")
                    if raw_real is not None
                    else True
                )
                output["real_image"] = image if valid else self._real_image_template.copy()
                output["real_image_valid"] = numpy.bool_(valid)
            else:
                if self._real_image_template is None:
                    raise VisualBehaviorError(
                        "initial root did not establish the real image schema"
                    )
                output["real_image"] = self._real_image_template.copy()
                output["real_image_valid"] = numpy.bool_(False)
        if self.images in {"imagined", "all"}:
            image = self._display_image(raw_imagined)
            if self._imagined_image_template is None:
                self._imagined_image_template = numpy.zeros_like(image)
            elif image.shape != self._imagined_image_template.shape:
                raise VisualBehaviorError("imagined image shape changed during replay")
            valid = self._scalar_bool(imagined_valid, "imagined image validity")
            output["imagined_image"] = (
                image if valid else self._imagined_image_template.copy()
            )
            output["imagined_image_valid"] = numpy.bool_(valid)
        return output

    def _display_image(self, value: Any) -> Any:
        numpy = importlib.import_module("numpy")
        array = numpy.asarray(_freeze_event(value))
        while array.ndim > 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 3:
            raise VisualBehaviorError(
                f"runner image must reduce to CHW, got {array.shape}"
            )
        channels = 1 if self.grayscale else 3
        if array.shape[0] < channels:
            raise VisualBehaviorError(f"runner image has too few channels: {array.shape}")
        array = array[-channels:]
        if numpy.issubdtype(array.dtype, numpy.floating):
            array = numpy.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
            if array.size and float(numpy.max(array)) <= 1.0:
                array = array * 255.0
            array = numpy.rint(numpy.clip(array, 0.0, 255.0)).astype(numpy.uint8)
        else:
            array = numpy.clip(array, 0, 255).astype(numpy.uint8, copy=False)
        return numpy.ascontiguousarray(array)

    @staticmethod
    def _scalar_bool(value: Any, name: str) -> bool:
        numpy = importlib.import_module("numpy")
        array = numpy.asarray(_freeze_event(value))
        if array.size != 1 or array.dtype.kind != "b":
            raise VisualBehaviorError(f"trace field {name!r} must be one boolean")
        return bool(array.reshape(-1)[0])

    def write_event(self, event: Mapping[str, Any]) -> None:
        if not isinstance(event, Mapping):
            raise VisualBehaviorError("runner event sink payload must be a mapping")
        numpy = importlib.import_module("numpy")
        step_times = numpy.asarray(_freeze_event(event.get("step_times")))
        if step_times.shape != (1,) or step_times.dtype != numpy.dtype("float32"):
            raise VisualBehaviorError(
                "timing_mode=none requires step_times float32 sentinel shape (1,)"
            )
        if self._scalar_bool(event.get("step_times_valid"), "step_times_valid"):
            raise VisualBehaviorError(
                "timing_mode=none cannot claim valid per-step timing"
            )
        cursor_array = importlib.import_module("numpy").asarray(
            _freeze_event(event.get("real_cursor"))
        )
        if cursor_array.size != 1:
            raise VisualBehaviorError("runner real_cursor must be one integer")
        cursor = int(cursor_array.reshape(-1)[0])
        if not 0 <= cursor < len(self.source_frame_indices):
            raise VisualBehaviorError(
                f"runner real_cursor {cursor} is outside source episode alignment"
            )
        real_transition = self._scalar_bool(
            event.get("real_transition"), "real_transition"
        )
        alignment_cursor = cursor
        if real_transition:
            after = importlib.import_module("numpy").asarray(
                _freeze_event(event.get("cursor_after"))
            )
            if after.size != 1:
                raise VisualBehaviorError("runner cursor_after must be one integer")
            alignment_cursor = int(after.reshape(-1)[0])
        if not 0 <= alignment_cursor < len(self.source_frame_indices):
            raise VisualBehaviorError(
                f"runner aligned cursor {alignment_cursor} is outside source episode alignment"
            )
        event = self._filter_images(event, cursor)
        numpy = importlib.import_module("numpy")
        event["source_frame_index"] = numpy.int64(
            self.source_frame_indices[alignment_cursor]
        )
        event["source_frame_index_valid"] = numpy.bool_(True)
        # A real row displays the post-edge frame, but its decision timestamp
        # belongs to the human action selected at cursor_before.
        event["decision_time"] = numpy.float64(self.decision_times[cursor])
        event["decision_time_valid"] = numpy.bool_(True)
        missing = sorted(self._REQUIRED - set(event))
        if missing:
            raise VisualBehaviorError(
                "runner did not emit the canonical trace schema; missing=" + repr(missing)
            )
        initial = self._scalar_bool(event["initial_root"], "initial_root")
        if initial != (self.event_count == 0):
            raise VisualBehaviorError(
                "exactly the first native event must have initial_root=true"
            )
        self._scalar_bool(event["forced_root_reset"], "forced_root_reset")
        # Derive the display image and validate the full runner contract first.
        # Only the disk payload changes; model inputs, RNG and control do not.
        if not self.store_predicted_xs:
            event.pop("predicted_xs")
            event.pop("predicted_xs_valid")
        schema = self._numeric_schema(event)
        if self._schema is None:
            self._schema = schema
        elif schema != self._schema:
            raise VisualBehaviorError("runner trace schema changed after the initial root")
        self.output.write_event(event)
        self.event_count += 1

    def assert_complete(self) -> None:
        if self.event_count < 1:
            raise VisualBehaviorError("runner completed without emitting an initial-root event")


class _SourceLoaderView:
    """One-source BC-loader view retaining the loader's preprocessing authority."""

    def __init__(self, task: SourceTask) -> None:
        if not task.episodes:
            raise VisualBehaviorError(f"source task has no episodes: {task.source.source_id}")
        first = task.episodes[0]
        self.file_records = [
            _BehaviorFileView(
                path=Path(task.source.path),
                subject=int(first.subject),
                session=int(first.session),
                block=int(first.block),
                game=int(first.game),
            )
        ]
        self._episodes = [
            _EpisodeView(
                file_index=0,
                episode_index=int(descriptor.local_episode_index),
                observation_source_index=descriptor.observation_source_index,
                decision_times=descriptor.decision_times,
                actions=descriptor.actions,
                rewards=descriptor.rewards,
                done=descriptor.done,
                truncated=descriptor.truncated,
            )
            for descriptor in task.episodes
        ]
        self.frame_stack_n = int(task.preprocess.frame_stack_n)
        self.target_size = tuple(int(value) for value in task.preprocess.target_size)
        self.grayscale = bool(task.preprocess.grayscale)
        self.normalize = bool(task.preprocess.normalize)

    def _preprocess_frame(self, frame: Any) -> Any:
        from thinker.bc_loader import FrameStackedBehavioralDataLoader

        return FrameStackedBehavioralDataLoader._preprocess_frame(self, frame)

    def _stack_source_indices(self, episode: Any, observation_position: int) -> Any:
        from thinker.bc_loader import FrameStackedBehavioralDataLoader

        return FrameStackedBehavioralDataLoader._stack_source_indices(
            self, episode, observation_position
        )


class _LazyObservationProvider:
    """Lazy mmap provider with one bounded CPU frame/decompression prefetch."""

    def __init__(self, behavior_source: Any, descriptor: Any, positions: Any) -> None:
        self._source = behavior_source
        self._descriptor = descriptor
        self._positions = positions
        loader = behavior_source.loader
        self._cache: OrderedDict[int, Any] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._cache_limit = max(8, int(loader.frame_stack_n) * 3)
        channels = 1 if bool(loader.grayscale) else 3
        self.observation_shape = (
            channels * int(loader.frame_stack_n),
            int(loader.target_size[0]),
            int(loader.target_size[1]),
        )
        self.shape = (len(self._positions),) + self.observation_shape
        numpy = importlib.import_module("numpy")
        self.dtype = numpy.dtype("float32" if loader.normalize else "uint8")
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="behavior-frame-prefetch"
        )
        self._prefetch_index: Optional[int] = None
        self._prefetch_future: Any = None
        self._closed = False
        if len(self):
            # This first task performs streaming NPZ->NPY extraction as well
            # as the first stack.  Later one-frame lookahead overlaps CPU
            # preprocessing with actor/model inference without a second mmap.
            self._schedule_prefetch(0)

    def __len__(self) -> int:
        return int(len(self._positions))

    def _materialize(self, index: int) -> Any:
        numpy = importlib.import_module("numpy")
        position = int(self._positions[index])
        return numpy.ascontiguousarray(
            self._source.stacked_frame(self._descriptor, position)
        )

    def _schedule_prefetch(self, index: int) -> None:
        if self._closed or not 0 <= index < len(self):
            return
        # Keep even a completed lookahead until its intended consumer takes
        # it; overwriting a done Future would redundantly preprocess a frame.
        if self._prefetch_future is not None:
            return
        self._prefetch_index = index
        self._prefetch_future = self._executor.submit(self._materialize, index)

    def _remember(self, index: int, value: Any) -> Any:
        position = int(self._positions[index])
        with self._cache_lock:
            self._cache[position] = value
            self._cache.move_to_end(position)
            while len(self._cache) > self._cache_limit:
                self._cache.popitem(last=False)
        self._schedule_prefetch(index + 1)
        return value

    def __getitem__(self, observation_index: int) -> Any:
        if self._closed:
            raise VisualBehaviorError("observation provider is closed")
        index = int(observation_index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        position = int(self._positions[index])
        with self._cache_lock:
            value = self._cache.pop(position, None)
            if value is not None:
                self._cache[position] = value
        if value is not None:
            self._schedule_prefetch(index + 1)
            return value
        if self._prefetch_index == index and self._prefetch_future is not None:
            value = self._prefetch_future.result()
            self._prefetch_future = None
            self._prefetch_index = None
        else:
            value = self._materialize(index)
        return self._remember(index, value)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)
        with self._cache_lock:
            self._cache.clear()


def _episode_batch(behavior_source: Any, descriptor: Any) -> Mapping[str, Any]:
    numpy = importlib.import_module("numpy")
    replay = behavior_source.replay_spec(descriptor)
    provider = _LazyObservationProvider(
        behavior_source, descriptor, replay["observation_positions"]
    )
    actions = numpy.asarray(replay["actions_seq"], dtype=numpy.int64)
    if len(actions) < 1 or len(provider) != len(actions) + 1:
        raise VisualBehaviorError("logical episode observation/action lengths disagree")
    score_mask = numpy.asarray(replay["score_mask"], dtype=numpy.bool_)
    from gymnasium import spaces

    high = 1.0 if provider.dtype.name == "float32" else 255
    observation_space = spaces.Box(
        low=0.0, high=high, shape=provider.observation_shape, dtype=provider.dtype
    )
    return {
        "observation_provider": provider,
        "observation_source_index": numpy.asarray(
            replay["observation_source_index"], dtype=numpy.int64
        ),
        "actions_seq": actions[numpy.newaxis, :],
        "rewards_seq": numpy.asarray(replay["rewards_seq"], dtype=numpy.float32)[numpy.newaxis, :],
        "done_seq": numpy.asarray(replay["done_seq"], dtype=numpy.bool_)[numpy.newaxis, :],
        "truncated_seq": numpy.asarray(replay["truncated_seq"], dtype=numpy.bool_)[numpy.newaxis, :],
        "initial_prev_action": numpy.asarray([replay["initial_prev_action"]], dtype=numpy.int64),
        "score_mask": score_mask,
        "observation_space": observation_space,
    }


def _probe_batch(batch: Mapping[str, Any]) -> Mapping[str, Any]:
    numpy = importlib.import_module("numpy")
    provider = batch["observation_provider"]
    edge_count = min(2, int(batch["actions_seq"].shape[1]))
    observations = numpy.stack(
        [provider[index] for index in range(edge_count + 1)], axis=0
    )[numpy.newaxis, :]
    probe = {
        "obs_seq": observations,
        "actions_seq": batch["actions_seq"][:, :edge_count],
        "rewards_seq": batch["rewards_seq"][:, :edge_count],
        "done_seq": batch["done_seq"][:, :edge_count].copy(),
        "truncated_seq": batch["truncated_seq"][:, :edge_count].copy(),
        "initial_prev_action": batch["initial_prev_action"],
        "score_mask": numpy.asarray([False] + [True] * (edge_count - 1), dtype=numpy.bool_),
    }
    if edge_count > 1:
        probe["done_seq"][:, :-1] = False
        probe["truncated_seq"][:, :-1] = False
    return probe


def _vector_actor_observation_space(template_space: Any, batch_size: int) -> Any:
    """Restore the vector axis expected by ActorNet without evaluator policy checks."""

    numpy = importlib.import_module("numpy")
    from gymnasium import spaces

    if not isinstance(template_space, spaces.Dict):
        raise VisualBehaviorError("Dynamic wrapper observation_space must be Dict")
    vector_spaces = dict(template_space.spaces)
    for name in ("real_states", "xs"):
        space = vector_spaces.get(name)
        if space is None:
            continue
        vector_spaces[name] = spaces.Box(
            low=numpy.broadcast_to(space.low, (batch_size,) + tuple(space.shape)),
            high=numpy.broadcast_to(space.high, (batch_size,) + tuple(space.shape)),
            dtype=space.dtype,
        )
    for name in ("tree_reps", "hs"):
        space = vector_spaces.get(name)
        if space is not None and int(space.shape[0]) != batch_size:
            raise VisualBehaviorError(
                f"Dynamic wrapper {name} does not begin with batch size {batch_size}"
            )
    return spaces.Dict(vector_spaces)


def _build_recorder_networks(
    first_batch: Mapping[str, Any],
    device: Any,
    flags: Any,
    spec: Any,
    actor_checkpoint: Mapping[str, Any],
    model_checkpoint: Mapping[str, Any],
) -> tuple[Any, Any, Any]:
    """Reconstruct networks while honoring the checkpoint's own protocol.

    The qualification evaluator's public builder calls its historical
    fixed-length holdout validator.  Full-episode recording has a different
    shape contract, so this narrowly copies only the low-level construction
    path and retains strict state-dict/action/observation checks.
    """

    numpy = importlib.import_module("numpy")
    from gymnasium import spaces
    from thinker import util
    from thinker.actor_net import ActorNet
    from thinker.cenv import cModelWrapper
    from thinker.dataset_env import BehaviorSequenceVectorEnv
    from thinker.model_net import ModelNet

    obs = numpy.asarray(first_batch["obs_seq"])
    actions = numpy.asarray(first_batch["actions_seq"], dtype=numpy.int64)
    if actions.ndim != 2 or actions.shape[0] != 1:
        raise VisualBehaviorError("network probe actions must have shape [1,E]")
    if obs.shape[:2] != (1, actions.shape[1] + 1):
        raise VisualBehaviorError("network probe observations/actions disagree")
    if tuple(obs.shape[2:]) != tuple(spec.observation_shape):
        raise VisualBehaviorError(
            "network probe observation shape disagrees with live environment"
        )
    if numpy.dtype(obs.dtype).name != str(spec.observation_dtype):
        raise VisualBehaviorError(
            "network probe observation dtype disagrees with live environment"
        )
    base_env = BehaviorSequenceVectorEnv(
        obs_seq=obs,
        actions_seq=actions,
        rewards_seq=numpy.asarray(first_batch["rewards_seq"], dtype=numpy.float32),
        done_seq=numpy.asarray(first_batch["done_seq"], dtype=numpy.bool_),
        truncated_seq=numpy.asarray(
            first_batch["truncated_seq"], dtype=numpy.bool_
        ),
        initial_prev_action=numpy.asarray(
            first_batch["initial_prev_action"], dtype=numpy.int64
        ),
        score_mask=numpy.asarray(first_batch["score_mask"], dtype=numpy.bool_),
        num_actions=int(spec.num_actions),
        observation_space=spaces.Box(
            low=spec.observation_low,
            high=spec.observation_high,
            shape=tuple(spec.observation_shape),
            dtype=numpy.dtype(spec.observation_dtype),
        ),
    )
    template = None
    try:
        primary_action_space = base_env.action_space[0]
        if int(primary_action_space.n) != int(spec.num_actions):
            raise VisualBehaviorError(
                "behavior action space disagrees with checkpoint environment"
            )
        model_net = ModelNet(
            obs_space=base_env.observation_space,
            action_space=primary_action_space,
            flags=flags,
            frame_stack_n=int(getattr(flags, "frame_stack_n", 4)),
        ).to(device)
        model_net.set_weights(model_checkpoint["model_net_state_dict"])
        model_net.eval()
        template = cModelWrapper(
            env=base_env,
            env_n=actions.shape[0],
            flags=flags,
            model_net=model_net,
            device=device,
            timing=False,
        )
        actor_space = _vector_actor_observation_space(
            template.observation_space, actions.shape[0]
        )
        actor_net = ActorNet(
            obs_space=actor_space,
            action_space=template.action_space,
            flags=flags,
            tree_rep_meaning=util.get_tree_rep_meaning(
                int(spec.num_actions), 1, flags
            ),
        ).to(device)
        actor_net.set_weights(actor_checkpoint["actor_net_state_dict"])
        actor_net.eval()
        if int(actor_net.num_actions) != int(spec.num_actions):
            raise VisualBehaviorError(
                "actor action dimension disagrees with live environment"
            )
        return actor_net, model_net, flags
    finally:
        if template is not None:
            template.close()
        else:
            close = getattr(base_env, "close", None)
            if callable(close):
                close()


def _normalise_tree_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in schema.items():
        if isinstance(value, slice):
            result[str(key)] = [value.start, value.stop, value.step]
        elif isinstance(value, int):
            result[str(key)] = [int(value), int(value) + 1, None]
        else:
            raise VisualBehaviorError(f"unsupported tree schema entry {key!r}: {value!r}")
    return result


def _set_episode_seed(seed: int, torch: Any, numpy: Any) -> None:
    random.seed(seed)
    numpy.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _assert_unchanged(record: StableFile | SourceBundle) -> None:
    identity = _identity(Path(record.path))
    expected = (record.device, record.inode, record.size, record.mtime_ns)
    if identity != expected:
        raise VisualBehaviorError(f"immutable input identity changed: {record.path}")


def _import_evaluator() -> Any:
    last_error: Optional[BaseException] = None
    for name in ("evaluate_dynamic_imitation", "thinker.evaluate_dynamic_imitation"):
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as error:
            last_error = error
    raise RuntimeDependencyError(
        "evaluate_dynamic_imitation.py is unavailable; cannot reconstruct checkpoint networks"
    ) from last_error


def _coordinator_source_tasks(
    config: CaptureConfig, sources: Sequence[SourceBundle]
) -> list[SourceTask]:
    """Index episodes once, then send only one-source descriptor bundles."""

    evaluator = _import_evaluator()
    flags = _flags_from_stable_config(config.checkpoint.config, config.checkpoint.directory)
    spec = evaluator.resolve_evaluation_spec(flags)
    from thinker.bc_loader import FrameStackedBehavioralDataLoader

    loader = FrameStackedBehavioralDataLoader(
        base_path=config.data_root,
        subjects=None,
        game_id=spec.game_id,
        sessions=None,
        split=None,
        num_actions=spec.num_actions,
        source_paths=[source.path for source in sources],
        scored_length=max(1, spec.scored_length),
        frame_stack_n=spec.frame_stack_n,
        target_size=spec.target_size,
        grayscale=spec.grayscale,
        normalize=spec.observation_dtype == "float32",
        decision_hz=15.0,
        seed=config.base_seed,
    )
    trace = _load_behavior_trace()
    allowed = [source.path for source in sources]
    expected = {
        source.path: {
            "size": source.size,
            "sha256": source.sha256,
            "device": source.device,
            "inode": source.inode,
            "mtime_ns": source.mtime_ns,
        }
        for source in sources
    }
    descriptor_source = trace.BehaviorEpisodeSource(
        loader,
        expected_sources=expected,
        allowed_source_paths=allowed,
    )
    try:
        by_path: dict[str, list[Any]] = {}
        for descriptor in descriptor_source.episodes:
            by_path.setdefault(str(Path(descriptor.source_path).resolve()), []).append(
                descriptor
            )
        preprocess = ReplayPreprocessSpec(
            frame_stack_n=int(spec.frame_stack_n),
            target_size=tuple(int(value) for value in spec.target_size),
            grayscale=bool(spec.grayscale),
            normalize=spec.observation_dtype == "float32",
        )
        tasks: list[SourceTask] = []
        for source in sources:
            descriptors = tuple(
                sorted(
                    (
                        descriptor
                        for descriptor in by_path.get(
                            str(Path(source.path).resolve()), ()
                        )
                        if not config.episode_indices
                        or int(descriptor.local_episode_index)
                        in config.episode_indices
                    ),
                    key=lambda item: int(item.local_episode_index),
                )
            )
            if not descriptors:
                raise VisualBehaviorError(
                    "selected source has no usable checkpoint-game episodes: "
                    f"{source.source_id}"
                )
            if config.episode_indices and {
                int(descriptor.local_episode_index) for descriptor in descriptors
            } != set(config.episode_indices):
                raise VisualBehaviorError(
                    "selected source does not contain every requested episode index: "
                    f"{source.source_id}"
                )
            tasks.append(SourceTask(source=source, episodes=descriptors, preprocess=preprocess))
        return tasks
    finally:
        descriptor_source.close()


class _CaptureRuntime:
    """Heavy worker-local runtime; one instance owns exactly one device."""

    def __init__(self, config: CaptureConfig, worker: WorkerSpec) -> None:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        self.config = config
        self.worker = worker
        self.numpy = importlib.import_module("numpy")
        self.torch = importlib.import_module("torch")
        self.torch.set_num_threads(int(config.torch_threads_per_worker))
        try:
            self.torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        self.trace_api = _load_behavior_trace()
        if worker.device.startswith("cuda"):
            if not self.torch.cuda.is_available():
                raise VisualBehaviorError(f"CUDA worker requested but unavailable: {worker.device}")
            self.torch.cuda.set_device(self.torch.device(worker.device))
        self.device = self.torch.device(worker.device)
        self.evaluator = _import_evaluator()
        self.flags = _flags_from_stable_config(
            config.checkpoint.config, config.checkpoint.directory
        )
        self.spec = self.evaluator.resolve_evaluation_spec(self.flags)
        self.actor_checkpoint = _torch_load_stable(
            config.checkpoint.actor,
            label="actor checkpoint",
            torch=self.torch,
        )
        self.model_checkpoint = _torch_load_stable(
            config.checkpoint.model,
            label="model checkpoint",
            torch=self.torch,
        )
        self._validate_checkpoint_steps()
        self._bind_voc_gate_policy_schema()
        # Do not call evaluate_dynamic_imitation.validate_* here: that
        # qualification tool intentionally freezes the historical 20/20/20
        # protocol and margin/prior coefficients.  Recorder checkpoints are
        # instead bound byte-for-byte and checked against their own embedded
        # config above, so valid b30/d30/u30 AP0/M0 campaign milestones remain
        # replayable without weakening the evaluator's separate contract.
        self.voc_ema_head_state_dict = self.actor_checkpoint.get(
            "voc_ema_gate_head_state_dict"
        )
        if bool(getattr(self.flags, "think_cost_anneal", False)):
            raise VisualBehaviorError(
                "checkpoint uses annealed think_cost, but no exact per-step cost "
                "is bound to this replay; refusing an inexact VoC trace"
            )
        self.effective_think_cost = float(getattr(self.flags, "think_cost", 0.0))
        self.actor_net = None
        self.model_net = None
        self.runner = None
        from thinker import util

        self.tree_rep_schema = _normalise_tree_schema(
            util.get_tree_rep_meaning(self.spec.num_actions, 1, self.flags)
        )

    def _bind_voc_gate_policy_schema(self) -> None:
        top_level = self.actor_checkpoint.get("voc_gate_policy_schema_version")
        self.voc_gate_schema_validation: Optional[Mapping[str, Any]] = None
        declarations = {
            "actor embedded": _checkpoint_flags_mapping(
                self.actor_checkpoint, "actor"
            ).get("voc_gate_policy_schema_version"),
            "model top-level": self.model_checkpoint.get(
                "voc_gate_policy_schema_version"
            ),
            "model embedded": _checkpoint_flags_mapping(
                self.model_checkpoint, "model"
            ).get("voc_gate_policy_schema_version"),
            "config": getattr(self.flags, "voc_gate_policy_schema_version", None),
        }
        declared = {
            label: value for label, value in declarations.items() if value is not None
        }
        if top_level is None:
            if declared:
                raise VisualBehaviorError(
                    "VoC gate-policy schema exists outside the actor top-level: "
                    + repr(declared)
                )
            return
        if type(top_level) is not int or top_level < 1:
            raise VisualBehaviorError("actor VoC gate-policy schema is invalid")
        contradictory = {
            label: value for label, value in declared.items() if value != top_level
        }
        if contradictory:
            raise VisualBehaviorError(
                "VoC gate-policy schema declarations disagree with actor checkpoint: "
                + repr(contradictory)
            )
        embedded = _checkpoint_flags_mapping(self.actor_checkpoint, "actor")
        if embedded.get("dynamic_voc_mode", "off") != "off":
            validator = getattr(
                self.evaluator, "_validate_voc_gate_policy_schema", None
            )
            if not callable(validator):
                raise RuntimeDependencyError(
                    "checkpoint evaluator lacks the strict VoC schema validator"
                )
            try:
                gate_state = validator(
                    self.actor_checkpoint,
                    embedded,
                    label="behavior recorder actor",
                )
            except (TypeError, ValueError, RuntimeError) as error:
                raise VisualBehaviorError(
                    f"actor VoC gate-policy schema is invalid: {error}"
                ) from error
            if int(gate_state["voc_gate_policy_schema_version"]) != top_level:
                raise VisualBehaviorError(
                    "validated VoC schema disagrees with actor top-level"
                )
            self.voc_gate_schema_validation = dict(gate_state)
        # Historical canonical milestones published schema=3 only at the
        # actor top level.  Bind that verified declaration into this private
        # runtime namespace before ActorNet construction; never rewrite the
        # checkpoint/config bytes.
        self.flags.voc_gate_policy_schema_version = int(top_level)

    def _validate_checkpoint_steps(self) -> None:
        _validate_loaded_checkpoint_payloads(
            self.actor_checkpoint,
            self.model_checkpoint,
            self.config.checkpoint,
            config_xpid=str(getattr(self.flags, "xpid", "")).strip() or None,
            config_mapping=vars(self.flags),
        )

    def _ensure_runner(self, batch: Mapping[str, Any]) -> None:
        if self.runner is not None:
            return
        self.actor_net, self.model_net, self.flags = _build_recorder_networks(
            _probe_batch(batch), self.device, self.flags, self.spec,
            self.actor_checkpoint, self.model_checkpoint,
        )
        checkpoint_schema = self.actor_checkpoint.get(
            "voc_gate_policy_schema_version"
        )
        resolved_schema = getattr(self.flags, "voc_gate_policy_schema_version", None)
        if checkpoint_schema is not None and resolved_schema != checkpoint_schema:
            raise VisualBehaviorError(
                "resolved VoC gate-policy schema disagrees with actor checkpoint: "
                f"{resolved_schema!r} != {checkpoint_schema!r}"
            )
        from thinker import dynamic_imitation

        runtime_trace_schema = int(
            getattr(dynamic_imitation, "DYNAMIC_TRACE_EVENT_SCHEMA_VERSION", -1)
        )
        if runtime_trace_schema != TRACE_EVENT_SCHEMA_VERSION:
            raise RuntimeDependencyError(
                "DynamicImitationRunner trace schema disagrees with recorder: "
                f"{runtime_trace_schema} != {TRACE_EVENT_SCHEMA_VERSION}"
            )

        self.runner = dynamic_imitation.DynamicImitationRunner(
            self.actor_net, self.model_net, self.flags, device=self.device
        )
        if not callable(getattr(self.runner, "stream_rollout_episode", None)):
            raise RuntimeDependencyError(
                "DynamicImitationRunner.stream_rollout_episode is unavailable; "
                "this build cannot truthfully perform full-episode capture"
            )

    def _episode_metadata(
        self, source: SourceBundle, descriptor: Any, seed: int, store_id: str
    ) -> dict[str, Any]:
        source_observations = int(descriptor.logical_observation_count)
        source_edges = int(descriptor.edge_count)
        replay_observations = source_observations - 1
        replay_edges = source_edges - 1
        scored_edges = int(
            getattr(descriptor, "scored_edge_count", max(0, replay_edges - 1))
        )
        return {
            "schema_version": CAPTURE_SCHEMA_VERSION,
            "episode_id": store_id,
            "episode_seed": seed,
            "seed_protocol": SEED_PROTOCOL,
            "source_id": source.source_id,
            "source_path": source.path,
            "source_sha256": source.sha256,
            "source_size": source.size,
            "source_episode_index": int(descriptor.local_episode_index),
            "subject": int(descriptor.subject), "session": int(descriptor.session),
            "block": int(descriptor.block), "game": int(descriptor.game),
            # The unqualified logical counts describe the replay handed to the
            # runner and therefore retain obs = edges + 1.  Source-domain
            # counts make the consumed burn-in edge explicit.
            "logical_observation_count": replay_observations,
            "logical_action_count": replay_edges,
            "source_logical_observation_count": source_observations,
            "source_logical_edge_count": source_edges,
            "replay_observation_count": replay_observations,
            "replay_edge_count": replay_edges,
            "scored_edge_count": scored_edges,
            "checkpoint_bundle_sha256": self.config.checkpoint.bundle_sha256,
            "checkpoint_actor_sha256": self.config.checkpoint.actor.sha256,
            "checkpoint_model_sha256": self.config.checkpoint.model.sha256,
            "checkpoint_config_sha256": self.config.checkpoint.config.sha256,
            "checkpoint_real_step": self.config.checkpoint.requested_step,
            "tree_carry": self.config.tree_carry,
            "policy_mode": self.config.policy_mode,
            "determinism_backend_fingerprint": dict(
                self.config.determinism_backend_fingerprint
            ),
            "timing_mode": TIMING_MODE,
            "trace_event_schema_version": TRACE_EVENT_SCHEMA_VERSION,
            "images": self.config.images,
            "chunk_events": self.config.chunk_events,
            "compression_level": self.config.compression_level,
            "store_predicted_xs": self.config.store_predicted_xs,
            "frame_stack_n": int(self.spec.frame_stack_n),
            "target_size": [int(value) for value in self.spec.target_size],
            "grayscale": bool(self.spec.grayscale),
            "observation_dtype": str(self.spec.observation_dtype),
            "voc_gate_policy_schema_version": getattr(
                self.flags, "voc_gate_policy_schema_version", None
            ),
            "voc_gate_policy_schema_source": (
                "actor_checkpoint_top_level"
                if self.actor_checkpoint.get("voc_gate_policy_schema_version")
                is not None
                else "config_or_off"
            ),
            "voc_ema_head_available": self.voc_ema_head_state_dict is not None,
            "effective_think_cost": self.effective_think_cost,
        }

    def _stream_episode(
        self, batch: Mapping[str, Any], event_sink: Any
    ) -> Mapping[str, Any]:
        """Run one episode while preserving the configured policy contract."""

        with self.torch.no_grad():
            return self.runner.stream_rollout_episode(
                batch,
                event_sink=event_sink,
                tree_carry=self.config.tree_carry,
                policy_mode=self.config.policy_mode,
                voc_ema_head_state_dict=self.voc_ema_head_state_dict,
                effective_think_cost=self.effective_think_cost,
            )

    def _existing_store(
        self, final_path: Path, expected_metadata: Mapping[str, Any]
    ) -> Optional[Mapping[str, Any]]:
        if not final_path.exists():
            return None
        if not self.config.resume:
            raise VisualBehaviorError(
                f"episode store already exists (use --resume): {final_path}"
            )
        validation = self.trace_api.validate_episode_store(final_path)
        if not isinstance(validation, Mapping):
            raise VisualBehaviorError(f"resume validation returned no metadata: {final_path}")
        metadata = validation.get("metadata", validation)
        if not isinstance(metadata, Mapping):
            raise VisualBehaviorError(f"existing store lacks metadata: {final_path}")
        for key, expected in expected_metadata.items():
            if metadata.get(key) != expected:
                raise VisualBehaviorError(f"existing store {key} mismatch: {final_path}")
        if validation.get("images") != self.config.images:
            raise VisualBehaviorError(f"existing store image mode mismatch: {final_path}")
        return validation

    def _local_paths(self, store_id: str) -> tuple[Path, Path, Path]:
        canonical = Path(self.config.output_root) / "episodes" / f"{store_id}.zarr"
        if self.config.scratch_root:
            local_final = (
                Path(self.config.scratch_root) / f"worker-{self.worker.rank:03d}"
                / "episodes" / f"{store_id}.zarr"
            )
        else:
            local_final = canonical
        partial = local_final.with_name(f".{local_final.name}.partial-{os.getpid()}")
        return partial, local_final, canonical

    def _quarantine_stale_partials(self, local_final: Path) -> tuple[Path, ...]:
        stale = sorted(local_final.parent.glob(f".{local_final.name}.partial-*"))
        if not stale:
            return ()
        if not self.config.resume:
            raise VisualBehaviorError(
                f"stale episode partial exists (use --resume): {stale[0]}"
            )
        quarantine = local_final.parent / ".aborted-partials"
        quarantine.mkdir(parents=True, exist_ok=True)
        moved: list[Path] = []
        for source in stale:
            destination = quarantine / f"{source.name}.{time.time_ns()}"
            os.rename(source, destination)
            moved.append(destination)
        _fsync_directory(quarantine)
        _fsync_directory(local_final.parent)
        return tuple(moved)

    def _recover_local_final(
        self,
        local_final: Path,
        canonical: Path,
        metadata: Mapping[str, Any],
    ) -> Optional[Mapping[str, Any]]:
        if local_final == canonical or not local_final.exists():
            return None
        validation = self._existing_store(local_final, metadata)
        if validation is None:
            return None
        self._publish(local_final, canonical)
        return self._existing_store(canonical, metadata)

    def _publish(self, local_final: Path, canonical: Path) -> None:
        if local_final == canonical:
            return
        canonical.parent.mkdir(parents=True, exist_ok=True)
        self._cleanup_stale_transfers(canonical)
        transfer = canonical.with_name(f".{canonical.name}.transfer-partial-{os.getpid()}")
        if transfer.exists():
            raise VisualBehaviorError(f"stale transfer directory exists: {transfer}")
        try:
            shutil.copytree(local_final, transfer)
            self.trace_api.validate_episode_store(transfer)
            os.rename(transfer, canonical)
            _fsync_directory(canonical.parent)
            self.trace_api.validate_episode_store(canonical)
        except FileExistsError as error:
            raise VisualBehaviorError(
                f"canonical episode store appeared concurrently: {canonical}"
            ) from error
        finally:
            if transfer.exists():
                shutil.rmtree(transfer)
        if local_final.is_symlink() or not local_final.is_dir():
            raise VisualBehaviorError(
                f"published scratch source is not a safe directory: {local_final}"
            )
        shutil.rmtree(local_final)
        _fsync_directory(local_final.parent)

    def _cleanup_stale_transfers(self, canonical: Path) -> tuple[Path, ...]:
        """Atomically retire dead-PID transfer copies, never writer partials."""

        prefix = f".{canonical.name}.transfer-partial-"
        quarantine = canonical.parent / ".aborted-transfers"
        quarantined: list[Path] = []
        if quarantine.is_dir():
            for candidate in sorted(quarantine.glob(f"{prefix}*")):
                suffix = candidate.name[len(prefix):]
                pieces = suffix.split(".", 1)
                if len(pieces) == 2 and all(piece.isdigit() for piece in pieces):
                    quarantined.append(candidate)
        stale: list[Path] = []
        for candidate in sorted(canonical.parent.glob(f"{prefix}*")):
            suffix = candidate.name[len(prefix):]
            if not suffix.isdigit():
                continue
            pid = int(suffix)
            if _pid_is_alive(pid):
                raise VisualBehaviorError(
                    f"active episode transfer already exists: {candidate}"
                )
            stale.append(candidate)
        if not stale and not quarantined:
            return ()
        if not self.config.resume:
            raise VisualBehaviorError(
                "stale episode transfer exists (use --resume): "
                f"{(stale or quarantined)[0]}"
            )
        quarantine.mkdir(parents=True, exist_ok=True)
        retired: list[Path] = list(quarantined)
        for candidate in stale:
            if candidate.is_symlink() or not candidate.is_dir():
                raise VisualBehaviorError(
                    f"refusing to remove unsafe transfer entry: {candidate}"
                )
            destination = quarantine / f"{candidate.name}.{time.time_ns()}"
            os.rename(candidate, destination)
            retired.append(destination)
        _fsync_directory(canonical.parent)
        _fsync_directory(quarantine)
        # The canonical name is now free.  Transfer copies are redundant—the
        # validated scratch local_final remains the recovery authority—so
        # remove the quarantined copies and keep failure partials untouched.
        for destination in retired:
            if destination.is_symlink() or not destination.is_dir():
                raise VisualBehaviorError(
                    f"refusing to remove unsafe quarantined transfer: {destination}"
                )
            shutil.rmtree(destination)
        _fsync_directory(quarantine)
        return tuple(stale)

    def process_source(self, task: SourceTask) -> Mapping[str, Any]:
        source = task.source
        _assert_unchanged(source)
        loader_view = _SourceLoaderView(task)
        image_scratch = (
            Path(self.config.scratch_root) / f"worker-{self.worker.rank:03d}"
            if self.config.scratch_root
            else None
        )
        expected = {
            source.path: {
                "size": source.size,
                "sha256": source.sha256,
                "device": source.device,
                "inode": source.inode,
                "mtime_ns": source.mtime_ns,
            }
        }
        behavior_source = self.trace_api.BehaviorEpisodeSource(
            loader_view,
            scratch_dir=image_scratch,
            expected_sources=expected,
            allowed_source_paths=[source.path],
        )
        completed: list[dict[str, Any]] = []
        try:
            for descriptor in sorted(
                behavior_source.episodes,
                key=lambda item: int(item.local_episode_index),
            ):
                local_index = int(descriptor.local_episode_index)
                seed = episode_seed(self.config.base_seed, source, local_index)
                store_id = episode_store_id(source, local_index)
                metadata = self._episode_metadata(source, descriptor, seed, store_id)
                partial, local_final, canonical = self._local_paths(store_id)
                existing = self._existing_store(canonical, metadata)
                if existing is not None:
                    completed.append({
                        "episode_id": store_id,
                        "path": str(canonical),
                        "status": "reused",
                        "event_count": int(existing["event_count"]),
                        "files_sha256": str(existing["files_sha256"]),
                        "images": str(existing["images"]),
                    })
                    continue
                self._quarantine_stale_partials(local_final)
                recovered = self._recover_local_final(
                    local_final, canonical, metadata
                )
                if recovered is not None:
                    completed.append({
                        "episode_id": store_id,
                        "path": str(canonical),
                        "status": "recovered-local-final",
                        "event_count": int(recovered["event_count"]),
                        "files_sha256": str(recovered["files_sha256"]),
                        "images": str(recovered["images"]),
                    })
                    continue
                if local_final.exists():
                    raise VisualBehaviorError(
                        f"unrecoverable local episode staging exists: {local_final}"
                    )
                partial.parent.mkdir(parents=True, exist_ok=True)
                canonical.parent.mkdir(parents=True, exist_ok=True)
                batch = _episode_batch(behavior_source, descriptor)
                try:
                    self._ensure_runner(batch)
                    writer = _BoundedWriter(
                        self.trace_api, partial_path=partial, final_path=local_final,
                        metadata=metadata, num_actions=self.spec.num_actions,
                        tree_rep_schema=self.tree_rep_schema,
                        chunk_events=self.config.chunk_events, images=self.config.images,
                        queue_size=self.config.queue_size,
                        compression_level=self.config.compression_level,
                    )
                    replay_alignment = behavior_source.replay_spec(descriptor)
                    event_sink = _CanonicalEventSink(
                        writer,
                        images=self.config.images,
                        source_frame_indices=replay_alignment["observation_source_index"],
                        decision_times=replay_alignment["decision_times"],
                        grayscale=bool(task.preprocess.grayscale),
                        observation_provider=batch["observation_provider"],
                        store_predicted_xs=self.config.store_predicted_xs,
                    )
                    try:
                        _set_episode_seed(seed, self.torch, self.numpy)
                        summary = self._stream_episode(batch, event_sink)
                        event_sink.assert_complete()
                        _validate_runner_summary(
                            summary, metadata, event_sink.event_count
                        )
                        writer.finalize()
                    except BaseException:
                        writer.abort()
                        raise
                    self.trace_api.validate_episode_store(local_final)
                    self._publish(local_final, canonical)
                    final_validation = self.trace_api.validate_episode_store(canonical)
                    completed.append({
                        "episode_id": store_id, "path": str(canonical),
                        "status": "captured", "seed": seed,
                        "runner_summary": _json_safe(summary),
                        "event_count": int(final_validation["event_count"]),
                        "files_sha256": str(final_validation["files_sha256"]),
                        "images": str(final_validation["images"]),
                    })
                    del final_validation
                finally:
                    close_provider = getattr(
                        batch["observation_provider"], "close", None
                    )
                    if callable(close_provider):
                        close_provider()
            behavior_source.release_source(source.sha256, remove_scratch=True)
        finally:
            behavior_source.close()
        _assert_unchanged(source)
        return {"source_index": source.source_index, "source_id": source.source_id,
                "source_sha256": source.sha256, "episodes": completed}

    def close(self) -> None:
        if self.runner is not None:
            self.runner.close()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except (TypeError, ValueError):
            pass
    return repr(value)


def _validate_runner_summary(
    summary: Mapping[str, Any], metadata: Mapping[str, Any], event_count: int
) -> None:
    if not isinstance(summary, Mapping):
        raise VisualBehaviorError("stream runner returned no episode summary")
    expected = {
        "initial_root_count": 1,
        "edge_count": int(metadata["replay_edge_count"]),
        "real_transition_count": int(metadata["replay_edge_count"]),
        "scored_edge_count": int(metadata["scored_edge_count"]),
        "native_event_count": int(event_count),
    }
    for key, value in expected.items():
        try:
            observed = int(summary[key])
        except (KeyError, TypeError, ValueError) as error:
            raise VisualBehaviorError(
                f"stream runner summary lacks valid {key}"
            ) from error
        if observed != value:
            raise VisualBehaviorError(
                f"stream runner summary {key} disagrees with episode alignment: "
                f"{observed} != {value}"
            )


def _worker_entry(config: CaptureConfig, worker: WorkerSpec, task_queue: Any, result_queue: Any) -> None:
    runtime: Optional[_CaptureRuntime] = None
    try:
        runtime = _CaptureRuntime(config, worker)
        while True:
            task = task_queue.get()
            if task is None:
                break
            try:
                payload = runtime.process_source(task)
                result_queue.put({"kind": "source", "ok": True,
                                  "worker": asdict(worker), "payload": payload})
            except BaseException as error:
                result_queue.put({"kind": "source", "ok": False,
                                  "worker": asdict(worker),
                                  "source_index": task.source.source_index,
                                  "source_id": task.source.source_id, "error": str(error),
                                  "traceback": traceback.format_exc()})
    except BaseException as error:
        result_queue.put({"kind": "worker_fatal", "ok": False,
                          "worker": asdict(worker), "error": str(error),
                          "traceback": traceback.format_exc()})
    finally:
        if runtime is not None:
            runtime.close()
        result_queue.put({"kind": "worker_done", "ok": True, "worker": asdict(worker)})


def _pid_is_alive(pid: int) -> bool:
    if pid < 1:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    if temporary.exists():
        raise VisualBehaviorError(f"temporary manifest already exists: {temporary}")
    encoded = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True).encode("utf-8") + b"\n"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as output:
            descriptor = -1
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _prepare_capture_plan(output_root: Path, plan: Mapping[str, Any]) -> Path:
    """Freeze semantics while allowing a different worker topology on resume."""

    plan_path = output_root / "capture_plan.json"
    semantic_record = {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "kind": "dynamic-thinker-behavior-capture-plan",
        "semantic_sha256": plan["semantic_sha256"],
        "semantic": plan["semantic"],
    }
    if plan_path.exists():
        try:
            existing = json.loads(plan_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise VisualBehaviorError(f"cannot read capture plan {plan_path}: {error}") from error
        if existing.get("semantic_sha256") != plan["semantic_sha256"]:
            raise VisualBehaviorError(
                f"output contains different capture semantics: {plan_path}"
            )
        if existing.get("semantic") != plan["semantic"]:
            raise VisualBehaviorError(
                f"capture semantic digest collision or corrupt plan: {plan_path}"
            )
    else:
        _atomic_json(plan_path, semantic_record)

    attempts = output_root / "attempts"
    attempt_path = attempts / f"attempt-{time.time_ns()}-{os.getpid()}.json"
    _atomic_json(
        attempt_path,
        {
            "schema_version": CAPTURE_SCHEMA_VERSION,
            "kind": "dynamic-thinker-behavior-capture-attempt",
            "status": "running",
            "semantic_sha256": plan["semantic_sha256"],
            "execution_plan_sha256": plan["plan_sha256"],
            "execution": plan["execution"],
        },
    )
    return attempt_path


def _finish_attempt(
    attempt_path: Path,
    plan: Mapping[str, Any],
    *,
    status: str,
    records: Sequence[Mapping[str, Any]],
) -> None:
    _atomic_json(
        attempt_path,
        {
            "schema_version": CAPTURE_SCHEMA_VERSION,
            "kind": "dynamic-thinker-behavior-capture-attempt",
            "status": status,
            "semantic_sha256": plan["semantic_sha256"],
            "execution_plan_sha256": plan["plan_sha256"],
            "execution": plan["execution"],
            "record_count": len(records),
        },
    )


def _validate_episode_semantics(
    metadata: Mapping[str, Any],
    validation: Mapping[str, Any],
    semantic: Mapping[str, Any],
    path: Path,
) -> None:
    checkpoint = semantic.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise VisualBehaviorError("capture semantic checkpoint is invalid")
    if semantic.get("seed_protocol") != SEED_PROTOCOL:
        raise VisualBehaviorError("capture semantic seed protocol is unsupported")
    backend_fingerprint = _validate_determinism_backend_fingerprint(
        semantic.get("determinism_backend_fingerprint")
    )
    expected_pairs = (
        ("checkpoint_bundle_sha256", checkpoint.get("bundle_sha256")),
        ("seed_protocol", SEED_PROTOCOL),
        ("policy_mode", semantic.get("policy_mode")),
        ("determinism_backend_fingerprint", backend_fingerprint),
        ("tree_carry", semantic.get("tree_carry")),
        ("images", semantic.get("images")),
        ("chunk_events", semantic.get("chunk_events")),
        ("timing_mode", semantic.get("timing_mode")),
        (
            "trace_event_schema_version",
            semantic.get("trace_event_schema_version"),
        ),
    )
    for key, expected in expected_pairs:
        if metadata.get(key) != expected:
            raise VisualBehaviorError(f"episode {key} semantic mismatch: {path}")
    for key, default in (("compression_level", 3), ("store_predicted_xs", True)):
        if metadata.get(key, default) != semantic.get(key, default):
            raise VisualBehaviorError(f"episode {key} semantic mismatch: {path}")
    base_seed = semantic.get("base_seed")
    episode_index = metadata.get("source_episode_index")
    observed_seed = metadata.get("episode_seed")
    if (
        type(base_seed) is not int
        or type(episode_index) is not int
        or episode_index < 0
        or type(observed_seed) is not int
    ):
        raise VisualBehaviorError(f"episode seed metadata is invalid: {path}")
    try:
        expected_seed = _episode_seed_from_identity(
            base_seed, metadata.get("source_sha256"), episode_index
        )
    except (TypeError, ValueError) as error:
        raise VisualBehaviorError(
            f"episode seed source identity is invalid: {path}"
        ) from error
    if observed_seed != expected_seed:
        raise VisualBehaviorError(
            f"episode_seed does not match the canonical seed derivation: {path}"
        )
    if validation.get("images") != semantic.get("images"):
        raise VisualBehaviorError(f"episode image-store mode mismatch: {path}")
    preprocessing = semantic.get("preprocessing")
    if isinstance(preprocessing, Mapping):
        expected_dtype = "float32" if preprocessing["normalize"] else "uint8"
        for key, expected in (
            ("frame_stack_n", preprocessing["frame_stack_n"]),
            ("target_size", list(preprocessing["target_size"])),
            ("grayscale", preprocessing["grayscale"]),
            ("observation_dtype", expected_dtype),
        ):
            if metadata.get(key) != expected:
                raise VisualBehaviorError(
                    f"episode preprocessing {key} mismatch: {path}"
                )
    try:
        source_observations = int(metadata["source_logical_observation_count"])
        source_edges = int(metadata["source_logical_edge_count"])
        replay_observations = int(metadata["replay_observation_count"])
        replay_edges = int(metadata["replay_edge_count"])
        scored_edges = int(metadata["scored_edge_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise VisualBehaviorError(f"episode alignment metadata is incomplete: {path}") from error
    if not (
        source_observations == source_edges + 1
        and replay_observations == replay_edges + 1
        and replay_observations == source_observations - 1
        and replay_edges == source_edges - 1
        and scored_edges == max(0, replay_edges - 1)
        and int(metadata.get("logical_observation_count", -1)) == replay_observations
        and int(metadata.get("logical_action_count", -1)) == replay_edges
    ):
        raise VisualBehaviorError(f"episode alignment counts are inconsistent: {path}")


def _validate_native_trace_contract(
    trace_api: Any,
    path: Path,
    validation: Mapping[str, Any],
    expected_version: int,
) -> None:
    raw_schema = validation.get("field_schema")
    if not isinstance(raw_schema, list):
        raise VisualBehaviorError(f"episode store lacks a field schema: {path}")
    by_name = {
        str(item.get("name")): item
        for item in raw_schema
        if isinstance(item, Mapping) and item.get("name") is not None
    }
    required = {
        "trace_schema_version",
        "step_times",
        "step_times_valid",
    }
    if expected_version >= 3:
        required.update((
            "actor_real_logits_pre_override", "actor_real_logits_pre_override_valid",
            "model_done_logit", "model_done_logit_valid",
            "model_done_target_depth", "model_done_target_action",
        ))
    store_xs = validation.get("metadata", {}).get("store_predicted_xs", True)
    if type(store_xs) is not bool:
        raise VisualBehaviorError(f"Invalid predicted_xs storage profile: {path}")
    if store_xs:
        required.update(("predicted_xs", "predicted_xs_valid"))
    else:
        required.update(("imagined_image", "imagined_image_valid"))
        if {"predicted_xs", "predicted_xs_valid"} & set(by_name):
            raise VisualBehaviorError(f"Omitted predicted_xs unexpectedly stored: {path}")
    missing = sorted(required - set(by_name))
    if missing:
        raise VisualBehaviorError(
            f"episode store lacks trace-v{expected_version} fields {missing}: {path}"
        )
    if store_xs and str(by_name["predicted_xs"].get("dtype")) not in {"<f4", ">f4", "=f4"}:
        raise VisualBehaviorError(f"episode predicted_xs is not float32: {path}")
    if store_xs and str(by_name["predicted_xs_valid"].get("dtype")) != "|b1":
        raise VisualBehaviorError(f"episode predicted_xs_valid is not boolean: {path}")
    if store_xs and len(by_name["predicted_xs"].get("shape", ())) != 3:
        raise VisualBehaviorError(f"episode predicted_xs is not CHW: {path}")
    if (
        str(by_name["step_times"].get("dtype")) not in {"<f4", ">f4", "=f4"}
        or by_name["step_times"].get("shape") != [1]
        or str(by_name["step_times_valid"].get("dtype")) != "|b1"
    ):
        raise VisualBehaviorError(f"episode timing sentinel schema is invalid: {path}")
    opened = trace_api.open_video_stat(path)
    native = getattr(opened, "native", opened)
    field = getattr(native, "field", None)
    if not callable(field):
        raise VisualBehaviorError(f"episode reader lacks native field access: {path}")
    numpy = importlib.import_module("numpy")
    versions = numpy.asarray(field("trace_schema_version"))
    if versions.shape[:1] != (int(validation.get("event_count", -1)),) or not numpy.all(
        versions == int(expected_version)
    ):
        raise VisualBehaviorError(f"episode trace schema rows disagree: {path}")
    if numpy.any(numpy.asarray(field("step_times_valid"), dtype=numpy.bool_)):
        raise VisualBehaviorError(f"timing_mode=none store claims valid timing: {path}")
    if "compression_level" in validation.get("metadata", {}):
        expected = validation["metadata"]["compression_level"]
        for pack in validation["pack_schema"]:
            codec = json.loads((path / "packs" / pack["name"] / ".zarray").read_text())["compressor"]
            if (codec.get("id"), codec.get("cname"), codec.get("clevel"), codec.get("shuffle")) != ("blosc", "zstd", expected, 2):
                raise VisualBehaviorError(f"Compression profile mismatch: {path}")


def _validate_episode_coverage(
    semantic: Mapping[str, Any], episodes: Sequence[Mapping[str, Any]]
) -> None:
    sources = semantic.get("sources")
    counts = semantic.get("episode_counts")
    if not isinstance(sources, list) or not isinstance(counts, list):
        raise VisualBehaviorError("capture semantic source coverage is incomplete")
    if len(sources) != len(counts) or int(semantic.get("source_count", -1)) != len(sources):
        raise VisualBehaviorError("capture semantic source counts disagree")
    expected: dict[tuple[str, str], tuple[int, int]] = {}
    for source, count in zip(sources, counts):
        if not isinstance(source, Mapping):
            raise VisualBehaviorError("capture semantic source entry is invalid")
        key = (str(source.get("source_id", "")), str(source.get("sha256", "")))
        if not all(key) or key in expected:
            raise VisualBehaviorError("capture semantic source identity is invalid")
        expected[key] = (int(source.get("size", -1)), int(count))
    observed: dict[tuple[str, str], list[int]] = {}
    for episode in episodes:
        metadata = episode.get("metadata")
        if not isinstance(metadata, Mapping):
            raise VisualBehaviorError("capture episode proof lacks metadata")
        key = (str(metadata.get("source_id", "")), str(metadata.get("source_sha256", "")))
        if key not in expected or int(metadata.get("source_size", -1)) != expected[key][0]:
            raise VisualBehaviorError("capture episode source disagrees with semantics")
        observed.setdefault(key, []).append(int(metadata.get("source_episode_index", -1)))
    if set(observed) != set(expected):
        raise VisualBehaviorError("capture source coverage set disagrees")
    requested = {int(value) for value in semantic.get("episode_indices", ())}
    for key, (_, expected_count) in expected.items():
        indices = observed[key]
        if len(indices) != expected_count or len(set(indices)) != len(indices):
            raise VisualBehaviorError("capture per-source episode coverage count disagrees")
        if requested and set(indices) != requested:
            raise VisualBehaviorError("capture selected episode indices disagree")


def _canonical_episode_manifest(
    trace_api: Any,
    output_root: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    expected_episodes: Mapping[str, str],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    episodes: dict[str, dict[str, Any]] = {}
    verified_paths: set[Path] = set()
    canonical_episode_root = (output_root / "episodes").resolve()
    for source_record in records:
        if source_record.get("kind") != "source" or not source_record.get("ok"):
            continue
        payload = source_record.get("payload")
        if not isinstance(payload, Mapping):
            raise VisualBehaviorError("successful source result lacks a payload")
        for episode in payload.get("episodes", ()):
            if not isinstance(episode, Mapping):
                raise VisualBehaviorError("source result contains an invalid episode record")
            episode_id = str(episode.get("episode_id", ""))
            if not episode_id or episode_id in episodes:
                raise VisualBehaviorError(
                    f"duplicate or empty episode result identifier: {episode_id!r}"
                )
            path = Path(str(episode.get("path", ""))).expanduser().resolve(strict=True)
            if not _is_within(path, canonical_episode_root):
                raise VisualBehaviorError(f"episode result escaped episodes root: {path}")
            validation = trace_api.validate_episode_store(path)
            if not isinstance(validation, Mapping):
                raise VisualBehaviorError(f"episode validation returned no manifest: {path}")
            metadata = validation.get("metadata")
            if not isinstance(metadata, Mapping) or metadata.get("episode_id") != episode_id:
                raise VisualBehaviorError(f"episode metadata identity mismatch: {path}")
            expected_source_sha = expected_episodes.get(episode_id)
            if (
                expected_source_sha is not None
                and metadata.get("source_sha256") != expected_source_sha
            ):
                raise VisualBehaviorError(f"episode source identity mismatch: {path}")
            semantic = plan["semantic"]
            _validate_episode_semantics(metadata, validation, semantic, path)
            _validate_native_trace_contract(
                trace_api,
                path,
                validation,
                int(semantic["trace_event_schema_version"]),
            )
            episodes[episode_id] = {
                "episode_id": episode_id,
                "path": path.relative_to(output_root).as_posix(),
                "event_count": int(validation["event_count"]),
                "files_sha256": str(validation["files_sha256"]),
                "images": str(validation["images"]),
                "metadata": _json_safe(dict(metadata)),
            }
            verified_paths.add(path)
    if set(episodes) != set(expected_episodes):
        missing = sorted(set(expected_episodes) - set(episodes))
        extra = sorted(set(episodes) - set(expected_episodes))
        raise VisualBehaviorError(
            "verified episode identity set mismatch: "
            f"missing={missing}, extra={extra}"
        )
    disk_paths = _episode_store_paths(output_root)
    if disk_paths != verified_paths:
        missing = sorted(str(path) for path in verified_paths - disk_paths)
        orphaned = sorted(str(path) for path in disk_paths - verified_paths)
        raise VisualBehaviorError(
            "canonical episodes directory disagrees with verified result set: "
            f"missing={missing}, orphaned={orphaned}"
        )
    ordered = [episodes[key] for key in sorted(episodes)]
    _validate_episode_coverage(plan["semantic"], ordered)
    manifest: dict[str, Any] = {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "kind": "dynamic-thinker-behavior-capture-manifest",
        "state": "complete",
        "semantic_sha256": plan["semantic_sha256"],
        "semantic": plan["semantic"],
        "episode_count": len(ordered),
        "episodes": ordered,
        "episodes_sha256": hashlib.sha256(_canonical_json(ordered)).hexdigest(),
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        _canonical_json(manifest)
    ).hexdigest()
    return manifest


def _publish_capture_completion(output_root: Path, manifest: Mapping[str, Any]) -> None:
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise VisualBehaviorError(
                f"cannot read root manifest {manifest_path}: {error}"
            ) from error
        if existing != manifest:
            raise VisualBehaviorError(
                f"existing root manifest disagrees with verified stores: {manifest_path}"
            )
    else:
        _atomic_json(manifest_path, manifest)
    completed = {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "kind": "dynamic-thinker-behavior-capture-completion",
        "status": "COMPLETED",
        "semantic_sha256": manifest["semantic_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "episode_count": manifest["episode_count"],
    }
    completed_path = output_root / "COMPLETED.json"
    if completed_path.exists():
        try:
            existing_completed = json.loads(
                completed_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise VisualBehaviorError(
                f"cannot read completion marker {completed_path}: {error}"
            ) from error
        if existing_completed != completed:
            raise VisualBehaviorError(
                f"completion marker disagrees with root manifest: {completed_path}"
            )
    else:
        _atomic_json(completed_path, completed)


def _episode_store_paths(output_root: Path) -> set[Path]:
    """Return direct, non-partial canonical ``*.zarr`` episode entries."""

    episode_root = output_root / "episodes"
    if not episode_root.is_dir():
        return set()
    return {
        path.resolve(strict=True)
        for path in episode_root.iterdir()
        if path.name.endswith(".zarr")
    }


def validate_capture_output(
    output_root: str | Path,
    *,
    trace_api: Any = None,
    expected_semantic_sha256: Optional[str] = None,
    expected_episodes: Optional[Mapping[str, str]] = None,
) -> dict[str, Any]:
    """Validate a completed multi-episode capture without materializing arrays."""

    root = Path(output_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise VisualBehaviorError(f"capture output is not a directory: {root}")
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        completed = json.loads((root / "COMPLETED.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VisualBehaviorError(
            f"completed capture lacks readable root proof files: {root}: {error}"
        ) from error
    if not isinstance(manifest, Mapping) or not isinstance(completed, Mapping):
        raise VisualBehaviorError("capture root proofs must be JSON objects")
    unsigned = dict(manifest)
    claimed_manifest_sha = unsigned.pop("manifest_sha256", None)
    actual_manifest_sha = hashlib.sha256(_canonical_json(unsigned)).hexdigest()
    if claimed_manifest_sha != actual_manifest_sha:
        raise VisualBehaviorError("capture root manifest SHA-256 is invalid")
    semantic = manifest.get("semantic")
    semantic_sha = manifest.get("semantic_sha256")
    if not isinstance(semantic, Mapping) or hashlib.sha256(
        _canonical_json(semantic)
    ).hexdigest() != semantic_sha:
        raise VisualBehaviorError("capture semantic SHA-256 is invalid")
    if expected_semantic_sha256 is not None and semantic_sha != expected_semantic_sha256:
        raise VisualBehaviorError("completed capture has different semantics")
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list):
        raise VisualBehaviorError("capture manifest episodes must be a list")
    if hashlib.sha256(_canonical_json(episodes)).hexdigest() != manifest.get(
        "episodes_sha256"
    ):
        raise VisualBehaviorError("capture episode-list SHA-256 is invalid")
    if int(manifest.get("episode_count", -1)) != len(episodes):
        raise VisualBehaviorError("capture episode count is invalid")
    expected_completion = {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "kind": "dynamic-thinker-behavior-capture-completion",
        "status": "COMPLETED",
        "semantic_sha256": semantic_sha,
        "manifest_sha256": claimed_manifest_sha,
        "episode_count": len(episodes),
    }
    if dict(completed) != expected_completion:
        raise VisualBehaviorError("capture COMPLETED marker disagrees with manifest")
    trace = _load_behavior_trace() if trace_api is None else trace_api
    observed: dict[str, str] = {}
    manifest_paths: set[Path] = set()
    for episode in episodes:
        if not isinstance(episode, Mapping):
            raise VisualBehaviorError("capture manifest contains an invalid episode")
        episode_id = str(episode.get("episode_id", ""))
        relative = Path(str(episode.get("path", "")))
        if not episode_id or relative.is_absolute():
            raise VisualBehaviorError("capture episode identity/path is invalid")
        path = (root / relative).resolve(strict=True)
        if not _is_within(path, root / "episodes"):
            raise VisualBehaviorError(f"capture episode escaped episodes root: {path}")
        if episode_id in observed:
            raise VisualBehaviorError(f"duplicate capture episode: {episode_id}")
        manifest_paths.add(path)
        validation = trace.validate_episode_store(path)
        metadata = validation.get("metadata")
        if not isinstance(metadata, Mapping) or metadata.get("episode_id") != episode_id:
            raise VisualBehaviorError(f"capture episode metadata mismatch: {path}")
        for key in ("event_count", "files_sha256", "images"):
            if validation.get(key) != episode.get(key):
                raise VisualBehaviorError(
                    f"capture episode {key} proof mismatch: {path}"
                )
        if _json_safe(dict(metadata)) != episode.get("metadata"):
            raise VisualBehaviorError(f"capture episode metadata proof mismatch: {path}")
        _validate_episode_semantics(metadata, validation, semantic, path)
        _validate_native_trace_contract(
            trace,
            path,
            validation,
            int(semantic.get("trace_event_schema_version", -1)),
        )
        observed[episode_id] = str(metadata.get("source_sha256", ""))
    if expected_episodes is not None and observed != dict(expected_episodes):
        raise VisualBehaviorError("completed capture episode/source set disagrees")
    disk_paths = _episode_store_paths(root)
    if disk_paths != manifest_paths:
        missing = sorted(str(path) for path in manifest_paths - disk_paths)
        orphaned = sorted(str(path) for path in disk_paths - manifest_paths)
        raise VisualBehaviorError(
            "completed capture has missing or orphaned canonical episode stores: "
            f"missing={missing}, orphaned={orphaned}"
        )
    _validate_episode_coverage(semantic, episodes)
    return dict(manifest)


def _run_workers(
    config: CaptureConfig,
    source_tasks: Sequence[SourceTask],
    workers: Sequence[WorkerSpec],
) -> list[Mapping[str, Any]]:
    context = multiprocessing.get_context("spawn")
    tasks = context.Queue(maxsize=max(len(workers) * 2, 1))
    # Worker payloads are deliberately compact, and the result channel is also
    # bounded so a slow coordinator cannot accumulate an unbounded backlog.
    results = context.Queue(maxsize=max(len(workers) * 2, 1))
    processes: list[Any] = []
    feeder_errors: queue.Queue[tuple[BaseException, str]] = queue.Queue(maxsize=1)
    stop_feeder = threading.Event()

    def put_task(value: Any) -> bool:
        while not stop_feeder.is_set():
            try:
                tasks.put(value, timeout=0.25)
                return True
            except queue.Full:
                if processes and all(not process.is_alive() for process in processes):
                    return False
        return False

    def feed() -> None:
        try:
            for source_task in source_tasks:
                if not put_task(source_task):
                    return
        except BaseException as error:
            try:
                feeder_errors.put_nowait((error, traceback.format_exc()))
            except queue.Full:
                pass
        finally:
            # On a normal or feeder-error path, every surviving worker gets a
            # sentinel.  The coordinator's outer finally handles interruption.
            for _ in processes:
                if not put_task(None):
                    break

    feeder: Optional[threading.Thread] = None
    records: list[Mapping[str, Any]] = []
    try:
        for worker in workers:
            process = context.Process(
                target=_worker_entry,
                args=(config, worker, tasks, results),
                name=(
                    f"behavior-capture-{worker.rank}-"
                    f"{worker.device.replace(':', '-')}"
                ),
            )
            process.start()
            processes.append(process)
        feeder = threading.Thread(
            target=feed, name="source-bundle-feeder", daemon=True
        )
        feeder.start()
        done_workers = 0
        feeder_error_reported = False
        while done_workers < len(processes):
            if not feeder_error_reported and not feeder_errors.empty():
                error, formatted = feeder_errors.get_nowait()
                records.append(
                    {
                        "kind": "orchestrator_error",
                        "ok": False,
                        "error": f"source feeder failed: {error}",
                        "traceback": formatted,
                    }
                )
                feeder_error_reported = True
            try:
                record = results.get(timeout=0.5)
            except queue.Empty:
                if all(not process.is_alive() for process in processes):
                    break
                continue
            if record.get("kind") == "worker_done":
                done_workers += 1
            else:
                records.append(record)
                if record.get("kind") == "source":
                    source_id = record.get(
                        "source_id", record.get("payload", {}).get("source_id")
                    )
                    print(
                        f"[{'ok' if record.get('ok') else 'failed'}] {source_id}",
                        flush=True,
                    )
    finally:
        stop_feeder.set()
        # Best effort wake-up before terminating.  Queue capacity may already
        # be full, so this must never block cleanup.
        for _ in processes:
            try:
                tasks.put_nowait(None)
            except (queue.Full, ValueError, OSError):
                break
        if feeder is not None:
            feeder.join(timeout=5.0)
        for process in processes:
            process.join(timeout=5.0)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
        for channel in (tasks, results):
            try:
                channel.close()
                channel.join_thread()
            except (AttributeError, OSError, ValueError):
                pass
    bad_exits = [process for process in processes if process.exitcode != 0]
    if bad_exits:
        records.append({"kind": "orchestrator_error", "ok": False,
                        "error": "worker process exited abnormally",
                        "workers": [{"name": process.name, "exitcode": process.exitcode}
                                    for process in bad_exits]})
    source_records = [record for record in records if record.get("kind") == "source"]
    seen = {
        int(record["payload"]["source_index"] if record.get("ok") else record["source_index"])
        for record in source_records
    }
    missing = sorted(set(range(len(source_tasks))) - seen)
    if missing:
        records.append({"kind": "orchestrator_error", "ok": False,
                        "error": "source tasks completed without a result",
                        "missing_source_indices": missing})
    return records


def _capture(args: argparse.Namespace) -> int:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    checkpoint = resolve_checkpoint_bundle(
        args.checkpoint_dir, requested_step=args.checkpoint_step,
        stable_polls=args.stable_polls, stable_interval_seconds=args.stable_interval,
    )
    checkpoint_payload_validation = validate_checkpoint_payload_bundle(checkpoint)
    sources = enumerate_source_bundles(
        args.data_root, selectors=args.source, pattern=args.pattern,
        stable_polls=args.stable_polls, stable_interval_seconds=args.stable_interval,
        hash_workers=args.cpu_workers,
        game_id=int(checkpoint_payload_validation["game_id"]),
    )
    workers = resolve_worker_specs(
        args.devices, source_count=len(sources), cpu_workers=args.cpu_workers,
        dry_run=args.dry_run,
        checkpoint_bytes=checkpoint.actor.size + checkpoint.model.size,
    )
    determinism_backend_fingerprint = resolve_determinism_backend_fingerprint(
        workers, dry_run=bool(args.dry_run)
    )
    if int(args.workers_per_device) != 1:
        raise VisualBehaviorError(
            "--workers-per-device must be exactly 1; multiple CUDA owners break the "
            "episode/device memory contract"
        )
    torch_threads = resolve_torch_threads(
        str(args.torch_threads_per_worker), len(workers)
    )
    output_root = Path(args.output).expanduser().absolute()
    config = CaptureConfig(
        checkpoint=checkpoint,
        data_root=str(Path(args.data_root).expanduser().resolve()),
        output_root=str(output_root),
        scratch_root=str(Path(args.scratch_dir).expanduser().absolute()) if args.scratch_dir else None,
        base_seed=int(args.base_seed), queue_size=int(args.queue_size),
        chunk_events=int(args.chunk_events), images=str(args.images),
        tree_carry=bool(args.tree_carry), policy_mode=str(args.policy_mode),
        determinism_backend_fingerprint=determinism_backend_fingerprint,
        resume=bool(args.resume),
        torch_threads_per_worker=torch_threads,
        source_hash_workers=resolve_hash_workers(args.cpu_workers, len(sources)),
        episode_indices=tuple(sorted(set(int(value) for value in args.episode_index))),
        compression_level=int(args.compression_level),
        store_predicted_xs=not args.omit_predicted_xs,
    )
    if args.dry_run:
        plan = build_capture_plan(config, sources, workers)
        print(
            json.dumps(
                {
                    "checkpoint_payload_validation": checkpoint_payload_validation,
                    "plan": plan,
                },
                sort_keys=True,
                indent=2,
            )
        )
        return 0

    source_tasks = _coordinator_source_tasks(config, sources)
    plan = build_capture_plan(config, sources, workers, source_tasks)
    output_root.mkdir(parents=True, exist_ok=True)
    expected_episodes: dict[str, str] = {}
    for task in source_tasks:
        for descriptor in task.episodes:
            store_id = episode_store_id(
                task.source, int(descriptor.local_episode_index)
            )
            if store_id in expected_episodes:
                raise VisualBehaviorError(
                    f"duplicate canonical episode identifier: {store_id}"
                )
            expected_episodes[store_id] = task.source.sha256
    if (output_root / "COMPLETED.json").exists():
        if not config.resume:
            raise VisualBehaviorError(
                f"capture output is already complete; pass --resume to validate: {output_root}"
            )
        validate_capture_output(
            output_root,
            trace_api=_load_behavior_trace(),
            expected_semantic_sha256=plan["semantic_sha256"],
            expected_episodes=expected_episodes,
        )
        print(f"validated existing completed capture at {output_root}")
        return 0
    attempt_path = _prepare_capture_plan(output_root, plan)
    records: list[Mapping[str, Any]] = []
    try:
        records = _run_workers(config, source_tasks, workers)
        records = sorted(records, key=lambda item: (
            int(item.get("payload", {}).get("source_index", item.get("source_index", 2**63 - 1))),
            str(item.get("kind", "")),
        ))
        ok = all(bool(record.get("ok")) for record in records) and len(
            [record for record in records if record.get("kind") == "source"]
        ) == len(sources)
        final = {"schema_version": CAPTURE_SCHEMA_VERSION,
                 "kind": "dynamic-thinker-behavior-capture-result",
                 "status": "workers_complete" if ok else "failed",
                 "semantic_sha256": plan["semantic_sha256"],
                 "execution_plan_sha256": plan["plan_sha256"],
                 "source_count": len(sources), "records": records}
        _atomic_json(output_root / "capture_result.json", final)
        bound_inputs: list[StableFile] = [
            checkpoint.config,
            checkpoint.actor,
            checkpoint.model,
        ]
        if checkpoint.milestone is not None:
            bound_inputs.append(checkpoint.milestone)
        for record in bound_inputs:
            if stable_file(record.path, polls=1).sha256 != record.sha256:
                raise VisualBehaviorError(
                    f"immutable input changed during capture: {record.path}"
                )
        # Bind completion to the bytes still present after replay.  A metadata
        # fence alone cannot detect same-size, same-inode content replacement
        # whose timestamp was restored, so every selected source is re-hashed.
        validate_source_snapshots(
            sources, hash_workers=config.source_hash_workers
        )
        if not ok:
            raise VisualBehaviorError(
                f"capture failed; see {output_root / 'capture_result.json'}"
            )
        root_manifest = _canonical_episode_manifest(
            _load_behavior_trace(),
            output_root,
            records,
            expected_episodes=expected_episodes,
            plan=plan,
        )
        final["status"] = "complete"
        final["manifest_sha256"] = root_manifest["manifest_sha256"]
        _atomic_json(output_root / "capture_result.json", final)
        # COMPLETED is deliberately the final canonical publish after every
        # other proof and diagnostic result is durable.
        _publish_capture_completion(output_root, root_manifest)
    except BaseException as error:
        failed_result = {
            "schema_version": CAPTURE_SCHEMA_VERSION,
            "kind": "dynamic-thinker-behavior-capture-result",
            "status": "failed",
            "semantic_sha256": plan["semantic_sha256"],
            "execution_plan_sha256": plan["plan_sha256"],
            "source_count": len(sources),
            "error": str(error),
            "records": records,
        }
        try:
            _atomic_json(output_root / "capture_result.json", failed_result)
        except BaseException:
            pass
        _finish_attempt(attempt_path, plan, status="failed", records=records)
        raise
    _finish_attempt(attempt_path, plan, status="complete", records=records)
    print(f"captured {len(sources)} source bundles into {output_root}")
    return 0


def _array_description(value: Any) -> Mapping[str, Any]:
    shape, dtype = getattr(value, "shape", None), getattr(value, "dtype", None)
    if shape is not None and dtype is not None:
        return {"shape": [int(item) for item in shape], "dtype": str(dtype)}
    return {"type": type(value).__name__}


def _describe_opened(value: Any) -> Mapping[str, Any]:
    result: dict[str, Any] = {"type": type(value).__name__}
    native = getattr(value, "native", None)
    if native is not None:
        result["native_event_count"] = int(native.event_count)
        result["native_fields"] = {
            name: _array_description(native.field(name))
            for name in sorted(native.field_names)
        }
        result["metadata"] = _json_safe(dict(native.metadata))
        # LegacyProjection implements Mapping, but iterating its items would
        # materialize every projected tree/latent array during an inspection.
        result["legacy_projection"] = "lazy-not-materialized"
        return result
    attrs = getattr(value, "attrs", None)
    if attrs is not None:
        try:
            result["attrs"] = _json_safe(dict(attrs))
        except (TypeError, ValueError):
            pass
    if isinstance(value, Mapping):
        result["fields"] = {str(key): _array_description(item) for key, item in value.items()}
        return result
    array_keys, group_keys = getattr(value, "array_keys", None), getattr(value, "group_keys", None)
    if callable(array_keys):
        result["arrays"] = {str(key): _array_description(value[key]) for key in sorted(array_keys())}
    if callable(group_keys):
        result["groups"] = sorted(str(key) for key in group_keys())
    summary = getattr(value, "summary", None)
    if callable(summary):
        result["summary"] = _json_safe(summary())
    return result


def _inspect(args: argparse.Namespace) -> int:
    trace = _load_behavior_trace()
    path = Path(args.store).expanduser().resolve(strict=True)
    if path.is_dir() and (path / "COMPLETED.json").is_file():
        validation = validate_capture_output(path, trace_api=trace)
        print(
            json.dumps(
                {"path": str(path), "capture": _json_safe(validation)},
                sort_keys=True,
                indent=2,
            )
        )
        return 0
    opened = trace.open_video_stat(path)
    validation = (
        trace.validate_episode_store(path)
        if path.is_dir()
        else {"format": "legacy-npy", "state": "opened-as-trusted-legacy"}
    )
    print(json.dumps({"path": str(path), "validation": _json_safe(validation),
                      "store": _describe_opened(opened)}, sort_keys=True, indent=2))
    return 0


def _lookup_array(opened: Any, candidates: Sequence[str]) -> Any:
    native = getattr(opened, "native", None)
    for candidate in candidates:
        if native is not None and callable(getattr(native, "has_field", None)):
            if native.has_field(candidate):
                return native.field(candidate)
        try:
            if isinstance(opened, Mapping) and candidate in opened:
                return opened[candidate]
            return opened[candidate]
        except (KeyError, IndexError, TypeError):
            continue
    return None


def _uint8_hwc(frame: Any) -> Any:
    numpy = importlib.import_module("numpy")
    array = numpy.asarray(frame)
    while array.ndim > 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 2:
        array = array[..., numpy.newaxis]
    if array.ndim != 3:
        raise VisualBehaviorError(f"cannot render image with shape {array.shape}")
    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = numpy.transpose(array, (1, 2, 0))
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.shape[-1] == 1:
        array = numpy.repeat(array, 3, axis=-1)
    if numpy.issubdtype(array.dtype, numpy.floating):
        array = numpy.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
        if array.size and float(numpy.max(array)) <= 1.0:
            array = array * 255.0
        array = numpy.rint(numpy.clip(array, 0.0, 255.0)).astype(numpy.uint8)
    else:
        array = numpy.clip(array, 0, 255).astype(numpy.uint8, copy=False)
    return numpy.ascontiguousarray(array)


def _iter_forward_filled_render_rows(
    arrays: Sequence[Any], validities: Sequence[Any], indices: Iterable[int]
) -> Iterator[tuple[int, list[Any]]]:
    """Yield panels while holding each one's latest valid native frame."""

    numpy = importlib.import_module("numpy")
    last_valid: list[Any] = [None] * len(arrays)
    for index_value in indices:
        index = int(index_value)
        for panel, (array, validity) in enumerate(zip(arrays, validities)):
            valid = (
                True
                if validity is None
                else bool(numpy.asarray(validity[index]).reshape(-1)[0])
            )
            if valid:
                last_valid[panel] = _uint8_hwc(array[index])
        # A sparse panel may have no state before its first valid row.  Skip
        # only that prefix; once established, invalid rows hold the last root.
        if all(frame is not None for frame in last_valid):
            yield index, list(last_valid)


def _render(args: argparse.Namespace) -> int:
    trace = _load_behavior_trace()
    store = Path(args.store).expanduser().resolve(strict=True)
    if store.is_dir():
        trace.validate_episode_store(store)
    opened = trace.open_video_stat(store)
    real = _lookup_array(
        opened,
        (
            "real_image", "real_imgs", "real_images", "real/images",
            "post_real_states", "pre_real_states",
        ),
    )
    imagined = _lookup_array(
        opened,
        (
            "imagined_image", "im_imgs", "imagined_images", "imagined/images",
            "post_xs", "pre_xs",
        ),
    )
    real_valid = _lookup_array(opened, ("real_image_valid",))
    imagined_valid = _lookup_array(opened, ("imagined_image_valid",))
    selected = [real] if args.view == "real" else [imagined] if args.view == "imagined" else [real, imagined]
    selected_validities = (
        [real_valid]
        if args.view == "real"
        else [imagined_valid]
        if args.view == "imagined"
        else [real_valid, imagined_valid]
    )
    if any(value is None for value in selected):
        raise VisualBehaviorError(
            f"store lacks images required by --view={args.view}; legacy 10-key NPY "
            "does not contain display frames, and native stores must be captured "
            "with the matching --images mode"
        )
    native_event_count = min(len(value) for value in selected)
    numpy = importlib.import_module("numpy")
    if args.view == "real" and real_valid is not None:
        render_indices = numpy.flatnonzero(numpy.asarray(real_valid[:], dtype=numpy.bool_))
    elif args.view == "imagined" and imagined_valid is not None:
        render_indices = numpy.flatnonzero(
            numpy.asarray(imagined_valid[:], dtype=numpy.bool_)
        )
    elif args.view == "both" and real_valid is not None and imagined_valid is not None:
        render_indices = numpy.flatnonzero(
            numpy.asarray(real_valid[:], dtype=numpy.bool_)
            | numpy.asarray(imagined_valid[:], dtype=numpy.bool_)
        )
    else:
        render_indices = numpy.arange(native_event_count, dtype=numpy.int64)
    render_indices = render_indices[render_indices < native_event_count]
    if args.max_frames is not None:
        render_indices = render_indices[: int(args.max_frames)]
    rows = _iter_forward_filled_render_rows(
        selected, selected_validities, render_indices
    )
    try:
        _, first = next(rows)
    except StopIteration as error:
        raise VisualBehaviorError("store has no renderable frames") from error
    try:
        cv2 = importlib.import_module("cv2")
    except ModuleNotFoundError as error:
        raise RuntimeDependencyError("render requires OpenCV (cv2)") from error
    output = Path(args.output).expanduser().absolute()
    if output.exists() and not args.overwrite:
        raise VisualBehaviorError(f"output exists; pass --overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.partial-{os.getpid()}{output.suffix or '.avi'}")
    height = max(frame.shape[0] for frame in first)
    width = sum(int(round(frame.shape[1] * height / frame.shape[0])) for frame in first)
    scale = int(args.scale)
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"FFV1"),
                             float(args.fps), (width * scale, height * scale))
    if not writer.isOpened():
        raise VisualBehaviorError(f"OpenCV could not open video output: {temporary}")
    frame_count = 0

    def write_frames(frames: Sequence[Any]) -> None:
        nonlocal frame_count
        frames = [cv2.resize(frame,
                             (int(round(frame.shape[1] * height / frame.shape[0])), height),
                             interpolation=cv2.INTER_NEAREST)
                  if frame.shape[0] != height else frame for frame in frames]
        rgb = numpy.concatenate(frames, axis=1)
        if scale != 1:
            rgb = cv2.resize(rgb, (width * scale, height * scale),
                             interpolation=cv2.INTER_NEAREST)
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        frame_count += 1

    try:
        write_frames(first)
        for _, frames in rows:
            write_frames(frames)
    finally:
        writer.release()
    os.replace(temporary, output)
    print(f"rendered {frame_count} frames to {output}")
    return 0


def _export_legacy(args: argparse.Namespace) -> int:
    trace = _load_behavior_trace()
    store = Path(args.store).expanduser().resolve(strict=True)
    output = Path(args.output).expanduser().absolute()
    if output.exists() and not args.overwrite:
        raise VisualBehaviorError(f"output exists; pass --overwrite: {output}")
    trace.validate_episode_store(store)
    start = args.start
    stop = args.stop
    if start is not None and start < 0:
        raise VisualBehaviorError("--start must be nonnegative")
    if stop is not None and stop < 0:
        raise VisualBehaviorError("--stop must be nonnegative")
    if start is not None and stop is not None and start >= stop:
        raise VisualBehaviorError("--start must be smaller than --stop")
    destination = output
    replacing = output.exists()
    if replacing:
        if not output.is_file():
            raise VisualBehaviorError(f"legacy output is not a regular file: {output}")
        destination = output.with_name(
            f".{output.name}.replace-partial-{os.getpid()}-{time.time_ns()}"
        )
    try:
        result = trace.export_legacy(
            store,
            destination,
            episode_slice=slice(start, stop),
        )
        if replacing:
            os.replace(destination, output)
            result = output
    finally:
        if destination != output and destination.exists():
            destination.unlink()
    print(json.dumps({"output": str(output), "result": _json_safe(result)}, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture", help="capture complete human episodes")
    capture.add_argument("--checkpoint-dir", required=True)
    capture.add_argument("--checkpoint-step", type=int, default=None)
    capture.add_argument("--data-root", required=True)
    capture.add_argument("--source", action="append", default=[],
                         help="relative file, directory or glob under data root; repeatable")
    capture.add_argument(
        "--episode-index",
        action="append",
        type=int,
        default=[],
        help="capture only this source-local complete episode; repeatable",
    )
    capture.add_argument("--pattern", default=DEFAULT_SOURCE_PATTERN)
    capture.add_argument("--output", required=True)
    capture.add_argument("--scratch-dir", default=None)
    capture.add_argument("--devices", default="auto",
                         help="auto, cpu, or comma-separated logical CUDA indices")
    capture.add_argument("--cpu-workers", default="auto")
    capture.add_argument("--base-seed", type=int, default=0)
    capture.add_argument(
        "--max-inflight-chunks", "--queue-size", dest="queue_size",
        type=int, default=DEFAULT_QUEUE_SIZE,
    )
    capture.add_argument("--chunk-events", type=int, default=DEFAULT_CHUNK_EVENTS)
    capture.add_argument("--compression-level", type=int, choices=range(10), default=3,
                         help="Lossless Zstd/bitshuffle compression level (0..9)")
    capture.add_argument("--omit-predicted-xs", action="store_true",
                         help="Omit raw float32 XS from disk only; requires --images imagined/all")
    capture.add_argument(
        "--images", choices=("none", "real", "imagined", "all"), default="none"
    )
    capture.add_argument(
        "--policy-mode", choices=CAPTURE_POLICY_MODES, default="checkpoint"
    )
    capture.add_argument("--workers-per-device", type=int, default=1)
    capture.add_argument(
        "--torch-threads-per-worker",
        default="4",
        help="positive integer (default: 4), or auto",
    )
    capture.add_argument("--no-tree-carry", dest="tree_carry", action="store_false", default=True)
    capture.add_argument("--resume", action="store_true")
    capture.add_argument("--dry-run", action="store_true")
    capture.add_argument("--stable-polls", type=int, default=2)
    capture.add_argument("--stable-interval", type=float, default=0.05)
    capture.set_defaults(handler=_capture)

    inspect_parser = subparsers.add_parser("inspect", help="validate and describe a store")
    inspect_parser.add_argument("store")
    inspect_parser.set_defaults(handler=_inspect)
    render = subparsers.add_parser("render", help="stream stored images to a video")
    render.add_argument("store")
    render.add_argument("--output", required=True)
    render.add_argument("--view", choices=("real", "imagined", "both"), default="both")
    render.add_argument("--fps", type=float, default=5.0)
    render.add_argument("--scale", type=int, default=3)
    render.add_argument("--max-frames", type=int, default=None)
    render.add_argument("--overwrite", action="store_true")
    render.set_defaults(handler=_render)
    legacy = subparsers.add_parser("export-legacy", help="create compatibility NPY")
    legacy.add_argument("store")
    legacy.add_argument("--output", required=True)
    legacy.add_argument("--start", type=int, default=None,
                        help="first projected legacy row (0-based, inclusive)")
    legacy.add_argument("--stop", type=int, default=None,
                        help="last projected legacy row (0-based, exclusive)")
    legacy.add_argument("--overwrite", action="store_true")
    legacy.set_defaults(handler=_export_legacy)
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.command == "capture":
        if args.checkpoint_step is not None and args.checkpoint_step < 1:
            parser.error("--checkpoint-step must be positive")
        if args.queue_size < 1:
            parser.error("--max-inflight-chunks must be positive")
        if args.workers_per_device != 1:
            parser.error("--workers-per-device must be exactly 1")
        if args.chunk_events < 1:
            parser.error("--chunk-events must be positive")
        if args.stable_polls < 1:
            parser.error("--stable-polls must be positive")
        if args.stable_interval < 0:
            parser.error("--stable-interval must be nonnegative")
        if any(value < 0 for value in args.episode_index):
            parser.error("--episode-index must be nonnegative")
    elif args.command == "render":
        if args.fps <= 0:
            parser.error("--fps must be positive")
        if args.scale < 1:
            parser.error("--scale must be positive")
        if args.max_frames is not None and args.max_frames < 1:
            parser.error("--max-frames must be positive")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(args, parser)
    try:
        return int(args.handler(args))
    except (VisualBehaviorError, FileNotFoundError, NotADirectoryError) as error:
        print(f"visual_behav: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
