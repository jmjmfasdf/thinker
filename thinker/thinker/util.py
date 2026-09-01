__version__ = "1.3.0"
__project__ = "thinker"

import collections
import copy
import hashlib
import io
import json
import random
import time
import timeit
import yaml
import argparse
import subprocess
from collections import namedtuple
import os
import re
import stat as stat_module
import sys
import math
import logging
from types import MappingProxyType
from matplotlib import pyplot as plt
from gymnasium import spaces
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


_COMPLETION_CHECKPOINT_FILES = (
    "config_c.yaml",
    "ckp_actor.tar",
    "ckp_model.tar",
)
_SCHEMA13_COMPLETION_CHECKPOINT_FILES = (
    *_COMPLETION_CHECKPOINT_FILES,
    "voc_telemetry_manifest.json",
)
_TRAINING_IMPLEMENTATION_FILES = (
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
)
_SCHEMA13_TRAINING_IMPLEMENTATION_FILES = (
    *_TRAINING_IMPLEMENTATION_FILES,
    "thinker/voc_telemetry.py",
)


def _stable_regular_file_bytes(path, *, label):
    """Read one single-link regular inode and prove the path stayed bound."""

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    fd = os.open(os.fspath(path), flags)
    try:
        before = os.fstat(fd)
        if not stat_module.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"{label} must be a single-link regular file")
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
        after = os.fstat(fd)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_after != identity_before or len(payload) != before.st_size:
            raise RuntimeError(f"{label} changed during its descriptor read")
        path_state = os.stat(os.fspath(path), follow_symlinks=False)
        if (
            not stat_module.S_ISREG(path_state.st_mode)
            or path_state.st_nlink != 1
            or (path_state.st_dev, path_state.st_ino)
            != (before.st_dev, before.st_ino)
        ):
            raise RuntimeError(f"{label} pathname changed around its bound read")
        return payload, before
    finally:
        os.close(fd)


def _file_sha256(path):
    # Preserve the inherited schema<=12 pathname/link/error behavior exactly.
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clear_run_completion(ckpdir):
    """Invalidate an earlier success marker before starting or resuming."""

    marker = os.path.join(os.path.abspath(ckpdir), "finish")
    if os.path.lexists(marker):
        if not (os.path.isfile(marker) or os.path.islink(marker)):
            raise IsADirectoryError(f"completion marker is not a file: {marker}")
        os.unlink(marker)


def _collect_run_completion_snapshot_raw(ckpdir, *, gate_schema=None):
    """Collect completion bytes without invoking a semantic final validator."""

    root = os.path.abspath(ckpdir)
    schema13 = (
        type(gate_schema) is int
        and gate_schema == VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION
    )
    if gate_schema is None:
        checkpoint_names = _COMPLETION_CHECKPOINT_FILES
        implementation_names = _TRAINING_IMPLEMENTATION_FILES
    elif schema13:
        checkpoint_names = _SCHEMA13_COMPLETION_CHECKPOINT_FILES
        implementation_names = _SCHEMA13_TRAINING_IMPLEMENTATION_FILES
    elif type(gate_schema) is int and gate_schema in (
        VOC_GATE_POLICY_VERSION_BARRIER_SCHEMA_VERSION,
        VOC_GATE_POLICY_MODEL_INPUT_SEAL_SCHEMA_VERSION,
        VOC_GATE_POLICY_HALF_SQUARED_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_COMMON_MODE_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_HUBER_COMMON_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION,
    ):
        checkpoint_names = _COMPLETION_CHECKPOINT_FILES
        implementation_names = _TRAINING_IMPLEMENTATION_FILES
    else:
        raise ValueError(
            "completion evidence gate_schema must be an exact supported "
            "Python integer"
        )
    checkpoint_files = {}
    checkpoint_payloads = {}
    for name in checkpoint_names:
        path = os.path.join(root, name)
        if not (os.path.lexists(path) if schema13 else os.path.isfile(path)):
            raise FileNotFoundError(
                f"cannot complete run without final checkpoint file: {path}"
            )
        if schema13:
            payload, _ = _stable_regular_file_bytes(
                path, label=f"completion checkpoint {name}"
            )
            checkpoint_payloads[name] = payload
            checkpoint_files[name] = {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size": len(payload),
            }
        else:
            checkpoint_files[name] = {
                "sha256": _file_sha256(path),
                "size": os.path.getsize(path),
            }

    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sources = {}
    for relative in implementation_names:
        path = os.path.join(package_root, relative)
        if not (os.path.lexists(path) if schema13 else os.path.isfile(path)):
            raise FileNotFoundError(
                f"cannot complete run without implementation source: {path}"
            )
        if schema13:
            payload, _ = _stable_regular_file_bytes(
                path, label=f"completion implementation source {relative}"
            )
            digest = hashlib.sha256(payload).hexdigest()
        else:
            digest = _file_sha256(path)
        sources[relative] = {"sha256": digest}

    loaded_extensions = {}
    cenv_module = sys.modules.get("thinker.cenv")
    cenv_path = getattr(cenv_module, "__file__", None)
    if cenv_path and (
        os.path.lexists(cenv_path) if schema13 else os.path.isfile(cenv_path)
    ):
        relative = os.path.relpath(os.path.abspath(cenv_path), package_root)
        if schema13:
            payload, _ = _stable_regular_file_bytes(
                cenv_path, label=f"completion extension {relative}"
            )
            digest = hashlib.sha256(payload).hexdigest()
        else:
            digest = _file_sha256(cenv_path)
        loaded_extensions[relative] = {
            "sha256": digest
        }
    if not loaded_extensions:
        raise RuntimeError(
            "cannot complete run before the thinker.cenv extension is loaded"
        )

    evidence = {
        "checkpoint_files": checkpoint_files,
        "implementation_sources": sources,
        "loaded_extensions": loaded_extensions,
    }
    return evidence, checkpoint_payloads


def _collect_run_completion_evidence_raw(ckpdir, *, gate_schema=None):
    evidence, _ = _collect_run_completion_snapshot_raw(
        ckpdir, gate_schema=gate_schema
    )
    return evidence


def _completion_claims_schema13(ckpdir):
    """Classify schema-13 logger intent without importing telemetry."""

    root = os.path.abspath(ckpdir)
    # Every accepted V20 run directory is the exact lexical stage xpid.  This
    # basename-only dispatch leaves schema<=12 logger/path/exception behavior
    # byte-for-byte inherited and still routes malformed V20 directories to
    # the dedicated fail-closed validator.
    return _schema13_stage_xpid_candidate(os.path.basename(root))


def collect_run_completion_evidence(ckpdir, *, gate_schema=None):
    """Collect exact completion evidence, fully validating schema 13."""

    if gate_schema is None and _completion_claims_schema13(ckpdir):
        bundle = validate_schema13_final_bundle(
            ckpdir, label="schema-13 logger completion bundle"
        )
        return copy.deepcopy(bundle["completion_evidence"])
    return _collect_run_completion_evidence_raw(
        ckpdir, gate_schema=gate_schema
    )


def write_run_completion(
    ckpdir, *, expected_evidence=None, actor_policy_logger_completion=None,
    validated_actor_policy=None, completion_schema_version=1,
    gate_schema=None,
):
    """Atomically bind a success marker to final checkpoints and source."""

    root = os.path.abspath(ckpdir)
    if completion_schema_version == 1 and gate_schema is None:
        normalized_completion_schema = 1
    elif (
        type(completion_schema_version) is int
        and completion_schema_version
        == VOC_ACTOR_POLICY_LOGGER_SCHEMA13_COMPLETION_SCHEMA_VERSION
        and type(gate_schema) is int
        and gate_schema == VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION
    ):
        normalized_completion_schema = (
            VOC_ACTOR_POLICY_LOGGER_SCHEMA13_COMPLETION_SCHEMA_VERSION
        )
    else:
        raise ValueError(
            "run completion schema/gate combination is invalid"
        )
    evidence = (
        collect_run_completion_evidence(root)
        if gate_schema is None
        else collect_run_completion_evidence(root, gate_schema=gate_schema)
    )
    if expected_evidence is not None:
        if (
            not isinstance(expected_evidence, collections.abc.Mapping)
            or dict(expected_evidence) != evidence
        ):
            raise RuntimeError(
                "run completion evidence changed before public commit"
            )
    logger_completion = None
    if actor_policy_logger_completion is not None:
        logger_completion = validate_actor_policy_logger_completion(
            actor_policy_logger_completion
        )
        if logger_completion["schema_version"] != normalized_completion_schema:
            raise ValueError(
                "actor-policy logger schema disagrees at public commit"
            )
        if logger_completion["checkpoint_files"] != evidence["checkpoint_files"]:
            raise RuntimeError(
                "actor-policy logger checkpoint files disagree at public commit"
            )
        if (
            not isinstance(validated_actor_policy, collections.abc.Mapping)
            or validated_actor_policy.get("voc_actor_policy_terminal") is not True
        ):
            raise ValueError(
                "public schema-6 commit requires validated terminal actor evidence"
            )
        for completion_name, actor_name in (
            ("policy_version", "voc_actor_policy_version"),
            ("state_sha256", "voc_actor_policy_state_sha256"),
            (
                "publication_history_sha256",
                "voc_actor_policy_publication_history_sha256",
            ),
        ):
            if logger_completion[completion_name] != validated_actor_policy.get(
                actor_name
            ):
                raise ValueError(
                    "logger completion disagrees with validated actor "
                    f"evidence: {completion_name}"
                )
        for name in (
            VOC_ACTOR_POLICY_LOGGER_FINISH_REQUEST_FILE,
            VOC_ACTOR_POLICY_LOGGER_FINISH_ACK_FILE,
        ):
            if os.path.lexists(_actor_policy_logger_marker_path(root, name)):
                raise RuntimeError(
                    "private actor-policy logger marker remains at public commit"
                )
        with open(
            os.path.join(root, "config_c.yaml"), "r", encoding="utf-8"
        ) as handle:
            saved_config = yaml.safe_load(handle)
        if not isinstance(saved_config, collections.abc.Mapping):
            raise ValueError("config_c.yaml must contain a mapping")
        configured_wandb = saved_config.get("use_wandb")
        if type(configured_wandb) is not bool:
            raise ValueError("schema-6 config use_wandb must be a strict boolean")
        if configured_wandb != logger_completion["use_wandb"]:
            raise ValueError(
                "logger completion disagrees with config_c.yaml use_wandb"
            )

    payload = {
        "schema_version": normalized_completion_schema,
        "status": "complete",
        "completed_unix": time.time(),
        **evidence,
    }
    if logger_completion is not None:
        payload["voc_actor_policy_logger_completion"] = logger_completion
    os.makedirs(root, exist_ok=True)
    marker = os.path.join(root, "finish")
    published_identity = _atomic_write_json(marker, payload, indent=2)
    try:
        marker_status = _require_single_link_regular_file(
            marker, label="public completion marker"
        )
        marker_identity = (
            marker_status.st_dev,
            marker_status.st_ino,
            marker_status.st_ctime_ns,
        )
        if marker_identity != published_identity:
            raise RuntimeError("public completion marker identity changed")
        post_commit_evidence = (
            collect_run_completion_evidence(root)
            if gate_schema is None
            else collect_run_completion_evidence(
                root, gate_schema=gate_schema
            )
        )
        if post_commit_evidence != evidence:
            raise RuntimeError(
                "run completion evidence changed during public commit"
            )
    except Exception:
        _unlink_exact_published_path(marker, published_identity)
        raise
    return payload


def _atomic_write_json(path, payload, *, indent=None):
    """Publish JSON exactly once without overwriting a raced target."""

    root = os.path.dirname(os.path.abspath(path))
    os.makedirs(root, exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}"
    published_identity = None
    try:
        with open(temporary, "x", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                sort_keys=True,
                ensure_ascii=True,
                separators=(",", ":") if indent is None else None,
                indent=indent,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary_stat = os.stat(temporary, follow_symlinks=False)
        os.link(temporary, path)
        published_identity = (temporary_stat.st_dev, temporary_stat.st_ino)
        directory_fd = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.unlink(temporary)
        directory_fd = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        target_status = os.stat(path, follow_symlinks=False)
        if (target_status.st_dev, target_status.st_ino) != published_identity:
            raise RuntimeError("atomic JSON target inode changed after publish")
        published_identity = (
            target_status.st_dev,
            target_status.st_ino,
            target_status.st_ctime_ns,
        )
        return published_identity
    except Exception:
        if published_identity is not None:
            _unlink_exact_published_path(path, published_identity)
        raise
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _unlink_exact_published_path(path, published_identity):
    """Remove only the inode this process linked, never a raced replacement."""

    try:
        current = os.stat(path, follow_symlinks=False)
    except FileNotFoundError:
        return False
    current_identity = (current.st_dev, current.st_ino, current.st_ctime_ns)
    expected_identity = tuple(published_identity)
    if len(expected_identity) not in (2, 3):
        raise ValueError("published path identity must have two or three fields")
    if current_identity[:len(expected_identity)] != expected_identity:
        return False
    os.unlink(path)
    root = os.path.dirname(os.path.abspath(path))
    directory_fd = os.open(root, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return True


def _actor_policy_logger_marker_path(ckpdir, name):
    return os.path.join(os.path.abspath(ckpdir), name)


def _require_single_link_regular_file(path, *, label):
    try:
        status = os.lstat(path)
    except FileNotFoundError:
        raise
    if not stat_module.S_ISREG(status.st_mode) or status.st_nlink != 1:
        raise ValueError(f"{label} must be a single-link regular file")
    return status


def _reject_duplicate_json_pairs(pairs):
    payload = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"JSON object contains duplicate key {key!r}")
        payload[key] = value
    return payload


def clear_actor_policy_logger_completion(
    ckpdir, *, expected_request=None, expected_request_identity=None
):
    """Remove private schema-6 logger request/ack records only."""

    root = os.path.abspath(ckpdir)
    if (expected_request is None) != (expected_request_identity is None):
        raise ValueError(
            "owned logger cleanup requires request and inode identity together"
        )
    if expected_request is not None:
        expected_request = _validate_actor_policy_logger_finish_request(
            expected_request
        )
        request_path = _actor_policy_logger_marker_path(
            root, VOC_ACTOR_POLICY_LOGGER_FINISH_REQUEST_FILE
        )
        request_status = _require_single_link_regular_file(
            request_path, label="owned actor-policy logger finish request"
        )
        request_identity = (
            request_status.st_dev,
            request_status.st_ino,
            request_status.st_ctime_ns,
        )
        if request_identity != tuple(expected_request_identity):
            raise RuntimeError(
                "actor-policy logger finish request ownership changed"
            )
        if read_actor_policy_logger_finish_request(root) != expected_request:
            raise RuntimeError(
                "actor-policy logger finish request payload changed"
            )
        owned_paths = [(request_path, request_identity)]
        ack_path = _actor_policy_logger_marker_path(
            root, VOC_ACTOR_POLICY_LOGGER_FINISH_ACK_FILE
        )
        if os.path.lexists(ack_path):
            ack_status = _require_single_link_regular_file(
                ack_path, label="owned actor-policy logger finish ack"
            )
            if read_actor_policy_logger_finish_ack(
                root, expected_request
            ) is None:
                raise RuntimeError("actor-policy logger finish ack vanished")
            owned_paths.insert(
                0,
                (
                    ack_path,
                    (
                        ack_status.st_dev,
                        ack_status.st_ino,
                        ack_status.st_ctime_ns,
                    ),
                ),
            )
        for marker, identity in owned_paths:
            if not _unlink_exact_published_path(marker, identity):
                raise RuntimeError(
                    "actor-policy logger marker changed during owned cleanup"
                )
        for marker, _ in owned_paths:
            if os.path.lexists(marker):
                raise RuntimeError(
                    "actor-policy logger marker cleanup was not durable"
                )
        return

    removed = False
    for name in (
        VOC_ACTOR_POLICY_LOGGER_FINISH_REQUEST_FILE,
        VOC_ACTOR_POLICY_LOGGER_FINISH_ACK_FILE,
    ):
        marker = _actor_policy_logger_marker_path(ckpdir, name)
        if os.path.lexists(marker):
            _require_single_link_regular_file(
                marker, label="actor-policy logger marker"
            )
            os.unlink(marker)
            removed = True
    if removed:
        directory_fd = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    for name in (
        VOC_ACTOR_POLICY_LOGGER_FINISH_REQUEST_FILE,
        VOC_ACTOR_POLICY_LOGGER_FINISH_ACK_FILE,
    ):
        if os.path.lexists(_actor_policy_logger_marker_path(root, name)):
            raise RuntimeError("actor-policy logger marker cleanup was not durable")


def validate_schema6_fresh_run_directory(ckpdir):
    """Reject xpid reuse after create_setting wrote the sole config file."""

    root = os.path.abspath(ckpdir)
    if not os.path.isdir(root) or os.path.islink(root):
        raise ValueError("schema-6 fresh run directory must be a real directory")
    entries = os.listdir(root)
    if entries != ["config_c.yaml"] and set(entries) != {"config_c.yaml"}:
        raise FileExistsError(
            "schema-6 fresh run directory contains pre-existing artifacts: "
            f"{sorted(entries)}"
        )
    config_path = os.path.join(root, "config_c.yaml")
    if not os.path.isfile(config_path) or os.path.islink(config_path):
        raise ValueError("schema-6 config_c.yaml must be a regular file")
    return True


def create_schema6_fresh_run_directory(ckpdir):
    """Create a new schema-6 xpid directory with no reuse or replacement."""

    root = os.path.abspath(ckpdir)
    os.makedirs(os.path.dirname(root), exist_ok=True)
    try:
        os.mkdir(root)
    except FileExistsError as error:
        raise FileExistsError(
            f"schema-6 fresh xpid directory already exists: {root}"
        ) from error
    return root


def _validate_completion_checkpoint_files(
    checkpoint_files, *, label, schema_version=1
):
    if schema_version == 1:
        required_names = _COMPLETION_CHECKPOINT_FILES
        strict_schema13_types = False
    elif (
        type(schema_version) is int
        and schema_version
        == VOC_ACTOR_POLICY_LOGGER_SCHEMA13_COMPLETION_SCHEMA_VERSION
    ):
        required_names = _SCHEMA13_COMPLETION_CHECKPOINT_FILES
        strict_schema13_types = True
    else:
        raise ValueError(f"{label} has unsupported completion schema")
    required = set(required_names)
    if (
        not isinstance(checkpoint_files, collections.abc.Mapping)
        or set(checkpoint_files) != required
    ):
        raise ValueError(f"{label} must contain exactly {sorted(required)}")
    canonical = {}
    for name in required_names:
        record = checkpoint_files[name]
        if (
            not isinstance(record, collections.abc.Mapping)
            or set(record) != {"sha256", "size"}
        ):
            raise ValueError(f"{label} {name} record is malformed")
        digest = record["sha256"]
        size = record["size"]
        if (
            (type(digest) is not str if strict_schema13_types else not isinstance(digest, str))
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise ValueError(f"{label} {name} sha256 is invalid")
        if strict_schema13_types:
            size_invalid = type(size) is not int or size <= 0
        else:
            size_invalid = (
                isinstance(size, (bool, np.bool_))
                or not isinstance(size, (int, np.integer))
                or int(size) <= 0
            )
        if size_invalid:
            raise ValueError(f"{label} {name} size is invalid")
        canonical[name] = {"sha256": digest, "size": int(size)}
    return canonical


def _validate_actor_policy_logger_finish_request(payload):
    required = {
        "schema_version",
        "status",
        "policy_version",
        "state_sha256",
        "publication_history_sha256",
        "checkpoint_files",
    }
    if not isinstance(payload, collections.abc.Mapping) or set(payload) != required:
        raise ValueError(
            "actor-policy logger finish request has invalid fields"
        )
    schema = payload["schema_version"]
    if (
        isinstance(schema, (bool, np.bool_))
        or not isinstance(schema, (int, np.integer))
    ):
        raise ValueError("actor-policy logger finish request has invalid schema")
    if int(schema) == VOC_ACTOR_POLICY_LOGGER_COMPLETION_SCHEMA_VERSION:
        normalized_schema = VOC_ACTOR_POLICY_LOGGER_COMPLETION_SCHEMA_VERSION
    elif (
        type(schema) is int
        and schema
        == VOC_ACTOR_POLICY_LOGGER_SCHEMA13_COMPLETION_SCHEMA_VERSION
    ):
        normalized_schema = (
            VOC_ACTOR_POLICY_LOGGER_SCHEMA13_COMPLETION_SCHEMA_VERSION
        )
    else:
        raise ValueError("actor-policy logger finish request has invalid schema")
    version = payload["policy_version"]
    if (
        payload["status"] != "finish_requested"
        or (
            normalized_schema
            == VOC_ACTOR_POLICY_LOGGER_SCHEMA13_COMPLETION_SCHEMA_VERSION
            and type(payload["status"]) is not str
        )
    ):
        raise ValueError("actor-policy logger finish request has invalid status")
    if normalized_schema == VOC_ACTOR_POLICY_LOGGER_SCHEMA13_COMPLETION_SCHEMA_VERSION:
        version_invalid = type(version) is not int or version < 0
    else:
        version_invalid = (
            isinstance(version, (bool, np.bool_))
            or not isinstance(version, (int, np.integer))
            or int(version) < 0
        )
    if version_invalid:
        raise ValueError("actor-policy logger finish request has invalid version")
    for name in ("state_sha256", "publication_history_sha256"):
        value = payload[name]
        if (
            (
                type(value) is not str
                if normalized_schema
                == VOC_ACTOR_POLICY_LOGGER_SCHEMA13_COMPLETION_SCHEMA_VERSION
                else not isinstance(value, str)
            )
            or re.fullmatch(r"[0-9a-f]{64}", value) is None
        ):
            raise ValueError(
                f"actor-policy logger finish request has invalid {name}"
            )
    checkpoint_files = _validate_completion_checkpoint_files(
        payload["checkpoint_files"],
        label="actor-policy logger finish request checkpoint_files",
        schema_version=normalized_schema,
    )
    return {
        "schema_version": normalized_schema,
        "status": "finish_requested",
        "policy_version": int(version),
        "state_sha256": payload["state_sha256"],
        "publication_history_sha256": payload[
            "publication_history_sha256"
        ],
        "checkpoint_files": checkpoint_files,
    }


def actor_policy_logger_finish_request_sha256(payload):
    canonical = _validate_actor_policy_logger_finish_request(payload)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_actor_policy_logger_finish_request(
    ckpdir,
    checkpoint_evidence,
    completion_evidence,
    *,
    return_identity=False,
    schema_version=1,
):
    """Publish the private schema-6 logger close request atomically."""

    if not isinstance(checkpoint_evidence, collections.abc.Mapping):
        raise ValueError("validated actor-policy evidence must be a mapping")
    if checkpoint_evidence.get("voc_actor_policy_terminal") is not True:
        raise ValueError("logger finish requires a terminal actor-policy bundle")
    if not isinstance(completion_evidence, collections.abc.Mapping):
        raise ValueError("completion evidence must be a mapping")
    payload = _validate_actor_policy_logger_finish_request({
        "schema_version": schema_version,
        "status": "finish_requested",
        "policy_version": checkpoint_evidence.get("voc_actor_policy_version"),
        "state_sha256": checkpoint_evidence.get(
            "voc_actor_policy_state_sha256"
        ),
        "publication_history_sha256": checkpoint_evidence.get(
            "voc_actor_policy_publication_history_sha256"
        ),
        "checkpoint_files": completion_evidence.get("checkpoint_files"),
    })
    path = _actor_policy_logger_marker_path(
        ckpdir, VOC_ACTOR_POLICY_LOGGER_FINISH_REQUEST_FILE
    )
    if os.path.lexists(path):
        raise FileExistsError(f"actor-policy logger finish request exists: {path}")
    published_identity = _atomic_write_json(path, payload)
    if return_identity:
        return payload, published_identity
    return payload


def read_actor_policy_logger_finish_request(ckpdir):
    path = _actor_policy_logger_marker_path(
        ckpdir, VOC_ACTOR_POLICY_LOGGER_FINISH_REQUEST_FILE
    )
    if not os.path.exists(path):
        return None
    _require_single_link_regular_file(
        path, label="actor-policy logger finish request"
    )
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle, object_pairs_hook=_reject_duplicate_json_pairs)
    return _validate_actor_policy_logger_finish_request(payload)


def _expected_actor_policy_logger_finish_ack(request):
    request = _validate_actor_policy_logger_finish_request(request)
    return {
        "schema_version": request["schema_version"],
        "status": "finish_acknowledged",
        "request_sha256": actor_policy_logger_finish_request_sha256(request),
    }


def write_actor_policy_logger_finish_ack(ckpdir, request):
    """Acknowledge final upload/close for the exact private request."""

    payload = _expected_actor_policy_logger_finish_ack(request)
    path = _actor_policy_logger_marker_path(
        ckpdir, VOC_ACTOR_POLICY_LOGGER_FINISH_ACK_FILE
    )
    if os.path.lexists(path):
        raise FileExistsError(f"actor-policy logger finish ack exists: {path}")
    _atomic_write_json(path, payload)
    return payload


def read_actor_policy_logger_finish_ack(ckpdir, request):
    path = _actor_policy_logger_marker_path(
        ckpdir, VOC_ACTOR_POLICY_LOGGER_FINISH_ACK_FILE
    )
    if not os.path.exists(path):
        return None
    _require_single_link_regular_file(
        path, label="actor-policy logger finish ack"
    )
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle, object_pairs_hook=_reject_duplicate_json_pairs)
    expected = _expected_actor_policy_logger_finish_ack(request)
    if not isinstance(payload, collections.abc.Mapping) or dict(payload) != expected:
        raise ValueError("actor-policy logger finish ack disagrees with request")
    return expected


def validate_actor_policy_logger_completion(completion):
    required_keys = {
        "schema_version",
        "required",
        "use_wandb",
        "request_sha256",
        "ack_verified",
        "private_markers_cleaned",
        "policy_version",
        "state_sha256",
        "publication_history_sha256",
        "checkpoint_files",
    }
    if not isinstance(completion, collections.abc.Mapping) or set(completion) != required_keys:
        raise ValueError("actor-policy logger completion has invalid fields")
    schema = completion["schema_version"]
    if (
        isinstance(schema, (bool, np.bool_))
        or not isinstance(schema, (int, np.integer))
    ):
        raise ValueError("actor-policy logger completion has invalid schema")
    if int(schema) == VOC_ACTOR_POLICY_LOGGER_COMPLETION_SCHEMA_VERSION:
        normalized_schema = VOC_ACTOR_POLICY_LOGGER_COMPLETION_SCHEMA_VERSION
    elif (
        type(schema) is int
        and schema
        == VOC_ACTOR_POLICY_LOGGER_SCHEMA13_COMPLETION_SCHEMA_VERSION
    ):
        normalized_schema = (
            VOC_ACTOR_POLICY_LOGGER_SCHEMA13_COMPLETION_SCHEMA_VERSION
        )
    else:
        raise ValueError("actor-policy logger completion has invalid schema")
    required = completion["required"]
    use_wandb = completion["use_wandb"]
    ack_verified = completion["ack_verified"]
    cleaned = completion["private_markers_cleaned"]
    if any(
        type(value) is not bool
        for value in (required, use_wandb, ack_verified, cleaned)
    ):
        raise ValueError("actor-policy logger completion booleans are invalid")
    if required != use_wandb:
        raise ValueError("actor-policy logger required must equal use_wandb")
    if cleaned is not True:
        raise ValueError("actor-policy logger private markers are not cleaned")
    request_digest = completion["request_sha256"]
    if required:
        if (
            (
                type(request_digest) is not str
                if normalized_schema
                == VOC_ACTOR_POLICY_LOGGER_SCHEMA13_COMPLETION_SCHEMA_VERSION
                else not isinstance(request_digest, str)
            )
            or re.fullmatch(r"[0-9a-f]{64}", request_digest) is None
            or ack_verified is not True
        ):
            raise ValueError("required actor-policy logger ack is invalid")
    elif request_digest is not None or ack_verified is not False:
        raise ValueError("disabled actor-policy logger completion is invalid")
    version = completion["policy_version"]
    if normalized_schema == VOC_ACTOR_POLICY_LOGGER_SCHEMA13_COMPLETION_SCHEMA_VERSION:
        version_invalid = type(version) is not int or version < 1
    else:
        version_invalid = (
            isinstance(version, (bool, np.bool_))
            or not isinstance(version, (int, np.integer))
            or int(version) < 1
        )
    if version_invalid:
        raise ValueError("actor-policy logger completion version is invalid")
    for name in ("state_sha256", "publication_history_sha256"):
        value = completion[name]
        if (
            (
                type(value) is not str
                if normalized_schema
                == VOC_ACTOR_POLICY_LOGGER_SCHEMA13_COMPLETION_SCHEMA_VERSION
                else not isinstance(value, str)
            )
            or re.fullmatch(r"[0-9a-f]{64}", value) is None
        ):
            raise ValueError(f"actor-policy logger completion {name} is invalid")
    return {
        "schema_version": normalized_schema,
        "required": required,
        "use_wandb": use_wandb,
        "request_sha256": request_digest,
        "ack_verified": ack_verified,
        "private_markers_cleaned": True,
        "policy_version": int(version),
        "state_sha256": completion["state_sha256"],
        "publication_history_sha256": completion[
            "publication_history_sha256"
        ],
        "checkpoint_files": _validate_completion_checkpoint_files(
            completion["checkpoint_files"],
            label="actor-policy logger completion checkpoint_files",
            schema_version=normalized_schema,
        ),
    }

# Dynamic Thinker public action/phase contracts.  Keep these values in sync
# with cenv.pyx; tests assert the mapping at the Python/Cython boundary.
PROCEED = 0
RESET = 1
STOP = 2

SEARCH_PHASE = 0
NEED_REAL_ACTION_PHASE = 1
WAIT_PHASE = 2

POLICY_NONE = 0
POLICY_SEARCH = 1
POLICY_REAL = 2

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
    # Schema-6 separates the soft learning epsilon from the actual
    # sign-policy execution epsilon.  Legacy schemas continue to execute with
    # voc_train_epsilon and therefore canonicalize this field to 0.02.
    "voc_gate_execution_epsilon": 0.02,
    "voc_actor_policy_version_barrier": False,
    # Schema 7 seals the model-input stream after the terminal actor-policy
    # acknowledgement.  Older schemas resolve the absent field to zero.
    "voc_model_input_seal_schema_version": 0,
    "voc_actor_policy_bundle_schema_version": 1,
    "voc_actor_policy_barrier_timeout_s": 120.0,
    "voc_actor_policy_ray_max_restarts": 0,
    "voc_actor_policy_ray_max_task_retries": 0,
    "actor_amp_init_scale": 256.0,
    "voc_gate_learning_rate": 0.0003,
    "voc_gate_grad_norm_clipping": 1.0,
    # Active VoC targets raw environment return.  A non-zero actor entropy
    # reward would make the shared value baseline optimize a different return.
    "entropy_r_cost": 0.0,
}
VOC_MODES = frozenset(("off", "shadow", "control"))
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
        "voc_model_input_seal_schema_version",
        "voc_actor_policy_bundle_schema_version",
        "voc_actor_policy_barrier_timeout_s",
        "voc_actor_policy_ray_max_restarts",
        "voc_actor_policy_ray_max_task_retries",
        "actor_amp_init_scale",
        "voc_gate_learning_rate",
        "voc_gate_grad_norm_clipping",
    )
)
VOC_CONTROL_ORIGIN_FRESH = "fresh"
VOC_CONTROL_ORIGIN_SHADOW_PARENT = "shadow_parent"
VOC_CONTROL_ORIGINS = frozenset(
    (VOC_CONTROL_ORIGIN_FRESH, VOC_CONTROL_ORIGIN_SHADOW_PARENT)
)
VOC_HOLDOUT_SPLIT_VERSION = 1
VOC_HOLDOUT_ACTOR_MODULUS = 8
VOC_EMA_GATE_SCHEMA_VERSION = 1
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
VOC_MODEL_INPUT_SEAL_SCHEMA_VERSION = 1
VOC_GATE_POLICY_ATOMIC_SCHEMA_VERSIONS = frozenset((
    VOC_GATE_POLICY_VERSION_BARRIER_SCHEMA_VERSION,
    VOC_GATE_POLICY_MODEL_INPUT_SEAL_SCHEMA_VERSION,
    VOC_GATE_POLICY_HALF_SQUARED_Q_SCHEMA_VERSION,
    VOC_GATE_POLICY_COMMON_MODE_Q_SCHEMA_VERSION,
    VOC_GATE_POLICY_HUBER_COMMON_Q_SCHEMA_VERSION,
    VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION,
    VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION,
    VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION,
))
VOC_ACTOR_POLICY_BUNDLE_SCHEMA_VERSION = 1
VOC_ACTOR_POLICY_BARRIER_TIMEOUT_SECONDS = 120.0
# Schema 6 is an atomic protocol label, not a menu of independently tunable
# VoC settings.  Keep this table separate from the legacy defaults above:
# schemas 1--5 retain their historical defaults, while a schema-6 record must
# carry every value below explicitly and exactly.
VOC_GATE_POLICY_SCHEMA6_ATOMIC_REQUIREMENTS = {
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
    "voc_gate_confidence_weighted": False,
    "voc_gate_adam_beta1": 0.0,
    "voc_gate_param_align": False,
    "voc_gate_param_align_coef": 1.0,
    "voc_gate_exact_projection": True,
    "voc_gate_epsilon_greedy_execution": True,
    "voc_gate_execution_epsilon": 0.25,
    "voc_gate_learning_rate": 0.001,
    "voc_gate_grad_norm_clipping": 1.0,
    "entropy_r_cost": 0.0,
    "model_state_range_loss_cost": 1.0,
}
VOC_GATE_POLICY_SCHEMA6_OPTIMIZER_REQUIREMENTS = {
    "actor_use_rms": False,
    "actor_learning_rate": 0.0003,
    "actor_adam_eps": 1e-8,
    "model_optimizer": "adam",
    "model_learning_rate": 5e-5,
}
VOC_GATE_POLICY_SCHEMA6_ENDURO_REQUIREMENTS = {
    "name": "Enduro-v5",
    "icopro_game_id": 0,
    "icopro_supervised_freq": 1,
    "envpool": True,
    "frame_stack_n": 4,
    "grayscale": False,
    "wrapper_type": 0,
    "dynamic_search": True,
    "dynamic_factorized_control": True,
    "max_search_steps": 20,
    "max_depth": 20,
    "rec_t": 20,
    "has_action_seq": False,
    "return_h": True,
    "return_x": True,
    "model_size_nn": 2,
    "model_disable_bn": False,
    "model_decoder_depth": 0,
    "model_state_projection": "clamp",
}
VOC_GATE_POLICY_SCHEMA6_V12_BASELINE_SHA256 = (
    "bd386ad1c78f030409562dea5c34d8866cc128cf99b7f39938f6a090f5987407"
)
# Frozen canonical JSON (UTF-8, no trailing LF) of the exact 209-key
# production-v12 projection preregistered for schema 6.  This is deliberately
# source-resident: validation must not depend on a mutable or historical run
# directory.
_VOC_GATE_POLICY_SCHEMA6_V12_BASELINE_JSON = r'''{"__version__":"1.3.0","action_prior_ema":0.05,"action_prior_weight":1.0,"actor_adam_eps":1e-08,"actor_amp_max_consecutive_skips":8,"actor_batch_size":16,"actor_grad_norm_clipping":0.5,"actor_learning_rate":0.0003,"actor_max_std":10,"actor_min_std":0.003,"actor_ordinal":false,"actor_use_rms":false,"auto_res":false,"autotune":false,"baseline_cost":0.5,"batch_length":4,"buffer_save_size":1,"checkpoint_interval":250000,"ckp":false,"critic_enc_f_type":0,"critic_enc_type":0,"critic_zero_init":true,"cur_cost":0.0,"cur_cost_anneal":true,"detect_dan_num":0,"discounting":0.99,"discrete_k":-1,"drc":false,"dual_net":true,"dynamic_factorized_control":true,"dynamic_search":true,"dynamic_search_hidden_dim":100,"dynamic_voc_mode":"control","enc_1d_block":2,"enc_1d_hs":256,"enc_1d_norm":true,"enc_1d_shallow":false,"entropy_cost":0.001,"entropy_r_cost":0.0,"env_n":16,"envpool":true,"fea_loss_inf_bn":true,"float16":true,"frame_stack_n":4,"git_revision":null,"gpu_learn":1.0,"gpu_learn_actor":0.5,"gpu_self_play":0.5,"grayscale":false,"h_rnn":false,"has_action_seq":false,"has_model":true,"icopro_action_diff_coef":1.0,"icopro_batch_size":16,"icopro_coef":1.0,"icopro_device":"cuda","icopro_game_id":0,"icopro_holdout_sessions":"4","icopro_margin":1.0,"icopro_margin_coef":1.0,"icopro_pvp_coef":0.0,"icopro_subjects":"1","icopro_supervised_freq":1,"icopro_train_sessions":"1,2,3","im_cost":1,"im_cost_anneal":true,"im_enable":true,"im_entropy_cost":0.0005,"img_fea_cos":true,"last_layer_n":0,"legacy":false,"max_depth":20,"max_replay_ratio":5.0,"max_search_steps":20,"mcts":false,"min_replay_ratio":4.0,"model_batch_size":32,"model_buffer_n":200000,"model_decoder_depth":0,"model_disable_bn":false,"model_done_loss_cost":1.0,"model_downscale_c":2,"model_downscale_c_vp":2,"model_enc_f_type":0,"model_enc_type":0,"model_fea_loss_cost":10.0,"model_float16":false,"model_grad_norm_clipping":10000.0,"model_has_memory":false,"model_img_loss_cost":0.0,"model_learning_rate":5e-05,"model_mem_unroll_len":0,"model_noise_loss_cost":0.0,"model_optimizer":"adam","model_ordinal":false,"model_policy_loss_cost":0.5,"model_reg_loss_cost":0.0,"model_return_n":5,"model_rs_loss_cost":1.0,"model_sgd_momentum":0.9,"model_sgd_weight_decay":0.0001,"model_size_nn":2,"model_state_projection":"clamp","model_state_range_loss_cost":1.0,"model_unroll_len":20,"model_vs_loss_cost":0.25,"model_zero_init":true,"name":"Enduro-v5","noise_alpha":0.8,"noise_d":10,"noise_enable":false,"noise_mlp":false,"noise_n":16,"obs_clip":-1,"obs_norm":false,"parallel":true,"parallel_actor":true,"policy_vis_freq":-1,"policy_vis_length":20,"ppo_clip":0.3,"ppo_early_stop":false,"ppo_k":1,"ppo_kl_coef":0.0,"ppo_kl_targ":0.04,"ppo_n":64,"ppo_syn":false,"ppo_v_trace":true,"preload":"","preload_actor":"","priority_alpha":0.6,"priority_beta":0.4,"profile":false,"project":"thinker","ray_cpu":16,"ray_gpu":2,"ray_mem":-1,"real_state_ch":-1,"real_state_rnn":false,"rec_t":20,"reg_cost":0.001,"require_prob":true,"reset_mode":0,"return_h":true,"return_norm_type":-1,"return_x":true,"reward_clip":1.0,"reward_norm":false,"sample_n":-1,"sample_replace":true,"sample_temp":4,"schedule_total_steps":100000000,"se_buffer_n":20,"se_lstm_table":true,"se_manual_stat":false,"se_query_cur":2,"se_query_size":20,"se_td_lambda":0.9,"se_tree_carry":true,"see_h":true,"see_real_state":true,"see_tree_rep":true,"see_x":true,"self_play_n":1,"sep_actor_critic":false,"sep_im_head":true,"stat_mask_type":0,"tanh_action":true,"tar_entropy_scale":0.4,"tar_im_entropy_scale":0.4,"test_rec_t":-1,"think_cost":0.0005,"think_cost_anneal":false,"train_actor":true,"train_model":true,"tran_attn_b":5,"tran_dim":128,"tran_head_n":8,"tran_layer_n":3,"tran_lstm_no_attn":false,"tran_mem_n":40,"tran_reset_mode":0,"tran_t":1,"tree_carry":true,"tree_rep_rnn":false,"v_trace_lamb":1.0,"voc_dedicated_gate":true,"voc_dueling_q":true,"voc_ema_gate_target":true,"voc_eval_stochastic":true,"voc_expected_gate_loss":true,"voc_gate_adam_beta1":0.0,"voc_gate_confidence_weighted":false,"voc_gate_epsilon_greedy_execution":true,"voc_gate_exact_projection":true,"voc_gate_grad_norm_clipping":1.0,"voc_gate_learning_rate":0.001,"voc_gate_param_align":false,"voc_gate_param_align_coef":1.0,"voc_gate_q_temperature":0.05,"voc_gate_target_tau":0.1,"voc_gate_temperature":1.0,"voc_loss_cost":1.0,"voc_parent_checkpoint":"","voc_soft_q_bce_gate":true,"voc_train_epsilon":0.02,"vp_fix_bootstrap":false,"wandb_ckp_freq":0,"wrapper_type":0,"x_rnn":false}'''
VOC_GATE_POLICY_SCHEMA6_V12_BASELINE = MappingProxyType(
    json.loads(_VOC_GATE_POLICY_SCHEMA6_V12_BASELINE_JSON)
)
VOC_GATE_POLICY_SCHEMA12_V12_PROJECTION_SHA256 = (
    "ad22b91fdd06a30ac7f53c0135b32fac2530687c3c36dad5dccf06d700f83f82"
)
VOC_GATE_POLICY_SCHEMA12_V12_PROJECTION = MappingProxyType({
    **dict(VOC_GATE_POLICY_SCHEMA6_V12_BASELINE),
    "voc_gate_target_tau": 1.0,
})
VOC_GATE_POLICY_SCHEMA6_STAGE_FIELDS = frozenset({
    "xpid",
    "base_seed",
    "total_steps",
    "model_warm_up_n",
    "actor_unroll_len",
    "use_wandb",
})
VOC_GATE_POLICY_SCHEMA6_PATH_FIELDS = frozenset({
    "savedir",
    "ckpdir",
    "cmd",
    "icopro_data_path",
})
VOC_GATE_POLICY_SCHEMA6_NEW_FIELDS = frozenset({
    "voc_gate_policy_schema_version",
    "voc_gate_execution_epsilon",
    "voc_actor_policy_version_barrier",
    "voc_actor_policy_bundle_schema_version",
    "voc_actor_policy_barrier_timeout_s",
    "voc_actor_policy_ray_max_restarts",
    "voc_actor_policy_ray_max_task_retries",
    "actor_amp_init_scale",
    "voc_actor_policy_barrier_runtime",
})
VOC_GATE_POLICY_SCHEMA6_STAGE_PROFILES = (
    (
        "enduro-voc-v13-versioned-eps25-sentinel-wire1200",
        1,
        1200,
        512,
        41,
        False,
    ),
    (
        "enduro-voc-v13-versioned-eps25-seed1-qual-fresh-100k",
        1,
        100000,
        10000,
        201,
        True,
    ),
    (
        "enduro-voc-v13-versioned-eps25-seed5-strict-fresh-300k",
        5,
        300000,
        10000,
        201,
        True,
    ),
)
VOC_GATE_POLICY_SCHEMA7_NEW_FIELDS = frozenset(
    set(VOC_GATE_POLICY_SCHEMA6_NEW_FIELDS)
    | {"voc_model_input_seal_schema_version"}
)
VOC_GATE_POLICY_SCHEMA7_STAGE_PROFILES = (
    (
        "enduro-voc-v14-sealed-eps25-sentinel-wire1200",
        1,
        1200,
        512,
        41,
        False,
    ),
    (
        "enduro-voc-v14-sealed-eps25-seed1-qual-fresh-100k",
        1,
        100000,
        10000,
        201,
        True,
    ),
    (
        "enduro-voc-v14-sealed-eps25-seed5-strict-fresh-300k",
        5,
        300000,
        10000,
        201,
        True,
    ),
)
VOC_GATE_POLICY_SCHEMA8_NEW_FIELDS = frozenset(
    VOC_GATE_POLICY_SCHEMA7_NEW_FIELDS
)
VOC_GATE_POLICY_SCHEMA8_STAGE_PROFILES = (
    (
        "enduro-voc-v15-halfsq-eps25-sentinel-wire1200",
        1,
        1200,
        512,
        41,
        False,
    ),
    (
        "enduro-voc-v15-halfsq-eps25-seed1-qual-fresh-100k",
        1,
        100000,
        10000,
        201,
        True,
    ),
    (
        "enduro-voc-v15-halfsq-eps25-seed5-strict-fresh-300k",
        5,
        300000,
        10000,
        201,
        True,
    ),
)
VOC_GATE_POLICY_SCHEMA9_NEW_FIELDS = frozenset(
    VOC_GATE_POLICY_SCHEMA8_NEW_FIELDS
)
VOC_GATE_POLICY_SCHEMA9_STAGE_PROFILES = (
    (
        "enduro-voc-v16-commonmode-eps25-sentinel-wire1200",
        1,
        1200,
        512,
        41,
        False,
    ),
    (
        "enduro-voc-v16-commonmode-eps25-seed1-qual-fresh-100k",
        1,
        100000,
        10000,
        201,
        True,
    ),
    (
        "enduro-voc-v16-commonmode-eps25-seed5-strict-fresh-300k",
        5,
        300000,
        10000,
        201,
        True,
    ),
)
VOC_GATE_POLICY_SCHEMA10_NEW_FIELDS = frozenset(
    VOC_GATE_POLICY_SCHEMA9_NEW_FIELDS
)
VOC_GATE_POLICY_SCHEMA10_STAGE_PROFILES = (
    (
        "enduro-voc-v17-huber-common-eps25-sentinel-wire1200",
        1,
        1200,
        512,
        41,
        False,
    ),
    (
        "enduro-voc-v17-huber-common-eps25-seed1-qual-fresh-100k",
        1,
        100000,
        10000,
        201,
        True,
    ),
    (
        "enduro-voc-v17-huber-common-eps25-seed5-strict-fresh-300k",
        5,
        300000,
        10000,
        201,
        True,
    ),
)
VOC_GATE_POLICY_SCHEMA11_NEW_FIELDS = frozenset(
    VOC_GATE_POLICY_SCHEMA10_NEW_FIELDS
)
VOC_GATE_POLICY_SCHEMA11_STAGE_PROFILES = (
    (
        "enduro-voc-v18-orthocd-adam-eps25-sentinel-wire1200",
        1,
        1200,
        512,
        41,
        False,
    ),
    (
        "enduro-voc-v18-orthocd-adam-eps25-seed1-qual-fresh-100k",
        1,
        100000,
        10000,
        201,
        True,
    ),
    (
        "enduro-voc-v18-orthocd-adam-eps25-seed5-strict-fresh-300k",
        5,
        300000,
        10000,
        201,
        True,
    ),
)
VOC_GATE_POLICY_SCHEMA12_NEW_FIELDS = frozenset(
    VOC_GATE_POLICY_SCHEMA11_NEW_FIELDS
)
VOC_GATE_POLICY_SCHEMA12_STAGE_PROFILES = (
    (
        "enduro-voc-v19-tau1-orthocd-adam-eps25-sentinel-wire1200",
        1,
        1200,
        512,
        41,
        False,
    ),
    (
        "enduro-voc-v19-tau1-orthocd-adam-eps25-seed1-qual-fresh-100k",
        1,
        100000,
        10000,
        201,
        True,
    ),
    (
        "enduro-voc-v19-tau1-orthocd-adam-eps25-seed5-strict-fresh-300k",
        5,
        300000,
        10000,
        201,
        True,
    ),
)
VOC_GATE_POLICY_SCHEMA13_NEW_FIELDS = frozenset(
    VOC_GATE_POLICY_SCHEMA12_NEW_FIELDS
)
VOC_GATE_POLICY_SCHEMA13_STAGE_PROFILES = (
    (
        "enduro-voc-v20-telemetry-tau1-orthocd-adam-eps25-"
        "sentinel-wire1200",
        1,
        1200,
        512,
        41,
        False,
    ),
    (
        "enduro-voc-v20-telemetry-tau1-orthocd-adam-eps25-"
        "seed1-qual-fresh-100k",
        1,
        100000,
        10000,
        201,
        True,
    ),
    (
        "enduro-voc-v20-telemetry-tau1-orthocd-adam-eps25-"
        "seed5-strict-fresh-300k",
        5,
        300000,
        10000,
        201,
        True,
    ),
)

# Schema 13 intentionally has no schema-version CLI flag.  Its three public
# stages inherit one exact 96-pair vector and infer the discriminator from the
# lexical xpid.  Keep this ordered flag surface independent of argparse so a
# duplicate, reordered, missing, or 97th pair cannot be silently normalized.
_SCHEMA13_CLI_FLAG_ORDER = (
    "--name",
    "--xpid",
    "--savedir",
    "--ckp",
    "--preload",
    "--preload_actor",
    "--voc_parent_checkpoint",
    "--total_steps",
    "--schedule_total_steps",
    "--model_warm_up_n",
    "--actor_unroll_len",
    "--dynamic_search",
    "--dynamic_factorized_control",
    "--dynamic_voc_mode",
    "--voc_loss_cost",
    "--voc_gate_temperature",
    "--voc_train_epsilon",
    "--voc_eval_stochastic",
    "--voc_dueling_q",
    "--voc_expected_gate_loss",
    "--voc_ema_gate_target",
    "--voc_gate_target_tau",
    "--voc_dedicated_gate",
    "--voc_soft_q_bce_gate",
    "--voc_gate_q_temperature",
    "--voc_gate_confidence_weighted",
    "--voc_gate_adam_beta1",
    "--voc_gate_param_align",
    "--voc_gate_param_align_coef",
    "--voc_gate_exact_projection",
    "--voc_gate_epsilon_greedy_execution",
    "--voc_gate_execution_epsilon",
    "--voc_actor_policy_version_barrier",
    "--voc_actor_policy_bundle_schema_version",
    "--voc_actor_policy_barrier_timeout_s",
    "--voc_actor_policy_ray_max_restarts",
    "--voc_actor_policy_ray_max_task_retries",
    "--actor_amp_init_scale",
    "--voc_gate_learning_rate",
    "--voc_gate_grad_norm_clipping",
    "--entropy_r_cost",
    "--wrapper_type",
    "--rec_t",
    "--max_search_steps",
    "--max_depth",
    "--model_unroll_len",
    "--think_cost",
    "--think_cost_anneal",
    "--tree_carry",
    "--train_model",
    "--float16",
    "--actor_amp_max_consecutive_skips",
    "--model_float16",
    "--model_learning_rate",
    "--model_grad_norm_clipping",
    "--model_disable_bn",
    "--model_state_projection",
    "--model_state_range_loss_cost",
    "--model_batch_size",
    "--actor_batch_size",
    "--env_n",
    "--self_play_n",
    "--parallel_actor",
    "--ppo_k",
    "--icopro_data_path",
    "--icopro_subjects",
    "--icopro_game_id",
    "--icopro_train_sessions",
    "--icopro_holdout_sessions",
    "--icopro_batch_size",
    "--batch_length",
    "--icopro_margin",
    "--icopro_margin_coef",
    "--icopro_action_diff_coef",
    "--icopro_pvp_coef",
    "--icopro_coef",
    "--icopro_supervised_freq",
    "--action_prior_weight",
    "--action_prior_ema",
    "--icopro_device",
    "--reward_clip",
    "--model_size_nn",
    "--discounting",
    "--envpool",
    "--grayscale",
    "--frame_stack_n",
    "--auto_res",
    "--ray_cpu",
    "--ray_gpu",
    "--gpu_learn",
    "--gpu_learn_actor",
    "--gpu_self_play",
    "--use_wandb",
    "--base_seed",
    "--actor_use_rms",
    "--voc_model_input_seal_schema_version",
)
_SCHEMA13_CLI_WIRE_VALUE_TEXT = (
    "Enduro-v5",
    VOC_GATE_POLICY_SCHEMA13_STAGE_PROFILES[0][0],
    None,  # canonical absolute candidate_root/runs
    "False", "", "", "", "1200", "100000000", "512", "41",
    "True", "True", "control", "1.0", "1.0", "0.02", "True",
    "True", "True", "True", "1.0", "True", "True", "0.05",
    "False", "0.0", "False", "1.0", "True", "True", "0.25",
    "True", "1", "120", "0", "0", "32", "0.001", "1.0", "0",
    "0", "20", "20", "20", "20", "0.0005", "False", "True",
    "True", "True", "8", "False", "0.00005", "10000", "False",
    "clamp", "1.0", "32", "16", "16", "1", "True", "1",
    None,  # canonical absolute candidate_root/data/behavioral_data_block
    "1", "0", "1,2,3", "4", "16", "4", "1.0", "1.0", "1.0",
    "0.0", "1.0", "1", "1.0", "0.05", "cuda", "1", "2", "0.99",
    "True", "False", "4", "False", "16", "2", "1", "0.5", "0.5",
    "False", "1", "False", "1",
)
if len(_SCHEMA13_CLI_WIRE_VALUE_TEXT) != len(_SCHEMA13_CLI_FLAG_ORDER):
    raise RuntimeError("schema-13 frozen CLI flag/value vector is malformed")


def _schema9_stage_xpid_candidate(value):
    """Classify v16 intent without normalizing the value into validity."""

    return (
        type(value) is str
        and value.strip().startswith("enduro-voc-v16-commonmode-eps25-")
    )


def _schema10_stage_xpid_candidate(value):
    """Classify v17 intent without normalizing the value into validity."""

    try:
        lexical_value = os.fspath(value) if isinstance(value, os.PathLike) else value
        if isinstance(
            lexical_value,
            (bytes, bytearray, memoryview, np.bytes_),
        ):
            lexical_value = bytes(lexical_value).decode("utf-8")
        else:
            lexical_value = str(lexical_value)
    except (TypeError, UnicodeError) as exc:
        raise ValueError(
            "schema-10 xpid intent could not be classified before persisted "
            "configuration I/O"
        ) from exc
    return lexical_value.strip().startswith(
        "enduro-voc-v17-huber-common-eps25-"
    )


def _schema11_stage_xpid_candidate(value):
    """Classify v18 intent without normalizing the value into validity."""

    try:
        lexical_value = os.fspath(value) if isinstance(value, os.PathLike) else value
        if isinstance(
            lexical_value,
            (bytes, bytearray, memoryview, np.bytes_),
        ):
            lexical_value = bytes(lexical_value).decode("utf-8")
        else:
            lexical_value = str(lexical_value)
    except (TypeError, UnicodeError) as exc:
        raise ValueError(
            "schema-11 xpid intent could not be classified before persisted "
            "configuration I/O"
        ) from exc
    return lexical_value.strip().startswith(
        "enduro-voc-v18-orthocd-adam-eps25-"
    )


def _schema12_stage_xpid_candidate(value):
    """Classify v19 intent without normalizing the value into validity."""

    try:
        lexical_value = os.fspath(value) if isinstance(value, os.PathLike) else value
        if isinstance(
            lexical_value,
            (bytes, bytearray, memoryview, np.bytes_),
        ):
            lexical_value = bytes(lexical_value).decode("utf-8")
        else:
            lexical_value = str(lexical_value)
    except (TypeError, UnicodeError) as exc:
        raise ValueError(
            "schema-12 xpid intent could not be classified before persisted "
            "configuration I/O"
        ) from exc
    return lexical_value.strip().startswith(
        "enduro-voc-v19-tau1-orthocd-adam-eps25-"
    )


def _schema13_stage_xpid_candidate(value):
    """Classify v20 intent without normalizing the value into validity."""

    try:
        lexical_value = os.fspath(value) if isinstance(value, os.PathLike) else value
        if isinstance(
            lexical_value,
            (bytes, bytearray, memoryview, np.bytes_),
        ):
            lexical_value = bytes(lexical_value).decode("utf-8")
        else:
            lexical_value = str(lexical_value)
    except (TypeError, UnicodeError) as exc:
        raise ValueError(
            "schema-13 xpid intent could not be classified before persisted "
            "configuration I/O"
        ) from exc
    return lexical_value.strip().startswith(
        "enduro-voc-v20-telemetry-tau1-orthocd-adam-eps25-"
    )


def _validate_schema13_cli_vector(raw_args, *, keyword_xpid=None):
    """Fail closed on any drift from the frozen schema-13 CLI vector."""

    if raw_args is None:
        raw_args = sys.argv[1:]
    try:
        tokens = list(raw_args)
    except TypeError as error:
        raise ValueError("schema-13 CLI arguments must be a token sequence") from error
    raw_xpid = None
    for index, token in enumerate(tokens[:-1]):
        if type(token) is str and token == "--xpid":
            raw_xpid = tokens[index + 1]
            break
    claims_schema13 = _schema13_stage_xpid_candidate(raw_xpid) or (
        keyword_xpid is not None
        and _schema13_stage_xpid_candidate(keyword_xpid)
    )
    if not claims_schema13:
        return
    if any(type(token) is not str for token in tokens):
        raise ValueError("schema-13 CLI tokens must be exact Python strings")
    if len(tokens) != 2 * len(_SCHEMA13_CLI_FLAG_ORDER):
        raise ValueError("schema-13 requires exactly 192 CLI tokens / 96 pairs")
    if tuple(tokens[::2]) != _SCHEMA13_CLI_FLAG_ORDER:
        raise ValueError(
            "schema-13 CLI flag order/keyset differs from the exact 96-pair vector"
        )
    values = tuple(tokens[1::2])
    profiles = {
        profile[0]: profile for profile in VOC_GATE_POLICY_SCHEMA13_STAGE_PROFILES
    }
    profile = profiles.get(values[_SCHEMA13_CLI_FLAG_ORDER.index("--xpid")])
    if profile is None:
        raise ValueError("schema-13 CLI xpid is not an exact frozen stage")
    expected = list(_SCHEMA13_CLI_WIRE_VALUE_TEXT)
    for name, value in zip(
        (
            "--xpid",
            "--base_seed",
            "--total_steps",
            "--model_warm_up_n",
            "--actor_unroll_len",
            "--use_wandb",
        ),
        profile,
    ):
        expected[_SCHEMA13_CLI_FLAG_ORDER.index(name)] = str(value)
    savedir_index = _SCHEMA13_CLI_FLAG_ORDER.index("--savedir")
    data_index = _SCHEMA13_CLI_FLAG_ORDER.index("--icopro_data_path")
    savedir = values[savedir_index]
    data_path = values[data_index]
    if (
        not os.path.isabs(savedir)
        or os.path.normpath(savedir) != savedir
        or os.path.basename(savedir) != "runs"
        or not os.path.isabs(data_path)
        or os.path.normpath(data_path) != data_path
        or os.path.basename(data_path) != "behavioral_data_block"
        or os.path.basename(os.path.dirname(data_path)) != "data"
        or os.path.dirname(savedir) != os.path.dirname(os.path.dirname(data_path))
    ):
        raise ValueError("schema-13 CLI candidate/data paths are not canonical")
    expected[savedir_index] = savedir
    expected[data_index] = data_path
    if values != tuple(expected):
        raise ValueError(
            "schema-13 CLI values differ from the exact frozen stage vector"
        )


_VOC_GATE_POLICY_SCHEMA6_BASELINE_CANONICAL = json.dumps(
    dict(VOC_GATE_POLICY_SCHEMA6_V12_BASELINE),
    sort_keys=True,
    ensure_ascii=True,
    separators=(",", ":"),
    allow_nan=False,
)
if (
    len(VOC_GATE_POLICY_SCHEMA6_V12_BASELINE) != 209
    or _VOC_GATE_POLICY_SCHEMA6_BASELINE_CANONICAL
    != _VOC_GATE_POLICY_SCHEMA6_V12_BASELINE_JSON
    or hashlib.sha256(
        _VOC_GATE_POLICY_SCHEMA6_BASELINE_CANONICAL.encode("utf-8")
    ).hexdigest() != VOC_GATE_POLICY_SCHEMA6_V12_BASELINE_SHA256
):
    raise RuntimeError("corrupt source-frozen schema-6 v12 baseline")
_VOC_GATE_POLICY_SCHEMA12_PROJECTION_CANONICAL = json.dumps(
    dict(VOC_GATE_POLICY_SCHEMA12_V12_PROJECTION),
    sort_keys=True,
    ensure_ascii=True,
    separators=(",", ":"),
    allow_nan=False,
)
if (
    len(VOC_GATE_POLICY_SCHEMA12_V12_PROJECTION) != 209
    or len(_VOC_GATE_POLICY_SCHEMA12_PROJECTION_CANONICAL.encode("utf-8"))
    != 4457
    or hashlib.sha256(
        _VOC_GATE_POLICY_SCHEMA12_PROJECTION_CANONICAL.encode("utf-8")
    ).hexdigest() != VOC_GATE_POLICY_SCHEMA12_V12_PROJECTION_SHA256
):
    raise RuntimeError("corrupt source-frozen schema-12 v12 projection")
_VOC_GATE_POLICY_SCHEMA6_COMPLETE_KEYS = (
    set(VOC_GATE_POLICY_SCHEMA6_V12_BASELINE)
    | set(VOC_GATE_POLICY_SCHEMA6_STAGE_FIELDS)
    | set(VOC_GATE_POLICY_SCHEMA6_PATH_FIELDS)
    | set(VOC_GATE_POLICY_SCHEMA6_NEW_FIELDS)
)
if (
    len(_VOC_GATE_POLICY_SCHEMA6_COMPLETE_KEYS) != 228
    or set(VOC_GATE_POLICY_SCHEMA6_V12_BASELINE)
    & (
        set(VOC_GATE_POLICY_SCHEMA6_STAGE_FIELDS)
        | set(VOC_GATE_POLICY_SCHEMA6_PATH_FIELDS)
        | set(VOC_GATE_POLICY_SCHEMA6_NEW_FIELDS)
    )
    or set(VOC_GATE_POLICY_SCHEMA6_STAGE_FIELDS)
    & (
        set(VOC_GATE_POLICY_SCHEMA6_PATH_FIELDS)
        | set(VOC_GATE_POLICY_SCHEMA6_NEW_FIELDS)
    )
    or set(VOC_GATE_POLICY_SCHEMA6_PATH_FIELDS)
    & set(VOC_GATE_POLICY_SCHEMA6_NEW_FIELDS)
):
    raise RuntimeError("schema-6 228-key surface partitions overlap")
_VOC_GATE_POLICY_SCHEMA7_COMPLETE_KEYS = (
    set(VOC_GATE_POLICY_SCHEMA6_V12_BASELINE)
    | set(VOC_GATE_POLICY_SCHEMA6_STAGE_FIELDS)
    | set(VOC_GATE_POLICY_SCHEMA6_PATH_FIELDS)
    | set(VOC_GATE_POLICY_SCHEMA7_NEW_FIELDS)
)
if (
    len(_VOC_GATE_POLICY_SCHEMA7_COMPLETE_KEYS) != 229
    or set(VOC_GATE_POLICY_SCHEMA6_V12_BASELINE)
    & (
        set(VOC_GATE_POLICY_SCHEMA6_STAGE_FIELDS)
        | set(VOC_GATE_POLICY_SCHEMA6_PATH_FIELDS)
        | set(VOC_GATE_POLICY_SCHEMA7_NEW_FIELDS)
    )
    or set(VOC_GATE_POLICY_SCHEMA6_STAGE_FIELDS)
    & (
        set(VOC_GATE_POLICY_SCHEMA6_PATH_FIELDS)
        | set(VOC_GATE_POLICY_SCHEMA7_NEW_FIELDS)
    )
    or set(VOC_GATE_POLICY_SCHEMA6_PATH_FIELDS)
    & set(VOC_GATE_POLICY_SCHEMA7_NEW_FIELDS)
):
    raise RuntimeError("schema-7 229-key surface partitions overlap")
_VOC_GATE_POLICY_SCHEMA8_COMPLETE_KEYS = (
    set(VOC_GATE_POLICY_SCHEMA6_V12_BASELINE)
    | set(VOC_GATE_POLICY_SCHEMA6_STAGE_FIELDS)
    | set(VOC_GATE_POLICY_SCHEMA6_PATH_FIELDS)
    | set(VOC_GATE_POLICY_SCHEMA8_NEW_FIELDS)
)
if (
    len(_VOC_GATE_POLICY_SCHEMA8_COMPLETE_KEYS) != 229
    or set(VOC_GATE_POLICY_SCHEMA6_V12_BASELINE)
    & (
        set(VOC_GATE_POLICY_SCHEMA6_STAGE_FIELDS)
        | set(VOC_GATE_POLICY_SCHEMA6_PATH_FIELDS)
        | set(VOC_GATE_POLICY_SCHEMA8_NEW_FIELDS)
    )
    or set(VOC_GATE_POLICY_SCHEMA6_STAGE_FIELDS)
    & (
        set(VOC_GATE_POLICY_SCHEMA6_PATH_FIELDS)
        | set(VOC_GATE_POLICY_SCHEMA8_NEW_FIELDS)
    )
    or set(VOC_GATE_POLICY_SCHEMA6_PATH_FIELDS)
    & set(VOC_GATE_POLICY_SCHEMA8_NEW_FIELDS)
):
    raise RuntimeError("schema-8 229-key surface partitions overlap")
_VOC_GATE_POLICY_SCHEMA9_COMPLETE_KEYS = (
    set(VOC_GATE_POLICY_SCHEMA6_V12_BASELINE)
    | set(VOC_GATE_POLICY_SCHEMA6_STAGE_FIELDS)
    | set(VOC_GATE_POLICY_SCHEMA6_PATH_FIELDS)
    | set(VOC_GATE_POLICY_SCHEMA9_NEW_FIELDS)
)
if (
    len(_VOC_GATE_POLICY_SCHEMA9_COMPLETE_KEYS) != 229
    or set(VOC_GATE_POLICY_SCHEMA6_V12_BASELINE)
    & (
        set(VOC_GATE_POLICY_SCHEMA6_STAGE_FIELDS)
        | set(VOC_GATE_POLICY_SCHEMA6_PATH_FIELDS)
        | set(VOC_GATE_POLICY_SCHEMA9_NEW_FIELDS)
    )
    or set(VOC_GATE_POLICY_SCHEMA6_STAGE_FIELDS)
    & (
        set(VOC_GATE_POLICY_SCHEMA6_PATH_FIELDS)
        | set(VOC_GATE_POLICY_SCHEMA9_NEW_FIELDS)
    )
    or set(VOC_GATE_POLICY_SCHEMA6_PATH_FIELDS)
    & set(VOC_GATE_POLICY_SCHEMA9_NEW_FIELDS)
):
    raise RuntimeError("schema-9 229-key surface partitions overlap")
_VOC_GATE_POLICY_SCHEMA10_COMPLETE_KEYS = (
    set(VOC_GATE_POLICY_SCHEMA6_V12_BASELINE)
    | set(VOC_GATE_POLICY_SCHEMA6_STAGE_FIELDS)
    | set(VOC_GATE_POLICY_SCHEMA6_PATH_FIELDS)
    | set(VOC_GATE_POLICY_SCHEMA10_NEW_FIELDS)
)
if (
    len(_VOC_GATE_POLICY_SCHEMA10_COMPLETE_KEYS) != 229
    or set(VOC_GATE_POLICY_SCHEMA6_V12_BASELINE)
    & (
        set(VOC_GATE_POLICY_SCHEMA6_STAGE_FIELDS)
        | set(VOC_GATE_POLICY_SCHEMA6_PATH_FIELDS)
        | set(VOC_GATE_POLICY_SCHEMA10_NEW_FIELDS)
    )
    or set(VOC_GATE_POLICY_SCHEMA6_STAGE_FIELDS)
    & (
        set(VOC_GATE_POLICY_SCHEMA6_PATH_FIELDS)
        | set(VOC_GATE_POLICY_SCHEMA10_NEW_FIELDS)
    )
    or set(VOC_GATE_POLICY_SCHEMA6_PATH_FIELDS)
    & set(VOC_GATE_POLICY_SCHEMA10_NEW_FIELDS)
):
    raise RuntimeError("schema-10 229-key surface partitions overlap")
_VOC_GATE_POLICY_SCHEMA11_COMPLETE_KEYS = (
    set(VOC_GATE_POLICY_SCHEMA6_V12_BASELINE)
    | set(VOC_GATE_POLICY_SCHEMA6_STAGE_FIELDS)
    | set(VOC_GATE_POLICY_SCHEMA6_PATH_FIELDS)
    | set(VOC_GATE_POLICY_SCHEMA11_NEW_FIELDS)
)
if (
    len(_VOC_GATE_POLICY_SCHEMA11_COMPLETE_KEYS) != 229
    or set(VOC_GATE_POLICY_SCHEMA6_V12_BASELINE)
    & (
        set(VOC_GATE_POLICY_SCHEMA6_STAGE_FIELDS)
        | set(VOC_GATE_POLICY_SCHEMA6_PATH_FIELDS)
        | set(VOC_GATE_POLICY_SCHEMA11_NEW_FIELDS)
    )
    or set(VOC_GATE_POLICY_SCHEMA6_STAGE_FIELDS)
    & (
        set(VOC_GATE_POLICY_SCHEMA6_PATH_FIELDS)
        | set(VOC_GATE_POLICY_SCHEMA11_NEW_FIELDS)
    )
    or set(VOC_GATE_POLICY_SCHEMA6_PATH_FIELDS)
    & set(VOC_GATE_POLICY_SCHEMA11_NEW_FIELDS)
):
    raise RuntimeError("schema-11 229-key surface partitions overlap")
_VOC_GATE_POLICY_SCHEMA12_COMPLETE_KEYS = (
    set(VOC_GATE_POLICY_SCHEMA6_V12_BASELINE)
    | set(VOC_GATE_POLICY_SCHEMA6_STAGE_FIELDS)
    | set(VOC_GATE_POLICY_SCHEMA6_PATH_FIELDS)
    | set(VOC_GATE_POLICY_SCHEMA12_NEW_FIELDS)
)
if (
    len(_VOC_GATE_POLICY_SCHEMA12_COMPLETE_KEYS) != 229
    or set(VOC_GATE_POLICY_SCHEMA6_V12_BASELINE)
    & (
        set(VOC_GATE_POLICY_SCHEMA6_STAGE_FIELDS)
        | set(VOC_GATE_POLICY_SCHEMA6_PATH_FIELDS)
        | set(VOC_GATE_POLICY_SCHEMA12_NEW_FIELDS)
    )
    or set(VOC_GATE_POLICY_SCHEMA6_STAGE_FIELDS)
    & (
        set(VOC_GATE_POLICY_SCHEMA6_PATH_FIELDS)
        | set(VOC_GATE_POLICY_SCHEMA12_NEW_FIELDS)
    )
    or set(VOC_GATE_POLICY_SCHEMA6_PATH_FIELDS)
    & set(VOC_GATE_POLICY_SCHEMA12_NEW_FIELDS)
):
    raise RuntimeError("schema-12 229-key surface partitions overlap")
_VOC_GATE_POLICY_SCHEMA13_COMPLETE_KEYS = (
    set(VOC_GATE_POLICY_SCHEMA12_V12_PROJECTION)
    | set(VOC_GATE_POLICY_SCHEMA6_STAGE_FIELDS)
    | set(VOC_GATE_POLICY_SCHEMA6_PATH_FIELDS)
    | set(VOC_GATE_POLICY_SCHEMA13_NEW_FIELDS)
)
if (
    len(_VOC_GATE_POLICY_SCHEMA13_COMPLETE_KEYS) != 229
    or set(VOC_GATE_POLICY_SCHEMA12_V12_PROJECTION)
    & (
        set(VOC_GATE_POLICY_SCHEMA6_STAGE_FIELDS)
        | set(VOC_GATE_POLICY_SCHEMA6_PATH_FIELDS)
        | set(VOC_GATE_POLICY_SCHEMA13_NEW_FIELDS)
    )
    or set(VOC_GATE_POLICY_SCHEMA6_STAGE_FIELDS)
    & (
        set(VOC_GATE_POLICY_SCHEMA6_PATH_FIELDS)
        | set(VOC_GATE_POLICY_SCHEMA13_NEW_FIELDS)
    )
    or set(VOC_GATE_POLICY_SCHEMA6_PATH_FIELDS)
    & set(VOC_GATE_POLICY_SCHEMA13_NEW_FIELDS)
):
    raise RuntimeError("schema-13 229-key surface partitions overlap")


def _schema6_canonical_json(value, *, label):
    try:
        return json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not canonical JSON data") from exc


def _validate_schema6_stage_profile(surface, *, label):
    values = []
    for name in (
        "xpid",
        "base_seed",
        "total_steps",
        "model_warm_up_n",
        "actor_unroll_len",
        "use_wandb",
    ):
        if name not in surface:
            raise ValueError(f"{label} lacks schema-6 stage field {name}")
        value = surface[name]
        if name == "xpid":
            if type(value) is not str:
                raise ValueError(f"{label} xpid must be an exact string")
        elif name == "use_wandb":
            if type(value) is not bool:
                raise ValueError(
                    f"{label} use_wandb must be a Python bool"
                )
        elif type(value) is not int:
            raise ValueError(
                f"{label} {name} must be a Python non-bool integer"
            )
        values.append(value)
    stage = tuple(values)
    if stage not in VOC_GATE_POLICY_SCHEMA6_STAGE_PROFILES:
        raise ValueError(f"{label} has unregistered schema-6 stage {stage!r}")
    return stage


def _validate_schema7_stage_profile(surface, *, label):
    values = []
    for name in (
        "xpid",
        "base_seed",
        "total_steps",
        "model_warm_up_n",
        "actor_unroll_len",
        "use_wandb",
    ):
        if name not in surface:
            raise ValueError(f"{label} lacks schema-7 stage field {name}")
        value = surface[name]
        if name == "xpid":
            if type(value) is not str:
                raise ValueError(f"{label} xpid must be an exact string")
        elif name == "use_wandb":
            if type(value) is not bool:
                raise ValueError(
                    f"{label} use_wandb must be a Python bool"
                )
        elif type(value) is not int:
            raise ValueError(
                f"{label} {name} must be a Python non-bool integer"
            )
        values.append(value)
    stage = tuple(values)
    if stage not in VOC_GATE_POLICY_SCHEMA7_STAGE_PROFILES:
        raise ValueError(f"{label} has unregistered schema-7 stage {stage!r}")
    return stage


def _validate_schema8_stage_profile(surface, *, label):
    values = []
    for name in (
        "xpid",
        "base_seed",
        "total_steps",
        "model_warm_up_n",
        "actor_unroll_len",
        "use_wandb",
    ):
        if name not in surface:
            raise ValueError(f"{label} lacks schema-8 stage field {name}")
        value = surface[name]
        if name == "xpid":
            if type(value) is not str:
                raise ValueError(f"{label} xpid must be an exact string")
        elif name == "use_wandb":
            if type(value) is not bool:
                raise ValueError(
                    f"{label} use_wandb must be a Python bool"
                )
        elif type(value) is not int:
            raise ValueError(
                f"{label} {name} must be a Python non-bool integer"
            )
        values.append(value)
    stage = tuple(values)
    if stage not in VOC_GATE_POLICY_SCHEMA8_STAGE_PROFILES:
        raise ValueError(f"{label} has unregistered schema-8 stage {stage!r}")
    return stage


def _validate_schema9_stage_profile(surface, *, label):
    values = []
    for name in (
        "xpid",
        "base_seed",
        "total_steps",
        "model_warm_up_n",
        "actor_unroll_len",
        "use_wandb",
    ):
        if name not in surface:
            raise ValueError(f"{label} lacks schema-9 stage field {name}")
        value = surface[name]
        if name == "xpid":
            if type(value) is not str:
                raise ValueError(f"{label} xpid must be an exact string")
        elif name == "use_wandb":
            if type(value) is not bool:
                raise ValueError(
                    f"{label} use_wandb must be a Python bool"
                )
        elif type(value) is not int:
            raise ValueError(
                f"{label} {name} must be a Python non-bool integer"
            )
        values.append(value)
    stage = tuple(values)
    if stage not in VOC_GATE_POLICY_SCHEMA9_STAGE_PROFILES:
        raise ValueError(f"{label} has unregistered schema-9 stage {stage!r}")
    return stage


def _validate_schema10_stage_profile(surface, *, label):
    values = []
    for name in (
        "xpid",
        "base_seed",
        "total_steps",
        "model_warm_up_n",
        "actor_unroll_len",
        "use_wandb",
    ):
        if name not in surface:
            raise ValueError(f"{label} lacks schema-10 stage field {name}")
        value = surface[name]
        if name == "xpid":
            if type(value) is not str:
                raise ValueError(f"{label} xpid must be an exact string")
        elif name == "use_wandb":
            if type(value) is not bool:
                raise ValueError(
                    f"{label} use_wandb must be a Python bool"
                )
        elif type(value) is not int:
            raise ValueError(
                f"{label} {name} must be a Python non-bool integer"
            )
        values.append(value)
    stage = tuple(values)
    if stage not in VOC_GATE_POLICY_SCHEMA10_STAGE_PROFILES:
        raise ValueError(f"{label} has unregistered schema-10 stage {stage!r}")
    return stage


def _validate_schema11_stage_profile(surface, *, label):
    values = []
    for name in (
        "xpid",
        "base_seed",
        "total_steps",
        "model_warm_up_n",
        "actor_unroll_len",
        "use_wandb",
    ):
        if name not in surface:
            raise ValueError(f"{label} lacks schema-11 stage field {name}")
        value = surface[name]
        if name == "xpid":
            if type(value) is not str:
                raise ValueError(f"{label} xpid must be an exact string")
        elif name == "use_wandb":
            if type(value) is not bool:
                raise ValueError(
                    f"{label} use_wandb must be a Python bool"
                )
        elif type(value) is not int:
            raise ValueError(
                f"{label} {name} must be a Python non-bool integer"
            )
        values.append(value)
    stage = tuple(values)
    if stage not in VOC_GATE_POLICY_SCHEMA11_STAGE_PROFILES:
        raise ValueError(f"{label} has unregistered schema-11 stage {stage!r}")
    return stage


def _validate_schema12_stage_profile(surface, *, label):
    values = []
    for name in (
        "xpid",
        "base_seed",
        "total_steps",
        "model_warm_up_n",
        "actor_unroll_len",
        "use_wandb",
    ):
        if name not in surface:
            raise ValueError(f"{label} lacks schema-12 stage field {name}")
        value = surface[name]
        if name == "xpid":
            if type(value) is not str:
                raise ValueError(f"{label} xpid must be an exact string")
        elif name == "use_wandb":
            if type(value) is not bool:
                raise ValueError(
                    f"{label} use_wandb must be a Python bool"
                )
        elif type(value) is not int:
            raise ValueError(
                f"{label} {name} must be a Python non-bool integer"
            )
        values.append(value)
    stage = tuple(values)
    if stage not in VOC_GATE_POLICY_SCHEMA12_STAGE_PROFILES:
        raise ValueError(f"{label} has unregistered schema-12 stage {stage!r}")
    return stage


def _validate_schema13_stage_profile(surface, *, label):
    values = []
    for name in (
        "xpid",
        "base_seed",
        "total_steps",
        "model_warm_up_n",
        "actor_unroll_len",
        "use_wandb",
    ):
        if name not in surface:
            raise ValueError(f"{label} lacks schema-13 stage field {name}")
        value = surface[name]
        if name == "xpid":
            if type(value) is not str:
                raise ValueError(f"{label} xpid must be an exact string")
        elif name == "use_wandb":
            if type(value) is not bool:
                raise ValueError(
                    f"{label} use_wandb must be a Python bool"
                )
        elif type(value) is not int:
            raise ValueError(
                f"{label} {name} must be a Python non-bool integer"
            )
        values.append(value)
    stage = tuple(values)
    if stage not in VOC_GATE_POLICY_SCHEMA13_STAGE_PROFILES:
        raise ValueError(f"{label} has unregistered schema-13 stage {stage!r}")
    return stage


def _validate_schema6_paths(surface, *, label, expected_ckpdir=None):
    for name in VOC_GATE_POLICY_SCHEMA6_PATH_FIELDS:
        value = surface.get(name)
        if type(value) is not str:
            raise ValueError(f"{label} {name} must be an exact string")
        if name != "cmd":
            if not value or not os.path.isabs(value):
                raise ValueError(f"{label} {name} must be an absolute path")
            if os.path.normpath(value) != value or os.path.realpath(value) != value:
                raise ValueError(
                    f"{label} {name} must be normalized and symlink-free"
                )
        elif not value:
            raise ValueError(f"{label} cmd must be a nonempty captured command")
    savedir = surface["savedir"]
    xpid = surface["xpid"]
    ckpdir = surface["ckpdir"]
    expected_run_dir = os.path.join(savedir, xpid)
    if ckpdir != expected_run_dir or os.path.basename(ckpdir) != xpid:
        raise ValueError(
            f"{label} ckpdir must equal join(savedir, xpid) exactly"
        )
    expected_data = os.path.join(
        os.path.dirname(savedir), "data", "behavioral_data_block"
    )
    if surface["icopro_data_path"] != expected_data:
        raise ValueError(
            f"{label} icopro_data_path does not match the staged data path"
        )
    if expected_ckpdir is not None:
        if type(expected_ckpdir) is not str:
            raise ValueError(f"{label} expected ckpdir must be a string")
        expected_ckpdir = os.path.realpath(os.path.abspath(expected_ckpdir))
        if ckpdir != expected_ckpdir:
            raise ValueError(
                f"{label} persisted ckpdir does not identify its run directory"
            )
    return {
        "savedir": savedir,
        "ckpdir": ckpdir,
        "cmd": surface["cmd"],
        "icopro_data_path": surface["icopro_data_path"],
    }


def _validate_schema6_complete_surface(
    surface, *, label, expected_ckpdir=None
):
    """Validate one exact 228-key immutable schema-6 flag surface."""

    if not isinstance(surface, collections.abc.Mapping):
        raise ValueError(f"{label} must be a mapping")
    expected_keys = _VOC_GATE_POLICY_SCHEMA6_COMPLETE_KEYS
    actual_keys = set(surface)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys, key=repr)
        extra = sorted(actual_keys - expected_keys, key=repr)
        raise ValueError(
            f"{label} requires exact 228-key schema-6 surface; "
            f"missing={missing!r}, extra={extra!r}"
        )
    projection = {
        key: surface[key]
        for key in VOC_GATE_POLICY_SCHEMA6_V12_BASELINE
    }
    for name, expected in VOC_GATE_POLICY_SCHEMA6_V12_BASELINE.items():
        value = projection[name]
        if type(value) is not type(expected) or value != expected:
            kind = "Python boolean" if isinstance(expected, bool) else "canonical value/type"
            raise ValueError(
                f"{label} frozen v12 {name} has wrong {kind}; "
                f"expected {expected!r}, got {value!r}"
            )
    projection_json = _schema6_canonical_json(
        projection, label=f"{label} v12 projection"
    )
    projection_sha256 = hashlib.sha256(
        projection_json.encode("utf-8")
    ).hexdigest()
    if (
        projection_json != _VOC_GATE_POLICY_SCHEMA6_V12_BASELINE_JSON
        or projection_sha256
        != VOC_GATE_POLICY_SCHEMA6_V12_BASELINE_SHA256
    ):
        raise ValueError(
            f"{label} does not match the frozen 209-key v12 baseline"
        )
    stage = _validate_schema6_stage_profile(surface, label=label)
    paths = _validate_schema6_paths(
        surface, label=label, expected_ckpdir=expected_ckpdir
    )
    new_values = {
        "voc_gate_policy_schema_version": 6,
        "voc_gate_execution_epsilon": 0.25,
        "voc_actor_policy_version_barrier": True,
        "voc_actor_policy_bundle_schema_version": 1,
        "voc_actor_policy_barrier_timeout_s": 120.0,
        "voc_actor_policy_ray_max_restarts": 0,
        "voc_actor_policy_ray_max_task_retries": 0,
        "actor_amp_init_scale": 32.0,
        "voc_actor_policy_barrier_runtime": True,
    }
    for name, expected in new_values.items():
        value = surface[name]
        if isinstance(expected, bool):
            matches = type(value) is bool and value is expected
        elif isinstance(expected, int):
            matches = type(value) is int and value == expected
        else:
            matches = (
                type(value) is float
                and np.isfinite(value)
                and value == expected
            )
        if not matches:
            kind = "Python boolean" if isinstance(expected, bool) else "canonical type"
            raise ValueError(
                f"{label} requires schema-6 {name}={expected!r} with "
                f"its {kind}; got {value!r}"
            )
    full_json = _schema6_canonical_json(
        dict(surface), label=f"{label} complete surface"
    )
    return {
        "key_count": len(surface),
        "v12_projection_key_count": len(projection),
        "v12_projection_sha256": projection_sha256,
        "complete_surface_sha256": hashlib.sha256(
            full_json.encode("utf-8")
        ).hexdigest(),
        "stage": stage,
        "paths": paths,
        "canonical_json": full_json,
    }


def _validate_schema7_complete_surface(
    surface, *, label, expected_ckpdir=None
):
    """Validate one exact 229-key immutable schema-7 flag surface."""

    if not isinstance(surface, collections.abc.Mapping):
        raise ValueError(f"{label} must be a mapping")
    expected_keys = _VOC_GATE_POLICY_SCHEMA7_COMPLETE_KEYS
    actual_keys = set(surface)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys, key=repr)
        extra = sorted(actual_keys - expected_keys, key=repr)
        raise ValueError(
            f"{label} requires exact 229-key schema-7 surface; "
            f"missing={missing!r}, extra={extra!r}"
        )
    projection = {
        key: surface[key]
        for key in VOC_GATE_POLICY_SCHEMA6_V12_BASELINE
    }
    for name, expected in VOC_GATE_POLICY_SCHEMA6_V12_BASELINE.items():
        value = projection[name]
        if type(value) is not type(expected) or value != expected:
            kind = (
                "Python boolean"
                if isinstance(expected, bool)
                else "canonical value/type"
            )
            raise ValueError(
                f"{label} frozen v12 {name} has wrong {kind}; "
                f"expected {expected!r}, got {value!r}"
            )
    projection_json = _schema6_canonical_json(
        projection, label=f"{label} v12 projection"
    )
    projection_sha256 = hashlib.sha256(
        projection_json.encode("utf-8")
    ).hexdigest()
    if (
        projection_json != _VOC_GATE_POLICY_SCHEMA6_V12_BASELINE_JSON
        or projection_sha256
        != VOC_GATE_POLICY_SCHEMA6_V12_BASELINE_SHA256
    ):
        raise ValueError(
            f"{label} does not match the frozen 209-key v12 baseline"
        )
    stage = _validate_schema7_stage_profile(surface, label=label)
    paths = _validate_schema6_paths(
        surface, label=label, expected_ckpdir=expected_ckpdir
    )
    new_values = {
        "voc_gate_policy_schema_version": (
            VOC_GATE_POLICY_MODEL_INPUT_SEAL_SCHEMA_VERSION
        ),
        "voc_gate_execution_epsilon": 0.25,
        "voc_actor_policy_version_barrier": True,
        "voc_actor_policy_bundle_schema_version": 1,
        "voc_actor_policy_barrier_timeout_s": 120.0,
        "voc_actor_policy_ray_max_restarts": 0,
        "voc_actor_policy_ray_max_task_retries": 0,
        "actor_amp_init_scale": 32.0,
        "voc_actor_policy_barrier_runtime": True,
        "voc_model_input_seal_schema_version": (
            VOC_MODEL_INPUT_SEAL_SCHEMA_VERSION
        ),
    }
    for name, expected in new_values.items():
        value = surface[name]
        if isinstance(expected, bool):
            matches = type(value) is bool and value is expected
        elif isinstance(expected, int):
            matches = type(value) is int and value == expected
        else:
            matches = (
                type(value) is float
                and np.isfinite(value)
                and value == expected
            )
        if not matches:
            kind = (
                "Python boolean"
                if isinstance(expected, bool)
                else "canonical type"
            )
            raise ValueError(
                f"{label} requires schema-7 {name}={expected!r} with "
                f"its {kind}; got {value!r}"
            )
    full_json = _schema6_canonical_json(
        dict(surface), label=f"{label} complete surface"
    )
    return {
        "key_count": len(surface),
        "v12_projection_key_count": len(projection),
        "v12_projection_sha256": projection_sha256,
        "complete_surface_sha256": hashlib.sha256(
            full_json.encode("utf-8")
        ).hexdigest(),
        "stage": stage,
        "paths": paths,
        "canonical_json": full_json,
    }


def _validate_schema8_complete_surface(
    surface, *, label, expected_ckpdir=None
):
    """Validate one exact 229-key immutable schema-8 flag surface."""

    if not isinstance(surface, collections.abc.Mapping):
        raise ValueError(f"{label} must be a mapping")
    expected_keys = _VOC_GATE_POLICY_SCHEMA8_COMPLETE_KEYS
    actual_keys = set(surface)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys, key=repr)
        extra = sorted(actual_keys - expected_keys, key=repr)
        raise ValueError(
            f"{label} requires exact 229-key schema-8 surface; "
            f"missing={missing!r}, extra={extra!r}"
        )
    projection = {
        key: surface[key]
        for key in VOC_GATE_POLICY_SCHEMA6_V12_BASELINE
    }
    for name, expected in VOC_GATE_POLICY_SCHEMA6_V12_BASELINE.items():
        value = projection[name]
        if type(value) is not type(expected) or value != expected:
            kind = (
                "Python boolean"
                if isinstance(expected, bool)
                else "canonical value/type"
            )
            raise ValueError(
                f"{label} frozen v12 {name} has wrong {kind}; "
                f"expected {expected!r}, got {value!r}"
            )
    projection_json = _schema6_canonical_json(
        projection, label=f"{label} v12 projection"
    )
    projection_sha256 = hashlib.sha256(
        projection_json.encode("utf-8")
    ).hexdigest()
    if (
        projection_json != _VOC_GATE_POLICY_SCHEMA6_V12_BASELINE_JSON
        or projection_sha256
        != VOC_GATE_POLICY_SCHEMA6_V12_BASELINE_SHA256
    ):
        raise ValueError(
            f"{label} does not match the frozen 209-key v12 baseline"
        )
    stage = _validate_schema8_stage_profile(surface, label=label)
    paths = _validate_schema6_paths(
        surface, label=label, expected_ckpdir=expected_ckpdir
    )
    new_values = {
        "voc_gate_policy_schema_version": (
            VOC_GATE_POLICY_HALF_SQUARED_Q_SCHEMA_VERSION
        ),
        "voc_gate_execution_epsilon": 0.25,
        "voc_actor_policy_version_barrier": True,
        "voc_actor_policy_bundle_schema_version": 1,
        "voc_actor_policy_barrier_timeout_s": 120.0,
        "voc_actor_policy_ray_max_restarts": 0,
        "voc_actor_policy_ray_max_task_retries": 0,
        "actor_amp_init_scale": 32.0,
        "voc_actor_policy_barrier_runtime": True,
        "voc_model_input_seal_schema_version": (
            VOC_MODEL_INPUT_SEAL_SCHEMA_VERSION
        ),
    }
    for name, expected in new_values.items():
        value = surface[name]
        if isinstance(expected, bool):
            matches = type(value) is bool and value is expected
        elif isinstance(expected, int):
            matches = type(value) is int and value == expected
        else:
            matches = (
                type(value) is float
                and np.isfinite(value)
                and value == expected
            )
        if not matches:
            kind = (
                "Python boolean"
                if isinstance(expected, bool)
                else "canonical type"
            )
            raise ValueError(
                f"{label} requires schema-8 {name}={expected!r} with "
                f"its {kind}; got {value!r}"
            )
    full_json = _schema6_canonical_json(
        dict(surface), label=f"{label} complete surface"
    )
    return {
        "key_count": len(surface),
        "v12_projection_key_count": len(projection),
        "v12_projection_sha256": projection_sha256,
        "complete_surface_sha256": hashlib.sha256(
            full_json.encode("utf-8")
        ).hexdigest(),
        "stage": stage,
        "paths": paths,
        "canonical_json": full_json,
    }


def _validate_schema9_complete_surface(
    surface, *, label, expected_ckpdir=None
):
    """Validate one exact 229-key immutable schema-9 flag surface."""

    if not isinstance(surface, collections.abc.Mapping):
        raise ValueError(f"{label} must be a mapping")
    expected_keys = _VOC_GATE_POLICY_SCHEMA9_COMPLETE_KEYS
    actual_keys = set(surface)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys, key=repr)
        extra = sorted(actual_keys - expected_keys, key=repr)
        raise ValueError(
            f"{label} requires exact 229-key schema-9 surface; "
            f"missing={missing!r}, extra={extra!r}"
        )
    projection = {
        key: surface[key]
        for key in VOC_GATE_POLICY_SCHEMA6_V12_BASELINE
    }
    for name, expected in VOC_GATE_POLICY_SCHEMA6_V12_BASELINE.items():
        value = projection[name]
        if type(value) is not type(expected) or value != expected:
            kind = (
                "Python boolean"
                if isinstance(expected, bool)
                else "canonical value/type"
            )
            raise ValueError(
                f"{label} frozen v12 {name} has wrong {kind}; "
                f"expected {expected!r}, got {value!r}"
            )
    projection_json = _schema6_canonical_json(
        projection, label=f"{label} v12 projection"
    )
    projection_sha256 = hashlib.sha256(
        projection_json.encode("utf-8")
    ).hexdigest()
    if (
        projection_json != _VOC_GATE_POLICY_SCHEMA6_V12_BASELINE_JSON
        or projection_sha256
        != VOC_GATE_POLICY_SCHEMA6_V12_BASELINE_SHA256
    ):
        raise ValueError(
            f"{label} does not match the frozen 209-key v12 baseline"
        )
    stage = _validate_schema9_stage_profile(surface, label=label)
    paths = _validate_schema6_paths(
        surface, label=label, expected_ckpdir=expected_ckpdir
    )
    new_values = {
        "voc_gate_policy_schema_version": (
            VOC_GATE_POLICY_COMMON_MODE_Q_SCHEMA_VERSION
        ),
        "voc_gate_execution_epsilon": 0.25,
        "voc_actor_policy_version_barrier": True,
        "voc_actor_policy_bundle_schema_version": 1,
        "voc_actor_policy_barrier_timeout_s": 120.0,
        "voc_actor_policy_ray_max_restarts": 0,
        "voc_actor_policy_ray_max_task_retries": 0,
        "actor_amp_init_scale": 32.0,
        "voc_actor_policy_barrier_runtime": True,
        "voc_model_input_seal_schema_version": (
            VOC_MODEL_INPUT_SEAL_SCHEMA_VERSION
        ),
    }
    for name, expected in new_values.items():
        value = surface[name]
        if isinstance(expected, bool):
            matches = type(value) is bool and value is expected
        elif isinstance(expected, int):
            matches = type(value) is int and value == expected
        else:
            matches = (
                type(value) is float
                and np.isfinite(value)
                and value == expected
            )
        if not matches:
            kind = (
                "Python boolean"
                if isinstance(expected, bool)
                else "canonical type"
            )
            raise ValueError(
                f"{label} requires schema-9 {name}={expected!r} with "
                f"its {kind}; got {value!r}"
            )
    full_json = _schema6_canonical_json(
        dict(surface), label=f"{label} complete surface"
    )
    return {
        "key_count": len(surface),
        "v12_projection_key_count": len(projection),
        "v12_projection_sha256": projection_sha256,
        "complete_surface_sha256": hashlib.sha256(
            full_json.encode("utf-8")
        ).hexdigest(),
        "stage": stage,
        "paths": paths,
        "canonical_json": full_json,
    }


def _validate_schema10_complete_surface(
    surface, *, label, expected_ckpdir=None
):
    """Validate one exact 229-key immutable schema-10 flag surface."""

    if not isinstance(surface, collections.abc.Mapping):
        raise ValueError(f"{label} must be a mapping")
    expected_keys = _VOC_GATE_POLICY_SCHEMA10_COMPLETE_KEYS
    actual_keys = set(surface)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys, key=repr)
        extra = sorted(actual_keys - expected_keys, key=repr)
        raise ValueError(
            f"{label} requires exact 229-key schema-10 surface; "
            f"missing={missing!r}, extra={extra!r}"
        )
    projection = {
        key: surface[key]
        for key in VOC_GATE_POLICY_SCHEMA6_V12_BASELINE
    }
    for name, expected in VOC_GATE_POLICY_SCHEMA6_V12_BASELINE.items():
        value = projection[name]
        if type(value) is not type(expected) or value != expected:
            kind = (
                "Python boolean"
                if isinstance(expected, bool)
                else "canonical value/type"
            )
            raise ValueError(
                f"{label} frozen v12 {name} has wrong {kind}; "
                f"expected {expected!r}, got {value!r}"
            )
    projection_json = _schema6_canonical_json(
        projection, label=f"{label} v12 projection"
    )
    projection_sha256 = hashlib.sha256(
        projection_json.encode("utf-8")
    ).hexdigest()
    if (
        projection_json != _VOC_GATE_POLICY_SCHEMA6_V12_BASELINE_JSON
        or projection_sha256
        != VOC_GATE_POLICY_SCHEMA6_V12_BASELINE_SHA256
    ):
        raise ValueError(
            f"{label} does not match the frozen 209-key v12 baseline"
        )
    stage = _validate_schema10_stage_profile(surface, label=label)
    paths = _validate_schema6_paths(
        surface, label=label, expected_ckpdir=expected_ckpdir
    )
    new_values = {
        "voc_gate_policy_schema_version": (
            VOC_GATE_POLICY_HUBER_COMMON_Q_SCHEMA_VERSION
        ),
        "voc_gate_execution_epsilon": 0.25,
        "voc_actor_policy_version_barrier": True,
        "voc_actor_policy_bundle_schema_version": 1,
        "voc_actor_policy_barrier_timeout_s": 120.0,
        "voc_actor_policy_ray_max_restarts": 0,
        "voc_actor_policy_ray_max_task_retries": 0,
        "actor_amp_init_scale": 32.0,
        "voc_actor_policy_barrier_runtime": True,
        "voc_model_input_seal_schema_version": (
            VOC_MODEL_INPUT_SEAL_SCHEMA_VERSION
        ),
    }
    for name, expected in new_values.items():
        value = surface[name]
        if isinstance(expected, bool):
            matches = type(value) is bool and value is expected
        elif isinstance(expected, int):
            matches = type(value) is int and value == expected
        else:
            matches = (
                type(value) is float
                and np.isfinite(value)
                and value == expected
            )
        if not matches:
            kind = (
                "Python boolean"
                if isinstance(expected, bool)
                else "canonical type"
            )
            raise ValueError(
                f"{label} requires schema-10 {name}={expected!r} with "
                f"its {kind}; got {value!r}"
            )
    full_json = _schema6_canonical_json(
        dict(surface), label=f"{label} complete surface"
    )
    return {
        "key_count": len(surface),
        "v12_projection_key_count": len(projection),
        "v12_projection_sha256": projection_sha256,
        "complete_surface_sha256": hashlib.sha256(
            full_json.encode("utf-8")
        ).hexdigest(),
        "stage": stage,
        "paths": paths,
        "canonical_json": full_json,
    }


def _validate_schema11_complete_surface(
    surface, *, label, expected_ckpdir=None
):
    """Validate one exact 229-key immutable schema-11 flag surface."""

    if not isinstance(surface, collections.abc.Mapping):
        raise ValueError(f"{label} must be a mapping")
    expected_keys = _VOC_GATE_POLICY_SCHEMA11_COMPLETE_KEYS
    actual_keys = set(surface)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys, key=repr)
        extra = sorted(actual_keys - expected_keys, key=repr)
        raise ValueError(
            f"{label} requires exact 229-key schema-11 surface; "
            f"missing={missing!r}, extra={extra!r}"
        )
    projection = {
        key: surface[key]
        for key in VOC_GATE_POLICY_SCHEMA6_V12_BASELINE
    }
    for name, expected in VOC_GATE_POLICY_SCHEMA6_V12_BASELINE.items():
        value = projection[name]
        if type(value) is not type(expected) or value != expected:
            kind = (
                "Python boolean"
                if isinstance(expected, bool)
                else "canonical value/type"
            )
            raise ValueError(
                f"{label} frozen v12 {name} has wrong {kind}; "
                f"expected {expected!r}, got {value!r}"
            )
    projection_json = _schema6_canonical_json(
        projection, label=f"{label} v12 projection"
    )
    projection_sha256 = hashlib.sha256(
        projection_json.encode("utf-8")
    ).hexdigest()
    if (
        projection_json != _VOC_GATE_POLICY_SCHEMA6_V12_BASELINE_JSON
        or projection_sha256
        != VOC_GATE_POLICY_SCHEMA6_V12_BASELINE_SHA256
    ):
        raise ValueError(
            f"{label} does not match the frozen 209-key v12 baseline"
        )
    stage = _validate_schema11_stage_profile(surface, label=label)
    paths = _validate_schema6_paths(
        surface, label=label, expected_ckpdir=expected_ckpdir
    )
    new_values = {
        "voc_gate_policy_schema_version": (
            VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION
        ),
        "voc_gate_execution_epsilon": 0.25,
        "voc_actor_policy_version_barrier": True,
        "voc_actor_policy_bundle_schema_version": 1,
        "voc_actor_policy_barrier_timeout_s": 120.0,
        "voc_actor_policy_ray_max_restarts": 0,
        "voc_actor_policy_ray_max_task_retries": 0,
        "actor_amp_init_scale": 32.0,
        "voc_actor_policy_barrier_runtime": True,
        "voc_model_input_seal_schema_version": (
            VOC_MODEL_INPUT_SEAL_SCHEMA_VERSION
        ),
    }
    for name, expected in new_values.items():
        value = surface[name]
        if isinstance(expected, bool):
            matches = type(value) is bool and value is expected
        elif isinstance(expected, int):
            matches = type(value) is int and value == expected
        else:
            matches = (
                type(value) is float
                and np.isfinite(value)
                and value == expected
            )
        if not matches:
            kind = (
                "Python boolean"
                if isinstance(expected, bool)
                else "canonical type"
            )
            raise ValueError(
                f"{label} requires schema-11 {name}={expected!r} with "
                f"its {kind}; got {value!r}"
            )
    full_json = _schema6_canonical_json(
        dict(surface), label=f"{label} complete surface"
    )
    return {
        "key_count": len(surface),
        "v12_projection_key_count": len(projection),
        "v12_projection_sha256": projection_sha256,
        "complete_surface_sha256": hashlib.sha256(
            full_json.encode("utf-8")
        ).hexdigest(),
        "stage": stage,
        "paths": paths,
        "canonical_json": full_json,
    }


def _validate_schema12_complete_surface(
    surface, *, label, expected_ckpdir=None
):
    """Validate one exact 229-key immutable schema-12 flag surface."""

    if not isinstance(surface, collections.abc.Mapping):
        raise ValueError(f"{label} must be a mapping")
    expected_keys = _VOC_GATE_POLICY_SCHEMA12_COMPLETE_KEYS
    actual_keys = set(surface)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys, key=repr)
        extra = sorted(actual_keys - expected_keys, key=repr)
        raise ValueError(
            f"{label} requires exact 229-key schema-12 surface; "
            f"missing={missing!r}, extra={extra!r}"
        )
    projection = {
        key: surface[key]
        for key in VOC_GATE_POLICY_SCHEMA12_V12_PROJECTION
    }
    for name, expected in VOC_GATE_POLICY_SCHEMA12_V12_PROJECTION.items():
        value = projection[name]
        if type(value) is not type(expected) or value != expected:
            kind = (
                "Python boolean"
                if isinstance(expected, bool)
                else "canonical value/type"
            )
            raise ValueError(
                f"{label} frozen schema-12 v12 {name} has wrong {kind}; "
                f"expected {expected!r}, got {value!r}"
            )
    projection_json = _schema6_canonical_json(
        projection, label=f"{label} v12 projection"
    )
    projection_sha256 = hashlib.sha256(
        projection_json.encode("utf-8")
    ).hexdigest()
    if (
        projection_json != _VOC_GATE_POLICY_SCHEMA12_PROJECTION_CANONICAL
        or projection_sha256
        != VOC_GATE_POLICY_SCHEMA12_V12_PROJECTION_SHA256
    ):
        raise ValueError(
            f"{label} does not match the frozen schema-12 209-key projection"
        )
    stage = _validate_schema12_stage_profile(surface, label=label)
    paths = _validate_schema6_paths(
        surface, label=label, expected_ckpdir=expected_ckpdir
    )
    new_values = {
        "voc_gate_policy_schema_version": VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION,
        "voc_gate_execution_epsilon": 0.25,
        "voc_actor_policy_version_barrier": True,
        "voc_actor_policy_bundle_schema_version": 1,
        "voc_actor_policy_barrier_timeout_s": 120.0,
        "voc_actor_policy_ray_max_restarts": 0,
        "voc_actor_policy_ray_max_task_retries": 0,
        "actor_amp_init_scale": 32.0,
        "voc_actor_policy_barrier_runtime": True,
        "voc_model_input_seal_schema_version": (
            VOC_MODEL_INPUT_SEAL_SCHEMA_VERSION
        ),
    }
    for name, expected in new_values.items():
        value = surface[name]
        if isinstance(expected, bool):
            matches = type(value) is bool and value is expected
        elif isinstance(expected, int):
            matches = type(value) is int and value == expected
        else:
            matches = (
                type(value) is float
                and np.isfinite(value)
                and value == expected
            )
        if not matches:
            kind = (
                "Python boolean"
                if isinstance(expected, bool)
                else "canonical type"
            )
            raise ValueError(
                f"{label} requires schema-12 {name}={expected!r} with "
                f"its {kind}; got {value!r}"
            )
    full_json = _schema6_canonical_json(
        dict(surface), label=f"{label} complete surface"
    )
    return {
        "key_count": len(surface),
        "v12_projection_key_count": len(projection),
        "v12_projection_sha256": projection_sha256,
        "complete_surface_sha256": hashlib.sha256(
            full_json.encode("utf-8")
        ).hexdigest(),
        "stage": stage,
        "paths": paths,
        "canonical_json": full_json,
    }


def _validate_schema13_complete_surface(
    surface, *, label, expected_ckpdir=None
):
    """Validate one exact 229-key immutable schema-13 flag surface."""

    if not isinstance(surface, collections.abc.Mapping):
        raise ValueError(f"{label} must be a mapping")
    expected_keys = _VOC_GATE_POLICY_SCHEMA13_COMPLETE_KEYS
    actual_keys = set(surface)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys, key=repr)
        extra = sorted(actual_keys - expected_keys, key=repr)
        raise ValueError(
            f"{label} requires exact 229-key schema-13 surface; "
            f"missing={missing!r}, extra={extra!r}"
        )
    projection = {
        key: surface[key]
        for key in VOC_GATE_POLICY_SCHEMA12_V12_PROJECTION
    }
    for name, expected in VOC_GATE_POLICY_SCHEMA12_V12_PROJECTION.items():
        value = projection[name]
        if type(value) is not type(expected) or value != expected:
            kind = (
                "Python boolean"
                if isinstance(expected, bool)
                else "canonical value/type"
            )
            raise ValueError(
                f"{label} frozen schema-13 v12 {name} has wrong {kind}; "
                f"expected {expected!r}, got {value!r}"
            )
    projection_json = _schema6_canonical_json(
        projection, label=f"{label} v12 projection"
    )
    projection_sha256 = hashlib.sha256(
        projection_json.encode("utf-8")
    ).hexdigest()
    if (
        projection_json != _VOC_GATE_POLICY_SCHEMA12_PROJECTION_CANONICAL
        or projection_sha256
        != VOC_GATE_POLICY_SCHEMA12_V12_PROJECTION_SHA256
    ):
        raise ValueError(
            f"{label} does not match the frozen schema-13 209-key projection"
        )
    stage = _validate_schema13_stage_profile(surface, label=label)
    paths = _validate_schema6_paths(
        surface, label=label, expected_ckpdir=expected_ckpdir
    )
    new_values = {
        "voc_gate_policy_schema_version": (
            VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION
        ),
        "voc_gate_execution_epsilon": 0.25,
        "voc_actor_policy_version_barrier": True,
        "voc_actor_policy_bundle_schema_version": 1,
        "voc_actor_policy_barrier_timeout_s": 120.0,
        "voc_actor_policy_ray_max_restarts": 0,
        "voc_actor_policy_ray_max_task_retries": 0,
        "actor_amp_init_scale": 32.0,
        "voc_actor_policy_barrier_runtime": True,
        "voc_model_input_seal_schema_version": (
            VOC_MODEL_INPUT_SEAL_SCHEMA_VERSION
        ),
    }
    for name, expected in new_values.items():
        value = surface[name]
        if isinstance(expected, bool):
            matches = type(value) is bool and value is expected
        elif isinstance(expected, int):
            matches = type(value) is int and value == expected
        else:
            matches = (
                type(value) is float
                and np.isfinite(value)
                and value == expected
            )
        if not matches:
            kind = (
                "Python boolean"
                if isinstance(expected, bool)
                else "canonical type"
            )
            raise ValueError(
                f"{label} requires schema-13 {name}={expected!r} with "
                f"its {kind}; got {value!r}"
            )
    full_json = _schema6_canonical_json(
        dict(surface), label=f"{label} complete surface"
    )
    return {
        "key_count": len(surface),
        "v12_projection_key_count": len(projection),
        "v12_projection_sha256": projection_sha256,
        "complete_surface_sha256": hashlib.sha256(
            full_json.encode("utf-8")
        ).hexdigest(),
        "stage": stage,
        "paths": paths,
        "canonical_json": full_json,
    }


VOC_ACTOR_POLICY_BUNDLE_KEY = "actor_policy_bundle"
VOC_ACTOR_POLICY_ACKS_KEY = "actor_policy_acks"
VOC_ACTOR_POLICY_ABORT_KEY = "actor_policy_abort"
VOC_ACTOR_POLICY_HEARTBEAT_KEY = "actor_policy_heartbeat"
VOC_ACTOR_POLICY_LOGGER_COMPLETION_SCHEMA_VERSION = 1
VOC_ACTOR_POLICY_LOGGER_SCHEMA13_COMPLETION_SCHEMA_VERSION = 2
VOC_ACTOR_POLICY_LOGGER_FINISH_REQUEST_FILE = (
    "voc_actor_policy_logger_finish_request"
)
VOC_ACTOR_POLICY_LOGGER_FINISH_ACK_FILE = "voc_actor_policy_logger_finish_ack"
VOC_GATE_ADAM_BETA1_LEGACY_DEFAULT = 0.9

_fields = ("real_states", "tree_reps", "xs", "hs")
_fields += ("reward", "episode_return", "episode_step")
_fields += ("done", "real_done", "truncated_done")
_fields += ("max_rollout_depth", "step_status")
_fields += ("last_pri", "last_reset", "cur_gate")
_fields += ("last_search_control", "phase", "legal_control_mask")
_fields += ("tree_token_valid", "search_state_reset", "real_transition")
_fields += (
    "root_carried",
    "carried_descendant_visit_count",
    "carried_descendant_expanded_count",
    "useful_carry",
)
_fields += ("stage_end", "forced_stop", "search_steps")
EnvOut = namedtuple("EnvOut", _fields)


def dynamic_search_enabled(flags):
    """Return whether the opt-in Dynamic Thinker state machine is enabled."""
    return bool(getattr(flags, "dynamic_search", False))


def get_voc_protocol(flags):
    """Return the normalized public configuration of the VoC gate.

    Missing fields deliberately mean the legacy ``off`` protocol.  This makes
    old checkpoints reconstructible while keeping shadow/control checkpoints
    explicit and provenance-checkable.
    """

    return {
        name: getattr(flags, name, default)
        for name, default in VOC_PROTOCOL_DEFAULTS.items()
    }


def _require_environment_return_only_voc(value, *, label):
    """Validate the shared-value anchor used by active VoC training."""

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
    return 0.0


def _require_voc_ema_gate_protocol(enabled, tau, *, label):
    """Validate the mandatory frozen gate-Q target for active VoC."""

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
    return True, float(tau)


def _require_voc_gate_policy_protocol(
    dedicated,
    soft_q_bce,
    q_temperature,
    confidence_weighted,
    adam_beta1,
    learning_rate,
    grad_norm_clipping,
    param_align,
    param_align_coef,
    exact_projection,
    epsilon_greedy_execution,
    *,
    label,
):
    """Validate the isolated soft-Q gate-policy protocol."""

    for name, value in (
        ("voc_dedicated_gate", dedicated),
        ("voc_soft_q_bce_gate", soft_q_bce),
    ):
        if not isinstance(value, (bool, np.bool_)) or not bool(value):
            raise ValueError(f"{label} requires {name}=true")
    if not isinstance(confidence_weighted, (bool, np.bool_)):
        raise ValueError(
            f"{label} requires voc_gate_confidence_weighted to be boolean"
        )
    if not isinstance(param_align, (bool, np.bool_)):
        raise ValueError(
            f"{label} requires voc_gate_param_align to be boolean"
        )
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
    if not isinstance(exact_projection, (bool, np.bool_)):
        raise ValueError(
            f"{label} requires voc_gate_exact_projection to be boolean"
        )
    if bool(exact_projection) and bool(param_align):
        raise ValueError(
            f"{label} requires voc_gate_exact_projection and "
            "voc_gate_param_align to be mutually exclusive"
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
    for name, value in (
        ("voc_gate_q_temperature", q_temperature),
        ("voc_gate_learning_rate", learning_rate),
        ("voc_gate_grad_norm_clipping", grad_norm_clipping),
    ):
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


def validate_voc_gate_policy_schema(checkpoint, *, label="VoC checkpoint"):
    """Resolve schema-versioned gate metadata without relabelling behavior.

    Schema 1 predates both the explicit Adam beta and parameter alignment;
    schema 2 makes beta explicit but still predates alignment.  Schema 3
    requires both alignment fields explicitly.  Schema 4 additionally
    requires the deterministic projection switch explicitly so a v11
    checkpoint cannot silently resume or promote under a different update
    rule.  Schema 5 additionally requires the epsilon-greedy execution switch
    explicitly and can only describe that enabled v12 behavior.  Schema 6 is
    the atomic v13 protocol: a distinct 0.25 execution epsilon, strict actor
    policy-version barrier/bundle schema 1, and main-actor AMP initial scale
    32.  Schema 7 preserves that protocol and additionally requires the
    schema-1 model-input seal.  Schema 8 preserves the sealed protocol and
    selects half-squared selected-action Q regression.  Schema 9 preserves
    that loss and selects common-mode-Q reconstruction.  Schema 10 preserves
    that reconstruction and restores beta-1 SmoothL1 selected-action Q
    regression.  Schema 11 preserves schema-10 Q behavior and selects the
    orthonormal common/difference Adam coordinate adapter.  Schema 12 retains
    schema-11 Q behavior and changes only the existing EMA target rate to the
    exact built-in float 1.0.  Schema 13 retains schema-12 behavior and adds
    only hash-bound telemetry evidence.  Every field introduced after a
    legacy schema canonicalizes to its historical value.
    """

    if not isinstance(checkpoint, collections.abc.Mapping):
        raise ValueError(f"{label} must be a mapping")
    embedded = checkpoint.get("flags", {})
    if not isinstance(embedded, collections.abc.Mapping):
        raise ValueError(f"{label} lacks embedded training flags")
    schema = checkpoint.get("voc_gate_policy_schema_version")
    if (
        isinstance(schema, (bool, np.bool_))
        or not isinstance(schema, (int, np.integer))
        or int(schema) not in (
            VOC_GATE_POLICY_LEGACY_SCHEMA_VERSION,
            VOC_GATE_POLICY_INTERMEDIATE_SCHEMA_VERSION,
            VOC_GATE_POLICY_SCHEMA_VERSION,
            VOC_GATE_POLICY_EXACT_PROJECTION_SCHEMA_VERSION,
            VOC_GATE_POLICY_EPSILON_GREEDY_EXECUTION_SCHEMA_VERSION,
            VOC_GATE_POLICY_VERSION_BARRIER_SCHEMA_VERSION,
            VOC_GATE_POLICY_MODEL_INPUT_SEAL_SCHEMA_VERSION,
            VOC_GATE_POLICY_HALF_SQUARED_Q_SCHEMA_VERSION,
            VOC_GATE_POLICY_COMMON_MODE_Q_SCHEMA_VERSION,
            VOC_GATE_POLICY_HUBER_COMMON_Q_SCHEMA_VERSION,
            VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION,
            VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION,
            VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION,
        )
    ):
        raise ValueError(
            f"{label} has unsupported "
            f"voc_gate_policy_schema_version={schema!r}"
        )
    if (
        int(schema) == VOC_GATE_POLICY_COMMON_MODE_Q_SCHEMA_VERSION
        and type(schema) is not int
    ):
        raise ValueError(
            f"{label} schema 9 requires a built-in Python integer; "
            f"got {schema!r}"
        )
    if (
        int(schema) == VOC_GATE_POLICY_HUBER_COMMON_Q_SCHEMA_VERSION
        and type(schema) is not int
    ):
        raise ValueError(
            f"{label} schema 10 requires a built-in Python integer; "
            f"got {schema!r}"
        )
    if (
        int(schema) == VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION
        and type(schema) is not int
    ):
        raise ValueError(
            f"{label} schema 11 requires a built-in Python integer; "
            f"got {schema!r}"
        )
    if (
        int(schema) == VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION
        and type(schema) is not int
    ):
        raise ValueError(
            f"{label} schema 12 requires a built-in Python integer; "
            f"got {schema!r}"
        )
    if (
        int(schema) == VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION
        and type(schema) is not int
    ):
        raise ValueError(
            f"{label} schema 13 requires a built-in Python integer; "
            f"got {schema!r}"
        )
    schema = int(schema)
    if (
        schema in VOC_GATE_POLICY_ATOMIC_SCHEMA_VERSIONS
        and embedded.get("dynamic_voc_mode", "off") != "control"
    ):
        raise ValueError(
            f"{label} voc_gate_policy_schema_version {schema} requires "
            "dynamic_voc_mode='control'"
        )
    if schema == VOC_GATE_POLICY_VERSION_BARRIER_SCHEMA_VERSION:
        _validate_schema6_complete_surface(
            embedded, label=f"{label} embedded flags"
        )
    elif schema == VOC_GATE_POLICY_MODEL_INPUT_SEAL_SCHEMA_VERSION:
        _validate_schema7_complete_surface(
            embedded, label=f"{label} embedded flags"
        )
    elif schema == VOC_GATE_POLICY_HALF_SQUARED_Q_SCHEMA_VERSION:
        _validate_schema8_complete_surface(
            embedded, label=f"{label} embedded flags"
        )
    elif schema == VOC_GATE_POLICY_COMMON_MODE_Q_SCHEMA_VERSION:
        _validate_schema9_complete_surface(
            embedded, label=f"{label} embedded flags"
        )
    elif schema == VOC_GATE_POLICY_HUBER_COMMON_Q_SCHEMA_VERSION:
        _validate_schema10_complete_surface(
            embedded, label=f"{label} embedded flags"
        )
    elif schema == VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION:
        _validate_schema11_complete_surface(
            embedded, label=f"{label} embedded flags"
        )
    elif schema == VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION:
        _validate_schema12_complete_surface(
            embedded, label=f"{label} embedded flags"
        )
    elif schema == VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION:
        _validate_schema13_complete_surface(
            embedded, label=f"{label} embedded flags"
        )
    beta1_legacy_defaulted = "voc_gate_adam_beta1" not in embedded
    if beta1_legacy_defaulted:
        if schema != VOC_GATE_POLICY_LEGACY_SCHEMA_VERSION:
            raise ValueError(
                f"{label} schema {schema} lacks embedded "
                "voc_gate_adam_beta1"
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
                f"{label} requires 0 <= voc_gate_adam_beta1 < 1; "
                f"got {beta1!r}"
            )
        beta1 = float(beta1)
        if (
            schema == VOC_GATE_POLICY_LEGACY_SCHEMA_VERSION
            and beta1 != VOC_GATE_ADAM_BETA1_LEGACY_DEFAULT
        ):
            raise ValueError(
                f"{label} schema 1 requires legacy "
                f"voc_gate_adam_beta1={VOC_GATE_ADAM_BETA1_LEGACY_DEFAULT}; "
                f"got {beta1!r}"
            )

    align_present = "voc_gate_param_align" in embedded
    coefficient_present = "voc_gate_param_align_coef" in embedded
    if schema >= VOC_GATE_POLICY_SCHEMA_VERSION:
        for name, present in (
            ("voc_gate_param_align", align_present),
            ("voc_gate_param_align_coef", coefficient_present),
        ):
            if not present:
                raise ValueError(
                    f"{label} schema {schema} lacks embedded {name}"
                )
    elif align_present != coefficient_present:
        missing = (
            "voc_gate_param_align_coef"
            if align_present
            else "voc_gate_param_align"
        )
        raise ValueError(
            f"{label} schema {schema} has partial legacy alignment "
            f"metadata; lacks embedded {missing}"
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
                f"{label} requires voc_gate_param_align to be boolean; "
                f"got {param_align!r}"
            )
        param_align = bool(param_align)
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
        param_align_coef = 1.0
        if schema < VOC_GATE_POLICY_SCHEMA_VERSION and param_align:
            raise ValueError(
                f"{label} schema {schema} predates parameter alignment and "
                "requires voc_gate_param_align=false"
            )
    projection_present = "voc_gate_exact_projection" in embedded
    if (
        schema >= VOC_GATE_POLICY_EXACT_PROJECTION_SCHEMA_VERSION
        and not projection_present
    ):
        raise ValueError(
            f"{label} schema {schema} lacks embedded "
            "voc_gate_exact_projection"
        )
    projection_legacy_defaulted = not projection_present
    if projection_legacy_defaulted:
        exact_projection = False
    else:
        exact_projection = embedded["voc_gate_exact_projection"]
        if not isinstance(exact_projection, (bool, np.bool_)):
            raise ValueError(
                f"{label} requires voc_gate_exact_projection to be boolean; "
                f"got {exact_projection!r}"
            )
        exact_projection = bool(exact_projection)
        if (
            schema < VOC_GATE_POLICY_EXACT_PROJECTION_SCHEMA_VERSION
            and exact_projection
        ):
            raise ValueError(
                f"{label} schema {schema} predates exact projection and "
                "requires voc_gate_exact_projection=false"
            )
    if (
        schema >= VOC_GATE_POLICY_EXACT_PROJECTION_SCHEMA_VERSION
        and not exact_projection
    ):
        raise ValueError(
            f"{label} schema {schema} requires "
            "voc_gate_exact_projection=true"
        )
    if exact_projection and param_align:
        raise ValueError(
            f"{label} requires voc_gate_exact_projection and "
            "voc_gate_param_align to be mutually exclusive"
        )
    execution_present = "voc_gate_epsilon_greedy_execution" in embedded
    if schema >= VOC_GATE_POLICY_EPSILON_GREEDY_EXECUTION_SCHEMA_VERSION and not execution_present:
        raise ValueError(
            f"{label} schema {schema} lacks embedded "
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
                f"{label} requires voc_gate_epsilon_greedy_execution to be "
                f"boolean; got {epsilon_greedy_execution!r}"
            )
        epsilon_greedy_execution = bool(epsilon_greedy_execution)
        if (
            schema < VOC_GATE_POLICY_EPSILON_GREEDY_EXECUTION_SCHEMA_VERSION
            and epsilon_greedy_execution
        ):
            raise ValueError(
                f"{label} schema {schema} predates epsilon-greedy execution "
                "and requires voc_gate_epsilon_greedy_execution=false"
            )
    if schema >= VOC_GATE_POLICY_EPSILON_GREEDY_EXECUTION_SCHEMA_VERSION and not epsilon_greedy_execution:
        raise ValueError(
            f"{label} schema {schema} requires "
            "voc_gate_epsilon_greedy_execution=true"
        )
    if epsilon_greedy_execution and not exact_projection:
        raise ValueError(
            f"{label} epsilon-greedy execution requires exact projection"
        )

    v13_defaults = {
        "voc_gate_execution_epsilon": 0.02,
        "voc_actor_policy_version_barrier": False,
        "voc_actor_policy_bundle_schema_version": (
            VOC_ACTOR_POLICY_BUNDLE_SCHEMA_VERSION
        ),
        "voc_actor_policy_barrier_timeout_s": (
            VOC_ACTOR_POLICY_BARRIER_TIMEOUT_SECONDS
        ),
        "voc_actor_policy_ray_max_restarts": 0,
        "voc_actor_policy_ray_max_task_retries": 0,
        "actor_amp_init_scale": 256.0,
    }
    v13_values = {}
    v13_defaulted = {}
    for name, default in v13_defaults.items():
        present = name in embedded
        if schema in VOC_GATE_POLICY_ATOMIC_SCHEMA_VERSIONS and not present:
            raise ValueError(f"{label} schema {schema} lacks embedded {name}")
        value = embedded[name] if present else default
        if name == "voc_actor_policy_version_barrier":
            if (
                schema in VOC_GATE_POLICY_ATOMIC_SCHEMA_VERSIONS
                and type(value) is not bool
            ) or (
                schema not in VOC_GATE_POLICY_ATOMIC_SCHEMA_VERSIONS
                and not isinstance(value, (bool, np.bool_))
            ):
                raise ValueError(f"{label} requires {name} to be boolean")
            value = bool(value)
        elif name == "voc_actor_policy_bundle_schema_version":
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or int(value) != VOC_ACTOR_POLICY_BUNDLE_SCHEMA_VERSION
            ):
                raise ValueError(
                    f"{label} requires {name}="
                    f"{VOC_ACTOR_POLICY_BUNDLE_SCHEMA_VERSION} exactly; "
                    f"got {value!r}"
                )
            value = int(value)
        elif name in (
            "voc_actor_policy_ray_max_restarts",
            "voc_actor_policy_ray_max_task_retries",
        ):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or int(value) != 0
            ):
                raise ValueError(f"{label} requires {name}=0 exactly")
            value = int(value)
        else:
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.number))
                or not np.isfinite(value)
            ):
                raise ValueError(f"{label} requires finite numeric {name}")
            value = float(value)
        if schema < VOC_GATE_POLICY_VERSION_BARRIER_SCHEMA_VERSION and value != default:
            raise ValueError(
                f"{label} schema {schema} predates the v13 atomic protocol "
                f"and requires {name}={default!r}; got {value!r}"
            )
        v13_values[name] = value
        v13_defaulted[name] = not present

    seal_present = "voc_model_input_seal_schema_version" in embedded
    seal_schema = embedded.get("voc_model_input_seal_schema_version", 0)
    if type(seal_schema) is not int:
        raise ValueError(
            f"{label} requires voc_model_input_seal_schema_version to be a "
            f"Python non-bool integer; got {seal_schema!r}"
        )
    expected_seal_schema = (
        VOC_MODEL_INPUT_SEAL_SCHEMA_VERSION
        if schema in (
            VOC_GATE_POLICY_MODEL_INPUT_SEAL_SCHEMA_VERSION,
            VOC_GATE_POLICY_HALF_SQUARED_Q_SCHEMA_VERSION,
            VOC_GATE_POLICY_COMMON_MODE_Q_SCHEMA_VERSION,
            VOC_GATE_POLICY_HUBER_COMMON_Q_SCHEMA_VERSION,
            VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION,
            VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION,
            VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION,
        )
        else 0
    )
    if seal_schema != expected_seal_schema:
        raise ValueError(
            f"{label} schema {schema} requires "
            "voc_model_input_seal_schema_version="
            f"{expected_seal_schema}; got {seal_schema!r}"
        )

    if schema in VOC_GATE_POLICY_ATOMIC_SCHEMA_VERSIONS:
        for name, expected in VOC_GATE_POLICY_SCHEMA6_ATOMIC_REQUIREMENTS.items():
            if (
                schema in (
                    VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION,
                    VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION,
                )
                and name == "voc_gate_target_tau"
            ):
                expected = 1.0
            if name not in embedded:
                raise ValueError(
                    f"{label} schema {schema} lacks embedded {name}"
                )
            value = embedded[name]
            if isinstance(expected, bool):
                matches = type(value) is bool and value is expected
            else:
                matches = (
                    not isinstance(value, (bool, np.bool_))
                    and isinstance(value, (int, float, np.number))
                    and np.isfinite(value)
                    and float(value) == expected
                )
            if not matches:
                if isinstance(expected, bool):
                        raise ValueError(
                            f"schema-{schema} {name} must be a Python bool "
                            f"equal to {expected!r}; got {value!r}"
                        )
                raise ValueError(
                    f"{label} schema {schema} requires embedded "
                    f"{name}={expected!r} exactly; got {value!r}"
                )
        for name, expected in VOC_GATE_POLICY_SCHEMA6_OPTIMIZER_REQUIREMENTS.items():
            if name not in embedded:
                raise ValueError(
                    f"{label} schema {schema} lacks embedded {name}"
                )
            value = embedded[name]
            if isinstance(expected, bool):
                matches = type(value) is bool and value is expected
            elif isinstance(expected, str):
                matches = type(value) is str and value == expected
            else:
                matches = (
                    not isinstance(value, (bool, np.bool_))
                    and isinstance(value, (int, float, np.number))
                    and np.isfinite(value)
                    and float(value) == expected
                )
            if not matches:
                raise ValueError(
                    f"{label} schema {schema} requires embedded "
                    f"{name}={expected!r} exactly; got {value!r}"
                )
        for name, expected in VOC_GATE_POLICY_SCHEMA6_ENDURO_REQUIREMENTS.items():
            if name not in embedded:
                raise ValueError(
                    f"{label} schema {schema} lacks embedded {name}"
                )
            value = embedded[name]
            if isinstance(expected, bool):
                matches = type(value) is bool and value is expected
            elif isinstance(expected, int):
                matches = (
                    not isinstance(value, (bool, np.bool_))
                    and isinstance(value, (int, np.integer))
                    and int(value) == expected
                )
            else:
                matches = type(value) is str and value == expected
            if not matches:
                raise ValueError(
                    f"{label} schema {schema} requires embedded "
                    f"{name}={expected!r} exactly; got {value!r}"
                )
        required = {
            "voc_gate_execution_epsilon": 0.25,
            "voc_actor_policy_version_barrier": True,
            "voc_actor_policy_bundle_schema_version": (
                VOC_ACTOR_POLICY_BUNDLE_SCHEMA_VERSION
            ),
            "voc_actor_policy_barrier_timeout_s": (
                VOC_ACTOR_POLICY_BARRIER_TIMEOUT_SECONDS
            ),
            "voc_actor_policy_ray_max_restarts": 0,
            "voc_actor_policy_ray_max_task_retries": 0,
            "actor_amp_init_scale": 32.0,
        }
        for name, expected in required.items():
            if v13_values[name] != expected:
                raise ValueError(
                    f"{label} schema {schema} requires {name}={expected!r} "
                    f"exactly; got {v13_values[name]!r}"
                )
        for name, expected in (
            ("ppo_k", 1),
            ("self_play_n", 1),
            ("env_n", 16),
            ("actor_batch_size", 16),
        ):
            value = embedded.get(name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or int(value) != expected
            ):
                raise ValueError(
                    f"{label} schema {schema} requires embedded "
                    f"{name}={expected}; got {value!r}"
                )
            v13_values[name] = int(value)
        for name in ("ckp", "train_actor", "parallel_actor"):
            value = embedded.get(name)
            if type(value) is not bool:
                raise ValueError(
                    f"{label} schema {schema} requires Python boolean "
                    f"embedded {name}"
                )
        if bool(embedded["ckp"]):
            raise ValueError(f"{label} schema {schema} is fresh-only")
        if not bool(embedded["train_actor"]) or not bool(
            embedded["parallel_actor"]
        ):
            raise ValueError(
                f"{label} schema {schema} requires parallel actor training"
            )
        for name, expected in (
            ("float16", True),
            ("model_float16", False),
            ("dual_net", True),
            ("train_model", True),
        ):
            value = embedded.get(name)
            if type(value) is not bool or value is not expected:
                raise ValueError(
                    f"{label} schema {schema} requires embedded "
                    f"{name}={expected!r} exactly; got {value!r}"
                )
        model_optimizer = embedded.get("model_optimizer")
        if type(model_optimizer) is not str or model_optimizer != "adam":
            raise ValueError(
                f"{label} schema {schema} requires embedded "
                f"model_optimizer='adam' exactly; got {model_optimizer!r}"
            )
        schedule_total_steps = embedded.get("schedule_total_steps")
        if (
            isinstance(schedule_total_steps, (bool, np.bool_))
            or not isinstance(schedule_total_steps, (int, np.integer))
            or int(schedule_total_steps) != 100_000_000
        ):
            raise ValueError(
                f"{label} schema {schema} requires embedded "
                "schedule_total_steps=100000000 exactly; got "
                f"{schedule_total_steps!r}"
            )
        for name in ("preload", "preload_actor", "voc_parent_checkpoint"):
            value = embedded.get(name)
            if not isinstance(value, str) or value != "":
                raise ValueError(
                    f"{label} schema {schema} fresh origin requires {name}=''"
                )
    return {
        "voc_gate_policy_schema_version": schema,
        "voc_gate_adam_beta1": float(beta1),
        "voc_gate_adam_beta1_legacy_defaulted": beta1_legacy_defaulted,
        "voc_gate_param_align": bool(param_align),
        "voc_gate_param_align_coef": float(param_align_coef),
        "voc_gate_param_align_legacy_defaulted": (
            align_legacy_defaulted
        ),
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
        **(
            {
                "voc_model_input_seal_schema_version": int(seal_schema),
                "voc_model_input_seal_schema_version_legacy_defaulted": (
                    not seal_present
                ),
            }
            if schema in (
                VOC_GATE_POLICY_MODEL_INPUT_SEAL_SCHEMA_VERSION,
                VOC_GATE_POLICY_HALF_SQUARED_Q_SCHEMA_VERSION,
                VOC_GATE_POLICY_COMMON_MODE_Q_SCHEMA_VERSION,
                VOC_GATE_POLICY_HUBER_COMMON_Q_SCHEMA_VERSION,
                VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION,
                VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION,
                VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION,
            )
            else {}
        ),
        **v13_values,
        **{
            f"{name}_legacy_defaulted": defaulted
            for name, defaulted in v13_defaulted.items()
        },
    }


def actor_policy_state_sha256(actor_state):
    """Hash sorted tensor identity metadata plus contiguous CPU bytes."""

    if not isinstance(actor_state, collections.abc.Mapping):
        raise ValueError("actor policy state must be a mapping")
    digest = hashlib.sha256()
    for key in sorted(actor_state):
        if not isinstance(key, str):
            raise ValueError("actor policy state keys must be strings")
        tensor = actor_state[key]
        if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
            raise ValueError(
                f"actor policy state {key!r} must be a strided tensor"
            )
        tensor = tensor.detach().cpu().contiguous()
        if not torch.isfinite(tensor).all():
            raise ValueError(f"actor policy state {key!r} is non-finite")
        header = json.dumps(
            [key, str(tensor.dtype), list(tensor.shape)],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        raw = tensor.view(torch.uint8).numpy().tobytes(order="C")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def actor_policy_publication_history_sha256(history):
    if not isinstance(history, (list, tuple)):
        raise ValueError("actor policy publication history must be a sequence")
    payload = json.dumps(
        list(history),
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def clone_actor_policy_state(actor_state):
    """Return an immutable-publication copy with no live parameter storage."""

    if not isinstance(actor_state, collections.abc.Mapping):
        raise ValueError("actor policy state must be a mapping")
    cloned = collections.OrderedDict()
    for key in sorted(actor_state):
        if not isinstance(key, str):
            raise ValueError("actor policy state keys must be strings")
        tensor = actor_state[key]
        if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
            raise ValueError(
                f"actor policy state {key!r} must be a strided tensor"
            )
        tensor = tensor.detach().cpu().contiguous().clone()
        if tensor.requires_grad:
            raise ValueError(f"actor policy state {key!r} remains trainable")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"actor policy state {key!r} is non-finite")
        cloned[key] = tensor
    return cloned


def validate_actor_policy_state(
    actor_state, *, expected_actor_state=None, require_equal=False,
    label="actor policy state",
):
    """Reject noncanonical persisted state; never normalize it into validity."""

    if not isinstance(actor_state, collections.abc.Mapping) or not actor_state:
        raise ValueError(f"{label} must be a nonempty mapping")
    if expected_actor_state is not None:
        if (
            not isinstance(expected_actor_state, collections.abc.Mapping)
            or not expected_actor_state
            or set(actor_state) != set(expected_actor_state)
        ):
            raise ValueError(f"{label} keys disagree with template")
    storage_ptrs = set()
    for key in sorted(actor_state):
        if not isinstance(key, str):
            raise ValueError(f"{label} keys must be strings")
        tensor = actor_state[key]
        if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
            raise ValueError(f"{label} {key!r} must be a strided tensor")
        if tensor.device.type != "cpu":
            raise ValueError(f"{label} {key!r} must be on CPU")
        if tensor.requires_grad:
            raise ValueError(f"{label} {key!r} must be detached")
        if not tensor.is_contiguous():
            raise ValueError(f"{label} {key!r} must be contiguous")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{label} {key!r} is non-finite")
        if tensor.numel():
            storage_ptr = tensor.untyped_storage().data_ptr()
            if storage_ptr in storage_ptrs:
                raise ValueError(f"{label} tensors alias storage")
            storage_ptrs.add(storage_ptr)
        if expected_actor_state is not None:
            template = expected_actor_state[key]
            if not isinstance(template, torch.Tensor):
                raise ValueError(f"{label} template {key!r} is not a tensor")
            if (
                tensor.dtype != template.dtype
                or tuple(tensor.shape) != tuple(template.shape)
            ):
                raise ValueError(f"{label} {key!r} metadata disagrees")
            if require_equal and not torch.equal(tensor, template.detach().cpu()):
                raise ValueError(f"{label} {key!r} values disagree")
    return clone_actor_policy_state(actor_state)


def _validate_atomic_gate_schema(value, *, label):
    if type(value) is not int or value not in VOC_GATE_POLICY_ATOMIC_SCHEMA_VERSIONS:
        raise ValueError(
            f"{label} must be exact Python integer 6 through 13; "
            f"got {value!r}"
        )
    return value


def make_actor_policy_bundle(
    actor_state,
    epoch,
    *,
    terminal=False,
    gate_schema=VOC_GATE_POLICY_VERSION_BARRIER_SCHEMA_VERSION,
):
    """Construct the exact schema-1 actor-policy publication atomically."""

    if (
        isinstance(epoch, (bool, np.bool_))
        or not isinstance(epoch, (int, np.integer))
        or int(epoch) < 0
    ):
        raise ValueError("actor policy bundle epoch must be non-negative")
    if type(terminal) is not bool:
        raise ValueError("actor policy bundle terminal must be boolean")
    gate_schema = _validate_atomic_gate_schema(
        gate_schema, label="actor policy bundle gate_schema"
    )
    actor_state = clone_actor_policy_state(actor_state)
    return {
        "bundle_schema_version": VOC_ACTOR_POLICY_BUNDLE_SCHEMA_VERSION,
        "policy_version": int(epoch),
        "terminal": bool(terminal),
        "gate_schema": int(gate_schema),
        "actor_state_dict": actor_state,
    }


def validate_actor_policy_bundle(
    bundle, *, expected_epoch=None, expected_terminal=None,
    expected_actor_state=None, require_equal_state=False,
    expected_gate_schema=VOC_GATE_POLICY_VERSION_BARRIER_SCHEMA_VERSION,
    label="actor policy bundle",
):
    """Validate the complete barrier bundle before loading any weights."""

    required = {
        "bundle_schema_version",
        "policy_version",
        "terminal",
        "gate_schema",
        "actor_state_dict",
    }
    if not isinstance(bundle, collections.abc.Mapping) or set(bundle) != required:
        raise ValueError(f"{label} must contain exactly {sorted(required)}")
    expected_gate_schema = _validate_atomic_gate_schema(
        expected_gate_schema, label=f"{label} expected gate_schema"
    )
    bundle_schema = bundle["bundle_schema_version"]
    if expected_gate_schema in (
        VOC_GATE_POLICY_MODEL_INPUT_SEAL_SCHEMA_VERSION,
        VOC_GATE_POLICY_HALF_SQUARED_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_COMMON_MODE_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_HUBER_COMMON_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION,
    ):
        valid_bundle_schema = (
            type(bundle_schema) is int
            and bundle_schema == VOC_ACTOR_POLICY_BUNDLE_SCHEMA_VERSION
        )
    else:
        valid_bundle_schema = (
            not isinstance(bundle_schema, (bool, np.bool_))
            and isinstance(bundle_schema, (int, np.integer))
            and int(bundle_schema) == VOC_ACTOR_POLICY_BUNDLE_SCHEMA_VERSION
        )
    if not valid_bundle_schema:
        raise ValueError(f"{label} has invalid bundle schema")
    gate_schema = bundle["gate_schema"]
    if expected_gate_schema in (
        VOC_GATE_POLICY_MODEL_INPUT_SEAL_SCHEMA_VERSION,
        VOC_GATE_POLICY_HALF_SQUARED_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_COMMON_MODE_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_HUBER_COMMON_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION,
    ):
        valid_gate_schema = (
            type(gate_schema) is int and gate_schema == expected_gate_schema
        )
    else:
        valid_gate_schema = (
            not isinstance(gate_schema, (bool, np.bool_))
            and isinstance(gate_schema, (int, np.integer))
            and int(gate_schema) == expected_gate_schema
        )
    if not valid_gate_schema:
        raise ValueError(f"{label} has invalid gate-policy schema")
    epoch = bundle["policy_version"]
    if (
        isinstance(epoch, (bool, np.bool_))
        or not isinstance(epoch, (int, np.integer))
        or int(epoch) < 0
    ):
        raise ValueError(f"{label} has invalid epoch")
    terminal = bundle["terminal"]
    if type(terminal) is not bool:
        raise ValueError(f"{label} has invalid terminal marker")
    if not isinstance(bundle["actor_state_dict"], collections.abc.Mapping):
        raise ValueError(f"{label} lacks actor weights")
    if expected_epoch is not None and (
        isinstance(expected_epoch, (bool, np.bool_))
        or not isinstance(expected_epoch, (int, np.integer))
        or int(expected_epoch) < 0
    ):
        raise ValueError(f"{label} expected epoch is invalid")
    if expected_epoch is not None and int(epoch) != int(expected_epoch):
        raise ValueError(
            f"{label} epoch {int(epoch)} != expected {int(expected_epoch)}"
        )
    if expected_terminal is not None and type(expected_terminal) is not bool:
        raise ValueError(f"{label} expected terminal marker is invalid")
    if expected_terminal is not None and bool(terminal) != bool(expected_terminal):
        raise ValueError(
            f"{label} terminal {bool(terminal)} != expected "
            f"{bool(expected_terminal)}"
        )
    raw_state = bundle["actor_state_dict"]
    if not isinstance(expected_actor_state, collections.abc.Mapping) or not expected_actor_state:
        raise ValueError(f"{label} requires a nonempty expected actor template")
    actor_state = validate_actor_policy_state(
        raw_state,
        expected_actor_state=expected_actor_state,
        require_equal=require_equal_state,
        label=f"{label} actor state",
    )
    return {
        "bundle_schema_version": VOC_ACTOR_POLICY_BUNDLE_SCHEMA_VERSION,
        "policy_version": int(epoch),
        "terminal": bool(terminal),
        "gate_schema": int(gate_schema),
        "actor_state_dict": actor_state,
    }


def make_actor_policy_ack(
    rank,
    epoch,
    *,
    terminal=False,
    gate_schema=VOC_GATE_POLICY_VERSION_BARRIER_SCHEMA_VERSION,
):
    if (
        isinstance(rank, (bool, np.bool_))
        or not isinstance(rank, (int, np.integer))
        or int(rank) < 0
    ):
        raise ValueError("actor policy ack rank must be non-negative")
    if (
        isinstance(epoch, (bool, np.bool_))
        or not isinstance(epoch, (int, np.integer))
        or int(epoch) < 0
    ):
        raise ValueError("actor policy ack epoch must be non-negative")
    if type(terminal) is not bool:
        raise ValueError("actor policy ack terminal must be boolean")
    gate_schema = _validate_atomic_gate_schema(
        gate_schema, label="actor policy ack gate_schema"
    )
    return {
        "bundle_schema_version": VOC_ACTOR_POLICY_BUNDLE_SCHEMA_VERSION,
        "gate_schema": gate_schema,
        "rank": int(rank),
        "policy_version": int(epoch),
        "terminal": bool(terminal),
    }


def validate_actor_policy_ack(
    ack,
    *,
    rank=None,
    epoch=None,
    terminal=None,
    expected_gate_schema=VOC_GATE_POLICY_VERSION_BARRIER_SCHEMA_VERSION,
    label="actor policy ack",
):
    required = {
        "bundle_schema_version", "gate_schema", "rank", "policy_version",
        "terminal",
    }
    if not isinstance(ack, collections.abc.Mapping) or set(ack) != required:
        raise ValueError(f"{label} must contain exactly {sorted(required)}")
    expected_gate_schema = _validate_atomic_gate_schema(
        expected_gate_schema, label=f"{label} expected gate_schema"
    )
    bundle_schema = ack["bundle_schema_version"]
    if expected_gate_schema in (
        VOC_GATE_POLICY_MODEL_INPUT_SEAL_SCHEMA_VERSION,
        VOC_GATE_POLICY_HALF_SQUARED_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_COMMON_MODE_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_HUBER_COMMON_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION,
    ):
        valid_bundle_schema = (
            type(bundle_schema) is int
            and bundle_schema == VOC_ACTOR_POLICY_BUNDLE_SCHEMA_VERSION
        )
    else:
        valid_bundle_schema = (
            not isinstance(bundle_schema, (bool, np.bool_))
            and isinstance(bundle_schema, (int, np.integer))
            and int(bundle_schema) == VOC_ACTOR_POLICY_BUNDLE_SCHEMA_VERSION
        )
    if not valid_bundle_schema:
        raise ValueError(f"{label} has invalid bundle_schema_version")
    gate_schema = ack["gate_schema"]
    if expected_gate_schema in (
        VOC_GATE_POLICY_MODEL_INPUT_SEAL_SCHEMA_VERSION,
        VOC_GATE_POLICY_HALF_SQUARED_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_COMMON_MODE_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_HUBER_COMMON_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION,
    ):
        valid_gate_schema = (
            type(gate_schema) is int and gate_schema == expected_gate_schema
        )
    else:
        valid_gate_schema = (
            not isinstance(gate_schema, (bool, np.bool_))
            and isinstance(gate_schema, (int, np.integer))
            and int(gate_schema) == expected_gate_schema
        )
    if not valid_gate_schema:
        raise ValueError(f"{label} has invalid gate_schema")
    expected_rank = ack["rank"] if rank is None else rank
    expected_epoch = ack["policy_version"] if epoch is None else epoch
    expected_terminal = ack["terminal"] if terminal is None else terminal
    canonical = make_actor_policy_ack(
        expected_rank,
        expected_epoch,
        terminal=expected_terminal,
        gate_schema=expected_gate_schema,
    )
    if dict(ack) != canonical:
        raise ValueError(f"{label} disagrees with the expected publication")
    return canonical


def validate_actor_policy_heartbeat(
    heartbeat, *, previous=None, label="actor policy heartbeat"
):
    """Validate the latest-value heartbeat and report genuine progress."""

    required = {"rank", "policy_version", "phase", "count"}
    if not isinstance(heartbeat, collections.abc.Mapping) or set(heartbeat) != required:
        raise ValueError(f"{label} must contain exactly {sorted(required)}")
    for name in ("rank", "policy_version", "count"):
        value = heartbeat[name]
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
        ):
            raise ValueError(f"{label} {name} must be a non-bool integer")
    rank = int(heartbeat["rank"])
    version = int(heartbeat["policy_version"])
    count = int(heartbeat["count"])
    phase = heartbeat["phase"]
    if rank != 0 or version < 0 or count <= 0:
        raise ValueError(f"{label} rank/version/count is invalid")
    if phase == "load_ack":
        expected_count = 2 * version + 1
    elif phase == "enqueue":
        expected_count = 2 * version + 2
    else:
        raise ValueError(f"{label} phase is invalid")
    if count != expected_count:
        raise ValueError(f"{label} count/phase/version relation is invalid")
    canonical = {
        "rank": 0,
        "policy_version": version,
        "phase": phase,
        "count": count,
    }
    if previous is None:
        return canonical, True
    previous_canonical, _ = validate_actor_policy_heartbeat(
        previous, label=f"previous {label}"
    )
    if count < previous_canonical["count"]:
        raise ValueError(f"{label} count regressed")
    if count == previous_canonical["count"]:
        if canonical != previous_canonical:
            raise ValueError(f"{label} changed without increasing count")
        return canonical, False
    return canonical, True


def validate_actor_policy_checkpoint(checkpoint, *, label="actor checkpoint"):
    """Validate persisted atomic lifecycle and terminal state identity."""

    schema = validate_voc_gate_policy_schema(checkpoint, label=label)
    gate_schema = schema["voc_gate_policy_schema_version"]
    if gate_schema not in VOC_GATE_POLICY_ATOMIC_SCHEMA_VERSIONS:
        raise ValueError(f"{label} is not an atomic gate-policy schema")
    version = checkpoint.get("voc_actor_policy_version")
    publication_count = checkpoint.get("voc_actor_policy_publication_count")
    for name, value in (
        ("voc_actor_policy_version", version),
        ("voc_actor_policy_publication_count", publication_count),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < 0
        ):
            raise ValueError(f"{label} has invalid {name}")
    if int(version) != int(publication_count):
        raise ValueError(f"{label} policy version/count are not lockstep")
    terminal = checkpoint.get("voc_actor_policy_terminal")
    if type(terminal) is not bool:
        raise ValueError(f"{label} has invalid policy terminal marker")
    if terminal and int(version) < 1:
        raise ValueError(
            f"{label} terminal publication must follow at least one batch"
        )
    expected_acks = checkpoint.get("voc_actor_policy_expected_ack_count")
    terminal_acks = checkpoint.get("voc_actor_policy_terminal_ack_count")
    for name, value in (
        ("voc_actor_policy_expected_ack_count", expected_acks),
        ("voc_actor_policy_terminal_ack_count", terminal_acks),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < 0
        ):
            raise ValueError(f"{label} has invalid {name}")
    if int(expected_acks) != 1:
        raise ValueError(
            f"{label} schema {gate_schema} requires exactly one worker ack"
        )
    if int(terminal_acks) != (int(expected_acks) if terminal else 0):
        raise ValueError(f"{label} terminal acknowledgement count disagrees")
    for name in (
        "voc_actor_policy_version_mismatch_count",
        "voc_actor_policy_malformed_bundle_count",
        "voc_actor_policy_barrier_timeout_count",
    ):
        value = checkpoint.get(name)
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) != 0
        ):
            raise ValueError(f"{label} requires {name}=0")
    raw_checkpoint_state = checkpoint.get("actor_net_state_dict")
    actor_state = validate_actor_policy_state(
        raw_checkpoint_state,
        label=f"{label} actor_net_state_dict",
    )
    raw_bundle = checkpoint.get("voc_actor_policy_bundle")
    if isinstance(raw_bundle, collections.abc.Mapping):
        raw_bundle_state = raw_bundle.get("actor_state_dict")
        if isinstance(raw_bundle_state, collections.abc.Mapping):
            checkpoint_ptrs = {
                tensor.untyped_storage().data_ptr()
                for tensor in raw_checkpoint_state.values()
                if isinstance(tensor, torch.Tensor) and tensor.numel()
            }
            bundle_ptrs = {
                tensor.untyped_storage().data_ptr()
                for tensor in raw_bundle_state.values()
                if isinstance(tensor, torch.Tensor) and tensor.numel()
            }
            if checkpoint_ptrs & bundle_ptrs:
                raise ValueError(
                    f"{label} bundle and checkpoint actor states alias storage"
                )
    bundle = validate_actor_policy_bundle(
        raw_bundle,
        expected_epoch=int(version),
        expected_terminal=bool(terminal),
        expected_actor_state=actor_state,
        require_equal_state=True,
        expected_gate_schema=gate_schema,
        label=f"{label} policy bundle",
    )
    digest = checkpoint.get("voc_actor_policy_state_sha256")
    expected_digest = actor_policy_state_sha256(actor_state)
    if (
        not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        or digest != expected_digest
        or actor_policy_state_sha256(bundle["actor_state_dict"]) != digest
    ):
        raise ValueError(f"{label} actor policy state digest disagrees")
    embedded = checkpoint["flags"]
    if embedded.get("float16") is not True:
        raise ValueError(
            f"{label} schema {gate_schema} requires main actor AMP enabled"
        )
    actor_amp_init_scale = embedded.get("actor_amp_init_scale")
    if (
        isinstance(actor_amp_init_scale, (bool, np.bool_))
        or not isinstance(actor_amp_init_scale, (int, float, np.number))
        or not np.isfinite(actor_amp_init_scale)
        or float(actor_amp_init_scale) != 32.0
    ):
        raise ValueError(f"{label} requires actor_amp_init_scale=32.0")
    actor_scaler = checkpoint.get("actor_grad_scaler_state_dict")
    scaler_keys = {
        "scale",
        "growth_factor",
        "backoff_factor",
        "growth_interval",
        "_growth_tracker",
    }
    if not isinstance(actor_scaler, collections.abc.Mapping) or set(actor_scaler) != scaler_keys:
        raise ValueError(f"{label} has invalid main actor GradScaler state")
    scale = actor_scaler["scale"]
    growth_factor = actor_scaler["growth_factor"]
    backoff_factor = actor_scaler["backoff_factor"]
    growth_interval = actor_scaler["growth_interval"]
    growth_tracker = actor_scaler["_growth_tracker"]
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.number))
        or not np.isfinite(value)
        for value in (scale, growth_factor, backoff_factor)
    ):
        raise ValueError(f"{label} main actor GradScaler floats are invalid")
    if (
        isinstance(growth_interval, (bool, np.bool_))
        or not isinstance(growth_interval, (int, np.integer))
        or int(growth_interval) != 2000
        or isinstance(growth_tracker, (bool, np.bool_))
        or not isinstance(growth_tracker, (int, np.integer))
    ):
        raise ValueError(f"{label} main actor GradScaler counters are invalid")
    expected_tracker = int(version) % 2000
    try:
        expected_scale = math.ldexp(32.0, int(version) // 2000)
    except OverflowError as error:
        raise ValueError(f"{label} main actor GradScaler scale overflowed") from error
    if (
        float(growth_factor) != 2.0
        or float(backoff_factor) != 0.5
        or int(growth_tracker) != expected_tracker
        or float(scale) != expected_scale
    ):
        raise ValueError(
            f"{label} main actor GradScaler is not reconstructible from "
            "publication count"
        )
    for name in ("actor_amp_skip_count", "actor_amp_consecutive_skips"):
        value = checkpoint.get(name)
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) != 0
        ):
            raise ValueError(f"{label} requires {name}=0")
    history = checkpoint.get("voc_actor_policy_publication_history")
    event_keys = {
        "predecessor_version",
        "policy_version",
        "publication_count",
        "terminal",
        "ack_ranks",
        "expected_ack_count",
        "state_sha256",
    }
    if not isinstance(history, (list, tuple)) or len(history) != int(version) + 1:
        raise ValueError(f"{label} has invalid actor policy publication history")
    for index, event in enumerate(history):
        if not isinstance(event, collections.abc.Mapping) or set(event) != event_keys:
            raise ValueError(f"{label} publication history event is malformed")
        for name, expected in (
            ("predecessor_version", index - 1),
            ("policy_version", index),
            ("publication_count", index),
            ("expected_ack_count", 1),
        ):
            value = event[name]
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or int(value) != expected
            ):
                raise ValueError(
                    f"{label} publication history {name} is not contiguous"
                )
        ack_ranks = event["ack_ranks"]
        if (
            type(ack_ranks) is not list
            or len(ack_ranks) != 1
            or isinstance(ack_ranks[0], (bool, np.bool_))
            or not isinstance(ack_ranks[0], (int, np.integer))
            or int(ack_ranks[0]) != 0
        ):
            raise ValueError(f"{label} publication history lacks full ack")
        event_terminal = event["terminal"]
        if type(event_terminal) is not bool:
            raise ValueError(f"{label} publication terminal is not boolean")
        if bool(event_terminal) != bool(terminal and index == len(history) - 1):
            raise ValueError(
                f"{label} publication history terminal is not final-only"
            )
        event_digest = event["state_sha256"]
        if (
            not isinstance(event_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", event_digest) is None
        ):
            raise ValueError(f"{label} publication history digest is invalid")
    if history[-1]["state_sha256"] != digest:
        raise ValueError(f"{label} final publication digest disagrees")
    history_digest = checkpoint.get(
        "voc_actor_policy_publication_history_sha256"
    )
    if history_digest != actor_policy_publication_history_sha256(history):
        raise ValueError(f"{label} publication history digest disagrees")
    real_step = checkpoint.get("real_step")
    total_steps = embedded.get("total_steps")
    if (
        isinstance(real_step, (bool, np.bool_))
        or not isinstance(real_step, (int, np.integer))
        or int(real_step) < 0
        or isinstance(total_steps, (bool, np.bool_))
        or not isinstance(total_steps, (int, np.integer))
        or int(total_steps) <= 0
    ):
        raise ValueError(f"{label} has invalid real_step/total_steps progress")
    if terminal != (int(real_step) >= int(total_steps)):
        raise ValueError(
            f"{label} terminal marker disagrees with exact progress boundary"
        )
    return {
        "voc_actor_policy_version": int(version),
        "voc_actor_policy_publication_count": int(publication_count),
        "voc_actor_policy_terminal": bool(terminal),
        "voc_actor_policy_version_mismatch_count": 0,
        "voc_actor_policy_malformed_bundle_count": 0,
        "voc_actor_policy_barrier_timeout_count": 0,
        "voc_actor_policy_terminal_ack_count": int(terminal_acks),
        "voc_actor_policy_expected_ack_count": int(expected_acks),
        "voc_actor_policy_state_sha256": digest,
        "actor_amp_init_scale": 32.0,
        "actor_amp_scale": float(scale),
        "actor_amp_growth_tracker": int(growth_tracker),
        "actor_amp_skip_count": 0,
        "actor_amp_consecutive_skips": 0,
        "voc_actor_policy_bundle_summary": {
            "bundle_schema_version": bundle["bundle_schema_version"],
            "policy_version": bundle["policy_version"],
            "terminal": bundle["terminal"],
            "gate_schema": bundle["gate_schema"],
            "actor_state_dict_sha256": actor_policy_state_sha256(
                bundle["actor_state_dict"]
            ),
            "actor_state_dict_key_count": len(bundle["actor_state_dict"]),
            "actor_state_dict_keys": list(bundle["actor_state_dict"].keys()),
            "actor_state_dict_metadata": [
                {
                    "key": key,
                    "dtype": str(bundle["actor_state_dict"][key].dtype),
                    "shape": list(bundle["actor_state_dict"][key].shape),
                    "numel": int(bundle["actor_state_dict"][key].numel()),
                }
                for key in bundle["actor_state_dict"]
            ],
        },
        "voc_actor_policy_publication_history": tuple(
            copy.deepcopy(list(history))
        ),
        "voc_actor_policy_publication_history_sha256": history_digest,
        "voc_actor_policy_publication_event_count": len(history),
        "voc_actor_policy_final_publication_event": copy.deepcopy(
            history[-1]
        ),
    }


def _validate_schema6_protocol_flags(flags, *, label, expected_ckpdir=None):
    if not isinstance(flags, collections.abc.Mapping):
        raise ValueError(f"{label} must be a mapping")
    _validate_schema6_complete_surface(
        flags, label=label, expected_ckpdir=expected_ckpdir
    )
    schema = flags.get("voc_gate_policy_schema_version")
    if (
        isinstance(schema, (bool, np.bool_))
        or not isinstance(schema, (int, np.integer))
        or int(schema) != VOC_GATE_POLICY_VERSION_BARRIER_SCHEMA_VERSION
    ):
        raise ValueError(f"{label} requires explicit integer schema 6")
    record = {
        "voc_gate_policy_schema_version": int(schema),
        "flags": flags,
    }
    resolved = validate_voc_gate_policy_schema(record, label=label)
    if (
        resolved["voc_gate_policy_schema_version"]
        != VOC_GATE_POLICY_VERSION_BARRIER_SCHEMA_VERSION
    ):
        raise ValueError(f"{label} is not schema 6")
    return resolved


def _validate_schema7_protocol_flags(flags, *, label, expected_ckpdir=None):
    if not isinstance(flags, collections.abc.Mapping):
        raise ValueError(f"{label} must be a mapping")
    _validate_schema7_complete_surface(
        flags, label=label, expected_ckpdir=expected_ckpdir
    )
    schema = flags.get("voc_gate_policy_schema_version")
    if (
        type(schema) is not int
        or schema != VOC_GATE_POLICY_MODEL_INPUT_SEAL_SCHEMA_VERSION
    ):
        raise ValueError(f"{label} requires explicit Python integer schema 7")
    record = {
        "voc_gate_policy_schema_version": schema,
        "flags": flags,
    }
    resolved = validate_voc_gate_policy_schema(record, label=label)
    if (
        resolved["voc_gate_policy_schema_version"]
        != VOC_GATE_POLICY_MODEL_INPUT_SEAL_SCHEMA_VERSION
    ):
        raise ValueError(f"{label} is not schema 7")
    return resolved


def _validate_schema8_protocol_flags(flags, *, label, expected_ckpdir=None):
    if not isinstance(flags, collections.abc.Mapping):
        raise ValueError(f"{label} must be a mapping")
    _validate_schema8_complete_surface(
        flags, label=label, expected_ckpdir=expected_ckpdir
    )
    schema = flags.get("voc_gate_policy_schema_version")
    if (
        type(schema) is not int
        or schema != VOC_GATE_POLICY_HALF_SQUARED_Q_SCHEMA_VERSION
    ):
        raise ValueError(f"{label} requires explicit Python integer schema 8")
    record = {
        "voc_gate_policy_schema_version": schema,
        "flags": flags,
    }
    resolved = validate_voc_gate_policy_schema(record, label=label)
    if (
        resolved["voc_gate_policy_schema_version"]
        != VOC_GATE_POLICY_HALF_SQUARED_Q_SCHEMA_VERSION
    ):
        raise ValueError(f"{label} is not schema 8")
    return resolved


def _validate_schema9_protocol_flags(flags, *, label, expected_ckpdir=None):
    if not isinstance(flags, collections.abc.Mapping):
        raise ValueError(f"{label} must be a mapping")
    _validate_schema9_complete_surface(
        flags, label=label, expected_ckpdir=expected_ckpdir
    )
    schema = flags.get("voc_gate_policy_schema_version")
    if (
        type(schema) is not int
        or schema != VOC_GATE_POLICY_COMMON_MODE_Q_SCHEMA_VERSION
    ):
        raise ValueError(f"{label} requires explicit Python integer schema 9")
    record = {
        "voc_gate_policy_schema_version": schema,
        "flags": flags,
    }
    resolved = validate_voc_gate_policy_schema(record, label=label)
    if (
        resolved["voc_gate_policy_schema_version"]
        != VOC_GATE_POLICY_COMMON_MODE_Q_SCHEMA_VERSION
    ):
        raise ValueError(f"{label} is not schema 9")
    return resolved


def _validate_schema10_protocol_flags(flags, *, label, expected_ckpdir=None):
    if not isinstance(flags, collections.abc.Mapping):
        raise ValueError(f"{label} must be a mapping")
    _validate_schema10_complete_surface(
        flags, label=label, expected_ckpdir=expected_ckpdir
    )
    schema = flags.get("voc_gate_policy_schema_version")
    if (
        type(schema) is not int
        or schema != VOC_GATE_POLICY_HUBER_COMMON_Q_SCHEMA_VERSION
    ):
        raise ValueError(f"{label} requires explicit Python integer schema 10")
    record = {
        "voc_gate_policy_schema_version": schema,
        "flags": flags,
    }
    resolved = validate_voc_gate_policy_schema(record, label=label)
    if (
        resolved["voc_gate_policy_schema_version"]
        != VOC_GATE_POLICY_HUBER_COMMON_Q_SCHEMA_VERSION
    ):
        raise ValueError(f"{label} is not schema 10")
    return resolved


def _validate_schema11_protocol_flags(flags, *, label, expected_ckpdir=None):
    if not isinstance(flags, collections.abc.Mapping):
        raise ValueError(f"{label} must be a mapping")
    _validate_schema11_complete_surface(
        flags, label=label, expected_ckpdir=expected_ckpdir
    )
    schema = flags.get("voc_gate_policy_schema_version")
    if (
        type(schema) is not int
        or schema != VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION
    ):
        raise ValueError(f"{label} requires explicit Python integer schema 11")
    record = {
        "voc_gate_policy_schema_version": schema,
        "flags": flags,
    }
    resolved = validate_voc_gate_policy_schema(record, label=label)
    if (
        resolved["voc_gate_policy_schema_version"]
        != VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION
    ):
        raise ValueError(f"{label} is not schema 11")
    return resolved


def _validate_schema12_protocol_flags(flags, *, label, expected_ckpdir=None):
    if not isinstance(flags, collections.abc.Mapping):
        raise ValueError(f"{label} must be a mapping")
    _validate_schema12_complete_surface(
        flags, label=label, expected_ckpdir=expected_ckpdir
    )
    schema = flags.get("voc_gate_policy_schema_version")
    if type(schema) is not int or schema != VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION:
        raise ValueError(f"{label} requires explicit Python integer schema 12")
    record = {
        "voc_gate_policy_schema_version": schema,
        "flags": flags,
    }
    resolved = validate_voc_gate_policy_schema(record, label=label)
    if (
        resolved["voc_gate_policy_schema_version"]
        != VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION
    ):
        raise ValueError(f"{label} is not schema 12")
    return resolved


def _validate_schema13_protocol_flags(flags, *, label, expected_ckpdir=None):
    if not isinstance(flags, collections.abc.Mapping):
        raise ValueError(f"{label} must be a mapping")
    _validate_schema13_complete_surface(
        flags, label=label, expected_ckpdir=expected_ckpdir
    )
    schema = flags.get("voc_gate_policy_schema_version")
    if (
        type(schema) is not int
        or schema != VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION
    ):
        raise ValueError(f"{label} requires explicit Python integer schema 13")
    record = {
        "voc_gate_policy_schema_version": schema,
        "flags": flags,
    }
    resolved = validate_voc_gate_policy_schema(record, label=label)
    if (
        resolved["voc_gate_policy_schema_version"]
        != VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION
    ):
        raise ValueError(f"{label} is not schema 13")
    return resolved


_SCHEMA13_TELEMETRY_EVIDENCE_KEYS = frozenset({
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
})


def validate_schema13_telemetry_manifest(
    ckpdir,
    *,
    expected_xpid=None,
    expected_terminal_policy_version=None,
    expected_terminal_real_step=None,
    expected_actor_state_sha256=None,
    expected_publication_history_sha256=None,
    expected_stage_total_steps=None,
    expected_actor_unroll_len=None,
    expected_terminal_ack_count=1,
    expected_manifest_sha256=None,
    expected_manifest_size=None,
    expected_q_initial_lr=None,
    expected_schedule_total_steps=None,
    expected_amp_initial_scale=None,
    expected_publication_history=None,
    expected_terminal_state=None,
):
    """Strictly validate schema-13 telemetry after classifying its config."""

    root = os.path.realpath(os.path.abspath(os.fspath(ckpdir)))
    config_path = os.path.join(root, "config_c.yaml")
    config_payload, _ = _stable_regular_file_bytes(
        config_path, label="schema-13 telemetry config"
    )
    try:
        config = yaml.safe_load(config_payload.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError(
            "schema-13 telemetry config must be strict UTF-8 YAML"
        ) from error
    if not isinstance(config, collections.abc.Mapping):
        raise ValueError("schema-13 telemetry config must be a mapping")
    _validate_schema13_complete_surface(
        config,
        label="schema-13 telemetry config",
        expected_ckpdir=root,
    )
    config_xpid = config["xpid"]
    if expected_xpid is None:
        expected_xpid = config_xpid
    elif type(expected_xpid) is not str or expected_xpid != config_xpid:
        raise ValueError("schema-13 telemetry expected xpid disagrees with config")
    if expected_stage_total_steps is None:
        expected_stage_total_steps = config["total_steps"]
    if expected_actor_unroll_len is None:
        expected_actor_unroll_len = config["actor_unroll_len"]

    # The telemetry module is a schema-13-only leaf and must never be imported
    # by schemas at most 12.
    from thinker import voc_telemetry

    if (
        getattr(voc_telemetry, "VOC_TELEMETRY_SCHEMA_VERSION", None)
        != VOC_TELEMETRY_SCHEMA_VERSION
        or getattr(voc_telemetry, "MANIFEST_FILENAME", None)
        != "voc_telemetry_manifest.json"
    ):
        raise RuntimeError("schema-13 telemetry module constants disagree")
    evidence = voc_telemetry.validate_schema13_telemetry_manifest(
        root,
        expected_xpid=expected_xpid,
        expected_terminal_policy_version=expected_terminal_policy_version,
        expected_terminal_real_step=expected_terminal_real_step,
        expected_actor_state_sha256=expected_actor_state_sha256,
        expected_publication_history_sha256=(
            expected_publication_history_sha256
        ),
        expected_stage_total_steps=expected_stage_total_steps,
        expected_actor_unroll_len=expected_actor_unroll_len,
        expected_terminal_ack_count=expected_terminal_ack_count,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_manifest_size=expected_manifest_size,
        expected_q_initial_lr=expected_q_initial_lr,
        expected_schedule_total_steps=expected_schedule_total_steps,
        expected_amp_initial_scale=expected_amp_initial_scale,
        expected_publication_history=expected_publication_history,
        expected_terminal_state=expected_terminal_state,
    )
    if (
        not isinstance(evidence, collections.abc.Mapping)
        or set(evidence) != _SCHEMA13_TELEMETRY_EVIDENCE_KEYS
    ):
        raise ValueError("schema-13 telemetry validator returned malformed evidence")
    if (
        type(evidence["telemetry_schema_version"]) is not int
        or evidence["telemetry_schema_version"] != VOC_TELEMETRY_SCHEMA_VERSION
        or type(evidence["gate_schema"]) is not int
        or evidence["gate_schema"]
        != VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION
        or evidence["manifest_name"] != "voc_telemetry_manifest.json"
    ):
        raise ValueError("schema-13 telemetry evidence identity is invalid")
    return copy.deepcopy(dict(evidence))


def _validate_final_model_state(model_state, *, label):
    if not isinstance(model_state, collections.abc.Mapping) or not model_state:
        raise ValueError(f"{label} must be a nonempty mapping")
    storage_ptrs = set()
    tensor_count = 0
    for key in sorted(model_state):
        if not isinstance(key, str):
            raise ValueError(f"{label} keys must be strings")
        tensor = model_state[key]
        if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
            raise ValueError(f"{label} {key!r} must be a strided tensor")
        if tensor.device.type != "cpu" or tensor.requires_grad:
            raise ValueError(f"{label} {key!r} must be detached on CPU")
        if not tensor.is_contiguous():
            raise ValueError(f"{label} {key!r} must be contiguous")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{label} {key!r} is non-finite")
        if tensor.numel():
            storage_ptr = tensor.untyped_storage().data_ptr()
            if storage_ptr in storage_ptrs:
                raise ValueError(f"{label} tensors alias storage")
            storage_ptrs.add(storage_ptr)
        tensor_count += 1
    return tensor_count


def _schema6_capture_rng_state():
    """Capture every process RNG surface that CPU template creation may touch."""

    cuda_initialized = bool(torch.cuda.is_initialized())
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state().clone(),
        "torch_cuda": (
            tuple(state.clone() for state in torch.cuda.get_rng_state_all())
            if cuda_initialized else None
        ),
    }


def _schema6_restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(list(state["torch_cuda"]))


def _reconstruct_schema6_enduro_networks(config, *, label):
    """Rebuild the frozen Enduro ActorNet/ModelNet templates on CPU.

    The stage contract is source-hardcoded rather than inferred from the
    checkpoint being checked.  Constructors do not create/reset/step an
    environment and do not read behavioral data.  Their random initializers
    are made observationally pure by restoring all process RNG states.
    """

    if not isinstance(config, collections.abc.Mapping):
        raise ValueError(f"{label} config must be a mapping")
    for name, expected in VOC_GATE_POLICY_SCHEMA6_ENDURO_REQUIREMENTS.items():
        if name not in config:
            raise ValueError(f"{label} lacks frozen Enduro field {name}")
        value = config[name]
        if isinstance(expected, bool):
            matches = type(value) is bool and value is expected
        elif isinstance(expected, int):
            matches = (
                not isinstance(value, (bool, np.bool_))
                and isinstance(value, (int, np.integer))
                and int(value) == expected
            )
        else:
            matches = type(value) is str and value == expected
        if not matches:
            raise ValueError(
                f"{label} requires frozen Enduro {name}={expected!r}; "
                f"got {value!r}"
            )

    # This protocol is intentionally Enduro-only.  Pong/Space Invaders need
    # a new preregistered profile rather than action/shape inference here.
    env_n = 16
    action_n = 9
    real_state_shape = (12, 84, 84)
    hidden_shape = (256, 6, 6)
    tree_width = 10 * action_n + 14
    flags = argparse.Namespace(**copy.deepcopy(dict(config)))
    rng_state = _schema6_capture_rng_state()
    try:
        # Lazy imports avoid util <-> network module import cycles.
        from thinker.actor_net import ActorNet
        from thinker.model_net import ModelNet

        single_observation_space = spaces.Box(
            low=0,
            high=255,
            shape=real_state_shape,
            dtype=np.uint8,
        )
        single_action_space = spaces.Discrete(action_n)
        model_net = ModelNet(
            obs_space=single_observation_space,
            action_space=single_action_space,
            flags=flags,
            frame_stack_n=4,
        ).cpu()
        if tuple(model_net.hidden_shape) != hidden_shape:
            raise ValueError(
                f"{label} reconstructed ModelNet hidden shape changed: "
                f"{tuple(model_net.hidden_shape)!r}"
            )
        actor_observation_space = spaces.Dict({
            "tree_reps": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(env_n, tree_width),
                dtype=np.float32,
            ),
            "real_states": spaces.Box(
                low=0,
                high=255,
                shape=(env_n,) + real_state_shape,
                dtype=np.uint8,
            ),
            "xs": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(env_n,) + real_state_shape,
                dtype=np.float32,
            ),
            "hs": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(env_n,) + hidden_shape,
                dtype=np.float32,
            ),
        })
        actor_action_space = spaces.Tuple((
            spaces.Tuple(tuple(
                spaces.Discrete(action_n) for _ in range(env_n)
            )),
            spaces.Tuple(tuple(spaces.Discrete(3) for _ in range(env_n))),
        ))
        actor_net = ActorNet(
            obs_space=actor_observation_space,
            action_space=actor_action_space,
            flags=flags,
            tree_rep_meaning=get_tree_rep_meaning(action_n, 1, flags),
        ).cpu()
        expected_counts = {
            "actor state": (len(actor_net.state_dict()), 119),
            "actor parameters": (len(tuple(actor_net.parameters())), 117),
            "model state": (len(model_net.state_dict()), 468),
            "model p parameters": (
                len(tuple(model_net.vp_net.parameters())), 103
            ),
            "model m parameters": (
                len(tuple(model_net.sr_net.parameters())), 147
            ),
        }
        for count_label, (actual, expected) in expected_counts.items():
            if actual != expected:
                raise ValueError(
                    f"{label} reconstructed Enduro {count_label} count "
                    f"changed: expected {expected}, got {actual}"
                )
    finally:
        _schema6_restore_rng_state(rng_state)
    return actor_net, model_net


def _validate_network_state_against_template(state, template, *, label):
    """Require an exact key/shape/dtype architecture match and strict load."""

    if not isinstance(state, collections.abc.Mapping):
        raise ValueError(f"{label} must be a state mapping")
    expected = template.state_dict()
    if set(state) != set(expected):
        missing = tuple(key for key in expected if key not in state)
        extra = tuple(key for key in state if key not in expected)
        raise ValueError(
            f"{label} architecture keys disagree; "
            f"missing={missing!r}, extra={extra!r}"
        )
    for key, expected_tensor in expected.items():
        tensor = state[key]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{label} {key!r} is not a tensor")
        if tensor.dtype != expected_tensor.dtype:
            raise ValueError(
                f"{label} {key!r} dtype disagrees with architecture"
            )
        if tuple(tensor.shape) != tuple(expected_tensor.shape):
            raise ValueError(
                f"{label} {key!r} shape disagrees with architecture"
            )
    try:
        template.load_state_dict(state, strict=True)
    except (RuntimeError, ValueError, TypeError) as exc:
        raise ValueError(f"{label} cannot strict-load architecture") from exc
    return len(expected)


def _validate_recursive_finite_training_state(value, *, label):
    """Reject missing-object tricks and non-finite persisted training state."""

    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided:
            raise ValueError(f"{label} tensor must use strided layout")
        if (torch.is_floating_point(value) or torch.is_complex(value)) and not (
            torch.isfinite(value).all()
        ):
            raise ValueError(f"{label} contains non-finite tensor values")
        return
    if isinstance(value, np.ndarray):
        if (
            np.issubdtype(value.dtype, np.floating)
            or np.issubdtype(value.dtype, np.complexfloating)
        ) and not np.isfinite(value).all():
            raise ValueError(f"{label} contains non-finite array values")
        return
    if isinstance(value, collections.abc.Mapping):
        for key, item in value.items():
            _validate_recursive_finite_training_state(
                item, label=f"{label}.{key}"
            )
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_recursive_finite_training_state(
                item, label=f"{label}[{index}]"
            )
        return
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise ValueError(f"{label} contains a non-finite scalar")
    if value is None or isinstance(
        value, (str, bytes, bool, np.bool_, int, np.integer, float, np.number)
    ):
        return
    raise ValueError(f"{label} contains unsupported state {type(value).__name__}")


def _validate_optimizer_checkpoint_state(
    value,
    *,
    expected_parameters,
    expected_step,
    initial_lr,
    current_lr,
    label,
):
    if (
        isinstance(expected_step, (bool, np.bool_))
        or not isinstance(expected_step, (int, np.integer))
        or int(expected_step) < 0
    ):
        raise ValueError(f"{label} expected step must be a nonnegative integer")
    expected_step = int(expected_step)
    if not isinstance(value, collections.abc.Mapping) or set(value) != {
        "state", "param_groups",
    }:
        raise ValueError(f"{label} must be an exact optimizer state mapping")
    state = value["state"]
    groups = value["param_groups"]
    if not isinstance(expected_parameters, (list, tuple)) or not expected_parameters:
        raise ValueError(f"{label} expected parameter manifest is empty")
    if not isinstance(groups, (list, tuple)) or len(groups) != 1:
        raise ValueError(f"{label} must have exactly one param_group")
    group = groups[0]
    expected_group_keys = {
        "lr", "betas", "eps", "weight_decay", "amsgrad", "maximize",
        "foreach", "capturable", "differentiable", "fused",
        "decoupled_weight_decay", "initial_lr", "params",
    }
    if not isinstance(group, collections.abc.Mapping) or set(group) != expected_group_keys:
        raise ValueError(f"{label} Adam param_group is incomplete")
    params = group["params"]
    if not isinstance(params, (list, tuple)) or len(params) != len(
        expected_parameters
    ):
        raise ValueError(f"{label} parameter coverage is incomplete")
    if any(
        isinstance(parameter_id, (bool, np.bool_))
        or not isinstance(parameter_id, (int, np.integer))
        for parameter_id in params
    ) or len(set(int(parameter_id) for parameter_id in params)) != len(params):
        raise ValueError(f"{label} parameter ids must be unique integers")
    params = [int(parameter_id) for parameter_id in params]
    if params != list(range(len(expected_parameters))):
        raise ValueError(
            f"{label} parameter ids/order must equal range(N)"
        )
    if (
        not isinstance(state, collections.abc.Mapping)
        or any(
            isinstance(parameter_id, (bool, np.bool_))
            or not isinstance(parameter_id, (int, np.integer))
            for parameter_id in state
        )
        or {int(parameter_id) for parameter_id in state} != set(params)
    ):
        raise ValueError(f"{label} optimizer state must exactly cover parameters")
    storage_ptrs = set()
    for name, expected in (
        ("lr", current_lr),
        ("initial_lr", initial_lr),
        ("eps", 1e-8),
        ("weight_decay", 0.0),
    ):
        item = group[name]
        if (
            isinstance(item, (bool, np.bool_))
            or not isinstance(item, (int, float, np.number))
            or not np.isfinite(item)
            or float(item) != float(expected)
        ):
            raise ValueError(f"{label} Adam {name} disagrees with protocol")
    betas = group["betas"]
    if (
        not isinstance(betas, (list, tuple))
        or len(betas) != 2
        or tuple(float(item) for item in betas) != (0.9, 0.999)
    ):
        raise ValueError(f"{label} Adam betas disagree with protocol")
    for name in ("amsgrad", "maximize", "capturable", "differentiable"):
        if type(group[name]) is not bool or group[name]:
            raise ValueError(f"{label} Adam {name} must be false")
    if group["foreach"] is not None or group["fused"] is not None:
        raise ValueError(f"{label} Adam foreach/fused must be null")
    if group["decoupled_weight_decay"] is not False:
        raise ValueError(f"{label} Adam decoupled_weight_decay must be false")
    for parameter_id, (parameter_name, parameter) in zip(
        params, expected_parameters
    ):
        parameter_state = state[parameter_id]
        if not isinstance(parameter_state, collections.abc.Mapping) or set(
            parameter_state
        ) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError(
                f"{label} Adam state is incomplete for {parameter_name}"
            )
        step = parameter_state["step"]
        if not isinstance(step, torch.Tensor):
            raise ValueError(
                f"{label} Adam step must be a tensor for {parameter_name}"
            )
        if not torch.isfinite(step).all():
            raise ValueError(
                f"{label} Adam step is non-finite for {parameter_name}"
            )
        if (
            step.layout != torch.strided
            or step.device.type != "cpu"
            or step.requires_grad
            or not step.is_contiguous()
            or step.dtype != torch.float32
            or tuple(step.shape) != ()
            or float(step.item()) != float(expected_step)
        ):
            raise ValueError(
                f"{label} Adam step disagrees for {parameter_name}"
            )
        for moment_name in ("exp_avg", "exp_avg_sq"):
            moment = parameter_state[moment_name]
            if isinstance(moment, torch.Tensor) and not torch.isfinite(
                moment
            ).all():
                raise ValueError(
                    f"{label} Adam {moment_name} is non-finite for "
                    f"{parameter_name}"
                )
            if (
                not isinstance(moment, torch.Tensor)
                or moment.layout != torch.strided
                or moment.device.type != "cpu"
                or moment.requires_grad
                or not moment.is_contiguous()
                or moment.dtype != parameter.dtype
                or tuple(moment.shape) != tuple(parameter.shape)
            ):
                raise ValueError(
                    f"{label} Adam {moment_name} disagrees with "
                    f"{parameter_name}"
                )
            if moment.numel():
                storage_ptr = moment.untyped_storage().data_ptr()
                if storage_ptr in storage_ptrs:
                    raise ValueError(f"{label} Adam moment tensors alias storage")
                storage_ptrs.add(storage_ptr)
    _validate_recursive_finite_training_state(value, label=label)
    return {
        "state_entry_count": len(state),
        "param_group_count": 1,
        "expected_step": int(expected_step),
    }


def _validate_scheduler_checkpoint_state(
    value, *, expected_step, real_step, initial_lr, current_lr, label
):
    for name, item in (("expected_step", expected_step), ("real_step", real_step)):
        if (
            isinstance(item, (bool, np.bool_))
            or not isinstance(item, (int, np.integer))
            or int(item) < 0
        ):
            raise ValueError(f"{label} {name} must be a nonnegative integer")
    expected_step = int(expected_step)
    real_step = int(real_step)
    required = {
        "base_lrs", "last_epoch", "_step_count", "_is_initial",
        "_get_lr_called_within_step", "_last_lr", "lr_lambdas",
    }
    if not isinstance(value, collections.abc.Mapping) or set(value) != required:
        raise ValueError(f"{label} scheduler state is incomplete")
    for name, expected in (
        ("last_epoch", real_step),
        ("_step_count", expected_step + 1),
    ):
        item = value[name]
        if (
            isinstance(item, (bool, np.bool_))
            or not isinstance(item, (int, np.integer))
            or int(item) != int(expected)
        ):
            raise ValueError(f"{label} has stale or invalid {name}")
    for name in ("_is_initial", "_get_lr_called_within_step"):
        if type(value[name]) is not bool or value[name]:
            raise ValueError(f"{label} has invalid {name}")
    if value["lr_lambdas"] != [None]:
        raise ValueError(f"{label} has invalid lr_lambdas")
    for name, expected in (
        ("base_lrs", initial_lr),
        ("_last_lr", current_lr),
    ):
        items = value[name]
        if (
            not isinstance(items, (list, tuple))
            or len(items) != 1
            or isinstance(items[0], (bool, np.bool_))
            or not isinstance(items[0], (int, float, np.number))
            or not np.isfinite(items[0])
            or float(items[0]) != float(expected)
        ):
            raise ValueError(f"{label} {name} disagrees with schedule")
    _validate_recursive_finite_training_state(value, label=label)
    return {
        "last_epoch": int(value["last_epoch"]),
        "step_count": int(value["_step_count"]),
    }


def _validate_grad_scaler_checkpoint_state(value, *, label):
    required = {
        "scale",
        "growth_factor",
        "backoff_factor",
        "growth_interval",
        "_growth_tracker",
    }
    if not isinstance(value, collections.abc.Mapping) or set(value) != required:
        raise ValueError(f"{label} has invalid GradScaler fields")
    _validate_recursive_finite_training_state(value, label=label)
    scale = value["scale"]
    growth_factor = value["growth_factor"]
    backoff_factor = value["backoff_factor"]
    growth_interval = value["growth_interval"]
    growth_tracker = value["_growth_tracker"]
    if any(
        isinstance(item, (bool, np.bool_))
        or not isinstance(item, (int, float, np.number))
        or not np.isfinite(item)
        for item in (scale, growth_factor, backoff_factor)
    ) or (
        isinstance(growth_interval, (bool, np.bool_))
        or not isinstance(growth_interval, (int, np.integer))
        or int(growth_interval) <= 0
        or isinstance(growth_tracker, (bool, np.bool_))
        or not isinstance(growth_tracker, (int, np.integer))
        or not 0 <= int(growth_tracker) < int(growth_interval)
    ):
        raise ValueError(f"{label} has invalid GradScaler values")
    if (
        float(scale) <= 0.0
        or float(growth_factor) != 2.0
        or float(backoff_factor) != 0.5
        or int(growth_interval) != 2000
    ):
        raise ValueError(f"{label} disagrees with GradScaler construction")
    return {
        "scale": float(scale),
        "growth_tracker": int(growth_tracker),
    }


_SCHEMA6_IDENTITY_BOOL_FIELDS = frozenset({
    "float16",
    "model_float16",
    "dual_net",
    "train_actor",
    "parallel_actor",
    "train_model",
    "use_wandb",
    "actor_use_rms",
    "voc_eval_stochastic",
    "voc_dueling_q",
    "voc_expected_gate_loss",
    "voc_ema_gate_target",
    "voc_dedicated_gate",
    "voc_soft_q_bce_gate",
    "voc_gate_confidence_weighted",
    "voc_gate_param_align",
    "voc_gate_exact_projection",
    "voc_gate_epsilon_greedy_execution",
    "voc_actor_policy_version_barrier",
    "envpool",
    "grayscale",
    "dynamic_search",
    "dynamic_factorized_control",
    "has_action_seq",
    "return_h",
    "return_x",
    "model_disable_bn",
})
_SCHEMA6_IDENTITY_INT_FIELDS = frozenset({
    "voc_gate_policy_schema_version",
    "voc_model_input_seal_schema_version",
    "voc_actor_policy_bundle_schema_version",
    "voc_actor_policy_ray_max_restarts",
    "voc_actor_policy_ray_max_task_retries",
    "ppo_k",
    "self_play_n",
    "env_n",
    "actor_batch_size",
    "actor_unroll_len",
    "total_steps",
    "schedule_total_steps",
    "base_seed",
    "icopro_game_id",
    "icopro_supervised_freq",
    "frame_stack_n",
    "wrapper_type",
    "max_search_steps",
    "max_depth",
    "rec_t",
    "model_size_nn",
    "model_decoder_depth",
})
_SCHEMA6_IDENTITY_FLOAT_FIELDS = frozenset({
    "voc_loss_cost",
    "voc_gate_temperature",
    "voc_train_epsilon",
    "voc_gate_target_tau",
    "voc_gate_q_temperature",
    "voc_gate_adam_beta1",
    "voc_gate_param_align_coef",
    "voc_gate_execution_epsilon",
    "voc_actor_policy_barrier_timeout_s",
    "actor_amp_init_scale",
    "voc_gate_learning_rate",
    "voc_gate_grad_norm_clipping",
    "entropy_r_cost",
    "actor_learning_rate",
    "actor_adam_eps",
    "model_learning_rate",
    "model_state_range_loss_cost",
})
_SCHEMA6_IDENTITY_STRING_FIELDS = frozenset({
    "dynamic_voc_mode",
    "model_optimizer",
    "name",
    "xpid",
    "model_state_projection",
})


def _validate_schema6_identity_value(name, value, *, label):
    """Return a typed canonical identity value without float coercion."""

    if name in _SCHEMA6_IDENTITY_BOOL_FIELDS:
        if type(value) is not bool:
            raise ValueError(f"{label} {name} must be a Python bool")
        return value
    if name in _SCHEMA6_IDENTITY_INT_FIELDS:
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
        ):
            raise ValueError(f"{label} {name} must be a non-bool integer")
        return int(value)
    if name in _SCHEMA6_IDENTITY_FLOAT_FIELDS:
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.number))
            or not np.isfinite(value)
        ):
            raise ValueError(f"{label} {name} must be finite numeric")
        return float(value)
    if name in _SCHEMA6_IDENTITY_STRING_FIELDS:
        if type(value) is not str:
            raise ValueError(f"{label} {name} must be a string")
        return value
    raise ValueError(f"{label} has unclassified identity field {name}")


_SCHEMA7_MODEL_INPUT_SEAL_EVIDENCE_FIELDS = (
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
)


def _validate_schema7_model_input_seal_evidence(
    model_checkpoint,
    config,
    *,
    model_p_steps,
    model_m_steps,
    label,
):
    """Validate the exact schema-1 terminal model-input seal attestation."""

    expected_fields = set(_SCHEMA7_MODEL_INPUT_SEAL_EVIDENCE_FIELDS)
    actual_fields = {
        name
        for name in model_checkpoint
        if type(name) is str and name.startswith("voc_model_")
    }
    if actual_fields != expected_fields:
        missing = sorted(expected_fields - actual_fields)
        extra = sorted(actual_fields - expected_fields)
        raise ValueError(
            f"{label} requires exact schema-7 model-input seal evidence; "
            f"missing={missing!r}, extra={extra!r}"
        )
    sealed = model_checkpoint["voc_model_input_sealed"]
    if type(sealed) is not bool or sealed is not True:
        raise ValueError(
            f"{label} voc_model_input_sealed must be Python bool True"
        )
    integer_names = tuple(
        name
        for name in _SCHEMA7_MODEL_INPUT_SEAL_EVIDENCE_FIELDS
        if name != "voc_model_input_sealed"
    )
    values = {}
    for name in integer_names:
        value = model_checkpoint[name]
        if type(value) is not int:
            raise ValueError(
                f"{label} {name} must be a Python non-bool integer"
            )
        values[name] = value
    if (
        values["voc_model_input_seal_schema_version"]
        != VOC_MODEL_INPUT_SEAL_SCHEMA_VERSION
        or values["voc_model_input_seal_schema_version"]
        != config["voc_model_input_seal_schema_version"]
    ):
        raise ValueError(
            f"{label} requires model-input seal schema version 1"
        )
    if values["voc_model_input_seal_count"] != 1:
        raise ValueError(f"{label} requires exactly one model-input seal")
    terminal_processed_n = values["voc_model_terminal_processed_n"]
    model_real_step = int(model_checkpoint["real_step"])
    if (
        terminal_processed_n <= 0
        or terminal_processed_n < int(config["total_steps"])
        or terminal_processed_n != model_real_step
    ):
        raise ValueError(
            f"{label} terminal processed count must equal final model "
            "real_step at or beyond total_steps"
        )
    drain_count = values["voc_model_terminal_drain_update_count"]
    if drain_count not in (0, 1):
        raise ValueError(
            f"{label} terminal drain update count must be exactly 0 or 1"
        )
    pre_real_step = values["voc_model_terminal_drain_pre_real_step"]
    pre_m_steps = values[
        "voc_model_terminal_drain_pre_grad_step_count_m"
    ]
    pre_p_steps = values[
        "voc_model_terminal_drain_pre_grad_step_count_p"
    ]
    if not 0 <= pre_real_step <= terminal_processed_n:
        raise ValueError(f"{label} terminal drain pre-real-step is invalid")
    if pre_m_steps < 0 or pre_p_steps < 0:
        raise ValueError(
            f"{label} terminal drain pre-update counts must be non-negative"
        )
    if (
        model_m_steps != pre_m_steps + drain_count
        or model_p_steps != pre_p_steps + drain_count
    ):
        raise ValueError(
            f"{label} final model update counts disagree with terminal drain"
        )
    if drain_count == 0 and pre_real_step != terminal_processed_n:
        raise ValueError(
            f"{label} zero-drain branch requires pre-real-step to be final"
        )
    if drain_count == 1 and pre_real_step >= terminal_processed_n:
        raise ValueError(
            f"{label} one-drain branch requires strict real-step progress"
        )
    if values["voc_model_input_late_write_count"] != 0:
        raise ValueError(f"{label} requires zero late model-input writes")
    if values["voc_model_input_abort_count"] != 0:
        raise ValueError(f"{label} requires zero model-input aborts")
    return {
        "voc_model_input_seal_schema_version": (
            VOC_MODEL_INPUT_SEAL_SCHEMA_VERSION
        ),
        "voc_model_input_sealed": True,
        "voc_model_input_seal_count": 1,
        "voc_model_terminal_processed_n": terminal_processed_n,
        "voc_model_terminal_drain_update_count": drain_count,
        "voc_model_terminal_drain_pre_real_step": pre_real_step,
        "voc_model_terminal_drain_pre_grad_step_count_m": pre_m_steps,
        "voc_model_terminal_drain_pre_grad_step_count_p": pre_p_steps,
        "voc_model_input_late_write_count": 0,
        "voc_model_input_abort_count": 0,
    }


def validate_schema6_final_bundle(ckpdir, *, label="schema-6 final bundle"):
    """Race-check schemas 6--13 before private/public finish."""

    root = os.path.abspath(ckpdir)
    config_path = os.path.join(root, "config_c.yaml")
    actor_path = os.path.join(root, "ckp_actor.tar")
    model_path = os.path.join(root, "ckp_model.tar")
    schema13_intent = _schema13_stage_xpid_candidate(os.path.basename(root))
    if schema13_intent:
        before, bound_payloads = _collect_run_completion_snapshot_raw(
            root, gate_schema=VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION
        )
        try:
            config = yaml.safe_load(
                bound_payloads["config_c.yaml"].decode("utf-8")
            )
        except (UnicodeDecodeError, yaml.YAMLError) as error:
            raise ValueError(
                f"{label} config_c.yaml must be strict UTF-8 YAML"
            ) from error
        if (
            not isinstance(config, collections.abc.Mapping)
            or config.get("voc_gate_policy_schema_version")
            != VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION
        ):
            raise RuntimeError(
                f"{label} schema changed during completion binding"
            )
        gate_schema = VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION
    else:
        # Preserve schemas<=12's inherited pathname and exception behavior.
        before, _ = _collect_run_completion_snapshot_raw(root)
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        if not isinstance(config, collections.abc.Mapping):
            raise ValueError(f"{label} config_c.yaml must contain a mapping")
        gate_schema = config.get("voc_gate_policy_schema_version")
        if gate_schema == VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION:
            raise ValueError(
                f"{label} schema 13 requires an exact lexical V20 run directory"
            )
    if gate_schema == VOC_GATE_POLICY_VERSION_BARRIER_SCHEMA_VERSION:
        surface_validator = _validate_schema6_complete_surface
        protocol_validator = _validate_schema6_protocol_flags
        actor_state_validator = validate_voc_schema6_final_actor_checkpoint
    elif gate_schema == VOC_GATE_POLICY_MODEL_INPUT_SEAL_SCHEMA_VERSION:
        surface_validator = _validate_schema7_complete_surface
        protocol_validator = _validate_schema7_protocol_flags
        actor_state_validator = validate_voc_schema7_final_actor_checkpoint
    elif gate_schema == VOC_GATE_POLICY_HALF_SQUARED_Q_SCHEMA_VERSION:
        surface_validator = _validate_schema8_complete_surface
        protocol_validator = _validate_schema8_protocol_flags
        actor_state_validator = validate_voc_schema8_final_actor_checkpoint
    elif gate_schema == VOC_GATE_POLICY_COMMON_MODE_Q_SCHEMA_VERSION:
        surface_validator = _validate_schema9_complete_surface
        protocol_validator = _validate_schema9_protocol_flags
        actor_state_validator = validate_voc_schema9_final_actor_checkpoint
    elif gate_schema == VOC_GATE_POLICY_HUBER_COMMON_Q_SCHEMA_VERSION:
        surface_validator = _validate_schema10_complete_surface
        protocol_validator = _validate_schema10_protocol_flags
        actor_state_validator = validate_voc_schema10_final_actor_checkpoint
    elif gate_schema == VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION:
        surface_validator = _validate_schema11_complete_surface
        protocol_validator = _validate_schema11_protocol_flags
        actor_state_validator = validate_voc_schema11_final_actor_checkpoint
    elif gate_schema == VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION:
        surface_validator = _validate_schema12_complete_surface
        protocol_validator = _validate_schema12_protocol_flags
        actor_state_validator = validate_voc_schema12_final_actor_checkpoint
    elif gate_schema == VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION:
        surface_validator = _validate_schema13_complete_surface
        protocol_validator = _validate_schema13_protocol_flags
        actor_state_validator = validate_voc_schema13_final_actor_checkpoint
    else:
        raise ValueError(
            f"{label} requires exact atomic gate-policy schema 6--13"
        )
    if gate_schema == VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION:
        actor_checkpoint = torch.load(
            io.BytesIO(bound_payloads["ckp_actor.tar"]),
            map_location=torch.device("cpu"),
            weights_only=False,
        )
        model_checkpoint = torch.load(
            io.BytesIO(bound_payloads["ckp_model.tar"]),
            map_location=torch.device("cpu"),
            weights_only=False,
        )
    else:
        actor_checkpoint = torch.load(
            actor_path, map_location=torch.device("cpu"), weights_only=False
        )
        model_checkpoint = torch.load(
            model_path, map_location=torch.device("cpu"), weights_only=False
        )
    if gate_schema == VOC_GATE_POLICY_HUBER_COMMON_Q_SCHEMA_VERSION:
        _reject_schema10_persisted_derived_identity(
            actor_checkpoint, label=f"{label} actor checkpoint"
        )
        _reject_schema10_persisted_derived_identity(
            model_checkpoint, label=f"{label} model checkpoint"
        )
    elif gate_schema == VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION:
        _reject_schema11_persisted_derived_identity(
            actor_checkpoint, label=f"{label} actor checkpoint"
        )
        _reject_schema11_persisted_derived_identity(
            model_checkpoint, label=f"{label} model checkpoint"
        )
    elif gate_schema == VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION:
        _reject_schema12_persisted_derived_identity(
            actor_checkpoint, label=f"{label} actor checkpoint"
        )
        _reject_schema12_persisted_derived_identity(
            model_checkpoint, label=f"{label} model checkpoint"
        )
    elif gate_schema == VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION:
        _reject_schema13_persisted_derived_identity(
            actor_checkpoint, label=f"{label} actor checkpoint"
        )
        _reject_schema13_persisted_derived_identity(
            model_checkpoint, label=f"{label} model checkpoint"
        )
    elif not isinstance(model_checkpoint, collections.abc.Mapping):
        raise ValueError(f"{label} model checkpoint must be a mapping")
    actor_evidence = validate_actor_policy_checkpoint(
        actor_checkpoint, label=f"{label} actor checkpoint"
    )
    actor_flags = actor_checkpoint["flags"]
    model_flags = model_checkpoint.get("flags")
    config_surface = surface_validator(
        config,
        label=f"{label} config",
        expected_ckpdir=root,
    )
    actor_surface = surface_validator(
        actor_flags,
        label=f"{label} actor embedded flags",
        expected_ckpdir=root,
    )
    model_surface = surface_validator(
        model_flags,
        label=f"{label} model embedded flags",
        expected_ckpdir=root,
    )
    protocol_validator(
        config, label=f"{label} config", expected_ckpdir=root
    )
    protocol_validator(
        actor_flags,
        label=f"{label} actor embedded flags",
        expected_ckpdir=root,
    )
    protocol_validator(
        model_flags,
        label=f"{label} model embedded flags",
        expected_ckpdir=root,
    )
    identity_field_set = (
        set(VOC_ACTIVE_ONLY_PROTOCOL_FIELDS)
        | set(VOC_GATE_POLICY_SCHEMA6_ATOMIC_REQUIREMENTS)
        | set(VOC_GATE_POLICY_SCHEMA6_OPTIMIZER_REQUIREMENTS)
        | set(VOC_GATE_POLICY_SCHEMA6_ENDURO_REQUIREMENTS)
        | {
        "dynamic_voc_mode",
        "voc_gate_policy_schema_version",
        "voc_loss_cost",
        "voc_gate_temperature",
        "voc_train_epsilon",
        "voc_eval_stochastic",
        "voc_dueling_q",
        "voc_expected_gate_loss",
        "float16",
        "model_float16",
        "dual_net",
        "model_optimizer",
        "actor_use_rms",
        "actor_learning_rate",
        "actor_adam_eps",
        "model_learning_rate",
        "actor_amp_init_scale",
        "ppo_k",
        "self_play_n",
        "env_n",
        "actor_batch_size",
        "actor_unroll_len",
        "train_actor",
        "parallel_actor",
        "train_model",
        "total_steps",
        "schedule_total_steps",
        "use_wandb",
        "base_seed",
        "name",
        "xpid",
    })
    if gate_schema == VOC_GATE_POLICY_VERSION_BARRIER_SCHEMA_VERSION:
        identity_field_set.discard("voc_model_input_seal_schema_version")
    identity_fields = tuple(sorted(identity_field_set))
    for name in identity_fields:
        if name not in config or name not in actor_flags or name not in model_flags:
            raise ValueError(f"{label} lacks cross-surface identity {name}")
        canonical_values = tuple(
            _validate_schema6_identity_value(
                name,
                surface[name],
                label=f"{label} {surface_name}",
            )
            for surface_name, surface in (
                ("config", config),
                ("actor embedded flags", actor_flags),
                ("model embedded flags", model_flags),
            )
        )
        if canonical_values[0] != canonical_values[1] or (
            canonical_values[0] != canonical_values[2]
        ):
            raise ValueError(f"{label} cross-surface identity disagrees on {name}")
    if not (
        config_surface["canonical_json"]
        == actor_surface["canonical_json"]
        == model_surface["canonical_json"]
    ):
        raise ValueError(
            f"{label} config/actor/model complete schema-{gate_schema} "
            "surfaces differ"
        )
    if config["train_model"] is not True:
        raise ValueError(f"{label} requires train_model=true")
    active_actor = actor_state_validator(
        actor_checkpoint,
        argparse.Namespace(**dict(config)),
        label=f"{label} actor active state",
    )
    policy_update_count = int(actor_evidence["voc_actor_policy_version"])
    for counter_name in (
        "voc_update_count",
        "voc_ema_gate_update_count",
        "voc_gate_update_count",
    ):
        if int(active_actor[counter_name]) != policy_update_count:
            raise ValueError(
                f"{label} actor policy version and {counter_name} must be "
                "lockstep"
            )
    for counter_name in (
        "imitation_update_count",
        "imitation_schedule_step",
    ):
        value = actor_checkpoint.get(counter_name)
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) != policy_update_count
        ):
            raise ValueError(
                f"{label} requires {counter_name} to equal the actor "
                "policy version"
            )
    actor_template, model_template = _reconstruct_schema6_enduro_networks(
        config, label=f"{label} architecture"
    )
    actor_state_count = _validate_network_state_against_template(
        actor_checkpoint.get("actor_net_state_dict"),
        actor_template,
        label=f"{label} actor state",
    )
    excluded_parameter_ids = {
        id(parameter)
        for module_name in ("voc_head", "voc_gate_head")
        for parameter in getattr(actor_template, module_name).parameters()
    }
    actor_parameters = [
        (name, parameter)
        for name, parameter in actor_template.named_parameters()
        if id(parameter) not in excluded_parameter_ids
    ]
    actor_real_step = int(actor_checkpoint["real_step"])
    actor_update_count = policy_update_count
    schedule_total_steps = int(config["schedule_total_steps"])
    actor_initial_lr = float(config["actor_learning_rate"])
    actor_current_lr = actor_initial_lr * (
        1.0 - actor_real_step / schedule_total_steps
    )
    actor_optimizer = _validate_optimizer_checkpoint_state(
        actor_checkpoint.get("actor_net_optimizer_state_dict"),
        expected_parameters=actor_parameters,
        expected_step=actor_update_count,
        initial_lr=actor_initial_lr,
        current_lr=actor_current_lr,
        label=f"{label} main actor optimizer",
    )
    actor_scheduler = _validate_scheduler_checkpoint_state(
        actor_checkpoint.get("actor_net_scheduler_state_dict"),
        expected_step=actor_update_count,
        real_step=actor_real_step,
        initial_lr=actor_initial_lr,
        current_lr=actor_current_lr,
        label=f"{label} main actor scheduler",
    )
    for name in ("step", "real_step"):
        value = model_checkpoint.get(name)
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) <= 0
        ):
            raise ValueError(f"{label} model {name} is invalid")
    if int(model_checkpoint["real_step"]) < int(config["total_steps"]):
        raise ValueError(f"{label} model checkpoint is stale at finalization")
    model_tensor_count = _validate_final_model_state(
        model_checkpoint.get("model_net_state_dict"),
        label=f"{label} model state",
    )
    _validate_network_state_against_template(
        model_checkpoint.get("model_net_state_dict"),
        model_template,
        label=f"{label} model state",
    )
    model_optimizers = {}
    model_schedulers = {}
    model_scalers = {}
    components = ("p", "m") if config["dual_net"] else ("p",)
    model_real_step = int(model_checkpoint["real_step"])
    model_initial_lr = float(config["model_learning_rate"])
    model_current_lr = model_initial_lr * (
        1.0 - model_real_step / schedule_total_steps
    )
    for component in components:
        component_net = (
            model_template.vp_net if component == "p" else model_template.sr_net
        )
        component_parameters = [
            (f"{component_net.__class__.__name__}.{name}", parameter)
            for name, parameter in component_net.named_parameters()
        ]
        component_update_count = model_checkpoint.get(
            f"model_grad_step_count_{component}"
        )
        if (
            isinstance(component_update_count, (bool, np.bool_))
            or not isinstance(component_update_count, (int, np.integer))
            or int(component_update_count) < 0
        ):
            raise ValueError(
                f"{label} has invalid model update count for {component}"
            )
        component_update_count = int(component_update_count)
        model_optimizers[component] = _validate_optimizer_checkpoint_state(
            model_checkpoint.get(
                f"model_net_optimizer_{component}_state_dict"
            ),
            expected_parameters=component_parameters,
            expected_step=component_update_count,
            initial_lr=model_initial_lr,
            current_lr=model_current_lr,
            label=f"{label} model optimizer {component}",
        )
        model_schedulers[component] = _validate_scheduler_checkpoint_state(
            model_checkpoint.get(
                f"model_net_scheduler_{component}_state_dict"
            ),
            expected_step=component_update_count,
            real_step=model_real_step,
            initial_lr=model_initial_lr,
            current_lr=model_current_lr,
            label=f"{label} model scheduler {component}",
        )
        scaler_key = f"model_scaler_{component}_state_dict"
        if config["model_float16"]:
            model_scalers[component] = _validate_grad_scaler_checkpoint_state(
                model_checkpoint.get(scaler_key),
                label=f"{label} model GradScaler {component}",
            )
        elif scaler_key in model_checkpoint:
            raise ValueError(
                f"{label} FP32 model must not persist {scaler_key}"
            )
    if not config["dual_net"]:
        for suffix in (
            "model_net_optimizer_m_state_dict",
            "model_net_scheduler_m_state_dict",
            "model_scaler_m_state_dict",
        ):
            if suffix in model_checkpoint:
                raise ValueError(
                    f"{label} single-network model unexpectedly stores {suffix}"
                )
    for name in (
        "model_grad_clip_count_m",
        "model_grad_step_count_m",
        "model_grad_clip_count_p",
        "model_grad_step_count_p",
    ):
        value = model_checkpoint.get(name)
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < 0
        ):
            raise ValueError(f"{label} has invalid model counter {name}")
    if (
        int(model_checkpoint["model_grad_clip_count_p"])
        > int(model_checkpoint["model_grad_step_count_p"])
        or int(model_checkpoint["model_grad_clip_count_m"])
        > int(model_checkpoint["model_grad_step_count_m"])
    ):
        raise ValueError(f"{label} model gradient-clip counters disagree")
    model_p_steps = int(model_checkpoint["model_grad_step_count_p"])
    model_m_steps = int(model_checkpoint["model_grad_step_count_m"])
    if model_p_steps <= 0 or model_m_steps <= 0 or model_p_steps != model_m_steps:
        raise ValueError(
            f"{label} dual ModelNet gradient-step counters must be positive "
            "and lockstep"
        )
    model_input_seal = None
    if gate_schema in (
        VOC_GATE_POLICY_MODEL_INPUT_SEAL_SCHEMA_VERSION,
        VOC_GATE_POLICY_HALF_SQUARED_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_COMMON_MODE_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_HUBER_COMMON_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION,
        VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION,
    ):
        model_input_seal = _validate_schema7_model_input_seal_evidence(
            model_checkpoint,
            config,
            model_p_steps=model_p_steps,
            model_m_steps=model_m_steps,
            label=f"{label} model checkpoint",
        )
    after = _collect_run_completion_evidence_raw(
        root,
        gate_schema=(
            VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION
            if gate_schema == VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION
            else None
        ),
    )
    if after != before:
        raise RuntimeError(f"{label} files changed during validation")
    return {
        "actor_policy": actor_evidence,
        "resolved_identity": (
            {
                "key_count": config_surface["key_count"],
                "v12_projection_key_count": config_surface[
                    "v12_projection_key_count"
                ],
                "v12_projection_sha256": config_surface[
                    "v12_projection_sha256"
                ],
                "complete_surface_sha256": config_surface[
                    "complete_surface_sha256"
                ],
                "stage": tuple(config_surface["stage"]),
                "paths": copy.deepcopy(config_surface["paths"]),
                "gate_schema": gate_schema,
                "voc_gate_policy_schema_version": gate_schema,
                "voc_model_input_seal_schema_version": (
                    VOC_MODEL_INPUT_SEAL_SCHEMA_VERSION
                ),
                "voc_q_regression_loss": (
                    "smooth_l1_beta1"
                    if gate_schema in (
                        VOC_GATE_POLICY_HUBER_COMMON_Q_SCHEMA_VERSION,
                        VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION,
                        VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION,
                        VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION,
                    )
                    else "half_squared_td"
                ),
                "voc_q_reconstruction": (
                    "detached_value_plus_raw_head_mean_plus_"
                    "policy_centered_raw_head"
                ),
                **(
                    {
                        "voc_q_optimizer_coordinates": (
                            "orthonormal_common_difference_adam"
                        )
                    }
                    if gate_schema
                    in (
                        VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION,
                        VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION,
                        VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION,
                    )
                    else {}
                ),
            }
            if gate_schema in (
                VOC_GATE_POLICY_COMMON_MODE_Q_SCHEMA_VERSION,
                VOC_GATE_POLICY_HUBER_COMMON_Q_SCHEMA_VERSION,
                VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION,
                VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION,
                VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION,
            )
            else {
                **(
                    {
                        "gate_schema": gate_schema,
                        "voc_gate_policy_schema_version": gate_schema,
                        "voc_model_input_seal_schema_version": (
                            VOC_MODEL_INPUT_SEAL_SCHEMA_VERSION
                        )
                    }
                    if gate_schema in (
                        VOC_GATE_POLICY_MODEL_INPUT_SEAL_SCHEMA_VERSION,
                        VOC_GATE_POLICY_HALF_SQUARED_Q_SCHEMA_VERSION,
                    )
                    else {}
                ),
                **(
                    {"voc_q_regression_loss": "half_squared_td"}
                    if gate_schema
                    == VOC_GATE_POLICY_HALF_SQUARED_Q_SCHEMA_VERSION
                    else {}
                ),
                "key_count": config_surface["key_count"],
                "v12_projection_key_count": config_surface[
                    "v12_projection_key_count"
                ],
                "v12_projection_sha256": config_surface[
                    "v12_projection_sha256"
                ],
                "complete_surface_sha256": config_surface[
                    "complete_surface_sha256"
                ],
                "stage": tuple(config_surface["stage"]),
                "paths": copy.deepcopy(config_surface["paths"]),
            }
        ),
        "actor_training_state": {
            "actor_state_tensor_count": actor_state_count,
            "actor_parameter_count": len(actor_parameters),
            "voc_update_count": active_actor["voc_update_count"],
            "voc_ema_gate_update_count": active_actor[
                "voc_ema_gate_update_count"
            ],
            "voc_gate_update_count": active_actor["voc_gate_update_count"],
            "voc_optimizer_state": actor_optimizer,
            "actor_scheduler_state": actor_scheduler,
        },
        "model_step": int(model_checkpoint["step"]),
        "model_real_step": int(model_checkpoint["real_step"]),
        "model_state_tensor_count": model_tensor_count,
        "model_optimizer_state": model_optimizers,
        "model_scheduler_state": model_schedulers,
        "model_scaler_state": model_scalers,
        "config_use_wandb": bool(config["use_wandb"]),
        "completion_evidence": before,
        **(
            {"model_input_seal": model_input_seal}
            if model_input_seal is not None
            else {}
        ),
    }


def validate_schema7_final_bundle(ckpdir, *, label="schema-7 final bundle"):
    """Validate a final bundle and require the schema-7 seal contract."""

    bundle = validate_schema6_final_bundle(ckpdir, label=label)
    if (
        bundle["resolved_identity"]["gate_schema"]
        != VOC_GATE_POLICY_MODEL_INPUT_SEAL_SCHEMA_VERSION
    ):
        raise ValueError(f"{label} is not schema 7")
    return bundle


def validate_schema8_final_bundle(ckpdir, *, label="schema-8 final bundle"):
    """Validate a final bundle and require the schema-8 loss identity."""

    bundle = validate_schema6_final_bundle(ckpdir, label=label)
    identity = bundle["resolved_identity"]
    if (
        identity.get("gate_schema")
        != VOC_GATE_POLICY_HALF_SQUARED_Q_SCHEMA_VERSION
    ):
        raise ValueError(f"{label} is not schema 8")
    if identity.get("voc_q_regression_loss") != "half_squared_td":
        raise ValueError(f"{label} lacks half-squared Q regression identity")
    return bundle


def validate_schema9_final_bundle(ckpdir, *, label="schema-9 final bundle"):
    """Validate a final bundle and require both schema-9 derived identities."""

    bundle = validate_schema6_final_bundle(ckpdir, label=label)
    identity = bundle["resolved_identity"]
    if (
        identity.get("gate_schema")
        != VOC_GATE_POLICY_COMMON_MODE_Q_SCHEMA_VERSION
        or identity.get("voc_gate_policy_schema_version")
        != VOC_GATE_POLICY_COMMON_MODE_Q_SCHEMA_VERSION
    ):
        raise ValueError(f"{label} is not schema 9")
    expected_keys = {
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
    }
    if set(identity) != expected_keys:
        raise ValueError(f"{label} has malformed schema-9 resolved identity")
    if identity.get("voc_q_regression_loss") != "half_squared_td":
        raise ValueError(f"{label} lacks half-squared Q regression identity")
    if identity.get("voc_q_reconstruction") != (
        "detached_value_plus_raw_head_mean_plus_policy_centered_raw_head"
    ):
        raise ValueError(f"{label} lacks common-mode Q reconstruction identity")
    return bundle


def validate_schema10_final_bundle(ckpdir, *, label="schema-10 final bundle"):
    """Validate a final bundle and require both schema-10 identities."""

    bundle = validate_schema6_final_bundle(ckpdir, label=label)
    identity = bundle["resolved_identity"]
    if (
        identity.get("gate_schema")
        != VOC_GATE_POLICY_HUBER_COMMON_Q_SCHEMA_VERSION
        or identity.get("voc_gate_policy_schema_version")
        != VOC_GATE_POLICY_HUBER_COMMON_Q_SCHEMA_VERSION
    ):
        raise ValueError(f"{label} is not schema 10")
    expected_keys = {
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
    }
    if set(identity) != expected_keys:
        raise ValueError(f"{label} has malformed schema-10 resolved identity")
    if identity.get("voc_q_regression_loss") != "smooth_l1_beta1":
        raise ValueError(f"{label} lacks beta-1 SmoothL1 Q regression identity")
    if identity.get("voc_q_reconstruction") != (
        "detached_value_plus_raw_head_mean_plus_policy_centered_raw_head"
    ):
        raise ValueError(f"{label} lacks common-mode Q reconstruction identity")
    return bundle


def validate_schema11_final_bundle(ckpdir, *, label="schema-11 final bundle"):
    """Validate a final bundle and require all schema-11 identities."""

    bundle = validate_schema6_final_bundle(ckpdir, label=label)
    identity = bundle["resolved_identity"]
    if (
        identity.get("gate_schema")
        != VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION
        or identity.get("voc_gate_policy_schema_version")
        != VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION
    ):
        raise ValueError(f"{label} is not schema 11")
    expected_keys = {
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
    if set(identity) != expected_keys:
        raise ValueError(f"{label} has malformed schema-11 resolved identity")
    if identity.get("voc_q_regression_loss") != "smooth_l1_beta1":
        raise ValueError(f"{label} lacks beta-1 SmoothL1 Q regression identity")
    if identity.get("voc_q_reconstruction") != (
        "detached_value_plus_raw_head_mean_plus_policy_centered_raw_head"
    ):
        raise ValueError(f"{label} lacks common-mode Q reconstruction identity")
    if identity.get("voc_q_optimizer_coordinates") != (
        "orthonormal_common_difference_adam"
    ):
        raise ValueError(f"{label} lacks orthonormal Q optimizer identity")
    return bundle


def validate_schema12_final_bundle(ckpdir, *, label="schema-12 final bundle"):
    """Validate a final bundle and require all schema-12 identities."""

    bundle = validate_schema6_final_bundle(ckpdir, label=label)
    identity = bundle["resolved_identity"]
    if (
        identity.get("gate_schema") != VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION
        or identity.get("voc_gate_policy_schema_version")
        != VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION
    ):
        raise ValueError(f"{label} is not schema 12")
    expected_keys = {
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
    if set(identity) != expected_keys:
        raise ValueError(f"{label} has malformed schema-12 resolved identity")
    if identity.get("voc_q_regression_loss") != "smooth_l1_beta1":
        raise ValueError(f"{label} lacks beta-1 SmoothL1 Q regression identity")
    if identity.get("voc_q_reconstruction") != (
        "detached_value_plus_raw_head_mean_plus_policy_centered_raw_head"
    ):
        raise ValueError(f"{label} lacks common-mode Q reconstruction identity")
    if identity.get("voc_q_optimizer_coordinates") != (
        "orthonormal_common_difference_adam"
    ):
        raise ValueError(f"{label} lacks orthonormal Q optimizer identity")
    return bundle


def _schema13_telemetry_terminal_expectations(
    actor_checkpoint, config, actor_policy, *, label
):
    """Normalize the exact private checkpoint state used for reconciliation."""

    optimizer = actor_checkpoint.get("voc_optimizer_state_dict")
    scheduler = actor_checkpoint.get("voc_scheduler_state_dict")
    if not isinstance(optimizer, collections.abc.Mapping) or not isinstance(
        scheduler, collections.abc.Mapping
    ):
        raise ValueError(f"{label} lacks Q optimizer/scheduler state")
    groups = optimizer.get("param_groups")
    state = optimizer.get("state")
    if (
        type(groups) is not list
        or len(groups) != 1
        or not isinstance(groups[0], collections.abc.Mapping)
        or not isinstance(state, collections.abc.Mapping)
    ):
        raise ValueError(f"{label} has malformed Q optimizer state")
    parameter_ids = groups[0].get("params")
    if type(parameter_ids) is not list or len(parameter_ids) != 2:
        raise ValueError(f"{label} Q optimizer parameters are not weight/bias")
    update_count = int(actor_checkpoint["voc_update_count"])
    parameter_states = []
    if update_count == 0:
        if state:
            raise ValueError(f"{label} zero-update Q optimizer has state")
    else:
        for parameter_id in parameter_ids:
            parameter_state = state.get(parameter_id)
            if not isinstance(parameter_state, collections.abc.Mapping):
                raise ValueError(f"{label} lacks a Q Adam parameter state")
            parameter_states.append(parameter_state)

    def scalar_step(parameter_state, name):
        value = torch.as_tensor(parameter_state.get("step"))
        if value.numel() != 1 or not torch.isfinite(value).all():
            raise ValueError(f"{label} has invalid Q Adam step for {name}")
        numeric = float(value.item())
        if not numeric.is_integer() or numeric < 0:
            raise ValueError(f"{label} has invalid Q Adam step for {name}")
        return int(numeric)

    if update_count == 0:
        adam_steps = (0, 0)
        adam_m_after = None
        adam_v_after = None
    else:
        adam_steps = tuple(
            scalar_step(parameter_state, name)
            for parameter_state, name in zip(parameter_states, ("weight", "bias"))
        )
        adam_m_after = tuple(
            torch.as_tensor(parameter_state["exp_avg"]).detach().cpu().clone()
            for parameter_state in parameter_states
        )
        adam_v_after = tuple(
            torch.as_tensor(parameter_state["exp_avg_sq"]).detach().cpu().clone()
            for parameter_state in parameter_states
        )
    scheduler_last_lr = scheduler.get("_last_lr")
    if not isinstance(scheduler_last_lr, (list, tuple)) or len(
        scheduler_last_lr
    ) != 1:
        raise ValueError(f"{label} has malformed Q scheduler last LR")
    if bool(config["float16"]):
        scaler = actor_checkpoint.get("voc_grad_scaler_state_dict")
        if not isinstance(scaler, collections.abc.Mapping):
            raise ValueError(f"{label} lacks Q GradScaler state")
        amp_scale = float(scaler["scale"])
        amp_growth_tracker = int(scaler["_growth_tracker"])
        # The inherited Q scaler has its own pinned 2**8 initialization;
        # actor_amp_init_scale applies only to the main actor optimizer.
        amp_initial_scale = 256.0
    else:
        if actor_checkpoint.get("voc_grad_scaler_state_dict") is not None:
            raise ValueError(f"{label} FP32 Q state unexpectedly has a scaler")
        amp_scale = 1.0
        amp_growth_tracker = None
        amp_initial_scale = 1.0
    expected_state = {
        "voc_update_count": update_count,
        "ema_update_count": int(
            actor_checkpoint["voc_ema_gate_update_count"]
        ),
        "projection_count": int(actor_checkpoint["voc_gate_update_count"]),
        "adam_step_weight": adam_steps[0],
        "adam_step_bias": adam_steps[1],
        "q_scheduler_last_epoch": int(scheduler["last_epoch"]),
        "q_scheduler_step_count": int(scheduler["_step_count"]),
        "q_optimizer_lr": float(groups[0]["lr"]),
        "q_scheduler_last_lr": float(scheduler_last_lr[0]),
        "amp_scale": amp_scale,
        "amp_growth_tracker": amp_growth_tracker,
        "amp_skip_count": int(actor_checkpoint["voc_amp_skip_count"]),
        "amp_consecutive_skips": int(
            actor_checkpoint["voc_amp_consecutive_skips"]
        ),
        "adam_m_after": adam_m_after,
        "adam_v_after": adam_v_after,
    }
    publication_history = actor_policy.get(
        "voc_actor_policy_publication_history"
    )
    if type(publication_history) is not tuple:
        raise ValueError(f"{label} lacks normalized publication history")
    return expected_state, publication_history, amp_initial_scale


def validate_schema13_final_bundle(ckpdir, *, label="schema-13 final bundle"):
    """Validate schema-13 terminal state plus its sealed telemetry evidence."""

    root = os.path.abspath(ckpdir)
    bundle = validate_schema6_final_bundle(root, label=label)
    identity = bundle["resolved_identity"]
    if (
        identity.get("gate_schema")
        != VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION
        or identity.get("voc_gate_policy_schema_version")
        != VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION
    ):
        raise ValueError(f"{label} is not schema 13")
    expected_identity_keys = {
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
    if set(identity) != expected_identity_keys:
        raise ValueError(f"{label} has malformed schema-13 resolved identity")
    if identity.get("voc_q_regression_loss") != "smooth_l1_beta1":
        raise ValueError(f"{label} lacks beta-1 SmoothL1 Q regression identity")
    if identity.get("voc_q_reconstruction") != (
        "detached_value_plus_raw_head_mean_plus_policy_centered_raw_head"
    ):
        raise ValueError(f"{label} lacks common-mode Q reconstruction identity")
    if identity.get("voc_q_optimizer_coordinates") != (
        "orthonormal_common_difference_adam"
    ):
        raise ValueError(f"{label} lacks orthonormal Q optimizer identity")

    completion_evidence = bundle["completion_evidence"]
    checkpoint_files = completion_evidence.get("checkpoint_files")
    if (
        not isinstance(checkpoint_files, collections.abc.Mapping)
        or set(checkpoint_files) != set(_SCHEMA13_COMPLETION_CHECKPOINT_FILES)
    ):
        raise ValueError(f"{label} lacks exact schema-13 completion files")
    manifest_record = checkpoint_files["voc_telemetry_manifest.json"]
    bound_evidence, bound_payloads = _collect_run_completion_snapshot_raw(
        root, gate_schema=VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION
    )
    if bound_evidence != completion_evidence:
        raise RuntimeError(
            f"{label} files changed before telemetry reconciliation"
        )
    try:
        config = yaml.safe_load(
            bound_payloads["config_c.yaml"].decode("utf-8")
        )
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError(f"{label} config must be strict UTF-8 YAML") from error
    if (
        not isinstance(config, collections.abc.Mapping)
        or config.get("voc_gate_policy_schema_version")
        != VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION
    ):
        raise ValueError(f"{label} bound config is not schema 13")
    actor_checkpoint = torch.load(
        io.BytesIO(bound_payloads["ckp_actor.tar"]),
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    actor_real_step = actor_checkpoint.get("real_step")
    if type(actor_real_step) is not int or actor_real_step <= 0:
        raise ValueError(f"{label} actor real_step is invalid")
    actor_policy = bundle["actor_policy"]
    (
        expected_terminal_state,
        expected_publication_history,
        expected_amp_initial_scale,
    ) = _schema13_telemetry_terminal_expectations(
        actor_checkpoint, config, actor_policy, label=label
    )
    telemetry = validate_schema13_telemetry_manifest(
        root,
        expected_xpid=identity["stage"][0],
        expected_terminal_policy_version=actor_policy[
            "voc_actor_policy_version"
        ],
        expected_terminal_real_step=actor_real_step,
        expected_actor_state_sha256=actor_policy[
            "voc_actor_policy_state_sha256"
        ],
        expected_publication_history_sha256=actor_policy[
            "voc_actor_policy_publication_history_sha256"
        ],
        expected_stage_total_steps=identity["stage"][2],
        expected_actor_unroll_len=identity["stage"][4],
        expected_terminal_ack_count=actor_policy[
            "voc_actor_policy_terminal_ack_count"
        ],
        expected_manifest_sha256=manifest_record["sha256"],
        expected_manifest_size=manifest_record["size"],
        expected_q_initial_lr=float(config["actor_learning_rate"]),
        expected_schedule_total_steps=int(config["schedule_total_steps"]),
        expected_amp_initial_scale=expected_amp_initial_scale,
        expected_publication_history=expected_publication_history,
        expected_terminal_state=expected_terminal_state,
    )
    after = collect_run_completion_evidence(
        root, gate_schema=VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION
    )
    if after != completion_evidence:
        raise RuntimeError(f"{label} files changed during telemetry validation")
    result = {**bundle, "telemetry": telemetry}
    if len(result) != 13:
        raise RuntimeError(f"{label} schema-13 final bundle is not exact13")
    return result


def resolve_voc_parent_checkpoint(flags):
    """Resolve the exact actor checkpoint used for a fresh control promotion."""

    explicit = str(getattr(flags, "voc_parent_checkpoint", "")).strip()
    if explicit:
        return os.path.abspath(os.path.expanduser(explicit))
    preload_actor = str(getattr(flags, "preload_actor", "")).strip()
    if preload_actor:
        return os.path.abspath(
            os.path.join(os.path.expanduser(preload_actor), "ckp_actor.tar")
        )
    return ""


def validate_voc_fresh_control_inputs(
    flags, *, label="Fresh VoC control configuration"
):
    """Require a genuinely parent-free model/actor start for fresh control.

    A control run with an actor parent follows the separately validated
    shadow-promotion path.  Without such a parent, calling the run ``fresh``
    is only truthful when ModelNet is not silently initialized from
    ``preload`` either.
    """

    if (
        getattr(flags, "dynamic_voc_mode", "off") != "control"
        or bool(getattr(flags, "ckp", False))
        or resolve_voc_parent_checkpoint(flags)
    ):
        return False
    for name in ("preload", "preload_actor", "voc_parent_checkpoint"):
        value = getattr(flags, name, "")
        if not isinstance(value, str):
            raise ValueError(f"{label} requires {name} to be a path string")
        if value != "":
            raise ValueError(
                f"{label} requires {name}='' when no actor parent is supplied"
            )
    return True


def validate_voc_shadow_preload(checkpoint, *, label="VoC shadow preload"):
    """Allow only legacy/off weights to initialize a new shadow lineage."""

    if not isinstance(checkpoint, collections.abc.Mapping):
        raise ValueError(f"{label} must be a mapping")
    embedded = checkpoint.get("flags", {})
    if not isinstance(embedded, collections.abc.Mapping):
        raise ValueError(f"{label} lacks embedded training flags")
    embedded_mode = embedded.get("dynamic_voc_mode", "off")
    top_level_mode = checkpoint.get("dynamic_voc_mode", embedded_mode)
    if top_level_mode != embedded_mode:
        raise ValueError(
            f"{label} top-level mode disagrees with embedded flags"
        )
    if embedded_mode != "off":
        raise ValueError(
            f"{label} requires an off/legacy source; got {embedded_mode!r}"
        )
    actor_state = checkpoint.get("actor_net_state_dict", {})
    if not isinstance(actor_state, collections.abc.Mapping):
        raise ValueError(f"{label} lacks actor_net_state_dict")
    learned_q_keys = sorted(
        key
        for key in actor_state
        if key.endswith("voc_head.weight") or key.endswith("voc_head.bias")
    )
    if learned_q_keys:
        raise ValueError(
            f"{label} off source contains active voc_head weights: "
            f"{learned_q_keys}"
        )
    for name in (
        "voc_update_count",
        "voc_continue_count",
        "voc_stop_count",
        "voc_ema_gate_update_count",
        "voc_ema_gate_parent_update_count",
    ):
        value = checkpoint.get(name, 0)
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) != 0
        ):
            raise ValueError(f"{label} off source has active {name}")
    for name in (
        "voc_optimizer_state_dict",
        "voc_scheduler_state_dict",
        "voc_ema_gate_head_state_dict",
        "voc_ema_gate_schema_version",
        "voc_parent_checkpoint_sha256",
        "voc_control_origin",
    ):
        if checkpoint.get(name) is not None:
            raise ValueError(f"{label} off source has active {name}")
    return {"dynamic_voc_mode": "off"}


def _is_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def validate_voc_control_checkpoint_provenance(
    checkpoint, *, label="VoC control checkpoint"
):
    """Validate whether control started fresh or from a shadow parent.

    New checkpoints carry an explicit ``voc_control_origin``.  A promoted
    control checkpoint written by the first VoC implementation did not carry
    that field, so a valid parent SHA remains a narrowly supported legacy
    representation of ``shadow_parent``.  There was no legacy fresh-control
    path, hence a missing origin plus a null parent is always rejected.

    Parent paths are audit metadata and are not required to remain readable
    after SLURM stage-out; the SHA-256 remains the stable identity.
    """

    if not isinstance(checkpoint, collections.abc.Mapping):
        raise ValueError(f"{label} must be a mapping")
    embedded = checkpoint.get("flags", {})
    if not isinstance(embedded, collections.abc.Mapping):
        raise ValueError(f"{label} lacks embedded training flags")
    embedded_mode = embedded.get("dynamic_voc_mode", "off")
    top_level_mode = checkpoint.get("dynamic_voc_mode", embedded_mode)
    if embedded_mode != "control" or top_level_mode != "control":
        raise ValueError(
            f"{label} must have matching control mode metadata; got "
            f"top-level={top_level_mode!r}, embedded={embedded_mode!r}"
        )

    origin_is_explicit = "voc_control_origin" in checkpoint
    origin = checkpoint.get("voc_control_origin")
    legacy_marker = checkpoint.get(
        "voc_control_origin_legacy_defaulted", False
    )
    if not isinstance(legacy_marker, (bool, np.bool_)):
        raise ValueError(
            f"{label} has invalid voc_control_origin_legacy_defaulted"
        )
    legacy_marker = bool(legacy_marker)
    parent_sha256 = checkpoint.get("voc_parent_checkpoint_sha256")
    parent_checkpoint = checkpoint.get("voc_parent_checkpoint")
    parent_data_signature = checkpoint.get(
        "voc_parent_imitation_data_signature"
    )
    legacy_origin_defaulted = False
    if not origin_is_explicit:
        if legacy_marker:
            raise ValueError(
                f"{label} cannot mark a missing voc_control_origin as an "
                "already-defaulted legacy origin"
            )
        if not _is_sha256(parent_sha256):
            raise ValueError(
                f"{label} lacks explicit voc_control_origin and a valid "
                "shadow-parent voc_parent_checkpoint_sha256"
            )
        origin = VOC_CONTROL_ORIGIN_SHADOW_PARENT
        legacy_origin_defaulted = True
    elif origin not in VOC_CONTROL_ORIGINS:
        raise ValueError(
            f"{label} has invalid voc_control_origin={origin!r}; expected "
            f"one of {sorted(VOC_CONTROL_ORIGINS)}"
        )

    if origin == VOC_CONTROL_ORIGIN_FRESH:
        if legacy_marker:
            raise ValueError(
                f"{label} cannot mark fresh control as a legacy-defaulted "
                "shadow promotion"
            )
        for name, value in (
            ("voc_parent_checkpoint_sha256", parent_sha256),
            ("voc_parent_checkpoint", parent_checkpoint),
            (
                "voc_parent_imitation_data_signature",
                parent_data_signature,
            ),
        ):
            if value is not None:
                raise ValueError(
                    f"{label} with fresh origin requires {name}=null"
                )
        for name in ("preload", "preload_actor", "voc_parent_checkpoint"):
            if name not in embedded:
                raise ValueError(
                    f"{label} with fresh origin lacks embedded {name}"
                )
            value = embedded[name]
            if not isinstance(value, str) or value != "":
                raise ValueError(
                    f"{label} with fresh origin requires embedded {name}=''"
                )
    else:
        if not _is_sha256(parent_sha256):
            raise ValueError(
                f"{label} with shadow_parent origin lacks a valid "
                "voc_parent_checkpoint_sha256"
            )
        # Old promoted checkpoints recorded only the digest.  New explicit
        # provenance records both audit fields as well.
        legacy_origin_defaulted = legacy_origin_defaulted or legacy_marker
        if origin_is_explicit and not legacy_origin_defaulted:
            if (
                not isinstance(parent_checkpoint, str)
                or not parent_checkpoint
                or not os.path.isabs(parent_checkpoint)
            ):
                raise ValueError(
                    f"{label} with shadow_parent origin lacks an absolute "
                    "voc_parent_checkpoint"
                )
            if not _is_sha256(parent_data_signature):
                raise ValueError(
                    f"{label} with shadow_parent origin lacks a valid "
                    "voc_parent_imitation_data_signature"
                )
        else:
            if parent_checkpoint is not None and (
                not isinstance(parent_checkpoint, str)
                or not parent_checkpoint
                or not os.path.isabs(parent_checkpoint)
            ):
                raise ValueError(
                    f"{label} has invalid legacy voc_parent_checkpoint"
                )
            if (
                parent_data_signature is not None
                and not _is_sha256(parent_data_signature)
            ):
                raise ValueError(
                    f"{label} has invalid legacy "
                    "voc_parent_imitation_data_signature"
                )

    activation_real_step = checkpoint.get("voc_activation_real_step")
    if (
        isinstance(activation_real_step, (bool, np.bool_))
        or not isinstance(activation_real_step, (int, np.integer))
        or int(activation_real_step) < 0
    ):
        raise ValueError(
            f"{label} must have a non-negative voc_activation_real_step"
        )
    checkpoint_real_step = checkpoint.get("real_step")
    if (
        isinstance(checkpoint_real_step, (bool, np.bool_))
        or not isinstance(checkpoint_real_step, (int, np.integer))
        or int(checkpoint_real_step) < int(activation_real_step)
    ):
        raise ValueError(
            f"{label} voc_activation_real_step must not exceed checkpoint "
            "real_step"
        )

    return {
        "voc_control_origin": origin,
        "voc_control_origin_legacy_defaulted": legacy_origin_defaulted,
        "voc_parent_checkpoint_sha256": parent_sha256,
        "voc_parent_checkpoint": parent_checkpoint,
        "voc_parent_imitation_data_signature": parent_data_signature,
        "voc_activation_real_step": int(activation_real_step),
    }


def validate_voc_shadow_checkpoint_provenance(
    checkpoint, *, label="VoC shadow checkpoint"
):
    """Require a shadow checkpoint to have no control lineage metadata."""

    if not isinstance(checkpoint, collections.abc.Mapping):
        raise ValueError(f"{label} must be a mapping")
    embedded = checkpoint.get("flags", {})
    if not isinstance(embedded, collections.abc.Mapping):
        raise ValueError(f"{label} lacks embedded training flags")
    embedded_mode = embedded.get("dynamic_voc_mode", "off")
    top_level_mode = checkpoint.get("dynamic_voc_mode", embedded_mode)
    if embedded_mode != "shadow" or top_level_mode != "shadow":
        raise ValueError(f"{label} must have matching shadow mode metadata")
    legacy_marker = checkpoint.get(
        "voc_control_origin_legacy_defaulted", False
    )
    if (
        not isinstance(legacy_marker, (bool, np.bool_))
        or bool(legacy_marker)
    ):
        raise ValueError(f"{label} has invalid control-origin legacy marker")
    parent_fields = {
        "voc_parent_checkpoint_sha256": checkpoint.get(
            "voc_parent_checkpoint_sha256"
        ),
        "voc_parent_checkpoint": checkpoint.get("voc_parent_checkpoint"),
        "voc_parent_imitation_data_signature": checkpoint.get(
            "voc_parent_imitation_data_signature"
        ),
        "voc_control_origin": checkpoint.get("voc_control_origin"),
    }
    non_null = [name for name, value in parent_fields.items() if value is not None]
    if non_null:
        raise ValueError(
            f"{label} must have null control provenance; found {non_null}"
        )
    activation_real_step = checkpoint.get("voc_activation_real_step", -1)
    if (
        isinstance(activation_real_step, (bool, np.bool_))
        or not isinstance(activation_real_step, (int, np.integer))
        or int(activation_real_step) != -1
    ):
        raise ValueError(f"{label} must have voc_activation_real_step=-1")
    return {
        **parent_fields,
        "voc_control_origin_legacy_defaulted": False,
        "voc_activation_real_step": -1,
    }


def validate_voc_resume_protocol(checkpoint, flags):
    """Require exact VoC mode/hyperparameters for a stateful resume."""

    if not isinstance(checkpoint, collections.abc.Mapping):
        raise ValueError("VoC resume checkpoint must be a mapping")
    embedded = checkpoint.get("flags", {})
    if not isinstance(embedded, collections.abc.Mapping):
        raise ValueError("VoC resume checkpoint lacks embedded training flags")
    embedded_mode = embedded.get("dynamic_voc_mode", "off")
    top_level_mode = checkpoint.get("dynamic_voc_mode", embedded_mode)
    if top_level_mode != embedded_mode:
        raise ValueError(
            "Cannot resume: checkpoint top-level dynamic_voc_mode "
            f"{top_level_mode!r} disagrees with embedded mode "
            f"{embedded_mode!r}"
        )
    run_protocol = get_voc_protocol(flags)
    if bool(run_protocol.get("voc_actor_policy_version_barrier", False)):
        raise ValueError(
            "Cannot resume schema-6 actor policy barrier runs; the protocol "
            "is fresh-origin only"
        )
    if top_level_mode != run_protocol["dynamic_voc_mode"]:
        raise ValueError(
            "Cannot resume across dynamic_voc_mode values: "
            f"{top_level_mode!r} != "
            f"{run_protocol['dynamic_voc_mode']!r}"
        )
    if top_level_mode != "off" and "dynamic_voc_mode" not in checkpoint:
        raise ValueError(
            "Active VoC resume checkpoint lacks top-level dynamic_voc_mode"
        )
    gate_schema = None
    if (
        top_level_mode != "off"
        and "voc_gate_policy_schema_version" in checkpoint
    ):
        gate_schema = validate_voc_gate_policy_schema(
            checkpoint, label="Active VoC resume checkpoint"
        )
    saved_protocol = {"dynamic_voc_mode": top_level_mode}
    for name, default in VOC_PROTOCOL_DEFAULTS.items():
        if name == "dynamic_voc_mode":
            continue
        if top_level_mode == "off" and name in VOC_ACTIVE_ONLY_PROTOCOL_FIELDS:
            # This field was not part of the legacy-off resume contract and is
            # irrelevant when no VoC learner/gate is active.
            saved_protocol[name] = default
            continue
        if top_level_mode != "off" and name not in embedded:
            if name in (
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
                "voc_model_input_seal_schema_version",
                "actor_amp_init_scale",
            ):
                if gate_schema is None:
                    gate_schema = validate_voc_gate_policy_schema(
                        checkpoint, label="Active VoC resume checkpoint"
                    )
                if (
                    name == "voc_model_input_seal_schema_version"
                    and gate_schema["voc_gate_policy_schema_version"]
                    != VOC_GATE_POLICY_MODEL_INPUT_SEAL_SCHEMA_VERSION
                ):
                    # Schema <= 6 predates the seal field.  Its authoritative
                    # public gate-schema result intentionally retains the
                    # historical shape, while the internal resume protocol
                    # resolves the absent field to the strict legacy zero.
                    saved = 0
                else:
                    saved = gate_schema[name]
            else:
                raise ValueError(
                    f"Active VoC resume checkpoint lacks embedded {name}"
                )
        else:
            saved = embedded.get(name, default)
        expected = run_protocol[name]
        if name == "entropy_r_cost":
            saved = _require_environment_return_only_voc(
                saved, label="Active VoC resume checkpoint"
            )
            expected = _require_environment_return_only_voc(
                expected, label="Active VoC resume configuration"
            )
        if name in (
            "voc_gate_adam_beta1",
            "voc_gate_param_align_coef",
        ):
            matches = (
                not isinstance(saved, (bool, np.bool_))
                and isinstance(saved, (int, float, np.number))
                and np.isfinite(saved)
                and float(saved) == float(expected)
            )
        elif isinstance(default, bool):
            matches = isinstance(saved, (bool, np.bool_)) and bool(saved) == expected
        else:
            matches = (
                not isinstance(saved, (bool, np.bool_))
                and isinstance(saved, (int, float, np.number))
                and np.isfinite(saved)
                and float(saved) == float(expected)
            )
        if not matches:
            raise ValueError(
                f"Cannot resume across VoC setting {name}: "
                f"{saved!r} != {expected!r}"
            )
        saved_protocol[name] = bool(saved) if isinstance(default, bool) else float(saved)
    if top_level_mode != "off":
        if gate_schema is None:
            # A complete active protocol with an explicit beta still needs a
            # versioned interpretation.  Delay this check until after the
            # field loop so a partially written checkpoint reports its first
            # missing protocol field rather than obscuring it with schema
            # fallout.
            gate_schema = validate_voc_gate_policy_schema(
                checkpoint, label="Active VoC resume checkpoint"
            )
        _require_voc_ema_gate_protocol(
            saved_protocol["voc_ema_gate_target"],
            saved_protocol["voc_gate_target_tau"],
            label="Active VoC resume checkpoint",
        )
        _require_voc_ema_gate_protocol(
            run_protocol["voc_ema_gate_target"],
            run_protocol["voc_gate_target_tau"],
            label="Active VoC resume configuration",
        )
        _require_voc_gate_policy_protocol(
            saved_protocol["voc_dedicated_gate"],
            saved_protocol["voc_soft_q_bce_gate"],
            saved_protocol["voc_gate_q_temperature"],
            saved_protocol["voc_gate_confidence_weighted"],
            saved_protocol["voc_gate_adam_beta1"],
            saved_protocol["voc_gate_learning_rate"],
            saved_protocol["voc_gate_grad_norm_clipping"],
            saved_protocol["voc_gate_param_align"],
            saved_protocol["voc_gate_param_align_coef"],
            saved_protocol["voc_gate_exact_projection"],
            saved_protocol["voc_gate_epsilon_greedy_execution"],
            label="Active VoC resume checkpoint",
        )
        _require_voc_gate_policy_protocol(
            run_protocol["voc_dedicated_gate"],
            run_protocol["voc_soft_q_bce_gate"],
            run_protocol["voc_gate_q_temperature"],
            run_protocol["voc_gate_confidence_weighted"],
            run_protocol["voc_gate_adam_beta1"],
            run_protocol["voc_gate_learning_rate"],
            run_protocol["voc_gate_grad_norm_clipping"],
            run_protocol["voc_gate_param_align"],
            run_protocol["voc_gate_param_align_coef"],
            run_protocol["voc_gate_exact_projection"],
            run_protocol["voc_gate_epsilon_greedy_execution"],
            label="Active VoC resume configuration",
        )
    return saved_protocol


def validate_voc_holdout_calibration(
    checkpoint, *, label="VoC checkpoint", require_positive_support=True
):
    """Validate held-out TD sufficient statistics and derived metrics.

    The raw sums are the auditable source of truth.  Recomputing bias, MAE,
    and RMSE here prevents a hand-edited or partially written checkpoint from
    qualifying for control promotion with unrelated derived values.
    """

    if not isinstance(checkpoint, collections.abc.Mapping):
        raise ValueError(f"{label} must be a mapping")
    counts = {}
    for key in (
        "voc_holdout_count",
        "voc_holdout_continue_count",
        "voc_holdout_stop_count",
    ):
        value = checkpoint.get(key)
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
        ):
            raise ValueError(f"{label} has invalid {key}")
        value = int(value)
        if value < 0 or (require_positive_support and value <= 0):
            qualifier = "positive " if require_positive_support else "non-negative "
            raise ValueError(f"{label} must have {qualifier}{key}; got {value}")
        counts[key] = value
    if (
        counts["voc_holdout_continue_count"]
        + counts["voc_holdout_stop_count"]
        != counts["voc_holdout_count"]
    ):
        raise ValueError(f"{label} held-out support counts disagree")

    sums = {}
    for key in (
        "voc_holdout_td_sum",
        "voc_holdout_td_abs_sum",
        "voc_holdout_td_sq_sum",
    ):
        value = checkpoint.get(key)
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.number))
            or not np.isfinite(value)
        ):
            raise ValueError(f"{label} has invalid {key}")
        value = float(value)
        if key != "voc_holdout_td_sum" and value < 0.0:
            raise ValueError(f"{label} has negative {key}")
        sums[key] = value

    count = counts["voc_holdout_count"]
    if count == 0:
        if any(value != 0.0 for value in sums.values()):
            raise ValueError(f"{label} has nonzero held-out sums with zero support")
        expected = {
            "voc_holdout_td_bias": None,
            "voc_holdout_td_mae": None,
            "voc_holdout_td_rmse": None,
        }
    else:
        tolerance = 1e-12 * max(
            1.0,
            sums["voc_holdout_td_abs_sum"],
            sums["voc_holdout_td_sq_sum"],
        )
        if sums["voc_holdout_td_abs_sum"] + tolerance < abs(
            sums["voc_holdout_td_sum"]
        ):
            raise ValueError(f"{label} held-out absolute sum is inconsistent")
        # Cauchy-Schwarz on absolute TD errors: sum(e^2) >= sum(|e|)^2 / n.
        # This also implies the weaker signed-sum bound.
        minimum_sq_sum = sums["voc_holdout_td_abs_sum"] ** 2 / count
        if sums["voc_holdout_td_sq_sum"] + tolerance < minimum_sq_sum:
            raise ValueError(f"{label} held-out squared sum is inconsistent")
        maximum_sq_sum = sums["voc_holdout_td_abs_sum"] ** 2
        if sums["voc_holdout_td_sq_sum"] > maximum_sq_sum + tolerance:
            raise ValueError(f"{label} held-out squared sum is inconsistent")
        expected = {
            "voc_holdout_td_bias": sums["voc_holdout_td_sum"] / count,
            "voc_holdout_td_mae": sums["voc_holdout_td_abs_sum"] / count,
            "voc_holdout_td_rmse": math.sqrt(
                sums["voc_holdout_td_sq_sum"] / count
            ),
        }

    calibration = {}
    for key, expected_value in expected.items():
        value = checkpoint.get(key)
        if expected_value is None:
            if value is not None:
                raise ValueError(f"{label} must store {key}=None with zero support")
            calibration[key] = None
            continue
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.number))
            or not np.isfinite(value)
        ):
            raise ValueError(f"{label} lacks finite held-out calibration {key}")
        value = float(value)
        if key != "voc_holdout_td_bias" and value < 0.0:
            raise ValueError(f"{label} has negative {key}")
        if not math.isclose(value, expected_value, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(
                f"{label} {key}={value!r} disagrees with raw sufficient "
                f"statistics ({expected_value!r})"
            )
        calibration[key] = value
    return {**counts, **sums, **calibration}


def validate_voc_holdout_split(checkpoint, *, flags=None, label="VoC checkpoint"):
    """Require an immutable actor-stream holdout assignment."""

    expected = {
        "voc_holdout_split_version": VOC_HOLDOUT_SPLIT_VERSION,
        "voc_holdout_actor_modulus": VOC_HOLDOUT_ACTOR_MODULUS,
    }
    result = {}
    for key, expected_value in expected.items():
        value = checkpoint.get(key)
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) != expected_value
        ):
            raise ValueError(
                f"{label} has unsupported {key}={value!r}; "
                f"expected {expected_value}"
            )
        result[key] = int(value)

    streams = checkpoint.get("voc_holdout_actor_streams")
    if (
        isinstance(streams, (bool, np.bool_))
        or not isinstance(streams, (int, np.integer))
        or int(streams) <= 0
    ):
        raise ValueError(f"{label} has invalid voc_holdout_actor_streams")
    streams = int(streams)
    embedded = checkpoint.get("flags", {})
    if not isinstance(embedded, collections.abc.Mapping):
        raise ValueError(f"{label} lacks embedded training flags")

    def topology(source, source_label):
        values = []
        for name in ("self_play_n", "env_n"):
            value = source.get(name) if isinstance(source, collections.abc.Mapping) else getattr(source, name, None)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or int(value) <= 0
            ):
                raise ValueError(f"{label} has invalid {source_label} {name}")
            values.append(int(value))
        return tuple(values), values[0] * values[1]

    embedded_topology, embedded_streams = topology(embedded, "embedded")
    if embedded_streams != streams:
        raise ValueError(
            f"{label} holdout stream count {streams} disagrees with embedded "
            f"topology {embedded_topology}"
        )
    if flags is not None:
        run_topology, run_streams = topology(flags, "run")
        if run_topology != embedded_topology or run_streams != streams:
            raise ValueError(
                f"{label} holdout topology {embedded_topology} does not match "
                f"run topology {run_topology}"
            )
    result["voc_holdout_actor_streams"] = streams
    result["voc_holdout_self_play_n"] = embedded_topology[0]
    result["voc_holdout_env_n"] = embedded_topology[1]
    return result


def validate_voc_ema_gate_checkpoint(checkpoint, *, label="VoC checkpoint"):
    """Validate the frozen FP32 gate-Q target and its lifetime counter."""

    if not isinstance(checkpoint, collections.abc.Mapping):
        raise ValueError(f"{label} must be a mapping")
    embedded = checkpoint.get("flags", {})
    if not isinstance(embedded, collections.abc.Mapping):
        raise ValueError(f"{label} lacks embedded training flags")
    mode = embedded.get("dynamic_voc_mode", "off")
    if mode not in ("shadow", "control"):
        raise ValueError(f"{label} EMA gate state requires active VoC mode")
    for name in ("voc_ema_gate_target", "voc_gate_target_tau"):
        if name not in embedded:
            raise ValueError(f"{label} lacks embedded {name}")
        if name not in checkpoint:
            raise ValueError(f"{label} lacks top-level {name}")
    embedded_enabled, embedded_tau = _require_voc_ema_gate_protocol(
        embedded["voc_ema_gate_target"],
        embedded["voc_gate_target_tau"],
        label=label,
    )
    top_enabled, top_tau = _require_voc_ema_gate_protocol(
        checkpoint["voc_ema_gate_target"],
        checkpoint["voc_gate_target_tau"],
        label=f"{label} top-level metadata",
    )
    if top_enabled != embedded_enabled or not math.isclose(
        top_tau, embedded_tau, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(f"{label} EMA gate protocol metadata disagrees")

    schema = checkpoint.get("voc_ema_gate_schema_version")
    if (
        isinstance(schema, (bool, np.bool_))
        or not isinstance(schema, (int, np.integer))
        or int(schema) != VOC_EMA_GATE_SCHEMA_VERSION
    ):
        raise ValueError(
            f"{label} has unsupported voc_ema_gate_schema_version={schema!r}"
        )
    update_count = checkpoint.get("voc_ema_gate_update_count")
    if (
        isinstance(update_count, (bool, np.bool_))
        or not isinstance(update_count, (int, np.integer))
        or int(update_count) < 0
    ):
        raise ValueError(f"{label} has invalid voc_ema_gate_update_count")
    update_count = int(update_count)
    parent_update_count = checkpoint.get("voc_ema_gate_parent_update_count")
    if (
        isinstance(parent_update_count, (bool, np.bool_))
        or not isinstance(parent_update_count, (int, np.integer))
        or int(parent_update_count) < 0
    ):
        raise ValueError(
            f"{label} has invalid voc_ema_gate_parent_update_count"
        )
    parent_update_count = int(parent_update_count)
    online_update_count = checkpoint.get("voc_update_count")
    if (
        isinstance(online_update_count, (bool, np.bool_))
        or not isinstance(online_update_count, (int, np.integer))
        or int(online_update_count) < 0
    ):
        raise ValueError(f"{label} has invalid voc_update_count")
    if update_count != parent_update_count + int(online_update_count):
        raise ValueError(
            f"{label} EMA count must equal parent count plus successful "
            "online Q updates"
        )
    control_origin = None
    if mode == "shadow":
        if parent_update_count != 0:
            raise ValueError(f"{label} shadow EMA parent count must be zero")
    else:
        provenance = validate_voc_control_checkpoint_provenance(
            checkpoint, label=label
        )
        control_origin = provenance["voc_control_origin"]
        if control_origin == VOC_CONTROL_ORIGIN_FRESH:
            if parent_update_count != 0:
                raise ValueError(
                    f"{label} fresh-control EMA parent count must be zero"
                )
        elif parent_update_count <= 0:
            raise ValueError(
                f"{label} promoted-control EMA parent count must be positive"
            )
        parent_path = provenance["voc_parent_checkpoint"]
        if (
            control_origin == VOC_CONTROL_ORIGIN_SHADOW_PARENT
            and isinstance(parent_path, str)
            and os.path.isfile(parent_path)
        ):
            with open(parent_path, "rb") as handle:
                parent_bytes = handle.read()
            parent_digest = hashlib.sha256(parent_bytes).hexdigest()
            if parent_digest != provenance["voc_parent_checkpoint_sha256"]:
                raise ValueError(f"{label} available parent digest disagrees")
            parent_checkpoint = torch.load(
                io.BytesIO(parent_bytes),
                map_location=torch.device("cpu"),
                weights_only=False,
            )
            parent_ema = validate_voc_ema_gate_checkpoint(
                parent_checkpoint, label=f"{label} shadow parent"
            )
            if (
                parent_ema["voc_ema_gate_update_count"]
                != parent_update_count
            ):
                raise ValueError(
                    f"{label} EMA parent count disagrees with parent checkpoint"
                )

    state = checkpoint.get("voc_ema_gate_head_state_dict")
    if not isinstance(state, collections.abc.Mapping) or set(state) != {
        "weight",
        "bias",
    }:
        raise ValueError(
            f"{label} EMA gate state must contain exactly weight and bias"
        )
    weight = torch.as_tensor(state["weight"])
    bias = torch.as_tensor(state["bias"])
    if weight.dtype != torch.float32 or bias.dtype != torch.float32:
        raise ValueError(f"{label} EMA gate state must use FP32 master tensors")

    actor_state = checkpoint.get("actor_net_state_dict")
    if not isinstance(actor_state, collections.abc.Mapping):
        raise ValueError(f"{label} lacks actor_net_state_dict")
    key_pairs = (
        ("voc_head.weight", "voc_head.bias"),
        ("critic.voc_head.weight", "critic.voc_head.bias"),
    )
    matched_pair = next(
        (pair for pair in key_pairs if all(key in actor_state for key in pair)),
        None,
    )
    if matched_pair is None:
        raise ValueError(f"{label} lacks online voc_head weights")
    online_weight = torch.as_tensor(actor_state[matched_pair[0]])
    online_bias = torch.as_tensor(actor_state[matched_pair[1]])
    if tuple(weight.shape) != tuple(online_weight.shape) or tuple(
        bias.shape
    ) != tuple(online_bias.shape):
        raise ValueError(f"{label} EMA gate state shape disagrees with voc_head")
    if weight.ndim != 2 or weight.shape[0] != 2 or bias.shape != (2,):
        raise ValueError(
            f"{label} EMA gate head must output [Q_continue, Q_stop]"
        )
    if not torch.isfinite(online_weight).all() or not torch.isfinite(
        online_bias
    ).all():
        raise ValueError(f"{label} online voc_head contains non-finite values")
    if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
        raise ValueError(f"{label} EMA gate state contains non-finite values")
    if update_count == 0 and (
        mode == "shadow" or control_origin == VOC_CONTROL_ORIGIN_FRESH
    ):
        if (
            torch.count_nonzero(online_weight).item() != 0
            or torch.count_nonzero(online_bias).item() != 0
            or torch.count_nonzero(weight).item() != 0
            or torch.count_nonzero(bias).item() != 0
            or not torch.equal(weight, online_weight.float())
            or not torch.equal(bias, online_bias.float())
        ):
            raise ValueError(
                f"{label} zero-update fresh EMA/online heads must be equal zero"
            )
    return {
        "voc_ema_gate_target": True,
        "voc_gate_target_tau": top_tau,
        "voc_ema_gate_schema_version": int(schema),
        "voc_ema_gate_head_state_dict": {
            "weight": weight.detach().clone(),
            "bias": bias.detach().clone(),
        },
        "voc_ema_gate_update_count": update_count,
        "voc_ema_gate_parent_update_count": parent_update_count,
    }


def _voc_lambda_lr_coordinates(
    checkpoint, embedded, update_count, *, flags=None, label, component
):
    """Return the only valid LambdaLR epoch/multiplier for saved progress."""

    if int(update_count) == 0:
        return 0, 1.0
    real_step = checkpoint.get("real_step")
    if (
        isinstance(real_step, (bool, np.bool_))
        or not isinstance(real_step, (int, np.integer))
        or int(real_step) < 0
    ):
        raise ValueError(f"{label} has invalid real_step for {component}")
    horizon = embedded.get("schedule_total_steps")
    if (
        isinstance(horizon, (bool, np.bool_))
        or not isinstance(horizon, (int, np.integer))
        or int(horizon) <= 0
    ):
        raise ValueError(
            f"{label} has invalid embedded schedule_total_steps for "
            f"{component}"
        )
    horizon = int(horizon)
    if flags is not None:
        run_horizon = getattr(flags, "schedule_total_steps", None)
        if (
            isinstance(run_horizon, (bool, np.bool_))
            or not isinstance(run_horizon, (int, np.integer))
            or int(run_horizon) != horizon
        ):
            raise ValueError(
                f"{label} schedule_total_steps disagrees with the run"
            )
    # LambdaLR performs its constructor step at epoch zero.  Thereafter the
    # learner explicitly sets last_epoch=real_step-1 before every successful
    # isolated optimizer step, so even direct zero-real-step unit updates land
    # at epoch one.
    expected_epoch = max(int(real_step), 1)
    multiplier = 1.0 - min(max(float(expected_epoch), 0.0) / horizon, 1.0)
    return expected_epoch, multiplier


def validate_voc_checkpoint_components(
    checkpoint, *, flags=None, label="VoC checkpoint"
):
    """Validate learned Q-head and exact optimizer/scheduler provenance."""

    if not isinstance(checkpoint, collections.abc.Mapping):
        raise ValueError(f"{label} must be a mapping")
    embedded = checkpoint.get("flags", {})
    if not isinstance(embedded, collections.abc.Mapping):
        raise ValueError(f"{label} lacks embedded training flags")

    state_dict = checkpoint.get("actor_net_state_dict")
    if not isinstance(state_dict, collections.abc.Mapping):
        raise ValueError(f"{label} lacks actor_net_state_dict")
    key_pairs = (
        ("voc_head.weight", "voc_head.bias"),
        ("critic.voc_head.weight", "critic.voc_head.bias"),
    )
    matched_pair = next(
        (pair for pair in key_pairs if all(key in state_dict for key in pair)),
        None,
    )
    if matched_pair is None:
        raise ValueError(f"{label} lacks voc_head weights")
    weight = torch.as_tensor(state_dict[matched_pair[0]])
    bias = torch.as_tensor(state_dict[matched_pair[1]])
    if weight.ndim != 2 or weight.shape[0] != 2 or bias.shape != (2,):
        raise ValueError(
            f"{label} voc_head must output exactly [Q_continue, Q_stop]"
        )
    if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
        raise ValueError(f"{label} voc_head contains non-finite weights")

    update_count = checkpoint.get("voc_update_count")
    if (
        isinstance(update_count, (bool, np.bool_))
        or not isinstance(update_count, (int, np.integer))
        or int(update_count) <= 0
    ):
        raise ValueError(f"{label} has invalid learned voc_update_count")
    update_count = int(update_count)

    optimizer = checkpoint.get("voc_optimizer_state_dict")
    if not isinstance(optimizer, collections.abc.Mapping):
        raise ValueError(f"{label} lacks voc_optimizer_state_dict")
    state = optimizer.get("state")
    param_groups = optimizer.get("param_groups")
    if not isinstance(state, collections.abc.Mapping):
        raise ValueError(f"{label} has invalid VoC optimizer state")
    if not isinstance(param_groups, (list, tuple)) or len(param_groups) != 1:
        raise ValueError(f"{label} VoC optimizer must have one param_group")
    group = param_groups[0]
    params = group.get("params") if isinstance(
        group, collections.abc.Mapping
    ) else None
    if (
        not isinstance(params, (list, tuple))
        or len(params) != 2
        or len(set(params)) != 2
        or any(
            isinstance(parameter_id, (bool, np.bool_))
            or not isinstance(parameter_id, (int, np.integer))
            for parameter_id in params
        )
    ):
        raise ValueError(
            f"{label} VoC optimizer must contain exactly the Q-head weight "
            "and bias parameters"
        )

    def require_finite(value, path):
        if torch.is_tensor(value):
            if torch.is_floating_point(value) and not torch.isfinite(value).all():
                raise ValueError(f"{label} has non-finite optimizer state {path}")
            return
        if isinstance(value, np.ndarray):
            if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
                raise ValueError(f"{label} has non-finite optimizer state {path}")
            return
        if isinstance(value, collections.abc.Mapping):
            for key, item in value.items():
                require_finite(item, f"{path}.{key}")
            return
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                require_finite(item, f"{path}[{index}]")
            return
        if (
            not isinstance(value, (bool, np.bool_))
            and isinstance(value, (float, np.floating))
            and not np.isfinite(value)
        ):
            raise ValueError(f"{label} has non-finite optimizer state {path}")

    require_finite(optimizer, "voc_optimizer_state_dict")
    for name in ("lr", "initial_lr", "eps", "weight_decay"):
        value = group.get(name)
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.number))
            or not np.isfinite(value)
        ):
            raise ValueError(f"{label} has invalid VoC optimizer {name}")
    expected_initial_lr = embedded.get("actor_learning_rate")
    if (
        isinstance(expected_initial_lr, (bool, np.bool_))
        or not isinstance(expected_initial_lr, (int, float, np.number))
        or not np.isfinite(expected_initial_lr)
        or float(expected_initial_lr) <= 0.0
    ):
        raise ValueError(f"{label} has invalid embedded actor_learning_rate")
    expected_initial_lr = float(expected_initial_lr)
    if flags is not None:
        run_lr = getattr(flags, "actor_learning_rate", None)
        if (
            isinstance(run_lr, (bool, np.bool_))
            or not isinstance(run_lr, (int, float, np.number))
            or not math.isclose(
                float(run_lr), expected_initial_lr,
                rel_tol=0.0, abs_tol=1e-15,
            )
        ):
            raise ValueError(
                f"{label} actor_learning_rate disagrees with the run"
            )
    if not math.isclose(
        float(group["initial_lr"]), expected_initial_lr,
        rel_tol=0.0, abs_tol=1e-12,
    ) or float(group["lr"]) < 0.0:
        raise ValueError(f"{label} VoC optimizer LR disagrees with protocol")
    if float(group["weight_decay"]) != 0.0:
        raise ValueError(f"{label} VoC optimizer weight_decay must be zero")
    for name, expected in (
        ("maximize", False),
        ("capturable", False),
        ("differentiable", False),
    ):
        if not isinstance(group.get(name), (bool, np.bool_)) or bool(
            group[name]
        ) != expected:
            raise ValueError(f"{label} has invalid VoC optimizer {name}")
    if group.get("foreach") is not None or group.get("fused") is not None:
        raise ValueError(
            f"{label} VoC optimizer requires foreach/fused=None"
        )

    use_rms = embedded.get("actor_use_rms")
    if not isinstance(use_rms, (bool, np.bool_)):
        raise ValueError(f"{label} has invalid embedded actor_use_rms")
    if flags is not None:
        run_use_rms = getattr(flags, "actor_use_rms", None)
        if (
            not isinstance(run_use_rms, (bool, np.bool_))
            or bool(run_use_rms) != bool(use_rms)
        ):
            raise ValueError(f"{label} actor_use_rms disagrees with the run")
    if bool(use_rms):
        if (
            float(group["eps"]) != 0.01
            or float(group.get("alpha", float("nan"))) != 0.99
            or float(group.get("momentum", float("nan"))) != 0.0
            or not isinstance(group.get("centered"), (bool, np.bool_))
            or bool(group["centered"])
        ):
            raise ValueError(f"{label} VoC RMSprop protocol disagrees")
        moment_names = ("square_avg",)
    else:
        actor_adam_eps = embedded.get("actor_adam_eps")
        if (
            isinstance(actor_adam_eps, (bool, np.bool_))
            or not isinstance(actor_adam_eps, (int, float, np.number))
            or not np.isfinite(actor_adam_eps)
            or float(actor_adam_eps) <= 0.0
        ):
            raise ValueError(f"{label} has invalid embedded actor_adam_eps")
        if flags is not None:
            run_eps = getattr(flags, "actor_adam_eps", None)
            if (
                isinstance(run_eps, (bool, np.bool_))
                or not isinstance(run_eps, (int, float, np.number))
                or not math.isclose(
                    float(run_eps), float(actor_adam_eps),
                    rel_tol=0.0, abs_tol=1e-15,
                )
            ):
                raise ValueError(
                    f"{label} actor_adam_eps disagrees with the run"
                )
        betas = group.get("betas")
        if (
            not isinstance(betas, (list, tuple))
            or tuple(float(beta) for beta in betas) != (0.9, 0.999)
            or float(group["eps"]) != float(actor_adam_eps)
            or not isinstance(group.get("amsgrad"), (bool, np.bool_))
            or bool(group["amsgrad"])
            or group.get("decoupled_weight_decay") is not False
        ):
            raise ValueError(f"{label} VoC Adam protocol disagrees")
        moment_names = ("exp_avg", "exp_avg_sq")

    if set(state) != set(params):
        raise ValueError(
            f"{label} VoC optimizer state must cover exactly the Q-head "
            "weight and bias parameters"
        )
    expected_shapes = (tuple(weight.shape), tuple(bias.shape))
    for parameter_id, expected_shape in zip(params, expected_shapes):
        parameter_state = state[parameter_id]
        if not isinstance(parameter_state, collections.abc.Mapping):
            raise ValueError(f"{label} has invalid VoC optimizer state")
        if set(parameter_state) != {"step", *moment_names}:
            raise ValueError(
                f"{label} VoC optimizer state fields disagree with the "
                "configured optimizer"
            )
        step = torch.as_tensor(parameter_state.get("step"))
        if (
            step.numel() != 1
            or not torch.isfinite(step).all()
            or float(step.item()) != float(update_count)
        ):
            raise ValueError(
                f"{label} VoC optimizer step disagrees with voc_update_count"
            )
        for name in moment_names:
            moment = parameter_state.get(name)
            if not torch.is_tensor(moment) or tuple(moment.shape) != (
                expected_shape
            ):
                raise ValueError(
                    f"{label} VoC optimizer {name} shape disagrees with "
                    "voc_head"
                )

    scheduler = checkpoint.get("voc_scheduler_state_dict")
    if not isinstance(scheduler, collections.abc.Mapping) or not scheduler:
        raise ValueError(f"{label} lacks voc_scheduler_state_dict")
    require_finite(scheduler, "voc_scheduler_state_dict")
    scheduler_keys = {
        "base_lrs",
        "last_epoch",
        "_step_count",
        "_is_initial",
        "_get_lr_called_within_step",
        "_last_lr",
        "lr_lambdas",
    }
    if set(scheduler) != scheduler_keys:
        raise ValueError(f"{label} VoC scheduler state fields are incomplete")
    expected_epoch, expected_multiplier = _voc_lambda_lr_coordinates(
        checkpoint,
        embedded,
        update_count,
        flags=flags,
        label=label,
        component="VoC Q scheduler",
    )
    expected_lr = expected_initial_lr * expected_multiplier
    base_lrs = scheduler["base_lrs"]
    last_lrs = scheduler["_last_lr"]
    if (
        not isinstance(base_lrs, (list, tuple))
        or len(base_lrs) != 1
        or not math.isclose(
            float(base_lrs[0]), expected_initial_lr,
            rel_tol=0.0, abs_tol=1e-12,
        )
        or not isinstance(last_lrs, (list, tuple))
        or len(last_lrs) != 1
        or not math.isclose(
            float(last_lrs[0]), expected_lr,
            rel_tol=0.0, abs_tol=1e-12,
        )
        or not math.isclose(
            float(group["lr"]), expected_lr,
            rel_tol=0.0, abs_tol=1e-12,
        )
        or scheduler["lr_lambdas"] != [None]
    ):
        raise ValueError(f"{label} VoC scheduler LR disagrees with schedule")
    if int(scheduler["last_epoch"]) != expected_epoch:
        raise ValueError(
            f"{label} VoC scheduler last_epoch disagrees with real_step"
        )
    if int(scheduler["_step_count"]) != update_count + 1:
        raise ValueError(
            f"{label} VoC scheduler step count disagrees with "
            "voc_update_count"
        )
    for name in ("last_epoch", "_step_count"):
        value = scheduler[name]
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < 0
        ):
            raise ValueError(f"{label} has invalid VoC scheduler {name}")
    for name in ("_is_initial", "_get_lr_called_within_step"):
        if (
            not isinstance(scheduler[name], (bool, np.bool_))
            or bool(scheduler[name])
        ):
            raise ValueError(f"{label} has invalid VoC scheduler {name}")
    validate_voc_ema_gate_checkpoint(checkpoint, label=label)
    return matched_pair


def validate_voc_gate_policy_checkpoint(
    checkpoint, *, flags=None, label="VoC checkpoint"
):
    """Validate the isolated soft-Q gate policy and optimizer state.

    Shadow runs carry an exactly-zero gate so the architecture promoted to
    control is explicit, but never update that gate.  Control checkpoints may
    start at zero updates; once an update succeeds, optimizer state must be
    present and resumeable.  The scaler/counters follow the actor precision
    mode independently of the actor and online-Q optimizers.
    """

    if not isinstance(checkpoint, collections.abc.Mapping):
        raise ValueError(f"{label} must be a mapping")
    embedded = checkpoint.get("flags", {})
    if not isinstance(embedded, collections.abc.Mapping):
        raise ValueError(f"{label} lacks embedded training flags")
    mode = embedded.get("dynamic_voc_mode", "off")
    if mode not in ("shadow", "control"):
        raise ValueError(
            f"{label} dedicated gate state requires active VoC mode"
        )
    schema_state = validate_voc_gate_policy_schema(checkpoint, label=label)
    protocol_names = (
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
    )
    for name in protocol_names:
        if name not in embedded and name not in (
            "voc_gate_adam_beta1",
            "voc_gate_param_align",
            "voc_gate_param_align_coef",
            "voc_gate_exact_projection",
            "voc_gate_epsilon_greedy_execution",
        ):
            raise ValueError(f"{label} lacks embedded {name}")
    protocol_values = dict(embedded)
    for name in (
        "voc_gate_adam_beta1",
        "voc_gate_param_align",
        "voc_gate_param_align_coef",
        "voc_gate_exact_projection",
        "voc_gate_epsilon_greedy_execution",
    ):
        protocol_values[name] = schema_state[name]
    protocol = _require_voc_gate_policy_protocol(
        *(protocol_values[name] for name in protocol_names), label=label
    )
    if flags is not None:
        run_values = get_voc_protocol(flags)
        run_protocol = _require_voc_gate_policy_protocol(
            *(run_values[name] for name in protocol_names),
            label=f"{label} run configuration",
        )
        for name, saved, expected in zip(
            protocol_names, protocol, run_protocol
        ):
            if saved != expected:
                raise ValueError(
                    f"{label} {name} disagrees with the run: "
                    f"{saved!r} != {expected!r}"
                )

    actor_state = checkpoint.get("actor_net_state_dict")
    if not isinstance(actor_state, collections.abc.Mapping):
        raise ValueError(f"{label} lacks actor_net_state_dict")
    key_pairs = (
        ("voc_gate_head.weight", "voc_gate_head.bias"),
        ("actor.voc_gate_head.weight", "actor.voc_gate_head.bias"),
    )
    matched_pair = next(
        (pair for pair in key_pairs if all(key in actor_state for key in pair)),
        None,
    )
    if matched_pair is None:
        raise ValueError(f"{label} lacks dedicated voc_gate_head weights")
    weight = torch.as_tensor(actor_state[matched_pair[0]])
    bias = torch.as_tensor(actor_state[matched_pair[1]])
    if weight.ndim != 2 or weight.shape[0] != 1 or bias.shape != (1,):
        raise ValueError(
            f"{label} voc_gate_head must output one CONTINUE log-odds scalar"
        )
    if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
        raise ValueError(f"{label} voc_gate_head contains non-finite values")

    counters = {}
    for name in (
        "voc_gate_update_count",
        "voc_gate_amp_skip_count",
        "voc_gate_amp_consecutive_skips",
    ):
        value = checkpoint.get(name)
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < 0
        ):
            raise ValueError(f"{label} has invalid {name}")
        counters[name] = int(value)
    if (
        counters["voc_gate_amp_consecutive_skips"]
        > counters["voc_gate_amp_skip_count"]
    ):
        raise ValueError(
            f"{label} has inconsistent dedicated gate AMP skip counters"
        )
    if mode == "shadow" and counters["voc_gate_update_count"] != 0:
        raise ValueError(f"{label} shadow gate update count must be zero")
    exact_projection = protocol[9]
    if exact_projection:
        if mode != "control":
            raise ValueError(
                f"{label} exact projection requires control mode"
            )
        provenance = validate_voc_control_checkpoint_provenance(
            checkpoint, label=label
        )
        if provenance["voc_control_origin"] != VOC_CONTROL_ORIGIN_FRESH:
            raise ValueError(
                f"{label} exact projection requires a fresh control origin"
            )
        if weight.dtype != torch.float32 or bias.dtype != torch.float32:
            raise ValueError(
                f"{label} exact projection requires an FP32 gate head"
            )
        if counters["voc_gate_amp_skip_count"] != 0 or counters[
            "voc_gate_amp_consecutive_skips"
        ] != 0:
            raise ValueError(
                f"{label} exact projection requires zero gate AMP skips"
            )
        if mode == "control":
            q_update_count = checkpoint.get("voc_update_count")
            if (
                isinstance(q_update_count, (bool, np.bool_))
                or not isinstance(q_update_count, (int, np.integer))
                or int(q_update_count) < 0
                or counters["voc_gate_update_count"] != int(q_update_count)
            ):
                raise ValueError(
                    f"{label} exact projection count must equal successful "
                    "online Q updates"
                )
        ema_state = validate_voc_ema_gate_checkpoint(checkpoint, label=label)
    if counters["voc_gate_update_count"] == 0 and (
        torch.count_nonzero(weight).item() != 0
        or torch.count_nonzero(bias).item() != 0
    ):
        raise ValueError(f"{label} zero-update dedicated gate must equal zero")
    if exact_projection:
        ema_weight = ema_state["voc_ema_gate_head_state_dict"]["weight"]
        ema_bias = ema_state["voc_ema_gate_head_state_dict"]["bias"]
        policy_temperature = embedded.get("voc_gate_temperature")
        q_temperature = protocol[2]
        if (
            isinstance(policy_temperature, (bool, np.bool_))
            or not isinstance(policy_temperature, (int, float, np.number))
            or not np.isfinite(policy_temperature)
            or float(policy_temperature) <= 0.0
        ):
            raise ValueError(
                f"{label} has invalid embedded voc_gate_temperature"
            )
        scale = float(policy_temperature) / float(q_temperature)
        expected_weight = scale * (
            ema_weight[0:1] - ema_weight[1:2]
        )
        expected_bias = scale * (ema_bias[0:1] - ema_bias[1:2])
        if not torch.equal(weight, expected_weight) or not torch.equal(
            bias, expected_bias
        ):
            raise ValueError(
                f"{label} exact-projection gate disagrees with EMA Q target"
            )

    def require_finite(value, path):
        if torch.is_tensor(value):
            if torch.is_floating_point(value) and not torch.isfinite(value).all():
                raise ValueError(f"{label} has non-finite gate state {path}")
            return
        if isinstance(value, np.ndarray):
            if np.issubdtype(value.dtype, np.floating) and not np.isfinite(
                value
            ).all():
                raise ValueError(f"{label} has non-finite gate state {path}")
            return
        if isinstance(value, collections.abc.Mapping):
            for key, item in value.items():
                require_finite(item, f"{path}.{key}")
            return
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                require_finite(item, f"{path}[{index}]")
            return
        if (
            not isinstance(value, (bool, np.bool_))
            and isinstance(value, (float, np.floating))
            and not np.isfinite(value)
        ):
            raise ValueError(f"{label} has non-finite gate state {path}")

    optimizer = checkpoint.get("voc_gate_optimizer_state_dict")
    if not isinstance(optimizer, collections.abc.Mapping):
        raise ValueError(f"{label} lacks voc_gate_optimizer_state_dict")
    optimizer_state = optimizer.get("state")
    param_groups = optimizer.get("param_groups")
    if not isinstance(optimizer_state, collections.abc.Mapping):
        raise ValueError(f"{label} has invalid dedicated gate optimizer state")
    if not isinstance(param_groups, (list, tuple)) or len(param_groups) != 1:
        raise ValueError(
            f"{label} dedicated gate optimizer must have one param_group"
        )
    group = param_groups[0]
    params = group.get("params") if isinstance(
        group, collections.abc.Mapping
    ) else None
    if (
        not isinstance(params, (list, tuple))
        or len(params) != 2
        or len(set(params)) != 2
        or any(
            isinstance(parameter_id, (bool, np.bool_))
            or not isinstance(parameter_id, (int, np.integer))
            for parameter_id in params
        )
    ):
        raise ValueError(
            f"{label} dedicated gate optimizer must contain exactly the "
            "weight and bias parameters"
        )
    require_finite(optimizer, "voc_gate_optimizer_state_dict")
    for name in ("lr", "initial_lr", "eps", "weight_decay"):
        if name not in group:
            raise ValueError(
                f"{label} dedicated gate optimizer lacks {name}"
            )
        value = group[name]
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.number))
            or not np.isfinite(value)
        ):
            raise ValueError(
                f"{label} has invalid dedicated gate optimizer {name}"
            )
    if float(group["lr"]) < 0.0 or float(group["initial_lr"]) <= 0.0:
        raise ValueError(
            f"{label} has invalid dedicated gate optimizer learning rate"
        )
    if float(group["eps"]) <= 0.0 or float(group["weight_decay"]) < 0.0:
        raise ValueError(
            f"{label} has invalid dedicated gate optimizer regularization"
        )

    use_rms = embedded.get("actor_use_rms")
    if not isinstance(use_rms, (bool, np.bool_)):
        raise ValueError(f"{label} has invalid embedded actor_use_rms")
    if flags is not None:
        run_use_rms = getattr(flags, "actor_use_rms", None)
        if (
            not isinstance(run_use_rms, (bool, np.bool_))
            or bool(run_use_rms) != bool(use_rms)
        ):
            raise ValueError(
                f"{label} actor_use_rms disagrees with the run"
            )
    expected_initial_lr = float(embedded["voc_gate_learning_rate"])
    if not math.isclose(
        float(group["initial_lr"]),
        expected_initial_lr,
        rel_tol=0.0,
        abs_tol=1e-12,
    ) or not 0.0 <= float(group["lr"]) <= expected_initial_lr:
        raise ValueError(
            f"{label} dedicated gate optimizer LR disagrees with protocol"
        )
    if float(group["weight_decay"]) != 0.0:
        raise ValueError(
            f"{label} dedicated gate optimizer weight_decay must be zero"
        )
    for name, expected in (
        ("maximize", False),
        ("capturable", False),
        ("differentiable", False),
    ):
        if not isinstance(group.get(name), (bool, np.bool_)) or bool(
            group[name]
        ) != expected:
            raise ValueError(
                f"{label} has invalid dedicated gate optimizer {name}"
            )
    for name in ("foreach", "fused"):
        if group.get(name) is not None:
            raise ValueError(
                f"{label} dedicated gate optimizer requires {name}=None"
            )
    if bool(use_rms):
        for name in ("alpha", "momentum"):
            value = group.get(name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.number))
                or not np.isfinite(value)
            ):
                raise ValueError(
                    f"{label} has invalid dedicated gate RMSprop {name}"
                )
        if not 0.0 <= float(group["alpha"]) < 1.0 or float(
            group["momentum"]
        ) < 0.0:
            raise ValueError(
                f"{label} has invalid dedicated gate RMSprop parameters"
            )
        if not isinstance(group.get("centered"), (bool, np.bool_)):
            raise ValueError(
                f"{label} has invalid dedicated gate RMSprop centered"
            )
        if (
            float(group["alpha"]) != 0.99
            or float(group["momentum"]) != 0.0
            or float(group["eps"]) != 0.01
            or bool(group["centered"])
        ):
            raise ValueError(
                f"{label} dedicated gate RMSprop protocol disagrees"
            )
    else:
        actor_adam_eps = embedded.get("actor_adam_eps")
        if (
            isinstance(actor_adam_eps, (bool, np.bool_))
            or not isinstance(actor_adam_eps, (int, float, np.number))
            or not np.isfinite(actor_adam_eps)
            or float(actor_adam_eps) <= 0.0
        ):
            raise ValueError(f"{label} has invalid embedded actor_adam_eps")
        if flags is not None:
            run_adam_eps = getattr(flags, "actor_adam_eps", None)
            if (
                isinstance(run_adam_eps, (bool, np.bool_))
                or not isinstance(run_adam_eps, (int, float, np.number))
                or not math.isclose(
                    float(actor_adam_eps),
                    float(run_adam_eps),
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
            ):
                raise ValueError(
                    f"{label} actor_adam_eps disagrees with the run"
                )
        betas = group.get("betas")
        if (
            not isinstance(betas, (list, tuple))
            or len(betas) != 2
            or any(
                isinstance(beta, (bool, np.bool_))
                or not isinstance(beta, (int, float, np.number))
                or not np.isfinite(beta)
                or not 0.0 <= float(beta) < 1.0
                for beta in betas
            )
        ):
            raise ValueError(
                f"{label} has invalid dedicated gate Adam betas"
            )
        if not isinstance(group.get("amsgrad"), (bool, np.bool_)):
            raise ValueError(
                f"{label} has invalid dedicated gate Adam amsgrad"
            )
        if (
            tuple(float(beta) for beta in betas) != (protocol[4], 0.999)
            or float(group["eps"]) != float(actor_adam_eps)
            or bool(group["amsgrad"])
            or group.get("decoupled_weight_decay") is not False
        ):
            raise ValueError(
                f"{label} dedicated gate Adam protocol disagrees"
            )

    update_count = counters["voc_gate_update_count"]
    expected_shapes = (tuple(weight.shape), tuple(bias.shape))
    if exact_projection:
        if optimizer_state:
            raise ValueError(
                f"{label} exact projection requires empty gate optimizer state"
            )
    elif update_count == 0:
        if optimizer_state:
            raise ValueError(
                f"{label} zero-update dedicated gate optimizer state "
                "must be empty"
            )
    else:
        if set(optimizer_state) != set(params):
            raise ValueError(
                f"{label} learned gate optimizer state must cover exactly "
                "the weight and bias parameters"
            )
        for parameter_id, expected_shape in zip(params, expected_shapes):
            state = optimizer_state[parameter_id]
            if not isinstance(state, collections.abc.Mapping):
                raise ValueError(
                    f"{label} has invalid learned gate optimizer state"
                )
            step = state.get("step")
            step_tensor = torch.as_tensor(step)
            if (
                step_tensor.numel() != 1
                or not torch.isfinite(step_tensor).all()
                or float(step_tensor.item()) != float(update_count)
            ):
                raise ValueError(
                    f"{label} gate optimizer step disagrees with "
                    "voc_gate_update_count"
                )
            moment_names = (
                ("square_avg",)
                if bool(use_rms)
                else ("exp_avg", "exp_avg_sq")
            )
            expected_state_keys = {"step", *moment_names}
            if set(state) != expected_state_keys:
                raise ValueError(
                    f"{label} gate optimizer state fields disagree with "
                    "the configured optimizer"
                )
            for name in moment_names:
                moment = state.get(name)
                if not torch.is_tensor(moment) or tuple(moment.shape) != (
                    expected_shape
                ):
                    raise ValueError(
                        f"{label} gate optimizer {name} shape disagrees "
                        "with voc_gate_head"
                    )
            for name in (
                "momentum_buffer",
                "grad_avg",
                "max_exp_avg_sq",
            ):
                if name in state and (
                    not torch.is_tensor(state[name])
                    or tuple(state[name].shape) != expected_shape
                ):
                    raise ValueError(
                        f"{label} gate optimizer {name} shape disagrees "
                        "with voc_gate_head"
                    )

    scheduler = checkpoint.get("voc_gate_scheduler_state_dict")
    if not isinstance(scheduler, collections.abc.Mapping) or not scheduler:
        raise ValueError(f"{label} lacks voc_gate_scheduler_state_dict")
    require_finite(scheduler, "voc_gate_scheduler_state_dict")
    scheduler_keys = {
        "base_lrs",
        "last_epoch",
        "_step_count",
        "_is_initial",
        "_get_lr_called_within_step",
        "_last_lr",
        "lr_lambdas",
    }
    if set(scheduler) != scheduler_keys:
        raise ValueError(
            f"{label} dedicated gate scheduler state fields are incomplete"
        )
    base_lrs = scheduler["base_lrs"]
    last_lrs = scheduler["_last_lr"]
    lr_lambdas = scheduler["lr_lambdas"]
    if (
        not isinstance(base_lrs, (list, tuple))
        or len(base_lrs) != 1
        or not math.isclose(
            float(base_lrs[0]),
            expected_initial_lr,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not isinstance(last_lrs, (list, tuple))
        or len(last_lrs) != 1
        or not math.isclose(
            float(last_lrs[0]),
            float(group["lr"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not isinstance(lr_lambdas, (list, tuple))
        or list(lr_lambdas) != [None]
    ):
        raise ValueError(
            f"{label} dedicated gate scheduler LR state disagrees"
        )
    for name in ("last_epoch", "_step_count"):
        value = scheduler[name]
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < 0
        ):
            raise ValueError(
                f"{label} has invalid dedicated gate scheduler {name}"
            )
    optimizer_update_count = 0 if exact_projection else update_count
    if int(scheduler["_step_count"]) != optimizer_update_count + 1:
        raise ValueError(
            f"{label} gate scheduler step count disagrees with "
            "voc_gate_update_count"
        )
    for name in ("_is_initial", "_get_lr_called_within_step"):
        if (
            not isinstance(scheduler[name], (bool, np.bool_))
            or bool(scheduler[name])
        ):
            raise ValueError(
                f"{label} has invalid dedicated gate scheduler {name}"
            )
    expected_epoch, expected_multiplier = _voc_lambda_lr_coordinates(
        checkpoint,
        embedded,
        optimizer_update_count,
        flags=flags,
        label=label,
        component="dedicated gate scheduler",
    )
    if int(scheduler["last_epoch"]) != expected_epoch:
        raise ValueError(
            f"{label} dedicated gate scheduler last_epoch disagrees with "
            "real_step"
        )
    expected_lr = expected_initial_lr * expected_multiplier
    if not math.isclose(
        float(group["lr"]), expected_lr,
        rel_tol=0.0, abs_tol=1e-12,
    ) or not math.isclose(
        float(last_lrs[0]), expected_lr,
        rel_tol=0.0, abs_tol=1e-12,
    ):
        raise ValueError(
            f"{label} dedicated gate LR disagrees with schedule"
        )
    if exact_projection and (
        int(scheduler["last_epoch"]) != 0
        or int(scheduler["_step_count"]) != 1
        or float(group["lr"]) != expected_initial_lr
    ):
        raise ValueError(
            f"{label} exact projection requires a pristine gate scheduler"
        )

    float16 = embedded.get("float16")
    if not isinstance(float16, (bool, np.bool_)):
        raise ValueError(f"{label} has invalid embedded float16")
    if flags is not None:
        run_float16 = getattr(flags, "float16", None)
        if (
            not isinstance(run_float16, (bool, np.bool_))
            or bool(run_float16) != bool(float16)
        ):
            raise ValueError(f"{label} float16 disagrees with the run")
    scaler = checkpoint.get("voc_gate_grad_scaler_state_dict")
    if bool(float16):
        if not isinstance(scaler, collections.abc.Mapping) or not scaler:
            raise ValueError(f"{label} lacks dedicated gate GradScaler state")
        scaler_fields = (
            "scale",
            "growth_factor",
            "backoff_factor",
            "growth_interval",
            "_growth_tracker",
        )
        if set(scaler) != set(scaler_fields):
            raise ValueError(
                f"{label} has invalid dedicated gate GradScaler fields"
            )
        for name in scaler_fields:
            value = scaler.get(name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.number))
                or not np.isfinite(value)
            ):
                raise ValueError(
                    f"{label} has invalid dedicated gate GradScaler {name}"
                )
        if float(scaler["scale"]) <= 0.0:
            raise ValueError(
                f"{label} has non-positive dedicated gate GradScaler scale"
            )
        if not 0.0 < float(scaler["backoff_factor"]) < 1.0:
            raise ValueError(
                f"{label} has invalid dedicated gate GradScaler backoff_factor"
            )
        if float(scaler["growth_factor"]) <= 1.0:
            raise ValueError(
                f"{label} has invalid dedicated gate GradScaler growth_factor"
            )
        growth_interval = scaler["growth_interval"]
        growth_tracker = scaler["_growth_tracker"]
        if (
            isinstance(growth_interval, (bool, np.bool_))
            or not isinstance(growth_interval, (int, np.integer))
            or int(growth_interval) <= 0
            or isinstance(growth_tracker, (bool, np.bool_))
            or not isinstance(growth_tracker, (int, np.integer))
            or int(growth_tracker) < 0
            or int(growth_tracker) >= int(growth_interval)
        ):
            raise ValueError(
                f"{label} has invalid dedicated gate GradScaler counters"
            )
        if (
            float(scaler["growth_factor"]) != 2.0
            or float(scaler["backoff_factor"]) != 0.5
            or int(growth_interval) != 2000
        ):
            raise ValueError(
                f"{label} dedicated gate GradScaler protocol disagrees "
                "with constructor"
            )
        if exact_projection and (
            float(scaler["scale"]) != 256.0
            or int(growth_tracker) != 0
        ):
            raise ValueError(
                f"{label} exact projection requires a pristine gate "
                "GradScaler"
            )
    elif scaler is not None:
        raise ValueError(
            f"{label} must not store dedicated gate GradScaler state in FP32"
        )

    barrier_state = {}
    if (
        schema_state["voc_gate_policy_schema_version"]
        in VOC_GATE_POLICY_ATOMIC_SCHEMA_VERSIONS
    ):
        barrier_state = validate_actor_policy_checkpoint(
            checkpoint, label=label
        )
    return {
        **schema_state,
        "voc_gate_head_keys": matched_pair,
        "voc_dedicated_gate": protocol[0],
        "voc_soft_q_bce_gate": protocol[1],
        "voc_gate_q_temperature": protocol[2],
        "voc_gate_confidence_weighted": protocol[3],
        "voc_gate_adam_beta1": protocol[4],
        "voc_gate_learning_rate": protocol[5],
        "voc_gate_grad_norm_clipping": protocol[6],
        "voc_gate_param_align": protocol[7],
        "voc_gate_param_align_coef": protocol[8],
        "voc_gate_exact_projection": protocol[9],
        "voc_gate_epsilon_greedy_execution": protocol[10],
        **counters,
        "voc_gate_optimizer_state_saved": True,
        "voc_gate_scheduler_state_saved": True,
        "voc_gate_grad_scaler_state_saved": scaler is not None,
        **barrier_state,
    }


def validate_voc_amp_checkpoint(checkpoint, *, label="VoC checkpoint"):
    """Validate the precision-dependent VoC scaler and skip counters."""

    embedded = checkpoint.get("flags", {})
    if not isinstance(embedded, collections.abc.Mapping):
        raise ValueError(f"{label} lacks embedded training flags")
    float16 = embedded.get("float16")
    if not isinstance(float16, (bool, np.bool_)):
        raise ValueError(f"{label} has invalid embedded float16")

    counters = {}
    for key in ("voc_amp_skip_count", "voc_amp_consecutive_skips"):
        value = checkpoint.get(key)
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < 0
        ):
            raise ValueError(f"{label} has invalid {key}")
        counters[key] = int(value)
    if counters["voc_amp_consecutive_skips"] > counters["voc_amp_skip_count"]:
        raise ValueError(f"{label} has inconsistent VoC AMP skip counters")

    scaler = checkpoint.get("voc_grad_scaler_state_dict")
    if bool(float16):
        if not isinstance(scaler, collections.abc.Mapping) or not scaler:
            raise ValueError(f"{label} lacks VoC GradScaler state")
        required = (
            "scale",
            "growth_factor",
            "backoff_factor",
            "growth_interval",
            "_growth_tracker",
        )
        if set(scaler) != set(required):
            raise ValueError(f"{label} has invalid VoC GradScaler fields")
        for key in required:
            value = scaler.get(key)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.number))
                or not np.isfinite(value)
            ):
                raise ValueError(f"{label} has invalid VoC GradScaler {key}")
        if float(scaler["scale"]) <= 0.0:
            raise ValueError(f"{label} has non-positive VoC GradScaler scale")
        if not 0.0 < float(scaler["backoff_factor"]) < 1.0:
            raise ValueError(f"{label} has invalid VoC GradScaler backoff_factor")
        if float(scaler["growth_factor"]) <= 1.0:
            raise ValueError(f"{label} has invalid VoC GradScaler growth_factor")
        growth_interval = scaler["growth_interval"]
        growth_tracker = scaler["_growth_tracker"]
        if (
            isinstance(growth_interval, (bool, np.bool_))
            or not isinstance(growth_interval, (int, np.integer))
            or int(growth_interval) <= 0
            or isinstance(growth_tracker, (bool, np.bool_))
            or not isinstance(growth_tracker, (int, np.integer))
            or int(growth_tracker) < 0
            or int(growth_tracker) >= int(growth_interval)
        ):
            raise ValueError(f"{label} has invalid VoC GradScaler counters")
        if (
            float(scaler["growth_factor"]) != 2.0
            or float(scaler["backoff_factor"]) != 0.5
            or int(growth_interval) != 2000
        ):
            raise ValueError(
                f"{label} VoC GradScaler protocol disagrees with constructor"
            )
    elif scaler is not None:
        raise ValueError(f"{label} must not store VoC GradScaler state in FP32")
    return {
        "voc_float16": bool(float16),
        **counters,
        "voc_grad_scaler_state_saved": scaler is not None,
    }


def _validate_voc_active_checkpoint_state(
    checkpoint, flags, protocol, *, label
):
    """Validate all active learner state after protocol identity is fixed."""

    mode = protocol["dynamic_voc_mode"]
    if mode == "off":
        return {"dynamic_voc_mode": "off", "voc_protocol": protocol}
    counters = {}
    for key in ("voc_update_count", "voc_continue_count", "voc_stop_count"):
        value = checkpoint.get(key)
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < 0
        ):
            raise ValueError(f"{label} has invalid {key}")
        counters[key] = int(value)
    holdout = validate_voc_holdout_calibration(
        checkpoint, label=label, require_positive_support=False
    )
    split = validate_voc_holdout_split(
        checkpoint, flags=flags, label=label
    )
    amp = validate_voc_amp_checkpoint(checkpoint, label=label)
    ema = validate_voc_ema_gate_checkpoint(checkpoint, label=label)
    gate_policy = validate_voc_gate_policy_checkpoint(
        checkpoint, flags=flags, label=label
    )
    if counters["voc_update_count"] > 0:
        validate_voc_checkpoint_components(
            checkpoint, flags=flags, label=label
        )
    provenance = (
        validate_voc_control_checkpoint_provenance(checkpoint, label=label)
        if mode == "control"
        else validate_voc_shadow_checkpoint_provenance(
            checkpoint, label=label
        )
    )
    if (
        mode == "control"
        and provenance["voc_control_origin"] == VOC_CONTROL_ORIGIN_FRESH
    ):
        for name in ("preload", "preload_actor", "voc_parent_checkpoint"):
            value = getattr(flags, name, "")
            if not isinstance(value, str) or value != "":
                raise ValueError(
                    f"{label} fresh-control resume requires run {name}=''"
                )
    return {
        "dynamic_voc_mode": mode,
        "voc_protocol": protocol,
        **counters,
        **holdout,
        **split,
        **amp,
        **ema,
        **gate_policy,
        **provenance,
    }


def validate_voc_active_resume_checkpoint(
    checkpoint, flags, *, label="Active VoC resume checkpoint"
):
    """Fail closed on all active learner state before publishing weights."""

    protocol = validate_voc_resume_protocol(checkpoint, flags)
    return _validate_voc_active_checkpoint_state(
        checkpoint, flags, protocol, label=label
    )


def validate_voc_schema6_final_actor_checkpoint(
    checkpoint, flags, *, label="Schema-6 final actor checkpoint"
):
    """Validate complete active VoC state without opening a resume path.

    Schema 6 is deliberately fresh-only, so the resume protocol validator
    rejects it.  Finalization nevertheless needs the identical Q/EMA/gate,
    optimizer, scheduler, scaler, holdout, and provenance checks.  This wrapper
    fixes the schema-6 protocol first and then reuses the shared state body.
    """

    return _validate_voc_atomic_final_actor_checkpoint(
        checkpoint,
        flags,
        expected_schema=VOC_GATE_POLICY_VERSION_BARRIER_SCHEMA_VERSION,
        label=label,
    )


def validate_voc_schema7_final_actor_checkpoint(
    checkpoint, flags, *, label="Schema-7 final actor checkpoint"
):
    """Validate schema-7 active state without opening a resume path."""

    return _validate_voc_atomic_final_actor_checkpoint(
        checkpoint,
        flags,
        expected_schema=VOC_GATE_POLICY_MODEL_INPUT_SEAL_SCHEMA_VERSION,
        label=label,
    )


def validate_voc_schema8_final_actor_checkpoint(
    checkpoint, flags, *, label="Schema-8 final actor checkpoint"
):
    """Validate schema-8 active state without opening a resume path."""

    return _validate_voc_atomic_final_actor_checkpoint(
        checkpoint,
        flags,
        expected_schema=VOC_GATE_POLICY_HALF_SQUARED_Q_SCHEMA_VERSION,
        label=label,
    )


def validate_voc_schema9_final_actor_checkpoint(
    checkpoint, flags, *, label="Schema-9 final actor checkpoint"
):
    """Validate schema-9 active state without opening a resume path."""

    validated = _validate_voc_atomic_final_actor_checkpoint(
        checkpoint,
        flags,
        expected_schema=VOC_GATE_POLICY_COMMON_MODE_Q_SCHEMA_VERSION,
        label=label,
    )
    return {
        **validated,
        "voc_q_regression_loss": "half_squared_td",
        "voc_q_reconstruction": (
            "detached_value_plus_raw_head_mean_plus_policy_centered_raw_head"
        ),
    }


def validate_voc_schema10_final_actor_checkpoint(
    checkpoint, flags, *, label="Schema-10 final actor checkpoint"
):
    """Validate schema-10 active state without opening a resume path."""

    _reject_schema10_persisted_derived_identity(checkpoint, label=label)
    validated = _validate_voc_atomic_final_actor_checkpoint(
        checkpoint,
        flags,
        expected_schema=VOC_GATE_POLICY_HUBER_COMMON_Q_SCHEMA_VERSION,
        label=label,
    )
    return {
        **validated,
        "voc_q_regression_loss": "smooth_l1_beta1",
        "voc_q_reconstruction": (
            "detached_value_plus_raw_head_mean_plus_policy_centered_raw_head"
        ),
    }


def validate_voc_schema11_final_actor_checkpoint(
    checkpoint, flags, *, label="Schema-11 final actor checkpoint"
):
    """Validate schema-11 active state without opening a resume path."""

    _reject_schema11_persisted_derived_identity(checkpoint, label=label)
    validated = _validate_voc_atomic_final_actor_checkpoint(
        checkpoint,
        flags,
        expected_schema=VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION,
        label=label,
    )
    return {
        **validated,
        "voc_q_regression_loss": "smooth_l1_beta1",
        "voc_q_reconstruction": (
            "detached_value_plus_raw_head_mean_plus_policy_centered_raw_head"
        ),
        "voc_q_optimizer_coordinates": (
            "orthonormal_common_difference_adam"
        ),
    }


def validate_voc_schema12_final_actor_checkpoint(
    checkpoint, flags, *, label="Schema-12 final actor checkpoint"
):
    """Validate schema-12 active state without opening a resume path."""

    _reject_schema12_persisted_derived_identity(checkpoint, label=label)
    validated = _validate_voc_atomic_final_actor_checkpoint(
        checkpoint,
        flags,
        expected_schema=VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION,
        label=label,
    )
    return {
        **validated,
        "voc_q_regression_loss": "smooth_l1_beta1",
        "voc_q_reconstruction": (
            "detached_value_plus_raw_head_mean_plus_policy_centered_raw_head"
        ),
        "voc_q_optimizer_coordinates": (
            "orthonormal_common_difference_adam"
        ),
    }


def validate_voc_schema13_final_actor_checkpoint(
    checkpoint, flags, *, label="Schema-13 final actor checkpoint"
):
    """Validate schema-13 active state without opening a resume path."""

    _reject_schema13_persisted_derived_identity(checkpoint, label=label)
    validated = _validate_voc_atomic_final_actor_checkpoint(
        checkpoint,
        flags,
        expected_schema=VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION,
        label=label,
    )
    return {
        **validated,
        "voc_q_regression_loss": "smooth_l1_beta1",
        "voc_q_reconstruction": (
            "detached_value_plus_raw_head_mean_plus_policy_centered_raw_head"
        ),
        "voc_q_optimizer_coordinates": (
            "orthonormal_common_difference_adam"
        ),
    }


_VOC_GATE_POLICY_SCHEMA10_DERIVED_IDENTITY_KEYS = (
    "voc_q_regression_loss",
    "voc_q_reconstruction",
)


def _reject_schema10_persisted_derived_identity(checkpoint, *, label):
    if not isinstance(checkpoint, collections.abc.Mapping):
        raise ValueError(f"{label} must be a mapping")
    pending = [checkpoint]
    seen = set()
    while pending:
        value = pending.pop()
        if not isinstance(
            value, (collections.abc.Mapping, list, tuple)
        ):
            continue
        value_id = id(value)
        if value_id in seen:
            continue
        seen.add(value_id)
        if isinstance(value, collections.abc.Mapping):
            present = [
                name
                for name in _VOC_GATE_POLICY_SCHEMA10_DERIVED_IDENTITY_KEYS
                if name in value
            ]
            if present:
                raise ValueError(
                    f"{label} persists reserved schema-10 derived identity "
                    f"keys {present!r}"
                )
            pending.extend(value.values())
        else:
            pending.extend(value)


_VOC_GATE_POLICY_SCHEMA11_DERIVED_IDENTITY_KEYS = (
    "voc_q_regression_loss",
    "voc_q_reconstruction",
    "voc_q_optimizer_coordinates",
)


def _reject_schema11_persisted_derived_identity(checkpoint, *, label):
    if not isinstance(checkpoint, collections.abc.Mapping):
        raise ValueError(f"{label} must be a mapping")
    pending = [checkpoint]
    seen = set()
    while pending:
        value = pending.pop()
        if not isinstance(
            value, (collections.abc.Mapping, list, tuple)
        ):
            continue
        value_id = id(value)
        if value_id in seen:
            continue
        seen.add(value_id)
        if isinstance(value, collections.abc.Mapping):
            present = [
                name
                for name in _VOC_GATE_POLICY_SCHEMA11_DERIVED_IDENTITY_KEYS
                if name in value
            ]
            if present:
                raise ValueError(
                    f"{label} persists reserved schema-11 derived identity "
                    f"keys {present!r}"
                )
            pending.extend(value.values())
        else:
            pending.extend(value)


_VOC_GATE_POLICY_SCHEMA12_DERIVED_IDENTITY_KEYS = (
    "voc_q_regression_loss",
    "voc_q_reconstruction",
    "voc_q_optimizer_coordinates",
)


def _reject_schema12_persisted_derived_identity(checkpoint, *, label):
    if not isinstance(checkpoint, collections.abc.Mapping):
        raise ValueError(f"{label} must be a mapping")
    pending = [checkpoint]
    seen = set()
    while pending:
        value = pending.pop()
        if not isinstance(
            value, (collections.abc.Mapping, list, tuple)
        ):
            continue
        value_id = id(value)
        if value_id in seen:
            continue
        seen.add(value_id)
        if isinstance(value, collections.abc.Mapping):
            present = [
                name
                for name in _VOC_GATE_POLICY_SCHEMA12_DERIVED_IDENTITY_KEYS
                if name in value
            ]
            if present:
                raise ValueError(
                    f"{label} persists reserved schema-12 derived identity "
                    f"keys {present!r}"
                )
            pending.extend(value.values())
        else:
            pending.extend(value)


_VOC_GATE_POLICY_SCHEMA13_DERIVED_IDENTITY_KEYS = (
    "voc_q_regression_loss",
    "voc_q_reconstruction",
    "voc_q_optimizer_coordinates",
)


def _reject_schema13_persisted_derived_identity(checkpoint, *, label):
    if not isinstance(checkpoint, collections.abc.Mapping):
        raise ValueError(f"{label} must be a mapping")
    pending = [checkpoint]
    seen = set()
    while pending:
        value = pending.pop()
        if not isinstance(
            value, (collections.abc.Mapping, list, tuple)
        ):
            continue
        value_id = id(value)
        if value_id in seen:
            continue
        seen.add(value_id)
        if isinstance(value, collections.abc.Mapping):
            present = [
                name
                for name in _VOC_GATE_POLICY_SCHEMA13_DERIVED_IDENTITY_KEYS
                if name in value
            ]
            if present:
                raise ValueError(
                    f"{label} persists reserved schema-13 derived identity "
                    f"keys {present!r}"
                )
            pending.extend(value.values())
        else:
            pending.extend(value)


def _validate_schema12_raw_ema_online_equality(checkpoint, *, label):
    """Bind a nonzero-update schema-12 EMA to its stored online raw head."""

    if not isinstance(checkpoint, collections.abc.Mapping):
        raise ValueError(f"{label} must be a mapping")
    update_count = checkpoint.get("voc_ema_gate_update_count")
    if type(update_count) is not int or update_count < 0:
        raise ValueError(
            f"{label} requires exact non-negative Python integer "
            "voc_ema_gate_update_count"
        )
    top_tau = checkpoint.get("voc_gate_target_tau")
    if type(top_tau) is not float or not np.isfinite(top_tau) or top_tau != 1.0:
        raise ValueError(
            f"{label} requires top-level voc_gate_target_tau=1.0 as an "
            f"exact built-in float; got {top_tau!r}"
        )
    if update_count == 0:
        return
    ema_state = checkpoint.get("voc_ema_gate_head_state_dict")
    online_state = checkpoint.get("actor_net_state_dict")
    if not isinstance(ema_state, collections.abc.Mapping) or set(ema_state) != {
        "weight",
        "bias",
    }:
        raise ValueError(f"{label} lacks exact raw EMA weight/bias state")
    if not isinstance(online_state, collections.abc.Mapping):
        raise ValueError(f"{label} lacks online raw Q state")
    matched_pair = next(
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
    if matched_pair is None:
        raise ValueError(f"{label} lacks online raw Q weight/bias")
    for ema_name, online_name in zip(("weight", "bias"), matched_pair):
        ema_tensor = ema_state[ema_name]
        online_tensor = online_state[online_name]
        if (
            not isinstance(ema_tensor, torch.Tensor)
            or not isinstance(online_tensor, torch.Tensor)
            or not torch.equal(ema_tensor, online_tensor)
        ):
            raise ValueError(
                f"{label} schema-12 raw EMA {ema_name} disagrees with "
                "the stored online raw Q head"
            )


def _validate_schema13_raw_ema_online_equality(checkpoint, *, label):
    """Bind a nonzero-update schema-13 EMA to its stored online raw head."""

    if not isinstance(checkpoint, collections.abc.Mapping):
        raise ValueError(f"{label} must be a mapping")
    update_count = checkpoint.get("voc_ema_gate_update_count")
    if type(update_count) is not int or update_count < 0:
        raise ValueError(
            f"{label} requires exact non-negative Python integer "
            "voc_ema_gate_update_count"
        )
    top_tau = checkpoint.get("voc_gate_target_tau")
    if type(top_tau) is not float or not np.isfinite(top_tau) or top_tau != 1.0:
        raise ValueError(
            f"{label} requires top-level voc_gate_target_tau=1.0 as an "
            f"exact built-in float; got {top_tau!r}"
        )
    if update_count == 0:
        return
    ema_state = checkpoint.get("voc_ema_gate_head_state_dict")
    online_state = checkpoint.get("actor_net_state_dict")
    if not isinstance(ema_state, collections.abc.Mapping) or set(ema_state) != {
        "weight",
        "bias",
    }:
        raise ValueError(f"{label} lacks exact raw EMA weight/bias state")
    if not isinstance(online_state, collections.abc.Mapping):
        raise ValueError(f"{label} lacks online raw Q state")
    matched_pair = next(
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
    if matched_pair is None:
        raise ValueError(f"{label} lacks online raw Q weight/bias")
    for ema_name, online_name in zip(("weight", "bias"), matched_pair):
        ema_tensor = ema_state[ema_name]
        online_tensor = online_state[online_name]
        if (
            not isinstance(ema_tensor, torch.Tensor)
            or not isinstance(online_tensor, torch.Tensor)
            or not torch.equal(ema_tensor, online_tensor)
        ):
            raise ValueError(
                f"{label} schema-13 raw EMA {ema_name} disagrees with "
                "the stored online raw Q head"
            )


def _validate_voc_atomic_final_actor_checkpoint(
    checkpoint, flags, *, expected_schema, label
):
    if not isinstance(checkpoint, collections.abc.Mapping):
        raise ValueError(f"{label} must be a mapping")
    if expected_schema == VOC_GATE_POLICY_VERSION_BARRIER_SCHEMA_VERSION:
        protocol_validator = _validate_schema6_protocol_flags
    elif expected_schema == VOC_GATE_POLICY_MODEL_INPUT_SEAL_SCHEMA_VERSION:
        protocol_validator = _validate_schema7_protocol_flags
    elif expected_schema == VOC_GATE_POLICY_HALF_SQUARED_Q_SCHEMA_VERSION:
        protocol_validator = _validate_schema8_protocol_flags
    elif expected_schema == VOC_GATE_POLICY_COMMON_MODE_Q_SCHEMA_VERSION:
        protocol_validator = _validate_schema9_protocol_flags
    elif expected_schema == VOC_GATE_POLICY_HUBER_COMMON_Q_SCHEMA_VERSION:
        protocol_validator = _validate_schema10_protocol_flags
    elif expected_schema == VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION:
        protocol_validator = _validate_schema11_protocol_flags
    elif expected_schema == VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION:
        protocol_validator = _validate_schema12_protocol_flags
    elif expected_schema == VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION:
        protocol_validator = _validate_schema13_protocol_flags
    else:
        raise ValueError(f"{label} has unsupported atomic schema")
    embedded = checkpoint.get("flags")
    if not isinstance(embedded, collections.abc.Mapping):
        raise ValueError(f"{label} lacks embedded training flags")
    protocol_validator(embedded, label=f"{label} flags")
    mode = embedded.get("dynamic_voc_mode")
    if mode != "control" or checkpoint.get("dynamic_voc_mode") != mode:
        raise ValueError(f"{label} requires exact top-level control mode")
    run_mapping = vars(flags) if isinstance(flags, argparse.Namespace) else flags
    protocol_validator(run_mapping, label=f"{label} run flags")
    protocol = get_voc_protocol(flags)
    if protocol["dynamic_voc_mode"] != "control":
        raise ValueError(f"{label} run flags require control mode")
    if expected_schema == VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION:
        _validate_schema12_raw_ema_online_equality(checkpoint, label=label)
    elif expected_schema == VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION:
        _validate_schema13_raw_ema_online_equality(checkpoint, label=label)
    return _validate_voc_active_checkpoint_state(
        checkpoint, flags, protocol, label=label
    )


def validate_voc_control_preload(checkpoint_path, *, checkpoint=None, flags=None):
    """Validate and fingerprint a shadow checkpoint promoted to control.

    This check intentionally happens at weight-preload time, not generic flag
    parsing: unit construction and checkpoint inspection need not have a file,
    while a real control promotion must be tied to one learned shadow snapshot.
    """

    path = os.path.abspath(os.path.expanduser(os.fspath(checkpoint_path)))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"VoC shadow parent does not exist: {path}")
    # Bind validation, content and digest to one immutable byte snapshot.  A
    # caller-provided decoded mapping is deliberately not trusted for identity:
    # the on-disk bytes are the promotion artifact that provenance records.
    with open(path, "rb") as handle:
        checkpoint_bytes = handle.read()
    digest = hashlib.sha256(checkpoint_bytes).hexdigest()
    checkpoint = torch.load(
        io.BytesIO(checkpoint_bytes),
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    if not isinstance(checkpoint, collections.abc.Mapping):
        raise ValueError("VoC control parent checkpoint must be a mapping")
    embedded = checkpoint.get("flags", {})
    if not isinstance(embedded, collections.abc.Mapping):
        raise ValueError("VoC control parent lacks embedded training flags")
    embedded_mode = embedded.get("dynamic_voc_mode", "off")
    top_level_mode = checkpoint.get("dynamic_voc_mode", embedded_mode)
    if top_level_mode != embedded_mode:
        raise ValueError(
            "VoC parent top-level mode disagrees with embedded flags: "
            f"{top_level_mode!r} versus {embedded_mode!r}"
        )
    if embedded_mode != "shadow":
        raise ValueError(
            "VoC control preload requires a shadow checkpoint; got "
            f"{embedded_mode!r}"
        )
    validate_voc_shadow_checkpoint_provenance(
        checkpoint, label="VoC shadow parent"
    )
    if flags is not None:
        for name in (
            "name",
            "icopro_game_id",
            "frame_stack_n",
            "grayscale",
            "wrapper_type",
        ):
            saved = embedded.get(name)
            expected = getattr(flags, name, None)
            if saved != expected:
                raise ValueError(
                    "VoC shadow parent identity mismatch: "
                    f"{name}={saved!r}, expected {expected!r}"
                )

        def canonical_int_list(value, name):
            if isinstance(value, (tuple, list)):
                items = value
            else:
                items = str(value).split(",")
            try:
                result = tuple(
                    int(item) for item in items if str(item).strip()
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"VoC shadow parent has invalid {name}={value!r}"
                ) from error
            if not result:
                raise ValueError(
                    f"VoC shadow parent has empty {name}"
                )
            return result

        for name in (
            "icopro_subjects",
            "icopro_train_sessions",
            "icopro_holdout_sessions",
        ):
            saved = canonical_int_list(embedded.get(name), name)
            expected = canonical_int_list(getattr(flags, name, None), name)
            if saved != expected:
                raise ValueError(
                    "VoC shadow parent identity mismatch: "
                    f"{name}={saved!r}, expected {expected!r}"
                )
    gate_schema = validate_voc_gate_policy_schema(
        checkpoint, label="VoC shadow parent"
    )
    for name in (
        "voc_ema_gate_target",
        "voc_gate_target_tau",
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
    ):
        if name not in embedded and name not in (
            "voc_gate_adam_beta1",
            "voc_gate_param_align",
            "voc_gate_param_align_coef",
            "voc_gate_exact_projection",
            "voc_gate_epsilon_greedy_execution",
        ):
            raise ValueError(f"VoC shadow parent lacks embedded {name}")
    expected_protocol = dict(
        VOC_PROTOCOL_DEFAULTS if flags is None else get_voc_protocol(flags)
    )
    if flags is None:
        # Legacy schema-3 alignment metadata may be true in shadow mode.  The
        # full gate validator below rejects schema-4 exact projection in
        # shadow mode even when this metadata-only inspection has no flags.
        expected_protocol["voc_gate_param_align"] = gate_schema[
            "voc_gate_param_align"
        ]
        expected_protocol["voc_gate_param_align_coef"] = gate_schema[
            "voc_gate_param_align_coef"
        ]
        expected_protocol["voc_gate_exact_projection"] = gate_schema[
            "voc_gate_exact_projection"
        ]
        expected_protocol["voc_gate_epsilon_greedy_execution"] = gate_schema[
            "voc_gate_epsilon_greedy_execution"
        ]
    _require_voc_ema_gate_protocol(
        embedded["voc_ema_gate_target"],
        embedded["voc_gate_target_tau"],
        label="VoC shadow parent checkpoint",
    )
    _require_voc_ema_gate_protocol(
        expected_protocol["voc_ema_gate_target"],
        expected_protocol["voc_gate_target_tau"],
        label="VoC control promotion configuration",
    )
    _require_voc_gate_policy_protocol(
        embedded.get("voc_dedicated_gate"),
        embedded.get("voc_soft_q_bce_gate"),
        embedded.get("voc_gate_q_temperature"),
        embedded.get("voc_gate_confidence_weighted"),
        gate_schema["voc_gate_adam_beta1"],
        embedded.get("voc_gate_learning_rate"),
        embedded.get("voc_gate_grad_norm_clipping"),
        gate_schema["voc_gate_param_align"],
        gate_schema["voc_gate_param_align_coef"],
        gate_schema["voc_gate_exact_projection"],
        gate_schema["voc_gate_epsilon_greedy_execution"],
        label="VoC shadow parent checkpoint",
    )
    _require_voc_gate_policy_protocol(
        expected_protocol["voc_dedicated_gate"],
        expected_protocol["voc_soft_q_bce_gate"],
        expected_protocol["voc_gate_q_temperature"],
        expected_protocol["voc_gate_confidence_weighted"],
        expected_protocol["voc_gate_adam_beta1"],
        expected_protocol["voc_gate_learning_rate"],
        expected_protocol["voc_gate_grad_norm_clipping"],
        expected_protocol["voc_gate_param_align"],
        expected_protocol["voc_gate_param_align_coef"],
        expected_protocol["voc_gate_exact_projection"],
        expected_protocol["voc_gate_epsilon_greedy_execution"],
        label="VoC control promotion configuration",
    )
    for name, default in VOC_PROTOCOL_DEFAULTS.items():
        if name == "dynamic_voc_mode":
            continue
        if name not in embedded:
            if name in (
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
                "voc_model_input_seal_schema_version",
                "actor_amp_init_scale",
            ):
                if (
                    name == "voc_model_input_seal_schema_version"
                    and gate_schema["voc_gate_policy_schema_version"]
                    != VOC_GATE_POLICY_MODEL_INPUT_SEAL_SCHEMA_VERSION
                ):
                    # Keep legacy shadow-parent validation compatible with
                    # checkpoints written before the schema-7 seal field.
                    saved = 0
                else:
                    saved = gate_schema[name]
            else:
                raise ValueError(f"VoC shadow parent lacks embedded {name}")
        else:
            saved = embedded[name]
        expected = expected_protocol[name]
        if name == "entropy_r_cost":
            saved = _require_environment_return_only_voc(
                saved, label="VoC shadow parent checkpoint"
            )
            expected = _require_environment_return_only_voc(
                expected, label="VoC control promotion configuration"
            )
        if name in (
            "voc_gate_adam_beta1",
            "voc_gate_param_align_coef",
        ):
            matches = (
                not isinstance(saved, (bool, np.bool_))
                and isinstance(saved, (int, float, np.number))
                and np.isfinite(saved)
                and float(saved) == float(expected)
            )
        elif isinstance(default, bool):
            matches = (
                isinstance(saved, (bool, np.bool_))
                and bool(saved) == bool(expected)
            )
        else:
            matches = (
                not isinstance(saved, (bool, np.bool_))
                and isinstance(saved, (int, float, np.number))
                and np.isfinite(saved)
                and float(saved) == float(expected)
            )
        if not matches:
            raise ValueError(
                f"VoC shadow parent {name}={saved!r}, expected {expected!r}"
            )
    for name, expected in {
        "dynamic_search": True,
        "dynamic_factorized_control": True,
        "think_cost": 0.0005,
        "think_cost_anneal": False,
    }.items():
        saved = embedded.get(name)
        if isinstance(expected, bool):
            matches = isinstance(saved, (bool, np.bool_)) and bool(saved) == expected
        else:
            matches = (
                not isinstance(saved, (bool, np.bool_))
                and isinstance(saved, (int, float, np.number))
                and np.isfinite(saved)
                and abs(float(saved) - expected) <= 1e-12
            )
        if not matches:
            raise ValueError(
                f"VoC shadow parent {name}={saved!r}, expected {expected!r}"
            )

    counters = {}
    for key in ("voc_update_count", "voc_continue_count", "voc_stop_count"):
        value = checkpoint.get(key)
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
        ):
            raise ValueError(f"VoC shadow parent has invalid {key}")
        value = int(value)
        if value <= 0:
            raise ValueError(
                f"VoC shadow parent must have positive {key}; got {value}"
            )
        counters[key] = value

    holdout_calibration = validate_voc_holdout_calibration(
        checkpoint, label="VoC shadow parent", require_positive_support=True
    )
    holdout_split = validate_voc_holdout_split(
        checkpoint, flags=flags, label="VoC shadow parent"
    )

    if checkpoint.get("voc_parent_checkpoint_sha256") is not None:
        raise ValueError(
            "VoC shadow parent must have voc_parent_checkpoint_sha256=null"
        )
    if checkpoint.get("voc_activation_real_step", -1) != -1:
        raise ValueError(
            "VoC shadow parent must have voc_activation_real_step=-1"
        )
    data_signature = checkpoint.get("imitation_data_signature")
    if (
        not isinstance(data_signature, str)
        or len(data_signature) != 64
        or any(c not in "0123456789abcdef" for c in data_signature.lower())
    ):
        raise ValueError(
            "VoC shadow parent lacks a valid imitation_data_signature"
        )
    matched_pair = validate_voc_checkpoint_components(
        checkpoint, flags=flags, label="VoC shadow parent"
    )
    ema_gate_state = validate_voc_ema_gate_checkpoint(
        checkpoint, label="VoC shadow parent"
    )
    amp_state = validate_voc_amp_checkpoint(
        checkpoint, label="VoC shadow parent"
    )
    gate_policy_state = validate_voc_gate_policy_checkpoint(
        checkpoint, flags=flags, label="VoC shadow parent"
    )

    return {
        "dynamic_voc_mode": "shadow",
        "voc_control_origin": VOC_CONTROL_ORIGIN_SHADOW_PARENT,
        **counters,
        **holdout_calibration,
        **holdout_split,
        **amp_state,
        **ema_gate_state,
        **gate_policy_state,
        "voc_head_keys": matched_pair,
        "voc_parent_checkpoint_sha256": digest,
        "voc_parent_checkpoint": path,
        "imitation_data_signature": data_signature,
        # Internal handoff lets self-play load the exact bytes whose digest and
        # EMA provenance were validated, eliminating a validate/load race.
        "_validated_checkpoint": checkpoint,
    }


def get_reward_names(flags):
    """Canonical actor reward-channel order.

    The Dynamic computation channel is appended so legacy reward indices stay
    stable during partial actor-checkpoint migration.
    """
    names = ["re"]
    if flags.im_cost > 0.0:
        names.append("im")
    if flags.cur_cost > 0.0:
        names.append("cur")
    if dynamic_search_enabled(flags):
        names.append("think")
    return names


def get_search_budget_stats(search_steps, stage_end):
    """Summarize imagination actions for completed Dynamic search stages.

    ``search_steps`` counts accepted PROCEED/RESET actions in a stage.  STOP,
    real-action storage, and WAIT calls are therefore excluded.  Incomplete
    stages are excluded as well so an unroll boundary cannot shorten a budget.
    """
    search_steps = torch.as_tensor(search_steps)
    stage_end = torch.as_tensor(stage_end, device=search_steps.device).bool()
    ended_steps = search_steps[stage_end].float()
    budget_bins = (
        ("0", ended_steps == 0),
        ("1", ended_steps == 1),
        ("2_3", (ended_steps >= 2) & (ended_steps <= 3)),
        ("4_7", (ended_steps >= 4) & (ended_steps <= 7)),
        ("8_15", (ended_steps >= 8) & (ended_steps <= 15)),
        # Dynamic production uses a finite watchdog, so this is the 16-to-cap
        # bin.  It remains well-defined as 16+ for an uncapped legacy run.
        ("16_cap", ended_steps >= 16),
    )
    bin_stats = {}
    stage_n = int(ended_steps.numel())
    for label, mask in budget_bins:
        count = int(mask.sum().item())
        bin_stats[f"search/budget_bin_{label}_count"] = count
        bin_stats[f"search/budget_bin_{label}_fraction"] = (
            count / stage_n if stage_n > 0 else 0.0
        )

    if ended_steps.numel() == 0:
        return {
            "max_budget": 0.0,
            "mean_budget": 0.0,
            "search/mean_steps": 0.0,
            "search/median_steps": 0.0,
            "search/p95_steps": 0.0,
            **bin_stats,
        }

    mean_budget = ended_steps.mean().item()
    return {
        "max_budget": ended_steps.max().item(),
        "mean_budget": mean_budget,
        # Keep the existing names for dashboard/checkpoint-log compatibility.
        "search/mean_steps": mean_budget,
        "search/median_steps": ended_steps.median().item(),
        "search/p95_steps": torch.quantile(ended_steps, 0.95).item(),
        **bin_stats,
    }


def get_search_depth_stop_stats(
        search_steps, search_control, control_valid, stop_probability):
    """Summarize behavior-policy STOP probability by decision depth.

    Dynamic environment fields are post-step while ``control_valid`` and
    ``search_control`` describe the action accepted on that row.  A
    PROCEED/RESET increments ``search_steps``; STOP does not.  Subtracting one
    only for accepted non-STOP controls therefore recovers the pre-decision
    depth without reclassifying WAIT or forced rows.

    Every field is always present.  Empty bins use count 0 and probability
    0.0 so CSV schemas remain stable across unrolls.
    """

    search_steps = torch.as_tensor(search_steps)
    search_control = torch.as_tensor(
        search_control, device=search_steps.device
    )
    control_valid = torch.as_tensor(
        control_valid, device=search_steps.device
    ).bool()
    stop_probability = torch.as_tensor(
        stop_probability, device=search_steps.device
    )
    expected_shape = tuple(search_steps.shape)
    for name, value in (
        ("search_control", search_control),
        ("control_valid", control_valid),
        ("stop_probability", stop_probability),
    ):
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"{name} must match search_steps shape {expected_shape}, "
                f"got {tuple(value.shape)}"
            )

    decision_depth = search_steps.long() - (
        control_valid & (search_control != STOP)
    ).long()
    valid_depth = decision_depth[control_valid].float()
    valid_stop_probability = stop_probability[control_valid].float()
    if torch.any(valid_depth < 0):
        raise ValueError("valid Dynamic control has negative decision depth")

    depth_bins = (
        ("0", valid_depth == 0),
        ("1", valid_depth == 1),
        ("2_3", (valid_depth >= 2) & (valid_depth <= 3)),
        ("4_7", (valid_depth >= 4) & (valid_depth <= 7)),
        ("8_15", (valid_depth >= 8) & (valid_depth <= 15)),
        ("16_plus", valid_depth >= 16),
    )
    stats = {}
    for label, mask in depth_bins:
        count = int(mask.sum().item())
        stats[f"search/depth_bin_{label}_count"] = count
        stats[f"search/depth_bin_{label}_stop_probability"] = (
            valid_stop_probability[mask].mean().item() if count > 0 else 0.0
        )

    sample_n = int(valid_depth.numel())
    slope = 0.0
    if sample_n >= 2:
        centered_depth = valid_depth - valid_depth.mean()
        denominator = torch.sum(centered_depth.square())
        if denominator.item() > 0.0:
            centered_probability = (
                valid_stop_probability - valid_stop_probability.mean()
            )
            slope = (
                torch.sum(centered_depth * centered_probability) / denominator
            ).item()
    stats["search/depth_stop_probability_slope"] = slope
    stats["search/depth_stop_probability_count"] = sample_n
    return stats

def init_env_out(state, info, flags, dim_actions, tuple_action):
    # minimum env_out for actor_net
    num_rewards = len(get_reward_names(flags))

    env_n = state["real_states"].shape[0]
    device = state["real_states"].device

    last_pri_shape = (env_n, dim_actions) if tuple_action else (env_n)
    out = {
        "last_pri": torch.zeros(last_pri_shape, dtype=torch.long, device=device),
        "last_reset": torch.zeros(env_n, dtype=torch.long, device=device),
        "last_search_control": torch.zeros(env_n, dtype=torch.long, device=device),
        "reward": torch.zeros((env_n, num_rewards), 
                            dtype=torch.float, device=device),
        "done": torch.zeros(env_n, dtype=torch.bool, device=device),
        "truncated_done": torch.zeros(env_n, dtype=torch.long, device=device),
    }

    if dynamic_search_enabled(flags):
        out.update({
            "phase": torch.full((env_n,), SEARCH_PHASE, dtype=torch.long, device=device),
            "legal_control_mask": torch.ones((env_n, 3), dtype=torch.bool, device=device),
            "tree_token_valid": torch.ones(env_n, dtype=torch.bool, device=device),
            "search_state_reset": torch.ones(env_n, dtype=torch.bool, device=device),
            "real_transition": torch.zeros(env_n, dtype=torch.bool, device=device),
            "root_carried": torch.zeros(env_n, dtype=torch.bool, device=device),
            "carried_descendant_visit_count": torch.zeros(
                env_n, dtype=torch.long, device=device
            ),
            "carried_descendant_expanded_count": torch.zeros(
                env_n, dtype=torch.long, device=device
            ),
            "useful_carry": torch.zeros(
                env_n, dtype=torch.bool, device=device
            ),
            "stage_end": torch.zeros(env_n, dtype=torch.bool, device=device),
            "forced_stop": torch.zeros(env_n, dtype=torch.bool, device=device),
            "search_steps": torch.zeros(env_n, dtype=torch.long, device=device),
        })

    # State/info are authoritative.  In particular, the Dynamic defaults
    # above describe only the reset observation; they must not shadow phase
    # and mask values returned by the wrapper.
    for field in EnvOut._fields:
        if field in state and state[field] is not None:
            out[field] = state[field]
        if field in info and info[field] is not None:
            out[field] = info[field]
        if field not in out:
            out[field] = None

    for k, v in out.items():
        if v is not None:
            out[k] = torch.unsqueeze(v, dim=0)
    env_out = EnvOut(**out)        
    return env_out     

def create_env_out(action, state, reward, done, truncated_done, info, flags):
    
    aug_reward = [reward]
    if flags.im_cost > 0:
        aug_reward.append(info["im_reward"][:, 0])
    if flags.cur_cost > 0:
        aug_reward.append(info["cur_reward"])
    if dynamic_search_enabled(flags):
        think_reward = info.get("think_reward")
        if think_reward is None:
            think_reward = torch.zeros_like(reward)
        if think_reward.ndim > reward.ndim:
            think_reward = think_reward[..., 0]
        aug_reward.append(think_reward)
    aug_reward = torch.stack(aug_reward, dim=-1)

    if 'episode_return' in info:
        aug_epsoide_return = [info['episode_return']]
        if flags.im_cost > 0:
            aug_epsoide_return.append(info["im_episode_return"])
        if flags.cur_cost > 0:
            aug_epsoide_return.append(info["cur_episode_return"])
        if dynamic_search_enabled(flags):
            think_episode_return = info.get("think_episode_return")
            if think_episode_return is None:
                think_episode_return = torch.zeros_like(info["episode_return"])
            aug_epsoide_return.append(think_episode_return)
        aug_epsoide_return = torch.stack(aug_epsoide_return, dim=-1)
    else:
        aug_epsoide_return = None
    
    out = {
        "reward": aug_reward, 
        "episode_return": aug_epsoide_return,
        "done": done,
        "truncated_done": truncated_done,           
    }
    if not flags.wrapper_type == 1:    
        last_pri = action[0]
        if dynamic_search_enabled(flags):
            effective_primary = info.get("effective_primary_action")
            if effective_primary is not None:
                last_pri = effective_primary.to(last_pri.device)
            elif info.get("executed_primary_action") is not None:
                real_transition = info.get("real_transition")
                if real_transition is not None:
                    mask = real_transition.bool()
                    executed = info["executed_primary_action"].to(last_pri.device)
                    if last_pri.ndim > mask.ndim:
                        mask = mask.unsqueeze(-1)
                    last_pri = torch.where(mask, executed, last_pri)
        out["last_pri"] = last_pri
        out["last_reset"] = action[1]
        effective_control = info.get("effective_search_control")
        out["last_search_control"] = (
            effective_control.to(action[1].device)
            if dynamic_search_enabled(flags) and effective_control is not None
            else action[1]
        )
    else:
        out["last_pri"] = action

    for field in EnvOut._fields:    
        if field not in out:
            out[field] = None
        else:
            continue
        if field in state.keys():
            out[field] = state[field]
        if field in info.keys():
            out[field] = info[field]
    
    for k, v in out.items():
        if v is not None:
            out[k] = torch.unsqueeze(v, dim=0)
    env_out = EnvOut(**out)
    return env_out    

def process_flags(flags):
    # Defaults are needed when resuming a configuration written by an older
    # Thinker version.
    initial_seal_schema = getattr(
        flags, "voc_model_input_seal_schema_version", 0
    )
    if type(initial_seal_schema) is not int or initial_seal_schema not in (0, 1):
        raise ValueError(
            "voc_model_input_seal_schema_version must be exact Python "
            f"integer 0 or 1; got {initial_seal_schema!r}"
        )
    initial_policy_barrier = getattr(
        flags,
        "voc_actor_policy_version_barrier",
        VOC_PROTOCOL_DEFAULTS["voc_actor_policy_version_barrier"],
    )
    if isinstance(initial_policy_barrier, np.bool_) and bool(
        initial_policy_barrier
    ):
        raise ValueError(
            "atomic voc_actor_policy_version_barrier must be a Python bool"
        )
    initial_atomic_requested = (
        type(initial_policy_barrier) is bool and initial_policy_barrier
    )
    initial_gate_schema = None
    if initial_atomic_requested:
        xpid = getattr(flags, "xpid", None)
        if (
            initial_seal_schema == VOC_MODEL_INPUT_SEAL_SCHEMA_VERSION
            and _schema13_stage_xpid_candidate(xpid)
        ):
            inferred_schema = VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION
        elif (
            initial_seal_schema == VOC_MODEL_INPUT_SEAL_SCHEMA_VERSION
            and _schema12_stage_xpid_candidate(xpid)
        ):
            inferred_schema = VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION
        elif (
            initial_seal_schema == VOC_MODEL_INPUT_SEAL_SCHEMA_VERSION
            and _schema11_stage_xpid_candidate(xpid)
        ):
            inferred_schema = VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION
        elif (
            initial_seal_schema == VOC_MODEL_INPUT_SEAL_SCHEMA_VERSION
            and _schema10_stage_xpid_candidate(xpid)
        ):
            inferred_schema = VOC_GATE_POLICY_HUBER_COMMON_Q_SCHEMA_VERSION
        elif (
            initial_seal_schema == VOC_MODEL_INPUT_SEAL_SCHEMA_VERSION
            and _schema9_stage_xpid_candidate(xpid)
        ):
            inferred_schema = VOC_GATE_POLICY_COMMON_MODE_Q_SCHEMA_VERSION
        elif (
            initial_seal_schema == VOC_MODEL_INPUT_SEAL_SCHEMA_VERSION
            and type(xpid) is str
            and xpid in {
                profile[0]
                for profile in VOC_GATE_POLICY_SCHEMA8_STAGE_PROFILES
            }
        ):
            inferred_schema = VOC_GATE_POLICY_HALF_SQUARED_Q_SCHEMA_VERSION
        elif initial_seal_schema == VOC_MODEL_INPUT_SEAL_SCHEMA_VERSION:
            inferred_schema = VOC_GATE_POLICY_MODEL_INPUT_SEAL_SCHEMA_VERSION
        else:
            inferred_schema = VOC_GATE_POLICY_VERSION_BARRIER_SCHEMA_VERSION
        initial_gate_schema = getattr(
            flags,
            "voc_gate_policy_schema_version",
            inferred_schema,
        )
        if (
            initial_gate_schema == VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION
            and not _schema13_stage_xpid_candidate(xpid)
        ):
            raise ValueError(
                "atomic flags require voc_gate_policy_schema_version to be "
                "exact Python integer 6, 7, 8, 9, 10, 11, or 12; got "
                f"{initial_gate_schema!r}"
            )
        if (
            initial_gate_schema == VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION
            and not _schema12_stage_xpid_candidate(xpid)
        ):
            raise ValueError(
                "atomic flags require voc_gate_policy_schema_version to be "
                "exact Python integer 6, 7, 8, 9, 10, or 11; got "
                f"{initial_gate_schema!r}"
            )
        if (
            initial_gate_schema
            == VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION
            and not _schema11_stage_xpid_candidate(xpid)
        ):
            # Preserve the schema<=10 rejection path byte-for-byte: before
            # schema 11 existed, an explicit 11 on a legacy/V17 xpid was an
            # out-of-range gate schema, not a request to enter a new validator
            # and report a schema-11 stage mismatch.
            raise ValueError(
                "atomic flags require voc_gate_policy_schema_version to be "
                "exact Python integer 6, 7, 8, 9, or 10; got "
                f"{initial_gate_schema!r}"
            )
        if type(initial_gate_schema) is not int or (
            initial_gate_schema not in VOC_GATE_POLICY_ATOMIC_SCHEMA_VERSIONS
        ):
            raise ValueError(
                "atomic flags require voc_gate_policy_schema_version to be "
                "exact Python integer 6 through 13; got "
                f"{initial_gate_schema!r}"
            )
        expected_seal_schema = (
            VOC_MODEL_INPUT_SEAL_SCHEMA_VERSION
            if initial_gate_schema in (
                VOC_GATE_POLICY_MODEL_INPUT_SEAL_SCHEMA_VERSION,
                VOC_GATE_POLICY_HALF_SQUARED_Q_SCHEMA_VERSION,
                VOC_GATE_POLICY_COMMON_MODE_Q_SCHEMA_VERSION,
                VOC_GATE_POLICY_HUBER_COMMON_Q_SCHEMA_VERSION,
                VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION,
                VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION,
                VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION,
            )
            else 0
        )
        if initial_seal_schema != expected_seal_schema:
            raise ValueError(
                f"schema-{initial_gate_schema} requires "
                "voc_model_input_seal_schema_version="
                f"{expected_seal_schema}; got {initial_seal_schema!r}"
            )
    elif initial_seal_schema != 0:
        raise ValueError(
            "voc_model_input_seal_schema_version=1 requires the atomic "
            "actor-policy barrier"
        )
    if initial_atomic_requested:
        # argparse derives the type of ``model_float16`` from the legacy
        # string default (``inherit``), so even an explicit
        # ``--model_float16 False`` arrives here as the string ``"False"``.
        # Canonicalize that parser representation before the raw schema-6
        # identity check.  Keep numpy booleans and arbitrary values intact so
        # the strict check below still rejects them, and let ``inherit`` bind
        # to the already-resolved actor precision exactly as the legacy path
        # does later in this function.
        raw_model_float16 = getattr(flags, "model_float16", "inherit")
        if raw_model_float16 is None:
            flags.model_float16 = bool(getattr(flags, "float16", False))
        elif type(raw_model_float16) is str:
            normalized_model_float16 = raw_model_float16.strip().lower()
            if normalized_model_float16 == "inherit":
                flags.model_float16 = bool(getattr(flags, "float16", False))
            elif normalized_model_float16 in {"true", "false"}:
                flags.model_float16 = normalized_model_float16 == "true"
        raw_requirements = {
            **VOC_GATE_POLICY_SCHEMA6_ATOMIC_REQUIREMENTS,
            **VOC_GATE_POLICY_SCHEMA6_OPTIMIZER_REQUIREMENTS,
            **VOC_GATE_POLICY_SCHEMA6_ENDURO_REQUIREMENTS,
            "float16": True,
            "model_float16": False,
            "dual_net": True,
            "train_model": True,
            "schedule_total_steps": 100_000_000,
        }
        if initial_gate_schema in (
            VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION,
            VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION,
        ):
            raw_requirements["voc_gate_target_tau"] = 1.0
        for name, expected in raw_requirements.items():
            actual = getattr(flags, name, None)
            if isinstance(expected, bool):
                matches = type(actual) is bool and actual is expected
            elif isinstance(expected, str):
                matches = type(actual) is str and actual == expected
            elif isinstance(expected, int):
                matches = (
                    not isinstance(actual, (bool, np.bool_))
                    and isinstance(actual, (int, np.integer))
                    and int(actual) == expected
                )
            else:
                if (
                    initial_gate_schema in (
                        VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION,
                        VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION,
                    )
                    and name == "voc_gate_target_tau"
                ):
                    matches = (
                        type(actual) is float
                        and np.isfinite(actual)
                        and actual == 1.0
                    )
                else:
                    matches = (
                        not isinstance(actual, (bool, np.bool_))
                        and isinstance(actual, (int, float, np.number))
                        and np.isfinite(actual)
                        and float(actual) == expected
                    )
            if not matches:
                if isinstance(expected, bool):
                    raise ValueError(
                        f"schema-{initial_gate_schema} {name} must be a "
                        "Python bool equal to "
                        f"{expected!r}; got {actual!r}"
                    )
                raise ValueError(
                    f"schema-{initial_gate_schema} raw configuration "
                    "atomically requires "
                    f"{name}={expected!r}; got {actual!r}"
                )
    total_steps = getattr(flags, "total_steps", 100_000_000)
    if (
        isinstance(total_steps, (bool, np.bool_))
        or not isinstance(total_steps, (int, np.integer))
        or int(total_steps) <= 0
    ):
        raise ValueError(f"total_steps must be a positive integer; got {total_steps!r}")
    flags.total_steps = int(total_steps)
    schedule_total_steps = getattr(flags, "schedule_total_steps", -1)
    if schedule_total_steps is None:
        schedule_total_steps = -1
    if (
        isinstance(schedule_total_steps, (bool, np.bool_))
        or not isinstance(schedule_total_steps, (int, np.integer))
    ):
        raise ValueError(
            "schedule_total_steps must be -1 or a positive integer; got "
            f"{schedule_total_steps!r}"
        )
    schedule_total_steps = int(schedule_total_steps)
    if schedule_total_steps == -1:
        schedule_total_steps = flags.total_steps
    elif schedule_total_steps <= 0:
        raise ValueError(
            "schedule_total_steps must be -1 or a positive integer; got "
            f"{schedule_total_steps!r}"
        )
    flags.schedule_total_steps = schedule_total_steps
    actor_amp_max_consecutive_skips = getattr(
        flags, "actor_amp_max_consecutive_skips", 8
    )
    if (
        isinstance(actor_amp_max_consecutive_skips, (bool, np.bool_))
        or not isinstance(actor_amp_max_consecutive_skips, (int, np.integer))
        or int(actor_amp_max_consecutive_skips) <= 0
    ):
        raise ValueError(
            "actor_amp_max_consecutive_skips must be a positive integer; got "
            f"{actor_amp_max_consecutive_skips!r}"
        )
    flags.actor_amp_max_consecutive_skips = int(
        actor_amp_max_consecutive_skips
    )
    if not hasattr(flags, "dynamic_search"):
        flags.dynamic_search = False
    dynamic_factorized_control = getattr(
        flags, "dynamic_factorized_control", False
    )
    if not isinstance(dynamic_factorized_control, (bool, np.bool_)):
        raise ValueError(
            "dynamic_factorized_control must be boolean; got "
            f"{dynamic_factorized_control!r}"
        )
    flags.dynamic_factorized_control = bool(dynamic_factorized_control)
    if flags.dynamic_factorized_control and not flags.dynamic_search:
        raise ValueError(
            "dynamic_factorized_control requires dynamic_search=true"
        )
    dynamic_voc_mode = getattr(
        flags, "dynamic_voc_mode", VOC_PROTOCOL_DEFAULTS["dynamic_voc_mode"]
    )
    if not isinstance(dynamic_voc_mode, str) or dynamic_voc_mode not in VOC_MODES:
        raise ValueError(
            "dynamic_voc_mode must be exactly 'off', 'shadow', or 'control'; "
            f"got {dynamic_voc_mode!r}"
        )
    flags.dynamic_voc_mode = dynamic_voc_mode

    # Canonicalize optional preload surfaces before any origin decision.  In
    # particular, whitespace must never survive as a truthy filesystem path
    # after a fresh-control check classified it as empty.
    for name in ("preload", "preload_actor"):
        value = getattr(flags, name, "")
        if value is None:
            value = ""
        if not isinstance(value, str):
            raise ValueError(f"{name} must be a path string; got {value!r}")
        setattr(flags, name, value.strip())

    for name, lower_inclusive, upper_inclusive in (
        ("voc_loss_cost", 0.0, None),
        ("voc_gate_temperature", None, None),
        ("voc_train_epsilon", 0.0, 1.0),
        ("voc_gate_target_tau", 0.0, 1.0),
        ("voc_gate_q_temperature", 0.0, None),
        ("voc_gate_adam_beta1", 0.0, None),
        ("voc_gate_param_align_coef", None, None),
        ("voc_gate_execution_epsilon", 0.0, 1.0),
        ("voc_actor_policy_barrier_timeout_s", 0.0, None),
        ("actor_amp_init_scale", 0.0, None),
        ("voc_gate_learning_rate", 0.0, None),
        ("voc_gate_grad_norm_clipping", 0.0, None),
    ):
        value = getattr(flags, name, VOC_PROTOCOL_DEFAULTS[name])
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.number))
            or not np.isfinite(value)
        ):
            raise ValueError(f"{name} must be a finite number; got {value!r}")
        value = float(value)
        if name in (
            "voc_gate_temperature",
            "voc_gate_q_temperature",
            "voc_gate_learning_rate",
            "voc_gate_grad_norm_clipping",
            "actor_amp_init_scale",
            "voc_actor_policy_barrier_timeout_s",
        ) and value <= 0.0:
            raise ValueError(
                f"{name} must be positive; got {value!r}"
            )
        if name == "voc_gate_target_tau" and value <= 0.0:
            raise ValueError(
                f"voc_gate_target_tau must be positive; got {value!r}"
            )
        if name == "voc_gate_adam_beta1" and value >= 1.0:
            raise ValueError(
                f"voc_gate_adam_beta1 must be less than 1; got {value!r}"
            )
        if name == "voc_gate_param_align_coef" and value != 1.0:
            raise ValueError(
                "voc_gate_param_align_coef must equal 1.0 exactly; "
                f"got {value!r}"
            )
        if lower_inclusive is not None and value < lower_inclusive:
            raise ValueError(
                f"{name} must be at least {lower_inclusive}; got {value!r}"
            )
        if upper_inclusive is not None and value > upper_inclusive:
            raise ValueError(
                f"{name} must be at most {upper_inclusive}; got {value!r}"
            )
        setattr(flags, name, value)

    raw_policy_barrier = getattr(
        flags,
        "voc_actor_policy_version_barrier",
        VOC_PROTOCOL_DEFAULTS["voc_actor_policy_version_barrier"],
    )
    if isinstance(raw_policy_barrier, np.bool_) and bool(raw_policy_barrier):
        raise ValueError(
            "atomic voc_actor_policy_version_barrier must be a Python bool"
        )
    atomic_requested = type(raw_policy_barrier) is bool and raw_policy_barrier
    if atomic_requested != initial_atomic_requested:
        raise ValueError("atomic barrier changed during flag normalization")
    for name in (
        "voc_eval_stochastic",
        "voc_dueling_q",
        "voc_expected_gate_loss",
        "voc_ema_gate_target",
        "voc_dedicated_gate",
        "voc_soft_q_bce_gate",
        "voc_gate_confidence_weighted",
        "voc_gate_param_align",
        "voc_gate_exact_projection",
        "voc_gate_epsilon_greedy_execution",
        "voc_actor_policy_version_barrier",
    ):
        value = getattr(flags, name, VOC_PROTOCOL_DEFAULTS[name])
        if atomic_requested and name in VOC_GATE_POLICY_SCHEMA6_ATOMIC_REQUIREMENTS:
            expected = VOC_GATE_POLICY_SCHEMA6_ATOMIC_REQUIREMENTS[name]
            if isinstance(expected, bool) and type(value) is not bool:
                raise ValueError(
                    f"schema-{initial_gate_schema} {name} must be a Python "
                    f"bool; got {value!r}"
                )
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be boolean; got {value!r}")
        setattr(flags, name, bool(value))
    bundle_schema = getattr(
        flags,
        "voc_actor_policy_bundle_schema_version",
        VOC_ACTOR_POLICY_BUNDLE_SCHEMA_VERSION,
    )
    if (
        isinstance(bundle_schema, (bool, np.bool_))
        or not isinstance(bundle_schema, (int, np.integer))
        or int(bundle_schema) != VOC_ACTOR_POLICY_BUNDLE_SCHEMA_VERSION
    ):
        raise ValueError(
            "voc_actor_policy_bundle_schema_version must equal "
            f"{VOC_ACTOR_POLICY_BUNDLE_SCHEMA_VERSION} exactly; got "
            f"{bundle_schema!r}"
        )
    flags.voc_actor_policy_bundle_schema_version = int(bundle_schema)
    for name in (
        "voc_actor_policy_ray_max_restarts",
        "voc_actor_policy_ray_max_task_retries",
    ):
        value = getattr(flags, name, 0)
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) != 0
        ):
            raise ValueError(f"{name} must equal integer 0 exactly; got {value!r}")
        setattr(flags, name, int(value))
    voc_parent_checkpoint = getattr(flags, "voc_parent_checkpoint", "")
    if not isinstance(voc_parent_checkpoint, str):
        raise ValueError(
            "voc_parent_checkpoint must be a path string; got "
            f"{voc_parent_checkpoint!r}"
        )
    voc_parent_checkpoint = voc_parent_checkpoint.strip()
    if voc_parent_checkpoint:
        voc_parent_checkpoint = os.path.abspath(
            os.path.expanduser(voc_parent_checkpoint)
        )
        if flags.dynamic_voc_mode != "control":
            raise ValueError(
                "voc_parent_checkpoint is valid only in control mode"
            )
    flags.voc_parent_checkpoint = voc_parent_checkpoint

    if (
        flags.voc_gate_exact_projection
        and flags.dynamic_voc_mode != "control"
    ):
        raise ValueError(
            "voc_gate_exact_projection requires dynamic_voc_mode=control"
        )
    if flags.voc_gate_exact_projection:
        for name in ("preload", "preload_actor", "voc_parent_checkpoint"):
            if getattr(flags, name, "") != "":
                raise ValueError(
                    "voc_gate_exact_projection requires fresh parent-free "
                    f"{name}=''"
                )
    if flags.voc_gate_epsilon_greedy_execution:
        if flags.dynamic_voc_mode != "control":
            raise ValueError(
                "voc_gate_epsilon_greedy_execution requires "
                "dynamic_voc_mode=control"
            )
        if not flags.voc_gate_exact_projection:
            raise ValueError(
                "voc_gate_epsilon_greedy_execution requires "
                "voc_gate_exact_projection=true"
            )
        if flags.voc_gate_param_align:
            raise ValueError(
                "voc_gate_epsilon_greedy_execution requires "
                "voc_gate_param_align=false"
            )
    if flags.voc_actor_policy_version_barrier:
        configured_schema = initial_gate_schema
        flags.voc_gate_policy_schema_version = configured_schema
        flags.voc_model_input_seal_schema_version = initial_seal_schema
        required = (
            ("dynamic_voc_mode", flags.dynamic_voc_mode, "control"),
            ("voc_gate_exact_projection", flags.voc_gate_exact_projection, True),
            (
                "voc_gate_epsilon_greedy_execution",
                flags.voc_gate_epsilon_greedy_execution,
                True,
            ),
            ("voc_gate_param_align", flags.voc_gate_param_align, False),
            ("voc_gate_param_align_coef", flags.voc_gate_param_align_coef, 1.0),
            ("voc_train_epsilon", flags.voc_train_epsilon, 0.02),
            ("voc_gate_execution_epsilon", flags.voc_gate_execution_epsilon, 0.25),
            (
                "voc_actor_policy_bundle_schema_version",
                flags.voc_actor_policy_bundle_schema_version,
                VOC_ACTOR_POLICY_BUNDLE_SCHEMA_VERSION,
            ),
            (
                "voc_actor_policy_barrier_timeout_s",
                flags.voc_actor_policy_barrier_timeout_s,
                VOC_ACTOR_POLICY_BARRIER_TIMEOUT_SECONDS,
            ),
            ("actor_amp_init_scale", flags.actor_amp_init_scale, 32.0),
            ("voc_actor_policy_ray_max_restarts", flags.voc_actor_policy_ray_max_restarts, 0),
            ("voc_actor_policy_ray_max_task_retries", flags.voc_actor_policy_ray_max_task_retries, 0),
            (
                "voc_model_input_seal_schema_version",
                flags.voc_model_input_seal_schema_version,
                (
                    VOC_MODEL_INPUT_SEAL_SCHEMA_VERSION
                    if configured_schema in (
                        VOC_GATE_POLICY_MODEL_INPUT_SEAL_SCHEMA_VERSION,
                        VOC_GATE_POLICY_HALF_SQUARED_Q_SCHEMA_VERSION,
                        VOC_GATE_POLICY_COMMON_MODE_Q_SCHEMA_VERSION,
                        VOC_GATE_POLICY_HUBER_COMMON_Q_SCHEMA_VERSION,
                        VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION,
                        VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION,
                        VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION,
                    )
                    else 0
                ),
            ),
        )
        required += tuple(
            (
                name,
                getattr(flags, name),
                (
                    1.0
                    if configured_schema in (
                        VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION,
                        VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION,
                    )
                    and name == "voc_gate_target_tau"
                    else expected
                ),
            )
            for name, expected in VOC_GATE_POLICY_SCHEMA6_ATOMIC_REQUIREMENTS.items()
        )
        required += tuple(
            (name, getattr(flags, name, None), expected)
            for name, expected in VOC_GATE_POLICY_SCHEMA6_OPTIMIZER_REQUIREMENTS.items()
        )
        required += tuple(
            (name, getattr(flags, name, None), expected)
            for name, expected in VOC_GATE_POLICY_SCHEMA6_ENDURO_REQUIREMENTS.items()
        )
        for name, actual, expected in required:
            if isinstance(expected, bool):
                matches = type(actual) is bool and actual is expected
            elif isinstance(expected, str):
                matches = type(actual) is str and actual == expected
            elif isinstance(expected, int):
                matches = (
                    not isinstance(actual, (bool, np.bool_))
                    and isinstance(actual, (int, np.integer))
                    and int(actual) == expected
                )
            else:
                matches = (
                    not isinstance(actual, (bool, np.bool_))
                    and isinstance(actual, (int, float, np.number))
                    and np.isfinite(actual)
                    and float(actual) == expected
                )
            if not matches:
                raise ValueError(
                    f"schema-{configured_schema} actor-policy barrier "
                    "atomically requires "
                    f"{name}={expected!r}; got {actual!r}"
                )
        for name in ("ckp", "train_actor", "parallel_actor"):
            if type(getattr(flags, name, None)) is not bool:
                raise ValueError(
                    f"schema-{configured_schema} {name} must be a Python "
                    "bool; got "
                    f"{getattr(flags, name, None)!r}"
                )
        for name, expected in (
            ("ppo_k", 1),
            ("self_play_n", 1),
            ("env_n", 16),
            ("actor_batch_size", 16),
        ):
            value = getattr(flags, name, None)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or int(value) != expected
            ):
                raise ValueError(
                    f"schema-{configured_schema} requires {name}={expected} "
                    "exactly; got "
                    f"{value!r}"
                )
        if flags.train_actor:
            if flags.ckp:
                raise ValueError(
                    f"voc_actor_policy_version_barrier schema "
                    f"{configured_schema} is fresh-only "
                    "and forbids checkpoint resume"
                )
            for name in ("preload", "preload_actor", "voc_parent_checkpoint"):
                if getattr(flags, name, "") != "":
                    raise ValueError(
                        f"voc_actor_policy_version_barrier schema "
                        f"{configured_schema} is fresh-only "
                        f"and requires {name}=''"
                    )
            if not flags.parallel_actor:
                raise ValueError(
                    "voc_actor_policy_version_barrier training requires "
                    "parallel_actor=true"
                )
    else:
        if hasattr(flags, "voc_gate_policy_schema_version"):
            raise ValueError(
                "voc_gate_policy_schema_version is embedded only by atomic "
                "schemas"
            )
        legacy = (
            ("voc_gate_execution_epsilon", flags.voc_gate_execution_epsilon, 0.02),
            ("actor_amp_init_scale", flags.actor_amp_init_scale, 256.0),
            (
                "voc_actor_policy_barrier_timeout_s",
                flags.voc_actor_policy_barrier_timeout_s,
                VOC_ACTOR_POLICY_BARRIER_TIMEOUT_SECONDS,
            ),
            ("voc_actor_policy_ray_max_restarts", flags.voc_actor_policy_ray_max_restarts, 0),
            ("voc_actor_policy_ray_max_task_retries", flags.voc_actor_policy_ray_max_task_retries, 0),
        )
        for name, actual, expected in legacy:
            if actual != expected:
                raise ValueError(
                    f"{name}={actual!r} is legal only in atomic schemas; "
                    f"legacy schemas require {expected!r}"
                )
    flags.voc_actor_policy_barrier_runtime = bool(
        flags.voc_actor_policy_version_barrier
        and bool(getattr(flags, "train_actor", False))
        and bool(getattr(flags, "parallel_actor", False))
    )

    validate_voc_fresh_control_inputs(
        flags, label="dynamic_voc_mode control"
    )

    if flags.dynamic_voc_mode != "off":
        if not flags.dynamic_search or not flags.dynamic_factorized_control:
            raise ValueError(
                "dynamic_voc_mode shadow/control requires dynamic_search=true "
                "and dynamic_factorized_control=true"
            )
        if flags.voc_loss_cost <= 0.0:
            raise ValueError(
                "voc_loss_cost must be positive in shadow/control mode"
            )
        for name in (
            "voc_dueling_q",
            "voc_expected_gate_loss",
            "voc_ema_gate_target",
            "voc_dedicated_gate",
            "voc_soft_q_bce_gate",
        ):
            if not getattr(flags, name):
                raise ValueError(
                    "dynamic_voc_mode shadow/control requires "
                    f"{name}=true"
                )
        _, flags.voc_gate_target_tau = _require_voc_ema_gate_protocol(
            flags.voc_ema_gate_target,
            flags.voc_gate_target_tau,
            label="dynamic_voc_mode shadow/control",
        )
        (
            flags.voc_dedicated_gate,
            flags.voc_soft_q_bce_gate,
            flags.voc_gate_q_temperature,
            flags.voc_gate_confidence_weighted,
            flags.voc_gate_adam_beta1,
            flags.voc_gate_learning_rate,
            flags.voc_gate_grad_norm_clipping,
            flags.voc_gate_param_align,
            flags.voc_gate_param_align_coef,
            flags.voc_gate_exact_projection,
            flags.voc_gate_epsilon_greedy_execution,
        ) = _require_voc_gate_policy_protocol(
            flags.voc_dedicated_gate,
            flags.voc_soft_q_bce_gate,
            flags.voc_gate_q_temperature,
            flags.voc_gate_confidence_weighted,
            flags.voc_gate_adam_beta1,
            flags.voc_gate_learning_rate,
            flags.voc_gate_grad_norm_clipping,
            flags.voc_gate_param_align,
            flags.voc_gate_param_align_coef,
            flags.voc_gate_exact_projection,
            flags.voc_gate_epsilon_greedy_execution,
            label="dynamic_voc_mode shadow/control",
        )
        flags.entropy_r_cost = _require_environment_return_only_voc(
            getattr(
                flags,
                "entropy_r_cost",
                VOC_PROTOCOL_DEFAULTS["entropy_r_cost"],
            ),
            label="dynamic_voc_mode shadow/control",
        )
    model_float16 = getattr(flags, "model_float16", "inherit")
    if model_float16 is None or str(model_float16).strip().lower() == "inherit":
        flags.model_float16 = bool(getattr(flags, "float16", False))
    elif isinstance(model_float16, bool):
        flags.model_float16 = model_float16
    elif str(model_float16).strip().lower() in {"true", "false"}:
        flags.model_float16 = str(model_float16).strip().lower() == "true"
    else:
        raise ValueError(
            "model_float16 must be true, false, or inherit; got "
            f"{model_float16!r}"
        )
    if flags.voc_actor_policy_version_barrier:
        configured_schema = flags.voc_gate_policy_schema_version
        for name, expected in (
            ("float16", True),
            ("model_float16", False),
            ("dual_net", True),
            ("train_model", True),
        ):
            value = getattr(flags, name, None)
            if type(value) is not bool or value is not expected:
                raise ValueError(
                    f"schema-{configured_schema} atomic protocol requires "
                    f"{name}={expected!r} exactly; got {value!r}"
                )
        model_optimizer = getattr(flags, "model_optimizer", None)
        if type(model_optimizer) is not str or model_optimizer != "adam":
            raise ValueError(
                f"schema-{configured_schema} atomic protocol requires "
                f"model_optimizer='adam' exactly; got {model_optimizer!r}"
            )
        if flags.schedule_total_steps != 100_000_000:
            raise ValueError(
                f"schema-{configured_schema} atomic protocol requires "
                "schedule_total_steps=100000000 exactly; got "
                f"{flags.schedule_total_steps!r}"
            )
    model_state_projection = getattr(flags, "model_state_projection", "none")
    if (
        not isinstance(model_state_projection, str)
        or model_state_projection not in {"none", "clamp"}
    ):
        raise ValueError(
            "model_state_projection must be exactly 'none' or 'clamp'; got "
            f"{model_state_projection!r}"
        )
    flags.model_state_projection = model_state_projection
    model_state_range_loss_cost = getattr(
        flags, "model_state_range_loss_cost", 0.0
    )
    if (
        isinstance(model_state_range_loss_cost, (bool, np.bool_))
        or not isinstance(model_state_range_loss_cost, (int, float, np.number))
        or not np.isfinite(model_state_range_loss_cost)
        or float(model_state_range_loss_cost) < 0.0
    ):
        raise ValueError(
            "model_state_range_loss_cost must be a finite non-negative "
            f"number; got {model_state_range_loss_cost!r}"
        )
    flags.model_state_range_loss_cost = float(model_state_range_loss_cost)
    if (
        flags.model_state_projection == "none"
        and flags.model_state_range_loss_cost != 0.0
    ):
        raise ValueError(
            "model_state_range_loss_cost requires "
            "model_state_projection='clamp'"
        )
    if (
        flags.model_state_projection == "clamp"
        and int(getattr(flags, "model_decoder_depth", 0)) != 0
    ):
        raise ValueError(
            "model_state_projection='clamp' is valid only when "
            "model_decoder_depth=0"
        )
    if not hasattr(flags, "max_search_steps"):
        flags.max_search_steps = -1
    if not hasattr(flags, "think_cost"):
        flags.think_cost = 0.002
    if not hasattr(flags, "think_cost_anneal"):
        flags.think_cost_anneal = False
    if not hasattr(flags, "dynamic_search_hidden_dim"):
        flags.dynamic_search_hidden_dim = 100

    if flags.dynamic_search:
        if flags.wrapper_type not in [0, 2]:
            raise ValueError(
                "dynamic_search supports wrapper_type 0 (learned) and "
                "2 (perfect)"
            )
        if flags.reset_mode != 0:
            raise ValueError(
                "dynamic_search currently preserves reset_mode=0 semantics only"
            )
        if not (flags.max_search_steps == -1 or flags.max_search_steps > 0):
            raise ValueError(
                "max_search_steps must be -1 (unbounded) or a positive integer"
            )
        # Dynamic history is represented along the temporal token axis.
        flags.has_action_seq = False

    if flags.dynamic_voc_mode != "off":
        if not np.isclose(float(flags.think_cost), 0.0005, rtol=0.0, atol=1e-12):
            raise ValueError(
                "dynamic_voc_mode shadow/control requires fixed "
                f"think_cost=0.0005; got {flags.think_cost!r}"
            )
        if bool(flags.think_cost_anneal):
            raise ValueError(
                "dynamic_voc_mode shadow/control requires "
                "think_cost_anneal=false"
            )

    if flags.wrapper_type == 1:
        flags.rec_t = 1
        # flags.train_model = False
        flags.im_enable = False
        flags.cur_enable = False
        flags.return_h = False
        flags.return_double = False

    if check_perfect_model(flags.wrapper_type):
        flags.dual_net = False
        flags.cur_enable = False
        flags.model_rs_loss_cost = 0
        flags.model_img_loss_cost = 0
        flags.model_done_loss_cost = 0

    if flags.voc_actor_policy_version_barrier:
        if (
            flags.voc_gate_policy_schema_version
            == VOC_GATE_POLICY_VERSION_BARRIER_SCHEMA_VERSION
        ):
            if hasattr(flags, "voc_model_input_seal_schema_version"):
                delattr(flags, "voc_model_input_seal_schema_version")
            _validate_schema6_complete_surface(
                vars(flags), label="processed schema-6 flags"
            )
        elif (
            flags.voc_gate_policy_schema_version
            == VOC_GATE_POLICY_MODEL_INPUT_SEAL_SCHEMA_VERSION
        ):
            _validate_schema7_complete_surface(
                vars(flags), label="processed schema-7 flags"
            )
        elif (
            flags.voc_gate_policy_schema_version
            == VOC_GATE_POLICY_HALF_SQUARED_Q_SCHEMA_VERSION
        ):
            _validate_schema8_complete_surface(
                vars(flags), label="processed schema-8 flags"
            )
        elif (
            flags.voc_gate_policy_schema_version
            == VOC_GATE_POLICY_COMMON_MODE_Q_SCHEMA_VERSION
        ):
            _validate_schema9_complete_surface(
                vars(flags), label="processed schema-9 flags"
            )
        elif (
            flags.voc_gate_policy_schema_version
            == VOC_GATE_POLICY_HUBER_COMMON_Q_SCHEMA_VERSION
        ):
            _validate_schema10_complete_surface(
                vars(flags), label="processed schema-10 flags"
            )
        elif (
            flags.voc_gate_policy_schema_version
            == VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION
        ):
            _validate_schema11_complete_surface(
                vars(flags), label="processed schema-11 flags"
            )
        elif (
            flags.voc_gate_policy_schema_version
            == VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION
        ):
            _validate_schema12_complete_surface(
                vars(flags), label="processed schema-12 flags"
            )
        else:
            _validate_schema13_complete_surface(
                vars(flags), label="processed schema-13 flags"
            )
    elif hasattr(flags, "voc_model_input_seal_schema_version"):
        delattr(flags, "voc_model_input_seal_schema_version")

    assert flags.wrapper_type != 5, "wrapper-type 5 (meta-learning) not yet supported"

    return flags


def schedule_progress(flags, real_step):
    """Return clamped schedule progress without changing the stop horizon."""

    horizon = getattr(flags, "schedule_total_steps", None)
    if horizon is None:
        horizon = getattr(flags, "total_steps")
    if (
        isinstance(horizon, (bool, np.bool_))
        or not isinstance(horizon, (int, np.integer))
        or int(horizon) <= 0
    ):
        raise ValueError(
            f"schedule_total_steps must be a positive integer; got {horizon!r}"
        )
    return min(max(float(real_step), 0.0) / float(horizon), 1.0)

def process_flags_actor(flags):    
    if flags.drc:
        flags.wrapper_type = 1

    if flags.wrapper_type == 1:
        flags.see_h = False
        flags.see_x = False
        flags.see_tree_rep = False
        flags.see_real_state = True
        flags.im_cost = 0.
        flags.cur_cost = 0.
        flags.policy_vis_freq = -1
        flags = process_flags(flags)

    if getattr(flags, "dynamic_search", False) and flags.mcts:
        raise ValueError("MCTS is not compatible with dynamic_search")

    if flags.mcts:
        flags.train_actor = False
        flags.policy_vis_freq = -1

    if "Safexp" in flags.name or flags.name.startswith("DM"):
        flags.policy_vis_freq = -1

    flags.return_h = flags.see_h
    flags.return_x = flags.see_x

    if check_perfect_model(flags.wrapper_type):
        flags.cur_cost = 0.
        flags.cur_enable = False    

    if not flags.has_model:
        flags.train_model = False
    
    return flags


def validate_voc_actor_policy_topology(flags):
    """Bind schema-6 to one complete 16-stream batch per policy epoch."""

    if not bool(getattr(flags, "voc_actor_policy_barrier_runtime", False)):
        return False
    for name, expected in (
        ("ppo_k", 1),
        ("self_play_n", 1),
        ("env_n", 16),
        ("actor_batch_size", 16),
    ):
        value = getattr(flags, name, None)
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) != expected
        ):
            raise ValueError(
                "schema-6 actor policy barrier requires exact topology "
                f"{name}={expected}; got {value!r}"
            )
        setattr(flags, name, int(value))
    return True

def alloc_res(flags, gpu_n):
    if flags.auto_res:
        flags.self_play_n = [1, 1, 2, 2][gpu_n]
        flags.env_n = [64, 32, 32, 32][gpu_n]
        flags.gpu_self_play = [0.25, 0.5, 0.5, 1][gpu_n]
        flags.gpu_learn_actor = [0.25, 0.5, 1, 1][gpu_n]
        flags.gpu_learn = [0.5, 1, 1, 1][gpu_n]
        if not flags.train_model:
            flags.gpu_learn = 0
            flags.self_play_n = [2, 2, 2, 2][gpu_n]
            flags.gpu_self_play = [0.25, 0.5, 1, 1][gpu_n]
        if not flags.train_actor:
            flags.gpu_learn_actor = 0
            flags.self_play_n = [2, 2, 2, 3][gpu_n]
            flags.gpu_self_play = [0.25, 0.5, 1, 1][gpu_n]
        if not flags.parallel:
            flags.self_play_n = 1
            flags.env_n = 64
            flags.gpu_self_play = [0.5, 1, 1, 1][gpu_n]
            flags.gpu_learn_actor = [0.5, 1, 1, 1][gpu_n]
            flags.gpu_learn = 0
        if not flags.parallel_actor:
            flags.self_play_n = 1
            flags.env_n = flags.actor_batch_size
            flags.gpu_self_play = [0.5, 1, 1, 1][gpu_n]
            flags.gpu_learn_actor = 0
            flags.gpu_learn = [0.5, 1, 1, 1][gpu_n]
        if not flags.parallel_actor and not flags.parallel:
            flags.self_play_n = 1
            flags.env_n = flags.actor_batch_size
            flags.gpu_self_play = 1
            flags.gpu_learn_actor = 0
            flags.gpu_learn = 0
    return flags

def add_parse(filename, parser=None, prefix=''):
    # Load default configuration
    if type(filename) is not list: 
        filename = [filename]
    config = {}
    for n in filename:
        default_config_path = os.path.join(os.path.dirname(__file__), '..', 'config', n)
        with open(default_config_path, 'r') as f:
            config.update(yaml.safe_load(f))

    # Set up command line argument parsing
    if parser is None:
        parser = argparse.ArgumentParser(description=f"{__project__} v{__version__}")
    try:
        parser.add_argument('--config', type=str, help="Path to user's thinker configuration file")
    except:
        # if there is dulplicate key, just ignore
        pass

    if prefix and prefix[-1] != "_": prefix = prefix + "_"
    # Dynamically add command line arguments based on the default config keys and their types
    for key, value in config.items():
        try:
            if isinstance(value, bool):
                parser.add_argument(f'--{prefix}{key}', type=lambda x: (str(x).lower() == 'true'), help=f"Override {key}")
            else:
                parser.add_argument(f'--{prefix}{key}', type=type(value), help=f"Override {key}")
        except:
            # if there is dulplicate key, just ignore
            pass
    return parser

def create_flags(filename, save_flags=True, post_fn=None, **kwargs):
    """create flags, a namespace object that contains the config; the load
       order is filename[0], filename[1], ..., kwargs['config'], kwargs       
       args:
            filename (str/list of str): the config file(s) to load
            save_flags (bool): weather to save the flags
            post_fn (function): a function that takes flags and output flags
            **kwargs: all other settings 
       return:
            flags (namespace): config                
    """
    if type(filename) is not list: 
        filename = [filename]

    config = {}
    explicit_config_keys = set()
    for n in filename:
        default_config_path = os.path.join(os.path.dirname(__file__), '..', 'config', n)
        with open(default_config_path, 'r') as f:
            config.update(yaml.safe_load(f))

    # If user provided their own YAML configuration, load it and update defaults
    if "config" in kwargs and kwargs["config"]:
        with open(kwargs["config"], 'r') as f:
            user_config = yaml.safe_load(f)
            config.update(user_config)
            explicit_config_keys.update(user_config)

    # Check for command line argument overrides and apply them
    for key in config.keys():
        if key in kwargs and kwargs[key] is not None:
            config[key] = kwargs[key]            
            explicit_config_keys.add(key)
    # The schema discriminator is derived for canonical atomic CLIs and is
    # therefore intentionally absent from the global YAML/parser surface.
    # A caller can nevertheless supply it directly.  Preserve that explicit
    # value for lexical V19 intent so a wrong or non-built-in schema cannot be
    # discarded and then fall through to git/checkpoint/run-directory I/O.
    if (
        "voc_gate_policy_schema_version" in kwargs
        and kwargs["voc_gate_policy_schema_version"] is not None
        and (
            (
                type(kwargs["voc_gate_policy_schema_version"]) is int
                and kwargs["voc_gate_policy_schema_version"]
                == VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION
            )
            or
            _schema13_stage_xpid_candidate(config.get("xpid"))
            or _schema12_stage_xpid_candidate(config.get("xpid"))
        )
    ):
        config["voc_gate_policy_schema_version"] = kwargs[
            "voc_gate_policy_schema_version"
        ]
        explicit_config_keys.add("voc_gate_policy_schema_version")

    # Atomic schemas validate their frozen raw identity before the normal flag
    # pipeline.  Three legacy fields are nevertheless derived later by that
    # same pipeline: dynamic search removes action-sequence tokens, while the
    # actor postprocessor mirrors see_h/see_x onto return_h/return_x.  Apply
    # those deterministic derivations early only when the corresponding raw
    # field was omitted.  An explicit conflicting CLI/config value is left
    # untouched and therefore fails the strict raw schema-6 guard.
    atomic_schema_requested = (
        type(config.get("voc_actor_policy_version_barrier")) is bool
        and config["voc_actor_policy_version_barrier"]
    )
    raw_gate_schema = config.get("voc_gate_policy_schema_version")
    schema8_requested = (
        type(raw_gate_schema) is int
        and raw_gate_schema
        == VOC_GATE_POLICY_HALF_SQUARED_Q_SCHEMA_VERSION
    ) or (
        type(config.get("xpid")) is str
        and config["xpid"] in {
            profile[0]
            for profile in VOC_GATE_POLICY_SCHEMA8_STAGE_PROFILES
        }
    )
    schema9_requested = (
        type(raw_gate_schema) is int
        and raw_gate_schema
        == VOC_GATE_POLICY_COMMON_MODE_Q_SCHEMA_VERSION
    ) or (
        _schema9_stage_xpid_candidate(config.get("xpid"))
    )
    schema10_requested = (
        type(raw_gate_schema) is int
        and raw_gate_schema
        == VOC_GATE_POLICY_HUBER_COMMON_Q_SCHEMA_VERSION
    ) or (
        _schema10_stage_xpid_candidate(config.get("xpid"))
    )
    schema11_requested = (
        type(raw_gate_schema) is int
        and raw_gate_schema
        == VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION
    ) or (
        _schema11_stage_xpid_candidate(config.get("xpid"))
    )
    schema12_requested = (
        type(raw_gate_schema) is int
        and raw_gate_schema == VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION
    ) or (
        _schema12_stage_xpid_candidate(config.get("xpid"))
    )
    schema13_requested = (
        type(raw_gate_schema) is int
        and raw_gate_schema == VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION
    ) or (
        _schema13_stage_xpid_candidate(config.get("xpid"))
    )
    if schema13_requested:
        if kwargs.get("config"):
            raise ValueError(
                "schema-13 forbids user-config indirection and requires the "
                "exact 96-pair CLI vector"
            )
        raw_ckp = config.get("ckp")
        if type(raw_ckp) is not bool or raw_ckp is not False:
            raise ValueError(
                "schema-13 is fresh-only and requires ckp to be exact "
                "Python bool False before loading persisted configuration; "
                f"got {raw_ckp!r}"
            )
        _validate_schema13_stage_profile(
            config, label="raw schema-13 create-flags intent"
        )
        raw_barrier = config.get("voc_actor_policy_version_barrier")
        if type(raw_barrier) is not bool or raw_barrier is not True:
            raise ValueError(
                "schema-13 intent requires "
                "voc_actor_policy_version_barrier=True as an exact Python "
                f"bool; got {raw_barrier!r}"
            )
        raw_seal = config.get("voc_model_input_seal_schema_version")
        if type(raw_seal) is not int or raw_seal != 1:
            raise ValueError(
                "schema-13 intent requires "
                "voc_model_input_seal_schema_version=1 as an exact Python "
                f"integer; got {raw_seal!r}"
            )
        if "voc_gate_policy_schema_version" in config:
            if (
                type(raw_gate_schema) is not int
                or raw_gate_schema
                != VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION
            ):
                raise ValueError(
                    "schema-13 intent requires an explicitly supplied "
                    "voc_gate_policy_schema_version=13 as an exact Python "
                    f"integer; got {raw_gate_schema!r}"
                )
        else:
            config["voc_gate_policy_schema_version"] = (
                VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION
            )
        if "voc_gate_target_tau" not in explicit_config_keys:
            config["voc_gate_target_tau"] = 1.0
        raw_tau = config.get("voc_gate_target_tau")
        if (
            type(raw_tau) is not float
            or not np.isfinite(raw_tau)
            or raw_tau != 1.0
        ):
            raise ValueError(
                "schema-13 intent requires voc_gate_target_tau=1.0 as an "
                f"exact built-in float; got {raw_tau!r}"
            )
        atomic_schema_requested = True
    elif schema12_requested:
        raw_ckp = config.get("ckp")
        if type(raw_ckp) is not bool or raw_ckp is not False:
            raise ValueError(
                "schema-12 is fresh-only and requires ckp to be exact "
                "Python bool False before loading persisted configuration; "
                f"got {raw_ckp!r}"
            )
        _validate_schema12_stage_profile(
            config, label="raw schema-12 create-flags intent"
        )
        raw_barrier = config.get("voc_actor_policy_version_barrier")
        if type(raw_barrier) is not bool or raw_barrier is not True:
            raise ValueError(
                "schema-12 intent requires "
                "voc_actor_policy_version_barrier=True as an exact Python "
                f"bool; got {raw_barrier!r}"
            )
        raw_seal = config.get("voc_model_input_seal_schema_version")
        if type(raw_seal) is not int or raw_seal != 1:
            raise ValueError(
                "schema-12 intent requires "
                "voc_model_input_seal_schema_version=1 as an exact Python "
                f"integer; got {raw_seal!r}"
            )
        if "voc_gate_policy_schema_version" in config:
            if (
                type(raw_gate_schema) is not int
                or raw_gate_schema != VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION
            ):
                raise ValueError(
                    "schema-12 intent requires an explicitly supplied "
                    "voc_gate_policy_schema_version=12 as an exact Python "
                    f"integer; got {raw_gate_schema!r}"
                )
        else:
            config["voc_gate_policy_schema_version"] = (
                VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION
            )
        if "voc_gate_target_tau" not in explicit_config_keys:
            config["voc_gate_target_tau"] = 1.0
        raw_tau = config.get("voc_gate_target_tau")
        if type(raw_tau) is not float or not np.isfinite(raw_tau) or raw_tau != 1.0:
            raise ValueError(
                "schema-12 intent requires voc_gate_target_tau=1.0 as an "
                f"exact built-in float; got {raw_tau!r}"
            )
        atomic_schema_requested = True
    elif schema11_requested:
        raw_ckp = config.get("ckp")
        if type(raw_ckp) is not bool or raw_ckp is not False:
            raise ValueError(
                "schema-11 is fresh-only and requires ckp to be exact "
                "Python bool False before loading persisted configuration; "
                f"got {raw_ckp!r}"
            )
        # A lexical V18 xpid is itself schema-11 intent.  It must never fall
        # through the legacy flag path merely because another atomic field is
        # absent or malformed.  The production CLI deliberately omits the
        # derived gate-schema argument, so infer only that one field after the
        # exact barrier/seal pair proves the canonical inference route.  Every
        # explicitly supplied schema value remains strict and fail-closed.
        _validate_schema11_stage_profile(
            config, label="raw schema-11 create-flags intent"
        )
        raw_barrier = config.get("voc_actor_policy_version_barrier")
        if type(raw_barrier) is not bool or raw_barrier is not True:
            raise ValueError(
                "schema-11 intent requires "
                "voc_actor_policy_version_barrier=True as an exact Python "
                f"bool; got {raw_barrier!r}"
            )
        raw_seal = config.get("voc_model_input_seal_schema_version")
        if type(raw_seal) is not int or raw_seal != 1:
            raise ValueError(
                "schema-11 intent requires "
                "voc_model_input_seal_schema_version=1 as an exact Python "
                f"integer; got {raw_seal!r}"
            )
        if "voc_gate_policy_schema_version" in config:
            if (
                type(raw_gate_schema) is not int
                or raw_gate_schema
                != VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION
            ):
                raise ValueError(
                    "schema-11 intent requires an explicitly supplied "
                    "voc_gate_policy_schema_version=11 as an exact Python "
                    f"integer; got {raw_gate_schema!r}"
                )
        else:
            config["voc_gate_policy_schema_version"] = (
                VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION
            )
        atomic_schema_requested = True
    elif schema10_requested:
        raw_ckp = config.get("ckp")
        if type(raw_ckp) is not bool or raw_ckp is not False:
            raise ValueError(
                "schema-10 is fresh-only and requires ckp to be exact "
                "Python bool False before loading persisted configuration; "
                f"got {raw_ckp!r}"
            )
    elif schema9_requested:
        raw_ckp = config.get("ckp")
        if type(raw_ckp) is not bool or raw_ckp is not False:
            raise ValueError(
                "schema-9 is fresh-only and requires ckp to be exact "
                "Python bool False before loading persisted configuration; "
                f"got {raw_ckp!r}"
            )
    elif schema8_requested:
        raw_ckp = config.get("ckp")
        if type(raw_ckp) is not bool or raw_ckp is not False:
            raise ValueError(
                "schema-8 is fresh-only and requires ckp to be exact "
                "Python bool False before loading persisted configuration; "
                f"got {raw_ckp!r}"
            )
    if atomic_schema_requested:
        if (
            type(config.get("dynamic_search")) is bool
            and config["dynamic_search"]
            and "has_action_seq" not in explicit_config_keys
        ):
            config["has_action_seq"] = False
        if post_fn is process_flags_actor:
            for return_name, see_name in (
                ("return_h", "see_h"),
                ("return_x", "see_x"),
            ):
                if return_name not in explicit_config_keys:
                    config[return_name] = config.get(see_name)

    if schema13_requested:
        # Validate the prospective schema-13 surface before git, persisted
        # configuration, checkpoint, or run-directory I/O.
        preflight_config = copy.deepcopy(config)
        if not preflight_config.get("project"):
            preflight_config["project"] = __project__
        preflight_config["savedir"] = preflight_config["savedir"].replace(
            "__project__", preflight_config["project"]
        )
        preflight_config["__version__"] = __version__
        preflight_config["cmd"] = " ".join(sys.argv)
        preflight_config["git_revision"] = None
        preflight_config["ckpdir"] = os.path.join(
            preflight_config["savedir"], preflight_config["xpid"]
        )
        preflight_flags = process_flags(argparse.Namespace(**preflight_config))
        _validate_schema13_complete_surface(
            vars(preflight_flags), label="raw schema-13 create-flags surface"
        )
    elif schema12_requested:
        # Validate the prospective schema-12 surface before any git, persisted
        # configuration, checkpoint, or run-directory I/O.
        preflight_config = copy.deepcopy(config)
        if not preflight_config.get("project"):
            preflight_config["project"] = __project__
        preflight_config["savedir"] = preflight_config["savedir"].replace(
            "__project__", preflight_config["project"]
        )
        preflight_config["__version__"] = __version__
        preflight_config["cmd"] = " ".join(sys.argv)
        preflight_config["git_revision"] = None
        preflight_config["ckpdir"] = os.path.join(
            preflight_config["savedir"], preflight_config["xpid"]
        )
        preflight_flags = process_flags(argparse.Namespace(**preflight_config))
        _validate_schema12_complete_surface(
            vars(preflight_flags), label="raw schema-12 create-flags surface"
        )
    elif schema11_requested:
        # Validate the complete prospective 229-key surface before the normal
        # git/config/checkpoint/run-directory path.  Use the same deterministic
        # values that create_flags adds below, but deliberately avoid invoking
        # the git subprocess or mutating the live configuration during this
        # read-only preflight.
        preflight_config = copy.deepcopy(config)
        if not preflight_config.get("project"):
            preflight_config["project"] = __project__
        preflight_config["savedir"] = preflight_config["savedir"].replace(
            "__project__", preflight_config["project"]
        )
        preflight_config["__version__"] = __version__
        preflight_config["cmd"] = " ".join(sys.argv)
        preflight_config["git_revision"] = None
        preflight_config["ckpdir"] = os.path.join(
            preflight_config["savedir"], preflight_config["xpid"]
        )
        preflight_flags = process_flags(argparse.Namespace(**preflight_config))
        _validate_schema11_complete_surface(
            vars(preflight_flags), label="raw schema-11 create-flags surface"
        )

    # Convert dictionary to named tuple
    flags = argparse.Namespace(**config)    

    # additional info
    if not flags.project: flags.project = __project__
    flags.savedir = flags.savedir.replace("__project__", flags.project)    
    flags.__version__ = __version__
    flags.cmd = " ".join(sys.argv) 

    try:
        flags.git_revision = get_git_revision_hash()
    except Exception:
        flags.git_revision = None

    if flags.ckp:
        # load setting from checkpoint yaml    
        xpid = 'latest' if not flags.xpid else flags.xpid
        config_path = os.path.join(flags.savedir, xpid, "config_c.yaml")        
        if os.path.islink(config_path): config_path = os.readlink(config_path)
        with open(config_path, 'r') as f:
            config_ = yaml.safe_load(f)
        for key, value in config_.items():
            if (key not in ['ckp', 'ray_mem', 'ray_gpu', 'savedir'] and
                not (key in kwargs and kwargs[key] is not None)):
                setattr(flags, key, value)
        print("Loaded config from %s" % config_path)

    if not flags.xpid:        
        flags.xpid = "%s-%s" % (flags.project, time.strftime("%Y%m%d-%H%M%S"))

    flags.ckpdir = os.path.join(flags.savedir, flags.xpid,)     

    flags = process_flags(flags)
    if post_fn is not None: flags = post_fn(flags)
    if bool(getattr(flags, "voc_actor_policy_version_barrier", False)):
        if (
            flags.voc_gate_policy_schema_version
            == VOC_GATE_POLICY_VERSION_BARRIER_SCHEMA_VERSION
        ):
            _validate_schema6_complete_surface(
                vars(flags), label="postprocessed schema-6 flags"
            )
        elif (
            flags.voc_gate_policy_schema_version
            == VOC_GATE_POLICY_MODEL_INPUT_SEAL_SCHEMA_VERSION
        ):
            _validate_schema7_complete_surface(
                vars(flags), label="postprocessed schema-7 flags"
            )
        elif (
            flags.voc_gate_policy_schema_version
            == VOC_GATE_POLICY_HALF_SQUARED_Q_SCHEMA_VERSION
        ):
            _validate_schema8_complete_surface(
                vars(flags), label="postprocessed schema-8 flags"
            )
        elif (
            flags.voc_gate_policy_schema_version
            == VOC_GATE_POLICY_COMMON_MODE_Q_SCHEMA_VERSION
        ):
            _validate_schema9_complete_surface(
                vars(flags), label="postprocessed schema-9 flags"
            )
        elif (
            flags.voc_gate_policy_schema_version
            == VOC_GATE_POLICY_HUBER_COMMON_Q_SCHEMA_VERSION
        ):
            _validate_schema10_complete_surface(
                vars(flags), label="postprocessed schema-10 flags"
            )
        elif (
            flags.voc_gate_policy_schema_version
            == VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION
        ):
            _validate_schema11_complete_surface(
                vars(flags), label="postprocessed schema-11 flags"
            )
        elif (
            flags.voc_gate_policy_schema_version
            == VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION
        ):
            _validate_schema12_complete_surface(
                vars(flags), label="postprocessed schema-12 flags"
            )
        else:
            _validate_schema13_complete_surface(
                vars(flags), label="postprocessed schema-13 flags"
            )

    if save_flags and not flags.ckp:        
        ckpdir = full_path(flags.ckpdir)
        schema6_fresh = bool(
            getattr(flags, "voc_actor_policy_version_barrier", False)
            and getattr(flags, "train_actor", False)
        )
        if schema6_fresh:
            create_schema6_fresh_run_directory(ckpdir)
        elif not os.path.exists(ckpdir):
            os.makedirs(ckpdir)
            
        # Set environment variable for the replay buffer to use
        os.environ['THINKER_LOG_DIR'] = ckpdir
        print(f"Set THINKER_LOG_DIR environment variable to: {ckpdir}")
            
        try:
            # create sym link for the latest run
            symlink = os.path.join(full_path(flags.savedir), "latest")
            if os.path.islink(symlink):
                os.remove(symlink)
            if not os.path.exists(symlink):
                os.symlink(flags.ckpdir, symlink)
                print("Symlinked log directory: %s" % symlink)
        except OSError:
            # os.remove() or os.symlink() raced. Don't do anything.
            pass

        config_path = os.path.join(full_path(flags.savedir), 
                                   flags.xpid, 
                                   "config_c.yaml")
        with open(config_path, 'x' if schema6_fresh else 'w') as outfile:
            yaml.dump(vars(flags), outfile)
        print("Wrote config file to %s" % config_path)  

    fs = ["savedir", "preload", "ckpdir"]
    for f in fs:
        path = getattr(flags, f)
        if path:            
            setattr(flags, f, full_path(path))
    return flags

def create_setting(args=None, save_flags=True, **kwargs):
    _validate_schema13_cli_vector(args, keyword_xpid=kwargs.get("xpid"))
    filenames = ['default_thinker.yaml', 'default_actor.yaml']
    parser = add_parse(filenames)
    if args is not None:
        parse_flags = parser.parse_args(args)
    else:
        parse_flags = parser.parse_args()

    parse_dict = vars(parse_flags)
    for key in parse_dict.keys():
        if key in kwargs and kwargs[key] is not None:
            parse_dict[key] = kwargs[key]            
    if "voc_gate_policy_schema_version" in kwargs:
        parse_dict["voc_gate_policy_schema_version"] = kwargs[
            "voc_gate_policy_schema_version"
        ]
    # argparse also accepts ``--flag=value`` and unique abbreviations.  Re-run
    # the raw-vector guard after parsing so either spelling cannot hide V20
    # intent from the exact pre-parser check above.
    _validate_schema13_cli_vector(args, keyword_xpid=parse_dict.get("xpid"))

    flags = create_flags(filenames, 
                         save_flags=save_flags, 
                         post_fn=process_flags_actor, 
                         **parse_dict)
    return flags

def full_path(path):
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.islink(path):
        path = os.readlink(path)
    return path

def tuple_map(x, f, skip_dict=False):
    def process_element(y):
        # Apply function to dictionary items
        if isinstance(y, dict):
            if not skip_dict:
                return {k: f(v) if v is not None else None for k, v in y.items()}
            else:
                return {}
        return f(y) if y is not None else None

    if type(x) == tuple:
        return tuple(process_element(y) for y in x)
    else:
        return type(x)(*(process_element(y) for y in x))

def dict_map(x, f):
    return {k:f(v) if v is not None else None for (k, v) in x.items()}

def safe_view(x, dims):
    if x is None:
        return None
    else:
        return x.view(*dims)
    
def safe_squeeze(x, dim=0):
    if x is None:
        return None
    else:
        return x.squeeze(dim)


def safe_unsqueeze(x, dim=0):
    if x is None:
        return None
    else:
        return x.unsqueeze(dim)


def safe_concat(xs, attr, dim=0):
    if len(xs) == 0:
        return None
    if getattr(xs[0], attr) is None:
        return None
    return torch.concat([getattr(i, attr).unsqueeze(dim) for i in xs], dim=dim)


def construct_tuple(x, **kwargs):
    return x(**{k: kwargs[k] if k in kwargs else None for k in x._fields})


def get_git_revision_hash():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()

def enc(x, f_type=0):
    if f_type == 0:
        return np.sign(x) * (np.sqrt(np.abs(x) + 1) - 1) + (0.001) * x
    else:
        return np.sign(x) * np.log(np.abs(x) + 1)

def dec(x, f_type=0):
    if f_type == 0:
        return np.sign(x) * (
            np.square(
                (np.sqrt(1 + 4 * 0.001 * (np.abs(x) + 1 + 0.001)) - 1) / (2 * 0.001)
            )
            - 1
        )
    else:
        return np.sign(x) * (np.exp(np.abs(x)) - 1)

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def copy_net(tar_net, net):
    for tar_module, new_module in zip(tar_net.modules(), net.modules()):
        if isinstance(tar_module, nn.modules.batchnorm._BatchNorm):
            # Copy BatchNorm running mean and variance
            tar_module.running_mean = new_module.running_mean.clone()
            tar_module.running_var = new_module.running_var.clone()
        for tar_param, new_param in zip(tar_module.parameters(), new_module.parameters()):
            tar_param.data = new_param.data.clone()

def load_optimizer(optimizer, optimizer_state_dict):
    # Preserve the run-configured base LR while restoring every optimizer
    # moment.  Work on a copy so validation/retry code never observes a
    # checkpoint mapping mutated by this helper.
    optimizer_state_dict = copy.deepcopy(optimizer_state_dict)
    current_lrs = [group['lr'] for group in optimizer.param_groups]
    for i, group in enumerate(optimizer_state_dict['param_groups']):
        if i < len(current_lrs):
            group['lr'] = current_lrs[i]
    optimizer.load_state_dict(optimizer_state_dict)

def load_scheduler(scheduler, scheduler_state_dict):
    # Keep a deliberately overridden configured base LR, but resume at the
    # *scheduled* multiplier saved by the checkpoint before the very next
    # optimizer update.  Previously the first resumed update ran at the fresh
    # base LR and only recovered its schedule afterward.
    scheduler_state_dict = copy.deepcopy(scheduler_state_dict)
    saved_base_lrs = scheduler_state_dict.pop('base_lrs', None)
    saved_last_lrs = scheduler_state_dict.get('_last_lr')
    scheduler.load_state_dict(scheduler_state_dict)
    resumed_lrs = None
    if (
        isinstance(saved_base_lrs, (list, tuple))
        and isinstance(saved_last_lrs, (list, tuple))
        and len(saved_base_lrs) == len(saved_last_lrs)
        and len(saved_last_lrs) == len(scheduler.optimizer.param_groups)
    ):
        resumed_lrs = []
        for current_base, saved_base, saved_last in zip(
            scheduler.base_lrs, saved_base_lrs, saved_last_lrs
        ):
            if not all(
                isinstance(value, (int, float, np.number))
                and not isinstance(value, (bool, np.bool_))
                and np.isfinite(value)
                for value in (current_base, saved_base, saved_last)
            ):
                raise ValueError("scheduler checkpoint has non-finite LR state")
            multiplier = (
                float(saved_last) / float(saved_base)
                if float(saved_base) != 0.0 else 1.0
            )
            resumed_lrs.append(float(current_base) * multiplier)
    elif (
        isinstance(saved_last_lrs, (list, tuple))
        and len(saved_last_lrs) == len(scheduler.optimizer.param_groups)
    ):
        resumed_lrs = [float(value) for value in saved_last_lrs]
    if resumed_lrs is not None:
        if not all(np.isfinite(value) and value >= 0.0 for value in resumed_lrs):
            raise ValueError("scheduler checkpoint has invalid resumed LR")
        for group, learning_rate in zip(
            scheduler.optimizer.param_groups, resumed_lrs
        ):
            group['lr'] = learning_rate
        scheduler._last_lr = list(resumed_lrs)

def logger():
    formatter = logging.Formatter("%(message)s")
    logger = logging.getLogger("logs/out")
    if not logger.hasHandlers():
        shandle = logging.StreamHandler()
        shandle.setFormatter(formatter)
        logger.addHandler(shandle)
    logger.setLevel(logging.INFO)
    return logger

class Timings:
    def __init__(self):
        self._means = collections.defaultdict(int)
        self._vars = collections.defaultdict(int)
        self._counts = collections.defaultdict(int)
        self._mean_deques = {}
        self.reset()

    def reset(self):
        self.last_time = timeit.default_timer()

    def time(self, name):
        now = timeit.default_timer()
        x = now - self.last_time
        self.last_time = now

        n = self._counts[name]

        mean = self._means[name] + (x - self._means[name]) / (n + 1)
        var = (
            n * self._vars[name] + n * (self._means[name] - mean) ** 2 + (x - mean) ** 2
        ) / (n + 1)

        self._means[name] = mean
        self._vars[name] = var
        self._counts[name] += 1

        if name not in self._mean_deques:
            self._mean_deques[name] = collections.deque(maxlen=5)
        self._mean_deques[name].append(x)

    def means(self):
        return self._means

    def vars(self):
        return self._vars

    def stds(self):
        return {k: v**0.5 for k, v in self._vars.items()}

    def summary(self, prefix=""):
        means = self.means()
        stds = self.stds()
        mean_deques = self._mean_deques
        total = sum(means.values())

        result = prefix
        for k in sorted(means, key=means.get, reverse=True):
            result += f"\n    %s: %.6fms (last 5: %.6fms) +- %.6fms (%.2f%%) " % (
                k,
                1000 * means[k],
                1000 * np.average(mean_deques[k]),
                1000 * stds[k],
                100 * means[k] / total,
            )
        result += "\nTotal: %.6fms" % (1000 * total)
        return result

class Wandb:
    def __init__(self, flags, subname=""):
        import wandb

        self.wandb = wandb
        xpid = flags.full_xpid if hasattr(flags, "full_xpid") else flags.xpid
        exp_name = xpid + subname
        tags = []
        if subname == "_model":
            tags.append("model")
        m = re.match(r"^v\d+", exp_name)
        if m:
            tags.append(m[0])
        self.wandb.init(
            project=flags.project,
            config=flags,
            entity=os.getenv("WANDB_USER", ""),
            reinit=True,
            # Restore parameters
            resume="allow",
            id=exp_name,
            name=exp_name,
            tags=tags,
        )
        self.wandb.config.update(flags, allow_val_change=True)

def compute_grad_norm(parameters, norm_type=2.0):
    grads = [p.grad for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.0)
    device = grads[0].device
    total_norm = torch.norm(
        torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]),
        norm_type,
    )
    return total_norm

def slice_tree_reps(num_actions, dim_actions, rec_t):
    idx1 = num_actions * 5 + 6
    idx2 = idx1
    idx3 = idx2 + num_actions * 5 + 3
    idx4 = idx3
    idx5 = idx4 + 2 + rec_t  
    tree_rep_map = [
        ["root_action", 0],
        ["root_r", num_actions],
        ["root_d", num_actions+1],
        ["root_v", num_actions+2],
        ["root_policy", num_actions+3],
        ["root_qs_mean", 2*num_actions+3],
        ["root_qs_max", 3*num_actions+3],
        ["root_ns", 4*num_actions+3],
        ["root_trail_r", 5*num_actions+3],
        ["rollout_return", 5*num_actions+4],
        ["max_rollout_return", 5*num_actions+5],
        ["root_raw_action", idx1],
        ["cur_action", idx2],
        ["cur_r", idx2+num_actions],
        ["cur_d", idx2+num_actions+1],
        ["cur_v", idx2+num_actions+2],
        ["cur_policy", idx2+num_actions+3],
        ["cur_qs_mean", idx2+2*num_actions+3],
        ["cur_qs_max", idx2+3*num_actions+3],
        ["cur_ns", idx2+4*num_actions+3],
        ["cur_raw_action", idx3],
        ["cur_reset", idx4],
        ["k", idx4+1],
        ["deprec", idx4+1+rec_t],
        ["action_seq", idx5]
        ]
    tree_rep_map_d = {}
    for n, (k, idx) in enumerate(tree_rep_map):
        next_idx = tree_rep_map[n+1][1] if n + 1 < len(tree_rep_map) else None
        tree_rep_map_d[k] = slice(idx, next_idx)    
    return tree_rep_map_d


def slice_dynamic_tree_reps(num_actions, dim_actions=1):
    """Return the budget-independent Dynamic Thinker token schema.

    A token contains the original detailed root and compact current-node
    statistics followed by five scalar search metadata values.  The feature
    width is therefore ``10 * num_actions + 14`` and is independent of both
    max_search_steps and max_depth.
    """
    del dim_actions  # the current wrappers expose discrete augmented actions
    root_end = num_actions * 5 + 6
    cur_end = root_end + num_actions * 5 + 3
    names = [
        ["root_action", 0],
        ["root_r", num_actions],
        ["root_d", num_actions + 1],
        ["root_v", num_actions + 2],
        ["root_policy", num_actions + 3],
        ["root_qs_mean", 2 * num_actions + 3],
        ["root_qs_max", 3 * num_actions + 3],
        ["root_ns", 4 * num_actions + 3],
        ["root_trail_r", 5 * num_actions + 3],
        ["rollout_return", 5 * num_actions + 4],
        ["max_rollout_return", 5 * num_actions + 5],
        ["cur_action", root_end],
        ["cur_r", root_end + num_actions],
        ["cur_d", root_end + num_actions + 1],
        ["cur_v", root_end + num_actions + 2],
        ["cur_policy", root_end + num_actions + 3],
        ["cur_qs_mean", root_end + 2 * num_actions + 3],
        ["cur_qs_max", root_end + 3 * num_actions + 3],
        ["cur_ns", root_end + 4 * num_actions + 3],
        ["tree_reset", cur_end],
        ["depth_discount", cur_end + 1],
        ["search_steps", cur_end + 2],
        ["rollout_depth", cur_end + 3],
        ["search_start", cur_end + 4],
    ]
    out = {}
    for i, (key, start) in enumerate(names):
        end = names[i + 1][1] if i + 1 < len(names) else cur_end + 5
        out[key] = slice(start, end)
    # Compatibility name used by visualizers for the reset event scalar.
    out["cur_reset"] = out["tree_reset"]
    return out


def get_tree_rep_meaning(num_actions, dim_actions, flags):
    if dynamic_search_enabled(flags):
        return slice_dynamic_tree_reps(num_actions, dim_actions)
    return slice_tree_reps(num_actions, dim_actions, flags.rec_t)

def _decode_tree_reps_with_schema(tree_reps, schema, enc_type=0, f_type=0):
    nd = [
            "root_r", "root_v", "root_qs_mean", "root_qs_max", 
            "root_trail_r", "rollout_return", "max_rollout_return", 
            "cur_r", "cur_v", "cur_qs_mean", "cur_qs_max"
        ]
    def dec_k(x, key):        
        if enc_type != 0 and key in nd:
            return dec(x, f_type)
        else:
            return x

    if len(tree_reps.shape) == 3:
        tree_reps = tree_reps[0]

    return {k: dec_k(tree_reps[:, v], k) for k, v in schema.items()}


def decode_tree_reps(tree_reps, num_actions, dim_actions, rec_t, enc_type=0, f_type=0):
    """Decode the legacy fixed-budget tree representation."""
    schema = slice_tree_reps(num_actions, dim_actions, rec_t)
    return _decode_tree_reps_with_schema(
        tree_reps, schema, enc_type=enc_type, f_type=f_type
    )


def decode_dynamic_tree_reps(
        tree_reps, num_actions, dim_actions=1, enc_type=0, f_type=0):
    """Decode a budget-independent Dynamic Thinker token."""
    schema = slice_dynamic_tree_reps(num_actions, dim_actions)
    return _decode_tree_reps_with_schema(
        tree_reps, schema, enc_type=enc_type, f_type=f_type
    )

def mask_tree_rep(tree_reps, num_actions, rec_t):
    # deprecated
    d = slice_tree_reps(num_actions, rec_t)  
    N, C = tree_reps.shape
    act_seq_len = C - (num_actions * 10 + 11 + rec_t)
    tree_reps_m = torch.zeros(N, 4+rec_t+act_seq_len, device=tree_reps.device)
    tree_reps_m[:, [0]] = tree_reps[:, d["reset"]]
    tree_reps_m[:, [1]] = tree_reps[:, d["cur_r"]] # imagainary reward
    tree_reps_m[:, [2]] = tree_reps[:, d["cur_d"]] # imagainary done
    tree_reps_m[:, [3]] = tree_reps[:, d["derec"]] # deprec
    tree_reps_m[:, 4:4+rec_t] = tree_reps[:, d["k"]] # time
    tree_reps_m[:, 4+rec_t:] = tree_reps[:, d["action_seq"]]
    return tree_reps_m

def encode_action(action, action_space, one_hot=False):
    if type(action_space) == spaces.discrete.Discrete:       
        if one_hot:
            return action
        else:
            return F.one_hot(action.squeeze(-1), num_classes=action_space.n).float()
    elif type(action_space) == spaces.tuple.Tuple:   
            if one_hot:
                action = torch.sum(action * torch.arange(action_space[0].n, device=action.device), dim=-1)   
            return action.float()/action_space[0].n
    elif type(action_space) == spaces.Box:  
            return action.float()
    
def process_action_space(action_space):
    if type(action_space) == spaces.discrete.Discrete:                        
        num_actions = action_space.n    
        dim_actions = 1
        dim_rep_actions = num_actions
        tuple_action = False        
        discrete_action = True
    elif type(action_space) == spaces.tuple.Tuple:              
        num_actions = action_space[0].n    
        dim_actions = len(action_space)    
        dim_rep_actions = dim_actions
        tuple_action = True
        discrete_action = True
    elif type(action_space) == spaces.Box:  
        num_actions = 1   
        dim_actions = action_space.shape[0] 
        dim_rep_actions = dim_actions
        tuple_action = True
        discrete_action = False
    else:
        raise AssertionError(f"Unsupported action space {action_space}")
    return num_actions, dim_actions, dim_rep_actions, tuple_action, discrete_action

def plot_raw_state(x, ax=None, title=None, savepath=None):
    if ax is None:
        _, ax = plt.subplots()
    if not isinstance(x, np.ndarray):
        x = x[-3:].detach().cpu().numpy()
    else:
        x = x[-3:]    
    # Swap axes
    x = np.swapaxes(np.swapaxes(x, 0, 2), 0, 1)
    if x.dtype in [float, np.float32]:
        x = np.clip(x, 0, 1)
    if x.dtype in [int, np.uint8]:
        x = np.clip(x, 0, 255)
    ax.imshow(x, interpolation="nearest", aspect="auto")
    if title is not None:
        ax.set_title(title)
    if savepath is not None:
        plt.savefig(os.path.join(savepath, title + ".png"))
        plt.close()

def check_perfect_model(wrapper_type):
    return wrapper_type in [2, 4, 5]        

class FifoBuffer:
    def __init__(self, size, device):
        self.size = size
        self.buffer = torch.empty(
            (self.size,), dtype=torch.float32, device=device
        ).fill_(float("nan"))
        self.current_index = 0
        self.num_elements = 0

    def push(self, data):
        num_entries = math.prod(data.shape)
        assert num_entries <= self.size, "Data too large for buffer"

        start_index = self.current_index
        end_index = (self.current_index + num_entries) % self.size

        if end_index < start_index:
            # The new data wraps around the buffer
            remaining_space = self.size - start_index
            self.buffer[start_index:] = data.flatten()[:remaining_space]
            self.buffer[:end_index] = data.flatten()[remaining_space:]
        else:
            # The new data fits within the remaining space
            self.buffer[start_index:end_index] = data.flatten()

        self.current_index = end_index
        self.num_elements = min(self.num_elements + num_entries, self.size)

    def get_percentile(self, percentile):
        num_valid_elements = min(self.num_elements, self.size)
        if num_valid_elements == 0:
            return None
        return torch.quantile(self.buffer[:num_valid_elements], q=percentile)

    def get_variance(self):
        num_valid_elements = min(self.num_elements, self.size)
        if num_valid_elements == 0:
            return None
        return torch.mean(torch.square(self.buffer[:num_valid_elements]))

    def get_mean(self):
        num_valid_elements = min(self.num_elements, self.size)
        if num_valid_elements == 0:
            return None
        return torch.mean(self.buffer[:num_valid_elements])

    def full(self):
        return self.num_elements >= self.size

# taken from https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_normalize.py
class RunningMeanStd:
    """Tracks the mean, variance and count of values."""

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        """Tracks the mean, variance and count of values."""
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        """Updates the mean, var and count from a batch of samples."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates from batch mean, variance and count moments."""
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )

def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count
    
def clone_bn_running_stats(module):
    """
    Traverse the module and its submodules to clone all BatchNorm layers' running mean and variance.
    
    Parameters:
    - module: The root module to traverse.
    
    Returns:
    - A dictionary containing the cloned running mean and variance for each BatchNorm layer.
    """
    cloned_stats = {}
    for name, submodule in module.named_modules():
        if isinstance(submodule, nn.modules.batchnorm._BatchNorm):
            # Use the module's name as a unique identifier
            cloned_stats[name] = {
                "running_mean": submodule.running_mean.clone(),
                "running_var": submodule.running_var.clone(),
            }
    return cloned_stats

def restore_bn_running_stats(module, cloned_stats):
    """
    Traverse the module and its submodules to restore BatchNorm layers' running mean and variance from cloned statistics.
    
    Parameters:
    - module: The root module to traverse.
    - cloned_stats: A dictionary containing the cloned running mean and variance for each BatchNorm layer.
    """
    for name, submodule in module.named_modules():
        if name in cloned_stats and isinstance(submodule, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            # Restore the running statistics from the cloned values
            submodule.running_mean = cloned_stats[name]["running_mean"]
            submodule.running_var = cloned_stats[name]["running_var"]
