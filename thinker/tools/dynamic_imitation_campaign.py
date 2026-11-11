#!/usr/bin/env python3
"""Build and audit the 72-fold Dynamic Imitation 5M campaign.

The tool has two deliberately separate responsibilities:

* audit the immutable behavioral snapshot and publish a deterministic 72-fold
  campaign manifest plus a 70-task pending ledger; and
* turn the exact 5,000,000-step milestones from the two pre-existing runs into
  immutable, tamper-evident canonical run directories.

Publishing is no-clobber.  Every directory is assembled beside its destination
and renamed only after all validation and hashing succeeds.  Legacy run
directories are read only, mutable ``ckp_*.tar`` aliases are never copied, and
no ``finish`` marker is created.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import errno
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import zipfile


TARGET_REAL_STEP = 5_000_000
SUBJECTS = tuple(range(1, 7))
SESSIONS = (1, 2, 3, 4)
EXPECTED_SESSION_FILE_COUNTS = {1: 2, 2: 3, 3: 3, 4: 3}

DEFAULT_REPOSITORY_ROOT = Path("/work/jeongmin/thinker-dynamic-imitation")
DEFAULT_LOGS_ROOT = DEFAULT_REPOSITORY_ROOT / "logs" / "thinker"
DEFAULT_CAMPAIGN_DIR = Path(
    "/work/jeongmin/dynamic-imitation-campaigns/20260902-72fold-5m"
)
DEFAULT_SOURCE_ROOT = Path(
    "/work/jeongmin/dynamic-imitation-snapshots/"
    "20260901-b30d30u30-ceonly-noprior-noop"
)
DEFAULT_SOURCE_MANIFEST = DEFAULT_SOURCE_ROOT.with_suffix(".sha256")
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "f32e83ba07d4b91bd454dcc84706e45bd1be2e33a66273ca3eeaa1ca5679b97f"
)
DEFAULT_BEHAVIOR_ROOT = Path(
    "/work/jeongmin/dynamic-imitation-snapshots/"
    "behavioral-data-20260902-72fold"
)
DEFAULT_BEHAVIOR_SHA256_MANIFEST = DEFAULT_BEHAVIOR_ROOT.with_suffix(".sha256")
EXPECTED_BEHAVIOR_SHA256_MANIFEST_SHA256 = (
    "d976ee71a94fa7bbe17ca53e6fdaf42c9fc64923266c1ff2469565c866e4c65f"
)

DATA_MANIFEST_NAME = "behavioral_data_manifest.json"
CAMPAIGN_MANIFEST_NAME = "campaign_manifest.json"
PENDING_LEDGER_NAME = "pending_tasks.jsonl"
SHA256SUMS_NAME = "SHA256SUMS"
MILESTONE_NAME = "milestone_5000000.json"


@dataclass(frozen=True)
class GameSpec:
    game_id: int
    game_label: str
    env_name: str
    num_actions: int


GAMES = (
    GameSpec(0, "enduro", "Enduro-v5", 9),
    GameSpec(1, "pong", "Pong-v5", 6),
    GameSpec(2, "spaceinvaders", "SpaceInvaders-v5", 6),
)
GAME_BY_ID = {game.game_id: game for game in GAMES}


@dataclass(frozen=True)
class ExistingRunSpec:
    game_id: int
    source_xpid: str
    canonical_xpid: str
    actor_sha256: str
    model_sha256: str

    @property
    def game(self) -> GameSpec:
        return GAME_BY_ID[self.game_id]


EXISTING_RUNS = {
    "enduro": ExistingRunSpec(
        game_id=0,
        source_xpid="dynamic-imitation-enduro-voc-b30-d30-u30-ap0-m0-s1-11774",
        canonical_xpid="dynamic-imitation-enduro-voc-sub1-leavesession4",
        actor_sha256=(
            "e7e3f993c64c94e415c4fdb2f4c3da917ee84e6785748f7af3e94e872f7ffc12"
        ),
        model_sha256=(
            "de688d3bb366b399109f9eab56f844da6805079318f609979b4c253d0fac6adc"
        ),
    ),
    "pong": ExistingRunSpec(
        game_id=1,
        source_xpid="dynamic-imitation-pong-voc-b30-d30-u30-ap0-m0-s1-11775",
        canonical_xpid="dynamic-imitation-pong-voc-sub1-leavesession4",
        actor_sha256=(
            "d47ae07ab4c8ecd4ec92a495328699ab4162e13f1ec149f3fb30467ec70644f0"
        ),
        model_sha256=(
            "523d79c370f5a36b9761210a074b87a26fffa949281ae5c549ab525da6567cd2"
        ),
    ),
}


CAMPAIGN_PROTOCOL = {
    "target_real_step": TARGET_REAL_STEP,
    "total_steps": 100_000_000,
    "planned_total_steps": 100_000_000,
    "schedule_total_steps": 100_000_000,
    "checkpoint_interval": 250_000,
    "max_search_steps": 30,
    "max_depth": 30,
    "model_unroll_len": 30,
    "rec_t": 20,
    "dynamic_voc_mode": "control",
    "margin_coef": 0.0,
    "pvp_coef": 0.0,
    "action_diff_coef": 1.0,
    "action_prior_weight": 0.0,
    "base_seed": 1,
    "stop_policy": {
        "kind": "external-controller-milestone",
        "target_real_step": TARGET_REAL_STEP,
        "checkpoint_window_start_inclusive": TARGET_REAL_STEP,
        "checkpoint_window_end_exclusive": TARGET_REAL_STEP + 250_000,
        "selection": "first-valid-actor-and-model-checkpoint-in-window",
        "expected_termination": True,
        "finish_marker_expected": False,
    },
}

_PATH_ENTITY_ALIASES = {
    "subject": ("sub",),
    "session": ("ses", "day"),
    "block": ("block",),
    "game": ("game",),
}


class CampaignError(RuntimeError):
    """Raised when immutable campaign or canonicalization checks fail."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _content_digest(value: Mapping[str, Any], digest_field: str) -> str:
    unsigned = dict(value)
    unsigned.pop(digest_field, None)
    return hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()


def _attach_digest(value: Mapping[str, Any], digest_field: str) -> dict[str, Any]:
    result = dict(value)
    result[digest_field] = _content_digest(result, digest_field)
    return result


def _verify_embedded_digest(value: Mapping[str, Any], digest_field: str) -> None:
    expected = value.get(digest_field)
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise CampaignError(f"{digest_field} is missing or malformed")
    actual = _content_digest(value, digest_field)
    if actual != expected:
        raise CampaignError(f"{digest_field} mismatch: {actual} != {expected}")


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())


def _atomic_directory(destination: Path, build) -> Path:
    """Build a new directory and publish it with a no-clobber rename."""

    destination = destination.expanduser().absolute()
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing destination: {destination}")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=str(parent))
    )
    try:
        build(temporary)
        _rename_noreplace(temporary, destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return destination


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without a check/rename race (Linux)."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise CampaignError("renameat2(RENAME_NOREPLACE) is required for publication")
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
        os.fsencode(destination),
        rename_noreplace,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(f"refusing to overwrite destination: {destination}")
    if error_number in {errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP}:
        _locked_atomic_rename(source, destination)
        return
    raise OSError(error_number, os.strerror(error_number), str(destination))


def _locked_atomic_rename(source: Path, destination: Path) -> None:
    """NFS fallback: cooperative O_EXCL lock plus one atomic rename."""

    lock_path = destination.parent / f".{destination.name}.publish.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise CampaignError(f"another publisher holds {lock_path}") from error
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"refusing to overwrite destination: {destination}")
        os.rename(source, destination)
    finally:
        os.close(descriptor)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _path_entity(path: Path, entity: str) -> int | None:
    text = path.as_posix().lower()
    for alias in _PATH_ENTITY_ALIASES[entity]:
        match = re.search(rf"(?:^|[/_-]){alias}[-_]?0*(\d+)(?=$|[/_.-])", text)
        if match is not None:
            return int(match.group(1))
    return None


def _parse_sha256_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-fA-F]{64}) [ *](.+)", raw_line)
        if match is None:
            raise CampaignError(f"malformed SHA256 line {path}:{line_number}")
        relative = match.group(2)
        if relative.startswith("./"):
            relative = relative[2:]
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise CampaignError(f"unsafe path in SHA256 manifest: {relative!r}")
        normalized = candidate.as_posix()
        if normalized in entries:
            raise CampaignError(f"duplicate SHA256 manifest path: {normalized}")
        entries[normalized] = match.group(1).lower()
    return entries


def _verify_external_manifest(
    path: Path, expected_manifest_sha256: str
) -> dict[str, str]:
    path = path.expanduser().resolve(strict=True)
    actual = _sha256_file(path)
    if actual != expected_manifest_sha256:
        raise CampaignError(
            f"external manifest hash mismatch for {path}: {actual} != "
            f"{expected_manifest_sha256}"
        )
    return _parse_sha256_manifest(path)


def _verify_tree_against_manifest(
    root: Path, manifest_entries: Mapping[str, str]
) -> None:
    """Require an exact regular-file path set and verify every file hash."""

    root = root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise CampaignError(f"manifest root is not a directory: {root}")
    symlinks = sorted(path for path in root.rglob("*") if path.is_symlink())
    if symlinks:
        raise CampaignError(f"immutable snapshot contains symlinks: {symlinks[:5]}")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    relative_paths = {path.relative_to(root).as_posix() for path in files}
    expected_paths = set(manifest_entries)
    if relative_paths != expected_paths:
        raise CampaignError(
            "snapshot manifest/file mismatch; "
            f"missing={sorted(relative_paths - expected_paths)}, "
            f"extra={sorted(expected_paths - relative_paths)}"
        )
    for path in files:
        relative = path.relative_to(root).as_posix()
        actual = _sha256_file(path)
        expected = manifest_entries[relative]
        if actual != expected:
            raise CampaignError(
                f"snapshot file hash mismatch for {relative}: {actual} != {expected}"
            )


def _load_production_loader():
    module_name = "_dynamic_imitation_campaign_bc_loader"
    module = sys.modules.get(module_name)
    if module is None:
        module_path = Path(__file__).resolve().parents[1] / "thinker" / "bc_loader.py"
        module_spec = importlib.util.spec_from_file_location(module_name, module_path)
        if module_spec is None or module_spec.loader is None:
            raise CampaignError(f"cannot load production behavioral loader: {module_path}")
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = module
        module_spec.loader.exec_module(module)
    return module.FrameStackedBehavioralDataLoader, module.behavioral_data_signature


def _npy_header(archive_path: Path, member_name: str) -> tuple[tuple[int, ...], str]:
    import numpy as np  # pylint: disable=import-outside-toplevel

    member = f"{member_name}.npy"
    with zipfile.ZipFile(archive_path) as archive:
        if member not in archive.namelist():
            raise CampaignError(f"{archive_path} lacks {member_name!r}")
        with archive.open(member) as source:
            version = np.lib.format.read_magic(source)
            if version == (1, 0):
                shape, _fortran, dtype = np.lib.format.read_array_header_1_0(source)
            else:
                shape, _fortran, dtype = np.lib.format.read_array_header_2_0(source)
    return tuple(int(value) for value in shape), str(dtype)


def _inspect_behavior_archive(path: Path, num_actions: int) -> dict[str, Any]:
    import numpy as np  # pylint: disable=import-outside-toplevel

    image_shape, image_dtype = _npy_header(path, "image")
    if not image_shape:
        raise CampaignError(f"image array has no time dimension: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if "action" not in archive.files:
            raise CampaignError(f"behavior archive lacks action array: {path}")
        actions = np.asarray(archive["action"])
    if actions.ndim == 1 or (actions.ndim == 2 and actions.shape[1] == 1):
        scalar = actions.reshape(-1)
        if not np.all(np.isfinite(scalar)) or not np.all(scalar == np.floor(scalar)):
            raise CampaignError(f"non-integral scalar actions in {path}")
        if np.any(scalar < 0) or np.any(scalar >= num_actions):
            raise CampaignError(f"scalar action outside runtime action space in {path}")
        encoding = "scalar"
        action_width = 1
    elif actions.ndim == 2:
        action_width = int(actions.shape[1])
        if action_width != num_actions:
            raise CampaignError(
                f"action width mismatch in {path}: {action_width} != {num_actions}"
            )
        if not np.all(np.logical_or(actions == 0, actions == 1)):
            raise CampaignError(f"non-binary one-hot action matrix in {path}")
        if not np.all(np.sum(actions, axis=1) == 1):
            raise CampaignError(f"invalid one-hot action rows in {path}")
        encoding = "strict_one_hot"
    else:
        raise CampaignError(f"unsupported action shape in {path}: {actions.shape}")
    action_rows = int(actions.shape[0])
    if int(image_shape[0]) != action_rows:
        raise CampaignError(
            f"image/action length mismatch in {path}: {image_shape[0]} != {action_rows}"
        )
    return {
        "action_encoding": encoding,
        "action_rows": action_rows,
        "action_width": action_width,
        "image_shape": list(image_shape),
        "image_dtype": image_dtype,
    }


def build_behavioral_data_manifest(
    data_root: Path,
    sha256_manifest: Path,
    *,
    expected_sha256_manifest_sha256: str,
) -> dict[str, Any]:
    """Validate all 72 cells and return a deterministic content manifest."""

    data_root = data_root.expanduser().resolve(strict=True)
    if not data_root.is_dir():
        raise CampaignError(f"behavior root is not a directory: {data_root}")
    sha256_manifest = sha256_manifest.expanduser().resolve(strict=True)
    external_hashes = _verify_external_manifest(
        sha256_manifest, expected_sha256_manifest_sha256
    )
    raw_paths = sorted(data_root.rglob("*.npz"))
    if len(raw_paths) != 198:
        raise CampaignError(f"expected 198 behavioral NPZ files, found {len(raw_paths)}")
    relative_paths = [path.relative_to(data_root).as_posix() for path in raw_paths]
    if set(relative_paths) != set(external_hashes):
        missing = sorted(set(relative_paths) - set(external_hashes))
        extra = sorted(set(external_hashes) - set(relative_paths))
        raise CampaignError(
            f"behavior SHA256 manifest/file mismatch; missing={missing}, extra={extra}"
        )

    groups: dict[tuple[int, int, int], list[Path]] = {}
    parsed: dict[Path, tuple[int, int, int, int]] = {}
    for path in raw_paths:
        subject = _path_entity(path, "subject")
        session = _path_entity(path, "session")
        block = _path_entity(path, "block")
        game_id = _path_entity(path, "game")
        if None in (subject, session, block, game_id):
            raise CampaignError(f"could not parse all entities from {path}")
        assert subject is not None and session is not None
        assert block is not None and game_id is not None
        if subject not in SUBJECTS or session not in SESSIONS or game_id not in GAME_BY_ID:
            raise CampaignError(
                f"out-of-campaign behavioral identity in {path}: "
                f"subject={subject}, session={session}, game={game_id}"
            )
        identity = (game_id, subject, session)
        groups.setdefault(identity, []).append(path)
        parsed[path] = (game_id, subject, session, block)

    expected_cells = {
        (game.game_id, subject, session)
        for game in GAMES
        for subject in SUBJECTS
        for session in SESSIONS
    }
    if set(groups) != expected_cells:
        raise CampaignError(
            f"72-cell identity mismatch; missing={sorted(expected_cells - set(groups))}, "
            f"extra={sorted(set(groups) - expected_cells)}"
        )

    Loader, signature_function = _load_production_loader()
    cells: list[dict[str, Any]] = []
    for game in GAMES:
        for subject in SUBJECTS:
            for session in SESSIONS:
                identity = (game.game_id, subject, session)
                expected_count = EXPECTED_SESSION_FILE_COUNTS[session]
                if len(groups[identity]) != expected_count:
                    raise CampaignError(
                        f"wrong file count for {identity}: "
                        f"{len(groups[identity])} != {expected_count}"
                    )
                loader = Loader(
                    base_path=data_root,
                    subjects=(subject,),
                    sessions=(session,),
                    game_id=game.game_id,
                    num_actions=game.num_actions,
                    scored_length=4,
                    frame_stack_n=4,
                    target_size=(84, 84),
                    grayscale=False,
                    normalize=False,
                    seed=1,
                )
                loader_files = {
                    Path(value).resolve().relative_to(data_root).as_posix()
                    for value in loader.data_files
                }
                expected_files = {
                    value.relative_to(data_root).as_posix() for value in groups[identity]
                }
                if loader_files != expected_files:
                    raise CampaignError(
                        f"production loader file mapping mismatch for {identity}"
                    )
                if loader.num_windows <= 0:
                    raise CampaignError(f"no length-4 windows for {identity}")
                cells.append(
                    {
                        "game_id": game.game_id,
                        "game_label": game.game_label,
                        "env_name": game.env_name,
                        "num_actions": game.num_actions,
                        "subject": subject,
                        "session": session,
                        "file_count": expected_count,
                        "scored_length_4_windows": int(loader.num_windows),
                        "files": sorted(expected_files),
                    }
                )

    file_records: list[dict[str, Any]] = []
    total_bytes = 0
    total_action_rows = 0
    for path in raw_paths:
        relative = path.relative_to(data_root).as_posix()
        actual_hash = _sha256_file(path)
        expected_hash = external_hashes[relative]
        if actual_hash != expected_hash:
            raise CampaignError(
                f"behavior file hash mismatch for {relative}: "
                f"{actual_hash} != {expected_hash}"
            )
        game_id, subject, session, block = parsed[path]
        inspection = _inspect_behavior_archive(path, GAME_BY_ID[game_id].num_actions)
        size_bytes = path.stat().st_size
        total_bytes += size_bytes
        total_action_rows += int(inspection["action_rows"])
        file_records.append(
            {
                "path": relative,
                "sha256": actual_hash,
                "size_bytes": size_bytes,
                "game_id": game_id,
                "subject": subject,
                "session": session,
                "block": block,
                **inspection,
            }
        )

    manifest = {
        "schema_version": 1,
        "kind": "dynamic-imitation-behavioral-data-manifest",
        "root": str(data_root),
        "source_sha256_manifest": str(sha256_manifest),
        "source_sha256_manifest_sha256": expected_sha256_manifest_sha256,
        "expected_subjects": list(SUBJECTS),
        "expected_sessions": list(SESSIONS),
        "expected_games": [
            {
                "game_id": game.game_id,
                "game_label": game.game_label,
                "env_name": game.env_name,
                "num_actions": game.num_actions,
            }
            for game in GAMES
        ],
        "scored_length_audited": 4,
        "cell_count": len(cells),
        "file_count": len(file_records),
        "total_bytes": total_bytes,
        "total_action_rows": total_action_rows,
        "cells": cells,
        "files": file_records,
    }
    cell_by_identity = {
        (cell["game_id"], cell["subject"], cell["session"]): cell
        for cell in cells
    }
    folds: list[dict[str, Any]] = []
    for game in GAMES:
        for subject in SUBJECTS:
            for leave_session in SESSIONS:
                train_sessions = tuple(
                    session for session in SESSIONS if session != leave_session
                )
                loader = Loader(
                    base_path=data_root,
                    subjects=(subject,),
                    sessions=train_sessions,
                    game_id=game.game_id,
                    num_actions=game.num_actions,
                    scored_length=4,
                    frame_stack_n=4,
                    target_size=(84, 84),
                    grayscale=False,
                    normalize=False,
                    seed=1,
                )
                train_files = sorted(
                    value
                    for session in train_sessions
                    for value in cell_by_identity[
                        (game.game_id, subject, session)
                    ]["files"]
                )
                loader_files = sorted(
                    Path(value).resolve().relative_to(data_root).as_posix()
                    for value in loader.data_files
                )
                if loader_files != train_files:
                    raise CampaignError(
                        "production loader fold mapping mismatch for "
                        f"game={game.game_id}, subject={subject}, "
                        f"leave_session={leave_session}"
                    )
                folds.append(
                    {
                        "game_id": game.game_id,
                        "subject": subject,
                        "leave_session": leave_session,
                        "train_sessions": list(train_sessions),
                        "train_file_count": len(train_files),
                        "holdout_file_count": int(
                            cell_by_identity[
                                (game.game_id, subject, leave_session)
                            ]["file_count"]
                        ),
                        "scored_length_4_train_windows": int(loader.num_windows),
                        "behavioral_data_signature": signature_function(
                            loader, data_root
                        ),
                    }
                )
    if len(folds) != 72:
        raise CampaignError(f"expected 72 leave-one-session-out folds, got {len(folds)}")
    manifest["folds"] = folds
    return _attach_digest(manifest, "manifest_sha256")


def _canonical_xpid(game: GameSpec, subject: int, leave_session: int) -> str:
    return (
        f"dynamic-imitation-{game.game_label}-voc-"
        f"sub{subject}-leavesession{leave_session}"
    )


def _task_id(game: GameSpec, subject: int, leave_session: int) -> str:
    return f"g{game.game_id}-{game.game_label}-s{subject:02d}-h{leave_session}"


def _existing_by_identity() -> dict[tuple[int, int, int], ExistingRunSpec]:
    return {(spec.game_id, 1, 4): spec for spec in EXISTING_RUNS.values()}


def _cell_index(data_manifest: Mapping[str, Any]) -> dict[tuple[int, int, int], Any]:
    result = {}
    for cell in data_manifest["cells"]:
        key = (int(cell["game_id"]), int(cell["subject"]), int(cell["session"]))
        if key in result:
            raise CampaignError(f"duplicate behavior cell in manifest: {key}")
        result[key] = cell
    return result


def _fold_index(data_manifest: Mapping[str, Any]) -> dict[tuple[int, int, int], Any]:
    result = {}
    for fold in data_manifest.get("folds", ()):
        key = (
            int(fold["game_id"]),
            int(fold["subject"]),
            int(fold["leave_session"]),
        )
        if key in result:
            raise CampaignError(f"duplicate behavioral fold in manifest: {key}")
        result[key] = fold
    if len(result) != 72:
        raise CampaignError(f"behavior manifest must contain 72 folds, found {len(result)}")
    return result


def build_campaign_payloads(
    data_manifest: Mapping[str, Any],
    *,
    source_root: Path,
    source_manifest: Path,
    source_manifest_sha256: str,
) -> tuple[dict[str, Any], bytes]:
    """Return the all-72 manifest and compact 70-task JSONL ledger."""

    _verify_embedded_digest(data_manifest, "manifest_sha256")
    if int(data_manifest.get("cell_count", -1)) != 72:
        raise CampaignError("behavior manifest must contain exactly 72 cells")
    cells = _cell_index(data_manifest)
    folds = _fold_index(data_manifest)
    existing = _existing_by_identity()
    tasks: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for game in GAMES:
        for subject in SUBJECTS:
            for leave_session in SESSIONS:
                campaign_index = game.game_id * 24 + (subject - 1) * 4 + leave_session - 1
                train_sessions = [value for value in SESSIONS if value != leave_session]
                train_files = sorted(
                    file_name
                    for session in train_sessions
                    for file_name in cells[(game.game_id, subject, session)]["files"]
                )
                holdout_files = list(
                    cells[(game.game_id, subject, leave_session)]["files"]
                )
                identity = (game.game_id, subject, leave_session)
                fold = folds[identity]
                if list(fold["train_sessions"]) != train_sessions:
                    raise CampaignError(f"non-complement train sessions in fold {identity}")
                existing_spec = existing.get(identity)
                task = {
                    "schema_version": 1,
                    "task_id": _task_id(game, subject, leave_session),
                    "campaign_index": campaign_index,
                    "array_task_id": None,
                    "state": (
                        "existing_milestone_available"
                        if existing_spec is not None
                        else "pending_submission"
                    ),
                    "logical": {
                        "xpid": _canonical_xpid(game, subject, leave_session),
                        "game_id": game.game_id,
                        "game_label": game.game_label,
                        "env_name": game.env_name,
                        "num_actions": game.num_actions,
                        "subject": subject,
                        "leave_session": leave_session,
                        "train_sessions": train_sessions,
                    },
                    "protocol": dict(CAMPAIGN_PROTOCOL),
                    "behavior": {
                        "manifest": DATA_MANIFEST_NAME,
                        "manifest_sha256": data_manifest["manifest_sha256"],
                        "behavioral_data_signature": fold[
                            "behavioral_data_signature"
                        ],
                        "scored_length_4_train_windows": fold[
                            "scored_length_4_train_windows"
                        ],
                        "train_file_count": len(train_files),
                        "holdout_file_count": len(holdout_files),
                        "train_files": train_files,
                        "holdout_files": holdout_files,
                    },
                    "existing_source": (
                        None
                        if existing_spec is None
                        else {
                            "xpid": existing_spec.source_xpid,
                            "source_total_steps": 100_000_000,
                            "milestone_real_step": TARGET_REAL_STEP,
                            "actor_sha256": existing_spec.actor_sha256,
                            "model_sha256": existing_spec.model_sha256,
                        }
                    ),
                }
                if existing_spec is None:
                    task["array_task_id"] = len(pending)
                    pending.append(task)
                tasks.append(task)

    if [task["campaign_index"] for task in tasks] != list(range(72)):
        raise CampaignError("campaign indices are not a contiguous 0..71 mapping")
    if len(pending) != 70:
        raise CampaignError(f"expected 70 pending tasks, built {len(pending)}")
    if [task["array_task_id"] for task in pending] != list(range(70)):
        raise CampaignError("pending array task IDs are not a contiguous 0..69 mapping")
    if {task["campaign_index"] for task in pending} != set(range(72)) - {3, 27}:
        raise CampaignError("pending task mapping does not skip only indices 3 and 27")

    ledger_bytes = b"".join(_canonical_json_bytes(task) for task in pending)
    campaign = {
        "schema_version": 1,
        "kind": "dynamic-imitation-72fold-5m-campaign",
        "target_real_step": TARGET_REAL_STEP,
        "source": {
            "root": str(source_root.expanduser().resolve(strict=True)),
            "manifest": str(source_manifest.expanduser().resolve(strict=True)),
            "manifest_sha256": source_manifest_sha256,
        },
        "behavior": {
            "root": data_manifest["root"],
            "source_sha256_manifest": data_manifest["source_sha256_manifest"],
            "source_sha256_manifest_sha256": data_manifest[
                "source_sha256_manifest_sha256"
            ],
            "audit_manifest": DATA_MANIFEST_NAME,
            "audit_manifest_sha256": data_manifest["manifest_sha256"],
        },
        "task_count": len(tasks),
        "existing_milestone_count": len(tasks) - len(pending),
        "pending_task_count": len(pending),
        "pending_ledger": PENDING_LEDGER_NAME,
        "pending_ledger_sha256": hashlib.sha256(ledger_bytes).hexdigest(),
        "tasks": tasks,
    }
    return _attach_digest(campaign, "campaign_manifest_sha256"), ledger_bytes


def _write_sha256sums(directory: Path, file_names: Iterable[str]) -> None:
    lines = []
    for file_name in sorted(file_names):
        lines.append(f"{_sha256_file(directory / file_name)}  {file_name}\n")
    _write_bytes(directory / SHA256SUMS_NAME, "".join(lines).encode("utf-8"))


def _seal_files(directory: Path, file_names: Iterable[str]) -> None:
    """Make published inputs non-writable while leaving the campaign root extensible."""

    for file_name in file_names:
        os.chmod(directory / file_name, 0o444)


def build_campaign_directory(
    destination: Path,
    *,
    data_root: Path = DEFAULT_BEHAVIOR_ROOT,
    behavior_sha256_manifest: Path = DEFAULT_BEHAVIOR_SHA256_MANIFEST,
    behavior_sha256_manifest_sha256: str = EXPECTED_BEHAVIOR_SHA256_MANIFEST_SHA256,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    source_manifest: Path = DEFAULT_SOURCE_MANIFEST,
    source_manifest_sha256: str = EXPECTED_SOURCE_MANIFEST_SHA256,
) -> Path:
    source_root = source_root.expanduser().resolve(strict=True)
    source_manifest = source_manifest.expanduser().resolve(strict=True)
    source_entries = _verify_external_manifest(source_manifest, source_manifest_sha256)
    _verify_tree_against_manifest(source_root, source_entries)
    data_manifest = build_behavioral_data_manifest(
        data_root,
        behavior_sha256_manifest,
        expected_sha256_manifest_sha256=behavior_sha256_manifest_sha256,
    )
    campaign_manifest, ledger_bytes = build_campaign_payloads(
        data_manifest,
        source_root=source_root,
        source_manifest=source_manifest,
        source_manifest_sha256=source_manifest_sha256,
    )

    def build(temporary: Path) -> None:
        _write_bytes(temporary / DATA_MANIFEST_NAME, _canonical_json_bytes(data_manifest))
        _write_bytes(
            temporary / CAMPAIGN_MANIFEST_NAME,
            _canonical_json_bytes(campaign_manifest),
        )
        _write_bytes(temporary / PENDING_LEDGER_NAME, ledger_bytes)
        _write_sha256sums(
            temporary,
            (DATA_MANIFEST_NAME, CAMPAIGN_MANIFEST_NAME, PENDING_LEDGER_NAME),
        )
        _seal_files(
            temporary,
            (
                DATA_MANIFEST_NAME,
                CAMPAIGN_MANIFEST_NAME,
                PENDING_LEDGER_NAME,
                SHA256SUMS_NAME,
            ),
        )

    return _atomic_directory(destination, build)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


def _verify_sha256sums(directory: Path) -> None:
    entries = _parse_sha256_manifest(directory / SHA256SUMS_NAME)
    if not entries:
        raise CampaignError(f"empty {SHA256SUMS_NAME} in {directory}")
    for relative, expected in entries.items():
        path = directory / relative
        if not path.is_file():
            raise CampaignError(f"missing sealed artifact: {path}")
        actual = _sha256_file(path)
        if actual != expected:
            raise CampaignError(f"sealed artifact hash mismatch: {path}")


def verify_campaign_directory(
    directory: Path,
    *,
    data_root: Path = DEFAULT_BEHAVIOR_ROOT,
    behavior_sha256_manifest: Path = DEFAULT_BEHAVIOR_SHA256_MANIFEST,
    behavior_sha256_manifest_sha256: str = EXPECTED_BEHAVIOR_SHA256_MANIFEST_SHA256,
) -> dict[str, Any]:
    directory = directory.expanduser().resolve(strict=True)
    _verify_sha256sums(directory)
    saved_data = _read_json(directory / DATA_MANIFEST_NAME)
    saved_campaign = _read_json(directory / CAMPAIGN_MANIFEST_NAME)
    _verify_embedded_digest(saved_data, "manifest_sha256")
    _verify_embedded_digest(saved_campaign, "campaign_manifest_sha256")
    rebuilt_data = build_behavioral_data_manifest(
        data_root,
        behavior_sha256_manifest,
        expected_sha256_manifest_sha256=behavior_sha256_manifest_sha256,
    )
    if rebuilt_data != saved_data:
        raise CampaignError("saved behavioral audit manifest differs from current snapshot")
    ledger_bytes = (directory / PENDING_LEDGER_NAME).read_bytes()
    if hashlib.sha256(ledger_bytes).hexdigest() != saved_campaign["pending_ledger_sha256"]:
        raise CampaignError("pending ledger hash differs from campaign manifest")
    ledger = [json.loads(line) for line in ledger_bytes.splitlines() if line]
    if len(ledger) != 70:
        raise CampaignError(f"pending ledger must contain 70 tasks, found {len(ledger)}")
    if ledger != [task for task in saved_campaign["tasks"] if task["state"] == "pending_submission"]:
        raise CampaignError("pending ledger rows differ from all-task campaign mapping")
    source = saved_campaign.get("source")
    if not isinstance(source, Mapping):
        raise CampaignError("campaign manifest lacks source snapshot mapping")
    source_root = Path(str(source.get("root", ""))).expanduser().resolve(strict=True)
    source_manifest = Path(str(source.get("manifest", ""))).expanduser().resolve(
        strict=True
    )
    source_manifest_sha256 = str(source.get("manifest_sha256", ""))
    source_entries = _verify_external_manifest(
        source_manifest, source_manifest_sha256
    )
    _verify_tree_against_manifest(source_root, source_entries)
    expected_campaign, expected_ledger = build_campaign_payloads(
        rebuilt_data,
        source_root=source_root,
        source_manifest=source_manifest,
        source_manifest_sha256=source_manifest_sha256,
    )
    if saved_campaign != expected_campaign:
        raise CampaignError("campaign manifest task mapping is not canonical")
    if ledger_bytes != expected_ledger:
        raise CampaignError("pending ledger bytes are not the canonical 70-task mapping")
    return {
        "campaign_manifest_sha256": saved_campaign["campaign_manifest_sha256"],
        "behavioral_manifest_sha256": saved_data["manifest_sha256"],
        "task_count": 72,
        "pending_task_count": 70,
    }


def _parse_id_list(value: Any, label: str) -> tuple[int, ...]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        parts = list(value)
    else:
        parts = [value]
    try:
        result = tuple(int(part) for part in parts)
    except (TypeError, ValueError) as error:
        raise CampaignError(f"invalid {label}: {value!r}") from error
    if not result:
        raise CampaignError(f"empty {label}")
    return result


def _value_equal(left: Any, right: Any) -> bool:
    try:
        result = left == right
    except Exception:
        return False
    if isinstance(result, bool):
        return result
    try:
        return bool(result)
    except Exception:
        return False


def _validate_flags(
    flags: Mapping[str, Any], spec: ExistingRunSpec, *, source_total_steps: int
) -> None:
    game = spec.game
    expected = {
        "name": game.env_name,
        "icopro_game_id": game.game_id,
        "max_search_steps": 30,
        "max_depth": 30,
        "model_unroll_len": 30,
        "action_prior_weight": 0.0,
        "icopro_margin_coef": 0.0,
        "icopro_pvp_coef": 0.0,
        "icopro_action_diff_coef": 1.0,
        "base_seed": 1,
        "dynamic_search": True,
        "dynamic_factorized_control": True,
        "dynamic_voc_mode": "control",
        "schedule_total_steps": 100_000_000,
        "total_steps": source_total_steps,
        "batch_length": 4,
        "frame_stack_n": 4,
        "grayscale": False,
        "train_model": True,
        "envpool": True,
    }
    for key, expected_value in expected.items():
        if key not in flags or not _value_equal(flags[key], expected_value):
            raise CampaignError(
                f"checkpoint/config {key} mismatch: {flags.get(key)!r} != "
                f"{expected_value!r}"
            )
    if _parse_id_list(flags.get("icopro_subjects"), "icopro_subjects") != (1,):
        raise CampaignError("existing canonical source must be subject 1")
    if _parse_id_list(flags.get("icopro_train_sessions"), "train sessions") != (
        1,
        2,
        3,
    ):
        raise CampaignError("existing canonical source must train on sessions 1,2,3")
    if _parse_id_list(flags.get("icopro_holdout_sessions"), "holdout sessions") != (4,):
        raise CampaignError("existing canonical source must hold out session 4")


def _validate_checkpoint_payloads(
    actor: Mapping[str, Any], model: Mapping[str, Any], spec: ExistingRunSpec
) -> Mapping[str, Any]:
    for label, checkpoint in (("actor", actor), ("model", model)):
        if int(checkpoint.get("real_step", -1)) != TARGET_REAL_STEP:
            raise CampaignError(
                f"{label} payload real_step is not {TARGET_REAL_STEP}: "
                f"{checkpoint.get('real_step')!r}"
            )
        if not isinstance(checkpoint.get("flags"), Mapping):
            raise CampaignError(f"{label} payload lacks embedded flags")
    actor_required = (
        "actor_net_state_dict",
        "actor_net_optimizer_state_dict",
        "actor_net_scheduler_state_dict",
        "imitation_data_signature",
        "imitation_rng_state",
        "python_rng_state",
        "numpy_rng_state",
        "torch_rng_state",
    )
    model_required = (
        "model_net_state_dict",
        "model_net_optimizer_p_state_dict",
        "model_net_scheduler_p_state_dict",
        "model_net_optimizer_m_state_dict",
        "model_net_scheduler_m_state_dict",
    )
    for key in actor_required:
        if key not in actor or actor[key] is None:
            raise CampaignError(f"actor payload lacks required state {key}")
    for key in model_required:
        if key not in model or model[key] is None:
            raise CampaignError(f"model payload lacks required state {key}")
    actor_flags = actor["flags"]
    model_flags = model["flags"]
    if set(actor_flags) != set(model_flags):
        raise CampaignError("actor/model embedded flag keysets differ")
    for key in actor_flags:
        if not _value_equal(actor_flags[key], model_flags[key]):
            raise CampaignError(f"actor/model embedded flag mismatch at {key}")
    _validate_flags(actor_flags, spec, source_total_steps=100_000_000)
    signature = actor.get("imitation_data_signature")
    if not isinstance(signature, str) or not re.fullmatch(r"[0-9a-f]{64}", signature):
        raise CampaignError("actor imitation_data_signature is malformed")
    return actor_flags


def _validate_config_and_meta(
    source_dir: Path,
    flags: Mapping[str, Any],
    spec: ExistingRunSpec,
) -> None:
    import yaml  # pylint: disable=import-outside-toplevel

    with (source_dir / "config_c.yaml").open("r", encoding="utf-8") as source:
        config = yaml.safe_load(source)
    if not isinstance(config, Mapping):
        raise CampaignError("config_c.yaml is not a mapping")
    _validate_flags(config, spec, source_total_steps=100_000_000)
    if set(config) != set(flags):
        raise CampaignError("config/checkpoint embedded flag keysets differ")
    allowed_path_differences = {"ckpdir", "savedir"}
    differences = {
        key for key in flags if not _value_equal(config[key], flags[key])
    }
    if not differences.issubset(allowed_path_differences):
        raise CampaignError(
            f"config/checkpoint flags differ outside relocated paths: "
            f"{sorted(differences)}"
        )
    if str(config.get("xpid")) != spec.source_xpid:
        raise CampaignError("source config xpid does not match the known legacy run")

    for meta_name in ("meta.json", "meta_model.json"):
        meta = _read_json(source_dir / meta_name)
        if not isinstance(meta, Mapping) or not isinstance(meta.get("args"), Mapping):
            raise CampaignError(f"{meta_name} lacks args mapping")
        _validate_flags(meta["args"], spec, source_total_steps=100_000_000)
        if str(meta["args"].get("xpid")) != spec.source_xpid:
            raise CampaignError(f"{meta_name} xpid mismatch")


def _truncate_csv(
    source_path: Path,
    destination_path: Path,
    *,
    require_exact_target: bool,
) -> dict[str, Any]:
    with source_path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.reader(source)
        try:
            header = next(reader)
        except StopIteration as error:
            raise CampaignError(f"empty CSV: {source_path}") from error
        if header.count("real_step") != 1:
            raise CampaignError(f"CSV must have one real_step column: {source_path}")
        step_column = header.index("real_step")
        selected: list[list[str]] = []
        exact_rows: list[list[str]] = []
        previous_step = -1
        source_row_count = 0
        for line_number, row in enumerate(reader, 2):
            source_row_count += 1
            if len(row) != len(header):
                raise CampaignError(
                    f"malformed CSV row {source_path}:{line_number}: "
                    f"{len(row)} columns != {len(header)}"
                )
            try:
                step_value = float(row[step_column])
            except ValueError as error:
                raise CampaignError(
                    f"non-numeric real_step at {source_path}:{line_number}"
                ) from error
            if not math.isfinite(step_value) or not step_value.is_integer():
                raise CampaignError(
                    f"invalid real_step at {source_path}:{line_number}: {step_value}"
                )
            step = int(step_value)
            if step < previous_step:
                raise CampaignError(f"real_step decreases in {source_path}:{line_number}")
            previous_step = step
            if step <= TARGET_REAL_STEP:
                selected.append(row)
                if step == TARGET_REAL_STEP:
                    exact_rows.append(row)
        if not selected:
            raise CampaignError(f"CSV has no rows through step {TARGET_REAL_STEP}: {source_path}")
        if require_exact_target and len(exact_rows) != 1:
            raise CampaignError(
                f"actor CSV must have exactly one step-{TARGET_REAL_STEP} row; "
                f"found {len(exact_rows)}"
            )

    with destination_path.open("x", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(selected)
        output.flush()
        os.fsync(output.fileno())
    last = selected[-1]
    return {
        "source_row_count": source_row_count,
        "canonical_row_count": len(selected),
        "last_real_step": int(float(last[step_column])),
        "header": header,
        "last_row": dict(zip(header, last)),
        "exact_target_row": (
            None if not exact_rows else dict(zip(header, exact_rows[0]))
        ),
    }


def _validate_noop_target_row(csv_info: Mapping[str, Any]) -> dict[str, Any]:
    row = csv_info.get("exact_target_row")
    if not isinstance(row, Mapping):
        raise CampaignError("actor CSV lacks the exact target row")
    keys = (
        "interaction/noop_action_index",
        "interaction/noop_supported",
        "interaction/real_action_count",
        "interaction/noop_count",
        "interaction/noop_frequency",
    )
    missing = [key for key in keys if key not in row]
    if missing:
        raise CampaignError(f"actor CSV lacks online NOOP columns: {missing}")
    try:
        noop_index = int(float(row[keys[0]]))
        supported = int(float(row[keys[1]]))
        real_count = int(float(row[keys[2]]))
        noop_count = int(float(row[keys[3]]))
        frequency = float(row[keys[4]])
    except (TypeError, ValueError) as error:
        raise CampaignError("actor target-row NOOP values are not numeric") from error
    if noop_index != 0 or supported != 1 or real_count <= 0:
        raise CampaignError("actor target-row NOOP support metadata is invalid")
    if noop_count < 0 or noop_count > real_count or not math.isfinite(frequency):
        raise CampaignError("actor target-row NOOP count/frequency is invalid")
    expected_frequency = noop_count / real_count
    if not math.isclose(frequency, expected_frequency, rel_tol=1e-12, abs_tol=1e-12):
        raise CampaignError(
            f"NOOP frequency does not equal count/real_count: "
            f"{frequency} != {expected_frequency}"
        )
    return {
        "noop_action_index": noop_index,
        "noop_supported": supported,
        "real_action_count": real_count,
        "noop_count": noop_count,
        "noop_frequency": frequency,
    }


def _verify_behavior_signature(
    checkpoint: Mapping[str, Any], spec: ExistingRunSpec, behavior_root: Path
) -> dict[str, Any]:
    Loader, signature_function = _load_production_loader()
    flags = checkpoint["flags"]
    loader = Loader(
        base_path=behavior_root,
        subjects=(1,),
        sessions=(1, 2, 3),
        game_id=spec.game_id,
        num_actions=spec.game.num_actions,
        scored_length=int(flags["batch_length"]),
        frame_stack_n=int(flags["frame_stack_n"]),
        target_size=(84, 84),
        grayscale=bool(flags["grayscale"]),
        normalize=False,
        seed=int(flags["base_seed"]),
    )
    actual = signature_function(loader, behavior_root)
    expected = checkpoint["imitation_data_signature"]
    if actual != expected:
        raise CampaignError(
            f"immutable behavior snapshot signature mismatch: {actual} != {expected}"
        )
    return {
        "imitation_data_signature": actual,
        "train_file_count": len(loader.data_files),
        "train_window_count": int(loader.num_windows),
    }


def canonicalize_existing_run(
    spec: ExistingRunSpec,
    *,
    source_logs_root: Path = DEFAULT_LOGS_ROOT,
    output_root: Path = DEFAULT_LOGS_ROOT,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    source_manifest: Path = DEFAULT_SOURCE_MANIFEST,
    source_manifest_sha256: str = EXPECTED_SOURCE_MANIFEST_SHA256,
    behavior_root: Path = DEFAULT_BEHAVIOR_ROOT,
    behavior_sha256_manifest: Path = DEFAULT_BEHAVIOR_SHA256_MANIFEST,
    behavior_sha256_manifest_sha256: str = EXPECTED_BEHAVIOR_SHA256_MANIFEST_SHA256,
) -> Path:
    """Validate and atomically publish one known exact 5M milestone."""

    source_logs_root = source_logs_root.expanduser().resolve(strict=True)
    source_dir = (source_logs_root / spec.source_xpid).resolve(strict=True)
    if source_dir.parent != source_logs_root:
        raise CampaignError("source run escaped the logs root")
    output_root = output_root.expanduser().absolute()
    destination = output_root / spec.canonical_xpid
    source_root = source_root.expanduser().resolve(strict=True)
    source_manifest = source_manifest.expanduser().resolve(strict=True)
    behavior_root = behavior_root.expanduser().resolve(strict=True)
    behavior_sha256_manifest = behavior_sha256_manifest.expanduser().resolve(strict=True)
    source_entries = _verify_external_manifest(source_manifest, source_manifest_sha256)
    _verify_tree_against_manifest(source_root, source_entries)
    behavior_entries = _verify_external_manifest(
        behavior_sha256_manifest, behavior_sha256_manifest_sha256
    )
    _verify_tree_against_manifest(behavior_root, behavior_entries)

    actor_name = f"ckp_actor.tar_step_{TARGET_REAL_STEP}"
    model_name = f"ckp_model.tar_step_{TARGET_REAL_STEP}"
    required_sources = {
        actor_name,
        model_name,
        "config_c.yaml",
        "meta.json",
        "meta_model.json",
        "logs.csv",
        "logs_model.csv",
    }
    for file_name in required_sources:
        if not (source_dir / file_name).is_file():
            raise CampaignError(f"required source artifact is missing: {source_dir / file_name}")
    actual_actor_hash = _sha256_file(source_dir / actor_name)
    actual_model_hash = _sha256_file(source_dir / model_name)
    if actual_actor_hash != spec.actor_sha256:
        raise CampaignError(
            f"known actor hash mismatch: {actual_actor_hash} != {spec.actor_sha256}"
        )
    if actual_model_hash != spec.model_sha256:
        raise CampaignError(
            f"known model hash mismatch: {actual_model_hash} != {spec.model_sha256}"
        )

    import torch  # pylint: disable=import-outside-toplevel

    # Hash verification happens before deserializing these trusted local payloads.
    actor = torch.load(source_dir / actor_name, map_location="cpu", weights_only=False)
    model = torch.load(source_dir / model_name, map_location="cpu", weights_only=False)
    if not isinstance(actor, Mapping) or not isinstance(model, Mapping):
        raise CampaignError("checkpoint payload is not a mapping")
    flags = _validate_checkpoint_payloads(actor, model, spec)
    _validate_config_and_meta(source_dir, flags, spec)
    behavior_signature = _verify_behavior_signature(actor, spec, behavior_root)

    def build(temporary: Path) -> None:
        for file_name in (
            actor_name,
            model_name,
            "config_c.yaml",
            "meta.json",
            "meta_model.json",
        ):
            shutil.copyfile(source_dir / file_name, temporary / file_name)
        actor_csv = _truncate_csv(
            source_dir / "logs.csv",
            temporary / "logs.csv",
            require_exact_target=True,
        )
        model_csv = _truncate_csv(
            source_dir / "logs_model.csv",
            temporary / "logs_model.csv",
            require_exact_target=False,
        )
        noop = _validate_noop_target_row(actor_csv)
        if model_csv["last_real_step"] > TARGET_REAL_STEP:
            raise CampaignError("truncated model CSV exceeds the target step")
        copied_names = (
            actor_name,
            model_name,
            "config_c.yaml",
            "meta.json",
            "meta_model.json",
            "logs.csv",
            "logs_model.csv",
        )
        artifacts = {
            file_name: {
                "size_bytes": (temporary / file_name).stat().st_size,
                "sha256": _sha256_file(temporary / file_name),
            }
            for file_name in copied_names
        }
        milestone = {
            "schema_version": 1,
            "kind": "dynamic-imitation-canonical-milestone",
            "target_real_step": TARGET_REAL_STEP,
            "logical": {
                "xpid": spec.canonical_xpid,
                "game_id": spec.game_id,
                "game_label": spec.game.game_label,
                "env_name": spec.game.env_name,
                "subject": 1,
                "leave_session": 4,
                "train_sessions": [1, 2, 3],
            },
            "source_run": {
                "root": str(source_logs_root),
                "xpid": spec.source_xpid,
                "configured_total_steps": 100_000_000,
                "milestone_real_step": TARGET_REAL_STEP,
            },
            "source_snapshot": {
                "root": str(source_root),
                "manifest": str(source_manifest),
                "manifest_sha256": source_manifest_sha256,
            },
            "behavior_snapshot": {
                "root": str(behavior_root),
                "manifest": str(behavior_sha256_manifest),
                "manifest_sha256": behavior_sha256_manifest_sha256,
                **behavior_signature,
            },
            "validation": {
                "known_checkpoint_hashes_verified": True,
                "actor_payload_real_step": int(actor["real_step"]),
                "model_payload_real_step": int(model["real_step"]),
                "actor_csv_last_real_step": actor_csv["last_real_step"],
                "model_csv_last_real_step": model_csv["last_real_step"],
                "embedded_flags_match": True,
                "config_match_except_relocated_savedir_ckpdir": True,
                "online_noop_at_target": noop,
            },
            "artifacts": artifacts,
            "wandb": {
                "entity": "jmme425",
                "project": "thinker",
                "source_run_id": spec.source_xpid,
                "derived_url": f"https://wandb.ai/jmme425/thinker/runs/{spec.source_xpid}",
                "online_noop_metric_keys_present_in_canonical_log": True,
                "remote_presence_claimed_by_canonicalizer": False,
            },
            "finish_present": False,
            "training_completion_claimed": False,
            "published_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_bytes(temporary / MILESTONE_NAME, _canonical_json_bytes(milestone))
        _write_sha256sums(temporary, (*copied_names, MILESTONE_NAME))
        if (temporary / "finish").exists():
            raise CampaignError("canonicalizer must never create a finish marker")
        _seal_files(
            temporary,
            (*copied_names, MILESTONE_NAME, SHA256SUMS_NAME),
        )

    published = _atomic_directory(destination, build)
    return published


def verify_canonical_run(directory: Path, spec: ExistingRunSpec) -> dict[str, Any]:
    directory = directory.expanduser().resolve(strict=True)
    if directory.name != spec.canonical_xpid:
        raise CampaignError(
            f"canonical directory name mismatch: {directory.name} != {spec.canonical_xpid}"
        )
    if (directory / "finish").exists():
        raise CampaignError("canonical milestone must not contain a finish marker")
    _verify_sha256sums(directory)
    milestone = _read_json(directory / MILESTONE_NAME)
    if int(milestone.get("target_real_step", -1)) != TARGET_REAL_STEP:
        raise CampaignError("canonical milestone target_real_step mismatch")
    if milestone.get("finish_present") is not False:
        raise CampaignError("canonical milestone must record finish_present=false")
    actor_name = f"ckp_actor.tar_step_{TARGET_REAL_STEP}"
    model_name = f"ckp_model.tar_step_{TARGET_REAL_STEP}"
    if _sha256_file(directory / actor_name) != spec.actor_sha256:
        raise CampaignError("canonical actor does not have the known exact hash")
    if _sha256_file(directory / model_name) != spec.model_sha256:
        raise CampaignError("canonical model does not have the known exact hash")
    return {
        "xpid": spec.canonical_xpid,
        "target_real_step": TARGET_REAL_STEP,
        "actor_sha256": spec.actor_sha256,
        "model_sha256": spec.model_sha256,
        "finish_present": False,
    }


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser(
        "build-campaign", help="audit data and atomically publish 72/70 manifests"
    )
    build.add_argument("--output-dir", type=Path, default=DEFAULT_CAMPAIGN_DIR)
    build.add_argument("--behavior-root", type=Path, default=DEFAULT_BEHAVIOR_ROOT)
    build.add_argument(
        "--behavior-sha256-manifest",
        type=Path,
        default=DEFAULT_BEHAVIOR_SHA256_MANIFEST,
    )
    build.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    build.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)

    verify = subparsers.add_parser(
        "verify-campaign", help="rehash data and verify the published campaign"
    )
    verify.add_argument("--campaign-dir", type=Path, default=DEFAULT_CAMPAIGN_DIR)
    verify.add_argument("--behavior-root", type=Path, default=DEFAULT_BEHAVIOR_ROOT)
    verify.add_argument(
        "--behavior-sha256-manifest",
        type=Path,
        default=DEFAULT_BEHAVIOR_SHA256_MANIFEST,
    )

    canonical = subparsers.add_parser(
        "canonicalize-existing",
        help="validate and atomically publish exact existing 5M milestones",
    )
    canonical.add_argument(
        "--run",
        choices=tuple(EXISTING_RUNS) + ("all",),
        default="all",
    )
    canonical.add_argument("--source-logs-root", type=Path, default=DEFAULT_LOGS_ROOT)
    canonical.add_argument("--output-root", type=Path, default=DEFAULT_LOGS_ROOT)
    canonical.add_argument("--behavior-root", type=Path, default=DEFAULT_BEHAVIOR_ROOT)
    canonical.add_argument(
        "--behavior-sha256-manifest",
        type=Path,
        default=DEFAULT_BEHAVIOR_SHA256_MANIFEST,
    )
    canonical.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    canonical.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)

    verify_canonical = subparsers.add_parser(
        "verify-canonical", help="verify sealed canonical 5M run directories"
    )
    verify_canonical.add_argument(
        "--run",
        choices=tuple(EXISTING_RUNS) + ("all",),
        default="all",
    )
    verify_canonical.add_argument("--logs-root", type=Path, default=DEFAULT_LOGS_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    if args.command == "build-campaign":
        destination = build_campaign_directory(
            args.output_dir,
            data_root=args.behavior_root,
            behavior_sha256_manifest=args.behavior_sha256_manifest,
            source_root=args.source_root,
            source_manifest=args.source_manifest,
        )
        result = verify_campaign_directory(
            destination,
            data_root=args.behavior_root,
            behavior_sha256_manifest=args.behavior_sha256_manifest,
        )
        print(json.dumps({"campaign_dir": str(destination), **result}, sort_keys=True))
        return 0
    if args.command == "verify-campaign":
        result = verify_campaign_directory(
            args.campaign_dir,
            data_root=args.behavior_root,
            behavior_sha256_manifest=args.behavior_sha256_manifest,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "canonicalize-existing":
        names = tuple(EXISTING_RUNS) if args.run == "all" else (args.run,)
        published = []
        for name in names:
            spec = EXISTING_RUNS[name]
            path = canonicalize_existing_run(
                spec,
                source_logs_root=args.source_logs_root,
                output_root=args.output_root,
                source_root=args.source_root,
                source_manifest=args.source_manifest,
                behavior_root=args.behavior_root,
                behavior_sha256_manifest=args.behavior_sha256_manifest,
            )
            published.append(verify_canonical_run(path, spec))
        print(json.dumps(published, sort_keys=True))
        return 0
    if args.command == "verify-canonical":
        names = tuple(EXISTING_RUNS) if args.run == "all" else (args.run,)
        results = [
            verify_canonical_run(args.logs_root / EXISTING_RUNS[name].canonical_xpid, EXISTING_RUNS[name])
            for name in names
        ]
        print(json.dumps(results, sort_keys=True))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CampaignError, FileExistsError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
