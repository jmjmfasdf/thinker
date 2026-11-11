"""Bounded-memory behavioral replay traces and legacy ``video_stat`` I/O.

The native format is one numeric Zarr v2 directory per behavioral episode.
Rows are the events produced by the dynamic planner; legacy reset rows are
*not* written by the runner.  :class:`NativeVideoStat` projects those events
to the historical ``video_stat.npy`` convention on demand.

The module deliberately imports Zarr lazily.  Reading trusted historical NPY
files and indexing behavioral source archives therefore remain usable in an
environment that has not installed the optional visualization dependencies.
Native stores never contain object arrays or pickle payloads.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
from types import MappingProxyType
from typing import Any, Optional, Sequence
import uuid
import zipfile

import numpy as np


LEGACY_KEYS = (
    "tree_reps",
    "thinker_action",
    "human_action",
    "status",
    "actor_policy",
    "step_times",
    "env_return",
    "cur_rewards",
    "im_vectors",
    "im_vp_vectors",
)

TRACE_FORMAT = "thinker.behavior-trace"
TRACE_VERSION = 2
MANIFEST_NAME = "manifest.json"
ALIGNMENT_PROTOCOL = "obs[t]-to-action[t]/burnin=1/v1"
STORAGE_LAYOUT = "packed-fields-v1"

PHASE_SEARCH = 0
CONTROL_PROCEED = 0
CONTROL_RESET = 1
CONTROL_STOP = 2


class BehaviorTraceError(RuntimeError):
    """Base error for native behavior-trace I/O."""


class TraceValidationError(BehaviorTraceError):
    """A trace is incomplete, corrupted, or violates its declared schema."""


class LegacyProjectionError(BehaviorTraceError):
    """A native event stream cannot be projected without inventing data."""


def _require_zarr():
    try:
        import zarr  # type: ignore
        import numcodecs  # type: ignore
        from numcodecs import Blosc  # type: ignore
    except ImportError as error:  # pragma: no cover - exercised without extras
        raise BehaviorTraceError(
            "Native behavior traces require zarr==2.18.3 and numcodecs==0.13.1. "
            "Install thinker/requirements-visual-behav.txt."
        ) from error
    if str(zarr.__version__) != "2.18.3" or str(numcodecs.__version__) != "0.13.1":
        raise BehaviorTraceError(
            "behavior_trace requires zarr==2.18.3 and numcodecs==0.13.1; "
            f"got zarr=={zarr.__version__}, numcodecs=={numcodecs.__version__}"
        )
    return zarr, Blosc


def file_sha256(path: str | os.PathLike[str], *, block_size: int = 8 << 20) -> str:
    """Return a streaming SHA-256 digest without materialising ``path``."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def behavior_source_identity(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Hash one immutable source for reuse by coordinator and GPU workers."""

    source = Path(path).expanduser().resolve()
    before = source.stat()
    digest = file_sha256(source)
    after = source.stat()
    keys_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    keys_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if keys_before != keys_after:
        raise TraceValidationError(f"behavior archive changed while hashing: {source}")
    return {
        "path": str(source),
        "size": int(after.st_size),
        "sha256": digest,
        "mtime_ns": int(after.st_mtime_ns),
        "inode": int(after.st_ino),
        "device": int(after.st_dev),
    }


def _json_safe(value: Any) -> Any:
    """Convert metadata to deterministic, finite JSON-compatible values."""

    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, slice):
        return {
            "start": value.start,
            "stop": value.stop,
            "step": value.step,
        }
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("metadata floats must be finite")
        return value
    raise TypeError(f"metadata value {type(value).__name__} is not JSON serialisable")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_regular_file(path: Path) -> None:
    """Flush one payload file without following a substituted leaf symlink."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise TraceValidationError(f"native trace payload is not regular: {path}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_payload_tree(root: Path) -> None:
    """Durably flush payload files, then every containing directory bottom-up."""

    entries = sorted(root.rglob("*"), key=lambda item: item.as_posix())
    files: list[Path] = []
    directories = [root]
    for path in entries:
        if path.is_symlink():
            raise TraceValidationError(f"native trace may not contain symlink {path}")
        if path.is_file():
            if path == root / MANIFEST_NAME:
                raise TraceValidationError(
                    f"partial trace unexpectedly already contains {MANIFEST_NAME}"
                )
            files.append(path)
        elif path.is_dir():
            directories.append(path)
        else:
            raise TraceValidationError(
                f"native trace contains unsupported payload entry {path}"
            )

    for path in files:
        _fsync_regular_file(path)
    for path in sorted(
        directories,
        key=lambda item: (-len(item.relative_to(root).parts), item.as_posix()),
    ):
        _fsync_directory(path)


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        with temporary.open("xb") as output:
            output.write(_canonical_json(value))
            output.write(b"\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _directory_file_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise TraceValidationError(f"native trace may not contain symlink {path}")
        if not path.is_file() or path == root / MANIFEST_NAME:
            continue
        relative = path.relative_to(root).as_posix()
        records.append(
            {
                "path": relative,
                "size": int(path.stat().st_size),
                "sha256": file_sha256(path),
            }
        )
    return records


def _records_digest(records: Sequence[Mapping[str, Any]]) -> str:
    compact = [
        {
            "path": str(record["path"]),
            "size": int(record["size"]),
            "sha256": str(record["sha256"]),
        }
        for record in records
    ]
    return hashlib.sha256(_canonical_json(compact)).hexdigest()


def read_episode_manifest(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read a final store manifest without opening any arrays."""

    manifest_path = Path(path) / MANIFEST_NAME
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TraceValidationError(f"cannot read {manifest_path}: {error}") from error
    if not isinstance(value, dict):
        raise TraceValidationError("episode manifest must be a JSON object")
    return value


def _to_numpy_numeric(value: Any, *, name: str) -> np.ndarray:
    """Detach Torch-like values and reject pickle/object/string payloads."""

    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach()
    if hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu()
    if hasattr(value, "numpy") and callable(value.numpy):
        value = value.numpy()
    array = np.asarray(value)
    if array.dtype.kind not in "biuf":
        raise TypeError(
            f"event field {name!r} must be bool/integer/float, got {array.dtype}"
        )
    if array.dtype.hasobject:
        raise TypeError(f"event field {name!r} may not use object dtype")
    # Use explicit little-endian values for stores that move between machines.
    if array.dtype.byteorder == ">" or (
        array.dtype.byteorder == "=" and not np.little_endian
    ):
        array = array.byteswap().view(array.dtype.newbyteorder("<"))
    # np.ascontiguousarray promotes a scalar to shape (1,), which would make
    # scalar event contracts dependent on this conversion helper.  np.array
    # preserves rank zero while still owning C-contiguous CPU storage.
    return np.array(array, copy=True, order="C")


def _flatten_event(
    mapping: Mapping[str, Any], prefix: str = ""
) -> dict[str, np.ndarray]:
    if not isinstance(mapping, Mapping):
        raise TypeError("write_event expects a mapping")
    flattened: dict[str, np.ndarray] = {}
    for key in sorted(mapping):
        if not isinstance(key, str) or not key or "/" in key or key in {".", ".."}:
            raise ValueError(f"invalid event field name {key!r}")
        name = f"{prefix}/{key}" if prefix else key
        value = mapping[key]
        if isinstance(value, Mapping):
            nested = _flatten_event(value, name)
            if not nested:
                raise ValueError(f"nested event mapping {name!r} may not be empty")
            flattened.update(nested)
        else:
            flattened[name] = _to_numpy_numeric(value, name=name)
    if not prefix and not flattened:
        raise ValueError("an event may not be empty")
    return flattened


@dataclass(frozen=True)
class _FieldSpec:
    name: str
    dtype: str
    shape: tuple[int, ...]
    storage_group: str
    storage_index: int

    def as_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "storage_group": self.storage_group,
            "storage_index": self.storage_index,
        }


@dataclass(frozen=True)
class _PackSpec:
    name: str
    dtype: str
    shape: tuple[int, ...]
    fields: tuple[str, ...]

    def as_json(self, chunks: Sequence[int]) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "field_count": len(self.fields),
            "fields": list(self.fields),
            "chunks": [int(value) for value in chunks],
        }


class EpisodeZarrWriter:
    """Append fixed-schema native events to an atomic numeric Zarr v2 store.

    The first event freezes the flattened key, dtype, and per-event shape
    schema.  All later events must match it exactly.  Callers represent an
    unavailable value with a numeric sentinel plus a companion ``*_valid``
    boolean; ``None`` is intentionally not accepted.
    """

    _IMAGE_MODES = {"none", "real", "imagined", "all"}

    def __init__(
        self,
        partial_path: str | os.PathLike[str],
        *,
        metadata: Mapping[str, Any],
        num_actions: int,
        tree_rep_schema: Mapping[str, Any],
        chunk_events: int = 256,
        images: str = "none",
        compression_level: int = 3,
    ) -> None:
        zarr, Blosc = _require_zarr()
        self.partial_path = Path(partial_path).expanduser().resolve()
        if self.partial_path.exists():
            raise FileExistsError(f"partial trace already exists: {self.partial_path}")
        if isinstance(num_actions, (bool, np.bool_)) or int(num_actions) < 1:
            raise ValueError("num_actions must be a positive integer")
        if isinstance(chunk_events, (bool, np.bool_)) or int(chunk_events) < 1:
            raise ValueError("chunk_events must be a positive integer")
        if isinstance(compression_level, (bool, np.bool_)) or not 0 <= int(compression_level) <= 9:
            raise ValueError("compression_level must be an integer in 0..9")
        images = str(images).lower()
        if images not in self._IMAGE_MODES:
            raise ValueError(f"images must be one of {sorted(self._IMAGE_MODES)}")

        self.metadata = _json_safe(metadata)
        self.num_actions = int(num_actions)
        self.tree_rep_schema = _json_safe(tree_rep_schema)
        self.chunk_events = int(chunk_events)
        self.images = images
        self._zarr = zarr
        self._Blosc = Blosc
        self._store = zarr.DirectoryStore(str(self.partial_path))
        self._root = zarr.group(store=self._store, overwrite=False)
        self._packs = self._root.require_group("packs")
        self._specs: dict[str, _FieldSpec] = {}
        self._pack_specs: dict[str, _PackSpec] = {}
        self._buffer: list[dict[str, np.ndarray]] = []
        self._event_count = 0
        self._closed = False
        self._finalized = False
        self._compressor = Blosc(
            cname="zstd", clevel=int(compression_level), shuffle=Blosc.BITSHUFFLE
        )
        self._root.attrs.update(
            {
                "format": TRACE_FORMAT,
                "format_version": TRACE_VERSION,
                "storage_layout": STORAGE_LAYOUT,
                "state": "partial",
                "num_actions": self.num_actions,
                "chunk_events": self.chunk_events,
                "images": self.images,
                "metadata": self.metadata,
                "tree_rep_schema": self.tree_rep_schema,
                "event_count": 0,
            }
        )

    @property
    def event_count(self) -> int:
        return self._event_count + len(self._buffer)

    @property
    def field_schema(self) -> Mapping[str, _FieldSpec]:
        return MappingProxyType(self._specs)

    def _check_open(self) -> None:
        if self._closed:
            raise BehaviorTraceError("episode writer is already closed")

    def _freeze_schema(self, event: Mapping[str, np.ndarray]) -> None:
        grouped: dict[tuple[str, tuple[int, ...]], list[str]] = {}
        for name, value in sorted(event.items()):
            key = (value.dtype.str, tuple(value.shape))
            grouped.setdefault(key, []).append(name)

        specs: dict[str, _FieldSpec] = {}
        packs: dict[str, _PackSpec] = {}
        for pack_index, key in enumerate(sorted(grouped)):
            dtype, shape = key
            pack_name = f"pack_{pack_index:04d}"
            fields = tuple(sorted(grouped[key]))
            packs[pack_name] = _PackSpec(pack_name, dtype, shape, fields)
            for storage_index, name in enumerate(fields):
                specs[name] = _FieldSpec(
                    name=name,
                    dtype=dtype,
                    shape=shape,
                    storage_group=pack_name,
                    storage_index=storage_index,
                )
        self._specs = specs
        self._pack_specs = packs
        self._root.attrs["field_schema"] = [
            self._specs[name].as_json() for name in sorted(self._specs)
        ]

    @staticmethod
    def _image_kind(name: str) -> Optional[str]:
        pieces = name.lower().split("/")
        leaf = pieces[-1]
        if leaf.endswith("_valid"):
            return None
        imagined_names = {
            "imagined_image",
            "imagined_images",
            "imagined_frame",
            "imagined_frames",
            "im_image",
            "im_images",
            "im_frame",
            "im_frames",
            "im_imgs",
            "pre_xs",
            "post_xs",
        }
        real_names = {
            "real_image",
            "real_images",
            "real_frame",
            "real_frames",
            "real_imgs",
            "pre_real_states",
            "post_real_states",
        }
        if leaf in imagined_names or (pieces[0] == "images" and leaf == "imagined"):
            return "imagined"
        if leaf in real_names or (pieces[0] == "images" and leaf == "real"):
            return "real"
        if leaf in {"image", "images", "frame", "frames"}:
            raise ValueError(
                f"ambiguous image field {name!r}; name it real_image or imagined_image"
            )
        return None

    def _validate_images(self, event: Mapping[str, np.ndarray]) -> None:
        for name, value in event.items():
            kind = self._image_kind(name)
            if kind is None:
                continue
            if self.images == "none":
                raise ValueError(
                    f"event contains image field {name!r}, but images='none'"
                )
            if self.images != "all" and kind != self.images:
                raise ValueError(
                    f"{kind} image field {name!r} is disabled by images={self.images!r}"
                )
            if value.dtype != np.dtype(np.uint8):
                raise TypeError(
                    f"stored image field {name!r} must be uint8, got {value.dtype}"
                )

    def write_event(self, mapping: Mapping[str, Any]) -> int:
        """Validate and buffer one event, returning its zero-based index."""

        self._check_open()
        event = _flatten_event(mapping)
        self._validate_images(event)
        if not self._specs:
            empty = [name for name, value in event.items() if 0 in value.shape]
            if empty:
                raise ValueError(f"event fields may not have empty dimensions: {empty}")
            self._freeze_schema(event)
        else:
            expected = set(self._specs)
            actual = set(event)
            if actual != expected:
                missing = sorted(expected - actual)
                extra = sorted(actual - expected)
                raise ValueError(
                    f"event schema differs from first row; missing={missing}, extra={extra}"
                )
            for name, value in event.items():
                spec = self._specs[name]
                if value.dtype.str != spec.dtype or tuple(value.shape) != spec.shape:
                    raise TypeError(
                        f"event field {name!r} differs from first row: "
                        f"expected dtype={spec.dtype}, shape={spec.shape}; "
                        f"got dtype={value.dtype.str}, shape={tuple(value.shape)}"
                    )

        index = self.event_count
        self._buffer.append(event)
        if len(self._buffer) >= self.chunk_events:
            self._flush()
        return index

    def _array_chunks(self, spec: _PackSpec) -> tuple[int, ...]:
        item_bytes = (
            len(spec.fields)
            * max(1, int(np.prod(spec.shape, dtype=np.int64)))
            * max(1, np.dtype(spec.dtype).itemsize)
        )
        target = max(1, (16 << 20) // item_bytes)
        first = min(self.chunk_events, target)
        # Every non-event dimension is stored in one chunk.  Consequently one
        # physical chunk contains the same event interval for every field in
        # this dtype/shape pack, rather than one filesystem object per field.
        return (int(first), len(spec.fields)) + spec.shape

    def _create_arrays(self) -> None:
        for name in sorted(self._pack_specs):
            spec = self._pack_specs[name]
            if name in self._packs:
                continue
            self._packs.create_dataset(
                name,
                shape=(0, len(spec.fields)) + spec.shape,
                chunks=self._array_chunks(spec),
                dtype=np.dtype(spec.dtype),
                compressor=self._compressor,
                overwrite=False,
            )

    def _flush(self) -> None:
        if not self._buffer:
            return
        self._create_arrays()
        start = self._event_count
        stop = start + len(self._buffer)
        for name in sorted(self._pack_specs):
            spec = self._pack_specs[name]
            output = self._packs[name]
            output.resize((stop, len(spec.fields)) + spec.shape)
            batch = np.stack(
                [
                    np.stack([event[field] for field in spec.fields], axis=0)
                    for event in self._buffer
                ],
                axis=0,
            )
            output[start:stop] = batch
        self._event_count = stop
        self._buffer.clear()
        self._root.attrs["event_count"] = self._event_count

    def _close_store(self) -> None:
        close = getattr(self._store, "close", None)
        if callable(close):
            close()

    def finalize(self, final_path: str | os.PathLike[str]) -> dict[str, Any]:
        """Verify, seal, and atomically rename the partial episode directory."""

        self._check_open()
        final = Path(final_path).expanduser().resolve()
        if final == self.partial_path:
            raise ValueError("final_path must differ from partial_path")
        if final.exists():
            raise FileExistsError(f"final trace already exists: {final}")
        final.parent.mkdir(parents=True, exist_ok=True)
        if final.parent.stat().st_dev != self.partial_path.parent.stat().st_dev:
            raise OSError("partial and final trace directories must share a filesystem")
        if not self._specs:
            raise BehaviorTraceError("cannot finalize an episode with no events")

        self._flush()
        self._root.attrs.update(
            {
                "state": "complete",
                "event_count": self._event_count,
            }
        )
        pack_schema = [
            self._pack_specs[name].as_json(self._packs[name].chunks)
            for name in sorted(self._pack_specs)
        ]
        self._root.attrs["pack_schema"] = pack_schema
        self._zarr.consolidate_metadata(self._store)
        self._close_store()

        # Zarr/codec close does not imply durable storage.  Flush every payload
        # file and then the directory topology (children before parents) before
        # writing the manifest that declares those bytes complete.
        _fsync_payload_tree(self.partial_path)
        records = _directory_file_records(self.partial_path)
        manifest = {
            "format": TRACE_FORMAT,
            "format_version": TRACE_VERSION,
            "storage_layout": STORAGE_LAYOUT,
            "state": "complete",
            "event_count": self._event_count,
            "num_actions": self.num_actions,
            "images": self.images,
            "metadata": self.metadata,
            "tree_rep_schema": self.tree_rep_schema,
            "field_schema": [
                self._specs[name].as_json() for name in sorted(self._specs)
            ],
            "pack_schema": pack_schema,
            "files": records,
            "files_sha256": _records_digest(records),
        }
        _atomic_write_json(self.partial_path / MANIFEST_NAME, manifest)
        # A partial directory is never externally canonical.  Fully validate
        # it before the single rename so a validation error cannot expose a
        # bad final path, even briefly.
        sealed_manifest = validate_episode_store(self.partial_path)
        partial_parent = self.partial_path.parent
        os.replace(self.partial_path, final)
        _fsync_directory(final.parent)
        if partial_parent != final.parent:
            _fsync_directory(partial_parent)
        self._closed = True
        self._finalized = True
        return sealed_manifest

    def abort(self) -> None:
        """Close and preserve a non-canonical partial store for diagnosis."""

        if self._closed:
            return
        try:
            self._flush()
            self._root.attrs.update(
                {"state": "aborted", "event_count": self._event_count}
            )
        finally:
            self._close_store()
            self._closed = True

    def __enter__(self) -> "EpisodeZarrWriter":
        self._check_open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if not self._finalized:
            self.abort()
        return False


def _zarr_array_names(group: Any, prefix: str = "") -> list[str]:
    names: list[str] = []
    for key in sorted(group.array_keys()):
        names.append(f"{prefix}/{key}" if prefix else key)
    for key in sorted(group.group_keys()):
        child_prefix = f"{prefix}/{key}" if prefix else key
        names.extend(_zarr_array_names(group[key], child_prefix))
    return sorted(names)


def _validated_packed_schema(
    manifest: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Validate and index the manifest's logical-to-packed field mapping."""

    raw_fields = manifest.get("field_schema")
    raw_packs = manifest.get("pack_schema")
    if not isinstance(raw_fields, list) or not raw_fields:
        raise TraceValidationError("manifest field_schema must be a nonempty list")
    if not isinstance(raw_packs, list) or not raw_packs:
        raise TraceValidationError("manifest pack_schema must be a nonempty list")

    fields: dict[str, dict[str, Any]] = {}
    grouped_fields: dict[str, dict[int, str]] = {}
    for raw in raw_fields:
        if not isinstance(raw, Mapping):
            raise TraceValidationError("manifest field_schema entry is not an object")
        name = raw.get("name")
        dtype_value = raw.get("dtype")
        shape_value = raw.get("shape")
        storage_group = raw.get("storage_group")
        storage_index = raw.get("storage_index")
        if not isinstance(name, str) or not name or name in fields:
            raise TraceValidationError("manifest field names must be unique nonempty strings")
        if not isinstance(dtype_value, str):
            raise TraceValidationError(f"field {name!r} has invalid dtype metadata")
        try:
            dtype = np.dtype(dtype_value)
        except TypeError as error:
            raise TraceValidationError(
                f"field {name!r} has invalid dtype metadata"
            ) from error
        if dtype.kind not in "biuf" or dtype.hasobject or dtype.str != dtype_value:
            raise TraceValidationError(f"field {name!r} has noncanonical numeric dtype")
        if not isinstance(shape_value, list) or any(
            type(value) is not int or value < 1 for value in shape_value
        ):
            # An empty list is the canonical scalar shape.  Only dimensions
            # that are present must be strictly positive.
            if shape_value != []:
                raise TraceValidationError(f"field {name!r} has invalid shape metadata")
        shape = tuple(int(value) for value in shape_value)
        if not isinstance(storage_group, str) or not storage_group:
            raise TraceValidationError(f"field {name!r} lacks a storage group")
        if type(storage_index) is not int or storage_index < 0:
            raise TraceValidationError(f"field {name!r} has invalid storage index")
        index_map = grouped_fields.setdefault(storage_group, {})
        if storage_index in index_map:
            raise TraceValidationError(
                f"storage group {storage_group!r} repeats column {storage_index}"
            )
        index_map[storage_index] = name
        fields[name] = {
            "name": name,
            "dtype": dtype.str,
            "shape": shape,
            "storage_group": storage_group,
            "storage_index": storage_index,
        }

    packs: dict[str, dict[str, Any]] = {}
    for raw in raw_packs:
        if not isinstance(raw, Mapping):
            raise TraceValidationError("manifest pack_schema entry is not an object")
        name = raw.get("name")
        if not isinstance(name, str) or not name or name in packs:
            raise TraceValidationError("manifest pack names must be unique nonempty strings")
        dtype_value = raw.get("dtype")
        shape_value = raw.get("shape")
        fields_value = raw.get("fields")
        field_count = raw.get("field_count")
        chunks_value = raw.get("chunks")
        if not isinstance(dtype_value, str):
            raise TraceValidationError(f"pack {name!r} has invalid dtype metadata")
        try:
            dtype = np.dtype(dtype_value)
        except TypeError as error:
            raise TraceValidationError(
                f"pack {name!r} has invalid dtype metadata"
            ) from error
        if dtype.kind not in "biuf" or dtype.hasobject or dtype.str != dtype_value:
            raise TraceValidationError(f"pack {name!r} has noncanonical numeric dtype")
        if not isinstance(shape_value, list) or any(
            type(value) is not int or value < 1 for value in shape_value
        ):
            if shape_value != []:
                raise TraceValidationError(f"pack {name!r} has invalid shape metadata")
        shape = tuple(int(value) for value in shape_value)
        if (
            not isinstance(fields_value, list)
            or not fields_value
            or any(not isinstance(value, str) for value in fields_value)
            or type(field_count) is not int
            or field_count != len(fields_value)
        ):
            raise TraceValidationError(f"pack {name!r} has invalid field membership")
        if not isinstance(chunks_value, list) or any(
            type(value) is not int or value < 1 for value in chunks_value
        ):
            raise TraceValidationError(f"pack {name!r} has invalid chunk metadata")
        chunks = tuple(int(value) for value in chunks_value)
        if len(chunks) != len(shape) + 2:
            raise TraceValidationError(f"pack {name!r} chunk rank disagrees")
        if chunks[1:] != (field_count,) + shape:
            raise TraceValidationError(
                f"pack {name!r} must store every non-event dimension in one chunk"
            )
        expected_index_map = grouped_fields.get(name)
        expected_indices = set(range(field_count))
        if expected_index_map is None or set(expected_index_map) != expected_indices:
            raise TraceValidationError(f"pack {name!r} field columns are not dense")
        expected_fields = [expected_index_map[index] for index in range(field_count)]
        if fields_value != expected_fields:
            raise TraceValidationError(f"pack {name!r} field order disagrees")
        for field_name in expected_fields:
            field = fields[field_name]
            if field["dtype"] != dtype.str or field["shape"] != shape:
                raise TraceValidationError(
                    f"field {field_name!r} disagrees with pack {name!r}"
                )
        packs[name] = {
            "name": name,
            "dtype": dtype.str,
            "shape": shape,
            "field_count": field_count,
            "fields": tuple(expected_fields),
            "chunks": chunks,
        }

    if set(grouped_fields) != set(packs):
        missing = sorted(set(grouped_fields) - set(packs))
        extra = sorted(set(packs) - set(grouped_fields))
        raise TraceValidationError(
            f"manifest pack membership disagrees: missing={missing}, extra={extra}"
        )
    return fields, packs


def validate_episode_store(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Hash and structurally validate a finalized native episode store."""

    zarr, _ = _require_zarr()
    root_path = Path(path).expanduser().resolve()
    if not root_path.is_dir():
        raise TraceValidationError(f"episode store is not a directory: {root_path}")
    manifest = read_episode_manifest(root_path)
    if manifest.get("format") != TRACE_FORMAT or manifest.get("format_version") != TRACE_VERSION:
        raise TraceValidationError("unsupported episode trace format/version")
    if manifest.get("storage_layout") != STORAGE_LAYOUT:
        raise TraceValidationError("unsupported episode trace storage layout")
    if manifest.get("state") != "complete":
        raise TraceValidationError("episode manifest is not complete")
    declared = manifest.get("files")
    if not isinstance(declared, list):
        raise TraceValidationError("manifest files must be a list")
    actual = _directory_file_records(root_path)
    if actual != declared:
        raise TraceValidationError("episode store file set, size, or SHA-256 changed")
    if _records_digest(actual) != manifest.get("files_sha256"):
        raise TraceValidationError("episode aggregate SHA-256 changed")

    try:
        root = zarr.open_consolidated(str(root_path), mode="r")
    except Exception:
        root = zarr.open_group(str(root_path), mode="r")
    if root.attrs.get("format") != TRACE_FORMAT or root.attrs.get("state") != "complete":
        raise TraceValidationError("Zarr root is not a complete behavior trace")
    if (
        root.attrs.get("format_version") != TRACE_VERSION
        or root.attrs.get("storage_layout") != STORAGE_LAYOUT
    ):
        raise TraceValidationError("Zarr root storage version/layout disagree")
    event_count = int(manifest.get("event_count", -1))
    if event_count < 1 or int(root.attrs.get("event_count", -1)) != event_count:
        raise TraceValidationError("manifest and Zarr event counts disagree")
    if set(root.group_keys()) != {"packs"} or tuple(root.array_keys()):
        raise TraceValidationError("native trace must contain only its packs group")
    fields, packs = _validated_packed_schema(manifest)
    pack_names = _zarr_array_names(root["packs"])
    if pack_names != sorted(packs):
        raise TraceValidationError("manifest and Zarr pack sets disagree")
    for name, spec in packs.items():
        array = root["packs"][name]
        if array.dtype.kind not in "biuf" or array.dtype.hasobject:
            raise TraceValidationError(f"event pack {name!r} is not numeric")
        expected_shape = (
            event_count,
            int(spec["field_count"]),
            *tuple(spec["shape"]),
        )
        if tuple(array.shape) != expected_shape:
            raise TraceValidationError(f"event pack {name!r} shape disagrees")
        if array.dtype.str != spec["dtype"]:
            raise TraceValidationError(f"event pack {name!r} dtype disagrees")
        if tuple(array.chunks) != tuple(spec["chunks"]):
            raise TraceValidationError(f"event pack {name!r} chunks disagree")
    if int(root.attrs.get("num_actions", -1)) != int(manifest.get("num_actions", -2)):
        raise TraceValidationError("num_actions metadata disagree")
    for name in (
        "images",
        "metadata",
        "tree_rep_schema",
        "field_schema",
        "pack_schema",
    ):
        if _json_safe(root.attrs.get(name)) != _json_safe(manifest.get(name)):
            raise TraceValidationError(f"manifest and Zarr {name} metadata disagree")
    return manifest


def _normalise_row_slice(value: Any, length: int) -> np.ndarray:
    if value is None:
        return np.arange(length, dtype=np.int64)
    if isinstance(value, tuple) and 1 <= len(value) <= 3:
        value = slice(*value)
    if isinstance(value, slice):
        start, stop, step = value.indices(length)
        return np.arange(start, stop, step, dtype=np.int64)
    raise TypeError("episode_slice must be None, a slice, or a 1-3 integer tuple")


class _LegacyMapping(Mapping[str, Any]):
    def __iter__(self) -> Iterator[str]:
        return iter(LEGACY_KEYS)

    def __len__(self) -> int:
        return len(LEGACY_KEYS)

    def legacy_view(self, *, materialize: bool = False):
        if materialize:
            return {key: self[key] for key in LEGACY_KEYS}
        return self


class LegacyVideoStat(_LegacyMapping):
    """Exact-key wrapper around a trusted historical object-NPY payload."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path).expanduser().resolve()
        try:
            loaded = np.load(self.path, allow_pickle=True)
            if not isinstance(loaded, np.ndarray) or loaded.shape != ():
                raise TraceValidationError("legacy NPY must contain one scalar mapping")
            payload = loaded.item()
        except Exception as error:
            if isinstance(error, TraceValidationError):
                raise
            raise TraceValidationError(f"cannot load legacy video stat: {error}") from error
        if not isinstance(payload, Mapping):
            raise TraceValidationError("legacy NPY scalar is not a mapping")
        missing = [key for key in LEGACY_KEYS if key not in payload]
        if missing:
            raise TraceValidationError(f"legacy video stat is missing keys {missing}")
        self._payload = {key: payload[key] for key in LEGACY_KEYS}

    @property
    def format(self) -> str:
        return "legacy-object-npy"

    def __getitem__(self, key: str) -> Any:
        if key not in LEGACY_KEYS:
            raise KeyError(key)
        return self._payload[key]


@dataclass(frozen=True)
class _ProjectedRow:
    source: int
    status: int
    synthetic: bool


def _as_event_vector(store: "NativeVideoStat", name: str) -> np.ndarray:
    value = store.field(name)
    value = np.asarray(value)
    if value.shape[0] != store.event_count:
        raise LegacyProjectionError(f"native field {name!r} has wrong event axis")
    while value.ndim > 1 and value.shape[-1] == 1:
        value = value[..., 0]
    if value.ndim != 1:
        raise LegacyProjectionError(f"native field {name!r} must be scalar per event")
    return value


class LegacyProjection(_LegacyMapping):
    """Lazy exact-key projection of native dynamic events."""

    def __init__(self, native: "NativeVideoStat") -> None:
        self.native = native
        self._rows = self._build_rows()
        self._source_event_indices = np.asarray(
            [row.source for row in self._rows], dtype=np.int64
        )
        self._synthetic_rows = np.asarray(
            [row.synthetic for row in self._rows], dtype=np.bool_
        )
        self._cache: dict[str, Any] = {}
        self._validity_cache: dict[str, np.ndarray] = {}

    def _optional_bool(self, name: str, default: bool = False) -> np.ndarray:
        if not self.native.has_field(name):
            return np.full(self.native.event_count, default, dtype=np.bool_)
        return _as_event_vector(self.native, name).astype(np.bool_, copy=False)

    def _build_rows(self) -> tuple[_ProjectedRow, ...]:
        initial = self._optional_bool("initial_root")
        search = self._optional_bool("search_computation")
        real = self._optional_bool("real_transition")
        forced_root_reset = self._optional_bool("forced_root_reset")
        actor_reset = self._optional_bool("actor_reset")

        if not self.native.has_field("search_computation"):
            required = {
                "phase_pre",
                "accepted_control",
                "accepted_primary_action",
            }
            if all(self.native.has_field(name) for name in required):
                phase = _as_event_vector(self.native, "phase_pre")
                control = _as_event_vector(self.native, "accepted_control")
                action = _as_event_vector(self.native, "accepted_primary_action")
                search = (phase == PHASE_SEARCH) & np.isin(
                    control, (CONTROL_PROCEED, CONTROL_RESET)
                ) & (action >= 0)
        overlap = initial.astype(np.int8) + search.astype(np.int8) + real.astype(np.int8)
        if np.any(overlap > 1):
            raise LegacyProjectionError(
                "initial_root, search_computation, and real_transition overlap"
            )
        if int(np.count_nonzero(initial)) != 1:
            raise LegacyProjectionError("native episode must contain exactly one initial_root")
        if not initial[0]:
            raise LegacyProjectionError("initial_root must be the first native event")

        control = (
            _as_event_vector(self.native, "accepted_control")
            if self.native.has_field("accepted_control")
            else np.full(self.native.event_count, -1, dtype=np.int64)
        )
        action = (
            _as_event_vector(self.native, "accepted_primary_action")
            if self.native.has_field("accepted_primary_action")
            else np.full(self.native.event_count, -1, dtype=np.int64)
        )
        accepted_valid = self.native.explicit_valid("accepted_control")
        action_valid = self.native.explicit_valid("accepted_primary_action")
        bad_search = search & (
            ~accepted_valid
            | ~action_valid
            | ~np.isin(control, (CONTROL_PROCEED, CONTROL_RESET))
            | (action < 0)
        )
        if np.any(bad_search):
            raise LegacyProjectionError(
                "search_computation must carry a valid PROCEED/RESET action"
            )
        if np.any(search & (actor_reset != (control == CONTROL_RESET))):
            raise LegacyProjectionError(
                "actor_reset must match accepted RESET on search computations"
            )

        rows: list[_ProjectedRow] = []
        for index in range(self.native.event_count):
            if initial[index]:
                rows.append(_ProjectedRow(index, 0, False))
            elif search[index]:
                rows.append(_ProjectedRow(index, 2, False))
                if actor_reset[index]:
                    rows.append(_ProjectedRow(index, 1, True))
            elif real[index]:
                rows.append(_ProjectedRow(index, 0, False))
                # Ordinary post-transition search_state_reset is merely the
                # next stage's initialization.  Status 3 is reserved for an
                # explicitly observed cenv-forced root reset and is never
                # inferred from dynamic forced_stop (STOP is not RESET).
                if forced_root_reset[index]:
                    rows.append(_ProjectedRow(index, 3, True))
        if not rows:
            raise LegacyProjectionError("native episode has no projectable rows")
        return tuple(rows)

    @property
    def row_count(self) -> int:
        return len(self._rows)

    @property
    def source_event_index(self) -> np.ndarray:
        return self._source_event_indices.copy()

    @property
    def synthetic_mask(self) -> np.ndarray:
        return self._synthetic_rows.copy()

    @staticmethod
    def _row_sources(rows: Sequence[_ProjectedRow]) -> np.ndarray:
        return np.asarray([row.source for row in rows], dtype=np.int64)

    def _field_validity(self, name: str) -> np.ndarray:
        if name not in self._validity_cache:
            self._validity_cache[name] = self.native.explicit_valid(name)
        return self._validity_cache[name]

    def _project_plain(
        self,
        name: str,
        rows: Sequence[_ProjectedRow],
        *,
        missing_none: bool = True,
    ) -> Any:
        if not self.native.has_field(name):
            if missing_none:
                return None
            raise LegacyProjectionError(f"native trace lacks {name!r}")
        validity = self._field_validity(name)
        # Historical projection decides whether an optional field exists over
        # the complete projectable episode.  Keep that decision stable for an
        # empty or all-invalid requested slice without reading the large value.
        if missing_none and not np.any(validity[self._source_event_indices]):
            return None
        sources = self._row_sources(rows)
        selected = self.native.read_field_rows(name, sources)
        valid = validity[sources]
        if not np.all(valid):
            selected = selected.copy()
            if selected.dtype.kind == "f":
                selected[~valid] = np.nan
            else:
                selected[~valid] = 0
        return selected

    def _project_actions(
        self, name: str, rows: Sequence[_ProjectedRow]
    ) -> np.ndarray:
        output = np.zeros((len(rows), self.native.num_actions), dtype=np.int64)
        if not self.native.has_field(name):
            raise LegacyProjectionError(f"native trace lacks real-action field {name!r}")
        valid = self._field_validity(name)
        needed = [
            (output_index, row)
            for output_index, row in enumerate(rows)
            if not row.synthetic and row.status == 0 and row.source != 0
        ]
        for _, row in needed:
            if not valid[row.source]:
                raise LegacyProjectionError(
                    f"real transition {row.source} lacks valid {name}"
                )
        values = self.native.read_field_rows(
            name,
            np.asarray([row.source for _, row in needed], dtype=np.int64),
        )
        for (output_index, row), item in zip(needed, values):
            if row.synthetic or row.status != 0 or row.source == 0:
                continue
            item = np.asarray(item)
            if item.size == 1:
                action = int(item.reshape(-1)[0])
            elif item.size == self.native.num_actions:
                flat = item.reshape(-1)
                if not np.all(np.logical_or(flat == 0, flat == 1)) or np.sum(flat) != 1:
                    raise LegacyProjectionError(f"{name} is not scalar or strict one-hot")
                action = int(np.argmax(flat))
            else:
                raise LegacyProjectionError(f"{name} has incompatible action shape")
            if not 0 <= action < self.native.num_actions:
                raise LegacyProjectionError(f"{name} action lies outside the action space")
            output[output_index, action] = 1
        return output

    def _project_actor_policy(self, rows: Sequence[_ProjectedRow]) -> Any:
        value = self._project_plain("actor_policy", rows)
        if value is None:
            return None
        if value.shape[-1] != self.native.num_actions:
            raise LegacyProjectionError("actor_policy width disagrees with num_actions")
        prefix = value.shape[1:-1]
        if all(size == 1 for size in prefix) and len(prefix) == 3:
            return value.astype(np.float32, copy=False)
        middle = int(np.prod(prefix, dtype=np.int64)) if prefix else 1
        squeezed = value.reshape((len(rows), middle, self.native.num_actions))
        if squeezed.shape[1] != 1:
            raise LegacyProjectionError("actor_policy has non-singleton legacy axes")
        return squeezed[:, 0].reshape(
            len(rows), 1, 1, 1, self.native.num_actions
        ).astype(np.float32, copy=False)

    def _project_tree(
        self, rows: Sequence[_ProjectedRow]
    ) -> dict[str, np.ndarray]:
        prefix = "tree_reps/"
        names = [
            name[len(prefix) :]
            for name in self.native.field_names
            if name.startswith(prefix)
        ]
        selected = self._row_sources(rows)
        if names:
            result: dict[str, np.ndarray] = {}
            for key in sorted(names):
                value = self.native.read_field_rows(prefix + key, selected)
                if value.ndim >= 3 and value.shape[1] == 1:
                    value = value[:, 0]
                result[key] = value
            return result

        if not self.native.has_field("tree_reps_raw"):
            raise LegacyProjectionError("native trace lacks tree_reps or tree_reps_raw")
        valid = self._field_validity("tree_reps_raw")
        if not np.all(valid[selected]):
            raise LegacyProjectionError("projected rows contain invalid tree_reps_raw")
        raw = self.native.read_field_rows("tree_reps_raw", selected)
        if raw.ndim == 3 and raw.shape[1] == 1:
            raw = raw[:, 0]
        if raw.ndim != 2:
            raise LegacyProjectionError("tree_reps_raw must have one feature axis")
        result = {}
        for key, descriptor in sorted(self.native.tree_rep_schema.items()):
            start, stop, step, shape = _decode_slice_descriptor(descriptor)
            part = raw[:, slice(start, stop, step)]
            if shape is not None:
                expected = int(np.prod(shape, dtype=np.int64))
                if part.shape[1] != expected:
                    raise LegacyProjectionError(
                        f"tree schema shape for {key!r} disagrees with its slice"
                    )
                part = part.reshape((len(rows), *shape))
            result[str(key)] = part
        if not result:
            raise LegacyProjectionError("tree_rep_schema is empty")
        return result

    def _project_key(self, key: str, rows: Sequence[_ProjectedRow]) -> Any:
        if key not in LEGACY_KEYS:
            raise KeyError(key)
        if key == "status":
            value: Any = np.asarray([row.status for row in rows], dtype=np.int64)
        elif key == "tree_reps":
            value = self._project_tree(rows)
        elif key in {"human_action", "thinker_action"}:
            value = self._project_actions(key, rows)
        elif key == "actor_policy":
            value = self._project_actor_policy(rows)
        else:
            value = self._project_plain(key, rows)
            if value is not None and key in {
                "step_times",
                "env_return",
                "cur_rewards",
                "im_vectors",
                "im_vp_vectors",
            }:
                value = value.astype(np.float32, copy=False)
            if key == "env_return" and value is not None:
                value = np.asarray(value).reshape(len(rows))
                # Synthetic reset rows historically carry no environment reward.
                value = value.copy()
                value[
                    np.asarray([row.synthetic for row in rows], dtype=np.bool_)
                ] = 0.0
        return value

    def materialize_slice(self, row_slice: Any = None) -> dict[str, Any]:
        """Materialize only selected projected rows from each native field."""

        indices = _normalise_row_slice(row_slice, self.row_count)
        rows = tuple(self._rows[int(index)] for index in indices)
        return {key: self._project_key(key, rows) for key in LEGACY_KEYS}

    def __getitem__(self, key: str) -> Any:
        if key not in LEGACY_KEYS:
            raise KeyError(key)
        if key in self._cache:
            return self._cache[key]
        value = self._project_key(key, self._rows)
        self._cache[key] = value
        return value


def _decode_slice_descriptor(value: Any) -> tuple[Optional[int], Optional[int], Optional[int], Optional[tuple[int, ...]]]:
    shape: Optional[tuple[int, ...]] = None
    if isinstance(value, Mapping):
        if "slice" in value:
            base = value["slice"]
            shape_value = value.get("shape")
        else:
            base = value
            shape_value = value.get("shape")
        if shape_value is not None:
            shape = tuple(int(item) for item in shape_value)
        if isinstance(base, Mapping):
            return base.get("start"), base.get("stop"), base.get("step"), shape
        value = base
    if isinstance(value, (tuple, list)):
        if not 1 <= len(value) <= 3:
            raise LegacyProjectionError("tree slice list must have 1-3 entries")
        parts = list(value) + [None] * (3 - len(value))
        return parts[0], parts[1], parts[2], shape
    if isinstance(value, int):
        return value, value + 1, None, shape
    raise LegacyProjectionError(f"invalid tree slice descriptor {value!r}")


class _PackedFieldView:
    """Array-like logical field backed by one column of a packed Zarr array."""

    def __init__(self, array: Any, storage_index: int, field_shape: Sequence[int]) -> None:
        self._array = array
        self._storage_index = int(storage_index)
        self._field_shape = tuple(int(value) for value in field_shape)

    @property
    def shape(self) -> tuple[int, ...]:
        return (int(self._array.shape[0]),) + self._field_shape

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self._array.dtype)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))

    @property
    def chunks(self) -> tuple[int, ...]:
        chunks = tuple(int(value) for value in self._array.chunks)
        return (chunks[0],) + chunks[2:]

    def __len__(self) -> int:
        return self.shape[0]

    def __array__(self, dtype: Any = None, copy: Optional[bool] = None) -> np.ndarray:
        value = np.asarray(self._array[:, self._storage_index, ...])
        if dtype is not None:
            value = value.astype(dtype, copy=False)
        if copy is True:
            value = value.copy()
        return value

    def read_rows(self, rows: Any) -> np.ndarray:
        """Read an event-axis subset without decoding the full packed column."""

        if isinstance(rows, slice):
            selection = rows
        else:
            selection = np.asarray(rows, dtype=np.int64)
            if selection.ndim != 1:
                raise IndexError("event row selection must be one-dimensional")
        physical = (selection, self._storage_index) + (slice(None),) * len(
            self._field_shape
        )
        return np.asarray(self._array.get_orthogonal_selection(physical))

    def __getitem__(self, key: Any) -> Any:
        if key is Ellipsis:
            return self._array[:, self._storage_index, ...]
        if isinstance(key, tuple):
            if not key:
                return np.asarray(self)[key]
            if any(item is Ellipsis or item is None for item in key):
                return np.asarray(self)[key]
            return self._array[(key[0], self._storage_index, *key[1:])]
        return self._array[key, self._storage_index, ...]

    def __repr__(self) -> str:
        return (
            f"PackedFieldView(shape={self.shape!r}, dtype={self.dtype!r}, "
            f"chunks={self.chunks!r})"
        )


class NativeVideoStat:
    """Read-only handle for a hash-verified native episode trace."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        zarr, _ = _require_zarr()
        self.path = Path(path).expanduser().resolve()
        self.manifest = validate_episode_store(self.path)
        try:
            self._root = zarr.open_consolidated(str(self.path), mode="r")
        except Exception:
            self._root = zarr.open_group(str(self.path), mode="r")
        self.event_count = int(self.manifest["event_count"])
        self.num_actions = int(self.manifest["num_actions"])
        self.metadata = MappingProxyType(dict(self.manifest.get("metadata", {})))
        self.tree_rep_schema = MappingProxyType(
            dict(self.manifest.get("tree_rep_schema", {}))
        )
        fields, _ = _validated_packed_schema(self.manifest)
        self._fields = {
            name: _PackedFieldView(
                self._root["packs"][spec["storage_group"]],
                spec["storage_index"],
                spec["shape"],
            )
            for name, spec in fields.items()
        }
        self.field_names = tuple(sorted(self._fields))

    @property
    def format(self) -> str:
        return TRACE_FORMAT

    def has_field(self, name: str) -> bool:
        return name in self.field_names

    def field(self, name: str) -> Any:
        if not self.has_field(name):
            raise KeyError(name)
        return self._fields[name]

    def read_field_rows(self, name: str, rows: Any) -> np.ndarray:
        """Read selected event rows for one logical field."""

        return self.field(name).read_rows(rows)

    def storage_valid(self, name: str) -> np.ndarray:
        if not self.has_field(name):
            raise KeyError(name)
        # Version 2 is strict-schema: every logical field is physically
        # present in every committed row.  Optional semantics remain encoded
        # by the explicit ``*_valid`` logical fields emitted by the runner.
        return np.ones(self.event_count, dtype=np.bool_)

    def explicit_valid(self, name: str) -> np.ndarray:
        """Return caller validity, falling back to the storage-row validity."""

        explicit = name + "_valid"
        if self.has_field(explicit):
            return _as_event_vector(self, explicit).astype(np.bool_, copy=False)
        return self.storage_valid(name)

    def legacy_view(self, *, materialize: bool = False):
        view = LegacyProjection(self)
        return view.legacy_view(materialize=materialize)


def open_video_stat(path: str | os.PathLike[str], *, materialize: bool = False):
    """Open trusted legacy NPY or a validated native Zarr episode store.

    ``materialize=False`` returns a format-aware read-only handle.  With
    ``materialize=True`` both formats return a historical exact-key mapping.
    Historical object-NPY files necessarily execute NumPy pickle loading and
    must therefore come from a trusted source.
    """

    source = Path(path).expanduser().resolve()
    if source.is_dir():
        native = NativeVideoStat(source)
        opened: Any = LegacyProjection(native)
    elif source.is_file() and source.suffix.lower() == ".npy":
        opened = LegacyVideoStat(source)
    else:
        raise FileNotFoundError(f"not a video-stat NPY or native store: {source}")
    if materialize:
        return opened.legacy_view(materialize=True)
    return opened


def _slice_legacy_payload(payload: Mapping[str, Any], row_slice: Any) -> dict[str, Any]:
    status = np.asarray(payload["status"])
    indices = _normalise_row_slice(row_slice, len(status))
    output: dict[str, Any] = {}
    for key in LEGACY_KEYS:
        value = payload[key]
        if value is None:
            output[key] = None
        elif key == "tree_reps":
            if not isinstance(value, Mapping):
                raise LegacyProjectionError("tree_reps must be a mapping")
            output[key] = {
                str(name): np.asarray(array)[indices]
                for name, array in sorted(value.items())
            }
        else:
            output[key] = np.asarray(value)[indices]
    return output


def export_legacy(
    path: str | os.PathLike[str],
    out: str | os.PathLike[str],
    episode_slice: Any = None,
) -> Path:
    """Atomically export an exact-key historical object-NPY compatibility file.

    ``episode_slice`` slices projected *legacy rows*.  It is named for command
    line compatibility; one native store already represents one episode.
    """

    opened = open_video_stat(path, materialize=False)
    if isinstance(opened, LegacyProjection):
        payload = opened.materialize_slice(episode_slice)
    else:
        # Historical object-NPY is intrinsically eager because loading its
        # trusted pickle produces the complete mapping.  Native stores use the
        # bounded row path above.
        payload = _slice_legacy_payload(opened, episode_slice)
    destination = Path(out).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"legacy export already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.partial"
    )
    try:
        with temporary.open("xb") as output:
            np.save(output, payload, allow_pickle=True)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _readonly_copy(value: Any, dtype: Any) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class BehaviorEpisodeDescriptor:
    """Immutable logical-episode identity produced by the current BC loader."""

    global_episode_index: int
    file_index: int
    local_episode_index: int
    source_path: Path
    source_size: int
    source_sha256: str
    subject: int
    session: int
    block: int
    game: int
    observation_source_index: np.ndarray
    decision_times: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    done: np.ndarray
    truncated: np.ndarray
    alignment_protocol: str = ALIGNMENT_PROTOCOL

    @property
    def logical_observation_count(self) -> int:
        return int(len(self.observation_source_index))

    @property
    def edge_count(self) -> int:
        return int(len(self.actions))

    @property
    def scored_edge_count(self) -> int:
        return max(0, self.edge_count - 2)


class BehaviorEpisodeSource:
    """Read-only, on-demand image access for a current BC-loader index.

    The loader remains the sole authority for episode splitting, causal 15-Hz
    resampling, reward alignment, preprocessing, and frame-stack padding.  A
    compressed NPZ ``image.npy`` member is streamed once to a scratch NPY and
    subsequently memory-mapped.  Repeated episodes/source paths share both the
    SHA-256 calculation and scratch mmap.
    """

    def __init__(
        self,
        loader: Any,
        *,
        scratch_dir: Optional[str | os.PathLike[str]] = None,
        expected_sources: Optional[Mapping[str | os.PathLike[str], Mapping[str, Any]]] = None,
        allowed_source_paths: Optional[Sequence[str | os.PathLike[str]]] = None,
    ) -> None:
        required = (
            "file_records",
            "_episodes",
            "_preprocess_frame",
            "_stack_source_indices",
            "frame_stack_n",
        )
        missing = [name for name in required if not hasattr(loader, name)]
        if missing:
            raise TypeError(f"loader lacks current BC-loader contract: {missing}")
        self.loader = loader
        self._temporary: Optional[tempfile.TemporaryDirectory[str]] = None
        if scratch_dir is None:
            self._temporary = tempfile.TemporaryDirectory(
                prefix="thinker-behavior-images-"
            )
            self.scratch_dir = Path(self._temporary.name).resolve()
        else:
            self.scratch_dir = Path(scratch_dir).expanduser().resolve()
            self.scratch_dir.mkdir(parents=True, exist_ok=True)
        self._image_mmaps: dict[str, np.ndarray] = {}
        self._closed = False

        allowed = (
            None
            if allowed_source_paths is None
            else {
                Path(path).expanduser().resolve() for path in allowed_source_paths
            }
        )
        expected = (
            None
            if expected_sources is None
            else {
                Path(path).expanduser().resolve(): dict(identity)
                for path, identity in expected_sources.items()
            }
        )

        identities: dict[Path, tuple[int, str]] = {}
        for record in loader.file_records:
            path = Path(record.path).expanduser().resolve()
            if allowed is not None and path not in allowed:
                continue
            if path not in identities:
                if expected is None:
                    identities[path] = self._stable_identity(path)
                else:
                    if path not in expected:
                        raise TraceValidationError(
                            f"selected behavior source lacks coordinator identity: {path}"
                        )
                    identities[path] = self._preverified_identity(
                        path, expected[path]
                    )

        descriptors: list[BehaviorEpisodeDescriptor] = []
        for global_index, episode in enumerate(loader._episodes):
            record = loader.file_records[int(episode.file_index)]
            path = Path(record.path).expanduser().resolve()
            if path not in identities:
                continue
            size, digest = identities[path]
            descriptors.append(
                BehaviorEpisodeDescriptor(
                    global_episode_index=global_index,
                    file_index=int(episode.file_index),
                    local_episode_index=int(episode.episode_index),
                    source_path=path,
                    source_size=size,
                    source_sha256=digest,
                    subject=int(record.subject),
                    session=int(record.session),
                    block=int(record.block),
                    game=int(record.game),
                    observation_source_index=_readonly_copy(
                        episode.observation_source_index, np.int64
                    ),
                    decision_times=_readonly_copy(episode.decision_times, np.float64),
                    actions=_readonly_copy(episode.actions, np.int64),
                    rewards=_readonly_copy(episode.rewards, np.float32),
                    done=_readonly_copy(episode.done, np.bool_),
                    truncated=_readonly_copy(episode.truncated, np.bool_),
                )
            )
        self.episodes = tuple(descriptors)
        if not self.episodes:
            raise ValueError("no logical episodes matched the selected source paths")
        self._descriptors_by_global_index = {
            descriptor.global_episode_index: descriptor
            for descriptor in self.episodes
        }
        grouped: dict[str, list[int]] = {}
        for descriptor in self.episodes:
            grouped.setdefault(descriptor.source_sha256, []).append(
                descriptor.global_episode_index
            )
        self.source_episode_indices = MappingProxyType(
            {digest: tuple(indices) for digest, indices in sorted(grouped.items())}
        )

    @staticmethod
    def _stable_identity(path: Path) -> tuple[int, str]:
        identity = behavior_source_identity(path)
        return int(identity["size"]), str(identity["sha256"])

    @staticmethod
    def _preverified_identity(
        path: Path, identity: Mapping[str, Any]
    ) -> tuple[int, str]:
        """Trust a coordinator digest after cheap immutable-snapshot checks."""

        try:
            expected_size = int(identity["size"])
            digest = str(identity["sha256"])
        except (KeyError, TypeError, ValueError) as error:
            raise TraceValidationError(
                f"invalid coordinator source identity for {path}"
            ) from error
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise TraceValidationError(f"invalid source SHA-256 for {path}")
        stat = path.stat()
        if int(stat.st_size) != expected_size:
            raise TraceValidationError(
                f"behavior source size changed since coordinator hashing: {path}"
            )
        optional_stat_fields = {
            "mtime_ns": int(stat.st_mtime_ns),
            "inode": int(stat.st_ino),
            "device": int(stat.st_dev),
        }
        for key, actual in optional_stat_fields.items():
            if key in identity and int(identity[key]) != actual:
                raise TraceValidationError(
                    f"behavior source {key} changed since coordinator hashing: {path}"
                )
        return expected_size, digest

    def __len__(self) -> int:
        return len(self.episodes)

    def descriptor(self, index: int) -> BehaviorEpisodeDescriptor:
        return self.episodes[int(index)]

    def _resolve_descriptor(
        self, value: int | BehaviorEpisodeDescriptor
    ) -> BehaviorEpisodeDescriptor:
        if isinstance(value, BehaviorEpisodeDescriptor):
            selected = self._descriptors_by_global_index.get(
                value.global_episode_index
            )
            if (
                selected is None
                or selected.source_sha256 != value.source_sha256
                or selected.local_episode_index != value.local_episode_index
            ):
                raise ValueError("episode descriptor does not belong to this source")
            return selected
        return self.descriptor(int(value))

    def descriptor_by_global_index(self, index: int) -> BehaviorEpisodeDescriptor:
        try:
            return self._descriptors_by_global_index[int(index)]
        except KeyError as error:
            raise IndexError(f"global episode index is not selected: {index}") from error

    def iter_source_groups(
        self,
    ) -> Iterator[tuple[str, tuple[BehaviorEpisodeDescriptor, ...]]]:
        """Yield deterministic source bundles for one-mmap-at-a-time workers."""

        for digest, indices in self.source_episode_indices.items():
            yield digest, tuple(
                self._descriptors_by_global_index[index] for index in indices
            )

    def _check_open(self) -> None:
        if self._closed:
            raise BehaviorTraceError("behavior episode source is closed")

    @staticmethod
    def _image_member(archive: zipfile.ZipFile) -> zipfile.ZipInfo:
        candidates = [
            info
            for info in archive.infolist()
            if info.filename == "image.npy" or info.filename.endswith("/image.npy")
        ]
        if len(candidates) != 1:
            raise TraceValidationError(
                f"behavior NPZ must contain exactly one image.npy, found {len(candidates)}"
            )
        return candidates[0]

    def _extract_images(self, descriptor: BehaviorEpisodeDescriptor) -> Path:
        target = self.scratch_dir / f"{descriptor.source_sha256}.image.npy"
        sidecar = target.with_suffix(target.suffix + ".json")
        if target.is_file() and sidecar.is_file():
            try:
                metadata = json.loads(sidecar.read_text(encoding="utf-8"))
                if (
                    metadata.get("source_sha256") == descriptor.source_sha256
                    and int(metadata.get("npy_size", -1)) == target.stat().st_size
                    and metadata.get("image_npy_sha256") == file_sha256(target)
                ):
                    return target
            except (OSError, ValueError, json.JSONDecodeError):
                pass

        temporary = self.scratch_dir / f".{target.name}.{uuid.uuid4().hex}.partial"
        image_digest = hashlib.sha256()
        try:
            with zipfile.ZipFile(descriptor.source_path, mode="r") as archive:
                member = self._image_member(archive)
                with archive.open(member, mode="r") as source, temporary.open("xb") as output:
                    for block in iter(lambda: source.read(8 << 20), b""):
                        image_digest.update(block)
                        output.write(block)
                    output.flush()
                    os.fsync(output.fileno())
            images = np.load(temporary, mmap_mode="r", allow_pickle=False)
            if images.dtype.hasobject:
                raise TraceValidationError("behavior image member may not be object dtype")
            del images
            os.replace(temporary, target)
            _atomic_write_json(
                sidecar,
                {
                    "source_path": str(descriptor.source_path),
                    "source_sha256": descriptor.source_sha256,
                    "npy_size": int(target.stat().st_size),
                    "image_npy_sha256": image_digest.hexdigest(),
                },
            )
        finally:
            if temporary.exists():
                temporary.unlink()
        return target

    def open_images(
        self, episode_index: int | BehaviorEpisodeDescriptor
    ) -> np.ndarray:
        """Return a shared read-only mmap for an episode's raw image member."""

        self._check_open()
        descriptor = self._resolve_descriptor(episode_index)
        cached = self._image_mmaps.get(descriptor.source_sha256)
        if cached is None:
            path = self._extract_images(descriptor)
            cached = np.load(path, mmap_mode="r", allow_pickle=False)
            self._image_mmaps[descriptor.source_sha256] = cached
        maximum = int(np.max(descriptor.observation_source_index, initial=-1))
        if maximum >= len(cached):
            raise TraceValidationError(
                "logical episode references an image beyond the NPZ image member"
            )
        return cached

    def release_source(
        self,
        source_sha256_or_episode_index: str | int | BehaviorEpisodeDescriptor,
        *,
        remove_scratch: bool = False,
    ) -> None:
        """Release one source mmap, optionally deleting its extracted cache.

        Grouping sibling episodes by source and calling this method between
        groups bounds open mappings to one archive.  Existing arrays returned
        by :meth:`open_images` must no longer be used after release.
        """

        if isinstance(source_sha256_or_episode_index, BehaviorEpisodeDescriptor):
            digest = self._resolve_descriptor(
                source_sha256_or_episode_index
            ).source_sha256
        elif isinstance(source_sha256_or_episode_index, (int, np.integer)):
            digest = self.descriptor(int(source_sha256_or_episode_index)).source_sha256
        else:
            digest = str(source_sha256_or_episode_index)
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError("source SHA-256 must be 64 lowercase hexadecimal characters")
        array = self._image_mmaps.pop(digest, None)
        if array is not None:
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()
        if remove_scratch:
            target = self.scratch_dir / f"{digest}.image.npy"
            sidecar = target.with_suffix(target.suffix + ".json")
            for path in (target, sidecar):
                if path.exists():
                    path.unlink()

    def stacked_frame(
        self,
        episode_index: int | BehaviorEpisodeDescriptor,
        observation_position: int,
    ) -> np.ndarray:
        """Materialise one preprocessed CHW frame stack with loader-exact padding."""

        descriptor = self._resolve_descriptor(episode_index)
        position = int(observation_position)
        if not 0 <= position < descriptor.logical_observation_count:
            raise IndexError("logical observation position outside the episode")
        episode = self.loader._episodes[descriptor.global_episode_index]
        source_indices = self.loader._stack_source_indices(episode, position)
        images = self.open_images(episode_index)
        frames = [
            self.loader._preprocess_frame(images[int(source_index)])
            for source_index in source_indices
        ]
        return np.concatenate(frames, axis=0)

    def replay_spec(
        self, episode_index: int | BehaviorEpisodeDescriptor
    ) -> dict[str, Any]:
        """Return full-episode transition metadata without materialising images.

        This mirrors a BC window beginning at ``burn_in_edge=1``: action zero
        is the previous action, action one is the unscored burn-in edge, and
        actions two onward are scored.
        """

        descriptor = self._resolve_descriptor(episode_index)
        if descriptor.edge_count < 3:
            raise ValueError("episode has fewer than the two predecessors plus target")
        score_mask = np.ones(descriptor.edge_count - 1, dtype=np.bool_)
        score_mask[0] = False
        return {
            "initial_prev_action": np.int64(descriptor.actions[0]),
            "actions_seq": descriptor.actions[1:].copy(),
            "rewards_seq": descriptor.rewards[1:].copy(),
            "done_seq": descriptor.done[1:].copy(),
            "truncated_seq": descriptor.truncated[1:].copy(),
            "score_mask": score_mask,
            "observation_positions": np.arange(
                1, descriptor.logical_observation_count, dtype=np.int64
            ),
            "decision_times": descriptor.decision_times[1:].copy(),
            "observation_source_index": descriptor.observation_source_index[1:].copy(),
        }

    def iter_replay_observations(
        self, episode_index: int | BehaviorEpisodeDescriptor
    ) -> Iterator[np.ndarray]:
        """Yield full-episode stacked observations one at a time."""

        positions = self.replay_spec(episode_index)["observation_positions"]
        for position in positions:
            yield self.stacked_frame(episode_index, int(position))

    def close(self) -> None:
        if self._closed:
            return
        for digest in tuple(self._image_mmaps):
            self.release_source(digest)
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None
        self._closed = True

    def __enter__(self) -> "BehaviorEpisodeSource":
        self._check_open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.close()
        return False


__all__ = [
    "ALIGNMENT_PROTOCOL",
    "BehaviorEpisodeDescriptor",
    "BehaviorEpisodeSource",
    "BehaviorTraceError",
    "CONTROL_PROCEED",
    "CONTROL_RESET",
    "CONTROL_STOP",
    "EpisodeZarrWriter",
    "LEGACY_KEYS",
    "LegacyProjectionError",
    "NativeVideoStat",
    "PHASE_SEARCH",
    "TRACE_FORMAT",
    "TRACE_VERSION",
    "TraceValidationError",
    "behavior_source_identity",
    "export_legacy",
    "file_sha256",
    "open_video_stat",
    "read_episode_manifest",
    "validate_episode_store",
]
