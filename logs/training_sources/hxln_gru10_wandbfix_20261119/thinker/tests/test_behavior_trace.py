from pathlib import Path
from types import SimpleNamespace
import json
import math

import numpy as np
import pytest


pytest.importorskip("zarr")
pytest.importorskip("numcodecs")

import thinker.behavior_trace as behavior_trace  # noqa: E402
from thinker.behavior_trace import (  # noqa: E402
    BehaviorEpisodeSource,
    EpisodeZarrWriter,
    LEGACY_KEYS,
    LegacyProjection,
    NativeVideoStat,
    TraceValidationError,
    behavior_source_identity,
    export_legacy,
    open_video_stat,
    validate_episode_store,
)


def _event(index, *, num_actions=3, **updates):
    event = {
        "initial_root": np.bool_(False),
        "phase_pre": np.int8(0),
        "accepted_control": np.int8(-1),
        "accepted_control_valid": np.bool_(False),
        "accepted_primary_action": np.int16(-1),
        "accepted_primary_action_valid": np.bool_(False),
        "search_computation": np.bool_(False),
        "real_transition": np.bool_(False),
        "search_state_reset": np.bool_(False),
        "forced_root_reset": np.bool_(False),
        "forced_stop": np.bool_(False),
        "actor_reset": np.bool_(False),
        "human_action": np.int16(-1),
        "human_action_valid": np.bool_(False),
        "thinker_action": np.int16(-1),
        "thinker_action_valid": np.bool_(False),
        "actor_policy": np.arange(num_actions, dtype=np.float32) + index,
        "actor_policy_valid": np.bool_(True),
        "step_times": np.arange(4, dtype=np.float32),
        "step_times_valid": np.bool_(True),
        "env_return": np.float32(index),
        "env_return_valid": np.bool_(True),
        "cur_rewards": np.asarray([index, -index], dtype=np.float32),
        "cur_rewards_valid": np.bool_(True),
        "im_vectors": np.full((1, 2, 2), index, dtype=np.float32),
        "im_vectors_valid": np.bool_(True),
        "im_vp_vectors": np.full((1, 2, 2), index + 0.5, dtype=np.float32),
        "im_vp_vectors_valid": np.bool_(True),
        "tree_reps_raw": np.arange(8, dtype=np.float32) + index,
        "tree_reps_raw_valid": np.bool_(True),
        "real_cursor": np.int64(index),
        "augmented_index": np.int64(index),
    }
    event.update(updates)
    return event


def _events():
    return [
        _event(0, initial_root=np.bool_(True)),
        _event(
            1,
            accepted_control=np.int8(0),
            accepted_control_valid=np.bool_(True),
            accepted_primary_action=np.int16(2),
            accepted_primary_action_valid=np.bool_(True),
            search_computation=np.bool_(True),
        ),
        _event(
            2,
            accepted_control=np.int8(1),
            accepted_control_valid=np.bool_(True),
            accepted_primary_action=np.int16(1),
            accepted_primary_action_valid=np.bool_(True),
            search_computation=np.bool_(True),
            actor_reset=np.bool_(True),
        ),
        # Explicit STOP consumes no model/search computation and is omitted.
        _event(
            3,
            accepted_control=np.int8(2),
            accepted_control_valid=np.bool_(True),
        ),
        # The watchdog's final accepted computation remains legacy status 2.
        _event(
            4,
            accepted_control=np.int8(0),
            accepted_control_valid=np.bool_(True),
            accepted_primary_action=np.int16(0),
            accepted_primary_action_valid=np.bool_(True),
            search_computation=np.bool_(True),
            forced_stop=np.bool_(True),
        ),
        # Ordinary next-stage initialization is not legacy forced reset.
        _event(
            5,
            real_transition=np.bool_(True),
            search_state_reset=np.bool_(True),
            human_action=np.int16(2),
            human_action_valid=np.bool_(True),
            thinker_action=np.int16(1),
            thinker_action_valid=np.bool_(True),
        ),
        _event(
            6,
            real_transition=np.bool_(True),
            search_state_reset=np.bool_(True),
            forced_root_reset=np.bool_(True),
            human_action=np.int16(0),
            human_action_valid=np.bool_(True),
            thinker_action=np.int16(2),
            thinker_action_valid=np.bool_(True),
        ),
    ]


def _write_store(partial, final, events=None, *, metadata=None, images="none"):
    writer = EpisodeZarrWriter(
        partial,
        metadata={"checkpoint_sha256": "a" * 64} if metadata is None else metadata,
        num_actions=3,
        tree_rep_schema={"root_action": slice(0, 3), "root_r": slice(3, 4)},
        chunk_events=2,
        images=images,
    )
    for event in _events() if events is None else events:
        writer.write_event(event)
    return writer.finalize(final)


def test_native_round_trip_and_exact_legacy_projection(tmp_path):
    partial = tmp_path / "episode.partial"
    final = tmp_path / "episode.zarr"
    manifest = _write_store(partial, final)

    assert not partial.exists()
    assert manifest["event_count"] == 7
    assert validate_episode_store(final) == manifest

    opened = open_video_stat(final)
    assert isinstance(opened, LegacyProjection)
    assert isinstance(opened.native, NativeVideoStat)
    assert tuple(opened) == LEGACY_KEYS
    assert tuple(opened.legacy_view(materialize=True)) == LEGACY_KEYS

    np.testing.assert_array_equal(opened["status"], [0, 2, 2, 1, 2, 0, 0, 3])
    real_boundaries = np.flatnonzero(opened["status"] == 0)
    status2_per_completed_stage = [
        int(np.count_nonzero(opened["status"][left + 1 : right] == 2))
        for left, right in zip(real_boundaries[:-1], real_boundaries[1:])
    ]
    assert status2_per_completed_stage == [3, 0]
    assert opened["status"].dtype == np.int64
    assert opened["human_action"].dtype == np.int64
    assert opened["thinker_action"].dtype == np.int64
    assert opened["human_action"].shape == (8, 3)
    np.testing.assert_array_equal(opened["human_action"][5], [0, 0, 1])
    np.testing.assert_array_equal(opened["thinker_action"][5], [0, 1, 0])
    np.testing.assert_array_equal(opened["human_action"][6], [1, 0, 0])
    # Synthetic status 1/3 rows never pretend to carry an executed action.
    np.testing.assert_array_equal(opened["human_action"][[3, 7]], 0)

    assert opened["actor_policy"].shape == (8, 1, 1, 1, 3)
    assert opened["actor_policy"].dtype == np.float32
    assert opened["step_times"].shape == (8, 4)
    assert opened["step_times"].dtype == np.float32
    assert opened["env_return"].shape == (8,)
    assert opened["env_return"].dtype == np.float32
    assert opened["env_return"][3] == 0
    assert opened["env_return"][7] == 0
    assert opened["cur_rewards"].shape == (8, 2)
    assert opened["im_vectors"].shape == (8, 1, 2, 2)
    assert opened["im_vp_vectors"].shape == (8, 1, 2, 2)
    assert set(opened["tree_reps"]) == {"root_action", "root_r"}
    assert opened["tree_reps"]["root_action"].shape == (8, 3)
    np.testing.assert_array_equal(
        opened["tree_reps"]["root_action"][2],
        opened["tree_reps"]["root_action"][3],
    )

    legacy_path = export_legacy(final, tmp_path / "video_stat.npy")
    legacy = open_video_stat(legacy_path)
    assert tuple(legacy) == LEGACY_KEYS
    for key in LEGACY_KEYS:
        if key == "tree_reps":
            for tree_key in opened[key]:
                np.testing.assert_array_equal(legacy[key][tree_key], opened[key][tree_key])
        else:
            np.testing.assert_array_equal(legacy[key], opened[key])


def test_export_slice_preserves_exact_keys(tmp_path):
    final = tmp_path / "episode.zarr"
    _write_store(tmp_path / "episode.partial", final)
    output = export_legacy(final, tmp_path / "slice.npy", episode_slice=slice(1, 5))
    loaded = open_video_stat(output, materialize=True)
    assert tuple(loaded) == LEGACY_KEYS
    np.testing.assert_array_equal(loaded["status"], [2, 2, 1, 2])
    assert loaded["tree_reps"]["root_r"].shape[0] == 4


def test_export_range_reads_only_selected_native_value_rows(tmp_path, monkeypatch):
    final = tmp_path / "episode.zarr"
    events = [
        {
            **event,
            "unused_native_tensor": np.full((5, 5, 5), index, dtype=np.float32),
        }
        for index, event in enumerate(_events())
    ]
    _write_store(tmp_path / "episode.partial", final, events=events)
    full = open_video_stat(final, materialize=True)
    expected = behavior_trace._slice_legacy_payload(full, (1, 5))

    original_array = behavior_trace._PackedFieldView.__array__
    original_read_rows = behavior_trace._PackedFieldView.read_rows
    row_reads = []

    def reject_eager_large_field(self, dtype=None, copy=None):
        if int(np.prod(self.shape[1:], dtype=np.int64)) > 1:
            raise AssertionError(f"eager full-field read: {self.shape}")
        return original_array(self, dtype=dtype, copy=copy)

    def record_selected_rows(self, rows):
        selection = np.asarray(rows, dtype=np.int64)
        row_reads.append((self.shape, selection.copy()))
        return original_read_rows(self, rows)

    monkeypatch.setattr(
        behavior_trace._PackedFieldView, "__array__", reject_eager_large_field
    )
    monkeypatch.setattr(
        behavior_trace._PackedFieldView, "read_rows", record_selected_rows
    )
    output = export_legacy(final, tmp_path / "range.npy", episode_slice=(1, 5))
    loaded = open_video_stat(output, materialize=True)

    assert tuple(loaded) == LEGACY_KEYS
    assert row_reads
    assert max(len(selection) for _, selection in row_reads) <= 4
    assert not any(shape[1:] == (5, 5, 5) for shape, _ in row_reads)
    for key in LEGACY_KEYS:
        if key == "tree_reps":
            assert loaded[key].keys() == expected[key].keys()
            for tree_key in loaded[key]:
                np.testing.assert_array_equal(
                    loaded[key][tree_key], expected[key][tree_key]
                )
        elif expected[key] is None:
            assert loaded[key] is None
        else:
            np.testing.assert_array_equal(loaded[key], expected[key])


def test_schema_is_frozen_on_first_event_and_abort_is_noncanonical(tmp_path):
    partial = tmp_path / "episode.partial"
    writer = EpisodeZarrWriter(
        partial,
        metadata={},
        num_actions=3,
        tree_rep_schema={"all": slice(0, 8)},
        chunk_events=8,
    )
    writer.write_event(_event(0, initial_root=np.bool_(True)))
    missing = _event(1)
    missing.pop("cur_rewards")
    with pytest.raises(ValueError, match="schema differs"):
        writer.write_event(missing)
    wrong_dtype = _event(1)
    wrong_dtype["env_return"] = np.float64(1)
    with pytest.raises(TypeError, match="differs from first row"):
        writer.write_event(wrong_dtype)
    with pytest.raises(TypeError, match="bool/integer/float"):
        writer.write_event({**_event(1), "cur_rewards": "not numeric"})
    writer.abort()
    assert partial.is_dir()
    assert not (partial / "manifest.json").exists()
    with pytest.raises(TraceValidationError):
        validate_episode_store(partial)


def test_store_hash_is_deterministic_and_tampering_fails(tmp_path):
    first = tmp_path / "first.zarr"
    second = tmp_path / "second.zarr"
    manifest_a = _write_store(tmp_path / "first.partial", first)
    manifest_b = _write_store(tmp_path / "second.partial", second)
    assert manifest_a["files_sha256"] == manifest_b["files_sha256"]

    chunk = next(
        path
        for path in first.rglob("*")
        if path.is_file() and path.name not in {"manifest.json", ".zattrs", ".zgroup", ".zmetadata", ".zarray"}
    )
    data = bytearray(chunk.read_bytes())
    data[-1] ^= 1
    chunk.write_bytes(data)
    with pytest.raises(TraceValidationError, match="SHA-256"):
        validate_episode_store(first)


def test_nested_manifest_named_payload_is_not_exempt_from_hashing(tmp_path):
    final = tmp_path / "episode.zarr"
    _write_store(tmp_path / "episode.partial", final)
    nested = final / "packs" / "manifest.json"
    nested.write_bytes(b"undeclared nested payload")

    with pytest.raises(TraceValidationError, match="file set, size, or SHA-256"):
        validate_episode_store(final)


def _representative_dynamic_event(index):
    from thinker.dynamic_imitation import DYNAMIC_TRACE_EVENT_FIELDS

    event = {
        name: np.float32(index)
        for name in DYNAMIC_TRACE_EVENT_FIELDS
    }
    for name in DYNAMIC_TRACE_EVENT_FIELDS:
        if name.endswith("_valid") or name in {
            "initial_root",
            "score_mask",
            "search_computation",
            "search_expansion",
            "search_revisit",
            "actor_reset",
            "stop",
            "real_transition",
            "real_action_stored",
            "wait",
            "tree_token_valid",
            "stage_end",
            "search_state_reset",
            "forced_root_reset",
            "forced_stop",
            "barrier_released",
            "done",
            "truncated",
            "root_carried",
            "useful_carry",
            "voc_enabled",
            "voc_ema_available",
        }:
            event[name] = np.bool_(index == 0)
    for name in {
        "trace_schema_version",
        "augmented_index",
        "real_cursor",
        "cursor_before",
        "cursor_after",
        "edge_count",
        "pre_observation_source_index",
        "post_observation_source_index",
    }:
        event[name] = np.int64(index)
    for name in {
        "phase_pre",
        "phase_post",
        "internal_status",
        "accepted_control",
        "control_proposal",
        "execution_control",
        "policy_type",
        "voc_reconstruction_mode",
        "voc_soft_probability_source",
    }:
        event[name] = np.int8(index)
    for name in {
        "accepted_primary_action",
        "executed_primary_action",
        "human_action",
        "thinker_action",
        "primary_proposal",
        "primary_argmax",
        "execution_primary",
        "current_node_action_pre",
        "current_node_action_post",
    }:
        event[name] = np.int16(index)
    event.update(
        {
            "actor_policy": np.full((6,), index, dtype=np.float32),
            "predicted_xs": np.full((6, 84, 84), index, dtype=np.float32),
            "tree_reps_raw": np.full((186,), index, dtype=np.float32),
            "tree_reps_pre": np.full((186,), index, dtype=np.float32),
            "tree_reps_post": np.full((186,), index, dtype=np.float32),
            "step_times": np.full((30,), index, dtype=np.float32),
            "im_vectors": np.full((1, 128, 6, 6), index, dtype=np.float32),
            "im_vp_vectors": np.full((1, 128, 6, 6), index, dtype=np.float32),
            "actor_control_logits": np.full((3,), index, dtype=np.float32),
            "actor_action_probability": np.full((6,), index, dtype=np.float32),
            "actor_baseline": np.full((3,), index, dtype=np.float32),
            "cur_rewards": np.full((2,), index, dtype=np.float32),
            "voc_features": np.full((256,), index, dtype=np.float32),
            "voc_online_raw_q": np.full((2,), index, dtype=np.float32),
            "voc_online_q": np.full((2,), index, dtype=np.float32),
            "voc_ema_raw_q": np.full((2,), index, dtype=np.float32),
            "voc_ema_q": np.full((2,), index, dtype=np.float32),
        }
    )
    return event


def test_dynamic_field_packing_bounds_committed_files_and_135k_projection(tmp_path):
    from thinker.dynamic_imitation import DYNAMIC_TRACE_EVENT_FIELDS

    partial = tmp_path / "packed.partial"
    final = tmp_path / "packed.zarr"
    writer = EpisodeZarrWriter(
        partial,
        metadata={"episode_id": "representative-dynamic-event"},
        num_actions=6,
        tree_rep_schema={"all": slice(0, 186)},
        chunk_events=256,
    )
    for index in range(5):
        writer.write_event(_representative_dynamic_event(index))
    manifest = writer.finalize(final)

    field_count = len(manifest["field_schema"])
    pack_count = len(manifest["pack_schema"])
    assert field_count == len(DYNAMIC_TRACE_EVENT_FIELDS)
    assert pack_count < field_count // 8
    assert not (final / "valid").exists()

    # Version 1 committed one event array and one always-true validity array
    # per logical field.  With one chunk each that is already four files per
    # field, before fixed group metadata.  Version 2 stores one array/chunk per
    # dtype-and-shape pack.
    legacy_v1_five_event_lower_bound = field_count * 4
    assert len(manifest["files"]) < legacy_v1_five_event_lower_bound // 10

    projected_events = 135_000
    packed_projected_chunks = sum(
        math.ceil(projected_events / pack["chunks"][0])
        for pack in manifest["pack_schema"]
    )
    legacy_v1_projected_chunks = (
        field_count * 2 * math.ceil(projected_events / 256)
    )
    assert packed_projected_chunks < legacy_v1_projected_chunks // 10

    native = NativeVideoStat(final)
    actor_policy = native.field("actor_policy")
    assert actor_policy.shape == (5, 6)
    assert actor_policy.dtype == np.dtype(np.float32)
    assert actor_policy.chunks[0] == 256
    np.testing.assert_array_equal(actor_policy[3], np.full(6, 3, np.float32))
    np.testing.assert_array_equal(
        actor_policy[1:4],
        np.stack([np.full(6, value, np.float32) for value in range(1, 4)]),
    )
    np.testing.assert_array_equal(
        np.asarray(actor_policy)[:, 0], np.arange(5, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        native.storage_valid("actor_policy"), np.ones(5, dtype=np.bool_)
    )


def test_packed_manifest_column_mutation_is_rejected(tmp_path):
    final = tmp_path / "packed.zarr"
    _write_store(tmp_path / "packed.partial", final)
    manifest_path = final / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    first_group = manifest["field_schema"][0]["storage_group"]
    same_pack = [
        item for item in manifest["field_schema"]
        if item["storage_group"] == first_group
    ]
    assert len(same_pack) >= 2
    same_pack[1]["storage_index"] = same_pack[0]["storage_index"]
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    with pytest.raises(TraceValidationError, match="repeats column"):
        validate_episode_store(final)


def test_finalize_fsyncs_manifest_and_atomic_publish_parents(tmp_path, monkeypatch):
    synced = []
    monkeypatch.setattr(
        behavior_trace,
        "_fsync_directory",
        lambda path: synced.append(Path(path).resolve()),
    )
    partial = tmp_path / "episode.partial"
    final = tmp_path / "episode.zarr"
    _write_store(partial, final, events=[_event(0, initial_root=np.bool_(True))])

    assert partial.resolve() in synced
    assert final.parent.resolve() in synced
    assert final.is_dir()
    assert not partial.exists()


def test_finalize_durability_order_is_payload_then_manifest_then_publish(
    tmp_path, monkeypatch
):
    partial = (tmp_path / "episode.partial").resolve()
    final = (tmp_path / "episode.zarr").resolve()
    calls = []
    real_file_sync = behavior_trace._fsync_regular_file
    real_dir_sync = behavior_trace._fsync_directory
    real_atomic_json = behavior_trace._atomic_write_json
    real_validate = behavior_trace.validate_episode_store
    real_replace = behavior_trace.os.replace

    def sync_file(path):
        path = Path(path).resolve()
        calls.append(("file", path.relative_to(partial).as_posix()))
        real_file_sync(path)

    def sync_directory(path):
        path = Path(path).resolve()
        label = (
            path.relative_to(partial).as_posix()
            if path == partial or partial in path.parents
            else str(path)
        )
        calls.append(("directory", label))
        real_dir_sync(path)

    def write_manifest(path, value):
        calls.append(("manifest", Path(path).name))
        real_atomic_json(path, value)

    def validate(path):
        calls.append(("validate", str(Path(path).resolve())))
        return real_validate(path)

    def replace(source, destination):
        kind = "publish" if Path(destination).resolve() == final else "replace"
        calls.append((kind, str(Path(destination))))
        real_replace(source, destination)

    monkeypatch.setattr(behavior_trace, "_fsync_regular_file", sync_file)
    monkeypatch.setattr(behavior_trace, "_fsync_directory", sync_directory)
    monkeypatch.setattr(behavior_trace, "_atomic_write_json", write_manifest)
    monkeypatch.setattr(behavior_trace, "validate_episode_store", validate)
    monkeypatch.setattr(behavior_trace.os, "replace", replace)

    _write_store(partial, final, events=[_event(0, initial_root=np.bool_(True))])

    manifest_at = next(index for index, call in enumerate(calls) if call[0] == "manifest")
    validate_at = next(index for index, call in enumerate(calls) if call[0] == "validate")
    publish_at = next(index for index, call in enumerate(calls) if call[0] == "publish")
    assert manifest_at < validate_at < publish_at
    pre_manifest_kinds = [kind for kind, _ in calls[:manifest_at]]
    assert {"file", "directory"}.issubset(pre_manifest_kinds)
    assert max(
        index for index, kind in enumerate(pre_manifest_kinds) if kind == "file"
    ) < min(
        index for index, kind in enumerate(pre_manifest_kinds) if kind == "directory"
    )
    assert not any(call[0] == "file" for call in calls[manifest_at:])

    directories = [
        label for kind, label in calls[:manifest_at] if kind == "directory"
    ]
    assert directories
    depths = [0 if label == "." else len(Path(label).parts) for label in directories]
    assert depths == sorted(depths, reverse=True)
    assert directories[-1] == "."

    synced_files = {label for kind, label in calls if kind == "file"}
    committed_files = {
        path.relative_to(final).as_posix()
        for path in final.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    assert synced_files == committed_files


def test_finalize_payload_fsync_failure_never_writes_manifest_or_publishes(
    tmp_path, monkeypatch
):
    partial = tmp_path / "episode.partial"
    final = tmp_path / "episode.zarr"
    writer = EpisodeZarrWriter(
        partial,
        metadata={},
        num_actions=3,
        tree_rep_schema={"all": slice(0, 8)},
    )
    writer.write_event(_event(0, initial_root=np.bool_(True)))

    def fail_sync(_path):
        raise OSError("deliberate payload fsync failure")

    monkeypatch.setattr(behavior_trace, "_fsync_regular_file", fail_sync)
    with pytest.raises(OSError, match="deliberate payload fsync failure"):
        writer.finalize(final)

    assert partial.is_dir()
    assert not (partial / "manifest.json").exists()
    assert not final.exists()


def test_finalize_never_publishes_a_store_that_fails_validation(
    tmp_path, monkeypatch
):
    partial = tmp_path / "episode.partial"
    final = tmp_path / "episode.zarr"
    writer = EpisodeZarrWriter(
        partial,
        metadata={},
        num_actions=3,
        tree_rep_schema={"all": slice(0, 8)},
    )
    writer.write_event(_event(0, initial_root=np.bool_(True)))

    def reject_store(_path):
        raise TraceValidationError("deliberate validation failure")

    monkeypatch.setattr(behavior_trace, "validate_episode_store", reject_store)
    with pytest.raises(TraceValidationError, match="deliberate"):
        writer.finalize(final)
    assert partial.is_dir()
    assert not final.exists()


def test_image_selection_modes_are_enforced(tmp_path):
    event = _event(0, initial_root=np.bool_(True))
    event["real_image"] = np.zeros((3, 4, 4), dtype=np.uint8)
    event["real_image_valid"] = np.bool_(True)
    writer = EpisodeZarrWriter(
        tmp_path / "none.partial",
        metadata={},
        num_actions=3,
        tree_rep_schema={"all": slice(0, 8)},
        images="none",
    )
    with pytest.raises(ValueError, match="images='none'"):
        writer.write_event(event)
    writer.abort()

    float_image = dict(event)
    float_image["real_image"] = np.zeros((3, 4, 4), dtype=np.float32)
    writer = EpisodeZarrWriter(
        tmp_path / "float.partial",
        metadata={},
        num_actions=3,
        tree_rep_schema={"all": slice(0, 8)},
        images="real",
    )
    with pytest.raises(TypeError, match="must be uint8"):
        writer.write_event(float_image)
    writer.abort()

    final = tmp_path / "real.zarr"
    _write_store(
        tmp_path / "real.partial",
        final,
        events=[event],
        images="real",
    )
    # Image rows share bounded multi-event chunks; sparse invalid search rows
    # therefore compress together instead of creating one filesystem object
    # per planner event.
    assert NativeVideoStat(final).field("real_image").chunks[0] > 1


class _FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)
        self.calls = []

    def detach(self):
        self.calls.append("detach")
        return self

    def cpu(self):
        self.calls.append("cpu")
        return self

    def numpy(self):
        self.calls.append("numpy")
        return self.value


def test_writer_detaches_tensor_like_values(tmp_path):
    value = _FakeTensor(np.int64(4))
    event = _event(0, initial_root=np.bool_(True))
    event["real_cursor"] = value
    final = tmp_path / "tensor.zarr"
    _write_store(tmp_path / "tensor.partial", final, events=[event])
    assert value.calls == ["detach", "cpu", "numpy"]
    assert np.asarray(open_video_stat(final).native.field("real_cursor")[0]).item() == 4


class _StubLoader:
    def __init__(self, archive):
        record = SimpleNamespace(
            path=archive, subject=1, session=2, block=3, game=0
        )
        # Repeating the same path verifies digest/extraction de-duplication.
        self.file_records = [record, record]
        base = dict(
            episode_index=0,
            observation_source_index=np.arange(5, dtype=np.int64),
            decision_times=np.arange(5, dtype=np.float64) / 15.0,
            actions=np.asarray([0, 1, 2, 1], dtype=np.int64),
            rewards=np.asarray([0, 1, 2, 3], dtype=np.float32),
            done=np.asarray([0, 0, 0, 1], dtype=np.bool_),
            truncated=np.zeros(4, dtype=np.bool_),
        )
        self._episodes = [
            SimpleNamespace(file_index=0, **base),
            SimpleNamespace(file_index=1, **base),
        ]
        self.frame_stack_n = 3

    def _stack_source_indices(self, episode, observation_position):
        positions = np.arange(
            observation_position - self.frame_stack_n + 1,
            observation_position + 1,
            dtype=np.int64,
        )
        positions = np.clip(positions, 0, len(episode.observation_source_index) - 1)
        return episode.observation_source_index[positions]

    @staticmethod
    def _preprocess_frame(frame):
        return np.transpose(np.asarray(frame), (2, 0, 1)).astype(np.uint8)


def test_behavior_episode_source_streams_npz_member_to_shared_mmap(tmp_path):
    archive = tmp_path / "behavior.npz"
    images = np.empty((5, 2, 2, 1), dtype=np.uint8)
    for index in range(len(images)):
        images[index].fill(index)
    np.savez_compressed(archive, image=images, action=np.arange(5))
    scratch = tmp_path / "scratch"
    loader = _StubLoader(archive)

    with BehaviorEpisodeSource(loader, scratch_dir=scratch) as source:
        assert len(source) == 2
        assert source.descriptor(0).source_sha256 == source.descriptor(1).source_sha256
        assert source.descriptor(0).scored_edge_count == 2
        frame = source.stacked_frame(0, 1)
        assert frame.shape == (3, 2, 2)
        np.testing.assert_array_equal(frame[:, 0, 0], [0, 0, 1])
        # A repeated source path shares one extracted image member.
        source.stacked_frame(1, 2)
        assert len(list(scratch.glob("*.image.npy"))) == 1
        source.release_source(0)
        assert source._image_mmaps == {}
        # The extracted file stays reusable for the sibling episode.
        source.stacked_frame(1, 2)
        assert len(list(scratch.glob("*.image.npy"))) == 1

        replay = source.replay_spec(0)
        assert int(replay["initial_prev_action"]) == 0
        np.testing.assert_array_equal(replay["actions_seq"], [1, 2, 1])
        np.testing.assert_array_equal(replay["score_mask"], [False, True, True])
        np.testing.assert_array_equal(replay["observation_positions"], [1, 2, 3, 4])
        assert len(list(source.iter_replay_observations(0))) == 4

    # The persistent scratch cache is content-checked, not trusted by size.
    cached = next(scratch.glob("*.image.npy"))
    original = cached.read_bytes()
    damaged = bytearray(original)
    damaged[-1] ^= 1
    cached.write_bytes(damaged)
    with BehaviorEpisodeSource(loader, scratch_dir=scratch) as source:
        frame = source.stacked_frame(0, 1)
        np.testing.assert_array_equal(frame[:, 0, 0], [0, 0, 1])
    assert cached.read_bytes() == original


def test_preverified_selected_sources_avoid_worker_rehash(tmp_path, monkeypatch):
    archive = tmp_path / "behavior.npz"
    images = np.zeros((5, 2, 2, 1), dtype=np.uint8)
    np.savez_compressed(archive, image=images, action=np.arange(5))
    identity = behavior_source_identity(archive)

    def unexpected_hash(*args, **kwargs):
        raise AssertionError("GPU worker re-hashed a coordinator-verified NPZ")

    monkeypatch.setattr("thinker.behavior_trace.file_sha256", unexpected_hash)
    source = BehaviorEpisodeSource(
        _StubLoader(archive),
        scratch_dir=tmp_path / "scratch",
        expected_sources={archive: identity},
        allowed_source_paths=[archive],
    )
    try:
        assert len(source) == 2
        groups = list(source.iter_source_groups())
        assert len(groups) == 1
        assert groups[0][0] == identity["sha256"]
        assert len(groups[0][1]) == 2
        source.stacked_frame(groups[0][1][0], 1)
    finally:
        source.close()


def test_episode_source_matches_current_bc_loader_alignment(tmp_path):
    from thinker.bc_loader import FrameStackedBehavioralDataLoader

    archive = (
        tmp_path
        / "sub-001"
        / "ses-01"
        / "sub001-ses01-block1-game0.npz"
    )
    archive.parent.mkdir(parents=True)
    images = np.empty((10, 2, 2, 3), dtype=np.uint8)
    for index in range(len(images)):
        images[index].fill(index)
    first = np.zeros(10, dtype=np.bool_)
    first[0] = True
    terminal = np.zeros(10, dtype=np.bool_)
    terminal[-1] = True
    np.savez_compressed(
        archive,
        image=images,
        action=np.arange(10, dtype=np.int64) % 3,
        reward=np.arange(10, dtype=np.float32),
        is_first=first,
        is_terminal=terminal,
    )
    loader = FrameStackedBehavioralDataLoader(
        tmp_path,
        subjects=(1,),
        game_id=0,
        sessions=(1,),
        split=None,
        num_actions=3,
        scored_length=2,
        frame_stack_n=3,
        target_size=(2, 2),
        decision_hz=None,
    )
    reference = loader._materialize_refs([loader._window_refs(2)[0]], 2)
    with BehaviorEpisodeSource(loader, scratch_dir=tmp_path / "scratch") as source:
        replay = source.replay_spec(0)
        np.testing.assert_array_equal(
            source.stacked_frame(0, 1), reference["obs_seq"][0, 0]
        )
        np.testing.assert_array_equal(
            replay["actions_seq"][:3], reference["actions_seq"][0]
        )
        assert int(replay["initial_prev_action"]) == int(
            reference["initial_prev_action"][0]
        )
        np.testing.assert_array_equal(replay["score_mask"][:3], [False, True, True])


def test_canonical_dynamic_events_round_trip_through_writer(tmp_path):
    import torch
    from thinker import dynamic_imitation as dynamic

    num_actions = 3
    tree_width = 8
    reward_channels = 2

    def env_out(tree_offset=0.0):
        return SimpleNamespace(
            tree_reps=torch.arange(
                tree_width, dtype=torch.float32
            ).reshape(1, 1, tree_width)
            + tree_offset,
            reward=torch.zeros((1, 1, reward_channels), dtype=torch.float32),
            done=torch.zeros((1, 1), dtype=torch.bool),
            truncated_done=torch.zeros((1, 1), dtype=torch.bool),
        )

    actor = SimpleNamespace(
        pri_param=torch.arange(num_actions, dtype=torch.float32).reshape(
            1, 1, 1, num_actions
        ),
        search_control_logits=torch.zeros((1, 1, 3), dtype=torch.float32),
        action_prob=None,
        baseline=None,
        policy_type=torch.zeros((1, 1), dtype=torch.int64),
        policy_valid=torch.ones((1, 1), dtype=torch.bool),
        primary_valid=torch.ones((1, 1), dtype=torch.bool),
        control_valid=torch.ones((1, 1), dtype=torch.bool),
    )
    decision = dynamic.ExecutionDecision(
        primary_proposal=torch.asarray([1]),
        primary_argmax=torch.asarray([2]),
        search_control_proposal=torch.asarray([0]),
        execution_primary=torch.asarray([1]),
        execution_control=torch.asarray([0]),
        real_policy_mask=torch.asarray([False]),
    )

    def snapshot(*, status=-1, control=-1, action=-1, phase=0, steps=0):
        return {
            "phase": np.asarray([phase], dtype=np.int8),
            "internal_status": np.asarray([status], dtype=np.int8),
            "accepted_control": np.asarray([control], dtype=np.int8),
            "accepted_primary_action": np.asarray([action], dtype=np.int16),
            "executed_primary_action": np.asarray([-1], dtype=np.int16),
            "current_node_hs": (np.zeros((4, 2, 2), dtype=np.float32),),
            "search_steps": np.asarray([steps], dtype=np.int32),
            "rollout_depth": np.asarray([steps], dtype=np.int32),
        }

    initial_predicates = dynamic._initial_trace_predicates()
    search_predicates = {
        "internal_status": dynamic.TRACE_SEARCH_EXPANSION,
        "is_search_computation": True,
        "is_search_expansion": True,
        "is_search_revisit": False,
        "is_reset_control": False,
        "is_stop": False,
        "is_real_transition": False,
        "is_real_action_stored": False,
        "is_wait": False,
        "tree_token_valid": True,
        "stage_end": False,
    }
    real_predicates = {
        **search_predicates,
        "internal_status": dynamic.TRACE_REAL_TRANSITION,
        "is_search_computation": False,
        "is_search_expansion": False,
        "is_real_transition": True,
        "tree_token_valid": False,
        "stage_end": True,
    }
    common = {
        "edge_count": 2,
        "score_mask": False,
        "human_action": 2,
        "actor_out": actor,
        "decision": decision,
        "done": torch.asarray([False]),
        "truncated": torch.asarray([False]),
        "raw_info": {},
        "observed_info": {},
        "voc": {"enabled": False},
        "pre_observation_source_index": 10,
        "post_observation_source_index": 11,
        "sr_hidden_shape": (2, 2, 2),
        "vp_hidden_shape": (2, 2, 2),
    }
    root_env = env_out()
    initial = dynamic.build_dynamic_trace_event(
        initial_root=True,
        augmented_index=-1,
        cursor_before=0,
        cursor_after=0,
        pre_env_out=root_env,
        post_env_out=root_env,
        reward=root_env.reward[-1],
        pre_cenv=snapshot(),
        post_cenv=snapshot(),
        predicates=initial_predicates,
        **common,
    )
    search = dynamic.build_dynamic_trace_event(
        initial_root=False,
        augmented_index=0,
        cursor_before=0,
        cursor_after=0,
        pre_env_out=root_env,
        post_env_out=env_out(1.0),
        reward=torch.asarray([0.0]),
        pre_cenv=snapshot(),
        post_cenv=snapshot(
            status=dynamic.TRACE_SEARCH_EXPANSION,
            control=0,
            action=1,
            steps=1,
        ),
        predicates=search_predicates,
        **common,
    )
    real = dynamic.build_dynamic_trace_event(
        initial_root=False,
        augmented_index=1,
        cursor_before=0,
        cursor_after=1,
        pre_env_out=env_out(1.0),
        post_env_out=env_out(2.0),
        reward=torch.asarray([1.0]),
        pre_cenv=snapshot(phase=1, steps=1),
        post_cenv=snapshot(status=dynamic.TRACE_REAL_TRANSITION, phase=0),
        predicates=real_predicates,
        **common,
    )
    assert set(initial) == set(dynamic.DYNAMIC_TRACE_EVENT_FIELDS)
    first_schema = {
        key: (np.asarray(value).dtype.str, np.asarray(value).shape)
        for key, value in initial.items()
    }
    for event in (search, real):
        assert {
            key: (np.asarray(value).dtype.str, np.asarray(value).shape)
            for key, value in event.items()
        } == first_schema

    final = tmp_path / "canonical.zarr"
    writer = EpisodeZarrWriter(
        tmp_path / "canonical.partial",
        metadata={"source_sha256": "b" * 64},
        num_actions=num_actions,
        tree_rep_schema={"root_action": slice(0, num_actions)},
        chunk_events=2,
    )
    for event in (initial, search, real):
        writer.write_event(event)
    writer.finalize(final)
    opened = open_video_stat(final)
    np.testing.assert_array_equal(opened["status"], [0, 2, 0])
    np.testing.assert_array_equal(opened["human_action"][-1], [0, 0, 1])
    assert opened["im_vectors"].shape == (3, 1, 2, 2, 2)
    assert opened["im_vp_vectors"].shape == (3, 1, 2, 2, 2)
