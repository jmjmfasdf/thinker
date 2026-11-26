from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

import visual_behav as recorder


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(path: Path, index: int = 0) -> recorder.SourceBundle:
    stat = path.stat()
    return recorder.SourceBundle(
        source_index=index,
        source_id=path.name,
        path=str(path.resolve()),
        size=stat.st_size,
        sha256=_sha256(path),
        device=stat.st_dev,
        inode=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
    )


def _checkpoint_bundle(tmp_path: Path, *, step: int = 5_000_000):
    torch = pytest.importorskip("torch")
    root = tmp_path / "dynamic-imitation-pong-voc-sub1-leavesession4"
    root.mkdir()
    flags = {
        "xpid": "source-run",
        "name": "Pong-v5",
        "icopro_game_id": 1,
        "dynamic_search": True,
        "dynamic_factorized_control": True,
        "max_search_steps": 30,
        "max_depth": 30,
        "model_unroll_len": 30,
        "rec_t": 20,
        "frame_stack_n": 4,
        "dynamic_voc_mode": "off",
    }
    config = root / "config_c.yaml"
    config.write_text(
        "\n".join(
            f"{key}: {json.dumps(value)}" for key, value in flags.items()
        )
        + "\n",
        encoding="utf-8",
    )
    actor = root / f"ckp_actor.tar_step_{step}"
    model = root / f"ckp_model.tar_step_{step}"
    torch.save(
        {
            "real_step": step,
            "flags": flags,
            "actor_net_state_dict": {"weight": torch.ones(1)},
        },
        actor,
    )
    torch.save(
        {
            "real_step": step,
            "flags": flags,
            "model_net_state_dict": {"weight": torch.ones(1)},
        },
        model,
    )
    artifacts = {}
    for path in (config, actor, model):
        artifacts[path.name] = {
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    marker = {
        "schema_version": 1,
        "kind": "dynamic-imitation-canonical-milestone",
        "status": "VERIFIED_5M_STOPPED",
        "target_real_step": step,
        "logical": {"xpid": root.name},
        "source_run": {"xpid": "source-run"},
        "artifacts": artifacts,
    }
    (root / "milestone_5000000.json").write_text(
        json.dumps(marker, sort_keys=True), encoding="utf-8"
    )
    return root, flags


def _capture_config(
    tmp_path: Path,
    checkpoint: recorder.CheckpointBundle,
    *,
    source_hash_workers: int = 1,
    policy_mode: str = "checkpoint",
    determinism_backend_fingerprint=None,
) -> recorder.CaptureConfig:
    if determinism_backend_fingerprint is None:
        determinism_backend_fingerprint = (
            recorder.resolve_determinism_backend_fingerprint(
                [recorder.WorkerSpec(0, "cpu")]
            )
        )
    return recorder.CaptureConfig(
        checkpoint=checkpoint,
        data_root=str(tmp_path / "data"),
        output_root=str(tmp_path / "output"),
        scratch_root=None,
        base_seed=17,
        queue_size=2,
        chunk_events=8,
        images="none",
        tree_carry=True,
        policy_mode=policy_mode,
        determinism_backend_fingerprint=determinism_backend_fingerprint,
        resume=True,
        torch_threads_per_worker=1,
        source_hash_workers=source_hash_workers,
        episode_indices=(),
    )


def _event(
    index: int,
    *,
    initial: bool = False,
    real: bool = False,
    search: bool = False,
    cursor_before: int = 0,
    cursor_after: int = 0,
    num_actions: int = 3,
):
    return {
        "initial_root": np.bool_(initial),
        "forced_root_reset": np.bool_(False),
        "phase_pre": np.int8(0),
        "search_computation": np.bool_(search),
        "real_transition": np.bool_(real),
        "real_cursor": np.int64(cursor_before),
        "cursor_after": np.int64(cursor_after),
        "human_action": np.int16(1 if real else -1),
        "human_action_valid": np.bool_(real),
        "thinker_action": np.int16(2 if real else -1),
        "thinker_action_valid": np.bool_(real),
        "accepted_control": np.int8(0 if search else -1),
        "accepted_control_valid": np.bool_(search),
        "accepted_primary_action": np.int16(1 if search else -1),
        "accepted_primary_action_valid": np.bool_(search),
        "actor_reset": np.bool_(False),
        "actor_policy": np.arange(num_actions, dtype=np.float32) + index,
        "actor_policy_valid": np.bool_(True),
        "actor_real_logits_pre_override": (
            np.arange(num_actions, dtype=np.float32) + index
            if real else np.zeros(num_actions, dtype=np.float32)
        ),
        "actor_real_logits_pre_override_valid": np.bool_(real),
        "model_done_logit": np.float32(-1.5 if search else 0),
        "model_done_logit_valid": np.bool_(search),
        "model_done_target_depth": np.int16(1 if search else -1),
        "model_done_target_action": np.int16(1 if search else -1),
        "tree_reps_raw": np.arange(4, dtype=np.float32) + index,
        "tree_reps_raw_valid": np.bool_(True),
        "predicted_xs": np.full((6, 2, 2), index / 10, dtype=np.float32),
        "predicted_xs_valid": np.bool_(True),
        "step_times": np.zeros((1,), dtype=np.float32),
        "step_times_valid": np.bool_(False),
        "env_return": np.float32(index),
        "env_return_valid": np.bool_(True),
        "cur_rewards": np.asarray([index], dtype=np.float32),
        "cur_rewards_valid": np.bool_(True),
        "im_vectors": np.full((1, 2), index, dtype=np.float32),
        "im_vectors_valid": np.bool_(True),
        "im_vp_vectors": np.full((1, 2), index, dtype=np.float32),
        "im_vp_vectors_valid": np.bool_(True),
    }


class _CollectingOutput:
    def __init__(self):
        self.events = []

    def write_event(self, event):
        self.events.append(event)


def test_checkpoint_marker_and_payload_are_stably_bound(tmp_path):
    root, _ = _checkpoint_bundle(tmp_path)
    bundle = recorder.resolve_checkpoint_bundle(
        root, stable_polls=1, stable_interval_seconds=0
    )
    result = recorder.validate_checkpoint_payload_bundle(bundle)
    assert result == {
        "actor_real_step": 5_000_000,
        "model_real_step": 5_000_000,
        "embedded_source_xpid": "source-run",
        "game_id": 1,
        "environment_name": "Pong-v5",
    }
    assert Path(bundle.actor.path).name == "ckp_actor.tar_step_5000000"
    assert bundle.logical_xpid == root.name
    assert bundle.source_xpid == "source-run"


def test_checkpoint_marker_rejects_contradictory_status(tmp_path):
    root, _ = _checkpoint_bundle(tmp_path)
    marker_path = root / "milestone_5000000.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["status"] = "FAILED"
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(recorder.VisualBehaviorError, match="status is contradictory"):
        recorder.resolve_checkpoint_bundle(
            root, stable_polls=1, stable_interval_seconds=0
        )


def test_payload_rejects_actor_model_step_mismatch(tmp_path):
    root, _ = _checkpoint_bundle(tmp_path)
    bundle = recorder.resolve_checkpoint_bundle(
        root, stable_polls=1, stable_interval_seconds=0
    )
    actor = {
        "real_step": 2,
        "flags": {"xpid": "x"},
        "actor_net_state_dict": {"w": np.ones(1)},
    }
    model = {
        "real_step": 3,
        "flags": {"xpid": "x"},
        "model_net_state_dict": {"w": np.ones(1)},
    }
    with pytest.raises(recorder.VisualBehaviorError, match="real_steps disagree"):
        recorder._validate_loaded_checkpoint_payloads(
            actor,
            model,
            recorder.CheckpointBundle(
                **{**bundle.__dict__, "requested_step": None, "source_xpid": "x"}
            ),
            config_xpid="x",
        )


def test_source_enumeration_filters_game_and_preserves_sorted_order(tmp_path):
    root = tmp_path / "data"
    paths = [
        root / "sub-002" / "ses-01" / "z_game0.npz",
        root / "sub-001" / "ses-01" / "a_game0.npz",
        root / "sub-001" / "ses-01" / "a_game1.npz",
    ]
    for index, path in enumerate(paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"source-{index}".encode())
    sources = recorder.enumerate_source_bundles(
        root,
        game_id=0,
        hash_workers=2,
        stable_polls=1,
        stable_interval_seconds=0,
    )
    assert [item.source_id for item in sources] == sorted(
        [paths[0].relative_to(root).as_posix(), paths[1].relative_to(root).as_posix()]
    )
    assert [item.source_index for item in sources] == [0, 1]


def test_source_enumeration_rejects_byte_identical_paths(tmp_path):
    root = tmp_path / "data"
    first = root / "sub-001" / "ses-01" / "a_game1.npz"
    second = root / "sub-002" / "ses-01" / "b_game1.npz"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_bytes(b"same immutable archive")
    second.write_bytes(b"same immutable archive")
    with pytest.raises(recorder.VisualBehaviorError, match="byte-identical"):
        recorder.enumerate_source_bundles(
            root,
            game_id=1,
            stable_polls=1,
            stable_interval_seconds=0,
        )


def test_coordinator_indexes_only_selected_sources(tmp_path, monkeypatch):
    data = tmp_path / "data"
    selected_path = data / "sub-001" / "ses-01" / "sub001-ses01-block1-game1.npz"
    malformed_path = data / "sub-002" / "ses-01" / "sub002-ses01-block1-game1.npz"
    selected_path.parent.mkdir(parents=True)
    malformed_path.parent.mkdir(parents=True)
    np.savez(
        selected_path,
        image=np.zeros((8, 2, 2, 3), dtype=np.uint8),
        action=np.zeros((8,), dtype=np.int64),
        is_first=np.asarray([True] + [False] * 7),
        is_terminal=np.asarray([False] * 7 + [True]),
    )
    malformed_path.write_bytes(b"not an npz")
    source = _source(selected_path)
    spec = SimpleNamespace(
        game_id=1,
        num_actions=3,
        scored_length=2,
        frame_stack_n=2,
        target_size=(2, 2),
        grayscale=False,
        observation_dtype="uint8",
    )
    monkeypatch.setattr(
        recorder,
        "_import_evaluator",
        lambda: SimpleNamespace(resolve_evaluation_spec=lambda flags: spec),
    )
    monkeypatch.setattr(
        recorder, "_flags_from_stable_config", lambda config, directory: object()
    )

    class DescriptorSource:
        def __init__(self, loader, **kwargs):
            assert [record.path for record in loader.file_records] == [
                selected_path.resolve()
            ]
            self.episodes = (
                SimpleNamespace(source_path=selected_path.resolve(), local_episode_index=0),
            )

        def close(self):
            return None

    monkeypatch.setattr(
        recorder,
        "_load_behavior_trace",
        lambda: SimpleNamespace(BehaviorEpisodeSource=DescriptorSource),
    )
    config = SimpleNamespace(
        checkpoint=SimpleNamespace(config=object(), directory="checkpoint"),
        data_root=str(data),
        base_seed=0,
        episode_indices=(),
    )
    tasks = recorder._coordinator_source_tasks(config, [source])
    assert len(tasks) == 1


def test_final_source_fence_rehashes_same_metadata_mutation(tmp_path):
    path = tmp_path / "game0.npz"
    path.write_bytes(b"original")
    source = _source(path)
    path.write_bytes(b"mutated!")
    os.utime(path, ns=(source.mtime_ns, source.mtime_ns))
    assert (path.stat().st_ino, path.stat().st_size, path.stat().st_mtime_ns) == (
        source.inode,
        source.size,
        source.mtime_ns,
    )
    with pytest.raises(recorder.VisualBehaviorError, match="source bytes changed"):
        recorder.validate_source_snapshots([source], hash_workers=1)


def test_episode_seed_is_worker_and_order_independent(tmp_path):
    path = tmp_path / "game0.npz"
    path.write_bytes(b"immutable")
    source = _source(path)
    assert recorder.episode_seed(7, source, 3) == recorder.episode_seed(7, source, 3)
    assert recorder.episode_seed(7, source, 3) != recorder.episode_seed(7, source, 4)
    renamed = recorder.SourceBundle(
        **{**source.__dict__, "source_id": "renamed/location/game0.npz"}
    )
    assert recorder.episode_seed(7, source, 3) == recorder.episode_seed(7, renamed, 3)
    assert recorder.episode_store_id(source, 3) == (
        f"{source.sha256}-e00003"
    )


def test_observation_provider_prefetches_next_frame_on_one_cpu_thread():
    started_second = threading.Event()
    release_second = threading.Event()
    calls = []

    class Source:
        loader = SimpleNamespace(
            frame_stack_n=2,
            grayscale=False,
            target_size=(2, 2),
            normalize=False,
        )

        def stacked_frame(self, descriptor, position):
            calls.append((position, threading.get_ident()))
            if position == 1:
                started_second.set()
                assert release_second.wait(timeout=5)
            return np.full((6, 2, 2), position, dtype=np.uint8)

    provider = recorder._LazyObservationProvider(
        Source(), object(), np.asarray([0, 1, 2], dtype=np.int64)
    )
    assert np.all(provider[0] == 0)
    assert started_second.wait(timeout=5)
    assert calls[0][1] != threading.get_ident()
    release_second.set()
    assert np.all(provider[1] == 1)
    provider.close()
    assert len({thread_id for _, thread_id in calls}) == 1
    assert len([position for position, _ in calls]) == len(
        {position for position, _ in calls}
    )


def test_cpu_auto_workers_are_memory_capped():
    checkpoint_bytes = 128 * 1024 * 1024
    per_worker = (
        checkpoint_bytes * recorder.CPU_MODEL_MEMORY_MULTIPLIER
        + recorder.CPU_WORKER_OVERHEAD_BYTES
    )
    sizing = recorder.cpu_worker_sizing(
        20,
        cpu_count=64,
        available_memory_bytes=2 * per_worker,
        checkpoint_bytes=checkpoint_bytes,
    )
    assert sizing["memory_limit"] == 1
    assert sizing["selected_workers"] == 1


def test_physical_core_count_respects_affinity_and_smt_pairs():
    cpuinfo = """
processor : 0
physical id : 0
core id : 0

processor : 1
physical id : 0
core id : 0

processor : 2
physical id : 0
core id : 1

processor : 3
physical id : 0
core id : 1
"""
    assert recorder._physical_core_count_from_cpuinfo(cpuinfo, {0, 1, 2, 3}) == 2
    assert recorder._physical_core_count_from_cpuinfo(cpuinfo, {1, 3}) == 2


def test_cpu_auto_workers_have_no_arbitrary_eight_worker_cap():
    sizing = recorder.cpu_worker_sizing(
        100,
        cpu_count=64,
        available_memory_bytes=1 << 50,
        checkpoint_bytes=1,
    )
    assert sizing["cpu_limit"] == 16
    assert sizing["selected_workers"] == 16


def test_cuda_worker_resolution_does_not_import_cuda(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,7")
    assert recorder.resolve_worker_specs(
        "auto", source_count=10, cpu_workers="auto", dry_run=True
    ) == [
        recorder.WorkerSpec(rank=0, device="cuda:0"),
        recorder.WorkerSpec(rank=1, device="cuda:1"),
    ]


def test_auto_dry_run_resolves_backend_instead_of_silently_assuming_cpu(
    monkeypatch,
):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    real_import = recorder.importlib.import_module

    def fake_import(name):
        if name == "torch":
            return SimpleNamespace(
                cuda=SimpleNamespace(device_count=lambda: 2)
            )
        return real_import(name)

    monkeypatch.setattr(recorder.importlib, "import_module", fake_import)
    assert recorder.resolve_worker_specs(
        "auto", source_count=3, cpu_workers="auto", dry_run=True
    ) == [
        recorder.WorkerSpec(rank=0, device="cuda:0"),
        recorder.WorkerSpec(rank=1, device="cuda:1"),
    ]


def test_cuda_workers_are_capped_by_source_bundle_count():
    assert recorder.resolve_worker_specs(
        "0,1,2", source_count=1, cpu_workers="auto", dry_run=True
    ) == [recorder.WorkerSpec(rank=0, device="cuda:0")]


def test_semantic_plan_allows_changed_worker_topology(tmp_path):
    root, _ = _checkpoint_bundle(tmp_path)
    bundle = recorder.resolve_checkpoint_bundle(
        root, stable_polls=1, stable_interval_seconds=0
    )
    data = tmp_path / "data"
    data.mkdir()
    source_path = data / "game1.npz"
    source_path.write_bytes(b"source")
    source = _source(source_path)
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_properties=lambda index: SimpleNamespace(
                name="NVIDIA RTX 3090", major=8, minor=6
            ),
        )
    )
    one_workers = [recorder.WorkerSpec(0, "cuda:0")]
    two_workers = [
        recorder.WorkerSpec(0, "cuda:4"),
        recorder.WorkerSpec(1, "cuda:7"),
    ]
    one_fingerprint = recorder.resolve_determinism_backend_fingerprint(
        one_workers, torch_module=fake_torch
    )
    two_fingerprint = recorder.resolve_determinism_backend_fingerprint(
        two_workers, torch_module=fake_torch
    )
    assert one_fingerprint == two_fingerprint
    config = _capture_config(
        tmp_path,
        bundle,
        policy_mode="sampled",
        determinism_backend_fingerprint=one_fingerprint,
    )
    one = recorder.build_capture_plan(
        config, [source], one_workers
    )
    two = recorder.build_capture_plan(
        config,
        [source],
        two_workers,
    )
    assert one["semantic_sha256"] == two["semantic_sha256"]
    assert one["plan_sha256"] != two["plan_sha256"]
    output = Path(config.output_root)
    output.mkdir()
    recorder._prepare_capture_plan(output, one)
    recorder._prepare_capture_plan(output, two)
    assert len(list((output / "attempts").glob("*.json"))) == 2


def test_resume_rejects_cpu_to_cuda_backend_change(tmp_path):
    root, _ = _checkpoint_bundle(tmp_path)
    bundle = recorder.resolve_checkpoint_bundle(
        root, stable_polls=1, stable_interval_seconds=0
    )
    source_path = tmp_path / "game1.npz"
    source_path.write_bytes(b"source")
    source = _source(source_path)
    cpu_workers = [recorder.WorkerSpec(0, "cpu")]
    cuda_workers = [recorder.WorkerSpec(0, "cuda:0")]
    cuda_fingerprint = recorder.resolve_determinism_backend_fingerprint(
        cuda_workers,
        torch_module=SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                get_device_properties=lambda index: SimpleNamespace(
                    name="NVIDIA RTX 3090", major=8, minor=6
                ),
            )
        ),
    )
    cpu_plan = recorder.build_capture_plan(
        _capture_config(tmp_path, bundle), [source], cpu_workers
    )
    cuda_plan = recorder.build_capture_plan(
        _capture_config(
            tmp_path,
            bundle,
            determinism_backend_fingerprint=cuda_fingerprint,
        ),
        [source],
        cuda_workers,
    )
    assert cpu_plan["semantic_sha256"] != cuda_plan["semantic_sha256"]
    output = Path(cpu_plan["execution"]["output_root"])
    output.mkdir()
    recorder._prepare_capture_plan(output, cpu_plan)
    with pytest.raises(recorder.VisualBehaviorError, match="different capture semantics"):
        recorder._prepare_capture_plan(output, cuda_plan)


def test_resume_rejects_changed_cuda_architecture(tmp_path):
    root, _ = _checkpoint_bundle(tmp_path)
    bundle = recorder.resolve_checkpoint_bundle(
        root, stable_polls=1, stable_interval_seconds=0
    )
    source_path = tmp_path / "game1.npz"
    source_path.write_bytes(b"source")
    source = _source(source_path)
    workers = [recorder.WorkerSpec(0, "cuda:0")]

    def fingerprint(name, major, minor):
        return recorder.resolve_determinism_backend_fingerprint(
            workers,
            torch_module=SimpleNamespace(
                cuda=SimpleNamespace(
                    is_available=lambda: True,
                    get_device_properties=lambda index: SimpleNamespace(
                        name=name, major=major, minor=minor
                    ),
                )
            ),
        )

    rtx_plan = recorder.build_capture_plan(
        _capture_config(
            tmp_path,
            bundle,
            determinism_backend_fingerprint=fingerprint(
                "NVIDIA RTX 3090", 8, 6
            ),
        ),
        [source],
        workers,
    )
    a100_plan = recorder.build_capture_plan(
        _capture_config(
            tmp_path,
            bundle,
            determinism_backend_fingerprint=fingerprint("NVIDIA A100", 8, 0),
        ),
        [source],
        workers,
    )
    output = Path(rtx_plan["execution"]["output_root"])
    output.mkdir()
    recorder._prepare_capture_plan(output, rtx_plan)
    with pytest.raises(recorder.VisualBehaviorError, match="different capture semantics"):
        recorder._prepare_capture_plan(output, a100_plan)


def test_backend_fingerprint_is_torch_optional_for_cpu_and_cuda_fail_closed(
    monkeypatch,
):
    def missing_torch(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(recorder.importlib, "import_module", missing_torch)
    cpu = recorder.resolve_determinism_backend_fingerprint(
        [recorder.WorkerSpec(0, "cpu")]
    )
    assert cpu["backend"] == "cpu"
    with pytest.raises(recorder.RuntimeDependencyError, match="CUDA --dry-run"):
        recorder.resolve_determinism_backend_fingerprint(
            [recorder.WorkerSpec(0, "cuda:0")], dry_run=True
        )


def test_backend_fingerprint_rejects_heterogeneous_cuda_architectures():
    properties = {
        0: SimpleNamespace(name="NVIDIA RTX 3090", major=8, minor=6),
        1: SimpleNamespace(name="NVIDIA A100", major=8, minor=0),
    }
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_properties=lambda index: properties[index],
        )
    )
    with pytest.raises(recorder.VisualBehaviorError, match="heterogeneous"):
        recorder.resolve_determinism_backend_fingerprint(
            [
                recorder.WorkerSpec(0, "cuda:0"),
                recorder.WorkerSpec(1, "cuda:1"),
            ],
            torch_module=fake_torch,
        )


def test_canonical_sink_stores_real_frame_once_per_real_edge():
    output = _CollectingOutput()
    provider = np.stack(
        [np.full((6, 2, 2), index / 10, dtype=np.float32) for index in range(3)]
    )
    sink = recorder._CanonicalEventSink(
        output,
        images="real",
        source_frame_indices=np.asarray([100, 101, 102]),
        decision_times=np.asarray([1.0, 2.0, 3.0]),
        grayscale=False,
        observation_provider=provider,
    )
    sink.write_event(_event(0, initial=True, cursor_before=0, cursor_after=0))
    sink.write_event(_event(1, search=True, cursor_before=0, cursor_after=0))
    sink.write_event(_event(2, real=True, cursor_before=0, cursor_after=1))
    sink.assert_complete()
    assert [bool(event["real_image_valid"]) for event in output.events] == [
        True,
        False,
        True,
    ]
    assert np.all(output.events[1]["real_image"] == 0)
    assert np.all(output.events[2]["real_image"] == 26)
    assert int(output.events[2]["source_frame_index"]) == 101
    assert float(output.events[2]["decision_time"]) == 1.0
    schemas = [recorder._CanonicalEventSink._numeric_schema(row) for row in output.events]
    assert schemas[0] == schemas[1] == schemas[2]


def test_canonical_sink_projects_all_valid_predicted_images():
    output = _CollectingOutput()
    provider = np.zeros((2, 6, 2, 2), dtype=np.uint8)
    sink = recorder._CanonicalEventSink(
        output,
        images="imagined",
        source_frame_indices=np.asarray([4, 5]),
        decision_times=np.asarray([1.0, 2.0]),
        grayscale=False,
        observation_provider=provider,
    )
    initial = _event(0, initial=True)
    initial.update(predicted_xs=np.full((6, 2, 2), 0.2, dtype=np.float32))
    search = _event(1, search=True)
    search.update(predicted_xs=np.full((6, 2, 2), 0.4, dtype=np.float32))
    sink.write_event(initial)
    sink.write_event(search)
    assert bool(output.events[0]["imagined_image_valid"]) is True
    assert np.all(output.events[0]["imagined_image"] == 51)
    assert bool(output.events[1]["imagined_image_valid"]) is True
    assert output.events[1]["imagined_image"].dtype == np.uint8
    assert output.events[0]["predicted_xs"].dtype == np.float32


def test_render_panels_hold_last_valid_root_during_search():
    real = np.asarray([[[[10]]], [[[0]]], [[[20]]]], dtype=np.uint8)
    imagined = np.asarray([[[[1]]], [[[2]]], [[[3]]]], dtype=np.uint8)
    rows = list(
        recorder._iter_forward_filled_render_rows(
            [real, imagined],
            [np.asarray([True, False, True]), np.asarray([True, True, True])],
            range(3),
        )
    )
    assert [int(panels[0][0, 0, 0]) for _, panels in rows] == [10, 10, 20]
    assert [int(panels[1][0, 0, 0]) for _, panels in rows] == [1, 2, 3]


def test_timing_mode_none_rejects_claimed_valid_timing():
    output = _CollectingOutput()
    sink = recorder._CanonicalEventSink(
        output,
        images="none",
        source_frame_indices=np.asarray([1]),
        decision_times=np.asarray([1.0]),
        grayscale=False,
        observation_provider=np.zeros((1, 6, 2, 2), dtype=np.uint8),
    )
    event = _event(0, initial=True)
    event["step_times_valid"] = np.bool_(True)
    with pytest.raises(recorder.VisualBehaviorError, match="cannot claim valid"):
        sink.write_event(event)


def test_bounded_writer_has_one_owner_and_finalizes(tmp_path):
    instances = []

    class FakeWriter:
        def __init__(self, partial, **kwargs):
            self.partial = partial
            self.events = []
            self.aborted = False
            instances.append(self)

        def write_event(self, event):
            self.events.append(event)

        def finalize(self, final):
            return {"event_count": len(self.events), "final": str(final)}

        def abort(self):
            self.aborted = True

    writer = recorder._BoundedWriter(
        SimpleNamespace(EpisodeZarrWriter=FakeWriter),
        partial_path=tmp_path / ".partial",
        final_path=tmp_path / "final.zarr",
        metadata={},
        num_actions=3,
        tree_rep_schema={"x": [0, 1, None]},
        chunk_events=2,
        images="none",
        queue_size=1,
    )
    writer.write_event({"x": np.int64(1)})
    result = writer.finalize()
    assert result["event_count"] == 1
    assert len(instances) == 1
    assert instances[0].aborted is False


def test_bounded_writer_applies_byte_backpressure(tmp_path):
    entered = threading.Event()
    release = threading.Event()

    class BlockingWriter:
        def __init__(self, partial, **kwargs):
            self.count = 0

        def write_event(self, event):
            self.count += 1
            if self.count == 1:
                entered.set()
                assert release.wait(timeout=5)

        def finalize(self, final):
            return {"event_count": self.count}

        def abort(self):
            release.set()

    event = {"payload": np.zeros((64,), dtype=np.uint8)}
    retained = recorder._retained_nbytes(recorder._freeze_event(event))
    writer = recorder._BoundedWriter(
        SimpleNamespace(EpisodeZarrWriter=BlockingWriter),
        partial_path=tmp_path / ".partial",
        final_path=tmp_path / "final.zarr",
        metadata={},
        num_actions=3,
        tree_rep_schema={"x": [0, 1, None]},
        chunk_events=8,
        images="none",
        queue_size=2,
        max_inflight_bytes=retained,
    )
    writer.write_event(event)
    assert entered.wait(timeout=5)
    completed = threading.Event()

    def produce_second():
        writer.write_event(event)
        completed.set()

    producer = threading.Thread(target=produce_second)
    producer.start()
    time.sleep(0.05)
    assert not completed.is_set()
    assert writer._peak_pending_bytes <= retained
    release.set()
    producer.join(timeout=5)
    assert completed.is_set()
    assert writer.finalize()["event_count"] == 2


def test_root_manifest_proves_exact_episode_set_then_publishes_completion(tmp_path):
    root, _ = _checkpoint_bundle(tmp_path)
    bundle = recorder.resolve_checkpoint_bundle(
        root, stable_polls=1, stable_interval_seconds=0
    )
    data = tmp_path / "data"
    data.mkdir()
    source_path = data / "game1.npz"
    source_path.write_bytes(b"source")
    source = _source(source_path)
    config = _capture_config(tmp_path, bundle, policy_mode="sampled")
    preprocess = recorder.ReplayPreprocessSpec(4, (84, 84), False, False)
    task = recorder.SourceTask(
        source=source,
        episodes=(SimpleNamespace(local_episode_index=0),),
        preprocess=preprocess,
    )
    plan = recorder.build_capture_plan(
        config, [source], [recorder.WorkerSpec(0, "cpu")], [task]
    )
    episode_id = recorder.episode_store_id(source, 0)
    episode_path = Path(config.output_root) / "episodes" / f"{episode_id}.zarr"
    episode_path.mkdir(parents=True)
    metadata = {
        "episode_id": episode_id,
        "episode_seed": recorder.episode_seed(config.base_seed, source, 0),
        "seed_protocol": recorder.SEED_PROTOCOL,
        "source_id": source.source_id,
        "source_sha256": source.sha256,
        "source_size": source.size,
        "source_episode_index": 0,
        "checkpoint_bundle_sha256": bundle.bundle_sha256,
        "policy_mode": "sampled",
        "determinism_backend_fingerprint": dict(
            config.determinism_backend_fingerprint
        ),
        "timing_mode": "none",
        "trace_event_schema_version": recorder.TRACE_EVENT_SCHEMA_VERSION,
        "tree_carry": True,
        "images": "none",
        "chunk_events": 8,
        "frame_stack_n": 4,
        "target_size": [84, 84],
        "grayscale": False,
        "observation_dtype": "uint8",
        "source_logical_observation_count": 9,
        "source_logical_edge_count": 8,
        "replay_observation_count": 8,
        "replay_edge_count": 7,
        "scored_edge_count": 6,
        "logical_observation_count": 8,
        "logical_action_count": 7,
    }
    validation = {
        "event_count": 7,
        "files_sha256": "a" * 64,
        "images": "none",
        "metadata": metadata,
        "field_schema": [
            {"name": "trace_schema_version", "dtype": "<i2", "shape": []},
            {"name": "predicted_xs", "dtype": "<f4", "shape": [12, 84, 84]},
            {"name": "predicted_xs_valid", "dtype": "|b1", "shape": []},
            {"name": "step_times", "dtype": "<f4", "shape": [1]},
            {"name": "step_times_valid", "dtype": "|b1", "shape": []},
            {"name": "actor_real_logits_pre_override", "dtype": "<f4", "shape": [3]},
            {"name": "actor_real_logits_pre_override_valid", "dtype": "|b1", "shape": []},
            {"name": "model_done_logit", "dtype": "<f4", "shape": []},
            {"name": "model_done_logit_valid", "dtype": "|b1", "shape": []},
            {"name": "model_done_target_depth", "dtype": "<i2", "shape": []},
            {"name": "model_done_target_action", "dtype": "<i2", "shape": []},
        ],
    }
    native_fields = {
        "trace_schema_version": np.full(
            (7,), recorder.TRACE_EVENT_SCHEMA_VERSION, dtype=np.int16
        ),
        "step_times_valid": np.zeros((7,), dtype=np.bool_),
    }
    native = SimpleNamespace(field=lambda name: native_fields[name])
    trace = SimpleNamespace(
        validate_episode_store=lambda path: validation,
        open_video_stat=lambda path: SimpleNamespace(native=native),
    )
    recorder._validate_episode_semantics(
        metadata, validation, plan["semantic"], episode_path
    )
    with pytest.raises(recorder.VisualBehaviorError, match="episode_seed"):
        recorder._validate_episode_semantics(
            {**metadata, "episode_seed": metadata["episode_seed"] + 1},
            validation,
            plan["semantic"],
            episode_path,
        )
    with pytest.raises(recorder.VisualBehaviorError, match="seed_protocol"):
        recorder._validate_episode_semantics(
            {**metadata, "seed_protocol": "tampered"},
            validation,
            plan["semantic"],
            episode_path,
        )
    with pytest.raises(recorder.VisualBehaviorError, match="seed protocol"):
        recorder._validate_episode_semantics(
            metadata,
            validation,
            {**plan["semantic"], "seed_protocol": "tampered"},
            episode_path,
        )
    records = [
        {
            "kind": "source",
            "ok": True,
            "payload": {
                "episodes": [{"episode_id": episode_id, "path": str(episode_path)}]
            },
        }
    ]
    manifest = recorder._canonical_episode_manifest(
        trace,
        Path(config.output_root),
        records,
        expected_episodes={episode_id: source.sha256},
        plan=plan,
    )
    assert manifest["semantic"]["policy_mode"] == "sampled"
    recorder._publish_capture_completion(Path(config.output_root), manifest)
    assert json.loads(
        (Path(config.output_root) / "COMPLETED.json").read_text(encoding="utf-8")
    )["manifest_sha256"] == manifest["manifest_sha256"]
    validated = recorder.validate_capture_output(
        config.output_root,
        trace_api=trace,
        expected_semantic_sha256=plan["semantic_sha256"],
        expected_episodes={episode_id: source.sha256},
    )
    assert validated["episode_count"] == 1
    orphan = Path(config.output_root) / "episodes" / "orphan.zarr"
    orphan.mkdir()
    with pytest.raises(recorder.VisualBehaviorError, match="orphaned canonical"):
        recorder.validate_capture_output(config.output_root, trace_api=trace)
    orphan.rmdir()
    completion_path = Path(config.output_root) / "COMPLETED.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["manifest_sha256"] = "0" * 64
    completion_path.write_text(json.dumps(completion), encoding="utf-8")
    with pytest.raises(recorder.VisualBehaviorError, match="disagrees with manifest"):
        recorder.validate_capture_output(config.output_root, trace_api=trace)
    with pytest.raises(recorder.VisualBehaviorError, match="identity set mismatch"):
        recorder._canonical_episode_manifest(
            trace,
            Path(config.output_root),
            records,
            expected_episodes={"different": source.sha256},
            plan=plan,
        )


def test_existing_store_resume_requires_all_semantic_metadata(tmp_path):
    root, _ = _checkpoint_bundle(tmp_path)
    bundle = recorder.resolve_checkpoint_bundle(
        root, stable_polls=1, stable_interval_seconds=0
    )
    config = _capture_config(tmp_path, bundle, policy_mode="sampled")
    final = tmp_path / "existing.zarr"
    final.mkdir()
    expected = {
        "episode_id": "ep",
        "episode_seed": 1,
        "seed_protocol": recorder.SEED_PROTOCOL,
        "source_sha256": "b" * 64,
        "checkpoint_bundle_sha256": bundle.bundle_sha256,
        "policy_mode": "sampled",
        "determinism_backend_fingerprint": dict(
            config.determinism_backend_fingerprint
        ),
        "images": "none",
    }
    runtime = object.__new__(recorder._CaptureRuntime)
    runtime.config = config
    runtime.trace_api = SimpleNamespace(
        validate_episode_store=lambda path: {
            "metadata": dict(expected),
            "images": "none",
        }
    )
    assert runtime._existing_store(final, expected) is not None
    runtime.trace_api = SimpleNamespace(
        validate_episode_store=lambda path: {
            "metadata": {**expected, "policy_mode": "checkpoint"},
            "images": "none",
        }
    )
    with pytest.raises(recorder.VisualBehaviorError, match="policy_mode mismatch"):
        runtime._existing_store(final, expected)
    cuda_fingerprint = recorder.resolve_determinism_backend_fingerprint(
        [recorder.WorkerSpec(0, "cuda:0")],
        torch_module=SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                get_device_properties=lambda index: SimpleNamespace(
                    name="NVIDIA RTX 3090", major=8, minor=6
                ),
            )
        ),
    )
    runtime.trace_api = SimpleNamespace(
        validate_episode_store=lambda path: {
            "metadata": {
                **expected,
                "determinism_backend_fingerprint": cuda_fingerprint,
            },
            "images": "none",
        }
    )
    with pytest.raises(
        recorder.VisualBehaviorError,
        match="determinism_backend_fingerprint mismatch",
    ):
        runtime._existing_store(final, expected)


def test_episode_metadata_binds_backend_and_canonical_seed(tmp_path):
    root, _ = _checkpoint_bundle(tmp_path)
    bundle = recorder.resolve_checkpoint_bundle(
        root, stable_polls=1, stable_interval_seconds=0
    )
    source_path = tmp_path / "game1.npz"
    source_path.write_bytes(b"source")
    source = _source(source_path)
    runtime = object.__new__(recorder._CaptureRuntime)
    runtime.config = _capture_config(tmp_path, bundle, policy_mode="sampled")
    runtime.spec = SimpleNamespace(
        frame_stack_n=4,
        target_size=(84, 84),
        grayscale=False,
        observation_dtype="uint8",
    )
    runtime.flags = SimpleNamespace(voc_gate_policy_schema_version=None)
    runtime.actor_checkpoint = {}
    runtime.voc_ema_head_state_dict = None
    runtime.effective_think_cost = 0.0
    descriptor = SimpleNamespace(
        logical_observation_count=9,
        edge_count=8,
        scored_edge_count=6,
        local_episode_index=2,
        subject=1,
        session=4,
        block=1,
        game=1,
    )
    seed = recorder.episode_seed(runtime.config.base_seed, source, 2)
    metadata = runtime._episode_metadata(source, descriptor, seed, "episode")
    assert metadata["episode_seed"] == seed
    assert metadata["seed_protocol"] == recorder.SEED_PROTOCOL
    assert metadata["determinism_backend_fingerprint"] == (
        runtime.config.determinism_backend_fingerprint
    )


def test_stream_episode_forwards_sampled_policy_to_runner():
    calls = []

    class NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    class FakeRunner:
        def stream_rollout_episode(self, batch, **kwargs):
            calls.append((batch, kwargs))
            return {"native_event_count": 1}

    runtime = object.__new__(recorder._CaptureRuntime)
    runtime.config = SimpleNamespace(tree_carry=False, policy_mode="sampled")
    runtime.torch = SimpleNamespace(no_grad=NoGrad)
    runtime.runner = FakeRunner()
    runtime.voc_ema_head_state_dict = {"ema": np.asarray([1], dtype=np.float32)}
    runtime.effective_think_cost = 0.125
    batch = {"actions_seq": np.asarray([1], dtype=np.int64)}
    sink = object()
    assert runtime._stream_episode(batch, sink) == {"native_event_count": 1}
    assert calls == [
        (
            batch,
            {
                "event_sink": sink,
                "tree_carry": False,
                "policy_mode": "sampled",
                "voc_ema_head_state_dict": runtime.voc_ema_head_state_dict,
                "effective_think_cost": 0.125,
            },
        )
    ]


def test_worker_entry_preserves_sampled_capture_config(tmp_path, monkeypatch):
    import queue

    root, _ = _checkpoint_bundle(tmp_path)
    bundle = recorder.resolve_checkpoint_bundle(
        root, stable_polls=1, stable_interval_seconds=0
    )
    config = _capture_config(tmp_path, bundle, policy_mode="sampled")
    observed = []

    class FakeRuntime:
        def __init__(self, received_config, worker):
            observed.append((received_config.policy_mode, worker.device))
            self.config = received_config

        def process_source(self, task):
            return {
                "source_index": task.source.source_index,
                "policy_mode_seen_by_worker": self.config.policy_mode,
            }

        def close(self):
            observed.append(("closed", self.config.policy_mode))

    monkeypatch.setattr(recorder, "_CaptureRuntime", FakeRuntime)
    tasks = queue.Queue()
    results = queue.Queue()
    task = SimpleNamespace(source=SimpleNamespace(source_index=4, source_id="source"))
    tasks.put(task)
    tasks.put(None)
    recorder._worker_entry(
        config, recorder.WorkerSpec(rank=0, device="cpu"), tasks, results
    )
    records = [results.get_nowait(), results.get_nowait()]
    assert records[0]["payload"]["policy_mode_seen_by_worker"] == "sampled"
    assert records[1]["kind"] == "worker_done"
    assert observed == [("sampled", "cpu"), ("closed", "sampled")]


def test_resume_quarantines_stale_partials_and_recovers_valid_scratch_final(tmp_path):
    runtime = object.__new__(recorder._CaptureRuntime)
    runtime.worker = recorder.WorkerSpec(0, "cpu")
    runtime.config = SimpleNamespace(
        resume=True,
        images="none",
        output_root=str(tmp_path / "output"),
        scratch_root=str(tmp_path / "scratch"),
    )
    expected = {"episode_id": "episode", "images": "none"}
    validation = {
        "metadata": expected,
        "images": "none",
        "event_count": 4,
        "files_sha256": "a" * 64,
    }
    runtime.trace_api = SimpleNamespace(validate_episode_store=lambda path: validation)
    partial, local_final, canonical = runtime._local_paths("episode")
    old_partial = partial.with_name(f".{local_final.name}.partial-123")
    old_partial.mkdir(parents=True)
    moved = runtime._quarantine_stale_partials(local_final)
    assert len(moved) == 1 and moved[0].is_dir()
    assert not old_partial.exists()
    local_final.mkdir(parents=True)
    recovered = runtime._recover_local_final(local_final, canonical, expected)
    assert recovered == validation
    assert canonical.is_dir()
    assert not local_final.exists()


def test_scratch_publish_failure_preserves_local_final(tmp_path):
    runtime = object.__new__(recorder._CaptureRuntime)
    local_final = tmp_path / "scratch" / "episode.zarr"
    canonical = tmp_path / "output" / "episodes" / "episode.zarr"
    local_final.mkdir(parents=True)
    (local_final / "payload").write_bytes(b"diagnostic")

    def reject_transfer(path):
        raise ValueError("invalid packed store")

    runtime.trace_api = SimpleNamespace(validate_episode_store=reject_transfer)
    with pytest.raises(ValueError, match="invalid packed store"):
        runtime._publish(local_final, canonical)
    assert local_final.is_dir()
    assert not canonical.exists()
    assert not any(canonical.parent.glob(".*.transfer-partial-*"))


def test_resume_cleans_only_dead_exact_transfer_partials(tmp_path, monkeypatch):
    runtime = object.__new__(recorder._CaptureRuntime)
    runtime.config = SimpleNamespace(resume=True)
    canonical = tmp_path / "episodes" / "episode.zarr"
    canonical.parent.mkdir(parents=True)
    stale_transfer = canonical.with_name(
        f".{canonical.name}.transfer-partial-123"
    )
    writer_partial = canonical.with_name(f".{canonical.name}.partial-123")
    malformed_transfer = canonical.with_name(
        f".{canonical.name}.transfer-partial-not-a-pid"
    )
    for path in (stale_transfer, writer_partial, malformed_transfer):
        path.mkdir()
    monkeypatch.setattr(recorder, "_pid_is_alive", lambda pid: pid == 456)
    retired = runtime._cleanup_stale_transfers(canonical)
    assert retired == (stale_transfer,)
    assert not stale_transfer.exists()
    assert writer_partial.is_dir()
    assert malformed_transfer.is_dir()
    assert not any((canonical.parent / ".aborted-transfers").iterdir())

    interrupted_cleanup = (
        canonical.parent
        / ".aborted-transfers"
        / f".{canonical.name}.transfer-partial-999.123456"
    )
    interrupted_cleanup.mkdir()
    assert runtime._cleanup_stale_transfers(canonical) == ()
    assert not interrupted_cleanup.exists()

    active_transfer = canonical.with_name(
        f".{canonical.name}.transfer-partial-456"
    )
    active_transfer.mkdir()
    with pytest.raises(recorder.VisualBehaviorError, match="active episode transfer"):
        runtime._cleanup_stale_transfers(canonical)
    assert active_transfer.is_dir()
    assert writer_partial.is_dir()


def test_partial_worker_start_failure_terminates_started_child(monkeypatch):
    created = []
    queue_sizes = []

    class FakeProcess:
        def __init__(self, *args, **kwargs):
            self.name = kwargs["name"]
            self.exitcode = None
            self.alive = False
            self.terminated = False
            created.append(self)

        def start(self):
            if len(created) == 2:
                raise RuntimeError("spawn failed")
            self.alive = True

        def is_alive(self):
            return self.alive

        def join(self, timeout=None):
            return None

        def terminate(self):
            self.terminated = True
            self.alive = False
            self.exitcode = -15

    class FakeContext:
        Process = FakeProcess

        @staticmethod
        def Queue(maxsize=0):
            import queue

            queue_sizes.append(maxsize)
            return queue.Queue(maxsize=maxsize)

    monkeypatch.setattr(recorder.multiprocessing, "get_context", lambda name: FakeContext())
    with pytest.raises(RuntimeError, match="spawn failed"):
        recorder._run_workers(
            SimpleNamespace(),
            [],
            [recorder.WorkerSpec(0, "cuda:0"), recorder.WorkerSpec(1, "cuda:1")],
        )
    assert created[0].terminated is True
    assert queue_sizes == [4, 4]


def test_dry_run_validates_without_zarr_or_cuda(tmp_path, capsys):
    root, _ = _checkpoint_bundle(tmp_path)
    data = tmp_path / "data" / "sub-001" / "ses-04"
    data.mkdir(parents=True)
    (data / "block-01_game1.npz").write_bytes(b"enumeration-only")
    result = recorder.main(
        [
            "capture",
            "--checkpoint-dir",
            str(root),
            "--data-root",
            str(tmp_path / "data"),
            "--output",
            str(tmp_path / "out"),
            "--devices",
            "cpu",
            "--cpu-workers",
            "1",
            "--stable-polls",
            "1",
            "--stable-interval",
            "0",
            "--policy-mode",
            "sampled",
            "--dry-run",
        ]
    )
    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["checkpoint_payload_validation"]["game_id"] == 1
    assert payload["plan"]["semantic"]["source_count"] == 1
    assert payload["plan"]["semantic"]["policy_mode"] == "sampled"
    assert payload["plan"]["semantic"]["determinism_backend_fingerprint"] == (
        recorder.resolve_determinism_backend_fingerprint(
            [recorder.WorkerSpec(0, "cpu")]
        )
    )


def test_export_parser_accepts_projected_row_slice():
    args = recorder.build_parser().parse_args(
        ["export-legacy", "episode.zarr", "--output", "out.npy", "--start", "2", "--stop", "9"]
    )
    assert (args.start, args.stop) == (2, 9)


def test_capture_parser_defaults_to_four_torch_threads():
    args = recorder.build_parser().parse_args(
        [
            "capture",
            "--checkpoint-dir", "checkpoint",
            "--data-root", "data",
            "--output", "output",
        ]
    )
    assert args.torch_threads_per_worker == "4"
    assert args.policy_mode == "checkpoint"


def test_capture_parser_accepts_sampled_and_rejects_invalid_policy_mode():
    base = [
        "capture",
        "--checkpoint-dir", "checkpoint",
        "--data-root", "data",
        "--output", "output",
    ]
    args = recorder.build_parser().parse_args(base + ["--policy-mode", "sampled"])
    assert args.policy_mode == "sampled"
    with pytest.raises(SystemExit):
        recorder.build_parser().parse_args(base + ["--policy-mode", "invalid"])


def test_capture_config_rejects_invalid_policy_mode(tmp_path):
    root, _ = _checkpoint_bundle(tmp_path)
    bundle = recorder.resolve_checkpoint_bundle(
        root, stable_polls=1, stable_interval_seconds=0
    )
    with pytest.raises(recorder.VisualBehaviorError, match="policy_mode must be"):
        _capture_config(tmp_path, bundle, policy_mode="invalid")


def test_actual_writer_reader_roundtrip_when_zarr_is_installed(tmp_path):
    pytest.importorskip("zarr")
    pytest.importorskip("numcodecs")
    from thinker.behavior_trace import (
        EpisodeZarrWriter,
        LEGACY_KEYS,
        open_video_stat,
        validate_episode_store,
    )

    partial = tmp_path / ".episode.partial"
    final = tmp_path / "episode.zarr"
    output = recorder._BoundedWriter(
        SimpleNamespace(EpisodeZarrWriter=EpisodeZarrWriter),
        partial_path=partial,
        final_path=final,
        metadata={"episode_id": "roundtrip"},
        num_actions=3,
        tree_rep_schema={"all": [0, 4, None]},
        chunk_events=2,
        images="none",
        queue_size=1,
    )
    sink = recorder._CanonicalEventSink(
        output,
        images="none",
        source_frame_indices=np.asarray([10, 11, 12]),
        decision_times=np.asarray([1.0, 2.0, 3.0]),
        grayscale=False,
        observation_provider=np.zeros((3, 6, 2, 2), dtype=np.uint8),
    )
    sink.write_event(_event(0, initial=True, cursor_before=0, cursor_after=0))
    sink.write_event(_event(1, real=True, cursor_before=0, cursor_after=1))
    output.finalize()
    manifest = validate_episode_store(final)
    opened = open_video_stat(final)
    legacy = opened.legacy_view(materialize=True)
    assert manifest["event_count"] == 2
    assert set(legacy) == set(LEGACY_KEYS)
    assert legacy["status"].tolist() == [0, 0]
