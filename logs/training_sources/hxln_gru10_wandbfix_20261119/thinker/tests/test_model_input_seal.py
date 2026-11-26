from types import SimpleNamespace
import shutil
import tempfile

import numpy as np
import pytest
import ray
import torch

import thinker.learn_model as learn_model_module
import thinker.main as main_module
from thinker.buffer import (
    MODEL_BUFFER_ABORT,
    ModelBuffer,
    SCHEMA7_MODEL_BUFFER_STATUS_FIELDS,
    SModelBuffer,
    validate_schema7_model_buffer_status,
)
from thinker.learn_model import SModelLearner
from thinker.main import Env, _resolve_model_input_seal_runtime
from thinker.self_play import SelfPlayWorker


def _buffer_data(batch_size=1):
    return {
        "x": np.arange(batch_size, dtype=np.float32).reshape(batch_size, 1)
    }


def _sealed_status(*, finish=False, aborted=False, processed_n=1344):
    return {
        "processed_n": processed_n,
        "warm_up_n": 512,
        "replay_ratio": 5.0,
        "running": True,
        "finish": finish,
        "voc_model_input_seal_schema_version": 1,
        "voc_model_input_sealed": True,
        "voc_model_input_seal_count": 1,
        "voc_model_terminal_processed_n": processed_n,
        "voc_model_input_late_write_count": 0,
        "voc_model_input_abort_count": 1 if aborted else 0,
        "voc_model_input_aborted": aborted,
        "voc_model_update_claim_active": False,
    }


def _unsealed_status(*, processed_n=0):
    status = _sealed_status(processed_n=processed_n)
    status.update({
        "running": processed_n >= 512,
        "voc_model_input_sealed": False,
        "voc_model_input_seal_count": 0,
        "voc_model_terminal_processed_n": None,
    })
    return status


def test_schema7_model_buffer_requires_seal_before_success():
    buffer = SModelBuffer(
        buffer_n=16,
        max_rank=1,
        batch_size=4,
        warm_up_n=0,
        model_input_seal_schema_version=1,
        total_steps=4,
    )

    write_ack = buffer.write(_buffer_data(4), rank=0)
    assert write_ack == {"rank": 0, "processed_n": 4}
    claim = buffer.begin_model_update(4)
    assert set(claim) == {"allowed", "token", "status"}
    assert claim["allowed"] is True
    assert type(claim["token"]) is int
    assert claim["status"]["voc_model_update_claim_active"] is True
    with pytest.raises(ValueError, match="expected_min"):
        buffer.seal_input(0, True)
    seal = buffer.seal_input(0, 4)
    assert seal["voc_model_input_sealed"] is True
    assert seal["voc_model_input_seal_count"] == 1
    assert seal["voc_model_terminal_processed_n"] == 4
    assert seal["voc_model_update_claim_active"] is True
    assert seal["finish"] is False
    with pytest.raises(RuntimeError, match="sealed twice"):
        buffer.seal_input(0, 4)

    with pytest.raises(RuntimeError, match="sealed input"):
        buffer.complete_success(4)
    ended = buffer.end_model_update(claim["token"])
    assert ended == {"token": claim["token"], "status": ended["status"]}
    assert ended["status"]["voc_model_update_claim_active"] is False
    with pytest.raises(RuntimeError, match="stale"):
        buffer.end_model_update(claim["token"])
    denied = buffer.begin_model_update(4)
    assert denied["allowed"] is False
    assert denied["token"] is None

    completed = buffer.complete_success(4)
    assert completed["finish"] is True
    assert completed["voc_model_input_abort_count"] == 0
    with pytest.raises(RuntimeError, match="complete_success"):
        buffer.set_finish()


def test_schema7_model_buffer_late_write_and_abort_never_launder_success():
    late = SModelBuffer(
        buffer_n=16,
        max_rank=1,
        batch_size=4,
        model_input_seal_schema_version=1,
        total_steps=4,
    )
    late.write(_buffer_data(4), rank=0)
    late.seal_input(0, 4)
    with pytest.raises(RuntimeError, match="after producer closure"):
        late.write(_buffer_data(4), rank=0)
    assert late.get_status()["voc_model_input_late_write_count"] == 1
    with pytest.raises(RuntimeError, match="sealed input"):
        late.complete_success(4)
    assert late.get_status()["finish"] is False

    aborted = SModelBuffer(
        buffer_n=16,
        max_rank=1,
        batch_size=4,
        model_input_seal_schema_version=1,
        total_steps=4,
    )
    active = aborted.begin_model_update(0)
    assert active["allowed"] is True
    first = aborted.abort_input()
    second = aborted.abort_input()
    assert first["voc_model_input_abort_count"] == 1
    assert second["voc_model_input_abort_count"] == 1
    assert second["finish"] is False
    assert second["voc_model_update_claim_active"] is False
    assert aborted.read(1, 1) == MODEL_BUFFER_ABORT
    with pytest.raises(RuntimeError, match="aborted"):
        aborted.complete_success(4)


def test_legacy_model_buffer_status_and_finish_are_unchanged():
    buffer = SModelBuffer(
        buffer_n=8, max_rank=1, batch_size=1, warm_up_n=0
    )
    assert set(buffer.get_status()) == {
        "processed_n",
        "warm_up_n",
        "replay_ratio",
        "running",
        "finish",
    }
    assert buffer.write(_buffer_data(), rank=0) is None
    buffer.set_finish()
    assert buffer.get_status()["finish"] is True
    with pytest.raises(RuntimeError, match="inactive"):
        buffer.seal_input(0, 1)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("processed_n", True),
        ("warm_up_n", np.int64(512)),
        ("replay_ratio", 5),
        ("replay_ratio", float("nan")),
        ("running", 1),
        ("finish", np.bool_(False)),
        ("voc_model_input_seal_schema_version", True),
        ("voc_model_input_seal_count", 1.0),
        ("voc_model_terminal_processed_n", np.int64(1344)),
        ("voc_model_update_claim_active", 0),
    ],
)
def test_schema7_status_validator_rejects_noncanonical_types(field, value):
    status = _sealed_status()
    status[field] = value
    with pytest.raises(RuntimeError):
        validate_schema7_model_buffer_status(
            status,
            total_steps=1200,
            self_play_n=1,
            warm_up_n=512,
        )


def test_schema7_status_validator_requires_exact_keys_and_relations():
    status = _sealed_status()
    assert set(status) == SCHEMA7_MODEL_BUFFER_STATUS_FIELDS
    assert validate_schema7_model_buffer_status(
        status,
        total_steps=1200,
        self_play_n=1,
        warm_up_n=512,
        require_sealed=True,
    ) is status
    for malformed in (
        {**status, "extra": 1},
        {name: value for name, value in status.items() if name != "running"},
        {**status, "running": False},
        {**_unsealed_status(), "voc_model_input_seal_count": 1},
    ):
        with pytest.raises(RuntimeError):
            validate_schema7_model_buffer_status(
                malformed,
                total_steps=1200,
                self_play_n=1,
                warm_up_n=512,
            )


@pytest.mark.parametrize("gate_schema", [7, 8, 9, 10, 11, 12, 13])
def test_versioned_sealed_runtime_is_training_only_without_rewriting_identity(
    gate_schema,
):
    offline = SimpleNamespace(
        voc_gate_policy_schema_version=gate_schema,
        voc_model_input_seal_schema_version=1,
        train_model=False,
        parallel=False,
    )
    assert main_module._resolve_model_input_seal_schema_version(offline) == 1
    assert learn_model_module._resolve_model_input_seal_schema_version(offline) == 1
    assert _resolve_model_input_seal_runtime(offline) is False
    assert offline.voc_model_input_seal_schema_version == 1

    training = SimpleNamespace(**{**vars(offline), "train_model": True, "parallel": True})
    assert _resolve_model_input_seal_runtime(training) is True
    invalid = SimpleNamespace(**{**vars(training), "parallel": False})
    with pytest.raises(ValueError, match="parallel=true"):
        _resolve_model_input_seal_runtime(invalid)


@pytest.mark.parametrize(
    ("gate_schema", "seal_schema"),
    [
        (7, 0),
        (8, 0),
        (9, 0),
        (10, 0),
        (11, 0),
        (12, 0),
        (13, 0),
        (8, True),
        (9, True),
        (10, True),
        (11, True),
        (12, True),
        (13, True),
        (8, np.int64(1)),
        (9, np.int64(1)),
        (10, np.int64(1)),
        (11, np.int64(1)),
        (12, np.int64(1)),
        (13, np.int64(1)),
        (np.int64(8), 1),
        (np.int64(9), 1),
        (np.int64(10), 1),
        (np.int64(11), 1),
        (np.int64(12), 1),
        (np.int64(13), 1),
        (True, 1),
        (6, 1),
    ],
)
def test_versioned_model_input_seal_schema_rejects_cross_surface_drift(
    gate_schema, seal_schema
):
    flags = SimpleNamespace(
        voc_gate_policy_schema_version=gate_schema,
        voc_model_input_seal_schema_version=seal_schema,
    )
    for resolver in (
        main_module._resolve_model_input_seal_schema_version,
        learn_model_module._resolve_model_input_seal_schema_version,
    ):
        with pytest.raises(ValueError, match="must be exact integer"):
            resolver(flags)


def test_schema6_model_input_seal_runtime_remains_inactive():
    flags = SimpleNamespace(
        voc_gate_policy_schema_version=6,
        voc_model_input_seal_schema_version=0,
    )
    assert main_module._resolve_model_input_seal_schema_version(flags) == 0
    assert learn_model_module._resolve_model_input_seal_schema_version(flags) == 0
    assert _resolve_model_input_seal_runtime(flags) is False


@pytest.fixture
def isolated_ray():
    started_here = not ray.is_initialized()
    ray_temp_dir = None
    if started_here:
        ray_temp_dir = tempfile.mkdtemp(prefix="s7ray-", dir="/tmp")
        ray.init(
            num_cpus=2,
            include_dashboard=False,
            _temp_dir=ray_temp_dir,
            logging_level="ERROR",
        )
    try:
        yield
    finally:
        if started_here:
            ray.shutdown()
            shutil.rmtree(ray_temp_dir)


def test_real_ray_model_buffer_orders_write_seal_read_success_and_abort(
    isolated_ray,
):
    buffer = ModelBuffer.options(num_cpus=0).remote(
        buffer_n=8,
        max_rank=1,
        batch_size=1,
        warm_up_n=0,
        model_input_seal_schema_version=1,
        total_steps=1,
    )
    last_write_ref = buffer.write.remote(_buffer_data(), rank=0)
    assert ray.get(last_write_ref) == {"rank": 0, "processed_n": 1}
    claim = ray.get(buffer.begin_model_update.remote(1))
    assert claim["allowed"] is True
    seal = ray.get(buffer.seal_input.remote(0, 1))
    assert seal["voc_model_terminal_processed_n"] == 1
    assert seal["voc_model_update_claim_active"] is True
    ended = ray.get(buffer.end_model_update.remote(claim["token"]))
    assert ended["status"]["voc_model_update_claim_active"] is False
    fresh = ray.get(buffer.read.remote(1, 1, beta=0.0))
    assert fresh["voc_model_input_sealed"] is True
    assert fresh["voc_model_terminal_processed_n"] == fresh["processed_n"] == 1
    assert ray.get(buffer.complete_success.remote(1))["finish"] is True

    failed = ModelBuffer.options(num_cpus=0).remote(
        buffer_n=8,
        max_rank=1,
        batch_size=1,
        warm_up_n=0,
        model_input_seal_schema_version=1,
        total_steps=1,
    )
    aborted = ray.get(failed.abort_input.remote())
    assert aborted["voc_model_input_abort_count"] == 1
    assert aborted["finish"] is False
    assert ray.get(failed.read.remote(1, 1)) == MODEL_BUFFER_ABORT


def test_fake_terminal_worker_seals_before_wait_and_never_steps_environment():
    events = []

    class Env:
        def seal_model_input_no_step(self, *, timeout):
            assert timeout == 120.0
            events.append("seal")
            return _sealed_status(finish=False)

        def poll_model_status_no_step(self, *, timeout):
            assert 0.0 < timeout <= 120.0
            events.append("poll")
            return _sealed_status(finish=True)

        def step(self, *args, **kwargs):
            events.append("step")
            raise AssertionError("post-terminal environment action")

        def close(self):
            events.append("close")

    worker_class = SelfPlayWorker.__ray_metadata__.modified_class
    worker = worker_class.__new__(worker_class)
    worker.env = Env()
    worker.rank = 1
    worker.voc_model_input_seal_runtime = True
    worker.voc_actor_policy_barrier_timeout_s = 120.0
    worker.flags = SimpleNamespace(
        total_steps=1200, self_play_n=1, model_warm_up_n=512
    )
    worker._monotonic = lambda: 0.0
    worker._barrier_sleep = lambda _seconds: None
    worker._logger = SimpleNamespace(info=lambda *args: None)

    assert worker._complete_terminal_policy(
        {"model_status": {"finish": False}}
    ) is True
    assert events == ["seal", "poll", "close"]
    assert "step" not in events


def test_fake_terminal_worker_treats_model_abort_as_failure_not_finish():
    worker_class = SelfPlayWorker.__ray_metadata__.modified_class
    worker = worker_class.__new__(worker_class)
    worker.voc_model_input_seal_runtime = True
    worker.voc_actor_policy_barrier_timeout_s = 120.0
    worker.rank = 0
    worker.flags = SimpleNamespace(
        total_steps=1200, self_play_n=1, model_warm_up_n=512
    )
    worker._monotonic = lambda: 0.0
    worker._barrier_sleep = lambda _seconds: None
    worker.env = SimpleNamespace(
        poll_model_status_no_step=lambda **_kwargs: pytest.fail(
            "aborted initial status must fail before another poll"
        )
    )
    with pytest.raises(RuntimeError, match="aborted"):
        worker._wait_for_model_finish_without_env_actions(
            {"model_status": _sealed_status(aborted=True)}
        )


def test_terminal_actor_learner_resolves_before_model_input_seal():
    events = []

    class Env:
        def seal_model_input_no_step(self, *, timeout):
            events.append(("seal", timeout))
            return _sealed_status()

        def poll_model_status_no_step(self, *, timeout):
            events.append(("model-finish", timeout))
            return _sealed_status(finish=True)

        def close(self):
            events.append(("close", None))

    worker_class = SelfPlayWorker.__ray_metadata__.modified_class
    worker = worker_class.__new__(worker_class)
    worker.rank = 0
    worker.r_learner = "actor-learner-ref"
    worker.env = Env()
    worker.flags = SimpleNamespace(
        total_steps=1200, self_play_n=1, model_warm_up_n=512
    )
    worker.voc_model_input_seal_runtime = True
    worker.voc_actor_policy_barrier_timeout_s = 120.0
    worker._monotonic = lambda: 0.0
    worker._barrier_sleep = lambda _seconds: None
    worker._logger = SimpleNamespace(info=lambda *args: None)

    def barrier_get(ref, *, deadline, label):
        assert ref == "actor-learner-ref"
        assert deadline == 120.0
        events.append(("actor-learner-true", label))
        return True

    worker._barrier_ray_get = barrier_get
    assert worker._complete_terminal_policy(
        {"model_status": _unsealed_status(processed_n=1344)}
    ) is True
    names = [name for name, _ in events]
    assert names == ["actor-learner-true", "seal", "model-finish", "close"]


def test_env_terminal_status_poll_shares_one_monotonic_deadline(monkeypatch):
    env = Env.__new__(Env)
    env.train_model = True
    env.parallel = True
    env.voc_model_input_seal_runtime = True
    env.rank = 0
    env.flags = SimpleNamespace(
        total_steps=1200, self_play_n=1, model_warm_up_n=512
    )
    env.status_ptr = "initial-status"
    env.status = _unsealed_status(processed_n=1344)
    env.model_buffer = SimpleNamespace(
        get_status=_RemoteMethod("fresh-status", [])
    )
    poll_calls = []

    def poll_learner(*, wait=False, timeout=None):
        poll_calls.append((wait, timeout))
        return True

    env._poll_model_learner = poll_learner
    monotonic_values = iter([0.0, 10.0, 40.0, 70.0])
    monkeypatch.setattr(
        main_module.time, "monotonic", lambda: next(monotonic_values)
    )
    get_calls = []

    def fake_get(ref, *, timeout):
        get_calls.append((ref, timeout))
        if ref == "initial-status":
            return _unsealed_status(processed_n=1344)
        if ref == "fresh-status-ref":
            return _sealed_status(finish=True)
        raise AssertionError(ref)

    monkeypatch.setattr(main_module.ray, "get", fake_get)
    status = env.poll_model_status_no_step(timeout=120.0)
    assert status["finish"] is True
    assert get_calls == [
        ("initial-status", 110.0),
        ("fresh-status-ref", 80.0),
    ]
    assert poll_calls[-1] == (True, 50.0)


def test_env_terminal_poll_surfaces_model_learner_failure_before_status(
    monkeypatch,
):
    env = Env.__new__(Env)
    env.train_model = True
    env.parallel = True
    env.voc_model_input_seal_runtime = True
    env.rank = 0
    env.flags = SimpleNamespace(
        total_steps=1200, self_play_n=1, model_warm_up_n=512
    )
    env.status_ptr = "must-not-block"
    env.status = _sealed_status()

    def failed_learner(**_kwargs):
        raise RuntimeError("model learner failed")

    env._poll_model_learner = failed_learner
    monkeypatch.setattr(
        main_module.ray,
        "get",
        lambda *_args, **_kwargs: pytest.fail("status RPC must not be awaited"),
    )
    with pytest.raises(RuntimeError, match="model learner failed"):
        env.poll_model_status_no_step(timeout=120.0)


class _RemoteMethod:
    def __init__(self, name, events):
        self.name = name
        self.events = events

    def remote(self, *args):
        self.events.append((self.name, args))
        return f"{self.name}-ref"


@pytest.mark.parametrize("replay_ratio", [True, 5, float("nan"), float("inf"), -1.0])
def test_schema7_replay_header_rejects_noncanonical_ratio(replay_ratio):
    learner = SModelLearner.__new__(SModelLearner)
    data = {
        "processed_n": 848,
        "replay_ratio": replay_ratio,
        "voc_model_input_seal_schema_version": 1,
        "voc_model_input_sealed": False,
        "voc_model_terminal_processed_n": None,
    }
    with pytest.raises(RuntimeError, match="header"):
        learner._validate_schema7_replay_data_header(data)


def test_schema7_ray_get_uses_exact_timeout_and_translates_timeout(monkeypatch):
    learner = SModelLearner.__new__(SModelLearner)
    learner.voc_actor_policy_barrier_timeout_s = 120.0
    observed = []

    def timeout_get(ref, *, timeout):
        observed.append((ref, timeout))
        raise ray.exceptions.GetTimeoutError("delayed RPC")

    monkeypatch.setattr(learn_model_module.ray, "get", timeout_get)
    with pytest.raises(TimeoutError, match="fresh post-seal replay read"):
        learner._schema7_ray_get(
            "delayed-ref", label="fresh post-seal replay read"
        )
    assert observed == [("delayed-ref", 120.0)]


def _fake_schema7_learner(events, counts, *, gate_schema=7):
    learner = SModelLearner.__new__(SModelLearner)
    learner.flags = SimpleNamespace(
        total_steps=1200,
        self_play_n=1,
        model_warm_up_n=512,
        max_replay_ratio=5.0,
        ckpdir="/unused-schema7-test",
        voc_gate_policy_schema_version=gate_schema,
    )
    learner.voc_model_input_seal_runtime = True
    learner.voc_model_input_seal_schema_version = 1
    learner.voc_actor_policy_barrier_timeout_s = 120.0
    learner._schema7_terminal_drain_active = False
    learner.real_step = learner.last_psteps = 848
    learner.replay_ratio = 1.0
    learner.start_training = True
    learner.model_buffer = SimpleNamespace(
        get_status=_RemoteMethod("status", events),
        begin_model_update=_RemoteMethod("begin", events),
        end_model_update=_RemoteMethod("end", events),
        complete_success=_RemoteMethod("complete", events),
        abort_input=_RemoteMethod("abort", events),
    )
    learner.signal_buffer = SimpleNamespace(
        update_dict_item=_RemoteMethod("signal", events)
    )
    learner._logger = SimpleNamespace(
        info=lambda *args: events.append(("log", args)),
        error=lambda *args: events.append(("error", args)),
    )
    learner._gradient_clip_checkpoint_state = lambda: {
        "model_grad_clip_count_m": 0,
        "model_grad_step_count_m": counts["m"],
        "model_grad_clip_count_p": 0,
        "model_grad_step_count_p": counts["p"],
    }
    learner._publish_schema7_model_update = lambda: events.append(
        ("publish", ())
    )
    learner._schema7_checkpoint_iteration = lambda: events.append(
        ("periodic", ())
    )
    learner.save_checkpoint = lambda force=False, terminal=False: events.append(
        ("save", (force, terminal))
    )
    learner.close = lambda successful=True: events.append(
        ("close", successful)
    )
    return learner


def test_schema7_claim_linearizes_update_before_concurrent_seal(monkeypatch):
    events = []
    counts = {"m": 8, "p": 8}
    learner = _fake_schema7_learner(events, counts)
    learner.data_ptr = "compatibility-read"
    refs = iter(["ordinary-read", "next-prefetch"])
    learner.read_buffer_ptr = lambda: next(refs)
    ordinary_data = {
        "processed_n": 1344,
        "replay_ratio": 1.0,
        "voc_model_input_seal_schema_version": 1,
        "voc_model_input_sealed": False,
        "voc_model_terminal_processed_n": None,
    }
    unsealed = _unsealed_status(processed_n=1344)
    claimed = {**unsealed, "voc_model_update_claim_active": True}
    sealed = _sealed_status()

    def consume(data):
        assert data["voc_model_input_sealed"] is False
        events.append(("consume-ordinary", data["processed_n"]))
        learner.real_step = learner.last_psteps = data["processed_n"]
        counts["m"] += 1
        counts["p"] += 1
        return True

    learner.consume_data = consume

    def fake_get(ref, *args, **kwargs):
        assert kwargs.get("timeout") == 120.0
        if ref == "compatibility-read":
            events.append(("get", ref))
            return None
        if ref == "ordinary-read":
            events.append(("get", ref))
            return ordinary_data
        if ref == "status-ref":
            return unsealed
        if ref == "begin-ref":
            return {"allowed": True, "token": 1, "status": claimed}
        if ref == "end-ref":
            return {"token": 1, "status": sealed}
        if ref == "complete-ref":
            return _sealed_status(finish=True)
        if ref == "abort-ref":
            return _sealed_status(aborted=True)
        raise AssertionError(f"unexpected ref {ref!r}")

    monkeypatch.setattr(learn_model_module.ray, "get", fake_get)
    monkeypatch.setattr(
        learn_model_module.ray.internal,
        "free",
        lambda ref: events.append(("free", ref)),
    )
    monkeypatch.setattr(
        learn_model_module.util,
        "validate_schema7_final_bundle",
        lambda *_args, **_kwargs: {
            "model_real_step": learner.real_step,
            "model_input_seal": learner._schema7_model_input_checkpoint_evidence(
                require_terminal=True
            ),
        },
    )

    assert learner._learn_data_schema7() is True
    assert learner.voc_model_terminal_drain_update_count == 0
    assert learner.voc_model_terminal_drain_pre_grad_step_count_m == 9
    assert learner.voc_model_terminal_drain_pre_grad_step_count_p == 9
    assert sum(name == "consume-ordinary" for name, _ in events) == 1
    assert events.index(("get", "compatibility-read")) < events.index(
        ("get", "ordinary-read")
    )
    assert ("free", "compatibility-read") in events
    assert not any(name == "periodic" for name, _ in events)
    assert ("free", "next-prefetch") in events
    assert events[-1] == ("close", True)


@pytest.mark.parametrize("seal_at", ["read", "begin"])
def test_schema7_seal_attestation_denies_ordinary_update(
    monkeypatch, seal_at,
):
    events = []
    counts = {"m": 8, "p": 8}
    learner = _fake_schema7_learner(events, counts)
    refs = iter(["ordinary-read", "next-prefetch", "fresh-post-seal"])
    learner.read_buffer_ptr = lambda: next(refs)
    ordinary_data = {
        "processed_n": 1344 if seal_at == "read" else 848,
        "replay_ratio": 1.0,
        "voc_model_input_seal_schema_version": 1,
        "voc_model_input_sealed": seal_at == "read",
        "voc_model_terminal_processed_n": (
            1344 if seal_at == "read" else None
        ),
    }
    fresh_data = {
        "processed_n": 1344,
        "replay_ratio": 5.0,
        "voc_model_input_seal_schema_version": 1,
        "voc_model_input_sealed": True,
        "voc_model_terminal_processed_n": 1344,
    }

    def consume(data):
        assert data["voc_model_input_sealed"] is True
        events.append(("consume-drain", data["processed_n"]))
        learner.real_step = learner.last_psteps = data["processed_n"]
        counts["m"] += 1
        counts["p"] += 1
        return True

    learner.consume_data = consume

    def fake_get(ref, *args, **kwargs):
        assert kwargs.get("timeout") == 120.0
        if ref == "ordinary-read":
            return ordinary_data
        if ref == "status-ref":
            return (
                _sealed_status()
                if seal_at == "read"
                else _unsealed_status(processed_n=1344)
            )
        if ref == "begin-ref":
            assert seal_at == "begin"
            return {"allowed": False, "token": None, "status": _sealed_status()}
        if ref == "fresh-post-seal":
            return fresh_data
        if ref == "complete-ref":
            return _sealed_status(finish=True)
        if ref == "abort-ref":
            return _sealed_status(aborted=True)
        raise AssertionError(f"unexpected ref {ref!r}")

    monkeypatch.setattr(learn_model_module.ray, "get", fake_get)
    monkeypatch.setattr(learn_model_module.ray.internal, "free", lambda _ref: None)
    monkeypatch.setattr(
        learn_model_module.util,
        "validate_schema7_final_bundle",
        lambda *_args, **_kwargs: {
            "model_real_step": learner.real_step,
            "model_input_seal": learner._schema7_model_input_checkpoint_evidence(
                require_terminal=True
            ),
        },
    )

    assert learner._learn_data_schema7() is True
    assert learner.voc_model_terminal_drain_update_count == 1
    assert not any(name == "consume-ordinary" for name, _ in events)
    assert sum(name == "consume-drain" for name, _ in events) == 1
    assert sum(name == "begin" for name, _ in events) == (seal_at == "begin")


@pytest.mark.parametrize("gate_schema", [7, 8, 9, 10, 11, 12, 13])
@pytest.mark.parametrize(
    ("pre_real_step", "expected_drain"), [(848, 1), (1344, 0)]
)
def test_versioned_model_learner_schemas_share_terminal_drain(
    monkeypatch, gate_schema, pre_real_step, expected_drain
):
    events = []
    counts = {"m": 8, "p": 8}
    status = _sealed_status()
    completed = _sealed_status(finish=True)
    drain_data = {
        "processed_n": 1344,
        "replay_ratio": 5.0,
        "voc_model_input_seal_schema_version": 1,
        "voc_model_input_sealed": True,
        "voc_model_terminal_processed_n": 1344,
    }
    refs = iter(["stale-prefetch", "fresh-post-seal"])

    learner = SModelLearner.__new__(SModelLearner)
    learner.flags = SimpleNamespace(
        total_steps=1200,
        self_play_n=1,
        model_warm_up_n=512,
        max_replay_ratio=5.0,
        ckpdir="/unused-schema7-test",
        voc_gate_policy_schema_version=gate_schema,
    )
    learner.voc_model_input_seal_runtime = True
    learner.voc_model_input_seal_schema_version = 1
    learner.voc_actor_policy_barrier_timeout_s = 120.0
    learner.real_step = pre_real_step
    learner.last_psteps = pre_real_step
    learner.replay_ratio = 5.0
    learner.model_buffer = SimpleNamespace(
        get_status=_RemoteMethod("status", events),
        complete_success=_RemoteMethod("complete", events),
        abort_input=_RemoteMethod("abort", events),
    )
    learner.signal_buffer = SimpleNamespace(
        update_dict_item=_RemoteMethod("signal", events)
    )
    learner.read_buffer_ptr = lambda: next(refs)
    learner._logger = SimpleNamespace(
        info=lambda *args: events.append(("log", args)),
        error=lambda *args: events.append(("error", args)),
    )
    learner._gradient_clip_checkpoint_state = lambda: {
        "model_grad_clip_count_m": 0,
        "model_grad_step_count_m": counts["m"],
        "model_grad_clip_count_p": 0,
        "model_grad_step_count_p": counts["p"],
    }

    def consume(data):
        events.append(("consume", data["processed_n"]))
        learner.real_step = data["processed_n"]
        learner.last_psteps = data["processed_n"]
        counts["m"] += 1
        counts["p"] += 1
        return True

    learner.consume_data = consume
    learner._publish_schema7_model_update = lambda: events.append(
        ("publish", ())
    )

    def save_checkpoint(force=False, terminal=False):
        assert force is True
        assert terminal is True
        assert learner.voc_model_input_sealed is True
        assert learner.voc_model_terminal_processed_n == 1344
        events.append(("save", ()))

    learner.save_checkpoint = save_checkpoint
    learner.close = lambda successful=True: events.append(
        ("close", successful)
    )

    def fake_get(ref, *args, **kwargs):
        assert kwargs.get("timeout") == 120.0
        if ref == "status-ref":
            return status
        if ref == "fresh-post-seal":
            return drain_data
        if ref == "complete-ref":
            return completed
        if ref == "abort-ref":
            return _sealed_status(aborted=True)
        raise AssertionError(f"unexpected ref {ref!r}")

    monkeypatch.setattr(learn_model_module.ray, "get", fake_get)
    monkeypatch.setattr(
        learn_model_module.ray.internal,
        "free",
        lambda ref: events.append(("free", ref)),
    )
    monkeypatch.setattr(learn_model_module.time, "sleep", lambda _s: None)
    validator_name = f"validate_schema{gate_schema}_final_bundle"
    monkeypatch.setattr(
        learn_model_module.util,
        validator_name,
        lambda *_args, **_kwargs: {
            "model_real_step": learner.real_step,
            "model_input_seal": (
                learner._schema7_model_input_checkpoint_evidence(
                    require_terminal=True
                )
            ),
        },
    )
    for other_schema in ({7, 8, 9, 10, 11, 12, 13} - {gate_schema}):
        monkeypatch.setattr(
            learn_model_module.util,
            f"validate_schema{other_schema}_final_bundle",
            lambda *_args, **_kwargs: pytest.fail(
                f"schema {gate_schema} selected the wrong final-bundle validator"
            ),
        )

    assert learner._learn_data_schema7() is True
    assert learner.real_step == learner.last_psteps == 1344
    assert learner.voc_model_terminal_drain_update_count == expected_drain
    assert counts == {"m": 8 + expected_drain, "p": 8 + expected_drain}
    assert sum(name == "consume" for name, _ in events) == expected_drain
    assert sum(name == "save" for name, _ in events) == 1
    assert sum(name == "complete" for name, _ in events) == 1
    assert not any(name == "abort" for name, _ in events)
    assert any(
        name == "log"
        and args
        == (
            f"Terminating schema-{gate_schema} model-learning thread after "
            "sealed input",
        )
        for name, args in events
    )
    assert events.index(("save", ())) < next(
        index for index, event in enumerate(events) if event[0] == "complete"
    )
    assert events[-1] == ("close", True)


@pytest.mark.parametrize(
    ("fresh_payload", "error_match"),
    [
        (None, "fresh replay batch"),
        (
            {
                "processed_n": 1344,
                "replay_ratio": 5.0,
                "voc_model_input_seal_schema_version": 1,
                "voc_model_input_sealed": True,
                "voc_model_terminal_processed_n": np.int64(1344),
            },
            "header",
        ),
        (
            {
                "processed_n": 1344,
                "replay_ratio": float("nan"),
                "voc_model_input_seal_schema_version": 1,
                "voc_model_input_sealed": True,
                "voc_model_terminal_processed_n": 1344,
            },
            "header",
        ),
    ],
)
def test_schema7_malformed_fresh_terminal_drain_aborts_without_success(
    monkeypatch, fresh_payload, error_match
):
    events = []
    refs = iter(["stale-prefetch", "fresh-post-seal"])
    learner = SModelLearner.__new__(SModelLearner)
    learner.flags = SimpleNamespace(
        total_steps=1200,
        self_play_n=1,
        model_warm_up_n=512,
        max_replay_ratio=5.0,
    )
    learner.voc_model_input_seal_runtime = True
    learner.voc_actor_policy_barrier_timeout_s = 120.0
    learner.real_step = learner.last_psteps = 848
    learner.replay_ratio = 5.0
    learner.model_buffer = SimpleNamespace(
        get_status=_RemoteMethod("status", events),
        complete_success=_RemoteMethod("complete", events),
        abort_input=_RemoteMethod("abort", events),
    )
    learner.read_buffer_ptr = lambda: next(refs)
    learner._logger = SimpleNamespace(
        info=lambda *args: None, error=lambda *args: events.append(("error", args))
    )
    learner._gradient_clip_checkpoint_state = lambda: {
        "model_grad_clip_count_m": 0,
        "model_grad_step_count_m": 8,
        "model_grad_clip_count_p": 0,
        "model_grad_step_count_p": 8,
    }
    learner.close = lambda successful=True: events.append(
        ("close", successful)
    )
    learner.save_checkpoint = lambda force=False: events.append(("save", force))

    def fake_get(ref, *args, **kwargs):
        assert kwargs.get("timeout") == 120.0
        if ref == "status-ref":
            return _sealed_status()
        if ref == "fresh-post-seal":
            return fresh_payload
        if ref == "abort-ref":
            return _sealed_status(aborted=True)
        raise AssertionError(f"unexpected ref {ref!r}")

    monkeypatch.setattr(learn_model_module.ray, "get", fake_get)
    monkeypatch.setattr(learn_model_module.ray.internal, "free", lambda _ref: None)
    monkeypatch.setattr(learn_model_module.time, "sleep", lambda _s: None)

    with pytest.raises(RuntimeError, match=error_match):
        learner._learn_data_schema7()
    assert any(name == "abort" for name, _ in events)
    assert not any(name == "complete" for name, _ in events)
    assert not any(name == "save" for name, _ in events)
    assert events[-1] == ("close", False)


def test_schema7_model_checkpoint_persists_exact_terminal_evidence(
    tmp_path, monkeypatch
):
    learner = SModelLearner.__new__(SModelLearner)
    learner.ckp_path = str(tmp_path / "ckp_model.tar")
    learner.model_net = torch.nn.Linear(2, 2)
    parameters = list(learner.model_net.parameters())
    learner.optimizer_p = torch.optim.Adam(parameters[:1], lr=0.001)
    learner.optimizer_m = torch.optim.Adam(parameters[1:], lr=0.001)
    learner.scheduler_p = torch.optim.lr_scheduler.LambdaLR(
        learner.optimizer_p, lambda _epoch: 1.0
    )
    learner.scheduler_m = torch.optim.lr_scheduler.LambdaLR(
        learner.optimizer_m, lambda _epoch: 1.0
    )
    learner.scaler_p = learner.scaler_m = None
    learner.flags = SimpleNamespace(
        dual_net=True,
        checkpoint_interval=0,
        total_steps=1200,
        self_play_n=1,
        model_warm_up_n=512,
        ckpdir=str(tmp_path),
    )
    learner.step = 6048
    learner.real_step = 1344
    learner.voc_model_input_seal_runtime = True
    learner.voc_model_input_seal_schema_version = 1
    learner.voc_model_input_sealed = True
    learner.voc_model_input_seal_count = 1
    learner.voc_model_terminal_processed_n = 1344
    learner.voc_model_terminal_drain_update_count = 1
    learner.voc_model_terminal_drain_pre_real_step = 848
    learner.voc_model_terminal_drain_pre_grad_step_count_m = 8
    learner.voc_model_terminal_drain_pre_grad_step_count_p = 8
    learner.voc_model_input_late_write_count = 0
    learner.voc_model_input_abort_count = 0
    learner._logger = SimpleNamespace(info=lambda *args: None, error=lambda *args: None)
    for name, value in (
        ("model_grad_clip_count_m", 0),
        ("model_grad_step_count_m", 9),
        ("model_grad_clip_count_p", 0),
        ("model_grad_step_count_p", 9),
    ):
        setattr(learner, "_" + name, torch.tensor(value, dtype=torch.int64))

    fsync_calls = []
    real_fsync = learn_model_module.os.fsync

    def recording_fsync(fd):
        fsync_calls.append(fd)
        return real_fsync(fd)

    monkeypatch.setattr(learn_model_module.os, "fsync", recording_fsync)
    learner.save_checkpoint(force=True, terminal=True)
    checkpoint = torch.load(learner.ckp_path, weights_only=False)
    expected = {
        "voc_model_input_seal_schema_version": 1,
        "voc_model_input_sealed": True,
        "voc_model_input_seal_count": 1,
        "voc_model_terminal_processed_n": 1344,
        "voc_model_terminal_drain_update_count": 1,
        "voc_model_terminal_drain_pre_real_step": 848,
        "voc_model_terminal_drain_pre_grad_step_count_m": 8,
        "voc_model_terminal_drain_pre_grad_step_count_p": 8,
        "voc_model_input_late_write_count": 0,
        "voc_model_input_abort_count": 0,
    }
    assert {name: checkpoint[name] for name in expected} == expected
    assert len(fsync_calls) == 4
    assert (tmp_path / "ckp_model.tar_step_1344").is_file()


def _terminal_evidence_only_learner():
    learner = SModelLearner.__new__(SModelLearner)
    learner.flags = SimpleNamespace(total_steps=1200)
    learner.voc_model_input_seal_runtime = True
    learner.voc_model_input_seal_schema_version = 1
    learner.voc_model_input_sealed = True
    learner.voc_model_input_seal_count = 1
    learner.voc_model_terminal_processed_n = 1344
    learner.voc_model_terminal_drain_update_count = 1
    learner.voc_model_terminal_drain_pre_real_step = 848
    learner.voc_model_terminal_drain_pre_grad_step_count_m = 8
    learner.voc_model_terminal_drain_pre_grad_step_count_p = 8
    learner.voc_model_input_late_write_count = 0
    learner.voc_model_input_abort_count = 0
    learner.real_step = 1344
    learner._gradient_clip_checkpoint_state = lambda: {
        "model_grad_clip_count_m": 0,
        "model_grad_step_count_m": 9,
        "model_grad_clip_count_p": 0,
        "model_grad_step_count_p": 9,
    }
    return learner


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("voc_model_input_sealed", np.bool_(True)),
        ("voc_model_input_seal_count", True),
        ("voc_model_terminal_processed_n", np.int64(1344)),
        ("voc_model_terminal_drain_update_count", 1.0),
        ("voc_model_terminal_drain_pre_real_step", "848"),
        ("voc_model_input_late_write_count", False),
    ],
)
def test_schema7_checkpoint_evidence_never_coerces_types(field, value):
    learner = _terminal_evidence_only_learner()
    setattr(learner, field, value)
    with pytest.raises(RuntimeError, match="Python"):
        learner._schema7_model_input_checkpoint_evidence(
            require_terminal=True
        )


def test_schema7_preterminal_evidence_is_exact_and_mid_drain_save_is_deferred():
    learner = _terminal_evidence_only_learner()
    learner.voc_model_input_sealed = False
    learner.voc_model_input_seal_count = 0
    learner.voc_model_terminal_processed_n = -1
    learner.voc_model_terminal_drain_update_count = 0
    learner.voc_model_terminal_drain_pre_real_step = -1
    learner.voc_model_terminal_drain_pre_grad_step_count_m = -1
    learner.voc_model_terminal_drain_pre_grad_step_count_p = -1
    learner.real_step = 848
    learner._schema7_terminal_drain_active = False
    learner._logger = SimpleNamespace(info=lambda *args: None)
    evidence = learner._schema7_model_input_checkpoint_evidence(
        require_terminal=False
    )
    assert evidence["voc_model_input_sealed"] is False
    with pytest.raises(RuntimeError, match="preterminal"):
        learner._schema7_model_input_checkpoint_evidence(
            require_terminal=True
        )
    learner._schema7_terminal_drain_active = True
    assert learner.save_checkpoint(force=False, terminal=False) is False


@pytest.mark.parametrize("gate_schema", [7, 8, 9, 10, 11, 12, 13])
def test_versioned_authoritative_validation_precedes_complete_success(
    monkeypatch, gate_schema
):
    events = []
    counts = {"m": 8, "p": 8}
    learner = _fake_schema7_learner(
        events, counts, gate_schema=gate_schema
    )
    learner.real_step = learner.last_psteps = 1344
    learner.model_buffer = SimpleNamespace(
        complete_success=_RemoteMethod("complete", events)
    )
    learner.save_checkpoint = lambda force=False, terminal=False: events.append(
        ("save", (force, terminal))
    )
    validator_name = f"validate_schema{gate_schema}_final_bundle"
    monkeypatch.setattr(
        learn_model_module.util,
        validator_name,
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("authoritative validation failed")
        ),
    )
    for other_schema in ({7, 8, 9, 10, 11, 12, 13} - {gate_schema}):
        monkeypatch.setattr(
            learn_model_module.util,
            f"validate_schema{other_schema}_final_bundle",
            lambda *_args, **_kwargs: pytest.fail(
                f"schema {gate_schema} selected the wrong final-bundle validator"
            ),
        )

    with pytest.raises(RuntimeError, match="authoritative validation failed"):
        learner._complete_schema7_terminal_drain(_sealed_status(), None)
    assert events[0][0] == "log"
    assert ("save", (True, True)) in events
    assert not any(name == "complete" for name, _ in events)
