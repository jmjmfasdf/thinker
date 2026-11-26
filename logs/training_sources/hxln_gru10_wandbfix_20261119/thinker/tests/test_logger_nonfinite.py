from types import SimpleNamespace
from types import MethodType

import math
import pytest

from thinker import logger as logger_module
from thinker import util
from thinker.logger import SLogWorker


def test_parse_line_preserves_nonfinite_diagnostics_instead_of_dropping_them():
    worker = SLogWorker.__new__(SLogWorker)
    worker.real_step = 12
    worker._logger = SimpleNamespace(error=lambda *_args, **_kwargs: None)

    row = worker.parse_line(
        ["tick", "loss", "positive", "negative", "finite"],
        "7,nan,inf,-inf,0.25",
    )

    assert row["tick"] == 7
    assert row["_tick"] == 7
    assert math.isnan(row["loss"])
    assert row["positive"] == float("inf")
    assert row["negative"] == float("-inf")
    assert row["finite"] == 0.25


def _logger_request(tmp_path):
    evidence = {
        "voc_actor_policy_terminal": True,
        "voc_actor_policy_version": 1,
        "voc_actor_policy_state_sha256": "a" * 64,
        "voc_actor_policy_publication_history_sha256": "b" * 64,
    }
    completion = {
        "checkpoint_files": {
            name: {"sha256": "c" * 64, "size": index + 1}
            for index, name in enumerate(util._COMPLETION_CHECKPOINT_FILES)
        }
    }
    return util.write_actor_policy_logger_finish_request(
        tmp_path, evidence, completion
    )


def _log_stat_worker(*, strict):
    worker = SLogWorker.__new__(SLogWorker)
    worker.voc_actor_policy_barrier_runtime = strict
    worker.log_actor = True
    worker.log_model = False
    worker.actor_fields = ["tick", "value"]
    worker.last_actor_tick = -1
    worker.video = None
    worker.real_step = 0
    worker.actor_log_path = "unused"
    worker._logger = SimpleNamespace(
        error=lambda *_args, **_kwargs: None,
        info=lambda *_args, **_kwargs: None,
    )
    return worker


def test_schema6_log_stat_propagates_parse_and_wandb_log_failures():
    worker = _log_stat_worker(strict=True)
    worker.read_stat = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("injected parse failure")
    )
    with pytest.raises(RuntimeError, match="parse failure"):
        worker.log_stat()

    worker = _log_stat_worker(strict=True)
    worker.read_stat = lambda *_args, **_kwargs: (
        {"_tick": 1, "real_step": 1, "value": 2.0},
        ["tick", "value"],
        1,
    )
    worker.wlogger = SimpleNamespace(
        wandb=SimpleNamespace(
            log=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("injected wandb.log failure")
            )
        )
    )
    with pytest.raises(RuntimeError, match="wandb.log failure"):
        worker.log_stat()


def test_legacy_log_stat_still_swallows_wandb_failure():
    worker = _log_stat_worker(strict=False)
    worker.read_stat = lambda *_args, **_kwargs: (
        {"_tick": 1, "real_step": 1, "value": 2.0},
        ["tick", "value"],
        1,
    )
    worker.wlogger = SimpleNamespace(
        wandb=SimpleNamespace(
            log=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("legacy upload failure")
            )
        )
    )
    assert worker.log_stat() is None


def test_log_stat_forwards_environment_noop_frequency_to_wandb():
    worker = _log_stat_worker(strict=False)
    worker.read_stat = lambda *_args, **_kwargs: (
        {
            "_tick": 1,
            "real_step": 32,
            "interaction/real_action_count": 16,
            "interaction/noop_count": 4,
            "interaction/noop_frequency": 0.25,
        },
        ["tick", "value"],
        1,
    )
    calls = []
    worker.wlogger = SimpleNamespace(
        wandb=SimpleNamespace(
            log=lambda payload, *, step: calls.append((payload, step))
        )
    )

    assert worker.log_stat() is True
    assert len(calls) == 1
    payload, step = calls[0]
    assert step == 32
    assert payload["interaction/real_action_count"] == 16
    assert payload["interaction/noop_count"] == 4
    assert payload["interaction/noop_frequency"] == pytest.approx(0.25)


def test_schema6_request_forces_complete_stat_and_unconditional_artifact_upload():
    worker = SLogWorker.__new__(SLogWorker)
    worker.voc_actor_policy_barrier_runtime = True
    worker.vis_policy = False
    worker.real_step = 10
    worker.last_real_step_c = 10
    worker.ckpdir = "/checkpoint"
    worker.flags = SimpleNamespace(wandb_ckp_freq=0, policy_vis_freq=-1)
    worker._logger = SimpleNamespace(info=lambda *_args, **_kwargs: None)
    calls = []
    worker.log_stat = lambda *, require_complete=False: calls.append(
        ("stat", require_complete)
    )
    worker.wlogger = SimpleNamespace(
        wandb=SimpleNamespace(
            save=lambda path: calls.append(("save", path))
        )
    )
    worker._run_schema6_log_iteration(
        force_artifact_upload=True, require_complete_stat=True
    )
    assert calls == [
        ("stat", True),
        ("save", "/checkpoint/*"),
    ]


def test_schema6_finish_failure_propagates_and_never_writes_ack(
    monkeypatch, tmp_path
):
    request = _logger_request(tmp_path)
    monkeypatch.setattr(
        util,
        "collect_run_completion_evidence",
        lambda _path: {"checkpoint_files": request["checkpoint_files"]},
    )
    worker = SLogWorker.__new__(SLogWorker)
    worker.voc_actor_policy_barrier_runtime = True
    worker.ckpdir = str(tmp_path)
    worker.real_step = 10
    worker._logger = SimpleNamespace(error=lambda *_args, **_kwargs: None)
    worker._run_schema6_log_iteration = lambda **_kwargs: None
    worker.close = lambda: (_ for _ in ()).throw(
        RuntimeError("injected wandb.finish failure")
    )
    with pytest.raises(RuntimeError, match="wandb.finish failure"):
        worker.start()
    assert util.read_actor_policy_logger_finish_ack(tmp_path, request) is None


def test_schema6_logger_keeps_periodic_cadence_then_runs_one_final_pass(
    monkeypatch, tmp_path
):
    worker = SLogWorker.__new__(SLogWorker)
    worker.voc_actor_policy_barrier_runtime = True
    worker.ckpdir = str(tmp_path)
    worker.real_step = 10
    worker.log_freq = 10
    worker._logger = SimpleNamespace(error=lambda *_args, **_kwargs: None)
    calls = []
    worker._run_schema6_log_iteration = MethodType(
        lambda self, **kwargs: calls.append(kwargs), worker
    )
    worker.close = lambda: calls.append({"close": True})
    sleeps = []

    def sleep_then_request(seconds):
        sleeps.append(seconds)
        request = _logger_request(tmp_path)
        monkeypatch.setattr(
            util,
            "collect_run_completion_evidence",
            lambda _path: {"checkpoint_files": request["checkpoint_files"]},
        )

    monkeypatch.setattr(logger_module.time, "sleep", sleep_then_request)
    assert worker.start() is True
    assert sleeps == [10]
    assert calls == [
        {},
        {"force_artifact_upload": True, "require_complete_stat": True},
        {"close": True},
    ]
    request = util.read_actor_policy_logger_finish_request(tmp_path)
    assert util.read_actor_policy_logger_finish_ack(tmp_path, request) is not None


def test_schema6_logger_rejects_triplet_change_after_final_upload(
    monkeypatch, tmp_path
):
    request = _logger_request(tmp_path)
    changed = {
        name: dict(value)
        for name, value in request["checkpoint_files"].items()
    }
    changed["ckp_actor.tar"]["sha256"] = "d" * 64
    evidence = iter((
        {"checkpoint_files": request["checkpoint_files"]},
        {"checkpoint_files": changed},
    ))
    monkeypatch.setattr(
        util, "collect_run_completion_evidence", lambda _path: next(evidence)
    )
    worker = SLogWorker.__new__(SLogWorker)
    worker.voc_actor_policy_barrier_runtime = True
    worker.ckpdir = str(tmp_path)
    worker.real_step = 10
    worker._logger = SimpleNamespace(error=lambda *_args, **_kwargs: None)
    worker._run_schema6_log_iteration = lambda **_kwargs: None
    worker.close = lambda: None
    with pytest.raises(RuntimeError, match="triplet changed"):
        worker.start()
    assert util.read_actor_policy_logger_finish_ack(tmp_path, request) is None
