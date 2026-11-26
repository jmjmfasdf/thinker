import builtins
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from gymnasium import spaces

import evaluate_dynamic_imitation as evaluation
from thinker import util


def _spec(
    *,
    game_id=0,
    env_name="Enduro-v5",
    num_actions=9,
    observation_shape=(12, 1, 1),
    subjects=(1,),
    train_sessions=(1, 2, 3),
    holdout_sessions=(4,),
):
    return evaluation.EvaluationSpec(
        subjects=subjects,
        train_sessions=train_sessions,
        holdout_sessions=holdout_sessions,
        game_id=game_id,
        env_name=env_name,
        num_actions=num_actions,
        scored_length=4,
        frame_stack_n=4,
        grayscale=False,
        observation_shape=observation_shape,
        observation_dtype="uint8",
        target_size=observation_shape[-2:],
    )


def _flag_values(
    *,
    game_id=0,
    env_name="Enduro-v5",
    subjects="1",
    train_sessions="1,2,3",
    holdout_sessions="4",
):
    return {
        **evaluation.DYNAMIC_PROTOCOL,
        **evaluation.IMITATION_PROTOCOL,
        "icopro_game_id": game_id,
        "dynamic_search": True,
        "icopro_data_path": "/staged/behavioral_data_block",
        "icopro_subjects": subjects,
        "icopro_train_sessions": train_sessions,
        "icopro_holdout_sessions": holdout_sessions,
        "name": env_name,
        "grayscale": False,
        "envpool": True,
        "train_model": True,
        "discounting": 0.99,
        "float16": False,
        "actor_use_rms": False,
        "actor_adam_eps": 1e-8,
        "actor_learning_rate": 0.0003,
        "schedule_total_steps": 100,
        "total_steps": 100,
        "model_float16": False,
        "self_play_n": 1,
        "env_n": 16,
        "preload": "",
        "preload_actor": "",
        "voc_parent_checkpoint": "",
    }


def test_actor_observation_space_restores_online_vector_shapes():
    template = spaces.Dict({
        "real_states": spaces.Box(
            0, 255, shape=(12, 84, 84), dtype=np.uint8
        ),
        "xs": spaces.Box(0.0, 1.0, shape=(12, 84, 84), dtype=np.float32),
        "hs": spaces.Box(-np.inf, np.inf, shape=(3, 32, 6, 6), dtype=np.float32),
        "tree_reps": spaces.Box(-np.inf, np.inf, shape=(3, 104), dtype=np.float32),
    })

    actor_space = evaluation._vector_actor_observation_space(template, 3)

    assert actor_space["real_states"].shape == (3, 12, 84, 84)
    assert actor_space["xs"].shape == (3, 12, 84, 84)
    assert actor_space["hs"].shape == (3, 32, 6, 6)
    assert actor_space["tree_reps"].shape == (3, 104)
    # ActorBase removes the first dimension and therefore reconstructs the
    # same feature shapes used when the checkpoint was trained online.
    assert actor_space["real_states"].shape[1:] == (12, 84, 84)


def _batch(tmp_path: Path, spec=None):
    spec = _spec() if spec is None else spec
    source = (
        tmp_path
        / f"sub-{spec.subject:03d}"
        / f"ses-{spec.holdout_session:02d}"
        / (
            f"sub{spec.subject:03d}-ses{spec.holdout_session:02d}-"
            f"block1-game{spec.game_id}.npz"
        )
    )
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"behavior")
    return {
        "obs_seq": np.zeros(
            (2, 6) + spec.observation_shape, dtype=np.uint8
        ),
        "actions_seq": (
            np.asarray([[8, 0, 1, 2, 3], [7, 4, 5, 6, 0]], dtype=np.int64)
            % spec.num_actions
        ),
        "initial_prev_action": np.asarray(
            [6 % spec.num_actions, 5 % spec.num_actions], dtype=np.int64
        ),
        "rewards_seq": np.zeros((2, 5), dtype=np.float32),
        "done_seq": np.zeros((2, 5), dtype=np.bool_),
        "truncated_seq": np.zeros((2, 5), dtype=np.bool_),
        "score_mask": np.asarray([False, True, True, True, True]),
        "source_file": np.asarray([str(source), str(source)]),
        "subject": np.asarray([spec.subject, spec.subject]),
        "session": np.asarray(
            [spec.holdout_session, spec.holdout_session]
        ),
        "block": np.asarray([1, 1]),
        "game": np.asarray([spec.game_id, spec.game_id]),
        "episode_index": np.asarray([0, 0]),
        "window_start": np.asarray([1, 5]),
        "decision_times": np.arange(12, dtype=np.float64).reshape(2, 6),
        "observation_source_index": np.arange(12, dtype=np.int64).reshape(2, 6),
    }


def _result(batch, nll, roots, visits=None, expanded=None):
    targets = batch["actions_seq"][:, 1:]
    roots = np.asarray(roots, dtype=np.bool_)
    visits = (
        np.zeros_like(roots, dtype=np.int64)
        if visits is None
        else np.asarray(visits, dtype=np.int64)
    )
    expanded = (
        np.zeros_like(roots, dtype=np.int64)
        if expanded is None
        else np.asarray(expanded, dtype=np.int64)
    )
    return SimpleNamespace(
        per_stage_nll=np.asarray(nll, dtype=np.float64),
        root_carried=roots,
        carried_descendant_visit_count=visits,
        carried_descendant_expanded_count=expanded,
        useful_carry=roots & (visits > 0),
        argmax=targets.copy(),
        proposal=targets.copy(),
        executed=targets.copy(),
        burnin_executed=batch["actions_seq"][:, 0].copy(),
        count=targets.size,
    )


def test_paired_aggregation_uses_no_minus_carry_and_actual_carry_subset(tmp_path):
    spec = _spec()
    batch = _batch(tmp_path, spec)
    no_carry = _result(
        batch,
        [[2, 4, 6, 8], [1, 3, 5, 7]],
        np.zeros((2, 4), dtype=np.bool_),
    )
    carry = _result(
        batch,
        [[1, 5, 3, 8], [2, 1, 6, 4]],
        [[True, False, True, False], [False, True, True, False]],
        visits=[[2, 0, 1, 0], [0, 0, 3, 0]],
        expanded=[[1, 0, 1, 0], [0, 0, 1, 0]],
    )

    rows = evaluation.build_paired_rows(batch, no_carry, carry, spec)
    summary = evaluation.summarize_rows(
        rows,
        {
            "n_windows": 2,
            "unique_scored_targets": 8,
            "eligible_scored_targets": 8,
            "skipped_tail_targets": 0,
        },
        stride=4,
        spec=spec,
    )

    assert len(rows) == 8
    assert summary["schema_version"] == evaluation.EVALUATION_SCHEMA_VERSION
    assert rows[0]["delta_nll"] == pytest.approx(1.0)
    assert summary["overall"]["nll_no_carry"] == pytest.approx(4.5)
    assert summary["overall"]["nll_carry"] == pytest.approx(3.75)
    assert summary["overall"]["delta_nll_no_minus_carry"] == pytest.approx(0.75)
    assert summary["root_carried_true"]["count"] == 4
    assert summary["root_carried_true"]["delta_nll_no_minus_carry"] == pytest.approx(1.25)
    assert summary["root_carry_coverage"]["fraction"] == pytest.approx(0.5)
    assert summary["useful_carry_true"]["count"] == 3
    assert summary["useful_carry_true"][
        "delta_nll_no_minus_carry"
    ] == pytest.approx(1.0)
    assert summary["useful_carry_coverage"] == {
        "count": 3,
        "total_scored_rows": 8,
        "fraction": pytest.approx(3 / 8),
        "support_status": "has_support",
        "descendant_visit_count_sum": 6,
        "descendant_expanded_count_sum": 3,
    }
    assert rows[0]["carried_descendant_visit_count_carry"] == 2
    assert rows[0]["carried_descendant_expanded_count_carry"] == 1
    assert rows[0]["useful_carry_carry"] is True


def test_useful_carry_summary_reports_no_support_for_leaf_promotions(tmp_path):
    spec = _spec()
    batch = _batch(tmp_path, spec)
    no_carry = _result(
        batch,
        np.ones((2, 4)),
        np.zeros((2, 4), dtype=np.bool_),
    )
    carry = _result(
        batch,
        np.ones((2, 4)),
        np.ones((2, 4), dtype=np.bool_),
    )

    rows = evaluation.build_paired_rows(batch, no_carry, carry, spec)
    summary = evaluation.summarize_rows(
        rows,
        {
            "n_windows": 2,
            "unique_scored_targets": 8,
            "eligible_scored_targets": 8,
            "skipped_tail_targets": 0,
        },
        stride=4,
        spec=spec,
    )

    assert summary["root_carry_coverage"]["count"] == 8
    assert summary["useful_carry_true"]["count"] == 0
    assert summary["useful_carry_true"]["nll_carry"] is None
    assert summary["useful_carry_coverage"]["support_status"] == "no_support"
    assert summary["useful_carry_coverage"]["count"] == 0


def test_holdout_validation_rejects_training_session_and_scored_burnin(tmp_path):
    spec = _spec()
    batch = _batch(tmp_path, spec)
    batch["session"] = np.asarray([3, 4])
    with pytest.raises(ValueError, match="session"):
        evaluation.validate_holdout_batch(batch, spec)

    batch = _batch(tmp_path, spec)
    batch["score_mask"] = np.ones(5, dtype=np.bool_)
    with pytest.raises(ValueError, match="score_mask"):
        evaluation.validate_holdout_batch(batch, spec)


def test_manifest_hashes_checkpoint_sources_and_results(tmp_path):
    import thinker.cenv  # noqa: F401 - completion marker records the loaded binary
    from thinker import util

    spec = _spec()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for name, content in {
        "config_c.yaml": b"dynamic_search: true\n",
        "ckp_actor.tar": b"actor",
        "ckp_model.tar": b"model",
    }.items():
        (checkpoint / name).write_bytes(content)
    util.write_run_completion(checkpoint)
    source = tmp_path / "sub-001" / "ses-04" / "data-game0.npz"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source")
    csv_path = tmp_path / "paired_steps.csv"
    summary_path = tmp_path / "summary.json"
    csv_path.write_text("delta_nll\n1.0\n", encoding="utf-8")
    summary_path.write_text("{}\n", encoding="utf-8")

    manifest = evaluation.make_manifest(
        checkpoint_dir=checkpoint,
        source_files=[source],
        output_files=[csv_path, summary_path],
        args={"device": "cpu", "session": 4},
        spec=spec,
    )

    assert manifest["checkpoint"]["files"]["ckp_actor.tar"]["sha256"] == (
        evaluation.sha256_file(checkpoint / "ckp_actor.tar")
    )
    assert manifest["behavioral_sources"][0]["sha256"] == evaluation.sha256_file(source)
    assert manifest["outputs"]["summary.json"]["sha256"] == evaluation.sha256_file(
        summary_path
    )
    assert "p_value" not in str(manifest)


def test_schema13_manifest_uses_real_completion_v2_and_exact_bound_hash_set(
    monkeypatch, tmp_path
):
    import thinker.cenv as cenv

    spec = _spec()
    checkpoint = tmp_path / "schema13-checkpoint"
    checkpoint.mkdir()
    for name in evaluation.SCHEMA13_BOUND_RUN_FILES:
        if name != "finish":
            (checkpoint / name).write_bytes(f"schema13:{name}\n".encode("utf-8"))
    package_root = Path(evaluation.__file__).resolve().parent
    implementation_sources = {
        relative: {
            "sha256": evaluation.sha256_file(package_root / relative),
        }
        for relative in evaluation.SCHEMA13_IMPLEMENTATION_SOURCES
    }
    extension_path = Path(cenv.__file__).resolve()
    extension_relative = str(extension_path.relative_to(package_root))
    marker = {
        "schema_version": 2,
        "status": "complete",
        "completed_unix": 1.0,
        "checkpoint_files": {
            name: {
                "sha256": evaluation.sha256_file(checkpoint / name),
                "size": (checkpoint / name).stat().st_size,
            }
            for name in evaluation.SCHEMA13_COMPLETION_CHECKPOINT_FILES
        },
        "implementation_sources": implementation_sources,
        "loaded_extensions": {
            extension_relative: {
                "sha256": evaluation.sha256_file(extension_path),
            }
        },
        "voc_actor_policy_logger_completion": {},
    }
    (checkpoint / "finish").write_text(
        json.dumps(marker, sort_keys=True), encoding="utf-8"
    )
    expected_hashes = evaluation._schema13_checkpoint_hashes(checkpoint)
    source = tmp_path / "source.npz"
    output = tmp_path / "summary.json"
    source.write_bytes(b"source")
    output.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        evaluation,
        "validate_completion_marker",
        lambda path: (_ for _ in ()).throw(
            AssertionError("schema-13 manifest used schema-1 completion")
        ),
    )

    manifest = evaluation.make_manifest(
        checkpoint_dir=checkpoint,
        source_files=[source],
        output_files=[output],
        args={"device": "cpu", "session": 4},
        spec=spec,
        flags=SimpleNamespace(voc_gate_policy_schema_version=13),
        expected_checkpoint_hashes=expected_hashes,
    )

    assert manifest["checkpoint"]["completion_marker"]["schema_version"] == 2
    assert set(manifest["checkpoint"]["files"]) == set(
        evaluation.SCHEMA13_BOUND_RUN_FILES
    )
    assert {
        name: record["sha256"]
        for name, record in manifest["checkpoint"]["files"].items()
    } == expected_hashes


def test_checkpoint_bundle_requires_success_marker(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for name in ("config_c.yaml", "ckp_actor.tar", "ckp_model.tar"):
        (checkpoint / name).write_bytes(b"placeholder")

    with pytest.raises(FileNotFoundError, match="finish"):
        evaluation.checkpoint_hashes(checkpoint)


def _write_public_completion_bundle(tmp_path):
    import thinker.cenv  # noqa: F401 - finish binds the loaded binary
    from thinker import util

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for name in ("config_c.yaml", "ckp_actor.tar", "ckp_model.tar"):
        (checkpoint / name).write_bytes(name.encode("utf-8"))
    util.write_run_completion(checkpoint)
    return checkpoint


def test_completion_marker_rejects_hardlinked_finish(tmp_path):
    checkpoint = _write_public_completion_bundle(tmp_path)
    os.link(checkpoint / "finish", checkpoint / "finish.alias")

    with pytest.raises(ValueError, match="regular single-link"):
        evaluation.validate_completion_marker(checkpoint)


def test_completion_marker_rejects_symlinked_finish(tmp_path):
    checkpoint = _write_public_completion_bundle(tmp_path)
    marker = checkpoint / "finish"
    backing = checkpoint / "finish.backing"
    marker.rename(backing)
    marker.symlink_to(backing.name)

    with pytest.raises(ValueError, match="regular single-link"):
        evaluation.validate_completion_marker(checkpoint)


def test_completion_marker_rejects_duplicate_json_names(tmp_path):
    checkpoint = _write_public_completion_bundle(tmp_path)
    marker = checkpoint / "finish"
    original = marker.read_text(encoding="utf-8")
    marker.write_text(
        original.replace("{", '{"status":"junk",', 1), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="duplicate JSON key 'status'"):
        evaluation.validate_completion_marker(checkpoint)


def test_completion_marker_rejects_identity_change_during_read(
    monkeypatch, tmp_path
):
    checkpoint = _write_public_completion_bundle(tmp_path)
    marker = checkpoint / "finish"
    original_lstat = os.lstat
    marker_calls = 0

    class ChangedIdentity:
        def __init__(self, value):
            self._value = value
            self.st_ctime_ns = value.st_ctime_ns + 1

        def __getattr__(self, name):
            return getattr(self._value, name)

    def racing_lstat(path):
        nonlocal marker_calls
        value = original_lstat(path)
        if Path(path) == marker:
            marker_calls += 1
            if marker_calls >= 2:
                return ChangedIdentity(value)
        return value

    monkeypatch.setattr(evaluation.os, "lstat", racing_lstat)
    with pytest.raises(RuntimeError, match="changed during read"):
        evaluation.validate_completion_marker(checkpoint)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("schema_bool", "completed v1 bundle"),
        ("status_nonstring", "completed v1 bundle"),
        ("size_float", "invalid size"),
        ("checkpoint_record_extra", "invalid fields"),
        ("checkpoint_set_extra", "invalid checkpoint file hashes"),
        ("source_record_extra", "invalid fields"),
        ("extension_digest_bool", "invalid sha256"),
    ),
)
def test_completion_marker_rejects_type_and_key_drift(
    tmp_path, mutation, message
):
    checkpoint = _write_public_completion_bundle(tmp_path)
    marker_path = checkpoint / "finish"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if mutation == "schema_bool":
        marker["schema_version"] = True
    elif mutation == "status_nonstring":
        marker["status"] = 1
    elif mutation == "size_float":
        record = marker["checkpoint_files"]["ckp_actor.tar"]
        record["size"] = float(record["size"])
    elif mutation == "checkpoint_record_extra":
        marker["checkpoint_files"]["ckp_actor.tar"]["extra"] = False
    elif mutation == "checkpoint_set_extra":
        marker["checkpoint_files"]["extra.tar"] = {
            "sha256": "0" * 64,
            "size": 1,
        }
    elif mutation == "source_record_extra":
        record = next(iter(marker["implementation_sources"].values()))
        record["extra"] = False
    elif mutation == "extension_digest_bool":
        record = next(iter(marker["loaded_extensions"].values()))
        record["sha256"] = True
    marker_path.write_text(
        json.dumps(marker, sort_keys=True), encoding="utf-8"
    )

    with pytest.raises(ValueError, match=message):
        evaluation.validate_completion_marker(checkpoint)


def test_completion_marker_is_bound_to_final_checkpoint_bytes(tmp_path):
    import thinker.cenv  # noqa: F401 - completion marker records the loaded binary
    from thinker import util

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for name in ("config_c.yaml", "ckp_actor.tar", "ckp_model.tar"):
        (checkpoint / name).write_bytes(name.encode("utf-8"))
    util.write_run_completion(checkpoint)
    hashes = evaluation.checkpoint_hashes(checkpoint)
    assert set(hashes) == {
        "config_c.yaml",
        "ckp_actor.tar",
        "ckp_model.tar",
        "finish",
    }

    (checkpoint / "ckp_actor.tar").write_bytes(b"later actor")
    with pytest.raises(ValueError, match="does not match final ckp_actor"):
        evaluation.checkpoint_hashes(checkpoint)


def test_completion_marker_is_bound_to_loaded_cython_binary(tmp_path):
    import thinker.cenv  # noqa: F401 - completion marker records the loaded binary
    from thinker import util

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for name in ("config_c.yaml", "ckp_actor.tar", "ckp_model.tar"):
        (checkpoint / name).write_bytes(name.encode("utf-8"))
    util.write_run_completion(checkpoint)
    marker_path = checkpoint / "finish"
    marker = evaluation.json.loads(marker_path.read_text(encoding="utf-8"))
    extension = next(iter(marker["loaded_extensions"]))
    marker["loaded_extensions"][extension]["sha256"] = "0" * 64
    marker_path.write_text(
        evaluation.json.dumps(marker, sort_keys=True), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Cython extension differs"):
        evaluation.checkpoint_hashes(checkpoint)


def test_starting_a_run_invalidates_an_old_completion_marker(tmp_path):
    from thinker import util

    marker = tmp_path / "finish"
    marker.write_text("stale", encoding="utf-8")
    util.clear_run_completion(tmp_path)
    assert not marker.exists()


def test_checkpoint_protocol_rejects_non_imitation_configuration(monkeypatch, tmp_path):
    from thinker import util

    values = _flag_values()
    flags = SimpleNamespace(**values)
    monkeypatch.setattr(util, "create_flags", lambda *args, **kwargs: flags)
    assert evaluation._load_flags(tmp_path).icopro_game_id == 0

    flags.icopro_data_path = ""
    with pytest.raises(ValueError, match="not an imitation-trained checkpoint"):
        evaluation._load_flags(tmp_path)


def test_actor_checkpoint_must_prove_completed_imitation_updates():
    values = _flag_values()
    flags = SimpleNamespace(**values)
    spec = _spec()
    checkpoint = {
        "step": 128,
        "real_step": 64,
        "imitation_update_count": 7,
        "imitation_schedule_step": 7,
        "imitation_data_signature": "a" * 64,
        "action_prior_ema": np.ones(9, dtype=np.float32),
        "flags": dict(values),
    }

    state = evaluation.validate_actor_imitation_checkpoint(checkpoint, flags, spec)
    assert state["imitation_update_count"] == 7
    assert state["embedded_protocol_verified"] is True
    assert state["voc"]["dynamic_voc_mode"] == "off"
    assert state["voc"]["legacy_voc_metadata_defaulted"] is True

    checkpoint["imitation_update_count"] = 0
    with pytest.raises(ValueError, match="no completed Dynamic imitation updates"):
        evaluation.validate_actor_imitation_checkpoint(checkpoint, flags, spec)


def test_actor_training_data_signature_must_match_recomputed_files():
    checkpoint = {"imitation_data_signature": "a" * 64}
    state = evaluation.verify_actor_behavioral_data_signature(
        checkpoint, "a" * 64
    )
    assert state["training_data_signature_recomputed"] is True

    with pytest.raises(ValueError, match="selected training sessions"):
        evaluation.verify_actor_behavioral_data_signature(checkpoint, "b" * 64)


class _LiveEnv:
    def __init__(
        self, num_actions, observation_shape=(12, 84, 84), action_start=0
    ):
        self.single_action_space = spaces.Discrete(
            num_actions, start=action_start
        )
        self.single_observation_space = spaces.Box(
            0, 255, shape=observation_shape, dtype=np.uint8
        )
        self.closed = False
        self.reset_count = 0
        self.step_actions = []

    def reset(self):
        self.reset_count += 1
        observation = np.zeros(
            (1,) + self.single_observation_space.shape,
            dtype=self.single_observation_space.dtype,
        )
        return observation, {}

    def step(self, action):
        action = np.asarray(action)
        self.step_actions.append(action.copy())
        observation = np.zeros(
            (1,) + self.single_observation_space.shape,
            dtype=self.single_observation_space.dtype,
        )
        return (
            observation,
            np.zeros(1, dtype=np.float32),
            np.zeros(1, dtype=np.bool_),
            np.zeros(1, dtype=np.bool_),
            {},
        )

    def close(self):
        self.closed = True


@pytest.mark.parametrize(
    ("game_id", "env_name", "num_actions"),
    [(0, "Enduro-v5", 9), (1, "Pong-v5", 6)],
)
def test_runtime_spec_uses_live_environment_action_dimension(
    monkeypatch, game_id, env_name, num_actions
):
    live = _LiveEnv(num_actions)
    monkeypatch.setattr(evaluation, "_make_live_environment", lambda flags: live)
    flags = SimpleNamespace(**_flag_values(game_id=game_id, env_name=env_name))

    spec = evaluation.resolve_evaluation_spec(
        flags, expected_env_name=env_name, expected_game_id=game_id
    )

    assert spec.game_id == game_id
    assert spec.env_name == env_name
    assert spec.num_actions == num_actions
    assert spec.observation_shape == (12, 84, 84)
    assert live.reset_count == 1
    assert len(live.step_actions) == 1
    assert live.step_actions[0].tolist() == [0]
    assert live.closed is True


def test_runtime_spec_assertions_never_override_checkpoint(monkeypatch):
    monkeypatch.setattr(
        evaluation, "_make_live_environment", lambda flags: _LiveEnv(6)
    )
    flags = SimpleNamespace(**_flag_values(game_id=1, env_name="Pong-v5"))

    with pytest.raises(ValueError, match="expected game id 0"):
        evaluation.resolve_evaluation_spec(flags, expected_game_id=0)
    with pytest.raises(ValueError, match="expected environment 'Enduro-v5'"):
        evaluation.resolve_evaluation_spec(
            flags, expected_env_name="Enduro-v5"
        )


def test_cli_expected_identity_flags_are_assertion_inputs():
    args = evaluation.parse_args(
        [
            "--checkpoint-dir",
            "/checkpoint",
            "--expected-env-name",
            "Pong-v5",
            "--expected-game-id",
            "1",
        ]
    )
    assert args.expected_env_name == "Pong-v5"
    assert args.expected_game_id == 1
    assert args.stride is None


def test_runtime_spec_supports_arbitrary_checkpoint_game_and_action_dimension(
    monkeypatch,
):
    monkeypatch.setattr(
        evaluation, "_make_live_environment", lambda flags: _LiveEnv(5)
    )
    flags = SimpleNamespace(
        **_flag_values(
            game_id=73,
            env_name="Synthetic-v0",
            subjects="7",
            train_sessions="2,5",
            holdout_sessions="9",
        )
    )

    spec = evaluation.resolve_evaluation_spec(flags)

    assert spec.subjects == (7,)
    assert spec.train_sessions == (2, 5)
    assert spec.holdout_sessions == (9,)
    assert spec.game_id == 73
    assert spec.env_name == "Synthetic-v0"
    assert spec.num_actions == 5


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"subjects": "7,8"}, "exactly one checkpoint subject"),
        ({"train_sessions": ""}, "at least one session"),
        ({"holdout_sessions": "9,10"}, "exactly one checkpoint holdout"),
        (
            {"train_sessions": "2,9", "holdout_sessions": "9"},
            "sessions overlap",
        ),
    ],
)
def test_runtime_spec_rejects_ambiguous_or_leaking_checkpoint_splits(
    monkeypatch, overrides, message
):
    monkeypatch.setattr(
        evaluation,
        "_make_live_environment",
        lambda flags: pytest.fail("invalid identity must fail before env creation"),
    )
    values = {
        "subjects": "7",
        "train_sessions": "2,5",
        "holdout_sessions": "9",
        **overrides,
    }
    flags = SimpleNamespace(
        **_flag_values(
            game_id=73,
            env_name="Synthetic-v0",
            **values,
        )
    )

    with pytest.raises(ValueError, match=message):
        evaluation.resolve_evaluation_spec(flags)


def test_runtime_spec_rejects_nonzero_based_live_action_space(monkeypatch):
    live = _LiveEnv(5, action_start=1)
    monkeypatch.setattr(evaluation, "_make_live_environment", lambda flags: live)
    flags = SimpleNamespace(**_flag_values(game_id=73, env_name="Synthetic-v0"))

    with pytest.raises(ValueError, match="zero-based"):
        evaluation.resolve_evaluation_spec(flags)

    assert live.reset_count == 0
    assert live.closed is True


def test_runtime_spec_rejects_bad_live_step_observation(monkeypatch):
    live = _LiveEnv(5)

    def bad_step(action):
        return (
            np.zeros((1, 11, 84, 84), dtype=np.uint8),
            np.zeros(1, dtype=np.float32),
            np.zeros(1, dtype=np.bool_),
            np.zeros(1, dtype=np.bool_),
            {},
        )

    live.step = bad_step
    monkeypatch.setattr(evaluation, "_make_live_environment", lambda flags: live)
    flags = SimpleNamespace(**_flag_values(game_id=73, env_name="Synthetic-v0"))

    with pytest.raises(ValueError, match="step observation has shape"):
        evaluation.resolve_evaluation_spec(flags)

    assert live.closed is True


def _checkpoint(values, num_actions):
    return {
        "step": 128,
        "real_step": 64,
        "imitation_update_count": 7,
        "imitation_schedule_step": 7,
        "imitation_data_signature": "a" * 64,
        "action_prior_ema": np.ones(num_actions, dtype=np.float32),
        "actor_net_state_dict": {},
        "model_net_state_dict": {},
        "flags": dict(values),
    }


@pytest.mark.parametrize(
    ("game_id", "env_name", "num_actions"),
    [(0, "Enduro-v5", 9), (1, "Pong-v5", 6)],
)
def test_checkpoint_prior_and_embedded_flags_follow_dynamic_action_dimension(
    game_id, env_name, num_actions
):
    values = _flag_values(game_id=game_id, env_name=env_name)
    flags = SimpleNamespace(**values)
    spec = _spec(game_id=game_id, env_name=env_name, num_actions=num_actions)
    actor = _checkpoint(values, num_actions)
    model = _checkpoint(values, num_actions)

    evaluation.validate_actor_imitation_checkpoint(actor, flags, spec)
    evaluation.validate_model_checkpoint(model, flags, spec)

    actor["action_prior_ema"] = np.ones(num_actions + 1, dtype=np.float32)
    with pytest.raises(ValueError, match="invalid action-prior EMA"):
        evaluation.validate_actor_imitation_checkpoint(actor, flags, spec)


def test_checkpoint_validation_rejects_untrained_actor_or_model():
    values = _flag_values()
    flags = SimpleNamespace(**values)
    spec = _spec()

    actor = _checkpoint(values, spec.num_actions)
    actor["real_step"] = 0
    with pytest.raises(ValueError, match="before any training progress"):
        evaluation.validate_actor_imitation_checkpoint(actor, flags, spec)

    model = _checkpoint(values, spec.num_actions)
    model["step"] = 0
    with pytest.raises(ValueError, match="before a completed model update"):
        evaluation.validate_model_checkpoint(model, flags, spec)

    model = _checkpoint(values, spec.num_actions)
    model["flags"]["train_model"] = False
    with pytest.raises(ValueError, match="train_model=True"):
        evaluation.validate_model_checkpoint(model, flags, spec)


def test_actor_and_model_embedded_environment_identity_must_match_config():
    values = _flag_values(game_id=1, env_name="Pong-v5")
    flags = SimpleNamespace(**values)
    spec = _spec(game_id=1, env_name="Pong-v5", num_actions=6)

    actor = _checkpoint(values, 6)
    actor["flags"]["name"] = "Enduro-v5"
    with pytest.raises(ValueError, match="actor checkpoint embedded identity mismatch"):
        evaluation.validate_actor_imitation_checkpoint(actor, flags, spec)

    model = _checkpoint(values, 6)
    model["flags"]["icopro_game_id"] = 0
    with pytest.raises(ValueError, match="model checkpoint embedded protocol mismatch"):
        evaluation.validate_model_checkpoint(model, flags, spec)


def test_checkpoint_runtime_semantics_must_match_config():
    values = _flag_values()
    flags = SimpleNamespace(**values)
    spec = _spec()
    actor = _checkpoint(values, spec.num_actions)
    flags.discounting = 0.5

    with pytest.raises(ValueError, match="runtime semantic discounting"):
        evaluation.validate_actor_imitation_checkpoint(actor, flags, spec)


def test_checkpoint_protocol_normalizes_legacy_voc_defaults():
    flags = SimpleNamespace(**_flag_values())

    protocol = evaluation.checkpoint_protocol(flags)

    for key, expected in evaluation.VOC_PROTOCOL_DEFAULTS.items():
        if key == "voc_model_input_seal_schema_version":
            assert key not in protocol
        else:
            assert protocol[key] == expected


@pytest.mark.parametrize("gate_schema", [7, 8, 9, 10, 11, 12, 13])
def test_checkpoint_protocol_records_model_input_seal_for_sealed_schemas(
    gate_schema,
):
    legacy_flags = SimpleNamespace(
        **{
            **_flag_values(),
            "voc_gate_policy_schema_version": 6,
            "voc_model_input_seal_schema_version": 0,
        }
    )
    sealed_flags = SimpleNamespace(
        **{
            **_flag_values(),
            "voc_gate_policy_schema_version": gate_schema,
            "voc_model_input_seal_schema_version": 1,
        }
    )

    legacy = evaluation.checkpoint_protocol(legacy_flags)
    sealed = evaluation.checkpoint_protocol(sealed_flags)

    assert "voc_model_input_seal_schema_version" not in legacy
    assert len(sealed) == len(legacy) + 1
    assert sealed["voc_model_input_seal_schema_version"] == 1


def test_checkpoint_protocol_matches_immutable_v13_record_when_available():
    config_path = Path(
        "/tmp/di-voc-v13-versioned-eps25-final-CnOCd9/runs/"
        "enduro-voc-v13-versioned-eps25-sentinel-wire1200/config_c.yaml"
    )
    if not config_path.is_file():
        pytest.skip("immutable v13 wire evidence is not present")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    protocol = evaluation.checkpoint_protocol(SimpleNamespace(**config))
    canonical = json.dumps(
        protocol, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")

    assert len(protocol) == 99
    assert "voc_model_input_seal_schema_version" not in protocol
    assert len(canonical) == 2477
    assert hashlib.sha256(canonical).hexdigest() == (
        "d900fd4af809b5fb425b54fc0401caeb3ce4d777e78ef58bf23736eb33194c07"
    )


def test_evaluator_voc_protocol_constants_match_runtime_contract():
    assert evaluation.VOC_PROTOCOL_DEFAULTS == util.VOC_PROTOCOL_DEFAULTS
    assert (
        evaluation.VOC_ACTIVE_ONLY_PROTOCOL_FIELDS
        == util.VOC_ACTIVE_ONLY_PROTOCOL_FIELDS
    )
    assert evaluation.VOC_GATE_POLICY_SCHEMA_VERSION == 3
    assert evaluation.VOC_GATE_POLICY_EXACT_PROJECTION_SCHEMA_VERSION == 4
    assert (
        evaluation.VOC_GATE_POLICY_EPSILON_GREEDY_EXECUTION_SCHEMA_VERSION
        == 5
    )
    assert evaluation.VOC_GATE_POLICY_MODEL_INPUT_SEAL_SCHEMA_VERSION == 7


def _voc_holdout_state():
    return {
        "voc_holdout_count": 10,
        "voc_holdout_split_version": 1,
        "voc_holdout_actor_modulus": 8,
        "voc_holdout_actor_streams": 16,
        "voc_holdout_continue_count": 6,
        "voc_holdout_stop_count": 4,
        "voc_holdout_td_bias": 0.1,
        "voc_holdout_td_mae": 0.2,
        "voc_holdout_td_rmse": 0.3,
        "voc_holdout_td_sum": 1.0,
        "voc_holdout_td_abs_sum": 2.0,
        "voc_holdout_td_sq_sum": 0.9,
    }


def _voc_optimizer_state():
    return {
        "voc_optimizer_state_dict": {
            "state": {
                0: {
                    "step": torch.tensor(5.0),
                    "exp_avg": torch.zeros(2, 4),
                    "exp_avg_sq": torch.zeros(2, 4),
                },
                1: {
                    "step": torch.tensor(5.0),
                    "exp_avg": torch.zeros(2),
                    "exp_avg_sq": torch.zeros(2),
                },
            },
            "param_groups": [{
                "params": [0, 1],
                "lr": 0.000108,
                "initial_lr": 0.0003,
                "eps": 1e-8,
                "weight_decay": 0.0,
                "betas": (0.9, 0.999),
                "amsgrad": False,
                "maximize": False,
                "foreach": None,
                "capturable": False,
                "differentiable": False,
                "fused": None,
                "decoupled_weight_decay": False,
            }],
        },
        "voc_scheduler_state_dict": {
            "base_lrs": [0.0003],
            "last_epoch": 64,
            "_step_count": 6,
            "_is_initial": False,
            "_get_lr_called_within_step": False,
            "_last_lr": [0.000108],
            "lr_lambdas": [None],
        },
        "voc_grad_scaler_state_dict": None,
        "voc_amp_skip_count": 0,
        "voc_amp_consecutive_skips": 0,
    }


def _voc_ema_state(*, update_count, parent_update_count=0):
    return {
        "voc_ema_gate_target": True,
        "voc_gate_target_tau": 0.1,
        "voc_ema_gate_schema_version": 1,
        "voc_ema_gate_head_state_dict": {
            "weight": torch.zeros(2, 4, dtype=torch.float32),
            "bias": torch.zeros(2, dtype=torch.float32),
        },
        "voc_ema_gate_update_count": update_count + parent_update_count,
        "voc_ema_gate_parent_update_count": parent_update_count,
    }


def _voc_gate_state(*, update_count=0, beta1=0.9, schema=3):
    optimizer_state = {}
    last_epoch = 0 if update_count == 0 else 64
    learning_rate = 0.0003 * (1.0 - last_epoch / 100.0)
    if update_count > 0:
        optimizer_state = {
            0: {
                "step": torch.tensor(float(update_count)),
                "exp_avg": torch.zeros(1, 4),
                "exp_avg_sq": torch.zeros(1, 4),
            },
            1: {
                "step": torch.tensor(float(update_count)),
                "exp_avg": torch.zeros(1),
                "exp_avg_sq": torch.zeros(1),
            },
        }
    return {
        "voc_gate_policy_schema_version": schema,
        "voc_gate_update_count": update_count,
        "voc_gate_amp_skip_count": 0,
        "voc_gate_amp_consecutive_skips": 0,
        "voc_gate_optimizer_state_dict": {
            "state": optimizer_state,
            "param_groups": [{
                "params": [0, 1],
                "lr": learning_rate,
                "initial_lr": 0.0003,
                "eps": 1e-8,
                "weight_decay": 0.0,
                "betas": (beta1, 0.999),
                "amsgrad": False,
                "maximize": False,
                "foreach": None,
                "capturable": False,
                "differentiable": False,
                "fused": None,
                "decoupled_weight_decay": False,
            }],
        },
        "voc_gate_scheduler_state_dict": {
            "base_lrs": [0.0003],
            "last_epoch": last_epoch,
            "_step_count": update_count + 1,
            "_is_initial": False,
            "_get_lr_called_within_step": False,
            "_last_lr": [learning_rate],
            "lr_lambdas": [None],
        },
        "voc_gate_grad_scaler_state_dict": None,
    }


def _voc_actor_state(*, gate_learned=False):
    return {
        "voc_head.weight": torch.zeros(2, 4),
        "voc_head.bias": torch.zeros(2),
        "voc_gate_head.weight": (
            torch.ones(1, 4) if gate_learned else torch.zeros(1, 4)
        ),
        "voc_gate_head.bias": (
            torch.ones(1) if gate_learned else torch.zeros(1)
        ),
    }


def _complete_active_voc_actor(values, *, mode, beta1=0.9, schema=2):
    actor = _checkpoint(values, _spec().num_actions)
    actor["actor_net_state_dict"] = _voc_actor_state(
        gate_learned=(mode == "control")
    )
    actor.update({
        **_voc_holdout_state(),
        **_voc_optimizer_state(),
        **_voc_ema_state(update_count=5),
        **_voc_gate_state(
            update_count=(5 if mode == "control" else 0),
            beta1=beta1,
            schema=schema,
        ),
        "dynamic_voc_mode": mode,
        "voc_update_count": 5,
        "voc_continue_count": 12,
        "voc_stop_count": 7,
        "voc_control_origin": "fresh" if mode == "control" else None,
        "voc_control_origin_legacy_defaulted": False,
        "voc_parent_checkpoint_sha256": None,
        "voc_parent_checkpoint": None,
        "voc_parent_imitation_data_signature": None,
        "voc_activation_real_step": 0 if mode == "control" else -1,
    })
    return actor


def _complete_exact_projection_actor(values):
    actor = _complete_active_voc_actor(
        values, mode="control", beta1=0.0, schema=4
    )
    actor["actor_net_state_dict"] = _voc_actor_state(gate_learned=False)
    actor.update(_voc_gate_state(update_count=0, beta1=0.0, schema=4))
    actor["voc_gate_update_count"] = actor["voc_update_count"]
    return actor


def _epsilon_greedy_execution_values():
    values = _exact_projection_values()
    values["voc_gate_epsilon_greedy_execution"] = True
    return values


def _complete_epsilon_greedy_execution_actor(values):
    actor = _complete_exact_projection_actor(values)
    actor["voc_gate_policy_schema_version"] = 5
    return actor


@pytest.mark.parametrize("mode", ["shadow", "control"])
@pytest.mark.parametrize("confidence_weighted", [True, False])
def test_checkpoint_voc_protocol_and_provenance_are_verified(
    mode, confidence_weighted
):
    values = _flag_values()
    values.update(evaluation.VOC_PROTOCOL_DEFAULTS)
    values["dynamic_voc_mode"] = mode
    values["voc_gate_confidence_weighted"] = confidence_weighted
    flags = SimpleNamespace(**values)
    spec = _spec()
    actor = _checkpoint(values, spec.num_actions)
    actor["actor_net_state_dict"] = _voc_actor_state(
        gate_learned=(mode == "control")
    )
    actor.update({
        **_voc_holdout_state(),
        **_voc_optimizer_state(),
        **_voc_ema_state(
            update_count=5,
            parent_update_count=(3 if mode == "control" else 0),
        ),
        **_voc_gate_state(update_count=(5 if mode == "control" else 0)),
        "dynamic_voc_mode": mode,
        "voc_update_count": 5,
        "voc_continue_count": 12,
        "voc_stop_count": 7,
        "voc_activation_real_step": -1 if mode == "shadow" else 0,
        "voc_parent_checkpoint_sha256": None if mode == "shadow" else "b" * 64,
    })

    state = evaluation.validate_actor_imitation_checkpoint(actor, flags, spec)

    assert state["voc"]["dynamic_voc_mode"] == mode
    assert state["voc"]["voc_activation_real_step"] == (
        -1 if mode == "shadow" else 0
    )
    assert state["voc"]["legacy_voc_metadata_defaulted"] is False


def test_evaluator_accepts_v2_zero_beta1_for_actor_and_model():
    values = _flag_values()
    values.update(evaluation.VOC_PROTOCOL_DEFAULTS)
    values.update(dynamic_voc_mode="control", voc_gate_adam_beta1=0.0)
    flags = SimpleNamespace(**values)
    spec = _spec()
    actor = _complete_active_voc_actor(
        values, mode="control", beta1=0.0, schema=2
    )
    model = _checkpoint(values, spec.num_actions)

    actor_state = evaluation.validate_actor_imitation_checkpoint(
        actor, flags, spec
    )
    model_state = evaluation.validate_model_checkpoint(model, flags, spec)

    assert actor_state["voc"]["voc_gate_adam_beta1"] == 0.0
    assert actor_state["voc"][
        "voc_gate_adam_beta1_legacy_defaulted"
    ] is False
    assert model_state["embedded_protocol_verified"] is True


def test_evaluator_model_requires_exact_beta1_config_identity():
    values = _flag_values()
    values.update(evaluation.VOC_PROTOCOL_DEFAULTS)
    values.update(dynamic_voc_mode="shadow", voc_gate_adam_beta1=0.5)
    flags = SimpleNamespace(**values)
    spec = _spec()
    model = _checkpoint(values, spec.num_actions)
    model["flags"]["voc_gate_adam_beta1"] = np.nextafter(0.5, 1.0)

    with pytest.raises(ValueError, match="voc_gate_adam_beta1"):
        evaluation.validate_model_checkpoint(model, flags, spec)


def test_evaluator_accepts_schema1_missing_beta_as_point9_for_actor_and_model():
    values = _flag_values()
    values.update(evaluation.VOC_PROTOCOL_DEFAULTS)
    values["dynamic_voc_mode"] = "shadow"
    flags = SimpleNamespace(**values)
    spec = _spec()
    actor = _complete_active_voc_actor(
        values, mode="shadow", schema=1
    )
    actor["flags"].pop("voc_gate_adam_beta1")
    model = _checkpoint(values, spec.num_actions)
    model["flags"].pop("voc_gate_adam_beta1")

    actor_state = evaluation.validate_actor_imitation_checkpoint(
        actor, flags, spec
    )
    model_state = evaluation.validate_model_checkpoint(model, flags, spec)

    assert actor_state["voc"]["voc_gate_adam_beta1"] == pytest.approx(0.9)
    assert actor_state["voc"][
        "voc_gate_adam_beta1_legacy_defaulted"
    ] is True
    assert model_state["embedded_protocol_verified"] is True


def test_evaluator_rejects_schema1_missing_beta_against_zero_config():
    values = _flag_values()
    values.update(evaluation.VOC_PROTOCOL_DEFAULTS)
    values.update(dynamic_voc_mode="shadow", voc_gate_adam_beta1=0.0)
    flags = SimpleNamespace(**values)
    spec = _spec()
    actor = _complete_active_voc_actor(
        values, mode="shadow", schema=1
    )
    actor["flags"].pop("voc_gate_adam_beta1")

    with pytest.raises(
        ValueError, match="voc_gate_adam_beta1.*0.9.*0.0"
    ):
        evaluation.validate_actor_imitation_checkpoint(actor, flags, spec)


@pytest.mark.parametrize("bad_beta1", [True, -0.1, 1.0, float("nan")])
def test_evaluator_rejects_corrupt_explicit_beta1(bad_beta1):
    values = _flag_values()
    values.update(evaluation.VOC_PROTOCOL_DEFAULTS)
    values["dynamic_voc_mode"] = "shadow"
    flags = SimpleNamespace(**values)
    spec = _spec()
    actor = _complete_active_voc_actor(values, mode="shadow", schema=2)
    actor["flags"]["voc_gate_adam_beta1"] = bad_beta1

    with pytest.raises(ValueError, match="voc_gate_adam_beta1"):
        evaluation.validate_actor_imitation_checkpoint(actor, flags, spec)


def test_evaluator_rejects_schema2_missing_beta1():
    values = _flag_values()
    values.update(evaluation.VOC_PROTOCOL_DEFAULTS)
    values["dynamic_voc_mode"] = "shadow"
    flags = SimpleNamespace(**values)
    spec = _spec()
    actor = _complete_active_voc_actor(values, mode="shadow", schema=2)
    actor["flags"].pop("voc_gate_adam_beta1")

    with pytest.raises(ValueError, match="lacks embedded voc_gate_adam_beta1"):
        evaluation.validate_actor_imitation_checkpoint(actor, flags, spec)


@pytest.mark.parametrize("schema", [1, 2])
def test_evaluator_legacy_gate_schemas_default_alignment_disabled(schema):
    values = _flag_values()
    values.update(evaluation.VOC_PROTOCOL_DEFAULTS)
    values["dynamic_voc_mode"] = "shadow"
    flags = SimpleNamespace(**values)
    spec = _spec()
    actor = _complete_active_voc_actor(values, mode="shadow", schema=schema)
    model = _checkpoint(values, spec.num_actions)
    for checkpoint in (actor, model):
        checkpoint["flags"].pop("voc_gate_param_align")
        checkpoint["flags"].pop("voc_gate_param_align_coef")
        checkpoint["flags"].pop("voc_gate_exact_projection")
        checkpoint["flags"].pop("voc_gate_epsilon_greedy_execution")

    actor_state = evaluation.validate_actor_imitation_checkpoint(
        actor, flags, spec
    )
    model_state = evaluation.validate_model_checkpoint(model, flags, spec)

    assert actor_state["voc"]["voc_gate_param_align"] is False
    assert actor_state["voc"]["voc_gate_param_align_coef"] == 1.0
    assert actor_state["voc"][
        "voc_gate_param_align_legacy_defaulted"
    ] is True
    assert actor_state["voc"]["voc_gate_exact_projection"] is False
    assert actor_state["voc"][
        "voc_gate_exact_projection_legacy_defaulted"
    ] is True
    assert actor_state["voc"]["voc_gate_epsilon_greedy_execution"] is False
    assert actor_state["voc"][
        "voc_gate_epsilon_greedy_execution_legacy_defaulted"
    ] is True
    assert model_state["embedded_protocol_verified"] is True


def test_evaluator_accepts_schema3_alignment_true_as_shadow_metadata():
    values = _flag_values()
    values.update(evaluation.VOC_PROTOCOL_DEFAULTS)
    values.update(dynamic_voc_mode="shadow", voc_gate_param_align=True)
    flags = SimpleNamespace(**values)
    spec = _spec()
    actor = _complete_active_voc_actor(values, mode="shadow", schema=3)
    model = _checkpoint(values, spec.num_actions)

    actor_state = evaluation.validate_actor_imitation_checkpoint(
        actor, flags, spec
    )
    model_state = evaluation.validate_model_checkpoint(model, flags, spec)

    assert actor_state["voc"]["voc_gate_policy_schema_version"] == 3
    assert actor_state["voc"]["voc_gate_param_align"] is True
    assert actor_state["voc"]["voc_gate_param_align_coef"] == 1.0
    assert actor_state["voc"][
        "voc_gate_param_align_legacy_defaulted"
    ] is False
    assert model_state["embedded_protocol_verified"] is True


@pytest.mark.parametrize(
    "missing", ["voc_gate_param_align", "voc_gate_param_align_coef"]
)
def test_evaluator_schema3_requires_explicit_alignment_pair(missing):
    values = _flag_values()
    values.update(evaluation.VOC_PROTOCOL_DEFAULTS)
    values["dynamic_voc_mode"] = "shadow"
    flags = SimpleNamespace(**values)
    spec = _spec()
    actor = _complete_active_voc_actor(values, mode="shadow", schema=3)
    actor["flags"].pop(missing)

    with pytest.raises(ValueError, match=rf"schema 3 lacks embedded {missing}"):
        evaluation.validate_actor_imitation_checkpoint(actor, flags, spec)


def _exact_projection_values():
    values = _flag_values()
    values.update(evaluation.VOC_PROTOCOL_DEFAULTS)
    values.update(
        dynamic_voc_mode="control",
        voc_gate_confidence_weighted=False,
        voc_gate_adam_beta1=0.0,
        voc_gate_param_align=False,
        voc_gate_param_align_coef=1.0,
        voc_gate_exact_projection=True,
    )
    return values


def test_evaluator_accepts_schema4_exact_projection_for_actor_and_model():
    values = _exact_projection_values()
    flags = SimpleNamespace(**values)
    spec = _spec()
    actor = _complete_exact_projection_actor(values)
    model = _checkpoint(values, spec.num_actions)

    actor_state = evaluation.validate_actor_imitation_checkpoint(
        actor, flags, spec
    )
    model_state = evaluation.validate_model_checkpoint(model, flags, spec)

    assert actor_state["voc"]["voc_gate_policy_schema_version"] == 4
    assert actor_state["voc"]["voc_gate_param_align"] is False
    assert actor_state["voc"]["voc_gate_exact_projection"] is True
    assert actor_state["voc"][
        "voc_gate_exact_projection_legacy_defaulted"
    ] is False
    assert model_state["embedded_protocol_verified"] is True


def test_evaluator_accepts_schema5_epsilon_greedy_execution_for_actor_and_model():
    values = _epsilon_greedy_execution_values()
    flags = SimpleNamespace(**values)
    spec = _spec()
    actor = _complete_epsilon_greedy_execution_actor(values)
    model = _checkpoint(values, spec.num_actions)

    actor_state = evaluation.validate_actor_imitation_checkpoint(
        actor, flags, spec
    )
    model_state = evaluation.validate_model_checkpoint(model, flags, spec)

    assert actor_state["voc"]["voc_gate_policy_schema_version"] == 5
    assert actor_state["voc"]["voc_gate_param_align"] is False
    assert actor_state["voc"]["voc_gate_exact_projection"] is True
    assert actor_state["voc"]["voc_gate_epsilon_greedy_execution"] is True
    assert actor_state["voc"][
        "voc_gate_epsilon_greedy_execution_legacy_defaulted"
    ] is False
    assert model_state["embedded_protocol_verified"] is True


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            "missing",
            "schema 5 lacks embedded voc_gate_epsilon_greedy_execution",
        ),
        (
            "disabled",
            "schema 5 requires voc_gate_epsilon_greedy_execution=true",
        ),
        (
            "nonboolean",
            "voc_gate_epsilon_greedy_execution to be boolean",
        ),
        ("no_projection", "schema 5 requires voc_gate_exact_projection=true"),
    ],
)
def test_evaluator_schema5_execution_metadata_fails_closed(
    mutation, message
):
    values = _epsilon_greedy_execution_values()
    flags = SimpleNamespace(**values)
    spec = _spec()
    actor = _complete_epsilon_greedy_execution_actor(values)
    if mutation == "missing":
        actor["flags"].pop("voc_gate_epsilon_greedy_execution")
    elif mutation == "disabled":
        actor["flags"]["voc_gate_epsilon_greedy_execution"] = False
    elif mutation == "nonboolean":
        actor["flags"]["voc_gate_epsilon_greedy_execution"] = 1
    else:
        actor["flags"]["voc_gate_exact_projection"] = False

    with pytest.raises(ValueError, match=message):
        evaluation.validate_actor_imitation_checkpoint(actor, flags, spec)


@pytest.mark.parametrize("schema", [3, 4])
def test_evaluator_schema3_and_4_default_missing_epsilon_execution_false(
    schema,
):
    values = (
        _exact_projection_values()
        if schema == 4
        else {
            **_flag_values(),
            **evaluation.VOC_PROTOCOL_DEFAULTS,
            "dynamic_voc_mode": "shadow",
        }
    )
    flags = SimpleNamespace(**values)
    spec = _spec()
    actor = (
        _complete_exact_projection_actor(values)
        if schema == 4
        else _complete_active_voc_actor(
            values, mode="shadow", schema=3
        )
    )
    model = _checkpoint(values, spec.num_actions)
    for checkpoint in (actor, model):
        checkpoint["flags"].pop("voc_gate_epsilon_greedy_execution")

    actor_state = evaluation.validate_actor_imitation_checkpoint(
        actor, flags, spec
    )
    model_state = evaluation.validate_model_checkpoint(model, flags, spec)

    assert actor_state["voc"]["voc_gate_epsilon_greedy_execution"] is False
    assert actor_state["voc"][
        "voc_gate_epsilon_greedy_execution_legacy_defaulted"
    ] is True
    assert model_state["embedded_protocol_verified"] is True


def test_evaluator_schema4_rejects_enabled_epsilon_greedy_execution():
    values = _epsilon_greedy_execution_values()
    flags = SimpleNamespace(**values)
    spec = _spec()
    actor = _complete_epsilon_greedy_execution_actor(values)
    actor["voc_gate_policy_schema_version"] = 4

    with pytest.raises(
        ValueError, match="predates epsilon-greedy execution.*requires"
    ):
        evaluation.validate_actor_imitation_checkpoint(actor, flags, spec)


@pytest.mark.parametrize("surface", ["actor", "model", "config"])
def test_evaluator_epsilon_greedy_execution_identity_mismatch_fails(surface):
    values = _epsilon_greedy_execution_values()
    flags = SimpleNamespace(**values)
    spec = _spec()
    actor = _complete_epsilon_greedy_execution_actor(values)
    model = _checkpoint(values, spec.num_actions)
    if surface == "actor":
        actor["flags"]["voc_gate_epsilon_greedy_execution"] = False
        with pytest.raises(
            ValueError, match="voc_gate_epsilon_greedy_execution"
        ):
            evaluation.validate_actor_imitation_checkpoint(actor, flags, spec)
    elif surface == "model":
        model["flags"]["voc_gate_epsilon_greedy_execution"] = False
        with pytest.raises(
            ValueError, match="voc_gate_epsilon_greedy_execution"
        ):
            evaluation.validate_model_checkpoint(model, flags, spec)
    else:
        flags.voc_gate_epsilon_greedy_execution = False
        with pytest.raises(
            ValueError, match="voc_gate_epsilon_greedy_execution"
        ):
            evaluation.validate_actor_imitation_checkpoint(actor, flags, spec)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "schema 4 lacks embedded voc_gate_exact_projection"),
        ("disabled", "schema 4 requires voc_gate_exact_projection=true"),
        ("nonboolean", "voc_gate_exact_projection to be boolean"),
        ("aligned", "mutually exclusive"),
    ],
)
def test_evaluator_schema4_projection_metadata_fails_closed(
    mutation, message
):
    values = _exact_projection_values()
    flags = SimpleNamespace(**values)
    spec = _spec()
    actor = _complete_exact_projection_actor(values)
    if mutation == "missing":
        actor["flags"].pop("voc_gate_exact_projection")
    elif mutation == "disabled":
        actor["flags"]["voc_gate_exact_projection"] = False
    elif mutation == "nonboolean":
        actor["flags"]["voc_gate_exact_projection"] = 1
    else:
        actor["flags"]["voc_gate_param_align"] = True

    with pytest.raises(ValueError, match=message):
        evaluation.validate_actor_imitation_checkpoint(actor, flags, spec)


def test_evaluator_schema3_defaults_missing_exact_projection_false():
    values = _flag_values()
    values.update(evaluation.VOC_PROTOCOL_DEFAULTS)
    values["dynamic_voc_mode"] = "shadow"
    flags = SimpleNamespace(**values)
    spec = _spec()
    actor = _complete_active_voc_actor(values, mode="shadow", schema=3)
    actor["flags"].pop("voc_gate_exact_projection")

    actor_state = evaluation.validate_actor_imitation_checkpoint(
        actor, flags, spec
    )

    assert actor_state["voc"]["voc_gate_exact_projection"] is False
    assert actor_state["voc"][
        "voc_gate_exact_projection_legacy_defaulted"
    ] is True


def test_evaluator_schema3_rejects_enabled_exact_projection():
    values = _exact_projection_values()
    flags = SimpleNamespace(**values)
    spec = _spec()
    actor = _complete_exact_projection_actor(values)
    actor["voc_gate_policy_schema_version"] = 3

    with pytest.raises(
        ValueError, match="predates exact projection.*requires"
    ):
        evaluation.validate_actor_imitation_checkpoint(actor, flags, spec)


@pytest.mark.parametrize("surface", ["actor", "model", "config"])
def test_evaluator_exact_projection_identity_mismatch_fails(surface):
    values = _exact_projection_values()
    flags = SimpleNamespace(**values)
    spec = _spec()
    actor = _complete_exact_projection_actor(values)
    model = _checkpoint(values, spec.num_actions)
    if surface == "actor":
        actor["flags"]["voc_gate_exact_projection"] = False
        with pytest.raises(ValueError, match="voc_gate_exact_projection"):
            evaluation.validate_actor_imitation_checkpoint(actor, flags, spec)
    elif surface == "model":
        model["flags"]["voc_gate_exact_projection"] = False
        with pytest.raises(ValueError, match="voc_gate_exact_projection"):
            evaluation.validate_model_checkpoint(model, flags, spec)
    else:
        flags.voc_gate_exact_projection = False
        with pytest.raises(ValueError, match="voc_gate_exact_projection"):
            evaluation.validate_actor_imitation_checkpoint(actor, flags, spec)


def test_evaluator_rejects_one_bit_exact_projection_target_mismatch():
    values = _exact_projection_values()
    flags = SimpleNamespace(**values)
    spec = _spec()
    actor = _complete_exact_projection_actor(values)
    gate_weight = actor["actor_net_state_dict"]["voc_gate_head.weight"]
    actor["actor_net_state_dict"]["voc_gate_head.weight"] = torch.nextafter(
        gate_weight,
        torch.ones_like(gate_weight),
    )

    with pytest.raises(ValueError, match="disagrees with EMA Q target"):
        evaluation.validate_actor_imitation_checkpoint(actor, flags, spec)


@pytest.mark.parametrize("schema", [1, 2])
def test_evaluator_legacy_gate_schemas_reject_enabled_alignment(schema):
    values = _flag_values()
    values.update(evaluation.VOC_PROTOCOL_DEFAULTS)
    values.update(dynamic_voc_mode="shadow", voc_gate_param_align=True)
    flags = SimpleNamespace(**values)
    spec = _spec()
    actor = _complete_active_voc_actor(values, mode="shadow", schema=schema)

    with pytest.raises(ValueError, match="requires voc_gate_param_align=false"):
        evaluation.validate_actor_imitation_checkpoint(actor, flags, spec)


@pytest.mark.parametrize(
    ("checkpoint_align", "config_align"), [(True, False), (False, True)]
)
def test_evaluator_model_requires_exact_alignment_config_identity(
    checkpoint_align, config_align
):
    values = _flag_values()
    values.update(evaluation.VOC_PROTOCOL_DEFAULTS)
    values.update(
        dynamic_voc_mode="shadow", voc_gate_param_align=config_align
    )
    flags = SimpleNamespace(**values)
    spec = _spec()
    model = _checkpoint(values, spec.num_actions)
    model["flags"]["voc_gate_param_align"] = checkpoint_align

    with pytest.raises(ValueError, match="voc_gate_param_align"):
        evaluation.validate_model_checkpoint(model, flags, spec)


@pytest.mark.parametrize("location", ["checkpoint", "config"])
def test_evaluator_requires_exact_unit_alignment_coefficient(location):
    values = _flag_values()
    values.update(evaluation.VOC_PROTOCOL_DEFAULTS)
    values.update(dynamic_voc_mode="shadow", voc_gate_param_align=True)
    flags = SimpleNamespace(**values)
    spec = _spec()
    model = _checkpoint(values, spec.num_actions)
    if location == "checkpoint":
        model["flags"]["voc_gate_param_align_coef"] = np.nextafter(1.0, 2.0)
    else:
        flags.voc_gate_param_align_coef = np.nextafter(1.0, 2.0)

    with pytest.raises(ValueError, match="voc_gate_param_align_coef=1.0 exactly"):
        evaluation.validate_model_checkpoint(model, flags, spec)


def test_active_checkpoint_requires_exact_confidence_bool_identity():
    values = _flag_values()
    values.update(evaluation.VOC_PROTOCOL_DEFAULTS)
    values["dynamic_voc_mode"] = "control"
    values["voc_gate_confidence_weighted"] = False
    flags = SimpleNamespace(**values)
    spec = _spec()
    actor = _checkpoint(values, spec.num_actions)
    actor["flags"]["voc_gate_confidence_weighted"] = True

    with pytest.raises(
        ValueError, match="voc_gate_confidence_weighted.*True.*False"
    ):
        evaluation.validate_actor_imitation_checkpoint(actor, flags, spec)


@pytest.mark.parametrize(
    "missing",
    [
        "voc_dueling_q",
        "voc_expected_gate_loss",
        "voc_dedicated_gate",
        "voc_soft_q_bce_gate",
        "voc_gate_q_temperature",
        "voc_gate_confidence_weighted",
        "voc_gate_adam_beta1",
        "voc_gate_param_align",
        "voc_gate_param_align_coef",
        "voc_gate_learning_rate",
        "voc_gate_grad_norm_clipping",
        "entropy_r_cost",
    ],
)
def test_active_checkpoint_requires_voc_protocol_fields(missing):
    values = _flag_values()
    values.update(evaluation.VOC_PROTOCOL_DEFAULTS)
    values["dynamic_voc_mode"] = "shadow"
    flags = SimpleNamespace(**values)
    spec = _spec()
    actor = _checkpoint(values, spec.num_actions)
    actor["flags"].pop(missing)
    actor["actor_net_state_dict"] = _voc_actor_state()
    actor.update({
        **_voc_holdout_state(),
        **_voc_optimizer_state(),
        **_voc_ema_state(update_count=5),
        **_voc_gate_state(),
        "dynamic_voc_mode": "shadow",
        "voc_update_count": 5,
        "voc_continue_count": 12,
        "voc_stop_count": 7,
        "voc_activation_real_step": -1,
        "voc_parent_checkpoint_sha256": None,
    })

    with pytest.raises(ValueError, match=rf"lacks embedded {missing}"):
        evaluation.validate_actor_imitation_checkpoint(actor, flags, spec)


@pytest.mark.parametrize(
    ("checkpoint_entropy", "config_entropy"),
    [(0.01, 0.0), (0.0, 0.01), (False, 0.0), (float("nan"), 0.0)],
)
def test_active_checkpoint_rejects_corrupt_environment_return_anchor(
    checkpoint_entropy, config_entropy
):
    values = _flag_values()
    values.update(evaluation.VOC_PROTOCOL_DEFAULTS)
    values["dynamic_voc_mode"] = "shadow"
    values["entropy_r_cost"] = config_entropy
    flags = SimpleNamespace(**values)
    spec = _spec()
    actor = _checkpoint(values, spec.num_actions)
    actor["flags"]["entropy_r_cost"] = checkpoint_entropy

    with pytest.raises(ValueError, match="entropy_r_cost=0"):
        evaluation.validate_actor_imitation_checkpoint(actor, flags, spec)


def test_legacy_off_checkpoint_ignores_entropy_return_anchor_metadata():
    values = _flag_values()
    values["entropy_r_cost"] = 0.125
    flags = SimpleNamespace(**values)
    spec = _spec()
    actor = _checkpoint(values, spec.num_actions)
    actor["flags"].pop("entropy_r_cost")

    state = evaluation.validate_actor_imitation_checkpoint(actor, flags, spec)

    assert state["voc"]["dynamic_voc_mode"] == "off"
    assert state["voc"]["legacy_voc_metadata_defaulted"] is True


def test_checkpoint_accepts_explicit_fresh_control_with_null_parent():
    values = _flag_values()
    values.update(evaluation.VOC_PROTOCOL_DEFAULTS)
    values["dynamic_voc_mode"] = "control"
    flags = SimpleNamespace(**values)
    spec = _spec()
    actor = _checkpoint(values, spec.num_actions)
    actor["actor_net_state_dict"] = _voc_actor_state(gate_learned=True)
    actor.update({
        **_voc_holdout_state(),
        **_voc_optimizer_state(),
        **_voc_ema_state(update_count=5),
        **_voc_gate_state(update_count=5),
        "dynamic_voc_mode": "control",
        "voc_update_count": 5,
        "voc_continue_count": 12,
        "voc_stop_count": 7,
        "voc_control_origin": "fresh",
        "voc_parent_checkpoint_sha256": None,
        "voc_parent_checkpoint": None,
        "voc_parent_imitation_data_signature": None,
        "voc_activation_real_step": 0,
    })

    state = evaluation.validate_actor_imitation_checkpoint(actor, flags, spec)

    assert state["voc"]["voc_control_origin"] == "fresh"
    assert state["voc"]["voc_control_origin_legacy_defaulted"] is False
    assert state["voc"]["voc_parent_checkpoint_sha256"] is None
    assert state["voc"]["voc_parent_checkpoint"] is None


def test_checkpoint_voc_mode_must_match_config_and_top_level_metadata():
    values = _flag_values()
    values.update(evaluation.VOC_PROTOCOL_DEFAULTS)
    values["dynamic_voc_mode"] = "shadow"
    flags = SimpleNamespace(**values)
    spec = _spec()
    actor = _checkpoint(values, spec.num_actions)
    actor.update({
        "dynamic_voc_mode": "control",
        "voc_update_count": 0,
        "voc_continue_count": 0,
        "voc_stop_count": 0,
        "voc_parent_checkpoint_sha256": "b" * 64,
        "voc_activation_real_step": 0,
    })

    with pytest.raises(ValueError, match="VoC mode disagrees"):
        evaluation.validate_actor_imitation_checkpoint(actor, flags, spec)

    actor["dynamic_voc_mode"] = "shadow"
    actor["flags"]["voc_train_epsilon"] = 0.5
    with pytest.raises(ValueError, match="voc_train_epsilon"):
        evaluation.validate_actor_imitation_checkpoint(actor, flags, spec)


@pytest.mark.parametrize(
    ("mode", "activation_step", "bad_key", "message"),
    [
        ("shadow", 0, None, "voc_activation_real_step=-1"),
        ("control", -1, None, "non-negative voc_activation_real_step"),
        ("control", 129, None, "must not exceed checkpoint real_step"),
        ("shadow", -1, "voc_update_count", "voc_update_count"),
        ("shadow", -1, "voc_continue_count", "voc_continue_count"),
        ("shadow", -1, "voc_stop_count", "voc_stop_count"),
        ("control", 0, "voc_parent_checkpoint_sha256", "parent_checkpoint"),
    ],
)
def test_checkpoint_voc_rejects_incomplete_provenance(
    mode, activation_step, bad_key, message
):
    values = _flag_values()
    values.update(evaluation.VOC_PROTOCOL_DEFAULTS)
    values["dynamic_voc_mode"] = mode
    flags = SimpleNamespace(**values)
    spec = _spec()
    actor = _checkpoint(values, spec.num_actions)
    actor.update({
        **_voc_holdout_state(),
        **_voc_optimizer_state(),
        "dynamic_voc_mode": mode,
        "voc_update_count": 1,
        "voc_continue_count": 1,
        "voc_stop_count": 1,
        "voc_parent_checkpoint_sha256": (
            "b" * 64 if mode == "control" else None
        ),
        "voc_activation_real_step": activation_step,
    })
    if bad_key is not None:
        actor.pop(bad_key)

    with pytest.raises(ValueError, match=message):
        evaluation.validate_actor_imitation_checkpoint(actor, flags, spec)


def test_checkpoint_voc_requires_two_output_q_head():
    values = _flag_values()
    values.update(evaluation.VOC_PROTOCOL_DEFAULTS)
    values["dynamic_voc_mode"] = "shadow"
    flags = SimpleNamespace(**values)
    spec = _spec()
    actor = _checkpoint(values, spec.num_actions)
    actor.update({
        **_voc_holdout_state(),
        **_voc_optimizer_state(),
        "dynamic_voc_mode": "shadow",
        "voc_update_count": 1,
        "voc_continue_count": 1,
        "voc_stop_count": 1,
        "voc_parent_checkpoint_sha256": None,
        "voc_activation_real_step": -1,
    })
    actor["actor_net_state_dict"] = {
        "voc_head.weight": torch.zeros(3, 4),
        "voc_head.bias": torch.zeros(3),
    }
    with pytest.raises(ValueError, match="voc_head must output exactly"):
        evaluation.validate_actor_imitation_checkpoint(actor, flags, spec)

    actor["actor_net_state_dict"] = {}
    with pytest.raises(ValueError, match="lacks voc_head"):
        evaluation.validate_actor_imitation_checkpoint(actor, flags, spec)


def test_checkpoint_requires_factorized_control_production_semantics():
    values = _flag_values()
    flags = SimpleNamespace(**values)
    spec = _spec()
    actor = _checkpoint(values, spec.num_actions)
    actor["flags"]["dynamic_factorized_control"] = False

    with pytest.raises(
        ValueError, match="dynamic_factorized_control=False, expected True"
    ):
        evaluation.validate_actor_imitation_checkpoint(actor, flags, spec)


def test_pong_holdout_batch_is_accepted_and_enduro_metadata_is_rejected(tmp_path):
    spec = _spec(game_id=1, env_name="Pong-v5", num_actions=6)
    batch = _batch(tmp_path, spec)
    assert evaluation.validate_holdout_batch(batch, spec) == 2

    batch["game"] = np.asarray([0, 0])
    with pytest.raises(ValueError, match="game"):
        evaluation.validate_holdout_batch(batch, spec)


def test_build_networks_propagates_synthetic_live_action_dimension(
    monkeypatch, tmp_path
):
    import thinker.actor_net as actor_module
    import thinker.cenv as cenv_module
    import thinker.dataset_env as dataset_module
    import thinker.model_net as model_module
    from thinker import util

    spec = _spec(num_actions=5)
    batch = _batch(tmp_path, spec)
    captured = {}

    class FakeBehaviorEnv:
        def __init__(self, obs_seq, actions_seq, num_actions, **kwargs):
            captured["replay_num_actions"] = num_actions
            self.action_space = spaces.Tuple(
                tuple(spaces.Discrete(num_actions) for _ in range(len(actions_seq)))
            )
            self.observation_space = spaces.Box(
                0, 255, shape=tuple(obs_seq.shape[2:]), dtype=np.uint8
            )

    class FakeModelNet:
        def __init__(self, action_space, **kwargs):
            captured["model_num_actions"] = action_space.n

        def to(self, device):
            return self

        def set_weights(self, weights):
            captured["model_weights"] = weights

        def eval(self):
            return self

    class FakeWrapper:
        def __init__(self, env, **kwargs):
            self.observation_space = spaces.Dict(
                {
                    "real_states": env.observation_space,
                    "xs": spaces.Box(
                        0.0,
                        1.0,
                        shape=env.observation_space.shape,
                        dtype=np.float32,
                    ),
                }
            )
            self.action_space = spaces.Tuple(
                (env.action_space[0], spaces.Discrete(4))
            )

        def close(self):
            captured["wrapper_closed"] = True

    class FakeActorNet:
        def __init__(self, action_space, tree_rep_meaning, **kwargs):
            self.num_actions = action_space[0].n
            captured["actor_num_actions"] = self.num_actions
            captured["tree_rep_meaning"] = tree_rep_meaning

        def to(self, device):
            return self

        def set_weights(self, weights):
            captured["actor_weights"] = weights

        def eval(self):
            return self

    monkeypatch.setattr(dataset_module, "BehaviorSequenceVectorEnv", FakeBehaviorEnv)
    monkeypatch.setattr(model_module, "ModelNet", FakeModelNet)
    monkeypatch.setattr(cenv_module, "cModelWrapper", FakeWrapper)
    monkeypatch.setattr(actor_module, "ActorNet", FakeActorNet)
    monkeypatch.setattr(
        util,
        "get_tree_rep_meaning",
        lambda num_actions, dim_actions, flags: (num_actions, dim_actions),
    )

    flags = SimpleNamespace(frame_stack_n=4)
    actor, model, returned_flags = evaluation.build_networks(
        batch,
        torch.device("cpu"),
        flags,
        spec,
        {"actor_net_state_dict": {}},
        {"model_net_state_dict": {}},
    )

    assert returned_flags is flags
    assert actor.num_actions == 5
    assert captured["replay_num_actions"] == 5
    assert captured["model_num_actions"] == 5
    assert captured["actor_num_actions"] == 5
    assert captured["tree_rep_meaning"] == (5, 1)
    assert captured["wrapper_closed"] is True


def _schema6_public_bundle_records():
    checkpoint_files = {
        name: {"sha256": character * 64, "size": index + 1}
        for index, (name, character) in enumerate(
            (
                ("config_c.yaml", "a"),
                ("ckp_actor.tar", "b"),
                ("ckp_model.tar", "c"),
            )
        )
    }
    evidence = {
        "checkpoint_files": checkpoint_files,
        "implementation_sources": {"train.py": {"sha256": "d" * 64}},
        "loaded_extensions": {"thinker/cenv.so": {"sha256": "e" * 64}},
    }
    actor_policy = {
        "voc_actor_policy_version": 2,
        "voc_actor_policy_state_sha256": "f" * 64,
        "voc_actor_policy_publication_history_sha256": "1" * 64,
        "voc_actor_policy_terminal": True,
        "voc_actor_policy_publication_history": (
            {
                "predecessor_version": -1,
                "policy_version": 0,
                "publication_count": 0,
                "terminal": False,
                "ack_ranks": [0],
                "expected_ack_count": 1,
                "state_sha256": "2" * 64,
            },
            {
                "predecessor_version": 0,
                "policy_version": 1,
                "publication_count": 1,
                "terminal": False,
                "ack_ranks": [0],
                "expected_ack_count": 1,
                "state_sha256": "3" * 64,
            },
            {
                "predecessor_version": 1,
                "policy_version": 2,
                "publication_count": 2,
                "terminal": True,
                "ack_ranks": [0],
                "expected_ack_count": 1,
                "state_sha256": "f" * 64,
            },
        ),
    }
    logger_completion = {
        "schema_version": 1,
        "required": False,
        "use_wandb": False,
        "request_sha256": None,
        "ack_verified": False,
        "private_markers_cleaned": True,
        "policy_version": 2,
        "state_sha256": "f" * 64,
        "publication_history_sha256": "1" * 64,
        "checkpoint_files": checkpoint_files,
    }
    marker = {
        "schema_version": 1,
        "status": "complete",
        **evidence,
        "voc_actor_policy_logger_completion": logger_completion,
    }
    full = {
        "actor_policy": actor_policy,
        "resolved_identity": {
            "key_count": 228,
            "v12_projection_key_count": 209,
            "v12_projection_sha256": util.VOC_GATE_POLICY_SCHEMA6_V12_BASELINE_SHA256,
            "complete_surface_sha256": "4" * 64,
            "stage": (
                "enduro-voc-v13-versioned-eps25-sentinel-wire1200",
                1,
                1200,
                512,
                41,
                False,
            ),
            "paths": {"ckpdir": "/sealed/run"},
        },
        "actor_training_state": {"voc_update_count": 2},
        "model_step": 2,
        "config_use_wandb": False,
        "completion_evidence": evidence,
    }
    return marker, full, logger_completion


def _schema7_public_bundle_records(*, drain=1):
    marker, full, logger_completion = _schema6_public_bundle_records()
    resolved = full["resolved_identity"]
    resolved.update(
        {
            "gate_schema": 7,
            "voc_gate_policy_schema_version": 7,
            "voc_model_input_seal_schema_version": 1,
            "key_count": 229,
        }
    )
    full["actor_policy"].update(
        {
            "voc_actor_policy_bundle_summary": {
                "bundle_schema_version": 1,
                "policy_version": 2,
                "terminal": True,
                "gate_schema": 7,
                "state_sha256": "f" * 64,
            },
            "voc_actor_policy_publication_count": 2,
        }
    )
    terminal = 1200
    pre_real = terminal if drain == 0 else 512
    pre_count = 4
    final_count = pre_count + drain
    full.update(
        {
            "model_real_step": terminal,
            "model_state_tensor_count": 1,
            "model_optimizer_state": {
                component: {"expected_step": final_count}
                for component in ("m", "p")
            },
            "model_scheduler_state": {
                component: {
                    "last_epoch": terminal,
                    "step_count": final_count + 1,
                }
                for component in ("m", "p")
            },
            "model_scaler_state": {},
            "model_input_seal": {
                "voc_model_input_seal_schema_version": 1,
                "voc_model_input_sealed": True,
                "voc_model_input_seal_count": 1,
                "voc_model_terminal_processed_n": terminal,
                "voc_model_terminal_drain_update_count": drain,
                "voc_model_terminal_drain_pre_real_step": pre_real,
                "voc_model_terminal_drain_pre_grad_step_count_m": pre_count,
                "voc_model_terminal_drain_pre_grad_step_count_p": pre_count,
                "voc_model_input_late_write_count": 0,
                "voc_model_input_abort_count": 0,
            },
        }
    )
    return marker, full, logger_completion


def _schema8_public_bundle_records(*, drain=1, checkpoint_dir=None):
    marker, full, logger_completion = _schema7_public_bundle_records(
        drain=drain
    )
    resolved = full["resolved_identity"]
    savedir = (
        Path(checkpoint_dir).resolve().parent
        if checkpoint_dir is not None
        else Path("/sealed/runs")
    )
    resolved.update(
        {
            "gate_schema": 8,
            "voc_gate_policy_schema_version": 8,
            "voc_q_regression_loss": "half_squared_td",
            "stage": evaluation.VOC_GATE_POLICY_SCHEMA8_STAGE_PROFILES[0],
            "paths": {
                "savedir": str(savedir),
                "ckpdir": str(
                    Path(checkpoint_dir).resolve()
                    if checkpoint_dir is not None
                    else savedir
                    / evaluation.VOC_GATE_POLICY_SCHEMA8_STAGE_PROFILES[0][0]
                ),
                "cmd": "train.py",
                "icopro_data_path": str(
                    savedir.parent / "data" / "behavioral_data_block"
                ),
            },
        }
    )
    full["actor_policy"]["voc_actor_policy_bundle_summary"] = {
        "bundle_schema_version": 1,
        "policy_version": 2,
        "terminal": True,
        "gate_schema": 8,
        "actor_state_dict_sha256": "f" * 64,
        "actor_state_dict_key_count": 1,
        "actor_state_dict_keys": ["weight"],
        "actor_state_dict_metadata": [
            {"key": "weight", "dtype": "torch.float32", "shape": [1], "numel": 1}
        ],
    }
    full["actor_policy"].update(
        {
            "voc_actor_policy_expected_ack_count": 1,
            "voc_actor_policy_terminal_ack_count": 1,
            "voc_actor_policy_version_mismatch_count": 0,
            "voc_actor_policy_malformed_bundle_count": 0,
            "voc_actor_policy_barrier_timeout_count": 0,
        }
    )
    canonical_history = json.dumps(
        list(full["actor_policy"]["voc_actor_policy_publication_history"]),
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    history_digest = hashlib.sha256(canonical_history).hexdigest()
    full["actor_policy"][
        "voc_actor_policy_publication_history_sha256"
    ] = history_digest
    logger_completion["publication_history_sha256"] = history_digest
    return marker, full, logger_completion


def _schema8_checkpoint_dir(tmp_path):
    checkpoint_dir = (
        tmp_path
        / "runs"
        / evaluation.VOC_GATE_POLICY_SCHEMA8_STAGE_PROFILES[0][0]
    )
    checkpoint_dir.mkdir(parents=True)
    return checkpoint_dir


def _schema9_public_bundle_records(*, drain=1, checkpoint_dir=None):
    marker, full, logger_completion = _schema8_public_bundle_records(
        drain=drain, checkpoint_dir=checkpoint_dir
    )
    resolved = full["resolved_identity"]
    resolved.update(
        {
            "gate_schema": 9,
            "voc_gate_policy_schema_version": 9,
            "voc_q_reconstruction": (
                "detached_value_plus_raw_head_mean_plus_"
                "policy_centered_raw_head"
            ),
            "stage": evaluation.VOC_GATE_POLICY_SCHEMA9_STAGE_PROFILES[0],
        }
    )
    full["actor_policy"]["voc_actor_policy_bundle_summary"]["gate_schema"] = 9
    return marker, full, logger_completion


def _schema9_checkpoint_dir(tmp_path):
    checkpoint_dir = (
        tmp_path
        / "runs"
        / evaluation.VOC_GATE_POLICY_SCHEMA9_STAGE_PROFILES[0][0]
    )
    checkpoint_dir.mkdir(parents=True)
    return checkpoint_dir


def _schema10_public_bundle_records(*, drain=1, checkpoint_dir=None):
    marker, full, logger_completion = _schema9_public_bundle_records(
        drain=drain, checkpoint_dir=checkpoint_dir
    )
    resolved = full["resolved_identity"]
    resolved.update(
        {
            "gate_schema": 10,
            "voc_gate_policy_schema_version": 10,
            "voc_q_regression_loss": "smooth_l1_beta1",
            "stage": evaluation.VOC_GATE_POLICY_SCHEMA10_STAGE_PROFILES[0],
        }
    )
    full["actor_policy"]["voc_actor_policy_bundle_summary"]["gate_schema"] = 10
    return marker, full, logger_completion


def _schema10_checkpoint_dir(tmp_path):
    checkpoint_dir = (
        tmp_path
        / "runs"
        / evaluation.VOC_GATE_POLICY_SCHEMA10_STAGE_PROFILES[0][0]
    )
    checkpoint_dir.mkdir(parents=True)
    return checkpoint_dir


def _schema11_public_bundle_records(*, drain=1, checkpoint_dir=None):
    marker, full, logger_completion = _schema10_public_bundle_records(
        drain=drain, checkpoint_dir=checkpoint_dir
    )
    resolved = full["resolved_identity"]
    resolved.update(
        {
            "gate_schema": 11,
            "voc_gate_policy_schema_version": 11,
            "voc_q_optimizer_coordinates": (
                "orthonormal_common_difference_adam"
            ),
            "stage": evaluation.VOC_GATE_POLICY_SCHEMA11_STAGE_PROFILES[0],
        }
    )
    full["actor_policy"]["voc_actor_policy_bundle_summary"]["gate_schema"] = 11
    actor_policy = full["actor_policy"]
    actor_policy.update(
        {
            "actor_amp_init_scale": 32.0,
            "actor_amp_scale": 32.0,
            "actor_amp_growth_tracker": 2,
            "actor_amp_skip_count": 0,
            "actor_amp_consecutive_skips": 0,
            "voc_actor_policy_publication_event_count": len(
                actor_policy["voc_actor_policy_publication_history"]
            ),
            "voc_actor_policy_final_publication_event": copy.deepcopy(
                actor_policy["voc_actor_policy_publication_history"][-1]
            ),
        }
    )
    return marker, full, logger_completion


def _schema11_checkpoint_dir(tmp_path):
    checkpoint_dir = (
        tmp_path
        / "runs"
        / evaluation.VOC_GATE_POLICY_SCHEMA11_STAGE_PROFILES[0][0]
    )
    checkpoint_dir.mkdir(parents=True)
    return checkpoint_dir


def _schema12_public_bundle_records(*, drain=1, checkpoint_dir=None):
    marker, full, logger_completion = _schema11_public_bundle_records(
        drain=drain, checkpoint_dir=checkpoint_dir
    )
    resolved = full["resolved_identity"]
    resolved.update(
        {
            "gate_schema": 12,
            "voc_gate_policy_schema_version": 12,
            "v12_projection_sha256": (
                evaluation.VOC_GATE_POLICY_SCHEMA12_V12_PROJECTION_SHA256
            ),
            "stage": evaluation.VOC_GATE_POLICY_SCHEMA12_STAGE_PROFILES[0],
        }
    )
    full["actor_policy"]["voc_actor_policy_bundle_summary"]["gate_schema"] = 12
    return marker, full, logger_completion


def _schema12_checkpoint_dir(tmp_path):
    checkpoint_dir = (
        tmp_path
        / "runs"
        / evaluation.VOC_GATE_POLICY_SCHEMA12_STAGE_PROFILES[0][0]
    )
    checkpoint_dir.mkdir(parents=True)
    return checkpoint_dir


def _write_schema12_public_actor(checkpoint_dir, marker, *, mismatch=None):
    online_weight = torch.tensor([[1.0, -0.0]], dtype=torch.float32)
    online_bias = torch.tensor([0.5, -0.0], dtype=torch.float32)
    ema_weight = online_weight.clone()
    ema_bias = online_bias.clone()
    if mismatch in ("weight", "both"):
        ema_weight[0, 0] += 1.0
    if mismatch in ("bias", "both"):
        ema_bias[0] += 1.0
    checkpoint = {
        "voc_gate_target_tau": 1.0,
        "voc_ema_gate_update_count": 1,
        "voc_ema_gate_head_state_dict": {
            "weight": ema_weight,
            "bias": ema_bias,
        },
        "actor_net_state_dict": {
            "voc_head.weight": online_weight,
            "voc_head.bias": online_bias,
        },
    }
    actor_path = checkpoint_dir / "ckp_actor.tar"
    torch.save(checkpoint, actor_path)
    record = {
        "sha256": hashlib.sha256(actor_path.read_bytes()).hexdigest(),
        "size": actor_path.stat().st_size,
    }
    marker["checkpoint_files"]["ckp_actor.tar"] = record
    if "voc_actor_policy_logger_completion" in marker:
        marker["voc_actor_policy_logger_completion"]["checkpoint_files"][
            "ckp_actor.tar"
        ] = record


def _schema13_public_bundle_records(*, checkpoint_dir=None, mismatch=None):
    marker, full, logger_completion = _schema12_public_bundle_records(
        checkpoint_dir=checkpoint_dir
    )
    resolved = full["resolved_identity"]
    resolved.update(
        {
            "gate_schema": 13,
            "voc_gate_policy_schema_version": 13,
            "stage": evaluation.VOC_GATE_POLICY_SCHEMA13_STAGE_PROFILES[0],
        }
    )
    full["actor_policy"]["voc_actor_policy_bundle_summary"]["gate_schema"] = 13
    manifest_record = {"sha256": "9" * 64, "size": 4096}
    marker["schema_version"] = 2
    marker["checkpoint_files"]["voc_telemetry_manifest.json"] = manifest_record
    logger_completion["schema_version"] = 2
    logger_completion["checkpoint_files"][
        "voc_telemetry_manifest.json"
    ] = manifest_record
    full["completion_evidence"]["checkpoint_files"][
        "voc_telemetry_manifest.json"
    ] = manifest_record
    telemetry = {
        "telemetry_schema_version": 1,
        "gate_schema": 13,
        "manifest_name": "voc_telemetry_manifest.json",
        "manifest_sha256": manifest_record["sha256"],
        "manifest_size": manifest_record["size"],
        "transaction_count": full["actor_training_state"]["voc_update_count"],
        "terminal_policy_version": full["actor_policy"][
            "voc_actor_policy_version"
        ],
        "terminal_real_step": 1200,
        "actor_state_sha256": full["actor_policy"][
            "voc_actor_policy_state_sha256"
        ],
        "publication_history_sha256": full["actor_policy"][
            "voc_actor_policy_publication_history_sha256"
        ],
    }
    full["telemetry"] = telemetry
    if mismatch is not None:
        telemetry[mismatch] = (
            "8" * 64 if mismatch.endswith("sha256") else telemetry[mismatch] + 1
        )
    return marker, full, logger_completion, telemetry


def _schema13_checkpoint_dir(tmp_path):
    checkpoint_dir = (
        tmp_path
        / "runs"
        / evaluation.VOC_GATE_POLICY_SCHEMA13_STAGE_PROFILES[0][0]
    )
    checkpoint_dir.mkdir(parents=True)
    return checkpoint_dir


def test_schema6_completed_bundle_records_full_json_safe_evidence(
    monkeypatch, tmp_path
):
    checkpoint_dir = tmp_path / "run"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "config_c.yaml").write_text(
        "voc_gate_policy_schema_version: 6\n", encoding="utf-8"
    )
    marker, full, logger_completion = _schema6_public_bundle_records()
    monkeypatch.setattr(
        evaluation, "validate_completion_marker", lambda path: marker
    )
    monkeypatch.setattr(
        util, "validate_schema6_final_bundle", lambda path, label: full
    )
    monkeypatch.setattr(
        util,
        "validate_actor_policy_logger_completion",
        lambda value: logger_completion if value == logger_completion else None,
    )

    record = evaluation.validate_schema6_completed_bundle(checkpoint_dir)

    assert record["public_finish_verified"] is True
    assert record["private_logger_markers_absent"] is True
    assert record["logger_completion"] == logger_completion
    assert set(record["stored_surface_identity"]) == {
        "config",
        "actor_checkpoint",
        "model_checkpoint",
    }
    assert all(
        value["key_count"] == 228
        for value in record["stored_surface_identity"].values()
    )
    json.dumps(record, sort_keys=True, allow_nan=False)


@pytest.mark.parametrize(
    "private_name",
    (
        util.VOC_ACTOR_POLICY_LOGGER_FINISH_REQUEST_FILE,
        util.VOC_ACTOR_POLICY_LOGGER_FINISH_ACK_FILE,
    ),
)
def test_schema6_completed_bundle_rejects_private_logger_markers(
    monkeypatch, tmp_path, private_name
):
    checkpoint_dir = tmp_path / "run"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "config_c.yaml").write_text(
        "voc_gate_policy_schema_version: 6\n", encoding="utf-8"
    )
    (checkpoint_dir / private_name).write_text("forensic\n", encoding="utf-8")
    marker, full, logger_completion = _schema6_public_bundle_records()
    monkeypatch.setattr(
        evaluation, "validate_completion_marker", lambda path: marker
    )
    monkeypatch.setattr(
        util, "validate_schema6_final_bundle", lambda path, label: full
    )
    monkeypatch.setattr(
        util, "validate_actor_policy_logger_completion", lambda value: logger_completion
    )

    with pytest.raises(ValueError, match="private logger marker"):
        evaluation.validate_schema6_completed_bundle(checkpoint_dir)


def test_schema6_completed_bundle_rejects_non_json_public_record(
    monkeypatch, tmp_path
):
    checkpoint_dir = tmp_path / "run"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "config_c.yaml").write_text(
        "voc_gate_policy_schema_version: 6\n", encoding="utf-8"
    )
    marker, full, logger_completion = _schema6_public_bundle_records()
    full["forbidden_tensor"] = torch.zeros(1)
    monkeypatch.setattr(
        evaluation, "validate_completion_marker", lambda path: marker
    )
    monkeypatch.setattr(
        util, "validate_schema6_final_bundle", lambda path, label: full
    )
    monkeypatch.setattr(
        util, "validate_actor_policy_logger_completion", lambda value: logger_completion
    )

    with pytest.raises(ValueError, match="not strict JSON-safe"):
        evaluation.validate_schema6_completed_bundle(checkpoint_dir)


def test_schema6_completed_bundle_rejects_logger_terminal_mismatch(
    monkeypatch, tmp_path
):
    checkpoint_dir = tmp_path / "run"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "config_c.yaml").write_text(
        "voc_gate_policy_schema_version: 6\n", encoding="utf-8"
    )
    marker, full, logger_completion = _schema6_public_bundle_records()
    mismatched = dict(logger_completion)
    mismatched["policy_version"] = 1
    marker["voc_actor_policy_logger_completion"] = mismatched
    monkeypatch.setattr(
        evaluation, "validate_completion_marker", lambda path: marker
    )
    monkeypatch.setattr(
        util, "validate_schema6_final_bundle", lambda path, label: full
    )
    monkeypatch.setattr(
        util,
        "validate_actor_policy_logger_completion",
        lambda value: dict(value),
    )

    with pytest.raises(ValueError, match="logger completion disagrees"):
        evaluation.validate_schema6_completed_bundle(checkpoint_dir)


def test_schema6_completed_bundle_rejects_public_finish_evidence_mismatch(
    monkeypatch, tmp_path
):
    checkpoint_dir = tmp_path / "run"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "config_c.yaml").write_text(
        "voc_gate_policy_schema_version: 6\n", encoding="utf-8"
    )
    marker, full, logger_completion = _schema6_public_bundle_records()
    full["completion_evidence"] = copy.deepcopy(full["completion_evidence"])
    full["completion_evidence"]["checkpoint_files"]["ckp_actor.tar"] = {
        "sha256": "9" * 64,
        "size": 11,
    }
    monkeypatch.setattr(
        evaluation, "validate_completion_marker", lambda path: marker
    )
    monkeypatch.setattr(
        util, "validate_schema6_final_bundle", lambda path, label: full
    )
    monkeypatch.setattr(
        util,
        "validate_actor_policy_logger_completion",
        lambda value: logger_completion,
    )

    with pytest.raises(ValueError, match="finish evidence disagrees"):
        evaluation.validate_schema6_completed_bundle(checkpoint_dir)


def test_schema6_evaluation_runtime_copy_preserves_immutable_identity():
    training = SimpleNamespace(
        voc_gate_policy_schema_version=6,
        train_actor=True,
        parallel=True,
        parallel_actor=True,
        use_wandb=True,
        voc_actor_policy_barrier_runtime=True,
        voc_train_epsilon=0.02,
        voc_gate_execution_epsilon=0.25,
    )

    runtime, record = evaluation.evaluation_runtime_flags(training)

    assert training.train_actor is True
    assert training.parallel_actor is True
    assert training.voc_actor_policy_barrier_runtime is True
    assert runtime is not training
    assert runtime.train_actor is False
    assert runtime.parallel is False
    assert runtime.parallel_actor is False
    assert runtime.use_wandb is False
    assert runtime.voc_actor_policy_barrier_runtime is False
    assert runtime.voc_train_epsilon == 0.02
    assert runtime.voc_gate_execution_epsilon == 0.25
    assert record["immutable_training"]["voc_train_epsilon"] == 0.02
    assert record["immutable_training"]["voc_gate_execution_epsilon"] == 0.25
    assert record["evaluation_copy"]["effective_soft_gate_epsilon"] == 0.0
    assert record["evaluation_copy"]["effective_execution_gate_epsilon"] == 0.0


@pytest.mark.parametrize("drain", [0, 1])
def test_schema7_completed_bundle_records_json_safe_model_seal(
    monkeypatch, tmp_path, drain
):
    checkpoint_dir = tmp_path / "run"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "config_c.yaml").write_text(
        "voc_gate_policy_schema_version: 7\n", encoding="utf-8"
    )
    marker, full, logger_completion = _schema7_public_bundle_records(
        drain=drain
    )
    monkeypatch.setattr(
        evaluation, "validate_completion_marker", lambda path: marker
    )
    monkeypatch.setattr(
        util, "validate_schema7_final_bundle", lambda path, label: full
    )
    monkeypatch.setattr(
        util,
        "validate_actor_policy_logger_completion",
        lambda value: logger_completion,
    )

    record = evaluation.validate_schema7_completed_bundle(checkpoint_dir)

    assert record["authoritative_validator"] == (
        "thinker.util.validate_schema7_final_bundle"
    )
    assert record["resolved_identity"]["key_count"] == 229
    assert record["model_input_seal"][
        "voc_model_terminal_drain_update_count"
    ] == drain
    assert record["private_logger_markers_absent"] is True
    json.dumps(record, sort_keys=True, allow_nan=False)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("voc_model_input_sealed", 1, "must be sealed"),
        ("voc_model_input_seal_count", True, "must be Python int"),
        ("voc_model_input_late_write_count", 1, "late input write"),
        ("voc_model_input_abort_count", 1, "was aborted"),
        ("voc_model_terminal_drain_update_count", 2, "zero or one"),
    ],
)
def test_schema7_completed_bundle_rejects_model_seal_mutations(
    monkeypatch, tmp_path, field, value, error
):
    checkpoint_dir = tmp_path / "run"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "config_c.yaml").write_text(
        "voc_gate_policy_schema_version: 7\n", encoding="utf-8"
    )
    marker, full, logger_completion = _schema7_public_bundle_records()
    full["model_input_seal"][field] = value
    monkeypatch.setattr(
        evaluation, "validate_completion_marker", lambda path: marker
    )
    monkeypatch.setattr(
        util, "validate_schema7_final_bundle", lambda path, label: full
    )
    monkeypatch.setattr(
        util,
        "validate_actor_policy_logger_completion",
        lambda value: logger_completion,
    )

    with pytest.raises(ValueError, match=error):
        evaluation.validate_schema7_completed_bundle(checkpoint_dir)


def test_schema7_evaluation_runtime_copy_disables_only_live_coordination():
    training = SimpleNamespace(
        voc_gate_policy_schema_version=7,
        voc_model_input_seal_schema_version=1,
        train_actor=True,
        train_model=True,
        parallel=True,
        parallel_actor=True,
        use_wandb=True,
        voc_actor_policy_barrier_runtime=True,
        voc_train_epsilon=0.02,
        voc_gate_execution_epsilon=0.25,
    )

    runtime, record = evaluation.evaluation_runtime_flags(training)

    assert training.train_model is True
    assert training.voc_model_input_seal_schema_version == 1
    assert runtime.train_model is False
    assert runtime.voc_model_input_seal_schema_version == 1
    assert record["immutable_training"]["train_model"] is True
    assert record["immutable_training"][
        "voc_model_input_seal_schema_version"
    ] == 1
    assert record["evaluation_copy"]["train_model"] is False
    assert record["evaluation_copy"][
        "effective_model_input_seal_coordination"
    ] is False


@pytest.mark.parametrize("drain", [0, 1])
def test_schema8_completed_bundle_records_half_squared_identity(
    monkeypatch, tmp_path, drain
):
    checkpoint_dir = _schema8_checkpoint_dir(tmp_path)
    (checkpoint_dir / "config_c.yaml").write_text(
        "voc_gate_policy_schema_version: 8\n", encoding="utf-8"
    )
    marker, full, logger_completion = _schema8_public_bundle_records(
        drain=drain, checkpoint_dir=checkpoint_dir
    )
    monkeypatch.setattr(
        evaluation, "validate_completion_marker", lambda path: marker
    )
    monkeypatch.setattr(
        util, "validate_schema8_final_bundle", lambda path, label: full
    )
    monkeypatch.setattr(
        util,
        "validate_actor_policy_logger_completion",
        lambda value: logger_completion,
    )

    record = evaluation.validate_schema8_completed_bundle(checkpoint_dir)

    assert record["authoritative_validator"] == (
        "thinker.util.validate_schema8_final_bundle"
    )
    assert record["resolved_identity"]["voc_q_regression_loss"] == (
        "half_squared_td"
    )
    assert record["resolved_identity"]["key_count"] == 229
    assert record["model_input_seal"][
        "voc_model_terminal_drain_update_count"
    ] == drain
    assert record["stored_surface_identity"]["config"] == (
        record["resolved_identity"]
    )
    json.dumps(record, sort_keys=True, allow_nan=False)


@pytest.mark.parametrize("drain", [0, 1])
def test_schema9_completed_bundle_records_exact_common_mode_identity(
    monkeypatch, tmp_path, drain
):
    checkpoint_dir = _schema9_checkpoint_dir(tmp_path)
    config_payload = b"voc_gate_policy_schema_version: 9\n"
    (checkpoint_dir / "config_c.yaml").write_bytes(config_payload)
    config_digest = hashlib.sha256(config_payload).hexdigest()
    marker, full, logger_completion = _schema9_public_bundle_records(
        drain=drain, checkpoint_dir=checkpoint_dir
    )
    monkeypatch.setattr(
        evaluation, "validate_completion_marker", lambda path: marker
    )
    monkeypatch.setattr(
        util, "validate_schema9_final_bundle", lambda path, label: full
    )
    monkeypatch.setattr(
        util,
        "validate_actor_policy_logger_completion",
        lambda value: logger_completion,
    )

    record = evaluation.validate_schema9_completed_bundle(
        checkpoint_dir,
        config_payload=config_payload,
        expected_config_sha256=config_digest,
    )

    assert record["authoritative_validator"] == (
        "thinker.util.validate_schema9_final_bundle"
    )
    assert set(record) == {
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
    assert set(record["resolved_identity"]) == {
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
    assert record["resolved_identity"]["voc_q_regression_loss"] == (
        "half_squared_td"
    )
    assert record["resolved_identity"]["voc_q_reconstruction"] == (
        "detached_value_plus_raw_head_mean_plus_policy_centered_raw_head"
    )
    assert record["model_input_seal"][
        "voc_model_terminal_drain_update_count"
    ] == drain
    assert "voc_q_reconstruction" not in (
        _schema8_public_bundle_records()[1]["resolved_identity"]
    )
    json.dumps(record, sort_keys=True, allow_nan=False)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("voc_q_regression_loss", None),
        ("voc_q_regression_loss", "smooth_l1"),
        ("voc_q_reconstruction", None),
        ("voc_q_reconstruction", "policy_centered_raw_head"),
    ],
)
def test_schema9_completed_bundle_rejects_missing_or_wrong_derived_identity(
    monkeypatch, tmp_path, field, value
):
    checkpoint_dir = _schema9_checkpoint_dir(tmp_path)
    (checkpoint_dir / "config_c.yaml").write_text(
        "voc_gate_policy_schema_version: 9\n", encoding="utf-8"
    )
    marker, full, logger_completion = _schema9_public_bundle_records(
        checkpoint_dir=checkpoint_dir
    )
    if value is None:
        full["resolved_identity"].pop(field)
    else:
        full["resolved_identity"][field] = value
    monkeypatch.setattr(
        evaluation, "validate_completion_marker", lambda path: marker
    )
    monkeypatch.setattr(
        util, "validate_schema9_final_bundle", lambda path, label: full
    )
    monkeypatch.setattr(
        util,
        "validate_actor_policy_logger_completion",
        lambda value: logger_completion,
    )

    with pytest.raises(ValueError, match="resolved identity|regression|reconstruction"):
        evaluation.validate_schema9_completed_bundle(checkpoint_dir)


@pytest.mark.parametrize("yaml_value", ["8", "10", "true", "'9'", "9.0"])
def test_dedicated_schema9_completed_route_rejects_other_schema_types(
    tmp_path, yaml_value
):
    checkpoint_dir = _schema9_checkpoint_dir(tmp_path)
    (checkpoint_dir / "config_c.yaml").write_text(
        f"voc_gate_policy_schema_version: {yaml_value}\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="exact Python integer"):
        evaluation.validate_schema9_completed_bundle(checkpoint_dir)


@pytest.mark.parametrize("drain", [0, 1])
def test_schema10_completed_bundle_preserves_schema9_record_shape_with_huber_identity(
    monkeypatch, tmp_path, drain
):
    checkpoint_dir = _schema10_checkpoint_dir(tmp_path)
    config_payload = b"voc_gate_policy_schema_version: 10\n"
    (checkpoint_dir / "config_c.yaml").write_bytes(config_payload)
    config_digest = hashlib.sha256(config_payload).hexdigest()
    marker, full, logger_completion = _schema10_public_bundle_records(
        drain=drain, checkpoint_dir=checkpoint_dir
    )
    monkeypatch.setattr(
        evaluation, "validate_completion_marker", lambda path: marker
    )
    monkeypatch.setattr(
        util, "validate_schema10_final_bundle", lambda path, label: full
    )
    monkeypatch.setattr(
        util,
        "validate_actor_policy_logger_completion",
        lambda value: logger_completion,
    )

    record = evaluation.validate_schema10_completed_bundle(
        checkpoint_dir,
        config_payload=config_payload,
        expected_config_sha256=config_digest,
    )

    schema9_marker, schema9_full, _ = _schema9_public_bundle_records(
        checkpoint_dir=checkpoint_dir
    )
    del schema9_marker
    assert set(record) == {
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
    assert set(record["resolved_identity"]) == set(
        schema9_full["resolved_identity"]
    )
    assert record["authoritative_validator"] == (
        "thinker.util.validate_schema10_final_bundle"
    )
    assert record["resolved_identity"]["voc_q_regression_loss"] == (
        "smooth_l1_beta1"
    )
    assert record["resolved_identity"]["voc_q_reconstruction"] == (
        "detached_value_plus_raw_head_mean_plus_policy_centered_raw_head"
    )
    assert record["model_input_seal"][
        "voc_model_terminal_drain_update_count"
    ] == drain
    json.dumps(record, sort_keys=True, allow_nan=False)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("wrong_loss", "smooth_l1_beta1"),
        ("missing_reconstruction", "resolved identity"),
        ("extra_identity", "resolved identity"),
        ("malformed_metadata", "metadata"),
        ("malformed_history", "history"),
        ("extra_authoritative_field", "invalid fields"),
    ],
)
def test_schema10_completed_bundle_rejects_identity_shape_and_forged_evidence(
    monkeypatch, tmp_path, mutation, error
):
    checkpoint_dir = _schema10_checkpoint_dir(tmp_path)
    (checkpoint_dir / "config_c.yaml").write_text(
        "voc_gate_policy_schema_version: 10\n", encoding="utf-8"
    )
    marker, full, logger_completion = _schema10_public_bundle_records(
        checkpoint_dir=checkpoint_dir
    )
    if mutation == "wrong_loss":
        full["resolved_identity"]["voc_q_regression_loss"] = "half_squared_td"
    elif mutation == "missing_reconstruction":
        full["resolved_identity"].pop("voc_q_reconstruction")
    elif mutation == "extra_identity":
        full["resolved_identity"]["forged"] = True
    elif mutation == "malformed_metadata":
        full["actor_policy"]["voc_actor_policy_bundle_summary"][
            "actor_state_dict_metadata"
        ][0]["numel"] = 2
    elif mutation == "malformed_history":
        full["actor_policy"]["voc_actor_policy_publication_history"][0][
            "forged"
        ] = True
    elif mutation == "extra_authoritative_field":
        full["forged"] = True
    monkeypatch.setattr(
        evaluation, "validate_completion_marker", lambda path: marker
    )
    monkeypatch.setattr(
        util, "validate_schema10_final_bundle", lambda path, label: full
    )
    monkeypatch.setattr(
        util,
        "validate_actor_policy_logger_completion",
        lambda value: logger_completion,
    )

    with pytest.raises(ValueError, match=error):
        evaluation.validate_schema10_completed_bundle(checkpoint_dir)


@pytest.mark.parametrize("yaml_value", ["9", "11", "true", "'10'", "10.0"])
def test_dedicated_schema10_completed_route_rejects_other_schema_types(
    tmp_path, yaml_value
):
    checkpoint_dir = _schema10_checkpoint_dir(tmp_path)
    (checkpoint_dir / "config_c.yaml").write_text(
        f"voc_gate_policy_schema_version: {yaml_value}\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="exact Python integer"):
        evaluation.validate_schema10_completed_bundle(checkpoint_dir)


@pytest.mark.parametrize(
    "private_name",
    (
        util.VOC_ACTOR_POLICY_LOGGER_FINISH_REQUEST_FILE,
        util.VOC_ACTOR_POLICY_LOGGER_FINISH_ACK_FILE,
    ),
)
def test_schema10_completed_bundle_rejects_private_logger_markers(
    monkeypatch, tmp_path, private_name
):
    checkpoint_dir = _schema10_checkpoint_dir(tmp_path)
    (checkpoint_dir / "config_c.yaml").write_text(
        "voc_gate_policy_schema_version: 10\n", encoding="utf-8"
    )
    (checkpoint_dir / private_name).write_text("forged\n", encoding="utf-8")
    marker, full, logger_completion = _schema10_public_bundle_records(
        checkpoint_dir=checkpoint_dir
    )
    monkeypatch.setattr(
        evaluation, "validate_completion_marker", lambda path: marker
    )
    monkeypatch.setattr(
        util, "validate_schema10_final_bundle", lambda path, label: full
    )
    monkeypatch.setattr(
        util,
        "validate_actor_policy_logger_completion",
        lambda value: logger_completion,
    )

    with pytest.raises(ValueError, match="private logger marker"):
        evaluation.validate_schema10_completed_bundle(checkpoint_dir)


@pytest.mark.parametrize("drain", [0, 1])
def test_schema11_completed_bundle_adds_only_optimizer_identity(
    monkeypatch, tmp_path, drain
):
    checkpoint_dir = _schema11_checkpoint_dir(tmp_path)
    config_payload = b"voc_gate_policy_schema_version: 11\n"
    (checkpoint_dir / "config_c.yaml").write_bytes(config_payload)
    config_digest = hashlib.sha256(config_payload).hexdigest()
    marker, full, logger_completion = _schema11_public_bundle_records(
        drain=drain, checkpoint_dir=checkpoint_dir
    )
    monkeypatch.setattr(
        evaluation, "validate_completion_marker", lambda path: marker
    )
    monkeypatch.setattr(
        util, "validate_schema11_final_bundle", lambda path, label: full
    )
    monkeypatch.setattr(
        util,
        "validate_actor_policy_logger_completion",
        lambda value: logger_completion,
    )

    record = evaluation.validate_schema11_completed_bundle(
        checkpoint_dir,
        config_payload=config_payload,
        expected_config_sha256=config_digest,
    )

    _, schema10_full, _ = _schema10_public_bundle_records(
        checkpoint_dir=checkpoint_dir
    )
    assert set(record["resolved_identity"]) == (
        set(schema10_full["resolved_identity"])
        | {"voc_q_optimizer_coordinates"}
    )
    assert record["authoritative_validator"] == (
        "thinker.util.validate_schema11_final_bundle"
    )
    assert record["resolved_identity"]["gate_schema"] == 11
    assert record["resolved_identity"][
        "voc_q_optimizer_coordinates"
    ] == "orthonormal_common_difference_adam"
    assert record["model_input_seal"][
        "voc_model_terminal_drain_update_count"
    ] == drain
    assert set(record["actor_policy"]) == (
        evaluation.ATOMIC_ACTOR_POLICY_EVIDENCE_FIELDS
    )
    assert not {
        "voc_q_regression_loss",
        "voc_q_reconstruction",
        "voc_q_optimizer_coordinates",
    } & set(record["actor_policy"])
    assert all(
        set(identity) == set(record["resolved_identity"])
        and identity == record["resolved_identity"]
        for identity in record["stored_surface_identity"].values()
    )
    json.dumps(record, sort_keys=True, allow_nan=False)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("voc_q_regression_loss", "smooth_l1_beta1"),
        ("voc_q_regression_loss", None),
        (
            "voc_q_reconstruction",
            "detached_value_plus_raw_head_mean_plus_policy_centered_raw_head",
        ),
        ("voc_q_reconstruction", "forged"),
        (
            "voc_q_optimizer_coordinates",
            "orthonormal_common_difference_adam",
        ),
        ("voc_q_optimizer_coordinates", None),
        ("forged_actor_policy_identity", True),
    ],
)
def test_schema11_completed_bundle_rejects_lifecycle_actor_identity_additions(
    monkeypatch, tmp_path, field, value
):
    checkpoint_dir = _schema11_checkpoint_dir(tmp_path)
    config_payload = b"voc_gate_policy_schema_version: 11\n"
    (checkpoint_dir / "config_c.yaml").write_bytes(config_payload)
    marker, full, logger_completion = _schema11_public_bundle_records(
        checkpoint_dir=checkpoint_dir
    )
    full["actor_policy"][field] = value
    monkeypatch.setattr(
        evaluation, "validate_completion_marker", lambda path: marker
    )
    monkeypatch.setattr(
        util, "validate_schema11_final_bundle", lambda path, label: full
    )
    monkeypatch.setattr(
        util,
        "validate_actor_policy_logger_completion",
        lambda value: logger_completion,
    )

    with pytest.raises(ValueError, match="exact schema-10 lifecycle keyset"):
        evaluation.validate_schema11_completed_bundle(
            checkpoint_dir,
            config_payload=config_payload,
            expected_config_sha256=hashlib.sha256(config_payload).hexdigest(),
        )


def test_schema11_completed_bundle_rejects_missing_lifecycle_actor_field(
    monkeypatch, tmp_path
):
    checkpoint_dir = _schema11_checkpoint_dir(tmp_path)
    config_payload = b"voc_gate_policy_schema_version: 11\n"
    (checkpoint_dir / "config_c.yaml").write_bytes(config_payload)
    marker, full, logger_completion = _schema11_public_bundle_records(
        checkpoint_dir=checkpoint_dir
    )
    full["actor_policy"].pop("actor_amp_scale")
    monkeypatch.setattr(
        evaluation, "validate_completion_marker", lambda path: marker
    )
    monkeypatch.setattr(
        util, "validate_schema11_final_bundle", lambda path, label: full
    )
    monkeypatch.setattr(
        util,
        "validate_actor_policy_logger_completion",
        lambda value: logger_completion,
    )

    with pytest.raises(ValueError, match="exact schema-10 lifecycle keyset"):
        evaluation.validate_schema11_completed_bundle(
            checkpoint_dir,
            config_payload=config_payload,
            expected_config_sha256=hashlib.sha256(config_payload).hexdigest(),
        )


@pytest.mark.parametrize("mutation", ["missing", "wrong", "extra"])
def test_schema11_completed_bundle_rejects_forged_optimizer_identity(
    monkeypatch, tmp_path, mutation
):
    checkpoint_dir = _schema11_checkpoint_dir(tmp_path)
    (checkpoint_dir / "config_c.yaml").write_text(
        "voc_gate_policy_schema_version: 11\n", encoding="utf-8"
    )
    marker, full, logger_completion = _schema11_public_bundle_records(
        checkpoint_dir=checkpoint_dir
    )
    if mutation == "missing":
        full["resolved_identity"].pop("voc_q_optimizer_coordinates")
    elif mutation == "wrong":
        full["resolved_identity"]["voc_q_optimizer_coordinates"] = (
            "raw_continue_stop_adam"
        )
    else:
        full["resolved_identity"]["forged_optimizer_state"] = True
    monkeypatch.setattr(
        evaluation, "validate_completion_marker", lambda path: marker
    )
    monkeypatch.setattr(
        util, "validate_schema11_final_bundle", lambda path, label: full
    )
    monkeypatch.setattr(
        util,
        "validate_actor_policy_logger_completion",
        lambda value: logger_completion,
    )

    with pytest.raises(ValueError, match="resolved|optimizer"):
        evaluation.validate_schema11_completed_bundle(checkpoint_dir)


@pytest.mark.parametrize("yaml_value", ["10", "12", "true", "'11'", "11.0"])
def test_dedicated_schema11_completed_route_rejects_other_schema_types(
    tmp_path, yaml_value
):
    checkpoint_dir = _schema11_checkpoint_dir(tmp_path)
    (checkpoint_dir / "config_c.yaml").write_text(
        f"voc_gate_policy_schema_version: {yaml_value}\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="exact Python integer"):
        evaluation.validate_schema11_completed_bundle(checkpoint_dir)


@pytest.mark.parametrize("drain", [0, 1])
def test_schema12_completed_bundle_preserves_schema11_keysets_and_tau_identity(
    monkeypatch, tmp_path, drain
):
    checkpoint_dir = _schema12_checkpoint_dir(tmp_path)
    config_payload = b"voc_gate_policy_schema_version: 12\nvoc_gate_target_tau: 1.0\n"
    (checkpoint_dir / "config_c.yaml").write_bytes(config_payload)
    marker, full, logger_completion = _schema12_public_bundle_records(
        drain=drain, checkpoint_dir=checkpoint_dir
    )
    _write_schema12_public_actor(checkpoint_dir, marker)
    monkeypatch.setattr(
        evaluation, "validate_completion_marker", lambda path: marker
    )
    monkeypatch.setattr(
        util, "validate_schema12_final_bundle", lambda path, label: full
    )
    monkeypatch.setattr(
        util,
        "validate_actor_policy_logger_completion",
        lambda value: logger_completion,
    )

    record = evaluation.validate_schema12_completed_bundle(
        checkpoint_dir,
        config_payload=config_payload,
        expected_config_sha256=hashlib.sha256(config_payload).hexdigest(),
    )
    _, schema11_full, _ = _schema11_public_bundle_records(
        checkpoint_dir=checkpoint_dir
    )
    assert set(record) == {
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
    assert set(record["resolved_identity"]) == set(
        schema11_full["resolved_identity"]
    )
    assert record["authoritative_validator"] == (
        "thinker.util.validate_schema12_final_bundle"
    )
    assert record["resolved_identity"]["gate_schema"] == 12
    assert record["resolved_identity"]["v12_projection_sha256"] == (
        evaluation.VOC_GATE_POLICY_SCHEMA12_V12_PROJECTION_SHA256
    )
    assert set(record["actor_policy"]) == (
        evaluation.ATOMIC_ACTOR_POLICY_EVIDENCE_FIELDS
    )
    assert all(
        identity == record["resolved_identity"]
        for identity in record["stored_surface_identity"].values()
    )


@pytest.mark.parametrize("mismatch", ["weight", "bias", "both"])
def test_schema12_public_rejects_each_raw_ema_online_mismatch(
    monkeypatch, tmp_path, mismatch
):
    checkpoint_dir = _schema12_checkpoint_dir(tmp_path)
    config_payload = b"voc_gate_policy_schema_version: 12\nvoc_gate_target_tau: 1.0\n"
    (checkpoint_dir / "config_c.yaml").write_bytes(config_payload)
    marker, full, logger_completion = _schema12_public_bundle_records(
        checkpoint_dir=checkpoint_dir
    )
    _write_schema12_public_actor(checkpoint_dir, marker, mismatch=mismatch)
    monkeypatch.setattr(
        evaluation, "validate_completion_marker", lambda path: marker
    )
    monkeypatch.setattr(
        util, "validate_schema12_final_bundle", lambda path, label: full
    )
    monkeypatch.setattr(
        util,
        "validate_actor_policy_logger_completion",
        lambda value: logger_completion,
    )

    with pytest.raises(ValueError, match="raw EMA (weight|bias)"):
        evaluation.validate_schema12_completed_bundle(
            checkpoint_dir,
            config_payload=config_payload,
            expected_config_sha256=hashlib.sha256(config_payload).hexdigest(),
        )


def test_schema12_public_uses_torch_equal_not_byte_equality(tmp_path):
    checkpoint_dir = _schema12_checkpoint_dir(tmp_path)
    marker = {"checkpoint_files": {}}
    _write_schema12_public_actor(checkpoint_dir, marker)
    actor_path = checkpoint_dir / "ckp_actor.tar"
    checkpoint = torch.load(actor_path, map_location="cpu", weights_only=False)
    checkpoint["voc_ema_gate_head_state_dict"]["weight"][0, 1] = 0.0
    checkpoint["voc_ema_gate_head_state_dict"]["bias"][1] = 0.0
    torch.save(checkpoint, actor_path)
    marker["checkpoint_files"]["ckp_actor.tar"] = {
        "sha256": hashlib.sha256(actor_path.read_bytes()).hexdigest(),
        "size": actor_path.stat().st_size,
    }

    evaluation._require_schema12_public_ema_online_equality(
        checkpoint_dir, marker
    )


@pytest.mark.parametrize(
    "schema_line",
    [
        pytest.param(b"voc_gate_policy_schema_version: 5\n", id="wrong-schema"),
        pytest.param(b"", id="missing-schema"),
    ],
)
def test_malformed_v19_prefix_dispatches_schema12_before_legacy_route(
    monkeypatch, tmp_path, schema_line
):
    checkpoint_dir = tmp_path / "malformed-v19-prefix"
    checkpoint_dir.mkdir()
    payload = schema_line + (
        b"xpid: enduro-voc-v19-tau1-orthocd-adam-eps25-malformed\n"
    )
    (checkpoint_dir / "config_c.yaml").write_bytes(payload)
    events = []

    def reject(*args, **kwargs):
        events.append("schema12")
        raise ValueError("invalid schema-12 intent")

    monkeypatch.setattr(
        evaluation, "validate_schema12_completed_bundle", reject
    )
    with pytest.raises(ValueError, match="invalid schema-12 intent"):
        evaluation.dispatch_schema12_completed_bundle(
            checkpoint_dir,
            config_payload=payload,
            expected_config_sha256=hashlib.sha256(payload).hexdigest(),
        )
    assert events == ["schema12"]


def test_schema12_public_missing_authoritative_api_fails_before_tensor_load(
    monkeypatch, tmp_path
):
    checkpoint_dir = _schema12_checkpoint_dir(tmp_path)
    config_payload = b"voc_gate_policy_schema_version: 12\nvoc_gate_target_tau: 1.0\n"
    (checkpoint_dir / "config_c.yaml").write_bytes(config_payload)
    marker, _, _ = _schema12_public_bundle_records(
        checkpoint_dir=checkpoint_dir
    )
    monkeypatch.delattr(util, "validate_schema12_final_bundle", raising=True)
    monkeypatch.setattr(
        torch,
        "load",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("tensor load ran before missing schema-12 API failure")
        ),
    )

    with pytest.raises(AttributeError, match="validate_schema12_final_bundle"):
        evaluation.validate_schema12_completed_bundle(
            checkpoint_dir,
            completion_state=marker,
            config_payload=config_payload,
            expected_config_sha256=hashlib.sha256(config_payload).hexdigest(),
        )


def test_schema13_completed_bundle_adds_only_telemetry_to_public_shape(
    monkeypatch, tmp_path
):
    checkpoint_dir = _schema13_checkpoint_dir(tmp_path)
    config_payload = b"voc_gate_policy_schema_version: 13\n"
    (checkpoint_dir / "config_c.yaml").write_bytes(config_payload)
    marker, full, logger_completion, telemetry = _schema13_public_bundle_records(
        checkpoint_dir=checkpoint_dir
    )
    _write_schema12_public_actor(checkpoint_dir, marker)
    monkeypatch.setattr(
        evaluation,
        "_validate_schema13_completion_marker_state",
        lambda root, state: marker,
    )
    monkeypatch.setattr(
        evaluation, "validate_schema13_completion_marker", lambda root: marker
    )
    monkeypatch.setattr(
        util, "validate_schema13_final_bundle", lambda path, label: full
    )
    monkeypatch.setattr(
        util,
        "validate_schema13_telemetry_manifest",
        lambda *args, **kwargs: telemetry,
    )
    monkeypatch.setattr(
        util,
        "validate_actor_policy_logger_completion",
        lambda value: logger_completion,
    )

    record = evaluation.validate_schema13_completed_bundle(
        checkpoint_dir,
        completion_state=marker,
        config_payload=config_payload,
        expected_config_sha256=hashlib.sha256(config_payload).hexdigest(),
    )
    assert len(record) == 18
    assert record["telemetry"] == telemetry
    assert record["authoritative_validator"] == (
        "thinker.util.validate_schema13_final_bundle"
    )
    assert record["resolved_identity"]["gate_schema"] == 13
    assert set(record["actor_policy"]) == (
        evaluation.ATOMIC_ACTOR_POLICY_EVIDENCE_FIELDS
    )


@pytest.mark.parametrize(
    "schema_line",
    [
        pytest.param(b"voc_gate_policy_schema_version: 5\n", id="wrong-schema"),
        pytest.param(b"", id="missing-schema"),
    ],
)
def test_malformed_v20_prefix_dispatches_schema13_before_legacy_route(
    monkeypatch, tmp_path, schema_line
):
    checkpoint_dir = tmp_path / "malformed-v20-prefix"
    checkpoint_dir.mkdir()
    payload = schema_line + (
        b"xpid: enduro-voc-v20-telemetry-tau1-orthocd-adam-eps25-malformed\n"
    )
    (checkpoint_dir / "config_c.yaml").write_bytes(payload)
    events = []

    def reject(*args, **kwargs):
        events.append("schema13")
        raise ValueError("invalid schema-13 intent")

    monkeypatch.setattr(
        evaluation, "validate_schema13_completed_bundle", reject
    )
    with pytest.raises(ValueError, match="invalid schema-13 intent"):
        evaluation.dispatch_schema13_completed_bundle(
            checkpoint_dir,
            config_payload=payload,
            expected_config_sha256=hashlib.sha256(payload).hexdigest(),
        )
    assert events == ["schema13"]


@pytest.mark.parametrize(
    "schema_line",
    [
        pytest.param(b"voc_gate_policy_schema_version: 5\n", id="wrong-schema"),
        pytest.param(b"", id="missing-schema"),
    ],
)
def test_malformed_v20_public_route_has_no_downstream_import_or_rng_side_effect(
    monkeypatch, tmp_path, schema_line
):
    checkpoint_dir = tmp_path / "malformed-v20"
    checkpoint_dir.mkdir()
    config_payload = schema_line + (
        b"xpid: enduro-voc-v20-telemetry-tau1-orthocd-adam-eps25-malformed\n"
    )
    (checkpoint_dir / "config_c.yaml").write_bytes(config_payload)
    config_digest = hashlib.sha256(config_payload).hexdigest()
    marker = {
        "schema_version": 2,
        "checkpoint_files": {
            "config_c.yaml": {
                "sha256": config_digest,
                "size": len(config_payload),
            }
        },
    }
    events = []

    def reject(*args, **kwargs):
        events.append("schema13_prevalidation")
        raise ValueError("invalid schema-13 intent")

    def forbidden(name):
        def fail(*args, **kwargs):
            events.append(name)
            raise AssertionError(f"{name} ran before schema-13 validation")

        return fail

    real_import = builtins.__import__
    downstream_modules = {
        "thinker.bc_loader",
        "thinker.dynamic_imitation",
    }

    def guarded_import(name, *args, **kwargs):
        if name in downstream_modules:
            return forbidden(f"downstream_import:{name}")()
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(
        evaluation, "validate_schema13_completion_marker", lambda path: marker
    )
    monkeypatch.setattr(
        evaluation,
        "_schema13_checkpoint_hashes",
        lambda path, **kwargs: {"config_c.yaml": config_digest},
    )
    monkeypatch.setattr(
        evaluation, "dispatch_schema13_completed_bundle", reject
    )
    monkeypatch.setattr(
        evaluation, "_set_pair_seed", forbidden("rng_seed")
    )
    monkeypatch.setattr(torch, "load", forbidden("tensor_load"))

    with pytest.raises(ValueError, match="invalid schema-13 intent"):
        evaluation.evaluate(
            SimpleNamespace(
                checkpoint_dir=checkpoint_dir,
                output_dir=tmp_path / "must-not-exist",
                seed=1,
            )
        )

    assert events == ["schema13_prevalidation"]
    assert not (tmp_path / "must-not-exist").exists()


def test_schema13_public_rejects_manifest_evidence_drift_before_tensor_use(
    monkeypatch, tmp_path
):
    checkpoint_dir = _schema13_checkpoint_dir(tmp_path)
    config_payload = b"voc_gate_policy_schema_version: 13\n"
    (checkpoint_dir / "config_c.yaml").write_bytes(config_payload)
    marker, full, logger_completion, telemetry = _schema13_public_bundle_records(
        checkpoint_dir=checkpoint_dir
    )
    _write_schema12_public_actor(checkpoint_dir, marker)
    monkeypatch.setattr(
        evaluation,
        "_validate_schema13_completion_marker_state",
        lambda root, state: marker,
    )
    monkeypatch.setattr(
        util, "validate_schema13_final_bundle", lambda path, label: full
    )
    monkeypatch.setattr(
        util,
        "validate_schema13_telemetry_manifest",
        lambda *args, **kwargs: {**telemetry, "manifest_size": 4097},
    )
    monkeypatch.setattr(
        util,
        "validate_actor_policy_logger_completion",
        lambda value: logger_completion,
    )
    with pytest.raises(ValueError, match="telemetry disagrees"):
        evaluation.validate_schema13_completed_bundle(
            checkpoint_dir,
            completion_state=marker,
            config_payload=config_payload,
            expected_config_sha256=hashlib.sha256(config_payload).hexdigest(),
        )


def test_schema13_checkpoint_hashes_bind_manifest_sidecars_logs_and_checkpoints(
    monkeypatch, tmp_path
):
    marker = {"schema_version": 2}
    for index, name in enumerate(evaluation.SCHEMA13_BOUND_RUN_FILES):
        (tmp_path / name).write_bytes(f"payload-{index}".encode("ascii"))
    monkeypatch.setattr(
        evaluation,
        "_validate_schema13_completion_marker_state",
        lambda root, state: marker,
    )
    monkeypatch.setattr(
        evaluation, "validate_schema13_completion_marker", lambda root: marker
    )

    hashes = evaluation._schema13_checkpoint_hashes(
        tmp_path, completion_state=marker
    )

    assert tuple(hashes) == evaluation.SCHEMA13_BOUND_RUN_FILES
    assert all(
        hashes[name] == hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        for name in evaluation.SCHEMA13_BOUND_RUN_FILES
    )


def test_schema8_completed_bundle_rejects_extra_public_record_field(
    monkeypatch, tmp_path
):
    checkpoint_dir = _schema8_checkpoint_dir(tmp_path)
    (checkpoint_dir / "config_c.yaml").write_text(
        "voc_gate_policy_schema_version: 8\n", encoding="utf-8"
    )
    marker, full, logger_completion = _schema8_public_bundle_records(
        checkpoint_dir=checkpoint_dir
    )
    full["unexpected"] = 0
    monkeypatch.setattr(
        evaluation, "validate_completion_marker", lambda path: marker
    )
    monkeypatch.setattr(
        util, "validate_schema8_final_bundle", lambda path, label: full
    )
    monkeypatch.setattr(
        util,
        "validate_actor_policy_logger_completion",
        lambda value: logger_completion,
    )

    with pytest.raises(ValueError, match="invalid fields|record has the wrong shape"):
        evaluation.validate_schema8_completed_bundle(checkpoint_dir)


@pytest.mark.parametrize(
    "yaml_value", ["7", "9", "true", "'8'", "8.0"]
)
def test_dedicated_schema8_completed_route_rejects_other_schema_types(
    tmp_path, yaml_value
):
    checkpoint_dir = _schema8_checkpoint_dir(tmp_path)
    (checkpoint_dir / "config_c.yaml").write_text(
        f"voc_gate_policy_schema_version: {yaml_value}\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="exact Python integer"):
        evaluation.validate_schema8_completed_bundle(checkpoint_dir)


@pytest.mark.parametrize("mutation", ["missing", "wrong"])
def test_schema8_completed_bundle_rejects_derived_loss_identity(
    monkeypatch, tmp_path, mutation
):
    checkpoint_dir = _schema8_checkpoint_dir(tmp_path)
    (checkpoint_dir / "config_c.yaml").write_text(
        "voc_gate_policy_schema_version: 8\n", encoding="utf-8"
    )
    marker, full, logger_completion = _schema8_public_bundle_records(
        checkpoint_dir=checkpoint_dir
    )
    if mutation == "missing":
        full["resolved_identity"].pop("voc_q_regression_loss")
    else:
        full["resolved_identity"]["voc_q_regression_loss"] = "smooth_l1"
    monkeypatch.setattr(
        evaluation, "validate_completion_marker", lambda path: marker
    )
    monkeypatch.setattr(
        util, "validate_schema8_final_bundle", lambda path, label: full
    )
    monkeypatch.setattr(
        util,
        "validate_actor_policy_logger_completion",
        lambda value: logger_completion,
    )

    with pytest.raises(ValueError, match="resolved identity|regression"):
        evaluation.validate_schema8_completed_bundle(checkpoint_dir)


@pytest.mark.parametrize(("index", "value"), [(1, 1.0), (5, 0)])
def test_schema8_completed_bundle_rejects_closed_stage_type_drift(
    monkeypatch, tmp_path, index, value
):
    checkpoint_dir = _schema8_checkpoint_dir(tmp_path)
    (checkpoint_dir / "config_c.yaml").write_text(
        "voc_gate_policy_schema_version: 8\n", encoding="utf-8"
    )
    marker, full, logger_completion = _schema8_public_bundle_records(
        checkpoint_dir=checkpoint_dir
    )
    stage = list(full["resolved_identity"]["stage"])
    stage[index] = value
    full["resolved_identity"]["stage"] = tuple(stage)
    monkeypatch.setattr(
        evaluation, "validate_completion_marker", lambda path: marker
    )
    monkeypatch.setattr(
        util, "validate_schema8_final_bundle", lambda path, label: full
    )
    monkeypatch.setattr(
        util,
        "validate_actor_policy_logger_completion",
        lambda value: logger_completion,
    )

    with pytest.raises(ValueError, match="closed stage"):
        evaluation.validate_schema8_completed_bundle(checkpoint_dir)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("savedir", "relative/runs"),
        ("ckpdir", "/sealed/runs/wrong-xpid"),
        ("cmd", ""),
        ("icopro_data_path", "/sealed/data"),
    ],
)
def test_schema8_completed_bundle_rejects_path_identity_drift(
    monkeypatch, tmp_path, field, replacement
):
    checkpoint_dir = _schema8_checkpoint_dir(tmp_path)
    (checkpoint_dir / "config_c.yaml").write_text(
        "voc_gate_policy_schema_version: 8\n", encoding="utf-8"
    )
    marker, full, logger_completion = _schema8_public_bundle_records(
        checkpoint_dir=checkpoint_dir
    )
    full["resolved_identity"]["paths"][field] = replacement
    monkeypatch.setattr(
        evaluation, "validate_completion_marker", lambda path: marker
    )
    monkeypatch.setattr(
        util, "validate_schema8_final_bundle", lambda path, label: full
    )
    monkeypatch.setattr(
        util,
        "validate_actor_policy_logger_completion",
        lambda value: logger_completion,
    )

    with pytest.raises(ValueError, match="path"):
        evaluation.validate_schema8_completed_bundle(checkpoint_dir)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("voc_model_input_sealed", 1, "must be sealed"),
        ("voc_model_input_seal_count", True, "must be Python int"),
        ("voc_model_input_late_write_count", 1, "requires.*=0"),
        ("voc_model_input_abort_count", 1, "requires.*=0"),
        ("voc_model_terminal_drain_update_count", 2, "drain evidence"),
    ],
)
def test_schema8_completed_bundle_rejects_exact10_type_and_branch_attacks(
    monkeypatch, tmp_path, field, value, error
):
    checkpoint_dir = _schema8_checkpoint_dir(tmp_path)
    (checkpoint_dir / "config_c.yaml").write_text(
        "voc_gate_policy_schema_version: 8\n", encoding="utf-8"
    )
    marker, full, logger_completion = _schema8_public_bundle_records(
        checkpoint_dir=checkpoint_dir
    )
    full["model_input_seal"][field] = value
    monkeypatch.setattr(
        evaluation, "validate_completion_marker", lambda path: marker
    )
    monkeypatch.setattr(
        util, "validate_schema8_final_bundle", lambda path, label: full
    )
    monkeypatch.setattr(
        util,
        "validate_actor_policy_logger_completion",
        lambda value: logger_completion,
    )

    with pytest.raises(ValueError, match=error):
        evaluation.validate_schema8_completed_bundle(checkpoint_dir)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "extra",
        "wrong_digest",
        "schema_bool",
        "policy_float",
        "gate_float",
        "ack_rank_bool",
        "metadata_none",
        "metadata_key",
        "metadata_shape_bool",
        "metadata_numel_float",
        "metadata_numel_product",
        "history_stale_digest",
    ],
)
def test_schema8_completed_bundle_rejects_actor_summary_attacks(
    monkeypatch, tmp_path, mutation
):
    checkpoint_dir = _schema8_checkpoint_dir(tmp_path)
    (checkpoint_dir / "config_c.yaml").write_text(
        "voc_gate_policy_schema_version: 8\n", encoding="utf-8"
    )
    marker, full, logger_completion = _schema8_public_bundle_records(
        checkpoint_dir=checkpoint_dir
    )
    summary = full["actor_policy"]["voc_actor_policy_bundle_summary"]
    if mutation == "missing":
        summary.pop("actor_state_dict_metadata")
    elif mutation == "extra":
        summary["state_sha256"] = "f" * 64
    elif mutation == "wrong_digest":
        summary["actor_state_dict_sha256"] = "9" * 64
    elif mutation == "schema_bool":
        summary["bundle_schema_version"] = True
    elif mutation == "policy_float":
        summary["policy_version"] = 2.0
    elif mutation == "gate_float":
        summary["gate_schema"] = 8.0
    elif mutation == "ack_rank_bool":
        full["actor_policy"]["voc_actor_policy_publication_history"][0][
            "ack_ranks"
        ] = [False]
    elif mutation == "metadata_none":
        summary["actor_state_dict_metadata"][0] = None
    elif mutation == "metadata_key":
        summary["actor_state_dict_metadata"][0]["key"] = "other"
    elif mutation == "metadata_shape_bool":
        summary["actor_state_dict_metadata"][0]["shape"] = [True]
    elif mutation == "metadata_numel_float":
        summary["actor_state_dict_metadata"][0]["numel"] = 1.0
    elif mutation == "metadata_numel_product":
        summary["actor_state_dict_metadata"][0]["numel"] = 2
    else:
        full["actor_policy"]["voc_actor_policy_publication_history"][0][
            "state_sha256"
        ] = "9" * 64
    monkeypatch.setattr(
        evaluation, "validate_completion_marker", lambda path: marker
    )
    monkeypatch.setattr(
        util, "validate_schema8_final_bundle", lambda path, label: full
    )
    monkeypatch.setattr(
        util,
        "validate_actor_policy_logger_completion",
        lambda value: logger_completion,
    )

    with pytest.raises(ValueError, match="actor"):
        evaluation.validate_schema8_completed_bundle(checkpoint_dir)


@pytest.mark.parametrize(
    "private_name",
    (
        util.VOC_ACTOR_POLICY_LOGGER_FINISH_REQUEST_FILE,
        util.VOC_ACTOR_POLICY_LOGGER_FINISH_ACK_FILE,
    ),
)
def test_schema8_completed_bundle_rejects_private_logger_markers(
    monkeypatch, tmp_path, private_name
):
    checkpoint_dir = _schema8_checkpoint_dir(tmp_path)
    (checkpoint_dir / "config_c.yaml").write_text(
        "voc_gate_policy_schema_version: 8\n", encoding="utf-8"
    )
    (checkpoint_dir / private_name).write_text("forensic\n", encoding="utf-8")
    marker, full, logger_completion = _schema8_public_bundle_records(
        checkpoint_dir=checkpoint_dir
    )
    monkeypatch.setattr(
        evaluation, "validate_completion_marker", lambda path: marker
    )
    monkeypatch.setattr(
        util, "validate_schema8_final_bundle", lambda path, label: full
    )
    monkeypatch.setattr(
        util,
        "validate_actor_policy_logger_completion",
        lambda value: logger_completion,
    )

    with pytest.raises(ValueError, match="private logger marker"):
        evaluation.validate_schema8_completed_bundle(checkpoint_dir)


def test_schema8_evaluation_runtime_copy_preserves_identity_and_disables_coordination():
    training = SimpleNamespace(
        voc_gate_policy_schema_version=8,
        voc_model_input_seal_schema_version=1,
        train_actor=True,
        train_model=True,
        parallel=True,
        parallel_actor=True,
        use_wandb=True,
        voc_actor_policy_barrier_runtime=True,
        voc_train_epsilon=0.02,
        voc_gate_execution_epsilon=0.25,
    )

    runtime, record = evaluation.evaluation_runtime_flags(training)

    assert training.train_actor is True and training.train_model is True
    assert runtime.voc_gate_policy_schema_version == 8
    assert runtime.voc_model_input_seal_schema_version == 1
    assert runtime.train_actor is False and runtime.train_model is False
    assert runtime.voc_train_epsilon == 0.02
    assert runtime.voc_gate_execution_epsilon == 0.25
    assert record["immutable_training"]["voc_model_input_seal_schema_version"] == 1
    assert record["evaluation_copy"][
        "effective_model_input_seal_coordination"
    ] is False


@pytest.mark.parametrize("gate_schema", [8, 9, 10, 11, 12, 13])
def test_checkpoint_protocol_new_schemas_keep_derived_identity_unpersisted(
    gate_schema,
):
    values = _flag_values()
    values.update(
        {
            "voc_gate_policy_schema_version": gate_schema,
            "voc_model_input_seal_schema_version": 1,
        }
    )
    protocol = evaluation.checkpoint_protocol(SimpleNamespace(**values))

    assert protocol["voc_model_input_seal_schema_version"] == 1
    assert "voc_q_regression_loss" not in protocol
    assert "voc_q_reconstruction" not in protocol


def test_schema9_training_protocol_preserves_schema8_shape_and_seal_identity():
    protocols = {}
    for gate_schema in (8, 9):
        values = _flag_values()
        values.update(
            {
                "voc_gate_policy_schema_version": gate_schema,
                "voc_model_input_seal_schema_version": 1,
            }
        )
        protocols[gate_schema] = evaluation.checkpoint_protocol(
            SimpleNamespace(**values)
        )

    assert set(protocols[9]) == set(protocols[8])
    assert protocols[9]["voc_model_input_seal_schema_version"] == 1
    assert {
        key: value
        for key, value in protocols[9].items()
        if key != "voc_gate_policy_schema_version"
    } == {
        key: value
        for key, value in protocols[8].items()
        if key != "voc_gate_policy_schema_version"
    }


def test_schema10_training_protocol_is_schema9_shape_differential_only():
    protocols = {}
    for gate_schema in (9, 10):
        values = _flag_values()
        values.update(
            {
                "voc_gate_policy_schema_version": gate_schema,
                "voc_model_input_seal_schema_version": 1,
            }
        )
        protocols[gate_schema] = evaluation.checkpoint_protocol(
            SimpleNamespace(**values)
        )

    assert set(protocols[10]) == set(protocols[9])
    assert {
        key: value
        for key, value in protocols[10].items()
        if key != "voc_gate_policy_schema_version"
    } == {
        key: value
        for key, value in protocols[9].items()
        if key != "voc_gate_policy_schema_version"
    }


def test_schema8_dispatch_preserves_schema7_route(monkeypatch, tmp_path):
    checkpoint_dir = tmp_path / "schema7"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "config_c.yaml").write_text(
        "voc_gate_policy_schema_version: 7\n"
        "xpid: enduro-voc-v14-sealed-eps25-sentinel-wire1200\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        evaluation,
        "validate_schema8_completed_bundle",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("legacy dispatch entered dedicated schema8 route")
        ),
    )

    assert evaluation.dispatch_schema8_completed_bundle(checkpoint_dir) is None


def test_schema9_dispatch_preserves_schema8_route(monkeypatch, tmp_path):
    checkpoint_dir = tmp_path / "schema8"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "config_c.yaml").write_text(
        "voc_gate_policy_schema_version: 8\n"
        "xpid: enduro-voc-v15-halfsq-eps25-sentinel-wire1200\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        evaluation,
        "validate_schema9_completed_bundle",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("schema-8 dispatch entered dedicated schema9 route")
        ),
    )

    assert evaluation.dispatch_schema9_completed_bundle(checkpoint_dir) is None


def test_schema10_dispatch_preserves_schema9_route(monkeypatch, tmp_path):
    checkpoint_dir = tmp_path / "schema9"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "config_c.yaml").write_text(
        "voc_gate_policy_schema_version: 9\n"
        "xpid: enduro-voc-v16-commonmode-eps25-sentinel-wire1200\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        evaluation,
        "validate_schema10_completed_bundle",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("schema-9 dispatch entered dedicated schema10 route")
        ),
    )

    assert evaluation.dispatch_schema10_completed_bundle(checkpoint_dir) is None


def test_schema11_dispatch_preserves_schema10_route(monkeypatch, tmp_path):
    checkpoint_dir = tmp_path / "schema10"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "config_c.yaml").write_text(
        "voc_gate_policy_schema_version: 10\n"
        "xpid: enduro-voc-v17-huber-common-eps25-sentinel-wire1200\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        evaluation,
        "validate_schema11_completed_bundle",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("schema-10 dispatch entered dedicated schema11 route")
        ),
    )

    assert evaluation.dispatch_schema11_completed_bundle(checkpoint_dir) is None


def test_schema9_evaluation_runtime_copy_preserves_identity_and_disables_coordination():
    training = SimpleNamespace(
        voc_gate_policy_schema_version=9,
        voc_model_input_seal_schema_version=1,
        train_actor=True,
        train_model=True,
        parallel=True,
        parallel_actor=True,
        use_wandb=True,
        voc_actor_policy_barrier_runtime=True,
        voc_train_epsilon=0.02,
        voc_gate_execution_epsilon=0.25,
    )

    runtime, record = evaluation.evaluation_runtime_flags(training)

    assert training.train_actor is True and training.train_model is True
    assert runtime.voc_gate_policy_schema_version == 9
    assert runtime.train_actor is False and runtime.train_model is False
    assert record["immutable_training"]["voc_train_epsilon"] == 0.02
    assert record["evaluation_copy"][
        "effective_model_input_seal_coordination"
    ] is False


def test_schema10_runtime_copy_preserves_identity_and_schema9_record_shape():
    training = SimpleNamespace(
        voc_gate_policy_schema_version=10,
        voc_model_input_seal_schema_version=1,
        train_actor=True,
        train_model=True,
        parallel=True,
        parallel_actor=True,
        use_wandb=True,
        voc_actor_policy_barrier_runtime=True,
        voc_train_epsilon=0.02,
        voc_gate_execution_epsilon=0.25,
    )

    runtime, record = evaluation.evaluation_runtime_flags(training)

    assert runtime.voc_gate_policy_schema_version == 10
    assert runtime.train_actor is False and runtime.train_model is False
    assert runtime.voc_train_epsilon == 0.02
    assert runtime.voc_gate_execution_epsilon == 0.25
    assert record["evaluation_copy"][
        "effective_model_input_seal_coordination"
    ] is False


@pytest.mark.parametrize("gate_schema", [11, 12, 13])
def test_schema11_runtime_copy_preserves_schema10_record_shape(gate_schema):
    training = SimpleNamespace(
        voc_gate_policy_schema_version=gate_schema,
        voc_model_input_seal_schema_version=1,
        train_actor=True,
        train_model=True,
        parallel=True,
        parallel_actor=True,
        use_wandb=True,
        voc_actor_policy_barrier_runtime=True,
        voc_train_epsilon=0.02,
        voc_gate_execution_epsilon=0.25,
    )

    runtime, record = evaluation.evaluation_runtime_flags(training)

    assert runtime.voc_gate_policy_schema_version == gate_schema
    assert runtime.train_actor is False and runtime.train_model is False
    assert training.use_wandb is True and runtime.use_wandb is False
    assert runtime.voc_train_epsilon == 0.02
    assert runtime.voc_gate_execution_epsilon == 0.25
    assert record["evaluation_copy"][
        "effective_model_input_seal_coordination"
    ] is False


def test_schema9_validation_detects_config_swap_before_return(monkeypatch, tmp_path):
    checkpoint_dir = _schema9_checkpoint_dir(tmp_path)
    config_payload = b"voc_gate_policy_schema_version: 9\n"
    config_path = checkpoint_dir / "config_c.yaml"
    config_path.write_bytes(config_payload)
    config_digest = hashlib.sha256(config_payload).hexdigest()
    marker, full, logger_completion = _schema9_public_bundle_records(
        checkpoint_dir=checkpoint_dir
    )
    monkeypatch.setattr(
        evaluation, "validate_completion_marker", lambda path: marker
    )

    def mutate_after_bound_read(path, label):
        config_path.write_text(
            "voc_gate_policy_schema_version: 8\n", encoding="utf-8"
        )
        return full

    monkeypatch.setattr(util, "validate_schema9_final_bundle", mutate_after_bound_read)
    monkeypatch.setattr(
        util,
        "validate_actor_policy_logger_completion",
        lambda value: logger_completion,
    )

    with pytest.raises(RuntimeError, match="config changed during validation"):
        evaluation.validate_schema9_completed_bundle(
            checkpoint_dir,
            config_payload=config_payload,
            expected_config_sha256=config_digest,
        )


def test_schema10_validation_detects_config_swap_before_return(monkeypatch, tmp_path):
    checkpoint_dir = _schema10_checkpoint_dir(tmp_path)
    config_payload = b"voc_gate_policy_schema_version: 10\n"
    config_path = checkpoint_dir / "config_c.yaml"
    config_path.write_bytes(config_payload)
    config_digest = hashlib.sha256(config_payload).hexdigest()
    marker, full, logger_completion = _schema10_public_bundle_records(
        checkpoint_dir=checkpoint_dir
    )
    monkeypatch.setattr(
        evaluation, "validate_completion_marker", lambda path: marker
    )

    def mutate_after_bound_read(path, label):
        config_path.write_text(
            "voc_gate_policy_schema_version: 9\n", encoding="utf-8"
        )
        return full

    monkeypatch.setattr(
        util, "validate_schema10_final_bundle", mutate_after_bound_read
    )
    monkeypatch.setattr(
        util,
        "validate_actor_policy_logger_completion",
        lambda value: logger_completion,
    )

    with pytest.raises(RuntimeError, match="config changed during validation"):
        evaluation.validate_schema10_completed_bundle(
            checkpoint_dir,
            config_payload=config_payload,
            expected_config_sha256=config_digest,
        )


def test_schema11_validation_detects_config_swap_before_return(monkeypatch, tmp_path):
    checkpoint_dir = _schema11_checkpoint_dir(tmp_path)
    config_payload = b"voc_gate_policy_schema_version: 11\n"
    config_path = checkpoint_dir / "config_c.yaml"
    config_path.write_bytes(config_payload)
    config_digest = hashlib.sha256(config_payload).hexdigest()
    marker, full, logger_completion = _schema11_public_bundle_records(
        checkpoint_dir=checkpoint_dir
    )
    monkeypatch.setattr(
        evaluation, "validate_completion_marker", lambda path: marker
    )

    def mutate_after_bound_read(path, label):
        config_path.write_text(
            "voc_gate_policy_schema_version: 10\n", encoding="utf-8"
        )
        return full

    monkeypatch.setattr(
        util, "validate_schema11_final_bundle", mutate_after_bound_read
    )
    monkeypatch.setattr(
        util,
        "validate_actor_policy_logger_completion",
        lambda value: logger_completion,
    )

    with pytest.raises(RuntimeError, match="config changed during validation"):
        evaluation.validate_schema11_completed_bundle(
            checkpoint_dir,
            config_payload=config_payload,
            expected_config_sha256=config_digest,
        )


def test_invalid_schema10_public_bundle_fails_before_every_downstream_call(
    monkeypatch, tmp_path
):
    import thinker.bc_loader as bc_loader
    import thinker.dynamic_imitation as dynamic_imitation

    checkpoint_dir = tmp_path / "invalid-v17"
    checkpoint_dir.mkdir()
    config_payload = b"voc_gate_policy_schema_version: 10\n"
    (checkpoint_dir / "config_c.yaml").write_bytes(config_payload)
    config_digest = hashlib.sha256(config_payload).hexdigest()
    output_dir = tmp_path / "output"
    events = []
    monkeypatch.setattr(
        evaluation,
        "checkpoint_hashes",
        lambda path: {"config_c.yaml": config_digest},
    )
    monkeypatch.setattr(
        evaluation,
        "validate_completion_marker",
        lambda path: {
            "status": "complete",
            "checkpoint_files": {
                "config_c.yaml": {
                    "sha256": config_digest,
                    "size": len(config_payload),
                }
            },
        },
    )
    monkeypatch.setattr(
        evaluation, "dispatch_schema8_completed_bundle", lambda *a, **k: None
    )
    monkeypatch.setattr(
        evaluation, "dispatch_schema9_completed_bundle", lambda *a, **k: None
    )

    def reject_schema10(*args, **kwargs):
        events.append("authoritative_schema10_validation")
        raise ValueError("invalid schema-10 evidence")

    def forbidden(name):
        def _forbidden(*args, **kwargs):
            events.append(name)
            raise AssertionError(f"{name} ran before schema-10 validation")

        return _forbidden

    monkeypatch.setattr(
        evaluation, "dispatch_schema10_completed_bundle", reject_schema10
    )
    monkeypatch.setattr(
        evaluation,
        "_load_flags_from_validated_config_bytes",
        forbidden("byte_bound_load_flags"),
    )
    monkeypatch.setattr(
        evaluation, "resolve_evaluation_spec", forbidden("live_spec")
    )
    monkeypatch.setattr(torch, "load", forbidden("downstream_torch_load"))
    monkeypatch.setattr(
        bc_loader,
        "FrameStackedBehavioralDataLoader",
        forbidden("data_loader"),
    )
    monkeypatch.setattr(
        dynamic_imitation, "DynamicImitationRunner", forbidden("rollout")
    )

    with pytest.raises(ValueError, match="invalid schema-10"):
        evaluation.evaluate(
            SimpleNamespace(
                checkpoint_dir=checkpoint_dir,
                output_dir=str(output_dir),
                seed=1,
            )
        )

    assert events == ["authoritative_schema10_validation"]
    assert not output_dir.exists()


def test_invalid_schema11_public_bundle_fails_before_every_downstream_call(
    monkeypatch, tmp_path
):
    import thinker.bc_loader as bc_loader
    import thinker.dynamic_imitation as dynamic_imitation

    checkpoint_dir = tmp_path / "invalid-v18"
    checkpoint_dir.mkdir()
    config_payload = b"voc_gate_policy_schema_version: 11\n"
    (checkpoint_dir / "config_c.yaml").write_bytes(config_payload)
    config_digest = hashlib.sha256(config_payload).hexdigest()
    output_dir = tmp_path / "output"
    events = []
    monkeypatch.setattr(
        evaluation,
        "checkpoint_hashes",
        lambda path: {"config_c.yaml": config_digest},
    )
    monkeypatch.setattr(
        evaluation,
        "validate_completion_marker",
        lambda path: {
            "status": "complete",
            "checkpoint_files": {
                "config_c.yaml": {
                    "sha256": config_digest,
                    "size": len(config_payload),
                }
            },
        },
    )
    monkeypatch.setattr(
        evaluation, "dispatch_schema8_completed_bundle", lambda *a, **k: None
    )
    monkeypatch.setattr(
        evaluation, "dispatch_schema9_completed_bundle", lambda *a, **k: None
    )
    monkeypatch.setattr(
        evaluation, "dispatch_schema10_completed_bundle", lambda *a, **k: None
    )

    def reject_schema11(*args, **kwargs):
        events.append("authoritative_schema11_validation")
        raise ValueError("invalid schema-11 evidence")

    def forbidden(name):
        def _forbidden(*args, **kwargs):
            events.append(name)
            raise AssertionError(f"{name} ran before schema-11 validation")

        return _forbidden

    monkeypatch.setattr(
        evaluation, "dispatch_schema11_completed_bundle", reject_schema11
    )
    monkeypatch.setattr(
        evaluation,
        "_load_flags_from_validated_config_bytes",
        forbidden("byte_bound_load_flags"),
    )
    monkeypatch.setattr(
        evaluation, "resolve_evaluation_spec", forbidden("live_spec")
    )
    monkeypatch.setattr(torch, "load", forbidden("downstream_torch_load"))
    monkeypatch.setattr(
        bc_loader,
        "FrameStackedBehavioralDataLoader",
        forbidden("data_loader"),
    )
    monkeypatch.setattr(
        dynamic_imitation, "DynamicImitationRunner", forbidden("rollout")
    )

    with pytest.raises(ValueError, match="invalid schema-11"):
        evaluation.evaluate(
            SimpleNamespace(
                checkpoint_dir=checkpoint_dir,
                output_dir=str(output_dir),
                seed=1,
            )
        )

    assert events == ["authoritative_schema11_validation"]
    assert not output_dir.exists()


@pytest.mark.parametrize(
    "schema_line",
    [
        pytest.param(b"voc_gate_policy_schema_version: 5\n", id="wrong-schema"),
        pytest.param(b"", id="missing-schema"),
    ],
)
def test_malformed_v18_prefix_routes_to_schema11_before_public_downstream(
    monkeypatch, tmp_path, schema_line
):
    import thinker.bc_loader as bc_loader
    import thinker.dynamic_imitation as dynamic_imitation

    checkpoint_dir = tmp_path / "malformed-v18-prefix"
    checkpoint_dir.mkdir()
    config_payload = schema_line + (
        b"xpid: enduro-voc-v18-orthocd-adam-eps25-malformed-stage\n"
    )
    (checkpoint_dir / "config_c.yaml").write_bytes(config_payload)
    config_digest = hashlib.sha256(config_payload).hexdigest()
    output_dir = tmp_path / "must-not-exist"
    events = []
    monkeypatch.setattr(
        evaluation,
        "checkpoint_hashes",
        lambda path: {"config_c.yaml": config_digest},
    )
    monkeypatch.setattr(
        evaluation,
        "validate_completion_marker",
        lambda path: {
            "status": "complete",
            "checkpoint_files": {
                "config_c.yaml": {
                    "sha256": config_digest,
                    "size": len(config_payload),
                }
            },
        },
    )
    monkeypatch.setattr(
        evaluation, "dispatch_schema8_completed_bundle", lambda *a, **k: None
    )
    monkeypatch.setattr(
        evaluation, "dispatch_schema9_completed_bundle", lambda *a, **k: None
    )
    monkeypatch.setattr(
        evaluation, "dispatch_schema10_completed_bundle", lambda *a, **k: None
    )

    def forbidden(name):
        def _forbidden(*args, **kwargs):
            events.append(name)
            raise AssertionError(f"{name} ran before schema-11 validation")

        return _forbidden

    monkeypatch.setattr(
        evaluation,
        "_load_flags_from_validated_config_bytes",
        forbidden("byte_bound_load_flags"),
    )
    monkeypatch.setattr(
        evaluation, "resolve_evaluation_spec", forbidden("live_spec")
    )
    monkeypatch.setattr(torch, "load", forbidden("downstream_torch_load"))
    monkeypatch.setattr(
        bc_loader,
        "FrameStackedBehavioralDataLoader",
        forbidden("data_loader"),
    )
    monkeypatch.setattr(
        dynamic_imitation, "DynamicImitationRunner", forbidden("rollout")
    )

    with pytest.raises(
        ValueError,
        match="dedicated schema-11 validation requires exact Python integer",
    ):
        evaluation.evaluate(
            SimpleNamespace(
                checkpoint_dir=checkpoint_dir,
                output_dir=str(output_dir),
                seed=1,
            )
        )

    assert events == []
    assert not output_dir.exists()


def test_invalid_schema9_public_bundle_fails_before_downstream_calls(
    monkeypatch, tmp_path
):
    import thinker.bc_loader as bc_loader
    import thinker.dynamic_imitation as dynamic_imitation

    checkpoint_dir = tmp_path / "invalid-v16"
    checkpoint_dir.mkdir()
    config_payload = b"voc_gate_policy_schema_version: 9\n"
    (checkpoint_dir / "config_c.yaml").write_bytes(config_payload)
    config_digest = hashlib.sha256(config_payload).hexdigest()
    output_dir = tmp_path / "output"
    events = []

    monkeypatch.setattr(
        evaluation,
        "checkpoint_hashes",
        lambda path: {"config_c.yaml": config_digest},
    )
    monkeypatch.setattr(
        evaluation,
        "validate_completion_marker",
        lambda path: {
            "status": "complete",
            "checkpoint_files": {
                "config_c.yaml": {
                    "sha256": config_digest,
                    "size": len(config_payload),
                }
            },
        },
    )
    monkeypatch.setattr(
        evaluation, "dispatch_schema8_completed_bundle", lambda *args, **kwargs: None
    )

    def reject_schema9(*args, **kwargs):
        events.append("authoritative_schema9_validation")
        raise ValueError("invalid schema-9 evidence")

    def forbidden(name):
        def _forbidden(*args, **kwargs):
            events.append(name)
            raise AssertionError(f"{name} ran before schema-9 validation")

        return _forbidden

    monkeypatch.setattr(
        evaluation, "dispatch_schema9_completed_bundle", reject_schema9
    )
    monkeypatch.setattr(
        evaluation,
        "_load_flags_from_validated_config_bytes",
        forbidden("byte_bound_load_flags"),
    )
    monkeypatch.setattr(
        evaluation, "resolve_evaluation_spec", forbidden("live_spec")
    )
    monkeypatch.setattr(torch, "load", forbidden("downstream_torch_load"))
    monkeypatch.setattr(
        bc_loader,
        "FrameStackedBehavioralDataLoader",
        forbidden("data_loader"),
    )
    monkeypatch.setattr(
        dynamic_imitation, "DynamicImitationRunner", forbidden("rollout")
    )

    with pytest.raises(ValueError, match="invalid schema-9"):
        evaluation.evaluate(
            SimpleNamespace(
                checkpoint_dir=checkpoint_dir,
                output_dir=str(output_dir),
                seed=1,
            )
        )

    assert events == ["authoritative_schema9_validation"]
    assert not output_dir.exists()


def test_invalid_schema8_public_bundle_fails_before_downstream_calls(
    monkeypatch, tmp_path
):
    import thinker.bc_loader as bc_loader
    import thinker.dynamic_imitation as dynamic_imitation

    checkpoint_dir = tmp_path / "invalid-v15"
    checkpoint_dir.mkdir()
    config_payload = b"voc_gate_policy_schema_version: 8\n"
    (checkpoint_dir / "config_c.yaml").write_bytes(config_payload)
    config_digest = hashlib.sha256(config_payload).hexdigest()
    output_dir = tmp_path / "output"
    events = []

    monkeypatch.setattr(
        evaluation,
        "checkpoint_hashes",
        lambda path: {"config_c.yaml": config_digest},
    )
    monkeypatch.setattr(
        evaluation,
        "validate_completion_marker",
        lambda path: {
            "status": "complete",
            "checkpoint_files": {
                "config_c.yaml": {
                    "sha256": config_digest,
                    "size": len(config_payload),
                }
            },
        },
    )

    def reject_schema8(*args, **kwargs):
        events.append("authoritative_schema8_validation")
        raise ValueError("invalid schema-8 evidence")

    def forbidden(name):
        def _forbidden(*args, **kwargs):
            events.append(name)
            raise AssertionError(f"{name} ran before schema-8 validation")

        return _forbidden

    monkeypatch.setattr(
        evaluation, "dispatch_schema8_completed_bundle", reject_schema8
    )
    monkeypatch.setattr(evaluation, "_load_flags", forbidden("load_flags"))
    monkeypatch.setattr(
        evaluation,
        "_load_flags_from_validated_config_bytes",
        forbidden("byte_bound_load_flags"),
    )
    monkeypatch.setattr(
        evaluation, "resolve_evaluation_spec", forbidden("live_spec")
    )
    monkeypatch.setattr(torch, "load", forbidden("downstream_torch_load"))
    monkeypatch.setattr(
        bc_loader,
        "FrameStackedBehavioralDataLoader",
        forbidden("data_loader"),
    )
    monkeypatch.setattr(
        dynamic_imitation, "DynamicImitationRunner", forbidden("rollout")
    )

    with pytest.raises(ValueError, match="invalid schema-8"):
        evaluation.evaluate(
            SimpleNamespace(
                checkpoint_dir=checkpoint_dir,
                output_dir=str(output_dir),
                seed=1,
            )
        )

    assert events == ["authoritative_schema8_validation"]
    assert not output_dir.exists()


def test_public_evaluation_uses_bound_config_bytes_and_detects_path_swap(
    monkeypatch, tmp_path
):
    import thinker.bc_loader as bc_loader
    import thinker.dynamic_imitation as dynamic_imitation

    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    config_path = checkpoint_dir / "config_c.yaml"
    config_payload = b"voc_gate_policy_schema_version: 5\nxpid: legacy\n"
    config_path.write_bytes(config_payload)
    config_digest = hashlib.sha256(config_payload).hexdigest()
    marker = {
        "status": "complete",
        "checkpoint_files": {
            "config_c.yaml": {
                "sha256": config_digest,
                "size": len(config_payload),
            }
        },
    }
    events = []

    def hashes(path):
        return {
            "config_c.yaml": hashlib.sha256(config_path.read_bytes()).hexdigest()
        }

    def dispatch(*args, config_payload, **kwargs):
        events.append("dispatch")
        assert config_payload == b"voc_gate_policy_schema_version: 5\nxpid: legacy\n"
        config_path.write_text(
            "voc_gate_policy_schema_version: 8\n",
            encoding="utf-8",
        )
        return None

    def byte_loader(path, payload, digest):
        events.append("byte_loader")
        assert payload == config_payload
        assert digest == config_digest
        return SimpleNamespace()

    def forbidden(name):
        def _forbidden(*args, **kwargs):
            events.append(name)
            raise AssertionError(f"{name} ran after checkpoint config swap")

        return _forbidden

    monkeypatch.setattr(evaluation, "checkpoint_hashes", hashes)
    monkeypatch.setattr(
        evaluation, "validate_completion_marker", lambda path: marker
    )
    monkeypatch.setattr(
        evaluation, "dispatch_schema8_completed_bundle", dispatch
    )
    monkeypatch.setattr(
        evaluation, "_load_flags_from_validated_config_bytes", byte_loader
    )
    monkeypatch.setattr(evaluation, "_load_flags", forbidden("pathname_loader"))
    monkeypatch.setattr(
        evaluation, "resolve_evaluation_spec", forbidden("live_spec")
    )
    monkeypatch.setattr(torch, "load", forbidden("downstream_torch_load"))
    monkeypatch.setattr(
        bc_loader,
        "FrameStackedBehavioralDataLoader",
        forbidden("data_loader"),
    )
    monkeypatch.setattr(
        dynamic_imitation, "DynamicImitationRunner", forbidden("rollout")
    )

    with pytest.raises(RuntimeError, match="changed before evaluation flag"):
        evaluation.evaluate(
            SimpleNamespace(
                checkpoint_dir=checkpoint_dir,
                output_dir=tmp_path / "output",
                seed=1,
            )
        )

    assert events == ["dispatch", "byte_loader"]
    assert not (tmp_path / "output").exists()


def test_legacy_evaluation_runtime_copy_preserves_schema_semantics():
    training = SimpleNamespace(
        voc_gate_policy_schema_version=5,
        train_actor=True,
        parallel=True,
        parallel_actor=True,
        use_wandb=True,
        voc_actor_policy_barrier_runtime=False,
        voc_train_epsilon=0.02,
        voc_gate_execution_epsilon=0.02,
    )

    runtime, record = evaluation.evaluation_runtime_flags(training)

    assert record is None
    assert runtime is not training
    assert runtime.train_actor is True
    assert runtime.voc_actor_policy_barrier_runtime is False
    assert runtime.voc_train_epsilon == 0.02
    assert runtime.voc_gate_execution_epsilon == 0.02
    assert runtime.parallel is False
    assert runtime.parallel_actor is False
    assert runtime.use_wandb is False


def _schema6_complete_surface(tmp_path):
    root = tmp_path.resolve()
    savedir = root / "runs"
    xpid = "enduro-voc-v13-versioned-eps25-sentinel-wire1200"
    return {
        **dict(util.VOC_GATE_POLICY_SCHEMA6_V12_BASELINE),
        "xpid": xpid,
        "base_seed": 1,
        "total_steps": 1200,
        "model_warm_up_n": 512,
        "actor_unroll_len": 41,
        "use_wandb": False,
        "savedir": str(savedir),
        "ckpdir": str(savedir / xpid),
        "cmd": "python train.py --wire",
        "icopro_data_path": str(root / "data" / "behavioral_data_block"),
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


def _schema11_to_13_complete_surface(tmp_path, schema):
    projection = dict(
        util.VOC_GATE_POLICY_SCHEMA6_V12_BASELINE
        if schema == 11
        else util.VOC_GATE_POLICY_SCHEMA12_V12_PROJECTION
    )
    stage_profiles = {
        11: util.VOC_GATE_POLICY_SCHEMA11_STAGE_PROFILES,
        12: util.VOC_GATE_POLICY_SCHEMA12_STAGE_PROFILES,
        13: util.VOC_GATE_POLICY_SCHEMA13_STAGE_PROFILES,
    }
    xpid, base_seed, total_steps, warm_up, unroll, use_wandb = (
        stage_profiles[schema][0]
    )
    root = tmp_path.resolve()
    savedir = root / "runs"
    return {
        **projection,
        "xpid": xpid,
        "base_seed": base_seed,
        "total_steps": total_steps,
        "model_warm_up_n": warm_up,
        "actor_unroll_len": unroll,
        "use_wandb": use_wandb,
        "savedir": str(savedir),
        "ckpdir": str(savedir / xpid),
        "cmd": "python train.py --wire",
        "icopro_data_path": str(root / "data" / "behavioral_data_block"),
        "voc_gate_policy_schema_version": schema,
        "voc_gate_execution_epsilon": 0.25,
        "voc_actor_policy_version_barrier": True,
        "voc_actor_policy_bundle_schema_version": 1,
        "voc_actor_policy_barrier_timeout_s": 120.0,
        "voc_actor_policy_ray_max_restarts": 0,
        "voc_actor_policy_ray_max_task_retries": 0,
        "actor_amp_init_scale": 32.0,
        "voc_actor_policy_barrier_runtime": True,
        "voc_model_input_seal_schema_version": 1,
    }


def _actual_schema13_flags(monkeypatch, tmp_path):
    surface = _schema11_to_13_complete_surface(tmp_path, 13)
    surface["cmd"] = " ".join(sys.argv)
    monkeypatch.setattr(util, "get_git_revision_hash", lambda: None)
    flags = util.create_flags(
        ["default_thinker.yaml", "default_actor.yaml"],
        save_flags=False,
        post_fn=util.process_flags_actor,
        **surface,
    )
    assert len(vars(flags)) == 229
    assert vars(flags) == surface
    return flags


def test_schema13_bound_flag_loader_reconstructs_actual_create_flags_surface(
    monkeypatch, tmp_path
):
    created = _actual_schema13_flags(monkeypatch, tmp_path)
    payload = yaml.safe_dump(vars(created), sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()

    loaded = evaluation._load_flags_from_validated_config_bytes(
        created.ckpdir, payload, digest
    )

    assert vars(loaded) == vars(created)
    explicit_config = tmp_path / "schema13-user-config.yaml"
    explicit_config.write_bytes(payload)
    with pytest.raises(ValueError, match="forbids user-config indirection"):
        util.create_flags(
            ["default_thinker.yaml", "default_actor.yaml"],
            save_flags=False,
            post_fn=util.process_flags_actor,
            config=str(explicit_config),
            ckp=False,
        )


@pytest.mark.parametrize("bad_schema", [12, None])
def test_schema13_bound_flag_loader_rejects_lexical_intent_without_legacy_loader(
    monkeypatch, tmp_path, bad_schema
):
    created = _actual_schema13_flags(monkeypatch, tmp_path)
    malformed = dict(vars(created))
    if bad_schema is None:
        malformed.pop("voc_gate_policy_schema_version")
    else:
        malformed["voc_gate_policy_schema_version"] = bad_schema
    payload = yaml.safe_dump(malformed, sort_keys=True).encode("utf-8")
    calls = []

    def forbidden_create_flags(*args, **kwargs):
        calls.append("create_flags")
        raise AssertionError("schema-13 lexical intent reached legacy flag loading")

    monkeypatch.setattr(util, "create_flags", forbidden_create_flags)
    with pytest.raises(ValueError, match="schema|surface|xpid"):
        evaluation._load_flags_from_validated_config_bytes(
            created.ckpdir,
            payload,
            hashlib.sha256(payload).hexdigest(),
        )

    assert calls == []


def test_completion_bound_checkpoint_loader_never_reopens_swapped_path(
    monkeypatch, tmp_path
):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / "ckp_actor.tar"
    bound_buffer = io.BytesIO()
    replacement_buffer = io.BytesIO()
    torch.save({"generation": "bound"}, bound_buffer)
    torch.save({"generation": "replacement"}, replacement_buffer)
    bound_payload = bound_buffer.getvalue()
    checkpoint_path.write_bytes(bound_payload)
    completion = {
        "checkpoint_files": {
            "ckp_actor.tar": {
                "sha256": hashlib.sha256(bound_payload).hexdigest(),
                "size": len(bound_payload),
            }
        }
    }
    stable_reader = evaluation._read_stable_single_link_bytes

    def swap_after_stable_read(path, *, label):
        payload = stable_reader(path, label=label)
        checkpoint_path.write_bytes(replacement_buffer.getvalue())
        return payload

    monkeypatch.setattr(
        evaluation, "_read_stable_single_link_bytes", swap_after_stable_read
    )

    checkpoint = evaluation._load_checkpoint_from_completion_bytes(
        checkpoint_dir,
        "ckp_actor.tar",
        completion,
        label="test actor checkpoint",
    )

    assert checkpoint["generation"] == "bound"
    assert torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )["generation"] == "replacement"


def test_runtime_checkpoint_loader_preserves_legacy_symlink_differential(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    target = tmp_path / "actor-target.tar"
    torch.save({"generation": "legacy"}, target)
    (checkpoint_dir / "ckp_actor.tar").symlink_to(target)
    payload = target.read_bytes()
    completion = {
        "checkpoint_files": {
            "ckp_actor.tar": {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size": len(payload),
            }
        }
    }

    legacy = evaluation._load_runtime_checkpoint(
        checkpoint_dir,
        "ckp_actor.tar",
        completion,
        schema13=False,
        label="legacy actor checkpoint",
    )

    assert legacy["generation"] == "legacy"
    with pytest.raises(ValueError, match="regular single-link"):
        evaluation._load_runtime_checkpoint(
            checkpoint_dir,
            "ckp_actor.tar",
            completion,
            schema13=True,
            label="schema-13 actor checkpoint",
        )


@pytest.mark.parametrize("schema", [11, 12, 13])
@pytest.mark.parametrize("top_level_schema", [True, False])
def test_public_gate_schema_uses_real_authoritative_successor_surface(
    tmp_path, schema, top_level_schema
):
    embedded = _schema11_to_13_complete_surface(tmp_path, schema)
    checkpoint = {"flags": embedded}
    if top_level_schema:
        checkpoint["voc_gate_policy_schema_version"] = schema
    authoritative_checkpoint = {
        **checkpoint,
        "voc_gate_policy_schema_version": schema,
    }

    expected = util.validate_voc_gate_policy_schema(
        authoritative_checkpoint, label="authoritative successor test"
    )
    actual = evaluation._validate_voc_gate_policy_schema(
        checkpoint, embedded, label="public successor test"
    )

    assert len(embedded) == 229
    assert actual == expected
    assert actual["voc_gate_policy_schema_version"] == schema
    assert actual["voc_model_input_seal_schema_version"] == 1


def test_public_gate_schema_uses_authoritative_schema6_surface(tmp_path):
    embedded = _schema6_complete_surface(tmp_path)
    checkpoint = {
        "voc_gate_policy_schema_version": 6,
        "flags": embedded,
    }

    state = evaluation._validate_voc_gate_policy_schema(
        checkpoint, embedded, label="public test"
    )

    assert state["voc_gate_policy_schema_version"] == 6
    assert state["voc_gate_execution_epsilon"] == 0.25
    assert state["voc_actor_policy_version_barrier"] is True
    assert state["voc_actor_policy_bundle_schema_version"] == 1
    assert state["voc_actor_policy_barrier_timeout_s"] == 120.0
    assert state["voc_actor_policy_ray_max_restarts"] == 0
    assert state["voc_actor_policy_ray_max_task_retries"] == 0
    assert state["actor_amp_init_scale"] == 32.0

    bad = dict(embedded)
    bad["unexpected"] = True
    with pytest.raises(ValueError, match="exact 228-key"):
        evaluation._validate_voc_gate_policy_schema(
            {"voc_gate_policy_schema_version": 6, "flags": bad},
            bad,
            label="public test",
        )


def test_public_gate_schema_uses_authoritative_schema7_surface(tmp_path):
    embedded = _schema6_complete_surface(tmp_path)
    xpid = "enduro-voc-v14-sealed-eps25-sentinel-wire1200"
    embedded.update(
        {
            "xpid": xpid,
            "ckpdir": str(Path(embedded["savedir"]) / xpid),
            "voc_gate_policy_schema_version": 7,
            "voc_model_input_seal_schema_version": 1,
        }
    )
    checkpoint = {
        "voc_gate_policy_schema_version": 7,
        "flags": embedded,
    }

    state = evaluation._validate_voc_gate_policy_schema(
        checkpoint, embedded, label="public schema-7 test"
    )

    assert state["voc_gate_policy_schema_version"] == 7
    assert state["voc_model_input_seal_schema_version"] == 1
    assert state[
        "voc_model_input_seal_schema_version_legacy_defaulted"
    ] is False
    bad = dict(embedded)
    bad["voc_model_input_seal_schema_version"] = True
    with pytest.raises(ValueError, match="seal_schema_version"):
        evaluation._validate_voc_gate_policy_schema(
            {"voc_gate_policy_schema_version": 7, "flags": bad},
            bad,
            label="public schema-7 test",
        )
