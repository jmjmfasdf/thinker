from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium import spaces

import evaluate_dynamic_imitation as evaluation


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


def test_checkpoint_bundle_requires_success_marker(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for name in ("config_c.yaml", "ckp_actor.tar", "ckp_model.tar"):
        (checkpoint / name).write_bytes(b"placeholder")

    with pytest.raises(FileNotFoundError, match="finish"):
        evaluation.checkpoint_hashes(checkpoint)


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
