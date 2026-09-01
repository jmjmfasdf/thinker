from argparse import Namespace
import builtins
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys

from gymnasium import spaces
import numpy as np
import pytest
import torch
import yaml

import evaluate_dynamic_imitation as evaluation
import smoke_dynamic_imitation as smoke
from thinker import util
from thinker.learn_actor import _validate_model_state_dict_compatibility


def test_smoke_checkpoint_loader_never_reopens_swapped_path(
    monkeypatch, tmp_path
):
    checkpoint_path = tmp_path / "ckp_actor.tar"
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

    checkpoint = smoke._load_stable_smoke_checkpoint(
        evaluation,
        tmp_path,
        "ckp_actor.tar",
        completion_state=completion,
        label="test smoke actor checkpoint",
    )

    assert checkpoint["generation"] == "bound"
    assert torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )["generation"] == "replacement"


def test_vector_actor_observation_space_adds_only_missing_batch_axes():
    template = spaces.Dict(
        {
            "real_states": spaces.Box(
                0, 255, shape=(12, 84, 84), dtype=np.uint8
            ),
            "xs": spaces.Box(0.0, 1.0, shape=(3, 84, 84), dtype=np.float32),
            "tree_reps": spaces.Box(
                -np.inf, np.inf, shape=(2, 104), dtype=np.float32
            ),
            "hs": spaces.Box(
                -np.inf, np.inf, shape=(2, 64, 6, 6), dtype=np.float32
            ),
        }
    )

    vector = smoke._vector_actor_observation_space(template, batch_size=2)

    assert vector["real_states"].shape == (2, 12, 84, 84)
    assert vector["xs"].shape == (2, 3, 84, 84)
    assert vector["tree_reps"].shape == (2, 104)
    assert vector["hs"].shape == (2, 64, 6, 6)


def test_smoke_state_dict_validation_reports_shape_and_key_errors():
    module = torch.nn.Linear(3, 2)
    valid = module.state_dict()
    smoke._validate_state_dict(module, valid, "test")

    invalid = {"weight": torch.zeros(4, 3), "extra": torch.zeros(1)}
    with pytest.raises(ValueError) as error:
        smoke._validate_state_dict(module, invalid, "test")
    message = str(error.value)
    assert "missing=['bias']" in message
    assert "unexpected=['extra']" in message
    assert "weight: incoming(4, 3) != expected(2, 3)" in message

    with pytest.raises(ValueError, match="shape_mismatch"):
        _validate_model_state_dict_compatibility(module, invalid, "test")


def test_fresh_smoke_flags_use_dynamic_20_20_20_protocol():
    args = Namespace(
        config=None,
        checkpoint_dir=None,
        env_name="Enduro-v5",
        rec_t=None,
        max_search_steps=None,
        max_depth=None,
        model_unroll_len=None,
        think_cost=None,
        model_size_nn=None,
        frame_stack_n=None,
        grayscale=None,
        tree_carry=None,
        scored_length=4,
        device=torch.device("cpu"),
    )

    flags = smoke._load_flags(args)

    assert flags.name == "Enduro-v5"
    assert flags.dynamic_search is True
    assert flags.sep_im_head is True
    assert flags.envpool is True
    assert (flags.rec_t, flags.max_search_steps, flags.max_depth) == (20, 20, 20)
    assert flags.model_unroll_len == 20
    assert flags.think_cost == pytest.approx(0.0005)
    assert flags.dynamic_voc_mode == "off"
    assert flags.voc_loss_cost == pytest.approx(1.0)
    assert flags.voc_gate_temperature == pytest.approx(1.0)
    assert flags.voc_train_epsilon == pytest.approx(0.02)
    assert flags.voc_eval_stochastic is True
    assert flags.voc_dueling_q is True
    assert flags.voc_expected_gate_loss is True
    assert flags.voc_dedicated_gate is True
    assert flags.voc_soft_q_bce_gate is True
    assert flags.voc_gate_q_temperature == pytest.approx(0.05)
    assert flags.voc_gate_confidence_weighted is True
    assert flags.voc_gate_adam_beta1 == pytest.approx(0.9)
    assert flags.voc_gate_param_align is False
    assert flags.voc_gate_param_align_coef == 1.0
    assert flags.voc_gate_exact_projection is False
    assert flags.voc_gate_epsilon_greedy_execution is False
    assert flags.voc_gate_execution_epsilon == pytest.approx(0.02)
    assert flags.voc_actor_policy_version_barrier is False
    assert flags.voc_actor_policy_bundle_schema_version == 1
    assert flags.voc_actor_policy_barrier_timeout_s == pytest.approx(120.0)
    assert flags.voc_actor_policy_ray_max_restarts == 0
    assert flags.voc_actor_policy_ray_max_task_retries == 0
    # Legacy/off smoke retains the historical Namespace shape; the runtime
    # contract still resolves a missing seal identity to exact legacy zero.
    assert not hasattr(flags, "voc_model_input_seal_schema_version")
    assert flags.actor_amp_init_scale == pytest.approx(256.0)
    assert flags.voc_gate_learning_rate == pytest.approx(0.0003)
    assert flags.voc_gate_grad_norm_clipping == pytest.approx(1.0)
    assert flags.model_size_nn == 2
    assert flags.frame_stack_n == 4
    assert flags.batch_length == 4
    assert flags.float16 is False
    assert flags.model_float16 is False

    args.dynamic_voc_mode = "shadow"
    shadow_flags = smoke._load_flags(args)
    assert shadow_flags.dynamic_voc_mode == "shadow"
    assert shadow_flags.voc_dueling_q is True
    assert shadow_flags.voc_expected_gate_loss is True
    assert shadow_flags.voc_dedicated_gate is True
    assert shadow_flags.voc_soft_q_bce_gate is True

    args.voc_dueling_q = False
    with pytest.raises(ValueError, match="voc_dueling_q=true"):
        smoke._load_flags(args)

    args.voc_dueling_q = True
    args.voc_expected_gate_loss = False
    with pytest.raises(ValueError, match="voc_expected_gate_loss=true"):
        smoke._load_flags(args)


def test_checkpoint_smoke_flags_preserve_configured_precision(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "config_c.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "Enduro-v5",
                "dynamic_search": True,
                "dynamic_factorized_control": True,
                "sep_im_head": True,
                "max_search_steps": 20,
                "float16": True,
                "model_float16": False,
                "parallel": True,
                "parallel_actor": True,
                "train_actor": True,
                "use_wandb": True,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    args = Namespace(
        config=None,
        checkpoint_dir=checkpoint_dir,
        env_name="Enduro-v5",
        rec_t=None,
        max_search_steps=None,
        max_depth=None,
        model_unroll_len=None,
        think_cost=None,
        model_size_nn=None,
        frame_stack_n=None,
        grayscale=None,
        tree_carry=None,
        scored_length=4,
        device=torch.device("cpu"),
    )

    flags = smoke._load_flags(args)

    assert flags.float16 is True
    assert flags.model_float16 is False
    assert flags.parallel is True
    assert flags.parallel_actor is True
    assert flags.train_actor is True
    assert flags.use_wandb is True


def test_bound_smoke_config_preserves_overrides_and_explicit_config_precedence(
    tmp_path,
):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    checkpoint_config = {
        "name": "Enduro-v5",
        "dynamic_search": True,
        "dynamic_factorized_control": True,
        "sep_im_head": True,
        "max_search_steps": 20,
        "batch_length": 4,
        "xpid": "bound-smoke-config-fixture",
    }
    checkpoint_bytes = yaml.safe_dump(
        checkpoint_config, sort_keys=True
    ).encode("utf-8")
    (checkpoint_dir / "config_c.yaml").write_bytes(checkpoint_bytes)
    args = Namespace(
        config=None,
        checkpoint_dir=checkpoint_dir,
        env_name="Enduro-v5",
        rec_t=17,
        max_search_steps=19,
        max_depth=None,
        model_unroll_len=None,
        think_cost=None,
        model_size_nn=None,
        frame_stack_n=2,
        grayscale=True,
        tree_carry=False,
        scored_length=4,
        device=torch.device("cpu"),
    )

    pathname_flags = smoke._load_flags(args)
    bound_flags = smoke._load_flags(
        args,
        config_payload=checkpoint_bytes,
        expected_sha256=hashlib.sha256(checkpoint_bytes).hexdigest(),
    )

    assert vars(bound_flags) == vars(pathname_flags)
    assert (
        bound_flags.frame_stack_n,
        bound_flags.grayscale,
        bound_flags.tree_carry,
        bound_flags.rec_t,
        bound_flags.max_search_steps,
    ) == (2, True, False, 17, 19)

    explicit_config = tmp_path / "explicit.yaml"
    explicit_values = dict(checkpoint_config)
    explicit_values["rec_t"] = 13
    explicit_bytes = yaml.safe_dump(explicit_values, sort_keys=True).encode(
        "utf-8"
    )
    explicit_config.write_bytes(explicit_bytes)
    args.config = explicit_config
    explicit_pathname_flags = smoke._load_flags(args)
    explicit_bound_flags = smoke._load_flags(
        args,
        config_payload=explicit_bytes,
        expected_sha256=hashlib.sha256(explicit_bytes).hexdigest(),
    )
    assert vars(explicit_bound_flags) == vars(explicit_pathname_flags)


def _actual_schema13_smoke_flags(monkeypatch, tmp_path):
    xpid, base_seed, total_steps, warm_up, unroll, use_wandb = (
        util.VOC_GATE_POLICY_SCHEMA13_STAGE_PROFILES[0]
    )
    root = tmp_path.resolve()
    savedir = root / "runs"
    surface = {
        **dict(util.VOC_GATE_POLICY_SCHEMA12_V12_PROJECTION),
        "xpid": xpid,
        "base_seed": base_seed,
        "total_steps": total_steps,
        "model_warm_up_n": warm_up,
        "actor_unroll_len": unroll,
        "use_wandb": use_wandb,
        "savedir": str(savedir),
        "ckpdir": str(savedir / xpid),
        "cmd": " ".join(sys.argv),
        "icopro_data_path": str(root / "data" / "behavioral_data_block"),
        "voc_gate_policy_schema_version": 13,
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


def _schema13_smoke_flag_args(flags):
    return Namespace(
        config=None,
        checkpoint_dir=Path(flags.ckpdir),
        env_name=flags.name,
        rec_t=None,
        max_search_steps=None,
        max_depth=None,
        model_unroll_len=None,
        think_cost=None,
        model_size_nn=None,
        frame_stack_n=None,
        grayscale=None,
        tree_carry=None,
        scored_length=flags.batch_length,
        device=torch.device("cpu"),
    )


def test_schema13_smoke_bound_flags_use_private_validated_byte_path(
    monkeypatch, tmp_path
):
    created = _actual_schema13_smoke_flags(monkeypatch, tmp_path)
    payload = yaml.safe_dump(vars(created), sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    args = _schema13_smoke_flag_args(created)

    loaded = smoke._load_flags(
        args,
        config_payload=payload,
        expected_sha256=digest,
        schema13_bound=True,
    )

    assert vars(loaded) == vars(created)
    args.config = tmp_path / "explicit-schema13.yaml"
    with pytest.raises(ValueError, match="forbids explicit user --config"):
        smoke._load_flags(
            args,
            config_payload=payload,
            expected_sha256=digest,
            schema13_bound=True,
        )


def test_schema13_smoke_bound_flags_reject_malformed_surface_without_legacy_loader(
    monkeypatch, tmp_path
):
    created = _actual_schema13_smoke_flags(monkeypatch, tmp_path)
    malformed = dict(vars(created))
    malformed["voc_gate_policy_schema_version"] = 12
    payload = yaml.safe_dump(malformed, sort_keys=True).encode("utf-8")
    calls = []

    def forbidden_create_flags(*args, **kwargs):
        calls.append("create_flags")
        raise AssertionError("malformed schema-13 smoke reached legacy loading")

    monkeypatch.setattr(util, "create_flags", forbidden_create_flags)
    with pytest.raises(ValueError, match="schema|surface|xpid"):
        smoke._load_flags(
            _schema13_smoke_flag_args(created),
            config_payload=payload,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            schema13_bound=True,
        )

    assert calls == []


def test_smoke_local_schema13_classifier_accepts_frozen_lexical_carriers():
    class StringSubclass(str):
        pass

    prefix = "enduro-voc-v20-telemetry-tau1-orthocd-adam-eps25-malformed"
    for value in (
        prefix,
        StringSubclass(prefix),
        prefix.encode("utf-8"),
        bytearray(prefix.encode("utf-8")),
        memoryview(prefix.encode("utf-8")),
        np.bytes_(prefix),
        Path(prefix),
    ):
        assert smoke._smoke_schema13_xpid_claims_intent(value) is True


def test_smoke_parser_exposes_voc_v3_options(tmp_path):
    args = smoke.parse_args([
        "--env-name",
        "Enduro-v5",
        "--game-id",
        "3",
        "--data-root",
        str(tmp_path),
        "--no-voc-dueling-q",
        "--voc-expected-gate-loss",
        "--no-voc-dedicated-gate",
        "--voc-soft-q-bce-gate",
        "--voc-gate-q-temperature",
        "0.07",
        "--no-voc-gate-confidence-weighted",
        "--voc-gate-adam-beta1",
        "0.25",
        "--voc-gate-param-align",
        "--voc-gate-param-align-coef",
        "1.0",
        "--no-voc-gate-exact-projection",
        "--no-voc-gate-epsilon-greedy-execution",
        "--voc-gate-execution-epsilon",
        "0.25",
        "--voc-actor-policy-version-barrier",
        "--voc-actor-policy-bundle-schema-version",
        "1",
        "--voc-actor-policy-barrier-timeout-s",
        "120.0",
        "--voc-actor-policy-ray-max-restarts",
        "0",
        "--voc-actor-policy-ray-max-task-retries",
        "0",
        "--actor-amp-init-scale",
        "32.0",
        "--voc-gate-learning-rate",
        "0.0004",
        "--voc-gate-grad-norm-clipping",
        "1.5",
    ])

    assert args.voc_dueling_q is False
    assert args.voc_expected_gate_loss is True
    assert args.voc_dedicated_gate is False
    assert args.voc_soft_q_bce_gate is True
    assert args.voc_gate_q_temperature == pytest.approx(0.07)
    assert args.voc_gate_confidence_weighted is False
    assert args.voc_gate_adam_beta1 == pytest.approx(0.25)
    assert args.voc_gate_param_align is True
    assert args.voc_gate_param_align_coef == 1.0
    assert args.voc_gate_exact_projection is False
    assert args.voc_gate_epsilon_greedy_execution is False
    assert args.voc_gate_execution_epsilon == 0.25
    assert args.voc_actor_policy_version_barrier is True
    assert args.voc_actor_policy_bundle_schema_version == 1
    assert args.voc_actor_policy_barrier_timeout_s == 120.0
    assert args.voc_actor_policy_ray_max_restarts == 0
    assert args.voc_actor_policy_ray_max_task_retries == 0
    assert args.actor_amp_init_scale == 32.0
    assert args.voc_gate_learning_rate == pytest.approx(0.0004)
    assert args.voc_gate_grad_norm_clipping == pytest.approx(1.5)


def test_smoke_active_flags_accept_unweighted_soft_target_gate(tmp_path):
    args = smoke.parse_args([
        "--env-name",
        "Enduro-v5",
        "--game-id",
        "3",
        "--data-root",
        str(tmp_path),
        "--device",
        "cpu",
        "--dynamic-voc-mode",
        "control",
        "--no-voc-gate-confidence-weighted",
        "--voc-gate-adam-beta1",
        "0",
        "--voc-gate-param-align",
    ])

    flags = smoke._load_flags(args)

    assert flags.dynamic_voc_mode == "control"
    assert flags.voc_gate_confidence_weighted is False
    assert flags.voc_gate_adam_beta1 == 0.0
    assert flags.voc_gate_param_align is True
    assert flags.voc_gate_param_align_coef == 1.0


@pytest.mark.parametrize("bad_beta1", ["-0.1", "1", "nan"])
def test_smoke_active_flags_reject_invalid_gate_adam_beta1(
    tmp_path, bad_beta1
):
    args = smoke.parse_args([
        "--env-name",
        "Enduro-v5",
        "--game-id",
        "3",
        "--data-root",
        str(tmp_path),
        "--device",
        "cpu",
        "--dynamic-voc-mode",
        "control",
        "--voc-gate-adam-beta1",
        bad_beta1,
    ])

    with pytest.raises(ValueError, match="voc_gate_adam_beta1"):
        smoke._load_flags(args)


def test_smoke_parser_exposes_negative_alignment_boolean(tmp_path):
    args = smoke.parse_args([
        "--env-name",
        "Enduro-v5",
        "--game-id",
        "3",
        "--data-root",
        str(tmp_path),
        "--no-voc-gate-param-align",
    ])

    assert args.voc_gate_param_align is False


def test_smoke_parser_and_flags_expose_exact_projection(tmp_path):
    args = smoke.parse_args([
        "--env-name",
        "Enduro-v5",
        "--game-id",
        "3",
        "--data-root",
        str(tmp_path),
        "--device",
        "cpu",
        "--dynamic-voc-mode",
        "control",
        "--no-voc-gate-param-align",
        "--voc-gate-exact-projection",
    ])

    flags = smoke._load_flags(args)

    assert args.voc_gate_exact_projection is True
    assert flags.voc_gate_param_align is False
    assert flags.voc_gate_param_align_coef == 1.0
    assert flags.voc_gate_exact_projection is True
    assert smoke.parse_args([
        "--env-name",
        "Enduro-v5",
        "--game-id",
        "3",
        "--data-root",
        str(tmp_path),
        "--no-voc-gate-exact-projection",
    ]).voc_gate_exact_projection is False


def test_smoke_parser_and_flags_expose_epsilon_greedy_execution(tmp_path):
    args = smoke.parse_args([
        "--env-name",
        "Enduro-v5",
        "--game-id",
        "3",
        "--data-root",
        str(tmp_path),
        "--device",
        "cpu",
        "--dynamic-voc-mode",
        "control",
        "--no-voc-gate-param-align",
        "--voc-gate-exact-projection",
        "--voc-gate-epsilon-greedy-execution",
    ])

    flags = smoke._load_flags(args)

    assert args.voc_gate_epsilon_greedy_execution is True
    assert flags.voc_gate_param_align is False
    assert flags.voc_gate_param_align_coef == 1.0
    assert flags.voc_gate_exact_projection is True
    assert flags.voc_gate_epsilon_greedy_execution is True
    assert smoke.parse_args([
        "--env-name",
        "Enduro-v5",
        "--game-id",
        "3",
        "--data-root",
        str(tmp_path),
        "--no-voc-gate-epsilon-greedy-execution",
    ]).voc_gate_epsilon_greedy_execution is False


def test_smoke_flags_reject_epsilon_greedy_execution_without_projection(
    tmp_path,
):
    args = smoke.parse_args([
        "--env-name",
        "Enduro-v5",
        "--game-id",
        "3",
        "--data-root",
        str(tmp_path),
        "--device",
        "cpu",
        "--dynamic-voc-mode",
        "control",
        "--no-voc-gate-param-align",
        "--no-voc-gate-exact-projection",
        "--voc-gate-epsilon-greedy-execution",
    ])

    with pytest.raises(ValueError, match="voc_gate_exact_projection=true"):
        smoke._load_flags(args)


def test_smoke_flags_reject_alignment_with_exact_projection(tmp_path):
    args = smoke.parse_args([
        "--env-name",
        "Enduro-v5",
        "--game-id",
        "3",
        "--data-root",
        str(tmp_path),
        "--device",
        "cpu",
        "--dynamic-voc-mode",
        "control",
        "--voc-gate-param-align",
        "--voc-gate-exact-projection",
    ])

    with pytest.raises(ValueError, match="mutually exclusive"):
        smoke._load_flags(args)


@pytest.mark.parametrize(
    "bad_coefficient", ["0.9999999999999999", "1.0000000000000002", "nan"]
)
def test_smoke_exact_projection_requires_exact_unit_coefficient(
    tmp_path, bad_coefficient
):
    args = smoke.parse_args([
        "--env-name",
        "Enduro-v5",
        "--game-id",
        "3",
        "--data-root",
        str(tmp_path),
        "--device",
        "cpu",
        "--dynamic-voc-mode",
        "control",
        "--no-voc-gate-param-align",
        "--voc-gate-exact-projection",
        "--voc-gate-param-align-coef",
        bad_coefficient,
    ])

    with pytest.raises(ValueError, match="voc_gate_param_align_coef"):
        smoke._load_flags(args)


@pytest.mark.parametrize(
    "bad_coefficient", ["0.9999999999999999", "1.0000000000000002", "nan"]
)
def test_smoke_flags_require_exact_unit_alignment_coefficient(
    tmp_path, bad_coefficient
):
    args = smoke.parse_args([
        "--env-name",
        "Enduro-v5",
        "--game-id",
        "3",
        "--data-root",
        str(tmp_path),
        "--device",
        "cpu",
        "--dynamic-voc-mode",
        "shadow",
        "--voc-gate-param-align",
        "--voc-gate-param-align-coef",
        bad_coefficient,
    ])

    with pytest.raises(ValueError, match="voc_gate_param_align_coef"):
        smoke._load_flags(args)


def _schema5_identity_values():
    return {
        "base_seed": 4,
        "total_steps": 300000,
        "schedule_total_steps": 100000000,
        "dynamic_voc_mode": "control",
        "voc_gate_adam_beta1": 0.0,
        "voc_gate_param_align": False,
        "voc_gate_param_align_coef": 1.0,
        "voc_gate_exact_projection": True,
        "voc_gate_epsilon_greedy_execution": True,
        "voc_train_epsilon": 0.02,
        "voc_eval_stochastic": True,
        "float16": True,
        "model_float16": False,
        "ckp": False,
        "preload": "",
        "preload_actor": "",
        "voc_parent_checkpoint": "",
    }


def _schema5_active_state():
    return {
        "dynamic_voc_mode": "control",
        "voc_gate_policy_schema_version": 5,
        "voc_gate_adam_beta1": 0.0,
        "voc_gate_param_align": False,
        "voc_gate_param_align_coef": 1.0,
        "voc_gate_exact_projection": True,
        "voc_gate_epsilon_greedy_execution": True,
    }


def _patch_public_checkpoint_validators(monkeypatch):
    calls = []

    def validate_actor(checkpoint, flags, spec):
        calls.append(("actor", checkpoint, flags, spec))
        return {"embedded_protocol_verified": True, "source": "actor"}

    def validate_model(checkpoint, flags, spec):
        calls.append(("model", checkpoint, flags, spec))
        return {"embedded_protocol_verified": True, "source": "model"}

    monkeypatch.setattr(
        evaluation, "validate_actor_imitation_checkpoint", validate_actor
    )
    monkeypatch.setattr(evaluation, "validate_model_checkpoint", validate_model)
    return calls


def test_smoke_checkpoint_metadata_validates_and_records_actor_and_model(
    monkeypatch,
):
    identity = _schema5_identity_values()
    flags = Namespace(**identity)
    actor = {"flags": dict(identity)}
    model = {"flags": dict(identity)}
    spec = object()
    calls = _patch_public_checkpoint_validators(monkeypatch)

    resolved = smoke._validate_smoke_checkpoint_metadata(
        actor, model, flags, spec, _schema5_active_state()
    )

    assert [(kind, checkpoint) for kind, checkpoint, _, _ in calls] == [
        ("actor", actor),
        ("model", model),
    ]
    assert all(call[2] is flags and call[3] is spec for call in calls)
    assert resolved["actor_public_validation"]["source"] == "actor"
    assert resolved["model_public_validation"]["source"] == "model"
    assert resolved["model_gate_policy"] == {
        "voc_gate_policy_schema_version": 5,
        "voc_gate_adam_beta1": 0.0,
        "voc_gate_adam_beta1_legacy_defaulted": False,
        "voc_gate_param_align": False,
        "voc_gate_param_align_coef": 1.0,
        "voc_gate_param_align_legacy_defaulted": False,
        "voc_gate_exact_projection": True,
        "voc_gate_exact_projection_legacy_defaulted": False,
        "voc_gate_epsilon_greedy_execution": True,
        "voc_gate_epsilon_greedy_execution_legacy_defaulted": False,
    }
    for source in ("config", "actor_checkpoint", "model_checkpoint"):
        assert resolved["resolved_identity"][source] == {
            **identity,
            "voc_gate_policy_schema_version": 5,
        }
    json.dumps(resolved, allow_nan=False)


def _schema6_active_state_for_smoke():
    return {
        "dynamic_voc_mode": "control",
        "voc_gate_policy_schema_version": 6,
        "voc_gate_adam_beta1": 0.0,
        "voc_gate_param_align": False,
        "voc_gate_param_align_coef": 1.0,
        "voc_gate_exact_projection": True,
        "voc_gate_epsilon_greedy_execution": True,
    }


def _schema7_active_state_for_smoke():
    return {
        **_schema6_active_state_for_smoke(),
        "voc_gate_policy_schema_version": 7,
        "voc_model_input_seal_schema_version": 1,
    }


def _schema8_active_state_for_smoke():
    return {
        **_schema7_active_state_for_smoke(),
        "voc_gate_policy_schema_version": 8,
    }


def _schema9_active_state_for_smoke():
    return {
        **_schema8_active_state_for_smoke(),
        "voc_gate_policy_schema_version": 9,
    }


def _schema10_active_state_for_smoke():
    return {
        **_schema9_active_state_for_smoke(),
        "voc_gate_policy_schema_version": 10,
    }


def _schema11_active_state_for_smoke():
    active_state = {
        name: None
        for name in smoke._SCHEMA10_FINAL_ACTOR_EVIDENCE_FIELDS
    }
    active_state.update({
        **_schema10_active_state_for_smoke(),
        "voc_gate_policy_schema_version": 11,
        "voc_q_regression_loss": "smooth_l1_beta1",
        "voc_q_reconstruction": (
            "detached_value_plus_raw_head_mean_plus_policy_centered_raw_head"
        ),
        "voc_q_optimizer_coordinates": (
            "orthonormal_common_difference_adam"
        ),
    })
    return active_state


def _schema12_active_state_for_smoke():
    return {
        **_schema11_active_state_for_smoke(),
        "voc_gate_policy_schema_version": 12,
        "voc_gate_target_tau": 1.0,
    }


def _schema13_active_state_for_smoke():
    return {
        **_schema12_active_state_for_smoke(),
        "voc_gate_policy_schema_version": 13,
        "voc_update_count": 2,
    }


def test_smoke_checkpoint_metadata_records_completed_schema6_surfaces(
    monkeypatch,
):
    _patch_public_checkpoint_validators(monkeypatch)
    active_state = _schema6_active_state_for_smoke()
    monkeypatch.setattr(
        evaluation,
        "_validate_voc_gate_policy_schema",
        lambda checkpoint, embedded, label: dict(active_state),
    )
    resolved_surface = {
        "key_count": 228,
        "v12_projection_key_count": 209,
        "v12_projection_sha256": "a" * 64,
        "complete_surface_sha256": "b" * 64,
        "stage": ["wire", 1, 1200, 512, 41, False],
        "paths": {"ckpdir": "/sealed/run"},
    }
    completed = {
        "stored_surface_identity": {
            name: dict(resolved_surface)
            for name in ("config", "actor_checkpoint", "model_checkpoint")
        },
        "public_finish_verified": True,
        "private_logger_markers_absent": True,
    }
    flags = {"dynamic_voc_mode": "control"}

    result = smoke._validate_smoke_checkpoint_metadata(
        {"flags": dict(flags)},
        {"flags": dict(flags)},
        Namespace(**flags),
        object(),
        active_state,
        completed,
    )

    assert result["resolved_identity"] == completed["stored_surface_identity"]
    assert result["schema6_completed_bundle_validation"] == completed
    json.dumps(result, sort_keys=True, allow_nan=False)


def test_smoke_checkpoint_metadata_rejects_schema6_without_completed_bundle(
    monkeypatch,
):
    _patch_public_checkpoint_validators(monkeypatch)
    active_state = _schema6_active_state_for_smoke()
    monkeypatch.setattr(
        evaluation,
        "_validate_voc_gate_policy_schema",
        lambda checkpoint, embedded, label: dict(active_state),
    )
    flags = {"dynamic_voc_mode": "control"}

    with pytest.raises(ValueError, match="lacks completed-bundle validation"):
        smoke._validate_smoke_checkpoint_metadata(
            {"flags": dict(flags)},
            {"flags": dict(flags)},
            Namespace(**flags),
            object(),
            active_state,
        )


def test_smoke_checkpoint_metadata_records_completed_schema7_surfaces(
    monkeypatch,
):
    _patch_public_checkpoint_validators(monkeypatch)
    active_state = _schema7_active_state_for_smoke()
    monkeypatch.setattr(
        evaluation,
        "_validate_voc_gate_policy_schema",
        lambda checkpoint, embedded, label: dict(active_state),
    )
    resolved_surface = {
        "gate_schema": 7,
        "voc_gate_policy_schema_version": 7,
        "voc_model_input_seal_schema_version": 1,
        "key_count": 229,
        "v12_projection_key_count": 209,
        "v12_projection_sha256": "a" * 64,
        "complete_surface_sha256": "b" * 64,
        "stage": ["v14-wire", 1, 1200, 512, 41, False],
        "paths": {"ckpdir": "/sealed/run"},
    }
    completed = {
        "stored_surface_identity": {
            name: dict(resolved_surface)
            for name in ("config", "actor_checkpoint", "model_checkpoint")
        },
        "model_input_seal": {
            "voc_model_input_seal_schema_version": 1,
            "voc_model_input_sealed": True,
        },
        "public_finish_verified": True,
        "private_logger_markers_absent": True,
    }
    flags = {
        "dynamic_voc_mode": "control",
        "voc_model_input_seal_schema_version": 1,
    }

    result = smoke._validate_smoke_checkpoint_metadata(
        {"flags": dict(flags)},
        {"flags": dict(flags)},
        Namespace(**flags),
        object(),
        active_state,
        None,
        completed,
    )

    assert result["resolved_identity"] == completed[
        "stored_surface_identity"
    ]
    assert result["schema7_completed_bundle_validation"] == completed
    assert result["schema6_completed_bundle_validation"] is None
    assert "schema8_completed_bundle_validation" not in result
    json.dumps(result, sort_keys=True, allow_nan=False)


def test_smoke_checkpoint_metadata_rejects_schema7_without_completed_bundle(
    monkeypatch,
):
    _patch_public_checkpoint_validators(monkeypatch)
    active_state = _schema7_active_state_for_smoke()
    monkeypatch.setattr(
        evaluation,
        "_validate_voc_gate_policy_schema",
        lambda checkpoint, embedded, label: dict(active_state),
    )
    flags = {
        "dynamic_voc_mode": "control",
        "voc_model_input_seal_schema_version": 1,
    }

    with pytest.raises(ValueError, match="schema-7.*completed-bundle"):
        smoke._validate_smoke_checkpoint_metadata(
            {"flags": dict(flags)},
            {"flags": dict(flags)},
            Namespace(**flags),
            object(),
            active_state,
        )


def _schema8_completed_smoke_record():
    resolved_surface = {
        "gate_schema": 8,
        "voc_gate_policy_schema_version": 8,
        "voc_model_input_seal_schema_version": 1,
        "voc_q_regression_loss": "half_squared_td",
        "key_count": 229,
        "v12_projection_key_count": 209,
        "v12_projection_sha256": "a" * 64,
        "complete_surface_sha256": "b" * 64,
        "stage": [
            "enduro-voc-v15-halfsq-eps25-sentinel-wire1200",
            1,
            1200,
            512,
            41,
            False,
        ],
        "paths": {
            "savedir": "/sealed/runs",
            "ckpdir": "/sealed/run",
            "cmd": "python train.py",
            "icopro_data_path": "/sealed/data",
        },
    }
    return {
        "stored_surface_identity": {
            name: copy.deepcopy(resolved_surface)
            for name in ("config", "actor_checkpoint", "model_checkpoint")
        },
        "model_input_seal": {
            "voc_model_input_seal_schema_version": 1,
            "voc_model_input_sealed": True,
        },
        "public_finish_verified": True,
        "private_logger_markers_absent": True,
    }


def _schema9_completed_smoke_record():
    completed = _schema8_completed_smoke_record()
    for surface in completed["stored_surface_identity"].values():
        surface.update(
            {
                "gate_schema": 9,
                "voc_gate_policy_schema_version": 9,
                "voc_q_reconstruction": (
                    "detached_value_plus_raw_head_mean_plus_"
                    "policy_centered_raw_head"
                ),
                "stage": [
                    "enduro-voc-v16-commonmode-eps25-sentinel-wire1200",
                    1,
                    1200,
                    512,
                    41,
                    False,
                ],
            }
        )
    return completed


def _schema10_completed_smoke_record():
    completed = _schema9_completed_smoke_record()
    for surface in completed["stored_surface_identity"].values():
        surface.update(
            {
                "gate_schema": 10,
                "voc_gate_policy_schema_version": 10,
                "voc_q_regression_loss": "smooth_l1_beta1",
                "stage": [
                    "enduro-voc-v17-huber-common-eps25-sentinel-wire1200",
                    1,
                    1200,
                    512,
                    41,
                    False,
                ],
            }
        )
    return completed


def _schema11_completed_smoke_record():
    completed = _schema10_completed_smoke_record()
    xpid = "enduro-voc-v18-orthocd-adam-eps25-sentinel-wire1200"
    for surface in completed["stored_surface_identity"].values():
        surface.update(
            {
                "gate_schema": 11,
                "voc_gate_policy_schema_version": 11,
                "voc_q_optimizer_coordinates": (
                    "orthonormal_common_difference_adam"
                ),
                "stage": [
                    xpid,
                    1,
                    1200,
                    512,
                    41,
                    False,
                ],
                "v12_projection_sha256": (
                    evaluation.VOC_GATE_POLICY_SCHEMA7_V12_PROJECTION_SHA256
                ),
                "paths": {
                    "savedir": "/sealed/runs",
                    "ckpdir": f"/sealed/runs/{xpid}",
                    "cmd": "python train.py",
                    "icopro_data_path": (
                        "/sealed/data/behavioral_data_block"
                    ),
                },
            }
        )
    completed["resolved_identity"] = copy.deepcopy(
        completed["stored_surface_identity"]["config"]
    )
    completed.update(
        {
            "authoritative_validator": (
                "thinker.util.validate_schema11_final_bundle"
            ),
            "actor_policy": {
                name: None
                for name in evaluation.ATOMIC_ACTOR_POLICY_EVIDENCE_FIELDS
            },
            "actor_training_state": {},
            "model_step": 1,
            "model_real_step": 1200,
            "model_state_tensor_count": 1,
            "model_optimizer_state": {},
            "model_scheduler_state": {},
            "model_scaler_state": {},
            "config_use_wandb": False,
            "completion_evidence": {},
            "logger_completion": {},
        }
    )
    return completed


def _schema12_completed_smoke_record():
    completed = _schema11_completed_smoke_record()
    xpid = "enduro-voc-v19-tau1-orthocd-adam-eps25-sentinel-wire1200"
    for surface in completed["stored_surface_identity"].values():
        surface.update(
            {
                "gate_schema": 12,
                "voc_gate_policy_schema_version": 12,
                "v12_projection_sha256": (
                    evaluation.VOC_GATE_POLICY_SCHEMA12_V12_PROJECTION_SHA256
                ),
                "stage": [xpid, 1, 1200, 512, 41, False],
                "paths": {
                    "savedir": "/sealed/runs",
                    "ckpdir": f"/sealed/runs/{xpid}",
                    "cmd": "python train.py",
                    "icopro_data_path": "/sealed/data/behavioral_data_block",
                },
            }
        )
    completed["resolved_identity"] = copy.deepcopy(
        completed["stored_surface_identity"]["config"]
    )
    completed["authoritative_validator"] = (
        "thinker.util.validate_schema12_final_bundle"
    )
    return completed


def _schema13_completed_smoke_record():
    completed = _schema12_completed_smoke_record()
    xpid = "enduro-voc-v20-telemetry-tau1-orthocd-adam-eps25-sentinel-wire1200"
    for surface in completed["stored_surface_identity"].values():
        surface.update(
            {
                "gate_schema": 13,
                "voc_gate_policy_schema_version": 13,
                "stage": [xpid, 1, 1200, 512, 41, False],
                "paths": {
                    "savedir": "/sealed/runs",
                    "ckpdir": f"/sealed/runs/{xpid}",
                    "cmd": "python train.py",
                    "icopro_data_path": "/sealed/data/behavioral_data_block",
                },
            }
        )
    completed["resolved_identity"] = copy.deepcopy(
        completed["stored_surface_identity"]["config"]
    )
    completed["authoritative_validator"] = (
        "thinker.util.validate_schema13_final_bundle"
    )
    completed["actor_policy"].update(
        {
            "voc_actor_policy_version": 2,
            "voc_actor_policy_state_sha256": "f" * 64,
            "voc_actor_policy_publication_history_sha256": "1" * 64,
        }
    )
    manifest_record = {"sha256": "9" * 64, "size": 4096}
    completed["completion_evidence"] = {
        "checkpoint_files": {
            "config_c.yaml": {"sha256": "a" * 64, "size": 1},
            "ckp_actor.tar": {"sha256": "b" * 64, "size": 2},
            "ckp_model.tar": {"sha256": "c" * 64, "size": 3},
            "voc_telemetry_manifest.json": manifest_record,
        },
        "implementation_sources": {},
        "loaded_extensions": {},
    }
    actor_policy = completed["actor_policy"]
    active_state = _schema13_active_state_for_smoke()
    completed["telemetry"] = {
        "telemetry_schema_version": 1,
        "gate_schema": 13,
        "manifest_name": "voc_telemetry_manifest.json",
        "manifest_sha256": manifest_record["sha256"],
        "manifest_size": manifest_record["size"],
        "transaction_count": active_state["voc_update_count"],
        "terminal_policy_version": actor_policy["voc_actor_policy_version"],
        "terminal_real_step": 1200,
        "actor_state_sha256": actor_policy["voc_actor_policy_state_sha256"],
        "publication_history_sha256": actor_policy[
            "voc_actor_policy_publication_history_sha256"
        ],
    }
    return completed


def test_smoke_checkpoint_metadata_records_completed_schema8_identity(
    monkeypatch,
):
    _patch_public_checkpoint_validators(monkeypatch)
    active_state = _schema8_active_state_for_smoke()
    monkeypatch.setattr(
        evaluation,
        "_validate_voc_gate_policy_schema",
        lambda checkpoint, embedded, label: dict(active_state),
    )
    completed = _schema8_completed_smoke_record()
    flags = {
        "dynamic_voc_mode": "control",
        "voc_model_input_seal_schema_version": 1,
    }

    result = smoke._validate_smoke_checkpoint_metadata(
        {"flags": dict(flags)},
        {"flags": dict(flags)},
        Namespace(**flags),
        object(),
        active_state,
        None,
        None,
        completed,
    )

    assert result["schema8_completed_bundle_validation"] == completed
    assert result["resolved_identity"]["config"][
        "voc_q_regression_loss"
    ] == "half_squared_td"
    assert "schema7_completed_bundle_validation" not in result
    json.dumps(result, sort_keys=True, allow_nan=False)


@pytest.mark.parametrize("mutation", ["missing", "wrong"])
def test_smoke_checkpoint_metadata_rejects_schema8_loss_identity(
    monkeypatch, mutation
):
    _patch_public_checkpoint_validators(monkeypatch)
    active_state = _schema8_active_state_for_smoke()
    monkeypatch.setattr(
        evaluation,
        "_validate_voc_gate_policy_schema",
        lambda checkpoint, embedded, label: dict(active_state),
    )
    completed = _schema8_completed_smoke_record()
    for surface in completed["stored_surface_identity"].values():
        if mutation == "missing":
            surface.pop("voc_q_regression_loss")
        else:
            surface["voc_q_regression_loss"] = "smooth_l1"
    flags = {
        "dynamic_voc_mode": "control",
        "voc_model_input_seal_schema_version": 1,
    }

    with pytest.raises(ValueError, match="half-squared identity"):
        smoke._validate_smoke_checkpoint_metadata(
            {"flags": dict(flags)},
            {"flags": dict(flags)},
            Namespace(**flags),
            object(),
            active_state,
            None,
            None,
            completed,
        )


def test_smoke_checkpoint_metadata_records_completed_schema9_identity(
    monkeypatch,
):
    _patch_public_checkpoint_validators(monkeypatch)
    active_state = _schema9_active_state_for_smoke()
    monkeypatch.setattr(
        evaluation,
        "_validate_voc_gate_policy_schema",
        lambda checkpoint, embedded, label: dict(active_state),
    )
    completed = _schema9_completed_smoke_record()
    flags = {
        "dynamic_voc_mode": "control",
        "voc_model_input_seal_schema_version": 1,
    }

    result = smoke._validate_smoke_checkpoint_metadata(
        {"flags": dict(flags)},
        {"flags": dict(flags)},
        Namespace(**flags),
        object(),
        active_state,
        None,
        None,
        None,
        completed,
    )

    assert result["schema9_completed_bundle_validation"] == completed
    assert result["resolved_identity"]["config"][
        "voc_q_regression_loss"
    ] == "half_squared_td"
    assert result["resolved_identity"]["config"][
        "voc_q_reconstruction"
    ] == (
        "detached_value_plus_raw_head_mean_plus_policy_centered_raw_head"
    )
    assert "schema8_completed_bundle_validation" not in result
    json.dumps(result, sort_keys=True, allow_nan=False)


@pytest.mark.parametrize(
    ("field", "mutation"),
    [
        ("voc_q_regression_loss", "missing"),
        ("voc_q_regression_loss", "wrong"),
        ("voc_q_reconstruction", "missing"),
        ("voc_q_reconstruction", "wrong"),
    ],
)
def test_smoke_checkpoint_metadata_rejects_schema9_derived_identity(
    monkeypatch, field, mutation
):
    _patch_public_checkpoint_validators(monkeypatch)
    active_state = _schema9_active_state_for_smoke()
    monkeypatch.setattr(
        evaluation,
        "_validate_voc_gate_policy_schema",
        lambda checkpoint, embedded, label: dict(active_state),
    )
    completed = _schema9_completed_smoke_record()
    for surface in completed["stored_surface_identity"].values():
        if mutation == "missing":
            surface.pop(field)
        else:
            surface[field] = "wrong"
    flags = {
        "dynamic_voc_mode": "control",
        "voc_model_input_seal_schema_version": 1,
    }

    with pytest.raises(ValueError, match="common-mode identity"):
        smoke._validate_smoke_checkpoint_metadata(
            {"flags": dict(flags)},
            {"flags": dict(flags)},
            Namespace(**flags),
            object(),
            active_state,
            None,
            None,
            None,
            completed,
        )


def test_smoke_checkpoint_metadata_records_exact_schema10_identity_shape(
    monkeypatch,
):
    _patch_public_checkpoint_validators(monkeypatch)
    active_state = _schema10_active_state_for_smoke()
    monkeypatch.setattr(
        evaluation,
        "_validate_voc_gate_policy_schema",
        lambda checkpoint, embedded, label: dict(active_state),
    )
    completed = _schema10_completed_smoke_record()
    flags = {
        "dynamic_voc_mode": "control",
        "voc_model_input_seal_schema_version": 1,
    }

    result = smoke._validate_smoke_checkpoint_metadata(
        {"flags": dict(flags)},
        {"flags": dict(flags)},
        Namespace(**flags),
        object(),
        active_state,
        None,
        None,
        None,
        None,
        completed,
    )

    expected_keys = set(
        _schema9_completed_smoke_record()["stored_surface_identity"]["config"]
    )
    assert result["schema10_completed_bundle_validation"] == completed
    assert set(result["resolved_identity"]) == {
        "config",
        "actor_checkpoint",
        "model_checkpoint",
    }
    assert all(
        set(identity) == expected_keys
        for identity in result["resolved_identity"].values()
    )
    assert {
        identity["voc_q_regression_loss"]
        for identity in result["resolved_identity"].values()
    } == {"smooth_l1_beta1"}
    assert {
        identity["voc_q_reconstruction"]
        for identity in result["resolved_identity"].values()
    } == {
        "detached_value_plus_raw_head_mean_plus_policy_centered_raw_head"
    }
    assert "schema9_completed_bundle_validation" not in result
    json.dumps(result, sort_keys=True, allow_nan=False)


@pytest.mark.parametrize(
    ("field", "mutation"),
    [
        ("voc_q_regression_loss", "missing"),
        ("voc_q_regression_loss", "wrong"),
        ("voc_q_reconstruction", "missing"),
        ("voc_q_reconstruction", "wrong"),
        ("forged_authoritative_validator", "extra"),
    ],
)
def test_smoke_checkpoint_metadata_rejects_schema10_forged_identity(
    monkeypatch, field, mutation
):
    _patch_public_checkpoint_validators(monkeypatch)
    active_state = _schema10_active_state_for_smoke()
    monkeypatch.setattr(
        evaluation,
        "_validate_voc_gate_policy_schema",
        lambda checkpoint, embedded, label: dict(active_state),
    )
    completed = _schema10_completed_smoke_record()
    for surface in completed["stored_surface_identity"].values():
        if mutation == "missing":
            surface.pop(field)
        elif mutation == "extra":
            surface[field] = "forged"
        else:
            surface[field] = "wrong"
    flags = {
        "dynamic_voc_mode": "control",
        "voc_model_input_seal_schema_version": 1,
    }

    with pytest.raises(ValueError, match="Huber-common identity"):
        smoke._validate_smoke_checkpoint_metadata(
            {"flags": dict(flags)},
            {"flags": dict(flags)},
            Namespace(**flags),
            object(),
            active_state,
            None,
            None,
            None,
            None,
            completed,
        )


def test_smoke_checkpoint_metadata_records_exact_schema11_identity_shape(
    monkeypatch,
):
    _patch_public_checkpoint_validators(monkeypatch)
    active_state = _schema11_active_state_for_smoke()
    monkeypatch.setattr(
        evaluation,
        "_validate_voc_gate_policy_schema",
        lambda checkpoint, embedded, label: dict(active_state),
    )
    completed = _schema11_completed_smoke_record()
    flags = {
        "dynamic_voc_mode": "control",
        "voc_model_input_seal_schema_version": 1,
    }

    result = smoke._validate_smoke_checkpoint_metadata(
        {"flags": dict(flags)},
        {"flags": dict(flags)},
        Namespace(**flags),
        object(),
        active_state,
        None,
        None,
        None,
        None,
        None,
        completed,
    )

    expected_keys = (
        set(
            _schema10_completed_smoke_record()["stored_surface_identity"][
                "config"
            ]
        )
        | {"voc_q_optimizer_coordinates"}
    )
    assert set(active_state) == smoke._SCHEMA11_FINAL_ACTOR_EVIDENCE_FIELDS
    assert smoke._SCHEMA11_FINAL_ACTOR_EVIDENCE_FIELDS == (
        smoke._SCHEMA10_FINAL_ACTOR_EVIDENCE_FIELDS
        | {"voc_q_optimizer_coordinates"}
    )
    assert {
        name: active_state[name]
        for name in smoke._SCHEMA11_DERIVED_IDENTITY
    } == smoke._SCHEMA11_DERIVED_IDENTITY
    assert result["schema11_completed_bundle_validation"] == completed
    assert set(completed) == smoke._SCHEMA11_COMPLETED_BUNDLE_FIELDS
    assert set(completed["actor_policy"]) == (
        evaluation.ATOMIC_ACTOR_POLICY_EVIDENCE_FIELDS
    )
    assert set(result["resolved_identity"]) == {
        "config",
        "actor_checkpoint",
        "model_checkpoint",
    }
    assert all(
        set(identity) == expected_keys
        and identity == completed["resolved_identity"]
        for identity in result["resolved_identity"].values()
    )
    assert completed["resolved_identity"] == (
        completed["stored_surface_identity"]["config"]
    )
    assert {
        identity["voc_q_optimizer_coordinates"]
        for identity in result["resolved_identity"].values()
    } == {"orthonormal_common_difference_adam"}
    assert "schema10_completed_bundle_validation" not in result
    json.dumps(result, sort_keys=True, allow_nan=False)


def test_smoke_checkpoint_metadata_records_exact_schema12_mapped_keysets(
    monkeypatch,
):
    _patch_public_checkpoint_validators(monkeypatch)
    active_state = _schema12_active_state_for_smoke()
    monkeypatch.setattr(
        evaluation,
        "_validate_voc_gate_policy_schema",
        lambda checkpoint, embedded, label: dict(active_state),
    )
    completed = _schema12_completed_smoke_record()
    flags = {
        "dynamic_voc_mode": "control",
        "voc_model_input_seal_schema_version": 1,
    }

    result = smoke._validate_smoke_checkpoint_metadata(
        {"flags": dict(flags)},
        {"flags": dict(flags)},
        Namespace(**flags),
        object(),
        active_state,
        None,
        None,
        None,
        None,
        None,
        None,
        completed,
    )

    schema11_completed = _schema11_completed_smoke_record()
    assert set(active_state) == smoke._SCHEMA12_FINAL_ACTOR_EVIDENCE_FIELDS
    assert smoke._SCHEMA12_FINAL_ACTOR_EVIDENCE_FIELDS == (
        smoke._SCHEMA11_FINAL_ACTOR_EVIDENCE_FIELDS
    )
    assert set(completed) == set(schema11_completed)
    assert result["schema12_completed_bundle_validation"] == completed
    assert "schema11_completed_bundle_validation" not in result
    assert set(result["resolved_identity"]) == {
        "config",
        "actor_checkpoint",
        "model_checkpoint",
    }
    assert all(
        identity == completed["resolved_identity"]
        and set(identity) == set(schema11_completed["resolved_identity"])
        for identity in result["resolved_identity"].values()
    )
    assert set(completed["actor_policy"]) == (
        evaluation.ATOMIC_ACTOR_POLICY_EVIDENCE_FIELDS
    )


def test_smoke_checkpoint_metadata_records_exact_schema13_telemetry_shape(
    monkeypatch,
):
    _patch_public_checkpoint_validators(monkeypatch)
    active_state = _schema13_active_state_for_smoke()
    monkeypatch.setattr(
        evaluation,
        "_validate_voc_gate_policy_schema",
        lambda checkpoint, embedded, label: dict(active_state),
    )
    completed = _schema13_completed_smoke_record()
    flags = {
        "dynamic_voc_mode": "control",
        "voc_model_input_seal_schema_version": 1,
    }

    result = smoke._validate_smoke_checkpoint_metadata(
        {"flags": dict(flags)},
        {"flags": dict(flags)},
        Namespace(**flags),
        object(),
        active_state,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        completed,
    )

    assert set(completed) == smoke._SCHEMA13_COMPLETED_BUNDLE_FIELDS
    assert len(completed) == 18
    assert result["schema13_completed_bundle_validation"] == completed
    assert result["resolved_identity"] == completed["stored_surface_identity"]
    assert completed["telemetry"]["gate_schema"] == 13
    assert set(completed["actor_policy"]) == (
        evaluation.ATOMIC_ACTOR_POLICY_EVIDENCE_FIELDS
    )


@pytest.mark.parametrize(
    ("surface", "mutation"),
    [
        ("completed", "extra"),
        ("identity", "schema"),
        ("identity", "cross-surface"),
        ("telemetry", "digest"),
        ("telemetry", "count"),
        ("active", "typed-schema"),
    ],
)
def test_smoke_schema13_rejects_container_identity_and_telemetry_drift(
    monkeypatch, surface, mutation
):
    _patch_public_checkpoint_validators(monkeypatch)
    active_state = _schema13_active_state_for_smoke()
    completed = _schema13_completed_smoke_record()
    if surface == "completed":
        completed["forged"] = True
    elif surface == "identity" and mutation == "schema":
        completed["resolved_identity"]["gate_schema"] = 12
    elif surface == "identity":
        completed["stored_surface_identity"]["actor_checkpoint"][
            "complete_surface_sha256"
        ] = "c" * 64
    elif surface == "telemetry" and mutation == "digest":
        completed["telemetry"]["manifest_sha256"] = "8" * 64
    elif surface == "telemetry":
        completed["telemetry"]["transaction_count"] += 1
    else:
        active_state["voc_gate_policy_schema_version"] = 13.0
    model_state = _schema13_active_state_for_smoke()
    if surface == "active":
        model_state["voc_gate_policy_schema_version"] = 13.0
    monkeypatch.setattr(
        evaluation,
        "_validate_voc_gate_policy_schema",
        lambda checkpoint, embedded, label: dict(model_state),
    )
    flags = {
        "dynamic_voc_mode": "control",
        "voc_model_input_seal_schema_version": 1,
    }

    with pytest.raises(ValueError, match="schema-13|telemetry|identity"):
        smoke._validate_smoke_checkpoint_metadata(
            {"flags": dict(flags)},
            {"flags": dict(flags)},
            Namespace(**flags),
            object(),
            active_state,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            completed,
        )


@pytest.mark.parametrize(
    ("surface", "mutation"),
    [
        ("active", "missing"),
        ("active", "wrong"),
        ("active", "extra"),
        ("stored", "missing"),
        ("stored", "wrong"),
        ("stored", "extra"),
        ("completed", "missing"),
        ("completed", "extra"),
    ],
)
def test_smoke_schema12_rejects_identity_and_container_drift(
    monkeypatch, surface, mutation
):
    _patch_public_checkpoint_validators(monkeypatch)
    active_state = _schema12_active_state_for_smoke()
    completed = _schema12_completed_smoke_record()
    if surface == "active":
        if mutation == "missing":
            active_state.pop("voc_q_optimizer_coordinates")
        elif mutation == "wrong":
            active_state["voc_q_optimizer_coordinates"] = "forged"
        else:
            active_state["forged_identity"] = True
    elif surface == "stored":
        target = completed["stored_surface_identity"]["actor_checkpoint"]
        if mutation == "missing":
            target.pop("voc_q_optimizer_coordinates")
        elif mutation == "wrong":
            target["complete_surface_sha256"] = "c" * 64
        else:
            target["forged_identity"] = True
    elif mutation == "missing":
        completed.pop("actor_training_state")
    else:
        completed["forged_completed_evidence"] = True
    model_state = _schema12_active_state_for_smoke()
    monkeypatch.setattr(
        evaluation,
        "_validate_voc_gate_policy_schema",
        lambda checkpoint, embedded, label: dict(model_state),
    )
    flags = {
        "dynamic_voc_mode": "control",
        "voc_model_input_seal_schema_version": 1,
    }

    with pytest.raises(ValueError, match="schema-12|identity|actor evidence"):
        smoke._validate_smoke_checkpoint_metadata(
            {"flags": dict(flags)},
            {"flags": dict(flags)},
            Namespace(**flags),
            object(),
            active_state,
            None,
            None,
            None,
            None,
            None,
            None,
            completed,
        )


@pytest.mark.parametrize("mismatch", [None, "weight", "bias", "both"])
def test_smoke_schema12_separately_enforces_raw_ema_online_equality(
    tmp_path, mismatch
):
    checkpoint_dir = tmp_path / "schema12"
    checkpoint_dir.mkdir()
    online_weight = torch.tensor([[1.0, -0.0]], dtype=torch.float32)
    online_bias = torch.tensor([0.5, -0.0], dtype=torch.float32)
    ema_weight = online_weight.clone()
    ema_bias = online_bias.clone()
    if mismatch in ("weight", "both"):
        ema_weight[0, 0] += 1.0
    if mismatch in ("bias", "both"):
        ema_bias[0] += 1.0
    if mismatch is None:
        ema_weight[0, 1] = 0.0
        ema_bias[1] = 0.0
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
    marker = {
        "checkpoint_files": {
            "ckp_actor.tar": {
                "sha256": hashlib.sha256(actor_path.read_bytes()).hexdigest(),
                "size": actor_path.stat().st_size,
            }
        }
    }

    if mismatch is None:
        smoke._require_schema12_smoke_ema_online_equality(
            checkpoint_dir, marker
        )
    else:
        with pytest.raises(ValueError, match="raw EMA (weight|bias)"):
            smoke._require_schema12_smoke_ema_online_equality(
                checkpoint_dir, marker
            )


@pytest.mark.parametrize("mutation", ["missing", "wrong", "extra"])
def test_smoke_checkpoint_metadata_rejects_schema11_forged_identity(
    monkeypatch, mutation
):
    _patch_public_checkpoint_validators(monkeypatch)
    active_state = _schema11_active_state_for_smoke()
    monkeypatch.setattr(
        evaluation,
        "_validate_voc_gate_policy_schema",
        lambda checkpoint, embedded, label: dict(active_state),
    )
    completed = _schema11_completed_smoke_record()
    for surface in completed["stored_surface_identity"].values():
        if mutation == "missing":
            surface.pop("voc_q_optimizer_coordinates")
        elif mutation == "wrong":
            surface["voc_q_optimizer_coordinates"] = "raw_continue_stop_adam"
        else:
            surface["forged_optimizer_state"] = True
    flags = {
        "dynamic_voc_mode": "control",
        "voc_model_input_seal_schema_version": 1,
    }

    with pytest.raises(ValueError, match="stored-surface identity"):
        smoke._validate_smoke_checkpoint_metadata(
            {"flags": dict(flags)},
            {"flags": dict(flags)},
            Namespace(**flags),
            object(),
            active_state,
            None,
            None,
            None,
            None,
            None,
            completed,
        )


@pytest.mark.parametrize(
    ("field", "mutation"),
    [
        ("voc_q_regression_loss", "missing"),
        ("voc_q_regression_loss", "wrong"),
        ("voc_q_regression_loss", "none"),
        ("voc_q_reconstruction", "missing"),
        ("voc_q_reconstruction", "wrong"),
        ("voc_q_reconstruction", "none"),
        ("voc_q_optimizer_coordinates", "missing"),
        ("voc_q_optimizer_coordinates", "wrong"),
        ("voc_q_optimizer_coordinates", "none"),
        ("forged_actor_identity", "extra"),
    ],
)
def test_smoke_checkpoint_metadata_rejects_schema11_actor_only_identity_drift(
    monkeypatch, field, mutation
):
    _patch_public_checkpoint_validators(monkeypatch)
    active_state = _schema11_active_state_for_smoke()
    if mutation == "missing":
        active_state.pop(field)
    elif mutation == "wrong":
        active_state[field] = "forged"
    elif mutation == "none":
        active_state[field] = None
    else:
        active_state[field] = "forged"
    monkeypatch.setattr(
        evaluation,
        "_validate_voc_gate_policy_schema",
        lambda checkpoint, embedded, label: {
            **_schema11_active_state_for_smoke()
        },
    )
    completed = _schema11_completed_smoke_record()
    flags = {
        "dynamic_voc_mode": "control",
        "voc_model_input_seal_schema_version": 1,
    }

    with pytest.raises(ValueError, match="final actor evidence|authoritative"):
        smoke._validate_smoke_checkpoint_metadata(
            {"flags": dict(flags)},
            {"flags": dict(flags)},
            Namespace(**flags),
            object(),
            active_state,
            None,
            None,
            None,
            None,
            None,
            completed,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "count",
        "count_type",
        "projection_digest",
        "complete_digest",
        "stage",
        "stage_type",
        "path",
        "cross_surface",
        "missing_surface",
    ],
)
def test_smoke_checkpoint_metadata_rejects_schema11_authoritative_surface_drift(
    monkeypatch, mutation
):
    _patch_public_checkpoint_validators(monkeypatch)
    active_state = _schema11_active_state_for_smoke()
    monkeypatch.setattr(
        evaluation,
        "_validate_voc_gate_policy_schema",
        lambda checkpoint, embedded, label: dict(active_state),
    )
    completed = _schema11_completed_smoke_record()
    identities = [
        completed["resolved_identity"],
        *completed["stored_surface_identity"].values(),
    ]
    if mutation == "count":
        for identity in identities:
            identity["key_count"] = 230
    elif mutation == "count_type":
        for identity in identities:
            identity["key_count"] = np.int64(229)
    elif mutation == "projection_digest":
        for identity in identities:
            identity["v12_projection_sha256"] = "0" * 64
    elif mutation == "complete_digest":
        for identity in identities:
            identity["complete_surface_sha256"] = "G" * 64
    elif mutation == "stage":
        for identity in identities:
            identity["stage"] = list(identity["stage"])
            identity["stage"][2] = 1201
    elif mutation == "stage_type":
        for identity in identities:
            identity["stage"] = list(identity["stage"])
            identity["stage"][1] = np.int64(1)
    elif mutation == "path":
        for identity in identities:
            identity["paths"]["ckpdir"] = "/sealed/runs/wrong"
    elif mutation == "cross_surface":
        completed["stored_surface_identity"]["actor_checkpoint"][
            "complete_surface_sha256"
        ] = "c" * 64
    else:
        completed["stored_surface_identity"].pop("model_checkpoint")
    flags = {
        "dynamic_voc_mode": "control",
        "voc_model_input_seal_schema_version": 1,
    }

    with pytest.raises(
        ValueError,
        match="schema-11|identity|digest|stage|path|integer",
    ):
        smoke._validate_smoke_checkpoint_metadata(
            {"flags": dict(flags)},
            {"flags": dict(flags)},
            Namespace(**flags),
            object(),
            active_state,
            None,
            None,
            None,
            None,
            None,
            completed,
        )


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_smoke_checkpoint_metadata_rejects_schema11_completed_container_drift(
    monkeypatch, mutation
):
    _patch_public_checkpoint_validators(monkeypatch)
    active_state = _schema11_active_state_for_smoke()
    monkeypatch.setattr(
        evaluation,
        "_validate_voc_gate_policy_schema",
        lambda checkpoint, embedded, label: dict(active_state),
    )
    completed = _schema11_completed_smoke_record()
    if mutation == "missing":
        completed.pop("actor_training_state")
    else:
        completed["forged_completed_evidence"] = True
    flags = {
        "dynamic_voc_mode": "control",
        "voc_model_input_seal_schema_version": 1,
    }

    with pytest.raises(ValueError, match="exact completed-bundle container"):
        smoke._validate_smoke_checkpoint_metadata(
            {"flags": dict(flags)},
            {"flags": dict(flags)},
            Namespace(**flags),
            object(),
            active_state,
            None,
            None,
            None,
            None,
            None,
            completed,
        )


@pytest.mark.parametrize(
    ("source", "field", "bad"),
    [
        ("actor", "voc_gate_policy_schema_version", 11.0),
        ("actor", "voc_gate_policy_schema_version", np.int64(11)),
        ("actor", "voc_gate_policy_schema_version", True),
        ("actor", "voc_model_input_seal_schema_version", 1.0),
        ("actor", "voc_model_input_seal_schema_version", np.int64(1)),
        ("model", "voc_gate_policy_schema_version", 11.0),
        ("model", "voc_gate_policy_schema_version", np.int64(11)),
        ("model", "voc_model_input_seal_schema_version", 1.0),
    ],
)
def test_smoke_checkpoint_metadata_rejects_schema11_schema_type_coercion(
    monkeypatch, source, field, bad
):
    _patch_public_checkpoint_validators(monkeypatch)
    active_state = _schema11_active_state_for_smoke()
    model_state = _schema11_active_state_for_smoke()
    if source == "actor":
        active_state[field] = bad
    else:
        model_state[field] = bad
    monkeypatch.setattr(
        evaluation,
        "_validate_voc_gate_policy_schema",
        lambda checkpoint, embedded, label: dict(model_state),
    )
    completed = _schema11_completed_smoke_record()
    flags = {
        "dynamic_voc_mode": "control",
        "voc_model_input_seal_schema_version": 1,
    }

    with pytest.raises(
        ValueError,
        match="exact Python integer|final actor|metadata disagree",
    ):
        smoke._validate_smoke_checkpoint_metadata(
            {"flags": dict(flags)},
            {"flags": dict(flags)},
            Namespace(**flags),
            object(),
            active_state,
            None,
            None,
            None,
            None,
            None,
            completed,
        )


def test_invalid_schema8_smoke_prevalidation_blocks_every_downstream_call(
    monkeypatch, tmp_path
):
    import thinker.bc_loader as bc_loader
    import thinker.gym_add.wrapper as wrapper

    events = []
    config_payload = b"voc_gate_policy_schema_version: 8\n"
    (tmp_path / "config_c.yaml").write_bytes(config_payload)
    config_digest = hashlib.sha256(config_payload).hexdigest()
    marker = {
        "checkpoint_files": {
            "config_c.yaml": {
                "sha256": config_digest,
                "size": len(config_payload),
            }
        }
    }

    def forbidden(name):
        def _forbidden(*args, **kwargs):
            events.append(name)
            raise AssertionError(f"{name} ran before schema-8 validation")

        return _forbidden

    def reject(*args, **kwargs):
        events.append("schema8_prevalidation")
        raise ValueError("invalid schema-8 bundle")

    monkeypatch.setattr(
        evaluation, "dispatch_schema8_completed_bundle", reject
    )
    monkeypatch.setattr(
        evaluation,
        "checkpoint_hashes",
        lambda path: {"config_c.yaml": config_digest},
    )
    monkeypatch.setattr(
        evaluation, "validate_completion_marker", lambda path: marker
    )
    monkeypatch.setattr(smoke, "_load_flags", forbidden("load_flags"))
    monkeypatch.setattr(
        evaluation,
        "_load_flags_from_validated_config_bytes",
        forbidden("byte_bound_load_flags"),
    )
    monkeypatch.setattr(wrapper, "create_envpool", forbidden("environment"))
    monkeypatch.setattr(
        bc_loader, "FrameStackedBehavioralDataLoader", forbidden("data")
    )
    monkeypatch.setattr(torch, "load", forbidden("downstream_torch_load"))

    with pytest.raises(ValueError, match="invalid schema-8"):
        smoke.run_smoke(
            Namespace(
                seed=1,
                device=torch.device("cpu"),
                checkpoint_dir=tmp_path,
            )
        )

    assert events == ["schema8_prevalidation"]


def test_invalid_schema9_smoke_prevalidation_blocks_every_downstream_call(
    monkeypatch, tmp_path
):
    import thinker.bc_loader as bc_loader
    import thinker.gym_add.wrapper as wrapper

    events = []
    config_payload = b"voc_gate_policy_schema_version: 9\n"
    (tmp_path / "config_c.yaml").write_bytes(config_payload)
    config_digest = hashlib.sha256(config_payload).hexdigest()
    marker = {
        "checkpoint_files": {
            "config_c.yaml": {
                "sha256": config_digest,
                "size": len(config_payload),
            }
        }
    }

    def forbidden(name):
        def _forbidden(*args, **kwargs):
            events.append(name)
            raise AssertionError(f"{name} ran before schema-9 validation")

        return _forbidden

    def reject(*args, **kwargs):
        events.append("schema9_prevalidation")
        raise ValueError("invalid schema-9 bundle")

    monkeypatch.setattr(
        evaluation, "dispatch_schema8_completed_bundle", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        evaluation, "dispatch_schema9_completed_bundle", reject
    )
    monkeypatch.setattr(
        evaluation,
        "checkpoint_hashes",
        lambda path: {"config_c.yaml": config_digest},
    )
    monkeypatch.setattr(
        evaluation, "validate_completion_marker", lambda path: marker
    )
    monkeypatch.setattr(smoke, "_load_flags", forbidden("load_flags"))
    monkeypatch.setattr(wrapper, "create_envpool", forbidden("environment"))
    monkeypatch.setattr(
        bc_loader, "FrameStackedBehavioralDataLoader", forbidden("data")
    )
    monkeypatch.setattr(torch, "load", forbidden("downstream_torch_load"))

    with pytest.raises(ValueError, match="invalid schema-9"):
        smoke.run_smoke(
            Namespace(
                seed=1,
                device=torch.device("cpu"),
                checkpoint_dir=tmp_path,
            )
        )

    assert events == ["schema9_prevalidation"]


def test_invalid_schema10_smoke_prevalidation_blocks_every_downstream_call(
    monkeypatch, tmp_path
):
    import thinker.bc_loader as bc_loader
    import thinker.gym_add.wrapper as wrapper

    events = []
    config_payload = (
        "voc_gate_policy_schema_version: 10\n"
        "xpid: enduro-voc-v17-huber-common-eps25-sentinel-wire1200\n"
    ).encode("utf-8")
    (tmp_path / "config_c.yaml").write_bytes(config_payload)
    config_digest = hashlib.sha256(config_payload).hexdigest()
    marker = {
        "checkpoint_files": {
            "config_c.yaml": {
                "sha256": config_digest,
                "size": len(config_payload),
            }
        }
    }

    def forbidden(name):
        def _forbidden(*args, **kwargs):
            events.append(name)
            raise AssertionError(f"{name} ran before schema-10 validation")

        return _forbidden

    def reject(*args, **kwargs):
        events.append("schema10_prevalidation")
        raise ValueError("invalid schema-10 bundle")

    monkeypatch.setattr(
        evaluation, "dispatch_schema8_completed_bundle", lambda *a, **k: None
    )
    monkeypatch.setattr(
        evaluation, "dispatch_schema9_completed_bundle", lambda *a, **k: None
    )
    monkeypatch.setattr(
        evaluation, "dispatch_schema10_completed_bundle", reject
    )
    monkeypatch.setattr(
        evaluation,
        "checkpoint_hashes",
        lambda path: {"config_c.yaml": config_digest},
    )
    monkeypatch.setattr(
        evaluation, "validate_completion_marker", lambda path: marker
    )
    monkeypatch.setattr(smoke, "_load_flags", forbidden("load_flags"))
    monkeypatch.setattr(wrapper, "create_envpool", forbidden("environment"))
    monkeypatch.setattr(
        bc_loader, "FrameStackedBehavioralDataLoader", forbidden("data")
    )
    monkeypatch.setattr(torch, "load", forbidden("downstream_torch_load"))

    with pytest.raises(ValueError, match="invalid schema-10"):
        smoke.run_smoke(
            Namespace(
                seed=1,
                device=torch.device("cpu"),
                checkpoint_dir=tmp_path,
            )
        )

    assert events == ["schema10_prevalidation"]


def test_invalid_schema11_smoke_prevalidation_blocks_every_downstream_call(
    monkeypatch, tmp_path
):
    import thinker.bc_loader as bc_loader
    import thinker.gym_add.wrapper as wrapper

    events = []
    config_payload = (
        "voc_gate_policy_schema_version: 11\n"
        "xpid: enduro-voc-v18-orthocd-adam-eps25-sentinel-wire1200\n"
    ).encode("utf-8")
    (tmp_path / "config_c.yaml").write_bytes(config_payload)
    config_digest = hashlib.sha256(config_payload).hexdigest()
    marker = {
        "checkpoint_files": {
            "config_c.yaml": {
                "sha256": config_digest,
                "size": len(config_payload),
            }
        }
    }

    def forbidden(name):
        def _forbidden(*args, **kwargs):
            events.append(name)
            raise AssertionError(f"{name} ran before schema-11 validation")

        return _forbidden

    def reject(*args, **kwargs):
        events.append("schema11_prevalidation")
        raise ValueError("invalid schema-11 bundle")

    monkeypatch.setattr(
        evaluation, "dispatch_schema8_completed_bundle", lambda *a, **k: None
    )
    monkeypatch.setattr(
        evaluation, "dispatch_schema9_completed_bundle", lambda *a, **k: None
    )
    monkeypatch.setattr(
        evaluation, "dispatch_schema10_completed_bundle", lambda *a, **k: None
    )
    monkeypatch.setattr(
        evaluation, "dispatch_schema11_completed_bundle", reject
    )
    monkeypatch.setattr(
        evaluation,
        "checkpoint_hashes",
        lambda path: {"config_c.yaml": config_digest},
    )
    monkeypatch.setattr(
        evaluation, "validate_completion_marker", lambda path: marker
    )
    monkeypatch.setattr(smoke, "_load_flags", forbidden("load_flags"))
    monkeypatch.setattr(wrapper, "create_envpool", forbidden("environment"))
    monkeypatch.setattr(
        bc_loader, "FrameStackedBehavioralDataLoader", forbidden("data")
    )
    monkeypatch.setattr(torch, "load", forbidden("downstream_torch_load"))

    with pytest.raises(ValueError, match="invalid schema-11"):
        smoke.run_smoke(
            Namespace(
                seed=1,
                device=torch.device("cpu"),
                checkpoint_dir=tmp_path,
            )
        )

    assert events == ["schema11_prevalidation"]


@pytest.mark.parametrize(
    "schema_line",
    [
        pytest.param(b"voc_gate_policy_schema_version: 5\n", id="wrong-schema"),
        pytest.param(b"", id="missing-schema"),
    ],
)
def test_malformed_v18_prefix_routes_to_schema11_before_smoke_downstream(
    monkeypatch, tmp_path, schema_line
):
    import thinker.bc_loader as bc_loader
    import thinker.gym_add.wrapper as wrapper

    events = []
    config_payload = schema_line + (
        b"xpid: enduro-voc-v18-orthocd-adam-eps25-malformed-stage\n"
    )
    (tmp_path / "config_c.yaml").write_bytes(config_payload)
    config_digest = hashlib.sha256(config_payload).hexdigest()
    marker = {
        "checkpoint_files": {
            "config_c.yaml": {
                "sha256": config_digest,
                "size": len(config_payload),
            }
        }
    }

    def forbidden(name):
        def _forbidden(*args, **kwargs):
            events.append(name)
            raise AssertionError(f"{name} ran before schema-11 validation")

        return _forbidden

    monkeypatch.setattr(
        evaluation, "dispatch_schema8_completed_bundle", lambda *a, **k: None
    )
    monkeypatch.setattr(
        evaluation, "dispatch_schema9_completed_bundle", lambda *a, **k: None
    )
    monkeypatch.setattr(
        evaluation, "dispatch_schema10_completed_bundle", lambda *a, **k: None
    )
    monkeypatch.setattr(
        evaluation,
        "checkpoint_hashes",
        lambda path: {"config_c.yaml": config_digest},
    )
    monkeypatch.setattr(
        evaluation, "validate_completion_marker", lambda path: marker
    )
    monkeypatch.setattr(smoke, "_load_flags", forbidden("load_flags"))
    monkeypatch.setattr(wrapper, "create_envpool", forbidden("environment"))
    monkeypatch.setattr(
        bc_loader, "FrameStackedBehavioralDataLoader", forbidden("data")
    )
    monkeypatch.setattr(torch, "load", forbidden("downstream_torch_load"))

    with pytest.raises(
        ValueError,
        match="dedicated schema-11 validation requires exact Python integer",
    ):
        smoke.run_smoke(
            Namespace(
                seed=1,
                device=torch.device("cpu"),
                checkpoint_dir=tmp_path,
            )
        )

    assert events == []


@pytest.mark.parametrize(
    "schema_line",
    [
        pytest.param(b"voc_gate_policy_schema_version: 5\n", id="wrong-schema"),
        pytest.param(b"", id="missing-schema"),
    ],
)
def test_malformed_v19_prefix_routes_schema12_before_smoke_downstream(
    monkeypatch, tmp_path, schema_line
):
    import thinker.bc_loader as bc_loader
    import thinker.gym_add.wrapper as wrapper

    events = []
    config_payload = schema_line + (
        b"xpid: enduro-voc-v19-tau1-orthocd-adam-eps25-malformed-stage\n"
    )
    (tmp_path / "config_c.yaml").write_bytes(config_payload)
    config_digest = hashlib.sha256(config_payload).hexdigest()
    marker = {
        "checkpoint_files": {
            "config_c.yaml": {
                "sha256": config_digest,
                "size": len(config_payload),
            }
        }
    }

    def forbidden(name):
        def _forbidden(*args, **kwargs):
            events.append(name)
            raise AssertionError(f"{name} ran before schema-12 validation")

        return _forbidden

    def reject(*args, **kwargs):
        events.append("schema12_prevalidation")
        raise ValueError("invalid schema-12 intent")

    for name in (
        "dispatch_schema8_completed_bundle",
        "dispatch_schema9_completed_bundle",
        "dispatch_schema10_completed_bundle",
        "dispatch_schema11_completed_bundle",
    ):
        monkeypatch.setattr(evaluation, name, lambda *a, **k: None)
    monkeypatch.setattr(
        evaluation, "dispatch_schema12_completed_bundle", reject
    )
    monkeypatch.setattr(
        evaluation,
        "checkpoint_hashes",
        lambda path: {"config_c.yaml": config_digest},
    )
    monkeypatch.setattr(
        evaluation, "validate_completion_marker", lambda path: marker
    )
    monkeypatch.setattr(smoke, "_load_flags", forbidden("load_flags"))
    monkeypatch.setattr(wrapper, "create_envpool", forbidden("environment"))
    monkeypatch.setattr(
        bc_loader, "FrameStackedBehavioralDataLoader", forbidden("data")
    )
    monkeypatch.setattr(torch, "load", forbidden("downstream_torch_load"))

    with pytest.raises(ValueError, match="invalid schema-12 intent"):
        smoke.run_smoke(
            Namespace(
                seed=1,
                device=torch.device("cpu"),
                checkpoint_dir=tmp_path,
            )
        )

    assert events == ["schema12_prevalidation"]


@pytest.mark.parametrize(
    "schema_line",
    [
        pytest.param(b"voc_gate_policy_schema_version: 5\n", id="wrong-schema"),
        pytest.param(b"", id="missing-schema"),
    ],
)
def test_malformed_v20_prefix_routes_schema13_before_smoke_downstream(
    monkeypatch, tmp_path, schema_line
):
    events = []
    config_payload = schema_line + (
        b"xpid: enduro-voc-v20-telemetry-tau1-orthocd-adam-eps25-malformed\n"
    )
    (tmp_path / "config_c.yaml").write_bytes(config_payload)
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

    def forbidden(name):
        def _forbidden(*args, **kwargs):
            events.append(name)
            raise AssertionError(f"{name} ran before schema-13 validation")

        return _forbidden

    def reject(*args, **kwargs):
        events.append("schema13_prevalidation")
        raise ValueError("invalid schema-13 intent")

    real_import = builtins.__import__
    downstream_modules = {
        "thinker.actor_net",
        "thinker.bc_loader",
        "thinker.cenv",
        "thinker.dataset_env",
        "thinker.dynamic_imitation",
        "thinker.gym_add.wrapper",
        "thinker.main",
        "thinker.model_net",
    }

    def guarded_import(name, *args, **kwargs):
        fromlist = args[2] if len(args) >= 3 else kwargs.get("fromlist", ())
        if name in downstream_modules or (
            name == "thinker" and "util" in (fromlist or ())
        ):
            return forbidden(f"downstream_import:{name}")()
        return real_import(name, *args, **kwargs)

    for name in (
        "dispatch_schema8_completed_bundle",
        "dispatch_schema9_completed_bundle",
        "dispatch_schema10_completed_bundle",
        "dispatch_schema11_completed_bundle",
        "dispatch_schema12_completed_bundle",
    ):
        monkeypatch.setattr(evaluation, name, lambda *a, **k: None)
    monkeypatch.setattr(
        evaluation, "dispatch_schema13_completed_bundle", reject
    )
    monkeypatch.setattr(
        evaluation, "validate_schema13_completion_marker", lambda path: marker
    )
    monkeypatch.setattr(
        evaluation,
        "_schema13_checkpoint_hashes",
        lambda path, **kwargs: {"config_c.yaml": config_digest},
    )
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(smoke, "_load_flags", forbidden("load_flags"))
    monkeypatch.setattr(smoke.random, "seed", forbidden("python_rng"))
    monkeypatch.setattr(smoke.np.random, "seed", forbidden("numpy_rng"))
    monkeypatch.setattr(torch, "manual_seed", forbidden("torch_rng"))
    monkeypatch.setattr(torch, "load", forbidden("downstream_torch_load"))

    with pytest.raises(ValueError, match="invalid schema-13 intent"):
        smoke.run_smoke(
            Namespace(
                seed=1,
                device=torch.device("cpu"),
                checkpoint_dir=tmp_path,
            )
        )

    assert events == ["schema13_prevalidation"]


@pytest.mark.parametrize(
    "raw_xpid",
    [
        pytest.param(
            "enduro-voc-v20-telemetry-tau1-orthocd-adam-eps25-malformed",
            id="plain",
        ),
        pytest.param(
            b"enduro-voc-v20-telemetry-tau1-orthocd-adam-eps25-malformed",
            id="bytes",
        ),
    ],
)
@pytest.mark.parametrize("raw_schema", [pytest.param(5, id="wrong"), pytest.param(None, id="missing")])
def test_smoke_local_v20_intent_fails_closed_without_public_schema13_features(
    monkeypatch, tmp_path, raw_xpid, raw_schema
):
    events = []
    config = {"xpid": raw_xpid}
    if raw_schema is not None:
        config["voc_gate_policy_schema_version"] = raw_schema
    (tmp_path / "config_c.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True), encoding="utf-8"
    )

    def forbidden(name):
        def fail(*args, **kwargs):
            events.append(name)
            raise AssertionError(f"{name} ran after malformed V20 intent")

        return fail

    real_import = builtins.__import__
    downstream_modules = {
        "thinker.actor_net",
        "thinker.bc_loader",
        "thinker.cenv",
        "thinker.dataset_env",
        "thinker.dynamic_imitation",
        "thinker.gym_add.wrapper",
        "thinker.main",
        "thinker.model_net",
    }

    def guarded_import(name, *args, **kwargs):
        fromlist = args[2] if len(args) >= 3 else kwargs.get("fromlist", ())
        if name in downstream_modules or (
            name == "thinker" and "util" in (fromlist or ())
        ):
            return forbidden(f"downstream_import:{name}")()
        return real_import(name, *args, **kwargs)

    monkeypatch.delattr(
        evaluation, "_schema13_xpid_claims_intent", raising=True
    )
    monkeypatch.delattr(
        evaluation, "dispatch_schema13_completed_bundle", raising=True
    )
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(smoke, "_load_flags", forbidden("load_flags"))
    monkeypatch.setattr(smoke.random, "seed", forbidden("python_rng"))
    monkeypatch.setattr(smoke.np.random, "seed", forbidden("numpy_rng"))
    monkeypatch.setattr(torch, "manual_seed", forbidden("torch_rng"))
    monkeypatch.setattr(torch, "load", forbidden("checkpoint_load"))

    with pytest.raises(RuntimeError, match="lacks schema-13"):
        smoke.run_smoke(
            Namespace(
                seed=1,
                device=torch.device("cpu"),
                checkpoint_dir=tmp_path,
            )
        )

    assert events == []


def test_schema12_smoke_missing_public_feature_gate_fails_before_downstream(
    monkeypatch, tmp_path
):
    import thinker.gym_add.wrapper as wrapper

    payload = (
        "voc_gate_policy_schema_version: 12\n"
        "voc_gate_target_tau: 1.0\n"
        "xpid: enduro-voc-v19-tau1-orthocd-adam-eps25-sentinel-wire1200\n"
    ).encode("utf-8")
    (tmp_path / "config_c.yaml").write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    marker = {
        "checkpoint_files": {
            "config_c.yaml": {"sha256": digest, "size": len(payload)}
        }
    }
    downstream = []
    monkeypatch.delattr(
        evaluation, "dispatch_schema12_completed_bundle", raising=True
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
    monkeypatch.setattr(
        evaluation, "dispatch_schema11_completed_bundle", lambda *a, **k: None
    )
    monkeypatch.setattr(
        evaluation, "checkpoint_hashes", lambda path: {"config_c.yaml": digest}
    )
    monkeypatch.setattr(
        evaluation, "validate_completion_marker", lambda path: marker
    )
    monkeypatch.setattr(
        smoke, "_load_flags", lambda *a, **k: downstream.append("load_flags")
    )
    monkeypatch.setattr(
        wrapper,
        "create_envpool",
        lambda *a, **k: downstream.append("environment"),
    )

    with pytest.raises(RuntimeError, match="lacks schema-12 dispatch"):
        smoke.run_smoke(
            Namespace(
                seed=1,
                device=torch.device("cpu"),
                checkpoint_dir=tmp_path,
            )
        )

    assert downstream == []


def test_schema10_smoke_missing_public_feature_gate_fails_closed(
    monkeypatch, tmp_path
):
    import thinker.gym_add.wrapper as wrapper

    payload = (
        "voc_gate_policy_schema_version: 10\n"
        "xpid: enduro-voc-v17-huber-common-eps25-sentinel-wire1200\n"
    ).encode("utf-8")
    (tmp_path / "config_c.yaml").write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    marker = {
        "checkpoint_files": {
            "config_c.yaml": {"sha256": digest, "size": len(payload)}
        }
    }
    downstream = []

    monkeypatch.delattr(
        evaluation, "dispatch_schema10_completed_bundle", raising=True
    )
    monkeypatch.setattr(
        evaluation, "dispatch_schema8_completed_bundle", lambda *a, **k: None
    )
    monkeypatch.setattr(
        evaluation, "dispatch_schema9_completed_bundle", lambda *a, **k: None
    )
    monkeypatch.setattr(
        evaluation, "checkpoint_hashes", lambda path: {"config_c.yaml": digest}
    )
    monkeypatch.setattr(
        evaluation, "validate_completion_marker", lambda path: marker
    )
    monkeypatch.setattr(
        smoke,
        "_load_flags",
        lambda *a, **k: downstream.append("load_flags"),
    )
    monkeypatch.setattr(
        wrapper,
        "create_envpool",
        lambda *a, **k: downstream.append("environment"),
    )

    with pytest.raises(RuntimeError, match="lacks schema-10 dispatch"):
        smoke.run_smoke(
            Namespace(
                seed=1,
                device=torch.device("cpu"),
                checkpoint_dir=tmp_path,
            )
        )

    assert downstream == []


def test_schema11_smoke_missing_public_feature_gate_fails_closed(
    monkeypatch, tmp_path
):
    import thinker.gym_add.wrapper as wrapper

    payload = (
        "voc_gate_policy_schema_version: 11\n"
        "xpid: enduro-voc-v18-orthocd-adam-eps25-sentinel-wire1200\n"
    ).encode("utf-8")
    (tmp_path / "config_c.yaml").write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    marker = {
        "checkpoint_files": {
            "config_c.yaml": {"sha256": digest, "size": len(payload)}
        }
    }
    downstream = []

    monkeypatch.delattr(
        evaluation, "dispatch_schema11_completed_bundle", raising=True
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
    monkeypatch.setattr(
        evaluation, "checkpoint_hashes", lambda path: {"config_c.yaml": digest}
    )
    monkeypatch.setattr(
        evaluation, "validate_completion_marker", lambda path: marker
    )
    monkeypatch.setattr(
        smoke,
        "_load_flags",
        lambda *a, **k: downstream.append("load_flags"),
    )
    monkeypatch.setattr(
        wrapper,
        "create_envpool",
        lambda *a, **k: downstream.append("environment"),
    )

    with pytest.raises(RuntimeError, match="lacks schema-11 dispatch"):
        smoke.run_smoke(
            Namespace(
                seed=1,
                device=torch.device("cpu"),
                checkpoint_dir=tmp_path,
            )
        )

    assert downstream == []


def test_legacy_smoke_rejects_explicit_v15_config_before_downstream(
    monkeypatch, tmp_path
):
    import thinker.bc_loader as bc_loader
    import thinker.gym_add.wrapper as wrapper

    checkpoint_dir = tmp_path / "legacy"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "config_c.yaml").write_text(
        "voc_gate_policy_schema_version: 5\nxpid: legacy\n",
        encoding="utf-8",
    )
    explicit_config = tmp_path / "v15.yaml"
    explicit_config.write_text(
        "voc_gate_policy_schema_version: 8\n"
        f"xpid: {evaluation.VOC_GATE_POLICY_SCHEMA8_STAGE_PROFILES[0][0]}\n",
        encoding="utf-8",
    )
    downstream = []

    def forbidden(name):
        def _forbidden(*args, **kwargs):
            downstream.append(name)
            raise AssertionError(f"{name} ran for cross-source schema-8 claim")

        return _forbidden

    monkeypatch.setattr(smoke, "_load_flags", forbidden("load_flags"))
    monkeypatch.setattr(wrapper, "create_envpool", forbidden("environment"))
    monkeypatch.setattr(
        bc_loader, "FrameStackedBehavioralDataLoader", forbidden("data")
    )
    monkeypatch.setattr(torch, "load", forbidden("downstream_torch_load"))

    with pytest.raises(ValueError, match="requires exact Python integer"):
        smoke.run_smoke(
            Namespace(
                seed=1,
                device=torch.device("cpu"),
                checkpoint_dir=checkpoint_dir,
                config=explicit_config,
            )
        )

    assert downstream == []


def test_v15_smoke_rejects_explicit_legacy_config_before_downstream(
    monkeypatch, tmp_path
):
    import thinker.bc_loader as bc_loader
    import thinker.gym_add.wrapper as wrapper

    checkpoint_dir = tmp_path / "v15"
    checkpoint_dir.mkdir()
    checkpoint_payload = (
        "voc_gate_policy_schema_version: 8\n"
        f"xpid: {evaluation.VOC_GATE_POLICY_SCHEMA8_STAGE_PROFILES[0][0]}\n"
    ).encode("utf-8")
    (checkpoint_dir / "config_c.yaml").write_bytes(checkpoint_payload)
    checkpoint_digest = hashlib.sha256(checkpoint_payload).hexdigest()
    explicit_config = tmp_path / "legacy.yaml"
    explicit_config.write_text(
        "voc_gate_policy_schema_version: 7\nxpid: legacy\n",
        encoding="utf-8",
    )
    marker = {
        "checkpoint_files": {
            "config_c.yaml": {
                "sha256": checkpoint_digest,
                "size": len(checkpoint_payload),
            }
        }
    }
    completed = _schema8_completed_smoke_record()
    completed["completion_evidence"] = copy.deepcopy(marker)
    downstream = []

    def dispatch(*args, config_payload, **kwargs):
        if b"schema_version: 8" in config_payload:
            return copy.deepcopy(completed)
        return None

    def forbidden(name):
        def _forbidden(*args, **kwargs):
            downstream.append(name)
            raise AssertionError(f"{name} ran for alternate legacy config")

        return _forbidden

    monkeypatch.setattr(
        evaluation, "dispatch_schema8_completed_bundle", dispatch
    )
    monkeypatch.setattr(
        evaluation,
        "checkpoint_hashes",
        lambda path: {"config_c.yaml": checkpoint_digest},
    )
    monkeypatch.setattr(
        evaluation, "validate_completion_marker", lambda path: marker
    )
    monkeypatch.setattr(smoke, "_load_flags", forbidden("load_flags"))
    monkeypatch.setattr(wrapper, "create_envpool", forbidden("environment"))
    monkeypatch.setattr(
        bc_loader, "FrameStackedBehavioralDataLoader", forbidden("data")
    )
    monkeypatch.setattr(torch, "load", forbidden("downstream_torch_load"))

    with pytest.raises(ValueError, match="authoritative checkpoint config"):
        smoke.run_smoke(
            Namespace(
                seed=1,
                device=torch.device("cpu"),
                checkpoint_dir=checkpoint_dir,
                config=explicit_config,
            )
        )

    assert downstream == []


@pytest.mark.parametrize("direction", ["schema10_to_legacy", "legacy_to_schema10"])
def test_schema10_smoke_rejects_reciprocal_explicit_config_before_downstream(
    monkeypatch, tmp_path, direction
):
    import thinker.bc_loader as bc_loader
    import thinker.gym_add.wrapper as wrapper

    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    schema10_payload = (
        "voc_gate_policy_schema_version: 10\n"
        "xpid: enduro-voc-v17-huber-common-eps25-sentinel-wire1200\n"
    ).encode("utf-8")
    legacy_payload = (
        b"voc_gate_policy_schema_version: 5\nxpid: historical-v12\n"
    )
    checkpoint_payload, explicit_payload = (
        (schema10_payload, legacy_payload)
        if direction == "schema10_to_legacy"
        else (legacy_payload, schema10_payload)
    )
    (checkpoint_dir / "config_c.yaml").write_bytes(checkpoint_payload)
    explicit_config = tmp_path / "alternate.yaml"
    explicit_config.write_bytes(explicit_payload)
    checkpoint_digest = hashlib.sha256(checkpoint_payload).hexdigest()
    marker = {
        "checkpoint_files": {
            "config_c.yaml": {
                "sha256": checkpoint_digest,
                "size": len(checkpoint_payload),
            }
        }
    }
    completed = _schema10_completed_smoke_record()
    completed["completion_evidence"] = copy.deepcopy(marker)
    downstream = []

    def dispatch_schema10(*args, config_payload, **kwargs):
        if b"schema_version: 10" in config_payload:
            return copy.deepcopy(completed)
        return None

    def forbidden(name):
        def _forbidden(*args, **kwargs):
            downstream.append(name)
            raise AssertionError(f"{name} ran for reciprocal schema-10 config")

        return _forbidden

    monkeypatch.setattr(
        evaluation, "dispatch_schema8_completed_bundle", lambda *a, **k: None
    )
    monkeypatch.setattr(
        evaluation, "dispatch_schema9_completed_bundle", lambda *a, **k: None
    )
    monkeypatch.setattr(
        evaluation, "dispatch_schema10_completed_bundle", dispatch_schema10
    )
    monkeypatch.setattr(
        evaluation,
        "checkpoint_hashes",
        lambda path: {"config_c.yaml": checkpoint_digest},
    )
    monkeypatch.setattr(
        evaluation, "validate_completion_marker", lambda path: marker
    )
    monkeypatch.setattr(smoke, "_load_flags", forbidden("load_flags"))
    monkeypatch.setattr(wrapper, "create_envpool", forbidden("environment"))
    monkeypatch.setattr(
        bc_loader, "FrameStackedBehavioralDataLoader", forbidden("data")
    )
    monkeypatch.setattr(torch, "load", forbidden("tensor_load"))

    with pytest.raises(ValueError, match="authoritative checkpoint config"):
        smoke.run_smoke(
            Namespace(
                seed=1,
                device=torch.device("cpu"),
                checkpoint_dir=checkpoint_dir,
                config=explicit_config,
            )
        )

    assert downstream == []


@pytest.mark.parametrize("direction", ["schema11_to_schema10", "schema10_to_schema11"])
def test_schema11_smoke_rejects_reciprocal_schema10_explicit_config(
    monkeypatch, tmp_path, direction
):
    import thinker.bc_loader as bc_loader
    import thinker.gym_add.wrapper as wrapper

    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    schema11_payload = (
        "voc_gate_policy_schema_version: 11\n"
        "xpid: enduro-voc-v18-orthocd-adam-eps25-sentinel-wire1200\n"
    ).encode("utf-8")
    schema10_payload = (
        "voc_gate_policy_schema_version: 10\n"
        "xpid: enduro-voc-v17-huber-common-eps25-sentinel-wire1200\n"
    ).encode("utf-8")
    checkpoint_payload, explicit_payload = (
        (schema11_payload, schema10_payload)
        if direction == "schema11_to_schema10"
        else (schema10_payload, schema11_payload)
    )
    (checkpoint_dir / "config_c.yaml").write_bytes(checkpoint_payload)
    explicit_config = tmp_path / "alternate.yaml"
    explicit_config.write_bytes(explicit_payload)
    checkpoint_digest = hashlib.sha256(checkpoint_payload).hexdigest()
    marker = {
        "checkpoint_files": {
            "config_c.yaml": {
                "sha256": checkpoint_digest,
                "size": len(checkpoint_payload),
            }
        }
    }
    schema10_completed = _schema10_completed_smoke_record()
    schema11_completed = _schema11_completed_smoke_record()
    for completed in (schema10_completed, schema11_completed):
        completed["completion_evidence"] = copy.deepcopy(marker)
    downstream = []

    def dispatch_schema10(*args, config_payload, **kwargs):
        if b"schema_version: 10" in config_payload:
            return copy.deepcopy(schema10_completed)
        return None

    def dispatch_schema11(*args, config_payload, **kwargs):
        if b"schema_version: 11" in config_payload:
            return copy.deepcopy(schema11_completed)
        return None

    def forbidden(name):
        def _forbidden(*args, **kwargs):
            downstream.append(name)
            raise AssertionError(f"{name} ran for reciprocal schema config")

        return _forbidden

    monkeypatch.setattr(
        evaluation, "dispatch_schema8_completed_bundle", lambda *a, **k: None
    )
    monkeypatch.setattr(
        evaluation, "dispatch_schema9_completed_bundle", lambda *a, **k: None
    )
    monkeypatch.setattr(
        evaluation, "dispatch_schema10_completed_bundle", dispatch_schema10
    )
    monkeypatch.setattr(
        evaluation, "dispatch_schema11_completed_bundle", dispatch_schema11
    )
    monkeypatch.setattr(
        evaluation,
        "checkpoint_hashes",
        lambda path: {"config_c.yaml": checkpoint_digest},
    )
    monkeypatch.setattr(
        evaluation, "validate_completion_marker", lambda path: marker
    )
    monkeypatch.setattr(smoke, "_load_flags", forbidden("load_flags"))
    monkeypatch.setattr(wrapper, "create_envpool", forbidden("environment"))
    monkeypatch.setattr(
        bc_loader, "FrameStackedBehavioralDataLoader", forbidden("data")
    )
    monkeypatch.setattr(torch, "load", forbidden("tensor_load"))

    with pytest.raises(ValueError, match="authoritative checkpoint config"):
        smoke.run_smoke(
            Namespace(
                seed=1,
                device=torch.device("cpu"),
                checkpoint_dir=checkpoint_dir,
                config=explicit_config,
            )
        )

    assert downstream == []


def test_smoke_uses_bound_config_bytes_and_detects_post_dispatch_path_swap(
    monkeypatch, tmp_path
):
    import thinker.bc_loader as bc_loader
    import thinker.gym_add.wrapper as wrapper

    checkpoint_dir = tmp_path / "legacy"
    checkpoint_dir.mkdir()
    config_path = checkpoint_dir / "config_c.yaml"
    config_payload = b"voc_gate_policy_schema_version: 5\nxpid: legacy\n"
    config_path.write_bytes(config_payload)
    config_digest = hashlib.sha256(config_payload).hexdigest()
    events = []

    def dispatch(*args, config_payload, **kwargs):
        events.append("dispatch")
        assert config_payload == b"voc_gate_policy_schema_version: 5\nxpid: legacy\n"
        config_path.write_text(
            "voc_gate_policy_schema_version: 8\n",
            encoding="utf-8",
        )
        return None

    def load_flags(args, *, config_payload, expected_sha256):
        events.append("load_flags")
        assert config_payload == b"voc_gate_policy_schema_version: 5\nxpid: legacy\n"
        assert expected_sha256 == config_digest
        return Namespace(name="Enduro-v5", max_search_steps=20, batch_length=4)

    def private_copy(flags):
        events.append("private_copy")
        return flags, None

    def forbidden(name):
        def _forbidden(*args, **kwargs):
            events.append(name)
            raise AssertionError(f"{name} ran after checkpoint config swap")

        return _forbidden

    monkeypatch.setattr(
        evaluation, "dispatch_schema8_completed_bundle", dispatch
    )
    monkeypatch.setattr(
        evaluation, "validate_schema6_completed_bundle", lambda path: None
    )
    monkeypatch.setattr(
        evaluation, "validate_schema7_completed_bundle", lambda path: None
    )
    monkeypatch.setattr(evaluation, "evaluation_runtime_flags", private_copy)
    monkeypatch.setattr(smoke, "_load_flags", load_flags)
    monkeypatch.setattr(wrapper, "create_envpool", forbidden("environment"))
    monkeypatch.setattr(
        bc_loader, "FrameStackedBehavioralDataLoader", forbidden("data")
    )
    monkeypatch.setattr(torch, "load", forbidden("downstream_torch_load"))

    with pytest.raises(RuntimeError, match="config changed before smoke"):
        smoke.run_smoke(
            Namespace(
                seed=1,
                device=torch.device("cpu"),
                checkpoint_dir=checkpoint_dir,
                config=None,
                env_name="Enduro-v5",
                scored_length=4,
            )
        )

    assert events == ["dispatch", "load_flags", "private_copy"]


def test_schema8_smoke_pre_copy_post_copy_order_precedes_environment(
    monkeypatch, tmp_path
):
    import thinker.gym_add.wrapper as wrapper

    events = []
    completed = _schema8_completed_smoke_record()
    config_payload = b"voc_gate_policy_schema_version: 8\n"
    (tmp_path / "config_c.yaml").write_bytes(config_payload)
    config_digest = hashlib.sha256(config_payload).hexdigest()
    hashes = {"config_c.yaml": config_digest}
    marker = {
        "checkpoint_files": {
            "config_c.yaml": {
                "sha256": config_digest,
                "size": len(config_payload),
            }
        }
    }
    completed["completion_evidence"] = copy.deepcopy(marker)
    training = Namespace(
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
        name="Enduro-v5",
        max_search_steps=20,
        batch_length=4,
    )

    def prevalidate(*args, **kwargs):
        events.append("prevalidate")
        return copy.deepcopy(completed)

    def load_flags(args, *, config_payload, expected_sha256):
        events.append("load_flags")
        assert config_payload == config_payload_bytes
        assert expected_sha256 == config_digest
        return training

    def private_copy(flags):
        events.append("private_copy")
        runtime = copy.deepcopy(flags)
        runtime.train_actor = False
        runtime.train_model = False
        runtime.parallel = False
        runtime.parallel_actor = False
        runtime.voc_actor_policy_barrier_runtime = False
        return runtime, {"evaluation_copy": {"train_model": False}}

    def postvalidate(*args, **kwargs):
        events.append("postvalidate")
        return copy.deepcopy(completed)

    def reach_environment(*args, **kwargs):
        events.append("environment")
        raise RuntimeError("environment sentinel")

    monkeypatch.setattr(
        evaluation, "dispatch_schema8_completed_bundle", prevalidate
    )
    monkeypatch.setattr(
        evaluation, "validate_schema8_completed_bundle", postvalidate
    )
    monkeypatch.setattr(evaluation, "checkpoint_hashes", lambda path: hashes)
    monkeypatch.setattr(
        evaluation, "validate_completion_marker", lambda path: marker
    )
    monkeypatch.setattr(
        evaluation, "validate_schema6_completed_bundle", lambda path: None
    )
    monkeypatch.setattr(
        evaluation, "validate_schema7_completed_bundle", lambda path: None
    )
    monkeypatch.setattr(evaluation, "evaluation_runtime_flags", private_copy)
    config_payload_bytes = config_payload
    monkeypatch.setattr(smoke, "_load_flags", load_flags)
    monkeypatch.setattr(wrapper, "create_envpool", reach_environment)

    with pytest.raises(RuntimeError, match="environment sentinel"):
        smoke.run_smoke(
            Namespace(
                seed=1,
                device=torch.device("cpu"),
                checkpoint_dir=tmp_path,
                config=None,
                scored_length=4,
                env_name="Enduro-v5",
                batch_size=1,
            )
        )

    assert events == [
        "prevalidate",
        "load_flags",
        "private_copy",
        "postvalidate",
        "environment",
    ]


def test_schema9_smoke_pre_copy_post_copy_order_precedes_environment(
    monkeypatch, tmp_path
):
    import thinker.gym_add.wrapper as wrapper

    events = []
    completed = _schema9_completed_smoke_record()
    config_payload = b"voc_gate_policy_schema_version: 9\n"
    (tmp_path / "config_c.yaml").write_bytes(config_payload)
    config_digest = hashlib.sha256(config_payload).hexdigest()
    hashes = {"config_c.yaml": config_digest}
    marker = {
        "checkpoint_files": {
            "config_c.yaml": {
                "sha256": config_digest,
                "size": len(config_payload),
            }
        }
    }
    completed["completion_evidence"] = copy.deepcopy(marker)
    training = Namespace(
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
        name="Enduro-v5",
        max_search_steps=20,
        batch_length=4,
    )

    def prevalidate(*args, **kwargs):
        events.append("prevalidate")
        return copy.deepcopy(completed)

    def load_flags(args, *, config_payload, expected_sha256):
        events.append("load_flags")
        assert config_payload == bound_payload
        assert expected_sha256 == config_digest
        return training

    def private_copy(flags):
        events.append("private_copy")
        runtime = copy.deepcopy(flags)
        runtime.train_actor = False
        runtime.train_model = False
        runtime.parallel = False
        runtime.parallel_actor = False
        runtime.voc_actor_policy_barrier_runtime = False
        return runtime, {"evaluation_copy": {"train_model": False}}

    def postvalidate(*args, **kwargs):
        events.append("postvalidate")
        return copy.deepcopy(completed)

    def reach_environment(*args, **kwargs):
        events.append("environment")
        raise RuntimeError("environment sentinel")

    monkeypatch.setattr(
        evaluation, "dispatch_schema8_completed_bundle", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        evaluation, "dispatch_schema9_completed_bundle", prevalidate
    )
    monkeypatch.setattr(
        evaluation, "validate_schema9_completed_bundle", postvalidate
    )
    monkeypatch.setattr(evaluation, "checkpoint_hashes", lambda path: hashes)
    monkeypatch.setattr(
        evaluation, "validate_completion_marker", lambda path: marker
    )
    monkeypatch.setattr(
        evaluation, "validate_schema6_completed_bundle", lambda path: None
    )
    monkeypatch.setattr(
        evaluation, "validate_schema7_completed_bundle", lambda path: None
    )
    monkeypatch.setattr(evaluation, "evaluation_runtime_flags", private_copy)
    bound_payload = config_payload
    monkeypatch.setattr(smoke, "_load_flags", load_flags)
    monkeypatch.setattr(wrapper, "create_envpool", reach_environment)

    with pytest.raises(RuntimeError, match="environment sentinel"):
        smoke.run_smoke(
            Namespace(
                seed=1,
                device=torch.device("cpu"),
                checkpoint_dir=tmp_path,
                config=None,
                scored_length=4,
                env_name="Enduro-v5",
                batch_size=1,
            )
        )

    assert events == [
        "prevalidate",
        "load_flags",
        "private_copy",
        "postvalidate",
        "environment",
    ]


def test_schema10_smoke_pre_copy_post_copy_order_precedes_environment(
    monkeypatch, tmp_path
):
    import thinker.gym_add.wrapper as wrapper

    events = []
    completed = _schema10_completed_smoke_record()
    config_payload = (
        "voc_gate_policy_schema_version: 10\n"
        "xpid: enduro-voc-v17-huber-common-eps25-sentinel-wire1200\n"
    ).encode("utf-8")
    (tmp_path / "config_c.yaml").write_bytes(config_payload)
    config_digest = hashlib.sha256(config_payload).hexdigest()
    hashes = {"config_c.yaml": config_digest}
    marker = {
        "checkpoint_files": {
            "config_c.yaml": {
                "sha256": config_digest,
                "size": len(config_payload),
            }
        }
    }
    completed["completion_evidence"] = copy.deepcopy(marker)
    training = Namespace(
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
        name="Enduro-v5",
        max_search_steps=20,
        batch_length=4,
    )

    def prevalidate(*args, **kwargs):
        events.append("prevalidate")
        return copy.deepcopy(completed)

    def load_flags(args, *, config_payload, expected_sha256):
        events.append("load_flags")
        assert config_payload == bound_payload
        assert expected_sha256 == config_digest
        return training

    def private_copy(flags):
        events.append("private_copy")
        runtime = copy.deepcopy(flags)
        runtime.train_actor = False
        runtime.train_model = False
        runtime.parallel = False
        runtime.parallel_actor = False
        runtime.voc_actor_policy_barrier_runtime = False
        return runtime, {"evaluation_copy": {"train_model": False}}

    def postvalidate(*args, **kwargs):
        events.append("postvalidate")
        return copy.deepcopy(completed)

    def reach_environment(*args, **kwargs):
        events.append("environment")
        raise RuntimeError("environment sentinel")

    monkeypatch.setattr(
        evaluation, "dispatch_schema8_completed_bundle", lambda *a, **k: None
    )
    monkeypatch.setattr(
        evaluation, "dispatch_schema9_completed_bundle", lambda *a, **k: None
    )
    monkeypatch.setattr(
        evaluation, "dispatch_schema10_completed_bundle", prevalidate
    )
    monkeypatch.setattr(
        evaluation, "validate_schema10_completed_bundle", postvalidate
    )
    monkeypatch.setattr(evaluation, "checkpoint_hashes", lambda path: hashes)
    monkeypatch.setattr(
        evaluation, "validate_completion_marker", lambda path: marker
    )
    monkeypatch.setattr(
        evaluation, "validate_schema6_completed_bundle", lambda path: None
    )
    monkeypatch.setattr(
        evaluation, "validate_schema7_completed_bundle", lambda path: None
    )
    monkeypatch.setattr(evaluation, "evaluation_runtime_flags", private_copy)
    bound_payload = config_payload
    monkeypatch.setattr(smoke, "_load_flags", load_flags)
    monkeypatch.setattr(wrapper, "create_envpool", reach_environment)

    with pytest.raises(RuntimeError, match="environment sentinel"):
        smoke.run_smoke(
            Namespace(
                seed=1,
                device=torch.device("cpu"),
                checkpoint_dir=tmp_path,
                config=None,
                scored_length=4,
                env_name="Enduro-v5",
                batch_size=1,
            )
        )

    assert events == [
        "prevalidate",
        "load_flags",
        "private_copy",
        "postvalidate",
        "environment",
    ]


@pytest.mark.parametrize("mutation", ["evidence", "hash"])
def test_schema8_smoke_post_copy_mutation_fails_before_environment(
    monkeypatch, tmp_path, mutation
):
    import thinker.gym_add.wrapper as wrapper

    completed = _schema8_completed_smoke_record()
    config_payload = b"voc_gate_policy_schema_version: 8\n"
    (tmp_path / "config_c.yaml").write_bytes(config_payload)
    config_digest = hashlib.sha256(config_payload).hexdigest()
    marker = {
        "checkpoint_files": {
            "config_c.yaml": {
                "sha256": config_digest,
                "size": len(config_payload),
            }
        }
    }
    completed["completion_evidence"] = copy.deepcopy(marker)
    post = copy.deepcopy(completed)
    if mutation == "evidence":
        post["resolved_identity"] = {"forged": True}
    hash_calls = 0
    environment_calls = 0
    training = Namespace(
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
        name="Enduro-v5",
        max_search_steps=20,
        batch_length=4,
    )

    monkeypatch.setattr(
        evaluation,
        "dispatch_schema8_completed_bundle",
        lambda *args, **kwargs: copy.deepcopy(completed),
    )
    monkeypatch.setattr(
        evaluation,
        "validate_schema8_completed_bundle",
        lambda *args, **kwargs: copy.deepcopy(post),
    )

    def hashes(path):
        nonlocal hash_calls
        hash_calls += 1
        digest = (
            "b" * 64
            if mutation == "hash" and hash_calls > 1
            else config_digest
        )
        return {"config_c.yaml": digest}

    def private_copy(flags):
        runtime = copy.deepcopy(flags)
        runtime.train_actor = False
        runtime.train_model = False
        runtime.parallel = False
        runtime.parallel_actor = False
        runtime.voc_actor_policy_barrier_runtime = False
        return runtime, {"evaluation_copy": {"train_model": False}}

    def environment(*args, **kwargs):
        nonlocal environment_calls
        environment_calls += 1
        raise AssertionError("environment ran after schema-8 mutation")

    monkeypatch.setattr(evaluation, "checkpoint_hashes", hashes)
    monkeypatch.setattr(
        evaluation, "validate_completion_marker", lambda path: marker
    )
    monkeypatch.setattr(
        evaluation, "validate_schema6_completed_bundle", lambda path: None
    )
    monkeypatch.setattr(
        evaluation, "validate_schema7_completed_bundle", lambda path: None
    )
    monkeypatch.setattr(evaluation, "evaluation_runtime_flags", private_copy)
    monkeypatch.setattr(
        smoke,
        "_load_flags",
        lambda args, *, config_payload, expected_sha256: training,
    )
    monkeypatch.setattr(wrapper, "create_envpool", environment)

    with pytest.raises(RuntimeError, match="changed across private"):
        smoke.run_smoke(
            Namespace(
                seed=1,
                device=torch.device("cpu"),
                checkpoint_dir=tmp_path,
                config=None,
                scored_length=4,
                env_name="Enduro-v5",
                batch_size=1,
            )
        )

    assert environment_calls == 0


@pytest.mark.parametrize("mutation", ["evidence", "hash"])
def test_schema10_smoke_post_copy_mutation_fails_before_environment(
    monkeypatch, tmp_path, mutation
):
    import thinker.gym_add.wrapper as wrapper

    completed = _schema10_completed_smoke_record()
    config_payload = (
        "voc_gate_policy_schema_version: 10\n"
        "xpid: enduro-voc-v17-huber-common-eps25-sentinel-wire1200\n"
    ).encode("utf-8")
    (tmp_path / "config_c.yaml").write_bytes(config_payload)
    config_digest = hashlib.sha256(config_payload).hexdigest()
    marker = {
        "checkpoint_files": {
            "config_c.yaml": {
                "sha256": config_digest,
                "size": len(config_payload),
            }
        }
    }
    completed["completion_evidence"] = copy.deepcopy(marker)
    post = copy.deepcopy(completed)
    if mutation == "evidence":
        post["stored_surface_identity"]["config"][
            "complete_surface_sha256"
        ] = "c" * 64
    hash_calls = 0
    environment_calls = 0
    training = Namespace(
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
        name="Enduro-v5",
        max_search_steps=20,
        batch_length=4,
    )

    monkeypatch.setattr(
        evaluation, "dispatch_schema8_completed_bundle", lambda *a, **k: None
    )
    monkeypatch.setattr(
        evaluation, "dispatch_schema9_completed_bundle", lambda *a, **k: None
    )
    monkeypatch.setattr(
        evaluation,
        "dispatch_schema10_completed_bundle",
        lambda *a, **k: copy.deepcopy(completed),
    )
    monkeypatch.setattr(
        evaluation,
        "validate_schema10_completed_bundle",
        lambda *a, **k: copy.deepcopy(post),
    )

    def hashes(path):
        nonlocal hash_calls
        hash_calls += 1
        digest = (
            "b" * 64
            if mutation == "hash" and hash_calls > 1
            else config_digest
        )
        return {"config_c.yaml": digest}

    def private_copy(flags):
        runtime = copy.deepcopy(flags)
        runtime.train_actor = False
        runtime.train_model = False
        runtime.parallel = False
        runtime.parallel_actor = False
        runtime.voc_actor_policy_barrier_runtime = False
        return runtime, {"evaluation_copy": {"train_model": False}}

    def environment(*args, **kwargs):
        nonlocal environment_calls
        environment_calls += 1
        raise AssertionError("environment ran after schema-10 mutation")

    monkeypatch.setattr(evaluation, "checkpoint_hashes", hashes)
    monkeypatch.setattr(
        evaluation, "validate_completion_marker", lambda path: marker
    )
    monkeypatch.setattr(
        evaluation, "validate_schema6_completed_bundle", lambda path: None
    )
    monkeypatch.setattr(
        evaluation, "validate_schema7_completed_bundle", lambda path: None
    )
    monkeypatch.setattr(evaluation, "evaluation_runtime_flags", private_copy)
    monkeypatch.setattr(
        smoke,
        "_load_flags",
        lambda args, *, config_payload, expected_sha256: training,
    )
    monkeypatch.setattr(wrapper, "create_envpool", environment)

    with pytest.raises(RuntimeError, match="changed across private"):
        smoke.run_smoke(
            Namespace(
                seed=1,
                device=torch.device("cpu"),
                checkpoint_dir=tmp_path,
                config=None,
                scored_length=4,
                env_name="Enduro-v5",
                batch_size=1,
            )
        )

    assert environment_calls == 0


@pytest.mark.parametrize(
    ("schema", "model_flags", "active_state"),
    [
        (
            1,
            {"dynamic_voc_mode": "control"},
            {
                "voc_gate_adam_beta1": 0.9,
                "voc_gate_param_align": False,
                "voc_gate_param_align_coef": 1.0,
                "voc_gate_exact_projection": False,
                "voc_gate_epsilon_greedy_execution": False,
            },
        ),
        (
            2,
            {
                "dynamic_voc_mode": "control",
                "voc_gate_adam_beta1": 0.0,
            },
            {
                "voc_gate_adam_beta1": 0.0,
                "voc_gate_param_align": False,
                "voc_gate_param_align_coef": 1.0,
                "voc_gate_exact_projection": False,
                "voc_gate_epsilon_greedy_execution": False,
            },
        ),
        (
            3,
            {
                "dynamic_voc_mode": "control",
                "voc_gate_adam_beta1": 0.0,
                "voc_gate_param_align": False,
                "voc_gate_param_align_coef": 1.0,
            },
            {
                "voc_gate_adam_beta1": 0.0,
                "voc_gate_param_align": False,
                "voc_gate_param_align_coef": 1.0,
                "voc_gate_exact_projection": False,
                "voc_gate_epsilon_greedy_execution": False,
            },
        ),
        (
            4,
            {
                "dynamic_voc_mode": "control",
                "voc_gate_adam_beta1": 0.0,
                "voc_gate_param_align": False,
                "voc_gate_param_align_coef": 1.0,
                "voc_gate_exact_projection": True,
            },
            {
                "voc_gate_adam_beta1": 0.0,
                "voc_gate_param_align": False,
                "voc_gate_param_align_coef": 1.0,
                "voc_gate_exact_projection": True,
                "voc_gate_epsilon_greedy_execution": False,
            },
        ),
    ],
)
def test_smoke_checkpoint_metadata_preserves_legacy_schema_meaning(
    monkeypatch, schema, model_flags, active_state
):
    _patch_public_checkpoint_validators(monkeypatch)
    actor_state = {
        "dynamic_voc_mode": "control",
        "voc_gate_policy_schema_version": schema,
        **active_state,
    }

    resolved = smoke._validate_smoke_checkpoint_metadata(
        {"flags": dict(model_flags)},
        {"flags": dict(model_flags)},
        Namespace(**model_flags),
        object(),
        actor_state,
    )

    assert resolved["model_gate_policy"][
        "voc_gate_policy_schema_version"
    ] == schema
    assert resolved["model_gate_policy"][
        "voc_gate_epsilon_greedy_execution"
    ] is False
    assert resolved["resolved_identity"] is None


@pytest.mark.parametrize("bad_execution", [False, 1, None])
def test_smoke_checkpoint_metadata_rejects_model_execution_mismatch(
    monkeypatch, bad_execution
):
    identity = _schema5_identity_values()
    model_identity = dict(identity)
    if bad_execution is None:
        model_identity.pop("voc_gate_epsilon_greedy_execution")
    else:
        model_identity["voc_gate_epsilon_greedy_execution"] = bad_execution
    _patch_public_checkpoint_validators(monkeypatch)

    with pytest.raises(
        ValueError,
        match=(
            "voc_gate_policy_schema_version|"
            "voc_gate_epsilon_greedy_execution"
        ),
    ):
        smoke._validate_smoke_checkpoint_metadata(
            {"flags": dict(identity)},
            {"flags": model_identity},
            Namespace(**identity),
            object(),
            _schema5_active_state(),
        )


def test_smoke_checkpoint_metadata_rejects_explicit_model_schema_mismatch(
    monkeypatch,
):
    identity = _schema5_identity_values()
    _patch_public_checkpoint_validators(monkeypatch)

    with pytest.raises(ValueError, match="schema 4 predates epsilon-greedy"):
        smoke._validate_smoke_checkpoint_metadata(
            {"flags": dict(identity)},
            {
                "flags": dict(identity),
                "voc_gate_policy_schema_version": 4,
            },
            Namespace(**identity),
            object(),
            _schema5_active_state(),
        )


def test_smoke_checkpoint_metadata_rejects_actor_schema5_identity_mismatch(
    monkeypatch,
):
    identity = _schema5_identity_values()
    actor_identity = dict(identity)
    actor_identity["base_seed"] = 3
    _patch_public_checkpoint_validators(monkeypatch)

    with pytest.raises(ValueError, match="base_seed"):
        smoke._validate_smoke_checkpoint_metadata(
            {"flags": actor_identity},
            {"flags": dict(identity)},
            Namespace(**identity),
            object(),
            _schema5_active_state(),
        )


@pytest.mark.parametrize(
    ("surface", "name", "bad_value"),
    [
        ("config", "float16", False),
        ("actor", "float16", False),
        ("model", "float16", False),
        ("config", "model_float16", True),
        ("actor", "model_float16", True),
        ("model", "model_float16", True),
    ],
)
def test_smoke_checkpoint_metadata_rejects_precision_identity_mismatch(
    monkeypatch, surface, name, bad_value
):
    identity = _schema5_identity_values()
    configured = dict(identity)
    actor_identity = dict(identity)
    model_identity = dict(identity)
    if surface == "config":
        configured[name] = bad_value
    elif surface == "actor":
        actor_identity[name] = bad_value
    else:
        model_identity[name] = bad_value
    _patch_public_checkpoint_validators(monkeypatch)

    with pytest.raises(ValueError, match=name):
        smoke._validate_smoke_checkpoint_metadata(
            {"flags": actor_identity},
            {"flags": model_identity},
            Namespace(**configured),
            object(),
            _schema5_active_state(),
        )


@pytest.mark.parametrize(
    ("name", "bad_value"),
    [("float16", False), ("model_float16", True)],
)
def test_smoke_checkpoint_metadata_rejects_consistently_rewritten_precision(
    monkeypatch, name, bad_value
):
    identity = _schema5_identity_values()
    rewritten = {**identity, name: bad_value}
    _patch_public_checkpoint_validators(monkeypatch)

    with pytest.raises(ValueError, match=name):
        smoke._validate_smoke_checkpoint_metadata(
            {"flags": dict(rewritten)},
            {"flags": dict(rewritten)},
            Namespace(**rewritten),
            object(),
            _schema5_active_state(),
        )


def test_smoke_checkpoint_metadata_rejects_missing_model_flags(monkeypatch):
    identity = _schema5_identity_values()
    _patch_public_checkpoint_validators(monkeypatch)

    with pytest.raises(ValueError, match="ModelNet.*embedded training flags"):
        smoke._validate_smoke_checkpoint_metadata(
            {"flags": dict(identity)},
            {},
            Namespace(**identity),
            object(),
            _schema5_active_state(),
        )


@pytest.mark.parametrize(
    ("name", "bad_value"),
    [
        ("base_seed", 3),
        ("total_steps", 100000),
        ("schedule_total_steps", 300000),
        ("voc_gate_param_align", True),
        ("voc_gate_param_align_coef", np.nextafter(1.0, 2.0)),
        ("voc_gate_exact_projection", False),
        ("voc_train_epsilon", np.nextafter(0.02, 1.0)),
        ("voc_eval_stochastic", False),
        ("float16", False),
        ("model_float16", True),
        ("ckp", True),
        ("preload", "/tmp/parent"),
        ("preload_actor", "/tmp/actor"),
        ("voc_parent_checkpoint", "/tmp/parent-actor.tar"),
    ],
)
def test_smoke_checkpoint_metadata_rejects_model_schema5_identity_mismatch(
    monkeypatch, name, bad_value
):
    identity = _schema5_identity_values()
    model_identity = dict(identity)
    model_identity[name] = bad_value
    _patch_public_checkpoint_validators(monkeypatch)

    with pytest.raises(ValueError, match=name):
        smoke._validate_smoke_checkpoint_metadata(
            {"flags": dict(identity)},
            {"flags": model_identity},
            Namespace(**identity),
            object(),
            _schema5_active_state(),
        )


@pytest.mark.parametrize(
    ("name", "bad_value"),
    [
        ("base_seed", True),
        ("total_steps", 0),
        ("schedule_total_steps", False),
        ("voc_train_epsilon", True),
        ("voc_train_epsilon", float("nan")),
        ("voc_eval_stochastic", 1),
        ("float16", 1),
        ("model_float16", None),
        ("ckp", 0),
        ("preload", None),
    ],
)
def test_smoke_checkpoint_metadata_rejects_malformed_model_identity(
    monkeypatch, name, bad_value
):
    identity = _schema5_identity_values()
    model_identity = dict(identity)
    model_identity[name] = bad_value
    _patch_public_checkpoint_validators(monkeypatch)

    with pytest.raises(ValueError, match=name):
        smoke._validate_smoke_checkpoint_metadata(
            {"flags": dict(identity)},
            {"flags": model_identity},
            Namespace(**identity),
            object(),
            _schema5_active_state(),
        )


_V12_WIRE_CHECKPOINT_DIR = Path(
    os.environ.get(
        "THINKER_V12_WIRE_CHECKPOINT_DIR",
        "/tmp/di-voc-v12-epsgreedy-final-QcxfH0/runs/"
        "enduro-voc-v12-epsgreedy-sentinel-wire1200",
    )
)


@pytest.fixture(scope="module")
def v12_production_wire_bundle():
    checkpoint_dir = _V12_WIRE_CHECKPOINT_DIR
    required = (
        checkpoint_dir / "config_c.yaml",
        checkpoint_dir / "ckp_actor.tar",
        checkpoint_dir / "ckp_model.tar",
    )
    if not all(path.is_file() for path in required):
        pytest.skip(
            "set THINKER_V12_WIRE_CHECKPOINT_DIR to the preserved v12 "
            "production wire bundle"
        )
    data_root = checkpoint_dir.parents[1] / "data" / "behavioral_data_block"
    if not data_root.is_dir():
        pytest.skip("preserved v12 production behavioral-data tree is absent")
    return checkpoint_dir, data_root


def _v12_production_smoke_args(checkpoint_dir, data_root):
    return smoke.parse_args(
        [
            "--env-name",
            "Enduro-v5",
            "--game-id",
            "0",
            "--data-root",
            str(data_root),
            "--subjects",
            "1",
            "--sessions",
            "1,2,3",
            "--batch-size",
            "1",
            "--scored-length",
            "4",
            "--device",
            "cpu",
            "--seed",
            "2026",
            "--checkpoint-dir",
            str(checkpoint_dir),
        ]
    )


def test_run_smoke_accepts_real_v12_production_wire_precision_identity(
    v12_production_wire_bundle,
):
    checkpoint_dir, data_root = v12_production_wire_bundle

    result = smoke.run_smoke(
        _v12_production_smoke_args(checkpoint_dir, data_root)
    )

    assert result["voc_checkpoint_validation"]["voc_float16"] is True
    assert result["actor_checkpoint_public_validation"]["voc"][
        "voc_float16"
    ] is True
    assert result["model_checkpoint_public_validation"][
        "embedded_protocol_verified"
    ] is True
    for surface in ("config", "actor_checkpoint", "model_checkpoint"):
        identity = result["voc_checkpoint_resolved_identity"][surface]
        assert identity["float16"] is True
        assert identity["model_float16"] is False
    assert result["actor_changed_parameter_count"] > 0
    assert result["model_versions_unchanged"] is True


def test_run_smoke_rejects_real_v12_bundle_with_rewritten_precision_identity(
    tmp_path, v12_production_wire_bundle
):
    checkpoint_dir, data_root = v12_production_wire_bundle
    bad_checkpoint_dir = tmp_path / "bad-precision-bundle"
    bad_checkpoint_dir.mkdir()
    config = yaml.safe_load((checkpoint_dir / "config_c.yaml").read_text())
    assert config["float16"] is True
    config["float16"] = False
    (bad_checkpoint_dir / "config_c.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True), encoding="utf-8"
    )
    for name in ("ckp_actor.tar", "ckp_model.tar"):
        (bad_checkpoint_dir / name).symlink_to(checkpoint_dir / name)

    with pytest.raises(ValueError, match="float16 disagrees with the run"):
        smoke.run_smoke(
            _v12_production_smoke_args(bad_checkpoint_dir, data_root)
        )
