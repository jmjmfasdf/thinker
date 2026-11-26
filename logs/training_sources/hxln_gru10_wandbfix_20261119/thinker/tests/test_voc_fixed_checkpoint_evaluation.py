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

import evaluate_dynamic_imitation as evaluation
import evaluate_voc_fixed_checkpoint as fixed_eval
from thinker import util


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_completed_bundle(tmp_path: Path):
    snapshot = tmp_path / "snapshot"
    source_root = snapshot / "src" / "thinker"
    package = source_root / "thinker"
    package.mkdir(parents=True)
    (source_root / "train.py").write_text("# training entry\n", encoding="utf-8")
    (package / "util.py").write_text("# util\n", encoding="utf-8")
    extension = package / "cenv.fake.so"
    extension.write_bytes(b"compiled-cenv")

    checkpoint_dir = snapshot / "runs" / "enduro-200k"
    checkpoint_dir.mkdir(parents=True)
    for name, payload in {
        "config_c.yaml": b"total_steps: 200000\n",
        "ckp_actor.tar": b"actor-final",
        "ckp_model.tar": b"model-final",
    }.items():
        (checkpoint_dir / name).write_bytes(payload)
    marker = {
        "schema_version": 1,
        "status": "complete",
        "checkpoint_files": {
            name: {
                "sha256": _sha(checkpoint_dir / name),
                "size": (checkpoint_dir / name).stat().st_size,
            }
            for name in fixed_eval.REQUIRED_CHECKPOINT_FILES
        },
        "implementation_sources": {
            "train.py": {"sha256": _sha(source_root / "train.py")},
            "thinker/util.py": {"sha256": _sha(package / "util.py")},
        },
        "loaded_extensions": {
            "thinker/cenv.fake.so": {"sha256": _sha(extension)}
        },
    }
    (checkpoint_dir / "finish").write_text(
        json.dumps(marker), encoding="utf-8"
    )
    source_manifest = snapshot / "source.sha256"
    source_files = {
        "thinker/train.py": source_root / "train.py",
        "thinker/thinker/util.py": package / "util.py",
        "thinker/thinker/cenv.fake.so": extension,
    }
    source_manifest.write_text(
        "".join(
            f"{_sha(path)}  ./{relative}\n"
            for relative, path in sorted(source_files.items())
        ),
        encoding="utf-8",
    )
    (source_root / "train.py").chmod(0o444)
    (package / "util.py").chmod(0o444)
    extension.chmod(0o555)
    package.chmod(0o555)
    source_root.chmod(0o555)
    (snapshot / "src").chmod(0o555)
    source_manifest.chmod(0o444)
    return checkpoint_dir, source_root


def _row(
    *,
    stream,
    step,
    depth,
    previous,
    control,
    delta,
    probability,
):
    gate = fixed_eval.GATE_STOP if control == fixed_eval.STOP else fixed_eval.GATE_CONTINUE
    selected = 0.25
    return {
        "stream_id": stream,
        "augmented_step": step,
        "decision_depth": depth,
        "predecision_last_control": previous,
        "sampled_control": control,
        "gate_action": gate,
        "continue_probability": probability,
        "ema_delta_q": delta,
        "online_delta_q": delta,
        "ema_selected_q": selected,
        "online_selected_q": selected,
        "calibration_net_target": selected,
    }


def test_bundle_validation_binds_final_files_source_and_extension(tmp_path):
    checkpoint_dir, source_root = _make_completed_bundle(tmp_path)
    validated = fixed_eval.validate_checkpoint_bundle(checkpoint_dir)
    assert validated.source_root == source_root.resolve()
    assert set(validated.file_hashes) == {
        "config_c.yaml",
        "ckp_actor.tar",
        "ckp_model.tar",
        "finish",
    }
    assert validated.source_manifest["file_count"] == 3
    assert validated.source_manifest["path_set_exact"] is True
    assert validated.source_manifest["writable_node_count"] == 0

    (checkpoint_dir / "ckp_actor.tar").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash disagrees"):
        fixed_eval.validate_checkpoint_bundle(
            checkpoint_dir, training_source_root=source_root
        )


@pytest.mark.parametrize("name", ["finish", "ckp_actor.tar"])
def test_bundle_validation_rejects_hardlinked_attested_file(tmp_path, name):
    checkpoint_dir, source_root = _make_completed_bundle(tmp_path)
    target = checkpoint_dir / name
    os.link(target, tmp_path / f"second-link-{name}")
    with pytest.raises(ValueError, match="exactly one hard link"):
        fixed_eval.validate_checkpoint_bundle(
            checkpoint_dir, training_source_root=source_root
        )


@pytest.mark.parametrize("name", ["finish", "ckp_actor.tar"])
def test_bundle_validation_rejects_symlinked_attested_file(tmp_path, name):
    checkpoint_dir, source_root = _make_completed_bundle(tmp_path)
    target = checkpoint_dir / name
    replacement = tmp_path / f"symlink-target-{name}"
    replacement.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(replacement)
    with pytest.raises(ValueError, match="regular non-symlink file"):
        fixed_eval.validate_checkpoint_bundle(
            checkpoint_dir, training_source_root=source_root
        )


@pytest.mark.parametrize(
    "needle,replacement",
    [
        (
            '"schema_version": 1',
            '"schema_version": 1, "schema_version": 1',
        ),
        ('"size": ', '"size": 1, "size": '),
    ],
)
def test_bundle_validation_rejects_duplicate_json_keys(
    tmp_path, needle, replacement
):
    checkpoint_dir, source_root = _make_completed_bundle(tmp_path)
    finish = checkpoint_dir / "finish"
    rendered = finish.read_text(encoding="utf-8")
    assert needle in rendered
    finish.write_text(rendered.replace(needle, replacement, 1), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        fixed_eval.validate_checkpoint_bundle(
            checkpoint_dir, training_source_root=source_root
        )


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("schema_version", True, "schema_version"),
        ("schema_version", 1.0, "schema_version"),
        ("status", True, "completed schema-v1"),
        ("status", 1, "completed schema-v1"),
    ],
)
def test_bundle_validation_rejects_completion_scalar_type_drift(
    tmp_path, field, value, match
):
    checkpoint_dir, source_root = _make_completed_bundle(tmp_path)
    finish = checkpoint_dir / "finish"
    marker = json.loads(finish.read_text(encoding="utf-8"))
    marker[field] = value
    finish.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        fixed_eval.validate_checkpoint_bundle(
            checkpoint_dir, training_source_root=source_root
        )


@pytest.mark.parametrize("operation", ["missing", "extra"])
def test_bundle_validation_requires_exact_checkpoint_file_names(
    tmp_path, operation
):
    checkpoint_dir, source_root = _make_completed_bundle(tmp_path)
    finish = checkpoint_dir / "finish"
    marker = json.loads(finish.read_text(encoding="utf-8"))
    if operation == "missing":
        del marker["checkpoint_files"]["ckp_model.tar"]
    else:
        marker["checkpoint_files"]["unexpected.tar"] = {
            "sha256": "0" * 64,
            "size": 1,
        }
    finish.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(ValueError, match="file names disagree"):
        fixed_eval.validate_checkpoint_bundle(
            checkpoint_dir, training_source_root=source_root
        )


@pytest.mark.parametrize("operation", ["missing", "extra"])
def test_bundle_validation_requires_exact_checkpoint_record_fields(
    tmp_path, operation
):
    checkpoint_dir, source_root = _make_completed_bundle(tmp_path)
    finish = checkpoint_dir / "finish"
    marker = json.loads(finish.read_text(encoding="utf-8"))
    record = marker["checkpoint_files"]["ckp_actor.tar"]
    if operation == "missing":
        del record["sha256"]
    else:
        record["unexpected"] = False
    finish.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid record fields"):
        fixed_eval.validate_checkpoint_bundle(
            checkpoint_dir, training_source_root=source_root
        )


@pytest.mark.parametrize("value", [True, 1.0, "1", 0, -1])
def test_bundle_validation_rejects_checkpoint_size_type_drift(tmp_path, value):
    checkpoint_dir, source_root = _make_completed_bundle(tmp_path)
    finish = checkpoint_dir / "finish"
    marker = json.loads(finish.read_text(encoding="utf-8"))
    marker["checkpoint_files"]["ckp_actor.tar"]["size"] = value
    finish.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid size"):
        fixed_eval.validate_checkpoint_bundle(
            checkpoint_dir, training_source_root=source_root
        )


@pytest.mark.parametrize("value", [True, 1, "A" * 64, "0" * 63])
def test_bundle_validation_rejects_checkpoint_hash_type_or_syntax_drift(
    tmp_path, value
):
    checkpoint_dir, source_root = _make_completed_bundle(tmp_path)
    finish = checkpoint_dir / "finish"
    marker = json.loads(finish.read_text(encoding="utf-8"))
    marker["checkpoint_files"]["ckp_actor.tar"]["sha256"] = value
    finish.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid SHA-256"):
        fixed_eval.validate_checkpoint_bundle(
            checkpoint_dir, training_source_root=source_root
        )


def test_single_link_reader_rejects_in_read_identity_drift(monkeypatch, tmp_path):
    path = tmp_path / "attested.json"
    path.write_bytes(b"{}")
    real_read = fixed_eval.os.read
    mutated = False

    def mutating_read(descriptor, count):
        nonlocal mutated
        if not mutated:
            mutated = True
            path.write_bytes(b'{"changed":true}')
        return real_read(descriptor, count)

    monkeypatch.setattr(fixed_eval.os, "read", mutating_read)
    with pytest.raises(RuntimeError, match="changed while it was read"):
        fixed_eval._read_stable_single_link_bytes(path, label="test file")


def test_fixed_checkpoint_loader_never_reopens_swapped_path(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "ckp_actor.tar"
    bound_buffer = io.BytesIO()
    replacement_buffer = io.BytesIO()
    torch.save({"generation": "bound"}, bound_buffer)
    torch.save({"generation": "replacement"}, replacement_buffer)
    bound_payload = bound_buffer.getvalue()
    checkpoint_path.write_bytes(bound_payload)
    checkpoint_files = {
        "ckp_actor.tar": {
            "sha256": hashlib.sha256(bound_payload).hexdigest(),
            "size": len(bound_payload),
        }
    }
    stable_reader = fixed_eval._read_stable_single_link_bytes

    def swap_after_stable_read(path, *, label):
        payload = stable_reader(path, label=label)
        checkpoint_path.write_bytes(replacement_buffer.getvalue())
        return payload

    monkeypatch.setattr(
        fixed_eval, "_read_stable_single_link_bytes", swap_after_stable_read
    )

    checkpoint = fixed_eval._load_checkpoint_from_bound_bytes(
        tmp_path,
        "ckp_actor.tar",
        checkpoint_files,
        label="test fixed actor checkpoint",
    )

    assert checkpoint["generation"] == "bound"
    assert torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )["generation"] == "replacement"


def test_fixed_runtime_loader_preserves_legacy_symlink_differential(tmp_path):
    target = tmp_path / "actor-target.tar"
    torch.save({"generation": "legacy"}, target)
    (tmp_path / "ckp_actor.tar").symlink_to(target)
    payload = target.read_bytes()
    checkpoint_files = {
        "ckp_actor.tar": {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size": len(payload),
        }
    }

    legacy = fixed_eval._load_fixed_runtime_checkpoint(
        tmp_path,
        "ckp_actor.tar",
        checkpoint_files,
        v20=False,
        label="legacy fixed actor checkpoint",
    )

    assert legacy["generation"] == "legacy"
    with pytest.raises(ValueError, match="regular non-symlink"):
        fixed_eval._load_fixed_runtime_checkpoint(
            tmp_path,
            "ckp_actor.tar",
            checkpoint_files,
            v20=True,
            label="v20 fixed actor checkpoint",
        )


def test_bundle_validation_rejects_source_drift(tmp_path):
    checkpoint_dir, source_root = _make_completed_bundle(tmp_path)
    util_path = source_root / "thinker" / "util.py"
    util_path.chmod(0o644)
    util_path.write_text(
        "# changed\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="training-source-root"):
        fixed_eval.validate_checkpoint_bundle(
            checkpoint_dir, training_source_root=source_root
        )


def test_full_manifest_rejects_writable_or_unlisted_source_node(tmp_path):
    checkpoint_dir, source_root = _make_completed_bundle(tmp_path)
    train_path = source_root / "train.py"
    train_path.chmod(0o644)
    with pytest.raises(PermissionError, match="writable file"):
        fixed_eval.validate_checkpoint_bundle(
            checkpoint_dir, training_source_root=source_root
        )

    train_path.chmod(0o444)
    package = source_root / "thinker"
    package.chmod(0o755)
    extra = package / "importable_drift.py"
    extra.write_text("DRIFT = True\n", encoding="utf-8")
    extra.chmod(0o444)
    package.chmod(0o555)
    with pytest.raises(ValueError, match="path set disagrees"):
        fixed_eval.validate_checkpoint_bundle(
            checkpoint_dir, training_source_root=source_root
        )


def test_rng_reset_repeats_stochastic_gate_draws():
    logits = torch.tensor([[0.4, -0.2], [-0.5, 0.8]], dtype=torch.float32)
    fixed_eval._set_deterministic_seed(20260827)
    first = torch.distributions.Categorical(logits=logits).sample((32,))
    first_numpy = __import__("numpy").random.uniform(size=8)
    first_python = [__import__("random").random() for _ in range(8)]

    fixed_eval._set_deterministic_seed(20260827)
    second = torch.distributions.Categorical(logits=logits).sample((32,))
    second_numpy = __import__("numpy").random.uniform(size=8)
    second_python = [__import__("random").random() for _ in range(8)]

    torch.testing.assert_close(first, second)
    assert (first_numpy == second_numpy).all()
    assert first_python == second_python


def _schema_five_actor_eval_fixture():
    soft_joint_probability = torch.tensor(
        [[[0.4, 0.4, 0.2], [0.1, 0.1, 0.8], [0.25, 0.25, 0.5]]],
        dtype=torch.float32,
    )
    execution_joint_probability = torch.tensor(
        [[[0.5, 0.5, 0.0], [0.0, 0.0, 1.0], [0.25, 0.25, 0.5]]],
        dtype=torch.float32,
    )

    def finite_log(probability):
        return torch.where(
            probability > 0.0,
            probability.clamp_min(torch.finfo(probability.dtype).tiny).log(),
            torch.full_like(probability, -1000.0),
        )

    actor_out = SimpleNamespace(
        search_control_logits=finite_log(execution_joint_probability),
        misc={
            "voc_gate_soft_control_logits": finite_log(
                soft_joint_probability
            ),
            "voc_gate_soft_continue_probability": torch.tensor(
                [[0.8, 0.2, 0.5]], dtype=torch.float32
            ),
            "voc_gate_execution_continue_probability": torch.tensor(
                [[1.0, 0.0, 0.5]], dtype=torch.float32
            ),
            "voc_gate_log_odds": torch.tensor(
                [[np.log(4.0), -np.log(4.0), 0.0]], dtype=torch.float32
            ),
        },
        voc_features=torch.zeros((1, 3, 2), dtype=torch.float32),
        baseline=torch.zeros((1, 3, 2), dtype=torch.float32),
        voc_q=torch.zeros((1, 3, 2), dtype=torch.float32),
        search_control=torch.tensor(
            [[fixed_eval.PROCEED, fixed_eval.STOP, fixed_eval.STOP]]
        ),
        control_valid=torch.ones((1, 3), dtype=torch.bool),
    )
    ema_weight = torch.zeros((2, 2), dtype=torch.float32)
    ema_bias = torch.zeros(2, dtype=torch.float32)
    return actor_out, ema_weight, ema_bias


def test_schema_five_actor_eval_uses_soft_probability_and_execution_action():
    actor_out, ema_weight, ema_bias = _schema_five_actor_eval_fixture()
    evaluated = fixed_eval._actor_eval_tensors(
        actor_out,
        ema_weight,
        ema_bias,
        reward_names=("re", "think"),
        think_cost=0.0,
        epsilon_greedy_execution=True,
    )

    torch.testing.assert_close(
        evaluated["continue_probability"],
        torch.tensor([0.8, 0.2, 0.5]),
    )
    assert torch.equal(
        evaluated["gate_action"],
        torch.tensor(
            [fixed_eval.GATE_CONTINUE, fixed_eval.GATE_STOP, fixed_eval.GATE_STOP]
        ),
    )


def test_schema_five_actor_eval_uses_recorded_soft_probability_as_authority():
    actor_out, ema_weight, ema_bias = _schema_five_actor_eval_fixture()
    recorded = actor_out.misc["voc_gate_soft_continue_probability"]
    recorded[0, 0] = torch.nextafter(
        recorded[0, 0], torch.tensor(1.0, dtype=recorded.dtype)
    )
    evaluated = fixed_eval._actor_eval_tensors(
        actor_out,
        ema_weight,
        ema_bias,
        reward_names=("re", "think"),
        think_cost=0.0,
        epsilon_greedy_execution=True,
    )
    assert torch.equal(evaluated["continue_probability"], recorded[-1])


def test_legacy_actor_eval_ignores_schema_five_soft_misc_surface():
    actor_out, ema_weight, ema_bias = _schema_five_actor_eval_fixture()
    evaluated = fixed_eval._actor_eval_tensors(
        actor_out,
        ema_weight,
        ema_bias,
        reward_names=("re", "think"),
        think_cost=0.0,
        epsilon_greedy_execution=False,
    )
    torch.testing.assert_close(
        evaluated["continue_probability"],
        torch.tensor([1.0, 0.0, 0.5]),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "missing",
    [
        "voc_gate_soft_control_logits",
        "voc_gate_soft_continue_probability",
        "voc_gate_execution_continue_probability",
        "voc_gate_log_odds",
    ],
)
def test_schema_five_actor_eval_requires_all_dual_gate_surfaces(missing):
    actor_out, ema_weight, ema_bias = _schema_five_actor_eval_fixture()
    del actor_out.misc[missing]
    with pytest.raises(RuntimeError, match=missing):
        fixed_eval._actor_eval_tensors(
            actor_out,
            ema_weight,
            ema_bias,
            reward_names=("re", "think"),
            think_cost=0.0,
            epsilon_greedy_execution=True,
        )


def test_schema_five_actor_eval_rejects_execution_or_action_drift():
    actor_out, ema_weight, ema_bias = _schema_five_actor_eval_fixture()
    actor_out.misc["voc_gate_execution_continue_probability"][0, 0] = 0.5
    with pytest.raises(RuntimeError, match="behavior logits/execution"):
        fixed_eval._actor_eval_tensors(
            actor_out,
            ema_weight,
            ema_bias,
            reward_names=("re", "think"),
            think_cost=0.0,
            epsilon_greedy_execution=True,
        )

    actor_out, ema_weight, ema_bias = _schema_five_actor_eval_fixture()
    actor_out.search_control[0, 0] = fixed_eval.STOP
    with pytest.raises(RuntimeError, match="zero execution probability"):
        fixed_eval._actor_eval_tensors(
            actor_out,
            ema_weight,
            ema_bias,
            reward_names=("re", "think"),
            think_cost=0.0,
            epsilon_greedy_execution=True,
        )


def test_ordered_decision_semantic_hash_is_repeatable_and_sensitive():
    row = {field: 0 for field in fixed_eval.DECISION_CSV_FIELDS}
    repeated_a = fixed_eval.decision_rows_semantic_sha256([row, dict(row)])
    repeated_b = fixed_eval.decision_rows_semantic_sha256([dict(row), dict(row)])
    assert repeated_a == repeated_b
    changed = dict(row)
    changed["continue_probability"] = 0.5
    assert fixed_eval.decision_rows_semantic_sha256([row, changed]) != repeated_a


def test_behavioral_training_data_signature_is_recomputed(monkeypatch, tmp_path):
    from thinker import bc_loader

    archive = tmp_path / "training.npz"
    archive.write_bytes(b"immutable behavioral bytes")

    class DummyLoader:
        def __init__(self, **kwargs):
            assert Path(kwargs["base_path"]) == tmp_path
            self.data_files = [archive]

    monkeypatch.setattr(bc_loader, "FrameStackedBehavioralDataLoader", DummyLoader)
    monkeypatch.setattr(
        bc_loader, "behavioral_data_signature", lambda loader, root: "a" * 64
    )

    def verify(checkpoint, signature):
        assert checkpoint["imitation_data_signature"] == signature
        return {
            "imitation_data_signature": signature,
            "training_data_signature_recomputed": True,
        }

    spec = SimpleNamespace(
        subjects=(1,),
        train_sessions=(1, 2, 3),
        game_id=0,
        scored_length=4,
        frame_stack_n=4,
        target_size=(84, 84),
        grayscale=False,
        observation_dtype="uint8",
        num_actions=9,
    )
    state = fixed_eval.validate_behavioral_training_data(
        flags=SimpleNamespace(),
        spec=spec,
        actor_checkpoint={"imitation_data_signature": "a" * 64},
        data_root=tmp_path,
        checkpoint_eval=SimpleNamespace(
            verify_actor_behavioral_data_signature=verify
        ),
    )
    assert state["training_data_signature_recomputed"] is True
    assert state["file_count"] == 1
    assert state["files"][0]["sha256"] == _sha(archive)
    post = fixed_eval.revalidate_behavioral_training_data(state)
    assert post["unchanged"] is True

    archive.write_bytes(b"drifted behavioral bytes")
    with pytest.raises(RuntimeError, match="behavioral training data changed"):
        fixed_eval.revalidate_behavioral_training_data(state)

    archive.write_bytes(b"immutable behavioral bytes")
    signatures = iter(("a" * 64, "b" * 64))
    monkeypatch.setattr(
        bc_loader,
        "behavioral_data_signature",
        lambda loader, root: next(signatures),
    )
    with pytest.raises(RuntimeError, match="between signature passes"):
        fixed_eval.validate_behavioral_training_data(
            flags=SimpleNamespace(),
            spec=spec,
            actor_checkpoint={"imitation_data_signature": "a" * 64},
            data_root=tmp_path,
            checkpoint_eval=SimpleNamespace(
                verify_actor_behavioral_data_signature=verify
            ),
        )


def test_dueling_q_and_on_policy_calibration_match_training_formula():
    raw = torch.tensor([[[2.0, 0.0]]])
    state = torch.tensor([[10.0]])
    probability = torch.tensor([[0.25]])
    q = fixed_eval.reconstruct_dueling_q(raw, state, probability)
    torch.testing.assert_close(q, torch.tensor([[[11.5, 9.5]]]))

    target = fixed_eval.on_policy_vtrace_target(
        rewards=torch.tensor([[1.0], [2.0]]),
        discounts=torch.tensor([[1.0], [0.0]]),
        values=torch.zeros((2, 1)),
        bootstrap_value=torch.zeros(1),
        lamb=1.0,
    )
    torch.testing.assert_close(target, torch.tensor([[3.0], [2.0]]))


def test_strict_useful_pair_rejects_gap_token_and_depth_mismatch():
    rows = [
        _row(
            stream=0,
            step=0,
            depth=4,
            previous=fixed_eval.STOP,
            control=fixed_eval.PROCEED,
            delta=0.2,
            probability=0.8,
        ),
        # Immediate but wrong predecision token.
        _row(
            stream=0,
            step=1,
            depth=5,
            previous=fixed_eval.RESET,
            control=fixed_eval.STOP,
            delta=-0.2,
            probability=0.2,
        ),
        _row(
            stream=1,
            step=0,
            depth=4,
            previous=fixed_eval.STOP,
            control=fixed_eval.RESET,
            delta=0.2,
            probability=0.8,
        ),
        # Correct token, wrong depth.
        _row(
            stream=1,
            step=1,
            depth=6,
            previous=fixed_eval.RESET,
            control=fixed_eval.STOP,
            delta=-0.2,
            probability=0.2,
        ),
        _row(
            stream=2,
            step=0,
            depth=4,
            previous=fixed_eval.STOP,
            control=fixed_eval.PROCEED,
            delta=0.2,
            probability=0.8,
        ),
        # Correct semantics but not an immediate adjacent augmented row.
        _row(
            stream=2,
            step=2,
            depth=5,
            previous=fixed_eval.PROCEED,
            control=fixed_eval.STOP,
            delta=-0.2,
            probability=0.2,
        ),
    ]
    candidates, pairs = fixed_eval._strict_useful_pairs(
        rows, tie_tolerance=fixed_eval.TIE_TOLERANCE
    )
    assert len(candidates) == 3
    assert pairs == []


def test_pooled_output_applies_all_four_exact_behavior_definitions():
    rows = []
    # Every three augmented rows form two strict useful transitions:
    # positive P/R -> positive P/R -> deep negative STOP.  This supplies
    # 512 P and 512 R pairs, and 512 observations in each next-Q branch.
    for group in range(512):
        control = fixed_eval.PROCEED if group < 256 else fixed_eval.RESET
        base = group * 3
        rows.extend(
            [
                _row(
                    stream=0,
                    step=base,
                    depth=8,
                    previous=fixed_eval.STOP,
                    control=control,
                    delta=0.2,
                    probability=0.8,
                ),
                _row(
                    stream=0,
                    step=base + 1,
                    depth=9,
                    previous=control,
                    control=control,
                    delta=0.2,
                    probability=0.8,
                ),
                _row(
                    stream=0,
                    step=base + 2,
                    depth=10,
                    previous=control,
                    control=fixed_eval.STOP,
                    delta=-0.2,
                    probability=0.2,
                ),
            ]
        )
    summary = fixed_eval.summarize_decision_rows(
        rows,
        q_temperature=0.05,
        stage_end_count=512,
        forced_stop_count=0,
    )
    assert summary["all_four_behaviors_pass"] is True
    assert all(block["pass"] for block in summary["behaviors"].values())
    transition = summary["behaviors"]["4_useful_compute_reevaluate"]
    assert transition["candidate_count"] == 1024
    assert transition["eligible_pair_count"] == 1024
    assert transition["coverage_rate"] == pytest.approx(1.0)
    assert transition["proceed_pair_count"] == 512
    assert transition["reset_pair_count"] == 512
    assert transition["next_decision"]["positive_count"] == 512
    assert transition["next_decision"]["negative_count"] == 512
    assert summary["selected_action_calibration"]["ema"]["rmse"] == 0.0


def _fixed_protocol_inputs(total_steps, *, profile=None):
    profile_value = profile.value if isinstance(
        profile, fixed_eval.ConfirmationProfile
    ) else profile
    is_v11 = profile_value == "v11-300k"
    is_v12 = profile_value == "v12-300k"
    is_v13 = profile_value == "v13-300k"
    is_v14 = profile_value == "v14-300k"
    is_v15 = profile_value == "v15-300k"
    is_v16 = profile_value == "v16-300k"
    is_v17 = profile_value == "v17-300k"
    is_v18 = profile_value == "v18-300k"
    is_v19 = profile_value == "v19-300k"
    is_v20 = profile_value == "v20-300k"
    base_seed = (
        5
        if is_v13 or is_v14 or is_v15 or is_v16 or is_v17 or is_v18 or is_v19 or is_v20
        else 4
        if is_v12
        else (3 if is_v11 else (2 if total_steps == 300_000 else 1))
    )
    flags = SimpleNamespace(
        name="Enduro-v5",
        dynamic_voc_mode="control",
        dynamic_search=True,
        dynamic_factorized_control=True,
        voc_eval_stochastic=True,
        voc_dueling_q=True,
        voc_ema_gate_target=True,
        voc_dedicated_gate=True,
        voc_soft_q_bce_gate=True,
        envpool=True,
        total_steps=total_steps,
        self_play_n=1,
        env_n=16,
        actor_unroll_len=201,
        base_seed=base_seed,
    )
    actor_checkpoint = {
        "flags": {"total_steps": total_steps},
        "voc_ema_gate_head_state_dict": {
            "weight": torch.zeros((2, 3), dtype=torch.float32),
            "bias": torch.zeros(2, dtype=torch.float32),
        },
    }
    model_checkpoint = {"flags": {"total_steps": total_steps}}
    actor_validation = {
        "real_step": total_steps + 224,
        "voc": {"voc_ema_gate_head_state_saved": True},
    }
    model_validation = {"real_step": total_steps}
    if total_steps == 300_000:
        profile_flags = {
            "base_seed": 2,
            "schedule_total_steps": 100_000_000,
            "voc_gate_param_align": True,
            "voc_gate_param_align_coef": 1.0,
        }
        for name, value in profile_flags.items():
            setattr(flags, name, value)
        actor_checkpoint["flags"].update(profile_flags)
        actor_checkpoint["voc_gate_policy_schema_version"] = 3
        model_checkpoint["flags"].update(profile_flags)
        actor_validation["voc"].update(
            {
                "voc_gate_policy_schema_version": 3,
                "voc_gate_param_align": True,
                "voc_gate_param_align_coef": 1.0,
            }
        )
    if (
        is_v11
        or is_v12
        or is_v13
        or is_v14
        or is_v15
        or is_v16
        or is_v17
        or is_v18
        or is_v19
        or is_v20
    ):
        profile_flags = {
            "base_seed": (
                5
                if is_v13 or is_v14 or is_v15 or is_v16 or is_v17 or is_v18 or is_v19 or is_v20
                else (4 if is_v12 else 3)
            ),
            "schedule_total_steps": 100_000_000,
            "dynamic_voc_mode": "control",
            "voc_dedicated_gate": True,
            "voc_soft_q_bce_gate": True,
            "voc_gate_temperature": 1.0,
            "voc_gate_q_temperature": 0.05,
            "voc_gate_param_align": False,
            "voc_gate_param_align_coef": 1.0,
            "voc_gate_exact_projection": True,
            "ckp": False,
            "preload": "",
            "preload_actor": "",
            "voc_parent_checkpoint": "",
            **(
                {
                    "voc_eval_stochastic": True,
                    "voc_train_epsilon": 0.02,
                    "voc_gate_epsilon_greedy_execution": True,
                }
                if (
                    is_v12
                    or is_v13
                    or is_v14
                    or is_v15
                    or is_v16
                    or is_v17
                    or is_v18
                    or is_v19
                    or is_v20
                )
                else {}
            ),
            **(
                {
                    "xpid": (
                        fixed_eval.V20_PRIMARY_XPID
                        if is_v20
                        else (
                            fixed_eval.V19_PRIMARY_XPID
                            if is_v19
                            else (
                                fixed_eval.V18_PRIMARY_XPID
                                if is_v18
                                else (
                                    fixed_eval.V17_PRIMARY_XPID
                                    if is_v17
                                    else (
                                        fixed_eval.V16_PRIMARY_XPID
                                        if is_v16
                                        else (
                                            fixed_eval.V15_PRIMARY_XPID
                                            if is_v15
                                            else (
                                                fixed_eval.V14_PRIMARY_XPID
                                                if is_v14
                                                else fixed_eval.V13_PRIMARY_XPID
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    ),
                    "voc_gate_policy_schema_version": (
                        13
                        if is_v20
                        else (
                            12
                            if is_v19
                            else (
                                11
                                if is_v18
                                else (
                                    10
                                    if is_v17
                                    else (
                                        9
                                        if is_v16
                                        else (8 if is_v15 else (7 if is_v14 else 6))
                                    )
                                )
                            )
                        )
                    ),
                    "model_warm_up_n": 10_000,
                    "actor_unroll_len": 201,
                    "use_wandb": True,
                    "voc_gate_execution_epsilon": 0.25,
                    "voc_actor_policy_version_barrier": True,
                    "voc_actor_policy_bundle_schema_version": 1,
                    "voc_actor_policy_barrier_timeout_s": 120.0,
                    "voc_actor_policy_ray_max_restarts": 0,
                    "voc_actor_policy_ray_max_task_retries": 0,
                    "voc_actor_policy_barrier_runtime": True,
                    "actor_amp_init_scale": 32.0,
                    "float16": True,
                    "model_float16": False,
                    "parallel_actor": True,
                    "ppo_k": 1,
                    "self_play_n": 1,
                    "env_n": 16,
                    "actor_batch_size": 16,
                    **({"voc_gate_target_tau": 1.0} if is_v19 or is_v20 else {}),
                    **(
                        {"voc_model_input_seal_schema_version": 1}
                        if is_v14 or is_v15 or is_v16 or is_v17 or is_v18 or is_v19 or is_v20
                        else {}
                    ),
                }
                if is_v13 or is_v14 or is_v15 or is_v16 or is_v17 or is_v18 or is_v19 or is_v20
                else {}
            ),
        }
        for name, value in profile_flags.items():
            setattr(flags, name, value)
        actor_checkpoint["flags"].update(profile_flags)
        model_checkpoint["flags"].update(profile_flags)
        actor_checkpoint.update(
            {
                "dynamic_voc_mode": "control",
                "voc_gate_policy_schema_version": (
                    13
                    if is_v20
                    else (
                        12
                        if is_v19
                        else (
                            11
                            if is_v18
                            else (
                                10
                                if is_v17
                                else (
                                    9
                                    if is_v16
                                    else (
                                        8
                                        if is_v15
                                        else (
                                            7
                                            if is_v14
                                            else (6 if is_v13 else (5 if is_v12 else 4))
                                        )
                                    )
                                )
                            )
                        )
                    )
                ),
                "voc_control_origin": "fresh",
                "voc_control_origin_legacy_defaulted": False,
                "voc_parent_checkpoint_sha256": None,
                "voc_parent_checkpoint": None,
                "voc_parent_imitation_data_signature": None,
                "voc_activation_real_step": 0,
            }
        )
        ema_weight = torch.tensor(
            [[0.125, -0.25, 0.5], [-0.375, 0.5, 0.25]],
            dtype=torch.float32,
        )
        ema_bias = torch.tensor([0.125, -0.25], dtype=torch.float32)
        actor_checkpoint["voc_ema_gate_head_state_dict"] = {
            "weight": ema_weight,
            "bias": ema_bias,
        }
        scale = (
            profile_flags["voc_gate_temperature"]
            / profile_flags["voc_gate_q_temperature"]
        )
        actor_checkpoint["actor_net_state_dict"] = {
            "voc_gate_head.weight": scale
            * (ema_weight[0:1] - ema_weight[1:2]),
            "voc_gate_head.bias": scale * (ema_bias[0:1] - ema_bias[1:2]),
            **(
                {
                    "voc_head.weight": ema_weight.clone(),
                    "voc_head.bias": ema_bias.clone(),
                }
                if is_v19 or is_v20
                else {}
            ),
        }
        if is_v19 or is_v20:
            actor_checkpoint.update(
                {
                    "voc_gate_target_tau": 1.0,
                    "voc_ema_gate_update_count": 1,
                }
            )
        actor_validation["voc"].update(
            {
                "dynamic_voc_mode": "control",
                "voc_gate_policy_schema_version": (
                    13
                    if is_v20
                    else (
                        12
                        if is_v19
                        else (
                            11
                            if is_v18
                            else (
                                10
                                if is_v17
                                else (
                                    9
                                    if is_v16
                                    else (
                                        8
                                        if is_v15
                                        else (
                                            7
                                            if is_v14
                                            else (6 if is_v13 else (5 if is_v12 else 4))
                                        )
                                    )
                                )
                            )
                        )
                    )
                ),
                "voc_gate_param_align": False,
                "voc_gate_param_align_coef": 1.0,
                "voc_gate_exact_projection": True,
                **(
                    {"voc_gate_epsilon_greedy_execution": True}
                    if (
                        is_v12
                        or is_v13
                        or is_v14
                        or is_v15
                        or is_v16
                        or is_v17
                        or is_v18
                        or is_v19
                        or is_v20
                    )
                    else {}
                ),
                **(
                    {"voc_model_input_seal_schema_version": 1}
                    if is_v14 or is_v15 or is_v16 or is_v17 or is_v18 or is_v19 or is_v20
                    else {}
                ),
                "voc_control_origin": "fresh",
                "voc_control_origin_legacy_defaulted": False,
                "voc_parent_checkpoint_sha256": None,
                "voc_parent_checkpoint": None,
                "voc_parent_imitation_data_signature": None,
                "voc_activation_real_step": 0,
                **({"voc_gate_target_tau": 1.0} if is_v19 or is_v20 else {}),
            }
        )
    return (
        flags,
        actor_checkpoint,
        model_checkpoint,
        actor_validation,
        model_validation,
    )


def _v13_bundle_evidence(*, stage=None):
    state_digest = "a" * 64
    history_digest = "b" * 64
    version = 1
    history = (
        {
            "predecessor_version": -1,
            "policy_version": 0,
            "publication_count": 0,
            "terminal": False,
            "ack_ranks": [0],
            "expected_ack_count": 1,
            "state_sha256": "c" * 64,
        },
        {
            "predecessor_version": 0,
            "policy_version": version,
            "publication_count": version,
            "terminal": True,
            "ack_ranks": [0],
            "expected_ack_count": 1,
            "state_sha256": state_digest,
        },
    )
    actor_policy = {
        "voc_actor_policy_version": version,
        "voc_actor_policy_publication_count": version,
        "voc_actor_policy_terminal": True,
        "voc_actor_policy_version_mismatch_count": 0,
        "voc_actor_policy_malformed_bundle_count": 0,
        "voc_actor_policy_barrier_timeout_count": 0,
        "voc_actor_policy_terminal_ack_count": 1,
        "voc_actor_policy_expected_ack_count": 1,
        "voc_actor_policy_state_sha256": state_digest,
        "actor_amp_init_scale": 32.0,
        "actor_amp_skip_count": 0,
        "actor_amp_consecutive_skips": 0,
        "voc_actor_policy_publication_history": history,
        "voc_actor_policy_publication_history_sha256": history_digest,
    }
    logger_completion = {
        "required": True,
        "use_wandb": True,
        "ack_verified": True,
        "private_markers_cleaned": True,
        "policy_version": version,
        "state_sha256": state_digest,
        "publication_history_sha256": history_digest,
    }
    return {
        "authoritative_validator": (
            "thinker.util.validate_schema6_final_bundle"
        ),
        "resolved_identity": {
            "key_count": fixed_eval.V13_COMPLETE_IDENTITY_KEY_COUNT,
            "v12_projection_key_count": (
                fixed_eval.V13_V12_PROJECTION_KEY_COUNT
            ),
            "v12_projection_sha256": (
                fixed_eval.V13_V12_PROJECTION_SHA256
            ),
            "complete_surface_sha256": "d" * 64,
            "stage": tuple(stage or fixed_eval.V13_PRIMARY_STAGE),
            "paths": {},
        },
        "actor_policy": actor_policy,
        "actor_training_state": {},
        "model_step": 1,
        "model_real_step": 300_000,
        "model_state_tensor_count": 1,
        "model_optimizer_state": {},
        "model_scheduler_state": {},
        "model_scaler_state": {},
        "config_use_wandb": True,
        "completion_evidence": {},
        "logger_completion": logger_completion,
        "finish_marker": {},
        "private_logger_markers": {
            name: {"path": f"/checkpoint/{name}", "absent": True}
            for name in fixed_eval.V13_PRIVATE_LOGGER_MARKERS
        },
    }


def _v14_bundle_evidence(*, stage=None, drain=1):
    evidence = copy.deepcopy(_v13_bundle_evidence())
    evidence["authoritative_validator"] = (
        "thinker.util.validate_schema7_final_bundle"
    )
    resolved = evidence["resolved_identity"]
    resolved.update(
        {
            "gate_schema": 7,
            "voc_gate_policy_schema_version": 7,
            "voc_model_input_seal_schema_version": 1,
            "key_count": fixed_eval.V14_COMPLETE_IDENTITY_KEY_COUNT,
            "v12_projection_key_count": (
                fixed_eval.V14_V12_PROJECTION_KEY_COUNT
            ),
            "v12_projection_sha256": fixed_eval.V14_V12_PROJECTION_SHA256,
            "stage": tuple(stage or fixed_eval.V14_PRIMARY_STAGE),
        }
    )
    actor_policy = evidence["actor_policy"]
    actor_policy["voc_actor_policy_bundle_summary"] = {
        "bundle_schema_version": 1,
        "policy_version": actor_policy["voc_actor_policy_version"],
        "terminal": True,
        "gate_schema": 7,
        "state_sha256": actor_policy["voc_actor_policy_state_sha256"],
    }
    terminal = 300_000
    pre_real = terminal if drain == 0 else terminal - 16
    pre_count = 5
    final_count = pre_count + drain
    seal = {
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
    }
    evidence.update(
        {
            "model_real_step": terminal,
            "model_input_seal": seal,
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
            "stored_surface_identity": {
                source: copy.deepcopy(resolved)
                for source in (
                    "config",
                    "actor_checkpoint",
                    "model_checkpoint",
                )
            },
            "private_logger_markers_absent": True,
            "public_finish_verified": True,
        }
    )
    return evidence


def _v15_bundle_evidence(*, stage=None, drain=1):
    evidence = copy.deepcopy(_v14_bundle_evidence(drain=drain))
    evidence.pop("finish_marker", None)
    evidence["authoritative_validator"] = (
        "thinker.util.validate_schema8_final_bundle"
    )
    resolved = evidence["resolved_identity"]
    resolved_stage = tuple(stage or fixed_eval.V15_PRIMARY_STAGE)
    xpid = resolved_stage[0]
    savedir = "/sealed/runs"
    data_path = "/sealed/data/behavioral_data_block"
    resolved.update(
        {
            "gate_schema": 8,
            "voc_gate_policy_schema_version": 8,
            "voc_q_regression_loss": "half_squared_td",
            "stage": resolved_stage,
            "paths": {
                "savedir": savedir,
                "ckpdir": f"{savedir}/{xpid}",
                "cmd": "train.py",
                "icopro_data_path": data_path,
            },
        }
    )
    actor_policy = evidence["actor_policy"]
    actor_policy["voc_actor_policy_bundle_summary"] = {
        "bundle_schema_version": 1,
        "policy_version": actor_policy["voc_actor_policy_version"],
        "terminal": True,
        "gate_schema": 8,
        "actor_state_dict_sha256": actor_policy[
            "voc_actor_policy_state_sha256"
        ],
        "actor_state_dict_key_count": 1,
        "actor_state_dict_keys": ["weight"],
        "actor_state_dict_metadata": [
            {
                "key": "weight",
                "dtype": "torch.float32",
                "shape": [1],
                "numel": 1,
            }
        ],
    }
    canonical_history = json.dumps(
        list(actor_policy["voc_actor_policy_publication_history"]),
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    history_digest = hashlib.sha256(canonical_history).hexdigest()
    actor_policy["voc_actor_policy_publication_history_sha256"] = history_digest
    evidence["stored_surface_identity"] = {
        source: copy.deepcopy(resolved)
        for source in ("config", "actor_checkpoint", "model_checkpoint")
    }
    checkpoint_files = {
        name: {"sha256": str(index) * 64, "size": index}
        for index, name in enumerate(
            fixed_eval.REQUIRED_CHECKPOINT_FILES, start=1
        )
    }
    evidence["completion_evidence"] = {
        "checkpoint_files": checkpoint_files,
        "implementation_sources": {
            "train.py": {"sha256": "d" * 64}
        },
        "loaded_extensions": {
            "thinker/cenv.so": {"sha256": "e" * 64}
        },
    }
    evidence["logger_completion"] = {
        "schema_version": 1,
        "required": True,
        "use_wandb": True,
        "request_sha256": "f" * 64,
        "ack_verified": True,
        "private_markers_cleaned": True,
        "policy_version": actor_policy["voc_actor_policy_version"],
        "state_sha256": actor_policy["voc_actor_policy_state_sha256"],
        "publication_history_sha256": actor_policy[
            "voc_actor_policy_publication_history_sha256"
        ],
        "checkpoint_files": copy.deepcopy(checkpoint_files),
    }
    evidence["private_logger_markers"] = {
        name: {
            "path": f"{resolved['paths']['ckpdir']}/{name}",
            "absent": True,
        }
        for name in fixed_eval.V15_PRIVATE_LOGGER_MARKERS
    }
    return evidence


def _v16_bundle_evidence(*, stage=None, drain=1):
    evidence = copy.deepcopy(_v15_bundle_evidence(drain=drain))
    evidence["authoritative_validator"] = (
        "thinker.util.validate_schema9_final_bundle"
    )
    resolved = evidence["resolved_identity"]
    resolved_stage = tuple(stage or fixed_eval.V16_PRIMARY_STAGE)
    resolved.update(
        {
            "gate_schema": 9,
            "voc_gate_policy_schema_version": 9,
            "voc_q_reconstruction": fixed_eval.V16_Q_RECONSTRUCTION,
            "stage": resolved_stage,
            "paths": {
                **resolved["paths"],
                "ckpdir": f"{resolved['paths']['savedir']}/{resolved_stage[0]}",
            },
        }
    )
    evidence["actor_policy"]["voc_actor_policy_bundle_summary"]["gate_schema"] = 9
    evidence["stored_surface_identity"] = {
        source: copy.deepcopy(resolved)
        for source in ("config", "actor_checkpoint", "model_checkpoint")
    }
    evidence["private_logger_markers"] = {
        name: {
            "path": f"{resolved['paths']['ckpdir']}/{name}",
            "absent": True,
        }
        for name in fixed_eval.V16_PRIVATE_LOGGER_MARKERS
    }
    return evidence


def _v17_bundle_evidence(*, stage=None, drain=1):
    evidence = copy.deepcopy(_v16_bundle_evidence(drain=drain))
    evidence["authoritative_validator"] = (
        "thinker.util.validate_schema10_final_bundle"
    )
    resolved = evidence["resolved_identity"]
    resolved_stage = tuple(stage or fixed_eval.V17_PRIMARY_STAGE)
    resolved.update(
        {
            "gate_schema": 10,
            "voc_gate_policy_schema_version": 10,
            "voc_q_regression_loss": fixed_eval.V17_Q_REGRESSION_LOSS,
            "voc_q_reconstruction": fixed_eval.V17_Q_RECONSTRUCTION,
            "stage": resolved_stage,
            "paths": {
                **resolved["paths"],
                "ckpdir": f"{resolved['paths']['savedir']}/{resolved_stage[0]}",
            },
        }
    )
    evidence["actor_policy"]["voc_actor_policy_bundle_summary"][
        "gate_schema"
    ] = 10
    evidence["stored_surface_identity"] = {
        source: copy.deepcopy(resolved)
        for source in ("config", "actor_checkpoint", "model_checkpoint")
    }
    evidence["private_logger_markers"] = {
        name: {
            "path": f"{resolved['paths']['ckpdir']}/{name}",
            "absent": True,
        }
        for name in fixed_eval.V17_PRIVATE_LOGGER_MARKERS
    }
    return evidence


def _v18_bundle_evidence(*, stage=None, drain=1):
    evidence = copy.deepcopy(_v17_bundle_evidence(drain=drain))
    evidence["authoritative_validator"] = (
        "thinker.util.validate_schema11_final_bundle"
    )
    resolved = evidence["resolved_identity"]
    resolved_stage = tuple(stage or fixed_eval.V18_PRIMARY_STAGE)
    resolved.update(
        {
            "gate_schema": 11,
            "voc_gate_policy_schema_version": 11,
            "voc_q_optimizer_coordinates": (
                fixed_eval.V18_Q_OPTIMIZER_COORDINATES
            ),
            "stage": resolved_stage,
            "paths": {
                **resolved["paths"],
                "ckpdir": f"{resolved['paths']['savedir']}/{resolved_stage[0]}",
            },
        }
    )
    evidence["actor_policy"]["voc_actor_policy_bundle_summary"][
        "gate_schema"
    ] = 11
    actor_policy = evidence["actor_policy"]
    actor_policy.update(
        {
            "actor_amp_scale": 32.0,
            "actor_amp_growth_tracker": 2,
            "voc_actor_policy_publication_event_count": len(
                actor_policy["voc_actor_policy_publication_history"]
            ),
            "voc_actor_policy_final_publication_event": copy.deepcopy(
                actor_policy["voc_actor_policy_publication_history"][-1]
            ),
        }
    )
    evidence["stored_surface_identity"] = {
        source: copy.deepcopy(resolved)
        for source in ("config", "actor_checkpoint", "model_checkpoint")
    }
    evidence["private_logger_markers"] = {
        name: {
            "path": f"{resolved['paths']['ckpdir']}/{name}",
            "absent": True,
        }
        for name in fixed_eval.V18_PRIVATE_LOGGER_MARKERS
    }
    return evidence


def _v19_bundle_evidence(*, stage=None, drain=1):
    evidence = copy.deepcopy(_v18_bundle_evidence(drain=drain))
    evidence["authoritative_validator"] = (
        "thinker.util.validate_schema12_final_bundle"
    )
    resolved = evidence["resolved_identity"]
    resolved_stage = tuple(stage or fixed_eval.V19_PRIMARY_STAGE)
    resolved.update(
        {
            "gate_schema": 12,
            "voc_gate_policy_schema_version": 12,
            "v12_projection_sha256": fixed_eval.V19_V12_PROJECTION_SHA256,
            "stage": resolved_stage,
            "paths": {
                **resolved["paths"],
                "ckpdir": f"{resolved['paths']['savedir']}/{resolved_stage[0]}",
            },
        }
    )
    evidence["actor_policy"]["voc_actor_policy_bundle_summary"][
        "gate_schema"
    ] = 12
    evidence["stored_surface_identity"] = {
        source: copy.deepcopy(resolved)
        for source in ("config", "actor_checkpoint", "model_checkpoint")
    }
    evidence["private_logger_markers"] = {
        name: {
            "path": f"{resolved['paths']['ckpdir']}/{name}",
            "absent": True,
        }
        for name in fixed_eval.V19_PRIVATE_LOGGER_MARKERS
    }
    return evidence


def _v20_bundle_evidence(*, stage=None, drain=1):
    evidence = copy.deepcopy(_v19_bundle_evidence(drain=drain))
    evidence["authoritative_validator"] = (
        "thinker.util.validate_schema13_final_bundle"
    )
    resolved = evidence["resolved_identity"]
    resolved_stage = tuple(stage or fixed_eval.V20_PRIMARY_STAGE)
    resolved.update(
        {
            "gate_schema": 13,
            "voc_gate_policy_schema_version": 13,
            "stage": resolved_stage,
            "paths": {
                **resolved["paths"],
                "ckpdir": f"{resolved['paths']['savedir']}/{resolved_stage[0]}",
            },
        }
    )
    actor_policy = evidence["actor_policy"]
    actor_policy["voc_actor_policy_bundle_summary"]["gate_schema"] = 13
    evidence["actor_training_state"]["voc_update_count"] = actor_policy[
        "voc_actor_policy_version"
    ]
    evidence["stored_surface_identity"] = {
        source: copy.deepcopy(resolved)
        for source in ("config", "actor_checkpoint", "model_checkpoint")
    }
    manifest_record = {"sha256": "9" * 64, "size": 4096}
    evidence["completion_evidence"]["checkpoint_files"][
        "voc_telemetry_manifest.json"
    ] = manifest_record
    evidence["logger_completion"]["schema_version"] = 2
    evidence["logger_completion"]["checkpoint_files"][
        "voc_telemetry_manifest.json"
    ] = manifest_record
    evidence["telemetry"] = {
        "telemetry_schema_version": 1,
        "gate_schema": 13,
        "manifest_name": "voc_telemetry_manifest.json",
        "manifest_sha256": manifest_record["sha256"],
        "manifest_size": manifest_record["size"],
        "transaction_count": evidence["actor_training_state"][
            "voc_update_count"
        ],
        "terminal_policy_version": actor_policy["voc_actor_policy_version"],
        "terminal_real_step": 300_224,
        "actor_state_sha256": actor_policy["voc_actor_policy_state_sha256"],
        "publication_history_sha256": actor_policy[
            "voc_actor_policy_publication_history_sha256"
        ],
    }
    evidence["private_logger_markers"] = {
        name: {
            "path": f"{resolved['paths']['ckpdir']}/{name}",
            "absent": True,
        }
        for name in fixed_eval.V20_PRIVATE_LOGGER_MARKERS
    }
    return evidence


def _retarget_v15_bundle_evidence(evidence, checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir).resolve()
    savedir = checkpoint_dir.parent
    data_path = savedir.parent / "data" / "behavioral_data_block"
    paths = {
        "savedir": str(savedir),
        "ckpdir": str(checkpoint_dir),
        "cmd": "train.py",
        "icopro_data_path": str(data_path),
    }
    evidence["resolved_identity"]["paths"] = copy.deepcopy(paths)
    for surface in evidence["stored_surface_identity"].values():
        surface["paths"] = copy.deepcopy(paths)
    if "private_logger_markers" in evidence:
        evidence["private_logger_markers"] = {
            name: {
                "path": str(checkpoint_dir / name),
                "absent": True,
            }
            for name in fixed_eval.V15_PRIVATE_LOGGER_MARKERS
        }
    return evidence


def _exact_protocol_call(profile, total_steps, *, diagnostic=False):
    inputs = _fixed_protocol_inputs(total_steps, profile=profile)
    return fixed_eval._require_fixed_protocol(
        *inputs,
        confirmation_profile=profile,
        seeds=range(
            fixed_eval.DEFAULT_SEED_BASE,
            fixed_eval.DEFAULT_SEED_BASE + fixed_eval.DEFAULT_NUM_SEEDS,
        ),
        real_steps_per_seed=fixed_eval.DEFAULT_REAL_STEPS_PER_SEED,
        calibration_unroll=fixed_eval.DEFAULT_CALIBRATION_UNROLL,
        diagnostic=diagnostic,
        schema6_bundle_validation=(
            _v13_bundle_evidence() if profile == "v13-300k" else None
        ),
        schema7_bundle_validation=(
            _v14_bundle_evidence() if profile == "v14-300k" else None
        ),
        schema8_bundle_validation=(
            _v15_bundle_evidence() if profile == "v15-300k" else None
        ),
        schema9_bundle_validation=(
            _v16_bundle_evidence() if profile == "v16-300k" else None
        ),
        schema10_bundle_validation=(
            _v17_bundle_evidence() if profile == "v17-300k" else None
        ),
        schema11_bundle_validation=(
            _v18_bundle_evidence() if profile == "v18-300k" else None
        ),
        schema12_bundle_validation=(
            _v19_bundle_evidence() if profile == "v19-300k" else None
        ),
        schema13_bundle_validation=(
            _v20_bundle_evidence() if profile == "v20-300k" else None
        ),
    )


@pytest.mark.parametrize("drain", [0, 1])
def test_v14_bundle_evidence_accepts_both_frozen_terminal_branches(drain):
    evidence = _v14_bundle_evidence(drain=drain)

    validated = fixed_eval._require_v14_bundle_evidence(evidence)

    assert validated["resolved_identity"]["key_count"] == 229
    assert validated["model_input_seal"][
        "voc_model_terminal_drain_update_count"
    ] == drain
    assert validated["private_logger_markers_absent"] is True
    json.dumps(validated, sort_keys=True, allow_nan=False)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("voc_model_input_sealed", 1, "not sealed"),
        ("voc_model_input_seal_count", True, "must be Python int"),
        ("voc_model_input_late_write_count", 1, "requires.*=0"),
        ("voc_model_input_abort_count", 1, "requires.*=0"),
        ("voc_model_terminal_drain_update_count", 2, "drain evidence"),
    ],
)
def test_v14_bundle_evidence_rejects_model_seal_mutations(
    field, value, error
):
    evidence = _v14_bundle_evidence()
    evidence["model_input_seal"][field] = value

    with pytest.raises(ValueError, match=error):
        fixed_eval._require_v14_bundle_evidence(evidence)


@pytest.mark.parametrize("drain", [0, 1])
def test_v15_bundle_evidence_accepts_exact_schema8_terminal_branches(drain):
    evidence = _v15_bundle_evidence(drain=drain)

    validated = fixed_eval._require_v15_bundle_evidence(evidence)

    assert validated["resolved_identity"]["gate_schema"] == 8
    assert validated["resolved_identity"]["voc_q_regression_loss"] == (
        "half_squared_td"
    )
    assert validated["model_input_seal"][
        "voc_model_terminal_drain_update_count"
    ] == drain
    assert validated["actor_policy"]["voc_actor_policy_bundle_summary"][
        "actor_state_dict_sha256"
    ] == validated["actor_policy"]["voc_actor_policy_state_sha256"]
    json.dumps(validated, sort_keys=True, allow_nan=False)


@pytest.mark.parametrize("drain", [0, 1])
def test_v16_bundle_evidence_accepts_exact_schema9_terminal_branches(drain):
    evidence = _v16_bundle_evidence(drain=drain)

    validated = fixed_eval._require_v16_bundle_evidence(evidence)

    assert set(validated) == set(_v15_bundle_evidence())
    assert validated["resolved_identity"]["gate_schema"] == 9
    assert validated["resolved_identity"]["voc_q_regression_loss"] == (
        "half_squared_td"
    )
    assert validated["resolved_identity"]["voc_q_reconstruction"] == (
        fixed_eval.V16_Q_RECONSTRUCTION
    )
    assert "voc_q_reconstruction" not in (
        _v15_bundle_evidence()["resolved_identity"]
    )
    assert validated["model_input_seal"][
        "voc_model_terminal_drain_update_count"
    ] == drain
    json.dumps(validated, sort_keys=True, allow_nan=False)


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
def test_v18_bundle_rejects_lifecycle_actor_identity_additions(field, value):
    evidence = _v18_bundle_evidence()
    evidence["actor_policy"][field] = value

    with pytest.raises(ValueError, match="exact schema-10 lifecycle keyset"):
        fixed_eval._require_v18_bundle_evidence(evidence)


def test_v18_bundle_rejects_missing_lifecycle_actor_field():
    evidence = _v18_bundle_evidence()
    evidence["actor_policy"].pop("actor_amp_scale")

    with pytest.raises(ValueError, match="exact schema-10 lifecycle keyset"):
        fixed_eval._require_v18_bundle_evidence(evidence)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("voc_q_regression_loss", None),
        ("voc_q_regression_loss", "smooth_l1"),
        ("voc_q_reconstruction", None),
        ("voc_q_reconstruction", "policy_centered_raw_head"),
    ],
)
def test_v16_bundle_evidence_rejects_derived_identity_drift(field, value):
    evidence = _v16_bundle_evidence()
    if value is None:
        evidence["resolved_identity"].pop(field)
    else:
        evidence["resolved_identity"][field] = value

    with pytest.raises(ValueError, match="resolved|regression|reconstruction"):
        fixed_eval._require_v16_bundle_evidence(evidence)


@pytest.mark.parametrize("drain", [0, 1])
def test_v17_bundle_evidence_accepts_exact_schema10_terminal_branches(drain):
    evidence = _v17_bundle_evidence(drain=drain)

    validated = fixed_eval._require_v17_bundle_evidence(evidence)

    assert set(validated) == set(_v16_bundle_evidence())
    assert set(validated["resolved_identity"]) == set(
        _v16_bundle_evidence()["resolved_identity"]
    )
    assert validated["resolved_identity"]["gate_schema"] == 10
    assert validated["resolved_identity"]["voc_q_regression_loss"] == (
        "smooth_l1_beta1"
    )
    assert validated["resolved_identity"]["voc_q_reconstruction"] == (
        fixed_eval.V16_Q_RECONSTRUCTION
    )
    assert validated["model_input_seal"][
        "voc_model_terminal_drain_update_count"
    ] == drain
    json.dumps(validated, sort_keys=True, allow_nan=False)


@pytest.mark.parametrize(
    ("target", "mutation"),
    [
        ("resolved_loss", "wrong"),
        ("resolved_reconstruction", "missing"),
        ("stored_surface", "wrong"),
        ("actor_metadata", "extra"),
        ("history", "wrong"),
        ("top_level", "extra"),
    ],
)
def test_v17_bundle_evidence_rejects_forged_schema10_evidence(
    target, mutation
):
    evidence = _v17_bundle_evidence()
    if target == "resolved_loss":
        evidence["resolved_identity"]["voc_q_regression_loss"] = (
            "half_squared_td"
        )
    elif target == "resolved_reconstruction":
        evidence["resolved_identity"].pop("voc_q_reconstruction")
    elif target == "stored_surface":
        evidence["stored_surface_identity"]["actor_checkpoint"][
            "gate_schema"
        ] = 9
    elif target == "actor_metadata":
        evidence["actor_policy"]["voc_actor_policy_bundle_summary"][
            "actor_state_dict_metadata"
        ][0]["forged"] = True
    elif target == "history":
        evidence["actor_policy"]["voc_actor_policy_publication_history"][0][
            "terminal"
        ] = True
    else:
        evidence["forged_authority"] = "private"

    with pytest.raises(ValueError):
        fixed_eval._require_v17_bundle_evidence(evidence)


@pytest.mark.parametrize("drain", [0, 1])
def test_v18_bundle_evidence_adds_only_optimizer_identity(drain):
    evidence = _v18_bundle_evidence(drain=drain)

    validated = fixed_eval._require_v18_bundle_evidence(evidence)

    assert set(validated) == set(_v17_bundle_evidence())
    assert set(validated["resolved_identity"]) == (
        set(_v17_bundle_evidence()["resolved_identity"])
        | {"voc_q_optimizer_coordinates"}
    )
    assert validated["resolved_identity"]["gate_schema"] == 11
    assert validated["resolved_identity"][
        "voc_q_optimizer_coordinates"
    ] == "orthonormal_common_difference_adam"
    assert validated["model_input_seal"][
        "voc_model_terminal_drain_update_count"
    ] == drain
    assert set(validated["actor_policy"]) == (
        fixed_eval.ATOMIC_ACTOR_POLICY_EVIDENCE_FIELDS
    )
    assert not {
        "voc_q_regression_loss",
        "voc_q_reconstruction",
        "voc_q_optimizer_coordinates",
    } & set(validated["actor_policy"])
    json.dumps(validated, sort_keys=True, allow_nan=False)


@pytest.mark.parametrize("mutation", ["missing", "wrong", "extra"])
def test_v18_bundle_evidence_rejects_forged_optimizer_identity(mutation):
    evidence = _v18_bundle_evidence()
    if mutation == "missing":
        evidence["resolved_identity"].pop("voc_q_optimizer_coordinates")
    elif mutation == "wrong":
        evidence["resolved_identity"]["voc_q_optimizer_coordinates"] = (
            "raw_continue_stop_adam"
        )
    else:
        evidence["resolved_identity"]["forged_optimizer_state"] = True

    with pytest.raises(ValueError, match="resolved|optimizer|shape"):
        fixed_eval._require_v18_bundle_evidence(evidence)


@pytest.mark.parametrize(
    "missing",
    [
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
        "private_logger_markers",
    ],
)
def test_v15_bundle_evidence_rejects_missing_top_level_field(missing):
    evidence = _v15_bundle_evidence()
    evidence.pop(missing)

    with pytest.raises(ValueError, match="exact top-level shape"):
        fixed_eval._require_v15_bundle_evidence(evidence)


def test_v15_bundle_evidence_rejects_extra_top_level_field():
    evidence = _v15_bundle_evidence()
    evidence["unexpected"] = 0

    with pytest.raises(ValueError, match="exact top-level shape"):
        fixed_eval._require_v15_bundle_evidence(evidence)


@pytest.mark.parametrize(
    "mutation", ["missing", "extra", "wrong_digest", "wrong_loss", "v14_stage"]
)
def test_v15_bundle_evidence_rejects_identity_and_actor_summary_attacks(
    mutation,
):
    evidence = _v15_bundle_evidence()
    summary = evidence["actor_policy"]["voc_actor_policy_bundle_summary"]
    if mutation == "missing":
        summary.pop("actor_state_dict_metadata")
    elif mutation == "extra":
        summary["state_sha256"] = "a" * 64
    elif mutation == "wrong_digest":
        summary["actor_state_dict_sha256"] = "9" * 64
    elif mutation == "wrong_loss":
        evidence["resolved_identity"]["voc_q_regression_loss"] = "smooth_l1"
        for surface in evidence["stored_surface_identity"].values():
            surface["voc_q_regression_loss"] = "smooth_l1"
    else:
        evidence["resolved_identity"]["stage"] = fixed_eval.V14_PRIMARY_STAGE
        for surface in evidence["stored_surface_identity"].values():
            surface["stage"] = fixed_eval.V14_PRIMARY_STAGE

    with pytest.raises(ValueError):
        fixed_eval._require_v15_bundle_evidence(evidence)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("bundle_schema_version", True),
        ("policy_version", 1.0),
        ("gate_schema", 8.0),
        ("actor_state_dict_sha256", 8),
    ],
)
def test_v15_bundle_evidence_rejects_actor_summary_type_drift(field, value):
    evidence = _v15_bundle_evidence()
    evidence["actor_policy"]["voc_actor_policy_bundle_summary"][field] = value

    with pytest.raises(ValueError, match="actor bundle identity"):
        fixed_eval._require_v15_bundle_evidence(evidence)


@pytest.mark.parametrize(
    "mutation",
    [
        "none",
        "key",
        "shape_bool",
        "numel_float",
        "numel_product",
    ],
)
def test_v15_bundle_evidence_rejects_actor_metadata_attacks(mutation):
    evidence = _v15_bundle_evidence()
    summary = evidence["actor_policy"]["voc_actor_policy_bundle_summary"]
    metadata = summary["actor_state_dict_metadata"]
    if mutation == "none":
        metadata[0] = None
    elif mutation == "key":
        metadata[0]["key"] = "other"
    elif mutation == "shape_bool":
        metadata[0]["shape"] = [True]
    elif mutation == "numel_float":
        metadata[0]["numel"] = 1.0
    else:
        metadata[0]["numel"] = 2

    with pytest.raises(ValueError, match="metadata"):
        fixed_eval._require_v15_bundle_evidence(evidence)


@pytest.mark.parametrize(
    ("kind", "value"),
    [
        ("seed", 5.0),
        ("steps", 300_000.0),
        ("wandb", 1),
        ("path_none", None),
        ("path_relative", "relative/runs"),
        ("path_ckp_basename", "/sealed/runs/wrong-xpid"),
        ("cmd_empty", ""),
        ("data_wrong", "/sealed/data"),
        ("ack_bool", False),
        ("ack_float", 0.0),
        ("history_digest", "not-a-digest"),
    ],
)
def test_v15_bundle_evidence_rejects_stage_path_and_history_types(kind, value):
    evidence = _v15_bundle_evidence()
    resolved = evidence["resolved_identity"]
    if kind in {"seed", "steps", "wandb"}:
        stage = list(resolved["stage"])
        index = {"seed": 1, "steps": 2, "wandb": 5}[kind]
        stage[index] = value
        resolved["stage"] = tuple(stage)
    elif kind == "path_none":
        resolved["paths"]["savedir"] = value
    elif kind == "path_relative":
        resolved["paths"]["savedir"] = value
    elif kind == "path_ckp_basename":
        resolved["paths"]["ckpdir"] = value
    elif kind == "cmd_empty":
        resolved["paths"]["cmd"] = value
    elif kind == "data_wrong":
        resolved["paths"]["icopro_data_path"] = value
    elif kind in {"ack_bool", "ack_float"}:
        resolved_event = evidence["actor_policy"][
            "voc_actor_policy_publication_history"
        ][0]
        resolved_event["ack_ranks"] = [value]
    else:
        evidence["actor_policy"]["voc_actor_policy_publication_history"][0][
            "state_sha256"
        ] = value
    evidence["stored_surface_identity"] = {
        name: copy.deepcopy(resolved)
        for name in ("config", "actor_checkpoint", "model_checkpoint")
    }

    with pytest.raises(ValueError, match="stage|path|history"):
        fixed_eval._require_v15_bundle_evidence(evidence)


def test_v15_bundle_evidence_rejects_stale_history_digest_after_event_drift():
    evidence = _v15_bundle_evidence()
    evidence["actor_policy"]["voc_actor_policy_publication_history"][0][
        "state_sha256"
    ] = "9" * 64

    with pytest.raises(ValueError, match="history digest"):
        fixed_eval._require_v15_bundle_evidence(evidence)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_checkpoint_files",
        "extra_logger_field",
        "schema_bool",
        "size_float",
        "size_bool",
        "checkpoint_digest_drift",
        "request_digest_type",
    ],
)
def test_v15_bundle_evidence_rejects_logger_completion_attacks(mutation):
    evidence = _v15_bundle_evidence()
    logger = evidence["logger_completion"]
    first_name = fixed_eval.REQUIRED_CHECKPOINT_FILES[0]
    if mutation == "missing_checkpoint_files":
        logger.pop("checkpoint_files")
    elif mutation == "extra_logger_field":
        logger["unexpected"] = 0
    elif mutation == "schema_bool":
        logger["schema_version"] = True
    elif mutation == "size_float":
        logger["checkpoint_files"][first_name]["size"] = 1.0
    elif mutation == "size_bool":
        logger["checkpoint_files"][first_name]["size"] = True
    elif mutation == "checkpoint_digest_drift":
        logger["checkpoint_files"][first_name]["sha256"] = "0" * 64
    else:
        logger["request_sha256"] = 1

    with pytest.raises(ValueError, match="logger"):
        fixed_eval._require_v15_bundle_evidence(evidence)


@pytest.mark.parametrize(
    "mutation", ["wrong_path", "path_type", "extra", "absent_type"]
)
def test_v15_bundle_evidence_rejects_private_marker_record_attacks(mutation):
    evidence = _v15_bundle_evidence()
    name = fixed_eval.V15_PRIVATE_LOGGER_MARKERS[0]
    record = evidence["private_logger_markers"][name]
    if mutation == "wrong_path":
        record["path"] = f"/wrong/{name}"
    elif mutation == "path_type":
        record["path"] = False
    elif mutation == "extra":
        record["unexpected"] = 0
    else:
        record["absent"] = 1

    with pytest.raises(ValueError, match="private logger marker"):
        fixed_eval._require_v15_bundle_evidence(evidence)


def test_validate_v15_final_bundle_uses_dedicated_public_schema8_route(
    tmp_path,
):
    checkpoint_dir = tmp_path / "runs" / fixed_eval.V15_PRIMARY_XPID
    checkpoint_dir.mkdir(parents=True)
    public_record = _retarget_v15_bundle_evidence(
        _v15_bundle_evidence(), checkpoint_dir
    )
    public_record.pop("private_logger_markers")

    class CheckpointEval:
        @staticmethod
        def validate_schema8_completed_bundle(path, *, completion_state):
            assert Path(path) == checkpoint_dir.resolve()
            assert completion_state == {"status": "complete"}
            return copy.deepcopy(public_record)

    validated = fixed_eval.validate_v15_final_bundle(
        checkpoint_dir,
        {"status": "complete"},
        checkpoint_eval=CheckpointEval,
    )

    assert validated["authoritative_validator"] == (
        "thinker.util.validate_schema8_final_bundle"
    )
    assert set(validated["private_logger_markers"]) == set(
        fixed_eval.V15_PRIVATE_LOGGER_MARKERS
    )


def test_validate_v15_final_bundle_rechecks_dispatched_public_evidence(tmp_path):
    checkpoint_dir = tmp_path / "runs" / fixed_eval.V15_PRIMARY_XPID
    checkpoint_dir.mkdir(parents=True)
    public_record = _retarget_v15_bundle_evidence(
        _v15_bundle_evidence(), checkpoint_dir
    )
    public_record.pop("private_logger_markers")

    class CheckpointEval:
        @staticmethod
        def validate_schema8_completed_bundle(*args, **kwargs):
            return copy.deepcopy(public_record)

    validated = fixed_eval.validate_v15_final_bundle(
        checkpoint_dir,
        {"status": "complete"},
        checkpoint_eval=CheckpointEval,
        completed_validation=public_record,
    )

    assert validated["resolved_identity"]["voc_q_regression_loss"] == (
        "half_squared_td"
    )


def test_validate_v15_final_bundle_rejects_forged_dispatched_evidence(tmp_path):
    checkpoint_dir = tmp_path / "run"
    checkpoint_dir.mkdir()
    public_record = _v15_bundle_evidence()
    public_record.pop("private_logger_markers")
    forged = copy.deepcopy(public_record)
    forged["resolved_identity"]["voc_q_regression_loss"] = "smooth_l1"

    class CheckpointEval:
        @staticmethod
        def validate_schema8_completed_bundle(*args, **kwargs):
            return copy.deepcopy(public_record)

    with pytest.raises(RuntimeError, match="dispatched and dedicated"):
        fixed_eval.validate_v15_final_bundle(
            checkpoint_dir,
            {"status": "complete"},
            checkpoint_eval=CheckpointEval,
            completed_validation=forged,
        )


@pytest.mark.parametrize("private_name", fixed_eval.V15_PRIVATE_LOGGER_MARKERS)
def test_validate_v15_final_bundle_rejects_private_markers(
    tmp_path, private_name
):
    checkpoint_dir = tmp_path / "run"
    checkpoint_dir.mkdir()
    (checkpoint_dir / private_name).write_text("forensic\n", encoding="utf-8")
    public_record = _v15_bundle_evidence()
    public_record.pop("private_logger_markers")

    class CheckpointEval:
        @staticmethod
        def validate_schema8_completed_bundle(path, *, completion_state):
            return copy.deepcopy(public_record)

    with pytest.raises(RuntimeError, match="private logger marker"):
        fixed_eval.validate_v15_final_bundle(
            checkpoint_dir,
            {"status": "complete"},
            checkpoint_eval=CheckpointEval,
        )


def test_validate_v16_final_bundle_uses_bound_dedicated_public_schema9_route(
    tmp_path,
):
    checkpoint_dir = tmp_path / "runs" / fixed_eval.V16_PRIMARY_XPID
    checkpoint_dir.mkdir(parents=True)
    public_record = _retarget_v15_bundle_evidence(
        _v16_bundle_evidence(), checkpoint_dir
    )
    public_record.pop("private_logger_markers")
    config_payload = b"voc_gate_policy_schema_version: 9\n"
    config_digest = hashlib.sha256(config_payload).hexdigest()

    class CheckpointEval:
        @staticmethod
        def validate_schema9_completed_bundle(
            path,
            *,
            completion_state,
            config_payload: bytes,
            expected_config_sha256: str,
        ):
            assert Path(path) == checkpoint_dir.resolve()
            assert completion_state == {"status": "complete"}
            assert config_payload == b"voc_gate_policy_schema_version: 9\n"
            assert expected_config_sha256 == config_digest
            return copy.deepcopy(public_record)

    validated = fixed_eval.validate_v16_final_bundle(
        checkpoint_dir,
        {"status": "complete"},
        checkpoint_eval=CheckpointEval,
        completed_validation=public_record,
        config_payload=config_payload,
        expected_config_sha256=config_digest,
    )

    assert validated["authoritative_validator"] == (
        "thinker.util.validate_schema9_final_bundle"
    )
    assert validated["resolved_identity"]["voc_q_reconstruction"] == (
        fixed_eval.V16_Q_RECONSTRUCTION
    )
    assert set(validated["private_logger_markers"]) == set(
        fixed_eval.V16_PRIVATE_LOGGER_MARKERS
    )


def test_validate_v17_final_bundle_uses_bound_dedicated_public_schema10_route(
    tmp_path,
):
    checkpoint_dir = tmp_path / "runs" / fixed_eval.V17_PRIMARY_XPID
    checkpoint_dir.mkdir(parents=True)
    public_record = _retarget_v15_bundle_evidence(
        _v17_bundle_evidence(), checkpoint_dir
    )
    public_record.pop("private_logger_markers")
    config_payload = b"voc_gate_policy_schema_version: 10\n"
    config_digest = hashlib.sha256(config_payload).hexdigest()

    class CheckpointEval:
        @staticmethod
        def validate_schema10_completed_bundle(
            path,
            *,
            completion_state,
            config_payload: bytes,
            expected_config_sha256: str,
        ):
            assert Path(path) == checkpoint_dir.resolve()
            assert completion_state == {"status": "complete"}
            assert config_payload == b"voc_gate_policy_schema_version: 10\n"
            assert expected_config_sha256 == config_digest
            return copy.deepcopy(public_record)

    validated = fixed_eval.validate_v17_final_bundle(
        checkpoint_dir,
        {"status": "complete"},
        checkpoint_eval=CheckpointEval,
        completed_validation=public_record,
        config_payload=config_payload,
        expected_config_sha256=config_digest,
    )

    assert validated["authoritative_validator"] == (
        "thinker.util.validate_schema10_final_bundle"
    )
    assert validated["resolved_identity"]["voc_q_regression_loss"] == (
        "smooth_l1_beta1"
    )
    assert validated["resolved_identity"]["voc_q_reconstruction"] == (
        fixed_eval.V16_Q_RECONSTRUCTION
    )


def test_validate_v17_final_bundle_rejects_forged_dispatched_evidence(
    tmp_path,
):
    checkpoint_dir = tmp_path / "runs" / fixed_eval.V17_PRIMARY_XPID
    checkpoint_dir.mkdir(parents=True)
    public_record = _retarget_v15_bundle_evidence(
        _v17_bundle_evidence(), checkpoint_dir
    )
    public_record.pop("private_logger_markers")
    forged = copy.deepcopy(public_record)
    forged["resolved_identity"]["voc_q_regression_loss"] = "half_squared_td"
    payload = b"voc_gate_policy_schema_version: 10\n"

    class CheckpointEval:
        @staticmethod
        def validate_schema10_completed_bundle(*args, **kwargs):
            return copy.deepcopy(public_record)

    with pytest.raises(RuntimeError, match="dispatched and dedicated"):
        fixed_eval.validate_v17_final_bundle(
            checkpoint_dir,
            {"status": "complete"},
            checkpoint_eval=CheckpointEval,
            completed_validation=forged,
            config_payload=payload,
            expected_config_sha256=hashlib.sha256(payload).hexdigest(),
        )


def test_validate_v18_final_bundle_uses_bound_dedicated_public_schema11_route(
    tmp_path,
):
    checkpoint_dir = tmp_path / "runs" / fixed_eval.V18_PRIMARY_XPID
    checkpoint_dir.mkdir(parents=True)
    public_record = _retarget_v15_bundle_evidence(
        _v18_bundle_evidence(), checkpoint_dir
    )
    public_record.pop("private_logger_markers")
    config_payload = b"voc_gate_policy_schema_version: 11\n"
    config_digest = hashlib.sha256(config_payload).hexdigest()

    class CheckpointEval:
        @staticmethod
        def validate_schema11_completed_bundle(
            path,
            *,
            completion_state,
            config_payload: bytes,
            expected_config_sha256: str,
        ):
            assert Path(path) == checkpoint_dir.resolve()
            assert completion_state == {"status": "complete"}
            assert config_payload == b"voc_gate_policy_schema_version: 11\n"
            assert expected_config_sha256 == config_digest
            return copy.deepcopy(public_record)

    validated = fixed_eval.validate_v18_final_bundle(
        checkpoint_dir,
        {"status": "complete"},
        checkpoint_eval=CheckpointEval,
        completed_validation=public_record,
        config_payload=config_payload,
        expected_config_sha256=config_digest,
    )

    assert validated["authoritative_validator"] == (
        "thinker.util.validate_schema11_final_bundle"
    )
    assert validated["resolved_identity"][
        "voc_q_optimizer_coordinates"
    ] == fixed_eval.V18_Q_OPTIMIZER_COORDINATES


def test_validate_v18_final_bundle_fails_closed_without_public_route(tmp_path):
    checkpoint_dir = tmp_path / "runs" / fixed_eval.V18_PRIMARY_XPID
    checkpoint_dir.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="lacks schema-11 completed route"):
        fixed_eval.validate_v18_final_bundle(
            checkpoint_dir,
            {"status": "complete"},
            checkpoint_eval=SimpleNamespace(),
            config_payload=b"voc_gate_policy_schema_version: 11\n",
            expected_config_sha256="a" * 64,
        )


def test_validate_v19_final_bundle_uses_dedicated_route_and_fixed_equality(
    tmp_path,
):
    checkpoint_dir = tmp_path / "runs" / fixed_eval.V19_PRIMARY_XPID
    checkpoint_dir.mkdir(parents=True)
    public_record = _retarget_v15_bundle_evidence(
        _v19_bundle_evidence(), checkpoint_dir
    )
    public_record.pop("private_logger_markers")
    online_weight = torch.tensor([[1.0, -0.0]], dtype=torch.float32)
    online_bias = torch.tensor([0.5, -0.0], dtype=torch.float32)
    actor_checkpoint = {
        "voc_gate_target_tau": 1.0,
        "voc_ema_gate_update_count": 1,
        "voc_ema_gate_head_state_dict": {
            "weight": online_weight.clone(),
            "bias": online_bias.clone(),
        },
        "actor_net_state_dict": {
            "voc_head.weight": online_weight,
            "voc_head.bias": online_bias,
        },
    }
    actor_path = checkpoint_dir / "ckp_actor.tar"
    torch.save(actor_checkpoint, actor_path)
    marker = {
        "status": "complete",
        "checkpoint_files": {
            "ckp_actor.tar": {
                "sha256": _sha(actor_path),
                "size": actor_path.stat().st_size,
            }
        },
    }
    config_payload = b"voc_gate_policy_schema_version: 12\nvoc_gate_target_tau: 1.0\n"
    config_digest = hashlib.sha256(config_payload).hexdigest()

    class CheckpointEval:
        @staticmethod
        def validate_schema12_completed_bundle(*args, **kwargs):
            assert kwargs["completion_state"] is marker
            return copy.deepcopy(public_record)

        @staticmethod
        def _read_stable_single_link_bytes(path, *, label):
            return Path(path).read_bytes()

    validated = fixed_eval.validate_v19_final_bundle(
        checkpoint_dir,
        marker,
        checkpoint_eval=CheckpointEval,
        completed_validation=public_record,
        config_payload=config_payload,
        expected_config_sha256=config_digest,
    )

    assert validated["authoritative_validator"] == (
        "thinker.util.validate_schema12_final_bundle"
    )
    assert validated["resolved_identity"]["gate_schema"] == 12


def test_validate_v19_final_bundle_fails_closed_without_public_route(tmp_path):
    checkpoint_dir = tmp_path / "runs" / fixed_eval.V19_PRIMARY_XPID
    checkpoint_dir.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="lacks schema-12 completed route"):
        fixed_eval.validate_v19_final_bundle(
            checkpoint_dir,
            {"status": "complete"},
            checkpoint_eval=SimpleNamespace(),
            config_payload=b"voc_gate_policy_schema_version: 12\n",
            expected_config_sha256="a" * 64,
        )


def test_v19_profile_missing_public_dispatch_fails_before_fixed_downstream(
    monkeypatch, tmp_path
):
    import evaluate_dynamic_imitation as checkpoint_eval

    checkpoint_dir = tmp_path / fixed_eval.V19_PRIMARY_XPID
    checkpoint_dir.mkdir()
    payload = (
        "voc_gate_policy_schema_version: 12\n"
        "voc_gate_target_tau: 1.0\n"
        f"xpid: {fixed_eval.V19_PRIMARY_XPID}\n"
    ).encode("utf-8")
    (checkpoint_dir / "config_c.yaml").write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    source_root = Path(fixed_eval.__file__).resolve().parent
    bundle = SimpleNamespace(
        source_root=source_root,
        marker={"status": "complete"},
        file_hashes={"config_c.yaml": digest},
        source_manifest={"path": source_root / "source.sha256"},
    )
    downstream = []
    monkeypatch.setattr(
        fixed_eval, "attest_regular_file", lambda path, label: {"path": str(path)}
    )
    monkeypatch.setattr(
        fixed_eval, "validate_checkpoint_bundle", lambda *args, **kwargs: bundle
    )
    monkeypatch.setattr(fixed_eval, "bind_training_runtime", lambda root: None)
    monkeypatch.delattr(
        checkpoint_eval, "dispatch_schema12_completed_bundle", raising=True
    )
    monkeypatch.setattr(
        checkpoint_eval,
        "_load_flags",
        lambda *a, **k: downstream.append("load_flags"),
    )
    monkeypatch.setattr(
        torch, "load", lambda *a, **k: downstream.append("tensor_load")
    )

    with pytest.raises(RuntimeError, match="lacks schema-12 dispatch"):
        fixed_eval.evaluate(
            SimpleNamespace(
                checkpoint_dir=checkpoint_dir,
                training_source_root=None,
                source_manifest=None,
                confirmation_profile="v19-300k",
                output_dir=tmp_path / "output",
            )
        )

    assert downstream == []
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("private_name", fixed_eval.V17_PRIVATE_LOGGER_MARKERS)
def test_validate_v17_final_bundle_rejects_private_markers(
    tmp_path, private_name
):
    checkpoint_dir = tmp_path / "runs" / fixed_eval.V17_PRIMARY_XPID
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / private_name).write_text("forensic\n", encoding="utf-8")
    public_record = _retarget_v15_bundle_evidence(
        _v17_bundle_evidence(), checkpoint_dir
    )
    public_record.pop("private_logger_markers")
    payload = b"voc_gate_policy_schema_version: 10\n"

    class CheckpointEval:
        @staticmethod
        def validate_schema10_completed_bundle(*args, **kwargs):
            return copy.deepcopy(public_record)

    with pytest.raises(RuntimeError, match="private logger marker"):
        fixed_eval.validate_v17_final_bundle(
            checkpoint_dir,
            {"status": "complete"},
            checkpoint_eval=CheckpointEval,
            config_payload=payload,
            expected_config_sha256=hashlib.sha256(payload).hexdigest(),
        )


def test_validate_v14_final_bundle_uses_public_schema7_validator(tmp_path):
    checkpoint_dir = tmp_path / "run"
    checkpoint_dir.mkdir()
    public_record = _v14_bundle_evidence()
    public_record.pop("private_logger_markers")

    class CheckpointEval:
        @staticmethod
        def validate_schema7_completed_bundle(path, *, completion_state):
            assert Path(path) == checkpoint_dir.resolve()
            assert completion_state == {"status": "complete"}
            return copy.deepcopy(public_record)

    validated = fixed_eval.validate_v14_final_bundle(
        checkpoint_dir,
        {"status": "complete"},
        checkpoint_eval=CheckpointEval,
    )

    assert validated["authoritative_validator"] == (
        "thinker.util.validate_schema7_final_bundle"
    )
    assert set(validated["private_logger_markers"]) == set(
        fixed_eval.V14_PRIVATE_LOGGER_MARKERS
    )


@pytest.mark.parametrize("private_name", fixed_eval.V14_PRIVATE_LOGGER_MARKERS)
def test_validate_v14_final_bundle_rejects_private_markers(
    tmp_path, private_name
):
    checkpoint_dir = tmp_path / "run"
    checkpoint_dir.mkdir()
    (checkpoint_dir / private_name).write_text("forensic\n", encoding="utf-8")
    public_record = _v14_bundle_evidence()
    public_record.pop("private_logger_markers")

    class CheckpointEval:
        @staticmethod
        def validate_schema7_completed_bundle(path, *, completion_state):
            return copy.deepcopy(public_record)

    with pytest.raises(RuntimeError, match="private logger marker"):
        fixed_eval.validate_v14_final_bundle(
            checkpoint_dir,
            {"status": "complete"},
            checkpoint_eval=CheckpointEval,
        )


def test_invalid_v14_bundle_fails_before_flags_live_probe_or_output(
    monkeypatch, tmp_path
):
    import evaluate_dynamic_imitation as checkpoint_eval

    checkpoint_dir = tmp_path / "invalid-v14"
    output_dir = tmp_path / "must-not-exist"
    source_root = Path(fixed_eval.__file__).resolve().parent
    bundle = SimpleNamespace(
        source_root=source_root,
        marker={"status": "complete"},
        file_hashes={},
        source_manifest={"path": source_root / "source.sha256"},
    )
    events = []
    monkeypatch.setattr(
        fixed_eval, "attest_regular_file", lambda path, label: {"path": str(path)}
    )
    monkeypatch.setattr(
        fixed_eval, "validate_checkpoint_bundle", lambda *args, **kwargs: bundle
    )
    monkeypatch.setattr(fixed_eval, "bind_training_runtime", lambda root: None)
    monkeypatch.setattr(
        checkpoint_eval,
        "dispatch_schema8_completed_bundle",
        lambda *args, **kwargs: None,
    )

    def reject_v14(*args, **kwargs):
        events.append("authoritative_v14_validation")
        raise ValueError("invalid schema-7 terminal bundle")

    def forbidden(name):
        def fail(*args, **kwargs):
            events.append(name)
            raise AssertionError(f"{name} ran before v14 validation")

        return fail

    monkeypatch.setattr(fixed_eval, "validate_v14_final_bundle", reject_v14)
    monkeypatch.setattr(checkpoint_eval, "_load_flags", forbidden("load_flags"))
    monkeypatch.setattr(
        checkpoint_eval,
        "resolve_evaluation_spec",
        forbidden("live_environment_spec_probe"),
    )
    monkeypatch.setattr(torch, "load", forbidden("checkpoint_tensor_load"))
    args = SimpleNamespace(
        checkpoint_dir=checkpoint_dir,
        training_source_root=None,
        source_manifest=None,
        confirmation_profile="v14-300k",
        output_dir=output_dir,
    )

    with pytest.raises(ValueError, match="invalid schema-7 terminal bundle"):
        fixed_eval.evaluate(args)

    assert events == ["authoritative_v14_validation"]
    assert not output_dir.exists()


def test_legacy_schema8_claim_probe_rejects_config_swap_before_downstream(
    monkeypatch, tmp_path
):
    import evaluate_dynamic_imitation as checkpoint_eval

    checkpoint_dir = tmp_path / "swapped-config"
    checkpoint_dir.mkdir()
    original = (
        "voc_gate_policy_schema_version: 8\n"
        f"xpid: {fixed_eval.V15_PRIMARY_XPID}\n"
    ).encode("utf-8")
    (checkpoint_dir / "config_c.yaml").write_text(
        "voc_gate_policy_schema_version: 6\n"
        "xpid: enduro-voc-v13-versioned-eps25-seed5-strict-fresh-300k\n",
        encoding="utf-8",
    )
    source_root = Path(fixed_eval.__file__).resolve().parent
    bundle = SimpleNamespace(
        source_root=source_root,
        marker={"status": "complete"},
        file_hashes={"config_c.yaml": hashlib.sha256(original).hexdigest()},
        source_manifest={"path": source_root / "source.sha256"},
    )
    events = []
    monkeypatch.setattr(
        fixed_eval, "attest_regular_file", lambda path, label: {"path": str(path)}
    )
    monkeypatch.setattr(
        fixed_eval, "validate_checkpoint_bundle", lambda *args, **kwargs: bundle
    )
    monkeypatch.setattr(fixed_eval, "bind_training_runtime", lambda root: None)

    def forbidden(name):
        def fail(*args, **kwargs):
            events.append(name)
            raise AssertionError(f"{name} ran after config swap")

        return fail

    monkeypatch.setattr(
        checkpoint_eval,
        "dispatch_schema8_completed_bundle",
        forbidden("schema8_dispatch"),
    )
    monkeypatch.setattr(checkpoint_eval, "_load_flags", forbidden("load_flags"))
    monkeypatch.setattr(
        checkpoint_eval, "resolve_evaluation_spec", forbidden("live_probe")
    )
    monkeypatch.setattr(torch, "load", forbidden("downstream_tensor_load"))
    output_dir = tmp_path / "output"

    with pytest.raises(RuntimeError, match="config changed"):
        fixed_eval.evaluate(
            SimpleNamespace(
                checkpoint_dir=checkpoint_dir,
                training_source_root=None,
                source_manifest=None,
                confirmation_profile="v13-300k",
                output_dir=output_dir,
            )
        )

    assert events == []
    assert not output_dir.exists()


def test_legacy_schema8_claim_probe_rejects_bound_config_deletion(
    monkeypatch, tmp_path
):
    import evaluate_dynamic_imitation as checkpoint_eval

    checkpoint_dir = tmp_path / "deleted-config"
    checkpoint_dir.mkdir()
    source_root = Path(fixed_eval.__file__).resolve().parent
    bundle = SimpleNamespace(
        source_root=source_root,
        marker={"status": "complete"},
        file_hashes={"config_c.yaml": "a" * 64},
        source_manifest={"path": source_root / "source.sha256"},
    )
    events = []
    monkeypatch.setattr(
        fixed_eval, "attest_regular_file", lambda path, label: {"path": str(path)}
    )
    monkeypatch.setattr(
        fixed_eval, "validate_checkpoint_bundle", lambda *args, **kwargs: bundle
    )
    monkeypatch.setattr(fixed_eval, "bind_training_runtime", lambda root: None)

    def forbidden(name):
        def fail(*args, **kwargs):
            events.append(name)
            raise AssertionError(f"{name} ran after bound config deletion")

        return fail

    monkeypatch.setattr(
        checkpoint_eval,
        "dispatch_schema8_completed_bundle",
        forbidden("schema8_dispatch"),
    )
    monkeypatch.setattr(checkpoint_eval, "_load_flags", forbidden("load_flags"))
    monkeypatch.setattr(
        checkpoint_eval, "resolve_evaluation_spec", forbidden("live_probe")
    )
    monkeypatch.setattr(torch, "load", forbidden("downstream_tensor_load"))
    output_dir = tmp_path / "output"

    with pytest.raises(FileNotFoundError):
        fixed_eval.evaluate(
            SimpleNamespace(
                checkpoint_dir=checkpoint_dir,
                training_source_root=None,
                source_manifest=None,
                confirmation_profile="v13-300k",
                output_dir=output_dir,
            )
        )

    assert events == []
    assert not output_dir.exists()


def test_invalid_v15_bundle_fails_before_every_downstream_probe_or_output(
    monkeypatch, tmp_path
):
    import evaluate_dynamic_imitation as checkpoint_eval

    checkpoint_dir = tmp_path / "invalid-v15"
    checkpoint_dir.mkdir()
    config_payload = (
        "voc_gate_policy_schema_version: 8\n"
        f"xpid: {fixed_eval.V15_PRIMARY_XPID}\n"
    ).encode("utf-8")
    (checkpoint_dir / "config_c.yaml").write_bytes(config_payload)
    config_digest = hashlib.sha256(config_payload).hexdigest()
    output_dir = tmp_path / "must-not-exist"
    source_root = Path(fixed_eval.__file__).resolve().parent
    bundle = SimpleNamespace(
        source_root=source_root,
        marker={"status": "complete"},
        file_hashes={"config_c.yaml": config_digest},
        source_manifest={"path": source_root / "source.sha256"},
    )
    events = []
    monkeypatch.setattr(
        fixed_eval, "attest_regular_file", lambda path, label: {"path": str(path)}
    )
    monkeypatch.setattr(
        fixed_eval, "validate_checkpoint_bundle", lambda *args, **kwargs: bundle
    )
    monkeypatch.setattr(fixed_eval, "bind_training_runtime", lambda root: None)
    monkeypatch.setattr(
        checkpoint_eval,
        "dispatch_schema8_completed_bundle",
        lambda *args, **kwargs: {"authoritative_validator": "schema8"},
    )

    def reject_v15(*args, **kwargs):
        events.append("authoritative_v15_validation")
        raise ValueError("invalid schema-8 terminal bundle")

    def forbidden(name):
        def fail(*args, **kwargs):
            events.append(name)
            raise AssertionError(f"{name} ran before v15 validation")

        return fail

    monkeypatch.setattr(fixed_eval, "validate_v15_final_bundle", reject_v15)
    monkeypatch.setattr(checkpoint_eval, "_load_flags", forbidden("load_flags"))
    monkeypatch.setattr(
        checkpoint_eval,
        "resolve_evaluation_spec",
        forbidden("live_environment_spec_probe"),
    )
    monkeypatch.setattr(torch, "load", forbidden("downstream_tensor_load"))
    monkeypatch.setattr(
        fixed_eval,
        "validate_behavioral_training_data",
        forbidden("behavioral_data"),
    )
    monkeypatch.setattr(
        fixed_eval, "run_fixed_rollouts", forbidden("rollout")
    )
    args = SimpleNamespace(
        checkpoint_dir=checkpoint_dir,
        training_source_root=None,
        source_manifest=None,
        confirmation_profile="v15-300k",
        output_dir=output_dir,
    )

    with pytest.raises(ValueError, match="invalid schema-8 terminal bundle"):
        fixed_eval.evaluate(args)

    assert events == ["authoritative_v15_validation"]
    assert not output_dir.exists()


def test_invalid_v16_bundle_fails_before_every_downstream_probe_or_output(
    monkeypatch, tmp_path
):
    import evaluate_dynamic_imitation as checkpoint_eval

    checkpoint_dir = tmp_path / "invalid-v16"
    checkpoint_dir.mkdir()
    config_payload = (
        "voc_gate_policy_schema_version: 9\n"
        f"xpid: {fixed_eval.V16_PRIMARY_XPID}\n"
    ).encode("utf-8")
    (checkpoint_dir / "config_c.yaml").write_bytes(config_payload)
    config_digest = hashlib.sha256(config_payload).hexdigest()
    output_dir = tmp_path / "must-not-exist"
    source_root = Path(fixed_eval.__file__).resolve().parent
    bundle = SimpleNamespace(
        source_root=source_root,
        marker={"status": "complete"},
        file_hashes={"config_c.yaml": config_digest},
        source_manifest={"path": source_root / "source.sha256"},
    )
    events = []
    monkeypatch.setattr(
        fixed_eval, "attest_regular_file", lambda path, label: {"path": str(path)}
    )
    monkeypatch.setattr(
        fixed_eval, "validate_checkpoint_bundle", lambda *args, **kwargs: bundle
    )
    monkeypatch.setattr(fixed_eval, "bind_training_runtime", lambda root: None)
    monkeypatch.setattr(
        checkpoint_eval,
        "dispatch_schema9_completed_bundle",
        lambda *args, **kwargs: {"authoritative_validator": "schema9"},
    )

    def reject_v16(*args, **kwargs):
        events.append("authoritative_v16_validation")
        raise ValueError("invalid schema-9 terminal bundle")

    def forbidden(name):
        def fail(*args, **kwargs):
            events.append(name)
            raise AssertionError(f"{name} ran before v16 validation")

        return fail

    monkeypatch.setattr(fixed_eval, "validate_v16_final_bundle", reject_v16)
    monkeypatch.setattr(
        checkpoint_eval,
        "_load_flags_from_validated_config_bytes",
        forbidden("byte_bound_load_flags"),
    )
    monkeypatch.setattr(
        checkpoint_eval,
        "resolve_evaluation_spec",
        forbidden("live_environment_spec_probe"),
    )
    monkeypatch.setattr(torch, "load", forbidden("downstream_tensor_load"))
    monkeypatch.setattr(
        fixed_eval,
        "validate_behavioral_training_data",
        forbidden("behavioral_data"),
    )
    monkeypatch.setattr(
        fixed_eval, "run_fixed_rollouts", forbidden("rollout")
    )

    with pytest.raises(ValueError, match="invalid schema-9 terminal bundle"):
        fixed_eval.evaluate(
            SimpleNamespace(
                checkpoint_dir=checkpoint_dir,
                training_source_root=None,
                source_manifest=None,
                confirmation_profile="v16-300k",
                output_dir=output_dir,
            )
        )

    assert events == ["authoritative_v16_validation"]
    assert not output_dir.exists()


def test_invalid_v17_bundle_fails_before_every_downstream_probe_or_output(
    monkeypatch, tmp_path
):
    import evaluate_dynamic_imitation as checkpoint_eval

    checkpoint_dir = tmp_path / "invalid-v17"
    checkpoint_dir.mkdir()
    config_payload = (
        "voc_gate_policy_schema_version: 10\n"
        f"xpid: {fixed_eval.V17_PRIMARY_XPID}\n"
    ).encode("utf-8")
    (checkpoint_dir / "config_c.yaml").write_bytes(config_payload)
    config_digest = hashlib.sha256(config_payload).hexdigest()
    output_dir = tmp_path / "must-not-exist"
    source_root = Path(fixed_eval.__file__).resolve().parent
    bundle = SimpleNamespace(
        source_root=source_root,
        marker={"status": "complete"},
        file_hashes={"config_c.yaml": config_digest},
        source_manifest={"path": source_root / "source.sha256"},
    )
    events = []
    monkeypatch.setattr(
        fixed_eval, "attest_regular_file", lambda path, label: {"path": str(path)}
    )
    monkeypatch.setattr(
        fixed_eval, "validate_checkpoint_bundle", lambda *args, **kwargs: bundle
    )
    monkeypatch.setattr(fixed_eval, "bind_training_runtime", lambda root: None)
    monkeypatch.setattr(
        checkpoint_eval,
        "dispatch_schema10_completed_bundle",
        lambda *args, **kwargs: {"authoritative_validator": "schema10"},
    )

    def reject_v17(*args, **kwargs):
        events.append("authoritative_v17_validation")
        raise ValueError("invalid schema-10 terminal bundle")

    def forbidden(name):
        def fail(*args, **kwargs):
            events.append(name)
            raise AssertionError(f"{name} ran before v17 validation")

        return fail

    monkeypatch.setattr(fixed_eval, "validate_v17_final_bundle", reject_v17)
    monkeypatch.setattr(
        checkpoint_eval,
        "_load_flags_from_validated_config_bytes",
        forbidden("byte_bound_load_flags"),
    )
    monkeypatch.setattr(
        checkpoint_eval,
        "resolve_evaluation_spec",
        forbidden("live_environment_spec_probe"),
    )
    monkeypatch.setattr(torch, "load", forbidden("downstream_tensor_load"))
    monkeypatch.setattr(
        fixed_eval,
        "validate_behavioral_training_data",
        forbidden("behavioral_data"),
    )
    monkeypatch.setattr(fixed_eval, "run_fixed_rollouts", forbidden("rollout"))

    with pytest.raises(ValueError, match="invalid schema-10 terminal bundle"):
        fixed_eval.evaluate(
            SimpleNamespace(
                checkpoint_dir=checkpoint_dir,
                training_source_root=None,
                source_manifest=None,
                confirmation_profile="v17-300k",
                output_dir=output_dir,
            )
        )

    assert events == ["authoritative_v17_validation"]
    assert not output_dir.exists()


def test_invalid_v18_bundle_fails_before_every_downstream_probe_or_output(
    monkeypatch, tmp_path
):
    import evaluate_dynamic_imitation as checkpoint_eval

    checkpoint_dir = tmp_path / "invalid-v18"
    checkpoint_dir.mkdir()
    config_payload = (
        "voc_gate_policy_schema_version: 11\n"
        f"xpid: {fixed_eval.V18_PRIMARY_XPID}\n"
    ).encode("utf-8")
    (checkpoint_dir / "config_c.yaml").write_bytes(config_payload)
    config_digest = hashlib.sha256(config_payload).hexdigest()
    output_dir = tmp_path / "must-not-exist"
    source_root = Path(fixed_eval.__file__).resolve().parent
    bundle = SimpleNamespace(
        source_root=source_root,
        marker={"status": "complete"},
        file_hashes={"config_c.yaml": config_digest},
        source_manifest={"path": source_root / "source.sha256"},
    )
    events = []
    monkeypatch.setattr(
        fixed_eval, "attest_regular_file", lambda path, label: {"path": str(path)}
    )
    monkeypatch.setattr(
        fixed_eval, "validate_checkpoint_bundle", lambda *args, **kwargs: bundle
    )
    monkeypatch.setattr(fixed_eval, "bind_training_runtime", lambda root: None)
    monkeypatch.setattr(
        checkpoint_eval,
        "dispatch_schema11_completed_bundle",
        lambda *args, **kwargs: {"authoritative_validator": "schema11"},
    )

    def reject_v18(*args, **kwargs):
        events.append("authoritative_v18_validation")
        raise ValueError("invalid schema-11 terminal bundle")

    def forbidden(name):
        def fail(*args, **kwargs):
            events.append(name)
            raise AssertionError(f"{name} ran before v18 validation")

        return fail

    monkeypatch.setattr(fixed_eval, "validate_v18_final_bundle", reject_v18)
    monkeypatch.setattr(
        checkpoint_eval,
        "_load_flags_from_validated_config_bytes",
        forbidden("byte_bound_load_flags"),
    )
    monkeypatch.setattr(
        checkpoint_eval,
        "resolve_evaluation_spec",
        forbidden("live_environment_spec_probe"),
    )
    monkeypatch.setattr(torch, "load", forbidden("downstream_tensor_load"))
    monkeypatch.setattr(
        fixed_eval,
        "validate_behavioral_training_data",
        forbidden("behavioral_data"),
    )
    monkeypatch.setattr(fixed_eval, "run_fixed_rollouts", forbidden("rollout"))

    with pytest.raises(ValueError, match="invalid schema-11 terminal bundle"):
        fixed_eval.evaluate(
            SimpleNamespace(
                checkpoint_dir=checkpoint_dir,
                training_source_root=None,
                source_manifest=None,
                confirmation_profile="v18-300k",
                output_dir=output_dir,
            )
        )

    assert events == ["authoritative_v18_validation"]
    assert not output_dir.exists()


def test_v17_profile_missing_public_schema10_feature_gate_fails_closed(
    monkeypatch, tmp_path
):
    import evaluate_dynamic_imitation as checkpoint_eval

    checkpoint_dir = tmp_path / "v17-missing-feature"
    checkpoint_dir.mkdir()
    payload = (
        "voc_gate_policy_schema_version: 10\n"
        f"xpid: {fixed_eval.V17_PRIMARY_XPID}\n"
    ).encode("utf-8")
    (checkpoint_dir / "config_c.yaml").write_bytes(payload)
    source_root = Path(fixed_eval.__file__).resolve().parent
    bundle = SimpleNamespace(
        source_root=source_root,
        marker={"status": "complete"},
        file_hashes={"config_c.yaml": hashlib.sha256(payload).hexdigest()},
        source_manifest={"path": source_root / "source.sha256"},
    )
    downstream = []
    monkeypatch.delattr(
        checkpoint_eval, "dispatch_schema10_completed_bundle", raising=True
    )
    monkeypatch.setattr(
        fixed_eval, "attest_regular_file", lambda path, label: {"path": str(path)}
    )
    monkeypatch.setattr(
        fixed_eval, "validate_checkpoint_bundle", lambda *args, **kwargs: bundle
    )
    monkeypatch.setattr(fixed_eval, "bind_training_runtime", lambda root: None)
    monkeypatch.setattr(
        checkpoint_eval,
        "_load_flags_from_validated_config_bytes",
        lambda *a, **k: downstream.append("load_flags"),
    )
    monkeypatch.setattr(
        torch, "load", lambda *a, **k: downstream.append("tensor_load")
    )

    with pytest.raises(RuntimeError, match="lacks schema-10 dispatch"):
        fixed_eval.evaluate(
            SimpleNamespace(
                checkpoint_dir=checkpoint_dir,
                training_source_root=None,
                source_manifest=None,
                confirmation_profile="v17-300k",
                output_dir=tmp_path / "output",
            )
        )

    assert downstream == []
    assert not (tmp_path / "output").exists()


def test_v18_profile_missing_public_schema11_feature_gate_fails_closed(
    monkeypatch, tmp_path
):
    import evaluate_dynamic_imitation as checkpoint_eval

    checkpoint_dir = tmp_path / "v18-missing-feature"
    checkpoint_dir.mkdir()
    payload = (
        "voc_gate_policy_schema_version: 11\n"
        f"xpid: {fixed_eval.V18_PRIMARY_XPID}\n"
    ).encode("utf-8")
    (checkpoint_dir / "config_c.yaml").write_bytes(payload)
    source_root = Path(fixed_eval.__file__).resolve().parent
    bundle = SimpleNamespace(
        source_root=source_root,
        marker={"status": "complete"},
        file_hashes={"config_c.yaml": hashlib.sha256(payload).hexdigest()},
        source_manifest={"path": source_root / "source.sha256"},
    )
    downstream = []
    monkeypatch.delattr(
        checkpoint_eval, "dispatch_schema11_completed_bundle", raising=True
    )
    monkeypatch.setattr(
        fixed_eval, "attest_regular_file", lambda path, label: {"path": str(path)}
    )
    monkeypatch.setattr(
        fixed_eval, "validate_checkpoint_bundle", lambda *args, **kwargs: bundle
    )
    monkeypatch.setattr(fixed_eval, "bind_training_runtime", lambda root: None)
    monkeypatch.setattr(
        checkpoint_eval,
        "_load_flags_from_validated_config_bytes",
        lambda *a, **k: downstream.append("load_flags"),
    )
    monkeypatch.setattr(
        torch, "load", lambda *a, **k: downstream.append("tensor_load")
    )

    with pytest.raises(RuntimeError, match="lacks schema-11 dispatch"):
        fixed_eval.evaluate(
            SimpleNamespace(
                checkpoint_dir=checkpoint_dir,
                training_source_root=None,
                source_manifest=None,
                confirmation_profile="v18-300k",
                output_dir=tmp_path / "output",
            )
        )

    assert downstream == []
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "wrong_profile",
    [
        "v7-200k",
        "v10-300k",
        "v11-300k",
        "v12-300k",
        "v13-300k",
        "v14-300k",
        "v15-300k",
        "v17-300k",
        "v18-300k",
    ],
)
def test_schema9_checkpoint_rejects_every_legacy_v15_profile_before_downstream(
    monkeypatch, tmp_path, wrong_profile
):
    import evaluate_dynamic_imitation as checkpoint_eval

    checkpoint_dir = tmp_path / "schema9"
    checkpoint_dir.mkdir()
    config_payload = (
        "voc_gate_policy_schema_version: 9\n"
        f"xpid: {fixed_eval.V16_PRIMARY_XPID}\n"
    ).encode("utf-8")
    (checkpoint_dir / "config_c.yaml").write_bytes(config_payload)
    source_root = Path(fixed_eval.__file__).resolve().parent
    bundle = SimpleNamespace(
        source_root=source_root,
        marker={"status": "complete"},
        file_hashes={
            "config_c.yaml": hashlib.sha256(config_payload).hexdigest()
        },
        source_manifest={"path": source_root / "source.sha256"},
    )
    events = []
    monkeypatch.setattr(
        fixed_eval, "attest_regular_file", lambda path, label: {"path": str(path)}
    )
    monkeypatch.setattr(
        fixed_eval, "validate_checkpoint_bundle", lambda *args, **kwargs: bundle
    )
    monkeypatch.setattr(fixed_eval, "bind_training_runtime", lambda root: None)

    def dispatch_schema9(*args, **kwargs):
        events.append("schema9_dispatch")
        return {"authoritative_validator": "thinker.util.validate_schema9_final_bundle"}

    def forbidden(name):
        def fail(*args, **kwargs):
            events.append(name)
            raise AssertionError(f"{name} ran before schema-9 profile rejection")

        return fail

    monkeypatch.setattr(
        checkpoint_eval, "dispatch_schema9_completed_bundle", dispatch_schema9
    )
    monkeypatch.setattr(
        checkpoint_eval, "dispatch_schema8_completed_bundle", forbidden("schema8")
    )
    monkeypatch.setattr(checkpoint_eval, "_load_flags", forbidden("load_flags"))
    monkeypatch.setattr(torch, "load", forbidden("tensor_load"))

    with pytest.raises(ValueError, match="eligible only.*v16-300k"):
        fixed_eval.evaluate(
            SimpleNamespace(
                checkpoint_dir=checkpoint_dir,
                training_source_root=None,
                source_manifest=None,
                confirmation_profile=wrong_profile,
                output_dir=tmp_path / "output",
            )
        )

    assert events == ["schema9_dispatch"]
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "wrong_profile",
    [
        "v7-200k",
        "v10-300k",
        "v11-300k",
        "v12-300k",
        "v13-300k",
        "v14-300k",
        "v15-300k",
        "v16-300k",
        "v18-300k",
    ],
)
def test_schema10_checkpoint_rejects_every_non_v17_profile_before_downstream(
    monkeypatch, tmp_path, wrong_profile
):
    import evaluate_dynamic_imitation as checkpoint_eval

    checkpoint_dir = tmp_path / "schema10"
    checkpoint_dir.mkdir()
    config_payload = (
        "voc_gate_policy_schema_version: 10\n"
        f"xpid: {fixed_eval.V17_PRIMARY_XPID}\n"
    ).encode("utf-8")
    (checkpoint_dir / "config_c.yaml").write_bytes(config_payload)
    source_root = Path(fixed_eval.__file__).resolve().parent
    bundle = SimpleNamespace(
        source_root=source_root,
        marker={"status": "complete"},
        file_hashes={
            "config_c.yaml": hashlib.sha256(config_payload).hexdigest()
        },
        source_manifest={"path": source_root / "source.sha256"},
    )
    events = []
    monkeypatch.setattr(
        fixed_eval, "attest_regular_file", lambda path, label: {"path": str(path)}
    )
    monkeypatch.setattr(
        fixed_eval, "validate_checkpoint_bundle", lambda *args, **kwargs: bundle
    )
    monkeypatch.setattr(fixed_eval, "bind_training_runtime", lambda root: None)

    def dispatch_schema10(*args, **kwargs):
        events.append("schema10_dispatch")
        return {
            "authoritative_validator": (
                "thinker.util.validate_schema10_final_bundle"
            )
        }

    def forbidden(name):
        def fail(*args, **kwargs):
            events.append(name)
            raise AssertionError(f"{name} ran before schema-10 profile rejection")

        return fail

    monkeypatch.setattr(
        checkpoint_eval, "dispatch_schema10_completed_bundle", dispatch_schema10
    )
    monkeypatch.setattr(
        checkpoint_eval, "dispatch_schema9_completed_bundle", forbidden("schema9")
    )
    monkeypatch.setattr(checkpoint_eval, "_load_flags", forbidden("load_flags"))
    monkeypatch.setattr(torch, "load", forbidden("tensor_load"))

    with pytest.raises(ValueError, match="eligible only.*v17-300k"):
        fixed_eval.evaluate(
            SimpleNamespace(
                checkpoint_dir=checkpoint_dir,
                training_source_root=None,
                source_manifest=None,
                confirmation_profile=wrong_profile,
                output_dir=tmp_path / "output",
            )
        )

    assert events == ["schema10_dispatch"]
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "wrong_profile",
    [
        "v7-200k",
        "v10-300k",
        "v11-300k",
        "v12-300k",
        "v13-300k",
        "v14-300k",
        "v15-300k",
        "v16-300k",
        "v17-300k",
    ],
)
def test_schema11_checkpoint_rejects_every_non_v18_profile_before_downstream(
    monkeypatch, tmp_path, wrong_profile
):
    import evaluate_dynamic_imitation as checkpoint_eval

    checkpoint_dir = tmp_path / "schema11-as-v17"
    checkpoint_dir.mkdir()
    config_payload = (
        "voc_gate_policy_schema_version: 11\n"
        f"xpid: {fixed_eval.V18_PRIMARY_XPID}\n"
    ).encode("utf-8")
    (checkpoint_dir / "config_c.yaml").write_bytes(config_payload)
    source_root = Path(fixed_eval.__file__).resolve().parent
    bundle = SimpleNamespace(
        source_root=source_root,
        marker={"status": "complete"},
        file_hashes={
            "config_c.yaml": hashlib.sha256(config_payload).hexdigest()
        },
        source_manifest={"path": source_root / "source.sha256"},
    )
    events = []
    monkeypatch.setattr(
        fixed_eval, "attest_regular_file", lambda path, label: {"path": str(path)}
    )
    monkeypatch.setattr(
        fixed_eval, "validate_checkpoint_bundle", lambda *args, **kwargs: bundle
    )
    monkeypatch.setattr(fixed_eval, "bind_training_runtime", lambda root: None)

    def dispatch_schema11(*args, **kwargs):
        events.append("schema11_dispatch")
        return {
            "authoritative_validator": (
                "thinker.util.validate_schema11_final_bundle"
            )
        }

    def forbidden(name):
        def fail(*args, **kwargs):
            events.append(name)
            raise AssertionError(f"{name} ran before schema-11 profile rejection")

        return fail

    monkeypatch.setattr(
        checkpoint_eval, "dispatch_schema11_completed_bundle", dispatch_schema11
    )
    monkeypatch.setattr(
        checkpoint_eval, "dispatch_schema10_completed_bundle", forbidden("schema10")
    )
    monkeypatch.setattr(checkpoint_eval, "_load_flags", forbidden("load_flags"))
    monkeypatch.setattr(torch, "load", forbidden("tensor_load"))

    with pytest.raises(ValueError, match="eligible only.*v18-300k"):
        fixed_eval.evaluate(
            SimpleNamespace(
                checkpoint_dir=checkpoint_dir,
                training_source_root=None,
                source_manifest=None,
                confirmation_profile=wrong_profile,
                output_dir=tmp_path / "output",
            )
        )

    assert events == ["schema11_dispatch"]
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "wrong_profile",
    [
        "v7-200k",
        "v10-300k",
        "v11-300k",
        "v12-300k",
        "v13-300k",
        "v14-300k",
        "v15-300k",
        "v16-300k",
        "v17-300k",
        "v18-300k",
    ],
)
def test_schema12_checkpoint_rejects_every_non_v19_profile_before_downstream(
    monkeypatch, tmp_path, wrong_profile
):
    import evaluate_dynamic_imitation as checkpoint_eval

    checkpoint_dir = tmp_path / "schema12-as-legacy"
    checkpoint_dir.mkdir()
    config_payload = (
        "voc_gate_policy_schema_version: 12\n"
        "voc_gate_target_tau: 1.0\n"
        f"xpid: {fixed_eval.V19_PRIMARY_XPID}\n"
    ).encode("utf-8")
    (checkpoint_dir / "config_c.yaml").write_bytes(config_payload)
    source_root = Path(fixed_eval.__file__).resolve().parent
    bundle = SimpleNamespace(
        source_root=source_root,
        marker={"status": "complete"},
        file_hashes={"config_c.yaml": hashlib.sha256(config_payload).hexdigest()},
        source_manifest={"path": source_root / "source.sha256"},
    )
    events = []
    monkeypatch.setattr(
        fixed_eval, "attest_regular_file", lambda path, label: {"path": str(path)}
    )
    monkeypatch.setattr(
        fixed_eval, "validate_checkpoint_bundle", lambda *args, **kwargs: bundle
    )
    monkeypatch.setattr(fixed_eval, "bind_training_runtime", lambda root: None)

    def dispatch_schema12(*args, **kwargs):
        events.append("schema12_dispatch")
        return {
            "authoritative_validator": (
                "thinker.util.validate_schema12_final_bundle"
            )
        }

    def forbidden(name):
        def fail(*args, **kwargs):
            events.append(name)
            raise AssertionError(f"{name} ran before schema-12 profile rejection")

        return fail

    monkeypatch.setattr(
        checkpoint_eval, "dispatch_schema12_completed_bundle", dispatch_schema12
    )
    monkeypatch.setattr(checkpoint_eval, "_load_flags", forbidden("load_flags"))
    monkeypatch.setattr(torch, "load", forbidden("tensor_load"))

    with pytest.raises(ValueError, match="eligible only.*v19-300k"):
        fixed_eval.evaluate(
            SimpleNamespace(
                checkpoint_dir=checkpoint_dir,
                training_source_root=None,
                source_manifest=None,
                confirmation_profile=wrong_profile,
                output_dir=tmp_path / "output",
            )
        )

    assert events == ["schema12_dispatch"]
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "wrong_profile",
    [
        "v7-200k",
        "v10-300k",
        "v11-300k",
        "v12-300k",
        "v13-300k",
        "v14-300k",
        "v15-300k",
        "v16-300k",
        "v17-300k",
        "v18-300k",
        "v19-300k",
    ],
)
def test_schema13_checkpoint_rejects_every_non_v20_profile_before_downstream(
    monkeypatch, tmp_path, wrong_profile
):
    import evaluate_dynamic_imitation as checkpoint_eval

    checkpoint_dir = tmp_path / "schema13-as-legacy"
    checkpoint_dir.mkdir()
    config_payload = (
        "voc_gate_policy_schema_version: 13\n"
        "voc_gate_target_tau: 1.0\n"
        f"xpid: {fixed_eval.V20_PRIMARY_XPID}\n"
    ).encode("utf-8")
    (checkpoint_dir / "config_c.yaml").write_bytes(config_payload)
    source_root = Path(fixed_eval.__file__).resolve().parent
    bundle = SimpleNamespace(
        source_root=source_root,
        marker={"schema_version": 2, "status": "complete"},
        file_hashes={"config_c.yaml": hashlib.sha256(config_payload).hexdigest()},
        source_manifest={"path": source_root / "source.sha256"},
    )
    events = []
    monkeypatch.setattr(
        fixed_eval, "attest_regular_file", lambda path, label: {"path": str(path)}
    )
    monkeypatch.setattr(
        fixed_eval, "validate_checkpoint_bundle", lambda *args, **kwargs: bundle
    )
    monkeypatch.setattr(fixed_eval, "bind_training_runtime", lambda root: None)

    def dispatch_schema13(*args, **kwargs):
        events.append("schema13_dispatch")
        return {
            "authoritative_validator": (
                "thinker.util.validate_schema13_final_bundle"
            )
        }

    def forbidden(name):
        def fail(*args, **kwargs):
            events.append(name)
            raise AssertionError(f"{name} ran before schema-13 profile rejection")

        return fail

    monkeypatch.setattr(
        checkpoint_eval, "dispatch_schema13_completed_bundle", dispatch_schema13
    )
    monkeypatch.setattr(checkpoint_eval, "_load_flags", forbidden("load_flags"))
    monkeypatch.setattr(torch, "load", forbidden("tensor_load"))

    with pytest.raises(ValueError, match="eligible only.*v20-300k"):
        fixed_eval.evaluate(
            SimpleNamespace(
                checkpoint_dir=checkpoint_dir,
                training_source_root=None,
                source_manifest=None,
                confirmation_profile=wrong_profile,
                output_dir=tmp_path / "output",
            )
        )

    assert events == ["schema13_dispatch"]
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "schema_line",
    [
        pytest.param(b"voc_gate_policy_schema_version: 5\n", id="wrong-schema"),
        pytest.param(b"", id="missing-schema"),
    ],
)
def test_malformed_v20_prefix_routes_schema13_before_fixed_downstream(
    monkeypatch, tmp_path, schema_line
):
    import evaluate_dynamic_imitation as checkpoint_eval

    checkpoint_dir = tmp_path / "malformed-v20-prefix"
    checkpoint_dir.mkdir()
    config_payload = schema_line + (
        b"xpid: enduro-voc-v20-telemetry-tau1-orthocd-adam-eps25-malformed\n"
    )
    (checkpoint_dir / "config_c.yaml").write_bytes(config_payload)
    source_root = Path(fixed_eval.__file__).resolve().parent
    bundle = SimpleNamespace(
        source_root=source_root,
        marker={"schema_version": 2, "status": "complete"},
        file_hashes={"config_c.yaml": hashlib.sha256(config_payload).hexdigest()},
        source_manifest={"path": source_root / "source.sha256"},
    )
    output_dir = tmp_path / "must-not-exist"
    events = []
    monkeypatch.setattr(
        fixed_eval, "attest_regular_file", lambda path, label: {"path": str(path)}
    )
    monkeypatch.setattr(
        fixed_eval, "validate_checkpoint_bundle", lambda *args, **kwargs: bundle
    )
    monkeypatch.setattr(fixed_eval, "bind_training_runtime", lambda root: None)

    def reject(*args, **kwargs):
        events.append("schema13_dispatch")
        raise ValueError("dedicated schema-13 requires exact integer 13")

    def forbidden(name):
        def fail(*args, **kwargs):
            events.append(name)
            raise AssertionError(f"{name} ran before schema-13 validation")

        return fail

    monkeypatch.setattr(
        checkpoint_eval, "dispatch_schema13_completed_bundle", reject
    )
    monkeypatch.setattr(
        checkpoint_eval,
        "_load_flags_from_validated_config_bytes",
        forbidden("load_flags"),
    )
    monkeypatch.setattr(torch, "load", forbidden("tensor_load"))

    with pytest.raises(ValueError, match="schema-13"):
        fixed_eval.evaluate(
            SimpleNamespace(
                checkpoint_dir=checkpoint_dir,
                training_source_root=None,
                source_manifest=None,
                confirmation_profile="v20-300k",
                output_dir=output_dir,
            )
        )

    assert events == ["schema13_dispatch"]
    assert not output_dir.exists()


def test_binary_v20_xpid_under_v19_profile_routes_before_rng_or_tensor_load(
    monkeypatch, tmp_path
):
    import evaluate_dynamic_imitation as checkpoint_eval

    checkpoint_dir = tmp_path / "binary-v20-as-v19"
    checkpoint_dir.mkdir()
    config_payload = (
        b'xpid: !!binary "ZW5kdXJvLXZvYy12MjAtdGVsZW1ldHJ5LXRhdTEt'
        b'b3J0aG9jZC1hZGFtLWVwczI1LW1hbGZvcm1lZA=="\n'
    )
    (checkpoint_dir / "config_c.yaml").write_bytes(config_payload)
    source_root = Path(fixed_eval.__file__).resolve().parent
    bundle = SimpleNamespace(
        source_root=source_root,
        marker={"schema_version": 2, "status": "complete"},
        file_hashes={
            "config_c.yaml": hashlib.sha256(config_payload).hexdigest()
        },
        source_manifest={"path": source_root / "source.sha256"},
    )
    events = []
    monkeypatch.setattr(
        fixed_eval, "attest_regular_file", lambda path, label: {"path": str(path)}
    )
    monkeypatch.setattr(
        fixed_eval, "validate_checkpoint_bundle", lambda *args, **kwargs: bundle
    )
    monkeypatch.setattr(fixed_eval, "bind_training_runtime", lambda root: None)

    def forbidden(name):
        def fail(*args, **kwargs):
            events.append(name)
            raise AssertionError(f"{name} ran before schema-13 validation")

        return fail

    def reject(*args, **kwargs):
        events.append("schema13_dispatch")
        raise ValueError("schema-13 bytes xpid is not a valid V20 surface")

    monkeypatch.setattr(
        checkpoint_eval, "dispatch_schema13_completed_bundle", reject
    )
    monkeypatch.setattr(
        checkpoint_eval,
        "_load_flags_from_validated_config_bytes",
        forbidden("load_flags"),
    )
    monkeypatch.setattr(
        fixed_eval, "_set_deterministic_seed", forbidden("rng_seed")
    )
    monkeypatch.setattr(torch, "load", forbidden("tensor_load"))

    with pytest.raises(ValueError, match="bytes xpid"):
        fixed_eval.evaluate(
            SimpleNamespace(
                checkpoint_dir=checkpoint_dir,
                training_source_root=None,
                source_manifest=None,
                confirmation_profile="v19-300k",
                output_dir=tmp_path / "must-not-exist",
            )
        )

    assert events == ["schema13_dispatch"]
    assert not (tmp_path / "must-not-exist").exists()


@pytest.mark.parametrize(
    "schema_line",
    [
        pytest.param(b"voc_gate_policy_schema_version: 5\n", id="wrong-schema"),
        pytest.param(b"", id="missing-schema"),
    ],
)
@pytest.mark.parametrize(
    "xpid_line",
    [
        pytest.param(
            b"xpid: enduro-voc-v20-telemetry-tau1-orthocd-adam-eps25-malformed\n",
            id="plain-xpid",
        ),
        pytest.param(
            b'xpid: !!binary "ZW5kdXJvLXZvYy12MjAtdGVsZW1ldHJ5LXRhdTEt'
            b'b3J0aG9jZC1hZGFtLWVwczI1LW1hbGZvcm1lZA=="\n',
            id="bytes-xpid",
        ),
    ],
)
def test_local_v20_classifier_fails_closed_without_external_classifier(
    monkeypatch, tmp_path, schema_line, xpid_line
):
    import evaluate_dynamic_imitation as checkpoint_eval

    checkpoint_dir = tmp_path / "forward-v20-with-frozen-v19-evaluator"
    checkpoint_dir.mkdir()
    config_payload = schema_line + xpid_line
    (checkpoint_dir / "config_c.yaml").write_bytes(config_payload)
    source_root = Path(fixed_eval.__file__).resolve().parent
    bundle = SimpleNamespace(
        source_root=source_root,
        marker={"schema_version": 2, "status": "complete"},
        file_hashes={
            "config_c.yaml": hashlib.sha256(config_payload).hexdigest()
        },
        source_manifest={"path": source_root / "source.sha256"},
    )
    events = []
    monkeypatch.setattr(
        fixed_eval, "attest_regular_file", lambda path, label: {"path": str(path)}
    )
    monkeypatch.setattr(
        fixed_eval, "validate_checkpoint_bundle", lambda *args, **kwargs: bundle
    )
    monkeypatch.setattr(fixed_eval, "bind_training_runtime", lambda root: None)
    monkeypatch.delattr(
        checkpoint_eval, "_schema13_xpid_claims_intent", raising=True
    )

    def forbidden(name):
        def fail(*args, **kwargs):
            events.append(name)
            raise AssertionError(f"{name} ran after forward V20 intent")

        return fail

    monkeypatch.setattr(
        checkpoint_eval,
        "dispatch_schema13_completed_bundle",
        forbidden("schema13_dispatch_without_classifier"),
    )
    monkeypatch.setattr(
        checkpoint_eval,
        "_load_flags_from_validated_config_bytes",
        forbidden("load_flags"),
    )
    monkeypatch.setattr(
        fixed_eval, "_set_deterministic_seed", forbidden("rng_seed")
    )
    monkeypatch.setattr(torch, "load", forbidden("tensor_load"))

    with pytest.raises(RuntimeError, match="lacks schema-13 lexical classifier"):
        fixed_eval.evaluate(
            SimpleNamespace(
                checkpoint_dir=checkpoint_dir,
                training_source_root=None,
                source_manifest=None,
                confirmation_profile="v19-300k",
                output_dir=tmp_path / "must-not-exist",
            )
        )

    assert events == []
    assert not (tmp_path / "must-not-exist").exists()


@pytest.mark.parametrize(
    "schema_line",
    [
        pytest.param(b"voc_gate_policy_schema_version: 5\n", id="wrong-schema"),
        pytest.param(b"", id="missing-schema"),
    ],
)
def test_malformed_v18_prefix_routes_to_schema11_before_fixed_downstream(
    monkeypatch, tmp_path, schema_line
):
    import evaluate_dynamic_imitation as checkpoint_eval

    checkpoint_dir = tmp_path / "malformed-v18-prefix"
    checkpoint_dir.mkdir()
    config_payload = schema_line + (
        b"xpid: enduro-voc-v18-orthocd-adam-eps25-malformed-stage\n"
    )
    (checkpoint_dir / "config_c.yaml").write_bytes(config_payload)
    config_digest = hashlib.sha256(config_payload).hexdigest()
    source_root = Path(fixed_eval.__file__).resolve().parent
    bundle = SimpleNamespace(
        source_root=source_root,
        marker={"status": "complete"},
        file_hashes={"config_c.yaml": config_digest},
        source_manifest={"path": source_root / "source.sha256"},
    )
    output_dir = tmp_path / "must-not-exist"
    events = []
    monkeypatch.setattr(
        fixed_eval, "attest_regular_file", lambda path, label: {"path": str(path)}
    )
    monkeypatch.setattr(
        fixed_eval, "validate_checkpoint_bundle", lambda *args, **kwargs: bundle
    )
    monkeypatch.setattr(fixed_eval, "bind_training_runtime", lambda root: None)

    def forbidden(name):
        def fail(*args, **kwargs):
            events.append(name)
            raise AssertionError(f"{name} ran before schema-11 validation")

        return fail

    monkeypatch.setattr(
        checkpoint_eval,
        "dispatch_schema10_completed_bundle",
        forbidden("schema10_dispatch"),
    )
    monkeypatch.setattr(
        checkpoint_eval,
        "_load_flags_from_validated_config_bytes",
        forbidden("byte_bound_load_flags"),
    )
    monkeypatch.setattr(torch, "load", forbidden("downstream_tensor_load"))
    monkeypatch.setattr(
        fixed_eval,
        "validate_behavioral_training_data",
        forbidden("behavioral_data"),
    )
    monkeypatch.setattr(fixed_eval, "run_fixed_rollouts", forbidden("rollout"))

    with pytest.raises(
        ValueError,
        match="dedicated schema-11 validation requires exact Python integer",
    ):
        fixed_eval.evaluate(
            SimpleNamespace(
                checkpoint_dir=checkpoint_dir,
                training_source_root=None,
                source_manifest=None,
                confirmation_profile="v17-300k",
                output_dir=output_dir,
            )
        )

    assert events == []
    assert not output_dir.exists()


@pytest.mark.parametrize(
    "schema_line",
    [
        pytest.param(b"voc_gate_policy_schema_version: 5\n", id="wrong-schema"),
        pytest.param(b"", id="missing-schema"),
    ],
)
def test_malformed_v19_prefix_routes_schema12_before_fixed_downstream(
    monkeypatch, tmp_path, schema_line
):
    import evaluate_dynamic_imitation as checkpoint_eval

    checkpoint_dir = tmp_path / "malformed-v19-prefix"
    checkpoint_dir.mkdir()
    config_payload = schema_line + (
        b"xpid: enduro-voc-v19-tau1-orthocd-adam-eps25-malformed-stage\n"
    )
    (checkpoint_dir / "config_c.yaml").write_bytes(config_payload)
    config_digest = hashlib.sha256(config_payload).hexdigest()
    source_root = Path(fixed_eval.__file__).resolve().parent
    bundle = SimpleNamespace(
        source_root=source_root,
        marker={"status": "complete"},
        file_hashes={"config_c.yaml": config_digest},
        source_manifest={"path": source_root / "source.sha256"},
    )
    output_dir = tmp_path / "must-not-exist"
    events = []
    monkeypatch.setattr(
        fixed_eval, "attest_regular_file", lambda path, label: {"path": str(path)}
    )
    monkeypatch.setattr(
        fixed_eval, "validate_checkpoint_bundle", lambda *args, **kwargs: bundle
    )
    monkeypatch.setattr(fixed_eval, "bind_training_runtime", lambda root: None)

    def reject(*args, **kwargs):
        events.append("schema12_dispatch")
        raise ValueError("invalid schema-12 intent")

    def forbidden(name):
        def fail(*args, **kwargs):
            events.append(name)
            raise AssertionError(f"{name} ran before schema-12 validation")

        return fail

    monkeypatch.setattr(
        checkpoint_eval, "dispatch_schema12_completed_bundle", reject
    )
    monkeypatch.setattr(
        checkpoint_eval,
        "_load_flags_from_validated_config_bytes",
        forbidden("byte_bound_load_flags"),
    )
    monkeypatch.setattr(torch, "load", forbidden("downstream_tensor_load"))
    monkeypatch.setattr(
        fixed_eval,
        "validate_behavioral_training_data",
        forbidden("behavioral_data"),
    )
    monkeypatch.setattr(fixed_eval, "run_fixed_rollouts", forbidden("rollout"))

    with pytest.raises(ValueError, match="invalid schema-12 intent"):
        fixed_eval.evaluate(
            SimpleNamespace(
                checkpoint_dir=checkpoint_dir,
                training_source_root=None,
                source_manifest=None,
                confirmation_profile="v18-300k",
                output_dir=output_dir,
            )
        )

    assert events == ["schema12_dispatch"]
    assert not output_dir.exists()


def test_schema8_checkpoint_rejects_v16_profile_before_downstream(
    monkeypatch, tmp_path
):
    import evaluate_dynamic_imitation as checkpoint_eval

    checkpoint_dir = tmp_path / "schema8-as-v16"
    checkpoint_dir.mkdir()
    config_payload = (
        "voc_gate_policy_schema_version: 8\n"
        f"xpid: {fixed_eval.V15_PRIMARY_XPID}\n"
    ).encode("utf-8")
    (checkpoint_dir / "config_c.yaml").write_bytes(config_payload)
    source_root = Path(fixed_eval.__file__).resolve().parent
    bundle = SimpleNamespace(
        source_root=source_root,
        marker={"status": "complete"},
        file_hashes={
            "config_c.yaml": hashlib.sha256(config_payload).hexdigest()
        },
        source_manifest={"path": source_root / "source.sha256"},
    )
    events = []
    monkeypatch.setattr(
        fixed_eval, "attest_regular_file", lambda path, label: {"path": str(path)}
    )
    monkeypatch.setattr(
        fixed_eval, "validate_checkpoint_bundle", lambda *args, **kwargs: bundle
    )
    monkeypatch.setattr(fixed_eval, "bind_training_runtime", lambda root: None)
    monkeypatch.setattr(
        checkpoint_eval,
        "dispatch_schema9_completed_bundle",
        lambda *args, **kwargs: None,
    )

    def reject_as_schema9(*args, **kwargs):
        events.append("schema9_dedicated")
        raise ValueError("v16-300k checkpoint is not a completed schema-9 bundle")

    def forbidden(name):
        def fail(*args, **kwargs):
            events.append(name)
            raise AssertionError(f"{name} ran before reciprocal rejection")

        return fail

    monkeypatch.setattr(fixed_eval, "validate_v16_final_bundle", reject_as_schema9)
    monkeypatch.setattr(
        checkpoint_eval,
        "_load_flags_from_validated_config_bytes",
        forbidden("flags"),
    )
    monkeypatch.setattr(torch, "load", forbidden("tensor_load"))

    with pytest.raises(ValueError, match="not a completed schema-9"):
        fixed_eval.evaluate(
            SimpleNamespace(
                checkpoint_dir=checkpoint_dir,
                training_source_root=None,
                source_manifest=None,
                confirmation_profile="v16-300k",
                output_dir=tmp_path / "output",
            )
        )

    assert events == ["schema9_dedicated"]
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "legacy_profile",
    [
        "v7-200k",
        "v10-300k",
        "v11-300k",
        "v12-300k",
        "v13-300k",
        "v14-300k",
    ],
)
def test_schema8_checkpoint_rejects_legacy_fixed_profile_before_downstream(
    monkeypatch, tmp_path, legacy_profile
):
    import evaluate_dynamic_imitation as checkpoint_eval

    checkpoint_dir = tmp_path / "schema8"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "config_c.yaml").write_text(
        "voc_gate_policy_schema_version: 8\n"
        f"xpid: {fixed_eval.V15_PRIMARY_XPID}\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "must-not-exist"
    source_root = Path(fixed_eval.__file__).resolve().parent
    bundle = SimpleNamespace(
        source_root=source_root,
        marker={"status": "complete"},
        file_hashes={
            "config_c.yaml": hashlib.sha256(
                (checkpoint_dir / "config_c.yaml").read_bytes()
            ).hexdigest()
        },
        source_manifest={"path": source_root / "source.sha256"},
    )
    events = []
    monkeypatch.setattr(
        fixed_eval, "attest_regular_file", lambda path, label: {"path": str(path)}
    )
    monkeypatch.setattr(
        fixed_eval, "validate_checkpoint_bundle", lambda *args, **kwargs: bundle
    )
    monkeypatch.setattr(fixed_eval, "bind_training_runtime", lambda root: None)

    def dispatch(*args, **kwargs):
        events.append("schema8_dispatch")
        return {"authoritative_validator": "thinker.util.validate_schema8_final_bundle"}

    def forbidden(name):
        def fail(*args, **kwargs):
            events.append(name)
            raise AssertionError(f"{name} ran before cross-profile rejection")

        return fail

    monkeypatch.setattr(
        checkpoint_eval, "dispatch_schema8_completed_bundle", dispatch
    )
    if legacy_profile == "v14-300k":
        def reject_schema8_as_v14(*args, **kwargs):
            events.append("schema7_validation")
            raise ValueError("schema-8 checkpoint is not schema-7")

        monkeypatch.setattr(
            fixed_eval, "validate_v14_final_bundle", reject_schema8_as_v14
        )
    monkeypatch.setattr(checkpoint_eval, "_load_flags", forbidden("load_flags"))
    monkeypatch.setattr(
        checkpoint_eval, "resolve_evaluation_spec", forbidden("live_probe")
    )
    monkeypatch.setattr(torch, "load", forbidden("downstream_tensor_load"))
    monkeypatch.setattr(
        fixed_eval, "validate_behavioral_training_data", forbidden("data")
    )
    monkeypatch.setattr(fixed_eval, "run_fixed_rollouts", forbidden("rollout"))

    expected_error = (
        "not schema-7"
        if legacy_profile == "v14-300k"
        else "eligible only for fixed profile v15"
    )
    with pytest.raises(ValueError, match=expected_error):
        fixed_eval.evaluate(
            SimpleNamespace(
                checkpoint_dir=checkpoint_dir,
                training_source_root=None,
                source_manifest=None,
                confirmation_profile=legacy_profile,
                output_dir=output_dir,
            )
        )

    assert events == [
        "schema7_validation"
        if legacy_profile == "v14-300k"
        else "schema8_dispatch"
    ]
    assert not output_dir.exists()


def test_legacy_bound_public_module_without_schema8_dispatch_still_reaches_v14_route(
    monkeypatch, tmp_path
):
    import evaluate_dynamic_imitation as checkpoint_eval

    source_root = Path(fixed_eval.__file__).resolve().parent
    bundle = SimpleNamespace(
        source_root=source_root,
        marker={"status": "complete"},
        file_hashes={},
        source_manifest={"path": source_root / "source.sha256"},
    )
    events = []
    monkeypatch.setattr(
        fixed_eval, "attest_regular_file", lambda path, label: {"path": str(path)}
    )
    monkeypatch.setattr(
        fixed_eval, "validate_checkpoint_bundle", lambda *args, **kwargs: bundle
    )
    monkeypatch.setattr(fixed_eval, "bind_training_runtime", lambda root: None)
    monkeypatch.delattr(
        checkpoint_eval, "dispatch_schema8_completed_bundle", raising=True
    )

    def reach_v14(*args, **kwargs):
        events.append("legacy_v14_validation")
        raise ValueError("legacy v14 sentinel")

    monkeypatch.setattr(fixed_eval, "validate_v14_final_bundle", reach_v14)

    with pytest.raises(ValueError, match="legacy v14 sentinel"):
        fixed_eval.evaluate(
            SimpleNamespace(
                checkpoint_dir=tmp_path / "v14",
                training_source_root=None,
                source_manifest=None,
                confirmation_profile="v14-300k",
                output_dir=tmp_path / "output",
            )
        )

    assert events == ["legacy_v14_validation"]


def test_v15_requested_without_schema8_dispatch_fails_before_downstream(
    monkeypatch, tmp_path
):
    import evaluate_dynamic_imitation as checkpoint_eval

    source_root = Path(fixed_eval.__file__).resolve().parent
    bundle = SimpleNamespace(
        source_root=source_root,
        marker={"status": "complete"},
        file_hashes={},
        source_manifest={"path": source_root / "source.sha256"},
    )
    events = []
    monkeypatch.setattr(
        fixed_eval, "attest_regular_file", lambda path, label: {"path": str(path)}
    )
    monkeypatch.setattr(
        fixed_eval, "validate_checkpoint_bundle", lambda *args, **kwargs: bundle
    )
    monkeypatch.setattr(fixed_eval, "bind_training_runtime", lambda root: None)
    monkeypatch.delattr(
        checkpoint_eval, "dispatch_schema8_completed_bundle", raising=True
    )

    def forbidden(name):
        def fail(*args, **kwargs):
            events.append(name)
            raise AssertionError(f"{name} ran without schema-8 dispatch")

        return fail

    monkeypatch.setattr(checkpoint_eval, "_load_flags", forbidden("load_flags"))
    monkeypatch.setattr(torch, "load", forbidden("downstream_tensor_load"))

    with pytest.raises(RuntimeError, match="lacks schema-8 dispatch"):
        fixed_eval.evaluate(
            SimpleNamespace(
                checkpoint_dir=tmp_path / "v15",
                training_source_root=None,
                source_manifest=None,
                confirmation_profile="v15-300k",
                output_dir=tmp_path / "output",
            )
        )

    assert events == []


@pytest.mark.parametrize("bound_hash", [None, "not-a-sha256"])
def test_v15_requires_bound_config_evidence_before_dispatch_or_downstream(
    monkeypatch, tmp_path, bound_hash
):
    import evaluate_dynamic_imitation as checkpoint_eval

    checkpoint_dir = tmp_path / "v15"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "config_c.yaml").write_text(
        "voc_gate_policy_schema_version: 8\n",
        encoding="utf-8",
    )
    source_root = Path(fixed_eval.__file__).resolve().parent
    bundle = SimpleNamespace(
        source_root=source_root,
        marker={"status": "complete"},
        file_hashes=(
            {} if bound_hash is None else {"config_c.yaml": bound_hash}
        ),
        source_manifest={"path": source_root / "source.sha256"},
    )
    events = []
    monkeypatch.setattr(
        fixed_eval, "attest_regular_file", lambda path, label: {"path": str(path)}
    )
    monkeypatch.setattr(
        fixed_eval, "validate_checkpoint_bundle", lambda *args, **kwargs: bundle
    )
    monkeypatch.setattr(fixed_eval, "bind_training_runtime", lambda root: None)

    def forbidden(name):
        def fail(*args, **kwargs):
            events.append(name)
            raise AssertionError(f"{name} ran without bound config evidence")

        return fail

    monkeypatch.setattr(
        checkpoint_eval,
        "dispatch_schema8_completed_bundle",
        forbidden("schema8_dispatch"),
    )
    monkeypatch.setattr(checkpoint_eval, "_load_flags", forbidden("load_flags"))
    monkeypatch.setattr(
        checkpoint_eval, "resolve_evaluation_spec", forbidden("live_probe")
    )
    monkeypatch.setattr(torch, "load", forbidden("downstream_tensor_load"))

    with pytest.raises(RuntimeError, match="config (digest|evidence)"):
        fixed_eval.evaluate(
            SimpleNamespace(
                checkpoint_dir=checkpoint_dir,
                training_source_root=None,
                source_manifest=None,
                confirmation_profile="v15-300k",
                output_dir=tmp_path / "output",
            )
        )

    assert events == []


def _actual_schema13_fixed_flags(monkeypatch, tmp_path):
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


def test_fixed_schema13_bound_flags_use_actual_public_validated_byte_loader(
    monkeypatch, tmp_path
):
    created = _actual_schema13_fixed_flags(monkeypatch, tmp_path)
    payload = yaml.safe_dump(vars(created), sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()

    loaded = fixed_eval._load_flags_from_bound_config_bytes(
        evaluation,
        Path(created.ckpdir),
        payload,
        digest,
        byte_loader=evaluation._load_flags_from_validated_config_bytes,
    )

    assert vars(loaded) == vars(created)


def test_fixed_schema13_bound_flags_reject_malformed_before_legacy_loader(
    monkeypatch, tmp_path
):
    created = _actual_schema13_fixed_flags(monkeypatch, tmp_path)
    malformed = dict(vars(created))
    malformed.pop("voc_gate_policy_schema_version")
    payload = yaml.safe_dump(malformed, sort_keys=True).encode("utf-8")
    calls = []

    def forbidden_create_flags(*args, **kwargs):
        calls.append("create_flags")
        raise AssertionError("fixed schema-13 intent reached legacy loading")

    monkeypatch.setattr(util, "create_flags", forbidden_create_flags)
    with pytest.raises(ValueError, match="schema|surface|xpid"):
        fixed_eval._load_flags_from_bound_config_bytes(
            evaluation,
            Path(created.ckpdir),
            payload,
            hashlib.sha256(payload).hexdigest(),
            byte_loader=evaluation._load_flags_from_validated_config_bytes,
        )

    assert calls == []


@pytest.mark.parametrize("mutation", [None, "replace", "delete"])
def test_frozen_public_module_loads_only_bound_config_bytes(
    monkeypatch, tmp_path, mutation
):
    import evaluate_dynamic_imitation as checkpoint_eval

    checkpoint_dir = tmp_path / "legacy"
    checkpoint_dir.mkdir()
    config_payload = (
        "voc_gate_policy_schema_version: 6\n"
        "xpid: legacy-v13\n"
    ).encode("utf-8")
    config_path = checkpoint_dir / "config_c.yaml"
    config_path.write_bytes(config_payload)
    config_digest = hashlib.sha256(config_payload).hexdigest()
    source_root = Path(fixed_eval.__file__).resolve().parent
    bundle = SimpleNamespace(
        source_root=source_root,
        marker={"status": "complete"},
        file_hashes={"config_c.yaml": config_digest},
        source_manifest={"path": source_root / "source.sha256"},
    )
    events = []
    private_paths = []
    monkeypatch.setattr(
        fixed_eval, "attest_regular_file", lambda path, label: {"path": str(path)}
    )
    monkeypatch.setattr(
        fixed_eval, "validate_checkpoint_bundle", lambda *args, **kwargs: bundle
    )
    monkeypatch.setattr(fixed_eval, "bind_training_runtime", lambda root: None)
    monkeypatch.delattr(
        checkpoint_eval, "_load_flags_from_validated_config_bytes", raising=True
    )

    def frozen_loader(private_checkpoint):
        private_checkpoint = Path(private_checkpoint)
        private_paths.append(private_checkpoint)
        assert private_checkpoint != checkpoint_dir
        assert (private_checkpoint / "config_c.yaml").read_bytes() == config_payload
        if mutation == "replace":
            config_path.write_text(
                "voc_gate_policy_schema_version: 8\n",
                encoding="utf-8",
            )
        elif mutation == "delete":
            config_path.unlink()
        return SimpleNamespace(ckpdir=str(private_checkpoint))

    def live_probe(flags, **kwargs):
        events.append("live_probe")
        assert flags.ckpdir == str(checkpoint_dir.resolve())
        assert not any(str(path) in str(value) for value in vars(flags).values() for path in private_paths)
        raise RuntimeError("live probe sentinel")

    monkeypatch.setattr(checkpoint_eval, "_load_flags", frozen_loader)
    monkeypatch.setattr(checkpoint_eval, "resolve_evaluation_spec", live_probe)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("tensor load ran")))

    args = SimpleNamespace(
        checkpoint_dir=checkpoint_dir,
        training_source_root=None,
        source_manifest=None,
        confirmation_profile="v13-300k",
        output_dir=tmp_path / "output",
        seed_base=4000,
        num_seeds=1,
        expected_game_id=None,
    )
    if mutation is None:
        with pytest.raises(RuntimeError, match="live probe sentinel"):
            fixed_eval.evaluate(args)
        assert events == ["live_probe"]
    else:
        with pytest.raises((FileNotFoundError, RuntimeError), match="config"):
            fixed_eval.evaluate(args)
        assert events == []
    assert private_paths and all(not path.exists() for path in private_paths)


@pytest.mark.parametrize(
    ("profile", "total_steps", "evaluation_mode"),
    [
        ("v7-200k", 200_000, "fixed_200k_confirmation"),
        ("v10-300k", 300_000, "fixed_300k_confirmation"),
        ("v11-300k", 300_000, "fixed_v11_300k_confirmation"),
        ("v12-300k", 300_000, "fixed_v12_300k_confirmation"),
        ("v13-300k", 300_000, "fixed_v13_300k_confirmation"),
        ("v14-300k", 300_000, "fixed_v14_300k_confirmation"),
        ("v15-300k", 300_000, "fixed_v15_300k_confirmation"),
    ],
)
def test_fixed_profiles_accept_only_their_exact_horizon(
    profile, total_steps, evaluation_mode
):
    confirmation = _exact_protocol_call(profile, total_steps)
    assert confirmation["confirmation_profile"] == profile
    assert confirmation["training_total_steps"] == total_steps
    assert confirmation["evaluation_mode"] == evaluation_mode
    assert confirmation["confirmation_eligible"] is True
    expected_execution = profile in (
        "v12-300k",
        "v13-300k",
        "v14-300k",
        "v15-300k",
    )
    assert (
        confirmation["voc_gate_epsilon_greedy_execution"]
        is expected_execution
    )
    assert set(confirmation["normalized_execution_identity"].values()) == {
        expected_execution
    }
    resolved = confirmation["resolved_profile_identity"]
    assert resolved["config"]["total_steps"] == total_steps
    assert resolved["actor_checkpoint"]["total_steps"] == total_steps
    assert resolved["model_checkpoint"]["total_steps"] == total_steps
    if profile == "v10-300k":
        for source in ("config", "actor_checkpoint", "model_checkpoint"):
            assert resolved[source]["base_seed"] == 2
            assert resolved[source]["schedule_total_steps"] == 100_000_000
            assert resolved[source]["voc_gate_param_align"] is True
            assert resolved[source]["voc_gate_param_align_coef"] == 1.0
        assert resolved["actor_checkpoint"][
            "voc_gate_policy_schema_version"
        ] == 3
        assert resolved["actor_checkpoint_validation"] == {
            "voc_gate_policy_schema_version": 3,
            "voc_gate_param_align": True,
            "voc_gate_param_align_coef": 1.0,
        }
    elif profile in ("v11-300k", "v12-300k"):
        is_v12 = profile == "v12-300k"
        for source in ("config", "actor_checkpoint", "model_checkpoint"):
            assert resolved[source] == {
                "total_steps": 300_000,
                "base_seed": 4 if is_v12 else 3,
                "schedule_total_steps": 100_000_000,
                "dynamic_voc_mode": "control",
                "voc_dedicated_gate": True,
                "voc_soft_q_bce_gate": True,
                "voc_gate_temperature": 1.0,
                "voc_gate_q_temperature": 0.05,
                "voc_gate_param_align": False,
                "voc_gate_param_align_coef": 1.0,
                "voc_gate_exact_projection": True,
                **(
                    {
                        "voc_eval_stochastic": True,
                        "voc_train_epsilon": 0.02,
                        "voc_gate_epsilon_greedy_execution": True,
                    }
                    if is_v12
                    else {}
                ),
                "ckp": False,
                "preload": "",
                "preload_actor": "",
                "voc_parent_checkpoint": "",
                **(
                    {"voc_gate_policy_schema_version": 5 if is_v12 else 4}
                    if source == "actor_checkpoint"
                    else {}
                ),
            }
        assert resolved["actor_checkpoint_validation"] == {
            "voc_gate_policy_schema_version": 5 if is_v12 else 4,
            "voc_gate_param_align": False,
            "voc_gate_param_align_coef": 1.0,
            "voc_gate_exact_projection": True,
            **(
                {"voc_gate_epsilon_greedy_execution": True}
                if is_v12
                else {}
            ),
        }
        assert resolved["fresh_actor_provenance"]["actor_checkpoint"] == (
            resolved["fresh_actor_provenance"][
                "actor_checkpoint_validation"
            ]
        )
        assert resolved["terminal_exact_projection"] == {
            "gate_head_keys": [
                "voc_gate_head.weight",
                "voc_gate_head.bias",
            ],
            "gate_dtype": "torch.float32",
            "ema_q_dtype": "torch.float32",
            "voc_gate_temperature": 1.0,
            "voc_gate_q_temperature": 0.05,
            "affine_scale": 20.0,
            "weight_torch_equal": True,
            "bias_torch_equal": True,
        }
    elif profile in ("v13-300k", "v14-300k", "v15-300k"):
        is_v14 = profile == "v14-300k"
        is_v15 = profile == "v15-300k"
        expected_identity = {
            "total_steps": 300_000,
            "xpid": (
                fixed_eval.V15_PRIMARY_XPID
                if is_v15
                else (
                    fixed_eval.V14_PRIMARY_XPID
                    if is_v14
                    else fixed_eval.V13_PRIMARY_XPID
                )
            ),
            "base_seed": 5,
            "schedule_total_steps": 100_000_000,
            "model_warm_up_n": 10_000,
            "actor_unroll_len": 201,
            "voc_actor_policy_bundle_schema_version": 1,
            "voc_actor_policy_ray_max_restarts": 0,
            "voc_actor_policy_ray_max_task_retries": 0,
            "ppo_k": 1,
            "self_play_n": 1,
            "env_n": 16,
            "actor_batch_size": 16,
            "dynamic_voc_mode": "control",
            "voc_dedicated_gate": True,
            "voc_soft_q_bce_gate": True,
            "voc_eval_stochastic": True,
            "use_wandb": True,
            "voc_actor_policy_version_barrier": True,
            "voc_actor_policy_barrier_runtime": True,
            "float16": True,
            "model_float16": False,
            "parallel_actor": True,
            "voc_gate_temperature": 1.0,
            "voc_gate_q_temperature": 0.05,
            "voc_train_epsilon": 0.02,
            "voc_gate_execution_epsilon": 0.25,
            "voc_actor_policy_barrier_timeout_s": 120.0,
            "actor_amp_init_scale": 32.0,
            "voc_gate_param_align": False,
            "voc_gate_param_align_coef": 1.0,
            "voc_gate_exact_projection": True,
            "voc_gate_epsilon_greedy_execution": True,
            "ckp": False,
            "preload": "",
            "preload_actor": "",
            "voc_parent_checkpoint": "",
            **(
                {"voc_model_input_seal_schema_version": 1}
                if is_v14 or is_v15
                else {}
            ),
        }
        for source in ("config", "actor_checkpoint", "model_checkpoint"):
            expected = dict(expected_identity)
            expected["voc_gate_policy_schema_version"] = (
                8 if is_v15 else (7 if is_v14 else 6)
            )
            assert resolved[source] == expected
        assert resolved["actor_checkpoint_validation"] == {
            "voc_gate_policy_schema_version": (
                8 if is_v15 else (7 if is_v14 else 6)
            ),
            "voc_gate_param_align": False,
            "voc_gate_param_align_coef": 1.0,
            "voc_gate_exact_projection": True,
            "voc_gate_epsilon_greedy_execution": True,
            **(
                {"voc_model_input_seal_schema_version": 1}
                if is_v14 or is_v15
                else {}
            ),
        }
        bundle_key = (
            "schema8_final_bundle"
            if is_v15
            else ("schema7_final_bundle" if is_v14 else "schema6_final_bundle")
        )
        assert resolved[bundle_key]["actor_policy"][
            "voc_actor_policy_terminal"
        ] is True
        assert resolved[bundle_key]["resolved_identity"][
            "key_count"
        ] == (229 if is_v14 or is_v15 else 228)
        assert confirmation["training_gate_soft_epsilon"] == 0.02
        assert confirmation["training_gate_execution_epsilon"] == 0.25
        assert confirmation["runtime_gate_soft_epsilon"] == 0.0
        assert confirmation["runtime_gate_execution_epsilon"] == 0.0
        assert confirmation["runtime_actor_policy_barrier_wait"] is False
        if is_v14 or is_v15:
            assert set(
                confirmation["normalized_model_input_seal_identity"].values()
            ) == {1}
            assert confirmation[
                "training_model_input_seal_schema_version"
            ] == 1
            assert confirmation[
                "runtime_model_input_seal_coordination"
            ] is False
        if is_v15:
            assert resolved["voc_q_regression_loss"] == "half_squared_td"

    diagnostic = _exact_protocol_call(profile, total_steps, diagnostic=True)
    assert diagnostic["confirmation_profile"] == profile
    assert diagnostic["evaluation_mode"] == "diagnostic"
    assert diagnostic["confirmation_eligible"] is False


def test_v16_fixed_profile_accepts_only_exact_schema9_primary_identity():
    confirmation = _exact_protocol_call("v16-300k", 300_000)

    assert confirmation["confirmation_profile"] == "v16-300k"
    assert confirmation["evaluation_mode"] == "fixed_v16_300k_confirmation"
    assert confirmation["confirmation_eligible"] is True
    resolved = confirmation["resolved_profile_identity"]
    assert resolved["config"]["xpid"] == fixed_eval.V16_PRIMARY_XPID
    assert resolved["config"]["voc_gate_policy_schema_version"] == 9
    assert resolved["schema9_final_bundle"]["resolved_identity"][
        "stage"
    ] == fixed_eval.V16_PRIMARY_STAGE
    assert resolved["voc_q_regression_loss"] == "half_squared_td"
    assert resolved["voc_q_reconstruction"] == fixed_eval.V16_Q_RECONSTRUCTION
    assert "schema8_final_bundle" not in resolved


def test_v16_fixed_profile_rejects_wire_stage_and_schema8_evidence():
    inputs = _fixed_protocol_inputs(300_000, profile="v16-300k")
    wire_stage = (
        "enduro-voc-v16-commonmode-eps25-sentinel-wire1200",
        1,
        1_200,
        512,
        41,
        False,
    )
    common_kwargs = {
        "confirmation_profile": "v16-300k",
        "seeds": range(
            fixed_eval.DEFAULT_SEED_BASE,
            fixed_eval.DEFAULT_SEED_BASE + fixed_eval.DEFAULT_NUM_SEEDS,
        ),
        "real_steps_per_seed": fixed_eval.DEFAULT_REAL_STEPS_PER_SEED,
        "calibration_unroll": fixed_eval.DEFAULT_CALIBRATION_UNROLL,
        "diagnostic": False,
    }

    with pytest.raises(ValueError, match="exact primary stage"):
        fixed_eval._require_fixed_protocol(
            *inputs,
            **common_kwargs,
            schema9_bundle_validation=_v16_bundle_evidence(stage=wire_stage),
        )
    with pytest.raises(ValueError, match="schema-9|resolved identity|validator"):
        fixed_eval._require_fixed_protocol(
            *inputs,
            **common_kwargs,
            schema9_bundle_validation=_v15_bundle_evidence(),
        )


def test_v17_fixed_profile_accepts_exact_primary_only_with_schema9_shape():
    confirmation = _exact_protocol_call("v17-300k", 300_000)

    assert confirmation["confirmation_profile"] == "v17-300k"
    assert confirmation["evaluation_mode"] == "fixed_v17_300k_confirmation"
    assert confirmation["confirmation_eligible"] is True
    resolved = confirmation["resolved_profile_identity"]
    assert resolved["config"]["xpid"] == fixed_eval.V17_PRIMARY_XPID
    assert resolved["config"]["voc_gate_policy_schema_version"] == 10
    assert set(resolved["schema10_final_bundle"]) == set(
        _v16_bundle_evidence()
    )
    assert set(
        resolved["schema10_final_bundle"]["resolved_identity"]
    ) == set(_v16_bundle_evidence()["resolved_identity"])
    assert resolved["schema10_final_bundle"]["resolved_identity"][
        "stage"
    ] == fixed_eval.V17_PRIMARY_STAGE
    assert resolved["voc_q_regression_loss"] == "smooth_l1_beta1"
    assert resolved["voc_q_reconstruction"] == fixed_eval.V16_Q_RECONSTRUCTION
    assert "schema9_final_bundle" not in resolved


def test_v17_fixed_profile_rejects_nonprimary_stage_and_schema9_evidence():
    inputs = _fixed_protocol_inputs(300_000, profile="v17-300k")
    wire_stage = (
        "enduro-voc-v17-huber-common-eps25-sentinel-wire1200",
        1,
        1_200,
        512,
        41,
        False,
    )
    common_kwargs = {
        "confirmation_profile": "v17-300k",
        "seeds": range(
            fixed_eval.DEFAULT_SEED_BASE,
            fixed_eval.DEFAULT_SEED_BASE + fixed_eval.DEFAULT_NUM_SEEDS,
        ),
        "real_steps_per_seed": fixed_eval.DEFAULT_REAL_STEPS_PER_SEED,
        "calibration_unroll": fixed_eval.DEFAULT_CALIBRATION_UNROLL,
        "diagnostic": False,
    }

    with pytest.raises(ValueError, match="exact primary stage"):
        fixed_eval._require_fixed_protocol(
            *inputs,
            **common_kwargs,
            schema10_bundle_validation=_v17_bundle_evidence(stage=wire_stage),
        )
    with pytest.raises(ValueError, match="schema-10|resolved identity|validator"):
        fixed_eval._require_fixed_protocol(
            *inputs,
            **common_kwargs,
            schema10_bundle_validation=_v16_bundle_evidence(),
        )


def test_v18_fixed_profile_accepts_exact_primary_only_with_identity12():
    confirmation = _exact_protocol_call("v18-300k", 300_000)

    assert confirmation["confirmation_profile"] == "v18-300k"
    assert confirmation["evaluation_mode"] == "fixed_v18_300k_confirmation"
    assert confirmation["confirmation_eligible"] is True
    resolved = confirmation["resolved_profile_identity"]
    assert resolved["config"]["xpid"] == fixed_eval.V18_PRIMARY_XPID
    assert resolved["config"]["voc_gate_policy_schema_version"] == 11
    assert set(resolved["schema11_final_bundle"]) == set(
        _v17_bundle_evidence()
    )
    assert set(resolved["schema11_final_bundle"]["resolved_identity"]) == (
        set(_v17_bundle_evidence()["resolved_identity"])
        | {"voc_q_optimizer_coordinates"}
    )
    assert resolved["schema11_final_bundle"]["resolved_identity"][
        "stage"
    ] == fixed_eval.V18_PRIMARY_STAGE
    assert resolved["voc_q_regression_loss"] == "smooth_l1_beta1"
    assert resolved["voc_q_reconstruction"] == fixed_eval.V17_Q_RECONSTRUCTION
    assert resolved["voc_q_optimizer_coordinates"] == (
        "orthonormal_common_difference_adam"
    )
    assert "schema10_final_bundle" not in resolved


def test_v18_fixed_profile_rejects_wire_stage_and_schema10_evidence():
    inputs = _fixed_protocol_inputs(300_000, profile="v18-300k")
    wire_stage = (
        "enduro-voc-v18-orthocd-adam-eps25-sentinel-wire1200",
        1,
        1_200,
        512,
        41,
        False,
    )
    common_kwargs = {
        "confirmation_profile": "v18-300k",
        "seeds": range(
            fixed_eval.DEFAULT_SEED_BASE,
            fixed_eval.DEFAULT_SEED_BASE + fixed_eval.DEFAULT_NUM_SEEDS,
        ),
        "real_steps_per_seed": fixed_eval.DEFAULT_REAL_STEPS_PER_SEED,
        "calibration_unroll": fixed_eval.DEFAULT_CALIBRATION_UNROLL,
        "diagnostic": False,
    }

    with pytest.raises(ValueError, match="exact primary stage"):
        fixed_eval._require_fixed_protocol(
            *inputs,
            **common_kwargs,
            schema11_bundle_validation=_v18_bundle_evidence(stage=wire_stage),
        )
    with pytest.raises(ValueError, match="schema-11|resolved identity|validator"):
        fixed_eval._require_fixed_protocol(
            *inputs,
            **common_kwargs,
            schema11_bundle_validation=_v17_bundle_evidence(),
        )


def test_v19_fixed_profile_accepts_only_exact_schema12_primary_identity():
    confirmation = _exact_protocol_call("v19-300k", 300_000)

    assert confirmation["confirmation_profile"] == "v19-300k"
    assert confirmation["evaluation_mode"] == "fixed_v19_300k_confirmation"
    assert confirmation["confirmation_eligible"] is True
    resolved = confirmation["resolved_profile_identity"]
    v18_resolved = _exact_protocol_call("v18-300k", 300_000)[
        "resolved_profile_identity"
    ]
    assert resolved["config"]["xpid"] == fixed_eval.V19_PRIMARY_XPID
    assert resolved["config"]["voc_gate_policy_schema_version"] == 12
    assert resolved["schema12_final_bundle"]["resolved_identity"]["stage"] == (
        fixed_eval.V19_PRIMARY_STAGE
    )
    assert resolved["schema12_final_bundle"]["resolved_identity"][
        "v12_projection_sha256"
    ] == fixed_eval.V19_V12_PROJECTION_SHA256
    mapped_v19_keys = (set(resolved) - {"schema12_final_bundle"}) | {
        "schema11_final_bundle"
    }
    assert mapped_v19_keys == set(v18_resolved)
    assert set(resolved["schema12_final_bundle"]) == set(
        v18_resolved["schema11_final_bundle"]
    )


def test_v20_fixed_profile_accepts_exact_schema13_telemetry_primary_only():
    confirmation = _exact_protocol_call("v20-300k", 300_000)

    assert confirmation["confirmation_profile"] == "v20-300k"
    assert confirmation["evaluation_mode"] == "fixed_v20_300k_confirmation"
    assert confirmation["confirmation_eligible"] is True
    resolved = confirmation["resolved_profile_identity"]
    assert resolved["config"]["xpid"] == fixed_eval.V20_PRIMARY_XPID
    assert resolved["config"]["voc_gate_policy_schema_version"] == 13
    bundle = resolved["schema13_final_bundle"]
    assert len(bundle) == 19
    assert bundle["resolved_identity"]["stage"] == fixed_eval.V20_PRIMARY_STAGE
    assert bundle["telemetry"]["gate_schema"] == 13
    assert bundle["telemetry"]["manifest_name"] == (
        "voc_telemetry_manifest.json"
    )
    assert set(bundle["actor_policy"]) == (
        fixed_eval.ATOMIC_ACTOR_POLICY_EVIDENCE_FIELDS
    )
    assert "schema12_final_bundle" not in resolved


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("manifest_sha256", "8" * 64),
        ("manifest_size", 4097),
        ("transaction_count", 2),
        ("gate_schema", 12),
    ],
)
def test_v20_fixed_profile_rejects_telemetry_cross_evidence_drift(field, bad):
    inputs = _fixed_protocol_inputs(300_000, profile="v20-300k")
    evidence = _v20_bundle_evidence()
    evidence["telemetry"][field] = bad

    with pytest.raises(ValueError, match="telemetry"):
        fixed_eval._require_fixed_protocol(
            *inputs,
            confirmation_profile="v20-300k",
            seeds=range(
                fixed_eval.DEFAULT_SEED_BASE,
                fixed_eval.DEFAULT_SEED_BASE + fixed_eval.DEFAULT_NUM_SEEDS,
            ),
            real_steps_per_seed=fixed_eval.DEFAULT_REAL_STEPS_PER_SEED,
            calibration_unroll=fixed_eval.DEFAULT_CALIBRATION_UNROLL,
            diagnostic=False,
            schema13_bundle_validation=evidence,
        )


@pytest.mark.parametrize(
    ("surface", "bad"),
    [
        ("config", 0.1),
        ("config", 1),
        ("actor", None),
        ("model", np.float64(1.0)),
        ("validation", True),
    ],
)
def test_v19_fixed_profile_rejects_wrong_or_typed_tau(surface, bad):
    inputs = list(_fixed_protocol_inputs(300_000, profile="v19-300k"))
    if surface == "config":
        inputs[0].voc_gate_target_tau = bad
    elif surface == "actor":
        inputs[1]["flags"]["voc_gate_target_tau"] = bad
    elif surface == "model":
        inputs[2]["flags"]["voc_gate_target_tau"] = bad
    else:
        inputs[3]["voc"]["voc_gate_target_tau"] = bad

    with pytest.raises(ValueError, match="voc_gate_target_tau"):
        fixed_eval._require_fixed_protocol(
            *inputs,
            confirmation_profile="v19-300k",
            seeds=range(
                fixed_eval.DEFAULT_SEED_BASE,
                fixed_eval.DEFAULT_SEED_BASE + fixed_eval.DEFAULT_NUM_SEEDS,
            ),
            real_steps_per_seed=fixed_eval.DEFAULT_REAL_STEPS_PER_SEED,
            calibration_unroll=fixed_eval.DEFAULT_CALIBRATION_UNROLL,
            diagnostic=False,
            schema12_bundle_validation=_v19_bundle_evidence(),
        )


@pytest.mark.parametrize("mismatch", [None, "weight", "bias", "both"])
def test_v19_fixed_separately_enforces_torch_equal_raw_ema(
    tmp_path, mismatch
):
    checkpoint_dir = tmp_path / "v19"
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
                "sha256": _sha(actor_path),
                "size": actor_path.stat().st_size,
            }
        }
    }

    class CheckpointEval:
        @staticmethod
        def _read_stable_single_link_bytes(path, *, label):
            return Path(path).read_bytes()

    if mismatch is None:
        fixed_eval._require_schema12_fixed_ema_online_equality(
            checkpoint_dir, marker, checkpoint_eval=CheckpointEval
        )
    else:
        with pytest.raises(ValueError, match="raw EMA (weight|bias)"):
            fixed_eval._require_schema12_fixed_ema_online_equality(
                checkpoint_dir, marker, checkpoint_eval=CheckpointEval
            )


@pytest.mark.parametrize(
    ("profile", "checkpoint_steps"),
    [
        ("v7-200k", 300_000),
        ("v10-300k", 200_000),
        ("v11-300k", 200_000),
        ("v12-300k", 200_000),
        ("v13-300k", 200_000),
        ("v14-300k", 200_000),
        ("v15-300k", 200_000),
    ],
)
def test_fixed_profiles_reject_cross_profile_horizons(profile, checkpoint_steps):
    with pytest.raises(ValueError, match="checkpoint config total_steps"):
        _exact_protocol_call(profile, checkpoint_steps)


@pytest.mark.parametrize(
    ("profile", "expected", "cross_profile_steps"),
    [
        ("v7-200k", 200_000, 300_000),
        ("v10-300k", 300_000, 200_000),
        ("v11-300k", 300_000, 200_000),
        ("v12-300k", 300_000, 200_000),
        ("v13-300k", 300_000, 200_000),
        ("v14-300k", 300_000, 200_000),
        ("v15-300k", 300_000, 200_000),
    ],
)
@pytest.mark.parametrize("checkpoint_name", ["actor", "model"])
def test_fixed_profiles_reject_cross_profile_embedded_checkpoint_horizon(
    profile, expected, cross_profile_steps, checkpoint_name
):
    inputs = list(_fixed_protocol_inputs(expected, profile=profile))
    checkpoint_index = 1 if checkpoint_name == "actor" else 2
    inputs[checkpoint_index]["flags"]["total_steps"] = cross_profile_steps
    with pytest.raises(
        ValueError,
        match=rf"{checkpoint_name} checkpoint embedded flags total_steps",
    ):
        fixed_eval._require_fixed_protocol(
            *inputs,
            confirmation_profile=profile,
            seeds=range(
                fixed_eval.DEFAULT_SEED_BASE,
                fixed_eval.DEFAULT_SEED_BASE + fixed_eval.DEFAULT_NUM_SEEDS,
            ),
            real_steps_per_seed=fixed_eval.DEFAULT_REAL_STEPS_PER_SEED,
            calibration_unroll=fixed_eval.DEFAULT_CALIBRATION_UNROLL,
            diagnostic=False,
        )


V10_PROFILE_FLAG_MISMATCHES = {
    "base_seed": 3,
    "schedule_total_steps": 99_999_999,
    "voc_gate_param_align": False,
    "voc_gate_param_align_coef": 0.5,
}


def _v10_identity_container(inputs, source):
    if source == "config":
        return inputs[0]
    if source == "actor":
        return inputs[1]["flags"]
    return inputs[2]["flags"]


@pytest.mark.parametrize("source", ["config", "actor", "model"])
@pytest.mark.parametrize(
    ("field", "mismatch"), list(V10_PROFILE_FLAG_MISMATCHES.items())
)
def test_v10_profile_rejects_identity_mismatch(source, field, mismatch):
    inputs = list(_fixed_protocol_inputs(300_000))
    container = _v10_identity_container(inputs, source)
    if isinstance(container, SimpleNamespace):
        setattr(container, field, mismatch)
    else:
        container[field] = mismatch
    with pytest.raises(ValueError, match=field):
        fixed_eval._require_fixed_protocol(
            *inputs,
            confirmation_profile="v10-300k",
            seeds=range(
                fixed_eval.DEFAULT_SEED_BASE,
                fixed_eval.DEFAULT_SEED_BASE + fixed_eval.DEFAULT_NUM_SEEDS,
            ),
            real_steps_per_seed=fixed_eval.DEFAULT_REAL_STEPS_PER_SEED,
            calibration_unroll=fixed_eval.DEFAULT_CALIBRATION_UNROLL,
            diagnostic=False,
        )


@pytest.mark.parametrize("source", ["config", "actor", "model"])
@pytest.mark.parametrize("field", list(V10_PROFILE_FLAG_MISMATCHES))
def test_v10_profile_rejects_missing_identity_field(source, field):
    inputs = list(_fixed_protocol_inputs(300_000))
    container = _v10_identity_container(inputs, source)
    if isinstance(container, SimpleNamespace):
        delattr(container, field)
    else:
        del container[field]
    with pytest.raises(ValueError, match=rf"lacks {field}"):
        fixed_eval._require_fixed_protocol(
            *inputs,
            confirmation_profile="v10-300k",
            seeds=range(
                fixed_eval.DEFAULT_SEED_BASE,
                fixed_eval.DEFAULT_SEED_BASE + fixed_eval.DEFAULT_NUM_SEEDS,
            ),
            real_steps_per_seed=fixed_eval.DEFAULT_REAL_STEPS_PER_SEED,
            calibration_unroll=fixed_eval.DEFAULT_CALIBRATION_UNROLL,
            diagnostic=False,
        )


@pytest.mark.parametrize("source", ["config", "actor", "model", "validation"])
def test_v10_profile_rejects_nextafter_alignment_coef(source):
    inputs = list(_fixed_protocol_inputs(300_000))
    nextafter = np.nextafter(np.float64(1.0), np.float64(2.0))
    if source == "validation":
        inputs[3]["voc"]["voc_gate_param_align_coef"] = nextafter
    else:
        container = _v10_identity_container(inputs, source)
        if isinstance(container, SimpleNamespace):
            setattr(container, "voc_gate_param_align_coef", nextafter)
        else:
            container["voc_gate_param_align_coef"] = nextafter
    with pytest.raises(ValueError, match="voc_gate_param_align_coef"):
        fixed_eval._require_fixed_protocol(
            *inputs,
            confirmation_profile="v10-300k",
            seeds=range(
                fixed_eval.DEFAULT_SEED_BASE,
                fixed_eval.DEFAULT_SEED_BASE + fixed_eval.DEFAULT_NUM_SEEDS,
            ),
            real_steps_per_seed=fixed_eval.DEFAULT_REAL_STEPS_PER_SEED,
            calibration_unroll=fixed_eval.DEFAULT_CALIBRATION_UNROLL,
            diagnostic=False,
        )


@pytest.mark.parametrize("missing", [True, False])
@pytest.mark.parametrize("source", ["checkpoint", "validation"])
def test_v10_profile_requires_actor_gate_schema_three(source, missing):
    inputs = list(_fixed_protocol_inputs(300_000))
    container = inputs[1] if source == "checkpoint" else inputs[3]["voc"]
    if missing:
        del container["voc_gate_policy_schema_version"]
    else:
        container["voc_gate_policy_schema_version"] = 2
    with pytest.raises(ValueError, match="voc_gate_policy_schema_version"):
        fixed_eval._require_fixed_protocol(
            *inputs,
            confirmation_profile="v10-300k",
            seeds=range(
                fixed_eval.DEFAULT_SEED_BASE,
                fixed_eval.DEFAULT_SEED_BASE + fixed_eval.DEFAULT_NUM_SEEDS,
            ),
            real_steps_per_seed=fixed_eval.DEFAULT_REAL_STEPS_PER_SEED,
            calibration_unroll=fixed_eval.DEFAULT_CALIBRATION_UNROLL,
            diagnostic=False,
        )


@pytest.mark.parametrize("field", ["voc_gate_param_align", "voc_gate_param_align_coef"])
@pytest.mark.parametrize("missing", [True, False])
def test_v10_profile_requires_validated_alignment_identity(field, missing):
    inputs = list(_fixed_protocol_inputs(300_000))
    if missing:
        del inputs[3]["voc"][field]
    else:
        inputs[3]["voc"][field] = False if field.endswith("align") else 0.5
    with pytest.raises(ValueError, match=field):
        fixed_eval._require_fixed_protocol(
            *inputs,
            confirmation_profile="v10-300k",
            seeds=range(
                fixed_eval.DEFAULT_SEED_BASE,
                fixed_eval.DEFAULT_SEED_BASE + fixed_eval.DEFAULT_NUM_SEEDS,
            ),
            real_steps_per_seed=fixed_eval.DEFAULT_REAL_STEPS_PER_SEED,
            calibration_unroll=fixed_eval.DEFAULT_CALIBRATION_UNROLL,
            diagnostic=False,
        )


V11_PROFILE_FLAG_MISMATCHES = {
    "base_seed": 2,
    "schedule_total_steps": 99_999_999,
    "dynamic_voc_mode": "shadow",
    "voc_dedicated_gate": False,
    "voc_soft_q_bce_gate": False,
    "voc_gate_temperature": np.nextafter(np.float64(1.0), np.float64(2.0)),
    "voc_gate_q_temperature": np.nextafter(
        np.float64(0.05), np.float64(1.0)
    ),
    "voc_gate_param_align": True,
    "voc_gate_param_align_coef": np.nextafter(
        np.float64(1.0), np.float64(2.0)
    ),
    "voc_gate_exact_projection": False,
    "ckp": True,
    "preload": "/tmp/model-parent",
    "preload_actor": "/tmp/actor-parent",
    "voc_parent_checkpoint": "/tmp/voc-parent.tar",
}


def _v11_identity_container(inputs, source):
    if source == "config":
        return inputs[0]
    if source == "actor":
        return inputs[1]["flags"]
    return inputs[2]["flags"]


@pytest.mark.parametrize("source", ["config", "actor", "model"])
@pytest.mark.parametrize(
    ("field", "mismatch"), list(V11_PROFILE_FLAG_MISMATCHES.items())
)
def test_v11_profile_rejects_identity_mismatch(source, field, mismatch):
    inputs = list(_fixed_protocol_inputs(300_000, profile="v11-300k"))
    container = _v11_identity_container(inputs, source)
    if isinstance(container, SimpleNamespace):
        setattr(container, field, mismatch)
    else:
        container[field] = mismatch
    with pytest.raises(ValueError, match=field):
        fixed_eval._require_fixed_protocol(
            *inputs,
            confirmation_profile="v11-300k",
            seeds=range(
                fixed_eval.DEFAULT_SEED_BASE,
                fixed_eval.DEFAULT_SEED_BASE + fixed_eval.DEFAULT_NUM_SEEDS,
            ),
            real_steps_per_seed=fixed_eval.DEFAULT_REAL_STEPS_PER_SEED,
            calibration_unroll=fixed_eval.DEFAULT_CALIBRATION_UNROLL,
            diagnostic=False,
        )


@pytest.mark.parametrize("source", ["config", "actor", "model"])
@pytest.mark.parametrize("field", list(V11_PROFILE_FLAG_MISMATCHES))
def test_v11_profile_rejects_missing_identity_field(source, field):
    inputs = list(_fixed_protocol_inputs(300_000, profile="v11-300k"))
    container = _v11_identity_container(inputs, source)
    if isinstance(container, SimpleNamespace):
        delattr(container, field)
    else:
        del container[field]
    with pytest.raises(ValueError, match=field):
        fixed_eval._require_fixed_protocol(
            *inputs,
            confirmation_profile="v11-300k",
            seeds=range(
                fixed_eval.DEFAULT_SEED_BASE,
                fixed_eval.DEFAULT_SEED_BASE + fixed_eval.DEFAULT_NUM_SEEDS,
            ),
            real_steps_per_seed=fixed_eval.DEFAULT_REAL_STEPS_PER_SEED,
            calibration_unroll=fixed_eval.DEFAULT_CALIBRATION_UNROLL,
            diagnostic=False,
        )


@pytest.mark.parametrize(
    ("field", "mismatch"),
    [
        ("voc_gate_policy_schema_version", 3),
        ("voc_gate_param_align", True),
        (
            "voc_gate_param_align_coef",
            np.nextafter(np.float64(1.0), np.float64(2.0)),
        ),
        ("voc_gate_exact_projection", False),
    ],
)
@pytest.mark.parametrize("missing", [False, True])
def test_v11_profile_requires_exact_public_validation_identity(
    field, mismatch, missing
):
    inputs = list(_fixed_protocol_inputs(300_000, profile="v11-300k"))
    if missing:
        del inputs[3]["voc"][field]
    else:
        inputs[3]["voc"][field] = mismatch
    with pytest.raises(ValueError, match=field):
        fixed_eval._require_fixed_protocol(
            *inputs,
            confirmation_profile="v11-300k",
            seeds=range(
                fixed_eval.DEFAULT_SEED_BASE,
                fixed_eval.DEFAULT_SEED_BASE + fixed_eval.DEFAULT_NUM_SEEDS,
            ),
            real_steps_per_seed=fixed_eval.DEFAULT_REAL_STEPS_PER_SEED,
            calibration_unroll=fixed_eval.DEFAULT_CALIBRATION_UNROLL,
            diagnostic=False,
        )


@pytest.mark.parametrize("missing", [False, True])
def test_v11_profile_requires_actor_checkpoint_schema_four(missing):
    inputs = list(_fixed_protocol_inputs(300_000, profile="v11-300k"))
    if missing:
        del inputs[1]["voc_gate_policy_schema_version"]
    else:
        inputs[1]["voc_gate_policy_schema_version"] = 3
    with pytest.raises(ValueError, match="voc_gate_policy_schema_version"):
        fixed_eval._require_fixed_protocol(
            *inputs,
            confirmation_profile="v11-300k",
            seeds=range(
                fixed_eval.DEFAULT_SEED_BASE,
                fixed_eval.DEFAULT_SEED_BASE + fixed_eval.DEFAULT_NUM_SEEDS,
            ),
            real_steps_per_seed=fixed_eval.DEFAULT_REAL_STEPS_PER_SEED,
            calibration_unroll=fixed_eval.DEFAULT_CALIBRATION_UNROLL,
            diagnostic=False,
        )


V12_PROFILE_FLAG_MISMATCHES = {
    **V11_PROFILE_FLAG_MISMATCHES,
    "base_seed": 3,
    "voc_eval_stochastic": False,
    "voc_train_epsilon": np.nextafter(
        np.float64(0.02), np.float64(1.0)
    ),
    "voc_gate_epsilon_greedy_execution": False,
}


@pytest.mark.parametrize("source", ["config", "actor", "model"])
@pytest.mark.parametrize(
    ("field", "mismatch"), list(V12_PROFILE_FLAG_MISMATCHES.items())
)
@pytest.mark.parametrize("missing", [False, True])
def test_v12_profile_rejects_mismatched_or_missing_identity(
    source, field, mismatch, missing
):
    inputs = list(_fixed_protocol_inputs(300_000, profile="v12-300k"))
    container = _v11_identity_container(inputs, source)
    if missing:
        if isinstance(container, SimpleNamespace):
            delattr(container, field)
        else:
            del container[field]
    elif isinstance(container, SimpleNamespace):
        setattr(container, field, mismatch)
    else:
        container[field] = mismatch
    with pytest.raises(ValueError, match=field):
        fixed_eval._require_fixed_protocol(
            *inputs,
            confirmation_profile="v12-300k",
            seeds=range(
                fixed_eval.DEFAULT_SEED_BASE,
                fixed_eval.DEFAULT_SEED_BASE + fixed_eval.DEFAULT_NUM_SEEDS,
            ),
            real_steps_per_seed=fixed_eval.DEFAULT_REAL_STEPS_PER_SEED,
            calibration_unroll=fixed_eval.DEFAULT_CALIBRATION_UNROLL,
            diagnostic=False,
        )


@pytest.mark.parametrize("source", ["config", "actor", "model"])
@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("voc_eval_stochastic", 1),
        ("voc_train_epsilon", True),
        ("voc_train_epsilon", float("nan")),
        ("voc_train_epsilon", float("inf")),
        (
            "voc_train_epsilon",
            np.nextafter(np.float64(0.02), np.float64(1.0)),
        ),
    ],
)
def test_v12_profile_requires_exact_execution_distribution_identity(
    source, field, invalid
):
    inputs = list(_fixed_protocol_inputs(300_000, profile="v12-300k"))
    container = _v11_identity_container(inputs, source)
    if isinstance(container, SimpleNamespace):
        setattr(container, field, invalid)
    else:
        container[field] = invalid
    with pytest.raises(ValueError, match=field):
        fixed_eval._require_fixed_protocol(
            *inputs,
            confirmation_profile="v12-300k",
            seeds=range(
                fixed_eval.DEFAULT_SEED_BASE,
                fixed_eval.DEFAULT_SEED_BASE + fixed_eval.DEFAULT_NUM_SEEDS,
            ),
            real_steps_per_seed=fixed_eval.DEFAULT_REAL_STEPS_PER_SEED,
            calibration_unroll=fixed_eval.DEFAULT_CALIBRATION_UNROLL,
            diagnostic=False,
        )


@pytest.mark.parametrize(
    ("field", "mismatch"),
    [
        ("voc_gate_policy_schema_version", 4),
        ("voc_gate_param_align", True),
        (
            "voc_gate_param_align_coef",
            np.nextafter(np.float64(1.0), np.float64(2.0)),
        ),
        ("voc_gate_exact_projection", False),
        ("voc_gate_epsilon_greedy_execution", False),
    ],
)
@pytest.mark.parametrize("missing", [False, True])
def test_v12_profile_requires_exact_public_validation_identity(
    field, mismatch, missing
):
    inputs = list(_fixed_protocol_inputs(300_000, profile="v12-300k"))
    if missing:
        del inputs[3]["voc"][field]
    else:
        inputs[3]["voc"][field] = mismatch
    with pytest.raises(ValueError, match=field):
        fixed_eval._require_fixed_protocol(
            *inputs,
            confirmation_profile="v12-300k",
            seeds=range(
                fixed_eval.DEFAULT_SEED_BASE,
                fixed_eval.DEFAULT_SEED_BASE + fixed_eval.DEFAULT_NUM_SEEDS,
            ),
            real_steps_per_seed=fixed_eval.DEFAULT_REAL_STEPS_PER_SEED,
            calibration_unroll=fixed_eval.DEFAULT_CALIBRATION_UNROLL,
            diagnostic=False,
        )


V13_PROFILE_FLAG_MISMATCHES = {
    **V12_PROFILE_FLAG_MISMATCHES,
    "xpid": "enduro-voc-v13-versioned-eps25-seed1-qual-fresh-100k",
    "base_seed": 1,
    "model_warm_up_n": 512,
    "actor_unroll_len": 41,
    "use_wandb": False,
    "voc_gate_policy_schema_version": 5,
    "voc_gate_execution_epsilon": np.nextafter(
        np.float64(0.25), np.float64(1.0)
    ),
    "voc_actor_policy_version_barrier": False,
    "voc_actor_policy_bundle_schema_version": 2,
    "voc_actor_policy_barrier_timeout_s": np.nextafter(
        np.float64(120.0), np.float64(121.0)
    ),
    "voc_actor_policy_ray_max_restarts": 1,
    "voc_actor_policy_ray_max_task_retries": 1,
    "voc_actor_policy_barrier_runtime": False,
    "actor_amp_init_scale": np.nextafter(
        np.float64(32.0), np.float64(33.0)
    ),
    "float16": False,
    "model_float16": True,
    "parallel_actor": False,
    "ppo_k": 2,
    "self_play_n": 2,
    "env_n": 8,
    "actor_batch_size": 8,
}


@pytest.mark.parametrize("source", ["config", "actor", "model"])
@pytest.mark.parametrize(
    ("field", "mismatch"), list(V13_PROFILE_FLAG_MISMATCHES.items())
)
@pytest.mark.parametrize("missing", [False, True])
def test_v13_profile_rejects_mismatched_or_missing_identity(
    source, field, mismatch, missing
):
    inputs = list(_fixed_protocol_inputs(300_000, profile="v13-300k"))
    container = _v11_identity_container(inputs, source)
    if missing:
        if isinstance(container, SimpleNamespace):
            delattr(container, field)
        else:
            del container[field]
    elif isinstance(container, SimpleNamespace):
        setattr(container, field, mismatch)
    else:
        container[field] = mismatch
    with pytest.raises(ValueError, match=field):
        fixed_eval._require_fixed_protocol(
            *inputs,
            confirmation_profile="v13-300k",
            seeds=range(
                fixed_eval.DEFAULT_SEED_BASE,
                fixed_eval.DEFAULT_SEED_BASE + fixed_eval.DEFAULT_NUM_SEEDS,
            ),
            real_steps_per_seed=fixed_eval.DEFAULT_REAL_STEPS_PER_SEED,
            calibration_unroll=fixed_eval.DEFAULT_CALIBRATION_UNROLL,
            diagnostic=False,
            schema6_bundle_validation=_v13_bundle_evidence(),
        )


@pytest.mark.parametrize("source", ["config", "actor", "model"])
@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("voc_train_epsilon", True),
        ("voc_train_epsilon", float("nan")),
        ("voc_gate_execution_epsilon", True),
        ("voc_gate_execution_epsilon", float("inf")),
        (
            "voc_gate_execution_epsilon",
            np.nextafter(np.float64(0.25), np.float64(1.0)),
        ),
        ("voc_actor_policy_version_barrier", 1),
        ("voc_actor_policy_barrier_runtime", np.bool_(True)),
        ("actor_amp_init_scale", True),
        ("use_wandb", 1),
        ("float16", np.bool_(True)),
        ("model_float16", 0),
    ],
)
def test_v13_profile_requires_strict_atomic_types(source, field, invalid):
    inputs = list(_fixed_protocol_inputs(300_000, profile="v13-300k"))
    container = _v11_identity_container(inputs, source)
    if isinstance(container, SimpleNamespace):
        setattr(container, field, invalid)
    else:
        container[field] = invalid
    with pytest.raises(ValueError, match=field):
        fixed_eval._require_fixed_protocol(
            *inputs,
            confirmation_profile="v13-300k",
            seeds=range(
                fixed_eval.DEFAULT_SEED_BASE,
                fixed_eval.DEFAULT_SEED_BASE + fixed_eval.DEFAULT_NUM_SEEDS,
            ),
            real_steps_per_seed=fixed_eval.DEFAULT_REAL_STEPS_PER_SEED,
            calibration_unroll=fixed_eval.DEFAULT_CALIBRATION_UNROLL,
            diagnostic=False,
            schema6_bundle_validation=_v13_bundle_evidence(),
        )


@pytest.mark.parametrize(
    ("field", "mismatch"),
    [
        ("voc_gate_policy_schema_version", 5),
        ("voc_gate_param_align", True),
        (
            "voc_gate_param_align_coef",
            np.nextafter(np.float64(1.0), np.float64(2.0)),
        ),
        ("voc_gate_exact_projection", False),
        ("voc_gate_epsilon_greedy_execution", False),
    ],
)
@pytest.mark.parametrize("missing", [False, True])
def test_v13_profile_requires_exact_public_gate_validation(
    field, mismatch, missing
):
    inputs = list(_fixed_protocol_inputs(300_000, profile="v13-300k"))
    if missing:
        del inputs[3]["voc"][field]
    else:
        inputs[3]["voc"][field] = mismatch
    with pytest.raises(ValueError, match=field):
        fixed_eval._require_fixed_protocol(
            *inputs,
            confirmation_profile="v13-300k",
            seeds=range(
                fixed_eval.DEFAULT_SEED_BASE,
                fixed_eval.DEFAULT_SEED_BASE + fixed_eval.DEFAULT_NUM_SEEDS,
            ),
            real_steps_per_seed=fixed_eval.DEFAULT_REAL_STEPS_PER_SEED,
            calibration_unroll=fixed_eval.DEFAULT_CALIBRATION_UNROLL,
            diagnostic=False,
            schema6_bundle_validation=_v13_bundle_evidence(),
        )


def test_v13_profile_requires_authoritative_primary_bundle_evidence():
    inputs = _fixed_protocol_inputs(300_000, profile="v13-300k")
    with pytest.raises(ValueError, match="authoritative schema-6"):
        fixed_eval._require_fixed_protocol(
            *inputs,
            confirmation_profile="v13-300k",
            seeds=range(
                fixed_eval.DEFAULT_SEED_BASE,
                fixed_eval.DEFAULT_SEED_BASE + fixed_eval.DEFAULT_NUM_SEEDS,
            ),
            real_steps_per_seed=fixed_eval.DEFAULT_REAL_STEPS_PER_SEED,
            calibration_unroll=fixed_eval.DEFAULT_CALIBRATION_UNROLL,
            diagnostic=False,
        )

    wire_stage = (
        "enduro-voc-v13-versioned-eps25-sentinel-wire1200",
        1,
        1200,
        512,
        41,
        False,
    )
    with pytest.raises(ValueError, match="exact primary stage"):
        fixed_eval._require_fixed_protocol(
            *inputs,
            confirmation_profile="v13-300k",
            seeds=range(
                fixed_eval.DEFAULT_SEED_BASE,
                fixed_eval.DEFAULT_SEED_BASE + fixed_eval.DEFAULT_NUM_SEEDS,
            ),
            real_steps_per_seed=fixed_eval.DEFAULT_REAL_STEPS_PER_SEED,
            calibration_unroll=fixed_eval.DEFAULT_CALIBRATION_UNROLL,
            diagnostic=False,
            schema6_bundle_validation=_v13_bundle_evidence(stage=wire_stage),
        )


@pytest.mark.parametrize(
    ("section", "field", "value", "match"),
    [
        ("resolved_identity", "key_count", 227, "key_count"),
        (
            "resolved_identity",
            "v12_projection_sha256",
            "0" * 64,
            "v12_projection_sha256",
        ),
        ("actor_policy", "voc_actor_policy_terminal", False, "terminal"),
        ("actor_policy", "actor_amp_skip_count", 1, "actor_amp_skip_count"),
        ("logger_completion", "ack_verified", False, "ack_verified"),
        (None, "config_use_wandb", False, "config_use_wandb"),
    ],
)
def test_v13_bundle_evidence_rejects_terminal_or_identity_drift(
    section, field, value, match
):
    evidence = _v13_bundle_evidence()
    target = evidence if section is None else evidence[section]
    target[field] = value
    with pytest.raises(ValueError, match=match):
        fixed_eval._require_v13_bundle_evidence(evidence)


def _v13_authoritative_bundle_fixture():
    evidence = _v13_bundle_evidence()
    checkpoint_files = {
        name: {"sha256": index * "1" + (64 - index) * "0", "size": index}
        for index, name in enumerate(
            fixed_eval.REQUIRED_CHECKPOINT_FILES, start=1
        )
    }
    completion = {
        "checkpoint_files": checkpoint_files,
        "implementation_sources": {"train.py": {"sha256": "e" * 64}},
        "loaded_extensions": {"thinker/cenv.so": {"sha256": "f" * 64}},
    }
    authoritative = {
        field: evidence[field]
        for field in (
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
        )
    }
    authoritative["completion_evidence"] = completion
    logger_completion = dict(evidence["logger_completion"])
    logger_completion["checkpoint_files"] = checkpoint_files
    marker = {
        "schema_version": 1,
        "status": "complete",
        "completed_unix": 1.0,
        **completion,
        "voc_actor_policy_logger_completion": logger_completion,
    }
    return authoritative, logger_completion, marker


def test_v13_authoritative_bundle_binds_history_logger_and_marker_absence(
    monkeypatch, tmp_path
):
    from thinker import util

    authoritative, logger_completion, marker = (
        _v13_authoritative_bundle_fixture()
    )
    calls = []

    def validate_bundle(path, *, label):
        calls.append((Path(path), label))
        return authoritative

    monkeypatch.setattr(util, "validate_schema6_final_bundle", validate_bundle)
    monkeypatch.setattr(
        util,
        "validate_actor_policy_logger_completion",
        lambda value: logger_completion,
    )
    validated = fixed_eval.validate_v13_final_bundle(tmp_path, marker)
    assert calls == [
        (tmp_path.resolve(), "v13-300k authoritative final bundle")
    ]
    assert validated["resolved_identity"]["key_count"] == 228
    assert validated["actor_policy"][
        "voc_actor_policy_publication_history"
    ] == authoritative["actor_policy"][
        "voc_actor_policy_publication_history"
    ]
    assert validated["logger_completion"]["ack_verified"] is True
    assert all(
        record["absent"] is True
        for record in validated["private_logger_markers"].values()
    )

    (tmp_path / fixed_eval.V13_PRIVATE_LOGGER_MARKERS[0]).write_text(
        "forensic marker", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="private actor-policy logger marker"):
        fixed_eval.validate_v13_final_bundle(tmp_path, marker)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("size", True, "invalid size"),
        ("size", 1.0, "invalid size"),
        ("sha256", True, "invalid SHA-256"),
        ("sha256", "A" * 64, "invalid SHA-256"),
    ],
)
def test_v13_authoritative_bundle_rejects_equal_cross_surface_record_type_drift(
    monkeypatch, tmp_path, field, value, match
):
    from thinker import util

    authoritative, logger_completion, marker = (
        _v13_authoritative_bundle_fixture()
    )
    # The fixture intentionally shares this mapping across marker,
    # authoritative evidence, and logger completion.  Equality alone would
    # therefore accept these consistently malformed values.
    marker["checkpoint_files"]["ckp_actor.tar"][field] = value
    monkeypatch.setattr(
        util,
        "validate_schema6_final_bundle",
        lambda path, *, label: authoritative,
    )
    monkeypatch.setattr(
        util,
        "validate_actor_policy_logger_completion",
        lambda value: logger_completion,
    )
    with pytest.raises(ValueError, match=match):
        fixed_eval.validate_v13_final_bundle(tmp_path, marker)


def test_v13_authoritative_bundle_rejects_equal_source_hash_type_drift(
    monkeypatch, tmp_path
):
    from thinker import util

    authoritative, logger_completion, marker = (
        _v13_authoritative_bundle_fixture()
    )
    marker["implementation_sources"]["train.py"]["sha256"] = True
    monkeypatch.setattr(
        util,
        "validate_schema6_final_bundle",
        lambda path, *, label: authoritative,
    )
    monkeypatch.setattr(
        util,
        "validate_actor_policy_logger_completion",
        lambda value: logger_completion,
    )
    with pytest.raises(ValueError, match="invalid SHA-256"):
        fixed_eval.validate_v13_final_bundle(tmp_path, marker)


def test_v13_authoritative_bundle_rejects_wire_after_full_validation(
    monkeypatch, tmp_path
):
    from thinker import util

    authoritative, logger_completion, marker = (
        _v13_authoritative_bundle_fixture()
    )
    authoritative["resolved_identity"] = dict(
        authoritative["resolved_identity"]
    )
    authoritative["resolved_identity"]["stage"] = (
        "enduro-voc-v13-versioned-eps25-sentinel-wire1200",
        1,
        1200,
        512,
        41,
        False,
    )
    calls = []

    def validate_bundle(path, *, label):
        calls.append(Path(path))
        return authoritative

    monkeypatch.setattr(util, "validate_schema6_final_bundle", validate_bundle)
    monkeypatch.setattr(
        util,
        "validate_actor_policy_logger_completion",
        lambda value: logger_completion,
    )
    with pytest.raises(ValueError, match="exact primary stage"):
        fixed_eval.validate_v13_final_bundle(tmp_path, marker)
    assert calls == [tmp_path.resolve()]


def test_v13_runtime_copy_preserves_training_identity_and_disables_wait(tmp_path):
    flags = SimpleNamespace(
        train_actor=True,
        train_model=True,
        parallel=True,
        parallel_actor=True,
        ckp=False,
        use_wandb=True,
        ckpdir="/immutable/checkpoint",
        savedir="/immutable/runs",
        preload="",
        preload_actor="",
        voc_parent_checkpoint="",
        voc_actor_policy_version_barrier=True,
        voc_actor_policy_barrier_runtime=True,
        voc_train_epsilon=0.02,
        voc_gate_execution_epsilon=0.25,
    )
    runtime = fixed_eval._runtime_flags(flags, tmp_path)
    assert flags.train_actor is True
    assert flags.voc_actor_policy_barrier_runtime is True
    assert flags.voc_train_epsilon == 0.02
    assert flags.voc_gate_execution_epsilon == 0.25
    assert runtime.train_actor is False
    assert runtime.train_model is False
    assert runtime.voc_actor_policy_version_barrier is True
    assert runtime.voc_actor_policy_barrier_runtime is False
    assert runtime.voc_train_epsilon == 0.02
    assert runtime.voc_gate_execution_epsilon == 0.25


@pytest.mark.parametrize(
    ("input_profile", "requested_profile"),
    [
        ("v11-300k", "v12-300k"),
        ("v12-300k", "v11-300k"),
        ("v12-300k", "v13-300k"),
        ("v13-300k", "v12-300k"),
        ("v11-300k", "v13-300k"),
        ("v13-300k", "v11-300k"),
        ("v10-300k", "v13-300k"),
        ("v13-300k", "v10-300k"),
    ],
)
def test_v11_v12_profiles_reject_cross_profile_bundle_identity(
    input_profile, requested_profile
):
    inputs = _fixed_protocol_inputs(300_000, profile=input_profile)
    with pytest.raises(ValueError, match="base_seed|xpid"):
        fixed_eval._require_fixed_protocol(
            *inputs,
            confirmation_profile=requested_profile,
            seeds=range(
                fixed_eval.DEFAULT_SEED_BASE,
                fixed_eval.DEFAULT_SEED_BASE + fixed_eval.DEFAULT_NUM_SEEDS,
            ),
            real_steps_per_seed=fixed_eval.DEFAULT_REAL_STEPS_PER_SEED,
            calibration_unroll=fixed_eval.DEFAULT_CALIBRATION_UNROLL,
            diagnostic=False,
            schema6_bundle_validation=(
                _v13_bundle_evidence()
                if requested_profile == "v13-300k"
                else None
            ),
        )


@pytest.mark.parametrize("profile", ["v7-200k", "v10-300k", "v11-300k"])
@pytest.mark.parametrize(
    "source", ["config", "actor", "model", "validation"]
)
def test_legacy_profiles_reject_epsilon_greedy_execution_true(profile, source):
    total_steps = 200_000 if profile == "v7-200k" else 300_000
    inputs = list(_fixed_protocol_inputs(total_steps, profile=profile))
    if source == "config":
        inputs[0].voc_gate_epsilon_greedy_execution = True
    elif source == "actor":
        inputs[1]["flags"]["voc_gate_epsilon_greedy_execution"] = True
    elif source == "model":
        inputs[2]["flags"]["voc_gate_epsilon_greedy_execution"] = True
    else:
        inputs[3]["voc"]["voc_gate_epsilon_greedy_execution"] = True
    with pytest.raises(
        ValueError, match="normalized voc_gate_epsilon_greedy_execution=false"
    ):
        fixed_eval._require_fixed_protocol(
            *inputs,
            confirmation_profile=profile,
            seeds=range(
                fixed_eval.DEFAULT_SEED_BASE,
                fixed_eval.DEFAULT_SEED_BASE + fixed_eval.DEFAULT_NUM_SEEDS,
            ),
            real_steps_per_seed=fixed_eval.DEFAULT_REAL_STEPS_PER_SEED,
            calibration_unroll=fixed_eval.DEFAULT_CALIBRATION_UNROLL,
            diagnostic=False,
        )


@pytest.mark.parametrize("profile", ["v7-200k", "v10-300k", "v11-300k"])
def test_legacy_profiles_accept_explicit_normalized_execution_false(profile):
    total_steps = 200_000 if profile == "v7-200k" else 300_000
    inputs = list(_fixed_protocol_inputs(total_steps, profile=profile))
    inputs[0].voc_gate_epsilon_greedy_execution = False
    inputs[1]["flags"]["voc_gate_epsilon_greedy_execution"] = False
    inputs[2]["flags"]["voc_gate_epsilon_greedy_execution"] = False
    inputs[3]["voc"]["voc_gate_epsilon_greedy_execution"] = False
    state = fixed_eval._require_fixed_protocol(
        *inputs,
        confirmation_profile=profile,
        seeds=range(
            fixed_eval.DEFAULT_SEED_BASE,
            fixed_eval.DEFAULT_SEED_BASE + fixed_eval.DEFAULT_NUM_SEEDS,
        ),
        real_steps_per_seed=fixed_eval.DEFAULT_REAL_STEPS_PER_SEED,
        calibration_unroll=fixed_eval.DEFAULT_CALIBRATION_UNROLL,
        diagnostic=False,
        schema6_bundle_validation=(
            _v13_bundle_evidence() if profile == "v13-300k" else None
        ),
    )
    assert state["voc_gate_epsilon_greedy_execution"] is False
    assert set(state["normalized_execution_identity"].values()) == {False}


@pytest.mark.parametrize(
    "profile", ["v7-200k", "v10-300k", "v11-300k", "v12-300k"]
)
@pytest.mark.parametrize("source", ["config", "actor", "model"])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("voc_actor_policy_version_barrier", True),
        ("voc_actor_policy_barrier_runtime", True),
        ("voc_gate_execution_epsilon", 0.25),
        ("actor_amp_init_scale", 32.0),
        ("voc_actor_policy_bundle_schema_version", 1),
        ("voc_gate_policy_schema_version", 6),
    ],
)
def test_legacy_profiles_reject_each_schema6_atomic_marker(
    profile, source, field, value
):
    total_steps = 200_000 if profile == "v7-200k" else 300_000
    inputs = list(_fixed_protocol_inputs(total_steps, profile=profile))
    container = _v11_identity_container(inputs, source)
    if isinstance(container, SimpleNamespace):
        setattr(container, field, value)
    else:
        container[field] = value
    with pytest.raises(ValueError):
        fixed_eval._require_fixed_protocol(
            *inputs,
            confirmation_profile=profile,
            seeds=range(
                fixed_eval.DEFAULT_SEED_BASE,
                fixed_eval.DEFAULT_SEED_BASE + fixed_eval.DEFAULT_NUM_SEEDS,
            ),
            real_steps_per_seed=fixed_eval.DEFAULT_REAL_STEPS_PER_SEED,
            calibration_unroll=fixed_eval.DEFAULT_CALIBRATION_UNROLL,
            diagnostic=False,
        )


def test_gate_protocol_description_uses_validated_execution_not_profile_name():
    legacy_execution = fixed_eval._evaluation_gate_protocol(
        {
            "confirmation_profile": "v12-300k",
            "voc_gate_epsilon_greedy_execution": False,
        }
    )
    assert legacy_execution == {
        "stochastic_gate": True,
        "gate_sampling": "checkpoint gate, epsilon zero, fixed RNG",
    }

    epsilon_greedy_execution = fixed_eval._evaluation_gate_protocol(
        {
            "confirmation_profile": "v7-200k",
            "voc_gate_epsilon_greedy_execution": True,
        }
    )
    assert epsilon_greedy_execution["stochastic_gate"] is False
    assert epsilon_greedy_execution["gate_execution"] == "epsilon_greedy"
    assert epsilon_greedy_execution[
        "gate_exact_zero_tie_sampling_probability"
    ] == 0.5

    v13 = fixed_eval._evaluation_gate_protocol(
        {
            "voc_gate_epsilon_greedy_execution": True,
            "training_gate_soft_epsilon": 0.02,
            "training_gate_execution_epsilon": 0.25,
            "runtime_gate_soft_epsilon": 0.0,
            "runtime_gate_execution_epsilon": 0.0,
            "training_actor_policy_version_barrier": True,
            "runtime_actor_policy_barrier_wait": False,
        }
    )
    assert v13["training_gate_soft_epsilon"] == 0.02
    assert v13["training_gate_execution_epsilon"] == 0.25
    assert v13["runtime_gate_soft_epsilon"] == 0.0
    assert v13["runtime_gate_execution_epsilon"] == 0.0
    assert v13["runtime_actor_policy_barrier_wait"] is False

    v14 = fixed_eval._evaluation_gate_protocol(
        {
            "voc_gate_epsilon_greedy_execution": True,
            "training_gate_soft_epsilon": 0.02,
            "training_gate_execution_epsilon": 0.25,
            "runtime_gate_soft_epsilon": 0.0,
            "runtime_gate_execution_epsilon": 0.0,
            "training_actor_policy_version_barrier": True,
            "runtime_actor_policy_barrier_wait": False,
            "training_model_input_seal_schema_version": 1,
            "runtime_model_input_seal_coordination": False,
        }
    )
    assert v14["training_model_input_seal_schema_version"] == 1
    assert v14["runtime_model_input_seal_coordination"] is False


V11_FRESH_PROVENANCE_MISMATCHES = {
    "dynamic_voc_mode": "shadow",
    "voc_control_origin": "shadow_parent",
    "voc_control_origin_legacy_defaulted": True,
    "voc_activation_real_step": 1,
    "voc_parent_checkpoint_sha256": "a" * 64,
    "voc_parent_checkpoint": "/tmp/parent.tar",
    "voc_parent_imitation_data_signature": "b" * 64,
}


@pytest.mark.parametrize("source", ["checkpoint", "validation"])
@pytest.mark.parametrize(
    ("field", "mismatch"), list(V11_FRESH_PROVENANCE_MISMATCHES.items())
)
@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize(
    "profile", ["v11-300k", "v12-300k", "v13-300k", "v14-300k"]
)
def test_exact_projection_profiles_require_exact_fresh_actor_provenance(
    source, field, mismatch, missing, profile
):
    inputs = list(_fixed_protocol_inputs(300_000, profile=profile))
    container = inputs[1] if source == "checkpoint" else inputs[3]["voc"]
    if missing:
        del container[field]
    else:
        container[field] = mismatch
    with pytest.raises(ValueError, match=field):
        fixed_eval._require_fixed_protocol(
            *inputs,
            confirmation_profile=profile,
            seeds=range(
                fixed_eval.DEFAULT_SEED_BASE,
                fixed_eval.DEFAULT_SEED_BASE + fixed_eval.DEFAULT_NUM_SEEDS,
            ),
            real_steps_per_seed=fixed_eval.DEFAULT_REAL_STEPS_PER_SEED,
            calibration_unroll=fixed_eval.DEFAULT_CALIBRATION_UNROLL,
            diagnostic=False,
            schema6_bundle_validation=(
                _v13_bundle_evidence() if profile == "v13-300k" else None
            ),
            schema7_bundle_validation=(
                _v14_bundle_evidence() if profile == "v14-300k" else None
            ),
        )


@pytest.mark.parametrize("parameter", ["weight", "bias"])
@pytest.mark.parametrize(
    "profile", ["v11-300k", "v12-300k", "v13-300k", "v14-300k"]
)
def test_exact_projection_profiles_reject_one_bit_gate_target_mismatch(
    parameter, profile
):
    inputs = list(_fixed_protocol_inputs(300_000, profile=profile))
    tensor = inputs[1]["actor_net_state_dict"][
        f"voc_gate_head.{parameter}"
    ]
    flat = tensor.view(-1)
    flat[0] = torch.nextafter(flat[0], torch.tensor(float("inf")))
    with pytest.raises(ValueError, match=rf"gate {parameter} disagrees"):
        fixed_eval._require_fixed_protocol(
            *inputs,
            confirmation_profile=profile,
            seeds=range(
                fixed_eval.DEFAULT_SEED_BASE,
                fixed_eval.DEFAULT_SEED_BASE + fixed_eval.DEFAULT_NUM_SEEDS,
            ),
            real_steps_per_seed=fixed_eval.DEFAULT_REAL_STEPS_PER_SEED,
            calibration_unroll=fixed_eval.DEFAULT_CALIBRATION_UNROLL,
            diagnostic=False,
            schema6_bundle_validation=(
                _v13_bundle_evidence() if profile == "v13-300k" else None
            ),
            schema7_bundle_validation=(
                _v14_bundle_evidence() if profile == "v14-300k" else None
            ),
        )


@pytest.mark.parametrize(
    "state_path",
    [
        ("actor_net_state_dict", "voc_gate_head.weight"),
        ("actor_net_state_dict", "voc_gate_head.bias"),
        ("voc_ema_gate_head_state_dict", "weight"),
        ("voc_ema_gate_head_state_dict", "bias"),
    ],
)
@pytest.mark.parametrize(
    "profile", ["v11-300k", "v12-300k", "v13-300k", "v14-300k"]
)
def test_exact_projection_profiles_require_stored_fp32_projection_tensors(
    state_path, profile
):
    inputs = list(_fixed_protocol_inputs(300_000, profile=profile))
    container, field = state_path
    inputs[1][container][field] = inputs[1][container][field].double()
    with pytest.raises(ValueError, match="must be an FP32 tensor"):
        fixed_eval._require_fixed_protocol(
            *inputs,
            confirmation_profile=profile,
            seeds=range(
                fixed_eval.DEFAULT_SEED_BASE,
                fixed_eval.DEFAULT_SEED_BASE + fixed_eval.DEFAULT_NUM_SEEDS,
            ),
            real_steps_per_seed=fixed_eval.DEFAULT_REAL_STEPS_PER_SEED,
            calibration_unroll=fixed_eval.DEFAULT_CALIBRATION_UNROLL,
            diagnostic=False,
            schema6_bundle_validation=(
                _v13_bundle_evidence() if profile == "v13-300k" else None
            ),
            schema7_bundle_validation=(
                _v14_bundle_evidence() if profile == "v14-300k" else None
            ),
        )


@pytest.mark.parametrize("profile", ["v7-200k", "v10-300k"])
def test_v11_profile_addition_preserves_legacy_resolved_identity_shape(profile):
    total_steps = 200_000 if profile == "v7-200k" else 300_000
    resolved = _exact_protocol_call(profile, total_steps)[
        "resolved_profile_identity"
    ]
    assert set(resolved) == {
        "config",
        "actor_checkpoint",
        "model_checkpoint",
        "actor_checkpoint_validation",
    }
    assert "voc_gate_exact_projection" not in resolved["config"]
    assert "terminal_exact_projection" not in resolved


@pytest.mark.parametrize(
    "profile", ["v7-200k", "v10-300k", "v11-300k", "v12-300k", "v13-300k"]
)
@pytest.mark.parametrize("missing_from", ["config", "actor", "model"])
def test_fixed_profiles_reject_missing_total_steps(profile, missing_from):
    expected = (
        fixed_eval.CONFIRMATION_PROFILE_SPECS[
            fixed_eval.ConfirmationProfile(profile)
        ].total_steps
    )
    inputs = list(_fixed_protocol_inputs(expected, profile=profile))
    if missing_from == "config":
        delattr(inputs[0], "total_steps")
    elif missing_from == "actor":
        del inputs[1]["flags"]["total_steps"]
    else:
        del inputs[2]["flags"]["total_steps"]
    with pytest.raises(ValueError, match="lacks total_steps|flags.total_steps"):
        fixed_eval._require_fixed_protocol(
            *inputs,
            confirmation_profile=profile,
            seeds=range(
                fixed_eval.DEFAULT_SEED_BASE,
                fixed_eval.DEFAULT_SEED_BASE + fixed_eval.DEFAULT_NUM_SEEDS,
            ),
            real_steps_per_seed=fixed_eval.DEFAULT_REAL_STEPS_PER_SEED,
            calibration_unroll=fixed_eval.DEFAULT_CALIBRATION_UNROLL,
            diagnostic=False,
        )


@pytest.mark.parametrize(
    "profile", ["v7-200k", "v10-300k", "v11-300k", "v12-300k", "v13-300k"]
)
def test_fixed_profiles_reject_arbitrary_training_horizon(profile):
    with pytest.raises(ValueError, match="checkpoint config total_steps"):
        _exact_protocol_call(profile, 250_000)


@pytest.mark.parametrize(
    ("profile", "total_steps"),
    [
        ("v7-200k", 200_000),
        ("v10-300k", 300_000),
        ("v11-300k", 300_000),
        ("v12-300k", 300_000),
        ("v13-300k", 300_000),
    ],
)
@pytest.mark.parametrize("checkpoint_name", ["actor", "model"])
@pytest.mark.parametrize("side", ["below", "above"])
def test_fixed_profiles_reject_nonfinal_checkpoint_steps(
    profile, total_steps, checkpoint_name, side
):
    inputs = list(_fixed_protocol_inputs(total_steps, profile=profile))
    validation_index = 3 if checkpoint_name == "actor" else 4
    maximum_overshoot = 1 * 16 * 201
    inputs[validation_index]["real_step"] = (
        total_steps - 1
        if side == "below"
        else total_steps + maximum_overshoot + 1
    )
    with pytest.raises(
        ValueError,
        match=rf"{checkpoint_name} checkpoint is not the bounded final",
    ):
        fixed_eval._require_fixed_protocol(
            *inputs,
            confirmation_profile=profile,
            seeds=range(
                fixed_eval.DEFAULT_SEED_BASE,
                fixed_eval.DEFAULT_SEED_BASE + fixed_eval.DEFAULT_NUM_SEEDS,
            ),
            real_steps_per_seed=fixed_eval.DEFAULT_REAL_STEPS_PER_SEED,
            calibration_unroll=fixed_eval.DEFAULT_CALIBRATION_UNROLL,
            diagnostic=False,
        )


@pytest.mark.parametrize(
    ("profile", "total_steps"),
    [
        ("v7-200k", 200_000),
        ("v10-300k", 300_000),
        ("v11-300k", 300_000),
        ("v12-300k", 300_000),
        ("v13-300k", 300_000),
    ],
)
def test_fixed_profiles_accept_inclusive_final_overshoot_boundary(
    profile, total_steps
):
    inputs = list(_fixed_protocol_inputs(total_steps, profile=profile))
    maximum_overshoot = 1 * 16 * 201
    inputs[3]["real_step"] = total_steps + maximum_overshoot
    inputs[4]["real_step"] = total_steps + maximum_overshoot
    state = fixed_eval._require_fixed_protocol(
        *inputs,
        confirmation_profile=profile,
        seeds=range(
            fixed_eval.DEFAULT_SEED_BASE,
            fixed_eval.DEFAULT_SEED_BASE + fixed_eval.DEFAULT_NUM_SEEDS,
        ),
        real_steps_per_seed=fixed_eval.DEFAULT_REAL_STEPS_PER_SEED,
        calibration_unroll=fixed_eval.DEFAULT_CALIBRATION_UNROLL,
        diagnostic=False,
        schema6_bundle_validation=(
            _v13_bundle_evidence() if profile == "v13-300k" else None
        ),
    )
    assert state["actor_final_real_step"] == total_steps + maximum_overshoot
    assert state["model_final_real_step"] == total_steps + maximum_overshoot


def test_fixed_protocol_rejects_unknown_profile():
    inputs = _fixed_protocol_inputs(300_000)
    with pytest.raises(ValueError, match="unknown fixed-confirmation profile"):
        fixed_eval._require_fixed_protocol(
            *inputs,
            confirmation_profile="v11-250k",
            seeds=range(
                fixed_eval.DEFAULT_SEED_BASE,
                fixed_eval.DEFAULT_SEED_BASE + fixed_eval.DEFAULT_NUM_SEEDS,
            ),
            real_steps_per_seed=fixed_eval.DEFAULT_REAL_STEPS_PER_SEED,
            calibration_unroll=fixed_eval.DEFAULT_CALIBRATION_UNROLL,
            diagnostic=False,
        )


def test_fixed_protocol_rejects_missing_ema_and_seed_overlap():
    (
        flags,
        actor_checkpoint,
        model_checkpoint,
        actor_validation,
        model_validation,
    ) = _fixed_protocol_inputs(200_000)
    actor_without_ema = {
        "flags": dict(actor_checkpoint["flags"]),
    }
    with pytest.raises(ValueError, match="EMA Q weight/bias"):
        fixed_eval._require_fixed_200k_protocol(
            flags,
            actor_without_ema,
            model_checkpoint,
            actor_validation,
            model_validation,
            seeds=[100, 101],
            real_steps_per_seed=1,
            calibration_unroll=201,
            diagnostic=True,
        )

    with pytest.raises(ValueError, match="overlap training"):
        fixed_eval._require_fixed_200k_protocol(
            flags,
            actor_checkpoint,
            model_checkpoint,
            actor_validation,
            model_validation,
            seeds=[1, 100],
            real_steps_per_seed=1,
            calibration_unroll=201,
            diagnostic=True,
        )

    with pytest.raises(ValueError, match="model checkpoint is not"):
        fixed_eval._require_fixed_200k_protocol(
            flags,
            actor_checkpoint,
            model_checkpoint,
            actor_validation,
            {"real_step": 1},
            seeds=range(
                fixed_eval.DEFAULT_SEED_BASE,
                fixed_eval.DEFAULT_SEED_BASE + fixed_eval.DEFAULT_NUM_SEEDS,
            ),
            real_steps_per_seed=fixed_eval.DEFAULT_REAL_STEPS_PER_SEED,
            calibration_unroll=fixed_eval.DEFAULT_CALIBRATION_UNROLL,
            diagnostic=False,
        )

    with pytest.raises(ValueError, match="require --diagnostic"):
        fixed_eval._require_fixed_200k_protocol(
            flags,
            actor_checkpoint,
            model_checkpoint,
            actor_validation,
            model_validation,
            seeds=[100, 101],
            real_steps_per_seed=1,
            calibration_unroll=201,
            diagnostic=False,
        )

    diagnostic = fixed_eval._require_fixed_200k_protocol(
        flags,
        actor_checkpoint,
        model_checkpoint,
        actor_validation,
        model_validation,
        seeds=[100, 101],
        real_steps_per_seed=1,
        calibration_unroll=201,
        diagnostic=True,
    )
    assert diagnostic["evaluation_mode"] == "diagnostic"
    assert diagnostic["confirmation_eligible"] is False
    assert diagnostic["confirmation_profile"] == "v7-200k"

    confirmation = fixed_eval._require_fixed_200k_protocol(
        flags,
        actor_checkpoint,
        model_checkpoint,
        actor_validation,
        model_validation,
        seeds=range(
            fixed_eval.DEFAULT_SEED_BASE,
            fixed_eval.DEFAULT_SEED_BASE + fixed_eval.DEFAULT_NUM_SEEDS,
        ),
        real_steps_per_seed=fixed_eval.DEFAULT_REAL_STEPS_PER_SEED,
        calibration_unroll=fixed_eval.DEFAULT_CALIBRATION_UNROLL,
        diagnostic=False,
    )
    assert confirmation["evaluation_mode"] == "fixed_200k_confirmation"
    assert confirmation["confirmation_eligible"] is True
    assert confirmation["confirmation_profile"] == "v7-200k"


def test_cli_defaults_preregister_100k_heldout_transitions(tmp_path):
    args = fixed_eval.parse_args(
        [
            "--checkpoint-dir",
            str(tmp_path / "checkpoint"),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )
    assert args.num_seeds == 16
    assert args.real_steps_per_seed == 6250
    assert args.num_seeds * args.real_steps_per_seed == 100000
    assert args.calibration_unroll == 201
    assert args.diagnostic is False
    assert args.confirmation_profile == "v7-200k"
    assert (
        args.expected_rom_sha256
        == fixed_eval.PREREGISTERED_ENDURO_ROM_SHA256
    )

    v10_args = fixed_eval.parse_args(
        [
            "--checkpoint-dir",
            str(tmp_path / "checkpoint"),
            "--output-dir",
            str(tmp_path / "output"),
            "--confirmation-profile",
            "v10-300k",
        ]
    )
    assert v10_args.confirmation_profile == "v10-300k"

    v11_args = fixed_eval.parse_args(
        [
            "--checkpoint-dir",
            str(tmp_path / "checkpoint"),
            "--output-dir",
            str(tmp_path / "output"),
            "--confirmation-profile",
            "v11-300k",
        ]
    )
    assert v11_args.confirmation_profile == "v11-300k"

    v12_args = fixed_eval.parse_args(
        [
            "--checkpoint-dir",
            str(tmp_path / "checkpoint"),
            "--output-dir",
            str(tmp_path / "output"),
            "--confirmation-profile",
            "v12-300k",
        ]
    )
    assert v12_args.confirmation_profile == "v12-300k"

    v13_args = fixed_eval.parse_args(
        [
            "--checkpoint-dir",
            str(tmp_path / "checkpoint"),
            "--output-dir",
            str(tmp_path / "output"),
            "--confirmation-profile",
            "v13-300k",
        ]
    )
    assert v13_args.confirmation_profile == "v13-300k"

    v14_args = fixed_eval.parse_args(
        [
            "--checkpoint-dir",
            str(tmp_path / "checkpoint"),
            "--output-dir",
            str(tmp_path / "output"),
            "--confirmation-profile",
            "v14-300k",
        ]
    )
    assert v14_args.confirmation_profile == "v14-300k"

    v15_args = fixed_eval.parse_args(
        [
            "--checkpoint-dir",
            str(tmp_path / "checkpoint"),
            "--output-dir",
            str(tmp_path / "output"),
            "--confirmation-profile",
            "v15-300k",
        ]
    )
    assert v15_args.confirmation_profile == "v15-300k"

    v16_args = fixed_eval.parse_args(
        [
            "--checkpoint-dir",
            str(tmp_path / "checkpoint"),
            "--output-dir",
            str(tmp_path / "output"),
            "--confirmation-profile",
            "v16-300k",
        ]
    )
    assert v16_args.confirmation_profile == "v16-300k"

    v17_args = fixed_eval.parse_args(
        [
            "--checkpoint-dir",
            str(tmp_path / "checkpoint"),
            "--output-dir",
            str(tmp_path / "output"),
            "--confirmation-profile",
            "v17-300k",
        ]
    )
    assert v17_args.confirmation_profile == "v17-300k"

    with pytest.raises(SystemExit):
        fixed_eval.parse_args(
            [
                "--checkpoint-dir",
                str(tmp_path / "checkpoint"),
                "--output-dir",
                str(tmp_path / "output"),
                "--confirmation-profile",
                "v11-250k",
            ]
        )


def test_file_attestation_and_rom_hash_fail_closed(monkeypatch, tmp_path):
    evaluator = tmp_path / "evaluator.py"
    evaluator.write_bytes(b"version one")
    before = fixed_eval.attest_regular_file(evaluator, label="test evaluator")
    evaluator.write_bytes(b"version two")
    after = fixed_eval.attest_regular_file(evaluator, label="test evaluator")
    with pytest.raises(RuntimeError, match="changed during"):
        fixed_eval.require_attestation_unchanged(
            before, after, label="test evaluator"
        )

    rom = tmp_path / "enduro.bin"
    rom.write_bytes(b"pinned-rom-bytes")
    expected = _sha(rom)
    monkeypatch.setattr(
        fixed_eval, "PREREGISTERED_ENDURO_ROM_SHA256", expected
    )
    state = fixed_eval.validate_enduro_rom(expected, rom_path=rom)
    assert state["sha256"] == expected
    rom.write_bytes(b"different-rom-bytes")
    with pytest.raises(ValueError, match="ROM hash disagrees"):
        fixed_eval.validate_enduro_rom(expected, rom_path=rom)


def _make_staged_outputs(tmp_path: Path):
    output_dir = tmp_path / "output"
    stage_dir = output_dir / "stage"
    stage_dir.mkdir(parents=True)
    staged = {
        "decisions": stage_dir / "decision_rows.csv",
        "summary": stage_dir / "summary.json",
        "manifest": stage_dir / "manifest.json",
    }
    final = {
        "decisions": output_dir / "decision_rows.csv",
        "summary": output_dir / "summary.json",
        "manifest": output_dir / "manifest.json",
    }
    for name, path in staged.items():
        path.write_text(f"new-{name}\n", encoding="utf-8")
    for name, path in final.items():
        path.write_text(f"old-{name}\n", encoding="utf-8")
    return output_dir, staged, final


def test_output_lock_rejects_concurrent_and_stale_writer(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    with fixed_eval.exclusive_output_lock(
        output_dir,
        generation_id="first",
        evaluator_attestation={"sha256": "a" * 64},
    ):
        with pytest.raises(FileExistsError, match="is locked"):
            with fixed_eval.exclusive_output_lock(
                output_dir,
                generation_id="second",
                evaluator_attestation={"sha256": "a" * 64},
            ):
                pass
    assert not (output_dir / fixed_eval.OUTPUT_LOCK_NAME).exists()

    stale = output_dir / fixed_eval.OUTPUT_LOCK_NAME
    stale.write_text("stale", encoding="utf-8")
    with pytest.raises(FileExistsError, match="is locked"):
        with fixed_eval.exclusive_output_lock(
            output_dir,
            generation_id="third",
            evaluator_attestation={"sha256": "a" * 64},
        ):
            pass
    assert stale.exists()


def test_staged_generation_commits_manifest_last_and_releases_lock(tmp_path):
    output_dir, staged, final = _make_staged_outputs(tmp_path)
    with fixed_eval.exclusive_output_lock(
        output_dir,
        generation_id="generation",
        evaluator_attestation={"sha256": "a" * 64},
    ) as lock:
        fixed_eval.commit_staged_generation(staged, final, lock=lock)
        assert lock.committed is True
    assert not (output_dir / fixed_eval.OUTPUT_LOCK_NAME).exists()
    assert final["decisions"].read_text(encoding="utf-8") == "new-decisions\n"
    assert final["summary"].read_text(encoding="utf-8") == "new-summary\n"
    assert final["manifest"].read_text(encoding="utf-8") == "new-manifest\n"


@pytest.mark.parametrize("failure_index", [0, 1, 2])
def test_publication_failure_at_each_replace_retains_lock(
    monkeypatch, tmp_path, failure_index
):
    output_dir, staged, final = _make_staged_outputs(tmp_path)
    real_replace = fixed_eval.os.replace
    calls = {"count": 0}

    def injected_failure(source, destination):
        index = calls["count"]
        calls["count"] += 1
        if index == failure_index:
            raise OSError(f"injected replace failure {failure_index}")
        return real_replace(source, destination)

    monkeypatch.setattr(fixed_eval.os, "replace", injected_failure)
    with pytest.raises(OSError, match="injected replace failure"):
        with fixed_eval.exclusive_output_lock(
            output_dir,
            generation_id="failed-generation",
            evaluator_attestation={"sha256": "a" * 64},
        ) as lock:
            fixed_eval.commit_staged_generation(staged, final, lock=lock)
    assert (output_dir / fixed_eval.OUTPUT_LOCK_NAME).exists()
