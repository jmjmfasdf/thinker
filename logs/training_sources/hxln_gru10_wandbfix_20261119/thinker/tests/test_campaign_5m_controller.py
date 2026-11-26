from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTROLLER_PATH = REPO_ROOT / "campaign_5m_controller.py"
LAUNCHER_PATH = REPO_ROOT / "train_dynamic_imitation_grid70_compute07.slurm"


@pytest.fixture(scope="module")
def campaign_module():
    spec = importlib.util.spec_from_file_location(
        "campaign_5m_controller_tested", CONTROLLER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_grid70_dry_run_is_unique_and_skips_only_two_cells():
    result = subprocess.run(
        ["bash", str(LAUNCHER_PATH), "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = list(csv.DictReader(result.stdout.splitlines(), delimiter="\t"))
    assert len(rows) == 70
    assert [int(row["array_task_id"]) for row in rows] == list(range(70))
    full_indices = [int(row["full_index"]) for row in rows]
    assert len(set(full_indices)) == 70
    assert set(range(72)).difference(full_indices) == {3, 27}
    assert len({row["task_id"] for row in rows}) == 70
    assert len({row["xpid"] for row in rows}) == 70
    for row in rows:
        holdout = int(row["holdout"])
        assert [int(value) for value in row["train_sessions"].split(",")] == [
            value for value in range(1, 5) if value != holdout
        ]
        assert row["xpid"] == (
            f"dynamic-imitation-{row['game']}-voc-sub{row['subject']}-"
            f"leavesession{row['holdout']}"
        )


def test_launcher_resource_and_stop_contracts_are_explicit():
    subprocess.run(["bash", "-n", str(LAUNCHER_PATH)], check=True)
    source = LAUNCHER_PATH.read_text(encoding="utf-8")
    for token in (
        "#SBATCH --array=0-69%2",
        "#SBATCH --gres=gpu:titanrtx:2",
        "#SBATCH --nodelist=bml-compute07",
        "#SBATCH --no-requeue",
        "--total_steps 100000000",
        "--schedule_total_steps 100000000",
        "--checkpoint_interval 250000",
        "--max_search_steps 30",
        "--max_depth 30",
        "--model_unroll_len 30",
        "--rec_t 20",
        "--dynamic_voc_mode control",
        "--icopro_margin_coef 0.0",
        "--icopro_pvp_coef 0.0",
        "--action_prior_weight 0.0",
        "--use_wandb True",
    ):
        assert token in source


def _flags(module, expected, run_dir, behavior_root, command):
    return {
        "xpid": expected.xpid,
        "ckpdir": str(run_dir),
        "cmd": command,
        "name": expected.env_name,
        "savedir": str(run_dir.parent),
        "icopro_data_path": str(behavior_root),
        "icopro_game_id": expected.game_id,
        "icopro_subjects": str(expected.subject),
        "icopro_train_sessions": ",".join(map(str, expected.train_sessions)),
        "icopro_holdout_sessions": str(expected.holdout_session),
        "dynamic_voc_mode": "control",
        "model_state_projection": "clamp",
        "total_steps": module.TOTAL_STEPS,
        "schedule_total_steps": module.SCHEDULE_TOTAL_STEPS,
        "checkpoint_interval": module.CHECKPOINT_INTERVAL,
        "max_search_steps": module.MAX_SEARCH_STEPS,
        "max_depth": module.MAX_DEPTH,
        "model_unroll_len": module.MODEL_UNROLL_LEN,
        "rec_t": module.REC_T,
        "base_seed": module.BASE_SEED,
        "icopro_margin_coef": 0.0,
        "icopro_pvp_coef": 0.0,
        "action_prior_weight": 0.0,
        "use_wandb": True,
    }


def _patch_reference(monkeypatch, module, flags):
    static = {
        key: value
        for key, value in flags.items()
        if key not in module.VARIABLE_FLAG_FIELDS
    }
    monkeypatch.setattr(module, "REFERENCE_FLAGS_KEY_COUNT", len(flags))
    monkeypatch.setattr(
        module, "REFERENCE_FLAGS_KEYSET_SHA256", module._canonical_json_sha256(sorted(flags))
    )
    monkeypatch.setattr(
        module, "REFERENCE_STATIC_FLAGS_SHA256", module._canonical_json_sha256(static)
    )


def _checkpoint_payload(role, flags, signature):
    real_step = 5_000_000
    if role == "actor":
        return {
            "real_step": real_step,
            "step": 123,
            "flags": flags,
            "dynamic_search": True,
            "dynamic_factorized_control": True,
            "dynamic_voc_mode": "control",
            "voc_gate_policy_schema_version": 3,
            "voc_actor_policy_terminal": False,
            "imitation_data_root": flags["icopro_data_path"],
            "imitation_data_signature": signature,
            "imitation_update_count": 1,
            "actor_net_state_dict": {"weight": torch.ones(1)},
            "actor_net_optimizer_state_dict": {"state": {}},
            "voc_optimizer_state_dict": {"state": {}},
            "voc_gate_optimizer_state_dict": {"state": {}},
            "voc_ema_gate_head_state_dict": {"weight": torch.ones(1)},
            "actor_net_scheduler_state_dict": {"last_epoch": real_step},
            "voc_scheduler_state_dict": {"last_epoch": real_step},
            "voc_gate_scheduler_state_dict": {"last_epoch": real_step},
        }
    return {
        "real_step": real_step,
        "step": 456,
        "flags": flags,
        "model_net_state_dict": {"weight": torch.ones(1)},
        "model_net_optimizer_p_state_dict": {"state": {}},
        "model_net_optimizer_m_state_dict": {"state": {}},
        "model_net_scheduler_p_state_dict": {"last_epoch": real_step},
        "model_net_scheduler_m_state_dict": {"last_epoch": real_step},
    }


def test_double_validation_and_atomic_publish(monkeypatch, tmp_path, campaign_module):
    module = campaign_module
    expected = module.ExpectedRun(
        xpid="dynamic-imitation-enduro-voc-sub2-leavesession1",
        game_label="enduro",
        env_name="Enduro-v5",
        game_id=0,
        subject=2,
        train_sessions=(2, 3, 4),
        holdout_session=1,
    )
    command_cwd = tmp_path / "p" / "thinker"
    run_dir = tmp_path / "p" / "logs" / "thinker" / expected.xpid
    behavior_root = tmp_path / "p" / "behavioral_data_block"
    command_cwd.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    behavior_root.mkdir(parents=True)
    command = "train.py --synthetic campaign-test"
    signature = "a" * 64
    flags = _flags(module, expected, run_dir, behavior_root, command)
    _patch_reference(monkeypatch, module, flags)
    for role in ("actor", "model"):
        torch.save(
            _checkpoint_payload(role, flags, signature),
            run_dir / f"ckp_{role}.tar_step_5000000",
        )
        torch.save(
            _checkpoint_payload(role, flags, signature),
            run_dir / f"ckp_{role}.tar_step_5000001",
        )
    (run_dir / "config_c.yaml").write_text(
        yaml.safe_dump(flags, sort_keys=True), encoding="utf-8"
    )
    with (run_dir / "logs.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "real_step",
                "interaction/noop_action_index",
                "interaction/noop_supported",
                "interaction/real_action_count",
                "interaction/noop_count",
                "interaction/noop_frequency",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "real_step": 5_000_000,
                "interaction/noop_action_index": 0,
                "interaction/noop_supported": 1,
                "interaction/real_action_count": 100,
                "interaction/noop_count": 25,
                "interaction/noop_frequency": 0.25,
            }
        )

    validation = module.double_validate_checkpoint_pair(
        run_dir,
        expected,
        stable_seconds=0,
        staged_behavior_root=behavior_root,
        expected_train_cmd=command,
        command_cwd=command_cwd,
        expected_imitation_data_signature=signature,
        sleeper=lambda _: None,
    )
    assert validation["artifacts"]["actor"]["step"] == 5_000_000
    assert validation["artifacts"]["model"]["step"] == 5_000_000
    metrics = module.validate_interaction_metrics(
        run_dir, checkpoint_step=5_000_000
    )
    assert metrics["checkpoint_row"]["noop_frequency"] == 0.25

    source_root = tmp_path / "source"
    original_behavior_root = tmp_path / "behavior"
    source_root.mkdir()
    original_behavior_root.mkdir()
    source_manifest = tmp_path / "source.sha256"
    behavior_manifest = tmp_path / "behavior.sha256"
    source_manifest.write_text("source manifest\n", encoding="utf-8")
    behavior_manifest.write_text("behavior manifest\n", encoding="utf-8")
    campaign_root = tmp_path / "campaign"
    publish_root = tmp_path / "published"
    publish_root.mkdir()
    target, milestone = module.publish_canonical_run(
        run_dir=run_dir,
        campaign_root=campaign_root,
        publish_root=publish_root,
        expected=expected,
        validation=validation,
        termination={"expected": True, "survivors_after_kill": []},
        source_root=source_root,
        source_manifest=source_manifest,
        behavior_root=original_behavior_root,
        behavior_manifest=behavior_manifest,
        launcher_path=LAUNCHER_PATH,
        controller_path=CONTROLLER_PATH,
        interaction_metrics=metrics,
        job_key="123_4",
        array_job_id="123",
        array_task_id=4,
    )
    assert target == publish_root / expected.xpid
    assert milestone["status"] == "VERIFIED_5M_STOPPED"
    assert (target / "milestone_5000000.json").is_file()
    assert not (target / "finish").exists()
    assert not (target / "ckp_actor.tar").exists()
    assert not (target / "ckp_model.tar").exists()
    for role in ("actor", "model"):
        checkpoint = target / f"ckp_{role}.tar_step_5000000"
        assert checkpoint.stat().st_nlink == 1
    external = campaign_root / "milestones" / f"{expected.task_id}.json"
    assert json.loads(external.read_text())["canonical_path"] == str(target)
