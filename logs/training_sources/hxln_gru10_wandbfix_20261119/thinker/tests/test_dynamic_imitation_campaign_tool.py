import csv
import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml

from tools import dynamic_imitation_campaign as campaign


def _data_manifest(tmp_path):
    cells = []
    folds = []
    for game in campaign.GAMES:
        for subject in campaign.SUBJECTS:
            for session in campaign.SESSIONS:
                count = campaign.EXPECTED_SESSION_FILE_COUNTS[session]
                cells.append(
                    {
                        "game_id": game.game_id,
                        "subject": subject,
                        "session": session,
                        "files": [
                            f"g{game.game_id}/s{subject}/h{session}/b{i}.npz"
                            for i in range(count)
                        ],
                        "file_count": count,
                    }
                )
            for leave_session in campaign.SESSIONS:
                folds.append(
                    {
                        "game_id": game.game_id,
                        "subject": subject,
                        "leave_session": leave_session,
                        "train_sessions": [
                            value
                            for value in campaign.SESSIONS
                            if value != leave_session
                        ],
                        "scored_length_4_train_windows": 123,
                        "behavioral_data_signature": hashlib.sha256(
                            f"{game.game_id}/{subject}/{leave_session}".encode()
                        ).hexdigest(),
                    }
                )
    payload = {
        "schema_version": 1,
        "root": str(tmp_path),
        "source_sha256_manifest": str(tmp_path / "behavior.sha256"),
        "source_sha256_manifest_sha256": "a" * 64,
        "cell_count": 72,
        "cells": cells,
        "folds": folds,
    }
    return campaign._attach_digest(payload, "manifest_sha256")


def test_campaign_mapping_has_72_tasks_and_exact_70_pending(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_manifest = tmp_path / "source.sha256"
    source_manifest.write_text("", encoding="utf-8")
    manifest, ledger_bytes = campaign.build_campaign_payloads(
        _data_manifest(tmp_path),
        source_root=source_root,
        source_manifest=source_manifest,
        source_manifest_sha256="b" * 64,
    )

    assert manifest["task_count"] == 72
    assert manifest["existing_milestone_count"] == 2
    assert manifest["pending_task_count"] == 70
    pending = [json.loads(line) for line in ledger_bytes.splitlines()]
    assert [row["array_task_id"] for row in pending] == list(range(70))
    assert {row["campaign_index"] for row in pending} == set(range(72)) - {3, 27}
    assert all(
        row["logical"]["train_sessions"]
        == [value for value in campaign.SESSIONS if value != row["logical"]["leave_session"]]
        for row in manifest["tasks"]
    )
    assert manifest["tasks"][3]["logical"]["xpid"] == (
        "dynamic-imitation-enduro-voc-sub1-leavesession4"
    )
    assert manifest["tasks"][27]["logical"]["xpid"] == (
        "dynamic-imitation-pong-voc-sub1-leavesession4"
    )
    assert manifest["tasks"][0]["protocol"]["planned_total_steps"] == 100_000_000
    assert manifest["tasks"][0]["protocol"]["stop_policy"][
        "checkpoint_window_end_exclusive"
    ] == 5_250_000


def _checkpoint_flags(xpid):
    return {
        "name": "Enduro-v5",
        "icopro_game_id": 0,
        "icopro_subjects": "1",
        "icopro_train_sessions": "1,2,3",
        "icopro_holdout_sessions": "4",
        "max_search_steps": 30,
        "max_depth": 30,
        "model_unroll_len": 30,
        "action_prior_weight": 0.0,
        "icopro_margin_coef": 0.0,
        "icopro_pvp_coef": 0.0,
        "icopro_action_diff_coef": 1.0,
        "base_seed": 1,
        "dynamic_search": True,
        "dynamic_factorized_control": True,
        "dynamic_voc_mode": "control",
        "schedule_total_steps": 100_000_000,
        "total_steps": 100_000_000,
        "batch_length": 4,
        "frame_stack_n": 4,
        "grayscale": False,
        "train_model": True,
        "envpool": True,
        "xpid": xpid,
        "ckpdir": "/scratch/run",
        "savedir": "/scratch",
    }


def _write_csv(path, *, actor):
    if actor:
        header = [
            "real_step",
            "interaction/noop_action_index",
            "interaction/noop_supported",
            "interaction/real_action_count",
            "interaction/noop_count",
            "interaction/noop_frequency",
        ]
        rows = [
            [4_999_000, 0, 1, 10, 2, 0.2],
            [5_000_000, 0, 1, 16, 4, 0.25],
            [5_001_000, 0, 1, 20, 5, 0.25],
        ]
    else:
        header = ["real_step", "model/loss"]
        rows = [[4_999_500, 1.0], [5_000_500, 0.9]]
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(header)
        writer.writerows(rows)


def _fake_source(tmp_path):
    xpid = "legacy-enduro"
    source_logs_root = tmp_path / "logs"
    source = source_logs_root / xpid
    source.mkdir(parents=True)
    flags = _checkpoint_flags(xpid)
    actor = {
        "real_step": campaign.TARGET_REAL_STEP,
        "flags": flags,
        "actor_net_state_dict": {"w": torch.ones(1)},
        "actor_net_optimizer_state_dict": {},
        "actor_net_scheduler_state_dict": {},
        "imitation_data_signature": "c" * 64,
        "imitation_rng_state": {},
        "python_rng_state": (),
        "numpy_rng_state": (),
        "torch_rng_state": torch.zeros(1, dtype=torch.uint8),
    }
    model = {
        "real_step": campaign.TARGET_REAL_STEP,
        "flags": flags,
        "model_net_state_dict": {"w": torch.ones(1)},
        "model_net_optimizer_p_state_dict": {},
        "model_net_scheduler_p_state_dict": {},
        "model_net_optimizer_m_state_dict": {},
        "model_net_scheduler_m_state_dict": {},
    }
    actor_path = source / f"ckp_actor.tar_step_{campaign.TARGET_REAL_STEP}"
    model_path = source / f"ckp_model.tar_step_{campaign.TARGET_REAL_STEP}"
    torch.save(actor, actor_path)
    torch.save(model, model_path)
    with (source / "config_c.yaml").open("w", encoding="utf-8") as output:
        yaml.safe_dump(flags, output)
    for name in ("meta.json", "meta_model.json"):
        (source / name).write_text(json.dumps({"args": flags}), encoding="utf-8")
    _write_csv(source / "logs.csv", actor=True)
    _write_csv(source / "logs_model.csv", actor=False)
    spec = campaign.ExistingRunSpec(
        game_id=0,
        source_xpid=xpid,
        canonical_xpid="dynamic-imitation-enduro-voc-sub1-leavesession4",
        actor_sha256=campaign._sha256_file(actor_path),
        model_sha256=campaign._sha256_file(model_path),
    )
    return source_logs_root, spec


def test_canonicalization_is_exact_truncated_atomic_and_no_finish(tmp_path, monkeypatch):
    source_logs_root, spec = _fake_source(tmp_path)
    source_root = tmp_path / "source-root"
    behavior_root = tmp_path / "behavior-root"
    source_root.mkdir()
    behavior_root.mkdir()
    source_manifest = tmp_path / "source.sha256"
    behavior_manifest = tmp_path / "behavior.sha256"
    source_manifest.write_text("", encoding="utf-8")
    behavior_manifest.write_text("", encoding="utf-8")
    monkeypatch.setattr(campaign, "_verify_external_manifest", lambda *_args: {})
    monkeypatch.setattr(campaign, "_verify_tree_against_manifest", lambda *_args: None)
    monkeypatch.setattr(
        campaign,
        "_verify_behavior_signature",
        lambda *_args: {
            "imitation_data_signature": "c" * 64,
            "train_file_count": 8,
            "train_window_count": 10,
        },
    )
    output_root = tmp_path / "canonical"

    destination = campaign.canonicalize_existing_run(
        spec,
        source_logs_root=source_logs_root,
        output_root=output_root,
        source_root=source_root,
        source_manifest=source_manifest,
        source_manifest_sha256="0" * 64,
        behavior_root=behavior_root,
        behavior_sha256_manifest=behavior_manifest,
        behavior_sha256_manifest_sha256="1" * 64,
    )

    assert destination.name == spec.canonical_xpid
    assert not (destination / "finish").exists()
    assert not (destination / "ckp_actor.tar").exists()
    assert not (destination / "ckp_model.tar").exists()
    with (destination / "logs.csv").open(newline="") as source:
        actor_rows = list(csv.DictReader(source))
    with (destination / "logs_model.csv").open(newline="") as source:
        model_rows = list(csv.DictReader(source))
    assert [int(row["real_step"]) for row in actor_rows] == [4_999_000, 5_000_000]
    assert [int(row["real_step"]) for row in model_rows] == [4_999_500]
    milestone = json.loads((destination / campaign.MILESTONE_NAME).read_text())
    assert milestone["finish_present"] is False
    assert milestone["training_completion_claimed"] is False
    assert milestone["validation"]["online_noop_at_target"]["noop_frequency"] == 0.25
    campaign.verify_canonical_run(destination, spec)
    with pytest.raises(FileExistsError):
        campaign.canonicalize_existing_run(
            spec,
            source_logs_root=source_logs_root,
            output_root=output_root,
            source_root=source_root,
            source_manifest=source_manifest,
            source_manifest_sha256="0" * 64,
            behavior_root=behavior_root,
            behavior_sha256_manifest=behavior_manifest,
            behavior_sha256_manifest_sha256="1" * 64,
        )


def test_known_hash_mismatch_publishes_nothing(tmp_path, monkeypatch):
    source_logs_root, good_spec = _fake_source(tmp_path)
    bad_spec = campaign.ExistingRunSpec(
        game_id=good_spec.game_id,
        source_xpid=good_spec.source_xpid,
        canonical_xpid=good_spec.canonical_xpid,
        actor_sha256="0" * 64,
        model_sha256=good_spec.model_sha256,
    )
    source_root = tmp_path / "source-root"
    behavior_root = tmp_path / "behavior-root"
    source_root.mkdir()
    behavior_root.mkdir()
    source_manifest = tmp_path / "source.sha256"
    behavior_manifest = tmp_path / "behavior.sha256"
    source_manifest.write_text("", encoding="utf-8")
    behavior_manifest.write_text("", encoding="utf-8")
    monkeypatch.setattr(campaign, "_verify_external_manifest", lambda *_args: {})
    monkeypatch.setattr(campaign, "_verify_tree_against_manifest", lambda *_args: None)
    output_root = tmp_path / "canonical"

    with pytest.raises(campaign.CampaignError, match="known actor hash mismatch"):
        campaign.canonicalize_existing_run(
            bad_spec,
            source_logs_root=source_logs_root,
            output_root=output_root,
            source_root=source_root,
            source_manifest=source_manifest,
            source_manifest_sha256="0" * 64,
            behavior_root=behavior_root,
            behavior_sha256_manifest=behavior_manifest,
            behavior_sha256_manifest_sha256="1" * 64,
        )
    assert not (output_root / bad_spec.canonical_xpid).exists()
