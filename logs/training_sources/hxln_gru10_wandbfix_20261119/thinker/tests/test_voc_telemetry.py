import hashlib
import json
import math
import os
import stat

import pytest
import torch

from thinker import voc_telemetry as telemetry


_ZERO_HASH = "0" * 64


def _td_inputs(*, train=True):
    optimized_t, batch = 2, 16
    target = torch.zeros((optimized_t, batch), dtype=torch.float32)
    q_values = torch.zeros((optimized_t, batch, 2), dtype=torch.float32)
    gate_action = torch.arange(batch).remainder(2).repeat(optimized_t, 1)
    control_action = torch.where(
        gate_action == 0,
        torch.full_like(gate_action, 0),
        torch.full_like(gate_action, 2),
    )
    search_steps = (gate_action == 0).to(torch.int64)
    support = torch.ones((optimized_t, batch), dtype=torch.bool)
    empty = torch.zeros_like(support)
    return {
        "target": target,
        "online_q_values": q_values,
        "ema_q_values": q_values.clone(),
        "valid_mask": support,
        "train_mask": support if train else empty,
        "holdout_mask": empty if train else support,
        "gate_action": gate_action,
        "control_action": control_action,
        "search_steps": search_steps,
    }


def _transaction_rows(*, status="no_support"):
    if status not in ("no_support", "stepped", "amp_skip"):
        raise AssertionError(
            "unit helper supports no_support, stepped, or amp_skip"
        )
    train = status != "no_support"
    td_rows = telemetry.build_td_cell_rows(
        transaction_id=1,
        source_policy_version=0,
        published_policy_version=1,
        real_step_after=32,
        **_td_inputs(train=train),
    )
    update = int(status == "stepped")
    replay_row = telemetry.build_replay_row(
        transaction_id=1,
        source_policy_version=0,
        published_policy_version=1,
        replay_t=3,
        optimized_t=2,
        replay_b=16,
        actor_ids=tuple(range(15, -1, -1)),
        real_step_before=0,
        real_step_delta=32,
        real_step_after=32,
        valid_count=32,
        train_count=32 if train else 0,
        holdout_count=0 if train else 32,
        train_continue_count=16 if train else 0,
        train_stop_count=16 if train else 0,
        holdout_continue_count=0 if train else 16,
        holdout_stop_count=0 if train else 16,
        q_status=status,
        voc_update_count_before=0,
        voc_update_count_after=update,
        ema_update_count_before=0,
        ema_update_count_after=update,
        projection_count_before=0,
        projection_count_after=update,
        q_scheduler_last_epoch_before=0,
        q_scheduler_last_epoch_after=32 if update else 0,
        q_scheduler_step_count_before=1,
        q_scheduler_step_count_after=1 + update,
        q_lr_before=1.0e-3,
        q_lr_used=1.0e-3,
        q_lr_after=0.0 if update else 1.0e-3,
        publication_count_after=1,
        ack_count=1,
        terminal=True,
        actor_state_sha256=_ZERO_HASH,
        publication_history_sha256=_ZERO_HASH,
    )
    diagnostics = None
    if status == "stepped":
        weight = torch.zeros((2, 3), dtype=torch.float32)
        bias = torch.zeros((2,), dtype=torch.float32)
        diagnostics = telemetry.build_stepped_q_diagnostics(
            clip_scale=1.0,
            raw_preclip=(weight, bias),
            raw_postclip=(weight.clone(), bias.clone()),
            md_postclip=(weight.clone(), bias.clone()),
            adam_m_before=(weight.clone(), bias.clone()),
            adam_v_before=(weight.clone(), bias.clone()),
            adam_m_after=(weight.clone(), bias.clone()),
            adam_v_after=(weight.clone(), bias.clone()),
            coordinate_delta=(weight.clone(), bias.clone()),
            mapped_delta=(weight.clone(), bias.clone()),
            q_lr_used=1.0e-3,
            adam_step_after=1,
        )
    q_row = telemetry.build_q_transaction_row(
        transaction_id=1,
        source_policy_version=0,
        published_policy_version=1,
        real_step_after=32,
        q_status=status,
        q_attempted=train,
        q_optimizer_committed=status == "stepped",
        q_loss_sum=0.0,
        clip_limit=24.0,
        amp_scale_before=256.0,
        amp_scale_after=128.0 if status == "amp_skip" else 256.0,
        nonfinite_gradient_parameter_count=1 if status == "amp_skip" else 0,
        adam_step_before=0,
        adam_step_after=update,
        diagnostics=diagnostics,
    )
    return td_rows, replay_row, q_row


def _build_terminal_set(tmp_path, monkeypatch, *, status="no_support"):
    writer = telemetry.Schema13TelemetryWriter(
        tmp_path,
        xpid="schema13-unit",
        actor_unroll_len=2,
        stage_total_steps=32,
    )
    td_rows, replay_row, q_row = _transaction_rows(status=status)
    commit = writer.append_transaction(
        td_rows=td_rows,
        replay_row=replay_row,
        q_row=q_row,
        terminal=True,
        actor_state_sha256=_ZERO_HASH,
        publication_history_sha256=_ZERO_HASH,
    )

    # Unit-isolate the legacy-log binding without duplicating the enormous
    # frozen V19 header.  Separate constants/tests bind the production oracle.
    log_header = b"_tick,value,actor/voc_q_loss\n"
    log_payload = log_header + b"0,1,0.0\r\n"
    log_path = tmp_path / telemetry.LEGACY_LOG_FILENAME
    log_path.write_bytes(log_payload)
    monkeypatch.setattr(telemetry, "LEGACY_LOG_HEADER_SIZE", len(log_header))
    monkeypatch.setattr(
        telemetry,
        "LEGACY_LOG_HEADER_SHA256",
        hashlib.sha256(log_header).hexdigest(),
    )
    monkeypatch.setattr(telemetry, "LEGACY_LOG_COLUMN_COUNT", 3)
    evidence = writer.seal(
        terminal_real_step=32,
        terminal_policy_version=1,
        terminal_publication_count=1,
        terminal_ack_count=1,
        legacy_actor_log_path=log_path,
    )
    return commit, evidence


def test_import_is_leaf_and_headers_match_independent_known_answers():
    assert "torch" not in telemetry.__dict__
    expected = (
        (
            telemetry.TD_HEADER,
            22,
            300,
            "37c82eea9a7bf7cbe05ee74ffb2b37b6190e4b715b05afbea5b5a06c406473fa",
        ),
        (
            telemetry.REPLAY_HEADER,
            39,
            713,
            "eed6226a8a591289125c7f5389b7d6705332b11e32746f103093b5dcd71592e2",
        ),
        (
            telemetry.Q_HEADER,
            59,
            1603,
            "e1574cf8c81306818abc2369b5270f98c74f3e3190cb8cb3ceddb000ad4096b3",
        ),
        (
            telemetry.COMMIT_HEADER,
            22,
            410,
            "9105b143dfd260a4c491a2821757c14ded2a32c74207e6f4e7140b717ae62929",
        ),
    )
    for header, columns, size, digest in expected:
        assert header.endswith(b"\n") and b"\r" not in header
        assert len(header[:-1].split(b",")) == columns
        assert len(header) == size
        assert hashlib.sha256(header).hexdigest() == digest
    assert telemetry.LEGACY_LOG_HEADER_SIZE == 43550
    assert telemetry.LEGACY_LOG_COLUMN_COUNT == 922
    assert telemetry.LEGACY_LOG_HEADER_SHA256 == (
        "82488231a631ca3571379e973122dd107007d14f4756fd839a811851dc6accbc"
    )


def test_canonical_float_csv_and_json_grammar_is_strict():
    assert telemetry.canonical_float(-0.0) == "0x0.0p+0"
    assert telemetry.canonical_float(0.5) == "0x1.0000000000000p-1"
    assert telemetry.parse_canonical_float("0x1.0000000000000p-1") == 0.5
    for spelling in ("0.5", "-0x0.0p+0", "0X1.0000000000000P-1", "nan"):
        with pytest.raises(ValueError):
            telemetry.parse_canonical_float(spelling)
    assert telemetry.canonical_json_bytes({"z": 1, "a": True}) == (
        b'{"a":true,"z":1}\n'
    )
    row = _transaction_rows()[2]
    encoded = telemetry.encode_csv_row(telemetry.Q_FIELDS, row)
    assert b",NA," in encoded and encoded.endswith(b"\n")
    assert b"\r" not in encoded and b'"' not in encoded


def test_stepped_q_rejects_zero_clip_scale():
    td_rows, replay_row, q_row = _transaction_rows(status="stepped")
    assert len(td_rows) == 720
    forged = dict(q_row)
    forged["clip_scale"] = 0.0
    with pytest.raises(ValueError, match=r"clip_scale must lie in \(0,1\]"):
        telemetry._validate_q_row(forged, replay_row)


def test_td_cube_exact_order_boundaries_and_fp32_reductions():
    target = torch.tensor([[-4.0, -2.0, -1.0, -0.5, 0.0, 0.5]])
    q_values = torch.zeros((1, 6, 2), dtype=torch.float32)
    rows = telemetry.build_td_cell_rows(
        transaction_id=1,
        source_policy_version=0,
        published_policy_version=1,
        real_step_after=6,
        target=target,
        online_q_values=q_values,
        ema_q_values=q_values,
        valid_mask=torch.ones((1, 6), dtype=torch.bool),
        train_mask=torch.ones((1, 6), dtype=torch.bool),
        holdout_mask=torch.zeros((1, 6), dtype=torch.bool),
        gate_action=torch.ones((1, 6), dtype=torch.int64),
        control_action=torch.full((1, 6), 2, dtype=torch.int64),
        search_steps=torch.tensor([[0, 1, 2, 4, 8, 16]]),
    )
    assert len(rows) == 720
    actual_order = tuple(
        tuple(
            row[name]
            for name in (
                "q_source",
                "split",
                "selected_action",
                "depth_bin",
                "td_sign",
                "abs_td_bin",
            )
        )
        for row in rows
    )
    assert actual_order == telemetry.TD_CATEGORY_ORDER
    populated = [row for row in rows if row["count"]]
    assert len(populated) == 12
    online = [row for row in populated if row["q_source"] == "online"]
    assert [row["depth_bin"] for row in online] == [
        "0", "1", "2_3", "4_7", "8_15", "16_plus"
    ]
    assert [row["abs_td_bin"] for row in online] == [
        "4_inf", "2_4", "1_2", "0p5_1", "0_0p5", "0p5_1"
    ]
    assert [row["td_sign"] for row in online] == [
        "negative", "negative", "negative", "negative", "zero", "positive"
    ]
    assert all(
        value == 0.0 and not torch.signbit(torch.tensor(value)).item()
        for row in rows
        if row["count"] == 0
        for name, value in row.items()
        if name.startswith("sum_") or name == "max_abs_td"
    )


def test_td_builder_rejects_action_mapping_and_valid_partition_drift():
    inputs = _td_inputs(train=True)
    inputs["control_action"] = inputs["control_action"].clone()
    inputs["control_action"][0, 0] = 1  # RESET is still CONTINUE.
    assert len(telemetry.build_td_cell_rows(
        transaction_id=1,
        source_policy_version=0,
        published_policy_version=1,
        real_step_after=32,
        **inputs,
    )) == 720

    bad_action = dict(inputs)
    bad_action["gate_action"] = inputs["gate_action"].clone()
    bad_action["gate_action"][0, 0] = 1
    with pytest.raises(ValueError, match="PROCEED/RESET"):
        telemetry.build_td_cell_rows(
            transaction_id=1,
            source_policy_version=0,
            published_policy_version=1,
            real_step_after=32,
            **bad_action,
        )

    bad_partition = dict(inputs)
    bad_partition["valid_mask"] = inputs["valid_mask"].clone()
    bad_partition["valid_mask"][0, 0] = False
    with pytest.raises(ValueError, match="partition valid"):
        telemetry.build_td_cell_rows(
            transaction_id=1,
            source_policy_version=0,
            published_policy_version=1,
            real_step_after=32,
            **bad_partition,
        )

    for name in ("valid_mask", "train_mask", "holdout_mask"):
        bad_dtype = dict(inputs)
        bad_dtype[name] = inputs[name].to(torch.float32)
        with pytest.raises(TypeError, match="exact bool dtype"):
            telemetry.build_td_cell_rows(
                transaction_id=1,
                source_policy_version=0,
                published_policy_version=1,
                real_step_after=32,
                **bad_dtype,
            )
    for name in ("gate_action", "control_action", "search_steps"):
        bad_dtype = dict(inputs)
        bad_dtype[name] = inputs[name].to(torch.float32).add_(0.9)
        with pytest.raises(TypeError, match="integer dtype"):
            telemetry.build_td_cell_rows(
                transaction_id=1,
                source_policy_version=0,
                published_policy_version=1,
                real_step_after=32,
                **bad_dtype,
            )


def test_stepped_diagnostic_reduction_matches_independent_l2_oracle():
    raw_weight = torch.tensor([[3.0, 4.0], [0.0, 12.0]])
    raw_bias = torch.tensor([5.0, 0.0])
    zero_weight = torch.zeros_like(raw_weight)
    zero_bias = torch.zeros_like(raw_bias)
    scale = torch.tensor([0x3F3504F3], dtype=torch.int32).view(torch.float32)[0]

    def md(raw):
        return torch.stack(
            (scale * (raw[0] + raw[1]), scale * (raw[0] - raw[1]))
        )

    md_weight, md_bias = md(raw_weight), md(raw_bias)
    m_weight, m_bias = 0.1 * md_weight, 0.1 * md_bias
    v_weight, v_bias = 0.001 * md_weight.square(), 0.001 * md_bias.square()
    lr = 1.0e-3

    def coordinate_delta(m_after, v_after):
        normalized = (m_after / 0.1) / (torch.sqrt(v_after / 0.001) + 1.0e-8)
        return (-lr * normalized).to(torch.float32)

    delta_weight = coordinate_delta(m_weight, v_weight)
    delta_bias = coordinate_delta(m_bias, v_bias)

    def mapped(delta):
        return torch.stack(
            (scale * (delta[0] + delta[1]), scale * (delta[0] - delta[1]))
        )

    diagnostics = telemetry.build_stepped_q_diagnostics(
        clip_scale=1.0,
        raw_preclip=(raw_weight, raw_bias),
        raw_postclip=(raw_weight, raw_bias),
        md_postclip=(md_weight, md_bias),
        adam_m_before=(zero_weight, zero_bias),
        adam_v_before=(zero_weight, zero_bias),
        adam_m_after=(m_weight, m_bias),
        adam_v_after=(v_weight, v_bias),
        coordinate_delta=(delta_weight, delta_bias),
        mapped_delta=(mapped(delta_weight), mapped(delta_bias)),
        q_lr_used=lr,
        adam_step_after=1,
    )
    assert diagnostics["raw_preclip_total_l2"] == 13.92838827718412
    assert diagnostics["raw_preclip_weight_continue_l2"] == 5.0
    assert diagnostics["raw_preclip_weight_stop_l2"] == 12.0
    assert diagnostics["raw_preclip_bias_continue_l2"] == 5.0
    assert diagnostics["raw_preclip_bias_stop_l2"] == 0.0


def test_raw_total_l2_uses_one_weight_then_bias_fsum_without_nested_rounding():
    raw_weight = torch.tensor([
        [-40.57423782348633, 113.40515899658203, -111.15385437011719,
         35.00675964355469, -77.02728271484375, -14.726622581481934,
         62.7178955078125],
        [109.34529113769531, 9.390315055847168, 123.80663299560547,
         -134.58941650390625, 51.189727783203125, -69.32776641845703,
         -16.676015853881836],
    ], dtype=torch.float32)
    raw_bias = torch.tensor(
        [-99.98821258544922, -164.7566375732422], dtype=torch.float32
    )
    zero_weight = torch.zeros_like(raw_weight)
    zero_bias = torch.zeros_like(raw_bias)
    scale = torch.tensor([0x3F3504F3], dtype=torch.int32).view(torch.float32)[0]

    def md(raw):
        return torch.stack((scale * (raw[0] + raw[1]), scale * (raw[0] - raw[1])))

    md_weight, md_bias = md(raw_weight), md(raw_bias)
    m_weight, m_bias = torch.lerp(zero_weight, md_weight, 0.1), torch.lerp(
        zero_bias, md_bias, 0.1
    )
    v_weight = zero_weight.mul(0.999).addcmul(md_weight, md_weight, value=0.001)
    v_bias = zero_bias.mul(0.999).addcmul(md_bias, md_bias, value=0.001)
    lr = 3.0e-4

    def delta(m_after, v_after):
        normalized = (m_after / 0.1) / (torch.sqrt(v_after / 0.001) + 1.0e-8)
        return (-lr * normalized).to(torch.float32)

    delta_weight, delta_bias = delta(m_weight, v_weight), delta(m_bias, v_bias)
    diagnostics = telemetry.build_stepped_q_diagnostics(
        clip_scale=1.0,
        raw_preclip=(raw_weight, raw_bias),
        raw_postclip=(raw_weight.clone(), raw_bias.clone()),
        md_postclip=(md_weight, md_bias),
        adam_m_before=(zero_weight, zero_bias),
        adam_v_before=(zero_weight, zero_bias),
        adam_m_after=(m_weight, m_bias),
        adam_v_after=(v_weight, v_bias),
        coordinate_delta=(delta_weight, delta_bias),
        mapped_delta=(md(delta_weight), md(delta_bias)),
        q_lr_used=lr,
        adam_step_after=1,
    )
    direct = math.sqrt(math.fsum(
        float(value) * float(value)
        for tensor in (raw_weight, raw_bias)
        for value in tensor.reshape(-1).tolist()
    ))
    nested = math.sqrt(math.fsum(
        telemetry.l2_norm(tensor) ** 2 for tensor in (raw_weight, raw_bias)
    ))
    assert direct.hex() == "0x1.6679fefebb8bep+8"
    assert nested.hex() == "0x1.6679fefebb8bfp+8"
    assert diagnostics["raw_preclip_total_l2"].hex() == direct.hex()
    assert diagnostics["raw_postclip_total_l2"].hex() == direct.hex()


@pytest.mark.parametrize("status", ["no_support", "stepped", "amp_skip"])
def test_terminal_writer_manifest_and_exact10_validate_end_to_end(
    tmp_path, monkeypatch, status
):
    commit, evidence = _build_terminal_set(
        tmp_path, monkeypatch, status=status
    )
    assert commit["td_first_data_row"] == 1
    assert commit["td_data_row_count"] == 720
    assert commit["replay_first_data_row"] == 1
    assert commit["q_first_data_row"] == 1
    assert set(evidence) == {
        "telemetry_schema_version",
        "gate_schema",
        "manifest_name",
        "manifest_sha256",
        "manifest_size",
        "transaction_count",
        "terminal_policy_version",
        "terminal_real_step",
        "actor_state_sha256",
        "publication_history_sha256",
    }
    assert evidence["transaction_count"] == 1
    assert evidence["terminal_policy_version"] == 1
    manifest_path = tmp_path / telemetry.MANIFEST_FILENAME
    manifest_payload = manifest_path.read_bytes()
    manifest = json.loads(manifest_payload)
    assert set(manifest) == telemetry._MANIFEST_FIELDS
    assert [record["name"] for record in manifest["artifacts"]] == list(
        telemetry.SIDECAR_FILENAMES
    )
    assert set(manifest["last_commit"]) == telemetry._LAST_COMMIT_FIELDS
    for name in telemetry.SIDECAR_FILENAMES + (telemetry.MANIFEST_FILENAME,):
        info = (tmp_path / name).stat()
        assert stat.S_IMODE(info.st_mode) == 0o400
        assert info.st_nlink == 1
    repeated = telemetry.validate_schema13_telemetry_manifest(
        tmp_path,
        expected_xpid="schema13-unit",
        expected_terminal_policy_version=1,
        expected_terminal_real_step=32,
        expected_actor_state_sha256=_ZERO_HASH,
        expected_publication_history_sha256=_ZERO_HASH,
        expected_stage_total_steps=32,
        expected_actor_unroll_len=2,
        expected_terminal_ack_count=1,
        expected_manifest_sha256=hashlib.sha256(manifest_payload).hexdigest(),
        expected_manifest_size=len(manifest_payload),
    )
    assert repeated == evidence


def test_mixed_stepped_amp_skip_no_support_transactions_reconcile(
    tmp_path, monkeypatch
):
    writer = telemetry.Schema13TelemetryWriter(
        tmp_path,
        xpid="schema13-mixed-statuses",
        actor_unroll_len=2,
        stage_total_steps=96,
        q_initial_lr=1.0e-3,
        schedule_total_steps=128,
        amp_initial_scale=256.0,
    )
    statuses = ("stepped", "amp_skip", "no_support")
    counts = ((0, 1), (1, 1), (1, 1))
    epochs = ((0, 32), (32, 32), (32, 32))
    scheduler_steps = ((1, 2), (2, 2), (2, 2))
    lrs = ((1.0e-3, 7.5e-4), (7.5e-4, 7.5e-4), (7.5e-4, 7.5e-4))
    amp_scales = ((256.0, 256.0), (256.0, 128.0), (128.0, 128.0))
    adam_steps = ((0, 1), (1, 1), (1, 1))
    history = [{"state_sha256": "f" * 64}]
    state_hashes = []

    for index, status in enumerate(statuses, start=1):
        train = status != "no_support"
        real_before = (index - 1) * 32
        real_after = index * 32
        td_rows = telemetry.build_td_cell_rows(
            transaction_id=index,
            source_policy_version=index - 1,
            published_policy_version=index,
            real_step_after=real_after,
            **_td_inputs(train=train),
        )
        before_count, after_count = counts[index - 1]
        epoch_before, epoch_after = epochs[index - 1]
        scheduler_before, scheduler_after = scheduler_steps[index - 1]
        lr_before, lr_after = lrs[index - 1]
        amp_before, amp_after = amp_scales[index - 1]
        adam_before, adam_after = adam_steps[index - 1]
        state_hash = hashlib.sha256(f"state-{index}".encode()).hexdigest()
        state_hashes.append(state_hash)
        history.append({"state_sha256": state_hash})
        history_hash = hashlib.sha256(json.dumps(
            history,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")).hexdigest()
        replay_row = telemetry.build_replay_row(
            transaction_id=index,
            source_policy_version=index - 1,
            published_policy_version=index,
            replay_t=3,
            optimized_t=2,
            replay_b=16,
            actor_ids=tuple(range(16)),
            real_step_before=real_before,
            real_step_delta=32,
            real_step_after=real_after,
            valid_count=32,
            train_count=32 if train else 0,
            holdout_count=0 if train else 32,
            train_continue_count=16 if train else 0,
            train_stop_count=16 if train else 0,
            holdout_continue_count=0 if train else 16,
            holdout_stop_count=0 if train else 16,
            q_status=status,
            voc_update_count_before=before_count,
            voc_update_count_after=after_count,
            ema_update_count_before=before_count,
            ema_update_count_after=after_count,
            projection_count_before=before_count,
            projection_count_after=after_count,
            q_scheduler_last_epoch_before=epoch_before,
            q_scheduler_last_epoch_after=epoch_after,
            q_scheduler_step_count_before=scheduler_before,
            q_scheduler_step_count_after=scheduler_after,
            q_lr_before=lr_before,
            q_lr_used=lr_before,
            q_lr_after=lr_after,
            publication_count_after=index,
            ack_count=1,
            terminal=index == len(statuses),
            actor_state_sha256=state_hash,
            publication_history_sha256=history_hash,
        )
        diagnostics = None
        if status == "stepped":
            weight = torch.zeros((2, 3), dtype=torch.float32)
            bias = torch.zeros((2,), dtype=torch.float32)
            diagnostics = telemetry.build_stepped_q_diagnostics(
                clip_scale=1.0,
                raw_preclip=(weight, bias),
                raw_postclip=(weight.clone(), bias.clone()),
                md_postclip=(weight.clone(), bias.clone()),
                adam_m_before=(weight.clone(), bias.clone()),
                adam_v_before=(weight.clone(), bias.clone()),
                adam_m_after=(weight.clone(), bias.clone()),
                adam_v_after=(weight.clone(), bias.clone()),
                coordinate_delta=(weight.clone(), bias.clone()),
                mapped_delta=(weight.clone(), bias.clone()),
                q_lr_used=lr_before,
                adam_step_after=adam_after,
            )
        q_row = telemetry.build_q_transaction_row(
            transaction_id=index,
            source_policy_version=index - 1,
            published_policy_version=index,
            real_step_after=real_after,
            q_status=status,
            q_attempted=train,
            q_optimizer_committed=status == "stepped",
            q_loss_sum=0.0,
            clip_limit=24.0,
            amp_scale_before=amp_before,
            amp_scale_after=amp_after,
            nonfinite_gradient_parameter_count=(
                1 if status == "amp_skip" else 0
            ),
            adam_step_before=adam_before,
            adam_step_after=adam_after,
            diagnostics=diagnostics,
        )
        writer.append_transaction(
            td_rows=td_rows,
            replay_row=replay_row,
            q_row=q_row,
            terminal=index == len(statuses),
            actor_state_sha256=state_hash,
            publication_history_sha256=history_hash,
        )

    header = b"_tick,value,actor/voc_q_loss\n"
    log_path = tmp_path / telemetry.LEGACY_LOG_FILENAME
    log_path.write_bytes(
        header + b"0,1,0.0\r\n1,2,0.0\r\n2,3,0.0\r\n"
    )
    monkeypatch.setattr(telemetry, "LEGACY_LOG_HEADER_SIZE", len(header))
    monkeypatch.setattr(
        telemetry, "LEGACY_LOG_HEADER_SHA256", hashlib.sha256(header).hexdigest()
    )
    monkeypatch.setattr(telemetry, "LEGACY_LOG_COLUMN_COUNT", 3)
    evidence = writer.seal(
        terminal_real_step=96,
        terminal_policy_version=3,
        terminal_publication_count=3,
        terminal_ack_count=1,
        legacy_actor_log_path=log_path,
    )

    assert evidence["transaction_count"] == 3
    assert evidence["terminal_policy_version"] == 3
    assert evidence["terminal_real_step"] == 96
    zero_weight = torch.zeros((2, 3), dtype=torch.float32)
    zero_bias = torch.zeros((2,), dtype=torch.float32)
    terminal_state = {
        "voc_update_count": 1,
        "ema_update_count": 1,
        "projection_count": 1,
        "adam_step_weight": 1,
        "adam_step_bias": 1,
        "q_scheduler_last_epoch": 32,
        "q_scheduler_step_count": 2,
        "q_optimizer_lr": 7.5e-4,
        "q_scheduler_last_lr": 7.5e-4,
        "amp_scale": 128.0,
        "amp_growth_tracker": 0,
        "amp_skip_count": 1,
        "amp_consecutive_skips": 1,
        "adam_m_after": (zero_weight, zero_bias),
        "adam_v_after": (zero_weight.clone(), zero_bias.clone()),
    }
    repeated = telemetry.validate_schema13_telemetry_manifest(
        tmp_path,
        expected_xpid="schema13-mixed-statuses",
        expected_terminal_policy_version=3,
        expected_terminal_real_step=96,
        expected_actor_state_sha256=state_hashes[-1],
        expected_publication_history_sha256=history_hash,
        expected_stage_total_steps=96,
        expected_actor_unroll_len=2,
        expected_terminal_ack_count=1,
        expected_q_initial_lr=1.0e-3,
        expected_schedule_total_steps=128,
        expected_amp_initial_scale=256.0,
        expected_publication_history=history,
        expected_terminal_state=terminal_state,
    )
    assert repeated == evidence


def test_validator_rejects_mode_tamper_extra_artifact_and_tick_drift(
    tmp_path, monkeypatch
):
    _build_terminal_set(tmp_path, monkeypatch)
    q_path = tmp_path / telemetry.Q_FILENAME
    os.chmod(q_path, 0o600)
    with pytest.raises(RuntimeError, match="mode"):
        telemetry.validate_schema13_telemetry_manifest(
            tmp_path, expected_actor_unroll_len=2
        )
    os.chmod(q_path, 0o400)
    extra = tmp_path / "voc_telemetry_orphan.csv"
    extra.write_bytes(b"orphan\n")
    with pytest.raises(ValueError, match="extra telemetry"):
        telemetry.validate_schema13_telemetry_manifest(
            tmp_path, expected_actor_unroll_len=2
        )
    extra.unlink()
    log_path = tmp_path / telemetry.LEGACY_LOG_FILENAME
    log_path.write_bytes(b"_tick,value,actor/voc_q_loss\n1,1,0.0\r\n")
    with pytest.raises(ValueError, match="tick sequence"):
        telemetry.validate_schema13_telemetry_manifest(
            tmp_path, expected_actor_unroll_len=2
        )


def test_legacy_rows_require_exact_columns_termination_and_q_loss(monkeypatch):
    header = b"_tick,value,actor/voc_q_loss\n"
    monkeypatch.setattr(telemetry, "LEGACY_LOG_HEADER_SIZE", len(header))
    monkeypatch.setattr(
        telemetry, "LEGACY_LOG_HEADER_SHA256", hashlib.sha256(header).hexdigest()
    )
    monkeypatch.setattr(telemetry, "LEGACY_LOG_COLUMN_COUNT", 3)
    replay = [{"train_count": 2}]
    q_rows = [{"q_loss_sum": 0.5}]
    record = telemetry._legacy_log_record_from_payload(
        header + b"0,1,0.25\r\n",
        1,
        expected_replay_rows=replay,
        expected_q_rows=q_rows,
    )
    assert record["column_count"] == 3
    for bad in (
        header + b"0,0.25\r\n",
        header + b"0,1,0.25,EXTRA\r\n",
        header + b"0,1,0.25",
        header + b"0,1,0.25\n",
    ):
        with pytest.raises(ValueError):
            telemetry._legacy_log_record_from_payload(bad, 1)
    with pytest.raises(ValueError, match="voc_q_loss disagrees"):
        telemetry._legacy_log_record_from_payload(
            header + b"0,1,0.5\r\n",
            1,
            expected_replay_rows=replay,
            expected_q_rows=q_rows,
        )


def test_validator_reconciles_history_terminal_checkpoint_and_lambda_lr(
    tmp_path, monkeypatch
):
    writer = telemetry.Schema13TelemetryWriter(
        tmp_path,
        xpid="schema13-reconcile",
        actor_unroll_len=2,
        stage_total_steps=32,
        q_initial_lr=1.0e-3,
        schedule_total_steps=32,
        amp_initial_scale=256.0,
    )
    td_rows, replay_row, q_row = _transaction_rows(status="stepped")
    history = [
        {"state_sha256": "f" * 64},
        {"state_sha256": _ZERO_HASH},
    ]
    history_digest = hashlib.sha256(json.dumps(
        history,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")).hexdigest()
    replay_row = dict(replay_row)
    replay_row["publication_history_sha256"] = history_digest
    writer.append_transaction(
        td_rows=td_rows,
        replay_row=replay_row,
        q_row=q_row,
        terminal=True,
        actor_state_sha256=_ZERO_HASH,
        publication_history_sha256=history_digest,
    )
    header = b"_tick,value,actor/voc_q_loss\n"
    log_path = tmp_path / telemetry.LEGACY_LOG_FILENAME
    log_path.write_bytes(header + b"0,1,0.0\r\n")
    monkeypatch.setattr(telemetry, "LEGACY_LOG_HEADER_SIZE", len(header))
    monkeypatch.setattr(
        telemetry, "LEGACY_LOG_HEADER_SHA256", hashlib.sha256(header).hexdigest()
    )
    monkeypatch.setattr(telemetry, "LEGACY_LOG_COLUMN_COUNT", 3)
    writer.seal(
        terminal_real_step=32,
        terminal_policy_version=1,
        terminal_publication_count=1,
        terminal_ack_count=1,
        legacy_actor_log_path=log_path,
    )
    zero_weight = torch.zeros((2, 3), dtype=torch.float32)
    zero_bias = torch.zeros((2,), dtype=torch.float32)
    terminal_state = {
        "voc_update_count": 1,
        "ema_update_count": 1,
        "projection_count": 1,
        "adam_step_weight": 1,
        "adam_step_bias": 1,
        "q_scheduler_last_epoch": 32,
        "q_scheduler_step_count": 2,
        "q_optimizer_lr": 0.0,
        "q_scheduler_last_lr": 0.0,
        "amp_scale": 256.0,
        "amp_growth_tracker": 1,
        "amp_skip_count": 0,
        "amp_consecutive_skips": 0,
        "adam_m_after": (zero_weight, zero_bias),
        "adam_v_after": (zero_weight.clone(), zero_bias.clone()),
    }
    evidence = telemetry.validate_schema13_telemetry_manifest(
        tmp_path,
        expected_actor_unroll_len=2,
        expected_q_initial_lr=1.0e-3,
        expected_schedule_total_steps=32,
        expected_amp_initial_scale=256.0,
        expected_publication_history=history,
        expected_terminal_state=terminal_state,
    )
    assert len(evidence) == 10
    scalar_mutations = {
        "voc_update_count": 0,
        "ema_update_count": 0,
        "projection_count": 0,
        "adam_step_weight": 0,
        "adam_step_bias": 0,
        "q_scheduler_last_epoch": 31,
        "q_scheduler_step_count": 1,
        "q_optimizer_lr": 1.0e-4,
        "q_scheduler_last_lr": 1.0e-4,
        "amp_scale": 128.0,
        "amp_growth_tracker": 0,
        "amp_skip_count": 1,
        "amp_consecutive_skips": 1,
    }
    for name, bad in scalar_mutations.items():
        forged = dict(terminal_state)
        forged[name] = bad
        with pytest.raises(ValueError):
            telemetry.validate_schema13_telemetry_manifest(
                tmp_path,
                expected_actor_unroll_len=2,
                expected_q_initial_lr=1.0e-3,
                expected_schedule_total_steps=32,
                expected_amp_initial_scale=256.0,
                expected_publication_history=history,
                expected_terminal_state=forged,
            )
    for name in ("adam_m_after", "adam_v_after"):
        forged = dict(terminal_state)
        forged_weight = zero_weight.clone()
        forged_weight[0, 0] = 1.0
        forged[name] = (forged_weight, zero_bias)
        with pytest.raises(ValueError, match="diagnostic"):
            telemetry.validate_schema13_telemetry_manifest(
                tmp_path,
                expected_actor_unroll_len=2,
                expected_q_initial_lr=1.0e-3,
                expected_schedule_total_steps=32,
                expected_amp_initial_scale=256.0,
                expected_publication_history=history,
                expected_terminal_state=forged,
            )
    bad_state_history = [history[0], {"state_sha256": "e" * 64}]
    with pytest.raises(ValueError, match="actor state"):
        telemetry.validate_schema13_telemetry_manifest(
            tmp_path,
            expected_actor_unroll_len=2,
            expected_publication_history=bad_state_history,
        )
    bad_prefix_history = [
        {"state_sha256": "e" * 64},
        history[1],
    ]
    with pytest.raises(ValueError, match="history digest"):
        telemetry.validate_schema13_telemetry_manifest(
            tmp_path,
            expected_actor_unroll_len=2,
            expected_publication_history=bad_prefix_history,
        )


def test_writer_rejects_wrong_stage_lambda_lr_before_manifest(tmp_path, monkeypatch):
    writer = telemetry.Schema13TelemetryWriter(
        tmp_path,
        xpid="schema13-bad-lr",
        actor_unroll_len=2,
        stage_total_steps=32,
        q_initial_lr=1.0e-3,
        schedule_total_steps=64,
        amp_initial_scale=256.0,
    )
    td_rows, replay_row, q_row = _transaction_rows(status="stepped")
    writer.append_transaction(
        td_rows=td_rows,
        replay_row=replay_row,
        q_row=q_row,
        terminal=True,
        actor_state_sha256=_ZERO_HASH,
        publication_history_sha256=_ZERO_HASH,
    )
    header = b"_tick,value,actor/voc_q_loss\n"
    log_path = tmp_path / telemetry.LEGACY_LOG_FILENAME
    log_path.write_bytes(header + b"0,1,0.0\r\n")
    monkeypatch.setattr(telemetry, "LEGACY_LOG_HEADER_SIZE", len(header))
    monkeypatch.setattr(
        telemetry, "LEGACY_LOG_HEADER_SHA256", hashlib.sha256(header).hexdigest()
    )
    monkeypatch.setattr(telemetry, "LEGACY_LOG_COLUMN_COUNT", 3)
    with pytest.raises(ValueError, match="LambdaLR"):
        writer.seal(
            terminal_real_step=32,
            terminal_policy_version=1,
            terminal_publication_count=1,
            terminal_ack_count=1,
            legacy_actor_log_path=log_path,
        )
    assert not (tmp_path / telemetry.MANIFEST_FILENAME).exists()


def test_validator_rejects_real_step_delta_above_optimized_support(
    tmp_path, monkeypatch
):
    writer = telemetry.Schema13TelemetryWriter(
        tmp_path, xpid="schema13-impossible-real-step", actor_unroll_len=2
    )
    td_rows, replay_row, q_row = _transaction_rows(status="no_support")
    td_rows = tuple({**row, "real_step_after": 33} for row in td_rows)
    replay_row = {
        **replay_row,
        "real_step_delta": 33,
        "real_step_after": 33,
    }
    q_row = {**q_row, "real_step_after": 33}
    writer.append_transaction(
        td_rows=td_rows,
        replay_row=replay_row,
        q_row=q_row,
        terminal=True,
        actor_state_sha256=_ZERO_HASH,
        publication_history_sha256=_ZERO_HASH,
    )
    header = b"_tick,value,actor/voc_q_loss\n"
    log_path = tmp_path / telemetry.LEGACY_LOG_FILENAME
    log_path.write_bytes(header + b"0,1,0.0\r\n")
    monkeypatch.setattr(telemetry, "LEGACY_LOG_HEADER_SIZE", len(header))
    monkeypatch.setattr(
        telemetry, "LEGACY_LOG_HEADER_SHA256", hashlib.sha256(header).hexdigest()
    )
    monkeypatch.setattr(telemetry, "LEGACY_LOG_COLUMN_COUNT", 3)

    with pytest.raises(ValueError, match="real-step delta exceeds"):
        writer.seal(
            terminal_real_step=33,
            terminal_policy_version=1,
            terminal_publication_count=1,
            terminal_ack_count=1,
            legacy_actor_log_path=log_path,
        )
    assert not (tmp_path / telemetry.MANIFEST_FILENAME).exists()


def test_seal_rejects_replaced_bound_legacy_log_inode(tmp_path, monkeypatch):
    writer = telemetry.Schema13TelemetryWriter(
        tmp_path, xpid="schema13-log-race", actor_unroll_len=2
    )
    td_rows, replay_row, q_row = _transaction_rows()
    writer.append_transaction(
        td_rows=td_rows,
        replay_row=replay_row,
        q_row=q_row,
        terminal=True,
        actor_state_sha256=_ZERO_HASH,
        publication_history_sha256=_ZERO_HASH,
    )
    header = b"_tick,value,actor/voc_q_loss\n"
    payload = header + b"0,1,0.0\r\n"
    log_path = tmp_path / telemetry.LEGACY_LOG_FILENAME
    log_path.write_bytes(payload)
    monkeypatch.setattr(telemetry, "LEGACY_LOG_HEADER_SIZE", len(header))
    monkeypatch.setattr(
        telemetry, "LEGACY_LOG_HEADER_SHA256", hashlib.sha256(header).hexdigest()
    )
    monkeypatch.setattr(telemetry, "LEGACY_LOG_COLUMN_COUNT", 3)
    log_fd = os.open(log_path, os.O_RDONLY)
    replacement = tmp_path / "replacement-log"
    replacement.write_bytes(payload)
    os.replace(replacement, log_path)
    try:
        with pytest.raises(RuntimeError, match="link count|pathname identity"):
            writer.seal(
                terminal_real_step=32,
                terminal_policy_version=1,
                terminal_publication_count=1,
                terminal_ack_count=1,
                legacy_actor_log_path=log_path,
                legacy_actor_log_fd=log_fd,
            )
    finally:
        os.close(log_fd)
    assert not (tmp_path / telemetry.MANIFEST_FILENAME).exists()


def test_seal_never_binds_replaced_sidecar_after_bound_read(tmp_path, monkeypatch):
    writer = telemetry.Schema13TelemetryWriter(
        tmp_path, xpid="schema13-sidecar-race", actor_unroll_len=2
    )
    td_rows, replay_row, q_row = _transaction_rows()
    writer.append_transaction(
        td_rows=td_rows,
        replay_row=replay_row,
        q_row=q_row,
        terminal=True,
        actor_state_sha256=_ZERO_HASH,
        publication_history_sha256=_ZERO_HASH,
    )
    header = b"_tick,value,actor/voc_q_loss\n"
    log_path = tmp_path / telemetry.LEGACY_LOG_FILENAME
    log_path.write_bytes(header + b"0,1,0.0\r\n")
    monkeypatch.setattr(telemetry, "LEGACY_LOG_HEADER_SIZE", len(header))
    monkeypatch.setattr(
        telemetry, "LEGACY_LOG_HEADER_SHA256", hashlib.sha256(header).hexdigest()
    )
    monkeypatch.setattr(telemetry, "LEGACY_LOG_COLUMN_COUNT", 3)
    original = telemetry._validated_payload_records
    replaced = False

    def replace_after_bound_read(payloads, **kwargs):
        nonlocal replaced
        result = original(payloads, **kwargs)
        if not replaced:
            replaced = True
            q_path = tmp_path / telemetry.Q_FILENAME
            replacement = tmp_path / "replacement-q"
            replacement.write_bytes(payloads[telemetry.Q_FILENAME])
            os.chmod(replacement, 0o400)
            os.replace(replacement, q_path)
        return result

    monkeypatch.setattr(
        telemetry, "_validated_payload_records", replace_after_bound_read
    )
    with pytest.raises(RuntimeError, match="expected bound inode"):
        writer.seal(
            terminal_real_step=32,
            terminal_policy_version=1,
            terminal_publication_count=1,
            terminal_ack_count=1,
            legacy_actor_log_path=log_path,
        )
    assert writer.poisoned is True


def test_fresh_exclusive_creation_and_post_ack_failure_poison_without_commit(
    tmp_path, monkeypatch
):
    (tmp_path / telemetry.TD_FILENAME).write_bytes(b"occupied")
    with pytest.raises(FileExistsError):
        telemetry.Schema13TelemetryWriter(
            tmp_path, xpid="collision", actor_unroll_len=2
        )

    second = tmp_path / "second"
    second.mkdir()
    writer = telemetry.Schema13TelemetryWriter(
        second, xpid="poison", actor_unroll_len=2
    )
    td_rows, replay_row, q_row = _transaction_rows()
    original = telemetry._write_once

    def fail_q(fd, payload, label):
        if label == f"{telemetry.Q_FILENAME} transaction block":
            raise OSError("injected q append failure")
        return original(fd, payload, label)

    monkeypatch.setattr(telemetry, "_write_once", fail_q)
    with pytest.raises(OSError, match="injected"):
        writer.append_transaction(
            td_rows=td_rows,
            replay_row=replay_row,
            q_row=q_row,
            terminal=True,
            actor_state_sha256=_ZERO_HASH,
            publication_history_sha256=_ZERO_HASH,
        )
    assert writer.poisoned is True
    assert writer.transaction_count == 0
    with pytest.raises(RuntimeError, match="poisoned"):
        writer.append_transaction(
            td_rows=td_rows,
            replay_row=replay_row,
            q_row=q_row,
            terminal=True,
            actor_state_sha256=_ZERO_HASH,
            publication_history_sha256=_ZERO_HASH,
        )
    assert (second / telemetry.COMMIT_FILENAME).read_bytes() == (
        telemetry.COMMIT_HEADER
    )


@pytest.mark.parametrize(
    "name", (telemetry.MANIFEST_FILENAME, *telemetry.SIDECAR_FILENAMES)
)
def test_stable_reader_rejects_fifo_without_blocking(tmp_path, monkeypatch, name):
    path = tmp_path / name
    os.mkfifo(path, mode=0o400)
    real_open = telemetry.os.open

    def require_nonblocking(target, flags, *args, **kwargs):
        assert flags & os.O_NONBLOCK
        return real_open(target, flags, *args, **kwargs)

    monkeypatch.setattr(telemetry.os, "open", require_nonblocking)
    with pytest.raises(RuntimeError, match="not a regular file"):
        telemetry._stable_read(path, label=name, mode=0o400)
