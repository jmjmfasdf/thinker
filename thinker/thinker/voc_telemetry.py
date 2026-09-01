"""Schema-13 VoC telemetry codecs, durable writer, and bound validator.

This module is deliberately a leaf.  Importing it performs no file, clock,
random-number, device, model, optimizer, logger, Ray, or W&B operation.  The
schema-13 learner imports it only after the strict V20 surface has been
accepted and passes detached observations rather than live training objects.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import re
import stat
from collections.abc import Mapping, Sequence


VOC_TELEMETRY_SCHEMA_VERSION = 1
VOC_GATE_SCHEMA_VERSION = 13

TD_FILENAME = "voc_td_cells.csv"
REPLAY_FILENAME = "voc_replay_events.csv"
Q_FILENAME = "voc_q_transactions.csv"
COMMIT_FILENAME = "voc_telemetry_commits.csv"
MANIFEST_FILENAME = "voc_telemetry_manifest.json"
SIDECAR_FILENAMES = (
    TD_FILENAME,
    REPLAY_FILENAME,
    Q_FILENAME,
    COMMIT_FILENAME,
)

TD_FIELDS = (
    "telemetry_schema_version",
    "gate_schema",
    "transaction_id",
    "source_policy_version",
    "published_policy_version",
    "real_step_after",
    "q_source",
    "split",
    "selected_action",
    "depth_bin",
    "td_sign",
    "abs_td_bin",
    "count",
    "sum_target",
    "sum_target_sq",
    "sum_selected_q",
    "sum_selected_q_sq",
    "sum_target_selected_q",
    "sum_td",
    "sum_abs_td",
    "sum_td_sq",
    "max_abs_td",
)

REPLAY_FIELDS = (
    "telemetry_schema_version",
    "gate_schema",
    "transaction_id",
    "source_policy_version",
    "published_policy_version",
    "replay_t",
    "optimized_t",
    "replay_b",
    "actor_ids",
    "actor_ids_sha256",
    "real_step_before",
    "real_step_delta",
    "real_step_after",
    "valid_count",
    "train_count",
    "holdout_count",
    "train_continue_count",
    "train_stop_count",
    "holdout_continue_count",
    "holdout_stop_count",
    "q_status",
    "voc_update_count_before",
    "voc_update_count_after",
    "ema_update_count_before",
    "ema_update_count_after",
    "projection_count_before",
    "projection_count_after",
    "q_scheduler_last_epoch_before",
    "q_scheduler_last_epoch_after",
    "q_scheduler_step_count_before",
    "q_scheduler_step_count_after",
    "q_lr_before",
    "q_lr_used",
    "q_lr_after",
    "publication_count_after",
    "ack_count",
    "terminal",
    "actor_state_sha256",
    "publication_history_sha256",
)

Q_FIELDS = (
    "telemetry_schema_version",
    "gate_schema",
    "transaction_id",
    "source_policy_version",
    "published_policy_version",
    "real_step_after",
    "q_status",
    "q_attempted",
    "q_optimizer_committed",
    "q_loss_sum",
    "clip_limit",
    "clip_scale",
    "raw_preclip_total_l2",
    "raw_postclip_total_l2",
    "amp_scale_before",
    "amp_scale_after",
    "nonfinite_gradient_parameter_count",
    "adam_step_before",
    "adam_step_after",
    "raw_preclip_weight_continue_l2",
    "raw_preclip_weight_stop_l2",
    "raw_postclip_weight_continue_l2",
    "raw_postclip_weight_stop_l2",
    "md_postclip_weight_common_l2",
    "md_postclip_weight_difference_l2",
    "adam_m_before_weight_common_l2",
    "adam_m_before_weight_difference_l2",
    "adam_v_before_weight_common_mean",
    "adam_v_before_weight_difference_mean",
    "adam_m_after_weight_common_l2",
    "adam_m_after_weight_difference_l2",
    "adam_v_after_weight_common_mean",
    "adam_v_after_weight_difference_mean",
    "normalized_update_weight_common_l2",
    "normalized_update_weight_difference_l2",
    "coordinate_delta_weight_common_l2",
    "coordinate_delta_weight_difference_l2",
    "mapped_delta_weight_continue_l2",
    "mapped_delta_weight_stop_l2",
    "raw_preclip_bias_continue_l2",
    "raw_preclip_bias_stop_l2",
    "raw_postclip_bias_continue_l2",
    "raw_postclip_bias_stop_l2",
    "md_postclip_bias_common_l2",
    "md_postclip_bias_difference_l2",
    "adam_m_before_bias_common_l2",
    "adam_m_before_bias_difference_l2",
    "adam_v_before_bias_common_mean",
    "adam_v_before_bias_difference_mean",
    "adam_m_after_bias_common_l2",
    "adam_m_after_bias_difference_l2",
    "adam_v_after_bias_common_mean",
    "adam_v_after_bias_difference_mean",
    "normalized_update_bias_common_l2",
    "normalized_update_bias_difference_l2",
    "coordinate_delta_bias_common_l2",
    "coordinate_delta_bias_difference_l2",
    "mapped_delta_bias_continue_l2",
    "mapped_delta_bias_stop_l2",
)

COMMIT_FIELDS = (
    "telemetry_schema_version",
    "gate_schema",
    "transaction_id",
    "source_policy_version",
    "published_policy_version",
    "terminal",
    "td_first_data_row",
    "td_data_row_count",
    "td_block_byte_count",
    "td_block_sha256",
    "replay_first_data_row",
    "replay_data_row_count",
    "replay_block_byte_count",
    "replay_block_sha256",
    "q_first_data_row",
    "q_data_row_count",
    "q_block_byte_count",
    "q_block_sha256",
    "publication_count",
    "ack_count",
    "actor_state_sha256",
    "publication_history_sha256",
)


def _header(fields):
    return (",".join(fields) + "\n").encode("ascii")


TD_HEADER = _header(TD_FIELDS)
REPLAY_HEADER = _header(REPLAY_FIELDS)
Q_HEADER = _header(Q_FIELDS)
COMMIT_HEADER = _header(COMMIT_FIELDS)
HEADERS = {
    TD_FILENAME: TD_HEADER,
    REPLAY_FILENAME: REPLAY_HEADER,
    Q_FILENAME: Q_HEADER,
    COMMIT_FILENAME: COMMIT_HEADER,
}
FIELDS_BY_FILENAME = {
    TD_FILENAME: TD_FIELDS,
    REPLAY_FILENAME: REPLAY_FIELDS,
    Q_FILENAME: Q_FIELDS,
    COMMIT_FILENAME: COMMIT_FIELDS,
}
HEADER_SHA256 = {
    name: hashlib.sha256(header).hexdigest()
    for name, header in HEADERS.items()
}

_EXPECTED_HEADER_CONTRACT = {
    TD_FILENAME: (
        300,
        "37c82eea9a7bf7cbe05ee74ffb2b37b6190e4b715b05afbea5b5a06c406473fa",
        22,
    ),
    REPLAY_FILENAME: (
        713,
        "eed6226a8a591289125c7f5389b7d6705332b11e32746f103093b5dcd71592e2",
        39,
    ),
    Q_FILENAME: (
        1603,
        "e1574cf8c81306818abc2369b5270f98c74f3e3190cb8cb3ceddb000ad4096b3",
        59,
    ),
    COMMIT_FILENAME: (
        410,
        "9105b143dfd260a4c491a2821757c14ded2a32c74207e6f4e7140b717ae62929",
        22,
    ),
}
for _name, (_size, _digest, _columns) in _EXPECTED_HEADER_CONTRACT.items():
    if (
        len(HEADERS[_name]) != _size
        or HEADER_SHA256[_name] != _digest
        or len(FIELDS_BY_FILENAME[_name]) != _columns
    ):
        raise AssertionError("schema-13 telemetry header constant drift")

LEGACY_LOG_FILENAME = "logs.csv"
LEGACY_LOG_HEADER_SIZE = 43550
LEGACY_LOG_COLUMN_COUNT = 922
LEGACY_LOG_HEADER_SHA256 = (
    "82488231a631ca3571379e973122dd107007d14f4756fd839a811851dc6accbc"
)

Q_SOURCES = ("online", "ema")
SPLITS = ("train", "holdout")
SELECTED_ACTIONS = ("continue", "stop")
DEPTH_BINS = ("0", "1", "2_3", "4_7", "8_15", "16_plus")
TD_SIGNS = ("negative", "zero", "positive")
ABS_TD_BINS = ("0_0p5", "0p5_1", "1_2", "2_4", "4_inf")
TD_CATEGORY_ORDER = tuple(
    itertools.product(
        Q_SOURCES,
        SPLITS,
        SELECTED_ACTIONS,
        DEPTH_BINS,
        TD_SIGNS,
        ABS_TD_BINS,
    )
)
if len(TD_CATEGORY_ORDER) != 720:
    raise AssertionError("schema-13 TD cube cardinality drift")

Q_STATUSES = ("stepped", "no_support", "amp_skip")
Q_DIAGNOSTIC_FIELDS = (
    "clip_scale",
    "raw_preclip_total_l2",
    "raw_postclip_total_l2",
) + Q_FIELDS[19:]
Q_BASE_FLOAT_FIELDS = (
    "q_loss_sum",
    "clip_limit",
    "amp_scale_before",
    "amp_scale_after",
)
TD_FLOAT_FIELDS = TD_FIELDS[13:]
REPLAY_FLOAT_FIELDS = ("q_lr_before", "q_lr_used", "q_lr_after")
Q_FLOAT_FIELDS = Q_BASE_FLOAT_FIELDS + Q_DIAGNOSTIC_FIELDS

_LOWER_HEX_RE = re.compile(r"[0-9a-f]{64}\Z")
_UINT_RE = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_ACTOR_IDS_RE = re.compile(r"(?:0|[1-9][0-9]*)(?:;(?:0|[1-9][0-9]*))*\Z")
_FORWARD_ABS = 2.0 ** -20
_FORWARD_ADD = 2.0 ** -30

_MANIFEST_FIELDS = {
    "telemetry_schema_version",
    "gate_schema",
    "status",
    "xpid",
    "fresh",
    "transaction_count",
    "terminal_policy_version",
    "terminal_real_step",
    "terminal_publication_count",
    "terminal_ack_count",
    "actor_state_sha256",
    "publication_history_sha256",
    "legacy_actor_log",
    "artifacts",
    "last_commit",
}
_FILE_RECORD_FIELDS = {
    "name",
    "sha256",
    "size",
    "header_sha256",
    "header_size",
    "column_count",
    "data_row_count",
}
_LAST_COMMIT_FIELDS = {
    "data_row",
    "transaction_id",
    "sha256",
    "actor_state_sha256",
    "publication_history_sha256",
}
_EVIDENCE_FIELDS = {
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


def _require_exact_int(name, value, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise TypeError(f"{name} must be a built-in integer >= {minimum}")
    return value


def _require_exact_string(name, value, *, nonempty=True):
    if type(value) is not str or (nonempty and not value):
        raise TypeError(f"{name} must be a nonempty built-in string")
    return value


def _require_hash(name, value):
    if type(value) is not str or _LOWER_HEX_RE.fullmatch(value) is None:
        raise TypeError(f"{name} must be lowercase SHA-256 hex")
    return value


def _finite_float(name, value):
    if type(value) not in (int, float) or type(value) is bool:
        raise TypeError(f"{name} must be a built-in finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return 0.0 if result == 0.0 else result


def canonical_float(value):
    """Return the frozen CPython binary64 spelling with positive zero."""

    result = _finite_float("telemetry float", value)
    return float.hex(result)


def parse_canonical_float(token, *, allow_na=False, name="telemetry float"):
    if allow_na and token == "NA":
        return None
    if type(token) is not str:
        raise TypeError(f"{name} token must be a built-in string")
    try:
        value = float.fromhex(token)
    except ValueError as error:
        raise ValueError(f"{name} is not hexadecimal binary64") from error
    if not math.isfinite(value) or (value == 0.0 and token != "0x0.0p+0"):
        raise ValueError(f"{name} is not canonical finite binary64")
    if float.hex(value) != token:
        raise ValueError(f"{name} has a noncanonical binary64 spelling")
    return value


def canonical_json_bytes(value, trailing_lf=True):
    if type(trailing_lf) is not bool:
        raise TypeError("trailing_lf must be a built-in boolean")
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("telemetry JSON value is not canonicalizable") from error
    return payload + (b"\n" if trailing_lf else b"")


def _encode_uint(name, value):
    return str(_require_exact_int(name, value))


def _encode_bool(name, value):
    if type(value) is bool:
        return "1" if value else "0"
    if type(value) is int and value in (0, 1):
        return str(value)
    raise TypeError(f"{name} must be a built-in boolean or integer bit")


def _encode_hash(name, value):
    return _require_hash(name, value)


def _encode_token(name, value):
    value = _require_exact_string(name, value)
    if any(character in value for character in (",", "\r", "\n", '"', " ")):
        raise ValueError(f"{name} is not an unquoted CSV token")
    if not value.isascii():
        raise ValueError(f"{name} must be ASCII")
    return value


def _field_kind(fields, name):
    if name in TD_FLOAT_FIELDS or name in REPLAY_FLOAT_FIELDS or name in Q_FLOAT_FIELDS:
        return "float"
    if name.endswith("sha256"):
        return "hash"
    if name in ("terminal", "q_attempted", "q_optimizer_committed"):
        return "bool"
    if name in (
        "q_source",
        "split",
        "selected_action",
        "depth_bin",
        "td_sign",
        "abs_td_bin",
        "q_status",
        "actor_ids",
    ):
        return "token"
    return "uint"


def encode_csv_row(fields, row):
    """Encode one exact unquoted LF-terminated schema-13 data row."""

    if type(fields) is not tuple or not all(type(name) is str for name in fields):
        raise TypeError("CSV fields must be an exact tuple of strings")
    if not isinstance(row, Mapping) or set(row) != set(fields):
        raise ValueError("telemetry CSV row has the wrong keyset")
    tokens = []
    for name in fields:
        value = row[name]
        kind = _field_kind(fields, name)
        if kind == "float":
            if value == "NA":
                if fields != Q_FIELDS or name not in Q_DIAGNOSTIC_FIELDS:
                    raise ValueError(f"NA is unavailable for {name}")
                token = "NA"
            else:
                token = canonical_float(value)
        elif kind == "hash":
            token = _encode_hash(name, value)
        elif kind == "bool":
            token = _encode_bool(name, value)
        elif kind == "token":
            token = _encode_token(name, value)
        else:
            token = _encode_uint(name, value)
        tokens.append(token)
    return (",".join(tokens) + "\n").encode("ascii")


def _to_fp32_cpu(value, name):
    # Torch is intentionally imported only at a schema-13 runtime call.
    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    tensor = value.detach().to(dtype=torch.float32).contiguous().cpu()
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} must be finite")
    return tensor


def _to_bool_cpu(value, name):
    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.dtype is not torch.bool:
        raise TypeError(f"{name} must have exact bool dtype")
    return value.detach().to(dtype=torch.bool).contiguous().cpu()


def _to_int64_cpu(value, name):
    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise TypeError(f"{name} must have an exact non-Boolean integer dtype")
    return value.detach().to(dtype=torch.int64).contiguous().cpu()


def _depth_label(value):
    if value < 0:
        raise ValueError("decision depth must be nonnegative")
    if value == 0:
        return "0"
    if value == 1:
        return "1"
    if value <= 3:
        return "2_3"
    if value <= 7:
        return "4_7"
    if value <= 15:
        return "8_15"
    return "16_plus"


def _td_sign_label(value):
    if value < 0.0:
        return "negative"
    if value > 0.0:
        return "positive"
    return "zero"


def _abs_td_label(value):
    absolute = abs(value)
    if absolute < 0.5:
        return "0_0p5"
    if absolute < 1.0:
        return "0p5_1"
    if absolute < 2.0:
        return "1_2"
    if absolute < 4.0:
        return "2_4"
    return "4_inf"


def _fsum(values):
    result = math.fsum(values)
    return 0.0 if result == 0.0 else result


def l2_norm(value):
    tensor = _to_fp32_cpu(value, "L2 source")
    values = (float(item) for item in tensor.reshape(-1).tolist())
    result = math.sqrt(math.fsum(item * item for item in values))
    return 0.0 if result == 0.0 else result


def row_l2_norms(value):
    tensor = _to_fp32_cpu(value, "row L2 source")
    if tensor.ndim < 1 or tensor.shape[0] != 2:
        raise ValueError("row L2 source must have exactly two leading rows")
    return tuple(
        math.sqrt(math.fsum(float(item) ** 2 for item in row.reshape(-1).tolist()))
        for row in tensor
    )


def row_means(value):
    tensor = _to_fp32_cpu(value, "row mean source")
    if tensor.ndim < 1 or tensor.shape[0] != 2:
        raise ValueError("row mean source must have exactly two leading rows")
    means = []
    for row in tensor:
        values = [float(item) for item in row.reshape(-1).tolist()]
        if not values:
            raise ValueError("row mean source rows must be nonempty")
        means.append(_fsum(values) / len(values))
    return tuple(means)


def build_stepped_q_diagnostics(
    *,
    clip_scale,
    raw_preclip,
    raw_postclip,
    md_postclip,
    adam_m_before,
    adam_v_before,
    adam_m_after,
    adam_v_after,
    coordinate_delta,
    mapped_delta,
    q_lr_used,
    adam_step_after,
):
    """Reduce detached weight/bias optimizer snapshots after publication/ack."""

    import torch

    groups = {
        "raw_preclip": raw_preclip,
        "raw_postclip": raw_postclip,
        "md_postclip": md_postclip,
        "adam_m_before": adam_m_before,
        "adam_v_before": adam_v_before,
        "adam_m_after": adam_m_after,
        "adam_v_after": adam_v_after,
        "coordinate_delta": coordinate_delta,
        "mapped_delta": mapped_delta,
    }
    normalized = {}
    for name, pair in groups.items():
        if type(pair) not in (tuple, list) or len(pair) != 2:
            raise ValueError(f"{name} must contain detached weight then bias")
        normalized[name] = tuple(
            _to_fp32_cpu(value, f"{name} {label}")
            for value, label in zip(pair, ("weight", "bias"))
        )
    for index, label in enumerate(("weight", "bias")):
        reference = normalized["raw_preclip"][index]
        expected_shape = (2,) + tuple(reference.shape[1:])
        if tuple(reference.shape) != expected_shape:
            raise ValueError(f"{label} telemetry snapshot must have two rows")
        for name in groups:
            if tuple(normalized[name][index].shape) != tuple(reference.shape):
                raise ValueError(f"{name} {label} shape disagrees")
    lr = _finite_float("q_lr_used", q_lr_used)
    if lr <= 0.0:
        raise ValueError("q_lr_used must be positive for a stepped Q update")
    step = _require_exact_int("adam_step_after", adam_step_after, minimum=1)
    scale = _finite_float("clip_scale", clip_scale)
    if not 0.0 < scale <= 1.0:
        raise ValueError("clip_scale must lie in (0,1]")

    # Independently validate the pinned Adam algebra against the actual
    # zero-base functional scratch delta.  This is observational only.
    beta1_correction = 1.0 - 0.9 ** step
    beta2_correction = 1.0 - 0.999 ** step
    for index, label in enumerate(("weight", "bias")):
        md_gradient = normalized["md_postclip"][index]
        m_before_fp32 = normalized["adam_m_before"][index]
        v_before_fp32 = normalized["adam_v_before"][index]
        m_after_fp32 = normalized["adam_m_after"][index]
        v_after_fp32 = normalized["adam_v_after"][index]
        if torch.any(v_before_fp32 < 0).item() or torch.any(v_after_fp32 < 0).item():
            raise ValueError(f"Adam variance state for {label} must be nonnegative")
        expected_m_after = torch.lerp(m_before_fp32, md_gradient, 0.1)
        expected_v_after = v_before_fp32.mul(0.999).addcmul(
            md_gradient, md_gradient, value=0.001
        )
        for actual, expected in zip(
            m_after_fp32.reshape(-1).tolist(),
            expected_m_after.reshape(-1).tolist(),
        ):
            if not _close_enough(float(actual), float(expected)):
                raise ValueError(f"Adam first-moment transition disagrees for {label}")
        for actual, expected in zip(
            v_after_fp32.reshape(-1).tolist(),
            expected_v_after.reshape(-1).tolist(),
        ):
            if not _close_enough(float(actual), float(expected)):
                raise ValueError(f"Adam second-moment transition disagrees for {label}")
        m_after = normalized["adam_m_after"][index].to(torch.float64)
        v_after = normalized["adam_v_after"][index].to(torch.float64)
        actual_delta = normalized["coordinate_delta"][index].to(torch.float64)
        if torch.any(v_after < 0).item():
            raise ValueError(f"Adam v_after {label} must be nonnegative")
        derived_u = (m_after / beta1_correction) / (
            torch.sqrt(v_after / beta2_correction) + 1e-8
        )
        derived_delta = -lr * derived_u
        for actual, expected in zip(
            actual_delta.reshape(-1).tolist(), derived_delta.reshape(-1).tolist()
        ):
            if not _close_enough(float(actual), float(expected)):
                raise ValueError(
                    f"functional Adam coordinate delta disagrees for {label}"
                )
        raw_c = normalized["raw_postclip"][index][0]
        raw_s = normalized["raw_postclip"][index][1]
        bits = torch.tensor([0x3F3504F3], dtype=torch.int32).view(torch.float32)[0]
        expected_md = torch.stack(
            (bits * (raw_c + raw_s), bits * (raw_c - raw_s)), dim=0
        )
        if not torch.equal(expected_md, normalized["md_postclip"][index]):
            raise ValueError(f"m/d postclip transform disagrees for {label}")
        delta_m = normalized["coordinate_delta"][index][0]
        delta_d = normalized["coordinate_delta"][index][1]
        expected_mapped = torch.stack(
            (bits * (delta_m + delta_d), bits * (delta_m - delta_d)), dim=0
        )
        if not torch.equal(expected_mapped, normalized["mapped_delta"][index]):
            raise ValueError(f"inverse-mapped delta disagrees for {label}")

    def total_l2(pair):
        # The frozen reduction has one weight-then-bias scalar stream.  Do not
        # round parameter-local square roots and then re-square them.
        scalars = (
            float(item)
            for value in pair
            for item in value.reshape(-1).tolist()
        )
        result = math.sqrt(math.fsum(item * item for item in scalars))
        return 0.0 if result == 0.0 else result

    result = {
        "clip_scale": scale,
        "raw_preclip_total_l2": total_l2(normalized["raw_preclip"]),
        "raw_postclip_total_l2": total_l2(normalized["raw_postclip"]),
    }
    for index, parameter in enumerate(("weight", "bias")):
        for prefix in ("raw_preclip", "raw_postclip"):
            first, second = row_l2_norms(normalized[prefix][index])
            result[f"{prefix}_{parameter}_continue_l2"] = first
            result[f"{prefix}_{parameter}_stop_l2"] = second
        first, second = row_l2_norms(normalized["md_postclip"][index])
        result[f"md_postclip_{parameter}_common_l2"] = first
        result[f"md_postclip_{parameter}_difference_l2"] = second
        for moment in ("adam_m_before", "adam_m_after"):
            first, second = row_l2_norms(normalized[moment][index])
            when = "before" if moment.endswith("before") else "after"
            result[f"adam_m_{when}_{parameter}_common_l2"] = first
            result[f"adam_m_{when}_{parameter}_difference_l2"] = second
        for moment in ("adam_v_before", "adam_v_after"):
            first, second = row_means(normalized[moment][index])
            when = "before" if moment.endswith("before") else "after"
            result[f"adam_v_{when}_{parameter}_common_mean"] = first
            result[f"adam_v_{when}_{parameter}_difference_mean"] = second
        coordinate = normalized["coordinate_delta"][index]
        normalized_update = coordinate / (-lr)
        first, second = row_l2_norms(normalized_update)
        result[f"normalized_update_{parameter}_common_l2"] = first
        result[f"normalized_update_{parameter}_difference_l2"] = second
        first, second = row_l2_norms(coordinate)
        result[f"coordinate_delta_{parameter}_common_l2"] = first
        result[f"coordinate_delta_{parameter}_difference_l2"] = second
        first, second = row_l2_norms(normalized["mapped_delta"][index])
        result[f"mapped_delta_{parameter}_continue_l2"] = first
        result[f"mapped_delta_{parameter}_stop_l2"] = second
    if set(result) != set(Q_DIAGNOSTIC_FIELDS):
        raise AssertionError("stepped Q diagnostic keyset drift")
    return result


def build_td_cell_rows(
    *,
    transaction_id,
    source_policy_version,
    published_policy_version,
    real_step_after,
    target,
    online_q_values,
    ema_q_values,
    valid_mask,
    train_mask,
    holdout_mask,
    gate_action,
    control_action,
    search_steps,
):
    """Reduce detached FP32 event tensors to the frozen dense 720-row cube."""

    import torch

    identity = {
        "telemetry_schema_version": VOC_TELEMETRY_SCHEMA_VERSION,
        "gate_schema": VOC_GATE_SCHEMA_VERSION,
        "transaction_id": _require_exact_int("transaction_id", transaction_id, minimum=1),
        "source_policy_version": _require_exact_int(
            "source_policy_version", source_policy_version
        ),
        "published_policy_version": _require_exact_int(
            "published_policy_version", published_policy_version, minimum=1
        ),
        "real_step_after": _require_exact_int("real_step_after", real_step_after),
    }
    if identity["source_policy_version"] + 1 != identity["published_policy_version"]:
        raise ValueError("TD source/published policy versions are not contiguous")
    if identity["transaction_id"] != identity["published_policy_version"]:
        raise ValueError("TD transaction and published policy versions disagree")

    target_cpu = _to_fp32_cpu(target, "TD target")
    online_cpu = _to_fp32_cpu(online_q_values, "online Q")
    ema_cpu = _to_fp32_cpu(ema_q_values, "EMA Q")
    valid_cpu = _to_bool_cpu(valid_mask, "valid mask")
    train_cpu = _to_bool_cpu(train_mask, "train mask")
    holdout_cpu = _to_bool_cpu(holdout_mask, "holdout mask")
    gate_cpu = _to_int64_cpu(gate_action, "gate action")
    control_cpu = _to_int64_cpu(control_action, "control action")
    steps_cpu = _to_int64_cpu(search_steps, "search steps")
    shape = tuple(target_cpu.shape)
    if len(shape) != 2:
        raise ValueError("TD event tensors must be time-by-batch matrices")
    for name, tensor in (
        ("valid mask", valid_cpu),
        ("train mask", train_cpu),
        ("holdout mask", holdout_cpu),
        ("gate action", gate_cpu),
        ("control action", control_cpu),
        ("search steps", steps_cpu),
    ):
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must match TD target shape")
    if tuple(online_cpu.shape) != shape + (2,) or tuple(ema_cpu.shape) != shape + (2,):
        raise ValueError("online and EMA Q must have shape [T,B,2]")
    if torch.any(train_cpu & holdout_cpu).item():
        raise ValueError("train and holdout masks must be disjoint")
    supported = train_cpu | holdout_cpu
    if not torch.equal(supported, valid_cpu):
        raise ValueError("train and holdout masks must exactly partition valid mask")
    if torch.any((gate_cpu < 0) | (gate_cpu > 1)).item():
        raise ValueError("gate actions must be CONTINUE=0 or STOP=1")
    if torch.any(supported & ((control_cpu < 0) | (control_cpu > 2))).item():
        raise ValueError("supported control actions must be PROCEED/RESET/STOP")
    expected_gate = (control_cpu == 2).to(torch.int64)
    if torch.any(supported & (gate_cpu != expected_gate)).item():
        raise ValueError(
            "supported gate actions must map PROCEED/RESET to CONTINUE and STOP to STOP"
        )
    decision_depth = steps_cpu - (supported & (control_cpu != 2)).to(torch.int64)
    if torch.any(supported & (decision_depth < 0)).item():
        raise ValueError("supported decision depth is negative")

    cells = {category: [] for category in TD_CATEGORY_ORDER}
    target_values = target_cpu.reshape(-1).tolist()
    gate_values = gate_cpu.reshape(-1).tolist()
    depth_values = decision_depth.reshape(-1).tolist()
    train_values = train_cpu.reshape(-1).tolist()
    holdout_values = holdout_cpu.reshape(-1).tolist()
    online_selected = torch.gather(online_cpu, -1, gate_cpu.unsqueeze(-1)).squeeze(-1)
    ema_selected = torch.gather(ema_cpu, -1, gate_cpu.unsqueeze(-1)).squeeze(-1)
    for q_source, selected_tensor in (
        ("online", online_selected),
        ("ema", ema_selected),
    ):
        selected_values = selected_tensor.reshape(-1).tolist()
        for index, (target_value, selected_value) in enumerate(
            zip(target_values, selected_values)
        ):
            split = "train" if train_values[index] else (
                "holdout" if holdout_values[index] else None
            )
            if split is None:
                continue
            target_float = float(target_value)
            selected_float = float(selected_value)
            # Preserve the exact FP32 subtraction used by the source path.
            td_float = float(
                torch.tensor(target_float, dtype=torch.float32)
                - torch.tensor(selected_float, dtype=torch.float32)
            )
            category = (
                q_source,
                split,
                "continue" if int(gate_values[index]) == 0 else "stop",
                _depth_label(int(depth_values[index])),
                _td_sign_label(td_float),
                _abs_td_label(td_float),
            )
            cells[category].append((target_float, selected_float, td_float))

    rows = []
    for category in TD_CATEGORY_ORDER:
        values = cells[category]
        targets = [value[0] for value in values]
        selected = [value[1] for value in values]
        residuals = [value[2] for value in values]
        absolute = [abs(value) for value in residuals]
        row = {
            **identity,
            "q_source": category[0],
            "split": category[1],
            "selected_action": category[2],
            "depth_bin": category[3],
            "td_sign": category[4],
            "abs_td_bin": category[5],
            "count": len(values),
            "sum_target": _fsum(targets),
            "sum_target_sq": _fsum(value * value for value in targets),
            "sum_selected_q": _fsum(selected),
            "sum_selected_q_sq": _fsum(value * value for value in selected),
            "sum_target_selected_q": _fsum(
                left * right for left, right in zip(targets, selected)
            ),
            "sum_td": _fsum(residuals),
            "sum_abs_td": _fsum(absolute),
            "sum_td_sq": _fsum(value * value for value in residuals),
            "max_abs_td": max(absolute, default=0.0),
        }
        rows.append(row)
    return tuple(rows)


def build_replay_row(**values):
    row = {
        "telemetry_schema_version": VOC_TELEMETRY_SCHEMA_VERSION,
        "gate_schema": VOC_GATE_SCHEMA_VERSION,
        **values,
    }
    actor_ids = row["actor_ids"]
    if isinstance(actor_ids, Sequence) and type(actor_ids) is not str:
        if len(actor_ids) != 16 or any(type(value) is not int for value in actor_ids):
            raise TypeError("actor_ids must contain 16 built-in integers")
        actor_ids = ";".join(str(value) for value in actor_ids)
        row["actor_ids"] = actor_ids
    _encode_token("actor_ids", actor_ids)
    row["actor_ids_sha256"] = hashlib.sha256(actor_ids.encode("ascii")).hexdigest()
    if set(row) != set(REPLAY_FIELDS):
        raise ValueError("replay row has the wrong keyset")
    encode_csv_row(REPLAY_FIELDS, row)
    return row


def build_q_transaction_row(*, diagnostics=None, **values):
    status = values.get("q_status")
    if status not in Q_STATUSES:
        raise ValueError("q_status is not a frozen schema-13 status")
    if diagnostics is None:
        diagnostics = {}
    if not isinstance(diagnostics, Mapping):
        raise TypeError("Q diagnostics must be a mapping")
    expected_diagnostic_keys = set(Q_DIAGNOSTIC_FIELDS) if status == "stepped" else set()
    if set(diagnostics) != expected_diagnostic_keys:
        raise ValueError("Q diagnostic availability disagrees with q_status")
    row = {
        "telemetry_schema_version": VOC_TELEMETRY_SCHEMA_VERSION,
        "gate_schema": VOC_GATE_SCHEMA_VERSION,
        **values,
        **(
            dict(diagnostics)
            if status == "stepped"
            else {name: "NA" for name in Q_DIAGNOSTIC_FIELDS}
        ),
    }
    if set(row) != set(Q_FIELDS):
        raise ValueError("Q transaction row has the wrong keyset")
    encode_csv_row(Q_FIELDS, row)
    return row


def _write_once(fd, payload, label):
    written = os.write(fd, payload)
    if written != len(payload):
        raise OSError(f"short write for {label}: {written}/{len(payload)}")


def _same_identity(left, right):
    return (
        left.st_dev,
        left.st_ino,
        left.st_mode,
        left.st_uid,
        left.st_gid,
        left.st_nlink,
        left.st_size,
    ) == (
        right.st_dev,
        right.st_ino,
        right.st_mode,
        right.st_uid,
        right.st_gid,
        right.st_nlink,
        right.st_size,
    )


def _validate_regular_stat(info, *, label, mode=None, links=1):
    if not stat.S_ISREG(info.st_mode):
        raise RuntimeError(f"{label} is not a regular file")
    if info.st_uid != os.geteuid() or info.st_gid != os.getegid():
        raise RuntimeError(f"{label} owner/group disagrees with launcher")
    if info.st_nlink != links:
        raise RuntimeError(f"{label} link count must be {links}")
    if mode is not None and stat.S_IMODE(info.st_mode) != mode:
        raise RuntimeError(f"{label} mode must be {mode:04o}")


def _stable_read_fd(fd, path, *, label, mode=None, expected_identity=None):
    """Read exact bytes from one caller-owned no-follow descriptor."""

    before = os.fstat(fd)
    _validate_regular_stat(before, label=label, mode=mode)
    if expected_identity is not None and (
        before.st_dev,
        before.st_ino,
    ) != tuple(expected_identity):
        raise RuntimeError(f"{label} is not the expected bound inode")
    os.lseek(fd, 0, os.SEEK_SET)
    chunks = []
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    after = os.fstat(fd)
    if not _same_identity(before, after):
        raise RuntimeError(f"{label} changed during descriptor-bound read")
    pathname = os.stat(path, follow_symlinks=False)
    if pathname.st_dev != after.st_dev or pathname.st_ino != after.st_ino:
        raise RuntimeError(f"{label} pathname identity changed during read")
    payload = b"".join(chunks)
    if len(payload) != after.st_size:
        raise RuntimeError(f"{label} size changed during read")
    return payload, after


def _read_only_nofollow_flags():
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    return flags


def _stable_read(path, *, label, mode=None, expected_identity=None):
    flags = _read_only_nofollow_flags()
    fd = os.open(path, flags)
    try:
        return _stable_read_fd(
            fd,
            path,
            label=label,
            mode=mode,
            expected_identity=expected_identity,
        )
    finally:
        os.close(fd)


def _fsync_directory(path):
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class Schema13TelemetryWriter:
    """Fresh-only commit-last writer for one schema-13 learner."""

    def __init__(
        self,
        run_dir,
        *,
        xpid,
        actor_unroll_len,
        stage_total_steps=None,
        q_initial_lr=None,
        schedule_total_steps=None,
        amp_initial_scale=None,
    ):
        self.run_dir = os.path.abspath(os.fspath(run_dir))
        self.xpid = _require_exact_string("xpid", xpid)
        self.actor_unroll_len = _require_exact_int(
            "actor_unroll_len", actor_unroll_len, minimum=1
        )
        self.stage_total_steps = (
            None
            if stage_total_steps is None
            else _require_exact_int("stage_total_steps", stage_total_steps, minimum=1)
        )
        self.q_initial_lr = (
            None if q_initial_lr is None else _finite_float("q_initial_lr", q_initial_lr)
        )
        if self.q_initial_lr is not None and self.q_initial_lr <= 0.0:
            raise ValueError("q_initial_lr must be positive")
        self.schedule_total_steps = (
            None
            if schedule_total_steps is None
            else _require_exact_int(
                "schedule_total_steps", schedule_total_steps, minimum=1
            )
        )
        if (self.q_initial_lr is None) != (self.schedule_total_steps is None):
            raise ValueError(
                "q_initial_lr and schedule_total_steps must be supplied together"
            )
        self.amp_initial_scale = (
            None
            if amp_initial_scale is None
            else _finite_float("amp_initial_scale", amp_initial_scale)
        )
        if self.amp_initial_scale is not None and self.amp_initial_scale not in (
            1.0,
            256.0,
        ):
            raise ValueError("amp_initial_scale must be frozen FP32 1 or FP16 256")
        directory = os.stat(self.run_dir, follow_symlinks=False)
        if not stat.S_ISDIR(directory.st_mode):
            raise ValueError("telemetry run_dir must be an existing directory")
        if directory.st_uid != os.geteuid() or directory.st_gid != os.getegid():
            raise RuntimeError("telemetry run_dir owner/group disagrees")
        self._fds = {}
        self._identities = {}
        self._sizes = {}
        self._transaction_count = 0
        self._last_commit_bytes = None
        self._last_replay_row = None
        self._poisoned = False
        self._sealed = False
        self._closed = False
        try:
            seen = set()
            for name in SIDECAR_FILENAMES:
                path = os.path.join(self.run_dir, name)
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW
                fd = os.open(path, flags, 0o600)
                self._fds[name] = fd
                os.fchmod(fd, 0o600)
                _write_once(fd, HEADERS[name], f"{name} header")
                os.fsync(fd)
                info = os.fstat(fd)
                _validate_regular_stat(info, label=name, mode=0o600)
                if info.st_size != len(HEADERS[name]):
                    raise RuntimeError(f"{name} header size disagrees")
                identity = (info.st_dev, info.st_ino)
                if identity in seen:
                    raise RuntimeError("telemetry sidecars alias one inode")
                seen.add(identity)
                self._identities[name] = identity
                self._sizes[name] = info.st_size
            _fsync_directory(self.run_dir)
        except BaseException:
            self._poisoned = True
            self._close_fds()
            raise

    @property
    def transaction_count(self):
        return self._transaction_count

    @property
    def poisoned(self):
        return self._poisoned

    def _require_writable(self):
        if self._poisoned:
            raise RuntimeError("schema-13 telemetry writer is poisoned")
        if self._sealed or self._closed:
            raise RuntimeError("schema-13 telemetry writer is not writable")

    def _validate_live_fd(self, name, *, expected_size):
        info = os.fstat(self._fds[name])
        _validate_regular_stat(info, label=name, mode=0o600)
        if (info.st_dev, info.st_ino) != self._identities[name]:
            raise RuntimeError(f"{name} bound inode changed")
        if info.st_size != expected_size:
            raise RuntimeError(f"{name} append offset drifted")
        pathname = os.stat(os.path.join(self.run_dir, name), follow_symlinks=False)
        if (pathname.st_dev, pathname.st_ino) != self._identities[name]:
            raise RuntimeError(f"{name} pathname was replaced")

    def _append_block(self, name, payload):
        expected = self._sizes[name]
        self._validate_live_fd(name, expected_size=expected)
        _write_once(self._fds[name], payload, f"{name} transaction block")
        os.fsync(self._fds[name])
        self._sizes[name] = expected + len(payload)
        self._validate_live_fd(name, expected_size=self._sizes[name])

    def append_transaction(
        self,
        *,
        td_rows,
        replay_row,
        q_row,
        terminal,
        actor_state_sha256,
        publication_history_sha256,
    ):
        self._require_writable()
        try:
            transaction_id = self._transaction_count + 1
            if type(td_rows) not in (tuple, list) or len(td_rows) != 720:
                raise ValueError("each telemetry transaction requires 720 TD rows")
            if not isinstance(replay_row, Mapping) or not isinstance(q_row, Mapping):
                raise TypeError("replay and Q rows must be mappings")
            common = {
                "telemetry_schema_version": VOC_TELEMETRY_SCHEMA_VERSION,
                "gate_schema": VOC_GATE_SCHEMA_VERSION,
                "transaction_id": transaction_id,
                "source_policy_version": transaction_id - 1,
                "published_policy_version": transaction_id,
            }
            for row in tuple(td_rows) + (replay_row, q_row):
                for name, expected in common.items():
                    if row.get(name) != expected or type(row.get(name)) is not int:
                        raise ValueError(f"transaction row disagrees on {name}")
            terminal_bit = _encode_bool("terminal", terminal)
            terminal_value = terminal_bit == "1"
            if replay_row.get("terminal") not in (terminal_value, int(terminal_value)):
                raise ValueError("replay terminal bit disagrees")
            actor_digest = _require_hash("actor_state_sha256", actor_state_sha256)
            history_digest = _require_hash(
                "publication_history_sha256", publication_history_sha256
            )
            if (
                replay_row.get("actor_state_sha256") != actor_digest
                or replay_row.get("publication_history_sha256") != history_digest
            ):
                raise ValueError("replay publication identity disagrees")
            td_block = b"".join(encode_csv_row(TD_FIELDS, row) for row in td_rows)
            replay_block = encode_csv_row(REPLAY_FIELDS, replay_row)
            q_block = encode_csv_row(Q_FIELDS, q_row)
            commit = {
                **common,
                "terminal": terminal_value,
                "td_first_data_row": 720 * (transaction_id - 1) + 1,
                "td_data_row_count": 720,
                "td_block_byte_count": len(td_block),
                "td_block_sha256": hashlib.sha256(td_block).hexdigest(),
                "replay_first_data_row": transaction_id,
                "replay_data_row_count": 1,
                "replay_block_byte_count": len(replay_block),
                "replay_block_sha256": hashlib.sha256(replay_block).hexdigest(),
                "q_first_data_row": transaction_id,
                "q_data_row_count": 1,
                "q_block_byte_count": len(q_block),
                "q_block_sha256": hashlib.sha256(q_block).hexdigest(),
                "publication_count": replay_row["publication_count_after"],
                "ack_count": replay_row["ack_count"],
                "actor_state_sha256": actor_digest,
                "publication_history_sha256": history_digest,
            }
            commit_block = encode_csv_row(COMMIT_FIELDS, commit)
            self._append_block(TD_FILENAME, td_block)
            self._append_block(REPLAY_FILENAME, replay_block)
            self._append_block(Q_FILENAME, q_block)
            self._append_block(COMMIT_FILENAME, commit_block)
            self._transaction_count = transaction_id
            self._last_commit_bytes = commit_block
            self._last_replay_row = dict(replay_row)
            return dict(commit)
        except BaseException:
            self._poisoned = True
            raise

    def _close_fds(self):
        for fd in tuple(self._fds.values()):
            try:
                os.close(fd)
            except OSError:
                pass
        self._fds.clear()
        self._closed = True

    def abort(self):
        self._poisoned = True
        self._close_fds()

    def _seal_sidecars(self):
        payloads = {}
        for name in SIDECAR_FILENAMES:
            fd = self._fds[name]
            self._validate_live_fd(name, expected_size=self._sizes[name])
            os.fsync(fd)
            os.fchmod(fd, 0o400)
            os.fsync(fd)
            info = os.fstat(fd)
            _validate_regular_stat(info, label=name, mode=0o400)
            if (info.st_dev, info.st_ino) != self._identities[name]:
                raise RuntimeError(f"{name} inode changed while sealing")
            os.close(fd)
            del self._fds[name]
            path = os.path.join(self.run_dir, name)
            read_flags = _read_only_nofollow_flags()
            read_fd = os.open(path, read_flags)
            try:
                payload, stable = _stable_read_fd(
                    read_fd,
                    path,
                    label=name,
                    mode=0o400,
                    expected_identity=self._identities[name],
                )
            finally:
                os.close(read_fd)
            if len(payload) != self._sizes[name]:
                raise RuntimeError(f"{name} seal validation disagrees")
            payloads[name] = payload
        records, parsed = _validated_payload_records(
            payloads,
            expected_actor_unroll_len=self.actor_unroll_len,
            expected_q_initial_lr=self.q_initial_lr,
            expected_schedule_total_steps=self.schedule_total_steps,
            expected_amp_initial_scale=self.amp_initial_scale,
        )
        return records, parsed

    def seal(
        self,
        *,
        terminal_real_step,
        terminal_policy_version,
        terminal_publication_count,
        terminal_ack_count,
        legacy_actor_log_path,
        legacy_actor_log_fd=None,
    ):
        self._require_writable()
        if self._transaction_count < 1 or self._last_replay_row is None:
            raise RuntimeError("cannot seal empty telemetry")
        terminal_real_step = _require_exact_int(
            "terminal_real_step", terminal_real_step, minimum=1
        )
        terminal_policy_version = _require_exact_int(
            "terminal_policy_version", terminal_policy_version, minimum=1
        )
        terminal_publication_count = _require_exact_int(
            "terminal_publication_count", terminal_publication_count, minimum=1
        )
        terminal_ack_count = _require_exact_int(
            "terminal_ack_count", terminal_ack_count, minimum=1
        )
        if (
            terminal_policy_version != self._transaction_count
            or terminal_publication_count != self._transaction_count
            or terminal_ack_count != 1
            or self._last_replay_row.get("terminal") not in (True, 1)
        ):
            raise ValueError("terminal telemetry identity/counts disagree")
        if self.stage_total_steps is not None and terminal_real_step < self.stage_total_steps:
            raise ValueError("terminal real step is below the stage total")
        log_path = os.path.abspath(os.fspath(legacy_actor_log_path))
        if os.path.dirname(log_path) != self.run_dir or os.path.basename(log_path) != LEGACY_LOG_FILENAME:
            raise ValueError("legacy actor log must be run_dir/logs.csv")
        try:
            records, parsed = self._seal_sidecars()
            if legacy_actor_log_fd is None:
                legacy_record, legacy_identity = _legacy_log_record(
                    log_path,
                    self._transaction_count,
                    expected_replay_rows=parsed[REPLAY_FILENAME]["rows"],
                    expected_q_rows=parsed[Q_FILENAME]["rows"],
                    return_identity=True,
                )
            else:
                if type(legacy_actor_log_fd) is not int or legacy_actor_log_fd < 0:
                    raise TypeError("legacy_actor_log_fd must be a non-negative fd")
                log_payload, log_info = _stable_read_fd(
                    legacy_actor_log_fd,
                    log_path,
                    label=LEGACY_LOG_FILENAME,
                )
                legacy_record = _legacy_log_record_from_payload(
                    log_payload,
                    self._transaction_count,
                    expected_replay_rows=parsed[REPLAY_FILENAME]["rows"],
                    expected_q_rows=parsed[Q_FILENAME]["rows"],
                )
                legacy_identity = (log_info.st_dev, log_info.st_ino)
            final_commit = parsed[COMMIT_FILENAME]["raw_rows"][-1]
            last_commit = {
                "data_row": self._transaction_count,
                "transaction_id": self._transaction_count,
                "sha256": hashlib.sha256(final_commit).hexdigest(),
                "actor_state_sha256": self._last_replay_row["actor_state_sha256"],
                "publication_history_sha256": self._last_replay_row[
                    "publication_history_sha256"
                ],
            }
            manifest = {
                "telemetry_schema_version": VOC_TELEMETRY_SCHEMA_VERSION,
                "gate_schema": VOC_GATE_SCHEMA_VERSION,
                "status": "sealed",
                "xpid": self.xpid,
                "fresh": True,
                "transaction_count": self._transaction_count,
                "terminal_policy_version": terminal_policy_version,
                "terminal_real_step": terminal_real_step,
                "terminal_publication_count": terminal_publication_count,
                "terminal_ack_count": terminal_ack_count,
                "actor_state_sha256": self._last_replay_row["actor_state_sha256"],
                "publication_history_sha256": self._last_replay_row[
                    "publication_history_sha256"
                ],
                "legacy_actor_log": legacy_record,
                "artifacts": [records[name] for name in SIDECAR_FILENAMES],
                "last_commit": last_commit,
            }
            manifest_bytes = canonical_json_bytes(manifest, trailing_lf=True)
            _publish_manifest(self.run_dir, manifest_bytes)
            evidence = validate_schema13_telemetry_manifest(
                self.run_dir,
                expected_xpid=self.xpid,
                expected_terminal_policy_version=terminal_policy_version,
                expected_terminal_real_step=terminal_real_step,
                expected_actor_state_sha256=manifest["actor_state_sha256"],
                expected_publication_history_sha256=manifest[
                    "publication_history_sha256"
                ],
                expected_stage_total_steps=self.stage_total_steps,
                expected_actor_unroll_len=self.actor_unroll_len,
                expected_terminal_ack_count=terminal_ack_count,
                expected_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                expected_manifest_size=len(manifest_bytes),
                expected_q_initial_lr=self.q_initial_lr,
                expected_schedule_total_steps=self.schedule_total_steps,
                expected_amp_initial_scale=self.amp_initial_scale,
                expected_sidecar_identities=self._identities,
                expected_legacy_log_identity=legacy_identity,
            )
            self._sealed = True
            self._closed = True
            return evidence
        except BaseException:
            self._poisoned = True
            self._close_fds()
            raise


def _publish_manifest(run_dir, payload):
    final_path = os.path.join(run_dir, MANIFEST_FILENAME)
    temp_path = os.path.join(run_dir, "." + MANIFEST_FILENAME + ".tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(temp_path, flags, 0o600)
    try:
        os.fchmod(fd, 0o600)
        _write_once(fd, payload, "telemetry manifest")
        os.fsync(fd)
        os.fchmod(fd, 0o400)
        os.fsync(fd)
        before = os.fstat(fd)
        _validate_regular_stat(before, label="telemetry manifest temporary", mode=0o400)
        if before.st_size != len(payload):
            raise RuntimeError("telemetry manifest temporary size disagrees")
        os.link(temp_path, final_path, follow_symlinks=False)
        linked = os.fstat(fd)
        _validate_regular_stat(
            linked, label="linked telemetry manifest", mode=0o400, links=2
        )
        os.unlink(temp_path)
        _fsync_directory(run_dir)
        final_payload, final_info = _stable_read(
            final_path, label="telemetry manifest", mode=0o400
        )
        if (
            final_info.st_dev != before.st_dev
            or final_info.st_ino != before.st_ino
            or final_payload != payload
        ):
            raise RuntimeError("published telemetry manifest identity disagrees")
    finally:
        os.close(fd)


def _parse_uint(token, *, name):
    if type(token) is not str or _UINT_RE.fullmatch(token) is None:
        raise ValueError(f"{name} is not a canonical unsigned integer")
    return int(token)


def _decode_csv_row(fields, tokens, *, label):
    if len(tokens) != len(fields):
        raise ValueError(f"{label} has the wrong column count")
    row = {}
    for name, token in zip(fields, tokens):
        kind = _field_kind(fields, name)
        if kind == "float":
            row[name] = parse_canonical_float(
                token,
                allow_na=(fields == Q_FIELDS and name in Q_DIAGNOSTIC_FIELDS),
                name=f"{label} {name}",
            )
        elif kind == "hash":
            if _LOWER_HEX_RE.fullmatch(token) is None:
                raise ValueError(f"{label} {name} is not lowercase SHA-256")
            row[name] = token
        elif kind == "bool":
            if token not in ("0", "1"):
                raise ValueError(f"{label} {name} is not a canonical bit")
            row[name] = token == "1"
        elif kind == "token":
            if (
                not token
                or not token.isascii()
                or any(character in token for character in (",", "\r", "\n", '"', " "))
            ):
                raise ValueError(f"{label} {name} is not a canonical token")
            row[name] = token
        else:
            row[name] = _parse_uint(token, name=f"{label} {name}")
    return row


def _parse_sidecar(name, payload):
    header = HEADERS[name]
    fields = FIELDS_BY_FILENAME[name]
    if (
        not payload.startswith(header)
        or payload.startswith(b"\xef\xbb\xbf")
        or b"\r" in payload
        or b'"' in payload
        or not payload.endswith(b"\n")
    ):
        raise ValueError(f"{name} violates the canonical sidecar grammar")
    try:
        payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError(f"{name} is not ASCII") from error
    lines = payload.splitlines(keepends=True)
    if not lines or lines[0] != header:
        raise ValueError(f"{name} header bytes disagree")
    raw_rows = lines[1:]
    if any(line in (b"", b"\n") or not line.endswith(b"\n") for line in raw_rows):
        raise ValueError(f"{name} contains a blank or unterminated row")
    rows = []
    for index, raw in enumerate(raw_rows, start=1):
        body = raw[:-1]
        if body.endswith(b" "):
            raise ValueError(f"{name} data row {index} has trailing space")
        tokens = body.decode("ascii").split(",")
        row = _decode_csv_row(fields, tokens, label=f"{name} data row {index}")
        rows.append(row)
    return {"rows": rows, "raw_rows": raw_rows, "payload": payload}


def _file_record(name, parsed):
    payload = parsed["payload"]
    return {
        "name": name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
        "header_sha256": HEADER_SHA256[name],
        "header_size": len(HEADERS[name]),
        "column_count": len(FIELDS_BY_FILENAME[name]),
        "data_row_count": len(parsed["rows"]),
    }


def _legacy_log_record_from_payload(
    payload,
    transaction_count,
    *,
    expected_replay_rows=None,
    expected_q_rows=None,
):
    header_end = payload.find(b"\n")
    if header_end < 0:
        raise ValueError("logs.csv lacks its frozen header terminator")
    header = payload[: header_end + 1]
    if (
        len(header) != LEGACY_LOG_HEADER_SIZE
        or hashlib.sha256(header).hexdigest() != LEGACY_LOG_HEADER_SHA256
        or len(header[:-1].split(b",")) != LEGACY_LOG_COLUMN_COUNT
    ):
        raise ValueError("logs.csv header differs from the frozen 922-column bytes")
    data = payload[header_end + 1 :]
    if transaction_count > 0 and not data.endswith(b"\r\n"):
        raise ValueError("logs.csv has an unterminated data row")
    row_bodies = data.split(b"\r\n")
    if row_bodies and row_bodies[-1] == b"":
        row_bodies.pop()
    if any(not row or b"\r" in row or b"\n" in row for row in row_bodies):
        raise ValueError("logs.csv contains a blank or malformed data row")
    rows = [row + b"\r\n" for row in row_bodies]
    if len(rows) != transaction_count:
        raise ValueError("logs.csv data-row count disagrees with telemetry")
    try:
        header_fields = header[:-1].decode("ascii").split(",")
    except UnicodeDecodeError as error:
        raise ValueError("logs.csv header is not ASCII") from error
    q_loss_name = "actor/voc_q_loss"
    if header_fields.count(q_loss_name) != 1:
        raise ValueError("logs.csv lacks its unique frozen actor/voc_q_loss column")
    q_loss_index = header_fields.index(q_loss_name)
    q_loss_values = []
    for expected_tick, body in enumerate(row_bodies):
        if b'"' in body:
            raise ValueError("logs.csv data rows must remain unquoted")
        try:
            tokens = body.decode("ascii").split(",")
        except UnicodeDecodeError as error:
            raise ValueError("logs.csv data row is not ASCII") from error
        if len(tokens) != LEGACY_LOG_COLUMN_COUNT:
            raise ValueError("logs.csv data row is not exactly 922 columns")
        if tokens[0] != str(expected_tick):
            raise ValueError(
                "logs.csv tick sequence disagrees with telemetry transactions"
            )
        q_loss_token = tokens[q_loss_index]
        if not q_loss_token or q_loss_token.strip() != q_loss_token:
            raise ValueError("logs.csv actor/voc_q_loss is malformed")
        try:
            q_loss_value = float(q_loss_token)
        except ValueError as error:
            raise ValueError("logs.csv actor/voc_q_loss is malformed") from error
        if not math.isfinite(q_loss_value):
            raise ValueError("logs.csv actor/voc_q_loss must be finite")
        q_loss_values.append(q_loss_value)
    if (expected_replay_rows is None) != (expected_q_rows is None):
        raise ValueError("legacy Q-loss reconciliation inputs must be paired")
    if expected_replay_rows is not None:
        if (
            len(expected_replay_rows) != transaction_count
            or len(expected_q_rows) != transaction_count
        ):
            raise ValueError("legacy Q-loss reconciliation row counts disagree")
        for index, (logged, replay, q_row) in enumerate(
            zip(q_loss_values, expected_replay_rows, expected_q_rows), start=1
        ):
            denominator = max(replay["train_count"], 1)
            expected = q_row["q_loss_sum"] / denominator
            if not _close_enough(logged, expected):
                raise ValueError(
                    f"logs.csv actor/voc_q_loss disagrees at transaction {index}"
                )
    return {
        "name": LEGACY_LOG_FILENAME,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
        "header_sha256": LEGACY_LOG_HEADER_SHA256,
        "header_size": LEGACY_LOG_HEADER_SIZE,
        "column_count": LEGACY_LOG_COLUMN_COUNT,
        "data_row_count": transaction_count,
    }


def _legacy_log_record(
    path,
    transaction_count,
    *,
    expected_replay_rows=None,
    expected_q_rows=None,
    expected_identity=None,
    return_identity=False,
):
    payload, info = _stable_read(
        path,
        label=LEGACY_LOG_FILENAME,
        expected_identity=expected_identity,
    )
    record = _legacy_log_record_from_payload(
        payload,
        transaction_count,
        expected_replay_rows=expected_replay_rows,
        expected_q_rows=expected_q_rows,
    )
    if return_identity:
        return record, (info.st_dev, info.st_ino)
    return record


def _close_enough(left, right):
    return abs(left - right) <= (
        _FORWARD_ABS * max(1.0, abs(left), abs(right)) + _FORWARD_ADD
    )


def _validate_td_row(row, *, expected_identity, expected_category):
    for name, expected in expected_identity.items():
        if row[name] != expected:
            raise ValueError(f"TD row disagrees on {name}")
    actual_category = tuple(
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
    if actual_category != expected_category:
        raise ValueError("TD cube category order or label disagrees")
    count = row["count"]
    for name in TD_FLOAT_FIELDS:
        value = row[name]
        if value is None or not math.isfinite(value):
            raise ValueError(f"TD row {name} must be finite")
    if count == 0 and any(row[name] != 0.0 for name in TD_FLOAT_FIELDS):
        raise ValueError("empty TD cell statistics must be positive zero")
    if (
        row["sum_target_sq"] < 0.0
        or row["sum_selected_q_sq"] < 0.0
        or row["sum_abs_td"] < 0.0
        or row["sum_td_sq"] < 0.0
        or row["max_abs_td"] < 0.0
    ):
        raise ValueError("TD absolute/squared/max statistics must be nonnegative")
    if count == 0:
        return
    if row["max_abs_td"] > row["sum_abs_td"] + _FORWARD_ADD:
        raise ValueError("TD max absolute residual exceeds its absolute sum")
    if row["sum_abs_td"] > count * row["max_abs_td"] + _FORWARD_ADD:
        raise ValueError("TD absolute sum exceeds count times maximum")
    sign = row["td_sign"]
    if sign == "zero":
        if (
            row["abs_td_bin"] != "0_0p5"
            or row["sum_td"] != 0.0
            or row["sum_abs_td"] != 0.0
            or row["sum_td_sq"] != 0.0
            or row["max_abs_td"] != 0.0
        ):
            raise ValueError("zero-sign TD cell has nonzero residual moments")
    elif sign == "positive":
        if row["sum_td"] <= 0.0 or not _close_enough(
            row["sum_td"], row["sum_abs_td"]
        ):
            raise ValueError("positive-sign TD cell has inconsistent moments")
    elif sign == "negative":
        if row["sum_td"] >= 0.0 or not _close_enough(
            -row["sum_td"], row["sum_abs_td"]
        ):
            raise ValueError("negative-sign TD cell has inconsistent moments")
    lower, upper = {
        "0_0p5": (0.0, 0.5),
        "0p5_1": (0.5, 1.0),
        "1_2": (1.0, 2.0),
        "2_4": (2.0, 4.0),
        "4_inf": (4.0, None),
    }[row["abs_td_bin"]]
    if (
        row["max_abs_td"] < lower
        or row["sum_abs_td"] < lower * count
        or row["sum_td_sq"] < lower * lower * count
    ):
        raise ValueError("TD residual moments fall below their absolute band")
    if upper is not None and (
        row["max_abs_td"] >= upper
        or row["sum_abs_td"] >= upper * count
        or row["sum_td_sq"] >= upper * upper * count
    ):
        raise ValueError("TD residual moments exceed their half-open absolute band")
    if row["sum_target_sq"] + _FORWARD_ADD < row["sum_target"] ** 2 / count:
        raise ValueError("TD target moments violate nonnegative variance")
    if (
        row["sum_selected_q_sq"] + _FORWARD_ADD
        < row["sum_selected_q"] ** 2 / count
    ):
        raise ValueError("TD selected-Q moments violate nonnegative variance")
    if not _close_enough(
        row["sum_td"], row["sum_target"] - row["sum_selected_q"]
    ):
        raise ValueError("TD first moment disagrees with target minus selected Q")
    reconstructed_td_sq = (
        row["sum_target_sq"]
        + row["sum_selected_q_sq"]
        - 2.0 * row["sum_target_selected_q"]
    )
    if not _close_enough(row["sum_td_sq"], reconstructed_td_sq):
        raise ValueError("TD squared moment disagrees with target/Q moments")


def _validate_q_row(row, replay):
    status = row["q_status"]
    if status not in Q_STATUSES or replay["q_status"] != status:
        raise ValueError("Q and replay status disagree")
    for name in Q_BASE_FLOAT_FIELDS:
        value = row[name]
        if value is None or not math.isfinite(value):
            raise ValueError(f"Q {name} must always be finite")
    if row["q_loss_sum"] < 0.0 or row["clip_limit"] < 0.0:
        raise ValueError("Q loss and clip limit must be nonnegative")
    if row["amp_scale_before"] <= 0.0 or row["amp_scale_after"] <= 0.0:
        raise ValueError("Q AMP scales must be positive")
    expected_clip = 0.5 * replay["replay_t"] * replay["replay_b"]
    if row["clip_limit"] != expected_clip:
        raise ValueError("Q clip limit does not use overlap-inclusive T*B")
    diagnostics = [row[name] for name in Q_DIAGNOSTIC_FIELDS]
    if status == "stepped":
        if (
            row["q_attempted"] is not True
            or row["q_optimizer_committed"] is not True
            or row["nonfinite_gradient_parameter_count"] != 0
            or row["adam_step_after"] != row["adam_step_before"] + 1
            or row["amp_scale_after"] < row["amp_scale_before"]
            or any(value is None or not math.isfinite(value) for value in diagnostics)
        ):
            raise ValueError("stepped Q status matrix disagrees")
        if any(value < 0.0 for value in diagnostics):
            raise ValueError("Q diagnostic norms/means must be nonnegative")
        if not 0.0 < row["clip_scale"] <= 1.0:
            raise ValueError("Q clip_scale must lie in (0,1]")
        if row["raw_preclip_total_l2"] == 0.0:
            if row["raw_postclip_total_l2"] != 0.0 or row["clip_scale"] != 1.0:
                raise ValueError("zero Q gradient clipping diagnostics disagree")
        else:
            expected_clip_scale = min(
                1.0,
                row["clip_limit"]
                / (row["raw_preclip_total_l2"] + 1.0e-6),
            )
            if not _close_enough(row["clip_scale"], expected_clip_scale):
                raise ValueError("Q clip_scale disagrees with the frozen clip rule")
            if not _close_enough(
                row["raw_postclip_total_l2"],
                row["raw_preclip_total_l2"] * row["clip_scale"],
            ):
                raise ValueError("Q post/preclip norm ratio disagrees")
        for prefix in ("raw_preclip", "raw_postclip"):
            total = row[f"{prefix}_total_l2"]
            components = [
                row[f"{prefix}_weight_continue_l2"],
                row[f"{prefix}_weight_stop_l2"],
                row[f"{prefix}_bias_continue_l2"],
                row[f"{prefix}_bias_stop_l2"],
            ]
            reconstructed = math.sqrt(math.fsum(value * value for value in components))
            if not _close_enough(total, reconstructed):
                raise ValueError(f"Q {prefix} total norm disagrees with row norms")
        lr = abs(replay["q_lr_used"])
        for parameter in ("weight", "bias"):
            for coordinate in ("common", "difference"):
                normalized = row[f"normalized_update_{parameter}_{coordinate}_l2"]
                delta = row[f"coordinate_delta_{parameter}_{coordinate}_l2"]
                if not _close_enough(delta, lr * normalized):
                    raise ValueError("Q normalized update and coordinate delta disagree")
            coordinate_total = math.sqrt(math.fsum(
                row[f"coordinate_delta_{parameter}_{coordinate}_l2"] ** 2
                for coordinate in ("common", "difference")
            ))
            mapped_total = math.sqrt(math.fsum(
                row[f"mapped_delta_{parameter}_{action}_l2"] ** 2
                for action in ("continue", "stop")
            ))
            if not _close_enough(coordinate_total, mapped_total):
                raise ValueError("Q orthonormal mapped delta norm disagrees")
    elif status == "no_support":
        if (
            row["q_attempted"] is not False
            or row["q_optimizer_committed"] is not False
            or row["nonfinite_gradient_parameter_count"] != 0
            or row["q_loss_sum"] != 0.0
            or row["amp_scale_after"] != row["amp_scale_before"]
            or row["adam_step_after"] != row["adam_step_before"]
            or any(value is not None for value in diagnostics)
        ):
            raise ValueError("no_support Q status matrix disagrees")
    else:
        if (
            row["q_attempted"] is not True
            or row["q_optimizer_committed"] is not False
            or row["nonfinite_gradient_parameter_count"] not in (1, 2)
            or row["amp_scale_after"] >= row["amp_scale_before"]
            or row["adam_step_after"] != row["adam_step_before"]
            or any(value is not None for value in diagnostics)
        ):
            raise ValueError("amp_skip Q status matrix disagrees")


def _validate_transaction_rows(
    parsed,
    *,
    expected_actor_unroll_len,
    expected_q_initial_lr=None,
    expected_schedule_total_steps=None,
    expected_amp_initial_scale=None,
):
    td_rows = parsed[TD_FILENAME]["rows"]
    replay_rows = parsed[REPLAY_FILENAME]["rows"]
    q_rows = parsed[Q_FILENAME]["rows"]
    commit_rows = parsed[COMMIT_FILENAME]["rows"]
    transaction_count = len(commit_rows)
    if transaction_count < 1:
        raise ValueError("telemetry must contain at least one committed transaction")
    if (
        len(td_rows) != 720 * transaction_count
        or len(replay_rows) != transaction_count
        or len(q_rows) != transaction_count
    ):
        raise ValueError("telemetry artifact row cardinalities disagree")
    if (expected_q_initial_lr is None) != (expected_schedule_total_steps is None):
        raise ValueError(
            "expected_q_initial_lr and expected_schedule_total_steps must be paired"
        )
    if expected_q_initial_lr is not None:
        expected_q_initial_lr = _finite_float(
            "expected_q_initial_lr", expected_q_initial_lr
        )
        if expected_q_initial_lr <= 0.0:
            raise ValueError("expected_q_initial_lr must be positive")
        expected_schedule_total_steps = _require_exact_int(
            "expected_schedule_total_steps",
            expected_schedule_total_steps,
            minimum=1,
        )
    if expected_amp_initial_scale is not None:
        expected_amp_initial_scale = _finite_float(
            "expected_amp_initial_scale", expected_amp_initial_scale
        )
        if expected_amp_initial_scale not in (1.0, 256.0):
            raise ValueError(
                "expected_amp_initial_scale must be frozen FP32 1 or FP16 256"
            )
    expected_amp_scale = expected_amp_initial_scale
    # The frozen learner has exactly two modes: its private FP16 scaler starts
    # at 2**8, while FP32 records observational scale 1.0 and has no scaler.
    # In particular, FP32 must not synthesize GradScaler growth at transaction
    # 2000 merely because its telemetry also carries a finite scale field.
    expected_amp_scaler_enabled = (
        expected_amp_initial_scale is not None
        and expected_amp_initial_scale != 1.0
    )
    expected_amp_growth_tracker = 0
    previous_replay = None
    previous_q = None
    for index in range(transaction_count):
        transaction_id = index + 1
        replay = replay_rows[index]
        q_row = q_rows[index]
        commit = commit_rows[index]
        identity = {
            "telemetry_schema_version": VOC_TELEMETRY_SCHEMA_VERSION,
            "gate_schema": VOC_GATE_SCHEMA_VERSION,
            "transaction_id": transaction_id,
            "source_policy_version": transaction_id - 1,
            "published_policy_version": transaction_id,
        }
        for row in (replay, q_row, commit):
            for name, expected in identity.items():
                if row[name] != expected:
                    raise ValueError(f"transaction {transaction_id} disagrees on {name}")
        td_block_rows = td_rows[index * 720 : (index + 1) * 720]
        for row, category in zip(td_block_rows, TD_CATEGORY_ORDER):
            _validate_td_row(
                row,
                expected_identity={
                    **identity,
                    "real_step_after": replay["real_step_after"],
                },
                expected_category=category,
            )
        if q_row["real_step_after"] != replay["real_step_after"]:
            raise ValueError("Q/replay real_step_after disagrees")
        if replay["replay_t"] != replay["optimized_t"] + 1:
            raise ValueError("replay T and optimized T disagree")
        if expected_actor_unroll_len is not None and (
            replay["optimized_t"] != expected_actor_unroll_len
            or replay["replay_t"] != expected_actor_unroll_len + 1
        ):
            raise ValueError("replay dimensions disagree with actor_unroll_len")
        if replay["replay_b"] != 16:
            raise ValueError("schema-13 replay batch must contain 16 actor streams")
        if _ACTOR_IDS_RE.fullmatch(replay["actor_ids"]) is None:
            raise ValueError("actor_ids spelling is malformed")
        actor_ids = tuple(int(token) for token in replay["actor_ids"].split(";"))
        if len(actor_ids) != 16 or set(actor_ids) != set(range(16)):
            raise ValueError("actor_ids must preserve a permutation of 0..15")
        if hashlib.sha256(replay["actor_ids"].encode("ascii")).hexdigest() != replay["actor_ids_sha256"]:
            raise ValueError("actor_ids digest disagrees")
        if replay["real_step_before"] + replay["real_step_delta"] != replay["real_step_after"]:
            raise ValueError("real-step before/delta/after relation disagrees")
        if index == 0:
            if replay["real_step_before"] != 0:
                raise ValueError("fresh telemetry must begin at real step zero")
            for name in (
                "voc_update_count_before",
                "ema_update_count_before",
                "projection_count_before",
            ):
                if replay[name] != 0:
                    raise ValueError(f"fresh telemetry must begin with {name}=0")
            if (
                replay["q_scheduler_last_epoch_before"] != 0
                or replay["q_scheduler_step_count_before"] != 1
                or q_row["adam_step_before"] != 0
            ):
                raise ValueError("fresh Q optimizer/scheduler state is not zero-base")
            if (
                expected_q_initial_lr is not None
                and replay["q_lr_before"] != expected_q_initial_lr
            ):
                raise ValueError("fresh Q learning rate disagrees with configuration")
        elif (
            replay["real_step_before"] != previous_replay["real_step_after"]
            or replay["voc_update_count_before"] != previous_replay["voc_update_count_after"]
            or replay["ema_update_count_before"] != previous_replay["ema_update_count_after"]
            or replay["projection_count_before"] != previous_replay["projection_count_after"]
            or replay["q_scheduler_last_epoch_before"] != previous_replay["q_scheduler_last_epoch_after"]
            or replay["q_scheduler_step_count_before"] != previous_replay["q_scheduler_step_count_after"]
            or replay["q_lr_before"] != previous_replay["q_lr_after"]
            or q_row["adam_step_before"] != previous_q["adam_step_after"]
        ):
            raise ValueError("telemetry transaction boundary continuity disagrees")
        maximum_support = replay["optimized_t"] * replay["replay_b"]
        if replay["real_step_delta"] > maximum_support:
            raise ValueError("real-step delta exceeds optimized T*B")
        count_names = (
            "valid_count",
            "train_count",
            "holdout_count",
            "train_continue_count",
            "train_stop_count",
            "holdout_continue_count",
            "holdout_stop_count",
        )
        if any(replay[name] > maximum_support for name in count_names):
            raise ValueError("replay support count exceeds optimized T*B")
        if (
            replay["valid_count"] != replay["train_count"] + replay["holdout_count"]
            or replay["train_count"]
            != replay["train_continue_count"] + replay["train_stop_count"]
            or replay["holdout_count"]
            != replay["holdout_continue_count"] + replay["holdout_stop_count"]
        ):
            raise ValueError("replay support/action count algebra disagrees")
        td_support = {}
        for source in Q_SOURCES:
            for split in SPLITS:
                for action in SELECTED_ACTIONS:
                    td_support[(source, split, action)] = sum(
                        row["count"]
                        for row in td_block_rows
                        if row["q_source"] == source
                        and row["split"] == split
                        and row["selected_action"] == action
                    )
        for source in Q_SOURCES:
            if (
                td_support[(source, "train", "continue")]
                != replay["train_continue_count"]
                or td_support[(source, "train", "stop")]
                != replay["train_stop_count"]
                or td_support[(source, "holdout", "continue")]
                != replay["holdout_continue_count"]
                or td_support[(source, "holdout", "stop")]
                != replay["holdout_stop_count"]
            ):
                raise ValueError("TD cube support does not reconstruct replay counts")
        if replay["q_lr_used"] != replay["q_lr_before"]:
            raise ValueError("q_lr_used must equal q_lr_before")
        if (
            replay["q_lr_before"] <= 0.0
            or replay["q_lr_used"] <= 0.0
            or replay["q_lr_after"] < 0.0
        ):
            raise ValueError("replay Q learning rates are outside the frozen range")
        if replay["q_status"] == "stepped":
            expected_delta = 1
        elif replay["q_status"] in ("no_support", "amp_skip"):
            expected_delta = 0
        else:
            raise ValueError("replay q_status is invalid")
        if (replay["train_count"] == 0) != (replay["q_status"] == "no_support"):
            raise ValueError("replay Q status disagrees with training support")
        for before, after in (
            ("voc_update_count_before", "voc_update_count_after"),
            ("ema_update_count_before", "ema_update_count_after"),
            ("projection_count_before", "projection_count_after"),
            ("q_scheduler_step_count_before", "q_scheduler_step_count_after"),
        ):
            if replay[after] != replay[before] + expected_delta:
                raise ValueError(f"replay {after} status relation disagrees")
        if expected_delta == 0 and (
            replay["q_scheduler_last_epoch_after"] != replay["q_scheduler_last_epoch_before"]
            or replay["q_lr_after"] != replay["q_lr_before"]
        ):
            raise ValueError("skipped/no-support Q scheduler or LR changed")
        if expected_delta == 1 and (
            replay["q_scheduler_last_epoch_after"]
            != max(replay["real_step_after"], 1)
            or replay["q_lr_after"] > replay["q_lr_before"]
        ):
            raise ValueError("stepped Q scheduler epoch or LR relation disagrees")
        if expected_delta == 1 and expected_q_initial_lr is not None:
            expected_epoch = max(replay["real_step_after"], 1)
            expected_multiplier = 1.0 - min(
                max(float(expected_epoch), 0.0)
                / float(expected_schedule_total_steps),
                1.0,
            )
            expected_lr_after = expected_q_initial_lr * expected_multiplier
            if replay["q_lr_after"] != expected_lr_after:
                raise ValueError("stepped Q learning rate disagrees with LambdaLR")
        if expected_amp_scale is not None:
            if q_row["amp_scale_before"] != expected_amp_scale:
                raise ValueError("Q AMP scale-before continuity disagrees")
            if replay["q_status"] == "stepped":
                if expected_amp_scaler_enabled:
                    expected_amp_growth_tracker += 1
                    if expected_amp_growth_tracker == 2000:
                        expected_amp_scale *= 2.0
                        expected_amp_growth_tracker = 0
            elif replay["q_status"] == "amp_skip":
                if not expected_amp_scaler_enabled:
                    raise ValueError("FP32 telemetry cannot contain an AMP skip")
                expected_amp_scale *= 0.5
                expected_amp_growth_tracker = 0
            if q_row["amp_scale_after"] != expected_amp_scale:
                raise ValueError("Q AMP scale transition disagrees with GradScaler")
        if replay["publication_count_after"] != transaction_id or replay["ack_count"] != 1:
            raise ValueError("publication/ack count disagrees with transaction")
        expected_terminal = transaction_id == transaction_count
        if replay["terminal"] is not expected_terminal or commit["terminal"] is not expected_terminal:
            raise ValueError("only the final telemetry transaction may be terminal")
        _validate_q_row(q_row, replay)
        huber_terms = []
        for row in td_block_rows:
            if row["q_source"] != "online" or row["split"] != "train":
                continue
            if row["abs_td_bin"] in ("0_0p5", "0p5_1"):
                huber_terms.append(0.5 * row["sum_td_sq"])
            else:
                huber_terms.append(row["sum_abs_td"] - 0.5 * row["count"])
        if not _close_enough(math.fsum(huber_terms), q_row["q_loss_sum"]):
            raise ValueError("online/train TD cells do not reconstruct beta-1 Huber sum")
        raw_td = b"".join(parsed[TD_FILENAME]["raw_rows"][index * 720 : (index + 1) * 720])
        raw_replay = parsed[REPLAY_FILENAME]["raw_rows"][index]
        raw_q = parsed[Q_FILENAME]["raw_rows"][index]
        expected_commit = {
            "td_first_data_row": 720 * index + 1,
            "td_data_row_count": 720,
            "td_block_byte_count": len(raw_td),
            "td_block_sha256": hashlib.sha256(raw_td).hexdigest(),
            "replay_first_data_row": transaction_id,
            "replay_data_row_count": 1,
            "replay_block_byte_count": len(raw_replay),
            "replay_block_sha256": hashlib.sha256(raw_replay).hexdigest(),
            "q_first_data_row": transaction_id,
            "q_data_row_count": 1,
            "q_block_byte_count": len(raw_q),
            "q_block_sha256": hashlib.sha256(raw_q).hexdigest(),
            "publication_count": transaction_id,
            "ack_count": 1,
            "actor_state_sha256": replay["actor_state_sha256"],
            "publication_history_sha256": replay["publication_history_sha256"],
        }
        for name, expected in expected_commit.items():
            if commit[name] != expected:
                raise ValueError(f"commit row disagrees on {name}")
        previous_replay = replay
        previous_q = q_row
    return transaction_count


def _validated_payload_records(
    payloads,
    *,
    expected_actor_unroll_len,
    expected_q_initial_lr=None,
    expected_schedule_total_steps=None,
    expected_amp_initial_scale=None,
):
    if type(payloads) is not dict or set(payloads) != set(SIDECAR_FILENAMES):
        raise ValueError("telemetry payload set is incomplete")
    parsed = {}
    records = {}
    for name in SIDECAR_FILENAMES:
        payload = payloads[name]
        if type(payload) is not bytes:
            raise TypeError(f"{name} payload must be exact bytes")
        parsed[name] = _parse_sidecar(name, payload)
        records[name] = _file_record(name, parsed[name])
    _validate_transaction_rows(
        parsed,
        expected_actor_unroll_len=expected_actor_unroll_len,
        expected_q_initial_lr=expected_q_initial_lr,
        expected_schedule_total_steps=expected_schedule_total_steps,
        expected_amp_initial_scale=expected_amp_initial_scale,
    )
    return records, parsed


def _validated_file_records(
    run_dir,
    *,
    expected_actor_unroll_len,
    expected_q_initial_lr=None,
    expected_schedule_total_steps=None,
    expected_amp_initial_scale=None,
    expected_identities=None,
):
    if expected_identities is not None and (
        type(expected_identities) is not dict
        or set(expected_identities) != set(SIDECAR_FILENAMES)
    ):
        raise ValueError("expected sidecar identities are incomplete")
    payloads = {}
    identities = set()
    for name in SIDECAR_FILENAMES:
        path = os.path.join(run_dir, name)
        expected_identity = (
            None if expected_identities is None else expected_identities[name]
        )
        payload, info = _stable_read(
            path,
            label=name,
            mode=0o400,
            expected_identity=expected_identity,
        )
        identity = (info.st_dev, info.st_ino)
        if identity in identities:
            raise RuntimeError("telemetry artifacts alias one inode")
        identities.add(identity)
        payloads[name] = payload
    return _validated_payload_records(
        payloads,
        expected_actor_unroll_len=expected_actor_unroll_len,
        expected_q_initial_lr=expected_q_initial_lr,
        expected_schedule_total_steps=expected_schedule_total_steps,
        expected_amp_initial_scale=expected_amp_initial_scale,
    )


def _no_duplicate_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate telemetry manifest key: {key}")
        result[key] = value
    return result


def _load_canonical_manifest(payload):
    if (
        not payload.endswith(b"\n")
        or b"\r" in payload
        or payload.startswith(b"\xef\xbb\xbf")
    ):
        raise ValueError("telemetry manifest is not canonical LF JSON")
    try:
        manifest = json.loads(
            payload[:-1].decode("utf-8"), object_pairs_hook=_no_duplicate_object
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("telemetry manifest is not strict UTF-8 JSON") from error
    if type(manifest) is not dict or set(manifest) != _MANIFEST_FIELDS:
        raise ValueError("telemetry manifest has the wrong top-level keyset")
    if canonical_json_bytes(manifest, trailing_lf=True) != payload:
        raise ValueError("telemetry manifest bytes are not canonical compact JSON")
    return manifest


def _validate_file_record(record, *, expected):
    if type(record) is not dict or set(record) != _FILE_RECORD_FIELDS:
        raise ValueError("telemetry manifest file record has the wrong keyset")
    for name in ("name", "sha256", "header_sha256"):
        if type(record[name]) is not str:
            raise TypeError(f"manifest file record {name} must be a built-in string")
    for name in ("size", "header_size", "column_count", "data_row_count"):
        _require_exact_int(f"manifest file record {name}", record[name])
    if record != expected:
        raise ValueError(f"manifest file record {record.get('name')!r} disagrees")


_EXPECTED_TERMINAL_STATE_FIELDS = frozenset({
    "voc_update_count",
    "ema_update_count",
    "projection_count",
    "adam_step_weight",
    "adam_step_bias",
    "q_scheduler_last_epoch",
    "q_scheduler_step_count",
    "q_optimizer_lr",
    "q_scheduler_last_lr",
    "amp_scale",
    "amp_growth_tracker",
    "amp_skip_count",
    "amp_consecutive_skips",
    "adam_m_after",
    "adam_v_after",
})


def _publication_history_sha256(history):
    try:
        payload = json.dumps(
            list(history),
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("expected publication history is not canonical JSON") from error
    return hashlib.sha256(payload).hexdigest()


def _validate_expected_publication_history(parsed, expected_history):
    replay_rows = parsed[REPLAY_FILENAME]["rows"]
    if type(expected_history) not in (tuple, list) or (
        len(expected_history) != len(replay_rows) + 1
    ):
        raise ValueError("expected publication history length disagrees")
    for index, replay in enumerate(replay_rows, start=1):
        event = expected_history[index]
        if not isinstance(event, Mapping):
            raise ValueError("expected publication history event is malformed")
        state_digest = event.get("state_sha256")
        _require_hash("expected publication state_sha256", state_digest)
        if replay["actor_state_sha256"] != state_digest:
            raise ValueError(
                f"replay actor state disagrees with publication history at {index}"
            )
        prefix_digest = _publication_history_sha256(expected_history[: index + 1])
        if replay["publication_history_sha256"] != prefix_digest:
            raise ValueError(
                f"replay history digest disagrees with publication prefix at {index}"
            )


def _validate_expected_terminal_state(
    parsed,
    expected_state,
    *,
    expected_amp_initial_scale=None,
):
    if type(expected_state) is not dict or set(expected_state) != set(
        _EXPECTED_TERMINAL_STATE_FIELDS
    ):
        raise ValueError("expected terminal state has the wrong keyset")
    replay_rows = parsed[REPLAY_FILENAME]["rows"]
    q_rows = parsed[Q_FILENAME]["rows"]
    final_replay = replay_rows[-1]
    final_q = q_rows[-1]
    integer_pairs = (
        ("voc_update_count", final_replay["voc_update_count_after"]),
        ("ema_update_count", final_replay["ema_update_count_after"]),
        ("projection_count", final_replay["projection_count_after"]),
        ("adam_step_weight", final_q["adam_step_after"]),
        ("adam_step_bias", final_q["adam_step_after"]),
        (
            "q_scheduler_last_epoch",
            final_replay["q_scheduler_last_epoch_after"],
        ),
        (
            "q_scheduler_step_count",
            final_replay["q_scheduler_step_count_after"],
        ),
    )
    for name, observed in integer_pairs:
        expected = _require_exact_int(f"expected terminal {name}", expected_state[name])
        if observed != expected:
            raise ValueError(f"terminal telemetry disagrees with checkpoint {name}")
    optimizer_lr = _finite_float(
        "expected terminal q_optimizer_lr", expected_state["q_optimizer_lr"]
    )
    scheduler_lr = _finite_float(
        "expected terminal q_scheduler_last_lr",
        expected_state["q_scheduler_last_lr"],
    )
    if (
        final_replay["q_lr_after"] != optimizer_lr
        or final_replay["q_lr_after"] != scheduler_lr
    ):
        raise ValueError("terminal Q LR disagrees with checkpoint optimizer/scheduler")

    checkpoint_amp_scale = _finite_float(
        "expected terminal amp_scale", expected_state["amp_scale"]
    )
    if final_q["amp_scale_after"] != checkpoint_amp_scale:
        raise ValueError("terminal AMP scale disagrees with checkpoint scaler")
    initial_scale = (
        q_rows[0]["amp_scale_before"]
        if expected_amp_initial_scale is None
        else _finite_float(
            "expected_amp_initial_scale", expected_amp_initial_scale
        )
    )
    scale = initial_scale
    growth_tracker = 0
    skip_count = 0
    consecutive_skips = 0
    checkpoint_tracker = expected_state["amp_growth_tracker"]
    fp32_no_scaler = checkpoint_tracker is None
    if (fp32_no_scaler and initial_scale != 1.0) or (
        not fp32_no_scaler and initial_scale != 256.0
    ):
        raise ValueError("terminal AMP mode disagrees with its frozen initial scale")
    for q_row in q_rows:
        if q_row["amp_scale_before"] != scale:
            raise ValueError("Q AMP scale history is not contiguous")
        if q_row["q_status"] == "stepped":
            if not fp32_no_scaler:
                growth_tracker += 1
                if growth_tracker == 2000:
                    scale *= 2.0
                    growth_tracker = 0
            consecutive_skips = 0
        elif q_row["q_status"] == "amp_skip":
            scale *= 0.5
            growth_tracker = 0
            skip_count += 1
            consecutive_skips += 1
        if q_row["amp_scale_after"] != scale:
            raise ValueError("Q AMP scale history disagrees with pinned GradScaler")
    if checkpoint_tracker is None:
        if initial_scale != 1.0 or scale != 1.0 or skip_count != 0:
            raise ValueError("FP32 telemetry has non-pristine AMP observations")
    else:
        checkpoint_tracker = _require_exact_int(
            "expected terminal amp_growth_tracker", checkpoint_tracker
        )
        if growth_tracker != checkpoint_tracker:
            raise ValueError("terminal AMP growth tracker disagrees with checkpoint")
    for name, observed in (
        ("amp_skip_count", skip_count),
        ("amp_consecutive_skips", consecutive_skips),
    ):
        expected = _require_exact_int(f"expected terminal {name}", expected_state[name])
        if observed != expected:
            raise ValueError(f"terminal telemetry disagrees with checkpoint {name}")

    update_count = final_replay["voc_update_count_after"]
    moment_m = expected_state["adam_m_after"]
    moment_v = expected_state["adam_v_after"]
    if update_count == 0:
        if moment_m is not None or moment_v is not None:
            raise ValueError("zero-update checkpoint unexpectedly has Adam moments")
        return
    if (
        type(moment_m) not in (tuple, list)
        or len(moment_m) != 2
        or type(moment_v) not in (tuple, list)
        or len(moment_v) != 2
    ):
        raise ValueError("checkpoint Adam moments must contain weight then bias")
    stepped_rows = [row for row in q_rows if row["q_status"] == "stepped"]
    if not stepped_rows:
        raise ValueError("positive terminal update count lacks a stepped Q row")
    terminal_step = stepped_rows[-1]
    for index, parameter in enumerate(("weight", "bias")):
        m_common, m_difference = row_l2_norms(moment_m[index])
        v_common, v_difference = row_means(moment_v[index])
        for field, expected in (
            (f"adam_m_after_{parameter}_common_l2", m_common),
            (f"adam_m_after_{parameter}_difference_l2", m_difference),
            (f"adam_v_after_{parameter}_common_mean", v_common),
            (f"adam_v_after_{parameter}_difference_mean", v_difference),
        ):
            if terminal_step[field] != expected:
                raise ValueError(
                    f"terminal Q diagnostic disagrees with checkpoint {field}"
                )


def validate_schema13_telemetry_manifest(
    run_dir,
    *,
    expected_xpid=None,
    expected_terminal_policy_version=None,
    expected_terminal_real_step=None,
    expected_actor_state_sha256=None,
    expected_publication_history_sha256=None,
    expected_stage_total_steps=None,
    expected_actor_unroll_len=None,
    expected_terminal_ack_count=1,
    expected_manifest_sha256=None,
    expected_manifest_size=None,
    expected_q_initial_lr=None,
    expected_schedule_total_steps=None,
    expected_amp_initial_scale=None,
    expected_publication_history=None,
    expected_terminal_state=None,
    expected_sidecar_identities=None,
    expected_legacy_log_identity=None,
):
    """Stable-read and fully validate the sealed schema-13 telemetry set."""

    root = os.path.abspath(os.fspath(run_dir))
    directory = os.stat(root, follow_symlinks=False)
    if not stat.S_ISDIR(directory.st_mode):
        raise ValueError("schema-13 telemetry root is not a directory")
    if expected_xpid is not None:
        _require_exact_string("expected_xpid", expected_xpid)
    for name, value in (
        ("expected_terminal_policy_version", expected_terminal_policy_version),
        ("expected_terminal_real_step", expected_terminal_real_step),
        ("expected_stage_total_steps", expected_stage_total_steps),
        ("expected_actor_unroll_len", expected_actor_unroll_len),
        ("expected_manifest_size", expected_manifest_size),
    ):
        if value is not None:
            _require_exact_int(name, value, minimum=1)
    expected_terminal_ack_count = _require_exact_int(
        "expected_terminal_ack_count", expected_terminal_ack_count, minimum=1
    )
    if expected_terminal_ack_count != 1:
        raise ValueError("schema-13 terminal ack count must be exactly one")
    if expected_actor_state_sha256 is not None:
        _require_hash("expected_actor_state_sha256", expected_actor_state_sha256)
    if expected_publication_history_sha256 is not None:
        _require_hash(
            "expected_publication_history_sha256",
            expected_publication_history_sha256,
        )
    if expected_manifest_sha256 is not None:
        _require_hash("expected_manifest_sha256", expected_manifest_sha256)
    if (expected_q_initial_lr is None) != (expected_schedule_total_steps is None):
        raise ValueError(
            "expected_q_initial_lr and expected_schedule_total_steps must be paired"
        )
    if expected_q_initial_lr is not None:
        expected_q_initial_lr = _finite_float(
            "expected_q_initial_lr", expected_q_initial_lr
        )
        if expected_q_initial_lr <= 0.0:
            raise ValueError("expected_q_initial_lr must be positive")
        expected_schedule_total_steps = _require_exact_int(
            "expected_schedule_total_steps",
            expected_schedule_total_steps,
            minimum=1,
        )
    if expected_amp_initial_scale is not None:
        expected_amp_initial_scale = _finite_float(
            "expected_amp_initial_scale", expected_amp_initial_scale
        )
        if expected_amp_initial_scale not in (1.0, 256.0):
            raise ValueError(
                "expected_amp_initial_scale must be frozen FP32 1 or FP16 256"
            )
    expected_telemetry_names = set(SIDECAR_FILENAMES) | {MANIFEST_FILENAME}
    reserved_extras = {
        entry
        for entry in os.listdir(root)
        if (
            entry.startswith("voc_td_")
            or entry.startswith("voc_replay_")
            or entry.startswith("voc_q_")
            or entry.startswith("voc_telemetry_")
        )
        and entry not in expected_telemetry_names
    }
    if reserved_extras:
        raise ValueError("schema-13 run contains an extra telemetry artifact")
    temp_path = os.path.join(root, "." + MANIFEST_FILENAME + ".tmp")
    if os.path.lexists(temp_path):
        raise ValueError("telemetry manifest temporary name remains")
    manifest_path = os.path.join(root, MANIFEST_FILENAME)
    manifest_payload, _ = _stable_read(
        manifest_path, label=MANIFEST_FILENAME, mode=0o400
    )
    manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    if expected_manifest_sha256 is not None and manifest_sha256 != expected_manifest_sha256:
        raise ValueError("telemetry manifest digest disagrees with completion evidence")
    if expected_manifest_size is not None and len(manifest_payload) != expected_manifest_size:
        raise ValueError("telemetry manifest size disagrees with completion evidence")
    manifest = _load_canonical_manifest(manifest_payload)
    for name, expected in (
        ("telemetry_schema_version", VOC_TELEMETRY_SCHEMA_VERSION),
        ("gate_schema", VOC_GATE_SCHEMA_VERSION),
        ("status", "sealed"),
        ("fresh", True),
    ):
        if type(manifest[name]) is not type(expected) or manifest[name] != expected:
            raise ValueError(f"telemetry manifest requires {name}={expected!r}")
    _require_exact_string("manifest xpid", manifest["xpid"])
    for name in (
        "transaction_count",
        "terminal_policy_version",
        "terminal_real_step",
        "terminal_publication_count",
        "terminal_ack_count",
    ):
        _require_exact_int(f"manifest {name}", manifest[name], minimum=1)
    _require_hash("manifest actor_state_sha256", manifest["actor_state_sha256"])
    _require_hash(
        "manifest publication_history_sha256",
        manifest["publication_history_sha256"],
    )
    transaction_count = manifest["transaction_count"]
    if (
        manifest["terminal_policy_version"] != transaction_count
        or manifest["terminal_publication_count"] != transaction_count
        or manifest["terminal_ack_count"] != 1
    ):
        raise ValueError("manifest terminal counts disagree")
    if expected_xpid is not None and manifest["xpid"] != expected_xpid:
        raise ValueError("manifest xpid disagrees")
    for field, expected in (
        ("terminal_policy_version", expected_terminal_policy_version),
        ("terminal_real_step", expected_terminal_real_step),
        ("actor_state_sha256", expected_actor_state_sha256),
        ("publication_history_sha256", expected_publication_history_sha256),
    ):
        if expected is not None and manifest[field] != expected:
            raise ValueError(f"manifest {field} disagrees with expected evidence")
    if expected_stage_total_steps is not None and manifest["terminal_real_step"] < expected_stage_total_steps:
        raise ValueError("manifest terminal real step is below the stage total")
    records, parsed = _validated_file_records(
        root,
        expected_actor_unroll_len=expected_actor_unroll_len,
        expected_q_initial_lr=expected_q_initial_lr,
        expected_schedule_total_steps=expected_schedule_total_steps,
        expected_amp_initial_scale=expected_amp_initial_scale,
        expected_identities=expected_sidecar_identities,
    )
    if len(parsed[COMMIT_FILENAME]["rows"]) != transaction_count:
        raise ValueError("manifest transaction count disagrees with commit rows")
    artifacts = manifest["artifacts"]
    if type(artifacts) is not list or len(artifacts) != 4:
        raise ValueError("manifest artifacts must be an ordered four-record list")
    for record, name in zip(artifacts, SIDECAR_FILENAMES):
        _validate_file_record(record, expected=records[name])
    legacy_expected = _legacy_log_record(
        os.path.join(root, LEGACY_LOG_FILENAME),
        transaction_count,
        expected_replay_rows=parsed[REPLAY_FILENAME]["rows"],
        expected_q_rows=parsed[Q_FILENAME]["rows"],
        expected_identity=expected_legacy_log_identity,
    )
    _validate_file_record(manifest["legacy_actor_log"], expected=legacy_expected)
    last_commit = manifest["last_commit"]
    if type(last_commit) is not dict or set(last_commit) != _LAST_COMMIT_FIELDS:
        raise ValueError("manifest last_commit has the wrong keyset")
    for name in ("data_row", "transaction_id"):
        _require_exact_int(f"last_commit {name}", last_commit[name], minimum=1)
    for name in ("sha256", "actor_state_sha256", "publication_history_sha256"):
        _require_hash(f"last_commit {name}", last_commit[name])
    commit_rows = parsed[COMMIT_FILENAME]["rows"]
    final_commit = commit_rows[-1]
    final_commit_bytes = parsed[COMMIT_FILENAME]["raw_rows"][-1]
    expected_last_commit = {
        "data_row": transaction_count,
        "transaction_id": transaction_count,
        "sha256": hashlib.sha256(final_commit_bytes).hexdigest(),
        "actor_state_sha256": final_commit["actor_state_sha256"],
        "publication_history_sha256": final_commit[
            "publication_history_sha256"
        ],
    }
    if last_commit != expected_last_commit:
        raise ValueError("manifest last_commit disagrees with terminal commit row")
    final_replay = parsed[REPLAY_FILENAME]["rows"][-1]
    if (
        final_commit["terminal"] is not True
        or final_commit["published_policy_version"] != transaction_count
        or final_replay["real_step_after"] != manifest["terminal_real_step"]
        or final_commit["actor_state_sha256"] != manifest["actor_state_sha256"]
        or final_commit["publication_history_sha256"]
        != manifest["publication_history_sha256"]
    ):
        raise ValueError("manifest terminal identity disagrees with sidecars")
    if expected_publication_history is not None:
        _validate_expected_publication_history(parsed, expected_publication_history)
    if expected_terminal_state is not None:
        _validate_expected_terminal_state(
            parsed,
            expected_terminal_state,
            expected_amp_initial_scale=expected_amp_initial_scale,
        )
    evidence = {
        "telemetry_schema_version": VOC_TELEMETRY_SCHEMA_VERSION,
        "gate_schema": VOC_GATE_SCHEMA_VERSION,
        "manifest_name": MANIFEST_FILENAME,
        "manifest_sha256": manifest_sha256,
        "manifest_size": len(manifest_payload),
        "transaction_count": transaction_count,
        "terminal_policy_version": manifest["terminal_policy_version"],
        "terminal_real_step": manifest["terminal_real_step"],
        "actor_state_sha256": manifest["actor_state_sha256"],
        "publication_history_sha256": manifest[
            "publication_history_sha256"
        ],
    }
    if set(evidence) != _EVIDENCE_FIELDS:
        raise AssertionError("schema-13 telemetry evidence keyset drift")
    return evidence


__all__ = (
    "VOC_TELEMETRY_SCHEMA_VERSION",
    "VOC_GATE_SCHEMA_VERSION",
    "TD_FILENAME",
    "REPLAY_FILENAME",
    "Q_FILENAME",
    "COMMIT_FILENAME",
    "MANIFEST_FILENAME",
    "SIDECAR_FILENAMES",
    "TD_FIELDS",
    "REPLAY_FIELDS",
    "Q_FIELDS",
    "COMMIT_FIELDS",
    "TD_HEADER",
    "REPLAY_HEADER",
    "Q_HEADER",
    "COMMIT_HEADER",
    "TD_CATEGORY_ORDER",
    "Q_DIAGNOSTIC_FIELDS",
    "canonical_float",
    "parse_canonical_float",
    "canonical_json_bytes",
    "encode_csv_row",
    "l2_norm",
    "row_l2_norms",
    "row_means",
    "build_stepped_q_diagnostics",
    "build_td_cell_rows",
    "build_replay_row",
    "build_q_transaction_row",
    "Schema13TelemetryWriter",
    "validate_schema13_telemetry_manifest",
)
