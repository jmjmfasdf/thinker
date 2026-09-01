import time
import timeit
import os
import hashlib
import stat
import numpy as np
import collections
from collections.abc import Mapping
import random
import copy
import traceback
import ray
import torch
import torch.nn.functional as F
from torch.optim import adam as _torch_adam
from torch.amp import grad_scaler as _torch_grad_scaler
from torch.cuda.amp import GradScaler, autocast

from thinker.core.vtrace import compute_v_trace
from thinker.core.file_writer import FileWriter
from thinker.core.module import guassian_kl_div
from thinker.actor_net import (
    ActorNet,
    ILLEGAL_CONTROL_LOGIT,
    atanh,
    compute_discrete_log_prob,
    compute_dynamic_control_entropy,
    compute_dynamic_control_log_probs,
)
from thinker.dynamic_imitation import (
    DynamicImitationRunner,
    empty_behavioral_action_metrics,
    imitation_checkpoint_state,
    resolve_noop_action_index,
    scale_imitation_for_online_rows,
)
import thinker.util as util
from thinker.buffer import RetBuffer


ActorGradientStepResult = collections.namedtuple(
    "ActorGradientStepResult",
    (
        "total_norm",
        "optimizer_stepped",
        "amp_scale_before",
        "amp_scale_after",
        "nonfinite_gradient_names",
    ),
)


_VOC_ORTHOCD_TORCH_VERSION = "2.13.0+cu130"
_VOC_ORTHOCD_ADAM_SOURCE_SHA256 = (
    "bde360b0bb9b7869f1cec04a3b41a90b8eabb84a613787d97b88d87f2f3ae1ec"
)
_VOC_ORTHOCD_GRAD_SCALER_SOURCE_SHA256 = (
    "97c411da028daaf6a6ed15d06b9b20c017404846db68203be1a586e276e44039"
)
_VOC_ORTHOCD_SCALE_BITS = 0x3F3504F3


def _voc_orthocd_source_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _VoCOrthoCDAdam(torch.optim.Adam):
    """Schema-11/12 Adam adapter with raw parameters and m/d moment rows."""

    @staticmethod
    def _orthocd_scale(device):
        bits = torch.tensor(
            [_VOC_ORTHOCD_SCALE_BITS], dtype=torch.int32, device=device
        )
        return bits.view(torch.float32)[0]

    @staticmethod
    def _validate_tensor(
        tensor,
        reference,
        label,
        *,
        require_finite=True,
    ):
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(f"{label} must be a tensor")
        if tensor.shape != reference.shape:
            raise RuntimeError(
                f"{label} shape {tuple(tensor.shape)} does not match "
                f"{tuple(reference.shape)}"
            )
        if tensor.dtype != reference.dtype:
            raise RuntimeError(
                f"{label} dtype {tensor.dtype} does not match "
                f"{reference.dtype}"
            )
        if tensor.device != reference.device:
            raise RuntimeError(
                f"{label} device {tensor.device} does not match "
                f"{reference.device}"
            )
        if tensor.layout != torch.strided:
            raise RuntimeError(f"{label} must use dense strided layout")
        if tensor.stride() != reference.stride():
            raise RuntimeError(
                f"{label} stride {tensor.stride()} does not match "
                f"{reference.stride()}"
            )
        if require_finite and not torch.isfinite(tensor).all().item():
            raise FloatingPointError(f"{label} must be finite")

    @staticmethod
    def _tensor_bytes_equal(left, right):
        if (
            left.shape != right.shape
            or left.dtype != right.dtype
            or left.device != right.device
            or left.layout != right.layout
        ):
            return False
        left_bytes = left.detach().contiguous().reshape(-1).view(torch.uint8)
        right_bytes = right.detach().contiguous().reshape(-1).view(torch.uint8)
        return torch.equal(left_bytes, right_bytes)

    @classmethod
    def _transform_raw_gradient(cls, raw_gradient, label):
        raw_c = raw_gradient[0].detach().clone(
            memory_format=torch.preserve_format
        )
        raw_s = raw_gradient[1].detach().clone(
            memory_format=torch.preserve_format
        )
        scale = cls._orthocd_scale(raw_gradient.device)
        gradient_m = torch.mul(scale, torch.add(raw_c, raw_s))
        gradient_d = torch.mul(scale, torch.sub(raw_c, raw_s))
        transformed = torch.stack((gradient_m, gradient_d), dim=0)
        cls._validate_tensor(
            transformed,
            raw_gradient,
            f"{label} m/d gradient",
        )
        return transformed

    @classmethod
    def _inverse_map_delta(cls, coordinate_delta, raw_parameter, label):
        delta_m = coordinate_delta[0].detach().clone(
            memory_format=torch.preserve_format
        )
        delta_d = coordinate_delta[1].detach().clone(
            memory_format=torch.preserve_format
        )
        scale = cls._orthocd_scale(coordinate_delta.device)
        delta_c = torch.mul(scale, torch.add(delta_m, delta_d))
        delta_s = torch.mul(scale, torch.sub(delta_m, delta_d))
        mapped_delta = torch.stack((delta_c, delta_s), dim=0)
        cls._validate_tensor(
            mapped_delta,
            raw_parameter,
            f"{label} mapped raw delta",
        )
        raw_c = raw_parameter[0].detach().clone(
            memory_format=torch.preserve_format
        )
        raw_s = raw_parameter[1].detach().clone(
            memory_format=torch.preserve_format
        )
        candidate = torch.stack(
            (torch.add(raw_c, delta_c), torch.add(raw_s, delta_s)), dim=0
        )
        cls._validate_tensor(candidate, raw_parameter, f"{label} candidate")
        return mapped_delta, candidate

    @staticmethod
    def _commit_injection_point(_label, _index):
        """No-op test seam; production commits are otherwise uninterrupted."""

    @staticmethod
    def _rollback_injection_point(_label, _index):
        """No-op test seam used to prove fatal rollback-failure handling."""

    def _require_runtime_and_group_contract(self):
        if str(torch.__version__) != _VOC_ORTHOCD_TORCH_VERSION:
            raise RuntimeError(
                "schema-11 VoC Adam requires torch "
                f"{_VOC_ORTHOCD_TORCH_VERSION}; got {torch.__version__}"
            )
        source_contract = (
            (
                "torch.optim.adam",
                _torch_adam.__file__,
                _VOC_ORTHOCD_ADAM_SOURCE_SHA256,
            ),
            (
                "torch.amp.grad_scaler",
                _torch_grad_scaler.__file__,
                _VOC_ORTHOCD_GRAD_SCALER_SOURCE_SHA256,
            ),
        )
        for label, path, expected_sha256 in source_contract:
            if not isinstance(path, str):
                raise RuntimeError(f"schema-11 cannot resolve {label} source")
            actual_sha256 = _voc_orthocd_source_sha256(path)
            if actual_sha256 != expected_sha256:
                raise RuntimeError(
                    f"schema-11 {label} source hash mismatch: "
                    f"{actual_sha256} != {expected_sha256}"
                )
        if len(self.param_groups) != 1:
            raise RuntimeError("schema-11 VoC Adam requires exactly one group")
        group = self.param_groups[0]
        parameters = list(group.get("params", ()))
        if (
            type(group.get("params")) is not list
            or len(parameters) != 2
            or parameters[0] is parameters[1]
        ):
            raise RuntimeError(
                "schema-11 VoC Adam requires raw weight then bias"
            )
        weight, bias = parameters
        if not isinstance(weight, torch.nn.Parameter) or not isinstance(
            bias, torch.nn.Parameter
        ):
            raise RuntimeError(
                "schema-11 VoC Adam requires raw weight then bias parameters"
            )
        if weight.ndim != 2 or weight.shape[0] != 2:
            raise RuntimeError(
                "schema-11 VoC Adam weight must have shape [2, D]"
            )
        if bias.ndim != 1 or bias.shape != (2,):
            raise RuntimeError(
                "schema-11 VoC Adam bias must have shape [2]"
            )
        for index, parameter in enumerate(parameters):
            label = ("weight", "bias")[index]
            if (
                not isinstance(parameter, torch.nn.Parameter)
                or parameter.dtype != torch.float32
                or parameter.layout != torch.strided
                or parameter.is_complex()
                or not parameter.requires_grad
                or not parameter.is_contiguous()
            ):
                raise RuntimeError(
                    f"schema-11 VoC Adam {label} must be dense real FP32"
                )
            gradient = parameter.grad
            if gradient is None:
                raise RuntimeError(
                    f"schema-11 VoC Adam {label} gradient is missing"
                )
            self._validate_tensor(
                gradient,
                parameter,
                f"schema-11 raw {label} gradient",
                require_finite=False,
            )
            if gradient.requires_grad:
                raise RuntimeError(
                    f"schema-11 raw {label} gradient must be detached"
                )
        if bias.device != weight.device:
            raise RuntimeError(
                "schema-11 VoC Adam weight and bias devices must match"
            )
        required_values = {
            "betas": (0.9, 0.999),
            "weight_decay": 0,
            "amsgrad": False,
            "maximize": False,
            "foreach": None,
            "capturable": False,
            "differentiable": False,
            "fused": None,
            "decoupled_weight_decay": False,
        }
        for key, expected in required_values.items():
            value = group.get(key)
            exact = (
                value is expected
                if expected is None or isinstance(expected, bool)
                else type(value) is type(expected) and value == expected
            )
            if key not in group or not exact:
                raise RuntimeError(
                    f"schema-11 VoC Adam requires {key}={expected!r}"
                )
        for key, positive in (("lr", False), ("eps", True)):
            value = group.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or (value <= 0.0 if positive else value < 0.0)
            ):
                raise RuntimeError(
                    f"schema-11 VoC Adam requires finite valid {key}"
                )
        return group, parameters

    @staticmethod
    def _stage_parameter_state(optimizer, parameter, label):
        had_entry = parameter in optimizer.state
        live_state = optimizer.state.get(parameter)
        if live_state is None:
            live_state = {}
        if not isinstance(live_state, dict):
            raise RuntimeError(f"schema-11 {label} Adam state must be a dict")
        if live_state:
            if set(live_state) != {"step", "exp_avg", "exp_avg_sq"}:
                raise RuntimeError(
                    f"schema-11 {label} Adam state keys are invalid"
                )
            step = live_state["step"]
            if (
                not isinstance(step, torch.Tensor)
                or step.shape != torch.Size([])
                or step.dtype != torch.float32
                or step.device.type != "cpu"
                or not torch.isfinite(step).item()
                or step.requires_grad
                or step.item() < 0.0
                or not float(step.item()).is_integer()
            ):
                raise RuntimeError(
                    f"schema-11 {label} Adam step must be finite CPU FP32"
                )
            exp_avg = live_state["exp_avg"]
            exp_avg_sq = live_state["exp_avg_sq"]
            _VoCOrthoCDAdam._validate_tensor(
                exp_avg,
                parameter,
                f"schema-11 {label} exp_avg",
                require_finite=False,
            )
            _VoCOrthoCDAdam._validate_tensor(
                exp_avg_sq,
                parameter,
                f"schema-11 {label} exp_avg_sq",
                require_finite=False,
            )
            if exp_avg.requires_grad or exp_avg_sq.requires_grad:
                raise RuntimeError(
                    f"schema-11 {label} Adam moments must be detached"
                )
            candidate_step = step.detach().clone()
            candidate_exp_avg = exp_avg.detach().clone(
                memory_format=torch.preserve_format
            )
            candidate_exp_avg_sq = exp_avg_sq.detach().clone(
                memory_format=torch.preserve_format
            )
        else:
            candidate_step = torch.tensor(
                0.0, dtype=torch.float32, device="cpu"
            )
            candidate_exp_avg = torch.zeros_like(
                parameter, memory_format=torch.preserve_format
            )
            candidate_exp_avg_sq = torch.zeros_like(
                parameter, memory_format=torch.preserve_format
            )
        expected_step = candidate_step.detach().clone()
        expected_step.add_(1.0)
        live_refs = dict(live_state)
        live_clones = {
            key: value.detach().clone(memory_format=torch.preserve_format)
            for key, value in live_refs.items()
        }
        return {
            "had_entry": had_entry,
            "live_state": live_state,
            "live_refs": live_refs,
            "live_clones": live_clones,
            "step": candidate_step,
            "expected_step": expected_step,
            "exp_avg": candidate_exp_avg,
            "exp_avg_sq": candidate_exp_avg_sq,
        }

    def _rollback_exact(self, parameters, raw_clones, staged_states):
        for index, (parameter, raw_clone) in enumerate(
            zip(parameters, raw_clones)
        ):
            parameter.copy_(raw_clone)
            self._rollback_injection_point("parameter", index)
        for index, (parameter, staged) in enumerate(
            zip(parameters, staged_states)
        ):
            if not staged["had_entry"]:
                self.state.pop(parameter, None)
                self._rollback_injection_point("state", index)
                continue
            live_state = staged["live_state"]
            live_state.clear()
            for key, live_value in staged["live_refs"].items():
                live_value.copy_(staged["live_clones"][key])
                live_state[key] = live_value
            self.state[parameter] = live_state
            self._rollback_injection_point("state", index)

    def _rollback_matches(self, parameters, raw_clones, staged_states):
        for parameter, raw_clone in zip(parameters, raw_clones):
            if not self._tensor_bytes_equal(parameter, raw_clone):
                return False
        for parameter, staged in zip(parameters, staged_states):
            if not staged["had_entry"]:
                if parameter in self.state:
                    return False
                continue
            if self.state.get(parameter) is not staged["live_state"]:
                return False
            live_state = self.state[parameter]
            if set(live_state) != set(staged["live_refs"]):
                return False
            for key, live_value in staged["live_refs"].items():
                if live_state[key] is not live_value:
                    return False
                if not self._tensor_bytes_equal(
                    live_value, staged["live_clones"][key]
                ):
                    return False
        return True

    def _commit_candidates(
        self,
        parameters,
        raw_candidates,
        staged_states,
    ):
        for index, (parameter, candidate) in enumerate(
            zip(parameters, raw_candidates)
        ):
            parameter.copy_(candidate)
            self._commit_injection_point("parameter", index)
        for index, (parameter, staged) in enumerate(
            zip(parameters, staged_states)
        ):
            candidate_state = {
                "step": staged["step"],
                "exp_avg": staged["exp_avg"],
                "exp_avg_sq": staged["exp_avg_sq"],
            }
            if staged["live_state"]:
                live_state = staged["live_state"]
                for key in ("step", "exp_avg", "exp_avg_sq"):
                    live_state[key].copy_(candidate_state[key])
                    self._commit_injection_point(f"state.{key}", index)
            else:
                live_state = staged["live_state"]
                for key in ("step", "exp_avg", "exp_avg_sq"):
                    live_state[key] = candidate_state[key]
                    self._commit_injection_point(f"state.{key}", index)
                self.state[parameter] = live_state

    def _build_schema13_telemetry_candidate(
        self,
        *,
        telemetry_before,
        staged_states,
        md_gradients,
        scratch_parameters,
        mapped_deltas,
    ):
        """Detach all schema-13 evidence before mutating live Adam state."""

        return {
            **telemetry_before,
            "adam_step_after": tuple(
                state["step"].detach().clone() for state in staged_states
            ),
            "md_postclip": tuple(
                value.detach().clone(memory_format=torch.preserve_format)
                for value in md_gradients
            ),
            "adam_m_after": tuple(
                state["exp_avg"].detach().clone(
                    memory_format=torch.preserve_format
                )
                for state in staged_states
            ),
            "adam_v_after": tuple(
                state["exp_avg_sq"].detach().clone(
                    memory_format=torch.preserve_format
                )
                for state in staged_states
            ),
            "coordinate_delta": tuple(
                value.detach().clone(memory_format=torch.preserve_format)
                for value in scratch_parameters
            ),
            "mapped_delta": tuple(
                value.detach().clone(memory_format=torch.preserve_format)
                for value in mapped_deltas
            ),
        }

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise RuntimeError("schema-11 VoC Adam does not accept a closure")
        group, parameters = self._require_runtime_and_group_contract()
        raw_clones = [
            parameter.detach().clone(memory_format=torch.preserve_format)
            for parameter in parameters
        ]
        staged_states = [
            self._stage_parameter_state(self, parameter, label)
            for parameter, label in zip(parameters, ("weight", "bias"))
        ]
        telemetry_capture = (
            getattr(self, "_schema13_telemetry_capture", False) is True
        )
        telemetry_before = None
        if telemetry_capture:
            telemetry_before = {
                "adam_step_before": tuple(
                    state["step"].detach().clone() for state in staged_states
                ),
                "adam_m_before": tuple(
                    state["exp_avg"].detach().clone(
                        memory_format=torch.preserve_format
                    )
                    for state in staged_states
                ),
                "adam_v_before": tuple(
                    state["exp_avg_sq"].detach().clone(
                        memory_format=torch.preserve_format
                    )
                    for state in staged_states
                ),
            }
        md_gradients = [
            self._transform_raw_gradient(parameter.grad, label)
            for parameter, label in zip(parameters, ("weight", "bias"))
        ]
        scratch_parameters = [
            torch.zeros_like(
                parameter, memory_format=torch.preserve_format
            ).detach()
            for parameter in parameters
        ]
        for scratch, parameter, label in zip(
            scratch_parameters, parameters, ("weight", "bias")
        ):
            self._validate_tensor(
                scratch, parameter, f"schema-11 {label} scratch"
            )
            if (
                torch.count_nonzero(scratch).item() != 0
                or torch.signbit(scratch).any().item()
                or scratch.requires_grad
            ):
                raise RuntimeError(
                    f"schema-11 {label} scratch must be detached positive zero"
                )
        candidate_exp_avgs = [state["exp_avg"] for state in staged_states]
        candidate_exp_avg_sqs = [
            state["exp_avg_sq"] for state in staged_states
        ]
        candidate_steps = [state["step"] for state in staged_states]
        _torch_adam.adam(
            scratch_parameters,
            md_gradients,
            candidate_exp_avgs,
            candidate_exp_avg_sqs,
            [],
            candidate_steps,
            foreach=True,
            fused=False,
            capturable=False,
            differentiable=False,
            decoupled_weight_decay=False,
            grad_scale=None,
            found_inf=None,
            has_complex=False,
            amsgrad=False,
            maximize=False,
            weight_decay=0,
            beta1=0.9,
            beta2=0.999,
            lr=group["lr"],
            eps=group["eps"],
        )
        raw_candidates = []
        mapped_deltas = []
        for scratch, raw_clone, label in zip(
            scratch_parameters, raw_clones, ("weight", "bias")
        ):
            self._validate_tensor(
                scratch, raw_clone, f"schema-11 {label} coordinate delta"
            )
            mapped_delta, raw_candidate = self._inverse_map_delta(
                scratch, raw_clone, label
            )
            mapped_deltas.append(mapped_delta)
            raw_candidates.append(raw_candidate)
        for index, (parameter, staged) in enumerate(
            zip(parameters, staged_states)
        ):
            label = ("weight", "bias")[index]
            self._validate_tensor(
                staged["exp_avg"], parameter, f"schema-11 {label} exp_avg"
            )
            self._validate_tensor(
                staged["exp_avg_sq"],
                parameter,
                f"schema-11 {label} exp_avg_sq",
            )
            step = staged["step"]
            if (
                step.shape != torch.Size([])
                or step.dtype != torch.float32
                or step.device.type != "cpu"
                or not torch.isfinite(step).item()
                or step.requires_grad
                or not self._tensor_bytes_equal(
                    step, staged["expected_step"]
                )
            ):
                raise RuntimeError(
                    f"schema-11 {label} candidate step is invalid"
                )
        telemetry_candidate = None
        if telemetry_capture:
            telemetry_candidate = self._build_schema13_telemetry_candidate(
                telemetry_before=telemetry_before,
                staged_states=staged_states,
                md_gradients=md_gradients,
                scratch_parameters=scratch_parameters,
                mapped_deltas=mapped_deltas,
            )
        try:
            self._commit_candidates(
                parameters, raw_candidates, staged_states
            )
        except BaseException as commit_error:
            try:
                self._rollback_exact(parameters, raw_clones, staged_states)
                if not self._rollback_matches(
                    parameters, raw_clones, staged_states
                ):
                    raise RuntimeError(
                        "schema-11 VoC Adam rollback verification failed"
                    )
            except BaseException as rollback_error:
                raise RuntimeError(
                    "schema-11 VoC Adam commit rollback failed"
                ) from rollback_error
            raise commit_error
        if telemetry_capture:
            self._schema13_telemetry_candidate = telemetry_candidate
        return None


DynamicVoCTarget = collections.namedtuple(
    "DynamicVoCTarget", ("task", "think", "net")
)
DynamicVoCLossResult = collections.namedtuple(
    "DynamicVoCLossResult",
    (
        "q_loss",
        "gate_pg_loss",
        "q_values",
        "target",
        "selected_q",
        "td_error",
        "delta_q",
        "continue_probability",
        "behavior_continue_probability",
        "selected_advantage",
        "gate_rho",
        "gate_action",
        "greedy_action",
        "valid",
        "q_train_valid",
    ),
)
DynamicVoCSoftGateLoss = collections.namedtuple(
    "DynamicVoCSoftGateLoss",
    (
        "loss",
        "bce",
        "student_continue_probability",
        "teacher_continue_probability",
        "confidence",
        "objective_weight",
        "delta_q",
        "directed_logit_gradient",
        "wrong_continue_saturation",
        "wrong_stop_saturation",
        "valid",
    ),
)
DynamicVoCGateParameterAlignment = collections.namedtuple(
    "DynamicVoCGateParameterAlignment",
    (
        "loss",
        "target_weight",
        "target_bias",
        "gate_weight_norm",
        "target_weight_norm",
        "weight_error_norm",
        "gate_bias",
        "bias_error_abs",
        "gate_parameter_norm",
        "target_parameter_norm",
        "parameter_error_norm",
        "relative_parameter_error",
        "relative_error_defined",
        "cosine",
        "cosine_defined",
    ),
)
DynamicVoCGateExactProjection = collections.namedtuple(
    "DynamicVoCGateExactProjection",
    (
        "target_weight",
        "target_bias",
        "pre_projection_error_norm",
        "post_projection_error_norm",
    ),
)


def compute_dynamic_voc_gate_parameter_alignment_loss(
    *,
    gate_weight,
    gate_bias,
    ema_q_weight,
    ema_q_bias,
    q_temperature,
    policy_temperature,
):
    """Align the scalar behavior head to the exact frozen soft-Q map.

    Both heads consume the same detached feature vector.  With Q order
    ``[CONTINUE, STOP]``, matching the behavior probability therefore has the
    exact parameter-space solution

    ``theta_gate = (T_policy / T_Q) * (theta_C - theta_S)``.

    The returned half-squared Euclidean loss is deliberately a sum over the
    one scalar head, not a feature-dimension mean.  The EMA/Q side is detached
    and all arithmetic is FP32, so only the dedicated gate can receive a
    gradient.
    """

    tensors = {
        "gate weight": gate_weight,
        "gate bias": gate_bias,
        "EMA Q weight": ema_q_weight,
        "EMA Q bias": ema_q_bias,
    }
    for name, value in tensors.items():
        if not torch.is_tensor(value) or not torch.is_floating_point(value):
            raise TypeError(f"VoC {name} must be a floating tensor")
    if gate_weight.ndim != 2 or gate_weight.shape[0] != 1:
        raise ValueError("VoC gate weight must have shape [1, feature_dim]")
    if tuple(gate_bias.shape) != (1,):
        raise ValueError("VoC gate bias must have shape [1]")
    if tuple(ema_q_weight.shape) != (2, gate_weight.shape[1]):
        raise ValueError(
            "VoC EMA Q weight must have shape [2, feature_dim]"
        )
    if tuple(ema_q_bias.shape) != (2,):
        raise ValueError("VoC EMA Q bias must have shape [2]")
    devices = {value.device for value in tensors.values()}
    if len(devices) != 1:
        raise ValueError("VoC gate and EMA Q parameters must share a device")
    for name, value in (
        ("q_temperature", q_temperature),
        ("policy_temperature", policy_temperature),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.number))
            or not np.isfinite(value)
            or float(value) <= 0.0
        ):
            raise ValueError(f"VoC gate {name} must be finite and positive")

    with autocast(enabled=False):
        gate_weight_fp32 = gate_weight.float()
        gate_bias_fp32 = gate_bias.float()
        frozen_q_weight = ema_q_weight.detach().float()
        frozen_q_bias = ema_q_bias.detach().float()
        scale = float(policy_temperature) / float(q_temperature)
        target_weight = scale * (
            frozen_q_weight[0:1] - frozen_q_weight[1:2]
        )
        target_bias = scale * (frozen_q_bias[0:1] - frozen_q_bias[1:2])
        weight_error = gate_weight_fp32 - target_weight
        bias_error = gate_bias_fp32 - target_bias
        weight_error_sq = torch.sum(weight_error.square())
        bias_error_sq = torch.sum(bias_error.square())
        loss = 0.5 * (weight_error_sq + bias_error_sq)

    for name, value in (
        ("gate weight", gate_weight_fp32),
        ("gate bias", gate_bias_fp32),
        ("EMA-derived target weight", target_weight),
        ("EMA-derived target bias", target_bias),
        ("parameter-alignment loss", loss),
    ):
        _require_finite_tensor(f"VoC {name}", value)

    # Diagnostics are not part of the backward graph.  Explicit defined flags
    # preserve the exact fresh zero/zero tie without manufacturing a cosine or
    # relative error through an arbitrary epsilon denominator.
    with torch.no_grad():
        gate_weight_norm = torch.linalg.vector_norm(gate_weight_fp32)
        target_weight_norm = torch.linalg.vector_norm(target_weight)
        weight_error_norm = weight_error_sq.sqrt()
        gate_bias_value = gate_bias_fp32.reshape(())
        bias_error_abs = bias_error.abs().reshape(())
        gate_parameter_sq = (
            torch.sum(gate_weight_fp32.square())
            + torch.sum(gate_bias_fp32.square())
        )
        target_parameter_sq = (
            torch.sum(target_weight.square())
            + torch.sum(target_bias.square())
        )
        gate_parameter_norm = gate_parameter_sq.sqrt()
        target_parameter_norm = target_parameter_sq.sqrt()
        parameter_error_norm = (weight_error_sq + bias_error_sq).sqrt()
        relative_defined = target_parameter_norm > 0.0
        relative_parameter_error = torch.where(
            relative_defined,
            parameter_error_norm
            / target_parameter_norm.clamp_min(torch.finfo(torch.float32).tiny),
            torch.zeros_like(parameter_error_norm),
        )
        cosine_denominator = gate_parameter_norm * target_parameter_norm
        cosine_defined = cosine_denominator > 0.0
        parameter_dot = (
            torch.sum(gate_weight_fp32 * target_weight)
            + torch.sum(gate_bias_fp32 * target_bias)
        )
        cosine = torch.where(
            cosine_defined,
            parameter_dot
            / cosine_denominator.clamp_min(torch.finfo(torch.float32).tiny),
            torch.zeros_like(parameter_dot),
        ).clamp(-1.0, 1.0)

    return DynamicVoCGateParameterAlignment(
        loss=loss,
        target_weight=target_weight.detach(),
        target_bias=target_bias.detach(),
        gate_weight_norm=gate_weight_norm,
        target_weight_norm=target_weight_norm,
        weight_error_norm=weight_error_norm,
        gate_bias=gate_bias_value,
        bias_error_abs=bias_error_abs,
        gate_parameter_norm=gate_parameter_norm,
        target_parameter_norm=target_parameter_norm,
        parameter_error_norm=parameter_error_norm,
        relative_parameter_error=relative_parameter_error,
        relative_error_defined=relative_defined.float(),
        cosine=cosine,
        cosine_defined=cosine_defined.float(),
    )


def project_dynamic_voc_gate_head_exact_(
    *,
    gate_weight,
    gate_bias,
    ema_q_weight,
    ema_q_bias,
    q_temperature,
    policy_temperature,
):
    """Project the exported scalar gate onto the exact frozen soft-Q map.

    The parameter target is identical to the optional v9 alignment target,
    but projection is a deterministic state transition rather than an
    optimizer objective.  Requiring FP32 gate and EMA tensors makes the saved
    head byte-exactly auditable with ``torch.equal``.
    """

    if gate_weight.dtype != torch.float32 or gate_bias.dtype != torch.float32:
        raise TypeError("VoC exact projection requires an FP32 gate head")
    if ema_q_weight.dtype != torch.float32 or ema_q_bias.dtype != torch.float32:
        raise TypeError("VoC exact projection requires FP32 EMA Q tensors")
    alignment = compute_dynamic_voc_gate_parameter_alignment_loss(
        gate_weight=gate_weight,
        gate_bias=gate_bias,
        ema_q_weight=ema_q_weight,
        ema_q_bias=ema_q_bias,
        q_temperature=q_temperature,
        policy_temperature=policy_temperature,
    )
    with torch.no_grad():
        gate_weight.copy_(alignment.target_weight)
        gate_bias.copy_(alignment.target_bias)
        if not torch.equal(gate_weight, alignment.target_weight) or not torch.equal(
            gate_bias, alignment.target_bias
        ):
            raise RuntimeError("VoC exact gate projection was not byte-exact")
        post_error = torch.linalg.vector_norm(
            torch.cat(
                (
                    (gate_weight - alignment.target_weight).reshape(-1),
                    (gate_bias - alignment.target_bias).reshape(-1),
                )
            )
        )
        if post_error.item() != 0.0:
            raise RuntimeError("VoC exact gate projection left residual error")
    return DynamicVoCGateExactProjection(
        target_weight=alignment.target_weight,
        target_bias=alignment.target_bias,
        pre_projection_error_norm=alignment.parameter_error_norm,
        post_projection_error_norm=post_error,
    )


def normalize_dynamic_voc_mode(mode):
    """Return the canonical opt-in VoC mode or fail on a typo."""

    mode = str(mode).strip().lower()
    if mode not in {"off", "shadow", "control"}:
        raise ValueError(
            "dynamic_voc_mode must be one of off, shadow, control; "
            f"got {mode!r}"
        )
    return mode


def compute_dynamic_voc_soft_q_gate_loss(
    *,
    gate_log_odds,
    q_values,
    valid,
    q_temperature,
    policy_temperature=1.0,
    confidence_weighted=True,
    tie_tolerance=1e-6,
    saturation_threshold=0.1,
):
    """Distill a detached soft-Q teacher into the dedicated binary gate.

    Q order is ``[CONTINUE, STOP]``.  The scalar student logit is the
    pre-temperature, pre-exploration ``log pi(C) - log pi(S)`` supplied by
    ActorNet.  ``confidence`` always reports the teacher's actual certainty.
    The separate ``objective_weight`` applies that certainty only when
    requested; an unweighted soft-target BCE therefore restores a stale
    student to 0.5 at an equal-Q state.
    """

    if not torch.is_tensor(gate_log_odds) or not torch.is_floating_point(
        gate_log_odds
    ):
        raise TypeError("VoC gate log-odds must be a floating tensor")
    if tuple(q_values.shape) != tuple(gate_log_odds.shape) + (2,):
        raise ValueError(
            "VoC soft-Q values must have shape gate_log_odds.shape + (2,)"
        )
    if tuple(valid.shape) != tuple(gate_log_odds.shape):
        raise ValueError("VoC soft-Q valid mask must match gate log-odds")
    if not np.isfinite(float(q_temperature)) or float(q_temperature) <= 0.0:
        raise ValueError("VoC gate Q temperature must be finite and positive")
    if (
        not np.isfinite(float(policy_temperature))
        or float(policy_temperature) <= 0.0
    ):
        raise ValueError(
            "VoC gate policy temperature must be finite and positive"
        )
    if not np.isfinite(float(tie_tolerance)) or float(tie_tolerance) < 0.0:
        raise ValueError("VoC gate tie tolerance must be finite and non-negative")
    if (
        not np.isfinite(float(saturation_threshold))
        or not 0.0 < float(saturation_threshold) < 0.5
    ):
        raise ValueError(
            "VoC gate saturation threshold must be finite and in (0, 0.5)"
        )

    valid = valid.to(device=gate_log_odds.device, dtype=torch.bool)
    q_values = q_values.to(device=gate_log_odds.device).detach().float()
    # ActorNet supplies the raw dedicated-head log-odds, while the sampled
    # behavior gate applies ``voc_gate_temperature`` before constructing its
    # binary distribution.  Distil the probability that is actually sampled,
    # not an untempered surrogate policy.
    raw_work_log_odds = gate_log_odds.float() / float(policy_temperature)
    _require_finite_tensor(
        "VoC dedicated gate log-odds", raw_work_log_odds[valid]
    )
    _require_finite_tensor("VoC dedicated gate Q", q_values[valid])
    # Invalid WAIT/forced/non-control rows are outside the objective.  Replace
    # them before nonlinear arithmetic so even diagnostic garbage there cannot
    # create ``0 * NaN`` and poison an otherwise valid gate update.
    work_log_odds = torch.where(
        valid, raw_work_log_odds, torch.zeros_like(raw_work_log_odds)
    )

    delta_q = torch.where(
        valid,
        q_values[..., 0] - q_values[..., 1],
        torch.zeros_like(q_values[..., 0]),
    ).detach()
    teacher = torch.sigmoid(delta_q / float(q_temperature)).detach()
    student = torch.sigmoid(work_log_odds)
    confidence = (2.0 * teacher - 1.0).abs().detach()
    if confidence_weighted:
        objective_weight = confidence
    else:
        objective_weight = torch.ones_like(teacher)
    bce = F.binary_cross_entropy_with_logits(
        work_log_odds, teacher, reduction="none"
    )
    valid_float = valid.float()
    denominator = valid_float.sum().clamp_min(1.0)
    loss = torch.sum(objective_weight * bce * valid_float) / denominator
    # This is the unnormalised per-row derivative with respect to the *raw*
    # dedicated-head log-odds before division by the valid-row denominator.
    # Logging it makes saturation and teacher/student disagreement observable
    # without an extra autograd pass.
    directed_logit_gradient = (
        objective_weight
        * (student - teacher)
        * valid_float
        / float(policy_temperature)
    ).detach()
    positive = valid & (delta_q > float(tie_tolerance))
    negative = valid & (delta_q < -float(tie_tolerance))
    wrong_continue_saturation = positive & (
        student.detach() < float(saturation_threshold)
    )
    wrong_stop_saturation = negative & (
        student.detach() > 1.0 - float(saturation_threshold)
    )

    _require_finite_tensor("VoC dedicated gate teacher", teacher[valid])
    _require_finite_tensor("VoC dedicated gate confidence", confidence[valid])
    _require_finite_tensor(
        "VoC dedicated gate objective weight", objective_weight[valid]
    )
    _require_finite_tensor("VoC dedicated gate BCE", bce[valid])
    _require_finite_tensor("VoC dedicated gate loss", loss)
    return DynamicVoCSoftGateLoss(
        loss=loss,
        bce=bce,
        student_continue_probability=student,
        teacher_continue_probability=teacher,
        confidence=confidence,
        objective_weight=objective_weight,
        delta_q=delta_q,
        directed_logit_gradient=directed_logit_gradient,
        wrong_continue_saturation=wrong_continue_saturation,
        wrong_stop_saturation=wrong_stop_saturation,
        valid=valid,
    )


def dynamic_voc_holdout_mask(
    actor_id, control_valid, *, total_actor_streams=None
):
    """Reserve stable actor streams for pre-update TD calibration.

    Actor ids divisible by ``VOC_HOLDOUT_ACTOR_MODULUS`` never contribute to
    the VoC Q regression, but they still follow the same behavior policy and
    therefore provide online held-out TD errors.  ``total_actor_streams`` is
    the full topology size, not the current PPO minibatch width, so a
    minibatch containing only reserved streams cannot leak back into Q
    training.  A genuinely single-stream learner keeps its only stream for
    training and reports zero holdout support.
    """

    if control_valid.ndim != 2:
        raise ValueError("VoC control_valid must have shape [T, B]")
    B = control_valid.shape[1]
    if total_actor_streams is None:
        total_actor_streams = B
    if (
        isinstance(total_actor_streams, (bool, np.bool_))
        or not isinstance(total_actor_streams, (int, np.integer))
        or int(total_actor_streams) <= 0
    ):
        raise ValueError("total_actor_streams must be a positive integer")
    if int(total_actor_streams) == 1:
        return torch.zeros_like(control_valid, dtype=torch.bool)
    ids = torch.as_tensor(actor_id, device=control_valid.device).reshape(-1)
    if ids.numel() != B:
        raise ValueError(
            f"VoC actor_id must contain B={B} ids, got {ids.numel()}"
        )
    if ids.dtype == torch.bool or torch.is_floating_point(ids):
        raise TypeError("VoC actor_id must use an integer dtype")
    held_actor = (
        torch.remainder(ids.long(), util.VOC_HOLDOUT_ACTOR_MODULUS) == 0
    )
    return control_valid.bool() & held_actor.unsqueeze(0)


def dynamic_voc_policy_log_probs(
    reward_log_probs, mode, *, detach_cur_gate=False
):
    """Remove only the legacy task/cost gate gradients in control mode.

    The unmodified likelihoods must still be used to form V-trace importance
    ratios.  This helper returns the likelihoods used by the old reward-channel
    policy losses.  In ``control`` mode, the new VoC advantage is the sole
    task/think reward gradient for the STOP/CONTINUE gate; primary and bout
    gradients remain unchanged.
    """

    mode = normalize_dynamic_voc_mode(mode)
    if mode != "control":
        return reward_log_probs
    if "re" not in reward_log_probs or "think" not in reward_log_probs:
        raise KeyError("VoC control requires re and think likelihood routes")
    out = dict(reward_log_probs)
    detached_gate = reward_log_probs["think"].detach()
    # Preserve the exact forward likelihood (and therefore PPO/V-trace
    # numerics) while replacing only its gate backward path.
    out["re"] = (
        reward_log_probs["re"]
        - reward_log_probs["think"]
        + detached_gate
    )
    out["think"] = detached_gate
    if detach_cur_gate and "cur" in reward_log_probs:
        out["cur"] = (
            reward_log_probs["cur"]
            - reward_log_probs["think"]
            + detached_gate
        )
    return out


def detach_dynamic_voc_gate_from_joint_logits(logits):
    """Keep a joint control policy's value while detaching its binary gate.

    Dedicated VoC action probabilities are represented as a normalized joint
    three-way distribution: ``P(C) P(PROCEED|C)``,
    ``P(C) P(RESET|C)``, and ``P(STOP)``.  PPO's KL may continue to regularize
    the conditional PROCEED/RESET head, but soft-Q BCE must be the dedicated
    STOP/CONTINUE head's sole gradient source.  Subtracting the aggregate gate
    log-probability and adding back its detached value preserves the exact
    forward logits while retaining only the conditional-bout derivative.
    """

    if not torch.is_tensor(logits) or not torch.is_floating_point(logits):
        raise TypeError("dynamic VoC joint logits must be a floating tensor")
    if logits.ndim < 1 or logits.shape[-1] != 3:
        raise ValueError(
            "dynamic VoC joint logits must end in three controls"
        )
    continue_log_prob = torch.logsumexp(logits[..., :2], dim=-1)
    stop_log_prob = logits[..., util.STOP]
    gate_log_prob = torch.stack(
        (continue_log_prob, continue_log_prob, stop_log_prob), dim=-1
    )
    return logits - gate_log_prob + gate_log_prob.detach()


def _dynamic_voc_one_step_target(rewards, discounts, vs, bootstrap_value):
    """Build ``r_t + discount_t * VTrace(s_{t+1})`` without normalization."""

    if rewards.shape != discounts.shape or rewards.shape != vs.shape:
        raise ValueError(
            "VoC reward, discount and V-trace value shapes must match: "
            f"{tuple(rewards.shape)}, {tuple(discounts.shape)}, {tuple(vs.shape)}"
        )
    if tuple(bootstrap_value.shape) != tuple(rewards.shape[1:]):
        raise ValueError(
            "VoC bootstrap shape must match a single time slice: "
            f"{tuple(bootstrap_value.shape)} != {tuple(rewards.shape[1:])}"
        )
    next_vs = torch.cat((vs[1:], bootstrap_value.unsqueeze(0)), dim=0)
    return rewards + discounts * next_vs


def compute_dynamic_voc_target(
    *,
    task_rewards,
    think_rewards,
    task_discounts,
    think_discounts,
    task_vs,
    think_vs,
    task_bootstrap_value,
    think_bootstrap_value,
    think_cost,
):
    """Combine recursive environment-return and computation-cost targets."""

    think_cost = float(think_cost)
    if not np.isfinite(think_cost) or think_cost < 0.0:
        raise ValueError(
            f"VoC think_cost must be finite and non-negative, got {think_cost}"
        )
    task_target = _dynamic_voc_one_step_target(
        task_rewards, task_discounts, task_vs, task_bootstrap_value
    )
    think_target = _dynamic_voc_one_step_target(
        think_rewards, think_discounts, think_vs, think_bootstrap_value
    )
    net_target = task_target + think_cost * think_target
    return DynamicVoCTarget(
        task=task_target.detach(),
        think=think_target.detach(),
        net=net_target.detach(),
    )


def resolve_dynamic_voc_learning_control_surface(
    *,
    execution_control_logits,
    actor_misc,
    control_valid,
    epsilon_greedy_execution,
):
    """Resolve the policy surface used by VoC Q/EMA learning.

    Schemas through v11 have one soft control surface, so this helper is an
    exact identity and deliberately does not inspect ``ActorOut.misc``.  In
    v12 the executed epsilon-greedy policy is a separate behavior surface;
    learning and calibration must instead consume the preserved soft logits.
    Both representations are validated before any optimizer loss is built.
    """

    if not epsilon_greedy_execution:
        return execution_control_logits, None
    if not torch.is_tensor(execution_control_logits):
        raise RuntimeError(
            "epsilon-greedy VoC execution requires target execution logits"
        )
    if not torch.is_floating_point(execution_control_logits):
        raise TypeError(
            "VoC target execution control logits must be a floating tensor"
        )
    if not isinstance(actor_misc, Mapping):
        raise RuntimeError(
            "epsilon-greedy VoC execution requires ActorOut.misc"
        )
    soft_control_logits = actor_misc.get("voc_gate_soft_control_logits")
    soft_continue_probability = actor_misc.get(
        "voc_gate_soft_continue_probability"
    )
    if soft_control_logits is None or soft_continue_probability is None:
        raise RuntimeError(
            "epsilon-greedy VoC execution requires separate soft gate "
            "logits/probability in ActorOut.misc"
        )
    if not torch.is_tensor(soft_control_logits) or not torch.is_floating_point(
        soft_control_logits
    ):
        raise TypeError("VoC soft control logits must be a floating tensor")
    if not torch.is_tensor(
        soft_continue_probability
    ) or not torch.is_floating_point(soft_continue_probability):
        raise TypeError(
            "VoC soft continue probability must be a floating tensor"
        )

    valid = control_valid.to(
        device=execution_control_logits.device, dtype=torch.bool
    )
    expected_logits_shape = tuple(valid.shape) + (3,)
    for name, logits in (
        ("target execution control logits", execution_control_logits),
        ("misc['voc_gate_soft_control_logits']", soft_control_logits),
    ):
        if tuple(logits.shape) != expected_logits_shape:
            raise ValueError(
                f"{name} must have shape {expected_logits_shape}, "
                f"got {tuple(logits.shape)}"
            )
    if tuple(soft_continue_probability.shape) != tuple(valid.shape):
        raise ValueError(
            "misc['voc_gate_soft_continue_probability'] must match the VoC "
            "control mask"
        )
    _require_finite_tensor(
        "VoC target execution control logits",
        execution_control_logits[valid],
    )
    _require_finite_tensor(
        "VoC soft control logits", soft_control_logits[valid]
    )
    _require_finite_tensor(
        "VoC soft continue probability",
        soft_continue_probability[valid],
    )
    if torch.any(
        (soft_continue_probability[valid] < 0.0)
        | (soft_continue_probability[valid] > 1.0)
    ):
        raise ValueError("VoC soft continue probability must be in [0, 1]")
    reconstructed_soft_probability = F.softmax(
        soft_control_logits, dim=-1
    )[..., :2].sum(dim=-1)
    if not torch.allclose(
        reconstructed_soft_probability[valid],
        soft_continue_probability[valid],
        rtol=1e-6,
        atol=1e-7,
    ):
        raise RuntimeError("VoC soft gate logits/probability disagree")
    return soft_control_logits, soft_continue_probability.detach()


def compute_dynamic_voc_loss(
    *,
    voc_q,
    target_control_logits,
    behavior_control_logits,
    target_behavior_control_logits=None,
    control_action,
    control_valid,
    voc_target,
    mode,
    q_train_valid=None,
    dueling_q=False,
    voc_state_value=None,
    expected_gate_loss=False,
    gate_policy_schema_version=None,
):
    """Selected-action Huber critic loss and soft off-policy gate loss.

    Q order is ``[CONTINUE, STOP]``.  PROCEED and RESET are both mapped to
    CONTINUE.  With ``dueling_q``, ``voc_q`` contains raw advantages and the
    public Q values are reconstructed around the detached return value
    ``voc_state_value`` with a policy-centered advantage.  Q values are
    detached from the gate policy loss, and gate logits are absent from the
    critic loss, keeping the two gradient paths explicit.

    ``target_control_logits`` is always the learned soft policy used for Q
    centering and calibration.  ``target_behavior_control_logits`` may carry
    a distinct execution policy solely for the behavior importance ratio; it
    defaults to the soft policy for schemas that have only one surface.
    Gate-policy schema 8 changes only the selected-action regression row from
    beta-1 Smooth-L1 to exact FP32 half-squared error.  Schema 9 retains that
    loss and adds the raw two-action mean to the reconstructed Q values.
    Schema 10 retains schema 9's reconstruction and restores the historical
    beta-1 Smooth-L1 selected-action loss.  ``None`` retains the historical
    unversioned/legacy loss path.
    """

    mode = normalize_dynamic_voc_mode(mode)
    if mode == "off":
        raise ValueError("compute_dynamic_voc_loss is invalid in off mode")
    expected_q_shape = tuple(control_action.shape) + (2,)
    if tuple(voc_q.shape) != expected_q_shape:
        raise ValueError(
            f"voc_q must have shape {expected_q_shape}, got {tuple(voc_q.shape)}"
        )
    expected_logits_shape = tuple(control_action.shape) + (3,)
    if target_behavior_control_logits is None:
        target_behavior_control_logits = target_control_logits
    for name, logits in (
        ("target_control_logits", target_control_logits),
        (
            "target_behavior_control_logits",
            target_behavior_control_logits,
        ),
        ("behavior_control_logits", behavior_control_logits),
    ):
        if tuple(logits.shape) != expected_logits_shape:
            raise ValueError(
                f"{name} must have shape {expected_logits_shape}, "
                f"got {tuple(logits.shape)}"
            )
    if tuple(control_valid.shape) != tuple(control_action.shape):
        raise ValueError("control_valid must match control_action")
    if tuple(voc_target.shape) != tuple(control_action.shape):
        raise ValueError("voc_target must match control_action")
    if dueling_q:
        if voc_state_value is None:
            raise ValueError("dueling_q requires voc_state_value")
        if tuple(voc_state_value.shape) != tuple(control_action.shape):
            raise ValueError("voc_state_value must match control_action")
    if control_action.dtype == torch.bool or torch.is_floating_point(
        control_action
    ):
        raise TypeError("control_action must use an integer dtype")

    valid = control_valid.to(device=voc_q.device, dtype=torch.bool)
    if q_train_valid is None:
        q_train_valid = valid
    else:
        if tuple(q_train_valid.shape) != tuple(control_action.shape):
            raise ValueError("q_train_valid must match control_action")
        q_train_valid = q_train_valid.to(device=voc_q.device, dtype=torch.bool)
        if torch.any(q_train_valid & ~valid):
            raise ValueError("q_train_valid must be a subset of control_valid")
    invalid_action = valid & (
        (control_action < util.PROCEED) | (control_action > util.STOP)
    )
    if torch.any(invalid_action):
        bad_values = torch.unique(control_action[invalid_action]).detach().cpu()
        raise ValueError(
            "VoC control_action contains values outside "
            f"[{util.PROCEED}, {util.STOP}]: {bad_values.tolist()}"
        )
    target = voc_target.to(device=voc_q.device, dtype=voc_q.dtype).detach()
    _require_finite_tensor("VoC Q outputs", voc_q[valid])
    _require_finite_tensor("VoC target", target[valid])
    _require_finite_tensor(
        "VoC target control logits", target_control_logits[valid]
    )
    _require_finite_tensor(
        "VoC target behavior control logits",
        target_behavior_control_logits[valid],
    )
    _require_finite_tensor(
        "VoC behavior control logits", behavior_control_logits[valid]
    )
    target_entropy = compute_dynamic_control_entropy(
        target_control_logits, project_gate_gradient=False
    )
    target_gate_probabilities = torch.stack(
        (target_entropy.continue_prob, target_entropy.stop_prob), dim=-1
    )
    if dueling_q:
        state_value = voc_state_value.to(
            device=voc_q.device, dtype=voc_q.dtype
        ).detach()
        _require_finite_tensor("VoC state value", state_value[valid])
        centered_advantage = voc_q - torch.sum(
            target_gate_probabilities.detach() * voc_q, dim=-1, keepdim=True
        )
        if gate_policy_schema_version in (
            util.VOC_GATE_POLICY_COMMON_MODE_Q_SCHEMA_VERSION,
            util.VOC_GATE_POLICY_HUBER_COMMON_Q_SCHEMA_VERSION,
            util.VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION,
            util.VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION,
            util.VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION,
        ):
            common_advantage = voc_q.mean(dim=-1, keepdim=True)
            q_values = (
                state_value.unsqueeze(-1)
                + common_advantage
                + centered_advantage
            )
        else:
            q_values = state_value.unsqueeze(-1) + centered_advantage
    else:
        q_values = voc_q
    _require_finite_tensor("VoC reconstructed Q values", q_values[valid])

    gate_action = torch.where(
        control_action == util.STOP,
        torch.ones_like(control_action),
        torch.zeros_like(control_action),
    ).long()
    selected_q = torch.gather(
        q_values, dim=-1, index=gate_action.unsqueeze(-1)
    ).squeeze(-1)
    td_error = target - selected_q
    if gate_policy_schema_version is not None and (
        type(gate_policy_schema_version) is not int
        or not 1 <= gate_policy_schema_version <= (
            util.VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION
        )
    ):
        raise ValueError(
            "gate_policy_schema_version must be a strict integer in [1, 13]"
        )
    selected_q_work = selected_q.float()
    target_work = target.float()
    half_squared_q = gate_policy_schema_version in (
        util.VOC_GATE_POLICY_HALF_SQUARED_Q_SCHEMA_VERSION,
        util.VOC_GATE_POLICY_COMMON_MODE_Q_SCHEMA_VERSION,
    )
    if half_squared_q:
        q_error = selected_q_work - target_work
        q_loss_rows = 0.5 * q_error.square()
    else:
        q_loss_rows = F.smooth_l1_loss(
            selected_q_work, target_work, reduction="none"
        )
    q_loss = torch.sum(q_loss_rows * q_train_valid.float())
    if half_squared_q:
        schema_label = (
            "schema-8"
            if gate_policy_schema_version
            == util.VOC_GATE_POLICY_HALF_SQUARED_Q_SCHEMA_VERSION
            else "schema-9"
        )
        _require_finite_tensor(
            f"{schema_label} VoC half-squared Q loss", q_loss
        )

    target_parts = compute_dynamic_control_log_probs(
        target_behavior_control_logits,
        control_action,
        valid,
        # In control mode these are already normalized joint logits produced
        # by compute_voc_gate_distribution.  Exact refactorization cancels the
        # conditional PROCEED/RESET terms; applying the legacy common-shift
        # projection a second time would leak bout gradients into the gate.
        project_gate_gradient=False,
    )
    with torch.no_grad():
        behavior_parts = compute_dynamic_control_log_probs(
            behavior_control_logits,
            control_action,
            valid,
            project_gate_gradient=False,
        )
        behavior_entropy = compute_dynamic_control_entropy(
            behavior_control_logits, project_gate_gradient=False
        )
        behavior_continue_probability = (
            behavior_entropy.continue_prob.detach()
        )
        gate_rho = torch.exp(
            target_parts.gate.detach() - behavior_parts.gate.detach()
        ).clamp(max=1.0)
        continue_probability = target_entropy.continue_prob.detach()
        q_detached = q_values.detach()
        policy_q = (
            continue_probability * q_detached[..., 0]
            + target_entropy.stop_prob.detach() * q_detached[..., 1]
        )
        selected_advantage = torch.gather(
            q_detached, dim=-1, index=gate_action.unsqueeze(-1)
        ).squeeze(-1) - policy_q
        greedy_action = torch.argmax(q_detached, dim=-1)

    if mode == "control":
        if expected_gate_loss:
            # The binary gate is small enough to sum its expectation exactly.
            # Work in FP32 because an actor-independent common return offset
            # can make the summed scalar large under AMP even though its gate
            # gradient depends only on Q_continue - Q_stop.
            gate_pg_loss = -torch.sum(
                target_gate_probabilities.float()
                * q_values.detach().float()
                * valid.unsqueeze(-1).float()
            )
        else:
            gate_pg_loss = -torch.sum(
                gate_rho
                * selected_advantage.detach()
                * target_parts.gate
                * valid.float()
            )
    else:
        # Keep a graph-connected exact zero so callers can sum losses without
        # special casing mixed-precision or empty SEARCH batches.
        gate_pg_loss = target_parts.gate.sum() * 0.0

    return DynamicVoCLossResult(
        q_loss=q_loss,
        gate_pg_loss=gate_pg_loss,
        q_values=q_values,
        target=target,
        selected_q=selected_q,
        td_error=td_error,
        delta_q=q_values[..., 0] - q_values[..., 1],
        continue_probability=continue_probability,
        behavior_continue_probability=behavior_continue_probability,
        selected_advantage=selected_advantage,
        gate_rho=gate_rho,
        gate_action=gate_action,
        greedy_action=greedy_action,
        valid=valid,
        q_train_valid=q_train_valid,
    )


def dynamic_voc_observability_metrics(
    *,
    delta_q,
    continue_probability,
    behavior_continue_probability=None,
    gate_action,
    control_valid,
    search_steps,
    control_action,
    predecision_last_control,
    q_temperature,
    tie_tolerance=1e-6,
):
    """Return detached diagnostics for state/depth-adaptive VoC behavior.

    ``search_steps`` and ``control_action`` are the learner's shifted,
    post-action environment/action pair.  ``predecision_last_control`` must be
    captured from ``TrainActorOut.last_search_control[:-1]`` *before* that
    shift; it is therefore the accepted PROCEED/RESET/STOP token entering the
    decision represented by ``delta_q``.  This alignment lets the post-compute
    slices include the first optimized row without guessing across unroll
    boundaries.  Forced-only decision rows are already excluded by
    ``control_valid``; a shifted ``forced_stop`` marker instead describes the
    post-action result of a legitimate P/R decision and must not filter it.

    The helper is deliberately behavior-neutral: it detaches every input,
    performs no sampling or model forward, and returns graph-free scalars.
    Empty slices are represented by a zero value plus an explicit zero count.
    """

    tie_tolerance = float(tie_tolerance)
    if not np.isfinite(tie_tolerance) or tie_tolerance < 0.0:
        raise ValueError(
            "VoC diagnostic tie_tolerance must be finite and non-negative"
        )
    if isinstance(q_temperature, (bool, np.bool_)):
        raise ValueError(
            "VoC diagnostic q_temperature must be finite and positive"
        )
    q_temperature = float(q_temperature)
    if not np.isfinite(q_temperature) or q_temperature <= 0.0:
        raise ValueError(
            "VoC diagnostic q_temperature must be finite and positive"
        )

    expected_shape = tuple(delta_q.shape)
    if len(expected_shape) != 2:
        raise ValueError(
            "VoC observability decisions must have shape [T, B], "
            f"got {expected_shape}"
        )
    values = {
        "continue_probability": continue_probability,
        "gate_action": gate_action,
        "control_valid": control_valid,
        "search_steps": search_steps,
        "control_action": control_action,
        "predecision_last_control": predecision_last_control,
    }
    if behavior_continue_probability is not None:
        values["behavior_continue_probability"] = (
            behavior_continue_probability
        )
    for name, value in values.items():
        if value is None or tuple(value.shape) != expected_shape:
            actual = None if value is None else tuple(value.shape)
            raise ValueError(
                f"VoC diagnostic {name} must have shape {expected_shape}, "
                f"got {actual}"
            )

    with torch.no_grad():
        delta = delta_q.detach().float()
        probability = continue_probability.detach().to(
            device=delta.device, dtype=torch.float
        )
        behavior_probability = None
        if behavior_continue_probability is not None:
            behavior_probability = behavior_continue_probability.detach().to(
                device=delta.device, dtype=torch.float
            )
        sampled_gate = gate_action.detach().to(
            device=delta.device, dtype=torch.long
        )
        valid = control_valid.detach().to(
            device=delta.device, dtype=torch.bool
        )
        steps = search_steps.detach().to(
            device=delta.device, dtype=torch.long
        )
        current_control = control_action.detach().to(
            device=delta.device, dtype=torch.long
        )
        previous_control = predecision_last_control.detach().to(
            device=delta.device, dtype=torch.long
        )

        if torch.any(valid & ((sampled_gate < 0) | (sampled_gate > 1))):
            raise ValueError("Valid VoC diagnostic gate_action must be 0 or 1")
        if torch.any(
            valid
            & (
                (current_control < util.PROCEED)
                | (current_control > util.STOP)
            )
        ):
            raise ValueError(
                "Valid VoC diagnostic control_action is outside P/R/STOP"
            )
        if torch.any(~torch.isfinite(delta[valid])):
            raise FloatingPointError("Non-finite VoC diagnostic delta_q")
        if torch.any(~torch.isfinite(probability[valid])):
            raise FloatingPointError(
                "Non-finite VoC diagnostic continue_probability"
            )
        if torch.any(valid & ((probability < 0.0) | (probability > 1.0))):
            raise ValueError(
                "VoC diagnostic continue_probability must be in [0, 1]"
            )
        if behavior_probability is not None:
            if torch.any(~torch.isfinite(behavior_probability[valid])):
                raise FloatingPointError(
                    "Non-finite VoC diagnostic behavior continue probability"
                )
            if torch.any(
                valid
                & (
                    (behavior_probability < 0.0)
                    | (behavior_probability > 1.0)
                )
            ):
                raise ValueError(
                    "VoC diagnostic behavior continue probability must be "
                    "in [0, 1]"
                )

        decision_depth = steps - (
            valid & (current_control != util.STOP)
        ).long()
        if torch.any(valid & (decision_depth < 0)):
            raise ValueError("Valid VoC diagnostic has negative decision depth")

        positive = valid & (delta > tie_tolerance)
        negative = valid & (delta < -tie_tolerance)
        tied = valid & ~(positive | negative)
        nontie = positive | negative
        sampled_continue = sampled_gate == 0
        probability_continue = probability > 0.5
        # Binary gate order is [CONTINUE, STOP], so torch.argmax resolves an
        # exact probability tie to CONTINUE (index 0).  Keep the historical
        # strict-threshold agreement metric above unchanged.
        argmax_continue = probability >= 0.5
        sign_agreement = nontie & (
            (positive & probability_continue)
            | (negative & ~probability_continue)
        )
        signed_margin = torch.sign(delta) * (2.0 * probability - 1.0)
        teacher_probability = torch.sigmoid(delta / q_temperature)
        acceptance_valid = valid

        zero = delta.new_zeros(())

        def count(mask):
            return mask.sum().to(dtype=delta.dtype)

        def rate(numerator, denominator):
            denominator_n = count(denominator)
            value = (
                count(numerator) / denominator_n.clamp_min(1.0)
            )
            return torch.where(denominator_n > 0.0, value, zero)

        def mean(value, mask):
            denominator_n = count(mask)
            # torch.where prevents a deliberately masked non-finite sentinel
            # from contaminating the sum; multiplication by zero would not.
            numerator = torch.where(mask, value, torch.zeros_like(value)).sum()
            result = numerator / denominator_n.clamp_min(1.0)
            return torch.where(denominator_n > 0.0, result, zero)

        metrics = {
            "voc_delta_q_positive_count": count(positive),
            "voc_delta_q_negative_count": count(negative),
            "voc_delta_q_tie_count": count(tied),
            "voc_delta_q_positive_rate": rate(positive, valid),
            "voc_delta_q_negative_rate": rate(negative, valid),
            "voc_delta_q_tie_rate": rate(tied, valid),
            "voc_q_greedy_nontie_count": count(nontie),
            "voc_q_greedy_continue_rate": rate(positive, nontie),
            "voc_sign_gate_agreement_count": count(nontie),
            "voc_sign_gate_agreement_rate": rate(sign_agreement, nontie),
            "voc_signed_gate_margin": mean(signed_margin, nontie),
            "voc_continue_probability_delta_positive": mean(
                probability, positive
            ),
            "voc_continue_probability_delta_negative": mean(
                probability, negative
            ),
        }

        def add_slice(prefix, mask, *, include_behavior_probability=False):
            slice_positive = mask & positive
            slice_negative = mask & negative
            slice_tied = mask & tied
            slice_nontie = mask & nontie
            slice_sign_agreement = mask & sign_agreement
            metrics[f"{prefix}_count"] = count(mask)
            metrics[f"{prefix}_delta_q"] = mean(delta, mask)
            metrics[f"{prefix}_delta_q_positive_rate"] = rate(
                slice_positive, mask
            )
            metrics[f"{prefix}_continue_probability"] = mean(
                probability, mask
            )
            metrics[f"{prefix}_sampled_continue_rate"] = rate(
                mask & sampled_continue, mask
            )
            metrics[f"{prefix}_sampled_stop_rate"] = rate(
                mask & ~sampled_continue, mask
            )
            metrics[f"{prefix}_signed_gate_margin"] = mean(
                signed_margin, slice_nontie
            )
            metrics[f"{prefix}_delta_q_positive_count"] = count(
                slice_positive
            )
            metrics[f"{prefix}_delta_q_negative_count"] = count(
                slice_negative
            )
            metrics[f"{prefix}_delta_q_tie_count"] = count(slice_tied)
            metrics[f"{prefix}_delta_q_nontie_count"] = count(
                slice_nontie
            )
            metrics[
                f"{prefix}_continue_probability_delta_positive"
            ] = mean(probability, slice_positive)
            metrics[
                f"{prefix}_continue_probability_delta_negative"
            ] = mean(probability, slice_negative)
            metrics[
                f"{prefix}_teacher_continue_probability_delta_positive"
            ] = mean(teacher_probability, slice_positive)
            metrics[
                f"{prefix}_teacher_continue_probability_delta_negative"
            ] = mean(teacher_probability, slice_negative)
            if include_behavior_probability:
                if behavior_probability is None:
                    raise RuntimeError(
                        "VoC behavior-probability telemetry was requested "
                        "without behavior logits"
                    )
                metrics[
                    f"{prefix}_behavior_continue_probability_delta_positive"
                ] = mean(behavior_probability, slice_positive)
                metrics[
                    f"{prefix}_behavior_continue_probability_delta_negative"
                ] = mean(behavior_probability, slice_negative)
            metrics[
                f"{prefix}_sampled_continue_given_delta_positive_rate"
            ] = rate(slice_positive & sampled_continue, slice_positive)
            metrics[
                f"{prefix}_sampled_stop_given_delta_negative_rate"
            ] = rate(slice_negative & ~sampled_continue, slice_negative)
            metrics[
                f"{prefix}_argmax_continue_given_delta_positive_rate"
            ] = rate(slice_positive & argmax_continue, slice_positive)
            metrics[
                f"{prefix}_argmax_stop_given_delta_negative_rate"
            ] = rate(slice_negative & ~argmax_continue, slice_negative)
            # As with the historical top-level metric, ``count`` is the
            # nontie support over which the agreement rate is defined.
            metrics[f"{prefix}_sign_gate_agreement_count"] = count(
                slice_nontie
            )
            metrics[f"{prefix}_sign_gate_agreement_rate"] = rate(
                slice_sign_agreement, slice_nontie
            )

        add_slice(
            "voc_acceptance",
            acceptance_valid,
            include_behavior_probability=(behavior_probability is not None),
        )
        add_slice(
            "voc_acceptance_depth_8_plus",
            acceptance_valid & (decision_depth >= 8),
            include_behavior_probability=(behavior_probability is not None),
        )

        depth_bins = (
            ("0", decision_depth == 0),
            ("1", decision_depth == 1),
            ("2_3", (decision_depth >= 2) & (decision_depth <= 3)),
            ("4_7", (decision_depth >= 4) & (decision_depth <= 7)),
            ("8_15", (decision_depth >= 8) & (decision_depth <= 15)),
            ("16_plus", decision_depth >= 16),
        )
        for label, depth_mask in depth_bins:
            add_slice(f"voc_depth_bin_{label}", valid & depth_mask)

        previous_continue = (
            (previous_control == util.PROCEED)
            | (previous_control == util.RESET)
        )
        post_compute = valid & (decision_depth > 0) & previous_continue
        add_slice("voc_post_compute", post_compute)
        add_slice(
            "voc_post_proceed",
            post_compute & (previous_control == util.PROCEED),
        )
        add_slice(
            "voc_post_reset",
            post_compute & (previous_control == util.RESET),
        )

        # A post-compute label is only causal when the preceding row is the
        # accepted computation from the same bout.  Never infer that relation
        # across the leading time boundary, invalid rows, a STOP, a mismatched
        # replay token, or a non-consecutive depth.  The prior positive-Q
        # condition identifies an accepted computation that the online/EMA
        # source (depending on the caller) judged useful at that decision.
        prior_accepted_compute = (
            (current_control[:-1] == util.PROCEED)
            | (current_control[:-1] == util.RESET)
        )
        strict_same_bout_pair = (
            acceptance_valid[:-1]
            & acceptance_valid[1:]
            & prior_accepted_compute
            & (previous_control[1:] == current_control[:-1])
            & (decision_depth[1:] == decision_depth[:-1] + 1)
        )
        prior_useful_pair = strict_same_bout_pair & (
            delta[:-1] > tie_tolerance
        )
        prior_useful_candidate = (
            acceptance_valid
            & (
                (current_control == util.PROCEED)
                | (current_control == util.RESET)
            )
            & (delta > tie_tolerance)
        )

        def add_post_useful_slice(
            prefix,
            pair_mask,
            candidate_mask,
            *,
            include_behavior_probability=False,
        ):
            # Prefixing one empty time row makes pair[t-1, b] describe the
            # current re-evaluation at [t, b], and makes t=0 unrepresentable.
            current_mask = torch.cat(
                (torch.zeros_like(valid[:1]), pair_mask), dim=0
            )
            metrics[f"{prefix}_prior_useful_count"] = count(pair_mask)
            metrics[f"{prefix}_prior_delta_q"] = mean(
                delta[:-1], pair_mask
            )
            metrics[f"{prefix}_prior_useful_candidate_count"] = count(
                candidate_mask
            )
            metrics[f"{prefix}_transition_coverage_rate"] = rate(
                pair_mask, candidate_mask
            )
            add_slice(
                prefix,
                current_mask,
                include_behavior_probability=include_behavior_probability,
            )

        add_post_useful_slice(
            "voc_post_useful_compute",
            prior_useful_pair,
            prior_useful_candidate,
            include_behavior_probability=(behavior_probability is not None),
        )
        add_post_useful_slice(
            "voc_post_useful_proceed",
            prior_useful_pair & (current_control[:-1] == util.PROCEED),
            prior_useful_candidate & (current_control == util.PROCEED),
        )
        add_post_useful_slice(
            "voc_post_useful_reset",
            prior_useful_pair & (current_control[:-1] == util.RESET),
            prior_useful_candidate & (current_control == util.RESET),
        )

    return metrics


def _validate_model_state_dict_compatibility(model, weights, label="ModelNet"):
    """Fail with actionable key/shape diagnostics before a strict refresh."""

    if not isinstance(weights, Mapping) or not weights:
        raise TypeError(f"{label} weights must be a non-empty state-dict mapping")
    expected = model.state_dict()
    expected_keys = set(expected)
    incoming_keys = set(weights)
    missing = sorted(expected_keys - incoming_keys)
    unexpected = sorted(incoming_keys - expected_keys)
    mismatched = []
    contract_mismatched = []
    nonfinite = []
    for key in sorted(expected_keys & incoming_keys):
        incoming_shape = tuple(np.shape(weights[key]))
        expected_shape = tuple(expected[key].shape)
        if incoming_shape != expected_shape:
            mismatched.append((key, incoming_shape, expected_shape))
            continue
        incoming_value_raw = weights[key]
        if torch.is_tensor(incoming_value_raw):
            if torch.is_floating_point(incoming_value_raw):
                finite = torch.isfinite(incoming_value_raw)
                if not torch.all(finite):
                    nonfinite.append(
                        (key, int((~finite).sum().detach().cpu()))
                    )
        else:
            incoming_array = np.asarray(incoming_value_raw)
            if (
                np.issubdtype(incoming_array.dtype, np.floating)
                and not np.all(np.isfinite(incoming_array))
            ):
                nonfinite.append(
                    (
                        key,
                        int(
                            np.size(incoming_array)
                            - np.count_nonzero(np.isfinite(incoming_array))
                        ),
                    )
                )
        if not nonfinite or nonfinite[-1][0] != key:
            if key not in {"norm_low", "norm_high"}:
                continue
            if torch.is_tensor(weights[key]):
                incoming_value = weights[key].detach().cpu()
            else:
                incoming_value = torch.tensor(np.asarray(weights[key]))
            expected_value = expected[key].detach().cpu()
            if not torch.equal(incoming_value, expected_value):
                contract_mismatched.append(key)
    if missing or unexpected or mismatched or contract_mismatched or nonfinite:
        detail = []
        if missing:
            detail.append(f"missing={missing[:8]}")
        if unexpected:
            detail.append(f"unexpected={unexpected[:8]}")
        if mismatched:
            rendered = [
                f"{key}: incoming{incoming_shape} != expected{expected_shape}"
                for key, incoming_shape, expected_shape in mismatched[:8]
            ]
            detail.append("shape_mismatch=[" + "; ".join(rendered) + "]")
        if contract_mismatched:
            detail.append(
                "observation_bound_mismatch=" + repr(contract_mismatched)
            )
        if nonfinite:
            detail.append(
                "nonfinite="
                + repr([f"{key} ({count} values)" for key, count in nonfinite[:8]])
            )
        raise ValueError(f"{label} state-dict is incompatible: " + ", ".join(detail))


def _require_finite_tensor(name, value):
    """Fail before a non-finite actor value can mutate optimizer/EMA state."""

    if value is None or not torch.is_tensor(value):
        return
    finite = torch.isfinite(value)
    if torch.all(finite):
        return
    count = int((~finite).sum().detach().cpu())
    raise FloatingPointError(
        f"non-finite {name}: {count}/{value.numel()} values"
    )


def _detached_absmax_rms(value, mask=None):
    """Return stable magnitude diagnostics without joining autograd."""

    if value is None or not torch.is_tensor(value) or value.numel() == 0:
        return 0.0, 0.0
    with torch.no_grad():
        selected = value.detach()
        if mask is not None:
            mask = torch.as_tensor(
                mask, device=selected.device, dtype=torch.bool
            )
            if tuple(selected.shape[: mask.ndim]) != tuple(mask.shape):
                raise ValueError(
                    "observability mask must match tensor leading dimensions: "
                    f"{tuple(mask.shape)} versus {tuple(selected.shape)}"
                )
            selected = selected[mask]
        if selected.numel() == 0:
            return 0.0, 0.0
        work = selected if selected.dtype == torch.float64 else selected.float()
        absmax = work.abs().max()
        scale = torch.where(absmax == 0, torch.ones_like(absmax), absmax)
        rms = scale * torch.sqrt(torch.mean((work / scale).square()))
        return float(absmax.detach().cpu()), float(rms.detach().cpu())


def dynamic_actor_observability_stats(actor_out, *, discrete_action):
    """Detached Dynamic actor input/head ranges with fixed numeric fields."""

    stats = {}

    def add(name, value, mask=None):
        absmax, rms = _detached_absmax_rms(value, mask)
        stats[f"actor/{name}_absmax"] = absmax
        stats[f"actor/{name}_rms"] = rms

    add("env_hs", getattr(actor_out, "hs", None))
    add("env_tree_reps", getattr(actor_out, "tree_reps", None))
    add("env_xs", getattr(actor_out, "xs", None))

    policy_type = getattr(actor_out, "policy_type", None)
    policy_valid = getattr(actor_out, "policy_valid", None)
    primary_valid = getattr(actor_out, "primary_valid", None)
    primary_logits = (
        getattr(actor_out, "pri_param", None) if discrete_action else None
    )
    if (
        policy_type is not None
        and policy_valid is not None
        and primary_valid is not None
    ):
        policy_type = policy_type.long()
        policy_valid = policy_valid.bool()
        primary_valid = primary_valid.bool()
        real_primary_mask = policy_valid & (
            policy_type == util.POLICY_REAL
        )
        search_primary_mask = primary_valid & (
            policy_type == util.POLICY_SEARCH
        )
    else:
        real_primary_mask = None
        search_primary_mask = None
        primary_logits = None
    add("real_primary_logits", primary_logits, real_primary_mask)
    add("search_primary_logits", primary_logits, search_primary_mask)

    control_logits = getattr(actor_out, "search_control_logits", None)
    masked_dynamic_logits = control_logits is not None
    if not masked_dynamic_logits:
        control_logits = getattr(actor_out, "reset_logits", None)
    control_valid = getattr(actor_out, "control_valid", None)
    if control_valid is None:
        control_logits = None
        control_slot_valid = None
    elif masked_dynamic_logits:
        control_slot_valid = (
            control_valid.bool().unsqueeze(-1)
            & control_logits.ne(ILLEGAL_CONTROL_LOGIT)
        )
    else:
        control_slot_valid = control_valid
    add("search_control_logits", control_logits, control_slot_valid)
    return stats


def environment_noop_observability_stats(
    last_primary_action,
    real_transition,
    *,
    num_actions,
    noop_action_index,
):
    """Summarize NOOPs among actions actually executed in the environment.

    Dynamic replay contains SEARCH proposals and WAIT rows as well as real
    Atari transitions.  ``EnvOut.last_pri`` is updated from the wrapper's
    effective/executed primary action, so only rows selected by
    ``real_transition`` represent direct environment interaction.
    """

    if (
        isinstance(num_actions, (bool, np.bool_))
        or not isinstance(num_actions, (int, np.integer))
        or int(num_actions) <= 0
    ):
        raise ValueError("num_actions must be a positive integer")
    num_actions = int(num_actions)
    if noop_action_index is not None:
        if (
            isinstance(noop_action_index, (bool, np.bool_))
            or not isinstance(noop_action_index, (int, np.integer))
        ):
            raise TypeError("noop_action_index must be an integer or None")
        noop_action_index = int(noop_action_index)
        if not 0 <= noop_action_index < num_actions:
            raise ValueError("noop_action_index lies outside the action space")

    with torch.no_grad():
        mask = torch.as_tensor(real_transition).detach()
        if mask.dtype != torch.bool:
            raise TypeError("real_transition must use boolean dtype")
        actions = torch.as_tensor(last_primary_action).detach()
        if actions.ndim == mask.ndim + 1 and actions.shape[-1] == 1:
            actions = actions.squeeze(-1)
        if tuple(actions.shape) != tuple(mask.shape):
            raise ValueError(
                "last_primary_action must match real_transition shape: "
                f"{tuple(actions.shape)} != {tuple(mask.shape)}"
            )
        if actions.dtype == torch.bool or torch.is_floating_point(actions):
            raise TypeError("last_primary_action must use an integer dtype")
        executed = actions[mask].to(device="cpu", dtype=torch.long)
        if torch.any(executed < 0) or torch.any(executed >= num_actions):
            raise ValueError("executed primary action lies outside the action space")

        real_action_count = int(executed.numel())
        noop_supported = noop_action_index is not None
        noop_count = (
            int(torch.count_nonzero(executed == noop_action_index).item())
            if noop_supported
            else 0
        )
        noop_frequency = (
            float(noop_count) / real_action_count
            if noop_supported and real_action_count > 0
            else 0.0
        )
        return {
            "interaction/noop_action_index": (
                int(noop_action_index) if noop_supported else -1
            ),
            "interaction/noop_supported": int(noop_supported),
            "interaction/real_action_count": real_action_count,
            "interaction/noop_count": noop_count,
            "interaction/noop_frequency": noop_frequency,
        }


def _box_contract_equal(left, right):
    """Compare Box shape, dtype, and exact finite/infinite bounds."""

    from gymnasium import spaces

    if not isinstance(left, spaces.Box) or not isinstance(right, spaces.Box):
        return False
    return (
        tuple(left.shape) == tuple(right.shape)
        and np.dtype(left.dtype) == np.dtype(right.dtype)
        and np.array_equal(left.low, right.low)
        and np.array_equal(left.high, right.high)
    )

def compute_baseline_loss(
    baseline,
    target_baseline,
    mask=None,
):
    target_baseline = target_baseline.detach()
    loss = (target_baseline - baseline)**2
    if mask is not None:
        loss = loss * mask
    return torch.sum(loss)

def compute_baseline_enc_loss(
    baseline_enc,
    target_baseline,
    rv_tran,
    enc_type,
    mask=None,
):
    target_baseline = target_baseline.detach()
    if enc_type == 1:
        baseline_enc = baseline_enc
        target_baseline_enc = rv_tran.encode(target_baseline)
        loss = (target_baseline_enc.detach() - baseline_enc)**2
    elif enc_type in [2, 3]:
        target_baseline_enc = rv_tran.encode(target_baseline)
        loss = (
            torch.nn.CrossEntropyLoss(reduction="none")(
                input=torch.flatten(baseline_enc, 0, 1),
                target=torch.flatten(target_baseline_enc, 0, 1).detach(),
            )            
        )
        loss = loss.view(baseline_enc.shape[:2])
    if mask is not None: loss = loss * mask
    return torch.sum(loss)


def dynamic_factorized_policy_log_probs(
    actor_out,
    control_valid,
    primary_valid,
    *,
    discrete_action,
    tanh_action=False,
):
    """Return reward-channel likelihoods for a Dynamic actor output.

    The actor continues to store the historical three-way joint likelihood.
    Components are reconstructed from the stored logits/actions so behavior
    and target policies use exactly the same factorization in V-trace and PPO.
    """

    control_logits = getattr(actor_out, "search_control_logits", None)
    if control_logits is None:
        control_logits = actor_out.reset_logits
    control_action = getattr(actor_out, "search_control", None)
    if control_action is None:
        control_action = actor_out.reset
    parts = compute_dynamic_control_log_probs(
        control_logits, control_action, control_valid
    )
    if discrete_action:
        primary_log_prob = compute_discrete_log_prob(
            actor_out.pri_param, actor_out.pri
        )
    else:
        pri_mean = actor_out.pri_param[..., 0]
        pri_log_var = actor_out.pri_param[..., 1]
        pri_std = torch.exp(pri_log_var / 2)
        pri_pre_tanh = (
            atanh(actor_out.pri) if tanh_action else actor_out.pri
        )
        primary_dist = torch.distributions.Normal(pri_mean, pri_std)
        primary_log_prob = primary_dist.log_prob(pri_pre_tanh)
        if tanh_action:
            primary_log_prob = primary_log_prob - torch.log(
                1.0 - actor_out.pri ** 2 + 1e-6
            )
        primary_log_prob = primary_log_prob.sum(dim=-1)
    primary_log_prob = torch.where(
        primary_valid,
        primary_log_prob,
        torch.zeros_like(primary_log_prob),
    )
    return {
        "re": primary_log_prob + parts.gate + parts.bout,
        "im": primary_log_prob + parts.bout,
        # Curiosity retains the historical unfactorized joint objective.
        "cur": actor_out.c_action_log_prob,
        "think": parts.gate,
    }

class SActorLearner:
    def __init__(
        self,
        ray_obj,
        actor_param,
        flags,
        actor_net=None,
        device=None,
        runtime_action_meanings=None,
    ):
        self.flags = flags
        self.time = flags.profile
        self._logger = util.logger()
        self.dynamic_search = util.dynamic_search_enabled(flags)
        self.dynamic_factorized_control = (
            self.dynamic_search
            and bool(getattr(flags, "dynamic_factorized_control", False))
        )
        self.dynamic_voc_mode = normalize_dynamic_voc_mode(
            getattr(flags, "dynamic_voc_mode", "off")
        )
        self.voc_dueling_q = bool(getattr(flags, "voc_dueling_q", False))
        self.voc_expected_gate_loss = bool(
            getattr(flags, "voc_expected_gate_loss", False)
        )
        self.voc_ema_gate_target = bool(
            getattr(flags, "voc_ema_gate_target", True)
        )
        raw_voc_gate_target_tau = getattr(flags, "voc_gate_target_tau", 0.1)
        raw_gate_schema_for_tau = getattr(
            flags, "voc_gate_policy_schema_version", 6
        )
        if (
            type(raw_gate_schema_for_tau) is int
            and raw_gate_schema_for_tau
            in (
                util.VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION,
                util.VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION,
            )
        ):
            if (
                type(raw_voc_gate_target_tau) is not float
                or not np.isfinite(raw_voc_gate_target_tau)
                or raw_voc_gate_target_tau != 1.0
            ):
                raise ValueError(
                    "schema-12/13 VoC Q requires exact built-in float "
                    "voc_gate_target_tau=1.0"
                )
            self.voc_gate_target_tau = raw_voc_gate_target_tau
        else:
            # Preserve the historical schemas' normalization and exceptions.
            self.voc_gate_target_tau = float(raw_voc_gate_target_tau)
        configured_voc_dedicated_gate = bool(
            getattr(flags, "voc_dedicated_gate", False)
        )
        configured_voc_soft_q_bce_gate = bool(
            getattr(flags, "voc_soft_q_bce_gate", False)
        )
        raw_voc_gate_param_align = getattr(
            flags, "voc_gate_param_align", False
        )
        if not isinstance(raw_voc_gate_param_align, (bool, np.bool_)):
            raise ValueError("voc_gate_param_align must be boolean")
        configured_voc_gate_param_align = bool(raw_voc_gate_param_align)
        raw_voc_gate_exact_projection = getattr(
            flags, "voc_gate_exact_projection", False
        )
        if not isinstance(raw_voc_gate_exact_projection, (bool, np.bool_)):
            raise ValueError("voc_gate_exact_projection must be boolean")
        configured_voc_gate_exact_projection = bool(
            raw_voc_gate_exact_projection
        )
        raw_voc_gate_epsilon_greedy_execution = getattr(
            flags, "voc_gate_epsilon_greedy_execution", False
        )
        if not isinstance(
            raw_voc_gate_epsilon_greedy_execution, (bool, np.bool_)
        ):
            raise ValueError(
                "voc_gate_epsilon_greedy_execution must be boolean"
            )
        configured_voc_gate_epsilon_greedy_execution = bool(
            raw_voc_gate_epsilon_greedy_execution
        )
        raw_policy_version_barrier = getattr(
            flags, "voc_actor_policy_version_barrier", False
        )
        if not isinstance(raw_policy_version_barrier, (bool, np.bool_)):
            raise ValueError("voc_actor_policy_version_barrier must be boolean")
        configured_policy_version_barrier = bool(raw_policy_version_barrier)
        raw_execution_epsilon = getattr(
            flags, "voc_gate_execution_epsilon", 0.02
        )
        raw_actor_amp_init_scale = getattr(
            flags, "actor_amp_init_scale", 256.0
        )
        raw_barrier_timeout = getattr(
            flags,
            "voc_actor_policy_barrier_timeout_s",
            util.VOC_ACTOR_POLICY_BARRIER_TIMEOUT_SECONDS,
        )
        for name, value in (
            ("voc_gate_execution_epsilon", raw_execution_epsilon),
            ("actor_amp_init_scale", raw_actor_amp_init_scale),
            ("voc_actor_policy_barrier_timeout_s", raw_barrier_timeout),
        ):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.number))
                or not np.isfinite(value)
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive")
        self.voc_gate_execution_epsilon = float(raw_execution_epsilon)
        self.actor_amp_init_scale = float(raw_actor_amp_init_scale)
        self.voc_actor_policy_barrier_timeout_s = float(raw_barrier_timeout)
        self.voc_actor_policy_version_barrier = configured_policy_version_barrier
        self.voc_actor_policy_barrier_runtime = bool(
            configured_policy_version_barrier
            and bool(getattr(flags, "train_actor", False))
            and bool(getattr(flags, "parallel_actor", False))
        )
        raw_gate_schema = getattr(
            flags, "voc_gate_policy_schema_version", 6
        )
        if configured_policy_version_barrier:
            if (
                type(raw_gate_schema) is not int
                or raw_gate_schema
                not in (
                    6,
                    7,
                    util.VOC_GATE_POLICY_HALF_SQUARED_Q_SCHEMA_VERSION,
                    util.VOC_GATE_POLICY_COMMON_MODE_Q_SCHEMA_VERSION,
                    util.VOC_GATE_POLICY_HUBER_COMMON_Q_SCHEMA_VERSION,
                    util.VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION,
                    util.VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION,
                    util.VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION,
                )
            ):
                raise ValueError(
                    "versioned actor policy barrier requires exact integer "
                    "gate schema 6, 7, 8, 9, 10, 11, 12, or 13"
                )
            self.voc_gate_policy_schema_version = raw_gate_schema
        else:
            self.voc_gate_policy_schema_version = None
        self._voc_telemetry_active = (
            configured_policy_version_barrier
            and raw_gate_schema
            == util.VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION
        )
        expected_model_input_seal_schema = (
            1
            if raw_gate_schema
            in (
                7,
                util.VOC_GATE_POLICY_HALF_SQUARED_Q_SCHEMA_VERSION,
                util.VOC_GATE_POLICY_COMMON_MODE_Q_SCHEMA_VERSION,
                util.VOC_GATE_POLICY_HUBER_COMMON_Q_SCHEMA_VERSION,
                util.VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION,
                util.VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION,
                util.VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION,
            )
            else 0
        )
        raw_model_input_seal_schema = getattr(
            flags, "voc_model_input_seal_schema_version", 0
        )
        if (
            type(raw_model_input_seal_schema) is not int
            or raw_model_input_seal_schema != expected_model_input_seal_schema
        ):
            raise ValueError(
                "versioned actor policy barrier atomically requires "
                "voc_model_input_seal_schema_version="
                f"{expected_model_input_seal_schema}; got "
                f"{raw_model_input_seal_schema!r}"
            )
        self.voc_model_input_seal_schema_version = (
            raw_model_input_seal_schema
        )
        # The raw actor bootstrap is needed before the rest of learner state
        # exists, so its barrier clock/counter must be live this early.
        self._monotonic = time.monotonic
        self._barrier_sleep = time.sleep
        self.voc_actor_policy_barrier_timeout_count = 0
        raw_bundle_schema = getattr(
            flags,
            "voc_actor_policy_bundle_schema_version",
            util.VOC_ACTOR_POLICY_BUNDLE_SCHEMA_VERSION,
        )
        if (
            isinstance(raw_bundle_schema, (bool, np.bool_))
            or not isinstance(raw_bundle_schema, (int, np.integer))
            or int(raw_bundle_schema)
            != util.VOC_ACTOR_POLICY_BUNDLE_SCHEMA_VERSION
        ):
            raise ValueError(
                "voc_actor_policy_bundle_schema_version must equal 1 exactly"
            )
        for name in (
            "voc_actor_policy_ray_max_restarts",
            "voc_actor_policy_ray_max_task_retries",
        ):
            value = getattr(flags, name, 0)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or int(value) != 0
            ):
                raise ValueError(f"{name} must equal integer 0 exactly")
            setattr(flags, name, int(value))
        raw_voc_gate_param_align_coef = getattr(
            flags, "voc_gate_param_align_coef", 1.0
        )
        if (
            configured_voc_gate_param_align
            or configured_voc_gate_exact_projection
            or configured_voc_gate_epsilon_greedy_execution
        ):
            if (
                isinstance(raw_voc_gate_param_align_coef, (bool, np.bool_))
                or not isinstance(
                    raw_voc_gate_param_align_coef,
                    (int, float, np.number),
                )
                or not np.isfinite(raw_voc_gate_param_align_coef)
                or float(raw_voc_gate_param_align_coef) != 1.0
            ):
                raise ValueError(
                    "active VoC gate alignment/projection requires "
                    "voc_gate_param_align_coef=1.0 exactly"
                )
        self.voc_gate_param_align_coef = 1.0
        # These are active-protocol switches.  Keeping them true in the
        # global defaults must remain harmless for ordinary ``off`` runs,
        # whose ActorNet intentionally has neither a VoC critic nor a gate
        # head.  Preserve the configured metadata on flags while using an
        # explicitly mode-gated runtime value below.
        self.voc_dedicated_gate = (
            self.dynamic_voc_mode != "off"
            and configured_voc_dedicated_gate
        )
        self.voc_soft_q_bce_gate = (
            self.dynamic_voc_mode != "off"
            and configured_voc_soft_q_bce_gate
        )
        self.voc_gate_param_align = (
            self.dynamic_voc_mode == "control"
            and configured_voc_gate_param_align
        )
        self.voc_gate_exact_projection = (
            self.dynamic_voc_mode == "control"
            and configured_voc_gate_exact_projection
        )
        self.voc_gate_epsilon_greedy_execution = (
            self.dynamic_voc_mode == "control"
            and configured_voc_gate_epsilon_greedy_execution
        )
        if configured_policy_version_barrier:
            atomic = (
                ("dynamic_voc_mode", self.dynamic_voc_mode, "control"),
                ("voc_gate_exact_projection", configured_voc_gate_exact_projection, True),
                ("voc_gate_epsilon_greedy_execution", configured_voc_gate_epsilon_greedy_execution, True),
                ("voc_gate_param_align", configured_voc_gate_param_align, False),
                ("voc_gate_execution_epsilon", self.voc_gate_execution_epsilon, 0.25),
                ("voc_train_epsilon", float(getattr(flags, "voc_train_epsilon", 0.02)), 0.02),
                ("actor_amp_init_scale", self.actor_amp_init_scale, 32.0),
                (
                    "voc_actor_policy_barrier_timeout_s",
                    self.voc_actor_policy_barrier_timeout_s,
                    util.VOC_ACTOR_POLICY_BARRIER_TIMEOUT_SECONDS,
                ),
            )
            for name, actual, expected in atomic:
                if actual != expected:
                    raise ValueError(
                        "voc_actor_policy_version_barrier atomically requires "
                        f"{name}={expected!r}; got {actual!r}"
                    )
            if bool(getattr(flags, "train_actor", False)):
                if bool(getattr(flags, "ckp", False)):
                    raise ValueError("schema-6 actor policy barrier is fresh-only")
                if not bool(getattr(flags, "parallel_actor", False)):
                    raise ValueError(
                        "schema-6 actor policy barrier training requires "
                        "parallel_actor=true"
                    )
                for name, expected in (
                    ("ppo_k", 1),
                    ("self_play_n", 1),
                    ("env_n", 16),
                    ("actor_batch_size", 16),
                ):
                    value = getattr(flags, name, None)
                    if (
                        isinstance(value, (bool, np.bool_))
                        or not isinstance(value, (int, np.integer))
                        or int(value) != expected
                    ):
                        raise ValueError(
                            "schema-6 actor policy barrier requires exact "
                            f"{name}={expected}; got {value!r}"
                        )
        elif self.voc_gate_execution_epsilon != 0.02 or self.actor_amp_init_scale != 256.0:
            raise ValueError(
                "legacy schemas require voc_gate_execution_epsilon=0.02 and "
                "actor_amp_init_scale=256.0"
            )
        self.voc_gate_q_temperature = float(
            getattr(flags, "voc_gate_q_temperature", 0.05)
        )
        self.voc_gate_confidence_weighted = bool(
            getattr(flags, "voc_gate_confidence_weighted", True)
        )
        self.voc_gate_learning_rate = float(
            getattr(flags, "voc_gate_learning_rate", 3e-4)
        )
        raw_voc_gate_adam_beta1 = getattr(
            flags, "voc_gate_adam_beta1", 0.9
        )
        if isinstance(raw_voc_gate_adam_beta1, (bool, np.bool_)):
            raise ValueError(
                "voc_gate_adam_beta1 must be finite and in [0, 1)"
            )
        try:
            self.voc_gate_adam_beta1 = float(raw_voc_gate_adam_beta1)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "voc_gate_adam_beta1 must be finite and in [0, 1)"
            ) from exc
        if (
            not np.isfinite(self.voc_gate_adam_beta1)
            or not 0.0 <= self.voc_gate_adam_beta1 < 1.0
        ):
            raise ValueError(
                "voc_gate_adam_beta1 must be finite and in [0, 1)"
            )
        self.voc_gate_grad_norm_clipping = float(
            getattr(flags, "voc_gate_grad_norm_clipping", 1.0)
        )
        # Test harnesses and embedding callers may construct flags directly
        # rather than through util.process_flags.  Persist the normalized
        # active protocol explicitly so checkpoints never omit it.
        self.flags.voc_ema_gate_target = self.voc_ema_gate_target
        self.flags.voc_gate_target_tau = self.voc_gate_target_tau
        self.flags.voc_dedicated_gate = configured_voc_dedicated_gate
        self.flags.voc_soft_q_bce_gate = configured_voc_soft_q_bce_gate
        self.flags.voc_gate_q_temperature = self.voc_gate_q_temperature
        self.flags.voc_gate_confidence_weighted = (
            self.voc_gate_confidence_weighted
        )
        self.flags.voc_gate_learning_rate = self.voc_gate_learning_rate
        self.flags.voc_gate_adam_beta1 = self.voc_gate_adam_beta1
        self.flags.voc_gate_grad_norm_clipping = (
            self.voc_gate_grad_norm_clipping
        )
        # A legacy VoC-off caller remains free of new checkpoint metadata when
        # it does not know about v9.  Every active VoC run stores canonical
        # false/1.0 fields so schema-3 validation can fail closed on omission.
        if (
            self.dynamic_voc_mode != "off"
            or hasattr(flags, "voc_gate_param_align")
        ):
            self.flags.voc_gate_param_align = (
                configured_voc_gate_param_align
            )
            self.flags.voc_gate_param_align_coef = (
                self.voc_gate_param_align_coef
            )
        if (
            self.dynamic_voc_mode != "off"
            or hasattr(flags, "voc_gate_exact_projection")
        ):
            self.flags.voc_gate_exact_projection = (
                configured_voc_gate_exact_projection
            )
        if (
            self.dynamic_voc_mode != "off"
            or hasattr(flags, "voc_gate_epsilon_greedy_execution")
        ):
            self.flags.voc_gate_epsilon_greedy_execution = (
                configured_voc_gate_epsilon_greedy_execution
            )
        for name, value in (
            ("voc_gate_execution_epsilon", self.voc_gate_execution_epsilon),
            ("voc_actor_policy_version_barrier", configured_policy_version_barrier),
            (
                "voc_actor_policy_bundle_schema_version",
                util.VOC_ACTOR_POLICY_BUNDLE_SCHEMA_VERSION,
            ),
            (
                "voc_actor_policy_barrier_timeout_s",
                self.voc_actor_policy_barrier_timeout_s,
            ),
            ("actor_amp_init_scale", self.actor_amp_init_scale),
        ):
            setattr(self.flags, name, value)
        if self.voc_gate_policy_schema_version in (
            7,
            util.VOC_GATE_POLICY_HALF_SQUARED_Q_SCHEMA_VERSION,
            util.VOC_GATE_POLICY_COMMON_MODE_Q_SCHEMA_VERSION,
            util.VOC_GATE_POLICY_HUBER_COMMON_Q_SCHEMA_VERSION,
            util.VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION,
            util.VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION,
            util.VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION,
        ):
            self.flags.voc_model_input_seal_schema_version = 1
        self.flags.voc_actor_policy_barrier_runtime = (
            self.voc_actor_policy_barrier_runtime
        )
        if self.dynamic_voc_mode != "off" and not (
            self.dynamic_search and self.dynamic_factorized_control
        ):
            raise ValueError(
                "dynamic_voc_mode requires dynamic_search=true and "
                "dynamic_factorized_control=true"
            )
        if configured_voc_dedicated_gate != configured_voc_soft_q_bce_gate:
            raise ValueError(
                "voc_dedicated_gate and voc_soft_q_bce_gate must be enabled "
                "together"
            )
        if self.voc_gate_param_align and not self.voc_soft_q_bce_gate:
            raise ValueError(
                "voc_gate_param_align requires the dedicated soft-Q BCE gate"
            )
        if (
            configured_voc_gate_exact_projection
            and self.dynamic_voc_mode != "control"
        ):
            raise ValueError(
                "voc_gate_exact_projection requires control mode"
            )
        if (
            configured_voc_gate_exact_projection
            and configured_voc_gate_param_align
        ):
            raise ValueError(
                "voc_gate_exact_projection and voc_gate_param_align must be "
                "mutually exclusive"
            )
        if self.voc_gate_exact_projection and not self.voc_soft_q_bce_gate:
            raise ValueError(
                "voc_gate_exact_projection requires the dedicated soft-Q "
                "BCE gate"
            )
        if configured_voc_gate_epsilon_greedy_execution:
            if self.dynamic_voc_mode != "control":
                raise ValueError(
                    "voc_gate_epsilon_greedy_execution requires control mode"
                )
            if not configured_voc_gate_exact_projection:
                raise ValueError(
                    "voc_gate_epsilon_greedy_execution requires "
                    "voc_gate_exact_projection=true"
                )
            if configured_voc_gate_param_align:
                raise ValueError(
                    "voc_gate_epsilon_greedy_execution requires "
                    "voc_gate_param_align=false"
                )
        if configured_voc_gate_exact_projection:
            for name in (
                "preload",
                "preload_actor",
                "voc_parent_checkpoint",
            ):
                value = getattr(flags, name, "")
                if not isinstance(value, str) or value.strip():
                    raise ValueError(
                        "voc_gate_exact_projection requires fresh "
                        f"parent-free {name}=''"
                    )
        if self.voc_soft_q_bce_gate:
            if (
                not np.isfinite(self.voc_gate_q_temperature)
                or self.voc_gate_q_temperature <= 0.0
            ):
                raise ValueError(
                    "voc_gate_q_temperature must be finite and positive"
                )
            if (
                not np.isfinite(self.voc_gate_learning_rate)
                or self.voc_gate_learning_rate <= 0.0
            ):
                raise ValueError(
                    "voc_gate_learning_rate must be finite and positive"
                )
            if (
                not np.isfinite(self.voc_gate_grad_norm_clipping)
                or self.voc_gate_grad_norm_clipping <= 0.0
            ):
                raise ValueError(
                    "voc_gate_grad_norm_clipping must be finite and positive"
                )
        self.voc_parent_checkpoint_sha256 = None
        self.voc_parent_imitation_data_signature = None
        self.voc_activation_real_step = -1
        self.voc_parent_checkpoint = None
        self.voc_control_origin = None
        self.voc_control_origin_legacy_defaulted = False
        self._voc_parent_ema_gate_state = None
        self._voc_parent_ema_gate_update_count = None
        if self.dynamic_voc_mode == "control" and not bool(
            getattr(flags, "ckp", False)
        ):
            # A new control run has two explicit, auditable origins.  With a
            # parent it is the existing strict shadow->control promotion.  In
            # the absence of both parent surfaces, it starts from the ActorNet
            # contract's equal zero Q head and learns Q/gate jointly from the
            # first on-policy batch.
            self.voc_activation_real_step = 0
            parent_path = util.resolve_voc_parent_checkpoint(flags)
            if self.voc_gate_exact_projection and parent_path:
                raise ValueError(
                    "voc_gate_exact_projection requires a fresh parent-free "
                    "control origin"
                )
            if parent_path:
                provenance = util.validate_voc_control_preload(
                    parent_path, flags=flags
                )
                resolved_parent_sha256 = getattr(
                    flags, "voc_resolved_parent_checkpoint_sha256", None
                )
                if (
                    not isinstance(resolved_parent_sha256, str)
                    or resolved_parent_sha256
                    != provenance["voc_parent_checkpoint_sha256"]
                ):
                    raise ValueError(
                        "VoC parent checkpoint changed between self-play load "
                        "and learner validation"
                    )
                self.voc_control_origin = provenance["voc_control_origin"]
                self.voc_parent_checkpoint = provenance[
                    "voc_parent_checkpoint"
                ]
                self.voc_parent_checkpoint_sha256 = provenance[
                    "voc_parent_checkpoint_sha256"
                ]
                self.voc_parent_imitation_data_signature = provenance[
                    "imitation_data_signature"
                ]
                self._voc_parent_ema_gate_state = provenance[
                    "voc_ema_gate_head_state_dict"
                ]
                self._voc_parent_ema_gate_update_count = provenance[
                    "voc_ema_gate_update_count"
                ]
            else:
                self.voc_control_origin = util.VOC_CONTROL_ORIGIN_FRESH
        learner_seed = int(getattr(flags, "base_seed", 0)) + 1_000_003
        random.seed(learner_seed)
        np.random.seed(learner_seed % (2**32))
        torch.manual_seed(learner_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(learner_seed)

        if flags.parallel_actor:
            self.actor_buffer = ray_obj["actor_buffer"]
            self.actor_param_buffer = ray_obj["actor_param_buffer"]
            self.model_param_buffer = ray_obj.get("model_param_buffer")
            self.actor_net = ActorNet(**actor_param)
            self.refresh_actor()
            self.actor_net.train(True)
            if self.flags.gpu_learn_actor > 0. and torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:           
                self.device = torch.device("cpu")
        else:
            assert actor_net is not None, "actor_net is required for non-parallel mode"
            assert device is not None, "device is required for non-parallel mode"
            self.actor_net = actor_net
            self.device = device
            self.model_param_buffer = None

        self.runtime_action_meanings = (
            None
            if runtime_action_meanings is None
            else tuple(runtime_action_meanings)
        )
        self.imitation_noop_action_index = resolve_noop_action_index(
            self.runtime_action_meanings,
            int(self.actor_net.num_actions),
        )

        if self.device == torch.device("cuda"):
            self._logger.info("Init. actor-learning: Using CUDA.")
        else:
            self._logger.info("Init. actor-learning: Not using CUDA.")

       # initialize learning setting

        self.voc_head_modules = self._find_voc_head_modules(self.actor_net)
        self.voc_parameters = self._find_voc_head_parameters(self.actor_net)
        self.voc_gate_head_modules = self._find_voc_gate_head_modules(
            self.actor_net
        )
        self.voc_gate_parameters = self._find_voc_gate_head_parameters(
            self.actor_net
        )
        if self.dynamic_voc_mode != "off" and not self.voc_parameters:
            raise RuntimeError(
                "VoC mode is enabled but ActorNet has no voc_head parameters"
            )
        if self.voc_dedicated_gate:
            if len(self.voc_gate_head_modules) != 1:
                raise RuntimeError(
                    "dedicated VoC control requires exactly one "
                    f"voc_gate_head module; found {len(self.voc_gate_head_modules)}"
                )
            if not self.voc_gate_parameters:
                raise RuntimeError(
                    "dedicated VoC control has no voc_gate_head parameters"
                )
        elif self.voc_gate_parameters:
            raise RuntimeError(
                "ActorNet exposes voc_gate_head parameters while the dedicated "
                "gate protocol is disabled"
            )
        self.voc_online_head = None
        self.voc_ema_gate_weight = None
        self.voc_ema_gate_bias = None
        self.voc_ema_gate_update_count = 0
        self.voc_ema_gate_parent_update_count = 0
        if self.dynamic_voc_mode != "off":
            if not self.voc_ema_gate_target:
                raise ValueError(
                    "active VoC requires voc_ema_gate_target=true"
                )
            if not 0.0 < self.voc_gate_target_tau <= 1.0:
                raise ValueError(
                    "active VoC requires 0 < voc_gate_target_tau <= 1"
                )
            if len(self.voc_head_modules) != 1:
                raise RuntimeError(
                    "EMA-gated VoC requires exactly one voc_head module; "
                    f"found {len(self.voc_head_modules)}"
                )
            self.voc_online_head = self.voc_head_modules[0]
            if not isinstance(self.voc_online_head, torch.nn.Linear):
                raise TypeError("EMA-gated VoC requires a linear voc_head")
            # Plain frozen FP32 tensors avoid module-construction RNG and keep
            # the target out of both optimizers and ActorNet/self-play state.
            self.voc_ema_gate_weight = (
                self.voc_online_head.weight.detach().float().clone()
            )
            self.voc_ema_gate_bias = (
                self.voc_online_head.bias.detach().float().clone()
            )
            if self._voc_parent_ema_gate_state is not None:
                self._load_voc_ema_gate_state(
                    self._voc_parent_ema_gate_state,
                    self._voc_parent_ema_gate_update_count,
                    parent_update_count=(
                        self._voc_parent_ema_gate_update_count
                    ),
                )
        if self.voc_control_origin == util.VOC_CONTROL_ORIGIN_FRESH:
            self._require_fresh_voc_zero_initialization(self.voc_parameters)
            if self.voc_dedicated_gate:
                self._require_fresh_voc_gate_zero_initialization(
                    self.voc_gate_parameters
                )
        elif self.dynamic_voc_mode == "shadow" and self.voc_dedicated_gate:
            self._require_fresh_voc_gate_zero_initialization(
                self.voc_gate_parameters
            )
        voc_parameter_ids = {id(parameter) for parameter in self.voc_parameters}
        voc_gate_parameter_ids = {
            id(parameter) for parameter in self.voc_gate_parameters
        }
        if not voc_parameter_ids.isdisjoint(voc_gate_parameter_ids):
            raise RuntimeError(
                "VoC critic and dedicated gate parameter sets must be disjoint"
            )
        actor_parameters = [
            parameter
            for parameter in self.actor_net.parameters()
            if id(parameter) not in voc_parameter_ids
            and id(parameter) not in voc_gate_parameter_ids
        ]
        self.optimizer = self._make_optimizer(
            actor_parameters, float(flags.actor_learning_rate)
        )
        self.voc_optimizer = None
        self.voc_gate_optimizer = None
        if self.dynamic_voc_mode != "off":
            self.voc_optimizer = self._make_voc_optimizer(
                self.voc_parameters, float(flags.actor_learning_rate)
            )
        if self.voc_dedicated_gate:
            self.voc_gate_optimizer = self._make_voc_gate_optimizer(
                self.voc_gate_parameters
            )
            if self.voc_gate_exact_projection:
                for parameter in self.voc_gate_parameters:
                    parameter.requires_grad_(False)
        active_optimizers = [self.optimizer]
        if self.voc_optimizer is not None:
            active_optimizers.append(self.voc_optimizer)
        if self.voc_gate_optimizer is not None:
            active_optimizers.append(self.voc_gate_optimizer)
        optimizer_parameter_sets = [
            {
                id(parameter)
                for group in optimizer.param_groups
                for parameter in group["params"]
            }
            for optimizer in active_optimizers
        ]
        for left_index, left in enumerate(optimizer_parameter_sets):
            for right in optimizer_parameter_sets[left_index + 1:]:
                if not left.isdisjoint(right):
                    raise RuntimeError(
                        "actor, VoC critic and dedicated gate optimizers must "
                        "be pairwise parameter-disjoint"
                    )
        if self.dynamic_voc_mode != "off":
            optimizer_parameter_ids = {
                id(parameter)
                for optimizer in active_optimizers
                for group in optimizer.param_groups
                for parameter in group["params"]
            }
            if (
                self.voc_ema_gate_weight.requires_grad
                or self.voc_ema_gate_bias.requires_grad
                or id(self.voc_ema_gate_weight) in optimizer_parameter_ids
                or id(self.voc_ema_gate_bias) in optimizer_parameter_ids
            ):
                raise RuntimeError(
                    "VoC EMA gate target must be frozen and optimizer-disjoint"
                )

        self.step = 0
        self.tot_eps = 0
        self.real_step = 0
        self.voc_update_count = 0
        self.voc_gate_update_count = 0
        self.voc_continue_count = 0
        self.voc_stop_count = 0
        self.voc_holdout_count = 0
        self.voc_holdout_continue_count = 0
        self.voc_holdout_stop_count = 0
        self.voc_holdout_td_sum = 0.0
        self.voc_holdout_td_abs_sum = 0.0
        self.voc_holdout_td_sq_sum = 0.0
        self._voc_pending_update = False
        self._pending_voc_continue_count = 0
        self._pending_voc_stop_count = 0
        self._pending_voc_holdout = None

        lr_lambda = lambda epoch: 1.0 - util.schedule_progress(self.flags, epoch)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self.voc_scheduler = None
        if self.voc_optimizer is not None:
            self.voc_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.voc_optimizer, lr_lambda
            )
        self.voc_gate_scheduler = None
        if self.voc_gate_optimizer is not None:
            self.voc_gate_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.voc_gate_optimizer, lr_lambda
            )

        # other init. variables for consume_data
        max_actor_id = (
            self.flags.self_play_n * self.flags.env_n
        )
        self.ret_buffers = {"re": RetBuffer(max_actor_id, mean_n=400)}
        if self.flags.im_cost > 0.:
            self.ret_buffers["im"] = RetBuffer(max_actor_id, mean_n=20000)
        if self.flags.cur_cost > 0.:
            self.ret_buffers["cur"] = RetBuffer(max_actor_id, mean_n=400)
        if self.dynamic_search:
            self.ret_buffers["think"] = RetBuffer(max_actor_id, mean_n=20000)
        self.ret_buffers["len"] = RetBuffer(max_actor_id, mean_n=400)
        self.im_discounting = self.flags.discounting ** (1 / self.flags.rec_t)

        self.rewards_ls = util.get_reward_names(flags)
        self.num_rewards = len(self.rewards_ls)

        if self.flags.return_norm_type in [0, 1]:
            self.norm_stats = [(None, None, None, util.FifoBuffer(100000 * self.flags.ppo_k, device=self.device),) for _ in range(self.num_rewards)] 
        else:
            self.norm_stats = [None,] * self.num_rewards
        self.anneal_c = 1
        self.n = 0

        # 버퍼 크기 설정 (오래된 데이터 저장 관련 코드 제거)
        self.buffer_save_size = getattr(self.flags, 'buffer_save_size', 1000)  # 기본값 1000

        self.crnorm = None

        # Checkpoint fields must exist before load_checkpoint.  Dataset/model
        # construction happens only after the actor and optimizer are moved to
        # their process device below.
        self.imitation_enabled = bool(
            str(getattr(self.flags, "icopro_data_path", "")).strip()
        )
        self.imitation_update_count = 0
        self.imitation_schedule_step = 0
        self._imitation_pending_update = False
        self.imitation_data_signature = None
        self.imitation_data_root = None
        self._checkpoint_imitation_data_signature = None
        self._checkpoint_imitation_rng_state = None
        self._checkpoint_python_rng_state = None
        self._checkpoint_numpy_rng_state = None
        self._checkpoint_torch_rng_state = None
        self._checkpoint_cuda_rng_state = None
        self._checkpoint_scaler_state = None
        self._checkpoint_voc_scaler_state = None
        self._checkpoint_voc_gate_scaler_state = None
        self.actor_amp_skip_count = 0
        self.actor_amp_consecutive_skips = 0
        self.voc_amp_skip_count = 0
        self.voc_amp_consecutive_skips = 0
        self.voc_gate_amp_skip_count = 0
        self.voc_gate_amp_consecutive_skips = 0
        self.voc_actor_policy_version = -1
        self.voc_actor_policy_publication_count = 0
        self.voc_actor_policy_terminal = False
        self.voc_actor_policy_version_mismatch_count = 0
        self.voc_actor_policy_malformed_bundle_count = 0
        self.voc_actor_policy_barrier_timeout_count = getattr(
            self, "voc_actor_policy_barrier_timeout_count", 0
        )
        self.voc_actor_policy_terminal_ack_count = 0
        self.voc_actor_policy_expected_ack_count = (
            int(getattr(self.flags, "self_play_n", 0))
            if self.voc_actor_policy_barrier_runtime else 0
        )
        self.voc_actor_policy_state_sha256 = ""
        self.voc_actor_policy_publication_history = []
        self.voc_actor_policy_publication_history_sha256 = ""
        self._voc_actor_policy_bundle = None
        self._voc_actor_policy_checkpoint_pending = False
        self._voc_actor_policy_checkpoint_force = False
        self._voc_actor_policy_transaction_open = False
        self._last_voc_gate_exact_projection_applied = False
        self._last_voc_gate_projection_pre_error_norm = 0.0
        self._last_voc_gate_projection_post_error_norm = 0.0
        self._last_voc_gate_postclip_total_norm = 0.0
        self._last_actor_gradient_step = ActorGradientStepResult(
            total_norm=0.0,
            optimizer_stepped=True,
            amp_scale_before=None,
            amp_scale_after=None,
            nonfinite_gradient_names=(),
        )
        self._last_voc_gradient_step = ActorGradientStepResult(
            total_norm=0.0,
            optimizer_stepped=(self.dynamic_voc_mode == "off"),
            amp_scale_before=None,
            amp_scale_after=None,
            nonfinite_gradient_names=(),
        )
        self._last_voc_gate_gradient_step = ActorGradientStepResult(
            total_norm=0.0,
            optimizer_stepped=(not self.voc_dedicated_gate),
            amp_scale_before=None,
            amp_scale_after=None,
            nonfinite_gradient_names=(),
        )
        self.action_prior = None
        self.action_prior_ema = None
        self._pending_action_prior_ema = None
        self.bc_loader = None
        self.bc_model_net = None
        self.bc_runner = None
        self._imitation_contract_validated = False

        self.ckp_path = os.path.join(flags.ckpdir, "ckp_actor.tar")
        if flags.ckp: self.load_checkpoint(self.ckp_path)

        # initialize file logs
        self.plogger = FileWriter(
            xpid=flags.xpid,
            xp_args=flags.__dict__,
            rootdir=flags.savedir,
            overwrite=not self.flags.ckp,
        )
        if self._voc_telemetry_active:
            self._init_schema13_telemetry()

        # move network and optimizer to process device
        self.actor_net.to(self.device)
        if self.voc_ema_gate_weight is not None:
            self.voc_ema_gate_weight = self.voc_ema_gate_weight.to(
                self.device, dtype=torch.float32
            )
            self.voc_ema_gate_bias = self.voc_ema_gate_bias.to(
                self.device, dtype=torch.float32
            )
        util.optimizer_to(self.optimizer, self.device)
        if self.voc_optimizer is not None:
            util.optimizer_to(self.voc_optimizer, self.device)
        if self.voc_gate_optimizer is not None:
            util.optimizer_to(self.voc_gate_optimizer, self.device)
        self._init_imitation_components()
        if (
            self.dynamic_voc_mode == "control"
            and self.voc_parent_imitation_data_signature is not None
            and self.imitation_data_signature is not None
            and self.voc_parent_imitation_data_signature
            != self.imitation_data_signature
        ):
            raise ValueError(
                "VoC shadow parent behavioral-data signature differs from "
                "the control run"
            )

        # variables for timing
        self.queue_n = 0
        self.timer = timeit.default_timer
        self.start_time = self.timer()
        self.sps_buffer = [(self.step, self.start_time)] * 36
        self.sps = 0
        self.sps_buffer_n = 0
        self.sps_start_time, self.sps_start_step = self.start_time, self.step
        self.ckp_start_time = int(time.strftime("%M")) // 10
        self.disable_thinker = flags.wrapper_type == 1

         # autotune
        self.autotune = flags.autotune
        if self.autotune:
            assert self.actor_net.discrete_action, "auto support discrete action set at the moment"
            self.tar_entropy = -flags.tar_entropy_scale * torch.log(1 / torch.tensor(self.actor_net.num_actions * self.actor_net.dim_actions))
            self.tar_entropy = self.tar_entropy.item()
            if not self.disable_thinker:
                if self.dynamic_search:
                    # The hierarchical SEARCH policy has one STOP outcome
                    # and A**D primary-action outcomes under each of PROCEED
                    # and RESET.  Its maximum joint entropy is therefore
                    # log(1 + 2*A**D), matching the conditional entropy used
                    # by ActorNet rather than the unattainable legacy
                    # log(3) + log(A*D) target.
                    primary_outcome_n = (
                        self.actor_net.num_actions
                        ** self.actor_net.dim_actions
                    )
                    self.tar_im_entropy = (
                        flags.tar_im_entropy_scale
                        * torch.log(torch.tensor(
                            1 + 2 * primary_outcome_n,
                            dtype=torch.float,
                        ))
                    ).item()
                else:
                    self.tar_im_entropy = -flags.tar_im_entropy_scale * torch.log(1 / torch.tensor(self.actor_net.num_actions * self.actor_net.dim_actions))
                    self.tar_im_entropy += -flags.tar_im_entropy_scale * torch.log(1 / torch.tensor(2))
                    self.tar_im_entropy = self.tar_im_entropy.item()

        self.voc_scaler = None
        self.voc_gate_scaler = None
        if self.flags.float16:
            self.scaler = GradScaler(init_scale=self.actor_amp_init_scale)
            if self.voc_optimizer is not None:
                self.voc_scaler = GradScaler(init_scale=2**8)
            if self.voc_gate_optimizer is not None:
                self.voc_gate_scaler = GradScaler(init_scale=2**8)

        self.ppo_enable = self.flags.ppo_k > 1
        if self.ppo_enable:
            self.ppo_n = self.flags.ppo_n
            self.ppo_k = self.flags.ppo_k
            self.ppo_b = self.flags.actor_batch_size
            if not self.flags.ppo_syn:
                assert (self.ppo_n > self.ppo_k and self.ppo_n % self.ppo_k == 0) or (
                    self.ppo_n < self.ppo_k and self.ppo_k % self.ppo_n == 0) or (
                    self.ppo_n == self.ppo_k
                    ), "ppo_k and ppo_n should be divisible"
                self.ppo_update_freq = 1 if self.ppo_k >= self.ppo_n else self.ppo_n // self.ppo_k
                self.ppo_update_time = 1 if self.ppo_n >= self.ppo_k else self.ppo_k // self.ppo_n
            else:
                self.ppo_update_freq = self.ppo_n
                self.ppo_update_time = self.ppo_k
            self.ppo_t = 0
            self.ppo_buffer = None
            self.ppo_buffer_n = self.ppo_n * self.ppo_b
            self.kl_losses = collections.deque(maxlen=100)
            self.ppo_is_abs = collections.deque(maxlen=100)
        self.dbg_adv = collections.deque(maxlen=100)
        self.dbg_start_time = self.timer()
        if self.flags.float16 and self._checkpoint_scaler_state is not None:
            self.scaler.load_state_dict(self._checkpoint_scaler_state)
        if (
            self.flags.float16
            and self.voc_optimizer is not None
            and self._checkpoint_voc_scaler_state is not None
        ):
            self.voc_scaler.load_state_dict(self._checkpoint_voc_scaler_state)
        if (
            self.flags.float16
            and self.voc_gate_optimizer is not None
            and self._checkpoint_voc_gate_scaler_state is not None
        ):
            self.voc_gate_scaler.load_state_dict(
                self._checkpoint_voc_gate_scaler_state
            )
        if self.voc_gate_exact_projection:
            self._assert_voc_gate_exact_projection_invariant()
        self._restore_training_rng_state()

    @staticmethod
    def _find_voc_head_modules(actor_net):
        """Return nested modules whose final path segment is ``voc_head``."""

        return [
            module
            for module_name, module in actor_net.named_modules()
            if module_name.split(".")[-1] == "voc_head"
        ]

    @staticmethod
    def _find_voc_head_parameters(actor_net):
        """Collect any nested module whose final path segment is voc_head."""

        parameters = []
        seen = set()
        for module_name, module in actor_net.named_modules():
            if module_name.split(".")[-1] != "voc_head":
                continue
            for parameter in module.parameters():
                if id(parameter) not in seen:
                    parameters.append(parameter)
                    seen.add(id(parameter))
        return parameters

    @staticmethod
    def _find_voc_gate_head_modules(actor_net):
        """Return nested modules whose final path is ``voc_gate_head``."""

        return [
            module
            for module_name, module in actor_net.named_modules()
            if module_name.split(".")[-1] == "voc_gate_head"
        ]

    @staticmethod
    def _find_voc_gate_head_parameters(actor_net):
        """Collect the dedicated gate parameters without duplicates."""

        parameters = []
        seen = set()
        for module in SActorLearner._find_voc_gate_head_modules(actor_net):
            for parameter in module.parameters():
                if id(parameter) not in seen:
                    parameters.append(parameter)
                    seen.add(id(parameter))
        return parameters

    def _load_voc_ema_gate_state(
        self, state, update_count, *, parent_update_count=None
    ):
        """Install a validated frozen EMA head without touching RNG state."""

        if self.voc_online_head is None:
            raise RuntimeError("cannot load an EMA gate target in VoC off mode")
        if not isinstance(state, Mapping) or set(state) != {"weight", "bias"}:
            raise ValueError(
                "VoC EMA gate state must contain exactly weight and bias"
            )
        if (
            isinstance(update_count, (bool, np.bool_))
            or not isinstance(update_count, (int, np.integer))
            or int(update_count) < 0
        ):
            raise ValueError("VoC EMA gate update count must be non-negative")
        weight = torch.as_tensor(state["weight"]).detach().float().clone()
        bias = torch.as_tensor(state["bias"]).detach().float().clone()
        if tuple(weight.shape) != tuple(self.voc_online_head.weight.shape):
            raise ValueError("VoC EMA gate weight shape does not match voc_head")
        if tuple(bias.shape) != tuple(self.voc_online_head.bias.shape):
            raise ValueError("VoC EMA gate bias shape does not match voc_head")
        _require_finite_tensor("VoC EMA gate weight", weight)
        _require_finite_tensor("VoC EMA gate bias", bias)
        device = self.voc_online_head.weight.device
        self.voc_ema_gate_weight = weight.to(device=device)
        self.voc_ema_gate_bias = bias.to(device=device)
        self.voc_ema_gate_update_count = int(update_count)
        if parent_update_count is not None:
            if (
                isinstance(parent_update_count, (bool, np.bool_))
                or not isinstance(parent_update_count, (int, np.integer))
                or int(parent_update_count) < 0
                or int(parent_update_count) > int(update_count)
            ):
                raise ValueError(
                    "VoC EMA parent update count must be within lifetime count"
                )
            self.voc_ema_gate_parent_update_count = int(parent_update_count)

    def _voc_ema_gate_state_dict(self):
        if self.voc_ema_gate_weight is None:
            return None
        return {
            "weight": self.voc_ema_gate_weight.detach().cpu().clone(),
            "bias": self.voc_ema_gate_bias.detach().cpu().clone(),
        }

    def _compute_ema_gate_loss(
        self, *, features, logits, valid, state_value,
        enable_policy_loss=True,
    ):
        """Compute exact soft gate loss from the batch-start frozen EMA Q.

        The target head and shared critic representation/base are detached.
        Only the current stochastic gate probabilities retain gradients.
        """

        if self.voc_ema_gate_weight is None:
            raise RuntimeError("active VoC has no EMA gate target")
        valid = valid.bool()
        expected_feature_shape = tuple(valid.shape) + (
            self.voc_ema_gate_weight.shape[1],
        )
        if tuple(features.shape) != expected_feature_shape:
            raise ValueError(
                "VoC EMA features must have shape "
                f"{expected_feature_shape}, got {tuple(features.shape)}"
            )
        if features.requires_grad:
            raise RuntimeError("VoC EMA features must be detached")
        if tuple(logits.shape) != tuple(valid.shape) + (3,):
            raise ValueError("VoC EMA logits must end in three controls")
        if tuple(state_value.shape) != tuple(valid.shape):
            raise ValueError("VoC EMA state value must match control mask")
        _require_finite_tensor("VoC EMA features", features[valid])
        _require_finite_tensor("VoC EMA target weight", self.voc_ema_gate_weight)
        _require_finite_tensor("VoC EMA target bias", self.voc_ema_gate_bias)

        entropy = compute_dynamic_control_entropy(
            logits, project_gate_gradient=False
        )
        gate_probabilities = torch.stack(
            (entropy.continue_prob, entropy.stop_prob), dim=-1
        )
        # Keep the master target and its inference in FP32 even under actor AMP.
        with torch.no_grad(), autocast(enabled=False):
            raw_advantage = F.linear(
                features.float().reshape(-1, features.shape[-1]),
                self.voc_ema_gate_weight,
                self.voc_ema_gate_bias,
            ).view(tuple(valid.shape) + (2,))
            centered_advantage = raw_advantage - torch.sum(
                gate_probabilities.detach().float() * raw_advantage,
                dim=-1,
                keepdim=True,
            )
            if getattr(self, "voc_gate_policy_schema_version", None) in (
                util.VOC_GATE_POLICY_COMMON_MODE_Q_SCHEMA_VERSION,
                util.VOC_GATE_POLICY_HUBER_COMMON_Q_SCHEMA_VERSION,
                util.VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION,
                util.VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION,
                util.VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION,
            ):
                common_advantage = raw_advantage.mean(
                    dim=-1, keepdim=True
                )
                q_values = (
                    state_value.detach().float().unsqueeze(-1)
                    + common_advantage
                    + centered_advantage
                )
            else:
                q_values = state_value.detach().float().unsqueeze(-1) + (
                    centered_advantage
                )
        _require_finite_tensor("VoC EMA gate Q", q_values[valid])
        if self.dynamic_voc_mode == "control" and enable_policy_loss:
            gate_loss = -torch.sum(
                gate_probabilities.float()
                * q_values
                * valid.unsqueeze(-1).float()
            )
        else:
            gate_loss = logits.sum() * 0.0
        _require_finite_tensor("VoC EMA gate loss", gate_loss)
        return gate_loss, q_values

    def _update_voc_ema_gate_target(self):
        """Atomically Polyak-update the FP32 target after a successful Q step."""

        if self.voc_online_head is None or self.voc_ema_gate_weight is None:
            raise RuntimeError("cannot update a missing VoC EMA gate target")
        tau = self.voc_gate_target_tau
        with torch.no_grad():
            online_weight = self.voc_online_head.weight.detach().float()
            online_bias = self.voc_online_head.bias.detach().float()
            _require_finite_tensor("post-step online VoC weight", online_weight)
            _require_finite_tensor("post-step online VoC bias", online_bias)
            _require_finite_tensor(
                "pre-step VoC EMA gate weight", self.voc_ema_gate_weight
            )
            _require_finite_tensor(
                "pre-step VoC EMA gate bias", self.voc_ema_gate_bias
            )
            candidate_weight = (
                (1.0 - tau) * self.voc_ema_gate_weight
                + tau * online_weight
            )
            candidate_bias = (
                (1.0 - tau) * self.voc_ema_gate_bias + tau * online_bias
            )
            # Validate both candidates before mutating either target tensor.
            _require_finite_tensor("updated VoC EMA gate weight", candidate_weight)
            _require_finite_tensor("updated VoC EMA gate bias", candidate_bias)
            if (
                getattr(self, "voc_gate_policy_schema_version", None)
                in (
                    util.VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION,
                    util.VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION,
                )
            ):
                if not torch.equal(candidate_weight, online_weight):
                    raise RuntimeError(
                        "schema-12 VoC EMA weight candidate must be "
                        "torch.equal to the post-step online weight"
                    )
                if not torch.equal(candidate_bias, online_bias):
                    raise RuntimeError(
                        "schema-12 VoC EMA bias candidate must be "
                        "torch.equal to the post-step online bias"
                    )
            self.voc_ema_gate_weight.copy_(candidate_weight)
            self.voc_ema_gate_bias.copy_(candidate_bias)
            self.voc_ema_gate_update_count += 1

    def _assert_voc_gate_exact_projection_invariant(
        self, *, require_count_lockstep=True
    ):
        """Fail closed if the deterministic gate update rule has drifted."""

        if not self.voc_gate_exact_projection:
            return
        if self.voc_gate_optimizer is None or self.voc_gate_scheduler is None:
            raise RuntimeError("VoC exact projection lacks gate state objects")
        if any(
            parameter.requires_grad or parameter.grad is not None
            for parameter in self.voc_gate_parameters
        ):
            raise RuntimeError(
                "VoC exact projection requires frozen, gradient-free gate "
                "parameters"
            )
        if self.voc_gate_optimizer.state:
            raise RuntimeError(
                "VoC exact projection requires empty gate optimizer state"
            )
        gate_head = self.voc_gate_head_modules[0]
        for name, value in (
            ("gate weight", gate_head.weight),
            ("gate bias", gate_head.bias),
            ("EMA Q weight", self.voc_ema_gate_weight),
            ("EMA Q bias", self.voc_ema_gate_bias),
        ):
            if value.dtype != torch.float32:
                raise RuntimeError(
                    f"VoC exact projection requires FP32 {name}"
                )
        scheduler_state = self.voc_gate_scheduler.state_dict()
        if (
            int(scheduler_state.get("last_epoch", -1)) != 0
            or int(scheduler_state.get("_step_count", -1)) != 1
        ):
            raise RuntimeError(
                "VoC exact projection requires a pristine gate scheduler"
            )
        if self.voc_gate_scaler is not None:
            scaler_state = self.voc_gate_scaler.state_dict()
            if (
                float(scaler_state.get("scale", -1.0)) != 256.0
                or int(scaler_state.get("_growth_tracker", -1)) != 0
            ):
                raise RuntimeError(
                    "VoC exact projection requires a pristine gate GradScaler"
                )
        if self.voc_ema_gate_parent_update_count != 0:
            raise RuntimeError(
                "VoC exact projection requires zero parent EMA updates"
            )
        if self.voc_gate_update_count != self.voc_ema_gate_update_count:
            raise RuntimeError(
                "VoC exact projection count disagrees with EMA updates"
            )
        if require_count_lockstep and self.voc_gate_update_count != (
            self.voc_update_count
        ):
            raise RuntimeError(
                "VoC exact projection count disagrees with successful Q "
                "updates"
            )
        if self.voc_gate_update_count == 0:
            if (
                torch.count_nonzero(gate_head.weight).item() != 0
                or torch.count_nonzero(gate_head.bias).item() != 0
            ):
                raise RuntimeError(
                    "VoC zero-count exact-projection gate must equal zero"
                )
        alignment = compute_dynamic_voc_gate_parameter_alignment_loss(
            gate_weight=gate_head.weight,
            gate_bias=gate_head.bias,
            ema_q_weight=self.voc_ema_gate_weight,
            ema_q_bias=self.voc_ema_gate_bias,
            q_temperature=self.voc_gate_q_temperature,
            policy_temperature=float(self.flags.voc_gate_temperature),
        )
        if (
            not torch.equal(gate_head.weight, alignment.target_weight)
            or not torch.equal(gate_head.bias, alignment.target_bias)
            or alignment.parameter_error_norm.item() != 0.0
        ):
            raise RuntimeError(
                "VoC exact-projection gate disagrees with EMA Q target"
            )

    def _project_voc_gate_head_to_ema_target(self):
        """Apply one post-EMA projection without touching optimizer state."""

        if not self.voc_gate_exact_projection:
            raise RuntimeError("VoC exact gate projection is disabled")
        if self.voc_gate_optimizer.state:
            raise RuntimeError(
                "VoC exact projection found stale gate optimizer state"
            )
        gate_head = self.voc_gate_head_modules[0]
        result = project_dynamic_voc_gate_head_exact_(
            gate_weight=gate_head.weight,
            gate_bias=gate_head.bias,
            ema_q_weight=self.voc_ema_gate_weight,
            ema_q_bias=self.voc_ema_gate_bias,
            q_temperature=self.voc_gate_q_temperature,
            policy_temperature=float(self.flags.voc_gate_temperature),
        )
        self.voc_gate_update_count += 1
        self._last_voc_gate_exact_projection_applied = True
        self._last_voc_gate_projection_pre_error_norm = float(
            result.pre_projection_error_norm.detach().cpu()
        )
        self._last_voc_gate_projection_post_error_norm = float(
            result.post_projection_error_norm.detach().cpu()
        )
        self._assert_voc_gate_exact_projection_invariant(
            require_count_lockstep=False
        )
        return result

    @staticmethod
    def _require_fresh_voc_zero_initialization(parameters):
        """Fail closed if a no-parent control run is not Q-neutral."""

        if not parameters:
            raise RuntimeError("Fresh VoC control has no voc_head parameters")
        for parameter in parameters:
            if not torch.isfinite(parameter).all().item():
                raise RuntimeError(
                    "Fresh VoC control requires finite voc_head parameters"
                )
            if torch.count_nonzero(parameter).item() != 0:
                raise RuntimeError(
                    "Fresh VoC control requires an equal zero-initialized "
                    "voc_head"
                )

    @staticmethod
    def _require_fresh_voc_gate_zero_initialization(parameters):
        """Require an RNG-neutral, probability-neutral fresh scalar gate."""

        if not parameters:
            raise RuntimeError("Fresh VoC control has no voc_gate_head parameters")
        for parameter in parameters:
            if not torch.isfinite(parameter).all().item():
                raise RuntimeError(
                    "Fresh VoC control requires finite voc_gate_head parameters"
                )
            if torch.count_nonzero(parameter).item() != 0:
                raise RuntimeError(
                    "Fresh VoC control requires an exactly zero-initialized "
                    "voc_gate_head"
                )

    def _make_optimizer(self, parameters, learning_rate):
        if not self.flags.actor_use_rms:
            return torch.optim.Adam(
                parameters,
                lr=learning_rate,
                eps=self.flags.actor_adam_eps,
            )
        return torch.optim.RMSprop(
            parameters,
            lr=learning_rate,
            momentum=0,
            eps=0.01,
            alpha=0.99,
        )

    def _make_voc_optimizer(self, parameters, learning_rate):
        """Select the schema-11/12 adapter without changing older schemas."""

        if self.voc_gate_policy_schema_version in (
            util.VOC_GATE_POLICY_ORTHOCD_ADAM_Q_SCHEMA_VERSION,
            util.VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION,
            util.VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION,
        ):
            if self.flags.actor_use_rms:
                raise ValueError("schema-11/12 VoC Q requires inherited Adam")
            return _VoCOrthoCDAdam(
                parameters,
                lr=learning_rate,
                eps=self.flags.actor_adam_eps,
            )
        return self._make_optimizer(parameters, learning_rate)

    def _make_voc_gate_optimizer(self, parameters):
        """Build the isolated gate optimizer without changing actor/Q rules."""

        if self.flags.actor_use_rms:
            # Preserve the legacy all-RMS optimizer selection exactly.
            return self._make_optimizer(
                parameters, self.voc_gate_learning_rate
            )
        return torch.optim.Adam(
            parameters,
            lr=self.voc_gate_learning_rate,
            eps=self.flags.actor_adam_eps,
            betas=(self.voc_gate_adam_beta1, 0.999),
        )

    @staticmethod
    def _parse_int_list(value, name):
        if isinstance(value, (list, tuple)):
            raw = value
        else:
            raw = str(value).split(",")
        try:
            parsed = tuple(int(item) for item in raw if str(item).strip())
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a comma-separated integer list") from exc
        if not parsed:
            raise ValueError(f"{name} cannot be empty")
        return parsed

    def _validate_resume_imitation_protocol(self, checkpoint_flags):
        """Prevent a resumed checkpoint from mixing two BC protocols."""

        if not isinstance(checkpoint_flags, Mapping):
            raise ValueError("imitation checkpoint lacks embedded training flags")
        id_fields = (
            "icopro_subjects",
            "icopro_train_sessions",
            "icopro_holdout_sessions",
        )
        for name in id_fields:
            saved = self._parse_int_list(checkpoint_flags.get(name), name)
            current = self._parse_int_list(getattr(self.flags, name), name)
            if saved != current:
                raise ValueError(
                    f"imitation resume protocol mismatch: {name}={saved}, "
                    f"expected {current}"
                )
        fields = (
            "name",
            "icopro_game_id",
            "frame_stack_n",
            "grayscale",
            "batch_length",
            "icopro_margin",
            "icopro_margin_coef",
            "icopro_action_diff_coef",
            "icopro_pvp_coef",
            "icopro_coef",
            "icopro_supervised_freq",
            "icopro_batch_size",
            "action_prior_weight",
            "action_prior_ema",
            "tree_carry",
            "rec_t",
            "max_search_steps",
            "max_depth",
            "model_unroll_len",
            "think_cost",
            "think_cost_anneal",
            "sep_im_head",
            "float16",
            "model_float16",
            "model_state_projection",
            "model_state_range_loss_cost",
            "dynamic_factorized_control",
            "schedule_total_steps",
            "actor_amp_max_consecutive_skips",
        )
        for name in fields:
            saved = checkpoint_flags.get(name)
            if name == "model_float16" and saved is None:
                saved = checkpoint_flags.get("float16")
            if name == "model_state_projection" and saved is None:
                saved = "none"
            if name == "model_state_range_loss_cost" and saved is None:
                saved = 0.0
            if name == "dynamic_factorized_control" and saved is None:
                saved = False
            if name == "schedule_total_steps" and saved is None:
                saved = checkpoint_flags.get("total_steps")
            if name == "actor_amp_max_consecutive_skips" and saved is None:
                saved = 8
            current = getattr(self.flags, name, None)
            if name == "model_state_projection" and current is None:
                current = "none"
            if name == "model_state_range_loss_cost" and current is None:
                current = 0.0
            if isinstance(current, float):
                matches = np.isclose(
                    float(saved), current, rtol=0.0, atol=1e-12
                ) if saved is not None else False
            else:
                matches = saved == current
            if not matches:
                raise ValueError(
                    "imitation resume protocol mismatch: "
                    f"{name}={saved!r}, expected {current!r}"
                )

    def _loader_rng_state(self):
        if self.bc_loader is None:
            return None
        if hasattr(self.bc_loader, "get_rng_state"):
            return self.bc_loader.get_rng_state()
        rng = getattr(self.bc_loader, "rng", None)
        if rng is not None and hasattr(rng, "bit_generator"):
            return copy.deepcopy(rng.bit_generator.state)
        return None

    def _restore_training_rng_state(self):
        if self._checkpoint_python_rng_state is not None:
            random.setstate(self._checkpoint_python_rng_state)
        if self._checkpoint_numpy_rng_state is not None:
            np.random.set_state(self._checkpoint_numpy_rng_state)
        if self._checkpoint_torch_rng_state is not None:
            torch.set_rng_state(self._checkpoint_torch_rng_state.cpu())
        if (
            self._checkpoint_cuda_rng_state is not None
            and torch.cuda.is_available()
        ):
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in self._checkpoint_cuda_rng_state]
            )

    def _restore_loader_rng_state(self, state):
        if self.bc_loader is None or state is None:
            return
        if hasattr(self.bc_loader, "set_rng_state"):
            self.bc_loader.set_rng_state(state)
            return
        rng = getattr(self.bc_loader, "rng", None)
        if rng is not None and hasattr(rng, "bit_generator"):
            rng.bit_generator.state = copy.deepcopy(state)

    def _make_imitation_data_signature(
            self, data_path, subjects, sessions, game_id, scored_length):
        del subjects, sessions, game_id, scored_length
        from thinker.bc_loader import behavioral_data_signature

        return behavioral_data_signature(self.bc_loader, data_path)

    def _init_imitation_components(self):
        """Create only the light dataset state; ModelNet/planner stay lazy."""
        if not self.imitation_enabled:
            if (
                self.imitation_update_count > 0
                or self._checkpoint_imitation_data_signature is not None
            ):
                raise ValueError(
                    "actor checkpoint contains Dynamic imitation state, but "
                    "icopro_data_path is empty; resume with the original "
                    "behavioral dataset instead of silently disabling imitation"
                )
            return
        if not self.dynamic_search:
            raise ValueError("icopro_data_path requires dynamic_search=true")
        if not bool(getattr(self.flags, "sep_im_head", False)):
            raise ValueError("dynamic imitation requires sep_im_head=true")
        if int(getattr(self.flags, "max_search_steps", -1)) <= 0:
            raise ValueError(
                "dynamic imitation requires positive max_search_steps as a watchdog"
            )
        if not self.actor_net.discrete_action or self.actor_net.dim_actions != 1:
            raise ValueError(
                "dynamic behavioral imitation currently supports one discrete "
                "primary action dimension"
            )
        frequency = int(getattr(self.flags, "icopro_supervised_freq", 1))
        if frequency <= 0:
            raise ValueError("icopro_supervised_freq must be a positive integer")

        from gymnasium import spaces
        from thinker.bc_loader import FrameStackedBehavioralDataLoader

        obs_space = getattr(self.actor_net, "online_real_state_space", None)
        if not isinstance(obs_space, spaces.Box):
            raise RuntimeError(
                "ActorNet did not retain its online real-state Box contract; "
                "rebuild the actor from the current online environment"
            )
        if not isinstance(self.actor_net.pri_action_space, spaces.Discrete):
            raise TypeError("dynamic imitation requires a Discrete primary action")
        if int(getattr(self.actor_net.pri_action_space, "start", 0)) != 0:
            raise ValueError("dynamic imitation requires zero-based Discrete actions")
        if int(self.actor_net.pri_action_space.n) != int(self.actor_net.num_actions):
            raise ValueError(
                "Actor action metadata disagree: "
                f"space={self.actor_net.pri_action_space.n}, "
                f"head={self.actor_net.num_actions}"
            )
        stack_n = int(self.flags.frame_stack_n)
        if len(obs_space.shape) != 3:
            raise ValueError(
                "behavioral Atari imitation requires CHW online observations, "
                f"got {obs_space.shape}"
            )
        channel_n = int(obs_space.shape[0])
        if stack_n <= 0 or channel_n % stack_n != 0:
            raise ValueError(
                "online observation channels must be divisible by frame_stack_n: "
                f"channels={channel_n}, frame_stack_n={stack_n}"
            )
        frame_ch = channel_n // stack_n
        expected_frame_ch = 1 if bool(self.flags.grayscale) else 3
        if frame_ch != expected_frame_ch:
            raise ValueError(
                "online frame channels disagree with grayscale: "
                f"channels_per_frame={frame_ch}, grayscale={self.flags.grayscale}"
            )
        dtype = np.dtype(obs_space.dtype)
        byte_contract = (
            dtype == np.dtype(np.uint8)
            and np.all(obs_space.low == 0)
            and np.all(obs_space.high == 255)
        )
        unit_float_contract = (
            dtype == np.dtype(np.float32)
            and np.all(obs_space.low == 0.0)
            and np.all(obs_space.high == 1.0)
        )
        if not byte_contract and not unit_float_contract:
            raise ValueError(
                "behavioral frame preprocessing supports only online uint8 "
                "[0,255] or float32 [0,1] observations"
            )
        self.imitation_obs_space = obs_space

        data_path = os.path.abspath(str(self.flags.icopro_data_path))
        self.imitation_data_root = os.path.realpath(data_path)
        subjects = self._parse_int_list(
            getattr(self.flags, "icopro_subjects", "1"), "icopro_subjects"
        )
        sessions = self._parse_int_list(
            getattr(self.flags, "icopro_train_sessions", "1,2,3"),
            "icopro_train_sessions",
        )
        holdout = set(self._parse_int_list(
            getattr(self.flags, "icopro_holdout_sessions", "4"),
            "icopro_holdout_sessions",
        ))
        overlap = set(sessions) & holdout
        if overlap:
            raise ValueError(
                "training and holdout behavioral sessions overlap: "
                f"{sorted(overlap)}"
            )
        scored_length = int(getattr(self.flags, "batch_length", 4))
        if scored_length <= 0:
            raise ValueError("batch_length must be the positive scored length")
        game_id = int(getattr(self.flags, "icopro_game_id", 0))
        self.bc_loader = FrameStackedBehavioralDataLoader(
            base_path=data_path,
            subjects=subjects,
            game_id=game_id,
            sessions=sessions,
            num_actions=int(self.actor_net.num_actions),
            scored_length=scored_length,
            frame_stack_n=stack_n,
            target_size=tuple(int(value) for value in obs_space.shape[-2:]),
            grayscale=bool(self.flags.grayscale),
            normalize=unit_float_contract,
            seed=int(getattr(self.flags, "base_seed", 0)),
        )
        if int(self.bc_loader.num_actions) != int(self.actor_net.num_actions):
            raise ValueError("behavior loader and Actor action counts disagree")
        if len(getattr(self.bc_loader, "data_files", ())) == 0:
            raise RuntimeError(
                f"no behavioral files found under {data_path} for sessions {sessions}"
            )
        self.imitation_data_signature = self._make_imitation_data_signature(
            data_path, subjects, sessions, game_id, scored_length
        )
        if (self._checkpoint_imitation_data_signature is not None
                and self._checkpoint_imitation_data_signature
                != self.imitation_data_signature):
            raise ValueError(
                "behavioral dataset signature differs from the actor checkpoint; "
                "refusing a non-deterministic imitation resume"
            )
        self._restore_loader_rng_state(self._checkpoint_imitation_rng_state)

        distribution = getattr(self.bc_loader, "action_distribution", None)
        if distribution is None and hasattr(self.bc_loader, "get_action_distribution"):
            distribution = self.bc_loader.get_action_distribution()
        if distribution is not None:
            distribution = np.asarray(distribution, dtype=np.float32)
            if distribution.shape != (self.actor_net.num_actions,):
                raise ValueError(
                    "human action prior has shape "
                    f"{distribution.shape}, expected {(self.actor_net.num_actions,)}"
                )
            if not np.all(np.isfinite(distribution)) or distribution.sum() <= 0:
                raise ValueError("human action prior is empty or non-finite")
            distribution = distribution / distribution.sum()
            self.action_prior = torch.tensor(
                distribution, device=self.device, dtype=torch.float32
            )
        if self.action_prior_ema is not None:
            self.action_prior_ema = self.action_prior_ema.to(self.device)
            if tuple(self.action_prior_ema.shape) != (self.actor_net.num_actions,):
                raise ValueError(
                    "checkpoint action-prior EMA has shape "
                    f"{tuple(self.action_prior_ema.shape)}, expected "
                    f"{(self.actor_net.num_actions,)}"
                )
            _require_finite_tensor(
                "checkpoint action-prior EMA", self.action_prior_ema
            )
        self._logger.info(
            "Dynamic imitation enabled: subjects=%s train_sessions=%s "
            "holdout_sessions=%s scored_length=%d frequency=%d",
            subjects,
            sessions,
            tuple(sorted(holdout)),
            scored_length,
            frequency,
        )

    def _refresh_imitation_model(self, require_weights=False):
        weights = None
        if self.model_param_buffer is not None:
            deadline = time.monotonic() + 120.0
            while weights is None:
                weights = ray.get(
                    self.model_param_buffer.get_data.remote("model_net")
                )
                if weights is not None or not require_weights:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "timed out after 120 seconds waiting for the initial "
                        "ModelNet weights needed by Dynamic imitation"
                    )
                time.sleep(0.1)
        if weights is None:
            candidate_paths = []
            if getattr(self.flags, "ckp", False):
                candidate_paths.append(os.path.join(self.flags.ckpdir, "ckp_model.tar"))
            preload = str(getattr(self.flags, "preload", ""))
            if preload:
                candidate_paths.append(os.path.join(preload, "ckp_model.tar"))
            for path in candidate_paths:
                if os.path.exists(path):
                    checkpoint = torch.load(
                        path, map_location=torch.device("cpu"), weights_only=False
                    )
                    weights = checkpoint["model_net_state_dict"]
                    break
        if weights is None:
            if require_weights:
                raise RuntimeError(
                    "dynamic imitation could not obtain current ModelNet weights"
                )
            return False
        _validate_model_state_dict_compatibility(
            self.bc_model_net, weights, label="behavioral ModelNet"
        )
        self.bc_model_net.set_weights(weights)
        self.bc_model_net.eval()
        for parameter in self.bc_model_net.parameters():
            parameter.requires_grad_(False)
        return True

    def _ensure_imitation_runner(self):
        if self.bc_runner is not None:
            self._refresh_imitation_model(require_weights=True)
            return
        from thinker.model_net import ModelNet

        device_name = str(getattr(self.flags, "icopro_device", "cpu"))
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"icopro_device={device_name!r}, but CUDA is unavailable"
            )
        model_device = torch.device(device_name)
        obs_space = self.imitation_obs_space
        fork_devices = []
        if model_device.type == "cuda":
            fork_devices = [
                model_device.index
                if model_device.index is not None
                else torch.cuda.current_device()
            ]
        # Network initialization is discarded immediately when authoritative
        # online weights are loaded.  Preserve the Actor sampling RNG so a
        # resumed run's next search/control draw is not shifted by lazy setup.
        with torch.random.fork_rng(devices=fork_devices):
            self.bc_model_net = ModelNet(
                obs_space=obs_space,
                action_space=self.actor_net.pri_action_space,
                flags=self.flags,
                frame_stack_n=int(self.flags.frame_stack_n),
            ).to(model_device)
        if int(self.bc_model_net.num_actions) != int(self.actor_net.num_actions):
            raise ValueError("behavioral ModelNet and Actor action counts disagree")
        if int(self.bc_model_net.frame_stack_n) != int(self.flags.frame_stack_n):
            raise ValueError("behavioral ModelNet frame-stack metadata disagree")
        if not _box_contract_equal(
            self.bc_model_net.observation_space, self.imitation_obs_space
        ):
            raise ValueError(
                "behavioral ModelNet observation contract differs from Actor's "
                "online environment"
            )
        self._refresh_imitation_model(require_weights=True)
        self.bc_runner = DynamicImitationRunner(
            self.actor_net,
            self.bc_model_net,
            self.flags,
            device=self.device,
            noop_action_index=self.imitation_noop_action_index,
        )

    def _sample_imitation_batch(self):
        scored_length = int(getattr(self.flags, "batch_length", 4))
        batch = self.bc_loader.get_sequence_batch(
            batch_size=int(getattr(self.flags, "icopro_batch_size", 32)),
            sequence_length=scored_length,
        )
        if not self._imitation_contract_validated:
            self._validate_imitation_batch_contract(batch)
            self._imitation_contract_validated = True
        return batch

    def _validate_imitation_batch_contract(self, batch):
        from thinker.dynamic_imitation import validate_behavior_batch

        batch_size, edge_count = validate_behavior_batch(batch)
        expected_edges = int(getattr(self.flags, "batch_length", 4)) + 1
        if edge_count != expected_edges:
            raise ValueError(
                f"behavior batch has {edge_count} edges, expected {expected_edges}"
            )
        observations = np.asarray(batch["obs_seq"])
        if tuple(observations.shape[2:]) != tuple(self.imitation_obs_space.shape):
            raise ValueError(
                "behavior/online observation shape mismatch: "
                f"behavior={observations.shape[2:]}, "
                f"online={self.imitation_obs_space.shape}"
            )
        if np.dtype(observations.dtype) != np.dtype(self.imitation_obs_space.dtype):
            raise TypeError(
                "behavior/online observation dtype mismatch: "
                f"behavior={observations.dtype}, "
                f"online={self.imitation_obs_space.dtype}"
            )
        low = np.asarray(self.imitation_obs_space.low)
        high = np.asarray(self.imitation_obs_space.high)
        if not np.all(np.isfinite(observations)):
            raise ValueError("behavior observations contain non-finite values")
        if np.any(observations < low) or np.any(observations > high):
            raise ValueError("behavior observations exceed the online Box bounds")
        actions = np.asarray(batch["actions_seq"])
        initial = np.asarray(batch["initial_prev_action"])
        action_n = int(self.actor_net.num_actions)
        if (
            np.any(actions < 0)
            or np.any(actions >= action_n)
            or np.any(initial < 0)
            or np.any(initial >= action_n)
        ):
            raise ValueError(f"behavior action lies outside [0,{action_n - 1}]")
        if int(self.bc_loader.num_actions) != action_n:
            raise ValueError("behavior loader and Actor action counts disagree")
        if int(self.bc_model_net.num_actions) != action_n:
            raise ValueError("behavioral ModelNet and Actor action counts disagree")
        if int(self.bc_model_net.frame_stack_n) != int(self.flags.frame_stack_n):
            raise ValueError("behavioral ModelNet frame-stack metadata disagree")
        return batch_size

    def _barrier_ray_get(self, object_ref, *, deadline, label):
        remaining = deadline - self._monotonic()
        if remaining <= 0.0:
            self.voc_actor_policy_barrier_timeout_count += 1
            raise TimeoutError(f"actor policy barrier timed out before {label}")
        try:
            return ray.get(object_ref, timeout=remaining)
        except ray.exceptions.GetTimeoutError as error:
            self.voc_actor_policy_barrier_timeout_count += 1
            raise TimeoutError(
                f"actor policy barrier RPC timed out during {label}"
            ) from error

    def _publish_actor_policy_bundle(self, policy_version, *, terminal):
        """Publish one complete policy epoch and make it the only live one."""

        if not self.voc_actor_policy_barrier_runtime:
            raise RuntimeError("actor policy bundle publication is not active")
        expected = self.voc_actor_policy_version + 1
        if policy_version != expected:
            self.voc_actor_policy_version_mismatch_count += 1
            raise RuntimeError(
                f"actor policy publication {policy_version} is not contiguous "
                f"after {self.voc_actor_policy_version}"
            )
        if self.voc_actor_policy_terminal:
            self.voc_actor_policy_version_mismatch_count += 1
            raise RuntimeError("cannot publish after terminal actor policy")
        state = util.clone_actor_policy_state(self.actor_net.state_dict())
        bundle = util.make_actor_policy_bundle(
            state,
            policy_version,
            terminal=terminal,
            gate_schema=self.voc_gate_policy_schema_version,
        )
        # Clearing stale acknowledgements and publishing the complete bundle
        # are individually acknowledged GeneralBuffer operations.  Workers
        # accept only their exact next version, so no stale bundle grants
        # rollout authority between these calls.
        deadline = self._monotonic() + self.voc_actor_policy_barrier_timeout_s
        self._barrier_ray_get(
            self.actor_param_buffer.set_data.remote(
                util.VOC_ACTOR_POLICY_ACKS_KEY, {}
            ),
            deadline=deadline,
            label="acknowledgement reset",
        )
        self._barrier_ray_get(
            self.actor_param_buffer.set_data.remote(
                util.VOC_ACTOR_POLICY_BUNDLE_KEY, bundle
            ),
            deadline=deadline,
            label="bundle publication",
        )
        self.voc_actor_policy_version = int(policy_version)
        if int(policy_version) > 0:
            self.voc_actor_policy_publication_count += 1
        if self.voc_actor_policy_publication_count != self.voc_actor_policy_version:
            self.voc_actor_policy_version_mismatch_count += 1
            raise RuntimeError(
                "actor policy publication count/version lost lockstep"
            )
        self.voc_actor_policy_terminal = bool(terminal)
        self.voc_actor_policy_state_sha256 = (
            util.actor_policy_state_sha256(state)
        )
        self._voc_actor_policy_bundle = bundle

    def _wait_for_actor_policy_acks(self, *, policy_version, terminal):
        """Wait monotonically for every active worker's exact load ack."""

        expected_n = self.voc_actor_policy_expected_ack_count
        if expected_n <= 0:
            raise RuntimeError("actor policy barrier has no expected workers")
        deadline = self._monotonic() + self.voc_actor_policy_barrier_timeout_s
        while True:
            acks = self._barrier_ray_get(
                self.actor_param_buffer.get_data.remote(
                    util.VOC_ACTOR_POLICY_ACKS_KEY
                ),
                deadline=deadline,
                label="acknowledgement poll",
            )
            if acks is None:
                acks = {}
            if not isinstance(acks, Mapping):
                self.voc_actor_policy_malformed_bundle_count += 1
                raise RuntimeError("actor policy acknowledgement set is malformed")
            acknowledged = 0
            for rank, ack in acks.items():
                if (
                    isinstance(rank, (bool, np.bool_))
                    or not isinstance(rank, (int, np.integer))
                    or not 0 <= int(rank) < expected_n
                ):
                    self.voc_actor_policy_malformed_bundle_count += 1
                    raise RuntimeError("actor policy acknowledgement rank is malformed")
                try:
                    util.validate_actor_policy_ack(
                        ack,
                        rank=int(rank),
                        epoch=policy_version,
                        terminal=terminal,
                        expected_gate_schema=(
                            self.voc_gate_policy_schema_version
                        ),
                        label=f"actor policy ack rank {int(rank)}",
                    )
                except ValueError as error:
                    observed = (
                        ack.get("policy_version")
                        if isinstance(ack, Mapping) else None
                    )
                    if isinstance(observed, (int, np.integer)) and not isinstance(
                        observed, (bool, np.bool_)
                    ):
                        self.voc_actor_policy_version_mismatch_count += 1
                    else:
                        self.voc_actor_policy_malformed_bundle_count += 1
                    raise RuntimeError("actor policy acknowledgement mismatch") from error
                acknowledged += 1
            if acknowledged == expected_n:
                if len(self.voc_actor_policy_publication_history) != int(
                    policy_version
                ):
                    self.voc_actor_policy_version_mismatch_count += 1
                    raise RuntimeError(
                        "actor policy publication history is not contiguous"
                    )
                event = {
                    "predecessor_version": int(policy_version) - 1,
                    "policy_version": int(policy_version),
                    "publication_count": int(
                        self.voc_actor_policy_publication_count
                    ),
                    "terminal": bool(terminal),
                    "ack_ranks": list(range(expected_n)),
                    "expected_ack_count": int(expected_n),
                    "state_sha256": self.voc_actor_policy_state_sha256,
                }
                self.voc_actor_policy_publication_history.append(event)
                self.voc_actor_policy_publication_history_sha256 = (
                    util.actor_policy_publication_history_sha256(
                        self.voc_actor_policy_publication_history
                    )
                )
                if terminal:
                    self.voc_actor_policy_terminal_ack_count = acknowledged
                return
            if self._monotonic() >= deadline:
                self.voc_actor_policy_barrier_timeout_count += 1
                raise TimeoutError(
                    "actor policy barrier timed out after exactly "
                    f"{self.voc_actor_policy_barrier_timeout_s:.1f}s: "
                    f"{acknowledged}/{expected_n} acknowledgements"
                )
            self._barrier_sleep(0.01)

    def _validate_actor_policy_batch(self, train_actor_out):
        policy_version = getattr(train_actor_out, "policy_version", None)
        episode_return = getattr(train_actor_out, "episode_return", None)
        integer_dtypes = {
            torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
        }
        unroll_len = getattr(self.flags, "actor_unroll_len", None)
        batch_size = getattr(self.flags, "actor_batch_size", None)
        if (
            isinstance(unroll_len, (bool, np.bool_))
            or not isinstance(unroll_len, (int, np.integer))
            or int(unroll_len) <= 0
            or isinstance(batch_size, (bool, np.bool_))
            or not isinstance(batch_size, (int, np.integer))
            or int(batch_size) != 16
        ):
            self.voc_actor_policy_malformed_bundle_count += 1
            raise RuntimeError("schema-6 replay topology flags are malformed")
        expected_shape = (
            int(unroll_len) + 1,
            int(batch_size),
        )
        if (
            policy_version is None
            or not isinstance(policy_version, torch.Tensor)
            or policy_version.dtype not in integer_dtypes
            or not isinstance(episode_return, torch.Tensor)
            or policy_version.ndim != 2
            or tuple(policy_version.shape) != tuple(episode_return.shape[:2])
            or tuple(policy_version.shape) != expected_shape
        ):
            self.voc_actor_policy_malformed_bundle_count += 1
            raise RuntimeError(
                "schema-6 replay policy_version must be an integer [T,B] "
                "tensor matching episode_return"
            )
        row0 = policy_version[0]
        rows = policy_version[1:]
        if not torch.all(row0 == -1) or not torch.all(
            rows == self.voc_actor_policy_version
        ):
            self.voc_actor_policy_version_mismatch_count += 1
            raise RuntimeError(
                "schema-6 replay policy versions are not homogeneous: "
                f"expected row0=-1 and rows1:={self.voc_actor_policy_version}"
            )
        actor_ids = getattr(train_actor_out, "id", None)
        expected_n = 16
        if (
            actor_ids is None
            or not isinstance(actor_ids, torch.Tensor)
            or actor_ids.dtype not in integer_dtypes
            or tuple(actor_ids.shape) != (1, expected_n)
        ):
            self.voc_actor_policy_malformed_bundle_count += 1
            raise RuntimeError("schema-6 replay actor ids are malformed")
        canonical_ids = torch.arange(
            expected_n, device=actor_ids.device, dtype=actor_ids.dtype
        )
        if not torch.equal(torch.sort(actor_ids.reshape(-1)).values, canonical_ids):
            self.voc_actor_policy_version_mismatch_count += 1
            raise RuntimeError(
                "schema-6 replay must contain every actor id exactly once"
            )

    def _init_schema13_telemetry(self):
        """Create the four fresh headers before policy version zero exists."""

        if not self._voc_telemetry_active:
            return
        from thinker import voc_telemetry

        run_dir = os.path.abspath(self.plogger.basepath)
        if run_dir != os.path.abspath(self.flags.ckpdir):
            raise RuntimeError(
                "schema-13 telemetry run directory disagrees with ckpdir"
            )
        self._voc_telemetry_writer = voc_telemetry.Schema13TelemetryWriter(
            run_dir,
            xpid=self.flags.xpid,
            actor_unroll_len=int(self.flags.actor_unroll_len),
            stage_total_steps=int(self.flags.total_steps),
            q_initial_lr=float(self.flags.actor_learning_rate),
            schedule_total_steps=int(self.flags.schedule_total_steps),
            amp_initial_scale=(256.0 if self.flags.float16 else 1.0),
        )
        self._voc_telemetry_pending = None
        self._voc_telemetry_evidence = None
        self._voc_telemetry_log_closed = False

    def _schema13_adam_steps(self):
        """Clone the common raw weight/bias Adam step without reading it."""

        steps = []
        for parameter in self.voc_parameters:
            state = self.voc_optimizer.state.get(parameter)
            if not state:
                steps.append(torch.tensor(0.0, dtype=torch.float32))
                continue
            if not isinstance(state, dict) or "step" not in state:
                raise RuntimeError(
                    "schema-13 VoC Adam state lacks its common step"
                )
            steps.append(state["step"].detach().clone())
        if len(steps) != 2:
            raise RuntimeError(
                "schema-13 VoC Adam must expose weight and bias steps"
            )
        return tuple(steps)

    def _stage_schema13_replay_batch(self, train_actor_out):
        """Stage detached batch-boundary tensors without reduction or I/O."""

        if not self._voc_telemetry_active:
            return
        if self._voc_telemetry_pending is not None:
            raise RuntimeError(
                "schema-13 telemetry transaction staging is already live"
            )
        replay_t, replay_b = train_actor_out.done.shape
        if replay_t != int(self.flags.actor_unroll_len) + 1 or replay_b != 16:
            raise RuntimeError("schema-13 replay dimensions are malformed")
        scale_snapshot = 1.0
        if self.flags.float16:
            live_scale = getattr(self.voc_scaler, "_scale", None)
            scale_snapshot = (
                float(getattr(self.voc_scaler, "_init_scale"))
                if live_scale is None
                else live_scale.detach().clone()
            )
        scheduler_state = self.voc_scheduler.state_dict()
        self._voc_telemetry_pending = {
            "source_policy_version": int(self.voc_actor_policy_version),
            "real_step_before": int(self.real_step),
            "replay_t": int(replay_t),
            "optimized_t": int(replay_t - 1),
            "replay_b": int(replay_b),
            "actor_ids": train_actor_out.id.detach().clone(),
            "real_transition": train_actor_out.real_transition[1:].detach().clone(),
            "voc_update_count_before": int(self.voc_update_count),
            "ema_update_count_before": int(self.voc_ema_gate_update_count),
            "projection_count_before": int(self.voc_gate_update_count),
            "q_scheduler_last_epoch_before": int(
                scheduler_state["last_epoch"]
            ),
            "q_scheduler_step_count_before": int(
                scheduler_state["_step_count"]
            ),
            "q_lr_before": float(self.voc_optimizer.param_groups[0]["lr"]),
            "amp_scale_snapshot": scale_snapshot,
            "adam_step_before": self._schema13_adam_steps(),
        }

    def _stage_schema13_voc_sources(
        self,
        *,
        voc_result,
        ema_q_values,
        train_mask,
        holdout_mask,
        control_action,
        search_steps,
    ):
        """Stage exact FP32 TD sources on the existing device stream."""

        if not self._voc_telemetry_active:
            return
        pending = self._voc_telemetry_pending
        if not isinstance(pending, dict) or "target" in pending:
            raise RuntimeError("schema-13 TD staging boundary is malformed")
        pending.update({
            "target": voc_result.target.detach().float().clone(),
            "online_q_values": voc_result.q_values.detach().float().clone(),
            "ema_q_values": ema_q_values.detach().float().clone(),
            "train_mask": train_mask.detach().clone(),
            "holdout_mask": holdout_mask.detach().clone(),
            "valid_mask": voc_result.valid.detach().clone(),
            "gate_action": voc_result.gate_action.detach().clone(),
            "control_action": control_action.detach().clone(),
            "search_steps": search_steps.detach().clone(),
            "q_loss_sum": voc_result.q_loss.detach().float().clone(),
        })

    def _stage_schema13_post_scheduler(self, voc_step_result):
        """Stage status/counters at the inherited post-scheduler boundary."""

        if not self._voc_telemetry_active:
            return
        pending = self._voc_telemetry_pending
        if not isinstance(pending, dict) or "q_loss_sum" not in pending:
            raise RuntimeError("schema-13 post-scheduler staging lacks TD sources")
        if voc_step_result is None:
            q_status = "no_support"
            scale_before = pending["amp_scale_snapshot"]
            scale_after = pending["amp_scale_snapshot"]
            nonfinite_count = 0
        elif voc_step_result.optimizer_stepped:
            q_status = "stepped"
            scale_before = (
                1.0
                if voc_step_result.amp_scale_before is None
                else voc_step_result.amp_scale_before
            )
            scale_after = (
                1.0
                if voc_step_result.amp_scale_after is None
                else voc_step_result.amp_scale_after
            )
            nonfinite_count = 0
        else:
            q_status = "amp_skip"
            scale_before = voc_step_result.amp_scale_before
            scale_after = voc_step_result.amp_scale_after
            nonfinite_count = len(voc_step_result.nonfinite_gradient_names)
        scheduler_state = self.voc_scheduler.state_dict()
        pending.update({
            "q_status": q_status,
            "amp_scale_before": scale_before,
            "amp_scale_after": scale_after,
            "nonfinite_gradient_parameter_count": nonfinite_count,
            "adam_step_after": self._schema13_adam_steps(),
            "voc_update_count_after": int(self.voc_update_count),
            "ema_update_count_after": int(self.voc_ema_gate_update_count),
            "projection_count_after": int(self.voc_gate_update_count),
            "q_scheduler_last_epoch_after": int(
                scheduler_state["last_epoch"]
            ),
            "q_scheduler_step_count_after": int(
                scheduler_state["_step_count"]
            ),
            "q_lr_after": float(self.voc_optimizer.param_groups[0]["lr"]),
            "real_step_after": int(self.real_step),
        })

    @staticmethod
    def _schema13_scalar(value, *, integer=False):
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise RuntimeError("schema-13 scalar staging tensor is not scalar")
            value = value.detach().cpu().item()
        if integer:
            numeric = float(value)
            if not np.isfinite(numeric) or not numeric.is_integer() or numeric < 0:
                raise RuntimeError("schema-13 staged integer scalar is invalid")
            return int(numeric)
        numeric = float(value)
        if not np.isfinite(numeric):
            raise RuntimeError("schema-13 staged finite scalar is invalid")
        return numeric

    @classmethod
    def _schema13_common_step(cls, staged, *, label):
        if type(staged) not in (tuple, list) or len(staged) != 2:
            raise RuntimeError(f"{label} must contain weight and bias steps")
        steps = tuple(cls._schema13_scalar(value, integer=True) for value in staged)
        if steps[0] != steps[1]:
            raise RuntimeError(f"{label} weight/bias Adam steps disagree")
        return steps[0]

    def _commit_schema13_telemetry_after_ack(self, *, terminal, ack_count):
        """Reduce and durably append exactly one post-ack transaction."""

        if not self._voc_telemetry_active:
            return None
        from thinker import voc_telemetry

        pending = self._voc_telemetry_pending
        if not isinstance(pending, dict) or "q_status" not in pending:
            raise RuntimeError("schema-13 post-ack commit lacks staged transaction")
        transaction_id = self.voc_actor_policy_version
        if (
            transaction_id != pending["source_policy_version"] + 1
            or transaction_id != self._voc_telemetry_writer.transaction_count + 1
        ):
            raise RuntimeError("schema-13 telemetry transaction version drift")
        actor_ids = tuple(
            int(value)
            for value in pending["actor_ids"].detach().reshape(-1).cpu().tolist()
        )
        real_step_delta = int(
            pending["real_transition"].detach().bool().sum().cpu().item()
        )
        masks = {
            name: pending[name].detach().bool()
            for name in ("valid_mask", "train_mask", "holdout_mask")
        }
        gate_action = pending["gate_action"].detach().long()
        counts = {
            "valid_count": int(masks["valid_mask"].sum().cpu().item()),
            "train_count": int(masks["train_mask"].sum().cpu().item()),
            "holdout_count": int(masks["holdout_mask"].sum().cpu().item()),
            "train_continue_count": int(
                (masks["train_mask"] & (gate_action == 0)).sum().cpu().item()
            ),
            "train_stop_count": int(
                (masks["train_mask"] & (gate_action == 1)).sum().cpu().item()
            ),
            "holdout_continue_count": int(
                (masks["holdout_mask"] & (gate_action == 0)).sum().cpu().item()
            ),
            "holdout_stop_count": int(
                (masks["holdout_mask"] & (gate_action == 1)).sum().cpu().item()
            ),
        }
        td_rows = voc_telemetry.build_td_cell_rows(
            transaction_id=transaction_id,
            source_policy_version=pending["source_policy_version"],
            published_policy_version=transaction_id,
            real_step_after=pending["real_step_after"],
            target=pending["target"],
            online_q_values=pending["online_q_values"],
            ema_q_values=pending["ema_q_values"],
            valid_mask=pending["valid_mask"],
            train_mask=pending["train_mask"],
            holdout_mask=pending["holdout_mask"],
            gate_action=pending["gate_action"],
            control_action=pending["control_action"],
            search_steps=pending["search_steps"],
        )
        q_status = pending["q_status"]
        q_lr_before = pending["q_lr_before"]
        replay_row = voc_telemetry.build_replay_row(
            transaction_id=transaction_id,
            source_policy_version=pending["source_policy_version"],
            published_policy_version=transaction_id,
            replay_t=pending["replay_t"],
            optimized_t=pending["optimized_t"],
            replay_b=pending["replay_b"],
            actor_ids=actor_ids,
            real_step_before=pending["real_step_before"],
            real_step_delta=real_step_delta,
            real_step_after=pending["real_step_after"],
            **counts,
            q_status=q_status,
            voc_update_count_before=pending["voc_update_count_before"],
            voc_update_count_after=pending["voc_update_count_after"],
            ema_update_count_before=pending["ema_update_count_before"],
            ema_update_count_after=pending["ema_update_count_after"],
            projection_count_before=pending["projection_count_before"],
            projection_count_after=pending["projection_count_after"],
            q_scheduler_last_epoch_before=pending[
                "q_scheduler_last_epoch_before"
            ],
            q_scheduler_last_epoch_after=pending[
                "q_scheduler_last_epoch_after"
            ],
            q_scheduler_step_count_before=pending[
                "q_scheduler_step_count_before"
            ],
            q_scheduler_step_count_after=pending[
                "q_scheduler_step_count_after"
            ],
            q_lr_before=q_lr_before,
            q_lr_used=q_lr_before,
            q_lr_after=pending["q_lr_after"],
            publication_count_after=self.voc_actor_policy_publication_count,
            ack_count=ack_count,
            terminal=terminal,
            actor_state_sha256=self.voc_actor_policy_state_sha256,
            publication_history_sha256=(
                self.voc_actor_policy_publication_history_sha256
            ),
        )
        adam_step_before = self._schema13_common_step(
            pending["adam_step_before"], label="adam_step_before"
        )
        adam_step_after = self._schema13_common_step(
            pending["adam_step_after"], label="adam_step_after"
        )
        diagnostics = None
        if q_status == "stepped":
            q_step = pending.get("q_step")
            candidate = None if not isinstance(q_step, dict) else q_step.get("candidate")
            if not isinstance(candidate, dict):
                raise RuntimeError("schema-13 stepped Q lacks candidate snapshots")
            clip_limit = float(self.flags.actor_grad_norm_clipping) * (
                pending["replay_t"] * pending["replay_b"]
            )
            clip_scale = self._schema13_scalar(q_step["clip_coefficient"])
            diagnostics = voc_telemetry.build_stepped_q_diagnostics(
                clip_scale=clip_scale,
                raw_preclip=q_step["raw_preclip"],
                raw_postclip=q_step["raw_postclip"],
                md_postclip=candidate["md_postclip"],
                adam_m_before=candidate["adam_m_before"],
                adam_v_before=candidate["adam_v_before"],
                adam_m_after=candidate["adam_m_after"],
                adam_v_after=candidate["adam_v_after"],
                coordinate_delta=candidate["coordinate_delta"],
                mapped_delta=candidate["mapped_delta"],
                q_lr_used=q_lr_before,
                adam_step_after=adam_step_after,
            )
        q_row = voc_telemetry.build_q_transaction_row(
            transaction_id=transaction_id,
            source_policy_version=pending["source_policy_version"],
            published_policy_version=transaction_id,
            real_step_after=pending["real_step_after"],
            q_status=q_status,
            q_attempted=(q_status != "no_support"),
            q_optimizer_committed=(q_status == "stepped"),
            q_loss_sum=self._schema13_scalar(pending["q_loss_sum"]),
            clip_limit=float(self.flags.actor_grad_norm_clipping)
            * pending["replay_t"]
            * pending["replay_b"],
            amp_scale_before=self._schema13_scalar(
                pending["amp_scale_before"]
            ),
            amp_scale_after=self._schema13_scalar(pending["amp_scale_after"]),
            nonfinite_gradient_parameter_count=pending[
                "nonfinite_gradient_parameter_count"
            ],
            adam_step_before=adam_step_before,
            adam_step_after=adam_step_after,
            diagnostics=diagnostics,
        )
        commit = self._voc_telemetry_writer.append_transaction(
            td_rows=td_rows,
            replay_row=replay_row,
            q_row=q_row,
            terminal=terminal,
            actor_state_sha256=self.voc_actor_policy_state_sha256,
            publication_history_sha256=(
                self.voc_actor_policy_publication_history_sha256
            ),
        )
        self._voc_telemetry_pending = None
        if hasattr(self.voc_optimizer, "_schema13_telemetry_candidate"):
            self.voc_optimizer._schema13_telemetry_candidate = None
        return commit

    def _maybe_compute_imitation(self):
        self._imitation_pending_update = False
        if not self.imitation_enabled:
            return None
        self.imitation_schedule_step += 1
        frequency = int(getattr(self.flags, "icopro_supervised_freq", 1))
        if (self.imitation_schedule_step - 1) % frequency != 0:
            return None
        self._ensure_imitation_runner()
        batch = self._sample_imitation_batch()
        if batch is None:
            raise RuntimeError("behavioral loader returned no valid sequence batch")
        result = self.bc_runner.rollout(
            batch,
            tree_carry=bool(getattr(self.flags, "tree_carry", True)),
            training=True,
        )
        for name in (
            "loss",
            "normalized_ce",
            "margin_loss",
            "pvp_loss",
            "nll_sum",
            "logits",
        ):
            _require_finite_tensor(f"dynamic imitation {name}", getattr(result, name))
        self._imitation_pending_update = True
        return result

    def _add_online_action_prior(
            self, total_loss, losses, new_actor_out, real_policy_mask):
        weight = float(getattr(self.flags, "action_prior_weight", 0.0))
        if weight <= 0.0 or self.action_prior is None:
            return total_loss
        logits = new_actor_out.pri_param
        if logits.ndim != 4 or logits.shape[-2] != 1:
            raise ValueError("action prior requires logits [T,B,1,A]")
        selected_logits = logits[:, :, 0][real_policy_mask]
        real_policy_n = selected_logits.shape[0]
        if real_policy_n == 0:
            return total_loss
        p_batch = F.softmax(selected_logits, dim=-1).mean(dim=0)
        _require_finite_tensor("online action-prior batch distribution", p_batch)
        beta = min(max(float(getattr(self.flags, "action_prior_ema", 0.05)), 0.0), 1.0)
        if self.action_prior_ema is None:
            p_smooth = p_batch
        else:
            _require_finite_tensor("stored action-prior EMA", self.action_prior_ema)
            p_smooth = (1.0 - beta) * self.action_prior_ema + beta * p_batch
        _require_finite_tensor("candidate action-prior EMA", p_smooth)
        self._pending_action_prior_ema = p_smooth.detach()
        eps = 1e-8
        p_smooth = p_smooth.clamp_min(eps)
        target = self.action_prior.to(p_smooth.device).clamp_min(eps)
        _require_finite_tensor("human action prior", target)
        prior_mean = torch.sum(p_smooth * (p_smooth.log() - target.log()))
        _require_finite_tensor("action-prior loss", prior_mean)
        # Actor RL losses are sums.  Treat the KL as a per-real-policy-row
        # regularizer so its coefficient is independent of unroll layout.
        prior_sum = prior_mean * real_policy_n
        total_loss = total_loss + weight * prior_sum
        losses["action_prior_loss"] = prior_sum
        return total_loss

    def learn_data(self):
        timing = util.Timings() if self.time else None
        # Legacy tests and failure-path fixtures can enter through ``__new__``
        # without the schema-6 lifecycle fields initialized.  Missing runtime
        # metadata is the legacy/off state; do not turn an unrelated learner
        # exception into an AttributeError while unwinding that path.
        barrier_runtime = bool(
            getattr(self, "voc_actor_policy_barrier_runtime", False)
        )
        if barrier_runtime:
            self._publish_actor_policy_bundle(0, terminal=False)
            self._wait_for_actor_policy_acks(
                policy_version=0, terminal=False
            )
        data_ptr = self.actor_buffer.read.remote()
        successful = False
        try:
            while self.real_step < self.flags.total_steps:
                if timing is not None:
                    timing.reset()
                # get data remotely
                read_deadline = (
                    self._monotonic()
                    + self.voc_actor_policy_barrier_timeout_s
                    if barrier_runtime
                    else None
                )
                while True:
                    if barrier_runtime:
                        data = self._barrier_ray_get(
                            data_ptr,
                            deadline=read_deadline,
                            label="actor replay read",
                        )
                    else:
                        data = ray.get(data_ptr)
                    ray.internal.free(data_ptr)
                    data_ptr = self.actor_buffer.read.remote()                    
                    if data is not None:
                        break
                    time.sleep(0.001)
                    self.queue_n += 0.001
                if timing is not None:
                    timing.time("get_data")
         
                train_actor_out, initial_actor_state = data
                train_actor_out = util.tuple_map(
                    train_actor_out, lambda x: torch.tensor(x, device=self.device)
                )
                initial_actor_state = util.tuple_map(
                    initial_actor_state, lambda x: torch.tensor(x, device=self.device)
                )
                if timing is not None:
                    timing.time("convert_data")
                if barrier_runtime:
                    self._validate_actor_policy_batch(train_actor_out)
                    if getattr(self, "_voc_telemetry_active", False):
                        self._stage_schema13_replay_batch(train_actor_out)
                data = (train_actor_out, initial_actor_state)
                # start consume data
                if barrier_runtime:
                    if self._voc_actor_policy_transaction_open:
                        raise RuntimeError(
                            "schema-6 actor policy transaction is already open"
                        )
                    self._voc_actor_policy_transaction_open = True
                self.consume_data(data, timing=timing)
                del train_actor_out, initial_actor_state, data
                
                if barrier_runtime:
                    next_version = self.voc_actor_policy_version + 1
                    terminal = self.real_step >= self.flags.total_steps
                    self._publish_actor_policy_bundle(
                        next_version, terminal=terminal
                    )
                    self._wait_for_actor_policy_acks(
                        policy_version=next_version, terminal=terminal
                    )
                    if getattr(self, "_voc_telemetry_active", False):
                        self._commit_schema13_telemetry_after_ack(
                            terminal=terminal,
                            ack_count=int(
                                self.voc_actor_policy_expected_ack_count
                            ),
                        )
                    self._voc_actor_policy_transaction_open = False
                    self._flush_pending_actor_policy_checkpoint()
                else:
                    self.actor_param_buffer.set_data.remote(
                        "actor_net", self._actor_weights_for_publication()
                    )
                if timing is not None:
                    timing.time("set weight")            
          
            self._logger.info("Terminating actor-learning thread")
            if (
                barrier_runtime
                and not self.voc_actor_policy_terminal
            ):
                raise RuntimeError(
                    "schema-6 learner reached completion without one terminal "
                    "post-last-batch publication"
                )
            self.save_checkpoint(force=True)
            successful = True
            return True
        except Exception as e:
            self._logger.error(f"Exception detected in learn_actor: {e}")
            self._logger.error(traceback.format_exc())
            raise
        finally:
            self.close(successful=successful)

    def _flush_pending_actor_policy_checkpoint(self):
        """Write at most once, and only beyond the publication/ack boundary."""

        if not self.voc_actor_policy_barrier_runtime:
            return False
        if self._voc_actor_policy_transaction_open:
            raise RuntimeError(
                "cannot flush schema-6 checkpoint inside policy transaction"
            )
        if not self._voc_actor_policy_checkpoint_pending:
            return False
        pending_force = self._voc_actor_policy_checkpoint_force
        self._voc_actor_policy_checkpoint_pending = False
        self._voc_actor_policy_checkpoint_force = False
        self.save_checkpoint(force=pending_force)
        return True
        
    def consume_data(self, data, timing=None):

        train_actor_out, initial_actor_state = data
        T, B, *_ = train_actor_out.episode_return.shape
        if self.dynamic_search:
            # Row zero is the recurrent bootstrap/overlap row and is dropped
            # by compute_losses.  Do not count its transition twice at unroll
            # boundaries.
            self.step += (T - 1) * B
            last_step_real = train_actor_out.real_transition[1:].bool()
            real_done_source = train_actor_out.real_done[1:]
        else:
            self.step += T * B
            last_step_real = ((train_actor_out.step_status == 0)
                              | (train_actor_out.step_status == 3))
            real_done_source = train_actor_out.real_done
        self.real_step += torch.sum(last_step_real).item()
        real_done_count = torch.sum(real_done_source).item()
        self.tot_eps += real_done_count
        
        # 디버깅: 에피소드 카운터 증가 추적
        if real_done_count > 0:
            self._logger.info(f"[DEBUG] Episode counter increased:")
            self._logger.info(f"  - real_done_count: {real_done_count}")
            self._logger.info(f"  - tot_eps: {self.tot_eps}")
            self._logger.info(f"  - train_actor_out.real_done: {train_actor_out.real_done}")
            self._logger.info(f"  - train_actor_out.done: {train_actor_out.done}")
            self._logger.info(f"  - train_actor_out.truncated_done: {train_actor_out.truncated_done}")
        
        # ActorBuffer의 real_step도 함께 업데이트
        if self.flags.parallel_actor and hasattr(self, 'actor_buffer'):
            try:
                # Ray 원격 객체 메서드 호출 방식으로 수정
                update_future = self.actor_buffer.update_real_step.remote(int(self.real_step))
                if self.voc_actor_policy_barrier_runtime:
                    self._barrier_ray_get(
                        update_future,
                        deadline=(
                            self._monotonic()
                            + self.voc_actor_policy_barrier_timeout_s
                        ),
                        label="actor replay real-step update",
                    )
                # Legacy schemas preserve the asynchronous call.
                if self.real_step % 1000 == 0:
                    self._logger.info(f"Sent real_step update to ActorBuffer: {self.real_step}")
            except Exception as e:
                self._logger.error(f"Error updating ActorBuffer real_step: {e}")
                traceback.print_exc()
                if self.voc_actor_policy_barrier_runtime:
                    raise

        if not self.ppo_enable: return self.consume_data_single(data, timing)        
        TrainActorOut= type(train_actor_out)

        if self.ppo_buffer is None:            
            out = {}
            for k in TrainActorOut._fields:
                out[k] = None
                v = getattr(train_actor_out, k)
                if v is None: continue
                out[k] = torch.zeros(size=(v.shape[0], self.ppo_buffer_n) + v.shape[2:], dtype=v.dtype, device=self.device)
            self.ppo_buffer = TrainActorOut(**out)            
            self.ppo_buffer_actor_state = []
            for v in initial_actor_state:
                self.ppo_buffer_actor_state.append(torch.zeros(size=(self.ppo_buffer_n,)+v.shape[1:], dtype=v.dtype, device=self.device))
            self.buffer_idx = 0
            self.buffer_wrote_n = 0

        for k in TrainActorOut._fields:
            v_ = getattr(self.ppo_buffer, k)
            if v_ is None: continue           
            v = getattr(train_actor_out, k)
            v_[:, self.buffer_idx:self.buffer_idx+self.ppo_b] = v
        for n, v in enumerate(initial_actor_state):
            self.ppo_buffer_actor_state[n][self.buffer_idx:self.buffer_idx+self.ppo_b] = v

        self.buffer_wrote_n = min(self.buffer_wrote_n + self.ppo_b, self.ppo_buffer_n) 
        self.buffer_idx = (self.buffer_idx + self.ppo_b) % self.ppo_buffer_n
        
        self.ppo_t += 1        
        r = False                      
        if self.ppo_t % self.ppo_update_freq == 0:
            self.ppo_early_stop = False
            for m in range(self.ppo_update_time):
                ns = random.sample(range(self.buffer_wrote_n), self.buffer_wrote_n)
                ns = [ns[i:i + self.ppo_b] for i in range(0, len(ns), self.ppo_b)]     
                for k, n in enumerate(ns):
                    out = {}
                    for k_ in TrainActorOut._fields:
                        out[k_] = None
                        v = getattr(self.ppo_buffer, k_)
                        if v is None: continue           
                        out[k_] = v[:, n]
                    train_actor_out = TrainActorOut(**out)  

                    initial_actor_state = []
                    for v in self.ppo_buffer_actor_state:
                        initial_actor_state.append(v[n])

                    data = (train_actor_out, initial_actor_state)
                    r = self.consume_data_single(data, timing=timing, first_iter=k<self.ppo_update_freq and m == 0, last_iter=k==len(ns)-1)
                    if self.ppo_early_stop: break
                if self.ppo_early_stop: break
        return r

    def _step_actor_optimizer(self, optimize_params, T, B):
        """Apply one actor optimizer step, recovering isolated AMP overflows."""

        raw_total_norm = util.compute_grad_norm(optimize_params)
        raw_total_norm = float(raw_total_norm.detach().cpu())
        nonfinite_names = ()
        if not np.isfinite(raw_total_norm):
            parameter_names = {
                id(parameter): name
                for name, parameter in self.actor_net.named_parameters()
            }
            nonfinite_names = tuple(
                parameter_names.get(id(parameter), f"parameter[{index}]")
                for index, parameter in enumerate(optimize_params)
                if parameter.grad is not None
                and not torch.isfinite(parameter.grad).all().item()
            )
            if not self.flags.float16:
                raise FloatingPointError(
                    "non-finite actor gradient norm in FP32: "
                    f"{raw_total_norm}; parameters={nonfinite_names[:8]}"
                )
            if not nonfinite_names:
                raise FloatingPointError(
                    "actor gradient norm is non-finite even though every "
                    "gradient element is finite"
                )
        elif self.flags.actor_grad_norm_clipping > 0:
            torch.nn.utils.clip_grad_norm_(
                optimize_params, self.flags.actor_grad_norm_clipping * T * B
            )

        amp_scale_before = None
        amp_scale_after = None
        optimizer_stepped = True
        if self.flags.float16:
            amp_scale_before = float(self.scaler.get_scale())
            self.scaler.step(self.optimizer)
            self.scaler.update()
            amp_scale_after = float(self.scaler.get_scale())
            optimizer_stepped = amp_scale_after >= amp_scale_before
        else:
            self.optimizer.step()

        if optimizer_stepped:
            self.actor_amp_consecutive_skips = 0
        else:
            self.actor_amp_skip_count += 1
            self.actor_amp_consecutive_skips += 1
            if amp_scale_after is None or amp_scale_after >= amp_scale_before:
                raise FloatingPointError(
                    "actor AMP skipped an optimizer step without reducing its "
                    f"scale ({amp_scale_before} -> {amp_scale_after})"
                )
            self._logger.warning(
                "Actor AMP overflow: skipped optimizer step %d; consecutive=%d; "
                "scale %.1f -> %.1f; nonfinite_parameters=%s",
                self.actor_amp_skip_count,
                self.actor_amp_consecutive_skips,
                amp_scale_before,
                amp_scale_after,
                list(nonfinite_names[:8]),
            )
            max_skips = int(self.flags.actor_amp_max_consecutive_skips)
            if self.actor_amp_consecutive_skips >= max_skips:
                raise FloatingPointError(
                    "actor AMP overflow persisted for "
                    f"{self.actor_amp_consecutive_skips} consecutive updates; "
                    f"scale={amp_scale_after}; parameters={nonfinite_names[:8]}"
                )

        result = ActorGradientStepResult(
            total_norm=(raw_total_norm if optimizer_stepped else 0.0),
            optimizer_stepped=optimizer_stepped,
            amp_scale_before=amp_scale_before,
            amp_scale_after=amp_scale_after,
            nonfinite_gradient_names=nonfinite_names,
        )
        self._last_actor_gradient_step = result
        return result

    def _step_voc_optimizer(self, T, B):
        """Step the isolated VoC critic optimizer and its independent scaler."""

        if self.voc_optimizer is None:
            raise RuntimeError("VoC optimizer step requested in off mode")
        optimize_params = self.voc_parameters
        telemetry_active = bool(
            getattr(self, "_voc_telemetry_active", False)
        )
        raw_preclip = None
        raw_postclip = None
        clip_returned_norm = None
        clip_coefficient = None
        if telemetry_active:
            if not isinstance(
                getattr(self, "_voc_telemetry_pending", None), dict
            ):
                raise RuntimeError(
                    "schema-13 Q step lacks pending telemetry transaction"
                )
            raw_preclip = tuple(
                parameter.grad.detach().clone(
                    memory_format=torch.preserve_format
                )
                for parameter in optimize_params
            )
        raw_total_norm = util.compute_grad_norm(optimize_params)
        raw_total_norm = float(raw_total_norm.detach().cpu())
        nonfinite_names = ()
        if not np.isfinite(raw_total_norm):
            parameter_names = {
                id(parameter): name
                for name, parameter in self.actor_net.named_parameters()
            }
            nonfinite_names = tuple(
                parameter_names.get(id(parameter), f"voc_parameter[{index}]")
                for index, parameter in enumerate(optimize_params)
                if parameter.grad is not None
                and not torch.isfinite(parameter.grad).all().item()
            )
            if not self.flags.float16:
                raise FloatingPointError(
                    "non-finite VoC gradient norm in FP32: "
                    f"{raw_total_norm}; parameters={nonfinite_names[:8]}"
                )
            if not nonfinite_names:
                raise FloatingPointError(
                    "VoC gradient norm is non-finite even though every "
                    "gradient element is finite"
                )
        else:
            clipping = float(self.flags.actor_grad_norm_clipping)
            if clipping > 0:
                clip_returned_norm = torch.nn.utils.clip_grad_norm_(
                    optimize_params, clipping * T * B
                )
                if telemetry_active:
                    # Match the pinned clip implementation's scalar expression
                    # in the returned norm's dtype/device.  This captures the
                    # coefficient without a second norm or clip call.
                    clip_coefficient = torch.clamp(
                        (clipping * T * B) / (clip_returned_norm + 1e-6),
                        max=1.0,
                    ).detach().clone()
                    raw_postclip = tuple(
                        parameter.grad.detach().clone(
                            memory_format=torch.preserve_format
                        )
                        for parameter in optimize_params
                    )
                    clip_returned_norm = (
                        clip_returned_norm.detach().clone()
                    )
            elif telemetry_active:
                raise RuntimeError(
                    "schema-13 Q telemetry requires the inherited positive clip"
                )

        amp_scale_before = None
        amp_scale_after = None
        optimizer_stepped = True
        if telemetry_active:
            self.voc_optimizer._schema13_telemetry_capture = True
            self.voc_optimizer._schema13_telemetry_candidate = None
        if self.flags.float16:
            amp_scale_before = float(self.voc_scaler.get_scale())
            try:
                self.voc_scaler.step(self.voc_optimizer)
                self.voc_scaler.update()
            finally:
                if telemetry_active:
                    self.voc_optimizer._schema13_telemetry_capture = False
            amp_scale_after = float(self.voc_scaler.get_scale())
            optimizer_stepped = amp_scale_after >= amp_scale_before
        else:
            try:
                self.voc_optimizer.step()
            finally:
                if telemetry_active:
                    self.voc_optimizer._schema13_telemetry_capture = False

        if optimizer_stepped:
            self.voc_amp_consecutive_skips = 0
        else:
            self.voc_amp_skip_count += 1
            self.voc_amp_consecutive_skips += 1
            if amp_scale_after is None or amp_scale_after >= amp_scale_before:
                raise FloatingPointError(
                    "VoC AMP skipped an optimizer step without reducing its "
                    f"scale ({amp_scale_before} -> {amp_scale_after})"
                )
            self._logger.warning(
                "VoC AMP overflow: skipped optimizer step %d; consecutive=%d; "
                "scale %.1f -> %.1f; nonfinite_parameters=%s",
                self.voc_amp_skip_count,
                self.voc_amp_consecutive_skips,
                amp_scale_before,
                amp_scale_after,
                list(nonfinite_names[:8]),
            )
            max_skips = int(self.flags.actor_amp_max_consecutive_skips)
            if self.voc_amp_consecutive_skips >= max_skips:
                raise FloatingPointError(
                    "VoC AMP overflow persisted for "
                    f"{self.voc_amp_consecutive_skips} consecutive updates; "
                    f"scale={amp_scale_after}; parameters={nonfinite_names[:8]}"
                )

        result = ActorGradientStepResult(
            total_norm=(raw_total_norm if optimizer_stepped else 0.0),
            optimizer_stepped=optimizer_stepped,
            amp_scale_before=amp_scale_before,
            amp_scale_after=amp_scale_after,
            nonfinite_gradient_names=nonfinite_names,
        )
        if telemetry_active:
            self._voc_telemetry_pending["q_step"] = {
                "raw_preclip": raw_preclip,
                "raw_postclip": raw_postclip,
                "clip_returned_norm": clip_returned_norm,
                "clip_coefficient": clip_coefficient,
                "candidate": getattr(
                    self.voc_optimizer,
                    "_schema13_telemetry_candidate",
                    None,
                ),
            }
        self._last_voc_gradient_step = result
        return result

    def _step_voc_gate_optimizer(self):
        """Step the dedicated gate independently of actor/Q clip and AMP."""

        if getattr(self, "voc_gate_exact_projection", False):
            raise RuntimeError(
                "dedicated VoC gate optimizer is bypassed by exact projection"
            )
        if self.voc_gate_optimizer is None:
            raise RuntimeError(
                "dedicated VoC gate optimizer step requested while disabled"
            )
        optimize_params = self.voc_gate_parameters
        raw_total_norm = util.compute_grad_norm(optimize_params)
        raw_total_norm = float(raw_total_norm.detach().cpu())
        nonfinite_names = ()
        if not np.isfinite(raw_total_norm):
            parameter_names = {
                id(parameter): name
                for name, parameter in self.actor_net.named_parameters()
            }
            nonfinite_names = tuple(
                parameter_names.get(
                    id(parameter), f"voc_gate_parameter[{index}]"
                )
                for index, parameter in enumerate(optimize_params)
                if parameter.grad is not None
                and not torch.isfinite(parameter.grad).all().item()
            )
            if not self.flags.float16:
                raise FloatingPointError(
                    "non-finite dedicated VoC gate gradient norm in FP32: "
                    f"{raw_total_norm}; parameters={nonfinite_names[:8]}"
                )
            if not nonfinite_names:
                raise FloatingPointError(
                    "dedicated VoC gate gradient norm is non-finite even "
                    "though every gradient element is finite"
                )
        else:
            torch.nn.utils.clip_grad_norm_(
                optimize_params, self.voc_gate_grad_norm_clipping
            )
            postclip_total_norm = util.compute_grad_norm(optimize_params)
            self._last_voc_gate_postclip_total_norm = float(
                postclip_total_norm.detach().cpu()
            )
            if not np.isfinite(self._last_voc_gate_postclip_total_norm):
                raise FloatingPointError(
                    "dedicated VoC gate clipping produced a non-finite norm"
                )
        if not np.isfinite(raw_total_norm):
            # A recoverable AMP overflow is a skipped update, not a finite
            # post-clip measurement.  Use the explicit finite sentinel that
            # the actor/Q optimizer paths already expose for skipped steps.
            self._last_voc_gate_postclip_total_norm = 0.0

        amp_scale_before = None
        amp_scale_after = None
        optimizer_stepped = True
        if self.flags.float16:
            amp_scale_before = float(self.voc_gate_scaler.get_scale())
            self.voc_gate_scaler.step(self.voc_gate_optimizer)
            self.voc_gate_scaler.update()
            amp_scale_after = float(self.voc_gate_scaler.get_scale())
            optimizer_stepped = amp_scale_after >= amp_scale_before
        else:
            self.voc_gate_optimizer.step()

        if optimizer_stepped:
            if not np.isfinite(raw_total_norm):
                raise FloatingPointError(
                    "dedicated VoC gate optimizer stepped despite a "
                    "non-finite AMP gradient"
                )
            self.voc_gate_amp_consecutive_skips = 0
        else:
            self._last_voc_gate_postclip_total_norm = 0.0
            self.voc_gate_amp_skip_count += 1
            self.voc_gate_amp_consecutive_skips += 1
            if amp_scale_after is None or amp_scale_after >= amp_scale_before:
                raise FloatingPointError(
                    "dedicated VoC gate AMP skipped without reducing its "
                    f"scale ({amp_scale_before} -> {amp_scale_after})"
                )
            self._logger.warning(
                "Dedicated VoC gate AMP overflow: skipped optimizer step %d; "
                "consecutive=%d; scale %.1f -> %.1f; "
                "nonfinite_parameters=%s",
                self.voc_gate_amp_skip_count,
                self.voc_gate_amp_consecutive_skips,
                amp_scale_before,
                amp_scale_after,
                list(nonfinite_names[:8]),
            )
            max_skips = int(self.flags.actor_amp_max_consecutive_skips)
            if self.voc_gate_amp_consecutive_skips >= max_skips:
                raise FloatingPointError(
                    "dedicated VoC gate AMP overflow persisted for "
                    f"{self.voc_gate_amp_consecutive_skips} consecutive "
                    f"updates; scale={amp_scale_after}; "
                    f"parameters={nonfinite_names[:8]}"
                )

        result = ActorGradientStepResult(
            total_norm=(raw_total_norm if optimizer_stepped else 0.0),
            optimizer_stepped=optimizer_stepped,
            amp_scale_before=amp_scale_before,
            amp_scale_after=amp_scale_after,
            nonfinite_gradient_names=nonfinite_names,
        )
        self._last_voc_gate_gradient_step = result
        return result

    def consume_data_single(self, data, timing=None, first_iter=True, last_iter=False):

        train_actor_out, initial_actor_state = data
        actor_id = train_actor_out.id
        T, B = train_actor_out.done.shape
        self._last_voc_gate_exact_projection_applied = False
        self._last_voc_gate_projection_pre_error_norm = 0.0
        self._last_voc_gate_projection_post_error_norm = 0.0

        # compute losses
        out = self.compute_losses(
            train_actor_out, initial_actor_state, first_iter, last_iter
        )
        losses, train_actor_out = out
        total_loss = losses["total_loss"]
        voc_total_loss = losses.pop("_voc_total_loss", None)
        voc_gate_total_loss = losses.pop("_voc_gate_total_loss", None)
        _require_finite_tensor("actor total loss", total_loss)
        _require_finite_tensor("VoC total loss", voc_total_loss)
        _require_finite_tensor(
            "dedicated VoC gate total loss", voc_gate_total_loss
        )
        if timing is not None:
            timing.time("compute loss")

        # gradient descent on loss
        self.optimizer.zero_grad()
        if self.voc_optimizer is not None:
            self.voc_optimizer.zero_grad()
        if self.voc_gate_optimizer is not None:
            self.voc_gate_optimizer.zero_grad()
        if self.flags.float16:
            self.scaler.scale(total_loss).backward(
                retain_graph=(voc_gate_total_loss is not None)
            )
            if voc_total_loss is not None:
                self.voc_scaler.scale(voc_total_loss).backward()
            if self.voc_dedicated_gate:
                # Fail-safe isolation: even if a future actor auxiliary loss
                # accidentally retains the dedicated logits, discard that
                # independently-scaled gradient.  Held-out-only batches also
                # leave no stale gate gradient behind.
                for parameter in self.voc_gate_parameters:
                    parameter.grad = None
            if voc_gate_total_loss is not None:
                self.voc_gate_scaler.scale(voc_gate_total_loss).backward()
        else:
            total_loss.backward(retain_graph=(voc_gate_total_loss is not None))
            if voc_total_loss is not None:
                voc_total_loss.backward()
            if self.voc_dedicated_gate:
                for parameter in self.voc_gate_parameters:
                    parameter.grad = None
            if voc_gate_total_loss is not None:
                voc_gate_total_loss.backward()
        if timing is not None:
            timing.time("compute gradient")

        optimize_params = self.optimizer.param_groups[0]["params"]
        if self.flags.float16:
            self.scaler.unscale_(self.optimizer)
            if voc_total_loss is not None:
                self.voc_scaler.unscale_(self.voc_optimizer)
            if voc_gate_total_loss is not None:
                self.voc_gate_scaler.unscale_(self.voc_gate_optimizer)
        voc_step_result = None
        if voc_total_loss is not None:
            voc_step_result = self._step_voc_optimizer(T, B)
        voc_gate_step_result = None
        if voc_gate_total_loss is not None:
            voc_gate_step_result = self._step_voc_gate_optimizer()
        step_result = self._step_actor_optimizer(optimize_params, T, B)
        total_norm = step_result.total_norm
        voc_gate_projection_result = None
        if voc_step_result is not None and voc_step_result.optimizer_stepped:
            # The gate used theta_bar_t above.  Advance the frozen target only
            # after the online Q and actor steps; an actor AMP skip is separate
            # and does not suppress a successful critic target update.
            self._update_voc_ema_gate_target()
            if self.voc_gate_exact_projection:
                voc_gate_projection_result = (
                    self._project_voc_gate_head_to_ema_target()
                )
        if self.dynamic_voc_mode != "off":
            losses["voc_ema_gate_updated"] = torch.tensor(
                float(
                    voc_step_result is not None
                    and voc_step_result.optimizer_stepped
                ),
                device=total_loss.device,
            )
            losses["voc_ema_gate_update_count"] = torch.tensor(
                float(self.voc_ema_gate_update_count),
                device=total_loss.device,
            )
            losses["voc_ema_gate_parent_update_count"] = torch.tensor(
                float(self.voc_ema_gate_parent_update_count),
                device=total_loss.device,
            )
        if self.voc_dedicated_gate:
            gate_stepped = bool(
                voc_gate_step_result is not None
                and voc_gate_step_result.optimizer_stepped
            )
            if self.voc_gate_exact_projection and voc_gate_step_result is not None:
                raise RuntimeError(
                    "VoC exact projection must bypass the gate optimizer"
                )
            if gate_stepped:
                self.voc_gate_update_count += 1
            losses["voc_gate_optimizer_stepped"] = torch.tensor(
                float(gate_stepped), device=total_loss.device
            )
            losses["voc_gate_update_count"] = torch.tensor(
                float(self.voc_gate_update_count), device=total_loss.device
            )
            losses["voc_gate_total_norm"] = torch.tensor(
                float(
                    voc_gate_step_result.total_norm
                    if voc_gate_step_result is not None else 0.0
                ),
                device=total_loss.device,
            )
            losses["voc_gate_postclip_total_norm"] = torch.tensor(
                float(
                    self._last_voc_gate_postclip_total_norm
                    if voc_gate_step_result is not None else 0.0
                ),
                device=total_loss.device,
            )
            if self.voc_gate_exact_projection:
                losses["voc_gate_exact_projection_enabled"] = torch.tensor(
                    1.0, device=total_loss.device
                )
                losses["voc_gate_exact_projection_applied"] = torch.tensor(
                    float(voc_gate_projection_result is not None),
                    device=total_loss.device,
                )
                losses["voc_gate_projection_pre_error_norm"] = torch.tensor(
                    self._last_voc_gate_projection_pre_error_norm,
                    device=total_loss.device,
                )
                losses["voc_gate_projection_post_error_norm"] = torch.tensor(
                    self._last_voc_gate_projection_post_error_norm,
                    device=total_loss.device,
                )
        if timing is not None:
            timing.time("compute norm")

        if not step_result.optimizer_stepped:
            self._pending_action_prior_ema = None
            self._imitation_pending_update = False
            if "icopro_update_count" in losses:
                losses["icopro_update_count"] = torch.tensor(
                    float(self.imitation_update_count), device=total_loss.device
                )
        else:
            if self._pending_action_prior_ema is not None:
                self.action_prior_ema = self._pending_action_prior_ema
                self._pending_action_prior_ema = None
            if self._imitation_pending_update:
                self.imitation_update_count += 1
                self._imitation_pending_update = False
                if "icopro_update_count" in losses:
                    losses["icopro_update_count"] = torch.tensor(
                        float(self.imitation_update_count), device=total_loss.device
                    )
        if voc_step_result is not None and voc_step_result.optimizer_stepped:
            if not self._voc_pending_update:
                raise RuntimeError(
                    "VoC optimizer stepped without pending support counters"
                )
            self.voc_update_count += 1
            self.voc_continue_count += self._pending_voc_continue_count
            self.voc_stop_count += self._pending_voc_stop_count
            if self.voc_gate_exact_projection:
                self._assert_voc_gate_exact_projection_invariant()
        # Held-out rows are observations, not optimizer support.  Commit their
        # finite pre-update TD errors even for an all-held PPO minibatch (which
        # deliberately has no VoC Q step) or a recoverable VoC AMP skip.
        if self._pending_voc_holdout is not None:
            (
                holdout_count,
                holdout_continue_count,
                holdout_stop_count,
                td_sum,
                td_abs_sum,
                td_sq_sum,
            ) = self._pending_voc_holdout
            self.voc_holdout_count += holdout_count
            self.voc_holdout_continue_count += holdout_continue_count
            self.voc_holdout_stop_count += holdout_stop_count
            self.voc_holdout_td_sum += td_sum
            self.voc_holdout_td_abs_sum += td_abs_sum
            self.voc_holdout_td_sq_sum += td_sq_sum
        if self.dynamic_voc_mode != "off":
            losses["voc_update_count"] = torch.tensor(
                float(self.voc_update_count), device=total_loss.device
            )
            losses["voc_continue_count"] = torch.tensor(
                float(self.voc_continue_count), device=total_loss.device
            )
            losses["voc_stop_count"] = torch.tensor(
                float(self.voc_stop_count), device=total_loss.device
            )
            losses["voc_holdout_total_count"] = torch.tensor(
                float(self.voc_holdout_count), device=total_loss.device
            )
            if self.voc_holdout_count > 0:
                losses["voc_holdout_cumulative_td_bias"] = torch.tensor(
                    self.voc_holdout_td_sum / self.voc_holdout_count,
                    device=total_loss.device,
                )
                losses["voc_holdout_cumulative_td_mae"] = torch.tensor(
                    self.voc_holdout_td_abs_sum / self.voc_holdout_count,
                    device=total_loss.device,
                )
                losses["voc_holdout_cumulative_td_rmse"] = torch.tensor(
                    np.sqrt(
                        self.voc_holdout_td_sq_sum / self.voc_holdout_count
                    ),
                    device=total_loss.device,
                )
            if voc_step_result is not None:
                losses["voc_optimizer_stepped"] = torch.tensor(
                    float(voc_step_result.optimizer_stepped),
                    device=total_loss.device,
                )
                losses["voc_total_norm"] = torch.tensor(
                    float(voc_step_result.total_norm),
                    device=total_loss.device,
                )
        self._voc_pending_update = False
        self._pending_voc_continue_count = 0
        self._pending_voc_stop_count = 0
        self._pending_voc_holdout = None
        if timing is not None:
            timing.time("grad descent")
    
        self.scheduler.last_epoch = (
            max(self.real_step - 1, 0)
        )  # scheduler does not support setting epoch directly
        self.scheduler.step()
        if (
            self.voc_scheduler is not None
            and voc_step_result is not None
            and voc_step_result.optimizer_stepped
        ):
            self.voc_scheduler.last_epoch = max(self.real_step - 1, 0)
            self.voc_scheduler.step()
        if (
            self.voc_gate_scheduler is not None
            and voc_gate_step_result is not None
            and voc_gate_step_result.optimizer_stepped
        ):
            self.voc_gate_scheduler.last_epoch = max(self.real_step - 1, 0)
            self.voc_gate_scheduler.step()
        self._stage_schema13_post_scheduler(voc_step_result)
        self.anneal_c = 1.0 - util.schedule_progress(self.flags, self.real_step)
        
        if not self.ppo_enable or first_iter:
            # statistic output
            for k in losses:
                # Imitation metrics are already normalized over scored real
                # decisions; SEARCH/WAIT unroll size must not rescale them.
                if not (
                    k.startswith("icopro_") or k.startswith("voc_")
                ):
                    losses[k] = losses[k] / T / B
            total_norm = total_norm / T / B
            stats = self.compute_stat(train_actor_out, losses, total_norm, actor_id)
            stats["sps"] = self.sps

            # write to log file
            self.plogger.log(stats)

            # print statistics
            if self.timer() - self.start_time > 5:
                self.sps_buffer[self.sps_buffer_n] = (self.step, self.timer())
                self.sps_buffer_n = (self.sps_buffer_n + 1) % len(self.sps_buffer)
                self.sps = (
                    self.sps_buffer[self.sps_buffer_n - 1][0]
                    - self.sps_buffer[self.sps_buffer_n][0]
                ) / (
                    self.sps_buffer[self.sps_buffer_n - 1][1]
                    - self.sps_buffer[self.sps_buffer_n][1]
                )
                tot_sps = (self.step - self.sps_start_step) / (
                    self.timer() - self.sps_start_time
                )
                print_str = (
                    "\033[1;34m[%s] Steps %i @ %.1f SPS (%.1f). (T_q: %.2f) Eps %i. \033[0m"
                    "Ret \033[1;31m%f\033[0m (%f/%f). Loss %.2f"
                    % (
                        self.flags.xpid,
                        self.real_step,
                        self.sps,
                        tot_sps,
                        self.queue_n,
                        self.tot_eps,
                        stats["rmean_episode_return"],
                        stats.get("rmean_im_episode_return", 0.),
                        stats.get("rmean_cur_episode_return", 0.),
                        total_loss.detach().item() / T / B,
                    )
                )
                print_stats = [
                    "actor/pg_loss",
                    "actor/entropy_loss",
                    "actor/reg_loss",
                    "actor/total_norm",
                    "actor/mean_abs_v",
                ]
                for k in print_stats:
                    print_str += " %s %.2f" % (k.replace("actor/", ""), stats[k])
                if "actor/action_prior_loss" in stats:
                    print_str += " action_prior_loss %.4f" % (
                        stats["actor/action_prior_loss"]
                    )
                if "actor/icopro_behavioral_support_count" in stats:
                    print_str += " bc_acc %.3f/%.3f" % (
                        stats["actor/icopro_behavioral_argmax_accuracy"],
                        stats["actor/icopro_behavioral_sampled_accuracy"],
                    )
                    if stats.get("actor/icopro_noop_supported", 0.0) > 0.0:
                        print_str += " noop_h/a/s %.3f/%.3f/%.3f" % (
                            stats["actor/icopro_target_noop_frequency"],
                            stats["actor/icopro_argmax_noop_frequency"],
                            stats["actor/icopro_sampled_noop_frequency"],
                        )
                    else:
                        print_str += " noop_h/a/s unsupported"
                if self.flags.return_norm_type in [0, 1]:
                    print_str += " norm_diff %.4f/%.4f" % (
                        stats["actor/norm_diff"],
                        stats.get("actor/im_norm_diff", 0.),
                    )
                    print_str += " cur_norm_diff %.4f" % (
                        stats.get("actor/cur_norm_diff", 0.),
                    )
                    if self.dynamic_search:
                        print_str += " think_ret %.4f search_len %.2f" % (
                            stats.get("rmean_think_episode_return", 0.),
                            stats.get("search/mean_steps", 0.),
                        )
                if self.ppo_enable:
                    print_str += " kl_beta %.4f" % self.actor_net.kl_beta
                    print_str += " kl_loss %.4f" % losses["kl_loss"]
                    print_str += " is_abs %.4f" % np.mean(self.ppo_is_abs)

                print_str += " last_lr: %.4e"  % self.optimizer.param_groups[0]['lr']

                # dbg_adv = torch.concat(list(self.dbg_adv))
                # print_str += " dbg_adv mean %.4f std %.4f abs %.4f" % (torch.mean(dbg_adv), torch.std(dbg_adv), torch.mean(torch.abs(dbg_adv)))

                self._logger.info(print_str)
                self.start_time = self.timer()
                self.queue_n = 0
                if timing is not None:
                    print(timing.summary())

            # 시간 기반 체크포인트 저장
            if int(time.strftime("%M")) // 10 != self.ckp_start_time:
                self.save_checkpoint()
                self.ckp_start_time = int(time.strftime("%M")) // 10
                
            # step 기반 체크포인트 저장
            has_interval = hasattr(self.flags, 'checkpoint_interval')
            if has_interval:
                interval = self.flags.checkpoint_interval
                if interval > 0:
                    # 더 직관적인 방법: 현재 step이 어떤 마일스톤에 속하는지 확인
                    current_milestone = (self.real_step // interval) * interval
                    next_milestone = current_milestone + interval
                    
                    # 정적 변수를 사용하여 마일스톤 지남 여부 추적
                    if not hasattr(self, 'last_checkpoint_milestone'):
                        self.last_checkpoint_milestone = -1
                    
                    # 새로운 마일스톤에 도달했는지 확인
                    milestone_reached = current_milestone > self.last_checkpoint_milestone
                    
                    #self._logger.info(f"Actor step checkpoint check: has_interval={has_interval}, interval={interval}, real_step={self.real_step}, current_milestone={current_milestone}, last_milestone={self.last_checkpoint_milestone}, milestone_reached={milestone_reached}")
                    
                    if milestone_reached:
                        self._logger.info(f"Triggering actor step-based checkpoint at step {self.real_step} (milestone {current_milestone})")
                        self.save_checkpoint(force=True)
                        self.last_checkpoint_milestone = current_milestone
            del train_actor_out, losses, total_loss, stats, total_norm
        else:
            del train_actor_out, losses, total_loss, total_norm

        if timing is not None:
            timing.time("misc")
        
        torch.cuda.empty_cache()

        # update shared buffer's weights
        self.n += 1
        r = self.real_step > self.flags.total_steps
        return r

    def compute_losses(self, train_actor_out, initial_actor_state, first_iter=True, last_iter=False):
        # compute loss and then discard the first step in train_actor_out

        T, B = train_actor_out.done.shape
        T = T - 1        
        voc_actor_id = train_actor_out.id
        if self.voc_gate_exact_projection:
            self._assert_voc_gate_exact_projection_invariant()
        
        if self.disable_thinker:
            clamp_action = train_actor_out.pri[1:]
        elif self.dynamic_search:
            search_control = getattr(train_actor_out, "search_control", None)
            if search_control is None:
                search_control = train_actor_out.reset
            clamp_action = (train_actor_out.pri[1:], search_control[1:])
        else:
            clamp_action = (train_actor_out.pri[1:], train_actor_out.reset[1:])
        
        new_actor_out, _ = self.actor_net(
            train_actor_out, 
            initial_actor_state,
            clamp_action = clamp_action,
            compute_loss = True,
        )

        # Take final value function slice for bootstrapping.
        if not self.ppo_enable:
            bootstrap_value = new_actor_out.baseline[-1]     
        else:
            bootstrap_value = train_actor_out.baseline[-1]    

        # Actor replay rows pair action a_k with the post-step EnvOut at
        # obs_{k+1}.  After the standard [1:]/[:-1] alignment below, the
        # pre-decision accepted control for a_{k+1} is therefore the EnvOut
        # token from the preceding, unshifted row.  Preserve it here so
        # post-compute VoC diagnostics also cover the first optimized row.
        voc_predecision_last_control = None
        if self.dynamic_voc_mode != "off":
            last_search_control = getattr(
                train_actor_out, "last_search_control", None
            )
            if last_search_control is None:
                raise RuntimeError(
                    "VoC replay is missing EnvOut.last_search_control"
                )
            voc_predecision_last_control = last_search_control[:-1]
    
        # Move from obs[t] -> action[t] to action[t] -> obs[t].
        train_actor_out = util.tuple_map(train_actor_out, lambda x: x[1:])
        new_actor_out = util.tuple_map(new_actor_out, lambda x: x[:-1])

        if self.ppo_enable:
            # record base policy for ppo
            base_actor_out = train_actor_out
            if self.actor_net.discrete_action:
                base_pri_logits = base_actor_out.pri_param.detach()
            else:
                pri_param = base_actor_out.pri_param.detach()
                base_pri_mean = pri_param[:, :, :, 0]
                base_pri_log_var = pri_param[:, :, :, 1]
            if not self.disable_thinker:
                if self.dynamic_search:
                    base_control_logits = getattr(
                        base_actor_out, "search_control_logits", None
                    )
                    if base_control_logits is None:
                        base_control_logits = base_actor_out.reset_logits
                    base_control_logits = base_control_logits.detach()
                else:
                    base_reset_logits = base_actor_out.reset_logits.detach()
        rewards = train_actor_out.reward

        # compute advantage and baseline        
        pg_losses = []
        baseline_losses = []
        done = train_actor_out.done | train_actor_out.truncated_done
        if self.dynamic_search:
            real_transition = train_actor_out.real_transition.bool()
            stage_end = train_actor_out.stage_end.bool()

            def dynamic_field(name):
                value = getattr(train_actor_out, name, None)
                if value is None:
                    value = getattr(new_actor_out, name, None)
                if value is None:
                    raise RuntimeError(
                        "dynamic_search rollout is missing ActorOut.%s" % name
                    )
                return value

            policy_valid = dynamic_field("policy_valid").bool()
            primary_valid = dynamic_field("primary_valid").bool()
            control_valid = dynamic_field("control_valid").bool()
            policy_type = dynamic_field("policy_type").long()
            real_policy_mask = policy_valid & (policy_type == util.POLICY_REAL)
            search_policy_mask = policy_valid & (policy_type == util.POLICY_SEARCH)

            # Augmented SEARCH/NEED_REAL/WAIT calls are zero-time transitions.
            # Apply the environment discount once, on the call that actually
            # crosses the synchronous real-step barrier.
            main_discount = (~done).float() * torch.where(
                real_transition,
                torch.full_like(rewards[:, :, 0], self.flags.discounting),
                torch.ones_like(rewards[:, :, 0]),
            )
            stage_discount = (~(done | stage_end)).float()
            discount_by_prefix = {
                "re": main_discount,
                "im": stage_discount,
                "cur": main_discount,
                "think": stage_discount,
            }
            pg_mask_by_prefix = {
                "re": policy_valid,
                "im": search_policy_mask,
                "cur": policy_valid,
                "think": search_policy_mask,
            }
            # WAIT states still need a task-value target: their discount-one,
            # reward-zero transitions carry credit back to STOP and the stored
            # real action.  Stage-local critics only have meaning in SEARCH.
            baseline_mask_by_prefix = {
                "re": None,
                "im": search_policy_mask,
                "cur": None,
                "think": search_policy_mask,
            }
            discounts = [discount_by_prefix[p] for p in self.rewards_ls]
            pg_masks = [pg_mask_by_prefix[p] for p in self.rewards_ls]
            baseline_masks = [baseline_mask_by_prefix[p] for p in self.rewards_ls]
            last_step_real = real_transition
        else:
            discounts = [(~done).float() * self.im_discounting]
            pg_masks = [None]
            baseline_masks = [None]

            last_step_real = ((train_actor_out.step_status == 0)
                              | (train_actor_out.step_status == 3))
            next_step_real = ((train_actor_out.step_status == 2)
                              | (train_actor_out.step_status == 3))
            if self.flags.im_cost > 0.:
                discounts.append((~next_step_real).float() * self.im_discounting)
                pg_masks.append((~last_step_real).float())
                baseline_masks.append((~last_step_real).float())
            if self.flags.cur_cost > 0.:
                discounts.append((~done).float() * self.im_discounting)
                pg_masks.append(None)
                baseline_masks.append(None)

        if self.dynamic_search and self.dynamic_factorized_control:
            behavior_log_prob_by_prefix = dynamic_factorized_policy_log_probs(
                train_actor_out,
                control_valid,
                primary_valid,
                discrete_action=self.actor_net.discrete_action,
                tanh_action=bool(getattr(self.flags, "tanh_action", False)),
            )
            target_log_prob_by_prefix = dynamic_factorized_policy_log_probs(
                new_actor_out,
                control_valid,
                primary_valid,
                discrete_action=self.actor_net.discrete_action,
                tanh_action=bool(getattr(self.flags, "tanh_action", False)),
            )
        else:
            behavior_log_prob_by_prefix = {
                prefix: train_actor_out.c_action_log_prob
                for prefix in self.rewards_ls
            }
            target_log_prob_by_prefix = {
                prefix: new_actor_out.c_action_log_prob
                for prefix in self.rewards_ls
            }

        # Keep the full behavior/target likelihoods above for V-trace.  Only
        # the legacy reward-channel PG backward route is altered in control
        # mode; exact forward likelihood values are preserved by detach.
        behavior_pg_log_prob_by_prefix = dynamic_voc_policy_log_probs(
            behavior_log_prob_by_prefix,
            self.dynamic_voc_mode,
            detach_cur_gate=self.voc_dedicated_gate,
        )
        target_pg_log_prob_by_prefix = dynamic_voc_policy_log_probs(
            target_log_prob_by_prefix,
            self.dynamic_voc_mode,
            detach_cur_gate=self.voc_dedicated_gate,
        )

        log_rhos_by_prefix = {}
        for i, prefix in enumerate(self.rewards_ls):
            if not self.ppo_enable or self.flags.ppo_v_trace:
                prefix_log_rho = (
                    target_log_prob_by_prefix[prefix]
                    - behavior_log_prob_by_prefix[prefix]
                )
                if self.dynamic_search:
                    # No action was sampled on WAIT. rho=1 keeps the identity
                    # transition in V-trace without inventing a likelihood.
                    prefix_log_rho = torch.where(
                        pg_masks[i],
                        prefix_log_rho,
                        torch.zeros_like(prefix_log_rho),
                    )
            else:
                prefix_log_rho = torch.zeros_like(
                    train_actor_out.c_action_log_prob
                )
            log_rhos_by_prefix[prefix] = prefix_log_rho

        v_trace_by_prefix = {}
        for i in range(self.num_rewards):
            prefix = self.rewards_ls[i]
            prefix_rewards = rewards[:, :, i]
            behavior_log_prob = behavior_pg_log_prob_by_prefix[prefix]
            target_log_prob = target_pg_log_prob_by_prefix[prefix]

            if self.flags.entropy_r_cost > 0. and prefix == "re":
                if self.dynamic_search:
                    prefix_rewards = prefix_rewards.clone()
                    prefix_rewards[real_policy_mask] += (
                        -self.flags.entropy_r_cost
                        * train_actor_out.c_action_log_prob[real_policy_mask]
                    )
                else:
                    prefix_rewards[last_step_real] += (
                        -self.flags.entropy_r_cost
                        * train_actor_out.c_action_log_prob[last_step_real]
                    )

            return_norm_type=self.flags.return_norm_type
            if not self.ppo_enable:
                values = new_actor_out.baseline[:, :, i]
            else:
                values = train_actor_out.baseline[:, :, i]
            v_trace = compute_v_trace(
                log_rhos=log_rhos_by_prefix[prefix],
                discounts=discounts[i],
                rewards=prefix_rewards,
                values=values,
                bootstrap_value=bootstrap_value[:, i],
                return_norm_type=return_norm_type,
                norm_stat=self.norm_stats[i],
                lamb=self.flags.v_trace_lamb,
                norm_mask=(
                    pg_masks[i] if self.dynamic_search else None
                ),
            )
            v_trace_by_prefix[prefix] = v_trace
            self.norm_stats[i] = v_trace.norm_stat
            if self.ppo_enable:
                log_is_de = behavior_log_prob
                adv = v_trace.pg_advantages_nois.detach()
                log_is_de = log_is_de.detach()
                vs = v_trace.vs.detach()

            if not self.ppo_enable:
                adv = v_trace.pg_advantages.detach()
                pg_loss = -adv * target_log_prob
            else:
                log_is = target_log_prob - log_is_de
                if self.dynamic_search:
                    log_is = torch.where(
                        pg_masks[i], log_is, torch.zeros_like(log_is)
                    )
                unclipped_is = torch.exp(log_is)
                if self.dynamic_search:
                    if torch.any(pg_masks[i]):
                        self.ppo_is_abs.append(
                            torch.mean(torch.abs(unclipped_is[pg_masks[i]] - 1))
                            .detach().item()
                        )
                    else:
                        self.ppo_is_abs.append(0.)
                else:
                    self.ppo_is_abs.append(torch.mean(torch.abs(unclipped_is-1)).detach().item())
                clipped_is = torch.clamp(unclipped_is, 1-self.flags.ppo_clip, 1+self.flags.ppo_clip)
                pg_loss = -torch.minimum(unclipped_is * adv, clipped_is * adv)

            if pg_masks[i] is not None: pg_loss = pg_loss * pg_masks[i]
            pg_loss = torch.sum(pg_loss)

            vs = v_trace.vs if not self.ppo_enable else vs
            pg_losses.append(pg_loss)
            if self.flags.critic_enc_type == 0:
                baseline_loss = compute_baseline_loss(
                    baseline=new_actor_out.baseline[:, :, i],
                    target_baseline=vs,
                    mask=baseline_masks[i]
                )
            else:
                baseline_loss = compute_baseline_enc_loss(
                    baseline_enc=new_actor_out.baseline_enc[:, :, i],
                    target_baseline=vs,
                    rv_tran=self.actor_net.rv_tran,
                    enc_type=self.flags.critic_enc_type,
                    mask=baseline_masks[i]
                )

            baseline_losses.append(baseline_loss)

        # sum all the losses
        total_loss = pg_losses[0] / self.actor_net.dim_actions
        total_loss += self.flags.baseline_cost * baseline_losses[0]

        losses = {
            "pg_loss": pg_losses[0],
            "baseline_loss": baseline_losses[0]
        }
        n = 0
        for prefix in self.rewards_ls[1:]:
            cost = getattr(self.flags, "%s_cost" % prefix)
            n += 1
            if getattr(self.flags, "%s_cost_anneal" % prefix):
                cost *= self.anneal_c
            total_loss += cost * pg_losses[n] / self.actor_net.dim_actions
            total_loss += (cost * self.flags.baseline_cost * baseline_losses[n])
            losses["%s_pg_loss" % prefix] = pg_losses[n]
            losses["%s_baseline_loss" % prefix] = baseline_losses[n]

        if self.dynamic_voc_mode != "off":
            voc_q = getattr(new_actor_out, "voc_q", None)
            if voc_q is None:
                raise RuntimeError(
                    "VoC mode requires differentiable ActorOut.voc_q"
                )
            behavior_control_logits = getattr(
                train_actor_out, "search_control_logits", None
            )
            if behavior_control_logits is None:
                if self.voc_gate_epsilon_greedy_execution:
                    raise RuntimeError(
                        "epsilon-greedy VoC execution requires stored "
                        "behavior control logits"
                    )
                behavior_control_logits = train_actor_out.reset_logits
            target_behavior_control_logits = getattr(
                new_actor_out, "search_control_logits", None
            )
            if target_behavior_control_logits is None:
                if self.voc_gate_epsilon_greedy_execution:
                    raise RuntimeError(
                        "epsilon-greedy VoC execution requires target "
                        "execution control logits"
                    )
                target_behavior_control_logits = new_actor_out.reset_logits
            control_action = getattr(train_actor_out, "search_control", None)
            if control_action is None:
                control_action = train_actor_out.reset
            actor_misc = getattr(new_actor_out, "misc", None)
            (
                target_control_logits,
                voc_gate_soft_continue_probability,
            ) = resolve_dynamic_voc_learning_control_surface(
                execution_control_logits=target_behavior_control_logits,
                actor_misc=actor_misc,
                control_valid=control_valid,
                epsilon_greedy_execution=(
                    self.voc_gate_epsilon_greedy_execution
                ),
            )

            task_v_trace = v_trace_by_prefix["re"]
            # entropy_r_cost is an optional actor regularizer, not environment
            # return.  Recompute the recursive task target from raw rewards if
            # it was injected into the ordinary task V-trace above.
            if self.flags.entropy_r_cost > 0.0:
                task_i = self.rewards_ls.index("re")
                task_values = (
                    new_actor_out.baseline[:, :, task_i]
                    if not self.ppo_enable
                    else train_actor_out.baseline[:, :, task_i]
                )
                task_v_trace = compute_v_trace(
                    log_rhos=log_rhos_by_prefix["re"],
                    discounts=discount_by_prefix["re"],
                    rewards=rewards[:, :, task_i],
                    values=task_values,
                    bootstrap_value=bootstrap_value[:, task_i],
                    # Normalization changes only PG advantages.  Disable it
                    # here to avoid mutating the ordinary return-stat buffer.
                    return_norm_type=-1,
                    norm_stat=None,
                    lamb=self.flags.v_trace_lamb,
                    norm_mask=policy_valid,
                )
            think_i = self.rewards_ls.index("think")
            task_i = self.rewards_ls.index("re")
            effective_think_cost = float(self.flags.think_cost)
            if self.flags.think_cost_anneal:
                effective_think_cost *= self.anneal_c
            voc_value_source = (
                new_actor_out.baseline
                if not self.ppo_enable
                else train_actor_out.baseline
            )
            voc_state_value = (
                voc_value_source[:, :, task_i]
                + effective_think_cost
                * voc_value_source[:, :, think_i]
            )
            voc_target_parts = compute_dynamic_voc_target(
                task_rewards=rewards[:, :, task_i],
                think_rewards=rewards[:, :, think_i],
                task_discounts=discount_by_prefix["re"],
                think_discounts=discount_by_prefix["think"],
                task_vs=task_v_trace.vs,
                think_vs=v_trace_by_prefix["think"].vs,
                task_bootstrap_value=bootstrap_value[:, task_i],
                think_bootstrap_value=bootstrap_value[:, think_i],
                think_cost=effective_think_cost,
            )
            voc_holdout_valid = dynamic_voc_holdout_mask(
                voc_actor_id,
                control_valid,
                total_actor_streams=(
                    int(self.flags.self_play_n) * int(self.flags.env_n)
                ),
            )
            voc_q_train_valid = control_valid & ~voc_holdout_valid
            voc_result = compute_dynamic_voc_loss(
                voc_q=voc_q,
                target_control_logits=target_control_logits,
                target_behavior_control_logits=(
                    target_behavior_control_logits
                ),
                behavior_control_logits=behavior_control_logits,
                control_action=control_action,
                control_valid=control_valid,
                voc_target=voc_target_parts.net,
                # Online Q is regression/calibration only.  The gate objective
                # below is driven exclusively by the frozen EMA head.
                mode="shadow",
                q_train_valid=voc_q_train_valid,
                dueling_q=self.voc_dueling_q,
                voc_state_value=voc_state_value,
                expected_gate_loss=self.voc_expected_gate_loss,
                gate_policy_schema_version=(
                    self.voc_gate_policy_schema_version
                ),
            )
            train_valid = voc_result.q_train_valid
            train_count = int(train_valid.sum().detach().cpu())
            voc_features = getattr(new_actor_out, "voc_features", None)
            if voc_features is None:
                raise RuntimeError(
                    "active VoC requires loss-only ActorOut.voc_features"
                )
            voc_gate_loss, voc_gate_q = self._compute_ema_gate_loss(
                features=voc_features,
                logits=target_control_logits,
                valid=voc_result.valid,
                state_value=voc_state_value,
                enable_policy_loss=not self.voc_soft_q_bce_gate,
            )
            voc_soft_gate_result = None
            voc_gate_param_alignment = None
            voc_gate_behavior_entropy = None
            if (
                self.voc_soft_q_bce_gate
                and self.dynamic_voc_mode == "control"
            ):
                if not isinstance(actor_misc, Mapping):
                    raise RuntimeError(
                        "dedicated VoC control requires ActorOut.misc"
                    )
                gate_log_odds = actor_misc.get("voc_gate_log_odds")
                if gate_log_odds is None:
                    raise RuntimeError(
                        "dedicated VoC control requires loss-only "
                        "misc['voc_gate_log_odds']"
                    )
                voc_soft_gate_result = compute_dynamic_voc_soft_q_gate_loss(
                    gate_log_odds=gate_log_odds,
                    q_values=voc_gate_q,
                    # Held-out actor streams are calibration-only for both
                    # critic and policy.  They must never train the dedicated
                    # gate, even though their EMA-Q diagnostics remain logged.
                    valid=voc_result.q_train_valid,
                    q_temperature=self.voc_gate_q_temperature,
                    policy_temperature=float(
                        self.flags.voc_gate_temperature
                    ),
                    confidence_weighted=(
                        self.voc_gate_confidence_weighted
                    ),
                )
                voc_gate_loss = voc_soft_gate_result.loss
                if self.voc_gate_param_align and train_count > 0:
                    gate_head = self.voc_gate_head_modules[0]
                    if not isinstance(gate_head, torch.nn.Linear):
                        raise TypeError(
                            "VoC parameter alignment requires a linear "
                            "voc_gate_head"
                        )
                    voc_gate_param_alignment = (
                        compute_dynamic_voc_gate_parameter_alignment_loss(
                            gate_weight=gate_head.weight,
                            gate_bias=gate_head.bias,
                            ema_q_weight=self.voc_ema_gate_weight,
                            ema_q_bias=self.voc_ema_gate_bias,
                            q_temperature=self.voc_gate_q_temperature,
                            policy_temperature=float(
                                self.flags.voc_gate_temperature
                            ),
                        )
                    )
                    voc_gate_loss = voc_gate_loss + (
                        self.voc_gate_param_align_coef
                        * voc_gate_param_alignment.loss
                    )
                    _require_finite_tensor(
                        "VoC dedicated gate BCE plus parameter alignment",
                        voc_gate_loss,
                    )
                voc_gate_behavior_entropy = actor_misc.get(
                    "voc_gate_entropy"
                )
                if voc_gate_behavior_entropy is not None:
                    if tuple(voc_gate_behavior_entropy.shape) != tuple(
                        voc_result.valid.shape
                    ):
                        raise ValueError(
                            "misc['voc_gate_entropy'] must match the VoC "
                            "control mask"
                        )
                    _require_finite_tensor(
                        "VoC dedicated gate entropy",
                        voc_gate_behavior_entropy[voc_result.valid],
                    )
            elif self.dynamic_voc_mode == "control":
                total_loss = total_loss + voc_gate_loss
            self._stage_schema13_voc_sources(
                voc_result=voc_result,
                ema_q_values=voc_gate_q,
                train_mask=voc_result.q_train_valid,
                holdout_mask=voc_holdout_valid,
                control_action=control_action,
                search_steps=train_actor_out.search_steps,
            )
            valid_count = int(voc_result.valid.sum().detach().cpu())
            continue_mask = train_valid & (voc_result.gate_action == 0)
            stop_mask = train_valid & (voc_result.gate_action == 1)
            continue_count = int(continue_mask.sum().detach().cpu())
            stop_count = int(stop_mask.sum().detach().cpu())
            holdout_continue_mask = voc_holdout_valid & (
                voc_result.gate_action == 0
            )
            holdout_stop_mask = voc_holdout_valid & (
                voc_result.gate_action == 1
            )
            holdout_count = int(voc_holdout_valid.sum().detach().cpu())
            holdout_continue_count = int(
                holdout_continue_mask.sum().detach().cpu()
            )
            holdout_stop_count = int(
                holdout_stop_mask.sum().detach().cpu()
            )
            if holdout_count > 0:
                holdout_td = voc_result.td_error.detach()[voc_holdout_valid]
                _require_finite_tensor("VoC held-out TD error", holdout_td)
                self._pending_voc_holdout = (
                    holdout_count,
                    holdout_continue_count,
                    holdout_stop_count,
                    float(holdout_td.double().sum().cpu()),
                    float(holdout_td.double().abs().sum().cpu()),
                    float(holdout_td.double().square().sum().cpu()),
                )
            if train_count > 0:
                if self._voc_pending_update:
                    raise RuntimeError(
                        "VoC pending counters were not consumed before a new loss"
                    )
                self._voc_pending_update = True
                self._pending_voc_continue_count = continue_count
                self._pending_voc_stop_count = stop_count
                losses["_voc_total_loss"] = (
                    float(self.flags.voc_loss_cost) * voc_result.q_loss
                )
            if (
                voc_soft_gate_result is not None
                and train_count > 0
                and not self.voc_gate_exact_projection
            ):
                losses["_voc_gate_total_loss"] = voc_gate_loss
            def voc_mean(value, mask=None):
                use_mask = voc_result.valid if mask is None else mask
                selected = value.detach()[use_mask]
                if selected.numel() == 0:
                    return torch.zeros(
                        (), device=total_loss.device, dtype=total_loss.dtype
                    )
                return selected.float().mean().to(dtype=total_loss.dtype)

            def voc_rmse(value, mask=None):
                use_mask = voc_result.valid if mask is None else mask
                selected = value.detach()[use_mask]
                if selected.numel() == 0:
                    return torch.zeros(
                        (), device=total_loss.device, dtype=total_loss.dtype
                    )
                return selected.float().square().mean().sqrt().to(
                    dtype=total_loss.dtype
                )

            def voc_correlation(left, right, mask):
                left_selected = left.detach()[mask].float()
                right_selected = right.detach()[mask].float()
                count = left_selected.numel()
                zero = torch.zeros(
                    (), device=total_loss.device, dtype=total_loss.dtype
                )
                count_tensor = torch.tensor(
                    float(count), device=total_loss.device,
                    dtype=total_loss.dtype,
                )
                if count < 2:
                    return zero, count_tensor, zero
                left_centered = left_selected - left_selected.mean()
                right_centered = right_selected - right_selected.mean()
                denominator_corr = torch.sqrt(
                    left_centered.square().sum()
                    * right_centered.square().sum()
                )
                if denominator_corr.item() == 0.0:
                    return zero, count_tensor, zero
                correlation = (
                    (left_centered * right_centered).sum()
                    / denominator_corr
                ).clamp(-1.0, 1.0)
                return (
                    correlation.to(dtype=total_loss.dtype),
                    count_tensor,
                    torch.ones_like(zero),
                )

            denominator = max(train_count, 1)
            gate_denominator = max(valid_count, 1)
            greedy_agreement = (
                (voc_result.greedy_action == voc_result.gate_action).float()
            )
            voc_gate_delta_q = voc_gate_q[..., 0] - voc_gate_q[..., 1]
            voc_online_gate_gap = voc_gate_q - voc_result.q_values.detach()
            voc_gate_selected_q = torch.gather(
                voc_gate_q,
                dim=-1,
                index=voc_result.gate_action.unsqueeze(-1),
            ).squeeze(-1)
            voc_gate_td_error = voc_result.target.detach() - voc_gate_selected_q
            voc_gate_greedy_action = torch.argmax(voc_gate_q, dim=-1)
            voc_gate_greedy_agreement = (
                voc_gate_greedy_action == voc_result.gate_action
            ).float()
            voc_gate_nontie = voc_result.valid & (
                voc_gate_delta_q.abs() > 1e-6
            )
            voc_online_nontie = voc_result.valid & (
                voc_result.delta_q.detach().abs() > 1e-6
            )
            voc_delta_sign_support = voc_gate_nontie & voc_online_nontie
            voc_delta_sign_agreement = (
                torch.sign(voc_gate_delta_q)
                == torch.sign(voc_result.delta_q.detach())
            ).float()
            (
                voc_delta_correlation,
                voc_delta_correlation_count,
                voc_delta_correlation_defined,
            ) = voc_correlation(
                voc_gate_delta_q, voc_result.delta_q, voc_result.valid
            )
            (
                voc_holdout_delta_correlation,
                voc_holdout_delta_correlation_count,
                voc_holdout_delta_correlation_defined,
            ) = voc_correlation(
                voc_gate_delta_q, voc_result.delta_q, voc_holdout_valid
            )
            losses.update({
                "voc_q_loss": (
                    voc_result.q_loss.detach() / denominator
                ),
                "voc_gate_pg_loss": (
                    voc_soft_gate_result.loss.detach()
                    if voc_soft_gate_result is not None
                    else voc_gate_loss.detach() / gate_denominator
                ),
                "voc_q_continue": voc_mean(voc_result.q_values[..., 0]),
                "voc_q_stop": voc_mean(voc_result.q_values[..., 1]),
                "voc_delta_q": voc_mean(voc_result.delta_q),
                "voc_gate_q_continue": voc_mean(voc_gate_q[..., 0]),
                "voc_gate_q_stop": voc_mean(voc_gate_q[..., 1]),
                "voc_gate_delta_q": voc_mean(voc_gate_delta_q),
                "voc_gate_online_q_gap_mae": voc_mean(
                    voc_online_gate_gap.abs().mean(dim=-1)
                ),
                "voc_gate_online_delta_gap": voc_mean(
                    voc_gate_delta_q - voc_result.delta_q.detach()
                ),
                "voc_gate_online_delta_gap_abs": voc_mean(
                    (voc_gate_delta_q - voc_result.delta_q.detach()).abs()
                ),
                "voc_gate_td_bias": voc_mean(voc_gate_td_error),
                "voc_gate_td_mae": voc_mean(voc_gate_td_error.abs()),
                "voc_gate_td_rmse": voc_rmse(voc_gate_td_error),
                "voc_gate_greedy_agreement": voc_mean(
                    voc_gate_greedy_agreement, voc_gate_nontie
                ),
                "voc_gate_greedy_agreement_count": torch.tensor(
                    float(voc_gate_nontie.sum().item()),
                    device=total_loss.device,
                ),
                "voc_gate_greedy_agreement_defined": torch.tensor(
                    float(voc_gate_nontie.any().item()),
                    device=total_loss.device,
                ),
                "voc_gate_online_delta_sign_agreement": voc_mean(
                    voc_delta_sign_agreement, voc_delta_sign_support
                ),
                "voc_gate_online_delta_sign_agreement_count": torch.tensor(
                    float(voc_delta_sign_support.sum().item()),
                    device=total_loss.device,
                ),
                "voc_gate_online_delta_sign_agreement_defined": torch.tensor(
                    float(voc_delta_sign_support.any().item()),
                    device=total_loss.device,
                ),
                "voc_gate_online_delta_correlation": voc_delta_correlation,
                "voc_gate_online_delta_correlation_count": (
                    voc_delta_correlation_count
                ),
                "voc_gate_online_delta_correlation_defined": (
                    voc_delta_correlation_defined
                ),
                "voc_gate_holdout_td_bias": voc_mean(
                    voc_gate_td_error, voc_holdout_valid
                ),
                "voc_gate_holdout_td_mae": voc_mean(
                    voc_gate_td_error.abs(), voc_holdout_valid
                ),
                "voc_gate_holdout_td_rmse": voc_rmse(
                    voc_gate_td_error, voc_holdout_valid
                ),
                "voc_gate_holdout_greedy_agreement": voc_mean(
                    voc_gate_greedy_agreement,
                    voc_holdout_valid & voc_gate_nontie,
                ),
                "voc_gate_holdout_greedy_agreement_count": torch.tensor(
                    float((voc_holdout_valid & voc_gate_nontie).sum().item()),
                    device=total_loss.device,
                ),
                "voc_gate_holdout_greedy_agreement_defined": torch.tensor(
                    float((voc_holdout_valid & voc_gate_nontie).any().item()),
                    device=total_loss.device,
                ),
                "voc_gate_holdout_online_delta_sign_agreement": voc_mean(
                    voc_delta_sign_agreement,
                    voc_holdout_valid & voc_delta_sign_support,
                ),
                "voc_gate_holdout_online_delta_sign_agreement_count": (
                    torch.tensor(
                        float(
                            (
                                voc_holdout_valid & voc_delta_sign_support
                            ).sum().item()
                        ),
                        device=total_loss.device,
                    )
                ),
                "voc_gate_holdout_online_delta_sign_agreement_defined": (
                    torch.tensor(
                        float(
                            (
                                voc_holdout_valid & voc_delta_sign_support
                            ).any().item()
                        ),
                        device=total_loss.device,
                    )
                ),
                "voc_gate_holdout_online_delta_correlation": (
                    voc_holdout_delta_correlation
                ),
                "voc_gate_holdout_online_delta_correlation_count": (
                    voc_holdout_delta_correlation_count
                ),
                "voc_gate_holdout_online_delta_correlation_defined": (
                    voc_holdout_delta_correlation_defined
                ),
                "voc_gate_target_tau": torch.tensor(
                    self.voc_gate_target_tau, device=total_loss.device
                ),
                "voc_gate_target_source_update_count": torch.tensor(
                    float(self.voc_ema_gate_update_count),
                    device=total_loss.device,
                ),
                "voc_target": voc_mean(voc_result.target),
                "voc_task_target": voc_mean(voc_target_parts.task),
                "voc_think_target": voc_mean(voc_target_parts.think),
                "voc_selected_advantage": voc_mean(
                    voc_result.selected_advantage
                ),
                "voc_td_bias": voc_mean(voc_result.td_error),
                "voc_td_mae": voc_mean(voc_result.td_error.abs()),
                "voc_td_rmse": voc_rmse(voc_result.td_error),
                "voc_continue_td_mae": voc_mean(
                    voc_result.td_error.abs(), continue_mask
                ),
                "voc_stop_td_mae": voc_mean(
                    voc_result.td_error.abs(), stop_mask
                ),
                "voc_continue_probability": voc_mean(
                    voc_result.continue_probability
                ),
                "voc_gate_rho": voc_mean(voc_result.gate_rho),
                "voc_greedy_agreement": voc_mean(greedy_agreement),
                "voc_continue_support": torch.tensor(
                    float(continue_count), device=total_loss.device
                ),
                "voc_stop_support": torch.tensor(
                    float(stop_count), device=total_loss.device
                ),
                "voc_continue_support_rate": torch.tensor(
                    continue_count / denominator, device=total_loss.device
                ),
                "voc_stop_support_rate": torch.tensor(
                    stop_count / denominator, device=total_loss.device
                ),
                "voc_holdout_td_bias": voc_mean(
                    voc_result.td_error, voc_holdout_valid
                ),
                "voc_holdout_td_mae": voc_mean(
                    voc_result.td_error.abs(), voc_holdout_valid
                ),
                "voc_holdout_td_rmse": voc_rmse(
                    voc_result.td_error, voc_holdout_valid
                ),
                "voc_holdout_count": torch.tensor(
                    float(holdout_count), device=total_loss.device
                ),
                "voc_holdout_continue_support": torch.tensor(
                    float(holdout_continue_count), device=total_loss.device
                ),
                "voc_holdout_stop_support": torch.tensor(
                    float(holdout_stop_count), device=total_loss.device
                ),
                "voc_effective_think_cost": torch.tensor(
                    effective_think_cost, device=total_loss.device
                ),
            })
            if voc_soft_gate_result is not None:
                soft = voc_soft_gate_result
                soft_mean = lambda value: voc_mean(value, soft.valid)
                soft_rmse = lambda value: voc_rmse(value, soft.valid)
                soft_positive = soft.valid & (soft.delta_q > 1e-6)
                soft_negative = soft.valid & (soft.delta_q < -1e-6)
                soft_low = soft.valid & (
                    soft.student_continue_probability.detach() < 0.1
                )
                soft_high = soft.valid & (
                    soft.student_continue_probability.detach() > 0.9
                )
                positive_count = max(int(soft_positive.sum().item()), 1)
                negative_count = max(int(soft_negative.sum().item()), 1)
                soft_valid_count = max(int(soft.valid.sum().item()), 1)
                losses.update({
                    "voc_gate_bce_loss": soft.loss.detach(),
                    "voc_gate_bce_unweighted": soft_mean(soft.bce),
                    "voc_gate_student_continue_probability": soft_mean(
                        soft.student_continue_probability
                    ),
                    "voc_gate_teacher_continue_probability": soft_mean(
                        soft.teacher_continue_probability
                    ),
                    "voc_gate_teacher_confidence": soft_mean(
                        soft.confidence
                    ),
                    "voc_gate_teacher_student_gap": soft_mean(
                        soft.teacher_continue_probability
                        - soft.student_continue_probability
                    ),
                    "voc_gate_teacher_student_gap_abs": soft_mean(
                        (
                            soft.teacher_continue_probability
                            - soft.student_continue_probability
                        ).abs()
                    ),
                    "voc_gate_student_saturation_jacobian": soft_mean(
                        soft.student_continue_probability
                        * (1.0 - soft.student_continue_probability)
                    ),
                    "voc_gate_directed_logit_gradient_mean": soft_mean(
                        soft.directed_logit_gradient
                    ),
                    "voc_gate_directed_logit_gradient_abs": soft_mean(
                        soft.directed_logit_gradient.abs()
                    ),
                    "voc_gate_directed_logit_gradient_rms": soft_rmse(
                        soft.directed_logit_gradient
                    ),
                    "voc_gate_wrong_continue_saturation_count": torch.tensor(
                        float(soft.wrong_continue_saturation.sum().item()),
                        device=total_loss.device,
                    ),
                    "voc_gate_wrong_continue_saturation_rate": torch.tensor(
                        float(soft.wrong_continue_saturation.sum().item())
                        / positive_count,
                        device=total_loss.device,
                    ),
                    "voc_gate_wrong_stop_saturation_count": torch.tensor(
                        float(soft.wrong_stop_saturation.sum().item()),
                        device=total_loss.device,
                    ),
                    "voc_gate_wrong_stop_saturation_rate": torch.tensor(
                        float(soft.wrong_stop_saturation.sum().item())
                        / negative_count,
                        device=total_loss.device,
                    ),
                    "voc_gate_low_saturation_count": torch.tensor(
                        float(soft_low.sum().item()), device=total_loss.device
                    ),
                    "voc_gate_low_saturation_rate": torch.tensor(
                        float(soft_low.sum().item()) / soft_valid_count,
                        device=total_loss.device,
                    ),
                    "voc_gate_high_saturation_count": torch.tensor(
                        float(soft_high.sum().item()), device=total_loss.device
                    ),
                    "voc_gate_high_saturation_rate": torch.tensor(
                        float(soft_high.sum().item()) / soft_valid_count,
                        device=total_loss.device,
                    ),
                    "voc_gate_positive_teacher_count": torch.tensor(
                        float(soft_positive.sum().item()),
                        device=total_loss.device,
                    ),
                    "voc_gate_negative_teacher_count": torch.tensor(
                        float(soft_negative.sum().item()),
                        device=total_loss.device,
                    ),
                    "voc_gate_q_temperature": torch.tensor(
                        self.voc_gate_q_temperature,
                        device=total_loss.device,
                    ),
                    "voc_gate_policy_temperature": torch.tensor(
                        float(self.flags.voc_gate_temperature),
                        device=total_loss.device,
                    ),
                    "voc_gate_confidence_weighted": torch.tensor(
                        float(self.voc_gate_confidence_weighted),
                        device=total_loss.device,
                    ),
                })
                if self.voc_gate_param_align:
                    zero = torch.zeros(
                        (), device=total_loss.device,
                        dtype=total_loss.dtype,
                    )
                    if train_count > 0 and voc_gate_param_alignment is None:
                        raise RuntimeError(
                            "VoC gate parameter alignment was not computed "
                            "for a supported training batch"
                        )
                    alignment = voc_gate_param_alignment

                    def alignment_metric(value):
                        if alignment is None:
                            return zero
                        return value.detach().to(
                            device=total_loss.device,
                            dtype=total_loss.dtype,
                        )

                    raw_alignment_loss = alignment_metric(
                        None if alignment is None else alignment.loss
                    )
                    losses.update({
                        "voc_gate_param_align_enabled": torch.ones_like(zero),
                        "voc_gate_param_align_coef": torch.tensor(
                            self.voc_gate_param_align_coef,
                            device=total_loss.device,
                            dtype=total_loss.dtype,
                        ),
                        "voc_gate_param_align_applied": torch.tensor(
                            float(alignment is not None),
                            device=total_loss.device,
                            dtype=total_loss.dtype,
                        ),
                        "voc_gate_param_align_train_support": torch.tensor(
                            float(train_count),
                            device=total_loss.device,
                            dtype=total_loss.dtype,
                        ),
                        "voc_gate_param_align_loss": raw_alignment_loss,
                        "voc_gate_param_align_scaled_loss": (
                            self.voc_gate_param_align_coef
                            * raw_alignment_loss
                        ),
                        "voc_gate_objective_loss": voc_gate_loss.detach(),
                        "voc_gate_param_gate_weight_norm": alignment_metric(
                            None
                            if alignment is None
                            else alignment.gate_weight_norm
                        ),
                        "voc_gate_param_target_weight_norm": alignment_metric(
                            None
                            if alignment is None
                            else alignment.target_weight_norm
                        ),
                        "voc_gate_param_weight_error_norm": alignment_metric(
                            None
                            if alignment is None
                            else alignment.weight_error_norm
                        ),
                        "voc_gate_param_gate_bias": alignment_metric(
                            None if alignment is None else alignment.gate_bias
                        ),
                        "voc_gate_param_target_bias": alignment_metric(
                            None
                            if alignment is None
                            else alignment.target_bias.reshape(())
                        ),
                        "voc_gate_param_bias_error_abs": alignment_metric(
                            None
                            if alignment is None
                            else alignment.bias_error_abs
                        ),
                        "voc_gate_param_gate_norm": alignment_metric(
                            None
                            if alignment is None
                            else alignment.gate_parameter_norm
                        ),
                        "voc_gate_param_target_norm": alignment_metric(
                            None
                            if alignment is None
                            else alignment.target_parameter_norm
                        ),
                        "voc_gate_param_error_norm": alignment_metric(
                            None
                            if alignment is None
                            else alignment.parameter_error_norm
                        ),
                        "voc_gate_param_relative_error": alignment_metric(
                            None
                            if alignment is None
                            else alignment.relative_parameter_error
                        ),
                        "voc_gate_param_relative_error_defined": (
                            alignment_metric(
                                None
                                if alignment is None
                                else alignment.relative_error_defined
                            )
                        ),
                        "voc_gate_param_cosine": alignment_metric(
                            None if alignment is None else alignment.cosine
                        ),
                        "voc_gate_param_cosine_defined": alignment_metric(
                            None
                            if alignment is None
                            else alignment.cosine_defined
                        ),
                    })
                elif self.voc_gate_exact_projection:
                    losses.update({
                        "voc_gate_exact_projection_enabled": torch.ones(
                            (),
                            device=total_loss.device,
                            dtype=total_loss.dtype,
                        ),
                        "voc_gate_objective_loss": soft.loss.detach(),
                        "voc_gate_projection_batch_start_error_norm": (
                            torch.zeros(
                                (),
                                device=total_loss.device,
                                dtype=total_loss.dtype,
                            )
                        ),
                    })
                if voc_gate_behavior_entropy is not None:
                    losses["voc_gate_behavior_entropy"] = voc_mean(
                        voc_gate_behavior_entropy, soft.valid
                    )
            observability_kwargs = dict(
                continue_probability=(
                    voc_gate_soft_continue_probability
                    if self.voc_gate_epsilon_greedy_execution
                    else voc_result.continue_probability
                ),
                behavior_continue_probability=(
                    voc_result.behavior_continue_probability
                    if self.voc_gate_exact_projection
                    else None
                ),
                gate_action=voc_result.gate_action,
                control_valid=voc_result.valid,
                search_steps=train_actor_out.search_steps,
                control_action=control_action,
                predecision_last_control=voc_predecision_last_control,
                q_temperature=self.voc_gate_q_temperature,
            )
            # Preserve every historical online-Q key and record the behavior
            # goals for the actual EMA gate source under an explicit prefix.
            losses.update(dynamic_voc_observability_metrics(
                delta_q=voc_result.delta_q,
                **observability_kwargs,
            ))
            ema_observability = dynamic_voc_observability_metrics(
                delta_q=voc_gate_delta_q,
                **observability_kwargs,
            )
            losses.update({
                "voc_gate_" + key[len("voc_"):]: value
                for key, value in ema_observability.items()
            })

        # process entropy loss
        if not self.autotune:
            entropy_cost = self.flags.entropy_cost
            im_entropy_cost = self.flags.im_entropy_cost
        else:            
            entropy_cost = self.actor_net.log_entropy_cost.exp().item()
            im_entropy_cost = self.actor_net.log_im_entropy_cost.exp().item()

        f_entropy_loss = new_actor_out.entropy_loss
        if self.dynamic_search:
            entropy_loss = torch.sum(f_entropy_loss * real_policy_mask.float())
            real_policy_n = real_policy_mask.sum()
            policy_entropy = -entropy_loss / real_policy_n.clamp_min(1)
            losses["entropy_loss"] = entropy_loss
            total_loss += entropy_cost * entropy_loss / self.actor_net.dim_actions

            im_entropy_loss = torch.sum(
                f_entropy_loss * search_policy_mask.float()
            )
            search_policy_n = search_policy_mask.sum()
            im_policy_entropy = -im_entropy_loss / search_policy_n.clamp_min(1)
            total_loss += im_entropy_cost * im_entropy_loss
            losses["im_entropy_loss"] = im_entropy_loss / self.actor_net.dim_actions

            if self.autotune:
                autotune_loss = torch.zeros(
                    (), device=f_entropy_loss.device, dtype=f_entropy_loss.dtype
                )
                if real_policy_n.item() > 0:
                    autotune_loss = autotune_loss + (
                        -self.actor_net.log_entropy_cost.exp()
                        * (self.tar_entropy - policy_entropy.detach())
                    )[0]
                if search_policy_n.item() > 0:
                    autotune_loss = autotune_loss + (
                        -self.actor_net.log_im_entropy_cost.exp()
                        * (self.tar_im_entropy - im_policy_entropy.detach())
                    )[0]
                losses["autotune_loss"] = autotune_loss
                total_loss += autotune_loss
        else:
            entropy_loss = f_entropy_loss * last_step_real.float()
            policy_entropy = -entropy_loss.sum() / last_step_real.sum()
            entropy_loss = torch.sum(entropy_loss)
            losses["entropy_loss"] = entropy_loss
            total_loss += entropy_cost * entropy_loss / self.actor_net.dim_actions

            if not self.disable_thinker:
                im_entropy_loss = f_entropy_loss * (~last_step_real).float()
                im_policy_entropy = -im_entropy_loss.sum() / (~last_step_real).sum()
                im_entropy_loss = torch.sum(im_entropy_loss)
                total_loss += im_entropy_cost * im_entropy_loss
                losses["im_entropy_loss"] = im_entropy_loss / self.actor_net.dim_actions

            if self.autotune:
                autotune_loss = -self.actor_net.log_entropy_cost.exp() * (self.tar_entropy - policy_entropy.detach())
                if not self.disable_thinker:
                    autotune_loss += -self.actor_net.log_im_entropy_cost.exp() * (self.tar_im_entropy - im_policy_entropy.detach())
                autotune_loss = autotune_loss[0]
                losses["autotune_loss"] = autotune_loss
                total_loss += autotune_loss

        if self.dynamic_search:
            reg_loss = torch.sum(
                new_actor_out.reg_loss * policy_valid.float()
            )
        else:
            reg_loss = torch.sum(new_actor_out.reg_loss)
        losses["reg_loss"] = reg_loss
        total_loss += self.flags.reg_cost * reg_loss

        if self.dynamic_search:
            total_loss = self._add_online_action_prior(
                total_loss, losses, new_actor_out, real_policy_mask
            )
            if self.imitation_enabled:
                # Keep the CSV/W&B schema present even on a deliberately
                # skipped supervised-frequency row.  Count zero distinguishes
                # these finite placeholders from a measured batch value.
                empty_behavior_metrics = empty_behavioral_action_metrics(
                    num_actions=int(self.actor_net.num_actions),
                    noop_action_index=self.imitation_noop_action_index,
                )
                for metric_name, metric_value in empty_behavior_metrics.items():
                    losses[f"icopro_{metric_name}"] = torch.tensor(
                        metric_value, device=total_loss.device
                    )
                losses["icopro_accuracy"] = losses[
                    "icopro_behavioral_argmax_accuracy"
                ]
                losses["icopro_sampled_accuracy"] = losses[
                    "icopro_behavioral_sampled_accuracy"
                ]
                losses["icopro_count"] = losses[
                    "icopro_behavioral_support_count"
                ]
            imitation_result = self._maybe_compute_imitation()
            if imitation_result is not None:
                # The online actor objective is represented as a sum.  Scale
                # the mean behavioral objective by the number of online real
                # policy rows, not by augmented SEARCH/WAIT rows.
                scaled_imitation_loss = scale_imitation_for_online_rows(
                    imitation_result.loss, real_policy_mask
                )
                total_loss = total_loss + scaled_imitation_loss
                metrics = imitation_result.detached_metrics()
                losses.update({
                    "icopro_loss": imitation_result.loss,
                    "icopro_scaled_loss": scaled_imitation_loss,
                    "icopro_nll": imitation_result.nll_sum
                    / max(imitation_result.count, 1),
                    "icopro_normalized_ce": imitation_result.normalized_ce,
                    "icopro_margin_loss": imitation_result.margin_loss,
                    "icopro_pvp_loss": imitation_result.pvp_loss,
                    "icopro_accuracy": torch.tensor(
                        metrics["accuracy"], device=total_loss.device
                    ),
                    "icopro_sampled_accuracy": torch.tensor(
                        metrics["sampled_accuracy"], device=total_loss.device
                    ),
                    "icopro_root_carried_rate": torch.tensor(
                        metrics["root_carried_rate"], device=total_loss.device
                    ),
                    "icopro_count": torch.tensor(
                        float(imitation_result.count), device=total_loss.device
                    ),
                    "icopro_update_count": torch.tensor(
                        float(self.imitation_update_count + 1),
                        device=total_loss.device,
                    ),
                })
                # Test doubles and old evaluator shims may only provide the
                # historical mean names.  Derive the count-backed accuracy
                # aliases while real DynamicImitationResult objects provide
                # exact integer numerators below.
                metrics.setdefault(
                    "behavioral_argmax_accuracy", metrics["accuracy"]
                )
                metrics.setdefault(
                    "behavioral_sampled_accuracy",
                    metrics["sampled_accuracy"],
                )
                metrics.setdefault(
                    "behavioral_support_count",
                    float(imitation_result.count),
                )
                metrics.setdefault(
                    "behavioral_argmax_correct_count",
                    metrics["behavioral_argmax_accuracy"]
                    * float(imitation_result.count),
                )
                metrics.setdefault(
                    "behavioral_sampled_correct_count",
                    metrics["behavioral_sampled_accuracy"]
                    * float(imitation_result.count),
                )
                for metric_name in (
                    "behavioral_argmax_accuracy",
                    "behavioral_sampled_accuracy",
                    "behavioral_argmax_correct_count",
                    "behavioral_sampled_correct_count",
                    "behavioral_support_count",
                    "noop_action_index",
                    "noop_supported",
                    "noop_support_count",
                    "target_noop_count",
                    "target_noop_frequency",
                    "argmax_noop_count",
                    "argmax_noop_frequency",
                    "sampled_noop_count",
                    "sampled_noop_frequency",
                ):
                    if metric_name in metrics:
                        losses[f"icopro_{metric_name}"] = torch.tensor(
                            metrics[metric_name], device=total_loss.device
                        )
                for metric_name in (
                    "nll_max",
                    "nll_p99",
                    "target_vs_best_other_logit_gap_max",
                    "target_vs_best_other_logit_gap_p99",
                    "scored_logits_absmax",
                    "scored_logits_rms",
                ):
                    losses[f"icopro_{metric_name}"] = torch.tensor(
                        metrics[metric_name], device=total_loss.device
                    )

        if self.ppo_enable:
            if self.actor_net.discrete_action:
                tar_pri_log_prob = F.log_softmax(base_pri_logits, dim=-1)
                pri_log_prob = F.log_softmax(new_actor_out.pri_param, dim=-1)
                pri_kl_loss = F.kl_div(pri_log_prob, tar_pri_log_prob, reduction="none", log_target=True)
                pri_kl_loss = torch.sum(pri_kl_loss, dim=-1)
            else:
                pri_kl_loss = guassian_kl_div(
                    base_pri_mean, 
                    base_pri_log_var,
                    new_actor_out.pri_param[:, :, :, 0],
                    new_actor_out.pri_param[:, :, :, 1]
                )            
            if self.dynamic_search:
                # Exact hierarchical-policy KL:
                #   KL(control) + P_old(non-STOP) KL(imaginary)
                # on SEARCH rows, and KL(real) on NEED_REAL_ACTION rows.
                # Unlike the sampled primary_valid mask used by PPO's action
                # ratio, this expectation also regularizes the conditional
                # imaginary branch on rows whose sampled control was STOP.
                base_control_probs = F.softmax(base_control_logits, dim=-1)
                search_primary_weight = (
                    search_policy_mask.float()
                    * (1.0 - base_control_probs[..., util.STOP])
                )
                pri_weight = (
                    real_policy_mask.float() + search_primary_weight
                )
                expanded_pri_weight = pri_weight
                while expanded_pri_weight.ndim < pri_kl_loss.ndim:
                    expanded_pri_weight = expanded_pri_weight.unsqueeze(-1)
                pri_kl_loss = torch.sum(
                    pri_kl_loss * expanded_pri_weight
                )
                kl_loss = pri_kl_loss

                control_logits = getattr(
                    new_actor_out, "search_control_logits", None
                )
                if control_logits is None:
                    control_logits = new_actor_out.reset_logits
                if self.voc_dedicated_gate:
                    control_logits = (
                        detach_dynamic_voc_gate_from_joint_logits(
                            control_logits
                        )
                    )
                tar_control_log_prob = F.log_softmax(
                    base_control_logits, dim=-1
                )
                control_log_prob = F.log_softmax(control_logits, dim=-1)
                control_kl_loss = F.kl_div(
                    control_log_prob,
                    tar_control_log_prob,
                    reduction="none",
                    log_target=True,
                ).sum(dim=-1)
                control_kl_loss = torch.sum(
                    control_kl_loss * control_valid.float()
                )
                kl_loss += control_kl_loss
                kl_denominator = (
                    pri_weight.sum() * self.actor_net.dim_actions
                    + control_valid.sum()
                ).clamp_min(1)
            else:
                pri_kl_loss = torch.sum(pri_kl_loss)
                kl_loss = pri_kl_loss

                if not self.disable_thinker:
                    tar_reset_log_prob = F.log_softmax(base_reset_logits, dim=-1)
                    reset_log_prob = F.log_softmax(new_actor_out.reset_logits, dim=-1)
                    reset_kl_loss = F.kl_div(reset_log_prob, tar_reset_log_prob, reduction="sum", log_target=True)
                    kl_loss += reset_kl_loss
                kl_denominator = T * B

            if self.flags.ppo_kl_coef > 0.:
                total_loss += self.flags.ppo_kl_coef * self.actor_net.kl_beta * kl_loss         
                avg_kl_loss = kl_loss / kl_denominator
                if last_iter:                
                    if avg_kl_loss < self.flags.ppo_kl_targ / 1.5:
                        self.actor_net.kl_beta /= 2
                    elif avg_kl_loss > self.flags.ppo_kl_targ * 1.5:
                        self.actor_net.kl_beta *= 2
                if self.flags.ppo_early_stop:
                    if avg_kl_loss > self.flags.ppo_kl_targ:
                        self.ppo_early_stop = True
                self.actor_net.kl_beta = torch.clamp(self.actor_net.kl_beta, 1e-6, 1e3)
            self.kl_losses.append(kl_loss.item())            
            losses["kl_loss"] = np.mean(self.kl_losses)
        losses["total_loss"] = total_loss

        return losses, train_actor_out

    def compute_stat(self, train_actor_out, losses, total_norm, actor_id):
        """Update step, real_step and tot_eps; return training stat for printing"""
        stats = {}
        T, B, *_ = train_actor_out.episode_return.shape
        if self.dynamic_search:
            last_step_real = train_actor_out.real_transition.bool()
            stage_end = train_actor_out.stage_end.bool()
            next_step_real = stage_end
        else:
            last_step_real = ((train_actor_out.step_status == 0)
                              | (train_actor_out.step_status == 3))
            next_step_real = ((train_actor_out.step_status == 2)
                              | (train_actor_out.step_status == 3))
        
        real_done = train_actor_out.real_done |  train_actor_out.truncated_done     

        # extract episode_returns
        episode_returns, done_ids = self.ret_buffers["re"].insert(
            train_actor_out.episode_return, ind=0, actor_id=actor_id, done=real_done
        )
        episode_lens, _ = self.ret_buffers["len"].insert(
            train_actor_out.episode_step.unsqueeze(-1), ind=0, actor_id=actor_id, done=real_done
        )

        stats = {"rmean_episode_return": self.ret_buffers["re"].get_mean(),
                 "max_episode_return": self.ret_buffers["re"].get_max(),
                 "rmean_len": self.ret_buffers["len"].get_mean(),}

        for prefix in self.rewards_ls[1:]:
            if prefix in ["im", "think"]:
                done = next_step_real
            elif prefix == "cur":
                done = real_done
            
            if prefix in self.rewards_ls:            
                n = self.rewards_ls.index(prefix)
                self.ret_buffers[prefix].insert(
                    train_actor_out.episode_return, ind=n, actor_id=actor_id, done=done,
                )
                r = self.ret_buffers[prefix].get_mean()
                stats["rmean_%s_episode_return" % prefix] = r

        if self.dynamic_search:
            control_valid = train_actor_out.control_valid.bool()
            controls = train_actor_out.search_control[control_valid]
            control_n = max(int(controls.numel()), 1)
            control_logits = getattr(
                train_actor_out, "search_control_logits", None
            )
            if control_logits is None:
                control_logits = train_actor_out.reset_logits
            with torch.no_grad():
                control_entropy = compute_dynamic_control_entropy(
                    control_logits
                )

                def valid_control_mean(value):
                    if not torch.any(control_valid):
                        return 0.0
                    return value[control_valid].float().mean().item()

                mean_gate_entropy = valid_control_mean(control_entropy.gate)
                mean_bout_entropy = valid_control_mean(control_entropy.bout)
                mean_control_entropy = valid_control_mean(
                    control_entropy.gate
                    + control_entropy.continue_prob * control_entropy.bout
                )
                if self.actor_net.discrete_action:
                    primary_log_probs = F.log_softmax(
                        train_actor_out.pri_param, dim=-1
                    )
                    primary_probs = primary_log_probs.exp()
                    primary_entropy = -(
                        primary_probs * primary_log_probs
                    ).sum(dim=-1).sum(dim=-1)
                    mean_primary_entropy = valid_control_mean(primary_entropy)
                    mean_search_entropy = valid_control_mean(
                        control_entropy.gate
                        + control_entropy.continue_prob
                        * (control_entropy.bout + primary_entropy)
                    )
                else:
                    primary_log_var = train_actor_out.pri_param[..., 1]
                    primary_entropy = 0.5 * (
                        1.0 + np.log(2.0 * np.pi) + primary_log_var
                    ).sum(dim=-1)
                    mean_primary_entropy = valid_control_mean(primary_entropy)
                    mean_search_entropy = valid_control_mean(
                        control_entropy.gate
                        + control_entropy.continue_prob
                        * (control_entropy.bout + primary_entropy)
                    )
            stats.update({
                "search/proceed_ratio": (
                    (controls == util.PROCEED).sum().item() / control_n
                ),
                "search/reset_ratio": (
                    (controls == util.RESET).sum().item() / control_n
                ),
                "search/stop_ratio": (
                    (controls == util.STOP).sum().item() / control_n
                ),
                "search/wait_fraction": (
                    (train_actor_out.policy_type == util.POLICY_NONE)
                    .float().mean().item()
                ),
                "search/active_batch_fraction": (
                    train_actor_out.policy_valid.float().mean().item()
                ),
                "search/active_batch_size": (
                    train_actor_out.policy_valid.float().sum(dim=1).mean().item()
                ),
                "search/mean_stop_probability": valid_control_mean(
                    control_entropy.stop_prob
                ),
                "search/mean_continue_probability": valid_control_mean(
                    control_entropy.continue_prob
                ),
                "search/mean_gate_entropy": mean_gate_entropy,
                "search/mean_bout_entropy": mean_bout_entropy,
                "search/mean_primary_entropy": mean_primary_entropy,
                "search/mean_control_entropy": mean_control_entropy,
                "search/mean_policy_entropy": mean_search_entropy,
                "search/normalized_gate_entropy": (
                    mean_gate_entropy / np.log(2.0)
                ),
                "search/normalized_bout_entropy": (
                    mean_bout_entropy / np.log(2.0)
                ),
            })
            stats.update(dynamic_actor_observability_stats(
                train_actor_out,
                discrete_action=self.actor_net.discrete_action,
            ))
            # Versioned schema-6..13 runs retain their frozen legacy CSV/W&B
            # keysets.  Unversioned Dynamic runs expose the actions that
            # actually crossed the environment barrier; SEARCH proposals and
            # WAIT placeholders are deliberately excluded.
            if (
                getattr(self, "voc_gate_policy_schema_version", None) is None
                and self.actor_net.discrete_action
            ):
                stats.update(environment_noop_observability_stats(
                    train_actor_out.last_pri,
                    train_actor_out.real_transition,
                    num_actions=int(self.actor_net.num_actions),
                    noop_action_index=self.imitation_noop_action_index,
                ))
            stats.update(util.get_search_budget_stats(
                train_actor_out.search_steps, stage_end
            ))
            stats.update(util.get_search_depth_stop_stats(
                train_actor_out.search_steps,
                train_actor_out.search_control,
                control_valid,
                control_entropy.stop_prob,
            ))
            stage_n = max(int(stage_end.sum().item()), 1)
            forced_stage_end = stage_end & train_actor_out.forced_stop.bool()
            learned_stage_end = stage_end & ~train_actor_out.forced_stop.bool()
            forced_stage_n = int(forced_stage_end.sum().item())
            learned_stage_n = int(learned_stage_end.sum().item())
            stats["search/forced_stop_rate"] = forced_stage_n / stage_n
            stats["search/learned_stop_rate"] = learned_stage_n / stage_n
            stats["search/forced_stop_count"] = forced_stage_n
            stats["search/learned_stop_count"] = learned_stage_n
            stats["search/stage_end_count"] = int(stage_end.sum().item())
            stats["search/real_transition_fraction"] = (
                last_step_real.float().mean().item()
            )
        elif not self.disable_thinker:
            max_rollout_depth = (
                (train_actor_out.max_rollout_depth[last_step_real & ~next_step_real])
                .detach()
                .cpu()
                .numpy()
            )
            max_rollout_depth = (
                np.average(max_rollout_depth) if len(max_rollout_depth) > 0 else 0.0
            )
            stats["max_rollout_depth"] = max_rollout_depth

        mean_abs_v = torch.mean(torch.abs(train_actor_out.baseline)).item()

        stats.update({
            "step": self.step,
            "real_step": self.real_step,
            "tot_eps": self.tot_eps,
            "episode_returns": episode_returns,
            "episode_lens": episode_lens,
            "done_ids": done_ids,
            "actor/total_norm": total_norm,
            "actor/mean_abs_v": mean_abs_v,
            "actor/learning_rate": self.optimizer.param_groups[0]["lr"],
            "actor/schedule_progress": util.schedule_progress(
                self.flags, self.real_step
            ),
            "actor/optimizer_stepped": int(
                self._last_actor_gradient_step.optimizer_stepped
            ),
            "actor/amp_skip_count": self.actor_amp_skip_count,
            "actor/amp_consecutive_skips": self.actor_amp_consecutive_skips,
            "actor/nonfinite_gradient_parameter_count": len(
                self._last_actor_gradient_step.nonfinite_gradient_names
            ),
        })
        if self.dynamic_voc_mode != "off":
            stats.update({
                "voc/mode_shadow": int(self.dynamic_voc_mode == "shadow"),
                "voc/mode_control": int(self.dynamic_voc_mode == "control"),
                "voc/update_count": self.voc_update_count,
                "voc/continue_count": self.voc_continue_count,
                "voc/stop_count": self.voc_stop_count,
                "voc/learning_rate": self.voc_optimizer.param_groups[0]["lr"],
                "voc/amp_skip_count": self.voc_amp_skip_count,
                "voc/amp_consecutive_skips": self.voc_amp_consecutive_skips,
                "voc/nonfinite_gradient_parameter_count": len(
                    self._last_voc_gradient_step.nonfinite_gradient_names
                ),
            })
            if self._last_voc_gradient_step.amp_scale_before is not None:
                stats["voc/amp_scale_before"] = (
                    self._last_voc_gradient_step.amp_scale_before
                )
                stats["voc/amp_scale_after"] = (
                    self._last_voc_gradient_step.amp_scale_after
                )
        if self.voc_actor_policy_version_barrier:
            stats.update({
                "voc_actor_policy_version": self.voc_actor_policy_version,
                "voc_actor_policy_publication_count": (
                    self.voc_actor_policy_publication_count
                ),
                "voc_actor_policy_terminal": int(
                    self.voc_actor_policy_terminal
                ),
                "voc_actor_policy_version_mismatch_count": (
                    self.voc_actor_policy_version_mismatch_count
                ),
                "voc_actor_policy_malformed_bundle_count": (
                    self.voc_actor_policy_malformed_bundle_count
                ),
                "voc_actor_policy_barrier_timeout_count": (
                    self.voc_actor_policy_barrier_timeout_count
                ),
                "voc_actor_policy_terminal_ack_count": (
                    self.voc_actor_policy_terminal_ack_count
                ),
                "voc_actor_policy_expected_ack_count": (
                    self.voc_actor_policy_expected_ack_count
                ),
                "voc_actor_policy_state_sha256": (
                    self.voc_actor_policy_state_sha256
                ),
                "voc_actor_policy_publication_history_sha256": (
                    self.voc_actor_policy_publication_history_sha256
                ),
                "voc_actor_policy_barrier_runtime": int(
                    self.voc_actor_policy_barrier_runtime
                ),
                "voc_gate_soft_training_epsilon": float(
                    getattr(self.flags, "voc_train_epsilon", 0.02)
                ),
                "voc_gate_execution_epsilon": self.voc_gate_execution_epsilon,
                "voc_actor_policy_bundle_schema_version": (
                    util.VOC_ACTOR_POLICY_BUNDLE_SCHEMA_VERSION
                ),
                "voc_actor_policy_barrier_timeout_s": (
                    self.voc_actor_policy_barrier_timeout_s
                ),
                "actor/amp_init_scale": self.actor_amp_init_scale,
            })
        if self.voc_dedicated_gate:
            stats.update({
                "voc_gate/learning_rate": (
                    self.voc_gate_optimizer.param_groups[0]["lr"]
                ),
                "voc_gate/update_count": self.voc_gate_update_count,
                "voc_gate/amp_skip_count": self.voc_gate_amp_skip_count,
                "voc_gate/amp_consecutive_skips": (
                    self.voc_gate_amp_consecutive_skips
                ),
                "voc_gate/nonfinite_gradient_parameter_count": len(
                    self._last_voc_gate_gradient_step.nonfinite_gradient_names
                ),
                "voc_gate/optimizer_stepped": int(
                    self._last_voc_gate_gradient_step.optimizer_stepped
                ),
            })
            if self.voc_gate_exact_projection:
                stats.update({
                    "voc_gate/exact_projection_enabled": 1,
                    "voc_gate/exact_projection_applied": int(
                        self._last_voc_gate_exact_projection_applied
                    ),
                })
            if self._last_voc_gate_gradient_step.amp_scale_before is not None:
                stats["voc_gate/amp_scale_before"] = (
                    self._last_voc_gate_gradient_step.amp_scale_before
                )
                stats["voc_gate/amp_scale_after"] = (
                    self._last_voc_gate_gradient_step.amp_scale_after
                )
        if self._last_actor_gradient_step.amp_scale_before is not None:
            stats["actor/amp_scale_before"] = (
                self._last_actor_gradient_step.amp_scale_before
            )
            stats["actor/amp_scale_after"] = (
                self._last_actor_gradient_step.amp_scale_after
            )

        if losses is not None:
            for k, v in losses.items():
                if v is not None:
                    stats["actor/"+k] = v.item()

        if self.flags.return_norm_type in [0, 1]:
            n = self.rewards_ls.index("re")
            stats["actor/norm_diff"] = (
                self.norm_stats[n][1] - self.norm_stats[n][0]
                ).item()            
            stats["norm_rmean_episode_return"] = (stats["rmean_episode_return"] / self.norm_stats[n][2]).item()
            if "im" in self.rewards_ls:
                n = self.rewards_ls.index("im")
                stats["actor/im_norm_diff"] = (
                    self.norm_stats[n][1] - self.norm_stats[n][0]
                ).item()
                stats["norm_rmean_im_episode_return"] = (stats["rmean_im_episode_return"] / self.norm_stats[n][2]).item()
            if "cur" in self.rewards_ls:
                n = self.rewards_ls.index("cur")
                stats["actor/cur_norm_diff"] = (
                    self.norm_stats[n][1] - self.norm_stats[n][0]
                ).item()
                stats["norm_rmean_cur_episode_return"] = (stats["rmean_cur_episode_return"] / self.norm_stats[n][2]).item()
            if "think" in self.rewards_ls:
                n = self.rewards_ls.index("think")
                stats["actor/think_norm_diff"] = (
                    self.norm_stats[n][1] - self.norm_stats[n][0]
                ).item()
                stats["norm_rmean_think_episode_return"] = (
                    stats["rmean_think_episode_return"] / self.norm_stats[n][2]
                ).item()
        return stats

    def save_checkpoint(self, force=False):
        if self.voc_gate_exact_projection:
            self._assert_voc_gate_exact_projection_invariant()
        if (
            self.voc_actor_policy_barrier_runtime
            and self._voc_actor_policy_transaction_open
        ):
            self._voc_actor_policy_checkpoint_pending = True
            self._voc_actor_policy_checkpoint_force |= bool(force)
            return
        actor_state_for_checkpoint = self.actor_net.state_dict()
        if self.voc_actor_policy_version_barrier:
            actor_state_for_checkpoint = util.clone_actor_policy_state(
                actor_state_for_checkpoint
            )
            if self.voc_actor_policy_barrier_runtime:
                if self._voc_actor_policy_bundle is None:
                    raise RuntimeError(
                        "schema-6 checkpoint lacks a published policy bundle"
                    )
                published_state = self._voc_actor_policy_bundle[
                    "actor_state_dict"
                ]
                same_live_state = (
                    set(published_state) == set(actor_state_for_checkpoint)
                    and all(
                        torch.equal(
                            published_state[key],
                            actor_state_for_checkpoint[key],
                        )
                        for key in published_state
                    )
                )
                if not same_live_state:
                    if self.voc_actor_policy_terminal:
                        raise RuntimeError(
                            "terminal versioned bundle disagrees with live actor"
                        )
                    self._voc_actor_policy_checkpoint_pending = True
                    self._voc_actor_policy_checkpoint_force |= bool(force)
                    return
                self._voc_actor_policy_bundle = (
                    util.validate_actor_policy_bundle(
                        self._voc_actor_policy_bundle,
                        expected_epoch=self.voc_actor_policy_version,
                        expected_terminal=self.voc_actor_policy_terminal,
                        expected_actor_state=actor_state_for_checkpoint,
                        require_equal_state=True,
                        expected_gate_schema=(
                            self.voc_gate_policy_schema_version
                        ),
                        label="versioned checkpoint policy bundle",
                    )
                )
                self.voc_actor_policy_state_sha256 = (
                    util.actor_policy_state_sha256(actor_state_for_checkpoint)
                )
        gate_schema_version = None
        if self.voc_dedicated_gate:
            if self.voc_actor_policy_version_barrier:
                gate_schema_version = self.voc_gate_policy_schema_version
            elif self.voc_gate_epsilon_greedy_execution:
                gate_schema_version = (
                    util.VOC_GATE_POLICY_EPSILON_GREEDY_EXECUTION_SCHEMA_VERSION
                )
            elif self.voc_gate_exact_projection:
                gate_schema_version = (
                    util.VOC_GATE_POLICY_EXACT_PROJECTION_SCHEMA_VERSION
                )
            else:
                gate_schema_version = util.VOC_GATE_POLICY_SCHEMA_VERSION
        self._logger.info("Saving actor checkpoint to %s" % self.ckp_path)
        d = {
                "step": self.step,
                "real_step": self.real_step,
                "tot_eps": self.tot_eps,
                "ret_buffers": self.ret_buffers,
                "norm_stats": self.norm_stats,
                "crnorm": self.crnorm, 
                "actor_net_optimizer_state_dict": self.optimizer.state_dict(),
                "actor_net_scheduler_state_dict": self.scheduler.state_dict(),
                "actor_net_state_dict": actor_state_for_checkpoint,
                "actor_arch_version": 2 if self.dynamic_search else 1,
                "dynamic_search": self.dynamic_search,
                "dynamic_factorized_control": self.dynamic_factorized_control,
                "dynamic_voc_mode": self.dynamic_voc_mode,
                "voc_ema_gate_target": self.voc_ema_gate_target,
                "voc_gate_target_tau": self.voc_gate_target_tau,
                "voc_gate_policy_schema_version": gate_schema_version,
                "voc_ema_gate_schema_version": (
                    util.VOC_EMA_GATE_SCHEMA_VERSION
                    if self.dynamic_voc_mode != "off" else None
                ),
                "voc_ema_gate_head_state_dict": (
                    self._voc_ema_gate_state_dict()
                ),
                "voc_ema_gate_update_count": self.voc_ema_gate_update_count,
                "voc_ema_gate_parent_update_count": (
                    self.voc_ema_gate_parent_update_count
                ),
                "voc_update_count": self.voc_update_count,
                "voc_continue_count": self.voc_continue_count,
                "voc_stop_count": self.voc_stop_count,
                "voc_holdout_count": self.voc_holdout_count,
                "voc_holdout_split_version": util.VOC_HOLDOUT_SPLIT_VERSION,
                "voc_holdout_actor_modulus": util.VOC_HOLDOUT_ACTOR_MODULUS,
                "voc_holdout_actor_streams": (
                    int(self.flags.self_play_n) * int(self.flags.env_n)
                ),
                "voc_holdout_continue_count": self.voc_holdout_continue_count,
                "voc_holdout_stop_count": self.voc_holdout_stop_count,
                "voc_holdout_td_sum": self.voc_holdout_td_sum,
                "voc_holdout_td_abs_sum": self.voc_holdout_td_abs_sum,
                "voc_holdout_td_sq_sum": self.voc_holdout_td_sq_sum,
                "voc_holdout_td_bias": (
                    self.voc_holdout_td_sum / self.voc_holdout_count
                    if self.voc_holdout_count else None
                ),
                "voc_holdout_td_mae": (
                    self.voc_holdout_td_abs_sum / self.voc_holdout_count
                    if self.voc_holdout_count else None
                ),
                "voc_holdout_td_rmse": (
                    np.sqrt(
                        self.voc_holdout_td_sq_sum / self.voc_holdout_count
                    ) if self.voc_holdout_count else None
                ),
                "voc_parent_checkpoint_sha256": (
                    self.voc_parent_checkpoint_sha256
                ),
                "voc_parent_checkpoint": self.voc_parent_checkpoint,
                "voc_parent_imitation_data_signature": (
                    self.voc_parent_imitation_data_signature
                ),
                "voc_control_origin": self.voc_control_origin,
                "voc_control_origin_legacy_defaulted": (
                    self.voc_control_origin_legacy_defaulted
                ),
                "voc_activation_real_step": self.voc_activation_real_step,
                "voc_optimizer_state_dict": (
                    self.voc_optimizer.state_dict()
                    if self.voc_optimizer is not None else None
                ),
                "voc_scheduler_state_dict": (
                    self.voc_scheduler.state_dict()
                    if self.voc_scheduler is not None else None
                ),
                "voc_grad_scaler_state_dict": (
                    self.voc_scaler.state_dict()
                    if self.voc_scaler is not None else None
                ),
                "voc_amp_skip_count": self.voc_amp_skip_count,
                "voc_amp_consecutive_skips": self.voc_amp_consecutive_skips,
                "voc_gate_optimizer_state_dict": (
                    self.voc_gate_optimizer.state_dict()
                    if self.voc_gate_optimizer is not None else None
                ),
                "voc_gate_scheduler_state_dict": (
                    self.voc_gate_scheduler.state_dict()
                    if self.voc_gate_scheduler is not None else None
                ),
                "voc_gate_grad_scaler_state_dict": (
                    self.voc_gate_scaler.state_dict()
                    if self.voc_gate_scaler is not None else None
                ),
                "voc_gate_update_count": self.voc_gate_update_count,
                "voc_gate_amp_skip_count": self.voc_gate_amp_skip_count,
                "voc_gate_amp_consecutive_skips": (
                    self.voc_gate_amp_consecutive_skips
                ),
                "reward_names": list(self.rewards_ls),
                "imitation_update_count": self.imitation_update_count,
                "imitation_schedule_step": self.imitation_schedule_step,
                "imitation_rng_state": self._loader_rng_state(),
                "imitation_data_signature": self.imitation_data_signature,
                # Audit-only: excluded from identity because SLURM scratch
                # paths legitimately change across resumed jobs.
                "imitation_data_root": self.imitation_data_root,
                "action_prior_ema": (
                    self.action_prior_ema.detach().cpu()
                    if self.action_prior_ema is not None else None
                ),
                "python_rng_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "torch_cuda_rng_state_all": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available() else None
                ),
                "actor_grad_scaler_state_dict": (
                    self.scaler.state_dict() if self.flags.float16 else None
                ),
                "actor_amp_skip_count": self.actor_amp_skip_count,
                "actor_amp_consecutive_skips": self.actor_amp_consecutive_skips,
                "voc_actor_policy_version": self.voc_actor_policy_version,
                "voc_actor_policy_publication_count": (
                    self.voc_actor_policy_publication_count
                ),
                "voc_actor_policy_terminal": self.voc_actor_policy_terminal,
                "voc_actor_policy_version_mismatch_count": (
                    self.voc_actor_policy_version_mismatch_count
                ),
                "voc_actor_policy_malformed_bundle_count": (
                    self.voc_actor_policy_malformed_bundle_count
                ),
                "voc_actor_policy_barrier_timeout_count": (
                    self.voc_actor_policy_barrier_timeout_count
                ),
                "voc_actor_policy_terminal_ack_count": (
                    self.voc_actor_policy_terminal_ack_count
                ),
                "voc_actor_policy_expected_ack_count": (
                    self.voc_actor_policy_expected_ack_count
                ),
                "voc_actor_policy_state_sha256": (
                    self.voc_actor_policy_state_sha256
                ),
                "voc_actor_policy_bundle": (
                    copy.deepcopy(self._voc_actor_policy_bundle)
                    if self.voc_actor_policy_version_barrier else None
                ),
                "voc_actor_policy_publication_history": (
                    tuple(copy.deepcopy(
                        self.voc_actor_policy_publication_history
                    ))
                    if self.voc_actor_policy_version_barrier else None
                ),
                "voc_actor_policy_publication_history_sha256": (
                    self.voc_actor_policy_publication_history_sha256
                    if self.voc_actor_policy_version_barrier else None
                ),
                "flags": vars(self.flags),
            }      
        try:
            # Save regular checkpoint
            torch.save(d, self.ckp_path + ".tmp")
            os.replace(self.ckp_path + ".tmp", self.ckp_path)
            
            # Save step-specific checkpoint if forced or at checkpoint interval
            if force or (hasattr(self.flags, 'checkpoint_interval') and 
                         self.flags.checkpoint_interval > 0 and 
                         self.real_step % self.flags.checkpoint_interval == 0):
                checkpoint_path = f"{self.ckp_path}_step_{self.real_step}"
                torch.save(d, checkpoint_path + ".tmp")
                os.replace(checkpoint_path + ".tmp", checkpoint_path)
                self._logger.info(f"Saved actor checkpoint at step {self.real_step} to {checkpoint_path}")
        except Exception as e:       
            self._logger.error(f"Error saving actor checkpoint: {e}")
            raise

    def _actor_weights_for_publication(self):
        """Return weights only after validating the exact gate boundary."""

        if self.voc_gate_exact_projection:
            self._assert_voc_gate_exact_projection_invariant()
        return self.actor_net.get_weights()

    def load_checkpoint(self, ckp_path: str):
        train_checkpoint = torch.load(ckp_path, torch.device("cpu"), weights_only = False)
        checkpoint_flags = train_checkpoint.get("flags", {})
        checkpoint_dynamic = bool(train_checkpoint.get(
            "dynamic_search", checkpoint_flags.get("dynamic_search", False)
        ))
        if checkpoint_dynamic != self.dynamic_search:
            raise ValueError(
                "Cannot resume actor checkpoint across dynamic_search modes "
                f"(checkpoint={checkpoint_dynamic}, run={self.dynamic_search}). "
                "Use preload_actor for weight-only legacy migration."
            )
        checkpoint_factorized = bool(train_checkpoint.get(
            "dynamic_factorized_control",
            checkpoint_flags.get("dynamic_factorized_control", False),
        ))
        if (
            self.dynamic_search
            and checkpoint_factorized != self.dynamic_factorized_control
        ):
            raise ValueError(
                "Cannot resume actor checkpoint across Dynamic control "
                "objectives "
                f"(checkpoint={checkpoint_factorized}, "
                f"run={self.dynamic_factorized_control})."
            )
        if hasattr(self, "flags"):
            util.validate_voc_resume_protocol(train_checkpoint, self.flags)
        run_voc_mode = normalize_dynamic_voc_mode(
            getattr(self, "dynamic_voc_mode", "off")
        )
        checkpoint_voc_mode = normalize_dynamic_voc_mode(
            train_checkpoint.get(
                "dynamic_voc_mode",
                checkpoint_flags.get("dynamic_voc_mode", "off"),
            )
        )
        if checkpoint_voc_mode != run_voc_mode:
            raise ValueError(
                "Cannot resume actor checkpoint across VoC modes "
                f"(checkpoint={checkpoint_voc_mode!r}, "
                f"run={run_voc_mode!r}). Use weight-only preload "
                "for off->shadow or validated shadow->control promotion."
            )
        if self.voc_dedicated_gate:
            # Validate the complete isolated gate bundle before mutating any
            # learner/optimizer state.  The shared validator also protects
            # evaluator, smoke and promotion paths with the same schema.
            util.validate_voc_gate_policy_checkpoint(
                train_checkpoint,
                flags=self.flags,
                label="Actor resume checkpoint",
            )
            gate_update_count = train_checkpoint.get("voc_gate_update_count")
            if (
                isinstance(gate_update_count, (bool, np.bool_))
                or not isinstance(gate_update_count, (int, np.integer))
                or int(gate_update_count) < 0
            ):
                raise ValueError(
                    "Actor checkpoint has invalid voc_gate_update_count"
                )
            self.voc_gate_update_count = int(gate_update_count)
            for key in (
                "voc_gate_amp_skip_count",
                "voc_gate_amp_consecutive_skips",
            ):
                value = train_checkpoint.get(key)
                if (
                    isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, np.integer))
                    or int(value) < 0
                ):
                    raise ValueError(f"Actor checkpoint has invalid {key}")
                setattr(self, key, int(value))
            if (
                checkpoint_voc_mode == "shadow"
                and self.voc_gate_update_count != 0
            ):
                raise ValueError(
                    "VoC shadow checkpoint cannot contain dedicated gate "
                    "optimizer updates"
                )
        if checkpoint_voc_mode != "off":
            ema_gate_state = util.validate_voc_ema_gate_checkpoint(
                train_checkpoint, label="Actor resume checkpoint"
            )
            self._load_voc_ema_gate_state(
                ema_gate_state["voc_ema_gate_head_state_dict"],
                ema_gate_state["voc_ema_gate_update_count"],
                parent_update_count=ema_gate_state[
                    "voc_ema_gate_parent_update_count"
                ],
            )
            for key in (
                "voc_update_count",
                "voc_continue_count",
                "voc_stop_count",
            ):
                value = train_checkpoint.get(key)
                if (
                    isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, np.integer))
                ):
                    raise ValueError(f"Actor checkpoint has invalid {key}")
                value = int(value)
                if value < 0:
                    raise ValueError(f"Actor checkpoint has negative {key}")
                setattr(self, key, value)
            holdout_state = util.validate_voc_holdout_calibration(
                train_checkpoint,
                label="Actor resume checkpoint",
                require_positive_support=False,
            )
            for key in (
                "voc_holdout_count",
                "voc_holdout_continue_count",
                "voc_holdout_stop_count",
                "voc_holdout_td_sum",
                "voc_holdout_td_abs_sum",
                "voc_holdout_td_sq_sum",
            ):
                setattr(self, key, holdout_state[key])
            util.validate_voc_holdout_split(
                train_checkpoint,
                flags=self.flags,
                label="Actor resume checkpoint",
            )
            util.validate_voc_amp_checkpoint(
                train_checkpoint, label="Actor resume checkpoint"
            )
            if self.voc_update_count > 0:
                util.validate_voc_checkpoint_components(
                    train_checkpoint,
                    flags=self.flags,
                    label="Actor resume checkpoint",
                )
            if checkpoint_voc_mode == "shadow":
                util.validate_voc_shadow_checkpoint_provenance(
                    train_checkpoint, label="Actor resume checkpoint"
                )
                self.voc_parent_checkpoint_sha256 = None
                self.voc_parent_checkpoint = None
                self.voc_parent_imitation_data_signature = None
                self.voc_control_origin = None
                self.voc_control_origin_legacy_defaulted = False
                self.voc_activation_real_step = -1
            else:
                provenance = util.validate_voc_control_checkpoint_provenance(
                    train_checkpoint, label="Actor resume checkpoint"
                )
                self.voc_control_origin = provenance["voc_control_origin"]
                self.voc_control_origin_legacy_defaulted = provenance[
                    "voc_control_origin_legacy_defaulted"
                ]
                self.voc_parent_checkpoint_sha256 = provenance[
                    "voc_parent_checkpoint_sha256"
                ]
                self.voc_parent_checkpoint = provenance[
                    "voc_parent_checkpoint"
                ]
                self.voc_parent_imitation_data_signature = provenance[
                    "voc_parent_imitation_data_signature"
                ]
                self.voc_activation_real_step = provenance[
                    "voc_activation_real_step"
                ]
        checkpoint_arch = train_checkpoint.get("actor_arch_version")
        expected_arch = 2 if self.dynamic_search else 1
        if self.dynamic_search and checkpoint_arch is None:
            raise ValueError(
                "Dynamic actor checkpoint is missing actor_arch_version metadata."
            )
        if checkpoint_arch is not None and checkpoint_arch != expected_arch:
            raise ValueError(
                f"Actor checkpoint architecture {checkpoint_arch} does not "
                f"match expected version {expected_arch}."
            )
        checkpoint_rewards = train_checkpoint.get("reward_names")
        if self.dynamic_search and checkpoint_rewards is None:
            raise ValueError(
                "Dynamic actor checkpoint is missing reward_names metadata."
            )
        if (checkpoint_rewards is not None
                and list(checkpoint_rewards) != list(self.rewards_ls)):
            raise ValueError(
                "Actor checkpoint reward channels do not match this run: "
                f"{checkpoint_rewards} != {self.rewards_ls}."
            )
        self.step = train_checkpoint["step"]
        self.real_step = train_checkpoint["real_step"]
        self.tot_eps = train_checkpoint["tot_eps"]
        self.ret_buffers = train_checkpoint["ret_buffers"]
        self.norm_stats = train_checkpoint["norm_stats"]
        self.crnorm = train_checkpoint["crnorm"]
        # These fields were introduced with Dynamic imitation.  Defaults keep
        # every pre-imitation actor checkpoint loadable.
        imitation_state = imitation_checkpoint_state(train_checkpoint)
        self.imitation_update_count = imitation_state["update_count"]
        self.imitation_schedule_step = imitation_state["schedule_step"]
        self._checkpoint_imitation_rng_state = imitation_state["rng_state"]
        self._checkpoint_imitation_data_signature = imitation_state[
            "data_signature"
        ]
        self.action_prior_ema = imitation_state["action_prior_ema"]
        if (
            self.imitation_update_count > 0
            or self._checkpoint_imitation_data_signature is not None
        ):
            self._validate_resume_imitation_protocol(checkpoint_flags)
        self._checkpoint_python_rng_state = train_checkpoint.get(
            "python_rng_state"
        )
        self._checkpoint_numpy_rng_state = train_checkpoint.get("numpy_rng_state")
        self._checkpoint_torch_rng_state = train_checkpoint.get("torch_rng_state")
        self._checkpoint_cuda_rng_state = train_checkpoint.get(
            "torch_cuda_rng_state_all"
        )
        self._checkpoint_scaler_state = train_checkpoint.get(
            "actor_grad_scaler_state_dict"
        )
        self._checkpoint_voc_scaler_state = train_checkpoint.get(
            "voc_grad_scaler_state_dict"
        )
        self._checkpoint_voc_gate_scaler_state = train_checkpoint.get(
            "voc_gate_grad_scaler_state_dict"
        )
        self.actor_amp_skip_count = int(
            train_checkpoint.get("actor_amp_skip_count", 0)
        )
        self.actor_amp_consecutive_skips = int(
            train_checkpoint.get("actor_amp_consecutive_skips", 0)
        )
        self.voc_amp_skip_count = int(
            train_checkpoint.get("voc_amp_skip_count", 0)
        )
        self.voc_amp_consecutive_skips = int(
            train_checkpoint.get("voc_amp_consecutive_skips", 0)
        )
        util.load_optimizer(self.optimizer, train_checkpoint["actor_net_optimizer_state_dict"])
        util.load_scheduler(self.scheduler, train_checkpoint["actor_net_scheduler_state_dict"])
        if self.voc_optimizer is not None:
            voc_optimizer_state = train_checkpoint.get(
                "voc_optimizer_state_dict"
            )
            voc_scheduler_state = train_checkpoint.get(
                "voc_scheduler_state_dict"
            )
            if voc_optimizer_state is None or voc_scheduler_state is None:
                raise ValueError(
                    "VoC checkpoint lacks optimizer/scheduler state"
                )
            util.load_optimizer(self.voc_optimizer, voc_optimizer_state)
            util.load_scheduler(self.voc_scheduler, voc_scheduler_state)
        if self.voc_gate_optimizer is not None:
            gate_optimizer_state = train_checkpoint.get(
                "voc_gate_optimizer_state_dict"
            )
            gate_scheduler_state = train_checkpoint.get(
                "voc_gate_scheduler_state_dict"
            )
            if gate_optimizer_state is None or gate_scheduler_state is None:
                raise ValueError(
                    "dedicated VoC gate checkpoint lacks optimizer/scheduler "
                    "state"
                )
            if self.flags.float16 and (
                self._checkpoint_voc_gate_scaler_state is None
            ):
                raise ValueError(
                    "FP16 dedicated VoC gate checkpoint lacks scaler state"
                )
            util.load_optimizer(
                self.voc_gate_optimizer, gate_optimizer_state
            )
            util.load_scheduler(
                self.voc_gate_scheduler, gate_scheduler_state
            )
        self.actor_net.set_weights(train_checkpoint["actor_net_state_dict"])
        if self.voc_dedicated_gate:
            for parameter in self.voc_gate_parameters:
                _require_finite_tensor(
                    "checkpoint dedicated VoC gate parameter", parameter
                )
            if checkpoint_voc_mode == "shadow":
                self._require_fresh_voc_gate_zero_initialization(
                    self.voc_gate_parameters
                )
        self._logger.info("Loaded actor checkpoint from %s" % ckp_path)

    def refresh_actor(self):
        deadline = (
            self._monotonic() + self.voc_actor_policy_barrier_timeout_s
            if self.voc_actor_policy_barrier_runtime else None
        )
        while True:
            ref = self.actor_param_buffer.get_data.remote("actor_net")
            weights = (
                self._barrier_ray_get(
                    ref,
                    deadline=deadline,
                    label="learner raw actor bootstrap",
                )
                if self.voc_actor_policy_barrier_runtime
                else ray.get(ref)
            )
            if weights is not None:
                self.actor_net.set_weights(weights)
                del weights
                break                
            self._barrier_sleep(0.1)

    def _seal_schema13_telemetry_before_finish(self):
        """Close legacy logs and publish the manifest before replay FINISH."""

        if not self._voc_telemetry_active:
            return None
        if self._voc_actor_policy_transaction_open:
            raise RuntimeError(
                "schema-13 terminal telemetry cannot seal an open transaction"
            )
        if self._voc_telemetry_pending is not None:
            raise RuntimeError(
                "schema-13 terminal telemetry retains volatile staging"
            )
        if (
            self._voc_telemetry_writer.transaction_count
            != self.voc_actor_policy_version
        ):
            raise RuntimeError(
                "schema-13 terminal telemetry count disagrees with policy version"
            )
        if (
            self.voc_actor_policy_terminal is not True
            or self.voc_actor_policy_publication_count
            != self.voc_actor_policy_version
            or self.voc_actor_policy_terminal_ack_count != 1
            or self.voc_actor_policy_expected_ack_count != 1
            or self.real_step < self.flags.total_steps
        ):
            raise RuntimeError(
                "schema-13 terminal publication evidence is incomplete"
            )
        log_path = self.plogger.paths["logs"]
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        if hasattr(os, "O_NONBLOCK"):
            flags |= os.O_NONBLOCK
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        # Bind the reader to the exact inode still owned by FileWriter before
        # closing its write handle.  A close/open pathname gap would otherwise
        # allow a complete, well-formed replacement log to be sealed.
        writer_log = getattr(self.plogger, "_logfile", None)
        if writer_log is None or writer_log.closed:
            raise RuntimeError("schema-13 legacy actor log writer is not open")
        writer_log.flush()
        writer_fd = writer_log.fileno()
        os.fsync(writer_fd)
        writer_info = os.fstat(writer_fd)
        if (
            not stat.S_ISREG(writer_info.st_mode)
            or writer_info.st_nlink != 1
            or writer_info.st_uid != os.geteuid()
            or writer_info.st_gid != os.getegid()
        ):
            raise RuntimeError(
                "schema-13 legacy actor log writer identity is malformed"
            )
        log_fd = os.open(log_path, flags)
        try:
            info = os.fstat(log_fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or info.st_uid != os.geteuid()
                or info.st_gid != os.getegid()
                or info.st_dev != writer_info.st_dev
                or info.st_ino != writer_info.st_ino
            ):
                raise RuntimeError(
                    "schema-13 legacy actor log identity is malformed"
                )
            self.plogger.close(successful=True)
            self._voc_telemetry_log_closed = True
            os.fsync(log_fd)
            self._voc_telemetry_evidence = self._voc_telemetry_writer.seal(
                terminal_real_step=int(self.real_step),
                terminal_policy_version=int(self.voc_actor_policy_version),
                terminal_publication_count=int(
                    self.voc_actor_policy_publication_count
                ),
                terminal_ack_count=int(self.voc_actor_policy_terminal_ack_count),
                legacy_actor_log_path=log_path,
                legacy_actor_log_fd=log_fd,
            )
        finally:
            os.close(log_fd)
        return copy.deepcopy(self._voc_telemetry_evidence)

    def close(self, successful=True):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        telemetry_error = None
        telemetry_active = bool(
            getattr(self, "_voc_telemetry_active", False)
        )
        if self.bc_runner is not None:
            try:
                self.bc_runner.close()
            except BaseException as error:
                if not telemetry_active:
                    raise
                successful = False
                telemetry_error = error
        if telemetry_active and successful:
            try:
                self._seal_schema13_telemetry_before_finish()
            except BaseException as error:
                successful = False
                telemetry_error = error
                writer = getattr(self, "_voc_telemetry_writer", None)
                if writer is not None and not writer.poisoned:
                    writer.abort()
        if telemetry_active and not successful:
            writer = getattr(self, "_voc_telemetry_writer", None)
            if writer is not None:
                writer.abort()
        if self.voc_actor_policy_barrier_runtime:
            if successful:
                if (
                    not self.voc_actor_policy_terminal
                    or self.voc_actor_policy_terminal_ack_count
                    != self.voc_actor_policy_expected_ack_count
                ):
                    raise RuntimeError(
                        "successful schema-6 close requires an acknowledged "
                        "terminal publication"
                    )
            else:
                # Never launder a failed run into a valid terminal bundle.
                # This separate diagnostic merely wakes observers; FINISH
                # below unblocks ActorBuffer readers without qualification.
                try:
                    self._barrier_ray_get(
                        self.actor_param_buffer.set_data.remote(
                            util.VOC_ACTOR_POLICY_ABORT_KEY,
                            {
                                "policy_version": self.voc_actor_policy_version,
                                "terminal": False,
                            },
                        ),
                        deadline=(
                            self._monotonic()
                            + self.voc_actor_policy_barrier_timeout_s
                        ),
                        label="learner abort diagnostic",
                    )
                except Exception:
                    self._logger.error(
                        "failed to publish schema-6 learner abort diagnostic"
                    )
        if hasattr(self, "actor_buffer") and self.actor_buffer is not None:
            if self.voc_actor_policy_barrier_runtime and not successful:
                # ActorBuffer has only a normal FINISH bit.  Killing this
                # no-restart actor makes pending reads fail immediately
                # without forging that normal completion state.
                ray.kill(self.actor_buffer, no_restart=True)
            else:
                try:
                    finish_ref = self.actor_buffer.set_finish.remote()
                    if self.voc_actor_policy_barrier_runtime and successful:
                        self._barrier_ray_get(
                            finish_ref,
                            deadline=(
                                self._monotonic()
                                + self.voc_actor_policy_barrier_timeout_s
                            ),
                            label="actor replay FINISH",
                        )
                except BaseException as error:
                    if not (
                        telemetry_active
                        and self.voc_actor_policy_barrier_runtime
                        and successful
                    ):
                        raise
                    # The telemetry manifest is already sealed at this
                    # boundary, so training state cannot be rolled back.
                    # Poison the run, kill the replay actor, rewrite legacy
                    # metadata as unsuccessful below, and never retry FINISH.
                    successful = False
                    telemetry_error = error
                    writer = getattr(self, "_voc_telemetry_writer", None)
                    if writer is not None:
                        writer.abort()
                    try:
                        ray.kill(self.actor_buffer, no_restart=True)
                    except Exception:
                        self._logger.error(
                            "failed to kill actor replay after FINISH failure"
                        )
        if not (
            telemetry_active
            and getattr(self, "_voc_telemetry_log_closed", False)
            and successful
        ):
            self.plogger.close(successful=bool(successful))
        if telemetry_error is not None:
            raise telemetry_error


@ray.remote
class ActorLearner(SActorLearner):
    pass
