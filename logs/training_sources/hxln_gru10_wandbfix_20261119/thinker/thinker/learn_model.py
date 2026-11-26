import os
import numpy as np
import time
import timeit
import traceback
import ray
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from thinker.core.file_writer import FileWriter
from thinker.core.module import guassian_kl_div
from thinker.buffer import (
    MODEL_BUFFER_ABORT,
    validate_priorities,
    validate_schema7_model_buffer_status,
)
from thinker.model_net import ModelNet, VPNet
import thinker.util as util
import gc
from collections import namedtuple


def _resolve_model_input_seal_schema_version(flags):
    gate_schema = getattr(flags, "voc_gate_policy_schema_version", None)
    seal_schema = getattr(flags, "voc_model_input_seal_schema_version", 0)
    expected = (
        1
        if type(gate_schema) is int and gate_schema in (7, 8, 9, 10, 11, 12, 13)
        else 0
    )
    if type(seal_schema) is not int or seal_schema != expected:
        raise ValueError(
            "voc_model_input_seal_schema_version must be exact integer "
            f"{expected} for gate schema {gate_schema!r}"
        )
    return seal_schema


def _raise_nonfinite_tensor(value, context):
    finite = torch.isfinite(value)
    invalid = ~finite
    flat_idx = int(torch.nonzero(invalid.reshape(-1), as_tuple=False)[0].item())
    invalid_n = int(invalid.sum().item())
    first_value = value.detach().reshape(-1)[flat_idx].cpu().item()
    raise FloatingPointError(
        f"{context} contains {invalid_n}/{value.numel()} non-finite values; "
        f"first invalid value is {first_value!r} at flat index {flat_idx} "
        f"(shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device})"
    )


def _assert_finite_tensor_entries(entries):
    by_device = {}
    for context, value in entries:
        if value.numel() > 0:
            by_device.setdefault(value.device, []).append((context, value))

    # Batch all healthy-path checks into one synchronization per device. Exact
    # field diagnostics are computed only if the combined check fails.
    for device_entries in by_device.values():
        checks = torch.stack(
            [torch.isfinite(value).all() for _, value in device_entries]
        )
        if bool(torch.all(checks).item()):
            continue
        for (context, value), finite in zip(device_entries, checks):
            if not bool(finite.item()):
                _raise_nonfinite_tensor(value, context)


def assert_finite_tensors(value, *, context):
    """Fail with a field-level diagnostic when a tensor tree is non-finite."""
    entries = []

    def collect(child, child_context):
        if child is None:
            return
        if torch.is_tensor(child):
            entries.append((child_context, child))
            return
        if isinstance(child, dict):
            for key, item in child.items():
                collect(item, f"{child_context}.{key}")
            return
        if hasattr(child, "_fields"):
            for key in child._fields:
                collect(getattr(child, key), f"{child_context}.{key}")
            return
        if isinstance(child, (tuple, list)):
            for index, item in enumerate(child):
                collect(item, f"{child_context}[{index}]")
            return
        if isinstance(child, (float, np.floating)) and not np.isfinite(child):
            raise FloatingPointError(f"{child_context} is non-finite: {child!r}")

    collect(value, context)
    _assert_finite_tensor_entries(entries)


def assert_optimizer_parameters_finite(optimizer, *, context):
    entries = [
        (
            f"{context}.param_groups[{group_index}].params[{parameter_index}]",
            parameter.data,
        )
        for group_index, group in enumerate(optimizer.param_groups)
        for parameter_index, parameter in enumerate(group["params"])
    ]
    _assert_finite_tensor_entries(entries)


GradientStepResult = namedtuple(
    "GradientStepResult",
    ["total_norm", "optimizer_stepped", "amp_scale_before", "amp_scale_after"],
)


_MODEL_OBSERVABILITY_PREFIXES = (
    "pred_sr_hs",
    "pred_vp_hs",
    "pred_policy_logits",
    "pred_value_head",
    "pred_reward_head",
)


def _empty_model_observability():
    return {
        f"{prefix}_{suffix}": None
        for prefix in _MODEL_OBSERVABILITY_PREFIXES
        for suffix in ("abs_max", "rms")
    }


def _record_tensor_scale(observability, prefix, value):
    """Record detached device scalars without synchronizing the accelerator."""
    if value is None or value.numel() == 0:
        return
    value = value.detach().float()
    observability[f"{prefix}_abs_max"] = torch.amax(torch.abs(value))
    observability[f"{prefix}_rms"] = torch.sqrt(torch.mean(torch.square(value)))

def compute_cross_entropy_loss(policy, target_policy, discrete_action, require_prob, is_weights, mask=None):
    k, b, d, _ = policy.shape
    if discrete_action:
        loss = torch.nn.CrossEntropyLoss(reduction="none")(
            input=torch.flatten(policy, 0, 2), target=torch.flatten(target_policy, 0, 2)
        )
        loss = loss.view(k, b, d)
        loss = torch.mean(loss, dim=2)
    elif require_prob:
        tar_mean = target_policy[:, :, :, 0]
        tar_log_var = target_policy[:, :, :, 1]
        mean = policy[:, :, :, 0]
        log_var = policy[:, :, :, 1]
        loss = guassian_kl_div(
            tar_mean, tar_log_var, mean, log_var, reduce="mean"
        )
    else:
        loss = 0.5 * (log_var + ((policy - mean) ** 2) /  torch.exp(log_var))
        loss = torch.mean(loss, dim=-1)
    if mask is not None: loss = loss * mask
    loss = torch.sum(loss, dim=0)
    loss = is_weights * loss
    return torch.sum(loss)
   
class SModelLearner:
    def __init__(self, name, ray_obj, model_param, flags, model_net=None, device=None):
        self.flags = flags
        self.voc_model_input_seal_schema_version = (
            _resolve_model_input_seal_schema_version(flags)
        )
        self.voc_model_input_seal_runtime = (
            self.voc_model_input_seal_schema_version == 1
        )
        if self.voc_model_input_seal_runtime:
            raw_timeout = getattr(
                flags,
                "voc_actor_policy_barrier_timeout_s",
                util.VOC_ACTOR_POLICY_BARRIER_TIMEOUT_SECONDS,
            )
            if (
                isinstance(raw_timeout, bool)
                or type(raw_timeout) not in (int, float)
                or not np.isfinite(raw_timeout)
                or float(raw_timeout)
                != util.VOC_ACTOR_POLICY_BARRIER_TIMEOUT_SECONDS
            ):
                raise ValueError(
                    "schema-7 ModelLearner requires the exact finite "
                    "actor-policy barrier timeout"
                )
            self.voc_actor_policy_barrier_timeout_s = float(raw_timeout)
            self.voc_model_input_sealed = False
            self.voc_model_input_seal_count = 0
            self.voc_model_terminal_processed_n = -1
            self.voc_model_terminal_drain_update_count = 0
            self.voc_model_terminal_drain_pre_real_step = -1
            self.voc_model_terminal_drain_pre_grad_step_count_m = -1
            self.voc_model_terminal_drain_pre_grad_step_count_p = -1
            self.voc_model_input_late_write_count = 0
            self.voc_model_input_abort_count = 0
            self._schema7_terminal_drain_active = False
        self.model_float16 = getattr(
            flags, "model_float16", getattr(flags, "float16", False)
        )
        if not isinstance(self.model_float16, bool):
            raise TypeError(
                "model_float16 must be resolved to bool before ModelLearner "
                f"construction, got {self.model_float16!r}"
            )
        self.time = flags.profile
        self._logger = util.logger()

        if flags.parallel:
            self.model_buffer = ray_obj["model_buffer"]
            self.param_buffer = ray_obj["param_buffer"]
            self.signal_buffer = ray_obj["signal_buffer"]
            self.model_net = ModelNet(**model_param)
            self.refresh_model()
            self.model_net.train(True)
            if self.flags.gpu_learn > 0. and torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:           
                self.device = torch.device("cpu")
        else:
            assert model_net is not None, "actor_net is required for non-parallel mode"
            assert device is not None, "device is required for non-parallel mode"
            self.model_net = model_net
            self.device = device

        self.reward_n = self.model_net.reward_n

        if self.device == torch.device("cuda"):
            self._logger.info("Init. model-learning: Using CUDA.")
        else:
            self._logger.info("Init. model-learning: Not using CUDA.")

        self.step = 0
        self.real_step = 0
        self._initialize_gradient_clip_counters()

        lr_lambda = lambda epoch: 1.0 - util.schedule_progress(self.flags, epoch)

        opt = getattr(flags, "model_optimizer", "adam")
        if opt == "adam":
            Optimizer = torch.optim.Adam
            opt_args = {}
        elif opt == "sgd":
            Optimizer = torch.optim.SGD
            opt_args = {
                "momentum": self.flags.model_sgd_momentum,
                "weight_decay": self.flags.model_sgd_weight_decay,
            }

        if self.flags.dual_net:
            self.optimizer_m = Optimizer(
                self.model_net.sr_net.parameters(), lr=flags.model_learning_rate, **opt_args
            )
            self.scheduler_m = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer_m, lr_lambda
            )
            self.scaler_m = GradScaler(init_scale=2**3) if self.model_float16 else None
        
        param_groups = self.model_net.vp_net.parameters()
        self.optimizer_p = Optimizer(param_groups, lr=flags.model_learning_rate, **opt_args)

        self.scheduler_p = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_p, lr_lambda
        )
        self.scaler_p = GradScaler(init_scale=2**3) if self.model_float16 else None

        self.ckp_path = os.path.join(flags.ckpdir, "ckp_model.tar")
        if flags.ckp: self.load_checkpoint(self.ckp_path)

        self.plogger = FileWriter(
            xpid=flags.xpid,
            xp_args=flags.__dict__,
            rootdir=flags.savedir,
            suffix="_model",
            overwrite=not self.flags.ckp,
        )

        # move network and optimizer to process device
        self.model_net.to(self.device)
        if self.flags.dual_net:
            util.optimizer_to(self.optimizer_m, self.device)
        util.optimizer_to(self.optimizer_p, self.device)               

        self.timing = util.Timings() if self.time else None
        self.perfect_model = util.check_perfect_model(flags.wrapper_type)

        # other init. variables for consume_data
        self.last_psteps = 0
        self.timer = timeit.default_timer
        self.start_step = self.step
        self.start_time = self.timer()
        self.sps_buffer = [(self.step, self.start_time)] * 36
        self.sps_start_time, self.sps_start_step = self.start_time, self.step
        self.sps_buffer_n = 0
        self.ckp_start_time = int(time.strftime("%M")) // 10
        self.n = 0

        self.model_T = flags.model_unroll_len + 1
        self.model_B = flags.model_batch_size
        self.numel_per_step = self.model_T * self.model_B
        self.replay_ratio = 0

        if flags.parallel:
            self.data_ptr = self.read_buffer_ptr()
        self.start_training = False
        self.finish = False

    def read_buffer_ptr(self):
        return self.model_buffer.read.remote(self.model_T, self.model_B, self.compute_beta(), add_t=self.flags.model_return_n+1)

    def compute_beta(self):
        c = util.schedule_progress(self.flags, self.real_step)
        return self.flags.priority_beta * (1 - c) + 1.0 * c
    
    def init_psteps(self, data):
        if data is not None and not self.start_training:                                    
            # record the last processed steps from buffer
            self.last_psteps = int(data["processed_n"])
            if not self.flags.ckp:
                self.real_step += self.last_psteps
                # if it is not loading from checkpoint, the steps
                # used to fill the model should also be counted
            self.start_training = True   
    
    def log_preload(self, status):
        if self.timer() - self.start_time > 5:
            self._logger.info(
                "[%s] Preloading: %d/%d"
                % (self.flags.xpid, status["processed_n"], status["warm_up_n"])
            )
            self.start_time = self.timer()

    def learn_data(self):
        if getattr(self, "voc_model_input_seal_runtime", False):
            return self._learn_data_schema7()
        return self._learn_data_legacy()

    def _schema7_ray_get(self, object_ref, *, label):
        timeout = getattr(self, "voc_actor_policy_barrier_timeout_s", None)
        if (
            type(timeout) is not float
            or not np.isfinite(timeout)
            or timeout != util.VOC_ACTOR_POLICY_BARRIER_TIMEOUT_SECONDS
        ):
            raise RuntimeError(
                "schema-7 ModelLearner RPC timeout is not the exact finite bound"
            )
        try:
            return ray.get(object_ref, timeout=timeout)
        except ray.exceptions.GetTimeoutError as error:
            raise TimeoutError(
                f"schema-7 ModelLearner RPC timed out during {label}"
            ) from error

    def _validate_schema7_model_status(self, status, *, require_sealed=False):
        return validate_schema7_model_buffer_status(
            status,
            total_steps=self.flags.total_steps,
            self_play_n=self.flags.self_play_n,
            warm_up_n=self.flags.model_warm_up_n,
            require_sealed=require_sealed,
            label="schema-7 ModelLearner ModelBuffer status",
        )

    def _get_schema7_model_status(self):
        status = self._schema7_ray_get(
            self.model_buffer.get_status.remote(),
            label="ModelBuffer status",
        )
        return self._validate_schema7_model_status(status)

    def _raise_if_schema7_model_aborted(self, status):
        if status["voc_model_input_aborted"]:
            raise RuntimeError("schema-7 model input was aborted")

    def _begin_schema7_model_update(self, expected_processed_n):
        response = self._schema7_ray_get(
            self.model_buffer.begin_model_update.remote(expected_processed_n),
            label="ModelBuffer begin_model_update",
        )
        if not isinstance(response, dict) or set(response) != {
            "allowed",
            "token",
            "status",
        }:
            raise RuntimeError("schema-7 model update claim response is malformed")
        allowed = response["allowed"]
        token = response["token"]
        if type(allowed) is not bool:
            raise RuntimeError("schema-7 model update claim allowed is not boolean")
        status = self._validate_schema7_model_status(response["status"])
        if allowed:
            if (
                type(token) is not int
                or token <= 0
                or status["voc_model_update_claim_active"] is not True
                or status["voc_model_input_sealed"]
                or status["voc_model_input_aborted"]
                or status["finish"]
            ):
                raise RuntimeError("schema-7 model update claim grant is invalid")
        elif (
            token is not None
            or status["voc_model_update_claim_active"]
            or not (
                status["voc_model_input_sealed"]
                or status["voc_model_input_aborted"]
                or status["finish"]
            )
        ):
            raise RuntimeError("schema-7 model update claim denial is invalid")
        return allowed, token, status

    def _end_schema7_model_update(self, token):
        response = self._schema7_ray_get(
            self.model_buffer.end_model_update.remote(token),
            label="ModelBuffer end_model_update",
        )
        if (
            not isinstance(response, dict)
            or set(response) != {"token", "status"}
            or type(response["token"]) is not int
            or response["token"] != token
        ):
            raise RuntimeError("schema-7 model update release response is malformed")
        status = self._validate_schema7_model_status(response["status"])
        if status["voc_model_update_claim_active"]:
            raise RuntimeError("schema-7 model update claim remained active")
        return status

    def _validate_schema7_replay_data_header(self, data):
        if not isinstance(data, dict):
            raise RuntimeError("schema-7 ModelBuffer replay data must be a mapping")
        processed_n = data.get("processed_n")
        replay_ratio = data.get("replay_ratio")
        sealed = data.get("voc_model_input_sealed")
        terminal_processed_n = data.get("voc_model_terminal_processed_n")
        if (
            type(data.get("voc_model_input_seal_schema_version")) is not int
            or data.get("voc_model_input_seal_schema_version") != 1
            or type(processed_n) is not int
            or processed_n < 0
            or type(replay_ratio) is not float
            or not np.isfinite(replay_ratio)
            or replay_ratio < 0.0
            or type(sealed) is not bool
        ):
            raise RuntimeError("schema-7 ModelBuffer replay header is malformed")
        if sealed:
            if (
                type(terminal_processed_n) is not int
                or terminal_processed_n != processed_n
            ):
                raise RuntimeError(
                    "schema-7 sealed replay header has invalid terminal progress"
                )
        elif terminal_processed_n is not None:
            raise RuntimeError(
                "schema-7 unsealed replay header has terminal progress"
            )
        return data

    def _publish_schema7_model_update(self):
        assert_finite_tensors(
            self.model_net.state_dict(),
            context=f"published ModelNet state at real_step={self.real_step}",
        )
        self.param_buffer.set_data.remote(
            "model_net", self.model_net.get_weights()
        )

    def _schema7_checkpoint_iteration(self):
        if int(time.strftime("%M")) // 10 != self.ckp_start_time:
            self.save_checkpoint()
            self.ckp_start_time = int(time.strftime("%M")) // 10
        if hasattr(self.flags, "checkpoint_interval"):
            interval = self.flags.checkpoint_interval
            if interval > 0:
                current_milestone = (self.real_step // interval) * interval
                if not hasattr(self, "last_checkpoint_milestone"):
                    self.last_checkpoint_milestone = -1
                if current_milestone > self.last_checkpoint_milestone:
                    self._logger.info(
                        "Triggering model step-based checkpoint at step %s "
                        "(milestone %s)",
                        self.real_step,
                        current_milestone,
                    )
                    self.save_checkpoint(force=True)
                    self.last_checkpoint_milestone = current_milestone

    def _complete_schema7_terminal_drain(self, status, prefetched_data_ptr):
        if getattr(self, "_schema7_terminal_drain_active", False):
            raise RuntimeError("schema-7 terminal drain started more than once")
        self._schema7_terminal_drain_active = True
        status = self._validate_schema7_model_status(
            status, require_sealed=True
        )
        self._raise_if_schema7_model_aborted(status)
        if status["finish"]:
            raise RuntimeError("ModelBuffer reported success before ModelLearner")
        if status["voc_model_input_late_write_count"] != 0:
            raise RuntimeError("schema-7 ModelBuffer observed a late write")
        if status["voc_model_update_claim_active"]:
            raise RuntimeError("schema-7 terminal drain observed an active update claim")
        if prefetched_data_ptr is not None:
            ray.internal.free(prefetched_data_ptr)

        terminal_processed_n = status["voc_model_terminal_processed_n"]
        if type(self.last_psteps) is not int or self.last_psteps < 0:
            raise RuntimeError("schema-7 model update cursor is invalid")
        if (
            type(self.real_step) is not int
            or self.real_step < 0
            or self.real_step != self.last_psteps
        ):
            raise RuntimeError(
                "schema-7 model real_step/update cursor lost lockstep before drain"
            )
        pre_real_step = self.real_step
        if pre_real_step > terminal_processed_n:
            raise RuntimeError("schema-7 model update cursor exceeds sealed input")

        pre_counts = self._gradient_clip_checkpoint_state()
        self.voc_model_input_sealed = True
        self.voc_model_input_seal_count = status[
            "voc_model_input_seal_count"
        ]
        self.voc_model_terminal_processed_n = terminal_processed_n
        self.voc_model_terminal_drain_pre_real_step = pre_real_step
        self.voc_model_terminal_drain_pre_grad_step_count_m = pre_counts[
            "model_grad_step_count_m"
        ]
        self.voc_model_terminal_drain_pre_grad_step_count_p = pre_counts[
            "model_grad_step_count_p"
        ]
        self.voc_model_input_late_write_count = status[
            "voc_model_input_late_write_count"
        ]
        self.voc_model_input_abort_count = status[
            "voc_model_input_abort_count"
        ]

        drain_update_count = 0
        if terminal_processed_n > pre_real_step:
            fresh_data = self._schema7_ray_get(
                self.read_buffer_ptr(), label="fresh post-seal replay read"
            )
            if not isinstance(fresh_data, dict):
                raise RuntimeError(
                    "schema-7 terminal drain requires one fresh replay batch"
                )
            fresh_data = self._validate_schema7_replay_data_header(fresh_data)
            if (
                fresh_data["voc_model_input_sealed"] is not True
                or fresh_data.get("voc_model_terminal_processed_n")
                != terminal_processed_n
                or fresh_data["processed_n"] != terminal_processed_n
            ):
                raise RuntimeError(
                    "schema-7 terminal drain batch is not post-seal fresh"
                )
            self.replay_ratio = fresh_data["replay_ratio"]
            if self.consume_data(fresh_data) is not True:
                raise RuntimeError(
                    "schema-7 terminal drain did not complete an optimizer step"
                )
            drain_update_count = 1
            self._publish_schema7_model_update()
        elif pre_real_step != terminal_processed_n:
            raise RuntimeError(
                "schema-7 ModelLearner progress disagrees with sealed cursor"
            )
        self.voc_model_terminal_drain_update_count = drain_update_count

        final_counts = self._gradient_clip_checkpoint_state()
        for component in ("m", "p"):
            expected = pre_counts[f"model_grad_step_count_{component}"] + (
                drain_update_count
            )
            if final_counts[f"model_grad_step_count_{component}"] != expected:
                raise RuntimeError(
                    "schema-7 terminal drain gradient-step count is invalid"
                )
        if (
            self.real_step != terminal_processed_n
            or self.last_psteps != terminal_processed_n
        ):
            raise RuntimeError(
                "schema-7 terminal drain did not bind final model progress"
            )
        gate_schema = getattr(
            self.flags, "voc_gate_policy_schema_version", None
        )
        if type(gate_schema) is not int:
            raise RuntimeError(
                "sealed ModelLearner terminal bundle has no exact gate schema"
            )
        if gate_schema == 7:
            validate_final_bundle = util.validate_schema7_final_bundle
        elif gate_schema == 8:
            validate_final_bundle = util.validate_schema8_final_bundle
        elif gate_schema == 9:
            validate_final_bundle = util.validate_schema9_final_bundle
        elif gate_schema == 10:
            validate_final_bundle = util.validate_schema10_final_bundle
        elif gate_schema == 11:
            validate_final_bundle = util.validate_schema11_final_bundle
        elif gate_schema == 12:
            validate_final_bundle = util.validate_schema12_final_bundle
        elif gate_schema == 13:
            validate_final_bundle = util.validate_schema13_final_bundle
        else:
            raise RuntimeError(
                "sealed ModelLearner terminal bundle requires gate schema 7--13"
            )
        self._logger.info(
            f"Terminating schema-{gate_schema} model-learning thread after "
            "sealed input"
        )
        self.save_checkpoint(force=True, terminal=True)
        validated_bundle = validate_final_bundle(
            self.flags.ckpdir,
            label=f"schema-{gate_schema} ModelLearner terminal bundle",
        )
        if (
            not isinstance(validated_bundle, dict)
            or validated_bundle.get("model_real_step") != terminal_processed_n
            or validated_bundle.get("model_input_seal")
            != self._schema7_model_input_checkpoint_evidence(
                require_terminal=True
            )
        ):
            raise RuntimeError(
                f"schema-{gate_schema} authoritative terminal bundle "
                "validation disagrees"
            )
        completed = self._schema7_ray_get(
            self.model_buffer.complete_success.remote(terminal_processed_n),
            label="ModelBuffer complete_success",
        )
        completed = self._validate_schema7_model_status(
            completed, require_sealed=True
        )
        if completed["finish"] is not True:
            raise RuntimeError("schema-7 ModelBuffer did not acknowledge success")
        self.signal_buffer.update_dict_item.remote(
            "self_play_signals", "halt", False
        )
        self._schema7_terminal_drain_active = False

    def _learn_data_schema7(self):
        successful = False
        data_ptr = None
        try:
            data_ptr = self.read_buffer_ptr()
            compatibility_data_ptr = getattr(self, "data_ptr", None)
            if compatibility_data_ptr is not None:
                try:
                    self._schema7_ray_get(
                        compatibility_data_ptr,
                        label="initial compatibility replay read",
                    )
                finally:
                    ray.internal.free(compatibility_data_ptr)
                    self.data_ptr = None
            while True:
                if (
                    self.real_step < self.flags.total_steps
                    and self.replay_ratio < self.flags.max_replay_ratio
                ):
                    current_data_ptr = data_ptr
                    data_ptr = None
                    data = self._schema7_ray_get(
                        current_data_ptr, label="ModelBuffer replay read"
                    )
                    ray.internal.free(current_data_ptr)
                    data_ptr = self.read_buffer_ptr()
                    if data == MODEL_BUFFER_ABORT:
                        raise RuntimeError("schema-7 model input was aborted")
                    if data == "FINISH":
                        raise RuntimeError(
                            "schema-7 ModelBuffer finished before learner success"
                        )
                    if data is not None:
                        data = self._validate_schema7_replay_data_header(data)
                    status = self._get_schema7_model_status()
                    self._raise_if_schema7_model_aborted(status)
                    if data is None:
                        if status["voc_model_input_sealed"]:
                            terminal_data_ptr = data_ptr
                            data_ptr = None
                            self._complete_schema7_terminal_drain(
                                status, terminal_data_ptr
                            )
                            break
                        self.log_preload(status)
                        time.sleep(0.01)
                        continue
                    if (
                        data["voc_model_input_sealed"]
                        or status["voc_model_input_sealed"]
                    ):
                        if not status["voc_model_input_sealed"]:
                            raise RuntimeError(
                                "schema-7 replay observed a seal absent from status"
                            )
                        terminal_data_ptr = data_ptr
                        data_ptr = None
                        self._complete_schema7_terminal_drain(
                            status, terminal_data_ptr
                        )
                        break
                    if data["processed_n"] > status["processed_n"]:
                        raise RuntimeError(
                            "schema-7 replay progress exceeds latest buffer status"
                        )
                    allowed, claim_token, claim_status = (
                        self._begin_schema7_model_update(data["processed_n"])
                    )
                    if not allowed:
                        self._raise_if_schema7_model_aborted(claim_status)
                        if claim_status["finish"]:
                            raise RuntimeError(
                                "schema-7 ModelBuffer finished before learner success"
                            )
                        terminal_data_ptr = data_ptr
                        data_ptr = None
                        self._complete_schema7_terminal_drain(
                            claim_status, terminal_data_ptr
                        )
                        break
                    if data["processed_n"] > claim_status["processed_n"]:
                        raise RuntimeError(
                            "schema-7 replay progress exceeds claimed buffer status"
                        )
                    self.init_psteps(data)
                    self.replay_ratio = data["replay_ratio"]
                    if self.consume_data(data) is not True:
                        raise RuntimeError(
                            "Model consume_data returned without completing an "
                            "optimizer step"
                        )
                    end_status = self._end_schema7_model_update(claim_token)
                    self._raise_if_schema7_model_aborted(end_status)
                    if end_status["finish"]:
                        raise RuntimeError(
                            "schema-7 ModelBuffer finished before learner success"
                        )
                    self._publish_schema7_model_update()
                    if end_status["voc_model_input_sealed"]:
                        terminal_data_ptr = data_ptr
                        data_ptr = None
                        self._complete_schema7_terminal_drain(
                            end_status, terminal_data_ptr
                        )
                        break
                    self._schema7_checkpoint_iteration()
                    del data
                    gc.collect()
                else:
                    time.sleep(0.01)
                    status = self._get_schema7_model_status()
                    self._raise_if_schema7_model_aborted(status)
                    if status["voc_model_input_sealed"]:
                        terminal_data_ptr = data_ptr
                        data_ptr = None
                        self._complete_schema7_terminal_drain(
                            status, terminal_data_ptr
                        )
                        break
                    self.replay_ratio = status["replay_ratio"]
            successful = True
            return True
        except Exception as error:
            self._logger.error(f"Exception detected in learn_model: {error}")
            self._logger.error(traceback.format_exc())
            try:
                self._schema7_ray_get(
                    self.model_buffer.abort_input.remote(),
                    label="ModelBuffer abort_input",
                )
            except Exception as abort_error:
                self._logger.error(
                    "Failed to acknowledge schema-7 ModelBuffer abort: %s",
                    abort_error,
                )
            raise
        finally:
            if data_ptr is not None:
                ray.internal.free(data_ptr)
            self.close(successful=successful)

    def _learn_data_legacy(self):
        successful = False
        try:
            data_ptr = self.read_buffer_ptr()

            while self.real_step < self.flags.total_steps:
                if self.time: self.timing.reset()
                # get data remotely
                if self.replay_ratio < self.flags.max_replay_ratio:
                    while True:                    
                        data = ray.get(data_ptr)
                        ray.internal.free(data_ptr)
                        data_ptr = self.read_buffer_ptr()
                        self.init_psteps(data)
                        if data is not None: break
                        time.sleep(0.01)
                        status = ray.get(self.model_buffer.get_status.remote())
                        self.log_preload(status)                    
                        if status["finish"]: 
                            self.finish = True
                            break                    

                    if self.time: self.timing.time("get_data")
                    if data == "FINISH" or self.finish: break
                    self.replay_ratio = data["replay_ratio"]

                    # start consume data
                    model_update = self.consume_data(data)
                    if model_update is not True:
                        raise RuntimeError(
                            "Model consume_data returned without completing an optimizer step"
                        )
                    del data                
                    gc.collect()
                else:
                    model_update = False

                # update shared buffer's weights
                if model_update:
                    assert_finite_tensors(
                        self.model_net.state_dict(),
                        context=(
                            "published ModelNet state at "
                            f"real_step={self.real_step}"
                        ),
                    )
                    self.param_buffer.set_data.remote(
                        "model_net", self.model_net.get_weights()
                    )
                if self.time: self.timing.time("update_weight")

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
                        
                        #self._logger.info(f"Model step checkpoint check: has_interval={has_interval}, interval={interval}, real_step={self.real_step}, current_milestone={current_milestone}, last_milestone={self.last_checkpoint_milestone}, milestone_reached={milestone_reached}")
                        
                        if milestone_reached:
                            self._logger.info(f"Triggering model step-based checkpoint at step {self.real_step} (milestone {current_milestone})")
                            self.save_checkpoint(force=True)
                            self.last_checkpoint_milestone = current_milestone
                if self.timing is not None:
                    self.timing.time("misc")

                if not model_update:
                    time.sleep(0.01)
                    status = ray.get(self.model_buffer.get_status.remote())
                    self.replay_ratio = status["replay_ratio"]
                    if status["finish"]: 
                        self.finish = True
                        break 

            self._logger.info("Terminating model-learning thread")
            self.save_checkpoint(force=True)
            self.model_buffer.set_finish.remote()
            self.signal_buffer.update_dict_item.remote(
                "self_play_signals", "halt", False
            )
            successful = True
            return True

        except Exception as e:
            self._logger.error(f"Exception detected in learn_model: {e}")
            self._logger.error(traceback.format_exc())
            raise
        finally:
            self.model_buffer.set_finish.remote()
            self.close(successful=successful)
        
    def update_real_step(self, data):
        new_psteps = data["processed_n"]
        if getattr(self, "voc_model_input_seal_runtime", False):
            if type(new_psteps) is not int or new_psteps < self.last_psteps:
                raise RuntimeError(
                    "schema-7 ModelBuffer processed_n must be a monotonic integer"
                )
            if self.real_step != self.last_psteps:
                raise RuntimeError(
                    "schema-7 model real_step/update cursor lost lockstep"
                )
        new_psteps = int(new_psteps)        
        self.real_step += new_psteps - self.last_psteps
        self.last_psteps = new_psteps

    def _initialize_gradient_clip_counters(self, checkpoint=None):
        checkpoint = {} if checkpoint is None else checkpoint
        for branch in ("m", "p"):
            count_key = f"model_grad_clip_count_{branch}"
            step_key = f"model_grad_step_count_{branch}"
            count = int(checkpoint.get(count_key, 0))
            # A checkpoint with only a cumulative count predates the
            # denominator field.  Using count as its lower-bound denominator
            # preserves a valid rate while ordinary old checkpoints start at 0.
            step_count = int(
                checkpoint.get(step_key, count if count_key in checkpoint else 0)
            )
            if count < 0 or step_count < count:
                raise ValueError(
                    f"invalid model gradient clip counters for {branch}: "
                    f"count={count}, steps={step_count}"
                )
            setattr(
                self,
                f"_model_grad_clip_count_{branch}",
                torch.tensor(count, dtype=torch.long, device=self.device),
            )
            setattr(
                self,
                f"_model_grad_step_count_{branch}",
                torch.tensor(step_count, dtype=torch.long, device=self.device),
            )

    def _record_gradient_clipping(self, branch, total_norm):
        if branch not in ("m", "p"):
            raise ValueError(f"unknown model gradient branch: {branch!r}")
        if not hasattr(self, f"_model_grad_clip_count_{branch}"):
            self._initialize_gradient_clip_counters()
        threshold = float(getattr(self.flags, "model_grad_norm_clipping", 0.0))
        if threshold > 0.0:
            clipped = total_norm.detach().reshape(()).to(self.device) > threshold
        else:
            clipped = torch.zeros((), dtype=torch.bool, device=self.device)
        getattr(self, f"_model_grad_clip_count_{branch}").add_(
            clipped.to(dtype=torch.long)
        )
        getattr(self, f"_model_grad_step_count_{branch}").add_(1)
        return clipped

    def _record_model_tensor_scale(self, prefix, value):
        if not hasattr(self, "_pending_model_observability"):
            self._pending_model_observability = _empty_model_observability()
        _record_tensor_scale(self._pending_model_observability, prefix, value)

    def _model_observability_log_stats(self):
        pending = getattr(
            self, "_pending_model_observability", _empty_model_observability()
        )
        return {
            "model/" + key: (
                float(value.detach().cpu().item()) if value is not None else None
            )
            for key, value in pending.items()
        }

    def _gradient_clip_log_stats(self, m_grad_clipped, p_grad_clipped):
        if not hasattr(self, "_model_grad_clip_count_m"):
            self._initialize_gradient_clip_counters()
        values = torch.stack(
            [
                m_grad_clipped.detach().to(self.device, dtype=torch.long),
                p_grad_clipped.detach().to(self.device, dtype=torch.long),
                self._model_grad_clip_count_m,
                self._model_grad_step_count_m,
                self._model_grad_clip_count_p,
                self._model_grad_step_count_p,
            ]
        ).cpu().tolist()
        m_current, p_current, m_count, m_steps, p_count, p_steps = (
            int(value) for value in values
        )
        return {
            "model/m_grad_clipped": m_current,
            "model/m_grad_clip_count": m_count,
            "model/m_grad_step_count": m_steps,
            "model/m_grad_clip_rate": m_count / m_steps if m_steps else 0.0,
            "model/p_grad_clipped": p_current,
            "model/p_grad_clip_count": p_count,
            "model/p_grad_step_count": p_steps,
            "model/p_grad_clip_rate": p_count / p_steps if p_steps else 0.0,
        }

    def _gradient_clip_checkpoint_state(self):
        if not hasattr(self, "_model_grad_clip_count_m"):
            self._initialize_gradient_clip_counters()
        return {
            key: int(getattr(self, "_" + key).detach().cpu().item())
            for key in (
                "model_grad_clip_count_m",
                "model_grad_step_count_m",
                "model_grad_clip_count_p",
                "model_grad_step_count_p",
            )
        }

    def consume_data(self, data, model_buffer=None):
        # model_buffer is only provided in non-parallel mode
        # which is required for updating the priorities of 
        # transition in the buffer
        self.n += 1
        self._pending_model_observability = _empty_model_observability()
        self.update_real_step(data)
        train_model_out, is_weights, idx = data["data"], data["weights"], data["idx"]
        TrainModelOut = namedtuple('TrainModelOut', train_model_out.keys())
        train_model_out = TrainModelOut(**train_model_out)
        # move the data to the process device to free memory
        train_model_out = util.tuple_map(
            train_model_out, lambda x: torch.tensor(x, device=self.device)
        )
        is_weights = torch.tensor(is_weights, dtype=torch.float32, device=self.device)
        del data

        assert_finite_tensors(
            train_model_out,
            context=f"model input at real_step={self.real_step}",
        )
        assert_finite_tensors(
            is_weights,
            context=f"model importance weights at real_step={self.real_step}",
        )
        if bool(torch.any(is_weights < 0).item()):
            raise ValueError(
                f"model importance weights must be non-negative at real_step={self.real_step}"
            )

        target = self.prepare_data(train_model_out)
        assert_finite_tensors(
            target,
            context=f"model target at real_step={self.real_step}",
        )
        if self.timing is not None:
            self.timing.time("convert_data")

        if self.flags.dual_net:
            torch.autograd.set_detect_anomaly(False)
            # compute losses for model_net
            with autocast(enabled=self.model_float16):
                losses_m, pred_xs, raw_pred_xs = self.compute_losses_m(
                    train_model_out, target, is_weights
                )
            assert_finite_tensors(
                losses_m,
                context=f"model SR losses at real_step={self.real_step}",
            )
            assert_finite_tensors(
                pred_xs,
                context=f"model predicted states at real_step={self.real_step}",
            )
            assert_finite_tensors(
                raw_pred_xs,
                context=(
                    "model raw predicted states before projection at "
                    f"real_step={self.real_step}"
                ),
            )
            if self.timing is not None:
                self.timing.time("compute_losses_m")

            # Backpropagate SR before constructing the VP graph.  Holding both
            # 20-step graphs at once approximately doubles peak memory in FP32.
            step_result_m = self.gradient_step(
                losses_m["total_loss_m"],
                self.optimizer_m,
                self.scheduler_m,
                self.scaler_m,
            )
            if not step_result_m.optimizer_stepped:
                raise FloatingPointError(
                    "AMP skipped the model SR optimizer step at "
                    f"real_step={self.real_step}; scale "
                    f"{step_result_m.amp_scale_before!r} -> "
                    f"{step_result_m.amp_scale_after!r}"
                )
            total_norm_m = step_result_m.total_norm
            m_grad_clipped = self._record_gradient_clipping("m", total_norm_m)
            if self.timing is not None:
                self.timing.time("gradient_step_m")
        else:
            losses_m = {}
            pred_xs = None
            raw_pred_xs = None
            step_result_m = GradientStepResult(
                torch.zeros(1, device=self.device), True, None, None
            )
            total_norm_m = step_result_m.total_norm
            m_grad_clipped = torch.zeros(
                (), dtype=torch.bool, device=self.device
            )

        with autocast(enabled=self.model_float16):
            losses_p, priorities = self.compute_losses_p(
                train_model_out, target, is_weights, pred_xs
            )
        assert_finite_tensors(
            losses_p,
            context=f"model VP losses at real_step={self.real_step}",
        )
        if self.flags.priority_alpha > 0:
            priorities = validate_priorities(
                priorities,
                context=f"model computed priority at real_step={self.real_step}",
                expected_shape=(int(is_weights.shape[0]),),
            )
        if self.timing is not None:
            self.timing.time("compute_losses_p")

        pred_xs_abs_max = (
            float(torch.max(torch.abs(pred_xs.detach())).item())
            if pred_xs is not None and pred_xs.numel() > 0
            else None
        )
        pred_xs_min = (
            float(torch.min(pred_xs.detach()).item())
            if pred_xs is not None and pred_xs.numel() > 0
            else None
        )
        pred_xs_max = (
            float(torch.max(pred_xs.detach()).item())
            if pred_xs is not None and pred_xs.numel() > 0
            else None
        )
        pred_raw_xs_abs_max = (
            float(torch.max(torch.abs(raw_pred_xs.detach())).item())
            if raw_pred_xs is not None and raw_pred_xs.numel() > 0
            else None
        )
        pred_raw_xs_oob_fraction = (
            float(
                torch.mean(
                    (
                        (raw_pred_xs.detach() < 0.0)
                        | (raw_pred_xs.detach() > 1.0)
                    ).float()
                ).item()
            )
            if raw_pred_xs is not None and raw_pred_xs.numel() > 0
            else None
        )
        priority_min = float(np.min(priorities)) if priorities is not None else None
        priority_max = float(np.max(priorities)) if priorities is not None else None

        step_result_p = self.gradient_step(
            losses_p["total_loss_p"], self.optimizer_p, self.scheduler_p, self.scaler_p
        )
        if not step_result_p.optimizer_stepped:
            raise FloatingPointError(
                "AMP skipped the model VP optimizer step at "
                f"real_step={self.real_step}; scale "
                f"{step_result_p.amp_scale_before!r} -> {step_result_p.amp_scale_after!r}"
            )
        total_norm_p = step_result_p.total_norm
        p_grad_clipped = self._record_gradient_clipping("p", total_norm_p)
        if self.timing is not None:
            self.timing.time("gradient_step_p")
        assert_finite_tensors(
            self.model_net.state_dict(),
            context=f"ModelNet state after real_step={self.real_step}",
        )
        if self.flags.priority_alpha > 0:
            if model_buffer is None:
                self.model_buffer.update_priority.remote(idx, priorities)
            else:
                model_buffer.update_priority(idx, priorities)
        self.step += self.numel_per_step
        if self.timing is not None:
            self.timing.time("update_priority")
        losses = losses_m
        losses.update(losses_p)
        # print statistics
        if self.timer() - self.start_time > 5:
            self.sps_buffer[self.sps_buffer_n] = (self.step, self.timer())
            self.sps_buffer_n = (self.sps_buffer_n + 1) % len(self.sps_buffer)
            sps = (
                self.sps_buffer[self.sps_buffer_n - 1][0]
                - self.sps_buffer[self.sps_buffer_n][0]
            ) / (
                self.sps_buffer[self.sps_buffer_n - 1][1]
                - self.sps_buffer[self.sps_buffer_n][1]
            )
            tot_sps = (self.step - self.sps_start_step) / (
                self.timer() - self.sps_start_time
            )
            observability_stats = self._model_observability_log_stats()
            gradient_clip_stats = self._gradient_clip_log_stats(
                m_grad_clipped, p_grad_clipped
            )
            print_str = (
                "[%s] Steps %i (%i[%.1f]) @ %.1f SPS (%.1f). norm_m %.2f norm_p %.2f"
                % (
                    self.flags.xpid,
                    self.real_step,
                    self.step,
                    self.step_per_transition(),
                    sps,
                    tot_sps,
                    total_norm_m.item(),
                    total_norm_p.item(),
                )
            )
            print_stats = [
                "total_loss_m",
                "total_loss_p",
                "img_loss",
                "fea_loss",
                "state_range_loss",
                "noise_loss",
                "done_loss",
                "reg_loss",
            ]
            for k in print_stats:
                if k in losses and losses[k] is not None:
                    print_str += " %s %.6f" % (
                        k,
                        losses[k].item() / self.numel_per_step,
                    )
            if pred_xs_abs_max is not None:
                print_str += " pred_xs_abs_max %.6f" % pred_xs_abs_max
                print_str += " pred_xs_min %.6f pred_xs_max %.6f" % (
                    pred_xs_min,
                    pred_xs_max,
                )
            if pred_raw_xs_abs_max is not None:
                print_str += (
                    " pred_raw_xs_abs_max %.6f pred_raw_xs_oob_fraction %.6f"
                    % (pred_raw_xs_abs_max, pred_raw_xs_oob_fraction)
                )
            if priority_min is not None:
                print_str += " priority_min %.6f priority_max %.6f" % (
                    priority_min,
                    priority_max,
                )
            for key, value in observability_stats.items():
                if value is not None:
                    print_str += " %s %.6f" % (key.removeprefix("model/"), value)
            print_str += (
                " m_grad_clipped %d m_grad_clip %d/%d"
                " p_grad_clipped %d p_grad_clip %d/%d"
                % (
                    gradient_clip_stats["model/m_grad_clipped"],
                    gradient_clip_stats["model/m_grad_clip_count"],
                    gradient_clip_stats["model/m_grad_step_count"],
                    gradient_clip_stats["model/p_grad_clipped"],
                    gradient_clip_stats["model/p_grad_clip_count"],
                    gradient_clip_stats["model/p_grad_step_count"],
                )
            )
            if step_result_m.amp_scale_after is not None:
                print_str += " amp_scale_m %.1f" % step_result_m.amp_scale_after
            if step_result_p.amp_scale_after is not None:
                print_str += " amp_scale_p %.1f" % step_result_p.amp_scale_after
            self._logger.info(print_str)
            self.start_time = self.timer()

            # write to log file
            stats = {
                "step": self.step,
                "real_step": self.real_step,
                "model/total_norm_m": total_norm_m.item(),
                "model/total_norm_p": total_norm_p.item(),
                "model/pred_xs_abs_max": pred_xs_abs_max,
                "model/pred_xs_min": pred_xs_min,
                "model/pred_xs_max": pred_xs_max,
                "model/pred_raw_xs_abs_max": pred_raw_xs_abs_max,
                "model/pred_raw_xs_oob_fraction": pred_raw_xs_oob_fraction,
                "model/priority_min": priority_min,
                "model/priority_max": priority_max,
                "model/optimizer_stepped_m": int(step_result_m.optimizer_stepped),
                "model/optimizer_stepped_p": int(step_result_p.optimizer_stepped),
                "model/amp_scale_before_m": step_result_m.amp_scale_before,
                "model/amp_scale_after_m": step_result_m.amp_scale_after,
                "model/amp_scale_before_p": step_result_p.amp_scale_before,
                "model/amp_scale_after_p": step_result_p.amp_scale_after,
                "model/model_float16": int(self.model_float16),
                "model/learning_rate": self.optimizer_p.param_groups[0]["lr"],
                "model/schedule_progress": util.schedule_progress(
                    self.flags, self.real_step
                ),
                "model/priority_beta": self.compute_beta(),
            }
            stats.update(observability_stats)
            stats.update(gradient_clip_stats)
            for k in losses.keys():
                stats["model/" + k] = (
                    losses[k].item() / self.numel_per_step
                    if k in losses and losses[k] is not None
                    else None
                )

            self.plogger.log(stats)
            if self.timing is not None:
                print(self.timing.summary())
        if int(time.strftime("%M")) // 10 != self.ckp_start_time:
            self.save_checkpoint()
            self.ckp_start_time = int(time.strftime("%M")) // 10
        if self.timing is not None:
            self.timing.time("misc")
        return True

    def compute_rs_loss(self, target, rs, r_enc_logits, rv_tran, is_weights):
        k, b = self.flags.model_unroll_len, target["rewards"].shape[1]
        done_mask = target["done_mask"]
        if self.flags.model_enc_type == 0:
            rs_loss = (rs - target["rewards"]) ** 2
            rs_loss = torch.sum(rs_loss, dim=-1)
        else:
            target_rs_enc_v = rv_tran.encode(target["rewards"])
            rs_loss = 0.
            for i in range(self.reward_n):
                target_rs_enc_v = rv_tran.encode(target["rewards"][:, :, i])
                rs_loss = rs_loss + torch.nn.CrossEntropyLoss(reduction="none")(
                    input=torch.flatten(r_enc_logits[:, :, i], 0, 1),
                    target=torch.flatten(target_rs_enc_v, 0, 1),
                )
            rs_loss = rs_loss.view(k, b)
        rs_loss = rs_loss * done_mask[:-1]
        rs_loss = torch.sum(rs_loss, dim=0)
        rs_loss = rs_loss * is_weights
        rs_loss = torch.sum(rs_loss)
        return rs_loss

    def compute_done_loss(self, target, pred_done_logits, is_weights):
        if self.flags.model_done_loss_cost > 0.0:
            done_loss = torch.nn.BCEWithLogitsLoss(reduction="none")(
                pred_done_logits, target["dones"]
            )
            done_loss = done_loss * (~target["trun_done"]).float()[:-1]
            done_loss = torch.sum(done_loss, dim=0)
            done_loss = done_loss * is_weights
            done_loss = torch.sum(done_loss)
        else:
            done_loss = None
        return done_loss
    
    def compute_state_loss(self, tar, pred, mask, is_weights, cos=False):        
        if not cos:
            diff = tar - pred
            if not self.model_net.oned_input:                        
                state_loss = torch.mean(torch.square(diff), dim=(2, 3, 4))
            else:
                state_loss = torch.mean(torch.square(diff), dim=2)
        else:
            tar_flat = torch.flatten(tar, 2)
            pred_flat = torch.flatten(pred, 2)
            cos_sim = F.cosine_similarity(tar_flat, pred_flat, dim=2, eps=1e-08)
            state_loss = 1 - cos_sim
        state_loss = state_loss * mask
        state_loss = torch.sum(state_loss, dim=0)
        state_loss = state_loss * is_weights
        state_loss = torch.sum(state_loss)
        return state_loss

    def compute_state_range_loss(self, raw_pred, mask, is_weights):
        """Penalize only violations of the normalized observation interval.

        The raw decoder emits the newest frame rather than the complete frame
        stack.  Smooth-L1 is averaged over feature/pixel/channel dimensions,
        then aggregated over valid rollout edges and PER importance weights in
        the same way as the other state losses.  Detaching the projected target
        makes gradients point back toward the nearest interval boundary.
        """

        interval_target = torch.clamp(raw_pred.detach(), 0.0, 1.0)
        range_error = F.smooth_l1_loss(
            raw_pred, interval_target, reduction="none", beta=1.0
        )
        reduce_dims = tuple(range(2, range_error.ndim))
        if reduce_dims:
            range_error = torch.mean(range_error, dim=reduce_dims)
        range_error = range_error * mask
        range_error = torch.sum(range_error, dim=0)
        range_error = range_error * is_weights
        return torch.sum(range_error)

    def compute_losses_m(self, train_model_out, target, is_weights):
        k, b = self.flags.model_unroll_len, train_model_out.real_state.shape[1]
        initial_per_state = {sk: getattr(train_model_out, sk)[0] for sk in train_model_out._fields if sk.startswith("per")}
        if self.flags.model_mem_unroll_len > 0:
            past_env_state_norm = self.model_net.normalize(train_model_out.initial_per_state["past_real_state"])
            past_done = train_model_out.initial_per_state["past_done"]
            past_action = train_model_out.initial_per_state["past_action"]
            past_action = util.encode_action(past_action, self.model_net.action_space, one_hot=False)
            _, per_state = self.model_net.sr_net.encoder(past_env_state_norm, past_done, past_action, initial_per_state, flatten=True)

            #dbg_per_state = {sk: sv[-1] for sk, sv in train_model_out.initial_per_state.items() if sk.startswith("per")}
            #for sk in per_state.keys(): print(sk, torch.sum(torch.abs(per_state[sk] - dbg_per_state[sk])))
        else:
            per_state = initial_per_state

        env_state_norm = self.model_net.normalize(train_model_out.real_state[0])
        out = self.model_net.sr_net.forward(
            env_state_norm=env_state_norm,
            done=train_model_out.done[0],
            actions=train_model_out.action[: k + 1],
            state=per_state,
            future_env_state_norm=self.model_net.normalize(train_model_out.real_state[1:k+1]) if self.flags.noise_enable else None,
            check_raw_finite=True,
        )
        rs_loss = self.compute_rs_loss(
            target,
            out.rs,
            out.r_enc_logits,
            self.model_net.sr_net.rv_tran,
            is_weights,
        )
        done_loss = self.compute_done_loss(target, out.done_logits, is_weights)
        target_env_state_norm = self.model_net.normalize(target["env_states"])
        action = util.encode_action(train_model_out.action[1 : k + 1], self.model_net.action_space, one_hot=False)        
        if not self.flags.fea_loss_inf_bn:
            bn_stat = util.clone_bn_running_stats(self.model_net.vp_net)
        else:
            self.model_net.vp_net.train(False)
        with torch.no_grad():  
            target_xs = self.model_net.vp_net.encoder.forward_pre_mem(
                    target_env_state_norm, action, flatten=True, end_depth=self.flags.model_decoder_depth
            )
        if self.flags.model_img_loss_cost > 0.:
            img_loss = self.compute_state_loss(target_xs, out.xs, target["done_mask"][1:], is_weights, self.flags.img_fea_cos)
        else:
            img_loss = None
        if self.flags.model_fea_loss_cost > 0.:
            with torch.no_grad():                
                target_enc = self.model_net.vp_net.encoder.forward_pre_mem(
                    target_xs, action, flatten=True, depth=self.flags.model_decoder_depth
                )
            pred_enc = self.model_net.vp_net.encoder.forward_pre_mem(out.xs, action, flatten=True, depth=self.flags.model_decoder_depth)
            fea_loss = self.compute_state_loss(target_enc, pred_enc, target["done_mask"][1:], is_weights, self.flags.img_fea_cos)
        else:
            fea_loss = None        
        if self.flags.model_state_projection == "clamp":
            state_range_loss = self.compute_state_range_loss(
                out.raw_xs,
                target["done_mask"][1:],
                is_weights,
            )
        else:
            state_range_loss = None
        if not self.flags.fea_loss_inf_bn:
            util.restore_bn_running_stats(self.model_net.vp_net, bn_stat)
        else:
            self.model_net.vp_net.train(True)

        if out.noise_loss is not None:
            noise_loss = out.noise_loss
            noise_loss = noise_loss * target["done_mask"][1:]
            noise_loss = torch.sum(noise_loss, dim=0)
            noise_loss = noise_loss * is_weights
            noise_loss = torch.sum(noise_loss)
        else:
            noise_loss = None

        total_loss = self.flags.model_rs_loss_cost * rs_loss
        if self.flags.model_img_loss_cost > 0.0:
            total_loss = total_loss + self.flags.model_img_loss_cost * img_loss
        if self.flags.model_fea_loss_cost > 0.0:
            total_loss = total_loss + self.flags.model_fea_loss_cost * fea_loss
        if self.flags.model_done_loss_cost > 0.0:
            total_loss = total_loss + self.flags.model_done_loss_cost * done_loss
        if self.flags.model_noise_loss_cost > 0.:
            total_loss = total_loss + self.flags.model_noise_loss_cost * noise_loss
        if self.flags.model_state_range_loss_cost > 0.0:
            total_loss = (
                total_loss
                + self.flags.model_state_range_loss_cost * state_range_loss
            )

        predicted_sr_hs = out.hs[1:] if out.hs.shape[0] > 1 else out.hs
        self._record_model_tensor_scale("pred_sr_hs", predicted_sr_hs)
        self._record_model_tensor_scale(
            "pred_reward_head",
            out.r_enc_logits if out.r_enc_logits is not None else out.rs,
        )

        return {
            "rs_loss": rs_loss,
            "done_loss": done_loss,
            "img_loss": img_loss,
            "fea_loss": fea_loss,
            "state_range_loss": state_range_loss,
            "noise_loss": noise_loss,
            "total_loss_m": total_loss,
        }, out.xs.detach(), out.raw_xs.detach()

    def compute_losses_p(self, train_model_out, target, is_weights, pred_xs):
        k, b = self.flags.model_unroll_len, train_model_out.real_state.shape[1]
        vp_net = self.model_net.vp_net
        initial_per_state = {sk: getattr(train_model_out, sk)[0] for sk in train_model_out._fields if sk.startswith("per")}

        if self.flags.model_mem_unroll_len > 0:
            past_env_state_norm = self.model_net.normalize(train_model_out.initial_per_state["past_real_state"])
            past_done = train_model_out.initial_per_state["past_done"]
            past_action = train_model_out.initial_per_state["past_action"]
            past_action = util.encode_action(past_action, self.model_net.action_space, one_hot=False)
            _, per_state = vp_net.encoder(past_env_state_norm, past_done, past_action, initial_per_state, flatten=True)
        else:
            per_state = initial_per_state      
        
        if self.perfect_model:            
            env_state_norm = self.model_net.normalize(train_model_out.real_state)
            out = vp_net.forward(
                env_state_norm=env_state_norm[:k+1].view(((k+1) * b,) + env_state_norm.shape[2:]),
                x0=None,
                xs=None,
                done=train_model_out.done[:k+1].view(1, (k+1) * b,),
                actions=train_model_out.action[:k+1].view(1, (k+1) * b, -1),
                state={},
            )
            vs = out.vs.view(k+1, b, self.reward_n)
            v_enc_logits = util.safe_view(out.v_enc_logits, (k+1, b, self.reward_n, -1))
            policy = out.policy.view((k+1, b) + out.policy.shape[2:])
        else:
            env_state_norm = self.model_net.normalize(train_model_out.real_state[0])
            out = vp_net.forward(
                env_state_norm=env_state_norm,
                x0=None,
                xs=pred_xs, 
                done=train_model_out.done[0],
                actions=train_model_out.action[: k + 1],  # a_-1, ..., a_k-1                
                state=per_state,
            )
            vs = out.vs.view(k+1, b, self.reward_n)
            v_enc_logits = util.safe_view(out.v_enc_logits, (k+1, b, self.reward_n, -1))
            policy = out.policy

        done_mask = target["done_mask"]
        if vp_net.predict_rd:
            rs_loss = self.compute_rs_loss(
                target,
                out.rs,
                out.r_enc_logits,
                vp_net.rv_tran,
                is_weights,
            )
            done_loss = self.compute_done_loss(target, out.done_logits, is_weights)

        # compute vs loss
        vs_loss = self.model_net.compute_vs_loss(
            vs=vs, 
            v_enc_logits=v_enc_logits, 
            target_vs=target["vs"],
        )
        vs_loss = vs_loss * done_mask
        vs_loss = torch.sum(vs_loss, dim=0)
        vs_loss = vs_loss * is_weights
        vs_loss = torch.sum(vs_loss)

        # compute policy loss
        if self.flags.require_prob:
            target_policy = target["action_probs"].detach()
        else:
            if self.model_net.discrete_action:
                target_policy = F.one_hot(
                    target["actions"], self.model_net.num_actions).detach().float()
            else:
                target_policy = target["actions"].detach().float()

        policy_loss = compute_cross_entropy_loss(
            policy, 
            target_policy, 
            self.model_net.discrete_action,
            self.flags.require_prob,
            is_weights, 
            mask=done_mask, 
        )

        # compute reg loss
        if self.flags.model_reg_loss_cost > 0.0:
            if self.perfect_model:
                pred_zs = out.pred_zs.view(k, b, -1)
            else:
                pred_zs = out.pred_zs.view(k + 1, b, -1)
            reg_loss = torch.mean(torch.square(pred_zs), dim=-1)
            if not self.perfect_model:
                reg_loss = reg_loss * done_mask
            reg_loss = torch.sum(reg_loss)
        else:
            reg_loss = None

        losses = {
            "vs_loss": vs_loss,
            "policy_loss": policy_loss,
            "reg_loss": reg_loss,
        }
        total_loss = (
            self.flags.model_vs_loss_cost * vs_loss
            + self.flags.model_policy_loss_cost * policy_loss
        )
        if self.model_net.vp_net.predict_rd:
            total_loss = total_loss + self.flags.model_rs_loss_cost * rs_loss
            losses["rs_loss"] = rs_loss
            if self.flags.model_done_loss_cost > 0.0:
                total_loss = total_loss + self.flags.model_done_loss_cost * done_loss
                losses["done_loss"] = done_loss
        if self.flags.model_reg_loss_cost > 0.0:
            total_loss = total_loss + self.flags.model_reg_loss_cost * reg_loss

        losses["total_loss_p"] = total_loss

        # compute priorities
        if self.flags.priority_alpha > 0.0:
            priorities = torch.absolute(vs[0, :, 0] - target["vs"][0, :, 0]) # vs error on first time step wrt primiary reward
            priorities = priorities.detach().cpu().numpy()
        else:
            priorities = None

        self._record_model_tensor_scale("pred_vp_hs", out.hs)
        self._record_model_tensor_scale("pred_policy_logits", policy)
        self._record_model_tensor_scale(
            "pred_value_head", v_enc_logits if v_enc_logits is not None else vs
        )
        if self._pending_model_observability["pred_reward_head_abs_max"] is None:
            self._record_model_tensor_scale(
                "pred_reward_head",
                out.r_enc_logits if out.r_enc_logits is not None else out.rs,
            )

        return losses, priorities

    def prepare_data(self, train_model_out):
        k, b = self.flags.model_unroll_len, train_model_out.real_state.shape[1]
        ret_n = self.flags.model_return_n
        target_env_states = train_model_out.real_state
        target_rewards = train_model_out.reward[1 : k + 1]  # true reward r_1, r_2, ..., r_k
        target_action_probs = train_model_out.action_prob[1 : k + 2]  # true logits l_0, l_1, ..., l_k-1        
        target_actions = train_model_out.action[1 : k + 2]  # true actions l_0, l_1, ..., l_k-1

        reward = train_model_out.reward
        done = train_model_out.done | train_model_out.truncated_done    
        baseline = train_model_out.baseline[:, :, :self.reward_n]

        if not self.flags.vp_fix_bootstrap:
            target_vs = train_model_out.baseline[ret_n + 1: ret_n + 2 + k]  # baseline ranges from v_k, v_k+1, ... v_2k
            for t in range(ret_n, 0, -1):
                target_vs = (
                    target_vs
                    * self.flags.discounting
                    * (~done[t : k + t + 1]).float().unsqueeze(-1)
                    + train_model_out.reward[t : k + t + 1]
                )
                t_done = train_model_out.truncated_done[t : k + t + 1]
                if torch.any(t_done):
                    target_vs[t_done] = train_model_out.baseline[t : k + t + 1][t_done]

        else:
            target_v = train_model_out.baseline[k + 1] # v is in the form of v_-1, v_0, .., v_k; this target_v is v_k
            target_vs = [target_v]
            for t in range(k, 0, -1):
                target_v = train_model_out.reward[t] + self.flags.discounting * target_v * (~done[t]).float().unsqueeze(-1)
                t_done = train_model_out.truncated_done[t]
                if torch.any(t_done):
                    target_v[t_done] = train_model_out.baseline[t][t_done]
                target_vs.append(target_v)
            
            target_vs.reverse()
            target_vs = torch.stack(target_vs)

        # if done on step j, r_j, v_j-1, a_j-1 has the last valid loss
        # we set all target r_j+1, v_j, a_j to 0, 0, and last a_{j+1}

        if not self.perfect_model:
            trun_done = torch.zeros(k + 1, b, dtype=torch.bool, device=self.device)
            true_done = torch.zeros(k + 1, b, dtype=torch.bool, device=self.device)
            # done_mask stores accumulated done: True, adone_1, adone_2, ..., adone_k
            for t in range(1, k + 1):
                trun_done[t] = torch.logical_or(
                    trun_done[t - 1], train_model_out.truncated_done[t]
                )
                true_done[t] = torch.logical_or(
                    true_done[t - 1], train_model_out.done[t]
                )
                if not self.flags.model_done_loss_cost > 0.0:
                    target_env_states[t, true_done[t]] = 0
                if t < k:
                    target_rewards[t, true_done[t]] = 0.0
                target_action_probs[t, true_done[t]] = target_action_probs[t - 1, true_done[t]]
                target_actions[t, true_done[t]] = target_actions[t - 1, true_done[t]]
                target_vs[t, true_done[t]] = 0.0
            if self.flags.model_done_loss_cost > 0.0:
                done_mask = (~torch.logical_or(trun_done, true_done)).float()
                target_done = torch.logical_and(~trun_done, true_done).float()[1:]
            else:
                done_mask = (~trun_done).float()
                target_done = None
        else:
            done_mask = torch.ones(k + 1, b, device=self.device)
            trun_done = None
            target_done = None

        return {
            "env_states": target_env_states[1 : k + 1],
            "rewards": target_rewards,            
            "actions": target_actions,
            "action_probs": target_action_probs,
            "vs": target_vs,
            "dones": target_done,
            "trun_done": trun_done,
            "done_mask": done_mask,
        }
    
    def gradient_step(self, loss, optimizer, scheduler, scaler=None):
        # gradient descent on loss
        if self.flags.model_optimizer == "sgd":
            loss = loss / self.numel_per_step
        assert_finite_tensors(loss, context="model optimizer loss")

        optimizer.zero_grad()
        scaler_enabled = scaler is not None and (
            not hasattr(scaler, "is_enabled") or scaler.is_enabled()
        )
        amp_scale_before = None
        amp_scale_after = None
        if scaler_enabled:
            amp_scale_before = float(scaler.get_scale())
            if not np.isfinite(amp_scale_before) or amp_scale_before <= 0:
                raise FloatingPointError(
                    "model AMP scale must be finite and positive before backward, "
                    f"got {amp_scale_before!r}"
                )
            scaler.scale(loss).backward()
        else:
            loss.backward()
                
        if scaler_enabled:
            scaler.unscale_(optimizer)
        
        optimize_params = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        if self.flags.model_grad_norm_clipping > 0:
            total_norm = torch.nn.utils.clip_grad_norm_(
                optimize_params, self.flags.model_grad_norm_clipping
            )
        else:
            total_norm = util.compute_grad_norm(optimize_params)

        if not bool(torch.isfinite(total_norm).item()):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(
                f"model gradient norm is non-finite: {total_norm.detach().cpu().item()!r}"
            )

        found_inf = None
        if scaler_enabled and hasattr(scaler, "_found_inf_per_device"):
            found_inf_by_device = scaler._found_inf_per_device(optimizer)
            found_inf = sum(
                float(value.detach().cpu().item())
                for value in found_inf_by_device.values()
            )

        if scaler_enabled:
            scaler.step(optimizer)
            scaler.update()
            amp_scale_after = float(scaler.get_scale())
            scale_is_valid = np.isfinite(amp_scale_after) and amp_scale_after > 0
            if found_inf is not None:
                optimizer_stepped = scale_is_valid and found_inf == 0.0
            else:
                # PyTorch backs the scale off exactly when scaler.step skipped
                # the optimizer. This fallback also supports small fake scalers
                # used by focused tests.
                optimizer_stepped = (
                    scale_is_valid
                    and amp_scale_after >= amp_scale_before
                )
        else:
            optimizer.step()
            optimizer_stepped = True

        if optimizer_stepped:
            assert_optimizer_parameters_finite(
                optimizer,
                context=f"model optimizer parameters after real_step={self.real_step}",
            )
            scheduler.last_epoch = (
                max(self.real_step - 1, 0)
            )  # scheduler does not support setting epoch directly
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        return GradientStepResult(
            total_norm,
            bool(optimizer_stepped),
            amp_scale_before,
            amp_scale_after,
        )

    def step_per_transition(self):
        return self.step / (self.real_step - self.flags.model_warm_up_n + 1)

    def refresh_model(self):
        while True:
            weights = ray.get(
                self.param_buffer.get_data.remote("model_net")
            )  
            if weights is not None:
                self.model_net.set_weights(weights)
                del weights
                break                
            time.sleep(0.1)  

    def refresh_actor(self):
        while True:
            weights = ray.get(
                self.actor_param_buffer.get_data.remote("actor_net")
            )  
            if weights is not None:
                self.actor_net.set_weights(weights)
                del weights
                break                
            time.sleep(0.1)  

    def _validate_schema7_model_input_checkpoint_evidence(
        self, checkpoint, *, require_terminal
    ):
        if type(require_terminal) is not bool:
            raise RuntimeError("schema-7 checkpoint terminal mode must be boolean")
        expected_fields = set(util._SCHEMA7_MODEL_INPUT_SEAL_EVIDENCE_FIELDS)
        actual_fields = {
            name
            for name in checkpoint
            if type(name) is str and name.startswith("voc_model_")
        }
        if actual_fields != expected_fields:
            raise RuntimeError(
                "schema-7 checkpoint has an inexact model-input evidence surface"
            )
        evidence = {name: checkpoint[name] for name in expected_fields}
        if type(evidence["voc_model_input_sealed"]) is not bool:
            raise RuntimeError(
                "schema-7 checkpoint seal evidence must be Python bool"
            )
        for name in expected_fields - {"voc_model_input_sealed"}:
            if type(evidence[name]) is not int:
                raise RuntimeError(
                    f"schema-7 checkpoint {name} must be a Python integer"
                )
        if evidence["voc_model_input_seal_schema_version"] != 1:
            raise RuntimeError("schema-7 checkpoint has the wrong seal schema")

        if not evidence["voc_model_input_sealed"]:
            preterminal = {
                "voc_model_input_seal_schema_version": 1,
                "voc_model_input_sealed": False,
                "voc_model_input_seal_count": 0,
                "voc_model_terminal_processed_n": -1,
                "voc_model_terminal_drain_update_count": 0,
                "voc_model_terminal_drain_pre_real_step": -1,
                "voc_model_terminal_drain_pre_grad_step_count_m": -1,
                "voc_model_terminal_drain_pre_grad_step_count_p": -1,
                "voc_model_input_late_write_count": 0,
                "voc_model_input_abort_count": 0,
            }
            if require_terminal or evidence != preterminal:
                raise RuntimeError(
                    "schema-7 preterminal checkpoint evidence is malformed"
                )
            return evidence

        if not require_terminal:
            raise RuntimeError(
                "schema-7 sealed evidence requires terminal durable save mode"
            )

        for name in (
            "real_step",
            "model_grad_step_count_m",
            "model_grad_step_count_p",
        ):
            if type(checkpoint.get(name)) is not int:
                raise RuntimeError(
                    f"schema-7 terminal checkpoint {name} must be a Python integer"
                )
        total_steps = getattr(self.flags, "total_steps", None)
        if type(total_steps) is not int or total_steps <= 0:
            raise RuntimeError("schema-7 terminal checkpoint total_steps is invalid")
        real_step = checkpoint["real_step"]
        terminal_processed_n = evidence["voc_model_terminal_processed_n"]
        drain_count = evidence["voc_model_terminal_drain_update_count"]
        pre_real_step = evidence["voc_model_terminal_drain_pre_real_step"]
        pre_m = evidence["voc_model_terminal_drain_pre_grad_step_count_m"]
        pre_p = evidence["voc_model_terminal_drain_pre_grad_step_count_p"]
        final_m = checkpoint["model_grad_step_count_m"]
        final_p = checkpoint["model_grad_step_count_p"]
        if (
            evidence["voc_model_input_seal_count"] != 1
            or terminal_processed_n != real_step
            or real_step < total_steps
            or drain_count not in (0, 1)
            or not 0 <= pre_real_step <= real_step
            or pre_m < 0
            or pre_p < 0
            or final_m <= 0
            or final_p <= 0
            or final_m != final_p
            or final_m != pre_m + drain_count
            or final_p != pre_p + drain_count
            or evidence["voc_model_input_late_write_count"] != 0
            or evidence["voc_model_input_abort_count"] != 0
        ):
            raise RuntimeError(
                "schema-7 terminal checkpoint evidence relations are invalid"
            )
        if (
            drain_count == 0 and pre_real_step != real_step
        ) or (
            drain_count == 1 and pre_real_step >= real_step
        ):
            raise RuntimeError(
                "schema-7 terminal checkpoint drain branch is invalid"
            )
        return evidence

    def _schema7_model_input_checkpoint_evidence(self, *, require_terminal):
        evidence = {
            "voc_model_input_seal_schema_version": (
                self.voc_model_input_seal_schema_version
            ),
            "voc_model_input_sealed": self.voc_model_input_sealed,
            "voc_model_input_seal_count": self.voc_model_input_seal_count,
            "voc_model_terminal_processed_n": (
                self.voc_model_terminal_processed_n
            ),
            "voc_model_terminal_drain_update_count": (
                self.voc_model_terminal_drain_update_count
            ),
            "voc_model_terminal_drain_pre_real_step": (
                self.voc_model_terminal_drain_pre_real_step
            ),
            "voc_model_terminal_drain_pre_grad_step_count_m": (
                self.voc_model_terminal_drain_pre_grad_step_count_m
            ),
            "voc_model_terminal_drain_pre_grad_step_count_p": (
                self.voc_model_terminal_drain_pre_grad_step_count_p
            ),
            "voc_model_input_late_write_count": (
                self.voc_model_input_late_write_count
            ),
            "voc_model_input_abort_count": self.voc_model_input_abort_count,
        }
        checkpoint = {
            **evidence,
            "real_step": self.real_step,
            **self._gradient_clip_checkpoint_state(),
        }
        self._validate_schema7_model_input_checkpoint_evidence(
            checkpoint, require_terminal=require_terminal
        )
        return evidence

    @staticmethod
    def _durable_torch_save(payload, path):
        temporary_path = path + ".tmp"
        with open(temporary_path, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        directory = os.path.dirname(os.path.abspath(path))
        directory_fd = os.open(
            directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def save_checkpoint(self, force=False, terminal=False):
        if type(terminal) is not bool:
            raise ValueError("model checkpoint terminal mode must be Python bool")
        if terminal and not getattr(self, "voc_model_input_seal_runtime", False):
            raise RuntimeError("terminal model-input evidence is inactive")
        if terminal and force is not True:
            raise RuntimeError("schema-7 terminal checkpoint must be force-saved")
        if (
            getattr(self, "voc_model_input_seal_runtime", False)
            and getattr(self, "_schema7_terminal_drain_active", False)
            and not terminal
        ):
            self._logger.info(
                "Deferring periodic checkpoint during schema-7 terminal drain"
            )
            return False
        self._logger.info("Saving model checkpoint to %s" % self.ckp_path)
        assert_finite_tensors(
            self.model_net.state_dict(),
            context=f"checkpoint ModelNet state at real_step={self.real_step}",
        )
        d = {
            "step": self.step,
            "real_step": self.real_step,
            "model_net_optimizer_p_state_dict": self.optimizer_p.state_dict(),
            "model_net_scheduler_p_state_dict": self.scheduler_p.state_dict(),
            "model_net_state_dict": self.model_net.state_dict(),
            "flags": vars(self.flags),
        }
        d.update(self._gradient_clip_checkpoint_state())
        if getattr(self, "voc_model_input_seal_runtime", False):
            d.update(self._schema7_model_input_checkpoint_evidence(
                require_terminal=terminal
            ))
        if self.scaler_p is not None:
            d["model_scaler_p_state_dict"] = self.scaler_p.state_dict()
        if self.flags.dual_net:
            d.update(
                {
                    "model_net_optimizer_m_state_dict": self.optimizer_m.state_dict(),
                    "model_net_scheduler_m_state_dict": self.scheduler_m.state_dict(),
                }
            )
            if self.scaler_m is not None:
                d["model_scaler_m_state_dict"] = self.scaler_m.state_dict()
        try:
            # Save regular checkpoint
            if terminal:
                self._durable_torch_save(d, self.ckp_path)
            else:
                torch.save(d, self.ckp_path + ".tmp")
                os.replace(self.ckp_path + ".tmp", self.ckp_path)
            
            # Save step-specific checkpoint if forced or at checkpoint interval
            if force or (hasattr(self.flags, 'checkpoint_interval') and 
                         self.flags.checkpoint_interval > 0 and 
                         self.real_step % self.flags.checkpoint_interval == 0):
                checkpoint_path = f"{self.ckp_path}_step_{self.real_step}"
                if terminal:
                    self._durable_torch_save(d, checkpoint_path)
                else:
                    torch.save(d, checkpoint_path + ".tmp")
                    os.replace(checkpoint_path + ".tmp", checkpoint_path)
                self._logger.info(f"Saved model checkpoint at step {self.real_step} to {checkpoint_path}")
        except Exception as e:       
            self._logger.error(f"Error saving model checkpoint: {e}")
            raise

    def load_checkpoint(self, ckp_path: str):
        train_checkpoint = torch.load(ckp_path, torch.device("cpu"))
        self.step = train_checkpoint["step"]
        self.real_step = train_checkpoint["real_step"]
        self._initialize_gradient_clip_counters(train_checkpoint)
        if self.flags.dual_net:
            util.load_optimizer(self.optimizer_m, train_checkpoint["model_net_optimizer_m_state_dict"])
            util.load_scheduler(self.scheduler_m, train_checkpoint["model_net_scheduler_m_state_dict"])
        util.load_optimizer(self.optimizer_p, train_checkpoint["model_net_optimizer_p_state_dict"])
        util.load_scheduler(self.scheduler_p, train_checkpoint["model_net_scheduler_p_state_dict"])
        if self.scaler_p is not None:
            scaler_state = train_checkpoint.get("model_scaler_p_state_dict")
            if scaler_state is None:
                self._logger.warning(
                    "model checkpoint has no VP AMP scaler state; resume is not bitwise exact"
                )
            else:
                self.scaler_p.load_state_dict(scaler_state)
        if self.flags.dual_net and self.scaler_m is not None:
            scaler_state = train_checkpoint.get("model_scaler_m_state_dict")
            if scaler_state is None:
                self._logger.warning(
                    "model checkpoint has no SR AMP scaler state; resume is not bitwise exact"
                )
            else:
                self.scaler_m.load_state_dict(scaler_state)
        self.model_net.set_weights(train_checkpoint["model_net_state_dict"])
        assert_finite_tensors(
            self.model_net.state_dict(),
            context=f"loaded ModelNet state from {ckp_path}",
        )
        self._logger.info("Loaded model checkpoint from %s" % ckp_path)

    def close(self, successful=True):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        self.plogger.close(successful=bool(successful))

@ray.remote
class ModelLearner(SModelLearner):
    pass
