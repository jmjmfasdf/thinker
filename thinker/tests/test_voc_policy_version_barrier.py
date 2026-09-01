from collections import OrderedDict, namedtuple
import copy
import io
import json
from pathlib import Path
import random
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

import train as train_driver
from thinker import learn_actor, util
from thinker.actor_net import compute_voc_gate_distribution
from thinker.learn_actor import SActorLearner
from thinker.self_play import (
    SelfPlayWorker,
    TrainActorOut,
    VersionedTrainActorOut,
)


_REAL_SCHEMA6_ENDURO_RECONSTRUCTOR = (
    util._reconstruct_schema6_enduro_networks
)
_REAL_SCHEMA6_STAGE_VALIDATOR = util._validate_schema6_stage_profile
_REAL_SCHEMA7_STAGE_VALIDATOR = util._validate_schema7_stage_profile
_REAL_SCHEMA8_STAGE_VALIDATOR = util._validate_schema8_stage_profile
_REAL_SCHEMA9_STAGE_VALIDATOR = util._validate_schema9_stage_profile
_REAL_SCHEMA10_STAGE_VALIDATOR = util._validate_schema10_stage_profile
_REAL_SCHEMA11_STAGE_VALIDATOR = util._validate_schema11_stage_profile
_REAL_SCHEMA12_STAGE_VALIDATOR = util._validate_schema12_stage_profile
_REAL_SCHEMA13_STAGE_VALIDATOR = util._validate_schema13_stage_profile


class _TinySchema6Actor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2, 3))
        self.bias = torch.nn.Parameter(torch.zeros(2))
        self.voc_head = torch.nn.Linear(3, 2)
        self.voc_gate_head = torch.nn.Linear(3, 1)


class _TinySchema6Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vp_net = torch.nn.Linear(2, 2)
        self.sr_net = torch.nn.Linear(2, 2)


@pytest.fixture(autouse=True)
def _use_tiny_schema6_architecture_for_unit_fixtures(monkeypatch):
    """Keep corruption tests small; one separate test runs real constructors."""

    monkeypatch.setattr(
        util,
        "_reconstruct_schema6_enduro_networks",
        lambda _config, *, label: (_TinySchema6Actor(), _TinySchema6Model()),
    )
    # Final-bundle corruption tests use pytest-owned temporary directory names.
    # Dedicated tests below exercise the unpatched closed stage validator.
    monkeypatch.setattr(
        util,
        "_validate_schema6_stage_profile",
        lambda surface, *, label: tuple(
            surface[name]
            for name in (
                "xpid",
                "base_seed",
                "total_steps",
                "model_warm_up_n",
                "actor_unroll_len",
                "use_wandb",
            )
        ),
    )
    monkeypatch.setattr(
        util,
        "_validate_schema7_stage_profile",
        lambda surface, *, label: tuple(
            surface[name]
            for name in (
                "xpid",
                "base_seed",
                "total_steps",
                "model_warm_up_n",
                "actor_unroll_len",
                "use_wandb",
            )
        ),
    )
    monkeypatch.setattr(
        util,
        "_validate_schema8_stage_profile",
        lambda surface, *, label: tuple(
            surface[name]
            for name in (
                "xpid",
                "base_seed",
                "total_steps",
                "model_warm_up_n",
                "actor_unroll_len",
                "use_wandb",
            )
        ),
    )
    monkeypatch.setattr(
        util,
        "_validate_schema9_stage_profile",
        lambda surface, *, label: tuple(
            surface[name]
            for name in (
                "xpid",
                "base_seed",
                "total_steps",
                "model_warm_up_n",
                "actor_unroll_len",
                "use_wandb",
            )
        ),
    )
    monkeypatch.setattr(
        util,
        "_validate_schema10_stage_profile",
        lambda surface, *, label: tuple(
            surface[name]
            for name in (
                "xpid",
                "base_seed",
                "total_steps",
                "model_warm_up_n",
                "actor_unroll_len",
                "use_wandb",
            )
        ),
    )
    monkeypatch.setattr(
        util,
        "_validate_schema11_stage_profile",
        lambda surface, *, label: tuple(
            surface[name]
            for name in (
                "xpid",
                "base_seed",
                "total_steps",
                "model_warm_up_n",
                "actor_unroll_len",
                "use_wandb",
            )
        ),
    )
    monkeypatch.setattr(
        util,
        "_validate_schema12_stage_profile",
        lambda surface, *, label: tuple(
            surface[name]
            for name in (
                "xpid",
                "base_seed",
                "total_steps",
                "model_warm_up_n",
                "actor_unroll_len",
                "use_wandb",
            )
        ),
    )
    monkeypatch.setattr(
        util,
        "_validate_schema13_stage_profile",
        lambda surface, *, label: tuple(
            surface[name]
            for name in (
                "xpid",
                "base_seed",
                "total_steps",
                "model_warm_up_n",
                "actor_unroll_len",
                "use_wandb",
            )
        ),
    )


def _schema6_embedded_flags(**overrides):
    values = dict(util.VOC_GATE_POLICY_SCHEMA6_V12_BASELINE)
    profile_index = 1 if overrides.get("use_wandb") is True else 0
    xpid, seed, total, warmup, unroll, use_wandb = (
        util.VOC_GATE_POLICY_SCHEMA6_STAGE_PROFILES[profile_index]
    )
    savedir = "/tmp/di-voc-v13-versioned-eps25-test/runs"
    values.update({
        "__version__": "1.3.0",
        "git_revision": None,
        "dynamic_voc_mode": "control",
        "voc_gate_policy_schema_version": 6,
        "voc_loss_cost": 1.0,
        "voc_gate_temperature": 1.0,
        "voc_train_epsilon": 0.02,
        "voc_eval_stochastic": True,
        "voc_dueling_q": True,
        "voc_expected_gate_loss": True,
        "voc_ema_gate_target": True,
        "voc_gate_target_tau": 0.1,
        "voc_dedicated_gate": True,
        "voc_soft_q_bce_gate": True,
        "voc_gate_q_temperature": 0.05,
        "voc_gate_confidence_weighted": False,
        "voc_gate_adam_beta1": 0.0,
        "voc_gate_param_align": False,
        "voc_gate_param_align_coef": 1.0,
        "voc_gate_exact_projection": True,
        "voc_gate_epsilon_greedy_execution": True,
        "voc_gate_execution_epsilon": 0.25,
        "voc_gate_learning_rate": 0.001,
        "voc_gate_grad_norm_clipping": 1.0,
        "entropy_r_cost": 0.0,
        "voc_actor_policy_version_barrier": True,
        "voc_actor_policy_bundle_schema_version": 1,
        "voc_actor_policy_barrier_timeout_s": 120.0,
        "voc_actor_policy_ray_max_restarts": 0,
        "voc_actor_policy_ray_max_task_retries": 0,
        "actor_amp_init_scale": 32.0,
        "float16": True,
        "model_float16": False,
        "dual_net": True,
        "model_optimizer": "adam",
        "model_learning_rate": 0.00005,
        "actor_use_rms": False,
        "actor_adam_eps": 1e-8,
        "actor_learning_rate": 0.0003,
        "ppo_k": 1,
        "self_play_n": 1,
        "env_n": 16,
        "actor_batch_size": 16,
        "ckp": False,
        "train_actor": True,
        "parallel_actor": True,
        "preload": "",
        "preload_actor": "",
        "voc_parent_checkpoint": "",
        "total_steps": total,
        "schedule_total_steps": 100_000_000,
        "model_warm_up_n": warmup,
        "actor_unroll_len": unroll,
        "train_model": True,
        "use_wandb": use_wandb,
        "base_seed": seed,
        "name": "Enduro-v5",
        "icopro_game_id": 0,
        "envpool": True,
        "frame_stack_n": 4,
        "grayscale": False,
        "wrapper_type": 0,
        "dynamic_search": True,
        "dynamic_factorized_control": True,
        "max_search_steps": 20,
        "max_depth": 20,
        "rec_t": 20,
        "has_action_seq": False,
        "return_h": True,
        "return_x": True,
        "model_size_nn": 2,
        "model_disable_bn": False,
        "model_state_projection": "clamp",
        "model_state_range_loss_cost": 1.0,
        "xpid": xpid,
        "savedir": savedir,
        "ckpdir": f"{savedir}/{xpid}",
        "cmd": "train.py --schema6-test",
        "icopro_data_path": (
            "/tmp/di-voc-v13-versioned-eps25-test/data/"
            "behavioral_data_block"
        ),
        "voc_actor_policy_barrier_runtime": True,
    })
    values.update(overrides)
    return values


def _schema7_embedded_flags(**overrides):
    values = _schema6_embedded_flags()
    profile_index = 1 if overrides.get("use_wandb") is True else 0
    xpid, seed, total, warmup, unroll, use_wandb = (
        util.VOC_GATE_POLICY_SCHEMA7_STAGE_PROFILES[profile_index]
    )
    savedir = "/tmp/di-voc-v14-sealed-eps25-test/runs"
    values.update({
        "voc_gate_policy_schema_version": 7,
        "voc_model_input_seal_schema_version": 1,
        "xpid": xpid,
        "base_seed": seed,
        "total_steps": total,
        "model_warm_up_n": warmup,
        "actor_unroll_len": unroll,
        "use_wandb": use_wandb,
        "savedir": savedir,
        "ckpdir": f"{savedir}/{xpid}",
        "cmd": "train.py --schema7-test",
        "icopro_data_path": (
            "/tmp/di-voc-v14-sealed-eps25-test/data/"
            "behavioral_data_block"
        ),
    })
    values.update(overrides)
    return values


def _schema8_embedded_flags(**overrides):
    values = _schema7_embedded_flags()
    profile_index = 1 if overrides.get("use_wandb") is True else 0
    xpid, seed, total, warmup, unroll, use_wandb = (
        util.VOC_GATE_POLICY_SCHEMA8_STAGE_PROFILES[profile_index]
    )
    savedir = "/tmp/di-voc-v15-halfsq-eps25-test/runs"
    values.update({
        "voc_gate_policy_schema_version": 8,
        "voc_model_input_seal_schema_version": 1,
        "xpid": xpid,
        "base_seed": seed,
        "total_steps": total,
        "model_warm_up_n": warmup,
        "actor_unroll_len": unroll,
        "use_wandb": use_wandb,
        "savedir": savedir,
        "ckpdir": f"{savedir}/{xpid}",
        "cmd": "train.py --schema8-test",
        "icopro_data_path": (
            "/tmp/di-voc-v15-halfsq-eps25-test/data/"
            "behavioral_data_block"
        ),
    })
    values.update(overrides)
    return values


def _schema9_embedded_flags(**overrides):
    values = _schema8_embedded_flags()
    profile_index = 1 if overrides.get("use_wandb") is True else 0
    xpid, seed, total, warmup, unroll, use_wandb = (
        util.VOC_GATE_POLICY_SCHEMA9_STAGE_PROFILES[profile_index]
    )
    savedir = "/tmp/di-voc-v16-commonmode-eps25-test/runs"
    values.update({
        "voc_gate_policy_schema_version": 9,
        "voc_model_input_seal_schema_version": 1,
        "xpid": xpid,
        "base_seed": seed,
        "total_steps": total,
        "model_warm_up_n": warmup,
        "actor_unroll_len": unroll,
        "use_wandb": use_wandb,
        "savedir": savedir,
        "ckpdir": f"{savedir}/{xpid}",
        "cmd": "train.py --schema9-test",
        "icopro_data_path": (
            "/tmp/di-voc-v16-commonmode-eps25-test/data/"
            "behavioral_data_block"
        ),
    })
    values.update(overrides)
    return values


def _schema10_embedded_flags(**overrides):
    values = _schema9_embedded_flags()
    profile_index = 1 if overrides.get("use_wandb") is True else 0
    xpid, seed, total, warmup, unroll, use_wandb = (
        util.VOC_GATE_POLICY_SCHEMA10_STAGE_PROFILES[profile_index]
    )
    savedir = "/tmp/di-voc-v17-huber-common-eps25-test/runs"
    values.update({
        "voc_gate_policy_schema_version": 10,
        "voc_model_input_seal_schema_version": 1,
        "xpid": xpid,
        "base_seed": seed,
        "total_steps": total,
        "model_warm_up_n": warmup,
        "actor_unroll_len": unroll,
        "use_wandb": use_wandb,
        "savedir": savedir,
        "ckpdir": f"{savedir}/{xpid}",
        "cmd": "train.py --schema10-test",
        "icopro_data_path": (
            "/tmp/di-voc-v17-huber-common-eps25-test/data/"
            "behavioral_data_block"
        ),
    })
    values.update(overrides)
    return values


def _schema11_embedded_flags(**overrides):
    values = _schema10_embedded_flags()
    profile_index = 1 if overrides.get("use_wandb") is True else 0
    xpid, seed, total, warmup, unroll, use_wandb = (
        util.VOC_GATE_POLICY_SCHEMA11_STAGE_PROFILES[profile_index]
    )
    savedir = "/tmp/di-voc-v18-orthocd-adam-eps25-test/runs"
    values.update({
        "voc_gate_policy_schema_version": 11,
        "voc_model_input_seal_schema_version": 1,
        "xpid": xpid,
        "base_seed": seed,
        "total_steps": total,
        "model_warm_up_n": warmup,
        "actor_unroll_len": unroll,
        "use_wandb": use_wandb,
        "savedir": savedir,
        "ckpdir": f"{savedir}/{xpid}",
        "cmd": "train.py --schema11-test",
        "icopro_data_path": (
            "/tmp/di-voc-v18-orthocd-adam-eps25-test/data/"
            "behavioral_data_block"
        ),
    })
    values.update(overrides)
    return values


def _schema12_embedded_flags(**overrides):
    values = _schema11_embedded_flags()
    profile_index = 1 if overrides.get("use_wandb") is True else 0
    xpid, seed, total, warmup, unroll, use_wandb = (
        util.VOC_GATE_POLICY_SCHEMA12_STAGE_PROFILES[profile_index]
    )
    savedir = "/tmp/di-voc-v19-tau1-orthocd-adam-eps25-test/runs"
    values.update({
        "voc_gate_policy_schema_version": 12,
        "voc_model_input_seal_schema_version": 1,
        "voc_gate_target_tau": 1.0,
        "xpid": xpid,
        "base_seed": seed,
        "total_steps": total,
        "model_warm_up_n": warmup,
        "actor_unroll_len": unroll,
        "use_wandb": use_wandb,
        "savedir": savedir,
        "ckpdir": f"{savedir}/{xpid}",
        "cmd": "train.py --schema12-test",
        "icopro_data_path": (
            "/tmp/di-voc-v19-tau1-orthocd-adam-eps25-test/data/"
            "behavioral_data_block"
        ),
    })
    values.update(overrides)
    return values


def _schema13_embedded_flags(**overrides):
    values = _schema12_embedded_flags()
    profile_index = 1 if overrides.get("use_wandb") is True else 0
    xpid, seed, total, warmup, unroll, use_wandb = (
        util.VOC_GATE_POLICY_SCHEMA13_STAGE_PROFILES[profile_index]
    )
    savedir = "/tmp/di-voc-v20-telemetry-tau1-orthocd-adam-eps25-test/runs"
    values.update({
        "voc_gate_policy_schema_version": 13,
        "voc_model_input_seal_schema_version": 1,
        "voc_gate_target_tau": 1.0,
        "xpid": xpid,
        "base_seed": seed,
        "total_steps": total,
        "model_warm_up_n": warmup,
        "actor_unroll_len": unroll,
        "use_wandb": use_wandb,
        "savedir": savedir,
        "ckpdir": f"{savedir}/{xpid}",
        "cmd": "train.py --schema13-test",
        "icopro_data_path": (
            "/tmp/di-voc-v20-telemetry-tau1-orthocd-adam-eps25-test/data/"
            "behavioral_data_block"
        ),
    })
    values.update(overrides)
    return values


def _adam_checkpoint_state(parameters, *, step, initial_lr, current_lr):
    state = {}
    for parameter_id, tensor in enumerate(parameters):
        state[parameter_id] = {
            "step": torch.tensor(float(step), dtype=torch.float32),
            "exp_avg": torch.zeros_like(tensor),
            "exp_avg_sq": torch.zeros_like(tensor),
        }
    return {
        "state": state,
        "param_groups": [{
            "params": list(range(len(parameters))),
            "lr": current_lr,
            "initial_lr": initial_lr,
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
    }


def _scheduler_checkpoint_state(*, step, real_step, initial_lr, current_lr):
    return {
        "base_lrs": [initial_lr],
        "last_epoch": real_step,
        "_step_count": step + 1,
        "_is_initial": False,
        "_get_lr_called_within_step": False,
        "_last_lr": [current_lr],
        "lr_lambdas": [None],
    }


def _terminal_checkpoint(flags=None):
    flags = (
        _schema6_embedded_flags()
        if flags is None
        else copy.deepcopy(flags)
    )
    gate_schema = flags["voc_gate_policy_schema_version"]
    real_step = flags["total_steps"]
    ema_weight = torch.tensor(
        [[0.125, -0.25, 0.5], [-0.375, 0.5, 0.25]],
        dtype=torch.float32,
    )
    ema_bias = torch.tensor([0.125, -0.25], dtype=torch.float32)
    projection_scale = 1.0 / 0.05
    state = OrderedDict(
        weight=torch.arange(6, dtype=torch.float32).view(2, 3).clone(),
        bias=torch.tensor([0.25, -0.5], dtype=torch.float32),
        **{
            "voc_head.weight": torch.zeros(2, 3, dtype=torch.float32),
            "voc_head.bias": torch.zeros(2, dtype=torch.float32),
            "voc_gate_head.weight": projection_scale
            * (ema_weight[0:1] - ema_weight[1:2]),
            "voc_gate_head.bias": projection_scale
            * (ema_bias[0:1] - ema_bias[1:2]),
        },
    )
    if gate_schema in (
        util.VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION,
        util.VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION,
    ):
        state["voc_head.weight"] = ema_weight.clone()
        state["voc_head.bias"] = ema_bias.clone()
    bundle = util.make_actor_policy_bundle(
        state, 1, terminal=True, gate_schema=gate_schema
    )
    digests = [
        "0" * 64,
        util.actor_policy_state_sha256(bundle["actor_state_dict"]),
    ]
    history = (
        {
            "predecessor_version": -1,
            "policy_version": 0,
            "publication_count": 0,
            "terminal": False,
            "ack_ranks": [0],
            "expected_ack_count": 1,
            "state_sha256": digests[0],
        },
        {
            "predecessor_version": 0,
            "policy_version": 1,
            "publication_count": 1,
            "terminal": True,
            "ack_ranks": [0],
            "expected_ack_count": 1,
            "state_sha256": digests[1],
        },
    )
    q_lr = 0.0003 * (1.0 - real_step / 100_000_000.0)
    checkpoint = {
        "voc_gate_policy_schema_version": gate_schema,
        "flags": flags,
        "dynamic_voc_mode": "control",
        "actor_net_state_dict": util.clone_actor_policy_state(
            bundle["actor_state_dict"]
        ),
        "voc_ema_gate_target": True,
        "voc_gate_target_tau": flags["voc_gate_target_tau"],
        "voc_ema_gate_schema_version": util.VOC_EMA_GATE_SCHEMA_VERSION,
        "voc_ema_gate_head_state_dict": {
            "weight": ema_weight.clone(),
            "bias": ema_bias.clone(),
        },
        "voc_ema_gate_update_count": 1,
        "voc_ema_gate_parent_update_count": 0,
        "voc_update_count": 1,
        "voc_continue_count": 1,
        "voc_stop_count": 1,
        "voc_holdout_count": 0,
        "voc_holdout_split_version": util.VOC_HOLDOUT_SPLIT_VERSION,
        "voc_holdout_actor_modulus": util.VOC_HOLDOUT_ACTOR_MODULUS,
        "voc_holdout_actor_streams": 16,
        "voc_holdout_continue_count": 0,
        "voc_holdout_stop_count": 0,
        "voc_holdout_td_sum": 0.0,
        "voc_holdout_td_abs_sum": 0.0,
        "voc_holdout_td_sq_sum": 0.0,
        "voc_holdout_td_bias": None,
        "voc_holdout_td_mae": None,
        "voc_holdout_td_rmse": None,
        "voc_control_origin": util.VOC_CONTROL_ORIGIN_FRESH,
        "voc_control_origin_legacy_defaulted": False,
        "voc_parent_checkpoint_sha256": None,
        "voc_parent_checkpoint": None,
        "voc_parent_imitation_data_signature": None,
        "voc_activation_real_step": 0,
        "imitation_data_signature": "a" * 64,
        "imitation_update_count": 1,
        "imitation_schedule_step": 1,
        "voc_optimizer_state_dict": {
            "state": {
                0: {
                    "step": torch.tensor(1.0),
                    "exp_avg": torch.zeros(2, 3),
                    "exp_avg_sq": torch.zeros(2, 3),
                },
                1: {
                    "step": torch.tensor(1.0),
                    "exp_avg": torch.zeros(2),
                    "exp_avg_sq": torch.zeros(2),
                },
            },
            "param_groups": [{
                "params": [0, 1],
                "lr": q_lr,
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
            "last_epoch": real_step,
            "_step_count": 2,
            "_is_initial": False,
            "_get_lr_called_within_step": False,
            "_last_lr": [q_lr],
            "lr_lambdas": [None],
        },
        "voc_grad_scaler_state_dict": {
            "scale": 256.0,
            "growth_factor": 2.0,
            "backoff_factor": 0.5,
            "growth_interval": 2000,
            "_growth_tracker": 1,
        },
        "voc_amp_skip_count": 0,
        "voc_amp_consecutive_skips": 0,
        "voc_gate_optimizer_state_dict": {
            "state": {},
            "param_groups": [{
                "params": [0, 1],
                "lr": 0.001,
                "initial_lr": 0.001,
                "eps": 1e-8,
                "weight_decay": 0.0,
                "betas": (0.0, 0.999),
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
            "base_lrs": [0.001],
            "last_epoch": 0,
            "_step_count": 1,
            "_is_initial": False,
            "_get_lr_called_within_step": False,
            "_last_lr": [0.001],
            "lr_lambdas": [None],
        },
        "voc_gate_grad_scaler_state_dict": {
            "scale": 256.0,
            "growth_factor": 2.0,
            "backoff_factor": 0.5,
            "growth_interval": 2000,
            "_growth_tracker": 0,
        },
        "voc_gate_update_count": 1,
        "voc_gate_amp_skip_count": 0,
        "voc_gate_amp_consecutive_skips": 0,
        "actor_net_optimizer_state_dict": _adam_checkpoint_state(
            (state["weight"], state["bias"]),
            step=1,
            initial_lr=0.0003,
            current_lr=q_lr,
        ),
        "actor_net_scheduler_state_dict": _scheduler_checkpoint_state(
            step=1,
            real_step=real_step,
            initial_lr=0.0003,
            current_lr=q_lr,
        ),
        "voc_actor_policy_version": 1,
        "voc_actor_policy_publication_count": 1,
        "voc_actor_policy_terminal": True,
        "voc_actor_policy_version_mismatch_count": 0,
        "voc_actor_policy_malformed_bundle_count": 0,
        "voc_actor_policy_barrier_timeout_count": 0,
        "voc_actor_policy_terminal_ack_count": 1,
        "voc_actor_policy_expected_ack_count": 1,
        "voc_actor_policy_state_sha256": digests[1],
        "voc_actor_policy_bundle": bundle,
        "voc_actor_policy_publication_history": history,
        "voc_actor_policy_publication_history_sha256": (
            util.actor_policy_publication_history_sha256(history)
        ),
        "actor_grad_scaler_state_dict": {
            "scale": 32.0,
            "growth_factor": 2.0,
            "backoff_factor": 0.5,
            "growth_interval": 2000,
            "_growth_tracker": 1,
        },
        "actor_amp_skip_count": 0,
        "actor_amp_consecutive_skips": 0,
        "real_step": real_step,
        "step": 1,
    }
    return checkpoint


def _completion_checkpoint_files(fill="a"):
    return {
        name: {"sha256": fill * 64, "size": index + 1}
        for index, name in enumerate(util._COMPLETION_CHECKPOINT_FILES)
    }


def _write_private_logger_request(tmp_path):
    evidence = util.validate_actor_policy_checkpoint(_terminal_checkpoint())
    return util.write_actor_policy_logger_finish_request(
        tmp_path,
        evidence,
        {"checkpoint_files": _completion_checkpoint_files()},
    )


def _write_schema6_final_bundle(
    tmp_path, *, use_wandb=False, flags=None, model_overrides=None
):
    flags = (
        _schema6_embedded_flags(use_wandb=use_wandb)
        if flags is None
        else copy.deepcopy(flags)
    )
    resolved_ckpdir = str(tmp_path.resolve())
    resolved_savedir = str(tmp_path.resolve().parent)
    flags.update({
        "xpid": tmp_path.name,
        "savedir": resolved_savedir,
        "ckpdir": resolved_ckpdir,
        "icopro_data_path": str(
            tmp_path.resolve().parent.parent
            / "data"
            / "behavioral_data_block"
        ),
    })
    actor = _terminal_checkpoint(flags)
    real_step = flags["total_steps"]
    model_lr = 0.00005 * (1.0 - real_step / 100_000_000.0)
    model_state = OrderedDict(
        **{
            "vp_net.weight": torch.arange(
                4, dtype=torch.float32
            ).view(2, 2).clone(),
            "vp_net.bias": torch.tensor([0.0, 1.0]),
            "sr_net.weight": torch.arange(
                4, dtype=torch.float32
            ).view(2, 2).add(4.0).clone(),
            "sr_net.bias": torch.tensor([2.0, 3.0]),
        }
    )
    model = {
        "step": 1,
        "real_step": real_step,
        "model_net_state_dict": model_state,
        "model_net_optimizer_p_state_dict": _adam_checkpoint_state(
            (model_state["vp_net.weight"], model_state["vp_net.bias"]),
            step=1,
            initial_lr=0.00005,
            current_lr=model_lr,
        ),
        "model_net_scheduler_p_state_dict": _scheduler_checkpoint_state(
            step=1,
            real_step=real_step,
            initial_lr=0.00005,
            current_lr=model_lr,
        ),
        "model_net_optimizer_m_state_dict": _adam_checkpoint_state(
            (model_state["sr_net.weight"], model_state["sr_net.bias"]),
            step=1,
            initial_lr=0.00005,
            current_lr=model_lr,
        ),
        "model_net_scheduler_m_state_dict": _scheduler_checkpoint_state(
            step=1,
            real_step=real_step,
            initial_lr=0.00005,
            current_lr=model_lr,
        ),
        "model_grad_clip_count_m": 0,
        "model_grad_step_count_m": 1,
        "model_grad_clip_count_p": 0,
        "model_grad_step_count_p": 1,
        "flags": copy.deepcopy(flags),
    }
    if model_overrides is not None:
        model.update(copy.deepcopy(model_overrides))
    with (tmp_path / "config_c.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(flags, handle, sort_keys=True)
    torch.save(actor, tmp_path / "ckp_actor.tar")
    torch.save(model, tmp_path / "ckp_model.tar")
    return flags


def _schema7_model_input_seal_evidence(
    *, real_step, grad_steps=1, drain_count=0
):
    return {
        "voc_model_input_seal_schema_version": 1,
        "voc_model_input_sealed": True,
        "voc_model_input_seal_count": 1,
        "voc_model_terminal_processed_n": real_step,
        "voc_model_terminal_drain_update_count": drain_count,
        "voc_model_terminal_drain_pre_real_step": (
            real_step if drain_count == 0 else real_step - 1
        ),
        "voc_model_terminal_drain_pre_grad_step_count_m": (
            grad_steps - drain_count
        ),
        "voc_model_terminal_drain_pre_grad_step_count_p": (
            grad_steps - drain_count
        ),
        "voc_model_input_late_write_count": 0,
        "voc_model_input_abort_count": 0,
    }


def _write_schema7_final_bundle(
    tmp_path, *, use_wandb=False, drain_count=0, model_overrides=None
):
    flags = _schema7_embedded_flags(use_wandb=use_wandb)
    evidence = _schema7_model_input_seal_evidence(
        real_step=flags["total_steps"], drain_count=drain_count
    )
    if model_overrides:
        evidence.update(model_overrides)
    return _write_schema6_final_bundle(
        tmp_path,
        use_wandb=use_wandb,
        flags=flags,
        model_overrides=evidence,
    )


def _write_schema8_final_bundle(
    tmp_path, *, use_wandb=False, drain_count=0, model_overrides=None
):
    flags = _schema8_embedded_flags(use_wandb=use_wandb)
    evidence = _schema7_model_input_seal_evidence(
        real_step=flags["total_steps"], drain_count=drain_count
    )
    if model_overrides:
        evidence.update(model_overrides)
    return _write_schema6_final_bundle(
        tmp_path,
        use_wandb=use_wandb,
        flags=flags,
        model_overrides=evidence,
    )


def _write_schema9_final_bundle(
    tmp_path, *, use_wandb=False, drain_count=0, model_overrides=None
):
    flags = _schema9_embedded_flags(use_wandb=use_wandb)
    evidence = _schema7_model_input_seal_evidence(
        real_step=flags["total_steps"], drain_count=drain_count
    )
    if model_overrides:
        evidence.update(model_overrides)
    return _write_schema6_final_bundle(
        tmp_path,
        use_wandb=use_wandb,
        flags=flags,
        model_overrides=evidence,
    )


def _write_schema10_final_bundle(
    tmp_path, *, use_wandb=False, drain_count=0, model_overrides=None
):
    flags = _schema10_embedded_flags(use_wandb=use_wandb)
    evidence = _schema7_model_input_seal_evidence(
        real_step=flags["total_steps"], drain_count=drain_count
    )
    if model_overrides:
        evidence.update(model_overrides)
    return _write_schema6_final_bundle(
        tmp_path,
        use_wandb=use_wandb,
        flags=flags,
        model_overrides=evidence,
    )


def _write_schema11_final_bundle(
    tmp_path, *, use_wandb=False, drain_count=0, model_overrides=None
):
    flags = _schema11_embedded_flags(use_wandb=use_wandb)
    evidence = _schema7_model_input_seal_evidence(
        real_step=flags["total_steps"], drain_count=drain_count
    )
    if model_overrides:
        evidence.update(model_overrides)
    return _write_schema6_final_bundle(
        tmp_path,
        use_wandb=use_wandb,
        flags=flags,
        model_overrides=evidence,
    )


def _write_schema12_final_bundle(
    tmp_path, *, use_wandb=False, drain_count=0, model_overrides=None
):
    flags = _schema12_embedded_flags(use_wandb=use_wandb)
    evidence = _schema7_model_input_seal_evidence(
        real_step=flags["total_steps"], drain_count=drain_count
    )
    if model_overrides:
        evidence.update(model_overrides)
    return _write_schema6_final_bundle(
        tmp_path,
        use_wandb=use_wandb,
        flags=flags,
        model_overrides=evidence,
    )


def _write_schema13_final_bundle(
    tmp_path, *, use_wandb=False, drain_count=0, model_overrides=None
):
    flags = _schema13_embedded_flags(use_wandb=use_wandb)
    evidence = _schema7_model_input_seal_evidence(
        real_step=flags["total_steps"], drain_count=drain_count
    )
    if model_overrides:
        evidence.update(model_overrides)
    result = _write_schema6_final_bundle(
        tmp_path,
        use_wandb=use_wandb,
        flags=flags,
        model_overrides=evidence,
    )
    (tmp_path / "voc_telemetry_manifest.json").write_bytes(
        b'{"telemetry_schema_version":1}\n'
    )
    return result


def _fake_schema13_telemetry_manifest(_root, **expected):
    assert expected["expected_q_initial_lr"] == 0.0003
    assert expected["expected_schedule_total_steps"] == 100_000_000
    assert expected["expected_amp_initial_scale"] == 256.0
    assert type(expected["expected_publication_history"]) is tuple
    assert set(expected["expected_terminal_state"]) == {
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
    }
    return {
        "telemetry_schema_version": 1,
        "gate_schema": 13,
        "manifest_name": "voc_telemetry_manifest.json",
        "manifest_sha256": expected["expected_manifest_sha256"],
        "manifest_size": expected["expected_manifest_size"],
        "transaction_count": expected["expected_terminal_policy_version"],
        "terminal_policy_version": expected[
            "expected_terminal_policy_version"
        ],
        "terminal_real_step": expected["expected_terminal_real_step"],
        "actor_state_sha256": expected["expected_actor_state_sha256"],
        "publication_history_sha256": expected[
            "expected_publication_history_sha256"
        ],
    }


def test_schema6_gate_execution_distribution_is_875_125_and_tie_half():
    raw = torch.zeros(1, 3, 3)
    raw_gate = torch.tensor([[1.0, -1.0, 0.0]])
    distribution = compute_voc_gate_distribution(
        raw,
        temperature=1.0,
        epsilon=0.25,
        raw_gate_log_odds=raw_gate,
        epsilon_greedy_execution=True,
    )
    torch.testing.assert_close(
        distribution.continue_prob,
        torch.tensor([[0.875, 0.125, 0.5]]),
        rtol=0.0,
        atol=0.0,
    )


def test_schema6_real_enduro_reconstruction_has_frozen_counts_and_no_rng_effect():
    random.seed(12345)
    np.random.seed(23456)
    torch.manual_seed(34567)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()
    cuda_before = (
        tuple(state.clone() for state in torch.cuda.get_rng_state_all())
        if torch.cuda.is_initialized() else None
    )

    actor, model = _REAL_SCHEMA6_ENDURO_RECONSTRUCTOR(
        _schema6_embedded_flags(), label="schema-6 Enduro unit reconstruction"
    )

    assert len(actor.state_dict()) == 119
    assert len(tuple(actor.parameters())) == 117
    assert len(model.state_dict()) == 468
    assert len(tuple(model.vp_net.parameters())) == 103
    assert len(tuple(model.sr_net.parameters())) == 147
    excluded = {
        id(parameter)
        for module in (actor.voc_head, actor.voc_gate_head)
        for parameter in module.parameters()
    }
    assert sum(
        id(parameter) not in excluded for parameter in actor.parameters()
    ) == 113
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_before)
    if cuda_before is not None:
        cuda_after = torch.cuda.get_rng_state_all()
        assert len(cuda_after) == len(cuda_before)
        assert all(
            torch.equal(after, before)
            for after, before in zip(cuda_after, cuda_before)
        )


@pytest.mark.parametrize(
    ("field", "bad"),
    [("name", "Pong-v5"), ("icopro_game_id", 1), ("frame_stack_n", 3)],
)
def test_schema6_reconstruction_is_source_hardcoded_to_enduro(field, bad):
    config = _schema6_embedded_flags()
    config[field] = bad
    with pytest.raises(ValueError, match="frozen Enduro"):
        _REAL_SCHEMA6_ENDURO_RECONSTRUCTOR(config, label="bad stage")


def test_legacy_train_actor_tuple_is_unchanged_and_schema6_is_separate():
    assert "policy_version" not in TrainActorOut._fields
    assert VersionedTrainActorOut._fields[:-1] == TrainActorOut._fields
    assert VersionedTrainActorOut._fields[-1] == "policy_version"


def test_bundle_exact_shape_clones_cpu_state_and_rejects_noncanonical_input():
    source = OrderedDict(
        weight=torch.arange(6, dtype=torch.float32).view(2, 3),
        bias=torch.tensor([1.0, 2.0]),
    )
    bundle = util.make_actor_policy_bundle(source, 0, terminal=False)
    assert set(bundle) == {
        "bundle_schema_version",
        "policy_version",
        "terminal",
        "gate_schema",
        "actor_state_dict",
    }
    source["weight"].zero_()
    assert not torch.equal(source["weight"], bundle["actor_state_dict"]["weight"])
    validated = util.validate_actor_policy_bundle(
        bundle,
        expected_epoch=0,
        expected_terminal=False,
        expected_actor_state=bundle["actor_state_dict"],
    )
    assert validated["policy_version"] == 0

    malformed = dict(bundle)
    malformed["bundle_schema_version"] = True
    with pytest.raises(ValueError, match="bundle schema"):
        util.validate_actor_policy_bundle(
            malformed,
            expected_actor_state=bundle["actor_state_dict"],
        )

    noncontiguous = OrderedDict(bundle["actor_state_dict"])
    noncontiguous["weight"] = torch.arange(12, dtype=torch.float32).view(3, 4).t()
    malformed = dict(bundle, actor_state_dict=noncontiguous)
    with pytest.raises(ValueError, match="contiguous"):
        util.validate_actor_policy_bundle(
            malformed,
            expected_actor_state=noncontiguous,
        )

    import numpy as np

    for bad_terminal in (np.bool_(False), 0, 1):
        with pytest.raises(ValueError, match="terminal"):
            util.make_actor_policy_bundle(source, 0, terminal=bad_terminal)
        with pytest.raises(ValueError, match="terminal"):
            util.make_actor_policy_ack(0, 0, terminal=bad_terminal)


def test_schema7_bundle_and_ack_keep_shape_and_bind_strict_gate_schema():
    state = OrderedDict(weight=torch.ones(1, 1))
    bundle = util.make_actor_policy_bundle(
        state, 3, terminal=True, gate_schema=7
    )
    ack = util.make_actor_policy_ack(
        0, 3, terminal=True, gate_schema=7
    )
    assert set(bundle) == {
        "bundle_schema_version",
        "policy_version",
        "terminal",
        "gate_schema",
        "actor_state_dict",
    }
    assert set(ack) == {
        "bundle_schema_version",
        "gate_schema",
        "rank",
        "policy_version",
        "terminal",
    }
    assert bundle["bundle_schema_version"] == 1
    assert ack["bundle_schema_version"] == 1
    assert bundle["gate_schema"] == ack["gate_schema"] == 7
    util.validate_actor_policy_bundle(
        bundle,
        expected_epoch=3,
        expected_terminal=True,
        expected_actor_state=bundle["actor_state_dict"],
        expected_gate_schema=7,
    )
    util.validate_actor_policy_ack(
        ack,
        rank=0,
        epoch=3,
        terminal=True,
        expected_gate_schema=7,
    )

    for bad in (True, np.int64(7), 7.0, np.nextafter(7.0, 8.0)):
        with pytest.raises(ValueError, match="gate_schema"):
            util.make_actor_policy_bundle(
                state, 3, terminal=True, gate_schema=bad
            )
        malformed_bundle = copy.deepcopy(bundle)
        malformed_bundle["gate_schema"] = bad
        with pytest.raises(ValueError, match="gate-policy schema"):
            util.validate_actor_policy_bundle(
                malformed_bundle,
                expected_actor_state=bundle["actor_state_dict"],
                expected_gate_schema=7,
            )
        malformed_ack = copy.deepcopy(ack)
        malformed_ack["gate_schema"] = bad
        with pytest.raises(ValueError, match="gate_schema"):
            util.validate_actor_policy_ack(
                malformed_ack, expected_gate_schema=7
            )


def test_schema8_bundle_and_ack_are_exact_five_key_strict_atomic_records():
    state = OrderedDict(weight=torch.ones(1, 1))
    bundle = util.make_actor_policy_bundle(
        state, 3, terminal=True, gate_schema=8
    )
    ack = util.make_actor_policy_ack(
        0, 3, terminal=True, gate_schema=8
    )
    assert set(bundle) == {
        "bundle_schema_version",
        "policy_version",
        "terminal",
        "gate_schema",
        "actor_state_dict",
    }
    assert set(ack) == {
        "bundle_schema_version",
        "gate_schema",
        "rank",
        "policy_version",
        "terminal",
    }
    validated = util.validate_actor_policy_bundle(
        bundle,
        expected_epoch=3,
        expected_terminal=True,
        expected_actor_state=bundle["actor_state_dict"],
        expected_gate_schema=8,
    )
    assert validated["gate_schema"] == 8
    assert util.validate_actor_policy_ack(
        ack,
        rank=0,
        epoch=3,
        terminal=True,
        expected_gate_schema=8,
    )["gate_schema"] == 8

    for bad in (True, np.int64(8), 8.0, "8", np.nextafter(8.0, 9.0)):
        with pytest.raises(ValueError, match="gate_schema"):
            util.make_actor_policy_bundle(
                state, 3, terminal=True, gate_schema=bad
            )
        malformed_bundle = copy.deepcopy(bundle)
        malformed_bundle["gate_schema"] = bad
        with pytest.raises(ValueError, match="gate-policy schema"):
            util.validate_actor_policy_bundle(
                malformed_bundle,
                expected_actor_state=bundle["actor_state_dict"],
                expected_gate_schema=8,
            )
        malformed_ack = copy.deepcopy(ack)
        malformed_ack["gate_schema"] = bad
        with pytest.raises(ValueError, match="gate_schema"):
            util.validate_actor_policy_ack(
                malformed_ack, expected_gate_schema=8
            )

    for record, validator, kwargs in (
        (
            bundle,
            util.validate_actor_policy_bundle,
            {
                "expected_actor_state": bundle["actor_state_dict"],
                "expected_gate_schema": 8,
            },
        ),
        (ack, util.validate_actor_policy_ack, {"expected_gate_schema": 8}),
    ):
        for mutation in ("missing", "extra"):
            malformed = copy.deepcopy(record)
            if mutation == "missing":
                malformed.pop("bundle_schema_version")
            else:
                malformed["voc_q_regression_loss"] = "half_squared_td"
            with pytest.raises(ValueError, match="exactly|contain exactly"):
                validator(malformed, **kwargs)

    with pytest.raises(ValueError, match="invalid gate-policy schema"):
        util.validate_actor_policy_bundle(
            bundle,
            expected_actor_state=bundle["actor_state_dict"],
            expected_gate_schema=7,
        )


def test_schema9_bundle_ack_and_history_keep_exact_shapes_without_identity_keys():
    state = OrderedDict(weight=torch.ones(1, 1))
    bundle = util.make_actor_policy_bundle(
        state, 3, terminal=True, gate_schema=9
    )
    ack = util.make_actor_policy_ack(
        0, 3, terminal=True, gate_schema=9
    )
    assert set(bundle) == {
        "bundle_schema_version",
        "policy_version",
        "terminal",
        "gate_schema",
        "actor_state_dict",
    }
    assert set(ack) == {
        "bundle_schema_version",
        "gate_schema",
        "rank",
        "policy_version",
        "terminal",
    }
    assert util.validate_actor_policy_bundle(
        bundle,
        expected_epoch=3,
        expected_terminal=True,
        expected_actor_state=bundle["actor_state_dict"],
        expected_gate_schema=9,
    )["gate_schema"] == 9
    assert util.validate_actor_policy_ack(
        ack,
        rank=0,
        epoch=3,
        terminal=True,
        expected_gate_schema=9,
    )["gate_schema"] == 9

    for bad in (True, np.int64(9), 9.0, "9", 8, 10):
        malformed_bundle = copy.deepcopy(bundle)
        malformed_bundle["gate_schema"] = bad
        with pytest.raises(ValueError, match="gate-policy schema"):
            util.validate_actor_policy_bundle(
                malformed_bundle,
                expected_actor_state=bundle["actor_state_dict"],
                expected_gate_schema=9,
            )
        malformed_ack = copy.deepcopy(ack)
        malformed_ack["gate_schema"] = bad
        with pytest.raises(ValueError, match="gate_schema"):
            util.validate_actor_policy_ack(
                malformed_ack, expected_gate_schema=9
            )

    checkpoint = _terminal_checkpoint(_schema9_embedded_flags())
    event_keys = {
        "predecessor_version",
        "policy_version",
        "publication_count",
        "terminal",
        "ack_ranks",
        "expected_ack_count",
        "state_sha256",
    }
    assert all(
        set(event) == event_keys
        for event in checkpoint["voc_actor_policy_publication_history"]
    )
    evidence = util.validate_actor_policy_checkpoint(checkpoint)
    assert evidence["voc_actor_policy_bundle_summary"]["gate_schema"] == 9
    for forbidden in ("voc_q_regression_loss", "voc_q_reconstruction"):
        assert forbidden not in checkpoint
        assert forbidden not in checkpoint["flags"]
        assert forbidden not in evidence


def test_schema10_bundle_ack_and_history_keep_exact_shapes_without_identity_keys():
    state = OrderedDict(weight=torch.ones(1, 1))
    bundle = util.make_actor_policy_bundle(
        state, 3, terminal=True, gate_schema=10
    )
    ack = util.make_actor_policy_ack(
        0, 3, terminal=True, gate_schema=10
    )
    assert set(bundle) == {
        "bundle_schema_version",
        "policy_version",
        "terminal",
        "gate_schema",
        "actor_state_dict",
    }
    assert set(ack) == {
        "bundle_schema_version",
        "gate_schema",
        "rank",
        "policy_version",
        "terminal",
    }
    assert util.validate_actor_policy_bundle(
        bundle,
        expected_epoch=3,
        expected_terminal=True,
        expected_actor_state=bundle["actor_state_dict"],
        expected_gate_schema=10,
    )["gate_schema"] == 10
    assert util.validate_actor_policy_ack(
        ack,
        rank=0,
        epoch=3,
        terminal=True,
        expected_gate_schema=10,
    )["gate_schema"] == 10

    for bad in (True, np.int64(10), 10.0, "10", 9, 11):
        malformed_bundle = copy.deepcopy(bundle)
        malformed_bundle["gate_schema"] = bad
        with pytest.raises(ValueError, match="gate-policy schema"):
            util.validate_actor_policy_bundle(
                malformed_bundle,
                expected_actor_state=bundle["actor_state_dict"],
                expected_gate_schema=10,
            )
        malformed_ack = copy.deepcopy(ack)
        malformed_ack["gate_schema"] = bad
        with pytest.raises(ValueError, match="gate_schema"):
            util.validate_actor_policy_ack(
                malformed_ack, expected_gate_schema=10
            )

    checkpoint = _terminal_checkpoint(_schema10_embedded_flags())
    event_keys = {
        "predecessor_version",
        "policy_version",
        "publication_count",
        "terminal",
        "ack_ranks",
        "expected_ack_count",
        "state_sha256",
    }
    assert all(
        set(event) == event_keys
        for event in checkpoint["voc_actor_policy_publication_history"]
    )
    evidence = util.validate_actor_policy_checkpoint(checkpoint)
    assert evidence["voc_actor_policy_bundle_summary"]["gate_schema"] == 10
    for forbidden in ("voc_q_regression_loss", "voc_q_reconstruction"):
        assert forbidden not in checkpoint
        assert forbidden not in checkpoint["flags"]
        assert forbidden not in evidence


def test_schema11_bundle_ack_and_history_keep_exact_shapes_without_identity_keys():
    state = OrderedDict(weight=torch.ones(1, 1))
    bundle = util.make_actor_policy_bundle(
        state, 3, terminal=True, gate_schema=11
    )
    ack = util.make_actor_policy_ack(
        0, 3, terminal=True, gate_schema=11
    )
    assert set(bundle) == {
        "bundle_schema_version",
        "policy_version",
        "terminal",
        "gate_schema",
        "actor_state_dict",
    }
    assert set(ack) == {
        "bundle_schema_version",
        "gate_schema",
        "rank",
        "policy_version",
        "terminal",
    }
    assert util.validate_actor_policy_bundle(
        bundle,
        expected_epoch=3,
        expected_terminal=True,
        expected_actor_state=bundle["actor_state_dict"],
        expected_gate_schema=11,
    )["gate_schema"] == 11
    assert util.validate_actor_policy_ack(
        ack,
        rank=0,
        epoch=3,
        terminal=True,
        expected_gate_schema=11,
    )["gate_schema"] == 11
    for bad in (True, np.int64(11), 11.0, "11", 10, 12):
        malformed_bundle = copy.deepcopy(bundle)
        malformed_bundle["gate_schema"] = bad
        with pytest.raises(ValueError, match="gate-policy schema"):
            util.validate_actor_policy_bundle(
                malformed_bundle,
                expected_actor_state=bundle["actor_state_dict"],
                expected_gate_schema=11,
            )
        malformed_ack = copy.deepcopy(ack)
        malformed_ack["gate_schema"] = bad
        with pytest.raises(ValueError, match="gate_schema"):
            util.validate_actor_policy_ack(
                malformed_ack, expected_gate_schema=11
            )
    checkpoint = _terminal_checkpoint(_schema11_embedded_flags())
    event_keys = {
        "predecessor_version",
        "policy_version",
        "publication_count",
        "terminal",
        "ack_ranks",
        "expected_ack_count",
        "state_sha256",
    }
    assert all(
        set(event) == event_keys
        for event in checkpoint["voc_actor_policy_publication_history"]
    )
    evidence = util.validate_actor_policy_checkpoint(checkpoint)
    assert evidence["voc_actor_policy_bundle_summary"]["gate_schema"] == 11
    for forbidden in util._VOC_GATE_POLICY_SCHEMA11_DERIVED_IDENTITY_KEYS:
        assert forbidden not in checkpoint
        assert forbidden not in checkpoint["flags"]
        assert forbidden not in evidence


def test_schema6_bundle_validator_retains_historical_numpy_integer_acceptance():
    state = OrderedDict(weight=torch.ones(1, 1))
    bundle = util.make_actor_policy_bundle(state, 0)
    bundle["bundle_schema_version"] = np.int64(1)
    bundle["gate_schema"] = np.int64(6)
    validated = util.validate_actor_policy_bundle(
        bundle, expected_actor_state=bundle["actor_state_dict"]
    )
    assert type(validated["gate_schema"]) is int
    assert validated["gate_schema"] == 6
    ack = util.make_actor_policy_ack(0, 0)
    ack["bundle_schema_version"] = np.int64(1)
    ack["gate_schema"] = np.int64(6)
    assert util.validate_actor_policy_ack(ack)["gate_schema"] == 6
    for bad in (True, np.int64(1), 1.0, np.nextafter(1.0, 2.0)):
        malformed_bundle = copy.deepcopy(bundle)
        malformed_bundle["bundle_schema_version"] = bad
        with pytest.raises(ValueError, match="bundle schema"):
            util.validate_actor_policy_bundle(
                malformed_bundle,
                expected_actor_state=bundle["actor_state_dict"],
                expected_gate_schema=7,
            )
        malformed_ack = copy.deepcopy(ack)
        malformed_ack["bundle_schema_version"] = bad
        with pytest.raises(ValueError, match="bundle_schema_version"):
            util.validate_actor_policy_ack(
                malformed_ack, expected_gate_schema=7
            )


def test_schema7_checkpoint_binds_bundle_schema_without_history_shape_change():
    checkpoint = _terminal_checkpoint(_schema7_embedded_flags())
    assert checkpoint["voc_actor_policy_bundle"]["gate_schema"] == 7
    assert all(
        set(event) == {
            "predecessor_version",
            "policy_version",
            "publication_count",
            "terminal",
            "ack_ranks",
            "expected_ack_count",
            "state_sha256",
        }
        for event in checkpoint["voc_actor_policy_publication_history"]
    )
    evidence = util.validate_actor_policy_checkpoint(checkpoint)
    assert evidence["voc_actor_policy_bundle_summary"]["gate_schema"] == 7

    checkpoint["voc_actor_policy_bundle"]["gate_schema"] = 6
    with pytest.raises(ValueError, match="gate-policy schema"):
        util.validate_actor_policy_checkpoint(checkpoint)


def test_schema8_checkpoint_binds_bundle_schema_with_exact_seven_key_history():
    checkpoint = _terminal_checkpoint(_schema8_embedded_flags())
    assert checkpoint["voc_actor_policy_bundle"]["gate_schema"] == 8
    event_keys = {
        "predecessor_version",
        "policy_version",
        "publication_count",
        "terminal",
        "ack_ranks",
        "expected_ack_count",
        "state_sha256",
    }
    assert all(
        set(event) == event_keys
        for event in checkpoint["voc_actor_policy_publication_history"]
    )
    evidence = util.validate_actor_policy_checkpoint(checkpoint)
    assert evidence["voc_actor_policy_bundle_summary"]["gate_schema"] == 8
    assert "voc_q_regression_loss" not in checkpoint
    assert "voc_q_regression_loss" not in checkpoint["flags"]
    assert "voc_q_regression_loss" not in evidence

    checkpoint["voc_actor_policy_publication_history"][0][
        "voc_q_regression_loss"
    ] = "half_squared_td"
    with pytest.raises(ValueError, match="history event is malformed"):
        util.validate_actor_policy_checkpoint(checkpoint)


def test_schema6_checkpoint_returns_full_validated_history_evidence():
    checkpoint = _terminal_checkpoint()
    evidence = util.validate_actor_policy_checkpoint(checkpoint)
    assert evidence["voc_actor_policy_publication_event_count"] == 2
    assert evidence["voc_actor_policy_publication_history"] == checkpoint[
        "voc_actor_policy_publication_history"
    ]
    assert evidence["voc_actor_policy_final_publication_event"]["terminal"] is True
    assert evidence["voc_actor_policy_publication_history_sha256"] == checkpoint[
        "voc_actor_policy_publication_history_sha256"
    ]
    assert "voc_actor_policy_bundle" not in evidence
    assert evidence["voc_actor_policy_bundle_summary"][
        "actor_state_dict_sha256"
    ] == checkpoint["voc_actor_policy_state_sha256"]
    json.dumps(evidence, allow_nan=False)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: c.update(voc_actor_policy_publication_count=0),
        lambda c: c["voc_actor_policy_publication_history"][1].update(
            predecessor_version=-1
        ),
        lambda c: c["voc_actor_policy_publication_history"][0].update(
            terminal=True
        ),
        lambda c: c.update(voc_actor_policy_state_sha256="f" * 64),
    ],
)
def test_schema6_checkpoint_history_corruption_fails_closed(mutation):
    import copy

    checkpoint = copy.deepcopy(_terminal_checkpoint())
    checkpoint["voc_actor_policy_publication_history"] = list(
        checkpoint["voc_actor_policy_publication_history"]
    )
    mutation(checkpoint)
    with pytest.raises(ValueError):
        util.validate_actor_policy_checkpoint(checkpoint)


@pytest.mark.parametrize("kind", ["noncontiguous", "requires_grad", "nonfinite", "alias", "cross_alias"])
def test_schema6_checkpoint_rejects_noncanonical_top_level_actor_state(kind):
    checkpoint = _terminal_checkpoint()
    if kind == "noncontiguous":
        checkpoint["actor_net_state_dict"]["weight"] = (
            torch.arange(12, dtype=torch.float32).view(2, 6)[:, ::2]
        )
    elif kind == "requires_grad":
        checkpoint["actor_net_state_dict"]["weight"].requires_grad_(True)
    elif kind == "nonfinite":
        checkpoint["actor_net_state_dict"]["weight"][0, 0] = float("nan")
    elif kind == "alias":
        base = torch.arange(6, dtype=torch.float32)
        checkpoint["actor_net_state_dict"] = OrderedDict(
            weight=base.view(2, 3), bias=base[:2]
        )
    else:
        checkpoint["actor_net_state_dict"] = checkpoint[
            "voc_actor_policy_bundle"
        ]["actor_state_dict"]
    with pytest.raises(ValueError):
        util.validate_actor_policy_checkpoint(checkpoint)


def _batch(policy_version, actor_ids):
    Batch = namedtuple("Batch", "policy_version episode_return id")
    return Batch(
        policy_version=policy_version,
        episode_return=torch.zeros(policy_version.shape),
        id=actor_ids,
    )


def _batch_validator():
    learner = SActorLearner.__new__(SActorLearner)
    learner.voc_actor_policy_version = 7
    learner.voc_actor_policy_malformed_bundle_count = 0
    learner.voc_actor_policy_version_mismatch_count = 0
    learner.flags = SimpleNamespace(actor_batch_size=16, actor_unroll_len=3)
    return learner


def test_legacy_partial_learner_defaults_missing_barrier_runtime_off(
    monkeypatch,
):
    """A pre-schema-6 partial learner must preserve its original exception."""

    class _RemoteMethod:
        def remote(self, *_args, **_kwargs):
            return object()

    learner = SActorLearner.__new__(SActorLearner)
    learner.time = False
    learner.real_step = 0
    learner.flags = SimpleNamespace(total_steps=1)
    learner.actor_buffer = SimpleNamespace(read=_RemoteMethod())
    learner.queue_n = 0.0
    learner._logger = SimpleNamespace(error=lambda *_args: None)
    learner.consume_data = lambda *_args, **_kwargs: (
        _ for _ in ()
    ).throw(RuntimeError("legacy learner failure"))
    close_calls = []
    learner.close = lambda successful=True: close_calls.append(successful)
    monkeypatch.setattr(learn_actor.ray, "get", lambda _ref: ((), ()))
    monkeypatch.setattr(
        learn_actor.ray.internal, "free", lambda _ref: None
    )
    monkeypatch.setattr(util, "tuple_map", lambda value, _fn: value)

    assert not hasattr(learner, "voc_actor_policy_barrier_runtime")
    with pytest.raises(RuntimeError, match="legacy learner failure"):
        learner.learn_data()
    assert close_calls == [False]


def test_batch_stamp_accepts_shuffled_complete_actor_ids():
    learner = _batch_validator()
    versions = torch.full((4, 16), 7, dtype=torch.int64)
    versions[0] = -1
    actor_ids = torch.tensor([[5, 0, 15, 2, 1, 9, 8, 7, 6, 4, 3, 10, 11, 12, 14, 13]])
    learner._validate_actor_policy_batch(_batch(versions, actor_ids))


@pytest.mark.parametrize(
    "kind",
    ["missing", "float", "bool", "wrong_t", "extra_t", "wrong_b", "b32"],
)
def test_batch_stamp_rejects_missing_dtype_and_shape_corruption(kind):
    learner = _batch_validator()
    versions = torch.full((4, 16), 7, dtype=torch.int64)
    versions[0] = -1
    ids = torch.arange(16, dtype=torch.int64).view(1, 16)
    batch = _batch(versions, ids)
    if kind == "missing":
        batch = batch._replace(policy_version=None)
    elif kind == "float":
        batch = batch._replace(policy_version=versions.float())
    elif kind == "bool":
        batch = batch._replace(policy_version=versions.bool())
    elif kind == "wrong_t":
        batch = batch._replace(policy_version=versions[:-1])
    elif kind == "extra_t":
        extended = torch.cat((versions, versions[-1:]), dim=0)
        batch = _batch(extended, ids)
    elif kind == "wrong_b":
        batch = batch._replace(policy_version=versions[:, :-1])
    else:
        wide = torch.full((4, 32), 7, dtype=torch.int64)
        wide[0] = -1
        batch = _batch(wide, ids)
    with pytest.raises(RuntimeError, match="policy_version"):
        learner._validate_actor_policy_batch(batch)


@pytest.mark.parametrize(
    "ids",
    [
        torch.tensor([[0] * 16]),
        torch.arange(1, 17).view(1, 16),
        torch.arange(16, dtype=torch.float32).view(1, 16),
        torch.zeros((1, 16), dtype=torch.bool),
    ],
)
def test_batch_stamp_rejects_duplicate_missing_out_of_range_and_bad_id_dtype(ids):
    learner = _batch_validator()
    versions = torch.full((4, 16), 7, dtype=torch.int64)
    versions[0] = -1
    with pytest.raises(RuntimeError, match="actor id"):
        learner._validate_actor_policy_batch(_batch(versions, ids))


def test_raw_learner_bootstrap_none_is_bounded_by_one_monotonic_deadline(monkeypatch):
    class Remote:
        def remote(self, _name):
            return object()

    learner = SActorLearner.__new__(SActorLearner)
    learner.voc_actor_policy_barrier_runtime = True
    learner.voc_actor_policy_barrier_timeout_s = 120.0
    learner.voc_actor_policy_barrier_timeout_count = 0
    learner.actor_param_buffer = SimpleNamespace(get_data=Remote())
    times = iter([0.0, 0.0, 121.0])
    learner._monotonic = lambda: next(times)
    learner._barrier_sleep = lambda _seconds: None
    observed_timeouts = []

    def fake_get(_ref, timeout=None):
        observed_timeouts.append(timeout)
        return None

    monkeypatch.setattr(learn_actor.ray, "get", fake_get)
    with pytest.raises(TimeoutError):
        learner.refresh_actor()
    assert observed_timeouts == [120.0]
    assert learner.voc_actor_policy_barrier_timeout_count == 1


def test_terminal_completion_returns_without_post_terminal_env_step():
    class Env:
        def __init__(self):
            self.env_steps = 0
            self.closed = False

        def poll_model_status_no_step(self, *, timeout):
            assert 0.0 < timeout <= 120.0
            return {"finish": True}

        def close(self):
            self.closed = True

        def step(self, *_args, **_kwargs):
            self.env_steps += 1
            raise AssertionError("post-terminal env action")

    worker_class = SelfPlayWorker.__ray_metadata__.modified_class
    worker = worker_class.__new__(worker_class)
    worker.env = Env()
    worker.rank = 1
    worker.voc_actor_policy_barrier_timeout_s = 120.0
    worker._monotonic = lambda: 0.0
    worker._barrier_sleep = lambda _seconds: None
    worker._logger = SimpleNamespace(info=lambda *_args, **_kwargs: None)
    assert worker._complete_terminal_policy(
        {"model_status": {"finish": False}}
    ) is True
    assert worker.env.env_steps == 0
    assert worker.env.closed is True


def test_failed_schema6_close_aborts_and_kills_without_terminal_or_finish(monkeypatch):
    calls = []

    class Remote:
        def __init__(self, name):
            self.name = name

        def remote(self, *args):
            calls.append((self.name, args))
            return True

    learner = SActorLearner.__new__(SActorLearner)
    learner._closed = False
    learner.bc_runner = None
    learner.voc_actor_policy_barrier_runtime = True
    learner.voc_actor_policy_version = 3
    learner.voc_actor_policy_barrier_timeout_s = 120.0
    learner._monotonic = lambda: 0.0
    learner.actor_param_buffer = SimpleNamespace(set_data=Remote("set_data"))
    learner.actor_buffer = SimpleNamespace(set_finish=Remote("set_finish"))
    learner._barrier_ray_get = MethodType(
        lambda self, ref, **_kwargs: ref, learner
    )
    learner._logger = SimpleNamespace(error=lambda *_args, **_kwargs: None)
    learner.plogger = SimpleNamespace(close=lambda **_kwargs: None)
    monkeypatch.setattr(
        learn_actor.ray,
        "kill",
        lambda actor, no_restart: calls.append(("kill", actor, no_restart)),
    )
    learner.close(successful=False)
    assert not any(name == "set_finish" for name, *_ in calls)
    assert any(name == "set_data" for name, *_ in calls)
    assert any(name == "kill" for name, *_ in calls)


def test_driver_watchdog_fails_closed_on_never_resolving_worker(monkeypatch):
    worker_ref = object()
    cancelled = []

    class Remote:
        def remote(self, _name):
            return object()

    times = iter([0.0, 1.0, 2.0, 3.0, 121.0])
    monkeypatch.setattr(
        train_driver.ray,
        "wait",
        lambda refs, **_kwargs: ([], refs),
    )
    monkeypatch.setattr(
        train_driver.ray,
        "get",
        lambda _ref, timeout=None: None,
    )
    monkeypatch.setattr(
        train_driver.ray,
        "cancel",
        lambda ref, force: cancelled.append((ref, force)),
    )
    with pytest.raises(TimeoutError, match="no policy heartbeat progress"):
        train_driver.wait_for_schema6_workers(
            [worker_ref],
            SimpleNamespace(get_data=Remote()),
            timeout_s=120.0,
            monotonic=lambda: next(times),
        )
    assert cancelled == [(worker_ref, False)]


def test_driver_watchdog_kills_actor_handle_before_nonforce_ref_cancel(
    monkeypatch,
):
    worker_ref = object()
    worker_handle = object()
    order = []

    class Remote:
        def remote(self, _name):
            return object()

    times = iter([0.0, 121.0])
    monkeypatch.setattr(
        train_driver.ray, "wait", lambda refs, **_kwargs: ([], refs)
    )
    monkeypatch.setattr(
        train_driver.ray,
        "kill",
        lambda actor, *, no_restart: order.append(
            ("kill", actor, no_restart)
        ),
    )

    def reject_force_cancel(ref, *, force):
        if force:
            raise ValueError("Ray actor tasks reject force=True")
        order.append(("cancel", ref, force))

    monkeypatch.setattr(train_driver.ray, "cancel", reject_force_cancel)
    with pytest.raises(TimeoutError, match="no policy heartbeat progress"):
        train_driver.wait_for_schema6_workers(
            [worker_ref],
            SimpleNamespace(get_data=Remote()),
            worker_handles=[worker_handle],
            monotonic=lambda: next(times),
        )
    assert order == [
        ("kill", worker_handle, True),
        ("cancel", worker_ref, False),
    ]


def test_history_rejects_bool_ack_rank_and_non_python_terminal():
    import copy
    import numpy as np

    checkpoint = copy.deepcopy(_terminal_checkpoint())
    checkpoint["voc_actor_policy_publication_history"] = list(
        checkpoint["voc_actor_policy_publication_history"]
    )
    checkpoint["voc_actor_policy_publication_history"][0]["ack_ranks"] = [False]
    checkpoint["voc_actor_policy_publication_history_sha256"] = (
        util.actor_policy_publication_history_sha256(
            checkpoint["voc_actor_policy_publication_history"]
        )
    )
    with pytest.raises(ValueError, match="full ack"):
        util.validate_actor_policy_checkpoint(checkpoint)

    checkpoint = copy.deepcopy(_terminal_checkpoint())
    checkpoint["voc_actor_policy_publication_history"] = list(
        checkpoint["voc_actor_policy_publication_history"]
    )
    checkpoint["voc_actor_policy_publication_history"][0]["terminal"] = np.bool_(False)
    with pytest.raises(ValueError, match="not boolean"):
        util.validate_actor_policy_checkpoint(checkpoint)


def test_heartbeat_validation_allows_canonical_jump_and_rejects_oscillation():
    first, progressed = util.validate_actor_policy_heartbeat(
        {"rank": 0, "policy_version": 0, "phase": "load_ack", "count": 1}
    )
    assert progressed is True
    jumped, progressed = util.validate_actor_policy_heartbeat(
        {"rank": 0, "policy_version": 2, "phase": "enqueue", "count": 6},
        previous=first,
    )
    assert progressed is True
    duplicate, progressed = util.validate_actor_policy_heartbeat(
        jumped, previous=jumped
    )
    assert duplicate == jumped
    assert progressed is False
    with pytest.raises(ValueError, match="regressed"):
        util.validate_actor_policy_heartbeat(first, previous=jumped)
    with pytest.raises(ValueError, match="relation"):
        util.validate_actor_policy_heartbeat(
            {"rank": 0, "policy_version": 1, "phase": "load_ack", "count": 4},
            previous=first,
        )


def test_checkpoint_request_is_deferred_during_open_policy_transaction():
    learner = SActorLearner.__new__(SActorLearner)
    learner.voc_gate_exact_projection = False
    learner.voc_actor_policy_barrier_runtime = True
    learner._voc_actor_policy_transaction_open = True
    learner._voc_actor_policy_checkpoint_pending = False
    learner._voc_actor_policy_checkpoint_force = False
    # No actor/checkpoint/logger fields are installed: reaching any file-write
    # path would fail this test immediately.
    learner.save_checkpoint(force=True)
    assert learner._voc_actor_policy_checkpoint_pending is True
    assert learner._voc_actor_policy_checkpoint_force is True


def test_terminal_v0_checkpoint_is_rejected_as_no_batch_laundering():
    checkpoint = copy.deepcopy(_terminal_checkpoint())
    checkpoint["voc_actor_policy_version"] = 0
    checkpoint["voc_actor_policy_publication_count"] = 0
    checkpoint["voc_actor_policy_bundle"]["policy_version"] = 0
    checkpoint["voc_actor_policy_publication_history"] = [
        {
            **checkpoint["voc_actor_policy_publication_history"][0],
            "terminal": True,
        }
    ]
    checkpoint["voc_actor_policy_publication_history_sha256"] = (
        util.actor_policy_publication_history_sha256(
            checkpoint["voc_actor_policy_publication_history"]
        )
    )
    with pytest.raises(ValueError, match="at least one batch"):
        util.validate_actor_policy_checkpoint(checkpoint)


@pytest.mark.parametrize("outer_rank", [False, np.int64(0)])
def test_driver_heartbeat_outer_rank_requires_python_integer_zero(
    monkeypatch, outer_rank
):
    worker_ref = object()
    cancelled = []
    heartbeat = {
        outer_rank: {
            "rank": 0,
            "policy_version": 0,
            "phase": "load_ack",
            "count": 1,
        }
    }

    class Remote:
        def remote(self, _name):
            return object()

    monkeypatch.setattr(
        train_driver.ray, "wait", lambda refs, **_kwargs: ([], refs)
    )
    monkeypatch.setattr(
        train_driver.ray, "get", lambda _ref, timeout=None: heartbeat
    )
    monkeypatch.setattr(
        train_driver.ray,
        "cancel",
        lambda ref, force: cancelled.append((ref, force)),
    )
    with pytest.raises(RuntimeError, match="Python integer 0"):
        train_driver.wait_for_schema6_workers(
            [worker_ref],
            SimpleNamespace(get_data=Remote()),
            monotonic=lambda: 0.0,
        )
    assert cancelled == [(worker_ref, False)]


def test_policy_transaction_coalesces_save_requests_until_post_ack_flush():
    learner = SActorLearner.__new__(SActorLearner)
    learner.voc_gate_exact_projection = False
    learner.voc_actor_policy_barrier_runtime = True
    learner._voc_actor_policy_transaction_open = True
    learner._voc_actor_policy_checkpoint_pending = False
    learner._voc_actor_policy_checkpoint_force = False

    SActorLearner.save_checkpoint(learner, force=False)
    SActorLearner.save_checkpoint(learner, force=True)
    assert learner._voc_actor_policy_checkpoint_pending is True
    assert learner._voc_actor_policy_checkpoint_force is True

    writes = []
    learner._voc_actor_policy_transaction_open = False
    learner.save_checkpoint = lambda *, force=False: writes.append(force)
    assert learner._flush_pending_actor_policy_checkpoint() is True
    assert learner._flush_pending_actor_policy_checkpoint() is False
    assert writes == [True]


def test_private_logger_request_ack_round_trip_and_cleanup(tmp_path):
    evidence = util.validate_actor_policy_checkpoint(_terminal_checkpoint())
    completion_evidence = {
        "checkpoint_files": _completion_checkpoint_files(),
    }
    request = util.write_actor_policy_logger_finish_request(
        tmp_path, evidence, completion_evidence
    )
    assert util.read_actor_policy_logger_finish_request(tmp_path) == request
    ack = util.write_actor_policy_logger_finish_ack(tmp_path, request)
    assert util.read_actor_policy_logger_finish_ack(tmp_path, request) == ack
    assert not (tmp_path / "finish").exists()
    util.clear_actor_policy_logger_completion(tmp_path)
    assert not (
        tmp_path / util.VOC_ACTOR_POLICY_LOGGER_FINISH_REQUEST_FILE
    ).exists()
    assert not (
        tmp_path / util.VOC_ACTOR_POLICY_LOGGER_FINISH_ACK_FILE
    ).exists()


def test_owned_private_logger_cleanup_rejects_replaced_request_inode(tmp_path):
    evidence = util.validate_actor_policy_checkpoint(_terminal_checkpoint())
    completion_evidence = {"checkpoint_files": _completion_checkpoint_files()}
    request, identity = util.write_actor_policy_logger_finish_request(
        tmp_path,
        evidence,
        completion_evidence,
        return_identity=True,
    )
    request_path = (
        tmp_path / util.VOC_ACTOR_POLICY_LOGGER_FINISH_REQUEST_FILE
    )
    request_path.unlink()
    util.write_actor_policy_logger_finish_request(
        tmp_path, evidence, completion_evidence
    )
    with pytest.raises(RuntimeError, match="ownership changed"):
        util.clear_actor_policy_logger_completion(
            tmp_path,
            expected_request=request,
            expected_request_identity=identity,
        )
    assert request_path.exists()


def test_schema6_request_collision_preserves_foreign_private_marker(
    monkeypatch, tmp_path
):
    evidence = util.validate_actor_policy_checkpoint(_terminal_checkpoint())
    completion_evidence = {
        "checkpoint_files": _completion_checkpoint_files(),
        "implementation_sources": {},
        "loaded_extensions": {},
    }
    existing = util.write_actor_policy_logger_finish_request(
        tmp_path, evidence, completion_evidence
    )
    flags = SimpleNamespace(
        ckpdir=str(tmp_path),
        use_wandb=True,
        voc_actor_policy_barrier_runtime=True,
        voc_actor_policy_barrier_timeout_s=120.0,
    )
    monkeypatch.setattr(
        train_driver.ray, "kill", lambda actor, no_restart: None
    )
    monkeypatch.setattr(
        train_driver.ray,
        "get",
        lambda _ref, timeout=None: (_ for _ in ()).throw(
            RuntimeError("confirmed terminal logger task")
        ),
    )
    with pytest.raises(FileExistsError, match="request exists"):
        train_driver.finish_schema6_log_worker(
            object(),
            flags,
            final_bundle={
                "actor_policy": evidence,
                "completion_evidence": completion_evidence,
            },
            logger_worker=object(),
            monotonic=lambda: 0.0,
        )
    assert util.read_actor_policy_logger_finish_request(tmp_path) == existing
    assert not (tmp_path / "finish").exists()


def test_driver_fails_immediately_if_logger_exits_before_private_request(
    monkeypatch,
):
    worker_ref = object()
    logger_ref = object()
    cancelled = []

    def fake_wait(refs, **_kwargs):
        if refs == [logger_ref]:
            return [logger_ref], []
        return [], refs

    monkeypatch.setattr(train_driver.ray, "wait", fake_wait)
    monkeypatch.setattr(
        train_driver.ray, "get", lambda _ref, timeout=None: False
    )
    monkeypatch.setattr(
        train_driver.ray,
        "cancel",
        lambda ref, force: cancelled.append((ref, force)),
    )
    with pytest.raises(RuntimeError, match="exited before"):
        train_driver.wait_for_schema6_workers(
            [worker_ref],
            SimpleNamespace(),
            logger_ref=logger_ref,
            monotonic=lambda: 0.0,
        )
    assert cancelled == [(worker_ref, False)]


def test_schema6_logger_timeout_cleans_private_markers_and_never_commits_finish(
    monkeypatch, tmp_path
):
    evidence = util.validate_actor_policy_checkpoint(_terminal_checkpoint())
    final_bundle = {
        "actor_policy": evidence,
        "completion_evidence": {
            "checkpoint_files": _completion_checkpoint_files(),
            "implementation_sources": {},
            "loaded_extensions": {},
        },
    }
    flags = SimpleNamespace(
        ckpdir=str(tmp_path),
        use_wandb=True,
        voc_actor_policy_barrier_runtime=True,
        voc_actor_policy_barrier_timeout_s=120.0,
    )
    logger_ref = object()
    logger_worker = object()
    cancelled = []
    get_calls = 0

    def timeout_get(_ref, timeout=None):
        nonlocal get_calls
        get_calls += 1
        if get_calls == 1:
            raise train_driver.ray.exceptions.GetTimeoutError("injected hang")
        raise RuntimeError("confirmed dead logger task")

    monkeypatch.setattr(train_driver.ray, "get", timeout_get)
    monkeypatch.setattr(
        train_driver.ray, "kill", lambda actor, no_restart: None
    )
    monkeypatch.setattr(
        train_driver.ray,
        "cancel",
        lambda ref, force: cancelled.append((ref, force)),
    )
    with pytest.raises(TimeoutError, match="did not acknowledge"):
        train_driver.finish_schema6_log_worker(
            logger_ref,
            flags,
            final_bundle=final_bundle,
            logger_worker=logger_worker,
            monotonic=lambda: 0.0,
        )
    assert cancelled == [(logger_ref, False)]
    assert not (tmp_path / "finish").exists()
    assert not (
        tmp_path / util.VOC_ACTOR_POLICY_LOGGER_FINISH_REQUEST_FILE
    ).exists()
    assert not (
        tmp_path / util.VOC_ACTOR_POLICY_LOGGER_FINISH_ACK_FILE
    ).exists()


def test_schema6_finalize_never_writes_public_finish_after_logger_failure(
    monkeypatch, tmp_path
):
    flags = SimpleNamespace(
        ckpdir=str(tmp_path),
        use_wandb=True,
        voc_actor_policy_barrier_runtime=True,
    )
    writes = []
    monkeypatch.setattr(
        train_driver,
        "finish_schema6_log_worker",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("injected logger failure")
        ),
    )
    monkeypatch.setattr(
        train_driver.util,
        "validate_schema6_final_bundle",
        lambda _path: {
            "actor_policy": util.validate_actor_policy_checkpoint(
                _terminal_checkpoint()
            ),
            "config_use_wandb": True,
            "completion_evidence": {},
        },
    )
    monkeypatch.setattr(
        train_driver.util,
        "write_run_completion",
        lambda _path, **_kwargs: writes.append("public"),
    )
    with pytest.raises(RuntimeError, match="injected logger failure"):
        train_driver.finalize_run(flags, logger_ref=object())
    assert writes == []


def test_schema6_finalize_commits_public_finish_only_after_logger_ack(
    monkeypatch, tmp_path
):
    flags = SimpleNamespace(
        ckpdir=str(tmp_path),
        use_wandb=True,
        voc_actor_policy_barrier_runtime=True,
    )
    order = []
    evidence = util.validate_actor_policy_checkpoint(_terminal_checkpoint())
    monkeypatch.setattr(
        train_driver.util,
        "validate_schema6_final_bundle",
        lambda _path: {
            "actor_policy": evidence,
            "config_use_wandb": True,
            "completion_evidence": {"frozen": True},
        },
    )
    monkeypatch.setattr(
        train_driver,
        "finish_schema6_log_worker",
        lambda *_args, **_kwargs: order.append("logger_ack") or True,
    )
    monkeypatch.setattr(
        train_driver.util,
        "write_run_completion",
        lambda _path, **_kwargs: (
            order.append("public_finish") or {"status": "complete"}
        ),
    )
    assert train_driver.finalize_run(flags, logger_ref=object()) == {
        "status": "complete"
    }
    assert order == ["logger_ack", "public_finish"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda checkpoint: checkpoint.update(actor_amp_skip_count=1),
        lambda checkpoint: checkpoint.update(actor_amp_consecutive_skips=1),
        lambda checkpoint: checkpoint["actor_grad_scaler_state_dict"].update(
            scale=64.0
        ),
        lambda checkpoint: checkpoint["actor_grad_scaler_state_dict"].update(
            _growth_tracker=0
        ),
        lambda checkpoint: checkpoint["actor_grad_scaler_state_dict"].update(
            growth_interval=True
        ),
    ],
)
def test_schema6_main_actor_amp_state_is_reconstructed_and_zero_skip(mutation):
    checkpoint = copy.deepcopy(_terminal_checkpoint())
    mutation(checkpoint)
    with pytest.raises(ValueError, match="actor|GradScaler"):
        util.validate_actor_policy_checkpoint(checkpoint)


@pytest.mark.parametrize(
    "name",
    [
        "voc_actor_policy_version_barrier",
        "ckp",
        "train_actor",
        "parallel_actor",
    ],
)
def test_schema6_fresh_and_barrier_booleans_require_python_bool(name):
    checkpoint = copy.deepcopy(_terminal_checkpoint())
    checkpoint["flags"][name] = np.bool_(checkpoint["flags"][name])
    with pytest.raises(ValueError, match="boolean"):
        util.validate_actor_policy_checkpoint(checkpoint)


def test_schema6_final_bundle_validates_config_actor_model_and_triplet(tmp_path):
    _write_schema6_final_bundle(tmp_path, use_wandb=False)
    bundle = util.validate_schema6_final_bundle(tmp_path)
    assert bundle["actor_policy"]["voc_actor_policy_terminal"] is True
    assert bundle["model_state_tensor_count"] == 4
    assert bundle["resolved_identity"]["key_count"] == 228
    assert bundle["resolved_identity"]["v12_projection_key_count"] == 209
    assert bundle["resolved_identity"]["v12_projection_sha256"] == (
        util.VOC_GATE_POLICY_SCHEMA6_V12_BASELINE_SHA256
    )
    assert set(bundle["resolved_identity"]) == {
        "key_count",
        "v12_projection_key_count",
        "v12_projection_sha256",
        "complete_surface_sha256",
        "stage",
        "paths",
    }
    assert "model_input_seal" not in bundle
    json.dumps(bundle, allow_nan=False)
    assert bundle["config_use_wandb"] is False
    assert set(bundle["completion_evidence"]["checkpoint_files"]) == {
        "config_c.yaml",
        "ckp_actor.tar",
        "ckp_model.tar",
    }


def test_schema6_final_bundle_accepts_real_closed_wire_stage(monkeypatch, tmp_path):
    monkeypatch.setattr(
        util, "_validate_schema6_stage_profile", _REAL_SCHEMA6_STAGE_VALIDATOR
    )
    xpid = util.VOC_GATE_POLICY_SCHEMA6_STAGE_PROFILES[0][0]
    ckpdir = tmp_path / xpid
    ckpdir.mkdir()
    _write_schema6_final_bundle(ckpdir, use_wandb=False)
    evidence = util.validate_schema6_final_bundle(ckpdir)
    assert tuple(evidence["resolved_identity"]["stage"]) == (
        util.VOC_GATE_POLICY_SCHEMA6_STAGE_PROFILES[0]
    )


@pytest.mark.parametrize("drain_count", [0, 1])
def test_schema7_final_bundle_validates_both_terminal_drain_branches(
    tmp_path, drain_count
):
    _write_schema7_final_bundle(tmp_path, drain_count=drain_count)
    bundle = util.validate_schema7_final_bundle(tmp_path)
    assert set(bundle) == {
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
    }
    assert set(bundle["resolved_identity"]) == {
        "gate_schema",
        "voc_gate_policy_schema_version",
        "voc_model_input_seal_schema_version",
        "key_count",
        "v12_projection_key_count",
        "v12_projection_sha256",
        "complete_surface_sha256",
        "stage",
        "paths",
    }
    assert bundle["resolved_identity"]["gate_schema"] == 7
    assert bundle["resolved_identity"]["voc_gate_policy_schema_version"] == 7
    assert bundle["resolved_identity"][
        "voc_model_input_seal_schema_version"
    ] == 1
    assert bundle["resolved_identity"]["key_count"] == 229
    assert bundle["resolved_identity"]["v12_projection_key_count"] == 209
    assert bundle["resolved_identity"]["v12_projection_sha256"] == (
        util.VOC_GATE_POLICY_SCHEMA6_V12_BASELINE_SHA256
    )
    seal = bundle["model_input_seal"]
    assert set(seal) == set(util._SCHEMA7_MODEL_INPUT_SEAL_EVIDENCE_FIELDS)
    assert seal["voc_model_input_seal_schema_version"] == 1
    assert seal["voc_model_input_sealed"] is True
    assert seal["voc_model_input_seal_count"] == 1
    assert seal["voc_model_terminal_drain_update_count"] == drain_count
    assert seal["voc_model_input_late_write_count"] == 0
    assert seal["voc_model_input_abort_count"] == 0
    assert seal["voc_model_terminal_processed_n"] == bundle["model_real_step"]
    json.dumps(bundle, allow_nan=False)


def test_schema7_final_bundle_accepts_real_closed_wire_stage(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        util, "_validate_schema7_stage_profile", _REAL_SCHEMA7_STAGE_VALIDATOR
    )
    xpid = util.VOC_GATE_POLICY_SCHEMA7_STAGE_PROFILES[0][0]
    ckpdir = tmp_path / xpid
    ckpdir.mkdir()
    _write_schema7_final_bundle(ckpdir)
    evidence = util.validate_schema7_final_bundle(ckpdir)
    assert tuple(evidence["resolved_identity"]["stage"]) == (
        util.VOC_GATE_POLICY_SCHEMA7_STAGE_PROFILES[0]
    )


@pytest.mark.parametrize("drain_count", [0, 1])
def test_schema8_final_bundle_adds_only_derived_half_squared_identity(
    tmp_path, drain_count
):
    _write_schema8_final_bundle(tmp_path, drain_count=drain_count)
    bundle = util.validate_schema8_final_bundle(tmp_path)
    assert set(bundle["resolved_identity"]) == {
        "gate_schema",
        "voc_gate_policy_schema_version",
        "voc_model_input_seal_schema_version",
        "voc_q_regression_loss",
        "key_count",
        "v12_projection_key_count",
        "v12_projection_sha256",
        "complete_surface_sha256",
        "stage",
        "paths",
    }
    identity = bundle["resolved_identity"]
    assert identity["gate_schema"] == 8
    assert identity["voc_gate_policy_schema_version"] == 8
    assert identity["voc_model_input_seal_schema_version"] == 1
    assert identity["voc_q_regression_loss"] == "half_squared_td"
    assert identity["key_count"] == 229
    assert identity["v12_projection_key_count"] == 209
    assert identity["v12_projection_sha256"] == (
        util.VOC_GATE_POLICY_SCHEMA6_V12_BASELINE_SHA256
    )
    assert set(bundle["model_input_seal"]) == set(
        util._SCHEMA7_MODEL_INPUT_SEAL_EVIDENCE_FIELDS
    )
    assert bundle["model_input_seal"][
        "voc_model_terminal_drain_update_count"
    ] == drain_count
    json.dumps(bundle, allow_nan=False)

    with (tmp_path / "config_c.yaml").open(
        "r", encoding="utf-8"
    ) as handle:
        config = yaml.safe_load(handle)
    actor = torch.load(tmp_path / "ckp_actor.tar", weights_only=False)
    model = torch.load(tmp_path / "ckp_model.tar", weights_only=False)
    for surface in (config, actor["flags"], model["flags"]):
        assert len(surface) == 229
        assert "voc_q_regression_loss" not in surface


def test_schema8_dedicated_final_route_rejects_schema7_and_cross_surface_drift(
    tmp_path
):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    _write_schema7_final_bundle(legacy)
    with pytest.raises(ValueError, match="not schema 8"):
        util.validate_schema8_final_bundle(legacy)

    drift = tmp_path / "drift"
    drift.mkdir()
    _write_schema8_final_bundle(drift)
    actor = torch.load(drift / "ckp_actor.tar", weights_only=False)
    actor["flags"]["cmd"] += " --drift"
    torch.save(actor, drift / "ckp_actor.tar")
    with pytest.raises(ValueError, match="surfaces differ"):
        util.validate_schema8_final_bundle(drift)


def test_schema8_final_bundle_accepts_real_closed_wire_stage(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        util, "_validate_schema8_stage_profile", _REAL_SCHEMA8_STAGE_VALIDATOR
    )
    xpid = util.VOC_GATE_POLICY_SCHEMA8_STAGE_PROFILES[0][0]
    ckpdir = tmp_path / xpid
    ckpdir.mkdir()
    _write_schema8_final_bundle(ckpdir)
    evidence = util.validate_schema8_final_bundle(ckpdir)
    assert tuple(evidence["resolved_identity"]["stage"]) == (
        util.VOC_GATE_POLICY_SCHEMA8_STAGE_PROFILES[0]
    )


@pytest.mark.parametrize("drain_count", [0, 1])
def test_schema9_final_bundle_has_exact_derived_only_eleven_key_identity(
    tmp_path, drain_count
):
    _write_schema9_final_bundle(tmp_path, drain_count=drain_count)
    bundle = util.validate_schema9_final_bundle(tmp_path)
    identity = bundle["resolved_identity"]
    assert tuple(identity) == (
        "key_count",
        "v12_projection_key_count",
        "v12_projection_sha256",
        "complete_surface_sha256",
        "stage",
        "paths",
        "gate_schema",
        "voc_gate_policy_schema_version",
        "voc_model_input_seal_schema_version",
        "voc_q_regression_loss",
        "voc_q_reconstruction",
    )
    assert identity["key_count"] == 229
    assert identity["v12_projection_key_count"] == 209
    assert identity["v12_projection_sha256"] == (
        util.VOC_GATE_POLICY_SCHEMA6_V12_BASELINE_SHA256
    )
    assert identity["gate_schema"] == 9
    assert identity["voc_gate_policy_schema_version"] == 9
    assert identity["voc_model_input_seal_schema_version"] == 1
    assert identity["voc_q_regression_loss"] == "half_squared_td"
    assert identity["voc_q_reconstruction"] == (
        "detached_value_plus_raw_head_mean_plus_policy_centered_raw_head"
    )
    assert bundle["model_input_seal"][
        "voc_model_terminal_drain_update_count"
    ] == drain_count
    json.dumps(bundle, allow_nan=False)

    with (tmp_path / "config_c.yaml").open(
        "r", encoding="utf-8"
    ) as handle:
        config = yaml.safe_load(handle)
    actor = torch.load(tmp_path / "ckp_actor.tar", weights_only=False)
    model = torch.load(tmp_path / "ckp_model.tar", weights_only=False)
    for surface in (config, actor["flags"], model["flags"]):
        assert len(surface) == 229
        assert set(surface) == util._VOC_GATE_POLICY_SCHEMA9_COMPLETE_KEYS
        assert not {
            "voc_q_regression_loss",
            "voc_q_reconstruction",
        } & surface.keys()
    for checkpoint in (actor, model):
        assert "voc_q_regression_loss" not in checkpoint
        assert "voc_q_reconstruction" not in checkpoint


def test_schema9_dedicated_final_and_actor_routes_reject_every_non9_schema(
    tmp_path
):
    schema8 = tmp_path / "schema8"
    schema8.mkdir()
    _write_schema8_final_bundle(schema8)
    with pytest.raises(ValueError, match="not schema 9"):
        util.validate_schema9_final_bundle(schema8)

    schema9_flags = _schema9_embedded_flags()
    schema9_actor = _terminal_checkpoint(schema9_flags)
    validated = util.validate_voc_schema9_final_actor_checkpoint(
        schema9_actor,
        util.argparse.Namespace(**schema9_flags),
    )
    assert validated["voc_gate_policy_schema_version"] == 9
    assert validated["voc_q_regression_loss"] == "half_squared_td"
    assert validated["voc_q_reconstruction"] == (
        "detached_value_plus_raw_head_mean_plus_policy_centered_raw_head"
    )

    schema8_flags = _schema8_embedded_flags()
    with pytest.raises(ValueError, match="schema-9|schema 9"):
        util.validate_voc_schema9_final_actor_checkpoint(
            _terminal_checkpoint(schema8_flags),
            util.argparse.Namespace(**schema8_flags),
        )


def test_schema9_final_bundle_rejects_cross_surface_stage_drift(tmp_path):
    _write_schema9_final_bundle(tmp_path)
    actor = torch.load(tmp_path / "ckp_actor.tar", weights_only=False)
    actor["flags"]["base_seed"] = 5
    torch.save(actor, tmp_path / "ckp_actor.tar")
    with pytest.raises(
        ValueError,
        match="unregistered schema-9 stage|cross-surface identity",
    ):
        util.validate_schema9_final_bundle(tmp_path)


def test_schema9_final_bundle_accepts_real_closed_wire_stage(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        util, "_validate_schema9_stage_profile", _REAL_SCHEMA9_STAGE_VALIDATOR
    )
    xpid = util.VOC_GATE_POLICY_SCHEMA9_STAGE_PROFILES[0][0]
    ckpdir = tmp_path / xpid
    ckpdir.mkdir()
    _write_schema9_final_bundle(ckpdir)
    evidence = util.validate_schema9_final_bundle(ckpdir)
    assert tuple(evidence["resolved_identity"]["stage"]) == (
        util.VOC_GATE_POLICY_SCHEMA9_STAGE_PROFILES[0]
    )


@pytest.mark.parametrize("drain_count", [0, 1])
def test_schema10_final_bundle_has_exact_derived_only_eleven_key_identity(
    tmp_path, drain_count
):
    _write_schema10_final_bundle(tmp_path, drain_count=drain_count)
    bundle = util.validate_schema10_final_bundle(tmp_path)
    identity = bundle["resolved_identity"]
    assert tuple(identity) == (
        "key_count",
        "v12_projection_key_count",
        "v12_projection_sha256",
        "complete_surface_sha256",
        "stage",
        "paths",
        "gate_schema",
        "voc_gate_policy_schema_version",
        "voc_model_input_seal_schema_version",
        "voc_q_regression_loss",
        "voc_q_reconstruction",
    )
    assert identity["key_count"] == 229
    assert identity["v12_projection_key_count"] == 209
    assert identity["v12_projection_sha256"] == (
        util.VOC_GATE_POLICY_SCHEMA6_V12_BASELINE_SHA256
    )
    assert identity["gate_schema"] == 10
    assert identity["voc_gate_policy_schema_version"] == 10
    assert identity["voc_model_input_seal_schema_version"] == 1
    assert identity["voc_q_regression_loss"] == "smooth_l1_beta1"
    assert identity["voc_q_reconstruction"] == (
        "detached_value_plus_raw_head_mean_plus_policy_centered_raw_head"
    )
    assert bundle["model_input_seal"][
        "voc_model_terminal_drain_update_count"
    ] == drain_count
    json.dumps(bundle, allow_nan=False)

    with (tmp_path / "config_c.yaml").open(
        "r", encoding="utf-8"
    ) as handle:
        config = yaml.safe_load(handle)
    actor = torch.load(tmp_path / "ckp_actor.tar", weights_only=False)
    model = torch.load(tmp_path / "ckp_model.tar", weights_only=False)
    for surface in (config, actor["flags"], model["flags"]):
        assert len(surface) == 229
        assert set(surface) == util._VOC_GATE_POLICY_SCHEMA10_COMPLETE_KEYS
        assert not {
            "voc_q_regression_loss",
            "voc_q_reconstruction",
        } & surface.keys()
    for checkpoint in (actor, model):
        assert "voc_q_regression_loss" not in checkpoint
        assert "voc_q_reconstruction" not in checkpoint


def test_schema10_dedicated_final_and_actor_routes_reject_non10_schema(tmp_path):
    schema9 = tmp_path / "schema9"
    schema9.mkdir()
    _write_schema9_final_bundle(schema9)
    with pytest.raises(ValueError, match="not schema 10"):
        util.validate_schema10_final_bundle(schema9)

    flags = _schema10_embedded_flags()
    actor = _terminal_checkpoint(flags)
    validated = util.validate_voc_schema10_final_actor_checkpoint(
        actor,
        util.argparse.Namespace(**flags),
    )
    assert validated["voc_gate_policy_schema_version"] == 10
    assert validated["voc_q_regression_loss"] == "smooth_l1_beta1"
    assert validated["voc_q_reconstruction"] == (
        "detached_value_plus_raw_head_mean_plus_policy_centered_raw_head"
    )

    schema9_flags = _schema9_embedded_flags()
    with pytest.raises(ValueError, match="schema-10|schema 10"):
        util.validate_voc_schema10_final_actor_checkpoint(
            _terminal_checkpoint(schema9_flags),
            util.argparse.Namespace(**schema9_flags),
        )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("voc_q_regression_loss", "smooth_l1_beta1"),
        ("voc_q_regression_loss", "half_squared_td"),
        ("voc_q_regression_loss", None),
        (
            "voc_q_reconstruction",
            "detached_value_plus_raw_head_mean_plus_policy_centered_raw_head",
        ),
        ("voc_q_reconstruction", "forged"),
        ("voc_q_reconstruction", None),
    ],
)
def test_schema10_actor_route_rejects_reserved_identity_presence_before_state_use(
    monkeypatch, key, value
):
    flags = _schema10_embedded_flags()
    actor = _terminal_checkpoint(flags)
    actor[key] = value
    monkeypatch.setattr(
        util,
        "_validate_voc_atomic_final_actor_checkpoint",
        lambda *_args, **_kwargs: pytest.fail(
            "schema-10 reserved identity reached actor state validation"
        ),
    )
    with pytest.raises(ValueError, match=rf"reserved schema-10.*{key}"):
        util.validate_voc_schema10_final_actor_checkpoint(
            actor, util.argparse.Namespace(**flags)
        )


def _nested_schema10_reserved_identity(nesting, key, value):
    reserved = {key: value}
    if nesting == "resolved_identity":
        return {"resolved_identity": reserved}
    if nesting == "arbitrary_mapping":
        return {"arbitrary": {"inner": reserved}}
    if nesting == "list_tuple_wrapped":
        return {"wrapped": [{"inner": (reserved,)}]}
    raise AssertionError(f"unexpected nesting {nesting!r}")


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("voc_q_regression_loss", "smooth_l1_beta1"),
        ("voc_q_regression_loss", "half_squared_td"),
        ("voc_q_regression_loss", None),
        (
            "voc_q_reconstruction",
            "detached_value_plus_raw_head_mean_plus_policy_centered_raw_head",
        ),
        ("voc_q_reconstruction", "forged"),
        ("voc_q_reconstruction", None),
    ],
)
@pytest.mark.parametrize(
    "nesting", ["resolved_identity", "arbitrary_mapping", "list_tuple_wrapped"]
)
def test_schema10_actor_route_rejects_nested_reserved_identity_before_state_use(
    monkeypatch, nesting, key, value
):
    flags = _schema10_embedded_flags()
    actor = _terminal_checkpoint(flags)
    actor.update(_nested_schema10_reserved_identity(nesting, key, value))
    monkeypatch.setattr(
        util,
        "_validate_voc_atomic_final_actor_checkpoint",
        lambda *_args, **_kwargs: pytest.fail(
            "nested schema-10 identity reached actor state validation"
        ),
    )
    with pytest.raises(ValueError, match=rf"reserved schema-10.*{key}"):
        util.validate_voc_schema10_final_actor_checkpoint(
            actor, util.argparse.Namespace(**flags)
        )


def test_schema10_reserved_identity_scan_terminates_on_benign_cycles():
    checkpoint = {}
    nested = []
    checkpoint["nested"] = nested
    checkpoint["self"] = checkpoint
    nested.extend([checkpoint, nested, ({"benign": "value"},)])
    assert (
        util._reject_schema10_persisted_derived_identity(
            checkpoint, label="cyclic schema-10 checkpoint"
        )
        is None
    )


@pytest.mark.parametrize(
    ("checkpoint_name", "key", "value"),
    [
        ("ckp_actor.tar", "voc_q_regression_loss", "smooth_l1_beta1"),
        (
            "ckp_actor.tar",
            "voc_q_reconstruction",
            "detached_value_plus_raw_head_mean_plus_policy_centered_raw_head",
        ),
        ("ckp_model.tar", "voc_q_regression_loss", "smooth_l1_beta1"),
        (
            "ckp_model.tar",
            "voc_q_reconstruction",
            "detached_value_plus_raw_head_mean_plus_policy_centered_raw_head",
        ),
    ],
)
def test_schema10_final_bundle_rejects_reserved_checkpoint_identity_before_state_use(
    monkeypatch, tmp_path, checkpoint_name, key, value
):
    _write_schema10_final_bundle(tmp_path)
    path = tmp_path / checkpoint_name
    checkpoint = torch.load(path, weights_only=False)
    checkpoint[key] = value
    torch.save(checkpoint, path)
    monkeypatch.setattr(
        util,
        "validate_actor_policy_checkpoint",
        lambda *_args, **_kwargs: pytest.fail(
            "schema-10 reserved identity reached tensor/state validation"
        ),
    )
    with pytest.raises(ValueError, match=rf"reserved schema-10.*{key}"):
        util.validate_schema10_final_bundle(tmp_path)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("voc_q_regression_loss", "smooth_l1_beta1"),
        ("voc_q_regression_loss", "half_squared_td"),
        ("voc_q_regression_loss", None),
        (
            "voc_q_reconstruction",
            "detached_value_plus_raw_head_mean_plus_policy_centered_raw_head",
        ),
        ("voc_q_reconstruction", "forged"),
        ("voc_q_reconstruction", None),
    ],
)
@pytest.mark.parametrize(
    "nesting", ["resolved_identity", "arbitrary_mapping", "list_tuple_wrapped"]
)
@pytest.mark.parametrize("checkpoint_name", ["ckp_actor.tar", "ckp_model.tar"])
def test_schema10_final_bundle_rejects_nested_reserved_checkpoint_identity(
    monkeypatch, tmp_path, checkpoint_name, nesting, key, value
):
    _write_schema10_final_bundle(tmp_path)
    path = tmp_path / checkpoint_name
    checkpoint = torch.load(path, weights_only=False)
    checkpoint.update(_nested_schema10_reserved_identity(nesting, key, value))
    torch.save(checkpoint, path)
    monkeypatch.setattr(
        util,
        "validate_actor_policy_checkpoint",
        lambda *_args, **_kwargs: pytest.fail(
            "nested schema-10 identity reached tensor/state validation"
        ),
    )
    with pytest.raises(ValueError, match=rf"reserved schema-10.*{key}"):
        util.validate_schema10_final_bundle(tmp_path)


def test_schema10_final_bundle_rejects_cross_surface_stage_drift(tmp_path):
    _write_schema10_final_bundle(tmp_path)
    actor = torch.load(tmp_path / "ckp_actor.tar", weights_only=False)
    actor["flags"]["base_seed"] = 5
    torch.save(actor, tmp_path / "ckp_actor.tar")
    with pytest.raises(
        ValueError,
        match="unregistered schema-10 stage|cross-surface identity",
    ):
        util.validate_schema10_final_bundle(tmp_path)


def test_schema10_final_bundle_accepts_real_closed_wire_stage(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        util, "_validate_schema10_stage_profile", _REAL_SCHEMA10_STAGE_VALIDATOR
    )
    xpid = util.VOC_GATE_POLICY_SCHEMA10_STAGE_PROFILES[0][0]
    ckpdir = tmp_path / xpid
    ckpdir.mkdir()
    _write_schema10_final_bundle(ckpdir)
    evidence = util.validate_schema10_final_bundle(ckpdir)
    assert tuple(evidence["resolved_identity"]["stage"]) == (
        util.VOC_GATE_POLICY_SCHEMA10_STAGE_PROFILES[0]
    )


@pytest.mark.parametrize("drain_count", [0, 1])
def test_schema11_final_bundle_has_exact_derived_only_twelve_key_identity(
    tmp_path, drain_count
):
    _write_schema11_final_bundle(tmp_path, drain_count=drain_count)
    bundle = util.validate_schema11_final_bundle(tmp_path)
    identity = bundle["resolved_identity"]
    assert tuple(identity) == (
        "key_count",
        "v12_projection_key_count",
        "v12_projection_sha256",
        "complete_surface_sha256",
        "stage",
        "paths",
        "gate_schema",
        "voc_gate_policy_schema_version",
        "voc_model_input_seal_schema_version",
        "voc_q_regression_loss",
        "voc_q_reconstruction",
        "voc_q_optimizer_coordinates",
    )
    assert identity["key_count"] == 229
    assert identity["v12_projection_key_count"] == 209
    assert identity["v12_projection_sha256"] == (
        util.VOC_GATE_POLICY_SCHEMA6_V12_BASELINE_SHA256
    )
    assert identity["gate_schema"] == 11
    assert identity["voc_gate_policy_schema_version"] == 11
    assert identity["voc_model_input_seal_schema_version"] == 1
    assert identity["voc_q_regression_loss"] == "smooth_l1_beta1"
    assert identity["voc_q_reconstruction"] == (
        "detached_value_plus_raw_head_mean_plus_policy_centered_raw_head"
    )
    assert identity["voc_q_optimizer_coordinates"] == (
        "orthonormal_common_difference_adam"
    )
    assert bundle["model_input_seal"][
        "voc_model_terminal_drain_update_count"
    ] == drain_count
    json.dumps(bundle, allow_nan=False)
    with (tmp_path / "config_c.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    actor = torch.load(tmp_path / "ckp_actor.tar", weights_only=False)
    model = torch.load(tmp_path / "ckp_model.tar", weights_only=False)
    for surface in (config, actor["flags"], model["flags"]):
        assert len(surface) == 229
        assert set(surface) == util._VOC_GATE_POLICY_SCHEMA11_COMPLETE_KEYS
        assert not set(util._VOC_GATE_POLICY_SCHEMA11_DERIVED_IDENTITY_KEYS) & set(
            surface
        )
    for checkpoint in (actor, model):
        assert not set(util._VOC_GATE_POLICY_SCHEMA11_DERIVED_IDENTITY_KEYS) & set(
            checkpoint
        )


def test_schema11_dedicated_routes_reject_schema10(tmp_path):
    schema10 = tmp_path / "schema10"
    schema10.mkdir()
    _write_schema10_final_bundle(schema10)
    with pytest.raises(ValueError, match="not schema 11"):
        util.validate_schema11_final_bundle(schema10)
    flags = _schema11_embedded_flags()
    validated = util.validate_voc_schema11_final_actor_checkpoint(
        _terminal_checkpoint(flags), util.argparse.Namespace(**flags)
    )
    assert validated["voc_gate_policy_schema_version"] == 11
    assert validated["voc_q_optimizer_coordinates"] == (
        "orthonormal_common_difference_adam"
    )
    schema10_flags = _schema10_embedded_flags()
    with pytest.raises(ValueError, match="schema-11|schema 11"):
        util.validate_voc_schema11_final_actor_checkpoint(
            _terminal_checkpoint(schema10_flags),
            util.argparse.Namespace(**schema10_flags),
        )


_SCHEMA11_RESERVED_CASES = (
    ("voc_q_regression_loss", "smooth_l1_beta1"),
    ("voc_q_regression_loss", "half_squared_td"),
    ("voc_q_regression_loss", None),
    (
        "voc_q_reconstruction",
        "detached_value_plus_raw_head_mean_plus_policy_centered_raw_head",
    ),
    ("voc_q_reconstruction", "forged"),
    ("voc_q_reconstruction", None),
    ("voc_q_optimizer_coordinates", "orthonormal_common_difference_adam"),
    ("voc_q_optimizer_coordinates", "forged"),
    ("voc_q_optimizer_coordinates", None),
)


def _nested_schema11_reserved_identity(nesting, key, value):
    reserved = {key: value}
    if nesting == "top":
        return reserved
    if nesting == "resolved_identity":
        return {"resolved_identity": reserved}
    if nesting == "arbitrary_mapping":
        return {"arbitrary": {"inner": reserved}}
    if nesting == "list_tuple_wrapped":
        return {"wrapped": [{"inner": (reserved,)}]}
    raise AssertionError(f"unexpected nesting {nesting!r}")


@pytest.mark.parametrize(("key", "value"), _SCHEMA11_RESERVED_CASES)
@pytest.mark.parametrize(
    "nesting", ["top", "resolved_identity", "arbitrary_mapping", "list_tuple_wrapped"]
)
def test_schema11_actor_route_rejects_reserved_identity_before_state_use(
    monkeypatch, nesting, key, value
):
    flags = _schema11_embedded_flags()
    actor = _terminal_checkpoint(flags)
    actor.update(_nested_schema11_reserved_identity(nesting, key, value))
    monkeypatch.setattr(
        util,
        "_validate_voc_atomic_final_actor_checkpoint",
        lambda *_args, **_kwargs: pytest.fail(
            "schema-11 reserved identity reached actor state validation"
        ),
    )
    with pytest.raises(ValueError, match=rf"reserved schema-11.*{key}"):
        util.validate_voc_schema11_final_actor_checkpoint(
            actor, util.argparse.Namespace(**flags)
        )


@pytest.mark.parametrize(("key", "value"), _SCHEMA11_RESERVED_CASES)
@pytest.mark.parametrize(
    "nesting", ["top", "resolved_identity", "list_tuple_wrapped"]
)
@pytest.mark.parametrize("checkpoint_name", ["ckp_actor.tar", "ckp_model.tar"])
def test_schema11_final_bundle_rejects_reserved_identity_before_state_use(
    monkeypatch, tmp_path, checkpoint_name, nesting, key, value
):
    _write_schema11_final_bundle(tmp_path)
    path = tmp_path / checkpoint_name
    checkpoint = torch.load(path, weights_only=False)
    checkpoint.update(_nested_schema11_reserved_identity(nesting, key, value))
    torch.save(checkpoint, path)
    monkeypatch.setattr(
        util,
        "validate_actor_policy_checkpoint",
        lambda *_args, **_kwargs: pytest.fail(
            "schema-11 reserved identity reached tensor/state validation"
        ),
    )
    with pytest.raises(ValueError, match=rf"reserved schema-11.*{key}"):
        util.validate_schema11_final_bundle(tmp_path)


def test_schema11_reserved_identity_scan_terminates_on_benign_cycles():
    checkpoint = {}
    nested = []
    checkpoint["nested"] = nested
    checkpoint["self"] = checkpoint
    nested.extend([checkpoint, nested, ({"benign": "value"},)])
    assert util._reject_schema11_persisted_derived_identity(
        checkpoint, label="cyclic schema-11 checkpoint"
    ) is None


def test_schema11_final_bundle_accepts_real_closed_wire_stage(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        util, "_validate_schema11_stage_profile", _REAL_SCHEMA11_STAGE_VALIDATOR
    )
    xpid = util.VOC_GATE_POLICY_SCHEMA11_STAGE_PROFILES[0][0]
    ckpdir = tmp_path / xpid
    ckpdir.mkdir()
    _write_schema11_final_bundle(ckpdir)
    evidence = util.validate_schema11_final_bundle(ckpdir)
    assert tuple(evidence["resolved_identity"]["stage"]) == (
        util.VOC_GATE_POLICY_SCHEMA11_STAGE_PROFILES[0]
    )


@pytest.mark.parametrize("drain_count", [0, 1])
def test_schema12_final_bundle_has_exact_tau1_identity_and_projection(
    tmp_path, drain_count
):
    _write_schema12_final_bundle(tmp_path, drain_count=drain_count)
    bundle = util.validate_schema12_final_bundle(tmp_path)
    identity = bundle["resolved_identity"]
    assert tuple(identity) == (
        "key_count",
        "v12_projection_key_count",
        "v12_projection_sha256",
        "complete_surface_sha256",
        "stage",
        "paths",
        "gate_schema",
        "voc_gate_policy_schema_version",
        "voc_model_input_seal_schema_version",
        "voc_q_regression_loss",
        "voc_q_reconstruction",
        "voc_q_optimizer_coordinates",
    )
    assert identity["key_count"] == 229
    assert identity["v12_projection_key_count"] == 209
    assert identity["v12_projection_sha256"] == (
        util.VOC_GATE_POLICY_SCHEMA12_V12_PROJECTION_SHA256
    )
    assert identity["gate_schema"] == 12
    assert identity["voc_gate_policy_schema_version"] == 12
    assert identity["voc_q_regression_loss"] == "smooth_l1_beta1"
    assert identity["voc_q_optimizer_coordinates"] == (
        "orthonormal_common_difference_adam"
    )
    assert bundle["actor_training_state"]["voc_update_count"] == 1
    assert bundle["model_input_seal"][
        "voc_model_terminal_drain_update_count"
    ] == drain_count
    with (tmp_path / "config_c.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    actor = torch.load(tmp_path / "ckp_actor.tar", weights_only=False)
    model = torch.load(tmp_path / "ckp_model.tar", weights_only=False)
    for surface in (config, actor["flags"], model["flags"]):
        assert len(surface) == 229
        assert set(surface) == util._VOC_GATE_POLICY_SCHEMA12_COMPLETE_KEYS
        assert surface["voc_gate_target_tau"] == 1.0
        assert not set(util._VOC_GATE_POLICY_SCHEMA12_DERIVED_IDENTITY_KEYS) & set(
            surface
        )


def test_schema12_bundle_ack_and_history_keep_exact_strict_shapes():
    state = OrderedDict(weight=torch.zeros(1, 1), bias=torch.zeros(1))
    bundle = util.make_actor_policy_bundle(
        state, 3, terminal=True, gate_schema=12
    )
    ack = util.make_actor_policy_ack(0, 3, terminal=True, gate_schema=12)
    assert set(bundle) == {
        "bundle_schema_version",
        "policy_version",
        "terminal",
        "gate_schema",
        "actor_state_dict",
    }
    assert set(ack) == {
        "bundle_schema_version",
        "gate_schema",
        "rank",
        "policy_version",
        "terminal",
    }
    assert util.validate_actor_policy_bundle(
        bundle,
        expected_epoch=3,
        expected_terminal=True,
        expected_actor_state=state,
        expected_gate_schema=12,
    )["gate_schema"] == 12
    assert util.validate_actor_policy_ack(
        ack, expected_gate_schema=12
    )["gate_schema"] == 12
    for bad in (True, np.int64(12), 12.0, "12"):
        with pytest.raises(ValueError, match="gate_schema"):
            util.make_actor_policy_bundle(state, 3, gate_schema=bad)


def test_schema12_dedicated_routes_reject_schema11(tmp_path):
    schema11 = tmp_path / "schema11"
    schema11.mkdir()
    _write_schema11_final_bundle(schema11)
    with pytest.raises(ValueError, match="not schema 12"):
        util.validate_schema12_final_bundle(schema11)
    flags = _schema12_embedded_flags()
    validated = util.validate_voc_schema12_final_actor_checkpoint(
        _terminal_checkpoint(flags), util.argparse.Namespace(**flags)
    )
    assert validated["voc_gate_policy_schema_version"] == 12
    schema11_flags = _schema11_embedded_flags()
    with pytest.raises(ValueError, match="schema-12|schema 12"):
        util.validate_voc_schema12_final_actor_checkpoint(
            _terminal_checkpoint(schema11_flags),
            util.argparse.Namespace(**schema11_flags),
        )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("voc_q_regression_loss", "smooth_l1_beta1"),
        ("voc_q_reconstruction", "forged"),
        ("voc_q_optimizer_coordinates", None),
    ],
)
def test_schema12_actor_route_rejects_nested_reserved_identity_before_state_use(
    monkeypatch, key, value
):
    flags = _schema12_embedded_flags()
    checkpoint = _terminal_checkpoint(flags)
    checkpoint["resolved_identity"] = {"wrapped": [({key: value},)]}
    monkeypatch.setattr(
        util,
        "_validate_voc_atomic_final_actor_checkpoint",
        lambda *_args, **_kwargs: pytest.fail(
            "schema-12 reserved identity reached actor state validation"
        ),
    )
    with pytest.raises(ValueError, match=rf"reserved schema-12.*{key}"):
        util.validate_voc_schema12_final_actor_checkpoint(
            checkpoint, util.argparse.Namespace(**flags)
        )


@pytest.mark.parametrize("checkpoint_name", ["ckp_actor.tar", "ckp_model.tar"])
@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("voc_q_regression_loss", "smooth_l1_beta1"),
        ("voc_q_reconstruction", None),
        ("voc_q_optimizer_coordinates", "forged"),
    ],
)
def test_schema12_final_bundle_rejects_nested_reserved_identity_before_state_use(
    monkeypatch, tmp_path, checkpoint_name, key, value
):
    _write_schema12_final_bundle(tmp_path)
    path = tmp_path / checkpoint_name
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["wrapped"] = [{"inner": ({key: value},)}]
    torch.save(checkpoint, path)
    monkeypatch.setattr(
        util,
        "validate_actor_policy_checkpoint",
        lambda *_args, **_kwargs: pytest.fail(
            "schema-12 reserved identity reached tensor/state validation"
        ),
    )
    with pytest.raises(ValueError, match=rf"reserved schema-12.*{key}"):
        util.validate_schema12_final_bundle(tmp_path)


def test_schema12_reserved_identity_scan_terminates_on_benign_cycles():
    checkpoint = {}
    nested = []
    checkpoint["nested"] = nested
    checkpoint["self"] = checkpoint
    nested.extend([checkpoint, nested, ({"benign": "value"},)])
    assert util._reject_schema12_persisted_derived_identity(
        checkpoint, label="cyclic schema-12 checkpoint"
    ) is None


@pytest.mark.parametrize("name", ["weight", "bias"])
def test_schema12_positive_update_rejects_raw_ema_online_mismatch(name):
    flags = _schema12_embedded_flags()
    checkpoint = _terminal_checkpoint(flags)
    checkpoint["voc_ema_gate_head_state_dict"][name] = (
        checkpoint["voc_ema_gate_head_state_dict"][name].clone()
    )
    checkpoint["voc_ema_gate_head_state_dict"][name].view(-1)[0] += 1.0
    with pytest.raises(ValueError, match=rf"raw EMA {name} disagrees"):
        util.validate_voc_schema12_final_actor_checkpoint(
            checkpoint, util.argparse.Namespace(**flags)
        )


@pytest.mark.parametrize("bad_count", [True, np.int64(1), 1.0, "1", -1])
def test_schema12_raw_ema_update_count_requires_exact_builtin_int(bad_count):
    flags = _schema12_embedded_flags()
    checkpoint = _terminal_checkpoint(flags)
    checkpoint["voc_ema_gate_update_count"] = bad_count
    with pytest.raises(ValueError, match="exact non-negative Python integer"):
        util.validate_voc_schema12_final_actor_checkpoint(
            checkpoint, util.argparse.Namespace(**flags)
        )


def test_schema12_raw_ema_online_equality_allows_signed_zero_bytes():
    checkpoint = {
        "voc_ema_gate_update_count": 1,
        "voc_gate_target_tau": 1.0,
        "voc_ema_gate_head_state_dict": {
            "weight": torch.tensor([[-0.0], [1.0]], dtype=torch.float32),
            "bias": torch.tensor([-0.0, 1.0], dtype=torch.float32),
        },
        "actor_net_state_dict": {
            "voc_head.weight": torch.tensor([[0.0], [1.0]], dtype=torch.float32),
            "voc_head.bias": torch.tensor([0.0, 1.0], dtype=torch.float32),
        },
    }
    assert util._validate_schema12_raw_ema_online_equality(
        checkpoint, label="signed-zero schema-12 checkpoint"
    ) is None


def test_schema12_final_bundle_accepts_real_closed_wire_stage(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        util, "_validate_schema12_stage_profile", _REAL_SCHEMA12_STAGE_VALIDATOR
    )
    xpid = util.VOC_GATE_POLICY_SCHEMA12_STAGE_PROFILES[0][0]
    ckpdir = tmp_path / xpid
    ckpdir.mkdir()
    _write_schema12_final_bundle(ckpdir)
    evidence = util.validate_schema12_final_bundle(ckpdir)
    assert tuple(evidence["resolved_identity"]["stage"]) == (
        util.VOC_GATE_POLICY_SCHEMA12_STAGE_PROFILES[0]
    )


@pytest.mark.parametrize("drain_count", [0, 1])
def test_schema13_final_bundle_is_exact13_with_separate_telemetry_evidence(
    monkeypatch, tmp_path, drain_count
):
    ckpdir = tmp_path / util.VOC_GATE_POLICY_SCHEMA13_STAGE_PROFILES[0][0]
    ckpdir.mkdir()
    _write_schema13_final_bundle(ckpdir, drain_count=drain_count)
    monkeypatch.setattr(
        util,
        "validate_schema13_telemetry_manifest",
        _fake_schema13_telemetry_manifest,
    )
    real_torch_load = torch.load
    bound_load_sources = []

    def require_bound_payload(source, *args, **kwargs):
        bound_load_sources.append(source)
        assert isinstance(source, io.BytesIO)
        return real_torch_load(source, *args, **kwargs)

    monkeypatch.setattr(util.torch, "load", require_bound_payload)
    bundle = util.validate_schema13_final_bundle(ckpdir)
    assert len(bound_load_sources) == 3
    assert len(bundle) == 13
    assert set(bundle) == {
        "completion_evidence",
        "actor_policy",
        "actor_training_state",
        "model_input_seal",
        "resolved_identity",
        "config_use_wandb",
        "model_real_step",
        "model_step",
        "model_state_tensor_count",
        "model_optimizer_state",
        "model_scheduler_state",
        "model_scaler_state",
        "telemetry",
    }
    identity = bundle["resolved_identity"]
    assert len(identity) == 12
    assert identity["gate_schema"] == 13
    assert identity["voc_gate_policy_schema_version"] == 13
    assert identity["v12_projection_key_count"] == 209
    assert identity["v12_projection_sha256"] == (
        util.VOC_GATE_POLICY_SCHEMA12_V12_PROJECTION_SHA256
    )
    assert bundle["telemetry"]["telemetry_schema_version"] == 1
    assert bundle["telemetry"]["gate_schema"] == 13
    assert set(bundle["completion_evidence"]["checkpoint_files"]) == set(
        util._SCHEMA13_COMPLETION_CHECKPOINT_FILES
    )
    assert len(bundle["completion_evidence"]["implementation_sources"]) == 15
    assert bundle["model_input_seal"][
        "voc_model_terminal_drain_update_count"
    ] == drain_count


def test_schema13_dedicated_actor_route_and_cross_schema_rejection():
    flags = _schema13_embedded_flags()
    actor = _terminal_checkpoint(flags)
    evidence = util.validate_voc_schema13_final_actor_checkpoint(
        actor, util.argparse.Namespace(**flags)
    )
    assert evidence["voc_gate_policy_schema_version"] == 13
    assert evidence["voc_q_regression_loss"] == "smooth_l1_beta1"
    assert evidence["voc_q_reconstruction"] == (
        "detached_value_plus_raw_head_mean_plus_policy_centered_raw_head"
    )
    assert evidence["voc_q_optimizer_coordinates"] == (
        "orthonormal_common_difference_adam"
    )
    schema12_flags = _schema12_embedded_flags()
    with pytest.raises(ValueError, match="schema-13|schema 13"):
        util.validate_voc_schema13_final_actor_checkpoint(
            _terminal_checkpoint(schema12_flags),
            util.argparse.Namespace(**schema12_flags),
        )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("voc_q_regression_loss", "smooth_l1_beta1"),
        ("voc_q_reconstruction", "forged"),
        ("voc_q_optimizer_coordinates", None),
    ],
)
def test_schema13_actor_rejects_nested_reserved_identity_before_state_use(
    monkeypatch, key, value
):
    flags = _schema13_embedded_flags()
    checkpoint = _terminal_checkpoint(flags)
    checkpoint["wrapped"] = [{"inner": ({key: value},)}]
    monkeypatch.setattr(
        util,
        "_validate_voc_atomic_final_actor_checkpoint",
        lambda *_args, **_kwargs: pytest.fail(
            "schema-13 reserved identity reached actor state validation"
        ),
    )
    with pytest.raises(ValueError, match=rf"reserved schema-13.*{key}"):
        util.validate_voc_schema13_final_actor_checkpoint(
            checkpoint, util.argparse.Namespace(**flags)
        )


@pytest.mark.parametrize("name", ["weight", "bias"])
def test_schema13_positive_update_rejects_raw_ema_online_mismatch(name):
    flags = _schema13_embedded_flags()
    checkpoint = _terminal_checkpoint(flags)
    checkpoint["voc_ema_gate_head_state_dict"][name] = (
        checkpoint["voc_ema_gate_head_state_dict"][name].clone()
    )
    checkpoint["voc_ema_gate_head_state_dict"][name].view(-1)[0] += 1.0
    with pytest.raises(ValueError, match=rf"raw EMA {name} disagrees"):
        util.validate_voc_schema13_final_actor_checkpoint(
            checkpoint, util.argparse.Namespace(**flags)
        )


def test_schema13_bundle_ack_and_history_keep_exact_strict_shapes():
    state = OrderedDict(weight=torch.zeros(1, 1), bias=torch.zeros(1))
    bundle = util.make_actor_policy_bundle(
        state, 3, terminal=True, gate_schema=13
    )
    ack = util.make_actor_policy_ack(0, 3, terminal=True, gate_schema=13)
    assert len(bundle) == 5
    assert len(ack) == 5
    assert util.validate_actor_policy_bundle(
        bundle,
        expected_epoch=3,
        expected_terminal=True,
        expected_actor_state=state,
        expected_gate_schema=13,
    )["gate_schema"] == 13
    assert util.validate_actor_policy_ack(
        ack, expected_gate_schema=13
    )["gate_schema"] == 13
    for bad in (True, np.int64(13), 13.0, "13"):
        with pytest.raises(ValueError, match="gate_schema"):
            util.make_actor_policy_bundle(state, 3, gate_schema=bad)


def test_schema13_final_bundle_accepts_real_closed_wire_stage(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        util, "_validate_schema13_stage_profile", _REAL_SCHEMA13_STAGE_VALIDATOR
    )
    monkeypatch.setattr(
        util,
        "validate_schema13_telemetry_manifest",
        _fake_schema13_telemetry_manifest,
    )
    xpid = util.VOC_GATE_POLICY_SCHEMA13_STAGE_PROFILES[0][0]
    ckpdir = tmp_path / xpid
    ckpdir.mkdir()
    _write_schema13_final_bundle(ckpdir)
    evidence = util.validate_schema13_final_bundle(ckpdir)
    assert tuple(evidence["resolved_identity"]["stage"]) == (
        util.VOC_GATE_POLICY_SCHEMA13_STAGE_PROFILES[0]
    )


def test_schema13_finalize_dispatches_schema2_exact4_before_public_finish(
    monkeypatch, tmp_path
):
    actor_policy = util.validate_actor_policy_checkpoint(
        _terminal_checkpoint(_schema13_embedded_flags())
    )
    checkpoint_files = {
        name: {"sha256": chr(ord("a") + index) * 64, "size": index + 1}
        for index, name in enumerate(
            util._SCHEMA13_COMPLETION_CHECKPOINT_FILES
        )
    }
    completion_evidence = {
        "checkpoint_files": checkpoint_files,
        "implementation_sources": {
            name: {"sha256": "e" * 64}
            for name in util._SCHEMA13_TRAINING_IMPLEMENTATION_FILES
        },
        "loaded_extensions": {"thinker/cenv.so": {"sha256": "f" * 64}},
    }
    final_bundle = {
        "actor_policy": actor_policy,
        "config_use_wandb": False,
        "completion_evidence": completion_evidence,
        "resolved_identity": {"gate_schema": 13},
        "telemetry": {"manifest_sha256": checkpoint_files[
            "voc_telemetry_manifest.json"
        ]["sha256"]},
    }
    monkeypatch.setattr(
        train_driver.util,
        "validate_schema13_final_bundle",
        lambda _path: final_bundle,
    )
    monkeypatch.setattr(
        train_driver.util,
        "validate_schema6_final_bundle",
        lambda *_args, **_kwargs: pytest.fail("schema-13 used legacy validator"),
    )
    calls = []

    def write_completion(path, **kwargs):
        calls.append((path, kwargs))
        return {"schema_version": 2, "status": "complete"}

    monkeypatch.setattr(
        train_driver.util, "write_run_completion", write_completion
    )
    flags = SimpleNamespace(
        ckpdir=str(tmp_path),
        use_wandb=False,
        voc_actor_policy_barrier_runtime=True,
        voc_gate_policy_schema_version=13,
    )
    assert train_driver.finalize_run(flags) == {
        "schema_version": 2,
        "status": "complete",
    }
    assert len(calls) == 1
    path, kwargs = calls[0]
    assert path == str(tmp_path)
    assert kwargs["expected_evidence"] == completion_evidence
    assert kwargs["validated_actor_policy"] == actor_policy
    assert kwargs["completion_schema_version"] == 2
    assert kwargs["gate_schema"] == 13
    logger_completion = kwargs["actor_policy_logger_completion"]
    assert len(logger_completion) == 10
    assert logger_completion["schema_version"] == 2
    assert logger_completion["required"] is False
    assert logger_completion["checkpoint_files"] == checkpoint_files


@pytest.mark.parametrize(
    "field", util._SCHEMA7_MODEL_INPUT_SEAL_EVIDENCE_FIELDS
)
def test_schema7_final_bundle_requires_every_model_seal_evidence_field(
    tmp_path, field
):
    _write_schema7_final_bundle(tmp_path)
    model = torch.load(tmp_path / "ckp_model.tar", weights_only=False)
    del model[field]
    torch.save(model, tmp_path / "ckp_model.tar")
    with pytest.raises(ValueError, match="model-input seal evidence"):
        util.validate_schema7_final_bundle(tmp_path)


def test_schema7_final_bundle_rejects_extra_model_seal_evidence_field(tmp_path):
    _write_schema7_final_bundle(tmp_path)
    model = torch.load(tmp_path / "ckp_model.tar", weights_only=False)
    model["voc_model_unregistered_terminal_evidence"] = 0
    torch.save(model, tmp_path / "ckp_model.tar")
    with pytest.raises(ValueError, match="extra=.*unregistered"):
        util.validate_schema7_final_bundle(tmp_path)


@pytest.mark.parametrize(
    "field", util._SCHEMA7_MODEL_INPUT_SEAL_EVIDENCE_FIELDS
)
def test_schema7_final_bundle_rejects_evidence_type_and_nextafter_drift(
    tmp_path, field
):
    _write_schema7_final_bundle(tmp_path)
    model = torch.load(tmp_path / "ckp_model.tar", weights_only=False)
    if field == "voc_model_input_sealed":
        model[field] = 1
    else:
        model[field] = np.nextafter(float(model[field]), np.inf)
    torch.save(model, tmp_path / "ckp_model.tar")
    with pytest.raises(ValueError, match=field):
        util.validate_schema7_final_bundle(tmp_path)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"voc_model_input_seal_schema_version": 0}, "schema version 1"),
        ({"voc_model_input_sealed": False}, "Python bool True"),
        ({"voc_model_input_seal_count": 2}, "exactly one"),
        ({"voc_model_terminal_processed_n": 1201}, "processed count"),
        ({"voc_model_terminal_drain_update_count": 2}, "exactly 0 or 1"),
        ({"voc_model_terminal_drain_pre_real_step": -1}, "pre-real-step"),
        (
            {"voc_model_terminal_drain_pre_grad_step_count_m": -1},
            "pre-update counts",
        ),
        (
            {"voc_model_terminal_drain_pre_grad_step_count_p": -1},
            "pre-update counts",
        ),
        ({"voc_model_input_late_write_count": 1}, "zero late"),
        ({"voc_model_input_abort_count": 1}, "zero model-input aborts"),
        (
            {"voc_model_terminal_drain_pre_real_step": 1199},
            "zero-drain branch",
        ),
        (
            {
                "voc_model_terminal_drain_update_count": 1,
                "voc_model_terminal_drain_pre_real_step": 1200,
                "voc_model_terminal_drain_pre_grad_step_count_m": 0,
                "voc_model_terminal_drain_pre_grad_step_count_p": 0,
            },
            "one-drain branch",
        ),
    ],
)
def test_schema7_final_bundle_rejects_model_seal_invariant_drift(
    tmp_path, overrides, match
):
    _write_schema7_final_bundle(tmp_path, model_overrides=overrides)
    with pytest.raises(ValueError, match=match):
        util.validate_schema7_final_bundle(tmp_path)


@pytest.mark.parametrize("checkpoint_name", ["ckp_actor.tar", "ckp_model.tar"])
def test_schema7_final_bundle_rejects_cross_surface_drift(
    tmp_path, checkpoint_name
):
    _write_schema7_final_bundle(tmp_path)
    checkpoint = torch.load(tmp_path / checkpoint_name, weights_only=False)
    checkpoint["flags"]["cmd"] = "train.py --different-schema7-command"
    torch.save(checkpoint, tmp_path / checkpoint_name)
    with pytest.raises(ValueError, match="surfaces differ"):
        util.validate_schema7_final_bundle(tmp_path)


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("discounting", 0.5),
        ("actor_grad_norm_clipping", 0.25),
        ("model_batch_size", 64),
        ("icopro_margin", 2.0),
        ("critic_zero_init", False),
        ("ray_cpu", 15),
    ],
)
def test_schema6_final_rejects_consistent_v12_baseline_rewrite(
    tmp_path, field, bad
):
    _write_schema6_final_bundle(tmp_path, use_wandb=False)
    with (tmp_path / "config_c.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    actor = torch.load(tmp_path / "ckp_actor.tar", weights_only=False)
    model = torch.load(tmp_path / "ckp_model.tar", weights_only=False)
    config[field] = bad
    actor["flags"][field] = bad
    model["flags"][field] = bad
    with (tmp_path / "config_c.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)
    torch.save(actor, tmp_path / "ckp_actor.tar")
    torch.save(model, tmp_path / "ckp_model.tar")
    with pytest.raises(ValueError, match=field):
        util.validate_schema6_final_bundle(tmp_path)


@pytest.mark.parametrize("mutation", ["missing", "extra", "typed"])
def test_schema6_final_rejects_noncanonical_228_key_surface(tmp_path, mutation):
    _write_schema6_final_bundle(tmp_path, use_wandb=False)
    actor = torch.load(tmp_path / "ckp_actor.tar", weights_only=False)
    if mutation == "missing":
        del actor["flags"]["discounting"]
    elif mutation == "extra":
        actor["flags"]["unexpected_schema6_field"] = 1
    else:
        actor["flags"]["model_batch_size"] = 32.0
    torch.save(actor, tmp_path / "ckp_actor.tar")
    with pytest.raises(ValueError, match="228-key|model_batch_size"):
        util.validate_schema6_final_bundle(tmp_path)


@pytest.mark.parametrize("corruption", ["model_nonfinite", "config_identity"])
def test_schema6_final_bundle_rejects_model_and_config_corruption(
    tmp_path, corruption
):
    flags = _write_schema6_final_bundle(tmp_path, use_wandb=False)
    if corruption == "model_nonfinite":
        model = torch.load(
            tmp_path / "ckp_model.tar", weights_only=False
        )
        model["model_net_state_dict"]["vp_net.weight"][0, 0] = float("nan")
        torch.save(model, tmp_path / "ckp_model.tar")
    else:
        flags["actor_batch_size"] = 32
        with (tmp_path / "config_c.yaml").open(
            "w", encoding="utf-8"
        ) as handle:
            yaml.safe_dump(flags, handle, sort_keys=True)
    with pytest.raises(ValueError):
        util.validate_schema6_final_bundle(tmp_path)


def test_schema6_wandb_off_malformed_terminal_cannot_commit_public_finish(
    tmp_path
):
    _write_schema6_final_bundle(tmp_path, use_wandb=False)
    actor = torch.load(tmp_path / "ckp_actor.tar", weights_only=False)
    actor["voc_actor_policy_terminal"] = np.bool_(True)
    torch.save(actor, tmp_path / "ckp_actor.tar")
    flags = SimpleNamespace(
        ckpdir=str(tmp_path),
        use_wandb=False,
        voc_actor_policy_barrier_runtime=True,
    )
    with pytest.raises(ValueError, match="terminal"):
        train_driver.finalize_run(flags)
    assert not (tmp_path / "finish").exists()


@pytest.mark.parametrize("return_codes", [[1], [True, True], [], [object()]])
def test_schema6_worker_results_require_exact_true_and_exact_count(return_codes):
    with pytest.raises(RuntimeError, match="exactly True"):
        train_driver.validate_schema6_worker_results(return_codes, 1)
    assert train_driver.validate_schema6_worker_results([True], 1) is True


def test_hung_worker_with_healthy_pending_logger_keeps_worker_timeout_attribution(
    monkeypatch,
):
    worker_ref = object()
    logger_ref = object()
    cancelled = []
    times = iter([0.0, 121.0])
    monkeypatch.setattr(
        train_driver.ray, "wait", lambda refs, **_kwargs: ([], refs)
    )
    monkeypatch.setattr(
        train_driver.ray,
        "cancel",
        lambda ref, force: cancelled.append((ref, force)),
    )
    with pytest.raises(TimeoutError, match="no policy heartbeat progress"):
        train_driver.wait_for_schema6_workers(
            [worker_ref],
            SimpleNamespace(),
            logger_ref=logger_ref,
            monotonic=lambda: next(times),
        )
    assert cancelled == [(worker_ref, False)]


def test_schema6_logger_ack_rejects_checkpoint_swap_during_close(
    monkeypatch, tmp_path
):
    _write_schema6_final_bundle(tmp_path, use_wandb=True)
    final_bundle = util.validate_schema6_final_bundle(tmp_path)
    flags = SimpleNamespace(
        ckpdir=str(tmp_path),
        use_wandb=True,
        voc_actor_policy_barrier_runtime=True,
        voc_actor_policy_barrier_timeout_s=120.0,
    )

    def swap_then_ack(_ref, timeout=None):
        request = util.read_actor_policy_logger_finish_request(tmp_path)
        with (tmp_path / "ckp_model.tar").open("ab") as handle:
            handle.write(b"injected-swap")
        util.write_actor_policy_logger_finish_ack(tmp_path, request)
        return True

    monkeypatch.setattr(train_driver.ray, "get", swap_then_ack)
    monkeypatch.setattr(
        train_driver.ray, "kill", lambda actor, no_restart: None
    )
    with pytest.raises(RuntimeError, match="changed"):
        train_driver.finish_schema6_log_worker(
            object(),
            flags,
            final_bundle=final_bundle,
            logger_worker=object(),
            monotonic=lambda: 0.0,
        )
    assert not (tmp_path / "finish").exists()
    assert not (
        tmp_path / util.VOC_ACTOR_POLICY_LOGGER_FINISH_REQUEST_FILE
    ).exists()
    assert not (
        tmp_path / util.VOC_ACTOR_POLICY_LOGGER_FINISH_ACK_FILE
    ).exists()


def test_atomic_json_publish_never_overwrites_file_or_symlink(tmp_path):
    target = tmp_path / "target"
    target.write_text("sentinel", encoding="utf-8")
    with pytest.raises(FileExistsError):
        util._atomic_write_json(target, {"value": 1})
    assert target.read_text(encoding="utf-8") == "sentinel"

    target.unlink()
    destination = tmp_path / "destination"
    destination.write_text("protected", encoding="utf-8")
    target.symlink_to(destination)
    with pytest.raises(FileExistsError):
        util._atomic_write_json(target, {"value": 2})
    assert destination.read_text(encoding="utf-8") == "protected"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda checkpoint: checkpoint.update(real_step=True),
        lambda checkpoint: checkpoint.update(real_step=10.0),
        lambda checkpoint: checkpoint["flags"].update(total_steps=True),
        lambda checkpoint: checkpoint["flags"].update(total_steps=10.0),
        lambda checkpoint: checkpoint.update(real_step=9),
    ],
)
def test_schema6_terminal_progress_requires_strict_integer_equivalence(mutation):
    checkpoint = copy.deepcopy(_terminal_checkpoint())
    mutation(checkpoint)
    with pytest.raises(ValueError, match="progress"):
        util.validate_actor_policy_checkpoint(checkpoint)


def test_schema6_nonterminal_checkpoint_at_or_after_total_is_rejected():
    checkpoint = copy.deepcopy(_terminal_checkpoint())
    checkpoint["voc_actor_policy_terminal"] = False
    checkpoint["voc_actor_policy_terminal_ack_count"] = 0
    checkpoint["voc_actor_policy_bundle"]["terminal"] = False
    checkpoint["voc_actor_policy_publication_history"] = list(
        checkpoint["voc_actor_policy_publication_history"]
    )
    checkpoint["voc_actor_policy_publication_history"][-1]["terminal"] = False
    checkpoint["voc_actor_policy_publication_history_sha256"] = (
        util.actor_policy_publication_history_sha256(
            checkpoint["voc_actor_policy_publication_history"]
        )
    )
    with pytest.raises(ValueError, match="progress"):
        util.validate_actor_policy_checkpoint(checkpoint)


def test_schema6_final_bundle_rejects_stale_model_progress(tmp_path):
    _write_schema6_final_bundle(tmp_path, use_wandb=False)
    model = torch.load(tmp_path / "ckp_model.tar", weights_only=False)
    model["real_step"] = 1
    torch.save(model, tmp_path / "ckp_model.tar")
    with pytest.raises(ValueError, match="stale"):
        util.validate_schema6_final_bundle(tmp_path)


def test_schema6_wandb_off_end_to_end_commits_validated_durable_attestation(
    tmp_path
):
    _write_schema6_final_bundle(tmp_path, use_wandb=False)
    flags = SimpleNamespace(
        ckpdir=str(tmp_path),
        use_wandb=False,
        voc_actor_policy_barrier_runtime=True,
    )
    payload = train_driver.finalize_run(flags)
    completion = payload["voc_actor_policy_logger_completion"]
    assert completion["required"] is False
    assert completion["request_sha256"] is None
    assert completion["ack_verified"] is False
    assert completion["private_markers_cleaned"] is True
    assert (tmp_path / "finish").is_file()


def test_schema6_wandb_on_end_to_end_binds_request_ack_and_public_finish(
    monkeypatch, tmp_path
):
    _write_schema6_final_bundle(tmp_path, use_wandb=True)
    flags = SimpleNamespace(
        ckpdir=str(tmp_path),
        use_wandb=True,
        voc_actor_policy_barrier_runtime=True,
        voc_actor_policy_barrier_timeout_s=120.0,
    )

    def ack_and_return(_ref, timeout=None):
        request = util.read_actor_policy_logger_finish_request(tmp_path)
        util.write_actor_policy_logger_finish_ack(tmp_path, request)
        return True

    monkeypatch.setattr(train_driver.ray, "get", ack_and_return)
    payload = train_driver.finalize_run(
        flags, logger_ref=object(), monotonic=lambda: 0.0
    )
    completion = payload["voc_actor_policy_logger_completion"]
    assert completion["required"] is True
    assert completion["ack_verified"] is True
    assert len(completion["request_sha256"]) == 64
    assert completion["checkpoint_files"] == payload["checkpoint_files"]
    assert not (
        tmp_path / util.VOC_ACTOR_POLICY_LOGGER_FINISH_REQUEST_FILE
    ).exists()
    assert not (
        tmp_path / util.VOC_ACTOR_POLICY_LOGGER_FINISH_ACK_FILE
    ).exists()


def test_public_completion_rejects_forged_actor_logger_binding(tmp_path):
    _write_schema6_final_bundle(tmp_path, use_wandb=False)
    bundle = util.validate_schema6_final_bundle(tmp_path)
    actor = bundle["actor_policy"]
    forged = {
        "schema_version": 1,
        "required": False,
        "use_wandb": False,
        "request_sha256": None,
        "ack_verified": False,
        "private_markers_cleaned": True,
        "policy_version": actor["voc_actor_policy_version"],
        "state_sha256": "f" * 64,
        "publication_history_sha256": actor[
            "voc_actor_policy_publication_history_sha256"
        ],
        "checkpoint_files": bundle["completion_evidence"][
            "checkpoint_files"
        ],
    }
    with pytest.raises(ValueError, match="validated actor"):
        util.write_run_completion(
            tmp_path,
            expected_evidence=bundle["completion_evidence"],
            actor_policy_logger_completion=forged,
            validated_actor_policy=actor,
        )
    assert not (tmp_path / "finish").exists()


def test_public_completion_removes_its_exact_inode_on_post_publish_failure(
    monkeypatch, tmp_path
):
    evidence = {
        "checkpoint_files": _completion_checkpoint_files(),
        "implementation_sources": {},
        "loaded_extensions": {},
    }
    calls = 0

    def collect(_path):
        nonlocal calls
        calls += 1
        if calls == 1:
            return evidence
        raise RuntimeError("injected post-publish failure")

    monkeypatch.setattr(util, "collect_run_completion_evidence", collect)
    with pytest.raises(RuntimeError, match="post-publish"):
        util.write_run_completion(tmp_path)
    assert not (tmp_path / "finish").exists()


def test_exact_inode_cleanup_never_unlinks_a_raced_replacement(tmp_path):
    target = tmp_path / "finish"
    target.write_text("old", encoding="utf-8")
    old_link = tmp_path / "old-link"
    old_link.hardlink_to(target)
    old_stat = target.stat()
    target.unlink()
    target.write_text("replacement", encoding="utf-8")
    assert util._unlink_exact_published_path(
        target, (old_stat.st_dev, old_stat.st_ino)
    ) is False
    assert target.read_text(encoding="utf-8") == "replacement"
    old_link.unlink()


def test_private_marker_cleanup_unlinks_before_fsync_failure(
    monkeypatch, tmp_path
):
    evidence = util.validate_actor_policy_checkpoint(_terminal_checkpoint())
    util.write_actor_policy_logger_finish_request(
        tmp_path,
        evidence,
        {"checkpoint_files": _completion_checkpoint_files()},
    )
    monkeypatch.setattr(
        util.os,
        "fsync",
        lambda _fd: (_ for _ in ()).throw(RuntimeError("injected fsync")),
    )
    with pytest.raises(RuntimeError, match="injected fsync"):
        util.clear_actor_policy_logger_completion(tmp_path)
    assert not (
        tmp_path / util.VOC_ACTOR_POLICY_LOGGER_FINISH_REQUEST_FILE
    ).exists()


def test_schema6_fresh_directory_rejects_any_prior_run_artifact(tmp_path):
    new_xpid = tmp_path / "new-xpid"
    assert util.create_schema6_fresh_run_directory(new_xpid) == str(new_xpid)
    with pytest.raises(FileExistsError, match="already exists"):
        util.create_schema6_fresh_run_directory(new_xpid)
    (new_xpid / "config_c.yaml").write_text("x: 1\n", encoding="utf-8")
    assert util.validate_schema6_fresh_run_directory(new_xpid) is True
    (new_xpid / "logs.csv").write_text("prior", encoding="utf-8")
    with pytest.raises(FileExistsError, match="pre-existing"):
        util.validate_schema6_fresh_run_directory(new_xpid)


def test_private_request_hash_excludes_trailing_lf_exactly(tmp_path):
    request = _write_private_logger_request(tmp_path)
    path = tmp_path / util.VOC_ACTOR_POLICY_LOGGER_FINISH_REQUEST_FILE
    raw = path.read_bytes()
    assert raw.endswith(b"\n")
    import hashlib

    assert hashlib.sha256(raw[:-1]).hexdigest() == (
        util.actor_policy_logger_finish_request_sha256(request)
    )
    assert hashlib.sha256(raw).hexdigest() != (
        util.actor_policy_logger_finish_request_sha256(request)
    )


def test_private_marker_read_rejects_hardlink_and_duplicate_json_key(tmp_path):
    _write_private_logger_request(tmp_path)
    path = tmp_path / util.VOC_ACTOR_POLICY_LOGGER_FINISH_REQUEST_FILE
    alias = tmp_path / "request-hardlink"
    alias.hardlink_to(path)
    with pytest.raises(ValueError, match="single-link"):
        util.read_actor_policy_logger_finish_request(tmp_path)
    alias.unlink()

    raw = path.read_text(encoding="utf-8")
    raw = raw.replace(
        '"schema_version":1',
        '"schema_version":1,"schema_version":1',
        1,
    )
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate key"):
        util.read_actor_policy_logger_finish_request(tmp_path)


@pytest.mark.parametrize(
    "field",
    sorted(util.VOC_GATE_POLICY_SCHEMA6_ATOMIC_REQUIREMENTS),
)
def test_schema6_atomic_protocol_requires_every_explicit_field(field):
    checkpoint = copy.deepcopy(_terminal_checkpoint())
    checkpoint["flags"].pop(field)
    with pytest.raises(ValueError, match=field):
        util.validate_actor_policy_checkpoint(checkpoint)


@pytest.mark.parametrize(
    "field", sorted(util.VOC_GATE_POLICY_SCHEMA6_ENDURO_REQUIREMENTS)
)
def test_schema6_enduro_stage_requires_every_explicit_field(field):
    checkpoint = copy.deepcopy(_terminal_checkpoint())
    checkpoint["flags"].pop(field)
    with pytest.raises(ValueError, match=field):
        util.validate_actor_policy_checkpoint(checkpoint)


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("voc_loss_cost", 0.5),
        ("voc_loss_cost", True),
        ("voc_loss_cost", np.nextafter(1.0, 2.0)),
        ("voc_gate_grad_norm_clipping", 0.5),
        ("voc_gate_grad_norm_clipping", True),
        ("voc_gate_grad_norm_clipping", np.nextafter(1.0, 2.0)),
        ("voc_gate_confidence_weighted", True),
        ("voc_gate_adam_beta1", 0.9),
        ("voc_gate_learning_rate", 0.0003),
        ("voc_train_epsilon", 0.3),
        ("model_state_range_loss_cost", 0.0),
        (
            "model_state_range_loss_cost",
            np.nextafter(1.0, 2.0),
        ),
    ],
)
def test_schema6_atomic_protocol_rejects_numeric_and_boolean_drift(field, bad):
    checkpoint = copy.deepcopy(_terminal_checkpoint())
    checkpoint["flags"][field] = bad
    with pytest.raises(ValueError, match=field):
        util.validate_actor_policy_checkpoint(checkpoint)


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("float16", False),
        ("float16", np.bool_(True)),
        ("model_float16", True),
        ("model_float16", np.bool_(False)),
        ("dual_net", False),
        ("train_model", False),
        ("model_optimizer", "sgd"),
        ("schedule_total_steps", 99_999_999),
        ("schedule_total_steps", 100_000_000.0),
    ],
)
def test_schema6_atomic_runtime_identity_is_absolute(field, bad):
    checkpoint = copy.deepcopy(_terminal_checkpoint())
    checkpoint["flags"][field] = bad
    with pytest.raises(ValueError, match=field):
        util.validate_actor_policy_checkpoint(checkpoint)


def test_schema6_final_rejects_same_wrong_soft_protocol_on_all_surfaces(tmp_path):
    _write_schema6_final_bundle(tmp_path, use_wandb=False)
    with (tmp_path / "config_c.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    actor = torch.load(tmp_path / "ckp_actor.tar", weights_only=False)
    model = torch.load(tmp_path / "ckp_model.tar", weights_only=False)
    for surface in (config, actor["flags"], model["flags"]):
        surface["voc_train_epsilon"] = 0.3
    with (tmp_path / "config_c.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)
    torch.save(actor, tmp_path / "ckp_actor.tar")
    torch.save(model, tmp_path / "ckp_model.tar")
    with pytest.raises(ValueError, match="voc_train_epsilon"):
        util.validate_schema6_final_bundle(tmp_path)


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("model_state_projection", "none"),
        ("model_state_range_loss_cost", 0.0),
    ],
)
def test_schema6_final_rejects_same_wrong_model_range_protocol_on_all_surfaces(
    tmp_path, field, bad
):
    _write_schema6_final_bundle(tmp_path, use_wandb=False)
    with (tmp_path / "config_c.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    actor = torch.load(tmp_path / "ckp_actor.tar", weights_only=False)
    model = torch.load(tmp_path / "ckp_model.tar", weights_only=False)
    for surface in (config, actor["flags"], model["flags"]):
        surface[field] = bad
    with (tmp_path / "config_c.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)
    torch.save(actor, tmp_path / "ckp_actor.tar")
    torch.save(model, tmp_path / "ckp_model.tar")
    with pytest.raises(ValueError, match=field):
        util.validate_schema6_final_bundle(tmp_path)


@pytest.mark.parametrize(
    ("file_name", "key"),
    [
        ("ckp_actor.tar", "voc_ema_gate_head_state_dict"),
        ("ckp_actor.tar", "voc_optimizer_state_dict"),
        ("ckp_actor.tar", "voc_scheduler_state_dict"),
        ("ckp_actor.tar", "voc_grad_scaler_state_dict"),
        ("ckp_actor.tar", "voc_gate_optimizer_state_dict"),
        ("ckp_actor.tar", "voc_gate_scheduler_state_dict"),
        ("ckp_actor.tar", "voc_gate_grad_scaler_state_dict"),
        ("ckp_actor.tar", "actor_net_optimizer_state_dict"),
        ("ckp_actor.tar", "actor_net_scheduler_state_dict"),
        ("ckp_model.tar", "model_net_optimizer_p_state_dict"),
        ("ckp_model.tar", "model_net_scheduler_p_state_dict"),
        ("ckp_model.tar", "model_net_optimizer_m_state_dict"),
        ("ckp_model.tar", "model_net_scheduler_m_state_dict"),
    ],
)
def test_schema6_final_requires_complete_training_state(tmp_path, file_name, key):
    _write_schema6_final_bundle(tmp_path, use_wandb=False)
    path = tmp_path / file_name
    checkpoint = torch.load(path, weights_only=False)
    checkpoint.pop(key)
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="EMA|optimizer|scheduler|GradScaler"):
        util.validate_schema6_final_bundle(tmp_path)


@pytest.mark.parametrize("surface", ["actor", "model"])
def test_schema6_final_rejects_nonfinite_optimizer_state(tmp_path, surface):
    _write_schema6_final_bundle(tmp_path, use_wandb=False)
    if surface == "actor":
        path = tmp_path / "ckp_actor.tar"
        checkpoint = torch.load(path, weights_only=False)
        checkpoint["actor_net_optimizer_state_dict"]["state"][0][
            "exp_avg"
        ].fill_(float("nan"))
    else:
        path = tmp_path / "ckp_model.tar"
        checkpoint = torch.load(path, weights_only=False)
        checkpoint["model_net_optimizer_p_state_dict"]["state"][0][
            "exp_avg"
        ].fill_(float("nan"))
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="non-finite"):
        util.validate_schema6_final_bundle(tmp_path)


def test_schema6_cross_surface_integer_identity_never_float_coerces(tmp_path):
    _write_schema6_final_bundle(tmp_path, use_wandb=False)
    actor = torch.load(tmp_path / "ckp_actor.tar", weights_only=False)
    actor["flags"]["base_seed"] = float(2**53)
    torch.save(actor, tmp_path / "ckp_actor.tar")
    with pytest.raises(ValueError, match="base_seed"):
        util.validate_schema6_final_bundle(tmp_path)


def test_schema6_logger_timeout_kills_actor_before_cleaning_late_ack(
    monkeypatch, tmp_path
):
    evidence = util.validate_actor_policy_checkpoint(_terminal_checkpoint())
    final_bundle = {
        "actor_policy": evidence,
        "completion_evidence": {
            "checkpoint_files": _completion_checkpoint_files(),
            "implementation_sources": {},
            "loaded_extensions": {},
        },
    }
    flags = SimpleNamespace(
        ckpdir=str(tmp_path),
        use_wandb=True,
        voc_actor_policy_barrier_runtime=True,
        voc_actor_policy_barrier_timeout_s=120.0,
    )
    logger_ref = object()
    logger_worker = object()
    order = []
    get_calls = 0

    def timeout_get(_ref, timeout=None):
        nonlocal get_calls
        get_calls += 1
        if get_calls == 1:
            raise train_driver.ray.exceptions.GetTimeoutError("injected hang")
        raise RuntimeError("confirmed dead logger task")

    def late_ack_on_kill(actor, *, no_restart):
        order.append(("kill", actor, no_restart))
        request = util.read_actor_policy_logger_finish_request(tmp_path)
        util.write_actor_policy_logger_finish_ack(tmp_path, request)

    original_clear = util.clear_actor_policy_logger_completion

    def record_clear(path, **kwargs):
        order.append(("clear",))
        return original_clear(path, **kwargs)

    monkeypatch.setattr(train_driver.ray, "get", timeout_get)
    monkeypatch.setattr(
        train_driver.ray,
        "cancel",
        lambda ref, force: order.append(("cancel", ref, force)),
    )
    monkeypatch.setattr(train_driver.ray, "kill", late_ack_on_kill)
    monkeypatch.setattr(util, "clear_actor_policy_logger_completion", record_clear)
    with pytest.raises(TimeoutError, match="did not acknowledge"):
        train_driver.finish_schema6_log_worker(
            logger_ref,
            flags,
            final_bundle=final_bundle,
            logger_worker=logger_worker,
            monotonic=lambda: 0.0,
        )
    assert order[:3] == [
        ("kill", logger_worker, True),
        ("cancel", logger_ref, False),
        ("clear",),
    ]
    assert not (
        tmp_path / util.VOC_ACTOR_POLICY_LOGGER_FINISH_REQUEST_FILE
    ).exists()
    assert not (
        tmp_path / util.VOC_ACTOR_POLICY_LOGGER_FINISH_ACK_FILE
    ).exists()
    assert not (tmp_path / "finish").exists()


def test_schema6_unconfirmed_logger_death_retains_private_request(
    monkeypatch, tmp_path
):
    evidence = util.validate_actor_policy_checkpoint(_terminal_checkpoint())
    final_bundle = {
        "actor_policy": evidence,
        "completion_evidence": {
            "checkpoint_files": _completion_checkpoint_files(),
            "implementation_sources": {},
            "loaded_extensions": {},
        },
    }
    flags = SimpleNamespace(
        ckpdir=str(tmp_path),
        use_wandb=True,
        voc_actor_policy_barrier_runtime=True,
        voc_actor_policy_barrier_timeout_s=120.0,
    )
    monkeypatch.setattr(
        train_driver.ray,
        "get",
        lambda _ref, timeout=None: (_ for _ in ()).throw(
            train_driver.ray.exceptions.GetTimeoutError("still alive")
        ),
    )
    monkeypatch.setattr(
        train_driver.ray, "kill", lambda actor, no_restart: None
    )
    with pytest.raises(TimeoutError, match="did not acknowledge") as error:
        train_driver.finish_schema6_log_worker(
            object(),
            flags,
            final_bundle=final_bundle,
            logger_worker=object(),
            monotonic=lambda: 0.0,
        )
    assert isinstance(error.value.__cause__, TimeoutError)
    assert "death could not be confirmed" in str(error.value.__cause__)
    assert (
        tmp_path / util.VOC_ACTOR_POLICY_LOGGER_FINISH_REQUEST_FILE
    ).exists()
    assert not (tmp_path / "finish").exists()


@pytest.mark.parametrize("corruption", ["missing", "wrong_shape", "wrong_dtype"])
def test_schema6_final_rejects_truncated_or_wrong_model_architecture(
    tmp_path, corruption
):
    _write_schema6_final_bundle(tmp_path, use_wandb=False)
    path = tmp_path / "ckp_model.tar"
    checkpoint = torch.load(path, weights_only=False)
    state = checkpoint["model_net_state_dict"]
    if corruption == "missing":
        state.pop("vp_net.bias")
    elif corruption == "wrong_shape":
        state["vp_net.weight"] = torch.zeros(3, 2)
    else:
        state["vp_net.weight"] = state["vp_net.weight"].double()
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="architecture"):
        util.validate_schema6_final_bundle(tmp_path)


@pytest.mark.parametrize(
    ("surface", "corruption"),
    [
        ("actor", "missing_state"),
        ("actor", "duplicate_param"),
        ("actor", "wrong_shape"),
        ("actor", "missing_moment"),
        ("actor", "noncanonical_ids"),
        ("actor", "python_step"),
        ("actor", "wrong_step_dtype"),
        ("actor", "aliased_moments"),
        ("actor", "noncontiguous_moment"),
        ("model", "missing_state"),
        ("model", "duplicate_param"),
        ("model", "wrong_shape"),
        ("model", "missing_moment"),
    ],
)
def test_schema6_final_optimizer_exactly_covers_reconstructed_parameters(
    tmp_path, surface, corruption
):
    _write_schema6_final_bundle(tmp_path, use_wandb=False)
    if surface == "actor":
        path = tmp_path / "ckp_actor.tar"
        optimizer_key = "actor_net_optimizer_state_dict"
    else:
        path = tmp_path / "ckp_model.tar"
        optimizer_key = "model_net_optimizer_p_state_dict"
    checkpoint = torch.load(path, weights_only=False)
    optimizer = checkpoint[optimizer_key]
    if corruption == "missing_state":
        optimizer["state"].pop(1)
    elif corruption == "duplicate_param":
        optimizer["param_groups"][0]["params"] = [0, 0]
    elif corruption == "wrong_shape":
        optimizer["state"][0]["exp_avg"] = torch.zeros(1)
    elif corruption == "missing_moment":
        optimizer["state"][0].pop("exp_avg_sq")
    elif corruption == "noncanonical_ids":
        optimizer["state"] = {
            key + 2: value for key, value in optimizer["state"].items()
        }
        optimizer["param_groups"][0]["params"] = [2, 3]
    elif corruption == "python_step":
        optimizer["state"][0]["step"] = 1
    elif corruption == "wrong_step_dtype":
        optimizer["state"][0]["step"] = torch.tensor(1.0, dtype=torch.float64)
    elif corruption == "aliased_moments":
        optimizer["state"][0]["exp_avg_sq"] = optimizer["state"][0][
            "exp_avg"
        ]
    else:
        source = torch.zeros(4, 3)
        optimizer["state"][0]["exp_avg"] = source[::2]
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="optimizer|parameter|Adam"):
        util.validate_schema6_final_bundle(tmp_path)


@pytest.mark.parametrize(
    ("surface", "field", "bad"),
    [
        ("actor", "_step_count", 99),
        ("actor", "last_epoch", 9),
        ("actor", "_last_lr", [0.0003]),
        ("model", "_step_count", 99),
        ("model", "last_epoch", 9),
        ("model", "_last_lr", [0.00005]),
    ],
)
def test_schema6_final_rejects_stale_scheduler_state(
    tmp_path, surface, field, bad
):
    _write_schema6_final_bundle(tmp_path, use_wandb=False)
    if surface == "actor":
        path = tmp_path / "ckp_actor.tar"
        scheduler_key = "actor_net_scheduler_state_dict"
    else:
        path = tmp_path / "ckp_model.tar"
        scheduler_key = "model_net_scheduler_p_state_dict"
    checkpoint = torch.load(path, weights_only=False)
    checkpoint[scheduler_key][field] = bad
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="scheduler|stale|schedule"):
        util.validate_schema6_final_bundle(tmp_path)


@pytest.mark.parametrize("surface", ["actor", "model"])
def test_schema6_final_binds_adam_step_to_publication_or_model_counter(
    tmp_path, surface
):
    _write_schema6_final_bundle(tmp_path, use_wandb=False)
    if surface == "actor":
        path = tmp_path / "ckp_actor.tar"
        key = "actor_net_optimizer_state_dict"
    else:
        path = tmp_path / "ckp_model.tar"
        key = "model_net_optimizer_p_state_dict"
    checkpoint = torch.load(path, weights_only=False)
    checkpoint[key]["state"][0]["step"] = torch.tensor(2.0)
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="Adam step"):
        util.validate_schema6_final_bundle(tmp_path)


@pytest.mark.parametrize(
    "counter",
    [
        "voc_update_count",
        "voc_ema_gate_update_count",
        "voc_gate_update_count",
        "imitation_update_count",
        "imitation_schedule_step",
    ],
)
def test_schema6_final_actor_transaction_counters_equal_policy_version(
    tmp_path, counter
):
    _write_schema6_final_bundle(tmp_path, use_wandb=False)
    path = tmp_path / "ckp_actor.tar"
    checkpoint = torch.load(path, weights_only=False)
    checkpoint[counter] = 0
    torch.save(checkpoint, path)
    with pytest.raises(ValueError):
        util.validate_schema6_final_bundle(tmp_path)


@pytest.mark.parametrize("kind", ["zero", "mismatch"])
def test_schema6_final_model_update_counters_are_positive_and_lockstep(
    tmp_path, kind
):
    _write_schema6_final_bundle(tmp_path, use_wandb=False)
    path = tmp_path / "ckp_model.tar"
    checkpoint = torch.load(path, weights_only=False)
    component = "p" if kind == "zero" else "m"
    step = 0 if kind == "zero" else 2
    checkpoint[f"model_grad_step_count_{component}"] = step
    optimizer = checkpoint[f"model_net_optimizer_{component}_state_dict"]
    for state in optimizer["state"].values():
        state["step"] = torch.tensor(float(step), dtype=torch.float32)
    checkpoint[f"model_net_scheduler_{component}_state_dict"][
        "_step_count"
    ] = step + 1
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="positive.*lockstep"):
        util.validate_schema6_final_bundle(tmp_path)
