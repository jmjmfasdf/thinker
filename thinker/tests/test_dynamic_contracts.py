from argparse import Namespace
import copy
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

from thinker import util


_SCHEMA6_FAILED_WIRE_CLI = """
--name Enduro-v5
--xpid enduro-voc-v13-versioned-eps25-sentinel-wire1200
--savedir /tmp/di-voc-v13-versioned-eps25-final-WjoXru/runs
--ckp False
--preload ''
--preload_actor ''
--voc_parent_checkpoint ''
--total_steps 1200
--schedule_total_steps 100000000
--model_warm_up_n 512
--actor_unroll_len 41
--dynamic_search True
--dynamic_factorized_control True
--dynamic_voc_mode control
--voc_loss_cost 1.0
--voc_gate_temperature 1.0
--voc_train_epsilon 0.02
--voc_eval_stochastic True
--voc_dueling_q True
--voc_expected_gate_loss True
--voc_ema_gate_target True
--voc_gate_target_tau 0.1
--voc_dedicated_gate True
--voc_soft_q_bce_gate True
--voc_gate_q_temperature 0.05
--voc_gate_confidence_weighted False
--voc_gate_adam_beta1 0.0
--voc_gate_param_align False
--voc_gate_param_align_coef 1.0
--voc_gate_exact_projection True
--voc_gate_epsilon_greedy_execution True
--voc_gate_execution_epsilon 0.25
--voc_actor_policy_version_barrier True
--voc_actor_policy_bundle_schema_version 1
--voc_actor_policy_barrier_timeout_s 120
--voc_actor_policy_ray_max_restarts 0
--voc_actor_policy_ray_max_task_retries 0
--actor_amp_init_scale 32
--voc_gate_learning_rate 0.001
--voc_gate_grad_norm_clipping 1.0
--entropy_r_cost 0
--wrapper_type 0
--rec_t 20
--max_search_steps 20
--max_depth 20
--model_unroll_len 20
--think_cost 0.0005
--think_cost_anneal False
--tree_carry True
--train_model True
--float16 True
--actor_amp_max_consecutive_skips 8
--model_float16 False
--model_learning_rate 0.00005
--model_grad_norm_clipping 10000
--model_disable_bn False
--model_state_projection clamp
--model_state_range_loss_cost 1.0
--model_batch_size 32
--actor_batch_size 16
--env_n 16
--self_play_n 1
--parallel_actor True
--ppo_k 1
--icopro_data_path /tmp/di-voc-v13-versioned-eps25-final-WjoXru/data/behavioral_data_block
--icopro_subjects 1
--icopro_game_id 0
--icopro_train_sessions 1,2,3
--icopro_holdout_sessions 4
--icopro_batch_size 16
--batch_length 4
--icopro_margin 1.0
--icopro_margin_coef 1.0
--icopro_action_diff_coef 1.0
--icopro_pvp_coef 0.0
--icopro_coef 1.0
--icopro_supervised_freq 1
--action_prior_weight 1.0
--action_prior_ema 0.05
--icopro_device cuda
--reward_clip 1
--model_size_nn 2
--discounting 0.99
--envpool True
--grayscale False
--frame_stack_n 4
--auto_res False
--ray_cpu 16
--ray_gpu 2
--gpu_learn 1
--gpu_learn_actor 0.5
--gpu_self_play 0.5
--use_wandb False
--base_seed 1
--actor_use_rms False
"""


def _schema6_failed_wire_cli_args():
    return shlex.split(_SCHEMA6_FAILED_WIRE_CLI)


def _schema7_wire_cli_args():
    args = _schema6_failed_wire_cli_args()
    replacements = {
        "xpid": util.VOC_GATE_POLICY_SCHEMA7_STAGE_PROFILES[0][0],
        "savedir": "/tmp/di-voc-v14-sealed-eps25-test/runs",
        "icopro_data_path": (
            "/tmp/di-voc-v14-sealed-eps25-test/data/"
            "behavioral_data_block"
        ),
    }
    for name, value in replacements.items():
        args = _replace_cli_value(args, name, value)
    args.extend(["--voc_model_input_seal_schema_version", "1"])
    return args


def _schema8_wire_cli_args():
    args = _schema7_wire_cli_args()
    replacements = {
        "xpid": util.VOC_GATE_POLICY_SCHEMA8_STAGE_PROFILES[0][0],
        "savedir": "/tmp/di-voc-v15-halfsq-eps25-test/runs",
        "icopro_data_path": (
            "/tmp/di-voc-v15-halfsq-eps25-test/data/"
            "behavioral_data_block"
        ),
    }
    for name, value in replacements.items():
        args = _replace_cli_value(args, name, value)
    return args


def _schema9_wire_cli_args():
    args = _schema8_wire_cli_args()
    replacements = {
        "xpid": util.VOC_GATE_POLICY_SCHEMA9_STAGE_PROFILES[0][0],
        "savedir": "/tmp/di-voc-v16-commonmode-eps25-test/runs",
        "icopro_data_path": (
            "/tmp/di-voc-v16-commonmode-eps25-test/data/"
            "behavioral_data_block"
        ),
    }
    for name, value in replacements.items():
        args = _replace_cli_value(args, name, value)
    return args


def _schema10_wire_cli_args():
    args = _schema9_wire_cli_args()
    replacements = {
        "xpid": util.VOC_GATE_POLICY_SCHEMA10_STAGE_PROFILES[0][0],
        "savedir": "/tmp/di-voc-v17-huber-common-eps25-test/runs",
        "icopro_data_path": (
            "/tmp/di-voc-v17-huber-common-eps25-test/data/"
            "behavioral_data_block"
        ),
    }
    for name, value in replacements.items():
        args = _replace_cli_value(args, name, value)
    return args


def _schema11_wire_cli_args():
    args = _schema10_wire_cli_args()
    replacements = {
        "xpid": util.VOC_GATE_POLICY_SCHEMA11_STAGE_PROFILES[0][0],
        "savedir": "/tmp/di-voc-v18-orthocd-adam-eps25-test/runs",
        "icopro_data_path": (
            "/tmp/di-voc-v18-orthocd-adam-eps25-test/data/"
            "behavioral_data_block"
        ),
    }
    for name, value in replacements.items():
        args = _replace_cli_value(args, name, value)
    return args


def _schema12_wire_cli_args():
    args = _schema11_wire_cli_args()
    replacements = {
        "xpid": util.VOC_GATE_POLICY_SCHEMA12_STAGE_PROFILES[0][0],
        "savedir": "/tmp/di-voc-v19-tau1-orthocd-adam-eps25-test/runs",
        "icopro_data_path": (
            "/tmp/di-voc-v19-tau1-orthocd-adam-eps25-test/data/"
            "behavioral_data_block"
        ),
        "voc_gate_target_tau": 1.0,
    }
    for name, value in replacements.items():
        args = _replace_cli_value(args, name, value)
    return args


def _schema13_wire_cli_args():
    args = _schema12_wire_cli_args()
    replacements = {
        "xpid": util.VOC_GATE_POLICY_SCHEMA13_STAGE_PROFILES[0][0],
        "savedir": "/tmp/di-voc-v20-telemetry-tau1-orthocd-adam-eps25-test/runs",
        "icopro_data_path": (
            "/tmp/di-voc-v20-telemetry-tau1-orthocd-adam-eps25-test/data/"
            "behavioral_data_block"
        ),
    }
    for name, value in replacements.items():
        args = _replace_cli_value(args, name, value)
    return args


def _replace_cli_value(args, name, value):
    args = list(args)
    index = args.index(f"--{name}")
    args[index + 1] = str(value)
    return args


def _flags(**overrides):
    values = dict(
        wrapper_type=0,
        dynamic_search=True,
        max_search_steps=-1,
        reset_mode=0,
        rec_t=40,
        has_action_seq=True,
        im_cost=1.0,
        cur_cost=0.0,
        think_cost=0.002,
        think_cost_anneal=False,
        dynamic_search_hidden_dim=100,
        dual_net=True,
        cur_enable=False,
        model_rs_loss_cost=1.0,
        model_img_loss_cost=0.0,
        model_done_loss_cost=1.0,
    )
    values.update(overrides)
    return Namespace(**values)


def _schema6_flags(**overrides):
    values = dict(util.VOC_GATE_POLICY_SCHEMA6_V12_BASELINE)
    xpid, seed, total, warmup, unroll, use_wandb = (
        util.VOC_GATE_POLICY_SCHEMA6_STAGE_PROFILES[0]
    )
    savedir = "/tmp/di-voc-v13-versioned-eps25-test/runs"
    values.update({
        "xpid": xpid,
        "base_seed": seed,
        "total_steps": total,
        "model_warm_up_n": warmup,
        "actor_unroll_len": unroll,
        "use_wandb": use_wandb,
        "savedir": savedir,
        "ckpdir": f"{savedir}/{xpid}",
        "cmd": "train.py --schema6-test",
        "icopro_data_path": (
            "/tmp/di-voc-v13-versioned-eps25-test/data/"
            "behavioral_data_block"
        ),
        "voc_gate_policy_schema_version": 6,
        "voc_gate_execution_epsilon": 0.25,
        "voc_actor_policy_version_barrier": True,
        "voc_actor_policy_bundle_schema_version": 1,
        "voc_actor_policy_barrier_timeout_s": 120.0,
        "voc_actor_policy_ray_max_restarts": 0,
        "voc_actor_policy_ray_max_task_retries": 0,
        "actor_amp_init_scale": 32.0,
        "voc_actor_policy_barrier_runtime": True,
    })
    values.update(overrides)
    return Namespace(**values)


def _schema7_flags(**overrides):
    values = vars(_schema6_flags()).copy()
    xpid, seed, total, warmup, unroll, use_wandb = (
        util.VOC_GATE_POLICY_SCHEMA7_STAGE_PROFILES[0]
    )
    savedir = "/tmp/di-voc-v14-sealed-eps25-test/runs"
    values.update({
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
        "voc_gate_policy_schema_version": 7,
        "voc_model_input_seal_schema_version": 1,
    })
    values.update(overrides)
    return Namespace(**values)


def _schema8_flags(**overrides):
    values = vars(_schema7_flags()).copy()
    xpid, seed, total, warmup, unroll, use_wandb = (
        util.VOC_GATE_POLICY_SCHEMA8_STAGE_PROFILES[0]
    )
    savedir = "/tmp/di-voc-v15-halfsq-eps25-test/runs"
    values.update({
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
        "voc_gate_policy_schema_version": 8,
        "voc_model_input_seal_schema_version": 1,
    })
    values.update(overrides)
    return Namespace(**values)


def _schema9_flags(**overrides):
    values = vars(_schema8_flags()).copy()
    xpid, seed, total, warmup, unroll, use_wandb = (
        util.VOC_GATE_POLICY_SCHEMA9_STAGE_PROFILES[0]
    )
    savedir = "/tmp/di-voc-v16-commonmode-eps25-test/runs"
    values.update({
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
        "voc_gate_policy_schema_version": 9,
        "voc_model_input_seal_schema_version": 1,
    })
    values.update(overrides)
    return Namespace(**values)


def _schema10_flags(**overrides):
    values = vars(_schema9_flags()).copy()
    xpid, seed, total, warmup, unroll, use_wandb = (
        util.VOC_GATE_POLICY_SCHEMA10_STAGE_PROFILES[0]
    )
    savedir = "/tmp/di-voc-v17-huber-common-eps25-test/runs"
    values.update({
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
        "voc_gate_policy_schema_version": 10,
        "voc_model_input_seal_schema_version": 1,
    })
    values.update(overrides)
    return Namespace(**values)


def _schema11_flags(**overrides):
    values = vars(_schema10_flags()).copy()
    xpid, seed, total, warmup, unroll, use_wandb = (
        util.VOC_GATE_POLICY_SCHEMA11_STAGE_PROFILES[0]
    )
    savedir = "/tmp/di-voc-v18-orthocd-adam-eps25-test/runs"
    values.update({
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
        "voc_gate_policy_schema_version": 11,
        "voc_model_input_seal_schema_version": 1,
    })
    values.update(overrides)
    return Namespace(**values)


def _schema12_flags(**overrides):
    values = vars(_schema11_flags()).copy()
    xpid, seed, total, warmup, unroll, use_wandb = (
        util.VOC_GATE_POLICY_SCHEMA12_STAGE_PROFILES[0]
    )
    savedir = "/tmp/di-voc-v19-tau1-orthocd-adam-eps25-test/runs"
    values.update({
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
        "voc_gate_policy_schema_version": 12,
        "voc_model_input_seal_schema_version": 1,
        "voc_gate_target_tau": 1.0,
    })
    values.update(overrides)
    return Namespace(**values)


def _schema13_flags(**overrides):
    values = vars(_schema12_flags()).copy()
    xpid, seed, total, warmup, unroll, use_wandb = (
        util.VOC_GATE_POLICY_SCHEMA13_STAGE_PROFILES[0]
    )
    savedir = "/tmp/di-voc-v20-telemetry-tau1-orthocd-adam-eps25-test/runs"
    values.update({
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
        "voc_gate_policy_schema_version": 13,
        "voc_model_input_seal_schema_version": 1,
        "voc_gate_target_tau": 1.0,
    })
    values.update(overrides)
    return Namespace(**values)


def test_schema13_real_package_import_has_no_forward_default_dependency():
    package_root = Path(util.__file__).resolve().parents[1]
    code = "\n".join((
        "import sys",
        f"sys.path.insert(0, {str(package_root)!r})",
        "from thinker import util, voc_telemetry",
        "assert util.VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION == 13",
        "assert voc_telemetry.VOC_TELEMETRY_SCHEMA_VERSION == 1",
    ))
    environment = os.environ.copy()
    environment.update({
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
    })
    result = subprocess.run(
        [sys.executable, "-B", "-I", "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 0, result.stderr


def test_schema12_fresh_process_does_not_import_schema13_telemetry_module():
    package_root = Path(util.__file__).resolve().parents[1]
    code = "\n".join((
        "import sys",
        f"sys.path.insert(0, {str(package_root)!r})",
        "from argparse import Namespace",
        "from thinker import util",
        "assert 'thinker.voc_telemetry' not in sys.modules",
        "flags = util.process_flags(Namespace(**{",
        "    **util.VOC_GATE_POLICY_SCHEMA12_V12_PROJECTION,",
        "    **dict(zip((",
        "        'xpid', 'base_seed', 'total_steps', 'model_warm_up_n',",
        "        'actor_unroll_len', 'use_wandb'),",
        "        util.VOC_GATE_POLICY_SCHEMA12_STAGE_PROFILES[0])),",
        "    'savedir': '/tmp/di-voc-v19-tau1-orthocd-adam-eps25-test/runs',",
        "    'ckpdir': '/tmp/di-voc-v19-tau1-orthocd-adam-eps25-test/runs/'",
        "        + util.VOC_GATE_POLICY_SCHEMA12_STAGE_PROFILES[0][0],",
        "    'cmd': 'train.py --schema12-fresh-process-test',",
        "    'icopro_data_path': '/tmp/di-voc-v19-tau1-orthocd-adam-eps25-test/'",
        "        'data/behavioral_data_block',",
        "    'voc_gate_policy_schema_version': 12,",
        "    'voc_gate_execution_epsilon': 0.25,",
        "    'voc_model_input_seal_schema_version': 1,",
        "    'voc_gate_target_tau': 1.0,",
        "    'voc_actor_policy_version_barrier': True,",
        "    'voc_actor_policy_bundle_schema_version': 1,",
        "    'voc_actor_policy_barrier_timeout_s': 120.0,",
        "    'voc_actor_policy_ray_max_restarts': 0,",
        "    'voc_actor_policy_ray_max_task_retries': 0,",
        "    'actor_amp_init_scale': 32.0,",
        "    'voc_actor_policy_barrier_runtime': True,",
        "}))",
        "assert flags.voc_gate_policy_schema_version == 12",
        "assert 'thinker.voc_telemetry' not in sys.modules",
    ))
    environment = os.environ.copy()
    environment.update({
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
    })
    result = subprocess.run(
        [sys.executable, "-B", "-I", "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 0, result.stderr


def test_dynamic_public_enum_values_are_stable():
    assert (util.PROCEED, util.RESET, util.STOP) == (0, 1, 2)
    assert (
        util.SEARCH_PHASE,
        util.NEED_REAL_ACTION_PHASE,
        util.WAIT_PHASE,
    ) == (0, 1, 2)


def test_factorized_control_legacy_default_is_disabled():
    flags = _flags()
    util.process_flags(flags)
    assert flags.dynamic_factorized_control is False


def test_factorized_control_opt_in_requires_dynamic_search():
    with pytest.raises(ValueError, match="requires dynamic_search=true"):
        util.process_flags(_flags(
            dynamic_search=False, dynamic_factorized_control=True
        ))


def test_voc_legacy_defaults_are_disabled_and_normalized():
    flags = _flags()

    util.process_flags(flags)

    assert util.get_voc_protocol(flags) == util.VOC_PROTOCOL_DEFAULTS


def test_schema6_process_flags_accepts_only_the_atomic_v13_mechanism():
    flags = util.process_flags(_schema6_flags())
    assert flags.voc_gate_policy_schema_version == 6
    assert flags.voc_gate_confidence_weighted is False
    assert flags.voc_gate_adam_beta1 == 0.0
    assert flags.voc_gate_learning_rate == 0.001
    assert flags.model_float16 is False
    assert flags.float16 is True
    evidence = util._validate_schema6_complete_surface(
        vars(flags), label="processed schema-6 unit flags"
    )
    assert evidence["key_count"] == 228
    assert evidence["v12_projection_key_count"] == 209
    assert evidence["v12_projection_sha256"] == (
        "bd386ad1c78f030409562dea5c34d8866cc128cf99b7f39938f6a090f5987407"
    )


def test_schema7_process_flags_accepts_exact_sealed_v14_surface():
    flags = util.process_flags(_schema7_flags())
    assert flags.voc_gate_policy_schema_version == 7
    assert flags.voc_model_input_seal_schema_version == 1
    evidence = util._validate_schema7_complete_surface(
        vars(flags), label="processed schema-7 unit flags"
    )
    assert evidence["key_count"] == 229
    assert evidence["v12_projection_key_count"] == 209
    assert evidence["v12_projection_sha256"] == (
        util.VOC_GATE_POLICY_SCHEMA6_V12_BASELINE_SHA256
    )


def test_schema8_process_flags_accepts_exact_half_squared_v15_surface():
    flags = util.process_flags(_schema8_flags())
    assert flags.voc_gate_policy_schema_version == 8
    assert flags.voc_model_input_seal_schema_version == 1
    assert not hasattr(flags, "voc_q_regression_loss")
    evidence = util._validate_schema8_complete_surface(
        vars(flags), label="processed schema-8 unit flags"
    )
    assert evidence["key_count"] == 229
    assert evidence["v12_projection_key_count"] == 209
    assert evidence["v12_projection_sha256"] == (
        util.VOC_GATE_POLICY_SCHEMA6_V12_BASELINE_SHA256
    )


def test_schema9_process_flags_accepts_exact_common_mode_v16_surface():
    flags = util.process_flags(_schema9_flags())
    assert flags.voc_gate_policy_schema_version == 9
    assert flags.voc_model_input_seal_schema_version == 1
    for derived in ("voc_q_regression_loss", "voc_q_reconstruction"):
        assert not hasattr(flags, derived)
    evidence = util._validate_schema9_complete_surface(
        vars(flags), label="processed schema-9 unit flags"
    )
    assert evidence["key_count"] == 229
    assert evidence["v12_projection_key_count"] == 209
    assert evidence["v12_projection_sha256"] == (
        util.VOC_GATE_POLICY_SCHEMA6_V12_BASELINE_SHA256
    )


def test_schema10_process_flags_accepts_exact_huber_common_v17_surface():
    flags = util.process_flags(_schema10_flags())
    assert flags.voc_gate_policy_schema_version == 10
    assert flags.voc_model_input_seal_schema_version == 1
    for derived in ("voc_q_regression_loss", "voc_q_reconstruction"):
        assert not hasattr(flags, derived)
    evidence = util._validate_schema10_complete_surface(
        vars(flags), label="processed schema-10 unit flags"
    )
    assert evidence["key_count"] == 229
    assert evidence["v12_projection_key_count"] == 209
    assert evidence["v12_projection_sha256"] == (
        util.VOC_GATE_POLICY_SCHEMA6_V12_BASELINE_SHA256
    )


def test_schema11_process_flags_accepts_exact_orthocd_v18_surface():
    flags = util.process_flags(_schema11_flags())
    assert flags.voc_gate_policy_schema_version == 11
    assert flags.voc_model_input_seal_schema_version == 1
    for derived in (
        "voc_q_regression_loss",
        "voc_q_reconstruction",
        "voc_q_optimizer_coordinates",
    ):
        assert not hasattr(flags, derived)
    evidence = util._validate_schema11_complete_surface(
        vars(flags), label="processed schema-11 unit flags"
    )
    assert evidence["key_count"] == 229
    assert evidence["v12_projection_key_count"] == 209
    assert evidence["v12_projection_sha256"] == (
        util.VOC_GATE_POLICY_SCHEMA6_V12_BASELINE_SHA256
    )


def test_schema7_surface_partitions_are_exact_new10_without_baseline_drift():
    assert len(util.VOC_GATE_POLICY_SCHEMA6_V12_BASELINE) == 209
    assert len(util.VOC_GATE_POLICY_SCHEMA6_STAGE_FIELDS) == 6
    assert len(util.VOC_GATE_POLICY_SCHEMA6_PATH_FIELDS) == 4
    assert len(util.VOC_GATE_POLICY_SCHEMA6_NEW_FIELDS) == 9
    assert len(util.VOC_GATE_POLICY_SCHEMA7_NEW_FIELDS) == 10
    assert util.VOC_GATE_POLICY_SCHEMA7_NEW_FIELDS == (
        util.VOC_GATE_POLICY_SCHEMA6_NEW_FIELDS
        | {"voc_model_input_seal_schema_version"}
    )
    assert len(util._VOC_GATE_POLICY_SCHEMA6_COMPLETE_KEYS) == 228
    assert len(util._VOC_GATE_POLICY_SCHEMA7_COMPLETE_KEYS) == 229
    assert util._VOC_GATE_POLICY_SCHEMA7_COMPLETE_KEYS == (
        util._VOC_GATE_POLICY_SCHEMA6_COMPLETE_KEYS
        | {"voc_model_input_seal_schema_version"}
    )
    assert util.VOC_GATE_POLICY_SCHEMA6_V12_BASELINE_SHA256 == (
        "bd386ad1c78f030409562dea5c34d8866cc128cf99b7f39938f6a090f5987407"
    )


def test_schema8_surface_reuses_exact_new10_and_never_persists_loss_name():
    assert util.VOC_GATE_POLICY_SCHEMA8_NEW_FIELDS == (
        util.VOC_GATE_POLICY_SCHEMA7_NEW_FIELDS
    )
    assert len(util.VOC_GATE_POLICY_SCHEMA8_NEW_FIELDS) == 10
    assert len(util._VOC_GATE_POLICY_SCHEMA8_COMPLETE_KEYS) == 229
    assert util._VOC_GATE_POLICY_SCHEMA8_COMPLETE_KEYS == (
        util._VOC_GATE_POLICY_SCHEMA7_COMPLETE_KEYS
    )
    assert "voc_q_regression_loss" not in (
        util._VOC_GATE_POLICY_SCHEMA8_COMPLETE_KEYS
    )


def test_schema9_surface_reuses_exact229_and_never_persists_derived_names():
    assert util.VOC_GATE_POLICY_SCHEMA9_NEW_FIELDS == (
        util.VOC_GATE_POLICY_SCHEMA8_NEW_FIELDS
    )
    assert len(util.VOC_GATE_POLICY_SCHEMA9_NEW_FIELDS) == 10
    assert util._VOC_GATE_POLICY_SCHEMA9_COMPLETE_KEYS == (
        util._VOC_GATE_POLICY_SCHEMA8_COMPLETE_KEYS
    )
    assert len(util._VOC_GATE_POLICY_SCHEMA9_COMPLETE_KEYS) == 229
    assert not {
        "voc_q_regression_loss",
        "voc_q_reconstruction",
    } & util._VOC_GATE_POLICY_SCHEMA9_COMPLETE_KEYS


def test_schema10_surface_reuses_exact229_and_never_persists_derived_names():
    assert util.VOC_GATE_POLICY_SCHEMA10_NEW_FIELDS == (
        util.VOC_GATE_POLICY_SCHEMA9_NEW_FIELDS
    )
    assert len(util.VOC_GATE_POLICY_SCHEMA10_NEW_FIELDS) == 10
    assert util._VOC_GATE_POLICY_SCHEMA10_COMPLETE_KEYS == (
        util._VOC_GATE_POLICY_SCHEMA9_COMPLETE_KEYS
    )
    assert len(util._VOC_GATE_POLICY_SCHEMA10_COMPLETE_KEYS) == 229
    assert not {
        "voc_q_regression_loss",
        "voc_q_reconstruction",
    } & util._VOC_GATE_POLICY_SCHEMA10_COMPLETE_KEYS


def test_schema11_surface_reuses_exact229_and_never_persists_derived_names():
    assert util.VOC_GATE_POLICY_SCHEMA11_NEW_FIELDS == (
        util.VOC_GATE_POLICY_SCHEMA10_NEW_FIELDS
    )
    assert util._VOC_GATE_POLICY_SCHEMA11_COMPLETE_KEYS == (
        util._VOC_GATE_POLICY_SCHEMA10_COMPLETE_KEYS
    )
    assert len(util._VOC_GATE_POLICY_SCHEMA11_COMPLETE_KEYS) == 229
    assert not {
        "voc_q_regression_loss",
        "voc_q_reconstruction",
        "voc_q_optimizer_coordinates",
    } & util._VOC_GATE_POLICY_SCHEMA11_COMPLETE_KEYS


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("voc_model_input_seal_schema_version", True),
        ("voc_model_input_seal_schema_version", np.int64(1)),
        ("voc_gate_policy_schema_version", True),
        ("voc_gate_policy_schema_version", np.int64(10)),
        ("voc_gate_policy_schema_version", 10.0),
        ("voc_gate_policy_schema_version", "10"),
        ("voc_gate_policy_schema_version", 9),
        ("voc_gate_policy_schema_version", 11),
    ],
)
def test_schema10_process_flags_rejects_coerced_or_wrong_schema(field, bad):
    with pytest.raises(
        ValueError,
        match=rf"{field}|unregistered schema-(9|10|11) stage|schema-(10|11)",
    ):
        util.process_flags(_schema10_flags(**{field: bad}))


def test_schema10_rejects_missing_extra_and_persisted_derived_identity_keys():
    missing = _schema10_flags()
    delattr(missing, "voc_model_input_seal_schema_version")
    with pytest.raises(ValueError, match="voc_model_input_seal_schema_version"):
        util.process_flags(missing)

    for key in (
        "unexpected_schema10_field",
        "voc_q_regression_loss",
        "voc_q_reconstruction",
    ):
        with pytest.raises(ValueError, match="exact 229-key"):
            util.process_flags(_schema10_flags(**{key: "forbidden"}))


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("voc_model_input_seal_schema_version", True),
        ("voc_model_input_seal_schema_version", np.int64(1)),
        ("voc_gate_policy_schema_version", True),
        ("voc_gate_policy_schema_version", np.int64(9)),
        ("voc_gate_policy_schema_version", 9.0),
        ("voc_gate_policy_schema_version", "9"),
        ("voc_gate_policy_schema_version", 8),
        ("voc_gate_policy_schema_version", 10),
    ],
)
def test_schema9_process_flags_rejects_every_coerced_or_wrong_schema(field, bad):
    with pytest.raises(
        ValueError,
        match=rf"{field}|unregistered schema-(8|9|10) stage|schema-9",
    ):
        util.process_flags(_schema9_flags(**{field: bad}))


def test_schema9_rejects_missing_extra_and_persisted_derived_identity_keys():
    missing = _schema9_flags()
    delattr(missing, "voc_model_input_seal_schema_version")
    with pytest.raises(ValueError, match="voc_model_input_seal_schema_version"):
        util.process_flags(missing)

    for key in (
        "unexpected_schema9_field",
        "voc_q_regression_loss",
        "voc_q_reconstruction",
    ):
        with pytest.raises(ValueError, match="exact 229-key"):
            util.process_flags(_schema9_flags(**{key: "forbidden"}))


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("voc_model_input_seal_schema_version", True),
        ("voc_model_input_seal_schema_version", 1.0),
        ("voc_model_input_seal_schema_version", np.int64(1)),
        (
            "voc_model_input_seal_schema_version",
            np.nextafter(1.0, 2.0),
        ),
        ("voc_gate_policy_schema_version", True),
        ("voc_gate_policy_schema_version", np.int64(7)),
        ("voc_gate_policy_schema_version", np.nextafter(7.0, 8.0)),
    ],
)
def test_schema7_process_flags_rejects_typed_and_nextafter_drift(field, bad):
    with pytest.raises(ValueError, match=field):
        util.process_flags(_schema7_flags(**{field: bad}))


def test_schema7_process_flags_rejects_missing_and_extra_surface_keys():
    missing = _schema7_flags()
    delattr(missing, "voc_model_input_seal_schema_version")
    with pytest.raises(ValueError, match="voc_model_input_seal_schema_version"):
        util.process_flags(missing)

    with pytest.raises(ValueError, match="exact 229-key"):
        util.process_flags(_schema7_flags(unexpected_schema7_field=1))


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("voc_model_input_seal_schema_version", True),
        ("voc_model_input_seal_schema_version", np.int64(1)),
        ("voc_gate_policy_schema_version", True),
        ("voc_gate_policy_schema_version", np.int64(8)),
        ("voc_gate_policy_schema_version", "8"),
        ("voc_gate_policy_schema_version", np.nextafter(8.0, 9.0)),
    ],
)
def test_schema8_process_flags_rejects_coerced_schema_identity(field, bad):
    with pytest.raises(ValueError, match=field):
        util.process_flags(_schema8_flags(**{field: bad}))


def test_schema8_process_flags_rejects_missing_extra_and_persisted_loss_key():
    missing = _schema8_flags()
    delattr(missing, "voc_model_input_seal_schema_version")
    with pytest.raises(ValueError, match="voc_model_input_seal_schema_version"):
        util.process_flags(missing)

    for key in ("unexpected_schema8_field", "voc_q_regression_loss"):
        with pytest.raises(ValueError, match="exact 229-key"):
            util.process_flags(_schema8_flags(**{key: "half_squared_td"}))


def test_schema6_and_legacy_resolve_seal_schema_to_zero_without_surface_drift():
    schema6 = util.process_flags(_schema6_flags())
    assert not hasattr(schema6, "voc_model_input_seal_schema_version")
    assert util.get_voc_protocol(schema6)[
        "voc_model_input_seal_schema_version"
    ] == 0
    legacy = util.process_flags(_flags())
    assert not hasattr(legacy, "voc_model_input_seal_schema_version")
    assert util.get_voc_protocol(legacy)[
        "voc_model_input_seal_schema_version"
    ] == 0
    with pytest.raises(ValueError, match="requires the atomic"):
        util.process_flags(
            _flags(voc_model_input_seal_schema_version=1)
        )


def test_schema7_gate_resolution_extends_only_the_historical_schema6_shape():
    historical_keys = (
        "voc_gate_policy_schema_version",
        "voc_gate_adam_beta1",
        "voc_gate_adam_beta1_legacy_defaulted",
        "voc_gate_param_align",
        "voc_gate_param_align_coef",
        "voc_gate_param_align_legacy_defaulted",
        "voc_gate_exact_projection",
        "voc_gate_exact_projection_legacy_defaulted",
        "voc_gate_epsilon_greedy_execution",
        "voc_gate_epsilon_greedy_execution_legacy_defaulted",
        "voc_gate_execution_epsilon",
        "voc_actor_policy_version_barrier",
        "voc_actor_policy_bundle_schema_version",
        "voc_actor_policy_barrier_timeout_s",
        "voc_actor_policy_ray_max_restarts",
        "voc_actor_policy_ray_max_task_retries",
        "actor_amp_init_scale",
        "ppo_k",
        "self_play_n",
        "env_n",
        "actor_batch_size",
        "voc_gate_execution_epsilon_legacy_defaulted",
        "voc_actor_policy_version_barrier_legacy_defaulted",
        "voc_actor_policy_bundle_schema_version_legacy_defaulted",
        "voc_actor_policy_barrier_timeout_s_legacy_defaulted",
        "voc_actor_policy_ray_max_restarts_legacy_defaulted",
        "voc_actor_policy_ray_max_task_retries_legacy_defaulted",
        "actor_amp_init_scale_legacy_defaulted",
    )
    schema6_flags = util.process_flags(_schema6_flags())
    schema6 = util.validate_voc_gate_policy_schema({
        "voc_gate_policy_schema_version": 6,
        "flags": vars(schema6_flags),
    })
    assert tuple(schema6) == historical_keys

    schema7_flags = util.process_flags(_schema7_flags())
    schema7 = util.validate_voc_gate_policy_schema({
        "voc_gate_policy_schema_version": 7,
        "flags": vars(schema7_flags),
    })
    assert set(schema7) == set(historical_keys) | {
        "voc_model_input_seal_schema_version",
        "voc_model_input_seal_schema_version_legacy_defaulted",
    }
    assert schema7["voc_model_input_seal_schema_version"] == 1
    assert schema7[
        "voc_model_input_seal_schema_version_legacy_defaulted"
    ] is False

    schema8_flags = util.process_flags(_schema8_flags())
    schema8 = util.validate_voc_gate_policy_schema({
        "voc_gate_policy_schema_version": 8,
        "flags": vars(schema8_flags),
    })
    assert tuple(schema8) == tuple(schema7)
    assert "voc_q_regression_loss" not in schema8
    assert "voc_q_regression_loss" not in schema7
    assert "voc_q_regression_loss" not in schema6

    schema9_flags = util.process_flags(_schema9_flags())
    schema9 = util.validate_voc_gate_policy_schema({
        "voc_gate_policy_schema_version": 9,
        "flags": vars(schema9_flags),
    })
    assert tuple(schema9) == tuple(schema8)
    assert "voc_q_regression_loss" not in schema9
    assert "voc_q_reconstruction" not in schema9

    schema10_flags = util.process_flags(_schema10_flags())
    schema10 = util.validate_voc_gate_policy_schema({
        "voc_gate_policy_schema_version": 10,
        "flags": vars(schema10_flags),
    })
    assert tuple(schema10) == tuple(schema9)
    assert "voc_q_regression_loss" not in schema10
    assert "voc_q_reconstruction" not in schema10

    with pytest.raises(ValueError, match="built-in Python integer"):
        util.validate_voc_gate_policy_schema({
            "voc_gate_policy_schema_version": np.int64(9),
            "flags": vars(schema9_flags),
        })
    with pytest.raises(ValueError, match="built-in Python integer"):
        util.validate_voc_gate_policy_schema({
            "voc_gate_policy_schema_version": np.int64(10),
            "flags": vars(schema10_flags),
        })


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
def test_schema6_process_flags_rejects_any_v12_baseline_drift(field, bad):
    with pytest.raises(ValueError, match=field):
        util.process_flags(_schema6_flags(**{field: bad}))


def test_schema6_process_flags_rejects_missing_extra_and_typed_drift():
    missing = _schema6_flags()
    delattr(missing, "discounting")
    with pytest.raises(ValueError, match="exact 228-key"):
        util.process_flags(missing)

    with pytest.raises(ValueError, match="extra=.*unexpected"):
        util.process_flags(_schema6_flags(unexpected="forbidden"))

    with pytest.raises(ValueError, match="actor_batch_size"):
        util.process_flags(_schema6_flags(actor_batch_size=16.0))


def test_schema6_process_flags_accepts_only_three_closed_stage_profiles():
    for profile in util.VOC_GATE_POLICY_SCHEMA6_STAGE_PROFILES:
        xpid, seed, total, warmup, unroll, use_wandb = profile
        flags = _schema6_flags(
            xpid=xpid,
            base_seed=seed,
            total_steps=total,
            model_warm_up_n=warmup,
            actor_unroll_len=unroll,
            use_wandb=use_wandb,
            ckpdir=(
                "/tmp/di-voc-v13-versioned-eps25-test/runs/" + xpid
            ),
        )
        assert util.process_flags(flags).xpid == xpid

    with pytest.raises(ValueError, match="unregistered schema-6 stage"):
        util.process_flags(_schema6_flags(xpid=" schema6-whitespace"))


def test_schema7_process_flags_accepts_only_three_closed_stage_profiles():
    for profile in util.VOC_GATE_POLICY_SCHEMA7_STAGE_PROFILES:
        xpid, seed, total, warmup, unroll, use_wandb = profile
        flags = _schema7_flags(
            xpid=xpid,
            base_seed=seed,
            total_steps=total,
            model_warm_up_n=warmup,
            actor_unroll_len=unroll,
            use_wandb=use_wandb,
            ckpdir=(
                "/tmp/di-voc-v14-sealed-eps25-test/runs/" + xpid
            ),
        )
        assert util.process_flags(flags).xpid == xpid

    for forbidden in (
        util.VOC_GATE_POLICY_SCHEMA6_STAGE_PROFILES[0][0],
        " enduro-voc-v14-sealed-eps25-sentinel-wire1200",
        "enduro-voc-v14-sealed-eps25-sentinel-wire1200 ",
    ):
        with pytest.raises(ValueError, match="unregistered schema-7 stage"):
            util.process_flags(_schema7_flags(xpid=forbidden))


def test_schema8_process_flags_accepts_only_three_closed_stage_profiles():
    for profile in util.VOC_GATE_POLICY_SCHEMA8_STAGE_PROFILES:
        xpid, seed, total, warmup, unroll, use_wandb = profile
        flags = _schema8_flags(
            xpid=xpid,
            base_seed=seed,
            total_steps=total,
            model_warm_up_n=warmup,
            actor_unroll_len=unroll,
            use_wandb=use_wandb,
            ckpdir=(
                "/tmp/di-voc-v15-halfsq-eps25-test/runs/" + xpid
            ),
        )
        assert util.process_flags(flags).xpid == xpid

    for forbidden in (
        util.VOC_GATE_POLICY_SCHEMA7_STAGE_PROFILES[0][0],
        " enduro-voc-v15-halfsq-eps25-sentinel-wire1200",
        "enduro-voc-v15-halfsq-eps25-sentinel-wire1200 ",
    ):
        with pytest.raises(ValueError, match="unregistered schema-8 stage"):
            util.process_flags(_schema8_flags(xpid=forbidden))


def test_schema9_process_flags_accepts_only_exact_three_closed_stage_tuples():
    for profile in util.VOC_GATE_POLICY_SCHEMA9_STAGE_PROFILES:
        xpid, seed, total, warmup, unroll, use_wandb = profile
        flags = _schema9_flags(
            xpid=xpid,
            base_seed=seed,
            total_steps=total,
            model_warm_up_n=warmup,
            actor_unroll_len=unroll,
            use_wandb=use_wandb,
            ckpdir=(
                "/tmp/di-voc-v16-commonmode-eps25-test/runs/" + xpid
            ),
        )
        assert util.process_flags(flags).xpid == xpid

    wire = util.VOC_GATE_POLICY_SCHEMA9_STAGE_PROFILES[0]
    qualification = util.VOC_GATE_POLICY_SCHEMA9_STAGE_PROFILES[1]
    primary = util.VOC_GATE_POLICY_SCHEMA9_STAGE_PROFILES[2]
    cross_stage_mutations = (
        {"xpid": util.VOC_GATE_POLICY_SCHEMA8_STAGE_PROFILES[0][0]},
        {"xpid": wire[0], "base_seed": primary[1]},
        {"xpid": wire[0], "total_steps": qualification[2]},
        {"xpid": qualification[0], "use_wandb": wire[5]},
        {"xpid": primary[0], "actor_unroll_len": wire[4]},
        {"xpid": " " + wire[0]},
        {"xpid": wire[0] + " "},
    )
    for mutation in cross_stage_mutations:
        with pytest.raises(ValueError, match="unregistered schema-9 stage"):
            util.process_flags(_schema9_flags(**mutation))


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("xpid", np.str_("enduro-voc-v16-commonmode-eps25-sentinel-wire1200")),
        ("base_seed", True),
        ("base_seed", np.int64(1)),
        ("total_steps", 1200.0),
        ("model_warm_up_n", "512"),
        ("actor_unroll_len", np.int64(41)),
        ("use_wandb", np.bool_(False)),
    ],
)
def test_schema9_stage_members_require_exact_builtin_types(field, bad):
    with pytest.raises(ValueError, match=field):
        util.process_flags(_schema9_flags(**{field: bad}))


@pytest.mark.parametrize(
    "field", ["preload", "preload_actor", "voc_parent_checkpoint"]
)
def test_schema9_is_fresh_only_and_rejects_all_parent_surfaces(field):
    with pytest.raises(ValueError, match=field):
        util.process_flags(_schema9_flags(**{field: "/tmp/forbidden"}))


def test_schema10_process_flags_accepts_only_exact_three_closed_stage_tuples():
    for profile in util.VOC_GATE_POLICY_SCHEMA10_STAGE_PROFILES:
        xpid, seed, total, warmup, unroll, use_wandb = profile
        flags = _schema10_flags(
            xpid=xpid,
            base_seed=seed,
            total_steps=total,
            model_warm_up_n=warmup,
            actor_unroll_len=unroll,
            use_wandb=use_wandb,
            ckpdir=(
                "/tmp/di-voc-v17-huber-common-eps25-test/runs/" + xpid
            ),
        )
        assert util.process_flags(flags).xpid == xpid

    wire = util.VOC_GATE_POLICY_SCHEMA10_STAGE_PROFILES[0]
    qualification = util.VOC_GATE_POLICY_SCHEMA10_STAGE_PROFILES[1]
    primary = util.VOC_GATE_POLICY_SCHEMA10_STAGE_PROFILES[2]
    cross_stage_mutations = (
        {"xpid": util.VOC_GATE_POLICY_SCHEMA9_STAGE_PROFILES[0][0]},
        {"xpid": wire[0], "base_seed": primary[1]},
        {"xpid": wire[0], "total_steps": qualification[2]},
        {"xpid": qualification[0], "use_wandb": wire[5]},
        {"xpid": primary[0], "actor_unroll_len": wire[4]},
        {"xpid": " " + wire[0]},
        {"xpid": wire[0] + " "},
    )
    for mutation in cross_stage_mutations:
        with pytest.raises(ValueError, match="unregistered schema-10 stage"):
            util.process_flags(_schema10_flags(**mutation))


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        (
            "xpid",
            np.str_("enduro-voc-v17-huber-common-eps25-sentinel-wire1200"),
        ),
        ("base_seed", True),
        ("base_seed", np.int64(1)),
        ("total_steps", 1200.0),
        ("model_warm_up_n", "512"),
        ("actor_unroll_len", np.int64(41)),
        ("use_wandb", np.bool_(False)),
    ],
)
def test_schema10_stage_members_require_exact_builtin_types(field, bad):
    with pytest.raises(ValueError, match=field):
        util.process_flags(_schema10_flags(**{field: bad}))


@pytest.mark.parametrize(
    "field", ["preload", "preload_actor", "voc_parent_checkpoint"]
)
def test_schema10_is_fresh_only_and_rejects_all_parent_surfaces(field):
    with pytest.raises(ValueError, match=field):
        util.process_flags(_schema10_flags(**{field: "/tmp/forbidden"}))


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("base_seed", True),
        ("base_seed", np.int64(1)),
        ("total_steps", 1200.0),
        ("model_warm_up_n", "512"),
        ("actor_unroll_len", np.int64(41)),
        ("use_wandb", np.bool_(False)),
    ],
)
def test_schema8_stage_members_are_strict_python_types(field, bad):
    with pytest.raises(ValueError, match=field):
        util.process_flags(_schema8_flags(**{field: bad}))


@pytest.mark.parametrize(
    "field", ["preload", "preload_actor", "voc_parent_checkpoint"]
)
def test_schema8_is_fresh_only_and_rejects_every_preload_surface(field):
    with pytest.raises(ValueError, match=field):
        util.process_flags(_schema8_flags(**{field: "/tmp/forbidden"}))

    if field == "preload":
        with pytest.raises(ValueError, match="fresh-only"):
            util.process_flags(_schema8_flags(ckp=True))


def test_schema6_process_flags_binds_run_and_behavioral_data_paths():
    with pytest.raises(ValueError, match=r"join\(savedir, xpid\)"):
        util.process_flags(_schema6_flags(ckpdir="/tmp/wrong/run"))
    with pytest.raises(ValueError, match="staged data path"):
        util.process_flags(
            _schema6_flags(icopro_data_path="/tmp/wrong/data")
        )


def test_schema6_create_flags_produces_exact_228_key_training_surface(
    monkeypatch
):
    expected = vars(_schema6_flags()).copy()
    for derived in (
        "__version__",
        "git_revision",
        "cmd",
        "ckpdir",
        "voc_gate_policy_schema_version",
        "voc_actor_policy_barrier_runtime",
    ):
        expected.pop(derived)
    monkeypatch.setattr(util, "get_git_revision_hash", lambda: None)
    monkeypatch.setattr(util.sys, "argv", ["train.py", "--schema6-wire"])
    flags = util.create_flags(
        ["default_thinker.yaml", "default_actor.yaml"],
        save_flags=False,
        post_fn=util.process_flags_actor,
        **expected,
    )
    evidence = util._validate_schema6_complete_surface(
        vars(flags), label="create_flags schema-6 output"
    )
    assert evidence["key_count"] == 228
    assert evidence["v12_projection_sha256"] == (
        util.VOC_GATE_POLICY_SCHEMA6_V12_BASELINE_SHA256
    )


def test_schema7_create_flags_produces_exact_229_key_training_surface(
    monkeypatch,
):
    expected = vars(_schema7_flags()).copy()
    for derived in (
        "__version__",
        "git_revision",
        "cmd",
        "ckpdir",
        "voc_gate_policy_schema_version",
        "voc_actor_policy_barrier_runtime",
    ):
        expected.pop(derived)
    monkeypatch.setattr(util, "get_git_revision_hash", lambda: None)
    monkeypatch.setattr(util.sys, "argv", ["train.py", "--schema7-wire"])
    flags = util.create_flags(
        ["default_thinker.yaml", "default_actor.yaml"],
        save_flags=False,
        post_fn=util.process_flags_actor,
        **expected,
    )
    evidence = util._validate_schema7_complete_surface(
        vars(flags), label="create_flags schema-7 output"
    )
    assert evidence["key_count"] == 229
    assert evidence["v12_projection_sha256"] == (
        util.VOC_GATE_POLICY_SCHEMA6_V12_BASELINE_SHA256
    )


def test_schema8_create_flags_produces_exact_229_key_training_surface(
    monkeypatch,
):
    expected = vars(_schema8_flags()).copy()
    for derived in (
        "__version__",
        "git_revision",
        "cmd",
        "ckpdir",
        "voc_gate_policy_schema_version",
        "voc_actor_policy_barrier_runtime",
    ):
        expected.pop(derived)
    monkeypatch.setattr(util, "get_git_revision_hash", lambda: None)
    monkeypatch.setattr(util.sys, "argv", ["train.py", "--schema8-wire"])
    flags = util.create_flags(
        ["default_thinker.yaml", "default_actor.yaml"],
        save_flags=False,
        post_fn=util.process_flags_actor,
        **expected,
    )
    evidence = util._validate_schema8_complete_surface(
        vars(flags), label="create_flags schema-8 output"
    )
    assert flags.voc_gate_policy_schema_version == 8
    assert not hasattr(flags, "voc_q_regression_loss")
    assert evidence["key_count"] == 229
    assert evidence["v12_projection_sha256"] == (
        util.VOC_GATE_POLICY_SCHEMA6_V12_BASELINE_SHA256
    )


def test_schema9_create_flags_produces_exact229_without_derived_keys(
    monkeypatch,
):
    expected = vars(_schema9_flags()).copy()
    for derived in (
        "__version__",
        "git_revision",
        "cmd",
        "ckpdir",
        "voc_gate_policy_schema_version",
        "voc_actor_policy_barrier_runtime",
    ):
        expected.pop(derived)
    monkeypatch.setattr(util, "get_git_revision_hash", lambda: None)
    monkeypatch.setattr(util.sys, "argv", ["train.py", "--schema9-wire"])
    flags = util.create_flags(
        ["default_thinker.yaml", "default_actor.yaml"],
        save_flags=False,
        post_fn=util.process_flags_actor,
        **expected,
    )
    evidence = util._validate_schema9_complete_surface(
        vars(flags), label="create_flags schema-9 output"
    )
    assert flags.voc_gate_policy_schema_version == 9
    assert evidence["key_count"] == 229
    assert evidence["v12_projection_key_count"] == 209
    assert evidence["v12_projection_sha256"] == (
        util.VOC_GATE_POLICY_SCHEMA6_V12_BASELINE_SHA256
    )
    assert not {
        "voc_q_regression_loss",
        "voc_q_reconstruction",
    } & vars(flags).keys()


def test_schema10_create_flags_produces_exact229_without_derived_keys(
    monkeypatch,
):
    expected = vars(_schema10_flags()).copy()
    for derived in (
        "__version__",
        "git_revision",
        "cmd",
        "ckpdir",
        "voc_gate_policy_schema_version",
        "voc_actor_policy_barrier_runtime",
    ):
        expected.pop(derived)
    monkeypatch.setattr(util, "get_git_revision_hash", lambda: None)
    monkeypatch.setattr(util.sys, "argv", ["train.py", "--schema10-wire"])
    flags = util.create_flags(
        ["default_thinker.yaml", "default_actor.yaml"],
        save_flags=False,
        post_fn=util.process_flags_actor,
        **expected,
    )
    evidence = util._validate_schema10_complete_surface(
        vars(flags), label="create_flags schema-10 output"
    )
    assert flags.voc_gate_policy_schema_version == 10
    assert evidence["key_count"] == 229
    assert evidence["v12_projection_key_count"] == 209
    assert evidence["v12_projection_sha256"] == (
        util.VOC_GATE_POLICY_SCHEMA6_V12_BASELINE_SHA256
    )
    assert not {
        "voc_q_regression_loss",
        "voc_q_reconstruction",
    } & vars(flags).keys()


@pytest.mark.parametrize(
    "profile", util.VOC_GATE_POLICY_SCHEMA7_STAGE_PROFILES
)
def test_schema7_actual_train_cli_infers_schema_from_exact_seal_flag(
    monkeypatch, profile
):
    args = _schema7_wire_cli_args()
    xpid, seed, total, warmup, unroll, use_wandb = profile
    for name, value in (
        ("xpid", xpid),
        ("base_seed", seed),
        ("total_steps", total),
        ("model_warm_up_n", warmup),
        ("actor_unroll_len", unroll),
        ("use_wandb", use_wandb),
    ):
        args = _replace_cli_value(args, name, value)

    parsed = util.add_parse(
        ["default_thinker.yaml", "default_actor.yaml"]
    ).parse_args(args)
    assert parsed.voc_model_input_seal_schema_version == 1
    assert not hasattr(parsed, "voc_gate_policy_schema_version")
    monkeypatch.setattr(util, "get_git_revision_hash", lambda: None)
    monkeypatch.setattr(util.sys, "argv", ["train.py", *args])
    flags = util.create_setting(args=args, save_flags=False)
    assert flags.voc_gate_policy_schema_version == 7
    assert flags.voc_model_input_seal_schema_version == 1
    assert (
        flags.xpid,
        flags.base_seed,
        flags.total_steps,
        flags.model_warm_up_n,
        flags.actor_unroll_len,
        flags.use_wandb,
    ) == profile
    assert util._validate_schema7_complete_surface(
        vars(flags), label="actual schema-7 train CLI"
    )["key_count"] == 229


@pytest.mark.parametrize(
    "profile", util.VOC_GATE_POLICY_SCHEMA8_STAGE_PROFILES
)
def test_schema8_actual_train_cli_infers_only_exact_v15_stage(
    monkeypatch, profile
):
    args = _schema8_wire_cli_args()
    xpid, seed, total, warmup, unroll, use_wandb = profile
    for name, value in (
        ("xpid", xpid),
        ("base_seed", seed),
        ("total_steps", total),
        ("model_warm_up_n", warmup),
        ("actor_unroll_len", unroll),
        ("use_wandb", use_wandb),
    ):
        args = _replace_cli_value(args, name, value)

    parsed = util.add_parse(
        ["default_thinker.yaml", "default_actor.yaml"]
    ).parse_args(args)
    assert parsed.voc_model_input_seal_schema_version == 1
    assert not hasattr(parsed, "voc_gate_policy_schema_version")
    assert not hasattr(parsed, "voc_q_regression_loss")
    monkeypatch.setattr(util, "get_git_revision_hash", lambda: None)
    monkeypatch.setattr(util.sys, "argv", ["train.py", *args])
    flags = util.create_setting(args=args, save_flags=False)
    assert flags.voc_gate_policy_schema_version == 8
    assert flags.voc_model_input_seal_schema_version == 1
    assert not hasattr(flags, "voc_q_regression_loss")
    assert (
        flags.xpid,
        flags.base_seed,
        flags.total_steps,
        flags.model_warm_up_n,
        flags.actor_unroll_len,
        flags.use_wandb,
    ) == profile
    evidence = util._validate_schema8_complete_surface(
        vars(flags), label="actual schema-8 train CLI"
    )
    assert evidence["key_count"] == 229
    assert evidence["v12_projection_key_count"] == 209


@pytest.mark.parametrize(
    "profile", util.VOC_GATE_POLICY_SCHEMA9_STAGE_PROFILES
)
def test_schema9_actual_train_cli_infers_only_exact_v16_stage(
    monkeypatch, profile
):
    args = _schema9_wire_cli_args()
    xpid, seed, total, warmup, unroll, use_wandb = profile
    for name, value in (
        ("xpid", xpid),
        ("base_seed", seed),
        ("total_steps", total),
        ("model_warm_up_n", warmup),
        ("actor_unroll_len", unroll),
        ("use_wandb", use_wandb),
    ):
        args = _replace_cli_value(args, name, value)

    parsed = util.add_parse(
        ["default_thinker.yaml", "default_actor.yaml"]
    ).parse_args(args)
    assert parsed.voc_model_input_seal_schema_version == 1
    assert not hasattr(parsed, "voc_gate_policy_schema_version")
    for derived in ("voc_q_regression_loss", "voc_q_reconstruction"):
        assert not hasattr(parsed, derived)
    monkeypatch.setattr(util, "get_git_revision_hash", lambda: None)
    monkeypatch.setattr(util.sys, "argv", ["train.py", *args])
    flags = util.create_setting(args=args, save_flags=False)
    assert flags.voc_gate_policy_schema_version == 9
    assert flags.voc_model_input_seal_schema_version == 1
    assert (
        flags.xpid,
        flags.base_seed,
        flags.total_steps,
        flags.model_warm_up_n,
        flags.actor_unroll_len,
        flags.use_wandb,
    ) == profile
    evidence = util._validate_schema9_complete_surface(
        vars(flags), label="actual schema-9 train CLI"
    )
    assert evidence["key_count"] == 229
    assert evidence["v12_projection_key_count"] == 209


@pytest.mark.parametrize(
    "profile", util.VOC_GATE_POLICY_SCHEMA10_STAGE_PROFILES
)
def test_schema10_actual_train_cli_infers_only_exact_v17_stage(
    monkeypatch, profile
):
    args = _schema10_wire_cli_args()
    xpid, seed, total, warmup, unroll, use_wandb = profile
    for name, value in (
        ("xpid", xpid),
        ("base_seed", seed),
        ("total_steps", total),
        ("model_warm_up_n", warmup),
        ("actor_unroll_len", unroll),
        ("use_wandb", use_wandb),
    ):
        args = _replace_cli_value(args, name, value)

    parsed = util.add_parse(
        ["default_thinker.yaml", "default_actor.yaml"]
    ).parse_args(args)
    assert parsed.voc_model_input_seal_schema_version == 1
    assert not hasattr(parsed, "voc_gate_policy_schema_version")
    for derived in ("voc_q_regression_loss", "voc_q_reconstruction"):
        assert not hasattr(parsed, derived)
    monkeypatch.setattr(util, "get_git_revision_hash", lambda: None)
    monkeypatch.setattr(util.sys, "argv", ["train.py", *args])
    flags = util.create_setting(args=args, save_flags=False)
    assert flags.voc_gate_policy_schema_version == 10
    assert flags.voc_model_input_seal_schema_version == 1
    assert (
        flags.xpid,
        flags.base_seed,
        flags.total_steps,
        flags.model_warm_up_n,
        flags.actor_unroll_len,
        flags.use_wandb,
    ) == profile
    evidence = util._validate_schema10_complete_surface(
        vars(flags), label="actual schema-10 train CLI"
    )
    assert evidence["key_count"] == 229
    assert evidence["v12_projection_key_count"] == 209
    assert evidence["v12_projection_sha256"] == (
        util.VOC_GATE_POLICY_SCHEMA6_V12_BASELINE_SHA256
    )


@pytest.mark.parametrize(
    "bad_ckp",
    [True, 1, 0, np.bool_(True), np.bool_(False), "True", "False"],
)
def test_schema9_resume_rejects_before_config_io_or_run_directory(
    monkeypatch, tmp_path, bad_ckp
):
    snapshot_root = tmp_path / "snapshot"
    savedir = snapshot_root / "runs"
    xpid = util.VOC_GATE_POLICY_SCHEMA9_STAGE_PROFILES[0][0]
    args = _replace_cli_value(_schema9_wire_cli_args(), "savedir", savedir)
    args = _replace_cli_value(
        args,
        "icopro_data_path",
        snapshot_root / "data" / "behavioral_data_block",
    )
    opened_configs = []
    real_open = open

    def tracking_open(path, *open_args, **open_kwargs):
        if str(path).endswith("config_c.yaml"):
            opened_configs.append(str(path))
        return real_open(path, *open_args, **open_kwargs)

    monkeypatch.setattr("builtins.open", tracking_open)
    monkeypatch.setattr(util, "get_git_revision_hash", lambda: None)
    with pytest.raises(ValueError, match="schema-9 is fresh-only"):
        util.create_setting(
            args=args,
            save_flags=True,
            ckp=bad_ckp,
        )
    assert opened_configs == []
    assert not (savedir / xpid).exists()


@pytest.mark.parametrize(
    "bad_xpid",
    [
        " enduro-voc-v16-commonmode-eps25-sentinel-wire1200",
        "enduro-voc-v16-commonmode-eps25-sentinel-wire1200 ",
        "enduro-voc-v16-commonmode-eps25-unregistered",
    ],
)
def test_schema9_malformed_v16_xpid_resume_still_fails_before_config_io(
    monkeypatch, tmp_path, bad_xpid
):
    snapshot_root = tmp_path / "snapshot"
    savedir = snapshot_root / "runs"
    args = _replace_cli_value(_schema9_wire_cli_args(), "xpid", bad_xpid)
    args = _replace_cli_value(args, "savedir", savedir)
    args = _replace_cli_value(
        args,
        "icopro_data_path",
        snapshot_root / "data" / "behavioral_data_block",
    )
    args = _replace_cli_value(args, "ckp", True)
    opened_configs = []
    real_open = open

    def tracking_open(path, *open_args, **open_kwargs):
        if str(path).endswith("config_c.yaml"):
            opened_configs.append(str(path))
        return real_open(path, *open_args, **open_kwargs)

    monkeypatch.setattr("builtins.open", tracking_open)
    monkeypatch.setattr(util, "get_git_revision_hash", lambda: None)
    with pytest.raises(ValueError, match="schema-9 is fresh-only"):
        util.create_setting(args=args, save_flags=True)
    assert opened_configs == []
    assert not (savedir / bad_xpid).exists()


@pytest.mark.parametrize(
    "bad_ckp",
    [True, 1, 0, np.bool_(True), np.bool_(False), "True", "False"],
)
def test_schema10_resume_rejects_before_config_io_or_run_directory(
    monkeypatch, tmp_path, bad_ckp
):
    snapshot_root = tmp_path / "snapshot"
    savedir = snapshot_root / "runs"
    xpid = util.VOC_GATE_POLICY_SCHEMA10_STAGE_PROFILES[0][0]
    args = _replace_cli_value(_schema10_wire_cli_args(), "savedir", savedir)
    args = _replace_cli_value(
        args,
        "icopro_data_path",
        snapshot_root / "data" / "behavioral_data_block",
    )
    opened_configs = []
    real_open = open

    def tracking_open(path, *open_args, **open_kwargs):
        if str(path).endswith("config_c.yaml"):
            opened_configs.append(str(path))
        return real_open(path, *open_args, **open_kwargs)

    monkeypatch.setattr("builtins.open", tracking_open)
    monkeypatch.setattr(util, "get_git_revision_hash", lambda: None)
    with pytest.raises(ValueError, match="schema-10 is fresh-only"):
        util.create_setting(
            args=args,
            save_flags=True,
            ckp=bad_ckp,
        )
    assert opened_configs == []
    assert not (savedir / xpid).exists()


@pytest.mark.parametrize(
    "bad_xpid",
    [
        " enduro-voc-v17-huber-common-eps25-sentinel-wire1200",
        "enduro-voc-v17-huber-common-eps25-sentinel-wire1200 ",
        "enduro-voc-v17-huber-common-eps25-unregistered",
    ],
)
def test_schema10_malformed_v17_xpid_resume_fails_before_config_io(
    monkeypatch, tmp_path, bad_xpid
):
    snapshot_root = tmp_path / "snapshot"
    savedir = snapshot_root / "runs"
    args = _replace_cli_value(_schema10_wire_cli_args(), "xpid", bad_xpid)
    args = _replace_cli_value(args, "savedir", savedir)
    args = _replace_cli_value(
        args,
        "icopro_data_path",
        snapshot_root / "data" / "behavioral_data_block",
    )
    args = _replace_cli_value(args, "ckp", True)
    opened_configs = []
    real_open = open

    def tracking_open(path, *open_args, **open_kwargs):
        if str(path).endswith("config_c.yaml"):
            opened_configs.append(str(path))
        return real_open(path, *open_args, **open_kwargs)

    monkeypatch.setattr("builtins.open", tracking_open)
    monkeypatch.setattr(util, "get_git_revision_hash", lambda: None)
    with pytest.raises(ValueError, match="schema-10 is fresh-only"):
        util.create_setting(args=args, save_flags=True)
    assert opened_configs == []
    assert not (savedir / bad_xpid).exists()


class _Schema10StringSubclass(str):
    pass


class _Schema10PathLike:
    def __init__(self, value):
        self.value = value

    def __fspath__(self):
        return self.value


class _Schema10Stringable:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return self.value


@pytest.mark.parametrize(
    "typed_xpid",
    [
        np.str_("enduro-voc-v17-huber-common-eps25-sentinel-wire1200"),
        _Schema10StringSubclass(
            "enduro-voc-v17-huber-common-eps25-sentinel-wire1200"
        ),
    ],
)
@pytest.mark.parametrize("bad_ckp", [True, 1, np.bool_(True), "True"])
def test_schema10_typed_xpid_resume_intent_fails_before_config_io(
    monkeypatch, tmp_path, typed_xpid, bad_ckp
):
    snapshot_root = tmp_path / "snapshot"
    savedir = snapshot_root / "runs"
    args = _replace_cli_value(_schema10_wire_cli_args(), "savedir", savedir)
    args = _replace_cli_value(
        args,
        "icopro_data_path",
        snapshot_root / "data" / "behavioral_data_block",
    )
    opened_configs = []
    real_open = open

    def tracking_open(path, *open_args, **open_kwargs):
        if str(path).endswith("config_c.yaml"):
            opened_configs.append(str(path))
        return real_open(path, *open_args, **open_kwargs)

    monkeypatch.setattr("builtins.open", tracking_open)
    monkeypatch.setattr(util, "get_git_revision_hash", lambda: None)
    with pytest.raises(ValueError, match="schema-10 is fresh-only"):
        util.create_setting(
            args=args,
            save_flags=True,
            xpid=typed_xpid,
            ckp=bad_ckp,
            voc_gate_policy_schema_version=9,
        )
    assert opened_configs == []
    assert not (savedir / str(typed_xpid)).exists()


def test_schema8_actual_train_cli_rejects_wrong_stage_and_resume_before_load(
    monkeypatch,
):
    wrong_stage = _replace_cli_value(
        _schema8_wire_cli_args(),
        "base_seed",
        5,
    )
    monkeypatch.setattr(util, "get_git_revision_hash", lambda: None)
    with pytest.raises(ValueError, match="unregistered schema-8 stage"):
        util.create_setting(args=wrong_stage, save_flags=False)

    resume = _replace_cli_value(_schema8_wire_cli_args(), "ckp", True)
    opened = []
    real_open = open

    def tracking_open(path, *args, **kwargs):
        if str(path).endswith("config_c.yaml"):
            opened.append(str(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", tracking_open)
    with pytest.raises(ValueError, match="schema-8 is fresh-only"):
        util.create_setting(args=resume, save_flags=False)
    assert opened == []


@pytest.mark.parametrize(
    "bad_ckp",
    [True, 1, 0, np.bool_(True), np.bool_(False), "True", "False"],
)
def test_schema8_create_setting_rejects_nonexact_false_ckp_before_io(
    monkeypatch, tmp_path, bad_ckp
):
    snapshot_root = tmp_path / "snapshot"
    savedir = snapshot_root / "runs"
    xpid = util.VOC_GATE_POLICY_SCHEMA8_STAGE_PROFILES[0][0]
    args = _replace_cli_value(
        _schema8_wire_cli_args(), "savedir", savedir
    )
    args = _replace_cli_value(
        args,
        "icopro_data_path",
        snapshot_root / "data" / "behavioral_data_block",
    )
    opened_configs = []
    real_open = open

    def tracking_open(path, *args, **kwargs):
        if str(path).endswith("config_c.yaml"):
            opened_configs.append(str(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", tracking_open)
    monkeypatch.setattr(util, "get_git_revision_hash", lambda: None)
    with pytest.raises(ValueError, match="exact Python bool False"):
        util.create_setting(
            args=args,
            save_flags=True,
            ckp=bad_ckp,
        )
    assert opened_configs == []
    assert not (savedir / xpid).exists()


@pytest.mark.parametrize(
    "typed_xpid",
    [
        Path("enduro-voc-v17-huber-common-eps25-sentinel-wire1200"),
        _Schema10PathLike(
            "enduro-voc-v17-huber-common-eps25-sentinel-wire1200"
        ),
        _Schema10PathLike(
            b"enduro-voc-v17-huber-common-eps25-sentinel-wire1200"
        ),
        b"enduro-voc-v17-huber-common-eps25-sentinel-wire1200",
        np.bytes_(
            b"enduro-voc-v17-huber-common-eps25-sentinel-wire1200"
        ),
        bytearray(
            b"enduro-voc-v17-huber-common-eps25-sentinel-wire1200"
        ),
        memoryview(
            b"enduro-voc-v17-huber-common-eps25-sentinel-wire1200"
        ),
        _Schema10Stringable(
            "enduro-voc-v17-huber-common-eps25-sentinel-wire1200"
        ),
    ],
)
def test_schema10_lexical_xpid_resume_intent_fails_before_config_io(
    monkeypatch, tmp_path, typed_xpid
):
    snapshot_root = tmp_path / "snapshot"
    savedir = snapshot_root / "runs"
    args = _replace_cli_value(_schema10_wire_cli_args(), "savedir", savedir)
    args = _replace_cli_value(
        args,
        "icopro_data_path",
        snapshot_root / "data" / "behavioral_data_block",
    )
    opened_configs = []
    real_open = open

    def tracking_open(path, *open_args, **open_kwargs):
        if str(path).endswith("config_c.yaml"):
            opened_configs.append(str(path))
        return real_open(path, *open_args, **open_kwargs)

    monkeypatch.setattr("builtins.open", tracking_open)
    monkeypatch.setattr(util, "get_git_revision_hash", lambda: None)
    with pytest.raises(ValueError, match="schema-10 is fresh-only"):
        util.create_setting(
            args=args,
            save_flags=True,
            xpid=typed_xpid,
            ckp=True,
            voc_gate_policy_schema_version=9,
        )
    assert opened_configs == []
    assert not savedir.exists()


@pytest.mark.parametrize(
    "typed_xpid",
    [
        _Schema10PathLike(123),
        b"\xffenduro-voc-v17-huber-common-eps25-sentinel-wire1200",
    ],
)
def test_schema10_unclassifiable_xpid_fails_before_config_io(
    monkeypatch, tmp_path, typed_xpid
):
    snapshot_root = tmp_path / "snapshot"
    savedir = snapshot_root / "runs"
    args = _replace_cli_value(_schema10_wire_cli_args(), "savedir", savedir)
    args = _replace_cli_value(
        args,
        "icopro_data_path",
        snapshot_root / "data" / "behavioral_data_block",
    )
    opened_configs = []
    real_open = open

    def tracking_open(path, *open_args, **open_kwargs):
        if str(path).endswith("config_c.yaml"):
            opened_configs.append(str(path))
        return real_open(path, *open_args, **open_kwargs)

    monkeypatch.setattr("builtins.open", tracking_open)
    monkeypatch.setattr(util, "get_git_revision_hash", lambda: None)
    with pytest.raises(ValueError, match="could not be classified"):
        util.create_setting(
            args=args,
            save_flags=True,
            xpid=typed_xpid,
            ckp=True,
            voc_gate_policy_schema_version=9,
        )
    assert opened_configs == []
    assert not savedir.exists()


def test_schema11_accepts_only_three_exact_closed_stage_tuples():
    for profile in util.VOC_GATE_POLICY_SCHEMA11_STAGE_PROFILES:
        xpid, seed, total, warmup, unroll, use_wandb = profile
        flags = _schema11_flags(
            xpid=xpid,
            base_seed=seed,
            total_steps=total,
            model_warm_up_n=warmup,
            actor_unroll_len=unroll,
            use_wandb=use_wandb,
            ckpdir=(
                "/tmp/di-voc-v18-orthocd-adam-eps25-test/runs/" + xpid
            ),
        )
        assert util.process_flags(flags).xpid == xpid
    wire, qualification, primary = util.VOC_GATE_POLICY_SCHEMA11_STAGE_PROFILES
    with pytest.raises(
        ValueError,
        match=(
            "atomic flags require voc_gate_policy_schema_version to be exact "
            "Python integer 6, 7, 8, 9, or 10"
        ),
    ):
        util.process_flags(
            _schema11_flags(
                xpid=util.VOC_GATE_POLICY_SCHEMA10_STAGE_PROFILES[0][0]
            )
        )
    for mutation in (
        {"xpid": wire[0], "base_seed": primary[1]},
        {"xpid": wire[0], "total_steps": qualification[2]},
        {"xpid": qualification[0], "use_wandb": wire[5]},
        {"xpid": primary[0], "actor_unroll_len": wire[4]},
        {"xpid": " " + wire[0]},
        {"xpid": wire[0] + " "},
    ):
        with pytest.raises(ValueError, match="unregistered schema-11 stage"):
            util.process_flags(_schema11_flags(**mutation))


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("xpid", np.str_(util.VOC_GATE_POLICY_SCHEMA11_STAGE_PROFILES[0][0])),
        ("base_seed", True),
        ("base_seed", np.int64(1)),
        ("total_steps", 1200.0),
        ("model_warm_up_n", "512"),
        ("actor_unroll_len", np.int64(41)),
        ("use_wandb", np.bool_(False)),
        ("voc_gate_policy_schema_version", np.int64(11)),
        ("voc_gate_policy_schema_version", 11.0),
        ("voc_gate_policy_schema_version", "11"),
        ("voc_gate_policy_schema_version", 10),
    ],
)
def test_schema11_stage_and_schema_members_require_exact_builtin_types(
    field, bad
):
    with pytest.raises(ValueError, match=field + "|schema-(10|11)"):
        util.process_flags(_schema11_flags(**{field: bad}))


@pytest.mark.parametrize(
    "profile", util.VOC_GATE_POLICY_SCHEMA11_STAGE_PROFILES
)
def test_schema11_actual_train_cli_is_exact229_and_identity_free(
    monkeypatch, profile
):
    args = _schema11_wire_cli_args()
    for name, value in zip(
        (
            "xpid",
            "base_seed",
            "total_steps",
            "model_warm_up_n",
            "actor_unroll_len",
            "use_wandb",
        ),
        profile,
    ):
        args = _replace_cli_value(args, name, value)
    monkeypatch.setattr(util, "get_git_revision_hash", lambda: None)
    monkeypatch.setattr(util.sys, "argv", ["train.py", *args])
    flags = util.create_setting(args=args, save_flags=False)
    evidence = util._validate_schema11_complete_surface(
        vars(flags), label="actual schema-11 train CLI"
    )
    assert evidence["key_count"] == 229
    assert evidence["v12_projection_key_count"] == 209
    assert tuple(evidence["stage"]) == profile
    for derived in util._VOC_GATE_POLICY_SCHEMA11_DERIVED_IDENTITY_KEYS:
        assert not hasattr(flags, derived)


@pytest.mark.parametrize(
    "typed_xpid",
    [
        np.str_(util.VOC_GATE_POLICY_SCHEMA11_STAGE_PROFILES[0][0]),
        _Schema10StringSubclass(
            util.VOC_GATE_POLICY_SCHEMA11_STAGE_PROFILES[0][0]
        ),
        Path(util.VOC_GATE_POLICY_SCHEMA11_STAGE_PROFILES[0][0]),
        _Schema10PathLike(util.VOC_GATE_POLICY_SCHEMA11_STAGE_PROFILES[0][0]),
        _Schema10PathLike(
            util.VOC_GATE_POLICY_SCHEMA11_STAGE_PROFILES[0][0].encode()
        ),
        util.VOC_GATE_POLICY_SCHEMA11_STAGE_PROFILES[0][0].encode(),
        np.bytes_(util.VOC_GATE_POLICY_SCHEMA11_STAGE_PROFILES[0][0].encode()),
        _Schema10Stringable(
            util.VOC_GATE_POLICY_SCHEMA11_STAGE_PROFILES[0][0]
        ),
    ],
)
def test_schema11_lexical_resume_intent_fails_before_config_io(
    monkeypatch, tmp_path, typed_xpid
):
    savedir = tmp_path / "snapshot" / "runs"
    args = _replace_cli_value(_schema11_wire_cli_args(), "savedir", savedir)
    opened = []
    real_open = open

    def tracking_open(path, *args, **kwargs):
        if str(path).endswith("config_c.yaml"):
            opened.append(str(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", tracking_open)
    monkeypatch.setattr(util, "get_git_revision_hash", lambda: None)
    with pytest.raises(ValueError, match="schema-11 is fresh-only"):
        util.create_setting(
            args=args,
            save_flags=True,
            xpid=typed_xpid,
            ckp=True,
        )
    assert opened == []
    assert not savedir.exists()


@pytest.mark.parametrize(
    "typed_xpid",
    [
        "enduro-voc-v18-orthocd-adam-eps25-unregistered",
        " " + util.VOC_GATE_POLICY_SCHEMA11_STAGE_PROFILES[0][0],
        util.VOC_GATE_POLICY_SCHEMA11_STAGE_PROFILES[0][0] + " ",
        _Schema10StringSubclass(
            util.VOC_GATE_POLICY_SCHEMA11_STAGE_PROFILES[0][0]
        ),
        np.str_(util.VOC_GATE_POLICY_SCHEMA11_STAGE_PROFILES[0][0]),
        Path(util.VOC_GATE_POLICY_SCHEMA11_STAGE_PROFILES[0][0]),
        _Schema10PathLike(util.VOC_GATE_POLICY_SCHEMA11_STAGE_PROFILES[0][0]),
        _Schema10PathLike(
            util.VOC_GATE_POLICY_SCHEMA11_STAGE_PROFILES[0][0].encode()
        ),
        util.VOC_GATE_POLICY_SCHEMA11_STAGE_PROFILES[0][0].encode(),
        np.bytes_(util.VOC_GATE_POLICY_SCHEMA11_STAGE_PROFILES[0][0].encode()),
        bytearray(
            util.VOC_GATE_POLICY_SCHEMA11_STAGE_PROFILES[0][0].encode()
        ),
        memoryview(
            util.VOC_GATE_POLICY_SCHEMA11_STAGE_PROFILES[0][0].encode()
        ),
        _Schema10Stringable(
            util.VOC_GATE_POLICY_SCHEMA11_STAGE_PROFILES[0][0]
        ),
    ],
)
def test_schema11_lexical_intent_ckp_false_requires_strict_stage_before_io(
    monkeypatch, tmp_path, typed_xpid
):
    savedir = tmp_path / "snapshot" / "runs"
    args = _replace_cli_value(_schema11_wire_cli_args(), "savedir", savedir)
    args = _replace_cli_value(
        args,
        "icopro_data_path",
        tmp_path / "snapshot" / "data" / "behavioral_data_block",
    )
    opened = []
    git_calls = []
    real_open = open

    def tracking_open(path, *open_args, **open_kwargs):
        if str(path).endswith("config_c.yaml"):
            opened.append(str(path))
        return real_open(path, *open_args, **open_kwargs)

    monkeypatch.setattr("builtins.open", tracking_open)
    monkeypatch.setattr(
        util,
        "get_git_revision_hash",
        lambda: git_calls.append("git") or None,
    )
    with pytest.raises(ValueError, match="schema-11|xpid|stage"):
        util.create_setting(
            args=args,
            save_flags=True,
            xpid=typed_xpid,
            ckp=False,
        )
    assert opened == []
    assert git_calls == []
    assert not savedir.exists()


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("missing_barrier", "version_barrier"),
        ("wrong_barrier", "version_barrier"),
        ("missing_seal", "seal_schema_version"),
        ("wrong_seal", "seal_schema_version"),
        ("missing_explicit_schema", "schema_version"),
        ("wrong_schema", "schema_version"),
        ("wrong_atomic_surface", "actor_amp_init_scale"),
    ],
)
def test_schema11_exact_xpid_requires_complete_atomic_surface_before_io(
    monkeypatch, tmp_path, mutation, expected
):
    savedir = tmp_path / "snapshot" / "runs"
    args = _replace_cli_value(_schema11_wire_cli_args(), "savedir", savedir)
    args = _replace_cli_value(
        args,
        "icopro_data_path",
        tmp_path / "snapshot" / "data" / "behavioral_data_block",
    )

    def remove_cli_value(name):
        index = args.index(f"--{name}")
        del args[index:index + 2]

    kwargs = {"ckp": False}
    if mutation == "missing_barrier":
        # The gate schema is intentionally absent too, as it is on the
        # canonical inference CLI; without the atomic barrier it must not
        # fall through to legacy processing.
        remove_cli_value("voc_actor_policy_version_barrier")
    elif mutation == "wrong_barrier":
        kwargs["voc_actor_policy_version_barrier"] = np.bool_(True)
    elif mutation == "missing_seal":
        remove_cli_value("voc_model_input_seal_schema_version")
    elif mutation == "wrong_seal":
        kwargs["voc_model_input_seal_schema_version"] = np.int64(1)
    elif mutation in {"missing_explicit_schema", "wrong_schema"}:
        config_path = tmp_path / "schema11-intent.yaml"
        config_path.write_text(
            "voc_gate_policy_schema_version: "
            + ("null\n" if mutation == "missing_explicit_schema" else "10\n"),
            encoding="utf-8",
        )
        args.extend(["--config", str(config_path)])
    else:
        kwargs["actor_amp_init_scale"] = 64.0

    opened = []
    git_calls = []
    real_open = open

    def tracking_open(path, *open_args, **open_kwargs):
        if str(path).endswith("config_c.yaml"):
            opened.append(str(path))
        return real_open(path, *open_args, **open_kwargs)

    monkeypatch.setattr("builtins.open", tracking_open)
    monkeypatch.setattr(
        util,
        "get_git_revision_hash",
        lambda: git_calls.append("git") or None,
    )
    with pytest.raises(ValueError, match=expected):
        util.create_setting(args=args, save_flags=True, **kwargs)
    assert opened == []
    assert git_calls == []
    assert not savedir.exists()


def test_schema11_protocol_mapping_preserves_schema10_return_shape():
    schema10_flags = util.process_flags(_schema10_flags())
    schema11_flags = util.process_flags(_schema11_flags())
    schema10 = util.validate_voc_gate_policy_schema({
        "voc_gate_policy_schema_version": 10,
        "flags": vars(schema10_flags),
    })
    schema11 = util.validate_voc_gate_policy_schema({
        "voc_gate_policy_schema_version": 11,
        "flags": vars(schema11_flags),
    })
    assert tuple(schema11) == tuple(schema10)
    for derived in util._VOC_GATE_POLICY_SCHEMA11_DERIVED_IDENTITY_KEYS:
        assert derived not in schema11


def test_schema12_accepts_only_three_exact_closed_stage_tuples_and_tau1():
    for profile in util.VOC_GATE_POLICY_SCHEMA12_STAGE_PROFILES:
        xpid, seed, total, warmup, unroll, use_wandb = profile
        flags = _schema12_flags(
            xpid=xpid,
            base_seed=seed,
            total_steps=total,
            model_warm_up_n=warmup,
            actor_unroll_len=unroll,
            use_wandb=use_wandb,
            ckpdir=(
                "/tmp/di-voc-v19-tau1-orthocd-adam-eps25-test/runs/" + xpid
            ),
        )
        resolved = util.process_flags(flags)
        assert resolved.xpid == xpid
        assert type(resolved.voc_gate_target_tau) is float
        assert resolved.voc_gate_target_tau == 1.0
    wire, qualification, primary = util.VOC_GATE_POLICY_SCHEMA12_STAGE_PROFILES
    for mutation in (
        {"xpid": wire[0], "base_seed": primary[1]},
        {"xpid": qualification[0], "use_wandb": wire[5]},
        {"xpid": primary[0], "actor_unroll_len": wire[4]},
        {"xpid": " " + wire[0]},
        {"xpid": wire[0] + " "},
    ):
        with pytest.raises(ValueError, match="schema-12|stage"):
            util.process_flags(_schema12_flags(**mutation))


@pytest.mark.parametrize(
    "bad_tau",
    [0.1, 1, True, np.float64(1.0), "1.0", 1.0 - 2.0 ** -24, 1.0000001],
)
def test_schema12_tau_requires_exact_builtin_float_one(bad_tau):
    with pytest.raises(ValueError, match="voc_gate_target_tau|schema-12"):
        util.process_flags(_schema12_flags(voc_gate_target_tau=bad_tau))


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("xpid", np.str_(util.VOC_GATE_POLICY_SCHEMA12_STAGE_PROFILES[0][0])),
        ("base_seed", True),
        ("total_steps", 1200.0),
        ("model_warm_up_n", np.int64(512)),
        ("actor_unroll_len", "41"),
        ("use_wandb", np.bool_(False)),
        ("voc_gate_policy_schema_version", np.int64(12)),
        ("voc_gate_policy_schema_version", 12.0),
        ("voc_gate_policy_schema_version", "12"),
        ("voc_gate_policy_schema_version", 11),
    ],
)
def test_schema12_stage_and_schema_members_require_exact_builtin_types(field, bad):
    with pytest.raises(ValueError, match=field + "|schema-(11|12)"):
        util.process_flags(_schema12_flags(**{field: bad}))


@pytest.mark.parametrize("profile", util.VOC_GATE_POLICY_SCHEMA12_STAGE_PROFILES)
def test_schema12_actual_train_cli_is_exact229_tau1_and_identity_free(
    monkeypatch, profile
):
    args = _schema12_wire_cli_args()
    for name, value in zip(
        (
            "xpid",
            "base_seed",
            "total_steps",
            "model_warm_up_n",
            "actor_unroll_len",
            "use_wandb",
        ),
        profile,
    ):
        args = _replace_cli_value(args, name, value)
    monkeypatch.setattr(util, "get_git_revision_hash", lambda: None)
    monkeypatch.setattr(util.sys, "argv", ["train.py", *args])
    flags = util.create_setting(args=args, save_flags=False)
    evidence = util._validate_schema12_complete_surface(
        vars(flags), label="actual schema-12 train CLI"
    )
    assert len(vars(flags)) == 229
    assert evidence["key_count"] == 229
    assert evidence["v12_projection_key_count"] == 209
    assert evidence["v12_projection_sha256"] == (
        util.VOC_GATE_POLICY_SCHEMA12_V12_PROJECTION_SHA256
    )
    assert tuple(evidence["stage"]) == profile
    assert type(flags.voc_gate_target_tau) is float
    assert flags.voc_gate_target_tau == 1.0
    for derived in util._VOC_GATE_POLICY_SCHEMA12_DERIVED_IDENTITY_KEYS:
        assert not hasattr(flags, derived)


@pytest.mark.parametrize(
    "typed_xpid",
    [
        np.str_(util.VOC_GATE_POLICY_SCHEMA12_STAGE_PROFILES[0][0]),
        _Schema10StringSubclass(util.VOC_GATE_POLICY_SCHEMA12_STAGE_PROFILES[0][0]),
        Path(util.VOC_GATE_POLICY_SCHEMA12_STAGE_PROFILES[0][0]),
        _Schema10PathLike(util.VOC_GATE_POLICY_SCHEMA12_STAGE_PROFILES[0][0]),
        util.VOC_GATE_POLICY_SCHEMA12_STAGE_PROFILES[0][0].encode(),
        np.bytes_(util.VOC_GATE_POLICY_SCHEMA12_STAGE_PROFILES[0][0].encode()),
        _Schema10Stringable(util.VOC_GATE_POLICY_SCHEMA12_STAGE_PROFILES[0][0]),
    ],
)
def test_schema12_lexical_resume_intent_fails_before_config_io(
    monkeypatch, tmp_path, typed_xpid
):
    savedir = tmp_path / "snapshot" / "runs"
    args = _replace_cli_value(_schema12_wire_cli_args(), "savedir", savedir)
    opened = []
    real_open = open

    def tracking_open(path, *open_args, **open_kwargs):
        if str(path).endswith("config_c.yaml"):
            opened.append(str(path))
        return real_open(path, *open_args, **open_kwargs)

    monkeypatch.setattr("builtins.open", tracking_open)
    monkeypatch.setattr(util, "get_git_revision_hash", lambda: pytest.fail("git I/O"))
    with pytest.raises(ValueError, match="schema-12 is fresh-only"):
        util.create_setting(
            args=args, save_flags=True, xpid=typed_xpid, ckp=True
        )
    assert opened == []
    assert not savedir.exists()


@pytest.mark.parametrize(
    "typed_xpid",
    [
        np.str_(util.VOC_GATE_POLICY_SCHEMA12_STAGE_PROFILES[0][0]),
        _Schema10StringSubclass(
            util.VOC_GATE_POLICY_SCHEMA12_STAGE_PROFILES[0][0]
        ),
        Path(util.VOC_GATE_POLICY_SCHEMA12_STAGE_PROFILES[0][0]),
        _Schema10PathLike(util.VOC_GATE_POLICY_SCHEMA12_STAGE_PROFILES[0][0]),
        util.VOC_GATE_POLICY_SCHEMA12_STAGE_PROFILES[0][0].encode(),
        np.bytes_(
            util.VOC_GATE_POLICY_SCHEMA12_STAGE_PROFILES[0][0].encode()
        ),
        _Schema10Stringable(
            util.VOC_GATE_POLICY_SCHEMA12_STAGE_PROFILES[0][0]
        ),
    ],
)
def test_schema12_lexical_fresh_intent_rejects_nonbuiltin_xpid_before_io(
    monkeypatch, tmp_path, typed_xpid
):
    savedir = tmp_path / "snapshot" / "runs"
    args = _replace_cli_value(_schema12_wire_cli_args(), "savedir", savedir)
    args = _replace_cli_value(
        args,
        "icopro_data_path",
        tmp_path / "snapshot" / "data" / "behavioral_data_block",
    )
    opened = []
    real_open = open

    def tracking_open(path, *open_args, **open_kwargs):
        if str(path).endswith("config_c.yaml"):
            opened.append(str(path))
        return real_open(path, *open_args, **open_kwargs)

    monkeypatch.setattr("builtins.open", tracking_open)
    monkeypatch.setattr(util, "get_git_revision_hash", lambda: pytest.fail("git I/O"))
    with pytest.raises(ValueError, match="schema-12.*xpid"):
        util.create_setting(
            args=args, save_flags=True, xpid=typed_xpid, ckp=False
        )
    assert opened == []
    assert not savedir.exists()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("voc_actor_policy_version_barrier", False),
        ("voc_actor_policy_version_barrier", np.bool_(True)),
        ("voc_model_input_seal_schema_version", 0),
        ("voc_model_input_seal_schema_version", np.int64(1)),
        ("voc_gate_policy_schema_version", 11),
        ("voc_gate_policy_schema_version", np.int64(12)),
        ("voc_gate_target_tau", 0.1),
        ("voc_gate_target_tau", 1),
    ],
)
def test_schema12_exact_xpid_rejects_malformed_atomic_surface_before_io(
    monkeypatch, tmp_path, name, value
):
    savedir = tmp_path / "snapshot" / "runs"
    args = _replace_cli_value(_schema12_wire_cli_args(), "savedir", savedir)
    args = _replace_cli_value(
        args,
        "icopro_data_path",
        tmp_path / "snapshot" / "data" / "behavioral_data_block",
    )
    opened = []
    real_open = open

    def tracking_open(path, *open_args, **open_kwargs):
        if str(path).endswith("config_c.yaml"):
            opened.append(str(path))
        return real_open(path, *open_args, **open_kwargs)

    monkeypatch.setattr("builtins.open", tracking_open)
    monkeypatch.setattr(util, "get_git_revision_hash", lambda: pytest.fail("git I/O"))
    with pytest.raises(ValueError, match="schema-12|" + name):
        util.create_setting(args=args, save_flags=True, **{name: value})
    assert opened == []
    assert not savedir.exists()


def test_schema12_exact_xpid_rejects_cross_schema_config_before_io(
    monkeypatch, tmp_path
):
    savedir = tmp_path / "snapshot" / "runs"
    args = _replace_cli_value(_schema12_wire_cli_args(), "savedir", savedir)
    args = _replace_cli_value(
        args,
        "icopro_data_path",
        tmp_path / "snapshot" / "data" / "behavioral_data_block",
    )
    config_path = tmp_path / "cross-schema.yaml"
    config_path.write_text(
        "voc_gate_policy_schema_version: 11\n", encoding="utf-8"
    )
    args.extend(["--config", str(config_path)])
    monkeypatch.setattr(util, "get_git_revision_hash", lambda: pytest.fail("git I/O"))
    with pytest.raises(ValueError, match="schema-12.*schema_version"):
        util.create_setting(args=args, save_flags=True)
    assert not savedir.exists()


@pytest.mark.parametrize(
    "missing_name",
    ["voc_actor_policy_version_barrier", "voc_model_input_seal_schema_version"],
)
def test_schema12_exact_xpid_rejects_missing_atomic_field_before_io(
    monkeypatch, tmp_path, missing_name
):
    savedir = tmp_path / "snapshot" / "runs"
    args = _replace_cli_value(_schema12_wire_cli_args(), "savedir", savedir)
    index = args.index(f"--{missing_name}")
    del args[index:index + 2]
    monkeypatch.setattr(util, "get_git_revision_hash", lambda: pytest.fail("git I/O"))
    with pytest.raises(ValueError, match="schema-12|" + missing_name):
        util.create_setting(args=args, save_flags=True)
    assert not savedir.exists()


@pytest.mark.parametrize("tau", [0.25, 1, np.float64(0.75)])
def test_legacy_schema1_through5_nondefault_tau_normalization_is_unchanged(
    tau
):
    flags = _flags(voc_gate_target_tau=tau)
    processed = util.process_flags(flags)
    assert processed.voc_gate_target_tau == float(tau)
    assert type(processed.voc_gate_target_tau) is float


def test_schema12_protocol_mapping_preserves_schema11_return_shape():
    schema11_flags = util.process_flags(_schema11_flags())
    schema12_flags = util.process_flags(_schema12_flags())
    schema11 = util.validate_voc_gate_policy_schema({
        "voc_gate_policy_schema_version": 11,
        "flags": vars(schema11_flags),
    })
    schema12 = util.validate_voc_gate_policy_schema({
        "voc_gate_policy_schema_version": 12,
        "flags": vars(schema12_flags),
    })
    assert tuple(schema12) == tuple(schema11)
    for derived in util._VOC_GATE_POLICY_SCHEMA12_DERIVED_IDENTITY_KEYS:
        assert derived not in schema12


def test_schema13_accepts_only_three_exact_closed_stage_tuples_and_tau1():
    for profile in util.VOC_GATE_POLICY_SCHEMA13_STAGE_PROFILES:
        xpid, seed, total, warmup, unroll, use_wandb = profile
        flags = util.process_flags(_schema13_flags(
            xpid=xpid,
            base_seed=seed,
            total_steps=total,
            model_warm_up_n=warmup,
            actor_unroll_len=unroll,
            use_wandb=use_wandb,
            ckpdir=(
                "/tmp/di-voc-v20-telemetry-tau1-orthocd-adam-eps25-test/"
                "runs/" + xpid
            ),
        ))
        assert util._validate_schema13_stage_profile(
            vars(flags), label="schema-13 exact stage"
        ) == profile
        assert type(flags.voc_gate_policy_schema_version) is int
        assert flags.voc_gate_policy_schema_version == 13
        assert type(flags.voc_gate_target_tau) is float
        assert flags.voc_gate_target_tau == 1.0

    wire, qualification, primary = util.VOC_GATE_POLICY_SCHEMA13_STAGE_PROFILES
    for mutation in (
        {"xpid": wire[0], "base_seed": primary[1]},
        {"xpid": qualification[0], "use_wandb": wire[5]},
        {"xpid": primary[0], "actor_unroll_len": wire[4]},
        {"xpid": " " + wire[0]},
        {"xpid": wire[0] + " "},
    ):
        with pytest.raises(ValueError, match="schema-13|stage"):
            util.process_flags(_schema13_flags(**mutation))


@pytest.mark.parametrize(
    "bad_tau",
    [0.1, 1, True, np.float64(1.0), "1.0", 1.0 - 2.0 ** -24, 1.0000001],
)
def test_schema13_tau_requires_exact_builtin_float_one(bad_tau):
    with pytest.raises(ValueError, match="voc_gate_target_tau|schema-13"):
        util.process_flags(_schema13_flags(voc_gate_target_tau=bad_tau))


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("xpid", np.str_(util.VOC_GATE_POLICY_SCHEMA13_STAGE_PROFILES[0][0])),
        ("base_seed", True),
        ("total_steps", 1200.0),
        ("model_warm_up_n", np.int64(512)),
        ("actor_unroll_len", "41"),
        ("use_wandb", np.bool_(False)),
        ("voc_gate_policy_schema_version", np.int64(13)),
        ("voc_gate_policy_schema_version", 13.0),
        ("voc_gate_policy_schema_version", "13"),
        ("voc_gate_policy_schema_version", 12),
    ],
)
def test_schema13_stage_and_schema_members_require_exact_builtin_types(field, bad):
    with pytest.raises(ValueError, match=field + "|schema-(12|13)"):
        util.process_flags(_schema13_flags(**{field: bad}))


@pytest.mark.parametrize("profile", util.VOC_GATE_POLICY_SCHEMA13_STAGE_PROFILES)
def test_schema13_actual_train_cli_is_exact229_tau1_and_identity_free(
    monkeypatch, profile
):
    args = _schema13_wire_cli_args()
    assert "--voc_gate_policy_schema_version" not in args
    assert args.count("--voc_gate_target_tau") == 1
    for name, value in zip(
        (
            "xpid",
            "base_seed",
            "total_steps",
            "model_warm_up_n",
            "actor_unroll_len",
            "use_wandb",
        ),
        profile,
    ):
        args = _replace_cli_value(args, name, value)
    monkeypatch.setattr(util, "get_git_revision_hash", lambda: None)
    monkeypatch.setattr(util.sys, "argv", ["train.py", *args])
    flags = util.create_setting(args=args, save_flags=False)
    evidence = util._validate_schema13_complete_surface(
        vars(flags), label="actual schema-13 train CLI"
    )
    assert len(vars(flags)) == 229
    assert evidence["key_count"] == 229
    assert evidence["v12_projection_key_count"] == 209
    assert evidence["v12_projection_sha256"] == (
        util.VOC_GATE_POLICY_SCHEMA12_V12_PROJECTION_SHA256
    )
    assert "voc_gate_policy_schema_version" not in (
        util.VOC_GATE_POLICY_SCHEMA12_V12_PROJECTION
    )
    assert tuple(evidence["stage"]) == profile
    assert type(flags.voc_gate_policy_schema_version) is int
    assert flags.voc_gate_policy_schema_version == 13
    assert type(flags.voc_gate_target_tau) is float
    assert flags.voc_gate_target_tau == 1.0
    for derived in util._VOC_GATE_POLICY_SCHEMA13_DERIVED_IDENTITY_KEYS:
        assert not hasattr(flags, derived)


@pytest.mark.parametrize(
    "mutation",
    ("duplicate", "extra", "missing", "reorder", "equals", "abbreviation"),
)
def test_schema13_cli_rejects_any_drift_from_exact_96_pairs_before_io(
    monkeypatch, mutation
):
    args = _schema13_wire_cli_args()
    if mutation == "duplicate":
        args.extend(["--base_seed", "1"])
    elif mutation == "extra":
        args.extend(["--voc_gate_policy_schema_version", "13"])
    elif mutation == "missing":
        del args[-2:]
    elif mutation == "reorder":
        args[0:4] = args[2:4] + args[0:2]
    elif mutation == "equals":
        index = args.index("--xpid")
        args[index:index + 2] = [f"--xpid={args[index + 1]}"]
    else:
        args[args.index("--xpid")] = "--xpi"
    monkeypatch.setattr(
        util, "get_git_revision_hash", lambda: pytest.fail("git I/O")
    )
    with pytest.raises(ValueError, match="schema-13.*(192|96-pair)"):
        util.create_setting(args=args, save_flags=False)


@pytest.mark.parametrize(
    ("name", "value"),
    (("base_seed", "01"), ("ckp", "garbage"), ("voc_gate_target_tau", "1.00")),
)
def test_schema13_cli_rejects_parser_normalized_value_spellings(
    monkeypatch, name, value
):
    args = _replace_cli_value(_schema13_wire_cli_args(), name, value)
    monkeypatch.setattr(
        util, "get_git_revision_hash", lambda: pytest.fail("git I/O")
    )
    with pytest.raises(ValueError, match="schema-13 CLI values"):
        util.create_setting(args=args, save_flags=False)


def test_schema13_cli_rejects_user_config_indirection_before_io(
    monkeypatch, tmp_path
):
    parser = util.add_parse(["default_thinker.yaml", "default_actor.yaml"])
    parsed = vars(parser.parse_args(_schema13_wire_cli_args()))
    config_path = tmp_path / "schema13.yaml"
    config_path.write_text(yaml.safe_dump(parsed), encoding="utf-8")
    monkeypatch.setattr(
        util, "get_git_revision_hash", lambda: pytest.fail("git I/O")
    )
    with pytest.raises(ValueError, match="schema-13.*user-config"):
        util.create_setting(
            args=["--config", str(config_path)], save_flags=False
        )


@pytest.mark.parametrize(
    "typed_xpid",
    [
        np.str_(util.VOC_GATE_POLICY_SCHEMA13_STAGE_PROFILES[0][0]),
        _Schema10StringSubclass(util.VOC_GATE_POLICY_SCHEMA13_STAGE_PROFILES[0][0]),
        Path(util.VOC_GATE_POLICY_SCHEMA13_STAGE_PROFILES[0][0]),
        _Schema10PathLike(util.VOC_GATE_POLICY_SCHEMA13_STAGE_PROFILES[0][0]),
        util.VOC_GATE_POLICY_SCHEMA13_STAGE_PROFILES[0][0].encode(),
        np.bytes_(util.VOC_GATE_POLICY_SCHEMA13_STAGE_PROFILES[0][0].encode()),
        _Schema10Stringable(util.VOC_GATE_POLICY_SCHEMA13_STAGE_PROFILES[0][0]),
    ],
)
def test_schema13_lexical_intent_rejects_nonbuiltin_xpid_before_io(
    monkeypatch, tmp_path, typed_xpid
):
    savedir = tmp_path / "snapshot" / "runs"
    args = _replace_cli_value(_schema13_wire_cli_args(), "savedir", savedir)
    args = _replace_cli_value(
        args,
        "icopro_data_path",
        tmp_path / "snapshot" / "data" / "behavioral_data_block",
    )
    monkeypatch.setattr(util, "get_git_revision_hash", lambda: pytest.fail("git I/O"))
    with pytest.raises(ValueError, match="schema-13.*xpid"):
        util.create_setting(
            args=args, save_flags=True, xpid=typed_xpid, ckp=False
        )
    assert not savedir.exists()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("ckp", True),
        ("voc_actor_policy_version_barrier", False),
        ("voc_actor_policy_version_barrier", np.bool_(True)),
        ("voc_model_input_seal_schema_version", 0),
        ("voc_model_input_seal_schema_version", np.int64(1)),
        ("voc_gate_policy_schema_version", 12),
        ("voc_gate_policy_schema_version", np.int64(13)),
        ("voc_gate_target_tau", 0.1),
        ("voc_gate_target_tau", 1),
    ],
)
def test_schema13_exact_xpid_rejects_malformed_surface_before_io(
    monkeypatch, tmp_path, name, value
):
    savedir = tmp_path / "snapshot" / "runs"
    args = _replace_cli_value(_schema13_wire_cli_args(), "savedir", savedir)
    args = _replace_cli_value(
        args,
        "icopro_data_path",
        tmp_path / "snapshot" / "data" / "behavioral_data_block",
    )
    monkeypatch.setattr(util, "get_git_revision_hash", lambda: pytest.fail("git I/O"))
    with pytest.raises(ValueError, match="schema-13|" + name):
        util.create_setting(args=args, save_flags=True, **{name: value})
    assert not savedir.exists()


@pytest.mark.parametrize(
    "missing_name",
    ["voc_actor_policy_version_barrier", "voc_model_input_seal_schema_version"],
)
def test_schema13_exact_xpid_rejects_missing_atomic_field_before_io(
    monkeypatch, tmp_path, missing_name
):
    savedir = tmp_path / "snapshot" / "runs"
    args = _replace_cli_value(_schema13_wire_cli_args(), "savedir", savedir)
    index = args.index(f"--{missing_name}")
    del args[index:index + 2]
    monkeypatch.setattr(util, "get_git_revision_hash", lambda: pytest.fail("git I/O"))
    with pytest.raises(ValueError, match="schema-13|" + missing_name):
        util.create_setting(args=args, save_flags=True)
    assert not savedir.exists()


def test_schema13_protocol_mapping_preserves_schema12_return_shape():
    schema12_flags = util.process_flags(_schema12_flags())
    schema13_flags = util.process_flags(_schema13_flags())
    schema12 = util.validate_voc_gate_policy_schema({
        "voc_gate_policy_schema_version": 12,
        "flags": vars(schema12_flags),
    })
    schema13 = util.validate_voc_gate_policy_schema({
        "voc_gate_policy_schema_version": 13,
        "flags": vars(schema13_flags),
    })
    assert tuple(schema13) == tuple(schema12)
    for derived in util._VOC_GATE_POLICY_SCHEMA13_DERIVED_IDENTITY_KEYS:
        assert derived not in schema13


def test_schema13_no_keyword_completion_uses_full_validator_without_recursion(
    monkeypatch
):
    evidence = {
        "checkpoint_files": {"sentinel": {"sha256": "a" * 64, "size": 1}},
        "implementation_sources": {},
        "loaded_extensions": {},
    }
    calls = []
    monkeypatch.setattr(util, "_completion_claims_schema13", lambda root: True)
    monkeypatch.setattr(
        util,
        "validate_schema13_final_bundle",
        lambda root, *, label: calls.append((root, label))
        or {"completion_evidence": evidence},
    )
    monkeypatch.setattr(
        util,
        "_collect_run_completion_evidence_raw",
        lambda *args, **kwargs: pytest.fail("raw fallback"),
    )
    first = util.collect_run_completion_evidence("/tmp/schema13-run")
    second = util.collect_run_completion_evidence("/tmp/schema13-run")
    assert first == evidence == second
    assert first is not evidence and first is not second
    assert len(calls) == 2


def test_schema13_inherited_logger_binds_full_evidence_before_and_after_upload(
    monkeypatch
):
    from thinker.logger import SLogWorker

    checkpoint_files = {
        name: {"sha256": str(index + 1) * 64, "size": index + 1}
        for index, name in enumerate(util._SCHEMA13_COMPLETION_CHECKPOINT_FILES)
    }
    calls = []

    def full_evidence(root):
        calls.append(root)
        return {
            "checkpoint_files": checkpoint_files,
            "implementation_sources": {},
            "loaded_extensions": {},
        }

    monkeypatch.setattr(util, "collect_run_completion_evidence", full_evidence)
    worker = object.__new__(SLogWorker)
    worker.ckpdir = "/tmp/" + util.VOC_GATE_POLICY_SCHEMA13_STAGE_PROFILES[0][0]
    request = {"checkpoint_files": copy.deepcopy(checkpoint_files)}
    assert worker._validate_schema6_request_checkpoint_files(request) is True
    assert worker._validate_schema6_request_checkpoint_files(request) is True
    assert calls == [worker.ckpdir, worker.ckpdir]
    request["checkpoint_files"]["ckp_actor.tar"]["size"] += 1
    with pytest.raises(RuntimeError, match="changed around final upload"):
        worker._validate_schema6_request_checkpoint_files(request)


def test_schema13_explicit_completion_collection_stays_raw(monkeypatch):
    sentinel = {"checkpoint_files": {}, "implementation_sources": {}}
    monkeypatch.setattr(
        util,
        "validate_schema13_final_bundle",
        lambda *args, **kwargs: pytest.fail("recursive full validation"),
    )
    monkeypatch.setattr(
        util,
        "_collect_run_completion_evidence_raw",
        lambda root, *, gate_schema: (root, gate_schema, sentinel),
    )
    assert util.collect_run_completion_evidence(
        "/tmp/schema13-run",
        gate_schema=util.VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION,
    ) == (
        "/tmp/schema13-run",
        util.VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION,
        sentinel,
    )


def test_completion_bound_reader_rejects_links_and_path_replacement(
    monkeypatch, tmp_path
):
    original = tmp_path / "original"
    replacement = tmp_path / "replacement"
    original.write_bytes(b"original-bytes")
    replacement.write_bytes(b"replacement-bytes")

    hardlink = tmp_path / "hardlink"
    os.link(original, hardlink)
    expected_digest = util.hashlib.sha256(b"original-bytes").hexdigest()
    assert util._file_sha256(original) == expected_digest
    assert util._file_sha256(hardlink) == expected_digest
    with pytest.raises(ValueError, match="single-link regular"):
        util._stable_regular_file_bytes(original, label="hardlinked")
    hardlink.unlink()

    symlink = tmp_path / "symlink"
    symlink.symlink_to(original)
    assert util._file_sha256(symlink) == expected_digest
    with pytest.raises(OSError):
        util._stable_regular_file_bytes(symlink, label="symlinked")

    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="single-link regular"):
        util._stable_regular_file_bytes(fifo, label="fifo")

    real_fstat = util.os.fstat
    calls = 0

    def replace_after_read(fd):
        nonlocal calls
        result = real_fstat(fd)
        calls += 1
        if calls == 2:
            os.replace(replacement, original)
        return result

    monkeypatch.setattr(util.os, "fstat", replace_after_read)
    with pytest.raises(RuntimeError, match="pathname changed"):
        util._stable_regular_file_bytes(original, label="replaced")


def test_schema13_completion_rejects_fifo_config_without_blocking(
    monkeypatch, tmp_path
):
    root = tmp_path / util.VOC_GATE_POLICY_SCHEMA13_STAGE_PROFILES[0][0]
    root.mkdir()
    os.mkfifo(root / "config_c.yaml")
    real_open = util.os.open

    def require_nonblocking(path, flags, *args, **kwargs):
        assert flags & os.O_NONBLOCK
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(util.os, "open", require_nonblocking)
    with pytest.raises(ValueError, match="single-link regular"):
        util.validate_schema13_final_bundle(root)


@pytest.mark.parametrize(
    "bad_ckp",
    [True, 1, 0, np.bool_(True), np.bool_(False), "True", "False"],
)
def test_schema8_create_flags_rejects_nonexact_false_ckp_before_io(
    monkeypatch, tmp_path, bad_ckp
):
    snapshot_root = tmp_path / "snapshot"
    savedir = snapshot_root / "runs"
    xpid = util.VOC_GATE_POLICY_SCHEMA8_STAGE_PROFILES[0][0]
    raw = vars(_schema8_flags()).copy()
    for derived in (
        "__version__",
        "git_revision",
        "cmd",
        "ckpdir",
        "voc_gate_policy_schema_version",
        "voc_actor_policy_barrier_runtime",
    ):
        raw.pop(derived)
    raw.update({
        "savedir": str(savedir),
        "icopro_data_path": str(
            snapshot_root / "data" / "behavioral_data_block"
        ),
        "ckp": bad_ckp,
    })
    opened_configs = []
    real_open = open

    def tracking_open(path, *args, **kwargs):
        if str(path).endswith("config_c.yaml"):
            opened_configs.append(str(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", tracking_open)
    monkeypatch.setattr(util, "get_git_revision_hash", lambda: None)
    with pytest.raises(ValueError, match="exact Python bool False"):
        util.create_flags(
            ["default_thinker.yaml", "default_actor.yaml"],
            save_flags=True,
            post_fn=util.process_flags_actor,
            **raw,
        )
    assert opened_configs == []
    assert not (savedir / xpid).exists()


@pytest.mark.parametrize("entrypoint", ["create_flags", "create_setting"])
def test_schema8_missing_ckp_rejects_before_io_or_run_directory(
    monkeypatch, tmp_path, entrypoint
):
    snapshot_root = tmp_path / "snapshot"
    savedir = snapshot_root / "runs"
    xpid = util.VOC_GATE_POLICY_SCHEMA8_STAGE_PROFILES[0][0]
    real_safe_load = util.yaml.safe_load

    def safe_load_without_ckp(stream):
        loaded = real_safe_load(stream)
        if isinstance(loaded, dict):
            loaded.pop("ckp", None)
        return loaded

    opened_configs = []
    real_open = open

    def tracking_open(path, *args, **kwargs):
        if str(path).endswith("config_c.yaml"):
            opened_configs.append(str(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(util.yaml, "safe_load", safe_load_without_ckp)
    monkeypatch.setattr("builtins.open", tracking_open)
    monkeypatch.setattr(util, "get_git_revision_hash", lambda: None)
    if entrypoint == "create_flags":
        raw = vars(_schema8_flags()).copy()
        for derived in (
            "__version__",
            "git_revision",
            "cmd",
            "ckpdir",
            "voc_gate_policy_schema_version",
            "voc_actor_policy_barrier_runtime",
            "ckp",
        ):
            raw.pop(derived)
        raw.update({
            "savedir": str(savedir),
            "icopro_data_path": str(
                snapshot_root / "data" / "behavioral_data_block"
            ),
        })
        invoke = lambda: util.create_flags(
            ["default_thinker.yaml", "default_actor.yaml"],
            save_flags=True,
            post_fn=util.process_flags_actor,
            **raw,
        )
    else:
        args = _schema8_wire_cli_args()
        ckp_index = args.index("--ckp")
        del args[ckp_index:ckp_index + 2]
        args = _replace_cli_value(args, "savedir", savedir)
        args = _replace_cli_value(
            args,
            "icopro_data_path",
            snapshot_root / "data" / "behavioral_data_block",
        )
        invoke = lambda: util.create_setting(args=args, save_flags=True)
    with pytest.raises(ValueError, match="exact Python bool False"):
        invoke()
    assert opened_configs == []
    assert not (savedir / xpid).exists()


@pytest.mark.parametrize(
    "profile", util.VOC_GATE_POLICY_SCHEMA6_STAGE_PROFILES
)
def test_schema6_actual_train_cli_canonicalizes_only_omitted_derived_fields(
    monkeypatch, profile
):
    args = _schema6_failed_wire_cli_args()
    assert "--has_action_seq" not in args
    assert "--return_h" not in args
    assert "--return_x" not in args

    xpid, seed, total, warmup, unroll, use_wandb = profile
    for name, value in (
        ("xpid", xpid),
        ("base_seed", seed),
        ("total_steps", total),
        ("model_warm_up_n", warmup),
        ("actor_unroll_len", unroll),
        ("use_wandb", use_wandb),
    ):
        args = _replace_cli_value(args, name, value)

    parsed = util.add_parse(
        ["default_thinker.yaml", "default_actor.yaml"]
    ).parse_args(args)
    assert parsed.has_action_seq is None
    assert parsed.return_h is None
    assert parsed.return_x is None
    assert parsed.model_float16 == "False"

    def unexpected_write(*_args, **_kwargs):
        raise AssertionError("save_flags=False attempted a filesystem write")

    monkeypatch.setattr(util, "get_git_revision_hash", lambda: None)
    monkeypatch.setattr(
        util, "create_schema6_fresh_run_directory", unexpected_write
    )
    monkeypatch.setattr(util.os, "makedirs", unexpected_write)
    monkeypatch.setattr(util.sys, "argv", ["train.py", *args])
    flags = util.create_setting(args=args, save_flags=False)

    assert flags.has_action_seq is False
    assert flags.return_h is True
    assert flags.return_x is True
    assert flags.model_float16 is False
    assert flags.voc_gate_policy_schema_version == 6
    assert flags.voc_actor_policy_barrier_runtime is True
    assert (
        flags.xpid,
        flags.base_seed,
        flags.total_steps,
        flags.model_warm_up_n,
        flags.actor_unroll_len,
        flags.use_wandb,
    ) == profile
    evidence = util._validate_schema6_complete_surface(
        vars(flags), label="actual schema-6 train CLI"
    )
    assert evidence["key_count"] == 228
    assert evidence["v12_projection_key_count"] == 209
    assert evidence["v12_projection_sha256"] == (
        util.VOC_GATE_POLICY_SCHEMA6_V12_BASELINE_SHA256
    )


@pytest.mark.parametrize(
    ("explicit_args", "match"),
    [
        (("--has_action_seq", "True"), "has_action_seq"),
        (("--return_h", "False"), "return_h"),
        (("--return_x", "False"), "return_x"),
    ],
)
def test_schema6_actual_train_cli_rejects_explicit_derived_field_drift(
    monkeypatch, explicit_args, match
):
    args = [*_schema6_failed_wire_cli_args(), *explicit_args]
    monkeypatch.setattr(util, "get_git_revision_hash", lambda: None)
    with pytest.raises(ValueError, match=match):
        util.create_setting(args=args, save_flags=False)


@pytest.mark.parametrize("setting", ["True", "inherit", "sometimes"])
def test_schema6_actual_train_cli_rejects_nonfalse_model_precision(
    monkeypatch, setting
):
    args = _replace_cli_value(
        _schema6_failed_wire_cli_args(), "model_float16", setting
    )
    monkeypatch.setattr(util, "get_git_revision_hash", lambda: None)
    with pytest.raises(ValueError, match="model_float16"):
        util.create_setting(args=args, save_flags=False)


def test_legacy_create_setting_keeps_derived_normalization(monkeypatch):
    monkeypatch.setattr(util, "get_git_revision_hash", lambda: None)
    flags = util.create_setting(
        args=["--dynamic_search", "True"], save_flags=False
    )
    assert flags.voc_actor_policy_version_barrier is False
    assert flags.has_action_seq is False
    assert flags.return_h is flags.see_h
    assert flags.return_x is flags.see_x
    assert flags.model_float16 is flags.float16


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("voc_loss_cost", 0.5),
        ("voc_loss_cost", np.nextafter(1.0, 2.0)),
        ("voc_gate_grad_norm_clipping", 0.5),
        ("voc_gate_grad_norm_clipping", np.nextafter(1.0, 2.0)),
        ("voc_gate_confidence_weighted", True),
        ("voc_gate_adam_beta1", 0.9),
        ("voc_gate_learning_rate", 0.0003),
        ("voc_train_epsilon", 0.3),
        ("model_state_range_loss_cost", 0.0),
        ("model_state_range_loss_cost", np.nextafter(1.0, 2.0)),
        ("model_state_projection", "none"),
        ("actor_use_rms", True),
        ("actor_learning_rate", 0.0001),
        ("actor_adam_eps", np.nextafter(1e-8, 1.0)),
        ("model_learning_rate", 0.0001),
        ("name", "Pong-v5"),
        ("icopro_game_id", 1),
        ("float16", False),
        ("model_float16", True),
        ("model_float16", np.bool_(False)),
        ("dual_net", False),
        ("model_optimizer", "sgd"),
        ("schedule_total_steps", 300_000),
    ],
)
def test_schema6_process_flags_rejects_atomic_drift(field, bad):
    with pytest.raises(ValueError, match=field):
        util.process_flags(_schema6_flags(**{field: bad}))


def test_schema6_process_flags_rejects_numpy_boolean_before_normalization():
    with pytest.raises(ValueError, match="Python bool"):
        util.process_flags(
            _schema6_flags(voc_eval_stochastic=np.bool_(True))
        )


@pytest.mark.parametrize("mode", ["shadow", "control"])
def test_voc_modes_require_dynamic_factorized_fixed_cost(mode):
    flags = _flags(
        dynamic_factorized_control=True,
        dynamic_voc_mode=mode,
        think_cost=0.0005,
    )

    util.process_flags(flags)

    assert flags.dynamic_voc_mode == mode
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
    assert flags.voc_gate_learning_rate == pytest.approx(0.0003)
    assert flags.voc_gate_grad_norm_clipping == pytest.approx(1.0)
    assert flags.entropy_r_cost == 0.0


@pytest.mark.parametrize("bad_mode", [None, "", "Shadow", "enabled", 1, True])
def test_voc_rejects_unknown_mode(bad_mode):
    with pytest.raises(ValueError, match="dynamic_voc_mode"):
        util.process_flags(_flags(dynamic_voc_mode=bad_mode))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"voc_loss_cost": -1.0}, "voc_loss_cost"),
        ({"voc_loss_cost": float("nan")}, "voc_loss_cost"),
        ({"voc_gate_temperature": 0.0}, "voc_gate_temperature"),
        ({"voc_gate_temperature": float("inf")}, "voc_gate_temperature"),
        ({"voc_train_epsilon": -0.01}, "voc_train_epsilon"),
        ({"voc_train_epsilon": 1.01}, "voc_train_epsilon"),
        ({"voc_eval_stochastic": 1}, "voc_eval_stochastic"),
        ({"voc_dueling_q": 1}, "voc_dueling_q"),
        ({"voc_expected_gate_loss": "true"}, "voc_expected_gate_loss"),
        ({"voc_ema_gate_target": 1}, "voc_ema_gate_target"),
        ({"voc_gate_target_tau": 0.0}, "voc_gate_target_tau"),
        ({"voc_gate_target_tau": 1.01}, "voc_gate_target_tau"),
        ({"voc_gate_target_tau": float("nan")}, "voc_gate_target_tau"),
        ({"voc_dedicated_gate": 1}, "voc_dedicated_gate"),
        ({"voc_soft_q_bce_gate": "true"}, "voc_soft_q_bce_gate"),
        ({"voc_gate_q_temperature": 0.0}, "voc_gate_q_temperature"),
        ({"voc_gate_q_temperature": float("nan")}, "voc_gate_q_temperature"),
        ({"voc_gate_confidence_weighted": 1}, "voc_gate_confidence_weighted"),
        ({"voc_gate_adam_beta1": -0.01}, "voc_gate_adam_beta1"),
        ({"voc_gate_adam_beta1": 1.0}, "voc_gate_adam_beta1"),
        ({"voc_gate_adam_beta1": float("nan")}, "voc_gate_adam_beta1"),
        ({"voc_gate_adam_beta1": True}, "voc_gate_adam_beta1"),
        ({"voc_gate_param_align": 1}, "voc_gate_param_align"),
        ({"voc_gate_exact_projection": 1}, "voc_gate_exact_projection"),
        (
            {"voc_gate_epsilon_greedy_execution": 1},
            "voc_gate_epsilon_greedy_execution",
        ),
        ({"voc_gate_param_align_coef": True}, "voc_gate_param_align_coef"),
        ({"voc_gate_param_align_coef": 0.0}, "voc_gate_param_align_coef"),
        (
            {"voc_gate_param_align_coef": math.nextafter(1.0, 2.0)},
            "voc_gate_param_align_coef",
        ),
        (
            {"voc_gate_param_align_coef": float("nan")},
            "voc_gate_param_align_coef",
        ),
        ({"voc_gate_learning_rate": 0.0}, "voc_gate_learning_rate"),
        ({"voc_gate_grad_norm_clipping": 0.0}, "voc_gate_grad_norm_clipping"),
    ],
)
def test_voc_rejects_invalid_hyperparameters(overrides, message):
    with pytest.raises(ValueError, match=message):
        util.process_flags(_flags(**overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        {"dynamic_search": False, "dynamic_factorized_control": False},
        {"dynamic_factorized_control": False},
        {"think_cost": 0.002},
        {"think_cost": 0.0005, "think_cost_anneal": True},
        {"think_cost": 0.0005, "voc_loss_cost": 0.0},
        {"think_cost": 0.0005, "voc_dueling_q": False},
        {"think_cost": 0.0005, "voc_expected_gate_loss": False},
        {"think_cost": 0.0005, "voc_ema_gate_target": False},
        {"think_cost": 0.0005, "voc_gate_target_tau": 0.0},
        {"think_cost": 0.0005, "voc_dedicated_gate": False},
        {"think_cost": 0.0005, "voc_soft_q_bce_gate": False},
        {"think_cost": 0.0005, "voc_gate_q_temperature": 0.0},
        {"think_cost": 0.0005, "voc_gate_learning_rate": 0.0},
        {"think_cost": 0.0005, "voc_gate_grad_norm_clipping": 0.0},
        {"think_cost": 0.0005, "entropy_r_cost": 1e-12},
        {"think_cost": 0.0005, "entropy_r_cost": float("nan")},
        {"think_cost": 0.0005, "entropy_r_cost": False},
    ],
)
def test_active_voc_rejects_incompatible_protocol(overrides):
    values = dict(
        dynamic_voc_mode="shadow",
        dynamic_factorized_control=True,
        think_cost=0.0005,
    )
    values.update(overrides)
    with pytest.raises(ValueError):
        util.process_flags(_flags(**values))


@pytest.mark.parametrize("mode", ["shadow", "control"])
def test_active_voc_accepts_unweighted_soft_target_gate(mode):
    flags = _flags(
        dynamic_factorized_control=True,
        dynamic_voc_mode=mode,
        think_cost=0.0005,
        voc_gate_confidence_weighted=False,
    )

    util.process_flags(flags)

    assert flags.voc_gate_confidence_weighted is False


@pytest.mark.parametrize("mode", ["shadow", "control"])
def test_active_voc_accepts_zero_gate_adam_beta1(mode):
    flags = _flags(
        dynamic_factorized_control=True,
        dynamic_voc_mode=mode,
        think_cost=0.0005,
        voc_gate_adam_beta1=0,
    )

    util.process_flags(flags)

    assert flags.voc_gate_adam_beta1 == 0.0


@pytest.mark.parametrize("mode", ["shadow", "control"])
def test_active_voc_accepts_exact_parameter_alignment_protocol(mode):
    flags = _flags(
        dynamic_factorized_control=True,
        dynamic_voc_mode=mode,
        think_cost=0.0005,
        voc_gate_param_align=True,
        voc_gate_param_align_coef=1.0,
    )

    util.process_flags(flags)

    assert flags.voc_gate_param_align is True
    assert flags.voc_gate_param_align_coef == 1.0


def test_exact_projection_is_control_only_fresh_and_alignment_exclusive():
    flags = _flags(
        dynamic_factorized_control=True,
        dynamic_voc_mode="control",
        think_cost=0.0005,
        voc_gate_param_align=False,
        voc_gate_exact_projection=True,
    )

    util.process_flags(flags)

    assert flags.voc_gate_exact_projection is True
    assert flags.voc_gate_param_align is False

    for overrides in (
        {"dynamic_voc_mode": "shadow"},
        {"voc_gate_param_align": True},
        {"preload": "/model-parent"},
        {"preload_actor": "/actor-parent"},
        {"voc_parent_checkpoint": "/explicit-parent.tar"},
    ):
        invalid = vars(flags).copy()
        invalid.update(overrides)
        invalid["ckp"] = False
        with pytest.raises(
            ValueError,
            match="exact_projection|requires preload|parent-free",
        ):
            util.process_flags(Namespace(**invalid))


@pytest.mark.parametrize(
    "name", ("preload", "preload_actor", "voc_parent_checkpoint")
)
def test_exact_projection_resume_configuration_rejects_preload_surfaces(name):
    flags = _flags(
        dynamic_factorized_control=True,
        dynamic_voc_mode="control",
        think_cost=0.0005,
        voc_gate_exact_projection=True,
        ckp=True,
        **{name: "/forbidden-parent"},
    )

    with pytest.raises(ValueError, match=rf"{name}=''"):
        util.process_flags(flags)


def test_epsilon_greedy_execution_is_schema5_control_exact_only():
    flags = _flags(
        dynamic_factorized_control=True,
        dynamic_voc_mode="control",
        think_cost=0.0005,
        voc_gate_param_align=False,
        voc_gate_param_align_coef=1.0,
        voc_gate_exact_projection=True,
        voc_gate_epsilon_greedy_execution=True,
    )

    util.process_flags(flags)

    assert flags.voc_gate_epsilon_greedy_execution is True
    for overrides in (
        {"dynamic_voc_mode": "shadow"},
        {"voc_gate_exact_projection": False},
        {"voc_gate_param_align": True},
    ):
        invalid = vars(flags).copy()
        invalid.update(overrides)
        with pytest.raises(
            ValueError,
            match="epsilon_greedy_execution|exact_projection",
        ):
            util.process_flags(Namespace(**invalid))


def test_off_voc_preserves_legacy_entropy_reward_compatibility():
    flags = _flags(entropy_r_cost=0.125)

    util.process_flags(flags)

    assert flags.dynamic_voc_mode == "off"
    assert flags.entropy_r_cost == pytest.approx(0.125)


def test_optimizer_scheduler_resume_applies_saved_progress_before_next_step():
    source_parameter = torch.nn.Parameter(torch.zeros(()))
    source_optimizer = torch.optim.Adam([source_parameter], lr=0.001)
    source_scheduler = torch.optim.lr_scheduler.LambdaLR(
        source_optimizer, lambda step: max(0.0, 1.0 - 0.25 * step)
    )
    for _ in range(3):
        source_optimizer.step()
        source_scheduler.step()
    optimizer_state = source_optimizer.state_dict()
    scheduler_state = source_scheduler.state_dict()
    saved_optimizer_lr = optimizer_state["param_groups"][0]["lr"]
    saved_base_lr = scheduler_state["base_lrs"][0]

    resumed_parameter = torch.nn.Parameter(torch.zeros(()))
    resumed_optimizer = torch.optim.Adam([resumed_parameter], lr=0.002)
    resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(
        resumed_optimizer, lambda step: max(0.0, 1.0 - 0.25 * step)
    )
    util.load_optimizer(resumed_optimizer, optimizer_state)
    util.load_scheduler(resumed_scheduler, scheduler_state)

    assert resumed_optimizer.param_groups[0]["lr"] == pytest.approx(0.0005)
    assert resumed_scheduler.get_last_lr() == pytest.approx([0.0005])
    assert optimizer_state["param_groups"][0]["lr"] == saved_optimizer_lr
    assert scheduler_state["base_lrs"][0] == saved_base_lr


def _shadow_checkpoint(**overrides):
    embedded = {
        **util.VOC_PROTOCOL_DEFAULTS,
        "dynamic_voc_mode": "shadow",
        "dynamic_search": True,
        "dynamic_factorized_control": True,
        "think_cost": 0.0005,
        "think_cost_anneal": False,
        "float16": False,
        "actor_use_rms": False,
        "actor_adam_eps": 1e-8,
        "actor_learning_rate": 0.0003,
        "schedule_total_steps": 100,
        "total_steps": 100,
        "self_play_n": 1,
        "env_n": 16,
    }
    checkpoint = {
        "real_step": 3,
        "dynamic_voc_mode": "shadow",
        "voc_ema_gate_target": True,
        "voc_gate_target_tau": 0.1,
        "voc_ema_gate_schema_version": util.VOC_EMA_GATE_SCHEMA_VERSION,
        "voc_ema_gate_head_state_dict": {
            "weight": torch.zeros(2, 4, dtype=torch.float32),
            "bias": torch.zeros(2, dtype=torch.float32),
        },
        "voc_ema_gate_update_count": 3,
        "voc_ema_gate_parent_update_count": 0,
        "voc_gate_policy_schema_version": (
            util.VOC_GATE_POLICY_SCHEMA_VERSION
        ),
        "voc_gate_update_count": 0,
        "voc_gate_amp_skip_count": 0,
        "voc_gate_amp_consecutive_skips": 0,
        "voc_update_count": 3,
        "voc_continue_count": 8,
        "voc_stop_count": 5,
        "voc_holdout_count": 10,
        "voc_holdout_split_version": util.VOC_HOLDOUT_SPLIT_VERSION,
        "voc_holdout_actor_modulus": util.VOC_HOLDOUT_ACTOR_MODULUS,
        "voc_holdout_actor_streams": 16,
        "voc_holdout_continue_count": 6,
        "voc_holdout_stop_count": 4,
        "voc_holdout_td_bias": 0.1,
        "voc_holdout_td_mae": 0.2,
        "voc_holdout_td_rmse": 0.3,
        "voc_holdout_td_sum": 1.0,
        "voc_holdout_td_abs_sum": 2.0,
        "voc_holdout_td_sq_sum": 0.9,
        "voc_parent_checkpoint_sha256": None,
        "voc_parent_checkpoint": None,
        "voc_parent_imitation_data_signature": None,
        "voc_control_origin": None,
        "voc_control_origin_legacy_defaulted": False,
        "voc_activation_real_step": -1,
        "imitation_data_signature": "a" * 64,
        "flags": embedded,
        "actor_net_state_dict": {
            "voc_head.weight": torch.zeros(2, 4),
            "voc_head.bias": torch.zeros(2),
            "voc_gate_head.weight": torch.zeros(1, 4),
            "voc_gate_head.bias": torch.zeros(1),
        },
        "voc_optimizer_state_dict": {
            "state": {
                0: {
                    "step": torch.tensor(3.0),
                    "exp_avg": torch.zeros(2, 4),
                    "exp_avg_sq": torch.zeros(2, 4),
                },
                1: {
                    "step": torch.tensor(3.0),
                    "exp_avg": torch.zeros(2),
                    "exp_avg_sq": torch.zeros(2),
                },
            },
            "param_groups": [{
                "params": [0, 1],
                "lr": 0.000291,
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
            "last_epoch": 3,
            "_step_count": 4,
            "_is_initial": False,
            "_get_lr_called_within_step": False,
            "_last_lr": [0.000291],
            "lr_lambdas": [None],
        },
        "voc_grad_scaler_state_dict": None,
        "voc_amp_skip_count": 0,
        "voc_amp_consecutive_skips": 0,
        "voc_gate_optimizer_state_dict": {
            "state": {},
            "param_groups": [{
                "params": [0, 1],
                "lr": 0.0003,
                "initial_lr": 0.0003,
                "eps": 1e-8,
                "weight_decay": 0.0,
                "betas": (embedded["voc_gate_adam_beta1"], 0.999),
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
            "base_lrs": [0.0003],
            "last_epoch": 0,
            "_step_count": 1,
            "_is_initial": False,
            "_get_lr_called_within_step": False,
            "_last_lr": [0.0003],
            "lr_lambdas": [None],
        },
        "voc_gate_grad_scaler_state_dict": None,
    }
    checkpoint.update(overrides)
    return checkpoint


def _learned_gate_checkpoint():
    checkpoint = _shadow_checkpoint()
    checkpoint["dynamic_voc_mode"] = "control"
    checkpoint["flags"]["dynamic_voc_mode"] = "control"
    checkpoint["voc_gate_update_count"] = 3
    checkpoint["actor_net_state_dict"]["voc_gate_head.weight"].fill_(0.1)
    checkpoint["actor_net_state_dict"]["voc_gate_head.bias"].fill_(0.1)
    checkpoint["voc_gate_optimizer_state_dict"] = {
        "state": {
            0: {
                "step": torch.tensor(3.0),
                "exp_avg": torch.zeros(1, 4),
                "exp_avg_sq": torch.zeros(1, 4),
            },
            1: {
                "step": torch.tensor(3.0),
                "exp_avg": torch.zeros(1),
                "exp_avg_sq": torch.zeros(1),
            },
        },
        "param_groups": [{
            "params": [0, 1],
            "lr": 0.000291,
            "initial_lr": 0.0003,
            "eps": 1e-8,
            "weight_decay": 0.0,
            "betas": (
                checkpoint["flags"]["voc_gate_adam_beta1"],
                0.999,
            ),
            "amsgrad": False,
            "maximize": False,
            "foreach": None,
            "capturable": False,
            "differentiable": False,
            "fused": None,
            "decoupled_weight_decay": False,
        }],
    }
    checkpoint["voc_gate_scheduler_state_dict"] = {
        "base_lrs": [0.0003],
        "last_epoch": 3,
        "_step_count": 4,
        "_is_initial": False,
        "_get_lr_called_within_step": False,
        "_last_lr": [0.000291],
        "lr_lambdas": [None],
    }
    return checkpoint


def _exact_projection_checkpoint():
    checkpoint = _shadow_checkpoint()
    checkpoint["dynamic_voc_mode"] = "control"
    checkpoint["flags"].update({
        "dynamic_voc_mode": "control",
        "voc_gate_param_align": False,
        "voc_gate_param_align_coef": 1.0,
        "voc_gate_exact_projection": True,
        "preload": "",
        "preload_actor": "",
        "voc_parent_checkpoint": "",
    })
    checkpoint["voc_gate_policy_schema_version"] = (
        util.VOC_GATE_POLICY_EXACT_PROJECTION_SCHEMA_VERSION
    )
    checkpoint["voc_control_origin"] = util.VOC_CONTROL_ORIGIN_FRESH
    checkpoint["voc_control_origin_legacy_defaulted"] = False
    checkpoint["voc_activation_real_step"] = 0
    checkpoint["voc_parent_checkpoint_sha256"] = None
    checkpoint["voc_parent_checkpoint"] = None
    checkpoint["voc_parent_imitation_data_signature"] = None
    checkpoint["voc_gate_update_count"] = checkpoint["voc_update_count"]
    ema_weight = torch.tensor(
        [[0.125, -0.25, 0.5, -1.0], [-0.375, 0.5, 0.25, -0.5]],
        dtype=torch.float32,
    )
    ema_bias = torch.tensor([0.125, -0.25], dtype=torch.float32)
    checkpoint["voc_ema_gate_head_state_dict"] = {
        "weight": ema_weight,
        "bias": ema_bias,
    }
    scale = (
        checkpoint["flags"]["voc_gate_temperature"]
        / checkpoint["flags"]["voc_gate_q_temperature"]
    )
    checkpoint["actor_net_state_dict"]["voc_gate_head.weight"] = (
        scale * (ema_weight[0:1] - ema_weight[1:2])
    )
    checkpoint["actor_net_state_dict"]["voc_gate_head.bias"] = (
        scale * (ema_bias[0:1] - ema_bias[1:2])
    )
    return checkpoint


def _epsilon_greedy_execution_checkpoint():
    checkpoint = _exact_projection_checkpoint()
    checkpoint["voc_gate_policy_schema_version"] = (
        util.VOC_GATE_POLICY_EPSILON_GREEDY_EXECUTION_SCHEMA_VERSION
    )
    checkpoint["flags"]["voc_gate_epsilon_greedy_execution"] = True
    return checkpoint


def test_voc_control_preload_accepts_learned_shadow_parent(tmp_path):
    checkpoint = _shadow_checkpoint()
    checkpoint["flags"].pop("voc_model_input_seal_schema_version")
    path = tmp_path / "ckp_actor.tar"
    torch.save(checkpoint, path)

    state = util.validate_voc_control_preload(path)

    assert state["dynamic_voc_mode"] == "shadow"
    assert state["voc_control_origin"] == "shadow_parent"
    assert state["voc_update_count"] == 3
    assert len(state["voc_parent_checkpoint_sha256"]) == 64
    assert state["voc_parent_checkpoint"] == str(path)


def test_voc_control_preload_accepts_schema3_shadow_alignment_metadata(
    tmp_path,
):
    checkpoint = _shadow_checkpoint()
    checkpoint["flags"]["voc_gate_param_align"] = True
    path = tmp_path / "ckp_actor.tar"
    torch.save(checkpoint, path)

    state = util.validate_voc_control_preload(path)

    assert state["voc_gate_policy_schema_version"] == 3
    assert state["voc_gate_param_align"] is True
    assert state["voc_gate_param_align_coef"] == 1.0


@pytest.mark.parametrize("schema", [1, 2])
def test_voc_control_preload_accepts_legacy_schema_alignment_defaults(
    schema, tmp_path
):
    checkpoint = _shadow_checkpoint()
    checkpoint["voc_gate_policy_schema_version"] = schema
    checkpoint["flags"].pop("voc_gate_param_align")
    checkpoint["flags"].pop("voc_gate_param_align_coef")
    if schema == 1:
        checkpoint["flags"].pop("voc_gate_adam_beta1")
    path = tmp_path / "ckp_actor.tar"
    torch.save(checkpoint, path)

    state = util.validate_voc_control_preload(path)

    assert state["voc_gate_policy_schema_version"] == schema
    assert state["voc_gate_param_align"] is False
    assert state["voc_gate_param_align_coef"] == 1.0
    assert state["voc_gate_param_align_legacy_defaulted"] is True


def test_voc_gate_policy_validator_accepts_complete_learned_adam_state():
    state = util.validate_voc_gate_policy_checkpoint(
        _learned_gate_checkpoint()
    )

    assert state["voc_gate_policy_schema_version"] == 3
    assert state["voc_gate_param_align"] is False
    assert state["voc_gate_param_align_coef"] == 1.0
    assert state["voc_gate_param_align_legacy_defaulted"] is False
    assert state["voc_gate_update_count"] == 3
    assert state["voc_gate_optimizer_state_saved"] is True


@pytest.mark.parametrize("confidence_weighted", [True, False])
def test_voc_gate_policy_validator_preserves_confidence_protocol(
    confidence_weighted,
):
    checkpoint = _learned_gate_checkpoint()
    checkpoint["flags"]["voc_gate_confidence_weighted"] = confidence_weighted

    state = util.validate_voc_gate_policy_checkpoint(checkpoint)

    assert state["voc_gate_confidence_weighted"] is confidence_weighted


def test_voc_gate_policy_validator_accepts_v2_zero_adam_beta1():
    checkpoint = _learned_gate_checkpoint()
    checkpoint["voc_gate_policy_schema_version"] = 2
    checkpoint["flags"].pop("voc_gate_param_align")
    checkpoint["flags"].pop("voc_gate_param_align_coef")
    checkpoint["flags"]["voc_gate_adam_beta1"] = 0.0
    checkpoint["voc_gate_optimizer_state_dict"]["param_groups"][0][
        "betas"
    ] = (0.0, 0.999)

    state = util.validate_voc_gate_policy_checkpoint(checkpoint)

    assert state["voc_gate_policy_schema_version"] == 2
    assert state["voc_gate_adam_beta1"] == 0.0
    assert state["voc_gate_adam_beta1_legacy_defaulted"] is False
    assert state["voc_gate_param_align"] is False
    assert state["voc_gate_param_align_coef"] == 1.0
    assert state["voc_gate_param_align_legacy_defaulted"] is True


def test_voc_gate_policy_validator_accepts_schema1_missing_beta_as_legacy_point9():
    checkpoint = _learned_gate_checkpoint()
    checkpoint["voc_gate_policy_schema_version"] = 1
    checkpoint["flags"].pop("voc_gate_adam_beta1")
    checkpoint["flags"].pop("voc_gate_param_align")
    checkpoint["flags"].pop("voc_gate_param_align_coef")

    state = util.validate_voc_gate_policy_checkpoint(checkpoint)

    assert state["voc_gate_policy_schema_version"] == 1
    assert state["voc_gate_adam_beta1"] == pytest.approx(0.9)
    assert state["voc_gate_adam_beta1_legacy_defaulted"] is True
    assert state["voc_gate_param_align"] is False
    assert state["voc_gate_param_align_coef"] == 1.0
    assert state["voc_gate_param_align_legacy_defaulted"] is True


@pytest.mark.parametrize("schema", [1, 2])
def test_voc_gate_policy_legacy_schema_canonicalizes_missing_alignment(schema):
    checkpoint = _learned_gate_checkpoint()
    checkpoint["voc_gate_policy_schema_version"] = schema
    checkpoint["flags"].pop("voc_gate_param_align")
    checkpoint["flags"].pop("voc_gate_param_align_coef")
    if schema == 1:
        checkpoint["flags"].pop("voc_gate_adam_beta1")

    state = util.validate_voc_gate_policy_checkpoint(checkpoint)

    assert state["voc_gate_param_align"] is False
    assert state["voc_gate_param_align_coef"] == 1.0
    assert state["voc_gate_param_align_legacy_defaulted"] is True


@pytest.mark.parametrize("schema", [1, 2, 3])
def test_pre_projection_schemas_canonicalize_missing_projection_false(schema):
    checkpoint = _learned_gate_checkpoint()
    checkpoint["voc_gate_policy_schema_version"] = schema
    checkpoint["flags"].pop("voc_gate_exact_projection", None)
    if schema < 3:
        checkpoint["flags"].pop("voc_gate_param_align")
        checkpoint["flags"].pop("voc_gate_param_align_coef")
    if schema == 1:
        checkpoint["flags"].pop("voc_gate_adam_beta1")

    state = util.validate_voc_gate_policy_schema(checkpoint)

    assert state["voc_gate_exact_projection"] is False
    assert state["voc_gate_exact_projection_legacy_defaulted"] is True


def test_schema4_requires_explicit_true_projection_and_exact_coefficient():
    checkpoint = _exact_projection_checkpoint()
    state = util.validate_voc_gate_policy_schema(checkpoint)
    assert state["voc_gate_policy_schema_version"] == 4
    assert state["voc_gate_exact_projection"] is True
    assert state["voc_gate_exact_projection_legacy_defaulted"] is False

    for value in (None, False, 0, 1, "true"):
        corrupt = _exact_projection_checkpoint()
        if value is None:
            corrupt["flags"].pop("voc_gate_exact_projection")
        else:
            corrupt["flags"]["voc_gate_exact_projection"] = value
        with pytest.raises(ValueError, match="exact_projection"):
            util.validate_voc_gate_policy_schema(corrupt)

    corrupt = _exact_projection_checkpoint()
    corrupt["flags"]["voc_gate_param_align"] = True
    with pytest.raises(ValueError, match="mutually exclusive"):
        util.validate_voc_gate_policy_schema(corrupt)

    corrupt = _exact_projection_checkpoint()
    corrupt["flags"]["voc_gate_param_align_coef"] = math.nextafter(1.0, 2.0)
    with pytest.raises(ValueError, match="coef=1.0 exactly"):
        util.validate_voc_gate_policy_schema(corrupt)


def test_gate_protocol_tuple_appends_v12_without_reordering_prior_fields():
    protocol = util._require_voc_gate_policy_protocol(
        True,
        True,
        0.05,
        False,
        0.9,
        0.0003,
        1.0,
        False,
        1.0,
        True,
        False,
        label="test protocol",
    )

    assert protocol == (
        True,
        True,
        0.05,
        False,
        0.9,
        0.0003,
        1.0,
        False,
        1.0,
        True,
        False,
    )


@pytest.mark.parametrize("schema", [1, 2, 3, 4])
def test_pre_v12_schemas_canonicalize_missing_execution_false(schema):
    checkpoint = (
        _exact_projection_checkpoint()
        if schema == 4
        else _learned_gate_checkpoint()
    )
    checkpoint["voc_gate_policy_schema_version"] = schema
    checkpoint["flags"].pop("voc_gate_epsilon_greedy_execution", None)
    if schema < 4:
        checkpoint["flags"].pop("voc_gate_exact_projection", None)
    if schema < 3:
        checkpoint["flags"].pop("voc_gate_param_align")
        checkpoint["flags"].pop("voc_gate_param_align_coef")
    if schema == 1:
        checkpoint["flags"].pop("voc_gate_adam_beta1")

    state = util.validate_voc_gate_policy_schema(checkpoint)

    assert state["voc_gate_epsilon_greedy_execution"] is False
    assert state[
        "voc_gate_epsilon_greedy_execution_legacy_defaulted"
    ] is True


def test_schema5_requires_explicit_true_execution_and_exact_v11_base():
    checkpoint = _epsilon_greedy_execution_checkpoint()
    state = util.validate_voc_gate_policy_checkpoint(checkpoint)

    assert state["voc_gate_policy_schema_version"] == 5
    assert state["voc_gate_exact_projection"] is True
    assert state["voc_gate_param_align"] is False
    assert state["voc_gate_param_align_coef"] == 1.0
    assert state["voc_gate_epsilon_greedy_execution"] is True
    assert state[
        "voc_gate_epsilon_greedy_execution_legacy_defaulted"
    ] is False

    for value in (None, False, 0, 1, "true"):
        corrupt = _epsilon_greedy_execution_checkpoint()
        if value is None:
            corrupt["flags"].pop("voc_gate_epsilon_greedy_execution")
        else:
            corrupt["flags"]["voc_gate_epsilon_greedy_execution"] = value
        with pytest.raises(ValueError, match="epsilon_greedy_execution"):
            util.validate_voc_gate_policy_schema(corrupt)

    corrupt = _epsilon_greedy_execution_checkpoint()
    corrupt["flags"]["voc_gate_exact_projection"] = False
    with pytest.raises(ValueError, match="exact_projection"):
        util.validate_voc_gate_policy_schema(corrupt)

    corrupt = _epsilon_greedy_execution_checkpoint()
    corrupt["flags"]["voc_gate_param_align"] = True
    with pytest.raises(ValueError, match="mutually exclusive"):
        util.validate_voc_gate_policy_schema(corrupt)

    corrupt = _epsilon_greedy_execution_checkpoint()
    corrupt["flags"]["voc_gate_param_align_coef"] = math.nextafter(1.0, 2.0)
    with pytest.raises(ValueError, match="coef=1.0 exactly"):
        util.validate_voc_gate_policy_schema(corrupt)


def test_pre_v12_schema_rejects_enabled_execution_metadata():
    checkpoint = _exact_projection_checkpoint()
    checkpoint["flags"]["voc_gate_epsilon_greedy_execution"] = True

    with pytest.raises(ValueError, match="predates epsilon-greedy execution"):
        util.validate_voc_gate_policy_schema(checkpoint)

def test_pre_projection_schema_rejects_true_projection_metadata():
    checkpoint = _learned_gate_checkpoint()
    checkpoint["flags"]["voc_gate_exact_projection"] = True

    with pytest.raises(ValueError, match="predates exact projection"):
        util.validate_voc_gate_policy_schema(checkpoint)


def test_exact_projection_checkpoint_validates_fresh_bit_exact_bundle():
    checkpoint = _exact_projection_checkpoint()

    state = util.validate_voc_gate_policy_checkpoint(checkpoint)

    assert state["voc_gate_policy_schema_version"] == 4
    assert state["voc_gate_exact_projection"] is True
    assert state["voc_gate_update_count"] == checkpoint["voc_update_count"]
    assert checkpoint["voc_gate_optimizer_state_dict"]["state"] == {}
    assert checkpoint["voc_gate_scheduler_state_dict"]["last_epoch"] == 0
    ema = checkpoint["voc_ema_gate_head_state_dict"]
    scale = (
        checkpoint["flags"]["voc_gate_temperature"]
        / checkpoint["flags"]["voc_gate_q_temperature"]
    )
    assert torch.equal(
        checkpoint["actor_net_state_dict"]["voc_gate_head.weight"],
        scale * (ema["weight"][0:1] - ema["weight"][1:2]),
    )
    assert torch.equal(
        checkpoint["actor_net_state_dict"]["voc_gate_head.bias"],
        scale * (ema["bias"][0:1] - ema["bias"][1:2]),
    )


def test_exact_projection_fp16_checkpoint_requires_pristine_gate_scaler():
    checkpoint = _exact_projection_checkpoint()
    checkpoint["flags"]["float16"] = True
    checkpoint["voc_gate_grad_scaler_state_dict"] = {
        "scale": 256.0,
        "growth_factor": 2.0,
        "backoff_factor": 0.5,
        "growth_interval": 2000,
        "_growth_tracker": 0,
    }

    state = util.validate_voc_gate_policy_checkpoint(checkpoint)
    assert state["voc_gate_grad_scaler_state_saved"] is True

    for field, value in (("scale", 512.0), ("_growth_tracker", 1)):
        corrupt = _exact_projection_checkpoint()
        corrupt["flags"]["float16"] = True
        corrupt["voc_gate_grad_scaler_state_dict"] = dict(
            checkpoint["voc_gate_grad_scaler_state_dict"]
        )
        corrupt["voc_gate_grad_scaler_state_dict"][field] = value
        with pytest.raises(ValueError, match="pristine gate GradScaler"):
            util.validate_voc_gate_policy_checkpoint(corrupt)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("weight_nextafter", "disagrees with EMA Q target"),
        ("bias_nextafter", "disagrees with EMA Q target"),
        ("gate_float64", "FP32 gate head"),
        ("ema_float64", "FP32 master tensors"),
        ("optimizer_state", "empty gate optimizer state"),
        ("scheduler_step", "scheduler step count"),
        ("gate_count", "count must equal"),
        ("promoted_origin", "fresh control origin"),
    ],
)
def test_exact_projection_checkpoint_rejects_corruption(mutation, message):
    checkpoint = _exact_projection_checkpoint()
    if mutation == "weight_nextafter":
        weight = checkpoint["actor_net_state_dict"]["voc_gate_head.weight"]
        weight[0, 0] = torch.nextafter(
            weight[0, 0], torch.tensor(float("inf"))
        )
    elif mutation == "bias_nextafter":
        bias = checkpoint["actor_net_state_dict"]["voc_gate_head.bias"]
        bias[0] = torch.nextafter(bias[0], torch.tensor(float("inf")))
    elif mutation == "gate_float64":
        checkpoint["actor_net_state_dict"]["voc_gate_head.weight"] = (
            checkpoint["actor_net_state_dict"]["voc_gate_head.weight"].double()
        )
    elif mutation == "ema_float64":
        checkpoint["voc_ema_gate_head_state_dict"]["weight"] = checkpoint[
            "voc_ema_gate_head_state_dict"
        ]["weight"].double()
    elif mutation == "optimizer_state":
        checkpoint["voc_gate_optimizer_state_dict"]["state"] = {
            0: {"step": torch.tensor(1.0)}
        }
    elif mutation == "scheduler_step":
        checkpoint["voc_gate_scheduler_state_dict"]["_step_count"] = 2
    elif mutation == "gate_count":
        checkpoint["voc_gate_update_count"] -= 1
    elif mutation == "promoted_origin":
        checkpoint["voc_control_origin"] = util.VOC_CONTROL_ORIGIN_SHADOW_PARENT
        checkpoint["voc_parent_checkpoint_sha256"] = "a" * 64
        checkpoint["voc_parent_checkpoint"] = "/tmp/parent.tar"
        checkpoint["voc_parent_imitation_data_signature"] = "b" * 64

    with pytest.raises(ValueError, match=message):
        util.validate_voc_gate_policy_checkpoint(checkpoint)


def test_schema4_resume_binds_projection_identity_and_rejects_promotion(
    tmp_path,
):
    checkpoint = _exact_projection_checkpoint()
    flags = Namespace(**checkpoint["flags"])
    assert util.validate_voc_resume_protocol(checkpoint, flags)[
        "voc_gate_exact_projection"
    ] is True

    flags.voc_gate_exact_projection = False
    with pytest.raises(ValueError, match="voc_gate_exact_projection"):
        util.validate_voc_resume_protocol(checkpoint, flags)

    path = tmp_path / "schema4-control.tar"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="requires a shadow checkpoint"):
        util.validate_voc_control_preload(path)


def test_schema4_shadow_metadata_is_fail_closed(tmp_path):
    checkpoint = _shadow_checkpoint()
    checkpoint["voc_gate_policy_schema_version"] = 4
    checkpoint["flags"]["voc_gate_exact_projection"] = True
    path = tmp_path / "schema4-shadow.tar"
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="exact projection requires control"):
        util.validate_voc_control_preload(path)


def test_schema5_resume_binds_execution_identity_and_rejects_promotion(
    tmp_path,
):
    checkpoint = _epsilon_greedy_execution_checkpoint()
    checkpoint["flags"].pop("voc_model_input_seal_schema_version")
    flags = Namespace(**checkpoint["flags"])

    protocol = util.validate_voc_resume_protocol(checkpoint, flags)

    assert protocol["voc_gate_epsilon_greedy_execution"] is True
    assert protocol["voc_model_input_seal_schema_version"] == 0
    flags.voc_gate_epsilon_greedy_execution = False
    with pytest.raises(
        ValueError, match="voc_gate_epsilon_greedy_execution"
    ):
        util.validate_voc_resume_protocol(checkpoint, flags)

    path = tmp_path / "schema5-control.tar"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="requires a shadow checkpoint"):
        util.validate_voc_control_preload(path)


def test_schema5_shadow_origin_is_fail_closed(tmp_path):
    checkpoint = _epsilon_greedy_execution_checkpoint()
    checkpoint["dynamic_voc_mode"] = "shadow"
    checkpoint["flags"]["dynamic_voc_mode"] = "shadow"
    checkpoint["voc_control_origin"] = None
    checkpoint["voc_activation_real_step"] = -1
    checkpoint["voc_gate_update_count"] = 0
    path = tmp_path / "schema5-shadow.tar"
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="exact projection requires control"):
        util.validate_voc_control_preload(path)


@pytest.mark.parametrize(
    "missing", ["voc_gate_param_align", "voc_gate_param_align_coef"]
)
def test_voc_gate_policy_schema3_requires_explicit_alignment_metadata(missing):
    checkpoint = _learned_gate_checkpoint()
    checkpoint["flags"].pop(missing)

    with pytest.raises(ValueError, match=rf"schema 3 lacks embedded {missing}"):
        util.validate_voc_gate_policy_schema(checkpoint)


@pytest.mark.parametrize("value", [1, 0, "true", None])
def test_voc_gate_policy_schema3_rejects_nonboolean_alignment(value):
    checkpoint = _learned_gate_checkpoint()
    checkpoint["flags"]["voc_gate_param_align"] = value

    with pytest.raises(ValueError, match="voc_gate_param_align to be boolean"):
        util.validate_voc_gate_policy_schema(checkpoint)


@pytest.mark.parametrize(
    "value",
    [
        True,
        0.0,
        0.5,
        math.nextafter(1.0, 2.0),
        float("nan"),
        float("inf"),
    ],
)
def test_voc_gate_policy_schema3_rejects_nonexact_alignment_coefficient(value):
    checkpoint = _learned_gate_checkpoint()
    checkpoint["flags"]["voc_gate_param_align_coef"] = value

    with pytest.raises(ValueError, match="voc_gate_param_align_coef=1.0 exactly"):
        util.validate_voc_gate_policy_schema(checkpoint)


@pytest.mark.parametrize("schema", [1, 2])
def test_voc_gate_policy_legacy_schema_rejects_enabled_alignment(schema):
    checkpoint = _learned_gate_checkpoint()
    checkpoint["voc_gate_policy_schema_version"] = schema
    checkpoint["flags"]["voc_gate_param_align"] = True
    if schema == 1:
        checkpoint["flags"]["voc_gate_adam_beta1"] = 0.9

    with pytest.raises(ValueError, match="predates parameter alignment"):
        util.validate_voc_gate_policy_schema(checkpoint)


@pytest.mark.parametrize(
    ("present", "missing"),
    [
        ("voc_gate_param_align", "voc_gate_param_align_coef"),
        ("voc_gate_param_align_coef", "voc_gate_param_align"),
    ],
)
def test_voc_gate_policy_legacy_schema_rejects_partial_alignment_metadata(
    present, missing
):
    checkpoint = _learned_gate_checkpoint()
    checkpoint["voc_gate_policy_schema_version"] = 2
    checkpoint["flags"].pop(missing)

    with pytest.raises(ValueError, match=rf"lacks embedded {missing}"):
        util.validate_voc_gate_policy_schema(checkpoint)


@pytest.mark.parametrize(
    ("schema", "value", "message"),
    [
        (2, None, "schema 2 lacks embedded voc_gate_adam_beta1"),
        (2, True, "voc_gate_adam_beta1"),
        (2, -0.1, "voc_gate_adam_beta1"),
        (2, 1.0, "voc_gate_adam_beta1"),
        (2, float("nan"), "voc_gate_adam_beta1"),
        (1, 0.0, "schema 1 requires legacy voc_gate_adam_beta1=0.9"),
    ],
)
def test_voc_gate_policy_validator_rejects_missing_or_corrupt_beta1(
    schema, value, message
):
    checkpoint = _learned_gate_checkpoint()
    checkpoint["voc_gate_policy_schema_version"] = schema
    if value is None:
        checkpoint["flags"].pop("voc_gate_adam_beta1")
    else:
        checkpoint["flags"]["voc_gate_adam_beta1"] = value

    with pytest.raises(ValueError, match=message):
        util.validate_voc_gate_policy_checkpoint(checkpoint)


def test_voc_gate_policy_validator_rejects_beta1_optimizer_mismatch():
    checkpoint = _learned_gate_checkpoint()
    checkpoint["flags"]["voc_gate_adam_beta1"] = 0.0

    with pytest.raises(ValueError, match="Adam protocol disagrees"):
        util.validate_voc_gate_policy_checkpoint(checkpoint)


def test_voc_gate_policy_validator_rejects_legacy_beta1_against_zero_run():
    checkpoint = _learned_gate_checkpoint()
    checkpoint["voc_gate_policy_schema_version"] = 1
    checkpoint["flags"].pop("voc_gate_adam_beta1")
    flags = Namespace(
        actor_use_rms=False,
        actor_adam_eps=1e-8,
        actor_learning_rate=0.0003,
        schedule_total_steps=100,
        float16=False,
        voc_gate_adam_beta1=0.0,
    )

    with pytest.raises(ValueError, match="voc_gate_adam_beta1 disagrees"):
        util.validate_voc_gate_policy_checkpoint(checkpoint, flags=flags)


@pytest.mark.parametrize("beta1", [0.9, 0.0])
def test_voc_gate_policy_validator_accepts_complete_learned_rmsprop_state(
    beta1,
):
    checkpoint = _learned_gate_checkpoint()
    checkpoint["flags"]["actor_use_rms"] = True
    checkpoint["flags"]["voc_gate_adam_beta1"] = beta1
    checkpoint["voc_gate_optimizer_state_dict"] = {
        "state": {
            0: {
                "step": torch.tensor(3.0),
                "square_avg": torch.zeros(1, 4),
            },
            1: {
                "step": torch.tensor(3.0),
                "square_avg": torch.zeros(1),
            },
        },
        "param_groups": [{
            "params": [0, 1],
            "lr": 0.000291,
            "initial_lr": 0.0003,
            "eps": 0.01,
            "weight_decay": 0.0,
            "alpha": 0.99,
            "momentum": 0.0,
            "centered": False,
            "maximize": False,
            "foreach": None,
            "capturable": False,
            "differentiable": False,
            "fused": None,
        }],
    }
    flags = Namespace(
        actor_use_rms=True,
        actor_learning_rate=0.0003,
        schedule_total_steps=100,
        float16=False,
        voc_gate_adam_beta1=beta1,
    )

    state = util.validate_voc_gate_policy_checkpoint(checkpoint, flags=flags)

    assert state["voc_gate_update_count"] == 3
    assert state["voc_gate_adam_beta1"] == beta1

    flags.voc_gate_adam_beta1 = 0.5 if beta1 != 0.5 else 0.0
    with pytest.raises(ValueError, match="voc_gate_adam_beta1 disagrees"):
        util.validate_voc_gate_policy_checkpoint(checkpoint, flags=flags)


def test_voc_q_validator_accepts_complete_learned_rmsprop_state():
    checkpoint = _shadow_checkpoint()
    checkpoint["flags"]["actor_use_rms"] = True
    checkpoint["voc_optimizer_state_dict"] = {
        "state": {
            0: {
                "step": torch.tensor(3.0),
                "square_avg": torch.zeros(2, 4),
            },
            1: {
                "step": torch.tensor(3.0),
                "square_avg": torch.zeros(2),
            },
        },
        "param_groups": [{
            "params": [0, 1],
            "lr": 0.000291,
            "initial_lr": 0.0003,
            "eps": 0.01,
            "weight_decay": 0.0,
            "alpha": 0.99,
            "momentum": 0.0,
            "centered": False,
            "maximize": False,
            "foreach": None,
            "capturable": False,
            "differentiable": False,
            "fused": None,
        }],
    }

    assert util.validate_voc_checkpoint_components(checkpoint) == (
        "voc_head.weight",
        "voc_head.bias",
    )


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        ("missing_bias_state", "cover exactly"),
        ("one_param", "exactly the weight and bias"),
        ("wrong_moment_shape", "exp_avg shape"),
        ("wrong_step", "step disagrees"),
        ("nan_eps", "non-finite gate state.*eps"),
        ("bad_maximize", "optimizer maximize"),
        ("bad_weight_decay", "weight_decay must be zero"),
        ("bad_initial_lr", "LR disagrees with protocol"),
        ("partial_scheduler", "scheduler state fields are incomplete"),
        ("wrong_scheduler_step", "scheduler step count disagrees"),
        ("wrong_scheduler_lr", "scheduler LR state disagrees"),
    ],
)
def test_voc_gate_policy_validator_rejects_unresumeable_adam_state(
    corrupt, message
):
    checkpoint = _learned_gate_checkpoint()
    optimizer = checkpoint["voc_gate_optimizer_state_dict"]
    if corrupt == "missing_bias_state":
        optimizer["state"].pop(1)
    elif corrupt == "one_param":
        optimizer["param_groups"][0]["params"] = [0]
    elif corrupt == "wrong_moment_shape":
        optimizer["state"][0]["exp_avg"] = torch.zeros(2, 4)
    elif corrupt == "wrong_step":
        optimizer["state"][1]["step"] = torch.tensor(2.0)
    elif corrupt == "nan_eps":
        optimizer["param_groups"][0]["eps"] = float("nan")
    elif corrupt == "bad_maximize":
        optimizer["param_groups"][0]["maximize"] = "truthy"
    elif corrupt == "bad_weight_decay":
        optimizer["param_groups"][0]["weight_decay"] = 0.75
    elif corrupt == "bad_initial_lr":
        optimizer["param_groups"][0]["initial_lr"] = 999.0
    elif corrupt == "partial_scheduler":
        checkpoint["voc_gate_scheduler_state_dict"] = {"last_epoch": 3}
    elif corrupt == "wrong_scheduler_step":
        checkpoint["voc_gate_scheduler_state_dict"]["_step_count"] = 3
    elif corrupt == "wrong_scheduler_lr":
        checkpoint["voc_gate_scheduler_state_dict"]["_last_lr"] = [0.5]

    with pytest.raises(ValueError, match=message):
        util.validate_voc_gate_policy_checkpoint(checkpoint)


@pytest.mark.parametrize(
    "corrupt",
    [
        "missing_bias_state",
        "one_param",
        "wrong_moment_shape",
        "wrong_step",
        "partial_scheduler",
        "wrong_scheduler_step",
        "wrong_scheduler_epoch",
        "wrong_scheduler_lr",
    ],
)
def test_voc_q_validator_rejects_unresumeable_optimizer_scheduler_state(
    corrupt,
):
    checkpoint = _shadow_checkpoint()
    optimizer = checkpoint["voc_optimizer_state_dict"]
    if corrupt == "missing_bias_state":
        optimizer["state"].pop(1, None)
    elif corrupt == "one_param":
        optimizer["param_groups"][0]["params"] = [0]
    elif corrupt == "wrong_moment_shape":
        optimizer["state"][0]["exp_avg"] = torch.zeros(3, 4)
    elif corrupt == "wrong_step":
        optimizer["state"][1]["step"] = torch.tensor(2.0)
    elif corrupt == "partial_scheduler":
        checkpoint["voc_scheduler_state_dict"] = {"last_epoch": 3}
    elif corrupt == "wrong_scheduler_step":
        checkpoint["voc_scheduler_state_dict"]["_step_count"] = 3
    elif corrupt == "wrong_scheduler_epoch":
        checkpoint["voc_scheduler_state_dict"]["last_epoch"] = 50
        checkpoint["voc_scheduler_state_dict"]["_last_lr"] = [0.00015]
        optimizer["param_groups"][0]["lr"] = 0.00015
    elif corrupt == "wrong_scheduler_lr":
        checkpoint["voc_scheduler_state_dict"]["_last_lr"] = [0.0002]
        optimizer["param_groups"][0]["lr"] = 0.0002

    with pytest.raises(ValueError):
        util.validate_voc_checkpoint_components(checkpoint)


def test_voc_gate_scheduler_rejects_progress_unbound_to_real_step():
    checkpoint = _learned_gate_checkpoint()
    checkpoint["real_step"] = 3
    checkpoint["flags"]["schedule_total_steps"] = 100
    scheduler = checkpoint["voc_gate_scheduler_state_dict"]
    scheduler["last_epoch"] = 50
    scheduler["_last_lr"] = [0.00015]
    checkpoint["voc_gate_optimizer_state_dict"]["param_groups"][0]["lr"] = (
        0.00015
    )

    with pytest.raises(ValueError, match="scheduler.*real_step|LR.*schedule"):
        util.validate_voc_gate_policy_checkpoint(checkpoint)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("actor_use_rms", True),
        ("actor_adam_eps", 1e-7),
        ("voc_gate_adam_beta1", 0.0),
        ("voc_gate_param_align", True),
        ("voc_gate_param_align_coef", math.nextafter(1.0, 2.0)),
        ("float16", True),
    ],
)
def test_voc_gate_policy_validator_binds_run_optimizer_protocol(name, value):
    checkpoint = _learned_gate_checkpoint()
    flags = Namespace(
        actor_use_rms=False,
        actor_adam_eps=1e-8,
        actor_learning_rate=0.0003,
        schedule_total_steps=100,
        float16=False,
    )
    setattr(flags, name, value)

    with pytest.raises(ValueError, match=name):
        util.validate_voc_gate_policy_checkpoint(checkpoint, flags=flags)


def test_shadow_preload_accepts_only_clean_off_sources():
    clean_off = {
        "dynamic_voc_mode": "off",
        "flags": {"dynamic_voc_mode": "off"},
        "actor_net_state_dict": {"policy.weight": torch.zeros(1, 1)},
    }
    assert util.validate_voc_shadow_preload(clean_off) == {
        "dynamic_voc_mode": "off"
    }

    with pytest.raises(ValueError, match="off/legacy source"):
        util.validate_voc_shadow_preload(_shadow_checkpoint())

    control = _shadow_checkpoint()
    control["dynamic_voc_mode"] = "control"
    control["flags"]["dynamic_voc_mode"] = "control"
    with pytest.raises(ValueError, match="off/legacy source"):
        util.validate_voc_shadow_preload(control)

    stripped = _shadow_checkpoint()
    stripped["dynamic_voc_mode"] = "off"
    stripped["flags"]["dynamic_voc_mode"] = "off"
    with pytest.raises(ValueError, match="active voc_head weights"):
        util.validate_voc_shadow_preload(stripped)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("voc_ema_gate_target", False, "requires voc_ema_gate_target=true"),
        ("voc_ema_gate_schema_version", 2, "unsupported.*schema_version"),
        ("voc_ema_gate_update_count", 4, "parent count plus"),
        ("voc_ema_gate_parent_update_count", 1, "parent count plus"),
    ],
)
def test_ema_gate_checkpoint_rejects_protocol_and_counter_corruption(
    field, value, message
):
    checkpoint = _shadow_checkpoint(**{field: value})

    with pytest.raises(ValueError, match=message):
        util.validate_voc_ema_gate_checkpoint(checkpoint)


def test_ema_gate_checkpoint_rejects_nonfinite_online_and_target_heads():
    checkpoint = _shadow_checkpoint()
    checkpoint["actor_net_state_dict"]["voc_head.weight"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="online voc_head.*non-finite"):
        util.validate_voc_ema_gate_checkpoint(checkpoint)

    checkpoint = _shadow_checkpoint()
    checkpoint["voc_ema_gate_head_state_dict"]["bias"][0] = float("inf")
    with pytest.raises(ValueError, match="EMA gate state.*non-finite"):
        util.validate_voc_ema_gate_checkpoint(checkpoint)


def test_zero_update_shadow_requires_identical_zero_online_and_ema_heads():
    checkpoint = _shadow_checkpoint(
        voc_update_count=0,
        voc_ema_gate_update_count=0,
        voc_ema_gate_parent_update_count=0,
    )
    checkpoint["actor_net_state_dict"]["voc_head.bias"][0] = 1e-12

    with pytest.raises(ValueError, match="equal zero"):
        util.validate_voc_ema_gate_checkpoint(checkpoint)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("voc_parent_checkpoint_sha256", "a" * 64),
        ("voc_parent_checkpoint", "/tmp/parent.tar"),
        ("voc_parent_imitation_data_signature", "b" * 64),
        ("voc_control_origin", "fresh"),
        ("voc_activation_real_step", 0),
        ("voc_control_origin_legacy_defaulted", True),
    ],
)
def test_shadow_provenance_rejects_control_lineage(field, value):
    checkpoint = _shadow_checkpoint(**{field: value})

    with pytest.raises(ValueError):
        util.validate_voc_shadow_checkpoint_provenance(checkpoint)


def _control_provenance_checkpoint(origin, **overrides):
    checkpoint = {
        "dynamic_voc_mode": "control",
        "flags": {
            "dynamic_voc_mode": "control",
            "preload": "",
            "preload_actor": "",
            "voc_parent_checkpoint": "",
        },
        "real_step": 0,
        "voc_activation_real_step": 0,
        "voc_control_origin": origin,
        "voc_parent_checkpoint_sha256": None,
        "voc_parent_checkpoint": None,
        "voc_parent_imitation_data_signature": None,
    }
    checkpoint.update(overrides)
    return checkpoint


def test_fresh_control_provenance_requires_explicit_null_parent():
    checkpoint = _control_provenance_checkpoint("fresh")

    state = util.validate_voc_control_checkpoint_provenance(checkpoint)

    assert state["voc_control_origin"] == "fresh"
    assert state["voc_control_origin_legacy_defaulted"] is False
    assert state["voc_parent_checkpoint_sha256"] is None
    assert state["voc_parent_checkpoint"] is None
    assert state["voc_parent_imitation_data_signature"] is None
    assert state["voc_activation_real_step"] == 0

    checkpoint["voc_parent_checkpoint_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="fresh.*parent_checkpoint_sha256"):
        util.validate_voc_control_checkpoint_provenance(checkpoint)


@pytest.mark.parametrize(
    "name", ["preload", "preload_actor", "voc_parent_checkpoint"]
)
def test_fresh_control_provenance_rejects_embedded_preloads(name):
    checkpoint = _control_provenance_checkpoint("fresh")
    checkpoint["flags"][name] = "/forged/parent"

    with pytest.raises(ValueError, match=rf"fresh origin.*{name}=''"):
        util.validate_voc_control_checkpoint_provenance(checkpoint)

    checkpoint["flags"][name] = "   "
    with pytest.raises(ValueError, match=rf"fresh origin.*{name}=''"):
        util.validate_voc_control_checkpoint_provenance(checkpoint)


def test_fresh_control_launch_rejects_model_preload_without_actor_parent():
    flags = _flags(
        dynamic_factorized_control=True,
        dynamic_voc_mode="control",
        think_cost=0.0005,
        preload="/model-only-parent",
        preload_actor="",
        voc_parent_checkpoint="",
        ckp=False,
    )

    with pytest.raises(ValueError, match="requires preload=''" ):
        util.process_flags(flags)

    flags.preload = ""
    util.process_flags(flags)


def test_fresh_control_launch_canonicalizes_whitespace_preload_surfaces():
    flags = _flags(
        dynamic_factorized_control=True,
        dynamic_voc_mode="control",
        think_cost=0.0005,
        preload="   ",
        preload_actor="\t",
        voc_parent_checkpoint="\n",
        ckp=False,
    )

    util.process_flags(flags)

    assert flags.preload == ""
    assert flags.preload_actor == ""
    assert flags.voc_parent_checkpoint == ""


@pytest.mark.parametrize(
    "name", ["preload", "preload_actor", "voc_parent_checkpoint"]
)
def test_unprocessed_fresh_control_rejects_whitespace_preloads(name):
    flags = _flags(
        dynamic_voc_mode="control",
        ckp=False,
        preload="",
        preload_actor="",
        voc_parent_checkpoint="",
    )
    setattr(flags, name, "   ")

    with pytest.raises(ValueError, match=rf"requires {name}=''"):
        util.validate_voc_fresh_control_inputs(flags)


def test_fresh_control_resume_rejects_current_preload_surface():
    checkpoint = _shadow_checkpoint()
    checkpoint.update({
        "dynamic_voc_mode": "control",
        "real_step": 3,
        "voc_control_origin": "fresh",
        "voc_activation_real_step": 0,
    })
    checkpoint["flags"].update({
        "dynamic_voc_mode": "control",
        "preload": "",
        "preload_actor": "",
        "voc_parent_checkpoint": "",
    })
    flags = Namespace(**checkpoint["flags"])
    flags.preload = "/ignored/model-parent"

    with pytest.raises(ValueError, match="resume requires run preload=''" ):
        util.validate_voc_active_resume_checkpoint(checkpoint, flags)


def test_promoted_control_provenance_remains_strict_and_legacy_compatible(
    tmp_path,
):
    parent = str((tmp_path / "ckp_actor.tar").resolve())
    explicit = _control_provenance_checkpoint(
        "shadow_parent",
        voc_parent_checkpoint_sha256="b" * 64,
        voc_parent_checkpoint=parent,
        voc_parent_imitation_data_signature="c" * 64,
    )

    state = util.validate_voc_control_checkpoint_provenance(explicit)

    assert state["voc_control_origin"] == "shadow_parent"
    assert state["voc_control_origin_legacy_defaulted"] is False
    assert state["voc_parent_checkpoint"] == parent

    legacy = dict(explicit)
    for key in (
        "voc_control_origin",
        "voc_parent_checkpoint",
        "voc_parent_imitation_data_signature",
    ):
        legacy.pop(key)
    legacy_state = util.validate_voc_control_checkpoint_provenance(legacy)
    assert legacy_state["voc_control_origin"] == "shadow_parent"
    assert legacy_state["voc_control_origin_legacy_defaulted"] is True

    # A legacy promotion remains resumable after a new learner explicitly
    # records the inferred origin, without fabricating a historical path.
    migrated = dict(legacy)
    migrated["voc_control_origin"] = legacy_state["voc_control_origin"]
    migrated["voc_control_origin_legacy_defaulted"] = True
    migrated_state = util.validate_voc_control_checkpoint_provenance(migrated)
    assert migrated_state["voc_control_origin"] == "shadow_parent"
    assert migrated_state["voc_control_origin_legacy_defaulted"] is True

    explicit["voc_parent_checkpoint"] = "relative/ckp_actor.tar"
    with pytest.raises(ValueError, match="absolute voc_parent_checkpoint"):
        util.validate_voc_control_checkpoint_provenance(explicit)


def test_missing_control_origin_cannot_implicitly_mean_fresh():
    checkpoint = _control_provenance_checkpoint("fresh")
    checkpoint.pop("voc_control_origin")

    with pytest.raises(ValueError, match="voc_control_origin"):
        util.validate_voc_control_checkpoint_provenance(checkpoint)


def test_voc_control_preload_requires_exact_environment_and_data_identity(
    tmp_path,
):
    identity = {
        "name": "Pong-v5",
        "icopro_game_id": 1,
        "frame_stack_n": 4,
        "grayscale": False,
        "wrapper_type": 0,
        "icopro_subjects": "1",
        "icopro_train_sessions": "1,2,3",
        "icopro_holdout_sessions": "4",
        "self_play_n": 1,
        "env_n": 16,
    }
    checkpoint = _shadow_checkpoint()
    checkpoint["flags"].update(identity)
    checkpoint["flags"]["voc_gate_param_align"] = True
    path = tmp_path / "ckp_actor.tar"
    torch.save(checkpoint, path)
    flags = Namespace(
        **identity,
        **util.VOC_PROTOCOL_DEFAULTS,
        dynamic_search=True,
            dynamic_factorized_control=True,
            think_cost=0.0005,
            think_cost_anneal=False,
            actor_use_rms=False,
            actor_adam_eps=1e-8,
            actor_learning_rate=0.0003,
            schedule_total_steps=100,
            float16=False,
        )
    flags.dynamic_voc_mode = "control"
    flags.voc_gate_param_align = True

    util.validate_voc_control_preload(path, flags=flags)

    flags.voc_gate_param_align = False
    with pytest.raises(ValueError, match="voc_gate_param_align"):
        util.validate_voc_control_preload(path, flags=flags)

    flags.voc_gate_param_align = True
    flags.name = "Enduro-v5"
    with pytest.raises(ValueError, match="identity mismatch"):
        util.validate_voc_control_preload(path, flags=flags)


def test_voc_parent_path_defaults_to_preload_actor_and_allows_exact_override(
    tmp_path,
):
    flags = _flags(preload_actor=str(tmp_path / "shadow"))
    expected = tmp_path / "shadow" / "ckp_actor.tar"
    assert util.resolve_voc_parent_checkpoint(flags) == str(expected)

    exact = tmp_path / "promoted.tar"
    flags.voc_parent_checkpoint = str(exact)
    assert util.resolve_voc_parent_checkpoint(flags) == str(exact)


def test_voc_resume_protocol_legacy_missing_fields_means_off():
    flags = _flags(entropy_r_cost=0.125)
    util.process_flags(flags)

    protocol = util.validate_voc_resume_protocol({"flags": {}}, flags)

    assert protocol == util.VOC_PROTOCOL_DEFAULTS


def test_voc_resume_protocol_requires_exact_active_mode_and_hyperparameters():
    flags = _flags(
        dynamic_factorized_control=True,
        dynamic_voc_mode="shadow",
        think_cost=0.0005,
    )
    util.process_flags(flags)
    embedded = dict(util.get_voc_protocol(flags))
    checkpoint = {
        "dynamic_voc_mode": "shadow",
        "voc_gate_policy_schema_version": util.VOC_GATE_POLICY_SCHEMA_VERSION,
        "flags": embedded,
    }

    assert util.validate_voc_resume_protocol(checkpoint, flags) == embedded

    checkpoint["dynamic_voc_mode"] = "control"
    with pytest.raises(ValueError, match="top-level dynamic_voc_mode"):
        util.validate_voc_resume_protocol(checkpoint, flags)

    checkpoint["dynamic_voc_mode"] = "shadow"
    checkpoint["flags"]["voc_train_epsilon"] = 0.5
    with pytest.raises(ValueError, match="voc_train_epsilon"):
        util.validate_voc_resume_protocol(checkpoint, flags)


def test_voc_resume_protocol_schema1_missing_beta_is_point9_only():
    flags = _flags(
        dynamic_factorized_control=True,
        dynamic_voc_mode="control",
        think_cost=0.0005,
    )
    util.process_flags(flags)
    embedded = dict(util.get_voc_protocol(flags))
    embedded.pop("voc_gate_adam_beta1")
    embedded.pop("voc_gate_param_align")
    embedded.pop("voc_gate_param_align_coef")
    checkpoint = {
        "dynamic_voc_mode": "control",
        "voc_gate_policy_schema_version": 1,
        "flags": embedded,
    }

    protocol = util.validate_voc_resume_protocol(checkpoint, flags)

    assert protocol["voc_gate_adam_beta1"] == pytest.approx(0.9)
    assert protocol["voc_gate_param_align"] is False
    assert protocol["voc_gate_param_align_coef"] == 1.0
    flags.voc_gate_adam_beta1 = 0.0
    with pytest.raises(ValueError, match="voc_gate_adam_beta1"):
        util.validate_voc_resume_protocol(checkpoint, flags)


@pytest.mark.parametrize("beta1", [0.0, 0.9])
def test_voc_resume_protocol_roundtrips_exact_v2_beta1(beta1):
    flags = _flags(
        dynamic_factorized_control=True,
        dynamic_voc_mode="control",
        think_cost=0.0005,
        voc_gate_adam_beta1=beta1,
    )
    util.process_flags(flags)
    checkpoint = {
        "dynamic_voc_mode": "control",
        "voc_gate_policy_schema_version": 2,
        "flags": dict(util.get_voc_protocol(flags)),
    }
    checkpoint["flags"].pop("voc_gate_param_align")
    checkpoint["flags"].pop("voc_gate_param_align_coef")

    protocol = util.validate_voc_resume_protocol(checkpoint, flags)

    assert protocol["voc_gate_adam_beta1"] == beta1
    assert protocol["voc_gate_param_align"] is False
    assert protocol["voc_gate_param_align_coef"] == 1.0
    flags.voc_gate_adam_beta1 = 0.5 if beta1 != 0.5 else 0.0
    with pytest.raises(ValueError, match="voc_gate_adam_beta1"):
        util.validate_voc_resume_protocol(checkpoint, flags)


def test_voc_resume_protocol_rejects_near_but_not_exact_beta1():
    flags = _flags(
        dynamic_factorized_control=True,
        dynamic_voc_mode="control",
        think_cost=0.0005,
        voc_gate_adam_beta1=0.5,
    )
    util.process_flags(flags)
    checkpoint = {
        "dynamic_voc_mode": "control",
        "voc_gate_policy_schema_version": 2,
        "flags": dict(util.get_voc_protocol(flags)),
    }
    checkpoint["flags"].pop("voc_gate_param_align")
    checkpoint["flags"].pop("voc_gate_param_align_coef")
    checkpoint["flags"]["voc_gate_adam_beta1"] = 0.5000000000000001

    with pytest.raises(ValueError, match="voc_gate_adam_beta1"):
        util.validate_voc_resume_protocol(checkpoint, flags)


@pytest.mark.parametrize("confidence_weighted", [True, False])
def test_voc_resume_protocol_roundtrips_exact_confidence_bool(
    confidence_weighted,
):
    flags = _flags(
        dynamic_factorized_control=True,
        dynamic_voc_mode="control",
        think_cost=0.0005,
        voc_gate_confidence_weighted=confidence_weighted,
    )
    util.process_flags(flags)
    checkpoint = {
        "dynamic_voc_mode": "control",
        "voc_gate_policy_schema_version": util.VOC_GATE_POLICY_SCHEMA_VERSION,
        "flags": dict(util.get_voc_protocol(flags)),
    }

    protocol = util.validate_voc_resume_protocol(checkpoint, flags)

    assert protocol["voc_gate_confidence_weighted"] is confidence_weighted
    checkpoint["flags"]["voc_gate_confidence_weighted"] = (
        not confidence_weighted
    )
    with pytest.raises(ValueError, match="voc_gate_confidence_weighted"):
        util.validate_voc_resume_protocol(checkpoint, flags)


def test_voc_resume_protocol_binds_schema3_alignment_identity():
    flags = _flags(
        dynamic_factorized_control=True,
        dynamic_voc_mode="control",
        think_cost=0.0005,
        voc_gate_param_align=True,
    )
    util.process_flags(flags)
    checkpoint = {
        "dynamic_voc_mode": "control",
        "voc_gate_policy_schema_version": util.VOC_GATE_POLICY_SCHEMA_VERSION,
        "flags": dict(util.get_voc_protocol(flags)),
    }

    protocol = util.validate_voc_resume_protocol(checkpoint, flags)

    assert protocol["voc_gate_param_align"] is True
    assert protocol["voc_gate_param_align_coef"] == 1.0
    flags.voc_gate_param_align = False
    with pytest.raises(ValueError, match="voc_gate_param_align"):
        util.validate_voc_resume_protocol(checkpoint, flags)


def test_voc_resume_protocol_rejects_schema3_nextafter_alignment_coefficient():
    flags = _flags(
        dynamic_factorized_control=True,
        dynamic_voc_mode="control",
        think_cost=0.0005,
        voc_gate_param_align=True,
    )
    util.process_flags(flags)
    checkpoint = {
        "dynamic_voc_mode": "control",
        "voc_gate_policy_schema_version": util.VOC_GATE_POLICY_SCHEMA_VERSION,
        "flags": dict(util.get_voc_protocol(flags)),
    }
    checkpoint["flags"]["voc_gate_param_align_coef"] = math.nextafter(
        1.0, 2.0
    )

    with pytest.raises(ValueError, match="voc_gate_param_align_coef=1.0 exactly"):
        util.validate_voc_resume_protocol(checkpoint, flags)


def test_voc_resume_protocol_rejects_active_checkpoint_without_gate_schema():
    flags = _flags(
        dynamic_factorized_control=True,
        dynamic_voc_mode="shadow",
        think_cost=0.0005,
    )
    util.process_flags(flags)
    checkpoint = {
        "dynamic_voc_mode": "shadow",
        "flags": dict(util.get_voc_protocol(flags)),
    }

    with pytest.raises(ValueError, match="voc_gate_policy_schema_version"):
        util.validate_voc_resume_protocol(checkpoint, flags)


def test_voc_resume_protocol_rejects_missing_active_metadata():
    flags = _flags(
        dynamic_factorized_control=True,
        dynamic_voc_mode="shadow",
        think_cost=0.0005,
    )
    util.process_flags(flags)
    checkpoint = {
        "dynamic_voc_mode": "shadow",
        "flags": {"dynamic_voc_mode": "shadow"},
    }

    with pytest.raises(ValueError, match="lacks embedded voc_loss_cost"):
        util.validate_voc_resume_protocol(checkpoint, flags)


@pytest.mark.parametrize(
    "missing",
    [
        "voc_dueling_q",
        "voc_expected_gate_loss",
        "voc_ema_gate_target",
        "voc_gate_target_tau",
        "voc_dedicated_gate",
        "voc_soft_q_bce_gate",
        "voc_gate_q_temperature",
        "voc_gate_confidence_weighted",
        "voc_gate_adam_beta1",
        "voc_gate_param_align",
        "voc_gate_param_align_coef",
        "voc_gate_learning_rate",
        "voc_gate_grad_norm_clipping",
        "entropy_r_cost",
    ],
)
def test_voc_resume_protocol_requires_active_protocol_fields(missing):
    flags = _flags(
        dynamic_factorized_control=True,
        dynamic_voc_mode="shadow",
        think_cost=0.0005,
    )
    util.process_flags(flags)
    embedded = dict(util.get_voc_protocol(flags))
    embedded.pop(missing)
    checkpoint = {
        "dynamic_voc_mode": "shadow",
        "voc_gate_policy_schema_version": util.VOC_GATE_POLICY_SCHEMA_VERSION,
        "flags": embedded,
    }

    with pytest.raises(ValueError, match=rf"lacks embedded {missing}"):
        util.validate_voc_resume_protocol(checkpoint, flags)


@pytest.mark.parametrize("bad_entropy", [0.01, 1e-13])
def test_voc_resume_protocol_rejects_corrupt_return_anchor(bad_entropy):
    flags = _flags(
        dynamic_factorized_control=True,
        dynamic_voc_mode="shadow",
        think_cost=0.0005,
    )
    util.process_flags(flags)
    embedded = dict(util.get_voc_protocol(flags))
    embedded["entropy_r_cost"] = bad_entropy
    checkpoint = {
        "dynamic_voc_mode": "shadow",
        "flags": embedded,
    }

    with pytest.raises(ValueError, match="entropy_r_cost"):
        util.validate_voc_resume_protocol(checkpoint, flags)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"dynamic_voc_mode": "control"}, "top-level mode disagrees"),
        ({"voc_update_count": 0}, "positive voc_update_count"),
        ({"voc_continue_count": 0}, "positive voc_continue_count"),
        ({"voc_stop_count": 0}, "positive voc_stop_count"),
        ({"voc_holdout_continue_count": 0}, "voc_holdout_continue_count"),
        ({"voc_holdout_stop_count": 0}, "voc_holdout_stop_count"),
        ({"voc_holdout_td_rmse": float("nan")}, "held-out calibration"),
        ({"voc_holdout_td_sum": float("nan")}, "voc_holdout_td_sum"),
        ({"voc_holdout_td_bias": 0.25}, "disagrees with raw"),
        ({"voc_holdout_split_version": 2}, "unsupported.*split_version"),
        ({"voc_holdout_actor_streams": 15}, "disagrees with embedded topology"),
        ({"voc_amp_skip_count": 0.5}, "voc_amp_skip_count"),
        (
            {"voc_grad_scaler_state_dict": {"scale": 256.0}},
            "must not store VoC GradScaler",
        ),
        (
            {
                "voc_holdout_td_sum": 0.0,
                "voc_holdout_td_abs_sum": 100.0,
                "voc_holdout_td_sq_sum": 1.0,
                "voc_holdout_td_bias": 0.0,
                "voc_holdout_td_mae": 10.0,
                "voc_holdout_td_rmse": 10 ** -0.5,
            },
            "squared sum is inconsistent",
        ),
        (
            {
                "voc_holdout_td_sum": 0.0,
                "voc_holdout_td_abs_sum": 0.0,
                "voc_holdout_td_sq_sum": 1.0,
                "voc_holdout_td_bias": 0.0,
                "voc_holdout_td_mae": 0.0,
                "voc_holdout_td_rmse": 10 ** -0.5,
            },
            "squared sum is inconsistent",
        ),
        (
            {"voc_optimizer_state_dict": {"state": {}, "param_groups": []}},
            "VoC optimizer must have one param_group",
        ),
        ({"voc_scheduler_state_dict": {}}, "voc_scheduler_state_dict"),
        ({"actor_net_state_dict": {}}, "lacks voc_head"),
        (
            {"voc_gate_policy_schema_version": 6},
            "voc_gate_policy_schema_version",
        ),
        ({"voc_gate_update_count": 1}, "shadow gate update count"),
        (
            {
                "voc_gate_optimizer_state_dict": {
                    "state": {},
                    "param_groups": [],
                }
            },
            "gate optimizer must have one param_group",
        ),
        (
            {"voc_gate_scheduler_state_dict": {}},
            "voc_gate_scheduler_state_dict",
        ),
        (
            {"voc_gate_grad_scaler_state_dict": {"scale": 256.0}},
            "must not store dedicated gate GradScaler",
        ),
    ],
)
def test_voc_control_preload_rejects_invalid_shadow_parent(
    overrides, message, tmp_path
):
    path = tmp_path / "ckp_actor.tar"
    torch.save(_shadow_checkpoint(**overrides), path)
    with pytest.raises(ValueError, match=message):
        util.validate_voc_control_preload(path)


def test_voc_control_preload_rejects_nonfinite_q_head(tmp_path):
    checkpoint = _shadow_checkpoint()
    checkpoint["actor_net_state_dict"]["voc_head.weight"][0, 0] = float("nan")
    path = tmp_path / "ckp_actor.tar"
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="non-finite weights"):
        util.validate_voc_control_preload(path)


def test_voc_control_preload_rejects_nonfinite_gate_head(tmp_path):
    checkpoint = _shadow_checkpoint()
    checkpoint["actor_net_state_dict"]["voc_gate_head.weight"][0, 0] = (
        float("nan")
    )
    path = tmp_path / "ckp_actor.tar"
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="voc_gate_head contains non-finite"):
        util.validate_voc_control_preload(path)


def test_voc_control_preload_validates_fp16_scaler_provenance(tmp_path):
    checkpoint = _shadow_checkpoint()
    checkpoint["flags"]["float16"] = True
    checkpoint["voc_grad_scaler_state_dict"] = {
        "scale": 128.0,
        "growth_factor": 2.0,
        "backoff_factor": 0.5,
        "growth_interval": 2000,
        "_growth_tracker": 7,
    }
    checkpoint["voc_gate_grad_scaler_state_dict"] = dict(
        checkpoint["voc_grad_scaler_state_dict"]
    )
    path = tmp_path / "ckp_actor.tar"
    torch.save(checkpoint, path)
    state = util.validate_voc_control_preload(path)
    assert state["voc_float16"] is True
    assert state["voc_grad_scaler_state_saved"] is True

    checkpoint["voc_grad_scaler_state_dict"] = None
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="GradScaler state"):
        util.validate_voc_control_preload(path)


@pytest.mark.parametrize(
    ("state_key", "field", "value"),
    [
        ("voc_grad_scaler_state_dict", "growth_interval", 1.5),
        ("voc_grad_scaler_state_dict", "_growth_tracker", 2000),
        ("voc_gate_grad_scaler_state_dict", "growth_interval", 1.5),
        ("voc_gate_grad_scaler_state_dict", "_growth_tracker", 2000),
    ],
)
def test_voc_control_preload_rejects_invalid_scaler_counters(
    state_key, field, value, tmp_path
):
    checkpoint = _shadow_checkpoint()
    checkpoint["flags"]["float16"] = True
    scaler = {
        "scale": 128.0,
        "growth_factor": 2.0,
        "backoff_factor": 0.5,
        "growth_interval": 2000,
        "_growth_tracker": 7,
    }
    checkpoint["voc_grad_scaler_state_dict"] = dict(scaler)
    checkpoint["voc_gate_grad_scaler_state_dict"] = dict(scaler)
    checkpoint[state_key][field] = value
    path = tmp_path / "ckp_actor.tar"
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="GradScaler counters"):
        util.validate_voc_control_preload(path)


@pytest.mark.parametrize(
    ("state_key", "field", "value"),
    [
        ("voc_grad_scaler_state_dict", "growth_factor", 3.0),
        ("voc_grad_scaler_state_dict", "backoff_factor", 0.9),
        ("voc_grad_scaler_state_dict", "growth_interval", 1000),
        ("voc_gate_grad_scaler_state_dict", "growth_factor", 3.0),
        ("voc_gate_grad_scaler_state_dict", "backoff_factor", 0.9),
        ("voc_gate_grad_scaler_state_dict", "growth_interval", 1000),
    ],
)
def test_voc_control_preload_binds_fixed_scaler_protocol(
    state_key, field, value, tmp_path
):
    checkpoint = _shadow_checkpoint()
    checkpoint["flags"]["float16"] = True
    scaler = {
        "scale": 128.0,
        "growth_factor": 2.0,
        "backoff_factor": 0.5,
        "growth_interval": 2000,
        "_growth_tracker": 7,
    }
    checkpoint["voc_grad_scaler_state_dict"] = dict(scaler)
    checkpoint["voc_gate_grad_scaler_state_dict"] = dict(scaler)
    checkpoint[state_key][field] = value
    path = tmp_path / "ckp_actor.tar"
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="GradScaler protocol disagrees"):
        util.validate_voc_control_preload(path)


def test_voc_control_preload_rejects_shadow_protocol_mismatch(tmp_path):
    checkpoint = _shadow_checkpoint()
    checkpoint["flags"]["voc_gate_temperature"] = 0.5
    path = tmp_path / "ckp_actor.tar"
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="voc_gate_temperature"):
        util.validate_voc_control_preload(path)


@pytest.mark.parametrize("bad_entropy", [0.01, 1e-13])
def test_voc_control_preload_rejects_corrupt_return_anchor(
    bad_entropy, tmp_path
):
    checkpoint = _shadow_checkpoint()
    checkpoint["flags"]["entropy_r_cost"] = bad_entropy
    path = tmp_path / "ckp_actor.tar"
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="entropy_r_cost"):
        util.validate_voc_control_preload(path)


@pytest.mark.parametrize(
    "missing",
    [
        "voc_dueling_q",
        "voc_expected_gate_loss",
        "voc_ema_gate_target",
        "voc_gate_target_tau",
        "voc_dedicated_gate",
        "voc_soft_q_bce_gate",
        "voc_gate_q_temperature",
        "voc_gate_confidence_weighted",
        "voc_gate_adam_beta1",
        "voc_gate_param_align",
        "voc_gate_param_align_coef",
        "voc_gate_learning_rate",
        "voc_gate_grad_norm_clipping",
        "entropy_r_cost",
    ],
)
def test_voc_control_preload_requires_active_protocol_fields(missing, tmp_path):
    checkpoint = _shadow_checkpoint()
    checkpoint["flags"].pop(missing)
    path = tmp_path / "ckp_actor.tar"
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match=rf"lacks embedded {missing}"):
        util.validate_voc_control_preload(path)


@pytest.mark.parametrize("bad_value", [1, 0, "true", None])
def test_factorized_control_rejects_non_boolean_values(bad_value):
    with pytest.raises(ValueError, match="must be boolean"):
        util.process_flags(_flags(dynamic_factorized_control=bad_value))


@pytest.mark.parametrize("num_actions", [2, 5, 11])
def test_dynamic_tree_schema_is_budget_independent(num_actions):
    expected_width = 10 * num_actions + 14
    schemas = []
    for max_search_steps, max_depth in [(-1, 5), (8, 40), (40, 100)]:
        flags = _flags(max_search_steps=max_search_steps, max_depth=max_depth)
        util.process_flags(flags)
        schema = util.get_tree_rep_meaning(num_actions, 1, flags)
        schemas.append(schema)
        assert schema["search_start"].stop == expected_width
    assert schemas[0] == schemas[1] == schemas[2]


def test_dynamic_reward_channel_is_appended():
    flags = _flags(im_cost=1.0, cur_cost=1.0)
    assert util.get_reward_names(flags) == ["re", "im", "cur", "think"]


def test_dynamic_budget_stats_use_only_completed_search_stages():
    search_steps = torch.tensor([
        [0, 99, 2],
        [4, 7, 6],
    ])
    stage_end = torch.tensor([
        [True, False, True],
        [True, False, True],
    ])

    stats = util.get_search_budget_stats(search_steps, stage_end)

    # The unfinished 99/7 stages do not contribute.  A zero-step STOP does.
    assert stats["max_budget"] == 6.0
    assert stats["mean_budget"] == 3.0
    assert stats["search/mean_steps"] == stats["mean_budget"]
    assert stats["search/median_steps"] == 2.0
    assert stats["search/p95_steps"] == pytest.approx(5.7)
    assert stats["search/budget_bin_0_count"] == 1
    assert stats["search/budget_bin_0_fraction"] == pytest.approx(0.25)
    assert stats["search/budget_bin_1_count"] == 0
    assert stats["search/budget_bin_1_fraction"] == 0.0
    assert stats["search/budget_bin_2_3_count"] == 1
    assert stats["search/budget_bin_2_3_fraction"] == pytest.approx(0.25)
    assert stats["search/budget_bin_4_7_count"] == 2
    assert stats["search/budget_bin_4_7_fraction"] == pytest.approx(0.5)
    assert stats["search/budget_bin_8_15_count"] == 0
    assert stats["search/budget_bin_16_cap_count"] == 0


def test_dynamic_budget_stats_are_zero_without_completed_stage():
    stats = util.get_search_budget_stats(
        torch.tensor([[8, 13]]), torch.tensor([[False, False]])
    )

    assert stats["max_budget"] == 0.0
    assert stats["mean_budget"] == 0.0
    for label in ("0", "1", "2_3", "4_7", "8_15", "16_cap"):
        assert stats[f"search/budget_bin_{label}_count"] == 0
        assert stats[f"search/budget_bin_{label}_fraction"] == 0.0


def test_dynamic_depth_stop_stats_use_pre_decision_depth_and_stable_empty_bins():
    search_steps = torch.tensor([[0, 1, 1, 3, 4, 8, 16, 99]])
    controls = torch.tensor([[
        util.STOP,
        util.PROCEED,
        util.STOP,
        util.RESET,
        util.STOP,
        util.STOP,
        util.STOP,
        util.PROCEED,
    ]])
    valid = torch.tensor([[True, True, True, True, True, True, True, False]])
    stop_probability = torch.tensor([[
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0,
    ]])

    stats = util.get_search_depth_stop_stats(
        search_steps, controls, valid, stop_probability
    )

    # Accepted PROCEED/RESET rows are reported at their pre-increment depth.
    assert stats["search/depth_bin_0_count"] == 2
    assert stats["search/depth_bin_0_stop_probability"] == pytest.approx(0.15)
    assert stats["search/depth_bin_1_count"] == 1
    assert stats["search/depth_bin_1_stop_probability"] == pytest.approx(0.3)
    assert stats["search/depth_bin_2_3_count"] == 1
    assert stats["search/depth_bin_2_3_stop_probability"] == pytest.approx(0.4)
    assert stats["search/depth_bin_4_7_count"] == 1
    assert stats["search/depth_bin_4_7_stop_probability"] == pytest.approx(0.5)
    assert stats["search/depth_bin_8_15_count"] == 1
    assert stats["search/depth_bin_8_15_stop_probability"] == pytest.approx(0.6)
    assert stats["search/depth_bin_16_plus_count"] == 1
    assert stats["search/depth_bin_16_plus_stop_probability"] == pytest.approx(0.7)
    assert stats["search/depth_stop_probability_count"] == 7
    assert stats["search/depth_stop_probability_slope"] > 0.0

    empty = util.get_search_depth_stop_stats(
        search_steps,
        controls,
        torch.zeros_like(valid),
        stop_probability,
    )
    assert empty["search/depth_stop_probability_count"] == 0
    assert empty["search/depth_stop_probability_slope"] == 0.0
    for label in ("0", "1", "2_3", "4_7", "8_15", "16_plus"):
        assert empty[f"search/depth_bin_{label}_count"] == 0
        assert empty[f"search/depth_bin_{label}_stop_probability"] == 0.0


@pytest.mark.parametrize("bad_cap", [0, -2])
def test_dynamic_rejects_invalid_search_cap(bad_cap):
    with pytest.raises(ValueError):
        util.process_flags(_flags(max_search_steps=bad_cap))


def test_dynamic_requires_reset_mode_zero():
    with pytest.raises(ValueError):
        util.process_flags(_flags(reset_mode=1))


@pytest.mark.parametrize("wrapper_type", [1, 3, 4])
def test_dynamic_rejects_unsupported_wrapper_types(wrapper_type):
    with pytest.raises(ValueError):
        util.process_flags(_flags(wrapper_type=wrapper_type))


def test_dynamic_rejects_mcts_actor():
    with pytest.raises(ValueError):
        util.process_flags_actor(_flags(drc=False, mcts=True))


def test_fixed_mode_keeps_action_sequence_setting():
    flags = _flags(dynamic_search=False, has_action_seq=True)
    util.process_flags(flags)
    assert flags.has_action_seq is True


@pytest.mark.parametrize(
    ("model_setting", "global_setting", "expected"),
    [("inherit", True, True), ("inherit", False, False), ("false", True, False),
     ("true", False, True), (False, True, False), (True, False, True)],
)
def test_model_float16_can_override_or_inherit_actor_precision(
        model_setting, global_setting, expected):
    flags = _flags(
        dynamic_search=False,
        model_float16=model_setting,
        float16=global_setting,
    )
    util.process_flags(flags)
    assert flags.model_float16 is expected


def test_model_float16_rejects_unknown_setting():
    with pytest.raises(ValueError, match="model_float16"):
        util.process_flags(
            _flags(dynamic_search=False, model_float16="sometimes", float16=True)
        )


def test_model_state_projection_legacy_defaults_are_backward_compatible():
    flags = _flags(dynamic_search=False)

    util.process_flags(flags)

    assert flags.model_state_projection == "none"
    assert flags.model_state_range_loss_cost == 0.0


@pytest.mark.parametrize("bad_mode", [None, "sigmoid", "Clamp", True, 1])
def test_model_state_projection_rejects_unknown_mode(bad_mode):
    with pytest.raises(ValueError, match="model_state_projection"):
        util.process_flags(
            _flags(dynamic_search=False, model_state_projection=bad_mode)
        )


@pytest.mark.parametrize("bad_cost", [-0.1, float("nan"), float("inf"), True])
def test_model_state_range_loss_rejects_invalid_cost(bad_cost):
    with pytest.raises(ValueError, match="model_state_range_loss_cost"):
        util.process_flags(
            _flags(
                dynamic_search=False,
                model_state_projection="clamp",
                model_state_range_loss_cost=bad_cost,
            )
        )


def test_model_state_range_loss_requires_projection():
    with pytest.raises(ValueError, match="requires"):
        util.process_flags(
            _flags(
                dynamic_search=False,
                model_state_projection="none",
                model_state_range_loss_cost=1.0,
            )
        )


def test_model_state_projection_rejects_latent_decoder_depth():
    with pytest.raises(ValueError, match="model_decoder_depth=0"):
        util.process_flags(
            _flags(
                dynamic_search=False,
                model_decoder_depth=1,
                model_state_projection="clamp",
            )
        )


def test_schedule_horizon_inherits_total_steps():
    flags = _flags(dynamic_search=False, total_steps=30_000)
    util.process_flags(flags)
    assert flags.schedule_total_steps == 30_000
    assert util.schedule_progress(flags, 15_000) == pytest.approx(0.5)


def test_schedule_horizon_can_outlive_bounded_run():
    flags = _flags(
        dynamic_search=False,
        total_steps=30_000,
        schedule_total_steps=100_000_000,
    )
    util.process_flags(flags)
    assert flags.total_steps == 30_000
    assert flags.schedule_total_steps == 100_000_000
    assert util.schedule_progress(flags, 30_000) == pytest.approx(0.0003)
    assert util.schedule_progress(flags, 200_000_000) == 1.0


@pytest.mark.parametrize("bad_horizon", [0, -2, 1.5, True, "100000000"])
def test_schedule_horizon_rejects_invalid_values(bad_horizon):
    with pytest.raises(ValueError, match="schedule_total_steps"):
        util.process_flags(
            _flags(
                dynamic_search=False,
                total_steps=30_000,
                schedule_total_steps=bad_horizon,
            )
        )


@pytest.mark.parametrize("bad_limit", [0, -1, 1.5, True, "8"])
def test_actor_amp_skip_limit_rejects_invalid_values(bad_limit):
    with pytest.raises(ValueError, match="actor_amp_max_consecutive_skips"):
        util.process_flags(
            _flags(
                dynamic_search=False,
                actor_amp_max_consecutive_skips=bad_limit,
            )
        )


def test_initial_env_out_uses_wrapper_phase_and_masks_over_defaults():
    flags = _flags()
    state = {
        "real_states": torch.zeros(2, 1),
        "phase": torch.tensor([util.WAIT_PHASE, util.SEARCH_PHASE]),
        "tree_token_valid": torch.tensor([False, True]),
    }
    info = {
        "legal_control_mask": torch.tensor(
            [[False, False, False], [True, True, True]]
        ),
        "search_state_reset": torch.tensor([False, True]),
    }

    env_out = util.init_env_out(
        state, info, flags, dim_actions=1, tuple_action=False
    )

    assert env_out.phase.tolist() == [[util.WAIT_PHASE, util.SEARCH_PHASE]]
    assert env_out.tree_token_valid.tolist() == [[False, True]]
    assert env_out.legal_control_mask.tolist() == [
        [[False, False, False], [True, True, True]]
    ]


def test_dynamic_decoder_uses_budget_independent_schema():
    num_actions = 5
    width = 10 * num_actions + 14
    token = torch.arange(width, dtype=torch.float32).view(1, width)

    decoded = util.decode_dynamic_tree_reps(token, num_actions)

    assert decoded["search_start"].item() == width - 1
    assert torch.equal(decoded["cur_reset"], decoded["tree_reset"])
