#!/usr/bin/env python3
"""Fail-closed fixed-checkpoint evaluation for learned Dynamic VoC control.

This evaluator is intentionally separate from training.  It loads a completed
Enduro checkpoint bundle, disables every learner, runs the profile's
epsilon-zero gate execution policy under fixed held-out seeds, and records
enough per-decision evidence to re-pool the four behaviours frozen in
``VOC_V7_200K_ACCEPTANCE.md``.  Schemas 1--4 sample the learned soft gate;
schemas 5--6 execute non-ties by deterministic sign and sample only an exact
zero tie at probability one half.

The EMA Q head is diagnostic only.  It is evaluated *after* ActorNet samples
the control action and is never used to clamp, replace, or reject that sample.
"""

from __future__ import annotations

import argparse
import copy
import csv
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import io
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import stat
import sys
import tempfile
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple
import uuid

import numpy as np
import torch
import yaml


SCHEMA_VERSION = 1
TIE_TOLERANCE = 1e-6
PROCEED = 0
RESET = 1
STOP = 2
GATE_CONTINUE = 0
GATE_STOP = 1

DEFAULT_SEED_BASE = 20_260_827
DEFAULT_NUM_SEEDS = 16
DEFAULT_REAL_STEPS_PER_SEED = 6_250
DEFAULT_CALIBRATION_UNROLL = 201
PREREGISTERED_ENDURO_ROM_SHA256 = (
    "6045c8be78c7d0bec29040022543a8c0b9e3672b50005a94bf0166f0f73be3d9"
)
OUTPUT_LOCK_NAME = ".fixed-checkpoint-evaluation.lock"
V13_PRIMARY_XPID = (
    "enduro-voc-v13-versioned-eps25-seed5-strict-fresh-300k"
)
V13_PRIMARY_STAGE = (
    V13_PRIMARY_XPID,
    5,
    300_000,
    10_000,
    201,
    True,
)
V13_V12_PROJECTION_KEY_COUNT = 209
V13_COMPLETE_IDENTITY_KEY_COUNT = 228
V13_V12_PROJECTION_SHA256 = (
    "bd386ad1c78f030409562dea5c34d8866cc128cf99b7f39938f6a090f5987407"
)
V13_PRIVATE_LOGGER_MARKERS = (
    "voc_actor_policy_logger_finish_request",
    "voc_actor_policy_logger_finish_ack",
)
V14_PRIMARY_XPID = (
    "enduro-voc-v14-sealed-eps25-seed5-strict-fresh-300k"
)
V14_PRIMARY_STAGE = (
    V14_PRIMARY_XPID,
    5,
    300_000,
    10_000,
    201,
    True,
)
V14_V12_PROJECTION_KEY_COUNT = 209
V14_COMPLETE_IDENTITY_KEY_COUNT = 229
V14_V12_PROJECTION_SHA256 = V13_V12_PROJECTION_SHA256
V14_PRIVATE_LOGGER_MARKERS = V13_PRIVATE_LOGGER_MARKERS
V15_PRIMARY_XPID = (
    "enduro-voc-v15-halfsq-eps25-seed5-strict-fresh-300k"
)
V15_PRIMARY_STAGE = (
    V15_PRIMARY_XPID,
    5,
    300_000,
    10_000,
    201,
    True,
)
V15_V12_PROJECTION_KEY_COUNT = 209
V15_COMPLETE_IDENTITY_KEY_COUNT = 229
V15_V12_PROJECTION_SHA256 = V14_V12_PROJECTION_SHA256
V15_Q_REGRESSION_LOSS = "half_squared_td"
V15_PRIVATE_LOGGER_MARKERS = V14_PRIVATE_LOGGER_MARKERS
V16_PRIMARY_XPID = (
    "enduro-voc-v16-commonmode-eps25-seed5-strict-fresh-300k"
)
V16_PRIMARY_STAGE = (
    V16_PRIMARY_XPID,
    5,
    300_000,
    10_000,
    201,
    True,
)
V16_V12_PROJECTION_KEY_COUNT = 209
V16_COMPLETE_IDENTITY_KEY_COUNT = 229
V16_V12_PROJECTION_SHA256 = V15_V12_PROJECTION_SHA256
V16_Q_REGRESSION_LOSS = "half_squared_td"
V16_Q_RECONSTRUCTION = (
    "detached_value_plus_raw_head_mean_plus_policy_centered_raw_head"
)
V16_PRIVATE_LOGGER_MARKERS = V15_PRIVATE_LOGGER_MARKERS
V17_PRIMARY_XPID = (
    "enduro-voc-v17-huber-common-eps25-seed5-strict-fresh-300k"
)
V17_PRIMARY_STAGE = (
    V17_PRIMARY_XPID,
    5,
    300_000,
    10_000,
    201,
    True,
)
V17_V12_PROJECTION_KEY_COUNT = 209
V17_COMPLETE_IDENTITY_KEY_COUNT = 229
V17_V12_PROJECTION_SHA256 = V16_V12_PROJECTION_SHA256
V17_Q_REGRESSION_LOSS = "smooth_l1_beta1"
V17_Q_RECONSTRUCTION = V16_Q_RECONSTRUCTION
V17_PRIVATE_LOGGER_MARKERS = V16_PRIVATE_LOGGER_MARKERS
ATOMIC_ACTOR_POLICY_EVIDENCE_FIELDS = frozenset(
    {
        "actor_amp_consecutive_skips",
        "actor_amp_growth_tracker",
        "actor_amp_init_scale",
        "actor_amp_scale",
        "actor_amp_skip_count",
        "voc_actor_policy_barrier_timeout_count",
        "voc_actor_policy_bundle_summary",
        "voc_actor_policy_expected_ack_count",
        "voc_actor_policy_final_publication_event",
        "voc_actor_policy_malformed_bundle_count",
        "voc_actor_policy_publication_count",
        "voc_actor_policy_publication_event_count",
        "voc_actor_policy_publication_history",
        "voc_actor_policy_publication_history_sha256",
        "voc_actor_policy_state_sha256",
        "voc_actor_policy_terminal",
        "voc_actor_policy_terminal_ack_count",
        "voc_actor_policy_version",
        "voc_actor_policy_version_mismatch_count",
    }
)
V18_PRIMARY_XPID = (
    "enduro-voc-v18-orthocd-adam-eps25-seed5-strict-fresh-300k"
)
V18_PRIMARY_STAGE = (
    V18_PRIMARY_XPID,
    5,
    300_000,
    10_000,
    201,
    True,
)
V18_V12_PROJECTION_KEY_COUNT = 209
V18_COMPLETE_IDENTITY_KEY_COUNT = 229
V18_V12_PROJECTION_SHA256 = V17_V12_PROJECTION_SHA256
V18_Q_REGRESSION_LOSS = V17_Q_REGRESSION_LOSS
V18_Q_RECONSTRUCTION = V17_Q_RECONSTRUCTION
V18_Q_OPTIMIZER_COORDINATES = "orthonormal_common_difference_adam"
V18_PRIVATE_LOGGER_MARKERS = V17_PRIVATE_LOGGER_MARKERS
V19_PRIMARY_XPID = (
    "enduro-voc-v19-tau1-orthocd-adam-eps25-seed5-strict-fresh-300k"
)
V19_PRIMARY_STAGE = (
    V19_PRIMARY_XPID,
    5,
    300_000,
    10_000,
    201,
    True,
)
V19_V12_PROJECTION_KEY_COUNT = 209
V19_COMPLETE_IDENTITY_KEY_COUNT = 229
V19_V12_PROJECTION_SHA256 = (
    "ad22b91fdd06a30ac7f53c0135b32fac2530687c3c36dad5dccf06d700f83f82"
)
V19_Q_REGRESSION_LOSS = V18_Q_REGRESSION_LOSS
V19_Q_RECONSTRUCTION = V18_Q_RECONSTRUCTION
V19_Q_OPTIMIZER_COORDINATES = V18_Q_OPTIMIZER_COORDINATES
V19_PRIVATE_LOGGER_MARKERS = V18_PRIVATE_LOGGER_MARKERS
V20_PRIMARY_XPID = (
    "enduro-voc-v20-telemetry-tau1-orthocd-adam-eps25-seed5-strict-fresh-300k"
)
V20_PRIMARY_STAGE = (
    V20_PRIMARY_XPID,
    5,
    300_000,
    10_000,
    201,
    True,
)
V20_V12_PROJECTION_KEY_COUNT = 209
V20_COMPLETE_IDENTITY_KEY_COUNT = 229
V20_V12_PROJECTION_SHA256 = V19_V12_PROJECTION_SHA256
V20_Q_REGRESSION_LOSS = V19_Q_REGRESSION_LOSS
V20_Q_RECONSTRUCTION = V19_Q_RECONSTRUCTION
V20_Q_OPTIMIZER_COORDINATES = V19_Q_OPTIMIZER_COORDINATES
V20_PRIVATE_LOGGER_MARKERS = V19_PRIVATE_LOGGER_MARKERS


def _fixed_schema13_xpid_claims_intent(value: Any) -> bool:
    """Classify forward V20 lexical intent without normalizing validity."""

    try:
        lexical_value = os.fspath(value) if isinstance(value, os.PathLike) else value
        if isinstance(
            lexical_value,
            (bytes, bytearray, memoryview, np.bytes_),
        ):
            lexical_value = bytes(lexical_value).decode("utf-8")
        else:
            lexical_value = str(lexical_value)
    except (TypeError, UnicodeError) as error:
        raise ValueError(
            "fixed schema-13 xpid intent could not be classified before "
            "downstream I/O"
        ) from error
    return lexical_value.strip().startswith(
        "enduro-voc-v20-telemetry-tau1-orthocd-adam-eps25-"
    )


class ConfirmationProfile(str, Enum):
    """Closed set of preregistered training-horizon confirmations."""

    V7_200K = "v7-200k"
    V10_300K = "v10-300k"
    V11_300K = "v11-300k"
    V12_300K = "v12-300k"
    V13_300K = "v13-300k"
    V14_300K = "v14-300k"
    V15_300K = "v15-300k"
    V16_300K = "v16-300k"
    V17_300K = "v17-300k"
    V18_300K = "v18-300k"
    V19_300K = "v19-300k"
    V20_300K = "v20-300k"


@dataclass(frozen=True)
class ConfirmationProfileSpec:
    total_steps: int
    evaluation_mode: str
    xpid: Optional[str] = None
    base_seed: Optional[int] = None
    schedule_total_steps: Optional[int] = None
    model_warm_up_n: Optional[int] = None
    actor_unroll_len: Optional[int] = None
    use_wandb: Optional[bool] = None
    dynamic_voc_mode: Optional[str] = None
    voc_dedicated_gate: Optional[bool] = None
    voc_soft_q_bce_gate: Optional[bool] = None
    voc_eval_stochastic: Optional[bool] = None
    voc_gate_temperature: Optional[float] = None
    voc_gate_q_temperature: Optional[float] = None
    voc_train_epsilon: Optional[float] = None
    voc_gate_param_align: Optional[bool] = None
    voc_gate_param_align_coef: Optional[float] = None
    voc_gate_exact_projection: Optional[bool] = None
    voc_gate_epsilon_greedy_execution: Optional[bool] = None
    voc_gate_execution_epsilon: Optional[float] = None
    voc_gate_target_tau: Optional[float] = None
    voc_actor_policy_version_barrier: Optional[bool] = None
    voc_actor_policy_bundle_schema_version: Optional[int] = None
    voc_actor_policy_barrier_timeout_s: Optional[float] = None
    voc_actor_policy_ray_max_restarts: Optional[int] = None
    voc_actor_policy_ray_max_task_retries: Optional[int] = None
    voc_actor_policy_barrier_runtime: Optional[bool] = None
    voc_model_input_seal_schema_version: Optional[int] = None
    actor_amp_init_scale: Optional[float] = None
    float16: Optional[bool] = None
    model_float16: Optional[bool] = None
    parallel_actor: Optional[bool] = None
    ppo_k: Optional[int] = None
    self_play_n: Optional[int] = None
    env_n: Optional[int] = None
    actor_batch_size: Optional[int] = None
    ckp: Optional[bool] = None
    preload: Optional[str] = None
    preload_actor: Optional[str] = None
    voc_parent_checkpoint: Optional[str] = None
    voc_gate_policy_schema_version: Optional[int] = None


CONFIRMATION_PROFILE_SPECS = MappingProxyType(
    {
        ConfirmationProfile.V7_200K: ConfirmationProfileSpec(
            total_steps=200_000,
            evaluation_mode="fixed_200k_confirmation",
        ),
        ConfirmationProfile.V10_300K: ConfirmationProfileSpec(
            total_steps=300_000,
            evaluation_mode="fixed_300k_confirmation",
            base_seed=2,
            schedule_total_steps=100_000_000,
            voc_gate_param_align=True,
            voc_gate_param_align_coef=1.0,
            voc_gate_policy_schema_version=3,
        ),
        ConfirmationProfile.V11_300K: ConfirmationProfileSpec(
            total_steps=300_000,
            evaluation_mode="fixed_v11_300k_confirmation",
            base_seed=3,
            schedule_total_steps=100_000_000,
            dynamic_voc_mode="control",
            voc_dedicated_gate=True,
            voc_soft_q_bce_gate=True,
            voc_gate_temperature=1.0,
            voc_gate_q_temperature=0.05,
            voc_gate_param_align=False,
            voc_gate_param_align_coef=1.0,
            voc_gate_exact_projection=True,
            ckp=False,
            preload="",
            preload_actor="",
            voc_parent_checkpoint="",
            voc_gate_policy_schema_version=4,
        ),
        ConfirmationProfile.V12_300K: ConfirmationProfileSpec(
            total_steps=300_000,
            evaluation_mode="fixed_v12_300k_confirmation",
            base_seed=4,
            schedule_total_steps=100_000_000,
            dynamic_voc_mode="control",
            voc_dedicated_gate=True,
            voc_soft_q_bce_gate=True,
            voc_eval_stochastic=True,
            voc_gate_temperature=1.0,
            voc_gate_q_temperature=0.05,
            voc_train_epsilon=0.02,
            voc_gate_param_align=False,
            voc_gate_param_align_coef=1.0,
            voc_gate_exact_projection=True,
            voc_gate_epsilon_greedy_execution=True,
            ckp=False,
            preload="",
            preload_actor="",
            voc_parent_checkpoint="",
            voc_gate_policy_schema_version=5,
        ),
        ConfirmationProfile.V13_300K: ConfirmationProfileSpec(
            total_steps=300_000,
            evaluation_mode="fixed_v13_300k_confirmation",
            xpid=V13_PRIMARY_XPID,
            base_seed=5,
            schedule_total_steps=100_000_000,
            model_warm_up_n=10_000,
            actor_unroll_len=201,
            use_wandb=True,
            dynamic_voc_mode="control",
            voc_dedicated_gate=True,
            voc_soft_q_bce_gate=True,
            voc_eval_stochastic=True,
            voc_gate_temperature=1.0,
            voc_gate_q_temperature=0.05,
            voc_train_epsilon=0.02,
            voc_gate_param_align=False,
            voc_gate_param_align_coef=1.0,
            voc_gate_exact_projection=True,
            voc_gate_epsilon_greedy_execution=True,
            voc_gate_execution_epsilon=0.25,
            voc_actor_policy_version_barrier=True,
            voc_actor_policy_bundle_schema_version=1,
            voc_actor_policy_barrier_timeout_s=120.0,
            voc_actor_policy_ray_max_restarts=0,
            voc_actor_policy_ray_max_task_retries=0,
            voc_actor_policy_barrier_runtime=True,
            actor_amp_init_scale=32.0,
            float16=True,
            model_float16=False,
            parallel_actor=True,
            ppo_k=1,
            self_play_n=1,
            env_n=16,
            actor_batch_size=16,
            ckp=False,
            preload="",
            preload_actor="",
            voc_parent_checkpoint="",
            voc_gate_policy_schema_version=6,
        ),
        ConfirmationProfile.V14_300K: ConfirmationProfileSpec(
            total_steps=300_000,
            evaluation_mode="fixed_v14_300k_confirmation",
            xpid=V14_PRIMARY_XPID,
            base_seed=5,
            schedule_total_steps=100_000_000,
            model_warm_up_n=10_000,
            actor_unroll_len=201,
            use_wandb=True,
            dynamic_voc_mode="control",
            voc_dedicated_gate=True,
            voc_soft_q_bce_gate=True,
            voc_eval_stochastic=True,
            voc_gate_temperature=1.0,
            voc_gate_q_temperature=0.05,
            voc_train_epsilon=0.02,
            voc_gate_param_align=False,
            voc_gate_param_align_coef=1.0,
            voc_gate_exact_projection=True,
            voc_gate_epsilon_greedy_execution=True,
            voc_gate_execution_epsilon=0.25,
            voc_actor_policy_version_barrier=True,
            voc_actor_policy_bundle_schema_version=1,
            voc_actor_policy_barrier_timeout_s=120.0,
            voc_actor_policy_ray_max_restarts=0,
            voc_actor_policy_ray_max_task_retries=0,
            voc_actor_policy_barrier_runtime=True,
            voc_model_input_seal_schema_version=1,
            actor_amp_init_scale=32.0,
            float16=True,
            model_float16=False,
            parallel_actor=True,
            ppo_k=1,
            self_play_n=1,
            env_n=16,
            actor_batch_size=16,
            ckp=False,
            preload="",
            preload_actor="",
            voc_parent_checkpoint="",
            voc_gate_policy_schema_version=7,
        ),
        ConfirmationProfile.V15_300K: ConfirmationProfileSpec(
            total_steps=300_000,
            evaluation_mode="fixed_v15_300k_confirmation",
            xpid=V15_PRIMARY_XPID,
            base_seed=5,
            schedule_total_steps=100_000_000,
            model_warm_up_n=10_000,
            actor_unroll_len=201,
            use_wandb=True,
            dynamic_voc_mode="control",
            voc_dedicated_gate=True,
            voc_soft_q_bce_gate=True,
            voc_eval_stochastic=True,
            voc_gate_temperature=1.0,
            voc_gate_q_temperature=0.05,
            voc_train_epsilon=0.02,
            voc_gate_param_align=False,
            voc_gate_param_align_coef=1.0,
            voc_gate_exact_projection=True,
            voc_gate_epsilon_greedy_execution=True,
            voc_gate_execution_epsilon=0.25,
            voc_actor_policy_version_barrier=True,
            voc_actor_policy_bundle_schema_version=1,
            voc_actor_policy_barrier_timeout_s=120.0,
            voc_actor_policy_ray_max_restarts=0,
            voc_actor_policy_ray_max_task_retries=0,
            voc_actor_policy_barrier_runtime=True,
            voc_model_input_seal_schema_version=1,
            actor_amp_init_scale=32.0,
            float16=True,
            model_float16=False,
            parallel_actor=True,
            ppo_k=1,
            self_play_n=1,
            env_n=16,
            actor_batch_size=16,
            ckp=False,
            preload="",
            preload_actor="",
            voc_parent_checkpoint="",
            voc_gate_policy_schema_version=8,
        ),
        ConfirmationProfile.V16_300K: ConfirmationProfileSpec(
            total_steps=300_000,
            evaluation_mode="fixed_v16_300k_confirmation",
            xpid=V16_PRIMARY_XPID,
            base_seed=5,
            schedule_total_steps=100_000_000,
            model_warm_up_n=10_000,
            actor_unroll_len=201,
            use_wandb=True,
            dynamic_voc_mode="control",
            voc_dedicated_gate=True,
            voc_soft_q_bce_gate=True,
            voc_eval_stochastic=True,
            voc_gate_temperature=1.0,
            voc_gate_q_temperature=0.05,
            voc_train_epsilon=0.02,
            voc_gate_param_align=False,
            voc_gate_param_align_coef=1.0,
            voc_gate_exact_projection=True,
            voc_gate_epsilon_greedy_execution=True,
            voc_gate_execution_epsilon=0.25,
            voc_actor_policy_version_barrier=True,
            voc_actor_policy_bundle_schema_version=1,
            voc_actor_policy_barrier_timeout_s=120.0,
            voc_actor_policy_ray_max_restarts=0,
            voc_actor_policy_ray_max_task_retries=0,
            voc_actor_policy_barrier_runtime=True,
            voc_model_input_seal_schema_version=1,
            actor_amp_init_scale=32.0,
            float16=True,
            model_float16=False,
            parallel_actor=True,
            ppo_k=1,
            self_play_n=1,
            env_n=16,
            actor_batch_size=16,
            ckp=False,
            preload="",
            preload_actor="",
            voc_parent_checkpoint="",
            voc_gate_policy_schema_version=9,
        ),
        ConfirmationProfile.V17_300K: ConfirmationProfileSpec(
            total_steps=300_000,
            evaluation_mode="fixed_v17_300k_confirmation",
            xpid=V17_PRIMARY_XPID,
            base_seed=5,
            schedule_total_steps=100_000_000,
            model_warm_up_n=10_000,
            actor_unroll_len=201,
            use_wandb=True,
            dynamic_voc_mode="control",
            voc_dedicated_gate=True,
            voc_soft_q_bce_gate=True,
            voc_eval_stochastic=True,
            voc_gate_temperature=1.0,
            voc_gate_q_temperature=0.05,
            voc_train_epsilon=0.02,
            voc_gate_param_align=False,
            voc_gate_param_align_coef=1.0,
            voc_gate_exact_projection=True,
            voc_gate_epsilon_greedy_execution=True,
            voc_gate_execution_epsilon=0.25,
            voc_actor_policy_version_barrier=True,
            voc_actor_policy_bundle_schema_version=1,
            voc_actor_policy_barrier_timeout_s=120.0,
            voc_actor_policy_ray_max_restarts=0,
            voc_actor_policy_ray_max_task_retries=0,
            voc_actor_policy_barrier_runtime=True,
            voc_model_input_seal_schema_version=1,
            actor_amp_init_scale=32.0,
            float16=True,
            model_float16=False,
            parallel_actor=True,
            ppo_k=1,
            self_play_n=1,
            env_n=16,
            actor_batch_size=16,
            ckp=False,
            preload="",
            preload_actor="",
            voc_parent_checkpoint="",
            voc_gate_policy_schema_version=10,
        ),
        ConfirmationProfile.V18_300K: ConfirmationProfileSpec(
            total_steps=300_000,
            evaluation_mode="fixed_v18_300k_confirmation",
            xpid=V18_PRIMARY_XPID,
            base_seed=5,
            schedule_total_steps=100_000_000,
            model_warm_up_n=10_000,
            actor_unroll_len=201,
            use_wandb=True,
            dynamic_voc_mode="control",
            voc_dedicated_gate=True,
            voc_soft_q_bce_gate=True,
            voc_eval_stochastic=True,
            voc_gate_temperature=1.0,
            voc_gate_q_temperature=0.05,
            voc_train_epsilon=0.02,
            voc_gate_param_align=False,
            voc_gate_param_align_coef=1.0,
            voc_gate_exact_projection=True,
            voc_gate_epsilon_greedy_execution=True,
            voc_gate_execution_epsilon=0.25,
            voc_actor_policy_version_barrier=True,
            voc_actor_policy_bundle_schema_version=1,
            voc_actor_policy_barrier_timeout_s=120.0,
            voc_actor_policy_ray_max_restarts=0,
            voc_actor_policy_ray_max_task_retries=0,
            voc_actor_policy_barrier_runtime=True,
            voc_model_input_seal_schema_version=1,
            actor_amp_init_scale=32.0,
            float16=True,
            model_float16=False,
            parallel_actor=True,
            ppo_k=1,
            self_play_n=1,
            env_n=16,
            actor_batch_size=16,
            ckp=False,
            preload="",
            preload_actor="",
            voc_parent_checkpoint="",
            voc_gate_policy_schema_version=11,
        ),
        ConfirmationProfile.V19_300K: ConfirmationProfileSpec(
            total_steps=300_000,
            evaluation_mode="fixed_v19_300k_confirmation",
            xpid=V19_PRIMARY_XPID,
            base_seed=5,
            schedule_total_steps=100_000_000,
            model_warm_up_n=10_000,
            actor_unroll_len=201,
            use_wandb=True,
            dynamic_voc_mode="control",
            voc_dedicated_gate=True,
            voc_soft_q_bce_gate=True,
            voc_eval_stochastic=True,
            voc_gate_temperature=1.0,
            voc_gate_q_temperature=0.05,
            voc_train_epsilon=0.02,
            voc_gate_param_align=False,
            voc_gate_param_align_coef=1.0,
            voc_gate_exact_projection=True,
            voc_gate_epsilon_greedy_execution=True,
            voc_gate_execution_epsilon=0.25,
            voc_gate_target_tau=1.0,
            voc_actor_policy_version_barrier=True,
            voc_actor_policy_bundle_schema_version=1,
            voc_actor_policy_barrier_timeout_s=120.0,
            voc_actor_policy_ray_max_restarts=0,
            voc_actor_policy_ray_max_task_retries=0,
            voc_actor_policy_barrier_runtime=True,
            voc_model_input_seal_schema_version=1,
            actor_amp_init_scale=32.0,
            float16=True,
            model_float16=False,
            parallel_actor=True,
            ppo_k=1,
            self_play_n=1,
            env_n=16,
            actor_batch_size=16,
            ckp=False,
            preload="",
            preload_actor="",
            voc_parent_checkpoint="",
            voc_gate_policy_schema_version=12,
        ),
        ConfirmationProfile.V20_300K: ConfirmationProfileSpec(
            total_steps=300_000,
            evaluation_mode="fixed_v20_300k_confirmation",
            xpid=V20_PRIMARY_XPID,
            base_seed=5,
            schedule_total_steps=100_000_000,
            model_warm_up_n=10_000,
            actor_unroll_len=201,
            use_wandb=True,
            dynamic_voc_mode="control",
            voc_dedicated_gate=True,
            voc_soft_q_bce_gate=True,
            voc_eval_stochastic=True,
            voc_gate_temperature=1.0,
            voc_gate_q_temperature=0.05,
            voc_train_epsilon=0.02,
            voc_gate_param_align=False,
            voc_gate_param_align_coef=1.0,
            voc_gate_exact_projection=True,
            voc_gate_epsilon_greedy_execution=True,
            voc_gate_execution_epsilon=0.25,
            voc_gate_target_tau=1.0,
            voc_actor_policy_version_barrier=True,
            voc_actor_policy_bundle_schema_version=1,
            voc_actor_policy_barrier_timeout_s=120.0,
            voc_actor_policy_ray_max_restarts=0,
            voc_actor_policy_ray_max_task_retries=0,
            voc_actor_policy_barrier_runtime=True,
            voc_model_input_seal_schema_version=1,
            actor_amp_init_scale=32.0,
            float16=True,
            model_float16=False,
            parallel_actor=True,
            ppo_k=1,
            self_play_n=1,
            env_n=16,
            actor_batch_size=16,
            ckp=False,
            preload="",
            preload_actor="",
            voc_parent_checkpoint="",
            voc_gate_policy_schema_version=13,
        ),
    }
)
DEFAULT_CONFIRMATION_PROFILE = ConfirmationProfile.V7_200K

REQUIRED_CHECKPOINT_FILES = (
    "config_c.yaml",
    "ckp_actor.tar",
    "ckp_model.tar",
)
SCHEMA13_REQUIRED_CHECKPOINT_FILES = (
    *REQUIRED_CHECKPOINT_FILES,
    "voc_telemetry_manifest.json",
)

DECISION_CSV_FIELDS = (
    "stream_id",
    "environment_seed",
    "episode_index",
    "augmented_step",
    "real_step_before",
    "decision_depth",
    "predecision_last_control",
    "sampled_control",
    "gate_action",
    "continue_probability",
    "stop_probability",
    "ema_q_continue",
    "ema_q_stop",
    "ema_delta_q",
    "online_q_continue",
    "online_q_stop",
    "online_delta_q",
    "ema_selected_q",
    "online_selected_q",
    "state_value",
    "task_reward",
    "think_reward",
    "task_discount",
    "think_discount",
    "real_transition",
    "stage_end",
    "forced_stop",
    "done",
    "truncated",
    "calibration_task_target",
    "calibration_think_target",
    "calibration_net_target",
    "ema_td_error",
    "online_td_error",
)


@dataclass(frozen=True)
class BundleValidation:
    checkpoint_dir: Path
    source_root: Path
    marker: Mapping[str, Any]
    file_hashes: Mapping[str, str]
    source_manifest: Mapping[str, Any]


@dataclass
class OutputGenerationLock:
    path: Path
    generation_id: str
    publication_started: bool = False
    committed: bool = False


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def attest_regular_file(path: str | Path, *, label: str) -> Dict[str, Any]:
    """Hash one non-symlink file and freeze its filesystem identity."""

    path = Path(path).expanduser().resolve()
    _require_regular_file(path, label=label)
    before = path.lstat()
    digest = sha256_file(path)
    after = path.lstat()
    identity_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(getattr(before, name) != getattr(after, name) for name in identity_fields):
        raise RuntimeError(f"{label} changed while it was being attested: {path}")
    mode = stat.S_IMODE(after.st_mode)
    return {
        "path": str(path),
        "sha256": digest,
        "size": int(after.st_size),
        "mode": f"{mode:04o}",
        "writable": bool(mode & 0o222),
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
        "mtime_ns": int(after.st_mtime_ns),
        "ctime_ns": int(after.st_ctime_ns),
    }


def require_attestation_unchanged(
    before: Mapping[str, Any], after: Mapping[str, Any], *, label: str
) -> None:
    if dict(before) != dict(after):
        raise RuntimeError(f"{label} changed during fixed evaluation")


def _safe_relative_path(value: Any, *, label: str) -> Path:
    path = Path(str(value))
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"unsafe {label} path: {value!r}")
    return path


def _require_regular_file(path: Path, *, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise FileNotFoundError(f"missing {label}: {path}") from error
    if path.is_symlink() or not stat.S_ISREG(mode):
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")


_STABLE_FILE_IDENTITY_FIELDS = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_nlink",
    "st_uid",
    "st_gid",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)


def _stable_file_identity(file_stat: os.stat_result) -> Tuple[int, ...]:
    return tuple(
        int(getattr(file_stat, field))
        for field in _STABLE_FILE_IDENTITY_FIELDS
    )


def _read_stable_single_link_bytes(path: Path, *, label: str) -> bytes:
    """Read one regular file through a stable, single-link filesystem identity."""

    path = Path(path)
    try:
        path_before = os.lstat(path)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"missing {label}: {path}") from error
    if not stat.S_ISREG(path_before.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    if path_before.st_nlink != 1:
        raise ValueError(f"{label} must have exactly one hard link: {path}")

    open_flags = os.O_RDONLY
    open_flags |= getattr(os, "O_CLOEXEC", 0)
    open_flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, open_flags)
    except OSError as error:
        raise ValueError(f"could not securely open {label}: {path}") from error
    try:
        descriptor_before = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_before.st_mode):
            raise ValueError(f"{label} descriptor is not a regular file: {path}")
        if descriptor_before.st_nlink != 1:
            raise ValueError(f"{label} must have exactly one hard link: {path}")
        if _stable_file_identity(path_before) != _stable_file_identity(
            descriptor_before
        ):
            raise RuntimeError(f"{label} changed while it was opened: {path}")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        descriptor_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        path_after = os.lstat(path)
    except FileNotFoundError as error:
        raise RuntimeError(f"{label} disappeared while it was read: {path}") from error
    identities = {
        _stable_file_identity(path_before),
        _stable_file_identity(descriptor_before),
        _stable_file_identity(descriptor_after),
        _stable_file_identity(path_after),
    }
    if len(identities) != 1:
        raise RuntimeError(f"{label} changed while it was read: {path}")
    payload = b"".join(chunks)
    if len(payload) != descriptor_after.st_size:
        raise RuntimeError(f"{label} size changed while it was read: {path}")
    return payload


def _load_checkpoint_from_bound_bytes(
    checkpoint_dir: Path,
    filename: str,
    checkpoint_files: Mapping[str, Any],
    *,
    label: str,
) -> Mapping[str, Any]:
    """Load the same stable checkpoint generation named by the bundle marker."""

    if not isinstance(checkpoint_files, Mapping) or filename not in checkpoint_files:
        raise ValueError(f"{label} is not bound by the completion marker")
    record = checkpoint_files[filename]
    if not isinstance(record, Mapping) or set(record) != {"sha256", "size"}:
        raise ValueError(f"{label} has an invalid completion record")
    expected_digest = record["sha256"]
    expected_size = record["size"]
    if (
        type(expected_digest) is not str
        or re.fullmatch(r"[0-9a-f]{64}", expected_digest) is None
    ):
        raise ValueError(f"{label} has an invalid completion digest")
    if type(expected_size) is not int or expected_size <= 0:
        raise ValueError(f"{label} has an invalid completion size")
    payload = _read_stable_single_link_bytes(
        Path(checkpoint_dir) / filename,
        label=label,
    )
    if len(payload) != expected_size:
        raise RuntimeError(f"{label} size disagrees with the completion marker")
    if hashlib.sha256(payload).hexdigest() != expected_digest:
        raise RuntimeError(f"{label} digest disagrees with the completion marker")
    checkpoint = torch.load(
        io.BytesIO(payload), map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"{label} must deserialize to a mapping")
    return checkpoint


def _load_fixed_runtime_checkpoint(
    checkpoint_dir: Path,
    filename: str,
    checkpoint_files: Mapping[str, Any],
    *,
    v20: bool,
    label: str,
) -> Any:
    """Use inherited pathname loads except for the V20 same-byte contract."""

    if v20:
        return _load_checkpoint_from_bound_bytes(
            checkpoint_dir,
            filename,
            checkpoint_files,
            label=label,
        )
    return torch.load(
        Path(checkpoint_dir) / filename,
        map_location="cpu",
        weights_only=False,
    )


def _load_flags_from_bound_config_bytes(
    checkpoint_eval: Any,
    checkpoint_dir: Path,
    config_payload: bytes,
    expected_sha256: str,
    *,
    byte_loader: Optional[Any],
) -> Any:
    """Load flags from bundle-bound bytes, including frozen legacy modules."""

    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    if type(config_payload) is not bytes:
        raise TypeError("fixed checkpoint config payload must be exact bytes")
    if (
        type(expected_sha256) is not str
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
        or hashlib.sha256(config_payload).hexdigest() != expected_sha256
    ):
        raise ValueError("fixed checkpoint config digest disagrees")
    if byte_loader is not None:
        return byte_loader(checkpoint_dir, config_payload, expected_sha256)

    legacy_loader = getattr(checkpoint_eval, "_load_flags", None)
    if legacy_loader is None:
        raise RuntimeError("bound checkpoint evaluator lacks a flag loader")
    with tempfile.TemporaryDirectory(
        prefix="voc-fixed-bound-config-"
    ) as temp_name:
        private_root = Path(temp_name).resolve()
        private_checkpoint = private_root / "checkpoint"
        private_checkpoint.mkdir(mode=0o700)
        private_config = private_checkpoint / "config_c.yaml"
        descriptor = os.open(
            private_config,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = -1
                handle.write(config_payload)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        flags = legacy_loader(private_checkpoint)
        flags.ckpdir = str(checkpoint_dir)
        private_prefix = str(private_root)
        leaked = sorted(
            name
            for name, value in vars(flags).items()
            if type(value) is str and private_prefix in value
        )
        if leaked:
            raise RuntimeError(
                "frozen checkpoint flag loader leaked its private config path: "
                + ", ".join(leaked)
            )
        return flags


def _reject_duplicate_json_pairs(
    pairs: Sequence[Tuple[str, Any]],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _load_strict_json_object(payload: bytes, *, label: str) -> Dict[str, Any]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not valid UTF-8") from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_pairs,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid {label} JSON") from error
    if type(value) is not dict:
        raise ValueError(f"{label} JSON must contain an object")
    return value


def _require_strict_checkpoint_file_records(
    value: Any,
    *,
    label: str,
    expected_names: Sequence[str] = REQUIRED_CHECKPOINT_FILES,
) -> Dict[str, Mapping[str, Any]]:
    expected_names = tuple(expected_names)
    if type(value) is not dict or set(value) != set(expected_names):
        raise ValueError(f"{label} checkpoint file names disagree with exact bundle")
    for name in expected_names:
        record = value[name]
        if type(record) is not dict or set(record) != {"sha256", "size"}:
            raise ValueError(f"{label} has invalid record fields for {name}")
        digest = record["sha256"]
        if type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"{label} has invalid SHA-256 for {name}")
        size = record["size"]
        if type(size) is not int or size <= 0:
            raise ValueError(f"{label} has invalid size for {name}")
    return value


def _require_strict_sha256_file_records(
    value: Any, *, label: str
) -> Dict[str, Mapping[str, Any]]:
    if type(value) is not dict or not value:
        raise ValueError(f"{label} must contain nonempty file records")
    for name, record in value.items():
        if type(name) is not str or not name:
            raise ValueError(f"{label} has an invalid path")
        if type(record) is not dict or set(record) != {"sha256"}:
            raise ValueError(f"{label} has invalid record fields for {name!r}")
        digest = record["sha256"]
        if type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"{label} has invalid SHA-256 for {name!r}")
    return value


def _source_root_matches(marker: Mapping[str, Any], root: Path) -> bool:
    records = marker.get("implementation_sources")
    if not isinstance(records, Mapping) or not records:
        return False
    try:
        for relative, record in records.items():
            relative_path = _safe_relative_path(relative, label="implementation")
            path = (root / relative_path).resolve()
            if root != path and root not in path.parents:
                return False
            if (
                not path.is_file()
                or not isinstance(record, Mapping)
                or sha256_file(path) != record.get("sha256")
            ):
                return False
    except (OSError, TypeError, ValueError):
        return False
    return True


def resolve_training_source_root(
    checkpoint_dir: str | Path,
    marker: Mapping[str, Any],
    explicit: Optional[str | Path] = None,
) -> Path:
    """Find the exact source tree hashed by the run completion marker."""

    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    candidates: List[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit).expanduser().resolve())
    for ancestor in (checkpoint_dir, *checkpoint_dir.parents):
        candidates.extend((ancestor / "src" / "thinker", ancestor / "thinker"))
    candidates.append(Path(__file__).resolve().parent)

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if _source_root_matches(marker, candidate):
            return candidate
    if explicit is not None:
        raise ValueError(
            "--training-source-root does not match the implementation hashes "
            "in the completion marker"
        )
    raise FileNotFoundError(
        "could not locate the source snapshot bound by finish; pass "
        "--training-source-root pointing to the directory containing train.py "
        "and thinker/"
    )


def validate_full_source_manifest(
    source_root: str | Path,
    *,
    source_manifest: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Validate the complete immutable snapshot tree, not only finish inputs.

    Snapshot creation hashes every regular file below ``SNAPSHOT/src`` into
    ``SNAPSHOT/source.sha256``.  Exact path-set comparison also catches an
    added importable module.  Any symlink, special node, or group/owner/world
    write bit below the source root is rejected before execution.
    """

    project_root = Path(source_root).expanduser().resolve()
    snapshot_source_root = project_root.parent
    manifest_path = (
        Path(source_manifest).expanduser().resolve()
        if source_manifest is not None
        else snapshot_source_root.parent / "source.sha256"
    )
    _require_regular_file(manifest_path, label="full source manifest")
    manifest_mode = stat.S_IMODE(manifest_path.lstat().st_mode)
    if manifest_mode & 0o222:
        raise PermissionError(
            f"full source manifest is writable (mode {manifest_mode:04o}): "
            f"{manifest_path}"
        )
    if not snapshot_source_root.is_dir() or project_root.parent != snapshot_source_root:
        raise NotADirectoryError(
            f"invalid snapshot source root for manifest validation: {project_root}"
        )

    records: Dict[str, str] = {}
    raw_lines = manifest_path.read_text(encoding="utf-8").splitlines()
    if not raw_lines:
        raise ValueError("full source manifest is empty")
    for line_number, line in enumerate(raw_lines, start=1):
        if len(line) < 69 or line[64:66] != "  ":
            raise ValueError(
                f"invalid source manifest syntax at line {line_number}"
            )
        digest = line[:64]
        rendered_path = line[66:]
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(
                f"invalid source manifest SHA-256 at line {line_number}"
            )
        if not rendered_path.startswith("./"):
            raise ValueError(
                f"source manifest path must begin './' at line {line_number}"
            )
        relative_text = rendered_path[2:]
        relative = Path(relative_text)
        if (
            not relative_text
            or relative.is_absolute()
            or ".." in relative.parts
            or "." in relative.parts
            or relative.as_posix() != relative_text
        ):
            raise ValueError(
                f"unsafe/non-canonical source manifest path at line "
                f"{line_number}: {rendered_path!r}"
            )
        if relative_text in records:
            raise ValueError(f"duplicate source manifest path: {rendered_path}")
        records[relative_text] = digest

    actual_files: Dict[str, Path] = {}
    file_modes: Dict[str, int] = {}
    directory_modes: Dict[str, int] = {}
    stack = [snapshot_source_root]
    while stack:
        directory = stack.pop()
        directory_stat = directory.lstat()
        if directory.is_symlink() or not stat.S_ISDIR(directory_stat.st_mode):
            raise ValueError(f"source tree directory is not a real directory: {directory}")
        directory_mode = stat.S_IMODE(directory_stat.st_mode)
        if directory_mode & 0o222:
            raise PermissionError(
                f"writable directory in immutable source tree "
                f"(mode {directory_mode:04o}): {directory}"
            )
        directory_modes[f"{directory_mode:04o}"] = (
            directory_modes.get(f"{directory_mode:04o}", 0) + 1
        )
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                node_stat = path.lstat()
                if stat.S_ISLNK(node_stat.st_mode):
                    raise ValueError(f"symlink in immutable source tree: {path}")
                if stat.S_ISDIR(node_stat.st_mode):
                    stack.append(path)
                    continue
                if not stat.S_ISREG(node_stat.st_mode):
                    raise ValueError(f"special node in immutable source tree: {path}")
                mode = stat.S_IMODE(node_stat.st_mode)
                if mode & 0o222:
                    raise PermissionError(
                        f"writable file in immutable source tree "
                        f"(mode {mode:04o}): {path}"
                    )
                relative = path.relative_to(snapshot_source_root).as_posix()
                if relative in actual_files:
                    raise ValueError(f"duplicate source-tree path: {relative}")
                actual_files[relative] = path
                file_modes[f"{mode:04o}"] = file_modes.get(f"{mode:04o}", 0) + 1

    expected_paths = set(records)
    actual_paths = set(actual_files)
    if expected_paths != actual_paths:
        missing = sorted(expected_paths - actual_paths)
        unexpected = sorted(actual_paths - expected_paths)
        raise ValueError(
            "full source manifest path set disagrees with snapshot tree: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    for relative in sorted(records):
        actual_hash = sha256_file(actual_files[relative])
        if actual_hash != records[relative]:
            raise ValueError(f"full source manifest hash disagrees: ./{relative}")

    canonical = "".join(
        f"{records[relative]}  ./{relative}\n" for relative in sorted(records)
    ).encode("utf-8")
    path_set_payload = "".join(f"./{relative}\n" for relative in sorted(records))
    return {
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "canonical_tree_sha256": hashlib.sha256(canonical).hexdigest(),
        "path_set_sha256": hashlib.sha256(path_set_payload.encode("utf-8")).hexdigest(),
        "source_tree_root": str(snapshot_source_root),
        "project_root": str(project_root),
        "file_count": len(actual_files),
        "directory_count": sum(directory_modes.values()),
        "file_mode_counts": dict(sorted(file_modes.items())),
        "directory_mode_counts": dict(sorted(directory_modes.items())),
        "manifest_mode": f"{manifest_mode:04o}",
        "writable_node_count": 0,
        "symlink_count": 0,
        "special_node_count": 0,
        "path_set_exact": True,
        "all_hashes_match": True,
        "immutable_modes_verified": True,
    }


def validate_checkpoint_bundle(
    checkpoint_dir: str | Path,
    *,
    training_source_root: Optional[str | Path] = None,
    source_manifest: Optional[str | Path] = None,
    completion_schema_version: int = 1,
) -> BundleValidation:
    """Validate final checkpoint, source, and compiled-extension provenance."""

    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(f"checkpoint directory does not exist: {checkpoint_dir}")
    marker_path = checkpoint_dir / "finish"
    marker_payload = _read_stable_single_link_bytes(
        marker_path, label="completion marker"
    )
    marker = _load_strict_json_object(
        marker_payload, label="completion marker"
    )
    if type(completion_schema_version) is not int or completion_schema_version not in (
        1,
        2,
    ):
        raise TypeError("completion schema version must be exact built-in int 1 or 2")
    if type(marker.get("schema_version")) is not int or (
        marker["schema_version"] != completion_schema_version
    ):
        raise ValueError(
            "checkpoint completion schema_version must be integer "
            f"{completion_schema_version}"
        )
    if type(marker.get("status")) is not str or marker["status"] != "complete":
        raise ValueError(
            "checkpoint is not a completed schema-v"
            f"{completion_schema_version} bundle"
        )

    required_checkpoint_files = (
        SCHEMA13_REQUIRED_CHECKPOINT_FILES
        if completion_schema_version == 2
        else REQUIRED_CHECKPOINT_FILES
    )
    recorded = _require_strict_checkpoint_file_records(
        marker.get("checkpoint_files"),
        label="completion marker",
        expected_names=required_checkpoint_files,
    )
    hashes: Dict[str, str] = {}
    for name in required_checkpoint_files:
        path = checkpoint_dir / name
        payload = _read_stable_single_link_bytes(path, label=name)
        record = recorded[name]
        recorded_hash = record["sha256"]
        recorded_size = record["size"]
        actual_hash = hashlib.sha256(payload).hexdigest()
        actual_size = len(payload)
        if recorded_hash != actual_hash:
            raise ValueError(f"completion marker hash disagrees for {name}")
        if recorded_size != actual_size:
            raise ValueError(f"completion marker size disagrees for {name}")
        hashes[name] = actual_hash
    hashes["finish"] = hashlib.sha256(marker_payload).hexdigest()

    source_root = resolve_training_source_root(
        checkpoint_dir, marker, explicit=training_source_root
    )
    source_manifest_state = validate_full_source_manifest(
        source_root, source_manifest=source_manifest
    )
    # Resolve again with detailed errors now that the matching root is known.
    for relative, record in marker["implementation_sources"].items():
        relative_path = _safe_relative_path(relative, label="implementation")
        path = (source_root / relative_path).resolve()
        if source_root != path and source_root not in path.parents:
            raise ValueError(f"implementation path escapes source root: {relative}")
        _require_regular_file(path, label=f"training source {relative}")
        if not isinstance(record, Mapping) or record.get("sha256") != sha256_file(path):
            raise ValueError(f"training source hash disagrees: {relative}")

    extensions = marker.get("loaded_extensions")
    if not isinstance(extensions, Mapping) or not extensions:
        raise ValueError("completion marker lacks the loaded cenv extension")
    for relative, record in extensions.items():
        relative_path = _safe_relative_path(relative, label="extension")
        path = (source_root / relative_path).resolve()
        if source_root not in path.parents:
            raise ValueError(f"extension path escapes source root: {relative}")
        _require_regular_file(path, label=f"training extension {relative}")
        if not isinstance(record, Mapping) or record.get("sha256") != sha256_file(path):
            raise ValueError(f"training extension hash disagrees: {relative}")
    return BundleValidation(
        checkpoint_dir,
        source_root,
        marker,
        hashes,
        source_manifest_state,
    )


def bind_training_runtime(source_root: str | Path) -> None:
    """Put the marker-matched source first and reject a mixed thinker import."""

    source_root = Path(source_root).resolve()
    source_text = str(source_root)
    sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != source_root]
    sys.path.insert(0, source_text)
    existing = sys.modules.get("thinker")
    if existing is not None:
        existing_file = Path(getattr(existing, "__file__", "")).resolve()
        if source_root not in existing_file.parents:
            raise RuntimeError(
                "thinker was imported before binding the training source root: "
                f"{existing_file}"
            )
    import thinker  # pylint: disable=import-outside-toplevel

    loaded = Path(thinker.__file__).resolve()
    if source_root not in loaded.parents:
        raise RuntimeError(
            f"runtime thinker import escaped training source root: {loaded}"
        )


def validate_loaded_training_modules(
    source_manifest_state: Mapping[str, Any],
) -> List[Dict[str, str]]:
    """Require every loaded ``thinker.*`` module to come from the frozen tree."""

    source_tree_root = Path(source_manifest_state["source_tree_root"]).resolve()
    manifest_path = Path(source_manifest_state["path"]).resolve()
    manifest_hashes = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        manifest_hashes[line[68:]] = line[:64]
    loaded = []
    for name, module in sorted(sys.modules.items()):
        if name != "thinker" and not name.startswith("thinker."):
            continue
        value = getattr(module, "__file__", None)
        if value is None:
            continue
        path = Path(value).resolve()
        if source_tree_root not in path.parents:
            raise RuntimeError(f"loaded training module escaped snapshot: {name}={path}")
        relative = path.relative_to(source_tree_root).as_posix()
        expected_hash = manifest_hashes.get(relative)
        if expected_hash is None:
            raise RuntimeError(
                f"loaded training module is absent from source manifest: {name}={relative}"
            )
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"loaded training module drifted: {name}={relative}")
        loaded.append({"module": name, "path": relative, "sha256": actual_hash})
    if not any(item["module"] == "thinker.cenv" for item in loaded):
        raise RuntimeError("fixed evaluation did not load the frozen thinker.cenv")
    return loaded


def loaded_training_modules_attestation(
    source_manifest_state: Mapping[str, Any],
) -> Dict[str, Any]:
    modules = validate_loaded_training_modules(source_manifest_state)
    payload = json.dumps(
        modules, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return {
        "count": len(modules),
        "semantic_sha256": hashlib.sha256(payload).hexdigest(),
        "modules": modules,
    }


def validate_enduro_rom(
    expected_sha256: str,
    *,
    rom_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Bind EnvPool evaluation to the preregistered Enduro ROM bytes."""

    expected = str(expected_sha256).lower()
    if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError("expected Enduro ROM SHA-256 must be 64 lowercase hex digits")
    if expected != PREREGISTERED_ENDURO_ROM_SHA256:
        raise ValueError(
            "fixed confirmation requires the preregistered Enduro ROM SHA-256"
        )
    if rom_path is None:
        import envpool  # pylint: disable=import-outside-toplevel

        path = Path(envpool.__file__).resolve().parent / "atari" / "roms" / "enduro.bin"
    else:
        path = Path(rom_path).expanduser().resolve()
    attestation = attest_regular_file(path, label="EnvPool Enduro ROM")
    if attestation["sha256"] != expected:
        raise ValueError(
            "EnvPool Enduro ROM hash disagrees with the preregistered bytes: "
            f"{attestation['sha256']} != {expected}"
        )
    return {"expected_sha256": expected, **attestation}


def collect_runtime_attestation(
    device: torch.device,
    *,
    expected_rom_sha256: str,
) -> Dict[str, Any]:
    """Collect reproducibility-critical software, GPU, and ROM identity."""

    packages = {}
    for output_name, distribution_name in (
        ("envpool", "envpool"),
        ("gymnasium", "gymnasium"),
        ("ale_py", "ale-py"),
    ):
        try:
            packages[output_name] = importlib_metadata.version(distribution_name)
        except importlib_metadata.PackageNotFoundError as error:
            raise RuntimeError(
                f"required runtime package is not installed: {distribution_name}"
            ) from error

    gpu = None
    if device.type == "cuda":
        index = device.index
        if index is None:
            index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        gpu = {
            "logical_index": int(index),
            "name": str(properties.name),
            "compute_capability": f"{properties.major}.{properties.minor}",
            "compute_capability_major": int(properties.major),
            "compute_capability_minor": int(properties.minor),
            "total_memory_bytes": int(properties.total_memory),
        }
    return {
        "python": platform.python_version(),
        "numpy": str(np.__version__),
        "torch": str(torch.__version__),
        "torch_cuda": None if torch.version.cuda is None else str(torch.version.cuda),
        "cudnn_version": torch.backends.cudnn.version(),
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": gpu,
        "packages": packages,
        "enduro_rom": validate_enduro_rom(expected_rom_sha256),
    }


def _set_deterministic_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    seed = int(seed)
    if seed < 0 or seed >= 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _finite_float(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def reconstruct_dueling_q(
    raw_advantage: torch.Tensor,
    state_value: torch.Tensor,
    continue_probability: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct [CONTINUE, STOP] Q exactly as the EMA learner path does."""

    if raw_advantage.shape[-1:] != (2,):
        raise ValueError("raw_advantage must end in [CONTINUE, STOP]")
    if tuple(state_value.shape) != tuple(raw_advantage.shape[:-1]):
        raise ValueError("state_value shape does not match raw_advantage")
    if tuple(continue_probability.shape) != tuple(state_value.shape):
        raise ValueError("continue_probability shape does not match state_value")
    raw = raw_advantage.float()
    state = state_value.float()
    probability = continue_probability.float()
    if torch.any(~torch.isfinite(raw)) or torch.any(~torch.isfinite(state)):
        raise FloatingPointError("non-finite Q reconstruction input")
    if torch.any(~torch.isfinite(probability)) or torch.any(
        (probability < 0.0) | (probability > 1.0)
    ):
        raise ValueError("continue_probability must be finite and in [0,1]")
    gate_probability = torch.stack((probability, 1.0 - probability), dim=-1)
    centered = raw - torch.sum(gate_probability * raw, dim=-1, keepdim=True)
    result = state.unsqueeze(-1) + centered
    if torch.any(~torch.isfinite(result)):
        raise FloatingPointError("non-finite reconstructed Q")
    return result


def on_policy_vtrace_target(
    rewards: torch.Tensor,
    discounts: torch.Tensor,
    values: torch.Tensor,
    bootstrap_value: torch.Tensor,
    *,
    lamb: float,
) -> torch.Tensor:
    """Return the selected-action target used by rho=1 fixed-policy V-trace."""

    if rewards.shape != discounts.shape or rewards.shape != values.shape:
        raise ValueError("rewards, discounts, and values must have equal [T,B] shape")
    if rewards.ndim != 2 or tuple(bootstrap_value.shape) != tuple(rewards.shape[1:]):
        raise ValueError("bootstrap_value must have shape [B]")
    lamb = _finite_float(lamb, label="V-trace lambda")
    if not 0.0 <= lamb <= 1.0:
        raise ValueError("V-trace lambda must be in [0,1]")
    for label, value in (
        ("rewards", rewards),
        ("discounts", discounts),
        ("values", values),
        ("bootstrap_value", bootstrap_value),
    ):
        if torch.any(~torch.isfinite(value)):
            raise FloatingPointError(f"non-finite {label}")

    next_values = torch.cat((values[1:], bootstrap_value.unsqueeze(0)), dim=0)
    deltas = rewards + discounts * next_values - values
    accumulator = torch.zeros_like(bootstrap_value)
    corrections: List[torch.Tensor] = []
    for index in range(rewards.shape[0] - 1, -1, -1):
        accumulator = deltas[index] + discounts[index] * lamb * accumulator
        corrections.append(accumulator)
    corrections.reverse()
    vs = values + torch.stack(corrections)
    next_vs = torch.cat((vs[1:], bootstrap_value.unsqueeze(0)), dim=0)
    return rewards + discounts * next_vs


def _mean(values: Sequence[float]) -> Optional[float]:
    return float(math.fsum(values) / len(values)) if values else None


def _rate(numerator: int, denominator: int) -> Optional[float]:
    return float(numerator / denominator) if denominator else None


def _slice_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    tie_tolerance: float = TIE_TOLERANCE,
    q_temperature: float,
) -> Dict[str, Any]:
    """Pool event-weighted sufficient statistics for one decision slice."""

    tolerance = _finite_float(tie_tolerance, label="tie tolerance")
    temperature = _finite_float(q_temperature, label="Q temperature")
    if tolerance < 0.0 or temperature <= 0.0:
        raise ValueError("tie tolerance must be non-negative and Q temperature positive")

    positive: List[Mapping[str, Any]] = []
    negative: List[Mapping[str, Any]] = []
    tied: List[Mapping[str, Any]] = []
    for row in rows:
        delta = _finite_float(row["ema_delta_q"], label="ema_delta_q")
        probability = _finite_float(
            row["continue_probability"], label="continue_probability"
        )
        if not 0.0 <= probability <= 1.0:
            raise ValueError("continue_probability must be in [0,1]")
        control = int(row["sampled_control"])
        gate_action = int(row["gate_action"])
        if control not in (PROCEED, RESET, STOP):
            raise ValueError("sampled_control must be PROCEED, RESET, or STOP")
        expected_gate = GATE_STOP if control == STOP else GATE_CONTINUE
        if gate_action != expected_gate:
            raise ValueError("gate_action disagrees with sampled_control")
        if delta > tolerance:
            positive.append(row)
        elif delta < -tolerance:
            negative.append(row)
        else:
            tied.append(row)

    nontie = positive + negative
    probabilities_positive = [float(row["continue_probability"]) for row in positive]
    probabilities_negative = [float(row["continue_probability"]) for row in negative]
    teacher_positive = [
        1.0 / (1.0 + math.exp(-float(row["ema_delta_q"]) / temperature))
        for row in positive
    ]
    teacher_negative = [
        1.0 / (1.0 + math.exp(-float(row["ema_delta_q"]) / temperature))
        for row in negative
    ]
    student_positive = _mean(probabilities_positive)
    student_negative = _mean(probabilities_negative)
    teacher_positive_mean = _mean(teacher_positive)
    teacher_negative_mean = _mean(teacher_negative)
    student_gap = (
        student_positive - student_negative
        if student_positive is not None and student_negative is not None
        else None
    )
    teacher_gap = (
        teacher_positive_mean - teacher_negative_mean
        if teacher_positive_mean is not None and teacher_negative_mean is not None
        else None
    )
    retention = (
        student_gap / teacher_gap
        if student_gap is not None and teacher_gap not in (None, 0.0)
        else None
    )
    signed_margins = [
        math.copysign(1.0, float(row["ema_delta_q"]))
        * (2.0 * float(row["continue_probability"]) - 1.0)
        for row in nontie
    ]
    return {
        "count": len(rows),
        "positive_count": len(positive),
        "negative_count": len(negative),
        "tie_count": len(tied),
        "nontie_count": len(nontie),
        "positive_support_fraction": _rate(len(positive), len(nontie)),
        "negative_support_fraction": _rate(len(negative), len(nontie)),
        "continue_probability_positive": student_positive,
        "continue_probability_negative": student_negative,
        "teacher_continue_probability_positive": teacher_positive_mean,
        "teacher_continue_probability_negative": teacher_negative_mean,
        "student_conditional_gap": student_gap,
        "teacher_conditional_gap": teacher_gap,
        "student_teacher_gap_retention": retention,
        "signed_margin": _mean(signed_margins),
        "sampled_continue_given_positive_rate": _rate(
            sum(int(int(row["sampled_control"]) != STOP) for row in positive),
            len(positive),
        ),
        "sampled_stop_given_negative_rate": _rate(
            sum(int(int(row["sampled_control"]) == STOP) for row in negative),
            len(negative),
        ),
        # Binary argmax order is [CONTINUE, STOP]; p=0.5 resolves CONTINUE.
        "argmax_continue_given_positive_rate": _rate(
            sum(int(float(row["continue_probability"]) >= 0.5) for row in positive),
            len(positive),
        ),
        "argmax_stop_given_negative_rate": _rate(
            sum(int(float(row["continue_probability"]) < 0.5) for row in negative),
            len(negative),
        ),
        "wrong_continue_saturation_rate": _rate(
            sum(int(float(row["continue_probability"]) < 0.1) for row in positive),
            len(positive),
        ),
        "wrong_stop_saturation_rate": _rate(
            sum(int(float(row["continue_probability"]) > 0.9) for row in negative),
            len(negative),
        ),
    }


def _strict_useful_pairs(
    rows: Sequence[Mapping[str, Any]],
    *,
    tie_tolerance: float,
) -> Tuple[List[Mapping[str, Any]], List[Tuple[Mapping[str, Any], Mapping[str, Any]]]]:
    """Return prior-useful candidates and exact adjacent same-stream pairs."""

    ordered = sorted(
        rows, key=lambda row: (int(row["stream_id"]), int(row["augmented_step"]))
    )
    by_location = {
        (int(row["stream_id"]), int(row["augmented_step"])): row
        for row in ordered
    }
    if len(by_location) != len(ordered):
        raise ValueError("duplicate decision row for one stream/augmented step")
    candidates = [
        row
        for row in ordered
        if float(row["ema_delta_q"]) > tie_tolerance
        and int(row["sampled_control"]) in (PROCEED, RESET)
    ]
    pairs = []
    for prior in candidates:
        current = by_location.get(
            (int(prior["stream_id"]), int(prior["augmented_step"]) + 1)
        )
        if current is None:
            continue
        if (
            int(current["predecision_last_control"])
            != int(prior["sampled_control"])
            or int(current["decision_depth"])
            != int(prior["decision_depth"]) + 1
        ):
            continue
        pairs.append((prior, current))
    return candidates, pairs


def _calibration_metrics(
    rows: Sequence[Mapping[str, Any]], *, q_key: str
) -> Dict[str, Any]:
    selected_key = f"{q_key}_selected_q"
    errors = [
        _finite_float(row["calibration_net_target"], label="calibration target")
        - _finite_float(row[selected_key], label=selected_key)
        for row in rows
    ]
    continue_errors = [
        error
        for row, error in zip(rows, errors)
        if int(row["gate_action"]) == GATE_CONTINUE
    ]
    stop_errors = [
        error
        for row, error in zip(rows, errors)
        if int(row["gate_action"]) == GATE_STOP
    ]

    def block(values: Sequence[float]) -> Dict[str, Any]:
        return {
            "count": len(values),
            "bias": _mean(values),
            "mae": _mean([abs(value) for value in values]),
            "rmse": (
                math.sqrt(math.fsum(value * value for value in values) / len(values))
                if values
                else None
            ),
        }

    return {
        **block(errors),
        "continue": block(continue_errors),
        "stop": block(stop_errors),
    }


def summarize_decision_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    q_temperature: float,
    stage_end_count: int,
    forced_stop_count: int,
    tie_tolerance: float = TIE_TOLERANCE,
) -> Dict[str, Any]:
    """Pool raw fixed-rollout rows and evaluate all four frozen behaviours."""

    if stage_end_count < 0 or forced_stop_count < 0 or forced_stop_count > stage_end_count:
        raise ValueError("invalid stage-end/forced-stop counts")
    overall = _slice_metrics(
        rows, tie_tolerance=tie_tolerance, q_temperature=q_temperature
    )
    deep_negative = [
        row
        for row in rows
        if int(row["decision_depth"]) >= 8
        and float(row["ema_delta_q"]) < -tie_tolerance
    ]
    candidates, pairs = _strict_useful_pairs(rows, tie_tolerance=tie_tolerance)
    proceed_pairs = [pair for pair in pairs if int(pair[0]["sampled_control"]) == PROCEED]
    reset_pairs = [pair for pair in pairs if int(pair[0]["sampled_control"]) == RESET]
    current_rows = [current for _, current in pairs]
    current_metrics = _slice_metrics(
        current_rows, tie_tolerance=tie_tolerance, q_temperature=q_temperature
    )
    proceed_metrics = _slice_metrics(
        [current for _, current in proceed_pairs],
        tie_tolerance=tie_tolerance,
        q_temperature=q_temperature,
    )
    reset_metrics = _slice_metrics(
        [current for _, current in reset_pairs],
        tie_tolerance=tie_tolerance,
        q_temperature=q_temperature,
    )

    deep_continue_probability = _mean(
        [float(row["continue_probability"]) for row in deep_negative]
    )
    deep_sampled_stop = _rate(
        sum(int(int(row["sampled_control"]) == STOP) for row in deep_negative),
        len(deep_negative),
    )
    deep_argmax_stop = _rate(
        sum(int(float(row["continue_probability"]) < 0.5) for row in deep_negative),
        len(deep_negative),
    )
    forced_rate = _rate(forced_stop_count, stage_end_count)

    online_ema_sign_support = [
        row
        for row in rows
        if abs(float(row["ema_delta_q"])) > tie_tolerance
        and abs(float(row["online_delta_q"])) > tie_tolerance
    ]
    online_ema_sign_agreement = _rate(
        sum(
            int(
                math.copysign(1.0, float(row["ema_delta_q"]))
                == math.copysign(1.0, float(row["online_delta_q"]))
            )
            for row in online_ema_sign_support
        ),
        len(online_ema_sign_support),
    )

    sign_support_ok = (
        overall["positive_support_fraction"] is not None
        and overall["positive_support_fraction"] > 0.05
        and overall["negative_support_fraction"] is not None
        and overall["negative_support_fraction"] > 0.05
    )
    behavior_1 = {
        "name": "easy_negative_q_states_stop",
        "sign_support_ok": sign_support_ok,
        "continue_probability_ok": (
            overall["continue_probability_negative"] is not None
            and overall["continue_probability_negative"] <= 0.475
        ),
        "sampled_stop_ok": (
            overall["sampled_stop_given_negative_rate"] is not None
            and overall["sampled_stop_given_negative_rate"] >= 0.525
        ),
        "argmax_stop_ok": (
            overall["argmax_stop_given_negative_rate"] is not None
            and overall["argmax_stop_given_negative_rate"] >= 0.60
        ),
    }
    behavior_1["pass"] = all(
        behavior_1[key]
        for key in (
            "sign_support_ok",
            "continue_probability_ok",
            "sampled_stop_ok",
            "argmax_stop_ok",
        )
    )
    behavior_2 = {
        "name": "hard_positive_q_states_continue",
        "continue_probability_ok": (
            overall["continue_probability_positive"] is not None
            and overall["continue_probability_positive"] >= 0.525
        ),
        "sampled_continue_ok": (
            overall["sampled_continue_given_positive_rate"] is not None
            and overall["sampled_continue_given_positive_rate"] >= 0.525
        ),
        "argmax_continue_ok": (
            overall["argmax_continue_given_positive_rate"] is not None
            and overall["argmax_continue_given_positive_rate"] >= 0.60
        ),
    }
    behavior_2["pass"] = all(
        behavior_2[key]
        for key in (
            "continue_probability_ok",
            "sampled_continue_ok",
            "argmax_continue_ok",
        )
    )
    behavior_3 = {
        "name": "deep_negative_q_states_stop",
        "support": len(deep_negative),
        "continue_probability": deep_continue_probability,
        "sampled_stop_rate": deep_sampled_stop,
        "argmax_stop_accuracy": deep_argmax_stop,
        "forced_stop_count": forced_stop_count,
        "stage_end_count": stage_end_count,
        "forced_stop_rate": forced_rate,
        "support_ok": len(deep_negative) >= 256,
        "continue_probability_ok": (
            deep_continue_probability is not None
            and deep_continue_probability <= 0.475
        ),
        "sampled_stop_ok": deep_sampled_stop is not None and deep_sampled_stop >= 0.525,
        "argmax_stop_ok": deep_argmax_stop is not None and deep_argmax_stop >= 0.60,
        "forced_stop_rate_ok": forced_rate is not None and forced_rate < 0.01,
        "forced_stops_count_as_successes": False,
    }
    behavior_3["pass"] = all(
        behavior_3[key]
        for key in (
            "support_ok",
            "continue_probability_ok",
            "sampled_stop_ok",
            "argmax_stop_ok",
            "forced_stop_rate_ok",
        )
    )
    behavior_4 = {
        "name": "useful_compute_then_reevaluate",
        "candidate_count": len(candidates),
        "eligible_pair_count": len(pairs),
        "coverage_rate": _rate(len(pairs), len(candidates)),
        "proceed_pair_count": len(proceed_pairs),
        "reset_pair_count": len(reset_pairs),
        "next_decision": current_metrics,
        "proceed_next_decision": proceed_metrics,
        "reset_next_decision": reset_metrics,
    }
    behavior_4.update(
        {
            "coverage_ok": (
                behavior_4["coverage_rate"] is not None
                and behavior_4["coverage_rate"] >= 0.95
            ),
            "proceed_support_ok": len(proceed_pairs) >= 256,
            "reset_support_ok": len(reset_pairs) >= 256,
            "next_positive_support_ok": current_metrics["positive_count"] >= 128,
            "next_negative_support_ok": current_metrics["negative_count"] >= 128,
            "next_positive_probability_ok": (
                current_metrics["continue_probability_positive"] is not None
                and current_metrics["continue_probability_positive"] >= 0.525
            ),
            "next_negative_probability_ok": (
                current_metrics["continue_probability_negative"] is not None
                and current_metrics["continue_probability_negative"] <= 0.475
            ),
            "next_sampled_positive_ok": (
                current_metrics["sampled_continue_given_positive_rate"] is not None
                and current_metrics["sampled_continue_given_positive_rate"] >= 0.525
            ),
            "next_sampled_negative_ok": (
                current_metrics["sampled_stop_given_negative_rate"] is not None
                and current_metrics["sampled_stop_given_negative_rate"] >= 0.525
            ),
            "next_argmax_positive_ok": (
                current_metrics["argmax_continue_given_positive_rate"] is not None
                and current_metrics["argmax_continue_given_positive_rate"] >= 0.60
            ),
            "next_argmax_negative_ok": (
                current_metrics["argmax_stop_given_negative_rate"] is not None
                and current_metrics["argmax_stop_given_negative_rate"] >= 0.60
            ),
            "proceed_margin_ok": (
                proceed_metrics["signed_margin"] is not None
                and proceed_metrics["signed_margin"] > 0.0
            ),
            "reset_margin_ok": (
                reset_metrics["signed_margin"] is not None
                and reset_metrics["signed_margin"] > 0.0
            ),
        }
    )
    behavior_4["pass"] = all(
        behavior_4[key]
        for key in (
            "coverage_ok",
            "proceed_support_ok",
            "reset_support_ok",
            "next_positive_support_ok",
            "next_negative_support_ok",
            "next_positive_probability_ok",
            "next_negative_probability_ok",
            "next_sampled_positive_ok",
            "next_sampled_negative_ok",
            "next_argmax_positive_ok",
            "next_argmax_negative_ok",
            "proceed_margin_ok",
            "reset_margin_ok",
        )
    )

    ema_calibration = _calibration_metrics(rows, q_key="ema")
    online_calibration = _calibration_metrics(rows, q_key="online")
    return {
        "schema_version": SCHEMA_VERSION,
        "tie_tolerance": tie_tolerance,
        "q_temperature": q_temperature,
        "overall": overall,
        "online_ema_sign_agreement": {
            "count": len(online_ema_sign_support),
            "rate": online_ema_sign_agreement,
        },
        "behaviors": {
            "1_easy_stop": behavior_1,
            "2_hard_search": behavior_2,
            "3_deep_stop": behavior_3,
            "4_useful_compute_reevaluate": behavior_4,
        },
        "all_four_behaviors_pass": all(
            behavior["pass"]
            for behavior in (behavior_1, behavior_2, behavior_3, behavior_4)
        ),
        "selected_action_calibration": {
            "definition": (
                "on-policy rho=1 V-trace target over fixed held-out seed "
                "rollouts; task return + think_cost * stage-local think return"
            ),
            "ema": {
                **ema_calibration,
                "rmse_at_most_0_5": (
                    ema_calibration["rmse"] is not None
                    and ema_calibration["rmse"] <= 0.5
                ),
            },
            "online": online_calibration,
        },
    }


def _state_dict_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _atomic_write_json(value: Mapping[str, Any], path: Path) -> None:
    temporary: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def _atomic_write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    temporary: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            writer = csv.DictWriter(handle, fieldnames=DECISION_CSV_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row[field] for field in DECISION_CSV_FIELDS})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def exclusive_output_lock(
    output_dir: Path, *, generation_id: str, evaluator_attestation: Mapping[str, Any]
) -> Iterable[OutputGenerationLock]:
    """Hold an exclusive generation lock; a crash intentionally leaves it stale."""

    lock_path = output_dir / OUTPUT_LOCK_NAME
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except FileExistsError as error:
        raise FileExistsError(
            f"fixed-evaluation output directory is locked: {lock_path}"
        ) from error
    state = OutputGenerationLock(path=lock_path, generation_id=generation_id)
    try:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "generation_id": generation_id,
            "pid": os.getpid(),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "evaluator": dict(evaluator_attestation),
        }
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        descriptor = -1
        _fsync_directory(output_dir)
        yield state
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        # Before publication, a handled error leaves no public generation and
        # the lock can be released.  After publication starts, retain the lock
        # on every failure so a partial overwrite remains visibly invalid.
        # Only a fully fsynced manifest-last commit may release it.
        if not state.publication_started or state.committed:
            lock_path.unlink(missing_ok=True)
        _fsync_directory(output_dir)
        if state.committed and lock_path.exists():
            raise RuntimeError(f"failed to release output lock: {lock_path}")


def commit_staged_generation(
    staged_outputs: Mapping[str, Path],
    final_outputs: Mapping[str, Path],
    *,
    lock: OutputGenerationLock,
) -> None:
    """Commit decisions/summary first and manifest last as the commit marker."""

    if set(staged_outputs) != set(final_outputs):
        raise ValueError("staged and final output keys differ")
    commit_order = ("decisions", "summary", "manifest")
    if set(staged_outputs) != set(commit_order):
        raise ValueError("fixed generation requires decisions, summary, and manifest")
    for name in commit_order:
        _require_regular_file(staged_outputs[name], label=f"staged {name}")
    stage_parents = {path.parent.resolve() for path in staged_outputs.values()}
    final_parents = {path.parent.resolve() for path in final_outputs.values()}
    if len(stage_parents) != 1 or len(final_parents) != 1:
        raise ValueError("generation outputs must each share one directory")
    final_parent = next(iter(final_parents))
    if lock.path.parent.resolve() != final_parent or not lock.path.exists():
        raise RuntimeError("generation publication does not hold its output lock")
    _fsync_directory(next(iter(stage_parents)))
    lock.publication_started = True
    for name in commit_order:
        os.replace(staged_outputs[name], final_outputs[name])
    _fsync_directory(final_parent)
    lock.committed = True


def decision_rows_semantic_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash ordered decision evidence independently of CSV formatting."""

    payload = []
    for row in rows:
        normalized = {}
        for field in DECISION_CSV_FIELDS:
            if field not in row:
                raise KeyError(f"decision row lacks semantic-hash field {field!r}")
            value = row[field]
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, float) and not math.isfinite(value):
                raise FloatingPointError(
                    f"non-finite decision semantic-hash field {field}"
                )
            normalized[field] = value
        payload.append(normalized)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _ensure_output_is_external(output_dir: Path, bundle: BundleValidation) -> None:
    output_dir = output_dir.resolve()
    for protected, label in (
        (bundle.checkpoint_dir, "checkpoint directory"),
        (bundle.source_root, "immutable training source"),
    ):
        if output_dir == protected or protected in output_dir.parents:
            raise ValueError(f"output directory may not be inside the {label}")


def _parse_seeds(seed_base: int, num_seeds: int) -> Tuple[int, ...]:
    if seed_base < 0 or seed_base >= 2**32:
        raise ValueError("seed base must be in [0, 2**32)")
    if num_seeds < 1 or seed_base + num_seeds > 2**32:
        raise ValueError("seed range is empty or exceeds uint32")
    return tuple(range(seed_base, seed_base + num_seeds))


def validate_behavioral_training_data(
    *,
    flags: Any,
    spec: Any,
    actor_checkpoint: Mapping[str, Any],
    data_root: str | Path,
    checkpoint_eval: Any,
) -> Dict[str, Any]:
    """Recompute the exact behavioral corpus signature stored by training."""

    from thinker.bc_loader import (  # pylint: disable=import-outside-toplevel
        FrameStackedBehavioralDataLoader,
        behavioral_data_signature,
    )

    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"behavioral data root does not exist: {root}")
    loader = FrameStackedBehavioralDataLoader(
        base_path=root,
        subjects=spec.subjects,
        sessions=spec.train_sessions,
        game_id=spec.game_id,
        split=None,
        scored_length=spec.scored_length,
        frame_stack_n=spec.frame_stack_n,
        target_size=spec.target_size,
        grayscale=spec.grayscale,
        normalize=spec.observation_dtype == "float32",
        decision_hz=15.0,
        num_actions=spec.num_actions,
        seed=0,
    )
    signature_before = behavioral_data_signature(loader, root)
    files = []
    for value in loader.data_files:
        path = Path(value).expanduser().resolve()
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError(f"behavioral file escaped data root: {path}") from error
        attestation = attest_regular_file(path, label="behavioral training file")
        files.append({"path": relative, **{k: v for k, v in attestation.items() if k != "path"}})
    if not files:
        raise RuntimeError("behavioral training-data selection is empty")
    files = sorted(files, key=lambda item: item["path"])
    signature_after = behavioral_data_signature(loader, root)
    if signature_before != signature_after:
        raise RuntimeError(
            "behavioral training data changed between signature passes"
        )
    reattested_files = []
    for record in files:
        path = (root / record["path"]).resolve()
        attestation = attest_regular_file(path, label="behavioral training file")
        reattested_files.append(
            {
                "path": record["path"],
                **{key: value for key, value in attestation.items() if key != "path"},
            }
        )
    if reattested_files != files:
        raise RuntimeError(
            "behavioral training data changed while binding its signature"
        )
    signature_state = checkpoint_eval.verify_actor_behavioral_data_signature(
        actor_checkpoint, signature_after
    )
    file_payload = json.dumps(
        files, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return {
        **signature_state,
        "data_root": str(root),
        "subjects": list(spec.subjects),
        "sessions": list(spec.train_sessions),
        "game_id": int(spec.game_id),
        "file_count": len(files),
        "files_semantic_sha256": hashlib.sha256(file_payload).hexdigest(),
        "files": files,
    }


def revalidate_behavioral_training_data(
    state: Mapping[str, Any],
) -> Dict[str, Any]:
    """Re-attest the exact selected behavioral files after the rollout."""

    root = Path(state["data_root"]).resolve()
    records = state.get("files")
    if not isinstance(records, Sequence) or not records:
        raise ValueError("behavioral data attestation has no selected files")
    current = []
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("behavioral data file record must be a mapping")
        relative = _safe_relative_path(record.get("path"), label="behavioral data")
        path = (root / relative).resolve()
        if root not in path.parents:
            raise ValueError(f"behavioral data path escapes its root: {relative}")
        attestation = attest_regular_file(path, label="behavioral training file")
        current_record = {
            "path": relative.as_posix(),
            **{key: value for key, value in attestation.items() if key != "path"},
        }
        current.append(current_record)
    current.sort(key=lambda item: item["path"])
    if current != list(records):
        raise RuntimeError("behavioral training data changed during fixed evaluation")
    payload = json.dumps(
        current, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    semantic_sha = hashlib.sha256(payload).hexdigest()
    if semantic_sha != state.get("files_semantic_sha256"):
        raise RuntimeError("behavioral training data attestation digest changed")
    return {
        "file_count": len(current),
        "files_semantic_sha256": semantic_sha,
        "unchanged": True,
    }


def _resolve_confirmation_profile(
    value: Any,
) -> Tuple[ConfirmationProfile, ConfirmationProfileSpec]:
    try:
        profile = (
            value
            if isinstance(value, ConfirmationProfile)
            else ConfirmationProfile(value)
        )
    except (TypeError, ValueError) as error:
        allowed = ", ".join(item.value for item in ConfirmationProfile)
        raise ValueError(
            f"unknown fixed-confirmation profile {value!r}; allowed: {allowed}"
        ) from error
    try:
        return profile, CONFIRMATION_PROFILE_SPECS[profile]
    except KeyError as error:  # pragma: no cover - protects future enum edits.
        raise RuntimeError(
            f"fixed-confirmation profile has no closed specification: {profile.value}"
        ) from error


def _require_exact_total_steps(value: Any, *, label: str, expected: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} total_steps must be the integer {expected}")
    actual = int(value)
    if actual != expected:
        raise ValueError(
            f"{label} total_steps must equal {expected}, got {actual}"
        )
    return actual


def _require_exact_integer(
    value: Any, *, label: str, field: str, expected: int
) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} {field} must be the integer {expected}")
    actual = int(value)
    if actual != expected:
        raise ValueError(f"{label} {field} must equal {expected}, got {actual}")
    return actual


def _require_exact_boolean(
    value: Any, *, label: str, field: str, expected: bool
) -> bool:
    if not isinstance(value, bool) or value is not expected:
        rendered = str(expected).lower()
        raise ValueError(f"{label} {field} must be exactly {rendered}")
    return value


def _require_exact_float(
    value: Any, *, label: str, field: str, expected: float
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{label} {field} must be exactly {expected!r}")
    actual = float(value)
    if not math.isfinite(actual) or actual != expected:
        raise ValueError(
            f"{label} {field} must equal exactly {expected!r}, got {actual!r}"
        )
    return actual


def _require_exact_string(
    value: Any, *, label: str, field: str, expected: str
) -> str:
    if not isinstance(value, str) or value != expected:
        raise ValueError(f"{label} {field} must equal exactly {expected!r}")
    return value


def _require_exact_none(value: Any, *, label: str, field: str) -> None:
    if value is not None:
        raise ValueError(f"{label} {field} must be exactly null")
    return None


def _profile_identity_value(container: Any, *, label: str, field: str) -> Any:
    if isinstance(container, Mapping):
        if field not in container:
            raise ValueError(f"{label} lacks {field}")
        return container[field]
    if not hasattr(container, field):
        raise ValueError(f"{label} lacks {field}")
    return getattr(container, field)


def _normalized_optional_boolean(
    container: Any, *, label: str, field: str
) -> bool:
    """Normalize a schema-optional historical boolean to explicit false."""

    if isinstance(container, Mapping):
        if field not in container:
            return False
        value = container[field]
    else:
        if not hasattr(container, field):
            return False
        value = getattr(container, field)
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} {field} must be boolean when present")
    return bool(value)


def _normalized_optional_seal_schema(
    container: Any, *, label: str
) -> int:
    """Normalize the schema-7 seal identity without weakening legacy zero."""

    field = "voc_model_input_seal_schema_version"
    if isinstance(container, Mapping):
        if field not in container:
            return 0
        value = container[field]
    else:
        if not hasattr(container, field):
            return 0
        value = getattr(container, field)
    if type(value) is not int or value not in (0, 1):
        raise ValueError(f"{label} {field} must be exact Python integer 0 or 1")
    return value


def _require_profile_flag_identity(
    container: Any,
    *,
    label: str,
    profile_spec: ConfirmationProfileSpec,
) -> Dict[str, Any]:
    resolved: Dict[str, Any] = {}
    if profile_spec.xpid is not None:
        resolved["xpid"] = _require_exact_string(
            _profile_identity_value(container, label=label, field="xpid"),
            label=label,
            field="xpid",
            expected=profile_spec.xpid,
        )
    if profile_spec.base_seed is not None:
        resolved["base_seed"] = _require_exact_integer(
            _profile_identity_value(container, label=label, field="base_seed"),
            label=label,
            field="base_seed",
            expected=profile_spec.base_seed,
        )
    if profile_spec.schedule_total_steps is not None:
        resolved["schedule_total_steps"] = _require_exact_integer(
            _profile_identity_value(
                container, label=label, field="schedule_total_steps"
            ),
            label=label,
            field="schedule_total_steps",
            expected=profile_spec.schedule_total_steps,
        )
    for field in (
        "model_warm_up_n",
        "actor_unroll_len",
        "voc_actor_policy_bundle_schema_version",
        "voc_actor_policy_ray_max_restarts",
        "voc_actor_policy_ray_max_task_retries",
        "voc_model_input_seal_schema_version",
        "ppo_k",
        "self_play_n",
        "env_n",
        "actor_batch_size",
    ):
        expected = getattr(profile_spec, field)
        if expected is not None:
            resolved[field] = _require_exact_integer(
                _profile_identity_value(container, label=label, field=field),
                label=label,
                field=field,
                expected=expected,
            )
    if profile_spec.dynamic_voc_mode is not None:
        resolved["dynamic_voc_mode"] = _require_exact_string(
            _profile_identity_value(
                container, label=label, field="dynamic_voc_mode"
            ),
            label=label,
            field="dynamic_voc_mode",
            expected=profile_spec.dynamic_voc_mode,
        )
    for field in (
        "voc_dedicated_gate",
        "voc_soft_q_bce_gate",
        "voc_eval_stochastic",
        "use_wandb",
        "voc_actor_policy_version_barrier",
        "voc_actor_policy_barrier_runtime",
        "float16",
        "model_float16",
        "parallel_actor",
    ):
        expected = getattr(profile_spec, field)
        if expected is not None:
            resolved[field] = _require_exact_boolean(
                _profile_identity_value(container, label=label, field=field),
                label=label,
                field=field,
                expected=expected,
            )
    for field in (
        "voc_gate_temperature",
        "voc_gate_q_temperature",
        "voc_train_epsilon",
        "voc_gate_execution_epsilon",
        "voc_actor_policy_barrier_timeout_s",
        "actor_amp_init_scale",
    ):
        expected = getattr(profile_spec, field)
        if expected is not None:
            resolved[field] = _require_exact_float(
                _profile_identity_value(container, label=label, field=field),
                label=label,
                field=field,
                expected=expected,
            )
    if profile_spec.voc_gate_param_align is not None:
        resolved["voc_gate_param_align"] = _require_exact_boolean(
            _profile_identity_value(
                container, label=label, field="voc_gate_param_align"
            ),
            label=label,
            field="voc_gate_param_align",
            expected=profile_spec.voc_gate_param_align,
        )
    if profile_spec.voc_gate_param_align_coef is not None:
        resolved["voc_gate_param_align_coef"] = _require_exact_float(
            _profile_identity_value(
                container, label=label, field="voc_gate_param_align_coef"
            ),
            label=label,
            field="voc_gate_param_align_coef",
            expected=profile_spec.voc_gate_param_align_coef,
        )
    if profile_spec.voc_gate_exact_projection is not None:
        resolved["voc_gate_exact_projection"] = _require_exact_boolean(
            _profile_identity_value(
                container, label=label, field="voc_gate_exact_projection"
            ),
            label=label,
            field="voc_gate_exact_projection",
            expected=profile_spec.voc_gate_exact_projection,
        )
    if profile_spec.voc_gate_epsilon_greedy_execution is not None:
        resolved["voc_gate_epsilon_greedy_execution"] = (
            _require_exact_boolean(
                _profile_identity_value(
                    container,
                    label=label,
                    field="voc_gate_epsilon_greedy_execution",
                ),
                label=label,
                field="voc_gate_epsilon_greedy_execution",
                expected=profile_spec.voc_gate_epsilon_greedy_execution,
            )
        )
    if profile_spec.ckp is not None:
        resolved["ckp"] = _require_exact_boolean(
            _profile_identity_value(container, label=label, field="ckp"),
            label=label,
            field="ckp",
            expected=profile_spec.ckp,
        )
    for field in ("preload", "preload_actor", "voc_parent_checkpoint"):
        expected = getattr(profile_spec, field)
        if expected is not None:
            resolved[field] = _require_exact_string(
                _profile_identity_value(container, label=label, field=field),
                label=label,
                field=field,
                expected=expected,
            )
    return resolved


def _require_v11_fresh_actor_provenance(
    actor_checkpoint: Mapping[str, Any],
    actor_validation: Mapping[str, Any],
) -> Dict[str, Any]:
    """Bind the parent-free origin in both stored and validated metadata."""

    resolved: Dict[str, Any] = {}
    for key, label, container in (
        ("actor_checkpoint", "actor checkpoint", actor_checkpoint),
        (
            "actor_checkpoint_validation",
            "actor checkpoint validation",
            actor_validation,
        ),
    ):
        source = {
            "dynamic_voc_mode": _require_exact_string(
                _profile_identity_value(
                    container, label=label, field="dynamic_voc_mode"
                ),
                label=label,
                field="dynamic_voc_mode",
                expected="control",
            ),
            "voc_control_origin": _require_exact_string(
                _profile_identity_value(
                    container, label=label, field="voc_control_origin"
                ),
                label=label,
                field="voc_control_origin",
                expected="fresh",
            ),
            "voc_control_origin_legacy_defaulted": _require_exact_boolean(
                _profile_identity_value(
                    container,
                    label=label,
                    field="voc_control_origin_legacy_defaulted",
                ),
                label=label,
                field="voc_control_origin_legacy_defaulted",
                expected=False,
            ),
            "voc_activation_real_step": _require_exact_integer(
                _profile_identity_value(
                    container,
                    label=label,
                    field="voc_activation_real_step",
                ),
                label=label,
                field="voc_activation_real_step",
                expected=0,
            ),
        }
        for field in (
            "voc_parent_checkpoint_sha256",
            "voc_parent_checkpoint",
            "voc_parent_imitation_data_signature",
        ):
            source[field] = _require_exact_none(
                _profile_identity_value(container, label=label, field=field),
                label=label,
                field=field,
            )
        resolved[key] = source
    return resolved


def _require_v11_exact_projection_terminal(
    actor_checkpoint: Mapping[str, Any],
) -> Dict[str, Any]:
    """Require the stored gate affine map to be the exact EMA-Q projection.

    This is deliberately a raw parameter check.  Reconstructed per-state Q
    values and probabilities remain rollout diagnostics and are not subjected
    to bit equality.
    """

    embedded = actor_checkpoint.get("flags")
    if not isinstance(embedded, Mapping):
        raise ValueError("actor checkpoint lacks embedded training flags")
    policy_temperature = _require_exact_float(
        _profile_identity_value(
            embedded,
            label="actor checkpoint embedded flags",
            field="voc_gate_temperature",
        ),
        label="actor checkpoint embedded flags",
        field="voc_gate_temperature",
        expected=1.0,
    )
    q_temperature = _require_exact_float(
        _profile_identity_value(
            embedded,
            label="actor checkpoint embedded flags",
            field="voc_gate_q_temperature",
        ),
        label="actor checkpoint embedded flags",
        field="voc_gate_q_temperature",
        expected=0.05,
    )

    actor_state = actor_checkpoint.get("actor_net_state_dict")
    if not isinstance(actor_state, Mapping):
        raise ValueError("actor checkpoint lacks actor_net_state_dict")
    key_pairs = (
        ("voc_gate_head.weight", "voc_gate_head.bias"),
        ("actor.voc_gate_head.weight", "actor.voc_gate_head.bias"),
    )
    matched_pair = next(
        (pair for pair in key_pairs if all(key in actor_state for key in pair)),
        None,
    )
    if matched_pair is None:
        raise ValueError("actor checkpoint lacks stored voc_gate_head weight/bias")
    gate_weight = actor_state[matched_pair[0]]
    gate_bias = actor_state[matched_pair[1]]

    ema_state = actor_checkpoint.get("voc_ema_gate_head_state_dict")
    if not isinstance(ema_state, Mapping) or set(ema_state) != {"weight", "bias"}:
        raise ValueError("actor checkpoint lacks exact EMA Q weight/bias")
    ema_weight = ema_state["weight"]
    ema_bias = ema_state["bias"]
    for label, tensor in (
        ("stored gate weight", gate_weight),
        ("stored gate bias", gate_bias),
        ("stored EMA Q weight", ema_weight),
        ("stored EMA Q bias", ema_bias),
    ):
        if not torch.is_tensor(tensor) or tensor.dtype != torch.float32:
            raise ValueError(f"actor checkpoint {label} must be an FP32 tensor")
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"actor checkpoint {label} contains non-finite values")
    if (
        gate_weight.ndim != 2
        or gate_weight.shape[0] != 1
        or gate_bias.shape != (1,)
        or ema_weight.ndim != 2
        or ema_weight.shape[0] != 2
        or ema_bias.shape != (2,)
        or gate_weight.shape[1:] != ema_weight.shape[1:]
    ):
        raise ValueError(
            "actor checkpoint exact-projection gate/EMA affine shapes disagree"
        )

    scale = float(policy_temperature) / float(q_temperature)
    expected_weight = scale * (ema_weight[0:1] - ema_weight[1:2])
    expected_bias = scale * (ema_bias[0:1] - ema_bias[1:2])
    if not torch.equal(gate_weight, expected_weight):
        raise ValueError(
            "actor checkpoint exact-projection gate weight disagrees with "
            "the raw EMA affine target"
        )
    if not torch.equal(gate_bias, expected_bias):
        raise ValueError(
            "actor checkpoint exact-projection gate bias disagrees with "
            "the raw EMA affine target"
        )
    return {
        "gate_head_keys": list(matched_pair),
        "gate_dtype": str(gate_weight.dtype),
        "ema_q_dtype": str(ema_weight.dtype),
        "voc_gate_temperature": policy_temperature,
        "voc_gate_q_temperature": q_temperature,
        "affine_scale": scale,
        "weight_torch_equal": True,
        "bias_torch_equal": True,
    }


def _optional_profile_identity_value(
    container: Any, *, field: str
) -> Tuple[bool, Any]:
    if isinstance(container, Mapping):
        return (field in container, container.get(field))
    return (hasattr(container, field), getattr(container, field, None))


def _require_legacy_profile_excludes_schema6_identity(
    *,
    profile: ConfirmationProfile,
    flags: Any,
    actor_checkpoint: Mapping[str, Any],
    model_checkpoint: Mapping[str, Any],
    actor_validation: Mapping[str, Any],
) -> None:
    """Reject schema-6 atomic markers without changing legacy defaults."""

    actor_flags = actor_checkpoint.get("flags")
    model_flags = model_checkpoint.get("flags")
    surfaces = (
        (f"{profile.value} checkpoint config", flags),
        ("actor checkpoint embedded flags", actor_flags),
        ("model checkpoint embedded flags", model_flags),
    )
    for label, surface in surfaces:
        for field in (
            "voc_actor_policy_version_barrier",
            "voc_actor_policy_barrier_runtime",
        ):
            present, value = _optional_profile_identity_value(
                surface, field=field
            )
            if present:
                _require_exact_boolean(
                    value,
                    label=label,
                    field=field,
                    expected=False,
                )
        present, value = _optional_profile_identity_value(
            surface, field="voc_gate_execution_epsilon"
        )
        if present:
            _require_exact_float(
                value,
                label=label,
                field="voc_gate_execution_epsilon",
                expected=0.02,
            )
        present, value = _optional_profile_identity_value(
            surface, field="actor_amp_init_scale"
        )
        if present:
            _require_exact_float(
                value,
                label=label,
                field="actor_amp_init_scale",
                expected=256.0,
            )
        present, _ = _optional_profile_identity_value(
            surface, field="voc_actor_policy_bundle_schema_version"
        )
        if present:
            raise ValueError(
                f"{label} legacy profile forbids "
                "voc_actor_policy_bundle_schema_version"
            )
        present, value = _optional_profile_identity_value(
            surface, field="voc_gate_policy_schema_version"
        )
        if present:
            if isinstance(value, bool) or not isinstance(
                value, (int, np.integer)
            ):
                raise ValueError(
                    f"{label} voc_gate_policy_schema_version must be an integer"
                )
            if int(value) in (6, 7):
                raise ValueError(
                    f"{label} legacy profile forbids atomic gate-policy schema"
                )
        if _normalized_optional_seal_schema(surface, label=label) != 0:
            raise ValueError(f"{label} legacy profile forbids seal schema 1")

    voc = actor_validation.get("voc")
    for label, surface in (
        ("actor checkpoint", actor_checkpoint),
        ("actor checkpoint validation", voc),
    ):
        present, value = _optional_profile_identity_value(
            surface, field="voc_gate_policy_schema_version"
        )
        if present:
            if isinstance(value, bool) or not isinstance(
                value, (int, np.integer)
            ):
                raise ValueError(
                    f"{label} voc_gate_policy_schema_version must be an integer"
                )
            if int(value) in (6, 7):
                raise ValueError(
                    f"{label} legacy profile forbids atomic gate-policy schema"
                )
        if _normalized_optional_seal_schema(surface, label=label) != 0:
            raise ValueError(f"{label} legacy profile forbids seal schema 1")


def _require_v13_bundle_evidence(
    evidence: Any,
) -> Dict[str, Any]:
    """Require the JSON-safe result of the frozen authoritative validator."""

    if not isinstance(evidence, Mapping):
        raise ValueError(
            "v13-300k requires authoritative schema-6 final-bundle evidence"
        )
    if evidence.get("authoritative_validator") != (
        "thinker.util.validate_schema6_final_bundle"
    ):
        raise ValueError("v13-300k final-bundle validator identity disagrees")
    resolved = evidence.get("resolved_identity")
    if not isinstance(resolved, Mapping):
        raise ValueError("v13-300k final bundle lacks resolved 228-key identity")
    for field, expected in (
        ("key_count", V13_COMPLETE_IDENTITY_KEY_COUNT),
        ("v12_projection_key_count", V13_V12_PROJECTION_KEY_COUNT),
    ):
        _require_exact_integer(
            resolved.get(field),
            label="v13-300k final bundle",
            field=field,
            expected=expected,
        )
    _require_exact_string(
        resolved.get("v12_projection_sha256"),
        label="v13-300k final bundle",
        field="v12_projection_sha256",
        expected=V13_V12_PROJECTION_SHA256,
    )
    complete_digest = resolved.get("complete_surface_sha256")
    if not isinstance(complete_digest, str) or re.fullmatch(
        r"[0-9a-f]{64}", complete_digest
    ) is None:
        raise ValueError(
            "v13-300k final bundle has invalid complete_surface_sha256"
        )
    stage = resolved.get("stage")
    if not isinstance(stage, (list, tuple)) or tuple(stage) != V13_PRIMARY_STAGE:
        raise ValueError(
            "v13-300k requires exact primary stage "
            f"{V13_PRIMARY_STAGE!r}; got {stage!r}"
        )

    actor_policy = evidence.get("actor_policy")
    if not isinstance(actor_policy, Mapping):
        raise ValueError("v13-300k final bundle lacks actor-policy evidence")
    for field, expected in (
        ("voc_actor_policy_terminal", True),
        ("voc_actor_policy_version_mismatch_count", 0),
        ("voc_actor_policy_malformed_bundle_count", 0),
        ("voc_actor_policy_barrier_timeout_count", 0),
        ("actor_amp_init_scale", 32.0),
        ("actor_amp_skip_count", 0),
        ("actor_amp_consecutive_skips", 0),
    ):
        if actor_policy.get(field) != expected or type(
            actor_policy.get(field)
        ) is not type(expected):
            raise ValueError(
                f"v13-300k final actor evidence requires {field}={expected!r}"
            )
    version_value = actor_policy.get("voc_actor_policy_version")
    publication_value = actor_policy.get(
        "voc_actor_policy_publication_count"
    )
    for field, value in (
        ("voc_actor_policy_version", version_value),
        ("voc_actor_policy_publication_count", publication_value),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(
                f"v13-300k final actor evidence {field} must be an integer"
            )
    version = int(version_value)
    if version < 1 or int(publication_value) != version:
        raise ValueError(
            "v13-300k final actor policy version/count must be positive and lockstep"
        )
    _require_exact_integer(
        actor_policy.get("voc_actor_policy_expected_ack_count"),
        label="v13-300k final actor evidence",
        field="voc_actor_policy_expected_ack_count",
        expected=1,
    )
    _require_exact_integer(
        actor_policy.get("voc_actor_policy_terminal_ack_count"),
        label="v13-300k final actor evidence",
        field="voc_actor_policy_terminal_ack_count",
        expected=1,
    )
    state_digest = actor_policy.get("voc_actor_policy_state_sha256")
    history_digest = actor_policy.get(
        "voc_actor_policy_publication_history_sha256"
    )
    for field, digest in (
        ("voc_actor_policy_state_sha256", state_digest),
        ("voc_actor_policy_publication_history_sha256", history_digest),
    ):
        if not isinstance(digest, str) or re.fullmatch(
            r"[0-9a-f]{64}", digest
        ) is None:
            raise ValueError(f"v13-300k final actor evidence has invalid {field}")
    history = actor_policy.get("voc_actor_policy_publication_history")
    if not isinstance(history, (list, tuple)) or len(history) != version + 1:
        raise ValueError("v13-300k final actor evidence lacks complete history")
    final_event = history[-1]
    if (
        not isinstance(final_event, Mapping)
        or final_event.get("policy_version") != version
        or final_event.get("publication_count") != version
        or final_event.get("terminal") is not True
        or final_event.get("state_sha256") != state_digest
    ):
        raise ValueError("v13-300k final publication event disagrees")

    logger_completion = evidence.get("logger_completion")
    if not isinstance(logger_completion, Mapping):
        raise ValueError("v13-300k final bundle lacks logger completion")
    for field in (
        "required",
        "use_wandb",
        "ack_verified",
        "private_markers_cleaned",
    ):
        if logger_completion.get(field) is not True:
            raise ValueError(
                f"v13-300k logger completion requires {field}=true"
            )
    for logger_field, actor_field in (
        ("policy_version", "voc_actor_policy_version"),
        ("state_sha256", "voc_actor_policy_state_sha256"),
        (
            "publication_history_sha256",
            "voc_actor_policy_publication_history_sha256",
        ),
    ):
        if logger_completion.get(logger_field) != actor_policy.get(actor_field):
            raise ValueError(
                f"v13-300k logger completion {logger_field} disagrees"
            )
    if evidence.get("config_use_wandb") is not True:
        raise ValueError("v13-300k final bundle requires config_use_wandb=true")
    private_markers = evidence.get("private_logger_markers")
    if not isinstance(private_markers, Mapping) or set(private_markers) != set(
        V13_PRIVATE_LOGGER_MARKERS
    ):
        raise ValueError("v13-300k private logger-marker evidence is incomplete")
    for name, record in private_markers.items():
        if not isinstance(record, Mapping) or record.get("absent") is not True:
            raise ValueError(f"v13-300k private logger marker remains: {name}")
    return copy.deepcopy(dict(evidence))


def _require_v14_bundle_evidence(evidence: Any) -> Dict[str, Any]:
    """Require the JSON-safe authoritative schema-7 terminal bundle."""

    if not isinstance(evidence, Mapping):
        raise ValueError(
            "v14-300k requires authoritative schema-7 final-bundle evidence"
        )
    if evidence.get("authoritative_validator") != (
        "thinker.util.validate_schema7_final_bundle"
    ):
        raise ValueError("v14-300k final-bundle validator identity disagrees")
    resolved = evidence.get("resolved_identity")
    if not isinstance(resolved, Mapping):
        raise ValueError("v14-300k final bundle lacks resolved 229-key identity")
    for field, expected in (
        ("gate_schema", 7),
        ("voc_gate_policy_schema_version", 7),
        ("voc_model_input_seal_schema_version", 1),
        ("key_count", V14_COMPLETE_IDENTITY_KEY_COUNT),
        ("v12_projection_key_count", V14_V12_PROJECTION_KEY_COUNT),
    ):
        _require_exact_integer(
            resolved.get(field),
            label="v14-300k final bundle",
            field=field,
            expected=expected,
        )
    _require_exact_string(
        resolved.get("v12_projection_sha256"),
        label="v14-300k final bundle",
        field="v12_projection_sha256",
        expected=V14_V12_PROJECTION_SHA256,
    )
    complete_digest = resolved.get("complete_surface_sha256")
    if not isinstance(complete_digest, str) or re.fullmatch(
        r"[0-9a-f]{64}", complete_digest
    ) is None:
        raise ValueError("v14-300k final bundle has invalid surface digest")
    stage = resolved.get("stage")
    if not isinstance(stage, (list, tuple)) or tuple(stage) != V14_PRIMARY_STAGE:
        raise ValueError(
            f"v14-300k requires exact primary stage {V14_PRIMARY_STAGE!r}"
        )
    stored_surfaces = evidence.get("stored_surface_identity")
    if (
        not isinstance(stored_surfaces, Mapping)
        or set(stored_surfaces)
        != {"config", "actor_checkpoint", "model_checkpoint"}
        or any(value != resolved for value in stored_surfaces.values())
    ):
        raise ValueError("v14-300k lacks exact three-surface identity evidence")

    actor_policy = evidence.get("actor_policy")
    if not isinstance(actor_policy, Mapping):
        raise ValueError("v14-300k final bundle lacks actor-policy evidence")
    bundle_summary = actor_policy.get("voc_actor_policy_bundle_summary")
    if (
        not isinstance(bundle_summary, Mapping)
        or bundle_summary.get("gate_schema") != 7
        or bundle_summary.get("bundle_schema_version") != 1
        or bundle_summary.get("terminal") is not True
    ):
        raise ValueError("v14-300k final actor bundle is not terminal schema 7")
    for field, expected in (
        ("voc_actor_policy_terminal", True),
        ("voc_actor_policy_version_mismatch_count", 0),
        ("voc_actor_policy_malformed_bundle_count", 0),
        ("voc_actor_policy_barrier_timeout_count", 0),
        ("actor_amp_init_scale", 32.0),
        ("actor_amp_skip_count", 0),
        ("actor_amp_consecutive_skips", 0),
    ):
        if actor_policy.get(field) != expected or type(
            actor_policy.get(field)
        ) is not type(expected):
            raise ValueError(
                f"v14-300k final actor evidence requires {field}={expected!r}"
            )
    version = actor_policy.get("voc_actor_policy_version")
    publication_count = actor_policy.get("voc_actor_policy_publication_count")
    if (
        type(version) is not int
        or version < 1
        or type(publication_count) is not int
        or publication_count != version
    ):
        raise ValueError("v14-300k actor policy version/count disagree")
    for field in (
        "voc_actor_policy_expected_ack_count",
        "voc_actor_policy_terminal_ack_count",
    ):
        _require_exact_integer(
            actor_policy.get(field),
            label="v14-300k final actor evidence",
            field=field,
            expected=1,
        )
    state_digest = actor_policy.get("voc_actor_policy_state_sha256")
    history_digest = actor_policy.get(
        "voc_actor_policy_publication_history_sha256"
    )
    for field, digest in (
        ("voc_actor_policy_state_sha256", state_digest),
        ("voc_actor_policy_publication_history_sha256", history_digest),
    ):
        if not isinstance(digest, str) or re.fullmatch(
            r"[0-9a-f]{64}", digest
        ) is None:
            raise ValueError(f"v14-300k actor evidence has invalid {field}")
    history = actor_policy.get("voc_actor_policy_publication_history")
    if not isinstance(history, (list, tuple)) or len(history) != version + 1:
        raise ValueError("v14-300k actor evidence lacks complete history")
    final_event = history[-1]
    if (
        not isinstance(final_event, Mapping)
        or final_event.get("policy_version") != version
        or final_event.get("publication_count") != version
        or final_event.get("terminal") is not True
        or final_event.get("state_sha256") != state_digest
    ):
        raise ValueError("v14-300k final publication event disagrees")

    seal_fields = {
        "voc_model_input_seal_schema_version",
        "voc_model_input_sealed",
        "voc_model_input_seal_count",
        "voc_model_terminal_processed_n",
        "voc_model_terminal_drain_update_count",
        "voc_model_terminal_drain_pre_real_step",
        "voc_model_terminal_drain_pre_grad_step_count_m",
        "voc_model_terminal_drain_pre_grad_step_count_p",
        "voc_model_input_late_write_count",
        "voc_model_input_abort_count",
    }
    seal = evidence.get("model_input_seal")
    if not isinstance(seal, Mapping) or set(seal) != seal_fields:
        raise ValueError("v14-300k lacks exact ModelNet seal evidence")
    if seal.get("voc_model_input_sealed") is not True:
        raise ValueError("v14-300k ModelNet input is not sealed")
    for field in seal_fields - {"voc_model_input_sealed"}:
        if type(seal.get(field)) is not int:
            raise ValueError(f"v14-300k ModelNet seal {field} must be Python int")
    for field, expected in (
        ("voc_model_input_seal_schema_version", 1),
        ("voc_model_input_seal_count", 1),
        ("voc_model_input_late_write_count", 0),
        ("voc_model_input_abort_count", 0),
    ):
        if seal[field] != expected:
            raise ValueError(f"v14-300k ModelNet seal requires {field}={expected}")
    drain = seal["voc_model_terminal_drain_update_count"]
    terminal = seal["voc_model_terminal_processed_n"]
    pre_real = seal["voc_model_terminal_drain_pre_real_step"]
    pre_m = seal["voc_model_terminal_drain_pre_grad_step_count_m"]
    pre_p = seal["voc_model_terminal_drain_pre_grad_step_count_p"]
    if drain not in (0, 1) or min(terminal, pre_real, pre_m, pre_p) < 0:
        raise ValueError("v14-300k ModelNet drain evidence is invalid")
    model_real_step = evidence.get("model_real_step")
    if type(model_real_step) is not int or terminal != model_real_step:
        raise ValueError("v14-300k ModelNet terminal progress disagrees")
    if drain == 0 and pre_real != terminal:
        raise ValueError("v14-300k zero-drain progress disagrees")
    if drain == 1 and pre_real >= terminal:
        raise ValueError("v14-300k one-drain progress did not advance")
    optimizers = evidence.get("model_optimizer_state")
    schedulers = evidence.get("model_scheduler_state")
    if (
        not isinstance(optimizers, Mapping)
        or set(optimizers) != {"m", "p"}
        or not isinstance(schedulers, Mapping)
        or set(schedulers) != {"m", "p"}
    ):
        raise ValueError("v14-300k ModelNet state summaries are incomplete")
    for component, pre_count in (("m", pre_m), ("p", pre_p)):
        optimizer = optimizers[component]
        scheduler = schedulers[component]
        if (
            not isinstance(optimizer, Mapping)
            or type(optimizer.get("expected_step")) is not int
            or optimizer["expected_step"] != pre_count + drain
            or optimizer["expected_step"] <= 0
            or not isinstance(scheduler, Mapping)
            or scheduler.get("last_epoch") != terminal
            or scheduler.get("step_count") != optimizer["expected_step"] + 1
        ):
            raise ValueError(f"v14-300k ModelNet {component} state disagrees")
    if optimizers["m"]["expected_step"] != optimizers["p"]["expected_step"]:
        raise ValueError("v14-300k ModelNet m/p counters are not lockstep")
    if evidence.get("model_scaler_state") != {}:
        raise ValueError("v14-300k FP32 ModelNet unexpectedly has scaler state")

    logger_completion = evidence.get("logger_completion")
    if not isinstance(logger_completion, Mapping):
        raise ValueError("v14-300k lacks logger completion")
    for field in (
        "required",
        "use_wandb",
        "ack_verified",
        "private_markers_cleaned",
    ):
        if logger_completion.get(field) is not True:
            raise ValueError(f"v14-300k logger completion requires {field}=true")
    for logger_field, actor_field in (
        ("policy_version", "voc_actor_policy_version"),
        ("state_sha256", "voc_actor_policy_state_sha256"),
        (
            "publication_history_sha256",
            "voc_actor_policy_publication_history_sha256",
        ),
    ):
        if logger_completion.get(logger_field) != actor_policy.get(actor_field):
            raise ValueError(f"v14-300k logger {logger_field} disagrees")
    if evidence.get("config_use_wandb") is not True:
        raise ValueError("v14-300k requires config_use_wandb=true")
    if evidence.get("private_logger_markers_absent") is not True:
        raise ValueError("v14-300k private logger markers remain")
    private_markers = evidence.get("private_logger_markers")
    if not isinstance(private_markers, Mapping) or set(private_markers) != set(
        V14_PRIVATE_LOGGER_MARKERS
    ):
        raise ValueError("v14-300k private marker evidence is incomplete")
    if any(
        not isinstance(record, Mapping) or record.get("absent") is not True
        for record in private_markers.values()
    ):
        raise ValueError("v14-300k retains a private logger marker")
    if evidence.get("public_finish_verified") is not True:
        raise ValueError("v14-300k public finish was not verified")
    try:
        json.dumps(
            evidence,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("v14-300k final-bundle evidence is not JSON-safe") from error
    return copy.deepcopy(dict(evidence))


def validate_v14_final_bundle(
    checkpoint_dir: str | Path,
    completion_marker: Mapping[str, Any],
    *,
    checkpoint_eval: Any,
) -> Dict[str, Any]:
    """Bind the closed v14 primary to public schema-7 validation."""

    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    if not isinstance(completion_marker, Mapping):
        raise ValueError("v14-300k completion marker must be a mapping")
    validated = checkpoint_eval.validate_schema7_completed_bundle(
        checkpoint_dir, completion_state=completion_marker
    )
    if not isinstance(validated, Mapping):
        raise ValueError("v14-300k checkpoint is not a completed schema-7 bundle")
    private_markers: Dict[str, Any] = {}
    for name in V14_PRIVATE_LOGGER_MARKERS:
        path = checkpoint_dir / name
        if os.path.lexists(path):
            raise RuntimeError(f"v14-300k private logger marker remains: {path}")
        private_markers[name] = {"path": str(path), "absent": True}
    evidence = {
        **copy.deepcopy(dict(validated)),
        "private_logger_markers": private_markers,
    }
    return _require_v14_bundle_evidence(evidence)


def _require_schema89_bundle_evidence(
    evidence: Any,
    *,
    label: str,
    schema_version: int,
    authoritative_validator: str,
    complete_identity_key_count: int,
    projection_key_count: int,
    projection_sha256: str,
    q_regression_loss: str,
    q_reconstruction: Optional[str],
    primary_stage: Tuple[Any, ...],
    private_logger_markers: Sequence[str],
    q_optimizer_coordinates: Optional[str] = None,
    telemetry_required: bool = False,
) -> Dict[str, Any]:
    """Require a JSON-safe authoritative schema-8/9 terminal bundle."""

    if not isinstance(evidence, Mapping):
        raise ValueError(f"{label} requires authoritative final-bundle evidence")
    expected_evidence_fields = {
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
    }
    if telemetry_required:
        expected_evidence_fields.add("telemetry")
    if type(evidence) is not dict or set(evidence) != expected_evidence_fields:
        raise ValueError(f"{label} evidence has the wrong exact top-level shape")
    if evidence.get("authoritative_validator") != authoritative_validator:
        raise ValueError(f"{label} final-bundle validator identity disagrees")
    resolved = evidence.get("resolved_identity")
    expected_resolved_fields = {
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
    if q_reconstruction is not None:
        expected_resolved_fields.add("voc_q_reconstruction")
    if q_optimizer_coordinates is not None:
        expected_resolved_fields.add("voc_q_optimizer_coordinates")
    if not isinstance(resolved, Mapping) or set(resolved) != expected_resolved_fields:
        raise ValueError(f"{label} lacks exact schema-{schema_version} resolved identity")
    for field, expected in (
        ("gate_schema", schema_version),
        ("voc_gate_policy_schema_version", schema_version),
        ("voc_model_input_seal_schema_version", 1),
        ("key_count", complete_identity_key_count),
        ("v12_projection_key_count", projection_key_count),
    ):
        _require_exact_integer(
            resolved.get(field), label=label, field=field, expected=expected
        )
    _require_exact_string(
        resolved.get("voc_q_regression_loss"),
        label=label,
        field="voc_q_regression_loss",
        expected=q_regression_loss,
    )
    if q_reconstruction is not None:
        _require_exact_string(
            resolved.get("voc_q_reconstruction"),
            label=label,
            field="voc_q_reconstruction",
            expected=q_reconstruction,
        )
    if q_optimizer_coordinates is not None:
        _require_exact_string(
            resolved.get("voc_q_optimizer_coordinates"),
            label=label,
            field="voc_q_optimizer_coordinates",
            expected=q_optimizer_coordinates,
        )
    _require_exact_string(
        resolved.get("v12_projection_sha256"),
        label=label,
        field="v12_projection_sha256",
        expected=projection_sha256,
    )
    complete_digest = resolved.get("complete_surface_sha256")
    if not isinstance(complete_digest, str) or re.fullmatch(
        r"[0-9a-f]{64}", complete_digest
    ) is None:
        raise ValueError(f"{label} has invalid complete-surface digest")
    stage = resolved.get("stage")
    if (
        type(stage) not in (list, tuple)
        or len(stage) != len(primary_stage)
        or any(
            type(value) is not type(expected) or value != expected
            for value, expected in zip(stage, primary_stage)
        )
    ):
        raise ValueError(f"{label} requires exact primary stage {primary_stage!r}")
    paths = resolved.get("paths")
    if type(paths) is not dict or set(paths) != {
        "savedir",
        "ckpdir",
        "cmd",
        "icopro_data_path",
    }:
        raise ValueError(f"{label} lacks exact path identity")
    if any(
        type(paths[name]) is not str
        or not paths[name]
        for name in paths
    ):
        raise ValueError(f"{label} path identity requires exact nonempty strings")
    savedir = Path(paths["savedir"])
    ckpdir = Path(paths["ckpdir"])
    data_path = Path(paths["icopro_data_path"])
    xpid = stage[0]
    if (
        not savedir.is_absolute()
        or not ckpdir.is_absolute()
        or not data_path.is_absolute()
        or os.path.normpath(paths["savedir"]) != paths["savedir"]
        or os.path.realpath(paths["savedir"]) != paths["savedir"]
        or os.path.normpath(paths["ckpdir"]) != paths["ckpdir"]
        or os.path.realpath(paths["ckpdir"]) != paths["ckpdir"]
        or os.path.normpath(paths["icopro_data_path"])
        != paths["icopro_data_path"]
        or os.path.realpath(paths["icopro_data_path"])
        != paths["icopro_data_path"]
        or ckpdir.parent != savedir
        or ckpdir.name != xpid
        or data_path
        != savedir.parent / "data" / "behavioral_data_block"
    ):
        raise ValueError(f"{label} path identity relationships disagree")
    stored_surfaces = evidence.get("stored_surface_identity")
    if (
        not isinstance(stored_surfaces, Mapping)
        or set(stored_surfaces)
        != {"config", "actor_checkpoint", "model_checkpoint"}
        or any(value != resolved for value in stored_surfaces.values())
    ):
        raise ValueError(f"{label} lacks exact three-surface identity evidence")

    actor_policy = evidence.get("actor_policy")
    if not isinstance(actor_policy, Mapping):
        raise ValueError(f"{label} lacks actor-policy evidence")
    if schema_version in (11, 13) and (
        set(actor_policy) != ATOMIC_ACTOR_POLICY_EVIDENCE_FIELDS
    ):
        raise ValueError(
            f"{label} actor-policy evidence must preserve the exact "
            "schema-10 lifecycle keyset"
        )
    version = actor_policy.get("voc_actor_policy_version")
    publication_count = actor_policy.get("voc_actor_policy_publication_count")
    summary = actor_policy.get("voc_actor_policy_bundle_summary")
    summary_fields = {
        "bundle_schema_version",
        "policy_version",
        "terminal",
        "gate_schema",
        "actor_state_dict_sha256",
        "actor_state_dict_key_count",
        "actor_state_dict_keys",
        "actor_state_dict_metadata",
    }
    if (
        type(version) is not int
        or version < 1
        or type(publication_count) is not int
        or publication_count != version
        or type(summary) is not dict
        or set(summary) != summary_fields
        or type(summary.get("bundle_schema_version")) is not int
        or summary["bundle_schema_version"] != 1
        or type(summary.get("policy_version")) is not int
        or summary["policy_version"] != version
        or summary.get("terminal") is not True
        or type(summary.get("gate_schema")) is not int
        or summary["gate_schema"] != schema_version
        or type(summary.get("actor_state_dict_sha256")) is not str
        or re.fullmatch(
            r"[0-9a-f]{64}", summary["actor_state_dict_sha256"]
        )
        is None
        or type(summary.get("actor_state_dict_key_count")) is not int
        or summary["actor_state_dict_key_count"] <= 0
        or not isinstance(summary.get("actor_state_dict_keys"), list)
        or len(summary["actor_state_dict_keys"])
        != summary["actor_state_dict_key_count"]
        or any(type(key) is not str or not key for key in summary["actor_state_dict_keys"])
        or len(set(summary["actor_state_dict_keys"]))
        != summary["actor_state_dict_key_count"]
        or not isinstance(summary.get("actor_state_dict_metadata"), list)
        or len(summary["actor_state_dict_metadata"])
        != summary["actor_state_dict_key_count"]
    ):
        raise ValueError(f"{label} final actor bundle identity disagrees")
    for key, metadata in zip(
        summary["actor_state_dict_keys"], summary["actor_state_dict_metadata"]
    ):
        if (
            type(metadata) is not dict
            or set(metadata) != {"key", "dtype", "shape", "numel"}
            or type(metadata.get("key")) is not str
            or metadata["key"] != key
            or type(metadata.get("dtype")) is not str
            or not metadata["dtype"]
            or type(metadata.get("shape")) is not list
            or any(type(size) is not int or size < 0 for size in metadata["shape"])
            or type(metadata.get("numel")) is not int
            or metadata["numel"] < 0
        ):
            raise ValueError(f"{label} actor bundle metadata is malformed")
        if metadata["numel"] != math.prod(metadata["shape"]):
            raise ValueError(f"{label} actor bundle metadata numel disagrees")
    for field, expected in (
        ("voc_actor_policy_terminal", True),
        ("voc_actor_policy_version_mismatch_count", 0),
        ("voc_actor_policy_malformed_bundle_count", 0),
        ("voc_actor_policy_barrier_timeout_count", 0),
        ("voc_actor_policy_expected_ack_count", 1),
        ("voc_actor_policy_terminal_ack_count", 1),
        ("actor_amp_init_scale", 32.0),
        ("actor_amp_skip_count", 0),
        ("actor_amp_consecutive_skips", 0),
    ):
        value = actor_policy.get(field)
        if type(value) is not type(expected) or value != expected:
            raise ValueError(f"{label} actor evidence requires {field}={expected!r}")
    state_digest = actor_policy.get("voc_actor_policy_state_sha256")
    history_digest = actor_policy.get(
        "voc_actor_policy_publication_history_sha256"
    )
    for field, digest in (
        ("voc_actor_policy_state_sha256", state_digest),
        ("voc_actor_policy_publication_history_sha256", history_digest),
    ):
        if not isinstance(digest, str) or re.fullmatch(
            r"[0-9a-f]{64}", digest
        ) is None:
            raise ValueError(f"{label} actor evidence has invalid {field}")
    if summary.get("actor_state_dict_sha256") != state_digest:
        raise ValueError(f"{label} actor bundle state digest disagrees")
    history = actor_policy.get("voc_actor_policy_publication_history")
    event_fields = {
        "predecessor_version",
        "policy_version",
        "publication_count",
        "terminal",
        "ack_ranks",
        "expected_ack_count",
        "state_sha256",
    }
    if not isinstance(history, (list, tuple)) or len(history) != version + 1:
        raise ValueError(f"{label} actor evidence lacks complete history")
    for index, event in enumerate(history):
        if (
            type(event) is not dict
            or set(event) != event_fields
            or type(event.get("predecessor_version")) is not int
            or event["predecessor_version"] != index - 1
            or type(event.get("policy_version")) is not int
            or event["policy_version"] != index
            or type(event.get("publication_count")) is not int
            or event["publication_count"] != index
            or event.get("terminal") is not (index == version)
            or type(event.get("ack_ranks")) is not list
            or len(event["ack_ranks"]) != 1
            or type(event["ack_ranks"][0]) is not int
            or event["ack_ranks"][0] != 0
            or type(event.get("expected_ack_count")) is not int
            or event["expected_ack_count"] != 1
            or type(event.get("state_sha256")) is not str
            or re.fullmatch(r"[0-9a-f]{64}", event["state_sha256"])
            is None
        ):
            raise ValueError(f"{label} publication history is malformed")
    try:
        canonical_history = json.dumps(
            list(history),
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} publication history is not JSON-safe") from error
    if hashlib.sha256(canonical_history).hexdigest() != history_digest:
        raise ValueError(f"{label} publication history digest disagrees")
    if history[-1].get("state_sha256") != state_digest:
        raise ValueError(f"{label} final publication digest disagrees")

    seal = evidence.get("model_input_seal")
    seal_fields = {
        "voc_model_input_seal_schema_version",
        "voc_model_input_sealed",
        "voc_model_input_seal_count",
        "voc_model_terminal_processed_n",
        "voc_model_terminal_drain_update_count",
        "voc_model_terminal_drain_pre_real_step",
        "voc_model_terminal_drain_pre_grad_step_count_m",
        "voc_model_terminal_drain_pre_grad_step_count_p",
        "voc_model_input_late_write_count",
        "voc_model_input_abort_count",
    }
    if not isinstance(seal, Mapping) or set(seal) != seal_fields:
        raise ValueError(f"{label} lacks exact ModelNet seal evidence")
    if seal.get("voc_model_input_sealed") is not True:
        raise ValueError(f"{label} ModelNet input is not sealed")
    for field in seal_fields - {"voc_model_input_sealed"}:
        if type(seal.get(field)) is not int:
            raise ValueError(f"{label} ModelNet seal {field} must be Python int")
    for field, expected in (
        ("voc_model_input_seal_schema_version", 1),
        ("voc_model_input_seal_count", 1),
        ("voc_model_input_late_write_count", 0),
        ("voc_model_input_abort_count", 0),
    ):
        if seal[field] != expected:
            raise ValueError(f"{label} ModelNet seal requires {field}={expected}")
    drain = seal["voc_model_terminal_drain_update_count"]
    terminal = seal["voc_model_terminal_processed_n"]
    pre_real = seal["voc_model_terminal_drain_pre_real_step"]
    pre_m = seal["voc_model_terminal_drain_pre_grad_step_count_m"]
    pre_p = seal["voc_model_terminal_drain_pre_grad_step_count_p"]
    if drain not in (0, 1) or min(terminal, pre_real, pre_m, pre_p) < 0:
        raise ValueError(f"{label} ModelNet drain evidence is invalid")
    if terminal != evidence.get("model_real_step") or terminal < 300_000:
        raise ValueError(f"{label} ModelNet terminal progress disagrees")
    if (drain == 0 and pre_real != terminal) or (
        drain == 1 and pre_real >= terminal
    ):
        raise ValueError(f"{label} ModelNet drain branch disagrees")
    optimizers = evidence.get("model_optimizer_state")
    schedulers = evidence.get("model_scheduler_state")
    if (
        not isinstance(optimizers, Mapping)
        or set(optimizers) != {"m", "p"}
        or not isinstance(schedulers, Mapping)
        or set(schedulers) != {"m", "p"}
    ):
        raise ValueError(f"{label} ModelNet state summaries are incomplete")
    for component, pre_count in (("m", pre_m), ("p", pre_p)):
        optimizer = optimizers[component]
        scheduler = schedulers[component]
        if (
            not isinstance(optimizer, Mapping)
            or optimizer.get("expected_step") != pre_count + drain
            or type(optimizer.get("expected_step")) is not int
            or not isinstance(scheduler, Mapping)
            or scheduler.get("last_epoch") != terminal
            or type(scheduler.get("last_epoch")) is not int
            or scheduler.get("step_count") != optimizer["expected_step"] + 1
            or type(scheduler.get("step_count")) is not int
        ):
            raise ValueError(f"{label} ModelNet {component} state disagrees")
    if optimizers["m"]["expected_step"] != optimizers["p"]["expected_step"]:
        raise ValueError(f"{label} ModelNet m/p counters are not lockstep")
    if evidence.get("model_scaler_state") != {}:
        raise ValueError(f"{label} FP32 ModelNet unexpectedly has scaler state")

    completion_evidence = evidence.get("completion_evidence")
    if type(completion_evidence) is not dict or set(completion_evidence) != {
        "checkpoint_files",
        "implementation_sources",
        "loaded_extensions",
    }:
        raise ValueError(f"{label} completion evidence fields disagree")
    expected_checkpoint_files = (
        SCHEMA13_REQUIRED_CHECKPOINT_FILES
        if telemetry_required
        else REQUIRED_CHECKPOINT_FILES
    )
    checkpoint_records = _require_strict_checkpoint_file_records(
        completion_evidence["checkpoint_files"],
        label=f"{label} completion evidence",
        expected_names=expected_checkpoint_files,
    )
    _require_strict_sha256_file_records(
        completion_evidence["implementation_sources"],
        label=f"{label} completion implementation sources",
    )
    _require_strict_sha256_file_records(
        completion_evidence["loaded_extensions"],
        label=f"{label} completion loaded extensions",
    )

    logger_completion = evidence.get("logger_completion")
    logger_fields = {
        "schema_version",
        "required",
        "use_wandb",
        "request_sha256",
        "ack_verified",
        "private_markers_cleaned",
        "policy_version",
        "state_sha256",
        "publication_history_sha256",
        "checkpoint_files",
    }
    if type(logger_completion) is not dict or set(logger_completion) != logger_fields:
        raise ValueError(f"{label} logger completion fields disagree")
    if (
        type(logger_completion.get("schema_version")) is not int
        or logger_completion["schema_version"] != (2 if telemetry_required else 1)
    ):
        raise ValueError(f"{label} logger completion schema disagrees")
    for field in (
        "required",
        "use_wandb",
        "ack_verified",
        "private_markers_cleaned",
    ):
        if type(logger_completion.get(field)) is not bool or (
            logger_completion[field] is not True
        ):
            raise ValueError(f"{label} logger completion requires {field}=true")
    request_digest = logger_completion.get("request_sha256")
    if type(request_digest) is not str or re.fullmatch(
        r"[0-9a-f]{64}", request_digest
    ) is None:
        raise ValueError(f"{label} logger request digest is invalid")
    logger_checkpoint_records = _require_strict_checkpoint_file_records(
        logger_completion["checkpoint_files"],
        label=f"{label} logger completion",
        expected_names=expected_checkpoint_files,
    )
    if logger_checkpoint_records != checkpoint_records:
        raise ValueError(f"{label} logger checkpoint files disagree")
    for logger_field, actor_field in (
        ("policy_version", "voc_actor_policy_version"),
        ("state_sha256", "voc_actor_policy_state_sha256"),
        (
            "publication_history_sha256",
            "voc_actor_policy_publication_history_sha256",
        ),
    ):
        logger_value = logger_completion.get(logger_field)
        if logger_field == "policy_version":
            valid_type = type(logger_value) is int
        else:
            valid_type = type(logger_value) is str and re.fullmatch(
                r"[0-9a-f]{64}", logger_value
            ) is not None
        if not valid_type or logger_value != actor_policy.get(actor_field):
            raise ValueError(f"{label} logger {logger_field} disagrees")
    if evidence.get("config_use_wandb") is not True:
        raise ValueError(f"{label} requires config_use_wandb=true")
    if evidence.get("private_logger_markers_absent") is not True:
        raise ValueError(f"{label} private marker absence was not verified")
    if evidence.get("public_finish_verified") is not True:
        raise ValueError(f"{label} public finish was not verified")
    if telemetry_required:
        telemetry = evidence.get("telemetry")
        telemetry_fields = {
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
        if type(telemetry) is not dict or set(telemetry) != telemetry_fields:
            raise ValueError(f"{label} telemetry evidence fields disagree")
        for field, expected in (
            ("telemetry_schema_version", 1),
            ("gate_schema", 13),
            ("terminal_policy_version", version),
        ):
            if type(telemetry.get(field)) is not int or telemetry[field] != expected:
                raise ValueError(f"{label} telemetry {field} disagrees")
        if (
            telemetry.get("manifest_name") != "voc_telemetry_manifest.json"
            or type(telemetry.get("manifest_name")) is not str
            or type(telemetry.get("manifest_size")) is not int
            or telemetry["manifest_size"] <= 0
            or type(telemetry.get("transaction_count")) is not int
            or telemetry["transaction_count"]
            != evidence["actor_training_state"].get("voc_update_count")
            or type(telemetry.get("terminal_real_step")) is not int
            or telemetry["terminal_real_step"] < 300_000
        ):
            raise ValueError(f"{label} telemetry manifest/count evidence disagrees")
        for field, expected in (
            (
                "manifest_sha256",
                checkpoint_records["voc_telemetry_manifest.json"]["sha256"],
            ),
            ("actor_state_sha256", state_digest),
            ("publication_history_sha256", history_digest),
        ):
            value = telemetry.get(field)
            if (
                type(value) is not str
                or re.fullmatch(r"[0-9a-f]{64}", value) is None
                or value != expected
            ):
                raise ValueError(f"{label} telemetry {field} disagrees")
        if (
            telemetry["manifest_size"]
            != checkpoint_records["voc_telemetry_manifest.json"]["size"]
        ):
            raise ValueError(f"{label} telemetry manifest size disagrees")
    private_markers = evidence.get("private_logger_markers")
    if type(private_markers) is not dict or set(private_markers) != set(
        private_logger_markers
    ):
        raise ValueError(f"{label} private marker evidence is incomplete")
    for name, record in private_markers.items():
        expected_path = str(ckpdir / name)
        if (
            type(record) is not dict
            or set(record) != {"path", "absent"}
            or type(record.get("path")) is not str
            or record["path"] != expected_path
            or record.get("absent") is not True
        ):
            raise ValueError(f"{label} private logger marker evidence disagrees")
    try:
        json.dumps(
            evidence,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} evidence is not strict JSON-safe") from error
    return copy.deepcopy(dict(evidence))


def _require_v15_bundle_evidence(evidence: Any) -> Dict[str, Any]:
    """Require the JSON-safe authoritative schema-8 terminal bundle."""

    return _require_schema89_bundle_evidence(
        evidence,
        label="v15-300k",
        schema_version=8,
        authoritative_validator="thinker.util.validate_schema8_final_bundle",
        complete_identity_key_count=V15_COMPLETE_IDENTITY_KEY_COUNT,
        projection_key_count=V15_V12_PROJECTION_KEY_COUNT,
        projection_sha256=V15_V12_PROJECTION_SHA256,
        q_regression_loss=V15_Q_REGRESSION_LOSS,
        q_reconstruction=None,
        primary_stage=V15_PRIMARY_STAGE,
        private_logger_markers=V15_PRIVATE_LOGGER_MARKERS,
    )


def _require_v16_bundle_evidence(evidence: Any) -> Dict[str, Any]:
    """Require the JSON-safe authoritative schema-9 terminal bundle."""

    return _require_schema89_bundle_evidence(
        evidence,
        label="v16-300k",
        schema_version=9,
        authoritative_validator="thinker.util.validate_schema9_final_bundle",
        complete_identity_key_count=V16_COMPLETE_IDENTITY_KEY_COUNT,
        projection_key_count=V16_V12_PROJECTION_KEY_COUNT,
        projection_sha256=V16_V12_PROJECTION_SHA256,
        q_regression_loss=V16_Q_REGRESSION_LOSS,
        q_reconstruction=V16_Q_RECONSTRUCTION,
        primary_stage=V16_PRIMARY_STAGE,
        private_logger_markers=V16_PRIVATE_LOGGER_MARKERS,
    )


def _require_v17_bundle_evidence(evidence: Any) -> Dict[str, Any]:
    """Require the JSON-safe authoritative schema-10 terminal bundle."""

    return _require_schema89_bundle_evidence(
        evidence,
        label="v17-300k",
        schema_version=10,
        authoritative_validator="thinker.util.validate_schema10_final_bundle",
        complete_identity_key_count=V17_COMPLETE_IDENTITY_KEY_COUNT,
        projection_key_count=V17_V12_PROJECTION_KEY_COUNT,
        projection_sha256=V17_V12_PROJECTION_SHA256,
        q_regression_loss=V17_Q_REGRESSION_LOSS,
        q_reconstruction=V17_Q_RECONSTRUCTION,
        primary_stage=V17_PRIMARY_STAGE,
        private_logger_markers=V17_PRIVATE_LOGGER_MARKERS,
    )


def _require_v18_bundle_evidence(evidence: Any) -> Dict[str, Any]:
    """Require the JSON-safe authoritative schema-11 terminal bundle."""

    return _require_schema89_bundle_evidence(
        evidence,
        label="v18-300k",
        schema_version=11,
        authoritative_validator="thinker.util.validate_schema11_final_bundle",
        complete_identity_key_count=V18_COMPLETE_IDENTITY_KEY_COUNT,
        projection_key_count=V18_V12_PROJECTION_KEY_COUNT,
        projection_sha256=V18_V12_PROJECTION_SHA256,
        q_regression_loss=V18_Q_REGRESSION_LOSS,
        q_reconstruction=V18_Q_RECONSTRUCTION,
        q_optimizer_coordinates=V18_Q_OPTIMIZER_COORDINATES,
        primary_stage=V18_PRIMARY_STAGE,
        private_logger_markers=V18_PRIVATE_LOGGER_MARKERS,
    )


def _require_v19_bundle_evidence(evidence: Any) -> Dict[str, Any]:
    """Require the JSON-safe authoritative schema-12 terminal bundle."""

    return _require_schema89_bundle_evidence(
        evidence,
        label="v19-300k",
        schema_version=12,
        authoritative_validator="thinker.util.validate_schema12_final_bundle",
        complete_identity_key_count=V19_COMPLETE_IDENTITY_KEY_COUNT,
        projection_key_count=V19_V12_PROJECTION_KEY_COUNT,
        projection_sha256=V19_V12_PROJECTION_SHA256,
        q_regression_loss=V19_Q_REGRESSION_LOSS,
        q_reconstruction=V19_Q_RECONSTRUCTION,
        q_optimizer_coordinates=V19_Q_OPTIMIZER_COORDINATES,
        primary_stage=V19_PRIMARY_STAGE,
        private_logger_markers=V19_PRIVATE_LOGGER_MARKERS,
    )


def _require_v20_bundle_evidence(evidence: Any) -> Dict[str, Any]:
    """Require the JSON-safe authoritative schema-13 terminal bundle."""

    return _require_schema89_bundle_evidence(
        evidence,
        label="v20-300k",
        schema_version=13,
        authoritative_validator="thinker.util.validate_schema13_final_bundle",
        complete_identity_key_count=V20_COMPLETE_IDENTITY_KEY_COUNT,
        projection_key_count=V20_V12_PROJECTION_KEY_COUNT,
        projection_sha256=V20_V12_PROJECTION_SHA256,
        q_regression_loss=V20_Q_REGRESSION_LOSS,
        q_reconstruction=V20_Q_RECONSTRUCTION,
        q_optimizer_coordinates=V20_Q_OPTIMIZER_COORDINATES,
        primary_stage=V20_PRIMARY_STAGE,
        private_logger_markers=V20_PRIVATE_LOGGER_MARKERS,
        telemetry_required=True,
    )


def _require_schema12_fixed_ema_online_equality(
    checkpoint_dir: Path,
    completion_marker: Mapping[str, Any],
    *,
    checkpoint_eval: Any,
) -> None:
    """Independently compare bound raw EMA and online Q before fixed use."""

    checkpoint_files = completion_marker.get("checkpoint_files")
    actor_record = (
        checkpoint_files.get("ckp_actor.tar")
        if isinstance(checkpoint_files, Mapping)
        else None
    )
    if (
        not isinstance(actor_record, Mapping)
        or set(actor_record) != {"sha256", "size"}
    ):
        raise ValueError("v19-300k completion lacks exact actor checkpoint evidence")
    expected_sha = actor_record.get("sha256")
    expected_size = actor_record.get("size")
    if (
        type(expected_sha) is not str
        or len(expected_sha) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha)
        or type(expected_size) is not int
        or expected_size <= 0
    ):
        raise ValueError("v19-300k actor checkpoint evidence is malformed")
    reader = getattr(checkpoint_eval, "_read_stable_single_link_bytes", None)
    if not callable(reader):
        raise RuntimeError("v19-300k validator lacks stable bound byte reader")
    payload = reader(
        checkpoint_dir / "ckp_actor.tar",
        label="v19-300k fixed actor checkpoint",
    )
    if len(payload) != expected_size or hashlib.sha256(payload).hexdigest() != expected_sha:
        raise RuntimeError("v19-300k actor checkpoint disagrees with completion")
    checkpoint = torch.load(
        io.BytesIO(payload), map_location=torch.device("cpu"), weights_only=False
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError("v19-300k actor checkpoint must be a mapping")
    tau = checkpoint.get("voc_gate_target_tau")
    if type(tau) is not float or tau != 1.0:
        raise ValueError("v19-300k actor checkpoint requires exact tau 1.0")
    update_count = checkpoint.get("voc_ema_gate_update_count")
    if type(update_count) is not int or update_count < 0:
        raise ValueError("v19-300k EMA update count must be an integer >= 0")
    if update_count == 0:
        return
    ema_state = checkpoint.get("voc_ema_gate_head_state_dict")
    online_state = checkpoint.get("actor_net_state_dict")
    if not isinstance(ema_state, Mapping) or set(ema_state) != {"weight", "bias"}:
        raise ValueError("v19-300k actor checkpoint lacks exact raw EMA state")
    if not isinstance(online_state, Mapping):
        raise ValueError("v19-300k actor checkpoint lacks online raw Q state")
    online_pair = next(
        (
            pair
            for pair in (
                ("voc_head.weight", "voc_head.bias"),
                ("critic.voc_head.weight", "critic.voc_head.bias"),
            )
            if all(name in online_state for name in pair)
        ),
        None,
    )
    if online_pair is None:
        raise ValueError("v19-300k actor checkpoint lacks online raw Q weight/bias")
    for ema_name, online_name in zip(("weight", "bias"), online_pair):
        ema_tensor = ema_state.get(ema_name)
        online_tensor = online_state.get(online_name)
        if not torch.is_tensor(ema_tensor) or not torch.is_tensor(online_tensor):
            raise ValueError("v19-300k EMA/online equality requires tensors")
        if not torch.equal(ema_tensor, online_tensor):
            raise ValueError(
                f"v19-300k raw EMA {ema_name} must equal online raw Q {ema_name}"
            )


def _require_schema13_fixed_ema_online_equality(
    checkpoint_dir: Path,
    completion_marker: Mapping[str, Any],
    *,
    checkpoint_eval: Any,
) -> None:
    """Independently compare schema-13 bound raw EMA and online Q."""

    checkpoint_files = completion_marker.get("checkpoint_files")
    actor_record = (
        checkpoint_files.get("ckp_actor.tar")
        if isinstance(checkpoint_files, Mapping)
        else None
    )
    if (
        not isinstance(actor_record, Mapping)
        or set(actor_record) != {"sha256", "size"}
    ):
        raise ValueError("v20-300k completion lacks exact actor checkpoint evidence")
    expected_sha = actor_record.get("sha256")
    expected_size = actor_record.get("size")
    if (
        type(expected_sha) is not str
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None
        or type(expected_size) is not int
        or expected_size <= 0
    ):
        raise ValueError("v20-300k actor checkpoint evidence is malformed")
    reader = getattr(checkpoint_eval, "_read_stable_single_link_bytes", None)
    if not callable(reader):
        raise RuntimeError("v20-300k validator lacks stable bound byte reader")
    payload = reader(
        checkpoint_dir / "ckp_actor.tar",
        label="v20-300k fixed actor checkpoint",
    )
    if len(payload) != expected_size or hashlib.sha256(payload).hexdigest() != expected_sha:
        raise RuntimeError("v20-300k actor checkpoint disagrees with completion")
    checkpoint = torch.load(
        io.BytesIO(payload), map_location=torch.device("cpu"), weights_only=False
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError("v20-300k actor checkpoint must be a mapping")
    update_count = checkpoint.get("voc_ema_gate_update_count")
    if type(update_count) is not int or update_count < 0:
        raise ValueError("v20-300k EMA update count must be an integer >= 0")
    if update_count == 0:
        return
    ema_state = checkpoint.get("voc_ema_gate_head_state_dict")
    online_state = checkpoint.get("actor_net_state_dict")
    if not isinstance(ema_state, Mapping) or set(ema_state) != {"weight", "bias"}:
        raise ValueError("v20-300k actor checkpoint lacks exact raw EMA state")
    if not isinstance(online_state, Mapping):
        raise ValueError("v20-300k actor checkpoint lacks online raw Q state")
    online_pair = next(
        (
            pair
            for pair in (
                ("voc_head.weight", "voc_head.bias"),
                ("critic.voc_head.weight", "critic.voc_head.bias"),
            )
            if all(name in online_state for name in pair)
        ),
        None,
    )
    if online_pair is None:
        raise ValueError("v20-300k actor checkpoint lacks online raw Q weight/bias")
    for ema_name, online_name in zip(("weight", "bias"), online_pair):
        ema_tensor = ema_state.get(ema_name)
        online_tensor = online_state.get(online_name)
        if not torch.is_tensor(ema_tensor) or not torch.is_tensor(online_tensor):
            raise ValueError("v20-300k EMA/online equality requires tensors")
        if not torch.equal(ema_tensor, online_tensor):
            raise ValueError(
                f"v20-300k raw EMA {ema_name} must equal online raw Q {ema_name}"
            )


def validate_v15_final_bundle(
    checkpoint_dir: str | Path,
    completion_marker: Mapping[str, Any],
    *,
    checkpoint_eval: Any,
    completed_validation: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Bind the closed v15 primary to public schema-8 validation."""

    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    if not isinstance(completion_marker, Mapping):
        raise ValueError("v15-300k completion marker must be a mapping")
    validated = checkpoint_eval.validate_schema8_completed_bundle(
        checkpoint_dir, completion_state=completion_marker
    )
    if completed_validation is not None and validated != completed_validation:
        raise RuntimeError(
            "v15-300k dispatched and dedicated schema-8 evidence disagree"
        )
    if not isinstance(validated, Mapping):
        raise ValueError("v15-300k checkpoint is not a completed schema-8 bundle")
    private_markers: Dict[str, Any] = {}
    for name in V15_PRIVATE_LOGGER_MARKERS:
        path = checkpoint_dir / name
        if os.path.lexists(path):
            raise RuntimeError(f"v15-300k private logger marker remains: {path}")
        private_markers[name] = {"path": str(path), "absent": True}
    evidence = {
        **copy.deepcopy(dict(validated)),
        "private_logger_markers": private_markers,
    }
    return _require_v15_bundle_evidence(evidence)


def validate_v16_final_bundle(
    checkpoint_dir: str | Path,
    completion_marker: Mapping[str, Any],
    *,
    checkpoint_eval: Any,
    completed_validation: Optional[Mapping[str, Any]] = None,
    config_payload: Optional[bytes] = None,
    expected_config_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Bind the closed v16 primary to public schema-9 validation."""

    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    if not isinstance(completion_marker, Mapping):
        raise ValueError("v16-300k completion marker must be a mapping")
    validated = checkpoint_eval.validate_schema9_completed_bundle(
        checkpoint_dir,
        completion_state=completion_marker,
        config_payload=config_payload,
        expected_config_sha256=expected_config_sha256,
    )
    if completed_validation is not None and validated != completed_validation:
        raise RuntimeError(
            "v16-300k dispatched and dedicated schema-9 evidence disagree"
        )
    if not isinstance(validated, Mapping):
        raise ValueError("v16-300k checkpoint is not a completed schema-9 bundle")
    private_markers: Dict[str, Any] = {}
    for name in V16_PRIVATE_LOGGER_MARKERS:
        path = checkpoint_dir / name
        if os.path.lexists(path):
            raise RuntimeError(f"v16-300k private logger marker remains: {path}")
        private_markers[name] = {"path": str(path), "absent": True}
    evidence = {
        **copy.deepcopy(dict(validated)),
        "private_logger_markers": private_markers,
    }
    return _require_v16_bundle_evidence(evidence)


def validate_v17_final_bundle(
    checkpoint_dir: str | Path,
    completion_marker: Mapping[str, Any],
    *,
    checkpoint_eval: Any,
    completed_validation: Optional[Mapping[str, Any]] = None,
    config_payload: Optional[bytes] = None,
    expected_config_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Bind the closed v17 primary to public schema-10 validation."""

    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    if not isinstance(completion_marker, Mapping):
        raise ValueError("v17-300k completion marker must be a mapping")
    validator = getattr(
        checkpoint_eval, "validate_schema10_completed_bundle", None
    )
    if validator is None:
        raise RuntimeError(
            "v17-300k checkpoint validator lacks schema-10 completed route"
        )
    validated = validator(
        checkpoint_dir,
        completion_state=completion_marker,
        config_payload=config_payload,
        expected_config_sha256=expected_config_sha256,
    )
    if completed_validation is not None and validated != completed_validation:
        raise RuntimeError(
            "v17-300k dispatched and dedicated schema-10 evidence disagree"
        )
    if not isinstance(validated, Mapping):
        raise ValueError("v17-300k checkpoint is not a completed schema-10 bundle")
    private_markers: Dict[str, Any] = {}
    for name in V17_PRIVATE_LOGGER_MARKERS:
        path = checkpoint_dir / name
        if os.path.lexists(path):
            raise RuntimeError(f"v17-300k private logger marker remains: {path}")
        private_markers[name] = {"path": str(path), "absent": True}
    evidence = {
        **copy.deepcopy(dict(validated)),
        "private_logger_markers": private_markers,
    }
    return _require_v17_bundle_evidence(evidence)


def validate_v18_final_bundle(
    checkpoint_dir: str | Path,
    completion_marker: Mapping[str, Any],
    *,
    checkpoint_eval: Any,
    completed_validation: Optional[Mapping[str, Any]] = None,
    config_payload: Optional[bytes] = None,
    expected_config_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Bind the closed v18 primary to public schema-11 validation."""

    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    if not isinstance(completion_marker, Mapping):
        raise ValueError("v18-300k completion marker must be a mapping")
    validator = getattr(
        checkpoint_eval, "validate_schema11_completed_bundle", None
    )
    if validator is None:
        raise RuntimeError(
            "v18-300k checkpoint validator lacks schema-11 completed route"
        )
    validated = validator(
        checkpoint_dir,
        completion_state=completion_marker,
        config_payload=config_payload,
        expected_config_sha256=expected_config_sha256,
    )
    if completed_validation is not None and validated != completed_validation:
        raise RuntimeError(
            "v18-300k dispatched and dedicated schema-11 evidence disagree"
        )
    if not isinstance(validated, Mapping):
        raise ValueError("v18-300k checkpoint is not a completed schema-11 bundle")
    private_markers: Dict[str, Any] = {}
    for name in V18_PRIVATE_LOGGER_MARKERS:
        path = checkpoint_dir / name
        if os.path.lexists(path):
            raise RuntimeError(f"v18-300k private logger marker remains: {path}")
        private_markers[name] = {"path": str(path), "absent": True}
    evidence = {
        **copy.deepcopy(dict(validated)),
        "private_logger_markers": private_markers,
    }
    return _require_v18_bundle_evidence(evidence)


def validate_v19_final_bundle(
    checkpoint_dir: str | Path,
    completion_marker: Mapping[str, Any],
    *,
    checkpoint_eval: Any,
    completed_validation: Optional[Mapping[str, Any]] = None,
    config_payload: Optional[bytes] = None,
    expected_config_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Bind the closed v19 primary to public schema-12 validation."""

    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    if not isinstance(completion_marker, Mapping):
        raise ValueError("v19-300k completion marker must be a mapping")
    validator = getattr(
        checkpoint_eval, "validate_schema12_completed_bundle", None
    )
    if validator is None:
        raise RuntimeError(
            "v19-300k checkpoint validator lacks schema-12 completed route"
        )
    validated = validator(
        checkpoint_dir,
        completion_state=completion_marker,
        config_payload=config_payload,
        expected_config_sha256=expected_config_sha256,
    )
    if completed_validation is not None and validated != completed_validation:
        raise RuntimeError(
            "v19-300k dispatched and dedicated schema-12 evidence disagree"
        )
    if not isinstance(validated, Mapping):
        raise ValueError("v19-300k checkpoint is not a completed schema-12 bundle")
    _require_schema12_fixed_ema_online_equality(
        checkpoint_dir,
        completion_marker,
        checkpoint_eval=checkpoint_eval,
    )
    private_markers: Dict[str, Any] = {}
    for name in V19_PRIVATE_LOGGER_MARKERS:
        path = checkpoint_dir / name
        if os.path.lexists(path):
            raise RuntimeError(f"v19-300k private logger marker remains: {path}")
        private_markers[name] = {"path": str(path), "absent": True}
    evidence = {
        **copy.deepcopy(dict(validated)),
        "private_logger_markers": private_markers,
    }
    return _require_v19_bundle_evidence(evidence)


def validate_v20_final_bundle(
    checkpoint_dir: str | Path,
    completion_marker: Mapping[str, Any],
    *,
    checkpoint_eval: Any,
    completed_validation: Optional[Mapping[str, Any]] = None,
    config_payload: Optional[bytes] = None,
    expected_config_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Bind the closed v20 primary to public schema-13 validation."""

    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    if not isinstance(completion_marker, Mapping):
        raise ValueError("v20-300k completion marker must be a mapping")
    validator = getattr(
        checkpoint_eval, "validate_schema13_completed_bundle", None
    )
    if validator is None:
        raise RuntimeError(
            "v20-300k checkpoint validator lacks schema-13 completed route"
        )
    validated = validator(
        checkpoint_dir,
        completion_state=completion_marker,
        config_payload=config_payload,
        expected_config_sha256=expected_config_sha256,
    )
    if completed_validation is not None and validated != completed_validation:
        raise RuntimeError(
            "v20-300k dispatched and dedicated schema-13 evidence disagree"
        )
    if not isinstance(validated, Mapping):
        raise ValueError("v20-300k checkpoint is not a completed schema-13 bundle")
    _require_schema13_fixed_ema_online_equality(
        checkpoint_dir,
        completion_marker,
        checkpoint_eval=checkpoint_eval,
    )
    private_markers: Dict[str, Any] = {}
    for name in V20_PRIVATE_LOGGER_MARKERS:
        path = checkpoint_dir / name
        if os.path.lexists(path):
            raise RuntimeError(f"v20-300k private logger marker remains: {path}")
        private_markers[name] = {"path": str(path), "absent": True}
    evidence = {
        **copy.deepcopy(dict(validated)),
        "private_logger_markers": private_markers,
    }
    return _require_v20_bundle_evidence(evidence)


def validate_v13_final_bundle(
    checkpoint_dir: str | Path,
    completion_marker: Mapping[str, Any],
) -> Dict[str, Any]:
    """Bind the closed v13 primary to full state and logger completion."""

    from thinker import util  # pylint: disable=import-outside-toplevel

    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    if not isinstance(completion_marker, Mapping):
        raise ValueError("v13-300k completion marker must be a mapping")
    expected_marker_fields = {
        "schema_version",
        "status",
        "completed_unix",
        "checkpoint_files",
        "implementation_sources",
        "loaded_extensions",
        "voc_actor_policy_logger_completion",
    }
    if set(completion_marker) != expected_marker_fields:
        raise ValueError("v13-300k completion marker fields disagree")
    if type(completion_marker.get("schema_version")) is not int or (
        completion_marker["schema_version"] != 1
    ):
        raise ValueError("v13-300k completion marker schema must be integer 1")
    if type(completion_marker.get("status")) is not str or (
        completion_marker["status"] != "complete"
    ):
        raise ValueError("v13-300k completion marker is not complete")
    completed_unix = completion_marker.get("completed_unix")
    if (
        type(completed_unix) not in (int, float)
        or not math.isfinite(float(completed_unix))
        or float(completed_unix) <= 0.0
    ):
        raise ValueError("v13-300k completion timestamp is invalid")
    marker_checkpoint_records = _require_strict_checkpoint_file_records(
        completion_marker.get("checkpoint_files"),
        label="v13-300k completion marker",
    )
    marker_implementation_records = _require_strict_sha256_file_records(
        completion_marker.get("implementation_sources"),
        label="v13-300k completion implementation sources",
    )
    marker_extension_records = _require_strict_sha256_file_records(
        completion_marker.get("loaded_extensions"),
        label="v13-300k completion loaded extensions",
    )

    authoritative = util.validate_schema6_final_bundle(
        str(checkpoint_dir), label="v13-300k authoritative final bundle"
    )
    expected_authoritative_fields = {
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
    }
    if not isinstance(authoritative, Mapping) or set(authoritative) != (
        expected_authoritative_fields
    ):
        raise ValueError("v13-300k authoritative final-bundle fields disagree")
    completion_evidence = authoritative.get("completion_evidence")
    if type(completion_evidence) is not dict or set(completion_evidence) != {
        "checkpoint_files",
        "implementation_sources",
        "loaded_extensions",
    }:
        raise ValueError(
            "v13-300k authoritative completion-evidence fields disagree"
        )
    authoritative_checkpoint_records = _require_strict_checkpoint_file_records(
        completion_evidence.get("checkpoint_files"),
        label="v13-300k authoritative completion evidence",
    )
    authoritative_implementation_records = _require_strict_sha256_file_records(
        completion_evidence.get("implementation_sources"),
        label="v13-300k authoritative implementation sources",
    )
    authoritative_extension_records = _require_strict_sha256_file_records(
        completion_evidence.get("loaded_extensions"),
        label="v13-300k authoritative loaded extensions",
    )
    marker_completion_evidence = {
        "checkpoint_files": marker_checkpoint_records,
        "implementation_sources": marker_implementation_records,
        "loaded_extensions": marker_extension_records,
    }
    normalized_authoritative_evidence = {
        "checkpoint_files": authoritative_checkpoint_records,
        "implementation_sources": authoritative_implementation_records,
        "loaded_extensions": authoritative_extension_records,
    }
    if normalized_authoritative_evidence != marker_completion_evidence:
        raise ValueError(
            "v13-300k completion marker disagrees with authoritative bundle"
        )
    logger_completion = util.validate_actor_policy_logger_completion(
        completion_marker["voc_actor_policy_logger_completion"]
    )
    actor_policy = authoritative.get("actor_policy")
    if not isinstance(actor_policy, Mapping):
        raise ValueError("v13-300k authoritative actor policy is missing")
    logger_checkpoint_records = _require_strict_checkpoint_file_records(
        logger_completion.get("checkpoint_files"),
        label="v13-300k logger completion",
    )
    if logger_checkpoint_records != authoritative_checkpoint_records:
        raise ValueError("v13-300k logger checkpoint files disagree")

    actual_marker_names = (
        getattr(util, "VOC_ACTOR_POLICY_LOGGER_FINISH_REQUEST_FILE", None),
        getattr(util, "VOC_ACTOR_POLICY_LOGGER_FINISH_ACK_FILE", None),
    )
    if actual_marker_names != V13_PRIVATE_LOGGER_MARKERS:
        raise RuntimeError("frozen schema-6 private logger-marker names drifted")
    private_markers: Dict[str, Any] = {}
    for name in V13_PRIVATE_LOGGER_MARKERS:
        path = checkpoint_dir / name
        if os.path.lexists(path):
            raise RuntimeError(
                f"v13-300k private actor-policy logger marker remains: {path}"
            )
        private_markers[name] = {"path": str(path), "absent": True}

    evidence = {
        "authoritative_validator": (
            "thinker.util.validate_schema6_final_bundle"
        ),
        **copy.deepcopy(dict(authoritative)),
        "logger_completion": copy.deepcopy(dict(logger_completion)),
        "finish_marker": {
            "schema_version": 1,
            "status": "complete",
            "completed_unix": float(completed_unix),
            "field_names": sorted(expected_marker_fields),
            "completion_evidence_matches": True,
        },
        "private_logger_markers": private_markers,
    }
    validated = _require_v13_bundle_evidence(evidence)
    try:
        json.dumps(
            validated,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "v13-300k final-bundle evidence is not JSON-safe"
        ) from error
    return validated


def _checkpoint_total_steps(
    checkpoint: Mapping[str, Any], *, label: str, expected: int
) -> int:
    embedded_flags = checkpoint.get("flags")
    if not isinstance(embedded_flags, Mapping) or "total_steps" not in embedded_flags:
        raise ValueError(f"{label} checkpoint lacks embedded flags.total_steps")
    return _require_exact_total_steps(
        embedded_flags["total_steps"],
        label=f"{label} checkpoint embedded flags",
        expected=expected,
    )


def _require_fixed_protocol(
    flags: Any,
    actor_checkpoint: Mapping[str, Any],
    model_checkpoint: Mapping[str, Any],
    actor_validation: Mapping[str, Any],
    model_validation: Mapping[str, Any],
    *,
    confirmation_profile: str | ConfirmationProfile,
    seeds: Sequence[int],
    real_steps_per_seed: int,
    calibration_unroll: int,
    diagnostic: bool,
    schema6_bundle_validation: Optional[Mapping[str, Any]] = None,
    schema7_bundle_validation: Optional[Mapping[str, Any]] = None,
    schema8_bundle_validation: Optional[Mapping[str, Any]] = None,
    schema9_bundle_validation: Optional[Mapping[str, Any]] = None,
    schema10_bundle_validation: Optional[Mapping[str, Any]] = None,
    schema11_bundle_validation: Optional[Mapping[str, Any]] = None,
    schema12_bundle_validation: Optional[Mapping[str, Any]] = None,
    schema13_bundle_validation: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    profile, profile_spec = _resolve_confirmation_profile(confirmation_profile)
    if str(getattr(flags, "name", "")) != "Enduro-v5":
        raise ValueError("fixed confirmation requires Enduro-v5")
    if str(getattr(flags, "dynamic_voc_mode", "")) != "control":
        raise ValueError("fixed confirmation requires dynamic_voc_mode=control")
    required_true = (
        "dynamic_search",
        "dynamic_factorized_control",
        "voc_eval_stochastic",
        "voc_dueling_q",
        "voc_ema_gate_target",
        "voc_dedicated_gate",
        "voc_soft_q_bce_gate",
    )
    for name in required_true:
        if not bool(getattr(flags, name, False)):
            raise ValueError(f"fixed confirmation requires {name}=true")
    if not bool(getattr(flags, "envpool", False)):
        raise ValueError("fixed confirmation requires the EnvPool Atari runtime")
    if not hasattr(flags, "total_steps"):
        raise ValueError("checkpoint config lacks total_steps")
    total_steps = _require_exact_total_steps(
        getattr(flags, "total_steps"),
        label=f"{profile.value} checkpoint config",
        expected=profile_spec.total_steps,
    )
    actor_total_steps = _checkpoint_total_steps(
        actor_checkpoint,
        label="actor",
        expected=profile_spec.total_steps,
    )
    model_total_steps = _checkpoint_total_steps(
        model_checkpoint,
        label="model",
        expected=profile_spec.total_steps,
    )
    actor_embedded_flags = actor_checkpoint.get("flags")
    model_embedded_flags = model_checkpoint.get("flags")
    config_profile_identity = _require_profile_flag_identity(
        flags,
        label=f"{profile.value} checkpoint config",
        profile_spec=profile_spec,
    )
    actor_profile_identity = _require_profile_flag_identity(
        actor_embedded_flags,
        label="actor checkpoint embedded flags",
        profile_spec=profile_spec,
    )
    model_profile_identity = _require_profile_flag_identity(
        model_embedded_flags,
        label="model checkpoint embedded flags",
        profile_spec=profile_spec,
    )
    if profile in (
        ConfirmationProfile.V19_300K,
        ConfirmationProfile.V20_300K,
    ):
        for label, container in (
            (f"{profile.value} checkpoint config", flags),
            ("actor checkpoint embedded flags", actor_embedded_flags),
            ("model checkpoint embedded flags", model_embedded_flags),
        ):
            tau = _profile_identity_value(
                container,
                label=label,
                field="voc_gate_target_tau",
            )
            if (
                type(tau) is not float
                or tau != profile_spec.voc_gate_target_tau
            ):
                raise ValueError(
                    f"{label} voc_gate_target_tau must be exact built-in float 1.0"
                )
    if profile in (
        ConfirmationProfile.V13_300K,
        ConfirmationProfile.V14_300K,
        ConfirmationProfile.V15_300K,
        ConfirmationProfile.V16_300K,
        ConfirmationProfile.V17_300K,
        ConfirmationProfile.V18_300K,
        ConfirmationProfile.V19_300K,
        ConfirmationProfile.V20_300K,
    ):
        expected_atomic_schema = {
            ConfirmationProfile.V13_300K: 6,
            ConfirmationProfile.V14_300K: 7,
            ConfirmationProfile.V15_300K: 8,
            ConfirmationProfile.V16_300K: 9,
            ConfirmationProfile.V17_300K: 10,
            ConfirmationProfile.V18_300K: 11,
            ConfirmationProfile.V19_300K: 12,
            ConfirmationProfile.V20_300K: 13,
        }[profile]
        for label, container, resolved in (
            (
                "v13-300k checkpoint config",
                flags,
                config_profile_identity,
            ),
            (
                "actor checkpoint embedded flags",
                actor_embedded_flags,
                actor_profile_identity,
            ),
            (
                "model checkpoint embedded flags",
                model_embedded_flags,
                model_profile_identity,
            ),
        ):
            resolved["voc_gate_policy_schema_version"] = (
                _require_exact_integer(
                    _profile_identity_value(
                        container,
                        label=label,
                        field="voc_gate_policy_schema_version",
                    ),
                    label=label,
                    field="voc_gate_policy_schema_version",
                    expected=expected_atomic_schema,
                )
            )
    actor_real_step = int(actor_validation["real_step"])
    maximum_overshoot = (
        int(getattr(flags, "self_play_n", 1))
        * int(getattr(flags, "env_n", 1))
        * int(getattr(flags, "actor_unroll_len", 1))
    )
    if not total_steps <= actor_real_step <= total_steps + maximum_overshoot:
        raise ValueError(
            f"actor checkpoint is not the bounded final {profile.value} snapshot: "
            f"real_step={actor_real_step}, allowed=[{total_steps},"
            f"{total_steps + maximum_overshoot}]"
        )
    model_real_step = int(model_validation["real_step"])
    if not total_steps <= model_real_step <= total_steps + maximum_overshoot:
        raise ValueError(
            f"model checkpoint is not the bounded final {profile.value} snapshot: "
            f"real_step={model_real_step}, allowed=[{total_steps},"
            f"{total_steps + maximum_overshoot}]"
        )
    voc = actor_validation.get("voc")
    if not isinstance(voc, Mapping) or not voc.get("voc_ema_gate_head_state_saved"):
        raise ValueError("actor checkpoint lacks validated EMA Q state")
    if profile in (
        ConfirmationProfile.V19_300K,
        ConfirmationProfile.V20_300K,
    ):
        validated_tau = _profile_identity_value(
            voc,
            label="actor checkpoint validation",
            field="voc_gate_target_tau",
        )
        if (
            type(validated_tau) is not float
            or validated_tau != profile_spec.voc_gate_target_tau
        ):
            raise ValueError(
                "actor checkpoint validation voc_gate_target_tau must be "
                "exact built-in float 1.0"
            )
    if profile not in (
        ConfirmationProfile.V13_300K,
        ConfirmationProfile.V14_300K,
        ConfirmationProfile.V15_300K,
        ConfirmationProfile.V16_300K,
        ConfirmationProfile.V17_300K,
        ConfirmationProfile.V18_300K,
        ConfirmationProfile.V19_300K,
        ConfirmationProfile.V20_300K,
    ):
        _require_legacy_profile_excludes_schema6_identity(
            profile=profile,
            flags=flags,
            actor_checkpoint=actor_checkpoint,
            model_checkpoint=model_checkpoint,
            actor_validation=actor_validation,
        )
    v13_bundle_evidence: Mapping[str, Any] = {}
    if profile is ConfirmationProfile.V13_300K:
        v13_bundle_evidence = _require_v13_bundle_evidence(
            schema6_bundle_validation
        )
    v14_bundle_evidence: Mapping[str, Any] = {}
    if profile is ConfirmationProfile.V14_300K:
        v14_bundle_evidence = _require_v14_bundle_evidence(
            schema7_bundle_validation
        )
    v15_bundle_evidence: Mapping[str, Any] = {}
    if profile is ConfirmationProfile.V15_300K:
        v15_bundle_evidence = _require_v15_bundle_evidence(
            schema8_bundle_validation
        )
    v16_bundle_evidence: Mapping[str, Any] = {}
    if profile is ConfirmationProfile.V16_300K:
        v16_bundle_evidence = _require_v16_bundle_evidence(
            schema9_bundle_validation
        )
    v17_bundle_evidence: Mapping[str, Any] = {}
    if profile is ConfirmationProfile.V17_300K:
        v17_bundle_evidence = _require_v17_bundle_evidence(
            schema10_bundle_validation
        )
    v18_bundle_evidence: Mapping[str, Any] = {}
    if profile is ConfirmationProfile.V18_300K:
        v18_bundle_evidence = _require_v18_bundle_evidence(
            schema11_bundle_validation
        )
    v19_bundle_evidence: Mapping[str, Any] = {}
    if profile is ConfirmationProfile.V19_300K:
        v19_bundle_evidence = _require_v19_bundle_evidence(
            schema12_bundle_validation
        )
    v20_bundle_evidence: Mapping[str, Any] = {}
    if profile is ConfirmationProfile.V20_300K:
        v20_bundle_evidence = _require_v20_bundle_evidence(
            schema13_bundle_validation
        )
    normalized_seal_identity = {
        "config": _normalized_optional_seal_schema(
            flags, label=f"{profile.value} checkpoint config"
        ),
        "actor_checkpoint": _normalized_optional_seal_schema(
            actor_embedded_flags, label="actor checkpoint embedded flags"
        ),
        "model_checkpoint": _normalized_optional_seal_schema(
            model_embedded_flags, label="model checkpoint embedded flags"
        ),
        "actor_checkpoint_validation": _normalized_optional_seal_schema(
            voc, label="actor checkpoint validation"
        ),
    }
    expected_seal_schema = (
        1
        if profile
        in (
            ConfirmationProfile.V14_300K,
            ConfirmationProfile.V15_300K,
            ConfirmationProfile.V16_300K,
            ConfirmationProfile.V17_300K,
            ConfirmationProfile.V18_300K,
            ConfirmationProfile.V19_300K,
            ConfirmationProfile.V20_300K,
        )
        else 0
    )
    for source, actual_seal_schema in normalized_seal_identity.items():
        if actual_seal_schema != expected_seal_schema:
            raise ValueError(
                f"{profile.value} requires normalized "
                "voc_model_input_seal_schema_version="
                f"{expected_seal_schema}; {source} resolved to "
                f"{actual_seal_schema}"
            )
    execution_field = "voc_gate_epsilon_greedy_execution"
    normalized_execution_identity = {
        "config": _normalized_optional_boolean(
            flags,
            label=f"{profile.value} checkpoint config",
            field=execution_field,
        ),
        "actor_checkpoint": _normalized_optional_boolean(
            actor_embedded_flags,
            label="actor checkpoint embedded flags",
            field=execution_field,
        ),
        "model_checkpoint": _normalized_optional_boolean(
            model_embedded_flags,
            label="model checkpoint embedded flags",
            field=execution_field,
        ),
        "actor_checkpoint_validation": _normalized_optional_boolean(
            voc,
            label="actor checkpoint validation",
            field=execution_field,
        ),
    }
    expected_execution = profile in (
        ConfirmationProfile.V12_300K,
        ConfirmationProfile.V13_300K,
        ConfirmationProfile.V14_300K,
        ConfirmationProfile.V15_300K,
        ConfirmationProfile.V16_300K,
        ConfirmationProfile.V17_300K,
        ConfirmationProfile.V18_300K,
        ConfirmationProfile.V19_300K,
        ConfirmationProfile.V20_300K,
    )
    for source, actual_execution in normalized_execution_identity.items():
        if actual_execution is not expected_execution:
            raise ValueError(
                f"{profile.value} requires normalized {execution_field}="
                f"{str(expected_execution).lower()}; {source} resolved to "
                f"{str(actual_execution).lower()}"
            )
    gate_schema_version = None
    validated_gate_schema_version = None
    validated_param_align = None
    validated_param_align_coef = None
    validated_exact_projection = None
    validated_epsilon_greedy_execution = None
    if profile_spec.voc_gate_policy_schema_version is not None:
        gate_schema_version = _require_exact_integer(
            _profile_identity_value(
                actor_checkpoint,
                label="actor checkpoint",
                field="voc_gate_policy_schema_version",
            ),
            label="actor checkpoint",
            field="voc_gate_policy_schema_version",
            expected=profile_spec.voc_gate_policy_schema_version,
        )
        validated_gate_schema_version = _require_exact_integer(
            _profile_identity_value(
                voc,
                label="actor checkpoint validation",
                field="voc_gate_policy_schema_version",
            ),
            label="actor checkpoint validation",
            field="voc_gate_policy_schema_version",
            expected=profile_spec.voc_gate_policy_schema_version,
        )
        validated_param_align = _require_exact_boolean(
            _profile_identity_value(
                voc,
                label="actor checkpoint validation",
                field="voc_gate_param_align",
            ),
            label="actor checkpoint validation",
            field="voc_gate_param_align",
            expected=profile_spec.voc_gate_param_align,
        )
        validated_param_align_coef = _require_exact_float(
            _profile_identity_value(
                voc,
                label="actor checkpoint validation",
                field="voc_gate_param_align_coef",
            ),
            label="actor checkpoint validation",
            field="voc_gate_param_align_coef",
            expected=profile_spec.voc_gate_param_align_coef,
        )
        if profile_spec.voc_gate_exact_projection is not None:
            validated_exact_projection = _require_exact_boolean(
                _profile_identity_value(
                    voc,
                    label="actor checkpoint validation",
                    field="voc_gate_exact_projection",
                ),
                label="actor checkpoint validation",
                field="voc_gate_exact_projection",
                expected=profile_spec.voc_gate_exact_projection,
            )
        if profile_spec.voc_gate_epsilon_greedy_execution is not None:
            validated_epsilon_greedy_execution = _require_exact_boolean(
                _profile_identity_value(
                    voc,
                    label="actor checkpoint validation",
                    field="voc_gate_epsilon_greedy_execution",
                ),
                label="actor checkpoint validation",
                field="voc_gate_epsilon_greedy_execution",
                expected=profile_spec.voc_gate_epsilon_greedy_execution,
            )
    ema_state = actor_checkpoint.get("voc_ema_gate_head_state_dict")
    if not isinstance(ema_state, Mapping) or set(ema_state) != {"weight", "bias"}:
        raise ValueError("actor checkpoint lacks exact EMA Q weight/bias")
    exact_projection_fresh_provenance: Mapping[str, Any] = {}
    exact_projection_terminal: Mapping[str, Any] = {}
    exact_projection_profile = profile in (
        ConfirmationProfile.V11_300K,
        ConfirmationProfile.V12_300K,
        ConfirmationProfile.V13_300K,
        ConfirmationProfile.V14_300K,
        ConfirmationProfile.V15_300K,
        ConfirmationProfile.V16_300K,
        ConfirmationProfile.V17_300K,
        ConfirmationProfile.V18_300K,
        ConfirmationProfile.V19_300K,
        ConfirmationProfile.V20_300K,
    )
    if exact_projection_profile:
        exact_projection_fresh_provenance = _require_v11_fresh_actor_provenance(
            actor_checkpoint, voc
        )
        exact_projection_terminal = _require_v11_exact_projection_terminal(
            actor_checkpoint
        )
    training_seed_start = int(getattr(flags, "base_seed", 0))
    training_streams = int(getattr(flags, "self_play_n", 1)) * int(
        getattr(flags, "env_n", 1)
    )
    training_seeds = set(range(training_seed_start, training_seed_start + training_streams))
    overlap = sorted(training_seeds & set(int(seed) for seed in seeds))
    if overlap:
        raise ValueError(f"fixed evaluation seeds overlap training streams: {overlap}")
    exact_confirmation = (
        tuple(int(seed) for seed in seeds)
        == tuple(
            range(DEFAULT_SEED_BASE, DEFAULT_SEED_BASE + DEFAULT_NUM_SEEDS)
        )
        and int(real_steps_per_seed) == DEFAULT_REAL_STEPS_PER_SEED
        and int(calibration_unroll) == DEFAULT_CALIBRATION_UNROLL
    )
    if not exact_confirmation and not diagnostic:
        raise ValueError(
            "non-preregistered seed/horizon settings require --diagnostic; "
            "fixed confirmation is exactly seeds 20260827..20260842, "
            "6250 real steps per seed, calibration unroll 201"
        )
    evaluation_mode = "diagnostic" if diagnostic else profile_spec.evaluation_mode
    return {
        "confirmation_profile": profile.value,
        "voc_gate_epsilon_greedy_execution": expected_execution,
        "normalized_execution_identity": normalized_execution_identity,
        **(
            {"normalized_model_input_seal_identity": normalized_seal_identity}
            if profile
            in (
                ConfirmationProfile.V14_300K,
                ConfirmationProfile.V15_300K,
                ConfirmationProfile.V16_300K,
                ConfirmationProfile.V17_300K,
                ConfirmationProfile.V18_300K,
                ConfirmationProfile.V19_300K,
                ConfirmationProfile.V20_300K,
            )
            else {}
        ),
        "resolved_profile_identity": {
            "config": {
                "total_steps": total_steps,
                **config_profile_identity,
            },
            "actor_checkpoint": {
                "total_steps": actor_total_steps,
                **actor_profile_identity,
                **(
                    {"voc_gate_policy_schema_version": gate_schema_version}
                    if gate_schema_version is not None
                    else {}
                ),
            },
            "model_checkpoint": {
                "total_steps": model_total_steps,
                **model_profile_identity,
            },
            "actor_checkpoint_validation": (
                {
                    "voc_gate_policy_schema_version": validated_gate_schema_version,
                    "voc_gate_param_align": validated_param_align,
                    "voc_gate_param_align_coef": validated_param_align_coef,
                    **(
                        {
                            "voc_gate_exact_projection": (
                                validated_exact_projection
                            )
                        }
                        if validated_exact_projection is not None
                        else {}
                    ),
                    **(
                        {
                            "voc_gate_epsilon_greedy_execution": (
                                validated_epsilon_greedy_execution
                            )
                        }
                        if validated_epsilon_greedy_execution is not None
                        else {}
                    ),
                    **(
                        {
                            "voc_model_input_seal_schema_version": (
                                normalized_seal_identity[
                                    "actor_checkpoint_validation"
                                ]
                            )
                        }
                        if profile
                        in (
                            ConfirmationProfile.V14_300K,
                            ConfirmationProfile.V15_300K,
                            ConfirmationProfile.V16_300K,
                            ConfirmationProfile.V17_300K,
                            ConfirmationProfile.V18_300K,
                            ConfirmationProfile.V19_300K,
                            ConfirmationProfile.V20_300K,
                        )
                        else {}
                    ),
                }
                if validated_gate_schema_version is not None
                else {}
            ),
            **(
                {
                    "fresh_actor_provenance": dict(
                        exact_projection_fresh_provenance
                    ),
                    "terminal_exact_projection": dict(
                        exact_projection_terminal
                    ),
                }
                if exact_projection_profile
                else {}
            ),
            **(
                {
                    "schema6_final_bundle": copy.deepcopy(
                        dict(v13_bundle_evidence)
                    )
                }
                if profile is ConfirmationProfile.V13_300K
                else {}
            ),
            **(
                {
                    "schema7_final_bundle": copy.deepcopy(
                        dict(v14_bundle_evidence)
                    )
                }
                if profile is ConfirmationProfile.V14_300K
                else {}
            ),
            **(
                {
                    "schema8_final_bundle": copy.deepcopy(
                        dict(v15_bundle_evidence)
                    ),
                    "voc_q_regression_loss": V15_Q_REGRESSION_LOSS,
                }
                if profile is ConfirmationProfile.V15_300K
                else {}
            ),
            **(
                {
                    "schema9_final_bundle": copy.deepcopy(
                        dict(v16_bundle_evidence)
                    ),
                    "voc_q_regression_loss": V16_Q_REGRESSION_LOSS,
                    "voc_q_reconstruction": V16_Q_RECONSTRUCTION,
                }
                if profile is ConfirmationProfile.V16_300K
                else {}
            ),
            **(
                {
                    "schema10_final_bundle": copy.deepcopy(
                        dict(v17_bundle_evidence)
                    ),
                    "voc_q_regression_loss": V17_Q_REGRESSION_LOSS,
                    "voc_q_reconstruction": V17_Q_RECONSTRUCTION,
                }
                if profile is ConfirmationProfile.V17_300K
                else {}
            ),
            **(
                {
                    "schema11_final_bundle": copy.deepcopy(
                        dict(v18_bundle_evidence)
                    ),
                    "voc_q_regression_loss": V18_Q_REGRESSION_LOSS,
                    "voc_q_reconstruction": V18_Q_RECONSTRUCTION,
                    "voc_q_optimizer_coordinates": (
                        V18_Q_OPTIMIZER_COORDINATES
                    ),
                }
                if profile is ConfirmationProfile.V18_300K
                else {}
            ),
            **(
                {
                    "schema12_final_bundle": copy.deepcopy(
                        dict(v19_bundle_evidence)
                    ),
                    "voc_q_regression_loss": V19_Q_REGRESSION_LOSS,
                    "voc_q_reconstruction": V19_Q_RECONSTRUCTION,
                    "voc_q_optimizer_coordinates": (
                        V19_Q_OPTIMIZER_COORDINATES
                    ),
                }
                if profile is ConfirmationProfile.V19_300K
                else {}
            ),
            **(
                {
                    "schema13_final_bundle": copy.deepcopy(
                        dict(v20_bundle_evidence)
                    ),
                    "voc_q_regression_loss": V20_Q_REGRESSION_LOSS,
                    "voc_q_reconstruction": V20_Q_RECONSTRUCTION,
                    "voc_q_optimizer_coordinates": (
                        V20_Q_OPTIMIZER_COORDINATES
                    ),
                }
                if profile is ConfirmationProfile.V20_300K
                else {}
            ),
        },
        "evaluation_mode": evaluation_mode,
        "confirmation_eligible": bool(exact_confirmation and not diagnostic),
        "exact_preregistered_seed_horizon": exact_confirmation,
        "training_total_steps": total_steps,
        "actor_final_real_step": actor_real_step,
        "allowed_actor_overshoot": maximum_overshoot,
        "model_final_real_step": model_real_step,
        "allowed_model_overshoot": maximum_overshoot,
        "heldout_seed_overlap": [],
        **(
            {
                "training_gate_soft_epsilon": 0.02,
                "training_gate_execution_epsilon": 0.25,
                "runtime_gate_soft_epsilon": 0.0,
                "runtime_gate_execution_epsilon": 0.0,
                "training_actor_policy_version_barrier": True,
                "runtime_actor_policy_barrier_wait": False,
            }
            if profile in (
                ConfirmationProfile.V13_300K,
                ConfirmationProfile.V14_300K,
                ConfirmationProfile.V15_300K,
                ConfirmationProfile.V16_300K,
                ConfirmationProfile.V17_300K,
                ConfirmationProfile.V18_300K,
                ConfirmationProfile.V19_300K,
                ConfirmationProfile.V20_300K,
            )
            else {}
        ),
        **(
            {
                "training_model_input_seal_schema_version": 1,
                "runtime_model_input_seal_coordination": False,
            }
            if profile
            in (
                ConfirmationProfile.V14_300K,
                ConfirmationProfile.V15_300K,
                ConfirmationProfile.V16_300K,
                ConfirmationProfile.V17_300K,
                ConfirmationProfile.V18_300K,
                ConfirmationProfile.V19_300K,
                ConfirmationProfile.V20_300K,
            )
            else {}
        ),
    }


def _require_fixed_200k_protocol(
    flags: Any,
    actor_checkpoint: Mapping[str, Any],
    model_checkpoint: Mapping[str, Any],
    actor_validation: Mapping[str, Any],
    model_validation: Mapping[str, Any],
    *,
    seeds: Sequence[int],
    real_steps_per_seed: int,
    calibration_unroll: int,
    diagnostic: bool,
) -> Dict[str, Any]:
    """Legacy v7 protocol entry point with the hardened bundle checks."""

    return _require_fixed_protocol(
        flags,
        actor_checkpoint,
        model_checkpoint,
        actor_validation,
        model_validation,
        confirmation_profile=ConfirmationProfile.V7_200K,
        seeds=seeds,
        real_steps_per_seed=real_steps_per_seed,
        calibration_unroll=calibration_unroll,
        diagnostic=diagnostic,
    )


def _evaluation_gate_protocol(
    protocol_state: Mapping[str, Any],
) -> Dict[str, Any]:
    """Describe gate execution from the validated state, never a profile label."""

    execution = protocol_state.get("voc_gate_epsilon_greedy_execution")
    if not isinstance(execution, bool):
        raise RuntimeError(
            "fixed protocol lacks validated voc_gate_epsilon_greedy_execution"
        )
    if not execution:
        return {
            "stochastic_gate": True,
            "gate_sampling": "checkpoint gate, epsilon zero, fixed RNG",
        }
    result = {
        "stochastic_gate": False,
        "gate_sampling": (
            "projected gate sign, epsilon zero, deterministic non-ties; "
            "exact zero ties sampled 0.5 with fixed RNG"
        ),
        "gate_execution": "epsilon_greedy",
        "gate_non_tie_argmax": True,
        "gate_exact_zero_tie_sampling_probability": 0.5,
        "gate_behavior_logits_source": "search_control_logits",
        "gate_probability_and_calibration_source": (
            "ActorOut.misc['voc_gate_soft_continue_probability']"
        ),
        "gate_soft_logit_validation_source": (
            "ActorOut.misc['voc_gate_soft_control_logits']"
        ),
    }
    atomic_runtime_fields = {
        field: protocol_state[field]
        for field in (
            "training_gate_soft_epsilon",
            "training_gate_execution_epsilon",
            "runtime_gate_soft_epsilon",
            "runtime_gate_execution_epsilon",
            "training_actor_policy_version_barrier",
            "runtime_actor_policy_barrier_wait",
        )
        if field in protocol_state
    }
    if atomic_runtime_fields:
        if set(atomic_runtime_fields) != {
            "training_gate_soft_epsilon",
            "training_gate_execution_epsilon",
            "runtime_gate_soft_epsilon",
            "runtime_gate_execution_epsilon",
            "training_actor_policy_version_barrier",
            "runtime_actor_policy_barrier_wait",
        }:
            raise RuntimeError("incomplete validated atomic runtime gate protocol")
        result.update(atomic_runtime_fields)
    seal_runtime_fields = {
        field: protocol_state[field]
        for field in (
            "training_model_input_seal_schema_version",
            "runtime_model_input_seal_coordination",
        )
        if field in protocol_state
    }
    if seal_runtime_fields:
        if seal_runtime_fields != {
            "training_model_input_seal_schema_version": 1,
            "runtime_model_input_seal_coordination": False,
        }:
            raise RuntimeError("incomplete validated schema-7 runtime seal protocol")
        result.update(seal_runtime_fields)
    return result


def _runtime_flags(flags: Any, runtime_dir: Path) -> Any:
    runtime = copy.deepcopy(flags)
    runtime.train_actor = False
    runtime.train_model = False
    runtime.parallel = False
    runtime.parallel_actor = False
    runtime.ckp = False
    runtime.use_wandb = False
    runtime.ckpdir = str(runtime_dir)
    runtime.savedir = str(runtime_dir)
    # These paths are inert with load_net=False, but clearing them makes the
    # evaluation contract explicit and prevents an accidental fallback load.
    runtime.preload = ""
    runtime.preload_actor = ""
    runtime.voc_parent_checkpoint = ""
    if hasattr(runtime, "voc_actor_policy_barrier_runtime"):
        # Persisted schema-6 identity remains true.  Only this private deepcopy
        # disables the online actor-publication wait path for offline rollout.
        runtime.voc_actor_policy_barrier_runtime = False
    return runtime


def _update_effective_tokens(env_out: Any, next_env_out: Any, info: Mapping[str, Any]) -> Any:
    """Preserve accepted tokens on WAIT exactly as online SelfPlayWorker does."""

    accepted_primary = info.get("accepted_primary_action")
    real_transition = info.get("real_transition")
    if accepted_primary is not None:
        invalid = accepted_primary < 0
        if real_transition is not None:
            invalid = invalid & ~real_transition.bool()
        if torch.any(invalid):
            last_pri = next_env_out.last_pri.clone()
            mask = invalid
            if last_pri.ndim > mask.ndim + 1:
                mask = mask.unsqueeze(-1)
            last_pri[0] = torch.where(mask, env_out.last_pri[0], last_pri[0])
            next_env_out = next_env_out._replace(last_pri=last_pri)
    accepted_control = info.get("accepted_control")
    if accepted_control is not None and next_env_out.last_search_control is not None:
        invalid = accepted_control < 0
        if torch.any(invalid):
            last_control = next_env_out.last_search_control.clone()
            last_control[0] = torch.where(
                invalid, env_out.last_search_control[0], last_control[0]
            )
            next_env_out = next_env_out._replace(last_search_control=last_control)
    return next_env_out


def _actor_eval_tensors(
    actor_out: Any,
    ema_weight: torch.Tensor,
    ema_bias: torch.Tensor,
    *,
    reward_names: Sequence[str],
    think_cost: float,
    epsilon_greedy_execution: bool = False,
) -> Dict[str, torch.Tensor]:
    behavior_logits = actor_out.search_control_logits
    if (
        not torch.is_tensor(behavior_logits)
        or not torch.is_floating_point(behavior_logits)
        or behavior_logits.ndim < 2
        or behavior_logits.shape[-1] != 3
    ):
        raise RuntimeError(
            "fixed evaluation requires floating search_control_logits ending in 3"
        )
    if not torch.isfinite(behavior_logits).all().item():
        raise FloatingPointError("non-finite fixed-evaluation behavior logits")

    probability_logits = behavior_logits
    soft_continue_probability = None
    execution_continue_probability = None
    raw_gate_log_odds = None
    if epsilon_greedy_execution:
        misc = getattr(actor_out, "misc", None)
        if not isinstance(misc, Mapping):
            raise RuntimeError(
                "epsilon-greedy fixed evaluation requires ActorOut.misc"
            )
        required = {
            "voc_gate_soft_control_logits": tuple(behavior_logits.shape),
            "voc_gate_soft_continue_probability": tuple(
                behavior_logits.shape[:-1]
            ),
            "voc_gate_execution_continue_probability": tuple(
                behavior_logits.shape[:-1]
            ),
            "voc_gate_log_odds": tuple(behavior_logits.shape[:-1]),
        }
        tensors: Dict[str, torch.Tensor] = {}
        for name, expected_shape in required.items():
            value = misc.get(name)
            if (
                not torch.is_tensor(value)
                or not torch.is_floating_point(value)
                or tuple(value.shape) != expected_shape
                or value.device != behavior_logits.device
            ):
                raise RuntimeError(
                    f"epsilon-greedy fixed evaluation requires floating misc[{name!r}] "
                    f"with shape {expected_shape} on the behavior-logit device"
                )
            if not torch.isfinite(value).all().item():
                raise FloatingPointError(
                    f"non-finite epsilon-greedy fixed-evaluation misc[{name!r}]"
                )
            tensors[name] = value

        probability_logits = tensors["voc_gate_soft_control_logits"]
        soft_probability = torch.softmax(
            probability_logits.float(), dim=-1
        )[..., :2].sum(dim=-1)
        recorded_soft_probability = tensors[
            "voc_gate_soft_continue_probability"
        ].float()
        if not torch.allclose(
            soft_probability,
            recorded_soft_probability,
            rtol=1e-6,
            atol=1e-7,
        ):
            raise RuntimeError(
                "epsilon-greedy soft gate logits/probability disagree"
            )
        soft_continue_probability = recorded_soft_probability

        reconstructed_execution_probability = torch.softmax(
            behavior_logits.float(), dim=-1
        )[..., :2].sum(dim=-1)
        recorded_execution_probability = tensors[
            "voc_gate_execution_continue_probability"
        ].float()
        if not torch.allclose(
            reconstructed_execution_probability,
            recorded_execution_probability,
            rtol=1e-6,
            atol=1e-7,
        ):
            raise RuntimeError(
                "epsilon-greedy behavior logits/execution probability disagree"
            )
        execution_continue_probability = recorded_execution_probability
        raw_gate_log_odds = tensors["voc_gate_log_odds"].float()

    logits = probability_logits[-1].float()
    probabilities = torch.softmax(logits, dim=-1)
    continue_probability = (
        soft_continue_probability[-1]
        if epsilon_greedy_execution
        else probabilities[:, :2].sum(dim=-1)
    )
    features = actor_out.voc_features[-1].float()
    raw_ema = torch.nn.functional.linear(features, ema_weight, ema_bias)
    task_index = reward_names.index("re")
    think_index = reward_names.index("think")
    baseline = actor_out.baseline[-1].float()
    state_value = baseline[:, task_index] + think_cost * baseline[:, think_index]
    ema_q = reconstruct_dueling_q(raw_ema, state_value, continue_probability)
    online_q = reconstruct_dueling_q(
        actor_out.voc_q[-1].float(), state_value, continue_probability
    )
    control = actor_out.search_control[-1].long()
    control_valid = actor_out.control_valid[-1].bool()
    gate_action = torch.where(
        control == STOP,
        torch.full_like(control, GATE_STOP),
        torch.full_like(control, GATE_CONTINUE),
    )
    if epsilon_greedy_execution:
        execution_probability = execution_continue_probability[-1]
        raw_log_odds = raw_gate_log_odds[-1]
        allowed_execution_probability = (
            (execution_probability == 0.0)
            | (execution_probability == 0.5)
            | (execution_probability == 1.0)
        )
        if not torch.all(allowed_execution_probability[control_valid]).item():
            raise RuntimeError(
                "epsilon-greedy epsilon-zero execution probability must be "
                "0, 0.5, or 1"
            )
        both_gate_actions_legal = (
            (continue_probability > 0.0) & (continue_probability < 1.0)
        )
        sign_check = control_valid & both_gate_actions_legal
        expected_execution_probability = torch.where(
            raw_log_odds > 0.0,
            torch.ones_like(raw_log_odds),
            torch.where(
                raw_log_odds < 0.0,
                torch.zeros_like(raw_log_odds),
                torch.full_like(raw_log_odds, 0.5),
            ),
        )
        if not torch.equal(
            execution_probability[sign_check],
            expected_execution_probability[sign_check],
        ):
            raise RuntimeError(
                "epsilon-greedy epsilon-zero execution disagrees with raw gate sign"
            )
        selected_continue = gate_action == GATE_CONTINUE
        impossible_sample = control_valid & (
            ((execution_probability == 1.0) & ~selected_continue)
            | ((execution_probability == 0.0) & selected_continue)
        )
        if torch.any(impossible_sample).item():
            raise RuntimeError(
                "epsilon-greedy sampled gate action has zero execution probability"
            )
    selected_index = gate_action.unsqueeze(-1)
    return {
        "continue_probability": continue_probability,
        "ema_q": ema_q,
        "online_q": online_q,
        "ema_selected_q": torch.gather(ema_q, -1, selected_index).squeeze(-1),
        "online_selected_q": torch.gather(online_q, -1, selected_index).squeeze(-1),
        "state_value": state_value,
        "task_value": baseline[:, task_index],
        "think_value": baseline[:, think_index],
        "control": control,
        "gate_action": gate_action,
        "control_valid": control_valid,
    }


def _flush_calibration_chunk(
    chunk: List[MutableMapping[str, Any]],
    *,
    bootstrap_task: torch.Tensor,
    bootstrap_think: torch.Tensor,
    lamb: float,
    think_cost: float,
) -> None:
    if not chunk:
        return

    def stack(name: str) -> torch.Tensor:
        return torch.stack([record[name] for record in chunk], dim=0)

    task_target = on_policy_vtrace_target(
        stack("task_reward"),
        stack("task_discount"),
        stack("task_value"),
        bootstrap_task,
        lamb=lamb,
    )
    think_target = on_policy_vtrace_target(
        stack("think_reward"),
        stack("think_discount"),
        stack("think_value"),
        bootstrap_think,
        lamb=lamb,
    )
    net_target = task_target + think_cost * think_target
    ema_td = net_target - stack("ema_selected_q")
    online_td = net_target - stack("online_selected_q")
    for time_index, record in enumerate(chunk):
        for stream_index, row in enumerate(record["rows"]):
            if row is None:
                continue
            values = {
                "calibration_task_target": task_target[time_index, stream_index],
                "calibration_think_target": think_target[time_index, stream_index],
                "calibration_net_target": net_target[time_index, stream_index],
                "ema_td_error": ema_td[time_index, stream_index],
                "online_td_error": online_td[time_index, stream_index],
            }
            for name, value in values.items():
                scalar = float(value.detach().cpu())
                if not math.isfinite(scalar):
                    raise FloatingPointError(f"non-finite {name}")
                row[name] = scalar
    chunk.clear()


def _build_actor_and_environment(
    flags: Any,
    actor_checkpoint: Mapping[str, Any],
    model_checkpoint: Mapping[str, Any],
    *,
    env_n: int,
    device: torch.device,
    runtime_dir: Path,
) -> Tuple[Any, torch.nn.Module, Any]:
    from thinker import util  # pylint: disable=import-outside-toplevel
    from thinker.actor_net import ActorNet  # pylint: disable=import-outside-toplevel
    from thinker.main import Env  # pylint: disable=import-outside-toplevel

    runtime_flags = _runtime_flags(flags, runtime_dir)
    runtime_flags.env_n = int(env_n)
    environment = Env(
        gpu=device.type == "cuda",
        load_net=False,
        **vars(runtime_flags),
    )
    try:
        environment.model_net.set_weights(model_checkpoint["model_net_state_dict"])
        environment.model_net.eval()
        environment.model_net.requires_grad_(False)
        actor = ActorNet(
            obs_space=environment.observation_space,
            action_space=environment.action_space,
            flags=runtime_flags,
            tree_rep_meaning=environment.get_tree_rep_meaning(),
        ).to(device)
        actor.set_weights(actor_checkpoint["actor_net_state_dict"], strict=True)
        actor.eval()
        actor.requires_grad_(False)
        if actor.train_actor_enabled:
            raise RuntimeError("evaluation ActorNet still has training epsilon enabled")
        if not actor.voc_eval_stochastic:
            raise RuntimeError("evaluation checkpoint disabled stochastic gate sampling")
        configured_epsilon_greedy_execution = bool(
            getattr(runtime_flags, "voc_gate_epsilon_greedy_execution", False)
        )
        if bool(
            getattr(actor, "voc_gate_epsilon_greedy_execution", False)
        ) != configured_epsilon_greedy_execution:
            raise RuntimeError(
                "evaluation ActorNet gate-execution mode disagrees with config"
            )
        if bool(getattr(flags, "voc_actor_policy_version_barrier", False)):
            if getattr(flags, "voc_actor_policy_barrier_runtime", None) is not True:
                raise RuntimeError(
                    "validated schema-6 training flags lost barrier-runtime=true"
                )
            if getattr(runtime_flags, "voc_actor_policy_barrier_runtime", None) is not False:
                raise RuntimeError(
                    "fixed schema-6 runtime still enables actor-policy barrier waits"
                )
            if getattr(runtime_flags, "voc_actor_policy_version_barrier", None) is not True:
                raise RuntimeError(
                    "fixed schema-6 runtime mutated the persisted barrier identity"
                )
            for name, actual, expected in (
                (
                    "voc_train_epsilon",
                    getattr(actor, "voc_train_epsilon", None),
                    0.02,
                ),
                (
                    "voc_gate_execution_epsilon",
                    getattr(actor, "voc_gate_execution_epsilon", None),
                    0.25,
                ),
            ):
                if type(actual) is not float or actual != expected:
                    raise RuntimeError(
                        f"fixed schema-6 runtime mutated configured {name}"
                    )
            # ActorNet applies both configured epsilons only while training.
            # With this false, held-out soft/execution epsilon is exactly zero.
            if actor.train_actor_enabled:
                raise RuntimeError(
                    "fixed schema-6 runtime did not disable gate exploration"
                )
        sealed_gate_schema = getattr(
            flags, "voc_gate_policy_schema_version", None
        )
        if sealed_gate_schema in (7, 8, 9, 10):
            if (
                getattr(flags, "train_model", None) is not True
                or type(
                    getattr(flags, "voc_model_input_seal_schema_version", None)
                )
                is not int
                or flags.voc_model_input_seal_schema_version != 1
            ):
                raise RuntimeError(
                    f"validated schema-{sealed_gate_schema} training flags lost "
                    "ModelNet seal identity"
                )
            if (
                getattr(runtime_flags, "train_model", None) is not False
                or runtime_flags.voc_model_input_seal_schema_version != 1
                or getattr(
                    environment, "voc_model_input_seal_runtime", None
                )
                is not False
            ):
                raise RuntimeError(
                    f"fixed schema-{sealed_gate_schema} runtime did not disable "
                    "live seal coordination"
                )
        # Inspect Env itself: Gymnasium's wrapper-level ``hasattr`` fallback
        # delegates to the backing Atari environment and emits a deprecation
        # warning, while a learner (if constructed) is always an Env-owned
        # instance attribute.
        if environment.train_model or "model_learner" in vars(environment):
            raise RuntimeError("evaluation environment constructed a model learner")
        if any(module.training for module in actor.modules()):
            raise RuntimeError("ActorNet contains a module left in training mode")
        if any(module.training for module in environment.model_net.modules()):
            raise RuntimeError("ModelNet contains a module left in training mode")
        if any(parameter.requires_grad for parameter in actor.parameters()):
            raise RuntimeError("ActorNet evaluation parameter requires gradients")
        if any(parameter.requires_grad for parameter in environment.model_net.parameters()):
            raise RuntimeError("ModelNet evaluation parameter requires gradients")
        # Keep constants synchronized with the training runtime.
        if (util.PROCEED, util.RESET, util.STOP) != (PROCEED, RESET, STOP):
            raise RuntimeError("training runtime changed Dynamic control indices")
        return environment, actor, runtime_flags
    except BaseException:
        environment.close()
        raise


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def run_fixed_rollouts(
    *,
    flags: Any,
    actor_checkpoint: Mapping[str, Any],
    model_checkpoint: Mapping[str, Any],
    seeds: Sequence[int],
    real_steps_per_seed: int,
    calibration_unroll: int,
    device: torch.device,
    runtime_dir: Path,
    source_manifest_state: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    from thinker import util  # pylint: disable=import-outside-toplevel

    if real_steps_per_seed < 1 or calibration_unroll < 1:
        raise ValueError("rollout horizon and calibration unroll must be positive")
    environment, actor, runtime_flags = _build_actor_and_environment(
        flags,
        actor_checkpoint,
        model_checkpoint,
        env_n=len(seeds),
        device=device,
        runtime_dir=runtime_dir,
    )
    epsilon_greedy_execution = bool(
        getattr(runtime_flags, "voc_gate_epsilon_greedy_execution", False)
    )
    reward_names = tuple(util.get_reward_names(runtime_flags))
    if "re" not in reward_names or "think" not in reward_names:
        environment.close()
        raise RuntimeError("fixed VoC evaluation requires re and think reward channels")
    task_index = reward_names.index("re")
    think_index = reward_names.index("think")
    think_cost = float(runtime_flags.think_cost)
    if bool(runtime_flags.think_cost_anneal):
        environment.close()
        raise ValueError("fixed v7 confirmation requires think_cost_anneal=false")

    from thinker import util as thinker_util  # pylint: disable=import-outside-toplevel

    ema = thinker_util.validate_voc_ema_gate_checkpoint(
        actor_checkpoint, label="fixed-evaluation actor checkpoint"
    )
    ema_weight = ema["voc_ema_gate_head_state_dict"]["weight"].to(device).float()
    ema_bias = ema["voc_ema_gate_head_state_dict"]["bias"].to(device).float()
    actor_digest_before = _state_dict_digest(actor)
    model_digest_before = _state_dict_digest(environment.model_net)

    rows: List[Dict[str, Any]] = []
    calibration_chunk: List[MutableMapping[str, Any]] = []
    real_counts = np.zeros(len(seeds), dtype=np.int64)
    episode_indices = np.zeros(len(seeds), dtype=np.int64)
    stage_end_count = 0
    forced_stop_count = 0
    augmented_step = 0
    max_augmented_steps = int(
        real_steps_per_seed
        * (max(int(getattr(runtime_flags, "max_search_steps", 20)), 1) + 4)
        * 2
    )
    loaded_modules_before: Optional[Dict[str, Any]] = None

    try:
        state, info = environment.reset(seed=list(int(seed) for seed in seeds))
        env_out = util.init_env_out(
            state,
            info,
            runtime_flags,
            actor.dim_actions,
            actor.tuple_action,
        )
        actor_state = actor.initial_state(batch_size=len(seeds), device=device)

        with torch.inference_mode():
            actor_out, actor_state = actor(
                env_out=env_out,
                core_state=actor_state,
                compute_loss=True,
                # The full primary/bout policy is sampled.  Disabling training
                # makes gate epsilon exactly zero.  Schemas 1--4 retain soft
                # gate sampling; schema 5 samples only exact ties after its
                # deterministic non-tie sign transform.  No Q reaches here.
                greedy=False,
            )
            # Freeze the complete loaded thinker module set after environment
            # reset and first inference have exercised all runtime imports.
            loaded_modules_before = loaded_training_modules_attestation(
                source_manifest_state
            )
            while np.any(real_counts < real_steps_per_seed):
                if augmented_step >= max_augmented_steps:
                    raise RuntimeError(
                        "fixed rollout exceeded the bounded augmented-step budget"
                    )
                evaluated = _actor_eval_tensors(
                    actor_out,
                    ema_weight,
                    ema_bias,
                    reward_names=reward_names,
                    think_cost=think_cost,
                    epsilon_greedy_execution=epsilon_greedy_execution,
                )
                primary_action, control_action = actor_out.action
                state, reward, done, truncated, info = environment.step(
                    primary_action=primary_action,
                    search_control=control_action,
                    action_prob=actor_out.action_prob[-1],
                )
                next_env_out = util.create_env_out(
                    actor_out.action,
                    state,
                    reward,
                    done,
                    truncated,
                    info,
                    runtime_flags,
                )
                next_env_out = _update_effective_tokens(env_out, next_env_out, info)

                post_reward = next_env_out.reward[-1].float()
                post_done = (next_env_out.done[-1] | next_env_out.truncated_done[-1].bool())
                post_real_transition = next_env_out.real_transition[-1].bool()
                post_stage_end = next_env_out.stage_end[-1].bool()
                post_forced_stop = next_env_out.forced_stop[-1].bool()
                post_search_steps = next_env_out.search_steps[-1].long()
                pre_depth = env_out.search_steps[-1].long()
                pre_last_control = env_out.last_search_control[-1].long()
                included = torch.as_tensor(
                    real_counts < real_steps_per_seed,
                    dtype=torch.bool,
                    device=device,
                )
                valid = evaluated["control_valid"] & included
                derived_depth = post_search_steps - (
                    valid & (evaluated["control"] != STOP)
                ).long()
                if torch.any(valid & (derived_depth != pre_depth)):
                    raise RuntimeError(
                        "live pre-decision depth disagrees with shifted telemetry depth"
                    )
                if torch.any(valid & env_out.forced_stop[-1].bool()):
                    raise RuntimeError("forced-only row was marked as a valid control decision")

                step_rows: List[Optional[Dict[str, Any]]] = [None] * len(seeds)
                cpu = {
                    name: value.detach().cpu()
                    for name, value in evaluated.items()
                    if torch.is_tensor(value)
                }
                for stream_index, seed in enumerate(seeds):
                    if not bool(valid[stream_index]):
                        continue
                    control = int(cpu["control"][stream_index])
                    gate_action = int(cpu["gate_action"][stream_index])
                    probability = float(cpu["continue_probability"][stream_index])
                    ema_q = cpu["ema_q"][stream_index]
                    online_q = cpu["online_q"][stream_index]
                    row: Dict[str, Any] = {
                        "stream_id": stream_index,
                        "environment_seed": int(seed),
                        "episode_index": int(episode_indices[stream_index]),
                        "augmented_step": augmented_step,
                        "real_step_before": int(real_counts[stream_index]),
                        "decision_depth": int(pre_depth[stream_index].detach().cpu()),
                        "predecision_last_control": int(
                            pre_last_control[stream_index].detach().cpu()
                        ),
                        "sampled_control": control,
                        "gate_action": gate_action,
                        "continue_probability": probability,
                        "stop_probability": 1.0 - probability,
                        "ema_q_continue": float(ema_q[GATE_CONTINUE]),
                        "ema_q_stop": float(ema_q[GATE_STOP]),
                        "ema_delta_q": float(ema_q[GATE_CONTINUE] - ema_q[GATE_STOP]),
                        "online_q_continue": float(online_q[GATE_CONTINUE]),
                        "online_q_stop": float(online_q[GATE_STOP]),
                        "online_delta_q": float(
                            online_q[GATE_CONTINUE] - online_q[GATE_STOP]
                        ),
                        "ema_selected_q": float(cpu["ema_selected_q"][stream_index]),
                        "online_selected_q": float(
                            cpu["online_selected_q"][stream_index]
                        ),
                        "state_value": float(cpu["state_value"][stream_index]),
                        "task_reward": float(
                            post_reward[stream_index, task_index].detach().cpu()
                        ),
                        "think_reward": float(
                            post_reward[stream_index, think_index].detach().cpu()
                        ),
                        "task_discount": float(
                            ((~post_done[stream_index]).float()
                            * (
                                float(runtime_flags.discounting)
                                if bool(post_real_transition[stream_index])
                                else 1.0
                            )).detach().cpu()
                        ),
                        "think_discount": float(
                            (~(post_done[stream_index] | post_stage_end[stream_index]))
                            .float()
                            .detach()
                            .cpu()
                        ),
                        "real_transition": bool(post_real_transition[stream_index]),
                        "stage_end": bool(post_stage_end[stream_index]),
                        "forced_stop": bool(post_forced_stop[stream_index]),
                        "done": bool(next_env_out.done[-1, stream_index]),
                        "truncated": bool(next_env_out.truncated_done[-1, stream_index]),
                    }
                    for key, value in row.items():
                        if isinstance(value, float) and not math.isfinite(value):
                            raise FloatingPointError(f"non-finite decision field {key}")
                    rows.append(row)
                    step_rows[stream_index] = row

                included_stage_end = included & post_stage_end
                stage_end_count += int(included_stage_end.sum().detach().cpu())
                forced_stop_count += int(
                    (included_stage_end & post_forced_stop).sum().detach().cpu()
                )
                task_discount = (~post_done).float() * torch.where(
                    post_real_transition,
                    torch.full_like(post_reward[:, task_index], float(runtime_flags.discounting)),
                    torch.ones_like(post_reward[:, task_index]),
                )
                think_discount = (~(post_done | post_stage_end)).float()
                calibration_chunk.append(
                    {
                        "task_reward": post_reward[:, task_index],
                        "think_reward": post_reward[:, think_index],
                        "task_discount": task_discount,
                        "think_discount": think_discount,
                        "task_value": evaluated["task_value"],
                        "think_value": evaluated["think_value"],
                        "ema_selected_q": evaluated["ema_selected_q"],
                        "online_selected_q": evaluated["online_selected_q"],
                        "rows": step_rows,
                    }
                )

                real_counts += (
                    post_real_transition.detach().cpu().numpy().astype(np.int64)
                    * (real_counts < real_steps_per_seed)
                )
                real_done = info.get("real_done", done | truncated)
                episode_indices += (
                    real_done.detach().cpu().numpy().astype(np.int64)
                    * (real_counts <= real_steps_per_seed)
                )
                augmented_step += 1

                next_actor_out, next_actor_state = actor(
                    env_out=next_env_out,
                    core_state=actor_state,
                    compute_loss=True,
                    greedy=False,
                )
                next_eval = _actor_eval_tensors(
                    next_actor_out,
                    ema_weight,
                    ema_bias,
                    reward_names=reward_names,
                    think_cost=think_cost,
                    epsilon_greedy_execution=epsilon_greedy_execution,
                )
                if len(calibration_chunk) >= calibration_unroll:
                    _flush_calibration_chunk(
                        calibration_chunk,
                        bootstrap_task=next_eval["task_value"],
                        bootstrap_think=next_eval["think_value"],
                        lamb=float(runtime_flags.v_trace_lamb),
                        think_cost=think_cost,
                    )
                env_out = next_env_out
                actor_out = next_actor_out
                actor_state = next_actor_state
                if augmented_step % 1000 == 0:
                    print(
                        "fixed evaluation: "
                        f"augmented={augmented_step}, "
                        f"real_min={int(real_counts.min())}, "
                        f"decisions={len(rows)}",
                        flush=True,
                    )

            final_eval = _actor_eval_tensors(
                actor_out,
                ema_weight,
                ema_bias,
                reward_names=reward_names,
                think_cost=think_cost,
                epsilon_greedy_execution=epsilon_greedy_execution,
            )
            _flush_calibration_chunk(
                calibration_chunk,
                bootstrap_task=final_eval["task_value"],
                bootstrap_think=final_eval["think_value"],
                lamb=float(runtime_flags.v_trace_lamb),
                think_cost=think_cost,
            )
    finally:
        actor_digest_after = _state_dict_digest(actor)
        model_digest_after = _state_dict_digest(environment.model_net)
        environment.close()

    if actor_digest_before != actor_digest_after:
        raise RuntimeError("ActorNet state changed during fixed evaluation")
    if model_digest_before != model_digest_after:
        raise RuntimeError("ModelNet state changed during fixed evaluation")
    if loaded_modules_before is None:
        raise RuntimeError("training module attestation was not initialized")
    loaded_modules_after = loaded_training_modules_attestation(source_manifest_state)
    require_attestation_unchanged(
        loaded_modules_before,
        loaded_modules_after,
        label="loaded training module set",
    )
    if not rows:
        raise RuntimeError("fixed rollout produced no valid control decisions")
    missing_calibration = [
        index for index, row in enumerate(rows) if "calibration_net_target" not in row
    ]
    if missing_calibration:
        raise RuntimeError(
            f"calibration was not assigned to {len(missing_calibration)} decision rows"
        )
    return rows, {
        "augmented_steps": augmented_step,
        "real_steps_per_stream": real_counts.tolist(),
        "stage_end_count": stage_end_count,
        "forced_stop_count": forced_stop_count,
        "actor_state_digest_before": actor_digest_before,
        "actor_state_digest_after": actor_digest_after,
        "model_state_digest_before": model_digest_before,
        "model_state_digest_after": model_digest_after,
        "network_state_unchanged": True,
        "loaded_training_modules_count": loaded_modules_before["count"],
        "loaded_training_modules_digest_before": loaded_modules_before[
            "semantic_sha256"
        ],
        "loaded_training_modules_digest_after": loaded_modules_after[
            "semantic_sha256"
        ],
        "loaded_training_modules_unchanged": True,
    }


def evaluate(args: argparse.Namespace) -> Mapping[str, Path]:
    evaluator_path = Path(__file__).resolve()
    evaluator_attestation_before = attest_regular_file(
        evaluator_path, label="fixed evaluator source"
    )
    requested_profile, _ = _resolve_confirmation_profile(
        getattr(
            args, "confirmation_profile", DEFAULT_CONFIRMATION_PROFILE.value
        )
    )
    completion_schema_version = (
        2 if requested_profile is ConfirmationProfile.V20_300K else 1
    )
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    bundle = validate_checkpoint_bundle(
        checkpoint_dir,
        training_source_root=args.training_source_root,
        source_manifest=args.source_manifest,
        completion_schema_version=completion_schema_version,
    )
    bind_training_runtime(bundle.source_root)
    # Import only after binding the exact source snapshot used for training.
    import evaluate_dynamic_imitation as checkpoint_eval  # pylint: disable=import-outside-toplevel

    checkpoint_evaluator_path = Path(checkpoint_eval.__file__).resolve()
    if bundle.source_root not in checkpoint_evaluator_path.parents:
        raise RuntimeError(
            "checkpoint validator was not imported from the frozen source: "
            f"{checkpoint_evaluator_path}"
        )

    schema8_dispatch = getattr(
        checkpoint_eval, "dispatch_schema8_completed_bundle", None
    )
    schema9_dispatch = getattr(
        checkpoint_eval, "dispatch_schema9_completed_bundle", None
    )
    schema10_dispatch = getattr(
        checkpoint_eval, "dispatch_schema10_completed_bundle", None
    )
    schema11_dispatch = getattr(
        checkpoint_eval, "dispatch_schema11_completed_bundle", None
    )
    schema12_dispatch = getattr(
        checkpoint_eval, "dispatch_schema12_completed_bundle", None
    )
    schema13_dispatch = getattr(
        checkpoint_eval, "dispatch_schema13_completed_bundle", None
    )
    schema13_claims_intent = getattr(
        checkpoint_eval, "_schema13_xpid_claims_intent", None
    )
    validated_config_loader = getattr(
        checkpoint_eval, "_load_flags_from_validated_config_bytes", None
    )
    bound_config_digest = bundle.file_hashes.get("config_c.yaml")
    validated_config_payload = None
    claims_schema9 = False
    claims_schema10 = False
    claims_schema11 = False
    claims_schema12 = False
    claims_schema13 = False
    if bound_config_digest is not None:
        if (
            type(bound_config_digest) is not str
            or re.fullmatch(r"[0-9a-f]{64}", bound_config_digest) is None
        ):
            raise RuntimeError(
                "fixed checkpoint bundle has malformed config digest"
            )
        validated_config_payload = _read_stable_single_link_bytes(
            checkpoint_dir / "config_c.yaml", label="fixed checkpoint config"
        )
        if (
            hashlib.sha256(validated_config_payload).hexdigest()
            != bound_config_digest
        ):
            raise RuntimeError(
                "fixed checkpoint config changed after bundle validation"
            )
        try:
            bound_config_claim = yaml.safe_load(
                validated_config_payload.decode("utf-8")
            )
        except (UnicodeDecodeError, yaml.YAMLError) as error:
            raise ValueError(
                "fixed checkpoint config is not strict UTF-8 YAML"
            ) from error
        if not isinstance(bound_config_claim, Mapping):
            raise ValueError("fixed checkpoint config must be a mapping")
        bound_schema = bound_config_claim.get("voc_gate_policy_schema_version")
        bound_xpid = bound_config_claim.get("xpid")
        claims_schema9 = bound_schema == 9 or (
            type(bound_xpid) is str and bound_xpid == V16_PRIMARY_XPID
        )
        claims_schema10 = bound_schema == 10 or (
            type(bound_xpid) is str and bound_xpid == V17_PRIMARY_XPID
        )
        claims_schema11 = bound_schema == 11 or (
            type(bound_xpid) is str
            and bound_xpid.startswith(
                "enduro-voc-v18-orthocd-adam-eps25-"
            )
        )
        claims_schema12 = bound_schema == 12 or (
            type(bound_xpid) is str
            and bound_xpid.startswith(
                "enduro-voc-v19-tau1-orthocd-adam-eps25-"
            )
        )
        local_schema13_intent = _fixed_schema13_xpid_claims_intent(bound_xpid)
        if callable(schema13_claims_intent):
            external_schema13_intent = schema13_claims_intent(bound_xpid)
            if (
                type(external_schema13_intent) is not bool
                or external_schema13_intent != local_schema13_intent
            ):
                raise RuntimeError(
                    "fixed/public schema-13 lexical classifiers disagree"
                )
        claims_schema13 = bound_schema == 13 or local_schema13_intent
        if (
            (
                claims_schema13
                or requested_profile is ConfirmationProfile.V20_300K
            )
            and schema13_dispatch is not None
            and not callable(schema13_claims_intent)
        ):
            raise RuntimeError(
                "fixed checkpoint validator lacks schema-13 lexical classifier"
            )
    if (
        claims_schema9
        and requested_profile is not ConfirmationProfile.V16_300K
    ):
        if schema9_dispatch is None:
            raise RuntimeError(
                "schema-9 checkpoint validator lacks schema-9 dispatch"
            )
        schema9_dispatch(
            checkpoint_dir,
            completion_state=bundle.marker,
            config_payload=validated_config_payload,
            expected_config_sha256=bound_config_digest,
        )
        raise ValueError(
            "schema-9 checkpoint is eligible only for fixed profile v16-300k"
        )
    if (
        claims_schema10
        and requested_profile is not ConfirmationProfile.V17_300K
    ):
        if schema10_dispatch is None:
            raise RuntimeError(
                "schema-10 checkpoint validator lacks schema-10 dispatch"
            )
        schema10_dispatch(
            checkpoint_dir,
            completion_state=bundle.marker,
            config_payload=validated_config_payload,
            expected_config_sha256=bound_config_digest,
        )
        raise ValueError(
            "schema-10 checkpoint is eligible only for fixed profile v17-300k"
        )
    if (
        claims_schema11
        and requested_profile is not ConfirmationProfile.V18_300K
    ):
        if schema11_dispatch is None:
            raise RuntimeError(
                "schema-11 checkpoint validator lacks schema-11 dispatch"
            )
        schema11_dispatch(
            checkpoint_dir,
            completion_state=bundle.marker,
            config_payload=validated_config_payload,
            expected_config_sha256=bound_config_digest,
        )
        raise ValueError(
            "schema-11 checkpoint is eligible only for fixed profile v18-300k"
        )
    if (
        claims_schema12
        and requested_profile is not ConfirmationProfile.V19_300K
    ):
        if schema12_dispatch is None:
            raise RuntimeError(
                "schema-12 checkpoint validator lacks schema-12 dispatch"
            )
        schema12_dispatch(
            checkpoint_dir,
            completion_state=bundle.marker,
            config_payload=validated_config_payload,
            expected_config_sha256=bound_config_digest,
        )
        raise ValueError(
            "schema-12 checkpoint is eligible only for fixed profile v19-300k"
        )
    if (
        claims_schema13
        and requested_profile is not ConfirmationProfile.V20_300K
    ):
        if schema13_dispatch is None:
            raise RuntimeError(
                "schema-13 checkpoint validator lacks schema-13 dispatch"
            )
        schema13_dispatch(
            checkpoint_dir,
            completion_state=bundle.marker,
            config_payload=validated_config_payload,
            expected_config_sha256=bound_config_digest,
        )
        raise ValueError(
            "schema-13 checkpoint is eligible only for fixed profile v20-300k"
        )
    dispatched_schema8_validation = None
    if requested_profile is ConfirmationProfile.V15_300K:
        if schema8_dispatch is None:
            raise RuntimeError(
                "v15-300k checkpoint validator lacks schema-8 dispatch"
            )
        if validated_config_loader is None:
            raise RuntimeError(
                "v15-300k checkpoint validator lacks schema-8 byte-bound loader"
            )
        if (
            type(validated_config_payload) is not bytes
            or type(bound_config_digest) is not str
            or re.fullmatch(r"[0-9a-f]{64}", bound_config_digest) is None
        ):
            raise RuntimeError(
                "v15-300k checkpoint lacks bound config evidence"
            )
        dispatched_schema8_validation = schema8_dispatch(
            checkpoint_dir,
            completion_state=bundle.marker,
            config_payload=validated_config_payload,
            expected_config_sha256=bound_config_digest,
        )
    schema8_bundle_validation: Optional[Mapping[str, Any]] = None
    if requested_profile is ConfirmationProfile.V15_300K:
        # The requested v15 profile selects the strict dedicated schema-8
        # route before flags, live probes, evaluator-direct tensor loads, data,
        # rollout, or output mutation.
        schema8_bundle_validation = validate_v15_final_bundle(
            checkpoint_dir,
            bundle.marker,
            checkpoint_eval=checkpoint_eval,
            completed_validation=dispatched_schema8_validation,
        )
    dispatched_schema9_validation = None
    schema9_bundle_validation: Optional[Mapping[str, Any]] = None
    if requested_profile is ConfirmationProfile.V16_300K:
        if schema9_dispatch is None:
            raise RuntimeError(
                "v16-300k checkpoint validator lacks schema-9 dispatch"
            )
        if validated_config_loader is None:
            raise RuntimeError(
                "v16-300k checkpoint validator lacks schema-9 byte-bound loader"
            )
        if (
            type(validated_config_payload) is not bytes
            or type(bound_config_digest) is not str
            or re.fullmatch(r"[0-9a-f]{64}", bound_config_digest) is None
        ):
            raise RuntimeError(
                "v16-300k checkpoint lacks bound config evidence"
            )
        dispatched_schema9_validation = schema9_dispatch(
            checkpoint_dir,
            completion_state=bundle.marker,
            config_payload=validated_config_payload,
            expected_config_sha256=bound_config_digest,
        )
        schema9_bundle_validation = validate_v16_final_bundle(
            checkpoint_dir,
            bundle.marker,
            checkpoint_eval=checkpoint_eval,
            completed_validation=dispatched_schema9_validation,
            config_payload=validated_config_payload,
            expected_config_sha256=bound_config_digest,
        )
    dispatched_schema10_validation = None
    schema10_bundle_validation: Optional[Mapping[str, Any]] = None
    if requested_profile is ConfirmationProfile.V17_300K:
        if schema10_dispatch is None:
            raise RuntimeError(
                "v17-300k checkpoint validator lacks schema-10 dispatch"
            )
        if validated_config_loader is None:
            raise RuntimeError(
                "v17-300k checkpoint validator lacks schema-10 byte-bound loader"
            )
        if (
            type(validated_config_payload) is not bytes
            or type(bound_config_digest) is not str
            or re.fullmatch(r"[0-9a-f]{64}", bound_config_digest) is None
        ):
            raise RuntimeError(
                "v17-300k checkpoint lacks bound config evidence"
            )
        dispatched_schema10_validation = schema10_dispatch(
            checkpoint_dir,
            completion_state=bundle.marker,
            config_payload=validated_config_payload,
            expected_config_sha256=bound_config_digest,
        )
        schema10_bundle_validation = validate_v17_final_bundle(
            checkpoint_dir,
            bundle.marker,
            checkpoint_eval=checkpoint_eval,
            completed_validation=dispatched_schema10_validation,
            config_payload=validated_config_payload,
            expected_config_sha256=bound_config_digest,
        )
    dispatched_schema11_validation = None
    schema11_bundle_validation: Optional[Mapping[str, Any]] = None
    if requested_profile is ConfirmationProfile.V18_300K:
        if schema11_dispatch is None:
            raise RuntimeError(
                "v18-300k checkpoint validator lacks schema-11 dispatch"
            )
        if validated_config_loader is None:
            raise RuntimeError(
                "v18-300k checkpoint validator lacks schema-11 byte-bound loader"
            )
        if (
            type(validated_config_payload) is not bytes
            or type(bound_config_digest) is not str
            or re.fullmatch(r"[0-9a-f]{64}", bound_config_digest) is None
        ):
            raise RuntimeError(
                "v18-300k checkpoint lacks bound config evidence"
            )
        dispatched_schema11_validation = schema11_dispatch(
            checkpoint_dir,
            completion_state=bundle.marker,
            config_payload=validated_config_payload,
            expected_config_sha256=bound_config_digest,
        )
        schema11_bundle_validation = validate_v18_final_bundle(
            checkpoint_dir,
            bundle.marker,
            checkpoint_eval=checkpoint_eval,
            completed_validation=dispatched_schema11_validation,
            config_payload=validated_config_payload,
            expected_config_sha256=bound_config_digest,
        )
    dispatched_schema12_validation = None
    schema12_bundle_validation: Optional[Mapping[str, Any]] = None
    if requested_profile is ConfirmationProfile.V19_300K:
        if schema12_dispatch is None:
            raise RuntimeError(
                "v19-300k checkpoint validator lacks schema-12 dispatch"
            )
        if validated_config_loader is None:
            raise RuntimeError(
                "v19-300k checkpoint validator lacks schema-12 byte-bound loader"
            )
        if (
            type(validated_config_payload) is not bytes
            or type(bound_config_digest) is not str
            or re.fullmatch(r"[0-9a-f]{64}", bound_config_digest) is None
        ):
            raise RuntimeError(
                "v19-300k checkpoint lacks bound config evidence"
            )
        dispatched_schema12_validation = schema12_dispatch(
            checkpoint_dir,
            completion_state=bundle.marker,
            config_payload=validated_config_payload,
            expected_config_sha256=bound_config_digest,
        )
        schema12_bundle_validation = validate_v19_final_bundle(
            checkpoint_dir,
            bundle.marker,
            checkpoint_eval=checkpoint_eval,
            completed_validation=dispatched_schema12_validation,
            config_payload=validated_config_payload,
            expected_config_sha256=bound_config_digest,
        )
    dispatched_schema13_validation = None
    schema13_bundle_validation: Optional[Mapping[str, Any]] = None
    if requested_profile is ConfirmationProfile.V20_300K:
        if schema13_dispatch is None:
            raise RuntimeError(
                "v20-300k checkpoint validator lacks schema-13 dispatch"
            )
        if validated_config_loader is None:
            raise RuntimeError(
                "v20-300k checkpoint validator lacks schema-13 byte-bound loader"
            )
        if (
            type(validated_config_payload) is not bytes
            or type(bound_config_digest) is not str
            or re.fullmatch(r"[0-9a-f]{64}", bound_config_digest) is None
        ):
            raise RuntimeError(
                "v20-300k checkpoint lacks bound config evidence"
            )
        dispatched_schema13_validation = schema13_dispatch(
            checkpoint_dir,
            completion_state=bundle.marker,
            config_payload=validated_config_payload,
            expected_config_sha256=bound_config_digest,
        )
        schema13_bundle_validation = validate_v20_final_bundle(
            checkpoint_dir,
            bundle.marker,
            checkpoint_eval=checkpoint_eval,
            completed_validation=dispatched_schema13_validation,
            config_payload=validated_config_payload,
            expected_config_sha256=bound_config_digest,
        )
    schema7_bundle_validation: Optional[Mapping[str, Any]] = None
    if requested_profile is ConfirmationProfile.V14_300K:
        # Schema 7 is validated before even the unscored live-environment
        # contract probe.  A malformed terminal artifact must not trigger an
        # environment reset/action, rollout, or output-directory mutation.
        schema7_bundle_validation = validate_v14_final_bundle(
            checkpoint_dir,
            bundle.marker,
            checkpoint_eval=checkpoint_eval,
        )
    elif requested_profile not in (
        ConfirmationProfile.V15_300K,
        ConfirmationProfile.V16_300K,
        ConfirmationProfile.V17_300K,
        ConfirmationProfile.V18_300K,
        ConfirmationProfile.V19_300K,
        ConfirmationProfile.V20_300K,
    ):
        # Preserve the historical v7-v13 path for ordinary legacy artifacts,
        # including the immutable v14 synthetic ordering corpus.  A real
        # schema-8/v15 claim is still routed to the strict validator before
        # _load_flags, any live probe, evaluator-direct tensor/data access, or
        # output.  Production bundles always contain config_c.yaml; the
        # missing-file case exists only in the historical mocked corpus.
        claims_schema8 = False
        if bound_config_digest is not None:
            config_payload = validated_config_payload
            if config_payload is None:
                config_payload = _read_stable_single_link_bytes(
                    checkpoint_dir / "config_c.yaml",
                    label="fixed checkpoint config",
                )
            if (
                type(bound_config_digest) is not str
                or re.fullmatch(r"[0-9a-f]{64}", bound_config_digest) is None
                or hashlib.sha256(config_payload).hexdigest()
                != bound_config_digest
            ):
                raise RuntimeError(
                    "fixed checkpoint config changed before schema-8 claim probe"
                )
            try:
                config_claim = yaml.safe_load(config_payload.decode("utf-8"))
            except (UnicodeDecodeError, yaml.YAMLError) as error:
                raise ValueError(
                    "fixed checkpoint config is not strict UTF-8 YAML"
                ) from error
            if not isinstance(config_claim, Mapping):
                raise ValueError("fixed checkpoint config must be a mapping")
            raw_schema = config_claim.get("voc_gate_policy_schema_version")
            raw_xpid = config_claim.get("xpid")
            claims_schema8 = raw_schema == 8 or (
                type(raw_xpid) is str and raw_xpid == V15_PRIMARY_XPID
            )
        if claims_schema8:
            if schema8_dispatch is None:
                raise RuntimeError(
                    "schema-8 checkpoint validator lacks schema-8 dispatch"
                )
            dispatched_schema8_validation = schema8_dispatch(
                checkpoint_dir,
                completion_state=bundle.marker,
                config_payload=config_payload,
                expected_config_sha256=bound_config_digest,
            )
            raise ValueError(
                "schema-8 checkpoint is eligible only for fixed profile "
                "v15-300k"
            )

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seeds = _parse_seeds(int(args.seed_base), int(args.num_seeds))
    _set_deterministic_seed(int(args.seed_base))
    initial_hashes = dict(bundle.file_hashes)
    if validated_config_payload is not None:
        flags = _load_flags_from_bound_config_bytes(
            checkpoint_eval,
            checkpoint_dir,
            validated_config_payload,
            bound_config_digest,
            byte_loader=validated_config_loader,
        )
        current_config = _read_stable_single_link_bytes(
            checkpoint_dir / "config_c.yaml",
            label="fixed checkpoint config after flag resolution",
        )
        if hashlib.sha256(current_config).hexdigest() != bound_config_digest:
            raise RuntimeError(
                "fixed checkpoint config changed before downstream evaluation"
            )
    else:
        flags = checkpoint_eval._load_flags(  # pylint: disable=protected-access
            checkpoint_dir
        )
    spec = checkpoint_eval.resolve_evaluation_spec(
        flags,
        expected_env_name="Enduro-v5",
        expected_game_id=args.expected_game_id,
    )
    # Live-contract probing above is deliberately unscored.  Reset all RNGs
    # before the fixed seed rollout so it cannot perturb gate sampling.
    _set_deterministic_seed(int(args.seed_base))
    checkpoint_files = bundle.marker.get("checkpoint_files")
    is_v20 = requested_profile is ConfirmationProfile.V20_300K
    actor_checkpoint = _load_fixed_runtime_checkpoint(
        checkpoint_dir,
        "ckp_actor.tar",
        checkpoint_files,
        v20=is_v20,
        label="fixed actor checkpoint",
    )
    model_checkpoint = _load_fixed_runtime_checkpoint(
        checkpoint_dir,
        "ckp_model.tar",
        checkpoint_files,
        v20=is_v20,
        label="fixed model checkpoint",
    )
    if not is_v20:
        if not isinstance(actor_checkpoint, Mapping) or not isinstance(
            model_checkpoint, Mapping
        ):
            raise TypeError("actor and model checkpoints must be mappings")
    actor_validation = checkpoint_eval.validate_actor_imitation_checkpoint(
        actor_checkpoint, flags, spec
    )
    model_validation = checkpoint_eval.validate_model_checkpoint(
        model_checkpoint, flags, spec
    )
    schema6_bundle_validation: Optional[Mapping[str, Any]] = None
    if requested_profile is ConfirmationProfile.V13_300K:
        # The authoritative validator binds all 228 config/actor/model fields,
        # terminal history, training state, logger completion, and marker
        # cleanup before a stage mismatch can reach rollout or output creation.
        schema6_bundle_validation = validate_v13_final_bundle(
            checkpoint_dir, bundle.marker
        )
    configured_data_root = str(getattr(flags, "icopro_data_path", "")).strip()
    data_root = args.data_root if args.data_root is not None else configured_data_root
    if not str(data_root).strip():
        raise ValueError(
            "checkpoint has no behavioral data root; pass --data-root to "
            "recompute its training-corpus signature"
        )
    behavioral_data_state = validate_behavioral_training_data(
        flags=flags,
        spec=spec,
        actor_checkpoint=actor_checkpoint,
        data_root=data_root,
        checkpoint_eval=checkpoint_eval,
    )
    actor_validation = {
        **actor_validation,
        "training_data_signature_recomputed": True,
    }
    protocol_state = _require_fixed_protocol(
        flags,
        actor_checkpoint,
        model_checkpoint,
        actor_validation,
        model_validation,
        confirmation_profile=requested_profile,
        seeds=seeds,
        real_steps_per_seed=int(args.real_steps_per_seed),
        calibration_unroll=int(args.calibration_unroll),
        diagnostic=bool(args.diagnostic),
        schema6_bundle_validation=schema6_bundle_validation,
        schema7_bundle_validation=schema7_bundle_validation,
        schema8_bundle_validation=schema8_bundle_validation,
        schema9_bundle_validation=schema9_bundle_validation,
        schema10_bundle_validation=schema10_bundle_validation,
        schema11_bundle_validation=schema11_bundle_validation,
        schema12_bundle_validation=schema12_bundle_validation,
        schema13_bundle_validation=schema13_bundle_validation,
    )
    current_bundle = validate_checkpoint_bundle(
        checkpoint_dir,
        training_source_root=bundle.source_root,
        source_manifest=bundle.source_manifest["path"],
        completion_schema_version=completion_schema_version,
    )
    if (
        dict(current_bundle.file_hashes) != initial_hashes
        or dict(current_bundle.source_manifest) != dict(bundle.source_manifest)
    ):
        raise RuntimeError("checkpoint bundle changed while being loaded")

    output_dir = Path(args.output_dir).expanduser().resolve()
    _ensure_output_is_external(output_dir, bundle)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "decisions": output_dir / "decision_rows.csv",
        "summary": output_dir / "summary.json",
        "manifest": output_dir / "manifest.json",
    }
    generation_id = uuid.uuid4().hex
    device = _resolve_device(args.device)
    with exclusive_output_lock(
        output_dir,
        generation_id=generation_id,
        evaluator_attestation=evaluator_attestation_before,
    ) as output_lock:
        existing = [path for path in outputs.values() if path.exists()]
        if existing and not args.overwrite:
            raise FileExistsError(
                "fixed-evaluation outputs already exist; pass --overwrite: "
                + ", ".join(str(path) for path in existing)
            )

        runtime_attestation_before = collect_runtime_attestation(
            device, expected_rom_sha256=args.expected_rom_sha256
        )
        with tempfile.TemporaryDirectory(
            prefix="voc-fixed-runtime-", dir=str(output_dir)
        ) as runtime_name:
            rows, rollout_state = run_fixed_rollouts(
                flags=flags,
                actor_checkpoint=actor_checkpoint,
                model_checkpoint=model_checkpoint,
                seeds=seeds,
                real_steps_per_seed=int(args.real_steps_per_seed),
                calibration_unroll=int(args.calibration_unroll),
                device=device,
                runtime_dir=Path(runtime_name),
                source_manifest_state=bundle.source_manifest,
            )

        summary = summarize_decision_rows(
            rows,
            q_temperature=float(flags.voc_gate_q_temperature),
            stage_end_count=int(rollout_state["stage_end_count"]),
            forced_stop_count=int(rollout_state["forced_stop_count"]),
        )
        decision_semantic_hash = decision_rows_semantic_sha256(rows)
        created_utc = datetime.now(timezone.utc).isoformat()
        gate_execution_protocol = _evaluation_gate_protocol(protocol_state)
        evaluation_protocol = {
            "confirmation_profile": protocol_state["confirmation_profile"],
            "resolved_profile_identity": protocol_state[
                "resolved_profile_identity"
            ],
            "evaluation_mode": protocol_state["evaluation_mode"],
            "confirmation_eligible": protocol_state["confirmation_eligible"],
            "training_disabled": True,
            "actor_eval_mode": True,
            "model_eval_mode": True,
            "gradient_enabled": False,
            "epsilon": 0.0,
            **gate_execution_protocol,
            "sample_full_policy": True,
            "q_action_override": False,
            "fixed_seeds": list(seeds),
            "real_steps_per_seed": int(args.real_steps_per_seed),
            "total_requested_real_steps": len(seeds)
            * int(args.real_steps_per_seed),
            "calibration_unroll": int(args.calibration_unroll),
            "calibration_vtrace_lambda": float(flags.v_trace_lamb),
            "think_cost": float(flags.think_cost),
            "primary_and_bout_sampling": "checkpoint policy, fixed RNG",
        }
        summary.update(
            {
                "generation_id": generation_id,
                "created_utc": created_utc,
                "environment": spec.env_name,
                "checkpoint_protocol": protocol_state,
                "evaluation_protocol": evaluation_protocol,
                "fixed_confirmation_pass": bool(
                    protocol_state["confirmation_eligible"]
                    and summary["all_four_behaviors_pass"]
                ),
                "rollout": rollout_state,
                "decision_rows_semantic_sha256": decision_semantic_hash,
                "actor_checkpoint_validation": actor_validation,
                "model_checkpoint_validation": model_validation,
                "behavioral_training_data_validation": behavioral_data_state,
            }
        )

        with tempfile.TemporaryDirectory(
            prefix=".fixed-generation-", dir=str(output_dir)
        ) as staging_name:
            staging_dir = Path(staging_name)
            staged_outputs = {
                key: staging_dir / path.name for key, path in outputs.items()
            }
            # Large evidence is written only to the private staging directory.
            # No public output is replaced until every post-attestation passes.
            _atomic_write_csv(rows, staged_outputs["decisions"])

            final_bundle = validate_checkpoint_bundle(
                checkpoint_dir,
                training_source_root=bundle.source_root,
                source_manifest=bundle.source_manifest["path"],
                completion_schema_version=completion_schema_version,
            )
            if (
                dict(final_bundle.file_hashes) != initial_hashes
                or dict(final_bundle.source_manifest) != dict(bundle.source_manifest)
            ):
                raise RuntimeError("checkpoint bundle changed during fixed evaluation")
            schema6_final_revalidated = False
            schema7_final_revalidated = False
            schema8_final_revalidated = False
            schema9_final_revalidated = False
            schema10_final_revalidated = False
            schema11_final_revalidated = False
            schema12_final_revalidated = False
            schema13_final_revalidated = False
            if requested_profile is ConfirmationProfile.V13_300K:
                schema6_bundle_post = validate_v13_final_bundle(
                    checkpoint_dir, final_bundle.marker
                )
                if dict(schema6_bundle_post) != dict(
                    schema6_bundle_validation
                ):
                    raise RuntimeError(
                        "v13-300k final-bundle evidence changed during fixed evaluation"
                    )
                schema6_final_revalidated = True
                protocol_state[
                    "schema6_final_bundle_revalidated_after_rollout"
                ] = True
                evaluation_protocol[
                    "schema6_final_bundle_revalidated_after_rollout"
                ] = True
            elif requested_profile is ConfirmationProfile.V14_300K:
                schema7_bundle_post = validate_v14_final_bundle(
                    checkpoint_dir,
                    final_bundle.marker,
                    checkpoint_eval=checkpoint_eval,
                )
                if dict(schema7_bundle_post) != dict(
                    schema7_bundle_validation
                ):
                    raise RuntimeError(
                        "v14-300k final-bundle evidence changed during fixed "
                        "evaluation"
                    )
                schema7_final_revalidated = True
                protocol_state[
                    "schema7_final_bundle_revalidated_after_rollout"
                ] = True
                evaluation_protocol[
                    "schema7_final_bundle_revalidated_after_rollout"
                ] = True
            elif requested_profile is ConfirmationProfile.V15_300K:
                schema8_bundle_post = validate_v15_final_bundle(
                    checkpoint_dir,
                    final_bundle.marker,
                    checkpoint_eval=checkpoint_eval,
                )
                if dict(schema8_bundle_post) != dict(
                    schema8_bundle_validation
                ):
                    raise RuntimeError(
                        "v15-300k final-bundle evidence changed during fixed "
                        "evaluation"
                    )
                schema8_final_revalidated = True
                protocol_state[
                    "schema8_final_bundle_revalidated_after_rollout"
                ] = True
                evaluation_protocol[
                    "schema8_final_bundle_revalidated_after_rollout"
                ] = True
            elif requested_profile is ConfirmationProfile.V16_300K:
                schema9_bundle_post = validate_v16_final_bundle(
                    checkpoint_dir,
                    final_bundle.marker,
                    checkpoint_eval=checkpoint_eval,
                    config_payload=validated_config_payload,
                    expected_config_sha256=bound_config_digest,
                )
                if dict(schema9_bundle_post) != dict(
                    schema9_bundle_validation
                ):
                    raise RuntimeError(
                        "v16-300k final-bundle evidence changed during fixed "
                        "evaluation"
                    )
                schema9_final_revalidated = True
                protocol_state[
                    "schema9_final_bundle_revalidated_after_rollout"
                ] = True
                evaluation_protocol[
                    "schema9_final_bundle_revalidated_after_rollout"
                ] = True
            elif requested_profile is ConfirmationProfile.V17_300K:
                schema10_bundle_post = validate_v17_final_bundle(
                    checkpoint_dir,
                    final_bundle.marker,
                    checkpoint_eval=checkpoint_eval,
                    config_payload=validated_config_payload,
                    expected_config_sha256=bound_config_digest,
                )
                if dict(schema10_bundle_post) != dict(
                    schema10_bundle_validation
                ):
                    raise RuntimeError(
                        "v17-300k final-bundle evidence changed during fixed "
                        "evaluation"
                    )
                schema10_final_revalidated = True
                protocol_state[
                    "schema10_final_bundle_revalidated_after_rollout"
                ] = True
                evaluation_protocol[
                    "schema10_final_bundle_revalidated_after_rollout"
                ] = True
            elif requested_profile is ConfirmationProfile.V18_300K:
                schema11_bundle_post = validate_v18_final_bundle(
                    checkpoint_dir,
                    final_bundle.marker,
                    checkpoint_eval=checkpoint_eval,
                    config_payload=validated_config_payload,
                    expected_config_sha256=bound_config_digest,
                )
                if dict(schema11_bundle_post) != dict(
                    schema11_bundle_validation
                ):
                    raise RuntimeError(
                        "v18-300k final-bundle evidence changed during fixed "
                        "evaluation"
                    )
                schema11_final_revalidated = True
                protocol_state[
                    "schema11_final_bundle_revalidated_after_rollout"
                ] = True
                evaluation_protocol[
                    "schema11_final_bundle_revalidated_after_rollout"
                ] = True
            elif requested_profile is ConfirmationProfile.V19_300K:
                schema12_bundle_post = validate_v19_final_bundle(
                    checkpoint_dir,
                    final_bundle.marker,
                    checkpoint_eval=checkpoint_eval,
                    config_payload=validated_config_payload,
                    expected_config_sha256=bound_config_digest,
                )
                if dict(schema12_bundle_post) != dict(
                    schema12_bundle_validation
                ):
                    raise RuntimeError(
                        "v19-300k final-bundle evidence changed during fixed "
                        "evaluation"
                    )
                schema12_final_revalidated = True
                protocol_state[
                    "schema12_final_bundle_revalidated_after_rollout"
                ] = True
                evaluation_protocol[
                    "schema12_final_bundle_revalidated_after_rollout"
                ] = True
            elif requested_profile is ConfirmationProfile.V20_300K:
                schema13_bundle_post = validate_v20_final_bundle(
                    checkpoint_dir,
                    final_bundle.marker,
                    checkpoint_eval=checkpoint_eval,
                    config_payload=validated_config_payload,
                    expected_config_sha256=bound_config_digest,
                )
                if dict(schema13_bundle_post) != dict(
                    schema13_bundle_validation
                ):
                    raise RuntimeError(
                        "v20-300k final-bundle evidence changed during fixed "
                        "evaluation"
                    )
                schema13_final_revalidated = True
                protocol_state[
                    "schema13_final_bundle_revalidated_after_rollout"
                ] = True
                evaluation_protocol[
                    "schema13_final_bundle_revalidated_after_rollout"
                ] = True
            behavioral_data_post = revalidate_behavioral_training_data(
                behavioral_data_state
            )
            loaded_modules_final = loaded_training_modules_attestation(
                final_bundle.source_manifest
            )
            if (
                loaded_modules_final["semantic_sha256"]
                != rollout_state["loaded_training_modules_digest_after"]
                or loaded_modules_final["count"]
                != rollout_state["loaded_training_modules_count"]
            ):
                raise RuntimeError(
                    "loaded training module set changed after the rollout"
                )
            runtime_attestation_after = collect_runtime_attestation(
                device, expected_rom_sha256=args.expected_rom_sha256
            )
            require_attestation_unchanged(
                runtime_attestation_before,
                runtime_attestation_after,
                label="runtime/ROM attestation",
            )
            evaluator_attestation_after = attest_regular_file(
                evaluator_path, label="fixed evaluator source"
            )
            require_attestation_unchanged(
                evaluator_attestation_before,
                evaluator_attestation_after,
                label="fixed evaluator source",
            )

            _atomic_write_json(summary, staged_outputs["summary"])
            staged_file_records = {
                outputs[key].name: {
                    "path": str(outputs[key]),
                    "size": staged_outputs[key].stat().st_size,
                    "sha256": sha256_file(staged_outputs[key]),
                }
                for key in ("decisions", "summary")
            }
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "confirmation_profile": protocol_state["confirmation_profile"],
                "resolved_profile_identity": protocol_state[
                    "resolved_profile_identity"
                ],
                "generation_id": generation_id,
                "created_utc": created_utc,
                "command": list(sys.argv),
                "commit": {
                    "state": "committed",
                    "manifest_is_commit_marker": True,
                    "output_lock": OUTPUT_LOCK_NAME,
                    "publication_order": [
                        "decision_rows.csv",
                        "summary.json",
                        "manifest.json",
                    ],
                    "valid_only_when_output_lock_absent": True,
                },
                "runtime": {
                    "attestation_before": runtime_attestation_before,
                    "attestation_after": runtime_attestation_after,
                    "attestation_unchanged": True,
                    "deterministic_algorithms": (
                        torch.are_deterministic_algorithms_enabled()
                    ),
                    "cudnn_deterministic": torch.backends.cudnn.deterministic,
                    "cudnn_benchmark": torch.backends.cudnn.benchmark,
                    "cublas_workspace_config": os.environ.get(
                        "CUBLAS_WORKSPACE_CONFIG"
                    ),
                    "evaluator_source": {
                        "before": evaluator_attestation_before,
                        "after": evaluator_attestation_after,
                        "unchanged": True,
                    },
                    "checkpoint_validator_source": attest_regular_file(
                        checkpoint_evaluator_path,
                        label="frozen checkpoint validator source",
                    ),
                    "loaded_training_modules": loaded_modules_final,
                },
                "checkpoint": {
                    "directory": str(checkpoint_dir),
                    "source_root": str(bundle.source_root),
                    "full_source_manifest": bundle.source_manifest,
                    "file_hashes_before_and_after": initial_hashes,
                    "unchanged": True,
                    "completion_marker": bundle.marker,
                    **(
                        {
                            "schema6_final_bundle_revalidated_after_rollout": (
                                schema6_final_revalidated
                            )
                        }
                        if requested_profile is ConfirmationProfile.V13_300K
                        else {}
                    ),
                    **(
                        {
                            "schema8_final_bundle_revalidated_after_rollout": (
                                schema8_final_revalidated
                            )
                        }
                        if requested_profile is ConfirmationProfile.V15_300K
                        else {}
                    ),
                    **(
                        {
                            "schema9_final_bundle_revalidated_after_rollout": (
                                schema9_final_revalidated
                            )
                        }
                        if requested_profile is ConfirmationProfile.V16_300K
                        else {}
                    ),
                    **(
                        {
                            "schema10_final_bundle_revalidated_after_rollout": (
                                schema10_final_revalidated
                            )
                        }
                        if requested_profile is ConfirmationProfile.V17_300K
                        else {}
                    ),
                    **(
                        {
                            "schema11_final_bundle_revalidated_after_rollout": (
                                schema11_final_revalidated
                            )
                        }
                        if requested_profile is ConfirmationProfile.V18_300K
                        else {}
                    ),
                    **(
                        {
                            "schema12_final_bundle_revalidated_after_rollout": (
                                schema12_final_revalidated
                            )
                        }
                        if requested_profile is ConfirmationProfile.V19_300K
                        else {}
                    ),
                    **(
                        {
                            "schema13_final_bundle_revalidated_after_rollout": (
                                schema13_final_revalidated
                            )
                        }
                        if requested_profile is ConfirmationProfile.V20_300K
                        else {}
                    ),
                    **(
                        {
                            "schema7_final_bundle_revalidated_after_rollout": (
                                schema7_final_revalidated
                            )
                        }
                        if requested_profile is ConfirmationProfile.V14_300K
                        else {}
                    ),
                },
                "behavioral_training_data": {
                    **behavioral_data_state,
                    "post_validation": behavioral_data_post,
                },
                "evaluation_protocol": evaluation_protocol,
                "decision_rows_semantic_sha256": decision_semantic_hash,
                "network_state": {
                    key: value
                    for key, value in rollout_state.items()
                    if "digest" in key
                    or key
                    in (
                        "network_state_unchanged",
                        "loaded_training_modules_unchanged",
                    )
                },
                "outputs": staged_file_records,
            }
            _atomic_write_json(manifest, staged_outputs["manifest"])
            # Manifest is the last published file.  A failure after the first
            # replace retains the output lock, making a mixed generation
            # explicitly invalid until audited and recovered.
            commit_staged_generation(
                staged_outputs,
                outputs,
                lock=output_lock,
            )

    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    print(f"wrote fixed-checkpoint evaluation to {output_dir}")
    return outputs


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Completed profiled Enduro run containing finish and final checkpoints",
    )
    parser.add_argument(
        "--confirmation-profile",
        choices=tuple(profile.value for profile in ConfirmationProfile),
        default=DEFAULT_CONFIRMATION_PROFILE.value,
        help=(
            "Closed preregistered checkpoint horizon; default preserves the "
            "legacy v7 200k confirmation"
        ),
    )
    parser.add_argument(
        "--training-source-root",
        default=None,
        help="Marker-matched source snapshot root; auto-detected for snapshot runs",
    )
    parser.add_argument(
        "--source-manifest",
        default=None,
        help=(
            "Full snapshot source.sha256; defaults to the manifest adjacent "
            "to the auto-detected SNAPSHOT/src tree"
        ),
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help=(
            "Behavioral corpus root used to recompute the checkpoint training "
            "signature; defaults to checkpoint icopro_data_path"
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="External directory for decision_rows.csv, summary.json, and manifest.json",
    )
    parser.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    parser.add_argument("--num-seeds", type=int, default=DEFAULT_NUM_SEEDS)
    parser.add_argument(
        "--real-steps-per-seed",
        type=int,
        default=DEFAULT_REAL_STEPS_PER_SEED,
        help="Default gives 100k held-out real transitions across 16 streams",
    )
    parser.add_argument(
        "--calibration-unroll",
        type=int,
        default=DEFAULT_CALIBRATION_UNROLL,
        help="V-trace horizon; 201 matches the production actor unroll",
    )
    parser.add_argument(
        "--expected-rom-sha256",
        default=PREREGISTERED_ENDURO_ROM_SHA256,
        help="Pinned EnvPool Enduro ROM SHA-256 (must equal the preregistered value)",
    )
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help=(
            "Allow non-preregistered seed/horizon settings while marking the "
            "result ineligible for fixed-checkpoint confirmation"
        ),
    )
    parser.add_argument("--expected-game-id", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parsed = parser.parse_args(argv)
    if parsed.num_seeds < 1:
        parser.error("--num-seeds must be positive")
    if parsed.real_steps_per_seed < 1:
        parser.error("--real-steps-per-seed must be positive")
    if parsed.calibration_unroll < 1:
        parser.error("--calibration-unroll must be positive")
    if (
        re.fullmatch(r"[0-9a-f]{64}", str(parsed.expected_rom_sha256)) is None
        or parsed.expected_rom_sha256 != PREREGISTERED_ENDURO_ROM_SHA256
    ):
        parser.error(
            "--expected-rom-sha256 must equal the preregistered Enduro ROM hash"
        )
    return parsed


def main(argv: Optional[Sequence[str]] = None) -> int:
    evaluate(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
