import time
import timeit
import os
import numpy as np
import collections
from collections.abc import Mapping
import random
import copy
import traceback
import ray
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler

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
    imitation_checkpoint_state,
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
    def __init__(self, ray_obj, actor_param, flags, actor_net=None, device=None):
        self.flags = flags
        self.time = flags.profile
        self._logger = util.logger()
        self.dynamic_search = util.dynamic_search_enabled(flags)
        self.dynamic_factorized_control = (
            self.dynamic_search
            and bool(getattr(flags, "dynamic_factorized_control", False))
        )
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

        if self.device == torch.device("cuda"):
            self._logger.info("Init. actor-learning: Using CUDA.")
        else:
            self._logger.info("Init. actor-learning: Not using CUDA.")

       # initialize learning setting

        if not self.flags.actor_use_rms:
            self.optimizer = torch.optim.Adam(
                self.actor_net.parameters(), lr=flags.actor_learning_rate, eps=flags.actor_adam_eps
            )
        else:
            self.optimizer = torch.optim.RMSprop(
                self.actor_net.parameters(),
                lr=flags.actor_learning_rate,
                momentum=0,
                eps=0.01,
                alpha=0.99,
            )

        self.step = 0
        self.tot_eps = 0
        self.real_step = 0

        lr_lambda = lambda epoch: 1.0 - util.schedule_progress(self.flags, epoch)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)        

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
        self.actor_amp_skip_count = 0
        self.actor_amp_consecutive_skips = 0
        self._last_actor_gradient_step = ActorGradientStepResult(
            total_norm=0.0,
            optimizer_stepped=True,
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
        
        # move network and optimizer to process device
        self.actor_net.to(self.device)
        util.optimizer_to(self.optimizer, self.device)    
        self._init_imitation_components()

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
    
        if self.flags.float16:
            self.scaler = GradScaler(init_scale=2**8)
        
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
        self._restore_training_rng_state()

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
            self.actor_net, self.bc_model_net, self.flags, device=self.device
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
        data_ptr = self.actor_buffer.read.remote()                    
        successful = False
        try:
            while self.real_step < self.flags.total_steps:
                if timing is not None:
                    timing.reset()
                # get data remotely
           
                while True:
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
                data = (train_actor_out, initial_actor_state)
                # start consume data
                self.consume_data(data, timing=timing)
                del train_actor_out, initial_actor_state, data
                
                self.actor_param_buffer.set_data.remote(
                    "actor_net", self.actor_net.get_weights()
                )
                if timing is not None:
                    timing.time("set weight")            
          
            self._logger.info("Terminating actor-learning thread")
            self.save_checkpoint(force=True)
            successful = True
            return True
        except Exception as e:
            self._logger.error(f"Exception detected in learn_actor: {e}")
            self._logger.error(traceback.format_exc())
            raise
        finally:
            self.close(successful=successful)
        
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
                # 비동기 호출이므로 결과를 기다리지 않음
                if self.real_step % 1000 == 0:
                    self._logger.info(f"Sent real_step update to ActorBuffer: {self.real_step}")
            except Exception as e:
                self._logger.error(f"Error updating ActorBuffer real_step: {e}")
                traceback.print_exc()

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

    def consume_data_single(self, data, timing=None, first_iter=True, last_iter=False):

        train_actor_out, initial_actor_state = data
        actor_id = train_actor_out.id
        T, B = train_actor_out.done.shape

        # compute losses
        out = self.compute_losses(
            train_actor_out, initial_actor_state, first_iter, last_iter
        )
        losses, train_actor_out = out
        total_loss = losses["total_loss"]
        _require_finite_tensor("actor total loss", total_loss)
        if timing is not None:
            timing.time("compute loss")

        # gradient descent on loss
        self.optimizer.zero_grad()
        if self.flags.float16:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()
        if timing is not None:
            timing.time("compute gradient")

        optimize_params = self.optimizer.param_groups[0]["params"]
        if self.flags.float16:
            self.scaler.unscale_(self.optimizer)
        step_result = self._step_actor_optimizer(optimize_params, T, B)
        total_norm = step_result.total_norm
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
        if timing is not None:
            timing.time("grad descent")
    
        self.scheduler.last_epoch = (
            max(self.real_step - 1, 0)
        )  # scheduler does not support setting epoch directly
        self.scheduler.step()
        self.anneal_c = 1.0 - util.schedule_progress(self.flags, self.real_step)
        
        if not self.ppo_enable or first_iter:
            # statistic output
            for k in losses:
                # Imitation metrics are already normalized over scored real
                # decisions; SEARCH/WAIT unroll size must not rescale them.
                if not k.startswith("icopro_"):
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
                        total_loss/T/B,
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

        for i in range(self.num_rewards):
            prefix = self.rewards_ls[i]
            prefix_rewards = rewards[:, :, i]
            behavior_log_prob = behavior_log_prob_by_prefix[prefix]
            target_log_prob = target_log_prob_by_prefix[prefix]
            
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
                "actor_net_state_dict": self.actor_net.state_dict(),
                "actor_arch_version": 2 if self.dynamic_search else 1,
                "dynamic_search": self.dynamic_search,
                "dynamic_factorized_control": self.dynamic_factorized_control,
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
        self.actor_amp_skip_count = int(
            train_checkpoint.get("actor_amp_skip_count", 0)
        )
        self.actor_amp_consecutive_skips = int(
            train_checkpoint.get("actor_amp_consecutive_skips", 0)
        )
        util.load_optimizer(self.optimizer, train_checkpoint["actor_net_optimizer_state_dict"])
        util.load_scheduler(self.scheduler, train_checkpoint["actor_net_scheduler_state_dict"])
        self.actor_net.set_weights(train_checkpoint["actor_net_state_dict"])
        self._logger.info("Loaded actor checkpoint from %s" % ckp_path)

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

    def close(self, successful=True):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        if self.bc_runner is not None:
            self.bc_runner.close()
        if hasattr(self, "actor_buffer") and self.actor_buffer is not None:
            self.actor_buffer.set_finish.remote()
        self.plogger.close(successful=bool(successful))


@ray.remote
class ActorLearner(SActorLearner):
    pass
