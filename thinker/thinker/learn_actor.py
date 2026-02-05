import time
import timeit
import os
import numpy as np
import collections
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
from thinker.actor_net import ActorNet
import thinker.util as util
from thinker.buffer import RetBuffer
from gymnasium import spaces
from thinker.bc_loader import FrameStackedBehavioralDataLoader
from thinker.dataset_env import BehaviorSequenceVectorEnv
from thinker.model_net import ModelNet
from thinker.cenv import cModelWrapper as OnlineCModelWrapper
from thinker.cenv_bc import cModelWrapper as BCCModelWrapper

def dqfd_margin_loss(q_values: torch.Tensor, actions: torch.Tensor, margin: torch.Tensor) -> torch.Tensor:
    if q_values.ndim != 2:
        raise ValueError("q_values must be 2D")
    batch_size, num_actions = q_values.shape
    actions = actions.view(batch_size, 1)
    margin = margin.view(batch_size, 1)
    margin_matrix = margin.repeat(1, num_actions)
    zeros = torch.zeros_like(margin)
    margin_matrix.scatter_(1, actions, zeros)
    q_selected = q_values.gather(1, actions)
    max_margin = torch.max(q_values + margin_matrix, dim=1, keepdim=True)[0]
    loss = max_margin - q_selected
    return loss.mean()

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

class SActorLearner:
    def __init__(self, ray_obj, actor_param, flags, actor_net=None, device=None):
        self.flags = flags
        self.time = flags.profile
        self._logger = util.logger()

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
            if self.actor_param_buffer is not None:
                try:
                    self.actor_param_buffer.set_data.remote("actor_net_init_params", actor_param)
                except Exception as exc:
                    self._logger.warning(f"Failed to store actor init params: {exc}")
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

        icopro_dev = getattr(self.flags, "icopro_device", "cpu")
        try:
            self.icopro_device = torch.device(icopro_dev if isinstance(icopro_dev, str) else "cpu")
        except Exception:
            self.icopro_device = torch.device("cpu")
            setattr(self.flags, "icopro_device", "cpu")

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

        lr_lambda = (
            lambda epoch: 1
            - min(epoch, self.flags.total_steps) / self.flags.total_steps
        )
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
        self.ret_buffers["len"] = RetBuffer(max_actor_id, mean_n=400)
        self.im_discounting = self.flags.discounting ** (1 / self.flags.rec_t)

        self.rewards_ls = ["re"]
        if flags.im_cost > 0.0:
            self.rewards_ls += ["im"]
        if flags.cur_cost > 0.0:
            self.rewards_ls += ["cur"]
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
                self.tar_im_entropy = -flags.tar_im_entropy_scale * torch.log(1 / torch.tensor(self.actor_net.num_actions * self.actor_net.dim_actions))   
                self.tar_im_entropy += -flags.tar_im_entropy_scale * torch.log(1 / torch.tensor(2))   # reset action
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
        self._init_bc_components()
        self.latest_icopro_actor_stats = None

    def _init_bc_components(self):
        self.bc_loader = None
        self.bc_model_net = None
        self.bc_policy_adapter = None
        self.bc_optimizer = None
        self.bc_enabled = False
        self.action_prior = None
        self.action_prior_ema = None
        self.action_prior_ema_beta = float(getattr(self.flags, "action_prior_ema", 0.05))
        self.bc_step = 0
        self.bc_batch_size = max(
            1, int(getattr(self.flags, "icopro_batch_size", 32))
        )
        self.bc_supervised_freq = max(
            1, int(getattr(self.flags, "icopro_supervised_freq", 1))
        )
        self.bc_margin = float(getattr(self.flags, "icopro_margin", 0.05))
        self.bc_margin_coef = float(getattr(self.flags, "icopro_margin_coef", 1.0))
        self.bc_action_diff_coef = float(
            getattr(self.flags, "icopro_action_diff_coef", 1.0)
        )
        self.bc_pvp_coef = float(getattr(self.flags, "icopro_pvp_coef", 0.0))
        self.bc_tree_coef = float(getattr(self.flags, "icopro_tree_coef", 0.0))
        self.bc_kl_coef = float(getattr(self.flags, "icopro_actor_kl_coef", 0.0))
        self.bc_seq_len = max(1, int(getattr(self.flags, "batch_length", 1)))
        self.bc_noop_window = collections.deque(maxlen=150)
        # Cached planner/env for IcoPro BC to avoid repeated cModelWrapper
        # allocations and reduce GPU memory growth.
        self.bc_planner = None
        self.bc_planner_env = None
        self.bc_planner_batch_size = 0
        self.bc_planner_seq_len = 0
        self.bc_planner_device = None

        data_path = getattr(self.flags, "icopro_data_path", "")
        if not data_path:
            return
        data_path = os.path.abspath(data_path)
        subjects_raw = str(getattr(self.flags, "icopro_subjects", ""))
        try:
            subjects = [int(s.strip()) for s in subjects_raw.split(",") if s.strip()]
        except ValueError:
            self._logger.warning(f"Invalid icopro_subjects '{subjects_raw}'; disabling supervised actor loss.")
            return
        if not subjects:
            self._logger.warning("No valid icopro_subjects provided; disabling supervised actor loss.")
            return
        game_id = int(getattr(self.flags, "icopro_game_id", 0))
        try:
            self.bc_loader = FrameStackedBehavioralDataLoader(
                base_path=data_path,
                subjects=subjects,
                game_id=game_id,
                frame_stack_n=self.flags.frame_stack_n,
                target_size=(84, 84),
                grayscale=self.flags.grayscale,
                normalize=True,
            )
        except Exception as exc:
            self._logger.warning(f"Failed to initialise IcoPro actor data loader: {exc}")
            self.bc_loader = None
            return
        if len(self.bc_loader.data_files) == 0:
            self._logger.warning("IcoPro actor data loader found no files; disabling supervised actor loss.")
            self.bc_loader = None
            return
        action_prior = np.asarray(self.bc_loader.action_distribution, dtype=np.float32)
        if action_prior.shape[0] != self.actor_net.num_actions:
            self._logger.warning(
                "Human action prior size mismatch: %d (data) vs %d (policy); skipping prior regularizer.",
                action_prior.shape[0],
                self.actor_net.num_actions,
            )
        else:
            prior_sum = float(np.sum(action_prior))
            if prior_sum <= 0.0 or not np.isfinite(prior_sum):
                action_prior = np.full(
                    self.actor_net.num_actions,
                    1.0 / self.actor_net.num_actions,
                    dtype=np.float32,
                )
            else:
                action_prior = action_prior / prior_sum
            self.action_prior = torch.tensor(
                action_prior, dtype=torch.float32, device=self.device
            )
            self._logger.info("Loaded human action prior for RL regularizer.")
        lr = float(getattr(self.flags, "icopro_actor_lr", 0.0))
        if lr <= 0:
            self._logger.info("icopro_actor_lr <= 0; supervised actor updates disabled.")
            return
        self.bc_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=lr)
        self.bc_enabled = True
        self._logger.info(f"IcoPro actor data loader initialised with {len(self.bc_loader.data_files)} files (subjects={subjects}, game_id={game_id}).")

    def _ensure_bc_model_net(self):
        if not self.bc_enabled or self.bc_model_net is not None:
            return
        real_state_shape = getattr(self.actor_net, "real_states_shape", None)
        pri_action_space = getattr(self.actor_net, "pri_action_space", None)
        if real_state_shape is None or pri_action_space is None:
            self._logger.warning("Unable to determine actor spaces for IcoPro model; disabling supervised loss.")
            self.bc_enabled = False
            return
        model_obs_space = spaces.Box(low=0, high=255, shape=real_state_shape, dtype=np.uint8)
        icopro_device = self.icopro_device
        try:
            self.bc_model_net = ModelNet(obs_space=model_obs_space, action_space=pri_action_space, flags=self.flags).to(icopro_device)
        except Exception as exc:
            self._logger.warning(
                f"Failed to initialise IcoPro model net on device '{icopro_device}': {exc}. Falling back to CPU."
            )
            try:
                self.bc_model_net = ModelNet(obs_space=model_obs_space, action_space=pri_action_space, flags=self.flags).to("cpu")
                icopro_device = torch.device("cpu")
                setattr(self.flags, "icopro_device", "cpu")
                self.icopro_device = icopro_device
            except Exception as exc2:
                self._logger.warning(f"Unable to create IcoPro model net on CPU: {exc2}; disabling supervised loss.")
                self.bc_model_net = None
                self.bc_enabled = False
                return
        self.bc_model_net.eval()
        for param in self.bc_model_net.parameters():
            param.requires_grad = False
        if self.model_param_buffer is not None:
            self._refresh_bc_model()
        self._logger.info("IcoPro model net initialised for BC planning.")

    def _ensure_bc_planner(
        self,
        obs_seq_np: np.ndarray,
        rewards_seq_np: np.ndarray,
        actions_seq_np: np.ndarray,
    ):
        """Reuse a single cModelWrapper planner for IcoPro BC.

        This avoids repeatedly allocating large planner buffers on the device.
        """
        batch_size, seq_len = actions_seq_np.shape
        model_device = self.icopro_device
        rebuild = (
            self.bc_planner is None
            or self.bc_planner_env is None
            or self.bc_planner_batch_size != batch_size
            or self.bc_planner_seq_len != seq_len
            or self.bc_planner_device != model_device
        )
        if rebuild:
            if self.bc_planner is not None and hasattr(self.bc_planner, "close"):
                try:
                    self.bc_planner.close()
                except Exception:
                    pass
            base_env = BehaviorSequenceVectorEnv(
                obs_seq=obs_seq_np,
                rewards_seq=rewards_seq_np,
                actions_seq=actions_seq_np,
                num_actions=self.actor_net.num_actions,
            )
            planner = BCCModelWrapper(
                env=base_env,
                env_n=batch_size,
                flags=self.flags,
                model_net=self.bc_model_net,
                device=model_device,
                timing=False,
            )
            self.bc_planner_env = base_env
            self.bc_planner = planner
            self.bc_planner_batch_size = batch_size
            self.bc_planner_seq_len = seq_len
            self.bc_planner_device = model_device
        else:
            try:
                self.bc_planner_env.update_sequences(
                    obs_seq_np, rewards_seq_np, actions_seq_np
                )
            except Exception:
                if self.bc_planner is not None and hasattr(self.bc_planner, "close"):
                    try:
                        self.bc_planner.close()
                    except Exception:
                        pass
                base_env = BehaviorSequenceVectorEnv(
                    obs_seq=obs_seq_np,
                    rewards_seq=rewards_seq_np,
                    actions_seq=actions_seq_np,
                    num_actions=self.actor_net.num_actions,
                )
                planner = BCCModelWrapper(
                    env=base_env,
                    env_n=batch_size,
                    flags=self.flags,
                    model_net=self.bc_model_net,
                    device=model_device,
                    timing=False,
                )
                self.bc_planner_env = base_env
                self.bc_planner = planner
                self.bc_planner_batch_size = batch_size
                self.bc_planner_seq_len = seq_len
                self.bc_planner_device = model_device

        return self.bc_planner, model_device

    def _refresh_bc_model(self):
        if self.model_param_buffer is None or self.bc_model_net is None:
            return
        try:
            weights = ray.get(self.model_param_buffer.get_data.remote("model_net"))
        except Exception:
            return
        if weights is not None:
            try:
                self.bc_model_net.set_weights(weights)
            except Exception as exc:
                self._logger.warning(f"Failed to refresh IcoPro model weights: {exc}")

    def _compute_bc_seq_loss(self, batch):
        """Sequential imitation loss over length-L non-overlapping stacks using cModelWrapper planning."""
        self._ensure_bc_model_net()
        if self.bc_model_net is None:
            return None
        if self.model_param_buffer is not None:
            self._refresh_bc_model()

        obs_seq_full = batch["obs_seq"]
        actions_seq_full = batch["actions_seq"]
        rewards_seq_full = batch.get("rewards_seq")

        # Drop the first frame-stack; imitate on the remaining L steps
        obs_seq = obs_seq_full[:, 1:]
        human_actions_seq = actions_seq_full[:, 1:]
        rewards_seq = (
            rewards_seq_full[:, 1:] if rewards_seq_full is not None else None
        )

        batch_size, seq_len = human_actions_seq.shape
        actor_device = self.device

        # Planner runs on icopro_device (can be CPU or GPU). Only minimal
        # tensors are moved to the actor device.
        obs_np = (obs_seq.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        if rewards_seq is not None:
            rewards_np = rewards_seq.cpu().numpy().astype(np.float32)
        else:
            rewards_np = np.zeros((batch_size, seq_len), dtype=np.float32)
        human_actions_seq_np = human_actions_seq.cpu().numpy()

        planner, model_device = self._ensure_bc_planner(
            obs_np, rewards_np, human_actions_seq_np
        )

        with torch.no_grad():
            states, info = planner.reset(self.bc_model_net)

        env_out = util.init_env_out(
            states,
            info,
            self.flags,
            dim_actions=self.actor_net.dim_actions,
            tuple_action=self.actor_net.tuple_action,
        )

        def _to_actor(x):
            return x.to(actor_device) if torch.is_tensor(x) else x

        env_out = env_out._replace(
            real_states=_to_actor(env_out.real_states),
            tree_reps=_to_actor(env_out.tree_reps),
            xs=_to_actor(env_out.xs) if env_out.xs is not None else None,
            hs=_to_actor(env_out.hs) if env_out.hs is not None else None,
            done=_to_actor(env_out.done),
            real_done=_to_actor(env_out.real_done),
            truncated_done=_to_actor(env_out.truncated_done),
            last_pri=_to_actor(env_out.last_pri),
            last_reset=_to_actor(env_out.last_reset),
            reward=_to_actor(env_out.reward),
            step_status=_to_actor(env_out.step_status),
        )

        actor_state = self.actor_net.initial_state(
            batch_size=batch_size, device=actor_device
        )

        # Track imaginary / curiosity episode returns (PostWrapper-lite for BC planner)
        im_ep_ret = torch.zeros(batch_size, device=actor_device)
        cur_ep_ret = torch.zeros(batch_size, device=actor_device)

        margin_losses, pvp_losses, action_diff_losses = [], [], []
        all_pred_actions = []
        all_human_actions = []

        for t in range(seq_len):
            actor_out, actor_state = self.actor_net(
                env_out, actor_state, compute_loss=False, greedy=False
            )
            if not self.actor_net.disable_thinker:
                primary_action, reset_action = actor_out.action
            else:
                primary_action, reset_action = actor_out.action, None

            with torch.no_grad():
                states, reward, done, truncated, info = planner.step(
                    (
                        primary_action.detach().cpu().numpy(),
                        reset_action.detach().cpu().numpy()
                        if reset_action is not None
                        else np.zeros_like(primary_action.detach().cpu().numpy()),
                    ),
                    self.bc_model_net,
                )

            # accumulate episode returns for imaginary / curiosity rewards if present
            real_done = info.get("real_done", done)
            if isinstance(real_done, torch.Tensor):
                real_done_t = real_done.to(actor_device)
            else:
                real_done_t = torch.tensor(real_done, device=actor_device)

            im_reward = info.get("im_reward")
            if im_reward is not None:
                if isinstance(im_reward, torch.Tensor):
                    im_r = im_reward.view(-1).to(actor_device)
                else:
                    im_r = torch.tensor(im_reward).view(-1).to(actor_device)
                im_ep_ret += im_r
            cur_reward = info.get("cur_reward")
            if cur_reward is not None:
                if isinstance(cur_reward, torch.Tensor):
                    cur_r = cur_reward.view(-1).to(actor_device)
                else:
                    cur_r = torch.tensor(cur_reward).view(-1).to(actor_device)
                cur_ep_ret += cur_r

            # inject episode return fields expected by create_env_out
            info["im_episode_return"] = im_ep_ret.clone()
            info["cur_episode_return"] = cur_ep_ret.clone()

            # reset accumulators on real episode end
            if real_done_t is not None:
                im_ep_ret = im_ep_ret * (~real_done_t)
                cur_ep_ret = cur_ep_ret * (~real_done_t)

            env_out = util.create_env_out(
                actor_out.action, states, reward, done, truncated, info, flags=self.flags
            )

            env_out = env_out._replace(
                real_states=_to_actor(env_out.real_states),
                tree_reps=_to_actor(env_out.tree_reps),
                xs=_to_actor(env_out.xs) if env_out.xs is not None else None,
                hs=_to_actor(env_out.hs) if env_out.hs is not None else None,
                done=_to_actor(env_out.done),
                real_done=_to_actor(env_out.real_done),
                truncated_done=_to_actor(env_out.truncated_done),
                last_pri=_to_actor(env_out.last_pri),
                last_reset=_to_actor(env_out.last_reset),
                reward=_to_actor(env_out.reward),
                step_status=_to_actor(env_out.step_status),
            )

            logits_step = actor_out.pri_param[-1]
            # ensure 2D [B, num_actions]
            if logits_step.dim() == 4:
                logits_step = logits_step[:, 0]
            if logits_step.dim() == 3:
                logits_step = logits_step[:, 0, :]
            human_actions = human_actions_seq[:, t]
            all_human_actions.append(human_actions)
            pred_actions = torch.argmax(logits_step.detach(), dim=-1)
            all_pred_actions.append(pred_actions)

            margin_tensor = torch.full(
                (human_actions.shape[0],),
                self.bc_margin,
                dtype=torch.float32,
                device=actor_device,
            )
            margin_losses.append(dqfd_margin_loss(logits_step, human_actions, margin_tensor))
            if self.bc_pvp_coef > 0.0:
                q_human = logits_step.gather(1, human_actions.unsqueeze(1)).squeeze(1)
                q_agent = logits_step.gather(1, pred_actions.unsqueeze(1)).squeeze(1)
                pvp_pos = F.mse_loss(q_human, torch.ones_like(q_human))
                diff = (pred_actions != human_actions).float()
                if diff.sum() > 0:
                    pvp_neg = F.mse_loss(diff * q_agent, diff * (-torch.ones_like(q_agent)))
                else:
                    pvp_neg = torch.zeros_like(pvp_pos)
                pvp_losses.append(pvp_pos + pvp_neg)
            # action difference loss: cross-entropy between logits and human action
            step_diff = F.cross_entropy(logits_step, human_actions)
            action_diff_losses.append(step_diff)

        margin_loss = (
            torch.stack(margin_losses).mean()
            if margin_losses
            else torch.zeros((), device=actor_device)
        )
        pvp_loss = (
            torch.stack(pvp_losses).mean()
            if pvp_losses
            else torch.zeros((), device=actor_device)
        )
        kl_loss = torch.zeros((), device=actor_device)
        action_diff_loss = (
            torch.stack(action_diff_losses).mean()
            if action_diff_losses
            else torch.zeros((), device=actor_device)
        )

        total_loss = (
            self.bc_margin_coef * margin_loss
            + self.bc_pvp_coef * pvp_loss
            + self.bc_kl_coef * kl_loss
            + self.bc_action_diff_coef * action_diff_loss
        )

        if all_pred_actions and all_human_actions:
            pred_cat = torch.cat(all_pred_actions, dim=0)
            human_cat = torch.cat(all_human_actions, dim=0)
            accuracy = float((pred_cat == human_cat).float().mean().item())
            self.bc_noop_window.extend(pred_cat.tolist())
            # per-batch noop frequency (action index 0 proportion)
            batch_noop_freq = float((pred_cat == 0).float().mean().item())
        else:
            accuracy = 0.0
            batch_noop_freq = 0.0

        noop_freq = 0.0
        if len(self.bc_noop_window) > 0:
            noop_freq = sum(1 for a in self.bc_noop_window if a == 0) / len(self.bc_noop_window)

        return {
            "total_loss": total_loss,
            "margin_loss": margin_loss,
            "action_diff_loss": action_diff_loss,
            "pvp_loss": pvp_loss,
            "kl_loss": kl_loss,
            "accuracy": accuracy,
            "noop_freq": noop_freq,
            "noop_freq_batch": batch_noop_freq,
        }

    def _sample_bc_batch(self):
        if self.bc_loader is None:
            return None
        if self.bc_seq_len > 1 and hasattr(self.bc_loader, "get_sequence_batch"):
            seq_batch = self.bc_loader.get_sequence_batch(
                batch_size=self.bc_batch_size,
                sequence_length=self.bc_seq_len + 1,
            )
            if seq_batch is not None and "obs_seq" in seq_batch:
                device = self.device
                obs_seq = torch.from_numpy(seq_batch["obs_seq"]).float().to(device)
                actions_seq = torch.from_numpy(
                    np.asarray(seq_batch["actions_seq"], dtype=np.int64)
                ).long().to(device)
                rewards_seq = torch.from_numpy(
                    np.asarray(
                        seq_batch.get(
                            "rewards_seq",
                            np.zeros_like(seq_batch["actions_seq"], dtype=np.float32),
                        ),
                        dtype=np.float32,
                    )
                ).to(device)
                sequence_starts = torch.from_numpy(
                    np.asarray(
                        seq_batch.get(
                            "sequence_starts",
                            np.ones(actions_seq.shape[0], dtype=np.bool_),
                        ),
                        dtype=np.bool_,
                    )
                ).to(device)
                return {
                    "obs_seq": obs_seq,
                    "actions_seq": actions_seq,
                    "rewards_seq": rewards_seq,
                    "sequence_starts": sequence_starts,
                }
        batch = self.bc_loader.get_paired_batch(batch_size=self.bc_batch_size)
        if batch is None:
            return None
        device = self.device
        obs = torch.from_numpy(batch["next_obs"]).float().to(device)
        prev_obs = torch.from_numpy(batch["obs"]).float().to(device)
        rewards = torch.from_numpy(
            np.asarray(batch.get("rewards", np.zeros(obs.shape[0], dtype=np.float32)), dtype=np.float32)
        ).to(device)
        dones_np = batch.get("dones")
        if dones_np is not None:
            dones = torch.from_numpy(np.asarray(dones_np, dtype=np.float32)).to(device)
        else:
            dones = torch.zeros_like(rewards)
        actions_np = np.asarray(batch["actions"])
        if actions_np.ndim > 1 and actions_np.shape[-1] > 1:
            actions_idx = actions_np.argmax(axis=-1)
        else:
            actions_idx = actions_np.reshape(-1)
        actions = torch.from_numpy(actions_idx.astype(np.int64)).long().to(device)
        prev_actions = torch.from_numpy(
            np.asarray(batch.get("prev_actions", actions_idx), dtype=np.int64)
        ).long().to(device)
        action_probs = torch.from_numpy(batch["curr_action_onehot"]).float().to(device)
        sequence_starts_np = batch.get("sequence_starts")
        if sequence_starts_np is not None:
            sequence_starts = torch.from_numpy(sequence_starts_np.astype(np.bool_)).to(device)
        else:
            sequence_starts = torch.ones_like(prev_actions, dtype=torch.bool)
        return {
            "obs": obs,
            "prev_obs": prev_obs,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "prev_actions": prev_actions,
            "sequence_starts": sequence_starts,
            "action_probs": action_probs,
        }

    def _compute_bc_loss(self):
        if not self.bc_enabled or self.bc_loader is None:
            return None
        self._ensure_bc_model_net()
        if self.bc_model_net is None:
            return None
        batch = self._sample_bc_batch()
        if batch is None:
            return None
        if "obs_seq" in batch:
            return self._compute_bc_seq_loss(batch)
        # Fallback: simple BC without planner when only paired batch is available
        obs = batch["obs"]
        human_actions = batch["actions"]
        logits = self.actor_net.policy(self.actor_net.normalize(obs))
        logits = logits.view(obs.shape[0], self.actor_net.dim_actions, self.actor_net.num_actions)
        if logits.dim() == 3:
            logits = logits[:, 0, :]
        margin_tensor = torch.full((human_actions.shape[0],), self.bc_margin, dtype=torch.float32, device=self.device)
        margin_loss = dqfd_margin_loss(logits, human_actions, margin_tensor)
        # action difference loss: cross-entropy between logits and human action
        action_diff_loss = F.cross_entropy(logits, human_actions)
        kl_loss = torch.zeros((), device=self.device)
        pvp_loss = torch.zeros((), device=self.device)
        total_loss = (
            self.bc_margin_coef * margin_loss
            + self.bc_pvp_coef * pvp_loss
            + self.bc_kl_coef * kl_loss
            + self.bc_action_diff_coef * action_diff_loss
        )
        accuracy = float((torch.argmax(logits.detach(), dim=-1) == human_actions).float().mean().item())
        return {
            "total_loss": total_loss,
            "margin_loss": margin_loss,
            "action_diff_loss": action_diff_loss,
            "pvp_loss": pvp_loss,
            "kl_loss": kl_loss,
            "accuracy": accuracy,
        }

    def _maybe_run_bc_update(self):
        if not self.bc_enabled or self.bc_loader is None or self.bc_optimizer is None:
            return None
        self.bc_step += 1
        if self.bc_step % self.bc_supervised_freq != 0:
            return None
        metrics = self._compute_bc_loss()
        if metrics is None:
            return None
        total_loss = metrics["total_loss"]
        self.bc_optimizer.zero_grad()
        total_loss.backward()
        if self.flags.actor_grad_norm_clipping > 0:
            torch.nn.utils.clip_grad_norm_(
                self.actor_net.parameters(), self.flags.actor_grad_norm_clipping
            )
        self.bc_optimizer.step()
        out = {
            "total_loss": total_loss.detach().cpu().item(),
            "margin_loss": metrics["margin_loss"].detach().cpu().item(),
            "pvp_loss": metrics["pvp_loss"].detach().cpu().item(),
            "kl_loss": metrics["kl_loss"].detach().cpu().item() if torch.is_tensor(metrics["kl_loss"]) else float(metrics["kl_loss"]),
            "accuracy": metrics["accuracy"],
        }
        if "action_diff_loss" in metrics:
            val = metrics["action_diff_loss"]
            if val is None:
                out["action_diff_loss"] = 0.0
            else:
                out["action_diff_loss"] = val.detach().cpu().item() if torch.is_tensor(val) else float(val)
        if "noop_freq" in metrics:
            out["noop_freq"] = metrics["noop_freq"]
            self._logger.info(
                f"IcoPro actor BC seq accuracy={metrics.get('accuracy', 0.0):.3f} noop_freq_150={metrics.get('noop_freq', 0.0):.3f}"
            )
        if "noop_freq_batch" in metrics:
            out["noop_freq_batch"] = metrics["noop_freq_batch"]
        return out

    def learn_data(self):
        timing = util.Timings() if self.time else None
        data_ptr = self.actor_buffer.read.remote()                    
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
            self.close()
            return True
        except Exception as e:
            self._logger.error(f"Exception detected in learn_actor: {e}")
            self._logger.error(traceback.format_exc())
        finally:
            self.close()
            return True
        
    def consume_data(self, data, timing=None):

        train_actor_out, initial_actor_state = data
        T, B, *_ = train_actor_out.episode_return.shape
        self.step += T * B
        last_step_real = (train_actor_out.step_status == 0) | (train_actor_out.step_status == 3)
        self.real_step += torch.sum(last_step_real).item()
        real_done_count = torch.sum(train_actor_out.real_done).item()
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

    def consume_data_single(self, data, timing=None, first_iter=True, last_iter=False):

        train_actor_out, initial_actor_state = data
        actor_id = train_actor_out.id
        T, B = train_actor_out.done.shape

        if self.actor_param_buffer is not None:
            try:
                weights = ray.get(self.actor_param_buffer.get_data.remote("actor_net"))
            except Exception:
                weights = None
            if weights is not None:
                self.actor_net.set_weights(weights)

        # compute losses
        out = self.compute_losses(
            train_actor_out, initial_actor_state, first_iter, last_iter
        )
        losses, train_actor_out = out
        total_loss = losses["total_loss"]
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
        if self.flags.actor_grad_norm_clipping > 0:
            total_norm = torch.nn.utils.clip_grad_norm_(
                optimize_params, self.flags.actor_grad_norm_clipping * T * B
            )
            total_norm = total_norm.detach().cpu().item()
        else:
            total_norm = util.compute_grad_norm(optimize_params)
        if timing is not None:
            timing.time("compute norm")

        if self.flags.float16:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        if timing is not None:
            timing.time("grad descent")
    
        self.scheduler.last_epoch = (
            max(self.real_step - 1, 0)
        )  # scheduler does not support setting epoch directly
        self.scheduler.step()
        self.anneal_c = max(1 - self.real_step / self.flags.total_steps, 0)

        icopro_stats = self._maybe_run_bc_update()
        if icopro_stats is not None:
            self.latest_icopro_actor_stats = icopro_stats

        if not self.ppo_enable or first_iter:
            # statistic output
            for k in losses:
                if not k.startswith('icopro_'):
                    losses[k] = losses[k] / T / B
            total_norm = total_norm / T / B
            stats = self.compute_stat(train_actor_out, losses, total_norm, actor_id)
            stats["sps"] = self.sps
            if self.latest_icopro_actor_stats:
                stats["icopro/actor/total_loss"] = self.latest_icopro_actor_stats["total_loss"]
                stats["icopro/actor/margin_loss"] = self.latest_icopro_actor_stats["margin_loss"]
                stats["icopro/actor/pvp_loss"] = self.latest_icopro_actor_stats["pvp_loss"]
                stats["icopro/actor/kl_loss"] = self.latest_icopro_actor_stats.get("kl_loss", 0.0)
                if "action_diff_loss" in self.latest_icopro_actor_stats:
                    stats["icopro/actor/action_diff_loss"] = self.latest_icopro_actor_stats["action_diff_loss"]
                if "noop_freq" in self.latest_icopro_actor_stats:
                    stats["icopro/actor/noop_freq_150"] = self.latest_icopro_actor_stats["noop_freq"]
                stats["icopro/actor/accuracy"] = self.latest_icopro_actor_stats["accuracy"]

            # write to log file
            self.plogger.log(stats)
            self.latest_icopro_actor_stats = None

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
                if self.flags.return_norm_type in [0, 1]:
                    print_str += " norm_diff %.4f/%.4f" % (
                        stats["actor/norm_diff"],
                        stats.get("actor/im_norm_diff", 0.),
                    )
                    print_str += " cur_norm_diff %.4f" % (
                        stats.get("actor/cur_norm_diff", 0.),
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
                base_reset_logits = base_actor_out.reset_logits.detach()
        rewards = train_actor_out.reward

        # compute advantage and baseline        
        pg_losses = []
        baseline_losses = []
        done = train_actor_out.done | train_actor_out.truncated_done
        discounts = [(~done).float() * self.im_discounting]
        masks = [None]

        last_step_real = (train_actor_out.step_status == 0) | (train_actor_out.step_status == 3)
        next_step_real = (train_actor_out.step_status == 2) | (train_actor_out.step_status == 3)        
        
        if self.flags.im_cost > 0.:
            discounts.append((~next_step_real).float() * self.im_discounting)            
            masks.append((~last_step_real).float())
        if self.flags.cur_cost > 0.:
            discounts.append((~done).float() * self.im_discounting)            
            masks.append(None)

        # optional gradient re-weighting based on step times (no reward shaping)
        step_time_cost = float(getattr(self.flags, "step_time_cost", 0.0))
        step_time_cost_reward = float(getattr(self.flags, "step_time_cost_reward", 0.0))
        step_time_cost_weight = float(getattr(self.flags, "step_time_cost_weight", step_time_cost))
        time_weights = None
        step_time_loss = None
        step_time_penalty = None
        if (step_time_cost_weight > 0.0 or step_time_cost_reward > 0.0) and not self.disable_thinker:
            step_times = getattr(train_actor_out, "step_times", None)
            if step_times is not None:
                st = step_times
                if not isinstance(st, torch.Tensor):
                    st = torch.tensor(st, device=self.device, dtype=torch.float32)
                if st.dim() > 2:
                    st = st.sum(dim=-1)  # (T,B,K) -> (T,B)
                st = st.clone()
                st[torch.isnan(st)] = 0.0
                if st.numel() > 0:
                    # log-only metric
                    step_time_loss = st.mean()
                    # normalize and build per-step weights in [0,1]
                    st_norm_weight = (st * 1e6).clamp(min=0.0)
                    time_weights = 1.0 / (1.0 + step_time_cost_weight * st_norm_weight)
                    if step_time_cost_reward > 0.0:
                        # reward shaping uses a milder ms scale
                        st_norm_reward = (st * 1e3).clamp(min=0.0)
                        step_time_penalty = step_time_cost_reward * st_norm_reward

        if step_time_penalty is not None:
            st_pen = step_time_penalty
            if st_pen.dim() == 1:
                st_pen = st_pen.view(-1, 1)
            if st_pen.dim() == 2:
                # align batch dim; allow (B,T) transpose case
                if st_pen.shape[0] == rewards.shape[1] and st_pen.shape[1] == rewards.shape[0]:
                    st_pen = st_pen.transpose(0, 1)
                elif st_pen.shape[1] == 1 and rewards.shape[1] > 1:
                    st_pen = st_pen.expand(-1, rewards.shape[1])
            if st_pen.shape[0] != rewards.shape[0]:
                st_pen = st_pen[-rewards.shape[0]:]
            if st_pen.shape[1] != rewards.shape[1]:
                if st_pen.shape[1] == 1:
                    st_pen = st_pen.expand(-1, rewards.shape[1])
                else:
                    raise ValueError(f"step_time_penalty shape {st_pen.shape} incompatible with rewards {rewards.shape[:2]}")
            st_pen = st_pen.unsqueeze(-1)  # (T,B,1)
            rewards = rewards.clone()
            rewards[:, :, 0] = rewards[:, :, 0] - st_pen.squeeze(-1)

        if not self.ppo_enable or self.flags.ppo_v_trace:
            log_rhos = new_actor_out.c_action_log_prob - train_actor_out.c_action_log_prob
        else:
            log_rhos = torch.zeros_like(train_actor_out.c_action_log_prob)

        for i in range(self.num_rewards):
            prefix = self.rewards_ls[i]
            prefix_rewards = rewards[:, :, i]
            
            if self.flags.entropy_r_cost > 0. and prefix == "re":
                prefix_rewards[last_step_real] += -self.flags.entropy_r_cost * train_actor_out.c_action_log_prob[last_step_real]

            return_norm_type=self.flags.return_norm_type 
            if not self.ppo_enable:
                values = new_actor_out.baseline[:, :, i]
            else:
                values = train_actor_out.baseline[:, :, i]
            v_trace = compute_v_trace(
                log_rhos=log_rhos,
                discounts=discounts[i],
                rewards=prefix_rewards,
                values=values,
                bootstrap_value=bootstrap_value[:, i],
                return_norm_type=return_norm_type,
                norm_stat=self.norm_stats[i], 
                lamb=self.flags.v_trace_lamb,
            )                
            self.norm_stats[i] = v_trace.norm_stat
            if self.ppo_enable:                
                log_is_de = train_actor_out.c_action_log_prob
                adv = v_trace.pg_advantages_nois.detach()
                log_is_de = log_is_de.detach()
                vs = v_trace.vs.detach()

            if not self.ppo_enable:
                adv = v_trace.pg_advantages.detach()
                pg_loss = -adv * new_actor_out.c_action_log_prob
            else:                
                log_is = new_actor_out.c_action_log_prob - log_is_de
                unclipped_is = torch.exp(log_is) 
                self.ppo_is_abs.append(torch.mean(torch.abs(unclipped_is-1)).detach().item())
                clipped_is = torch.clamp(unclipped_is, 1-self.flags.ppo_clip, 1+self.flags.ppo_clip)
                pg_loss = -torch.minimum(unclipped_is * adv, clipped_is * adv)

            if masks[i] is not None: pg_loss = pg_loss * masks[i]
            if time_weights is not None: pg_loss = pg_loss * time_weights
            pg_loss = torch.sum(pg_loss)

            vs = v_trace.vs if not self.ppo_enable else vs
            pg_losses.append(pg_loss)
            # combine original masks with time-based weights for baseline loss
            mask_i = masks[i]
            if time_weights is not None:
                mask_i = time_weights if mask_i is None else mask_i * time_weights
            if self.flags.critic_enc_type == 0:
                baseline_loss = compute_baseline_loss(
                    baseline=new_actor_out.baseline[:, :, i],
                    target_baseline=vs,
                    mask=mask_i
                )
            else:
                baseline_loss = compute_baseline_enc_loss(
                    baseline_enc=new_actor_out.baseline_enc[:, :, i],
                    target_baseline=vs,
                    rv_tran=self.actor_net.rv_tran,
                    enc_type=self.flags.critic_enc_type,
                    mask=mask_i
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
        for prefix in ["im", "cur"]:
            cost = getattr(self.flags, "%s_cost" % prefix)
            if cost > 0.:
                n += 1
                if getattr(self.flags, "%s_cost_anneal" % prefix):
                    cost *= self.anneal_c
                total_loss += cost * pg_losses[n] / self.actor_net.dim_actions
                total_loss += (cost * self.flags.baseline_cost * 
                            baseline_losses[n])
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

        reg_loss = torch.sum(new_actor_out.reg_loss)        
        losses["reg_loss"] = reg_loss
        total_loss += self.flags.reg_cost * reg_loss

        action_prior_weight = float(getattr(self.flags, "action_prior_weight", 0.0))
        if (
            action_prior_weight > 0.0
            and self.action_prior is not None
            and self.actor_net.discrete_action
        ):
            pri_logits = new_actor_out.pri_param
            if pri_logits.dim() == 4:
                pri_logits = pri_logits[:, :, 0, :]
            elif pri_logits.dim() == 2:
                pri_logits = pri_logits.unsqueeze(0)
            pri_probs = F.softmax(pri_logits, dim=-1)
            real_mask = last_step_real.float().unsqueeze(-1)
            denom = real_mask.sum()
            if denom > 0:
                p_batch = (pri_probs * real_mask).sum(dim=(0, 1)) / denom
                p_batch = p_batch / p_batch.sum()
                ema_beta = min(max(self.action_prior_ema_beta, 0.0), 1.0)
                if self.action_prior_ema is None:
                    p_smooth = p_batch
                else:
                    p_smooth = (
                        (1.0 - ema_beta) * self.action_prior_ema
                        + ema_beta * p_batch
                    )
                self.action_prior_ema = p_smooth.detach()
                eps = 1e-8
                p_smooth = p_smooth.clamp_min(eps)
                p_target = self.action_prior.clamp_min(eps)
                action_prior_loss = torch.sum(
                    p_smooth * (torch.log(p_smooth) - torch.log(p_target))
                )
                losses["action_prior_loss"] = action_prior_loss
                total_loss += action_prior_weight * action_prior_loss

        # log average step time (no direct penalty on total_loss)
        if step_time_loss is not None:
            losses["step_time_loss"] = step_time_loss

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
            pri_kl_loss = torch.sum(pri_kl_loss)
            kl_loss = pri_kl_loss

            if not self.disable_thinker:                
                tar_reset_log_prob = F.log_softmax(base_reset_logits, dim=-1)
                reset_log_prob = F.log_softmax(new_actor_out.reset_logits, dim=-1)
                reset_kl_loss = F.kl_div(reset_log_prob, tar_reset_log_prob, reduction="sum", log_target=True)
                kl_loss += reset_kl_loss

            if self.flags.ppo_kl_coef > 0.:
                total_loss += self.flags.ppo_kl_coef * self.actor_net.kl_beta * kl_loss         
                avg_kl_loss = kl_loss / T / B  
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
        last_step_real = (train_actor_out.step_status == 0) | (train_actor_out.step_status == 3)
        next_step_real = (train_actor_out.step_status == 2) | (train_actor_out.step_status == 3)
        
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

        for prefix in ["im", "cur"]:            
            if prefix == "im":
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

        if not self.disable_thinker:
            mask_stats = last_step_real & ~next_step_real
            max_rollout_depth = (
                train_actor_out.max_rollout_depth[mask_stats]
                .detach()
                .cpu()
                .numpy()
            )
            max_rollout_depth = (
                np.average(max_rollout_depth) if len(max_rollout_depth) > 0 else 0.0
            )
            stats["max_rollout_depth"] = max_rollout_depth

            # fraction of primary actions equal to index 0 (noop frequency)
            last_pri = getattr(train_actor_out, "last_pri", None)
            if last_pri is not None:
                last_pri_sel = last_pri[mask_stats]
                if last_pri_sel.numel() > 0:
                    # handle possible extra action dimension
                    if last_pri_sel.dim() > 1:
                        last_pri_sel = last_pri_sel[..., 0]
                    last_pri_flat = last_pri_sel.view(-1)
                    noop_frequency = (last_pri_flat == 0).float().mean().item()
                else:
                    noop_frequency = 0.0
                stats["noop_frequency"] = noop_frequency

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
        })

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

    def load_checkpoint(self, ckp_path: str):
        train_checkpoint = torch.load(ckp_path, torch.device("cpu"), weights_only = False)
        self.step = train_checkpoint["step"]
        self.real_step = train_checkpoint["real_step"]
        self.tot_eps = train_checkpoint["tot_eps"]
        self.ret_buffers = train_checkpoint["ret_buffers"]
        self.norm_stats = train_checkpoint["norm_stats"]
        self.crnorm = train_checkpoint["crnorm"]
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

    def close(self):
        if hasattr(self, 'actor_buffer') and self.actor_buffer is not None:
            self.actor_buffer.set_finish.remote()
        if getattr(self, 'bc_policy_adapter', None) is not None:
            self.bc_policy_adapter.close()
        if getattr(self, "bc_planner", None) is not None and hasattr(
            self.bc_planner, "close"
        ):
            try:
                self.bc_planner.close()
            except Exception:
                pass
        self.plogger.close()


@ray.remote
class ActorLearner(SActorLearner):
    pass
