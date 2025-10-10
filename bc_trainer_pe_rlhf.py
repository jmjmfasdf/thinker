"""PE-RLHF offline trainer for Thinker networks."""
from __future__ import annotations

import dataclasses
import itertools
import math
import os
import shutil
from pathlib import Path
from typing import Dict, NamedTuple, Optional, Tuple
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from thinker.bc_loader import FrameStackedBehavioralDataLoader
from thinker.actor_net import ActorNet
from thinker.model_net import ModelNet
from thinker import util
from python_tree import TreeManager


class PolicyBatch(NamedTuple):
    """Container for policy outputs used during training."""

    logits: torch.Tensor
    log_probs: torch.Tensor
    probs: torch.Tensor
    features: torch.Tensor


@dataclasses.dataclass
class PERLHFConfig:
    """Hyper-parameters for the PE-RLHF offline trainer."""

    gamma: float = 0.99
    alpha: float = 0.1
    tau: float = 0.005
    cost_coef: float = 1.0
    bc_coef: float = 1.0
    cql_alpha: float = 0.0
    default_takeover_cost: float = 1.0
    target_entropy: Optional[float] = None
    updates_per_epoch: int = 128
    max_grad_norm: Optional[float] = 10.0


class ThinkerPolicyAdapter:
    """Wraps ``ActorNet`` to expose logits and latent features for training."""

    def __init__(self, actor_net: ActorNet, model_net: ModelNet, flags, device: torch.device):
        self.actor_net = actor_net
        self.model_net = model_net
        self.flags = flags
        self.device = device
        self.num_actions = getattr(actor_net, "num_actions", None)
        if self.num_actions is None:
            raise ValueError("Actor network must define 'num_actions'.")
        self._latent_buffer: Optional[torch.Tensor] = None
        if not hasattr(self.actor_net, "policy"):
            raise AttributeError("Actor network must expose a 'policy' layer for PE-RLHF training")
        self._hook = self.actor_net.policy.register_forward_pre_hook(self._capture_latent)

    def close(self) -> None:
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def _capture_latent(self, _module, inputs):
        if not inputs:
            self._latent_buffer = None
        else:
            self._latent_buffer = inputs[0]

    def _clone_state(self, state):
        if isinstance(state, dict):
            return {k: self._clone_state(v) for k, v in state.items()}
        if isinstance(state, (list, tuple)):
            return type(state)(self._clone_state(v) for v in state)
        if torch.is_tensor(state):
            return state.detach().clone()
        return state

    def _prepare_model_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if getattr(self.model_net, "state_dtype_n", 0) == 0:
            return torch.clamp(obs * 255.0, 0, 255).to(torch.uint8)
        return obs

    def _extract_action(self, actor_out) -> torch.Tensor:
        if getattr(actor_out, "action_prob", None) is not None and actor_out.action_prob is not None:
            probs = actor_out.action_prob
            while probs.dim() > 2:
                probs = probs[0]
            return torch.argmax(probs, dim=-1)
        logits = getattr(actor_out, "pri_param", None)
        if logits is None:
            raise ValueError("Actor output does not contain logits or probabilities")
        while logits.dim() > 2:
            logits = logits[0]
        return torch.argmax(logits, dim=-1)


    def _squeeze_policy_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 4:
            return tensor[0, :, 0, :]
        if tensor.dim() == 3:
            return tensor[0, :, :]
        if tensor.dim() == 2:
            return tensor
        raise ValueError(f"Unsupported policy tensor shape: {tuple(tensor.shape)}")

    def _make_env_out(
        self,
        visual_input: torch.Tensor,
        tree_reps: torch.Tensor,
        last_action: torch.Tensor,
        last_reset: torch.Tensor,
        step_status: int | torch.Tensor,
        current_xs: Optional[torch.Tensor] = None,
        current_hs: Optional[torch.Tensor] = None,
    ):
        batch_size = visual_input.shape[0]
        env_out = SimpleNamespace()
        env_out.real_states = visual_input.unsqueeze(0)
        env_out.tree_reps = tree_reps.unsqueeze(0)
        if current_xs is not None:
            env_out.xs = current_xs.unsqueeze(0)
        if current_hs is not None:
            env_out.hs = current_hs.unsqueeze(0)
        if isinstance(step_status, torch.Tensor):
            env_out.step_status = step_status.view(1, -1)
        else:
            env_out.step_status = torch.full((1, batch_size), int(step_status), dtype=torch.long, device=self.device)
        env_out.done = torch.zeros(1, batch_size, dtype=torch.bool, device=self.device)
        env_out.real_done = torch.zeros(1, batch_size, dtype=torch.bool, device=self.device)
        env_out.truncated_done = torch.zeros(1, batch_size, dtype=torch.long, device=self.device)
        env_out.last_pri = last_action.long().unsqueeze(0)
        env_out.last_reset = last_reset.long().unsqueeze(0)
        reward_dim = getattr(self.actor_net, "num_rewards", 1)
        env_out.reward = torch.zeros(1, batch_size, reward_dim, device=self.device)
        return env_out

    def _rollout(
        self,
        obs: torch.Tensor,
        initial_action: Optional[torch.Tensor],
        requires_grad: bool,
    ):
        device = self.device
        obs_float = obs.to(device).float()
        model_input = self._prepare_model_obs(obs_float)
        batch_size = obs_float.shape[0]
        init_state = self.model_net.initial_state(batch_size=batch_size, device=device)
        dummy_actions = torch.zeros(1, batch_size, 1, dtype=torch.long, device=device)
        dummy_done = torch.zeros(batch_size, dtype=torch.bool, device=device)
        with torch.enable_grad():
            initial_model_out = self.model_net.forward(
                env_state=model_input,
                done=dummy_done,
                actions=dummy_actions,
                state=init_state,
                training=False,
            )
        initial_policy = getattr(initial_model_out, "policy", None)
        if initial_policy is not None:
            init_policy = initial_policy[0]
            if init_policy.dim() == 3:
                init_policy = init_policy.squeeze(1)
            init_policy = init_policy.detach()
        else:
            init_policy = torch.full((batch_size, self.num_actions), 1.0 / self.num_actions, device=device)
        initial_values = getattr(initial_model_out, "vs", None)
        if initial_values is not None:
            init_values = initial_values[0]
            if init_values.dim() == 2:
                init_values = init_values.squeeze(-1)
            init_values = init_values.detach()
        else:
            init_values = torch.zeros(batch_size, device=device)
        initial_rewards = getattr(initial_model_out, "rs", None)
        if initial_rewards is not None:
            init_rewards = initial_rewards[0]
            if init_rewards.dim() == 2:
                init_rewards = init_rewards.squeeze(-1)
            init_rewards = init_rewards.detach()
        else:
            init_rewards = None
        initial_xs = initial_model_out.xs[0].detach() if getattr(initial_model_out, "xs", None) is not None else None
        initial_hs = initial_model_out.hs[0].detach() if getattr(initial_model_out, "hs", None) is not None else None
        root_payload = []
        for idx in range(batch_size):
            encoded = {"real_states": obs_float[idx]}
            if initial_xs is not None:
                encoded["xs"] = initial_xs[idx]
            if initial_hs is not None:
                encoded["hs"] = initial_hs[idx]
            root_payload.append(encoded)
        tree_manager = TreeManager(batch_size=batch_size, num_actions=self.num_actions, flags=self.flags, device=device)
        tree_manager.expand_root(init_rewards, init_values, init_policy, root_payload)
        tree_reps = tree_manager.compute_tree_reps()
        current_xs = initial_xs.clone() if initial_xs is not None else None
        current_hs = initial_hs.clone() if initial_hs is not None else None
        model_state = self._clone_state(initial_model_out.state)
        actor_core_state = self.actor_net.initial_state(batch_size=batch_size, device=device)
        last_reset = torch.zeros(batch_size, dtype=torch.long, device=device)
        if initial_action is not None:
            initial_action = initial_action.detach()
            last_action = initial_action.to(device=device, dtype=torch.long).view(batch_size)
        else:
            self._latent_buffer = None
            env_out_root = self._make_env_out(
                current_xs if current_xs is not None else obs_float,
                tree_reps,
                torch.zeros(batch_size, dtype=torch.long, device=device),
                last_reset,
                0,
                current_xs=current_xs,
                current_hs=current_hs,
            )
            with torch.no_grad():
                actor_out, actor_core_state = self.actor_net(env_out_root, core_state=actor_core_state)
            last_action = self._extract_action(actor_out)
            self._latent_buffer = None
        rollout_steps = max(0, int(getattr(self.flags, "rec_t", 1)) - 1)
        for step in range(rollout_steps):
            tree_manager.cur_t += 1
            tree_manager.rollout_depth += 1
            with torch.enable_grad():
                model_out = self.model_net.forward_single(state=model_state, action=last_action, training=False)
            xs = model_out.xs[0].detach() if getattr(model_out, "xs", None) is not None else None
            hs = model_out.hs[0].detach() if getattr(model_out, "hs", None) is not None else None
            if xs is not None:
                current_xs = xs.clone()
            if hs is not None:
                current_hs = hs.clone()
            model_state = self._clone_state(model_out.state)
            logits = model_out.policy[0] if getattr(model_out, "policy", None) is not None else torch.zeros(batch_size, self.num_actions, device=device)
            if logits.dim() == 3:
                logits = logits.squeeze(1)
            logits = logits.detach()
            rewards = model_out.rs[0] if getattr(model_out, "rs", None) is not None else torch.zeros(batch_size, device=device)
            if rewards.dim() == 2:
                rewards = rewards.squeeze(-1)
            rewards = rewards.detach() if torch.is_tensor(rewards) else rewards
            dones = model_out.dones[0] if getattr(model_out, "dones", None) is not None else torch.zeros(batch_size, dtype=torch.bool, device=device)
            dones = dones.detach() if torch.is_tensor(dones) else dones
            values = model_out.vs[0] if getattr(model_out, "vs", None) is not None else torch.zeros(batch_size, device=device)
            if values.dim() == 2:
                values = values.squeeze(-1)
            values = values.detach() if torch.is_tensor(values) else values
            payload = []
            for idx in range(batch_size):
                encoded = {}
                if xs is not None:
                    encoded["xs"] = xs[idx]
                if hs is not None:
                    encoded["hs"] = hs[idx]
                payload.append(encoded)
            tree_manager.expand_current(rewards, values, dones, logits, payload)
            tree_reps = tree_manager.compute_tree_reps()
            step_status = 2 if step == rollout_steps - 1 else 1
            env_out_im = self._make_env_out(
                current_xs if current_xs is not None else obs_float,
                tree_reps,
                last_action,
                last_reset,
                step_status,
                current_xs=current_xs,
                current_hs=current_hs,
            )
            self._latent_buffer = None
            with torch.no_grad():
                actor_out, actor_core_state = self.actor_net(env_out_im, core_state=actor_core_state)
            next_action = self._extract_action(actor_out).long()
            self._latent_buffer = None
            tree_manager.advance(next_action)
            last_action = next_action
            last_reset.zero_()
        tree_manager.reset_real_step()
        tree_reps = tree_manager.compute_tree_reps()
        final_reset = torch.zeros_like(last_reset)
        final_visual = current_xs if current_xs is not None else obs_float
        env_out_final = self._make_env_out(
            final_visual,
            tree_reps,
            last_action,
            final_reset,
            0,
            current_xs=current_xs,
            current_hs=current_hs,
        )
        self._latent_buffer = None
        if requires_grad:
            actor_out, _ = self.actor_net(env_out_final, core_state=actor_core_state, compute_loss=True)
        else:
            with torch.no_grad():
                actor_out, _ = self.actor_net(env_out_final, core_state=actor_core_state, compute_loss=True)
        return actor_out

    def forward(
        self,
        obs: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        requires_grad: bool = True,
    ) -> PolicyBatch:
        actor_out = self._rollout(obs, actions, requires_grad)
        logits_tensor = getattr(actor_out, "pri_param", None)
        if logits_tensor is not None:
            logits = self._squeeze_policy_tensor(logits_tensor)
            log_probs = torch.log_softmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)
        else:
            probs_tensor = getattr(actor_out, "action_prob", None)
            if probs_tensor is None:
                raise ValueError("Actor output does not contain policy logits or probabilities.")
            probs = self._squeeze_policy_tensor(probs_tensor)
            log_probs = torch.log(probs + 1e-8)
            logits = log_probs
        if self._latent_buffer is None:
            raise RuntimeError("Actor latent representation hook did not trigger.")
        latent = torch.relu(self._latent_buffer)
        latent = latent.view(latent.shape[0], -1)
        if not requires_grad:
            latent = latent.detach()
            logits = logits.detach()
            log_probs = log_probs.detach()
            probs = probs.detach()
        return PolicyBatch(logits=logits, log_probs=log_probs, probs=probs, features=latent)
class DiscreteQNetwork(nn.Module):
    """Simple MLP that outputs Q-values for all discrete actions."""

    def __init__(self, input_dim: int, num_actions: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class PERLHFOfflineTrainer:
    """Trains Thinker's PyTorch networks with a PE-RLHF style objective."""

    def __init__(
        self,
        actor_net: ActorNet,
        model_net: ModelNet,
        data_loader: FrameStackedBehavioralDataLoader,
        config: PERLHFConfig,
        flags,
        logger=None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.actor_net = actor_net
        self.model_net = model_net
        self.data_loader = data_loader
        self.config = config
        self.flags = flags
        self.logger = logger
        self.device = device or next(actor_net.parameters()).device
        self.mode = getattr(flags, "mode", 1)

        self.policy_adapter = ThinkerPolicyAdapter(self.actor_net, self.model_net, flags, self.device)

        warm_batch = self._sample_batch(batch_size=max(1, getattr(flags, "bc_batch_size", 32)))
        if warm_batch is None:
            raise RuntimeError("Unable to load behavioral cloning data for warm-up.")
        obs = torch.from_numpy(warm_batch["obs"]).float().to(self.device)[:1]
        warm_actions = warm_batch.get("actions")
        if warm_actions is not None:
            warm_actions = np.asarray(warm_actions)
            if warm_actions.ndim > 1 and warm_actions.shape[-1] > 1:
                warm_actions_tensor = torch.from_numpy(np.argmax(warm_actions, axis=-1)).long().to(self.device)
            else:
                warm_actions_tensor = torch.from_numpy(warm_actions.reshape(-1)).long().to(self.device)
            warm_actions_tensor = warm_actions_tensor[: obs.shape[0]]
        else:
            warm_actions_tensor = None
        warm_policy = self.policy_adapter.forward(obs, actions=warm_actions_tensor, requires_grad=False)
        self.feature_dim = warm_policy.features.shape[-1]
        self.num_actions = warm_policy.logits.shape[-1]

        self.q_net1 = DiscreteQNetwork(self.feature_dim, self.num_actions).to(self.device)
        self.q_net2 = DiscreteQNetwork(self.feature_dim, self.num_actions).to(self.device)
        self.cost_q_net1 = DiscreteQNetwork(self.feature_dim, self.num_actions).to(self.device)
        self.cost_q_net2 = DiscreteQNetwork(self.feature_dim, self.num_actions).to(self.device)
        self.q_target1 = DiscreteQNetwork(self.feature_dim, self.num_actions).to(self.device)
        self.q_target2 = DiscreteQNetwork(self.feature_dim, self.num_actions).to(self.device)
        self.cost_target1 = DiscreteQNetwork(self.feature_dim, self.num_actions).to(self.device)
        self.cost_target2 = DiscreteQNetwork(self.feature_dim, self.num_actions).to(self.device)
        self.q_target1.load_state_dict(self.q_net1.state_dict())
        self.q_target2.load_state_dict(self.q_net2.state_dict())
        self.cost_target1.load_state_dict(self.cost_q_net1.state_dict())
        self.cost_target2.load_state_dict(self.cost_q_net2.state_dict())

        initial_alpha = max(self.config.alpha, 1e-4)
        self.log_alpha = torch.tensor(math.log(initial_alpha), device=self.device, requires_grad=True)
        target_entropy = self.config.target_entropy
        if target_entropy is None:
            target_entropy = -float(self.num_actions)
        self.target_entropy = torch.tensor(target_entropy, device=self.device)

        lr_actor = getattr(flags, "bc_lr", 1e-4)
        lr_model = getattr(flags, "bc_model_lr", lr_actor)
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(
            itertools.chain(self.q_net1.parameters(), self.q_net2.parameters()), lr=lr_actor
        )
        self.cost_optimizer = torch.optim.Adam(
            itertools.chain(self.cost_q_net1.parameters(), self.cost_q_net2.parameters()), lr=lr_actor
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr_actor)
        self.model_optimizer = (
            torch.optim.Adam(self.model_net.parameters(), lr=lr_model)
            if self.mode in (2, 3)
            else None
        )

        self.actor_net.train()
        self.model_net.train()
        self._best_metric = float("inf")
        self.global_update_step = 0

        base_savedir = Path(getattr(self.flags, "savedir", "./logs"))
        proj_name = getattr(self.flags, "proj_name", "pe_rlhf")
        self.output_dir = base_savedir / proj_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        existing_ckpdir = Path(getattr(self.flags, "ckpdir", str(self.output_dir)))
        if existing_ckpdir.exists() and existing_ckpdir.resolve() != self.output_dir.resolve():
            for item in existing_ckpdir.iterdir():
                dest = self.output_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                else:
                    if not dest.exists():
                        shutil.copy2(item, dest)
        config_source = existing_ckpdir / "config_c.yaml"
        if config_source.exists():
            shutil.copy2(config_source, self.output_dir / "config_c.yaml")

        os.environ["THINKER_LOG_DIR"] = str(self.output_dir)
        self.flags.ckpdir = str(self.output_dir)

        def _filter_value(value):
            if isinstance(value, (int, float, str, bool)) or value is None:
                return value
            if isinstance(value, (list, tuple)):
                if all(isinstance(item, (int, float, str, bool)) or item is None for item in value):
                    return list(value)
            return None

        self.serialized_flags = {}
        for key, value in vars(self.flags).items():
            filtered = _filter_value(value)
            if filtered is not None:
                self.serialized_flags[key] = filtered

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def train(self) -> Tuple[float, list]:
        history: list = []
        epochs = getattr(self.flags, "bc_epochs", 1)
        for epoch in range(epochs):
            metrics = self._run_epoch(epoch)
            history.append(metrics)

            reference_metric = metrics.get("actor_loss", metrics.get("model_loss", 0.0))
            if reference_metric < self._best_metric:
                self._best_metric = reference_metric
                self._save_checkpoint(epoch, metrics, best=True)

            save_interval = max(1, getattr(self.flags, "bc_save_interval", 1))
            if (epoch + 1) % save_interval == 0:
                self._save_checkpoint(epoch, metrics, best=False)

            if self.logger:
                parts = [
                    f"epoch={epoch + 1}",
                    f"actor={metrics.get('actor_loss', 0.0):.4f}",
                    f"critic={metrics.get('critic_loss', 0.0):.4f}",
                    f"cost={metrics.get('cost_loss', 0.0):.4f}",
                    f"model={metrics.get('model_loss', 0.0):.4f}",
                ]
                self.logger.info("[PE-RLHF] " + " | ".join(parts))

            eval_interval = getattr(self.flags, "bc_eval_interval", 0)
            if eval_interval and (epoch + 1) % eval_interval == 0:
                eval_stats = self.evaluate(
                    num_batches=getattr(self.flags, "bc_eval_batches", 1),
                    batch_size=getattr(self.flags, "bc_eval_batch_size", getattr(self.flags, "bc_batch_size", 32)),
                )
                metrics.update({f"eval_{k}": v for k, v in eval_stats.items()})
                if self.logger:
                    self.logger.info(
                        "[PE-RLHF] eval "
                        + ", ".join(f"{k}={v:.4f}" for k, v in eval_stats.items())
                    )

        self.policy_adapter.close()
        return self._best_metric, history

    def _log_training_step(self, epoch: int, update_idx: int, metrics: Dict[str, float]) -> None:
        if not metrics:
            return
        msg_parts = [f"epoch={epoch + 1}", f"update={update_idx + 1}", f"step={self.global_update_step}"]
        log_keys = ["actor_loss", "actor_core_loss", "bc_loss", "alpha_loss", "critic_loss", "cost_loss", "model_loss"]
        for key in log_keys:
            value = metrics.get(key)
            if isinstance(value, torch.Tensor):
                value = float(value.detach().cpu().item())
            if value is None:
                msg_parts.append(f"{key}=NA")
            else:
                msg_parts.append(f"{key}={value:.4f}")
        message = " | ".join(msg_parts)
        if self.logger is not None:
            self.logger.info("[PE-RLHF] " + message)
        else:
            print("[PE-RLHF] " + message)

    def _run_epoch(self, epoch: int) -> Dict[str, float]:
        updates = max(1, self.config.updates_per_epoch)
        actor_loss_sum = 0.0
        actor_core_loss_sum = 0.0
        bc_loss_sum = 0.0
        alpha_loss_sum = 0.0
        critic_loss_sum = 0.0
        cost_loss_sum = 0.0
        model_loss_sum = 0.0
        actor_steps = 0

        for update_idx in range(updates):
            batch = self._sample_batch(batch_size=getattr(self.flags, "bc_batch_size", 32))
            if batch is None:
                continue
            obs, next_obs, actions, rewards, takeover = self._prepare_batch(batch)

            step_metrics = {
                "actor_loss": None,
                "actor_core_loss": None,
                "bc_loss": None,
                "alpha_loss": None,
                "critic_loss": None,
                "cost_loss": None,
                "model_loss": None,
            }

            if self.mode in (1, 3):
                critic_loss, cost_loss = self._update_critics(obs, next_obs, actions, rewards, takeover)
                total_actor_loss, actor_core_loss, bc_loss, alpha_loss = self._update_actor_and_alpha(obs, actions)
                actor_loss_sum += total_actor_loss
                actor_core_loss_sum += actor_core_loss
                bc_loss_sum += bc_loss
                alpha_loss_sum += alpha_loss
                critic_loss_sum += critic_loss
                cost_loss_sum += cost_loss
                actor_steps += 1
                step_metrics.update(
                    {
                        "actor_loss": float(total_actor_loss),
                        "actor_core_loss": float(actor_core_loss),
                        "bc_loss": float(bc_loss),
                        "alpha_loss": float(alpha_loss),
                        "critic_loss": float(critic_loss),
                        "cost_loss": float(cost_loss),
                    }
                )

            if self.mode in (2, 3):
                model_loss = self._update_model(obs, next_obs, actions, rewards)
                model_loss_sum += model_loss
                step_metrics["model_loss"] = float(model_loss)

            if step_metrics:
                self.global_update_step += 1
                self._log_training_step(epoch, update_idx, step_metrics)

        divisor = max(1, actor_steps)
        metrics = {
            "actor_loss": actor_loss_sum / divisor,
            "actor_core_loss": actor_core_loss_sum / divisor,
            "bc_loss": bc_loss_sum / divisor,
            "alpha_loss": alpha_loss_sum / divisor,
            "critic_loss": critic_loss_sum / divisor,
            "cost_loss": cost_loss_sum / divisor,
            "model_loss": model_loss_sum / max(1, self.config.updates_per_epoch),
        }
        return metrics

    def _sample_batch(self, batch_size: int) -> Optional[Dict[str, np.ndarray]]:
        for _ in range(4):
            batch = self.data_loader.get_paired_batch(batch_size=batch_size)
            if batch is not None:
                return batch
            self.data_loader.reset()
        return None

    def _prepare_batch(
        self, batch: Dict[str, np.ndarray]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = torch.from_numpy(batch["obs"]).float().to(self.device)
        next_obs = torch.from_numpy(batch["next_obs"]).float().to(self.device)

        actions_np = batch["actions"]
        if actions_np.ndim > 1 and actions_np.shape[-1] > 1:
            action_indices = np.argmax(actions_np, axis=-1)
        else:
            action_indices = actions_np.reshape(-1)
        actions = torch.from_numpy(action_indices.astype(np.int64)).long().to(self.device)

        rewards = torch.from_numpy(batch["rewards"].astype(np.float32)).to(self.device)
        takeover_np = batch.get("takeover")
        if takeover_np is None:
            takeover = torch.full_like(rewards, float(self.config.default_takeover_cost))
        else:
            takeover = torch.from_numpy(takeover_np.astype(np.float32)).to(self.device)
        return obs, next_obs, actions, rewards, takeover

    def _update_critics(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        takeover: torch.Tensor,
    ) -> Tuple[float, float]:
        with torch.no_grad():
            current_policy = self.policy_adapter.forward(obs, actions=actions, requires_grad=False)
            next_policy = self.policy_adapter.forward(next_obs, requires_grad=False)

        q1_all = self.q_net1(current_policy.features)
        q2_all = self.q_net2(current_policy.features)
        action_idx = actions.unsqueeze(-1)
        q1_pred = q1_all.gather(1, action_idx).squeeze(1)
        q2_pred = q2_all.gather(1, action_idx).squeeze(1)

        q1_next = self.q_target1(next_policy.features)
        q2_next = self.q_target2(next_policy.features)
        q_next_min = torch.min(q1_next, q2_next)
        alpha = self.alpha.detach()
        v_next = torch.sum(next_policy.probs * (q_next_min - alpha * next_policy.log_probs), dim=-1)
        target_q = rewards + self.config.gamma * v_next

        critic_loss = F.mse_loss(q1_pred, target_q) + F.mse_loss(q2_pred, target_q)

        if self.config.cql_alpha > 0.0:
            logsum_q1 = torch.logsumexp(q1_all, dim=-1)
            logsum_q2 = torch.logsumexp(q2_all, dim=-1)
            cql_term = (logsum_q1 + logsum_q2).mean() - (q1_pred + q2_pred).mean()
            critic_loss = critic_loss + self.config.cql_alpha * cql_term

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                itertools.chain(self.q_net1.parameters(), self.q_net2.parameters()),
                self.config.max_grad_norm,
            )
        self.critic_optimizer.step()

        cost_q1_all = self.cost_q_net1(current_policy.features.detach())
        cost_q2_all = self.cost_q_net2(current_policy.features.detach())
        cost_q1_pred = cost_q1_all.gather(1, action_idx).squeeze(1)
        cost_q2_pred = cost_q2_all.gather(1, action_idx).squeeze(1)

        cost_q1_next = self.cost_target1(next_policy.features)
        cost_q2_next = self.cost_target2(next_policy.features)
        cost_next_min = torch.min(cost_q1_next, cost_q2_next)
        cost_target = takeover + self.config.gamma * torch.sum(next_policy.probs * cost_next_min, dim=-1)

        cost_loss = F.mse_loss(cost_q1_pred, cost_target) + F.mse_loss(cost_q2_pred, cost_target)

        self.cost_optimizer.zero_grad()
        cost_loss.backward()
        if self.config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                itertools.chain(self.cost_q_net1.parameters(), self.cost_q_net2.parameters()),
                self.config.max_grad_norm,
            )
        self.cost_optimizer.step()

        self._soft_update(self.q_target1, self.q_net1)
        self._soft_update(self.q_target2, self.q_net2)
        self._soft_update(self.cost_target1, self.cost_q_net1)
        self._soft_update(self.cost_target2, self.cost_q_net2)

        return critic_loss.item(), cost_loss.item()

    def _update_actor_and_alpha(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[float, float, float, float]:
        policy = self.policy_adapter.forward(obs, actions=actions, requires_grad=True)
        q1_all = self.q_net1(policy.features)
        q2_all = self.q_net2(policy.features)
        cost_q1_all = self.cost_q_net1(policy.features)
        cost_q2_all = self.cost_q_net2(policy.features)

        q_min = torch.min(q1_all, q2_all)
        cost_min = torch.min(cost_q1_all, cost_q2_all)
        alpha = self.alpha
        inside = alpha * policy.log_probs - q_min + self.config.cost_coef * cost_min
        actor_core_loss = torch.mean(torch.sum(policy.probs * inside, dim=-1))

        if self.config.bc_coef > 0.0:
            bc_loss = F.nll_loss(policy.log_probs, actions)
        else:
            bc_loss = torch.zeros(1, device=self.device)

        total_actor_loss = actor_core_loss + self.config.bc_coef * bc_loss

        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        if self.config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.config.max_grad_norm)
        self.actor_optimizer.step()

        entropy = -torch.sum(policy.probs.detach() * policy.log_probs.detach(), dim=-1)
        alpha_loss = -(self.log_alpha * (entropy - self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        return (
            total_actor_loss.item(),
            actor_core_loss.item(),
            bc_loss.item(),
            alpha_loss.item(),
        )

    def _update_model(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
    ) -> float:
        if self.model_optimizer is None:
            return 0.0

        if getattr(self.model_net, "state_dtype_n", 0) == 0:
            obs_input = (obs * 255.0).clamp(0, 255).to(torch.uint8)
            next_input = (next_obs * 255.0).clamp(0, 255).to(torch.uint8)
        else:
            obs_input = obs
            next_input = next_obs

        batch_size = obs.shape[0]
        done = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        prev_actions = torch.zeros(batch_size, 1, dtype=torch.long, device=self.device)
        current_actions = actions.view(batch_size, 1)
        action_seq = torch.stack([prev_actions, current_actions], dim=0)
        state = self.model_net.initial_state(batch_size=batch_size, device=self.device)

        model_out = self.model_net.forward(
            env_state=obs_input,
            done=done,
            actions=action_seq,
            state=state,
            future_env_state=next_input,
            training=True,
        )

        losses: list = []
        if getattr(model_out, "rs", None) is not None:
            predicted_rewards = model_out.rs[0]
            if predicted_rewards.ndim == 2:
                predicted_rewards = predicted_rewards[:, 0]
            losses.append(F.mse_loss(predicted_rewards.float(), rewards.float()))
        if getattr(model_out, "policy", None) is not None:
            policy_logits = model_out.policy[0].float()
            policy_logits = policy_logits.view(policy_logits.shape[0], -1)
            losses.append(F.cross_entropy(policy_logits, actions.long()))
        if getattr(model_out, "xs", None) is not None:
            predicted_next = model_out.xs[0]
            target_next = self.model_net.normalize(next_input)
            predicted_next = predicted_next.float()
            losses.append(F.mse_loss(predicted_next, target_next))

        if not losses:
            return 0.0

        model_loss = sum(losses)
        self.model_optimizer.zero_grad()
        model_loss.backward()
        if self.config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model_net.parameters(), self.config.max_grad_norm)
        self.model_optimizer.step()
        return float(model_loss.item())

    def evaluate(self, num_batches: int, batch_size: int) -> Dict[str, float]:
        total = 0
        correct = 0
        nll = 0.0
        self.actor_net.eval()
        for _ in range(num_batches):
            batch = self._sample_batch(batch_size=batch_size)
            if batch is None:
                continue
            obs, _, actions, _, _ = self._prepare_batch(batch)
            with torch.no_grad():
                policy = self.policy_adapter.forward(obs, actions=actions, requires_grad=False)
            preds = policy.probs.argmax(dim=-1)
            correct += (preds == actions).sum().item()
            nll += F.nll_loss(policy.log_probs, actions, reduction="sum").item()
            total += actions.numel()
        self.actor_net.train()
        if total == 0:
            return {"accuracy": 0.0, "nll": float("nan")}
        return {"accuracy": correct / total, "nll": nll / total}

    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        tau = self.config.tau
        for tgt, src in zip(target.parameters(), source.parameters()):
            tgt.data.copy_(tgt.data * (1.0 - tau) + src.data * tau)

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float], best: bool) -> None:
        step = max(1, self.global_update_step)
        base_dir = getattr(self, "output_dir", None)
        if base_dir is None:
            base_dir = Path(getattr(self.flags, "savedir", "./logs")) / getattr(self.flags, "proj_name", "pe_rlhf")
        else:
            base_dir = Path(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        def _atomic_save(payload, file_path: Path) -> None:
            tmp_path = file_path.with_name(file_path.name + '.tmp')
            torch.save(payload, tmp_path)
            os.replace(tmp_path, file_path)

        actor_payload = {
            "epoch": epoch + 1,
            "step": step,
            "metrics": metrics,
            "actor_net_state_dict": self.actor_net.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "flags": dict(self.serialized_flags),
            "config": dataclasses.asdict(self.config),
        }
        critic_payload = {
            "q1": self.q_net1.state_dict(),
            "q2": self.q_net2.state_dict(),
            "cost_q1": self.cost_q_net1.state_dict(),
            "cost_q2": self.cost_q_net2.state_dict(),
            "target_q1": self.q_target1.state_dict(),
            "target_q2": self.q_target2.state_dict(),
            "target_cost_q1": self.cost_target1.state_dict(),
            "target_cost_q2": self.cost_target2.state_dict(),
            "alpha": self.alpha.detach().cpu().item(),
        }
        model_payload = {
            "epoch": epoch + 1,
            "step": step,
            "metrics": metrics,
            "model_net_state_dict": self.model_net.state_dict(),
            "critic_state": critic_payload,
            "flags": dict(self.serialized_flags),
            "config": dataclasses.asdict(self.config),
        }
        if self.model_optimizer is None:
            self.model_optimizer = torch.optim.Adam(self.model_net.parameters(), lr=getattr(self.flags, "bc_model_lr", getattr(self.flags, "bc_lr", 1e-4)))
        model_payload["model_optimizer_state_dict"] = self.model_optimizer.state_dict()

        actor_path = base_dir / "ckp_actor.tar"
        model_path = base_dir / "ckp_model.tar"
        _atomic_save(actor_payload, actor_path)
        _atomic_save(model_payload, model_path)

        step_actor_path = base_dir / f"ckp_actor.tar_step_{step}"
        step_model_path = base_dir / f"ckp_model.tar_step_{step}"
        _atomic_save(actor_payload, step_actor_path)
        _atomic_save(model_payload, step_model_path)

        if best:
            _atomic_save(actor_payload, base_dir / "ckp_actor_best.tar")
            _atomic_save(model_payload, base_dir / "ckp_model_best.tar")

        if self.logger:
            status = "best" if best else "epoch"
            self.logger.info(f"Saved {status} checkpoint to {base_dir}")


def create_bc_data_loader(flags) -> FrameStackedBehavioralDataLoader:
    subjects = [int(s.strip()) for s in str(flags.bc_subjects).split(',')]
    loader = FrameStackedBehavioralDataLoader(
        base_path=flags.bc_data_path,
        subjects=subjects,
        game_id=flags.bc_game_id,
        frame_stack_n=flags.frame_stack_n,
        target_size=(84, 84),
        grayscale=flags.grayscale,
        normalize=True,
    )
    return loader


def run_pe_rlhf_training(
    flags,
    model_net: ModelNet,
    actor_net: ActorNet,
    logger=None,
) -> Tuple[float, list]:
    loader = create_bc_data_loader(flags)
    updates_per_epoch = getattr(flags, "bc_updates_per_epoch", len(loader.data_files))
    updates_per_epoch = max(1, updates_per_epoch)
    config = PERLHFConfig(
        gamma=getattr(flags, "bc_gamma", getattr(flags, "discounting", 0.99)),
        alpha=getattr(flags, "bc_alpha", 0.1),
        tau=0.005,
        cost_coef=getattr(flags, "bc_cost_coef", 1.0),
        bc_coef=getattr(flags, "bc_prior_coef", 1.0),
        cql_alpha=getattr(flags, "bc_cql_alpha", 0.0),
        default_takeover_cost=getattr(flags, "bc_default_takeover_cost", 1.0),
        target_entropy=getattr(flags, "bc_target_entropy", None),
        updates_per_epoch=updates_per_epoch,
        max_grad_norm=getattr(flags, "bc_max_grad_norm", 10.0),
    )
    trainer = PERLHFOfflineTrainer(
        actor_net=actor_net,
        model_net=model_net,
        data_loader=loader,
        config=config,
        flags=flags,
        logger=logger,
    )
    return trainer.train()
