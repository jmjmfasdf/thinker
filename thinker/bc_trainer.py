"""IcoPro-style behavioral cloning trainer for Thinker networks."""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from thinker.bc_loader import FrameStackedBehavioralDataLoader
from thinker.actor_net import ActorNet
from thinker.model_net import ModelNet
from python_tree import TreeManager


class PolicyBatch(NamedTuple):
    """Container for policy outputs used during training."""

    logits: torch.Tensor
    log_probs: torch.Tensor
    probs: torch.Tensor
    features: torch.Tensor


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
        self._last_tree_q: Optional[torch.Tensor] = None
        self._last_tree_reps: Optional[torch.Tensor] = None
        if not hasattr(self.actor_net, "policy"):
            raise AttributeError("Actor network must expose a 'policy' layer for BC training")
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
        self._last_tree_q = None
        self._last_tree_reps = None
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
            if initial_rewards.dim() == 2:
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
        q_values = torch.zeros(batch_size, self.num_actions, dtype=torch.float32, device=self.device)
        for idx, root in enumerate(tree_manager.root_nodes):
            for action_idx, child in enumerate(root.children):
                q_values[idx, action_idx] = float(child.rollout_q)
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
        self._last_tree_q = q_values
        self._last_tree_reps = tree_reps.detach()
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

    @property
    def last_tree_q(self) -> Optional[torch.Tensor]:
        return self._last_tree_q

    @property
    def last_tree_reps(self) -> Optional[torch.Tensor]:
        return self._last_tree_reps


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


class IcoProBehaviorCloningTrainer:
    """IcoPro-inspired behavioral cloning trainer for Thinker."""

    def __init__(
        self,
        actor_net: ActorNet,
        model_net: ModelNet,
        data_loader: FrameStackedBehavioralDataLoader,
        flags,
        logger=None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.actor_net = actor_net
        self.model_net = model_net
        self.data_loader = data_loader
        self.flags = flags
        self.logger = logger
        self.device = device or next(actor_net.parameters()).device
        self.mode = int(getattr(flags, "mode", 3))

        self.policy_adapter = ThinkerPolicyAdapter(self.actor_net, self.model_net, flags, self.device)

        self.batch_size = max(1, getattr(flags, "bc_batch_size", 32))
        self.eval_batch_size = max(1, getattr(flags, "bc_eval_batch_size", self.batch_size))
        self.margin_value = float(getattr(flags, "bc_margin", 0.05))
        self.margin_coef = float(getattr(flags, "bc_margin_coef", 1.0))
        self.ce_coef = float(getattr(flags, "bc_ce_coef", 1.0))
        self.tree_coef = float(getattr(flags, "bc_tree_coef", 0.0))
        self.max_grad_norm = getattr(flags, "bc_max_grad_norm", None)

        warm_batch = self._sample_batch(batch_size=self.batch_size)
        if warm_batch is None:
            raise RuntimeError("Unable to load behavioral cloning data for warm-up.")
        obs = warm_batch["obs"][:1]
        warm_actions = warm_batch["actions"][:1]
        warm_policy = self.policy_adapter.forward(obs, actions=warm_actions, requires_grad=False)
        self.num_actions = warm_policy.logits.shape[-1]

        lr_actor = getattr(flags, "bc_lr", 1e-4)
        lr_model = getattr(flags, "bc_model_lr", lr_actor)
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=lr_actor)
        self.model_optimizer = (
            torch.optim.Adam(self.model_net.parameters(), lr=lr_model)
            if self.mode in (2, 3)
            else None
        )

        self.actor_net.train()
        self.model_net.train()
        self._best_metric = float("inf")

        base_savedir = Path(getattr(self.flags, "savedir", "./logs"))
        proj_name = getattr(self.flags, "proj_name", "bc_icopro")
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

        self.updates_per_epoch = max(1, getattr(self.flags, "bc_updates_per_epoch", len(self.data_loader.data_files)))

    def _sample_batch(self, batch_size: int) -> Optional[Dict[str, torch.Tensor]]:
        batch = self.data_loader.get_paired_batch(batch_size=batch_size)
        if batch is None:
            return None
        obs = torch.from_numpy(batch["obs"]).float().to(self.device)
        next_obs = torch.from_numpy(batch["next_obs"]).float().to(self.device)
        actions_np = np.asarray(batch["actions"])
        if actions_np.ndim > 1 and actions_np.shape[-1] > 1:
            actions_np = actions_np.argmax(axis=-1)
        actions = torch.from_numpy(actions_np.astype(np.int64)).to(self.device)
        rewards = torch.from_numpy(np.asarray(batch["rewards"], dtype=np.float32)).to(self.device)
        return {
            "obs": obs,
            "next_obs": next_obs,
            "actions": actions,
            "rewards": rewards,
        }

    def _actor_step(self, batch: Dict[str, torch.Tensor], train: bool) -> Dict[str, float]:
        obs = batch["obs"]
        actions = batch["actions"]
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            policy = self.policy_adapter.forward(obs, actions=actions, requires_grad=train)
            tree_q = self.policy_adapter.last_tree_q
            q_policy = policy.logits
            if tree_q is not None:
                q_values = q_policy + self.tree_coef * tree_q
            else:
                q_values = q_policy
            margin = torch.full((obs.shape[0],), self.margin_value, dtype=torch.float32, device=self.device)
            margin_loss = dqfd_margin_loss(q_values, actions, margin)
            if self.ce_coef > 0:
                ce_loss = F.cross_entropy(q_policy, actions)
            else:
                ce_loss = torch.zeros(1, device=self.device)
            total_loss = self.margin_coef * margin_loss + self.ce_coef * ce_loss
            pred = torch.argmax(q_policy, dim=-1)
            accuracy = (pred == actions).float().mean()
        if train:
            self.actor_optimizer.zero_grad()
            total_loss.backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
        return {
            "loss": float(total_loss.detach().cpu().item()),
            "margin": float(margin_loss.detach().cpu().item()),
            "ce": float(ce_loss.detach().cpu().item()),
            "acc": float(accuracy.detach().cpu().item()),
        }

    def _model_step(self, batch: Dict[str, torch.Tensor], train: bool) -> float:
        if self.model_optimizer is None or not train:
            return 0.0
        obs = batch["obs"]
        next_obs = batch["next_obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]

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

        losses: List[torch.Tensor] = []
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
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model_net.parameters(), self.max_grad_norm)
        self.model_optimizer.step()
        return float(model_loss.detach().cpu().item())

    def _run_epoch(self, epoch: int) -> Dict[str, float]:
        actor_losses, margin_losses, ce_losses, accuracies, model_losses = [], [], [], [], []
        updates = 0
        for _ in range(self.updates_per_epoch):
            batch = self._sample_batch(self.batch_size)
            if batch is None:
                continue
            updates += 1
            actor_metrics = self._actor_step(batch, train=self.mode in (1, 3))
            actor_losses.append(actor_metrics["loss"])
            margin_losses.append(actor_metrics["margin"])
            ce_losses.append(actor_metrics["ce"])
            accuracies.append(actor_metrics["acc"])
            model_loss = self._model_step(batch, train=self.mode in (2, 3))
            model_losses.append(model_loss)
        metrics = {
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
            "margin_loss": float(np.mean(margin_losses)) if margin_losses else 0.0,
            "ce_loss": float(np.mean(ce_losses)) if ce_losses else 0.0,
            "accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
            "model_loss": float(np.mean(model_losses)) if model_losses else 0.0,
            "updates": updates,
        }
        return metrics

    def train(self) -> Tuple[float, List[Dict[str, float]]]:
        history: List[Dict[str, float]] = []
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
                self.logger.info(
                    "[BC] epoch=%d actor=%.4f margin=%.4f ce=%.4f acc=%.4f model=%.4f updates=%d",
                    epoch + 1,
                    metrics.get("actor_loss", 0.0),
                    metrics.get("margin_loss", 0.0),
                    metrics.get("ce_loss", 0.0),
                    metrics.get("accuracy", 0.0),
                    metrics.get("model_loss", 0.0),
                    metrics.get("updates", 0),
                )

            eval_interval = getattr(self.flags, "bc_eval_interval", 0)
            if eval_interval and (epoch + 1) % eval_interval == 0:
                eval_stats = self.evaluate(
                    num_batches=getattr(self.flags, "bc_eval_batches", 1),
                    batch_size=self.eval_batch_size,
                )
                metrics.update({f"eval_{k}": v for k, v in eval_stats.items()})
                if self.logger:
                    self.logger.info(
                        "[BC] eval " + ", ".join(f"{k}={v:.4f}" for k, v in eval_stats.items())
                    )

        self.policy_adapter.close()
        return self._best_metric, history

    def evaluate(self, num_batches: int, batch_size: int) -> Dict[str, float]:
        if num_batches <= 0:
            return {"actor_loss": 0.0, "accuracy": 0.0}
        actor_losses, accuracies = [], []
        for _ in range(num_batches):
            batch = self._sample_batch(batch_size)
            if batch is None:
                continue
            metrics = self._actor_step(batch, train=False)
            actor_losses.append(metrics["loss"])
            accuracies.append(metrics["acc"])
        if not actor_losses:
            return {"actor_loss": 0.0, "accuracy": 0.0}
        return {
            "actor_loss": float(np.mean(actor_losses)),
            "accuracy": float(np.mean(accuracies)),
        }

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float], best: bool) -> None:
        def _atomic_save(payload, file_path: Path) -> None:
            tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
            torch.save(payload, tmp_path)
            tmp_path.replace(file_path)

        base_dir = self.output_dir
        actor_payload = {
            "state_dict": self.actor_net.state_dict(),
            "flags": self.serialized_flags,
            "epoch": epoch,
            "metrics": metrics,
        }
        model_payload = {
            "state_dict": self.model_net.state_dict(),
            "flags": self.serialized_flags,
            "epoch": epoch,
            "metrics": metrics,
        }

        if best:
            _atomic_save(actor_payload, base_dir / "ckp_actor_best.tar")
            _atomic_save(model_payload, base_dir / "ckp_model_best.tar")
        else:
            _atomic_save(actor_payload, base_dir / f"ckp_actor_epoch{epoch + 1:04d}.tar")
            _atomic_save(model_payload, base_dir / f"ckp_model_epoch{epoch + 1:04d}.tar")

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
) -> Tuple[float, List[Dict[str, float]]]:
    loader = create_bc_data_loader(flags)
    trainer = IcoProBehaviorCloningTrainer(
        actor_net=actor_net,
        model_net=model_net,
        data_loader=loader,
        flags=flags,
        logger=logger,
    )
    return trainer.train()
