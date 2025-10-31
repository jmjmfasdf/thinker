from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, NamedTuple

import numpy as np
from thinker import util
import torch
import torch.nn.functional as F

from python_tree import TreeManager, node_expand, node_visit
from thinker.cenv import cModelWrapper
from thinker.dataset_env import BehaviorBatchEnv


class PolicyBatch(NamedTuple):
    logits: torch.Tensor
    log_probs: torch.Tensor
    probs: torch.Tensor
    features: torch.Tensor


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

def compute_icopro_actor_losses(
    policy_adapter: "ThinkerPolicyAdapter",
    obs: torch.Tensor,
    actions: torch.Tensor,
    *,
    prev_actions: Optional[torch.Tensor] = None,
    sequence_starts: Optional[torch.Tensor] = None,
    margin_value: float,
    margin_coef: float,
    ce_coef: float,
    tree_coef: float,
    requires_grad: bool = True,
) -> dict[str, torch.Tensor | float]:
    """Run Thinker forward pass and compute IcoPro-style actor losses."""
    policy = policy_adapter.forward(
        obs,
        actions=actions,
        prev_actions=prev_actions,
        sequence_starts=sequence_starts,
        requires_grad=requires_grad,
    )
    q_policy = policy.logits
    tree_q = policy_adapter.last_tree_q
    if tree_q is not None and tree_coef != 0.0:
        q_values = q_policy + tree_coef * tree_q
    else:
        q_values = q_policy

    margin_tensor = torch.full(
        (actions.shape[0],),
        margin_value,
        dtype=torch.float32,
        device=q_policy.device,
    )
    margin_loss = dqfd_margin_loss(q_values, actions, margin_tensor)
    if ce_coef > 0.0:
        ce_loss = F.cross_entropy(q_policy, actions)
    else:
        ce_loss = torch.zeros_like(margin_loss)
    total_loss = margin_coef * margin_loss + ce_coef * ce_loss

    with torch.no_grad():
        pred = torch.argmax(q_policy, dim=-1)
        accuracy = (pred == actions).float().mean().item()

    return {
        "total_loss": total_loss,
        "margin_loss": margin_loss,
        "ce_loss": ce_loss,
        "accuracy": accuracy,
    }


class ThinkerPolicyAdapter:
    """Run Thinker imagination to expose logits, features, and tree Q values."""

    def __init__(self, actor_net, model_net, flags, device: torch.device):
        self.actor_net = actor_net
        self.model_net = model_net
        self.flags = flags
        self.device = device
        self.num_actions = getattr(actor_net, "num_actions", None)
        if self.num_actions is None:
            raise ValueError("Actor network must define 'num_actions'.")
        self._latent_buffer: Optional[torch.Tensor] = None
        self._tree_manager: Optional[TreeManager] = None
        self._time_step: Optional[torch.Tensor] = None
        self._last_real_actions: Optional[torch.Tensor] = None
        self._last_tree_q: Optional[torch.Tensor] = None
        self._last_tree_reps: Optional[torch.Tensor] = None
        self._last_rollout_history: Optional[List[List[Dict[str, Any]]]] = None
        self._last_imagined_actions: Optional[torch.Tensor] = None
        self._pending_force_reset: Optional[torch.Tensor] = None
        if not hasattr(self.actor_net, "policy"):
            raise AttributeError("Actor network must expose a 'policy' layer for imitation training")
        self._hook = self.actor_net.policy.register_forward_pre_hook(self._capture_latent)
        self.training = self.actor_net.training or self.model_net.training
        # Backend selection: use cenv wrapper for offline rollout
        self._use_cenv_backend = bool(getattr(flags, "icopro_use_cenv", True))
        # Device for IcoPro planner rollout (default CPU to avoid GPU spikes)
        try:
            icopro_dev = getattr(flags, "icopro_device", "cpu")
        except Exception:
            icopro_dev = "cpu"
        self.icopro_device = torch.device(icopro_dev if isinstance(icopro_dev, str) else "cpu")
        self._cenv_dataset_env: Optional[BehaviorBatchEnv] = None
        self._cenv_wrapper: Optional[cModelWrapper] = None
        self._cenv_model_device: Optional[torch.device] = None
        self._cenv_batch_size: int = 0

    def train(self, mode: bool = True):
        self.training = bool(mode)
        self.actor_net.train(mode)
        self.model_net.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def close(self) -> None:
        if self._hook is not None:
            self._hook.remove()
            self._hook = None
        self._tree_manager = None
        self._time_step = None
        self._last_real_actions = None
        self._last_rollout_history = None
        self._last_imagined_actions = None
        if self._cenv_wrapper is not None and hasattr(self._cenv_wrapper, "close"):
            try:
                self._cenv_wrapper.close()
            except Exception:
                pass
        self._cenv_wrapper = None
        self._cenv_dataset_env = None
        self._cenv_model_device = None
        self._cenv_batch_size = 0

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

    def _extract_single_state(self, state, idx):
        if isinstance(state, dict):
            return {k: self._extract_single_state(v, idx) for k, v in state.items()}
        if isinstance(state, (list, tuple)):
            return type(state)(self._extract_single_state(v, idx) for v in state)
        if torch.is_tensor(state):
            return state[idx].detach().clone()
        return state

    def _restore_state_indices(self, target, reference, indices):
        if not indices.numel():
            return
        if isinstance(target, dict):
            for key in target:
                if key in reference:
                    self._restore_state_indices(target[key], reference[key], indices)
        elif isinstance(target, list):
            for tgt, ref in zip(target, reference):
                self._restore_state_indices(tgt, ref, indices)
        elif torch.is_tensor(target):
            target[indices] = reference[indices].detach().clone()

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

    def _ensure_cenv_runner(self, obs_batch: np.ndarray, batch_size: int, model_device: torch.device) -> cModelWrapper:
        rebuild = (
            self._cenv_wrapper is None
            or self._cenv_dataset_env is None
            or self._cenv_batch_size != batch_size
            or self._cenv_model_device != model_device
        )
        if rebuild:
            if self._cenv_wrapper is not None and hasattr(self._cenv_wrapper, "close"):
                try:
                    self._cenv_wrapper.close()
                except Exception:
                    pass
            self._cenv_dataset_env = BehaviorBatchEnv(obs_batch, num_actions=self.num_actions)
            self._cenv_wrapper = cModelWrapper(
                env=self._cenv_dataset_env,
                env_n=batch_size,
                flags=self.flags,
                model_net=self.model_net,
                device=model_device,
                timing=False,
            )
            self._cenv_model_device = model_device
            self._cenv_batch_size = batch_size
        else:
            try:
                self._cenv_dataset_env.update_batch(obs_batch)
            except ValueError:
                # Batch size changed unexpectedly; rebuild from scratch.
                if self._cenv_wrapper is not None and hasattr(self._cenv_wrapper, "close"):
                    try:
                        self._cenv_wrapper.close()
                    except Exception:
                        pass
                self._cenv_dataset_env = BehaviorBatchEnv(obs_batch, num_actions=self.num_actions)
                self._cenv_wrapper = cModelWrapper(
                    env=self._cenv_dataset_env,
                    env_n=batch_size,
                    flags=self.flags,
                    model_net=self.model_net,
                    device=model_device,
                    timing=False,
                )
                self._cenv_model_device = model_device
                self._cenv_batch_size = batch_size
        return self._cenv_wrapper

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
        # xs/hs optional features
        if current_xs is not None:
            env_out.xs = current_xs.unsqueeze(0)
        elif getattr(self.actor_net, "see_x", False):
            xs_shape = getattr(self.actor_net, "xs_shape", None)
            if xs_shape is not None:
                env_out.xs = torch.zeros((1, batch_size, *xs_shape), dtype=torch.float32, device=self.device)
        if current_hs is not None:
            env_out.hs = current_hs.unsqueeze(0)
        elif getattr(self.actor_net, "see_h", False):
            hs_shape = getattr(self.actor_net, "hs_shape", None)
            if hs_shape is not None:
                env_out.hs = torch.zeros((1, batch_size, *hs_shape), dtype=torch.float32, device=self.device)
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

    def _compute_sr_vectors(self, real_state: torch.Tensor, model_state) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not hasattr(self.model_net, "sr_net"):
            return None, None
        sr = self.model_net.sr_net
        with torch.no_grad():
            state = real_state.unsqueeze(0)
            state = self._prepare_model_obs(state)
            state = self.model_net.normalize(state)
            dim_actions = getattr(sr, "dim_rep_actions", 1)
            dummy_action = torch.zeros(1, dim_actions, device=state.device)
            dummy_done = torch.zeros(1, dtype=torch.bool, device=state.device)
            real_vec, _ = sr.encoder(state, dummy_done, dummy_action, {}, flatten=False)
            real_vec = real_vec.squeeze(0).detach().cpu().numpy()
            im_vec = None
            if isinstance(model_state, dict) and "sr_h" in model_state:
                im_vec = model_state["sr_h"].detach().cpu().numpy()
        if im_vec is None and real_vec is not None:
            im_vec = np.copy(real_vec)
        return real_vec, im_vec
    
    def _build_history_entry(
        self,
        real_state: torch.Tensor,
        encoded: Dict[str, Any],
        tree_rep: torch.Tensor,
        *,
        status: int,
        human_action: Optional[int],
        imagined_action: Optional[int],
        forced_reset: bool = False,
    ) -> Dict[str, Any]:
        real_img = torch.clamp(real_state, 0.0, 1.0).detach().cpu().numpy()
        real_img_uint8 = (real_img * 255.0).clip(0, 255).astype(np.uint8)
    
        xs = encoded.get("xs")
        if xs is not None:
            im_tensor = torch.clamp(xs.detach().cpu(), 0.0, 1.0)
            im_img = im_tensor.numpy().astype(np.float32)
        else:
            im_img = np.zeros_like(real_img, dtype=np.float32)
    
        model_state = encoded.get("model_state")
        real_vec, im_vec = self._compute_sr_vectors(real_state, model_state)
    
        if torch.is_tensor(tree_rep):
            tree_rep_np = tree_rep.detach().cpu().numpy()
        else:
            tree_rep_np = np.asarray(tree_rep)
    
        entry = {
            "status": int(status),
            "real_img": real_img_uint8,
            "im_img": im_img,
            "tree_reps": tree_rep_np,
            "real_vectors": real_vec,
            "im_vectors": im_vec,
            "human_action": int(human_action) if human_action is not None else -1,
            "imagined_action": int(imagined_action) if imagined_action is not None else None,
            "forced_reset": bool(forced_reset),
        }
        return entry
    
    
    def _rollout(
        self,
        obs: torch.Tensor,
        initial_action: Optional[torch.Tensor],
        requires_grad: bool,
        real_rewards: Optional[torch.Tensor] = None,
        real_dones: Optional[torch.Tensor] = None,
        sequence_starts: Optional[torch.Tensor] = None,
        prev_actions: Optional[torch.Tensor] = None,
        record_history: bool = False,
    ):
        if self._use_cenv_backend:
            return self._rollout_cenv(
                obs,
                initial_action,
                requires_grad,
                sequence_starts=sequence_starts,
                prev_actions=prev_actions,
                record_history=record_history,
            )
        device = self.device  # actor/device for final supervised loss
        obs_float = obs.to(device).float()
        batch_size = obs_float.shape[0]
    
        if sequence_starts is not None:
            sequence_starts = sequence_starts.to(device=device, dtype=torch.bool).view(batch_size)
        else:
            sequence_starts = torch.ones(batch_size, dtype=torch.bool, device=device)
    
        if prev_actions is not None:
            prev_actions = prev_actions.to(device=device, dtype=torch.long).view(batch_size)
        elif self._last_real_actions is not None and self._last_real_actions.shape[0] == batch_size:
            prev_actions = self._last_real_actions.to(device=device, dtype=torch.long)
        else:
            prev_actions = torch.zeros(batch_size, dtype=torch.long, device=device)
    
        teacher_actions = (
            initial_action.to(device=device, dtype=torch.long).view(batch_size)
            if initial_action is not None
            else None
        )
    
        model_input = self._prepare_model_obs(obs_float)
        init_state = self.model_net.initial_state(batch_size=batch_size, device=device)
        dummy_actions = torch.zeros(1, batch_size, 1, dtype=torch.long, device=device)
        dummy_done = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
        self._last_tree_q = None
        self._last_tree_reps = None
    
        grad_context = torch.enable_grad if requires_grad else torch.no_grad
        with grad_context():
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
    
        if real_rewards is not None:
            root_rewards = real_rewards.to(device=device, dtype=torch.float32).view(-1).detach()
        else:
            root_rewards = init_rewards
        if root_rewards is not None:
            root_rewards = root_rewards.detach()
    
        if real_dones is not None:
            root_dones = real_dones.to(device=device, dtype=torch.bool).view(-1).detach()
        else:
            root_dones = None
    
        initial_xs = initial_model_out.xs[0].detach() if getattr(initial_model_out, "xs", None) is not None else None
        initial_hs = initial_model_out.hs[0].detach() if getattr(initial_model_out, "hs", None) is not None else None
    
        root_payload = []
        for idx in range(batch_size):
            encoded = {"real_states": obs_float[idx]}
            if initial_xs is not None:
                encoded["xs"] = initial_xs[idx]
            if initial_hs is not None:
                encoded["hs"] = initial_hs[idx]
            encoded["model_state"] = self._extract_single_state(initial_model_out.state, idx)
            root_payload.append(encoded)
    
        rewards_tensor = (
            root_rewards.to(device=device, dtype=torch.float32)
            if root_rewards is not None
            else torch.zeros(batch_size, dtype=torch.float32, device=device)
        )
        dones_tensor = (
            root_dones.to(device=device, dtype=torch.bool)
            if root_dones is not None
            else torch.zeros(batch_size, dtype=torch.bool, device=device)
        )
    
        if self._tree_manager is None or self._tree_manager.batch_size != batch_size:
            tree_manager = TreeManager(batch_size=batch_size, num_actions=self.num_actions, flags=self.flags, device=device)
            tree_manager.set_remember_path(bool(getattr(self.flags, "tree_carry", True)))
            tree_manager.expand_root(root_rewards, init_values, init_policy, root_payload, root_dones)
            self._tree_manager = tree_manager
            self._time_step = torch.zeros(batch_size, dtype=torch.long, device=device)
        else:
            tree_manager = self._tree_manager
            if self._time_step is None or self._time_step.shape[0] != batch_size:
                self._time_step = torch.zeros(batch_size, dtype=torch.long, device=device)
            tree_manager.set_remember_path(bool(getattr(self.flags, "tree_carry", True)))
            refresh_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
            for idx in range(batch_size):
                tree_manager.record_real_transition(idx)
                if sequence_starts[idx]:
                    tree_manager.reset_root(
                        idx,
                        rewards_tensor[idx],
                        init_values[idx],
                        init_policy[idx],
                        root_payload[idx],
                        dones_tensor[idx],
                        time_step=0,
                    )
                    self._time_step[idx] = 0
                else:
                    prev_act_val = int(prev_actions[idx].item())
                    done_flag = bool(dones_tensor[idx].item()) if dones_tensor is not None else False
                    if tree_manager.can_carry(idx, prev_act_val, done=done_flag):
                        carried = tree_manager.carry_root(idx, prev_act_val)
                    else:
                        carried = False
                    if carried:
                        refresh_mask[idx] = True
                        self._time_step[idx] += 1
                    else:
                        tree_manager.reset_root(
                            idx,
                            rewards_tensor[idx],
                            init_values[idx],
                            init_policy[idx],
                            root_payload[idx],
                            dones_tensor[idx],
                            time_step=0,
                        )
                        self._time_step[idx] = 0
            if refresh_mask.any():
                tree_manager.refresh_root(
                    rewards_tensor,
                    init_values,
                    init_policy,
                    root_payload,
                    dones_tensor,
                    time_step=self._time_step,
                    mask=refresh_mask,
                )
    
        self._tree_manager = tree_manager
    
        root_status = torch.ones(batch_size, dtype=torch.long, device=self.device)
        tree_reps = tree_manager.compute_tree_reps(status=root_status)
        history = [[] for _ in range(batch_size)] if record_history else None
        root_entry_idx: List[Optional[int]] = [None] * batch_size
    
        if record_history:
            tree_reps_cpu = tree_reps.detach().cpu()
            for idx in range(batch_size):
                encoded_root = tree_manager.root_nodes[idx].encoded or {}
                human_action = (
                    int(teacher_actions[idx].item())
                    if teacher_actions is not None
                    else int(prev_actions[idx].item())
                )
                entry = self._build_history_entry(
                    obs_float[idx],
                    encoded_root,
                    tree_reps_cpu[idx],
                    status=1,
                    human_action=human_action,
                    imagined_action=None,
                )
                history[idx].append(entry)
                root_entry_idx[idx] = len(history[idx]) - 1
    
        current_xs = tree_manager.get_current_encoded("xs")
        if current_xs is not None:
            current_xs = current_xs.detach()
        elif initial_xs is not None:
            current_xs = initial_xs.clone()
    
        current_hs = tree_manager.get_current_encoded("hs")
        if current_hs is not None:
            current_hs = current_hs.detach()
        elif initial_hs is not None:
            current_hs = initial_hs.clone()
    
        initial_model_state = self._clone_state(initial_model_out.state)
        model_state = self._clone_state(initial_model_state)
        actor_core_state = self.actor_net.initial_state(batch_size=batch_size, device=device)
        last_reset = torch.zeros(batch_size, dtype=torch.long, device=device)
        next_reset_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
        self._latent_buffer = None
        env_out_root = self._make_env_out(
            current_xs if current_xs is not None else obs_float,
            tree_reps,
            prev_actions,
            last_reset,
            0,
            current_xs=current_xs,
            current_hs=current_hs,
        )
        with torch.no_grad():
            actor_out, actor_core_state = self.actor_net(env_out_root, core_state=actor_core_state)
        root_actor_action = self._extract_action(actor_out).long()
    
        last_action = prev_actions.clone()
        if sequence_starts.any():
            last_action = torch.where(sequence_starts, root_actor_action, last_action)
    
        rollout_steps = max(0, int(getattr(self.flags, "rec_t", 1)) - 1)
        current_t = 0

        prev_force_reset = getattr(self, "_pending_force_reset", None)
        if prev_force_reset is None or prev_force_reset.shape[0] != batch_size:
            prev_force_reset = torch.zeros(batch_size, dtype=torch.bool, device=device)
        else:
            prev_force_reset = prev_force_reset.to(device=device, dtype=torch.bool)
        prev_force_reset = torch.where(sequence_starts, torch.zeros_like(prev_force_reset), prev_force_reset)
        current_force_reset = prev_force_reset.clone()
    
        for step in range(rollout_steps):
            current_t += 1
            status_mask = []
            needs_expansion_mask = []
            parent_nodes = list(tree_manager.cur_nodes)
            apply_reset_mask = next_reset_mask.clone()
    
            for batch_idx in range(batch_size):
                if apply_reset_mask[batch_idx]:
                    status_mask.append(5)
                    needs_expansion_mask.append(False)
                    continue
                current_node = parent_nodes[batch_idx]
                action_idx = int(last_action[batch_idx].item())
                if not current_node.children:
                    current_node.ensure_children(torch.zeros(self.num_actions, device=device))
                next_node = current_node.children[action_idx]
                is_expanded = len(next_node.children) > 0 and current_t <= next_node.time_step
                if is_expanded:
                    status_mask.append(2)
                    needs_expansion_mask.append(False)
                elif current_node.done:
                    status_mask.append(3)
                    needs_expansion_mask.append(False)
                else:
                    status_mask.append(4)
                    needs_expansion_mask.append(True)
    
            needs_expansion_mask = torch.tensor(needs_expansion_mask, device=device, dtype=torch.bool)
            any_needs_expansion = bool(needs_expansion_mask.any())
            xs = hs = logits = rewards = dones = values = None
            payload = [None] * batch_size

            if any_needs_expansion:
                with grad_context():
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
                if torch.is_tensor(logits):
                    logits = logits.detach()
                rewards = model_out.rs[0] if getattr(model_out, "rs", None) is not None else torch.zeros(batch_size, device=device)
                if torch.is_tensor(rewards) and rewards.dim() == 2:
                    rewards = rewards.squeeze(-1)
                if torch.is_tensor(rewards):
                    rewards = rewards.detach()
                dones = model_out.dones[0] if getattr(model_out, "dones", None) is not None else torch.zeros(batch_size, dtype=torch.bool, device=device)
                if torch.is_tensor(dones):
                    dones = dones.detach()
                values = model_out.vs[0] if getattr(model_out, "vs", None) is not None else torch.zeros(batch_size, device=device)
                if torch.is_tensor(values) and values.dim() == 2:
                    values = values.squeeze(-1)
                if torch.is_tensor(values):
                    values = values.detach()
    
                for idx in range(batch_size):
                    encoded = {}
                    if xs is not None:
                        encoded["xs"] = xs[idx]
                    if hs is not None:
                        encoded["hs"] = hs[idx]
                    encoded["model_state"] = self._extract_single_state(model_out.state, idx)
                    payload[idx] = encoded
    
            tree_manager.advance(last_action, resets=apply_reset_mask)
            if apply_reset_mask.any():
                tree_manager.cur_t[apply_reset_mask] = 0
                reset_indices = torch.nonzero(apply_reset_mask, as_tuple=False).view(-1)
                if reset_indices.numel():
                    if current_xs is not None and initial_xs is not None:
                        current_xs[reset_indices] = initial_xs[reset_indices]
                    if current_hs is not None and initial_hs is not None:
                        current_hs[reset_indices] = initial_hs[reset_indices]
                    self._restore_state_indices(model_state, initial_model_state, reset_indices)
    
            for batch_idx in range(batch_size):
                status = status_mask[batch_idx]
                current_node = tree_manager.cur_nodes[batch_idx]
                parent_node = parent_nodes[batch_idx]
    
                if status == 4:
                    encoded = payload[batch_idx] or {"model_state": self._extract_single_state(model_out.state, batch_idx)}
                    node_expand(
                        current_node,
                        rewards[batch_idx] if rewards is not None else torch.tensor(0.0, device=device),
                        values[batch_idx] if values is not None else torch.tensor(0.0, device=device),
                        t=current_t,
                        done=dones[batch_idx] if dones is not None else torch.tensor(False, device=device),
                        logits=logits[batch_idx] if logits is not None else torch.zeros(self.num_actions, device=device),
                        encoded=encoded,
                        override=True,
                    )
                    node_visit(current_node)
                    current_node.max_q = max(current_node.max_q, current_node.rollout_q)
                elif status == 3:
                    if parent_node.children:
                        logits_zero = torch.tensor([child.logit for child in parent_node.children], device=device)
                    else:
                        logits_zero = torch.zeros(self.num_actions, device=device)
                    encoded = dict(parent_node.encoded or {})
                    node_expand(
                        current_node,
                        torch.tensor(0.0, device=device),
                        torch.tensor(0.0, device=device),
                        t=current_t,
                        done=torch.tensor(True, device=device),
                        logits=logits_zero,
                        encoded=encoded,
                        override=True,
                    )
                    node_visit(current_node)
                    current_node.max_q = max(current_node.max_q, current_node.rollout_q)
                else:
                    node_visit(current_node)
    
            status_tensor = torch.tensor(status_mask, dtype=torch.long, device=self.device)
            tree_reps = tree_manager.compute_tree_reps(reset_flags=apply_reset_mask, status=status_tensor)
            step_status = 2 if step == rollout_steps - 1 else 1
            last_reset = apply_reset_mask.long()
    
            if record_history:
                tree_reps_cpu = tree_reps.detach().cpu()
                for idx in range(batch_size):
                    status_val = int(status_mask[idx])
                    encoded_node = tree_manager.cur_nodes[idx].encoded or {}
                    human_action = (
                        int(teacher_actions[idx].item())
                        if teacher_actions is not None
                        else int(prev_actions[idx].item())
                    )
                    imagined_action = int(last_action[idx].item())
                    forced_reset_flag = bool(prev_force_reset[idx].item()) if status_val == 5 else False
                    entry = self._build_history_entry(
                        obs_float[idx],
                        encoded_node,
                        tree_reps_cpu[idx],
                        status=status_val,
                        human_action=human_action,
                        imagined_action=imagined_action,
                        forced_reset=forced_reset_flag,
                    )
                    history[idx].append(entry)
    
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
            reset_action = torch.zeros(batch_size, dtype=torch.bool, device=device)
            if hasattr(actor_out, "reset") and actor_out.reset is not None:
                reset_probs = actor_out.reset[0]
                reset_action = reset_probs > 0.5
            force_reset = torch.zeros(batch_size, dtype=torch.bool, device=device)
            max_depth = getattr(self.flags, "max_depth", 0)
            if max_depth > 0:
                force_reset = tree_manager.rollout_depth >= max_depth
            next_reset_mask = (reset_action | force_reset).detach().bool()
            current_force_reset = force_reset.detach().bool()
            if record_history:
                tree_reps_cpu = tree_reps.detach().cpu()
                next_action_cpu = next_action.detach().cpu()
                for idx in range(batch_size):
                    status_val = int(status_mask[idx])
                    encoded_node = tree_manager.cur_nodes[idx].encoded or {}
                    human_action = (
                        int(teacher_actions[idx].item())
                        if teacher_actions is not None
                        else int(prev_actions[idx].item())
                    )
                    imagined_action = int(next_action_cpu[idx].item())
                    forced_reset_flag = bool(current_force_reset[idx].item()) if status_val == 5 else False
                    entry = self._build_history_entry(
                        obs_float[idx],
                        encoded_node,
                        tree_reps_cpu[idx],
                        status=status_val,
                        human_action=human_action,
                        imagined_action=imagined_action,
                        forced_reset=forced_reset_flag,
                    )
                    history[idx].append(entry)
            self._latent_buffer = None
            last_action = next_action
            prev_force_reset = current_force_reset.clone()

        self._pending_force_reset = current_force_reset.detach().clone()
        tree_manager.reset_real_step()
        tree_reps = tree_manager.compute_tree_reps(status=torch.ones(batch_size, dtype=torch.long, device=self.device))
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
    
        final_actions = self._extract_action(actor_out).long()
        self._last_real_actions = final_actions.detach().clone()
    
        if record_history:
            final_actions_cpu = final_actions.detach().cpu()
            for idx, entry_idx in enumerate(root_entry_idx):
                if entry_idx is not None:
                    history[idx][entry_idx]["imagined_action"] = int(final_actions_cpu[idx].item())
            self._last_rollout_history = history
            self._last_imagined_actions = final_actions_cpu
        else:
            self._last_rollout_history = None
            self._last_imagined_actions = None
    
        self._last_tree_q = q_values
        self._last_tree_reps = tree_reps.detach()
        return actor_out

    def _rollout_cenv(
        self,
        obs: torch.Tensor,
        initial_action: Optional[torch.Tensor],
        requires_grad: bool,
        *,
        sequence_starts: Optional[torch.Tensor] = None,
        prev_actions: Optional[torch.Tensor] = None,
        record_history: bool = False,
    ):
        device = self.device
        obs_float = obs.to(device).float()
        batch_size = obs_float.shape[0]

        # Prepare uint8/float input for model and env
        # Prepare observations on IcoPro planner device
        if getattr(self.model_net, "state_dtype_n", 0) == 0:
            obs_env = torch.clamp(obs_float * 255.0, 0, 255).to(torch.uint8)
        else:
            obs_env = obs_float
        obs_np = obs_env.detach().cpu().numpy()  # BehaviorBatchEnv expects numpy

        model_dev = self.icopro_device
        try:
            original_device = next(self.model_net.parameters()).device
        except StopIteration:
            original_device = model_dev
        if original_device != model_dev:
            self.model_net.to(model_dev)
        model_train_mode = self.model_net.training
        self.model_net.eval()
        core_env = self._ensure_cenv_runner(obs_np, batch_size, model_dev)
        states, info = core_env.reset(self.model_net)

        # Initialize actor state
        actor_core_state = self.actor_net.initial_state(batch_size=batch_size, device=device)
        last_action = prev_actions.to(device=device, dtype=torch.long).view(batch_size) if prev_actions is not None else torch.zeros(batch_size, dtype=torch.long, device=device)
        last_reset = torch.zeros(batch_size, dtype=torch.long, device=device)

        # Rec_t rollout with imagination + dummy real steps
        for step in range(self.flags.rec_t):
            env_out = self._make_env_out(
                states["real_states"],
                states["tree_reps"],
                last_action,
                last_reset,
                1 if step < self.flags.rec_t - 1 else 0,
                current_xs=states.get("xs"),
                current_hs=states.get("hs"),
            )
            self._latent_buffer = None
            # Move env_out tensors to actor device for the call
            def _to_dev(x):
                return x.to(device) if torch.is_tensor(x) else x
            env_out.real_states = _to_dev(env_out.real_states)
            env_out.tree_reps = _to_dev(env_out.tree_reps)
            if getattr(env_out, "xs", None) is not None:
                env_out.xs = _to_dev(env_out.xs)
            if getattr(env_out, "hs", None) is not None:
                env_out.hs = _to_dev(env_out.hs)
            env_out.done = _to_dev(env_out.done)
            env_out.real_done = _to_dev(env_out.real_done)
            env_out.truncated_done = _to_dev(env_out.truncated_done)
            env_out.last_pri = _to_dev(env_out.last_pri)
            env_out.last_reset = _to_dev(env_out.last_reset)
            env_out.reward = _to_dev(env_out.reward)
            env_out.step_status = _to_dev(env_out.step_status)
            # Always run rollout steps without autograd; only the final pass computes loss
            with torch.no_grad():
                actor_out, actor_core_state = self.actor_net(env_out, core_state=actor_core_state)
            # Detach core state to prevent graph retention between steps
            actor_core_state = tuple(
                (t.detach() if torch.is_tensor(t) else t) for t in actor_core_state
            )
            logits = actor_out.pri_param
            if logits.dim() == 4:
                logits = logits[0, :, 0, :]
            elif logits.dim() == 3:
                logits = logits[0]
            next_action = torch.argmax(logits, dim=-1)
            # Reset handling
            if getattr(actor_out, "reset", None) is not None and actor_out.reset is not None:
                rst = actor_out.reset[-1] if actor_out.reset.dim() > 1 else actor_out.reset
            elif getattr(actor_out, "reset_logits", None) is not None and actor_out.reset_logits is not None:
                rl = actor_out.reset_logits
                if rl.dim() == 4:
                    rl = rl[0, :, 0, :]
                elif rl.dim() == 3:
                    rl = rl[-1]
                rst = torch.argmax(rl, dim=-1).long()
            else:
                rst = torch.zeros_like(next_action)

            # Step wrapper; env ignores actions for real step
            states, reward, done, truncated, info = core_env.step((next_action.detach().cpu().numpy(), rst.detach().cpu().numpy()), self.model_net)
            last_action = next_action.view(batch_size)
            last_reset = rst.view(batch_size)

        # Compute q-values from root tree reps (mean rollout value per action)
        tree_reps = states["tree_reps"].detach()
        dec = util.decode_tree_reps(tree_reps, self.num_actions, getattr(self.actor_net, "dim_actions", 1), self.flags.rec_t, self.flags.critic_enc_type, self.flags.critic_enc_f_type)
        root_qs_mean = dec.get("root_qs_mean")
        if torch.is_tensor(root_qs_mean):
            q_values = root_qs_mean.float().to(device)
        else:
            q_values = torch.tensor(root_qs_mean, dtype=torch.float32, device=device)

        # Final actor forward with loss computation hooks
        final_reset = torch.zeros_like(last_reset)
        # Final visuals are the input observations on the actor device
        final_visual = obs_float
        xs_final = states.get("xs")
        if isinstance(xs_final, torch.Tensor):
            xs_final = xs_final.to(device)
        tree_reps = tree_reps.to(device)
        env_out_final = self._make_env_out(
            final_visual,
            tree_reps,
            last_action,
            final_reset,
            0,
            current_xs=xs_final,
            current_hs=None,
        )
        self._latent_buffer = None
        if requires_grad:
            actor_out, _ = self.actor_net(env_out_final, core_state=actor_core_state, compute_loss=True)
        else:
            with torch.no_grad():
                actor_out, _ = self.actor_net(env_out_final, core_state=actor_core_state, compute_loss=True)

        self._last_real_actions = torch.argmax(getattr(actor_out, "pri_param", self._squeeze_policy_tensor(actor_out.action_prob)), dim=-1)
        self._last_tree_q = q_values
        self._last_tree_reps = tree_reps
        if original_device != model_dev:
            self.model_net.to(original_device)
        self.model_net.train(model_train_mode)
        return actor_out
    def forward(
        self,
        obs: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        *,
        prev_actions: Optional[torch.Tensor] = None,
        requires_grad: Optional[bool] = None,
        real_rewards: Optional[torch.Tensor] = None,
        real_dones: Optional[torch.Tensor] = None,
        sequence_starts: Optional[torch.Tensor] = None,
        record_history: bool = False,
    ) -> PolicyBatch:
        if requires_grad is None:
            requires_grad = self.actor_net.training or self.model_net.training or self.training
        self.training = bool(requires_grad)

        actor_out = self._rollout(
            obs,
            actions,
            requires_grad,
            real_rewards=real_rewards,
            real_dones=real_dones,
            sequence_starts=sequence_starts,
            prev_actions=prev_actions,
            record_history=record_history,
        )
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
        self._last_real_actions = torch.argmax(logits.detach(), dim=-1).to(device=self.device)
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

    @property
    def last_rollout_history(self) -> Optional[List[List[Dict[str, Any]]]]:
        return self._last_rollout_history

    @property
    def last_imagined_actions(self) -> Optional[torch.Tensor]:
        return self._last_imagined_actions
