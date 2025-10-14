from __future__ import annotations

from types import SimpleNamespace
from typing import Optional, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

from python_tree import TreeManager


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
    margin_value: float,
    margin_coef: float,
    ce_coef: float,
    tree_coef: float,
    requires_grad: bool = True,
) -> dict[str, torch.Tensor | float]:
    """Run Thinker forward pass and compute IcoPro-style actor losses."""
    policy = policy_adapter.forward(obs, actions=actions, requires_grad=requires_grad)
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
        self._last_tree_q: Optional[torch.Tensor] = None
        self._last_tree_reps: Optional[torch.Tensor] = None
        if not hasattr(self.actor_net, "policy"):
            raise AttributeError("Actor network must expose a 'policy' layer for imitation training")
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
        real_rewards: Optional[torch.Tensor] = None,
        real_dones: Optional[torch.Tensor] = None,
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
            root_dones = real_dones.to(device=device)
            if root_dones.dtype != torch.bool:
                root_dones = root_dones.bool()
            root_dones = root_dones.view(-1).detach()
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
            root_payload.append(encoded)
        tree_manager = TreeManager(batch_size=batch_size, num_actions=self.num_actions, flags=self.flags, device=device)
        tree_manager.expand_root(root_rewards, init_values, init_policy, root_payload, root_dones)
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
        
        # Track current time step for node expansion checking
        current_t = 0
        
        for step in range(rollout_steps):
            current_t += 1
            
            # Check node expansion status BEFORE advancing (mimicking cenv.pyx lines 762-773)
            # Determine which nodes need expansion (status 4) vs already expanded (status 2) vs done (status 3)
            needs_expansion_mask = []
            status_mask = []
            
            for batch_idx in range(batch_size):
                current_node = tree_manager.cur_nodes[batch_idx]
                action_idx = int(last_action[batch_idx].item())
                
                # Ensure children exist
                if not current_node.children:
                    current_node.ensure_children(torch.zeros(self.num_actions, device=device))
                
                next_node = current_node.children[action_idx]
                
                # Check if node is already expanded (mimicking cenv.pyx node_expanded())
                # bool node_expanded(Node* pnode, int t):
                #     return pnode[0].ppchildren[0].size() > 0 and t <= pnode[0].t
                is_expanded = len(next_node.children) > 0 and current_t <= next_node.time_step
                
                if is_expanded:
                    # Status 2: already expanded
                    needs_expansion_mask.append(False)
                    status_mask.append(2)
                elif current_node.done:
                    # Status 3: done already
                    needs_expansion_mask.append(False)
                    status_mask.append(3)
                else:
                    # Status 4: need expand
                    needs_expansion_mask.append(True)
                    status_mask.append(4)
            
            needs_expansion_mask = torch.tensor(needs_expansion_mask, device=device)
            any_needs_expansion = needs_expansion_mask.any()
            
            # Only run model forward if ANY node needs expansion (status 4)
            # Mimicking cenv.pyx lines 835-851
            if any_needs_expansion:
                # For efficiency, only forward the subset that needs expansion
                # But for simplicity in batched operations, we forward all and only use needed outputs
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
            
            # Advance to next node (mimicking cenv.pyx lines 887, 890, 901, 923)
            tree_manager.advance(last_action)
            
            # Now tree_manager.cur_nodes point to the next nodes
            # Expand if needed based on status (mimicking cenv.pyx status handling)
            for batch_idx in range(batch_size):
                status = status_mask[batch_idx]
                current_node = tree_manager.cur_nodes[batch_idx]
                
                if status == 4:
                    # Status 4: need expand - use model outputs
                    # Mimicking cenv.pyx lines 903-923
                    from python_tree import node_expand, node_visit
                    node_expand(
                        current_node,
                        rewards[batch_idx],
                        values[batch_idx],
                        t=current_t,
                        done=dones[batch_idx],
                        logits=logits[batch_idx],
                        encoded=payload[batch_idx],
                        override=True  # Override because child was created by ensure_children
                    )
                    node_visit(current_node)
                    current_node.max_q = max(current_node.max_q, current_node.rollout_q)
                elif status == 3:
                    # Status 3: done node - expand with zero values
                    # Mimicking cenv.pyx lines 894-901
                    from python_tree import node_expand, node_visit
                    parent_node = tree_manager.root_nodes[batch_idx]  # Get parent for logits
                    if len(parent_node.children) > 0:
                        logits_zero = torch.tensor([child.logit for child in parent_node.children], device=device)
                    else:
                        logits_zero = torch.zeros(self.num_actions, device=device)
                    node_expand(
                        current_node,
                        torch.tensor(0.0, device=device),
                        torch.tensor(0.0, device=device),
                        t=current_t,
                        done=torch.tensor(True, device=device),
                        logits=logits_zero,
                        encoded=parent_node.encoded or {},
                        override=True
                    )
                    node_visit(current_node)
                    current_node.max_q = max(current_node.max_q, current_node.rollout_q)
                else:
                    # Status 2: already expanded - just visit
                    # Mimicking cenv.pyx lines 887-890
                    from python_tree import node_visit
                    # Force visit to accumulate rollout statistics
                    current_node.visited = False
                    node_visit(current_node)
            
            # Update rollout depth after processing
            tree_manager.cur_t[:] = current_t
            tree_manager.rollout_depth += 1
            
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
        real_rewards: Optional[torch.Tensor] = None,
        real_dones: Optional[torch.Tensor] = None,
    ) -> PolicyBatch:
        actor_out = self._rollout(
            obs,
            actions,
            requires_grad,
            real_rewards=real_rewards,
            real_dones=real_dones,
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
