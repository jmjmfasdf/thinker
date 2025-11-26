from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, NamedTuple

import numpy as np
from thinker import util
import torch
import torch.nn.functional as F

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
        self._last_real_actions: Optional[torch.Tensor] = None
        self._last_tree_q: Optional[torch.Tensor] = None
        self._last_tree_reps: Optional[torch.Tensor] = None
        self._last_rollout_history: Optional[List[List[Dict[str, Any]]]] = None
        self._last_imagined_actions: Optional[torch.Tensor] = None
        if not hasattr(self.actor_net, "policy"):
            raise AttributeError("Actor network must expose a 'policy' layer for imitation training")
        self._hook = self.actor_net.policy.register_forward_pre_hook(self._capture_latent)
        self.training = self.actor_net.training or self.model_net.training
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
                timing=True,
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
                    timing=True,
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

    def _compute_sr_vectors(self, real_state: torch.Tensor, model_state) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        if not hasattr(self.model_net, "sr_net"):
            return None, None, None
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
            elif hasattr(sr, "state") and isinstance(sr.state, dict) and "sr_h" in sr.state:
                im_vec = sr.state["sr_h"].detach().cpu().numpy()
        if im_vec is None and real_vec is not None:
            im_vec = np.copy(real_vec)
        im_vp_vec = None
        vp = getattr(self.model_net, "vp_net", None)
        if vp is not None:
            if isinstance(model_state, dict) and "vp_h" in model_state:
                im_vp_vec = model_state["vp_h"].detach().cpu().numpy()
            elif hasattr(vp, "state") and isinstance(vp.state, dict) and "vp_h" in vp.state:
                im_vp_vec = vp.state["vp_h"].detach().cpu().numpy()
        return real_vec, im_vec, im_vp_vec

    def _extract_tree_rep_vector(self, actor_out) -> Optional[np.ndarray]:
        misc = getattr(actor_out, "misc", None)
        if not isinstance(misc, dict):
            return None
        tree_vec = misc.get("tree_rep_enc")
        if tree_vec is None:
            return None
        if torch.is_tensor(tree_vec):
            tensor = tree_vec.detach()
        else:
            tensor = torch.tensor(tree_vec)
        if tensor.dim() < 3:
            return None
        vec = tensor[-1, 0]
        return vec.cpu().numpy()
    
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
        im_vp_vectors: Optional[np.ndarray] = None,
        tree_rep_vector: Optional[np.ndarray] = None,
        step_times: Optional[np.ndarray] = None,
        env_return: Optional[float] = None,
        cur_rewards: Optional[np.ndarray] = None,
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
        real_vec, im_vec, im_vp_vec = self._compute_sr_vectors(real_state, model_state)
        if im_vp_vectors is not None:
            im_vp_vec = im_vp_vectors
    
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
            "im_vp_vectors": im_vp_vec,
            "tree_reps_vector": tree_rep_vector,
            "step_times": step_times,
            "env_return": env_return,
            "cur_rewards": cur_rewards,
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
        return self._rollout_cenv(
            obs,
            initial_action,
            requires_grad,
            sequence_starts=sequence_starts,
            prev_actions=prev_actions,
            record_history=record_history,
            real_rewards=real_rewards,
            real_dones=real_dones,
        )

    def _rollout_cenv(
        self,
        obs: torch.Tensor,
        initial_action: Optional[torch.Tensor],
        requires_grad: bool,
        *,
        sequence_starts: Optional[torch.Tensor] = None,
        prev_actions: Optional[torch.Tensor] = None,
        record_history: bool = False,
        real_rewards: Optional[torch.Tensor] = None,
        real_dones: Optional[torch.Tensor] = None,
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

        if sequence_starts is not None:
            sequence_starts = sequence_starts.to(device=device, dtype=torch.bool).view(batch_size)
        else:
            sequence_starts = torch.ones(batch_size, dtype=torch.bool, device=device)
        teacher_actions = (
            initial_action.to(device=device, dtype=torch.long).view(batch_size)
            if initial_action is not None
            else None
        )
        if prev_actions is not None:
            last_action = prev_actions.to(device=device, dtype=torch.long).view(batch_size)
        elif self._last_real_actions is not None and self._last_real_actions.shape[0] == batch_size:
            last_action = self._last_real_actions.to(device=device, dtype=torch.long)
        else:
            last_action = torch.zeros(batch_size, dtype=torch.long, device=device)
        if sequence_starts.any() and teacher_actions is not None:
            last_action = torch.where(sequence_starts, teacher_actions, last_action)

        # Initialize actor state
        actor_core_state = self.actor_net.initial_state(batch_size=batch_size, device=device)
        last_reset = torch.zeros(batch_size, dtype=torch.long, device=device)
        history = [[] for _ in range(batch_size)] if record_history else None

        def _record_history(states_snapshot, status_snapshot, imagined_action_tensor=None, forced_reset_tensor=None, tree_rep_vector=None, step_times=None):
            if history is None:
                return
            tree_reps_cpu = states_snapshot["tree_reps"].detach().cpu()
            real_states_tensor = states_snapshot["real_states"]
            xs_tensor = states_snapshot.get("xs")
            hs_tensor = states_snapshot.get("hs")
            if torch.is_tensor(xs_tensor):
                xs_tensor = xs_tensor.detach()
            if torch.is_tensor(hs_tensor):
                hs_tensor = hs_tensor.detach()
            if status_snapshot is None:
                status_tensor = torch.zeros(batch_size, dtype=torch.long, device=real_states_tensor.device)
            elif torch.is_tensor(status_snapshot):
                status_tensor = status_snapshot.view(-1)
            else:
                status_tensor = torch.tensor(status_snapshot, device=real_states_tensor.device).view(-1)
            imagined_tensor = imagined_action_tensor.detach() if torch.is_tensor(imagined_action_tensor) else imagined_action_tensor
            forced_tensor = forced_reset_tensor.detach() if torch.is_tensor(forced_reset_tensor) else forced_reset_tensor
            human_tensor = teacher_actions if teacher_actions is not None else last_action
            step_times_tensor = step_times
            if torch.is_tensor(step_times_tensor):
                step_times_tensor = step_times_tensor.detach()
            tree_rep_vector_tensor = tree_rep_vector
            if torch.is_tensor(tree_rep_vector_tensor):
                tree_rep_vector_tensor = tree_rep_vector_tensor.detach()
            for idx in range(batch_size):
                real_state = real_states_tensor[idx]
                if real_state.dtype == torch.uint8:
                    real_state = real_state.float() / 255.0
                encoded = {
                    "xs": xs_tensor[idx] if xs_tensor is not None else None,
                    "hs": hs_tensor[idx] if hs_tensor is not None else None,
                    "model_state": None,
                }
                human_val = int(human_tensor[idx].item()) if human_tensor is not None else -1
                if imagined_tensor is None:
                    imagined_val = None
                elif torch.is_tensor(imagined_tensor):
                    imagined_val = int(imagined_tensor[idx].item())
                else:
                    imagined_val = int(imagined_tensor[idx])
                forced_val = bool(forced_tensor[idx].item()) if forced_tensor is not None else False
                status_val = int(status_tensor[idx].item()) if torch.is_tensor(status_tensor) else int(status_tensor[idx])
                trv = None
                if tree_rep_vector_tensor is not None:
                    if isinstance(tree_rep_vector_tensor, (list, tuple, np.ndarray)) or torch.is_tensor(tree_rep_vector_tensor):
                        trv_elem = tree_rep_vector_tensor[idx]
                    else:
                        trv_elem = tree_rep_vector_tensor
                    if torch.is_tensor(trv_elem):
                        trv = trv_elem.detach().cpu().numpy()
                    else:
                        trv = np.asarray(trv_elem)
                st_entry = None
                if step_times_tensor is not None:
                    st_elem = step_times_tensor[idx] if hasattr(step_times_tensor, "__len__") else step_times_tensor
                    if torch.is_tensor(st_elem):
                        st_entry = st_elem.detach().cpu().numpy().astype(np.float32)
                    else:
                        st_entry = np.asarray(st_elem, dtype=np.float32)
                entry = self._build_history_entry(
                    real_state,
                    encoded,
                    tree_reps_cpu[idx],
                    status=status_val,
                    human_action=human_val,
                    imagined_action=imagined_val,
                    forced_reset=forced_val,
                    im_vp_vectors=None,
                    tree_rep_vector=trv,
                    step_times=st_entry,
                )
                history[idx].append(entry)

        status_tensor = info.get("step_status") if isinstance(info, dict) else None
        _record_history(states, status_tensor, imagined_action_tensor=None, forced_reset_tensor=None, tree_rep_vector=None, step_times=info.get("step_times") if isinstance(info, dict) else None)

        # Rec_t rollout with imagination + dummy real steps
        for step in range(self.flags.rec_t):
            status_tensor = info.get("step_status") if isinstance(info, dict) else None
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

            tree_vec = self._extract_tree_rep_vector(actor_out)
            _record_history(states, status_tensor, imagined_action_tensor=next_action, forced_reset_tensor=None, tree_rep_vector=tree_vec, step_times=info.get("step_times") if isinstance(info, dict) else None)

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
        self._last_tree_reps = tree_reps.detach()
        if history is not None:
            self._last_rollout_history = history
            self._last_imagined_actions = last_action.detach().cpu()
        else:
            self._last_rollout_history = None
            self._last_imagined_actions = None
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
