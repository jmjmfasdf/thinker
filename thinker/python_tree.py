"""Pure-Python reimplementation of Thinker's tree bookkeeping utilities.

This module mirrors the key pieces of the Cython cenv implementation so
that behavioural cloning can build identical tree representations without
linking against the C++ extension.  The implementation favours clarity
and API compatibility over raw speed – BC workloads operate offline with
small batch sizes, so the overhead is acceptable.

The code is organised as follows:
    * Node dataclass replicates the C struct used by Thinker.
    * Helper functions ``node_new``/``node_expand``/``node_visit``/... copy
      the semantics of their Cython counterparts.
    * ``TreeManager`` wraps a batch of trees (one per environment index)
      and exposes ``expand_current`` / ``advance`` / ``compute_tree_reps``
      utilities that mimic ``cModelWrapper``.

Only the features needed by the BC pipeline are implemented so far, but
additional flags (masking, remember_path, etc.) can be added as needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Encoding helpers – copied from thinker/thinker/cenv.pyx
# ---------------------------------------------------------------------------

def _sign(x: float) -> float:
    if x > 0.0:
        return 1.0
    if x < 0.0:
        return -1.0
    return 0.0


def _enc_mu_zero(x: float) -> float:
    """MuZero-style value encoding (enc_0 in Cython)."""
    return _sign(x) * (math.sqrt(abs(x) + 1.0) - 1.0) + 0.001 * x


def _enc_log(x: float) -> float:
    """Dreamer-style log encoding (enc_1 in Cython)."""
    return _sign(x) * math.log(abs(x) + 1.0)


def _apply_encoding_scalar(x: float, enc_type: int, enc_f_type: int) -> float:
    if enc_type == 0:
        return x
    if enc_f_type == 0:
        return _enc_mu_zero(x)
    if enc_f_type == 1:
        return _enc_log(x)
    raise ValueError(f"Unsupported enc_f_type={enc_f_type}")


def _safe_average(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_max(values: Sequence[float]) -> float:
    return float(np.max(values)) if values else 0.0


def _as_float_list(values: Any, expected_len: int) -> List[float]:
    if isinstance(values, torch.Tensor):
        flat = values.reshape(-1).tolist()
    elif isinstance(values, np.ndarray):
        flat = values.reshape(-1).tolist()
    else:
        flat = list(values)
    if len(flat) != expected_len:
        raise ValueError(f"Expected {expected_len} logits, got {len(flat)}")
    return [float(v) for v in flat]


# ---------------------------------------------------------------------------
# Node definition – mirrors the Thinker cenv Node struct
# ---------------------------------------------------------------------------

@dataclass
class Node:
    action: int
    num_actions: int
    discounting: float
    rec_t: int
    logit: float = 0.0
    parent: Optional["Node"] = None
    remember_path: bool = False

    reward: float = 0.0
    value: float = 0.0
    time_step: int = 0
    done: bool = False

    trail_r: float = 0.0
    trail_discount: float = 1.0
    rollout_q: float = 0.0
    visited: bool = False

    rollout_qs: List[float] = field(default_factory=list)
    paths: List[List["Node"]] = field(default_factory=list)
    rollout_n: int = 0
    max_q: float = 0.0

    encoded: Optional[Dict[str, Any]] = None
    children: List["Node"] = field(default_factory=list)

    def ensure_children(self, logits: Any) -> None:
        logits_list = _as_float_list(logits, self.num_actions)
        if self.children and len(self.children) == self.num_actions:
            # Just refresh logits
            for child, logit in zip(self.children, logits_list):
                child.logit = float(logit)
            return
        self.children = [
            Node(
                action=a,
                num_actions=self.num_actions,
                discounting=self.discounting,
                rec_t=self.rec_t,
                parent=self,
                logit=float(logits_list[a]),
                remember_path=self.remember_path,
            )
            for a in range(self.num_actions)
        ]

    # Alias properties to mirror C struct naming -------------------------
    @property
    def r(self) -> float:
        return self.reward

    @r.setter
    def r(self, value: float) -> None:
        self.reward = float(value)

    @property
    def v(self) -> float:
        return self.value

    @v.setter
    def v(self, value: float) -> None:
        self.value = float(value)

    @property
    def t(self) -> int:
        return self.time_step

    @t.setter
    def t(self, value: int) -> None:
        self.time_step = int(value)


# ---------------------------------------------------------------------------
# Node manipulation helpers
# ---------------------------------------------------------------------------


def node_new(
    parent: Optional[Node],
    action: int,
    logit: float,
    num_actions: int,
    discounting: float,
    rec_t: int,
    remember_path: bool,
) -> Node:
    return Node(
        action=action,
        parent=parent,
        num_actions=num_actions,
        discounting=discounting,
        rec_t=rec_t,
        logit=logit,
        remember_path=remember_path,
    )


def node_expanded(node: Node, t: int) -> bool:
    """Mirror the C helper that checks whether a node is already expanded."""
    return bool(node.children) and t <= node.time_step


def node_expand(
    node: Node,
    reward: Any,
    value: Any,
    t: int,
    done: Any,
    logits: Any,
    encoded: Optional[Dict[str, Any]],
    override: bool = False,
) -> None:
    """Expand or refresh node statistics using current model outputs."""
    reward_f = float(reward.item()) if torch.is_tensor(reward) else float(reward)
    value_f = float(value.item()) if torch.is_tensor(value) else float(value)
    done_f = bool(done.item()) if torch.is_tensor(done) else bool(done)

    if override and not node_expanded(node, -1):
        override = False  # match Cython behaviour

    if not override:
        assert not node_expanded(node, -1), "node should not be expanded"
    else:
        if node.rollout_qs:
            base_q = reward_f + value_f * node.discounting
            old_reward = node.reward
            old_value = node.value
            node.rollout_qs[0] = base_q
            for idx in range(1, len(node.rollout_qs)):
                node.rollout_qs[idx] = node.rollout_qs[idx] - old_reward + reward_f
            if node.parent and node.remember_path:
                node_refresh(
                    node.parent,
                    node,
                    r_diff=reward_f - old_reward,
                    v_diff=value_f - old_value,
                    discounting=node.discounting,
                    depth=1,
                )

    node.reward = reward_f
    node.value = value_f
    node.time_step = int(t)
    node.done = done_f
    node.encoded = encoded
    node.ensure_children(logits)


def node_refresh(node: Node, target: Node, r_diff: float, v_diff: float, discounting: float, depth: int) -> None:
    """Propagate differential updates to stored rollout values."""
    for path_idx, path in enumerate(node.paths):
        k = len(path) - 1 - depth
        if k < 0:
            continue
        if path[k] is target:
            node.rollout_qs[path_idx] += discounting * r_diff
            if k == 0:
                node.rollout_qs[path_idx] += discounting * node.discounting * v_diff
    if node.parent is not None:
        node_refresh(node.parent, target, r_diff, v_diff, discounting * node.discounting, depth + 1)


def node_propagate(node: Node, reward: float, value: float, new_rollout: bool, path: Optional[List[Node]]) -> None:
    node.trail_r += node.trail_discount * reward
    node.trail_discount *= node.discounting
    node.rollout_q = node.trail_r + node.trail_discount * value
    if new_rollout:
        if node.remember_path:
            base_path = list(path) if path else []
            stored_path = base_path + [node]
            node.paths.append(stored_path)
            next_path = stored_path
        else:
            next_path = None
        node.rollout_qs.append(node.rollout_q)
    else:
        next_path = list(path) + [node] if path is not None else None
    if node.parent is not None:
        node_propagate(node.parent, reward, value, new_rollout, next_path)


def node_visit(node: Node) -> None:
    path: Optional[List[Node]] = [] if node.remember_path else None
    node.trail_r = 0.0
    node.trail_discount = 1.0
    new_rollout = not node.visited
    node_propagate(node, node.reward, node.value, new_rollout=new_rollout, path=path)
    node.visited = True
    node.rollout_n += 1


def node_stat(
    node: Node,
    detailed: bool,
    enc_type: int,
    enc_f_type: int,
    mask_type: int,
    raw_num_actions: Optional[int] = None,
) -> torch.Tensor:

    num_actions = node.num_actions if raw_num_actions in (None, -1) else raw_num_actions
    obs_n = num_actions * 5 + 3
    if detailed:
        obs_n += 3

    result = torch.zeros(obs_n, dtype=torch.float32)
    result[node.action] = 1.0

    node.max_q = 0.0
    if node.discounting != 0.0:
        node.max_q = (_safe_max(node.rollout_qs) - node.reward) / node.discounting

    if mask_type == 3:
        return result

    result[num_actions] = _apply_encoding_scalar(node.reward, enc_type, enc_f_type)
    result[num_actions + 1] = float(node.done)
    if mask_type not in (2,):
        result[num_actions + 2] = _apply_encoding_scalar(node.value, enc_type, enc_f_type)

    base_logits = num_actions + 3
    mean_base = num_actions * 2 + 3
    max_base = num_actions * 3 + 3
    visit_base = num_actions * 4 + 3

    for idx, child in enumerate(node.children):
        if mask_type not in (2,):
            result[base_logits + idx] = float(child.logit)
        if mask_type in (1, 2) or (mask_type == 5 and not detailed):
            continue
        mean_q = _apply_encoding_scalar(_safe_average(child.rollout_qs), enc_type, enc_f_type)
        max_q = _apply_encoding_scalar(_safe_max(child.rollout_qs), enc_type, enc_f_type)
        result[mean_base + idx] = mean_q
        if mask_type not in (3, 4):
            result[max_base + idx] = max_q
        rec_t = float(node.rec_t) if node.rec_t > 0 else 1.0
        result[visit_base + idx] = child.rollout_n / rec_t

    base_idx = num_actions * 5 + 3
    if detailed and mask_type not in (1, 2, 4):
        discount = node.discounting if node.discounting != 0.0 else 1.0
        trail_term = (node.trail_r - node.reward) / discount
        rollout_term = (node.rollout_q - node.reward) / discount
        result[base_idx] = _apply_encoding_scalar(trail_term, enc_type, enc_f_type)
        result[base_idx + 1] = _apply_encoding_scalar(rollout_term, enc_type, enc_f_type)
        result[base_idx + 2] = _apply_encoding_scalar(node.max_q, enc_type, enc_f_type)

    return result


# ---------------------------------------------------------------------------
# Tree maintenance helpers for carry behaviour
# ---------------------------------------------------------------------------


def node_delete(node: Node, except_idx: int = -1) -> Optional[Node]:
    children = list(node.children)
    survivor: Optional[Node] = None
    for idx, child in enumerate(children):
        if except_idx >= 0 and idx == except_idx:
            child.parent = None
            survivor = child
            continue
        node_delete(child, -1)
    node.children = []
    node.rollout_qs.clear()
    node.paths.clear()
    node.rollout_n = 0
    node.trail_r = 0.0
    node.trail_discount = 1.0
    node.rollout_q = 0.0
    node.max_q = 0.0
    return survivor


# ---------------------------------------------------------------------------
# High-level tree wrapper used by the BC pipeline
# ---------------------------------------------------------------------------


class TreeManager:
    """Manages a batch of Thinker trees fully in Python."""

    def __init__(self, batch_size: int, num_actions: int, *, flags: Any, device: torch.device):
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.flags = flags
        self.device = device

        self.discounting = float(getattr(flags, "discounting", 0.99))
        base_rec_t = int(getattr(flags, "rec_t", 40))
        test_rec_t = int(getattr(flags, "test_rec_t", -1))
        if test_rec_t > 0:
            self.rec_t = test_rec_t
            self.rep_rec_t = base_rec_t
        else:
            self.rec_t = base_rec_t
            self.rep_rec_t = base_rec_t
        self.has_action_seq = bool(getattr(flags, "has_action_seq", False))
        self.max_depth = int(getattr(flags, "max_depth", 40))
        self.reset_mode = int(getattr(flags, "reset_mode", 0))
        self.enc_type = int(
            getattr(flags, "model_enc_type", getattr(flags, "critic_enc_type", 0))
        )
        self.enc_f_type = int(
            getattr(flags, "model_enc_f_type", getattr(flags, "critic_enc_f_type", 0))
        )
        self.mask_type = int(getattr(flags, "stat_mask_type", 0))
        self.tree_carry = bool(getattr(flags, "tree_carry", True))

        self.remember_path = bool(getattr(flags, "remember_path", True))
        self.raw_num_actions = getattr(flags, "raw_num_actions", num_actions)

        self.root_nodes: List[Node] = [
            node_new(
                parent=None,
                action=0,
                logit=0.0,
                num_actions=self.num_actions,
                discounting=self.discounting,
                rec_t=self.rec_t,
                remember_path=self.remember_path,
            )
            for _ in range(batch_size)
        ]
        self.cur_nodes: List[Node] = list(self.root_nodes)
        self.rollout_depth = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.cur_t = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.total_step = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.max_rollout_depth = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.max_rollout_depth_snapshot = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.root_nodes_qmax = torch.zeros(batch_size, dtype=torch.float32, device=device)
        self.baseline_mean_q = torch.zeros(batch_size, dtype=torch.float32, device=device)
        self.step_status = torch.zeros(batch_size, dtype=torch.long, device=device)

    # ------------------------------------------------------------------
    # Tree mutation APIs
    # ------------------------------------------------------------------

    def _propagate_remember_path(self, node: Node, enabled: bool) -> None:
        node.remember_path = enabled
        if not enabled:
            node.paths.clear()
        for child in node.children:
            self._propagate_remember_path(child, enabled)

    def set_remember_path(self, enabled: bool) -> None:
        if self.remember_path == enabled:
            return
        self.remember_path = enabled
        for root in self.root_nodes:
            self._propagate_remember_path(root, enabled)

    def record_real_transition(self, index: int) -> None:
        root = self.root_nodes[index]
        mean_q = _safe_average(root.rollout_qs) if root.rollout_qs else root.rollout_q
        discount = root.discounting if root.discounting != 0.0 else 1.0
        self.baseline_mean_q[index] = (mean_q - root.reward) / discount
        self.total_step[index] += 1

    def _update_root_stats(self, index: int) -> None:
        root = self.root_nodes[index]
        self.root_nodes_qmax[index] = float(root.max_q)
        self.max_rollout_depth_snapshot[index] = self.max_rollout_depth[index]

    def _update_step_status(self) -> None:
        for idx in range(self.batch_size):
            cur_t_val = int(self.cur_t[idx].item())
            if cur_t_val == 0:
                self.step_status[idx] = 0
            elif self.rec_t <= 1:
                self.step_status[idx] = 3
            elif cur_t_val < self.rec_t - 1:
                self.step_status[idx] = 1
            else:
                self.step_status[idx] = 2

    def expand_root(
        self,
        rewards: Optional[torch.Tensor],
        values: torch.Tensor,
        logits: torch.Tensor,
        encoded: Optional[Sequence[Optional[Dict[str, Any]]]],
        dones: Optional[torch.Tensor] = None,
        time_step: Optional[Any] = None,
        mask: Optional[Any] = None,
    ) -> None:
        def _resolve_time_step(idx: int) -> int:
            if time_step is None:
                return 0
            if isinstance(time_step, torch.Tensor):
                return int(time_step[idx].item() if time_step.ndim > 0 else time_step.item())
            if isinstance(time_step, (list, tuple)):
                return int(time_step[idx])
            return int(time_step)

        for idx, root in enumerate(self.root_nodes):
            if mask is not None:
                active = bool(mask[idx].item()) if torch.is_tensor(mask) else bool(mask[idx])
                if not active:
                    continue
            if rewards is not None:
                r = rewards[idx]
            else:
                r = 0.0
            v = values[idx]
            logit_vec = logits[idx]
            if dones is not None:
                done_flag = dones[idx]
            else:
                done_flag = False
            node_expand(
                root,
                r,
                v,
                t=_resolve_time_step(idx),
                done=done_flag,
                logits=logit_vec,
                encoded=encoded[idx] if encoded is not None else None,
                override=False,
            )
            node_visit(root)
            self._update_root_stats(idx)



    def can_carry(self, index: int, action: int, *, done: bool = False) -> bool:
        if not self.tree_carry or done:
            return False
        if index < 0 or index >= self.batch_size:
            return False
        root = self.root_nodes[index]
        if not root.children or action < 0 or action >= len(root.children):
            return False
        child = root.children[action]
        current_step = int(self.total_step[index].item())
        return node_expanded(child, current_step)

    def carry_root(self, index: int, action: int) -> bool:
        if not self.can_carry(index, action):
            return False
        root = self.root_nodes[index]
        child = root.children[action]
        survivor = node_delete(root, action)
        new_root = survivor if survivor is not None else child
        new_root.parent = None
        self._propagate_remember_path(new_root, self.remember_path)
        self.root_nodes[index] = new_root
        self.cur_nodes[index] = new_root
        self.rollout_depth[index] = 0
        self.cur_t[index] = 0
        self.max_rollout_depth[index] = 0
        self.max_rollout_depth_snapshot[index] = 0
        self._update_root_stats(index)
        return True


    def reset_root(
        self,
        index: int,
        reward: Any,
        value: Any,
        logits: Any,
        encoded: Optional[Dict[str, Any]],
        done: Any,
        time_step: int = 0,
    ) -> None:
        if index < 0 or index >= self.batch_size:
            raise IndexError("TreeManager.reset_root index out of range")
        root = node_new(
            parent=None,
            action=0,
            logit=0.0,
            num_actions=self.num_actions,
            discounting=self.discounting,
            rec_t=self.rec_t,
            remember_path=self.remember_path,
        )
        node_expand(
            root,
            reward,
            value,
            t=time_step,
            done=done,
            logits=logits,
            encoded=encoded,
            override=False,
        )
        node_visit(root)
        self._propagate_remember_path(root, self.remember_path)
        self.root_nodes[index] = root
        self.cur_nodes[index] = root
        self.rollout_depth[index] = 0
        self.cur_t[index] = 0
        self.max_rollout_depth[index] = 0
        self.max_rollout_depth_snapshot[index] = 0
        self._update_root_stats(index)

    def refresh_root(
        self,
        rewards: Optional[torch.Tensor],
        values: torch.Tensor,
        logits: torch.Tensor,
        encoded: Sequence[Optional[Dict[str, Any]]],
        dones: Optional[torch.Tensor] = None,
        time_step: Optional[Any] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        def _resolve_time_step(idx: int) -> int:
            if time_step is None:
                return 0
            if isinstance(time_step, torch.Tensor):
                return int(time_step[idx].item() if time_step.ndim > 0 else time_step.item())
            if isinstance(time_step, (list, tuple)):
                return int(time_step[idx])
            return int(time_step)

        for idx, root in enumerate(self.root_nodes):
            if mask is not None:
                active = bool(mask[idx].item()) if torch.is_tensor(mask) else bool(mask[idx])
                if not active:
                    continue
            if rewards is not None:
                r = rewards[idx]
            else:
                r = 0.0
            v = values[idx]
            logit_vec = logits[idx]
            if dones is not None:
                done_flag = dones[idx]
            else:
                done_flag = False
            node_expand(
                root,
                r,
                v,
                t=_resolve_time_step(idx),
                done=done_flag,
                logits=logit_vec,
                encoded=encoded[idx] if encoded is not None else None,
                override=True,
            )
            node_visit(root)
            self._update_root_stats(idx)


    def expand_current(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        logits: torch.Tensor,
        encoded: Sequence[Optional[Dict[str, Any]]],
        override: bool = False,
        time_step: Optional[Any] = None,
    ) -> None:
        def _resolve_time_step(idx: int) -> int:
            if time_step is None:
                return int(self.cur_t[idx].item())
            if isinstance(time_step, torch.Tensor):
                return int(time_step[idx].item() if time_step.ndim > 0 else time_step.item())
            if isinstance(time_step, (list, tuple)):
                return int(time_step[idx])
            return int(time_step)

        for idx in range(self.batch_size):
            node = self.cur_nodes[idx]
            node_expand(
                node,
                rewards[idx],
                values[idx],
                t=_resolve_time_step(idx),
                done=dones[idx],
                logits=logits[idx],
                encoded=encoded[idx] if encoded is not None else None,
                override=override,
            )
            node_visit(node)
    def advance(self, actions: torch.Tensor, resets: Optional[torch.Tensor] = None) -> None:
        if resets is None:
            if torch.is_tensor(actions):
                resets = torch.zeros_like(actions, dtype=torch.bool)
            else:
                resets = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        new_cur_nodes: List[Node] = []
        new_cur_t = self.cur_t.clone()
        for idx in range(self.batch_size):
            reset_flag = bool(resets[idx].item()) if torch.is_tensor(resets) else bool(resets[idx])
            if reset_flag:
                new_cur_nodes.append(self.root_nodes[idx])
                self.rollout_depth[idx] = 0
                new_cur_t[idx] = 0
                self.max_rollout_depth[idx] = 0
                self.max_rollout_depth_snapshot[idx] = 0
                continue
            if torch.is_tensor(actions):
                action = int(actions[idx].item())
            elif isinstance(actions, (list, tuple)):
                action = int(actions[idx])
            else:
                action = int(actions)
            current = self.cur_nodes[idx]
            if not current.children:
                current.ensure_children([0.0] * self.num_actions)
            next_node = current.children[action]
            new_cur_nodes.append(next_node)
            depth = int(self.rollout_depth[idx].item()) + 1
            if self.max_depth > 0:
                depth = min(depth, self.max_depth)
            self.rollout_depth[idx] = depth
            current_max = int(self.max_rollout_depth[idx].item())
            if depth > current_max:
                self.max_rollout_depth[idx] = depth
            self.max_rollout_depth_snapshot[idx] = self.max_rollout_depth[idx]
            new_cur_t[idx] = int(new_cur_t[idx].item()) + 1
        self.cur_nodes = new_cur_nodes
        self.cur_t = new_cur_t
        self._update_step_status()

    def reset_real_step(self) -> None:
        self.cur_t.zero_()
        self.rollout_depth.zero_()
        self.max_rollout_depth.zero_()
        self.max_rollout_depth_snapshot.zero_()
        self.cur_nodes = [self.root_nodes[idx] for idx in range(self.batch_size)]
        self._update_step_status()

    # ------------------------------------------------------------------
    # Tree representation utilities
    # ------------------------------------------------------------------

    def compute_tree_reps(
        self,
        reset_flags: Optional[torch.Tensor] = None,
        status: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Replicate compute_tree_reps from the Cython implementation."""
        self._update_step_status()
        self.max_rollout_depth_snapshot.copy_(self.max_rollout_depth)
        if reset_flags is None:
            reset_tensor = torch.ones(self.batch_size, dtype=torch.float32, device=self.device)
            reset_is_none = True
        else:
            if torch.is_tensor(reset_flags):
                reset_tensor = reset_flags.to(dtype=torch.float32, device=self.device).view(-1)
            else:
                reset_tensor = torch.tensor(reset_flags, dtype=torch.float32, device=self.device)
            reset_is_none = False

        if status is not None:
            if torch.is_tensor(status):
                status_tensor = status.to(dtype=torch.long, device=self.device).view(-1)
            else:
                status_tensor = torch.tensor(status, dtype=torch.long, device=self.device)
        else:
            status_tensor = None

        idx1 = self.num_actions * 5 + 6
        idx2 = idx1
        idx3 = idx2 + self.num_actions * 5 + 3
        idx4 = idx3
        idx5 = idx4 + 2 + self.rep_rec_t
        obs_n = idx5
        if self.has_action_seq:
            obs_n += self.max_depth * self.num_actions
            if self.reset_mode == 0:
                obs_n += self.num_actions

        reps = torch.zeros(self.batch_size, obs_n, dtype=torch.float32, device=self.device)

        for idx in range(self.batch_size):
            root = self.root_nodes[idx]
            current = self.cur_nodes[idx]
            reps[idx, :idx1] = node_stat(
                root,
                detailed=True,
                enc_type=self.enc_type,
                enc_f_type=self.enc_f_type,
                mask_type=self.mask_type,
                raw_num_actions=self.raw_num_actions,
            )
            reps[idx, idx2:idx3] = node_stat(
                current,
                detailed=False,
                enc_type=self.enc_type,
                enc_f_type=self.enc_f_type,
                mask_type=self.mask_type,
                raw_num_actions=self.raw_num_actions,
            )

            if reset_is_none or (status_tensor is not None and int(status_tensor[idx].item()) == 1):
                reps[idx, idx4] = 1.0
            else:
                reps[idx, idx4] = float(reset_tensor[idx].item())

            cur_t_int = int(self.cur_t[idx].item())
            if cur_t_int < self.rep_rec_t:
                reps[idx, idx4 + 1 + cur_t_int] = 1.0

            disc_depth = float(self.rollout_depth[idx].item())
            reps[idx, idx4 + self.rep_rec_t + 1] = float(self.discounting ** disc_depth)

            if self.has_action_seq:
                base = idx5
                node = current
                depth = int(self.rollout_depth[idx].item())
                # Trace back through parent nodes to record full action sequence
                # Mimicking cenv.pyx lines 454-458
                for j in range(depth + 1):
                    if node is None:
                        break
                    seq_idx = base + (depth - j) * self.num_actions + node.action
                    if seq_idx < reps.shape[1]:
                        reps[idx, seq_idx] = 1.0
                    node = node.parent

        return reps

    # Convenience helpers ------------------------------------------------

    def get_real_states(self) -> Optional[torch.Tensor]:
        states = []
        for node in self.root_nodes:
            encoded = node.encoded or {}
            state = encoded.get("real_states")
            if state is None:
                return None
            states.append(state)
        return torch.stack(states, dim=0)

    def get_current_encoded(self, key: str) -> Optional[torch.Tensor]:
        outputs = []
        for node in self.cur_nodes:
            encoded = node.encoded or {}
            value = encoded.get(key)
            if value is None:
                return None
            outputs.append(value)
        return torch.stack(outputs, dim=0)
