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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Encoding helpers – copied from thinker/thinker/cenv.pyx
# ---------------------------------------------------------------------------

def _enc_mu_zero(x: torch.Tensor) -> torch.Tensor:
    """MuZero-style value encoding (enc_0 in Cython)."""
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1.0) - 1.0) + 0.001 * x


def _enc_log(x: torch.Tensor) -> torch.Tensor:
    """Dreamer-style log encoding (enc_1 in Cython)."""
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def _apply_encoding(x: torch.Tensor, enc_type: int, enc_f_type: int) -> torch.Tensor:
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

    def ensure_children(self, logits: torch.Tensor) -> None:
        if self.children and len(self.children) == self.num_actions:
            # Just refresh logits
            for child, logit in zip(self.children, logits.tolist()):
                child.logit = float(logit)
            return
        self.children = [
            Node(
                action=a,
                num_actions=self.num_actions,
                discounting=self.discounting,
                rec_t=self.rec_t,
                parent=self,
                logit=float(logits[a]),
                remember_path=self.remember_path,
            )
            for a in range(self.num_actions)
        ]


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


def node_expand(
    node: Node,
    reward: torch.Tensor,
    value: torch.Tensor,
    t: int,
    done: torch.Tensor,
    logits: torch.Tensor,
    encoded: Dict[str, Any],
    override: bool = False,
) -> None:
    """Expand or refresh node statistics using current model outputs."""
    reward_f = float(reward)
    value_f = float(value)
    done_f = bool(done)

    if override and not node.children:
        override = False  # match Cython behaviour

    if override and node.rollout_qs:
        # Refresh stored rollout returns when overriding
        base_q = reward_f + value_f * node.discounting
        delta_r = reward_f - node.reward
        delta_v = value_f - node.value
        node.rollout_qs[0] = base_q
        for idx in range(1, len(node.rollout_qs)):
            node.rollout_qs[idx] = node.rollout_qs[idx] - node.reward + reward_f
        node.max_q = max(node.max_q, base_q)
        if node.parent and node.remember_path:
            node_refresh(node.parent, node, delta_r, delta_v, node.discounting, depth=1)

    node.reward = reward_f
    node.value = value_f
    node.time_step = t
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
            stored_path = list(path or [])
            stored_path.append(node)
            node.paths.append(stored_path)
        node.rollout_qs.append(node.rollout_q)
    if node.parent is not None:
        if path is not None:
            path.append(node)
        node_propagate(node.parent, reward, value, new_rollout, path)


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
    num_actions = node.num_actions if raw_num_actions is None else raw_num_actions
    obs_n = num_actions * 5 + 3
    if detailed:
        obs_n += 3

    result = torch.zeros(obs_n, dtype=torch.float32)
    result[node.action] = 1.0

    reward = torch.tensor([node.reward], dtype=torch.float32)
    value = torch.tensor([node.value], dtype=torch.float32)
    enc_reward = _apply_encoding(reward, enc_type, enc_f_type)[0]
    enc_value = _apply_encoding(value, enc_type, enc_f_type)[0]

    result[num_actions] = enc_reward
    result[num_actions + 1] = float(node.done)
    if mask_type not in (2,):
        result[num_actions + 2] = enc_value

    for idx, child in enumerate(node.children):
        base = num_actions + 3
        if mask_type not in (2,):
            result[base + idx] = child.logit
        if mask_type in (1, 2):
            continue
        qs = child.rollout_qs if child.rollout_qs else [child.rollout_q]
        mean_q = _apply_encoding(torch.tensor([_safe_average(qs)]), enc_type, enc_f_type)[0]
        max_q = _apply_encoding(torch.tensor([_safe_max(qs)]), enc_type, enc_f_type)[0]
        result[num_actions * 2 + 3 + idx] = mean_q
        if mask_type not in (3, 4):
            result[num_actions * 3 + 3 + idx] = max_q
        visits = child.rollout_n / max(1.0, float(node.rec_t))
        result[num_actions * 4 + 3 + idx] = visits

    base_idx = num_actions * 5 + 3
    if detailed and mask_type not in (1, 2, 4):
        trail_r = _apply_encoding(torch.tensor([(node.trail_r - node.reward) / node.discounting]), enc_type, enc_f_type)[0]
        rollout_q = _apply_encoding(torch.tensor([(node.rollout_q - node.reward) / node.discounting]), enc_type, enc_f_type)[0]
        max_q = _apply_encoding(torch.tensor([node.max_q]), enc_type, enc_f_type)[0]
        result[base_idx] = trail_r
        result[base_idx + 1] = rollout_q
        result[base_idx + 2] = max_q

    return result


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
        self.rec_t = int(getattr(flags, "rec_t", 40))
        self.has_action_seq = bool(getattr(flags, "has_action_seq", False))
        self.max_depth = int(getattr(flags, "max_depth", 40))
        self.reset_mode = int(getattr(flags, "reset_mode", 0))
        self.enc_type = int(getattr(flags, "critic_enc_type", 0))
        self.enc_f_type = int(getattr(flags, "critic_enc_f_type", 0))
        self.mask_type = int(getattr(flags, "stat_mask_type", 0))

        self.remember_path = False  # can be toggled if needed
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
        for root in self.root_nodes:
            root.ensure_children(torch.zeros(self.num_actions))
        self.cur_nodes: List[Node] = list(self.root_nodes)
        self.rollout_depth = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.cur_t = torch.zeros(batch_size, dtype=torch.long, device=device)

    # ------------------------------------------------------------------
    # Tree mutation APIs
    # ------------------------------------------------------------------

    def expand_root(
        self,
        rewards: Optional[torch.Tensor],
        values: torch.Tensor,
        logits: torch.Tensor,
        encoded: Sequence[Dict[str, Any]],
    ) -> None:
        for idx, root in enumerate(self.root_nodes):
            r = rewards[idx] if rewards is not None else torch.tensor(0.0, device=values.device)
            v = values[idx]
            logit_vec = logits[idx]
            node_expand(root, r, v, t=0, done=torch.tensor(False), logits=logit_vec, encoded=encoded[idx], override=False)
            node_visit(root)
            root.max_q = max(root.max_q, root.rollout_q)

    def expand_current(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        logits: torch.Tensor,
        encoded: Sequence[Dict[str, Any]],
        override: bool = False,
    ) -> None:
        for idx in range(self.batch_size):
            node = self.cur_nodes[idx]
            node_expand(
                node,
                rewards[idx],
                values[idx],
                t=int(self.cur_t[idx].item()),
                done=dones[idx],
                logits=logits[idx],
                encoded=encoded[idx],
                override=override,
            )
            node_visit(node)
            node.max_q = max(node.max_q, node.rollout_q)

    def advance(self, actions: torch.Tensor, resets: Optional[torch.Tensor] = None) -> None:
        resets = resets if resets is not None else torch.zeros_like(actions, dtype=torch.bool)
        new_cur_nodes: List[Node] = []
        for idx in range(self.batch_size):
            if bool(resets[idx]):
                new_cur_nodes.append(self.root_nodes[idx])
                self.rollout_depth[idx] = 0
                continue
            action = int(actions[idx].item())
            current = self.cur_nodes[idx]
            if not current.children:
                current.ensure_children(torch.zeros(self.num_actions))
            new_cur_nodes.append(current.children[action])
            self.rollout_depth[idx] += 1
            if self.max_depth > 0:
                self.rollout_depth[idx] = torch.clamp(self.rollout_depth[idx], max=self.max_depth)
        self.cur_nodes = new_cur_nodes
        self.cur_t += 1

    def reset_real_step(self) -> None:
        self.cur_t.zero_()
        self.rollout_depth.zero_()
        self.cur_nodes = [self.root_nodes[idx] for idx in range(self.batch_size)]

    # ------------------------------------------------------------------
    # Tree representation utilities
    # ------------------------------------------------------------------

    def compute_tree_reps(self, reset_flags: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Replicate compute_tree_reps from the Cython implementation."""
        reset_flags = reset_flags if reset_flags is not None else torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)

        idx1 = self.num_actions * 5 + 6
        idx2 = idx1
        idx3 = idx2 + self.num_actions * 5 + 3
        idx4 = idx3
        idx5 = idx4 + 2 + self.rec_t
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

            if reset_flags[idx] or self.cur_t[idx].item() == 0:
                reps[idx, idx4] = 1.0
            else:
                reps[idx, idx4] = 0.0

            cur_t_int = int(self.cur_t[idx].item())
            if cur_t_int < self.rec_t:
                reps[idx, idx4 + 1 + cur_t_int] = 1.0

            reps[idx, idx4 + self.rec_t + 1] = float(self.discounting ** float(self.rollout_depth[idx].item()))

            if self.has_action_seq:
                base = idx5
                depth = int(self.rollout_depth[idx].item())
                if depth >= 0:
                    action = current.action
                    seq_idx = base + depth * self.num_actions + action
                    if seq_idx < reps.shape[1]:
                        reps[idx, seq_idx] = 1.0

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
