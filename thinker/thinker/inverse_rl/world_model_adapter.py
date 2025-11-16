"""Penalty utilities derived from Thinker's world-model hints."""

from __future__ import annotations

import torch


class WorldModelAdapter:
    """Approximates Offline ML-IRL's conservative penalty using Thinker tree hints."""

    def __init__(
        self,
        gamma: float,
        penalty_scale: float = 1.0,
        penalty_clamp: float = 5.0,
        eps: float = 1e-6,
    ) -> None:
        self.gamma = float(gamma)
        self.penalty_scale = float(max(penalty_scale, 0.0))
        self.penalty_clamp = float(max(penalty_clamp, 0.0))
        self.eps = float(eps)

    def compute_penalty(
        self,
        tree_q: torch.Tensor | None,
        expert_actions: torch.Tensor,
        target_returns: torch.Tensor,
    ) -> torch.Tensor:
        if tree_q is None or self.penalty_scale == 0.0:
            return torch.zeros_like(target_returns)
        if tree_q.dim() == 3:
            tree_q = tree_q[0]
        q_values = tree_q.gather(-1, expert_actions.view(-1, 1)).squeeze(-1)
        penalty = (q_values - target_returns).abs()
        if self.penalty_clamp > 0:
            penalty = penalty.clamp(max=self.penalty_clamp)
        return penalty * self.penalty_scale
