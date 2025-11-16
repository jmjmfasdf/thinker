"""Policy improvement wrapper that reuses Thinker's actor network."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from imitation import PolicyBatch
from thinker.actor_net import ActorNet


class OfflinePolicyUpdater:
    def __init__(
        self,
        actor_net: ActorNet,
        optimizer: torch.optim.Optimizer,
        entropy_coef: float = 0.01,
        max_grad_norm: Optional[float] = None,
    ) -> None:
        self.actor_net = actor_net
        self.optimizer = optimizer
        self.entropy_coef = float(entropy_coef)
        self.max_grad_norm = max_grad_norm
        self.step = 0

    def update(
        self,
        batch: PolicyBatch,
        expert_actions: torch.Tensor,
        advantages: torch.Tensor,
    ) -> Dict[str, float]:
        log_probs = batch.log_probs.gather(-1, expert_actions.view(-1, 1)).squeeze(-1)
        entropy = -(batch.probs * batch.log_probs).sum(dim=-1)
        weights = advantages.detach()
        loss = -(weights * log_probs).mean() - self.entropy_coef * entropy.mean()

        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.step += 1
        with torch.no_grad():
            preds = torch.argmax(batch.logits, dim=-1)
            accuracy = (preds == expert_actions).float().mean().item()
        return {
            "loss": float(loss.detach().cpu().item()),
            "entropy": float(entropy.mean().detach().cpu().item()),
            "accuracy": accuracy,
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "optimizer": self.optimizer.state_dict(),
            "step": self.step,
        }
