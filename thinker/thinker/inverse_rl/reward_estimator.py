"""Reward head used by the offline ML-IRL trainer."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import nn


class RewardEstimator(nn.Module):
    """Small MLP that predicts per-state rewards from Thinker features."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        residual: bool = False,
        clamp_magnitude: float = 10.0,
    ) -> None:
        super().__init__()
        dims = [input_dim, *hidden_dims]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            block = [nn.Linear(in_dim, out_dim), nn.ReLU()]
            layers.append(nn.Sequential(*block))
        self.backbone = nn.ModuleList(layers)
        self.output = nn.Linear(dims[-1], 1)
        self.residual = residual
        self.clamp_magnitude = float(clamp_magnitude)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features
        for layer in self.backbone:
            out = layer(x)
            x = out if not self.residual else x + out
        reward = self.output(x).squeeze(-1)
        if self.clamp_magnitude > 0:
            reward = torch.clamp(reward, -self.clamp_magnitude, self.clamp_magnitude)
        return reward

    def parameters_without_output(self) -> Iterable[nn.Parameter]:
        for layer in self.backbone:
            yield from layer.parameters()
