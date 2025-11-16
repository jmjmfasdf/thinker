"""Shared helpers for extracting Thinker features for reward learning."""

from __future__ import annotations

from typing import Optional

import torch


def extract_features(policy_adapter, policy_batch, source: str | None) -> torch.Tensor:
    """Return the feature tensor requested by ``source`` ('sr', 'vp', or default)."""
    source = (source or "").lower()
    if source == "sr":
        feat = getattr(policy_adapter, "last_sr_features", None)
        if feat is not None:
            return feat
    elif source == "vp":
        feat = getattr(policy_adapter, "last_vp_features", None)
        if feat is not None:
            return feat
    return policy_batch.features


class FeatureScaler:
    """Running mean/variance normalizer for high-dimensional Thinker features."""

    def __init__(self, momentum: float = 0.01, eps: float = 1e-5) -> None:
        self.momentum = float(momentum)
        self.eps = float(eps)
        self._mean: Optional[torch.Tensor] = None
        self._var: Optional[torch.Tensor] = None

    def _to_device(self, ref: torch.Tensor) -> None:
        if self._mean is not None and self._mean.device != ref.device:
            self._mean = self._mean.to(ref.device)
            self._var = self._var.to(ref.device)

    @torch.no_grad()
    def update(self, batch: torch.Tensor) -> None:
        if batch.dim() == 1:
            batch = batch.unsqueeze(0)
        self._to_device(batch)
        mean = batch.mean(dim=0)
        var = batch.var(dim=0, unbiased=False)
        if self._mean is None:
            self._mean = mean
            self._var = var
        else:
            mom = self.momentum
            self._mean = (1 - mom) * self._mean + mom * mean
            self._var = (1 - mom) * self._var + mom * var

    def normalize(self, batch: torch.Tensor) -> torch.Tensor:
        if self._mean is None or self._var is None:
            return batch
        self._to_device(batch)
        return (batch - self._mean) / torch.sqrt(self._var + self.eps)

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        self.update(batch.detach())
        return self.normalize(batch)
