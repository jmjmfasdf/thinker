"""Dataset utilities for offline ML-IRL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from thinker.bc_loader import FrameStackedBehavioralDataLoader


@dataclass
class DemonstrationBatch:
    images: np.ndarray            # (B, T, C, H, W)
    actions: np.ndarray           # (B, T)
    prev_actions: np.ndarray      # (B, T)
    rewards: np.ndarray           # (B, T)
    is_first: np.ndarray          # (B, T)
    is_terminal: np.ndarray       # (B, T)


class ThinkerBehaviorDataset:
    """Thin wrapper around ``FrameStackedBehavioralDataLoader`` with return targets."""

    def __init__(
        self,
        loader: FrameStackedBehavioralDataLoader,
        sequence_length: int,
        gamma: float,
    ) -> None:
        self.loader = loader
        self.sequence_length = sequence_length
        self.gamma = float(gamma)

    def sample_batch(self, batch_size: int) -> Optional[DemonstrationBatch]:
        sample = self.loader.get_sequence_batch(
            batch_size=batch_size,
            sequence_length=self.sequence_length,
        )
        if sample is None:
            return None

        images = sample["images"]  # (B, T, C, H, W)
        actions = sample["actions"]
        rewards = sample["rewards"]
        is_terminal = sample["is_terminal"]
        is_first = sample["is_first"].astype(bool)
        prev_actions = sample.get("prev_actions")

        action_idx = self._action_to_indices(actions)
        if prev_actions is None:
            prev_action_idx = self._infer_prev_actions(action_idx, is_first)
        else:
            prev_action_idx = np.asarray(prev_actions, dtype=np.int64)

        return DemonstrationBatch(
            images=images.astype(np.float32),
            actions=action_idx.astype(np.int64),
            prev_actions=prev_action_idx.astype(np.int64),
            rewards=rewards.astype(np.float32),
            is_first=is_first,
            is_terminal=is_terminal.astype(bool),
        )

    @staticmethod
    def _action_to_indices(actions: np.ndarray) -> np.ndarray:
        if actions.ndim == 2:
            return actions
        if actions.ndim >= 3 and actions.shape[-1] > 1:
            flat = actions.reshape(*actions.shape[:-1], actions.shape[-1])
            return np.argmax(flat, axis=-1)
        raise ValueError(f"Unsupported action tensor shape {actions.shape}")

    @staticmethod
    def _infer_prev_actions(actions: np.ndarray, is_first: np.ndarray) -> np.ndarray:
        """Fallback for legacy data without explicit prev_action annotations."""
        prev = np.roll(actions, shift=1, axis=1)
        prev[:, 0] = actions[:, 0]
        seq_start_mask = np.zeros_like(is_first, dtype=bool)
        seq_start_mask[:, 0] = True
        prev[np.logical_or(is_first, seq_start_mask)] = actions[np.logical_or(is_first, seq_start_mask)]
        return prev
