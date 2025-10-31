from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces
import cv2


class BehaviorDatasetVectorEnv:
    """Minimal vectorized env (env_n=1) backed by behavioral data.

    It exposes a Thinker-compatible vector env API so cenv.cModelWrapper can
    drive imagination and real steps using pre-recorded observations.

    Observations are frame-stacked and resized on the fly.
    Real steps advance through precomputed logical indices (every
    `frame_stack_n` frames within an episode). When an episode ends, the env
    immediately resets and returns the first observation of the next episode in
    the same step, mirroring EnvPoolWrap behavior.
    """

    def __init__(
        self,
        *,
        images: np.ndarray,            # (T, H, W, C)
        actions: np.ndarray,           # (T, K) one-hot or (T,) indices
        rewards: np.ndarray,           # (T,)
        is_first: np.ndarray,          # (T,)
        is_terminal: np.ndarray,       # (T,)
        frame_stack_n: int,
        target_size: Tuple[int, int],  # (H, W)
        grayscale: bool,
        num_actions: int,
    ) -> None:
        assert images.ndim == 4, f"images should be (T, H, W, C), not {images.shape}"
        self._raw_images = images
        self._raw_actions = actions
        self._rewards = rewards.astype(np.float32)
        self._is_first = is_first.astype(bool)
        self._is_terminal = is_terminal.astype(bool)
        self.frame_stack_n = int(frame_stack_n)
        self.target_h, self.target_w = int(target_size[0]), int(target_size[1])
        self.grayscale = bool(grayscale)
        self.num_actions = int(num_actions)

        # Derived channels and spaces
        raw_ch = images.shape[-1]
        ch_per_frame = 1 if self.grayscale else raw_ch
        self._stack_ch = ch_per_frame * self.frame_stack_n
        self.env_n = 1

        # Observation space: Box of uint8 with CHW stacking
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self._stack_ch, self.target_h, self.target_w),
            dtype=np.uint8,
        )
        # Vector action space: Tuple of length env_n
        self.action_space = spaces.Tuple((spaces.Discrete(self.num_actions),) * self.env_n)
        self.reward_range = (-np.inf, np.inf)
        self.metadata: Dict[str, Any] = {}

        # Precompute logical step indices and episode boundaries
        self._logical_indices, self._segment_reward_range, self._segment_actions_idx, self._episode_start_flags = self._prepare_logical_steps()
        self._num_steps = len(self._logical_indices)
        self._pos = 0

    # ---------------------- Public helpers ----------------------
    def has_more(self) -> bool:
        return self._pos < self._num_steps

    def current_human_action(self) -> int:
        if self._pos <= 0:
            idx = 0
        else:
            idx = min(self._pos, self._num_steps - 1)
        return int(self._segment_actions_idx[idx])

    # ---------------------- Vector env API ----------------------
    def reset(self, *, reset_stat: bool = False, seed: Optional[int] = None):
        # Reset to the beginning if past end
        if self._pos >= self._num_steps:
            self._pos = 0
        obs = self._build_stack(self._logical_indices[self._pos])
        info = {
            "real_done": np.array([False], dtype=bool),
            "episode_return": np.array([0.0], dtype=np.float32),
            "episode_step": np.array([0], dtype=np.int64),
        }
        return np.expand_dims(obs, axis=0), info

    def step(self, action, *, env_id: Optional[List[int]] = None):
        # We ignore the action for dataset playback.
        if env_id is None:
            env_id = [0]

        # Advance one logical step
        self._pos += 1
        if self._pos >= self._num_steps:
            # End of dataset: emit zeros and mark done
            obs = np.zeros((1,) + self.observation_space.shape, dtype=np.uint8)
            reward = np.array([0.0], dtype=np.float32)
            done = np.array([True], dtype=bool)
            truncated = np.array([True], dtype=bool)
            info = {"real_done": done.copy()}
            return obs, reward, done, truncated, info

        idx = self._logical_indices[self._pos]
        obs = self._build_stack(idx)

        # Compute reward as sum over the segment that produced this obs
        seg_lo, seg_hi = self._segment_reward_range[self._pos]
        reward_val = float(np.sum(self._rewards[seg_lo:seg_hi]))

        # Determine termination at this step
        # Done if this logical index is terminal, or next step starts a new episode
        is_term = bool(self._is_terminal[idx])
        done = np.array([is_term], dtype=bool)
        truncated = np.array([False], dtype=bool)

        if is_term:
            # If terminal, jump to next episode's first logical step (if any)
            next_pos = self._find_next_episode_start(self._pos)
            if next_pos is not None:
                self._pos = next_pos
                obs = self._build_stack(self._logical_indices[self._pos])
        
        info = {"real_done": done.copy()}
        return np.expand_dims(obs, axis=0), np.array([reward_val], dtype=np.float32), done, truncated, info

    # ---------------------- Internal helpers ----------------------
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if (h, w) != (self.target_h, self.target_w):
            frame = cv2.resize(frame, (self.target_w, self.target_h), interpolation=cv2.INTER_AREA)
        if self.grayscale:
            if frame.shape[-1] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = frame[..., np.newaxis]
        else:
            if frame.shape[-1] == 1:
                frame = np.repeat(frame, 3, axis=-1)
            elif frame.shape[-1] == 4:
                frame = frame[..., :3]
        frame = np.transpose(frame, (2, 0, 1))  # CHW
        return frame.astype(np.uint8)

    def _build_stack(self, index: int) -> np.ndarray:
        # Build CHW stacked observation ending at frame index `index`.
        frames: List[np.ndarray] = []
        tgt_ep = self._episode_id(index)
        for offset in range(self.frame_stack_n - 1, -1, -1):
            src = index - offset
            valid = src >= 0 and self._episode_id(src) == tgt_ep
            if not valid:
                ch = 1 if self.grayscale else self._raw_images.shape[-1]
                frames.append(np.zeros((ch, self.target_h, self.target_w), dtype=np.uint8))
            else:
                frames.append(self._preprocess_frame(self._raw_images[src]))
        return np.concatenate(frames, axis=0)

    def _episode_id(self, idx: int) -> int:
        # Assign episode ids from is_first flags
        # Start at 0; increment when is_first True
        # Note: robust even if idx==0 and is_first[0]==False
        count = -1
        for i in range(0, idx + 1):
            if i == 0 or bool(self._is_first[i]):
                count += 1
        return max(0, count)

    def _prepare_logical_steps(self):
        logical_indices: List[int] = []
        reward_ranges: List[Tuple[int, int]] = []
        action_indices: List[int] = []
        episode_starts: List[bool] = []

        # Derive valid human action indices if one-hot
        if self._raw_actions.ndim == 2:
            human_actions = np.argmax(self._raw_actions, axis=1).astype(np.int64)
            valid_mask = self._raw_actions.sum(axis=1) > 0.0
        else:
            human_actions = self._raw_actions.astype(np.int64)
            valid_mask = human_actions >= 0

        frames_since_reset = 0
        ep_start_ptr = -1
        for idx in range(len(self._raw_images)):
            new_ep = idx == 0 or bool(self._is_first[idx])
            if new_ep:
                frames_since_reset = 1
                ep_start_ptr = idx
            else:
                frames_since_reset += 1

            if not valid_mask[idx]:
                continue
            if frames_since_reset >= self.frame_stack_n and frames_since_reset % self.frame_stack_n == 0:
                prev_idx = idx - self.frame_stack_n
                logical_indices.append(idx)
                reward_ranges.append((max(ep_start_ptr, idx - self.frame_stack_n + 1), idx + 1))
                action_indices.append(int(human_actions[prev_idx if prev_idx >= 0 else idx]))
                episode_starts.append(new_ep or len(logical_indices) == 1)

        return logical_indices, reward_ranges, action_indices, episode_starts

    def _find_next_episode_start(self, pos: int) -> Optional[int]:
        # Find the next logical index that begins a new episode
        for i in range(pos + 1, self._num_steps):
            if self._episode_start_flags[i]:
                return i
        return None


class BehaviorBatchEnv:
    """Vector env that serves a fixed batch of observations for offline training.

    - reset(reset_stat=True) returns the provided batch observations.
    - step(action, env_id=...) returns the same observations, zero rewards,
      and all dones=False. Agent actions are ignored.
    """
    def __init__(self, obs_batch: np.ndarray, num_actions: int):
        assert obs_batch.ndim == 4, f"obs_batch must be (B,C,H,W), got {obs_batch.shape}"
        self._update_internal_obs(obs_batch)
        self.env_n = obs_batch.shape[0]
        self.observation_space = spaces.Box(
            low=0,
            high=255 if self._obs.dtype == np.uint8 else np.inf,
            shape=self._obs.shape[1:],
            dtype=self._obs.dtype,
        )
        self.action_space = spaces.Tuple((spaces.Discrete(int(num_actions)),) * self.env_n)
        self.reward_range = (-np.inf, np.inf)
        self.metadata = {}

    def _update_internal_obs(self, obs_batch: np.ndarray) -> None:
        dtype = np.uint8 if obs_batch.dtype == np.uint8 else np.float32
        self._obs = obs_batch.astype(dtype, copy=True)

    def update_batch(self, obs_batch: np.ndarray) -> None:
        """Replace the stored batch while keeping the same env instance."""
        assert obs_batch.ndim == 4, f"obs_batch must be (B,C,H,W), got {obs_batch.shape}"
        if obs_batch.shape[0] != self.env_n:
            raise ValueError(f"Batch size changed from {self.env_n} to {obs_batch.shape[0]}; rebuild env.")
        self._update_internal_obs(obs_batch)

    def reset(self, *, reset_stat: bool = False, seed=None):
        info = {
            "real_done": np.zeros(self.env_n, dtype=bool),
            "episode_return": np.zeros(self.env_n, dtype=np.float32),
            "episode_step": np.zeros(self.env_n, dtype=np.int64),
        }
        return np.copy(self._obs), info

    def step(self, action, *, env_id=None):
        if env_id is None:
            env_id = list(range(self.env_n))
        obs = np.copy(self._obs[env_id])
        reward = np.zeros(len(env_id), dtype=np.float32)
        done = np.zeros(len(env_id), dtype=bool)
        truncated = np.zeros(len(env_id), dtype=bool)
        info = {"real_done": np.copy(done)}
        return obs, reward, done, truncated, info
