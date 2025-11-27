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
        # Action space: single Discrete (env_n=1) to match Thinker checkpoints expecting Discrete
        self.action_space = spaces.Discrete(self.num_actions)
        self.reward_range = (-np.inf, np.inf)
        self.metadata: Dict[str, Any] = {}

        # Precompute logical step indices and episode boundaries
        self._logical_indices, self._segment_reward_range, self._segment_actions_idx, self._episode_start_flags = self._prepare_logical_steps()
        self._num_steps = len(self._logical_indices)
        self._pos = 0
        self._pos_stack: List[int] = []

    # ---------------------- Public helpers ----------------------
    def has_more(self) -> bool:
        return self._pos < self._num_steps

    def current_human_action(self) -> int:
        if self._pos <= 0:
            idx = 0
        else:
            idx = min(self._pos, self._num_steps - 1)
        return int(self._segment_actions_idx[idx])

    def close(self):
        # compatibility with gym VectorEnv interface
        return None

    # ---------------------- Vector env API ----------------------
    def reset(self, *, reset_stat: bool = False, seed: Optional[int] = None):
        # Reset to the beginning if past end
        if self._pos >= self._num_steps:
            self._pos = 0
        obs = self._build_stack(self._logical_indices[self._pos])
        info = {
            "real_done": False,
            "episode_return": 0.0,
            "episode_step": 0,
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
            reward = 0.0
            done = True
            truncated = True
            info = {"real_done": True}
            return obs, reward, done, truncated, info

        idx = self._logical_indices[self._pos]
        obs = self._build_stack(idx)

        # Compute reward as sum over the segment that produced this obs
        seg_lo, seg_hi = self._segment_reward_range[self._pos]
        reward_val = float(np.sum(self._rewards[seg_lo:seg_hi]))

        # Determine termination at this step
        # Done if this logical index is terminal, or next step starts a new episode
        is_term = bool(self._is_terminal[idx])
        done = bool(is_term)
        truncated = False

        if is_term:
            # If terminal, jump to next episode's first logical step (if any)
            next_pos = self._find_next_episode_start(self._pos)
            if next_pos is not None:
                self._pos = next_pos
                obs = self._build_stack(self._logical_indices[self._pos])
        
        info = {"real_done": bool(done)}
        return np.expand_dims(obs, axis=0), reward_val, done, truncated, info

    # ---------------------- Save/Load for imagination ----------------------
    def quick_save(self, env_id: Optional[List[int]] = None):
        # Save current cursor so rollout can restore later
        self._pos_stack.append(self._pos)

    def quick_load(self, env_id: Optional[List[int]] = None):
        if not self._pos_stack:
            raise ValueError("No saved state found. Call quick_save() before quick_load().")
        self._pos = self._pos_stack.pop()

    def save_ckp(self):
        # Return lightweight state for checkpointing
        return {"behavior_pos": np.array([self._pos], dtype=np.int64)}

    def load_ckp(self, data):
        # Restore cursor if present
        if "behavior_pos" in data:
            pos = int(np.array(data["behavior_pos"]).reshape(-1)[0])
            self._pos = max(0, min(pos, self._num_steps - 1))

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
        """Generate logical steps with non-overlapping stacks (stride=frame_stack_n)."""
        logical_indices: List[int] = []
        reward_ranges: List[Tuple[int, int]] = []
        action_indices: List[int] = []
        episode_starts: List[bool] = []

        # Derive action indices (supports one-hot or index)
        if self._raw_actions.ndim == 2:
            human_actions = np.argmax(self._raw_actions, axis=1).astype(np.int64)
        else:
            human_actions = self._raw_actions.astype(np.int64)

        ep_start_ptr = 0
        frames_since_reset = 0
        for idx in range(len(self._raw_images)):
            new_ep = idx == 0 or bool(self._is_first[idx])
            if new_ep:
                ep_start_ptr = idx
                frames_since_reset = 1
            else:
                frames_since_reset += 1

            # emit only every frame_stack_n frames (non-overlapping)
            if frames_since_reset % self.frame_stack_n == 0:
                start_idx = max(ep_start_ptr, idx - self.frame_stack_n + 1)
                logical_indices.append(idx)
                reward_ranges.append((start_idx, idx + 1))
                # action aligned with the latest frame in the block
                action_indices.append(int(human_actions[idx]))
                episode_starts.append(new_ep or len(logical_indices) == 1)

        return logical_indices, reward_ranges, action_indices, episode_starts

    def _find_next_episode_start(self, pos: int) -> Optional[int]:
        # Find the next logical index that begins a new episode
        for i in range(pos + 1, self._num_steps):
            if self._episode_start_flags[i]:
                return i
        return None


class BehaviorSequenceVectorEnv:
    """Vector env over a fixed-length batch of stacked observations.

    - Supports env_n = batch_size
    - Steps through a provided length-L sequence, emits rewards/dones at the last step,
      and automatically resets finished envs to the start (like EnvPoolWrap).
    - Implements quick_save/quick_load so cModelWrapper can branch/reset during planning.
    """

    def __init__(
        self,
        obs_seq: np.ndarray,  # (B, L, C, H, W) stacked observations
        *,
        rewards_seq: Optional[np.ndarray] = None,  # (B, L)
        actions_seq: Optional[np.ndarray] = None,  # (B, L) for bookkeeping only
        num_actions: int = 1,
    ) -> None:
        assert obs_seq.ndim == 5, f"obs_seq must be (B,L,C,H,W), got {obs_seq.shape}"
        self.obs_seq = obs_seq.astype(np.uint8 if obs_seq.dtype == np.uint8 else np.float32, copy=True)
        self.rewards_seq = rewards_seq.astype(np.float32, copy=True) if rewards_seq is not None else np.zeros(
            (obs_seq.shape[0], obs_seq.shape[1]), dtype=np.float32
        )
        self.actions_seq = actions_seq.astype(np.int64, copy=True) if actions_seq is not None else None
        self.env_n, self.seq_len = obs_seq.shape[:2]
        self._pos = np.zeros(self.env_n, dtype=np.int64)
        self._pos_stack: List[np.ndarray] = []
        obs_low = 0 if self.obs_seq.dtype == np.uint8 else -np.inf
        obs_high = 255 if self.obs_seq.dtype == np.uint8 else np.inf
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            shape=self.obs_seq.shape[2:],
            dtype=self.obs_seq.dtype,
        )
        self.action_space = spaces.Tuple((spaces.Discrete(int(num_actions)),) * self.env_n)
        self.reward_range = (-np.inf, np.inf)
        self.metadata: Dict[str, Any] = {}

    # ------------- update helpers for planner reuse -------------
    def update_sequences(
        self,
        obs_seq: np.ndarray,
        rewards_seq: Optional[np.ndarray] = None,
        actions_seq: Optional[np.ndarray] = None,
    ) -> None:
        """Update stored sequences in-place while keeping the same env instance.

        This is used by IcoPro BC to reuse a single planner and avoid repeated
        GPU allocations when building cModelWrapper.
        """
        assert obs_seq.ndim == 5, f"obs_seq must be (B,L,C,H,W), got {obs_seq.shape}"
        if obs_seq.shape != self.obs_seq.shape:
            raise ValueError(
                f"Shape mismatch in update_sequences: {obs_seq.shape} vs {self.obs_seq.shape}"
            )

        # Replace backing arrays
        self.obs_seq = obs_seq.astype(
            np.uint8 if obs_seq.dtype == np.uint8 else np.float32, copy=True
        )
        if rewards_seq is not None:
            assert rewards_seq.shape == self.rewards_seq.shape, (
                f"rewards_seq shape {rewards_seq.shape} does not match "
                f"{self.rewards_seq.shape}"
            )
            self.rewards_seq = rewards_seq.astype(np.float32, copy=True)
        if actions_seq is not None:
            assert actions_seq.shape == self.actions_seq.shape, (
                f"actions_seq shape {actions_seq.shape} does not match "
                f"{self.actions_seq.shape}"
            )
            self.actions_seq = actions_seq.astype(np.int64, copy=True)

        # Reset cursors / stacks
        self._pos[...] = 0
        self._pos_stack.clear()

    # ------------- helpers -------------
    def _current_obs(self, env_id: np.ndarray) -> np.ndarray:
        return self.obs_seq[env_id, self._pos[env_id]]

    def _current_reward(self, env_id: np.ndarray) -> np.ndarray:
        return self.rewards_seq[env_id, self._pos[env_id]]

    # ------------- gym-like API -------------
    def reset(self, *, reset_stat: bool = False, seed: Optional[int] = None):
        if reset_stat:
            self._pos[...] = 0
        obs = self._current_obs(np.arange(self.env_n))
        info = {
            "real_done": np.zeros(self.env_n, dtype=bool),
            "episode_return": np.zeros(self.env_n, dtype=np.float32),
            "episode_step": np.zeros(self.env_n, dtype=np.int64),
        }
        return obs, info

    def step(self, action, *, env_id: Optional[List[int]] = None):
        if env_id is None:
            env_id = np.arange(self.env_n, dtype=np.int64)
        else:
            env_id = np.asarray(env_id, dtype=np.int64)

        # advance positions
        self._pos[env_id] = np.minimum(self._pos[env_id] + 1, self.seq_len - 1)

        obs = self._current_obs(env_id)
        reward = self._current_reward(env_id)
        done = self._pos[env_id] == (self.seq_len - 1)
        truncated = np.zeros_like(done, dtype=bool)

        # auto-reset finished envs to start for subsequent steps
        finished = np.where(done)[0]
        if finished.size > 0:
            self._pos[env_id[finished]] = 0

        info = {"real_done": done.copy()}
        return obs, reward, done, truncated, info

    # ------------- planning compatibility -------------
    def quick_save(self, env_id=None):
        self._pos_stack.append(self._pos.copy())

    def quick_load(self, env_id=None):
        if not self._pos_stack:
            raise ValueError("No saved state found. Call quick_save() before quick_load().")
        self._pos = self._pos_stack.pop()

    def save_ckp(self):
        return {"sequence_pos": self._pos.copy()}

    def load_ckp(self, data):
        if data is None:
            return
        pos = data.get("sequence_pos")
        if pos is None:
            return
        pos = np.asarray(pos, dtype=np.int64)
        if pos.shape != self._pos.shape:
            raise ValueError(f"Checkpoint position shape {pos.shape} does not match env_n {self.env_n}")
        self._pos = pos

class BehaviorBatchEnv:
    """Vector env that serves a fixed batch of observations for offline training.

    - reset(reset_stat=True) returns the provided batch observations.
    - step(action, env_id=...) returns the same observations, zero rewards,
      and all dones=False. Agent actions are ignored.

    It now implements quick_save/quick_load to be compatible with Thinker
    planning which expects env-level save & restore during rollouts.
    """
    def __init__(self, obs_batch: np.ndarray, num_actions: int):
        assert obs_batch.ndim == 4, f"obs_batch must be (B,C,H,W), got {obs_batch.shape}"
        self._update_internal_obs(obs_batch)
        self._obs_stack: List[np.ndarray] = []
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

    # -------- Planning compatibility helpers --------
    def quick_save(self, env_id=None):
        """Save current observation batch; env_id is unused but kept for API parity."""
        self._obs_stack.append(np.copy(self._obs))

    def quick_load(self, env_id=None):
        if not self._obs_stack:
            raise ValueError("No saved state found. Call quick_save() before quick_load().")
        self._obs = self._obs_stack.pop()

    def save_ckp(self):
        return {"behavior_obs": np.copy(self._obs)}

    def load_ckp(self, data):
        if data is None:
            return
        obs = data.get("behavior_obs")
        if obs is None:
            return
        obs = np.asarray(obs)
        if obs.shape[0] != self.env_n:
            raise ValueError(f"Checkpoint batch size {obs.shape[0]} does not match env_n {self.env_n}")
        self._update_internal_obs(obs)
