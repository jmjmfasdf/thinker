"""Teacher-forced vector environment for fixed behavioral sequences."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from gymnasium import spaces
import numpy as np


def _as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


class BehaviorSequenceVectorEnv:
    """Replay a batch of human transitions without auto-reset or wrapping.

    ``obs_seq[:, 0]`` is the burn-in root.  At cursor ``j``, ``step`` accepts
    only ``actions_seq[:, j]`` and returns ``obs_seq[:, j + 1]`` together with
    reward/done/truncated for that same edge.  This validation makes it
    impossible for a Thinker proposal to silently replace the teacher-forced
    human action.
    """

    def __init__(
        self,
        obs_seq: np.ndarray,
        *,
        actions_seq: np.ndarray,
        rewards_seq: Optional[np.ndarray] = None,
        done_seq: Optional[np.ndarray] = None,
        truncated_seq: Optional[np.ndarray] = None,
        initial_prev_action: Optional[np.ndarray] = None,
        score_mask: Optional[np.ndarray] = None,
        num_actions: int = 1,
        observation_space: Optional[spaces.Box] = None,
    ) -> None:
        self.num_actions = int(num_actions)
        if self.num_actions < 1:
            raise ValueError("num_actions must be positive")
        self._saved_positions: List[np.ndarray] = []
        self._checkpoint_slots: Dict[int, np.ndarray] = {}
        self._install_sequences(
            obs_seq=obs_seq,
            actions_seq=actions_seq,
            rewards_seq=rewards_seq,
            done_seq=done_seq,
            truncated_seq=truncated_seq,
            initial_prev_action=initial_prev_action,
            score_mask=score_mask,
            observation_space=observation_space,
            initialise=True,
        )

    def _install_sequences(
        self,
        *,
        obs_seq: np.ndarray,
        actions_seq: np.ndarray,
        rewards_seq: Optional[np.ndarray],
        done_seq: Optional[np.ndarray],
        truncated_seq: Optional[np.ndarray],
        initial_prev_action: Optional[np.ndarray],
        score_mask: Optional[np.ndarray],
        observation_space: Optional[spaces.Box],
        initialise: bool,
    ) -> None:
        observations = np.asarray(obs_seq)
        actions = np.asarray(actions_seq, dtype=np.int64)
        if observations.ndim != 5:
            raise ValueError(
                f"obs_seq must be [B,L+2,C,H,W], got {observations.shape}"
            )
        if actions.ndim != 2:
            raise ValueError(f"actions_seq must be [B,L+1], got {actions.shape}")
        batch_n, obs_n = observations.shape[:2]
        edge_n = obs_n - 1
        if actions.shape != (batch_n, edge_n):
            raise ValueError(
                f"actions_seq shape {actions.shape} must equal {(batch_n, edge_n)}"
            )
        if np.any(actions < 0) or np.any(actions >= self.num_actions):
            raise ValueError(f"actions_seq contains values outside [0,{self.num_actions - 1}]")
        if not initialise and (
            batch_n != self.env_n or observations.shape[2:] != self.obs_seq.shape[2:]
        ):
            raise ValueError(
                "update_sequences cannot change env_n or the observation shape; "
                "rebuild the planner environment instead"
            )

        def edge_array(value, dtype, default, name):
            if value is None:
                return np.full((batch_n, edge_n), default, dtype=dtype)
            array = np.asarray(value, dtype=dtype)
            if array.shape != (batch_n, edge_n):
                raise ValueError(
                    f"{name} shape {array.shape} must equal {(batch_n, edge_n)}"
                )
            return array.copy()

        rewards = edge_array(rewards_seq, np.float32, 0.0, "rewards_seq")
        done = edge_array(done_seq, np.bool_, False, "done_seq")
        truncated = edge_array(
            truncated_seq, np.bool_, False, "truncated_seq"
        )
        # Continuing after a terminal would necessarily violate the episode
        # contract.  A recorded terminal may occur on the final edge only.
        if edge_n > 1 and np.any(done[:, :-1] | truncated[:, :-1]):
            raise ValueError("done/truncated may only occur on the final sequence edge")

        if initial_prev_action is None:
            previous = np.zeros(batch_n, dtype=np.int64)
        else:
            previous = np.asarray(initial_prev_action, dtype=np.int64).reshape(-1)
            if previous.shape != (batch_n,):
                raise ValueError(
                    f"initial_prev_action shape {previous.shape} must equal {(batch_n,)}"
                )
            if np.any(previous < 0) or np.any(previous >= self.num_actions):
                raise ValueError("initial_prev_action contains an invalid action")
            previous = previous.copy()

        if score_mask is None:
            mask = np.ones(edge_n, dtype=np.bool_)
            mask[0] = False
        else:
            mask = np.asarray(score_mask, dtype=np.bool_)
            if mask.shape == (batch_n, edge_n):
                if not np.all(mask == mask[0]):
                    raise ValueError("score_mask must be identical across the batch")
                mask = mask[0]
            if mask.shape != (edge_n,):
                raise ValueError(
                    f"score_mask shape {mask.shape} must equal {(edge_n,)}"
                )
            if mask[0] or not np.all(mask[1:]):
                raise ValueError("score_mask must be [False, True, ..., True]")
            mask = mask.copy()

        if observation_space is not None:
            if not isinstance(observation_space, spaces.Box):
                raise TypeError("observation_space must be a gymnasium Box")
            if tuple(observation_space.shape) != tuple(observations.shape[2:]):
                raise ValueError(
                    "observation_space shape disagrees with obs_seq: "
                    f"{observation_space.shape} versus {observations.shape[2:]}"
                )
            obs_dtype = np.dtype(observation_space.dtype)
            if np.dtype(observations.dtype) != obs_dtype:
                raise ValueError(
                    "observation_space dtype disagrees with obs_seq: "
                    f"{obs_dtype} versus {observations.dtype}"
                )
            if np.any(observations < observation_space.low) or np.any(
                observations > observation_space.high
            ):
                raise ValueError("obs_seq contains values outside observation_space")
            replay_observation_space = spaces.Box(
                low=observation_space.low.copy(),
                high=observation_space.high.copy(),
                dtype=observation_space.dtype,
            )
        else:
            obs_dtype = np.dtype(
                np.uint8 if observations.dtype == np.uint8 else np.float32
            )
            high = 255 if obs_dtype == np.dtype(np.uint8) else np.inf
            replay_observation_space = spaces.Box(
                low=0.0,
                high=high,
                shape=observations.shape[2:],
                dtype=obs_dtype,
            )
        self.obs_seq = observations.astype(obs_dtype, copy=True)
        self.actions_seq = actions.copy()
        self.rewards_seq = rewards
        self.done_seq = done
        self.truncated_seq = truncated
        self.initial_prev_action = previous
        self.score_mask = mask
        self.env_n = batch_n
        self.seq_len = obs_n
        self.edge_n = edge_n
        self._pos = np.zeros(batch_n, dtype=np.int64)
        self._finished = np.zeros(batch_n, dtype=np.bool_)
        self._saved_positions.clear()
        self._checkpoint_slots.clear()

        self.observation_space = replay_observation_space
        self.action_space = spaces.Tuple(
            tuple(spaces.Discrete(self.num_actions) for _ in range(self.env_n))
        )
        self.reward_range = (-np.inf, np.inf)
        self.metadata: Dict[str, Any] = {}

    def update_sequences(
        self,
        obs_seq: np.ndarray,
        *,
        actions_seq: np.ndarray,
        rewards_seq: Optional[np.ndarray] = None,
        done_seq: Optional[np.ndarray] = None,
        truncated_seq: Optional[np.ndarray] = None,
        initial_prev_action: Optional[np.ndarray] = None,
        score_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Install the next batch while retaining the vector-env object."""

        self._install_sequences(
            obs_seq=obs_seq,
            actions_seq=actions_seq,
            rewards_seq=rewards_seq,
            done_seq=done_seq,
            truncated_seq=truncated_seq,
            initial_prev_action=initial_prev_action,
            score_mask=score_mask,
            observation_space=self.observation_space,
            initialise=False,
        )

    def reset(self, *, reset_stat: bool = False, seed: Optional[int] = None):
        del reset_stat, seed
        self._pos.fill(0)
        self._finished.fill(False)
        info = {
            "real_done": np.zeros(self.env_n, dtype=np.bool_),
            "episode_return": np.zeros(self.env_n, dtype=np.float32),
            "episode_step": np.zeros(self.env_n, dtype=np.int64),
            "initial_prev_action": self.initial_prev_action.copy(),
            "sequence_pos": self._pos.copy(),
        }
        return self.obs_seq[:, 0].copy(), info

    def _env_ids(self, env_id) -> np.ndarray:
        if env_id is None:
            ids = np.arange(self.env_n, dtype=np.int64)
        else:
            ids = np.asarray(env_id, dtype=np.int64).reshape(-1)
        if np.any(ids < 0) or np.any(ids >= self.env_n):
            raise IndexError(f"env_id contains an index outside [0,{self.env_n - 1}]")
        if len(np.unique(ids)) != len(ids):
            raise ValueError("env_id cannot contain duplicates")
        return ids

    def current_human_action(self, env_id=None) -> np.ndarray:
        ids = self._env_ids(env_id)
        if np.any(self._finished[ids]) or np.any(self._pos[ids] >= self.edge_n):
            raise RuntimeError("No human action remains for an exhausted sequence")
        return self.actions_seq[ids, self._pos[ids]].copy()

    def current_score_mask(self, env_id=None) -> np.ndarray:
        ids = self._env_ids(env_id)
        if np.any(self._pos[ids] >= self.edge_n):
            raise RuntimeError("No score-mask entry remains for an exhausted sequence")
        return self.score_mask[self._pos[ids]].copy()

    def has_more(self, env_id=None):
        ids = self._env_ids(env_id)
        value = (~self._finished[ids]) & (self._pos[ids] < self.edge_n)
        if env_id is None:
            return value
        return value.copy()

    def step(self, action, *, env_id=None):
        ids = self._env_ids(env_id)
        if np.any(self._finished[ids]) or np.any(self._pos[ids] >= self.edge_n):
            raise RuntimeError(
                "Behavioral sequence exhausted; call update_sequences/reset instead of wrapping"
            )
        supplied = _as_numpy(action)
        if supplied.ndim == 0 and len(ids) == 1:
            supplied = supplied.reshape(1)
        elif supplied.ndim == 2 and supplied.shape[1] == 1:
            supplied = supplied[:, 0]
        supplied = np.asarray(supplied, dtype=np.int64).reshape(-1)
        if supplied.shape == (self.env_n,) and len(ids) != self.env_n:
            supplied = supplied[ids]
        if supplied.shape != (len(ids),):
            raise ValueError(
                f"Action shape {supplied.shape} does not match selected env count {len(ids)}"
            )
        positions = self._pos[ids].copy()
        expected = self.actions_seq[ids, positions]
        mismatch = supplied != expected
        if np.any(mismatch):
            bad = ids[np.flatnonzero(mismatch)]
            raise ValueError(
                "Teacher-forced action mismatch for env(s) "
                f"{bad.tolist()}: supplied={supplied[mismatch].tolist()}, "
                f"human={expected[mismatch].tolist()}"
            )

        reward = self.rewards_seq[ids, positions].copy()
        done = self.done_seq[ids, positions].copy()
        truncated = self.truncated_seq[ids, positions].copy()
        scored = self.score_mask[positions].copy()
        self._pos[ids] = positions + 1
        self._finished[ids] = done | truncated | (self._pos[ids] >= self.edge_n)
        obs = self.obs_seq[ids, self._pos[ids]].copy()
        info = {
            "real_done": done.copy(),
            "human_action": expected.copy(),
            "score_mask": scored,
            "sequence_pos": self._pos[ids].copy(),
        }
        return obs, reward, done, truncated, info

    def quick_save(self, env_id=None, slot_id=None):
        del env_id
        if slot_id is None:
            self._saved_positions.append(self._pos.copy())
        else:
            self._checkpoint_slots[int(slot_id)] = self._pos.copy()

    def quick_load(self, env_id=None, slot_id=None):
        del env_id
        if slot_id is None:
            if not self._saved_positions:
                raise ValueError("No saved sequence cursor")
            position = self._saved_positions.pop()
        else:
            key = int(slot_id)
            if key not in self._checkpoint_slots:
                raise ValueError(f"No sequence cursor in slot {key}")
            position = self._checkpoint_slots[key]
        self._pos = position.copy()
        self._finished = self._pos >= self.edge_n

    def quick_delete(self, env_id=None, slot_id=0):
        del env_id
        self._checkpoint_slots.pop(int(slot_id), None)

    def save_ckp(self):
        return {
            "sequence_pos": self._pos.copy(),
            "sequence_finished": self._finished.copy(),
        }

    def load_ckp(self, data):
        if data is None or "sequence_pos" not in data:
            return
        position = np.asarray(data["sequence_pos"], dtype=np.int64)
        if position.shape != (self.env_n,) or np.any(position < 0) or np.any(
            position > self.edge_n
        ):
            raise ValueError("Invalid sequence_pos checkpoint")
        self._pos = position.copy()
        if "sequence_finished" in data:
            finished = np.asarray(data["sequence_finished"], dtype=np.bool_)
            if finished.shape != (self.env_n,):
                raise ValueError("Invalid sequence_finished checkpoint")
            self._finished = finished.copy()
        else:
            self._finished = self._pos >= self.edge_n

    def close(self):
        self._saved_positions.clear()
        self._checkpoint_slots.clear()
