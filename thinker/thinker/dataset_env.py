"""Teacher-forced environments for fixed behavioral sequences.

``BehaviorSequenceVectorEnv`` is the batched, in-memory training path.  The
single-environment ``LazyBehaviorEpisodeVectorEnv`` at the bottom of this file
has the same edge semantics but reads observations only when a real edge is
executed.  It is intended for long recorder/evaluation episodes where copying
the complete frame sequence into RAM would defeat streaming output.
"""

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


class LazyBehaviorEpisodeVectorEnv:
    """Replay one human episode from an on-demand observation provider.

    The provider is either callable as ``provider(index)`` or implements
    ``provider[index]``.  It returns one observation (without a batch axis).
    Only observation zero is read by :meth:`reset`; every subsequent
    observation is read after its matching teacher-forced real action.  Search
    calls made inside Dynamic Thinker therefore never advance or read the
    provider.

    Actions and edge metadata may be one-dimensional or have a leading
    singleton batch dimension.  The environment itself is always vectorized
    with ``env_n == 1`` so it can be passed directly to ``cModelWrapper``.
    """

    def __init__(
        self,
        observation_provider: Any,
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
        if not callable(observation_provider) and not hasattr(
            observation_provider, "__getitem__"
        ):
            raise TypeError(
                "observation_provider must be callable or support __getitem__"
            )
        self.observation_provider = observation_provider
        self.env_n = 1
        self._saved_positions: List[np.ndarray] = []
        self._checkpoint_slots: Dict[int, np.ndarray] = {}

        actions = self._episode_array(
            actions_seq, np.int64, "actions_seq", allow_scalar=False
        )
        if actions.size < 1:
            raise ValueError("actions_seq must contain at least one edge")
        if np.any(actions < 0) or np.any(actions >= self.num_actions):
            raise ValueError(
                f"actions_seq contains values outside [0,{self.num_actions - 1}]"
            )
        self.actions_seq = actions.copy()
        self.edge_n = int(actions.size)
        self.seq_len = self.edge_n + 1

        self.rewards_seq = self._edge_array(
            rewards_seq, np.float32, 0.0, "rewards_seq"
        )
        self.done_seq = self._edge_array(
            done_seq, np.bool_, False, "done_seq"
        )
        self.truncated_seq = self._edge_array(
            truncated_seq, np.bool_, False, "truncated_seq"
        )
        if self.edge_n > 1 and np.any(
            self.done_seq[:-1] | self.truncated_seq[:-1]
        ):
            raise ValueError(
                "done/truncated may only occur on the final episode edge"
            )

        if initial_prev_action is None:
            previous = np.zeros(1, dtype=np.int64)
        else:
            previous_values = self._episode_array(
                initial_prev_action,
                np.int64,
                "initial_prev_action",
                allow_scalar=True,
            )
            if previous_values.size != 1:
                raise ValueError("initial_prev_action must be scalar for batch=1")
            previous = previous_values.reshape(1).copy()
        if previous[0] < 0 or previous[0] >= self.num_actions:
            raise ValueError("initial_prev_action contains an invalid action")
        self.initial_prev_action = previous

        if score_mask is None:
            mask = np.ones(self.edge_n, dtype=np.bool_)
            mask[0] = False
        else:
            mask = self._episode_array(
                score_mask, np.bool_, "score_mask", allow_scalar=False
            )
            if mask.shape != (self.edge_n,):
                raise ValueError(
                    f"score_mask shape {mask.shape} must equal {(self.edge_n,)}"
                )
            if mask[0] or not np.all(mask[1:]):
                raise ValueError("score_mask must be [False, True, ..., True]")
            mask = mask.copy()
        self.score_mask = mask

        provider_shape = getattr(observation_provider, "shape", None)
        provider_dtype = getattr(observation_provider, "dtype", None)
        if provider_shape is not None:
            provider_shape = tuple(provider_shape)
            if len(provider_shape) < 2:
                raise ValueError(
                    "observation_provider.shape must be [E+1,...]"
                )
            if int(provider_shape[0]) != self.seq_len:
                raise ValueError(
                    "observation provider length disagrees with actions_seq: "
                    f"{provider_shape[0]} versus {self.seq_len}"
                )

        if observation_space is not None:
            if not isinstance(observation_space, spaces.Box):
                raise TypeError("observation_space must be a gymnasium Box")
            if provider_shape is not None and tuple(
                observation_space.shape
            ) != tuple(provider_shape[1:]):
                raise ValueError(
                    "observation_space shape disagrees with provider.shape"
                )
            if provider_dtype is not None and np.dtype(
                observation_space.dtype
            ) != np.dtype(provider_dtype):
                raise ValueError(
                    "observation_space dtype disagrees with provider.dtype"
                )
            self.observation_space = spaces.Box(
                low=observation_space.low.copy(),
                high=observation_space.high.copy(),
                dtype=observation_space.dtype,
            )
        else:
            # Reading one frame to infer metadata is bounded and does not retain
            # it.  No other observation is touched before reset/step requests it.
            first = self._provider_observation(0)
            obs_dtype = np.dtype(
                np.uint8 if first.dtype == np.uint8 else np.float32
            )
            if provider_dtype is not None and np.dtype(provider_dtype) != obs_dtype:
                raise ValueError("provider dtype metadata disagrees with observation 0")
            high = 255 if obs_dtype == np.dtype(np.uint8) else np.inf
            self.observation_space = spaces.Box(
                low=0.0,
                high=high,
                shape=first.shape,
                dtype=obs_dtype,
            )

        self.action_space = spaces.Tuple((spaces.Discrete(self.num_actions),))
        self.reward_range = (-np.inf, np.inf)
        self.metadata: Dict[str, Any] = {}
        self._pos = np.zeros(1, dtype=np.int64)
        self._finished = np.zeros(1, dtype=np.bool_)

    @staticmethod
    def _episode_array(value, dtype, name: str, *, allow_scalar: bool) -> np.ndarray:
        array = np.asarray(_as_numpy(value), dtype=dtype)
        if allow_scalar and array.ndim == 0:
            return array.reshape(1)
        if array.ndim == 2 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 1:
            raise ValueError(f"{name} must be [E] or [1,E] for batch=1")
        return array

    def _edge_array(self, value, dtype, default, name: str) -> np.ndarray:
        if value is None:
            return np.full(self.edge_n, default, dtype=dtype)
        array = self._episode_array(value, dtype, name, allow_scalar=False)
        if array.shape != (self.edge_n,):
            raise ValueError(
                f"{name} shape {array.shape} must equal {(self.edge_n,)}"
            )
        return array.copy()

    def _provider_observation(self, index: int) -> np.ndarray:
        if not 0 <= int(index) < self.seq_len:
            raise IndexError(f"observation index {index} outside [0,{self.seq_len - 1}]")
        if callable(self.observation_provider):
            value = self.observation_provider(int(index))
        else:
            value = self.observation_provider[int(index)]
        observation = np.asarray(_as_numpy(value))
        if observation.ndim < 1:
            raise ValueError("observation provider returned a scalar")
        return observation

    def _read_observation(self, index: int) -> np.ndarray:
        observation = self._provider_observation(index)
        expected_shape = tuple(self.observation_space.shape)
        expected_dtype = np.dtype(self.observation_space.dtype)
        if tuple(observation.shape) != expected_shape:
            raise ValueError(
                "observation provider returned shape "
                f"{observation.shape}; expected {expected_shape}"
            )
        if np.dtype(observation.dtype) != expected_dtype:
            raise ValueError(
                "observation provider returned dtype "
                f"{observation.dtype}; expected {expected_dtype}"
            )
        if np.issubdtype(expected_dtype, np.floating) and not np.all(
            np.isfinite(observation)
        ):
            raise ValueError("observation provider returned non-finite values")
        if np.any(observation < self.observation_space.low) or np.any(
            observation > self.observation_space.high
        ):
            raise ValueError("observation lies outside observation_space")
        return observation.copy()

    def reset(self, *, reset_stat: bool = False, seed: Optional[int] = None):
        del reset_stat, seed
        self._pos.fill(0)
        self._finished.fill(False)
        info = {
            "real_done": np.zeros(1, dtype=np.bool_),
            "episode_return": np.zeros(1, dtype=np.float32),
            "episode_step": np.zeros(1, dtype=np.int64),
            "initial_prev_action": self.initial_prev_action.copy(),
            "sequence_pos": self._pos.copy(),
        }
        return self._read_observation(0)[None, ...], info

    @staticmethod
    def _env_ids(env_id) -> np.ndarray:
        if env_id is None:
            return np.zeros(1, dtype=np.int64)
        ids = np.asarray(env_id, dtype=np.int64).reshape(-1)
        if ids.shape != (1,) or ids[0] != 0:
            raise IndexError("LazyBehaviorEpisodeVectorEnv only supports env_id=[0]")
        return ids

    def current_human_action(self, env_id=None) -> np.ndarray:
        self._env_ids(env_id)
        if self._finished[0] or self._pos[0] >= self.edge_n:
            raise RuntimeError("No human action remains for an exhausted episode")
        return self.actions_seq[self._pos].copy()

    def current_score_mask(self, env_id=None) -> np.ndarray:
        self._env_ids(env_id)
        if self._pos[0] >= self.edge_n:
            raise RuntimeError("No score-mask entry remains for an exhausted episode")
        return self.score_mask[self._pos].copy()

    def has_more(self, env_id=None):
        self._env_ids(env_id)
        value = (~self._finished) & (self._pos < self.edge_n)
        return value.copy()

    def step(self, action, *, env_id=None):
        self._env_ids(env_id)
        if self._finished[0] or self._pos[0] >= self.edge_n:
            raise RuntimeError(
                "Behavioral episode exhausted; create/reset an episode instead of wrapping"
            )
        supplied = np.asarray(_as_numpy(action), dtype=np.int64).reshape(-1)
        if supplied.shape != (1,):
            raise ValueError("Action must have shape (1,) for the lazy episode env")
        position = int(self._pos[0])
        expected = int(self.actions_seq[position])
        if int(supplied[0]) != expected:
            raise ValueError(
                "Teacher-forced action mismatch for env 0: "
                f"supplied={int(supplied[0])}, human={expected}"
            )

        reward = self.rewards_seq[position : position + 1].copy()
        done = self.done_seq[position : position + 1].copy()
        truncated = self.truncated_seq[position : position + 1].copy()
        scored = self.score_mask[position : position + 1].copy()
        self._pos[0] = position + 1
        self._finished[0] = bool(
            done[0] or truncated[0] or self._pos[0] >= self.edge_n
        )
        observation = self._read_observation(int(self._pos[0]))[None, ...]
        info = {
            "real_done": done.copy(),
            "human_action": np.array([expected], dtype=np.int64),
            "score_mask": scored,
            "sequence_pos": self._pos.copy(),
        }
        return observation, reward, done, truncated, info

    def quick_save(self, env_id=None, slot_id=None):
        self._env_ids(env_id)
        if slot_id is None:
            self._saved_positions.append(self._pos.copy())
        else:
            self._checkpoint_slots[int(slot_id)] = self._pos.copy()

    def quick_load(self, env_id=None, slot_id=None):
        self._env_ids(env_id)
        if slot_id is None:
            if not self._saved_positions:
                raise ValueError("No saved episode cursor")
            position = self._saved_positions.pop()
        else:
            key = int(slot_id)
            if key not in self._checkpoint_slots:
                raise ValueError(f"No episode cursor in slot {key}")
            position = self._checkpoint_slots[key]
        self._pos = position.copy()
        self._finished = self._pos >= self.edge_n

    def quick_delete(self, env_id=None, slot_id=0):
        self._env_ids(env_id)
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
        if position.shape != (1,) or np.any(position < 0) or np.any(
            position > self.edge_n
        ):
            raise ValueError("Invalid sequence_pos checkpoint")
        self._pos = position.copy()
        if "sequence_finished" in data:
            finished = np.asarray(data["sequence_finished"], dtype=np.bool_)
            if finished.shape != (1,):
                raise ValueError("Invalid sequence_finished checkpoint")
            self._finished = finished.copy()
        else:
            self._finished = self._pos >= self.edge_n

    def close(self):
        self._saved_positions.clear()
        self._checkpoint_slots.clear()
