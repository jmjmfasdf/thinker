"""Behavioral sequence loading for teacher-forced Dynamic Thinker training.

The behavioral archives contain timestamped display frames, whereas Atari
agents act once every four 60 Hz emulator frames.  This module constructs a
causal 15 Hz view of each recorded episode and samples windows with the exact
contract used by dynamic imitation:

``obs[0]`` / ``initial_prev_action`` initialise the burn-in root, edge zero is
the unscored burn-in human action, and edges one through ``L`` are scored.
No window is allowed to cross a recorded episode/file/time-gap boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from numbers import Integral
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


ATARI_DECISION_INTERVAL_SECONDS = 4.0 / 60.0


def behavioral_data_signature(loader: Any, data_root: str | Path) -> str:
    """Hash the exact behavioral training contract and selected archives.

    Absolute paths are deliberately excluded so a checkpoint can be resumed
    after the same files are staged beneath a job-local scratch directory.
    File content, relative identity, and every preprocessing choice that can
    change a sampled sequence remain part of the signature.
    """

    root = Path(data_root).expanduser().resolve()
    file_manifest = []
    for value in getattr(loader, "data_files", ()):
        path = Path(value).expanduser().resolve()
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        try:
            relative_path = path.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError(
                f"behavior file {path} is outside declared data root {root}"
            ) from error
        file_manifest.append(
            {
                "path": relative_path,
                "size": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )

    payload = {
        "protocol": "obs[t]-to-action[t]/burnin=1/v1",
        "subjects": list(getattr(loader, "subjects", ()) or ()),
        "sessions": list(getattr(loader, "sessions", ()) or ()),
        "game_id": int(loader.game_id),
        "num_actions": int(loader.num_actions),
        "scored_length": int(loader.scored_length),
        "frame_stack_n": int(loader.frame_stack_n),
        "target_size": list(loader.target_size),
        "grayscale": bool(loader.grayscale),
        "normalize": bool(loader.normalize),
        "observation_dtype": "float32" if loader.normalize else "uint8",
        "observation_range": [0.0, 1.0] if loader.normalize else [0, 255],
        "decision_hz": loader.decision_hz,
        "max_time_gap": loader.max_time_gap,
        "files": sorted(file_manifest, key=lambda item: item["path"]),
        "loader_signature": str(getattr(loader, "data_signature", None)),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BehaviorFile:
    """One source archive and the entities parsed from its path."""

    path: Path
    subject: int
    session: int
    block: int
    game: int


@dataclass
class _LogicalEpisode:
    """A source episode after causal decision-time resampling."""

    file_index: int
    episode_index: int
    observation_source_index: np.ndarray
    decision_times: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    done: np.ndarray
    truncated: np.ndarray


@dataclass(frozen=True)
class _WindowRef:
    episode_index: int
    burn_in_edge: int


def _normalise_ids(values: Optional[Iterable[int]]) -> Optional[Tuple[int, ...]]:
    if values is None:
        return None
    result = tuple(sorted({int(value) for value in values}))
    if not result:
        raise ValueError("An explicitly supplied ID filter cannot be empty")
    return result


def _path_entity(path: Path, entity: str) -> Optional[int]:
    """Parse modern BIDS-like and historical underscore path entities."""

    aliases = {
        "subject": ("sub",),
        "session": ("ses", "day"),
        "block": ("block",),
        "game": ("game",),
    }[entity]
    text = path.as_posix().lower()
    for alias in aliases:
        match = re.search(rf"(?:^|[/_-]){alias}[-_]?0*(\d+)(?=$|[/_.-])", text)
        if match is not None:
            return int(match.group(1))
    return None


class FrameStackedBehavioralDataLoader:
    """Load auditable, episode-safe behavioral imitation sequences.

    Parameters
    ----------
    base_path:
        Root containing either ``sub-001/ses-01/*game0.npz`` archives or the
        historical ``sub_1/game_0/day_1/block_*/**/*.npz`` layout.
    split / sessions:
        ``split='train'`` selects sessions 1--3 and ``split='holdout'`` (also
        ``'test'``/``'eval'``) selects session 4.  An explicit ``sessions``
        filter takes precedence.
    scored_length:
        Number of scored actions.  Every sampled sequence additionally has
        one unscored burn-in edge and one final next observation.
    num_actions:
        Size of the runtime policy's discrete action space.  This is supplied
        by the environment/Actor rather than inferred from ``game_id`` so new
        games cannot silently inherit a stale hard-coded action count.
    decision_hz:
        Causal sampling rate.  The Atari frame-skip-four default is 15 Hz.
        Archives without timestamps are treated as already decision sampled.
    """

    def __init__(
        self,
        base_path: str | Path = "behavioral_data_block",
        subjects: Sequence[int] = (1,),
        game_id: int = 0,
        sessions: Optional[Sequence[int]] = None,
        *,
        num_actions: int,
        split: Optional[str] = "train",
        scored_length: int = 4,
        frame_stack_n: int = 4,
        target_size: Tuple[int, int] = (84, 84),
        grayscale: bool = False,
        normalize: bool = False,
        decision_hz: Optional[float] = 1.0 / ATARI_DECISION_INTERVAL_SECONDS,
        max_time_gap: Optional[float] = None,
        seed: int = 0,
    ) -> None:
        self.base_path = Path(base_path).expanduser().resolve()
        self.subjects = _normalise_ids(subjects)
        self.game_id = int(game_id)
        if isinstance(num_actions, (bool, np.bool_)) or not isinstance(
            num_actions, Integral
        ):
            raise ValueError("num_actions must be a positive integer")
        self.num_actions = int(num_actions)
        self.sessions = self._resolve_sessions(sessions, split)
        self.split = split
        self.scored_length = int(scored_length)
        self.frame_stack_n = int(frame_stack_n)
        self.target_size = (int(target_size[0]), int(target_size[1]))
        self.grayscale = bool(grayscale)
        self.normalize = bool(normalize)
        self.decision_hz = None if decision_hz is None else float(decision_hz)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        if self.num_actions < 1:
            raise ValueError("num_actions must be a positive integer")
        if self.scored_length < 1:
            raise ValueError("scored_length must be at least one")
        if self.frame_stack_n < 1:
            raise ValueError("frame_stack_n must be at least one")
        if min(self.target_size) < 1:
            raise ValueError("target_size dimensions must be positive")
        if self.decision_hz is not None and self.decision_hz <= 0:
            raise ValueError("decision_hz must be positive or None")

        self.decision_interval = (
            None if self.decision_hz is None else 1.0 / self.decision_hz
        )
        if max_time_gap is None:
            self.max_time_gap = (
                None
                if self.decision_interval is None
                else 2.5 * self.decision_interval
            )
        else:
            self.max_time_gap = float(max_time_gap)
            if self.max_time_gap <= 0:
                raise ValueError("max_time_gap must be positive")

        self.file_records = self._discover_files()
        if not self.file_records:
            raise FileNotFoundError(
                "No behavioral NPZ files matched "
                f"root={self.base_path}, subjects={self.subjects}, "
                f"sessions={self.sessions}, game={self.game_id}"
            )
        # Kept for compatibility with the previous sequential-IL loader.
        self.data_files = [str(record.path) for record in self.file_records]

        self._episodes: List[_LogicalEpisode] = []
        self._index_archives()
        if not self._episodes:
            raise ValueError("Matched behavioral files contain no usable episodes")
        if not self._window_refs(self.scored_length):
            raise ValueError(
                "Matched behavioral episodes contain no windows satisfying the "
                "two-predecessor and within-episode sequence contract"
            )
        self.action_distribution = self._compute_action_distribution()

    @staticmethod
    def _resolve_sessions(
        sessions: Optional[Sequence[int]], split: Optional[str]
    ) -> Optional[Tuple[int, ...]]:
        explicit = _normalise_ids(sessions)
        if explicit is not None:
            return explicit
        if split is None or str(split).lower() in {"all", "none"}:
            return None
        split_name = str(split).lower()
        if split_name == "train":
            return (1, 2, 3)
        if split_name in {"holdout", "test", "eval", "validation", "val"}:
            return (4,)
        raise ValueError(f"Unknown behavioral split: {split!r}")

    def _discover_files(self) -> List[BehaviorFile]:
        if not self.base_path.is_dir():
            return []
        records: List[BehaviorFile] = []
        for path in self.base_path.rglob("*.npz"):
            subject = _path_entity(path, "subject")
            session = _path_entity(path, "session")
            block = _path_entity(path, "block")
            game = _path_entity(path, "game")
            # All entities except block are necessary to enforce leakage-safe
            # filtering.  A missing legacy block is represented by zero.
            if subject is None or session is None or game is None:
                continue
            if self.subjects is not None and subject not in self.subjects:
                continue
            if self.sessions is not None and session not in self.sessions:
                continue
            if game != self.game_id:
                continue
            records.append(
                BehaviorFile(
                    path=path.resolve(),
                    subject=subject,
                    session=session,
                    block=0 if block is None else block,
                    game=game,
                )
            )
        return sorted(
            records,
            key=lambda item: (
                item.subject,
                item.session,
                item.block,
                item.path.as_posix(),
            ),
        )

    def _action_indices(self, actions: np.ndarray) -> np.ndarray:
        """Decode scalar indices or a strict one-hot action matrix.

        Treating arbitrary matrices as logits and applying ``argmax`` can hide
        a wrong action vocabulary (for example a six-column archive used with
        a nine-action Actor).  Behavioral archives are therefore accepted only
        as integer-valued scalar indices or exact one-hot rows whose width
        matches the runtime action space.
        """

        array = np.asarray(actions)
        if array.ndim not in (1, 2):
            raise ValueError(
                "action must be scalar indices [T] / [T,1] or one-hot [T,A], "
                f"got {array.shape}"
            )
        if array.ndim == 2 and array.shape[1] == 0:
            raise ValueError("action one-hot width must be positive")

        if not (
            np.issubdtype(array.dtype, np.bool_)
            or np.issubdtype(array.dtype, np.integer)
            or np.issubdtype(array.dtype, np.floating)
        ):
            raise ValueError(f"action values must be numeric, got dtype {array.dtype}")
        if not np.all(np.isfinite(array)):
            raise ValueError("action values must all be finite")

        scalar_encoding = array.ndim == 1 or array.shape[1] == 1
        if scalar_encoding:
            scalar = array if array.ndim == 1 else array[:, 0]
            if np.issubdtype(scalar.dtype, np.floating) and not np.all(
                scalar == np.floor(scalar)
            ):
                raise ValueError("scalar actions must be integer-valued")
            if np.any(scalar < 0) or np.any(scalar >= self.num_actions):
                raise ValueError(
                    "scalar actions fall outside "
                    f"[0, {self.num_actions - 1}]"
                )
            return np.asarray(scalar, dtype=np.int64).reshape(-1)

        width = int(array.shape[1])
        if width != self.num_actions:
            raise ValueError(
                f"action one-hot width {width} does not match "
                f"num_actions={self.num_actions}"
            )
        binary = np.logical_or(array == 0, array == 1)
        if not np.all(binary) or not np.all(np.sum(array, axis=1) == 1):
            raise ValueError(
                "action matrix must contain strict one-hot rows with exactly "
                "one 1 and all remaining values 0"
            )
        return np.argmax(array, axis=1).astype(np.int64, copy=False)

    @staticmethod
    def _read_optional(
        archive: np.lib.npyio.NpzFile,
        names: Sequence[str],
        length: int,
        dtype,
        default,
    ) -> np.ndarray:
        for name in names:
            if name in archive.files:
                value = np.asarray(archive[name], dtype=dtype).reshape(-1)
                if len(value) != length:
                    raise ValueError(
                        f"{name} length {len(value)} does not match actions length {length}"
                    )
                return value
        return np.full(length, default, dtype=dtype)

    def _index_archives(self) -> None:
        for file_index, record in enumerate(self.file_records):
            try:
                with np.load(record.path, allow_pickle=False) as archive:
                    missing = {"image", "action"}.difference(archive.files)
                    if missing:
                        raise ValueError(f"missing required arrays {sorted(missing)}")
                    actions = self._action_indices(archive["action"])
                    n_steps = len(actions)
                    if n_steps < 2:
                        continue
                    if np.any(actions < 0) or np.any(actions >= self.num_actions):
                        raise ValueError(
                            f"actions fall outside [0, {self.num_actions - 1}]"
                        )
                    rewards = self._read_optional(
                        archive, ("reward", "rewards"), n_steps, np.float32, 0.0
                    )
                    is_first = self._read_optional(
                        archive,
                        ("is_first", "first"),
                        n_steps,
                        np.bool_,
                        False,
                    )
                    is_terminal = self._read_optional(
                        archive,
                        ("is_terminal", "terminal", "done"),
                        n_steps,
                        np.bool_,
                        False,
                    )
                    truncated = self._read_optional(
                        archive,
                        ("is_truncated", "truncated"),
                        n_steps,
                        np.bool_,
                        False,
                    )
                    if "time" in archive.files:
                        times = np.asarray(archive["time"], dtype=np.float64).reshape(-1)
                        if len(times) != n_steps:
                            raise ValueError(
                                f"time length {len(times)} does not match actions length {n_steps}"
                            )
                        if not np.all(np.isfinite(times)):
                            raise ValueError("timestamps contain non-finite values")
                        has_timestamps = True
                    else:
                        times = np.arange(n_steps, dtype=np.float64)
                        has_timestamps = False
            except Exception as error:
                raise ValueError(f"Invalid behavioral archive {record.path}: {error}") from error

            # A file boundary is always a hard sequence boundary, but it is
            # not labelled as a terminal transition.
            is_first = np.asarray(is_first, dtype=np.bool_).copy()
            is_first[0] = True
            spans = self._episode_spans(
                times=times,
                is_first=is_first,
                is_terminal=is_terminal,
                truncated=truncated,
                has_timestamps=has_timestamps,
            )
            for local_episode_index, (start, end) in enumerate(spans):
                episode = self._make_logical_episode(
                    file_index=file_index,
                    episode_index=local_episode_index,
                    start=start,
                    end=end,
                    times=times,
                    actions=actions,
                    rewards=rewards,
                    is_terminal=is_terminal,
                    truncated=truncated,
                    has_timestamps=has_timestamps,
                )
                if episode is not None:
                    self._episodes.append(episode)

    def _episode_spans(
        self,
        *,
        times: np.ndarray,
        is_first: np.ndarray,
        is_terminal: np.ndarray,
        truncated: np.ndarray,
        has_timestamps: bool,
    ) -> List[Tuple[int, int]]:
        spans: List[Tuple[int, int]] = []
        start = 0
        for index in range(len(times) - 1):
            delta = float(times[index + 1] - times[index])
            time_break = has_timestamps and (
                delta <= 0
                or (self.max_time_gap is not None and delta > self.max_time_gap)
            )
            boundary = bool(
                is_terminal[index]
                or truncated[index]
                or is_first[index + 1]
                or time_break
            )
            if boundary:
                if index >= start:
                    spans.append((start, index))
                start = index + 1
        if start < len(times):
            spans.append((start, len(times) - 1))
        return spans

    def _make_logical_episode(
        self,
        *,
        file_index: int,
        episode_index: int,
        start: int,
        end: int,
        times: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        is_terminal: np.ndarray,
        truncated: np.ndarray,
        has_timestamps: bool,
    ) -> Optional[_LogicalEpisode]:
        if end - start < 1:
            return None
        if has_timestamps and self.decision_interval is not None:
            start_time = float(times[start])
            duration = max(0.0, float(times[end]) - start_time)
            count = int(np.floor(duration / self.decision_interval + 1e-9)) + 1
            if count < 2:
                return None
            decision_times = start_time + self.decision_interval * np.arange(
                count, dtype=np.float64
            )
            local_times = times[start : end + 1]
            source = np.searchsorted(local_times, decision_times, side="right") - 1
            source = np.clip(source, 0, len(local_times) - 1) + start
        else:
            source = np.arange(start, end + 1, dtype=np.int64)
            if has_timestamps:
                decision_times = times[source].astype(np.float64, copy=True)
            else:
                # The actual unit is unavailable in legacy chunks; preserving
                # row order is preferable to inventing a timestamp alignment.
                decision_times = np.arange(len(source), dtype=np.float64)

        source = np.asarray(source, dtype=np.int64)
        decision_times = np.asarray(decision_times, dtype=np.float64)
        # A regular 15-Hz grid will usually stop a fraction of a capture frame
        # before an off-grid terminal timestamp.  Preserve the genuine terminal
        # image as one final (possibly shorter) transition endpoint rather than
        # either dropping the terminal flag or fabricating a reset observation.
        endpoint_is_terminal = bool(is_terminal[end] or truncated[end])
        if endpoint_is_terminal and int(source[-1]) != int(end):
            source = np.concatenate(
                [source, np.asarray([end], dtype=np.int64)]
            )
            decision_times = np.concatenate(
                [decision_times, np.asarray([times[end]], dtype=np.float64)]
            )
        if len(source) < 2:
            return None
        edge_n = len(source) - 1
        logical_actions = actions[source[:-1]].astype(np.int64, copy=True)
        logical_rewards = np.zeros(edge_n, dtype=np.float32)
        logical_done = np.zeros(edge_n, dtype=np.bool_)
        logical_truncated = np.zeros(edge_n, dtype=np.bool_)

        # Row i stores the human action aligned with image i.  Timestamped
        # edge rewards cover the *ensuing* half-open interval [tau, tau_next),
        # never the interval preceding the selected action.  The causal image
        # may be slightly before tau, so its source index cannot safely serve
        # as the reward-bin boundary.  Timestamp-free legacy rows are already
        # decision sampled and use their current-to-next row interval.
        for edge in range(edge_n):
            if has_timestamps and self.decision_interval is not None:
                lo = int(np.searchsorted(times, decision_times[edge], side="left"))
                hi = int(
                    np.searchsorted(times, decision_times[edge + 1], side="left")
                )
                lo = min(max(lo, start), end + 1)
                hi = min(max(hi, lo), end + 1)
            else:
                lo = int(source[edge])
                hi = int(source[edge + 1])
            if hi <= lo:
                hi = min(lo + 1, end + 1)
            logical_rewards[edge] = np.sum(rewards[lo:hi], dtype=np.float32)
            # ``image[source[edge + 1]]`` is the recorded next observation.
            # Episode-end flags live on that endpoint row in the behavioral
            # archives, whereas reward integration above is deliberately the
            # half-open action interval.  Include the endpoint for termination
            # only, otherwise the final recorded transition is mislabeled as
            # continuing and an expanded child can be carried across death.
            status_stop = min(int(source[edge + 1]) + 1, end + 1)
            status_stop = max(status_stop, lo + 1)
            logical_done[edge] = bool(np.any(is_terminal[lo:status_stop]))
            logical_truncated[edge] = bool(np.any(truncated[lo:status_stop]))

        return _LogicalEpisode(
            file_index=file_index,
            episode_index=episode_index,
            observation_source_index=source,
            decision_times=decision_times,
            actions=logical_actions,
            rewards=logical_rewards,
            done=logical_done,
            truncated=logical_truncated,
        )

    def _window_refs(self, scored_length: int) -> List[_WindowRef]:
        scored_length = int(scored_length)
        if scored_length < 1:
            raise ValueError("sequence_length must be at least one")
        refs: List[_WindowRef] = []
        required_edges = scored_length + 1
        for episode_index, episode in enumerate(self._episodes):
            max_burn_edge = len(episode.actions) - required_edges
            # burn_edge >= 1 ensures that a_{t-2}, a_{t-1} both exist before
            # the first scored target a_t.
            for burn_edge in range(1, max_burn_edge + 1):
                stop = burn_edge + required_edges
                # A real terminal is legal only on the final returned edge;
                # otherwise the fixed-length sequence would continue past it.
                if np.any(episode.done[burn_edge : stop - 1]):
                    continue
                if np.any(episode.truncated[burn_edge : stop - 1]):
                    continue
                refs.append(_WindowRef(episode_index, burn_edge))
        return refs

    def _compute_action_distribution(self) -> np.ndarray:
        counts = np.zeros(self.num_actions, dtype=np.float64)
        for episode in self._episodes:
            counts += np.bincount(
                episode.actions, minlength=self.num_actions
            )[: self.num_actions]
        total = float(np.sum(counts))
        if total == 0:
            return np.full(self.num_actions, 1.0 / self.num_actions, dtype=np.float64)
        return counts / total

    @property
    def num_windows(self) -> int:
        return len(self._window_refs(self.scored_length))

    def reseed(self, seed: int) -> None:
        """Restart deterministic window sampling without rebuilding the index."""

        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        array = np.asarray(frame)
        if array.ndim == 2:
            array = array[..., None]
        elif array.ndim == 3 and array.shape[0] in {1, 3, 4} and array.shape[-1] > 4:
            array = np.transpose(array, (1, 2, 0))
        if array.ndim != 3:
            raise ValueError(f"Each image must be HWC or CHW, got {array.shape}")
        if array.shape[-1] == 4:
            array = array[..., :3]
        target_h, target_w = self.target_size
        if array.shape[:2] != (target_h, target_w):
            array = cv2.resize(
                array, (target_w, target_h), interpolation=cv2.INTER_AREA
            )
            if array.ndim == 2:
                array = array[..., None]
        if self.grayscale:
            if array.shape[-1] == 3:
                array = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)[..., None]
            elif array.shape[-1] != 1:
                raise ValueError(f"Cannot grayscale image with shape {array.shape}")
        elif array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        array = np.transpose(array, (2, 0, 1))
        if self.normalize:
            return array.astype(np.float32) / 255.0
        return array.astype(np.uint8, copy=False)

    def _stack_source_indices(
        self, episode: _LogicalEpisode, observation_position: int
    ) -> np.ndarray:
        positions = np.arange(
            observation_position - self.frame_stack_n + 1,
            observation_position + 1,
            dtype=np.int64,
        )
        # Atari FrameStack repeats the reset observation before enough real
        # decisions exist; importantly, padding never reaches another episode.
        positions = np.clip(positions, 0, len(episode.observation_source_index) - 1)
        return episode.observation_source_index[positions]

    def get_sequence_batch(
        self,
        batch_size: int = 1,
        sequence_length: Optional[int] = None,
        *,
        replace: Optional[bool] = None,
    ) -> Dict[str, np.ndarray]:
        """Sample a batch satisfying the burn-in plus scored-edge contract."""

        batch_size = int(batch_size)
        scored_length = (
            self.scored_length if sequence_length is None else int(sequence_length)
        )
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        refs = self._window_refs(scored_length)
        if not refs:
            raise ValueError(
                f"No valid within-episode windows for scored length {scored_length}"
            )
        if replace is None:
            replace = batch_size > len(refs)
        if not replace and batch_size > len(refs):
            raise ValueError(
                f"Requested {batch_size} unique windows, but only {len(refs)} exist"
            )
        selected = self.rng.choice(len(refs), size=batch_size, replace=bool(replace))
        selected_refs = [refs[int(index)] for index in np.asarray(selected).reshape(-1)]

        return self._materialize_refs(selected_refs, scored_length)

    def _strided_window_refs(
        self, scored_length: int, stride: Optional[int]
    ) -> List[_WindowRef]:
        if stride is None:
            stride = int(scored_length)
        stride = int(stride)
        if stride < 1:
            raise ValueError("stride must be positive")
        return [
            ref
            for ref in self._window_refs(scored_length)
            if (ref.burn_in_edge - 1) % stride == 0
        ]

    def evaluation_coverage(
        self, *, sequence_length: Optional[int] = None, stride: Optional[int] = None
    ) -> Dict[str, int | float]:
        """Describe deterministic fixed-length evaluation target coverage.

        With the default stride ``L``, one window scores edges ``2..L+1`` and
        the next window scores ``L+2..2L+1``: the prior window's final scored
        action is reused only as the next window's burn-in action.  A trailing
        fragment shorter than ``L`` is intentionally skipped rather than
        padded or wrapped.
        """

        scored_length = (
            self.scored_length if sequence_length is None else int(sequence_length)
        )
        refs = self._strided_window_refs(scored_length, stride)
        covered = len(refs) * scored_length
        eligible = sum(max(0, len(episode.actions) - 2) for episode in self._episodes)
        # stride < L intentionally overlaps targets, so unique coverage cannot
        # be inferred from n_windows * L.  Compute it explicitly.
        unique_targets = {
            (ref.episode_index, edge)
            for ref in refs
            for edge in range(
                ref.burn_in_edge + 1,
                ref.burn_in_edge + scored_length + 1,
            )
        }
        return {
            "n_windows": len(refs),
            "scored_targets_emitted": covered,
            "unique_scored_targets": len(unique_targets),
            "eligible_scored_targets": eligible,
            "skipped_tail_targets": max(0, eligible - len(unique_targets)),
            "coverage_fraction": (
                float(len(unique_targets) / eligible) if eligible else 0.0
            ),
        }

    def iter_batches(
        self,
        batch_size: int,
        *,
        shuffle: bool = False,
        drop_last: bool = False,
        stride: Optional[int] = None,
        sequence_length: Optional[int] = None,
    ):
        """Yield a deterministic exhaustive pass over fixed-length windows.

        ``stride=None`` uses the scored length so scored targets neither
        overlap nor leave gaps except for the reported final short fragment.
        Pass ``stride=1`` to enumerate every valid sliding window.  The final
        batch may contain fewer environments unless ``drop_last`` is true;
        individual sequences always retain the exact fixed schema.
        """

        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        scored_length = (
            self.scored_length if sequence_length is None else int(sequence_length)
        )
        refs = self._strided_window_refs(scored_length, stride)
        if shuffle:
            order = self.rng.permutation(len(refs))
            refs = [refs[int(index)] for index in order]
        for start in range(0, len(refs), batch_size):
            batch_refs = refs[start : start + batch_size]
            if len(batch_refs) < batch_size and drop_last:
                break
            if batch_refs:
                yield self._materialize_refs(batch_refs, scored_length)

    def _materialize_refs(
        self, selected_refs: Sequence[_WindowRef], scored_length: int
    ) -> Dict[str, np.ndarray]:
        """Decode image arrays for an already selected set of window refs."""

        batch_size = len(selected_refs)

        # Decompress at most one large image archive at a time.  Result slots
        # preserve RNG order even though source files are processed in groups.
        result_obs: List[Optional[np.ndarray]] = [None] * batch_size
        actions_out = np.empty((batch_size, scored_length + 1), dtype=np.int64)
        rewards_out = np.empty((batch_size, scored_length + 1), dtype=np.float32)
        done_out = np.empty((batch_size, scored_length + 1), dtype=np.bool_)
        truncated_out = np.empty((batch_size, scored_length + 1), dtype=np.bool_)
        initial_prev = np.empty(batch_size, dtype=np.int64)
        source_file: List[str] = [""] * batch_size
        subject = np.empty(batch_size, dtype=np.int64)
        session = np.empty(batch_size, dtype=np.int64)
        block = np.empty(batch_size, dtype=np.int64)
        game = np.empty(batch_size, dtype=np.int64)
        local_episode = np.empty(batch_size, dtype=np.int64)
        window_start = np.empty(batch_size, dtype=np.int64)
        decision_times = np.empty(
            (batch_size, scored_length + 2), dtype=np.float64
        )
        observation_source = np.empty(
            (batch_size, scored_length + 2), dtype=np.int64
        )

        by_file: Dict[int, List[Tuple[int, _WindowRef]]] = {}
        for output_index, ref in enumerate(selected_refs):
            episode = self._episodes[ref.episode_index]
            by_file.setdefault(episode.file_index, []).append((output_index, ref))

        for file_index, group in by_file.items():
            record = self.file_records[file_index]
            with np.load(record.path, allow_pickle=False) as archive:
                images = np.asarray(archive["image"])
                for output_index, ref in group:
                    episode = self._episodes[ref.episode_index]
                    burn = ref.burn_in_edge
                    obs_stop = burn + scored_length + 2
                    edge_stop = burn + scored_length + 1
                    if np.max(
                        episode.observation_source_index[burn:obs_stop], initial=-1
                    ) >= len(images):
                        raise ValueError(
                            f"Image length in {record.path} is inconsistent with action/time arrays"
                        )
                    stacked_observations: List[np.ndarray] = []
                    for obs_position in range(burn, obs_stop):
                        stack_indices = self._stack_source_indices(
                            episode, obs_position
                        )
                        frames = [
                            self._preprocess_frame(images[int(raw_index)])
                            for raw_index in stack_indices
                        ]
                        stacked_observations.append(np.concatenate(frames, axis=0))
                    result_obs[output_index] = np.stack(stacked_observations, axis=0)
                    actions_out[output_index] = episode.actions[burn:edge_stop]
                    rewards_out[output_index] = episode.rewards[burn:edge_stop]
                    done_out[output_index] = episode.done[burn:edge_stop]
                    truncated_out[output_index] = episode.truncated[burn:edge_stop]
                    initial_prev[output_index] = episode.actions[burn - 1]
                    source_file[output_index] = str(record.path)
                    subject[output_index] = record.subject
                    session[output_index] = record.session
                    block[output_index] = record.block
                    game[output_index] = record.game
                    local_episode[output_index] = episode.episode_index
                    window_start[output_index] = burn
                    decision_times[output_index] = episode.decision_times[burn:obs_stop]
                    observation_source[output_index] = (
                        episode.observation_source_index[burn:obs_stop]
                    )

        if any(observation is None for observation in result_obs):
            raise RuntimeError("Internal error: not every sampled window was materialised")
        obs_seq = np.stack(result_obs, axis=0)  # type: ignore[arg-type]
        score_mask = np.ones(scored_length + 1, dtype=np.bool_)
        score_mask[0] = False
        return {
            "obs_seq": obs_seq,
            "actions_seq": actions_out,
            "initial_prev_action": initial_prev,
            "rewards_seq": rewards_out,
            "done_seq": done_out,
            "truncated_seq": truncated_out,
            "score_mask": score_mask,
            # Manifest fields make train/holdout leakage and exact row identity
            # auditable without opening the source archives again.
            "source_file": np.asarray(source_file),
            "subject": subject,
            "session": session,
            "block": block,
            "game": game,
            "episode_index": local_episode,
            "window_start": window_start,
            "decision_times": decision_times,
            "observation_source_index": observation_source,
        }
