#!/usr/bin/env python3
"""Duration-aware event segmentation of continuous human gameplay.

The analysis unit is one subject x game combination.  Each NPZ episode is an
independent sequence; states never bridge terminal, block, or session
boundaries.  A hard-EM hidden semi-Markov model (HSMM) with state-specific
discrete-Weibull durations is compared with a matched memoryless HMM using
leave-one-session-out likelihood.  The final model is then fit to all
sessions and decoded with exact HSMM forward-backward inference.

Each Pong rally and Space Invaders enemy wave is an independent phase sequence.
A multinomial nuisance model first predicts human control-transition category
from current RAM-derived task state and causal time elapsed since phase onset.
The phase-controlled HMM/HSMM then estimates persistent state-specific logit
offsets relative to that baseline.  Task state and reward are not latent-state
emissions, so inferred modes represent residual control organization rather
than a direct discretization of game phase.

Every CSV contains ``subject`` and ``game`` columns and the group-facing files
under ``group_exports/`` use stable long-form schemas.  Later across-subject
analysis should concatenate those files and align states from their saved
profiles rather than assume that numerical state IDs are equivalent.

Examples
--------
python research_script/04_event_segmentation_hsmm.py --subject 3 --game 1
python research_script/04_event_segmentation_hsmm.py --subject 3 --game 2
python research_script/04_event_segmentation_hsmm.py --subject 3 --game 2 --figures-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from scipy import optimize
from scipy.optimize import linear_sum_assignment
from scipy.special import expit, logsumexp
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score

try:
    from numba import njit
except ImportError as exc:  # pragma: no cover - clear runtime error on other machines
    raise ImportError(
        "04_event_segmentation_hsmm.py requires numba for the explicit-duration "
        "dynamic program. Install it in the analysis environment with "
        "`python -m pip install numba`."
    ) from exc


ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = ROOT / "behavioral_data_block"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs" / "04_event_segmentation_hsmm"
ANALYSIS_VERSION = "04_hsmm_v3_phase_controlled"
RANDOM_SEED = 2026
EPS = 1e-12
NEG_INF = -1.0e300

GAME_NAMES = {1: "Pong", 2: "Space Invaders"}
ACTION_LABELS = ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"]
CONTROL_LABELS = [
    "NOOP persistence",
    "Active persistence",
    "Active to NOOP",
    "NOOP to active",
    "Active switch",
    "Phase onset",
]
REWARD_LABELS = ["No reward", "Positive reward", "Negative reward"]

GAME_FEATURES: Dict[int, Tuple[str, ...]] = {
    1: (
        "ball_x",
        "ball_y",
        "player_y",
        "cpu_y",
        "ball_player_y_distance",
        "ball_cpu_y_distance",
        "delta_ball_x",
        "delta_ball_y",
        "delta_player_y",
    ),
    2: (
        "player_x",
        "enemies_y",
        "enemy_depletion_fraction",
        "lives",
        "delta_player_x",
        "delta_enemies_y",
        "delta_enemy_depletion_fraction",
    ),
}

DATA_FILE_RE = re.compile(
    r"sub(?P<subject>\d+)-ses(?P<session>\d+)-block(?P<block>\d+)-game(?P<game>\d+)\.npz$"
)


def log(message: str, start_time: float) -> None:
    print(f"[{time.time() - start_time:8.1f}s] {message}", flush=True)


def parse_subject(value: str) -> int:
    match = re.search(r"(\d+)", str(value))
    if match is None:
        raise argparse.ArgumentTypeError(f"Could not parse subject from {value!r}")
    return int(match.group(1))


def parse_int_list(value: str) -> List[int]:
    values = sorted({int(token.strip()) for token in value.split(",") if token.strip()})
    if not values:
        raise argparse.ArgumentTypeError("At least one integer is required")
    return values


@dataclass
class SequenceData:
    sequence_id: str
    subject: int
    game: int
    session: int
    block: int
    episode: int
    phase_index: int
    source_file: str
    source_start: int
    source_stop: int
    action: np.ndarray
    previous_action: np.ndarray
    control: np.ndarray
    reward_category: np.ndarray
    reward: np.ndarray
    continuous_raw: np.ndarray
    task_feature_names: Tuple[str, ...]
    episode_progress: np.ndarray
    cycle_id: np.ndarray
    cycle_progress: np.ndarray
    cycle_boundary: np.ndarray
    enemy_count_raw: np.ndarray
    phase_baseline_prob: np.ndarray

    def __len__(self) -> int:
        return int(len(self.action))


@dataclass
class Scaler:
    mean: np.ndarray
    scale: np.ndarray
    feature_names: Tuple[str, ...]

    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.clip((values - self.mean) / self.scale, -8.0, 8.0)


@dataclass
class PhaseBaseline:
    """Multinomial control baseline conditioned only on observable game phase."""

    raw_feature_names: Tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    classes: np.ndarray
    coefficients: np.ndarray
    intercept: np.ndarray

    def to_json_dict(self) -> Dict[str, object]:
        return {
            "raw_feature_names": list(self.raw_feature_names),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "classes": self.classes.tolist(),
            "coefficients": self.coefficients.tolist(),
            "intercept": self.intercept.tolist(),
        }

    @staticmethod
    def from_json_dict(payload: Mapping[str, object]) -> "PhaseBaseline":
        return PhaseBaseline(
            raw_feature_names=tuple(payload["raw_feature_names"]),
            mean=np.asarray(payload["mean"], dtype=float),
            scale=np.asarray(payload["scale"], dtype=float),
            classes=np.asarray(payload["classes"], dtype=int),
            coefficients=np.asarray(payload["coefficients"], dtype=float),
            intercept=np.asarray(payload["intercept"], dtype=float),
        )


@dataclass
class ModelParams:
    family: str
    feature_set: str
    n_states: int
    max_duration: int
    init_prob: np.ndarray
    transition_prob: np.ndarray
    means: np.ndarray
    variances: np.ndarray
    control_prob: np.ndarray
    control_logit_offset: np.ndarray
    reward_prob: np.ndarray
    duration_q: np.ndarray
    duration_beta: np.ndarray
    duration_log_prob: np.ndarray
    scaler: Scaler

    def to_json_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["scaler"]["feature_names"] = list(self.scaler.feature_names)
        for key, value in list(payload.items()):
            if isinstance(value, np.ndarray):
                payload[key] = value.tolist()
        for key, value in list(payload["scaler"].items()):
            if isinstance(value, np.ndarray):
                payload["scaler"][key] = value.tolist()
        return payload

    @staticmethod
    def from_json_dict(payload: Mapping[str, object]) -> "ModelParams":
        scaler_payload = payload["scaler"]
        scaler = Scaler(
            mean=np.asarray(scaler_payload["mean"], dtype=float),
            scale=np.asarray(scaler_payload["scale"], dtype=float),
            feature_names=tuple(scaler_payload["feature_names"]),
        )
        return ModelParams(
            family=str(payload["family"]),
            feature_set=str(payload["feature_set"]),
            n_states=int(payload["n_states"]),
            max_duration=int(payload["max_duration"]),
            init_prob=np.asarray(payload["init_prob"], dtype=float),
            transition_prob=np.asarray(payload["transition_prob"], dtype=float),
            means=np.asarray(payload["means"], dtype=float),
            variances=np.asarray(payload["variances"], dtype=float),
            control_prob=np.asarray(payload["control_prob"], dtype=float),
            control_logit_offset=np.asarray(
                payload.get(
                    "control_logit_offset",
                    np.zeros_like(np.asarray(payload["control_prob"], dtype=float)),
                ),
                dtype=float,
            ),
            reward_prob=np.asarray(payload["reward_prob"], dtype=float),
            duration_q=np.asarray(payload["duration_q"], dtype=float),
            duration_beta=np.asarray(payload["duration_beta"], dtype=float),
            duration_log_prob=np.asarray(payload["duration_log_prob"], dtype=float),
            scaler=scaler,
        )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--subject", required=True, type=parse_subject)
    parser.add_argument("--game", required=True, type=int, choices=(1, 2))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--k-values",
        type=parse_int_list,
        default=[4, 5, 6, 7, 8],
        help="Comma-separated candidate state counts (default: 4,5,6,7,8).",
    )
    parser.add_argument(
        "--max-duration",
        type=int,
        default=512,
        help="Maximum explicit state duration in real steps (default: 512).",
    )
    parser.add_argument("--cv-iterations", type=int, default=2)
    parser.add_argument("--final-iterations", type=int, default=4)
    parser.add_argument("--cv-n-init", type=int, default=1)
    parser.add_argument("--n-init", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=RANDOM_SEED)
    parser.add_argument("--min-sequence-length", type=int, default=20)
    parser.add_argument("--event-window", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--reuse-cv",
        action="store_true",
        help=(
            "Reuse an existing compatible session-held-out model-selection table "
            "and refit only the selected final model."
        ),
    )
    parser.add_argument("--figures-only", action="store_true")
    return parser


def _control_categories(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    action = np.asarray(action, dtype=int)
    previous = np.full(len(action), -1, dtype=int)
    if len(action) > 1:
        previous[1:] = action[:-1]
    control = np.full(len(action), 5, dtype=np.int16)
    for t in range(1, len(action)):
        current = int(action[t])
        prior = int(previous[t])
        if current == 0 and prior == 0:
            control[t] = 0
        elif current > 0 and current == prior:
            control[t] = 1
        elif current == 0 and prior > 0:
            control[t] = 2
        elif current > 0 and prior == 0:
            control[t] = 3
        elif current > 0 and prior > 0 and current != prior:
            control[t] = 4
        else:
            control[t] = 5
    return previous, control


def _safe_delta(values: np.ndarray) -> np.ndarray:
    result = np.zeros(len(values), dtype=float)
    if len(values) > 1:
        result[1:] = np.diff(values.astype(float))
    return result


def _cycle_metadata(
    ram: np.ndarray, reward: np.ndarray, game: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return visualization-only progress variables for one episode.

    ``episode_progress`` and the cycle identifiers are never supplied to the
    HSMM.  Pong cycles are reward-delimited rallies and use elapsed fraction
    within each rally.  Space Invaders cycles are enemy waves and use the
    fraction of enemies removed; refill jumps start a new wave.  This keeps
    later waves comparable with the first wave without encoding absolute wave
    number as a monotonic proxy for elapsed time.
    """
    n = len(ram)
    episode_progress = (
        np.arange(n, dtype=float) / max(n - 1, 1) if n else np.empty(0, dtype=float)
    )
    cycle_start = np.zeros(n, dtype=bool)
    if n:
        cycle_start[0] = True
    enemy_count_raw = np.full(n, np.nan, dtype=float)

    if game == 1:
        # A scored point ends a rally; the next step begins the next rally.
        if n > 1:
            cycle_start[1:] = np.asarray(reward[:-1] != 0, dtype=bool)
    else:
        enemy_count_raw = np.asarray(ram[:, 17], dtype=float)
        if n > 1:
            # True refills are large positive jumps (usually 0 -> 36).  The
            # threshold also tolerates an episode that restarts from a partly
            # depleted RAM snapshot, while episode boundaries are handled
            # separately by sequence splitting.
            cycle_start[1:] = np.diff(enemy_count_raw) >= 5

    cycle_id = np.cumsum(cycle_start).astype(int) - 1
    cycle_progress = np.zeros(n, dtype=float)
    for cycle in np.unique(cycle_id):
        indices = np.flatnonzero(cycle_id == cycle)
        if not len(indices):
            continue
        if game == 1:
            cycle_progress[indices] = np.arange(len(indices), dtype=float) / max(
                len(indices) - 1, 1
            )
        else:
            wave_capacity = max(float(np.nanmax(enemy_count_raw[indices])), 1.0)
            cycle_progress[indices] = np.clip(
                1.0 - enemy_count_raw[indices] / wave_capacity, 0.0, 1.0
            )
    cycle_boundary = cycle_start.copy()
    if n:
        cycle_boundary[0] = False
    return (
        episode_progress,
        cycle_id,
        cycle_progress,
        cycle_boundary,
        enemy_count_raw,
    )


def _task_features(
    ram: np.ndarray, game: int, cycle_progress: np.ndarray
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    ram = np.asarray(ram, dtype=float)
    if game == 1:
        ball_x = ram[:, 49]
        ball_y = ram[:, 54]
        player_y = ram[:, 51]
        cpu_y = ram[:, 50]
        values = np.column_stack(
            [
                ball_x,
                ball_y,
                player_y,
                cpu_y,
                np.abs(ball_y - player_y),
                np.abs(ball_y - cpu_y),
                _safe_delta(ball_x),
                _safe_delta(ball_y),
                _safe_delta(player_y),
            ]
        )
    else:
        player_x = ram[:, 28]
        enemies_y = ram[:, 16]
        lives = ram[:, 73]
        values = np.column_stack(
            [
                player_x,
                enemies_y,
                cycle_progress,
                lives,
                _safe_delta(player_x),
                _safe_delta(enemies_y),
                _safe_delta(cycle_progress),
            ]
        )
    return values.astype(np.float64), GAME_FEATURES[game]


def load_sequences(
    data_root: Path,
    subject: int,
    game: int,
    min_sequence_length: int,
) -> Tuple[List[SequenceData], pd.DataFrame]:
    subject_dir = data_root / f"sub-{subject:03d}"
    paths = sorted(subject_dir.glob(f"ses-*/*game{game}.npz"))
    if not paths:
        raise FileNotFoundError(
            f"No behavioral NPZ files for sub{subject:03d}, game {game} under {data_root}"
        )
    sequences: List[SequenceData] = []
    index_rows: List[Dict[str, object]] = []
    for path in paths:
        match = DATA_FILE_RE.match(path.name)
        if match is None or int(match.group("subject")) != subject:
            continue
        session = int(match.group("session"))
        block = int(match.group("block"))
        payload = np.load(path, allow_pickle=True)
        action = np.argmax(np.asarray(payload["action"], dtype=float), axis=1).astype(np.int16)
        reward = np.asarray(payload["reward"], dtype=float).reshape(-1)
        terminal = np.asarray(payload["is_terminal"], dtype=bool).reshape(-1)
        first = np.asarray(payload["is_first"], dtype=bool).reshape(-1)
        ram = np.asarray(payload["RAM"], dtype=np.uint8)
        n = min(len(action), len(reward), len(terminal), len(first), len(ram))
        action, reward, terminal, first, ram = (
            action[:n], reward[:n], terminal[:n], first[:n], ram[:n]
        )
        starts = sorted({0, *np.flatnonzero(first).astype(int).tolist()})
        stops: List[int] = []
        for start_idx, start in enumerate(starts):
            next_start = starts[start_idx + 1] if start_idx + 1 < len(starts) else n
            terminal_candidates = np.flatnonzero(terminal[start:next_start])
            if len(terminal_candidates):
                stops.append(start + int(terminal_candidates[0]) + 1)
            else:
                stops.append(next_start)
        for episode, (start, stop) in enumerate(zip(starts, stops)):
            if stop - start < min_sequence_length:
                continue
            ep_action = action[start:stop].copy()
            ep_reward = reward[start:stop].copy()
            ep_ram = ram[start:stop].copy()
            previous, control = _control_categories(ep_action)
            reward_category = np.zeros(len(ep_reward), dtype=np.int16)
            reward_category[ep_reward > 0] = 1
            reward_category[ep_reward < 0] = 2
            (
                episode_progress,
                cycle_id,
                cycle_progress,
                cycle_boundary,
                enemy_count_raw,
            ) = _cycle_metadata(ep_ram, ep_reward, game)
            continuous, names = _task_features(ep_ram, game, cycle_progress)
            phase_starts = np.r_[0, np.flatnonzero(cycle_boundary)].astype(int)
            phase_starts = np.unique(phase_starts)
            phase_stops = np.r_[phase_starts[1:], len(ep_action)].astype(int)
            for phase_number, (phase_start, phase_stop) in enumerate(
                zip(phase_starts, phase_stops)
            ):
                if phase_stop - phase_start < min_sequence_length:
                    continue
                phase_slice = slice(phase_start, phase_stop)
                phase_action = ep_action[phase_slice].copy()
                phase_previous, phase_control = _control_categories(phase_action)
                phase_reward = ep_reward[phase_slice].copy()
                phase_reward_category = np.zeros(len(phase_reward), dtype=np.int16)
                phase_reward_category[phase_reward > 0] = 1
                phase_reward_category[phase_reward < 0] = 2
                phase_index = int(cycle_id[phase_start])
                sequence_id = (
                    f"sub{subject:03d}_game{game}_ses{session:02d}_"
                    f"block{block:02d}_episode{episode:02d}_phase{phase_index:02d}"
                )
                sequence = SequenceData(
                    sequence_id=sequence_id,
                    subject=subject,
                    game=game,
                    session=session,
                    block=block,
                    episode=episode,
                    phase_index=phase_index,
                    source_file=str(path.resolve()),
                    source_start=int(start + phase_start),
                    source_stop=int(start + phase_stop),
                    action=phase_action,
                    previous_action=phase_previous,
                    control=phase_control,
                    reward_category=phase_reward_category,
                    reward=phase_reward,
                    continuous_raw=continuous[phase_slice].copy(),
                    task_feature_names=names,
                    episode_progress=episode_progress[phase_slice].copy(),
                    cycle_id=cycle_id[phase_slice].copy(),
                    cycle_progress=cycle_progress[phase_slice].copy(),
                    cycle_boundary=cycle_boundary[phase_slice].copy(),
                    enemy_count_raw=enemy_count_raw[phase_slice].copy(),
                    phase_baseline_prob=np.full(
                        (len(phase_action), len(CONTROL_LABELS)),
                        1.0 / len(CONTROL_LABELS),
                        dtype=float,
                    ),
                )
                sequences.append(sequence)
                index_rows.append(
                    {
                        "analysis_version": ANALYSIS_VERSION,
                        "subject": subject,
                        "game": game,
                        "game_name": GAME_NAMES[game],
                        "session": session,
                        "block": block,
                        "episode": episode,
                        "phase_index": phase_index,
                        "phase_type": "rally" if game == 1 else "enemy_wave",
                        "sequence_id": sequence_id,
                        "n_steps": len(sequence),
                        "source_file": str(path.resolve()),
                        "source_start": int(start + phase_start),
                        "source_stop": int(start + phase_stop),
                    }
                )
    if not sequences:
        raise RuntimeError("No eligible episode sequences remained after filtering")
    return sequences, pd.DataFrame(index_rows)


def fit_scaler(sequences: Sequence[SequenceData]) -> Scaler:
    values = np.concatenate([sequence.continuous_raw for sequence in sequences], axis=0)
    mean = np.nanmean(values, axis=0)
    scale = np.nanstd(values, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    return Scaler(mean=mean, scale=scale, feature_names=sequences[0].task_feature_names)


def _phase_raw_covariates(sequence: SequenceData) -> np.ndarray:
    """Causal phase covariates; normalized phase completion is excluded."""
    elapsed = np.log1p(np.arange(len(sequence), dtype=float))[:, None]
    return np.column_stack([sequence.continuous_raw, elapsed])


def _phase_baseline_design(
    sequence: SequenceData, baseline: PhaseBaseline
) -> np.ndarray:
    raw = _phase_raw_covariates(sequence)
    standardized = np.clip((raw - baseline.mean) / baseline.scale, -8.0, 8.0)
    return np.column_stack([standardized, standardized * standardized])


def fit_phase_baseline(sequences: Sequence[SequenceData]) -> PhaseBaseline:
    """Fit the nuisance action-transition model without latent-state variables."""
    raw = np.concatenate([_phase_raw_covariates(sequence) for sequence in sequences])
    outcome = np.concatenate([sequence.control for sequence in sequences]).astype(int)
    mean = np.nanmean(raw, axis=0)
    scale = np.nanstd(raw, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    standardized = np.clip((raw - mean) / scale, -8.0, 8.0)
    design = np.column_stack([standardized, standardized * standardized])
    classifier = LogisticRegression(
        C=0.1,
        solver="newton-cg",
        max_iter=100,
        tol=1e-4,
    )
    classifier.fit(design, outcome)
    return PhaseBaseline(
        raw_feature_names=tuple(
            [*sequences[0].task_feature_names, "log_elapsed_since_phase_onset"]
        ),
        mean=mean,
        scale=scale,
        classes=classifier.classes_.astype(int),
        coefficients=np.asarray(classifier.coef_, dtype=float),
        intercept=np.asarray(classifier.intercept_, dtype=float),
    )


def predict_phase_baseline(
    sequence: SequenceData, baseline: PhaseBaseline
) -> np.ndarray:
    design = _phase_baseline_design(sequence, baseline)
    probability = np.zeros((len(sequence), len(CONTROL_LABELS)), dtype=float)
    linear = design @ baseline.coefficients.T + baseline.intercept
    if len(baseline.classes) == 2 and linear.shape[1] == 1:
        positive = expit(linear[:, 0])
        probability[:, int(baseline.classes[0])] = 1.0 - positive
        probability[:, int(baseline.classes[1])] = positive
    else:
        linear -= logsumexp(linear, axis=1, keepdims=True)
        probability[:, baseline.classes.astype(int)] = np.exp(linear)
    # Finite support for a category absent from a training fold.
    probability = probability + 1e-5
    probability /= probability.sum(axis=1, keepdims=True)
    return probability


def apply_phase_baseline(
    sequences: Sequence[SequenceData], baseline: PhaseBaseline
) -> List[SequenceData]:
    return [
        replace(
            sequence,
            phase_baseline_prob=predict_phase_baseline(sequence, baseline),
        )
        for sequence in sequences
    ]


def phase_baseline_log_likelihood(sequences: Sequence[SequenceData]) -> float:
    return float(
        sum(
            np.log(
                np.clip(
                    sequence.phase_baseline_prob[
                        np.arange(len(sequence)), sequence.control
                    ],
                    EPS,
                    1.0,
                )
            ).sum()
            for sequence in sequences
        )
    )


def build_phase_baseline_coefficients(
    baseline: PhaseBaseline, subject: int, game: int
) -> pd.DataFrame:
    feature_names = [
        *baseline.raw_feature_names,
        *[f"{name}_squared" for name in baseline.raw_feature_names],
    ]
    rows: List[Dict[str, object]] = []
    if len(baseline.classes) == 2 and baseline.coefficients.shape[0] == 1:
        coefficient_rows = [(int(baseline.classes[1]), baseline.coefficients[0])]
    else:
        coefficient_rows = [
            (int(category), values)
            for category, values in zip(baseline.classes, baseline.coefficients)
        ]
    for category, values in coefficient_rows:
        for feature, value in zip(feature_names, values):
            rows.append(
                {
                    "analysis_version": ANALYSIS_VERSION,
                    "subject": subject,
                    "game": game,
                    "game_name": GAME_NAMES[game],
                    "control_category": category,
                    "control_label": CONTROL_LABELS[category],
                    "phase_feature": feature,
                    "coefficient": float(value),
                }
            )
    return pd.DataFrame(rows)


def _model_features(
    sequences: Sequence[SequenceData],
    scaler: Scaler,
    feature_set: str,
) -> List[np.ndarray]:
    output: List[np.ndarray] = []
    for sequence in sequences:
        if feature_set != "phase_controlled":
            raise ValueError(f"Unsupported feature set: {feature_set}")
        residual = (
            np.eye(len(CONTROL_LABELS), dtype=float)[sequence.control]
            - sequence.phase_baseline_prob
        )
        columns = [residual]
        output.append(np.column_stack(columns))
    return output


def _initial_labels(
    sequences: Sequence[SequenceData],
    scaler: Scaler,
    feature_set: str,
    n_states: int,
    seed: int,
) -> List[np.ndarray]:
    feature_arrays = _model_features(sequences, scaler, feature_set)
    all_features = np.concatenate(feature_arrays, axis=0)
    rng = np.random.default_rng(seed)
    if len(all_features) > 60_000:
        sample_idx = rng.choice(len(all_features), size=60_000, replace=False)
        fit_values = all_features[sample_idx]
    else:
        fit_values = all_features
    clusterer = MiniBatchKMeans(
        n_clusters=n_states,
        random_state=seed,
        n_init=3,
        batch_size=4096,
        max_iter=100,
    ).fit(fit_values)
    return [clusterer.predict(values).astype(np.int16) for values in feature_arrays]


def _logit(value: float) -> float:
    value = float(np.clip(value, 1e-8, 1.0 - 1e-8))
    return float(np.log(value / (1.0 - value)))


def _fit_control_logit_offset(
    baseline_probability: np.ndarray,
    control: np.ndarray,
    regularization: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate a state-specific multinomial offset from phase baseline logits."""
    baseline_probability = np.clip(
        np.asarray(baseline_probability, dtype=float), EPS, 1.0
    )
    control = np.asarray(control, dtype=int)
    n_categories = baseline_probability.shape[1]
    if len(control) == 0:
        return np.zeros(n_categories), np.full(n_categories, 1.0 / n_categories)
    log_baseline = np.log(baseline_probability)

    def objective(offset: np.ndarray) -> Tuple[float, np.ndarray]:
        centered = offset - offset.mean()
        logits = log_baseline + centered[None, :]
        log_normalizer = logsumexp(logits, axis=1)
        loss = -float(
            np.sum(logits[np.arange(len(control)), control] - log_normalizer)
        )
        probability = np.exp(logits - log_normalizer[:, None])
        gradient = probability.sum(axis=0) - np.bincount(
            control, minlength=n_categories
        )
        gradient += regularization * centered
        gradient -= gradient.mean()
        loss += 0.5 * regularization * float(np.sum(centered * centered))
        return loss, gradient

    result = optimize.minimize(
        objective,
        np.zeros(n_categories, dtype=float),
        method="L-BFGS-B",
        jac=True,
    )
    offset = np.asarray(result.x if result.success else np.zeros(n_categories), float)
    offset -= offset.mean()
    logits = log_baseline + offset[None, :]
    adjusted = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
    return offset, adjusted.mean(axis=0)


def _conditional_control_probability(
    sequence: SequenceData, model: ModelParams
) -> np.ndarray:
    """Return time x state x control probability after phase adjustment."""
    logits = (
        np.log(np.clip(sequence.phase_baseline_prob, EPS, 1.0))[:, None, :]
        + model.control_logit_offset[None, :, :]
    )
    return np.exp(logits - logsumexp(logits, axis=2, keepdims=True))


def _fit_discrete_weibull(lengths: np.ndarray, events: np.ndarray) -> Tuple[float, float]:
    lengths = np.maximum(np.asarray(lengths, dtype=int), 1)
    events = np.asarray(events, dtype=bool)
    hazard = float(np.sum(events) / max(np.sum(lengths), 1))
    q0 = float(np.clip(1.0 - hazard, 1e-6, 1.0 - 1e-6))

    def objective(theta: np.ndarray) -> float:
        q = float(expit(theta[0]))
        beta = float(np.exp(theta[1]))
        log_q = math.log(float(np.clip(q, 1e-9, 1 - 1e-9)))
        previous = np.maximum(lengths.astype(float) - 1.0, 0.0) ** beta * log_q
        current = lengths.astype(float) ** beta * log_q
        ratio = np.minimum(current - previous, -1e-14)
        log_event = previous + np.log(-np.expm1(ratio))
        ll = np.where(events, log_event, current).sum()
        return float(-ll)

    best = None
    for beta0 in (0.6, 1.0, 1.8):
        result = optimize.minimize(
            objective,
            np.array([_logit(q0), math.log(beta0)]),
            method="L-BFGS-B",
            bounds=[(-12.0, 12.0), (math.log(0.15), math.log(5.0))],
        )
        if np.isfinite(result.fun) and (best is None or result.fun < best.fun):
            best = result
    if best is None:
        return q0, 1.0
    return float(expit(best.x[0])), float(np.exp(best.x[1]))


def _duration_log_prob(q: float, beta: float, max_duration: int) -> np.ndarray:
    times = np.arange(1, max_duration + 1, dtype=float)
    log_q = math.log(float(np.clip(q, 1e-9, 1.0 - 1e-9)))
    log_previous = np.maximum(times - 1.0, 0.0) ** beta * log_q
    log_current = times**beta * log_q
    result = log_previous + np.log(-np.expm1(np.minimum(log_current - log_previous, -1e-14)))
    # The last duration bin contains the remaining tail mass.
    result[-1] = log_previous[-1]
    result -= logsumexp(result)
    return result


def _run_segments(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int16)
    if len(labels) == 0:
        return (
            np.empty(0, dtype=np.int16),
            np.empty(0, dtype=int),
            np.empty(0, dtype=int),
        )
    starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
    stops = np.r_[starts[1:], len(labels)]
    return labels[starts], starts.astype(int), stops.astype(int)


def estimate_params(
    sequences: Sequence[SequenceData],
    labels: Sequence[np.ndarray],
    scaler: Scaler,
    feature_set: str,
    family: str,
    n_states: int,
    max_duration: int,
) -> ModelParams:
    continuous = [scaler.transform(sequence.continuous_raw) for sequence in sequences]
    n_features = continuous[0].shape[1]
    means = np.zeros((n_states, n_features), dtype=float)
    variances = np.ones((n_states, n_features), dtype=float)
    control_prob = np.full((n_states, len(CONTROL_LABELS)), 1.0 / len(CONTROL_LABELS))
    control_logit_offset = np.zeros((n_states, len(CONTROL_LABELS)), dtype=float)
    reward_prob = np.full((n_states, len(REWARD_LABELS)), 1.0 / len(REWARD_LABELS))
    global_values = np.concatenate(continuous, axis=0)
    global_mean = np.mean(global_values, axis=0)
    global_var = np.maximum(np.var(global_values, axis=0), 0.05)
    init_counts = np.full(n_states, 0.5, dtype=float)
    transition_counts = np.full((n_states, n_states), 0.25, dtype=float)
    duration_values: List[List[int]] = [[] for _ in range(n_states)]
    duration_events: List[List[bool]] = [[] for _ in range(n_states)]

    for k in range(n_states):
        state_values = [values[state_labels == k] for values, state_labels in zip(continuous, labels)]
        state_values = [values for values in state_values if len(values)]
        if state_values:
            joined = np.concatenate(state_values, axis=0)
            means[k] = np.mean(joined, axis=0)
            variances[k] = np.maximum(np.var(joined, axis=0), 0.05)
        else:
            means[k] = global_mean
            variances[k] = global_var

        controls = np.concatenate(
            [sequence.control[state_labels == k] for sequence, state_labels in zip(sequences, labels)]
        )
        baseline_probability = np.concatenate(
            [
                sequence.phase_baseline_prob[state_labels == k]
                for sequence, state_labels in zip(sequences, labels)
            ],
            axis=0,
        )
        rewards = np.concatenate(
            [sequence.reward_category[state_labels == k] for sequence, state_labels in zip(sequences, labels)]
        )
        if len(controls):
            if feature_set == "phase_controlled":
                offset, adjusted_probability = _fit_control_logit_offset(
                    baseline_probability, controls
                )
                control_logit_offset[k] = offset
                control_prob[k] = adjusted_probability
            else:
                counts = np.bincount(
                    controls, minlength=len(CONTROL_LABELS)
                ).astype(float) + 0.5
                control_prob[k] = counts / counts.sum()
        if len(rewards):
            counts = np.bincount(rewards, minlength=len(REWARD_LABELS)).astype(float) + 0.5
            reward_prob[k] = counts / counts.sum()

    for state_labels in labels:
        init_counts[int(state_labels[0])] += 1.0
        if family == "hmm":
            np.add.at(transition_counts, (state_labels[:-1], state_labels[1:]), 1.0)
        else:
            states, starts, stops = _run_segments(state_labels)
            for index, state in enumerate(states):
                duration_values[int(state)].append(int(stops[index] - starts[index]))
                duration_events[int(state)].append(index < len(states) - 1)
                if index + 1 < len(states):
                    transition_counts[int(state), int(states[index + 1])] += 1.0

    init_prob = init_counts / init_counts.sum()
    if family == "hsmm":
        np.fill_diagonal(transition_counts, 0.0)
        for k in range(n_states):
            if transition_counts[k].sum() <= 0:
                transition_counts[k] = 1.0
                transition_counts[k, k] = 0.0
    transition_prob = transition_counts / np.maximum(transition_counts.sum(axis=1, keepdims=True), EPS)

    duration_q = np.zeros(n_states, dtype=float)
    duration_beta = np.ones(n_states, dtype=float)
    duration_logs = np.full((n_states, max_duration), -math.log(max_duration), dtype=float)
    for k in range(n_states):
        if duration_values[k]:
            q, beta = _fit_discrete_weibull(
                np.asarray(duration_values[k]), np.asarray(duration_events[k])
            )
        else:
            q, beta = 0.8, 1.0
        duration_q[k] = q
        duration_beta[k] = beta
        duration_logs[k] = _duration_log_prob(q, beta, max_duration)

    return ModelParams(
        family=family,
        feature_set=feature_set,
        n_states=n_states,
        max_duration=max_duration,
        init_prob=init_prob,
        transition_prob=transition_prob,
        means=means,
        variances=variances,
        control_prob=control_prob,
        control_logit_offset=control_logit_offset,
        reward_prob=reward_prob,
        duration_q=duration_q,
        duration_beta=duration_beta,
        duration_log_prob=duration_logs,
        scaler=scaler,
    )


def emission_log_likelihood(sequence: SequenceData, model: ModelParams) -> np.ndarray:
    if model.feature_set == "phase_controlled":
        probability = _conditional_control_probability(sequence, model)
        observed = probability[
            np.arange(len(sequence))[:, None],
            np.arange(model.n_states)[None, :],
            sequence.control[:, None],
        ]
        return np.log(np.clip(observed, EPS, 1.0)).astype(np.float64)
    values = model.scaler.transform(sequence.continuous_raw)
    difference = values[:, None, :] - model.means[None, :, :]
    gaussian = -0.5 * np.sum(
        np.log(2.0 * np.pi * model.variances)[None, :, :]
        + difference * difference / model.variances[None, :, :],
        axis=2,
    )
    raise ValueError(f"Unsupported feature set: {model.feature_set}")


@njit(cache=False)
def _logadd(a: float, b: float) -> float:
    if a <= NEG_INF / 2:
        return b
    if b <= NEG_INF / 2:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


@njit(cache=False)
def _hsmm_viterbi(
    emission: np.ndarray,
    log_init: np.ndarray,
    log_transition: np.ndarray,
    log_duration: np.ndarray,
) -> Tuple[np.ndarray, float]:
    n_steps, n_states = emission.shape
    max_duration = log_duration.shape[1]
    prefix = np.zeros((n_steps + 1, n_states), dtype=np.float64)
    for t in range(n_steps):
        for k in range(n_states):
            prefix[t + 1, k] = prefix[t, k] + emission[t, k]
    score = np.full((n_steps, n_states), NEG_INF, dtype=np.float64)
    best_duration = np.ones((n_steps, n_states), dtype=np.int32)
    best_previous = np.full((n_steps, n_states), -1, dtype=np.int16)
    for end in range(n_steps):
        duration_limit = min(max_duration, end + 1)
        for k in range(n_states):
            best = NEG_INF
            chosen_duration = 1
            chosen_previous = -1
            for duration in range(1, duration_limit + 1):
                start = end - duration + 1
                segment_score = prefix[end + 1, k] - prefix[start, k]
                if start == 0:
                    candidate = log_init[k] + log_duration[k, duration - 1] + segment_score
                    previous = -1
                else:
                    prior_best = NEG_INF
                    previous = -1
                    for j in range(n_states):
                        candidate_prior = score[start - 1, j] + log_transition[j, k]
                        if candidate_prior > prior_best:
                            prior_best = candidate_prior
                            previous = j
                    candidate = prior_best + log_duration[k, duration - 1] + segment_score
                if candidate > best:
                    best = candidate
                    chosen_duration = duration
                    chosen_previous = previous
            score[end, k] = best
            best_duration[end, k] = chosen_duration
            best_previous[end, k] = chosen_previous
    final_state = 0
    final_score = score[n_steps - 1, 0]
    for k in range(1, n_states):
        if score[n_steps - 1, k] > final_score:
            final_score = score[n_steps - 1, k]
            final_state = k
    labels = np.zeros(n_steps, dtype=np.int16)
    end = n_steps - 1
    state = final_state
    while end >= 0:
        duration = best_duration[end, state]
        start = end - duration + 1
        for t in range(start, end + 1):
            labels[t] = state
        previous = best_previous[end, state]
        end = start - 1
        state = previous
        if end >= 0 and state < 0:
            state = 0
    return labels, final_score


@njit(cache=False)
def _hmm_viterbi(
    emission: np.ndarray,
    log_init: np.ndarray,
    log_transition: np.ndarray,
) -> Tuple[np.ndarray, float]:
    n_steps, n_states = emission.shape
    score = np.empty((n_steps, n_states), dtype=np.float64)
    previous = np.zeros((n_steps, n_states), dtype=np.int16)
    for k in range(n_states):
        score[0, k] = log_init[k] + emission[0, k]
    for t in range(1, n_steps):
        for k in range(n_states):
            best = score[t - 1, 0] + log_transition[0, k]
            chosen = 0
            for j in range(1, n_states):
                candidate = score[t - 1, j] + log_transition[j, k]
                if candidate > best:
                    best = candidate
                    chosen = j
            score[t, k] = best + emission[t, k]
            previous[t, k] = chosen
    state = 0
    final_score = score[n_steps - 1, 0]
    for k in range(1, n_states):
        if score[n_steps - 1, k] > final_score:
            final_score = score[n_steps - 1, k]
            state = k
    labels = np.zeros(n_steps, dtype=np.int16)
    labels[-1] = state
    for t in range(n_steps - 1, 0, -1):
        state = previous[t, state]
        labels[t - 1] = state
    return labels, final_score


@njit(cache=False)
def _hmm_forward(
    emission: np.ndarray,
    log_init: np.ndarray,
    log_transition: np.ndarray,
) -> float:
    n_steps, n_states = emission.shape
    alpha = np.empty((n_steps, n_states), dtype=np.float64)
    for k in range(n_states):
        alpha[0, k] = log_init[k] + emission[0, k]
    for t in range(1, n_steps):
        for k in range(n_states):
            total = NEG_INF
            for j in range(n_states):
                total = _logadd(total, alpha[t - 1, j] + log_transition[j, k])
            alpha[t, k] = total + emission[t, k]
    total = NEG_INF
    for k in range(n_states):
        total = _logadd(total, alpha[n_steps - 1, k])
    return total


@njit(cache=False)
def _hsmm_forward_arrays(
    emission: np.ndarray,
    log_init: np.ndarray,
    log_transition: np.ndarray,
    log_duration: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    n_steps, n_states = emission.shape
    max_duration = log_duration.shape[1]
    prefix = np.zeros((n_steps + 1, n_states), dtype=np.float64)
    for t in range(n_steps):
        for k in range(n_states):
            prefix[t + 1, k] = prefix[t, k] + emission[t, k]
    alpha = np.full((n_steps, n_states), NEG_INF, dtype=np.float64)
    for end in range(n_steps):
        duration_limit = min(max_duration, end + 1)
        for k in range(n_states):
            total = NEG_INF
            for duration in range(1, duration_limit + 1):
                start = end - duration + 1
                segment_score = prefix[end + 1, k] - prefix[start, k]
                if start == 0:
                    candidate = log_init[k] + log_duration[k, duration - 1] + segment_score
                else:
                    prior = NEG_INF
                    for j in range(n_states):
                        prior = _logadd(prior, alpha[start - 1, j] + log_transition[j, k])
                    candidate = prior + log_duration[k, duration - 1] + segment_score
                total = _logadd(total, candidate)
            alpha[end, k] = total
    log_likelihood = NEG_INF
    for k in range(n_states):
        log_likelihood = _logadd(log_likelihood, alpha[n_steps - 1, k])
    return alpha, prefix, log_likelihood


@njit(cache=False)
def _hsmm_forward(
    emission: np.ndarray,
    log_init: np.ndarray,
    log_transition: np.ndarray,
    log_duration: np.ndarray,
) -> float:
    _, _, log_likelihood = _hsmm_forward_arrays(
        emission, log_init, log_transition, log_duration
    )
    return log_likelihood


@njit(cache=False)
def _hsmm_posterior(
    emission: np.ndarray,
    log_init: np.ndarray,
    log_transition: np.ndarray,
    log_duration: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    alpha, prefix, log_likelihood = _hsmm_forward_arrays(
        emission, log_init, log_transition, log_duration
    )
    n_steps, n_states = emission.shape
    max_duration = log_duration.shape[1]
    beta = np.full((n_steps, n_states), NEG_INF, dtype=np.float64)
    for k in range(n_states):
        beta[n_steps - 1, k] = 0.0
    for end in range(n_steps - 2, -1, -1):
        duration_limit = min(max_duration, n_steps - end - 1)
        for previous_state in range(n_states):
            total = NEG_INF
            for next_state in range(n_states):
                transition = log_transition[previous_state, next_state]
                if transition <= NEG_INF / 2:
                    continue
                for duration in range(1, duration_limit + 1):
                    next_end = end + duration
                    segment_score = prefix[next_end + 1, next_state] - prefix[end + 1, next_state]
                    candidate = (
                        transition
                        + log_duration[next_state, duration - 1]
                        + segment_score
                        + beta[next_end, next_state]
                    )
                    total = _logadd(total, candidate)
            beta[end, previous_state] = total

    gamma_difference = np.zeros((n_steps + 1, n_states), dtype=np.float64)
    boundary = np.zeros(n_steps, dtype=np.float64)
    boundary[0] = 1.0
    for end in range(n_steps):
        duration_limit = min(max_duration, end + 1)
        for state in range(n_states):
            for duration in range(1, duration_limit + 1):
                start = end - duration + 1
                segment_score = prefix[end + 1, state] - prefix[start, state]
                if start == 0:
                    prior = log_init[state]
                else:
                    prior = NEG_INF
                    for previous_state in range(n_states):
                        prior = _logadd(
                            prior,
                            alpha[start - 1, previous_state]
                            + log_transition[previous_state, state],
                        )
                log_probability = (
                    prior
                    + log_duration[state, duration - 1]
                    + segment_score
                    + beta[end, state]
                    - log_likelihood
                )
                if log_probability > -40.0:
                    probability = math.exp(log_probability)
                    gamma_difference[start, state] += probability
                    gamma_difference[end + 1, state] -= probability
                    if start > 0:
                        boundary[start] += probability
    gamma = np.zeros((n_steps, n_states), dtype=np.float64)
    for state in range(n_states):
        running = 0.0
        for t in range(n_steps):
            running += gamma_difference[t, state]
            gamma[t, state] = running
    for t in range(n_steps):
        total = 0.0
        for state in range(n_states):
            if gamma[t, state] < 0.0:
                gamma[t, state] = 0.0
            total += gamma[t, state]
        if total > 0.0:
            for state in range(n_states):
                gamma[t, state] /= total
        boundary[t] = min(max(boundary[t], 0.0), 1.0)
    return gamma, boundary, log_likelihood


def _log_parameters(model: ModelParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    log_init = np.log(np.clip(model.init_prob, EPS, 1.0))
    log_transition = np.full_like(model.transition_prob, NEG_INF, dtype=float)
    valid = model.transition_prob > 0
    log_transition[valid] = np.log(model.transition_prob[valid])
    return log_init, log_transition, model.duration_log_prob


def decode_sequences(
    sequences: Sequence[SequenceData], model: ModelParams
) -> Tuple[List[np.ndarray], float]:
    log_init, log_transition, log_duration = _log_parameters(model)
    labels: List[np.ndarray] = []
    total_score = 0.0
    for sequence in sequences:
        emission = emission_log_likelihood(sequence, model)
        if model.family == "hsmm":
            state_labels, score = _hsmm_viterbi(
                emission, log_init, log_transition, log_duration
            )
        else:
            state_labels, score = _hmm_viterbi(emission, log_init, log_transition)
        labels.append(state_labels)
        total_score += float(score)
    return labels, total_score


def score_sequences(sequences: Sequence[SequenceData], model: ModelParams) -> float:
    log_init, log_transition, log_duration = _log_parameters(model)
    total = 0.0
    for sequence in sequences:
        emission = emission_log_likelihood(sequence, model)
        if model.family == "hsmm":
            total += float(_hsmm_forward(emission, log_init, log_transition, log_duration))
        else:
            total += float(_hmm_forward(emission, log_init, log_transition))
    return total


def fit_model(
    sequences: Sequence[SequenceData],
    feature_set: str,
    family: str,
    n_states: int,
    max_duration: int,
    n_iterations: int,
    n_init: int,
    random_state: int,
) -> Tuple[ModelParams, List[np.ndarray], pd.DataFrame]:
    scaler = fit_scaler(sequences)
    candidates: List[Tuple[float, ModelParams, List[np.ndarray]]] = []
    diagnostics: List[Dict[str, object]] = []
    for initialization in range(n_init):
        seed = random_state + initialization * 997 + n_states * 101
        labels = _initial_labels(sequences, scaler, feature_set, n_states, seed)
        prior_labels = [values.copy() for values in labels]
        score = NEG_INF
        model: Optional[ModelParams] = None
        for iteration in range(n_iterations):
            model = estimate_params(
                sequences,
                labels,
                scaler,
                feature_set,
                family,
                n_states,
                max_duration,
            )
            labels, score = decode_sequences(sequences, model)
            ari = adjusted_rand_score(
                np.concatenate(prior_labels), np.concatenate(labels)
            )
            occupancy = np.bincount(
                np.concatenate(labels), minlength=n_states
            ).astype(float)
            occupancy /= occupancy.sum()
            diagnostics.append(
                {
                    "initialization": initialization,
                    "iteration": iteration + 1,
                    "viterbi_score_per_step": score / sum(map(len, sequences)),
                    "label_ari_from_previous_iteration": ari,
                    "minimum_state_occupancy": float(occupancy.min()),
                }
            )
            prior_labels = [values.copy() for values in labels]
            if iteration >= 1 and ari > 0.9995:
                break
        assert model is not None
        # Re-estimate so the saved parameters exactly match the final labels.
        model = estimate_params(
            sequences,
            labels,
            scaler,
            feature_set,
            family,
            n_states,
            max_duration,
        )
        labels, score = decode_sequences(sequences, model)
        candidates.append((score, model, labels))
    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best_model, best_labels = candidates[0]
    for initialization, (candidate_score, _, candidate_labels) in enumerate(candidates):
        diagnostics.append(
            {
                "initialization": initialization,
                "iteration": "final",
                "viterbi_score_per_step": candidate_score / sum(map(len, sequences)),
                "label_ari_from_previous_iteration": adjusted_rand_score(
                    np.concatenate(best_labels), np.concatenate(candidate_labels)
                ),
                "minimum_state_occupancy": float(
                    np.bincount(np.concatenate(candidate_labels), minlength=n_states).min()
                    / sum(map(len, sequences))
                ),
            }
        )
    return best_model, best_labels, pd.DataFrame(diagnostics)


def canonicalize_states(
    model: ModelParams,
    sequences: Sequence[SequenceData],
    labels: Sequence[np.ndarray],
) -> Tuple[ModelParams, List[np.ndarray], np.ndarray]:
    joined_labels = np.concatenate(labels)
    joined_control = np.concatenate([sequence.control for sequence in sequences])
    signatures = []
    for state in range(model.n_states):
        mask = joined_labels == state
        controls = joined_control[mask]
        noop = float(np.mean(np.isin(controls, [0, 2]))) if len(controls) else 0.0
        active_persistence = float(np.mean(controls == 1)) if len(controls) else 0.0
        transitions = float(np.mean(np.isin(controls, [2, 3, 4]))) if len(controls) else 0.0
        signatures.append((-noop, -active_persistence, -transitions, state))
    order = np.array([entry[-1] for entry in sorted(signatures)], dtype=int)
    inverse = np.zeros(model.n_states, dtype=int)
    inverse[order] = np.arange(model.n_states)
    reordered = ModelParams(
        family=model.family,
        feature_set=model.feature_set,
        n_states=model.n_states,
        max_duration=model.max_duration,
        init_prob=model.init_prob[order],
        transition_prob=model.transition_prob[order][:, order],
        means=model.means[order],
        variances=model.variances[order],
        control_prob=model.control_prob[order],
        control_logit_offset=model.control_logit_offset[order],
        reward_prob=model.reward_prob[order],
        duration_q=model.duration_q[order],
        duration_beta=model.duration_beta[order],
        duration_log_prob=model.duration_log_prob[order],
        scaler=model.scaler,
    )
    relabeled = [inverse[values] for values in labels]
    return reordered, relabeled, order


def run_cross_validation(
    sequences: Sequence[SequenceData],
    k_values: Sequence[int],
    max_duration: int,
    n_iterations: int,
    n_init: int,
    random_state: int,
    start_time: float,
) -> pd.DataFrame:
    sessions = sorted({sequence.session for sequence in sequences})
    rows: List[Dict[str, object]] = []
    for heldout_session in sessions:
        train_raw = [
            sequence for sequence in sequences if sequence.session != heldout_session
        ]
        test_raw = [
            sequence for sequence in sequences if sequence.session == heldout_session
        ]
        phase_baseline = fit_phase_baseline(train_raw)
        train = apply_phase_baseline(train_raw, phase_baseline)
        test = apply_phase_baseline(test_raw, phase_baseline)
        baseline_ll = phase_baseline_log_likelihood(test)
        for n_states in k_values:
            for family in ("hmm", "hsmm"):
                log(
                    f"CV session {heldout_session}: fitting {family.upper()} K={n_states}",
                    start_time,
                )
                model, _, diagnostics = fit_model(
                    train,
                    feature_set="phase_controlled",
                    family=family,
                    n_states=n_states,
                    max_duration=max_duration,
                    n_iterations=n_iterations,
                    n_init=n_init,
                    random_state=random_state + heldout_session * 10_000,
                )
                heldout_ll = score_sequences(test, model)
                train_steps = sum(map(len, train))
                test_steps = sum(map(len, test))
                rows.append(
                    {
                        "analysis_version": ANALYSIS_VERSION,
                        "subject": train[0].subject,
                        "game": train[0].game,
                        "game_name": GAME_NAMES[train[0].game],
                        "feature_set": "phase_controlled",
                        "heldout_session": heldout_session,
                        "model_family": family,
                        "n_states": n_states,
                        "n_train_sequences": len(train),
                        "n_test_sequences": len(test),
                        "n_train_steps": train_steps,
                        "n_test_steps": test_steps,
                        "heldout_log_likelihood": heldout_ll,
                        "heldout_log_likelihood_per_step": heldout_ll / test_steps,
                        "phase_baseline_log_likelihood": baseline_ll,
                        "phase_baseline_log_likelihood_per_step": baseline_ll
                        / test_steps,
                        "minimum_training_state_occupancy": float(
                            diagnostics.loc[
                                diagnostics["iteration"].astype(str) == "final",
                                "minimum_state_occupancy",
                            ].iloc[0]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def build_state_count_selection(cv_results: pd.DataFrame) -> pd.DataFrame:
    """Apply a paired one-standard-error rule to select a parsimonious K."""
    hsmm = cv_results[cv_results["model_family"] == "hsmm"]
    pivot = hsmm.pivot(
        index="heldout_session",
        columns="n_states",
        values="heldout_log_likelihood_per_step",
    )
    best_k = int(pivot.mean(axis=0).idxmax())
    rows = []
    for n_states in sorted(pivot.columns.astype(int)):
        paired_deficit = pivot[best_k] - pivot[n_states]
        mean_deficit = float(paired_deficit.mean())
        sem_deficit = float(paired_deficit.sem()) if len(paired_deficit) > 1 else 0.0
        occupancy = hsmm.loc[
            hsmm["n_states"] == n_states, "minimum_training_state_occupancy"
        ]
        stable_occupancy = bool(len(occupancy) and float(occupancy.min()) >= 0.01)
        within_one_se = bool(mean_deficit <= max(sem_deficit, 1e-8))
        rows.append(
            {
                "analysis_version": ANALYSIS_VERSION,
                "subject": int(hsmm["subject"].iloc[0]),
                "game": int(hsmm["game"].iloc[0]),
                "game_name": str(hsmm["game_name"].iloc[0]),
                "n_states": n_states,
                "mean_hsmm_heldout_log_likelihood_per_step": float(
                    pivot[n_states].mean()
                ),
                "mean_paired_deficit_from_best_k": mean_deficit,
                "sem_paired_deficit_from_best_k": sem_deficit,
                "minimum_fold_state_occupancy": float(occupancy.min()),
                "within_one_se_of_best": within_one_se,
                "stable_occupancy": stable_occupancy,
                "unconstrained_best_k": best_k,
            }
        )
    selection = pd.DataFrame(rows)
    eligible = selection[
        selection["within_one_se_of_best"] & selection["stable_occupancy"]
    ]
    if len(eligible):
        selected_k = int(eligible["n_states"].min())
        selection_rule = "paired_one_se_with_stable_occupancy"
    else:
        # Never fall back to a formally best model containing an effectively
        # empty state.  When the one-SE and occupancy criteria do not overlap,
        # retain the best held-out likelihood among occupancy-stable models.
        stable = selection[selection["stable_occupancy"]]
        if len(stable):
            selected_k = int(
                stable.sort_values(
                    ["mean_hsmm_heldout_log_likelihood_per_step", "n_states"],
                    ascending=[False, True],
                )["n_states"].iloc[0]
            )
            selection_rule = "best_heldout_with_stable_occupancy"
        else:
            selected_k = best_k
            selection_rule = "unconstrained_best_no_stable_candidate"
    selection["selected_k"] = selected_k
    selection["selection_rule"] = selection_rule
    return selection


def select_state_count(cv_results: pd.DataFrame) -> int:
    return int(build_state_count_selection(cv_results)["selected_k"].iloc[0])


def infer_final_sequences(
    sequences: Sequence[SequenceData], model: ModelParams, start_time: float
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[float]]:
    log_init, log_transition, log_duration = _log_parameters(model)
    labels: List[np.ndarray] = []
    posteriors: List[np.ndarray] = []
    boundaries: List[np.ndarray] = []
    likelihoods: List[float] = []
    for index, sequence in enumerate(sequences):
        log(
            f"Posterior {model.feature_set}: sequence {index + 1}/{len(sequences)} "
            f"({sequence.sequence_id}, n={len(sequence):,})",
            start_time,
        )
        emission = emission_log_likelihood(sequence, model)
        posterior, boundary, likelihood = _hsmm_posterior(
            emission, log_init, log_transition, log_duration
        )
        posteriors.append(posterior)
        boundaries.append(boundary)
        labels.append(np.argmax(posterior, axis=1).astype(np.int16))
        likelihoods.append(float(likelihood))
    return labels, posteriors, boundaries, likelihoods


def build_decoded_steps(
    sequences: Sequence[SequenceData],
    labels: Sequence[np.ndarray],
    posteriors: Sequence[np.ndarray],
    boundaries: Sequence[np.ndarray],
    model: ModelParams,
) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for sequence, state, posterior, boundary in zip(
        sequences, labels, posteriors, boundaries
    ):
        n = len(sequence)
        adjusted_probability = _conditional_control_probability(sequence, model)
        posterior_adjusted_probability = np.einsum(
            "tk,tkc->tc", posterior, adjusted_probability
        )
        observed_index = np.arange(n)
        baseline_observed = sequence.phase_baseline_prob[
            observed_index, sequence.control
        ]
        adjusted_observed = posterior_adjusted_probability[
            observed_index, sequence.control
        ]
        frame = pd.DataFrame(
            {
                "analysis_version": ANALYSIS_VERSION,
                "subject": sequence.subject,
                "game": sequence.game,
                "game_name": GAME_NAMES[sequence.game],
                "feature_set": model.feature_set,
                "selected_k": model.n_states,
                "session": sequence.session,
                "block": sequence.block,
                "episode": sequence.episode,
                "phase_index": sequence.phase_index,
                "sequence_id": sequence.sequence_id,
                "source_file": sequence.source_file,
                "step_in_sequence": np.arange(n, dtype=int),
                "source_step": sequence.source_start + np.arange(n, dtype=int),
                "episode_progress": sequence.episode_progress,
                "cycle_id": sequence.cycle_id,
                "cycle_progress": sequence.cycle_progress,
                "cycle_boundary": sequence.cycle_boundary.astype(int),
                "enemy_count_raw": sequence.enemy_count_raw,
                "action": sequence.action,
                "action_label": [ACTION_LABELS[int(value)] for value in sequence.action],
                "previous_action": sequence.previous_action,
                "control_category": sequence.control,
                "control_label": [CONTROL_LABELS[int(value)] for value in sequence.control],
                "reward": sequence.reward,
                "state_id": state.astype(int) + 1,
                "boundary_probability": boundary,
                "phase_baseline_observed_probability": baseline_observed,
                "mode_adjusted_observed_probability": adjusted_observed,
                "mode_log_likelihood_gain": np.log(
                    np.clip(adjusted_observed, EPS, 1.0)
                )
                - np.log(np.clip(baseline_observed, EPS, 1.0)),
            }
        )
        for index, name in enumerate(sequence.task_feature_names):
            frame[f"task_{name}"] = sequence.continuous_raw[:, index]
        for k in range(model.n_states):
            frame[f"state_posterior_{k + 1:02d}"] = posterior[:, k]
        for category, label in enumerate(CONTROL_LABELS):
            safe_label = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
            frame[f"phase_baseline_{safe_label}"] = sequence.phase_baseline_prob[
                :, category
            ]
            frame[f"mode_adjusted_{safe_label}"] = posterior_adjusted_probability[
                :, category
            ]
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def build_state_profiles(
    sequences: Sequence[SequenceData],
    decoded: pd.DataFrame,
    model: ModelParams,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for state in range(1, model.n_states + 1):
        state_rows = decoded[decoded["state_id"] == state]
        occupancy = len(state_rows) / max(len(decoded), 1)
        base = {
            "analysis_version": ANALYSIS_VERSION,
            "subject": int(decoded["subject"].iloc[0]),
            "game": int(decoded["game"].iloc[0]),
            "game_name": str(decoded["game_name"].iloc[0]),
            "feature_set": model.feature_set,
            "selected_k": model.n_states,
            "state_id": state,
            "state_occupancy": occupancy,
        }
        for control_index, control_label in enumerate(CONTROL_LABELS):
            rows.append(
                {
                    **base,
                    "profile_domain": "control",
                    "profile_variable": control_label,
                    "profile_value": float(
                        np.mean(state_rows["control_category"] == control_index)
                    ),
                    "profile_unit": "probability",
                }
            )
            safe_label = re.sub(
                r"[^a-z0-9]+", "_", control_label.lower()
            ).strip("_")
            observed_probability = float(
                np.mean(state_rows["control_category"] == control_index)
            )
            baseline_probability = float(
                state_rows[f"phase_baseline_{safe_label}"].mean()
            )
            rows.append(
                {
                    **base,
                    "profile_domain": "phase_adjusted_control",
                    "profile_variable": control_label,
                    "profile_value": observed_probability - baseline_probability,
                    "profile_unit": "probability_difference",
                }
            )
            rows.append(
                {
                    **base,
                    "profile_domain": "control_logit_offset",
                    "profile_variable": control_label,
                    "profile_value": float(
                        model.control_logit_offset[state - 1, control_index]
                    ),
                    "profile_unit": "logit",
                }
            )
        for action_index, action_label in enumerate(ACTION_LABELS):
            rows.append(
                {
                    **base,
                    "profile_domain": "action",
                    "profile_variable": action_label,
                    "profile_value": float(np.mean(state_rows["action"] == action_index)),
                    "profile_unit": "probability",
                }
            )
        for name in model.scaler.feature_names:
            raw_column = f"task_{name}"
            raw_index = model.scaler.feature_names.index(name)
            standardized_mean = (
                float(state_rows[raw_column].mean()) - model.scaler.mean[raw_index]
            ) / model.scaler.scale[raw_index]
            rows.append(
                {
                    **base,
                    "profile_domain": "task_state",
                    "profile_variable": name,
                    "profile_value": standardized_mean,
                    "profile_unit": "training_SD",
                }
            )
        rows.extend(
            [
                {
                    **base,
                    "profile_domain": "duration",
                    "profile_variable": "discrete_weibull_q",
                    "profile_value": float(model.duration_q[state - 1]),
                    "profile_unit": "parameter",
                },
                {
                    **base,
                    "profile_domain": "duration",
                    "profile_variable": "discrete_weibull_beta",
                    "profile_value": float(model.duration_beta[state - 1]),
                    "profile_unit": "parameter",
                },
            ]
        )
    return pd.DataFrame(rows)


def build_duration_curves(decoded: pd.DataFrame, model: ModelParams) -> pd.DataFrame:
    durations = np.arange(1, model.max_duration + 1, dtype=int)
    rows = []
    for state in range(model.n_states):
        q = float(model.duration_q[state])
        beta = float(model.duration_beta[state])
        survival_previous = q ** np.maximum(durations - 1, 0) ** beta
        survival = q ** durations**beta
        hazard = 1.0 - survival / np.maximum(survival_previous, EPS)
        for duration, s, h in zip(durations, survival, hazard):
            rows.append(
                {
                    "analysis_version": ANALYSIS_VERSION,
                    "subject": int(decoded["subject"].iloc[0]),
                    "game": int(decoded["game"].iloc[0]),
                    "game_name": str(decoded["game_name"].iloc[0]),
                    "feature_set": model.feature_set,
                    "selected_k": model.n_states,
                    "state_id": state + 1,
                    "duration": int(duration),
                    "survival": float(s),
                    "exit_hazard": float(h),
                }
            )
    return pd.DataFrame(rows)


def build_transition_table(decoded: pd.DataFrame, model: ModelParams) -> pd.DataFrame:
    rows = []
    for source in range(model.n_states):
        for destination in range(model.n_states):
            rows.append(
                {
                    "analysis_version": ANALYSIS_VERSION,
                    "subject": int(decoded["subject"].iloc[0]),
                    "game": int(decoded["game"].iloc[0]),
                    "game_name": str(decoded["game_name"].iloc[0]),
                    "feature_set": model.feature_set,
                    "selected_k": model.n_states,
                    "source_state": source + 1,
                    "destination_state": destination + 1,
                    "transition_probability": float(
                        model.transition_prob[source, destination]
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_model_qa(decoded: pd.DataFrame, model: ModelParams) -> pd.DataFrame:
    """Return subject-game QA rows that remain meaningful after concatenation."""
    posterior_columns = sorted(
        column for column in decoded.columns if column.startswith("state_posterior_")
    )
    posterior_error = np.abs(decoded[posterior_columns].sum(axis=1).to_numpy() - 1.0)
    segment_lengths: List[int] = []
    segment_states: List[int] = []
    for _, sequence in decoded.groupby("sequence_id", sort=False):
        labels = sequence.sort_values("step_in_sequence")["state_id"].to_numpy(int)
        states, starts, stops = _run_segments(labels.astype(np.int16))
        segment_states.extend(states.astype(int).tolist())
        segment_lengths.extend((stops - starts).astype(int).tolist())
    lengths = np.asarray(segment_lengths, dtype=int)
    states = np.asarray(segment_states, dtype=int)
    rows: List[Dict[str, object]] = []
    common = {
        "analysis_version": ANALYSIS_VERSION,
        "subject": int(decoded["subject"].iloc[0]),
        "game": int(decoded["game"].iloc[0]),
        "game_name": str(decoded["game_name"].iloc[0]),
        "feature_set": model.feature_set,
        "selected_k": model.n_states,
    }
    metrics = {
        "n_steps": float(len(decoded)),
        "n_segments": float(len(lengths)),
        "mean_boundary_probability": float(decoded["boundary_probability"].mean()),
        "maximum_posterior_sum_error": float(np.max(posterior_error)),
        "fraction_segments_at_duration_cap": float(
            np.mean(lengths >= model.max_duration) if len(lengths) else np.nan
        ),
    }
    for name, value in metrics.items():
        rows.append({**common, "state_id": 0, "qa_metric": name, "qa_value": value})
    for state in range(1, model.n_states + 1):
        state_mask = decoded["state_id"].to_numpy(int) == state
        state_lengths = lengths[states == state]
        state_metrics = {
            "state_occupancy": float(np.mean(state_mask)),
            "state_mean_decoded_duration": float(
                np.mean(state_lengths) if len(state_lengths) else np.nan
            ),
            "state_fraction_segments_at_duration_cap": float(
                np.mean(state_lengths >= model.max_duration)
                if len(state_lengths)
                else np.nan
            ),
        }
        for name, value in state_metrics.items():
            rows.append(
                {**common, "state_id": state, "qa_metric": name, "qa_value": value}
            )
    return pd.DataFrame(rows)


def build_event_alignment(
    decoded_by_feature: Mapping[str, pd.DataFrame], event_window: int
) -> pd.DataFrame:
    """Aggregate event-centered values at sequence x event x lag grain.

    Saving one row for every event-lag pair creates millions of redundant rows
    in action-rich games.  Sequence-level means retain the independent run
    structure needed for plotting and later participant-level aggregation.
    ``n_events`` preserves the denominator for optional weighted summaries.
    """
    rows = []
    for feature_set, decoded in decoded_by_feature.items():
        posterior_columns = sorted(
            column for column in decoded.columns if column.startswith("state_posterior_")
        )
        for sequence_id, sequence in decoded.groupby("sequence_id", sort=False):
            sequence = sequence.sort_values("step_in_sequence").reset_index(drop=True)
            previous = sequence["previous_action"].to_numpy(int)
            current = sequence["action"].to_numpy(int)
            reward = sequence["reward"].to_numpy(float)
            event_masks = {
                "NOOP to active": (previous == 0) & (current > 0),
                "Active to NOOP": (previous > 0) & (current == 0),
                "Active switch": (previous > 0) & (current > 0) & (previous != current),
                "Reward event": reward != 0,
            }
            for event_name, event_mask in event_masks.items():
                event_indices = np.flatnonzero(event_mask)
                if len(event_indices) == 0:
                    continue
                for lag in range(-event_window, event_window + 1):
                    indices = event_indices + lag
                    indices = indices[(indices >= 0) & (indices < len(sequence))]
                    if len(indices) == 0:
                        continue
                    base = {
                        "analysis_version": ANALYSIS_VERSION,
                        "subject": int(sequence["subject"].iloc[0]),
                        "game": int(sequence["game"].iloc[0]),
                        "game_name": str(sequence["game_name"].iloc[0]),
                        "feature_set": feature_set,
                        "event_type": event_name,
                        "sequence_id": sequence_id,
                        "lag": lag,
                        "n_events": int(len(indices)),
                        "boundary_probability": float(
                            sequence["boundary_probability"].to_numpy(float)[indices].mean()
                        ),
                    }
                    for column in posterior_columns:
                        base[column] = float(
                            sequence[column].to_numpy(float)[indices].mean()
                        )
                    rows.append(base)
    return pd.DataFrame(rows)


def save_feature_cache(
    sequences: Sequence[SequenceData], output_path: Path
) -> None:
    frames = []
    for sequence in sequences:
        frame = pd.DataFrame(
            {
                "analysis_version": ANALYSIS_VERSION,
                "subject": sequence.subject,
                "game": sequence.game,
                "session": sequence.session,
                "block": sequence.block,
                "episode": sequence.episode,
                "phase_index": sequence.phase_index,
                "sequence_id": sequence.sequence_id,
                "source_file": sequence.source_file,
                "step_in_sequence": np.arange(len(sequence)),
                "source_step": sequence.source_start + np.arange(len(sequence)),
                "episode_progress": sequence.episode_progress,
                "cycle_id": sequence.cycle_id,
                "cycle_progress": sequence.cycle_progress,
                "cycle_boundary": sequence.cycle_boundary.astype(int),
                "enemy_count_raw": sequence.enemy_count_raw,
                "action": sequence.action,
                "previous_action": sequence.previous_action,
                "control_category": sequence.control,
                "reward_category": sequence.reward_category,
                "reward": sequence.reward,
            }
        )
        for index, name in enumerate(sequence.task_feature_names):
            frame[f"task_{name}"] = sequence.continuous_raw[:, index]
        for category, label in enumerate(CONTROL_LABELS):
            safe_label = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
            frame[f"phase_baseline_{safe_label}"] = sequence.phase_baseline_prob[
                :, category
            ]
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_csv(output_path, index=False, compression="gzip")


def _state_colors(n_states: int) -> List[Tuple[float, float, float, float]]:
    cmap = plt.get_cmap("viridis")
    if n_states == 1:
        return [cmap(0.5)]
    return [cmap(index / (n_states - 1)) for index in range(n_states)]


def plot_model_comparison(cv: pd.DataFrame, selected_k: int, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    colors = {"hmm": "#7A7A7A", "hsmm": "#3B7EA1"}
    for family in ("hmm", "hsmm"):
        subset = cv[cv["model_family"] == family]
        summary = subset.groupby("n_states")["heldout_log_likelihood_per_step"].agg(
            ["mean", "sem"]
        )
        x = summary.index.to_numpy(int)
        y = summary["mean"].to_numpy(float)
        error = summary["sem"].fillna(0).to_numpy(float)
        axes[0].plot(
            x,
            y,
            marker="o",
            lw=2,
            color=colors[family],
            label=family.upper(),
        )
        axes[0].fill_between(x, y - error, y + error, color=colors[family], alpha=0.15)
    baseline_by_session = cv.groupby("heldout_session")[
        "phase_baseline_log_likelihood_per_step"
    ].first()
    baseline_mean = float(baseline_by_session.mean())
    axes[0].axhline(
        baseline_mean,
        color="#D17A22",
        lw=1.8,
        ls=":",
        label="Phase baseline",
    )
    axes[0].axvline(selected_k, color="#222222", ls="--", lw=1)
    axes[0].set_xlabel("Number of latent states")
    axes[0].set_ylabel("Held-out conditional log likelihood per step")
    axes[0].set_title("Phase-controlled held-out model evidence")
    axes[0].legend(frameon=False)

    pivot = cv[cv["n_states"] == selected_k].pivot(
        index="heldout_session",
        columns="model_family",
        values="heldout_log_likelihood_per_step",
    )
    delta = pivot["hsmm"] - pivot["hmm"]
    axes[1].axhline(0, color="#777777", lw=1, ls="--")
    axes[1].scatter(
        np.arange(len(delta)), delta, s=55, color="#3B7EA1", edgecolor="white", zorder=3
    )
    axes[1].plot(np.arange(len(delta)), delta, color="#9BC4D8", lw=1)
    axes[1].set_xticks(np.arange(len(delta)), [f"Session {value}" for value in delta.index])
    axes[1].set_ylabel("HSMM − HMM held-out log likelihood per step")
    axes[1].set_title(f"Explicit-duration advantage at K={selected_k}")
    for ax in axes:
        ax.grid(True, color="#E5E7EB", lw=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_state_atlas(
    profiles: pd.DataFrame,
    durations: pd.DataFrame,
    transitions: pd.DataFrame,
    output_path: Path,
) -> None:
    feature_set = str(profiles["feature_set"].iloc[0])
    n_states = int(profiles["selected_k"].iloc[0])
    colors = _state_colors(n_states)
    control_matrix = profiles[
        profiles["profile_domain"] == "phase_adjusted_control"
    ].pivot(
        index="state_id", columns="profile_variable", values="profile_value"
    )
    control_matrix = control_matrix[[name for name in CONTROL_LABELS if name in control_matrix]]
    task_matrix = profiles[profiles["profile_domain"] == "task_state"].pivot(
        index="state_id", columns="profile_variable", values="profile_value"
    )
    task_matrix = task_matrix[
        [
            name
            for name in GAME_FEATURES[int(profiles["game"].iloc[0])]
            if name in task_matrix
        ]
    ]
    fig = plt.figure(figsize=(14, 10))
    grid = fig.add_gridspec(
        3,
        2,
        width_ratios=[1.35, 1.0],
        height_ratios=[0.72, 1.0, 1.0],
        hspace=0.62,
        wspace=0.30,
    )
    ax0 = fig.add_subplot(grid[0, 0])
    control_limit = max(0.05, float(np.nanmax(np.abs(control_matrix.to_numpy(float)))))
    image = ax0.imshow(
        control_matrix.to_numpy(float),
        aspect="auto",
        cmap="coolwarm",
        vmin=-control_limit,
        vmax=control_limit,
    )
    ax0.set_yticks(
        np.arange(n_states), [f"State {state}" for state in control_matrix.index]
    )
    ax0.set_xticks(
        np.arange(len(control_matrix.columns)),
        control_matrix.columns,
        rotation=30,
        ha="right",
        fontsize=8,
    )
    ax0.set_title("Phase-adjusted control-category deviations")
    fig.colorbar(
        image, ax=ax0, fraction=0.03, pad=0.02, label="Observed − phase baseline"
    )

    ax_task = fig.add_subplot(grid[0, 1])
    task_image = ax_task.imshow(
        task_matrix.to_numpy(float),
        aspect="auto",
        cmap="coolwarm",
        vmin=-1.5,
        vmax=1.5,
    )
    ax_task.set_yticks(
        np.arange(n_states), [f"State {state}" for state in task_matrix.index]
    )
    ax_task.set_xticks(
        np.arange(len(task_matrix.columns)),
        task_matrix.columns,
        rotation=30,
        ha="right",
        fontsize=8,
    )
    ax_task.set_title("Observed task-state means")
    fig.colorbar(
        task_image, ax=ax_task, fraction=0.045, pad=0.03, label="Training-set SD"
    )

    ax1 = fig.add_subplot(grid[1, 0])
    for state, group in durations.groupby("state_id"):
        show = group[group["survival"] >= 0.01]
        ax1.plot(show["duration"], show["survival"], color=colors[int(state) - 1], lw=2, label=f"State {state}")
    ax1.set_xlabel("State duration (real steps)")
    ax1.set_ylabel("Model survival probability")
    ax1.set_title("State-specific duration survival")
    ax1.legend(frameon=False)

    ax2 = fig.add_subplot(grid[1, 1])
    for state, group in durations.groupby("state_id"):
        show = group[group["survival"] >= 0.01]
        ax2.plot(show["duration"], show["exit_hazard"], color=colors[int(state) - 1], lw=2, label=f"State {state}")
    ax2.set_xlabel("Elapsed state duration (real steps)")
    ax2.set_ylabel("Conditional exit probability")
    ax2.set_title("State-specific exit hazard")

    ax3 = fig.add_subplot(grid[2, 0])
    transition_matrix = transitions.pivot(
        index="source_state", columns="destination_state", values="transition_probability"
    ).to_numpy(float)
    transition_image = ax3.imshow(transition_matrix, cmap="magma", vmin=0, vmax=max(0.5, np.nanmax(transition_matrix)))
    ax3.set_xticks(np.arange(n_states), [f"S{value}" for value in range(1, n_states + 1)])
    ax3.set_yticks(np.arange(n_states), [f"S{value}" for value in range(1, n_states + 1)])
    ax3.set_xlabel("Destination state")
    ax3.set_ylabel("Source state")
    ax3.set_title("State transition matrix")
    fig.colorbar(transition_image, ax=ax3, fraction=0.04, pad=0.03)

    ax4 = fig.add_subplot(grid[2, 1])
    occupancy = (
        profiles[["state_id", "state_occupancy"]]
        .drop_duplicates("state_id")
        .sort_values("state_id")
    )
    ax4.barh(
        np.arange(n_states),
        occupancy["state_occupancy"],
        color=colors,
        edgecolor="white",
    )
    ax4.set_yticks(
        np.arange(n_states), [f"State {state}" for state in occupancy["state_id"]]
    )
    ax4.invert_yaxis()
    ax4.set_xlim(0, max(0.4, float(occupancy["state_occupancy"].max()) * 1.15))
    ax4.set_xlabel("Fraction of real steps")
    ax4.set_title("Decoded state occupancy")
    ax4.grid(True, axis="x", color="#E5E7EB", lw=0.7)
    ax4.set_axisbelow(True)
    ax4.spines[["top", "right"]].set_visible(False)

    game_name = str(profiles["game_name"].iloc[0])
    fig.suptitle(
        f"Phase-controlled control-mode atlas · {game_name}",
        fontsize=16,
        y=0.995,
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _smallest_source_subset(decoded: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
    """Select the shortest complete source NPZ, retaining all its episodes."""
    source_lengths = (
        decoded.groupby("source_file", sort=False)["source_step"].max().astype(int) + 1
    )
    source_file = str(source_lengths.sort_values(kind="stable").index[0])
    subset = (
        decoded[decoded["source_file"] == source_file]
        .sort_values("source_step")
        .reset_index(drop=True)
    )
    return source_file, subset


def _mark_progress_boundaries(axes: Sequence[plt.Axes], sequence: pd.DataFrame) -> None:
    source_step = sequence["source_step"].to_numpy(int)
    episodes = sequence["episode"].to_numpy(int)
    episode_starts = source_step[np.r_[True, episodes[1:] != episodes[:-1]]]
    cycle_steps = source_step[sequence["cycle_boundary"].to_numpy(bool)]
    for ax in axes:
        for value in episode_starts[1:]:
            ax.axvline(value, color="#202020", lw=1.0, ls="--", alpha=0.8)
        for value in cycle_steps:
            ax.axvline(value, color="#D17A22", lw=0.65, alpha=0.45)


def plot_full_recording_progress(decoded: pd.DataFrame, output_path: Path) -> None:
    """Plot observed progression through the shortest complete source NPZ."""
    source_file, sequence = _smallest_source_subset(decoded)
    game = int(sequence["game"].iloc[0])
    x = sequence["source_step"].to_numpy(int)
    action = sequence["action"].to_numpy(int)
    action_colors = [
        "#4C78A8", "#E45756", "#54A24B", "#F2A541", "#9D77A8", "#72B7B2"
    ]
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(15, 8.8),
        sharex=True,
        gridspec_kw={"height_ratios": [0.35, 1.2, 1.0, 0.65]},
    )
    axes[0].imshow(
        action[np.newaxis, :],
        aspect="auto",
        interpolation="nearest",
        cmap=ListedColormap(action_colors),
        vmin=-0.5,
        vmax=5.5,
        extent=[x[0], x[-1], 0, 1],
    )
    axes[0].set_yticks([0.5], ["Human\naction"])

    if game == 1:
        task_columns = ["task_ball_x", "task_ball_y", "task_player_y", "task_cpu_y"]
        for column in task_columns:
            values = sequence[column].to_numpy(float)
            scale = np.nanstd(values) or 1.0
            axes[1].plot(
                x,
                (values - np.nanmean(values)) / scale,
                lw=0.75,
                label=column.replace("task_", "").replace("_", " "),
            )
        axes[1].set_ylabel("Game variables\n(within-file SD)")
        axes[1].legend(frameon=False, ncol=4, fontsize=8, loc="upper right")
        cycle_label = "Within-rally elapsed fraction"
    else:
        axes[1].plot(x, sequence["enemy_count_raw"], color="#3B7EA1", lw=1.0)
        axes[1].set_ylim(-1, max(37, float(sequence["enemy_count_raw"].max()) + 1))
        axes[1].set_ylabel("Remaining\nenemies")
        auxiliary = ["task_player_x", "task_enemies_y", "task_lives"]
        for column in auxiliary:
            values = sequence[column].to_numpy(float)
            scale = np.nanstd(values) or 1.0
            axes[2].plot(
                x,
                (values - np.nanmean(values)) / scale,
                lw=0.8,
                label=column.replace("task_", "").replace("_", " "),
            )
        axes[2].legend(frameon=False, ncol=3, fontsize=8, loc="upper right")
        axes[2].set_ylabel("Other game variables\n(within-file SD)")
        cycle_label = "Enemy depletion within wave"

    if game == 1:
        axes[2].plot(
            x,
            sequence["cycle_progress"],
            color="#3B7EA1",
            lw=1.1,
            label=cycle_label,
        )
        axes[2].plot(
            x,
            sequence["episode_progress"],
            color="#777777",
            lw=1.0,
            alpha=0.8,
            label="Episode progress",
        )
        axes[2].set_ylabel("Progress fraction")
        axes[2].legend(frameon=False, ncol=2, fontsize=8, loc="upper right")

    reward_steps = x[sequence["reward"].to_numpy(float) != 0]
    if game == 1:
        cumulative_reward = np.cumsum(sequence["reward"].to_numpy(float))
        axes[3].step(
            x,
            cumulative_reward,
            where="post",
            color="#3B7EA1",
            lw=1.1,
            label="Cumulative reward",
        )
        reward_mask = sequence["reward"].to_numpy(float) != 0
        axes[3].scatter(
            reward_steps,
            cumulative_reward[reward_mask],
            s=18,
            color="#D17A22",
            edgecolor="white",
            linewidth=0.4,
            label="Reward event",
            zorder=3,
        )
        axes[3].set_ylabel("Cumulative\nreward")
    else:
        axes[3].plot(
            x,
            sequence["cycle_progress"],
            color="#D17A22",
            lw=1.1,
            label=cycle_label,
        )
        axes[3].plot(
            x,
            sequence["episode_progress"],
            color="#777777",
            lw=0.9,
            alpha=0.75,
            label="Episode progress",
        )
        axes[3].scatter(
            reward_steps,
            np.full(len(reward_steps), 1.03),
            s=8,
            marker="|",
            color="#202020",
            label="Reward event",
            clip_on=False,
        )
        axes[3].set_ylim(-0.03, 1.08)
        axes[3].set_ylabel("Progress")
    axes[3].set_xlabel("Real step in complete source recording")
    axes[3].legend(frameon=False, ncol=3, fontsize=8, loc="upper right")

    _mark_progress_boundaries(axes, sequence)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, axis="x", color="#E5E7EB", lw=0.5)
        ax.set_axisbelow(True)
    fig.suptitle(
        f"Observed game progression in the shortest recording · {GAME_NAMES[game]}",
        fontsize=15,
        y=0.995,
    )
    fig.text(
        0.5,
        0.962,
        f"{Path(source_file).name}; orange lines mark {'rally resets' if game == 1 else 'enemy refills'}; dashed lines mark episode boundaries",
        ha="center",
        va="top",
        fontsize=9,
        color="#666666",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_full_recording_latent_states(
    decoded: pd.DataFrame, output_path: Path
) -> None:
    """Show latent-state inference over every step of the shortest source NPZ."""
    source_file, sequence = _smallest_source_subset(decoded)
    posterior_columns = sorted(
        column for column in sequence.columns if column.startswith("state_posterior_")
    )
    x = sequence["source_step"].to_numpy(int)
    n_states = len(posterior_columns)
    colors = _state_colors(n_states)
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(15, 9.0),
        sharex=True,
        gridspec_kw={"height_ratios": [0.8, 0.35, 1.65, 0.75]},
    )
    game = int(sequence["game"].iloc[0])
    if game == 1:
        axes[0].plot(
            x,
            sequence["cycle_progress"],
            color="#3B7EA1",
            lw=1.0,
            label="Within-rally elapsed fraction",
        )
        axes[0].plot(
            x,
            sequence["episode_progress"],
            color="#777777",
            lw=0.9,
            label="Episode progress",
        )
        axes[0].set_ylabel("Progress")
    else:
        axes[0].plot(
            x,
            sequence["enemy_count_raw"],
            color="#3B7EA1",
            lw=1.0,
            label="Remaining enemies",
        )
        axes[0].plot(
            x,
            36.0 * sequence["cycle_progress"],
            color="#D17A22",
            lw=0.9,
            label="Enemy depletion × 36",
        )
        axes[0].set_ylabel("Enemy-wave\nprogress")
    axes[0].legend(frameon=False, ncol=2, fontsize=8, loc="upper right")

    map_state = sequence["state_id"].to_numpy(int) - 1
    axes[1].imshow(
        map_state[np.newaxis, :],
        aspect="auto",
        interpolation="nearest",
        cmap=ListedColormap(colors),
        vmin=-0.5,
        vmax=n_states - 0.5,
        extent=[x[0], x[-1], 0, 1],
    )
    axes[1].set_yticks([0.5], ["MAP\nstate"])

    posterior = sequence[posterior_columns].to_numpy(float).T
    axes[2].imshow(
        posterior,
        aspect="auto",
        cmap="viridis",
        vmin=0,
        vmax=1,
        interpolation="nearest",
        extent=[x[0], x[-1], n_states + 0.5, 0.5],
    )
    axes[2].set_yticks(
        np.arange(1, n_states + 1), [f"State {value}" for value in range(1, n_states + 1)]
    )
    axes[2].set_ylabel("HSMM posterior")

    axes[3].plot(
        x, sequence["boundary_probability"], color="#C44E52", lw=1.0
    )
    axes[3].fill_between(
        x,
        0,
        sequence["boundary_probability"],
        color="#C44E52",
        alpha=0.2,
    )
    axes[3].set_ylim(0, 1.02)
    axes[3].set_ylabel("Boundary\nprobability")
    axes[3].set_xlabel("Real step in complete source recording")

    _mark_progress_boundaries(axes, sequence)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    feature_set = str(sequence["feature_set"].iloc[0]).replace("_", " ")
    progress_note = (
        "rally boundaries reset the HSMM; normalized rally completion is visualization-only"
        if game == 1
        else "enemy refills reset the HSMM; depletion enters the nuisance phase baseline, not the latent-mode emission"
    )
    fig.suptitle(
        f"Phase-controlled modes across the shortest recording · {GAME_NAMES[game]}",
        fontsize=15,
        y=0.995,
    )
    fig.text(
        0.5,
        0.962,
        f"{Path(source_file).name}; {progress_note}",
        ha="center",
        va="top",
        fontsize=9,
        color="#666666",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_phase_control_diagnostics(
    decoded: pd.DataFrame,
    profiles: pd.DataFrame,
    cv_results: pd.DataFrame,
    selected_k: int,
    output_path: Path,
) -> None:
    """Diagnose whether residual control modes collapse back onto game phase."""
    posterior_columns = sorted(
        column for column in decoded.columns if column.startswith("state_posterior_")
    )
    bins = np.linspace(0.0, 1.0, 11)
    phase_bin = pd.cut(
        decoded["cycle_progress"], bins=bins, labels=False, include_lowest=True
    )
    occupancy_rows = []
    for bin_index in range(10):
        mask = phase_bin == bin_index
        if mask.any():
            occupancy_rows.append(
                decoded.loc[mask, posterior_columns].mean(axis=0).to_numpy(float)
            )
        else:
            occupancy_rows.append(np.full(len(posterior_columns), np.nan))
    occupancy = np.asarray(occupancy_rows, dtype=float).T

    adjusted = profiles[
        profiles["profile_domain"] == "phase_adjusted_control"
    ].pivot(index="state_id", columns="profile_variable", values="profile_value")
    adjusted = adjusted[[name for name in CONTROL_LABELS if name in adjusted]]

    cv_selected = cv_results[cv_results["n_states"] == selected_k].copy()
    hsmm = cv_selected[cv_selected["model_family"] == "hsmm"].set_index(
        "heldout_session"
    )
    heldout_gain = (
        hsmm["heldout_log_likelihood_per_step"]
        - hsmm["phase_baseline_log_likelihood_per_step"]
    )

    gain_table = decoded.assign(phase_bin=phase_bin).dropna(subset=["phase_bin"])
    sequence_gain = (
        gain_table.groupby(["sequence_id", "phase_bin"])["mode_log_likelihood_gain"]
        .mean()
        .reset_index()
    )
    gain_summary = sequence_gain.groupby("phase_bin")["mode_log_likelihood_gain"].agg(
        ["mean", "sem"]
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    image0 = axes[0, 0].imshow(
        occupancy,
        aspect="auto",
        cmap="Blues",
        vmin=0,
        vmax=max(0.25, float(np.nanmax(occupancy))),
    )
    axes[0, 0].set_yticks(
        np.arange(len(posterior_columns)),
        [f"Mode {value}" for value in range(1, len(posterior_columns) + 1)],
    )
    axes[0, 0].set_xticks(
        np.arange(10), [f"{value / 10:.1f}" for value in range(10)], rotation=45
    )
    axes[0, 0].set_xlabel("Within-phase progress bin")
    axes[0, 0].set_title("Posterior mode occupancy across phase")
    fig.colorbar(image0, ax=axes[0, 0], fraction=0.04, pad=0.03, label="Probability")

    limit = max(0.05, float(np.nanmax(np.abs(adjusted.to_numpy(float)))))
    image1 = axes[0, 1].imshow(
        adjusted.to_numpy(float),
        aspect="auto",
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
    )
    axes[0, 1].set_yticks(
        np.arange(len(adjusted)), [f"Mode {value}" for value in adjusted.index]
    )
    axes[0, 1].set_xticks(
        np.arange(len(adjusted.columns)),
        adjusted.columns,
        rotation=35,
        ha="right",
        fontsize=8,
    )
    axes[0, 1].set_title("Control-category deviation after phase adjustment")
    fig.colorbar(
        image1,
        ax=axes[0, 1],
        fraction=0.04,
        pad=0.03,
        label="Observed − phase baseline",
    )

    axes[1, 0].axhline(0, color="#777777", lw=1, ls="--")
    axes[1, 0].scatter(
        heldout_gain.index.astype(int),
        heldout_gain.to_numpy(float),
        color="#3B7EA1",
        s=60,
        edgecolor="white",
        zorder=3,
    )
    axes[1, 0].plot(
        heldout_gain.index.astype(int),
        heldout_gain.to_numpy(float),
        color="#9BC4D8",
        lw=1,
    )
    axes[1, 0].set_xticks(
        heldout_gain.index.astype(int),
        [f"Session {value}" for value in heldout_gain.index.astype(int)],
    )
    axes[1, 0].set_ylabel("HSMM − phase baseline\nlog likelihood per step")
    axes[1, 0].set_title(f"Held-out incremental evidence at K={selected_k}")

    x = gain_summary.index.to_numpy(float)
    mean = gain_summary["mean"].to_numpy(float)
    sem = gain_summary["sem"].fillna(0).to_numpy(float)
    axes[1, 1].axhline(0, color="#777777", lw=1, ls="--")
    axes[1, 1].plot(x, mean, color="#3B7EA1", marker="o", lw=2)
    axes[1, 1].fill_between(x, mean - sem, mean + sem, color="#3B7EA1", alpha=0.18)
    axes[1, 1].set_xticks(np.arange(10), [f"{value / 10:.1f}" for value in range(10)])
    axes[1, 1].set_xlabel("Within-phase progress bin")
    axes[1, 1].set_ylabel("Descriptive log-probability gain")
    axes[1, 1].set_title("Mode contribution across phase")

    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
    game_name = str(decoded["game_name"].iloc[0])
    phase_name = "rallies" if int(decoded["game"].iloc[0]) == 1 else "enemy waves"
    fig.suptitle(
        f"Phase-controlled control-mode diagnostics · {game_name}", fontsize=15, y=0.985
    )
    fig.text(
        0.5,
        0.945,
        f"HSMM sequences are independently reset at {phase_name}; normalized completion is used only for descriptive binning",
        ha="center",
        va="top",
        color="#666666",
        fontsize=9,
    )
    fig.subplots_adjust(
        left=0.10, right=0.93, bottom=0.10, top=0.84, wspace=0.40, hspace=0.42
    )
    fig.savefig(output_path, dpi=220, facecolor="white")
    plt.close(fig)


def plot_event_alignment(event_alignment: pd.DataFrame, output_path: Path) -> None:
    event_types = [
        "NOOP to active",
        "Active to NOOP",
        "Active switch",
        "Reward event",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    colors = {"phase_controlled": "#3B7EA1"}
    for ax, event_type in zip(axes.flat, event_types):
        subset = event_alignment[event_alignment["event_type"] == event_type]
        for feature_set, group in subset.groupby("feature_set"):
            summary = group.groupby("lag")["boundary_probability"].agg(["mean", "sem"])
            x = summary.index.to_numpy(int)
            mean = summary["mean"].to_numpy(float)
            sem = summary["sem"].fillna(0).to_numpy(float)
            color = colors.get(feature_set, "#666666")
            ax.plot(x, mean, color=color, lw=2, label=feature_set.replace("_", " "))
            ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.15)
        ax.axvline(0, color="#222222", lw=1, ls="--")
        ax.set_title(event_type)
        ax.grid(True, color="#E5E7EB", lw=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[1, :]:
        ax.set_xlabel("Real steps relative to event")
    axes[0, 0].set_ylabel("Mean boundary probability")
    axes[1, 0].set_ylabel("Mean boundary probability")
    axes[0, 0].legend(frameon=False)
    game_name = str(event_alignment["game_name"].iloc[0])
    fig.suptitle(f"Latent-state boundaries around observed events · {game_name}", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def save_group_schema(output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    schema_path = output_root / "GROUP_EXPORT_SCHEMA.json"
    schema = {
        "analysis_version": ANALYSIS_VERSION,
        "merge_rule": (
            "Concatenate same-named files under sub*_game*/group_exports. "
            "Use subject and game as the independent analysis unit. Numerical "
            "state_id values are local; align states across subjects from "
            "state_profile_long.csv before state-specific inference."
        ),
        "files": {
            "model_selection_long.csv": (
                "One row per subject-game-session-model-K fold with phase-baseline, "
                "HMM, and HSMM conditional likelihoods."
            ),
            "state_count_selection.csv": "Paired one-standard-error state-count selection.",
            "phase_baseline_coefficients.csv": (
                "Phase-nuisance multinomial coefficients by control category."
            ),
            "state_profile_long.csv": "One row per subject-game-feature-set-state-profile variable.",
            "duration_curve_long.csv": "State-specific survival and hazard by duration.",
            "transition_matrix_long.csv": "State transition probabilities.",
            "decoded_steps.csv.gz": (
                "One row per real step with posterior, boundary probability, "
                "phase-baseline and mode-adjusted probabilities, phase identity, "
                "episode progress, and visualization-only normalized completion."
            ),
            "event_alignment_long.csv.gz": "Event-centered boundary and posterior values.",
            "model_qa_long.csv": "Posterior normalization, state occupancy, and duration-cap diagnostics.",
        },
    }
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")


def write_manifest(
    args: argparse.Namespace,
    sequences: Sequence[SequenceData],
    selected_k: int,
    output_dir: Path,
    output_files: Sequence[Path],
) -> None:
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "created_at_unix": time.time(),
        "subject": args.subject,
        "game": args.game,
        "game_name": GAME_NAMES[args.game],
        "analysis_unit": "subject_x_game",
        "selected_k": selected_k,
        "parameters": {
            "k_values": args.k_values,
            "max_duration": args.max_duration,
            "cv_iterations": args.cv_iterations,
            "final_iterations": args.final_iterations,
            "cv_n_init": args.cv_n_init,
            "n_init": args.n_init,
            "random_state": args.random_state,
            "phase_definition": "reward-delimited_rally"
            if args.game == 1
            else "enemy-refill-delimited_wave",
            "latent_emission": "phase-adjusted_control_transition_category",
        },
        "n_sequences": len(sequences),
        "n_steps": int(sum(map(len, sequences))),
        "sessions": sorted({sequence.session for sequence in sequences}),
        "source_files": sorted({sequence.source_file for sequence in sequences}),
        "state_id_warning": (
            "State IDs are local to this subject-game model. Across-subject "
            "analysis must align state profiles before comparing state-specific effects."
        ),
        "outputs": [
            {
                "path": str(path.resolve()),
                "sha256": _hash_file(path) if path.exists() and path.is_file() else None,
            }
            for path in output_files
        ],
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def load_existing_outputs(output_dir: Path) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    cv = pd.read_csv(output_dir / "tables" / "model_selection_long.csv")
    decoded = {}
    for feature_set in ("phase_controlled",):
        path = output_dir / "tables" / f"decoded_steps_{feature_set}.csv.gz"
        if path.exists():
            decoded[feature_set] = pd.read_csv(path)
    return cv, decoded


def main() -> None:
    args = build_argparser().parse_args()
    if any(value < 2 for value in args.k_values):
        raise ValueError("All --k-values must be >= 2")
    if args.max_duration < 5:
        raise ValueError("--max-duration must be >= 5")
    if args.cv_iterations < 1 or args.final_iterations < 1:
        raise ValueError("Iteration counts must be positive")
    start_time = time.time()
    output_dir = args.output_root / f"sub{args.subject:03d}_game{args.game}"
    cache_dir = output_dir / "cache"
    model_dir = output_dir / "models"
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    group_dir = output_dir / "group_exports"
    for directory in (cache_dir, model_dir, table_dir, figure_dir, group_dir):
        directory.mkdir(parents=True, exist_ok=True)
    save_group_schema(args.output_root)

    sequences, sequence_index = load_sequences(
        args.data_root.resolve(), args.subject, args.game, args.min_sequence_length
    )
    log(
        f"Loaded {len(sequences)} sequences, {sum(map(len, sequences)):,} steps, "
        f"sessions={sorted({sequence.session for sequence in sequences})}",
        start_time,
    )
    sequence_index.to_csv(cache_dir / "sequence_index.csv", index=False)
    feature_cache_path = cache_dir / "step_features.csv.gz"

    if args.figures_only:
        cv_results, decoded_by_feature = load_existing_outputs(output_dir)
        selected_k = select_state_count(cv_results)
        models = {
            feature_set: ModelParams.from_json_dict(
                json.loads((model_dir / f"final_{feature_set}_hsmm.json").read_text())
            )
            for feature_set in decoded_by_feature
        }
    else:
        cv_path = table_dir / "model_selection_long.csv"
        if args.reuse_cv:
            if not cv_path.exists():
                raise FileNotFoundError(f"Cannot --reuse-cv; missing {cv_path}")
            cv_results = pd.read_csv(cv_path)
            expected_k = set(map(int, args.k_values))
            observed_k = set(cv_results["n_states"].astype(int).unique())
            compatible = (
                set(cv_results["analysis_version"].astype(str).unique())
                == {ANALYSIS_VERSION}
                and set(cv_results["subject"].astype(int).unique())
                == {int(args.subject)}
                and set(cv_results["game"].astype(int).unique()) == {int(args.game)}
                and set(cv_results["feature_set"].astype(str).unique())
                == {"phase_controlled"}
                and observed_k == expected_k
            )
            if not compatible:
                raise ValueError(
                    "Existing CV table is incompatible with this analysis/version "
                    f"or candidate K values: {cv_path}"
                )
            log(f"Reusing compatible session-held-out CV table: {cv_path}", start_time)
        else:
            cv_results = run_cross_validation(
                sequences,
                args.k_values,
                args.max_duration,
                args.cv_iterations,
                args.cv_n_init,
                args.random_state,
                start_time,
            )
        selected_k = select_state_count(cv_results)
        log(f"Selected K={selected_k} from session-held-out HSMM likelihood", start_time)
        cv_results.to_csv(table_dir / "model_selection_long.csv", index=False)
        cv_results.to_csv(group_dir / "model_selection_long.csv", index=False)
        state_count_selection = build_state_count_selection(cv_results)
        state_count_selection.to_csv(
            table_dir / "state_count_selection.csv", index=False
        )
        state_count_selection.to_csv(
            group_dir / "state_count_selection.csv", index=False
        )

        phase_baseline = fit_phase_baseline(sequences)
        sequences = apply_phase_baseline(sequences, phase_baseline)
        (model_dir / "phase_control_baseline.json").write_text(
            json.dumps(phase_baseline.to_json_dict(), indent=2), encoding="utf-8"
        )
        baseline_coefficients = build_phase_baseline_coefficients(
            phase_baseline, args.subject, args.game
        )
        baseline_coefficients.to_csv(
            table_dir / "phase_baseline_coefficients.csv", index=False
        )
        baseline_coefficients.to_csv(
            group_dir / "phase_baseline_coefficients.csv", index=False
        )
        save_feature_cache(sequences, feature_cache_path)

        models: Dict[str, ModelParams] = {}
        decoded_by_feature: Dict[str, pd.DataFrame] = {}
        feature_sets = ["phase_controlled"]
        initialization_tables = []
        for feature_index, feature_set in enumerate(feature_sets):
            log(f"Fitting final phase-controlled HSMM with K={selected_k}", start_time)
            model, labels, diagnostics = fit_model(
                sequences,
                feature_set=feature_set,
                family="hsmm",
                n_states=selected_k,
                max_duration=args.max_duration,
                n_iterations=args.final_iterations,
                n_init=args.n_init,
                random_state=args.random_state + feature_index * 50_000,
            )
            model, labels, _ = canonicalize_states(model, sequences, labels)
            model_path = model_dir / f"final_{feature_set}_hsmm.json"
            model_path.write_text(
                json.dumps(model.to_json_dict(), indent=2), encoding="utf-8"
            )
            labels, posteriors, boundaries, likelihoods = infer_final_sequences(
                sequences, model, start_time
            )
            decoded = build_decoded_steps(
                sequences, labels, posteriors, boundaries, model
            )
            decoded.to_csv(
                table_dir / f"decoded_steps_{feature_set}.csv.gz",
                index=False,
                compression="gzip",
            )
            decoded_by_feature[feature_set] = decoded
            models[feature_set] = model
            diagnostics.insert(0, "analysis_version", ANALYSIS_VERSION)
            diagnostics.insert(1, "subject", args.subject)
            diagnostics.insert(2, "game", args.game)
            diagnostics.insert(3, "feature_set", feature_set)
            diagnostics["selected_k"] = selected_k
            initialization_tables.append(diagnostics)
            likelihood_table = pd.DataFrame(
                {
                    "analysis_version": ANALYSIS_VERSION,
                    "subject": args.subject,
                    "game": args.game,
                    "feature_set": feature_set,
                    "sequence_id": [sequence.sequence_id for sequence in sequences],
                    "n_steps": [len(sequence) for sequence in sequences],
                    "log_likelihood": likelihoods,
                    "log_likelihood_per_step": [
                        likelihood / len(sequence)
                        for likelihood, sequence in zip(likelihoods, sequences)
                    ],
                }
            )
            likelihood_table.to_csv(
                table_dir / f"final_sequence_likelihood_{feature_set}.csv", index=False
            )
        pd.concat(initialization_tables, ignore_index=True).to_csv(
            table_dir / "initialization_stability.csv", index=False
        )

    profile_tables = []
    duration_tables = []
    transition_tables = []
    qa_tables = []
    for feature_set, decoded in decoded_by_feature.items():
        model = models[feature_set]
        profiles = build_state_profiles(sequences, decoded, model)
        durations = build_duration_curves(decoded, model)
        transitions = build_transition_table(decoded, model)
        model_qa = build_model_qa(decoded, model)
        profiles.to_csv(table_dir / f"state_profiles_{feature_set}.csv", index=False)
        durations.to_csv(table_dir / f"duration_curves_{feature_set}.csv", index=False)
        transitions.to_csv(table_dir / f"transition_matrix_{feature_set}.csv", index=False)
        model_qa.to_csv(table_dir / f"model_qa_{feature_set}.csv", index=False)
        profile_tables.append(profiles)
        duration_tables.append(durations)
        transition_tables.append(transitions)
        qa_tables.append(model_qa)
        plot_state_atlas(
            profiles,
            durations,
            transitions,
            figure_dir / f"supplementary_state_atlas_{feature_set}.png",
        )

    all_profiles = pd.concat(profile_tables, ignore_index=True)
    all_durations = pd.concat(duration_tables, ignore_index=True)
    all_transitions = pd.concat(transition_tables, ignore_index=True)
    all_qa = pd.concat(qa_tables, ignore_index=True)
    all_decoded = pd.concat(decoded_by_feature.values(), ignore_index=True)
    all_profiles.to_csv(group_dir / "state_profile_long.csv", index=False)
    all_durations.to_csv(group_dir / "duration_curve_long.csv", index=False)
    all_transitions.to_csv(group_dir / "transition_matrix_long.csv", index=False)
    all_qa.to_csv(group_dir / "model_qa_long.csv", index=False)
    all_decoded.to_csv(
        group_dir / "decoded_steps.csv.gz", index=False, compression="gzip"
    )

    event_alignment = build_event_alignment(decoded_by_feature, args.event_window)
    event_alignment.to_csv(
        table_dir / "event_alignment_long.csv.gz", index=False, compression="gzip"
    )
    event_alignment.to_csv(
        group_dir / "event_alignment_long.csv.gz", index=False, compression="gzip"
    )

    plot_model_comparison(
        cv_results,
        selected_k,
        figure_dir / "figure1_hsmm_vs_hmm_model_evidence.png",
    )
    plot_full_recording_progress(
        decoded_by_feature["phase_controlled"],
        figure_dir / "figure2_full_recording_game_progress.png",
    )
    plot_full_recording_latent_states(
        decoded_by_feature["phase_controlled"],
        figure_dir / "figure3_full_recording_phase_controlled_hsmm.png",
    )
    phase_profiles = all_profiles[
        all_profiles["feature_set"] == "phase_controlled"
    ]
    plot_phase_control_diagnostics(
        decoded_by_feature["phase_controlled"],
        phase_profiles,
        cv_results,
        selected_k,
        figure_dir / "figure4_phase_control_diagnostics.png",
    )
    plot_event_alignment(
        event_alignment,
        figure_dir / "supplementary_event_aligned_boundaries.png",
    )

    output_files = [
        table_dir / "model_selection_long.csv",
        model_dir / "phase_control_baseline.json",
        model_dir / "final_phase_controlled_hsmm.json",
        group_dir / "state_count_selection.csv",
        group_dir / "state_profile_long.csv",
        group_dir / "duration_curve_long.csv",
        group_dir / "transition_matrix_long.csv",
        group_dir / "model_qa_long.csv",
        group_dir / "decoded_steps.csv.gz",
        group_dir / "event_alignment_long.csv.gz",
        group_dir / "phase_baseline_coefficients.csv",
        figure_dir / "figure1_hsmm_vs_hmm_model_evidence.png",
        figure_dir / "figure2_full_recording_game_progress.png",
        figure_dir / "figure3_full_recording_phase_controlled_hsmm.png",
        figure_dir / "figure4_phase_control_diagnostics.png",
        figure_dir / "supplementary_event_aligned_boundaries.png",
    ]
    write_manifest(args, sequences, selected_k, output_dir, output_files)
    log(f"Complete -> {output_dir}", start_time)


if __name__ == "__main__":
    main()
