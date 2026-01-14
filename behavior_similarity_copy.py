"""Utilities to compare human behavioral data with Thinker checkpoints."""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Score placeholders --------------------------------------------------------
# ---------------------------------------------------------------------------
#
# Fill in the dictionary below with observed Thinker scores before running the
# script. Use the checkpoint filename *without* the trailing extension as the
# key. Example:
# THINKER_SCORE_PLACEHOLDER = {
#     "spaceinvaders_1e6_0": 1234.5,
#     "spaceinvaders_1e6_1": 1188.0,
# }
# Any missing entry defaults to NaN, which will skip score-based correlations.
THINKER_SCORE_PLACEHOLDER: Dict[str, float] = {
    'spaceinvaders_1e6_0': 400.0,
    'spaceinvaders_1e6_1': 225.0,
    'spaceinvaders_1e6_2': 235.0,
    'spaceinvaders_2e6_0': 515.0,
    'spaceinvaders_2e6_1': 535.0,
    'spaceinvaders_2e6_2': 545.0,
    'spaceinvaders_3e6_0': 775.0,
    'spaceinvaders_3e6_1': 800.0,
    'spaceinvaders_3e6_2': 540.0,
    'spaceinvaders_4e6_0': 585.0,
    'spaceinvaders_4e6_1': 540.0,
    'spaceinvaders_4e6_2': 600.0,
    'spaceinvaders_5e6_0': 605.0,
    'spaceinvaders_5e6_1': 605.0,
    'spaceinvaders_5e6_2': 600.0,
    'spaceinvaders_6e6_0': 1110.0,
    'spaceinvaders_6e6_1': 615.0,
    'spaceinvaders_6e6_2': 1900.0,
    'spaceinvaders_7e6_0': 2160.0,
    'spaceinvaders_7e6_1': 1860.0,
    'spaceinvaders_7e6_2': 580.0,
    'spaceinvaders_8e6_0': 1895.0,
    'spaceinvaders_8e6_1': 1140.0,
    'spaceinvaders_8e6_2': 2915.0,
    'spaceinvaders_9e6_0': 2725.0,
    'spaceinvaders_9e6_1': 805.0,
    'spaceinvaders_9e6_2': 2960.0,
    'spaceinvaders_10e6_0': 2460.0,
    'spaceinvaders_10e6_1': 5210.0,
    'spaceinvaders_10e6_2': 4790.0,
    'pong_5e5_0': -21.0,
    'pong_5e5_1': -21.0,
    'pong_5e5_2': -21.0,
    'pong_10e5_0': -18.0,
    'pong_10e5_1': -19.0,
    'pong_10e5_2': -20.0,
    'pong_15e5_0': -13.0,
    'pong_15e5_1': -17.0,
    'pong_15e5_2': -14.0,
    'pong_20e5_0': -2.0,
    'pong_20e5_1': -6.0,
    'pong_20e5_2': 20.0,
    'pong_25e5_0': 20.0,
    'pong_25e5_1': 20.0,
    'pong_25e5_2': 20.0,
    'pong_30e5_0': 20.0,
    'pong_30e5_1': 20.0,
    'pong_30e5_2': 20.0,
    'pong_35e5_0': 20.0,
    'pong_35e5_1': 20.0,
    'pong_35e5_2': 20.0,
    'pong_40e5_0': 20.0,
    'pong_40e5_1': 20.0,
    'pong_40e5_2': 20.0,
    'pong_45e5_0': 20.0,
    'pong_45e5_1': 20.0,
    'pong_45e5_2': 20.0
}


@dataclass
class HumanMetrics:
    paths: List[Path]
    noop_ratios: np.ndarray
    reward_sums: np.ndarray
    action_counts: np.ndarray


@dataclass
class StepMetrics:
    step_key: str
    step_token: str
    thinker_noop_ratios: np.ndarray
    thinker_score_values: np.ndarray
    noop_stat: float
    noop_pvalue: float
    noop_method: str
    score_stat: float
    score_pvalue: float
    score_method: str
    human_mean_noop: float
    thinker_mean_noop: float
    human_mean_score: float
    thinker_mean_score: float
    shapiro_human_noop: float
    shapiro_thinker_noop: float
    shapiro_human_reward: float
    shapiro_thinker_reward: float


# Thinker 파일 단위 메트릭을 담기 위한 컨테이너
@dataclass
class ThinkerFileMetric:
    path: Path
    noop_ratio: float
    score: float


# ---------------------------------------------------------------------------
# Helpers -------------------------------------------------------------------
# ---------------------------------------------------------------------------

def _argmax_actions(actions: np.ndarray) -> np.ndarray:
    """Convert action arrays to indices."""
    if actions.ndim == 1:
        return actions.astype(int)
    if actions.ndim == 2:
        return actions.argmax(axis=1)
    raise ValueError(f"Unsupported action array shape {actions.shape}")


def _shapiro_safe(values: np.ndarray) -> float:
    """Shapiro-Wilk test p-value or NaN if undefined."""
    values = np.asarray(values, dtype=float)
    if values.size < 3 or values.size > 5000 or np.all(values == values[0]):
        return math.nan
    stat, p_value = stats.shapiro(values)
    return float(p_value)


def _compare_samples(human: np.ndarray, thinker: np.ndarray) -> Tuple[float, float, str]:
    """Compare two samples using Welch t-test or Mann-Whitney U."""
    human = np.asarray(human, dtype=float)
    thinker = np.asarray(thinker, dtype=float)
    mask_h = np.isfinite(human)
    mask_t = np.isfinite(thinker)
    human = human[mask_h]
    thinker = thinker[mask_t]
    if human.size < 2 or thinker.size < 2:
        return math.nan, math.nan, "insufficient"

    p_h = _shapiro_safe(human)
    p_t = _shapiro_safe(thinker)

    if (np.isnan(p_h) or p_h > 0.05) and (np.isnan(p_t) or p_t > 0.05):
        stat, p_value = stats.ttest_ind(human, thinker, equal_var=False)
        method = "welch-t"
    else:
        stat, p_value = stats.mannwhitneyu(human, thinker, alternative="two-sided")
        method = "mannwhitney"
    return float(stat), float(p_value), method


def _load_human_metrics(subject: int, game_number: int, human_dir: Path) -> HumanMetrics:
    target_root = human_dir / f"sub_{subject}" / f"game_{game_number}"
    if not target_root.exists():
        raise FileNotFoundError(f"Human data directory not found: {target_root}")

    paths = sorted(target_root.rglob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No human behavioral files under {target_root}")

    noop_ratios: List[float] = []
    reward_sums: List[float] = []
    action_counts = np.zeros((0,), dtype=np.int64)

    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            actions = _argmax_actions(data["action"])
            rewards = data["reward"].astype(float)
        noop_ratios.append(float(np.mean(actions == 0)))
        reward_sums.append(float(np.sum(rewards)))
        valid_actions = actions[actions >= 0]
        if valid_actions.size:
            max_action = int(valid_actions.max())
            if max_action >= action_counts.size:
                action_counts = np.pad(
                    action_counts, (0, max_action + 1 - action_counts.size)
                )
            action_counts[:max_action + 1] += np.bincount(
                valid_actions, minlength=max_action + 1
            )

    return HumanMetrics(paths=paths,
                        noop_ratios=np.asarray(noop_ratios, dtype=float),
                        reward_sums=np.asarray(reward_sums, dtype=float),
                        action_counts=action_counts)


def _load_thinker_file_metrics(thinker_dir: Path) -> Dict[str, List[ThinkerFileMetric]]:
    per_step: Dict[str, List[ThinkerFileMetric]] = {}
    for path in sorted(thinker_dir.glob("*.npy")):
        stem = path.stem  # e.g. spaceinvaders_1e6_0
        parts = stem.split("_")
        if len(parts) < 3:
            continue
        game_token = "_".join(parts[:-2])
        step_token = parts[-2]
        step_key = f"{game_token}_{step_token}"

        data = np.load(path, allow_pickle=True).item()
        status = np.asarray(data.get("status", []), dtype=int)
        tree_reps = data.get("tree_reps", {})
        raw_actions = tree_reps.get("cur_action")
        if raw_actions is None:
            continue
        action_indices = _argmax_actions(np.asarray(raw_actions))
        filtered = action_indices[status == 0]
        if filtered.size == 0:
            continue
        noop_ratio = float(np.mean(filtered == 0))
        score = float(THINKER_SCORE_PLACEHOLDER.get(stem, math.nan))
        metric = ThinkerFileMetric(path=path, noop_ratio=noop_ratio, score=score)
        per_step.setdefault(step_key, []).append(metric)
    return per_step


def _build_step_metrics(human: HumanMetrics,
                        thinker_files: Dict[str, List[ThinkerFileMetric]]) -> List[StepMetrics]:
    results: List[StepMetrics] = []
    if not thinker_files:
        return results

    for step_key, file_metrics in thinker_files.items():
        game_token, step_token = step_key.rsplit("_", 1)
        if not file_metrics:
            continue

        thinker_noop_full = np.asarray([fm.noop_ratio for fm in file_metrics], dtype=float)
        thinker_scores_full = np.asarray([fm.score for fm in file_metrics], dtype=float)

        noop_stat, noop_pvalue, noop_method = _compare_samples(human.noop_ratios, thinker_noop_full)
        score_stat, score_pvalue, score_method = _compare_samples(human.reward_sums, thinker_scores_full)

        finite_thinker_noop = thinker_noop_full[np.isfinite(thinker_noop_full)]
        finite_thinker_scores = thinker_scores_full[np.isfinite(thinker_scores_full)]

        human_mean_noop = float(np.mean(human.noop_ratios)) if human.noop_ratios.size else math.nan
        thinker_mean_noop = float(np.mean(finite_thinker_noop)) if finite_thinker_noop.size else math.nan
        human_mean_score = float(np.mean(human.reward_sums)) if human.reward_sums.size else math.nan
        thinker_mean_score = float(np.mean(finite_thinker_scores)) if finite_thinker_scores.size else math.nan

        results.append(StepMetrics(
            step_key=step_key,
            step_token=step_token,
            thinker_noop_ratios=thinker_noop_full,
            thinker_score_values=thinker_scores_full,
            noop_stat=noop_stat,
            noop_pvalue=noop_pvalue,
            noop_method=noop_method,
            score_stat=score_stat,
            score_pvalue=score_pvalue,
            score_method=score_method,
            human_mean_noop=human_mean_noop,
            thinker_mean_noop=thinker_mean_noop,
            human_mean_score=human_mean_score,
            thinker_mean_score=thinker_mean_score,
            shapiro_human_noop=_shapiro_safe(human.noop_ratios),
            shapiro_thinker_noop=_shapiro_safe(finite_thinker_noop),
            shapiro_human_reward=_shapiro_safe(human.reward_sums),
            shapiro_thinker_reward=_shapiro_safe(finite_thinker_scores),
        ))
    return results


def _step_sort_key(step_token: str) -> float:
    try:
        return float(step_token)
    except ValueError:
        try:
            return float(step_token.replace("e", "e"))
        except ValueError:
            return float("inf")


def _plot_metrics(step_metrics: List[StepMetrics],
                  output_dir: Path,
                  subject: int,
                  game_number: int) -> Path:
    if not step_metrics:
        raise ValueError("No step metrics available for plotting")

    step_metrics = sorted(step_metrics, key=lambda sm: _step_sort_key(sm.step_token))
    steps = [sm.step_token for sm in step_metrics]
    noop_diffs = [sm.human_mean_noop - sm.thinker_mean_noop for sm in step_metrics]
    score_diffs = [sm.human_mean_score - sm.thinker_mean_score for sm in step_metrics]

    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(steps, noop_diffs, marker="o", linestyle="-", color="#1f77b4")
    axes[0].axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    axes[0].set_ylabel("Mean noop diff")
    axes[0].set_title("Human vs Thinker noop difference")

    axes[1].plot(steps, score_diffs, marker="s", linestyle="-", color="#d62728")
    axes[1].axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    axes[1].set_ylabel("Mean score diff")
    axes[1].set_xlabel("Thinker step token")
    axes[1].set_title("Human vs Thinker score difference")

    fig.suptitle(f"Subject {subject} / Game {game_number}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / f"behavior_similarity_sub{subject}_game{game_number}.png"
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    return figure_path


def _summarize(step_metrics: Sequence[StepMetrics]) -> None:
    header = (
        "step_key",
        "noop_method",
        "noop_stat",
        "noop_p",
        "human_noop_mean",
        "thinker_noop_mean",
        "score_method",
        "score_stat",
        "score_p",
        "human_score_mean",
        "thinker_score_mean",
    )
    print("\t".join(header))
    for sm in step_metrics:
        row = [
            sm.step_key,
            sm.noop_method,
            f"{sm.noop_stat:.4f}" if math.isfinite(sm.noop_stat) else "nan",
            f"{sm.noop_pvalue:.4g}" if math.isfinite(sm.noop_pvalue) else "nan",
            f"{sm.human_mean_noop:.4f}" if math.isfinite(sm.human_mean_noop) else "nan",
            f"{sm.thinker_mean_noop:.4f}" if math.isfinite(sm.thinker_mean_noop) else "nan",
            sm.score_method,
            f"{sm.score_stat:.4f}" if math.isfinite(sm.score_stat) else "nan",
            f"{sm.score_pvalue:.4g}" if math.isfinite(sm.score_pvalue) else "nan",
            f"{sm.human_mean_score:.4f}" if math.isfinite(sm.human_mean_score) else "nan",
            f"{sm.thinker_mean_score:.4f}" if math.isfinite(sm.thinker_mean_score) else "nan",
        ]
        print("\t".join(row))


# ---------------------------------------------------------------------------
# Entry point ---------------------------------------------------------------
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare human behavior with Thinker checkpoints.")
    parser.add_argument("--subject", type=int, required=True, help="Human subject id (1-6)")
    parser.add_argument("--game-number", type=int, required=True, help="Game number (0-2)")
    parser.add_argument("--thinker-dir", type=Path, required=True, help="Directory with Thinker *.npy files")
    parser.add_argument("--human-dir", type=Path, required=True, help="Root directory for human behavioral data")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to store output plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    human = _load_human_metrics(args.subject, args.game_number, args.human_dir)
    thinker_files = _load_thinker_file_metrics(args.thinker_dir)
    step_metrics = _build_step_metrics(human, thinker_files)
    if not step_metrics:
        raise RuntimeError("No Thinker data available after filtering status == 0")
    figure_path = _plot_metrics(step_metrics, args.output_dir, args.subject, args.game_number)
    _summarize(step_metrics)
    print(f"Saved plot to {figure_path}")


if __name__ == "__main__":
    main()
