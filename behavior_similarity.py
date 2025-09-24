"""Utilities to compare human behavioral data with Thinker checkpoints."""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

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


@dataclass
class StepMetrics:
    step_key: str
    step_token: str
    thinker_noop_ratios: np.ndarray
    thinker_score_values: np.ndarray
    noop_corr: float
    noop_pvalue: float
    noop_method: str
    score_corr: float
    score_pvalue: float
    score_method: str
    shapiro_human_noop: float
    shapiro_thinker_noop: float
    shapiro_human_reward: float
    shapiro_thinker_reward: float


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


def _group_values(values: Sequence[float], groups: int) -> np.ndarray:
    """Aggregate values into a fixed number of groups via averaging."""
    if groups <= 0:
        return np.array([], dtype=float)
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.array([], dtype=float)
    groups = min(groups, values.size)
    split = np.array_split(values, groups)
    out = []
    for chunk in split:
        if chunk.size:
            out.append(float(np.mean(chunk)))
    return np.asarray(out, dtype=float)


def _group_action_ratios(actions: Sequence[int], groups: int) -> np.ndarray:
    """Aggregate noop ratios for a fixed number of groups."""
    if groups <= 0:
        return np.array([], dtype=float)
    actions = np.asarray(actions, dtype=int)
    if actions.size == 0:
        return np.array([], dtype=float)
    groups = max(1, min(groups, actions.size))
    split = np.array_split(actions, groups)
    ratios = []
    for chunk in split:
        if chunk.size:
            ratios.append(float(np.mean(chunk == 0)))
    return np.asarray(ratios, dtype=float)


def _shapiro_safe(values: np.ndarray) -> float:
    """Shapiro-Wilk test p-value or NaN if undefined."""
    values = np.asarray(values, dtype=float)
    if values.size < 3 or values.size > 5000 or np.all(values == values[0]):
        return math.nan
    stat, p_value = stats.shapiro(values)
    return float(p_value)


def _correlate(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, str]:
    """Choose Pearson vs Spearman based on normality."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return math.nan, math.nan, "insufficient"
    a = a[mask]
    b = b[mask]

    p_a = _shapiro_safe(a)
    p_b = _shapiro_safe(b)
    if (not np.isnan(p_a) and p_a <= 0.05) or (not np.isnan(p_b) and p_b <= 0.05):
        corr, p_value = stats.spearmanr(a, b)
        method = "spearman"
    else:
        corr, p_value = stats.pearsonr(a, b)
        method = "pearson"

    if isinstance(corr, np.ndarray):
        corr = corr[0, 1]
    return float(corr), float(p_value), method


def _load_human_metrics(subject: int, game_number: int, human_dir: Path) -> HumanMetrics:
    target_root = human_dir / f"sub_{subject}" / f"game_{game_number}"
    if not target_root.exists():
        raise FileNotFoundError(f"Human data directory not found: {target_root}")

    paths = sorted(target_root.rglob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No human behavioral files under {target_root}")

    noop_ratios: List[float] = []
    reward_sums: List[float] = []

    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            actions = _argmax_actions(data["action"])
            rewards = data["reward"].astype(float)
        noop_ratios.append(float(np.mean(actions == 0)))
        reward_sums.append(float(np.sum(rewards)))

    return HumanMetrics(paths=paths,
                        noop_ratios=np.asarray(noop_ratios, dtype=float),
                        reward_sums=np.asarray(reward_sums, dtype=float))


def _load_thinker_actions(thinker_dir: Path) -> Dict[str, List[int]]:
    step_actions: Dict[str, List[int]] = {}
    for path in sorted(thinker_dir.glob("*.npy")):
        stem = path.stem  # e.g. spaceinvaders_1e6_0
        parts = stem.split("_")
        if len(parts) < 3:
            continue
        game_token = "_".join(parts[:-2])
        step_token = parts[-2]
        step_key = f"{game_token}_{step_token}"

        data = np.load(path, allow_pickle=True).item()
        status = np.asarray(data["status"], dtype=int)
        if "tree_reps" not in data or "cur_action" not in data["tree_reps"]:
            continue
        raw_actions = np.asarray(data["tree_reps"]["cur_action"])
        action_indices = _argmax_actions(raw_actions)
        filtered = action_indices[status == 0]
        if filtered.size == 0:
            continue
        bucket = step_actions.setdefault(step_key, [])
        bucket.extend(filtered.tolist())
    return step_actions


def _collect_score_placeholders(file_stems: Iterable[str]) -> np.ndarray:
    scores = []
    for stem in file_stems:
        scores.append(float(THINKER_SCORE_PLACEHOLDER.get(stem, math.nan)))
    return np.asarray(scores, dtype=float)


def _group_score_placeholders(step_key: str,
                              file_stems: Sequence[str],
                              target_groups: int) -> Tuple[np.ndarray, float]:
    raw_scores = _collect_score_placeholders(file_stems)
    if raw_scores.size == 0:
        return np.array([], dtype=float), math.nan
    target_groups = max(1, min(target_groups, raw_scores.size))
    grouped: List[float] = []
    for chunk in np.array_split(raw_scores, target_groups):
        if chunk.size == 0:
            grouped.append(math.nan)
            continue
        finite = chunk[np.isfinite(chunk)]
        grouped.append(float(np.mean(finite)) if finite.size else math.nan)
    grouped_arr = np.asarray(grouped, dtype=float)
    finite_all = raw_scores[np.isfinite(raw_scores)]
    fallback = float(np.mean(finite_all)) if finite_all.size else math.nan
    return grouped_arr, fallback


def _build_step_metrics(human: HumanMetrics,
                        thinker_actions: Dict[str, List[int]],
                        thinker_dir: Path) -> List[StepMetrics]:
    results: List[StepMetrics] = []
    if not thinker_actions:
        return results

    for step_key, actions in thinker_actions.items():
        game_token, step_token = step_key.rsplit("_", 1)
        related_files = sorted(
            thinker_dir.glob(f"{game_token}_{step_token}_*.npy"))
        file_stems = [p.stem for p in related_files]

        group_base = len(related_files) if related_files else 1
        group_count = max(1, min(group_base, human.noop_ratios.size))

        thinker_noop = _group_action_ratios(actions, group_count)
        human_noop = _group_values(human.noop_ratios, group_count)
        noop_corr, noop_pvalue, noop_method = _correlate(human_noop, thinker_noop)

        thinker_scores, fallback = _group_score_placeholders(step_key, file_stems, group_count)
        if thinker_scores.size == 0:
            thinker_scores = np.full(group_count, fallback, dtype=float)
        human_rewards = _group_values(human.reward_sums, thinker_scores.size)
        score_corr, score_pvalue, score_method = _correlate(human_rewards, thinker_scores)

        finite_thinker_scores = thinker_scores[np.isfinite(thinker_scores)]
        finite_thinker_noop = thinker_noop[np.isfinite(thinker_noop)]

        results.append(StepMetrics(
            step_key=step_key,
            step_token=step_token,
            thinker_noop_ratios=thinker_noop,
            thinker_score_values=thinker_scores,
            noop_corr=noop_corr,
            noop_pvalue=noop_pvalue,
            noop_method=noop_method,
            score_corr=score_corr,
            score_pvalue=score_pvalue,
            score_method=score_method,
            shapiro_human_noop=_shapiro_safe(human_noop),
            shapiro_thinker_noop=_shapiro_safe(finite_thinker_noop),
            shapiro_human_reward=_shapiro_safe(human_rewards),
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
    noop_corrs = [sm.noop_corr for sm in step_metrics]
    score_corrs = [sm.score_corr for sm in step_metrics]

    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(steps, noop_corrs, marker="o", linestyle="-", color="#1f77b4")
    axes[0].axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    axes[0].set_ylabel("Noop Correlation")
    axes[0].set_title("Noop frequency similarity vs Thinker step")

    axes[1].plot(steps, score_corrs, marker="s", linestyle="-", color="#d62728")
    axes[1].axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    axes[1].set_ylabel("Score Correlation")
    axes[1].set_xlabel("Thinker step token")
    axes[1].set_title("Reward similarity vs Thinker step")

    fig.suptitle(f"Subject {subject} / Game {game_number}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / f"behavior_similarity_sub{subject}_game{game_number}.png"
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    return figure_path


def _summarize(step_metrics: Sequence[StepMetrics]) -> None:
    header = (
        "step_key", "noop_corr", "noop_p", "noop_method",
        "score_corr", "score_p", "score_method"
    )
    print("\t".join(header))
    for sm in step_metrics:
        row = [
            sm.step_key,
            f"{sm.noop_corr:.4f}" if math.isfinite(sm.noop_corr) else "nan",
            f"{sm.noop_pvalue:.4g}" if math.isfinite(sm.noop_pvalue) else "nan",
            sm.noop_method,
            f"{sm.score_corr:.4f}" if math.isfinite(sm.score_corr) else "nan",
            f"{sm.score_pvalue:.4g}" if math.isfinite(sm.score_pvalue) else "nan",
            sm.score_method,
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
    thinker_actions = _load_thinker_actions(args.thinker_dir)
    step_metrics = _build_step_metrics(human, thinker_actions, args.thinker_dir)
    if not step_metrics:
        raise RuntimeError("No Thinker data available after filtering status == 0")
    figure_path = _plot_metrics(step_metrics, args.output_dir, args.subject, args.game_number)
    _summarize(step_metrics)
    print(f"Saved plot to {figure_path}")


if __name__ == "__main__":
    main()
