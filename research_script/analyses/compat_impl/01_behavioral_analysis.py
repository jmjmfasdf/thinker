#!/usr/bin/env python3
"""
Behavioral analysis for the figure set in research plan 01.

Figures generated here (the numbering contract is shared by every dataset):
1-1 : Withholding bout schematic + action distribution
1-2 : NOOP prevalence and across-day repeatability
1-3 : Post-NOOP versus other-active reward over a fixed time horizon
1-4 : Post-NOOP reward difference in task context
1-5 : Leave-one-day-out duration-dependent hazard generalization
1-6 : Censoring-aware NOOP duration and hazard analysis
1-7 : Censoring-aware one- versus two-component duration distributions
1-8 : NOOP and active-action persistence with competing transition risks

Input  : ../behavioral_data_block/ or ../behavioral_data_block_mario/
Output : ../research_script/outputs/01_behavioral_analysis/{atari|mario}/
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mpl_colors
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from scipy import optimize, stats
from scipy.special import expit, logsumexp

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Behavioral withholding analysis by dataset.")
    parser.add_argument(
        "--dataset",
        choices=("auto", "atari", "mario"),
        default="auto",
        help=(
            "Dataset and output subfolder. With 'auto', behavioral_data_block_mario "
            "(or any input root containing manifest.tsv) is treated as Mario; otherwise Atari."
        ),
    )
    parser.add_argument("--data-root", type=Path, default=None, help="Optional input-root override.")
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=499,
        help="Parametric-bootstrap repetitions for duration-hazard tests.",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default=None,
        help="Optional comma-separated Mario stages, e.g. W1-1,W1-2 (smoke tests or subsets).",
    )
    parser.add_argument(
        "--reward-horizon-seconds",
        type=float,
        default=1.0,
        help=(
            "Canonical physical-time horizon for post-action reward comparisons. "
            "The numbered output contract is fixed at 1.0 s."
        ),
    )
    return parser.parse_args()


ARGS = parse_args()
requested_data_root = ARGS.data_root.expanduser().resolve() if ARGS.data_root is not None else None
if ARGS.dataset == "auto":
    if requested_data_root is None:
        DATASET = "atari"
    else:
        DATASET = (
            "mario"
            if (requested_data_root / "manifest.tsv").exists()
            or "mario" in requested_data_root.name.lower()
            else "atari"
        )
else:
    DATASET = ARGS.dataset
DATA_ROOT = requested_data_root or ROOT / (
    "behavioral_data_block" if DATASET == "atari" else "behavioral_data_block_mario"
)
THINKER_ROOT = DATA_ROOT / "thinker"
OUT_DIR = Path(__file__).parent / "outputs" / "01" / "behavior" / DATASET
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# These panels were explicitly retired from the shared figure contract.  Remove
# stale renders so a rerun cannot leave an obsolete figure beside new results.
RETIRED_FIGURES = {
    "atari": (
        "fig_1-2_individual_differences.png",
        "fig_1-4_noopratio_postnoop_reward.png",
        "fig_1-5_alternative_exclusion.png",
    ),
    "mario": (
        "fig_1-2_noop_prevalence_by_stage.png",
        "fig_1-5_temporal_structure_by_stage.png",
    ),
}
for retired_name in RETIRED_FIGURES[DATASET]:
    (FIG_DIR / retired_name).unlink(missing_ok=True)

RETIRED_TABLES = {
    "atari": (
        "1-2_icc_by_subject_game.csv",
        "1-3_bout_lengths.csv",
        "1-3_effect_direction.csv",
        "1-4_episode_position_thirds.csv",
        "1-4_noop_autocorrelation.csv",
    ),
    "mario": (
        "1-2_icc_by_subject_game.csv",
        "1-3_reward_by_subject_stage.csv",
        "1-4_postnoop_reward_by_subject_stage.csv",
        "1-4_postnoop_reward_episode_detail.csv",
    ),
}
for retired_name in RETIRED_TABLES[DATASET]:
    (OUT_DIR / retired_name).unlink(missing_ok=True)

MARIO_MANIFEST: pd.DataFrame | None = None
MARIO_STAGE_BY_FILENAME: Dict[str, int] = {}

if DATASET == "atari":
    SUBJECTS = [f"sub_{i}" for i in range(1, 7)]
    GAMES = {1: "Pong", 2: "SpaceInvaders"}
    GAME_TITLES = {1: "Pong", 2: "Space Invaders"}
    ACTION_LABELS = ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"]
else:
    manifest_path = DATA_ROOT / "manifest.tsv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Mario manifest not found: {manifest_path}")
    MARIO_MANIFEST = pd.read_csv(manifest_path, sep="\t")
    MARIO_MANIFEST["stage_id"] = (
        MARIO_MANIFEST["world"].astype(int) * 10 + MARIO_MANIFEST["level"].astype(int)
    )
    MARIO_MANIFEST["stage_name"] = (
        "W"
        + MARIO_MANIFEST["world"].astype(int).astype(str)
        + "-"
        + MARIO_MANIFEST["level"].astype(int).astype(str)
    )
    GAMES = {
        int(stage_id): str(group["stage_name"].iloc[0])
        for stage_id, group in MARIO_MANIFEST.groupby("stage_id", sort=True)
    }
    if ARGS.stages:
        requested = {token.strip().upper() for token in ARGS.stages.split(",") if token.strip()}
        GAMES = {
            stage_id: stage_name
            for stage_id, stage_name in GAMES.items()
            if stage_name.upper() in requested or str(stage_id) in requested
        }
        if not GAMES:
            raise ValueError(f"None of the requested Mario stages were found: {ARGS.stages}")
    GAME_TITLES = dict(GAMES)
    MARIO_STAGE_BY_FILENAME = {
        Path(str(row.output_path)).name: int(row.stage_id)
        for row in MARIO_MANIFEST.itertuples(index=False)
    }
    subject_ids = sorted(
        int(path.name.split("-")[1]) for path in DATA_ROOT.glob("sub-*") if path.is_dir()
    )
    SUBJECTS = [f"sub_{subject_id}" for subject_id in subject_ids]
    vocab_path = DATA_ROOT / "action_vocab.json"
    if not vocab_path.exists():
        raise FileNotFoundError(f"Mario action vocabulary not found: {vocab_path}")
    vocab_payload = json.loads(vocab_path.read_text())
    ACTION_LABELS = [
        str(item["name"])
        for item in sorted(vocab_payload["actions"], key=lambda item: int(item["index"]))
    ]

NOOP_ACTION = 0
N_ACTIONS = len(ACTION_LABELS)
ACTION_MAPPING = {label: index for index, label in enumerate(ACTION_LABELS)}
CHANCE_NOOP = 1.0 / N_ACTIONS
REWARD_HORIZON_SECONDS = float(ARGS.reward_horizon_seconds)
if not np.isfinite(REWARD_HORIZON_SECONDS) or REWARD_HORIZON_SECONDS <= 0:
    raise ValueError("--reward-horizon-seconds must be a positive finite value")
if not np.isclose(REWARD_HORIZON_SECONDS, 1.0, rtol=0.0, atol=1e-12):
    raise ValueError(
        "The canonical numbered outputs are fixed to a 1.0-s reward horizon; "
        "use a separate sensitivity script/output root for other horizons."
    )
SURVIVAL_BOOTSTRAPS = max(int(ARGS.bootstrap), 1)
SURVIVAL_RANDOM_SEED = 260803
ICC_BOOTSTRAPS = 2000
ICC_RANDOM_SEED = 120521
SUBJECT_DIFFERENCE_PERMUTATIONS = 199_999
SUBJECT_DIFFERENCE_RANDOM_SEED = 720521

EPS = 1e-12
WITH_COLOR = "#2ca02c"
NON_COLOR = "#d62728"
LINE_COLOR = "#7f7f7f"
MEAN_COLOR = "#111111"
RIGHT_AXIS_COLOR = NON_COLOR
RIGHT_AXIS_GAMES = {2} if DATASET == "atari" else set()
if DATASET == "atari":
    GAME_COLORS = {1: WITH_COLOR, 2: NON_COLOR}
else:
    stage_keys = list(GAMES)
    GAME_COLORS = {
        stage_id: plt.cm.viridis(index / max(len(stage_keys) - 1, 1))
        for index, stage_id in enumerate(stage_keys)
    }
GAME_NAME_COLORS = {GAMES[game]: GAME_COLORS[game] for game in GAMES}
SUBJECT_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
]
SUBJECT_COLOR_MAP = {
    subject: SUBJECT_COLORS[idx]
    for idx, subject in enumerate(SUBJECTS)
}


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def save_withholding_schematic(out_path: Path) -> None:
    """Save the English withholding-bout schematic extracted from the sub001 analysis."""
    fig, ax = plt.subplots(figsize=(6.8, 2.8))
    ax.plot([0, 1], [0, 0], color="black", lw=1.5)
    ax.plot([0.25, 0.65], [0, 0], color="#ffd166", lw=9, solid_capstyle="butt")
    ax.scatter([0.25, 0.65, 0.8], [0, 0, 0], color=["#ef476f", "#ef476f", "#06d6a0"], s=65, zorder=3)
    ax.text(0.25, 0.08, "NOOP onset", ha="center")
    ax.text(0.65, 0.08, "NOOP end", ha="center")
    ax.text(0.8, 0.08, "Action commit", ha="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.2, 0.25)
    ax.set_axis_off()
    ax.set_title("1-1A: Withholding bout schematic")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reward figure helpers
# ---------------------------------------------------------------------------

def stars_for_pvalue(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "n/a"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def format_pvalue(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "n/a"
    if p_value < 0.001:
        return "< 0.001"
    return f"= {p_value:.3f}"


def shapiro_safe(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 3 or len(arr) > 5000:
        return np.nan
    if np.nanstd(arr) < EPS:
        return np.nan
    return float(stats.shapiro(arr).pvalue)


def choose_paired_test(with_vals: Sequence[float], non_vals: Sequence[float]) -> Dict[str, float | str]:
    with_arr = np.asarray(with_vals, dtype=float)
    non_arr = np.asarray(non_vals, dtype=float)
    mask = np.isfinite(with_arr) & np.isfinite(non_arr)
    with_arr = with_arr[mask]
    non_arr = non_arr[mask]
    diff = with_arr - non_arr

    result: Dict[str, float | str] = {
        "n_subjects": int(len(with_arr)),
        "shapiro_p": np.nan,
        "test_name": "n/a",
        "statistic": np.nan,
        "p_value": np.nan,
        "significance": "n/a",
    }
    if len(with_arr) == 0:
        return result

    shapiro_p = shapiro_safe(diff)
    result["shapiro_p"] = shapiro_p

    if len(with_arr) >= 3 and np.isfinite(shapiro_p) and shapiro_p > 0.05:
        test = stats.ttest_rel(with_arr, non_arr, nan_policy="omit")
        result["test_name"] = "paired t-test"
        result["statistic"] = float(test.statistic)
        result["p_value"] = float(test.pvalue)
    else:
        if len(with_arr) < 2 or np.allclose(diff, 0.0):
            statistic = 0.0
            p_value = 1.0
        else:
            test = stats.wilcoxon(with_arr, non_arr, zero_method="wilcox")
            statistic = float(test.statistic)
            p_value = float(test.pvalue)
        result["test_name"] = "Wilcoxon signed-rank"
        result["statistic"] = statistic
        result["p_value"] = p_value

    result["significance"] = stars_for_pvalue(float(result["p_value"]))
    return result


def subject_sort_key(subject: str) -> int:
    match = re.search(r"(\d+)$", subject)
    return int(match.group(1)) if match else 9999


def make_limits(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return (-1.0, 1.0)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    span = max(hi - lo, 0.1)
    return (lo - 0.15 * span, hi + 0.30 * span)


def add_significance_annotation_with_limits(
    ax: plt.Axes,
    x_left: float,
    x_right: float,
    y_values: Sequence[float],
    label: str,
) -> None:
    finite = np.asarray(y_values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return

    y0, y1 = ax.get_ylim()
    span = max(y1 - y0, 0.1)
    h = 0.025 * span
    y = min(float(np.max(finite)) + 0.05 * span, y1 - h - 0.05 * span)
    ax.plot([x_left, x_left, x_right, x_right], [y, y + h, y + h, y], color="black", lw=1.0)
    ax.text(
        (x_left + x_right) / 2,
        y + h + 0.01 * span,
        label,
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )


def build_reward_episode_summary(
    episodes: Sequence[Dict],
    horizon_seconds: float = REWARD_HORIZON_SECONDS,
) -> pd.DataFrame:
    """Summarize reward in a fixed physical-time window after each active action.

    For an active action at time ``t``, the outcome is the sum of rewards at
    timestamps in ``(t, t + horizon_seconds]``.  A true terminal is followed
    conceptually by zero reward; only windows cut short by a nonterminal file
    boundary are excluded.  Thus every dataset has the same exposure time.
    """
    if not np.isfinite(horizon_seconds) or horizon_seconds <= 0:
        raise ValueError("horizon_seconds must be positive and finite")
    rows: List[Dict[str, object]] = []

    for ep in episodes:
        actions = np.asarray(ep["actions"], dtype=int).reshape(-1)
        rewards = np.asarray(ep["rewards"], dtype=float).reshape(-1)
        times = np.asarray(ep["times"], dtype=float).reshape(-1)
        n = min(len(actions), len(rewards), len(times))
        actions = actions[:n]
        rewards = rewards[:n]
        times = times[:n]
        if n < 2:
            continue
        if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
            raise RuntimeError(
                "Reward analysis requires finite, strictly increasing timestamps: "
                f"{ep.get('source_file', 'unknown source')}"
            )

        prev_actions = np.full(n, -1, dtype=int)
        prev_actions[1:] = actions[:-1]
        with_rewards: List[float] = []
        non_rewards: List[float] = []
        n_incomplete_horizon = 0

        for step_index in np.where(actions != NOOP_ACTION)[0]:
            horizon_end = float(times[step_index] + horizon_seconds)
            if float(times[-1]) + 1e-9 < horizon_end and not bool(ep.get("terminated", False)):
                n_incomplete_horizon += 1
                continue
            end_index = min(int(np.searchsorted(times, horizon_end, side="right")), n)
            future_reward = float(np.nansum(rewards[step_index + 1 : end_index]))
            if prev_actions[step_index] == NOOP_ACTION:
                with_rewards.append(future_reward)
            else:
                non_rewards.append(future_reward)

        subject = str(ep["subject"])
        rows.append(
            {
                "subject": subject,
                "subject_label": f"S{subject_sort_key(subject)}",
                "game": int(ep["game"]),
                "game_name": str(ep["game_name"]),
                "day": int(ep["day"]),
                "block": int(ep["block"]),
                "ep_idx": int(ep["ep_idx"]),
                "reward_horizon_seconds": float(horizon_seconds),
                "n_active_total": int(np.sum(actions != NOOP_ACTION)),
                "n_incomplete_horizon": int(n_incomplete_horizon),
                "n_withholding_preceded": int(len(with_rewards)),
                "n_not_preceded": int(len(non_rewards)),
                "reward_withholding_preceded_sum": float(np.sum(with_rewards)),
                "reward_not_preceded_sum": float(np.sum(non_rewards)),
                "reward_withholding_preceded_mean": float(np.mean(with_rewards)) if with_rewards else np.nan,
                "reward_not_preceded_mean": float(np.mean(non_rewards)) if non_rewards else np.nan,
            }
        )

    columns = [
        "subject",
        "subject_label",
        "game",
        "game_name",
        "day",
        "block",
        "ep_idx",
        "reward_horizon_seconds",
        "n_active_total",
        "n_incomplete_horizon",
        "n_withholding_preceded",
        "n_not_preceded",
        "reward_withholding_preceded_sum",
        "reward_not_preceded_sum",
        "reward_withholding_preceded_mean",
        "reward_not_preceded_mean",
        "reward_difference",
        "episode_label",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)

    summary = pd.DataFrame(rows).sort_values(
        ["subject", "game", "day", "block", "ep_idx"]
    ).reset_index(drop=True)
    summary["reward_difference"] = (
        summary["reward_withholding_preceded_mean"] - summary["reward_not_preceded_mean"]
    )
    summary["episode_label"] = summary.apply(
        lambda row: (
            f"day{int(row['day']):02d}-block{int(row['block']):02d}-ep{int(row['ep_idx']):02d}"
        ),
        axis=1,
    )
    return summary


def summarize_reward_episode_subject_stats(episode_summary: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    grouping = ["subject", "subject_label", "game", "game_name"]

    for keys, group in episode_summary.groupby(grouping, sort=True):
        subject, subject_label, game, game_name = keys
        group = group.dropna(
            subset=["reward_withholding_preceded_mean", "reward_not_preceded_mean"]
        )
        stats_result = choose_paired_test(
            group["reward_withholding_preceded_mean"].to_numpy(dtype=float),
            group["reward_not_preceded_mean"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "subject": subject,
                "subject_label": subject_label,
                "game": int(game),
                "game_name": game_name,
                "n_episodes": int(len(group)),
                "mean_withholding_preceded": float(group["reward_withholding_preceded_mean"].mean()) if len(group) > 0 else np.nan,
                "mean_not_preceded": float(group["reward_not_preceded_mean"].mean()) if len(group) > 0 else np.nan,
                **stats_result,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "subject",
                "subject_label",
                "game",
                "game_name",
                "n_episodes",
                "mean_withholding_preceded",
                "mean_not_preceded",
                "n_subjects",
                "shapiro_p",
                "test_name",
                "statistic",
                "p_value",
                "significance",
            ]
        )
    return pd.DataFrame(rows).sort_values(["subject", "game"]).reset_index(drop=True)


def summarize_reward_subject_conditions(episode_summary: pd.DataFrame) -> pd.DataFrame:
    """Pool eligible events within subject while retaining episode audit counts."""
    if episode_summary.empty:
        return pd.DataFrame(
            columns=[
                "subject", "subject_label", "game", "game_name", "n_episodes",
                "mean_withholding_preceded", "mean_not_preceded", "mean_reward_difference",
            ]
        )
    pooled = (
        episode_summary.groupby(["subject", "subject_label", "game", "game_name"], as_index=False)
        .agg(
            n_episodes=("ep_idx", "size"),
            n_paired_episodes=("reward_difference", "count"),
            n_withholding_preceded=("n_withholding_preceded", "sum"),
            n_not_preceded=("n_not_preceded", "sum"),
            reward_withholding_preceded_sum=("reward_withholding_preceded_sum", "sum"),
            reward_not_preceded_sum=("reward_not_preceded_sum", "sum"),
            n_incomplete_horizon=("n_incomplete_horizon", "sum"),
        )
        .sort_values(["game", "subject"])
        .reset_index(drop=True)
    )
    pooled["mean_withholding_preceded"] = (
        pooled["reward_withholding_preceded_sum"]
        / pooled["n_withholding_preceded"].replace(0, np.nan)
    )
    pooled["mean_not_preceded"] = (
        pooled["reward_not_preceded_sum"]
        / pooled["n_not_preceded"].replace(0, np.nan)
    )
    pooled["mean_reward_difference"] = (
        pooled["mean_withholding_preceded"] - pooled["mean_not_preceded"]
    )
    return pooled


def exact_sign_flip_p(values: Sequence[float]) -> float:
    """Two-sided exact sign-flip p-value for a one-sample mean."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        return np.nan
    observed = abs(float(np.mean(arr)))
    signs = 1.0 - 2.0 * (
        (np.arange(2**n, dtype=np.uint64)[:, None] >> np.arange(n, dtype=np.uint64)) & 1
    ).astype(float)
    permuted = np.abs(np.mean(signs * arr[None, :], axis=1))
    return float(np.mean(permuted >= observed - 1e-15))


def summarize_reward_group_inference(subject_summary: pd.DataFrame) -> pd.DataFrame:
    """Subject-level condition effects with 95% CI and across-condition BH-FDR."""
    rows: List[Dict[str, object]] = []
    for (game, game_name), group in subject_summary.groupby(["game", "game_name"], sort=True):
        values = group["mean_reward_difference"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        n = len(values)
        mean = float(np.mean(values)) if n else np.nan
        sd = float(np.std(values, ddof=1)) if n > 1 else np.nan
        sem = float(sd / np.sqrt(n)) if n > 1 else np.nan
        if n > 1 and np.isfinite(sem) and sem > 0:
            t_stat = float(mean / sem)
            p_value = float(2.0 * stats.t.sf(abs(t_stat), df=n - 1))
            half_width = float(stats.t.ppf(0.975, df=n - 1) * sem)
        elif n > 0 and np.allclose(values, 0.0):
            t_stat, p_value, half_width = 0.0, 1.0, 0.0
        else:
            t_stat, p_value, half_width = np.nan, np.nan, np.nan
        rows.append(
            {
                "game": int(game),
                "game_name": str(game_name),
                "n_subjects": int(n),
                "mean_reward_difference": mean,
                "sd_subject_difference": sd,
                "sem_subject_difference": sem,
                "ci95_low": mean - half_width if np.isfinite(half_width) else np.nan,
                "ci95_high": mean + half_width if np.isfinite(half_width) else np.nan,
                "t_statistic": t_stat,
                "p_value": p_value,
                "exact_sign_flip_p": exact_sign_flip_p(values),
                "n_positive": int(np.sum(values > 0)),
            }
        )
    result = pd.DataFrame(rows).sort_values("game").reset_index(drop=True)
    if result.empty:
        return result
    result["q_fdr"] = benjamini_hochberg(result["p_value"].to_numpy(dtype=float))
    result["significance_fdr"] = result["q_fdr"].map(stars_for_pvalue)
    result["q_fdr_exact_sign_flip"] = benjamini_hochberg(
        result["exact_sign_flip_p"].to_numpy(dtype=float)
    )
    result["significance_exact_fdr"] = result["q_fdr_exact_sign_flip"].map(stars_for_pvalue)
    return result


def save_reward_condition_comparison_figure(
    subject_summary: pd.DataFrame,
    group_inference: pd.DataFrame,
    out_path: Path,
    condition_axis_label: str,
) -> None:
    """Show raw subject pairs and group 95% CIs for every task condition."""
    games = list(GAMES)
    labels = [GAMES[game] for game in games]
    legend_handles = [
        Line2D([0], [0], marker="o", color="#555555", markerfacecolor="#555555", lw=0, label="Post-NOOP active"),
        Line2D([0], [0], marker="o", color="#555555", markerfacecolor="white", lw=0, label="Other active"),
        Line2D([0], [0], marker="D", color="#111111", lw=1.3, label="Group mean ± 95% CI"),
    ]

    # Atari reward units differ strongly between games.  Preserve the shared
    # visual grammar but give each game its own y scale so Pong is not flattened.
    if DATASET == "atari":
        fig, axes = plt.subplots(1, len(games), figsize=(6.0 * len(games), 5.7), squeeze=False)
        inference = group_inference.set_index("game")
        for game_index, game in enumerate(games):
            ax = axes[0, game_index]
            subset = subject_summary[subject_summary["game"] == game]
            for subject in SUBJECTS:
                row = subset[subset["subject"] == subject]
                if row.empty:
                    continue
                left = float(row.iloc[0]["mean_withholding_preceded"])
                right = float(row.iloc[0]["mean_not_preceded"])
                color = SUBJECT_COLOR_MAP[subject]
                ax.plot([0, 1], [left, right], color=color, alpha=0.42, lw=1.0, zorder=1)
                ax.scatter(0, left, color=color, s=30, zorder=2)
                ax.scatter(1, right, facecolor="white", edgecolor=color, s=30, zorder=2)
            for x_pos, column in [(0, "mean_withholding_preceded"), (1, "mean_not_preceded")]:
                values = subset[column].to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                mean = float(np.mean(values))
                ci = float(stats.t.ppf(0.975, len(values) - 1) * stats.sem(values)) if len(values) > 1 else 0.0
                ax.errorbar(x_pos, mean, yerr=ci, fmt="D", color="#111111", capsize=4, lw=1.6, zorder=4)
            ax.axhline(0, color="#666666", lw=0.8, ls="--", alpha=0.7)
            ax.set_xticks([0, 1], ["Post-NOOP\nactive", "Other active"])
            ax.set_title(labels[game_index])
            ax.set_ylabel(f"Mean reward in next {REWARD_HORIZON_SECONDS:g} s")
            ax.grid(axis="y", alpha=0.22)
            ax.spines[["top", "right"]].set_visible(False)
            if game in inference.index:
                info = inference.loc[game]
                ax.text(
                    0.5,
                    0.98,
                    f"Δ={float(info['mean_reward_difference']):+.3g}, "
                    f"q_t={float(info['q_fdr']):.3g} {info['significance_fdr']}; "
                    f"{int(info['n_positive'])}/{int(info['n_subjects'])} positive",
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=9,
                )
        fig.suptitle(
            f"1-3: Reward after an active action over a fixed {REWARD_HORIZON_SECONDS:g}-s horizon",
            fontsize=14,
            y=0.99,
        )
        fig.legend(handles=legend_handles, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.935))
        fig.tight_layout(rect=[0, 0, 1, 0.86])
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return

    x = np.arange(len(games), dtype=float)
    offset = 0.17
    fig_width = max(8.0, 0.78 * len(games) + 3.0)
    fig, ax = plt.subplots(figsize=(fig_width, 6.2))

    for subject_index, subject in enumerate(SUBJECTS):
        color = SUBJECT_COLOR_MAP[subject]
        subset = subject_summary[subject_summary["subject"] == subject].set_index("game")
        for game_index, game in enumerate(games):
            if game not in subset.index:
                continue
            row = subset.loc[game]
            left = float(row["mean_withholding_preceded"])
            right = float(row["mean_not_preceded"])
            ax.plot(
                [x[game_index] - offset, x[game_index] + offset],
                [left, right],
                color=color, alpha=0.38, lw=0.9, zorder=1,
            )
            ax.scatter(x[game_index] - offset, left, color=color, s=22, alpha=0.82, zorder=2)
            ax.scatter(
                x[game_index] + offset, right, facecolor="white", edgecolor=color,
                linewidth=1.0, s=22, alpha=0.92, zorder=2,
            )

    for game_index, game in enumerate(games):
        subset = subject_summary[subject_summary["game"] == game]
        for direction, column in [(-1.0, "mean_withholding_preceded"), (1.0, "mean_not_preceded")]:
            values = subset[column].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if not len(values):
                continue
            mean = float(np.mean(values))
            if len(values) > 1:
                ci = float(stats.t.ppf(0.975, len(values) - 1) * stats.sem(values))
            else:
                ci = 0.0
            ax.errorbar(
                x[game_index] + direction * offset,
                mean,
                yerr=ci,
                fmt="D",
                color="#111111",
                markersize=5,
                capsize=3,
                lw=1.5,
                zorder=4,
            )

    ax.axhline(0, color="#666666", lw=0.8, ls="--", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45 if len(games) > 8 else 0, ha="right" if len(games) > 8 else "center")
    ax.set_xlabel(condition_axis_label)
    ax.set_ylabel(f"Mean reward in next {REWARD_HORIZON_SECONDS:g} s")
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        f"1-3: Reward after an active action over a fixed {REWARD_HORIZON_SECONDS:g}-s horizon",
        fontsize=14,
        y=0.99,
    )
    fig.legend(handles=legend_handles, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.935))
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_reward_difference_figure(
    subject_summary: pd.DataFrame,
    group_inference: pd.DataFrame,
    out_path: Path,
    condition_axis_label: str,
) -> None:
    """Show subject-level reward differences, group 95% CIs, and FDR stars."""
    games = list(GAMES)
    labels = [GAMES[game] for game in games]
    y = np.arange(len(games), dtype=float)
    fig_height = max(4.2, 0.43 * len(games) + 2.0)
    fig, ax = plt.subplots(figsize=(11.5, fig_height))
    subject_offsets = np.linspace(-0.16, 0.16, max(len(SUBJECTS), 1))

    for subject_index, subject in enumerate(SUBJECTS):
        subset = subject_summary[subject_summary["subject"] == subject].set_index("game")
        xs = [float(subset.loc[game, "mean_reward_difference"]) if game in subset.index else np.nan for game in games]
        ax.scatter(
            xs,
            y + subject_offsets[subject_index],
            s=27,
            color=SUBJECT_COLOR_MAP[subject],
            alpha=0.8,
            label=f"S{subject_sort_key(subject)}",
            zorder=2,
        )

    inference = group_inference.set_index("game")
    for game_index, game in enumerate(games):
        if game not in inference.index:
            continue
        row = inference.loc[game]
        mean = float(row["mean_reward_difference"])
        low = float(row["ci95_low"])
        high = float(row["ci95_high"])
        ax.errorbar(
            mean,
            y[game_index],
            xerr=[[mean - low], [high - mean]],
            fmt="D",
            color="#111111",
            markersize=5.5,
            capsize=3,
            lw=1.5,
            zorder=4,
        )
        star = str(row["significance_fdr"])
        ax.text(
            0.995,
            y[game_index],
            star,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.axvline(0, color="#555555", lw=1.0, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(
        f"Post-NOOP active − other-active reward (next {REWARD_HORIZON_SECONDS:g} s)"
    )
    ax.set_ylabel(condition_axis_label)
    ax.grid(axis="x", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    handles, labels_for_legend = ax.get_legend_handles_labels()
    fig.suptitle("1-4: Subject-level withholding reward difference", fontsize=14, y=0.995)
    fig.legend(
        handles,
        labels_for_legend,
        frameon=False,
        ncol=min(len(SUBJECTS), 6),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
    )
    fig.text(
        0.995,
        0.01,
        "Diamonds: mean ± 95% t CI; stars: parametric subject t test + BH-FDR; "
        "exact sign-flip p/q are reported in the companion table",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout(rect=[0, 0.045, 1, 0.88])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def axis_values_for_reward_games(
    episode_summary: pd.DataFrame,
    game_ids: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    left_vals: List[float] = []
    right_vals: List[float] = []

    for _, row in episode_summary.iterrows():
        vals = [
            row["reward_withholding_preceded_mean"],
            row["reward_not_preceded_mean"],
        ]
        if int(row["game"]) in game_ids and int(row["game"]) not in RIGHT_AXIS_GAMES:
            left_vals.extend(vals)
        if int(row["game"]) in game_ids and int(row["game"]) in RIGHT_AXIS_GAMES:
            right_vals.extend(vals)

    return np.asarray(left_vals, dtype=float), np.asarray(right_vals, dtype=float)


def plot_reward_subject_episode_panel(
    ax: plt.Axes,
    ax_right: plt.Axes | None,
    subject_episode_summary: pd.DataFrame,
    subject_episode_stats: pd.DataFrame,
    subject_label: str,
    game_ids: Sequence[int],
    left_limits: Tuple[float, float],
    right_limits: Tuple[float, float] | None,
    horizon_seconds: float,
) -> None:
    rng = np.random.default_rng(subject_sort_key(subject_label))
    xticks: List[float] = []
    xlabels: List[str] = []
    annotation_specs: List[Tuple[plt.Axes, float, float, np.ndarray, str]] = []

    for game_idx, game_id in enumerate(game_ids):
        x_with = game_idx * 3.0
        x_non = x_with + 1.0
        xticks.extend([x_with, x_non])
        xlabels.extend(["With", "Not"])
        target_ax = ax_right if ax_right is not None and game_id in RIGHT_AXIS_GAMES else ax

        game_rows = subject_episode_summary[
            subject_episode_summary["game"] == game_id
        ].dropna(
            subset=["reward_withholding_preceded_mean", "reward_not_preceded_mean"]
        )

        ax.text(
            (x_with + x_non) / 2,
            1.02,
            GAMES[game_id],
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=9,
            color=GAME_COLORS[game_id],
        )

        if len(game_rows) == 0:
            continue

        jitter = rng.uniform(-0.10, 0.10, size=len(game_rows))
        with_vals = game_rows["reward_withholding_preceded_mean"].to_numpy(dtype=float)
        non_vals = game_rows["reward_not_preceded_mean"].to_numpy(dtype=float)

        for j, (_, row) in zip(jitter, game_rows.iterrows()):
            target_ax.plot(
                [x_with + j, x_non + j],
                [row["reward_withholding_preceded_mean"], row["reward_not_preceded_mean"]],
                color=LINE_COLOR,
                lw=0.8,
                alpha=0.35,
                zorder=1,
            )

        target_ax.scatter(x_with + jitter, with_vals, s=16, color=WITH_COLOR, alpha=0.7, zorder=2)
        target_ax.scatter(x_non + jitter, non_vals, s=16, color=NON_COLOR, alpha=0.7, zorder=2)

        mean_with = float(np.mean(with_vals))
        mean_non = float(np.mean(non_vals))
        sem_with = float(stats.sem(with_vals)) if len(with_vals) > 1 else 0.0
        sem_non = float(stats.sem(non_vals)) if len(non_vals) > 1 else 0.0
        target_ax.errorbar(
            [x_with, x_non],
            [mean_with, mean_non],
            yerr=[sem_with, sem_non],
            color=MEAN_COLOR,
            lw=1.8,
            fmt="o",
            markersize=5,
            capsize=3,
            zorder=3,
        )

        ax.text(
            (x_with + x_non) / 2,
            0.02,
            f"n={len(game_rows)} eps",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#444444",
        )

        game_stats_row = subject_episode_stats[subject_episode_stats["game"] == game_id]
        label = "n/a"
        if len(game_stats_row) > 0:
            label = str(game_stats_row.iloc[0]["significance"])
        annotation_specs.append(
            (
                target_ax,
                x_with,
                x_non,
                np.concatenate([with_vals, non_vals]),
                label,
            )
        )

    ax.set_ylim(*left_limits)
    if ax_right is not None and right_limits is not None:
        ax_right.set_ylim(*right_limits)
        ax_right.tick_params(axis="y", colors=RIGHT_AXIS_COLOR, labelsize=8)
        ax_right.spines["right"].set_color(RIGHT_AXIS_COLOR)

    ax.set_xlim(-0.65, (len(game_ids) - 1) * 3.0 + 1.65)
    ax.set_xticks(xticks, xlabels)
    ax.set_title(subject_label)
    ax.grid(axis="y", alpha=0.3)
    if ax_right is not None:
        ax_right.set_xlim(ax.get_xlim())
        ax_right.grid(False)

    for target_ax, x_with, x_non, vals, label in annotation_specs:
        add_significance_annotation_with_limits(target_ax, x_with, x_non, vals, label)


def save_reward_subject_episode_figure(
    episode_summary: pd.DataFrame,
    episode_subject_stats: pd.DataFrame,
    out_path: Path,
    horizon_seconds: float = REWARD_HORIZON_SECONDS,
) -> None:
    game_ids = [1, 2]
    subjects = sorted(episode_summary["subject"].unique(), key=subject_sort_key)
    subject_labels = {
        subject: episode_summary.loc[
            episode_summary["subject"] == subject, "subject_label"
        ].iloc[0]
        for subject in subjects
    }

    left_vals, right_vals = axis_values_for_reward_games(episode_summary, game_ids)
    left_limits = make_limits(left_vals)
    right_limits = make_limits(right_vals) if np.isfinite(right_vals).any() else None

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=False)
    axes_flat = axes.flatten()
    right_axes: List[plt.Axes | None] = []

    for ax, subject in zip(axes_flat, subjects):
        subject_rows = episode_summary[episode_summary["subject"] == subject].copy()
        subject_stats = episode_subject_stats[episode_subject_stats["subject"] == subject].copy()
        ax_right = ax.twinx()
        right_axes.append(ax_right)
        plot_reward_subject_episode_panel(
            ax=ax,
            ax_right=ax_right,
            subject_episode_summary=subject_rows,
            subject_episode_stats=subject_stats,
            subject_label=subject_labels[subject],
            game_ids=game_ids,
            left_limits=left_limits,
            right_limits=right_limits,
            horizon_seconds=horizon_seconds,
        )

    for ax in axes_flat[len(subjects) :]:
        ax.set_axis_off()

    for ax in axes[:, 0]:
        ax.set_ylabel(f"Episode mean next-{horizon_seconds:g}-s reward\n(Pong scale)")

    for ax_right in right_axes:
        ax_right.set_ylabel(
            f"Episode mean next-{horizon_seconds:g}-s reward\n(SpaceInvaders scale)",
            color=RIGHT_AXIS_COLOR,
        )

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=WITH_COLOR, markeredgecolor=WITH_COLOR, markersize=6, label="Withholding-preceded episode mean"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=NON_COLOR, markeredgecolor=NON_COLOR, markersize=6, label="Not-preceded episode mean"),
        Line2D([0], [0], color=LINE_COLOR, lw=1.0, alpha=0.5, label="Episode paired line"),
        Line2D([0], [0], marker="o", color=MEAN_COLOR, markersize=6, lw=1.8, label="Episode mean ± SEM"),
        Line2D([0], [0], color="black", lw=1.0, label="Episode-level significance bracket"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.98),
    )
    fig.suptitle(
        "1-3 Reward comparison: Pong and SpaceInvaders by subject and episode",
        y=1.03,
        fontsize=15,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

DATA_FILE_RE = re.compile(
    r"sub(?P<subject>\d+)-ses(?P<session>\d+)-block(?P<block>\d+)-game(?P<game>\d+)\.npz$"
)


def build_data_file_index() -> Dict[Tuple[str, int], List[Path]]:
    index: Dict[Tuple[str, int], List[Path]] = {}
    for path in sorted(DATA_ROOT.glob("sub-*/ses-*/*.npz")):
        match = DATA_FILE_RE.match(path.name)
        if match is None:
            continue
        subject = f"sub_{int(match.group('subject'))}"
        source_game = int(match.group("game"))
        if DATASET == "mario":
            condition = MARIO_STAGE_BY_FILENAME.get(path.name)
            if condition is None or condition not in GAMES:
                continue
        else:
            condition = source_game
            if condition not in GAMES:
                continue
        index.setdefault((subject, int(condition)), []).append(path)
    return index


DATA_FILE_INDEX = build_data_file_index()

def load_subject_game(subject: str, game: int) -> List[Dict]:
    """Return episode dictionaries for one subject × Atari game/Mario stage.

    Each episode dict has:
        subject, game, day, block, ep_idx (within block), actions (1-D int array),
        rewards (1-D), times (1-D seconds), n_steps, total_reward, noop_prop
    """
    files = DATA_FILE_INDEX.get((subject, int(game)), [])
    episodes = []
    for fpath in files:
        match = DATA_FILE_RE.match(Path(fpath).name)
        if match is None:
            continue
        day = int(match.group("session"))
        block = int(match.group("block"))
        source_game = int(match.group("game"))

        data = np.load(fpath, allow_pickle=True)
        action_oh = np.asarray(data["action"], dtype=float)
        actions = np.argmax(action_oh, axis=1).astype(int)
        rewards = np.asarray(data["reward"], dtype=float).reshape(-1)
        is_terminal = np.asarray(data["is_terminal"], dtype=bool)
        times = np.asarray(data["time"], dtype=float).reshape(-1)
        n_file = min(len(actions), len(rewards), len(is_terminal), len(times))
        actions = actions[:n_file]
        rewards = rewards[:n_file]
        is_terminal = is_terminal[:n_file]
        times = times[:n_file]
        if n_file > 1 and (not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0)):
            raise RuntimeError(f"Non-finite or non-increasing timestamps in {fpath}")

        # Split into episodes at is_terminal boundaries
        ep_starts = [0]
        term_pos = np.where(is_terminal)[0].tolist()
        for tp in term_pos:
            if tp + 1 < len(actions):
                ep_starts.append(tp + 1)
        ep_ends = term_pos + [len(actions) - 1]

        for local_ep_idx, (s, e) in enumerate(zip(ep_starts, ep_ends)):
            ep_actions = actions[s : e + 1]
            ep_rewards = rewards[s : e + 1]
            ep_times = times[s : e + 1]
            ep_terminal = is_terminal[s : e + 1]
            n = len(ep_actions)
            if n < 5:
                continue
            episodes.append(
                {
                    "subject": subject,
                    "game": game,
                    "game_name": GAMES[game],
                    "day": day,
                    "block": block,
                    "ep_idx": int(source_game * 1000 + local_ep_idx),
                    "source_game_index": source_game,
                    "source_file": str(fpath),
                    "actions": ep_actions,
                    "rewards": ep_rewards,
                    "times": ep_times,
                    "terminated": bool(ep_terminal[-1]),
                    "n_steps": n,
                    "total_reward": int(ep_rewards.sum()),
                    "noop_prop": float((ep_actions == NOOP_ACTION).mean()),
                }
            )
    return episodes


def decode_action_ids(action_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Decode action ids from either integer ids or one-hot rows."""
    arr = np.asarray(action_data)
    if arr.ndim == 1:
        action_ids = arr.astype(int)
        valid = action_ids >= 0
        return action_ids, valid

    flat = arr.reshape(arr.shape[0], -1)
    action_ids = np.argmax(flat, axis=1).astype(int)
    valid = flat.sum(axis=1) > 0
    return action_ids, valid


def compute_action_distribution(actions: np.ndarray) -> np.ndarray:
    action_ids = np.asarray(actions, dtype=int).reshape(-1)
    action_ids = action_ids[(action_ids >= 0) & (action_ids < N_ACTIONS)]
    if len(action_ids) == 0:
        return np.full(N_ACTIONS, np.nan, dtype=float)
    counts = np.bincount(action_ids, minlength=N_ACTIONS).astype(float)
    return counts / max(counts.sum(), 1.0)


def load_thinker_game(game: int) -> List[Dict]:
    """Load baseline thinker episodes from filtered per-episode .npy files for one game."""
    game_dir = THINKER_ROOT / f"game_{game}"
    files = sorted(game_dir.glob("*.npy"))
    if not files:
        print(
            f"  warning: no optional Thinker episode files for game_{game} under {game_dir}; "
            "the human behavioral analyses will continue"
        )
        return []

    episodes: List[Dict] = []
    for ep_idx, fpath in enumerate(files):
        data = np.load(fpath, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.shape == ():
            data = data.item()

        status = np.asarray(data["status"]).reshape(-1)
        tree_reps = data["tree_reps"]
        root_action, root_valid = decode_action_ids(np.asarray(tree_reps["root_action"]))
        t = min(len(status), len(root_action), len(root_valid))
        status = status[:t]
        root_action = root_action[:t]
        root_valid = root_valid[:t]

        real_mask = (status == 0) & root_valid
        real_actions = root_action[real_mask]
        if len(real_actions) == 0:
            continue

        episodes.append(
            {
                "subject": "thinker",
                "game": game,
                "game_name": GAMES[game],
                "day": 0,
                "block": 0,
                "ep_idx": ep_idx,
                "actions": real_actions,
                "rewards": np.array([], dtype=float),
                "n_steps": int(len(real_actions)),
                "total_reward": np.nan,
                "noop_prop": float((real_actions == NOOP_ACTION).mean()),
                "source": fpath.name,
            }
        )
    return episodes


def summarize_episode_mean_action_distributions(
    human_episodes: Sequence[Dict],
    thinker_episodes: Sequence[Dict],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    entity_order = [f"S{i}" for i in range(1, 7)] + ["Thinker"]

    for game, game_name in GAMES.items():
        for entity in entity_order:
            if entity == "Thinker":
                selected = [ep for ep in thinker_episodes if ep["game"] == game]
                entity_type = "thinker"
            else:
                subject = f"sub_{int(entity[1:])}"
                selected = [
                    ep for ep in human_episodes
                    if ep["game"] == game and ep["subject"] == subject
                ]
                entity_type = "human"

            episode_dists = []
            for ep in selected:
                dist = compute_action_distribution(ep["actions"])
                if np.all(np.isnan(dist)):
                    continue
                episode_dists.append(dist)

            if not episode_dists:
                continue

            dist_arr = np.vstack(episode_dists)
            mean_dist = np.nanmean(dist_arr, axis=0)
            if len(dist_arr) > 1:
                sem_dist = np.nanstd(dist_arr, axis=0, ddof=1) / np.sqrt(len(dist_arr))
            else:
                sem_dist = np.zeros(N_ACTIONS, dtype=float)

            for action_id in range(N_ACTIONS):
                rows.append(
                    {
                        "game": game,
                        "game_name": game_name,
                        "entity": entity,
                        "entity_type": entity_type,
                        "action_id": action_id,
                        "mean_prop": float(mean_dist[action_id]),
                        "sem_prop": float(sem_dist[action_id]),
                        "n_episodes": int(len(dist_arr)),
                    }
                )

    return pd.DataFrame(rows)


def plot_action_distribution_subjects_vs_thinker(
    summary: pd.DataFrame,
    out_path: Path,
) -> None:
    entity_order = [f"S{i}" for i in range(1, 7)] + ["Thinker"]
    entity_colors = {
        "S1": SUBJECT_COLORS[0],
        "S2": SUBJECT_COLORS[1],
        "S3": SUBJECT_COLORS[2],
        "S4": SUBJECT_COLORS[3],
        "S5": SUBJECT_COLORS[4],
        "S6": SUBJECT_COLORS[5],
        "Thinker": "#222222",
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.2), sharey=True)
    x = np.arange(N_ACTIONS)
    width = 0.11
    offsets = (np.arange(len(entity_order)) - (len(entity_order) - 1) / 2.0) * width

    for ax, game in zip(axes, GAMES.keys()):
        game_summary = summary[summary["game"] == game]
        ax.axvspan(-0.5, 0.5, color="#ffd166", alpha=0.18, lw=0)

        for idx, entity in enumerate(entity_order):
            entity_summary = (
                game_summary[game_summary["entity"] == entity]
                .sort_values("action_id")
                .reset_index(drop=True)
            )
            if len(entity_summary) != N_ACTIONS:
                continue

            ax.bar(
                x + offsets[idx],
                entity_summary["mean_prop"].to_numpy(dtype=float),
                width=width * 0.95,
                color=entity_colors[entity],
                edgecolor="white",
                linewidth=0.4,
                yerr=entity_summary["sem_prop"].to_numpy(dtype=float),
                capsize=2,
                label=entity,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(ACTION_LABELS, rotation=20)
        ax.set_xlabel("Action")
        ax.set_title(f"{GAME_TITLES[game]}")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 1)

    axes[0].set_ylabel("Episode-mean selection proportion")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=7, frameon=False, bbox_to_anchor=(0.5, 1.02))
    includes_thinker = bool(np.any(summary["entity"] == "Thinker"))
    comparison_label = "by subject and non-IL Thinker" if includes_thinker else "by human subject"
    fig.suptitle(f"1-1B: Action distribution {comparison_label}", y=1.08, fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def compute_bouts(actions: np.ndarray) -> List[int]:
    """Return uncensored NOOP-bout lengths followed by an active action."""
    bouts: List[int] = []
    actions = np.asarray(actions, dtype=int).reshape(-1)
    i = 0
    while i < len(actions):
        if actions[i] == NOOP_ACTION:
            j = i
            while j < len(actions) and actions[j] == NOOP_ACTION:
                j += 1
            # Only count bouts that are followed by an overt action
            if j < len(actions):
                bouts.append(j - i)
            i = j
        else:
            i += 1
    return bouts


def compute_lag1_autocorr(actions: np.ndarray) -> float:
    """Lag-1 autocorrelation of the is_noop binary sequence."""
    x = (actions == NOOP_ACTION).astype(float)
    if len(x) < 3:
        return np.nan
    x_centered = x - x.mean()
    denom = np.dot(x_centered, x_centered)
    if denom < EPS:
        return np.nan
    return float(np.dot(x_centered[:-1], x_centered[1:]) / denom)


def build_behavioral_bout_table(episodes: Sequence[Dict]) -> pd.DataFrame:
    """Return maximal same-action runs with episode-end censoring and exit cause."""
    rows: List[Dict[str, object]] = []
    for ep in episodes:
        actions = np.asarray(ep["actions"], dtype=int).reshape(-1)
        n_steps = len(actions)
        start = 0
        bout_index = 0
        while start < n_steps:
            action = int(actions[start])
            end = start + 1
            while end < n_steps and int(actions[end]) == action:
                end += 1

            censored = end == n_steps
            next_action = -1 if censored else int(actions[end])
            if censored:
                event_type = "episode_end"
            elif action == NOOP_ACTION:
                event_type = "to_active"
            elif next_action == NOOP_ACTION:
                event_type = "to_noop"
            else:
                event_type = "to_other_active"

            rows.append(
                {
                    "subject": str(ep["subject"]),
                    "game": int(ep["game"]),
                    "game_name": str(ep["game_name"]),
                    "day": int(ep["day"]),
                    "block": int(ep["block"]),
                    "ep_idx": int(ep["ep_idx"]),
                    "bout_index": int(bout_index),
                    "bout_type": "NOOP" if action == NOOP_ACTION else "Active",
                    "action": action,
                    "action_label": ACTION_LABELS[action] if 0 <= action < N_ACTIONS else str(action),
                    "start_step": int(start),
                    "end_step": int(end - 1),
                    "length": int(end - start),
                    "censor_limit": int(n_steps - start),
                    "censored": bool(censored),
                    "event_type": event_type,
                    "next_action": int(next_action),
                }
            )
            bout_index += 1
            start = end
    return pd.DataFrame(rows)


def kaplan_meier_survival_arrays(
    lengths: np.ndarray,
    events: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """Kaplan-Meier estimate for integer durations at requested times."""
    lengths = np.asarray(lengths, dtype=int)
    events = np.asarray(events, dtype=bool)
    times = np.asarray(times, dtype=int)
    if len(lengths) == 0 or len(times) == 0:
        return np.array([], dtype=float)
    max_time = max(int(np.max(lengths)), int(np.max(times)))
    length_counts = np.bincount(lengths, minlength=max_time + 1)
    event_counts = np.bincount(lengths[events], minlength=max_time + 1)
    at_risk = np.cumsum(length_counts[::-1])[::-1]
    hazards = np.zeros(max_time + 1, dtype=float)
    valid = at_risk > 0
    hazards[valid] = event_counts[valid] / at_risk[valid]
    step_survival = np.cumprod(1.0 - hazards[1:])
    clipped_times = np.clip(times, 1, max_time)
    return step_survival[clipped_times - 1]


def _discrete_weibull_log_terms(
    lengths: np.ndarray,
    q: float,
    beta: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Log event mass and log right-censor survival for S(t)=q^(t^beta)."""
    lengths = np.asarray(lengths, dtype=float)
    q = float(np.clip(q, 1e-8, 1.0 - 1e-8))
    beta = float(np.clip(beta, 0.05, 10.0))
    log_q = np.log(q)
    log_s_previous = np.power(np.maximum(lengths - 1.0, 0.0), beta) * log_q
    log_s_current = np.power(lengths, beta) * log_q
    log_ratio = np.minimum(log_s_current - log_s_previous, -1e-14)
    log_event = log_s_previous + np.log(-np.expm1(log_ratio))
    return log_event, log_s_current


def discrete_weibull_log_likelihood(
    lengths: np.ndarray,
    events: np.ndarray,
    q: float,
    beta: float,
) -> float:
    unique_lengths, event_counts, censor_counts = compress_duration_counts(lengths, events)
    return discrete_weibull_count_log_likelihood(
        unique_lengths, event_counts, censor_counts, q, beta
    )


def compress_duration_counts(
    lengths: np.ndarray,
    events: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lengths = np.asarray(lengths, dtype=int)
    events = np.asarray(events, dtype=bool)
    unique_lengths, inverse = np.unique(lengths, return_inverse=True)
    event_counts = np.bincount(inverse[events], minlength=len(unique_lengths)).astype(float)
    censor_counts = np.bincount(inverse[~events], minlength=len(unique_lengths)).astype(float)
    return unique_lengths, event_counts, censor_counts


def discrete_weibull_count_log_likelihood(
    unique_lengths: np.ndarray,
    event_counts: np.ndarray,
    censor_counts: np.ndarray,
    q: float,
    beta: float,
) -> float:
    log_event, log_censor = _discrete_weibull_log_terms(unique_lengths, q, beta)
    return float(np.sum(event_counts * log_event + censor_counts * log_censor))


def fit_constant_geometric(lengths: np.ndarray, events: np.ndarray) -> Dict[str, float]:
    lengths = np.asarray(lengths, dtype=int)
    events = np.asarray(events, dtype=bool)
    hazard = float(np.sum(events) / max(np.sum(lengths), 1))
    hazard = float(np.clip(hazard, 1e-8, 1.0 - 1e-8))
    q = 1.0 - hazard
    return {
        "q": q,
        "beta": 1.0,
        "hazard": hazard,
        "log_likelihood": discrete_weibull_log_likelihood(lengths, events, q, 1.0),
    }


def _logit(value: float) -> float:
    value = float(np.clip(value, 1e-8, 1.0 - 1e-8))
    return float(np.log(value / (1.0 - value)))


def fit_discrete_weibull(
    lengths: np.ndarray,
    events: np.ndarray,
    n_starts: int = 5,
) -> Dict[str, float]:
    lengths = np.asarray(lengths, dtype=int)
    events = np.asarray(events, dtype=bool)
    null_fit = fit_constant_geometric(lengths, events)
    unique_lengths, event_counts, censor_counts = compress_duration_counts(lengths, events)

    def objective(theta: np.ndarray) -> float:
        q = float(expit(theta[0]))
        beta = float(np.exp(theta[1]))
        return -discrete_weibull_count_log_likelihood(
            unique_lengths, event_counts, censor_counts, q, beta
        )

    beta_starts = np.array([0.45, 0.7, 1.0, 1.4, 2.0, 3.0], dtype=float)[:max(n_starts, 1)]
    best = None
    for beta_start in beta_starts:
        result = optimize.minimize(
            objective,
            x0=np.array([_logit(null_fit["q"]), np.log(beta_start)]),
            method="L-BFGS-B",
            bounds=[(-12.0, 12.0), (np.log(0.1), np.log(5.0))],
        )
        if np.isfinite(result.fun) and (best is None or result.fun < best.fun):
            best = result

    if best is None:
        return {**null_fit, "success": False}
    q = float(expit(best.x[0]))
    beta = float(np.exp(best.x[1]))
    return {
        "q": q,
        "beta": beta,
        "hazard": 1.0 - q,
        "log_likelihood": float(-best.fun),
        "success": bool(best.success),
    }


def _simulate_geometric_with_censoring(
    q: float,
    censor_limits: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    true_lengths = rng.geometric(1.0 - q, size=len(censor_limits))
    limits = np.maximum(np.asarray(censor_limits, dtype=int), 1)
    events = true_lengths <= limits
    observed = np.minimum(true_lengths, limits)
    return observed.astype(int), events.astype(bool)


def _fixed_log_hazard_edges(max_length: int) -> np.ndarray:
    """Pre-specified integer-duration bins; the final bin always includes the tail."""
    candidates = np.array(
        [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 145, 233, 377, 610, 987, 1597, 2584],
        dtype=int,
    )
    edges = candidates[candidates <= max_length]
    edges = np.unique(np.r_[1, edges, max_length + 1])
    return edges


def fit_piecewise_hazard(
    lengths: np.ndarray,
    events: np.ndarray,
    edges: np.ndarray | None = None,
) -> Dict[str, object]:
    """Fit a flexible piecewise-constant discrete exit hazard."""
    lengths = np.asarray(lengths, dtype=int)
    events = np.asarray(events, dtype=bool)
    max_length = int(np.max(lengths))
    if edges is None:
        edges = _fixed_log_hazard_edges(max_length)
    edges = np.asarray(edges, dtype=int)

    length_counts = np.bincount(lengths, minlength=max_length + 1)
    event_counts = np.bincount(lengths[events], minlength=max_length + 1)
    at_risk = np.cumsum(length_counts[::-1])[::-1]
    bin_hazards: List[float] = []
    bin_exposures: List[int] = []
    bin_events: List[int] = []
    log_likelihood = 0.0
    hazard_by_time = np.full(max_length + 1, np.nan, dtype=float)

    for left, right in zip(edges[:-1], edges[1:]):
        right_clipped = min(int(right), max_length + 1)
        if int(left) >= right_clipped:
            continue
        exposure = int(np.sum(at_risk[int(left) : right_clipped]))
        n_events = int(np.sum(event_counts[int(left) : right_clipped]))
        if exposure <= 0:
            continue
        hazard = float(np.clip(n_events / exposure, 1e-10, 1.0 - 1e-10))
        log_likelihood += n_events * np.log(hazard)
        log_likelihood += (exposure - n_events) * np.log1p(-hazard)
        bin_hazards.append(hazard)
        bin_exposures.append(exposure)
        bin_events.append(n_events)
        hazard_by_time[int(left) : right_clipped] = hazard

    return {
        "edges": edges,
        "hazards": np.asarray(bin_hazards, dtype=float),
        "exposures": np.asarray(bin_exposures, dtype=int),
        "events": np.asarray(bin_events, dtype=int),
        "hazard_by_time": hazard_by_time,
        "log_likelihood": float(log_likelihood),
        "n_bins": int(len(bin_hazards)),
    }


def bootstrap_duration_hazard_test(
    lengths: np.ndarray,
    events: np.ndarray,
    censor_limits: np.ndarray,
    times: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    null_fit = fit_constant_geometric(lengths, events)
    smooth_fit = fit_discrete_weibull(lengths, events)
    hazard_edges = _fixed_log_hazard_edges(int(np.max(lengths)))
    alternative_fit = fit_piecewise_hazard(lengths, events, hazard_edges)
    observed_lr = max(
        2.0 * (alternative_fit["log_likelihood"] - null_fit["log_likelihood"]),
        0.0,
    )

    rng = np.random.default_rng(seed)
    bootstrap_lr = np.full(n_bootstrap, np.nan, dtype=float)
    bootstrap_survival = np.full((n_bootstrap, len(times)), np.nan, dtype=float)
    for bootstrap_index in range(n_bootstrap):
        sim_lengths, sim_events = _simulate_geometric_with_censoring(
            null_fit["q"], censor_limits, rng
        )
        sim_null = fit_constant_geometric(sim_lengths, sim_events)
        sim_alt = fit_piecewise_hazard(sim_lengths, sim_events, hazard_edges)
        bootstrap_lr[bootstrap_index] = max(
            2.0 * (sim_alt["log_likelihood"] - sim_null["log_likelihood"]),
            0.0,
        )
        bootstrap_survival[bootstrap_index] = kaplan_meier_survival_arrays(
            sim_lengths, sim_events, times
        )

    valid_lr = bootstrap_lr[np.isfinite(bootstrap_lr)]
    p_bootstrap = float(
        (1 + np.sum(valid_lr >= observed_lr)) / (1 + len(valid_lr))
    )
    degrees_freedom = max(int(alternative_fit["n_bins"]) - 1, 1)
    p_chi_square = float(stats.chi2.sf(observed_lr, df=degrees_freedom))
    envelope_low = np.nanpercentile(bootstrap_survival, 2.5, axis=0)
    envelope_high = np.nanpercentile(bootstrap_survival, 97.5, axis=0)
    summary = {
        "n_bouts": int(len(lengths)),
        "n_events": int(np.sum(events)),
        "n_censored": int(np.sum(~events)),
        "constant_hazard": float(null_fit["hazard"]),
        "geometric_q": float(null_fit["q"]),
        "duration_q": float(smooth_fit["q"]),
        "duration_beta": float(smooth_fit["beta"]),
        "log_likelihood_constant": float(null_fit["log_likelihood"]),
        "log_likelihood_duration": float(alternative_fit["log_likelihood"]),
        "n_piecewise_hazard_bins": int(alternative_fit["n_bins"]),
        "lr_degrees_freedom": degrees_freedom,
        "lr_statistic": float(observed_lr),
        "p_chi_square": p_chi_square,
        "p_bootstrap": p_bootstrap,
        "n_bootstrap": int(len(valid_lr)),
    }
    return summary, envelope_low, envelope_high


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not np.any(finite):
        return adjusted
    finite_values = p[finite]
    order = np.argsort(finite_values)
    ranked = finite_values[order]
    m = len(ranked)
    q_sorted = ranked * m / np.arange(1, m + 1)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    q_finite = np.empty(m, dtype=float)
    q_finite[order] = q_sorted
    adjusted[finite] = q_finite
    return adjusted


def build_noop_day_level(episodes: Sequence[Dict]) -> pd.DataFrame:
    """Build exact step-weighted subject x condition x day NOOP rates."""
    rows: List[Dict[str, object]] = []
    for ep in episodes:
        actions = np.asarray(ep["actions"], dtype=int).reshape(-1)
        if len(actions) == 0:
            continue
        rows.append(
            {
                "subject": str(ep["subject"]),
                "game": int(ep["game"]),
                "game_name": str(ep["game_name"]),
                "day": int(ep["day"]),
                "n_episodes": 1,
                "n_steps": int(len(actions)),
                "n_noop": int(np.sum(actions == NOOP_ACTION)),
            }
        )

    columns = [
        "subject", "game", "game_name", "day", "n_episodes", "n_steps",
        "n_noop", "noop_rate", "is_common_day",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)

    day_level = (
        pd.DataFrame(rows)
        .groupby(["subject", "game", "game_name", "day"], as_index=False)
        .agg(
            n_episodes=("n_episodes", "sum"),
            n_steps=("n_steps", "sum"),
            n_noop=("n_noop", "sum"),
        )
        .sort_values(["game", "subject", "day"])
        .reset_index(drop=True)
    )
    if day_level.duplicated(["subject", "game", "day"]).any():
        raise RuntimeError("Day-level NOOP table is not unique by subject x condition x day")
    if np.any(day_level["n_steps"].to_numpy(dtype=int) <= 0):
        raise RuntimeError("Day-level NOOP table contains a non-positive step count")
    day_level["noop_rate"] = day_level["n_noop"] / day_level["n_steps"]
    day_level["is_common_day"] = False

    for game in GAMES:
        condition = day_level[day_level["game"] == game]
        day_sets = [
            set(condition.loc[condition["subject"] == subject, "day"].astype(int))
            for subject in SUBJECTS
        ]
        common_days = set.intersection(*day_sets) if day_sets else set()
        mask = (day_level["game"] == game) & day_level["day"].isin(common_days)
        day_level.loc[mask, "is_common_day"] = True

    return day_level[columns]


def _two_way_random_icc(matrix: np.ndarray) -> Dict[str, float]:
    """Two-way ICCs with subjects as targets and days as random repeats.

    Returns absolute-agreement ICC(A,1), consistency ICC(C,1), and the
    absolute-agreement average-day ICC(A,k). Negative estimates are retained.
    """
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("ICC matrix must be two dimensional")
    n_subjects, n_days = values.shape
    empty = {
        "icc_a1": np.nan,
        "icc_c1": np.nan,
        "icc_ak": np.nan,
        "ms_subject": np.nan,
        "ms_day": np.nan,
        "ms_error": np.nan,
    }
    if n_subjects < 2 or n_days < 2 or not np.all(np.isfinite(values)):
        return empty

    grand_mean = float(np.mean(values))
    subject_means = np.mean(values, axis=1)
    day_means = np.mean(values, axis=0)
    ss_subject = float(n_days * np.sum((subject_means - grand_mean) ** 2))
    ss_day = float(n_subjects * np.sum((day_means - grand_mean) ** 2))
    residual = values - subject_means[:, None] - day_means[None, :] + grand_mean
    ss_error = float(np.sum(residual**2))
    ms_subject = ss_subject / (n_subjects - 1)
    ms_day = ss_day / (n_days - 1)
    ms_error = ss_error / ((n_subjects - 1) * (n_days - 1))

    def ratio(numerator: float, denominator: float) -> float:
        if not np.isfinite(denominator) or abs(denominator) <= EPS:
            return np.nan
        return float(numerator / denominator)

    numerator = ms_subject - ms_error
    icc_a1 = ratio(
        numerator,
        ms_subject
        + (n_days - 1) * ms_error
        + n_days * (ms_day - ms_error) / n_subjects,
    )
    icc_c1 = ratio(numerator, ms_subject + (n_days - 1) * ms_error)
    icc_ak = ratio(numerator, ms_subject + (ms_day - ms_error) / n_subjects)
    return {
        "icc_a1": icc_a1,
        "icc_c1": icc_c1,
        "icc_ak": icc_ak,
        "ms_subject": float(ms_subject),
        "ms_day": float(ms_day),
        "ms_error": float(ms_error),
    }


def _bootstrap_icc_by_subject(
    matrix: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> Dict[str, float]:
    """Deterministic target bootstrap for the balanced ICC matrix."""
    values = np.asarray(matrix, dtype=float)
    n_subjects = values.shape[0]
    rng = np.random.default_rng(seed)
    draws = {name: [] for name in ("icc_a1", "icc_c1", "icc_ak")}
    for _ in range(int(n_bootstrap)):
        sampled = values[rng.integers(0, n_subjects, size=n_subjects), :]
        estimates = _two_way_random_icc(sampled)
        for name in draws:
            value = float(estimates[name])
            if np.isfinite(value):
                draws[name].append(value)

    result: Dict[str, float] = {}
    for name, raw_values in draws.items():
        finite = np.asarray(raw_values, dtype=float)
        result[f"{name}_ci95_low"] = (
            float(np.percentile(finite, 2.5)) if len(finite) else np.nan
        )
        result[f"{name}_ci95_high"] = (
            float(np.percentile(finite, 97.5)) if len(finite) else np.nan
        )
        result[f"{name}_bootstrap_valid"] = int(len(finite))
    return result


def summarize_noop_day_repeatability(
    day_level: pd.DataFrame,
    n_bootstrap: int = ICC_BOOTSTRAPS,
) -> pd.DataFrame:
    """Condition-specific balanced common-day ICCs across participants."""
    rows: List[Dict[str, object]] = []
    expected_subjects = list(SUBJECTS)
    if day_level.duplicated(["subject", "game", "day"]).any():
        raise RuntimeError("ICC input contains duplicate subject x condition x day rows")

    for game, game_name in GAMES.items():
        condition = day_level[day_level["game"] == game].copy()
        union_days = sorted(condition["day"].dropna().astype(int).unique().tolist())
        observed_subjects = sorted(condition["subject"].unique().tolist())
        day_sets = [
            set(condition.loc[condition["subject"] == subject, "day"].astype(int))
            for subject in expected_subjects
        ]
        common_days = sorted(set.intersection(*day_sets)) if day_sets else []
        n_possible_cells = len(expected_subjects) * len(union_days)
        n_observed_cells = int(len(condition))
        n_common_cells_used = int(len(expected_subjects) * len(common_days))
        coverage = (
            float(n_observed_cells / n_possible_cells) if n_possible_cells > 0 else np.nan
        )
        common_cell_retention = (
            float(n_common_cells_used / n_observed_cells) if n_observed_cells > 0 else np.nan
        )
        common_day_fraction = (
            float(len(common_days) / len(union_days)) if union_days else np.nan
        )

        estimates = _two_way_random_icc(np.empty((0, 0)))
        bootstrap = {
            f"{name}_{suffix}": np.nan
            for name in ("icc_a1", "icc_c1", "icc_ak")
            for suffix in ("ci95_low", "ci95_high", "bootstrap_valid")
        }
        if len(common_days) >= 2 and len(expected_subjects) >= 2:
            matrix_frame = (
                condition[condition["day"].isin(common_days)]
                .pivot(index="subject", columns="day", values="noop_rate")
                .reindex(index=expected_subjects, columns=common_days)
            )
            if matrix_frame.shape != (len(expected_subjects), len(common_days)):
                raise RuntimeError("Balanced common-day ICC matrix has an unexpected shape")
            if matrix_frame.isna().any().any():
                raise RuntimeError("Balanced common-day ICC matrix contains missing cells")
            matrix = matrix_frame.to_numpy(dtype=float)
            estimates = _two_way_random_icc(matrix)
            bootstrap = _bootstrap_icc_by_subject(
                matrix,
                n_bootstrap=n_bootstrap,
                seed=ICC_RANDOM_SEED + 1009 * int(game),
            )

        rows.append(
            {
                "game": int(game),
                "game_name": str(game_name),
                "n_subjects": int(len(expected_subjects)),
                "n_subjects_observed": int(len(observed_subjects)),
                "n_days": int(len(common_days)),
                "n_days_union": int(len(union_days)),
                "days_used": "|".join(str(day) for day in common_days),
                "n_subject_day_cells_observed": n_observed_cells,
                "n_subject_day_cells_possible": int(n_possible_cells),
                "n_subject_day_cells_used": n_common_cells_used,
                "subject_day_coverage": coverage,
                "common_day_cell_retention": common_cell_retention,
                "common_day_fraction": common_day_fraction,
                "balanced_common_days": bool(
                    len(common_days) >= 2
                    and len(observed_subjects) == len(expected_subjects)
                ),
                "bootstrap_requested": int(n_bootstrap),
                **estimates,
                **bootstrap,
            }
        )
    return pd.DataFrame(rows).sort_values("game").reset_index(drop=True)


def _friedman_subject_concordance(matrix: np.ndarray) -> Dict[str, float]:
    """Friedman Q and Kendall's W for day blocks x participant columns."""
    values = np.asarray(matrix, dtype=float)
    empty = {"friedman_q": np.nan, "kendall_w": np.nan, "tie_correction": np.nan}
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        return empty
    n_days, n_subjects = values.shape
    if n_days < 2 or n_subjects < 2:
        return empty

    ranks = np.vstack([stats.rankdata(row, method="average") for row in values])
    rank_sums = np.sum(ranks, axis=0)
    q_uncorrected = (
        12.0 * np.sum(rank_sums**2) / (n_days * n_subjects * (n_subjects + 1.0))
        - 3.0 * n_days * (n_subjects + 1.0)
    )
    tie_sum = 0.0
    for row in values:
        counts = np.unique(row, return_counts=True)[1]
        tie_sum += float(np.sum(counts**3 - counts))
    tie_correction = 1.0 - tie_sum / (
        n_days * n_subjects * (n_subjects**2 - 1.0)
    )
    if not np.isfinite(tie_correction) or tie_correction <= EPS:
        return empty
    friedman_q = max(float(q_uncorrected / tie_correction), 0.0)
    kendall_w = friedman_q / (n_days * (n_subjects - 1.0))
    return {
        "friedman_q": friedman_q,
        "kendall_w": float(kendall_w),
        "tie_correction": float(tie_correction),
    }


def _subject_label_permutation_pvalue(
    matrix: np.ndarray,
    n_permutations: int,
    seed: int,
) -> Tuple[float, float]:
    """Permute participant labels independently within each common-day block."""
    values = np.asarray(matrix, dtype=float)
    observed = _friedman_subject_concordance(values)
    observed_q = float(observed["friedman_q"])
    tie_correction = float(observed["tie_correction"])
    if not np.isfinite(observed_q) or not np.isfinite(tie_correction):
        return np.nan, np.nan
    n_permutations = int(n_permutations)
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")

    n_days, n_subjects = values.shape
    ranks = np.vstack([stats.rankdata(row, method="average") for row in values])
    rng = np.random.default_rng(int(seed))
    n_at_least_observed = 0
    batch_size = 10_000
    completed = 0
    while completed < n_permutations:
        batch = min(batch_size, n_permutations - completed)
        rank_sums = np.zeros((batch, n_subjects), dtype=float)
        for day_index in range(n_days):
            permutations = np.argsort(
                rng.random((batch, n_subjects)), axis=1
            )
            rank_sums += ranks[day_index][permutations]
        null_q = (
            12.0 * np.sum(rank_sums**2, axis=1)
            / (n_days * n_subjects * (n_subjects + 1.0))
            - 3.0 * n_days * (n_subjects + 1.0)
        ) / tie_correction
        n_at_least_observed += int(np.sum(null_q >= observed_q - 1e-12))
        completed += batch

    p_value = (n_at_least_observed + 1.0) / (n_permutations + 1.0)
    mc_standard_error = float(
        np.sqrt(p_value * (1.0 - p_value) / (n_permutations + 1.0))
    )
    return float(p_value), mc_standard_error


def summarize_noop_subject_differences(
    day_level: pd.DataFrame,
    n_permutations: int = SUBJECT_DIFFERENCE_PERMUTATIONS,
) -> pd.DataFrame:
    """Test whether participant rank differences recur across common days.

    Day is the blocking unit. Participant identities are independently shuffled
    within each day, preserving every day's marginal NOOP-rate distribution.
    """
    columns = [
        "game", "game_name", "n_subjects", "n_common_days", "days_used",
        "friedman_q", "kendall_w", "tie_correction", "p_permutation",
        "mc_standard_error", "n_permutations", "random_seed", "q_bh", "stars",
        "valid", "invalid_reason", "test_name", "permutation_unit",
        "multiple_testing_family",
    ]
    rows: List[Dict[str, object]] = []
    expected_subjects = list(SUBJECTS)
    for game, game_name in GAMES.items():
        condition = day_level[
            (day_level["game"] == game) & day_level["is_common_day"].astype(bool)
        ].copy()
        common_days = sorted(condition["day"].dropna().astype(int).unique().tolist())
        random_seed = SUBJECT_DIFFERENCE_RANDOM_SEED + 1009 * int(game)
        base: Dict[str, object] = {
            "game": int(game),
            "game_name": str(game_name),
            "n_subjects": int(len(expected_subjects)),
            "n_common_days": int(len(common_days)),
            "days_used": "|".join(str(day) for day in common_days),
            "friedman_q": np.nan,
            "kendall_w": np.nan,
            "tie_correction": np.nan,
            "p_permutation": np.nan,
            "mc_standard_error": np.nan,
            "n_permutations": int(n_permutations),
            "random_seed": int(random_seed),
            "q_bh": np.nan,
            "stars": "n/a",
            "valid": False,
            "invalid_reason": "",
            "test_name": "Friedman/Kendall concordance",
            "permutation_unit": "participant labels independently within common day",
            "multiple_testing_family": f"{DATASET}_figure_1-2_conditions_bh_fdr",
        }
        if len(common_days) < 3:
            base["invalid_reason"] = "fewer_than_3_common_days"
            rows.append(base)
            continue
        observed_subjects = set(condition["subject"].astype(str).unique())
        if observed_subjects != set(expected_subjects):
            base["invalid_reason"] = "incomplete_subject_set"
            rows.append(base)
            continue
        matrix_frame = (
            condition.pivot(index="day", columns="subject", values="noop_rate")
            .reindex(index=common_days, columns=expected_subjects)
        )
        if matrix_frame.shape != (len(common_days), len(expected_subjects)):
            base["invalid_reason"] = "unexpected_balanced_matrix_shape"
            rows.append(base)
            continue
        if matrix_frame.isna().any().any():
            base["invalid_reason"] = "missing_common_day_cell"
            rows.append(base)
            continue
        matrix = matrix_frame.to_numpy(dtype=float)
        estimates = _friedman_subject_concordance(matrix)
        if not np.isfinite(estimates["friedman_q"]):
            base["invalid_reason"] = "no_within_day_subject_variation"
            rows.append(base)
            continue
        p_value, mc_standard_error = _subject_label_permutation_pvalue(
            matrix,
            n_permutations=int(n_permutations),
            seed=int(random_seed),
        )
        base.update(estimates)
        base["p_permutation"] = p_value
        base["mc_standard_error"] = mc_standard_error
        base["valid"] = bool(np.isfinite(p_value))
        base["invalid_reason"] = "" if np.isfinite(p_value) else "permutation_failed"
        rows.append(base)

    result = pd.DataFrame(rows, columns=columns)
    if result.empty:
        return result
    result = result.sort_values("game").reset_index(drop=True)
    result["q_bh"] = benjamini_hochberg(
        result["p_permutation"].to_numpy(dtype=float)
    )
    result["stars"] = result["q_bh"].map(stars_for_pvalue)
    return result[columns]


def save_day_repeatability_figure(
    day_level: pd.DataFrame,
    repeatability: pd.DataFrame,
    subject_inference: pd.DataFrame,
    out_path: Path,
) -> None:
    """Render the shared Figure 1-2 contract for Atari or Mario."""
    summary = repeatability.set_index("game") if not repeatability.empty else pd.DataFrame()
    inference_by_game = (
        subject_inference.set_index("game")
        if not subject_inference.empty else pd.DataFrame()
    )
    common_all = day_level[day_level["is_common_day"].astype(bool)]
    plotted_days = sorted(common_all["day"].dropna().astype(int).unique().tolist())
    day_min = min(plotted_days) if plotted_days else 0
    day_max = max(plotted_days) if plotted_days else 1
    if day_min == day_max:
        day_min -= 0.5
        day_max += 0.5
    day_cmap = mpl_colors.LinearSegmentedColormap.from_list(
        "figure_1_2_days", ("#dceaf6", "#1f5a94")
    )
    day_norm = mpl_colors.Normalize(vmin=day_min, vmax=day_max)
    subject_positions = np.arange(len(SUBJECTS), dtype=float)
    subject_labels = [f"S{subject_sort_key(subject)}" for subject in SUBJECTS]

    def render_condition_boxplots(ax: plt.Axes, game: int) -> List[int]:
        """Plot each participant's common-day distribution and every day-level value.

        The horizontal line inside each box is the median. Point offsets are a
        deterministic function of sorted day order, so rerenders do not move the
        observations and the same day always occupies the same relative position.
        """
        condition = common_all[common_all["game"] == game].copy()
        days = sorted(condition["day"].dropna().astype(int).unique().tolist())
        if not days:
            ax.text(0.5, 0.5, "No common days", transform=ax.transAxes,
                    ha="center", va="center", color="#666666")
            return days

        values_by_subject = [
            (
                condition[condition["subject"] == subject]
                .set_index("day")["noop_rate"]
                .reindex(days)
                .dropna()
                .to_numpy(dtype=float)
            )
            for subject in SUBJECTS
        ]
        ax.boxplot(
            values_by_subject,
            positions=subject_positions,
            widths=0.56,
            patch_artist=True,
            showmeans=False,
            showfliers=False,
            manage_ticks=False,
            boxprops={
                "facecolor": "#e5edf4",
                "edgecolor": "#52697d",
                "linewidth": 0.9,
                "alpha": 0.80,
            },
            medianprops={"color": "#111111", "linewidth": 1.6},
            whiskerprops={"color": "#52697d", "linewidth": 0.85},
            capprops={"color": "#52697d", "linewidth": 0.85},
            zorder=2,
        )
        point_offsets = (
            np.linspace(-0.13, 0.13, len(days))
            if len(days) > 1 else np.zeros(1, dtype=float)
        )
        for day_index, day in enumerate(days):
            rates = (
                condition[condition["day"] == day]
                .set_index("subject")["noop_rate"]
                .reindex(SUBJECTS)
                .to_numpy(dtype=float)
            )
            ax.scatter(
                subject_positions + point_offsets[day_index],
                rates,
                s=25,
                color=day_cmap(day_norm(day)),
                edgecolor="white",
                linewidth=0.45,
                alpha=0.96,
                zorder=4,
            )
        return days

    def panel_title(game: int, game_name: str) -> str:
        star = "n/a"
        kendall_w = np.nan
        n_days = 0
        if game in inference_by_game.index:
            row = inference_by_game.loc[game]
            star = str(row["stars"])
            kendall_w = float(row["kendall_w"])
            n_days = int(row["n_common_days"])
        icc = float(summary.loc[game, "icc_a1"]) if game in summary.index else np.nan
        icc_text = f"{icc:.2f}" if np.isfinite(icc) else "n/a"
        w_text = f"{kendall_w:.2f}" if np.isfinite(kendall_w) else "n/a"
        return f"{game_name}  {star}\nICC(A,1)={icc_text}; W={w_text}; k={n_days} days"

    if DATASET == "atari":
        fig, axes = plt.subplots(
            1, len(GAMES), figsize=(6.2 * len(GAMES), 5.6), squeeze=False
        )
        for axis_index, (game, game_name) in enumerate(GAMES.items()):
            ax = axes[0, axis_index]
            render_condition_boxplots(ax, int(game))
            ax.set_ylim(0.0, 1.0)
            ax.set_xticks(subject_positions, subject_labels)
            ax.set_xlabel("Participant")
            ax.set_ylabel("Step-weighted NOOP rate")
            ax.set_title(panel_title(int(game), str(GAME_TITLES.get(game, game_name))),
                         fontsize=11)
            ax.grid(axis="y", alpha=0.22, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)
            if game in summary.index:
                row = summary.loc[game]
                ax.text(
                    0.03, 0.035,
                    f"ICC 95% CI [{float(row['icc_a1_ci95_low']):.2f}, "
                    f"{float(row['icc_a1_ci95_high']):.2f}]; "
                    f"cells kept={float(row['common_day_cell_retention']):.0%}",
                    transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5,
                    bbox=dict(facecolor="white", edgecolor="#cccccc", alpha=0.88, pad=4),
                )
        legend_handles = [
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=day_cmap(day_norm(day)),
                   markeredgecolor="white", markeredgewidth=0.45,
                   label=str(day), markersize=7)
            for day in plotted_days
        ]
        if legend_handles:
            fig.legend(
                handles=legend_handles, title="Day", loc="upper center",
                ncol=len(legend_handles), frameon=False, bbox_to_anchor=(0.5, 0.885),
            )
        fig.suptitle(
            "1-2: Participant differences in day-level NOOP prevalence\n"
            "Balanced common days; boxes summarize days and colored points are individual days",
            fontsize=14, y=0.995,
        )
        fig.text(
            0.5, 0.012,
            "Boxes span Q1-Q3; center lines are medians; whiskers extend to 1.5 x IQR; "
            "no mean tick is shown.\nPanel stars are omnibus within-day "
            "participant-label permutation tests "
            "(* q<.05, ** q<.01, *** q<.001; BH across games), not pairwise contrasts. "
            "ICC describes absolute-agreement repeatability.",
            ha="center", va="bottom", fontsize=8.5, color="#444444",
        )
        fig.tight_layout(rect=[0, 0.06, 1, 0.82])
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return

    games = list(GAMES)
    n_columns = min(4, max(len(games), 1))
    n_rows = int(np.ceil(len(games) / n_columns))
    fig, axes = plt.subplots(
        n_rows, n_columns,
        figsize=(3.7 * n_columns, 2.7 * n_rows + 1.8),
        squeeze=False,
    )
    visible_axes: List[plt.Axes] = []
    for panel_index, game in enumerate(games):
        ax = axes.flat[panel_index]
        visible_axes.append(ax)
        render_condition_boxplots(ax, int(game))
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(subject_positions, subject_labels, fontsize=7)
        ax.set_yticks((0.0, 0.5, 1.0))
        ax.tick_params(axis="y", labelsize=7)
        ax.set_xlabel("Participant", fontsize=8)
        if panel_index % n_columns == 0:
            ax.set_ylabel("NOOP rate", fontsize=8)
        ax.set_title(panel_title(int(game), str(GAMES[game])), fontsize=9, pad=4)
        ax.grid(axis="y", alpha=0.20, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    for panel_index in range(len(games), n_rows * n_columns):
        axes.flat[panel_index].set_axis_off()
    fig.suptitle(
        "1-2: Participant differences in day-level NOOP prevalence across Mario stages\n"
        "Stage is encoded by facet; boxes summarize common days and points are colored by day",
        fontsize=14, y=0.995,
    )
    mappable = ScalarMappable(norm=day_norm, cmap=day_cmap)
    mappable.set_array([])
    fig.subplots_adjust(top=0.91, bottom=0.15, hspace=0.70, wspace=0.25)
    colorbar_axis = fig.add_axes([0.20, 0.078, 0.60, 0.012])
    colorbar = fig.colorbar(
        mappable, cax=colorbar_axis, orientation="horizontal",
    )
    colorbar.set_label("Day", fontsize=9)
    if plotted_days:
        if len(plotted_days) <= 9:
            colorbar.set_ticks(plotted_days)
        else:
            colorbar.set_ticks(
                np.unique(np.rint(np.linspace(min(plotted_days), max(plotted_days), 7)).astype(int))
            )
    retention = repeatability["common_day_cell_retention"].to_numpy(dtype=float)
    retention = retention[np.isfinite(retention)]
    retention_text = (
        f"median cells kept={np.median(retention):.0%}"
        if len(retention) else "cells kept=n/a"
    )
    fig.text(
        0.5, 0.012,
        "Boxes span Q1-Q3; center lines are medians; whiskers extend to 1.5 x IQR; "
        "no mean tick is shown.\nPanel stars are omnibus within-day participant-label "
        "permutation tests with BH across stages; "
        "they do not identify participant pairs. "
        f"Only balanced common-day cells are plotted ({retention_text}). "
        "With n=5, stage-level effects and ICCs remain exploratory, not evidence of a population-stable trait.",
        ha="center", va="bottom", fontsize=8.5, color="#444444",
    )
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_day_repeatability_analysis(
    episodes: Sequence[Dict],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    day_level = build_noop_day_level(episodes)
    repeatability = summarize_noop_day_repeatability(day_level)
    subject_inference = summarize_noop_subject_differences(day_level)
    day_level.to_csv(OUT_DIR / "1-2_noop_day_level.csv", index=False)
    repeatability.to_csv(OUT_DIR / "1-2_noop_day_repeatability.csv", index=False)
    subject_inference.to_csv(
        OUT_DIR / "1-2_noop_subject_difference_inference.csv", index=False
    )
    save_day_repeatability_figure(
        day_level,
        repeatability,
        subject_inference,
        FIG_DIR / "fig_1-2_day_level_noop_repeatability.png",
    )
    return day_level, repeatability


def build_day_heldout_hazard_folds(bout_table: pd.DataFrame) -> pd.DataFrame:
    """Leave one day out when fitting and evaluating censored NOOP durations."""
    columns = [
        "subject", "game", "game_name", "heldout_day", "train_days", "n_train_days",
        "n_train_bouts", "n_train_events", "n_train_censored", "n_test_bouts",
        "n_test_events", "n_test_censored", "constant_q", "constant_beta",
        "duration_q", "duration_beta", "duration_fit_success",
        "training_log_likelihood_constant", "training_log_likelihood_duration",
        "heldout_log_likelihood_constant", "heldout_log_likelihood_duration",
        "delta_heldout_log_likelihood", "delta_heldout_log_likelihood_per_bout",
        "fold_valid", "invalid_reason",
    ]
    required = {"subject", "game", "game_name", "day", "length", "censored", "bout_type"}
    missing = required.difference(bout_table.columns)
    if missing:
        raise RuntimeError(f"Held-out hazard table is missing columns: {sorted(missing)}")
    noop = bout_table[bout_table["bout_type"] == "NOOP"].copy()
    if noop.empty:
        return pd.DataFrame(columns=columns)
    if np.any(noop["length"].to_numpy(dtype=int) <= 0):
        raise RuntimeError("NOOP bout table contains non-positive durations")

    rows: List[Dict[str, object]] = []
    for (subject, game, game_name), group in noop.groupby(
        ["subject", "game", "game_name"], sort=True
    ):
        all_days = sorted(group["day"].astype(int).unique().tolist())
        for heldout_day in all_days:
            train_days = [day for day in all_days if day != heldout_day]
            train = group[group["day"].isin(train_days)]
            test = group[group["day"] == heldout_day]
            train_lengths = train["length"].to_numpy(dtype=int)
            train_events = ~train["censored"].to_numpy(dtype=bool)
            test_lengths = test["length"].to_numpy(dtype=int)
            test_events = ~test["censored"].to_numpy(dtype=bool)
            base: Dict[str, object] = {
                "subject": str(subject),
                "game": int(game),
                "game_name": str(game_name),
                "heldout_day": int(heldout_day),
                "train_days": "|".join(str(day) for day in train_days),
                "n_train_days": int(len(train_days)),
                "n_train_bouts": int(len(train)),
                "n_train_events": int(np.sum(train_events)),
                "n_train_censored": int(np.sum(~train_events)),
                "n_test_bouts": int(len(test)),
                "n_test_events": int(np.sum(test_events)),
                "n_test_censored": int(np.sum(~test_events)),
                "constant_q": np.nan,
                "constant_beta": 1.0,
                "duration_q": np.nan,
                "duration_beta": np.nan,
                "duration_fit_success": False,
                "training_log_likelihood_constant": np.nan,
                "training_log_likelihood_duration": np.nan,
                "heldout_log_likelihood_constant": np.nan,
                "heldout_log_likelihood_duration": np.nan,
                "delta_heldout_log_likelihood": np.nan,
                "delta_heldout_log_likelihood_per_bout": np.nan,
                "fold_valid": False,
                "invalid_reason": "",
            }
            if heldout_day in train_days:
                raise RuntimeError("Held-out day leaked into its training-day list")
            if len(train_days) < 2:
                base["invalid_reason"] = "fewer_than_two_training_days"
                rows.append(base)
                continue
            if len(train_lengths) < 2 or len(test_lengths) < 1:
                base["invalid_reason"] = "insufficient_train_or_test_bouts"
                rows.append(base)
                continue
            try:
                constant_fit = fit_constant_geometric(train_lengths, train_events)
                duration_fit = fit_discrete_weibull(train_lengths, train_events)
            except (FloatingPointError, RuntimeError, ValueError) as error:
                base["invalid_reason"] = f"fit_error:{type(error).__name__}"
                rows.append(base)
                continue
            base["constant_q"] = float(constant_fit["q"])
            base["duration_q"] = float(duration_fit["q"])
            base["duration_beta"] = float(duration_fit["beta"])
            base["duration_fit_success"] = bool(duration_fit.get("success", False))
            base["training_log_likelihood_constant"] = float(
                constant_fit["log_likelihood"]
            )
            base["training_log_likelihood_duration"] = float(
                duration_fit["log_likelihood"]
            )
            if not base["duration_fit_success"]:
                base["invalid_reason"] = "duration_fit_not_converged"
                rows.append(base)
                continue
            parameter_values = np.asarray(
                [
                    base["constant_q"], base["duration_q"], base["duration_beta"],
                    base["training_log_likelihood_constant"],
                    base["training_log_likelihood_duration"],
                ],
                dtype=float,
            )
            if not np.all(np.isfinite(parameter_values)):
                base["invalid_reason"] = "nonfinite_fit_parameter"
                rows.append(base)
                continue
            ll_constant = discrete_weibull_log_likelihood(
                test_lengths, test_events, float(base["constant_q"]), 1.0
            )
            ll_duration = discrete_weibull_log_likelihood(
                test_lengths,
                test_events,
                float(base["duration_q"]),
                float(base["duration_beta"]),
            )
            if not np.all(np.isfinite([ll_constant, ll_duration])):
                base["invalid_reason"] = "nonfinite_heldout_log_likelihood"
                rows.append(base)
                continue
            delta = float(ll_duration - ll_constant)
            base["heldout_log_likelihood_constant"] = float(ll_constant)
            base["heldout_log_likelihood_duration"] = float(ll_duration)
            base["delta_heldout_log_likelihood"] = delta
            base["delta_heldout_log_likelihood_per_bout"] = delta / len(test_lengths)
            base["fold_valid"] = True
            rows.append(base)

    return pd.DataFrame(rows, columns=columns).sort_values(
        ["game", "subject", "heldout_day"]
    ).reset_index(drop=True)


def summarize_day_heldout_hazard_subjects(folds: pd.DataFrame) -> pd.DataFrame:
    """Pool held-out likelihood by bouts; retain an equal-day sensitivity."""
    columns = [
        "subject", "game", "game_name", "n_days_total", "n_days_valid",
        "n_days_invalid", "bout_weighted_delta_heldout_log_likelihood_per_bout",
        "day_equal_mean_delta_heldout_log_likelihood_per_bout",
        "sd_delta_across_days", "min_delta_across_days", "max_delta_across_days",
        "n_positive_heldout_days", "n_negative_heldout_days", "n_zero_heldout_days",
        "mean_n_train_bouts", "total_n_test_bouts", "min_n_test_bouts",
        "median_n_test_bouts", "max_n_test_bouts", "n_test_days_lt5",
        "sum_delta_heldout_log_likelihood",
    ]
    if folds.empty:
        return pd.DataFrame(columns=columns)
    rows: List[Dict[str, object]] = []
    for (subject, game, game_name), group in folds.groupby(
        ["subject", "game", "game_name"], sort=True
    ):
        valid = group[
            group["fold_valid"].astype(bool)
            & np.isfinite(group["delta_heldout_log_likelihood_per_bout"])
            & np.isfinite(group["delta_heldout_log_likelihood"])
            & (group["n_test_bouts"] > 0)
        ]
        values = valid["delta_heldout_log_likelihood_per_bout"].to_numpy(dtype=float)
        total_test_bouts = int(valid["n_test_bouts"].sum())
        total_delta = float(valid["delta_heldout_log_likelihood"].sum())
        rows.append(
            {
                "subject": str(subject),
                "game": int(game),
                "game_name": str(game_name),
                "n_days_total": int(len(group)),
                "n_days_valid": int(len(valid)),
                "n_days_invalid": int(len(group) - len(valid)),
                "bout_weighted_delta_heldout_log_likelihood_per_bout": (
                    total_delta / total_test_bouts if total_test_bouts > 0 else np.nan
                ),
                "day_equal_mean_delta_heldout_log_likelihood_per_bout": (
                    float(np.mean(values)) if len(values) else np.nan
                ),
                "sd_delta_across_days": (
                    float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
                ),
                "min_delta_across_days": float(np.min(values)) if len(values) else np.nan,
                "max_delta_across_days": float(np.max(values)) if len(values) else np.nan,
                "n_positive_heldout_days": int(np.sum(values > 0)),
                "n_negative_heldout_days": int(np.sum(values < 0)),
                "n_zero_heldout_days": int(np.sum(values == 0)),
                "mean_n_train_bouts": (
                    float(valid["n_train_bouts"].mean()) if len(valid) else np.nan
                ),
                "total_n_test_bouts": total_test_bouts,
                "min_n_test_bouts": int(valid["n_test_bouts"].min()) if len(valid) else 0,
                "median_n_test_bouts": (
                    float(valid["n_test_bouts"].median()) if len(valid) else np.nan
                ),
                "max_n_test_bouts": int(valid["n_test_bouts"].max()) if len(valid) else 0,
                "n_test_days_lt5": int(np.sum(valid["n_test_bouts"] < 5)),
                "sum_delta_heldout_log_likelihood": total_delta,
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["game", "subject"]
    ).reset_index(drop=True)


def summarize_day_heldout_hazard_inference(subject_summary: pd.DataFrame) -> pd.DataFrame:
    """Equal-subject condition inference for held-out-day predictive gains."""
    rows: List[Dict[str, object]] = []
    value_column = "bout_weighted_delta_heldout_log_likelihood_per_bout"
    for game, game_name in GAMES.items():
        group = subject_summary[subject_summary["game"] == game]
        values = group[value_column].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        n_subjects = len(values)
        mean = float(np.mean(values)) if n_subjects else np.nan
        sd = float(np.std(values, ddof=1)) if n_subjects > 1 else np.nan
        sem = float(sd / np.sqrt(n_subjects)) if n_subjects > 1 else np.nan
        if n_subjects > 1 and np.isfinite(sem) and sem > 0:
            t_statistic = float(mean / sem)
            p_value = float(2.0 * stats.t.sf(abs(t_statistic), df=n_subjects - 1))
            half_width = float(stats.t.ppf(0.975, df=n_subjects - 1) * sem)
        elif n_subjects > 0 and np.allclose(values, 0.0):
            t_statistic, p_value, half_width = 0.0, 1.0, 0.0
        else:
            t_statistic, p_value, half_width = np.nan, np.nan, np.nan
        rows.append(
            {
                "game": int(game),
                "game_name": str(game_name),
                "n_subjects": int(n_subjects),
                "mean_bout_weighted_delta_heldout_log_likelihood_per_bout": mean,
                "sd_subject_delta": sd,
                "sem_subject_delta": sem,
                "ci95_low": mean - half_width if np.isfinite(half_width) else np.nan,
                "ci95_high": mean + half_width if np.isfinite(half_width) else np.nan,
                "t_statistic": t_statistic,
                "p_value": p_value,
                "exact_sign_flip_p": exact_sign_flip_p(values),
                "n_positive": int(np.sum(values > 0)),
                "n_negative": int(np.sum(values < 0)),
                "n_zero": int(np.sum(values == 0)),
                "n_valid_heldout_days": int(group["n_days_valid"].sum()),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result.sort_values("game").reset_index(drop=True)
    result["q_fdr"] = benjamini_hochberg(result["p_value"].to_numpy(dtype=float))
    result["significance_fdr"] = result["q_fdr"].map(stars_for_pvalue)
    result["q_fdr_exact_sign_flip"] = benjamini_hochberg(
        result["exact_sign_flip_p"].to_numpy(dtype=float)
    )
    result["significance_exact_fdr"] = result["q_fdr_exact_sign_flip"].map(
        stars_for_pvalue
    )
    result["display_significance"] = result["significance_fdr"]
    result["estimand"] = "condition_specific_subject_bout_weighted_over_heldout_days"
    result["multiple_testing_family"] = "condition_rows_bh_fdr"
    return result


def summarize_day_heldout_hazard_common_effect(
    subject_summary: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Dataset-wide equal-condition mean, kept outside the condition FDR family."""
    subject_rows: List[Dict[str, object]] = []
    expected_conditions = len(GAMES)
    for subject in SUBJECTS:
        group = subject_summary[subject_summary["subject"] == subject]
        primary_values = group[
            "bout_weighted_delta_heldout_log_likelihood_per_bout"
        ].to_numpy(dtype=float)
        primary_values = primary_values[np.isfinite(primary_values)]
        sensitivity_values = group[
            "day_equal_mean_delta_heldout_log_likelihood_per_bout"
        ].to_numpy(dtype=float)
        sensitivity_values = sensitivity_values[np.isfinite(sensitivity_values)]
        n_conditions = int(len(primary_values))
        complete = n_conditions == expected_conditions
        subject_rows.append(
            {
                "subject": str(subject),
                "n_conditions_expected": int(expected_conditions),
                "n_conditions_valid": n_conditions,
                "condition_coverage": (
                    float(n_conditions / expected_conditions) if expected_conditions else np.nan
                ),
                "complete_condition_coverage": bool(complete),
                "bout_weighted_delta_heldout_log_likelihood_per_bout": (
                    float(np.mean(primary_values)) if complete else np.nan
                ),
                "day_equal_condition_mean_sensitivity": (
                    float(np.mean(sensitivity_values))
                    if len(sensitivity_values) == expected_conditions else np.nan
                ),
            }
        )
    common_subject = pd.DataFrame(subject_rows)
    values = common_subject[
        "bout_weighted_delta_heldout_log_likelihood_per_bout"
    ].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    n_subjects = len(values)
    mean = float(np.mean(values)) if n_subjects else np.nan
    sd = float(np.std(values, ddof=1)) if n_subjects > 1 else np.nan
    sem = float(sd / np.sqrt(n_subjects)) if n_subjects > 1 else np.nan
    if n_subjects > 1 and np.isfinite(sem) and sem > 0:
        t_statistic = float(mean / sem)
        p_value = float(2.0 * stats.t.sf(abs(t_statistic), df=n_subjects - 1))
        half_width = float(stats.t.ppf(0.975, df=n_subjects - 1) * sem)
    elif n_subjects > 0 and np.allclose(values, 0.0):
        t_statistic, p_value, half_width = 0.0, 1.0, 0.0
    else:
        t_statistic, p_value, half_width = np.nan, np.nan, np.nan
    exact_p = exact_sign_flip_p(values)
    label = "Across-game mean" if DATASET == "atari" else "Across-stage mean"
    common_inference = pd.DataFrame(
        [
            {
                "game": -1,
                "game_name": label,
                "n_subjects": int(n_subjects),
                "n_conditions_equal_weight": int(expected_conditions),
                "mean_bout_weighted_delta_heldout_log_likelihood_per_bout": mean,
                "sd_subject_delta": sd,
                "sem_subject_delta": sem,
                "ci95_low": mean - half_width if np.isfinite(half_width) else np.nan,
                "ci95_high": mean + half_width if np.isfinite(half_width) else np.nan,
                "t_statistic": t_statistic,
                "p_value": p_value,
                "q_fdr": np.nan,
                "significance_fdr": "not_applicable",
                "display_significance": stars_for_pvalue(p_value),
                "exact_sign_flip_p": exact_p,
                "q_fdr_exact_sign_flip": np.nan,
                "significance_exact_fdr": "not_applicable",
                "n_positive": int(np.sum(values > 0)),
                "n_negative": int(np.sum(values < 0)),
                "n_zero": int(np.sum(values == 0)),
                "estimand": (
                    "dataset_common_subject_bout_weighted_within_condition_"
                    "then_equal_condition_weight"
                ),
                "multiple_testing_family": "dataset_common_single_prespecified_test",
            }
        ]
    )
    return common_subject, common_inference


def summarize_day_heldout_hazard_consistency(
    subject_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Descriptive condition and held-out-fold direction counts by participant."""
    columns = [
        "subject", "n_conditions_expected", "n_conditions_valid",
        "n_positive_conditions", "n_nonpositive_conditions",
        "positive_condition_fraction", "n_valid_heldout_folds",
        "n_positive_heldout_folds", "n_negative_heldout_folds",
        "n_zero_heldout_folds", "n_nonpositive_heldout_folds",
        "positive_heldout_fold_fraction",
    ]
    value_column = "bout_weighted_delta_heldout_log_likelihood_per_bout"
    rows: List[Dict[str, object]] = []
    for subject in SUBJECTS:
        group = subject_summary[subject_summary["subject"] == subject].copy()
        condition_values = group[value_column].to_numpy(dtype=float)
        condition_values = condition_values[np.isfinite(condition_values)]
        n_valid_folds = int(group["n_days_valid"].sum()) if not group.empty else 0
        n_positive_folds = (
            int(group["n_positive_heldout_days"].sum()) if not group.empty else 0
        )
        n_negative_folds = (
            int(group["n_negative_heldout_days"].sum()) if not group.empty else 0
        )
        n_zero_folds = (
            int(group["n_zero_heldout_days"].sum()) if not group.empty else 0
        )
        n_positive_conditions = int(np.sum(condition_values > 0))
        rows.append(
            {
                "subject": str(subject),
                "n_conditions_expected": int(len(GAMES)),
                "n_conditions_valid": int(len(condition_values)),
                "n_positive_conditions": n_positive_conditions,
                "n_nonpositive_conditions": int(
                    len(condition_values) - n_positive_conditions
                ),
                "positive_condition_fraction": (
                    float(n_positive_conditions / len(condition_values))
                    if len(condition_values) else np.nan
                ),
                "n_valid_heldout_folds": n_valid_folds,
                "n_positive_heldout_folds": n_positive_folds,
                "n_negative_heldout_folds": n_negative_folds,
                "n_zero_heldout_folds": n_zero_folds,
                "n_nonpositive_heldout_folds": int(
                    n_negative_folds + n_zero_folds
                ),
                "positive_heldout_fold_fraction": (
                    float(n_positive_folds / n_valid_folds)
                    if n_valid_folds else np.nan
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def save_day_heldout_hazard_figure(
    subject_summary: pd.DataFrame,
    inference: pd.DataFrame,
    common_subject: pd.DataFrame,
    common_inference: pd.DataFrame,
    consistency: pd.DataFrame,
    out_path: Path,
) -> None:
    """Participant-first predictive gains and descriptive fold consistency."""
    value_column = "bout_weighted_delta_heldout_log_likelihood_per_bout"
    common_by_subject = common_subject.set_index("subject")
    consistency_by_subject = consistency.set_index("subject")
    participant_values = np.asarray(
        [
            float(common_by_subject.loc[subject, value_column])
            if subject in common_by_subject.index else np.nan
            for subject in SUBJECTS
        ],
        dtype=float,
    )
    common_row = common_inference.iloc[0] if not common_inference.empty else None
    common_mean = (
        float(common_row["mean_bout_weighted_delta_heldout_log_likelihood_per_bout"])
        if common_row is not None else np.nan
    )
    common_low = float(common_row["ci95_low"]) if common_row is not None else np.nan
    common_high = float(common_row["ci95_high"]) if common_row is not None else np.nan
    common_star = str(common_row["display_significance"]) if common_row is not None else "n/a"

    x_positions = np.arange(len(SUBJECTS) + 1, dtype=float)
    x_labels = [f"S{subject_sort_key(subject)}" for subject in SUBJECTS] + ["All"]
    upper_values = np.r_[participant_values, common_mean]
    participant_color = "#4f8fbf"
    all_color = "#263746"
    bar_colors = [participant_color] * len(SUBJECTS) + [all_color]

    positive_folds: List[int] = []
    valid_folds: List[int] = []
    positive_conditions: List[int] = []
    valid_conditions: List[int] = []
    for subject in SUBJECTS:
        if subject in consistency_by_subject.index:
            row = consistency_by_subject.loc[subject]
            positive_folds.append(int(row["n_positive_heldout_folds"]))
            valid_folds.append(int(row["n_valid_heldout_folds"]))
            positive_conditions.append(int(row["n_positive_conditions"]))
            valid_conditions.append(int(row["n_conditions_valid"]))
        else:
            positive_folds.append(0)
            valid_folds.append(0)
            positive_conditions.append(0)
            valid_conditions.append(0)
    positive_folds.append(int(np.sum(positive_folds)))
    valid_folds.append(int(np.sum(valid_folds)))
    positive_conditions.append(int(np.sum(positive_conditions)))
    valid_conditions.append(int(np.sum(valid_conditions)))
    fold_fractions = np.divide(
        positive_folds,
        valid_folds,
        out=np.full(len(valid_folds), np.nan, dtype=float),
        where=np.asarray(valid_folds, dtype=float) > 0,
    )

    fig, (ax_gain, ax_consistency) = plt.subplots(
        2, 1, figsize=(11.2, 8.2), sharex=True,
        gridspec_kw={"height_ratios": (1.65, 1.0), "hspace": 0.30},
    )
    ax_gain.bar(
        x_positions, upper_values, width=0.66, color=bar_colors,
        edgecolor="white", linewidth=0.6, zorder=2,
    )
    if np.all(np.isfinite([common_mean, common_low, common_high])):
        ax_gain.errorbar(
            x_positions[-1], common_mean,
            yerr=np.asarray(
                [[max(common_mean - common_low, 0.0)],
                 [max(common_high - common_mean, 0.0)]]
            ),
            fmt="none", ecolor="#111111", elinewidth=1.6,
            capsize=5, capthick=1.4, zorder=4,
        )
    plotted = np.r_[upper_values, common_low, common_high, 0.0]
    plotted = plotted[np.isfinite(plotted)]
    lower = float(np.min(plotted)) if len(plotted) else -0.01
    upper = float(np.max(plotted)) if len(plotted) else 0.01
    span = max(upper - lower, 0.02)
    annotation_offset = 0.035 * span
    for index, value in enumerate(upper_values):
        if not np.isfinite(value):
            continue
        y = value + annotation_offset if value >= 0 else value - annotation_offset
        ax_gain.text(
            x_positions[index], y,
            f"{positive_conditions[index]}/{valid_conditions[index]} cond. +",
            ha="center", va="bottom" if value >= 0 else "top",
            fontsize=8, color="#333333",
        )
    if np.isfinite(common_mean):
        star_anchor = max(
            candidate for candidate in (common_mean, common_high)
            if np.isfinite(candidate)
        )
        ax_gain.text(
            x_positions[-1], star_anchor + 0.14 * span, common_star,
            ha="center", va="bottom", fontsize=12, fontweight="bold",
        )
    ax_gain.set_ylim(lower - 0.15 * span, upper + 0.35 * span)
    ax_gain.axhline(0.0, color="#444444", lw=1.0, ls="--", zorder=1)
    ax_gain.axvline(len(SUBJECTS) - 0.5, color="#a7adb3", lw=0.9)
    ax_gain.set_ylabel("Delta held-out log likelihood\nper NOOP bout")
    ax_gain.set_title(
        "A. Predictive gain by participant (duration-dependent minus constant hazard)",
        fontsize=11,
    )
    ax_gain.grid(axis="y", alpha=0.22, zorder=0)
    ax_gain.spines[["top", "right"]].set_visible(False)

    positive_plot = np.nan_to_num(fold_fractions, nan=0.0)
    nonpositive_plot = np.where(np.isfinite(fold_fractions), 1.0 - fold_fractions, 0.0)
    ax_consistency.bar(
        x_positions, positive_plot, width=0.66, color=participant_color,
        edgecolor="white", linewidth=0.6, label="Delta LL > 0", zorder=2,
    )
    ax_consistency.bar(
        x_positions, nonpositive_plot, width=0.66, bottom=positive_plot,
        color="#d9dde1", edgecolor="white", linewidth=0.6,
        label="Delta LL <= 0", zorder=2,
    )
    for index, (positive, total) in enumerate(zip(positive_folds, valid_folds)):
        label = f"{positive}/{total}" if total else "n/a"
        ax_consistency.text(
            x_positions[index], 1.025, label,
            ha="center", va="bottom", fontsize=8, color="#333333",
        )
    ax_consistency.axvline(len(SUBJECTS) - 0.5, color="#a7adb3", lw=0.9)
    ax_consistency.set_ylim(0.0, 1.13)
    ax_consistency.set_yticks((0.0, 0.5, 1.0), ("0%", "50%", "100%"))
    ax_consistency.set_xticks(x_positions, x_labels)
    ax_consistency.set_xlabel("Participant")
    ax_consistency.set_ylabel("Valid held-out folds")
    ax_consistency.set_title(
        "B. Fraction of held-out days with positive predictive gain (descriptive)",
        fontsize=11,
    )
    ax_consistency.grid(axis="y", alpha=0.22, zorder=0)
    ax_consistency.spines[["top", "right"]].set_visible(False)
    ax_consistency.legend(
        loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.30)
    )
    fig.suptitle(
        "1-5: Day-held-out generalization of duration-dependent NOOP hazards",
        fontsize=14, y=0.995,
    )
    condition_q = inference["q_fdr"].to_numpy(dtype=float) if not inference.empty else np.asarray([])
    condition_q = condition_q[np.isfinite(condition_q)]
    condition_hits = int(np.sum(condition_q < 0.05))
    t_p = float(common_row["p_value"]) if common_row is not None else np.nan
    exact_p = float(common_row["exact_sign_flip_p"]) if common_row is not None else np.nan
    fig.text(
        0.5, 0.012,
        "A: held-out likelihood is pooled over bouts within condition, then conditions are equal-weighted within participant. "
        "All is the equal-participant mean with 95% t CI; its star is the single prespecified common test "
        f"(t p={t_p:.4f}, exact sign-flip p={exact_p:.4f}; no BH).\n"
        f"Condition-level BH-FDR hits: {condition_hits}/{len(condition_q)}. "
        "B is a descriptive fold-direction summary only; n/N labels give positive/valid held-out days.",
        ha="center", va="bottom", fontsize=8.5, color="#444444",
    )
    fig.subplots_adjust(top=0.91, bottom=0.25, left=0.10, right=0.98, hspace=0.36)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_day_heldout_hazard_analysis(
    bout_table: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    folds = build_day_heldout_hazard_folds(bout_table)
    subject_summary = summarize_day_heldout_hazard_subjects(folds)
    inference = summarize_day_heldout_hazard_inference(subject_summary)
    common_subject, common_inference = summarize_day_heldout_hazard_common_effect(
        subject_summary
    )
    consistency = summarize_day_heldout_hazard_consistency(subject_summary)
    folds.to_csv(OUT_DIR / "1-5_day_heldout_hazard_folds.csv", index=False)
    subject_summary.to_csv(OUT_DIR / "1-5_day_heldout_hazard_subject.csv", index=False)
    inference.to_csv(OUT_DIR / "1-5_day_heldout_hazard_inference.csv", index=False)
    common_subject.to_csv(
        OUT_DIR / "1-5_day_heldout_hazard_common_subject.csv", index=False
    )
    common_inference.to_csv(
        OUT_DIR / "1-5_day_heldout_hazard_common_inference.csv", index=False
    )
    consistency.to_csv(
        OUT_DIR / "1-5_day_heldout_hazard_consistency.csv", index=False
    )
    save_day_heldout_hazard_figure(
        subject_summary,
        inference,
        common_subject,
        common_inference,
        consistency,
        FIG_DIR / "fig_1-5_day_heldout_hazard_generalization.png",
    )
    return folds, subject_summary, inference, common_subject, common_inference


def discrete_weibull_median(q: float, beta: float) -> float:
    return float(np.power(np.log(0.5) / np.log(q), 1.0 / beta))


def fit_two_component_discrete_weibull(
    lengths: np.ndarray,
    events: np.ndarray,
) -> Dict[str, float]:
    lengths = np.asarray(lengths, dtype=int)
    events = np.asarray(events, dtype=bool)
    unique_lengths, event_counts, censor_counts = compress_duration_counts(lengths, events)
    uncensored = lengths[events]
    if len(uncensored) == 0:
        uncensored = lengths
    medians = np.quantile(uncensored, [0.3, 0.75])
    q_starts = [
        float(np.power(0.5, 1.0 / max(float(medians[0]), 1.0))),
        float(np.power(0.5, 1.0 / max(float(medians[1]), 1.0))),
    ]

    def objective(theta: np.ndarray) -> float:
        q1, q2 = float(expit(theta[0])), float(expit(theta[2]))
        beta1, beta2 = float(np.exp(theta[1])), float(np.exp(theta[3]))
        weight1 = float(expit(theta[4]))
        event1, censor1 = _discrete_weibull_log_terms(unique_lengths, q1, beta1)
        event2, censor2 = _discrete_weibull_log_terms(unique_lengths, q2, beta2)
        log_w1 = np.log(weight1)
        log_w2 = np.log1p(-weight1)
        log_event = logsumexp(np.vstack([log_w1 + event1, log_w2 + event2]), axis=0)
        log_censor = logsumexp(np.vstack([log_w1 + censor1, log_w2 + censor2]), axis=0)
        return -float(np.sum(event_counts * log_event + censor_counts * log_censor))

    starts: List[np.ndarray] = []
    for beta1 in (0.65, 1.0, 1.5):
        for beta2 in (0.65, 1.0, 1.5):
            for weight1 in (0.3, 0.6):
                starts.append(
                    np.array(
                        [
                            _logit(q_starts[0]),
                            np.log(beta1),
                            _logit(q_starts[1]),
                            np.log(beta2),
                            _logit(weight1),
                        ]
                    )
                )

    best = None
    for start in starts:
        result = optimize.minimize(
            objective,
            x0=start,
            method="L-BFGS-B",
            bounds=[
                (-12.0, 12.0),
                (np.log(0.1), np.log(5.0)),
                (-12.0, 12.0),
                (np.log(0.1), np.log(5.0)),
                (-3.0, 3.0),
            ],
            options={"maxiter": 800},
        )
        if np.isfinite(result.fun) and (best is None or result.fun < best.fun):
            best = result

    if best is None:
        raise RuntimeError("Two-component discrete-Weibull fit failed")

    q1, q2 = float(expit(best.x[0])), float(expit(best.x[2]))
    beta1, beta2 = float(np.exp(best.x[1])), float(np.exp(best.x[3]))
    weight1 = float(expit(best.x[4]))
    components = [
        {"q": q1, "beta": beta1, "weight": weight1},
        {"q": q2, "beta": beta2, "weight": 1.0 - weight1},
    ]
    for component in components:
        component["median"] = discrete_weibull_median(component["q"], component["beta"])
    components.sort(key=lambda component: component["median"])

    return {
        "q1": float(components[0]["q"]),
        "beta1": float(components[0]["beta"]),
        "weight1": float(components[0]["weight"]),
        "median1": float(components[0]["median"]),
        "q2": float(components[1]["q"]),
        "beta2": float(components[1]["beta"]),
        "weight2": float(components[1]["weight"]),
        "median2": float(components[1]["median"]),
        "log_likelihood": float(-best.fun),
        "success": bool(best.success),
    }


def fit_duration_models(
    bout_table: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[Tuple[str, int], Dict[str, np.ndarray]]]:
    noop_table = bout_table[bout_table["bout_type"] == "NOOP"].copy()
    summary_rows: List[Dict[str, object]] = []
    component_rows: List[Dict[str, object]] = []
    envelopes: Dict[Tuple[str, int], Dict[str, np.ndarray]] = {}

    for (subject, game), group in noop_table.groupby(["subject", "game"], sort=True):
        lengths = group["length"].to_numpy(dtype=int)
        events = (~group["censored"].to_numpy(dtype=bool))
        censor_limits = group["censor_limit"].to_numpy(dtype=int)
        t_max = max(2, int(np.percentile(lengths, 99.5)))
        if DATASET == "mario":
            t_max = min(t_max, 600)
        else:
            t_max = min(t_max, 300 if int(game) == 1 else 160)
        times = np.arange(1, t_max + 1, dtype=int)
        seed = SURVIVAL_RANDOM_SEED + 100 * subject_sort_key(str(subject)) + int(game)
        hazard_summary, envelope_low, envelope_high = bootstrap_duration_hazard_test(
            lengths,
            events,
            censor_limits,
            times,
            n_bootstrap=SURVIVAL_BOOTSTRAPS,
            seed=seed,
        )

        one_component = fit_discrete_weibull(lengths, events)
        two_component = fit_two_component_discrete_weibull(lengths, events)
        bic_one = 2 * np.log(len(lengths)) - 2 * one_component["log_likelihood"]
        bic_two = 5 * np.log(len(lengths)) - 2 * two_component["log_likelihood"]
        delta_bic = float(bic_one - bic_two)

        summary_rows.append(
            {
                "subject": subject,
                "game": int(game),
                "game_name": GAMES[int(game)],
                **hazard_summary,
                "one_component_q": float(one_component["q"]),
                "one_component_beta": float(one_component["beta"]),
                "one_component_log_likelihood": float(one_component["log_likelihood"]),
                "two_component_log_likelihood": float(two_component["log_likelihood"]),
                "bic_one_component": float(bic_one),
                "bic_two_component": float(bic_two),
                "delta_bic_one_minus_two": delta_bic,
                "preferred_distribution": "two-component" if delta_bic > 10.0 else "one-component",
            }
        )
        for component_index in (1, 2):
            component_rows.append(
                {
                    "subject": subject,
                    "game": int(game),
                    "game_name": GAMES[int(game)],
                    "component": component_index,
                    "weight": float(two_component[f"weight{component_index}"]),
                    "q": float(two_component[f"q{component_index}"]),
                    "beta": float(two_component[f"beta{component_index}"]),
                    "median_steps": float(two_component[f"median{component_index}"]),
                    "delta_bic_one_minus_two": delta_bic,
                }
            )
        envelopes[(str(subject), int(game))] = {
            "times": times,
            "low": envelope_low,
            "high": envelope_high,
        }
        subject_label = "Pooled" if str(subject) == "pooled" else f"S{subject_sort_key(str(subject))}"
        print(
            f"  {subject_label} {GAMES[int(game)]}: "
            f"beta={hazard_summary['duration_beta']:.3f}, "
            f"bootstrap p={hazard_summary['p_bootstrap']:.4f}, "
            f"delta BIC={delta_bic:.1f}"
        )

    summary = pd.DataFrame(summary_rows).sort_values(["subject", "game"]).reset_index(drop=True)
    summary["q_bh_bootstrap"] = benjamini_hochberg(summary["p_bootstrap"])
    summary["constant_hazard_rejected_fdr05"] = summary["q_bh_bootstrap"] < 0.05
    components = pd.DataFrame(component_rows).sort_values(
        ["subject", "game", "component"]
    ).reset_index(drop=True)
    return summary, components, envelopes


def _format_probability(value: float) -> str:
    if value < 0.001:
        return "<.001"
    return f"{value:.3f}".lstrip("0")


def save_noop_survival_figure(
    bout_table: pd.DataFrame,
    model_summary: pd.DataFrame,
    envelopes: Dict[Tuple[str, int], Dict[str, np.ndarray]],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, len(SUBJECTS), figsize=(19, 7.0), sharey=True)
    for row_index, game in enumerate(GAMES):
        for column_index, subject in enumerate(SUBJECTS):
            ax = axes[row_index, column_index]
            group = bout_table[
                (bout_table["subject"] == subject)
                & (bout_table["game"] == game)
                & (bout_table["bout_type"] == "NOOP")
            ]
            fit_row = model_summary[
                (model_summary["subject"] == subject) & (model_summary["game"] == game)
            ]
            if group.empty or fit_row.empty:
                ax.set_axis_off()
                continue
            fit = fit_row.iloc[0]
            envelope = envelopes[(subject, game)]
            times = envelope["times"]
            lengths = group["length"].to_numpy(dtype=int)
            events = ~group["censored"].to_numpy(dtype=bool)
            observed = kaplan_meier_survival_arrays(lengths, events, times)
            geometric = np.power(float(fit["geometric_q"]), times)
            piecewise_fit = fit_piecewise_hazard(lengths, events)
            fitted_hazard = np.asarray(piecewise_fit["hazard_by_time"], dtype=float)[1 : len(times) + 1]
            duration_fit = np.cumprod(1.0 - fitted_hazard)

            ax.fill_between(times, envelope["low"], envelope["high"], color="#b7b7b7", alpha=0.35)
            ax.step(times, observed, where="post", color="#1f5a94", lw=1.8)
            ax.plot(times, geometric, color="#555555", lw=1.2, ls="--")
            ax.plot(times, duration_fit, color="#d97706", lw=1.5)
            ax.set_xscale("log")
            ax.set_xlim(1, int(times[-1]))
            ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_title(
                f"S{subject_sort_key(subject)}  LR={fit['lr_statistic']:.1f}\n"
                f"BH-FDR q {_format_probability(float(fit['q_bh_bootstrap']))}",
                fontsize=9,
            )
            if row_index == 1:
                ax.set_xlabel("NOOP-bout duration (steps)")
            if column_index == 0:
                ax.set_ylabel(f"{GAME_TITLES[game]}\nSurvival probability")

    handles = [
        Line2D([0], [0], color="#1f5a94", lw=1.8, label="Kaplan-Meier estimate"),
        Line2D([0], [0], color="#555555", lw=1.2, ls="--", label="Constant-hazard geometric fit"),
        Line2D([0], [0], color="#d97706", lw=1.5, label="Piecewise duration-dependent fit"),
        Line2D([0], [0], color="#b7b7b7", lw=7, alpha=0.35, label="95% geometric-null envelope"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle("NOOP-bout survival under constant and duration-dependent hazards", fontsize=14, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _binned_empirical_hazard(
    lengths: np.ndarray,
    events: np.ndarray,
    t_max: int,
    n_bins: int = 10,
) -> pd.DataFrame:
    edges = np.unique(np.round(np.geomspace(1, t_max + 1, n_bins + 1)).astype(int))
    edges = np.unique(np.r_[1, edges, t_max + 1])
    rows: List[Dict[str, float]] = []
    for left, right in zip(edges[:-1], edges[1:]):
        integer_times = np.arange(left, right, dtype=int)
        exposures = int(sum(np.sum(lengths >= t) for t in integer_times))
        n_events = int(sum(np.sum((lengths == t) & events) for t in integer_times))
        if exposures == 0:
            continue
        estimate = n_events / exposures
        ci_low = 0.0 if n_events == 0 else float(stats.beta.ppf(0.025, n_events, exposures - n_events + 1))
        ci_high = 1.0 if n_events == exposures else float(stats.beta.ppf(0.975, n_events + 1, exposures - n_events))
        rows.append(
            {
                "time": float(np.sqrt(left * max(right - 1, left))),
                "left": int(left),
                "right": int(right),
                "exposure": exposures,
                "events": n_events,
                "hazard": estimate,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    return pd.DataFrame(rows)


def save_noop_hazard_figure(
    bout_table: pd.DataFrame,
    model_summary: pd.DataFrame,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, len(SUBJECTS), figsize=(19, 7.0), sharey="row")
    for row_index, game in enumerate(GAMES):
        for column_index, subject in enumerate(SUBJECTS):
            ax = axes[row_index, column_index]
            group = bout_table[
                (bout_table["subject"] == subject)
                & (bout_table["game"] == game)
                & (bout_table["bout_type"] == "NOOP")
            ]
            fit_row = model_summary[
                (model_summary["subject"] == subject) & (model_summary["game"] == game)
            ]
            if group.empty or fit_row.empty:
                ax.set_axis_off()
                continue
            fit = fit_row.iloc[0]
            lengths = group["length"].to_numpy(dtype=int)
            events = ~group["censored"].to_numpy(dtype=bool)
            t_max = min(max(2, int(np.percentile(lengths, 99.5))), 300 if game == 1 else 160)
            empirical = _binned_empirical_hazard(lengths, events, t_max)
            piecewise_fit = fit_piecewise_hazard(lengths, events)

            ax.errorbar(
                empirical["time"],
                empirical["hazard"],
                yerr=np.vstack([
                    empirical["hazard"] - empirical["ci_low"],
                    empirical["ci_high"] - empirical["hazard"],
                ]),
                fmt="o",
                ms=3.5,
                color="#1f5a94",
                ecolor="#7aa6c9",
                capsize=2,
            )
            ax.axhline(float(fit["constant_hazard"]), color="#555555", lw=1.2, ls="--")
            for left, right, hazard in zip(
                np.asarray(piecewise_fit["edges"])[:-1],
                np.asarray(piecewise_fit["edges"])[1:],
                np.asarray(piecewise_fit["hazards"]),
            ):
                if left > t_max:
                    break
                ax.hlines(
                    hazard,
                    left,
                    min(right, t_max),
                    color="#d97706",
                    lw=1.5,
                )
            ax.set_xscale("log")
            ax.set_xlim(1, t_max)
            ax.grid(alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_title(f"S{subject_sort_key(subject)}", fontsize=9)
            if row_index == 1:
                ax.set_xlabel("NOOP-bout duration (steps)")
            if column_index == 0:
                ax.set_ylabel(f"{GAME_TITLES[game]}\nExit hazard per step")

    handles = [
        Line2D([0], [0], marker="o", color="#1f5a94", lw=0, label="Binned empirical hazard (95% CI)"),
        Line2D([0], [0], color="#555555", lw=1.2, ls="--", label="Constant-hazard null"),
        Line2D([0], [0], color="#d97706", lw=1.5, label="Duration-dependent fit"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle("Probability of ending a NOOP bout as duration increases", fontsize=14, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _discrete_weibull_pmf(times: np.ndarray, q: float, beta: float) -> np.ndarray:
    log_event, _ = _discrete_weibull_log_terms(times, q, beta)
    return np.exp(log_event)


def save_duration_mixture_figure(
    bout_table: pd.DataFrame,
    model_summary: pd.DataFrame,
    components: pd.DataFrame,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, len(SUBJECTS), figsize=(19, 7.2), sharey="row")
    for row_index, game in enumerate(GAMES):
        for column_index, subject in enumerate(SUBJECTS):
            ax = axes[row_index, column_index]
            group = bout_table[
                (bout_table["subject"] == subject)
                & (bout_table["game"] == game)
                & (bout_table["bout_type"] == "NOOP")
            ]
            fit_row = model_summary[
                (model_summary["subject"] == subject) & (model_summary["game"] == game)
            ]
            component_rows = components[
                (components["subject"] == subject) & (components["game"] == game)
            ].sort_values("component")
            if group.empty or fit_row.empty or len(component_rows) != 2:
                ax.set_axis_off()
                continue
            fit = fit_row.iloc[0]
            observed = group.loc[~group["censored"], "length"].to_numpy(dtype=int)
            x_max = min(max(3, int(np.percentile(group["length"], 99.5))), 500)
            edges = np.unique(np.round(np.geomspace(1, x_max + 1, 24)).astype(int))
            edges = np.unique(np.r_[1, edges, x_max + 1])
            counts, bins = np.histogram(observed, bins=edges)
            widths = np.diff(bins)
            empirical_mass = counts / max(np.sum(counts), 1)
            centers = np.sqrt(bins[:-1] * np.maximum(bins[1:] - 1, bins[:-1]))
            ax.bar(
                centers,
                empirical_mass,
                width=widths * 0.9,
                color="#b9c3cc",
                edgecolor="#66717b",
                linewidth=0.4,
                alpha=0.7,
                align="center",
            )

            integer_times = np.arange(1, x_max + 1, dtype=int)
            one_pmf = _discrete_weibull_pmf(
                integer_times,
                float(fit["one_component_q"]),
                float(fit["one_component_beta"]),
            )
            component_density: List[np.ndarray] = []
            for _, component in component_rows.iterrows():
                component_density.append(
                    float(component["weight"])
                    * _discrete_weibull_pmf(
                        integer_times,
                        float(component["q"]),
                        float(component["beta"]),
                    )
                )
            mixture_pmf = component_density[0] + component_density[1]

            def aggregate_to_bins(pmf: np.ndarray) -> np.ndarray:
                values = []
                for left, right in zip(bins[:-1], bins[1:]):
                    values.append(float(np.sum(pmf[left - 1 : right - 1])))
                return np.asarray(values)

            ax.plot(centers, aggregate_to_bins(one_pmf), color="#555555", lw=1.2, ls="--")
            ax.plot(centers, aggregate_to_bins(mixture_pmf), color="#1f5a94", lw=1.8)
            ax.plot(centers, aggregate_to_bins(component_density[0]), color="#d97706", lw=1.2)
            ax.plot(centers, aggregate_to_bins(component_density[1]), color="#a23b72", lw=1.2)
            ax.set_xscale("log")
            ax.set_xlim(1, x_max)
            ax.grid(axis="y", alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_title(
                f"S{subject_sort_key(subject)}  delta BIC={fit['delta_bic_one_minus_two']:.1f}\n"
                f"medians={component_rows.iloc[0]['median_steps']:.1f}, {component_rows.iloc[1]['median_steps']:.1f}",
                fontsize=8.5,
            )
            if row_index == 1:
                ax.set_xlabel("Observed NOOP-bout duration (steps)")
            if column_index == 0:
                ax.set_ylabel(f"{GAME_TITLES[game]}\nProbability mass per log-duration bin")

    handles = [
        Line2D([0], [0], color="#66717b", lw=7, alpha=0.7, label="Observed uncensored bouts"),
        Line2D([0], [0], color="#555555", lw=1.2, ls="--", label="One-component fit"),
        Line2D([0], [0], color="#1f5a94", lw=1.8, label="Two-component mixture"),
        Line2D([0], [0], color="#d97706", lw=1.2, label="Shorter-duration component"),
        Line2D([0], [0], color="#a23b72", lw=1.2, label="Longer-duration component"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle("Fitted NOOP-bout duration distributions without a fixed cross point", fontsize=14, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def competing_risk_curves(group: pd.DataFrame, times: np.ndarray) -> pd.DataFrame:
    lengths = group["length"].to_numpy(dtype=int)
    event_types = group["event_type"].astype(str).to_numpy()
    max_time = max(int(np.max(lengths)), int(np.max(times)))
    length_counts = np.bincount(lengths, minlength=max_time + 1)
    at_risk = np.cumsum(length_counts[::-1])[::-1]
    noop_counts = np.bincount(
        lengths[event_types == "to_noop"], minlength=max_time + 1
    )
    other_counts = np.bincount(
        lengths[event_types == "to_other_active"], minlength=max_time + 1
    )
    survival = 1.0
    cif_noop = 0.0
    cif_other = 0.0
    rows: List[Dict[str, float]] = []
    for t in times:
        n_risk = int(at_risk[int(t)])
        d_noop = int(noop_counts[int(t)])
        d_other = int(other_counts[int(t)])
        survival_previous = survival
        if n_risk > 0:
            cif_noop += survival_previous * d_noop / n_risk
            cif_other += survival_previous * d_other / n_risk
            survival *= 1.0 - (d_noop + d_other) / n_risk
        rows.append(
            {
                "time": int(t),
                "n_risk": n_risk,
                "active_persistence": survival,
                "cif_to_noop": cif_noop,
                "cif_to_other_active": cif_other,
            }
        )
    return pd.DataFrame(rows)


def build_persistence_curves(bout_table: pd.DataFrame) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for (subject, game), subject_game in bout_table.groupby(["subject", "game"], sort=True):
        active = subject_game[subject_game["bout_type"] == "Active"]
        noop = subject_game[subject_game["bout_type"] == "NOOP"]
        if active.empty or noop.empty:
            continue
        t_max = max(2, int(np.percentile(np.r_[active["length"], noop["length"]], 99.0)))
        t_max = min(t_max, 300)
        times = np.arange(1, t_max + 1, dtype=int)
        active_curves = competing_risk_curves(active, times)
        noop_survival = kaplan_meier_survival_arrays(
            noop["length"].to_numpy(dtype=int),
            ~noop["censored"].to_numpy(dtype=bool),
            times,
        )
        active_curves["noop_persistence"] = noop_survival
        active_curves["subject"] = subject
        active_curves["game"] = int(game)
        active_curves["game_name"] = GAMES[int(game)]
        rows.append(active_curves)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def save_persistence_competing_risk_figure(
    curves: pd.DataFrame,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, len(SUBJECTS), figsize=(19, 7.0), sharey=True)
    for row_index, game in enumerate(GAMES):
        for column_index, subject in enumerate(SUBJECTS):
            ax = axes[row_index, column_index]
            group = curves[(curves["subject"] == subject) & (curves["game"] == game)]
            if group.empty:
                ax.set_axis_off()
                continue
            ax.plot(group["time"], group["noop_persistence"], color="#555555", lw=1.4, ls="--")
            ax.plot(group["time"], group["active_persistence"], color="#1f5a94", lw=1.8)
            ax.plot(group["time"], group["cif_to_noop"], color="#d97706", lw=1.5)
            ax.plot(group["time"], group["cif_to_other_active"], color="#a23b72", lw=1.5, ls=":")
            ax.set_xscale("log")
            ax.set_xlim(1, int(group["time"].max()))
            ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_title(f"S{subject_sort_key(subject)}", fontsize=9)
            if row_index == 1:
                ax.set_xlabel("Same-action run duration (steps)")
            if column_index == 0:
                ax.set_ylabel(f"{GAME_TITLES[game]}\nProbability")

    handles = [
        Line2D([0], [0], color="#555555", lw=1.4, ls="--", label="NOOP persistence"),
        Line2D([0], [0], color="#1f5a94", lw=1.8, label="Same-active-action persistence"),
        Line2D([0], [0], color="#d97706", lw=1.5, label="Cumulative incidence: active to NOOP"),
        Line2D([0], [0], color="#a23b72", lw=1.5, ls=":", label="Cumulative incidence: active to another active action"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle("Persistence and exit routes of NOOP and active-action runs", fontsize=14, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _stage_grid(n_panels: int, n_columns: int = 5) -> Tuple[plt.Figure, np.ndarray]:
    n_rows = int(math.ceil(n_panels / n_columns))
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(4.0 * n_columns, 3.15 * n_rows))
    axes_flat = np.atleast_1d(axes).reshape(-1)
    for ax in axes_flat[n_panels:]:
        ax.set_axis_off()
    return fig, axes_flat


def _plot_stage_subject_metric(
    summary: pd.DataFrame,
    value_column: str,
    title: str,
    colorbar_label: str,
    out_path: Path,
    reference: float | None = None,
    cmap: str = "viridis",
) -> None:
    stages = list(GAMES)
    stage_names = [GAMES[stage] for stage in stages]
    matrix = (
        summary.pivot(index="subject", columns="game", values=value_column)
        .reindex(index=SUBJECTS, columns=stages)
    )
    values = matrix.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(19, 8.0),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    image = axes[0].imshow(masked, aspect="auto", interpolation="nearest", cmap=cmap)
    axes[0].set_yticks(np.arange(len(SUBJECTS)))
    axes[0].set_yticklabels([f"S{subject_sort_key(subject)}" for subject in SUBJECTS])
    axes[0].set_ylabel("Participant")
    colorbar = fig.colorbar(image, ax=axes[0], fraction=0.02, pad=0.015)
    colorbar.set_label(colorbar_label)

    means = np.nanmean(values, axis=0)
    valid_counts = np.sum(np.isfinite(values), axis=0)
    sem = np.nanstd(values, axis=0, ddof=1) / np.sqrt(np.maximum(valid_counts, 1))
    x = np.arange(len(stages))
    axes[1].errorbar(x, means, yerr=sem, color="#1f5a94", marker="o", ms=4, lw=1.5, capsize=3)
    if reference is not None:
        axes[1].axhline(reference, color="#555555", lw=1.0, ls="--")
    axes[1].set_ylabel(f"Mean ± SE\n{colorbar_label}")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(stage_names, rotation=45, ha="right")
    axes[1].set_xlabel("Mario stage")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=15, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_mario_action_distribution(
    episodes: Sequence[Dict],
    out_path: Path,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for ep in episodes:
        distribution = compute_action_distribution(ep["actions"])
        for action_index, proportion in enumerate(distribution):
            rows.append(
                {
                    "subject": ep["subject"],
                    "game": int(ep["game"]),
                    "stage": ep["game_name"],
                    "day": int(ep["day"]),
                    "block": int(ep["block"]),
                    "ep_idx": int(ep["ep_idx"]),
                    "action": int(action_index),
                    "action_label": ACTION_LABELS[action_index],
                    "proportion": float(proportion),
                }
            )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["game", "stage", "action", "action_label"], as_index=False)
        .agg(mean_proportion=("proportion", "mean"), n_episodes=("proportion", "size"))
    )
    matrix = (
        summary.pivot(index="game", columns="action", values="mean_proportion")
        .reindex(index=list(GAMES), columns=np.arange(N_ACTIONS))
    )
    fig, ax = plt.subplots(figsize=(13, 11))
    image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap="viridis", vmin=0)
    ax.set_yticks(np.arange(len(GAMES)))
    ax.set_yticklabels([GAMES[stage] for stage in GAMES])
    ax.set_xticks(np.arange(N_ACTIONS))
    ax.set_xticklabels(ACTION_LABELS, rotation=35, ha="right")
    ax.set_xlabel("Human action")
    ax.set_ylabel("Mario stage")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label("Episode-mean action proportion")
    ax.set_title("Human action distribution across Mario stages", pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return detail


def _save_mario_temporal_structure(
    episodes: Sequence[Dict],
    out_path: Path | None,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for ep in episodes:
        actions = np.asarray(ep["actions"], dtype=int)
        thirds = np.array_split(actions, 3)
        rows.append(
            {
                "subject": ep["subject"],
                "game": int(ep["game"]),
                "stage": ep["game_name"],
                "lag1_noop_autocorrelation": compute_lag1_autocorr(actions),
                "early_noop": float(np.mean(thirds[0] == NOOP_ACTION)) if len(thirds[0]) else np.nan,
                "middle_noop": float(np.mean(thirds[1] == NOOP_ACTION)) if len(thirds[1]) else np.nan,
                "late_noop": float(np.mean(thirds[2] == NOOP_ACTION)) if len(thirds[2]) else np.nan,
            }
        )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["subject", "game", "stage"], as_index=False)
        .agg(
            lag1_noop_autocorrelation=("lag1_noop_autocorrelation", "mean"),
            early_noop=("early_noop", "mean"),
            middle_noop=("middle_noop", "mean"),
            late_noop=("late_noop", "mean"),
        )
    )
    summary["late_minus_early_noop"] = summary["late_noop"] - summary["early_noop"]
    if out_path is None:
        return summary
    stages = list(GAMES)
    fig, axes = plt.subplots(2, 1, figsize=(19, 8.2), sharex=True)
    specifications = [
        ("lag1_noop_autocorrelation", "NOOP lag-1 autocorrelation", "viridis", None),
        ("late_minus_early_noop", "Late minus early NOOP proportion", "coolwarm", 0.0),
    ]
    for ax, (column, label, cmap, center) in zip(axes, specifications):
        matrix = (
            summary.pivot(index="subject", columns="game", values=column)
            .reindex(index=SUBJECTS, columns=stages)
        )
        values = np.ma.masked_invalid(matrix.to_numpy(dtype=float))
        if center is None:
            image = ax.imshow(values, aspect="auto", interpolation="nearest", cmap=cmap)
        else:
            limit = float(np.nanmax(np.abs(values))) if np.any(np.isfinite(values)) else 1.0
            image = ax.imshow(values, aspect="auto", interpolation="nearest", cmap=cmap, vmin=-limit, vmax=limit)
        ax.set_yticks(np.arange(len(SUBJECTS)))
        ax.set_yticklabels([f"S{subject_sort_key(subject)}" for subject in SUBJECTS])
        ax.set_ylabel("Participant")
        colorbar = fig.colorbar(image, ax=ax, fraction=0.02, pad=0.015)
        colorbar.set_label(label)
    axes[-1].set_xticks(np.arange(len(stages)))
    axes[-1].set_xticklabels([GAMES[stage] for stage in stages], rotation=45, ha="right")
    axes[-1].set_xlabel("Mario stage")
    fig.suptitle("Temporal organization of NOOP behavior across Mario stages", fontsize=15, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return summary


def _fit_subject_stage_hazards(bout_table: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    noop = bout_table[bout_table["bout_type"] == "NOOP"]
    for (subject, stage), group in noop.groupby(["subject", "game"], sort=True):
        lengths = group["length"].to_numpy(dtype=int)
        events = ~group["censored"].to_numpy(dtype=bool)
        null_fit = fit_constant_geometric(lengths, events)
        flexible_fit = fit_piecewise_hazard(lengths, events)
        smooth_fit = fit_discrete_weibull(lengths, events)
        lr = max(2.0 * (float(flexible_fit["log_likelihood"]) - null_fit["log_likelihood"]), 0.0)
        df = max(int(flexible_fit["n_bins"]) - 1, 1)
        rows.append(
            {
                "subject": subject,
                "game": int(stage),
                "stage": GAMES[int(stage)],
                "n_bouts": int(len(group)),
                "n_censored": int(np.sum(~events)),
                "constant_hazard": float(null_fit["hazard"]),
                "duration_beta": float(smooth_fit["beta"]),
                "lr_statistic": lr,
                "lr_degrees_freedom": df,
                "p_chi_square": float(stats.chi2.sf(lr, df=df)),
            }
        )
    summary = pd.DataFrame(rows).sort_values(["subject", "game"]).reset_index(drop=True)
    summary["q_bh_chi_square"] = benjamini_hochberg(summary["p_chi_square"])
    return summary


def _save_stage_survival_figure(
    bout_table: pd.DataFrame,
    model_summary: pd.DataFrame,
    envelopes: Dict[Tuple[str, int], Dict[str, np.ndarray]],
    out_path: Path,
) -> None:
    fig, axes = _stage_grid(len(GAMES))
    for ax, stage in zip(axes, GAMES):
        group = bout_table[(bout_table["game"] == stage) & (bout_table["bout_type"] == "NOOP")]
        fit_row = model_summary[model_summary["game"] == stage]
        if group.empty or fit_row.empty:
            ax.set_axis_off()
            continue
        fit = fit_row.iloc[0]
        envelope = envelopes[("pooled", stage)]
        times = envelope["times"]
        lengths = group["length"].to_numpy(dtype=int)
        events = ~group["censored"].to_numpy(dtype=bool)
        observed = kaplan_meier_survival_arrays(lengths, events, times)
        geometric = np.power(float(fit["geometric_q"]), times)
        piecewise = fit_piecewise_hazard(lengths, events)
        hazards = np.asarray(piecewise["hazard_by_time"], dtype=float)[1 : len(times) + 1]
        fitted_survival = np.cumprod(1.0 - hazards)
        ax.fill_between(times, envelope["low"], envelope["high"], color="#b7b7b7", alpha=0.3)
        ax.step(times, observed, where="post", color="#1f5a94", lw=1.5)
        ax.plot(times, geometric, color="#555555", lw=1.0, ls="--")
        ax.plot(times, fitted_survival, color="#d97706", lw=1.3)
        ax.set_xscale("log")
        ax.set_xlim(1, int(times[-1]))
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"{GAMES[stage]}  BH-FDR q {_format_probability(float(fit['q_bh_bootstrap']))}")
        ax.set_xlabel("Duration (steps)")
        ax.set_ylabel("Survival probability")
    handles = [
        Line2D([0], [0], color="#1f5a94", lw=1.5, label="Kaplan-Meier"),
        Line2D([0], [0], color="#555555", lw=1.0, ls="--", label="Constant hazard"),
        Line2D([0], [0], color="#d97706", lw=1.3, label="Duration-dependent hazard"),
        Line2D([0], [0], color="#b7b7b7", lw=7, alpha=0.3, label="95% null envelope"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("NOOP-bout survival by Mario stage", fontsize=15, y=1.025)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_stage_hazard_figure(
    bout_table: pd.DataFrame,
    model_summary: pd.DataFrame,
    out_path: Path,
) -> None:
    fig, axes = _stage_grid(len(GAMES))
    for ax, stage in zip(axes, GAMES):
        group = bout_table[(bout_table["game"] == stage) & (bout_table["bout_type"] == "NOOP")]
        fit_row = model_summary[model_summary["game"] == stage]
        if group.empty or fit_row.empty:
            ax.set_axis_off()
            continue
        fit = fit_row.iloc[0]
        lengths = group["length"].to_numpy(dtype=int)
        events = ~group["censored"].to_numpy(dtype=bool)
        t_max = min(max(2, int(np.percentile(lengths, 99.5))), 600)
        empirical = _binned_empirical_hazard(lengths, events, t_max)
        piecewise = fit_piecewise_hazard(lengths, events)
        ax.errorbar(
            empirical["time"], empirical["hazard"],
            yerr=np.vstack([empirical["hazard"] - empirical["ci_low"], empirical["ci_high"] - empirical["hazard"]]),
            fmt="o", ms=3, color="#1f5a94", ecolor="#7aa6c9", capsize=2,
        )
        ax.axhline(float(fit["constant_hazard"]), color="#555555", lw=1.0, ls="--")
        for left, right, hazard in zip(
            np.asarray(piecewise["edges"])[:-1], np.asarray(piecewise["edges"])[1:], np.asarray(piecewise["hazards"])
        ):
            if left > t_max:
                break
            ax.hlines(hazard, left, min(right, t_max), color="#d97706", lw=1.3)
        ax.set_xscale("log")
        ax.set_xlim(1, t_max)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(GAMES[stage])
        ax.set_xlabel("Duration (steps)")
        ax.set_ylabel("Exit hazard per step")
    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", color="#1f5a94", lw=0, label="Binned empirical hazard"),
            Line2D([0], [0], color="#555555", lw=1.0, ls="--", label="Constant-hazard null"),
            Line2D([0], [0], color="#d97706", lw=1.3, label="Duration-dependent fit"),
        ],
        loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.995),
    )
    fig.suptitle("Probability of ending a NOOP bout by Mario stage", fontsize=15, y=1.025)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_stage_mixture_figure(
    bout_table: pd.DataFrame,
    model_summary: pd.DataFrame,
    components: pd.DataFrame,
    out_path: Path,
) -> None:
    fig, axes = _stage_grid(len(GAMES))
    for ax, stage in zip(axes, GAMES):
        group = bout_table[(bout_table["game"] == stage) & (bout_table["bout_type"] == "NOOP")]
        fit_row = model_summary[model_summary["game"] == stage]
        component_rows = components[components["game"] == stage].sort_values("component")
        if group.empty or fit_row.empty or len(component_rows) != 2:
            ax.set_axis_off()
            continue
        fit = fit_row.iloc[0]
        observed = group.loc[~group["censored"], "length"].to_numpy(dtype=int)
        x_max = min(max(3, int(np.percentile(group["length"], 99.5))), 600)
        bins = np.unique(np.r_[1, np.round(np.geomspace(1, x_max + 1, 22)).astype(int), x_max + 1])
        counts, bins = np.histogram(observed, bins=bins)
        widths = np.diff(bins)
        mass = counts / max(np.sum(counts), 1)
        centers = np.sqrt(bins[:-1] * np.maximum(bins[1:] - 1, bins[:-1]))
        ax.bar(centers, mass, width=widths * 0.9, color="#b9c3cc", edgecolor="#66717b", lw=0.35, alpha=0.7)
        integer_times = np.arange(1, x_max + 1, dtype=int)
        one_pmf = _discrete_weibull_pmf(integer_times, float(fit["one_component_q"]), float(fit["one_component_beta"]))
        component_pmfs = [
            float(row["weight"]) * _discrete_weibull_pmf(integer_times, float(row["q"]), float(row["beta"]))
            for _, row in component_rows.iterrows()
        ]

        def bin_mass(pmf: np.ndarray) -> np.ndarray:
            return np.asarray([float(np.sum(pmf[left - 1 : right - 1])) for left, right in zip(bins[:-1], bins[1:])])

        ax.plot(centers, bin_mass(one_pmf), color="#555555", lw=1.0, ls="--")
        ax.plot(centers, bin_mass(component_pmfs[0] + component_pmfs[1]), color="#1f5a94", lw=1.5)
        ax.plot(centers, bin_mass(component_pmfs[0]), color="#d97706", lw=1.0)
        ax.plot(centers, bin_mass(component_pmfs[1]), color="#a23b72", lw=1.0)
        ax.set_xscale("log")
        ax.set_xlim(1, x_max)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(
            f"{GAMES[stage]}  delta BIC={fit['delta_bic_one_minus_two']:.0f}\n"
            f"medians={component_rows.iloc[0]['median_steps']:.1f}, {component_rows.iloc[1]['median_steps']:.1f}",
            fontsize=9,
        )
        ax.set_xlabel("Duration (steps)")
        ax.set_ylabel("Probability mass")
    fig.legend(
        handles=[
            Line2D([0], [0], color="#66717b", lw=7, alpha=0.7, label="Observed"),
            Line2D([0], [0], color="#555555", lw=1.0, ls="--", label="One component"),
            Line2D([0], [0], color="#1f5a94", lw=1.5, label="Two-component mixture"),
            Line2D([0], [0], color="#d97706", lw=1.0, label="Shorter component"),
            Line2D([0], [0], color="#a23b72", lw=1.0, label="Longer component"),
        ],
        loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.995),
    )
    fig.suptitle("Fitted NOOP-bout duration distributions by Mario stage", fontsize=15, y=1.025)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_stage_persistence_figure(curves: pd.DataFrame, out_path: Path) -> None:
    fig, axes = _stage_grid(len(GAMES))
    for ax, stage in zip(axes, GAMES):
        group = curves[curves["game"] == stage]
        if group.empty:
            ax.set_axis_off()
            continue
        ax.plot(group["time"], group["noop_persistence"], color="#555555", lw=1.2, ls="--")
        ax.plot(group["time"], group["active_persistence"], color="#1f5a94", lw=1.5)
        ax.plot(group["time"], group["cif_to_noop"], color="#d97706", lw=1.2)
        ax.plot(group["time"], group["cif_to_other_active"], color="#a23b72", lw=1.2, ls=":")
        ax.set_xscale("log")
        ax.set_xlim(1, int(group["time"].max()))
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(GAMES[stage])
        ax.set_xlabel("Same-action duration")
        ax.set_ylabel("Probability")
    fig.legend(
        handles=[
            Line2D([0], [0], color="#555555", lw=1.2, ls="--", label="NOOP persistence"),
            Line2D([0], [0], color="#1f5a94", lw=1.5, label="Same-active persistence"),
            Line2D([0], [0], color="#d97706", lw=1.2, label="Active to NOOP"),
            Line2D([0], [0], color="#a23b72", lw=1.2, ls=":", label="Active to other active"),
        ],
        loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.995),
    )
    fig.suptitle("Action persistence and competing exit routes by Mario stage", fontsize=15, y=1.025)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_mario_analysis(episodes: Sequence[Dict]) -> None:
    print("\n=== Mario stage-wise behavioral analysis ===")
    episode_table = pd.DataFrame(
        [
            {key: value for key, value in ep.items() if key not in ("actions", "rewards", "times")}
            for ep in episodes
        ]
    )
    episode_table.to_csv(OUT_DIR / "1-0_episode_index_by_stage.csv", index=False)
    save_withholding_schematic(FIG_DIR / "fig_1-1_A_withholding_schematic.png")
    action_detail = _save_mario_action_distribution(
        episodes, FIG_DIR / "fig_1-1_B_action_distribution_by_stage.png"
    )
    action_detail.to_csv(OUT_DIR / "1-1_action_distribution_episode_detail.csv", index=False)

    noop_summary = (
        episode_table.groupby(["subject", "game", "game_name"], as_index=False)
        .agg(mean_noop=("noop_prop", "mean"), n_episodes=("noop_prop", "size"))
    )
    noop_summary.to_csv(OUT_DIR / "1-2_noop_prevalence_by_subject_stage.csv", index=False)
    run_day_repeatability_analysis(episodes)
    print("Saved Figure 1-2 day-level NOOP repeatability outputs.")

    reward_summary = (
        episode_table.groupby(["subject", "game", "game_name"], as_index=False)
        .agg(mean_episode_return=("total_reward", "mean"), n_episodes=("total_reward", "size"))
    )
    reward_summary.to_csv(OUT_DIR / "1-3_episode_return_diagnostic_by_subject_stage.csv", index=False)

    reward_episode = build_reward_episode_summary(
        episodes, horizon_seconds=REWARD_HORIZON_SECONDS
    )
    reward_subject = summarize_reward_subject_conditions(reward_episode)
    reward_inference = summarize_reward_group_inference(reward_subject)
    reward_episode.to_csv(OUT_DIR / "1-3_postaction_reward_episode_1s.csv", index=False)
    reward_subject.to_csv(OUT_DIR / "1-3_postaction_reward_subject_condition_1s.csv", index=False)
    reward_inference.to_csv(OUT_DIR / "1-3_postaction_reward_condition_inference_1s.csv", index=False)
    save_reward_condition_comparison_figure(
        reward_subject,
        reward_inference,
        FIG_DIR / "fig_1-3_reward_by_stage.png",
        condition_axis_label="Mario stage",
    )
    save_reward_difference_figure(
        reward_subject,
        reward_inference,
        FIG_DIR / "fig_1-4_postnoop_reward_by_stage.png",
        condition_axis_label="Mario stage",
    )

    temporal_summary = _save_mario_temporal_structure(episodes, out_path=None)
    temporal_summary.to_csv(OUT_DIR / "1-5_temporal_structure_by_subject_stage.csv", index=False)
    print("Legacy temporal summaries retained as diagnostic CSV only.")

    bout_table = build_behavioral_bout_table(episodes)
    run_day_heldout_hazard_analysis(bout_table)
    print("Saved Figure 1-5 day-held-out hazard generalization outputs.")
    bout_table.to_csv(OUT_DIR / "1-6_behavioral_bouts_with_censoring.csv", index=False)
    subject_stage_hazards = _fit_subject_stage_hazards(bout_table)
    subject_stage_hazards.to_csv(OUT_DIR / "1-6_subject_stage_hazard_summary.csv", index=False)

    pooled_bouts = bout_table.copy()
    pooled_bouts["subject"] = "pooled"
    model_summary, components, envelopes = fit_duration_models(pooled_bouts)
    model_summary.to_csv(OUT_DIR / "1-6_stage_pooled_duration_hazard_model_summary.csv", index=False)
    components.to_csv(OUT_DIR / "1-7_stage_pooled_duration_mixture_components.csv", index=False)
    _save_stage_survival_figure(
        pooled_bouts, model_summary, envelopes, FIG_DIR / "fig_1-6_A_noop_survival_by_stage.png"
    )
    _save_stage_hazard_figure(
        pooled_bouts, model_summary, FIG_DIR / "fig_1-6_B_noop_duration_hazard_by_stage.png"
    )
    _save_stage_mixture_figure(
        pooled_bouts, model_summary, components, FIG_DIR / "fig_1-7_noop_duration_mixture_by_stage.png"
    )

    persistence_curves = build_persistence_curves(pooled_bouts)
    persistence_curves.to_csv(OUT_DIR / "1-8_stage_pooled_competing_risk_curves.csv", index=False)
    active = bout_table[bout_table["bout_type"] == "Active"].copy()
    active_summary = (
        active.groupby(["subject", "game", "game_name"], as_index=False)
        .agg(
            n_bouts=("length", "size"),
            median_length=("length", "median"),
            mean_length=("length", "mean"),
            censored_fraction=("censored", "mean"),
            to_noop_fraction=("event_type", lambda values: float(np.mean(np.asarray(values) == "to_noop"))),
            to_other_active_fraction=("event_type", lambda values: float(np.mean(np.asarray(values) == "to_other_active"))),
        )
    )
    active_summary.to_csv(OUT_DIR / "1-8_active_bout_by_subject_stage.csv", index=False)
    _save_stage_persistence_figure(
        persistence_curves, FIG_DIR / "fig_1-8_action_persistence_competing_risk_by_stage.png"
    )
    print(f"Mario outputs: {OUT_DIR}")


# ---------------------------------------------------------------------------
# Load all data
# ---------------------------------------------------------------------------

print("Loading data …")
all_episodes: List[Dict] = []
for sub in SUBJECTS:
    for game in GAMES:
        eps = load_subject_game(sub, game)
        all_episodes.extend(eps)
        condition_label = f"game_{game}" if DATASET == "atari" else GAMES[game]
        print(f"  {sub} {condition_label}: {len(eps)} episodes")

if not all_episodes:
    raise RuntimeError(f"No episodes found under {DATA_ROOT}")

if DATASET == "mario":
    run_mario_analysis(all_episodes)
    print("\n=== Done ===")
    print(f"Outputs: {OUT_DIR}")
    print(f"Figures: {FIG_DIR}")
    sys.exit(0)

thinker_episodes: List[Dict] = []
for game in GAMES:
    eps = load_thinker_game(game)
    thinker_episodes.extend(eps)
    print(f"  thinker game_{game}: {len(eps)} episodes")

df_ep = pd.DataFrame([
    {k: v for k, v in ep.items() if k not in ("actions", "rewards", "times")}
    for ep in all_episodes
])

# ---------------------------------------------------------------------------
# 1-1 : Withholding bout schematic
# ---------------------------------------------------------------------------
print("\n=== Section 1-1: Action distribution + withholding bout schematic ===")
action_dist_summary = summarize_episode_mean_action_distributions(
    human_episodes=all_episodes,
    thinker_episodes=thinker_episodes,
)
has_thinker_action_data = len(thinker_episodes) > 0
action_distribution_csv = (
    OUT_DIR / "1-1_action_distribution_subject_thinker.csv"
    if has_thinker_action_data
    else OUT_DIR / "1-1_action_distribution_human_only.csv"
)
action_distribution_figure = (
    FIG_DIR / "fig_1-1_B_action_distribution.png"
    if has_thinker_action_data
    else FIG_DIR / "fig_1-1_B_action_distribution_human_only.png"
)
for alternate_name in (
    "fig_1-1_B_action_distribution.png",
    "fig_1-1_B_action_distribution_human_only.png",
):
    alternate_path = FIG_DIR / alternate_name
    if alternate_path != action_distribution_figure:
        alternate_path.unlink(missing_ok=True)
action_dist_summary.to_csv(action_distribution_csv, index=False)
plot_action_distribution_subjects_vs_thinker(
    summary=action_dist_summary,
    out_path=action_distribution_figure,
)
print(f"Saved {action_distribution_csv.name}")
print(f"Saved {action_distribution_figure.name}")
save_withholding_schematic(FIG_DIR / "fig_1-1_A_withholding_schematic.png")
print("Saved fig_1-1_A_withholding_schematic.png")

# ---------------------------------------------------------------------------
# 1-3 : Fixed-time reward comparison
# ---------------------------------------------------------------------------
print("\n=== Section 1-3: Fixed-time post-action reward comparison ===")

episode_reward_summary = build_reward_episode_summary(
    all_episodes, horizon_seconds=REWARD_HORIZON_SECONDS
)
if len(episode_reward_summary) == 0:
    raise RuntimeError("No episode-level reward summary could be computed for fig_1-3_reward_subject_episode.png")
episode_reward_stats = summarize_reward_episode_subject_stats(episode_reward_summary)
episode_reward_subject = summarize_reward_subject_conditions(episode_reward_summary)
episode_reward_inference = summarize_reward_group_inference(episode_reward_subject)
episode_reward_summary.to_csv(OUT_DIR / "1-3_postaction_reward_episode_1s.csv", index=False)
episode_reward_subject.to_csv(OUT_DIR / "1-3_postaction_reward_subject_condition_1s.csv", index=False)
episode_reward_inference.to_csv(OUT_DIR / "1-3_postaction_reward_condition_inference_1s.csv", index=False)
save_reward_condition_comparison_figure(
    episode_reward_subject,
    episode_reward_inference,
    FIG_DIR / "fig_1-3_reward_subject_episode.png",
    condition_axis_label="Atari game",
)
print("Saved fig_1-3_reward_subject_episode.png")

# ---------------------------------------------------------------------------
# 1-2 : Subject-level NOOP analysis
# ---------------------------------------------------------------------------
print("\n=== Section 1-2: NOOP prevalence and across-day repeatability ===")

# --- 1-2a : Subject × Game NOOP proportion ---
subj_game = (
    df_ep.groupby(["subject", "game_name"])["noop_prop"]
    .agg(["mean", "std", "count"])
    .reset_index()
    .rename(columns={"mean": "noop_mean", "std": "noop_std", "count": "n_episodes"})
)
subj_game.to_csv(OUT_DIR / "1-2_subject_game_noop.csv", index=False)
print(subj_game.to_string(index=False))

# --- 1-2b : Step-weighted day rates and balanced across-subject day ICC ---
noop_day_level, noop_day_repeatability = run_day_repeatability_analysis(all_episodes)
print("\nBalanced common-day repeatability (subjects=targets, days=repeats):")
print(noop_day_repeatability.to_string(index=False))

# --- 1-2c : NOOP ratio ↔ performance correlation ---
corr_rows = []
for game, gname in GAMES.items():
    sub_game_df = df_ep[df_ep["game"] == game]
    # Subject-level: mean NOOP vs mean score
    subj_agg = sub_game_df.groupby("subject").agg(
        mean_noop=("noop_prop", "mean"),
        mean_score=("total_reward", "mean"),
    ).reset_index()
    r, p = stats.pearsonr(subj_agg["mean_noop"], subj_agg["mean_score"])
    corr_rows.append({"game": gname, "level": "subject", "r": r, "p": p, "n": len(subj_agg)})

    # Session-level: day mean NOOP vs day mean score (within subject, so partial correlation)
    sess_agg = sub_game_df.groupby(["subject", "day"]).agg(
        mean_noop=("noop_prop", "mean"),
        mean_score=("total_reward", "mean"),
    ).reset_index()
    r2, p2 = stats.pearsonr(sess_agg["mean_noop"], sess_agg["mean_score"])
    corr_rows.append({"game": gname, "level": "session", "r": r2, "p": p2, "n": len(sess_agg)})

df_corr = pd.DataFrame(corr_rows)
df_corr.to_csv(OUT_DIR / "1-2_noop_performance_corr.csv", index=False)
print("\nNOOP ↔ performance correlations:")
print(df_corr.to_string(index=False))

# --- 1-2d : Cohen's d vs chance (1/N_ACTIONS) ---
cohend_rows = []
for game, gname in GAMES.items():
    sub_means = df_ep[df_ep["game"] == game].groupby("subject")["noop_prop"].mean().values
    d = (sub_means.mean() - CHANCE_NOOP) / (sub_means.std(ddof=1) + EPS)
    cohend_rows.append({"game": gname, "mean_noop": sub_means.mean(),
                        "chance": CHANCE_NOOP, "cohens_d": d})

df_cohend = pd.DataFrame(cohend_rows)
df_cohend.to_csv(OUT_DIR / "1-2_cohens_d_vs_chance.csv", index=False)
print("\nCohen's d vs chance NOOP:")
print(df_cohend.to_string(index=False))

# Shared cross-game summaries reused in fig 1-2C and the section 1-3 tables.
pong_noop = df_ep[df_ep["game"] == 1].groupby("subject")["noop_prop"].mean()
si_noop = df_ep[df_ep["game"] == 2].groupby("subject")["noop_prop"].mean()
t_stat, p_paired = stats.ttest_rel(pong_noop.values, si_noop.values)

direction_rows = []
for sub in SUBJECTS:
    pv = float(df_ep[(df_ep["subject"] == sub) & (df_ep["game"] == 1)]["noop_prop"].mean())
    sv = float(df_ep[(df_ep["subject"] == sub) & (df_ep["game"] == 2)]["noop_prop"].mean())
    direction_rows.append(
        {
            "subject": sub,
            "pong_noop": pv,
            "si_noop": sv,
            "pong_above_chance": pv > CHANCE_NOOP,
            "si_above_chance": sv > CHANCE_NOOP,
        }
    )
df_dir = pd.DataFrame(direction_rows)

# --- Figure 1-2 ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel A: subject-level NOOP proportion per game
ax = axes[0]
x = np.arange(len(SUBJECTS))
offset = 0.16
for idx, sub in enumerate(SUBJECTS):
    subject_color = SUBJECT_COLOR_MAP[sub]
    p_val = float(pong_noop.get(sub, np.nan))
    s_val = float(si_noop.get(sub, np.nan))
    if np.isfinite(p_val) and np.isfinite(s_val):
        ax.plot(
            [x[idx] - offset, x[idx] + offset],
            [p_val, s_val],
            color=subject_color,
            lw=1.2,
            alpha=0.45,
            zorder=1,
        )
    if np.isfinite(p_val):
        ax.scatter(
            x[idx] - offset,
            p_val,
            s=62,
            marker="o",
            facecolor=subject_color,
            zorder=3,
        )
    if np.isfinite(s_val):
        ax.scatter(
            x[idx] + offset,
            s_val,
            s=62,
            marker="s",
            facecolor=subject_color,
            zorder=3,
        )
ax.axhline(CHANCE_NOOP, color="gray", ls="--", lw=1.2)
ax.set_xticks(x)
ax.set_xticklabels([f"S{i+1}" for i in range(len(SUBJECTS))])
ax.set_ylabel("Mean NOOP proportion")
ax.set_title("1-2A: Subject-level NOOP proportion")
ax.legend(
    handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor=GAME_COLORS[1], markeredgecolor="none", markersize=7, label="Pong"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=GAME_COLORS[2], markeredgecolor="none", markersize=7, label="SpaceInvaders"),
        Line2D([0], [0], color="gray", lw=1.2, ls="--", label="Chance (1/6)"),
    ],
    fontsize=8,
    frameon=False,
)
ax.grid(axis="y", alpha=0.3)

# Panel B: NOOP ratio vs performance scatter (dual y-axis: SI left, Pong right)
ax_si = axes[1]
ax_pong = ax_si.twinx()

for game, gname in GAMES.items():
    sub_agg = df_ep[df_ep["game"] == game].groupby("subject").agg(
        mean_noop=("noop_prop", "mean"),
        mean_score=("total_reward", "mean"),
    ).reset_index()
    sub_agg["sub_label"] = sub_agg["subject"].str.extract(r"(\d+)").astype(int).apply(lambda x: f"S{x[0]}", axis=1)

    if game == 2:  # SpaceInvaders → left axis
        for _, row in sub_agg.iterrows():
            subject_color = SUBJECT_COLOR_MAP[str(row["subject"])]
            ax_si.scatter(
                row["mean_noop"],
                row["mean_score"],
                s=68,
                marker="o",
                facecolor=subject_color,
                zorder=3,
            )
            ax_si.annotate(
                row["sub_label"],
                (row["mean_noop"], row["mean_score"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
                color=subject_color,
            )
        coef = np.polyfit(sub_agg["mean_noop"], sub_agg["mean_score"], deg=1)
        xx = np.linspace(sub_agg["mean_noop"].min(), sub_agg["mean_noop"].max(), 50)
        ax_si.plot(xx, np.polyval(coef, xx), color=GAME_COLORS[2], lw=1.5, ls="--", alpha=0.8)
    else:  # Pong → right axis
        for _, row in sub_agg.iterrows():
            subject_color = SUBJECT_COLOR_MAP[str(row["subject"])]
            ax_pong.scatter(
                row["mean_noop"],
                row["mean_score"],
                s=68,
                marker="s",
                facecolor=subject_color,
                zorder=3,
            )
            ax_pong.annotate(
                row["sub_label"],
                (row["mean_noop"], row["mean_score"]),
                textcoords="offset points",
                xytext=(4, -10),
                fontsize=7,
                color=subject_color,
            )
        coef = np.polyfit(sub_agg["mean_noop"], sub_agg["mean_score"], deg=1)
        xx = np.linspace(sub_agg["mean_noop"].min(), sub_agg["mean_noop"].max(), 50)
        ax_pong.plot(xx, np.polyval(coef, xx), color=GAME_COLORS[1], lw=1.5, ls="--", alpha=0.8)

ax_si.set_xlabel("Mean NOOP proportion")
ax_si.set_ylabel("Mean episode score (SpaceInvaders)", color=GAME_COLORS[2])
ax_pong.set_ylabel("Mean episode score (Pong)", color=GAME_COLORS[1])
ax_si.tick_params(axis="y", labelcolor=GAME_COLORS[2])
ax_pong.tick_params(axis="y", labelcolor=GAME_COLORS[1])
ax_si.set_title("1-2B: NOOP ratio ↔ performance")
ax_si.legend(
    handles=[
        Line2D([0], [0], marker="o", color=GAME_COLORS[2], markerfacecolor=GAME_COLORS[2], markeredgecolor="none", markersize=7, lw=1.5, ls="--", label="SpaceInvaders"),
        Line2D([0], [0], marker="s", color=GAME_COLORS[1], markerfacecolor=GAME_COLORS[1], markeredgecolor="none", markersize=7, lw=1.5, ls="--", label="Pong"),
    ],
    fontsize=8,
    frameon=False,
    loc="upper right",
)
ax_si.grid(alpha=0.3)

# Panel C: per-subject NOOP above chance in both games (meta-analytic)
ax = axes[2]
all_above_p = (df_dir["pong_noop"] - CHANCE_NOOP).to_numpy(dtype=float)
all_above_s = (df_dir["si_noop"] - CHANCE_NOOP).to_numpy(dtype=float)
pad_x = max(float(np.ptp(all_above_p)) * 0.35, 0.02)
pad_y = max(float(np.ptp(all_above_s)) * 0.35, 0.02)
for _, row in df_dir.iterrows():
    above_p = float(row["pong_noop"] - CHANCE_NOOP)
    above_s = float(row["si_noop"] - CHANCE_NOOP)
    subject = str(row["subject"])
    subject_color = SUBJECT_COLOR_MAP[subject]
    ax.scatter(above_p, above_s, color=subject_color, s=68, zorder=3)
    ax.annotate(
        f"S{subject_sort_key(subject)}",
        (above_p, above_s),
        textcoords="offset points",
        xytext=(5, 4),
        fontsize=8,
        color=subject_color,
    )
ax.axhline(0, color="black", lw=0.8, ls="--")
ax.axvline(0, color="black", lw=0.8, ls="--")
ax.set_xlim(float(np.min(all_above_p)) - pad_x, float(np.max(all_above_p)) + pad_x)
ax.set_ylim(float(np.min(all_above_s)) - pad_y, float(np.max(all_above_s)) + pad_y)
ax.set_xlabel("Pong NOOP above chance", color=GAME_COLORS[1])
ax.set_ylabel("SpaceInvaders NOOP above chance", color=GAME_COLORS[2])
ax.set_title("1-2C: Meta-analytic direction check")
ax.grid(alpha=0.3)

fig.tight_layout()
plt.close(fig)
print(
    "\nSaved fig_1-2_day_level_noop_repeatability.png; "
    "the legacy three-panel draft remains unsaved."
)


# ---------------------------------------------------------------------------
# 1-2 : Supporting cross-game descriptive summaries
# ---------------------------------------------------------------------------
print("\n=== Section 1-2: Cross-game descriptive summaries ===")

print(f"Pong mean NOOP: {pong_noop.mean():.3f}  SI mean NOOP: {si_noop.mean():.3f}")
print(f"Paired t-test: t={t_stat:.3f}, p={p_paired:.4f}")

# --- Bout length distributions per game ---
bout_rows = []
for ep in all_episodes:
    bouts = compute_bouts(ep["actions"])
    for b in bouts:
        bout_rows.append({
            "subject": ep["subject"],
            "game": ep["game"],
            "game_name": ep["game_name"],
            "day": ep["day"],
            "bout_length": b,
        })

df_bouts = pd.DataFrame(bout_rows)
df_bouts.to_csv(OUT_DIR / "1-2_bout_lengths.csv", index=False)

bout_stats = df_bouts.groupby("game_name")["bout_length"].describe()
print("\nBout length statistics:")
print(bout_stats.to_string())

df_dir.to_csv(OUT_DIR / "1-2_effect_direction.csv", index=False)
print("\nNOOP above chance per subject:")
print(df_dir.to_string(index=False))
print("\nCross-game direction checks are retained as tables; no separate 1-2/1-3 panel is emitted.")


# ---------------------------------------------------------------------------
# 1-5 : Alternative explanation diagnostics
# ---------------------------------------------------------------------------
print("\n=== Section 1-5: Alternative explanation diagnostics ===")

# --- 1-4 Q1: Episode-position NOOP density ---
# For each step in each episode, record normalized position (0-1) and is_noop

pos_rows = []
for ep in all_episodes:
    n = ep["n_steps"]
    if n < 10:
        continue
    norm_pos = np.arange(n) / (n - 1)
    is_noop = (ep["actions"] == NOOP_ACTION).astype(int)
    for pos, noop in zip(norm_pos, is_noop):
        pos_rows.append({
            "subject": ep["subject"],
            "game": ep["game"],
            "game_name": ep["game_name"],
            "norm_pos": float(pos),
            "is_noop": int(noop),
        })

df_pos = pd.DataFrame(pos_rows)

# Compute mean NOOP probability per position bin (20 bins) per game
n_bins = 20
bin_edges = np.linspace(0, 1, n_bins + 1)
df_pos["pos_bin"] = pd.cut(df_pos["norm_pos"], bins=bin_edges, labels=False, include_lowest=True)

pos_summary = df_pos.groupby(["game_name", "pos_bin"])["is_noop"].mean().reset_index()
pos_summary["pos_bin_center"] = (pos_summary["pos_bin"] + 0.5) / n_bins

# Kruskal-Wallis test: does NOOP probability differ across thirds of episode?
thirds_rows = []
for game, gname in GAMES.items():
    g_data = df_pos[df_pos["game"] == game].copy()
    g_data["third"] = pd.cut(g_data["norm_pos"], bins=[0, 1/3, 2/3, 1],
                              labels=["early", "mid", "late"], include_lowest=True)
    means_by_third = g_data.groupby("third", observed=False)["is_noop"].mean()
    early = g_data[g_data["third"] == "early"]["is_noop"].values
    late = g_data[g_data["third"] == "late"]["is_noop"].values
    stat, p = stats.mannwhitneyu(early, late, alternative="two-sided")
    thirds_rows.append({
        "game": gname,
        "early_noop": means_by_third.get("early", np.nan),
        "mid_noop": means_by_third.get("mid", np.nan),
        "late_noop": means_by_third.get("late", np.nan),
        "early_vs_late_U": stat,
        "early_vs_late_p": p,
    })
    print(f"{gname}: early={means_by_third.get('early',np.nan):.3f}  "
          f"mid={means_by_third.get('mid',np.nan):.3f}  "
          f"late={means_by_third.get('late',np.nan):.3f}  "
          f"(early vs late U-test p={p:.4f})")

df_thirds = pd.DataFrame(thirds_rows)
df_thirds.to_csv(OUT_DIR / "1-5_episode_position_thirds.csv", index=False)

# --- 1-4 Q2: NOOP autocorrelation (lag-1 AR coefficient) ---
autocorr_rows = []
for ep in all_episodes:
    ac = compute_lag1_autocorr(ep["actions"])
    if not np.isnan(ac):
        autocorr_rows.append({
            "subject": ep["subject"],
            "game": ep["game"],
            "game_name": ep["game_name"],
            "day": ep["day"],
            "lag1_autocorr": ac,
            "n_steps": ep["n_steps"],
        })

df_ac = pd.DataFrame(autocorr_rows)
df_ac.to_csv(OUT_DIR / "1-5_noop_autocorrelation.csv", index=False)

ac_summary = df_ac.groupby(["subject", "game_name"])["lag1_autocorr"].mean().reset_index()
print("\nMean lag-1 NOOP autocorrelation by subject × game:")
print(ac_summary.pivot(index="subject", columns="game_name", values="lag1_autocorr").to_string())

for game, gname in GAMES.items():
    ac_vals = df_ac[df_ac["game"] == game]["lag1_autocorr"].values
    t, p = stats.ttest_1samp(ac_vals, popmean=0)
    print(f"{gname}: mean AC = {ac_vals.mean():.3f} ± {ac_vals.std():.3f}  "
          f"(t={t:.2f}, p={p:.4f}, N={len(ac_vals)})")

# --- Figure 1-5 ---
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5))

# Panel A: episode-position NOOP density (continuous + 3 thirds)
ax = axes[0]
for game, gname in GAMES.items():
    sub_data = pos_summary[pos_summary["game_name"] == gname]
    ax.plot(
        sub_data["pos_bin_center"],
        sub_data["is_noop"],
        lw=2,
        label=gname,
        color=GAME_COLORS[game],
    )
ax.axhline(CHANCE_NOOP, color="gray", ls="--", lw=1.2, label="Chance (1/6)")
# Shade thirds
for x_start, label in [(0, "early"), (1/3, "mid"), (2/3, "late")]:
    ax.axvspan(x_start, x_start + 1/3,
               alpha=0.06 if label == "early" else (0.12 if label == "mid" else 0.06),
               color="gray")
    ax.text(x_start + 1/6, 0.01, label, ha="center", fontsize=8, color="gray")
ax.set_xlabel("Normalized episode position (0=start, 1=end)")
ax.set_ylabel("NOOP probability")
ax.set_title("1-5A: Episode-position NOOP density\n(uniform/state-dep -> not fatigue)")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.3)

# Panel B: lag-1 autocorrelation distribution per game
ax = axes[1]
rng = np.random.default_rng(0)
for game, gname in GAMES.items():
    ac_vals = df_ac[df_ac["game"] == game]["lag1_autocorr"].values
    x_pos = 0 if game == 1 else 1
    ax.boxplot(
        ac_vals,
        positions=[x_pos],
        widths=0.3,
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor=GAME_COLORS[game], edgecolor=GAME_COLORS[game], alpha=0.35),
        medianprops=dict(color=GAME_COLORS[game], linewidth=1.4),
        whiskerprops=dict(color=GAME_COLORS[game]),
        capprops=dict(color=GAME_COLORS[game]),
    )
    subject_means = (
        df_ac[df_ac["game"] == game]
        .groupby("subject")["lag1_autocorr"]
        .mean()
        .sort_index()
    )
    for subject, mean_ac in subject_means.items():
        ax.scatter(
            x_pos + rng.uniform(-0.06, 0.06),
            mean_ac,
            color=SUBJECT_COLOR_MAP[str(subject)],
            s=58,
            zorder=3,
        )
ax.axhline(0, color="black", lw=0.8, ls="--")
ax.set_xticks([0, 1], ["Pong", "SpaceInvaders"])
ax.set_ylabel("Lag-1 autocorrelation")
ax.set_title("1-5B: NOOP lag-1 autocorrelation\n(high AC -> perseveration)")
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
plt.close(fig)
print("\nLegacy position/autocorrelation summaries retained as diagnostic CSV only.")

behavioral_bout_table = build_behavioral_bout_table(all_episodes)
run_day_heldout_hazard_analysis(behavioral_bout_table)
print("Saved fig_1-5_day_heldout_hazard_generalization.png")
print("Saved 1-5_day_heldout_hazard fold, subject, condition, and common-effect CSVs")


# ---------------------------------------------------------------------------
# 1-4 : Subject-level fixed-time reward difference
# ---------------------------------------------------------------------------
print("\n=== Figure 1-4: Subject-level withholding reward difference ===")
out_path = FIG_DIR / "fig_1-4_postnoop_reward_by_game.png"
save_reward_difference_figure(
    episode_reward_subject,
    episode_reward_inference,
    out_path,
    condition_axis_label="Atari game",
)
print(f"Saved {out_path.name}")


# ---------------------------------------------------------------------------
# 1-6 : Censoring-aware NOOP duration and hazard analysis
# ---------------------------------------------------------------------------
print("\n=== Figure 1-6: Censoring-aware NOOP duration and hazard analysis ===")
behavioral_bout_table.to_csv(OUT_DIR / "1-6_behavioral_bouts_with_censoring.csv", index=False)
duration_model_summary, duration_components, survival_envelopes = fit_duration_models(
    behavioral_bout_table
)
duration_model_summary.to_csv(OUT_DIR / "1-6_noop_duration_hazard_model_summary.csv", index=False)
save_noop_survival_figure(
    behavioral_bout_table,
    duration_model_summary,
    survival_envelopes,
    FIG_DIR / "fig_1-6_A_noop_survival_model_comparison.png",
)
save_noop_hazard_figure(
    behavioral_bout_table,
    duration_model_summary,
    FIG_DIR / "fig_1-6_B_noop_duration_hazard.png",
)
print("Saved 1-6_behavioral_bouts_with_censoring.csv")
print("Saved 1-6_noop_duration_hazard_model_summary.csv")
print("Saved fig_1-6_A_noop_survival_model_comparison.png")
print("Saved fig_1-6_B_noop_duration_hazard.png")


# ---------------------------------------------------------------------------
# 1-7 : Fitted NOOP-bout duration distributions (no fixed cross point)
# ---------------------------------------------------------------------------
print("\n=== Figure 1-7: Fitted NOOP-bout duration distributions ===")
duration_components.to_csv(OUT_DIR / "1-7_noop_duration_mixture_components.csv", index=False)
save_duration_mixture_figure(
    behavioral_bout_table,
    duration_model_summary,
    duration_components,
    FIG_DIR / "fig_1-7_noop_duration_mixture.png",
)
print("Saved 1-7_noop_duration_mixture_components.csv")
print("Saved fig_1-7_noop_duration_mixture.png")


# ---------------------------------------------------------------------------
# 1-8 : Active-action persistence and competing exit routes
# ---------------------------------------------------------------------------
print("\n=== Figure 1-8: Active-action persistence and competing exit routes ===")
persistence_curves = build_persistence_curves(behavioral_bout_table)
persistence_curves.to_csv(OUT_DIR / "1-8_action_persistence_competing_risk_curves.csv", index=False)
active_summary = (
    behavioral_bout_table[behavioral_bout_table["bout_type"] == "Active"]
    .groupby(["subject", "game", "game_name", "action", "action_label"], as_index=False)
    .agg(
        n_bouts=("length", "size"),
        median_length=("length", "median"),
        mean_length=("length", "mean"),
        episode_end_fraction=("censored", "mean"),
        to_noop_fraction=("event_type", lambda values: float(np.mean(np.asarray(values) == "to_noop"))),
        to_other_active_fraction=("event_type", lambda values: float(np.mean(np.asarray(values) == "to_other_active"))),
    )
)
active_summary.to_csv(OUT_DIR / "1-8_active_bout_action_summary.csv", index=False)
save_persistence_competing_risk_figure(
    persistence_curves,
    FIG_DIR / "fig_1-8_action_persistence_competing_risk.png",
)
print("Saved 1-8_action_persistence_competing_risk_curves.csv")
print("Saved 1-8_active_bout_action_summary.csv")
print("Saved fig_1-8_action_persistence_competing_risk.png")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n=== Done ===")
print(f"Outputs: {OUT_DIR}")
print(f"Figures: {FIG_DIR}")
