#!/usr/bin/env python3
"""
Behavioral analysis for the figure set in research plan 01.

Figures generated here:
1-1 : Withholding bout schematic + action distribution
1-2 : Individual differences
1-3 : Reward comparison by subject and episode
1-4 : NOOP ratio vs post-NOOP reward benefit
1-5 : Alternative explanation exclusion
1-6 : NOOP bout survival analysis
1-7 : Short vs Long bout distribution from behavior-only data

Input  : ../behavioral_data_block_old/sub_{1-6}/game_{1,2}/day_{1-4}/block_N/*.npz
Output : ../research_script/outputs/01_behavioral_analysis/
"""
from __future__ import annotations

import glob
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
DATA_ROOT = ROOT / "behavioral_data_block_old"
THINKER_ROOT = DATA_ROOT / "thinker"
OUT_DIR = Path(__file__).parent / "outputs" / "01_behavioral_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SUBJECTS = [f"sub_{i}" for i in range(1, 7)]
GAMES = {1: "Pong", 2: "SpaceInvaders"}
GAME_TITLES = {1: "Pong", 2: "Space Invaders"}
NOOP_ACTION = 0
N_ACTIONS = 6
ACTION_MAPPING = {
    "NOOP": 0,
    "FIRE": 1,
    "RIGHT": 2,
    "LEFT": 3,
    "RIGHTFIRE": 4,
    "LEFTFIRE": 5,
}
ACTION_LABELS = [label for label, _ in sorted(ACTION_MAPPING.items(), key=lambda item: item[1])]
CHANCE_NOOP = 1.0 / N_ACTIONS  # ~0.167
REWARD_K = 15

EPS = 1e-12
WITH_COLOR = "#2ca02c"
NON_COLOR = "#d62728"
LINE_COLOR = "#7f7f7f"
MEAN_COLOR = "#111111"
RIGHT_AXIS_COLOR = NON_COLOR
RIGHT_AXIS_GAMES = {2}
GAME_COLORS = {1: WITH_COLOR, 2: NON_COLOR}
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
    k_future: int = REWARD_K,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for ep in episodes:
        actions = np.asarray(ep["actions"], dtype=int).reshape(-1)
        rewards = np.asarray(ep["rewards"], dtype=float).reshape(-1)
        n = min(len(actions), len(rewards))
        actions = actions[:n]
        rewards = rewards[:n]
        if n < 2:
            continue

        prev_actions = np.full(n, -1, dtype=int)
        prev_actions[1:] = actions[:-1]
        with_rewards: List[float] = []
        non_rewards: List[float] = []

        for step_index in np.where(actions != NOOP_ACTION)[0]:
            k_reward = float(np.nansum(rewards[step_index + 1 : step_index + 1 + k_future]))
            if prev_actions[step_index] == NOOP_ACTION:
                with_rewards.append(k_reward)
            else:
                non_rewards.append(k_reward)

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
                "n_withholding_preceded": int(len(with_rewards)),
                "n_not_preceded": int(len(non_rewards)),
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
        "n_withholding_preceded",
        "n_not_preceded",
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
    k_future: int,
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
    k_future: int = REWARD_K,
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
            k_future=k_future,
        )

    for ax in axes_flat[len(subjects) :]:
        ax.set_axis_off()

    for ax in axes[:, 0]:
        ax.set_ylabel(f"Episode mean k={k_future} reward\n(Pong scale)")

    for ax_right in right_axes:
        ax_right.set_ylabel(
            f"Episode mean k={k_future} reward\n(SpaceInvaders scale)",
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

def load_subject_game(subject: str, game: int) -> List[Dict]:
    """Return a list of episode dicts for one subject × game.

    Each episode dict has:
        subject, game, day, block, ep_idx (within block), actions (1-D int array),
        rewards (1-D), n_steps, total_reward, noop_prop
    """
    pattern = str(DATA_ROOT / subject / f"game_{game}" / "*" / "*" / "*.npz")
    files = sorted(glob.glob(pattern))
    episodes = []
    for fpath in files:
        # Parse day/block from path  e.g. .../day_2/block_3/...npz
        parts = Path(fpath).parts
        day_str = next(p for p in parts if p.startswith("day_"))
        block_str = next(p for p in parts if p.startswith("block_"))
        day = int(day_str.split("_")[1])
        block = int(block_str.split("_")[1])

        data = np.load(fpath, allow_pickle=True)
        action_oh = np.asarray(data["action"], dtype=float)
        actions = np.argmax(action_oh, axis=1).astype(int)
        rewards = np.asarray(data["reward"], dtype=int)
        is_terminal = np.asarray(data["is_terminal"], dtype=bool)

        # Split into episodes at is_terminal boundaries
        ep_starts = [0]
        term_pos = np.where(is_terminal)[0].tolist()
        for tp in term_pos:
            if tp + 1 < len(actions):
                ep_starts.append(tp + 1)
        ep_ends = term_pos + [len(actions) - 1]

        for ep_idx, (s, e) in enumerate(zip(ep_starts, ep_ends)):
            ep_actions = actions[s : e + 1]
            ep_rewards = rewards[s : e + 1]
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
                    "ep_idx": ep_idx,
                    "actions": ep_actions,
                    "rewards": ep_rewards,
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
        raise FileNotFoundError(f"No thinker episode .npy files found for game_{game} under {game_dir}")

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
    fig.suptitle("1-1B: Action distribution by subject and non-IL Thinker", y=1.08, fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def compute_bouts(actions: np.ndarray) -> List[int]:
    """Return list of NOOP bout lengths for one episode."""
    bouts = []
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


def _km_survival_from_lengths(lengths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lengths = np.sort(np.asarray(lengths, dtype=int))
    if lengths.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    max_t = int(lengths.max())
    x = np.arange(1, max_t + 1)
    survival = np.array([(lengths >= t).mean() for t in x], dtype=float)
    return x, survival


def estimate_cross2_from_lengths(lengths: Sequence[int]) -> int:
    """Estimate Cross2 from a bout-length survival curve and exponential null."""
    arr = np.asarray(lengths, dtype=int)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1
    if arr.size < 5:
        return max(int(np.percentile(arr, 95)), 1)

    lam = 1.0 / max(float(np.mean(arr)), EPS)
    x, km = _km_survival_from_lengths(arr)
    exp_base = np.exp(-lam * x)
    km_above = km > exp_base

    if km_above[0]:
        cross1_candidates = np.where(~km_above)[0]
        cross1_idx = int(cross1_candidates[0]) if cross1_candidates.size > 0 else len(km) - 1
    else:
        cross1_idx = 0

    search_start = cross1_idx + 1
    if search_start >= len(km):
        return max(int(np.percentile(arr, 95)), 1)

    cross2_candidates = np.where(km_above[search_start:])[0]
    if cross2_candidates.size == 0:
        return max(int(np.percentile(arr, 95)), 1)
    return int(x[search_start + cross2_candidates[0]])


def annotate_short_long_bouts(df_bouts: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Add subject × game Cross2 split labels to behavior-only bout lengths."""
    if df_bouts.empty:
        detail_cols = list(df_bouts.columns) + ["cross2", "bout_class"]
        summary_cols = [
            "subject",
            "game",
            "game_name",
            "n_bouts",
            "cross2",
            "n_short",
            "n_long",
            "pct_long",
        ]
        return pd.DataFrame(columns=detail_cols), pd.DataFrame(columns=summary_cols)

    detail = df_bouts.copy()
    cross2_map: Dict[Tuple[str, int], int] = {}
    summary_rows: List[Dict[str, object]] = []

    for (subject, game), group in detail.groupby(["subject", "game"], sort=True):
        lengths = group["bout_length"].to_numpy(dtype=int)
        cross2 = estimate_cross2_from_lengths(lengths)
        cross2_map[(str(subject), int(game))] = cross2
        is_long = lengths >= cross2
        summary_rows.append(
            {
                "subject": subject,
                "game": int(game),
                "game_name": str(group["game_name"].iloc[0]),
                "n_bouts": int(len(lengths)),
                "cross2": int(cross2),
                "n_short": int(np.sum(~is_long)),
                "n_long": int(np.sum(is_long)),
                "pct_long": float(np.mean(is_long)) if len(is_long) > 0 else np.nan,
            }
        )

    detail["cross2"] = [
        cross2_map[(str(row.subject), int(row.game))]
        for row in detail.itertuples(index=False)
    ]
    detail["bout_class"] = np.where(
        detail["bout_length"].to_numpy(dtype=int) >= detail["cross2"].to_numpy(dtype=int),
        "Long",
        "Short",
    )
    summary = pd.DataFrame(summary_rows).sort_values(["subject", "game"]).reset_index(drop=True)
    return detail, summary


def save_short_long_bout_distribution_figure(
    bout_detail: pd.DataFrame,
    bout_summary: pd.DataFrame,
    out_path: Path,
) -> None:
    """Save behavior-only Fig 1-7 for all subjects and both games."""
    fig, axes = plt.subplots(len(GAMES), len(SUBJECTS), figsize=(18, 7.2), sharey="row")

    game_max = {
        game: max(
            int(bout_detail.loc[bout_detail["game"] == game, "bout_length"].max())
            if np.any(bout_detail["game"] == game)
            else 2,
            2,
        )
        for game in GAMES
    }

    for row_idx, (game, game_name) in enumerate(GAMES.items()):
        max_len = game_max[game]
        bins = np.unique(np.round(np.logspace(0, np.log10(max_len), 28)).astype(int))
        bins = np.r_[bins, max_len + 1]
        bins = np.unique(bins)

        for col_idx, subject in enumerate(SUBJECTS):
            ax = axes[row_idx, col_idx]
            sub = bout_detail[
                (bout_detail["subject"] == subject) &
                (bout_detail["game"] == game)
            ]
            summary_row = bout_summary[
                (bout_summary["subject"] == subject) &
                (bout_summary["game"] == game)
            ]

            if sub.empty or summary_row.empty:
                ax.set_axis_off()
                continue

            cross2 = int(summary_row.iloc[0]["cross2"])
            short_lengths = sub.loc[sub["bout_class"] == "Short", "bout_length"].to_numpy(dtype=int)
            long_lengths = sub.loc[sub["bout_class"] == "Long", "bout_length"].to_numpy(dtype=int)

            ax.hist(
                short_lengths,
                bins=bins,
                color="#1f77b4",
                alpha=0.72,
                label="Short",
            )
            ax.hist(
                long_lengths,
                bins=bins,
                color="#ff7f0e",
                alpha=0.72,
                label="Long",
            )
            ax.axvline(cross2, color="#222222", lw=1.0, ls=":", alpha=0.8)
            ax.set_xscale("log")
            ax.set_xlim(1, max_len)
            ax.grid(axis="y", alpha=0.25)
            ax.set_title(
                f"S{subject_sort_key(subject)} {game_name}\n"
                f"Cross2={cross2}, long={int(summary_row.iloc[0]['n_long'])}",
                fontsize=9,
                color=GAME_COLORS[game],
            )
            if row_idx == len(GAMES) - 1:
                ax.set_xlabel("Bout length")
            if col_idx == 0:
                ax.set_ylabel("Count")

    handles = [
        Line2D([0], [0], color="#1f77b4", lw=7, alpha=0.72, label="Short (< Cross2)"),
        Line2D([0], [0], color="#ff7f0e", lw=7, alpha=0.72, label="Long (>= Cross2)"),
        Line2D([0], [0], color="#222222", lw=1.0, ls=":", label="Cross2"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("1-7: Short vs Long NOOP bout distribution by subject and game", y=1.05, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


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


def extract_censored_bouts(actions: np.ndarray) -> List[Tuple[int, bool]]:
    """Return NOOP bout lengths for one episode with episode-end censoring."""
    bouts: List[Tuple[int, bool]] = []
    in_bout = False
    bout_len = 0

    for action in np.asarray(actions, dtype=int).reshape(-1):
        if action == NOOP_ACTION:
            if not in_bout:
                in_bout = True
                bout_len = 1
            else:
                bout_len += 1
        else:
            if in_bout:
                bouts.append((bout_len, False))
            in_bout = False
            bout_len = 0

    if in_bout:
        bouts.append((bout_len, True))
    return bouts


def bouts_for_subject_game(
    episodes: Sequence[Dict],
    subject: str,
    game: int,
) -> List[Tuple[int, bool]]:
    bouts: List[Tuple[int, bool]] = []
    for ep in episodes:
        if ep["subject"] != subject or int(ep["game"]) != game:
            continue
        bouts.extend(extract_censored_bouts(np.asarray(ep["actions"], dtype=int)))
    return bouts


def kaplan_meier_survival(
    bouts: Sequence[Tuple[int, bool]],
    t_max: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return stepwise Kaplan-Meier survival estimate for bout lengths."""
    if not bouts:
        return np.array([], dtype=int), np.array([], dtype=float)

    lengths = np.asarray([length for length, _ in bouts], dtype=int)
    events = np.asarray([not censored for _, censored in bouts], dtype=bool)

    if t_max is None:
        t_max = int(np.percentile(lengths, 99))
    t_max = max(int(t_max), 1)

    times = np.arange(1, t_max + 1, dtype=int)
    survival = np.ones(len(times), dtype=float)
    s = 1.0

    for i, t in enumerate(times):
        n_at_risk = int(np.sum(lengths >= t))
        n_events = int(np.sum((lengths == t) & events))
        if n_at_risk > 0:
            s *= (1.0 - (n_events / n_at_risk))
        survival[i] = s

    return times, survival


def exponential_baseline(
    bouts: Sequence[Tuple[int, bool]],
    times: np.ndarray,
) -> np.ndarray:
    uncensored = [length for length, censored in bouts if not censored]
    if not uncensored:
        uncensored = [length for length, _ in bouts]
    mean_length = float(np.mean(uncensored))
    return np.exp(-times / max(mean_length, EPS))


def print_survival_summary(episodes: Sequence[Dict]) -> None:
    print(f"{'Sub':>4} {'Game':>14} {'N_bouts':>8} {'Censored%':>10} {'Mean_len':>9} {'Median_len':>11}")
    print("-" * 62)
    for subject in SUBJECTS:
        subject_num = subject_sort_key(subject)
        for game_id, game_name in GAMES.items():
            bouts = bouts_for_subject_game(episodes, subject, game_id)
            if not bouts:
                continue
            lengths = np.asarray([length for length, _ in bouts], dtype=float)
            n_censored = sum(censored for _, censored in bouts)
            print(
                f"{subject_num:>4} {game_name:>14} {len(bouts):>8} "
                f"{100 * n_censored / len(bouts):>9.1f}% "
                f"{np.mean(lengths):>9.1f} {np.median(lengths):>11.1f}"
            )


def save_survival_by_subject_figure(
    episodes: Sequence[Dict],
    out_path: Path,
    t_max_by_game: Dict[int, int] | None = None,
) -> None:
    if t_max_by_game is None:
        t_max_by_game = {1: 150, 2: 80}

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes_flat = axes.flatten()

    for ax, subject in zip(axes_flat, SUBJECTS):
        for game_id, game_name in GAMES.items():
            bouts = bouts_for_subject_game(episodes, subject, game_id)
            if not bouts:
                continue

            times, survival = kaplan_meier_survival(
                bouts,
                t_max=t_max_by_game.get(game_id),
            )
            if len(times) == 0:
                continue

            baseline = exponential_baseline(bouts, times)
            color = GAME_COLORS[game_id]
            ax.plot(times, survival, color=color, lw=2, label=game_name)
            ax.plot(
                times,
                baseline,
                color=color,
                lw=1.2,
                ls="--",
                alpha=0.55,
                label=f"{game_name} exp. baseline",
            )

        ax.set_title(f"S{subject_sort_key(subject)}", fontsize=11)
        ax.set_xlabel("Bout length (steps)", fontsize=9)
        ax.set_ylabel("P(surviving)", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)

    fig.text(
        0.5,
        0.01,
        "Solid: KM estimate  |  Dashed: Exponential baseline (random omission null)",
        ha="center",
        fontsize=9,
        color="gray",
    )
    fig.suptitle(
        "1-6: NOOP Bout Survival Function by Subject\n"
        "Pong vs SpaceInvaders - deviation from exponential = planned delay",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def icc_1k(values_by_session: Dict) -> float:
    """Compute ICC(1,1) from a dict {session: [values_across_subjects]}.

    One-way random effects ICC: ICC = (MSB - MSW) / (MSB + (k-1)*MSW)
    where k = number of sessions (raters), subjects are targets.
    """
    sessions = sorted(values_by_session.keys())
    k = len(sessions)
    if k < 2:
        return np.nan
    # Build matrix: rows = subjects, cols = sessions
    # Use only subjects present in all sessions
    subjects_per_session = [set(range(len(values_by_session[s]))) for s in sessions]
    n = min(len(v) for v in values_by_session.values())
    mat = np.array([values_by_session[s][:n] for s in sessions], dtype=float).T  # (n, k)
    if np.any(np.isnan(mat)):
        return np.nan
    grand_mean = mat.mean()
    row_means = mat.mean(axis=1)  # subject means
    col_means = mat.mean(axis=0)  # session means
    SS_between = k * np.sum((row_means - grand_mean) ** 2)
    MS_between = SS_between / (n - 1) if n > 1 else np.nan
    SS_within = np.sum((mat - row_means[:, None]) ** 2)
    MS_within = SS_within / (n * (k - 1)) if n * (k - 1) > 0 else np.nan
    if MS_within is np.nan or MS_between is np.nan:
        return np.nan
    denom = MS_between + (k - 1) * MS_within
    if denom < EPS:
        return np.nan
    return float((MS_between - MS_within) / denom)


# ---------------------------------------------------------------------------
# Load all data
# ---------------------------------------------------------------------------

print("Loading data …")
all_episodes: List[Dict] = []
for sub in SUBJECTS:
    for game in GAMES:
        eps = load_subject_game(sub, game)
        all_episodes.extend(eps)
        print(f"  {sub} game_{game}: {len(eps)} episodes")

thinker_episodes: List[Dict] = []
for game in GAMES:
    eps = load_thinker_game(game)
    thinker_episodes.extend(eps)
    print(f"  thinker game_{game}: {len(eps)} episodes")

df_ep = pd.DataFrame([
    {k: v for k, v in ep.items() if k not in ("actions", "rewards")}
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
action_dist_summary.to_csv(OUT_DIR / "1-1_action_distribution_subject_thinker.csv", index=False)
plot_action_distribution_subjects_vs_thinker(
    summary=action_dist_summary,
    out_path=FIG_DIR / "fig_1-1_B_action_distribution.png",
)
print("Saved 1-1_action_distribution_subject_thinker.csv")
print("Saved fig_1-1_B_action_distribution.png")
save_withholding_schematic(FIG_DIR / "fig_1-1_A_withholding_schematic.png")
print("Saved fig_1-1_A_withholding_schematic.png")

# ---------------------------------------------------------------------------
# 1-2 : Subject-level NOOP analysis
# ---------------------------------------------------------------------------
print("\n=== Section 1-2: Individual differences ===")

episode_reward_summary = build_reward_episode_summary(all_episodes, k_future=REWARD_K)
if len(episode_reward_summary) == 0:
    raise RuntimeError("No episode-level reward summary could be computed for fig_1-3_reward_subject_episode.png")
episode_reward_stats = summarize_reward_episode_subject_stats(episode_reward_summary)
save_reward_subject_episode_figure(
    episode_summary=episode_reward_summary,
    episode_subject_stats=episode_reward_stats,
    out_path=FIG_DIR / "fig_1-3_reward_subject_episode.png",
    k_future=REWARD_K,
)
print("Saved fig_1-3_reward_subject_episode.png")

# --- 1-2a : Subject × Game NOOP proportion ---
subj_game = (
    df_ep.groupby(["subject", "game_name"])["noop_prop"]
    .agg(["mean", "std", "count"])
    .reset_index()
    .rename(columns={"mean": "noop_mean", "std": "noop_std", "count": "n_episodes"})
)
subj_game.to_csv(OUT_DIR / "1-2_subject_game_noop.csv", index=False)
print(subj_game.to_string(index=False))

# --- 1-2b : ICC across sessions (days) within subject × game ---
icc_rows = []
for sub in SUBJECTS:
    for game, gname in GAMES.items():
        sub_data = df_ep[(df_ep["subject"] == sub) & (df_ep["game"] == game)]
        by_day: Dict[int, List[float]] = {}
        for day, grp in sub_data.groupby("day"):
            by_day[int(day)] = grp["noop_prop"].tolist()
        # Compute ICC only if ≥2 days
        days_with_data = {d: v for d, v in by_day.items() if len(v) > 0}
        if len(days_with_data) >= 2:
            icc_val = icc_1k(days_with_data)
        else:
            icc_val = np.nan
        icc_rows.append({"subject": sub, "game": gname, "icc": icc_val,
                         "n_days": len(days_with_data)})

df_icc = pd.DataFrame(icc_rows)
df_icc.to_csv(OUT_DIR / "1-2_icc_by_subject_game.csv", index=False)
print("\nICC across sessions:")
print(df_icc.to_string(index=False))

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
fig.savefig(FIG_DIR / "fig_1-2_individual_differences.png", dpi=200)
plt.close(fig)
print("\nSaved fig_1-2_individual_differences.png")


# ---------------------------------------------------------------------------
# 1-3 : Cross-game generalization
# ---------------------------------------------------------------------------
print("\n=== Section 1-3: Cross-game generalization ===")

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
df_bouts.to_csv(OUT_DIR / "1-3_bout_lengths.csv", index=False)

bout_stats = df_bouts.groupby("game_name")["bout_length"].describe()
print("\nBout length statistics:")
print(bout_stats.to_string())

df_dir.to_csv(OUT_DIR / "1-3_effect_direction.csv", index=False)
print("\nNOOP above chance per subject:")
print(df_dir.to_string(index=False))
print(
    "\nSkipped fig_1-3_cross_game.png "
    "(meta-analytic direction check moved into fig_1-2_individual_differences.png)."
)


# ---------------------------------------------------------------------------
# 1-4 : Alternative explanation exclusion
# ---------------------------------------------------------------------------
print("\n=== Section 1-4: Alternative explanation exclusion ===")

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
df_thirds.to_csv(OUT_DIR / "1-4_episode_position_thirds.csv", index=False)

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
df_ac.to_csv(OUT_DIR / "1-4_noop_autocorrelation.csv", index=False)

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
fig.savefig(FIG_DIR / "fig_1-5_alternative_exclusion.png", dpi=200)
plt.close(fig)
print("\nSaved fig_1-5_alternative_exclusion.png")


# ---------------------------------------------------------------------------
# 1-4 : NOOP ratio ~ post-NOOP action k-step reward scatter
# Each dot = one episode; color = subject; subplot per game
# X: episode NOOP proportion
# Y: mean k-step reward of post-NOOP actions in that episode
# ---------------------------------------------------------------------------
print("\n=== Figure 1-4: NOOP ratio ~ post-NOOP reward scatter ===")

def compute_noop_ratio_reward_scatter(episodes, k_future=REWARD_K):
    """
    For each episode: compute NOOP ratio and mean k-step reward of
    post-NOOP commit actions (first non-NOOP after a NOOP run).
    Returns DataFrame with columns: subject, game, game_name, noop_ratio, post_noop_reward_mean.
    """
    rows = []
    for ep in episodes:
        actions = np.asarray(ep["actions"], dtype=int).reshape(-1)
        rewards = np.asarray(ep["rewards"], dtype=float).reshape(-1)
        n = min(len(actions), len(rewards))
        actions = actions[:n]
        rewards = rewards[:n]
        if n < 5:
            continue

        noop_ratio = float((actions == NOOP_ACTION).mean())

        # Find post-NOOP commit steps: non-NOOP preceded by NOOP
        prev_actions = np.full(n, -1, dtype=int)
        prev_actions[1:] = actions[:-1]
        post_noop_steps = np.where(
            (actions != NOOP_ACTION) & (prev_actions == NOOP_ACTION)
        )[0]

        if len(post_noop_steps) == 0:
            continue

        # k-step reward for each commit step (only if k steps available)
        k_rewards = []
        for s in post_noop_steps:
            end = min(s + 1 + k_future, n)
            if end > s + 1:
                k_rewards.append(float(np.nansum(rewards[s + 1 : end])))

        if not k_rewards:
            continue

        rows.append({
            "subject": ep["subject"],
            "subject_label": f"S{subject_sort_key(ep['subject'])}",
            "game": ep["game"],
            "game_name": ep["game_name"],
            "noop_ratio": noop_ratio,
            "post_noop_reward_mean": float(np.mean(k_rewards)),
            "n_commit": len(k_rewards),
        })

    return pd.DataFrame(rows)


# Join noop_ratio from df_ep into episode_reward_summary
ep_noop = df_ep[["subject", "game", "day", "block", "ep_idx", "noop_prop"]].copy()
df_scatter = episode_reward_summary.merge(
    ep_noop,
    on=["subject", "game", "day", "block", "ep_idx"],
    how="inner",
).dropna(subset=["reward_difference", "noop_prop"])

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax_idx, (game_id, game_name) in enumerate(GAMES.items()):
    ax = axes[ax_idx]
    game_df = df_scatter[df_scatter["game"] == game_id]

    all_x, all_y = [], []
    for sub_idx, sub in enumerate(SUBJECTS):
        sub_df = game_df[game_df["subject"] == sub]
        if sub_df.empty:
            continue
        color = SUBJECT_COLORS[sub_idx % len(SUBJECT_COLORS)]
        sub_num = sub_idx + 1
        ax.scatter(
            sub_df["noop_prop"],
            sub_df["reward_difference"],
            color=color, s=18, alpha=0.55, zorder=2,
            label=f"S{sub_num}",
        )
        # subject centroid
        ax.scatter(
            sub_df["noop_prop"].mean(),
            sub_df["reward_difference"].mean(),
            color=color, s=90, zorder=4,
            marker="D",
        )
        all_x.extend(sub_df["noop_prop"].tolist())
        all_y.extend(sub_df["reward_difference"].tolist())

    # overall regression line
    if len(all_x) > 2:
        coef = np.polyfit(all_x, all_y, 1)
        xx = np.linspace(min(all_x), max(all_x), 100)
        ax.plot(xx, np.polyval(coef, xx), color="black", lw=1.5, ls="--", alpha=0.6, zorder=3)
        r, p = stats.pearsonr(all_x, all_y)
        ax.text(0.05, 0.95, f"r = {r:.2f}, p {format_pvalue(p)}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    ax.axhline(0, color="gray", lw=0.9, ls="--", alpha=0.7)
    ax.set_xlabel("Episode NOOP ratio", fontsize=10)
    ax.set_ylabel(f"Post-NOOP − non-NOOP reward (k={REWARD_K})", fontsize=10)
    ax.set_title(
        f"{game_name}\nNOOP ratio ~ withholding benefit",
        fontsize=10,
        color=GAME_COLORS[game_id],
    )
    ax.legend(fontsize=7, loc="upper right", ncol=2, frameon=False)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle(
    f"1-4: NOOP ratio vs. withholding benefit (k={REWARD_K}) per episode",
    fontsize=11,
)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out_path = FIG_DIR / "fig_1-4_noopratio_postnoop_reward.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out_path.name}")


# ---------------------------------------------------------------------------
# 1-6 : NOOP bout survival analysis
# ---------------------------------------------------------------------------
print("\n=== Figure 1-6: NOOP bout survival analysis ===")
print_survival_summary(all_episodes)
save_survival_by_subject_figure(
    episodes=all_episodes,
    out_path=FIG_DIR / "fig_1-6_survival_by_subject.png",
)
print("Saved fig_1-6_survival_by_subject.png")


# ---------------------------------------------------------------------------
# 1-7 : Short vs Long NOOP bout distribution
# ---------------------------------------------------------------------------
print("\n=== Figure 1-7: Short vs Long NOOP bout distribution ===")
bout_split_detail, bout_split_summary = annotate_short_long_bouts(df_bouts)
bout_split_detail.to_csv(OUT_DIR / "1-7_bout_short_long_detail.csv", index=False)
bout_split_summary.to_csv(OUT_DIR / "1-7_bout_short_long_summary.csv", index=False)
save_short_long_bout_distribution_figure(
    bout_detail=bout_split_detail,
    bout_summary=bout_split_summary,
    out_path=FIG_DIR / "fig_1-7_short_long_bout_distribution.png",
)
print("Saved 1-7_bout_short_long_detail.csv")
print("Saved 1-7_bout_short_long_summary.csv")
print("Saved fig_1-7_short_long_bout_distribution.png")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n=== Done ===")
print(f"Outputs: {OUT_DIR}")
print(f"Figures: {FIG_DIR}")
