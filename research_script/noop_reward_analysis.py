#!/usr/bin/env python3
"""
Figure 4B-style NOOP reward analysis on behavioral_data_block.

For each overt action, mark whether it was directly preceded by a NOOP
(`withholding_preceded`) and compute the downstream k-step reward from the next
real steps within the same episode. Then aggregate to subject x game means and
plot paired comparisons with a normality check followed by either a paired
t-test or a Wilcoxon signed-rank test.

Outputs:
  - outputs/noop_reward_analysis/noop_reward_events_k5.csv
  - outputs/noop_reward_analysis/noop_reward_episode_summary_k5.csv
  - outputs/noop_reward_analysis/noop_reward_subject_summary_k5.csv
  - outputs/noop_reward_analysis/noop_reward_game_stats_k5.csv
  - outputs/noop_reward_analysis/figures/fig_4b_reward_all_games.png
  - outputs/noop_reward_analysis/figures/fig_4b_reward_pong_spaceinvaders.png
  - outputs/noop_reward_analysis/figures/fig_4b_reward_all_games_by_subject_episode.png
  - outputs/noop_reward_analysis/figures/fig_4b_reward_pong_spaceinvaders_by_subject_episode.png
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "behavioral_data_block"
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "noop_reward_analysis"
FIG_DIR = OUT_DIR / "figures"

NOOP_ACTION = 0
K_MAIN = 5
GAME_NAMES = {
    0: "Enduro",
    1: "Pong",
    2: "SpaceInvaders",
}
GAME_ORDERS = {
    "all_games": [0, 1, 2],
    "pong_spaceinvaders": [1, 2],
}
RIGHT_AXIS_GAME_IDS = {2}
EPS = 1e-12

WITH_COLOR = "#2ca02c"
NON_COLOR = "#d62728"
LINE_COLOR = "#7f7f7f"
MEAN_COLOR = "#111111"
RIGHT_AXIS_COLOR = "#1f4e79"


@dataclass(frozen=True)
class BlockMeta:
    subject: str
    subject_label: str
    session: int
    block: int
    game_id: int
    game_name: str


def parse_block_meta(path: Path) -> BlockMeta:
    match = re.match(
        r"sub(\d+)-ses(\d+)-block(\d+)-game(\d+)\.npz$",
        path.name,
    )
    if match is None:
        raise ValueError(f"Unexpected block filename: {path.name}")

    subject_num = int(match.group(1))
    session = int(match.group(2))
    block = int(match.group(3))
    game_id = int(match.group(4))
    return BlockMeta(
        subject=f"sub-{subject_num:03d}",
        subject_label=f"S{subject_num}",
        session=session,
        block=block,
        game_id=game_id,
        game_name=GAME_NAMES.get(game_id, f"game_{game_id}"),
    )


def iter_block_files(data_root: Path) -> Iterable[Path]:
    return sorted(data_root.glob("sub-*/ses-*/*.npz"))


def split_episode_bounds(is_terminal: np.ndarray) -> List[Tuple[int, int]]:
    is_terminal = np.asarray(is_terminal, dtype=bool).reshape(-1)
    if len(is_terminal) == 0:
        return []

    starts = [0]
    terminal_positions = np.where(is_terminal)[0].tolist()
    for pos in terminal_positions:
        if pos + 1 < len(is_terminal):
            starts.append(pos + 1)
    ends = terminal_positions + [len(is_terminal) - 1]
    return [(int(s), int(e)) for s, e in zip(starts, ends) if s <= e]


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


def build_event_table(data_root: Path, k_future: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for path in iter_block_files(data_root):
        meta = parse_block_meta(path)
        data = np.load(path, allow_pickle=True)
        actions = np.argmax(np.asarray(data["action"], dtype=float), axis=1).astype(int)
        rewards = np.asarray(data["reward"], dtype=float).reshape(-1)
        is_terminal = np.asarray(data["is_terminal"], dtype=bool).reshape(-1)

        n = min(len(actions), len(rewards), len(is_terminal))
        actions = actions[:n]
        rewards = rewards[:n]
        is_terminal = is_terminal[:n]

        for episode_index, (start, end) in enumerate(split_episode_bounds(is_terminal)):
            ep_actions = actions[start : end + 1]
            ep_rewards = rewards[start : end + 1]
            if len(ep_actions) < 2:
                continue

            prev_actions = np.full(len(ep_actions), -1, dtype=int)
            prev_actions[1:] = ep_actions[:-1]
            overt_mask = ep_actions != NOOP_ACTION
            preceded_mask = overt_mask & (prev_actions == NOOP_ACTION)

            for step_index in np.where(overt_mask)[0]:
                rows.append(
                    {
                        "subject": meta.subject,
                        "subject_label": meta.subject_label,
                        "session": meta.session,
                        "block": meta.block,
                        "game_id": meta.game_id,
                        "game_name": meta.game_name,
                        "episode_index": int(episode_index),
                        "step_index": int(step_index),
                        "action": int(ep_actions[step_index]),
                        "prev_action": int(prev_actions[step_index]),
                        "preceded_by_withholding": int(preceded_mask[step_index]),
                        "k_reward": float(np.nansum(ep_rewards[step_index + 1 : step_index + 1 + k_future])),
                    }
                )

    return pd.DataFrame(rows).sort_values(
        ["game_id", "subject", "session", "block", "episode_index", "step_index"]
    ).reset_index(drop=True)


def summarize_subject_game(events: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    grouping = ["subject", "subject_label", "game_id", "game_name"]
    for keys, group in events.groupby(grouping, sort=True):
        subject, subject_label, game_id, game_name = keys
        withhold = group[group["preceded_by_withholding"] == 1]
        non = group[group["preceded_by_withholding"] == 0]
        rows.append(
            {
                "subject": subject,
                "subject_label": subject_label,
                "game_id": int(game_id),
                "game_name": game_name,
                "n_withholding_preceded": int(len(withhold)),
                "n_not_preceded": int(len(non)),
                "reward_withholding_preceded_mean": float(withhold["k_reward"].mean()) if len(withhold) > 0 else np.nan,
                "reward_not_preceded_mean": float(non["k_reward"].mean()) if len(non) > 0 else np.nan,
            }
        )

    summary = pd.DataFrame(rows).sort_values(["game_id", "subject"]).reset_index(drop=True)
    summary["reward_difference"] = (
        summary["reward_withholding_preceded_mean"] - summary["reward_not_preceded_mean"]
    )
    return summary


def summarize_episode_game(events: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    grouping = [
        "subject",
        "subject_label",
        "session",
        "block",
        "game_id",
        "game_name",
        "episode_index",
    ]
    for keys, group in events.groupby(grouping, sort=True):
        subject, subject_label, session, block, game_id, game_name, episode_index = keys
        withhold = group[group["preceded_by_withholding"] == 1]
        non = group[group["preceded_by_withholding"] == 0]
        rows.append(
            {
                "subject": subject,
                "subject_label": subject_label,
                "session": int(session),
                "block": int(block),
                "game_id": int(game_id),
                "game_name": game_name,
                "episode_index": int(episode_index),
                "n_withholding_preceded": int(len(withhold)),
                "n_not_preceded": int(len(non)),
                "reward_withholding_preceded_mean": float(withhold["k_reward"].mean()) if len(withhold) > 0 else np.nan,
                "reward_not_preceded_mean": float(non["k_reward"].mean()) if len(non) > 0 else np.nan,
            }
        )

    summary = pd.DataFrame(rows).sort_values(
        ["subject", "game_id", "session", "block", "episode_index"]
    ).reset_index(drop=True)
    summary["reward_difference"] = (
        summary["reward_withholding_preceded_mean"] - summary["reward_not_preceded_mean"]
    )
    summary["episode_label"] = summary.apply(
        lambda row: f"ses{int(row['session']):02d}-block{int(row['block']):02d}-ep{int(row['episode_index']):02d}",
        axis=1,
    )
    return summary


def summarize_game_stats(subject_summary: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for game_id, game_name in GAME_NAMES.items():
        group = subject_summary[subject_summary["game_id"] == game_id].copy()
        group = group.dropna(
            subset=["reward_withholding_preceded_mean", "reward_not_preceded_mean"]
        )
        stats_result = choose_paired_test(
            group["reward_withholding_preceded_mean"].to_numpy(dtype=float),
            group["reward_not_preceded_mean"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "game_id": game_id,
                "game_name": game_name,
                "mean_withholding_preceded": float(group["reward_withholding_preceded_mean"].mean()) if len(group) > 0 else np.nan,
                "mean_not_preceded": float(group["reward_not_preceded_mean"].mean()) if len(group) > 0 else np.nan,
                **stats_result,
            }
        )
    return pd.DataFrame(rows)


def summarize_episode_subject_game_stats(episode_summary: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    grouping = ["subject", "subject_label", "game_id", "game_name"]
    for keys, group in episode_summary.groupby(grouping, sort=True):
        subject, subject_label, game_id, game_name = keys
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
                "game_id": int(game_id),
                "game_name": game_name,
                "n_episodes": int(len(group)),
                "mean_withholding_preceded": float(group["reward_withholding_preceded_mean"].mean()) if len(group) > 0 else np.nan,
                "mean_not_preceded": float(group["reward_not_preceded_mean"].mean()) if len(group) > 0 else np.nan,
                **stats_result,
            }
        )
    return pd.DataFrame(rows).sort_values(["subject", "game_id"]).reset_index(drop=True)


def add_significance_annotation(
    ax: plt.Axes,
    x_left: float,
    x_right: float,
    y_values: np.ndarray,
    label: str,
) -> None:
    finite = y_values[np.isfinite(y_values)]
    if len(finite) == 0:
        return
    y_min = float(np.min(finite))
    y_max = float(np.max(finite))
    y_range = max(y_max - y_min, 0.1)
    y = y_max + 0.12 * y_range
    h = 0.03 * y_range
    ax.plot([x_left, x_left, x_right, x_right], [y, y + h, y + h, y], color="black", lw=1.2)
    ax.text((x_left + x_right) / 2, y + h + 0.01 * y_range, label, ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_ylim(y_min - 0.15 * y_range, y + h + 0.18 * y_range)


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


def plot_game_panel(
    ax: plt.Axes,
    group: pd.DataFrame,
    stats_row: pd.Series,
    title: str,
) -> None:
    x_with, x_non = 0, 1

    with_vals = group["reward_withholding_preceded_mean"].to_numpy(dtype=float)
    non_vals = group["reward_not_preceded_mean"].to_numpy(dtype=float)

    for _, row in group.iterrows():
        ax.plot(
            [x_with, x_non],
            [row["reward_withholding_preceded_mean"], row["reward_not_preceded_mean"]],
            color=LINE_COLOR,
            lw=1.2,
            alpha=0.75,
            zorder=1,
        )
        ax.annotate(
            row["subject_label"],
            (x_with, row["reward_withholding_preceded_mean"]),
            textcoords="offset points",
            xytext=(-20, 0),
            fontsize=7,
            color=WITH_COLOR,
            va="center",
        )
        ax.annotate(
            row["subject_label"],
            (x_non, row["reward_not_preceded_mean"]),
            textcoords="offset points",
            xytext=(5, 0),
            fontsize=7,
            color=NON_COLOR,
            va="center",
        )

    ax.scatter(np.full(len(with_vals), x_with), with_vals, s=55, color=WITH_COLOR, zorder=3)
    ax.scatter(np.full(len(non_vals), x_non), non_vals, s=55, color=NON_COLOR, zorder=3)

    if len(with_vals) > 0:
        mean_with = float(np.mean(with_vals))
        mean_non = float(np.mean(non_vals))
        sem_with = float(stats.sem(with_vals)) if len(with_vals) > 1 else 0.0
        sem_non = float(stats.sem(non_vals)) if len(non_vals) > 1 else 0.0
        ax.errorbar(
            [x_with, x_non],
            [mean_with, mean_non],
            yerr=[sem_with, sem_non],
            color=MEAN_COLOR,
            lw=2.0,
            fmt="o",
            markersize=7,
            capsize=4,
            zorder=4,
        )

    ax.set_xticks([x_with, x_non], ["Withholding\npreceded", "Not\npreceded"])
    ax.set_ylabel(f"Mean k={K_MAIN} downstream reward")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

    stat_text = (
        f"n={int(stats_row['n_subjects'])}\n"
        f"Shapiro p {format_pvalue(float(stats_row['shapiro_p']))}\n"
        f"{stats_row['test_name']}: p {format_pvalue(float(stats_row['p_value']))}"
    )
    ax.text(
        0.02,
        0.98,
        stat_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )
    add_significance_annotation(
        ax,
        x_with,
        x_non,
        np.concatenate([with_vals, non_vals]),
        str(stats_row["significance"]),
    )


def make_figure(
    subject_summary: pd.DataFrame,
    game_stats: pd.DataFrame,
    game_ids: Sequence[int],
    out_path: Path,
    figure_title: str,
) -> None:
    fig, axes = plt.subplots(1, len(game_ids), figsize=(6.2 * len(game_ids), 5.6), sharey=False)
    if len(game_ids) == 1:
        axes = [axes]

    for ax, game_id in zip(axes, game_ids):
        group = subject_summary[subject_summary["game_id"] == game_id].copy()
        group = group.dropna(
            subset=["reward_withholding_preceded_mean", "reward_not_preceded_mean"]
        )
        stats_row = game_stats.loc[game_stats["game_id"] == game_id].iloc[0]
        plot_game_panel(ax, group, stats_row, GAME_NAMES[game_id])

    fig.suptitle(figure_title, y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def subject_sort_key(subject: str) -> int:
    match = re.search(r"(\d+)$", subject)
    return int(match.group(1)) if match else 9999


def axis_values_for_games(
    episode_summary: pd.DataFrame,
    game_ids: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    left_game_ids = [gid for gid in game_ids if gid not in RIGHT_AXIS_GAME_IDS]
    right_game_ids = [gid for gid in game_ids if gid in RIGHT_AXIS_GAME_IDS]

    left_vals: List[float] = []
    right_vals: List[float] = []
    for _, row in episode_summary.iterrows():
        vals = [
            row["reward_withholding_preceded_mean"],
            row["reward_not_preceded_mean"],
        ]
        if int(row["game_id"]) in left_game_ids:
            left_vals.extend(vals)
        if int(row["game_id"]) in right_game_ids:
            right_vals.extend(vals)
    return np.asarray(left_vals, dtype=float), np.asarray(right_vals, dtype=float)


def plot_subject_episode_panel(
    ax: plt.Axes,
    ax_right: plt.Axes | None,
    subject_episode_summary: pd.DataFrame,
    subject_episode_stats: pd.DataFrame,
    subject_label: str,
    game_ids: Sequence[int],
    left_limits: Tuple[float, float],
    right_limits: Tuple[float, float] | None,
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
        target_ax = ax_right if ax_right is not None and game_id in RIGHT_AXIS_GAME_IDS else ax

        game_rows = subject_episode_summary[
            subject_episode_summary["game_id"] == game_id
        ].dropna(
            subset=["reward_withholding_preceded_mean", "reward_not_preceded_mean"]
        )

        ax.text(
            (x_with + x_non) / 2,
            1.02,
            GAME_NAMES[game_id],
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
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

        game_stats_row = subject_episode_stats[
            subject_episode_stats["game_id"] == game_id
        ]
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
        add_significance_annotation_with_limits(
            target_ax,
            x_with,
            x_non,
            vals,
            label,
        )


def make_subject_episode_figure(
    episode_summary: pd.DataFrame,
    episode_subject_stats: pd.DataFrame,
    game_ids: Sequence[int],
    out_path: Path,
    figure_title: str,
) -> None:
    subjects = sorted(episode_summary["subject"].unique(), key=subject_sort_key)
    subject_labels = {
        subject: episode_summary.loc[episode_summary["subject"] == subject, "subject_label"].iloc[0]
        for subject in subjects
    }

    left_vals, right_vals = axis_values_for_games(episode_summary, game_ids)
    left_limits = make_limits(left_vals)
    right_limits = make_limits(right_vals) if np.isfinite(right_vals).any() else None

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=False)
    axes_flat = axes.flatten()
    right_axes: List[plt.Axes | None] = []

    for ax, subject in zip(axes_flat, subjects):
        subject_rows = episode_summary[episode_summary["subject"] == subject].copy()
        subject_stats = episode_subject_stats[episode_subject_stats["subject"] == subject].copy()
        ax_right = ax.twinx() if any(gid in RIGHT_AXIS_GAME_IDS for gid in game_ids) else None
        right_axes.append(ax_right)
        plot_subject_episode_panel(
            ax=ax,
            ax_right=ax_right,
            subject_episode_summary=subject_rows,
            subject_episode_stats=subject_stats,
            subject_label=subject_labels[subject],
            game_ids=game_ids,
            left_limits=left_limits,
            right_limits=right_limits,
        )

    for ax in axes_flat[ len(subjects) : ]:
        ax.set_axis_off()

    for ax in axes[:, 0]:
        ax.set_ylabel(f"Episode mean k={K_MAIN} reward\n(Enduro / Pong scale)")

    for ax_right in right_axes:
        if ax_right is not None:
            ax_right.set_ylabel(f"Episode mean k={K_MAIN} reward\n(SpaceInvaders scale)", color=RIGHT_AXIS_COLOR)

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
    fig.suptitle(figure_title, y=1.03, fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NOOP reward analysis on behavioral_data_block."
    )
    parser.add_argument(
        "--k-main",
        type=int,
        default=K_MAIN,
        help="Downstream reward horizon k to use for analysis and output filenames.",
    )
    return parser.parse_args()


def main() -> None:
    global K_MAIN

    args = parse_args()
    K_MAIN = int(args.k_main)
    if K_MAIN < 1:
        raise ValueError("--k-main must be >= 1.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Building NOOP reward event table ...")
    events = build_event_table(DATA_ROOT, k_future=K_MAIN)
    if len(events) == 0:
        raise RuntimeError(f"No events were extracted from {DATA_ROOT}")

    episode_summary = summarize_episode_game(events)
    subject_summary = summarize_subject_game(events)
    game_stats = summarize_game_stats(subject_summary)
    episode_subject_stats = summarize_episode_subject_game_stats(episode_summary)

    events.to_csv(OUT_DIR / f"noop_reward_events_k{K_MAIN}.csv", index=False)
    episode_summary.to_csv(OUT_DIR / f"noop_reward_episode_summary_k{K_MAIN}.csv", index=False)
    subject_summary.to_csv(OUT_DIR / f"noop_reward_subject_summary_k{K_MAIN}.csv", index=False)
    game_stats.to_csv(OUT_DIR / f"noop_reward_game_stats_k{K_MAIN}.csv", index=False)
    episode_subject_stats.to_csv(OUT_DIR / f"noop_reward_episode_subject_game_stats_k{K_MAIN}.csv", index=False)

    make_figure(
        subject_summary=subject_summary,
        game_stats=game_stats,
        game_ids=GAME_ORDERS["all_games"],
        out_path=FIG_DIR / "fig_4b_reward_all_games.png",
        figure_title="Figure 4B-style reward comparison across all games",
    )
    make_figure(
        subject_summary=subject_summary,
        game_stats=game_stats,
        game_ids=GAME_ORDERS["pong_spaceinvaders"],
        out_path=FIG_DIR / "fig_4b_reward_pong_spaceinvaders.png",
        figure_title="Figure 4B-style reward comparison: Pong and SpaceInvaders",
    )
    make_subject_episode_figure(
        episode_summary=episode_summary,
        episode_subject_stats=episode_subject_stats,
        game_ids=GAME_ORDERS["all_games"],
        out_path=FIG_DIR / "fig_4b_reward_all_games_by_subject_episode.png",
        figure_title="Episode-level reward comparison across all games by subject",
    )
    make_subject_episode_figure(
        episode_summary=episode_summary,
        episode_subject_stats=episode_subject_stats,
        game_ids=GAME_ORDERS["pong_spaceinvaders"],
        out_path=FIG_DIR / "fig_4b_reward_pong_spaceinvaders_by_subject_episode.png",
        figure_title="Episode-level reward comparison: Pong and SpaceInvaders by subject",
    )

    print("Saved outputs to:")
    print(f"  {OUT_DIR}")
    print("\nGame-level stats:")
    print(game_stats.to_string(index=False))


if __name__ == "__main__":
    main()
