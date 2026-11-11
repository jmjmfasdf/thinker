#!/usr/bin/env python3
"""Plot subject-matched imitation-learning calibration summaries.

The script reads each imitated Thinker's exported
``rmean_episode_return`` series, excluding the companion ``__MIN`` and
``__MAX`` columns.  For every subject x game pair it then adds:

* the matched human's mean recording-defined segment return, computed from the
  behavioral NPZ files with the same terminal-based split as
  ``01_behavioral_analysis.py`` and visibly marked as horizon-unmatched; and
* human/Thinker action distributions computed on the same aligned real-step
  rows in the canonical ``00_cache`` actor-policy table.

These are training-data calibration plots, not an independent held-out
behavioral validation of the imitated Thinker.

Examples
--------
python research_script/00_learningcurve.py
python research_script/00_learningcurve.py --max-step 2000000
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "game_return.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "00_learningcurve"
DEFAULT_BEHAVIORAL_ROOT = ROOT / "behavioral_data_block"
DEFAULT_ACTION_CACHE_ROOT = SCRIPT_DIR / "outputs" / "00_cache"
DEFAULT_MAX_STEP = 2_000_000

GAME_TITLES = {
    "pong": "Pong",
    "spaceinvaders": "Space Invaders",
}
GAME_IDS = {
    "pong": 1,
    "spaceinvaders": 2,
}
EXPECTED_CANONICAL_PAIRS = {
    (game, subject)
    for game in ("pong", "spaceinvaders")
    for subject in range(1, 7)
}

ACTION_LABELS = [
    "NOOP",
    "FIRE",
    "RIGHT",
    "LEFT",
    "RIGHTFIRE",
    "LEFTFIRE",
]
HUMAN_COLOR = "#E15759"
THINKER_COLOR = "#4E79A7"

# Subject identity is encoded redundantly by color and marker so the six
# learning curves remain distinguishable without relying on color alone.
# Human references use the same subject color with a dashed line.
SUBJECT_COLORS = {
    1: "#4E79A7",
    2: "#F28E2B",
    3: "#E15759",
    4: "#59A14F",
    5: "#B07AA1",
    6: "#9C755F",
}
SUBJECT_MARKERS = {
    1: "o",
    2: "s",
    3: "^",
    4: "D",
    5: "P",
    6: "X",
}

SERIES_PATTERN = re.compile(
    r"^(?P<game>pong|spaceinvaders)_sub(?P<subject>\d+)_game\d+"
    r".* - rmean_episode_return$"
)
BEHAVIOR_FILE_PATTERN = re.compile(
    r"sub(?P<subject>\d+)-ses(?P<session>\d+)-"
    r"block(?P<block>\d+)-game(?P<game>\d+)\.npz$"
)


def configure_matplotlib() -> None:
    """Use the bundled Arial font when available."""
    font_path = ROOT / "ARIAL.TTF"
    font_family = "DejaVu Sans"
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        font_family = font_manager.FontProperties(
            fname=str(font_path)
        ).get_name()
    matplotlib.rcParams.update(
        {
            "font.family": font_family,
            "axes.edgecolor": "#404040",
            "axes.labelcolor": "#303030",
            "axes.titlecolor": "#202020",
            "text.color": "#303030",
            "xtick.color": "#404040",
            "ytick.color": "#404040",
            "savefig.facecolor": "white",
        }
    )


def discover_series(columns: List[str]) -> Dict[Tuple[str, int], str]:
    """Map each game/subject pair to its unsuffixed rmean column."""
    discovered: Dict[Tuple[str, int], str] = {}
    for column in columns:
        match = SERIES_PATTERN.match(column)
        if match is None:
            continue
        key = (match.group("game"), int(match.group("subject")))
        if key in discovered:
            raise ValueError(
                f"Multiple rmean series found for {key}: "
                f"{discovered[key]!r} and {column!r}"
            )
        discovered[key] = column
    if not discovered:
        raise ValueError("No unsuffixed rmean_episode_return columns found")
    return discovered


def format_training_step(value: float, _: int) -> str:
    if abs(value) < 1:
        return "0"
    return f"{value / 1_000_000:g}M"


def _subject_color(subject: int) -> object:
    """Return a stable explicit color for a subject id."""
    if subject in SUBJECT_COLORS:
        return SUBJECT_COLORS[subject]
    return plt.get_cmap("tab20")((subject - 1) % 20)


def _subject_marker(subject: int) -> str:
    markers = tuple(SUBJECT_MARKERS.values())
    return SUBJECT_MARKERS.get(subject, markers[(subject - 1) % len(markers)])


def load_human_return_summaries(
    behavioral_root: Path,
    pairs: Sequence[Tuple[str, int]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load human return summaries with analysis-01-compatible splitting.

    Recording-defined segments end at ``is_terminal`` rows.  A final
    non-terminal block fragment is retained when it has at least five steps,
    matching ``01_behavioral_analysis.py``.  The segment label is deliberate:
    behavioral recording termination is not assumed to match the Thinker
    training environment's episode horizon or termination rule.
    """
    rows: List[Dict[str, object]] = []
    block_rows: List[Dict[str, object]] = []
    for game, subject in sorted(pairs):
        game_id = GAME_IDS[game]
        subject_dir = behavioral_root / f"sub-{subject:03d}"
        files = sorted(subject_dir.glob(f"ses-*/*game{game_id}.npz"))
        if not files:
            raise FileNotFoundError(
                "No behavioral NPZ files for "
                f"S{subject} {GAME_TITLES[game]} under {subject_dir}"
            )

        n_pair_segments = 0
        for path in files:
            match = BEHAVIOR_FILE_PATTERN.fullmatch(path.name)
            if match is None:
                raise ValueError(f"Unexpected behavioral filename: {path}")
            if (
                int(match.group("subject")) != subject
                or int(match.group("game")) != game_id
            ):
                raise ValueError(f"Behavioral path does not match pair: {path}")

            with np.load(path, allow_pickle=True) as data:
                actions = np.asarray(data["action"])
                rewards = np.asarray(data["reward"], dtype=float).reshape(-1)
                terminals = np.asarray(data["is_terminal"], dtype=bool).reshape(-1)
                times = np.asarray(data["time"], dtype=float).reshape(-1)

            n_file = min(
                int(actions.shape[0]),
                len(rewards),
                len(terminals),
                len(times),
            )
            if n_file == 0:
                raise ValueError(f"Empty behavioral arrays in {path}")
            rewards = rewards[:n_file]
            terminals = terminals[:n_file]
            times = times[:n_file]
            if n_file > 1 and (
                not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0)
            ):
                raise RuntimeError(
                    f"Non-finite or non-increasing timestamps in {path}"
                )

            block_rows.append(
                {
                    "game": game,
                    "game_id": game_id,
                    "game_title": GAME_TITLES[game],
                    "subject": subject,
                    "session": int(match.group("session")),
                    "block": int(match.group("block")),
                    "source_file": str(path.resolve()),
                    "n_steps": n_file,
                    "duration_seconds": (
                        float(times[-1] - times[0]) if n_file > 1 else 0.0
                    ),
                    "n_terminal_flags": int(terminals.sum()),
                    "terminal_flag_at_block_end": bool(terminals[-1]),
                    "block_return": float(rewards.sum()),
                }
            )

            terminal_positions = np.flatnonzero(terminals).tolist()
            episode_starts = [0]
            for terminal_position in terminal_positions:
                if terminal_position + 1 < n_file:
                    episode_starts.append(terminal_position + 1)
            episode_ends = terminal_positions + [n_file - 1]

            for local_episode, (start, stop) in enumerate(
                zip(episode_starts, episode_ends)
            ):
                n_steps = int(stop - start + 1)
                if n_steps < 5:
                    continue
                n_pair_segments += 1
                rows.append(
                    {
                        "game": game,
                        "game_id": game_id,
                        "game_title": GAME_TITLES[game],
                        "subject": subject,
                        "session": int(match.group("session")),
                        "block": int(match.group("block")),
                        "segment_index_within_file": local_episode,
                        "analysis01_episode_id": int(
                            game_id * 1000 + local_episode
                        ),
                        "source_file": str(path.resolve()),
                        "n_steps": n_steps,
                        "terminal_flag_at_segment_end": bool(terminals[stop]),
                        "segment_return": float(rewards[start : stop + 1].sum()),
                    }
                )
        if n_pair_segments == 0:
            raise ValueError(
                f"No usable human segments for S{subject} {GAME_TITLES[game]}"
            )

    segments = pd.DataFrame(rows)
    group_columns = ["game", "game_id", "game_title", "subject"]
    segment_summary = (
        segments.groupby(group_columns, as_index=False, sort=True)
        .agg(
            n_behavior_files=("source_file", "nunique"),
            n_segments=("segment_return", "size"),
            n_terminal_flag_ended_segments=(
                "terminal_flag_at_segment_end",
                "sum",
            ),
            mean_recording_defined_segment_return=("segment_return", "mean"),
            sd_recording_defined_segment_return=("segment_return", "std"),
            median_recording_defined_segment_return=("segment_return", "median"),
            minimum_recording_defined_segment_return=("segment_return", "min"),
            maximum_recording_defined_segment_return=("segment_return", "max"),
        )
        .sort_values(["game_id", "subject"])
        .reset_index(drop=True)
    )
    segment_summary["n_nonterminal_final_segments"] = (
        segment_summary["n_segments"]
        - segment_summary["n_terminal_flag_ended_segments"]
    ).astype(int)
    segment_summary["sem_recording_defined_segment_return"] = segment_summary[
        "sd_recording_defined_segment_return"
    ] / np.sqrt(
        segment_summary["n_segments"]
    )
    complete_means = (
        segments.loc[segments["terminal_flag_at_segment_end"]]
        .groupby(group_columns, as_index=False, sort=True)["segment_return"]
        .mean()
        .rename(
            columns={
                "segment_return": (
                    "mean_terminal_flag_ended_segment_return"
                ),
            }
        )
    )
    segment_summary = segment_summary.merge(
        complete_means,
        on=group_columns,
        how="left",
    )
    segment_summary["reference_definition"] = (
        "mean return across analysis-01 terminal-split recorded segments "
        "(>=5 steps); descriptive and not horizon/termination-matched to "
        "Thinker rmean_episode_return"
    )

    blocks = pd.DataFrame(block_rows)
    block_summary = (
        blocks.groupby(group_columns, as_index=False, sort=True)
        .agg(
            n_blocks=("block_return", "size"),
            mean_block_return=("block_return", "mean"),
            sd_block_return=("block_return", "std"),
            sem_block_return=("block_return", "sem"),
            median_block_return=("block_return", "median"),
            minimum_block_return=("block_return", "min"),
            maximum_block_return=("block_return", "max"),
            mean_block_duration_seconds=("duration_seconds", "mean"),
            minimum_block_duration_seconds=("duration_seconds", "min"),
            maximum_block_duration_seconds=("duration_seconds", "max"),
            n_terminal_flags=("n_terminal_flags", "sum"),
            all_blocks_end_with_terminal_flag=(
                "terminal_flag_at_block_end",
                "all",
            ),
        )
        .sort_values(["game_id", "subject"])
        .reset_index(drop=True)
    )
    block_summary["comparison_caveat"] = (
        "approximately 480-s recorded block; not comparable to Thinker "
        "rmean_episode_return because horizons and termination rules differ"
    )
    return segments, segment_summary, blocks, block_summary


def load_action_distribution_cache(
    action_cache_root: Path,
    pairs: Sequence[Tuple[str, int]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate paired human/model actions from canonical real-step caches."""
    count_rows: List[Dict[str, object]] = []
    qa_rows: List[Dict[str, object]] = []
    required_columns = [
        "subject",
        "game",
        "real_step_id",
        "cross_chunk_boundary",
        "human_action",
        "thinker_action",
    ]

    for game, subject in sorted(pairs):
        game_id = GAME_IDS[game]
        cache_path = (
            action_cache_root
            / f"sub{subject:03d}_game{game_id}"
            / "actor_policy"
            / "real_step_metrics.csv.gz"
        )
        if not cache_path.exists():
            raise FileNotFoundError(cache_path)
        cached = pd.read_csv(cache_path, usecols=required_columns)
        n_rows_read = int(len(cached))

        subject_values = pd.to_numeric(cached["subject"], errors="coerce")
        game_values = pd.to_numeric(cached["game"], errors="coerce")
        pair_match = (subject_values == subject) & (game_values == game_id)
        n_pair_mismatch = int((~pair_match).sum())
        cached = cached.loc[pair_match].copy()
        n_pair_rows_before_deduplication = int(len(cached))

        if cached["real_step_id"].isna().any():
            raise ValueError(f"Missing real_step_id in {cache_path}")
        boundary_numeric = pd.to_numeric(
            cached["cross_chunk_boundary"],
            errors="coerce",
        )
        duplicated = cached.duplicated("real_step_id", keep=False)
        n_duplicate_rows = int(duplicated.sum())
        if n_duplicate_rows:
            conflict = (
                cached.loc[duplicated]
                .groupby("real_step_id")[["human_action", "thinker_action"]]
                .nunique(dropna=False)
                .gt(1)
                .any(axis=1)
            )
            if bool(conflict.any()):
                bad_id = str(conflict.index[conflict][0])
                raise ValueError(
                    f"Conflicting duplicate actions for {bad_id} in {cache_path}"
                )
            # Prefer a non-boundary copy if a duplicated id occurs on both a
            # chunk edge and a within-chunk row.
            cached["_boundary_preference"] = np.where(
                boundary_numeric == 0,
                0,
                1,
            )
            cached = (
                cached.sort_values(
                    ["real_step_id", "_boundary_preference"],
                    kind="stable",
                )
                .drop_duplicates("real_step_id", keep="first")
                .drop(columns="_boundary_preference")
            )

        n_unique_after_deduplication = int(len(cached))
        boundary_numeric = pd.to_numeric(
            cached["cross_chunk_boundary"],
            errors="coerce",
        )
        within_chunk = boundary_numeric == 0
        n_boundary_excluded = int((~within_chunk).sum())
        cached = cached.loc[within_chunk].copy()

        human_numeric = pd.to_numeric(cached["human_action"], errors="coerce")
        thinker_numeric = pd.to_numeric(
            cached["thinker_action"], errors="coerce"
        )
        human_integer = human_numeric.notna() & (human_numeric % 1 == 0)
        thinker_integer = thinker_numeric.notna() & (thinker_numeric % 1 == 0)
        human_valid = human_integer & human_numeric.between(
            0, len(ACTION_LABELS) - 1
        )
        thinker_valid = thinker_integer & thinker_numeric.between(
            0, len(ACTION_LABELS) - 1
        )
        paired_valid = human_valid & thinker_valid
        paired = cached.loc[paired_valid].copy()
        paired["human_action"] = human_numeric.loc[paired_valid].astype(int)
        paired["thinker_action"] = thinker_numeric.loc[paired_valid].astype(int)
        if paired.empty:
            raise ValueError(f"No paired valid actions in {cache_path}")

        human_counts = np.bincount(
            paired["human_action"].to_numpy(dtype=int),
            minlength=len(ACTION_LABELS),
        )
        thinker_counts = np.bincount(
            paired["thinker_action"].to_numpy(dtype=int),
            minlength=len(ACTION_LABELS),
        )
        n_paired = int(len(paired))
        human_proportions = human_counts / n_paired
        thinker_proportions = thinker_counts / n_paired
        agreement = float(
            np.mean(
                paired["human_action"].to_numpy()
                == paired["thinker_action"].to_numpy()
            )
        )
        total_variation = float(
            0.5 * np.abs(human_proportions - thinker_proportions).sum()
        )

        for source, counts, proportions in (
            ("human_observed", human_counts, human_proportions),
            (
                "thinker_replay_recommendation",
                thinker_counts,
                thinker_proportions,
            ),
        ):
            for action_id, (count, proportion) in enumerate(
                zip(counts, proportions)
            ):
                count_rows.append(
                    {
                        "game": game,
                        "game_id": game_id,
                        "game_title": GAME_TITLES[game],
                        "subject": subject,
                        "source": source,
                        "action_id": action_id,
                        "action_label": ACTION_LABELS[action_id],
                        "count": int(count),
                        "proportion": float(proportion),
                        "n_paired_rows": n_paired,
                        "exact_action_agreement": agreement,
                        "total_variation_distance": total_variation,
                        "cache_file": str(cache_path.resolve()),
                    }
                )

        qa_rows.append(
            {
                "game": game,
                "game_id": game_id,
                "game_title": GAME_TITLES[game],
                "subject": subject,
                "cache_file": str(cache_path.resolve()),
                "n_rows_read": n_rows_read,
                "n_subject_or_game_mismatch": n_pair_mismatch,
                "n_duplicate_rows_before_deduplication": n_duplicate_rows,
                "n_rows_removed_as_duplicates": int(
                    n_pair_rows_before_deduplication
                    - n_unique_after_deduplication
                ),
                "n_unique_rows_after_deduplication": (
                    n_unique_after_deduplication
                ),
                "n_cross_chunk_boundary_rows_excluded": n_boundary_excluded,
                "n_candidate_rows_after_boundary_filter": int(len(cached)),
                "n_missing_human_action": int(human_numeric.isna().sum()),
                "n_noninteger_or_out_of_range_human_action": int(
                    ((~human_numeric.isna()) & (~human_valid)).sum()
                ),
                "n_missing_thinker_action": int(thinker_numeric.isna().sum()),
                "n_noninteger_or_out_of_range_thinker_action": int(
                    ((~thinker_numeric.isna()) & (~thinker_valid)).sum()
                ),
                "n_paired_valid_rows": n_paired,
                "n_invalid_missing_or_sentinel_pairs_excluded": int(
                    len(cached) - n_paired
                ),
                "n_total_rows_not_in_paired_distribution": int(
                    n_rows_read - n_paired
                ),
                "exact_action_agreement": agreement,
                "total_variation_distance": total_variation,
                "row_definition": (
                    "paired valid human_action and thinker_action on aligned "
                    "cached real decision rows"
                ),
                "thinker_action_semantics": (
                    "model recommendation during teacher-forced replay; "
                    "not a free-running generative action"
                ),
            }
        )

    return pd.DataFrame(count_rows), pd.DataFrame(qa_rows)


def make_human_performance_lookup(
    human_summary: pd.DataFrame,
) -> Dict[Tuple[str, int], Dict[str, object]]:
    """Index one human performance summary row per game x subject."""
    lookup: Dict[Tuple[str, int], Dict[str, object]] = {}
    for row in human_summary.to_dict(orient="records"):
        key = (str(row["game"]), int(row["subject"]))
        if key in lookup:
            raise ValueError(f"Duplicate human performance row for {key}")
        lookup[key] = row
    return lookup


def learning_curve_legend_handles(
    subjects: Sequence[int],
) -> List[Line2D]:
    """Create subject and line-semantics legend handles."""
    subject_handles = [
        Line2D(
            [0],
            [0],
            color=_subject_color(subject),
            marker=_subject_marker(subject),
            markersize=5,
            linewidth=2,
            label=f"S{subject}",
        )
        for subject in subjects
    ]
    semantics = [
        Line2D(
            [0],
            [0],
            color="#303030",
            linewidth=2.2,
            label="Imitated Thinker training rmean",
        ),
        Line2D(
            [0],
            [0],
            color="#303030",
            linewidth=1.5,
            linestyle=(0, (4, 3)),
            label=(
                "Human recording-defined segment mean; termination/horizon "
                "not matched to Thinker training episode"
            ),
        ),
    ]
    return subject_handles + semantics


def draw_learning_curve_axis(
    ax: plt.Axes,
    frame: pd.DataFrame,
    series: Dict[Tuple[str, int], str],
    human_lookup: Mapping[Tuple[str, int], Mapping[str, object]],
    game: str,
    max_step: int,
    title: str,
    show_legend: bool,
    compact: bool = False,
) -> Tuple[List[Dict[str, object]], List[int]]:
    """Draw each imitated Thinker and its matched human performance line."""
    subjects = sorted(
        subject
        for series_game, subject in series
        if series_game == game
    )
    if not subjects:
        raise ValueError(f"No series found for game={game}")

    summary: List[Dict[str, object]] = []
    plotted_subjects: List[int] = []
    for subject in subjects:
        column = series[(game, subject)]
        values = pd.to_numeric(frame[column], errors="coerce")
        valid = pd.DataFrame(
            {
                "step": frame["Step"],
                "return": values,
            }
        ).dropna()
        valid = valid.loc[
            valid["step"].between(0, max_step, inclusive="both")
        ].sort_values("step", kind="stable")
        valid = valid.drop_duplicates("step", keep="last")
        if valid.empty:
            continue

        human_row = human_lookup.get((game, subject))
        if human_row is None:
            raise ValueError(
                f"No matched human performance for {game}, subject={subject}"
            )
        steps = valid["step"].to_numpy(dtype=float)
        returns = valid["return"].to_numpy(dtype=float)
        color = _subject_color(subject)
        marker = _subject_marker(subject)
        markevery = max(1, len(valid) // (10 if compact else 14))
        ax.plot(
            steps,
            returns,
            color=color,
            linewidth=1.6 if compact else 2.0,
            alpha=0.92,
            marker=marker,
            markersize=2.8 if compact else 3.4,
            markevery=markevery,
            markeredgewidth=0,
            label=f"S{subject}",
            zorder=3,
        )
        human_mean = float(
            human_row["mean_recording_defined_segment_return"]
        )
        ax.axhline(
            human_mean,
            color=color,
            linewidth=1.05 if compact else 1.35,
            linestyle=(0, (4, 3)),
            alpha=0.82,
            zorder=2,
        )
        ax.plot(
            [max_step * 0.992],
            [human_mean],
            linestyle="none",
            marker=marker,
            markersize=3.8 if compact else 4.4,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.0,
            alpha=0.9,
            clip_on=True,
            zorder=4,
        )
        plotted_subjects.append(subject)
        summary.append(
            {
                "game": game,
                "subject": subject,
                "source_column": column,
                "n_points": int(len(valid)),
                "first_step": int(valid["step"].iloc[0]),
                "last_step": int(valid["step"].iloc[-1]),
                "minimum_return": float(valid["return"].min()),
                "maximum_return": float(valid["return"].max()),
                "final_observed_return": float(valid["return"].iloc[-1]),
                "human_mean_recording_defined_segment_return": human_mean,
                "human_n_recording_defined_segments": int(
                    human_row["n_segments"]
                ),
                "human_n_nonterminal_final_segments": int(
                    human_row["n_nonterminal_final_segments"]
                ),
                "human_reference_definition": str(
                    human_row["reference_definition"]
                ),
            }
        )

    if not plotted_subjects:
        raise ValueError(f"No usable learning curves found for game={game}")

    ax.set_xlim(0, max_step)
    ax.xaxis.set_major_locator(MultipleLocator(500_000))
    ax.xaxis.set_major_formatter(FuncFormatter(format_training_step))
    label_size = 10.0 if compact else 11.5
    ax.set_xlabel("Training Step", fontsize=label_size)
    ax.set_ylabel("Return", fontsize=label_size)
    ax.set_title(
        title,
        fontsize=13 if compact else 16,
        pad=11 if compact else 15,
        loc="center",
    )
    ax.grid(
        True,
        which="major",
        color="#D8DDE3",
        linewidth=0.65,
        alpha=0.75,
    )
    if show_legend:
        ax.legend(
            handles=learning_curve_legend_handles(plotted_subjects),
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=4,
            frameon=False,
            fontsize=9.5 if compact else 10.5,
        )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(labelsize=8.5 if compact else 10)
    return summary, plotted_subjects


def plot_game(
    frame: pd.DataFrame,
    series: Dict[Tuple[str, int], str],
    human_lookup: Mapping[Tuple[str, int], Mapping[str, object]],
    game: str,
    max_step: int,
    output_path: Path,
) -> List[Dict[str, object]]:
    """Plot subject-matched Thinker curves and human references for one game."""
    fig, ax = plt.subplots(figsize=(12.0, 6.8), facecolor="white")
    summary, _ = draw_learning_curve_axis(
        ax,
        frame,
        series,
        human_lookup,
        game,
        max_step,
        title=f"{GAME_TITLES[game]} · Per-subject training calibration",
        show_legend=True,
    )
    fig.subplots_adjust(
        left=0.09,
        right=0.98,
        bottom=0.27,
        top=0.88,
    )
    fig.text(
        0.5,
        0.015,
        (
            "Human recording-defined segment mean; termination/horizon not "
            "matched to Thinker training episode. Descriptive reference only."
        ),
        ha="center",
        va="bottom",
        fontsize=9.0,
        color="#5B6470",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    return summary


def draw_action_distribution_axis(
    ax: plt.Axes,
    action_counts: pd.DataFrame,
    game: str,
    subject: int,
    show_x_label: bool,
    show_y_label: bool,
) -> None:
    """Draw one subject's paired human/replay-recommendation distribution."""
    selected = action_counts.loc[
        (action_counts["game"] == game)
        & (action_counts["subject"] == subject)
    ]
    expected_sources = {
        "human_observed",
        "thinker_replay_recommendation",
    }
    if set(selected["source"]) != expected_sources:
        raise ValueError(
            f"Incomplete action sources for {game}, subject={subject}"
        )

    def source_values(source: str) -> np.ndarray:
        source_rows = selected.loc[selected["source"] == source].sort_values(
            "action_id"
        )
        if source_rows["action_id"].tolist() != list(
            range(len(ACTION_LABELS))
        ):
            raise ValueError(
                f"Invalid action ids for {game}, subject={subject}, {source}"
            )
        return source_rows["proportion"].to_numpy(dtype=float)

    human_prop = source_values("human_observed")
    thinker_prop = source_values("thinker_replay_recommendation")
    first_row = selected.iloc[0]
    n_paired = int(first_row["n_paired_rows"])
    agreement = float(first_row["exact_action_agreement"])
    total_variation = float(first_row["total_variation_distance"])

    x = np.arange(len(ACTION_LABELS))
    width = 0.37
    ax.bar(
        x - width / 2,
        human_prop,
        width,
        color=HUMAN_COLOR,
        alpha=0.86,
        edgecolor="white",
        linewidth=0.5,
        label="Human action",
    )
    ax.bar(
        x + width / 2,
        thinker_prop,
        width,
        color=THINKER_COLOR,
        alpha=0.86,
        edgecolor="white",
        linewidth=0.5,
        label="Thinker action during replay",
    )
    ax.set_xlim(-0.65, len(ACTION_LABELS) - 0.35)
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(
        ACTION_LABELS,
        rotation=35,
        ha="right",
        rotation_mode="anchor",
        fontsize=6.6,
    )
    ax.set_xlabel("Action" if show_x_label else "", fontsize=8.5)
    ax.set_ylabel("Proportion" if show_y_label else "", fontsize=8.5)
    ax.set_title(f"S{subject}", fontsize=10.5, pad=4)
    ax.grid(
        axis="y",
        color="#D8DDE3",
        linewidth=0.65,
        alpha=0.75,
    )
    ax.text(
        0.97,
        0.96,
        f"paired n={n_paired:,}\nagreement={agreement:.2f}; TV={total_variation:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.8,
        color="#5B6470",
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", labelsize=7.2, labelleft=show_y_label)


def plot_combined_imitation_results(
    frame: pd.DataFrame,
    series: Dict[Tuple[str, int], str],
    human_lookup: Mapping[Tuple[str, int], Mapping[str, object]],
    action_counts: pd.DataFrame,
    max_step: int,
    output_path: Path,
) -> None:
    """Create two game rows: learning curves plus subject action facets."""
    games = [
        game
        for game in ("pong", "spaceinvaders")
        if any(series_game == game for series_game, _ in series)
    ]
    if not games:
        raise ValueError("No supported games available for combined figure")
    all_subjects = sorted({subject for _, subject in series})

    fig = plt.figure(
        figsize=(22.5, 6.6 * len(games) + 2.0),
        facecolor="white",
    )
    outer = fig.add_gridspec(
        len(games),
        2,
        width_ratios=(1.08, 1.92),
        left=0.055,
        right=0.99,
        bottom=0.135,
        top=0.90,
        hspace=0.40,
        wspace=0.16,
    )

    for game_row, game in enumerate(games):
        subjects = sorted(
            subject for series_game, subject in series if series_game == game
        )
        curve_ax = fig.add_subplot(outer[game_row, 0])
        draw_learning_curve_axis(
            curve_ax,
            frame,
            series,
            human_lookup,
            game,
            max_step,
            title=f"{GAME_TITLES[game]} · Performance calibration",
            show_legend=False,
            compact=True,
        )

        n_columns = 3
        n_rows = int(math.ceil(len(subjects) / n_columns))
        right_grid = outer[game_row, 1].subgridspec(
            n_rows + 1,
            n_columns,
            height_ratios=[0.12, *([1.0] * n_rows)],
            hspace=0.55,
            wspace=0.24,
        )
        header_ax = fig.add_subplot(right_grid[0, :])
        header_ax.set_axis_off()
        header_ax.text(
            0.5,
            0.5,
            (
                f"{GAME_TITLES[game]} · Action distribution on aligned "
                "teacher-forced replay rows"
            ),
            ha="center",
            va="center",
            fontsize=12.5,
            color="#202020",
        )
        for subject_position in range(n_rows * n_columns):
            row = subject_position // n_columns + 1
            column = subject_position % n_columns
            action_ax = fig.add_subplot(right_grid[row, column])
            if subject_position >= len(subjects):
                action_ax.set_axis_off()
                continue
            subject = subjects[subject_position]
            draw_action_distribution_axis(
                action_ax,
                action_counts,
                game,
                subject,
                show_x_label=(row == n_rows),
                show_y_label=(column == 0),
            )

    action_handles = [
        Patch(facecolor=HUMAN_COLOR, alpha=0.86, label="Human action"),
        Patch(
            facecolor=THINKER_COLOR,
            alpha=0.86,
            label="Thinker action during replay",
        ),
    ]
    fig.legend(
        handles=learning_curve_legend_handles(all_subjects),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.952),
        ncol=min(len(all_subjects) + 2, 8),
        frameon=False,
        fontsize=10,
        handlelength=2.2,
        columnspacing=1.4,
    )
    fig.legend(
        handles=action_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.030),
        ncol=2,
        frameon=False,
        fontsize=10.5,
        handlelength=1.8,
        columnspacing=2.2,
    )
    fig.suptitle(
        "Per-subject imitation-learning calibration",
        x=0.5,
        y=0.995,
        fontsize=17,
        color="#202020",
    )
    fig.text(
        0.5,
        0.003,
        (
            "Human recording-defined segment mean; termination/horizon not "
            "matched to Thinker training episode. Actions are teacher-forced "
            "replay recommendations. Training-data calibration only—not "
            "independent held-out or free-running behavioral validation."
        ),
        ha="center",
        va="bottom",
        fontsize=9.2,
        color="#5B6470",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot per-subject imitated-Thinker learning calibration with "
            "descriptive human references and paired replay actions."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--behavioral-root",
        type=Path,
        default=DEFAULT_BEHAVIORAL_ROOT,
        help=(
            "Atari behavioral NPZ root used for human return summaries "
            f"(default: {DEFAULT_BEHAVIORAL_ROOT})"
        ),
    )
    parser.add_argument(
        "--action-cache-root",
        type=Path,
        default=DEFAULT_ACTION_CACHE_ROOT,
        help=(
            "Canonical 00_cache root containing paired actor-policy real-step "
            f"tables (default: {DEFAULT_ACTION_CACHE_ROOT})"
        ),
    )
    parser.add_argument(
        "--max-step",
        type=int,
        default=DEFAULT_MAX_STEP,
        help="Maximum training step shown on the x-axis.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_step <= 0:
        raise ValueError("--max-step must be positive")
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    behavioral_root = args.behavioral_root.expanduser().resolve()
    action_cache_root = args.action_cache_root.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if not behavioral_root.exists():
        raise FileNotFoundError(behavioral_root)
    if not action_cache_root.exists():
        raise FileNotFoundError(action_cache_root)

    configure_matplotlib()
    header = pd.read_csv(input_path, nrows=0)
    series = discover_series(list(header.columns))
    observed_pairs = set(series)
    if observed_pairs != EXPECTED_CANONICAL_PAIRS:
        missing_pairs = sorted(EXPECTED_CANONICAL_PAIRS - observed_pairs)
        unexpected_pairs = sorted(observed_pairs - EXPECTED_CANONICAL_PAIRS)
        raise ValueError(
            "Canonical learning-curve input must contain exactly Pong and "
            "Space Invaders for subjects S1-S6; "
            f"missing={missing_pairs}, unexpected={unexpected_pairs}"
        )
    use_columns = ["Step", *series.values()]
    frame = pd.read_csv(input_path, usecols=use_columns)
    frame["Step"] = pd.to_numeric(frame["Step"], errors="coerce")
    frame = frame.dropna(subset=["Step"]).copy()
    frame["Step"] = frame["Step"].astype(int)
    pairs = sorted(
        series,
        key=lambda pair: (GAME_IDS[pair[0]], pair[1]),
    )
    games = [
        game
        for game in ("pong", "spaceinvaders")
        if any(series_game == game for series_game, _ in pairs)
    ]

    (
        human_segments,
        human_segment_summary,
        human_blocks,
        human_block_summary,
    ) = load_human_return_summaries(behavioral_root, pairs)
    human_lookup = make_human_performance_lookup(human_segment_summary)
    action_counts, action_qa = load_action_distribution_cache(
        action_cache_root,
        pairs,
    )

    figure_dir = output_dir / "figures"
    result_dir = output_dir / "results"
    figure_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    all_summary: List[Dict[str, object]] = []
    output_files = {
        "pong": figure_dir / "fig_pong_rmean_episode_return.png",
        "spaceinvaders": (
            figure_dir / "fig_spaceinvaders_rmean_episode_return.png"
        ),
        "combined": (
            figure_dir / "fig_imitation_learning_results_combined.png"
        ),
    }
    for game in games:
        all_summary.extend(
            plot_game(
                frame,
                series,
                human_lookup,
                game,
                args.max_step,
                output_files[game],
            )
        )
        print(f"Saved {output_files[game]}", flush=True)

    plot_combined_imitation_results(
        frame,
        series,
        human_lookup,
        action_counts,
        args.max_step,
        output_files["combined"],
    )
    print(f"Saved {output_files['combined']}", flush=True)

    summary_frame = pd.DataFrame(all_summary)
    table_files = {
        "learning_curve_series_summary": (
            result_dir / "learning_curve_series_summary.csv"
        ),
        "human_terminal_split_segments": (
            result_dir / "human_terminal_split_segments.csv"
        ),
        "human_terminal_split_segment_summary": (
            result_dir / "human_terminal_split_segment_summary.csv"
        ),
        "human_480s_block_returns": (
            result_dir / "human_480s_block_returns.csv"
        ),
        "human_480s_block_summary": (
            result_dir / "human_480s_block_summary.csv"
        ),
        "subject_action_distribution_counts": (
            result_dir / "subject_action_distribution_counts.csv"
        ),
        "action_distribution_pair_qa": (
            result_dir / "action_distribution_pair_qa.csv"
        ),
    }
    summary_frame.to_csv(
        table_files["learning_curve_series_summary"],
        index=False,
    )
    human_segments.to_csv(
        table_files["human_terminal_split_segments"],
        index=False,
    )
    human_segment_summary.to_csv(
        table_files["human_terminal_split_segment_summary"],
        index=False,
    )
    human_blocks.to_csv(
        table_files["human_480s_block_returns"],
        index=False,
    )
    human_block_summary.to_csv(
        table_files["human_480s_block_summary"],
        index=False,
    )
    action_counts.to_csv(
        table_files["subject_action_distribution_counts"],
        index=False,
    )
    action_qa.to_csv(
        table_files["action_distribution_pair_qa"],
        index=False,
    )
    # Do not leave the obsolete sub003-only aggregate beside the new complete
    # subject-level table after an official rerun.
    (result_dir / "sub003_action_distribution_counts.csv").unlink(
        missing_ok=True
    )
    manifest = {
        "analysis_scope": (
            "training-data calibration/sanity check; not independent held-out "
            "or free-running behavioral validation"
        ),
        "source_csv": str(input_path),
        "behavioral_root": str(behavioral_root),
        "action_cache_root": str(action_cache_root),
        "metric": "imitated Thinker rmean_episode_return",
        "max_step": args.max_step,
        "additional_smoothing": False,
        "canonical_pair_contract": {
            "required_exact_pairs": [
                {"game": game, "subject": subject}
                for game, subject in sorted(
                    EXPECTED_CANONICAL_PAIRS,
                    key=lambda pair: (GAME_IDS[pair[0]], pair[1]),
                )
            ],
            "policy": (
                "fail closed before data loading when any required pair is "
                "missing or any unexpected pair is present"
            ),
        },
        "learning_curves": {
            "unit": "one solid curve per imitated subject",
            "aggregation": "none across subjects",
            "source_metric": "exported rmean_episode_return",
            "checkpoint_alignment": (
                "not established: the 00_cache source manifests do not identify "
                "the replay checkpoint, and CSV run names are not proof of the "
                "checkpoint used to generate replay traces"
            ),
        },
        "human_descriptive_reference": {
            "plotted_statistic": (
                "per-subject mean return across recording-defined terminal-split "
                "segments of at least five steps"
            ),
            "visible_label": (
                "Human recording-defined segment mean; termination/horizon not "
                "matched to Thinker training episode"
            ),
            "episode_split": (
                "same is_terminal boundary algorithm as 01_behavioral_analysis.py; "
                "a non-terminal final fragment would be retained when >=5 steps"
            ),
            "comparability_caveat": (
                "behavioral recording segments and Thinker training episodes do "
                "not have established matching horizon/termination semantics; "
                "the dashed lines are descriptive references only"
            ),
            "block_audit": (
                "separate approximately 480-s block returns and durations are "
                "saved but deliberately not plotted on the learning-curve axis"
            ),
        },
        "action_distribution": {
            "subjects": sorted({int(subject) for _, subject in pairs}),
            "definition": (
                "real_step_id deduplication, cross_chunk_boundary==0, then "
                "jointly valid human/Thinker action ids 0..5 in "
                "actor_policy/real_step_metrics.csv.gz"
            ),
            "source": (
                "canonical subject x game 00_cache actor-policy real-step tables"
            ),
            "thinker_action_semantics": (
                "model recommendation during teacher-forced replay of the human "
                "trajectory; not a free-running generative action distribution"
            ),
            "cache_scope_caveat": (
                "the aligned cache starts after initial rows required by its "
                "real-step feature contract; distributions therefore describe "
                "cached paired decision rows, not every raw status==0 row"
            ),
        },
        "subjects_by_game": {
            game: sorted(
                int(item["subject"])
                for item in all_summary
                if item["game"] == game
            )
            for game in games
        },
        "figures": {
            name: str(path)
            for name, path in output_files.items()
        },
        "tables": {
            name: str(path)
            for name, path in table_files.items()
        },
    }
    (result_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
