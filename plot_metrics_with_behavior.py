"""Combine exported Thinker metrics with behavioral similarity overlays."""

import argparse
import glob
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from behavior_similarity_copy import (
    THINKER_SCORE_PLACEHOLDER,
    StepMetrics as BehaviorStepMetrics,
    ThinkerFileMetric,
    _build_step_metrics,
    _load_human_metrics,
    _plot_metrics,
    _summarize,
)
from plot_exported_metrics import (
    CORRELATION_KEYS,
    _combine_fisher_stouffer,
    _load_metrics_file,
    _parse_metrics_filename,
    calculate_metrics_correlations,
)

CORRELATION_LABELS = {
    "noop_freq_vs_planning_depth": "Noop freq vs planning depth",
    "noop_freq_vs_image_similarity": "Noop freq vs image similarity",
    "real_step_image_sim_vs_planning_depth": "Image similarity vs planning depth",
    "real_step_image_sim_vs_action_diversity": "Image similarity vs action diversity",
    "noop_freq_vs_action_diversity": "Noop freq vs action diversity",
    "planning_depth_vs_action_diversity": "Planning depth vs action diversity",
    "real_step_image_sim_vs_imagination_diversity": "Image similarity vs imagination diversity",
    "noop_freq_vs_imagination_diversity": "Noop freq vs imagination diversity",
    "planning_depth_vs_imagination_diversity": "Planning depth vs imagination diversity",
}

DIFF_SERIES_LABELS = {
    "noop": "noop freq difference",
    "score": "score diff",
}


def _to_step_number(step_token: str) -> Optional[int]:
    """Convert filenames' scientific notation tokens to integer training steps."""

    if step_token is None:
        return None
    token = step_token.replace("E", "e")
    try:
        return int(float(token))
    except (TypeError, ValueError):
        return None


def _load_exported_step_metrics(metrics_dir: Path) -> Dict[str, List[ThinkerFileMetric]]:
    """Convert exported metric files into Thinker-style per-step aggregates."""

    per_step: Dict[str, List[ThinkerFileMetric]] = defaultdict(list)
    metrics_dir = Path(metrics_dir)
    metric_files = sorted(metrics_dir.glob("*.npz"))

    for path in metric_files:
        stem = path.stem
        parts = stem.split("_")
        if len(parts) < 3:
            continue
        game_token = "_".join(parts[:-2])
        step_token = parts[-2]
        step_key = f"{game_token}_{step_token}"

        try:
            metrics = _load_metrics_file(str(path))
        except RuntimeError as exc:
            print(f"Failed to load thinker metrics from {path}: {exc}")
            continue

        actions = metrics.get("action")
        if actions is None:
            continue

        actions_arr = np.asarray(actions).reshape(-1)
        if actions_arr.size == 0:
            continue

        finite_actions = actions_arr[np.isfinite(actions_arr)]
        if finite_actions.size == 0:
            continue

        noop_ratio = float(np.mean(finite_actions == 0))
        score = float(THINKER_SCORE_PLACEHOLDER.get(stem, math.nan))
        per_step[step_key].append(
            ThinkerFileMetric(path=path, noop_ratio=noop_ratio, score=score)
        )

    return per_step


def _collect_behavior_differences(subject: int,
                                  game_number: int,
                                  metrics_dir: Path,
                                  human_dir: Path) -> Tuple[
                                      Tuple[BehaviorStepMetrics, ...],
                                      Dict[str, Dict[int, Tuple[float, float]]],
                                  ]:
    """Compute noop and score differences per Thinker step grouped by game."""

    human = _load_human_metrics(subject, game_number, human_dir)
    thinker_files = _load_exported_step_metrics(metrics_dir)
    step_metrics = tuple(_build_step_metrics(human, thinker_files))

    per_game: Dict[str, Dict[int, Tuple[float, float]]] = defaultdict(dict)
    for sm in step_metrics:
        try:
            game_token, step_token = sm.step_key.rsplit("_", 1)
        except ValueError:
            continue
        step_number = _to_step_number(step_token)
        if step_number is None:
            continue

        noop_diff = math.nan
        if math.isfinite(sm.human_mean_noop) and math.isfinite(sm.thinker_mean_noop):
            noop_diff = sm.thinker_mean_noop - sm.human_mean_noop

        score_diff = math.nan
        if math.isfinite(sm.human_mean_score) and math.isfinite(sm.thinker_mean_score):
            score_diff = sm.thinker_mean_score - sm.human_mean_score

        per_game[game_token][step_number] = (noop_diff, score_diff)

    return step_metrics, per_game


def _collect_metric_correlations(folder: str,
                                 window_size: int,
                                 stride: int) -> Dict[str, Dict[int, Dict[str, dict]]]:
    """Gather metric correlations grouped by game and training step."""

    metric_files = sorted(glob.glob(os.path.join(folder, "*.npz")))
    if not metric_files:
        return {}

    file_groups = defaultdict(lambda: defaultdict(list))
    for file_path in metric_files:
        filename = os.path.basename(file_path)
        gamename, step, number = _parse_metrics_filename(filename)
        if gamename is None:
            continue
        file_groups[gamename][step].append((file_path, number))

    correlations_by_game: Dict[str, Dict[int, Dict[str, dict]]] = {}

    for gamename, step_groups in file_groups.items():
        step_data: Dict[int, Dict[str, dict]] = {}
        print(f"\n=== {gamename} ===")

        for step, files in sorted(step_groups.items()):
            print(f"  Step {step}: {len(files)} file(s)")
            correlations_per_key = {key: [] for key in CORRELATION_KEYS}

            for file_path, number in sorted(files, key=lambda item: item[1]):
                print(f"    Processing file {number} ({os.path.basename(file_path)})...")
                try:
                    metrics = _load_metrics_file(file_path)
                    correlations = calculate_metrics_correlations(metrics, window_size, stride)
                except Exception as exc:  # pragma: no cover - diagnostics
                    print(f"      Failed: {exc}")
                    continue

                for key, value in correlations.items():
                    if value is not None:
                        correlations_per_key[key].append(value)

            combined = {key: _combine_fisher_stouffer(correlations_per_key[key])
                        for key in CORRELATION_KEYS}
            step_data[step] = combined

        correlations_by_game[gamename] = step_data

    return correlations_by_game


def _plot_combined_step_summary(gamename: str,
                                step_data: Dict[int, Dict[str, dict]],
                                outdir: Path,
                                behavior_series: Optional[Dict[int, float]] = None,
                                score_series: Optional[Dict[int, float]] = None) -> Optional[Path]:
    """Render all correlation and behavioral differences into a single figure."""

    step_keys = set(step_data.keys())
    if behavior_series:
        step_keys.update(behavior_series.keys())
    if score_series:
        step_keys.update(score_series.keys())

    steps = sorted(step_keys)
    if not steps:
        print(f"No step data available to plot for {gamename}.")
        return None

    fig, ax_r = plt.subplots(figsize=(20, 8))
    ax_diff = ax_r.twinx()
    palette = plt.get_cmap("tab20").colors

    handles: List = []
    for idx, key in enumerate(CORRELATION_KEYS):
        series = []
        for step in steps:
            correlations = step_data.get(step)
            if not correlations:
                series.append(math.nan)
                continue
            corr_entry = correlations.get(key)
            if not corr_entry:
                series.append(math.nan)
            else:
                series.append(float(corr_entry.get("r", math.nan)))

        values = np.asarray(series, dtype=float)
        if np.all(np.isnan(values)):
            continue

        label = CORRELATION_LABELS.get(key, key)
        color = palette[idx % len(palette)]
        line, = ax_r.plot(
            steps,
            values,
            marker="o",
            linewidth=2,
            label=label,
            color=color,
        )
        handles.append(line)

    diff_handles: List = []
    if behavior_series:
        noop_values = np.asarray(
            [behavior_series.get(step, math.nan) for step in steps], dtype=float
        )
        if not np.all(np.isnan(noop_values)):
            line, = ax_r.plot(
                steps,
                noop_values,
                marker="s",
                linewidth=2,
                linestyle="--",
                color="#2ca02c",
                label=DIFF_SERIES_LABELS["noop"],
            )
            diff_handles.append(line)

    if score_series:
        score_values = np.asarray(
            [score_series.get(step, math.nan) for step in steps], dtype=float
        )
        if not np.all(np.isnan(score_values)):
            line, = ax_diff.plot(
                steps,
                score_values,
                marker="^",
                linewidth=2,
                linestyle=":",
                color="#d62728",
                label=DIFF_SERIES_LABELS["score"],
            )
            diff_handles.append(line)

    if not handles and not diff_handles:
        plt.close(fig)
        print(f"No plottable correlations for {gamename}.")
        return None

    ax_r.axhline(0.0, color="#666666", linewidth=0.8, linestyle="--")
    ax_r.set_ylabel("Pearson r / noop freq diff")
    ax_r.set_xlabel("Training steps")
    ax_r.set_ylim(-1.05, 1.05)
    ax_r.grid(True, axis="y", alpha=0.3)

    ax_diff.set_ylabel("Score diff (model - human)")

    legend_handles = handles + diff_handles
    legend_labels = [handle.get_label() for handle in legend_handles]
    ax_r.legend(legend_handles, legend_labels, loc="center left", bbox_to_anchor=(1.2, 0.5))
    fig.suptitle(f"{gamename}: correlations and behavioral differences")
    fig.tight_layout(rect=[0, 0.04, 0.7, 0.94])

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    output_path = outdir / f"{gamename}_combined_metrics.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recreate exported Thinker metric plots and overlay human similarity "
            "statistics in a single run."
        )
    )
    parser.add_argument("--folder", required=True, help="Folder containing exported metric files")
    parser.add_argument("--outdir", required=True, help="Directory to store combined correlation plots")
    parser.add_argument("--windowsize", type=int, default=150,
                        help="Sliding window size for windowed correlations (default: 150)")
    parser.add_argument("--stride", type=int, default=1,
                        help="Stride between sliding windows (default: 1)")
    parser.add_argument("--subject", type=int, required=True, help="Human subject id (1-6)")
    parser.add_argument("--game-number", type=int, required=True, help="Game number (0-2)")
    parser.add_argument("--human-dir", type=Path, required=True,
                        help="Root directory for human behavioral data")
    parser.add_argument("--behavior-outdir", "--output-dir", dest="behavior_outdir",
                        help="Directory to store behavior-only plots (default: same as --outdir)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    behavior_outdir = Path(args.behavior_outdir) if args.behavior_outdir else outdir
    behavior_outdir.mkdir(parents=True, exist_ok=True)

    print("Collecting behavior similarity metrics...")
    step_metrics, behavior_map = _collect_behavior_differences(
        args.subject,
        args.game_number,
        Path(args.folder),
        args.human_dir,
    )

    if step_metrics:
        behavior_plot_path = _plot_metrics(step_metrics, behavior_outdir,
                                           args.subject, args.game_number)
        _summarize(step_metrics)
        print(f"Saved behavior similarity plot to {behavior_plot_path}")
    else:
        print("No Thinker data available after filtering status == 0")

    print("\nAggregating exported metric correlations...")
    correlations_by_game = _collect_metric_correlations(args.folder, args.windowsize, args.stride)
    if not correlations_by_game:
        print(f"No metric files found in {args.folder}.")
        return

    for gamename, step_data in correlations_by_game.items():
        behavior_series: Optional[Dict[int, float]] = None
        score_series: Optional[Dict[int, float]] = None

        if gamename in behavior_map:
            behavior_series = {
                step: diffs[0]
                for step, diffs in behavior_map[gamename].items()
                if diffs[0] == diffs[0]  # filter NaN
            }
            score_series = {
                step: diffs[1]
                for step, diffs in behavior_map[gamename].items()
                if diffs[1] == diffs[1]
            }

        combined_path = _plot_combined_step_summary(
            gamename,
            step_data,
            outdir,
            behavior_series=behavior_series,
            score_series=score_series,
        )
        if combined_path:
            print(f"Saved combined plot to {combined_path}")

    print(f"\nAll combined plots saved to {outdir}.")


if __name__ == "__main__":
    main()
