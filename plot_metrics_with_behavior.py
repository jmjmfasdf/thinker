"""Combine exported Thinker metrics with behavioral similarity overlays."""

import argparse
import glob
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

import fig_pong
from behavior_similarity_copy import (
    StepMetrics as BehaviorStepMetrics,
    _build_step_metrics,
    _load_human_metrics,
    _load_thinker_file_metrics,
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


def _to_step_number(step_token: str) -> Optional[int]:
    """Convert filenames' scientific notation tokens to integer training steps."""

    if step_token is None:
        return None
    token = step_token.replace("E", "e")
    try:
        return int(float(token))
    except (TypeError, ValueError):
        return None


def _collect_behavior_differences(subject: int,
                                  game_number: int,
                                  thinker_dir: Path,
                                  human_dir: Path) -> Tuple[
                                      Tuple[BehaviorStepMetrics, ...],
                                      Dict[str, Dict[int, Tuple[float, float]]],
                                  ]:
    """Compute noop and score differences per Thinker step grouped by game."""

    human = _load_human_metrics(subject, game_number, human_dir)
    thinker_files = _load_thinker_file_metrics(thinker_dir)
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
            noop_diff = abs(sm.human_mean_noop - sm.thinker_mean_noop)

        score_diff = math.nan
        if math.isfinite(sm.human_mean_score) and math.isfinite(sm.thinker_mean_score):
            score_diff = sm.human_mean_score - sm.thinker_mean_score

        per_game[game_token][step_number] = (noop_diff, score_diff)

    return step_metrics, per_game


def _collect_metric_correlations(folder: str,
                                 window_size: int,
                                 stride: int) -> Dict[str, Dict[int, Dict[str, dict]]]:
    """Gather metric correlations grouped by game and training step."""

    metric_files = sorted(
        glob.glob(os.path.join(folder, "*.npz"))
        + glob.glob(os.path.join(folder, "*.npy"))
    )
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
    parser.add_argument("--thinker-dir", type=Path, required=True,
                        help="Directory with Thinker *.npy files")
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
        args.thinker_dir,
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

        fig_pong.create_step_analysis_plots(
            gamename,
            step_data,
            str(outdir),
            behavior_series=behavior_series,
            score_series=score_series,
        )

    print(f"\nAll combined plots saved to {outdir}.")


if __name__ == "__main__":
    main()
