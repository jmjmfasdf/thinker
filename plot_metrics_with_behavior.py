"""Combine exported Thinker metrics with behavioral similarity overlays."""

import argparse
import glob
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import jensenshannon

import fig_pong
from behavior_similarity_copy import (
    THINKER_SCORE_PLACEHOLDER,
    StepMetrics as BehaviorStepMetrics,
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


def _to_step_number(step_token: str) -> Optional[int]:
    """Convert filenames' scientific notation tokens to integer training steps."""

    if step_token is None:
        return None
    token = step_token.replace("E", "e")
    try:
        return int(float(token))
    except (TypeError, ValueError):
        return None


def _merge_action_counts(existing: Optional[np.ndarray], actions: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=int)
    actions = actions[actions >= 0]
    if actions.size == 0:
        return existing if existing is not None else np.zeros((0,), dtype=np.int64)
    max_action = int(actions.max())
    counts = np.bincount(actions, minlength=max_action + 1).astype(np.int64, copy=False)
    if existing is None or existing.size == 0:
        return counts
    if counts.size > existing.size:
        existing = np.pad(existing, (0, counts.size - existing.size))
    elif counts.size < existing.size:
        counts = np.pad(counts, (0, existing.size - counts.size))
    return existing + counts


def _normalize_counts(counts: np.ndarray, size: int) -> Optional[np.ndarray]:
    if counts.size == 0:
        return None
    if counts.size < size:
        counts = np.pad(counts, (0, size - counts.size))
    elif counts.size > size:
        counts = counts[:size]
    total = float(np.sum(counts))
    if total <= 0.0:
        return None
    return counts.astype(float, copy=False) / total


def _compute_js_similarity(model_counts: np.ndarray, human_counts: np.ndarray) -> float:
    size = max(model_counts.size, human_counts.size)
    if size == 0:
        return math.nan
    model_dist = _normalize_counts(model_counts, size)
    human_dist = _normalize_counts(human_counts, size)
    if model_dist is None or human_dist is None:
        return math.nan
    dist = jensenshannon(model_dist, human_dist, base=2.0)
    if not np.isfinite(dist):
        return math.nan
    similarity = 1.0 - float(dist)
    return float(np.clip(similarity, 0.0, 1.0))


def _compute_noop_frequency(counts: np.ndarray) -> float:
    if counts.size == 0:
        return math.nan
    total = float(np.sum(counts))
    if total <= 0.0:
        return math.nan
    noop_count = float(counts[0]) if counts.size > 0 else 0.0
    return noop_count / total


def _collect_behavior_differences(subject: int,
                                  game_number: int,
                                  metrics_dir: Path,
                                  human_dir: Path) -> Tuple[
                                      Tuple[BehaviorStepMetrics, ...],
                                      Dict[str, Dict[int, Tuple[float, float]]],
                                      Dict[str, Dict[int, Tuple[float, float]]],
                                  ]:
    """Compute noop and score differences per Thinker step grouped by game."""

    human = _load_human_metrics(subject, game_number, human_dir)
    thinker_files: Dict[str, list] = {}
    action_counts_by_step: Dict[str, np.ndarray] = {}

    metric_paths = sorted(
        glob.glob(os.path.join(metrics_dir, "*.npz"))
        + glob.glob(os.path.join(metrics_dir, "*.npy"))
    )
    for file_path in metric_paths:
        filename = os.path.basename(file_path)
        stem, _ = os.path.splitext(filename)  # e.g. spaceinvaders_1e6_0
        parts = stem.split("_")
        if len(parts) < 3:
            continue
        game_token = "_".join(parts[:-2])
        step_token = parts[-2]
        step_key = f"{game_token}_{step_token}"

        try:
            metrics = _load_metrics_file(file_path)
        except Exception as exc:  # pragma: no cover - diagnostics
            print(f"Failed to load metrics from {file_path}: {exc}")
            continue

        actions = np.asarray(metrics.get("action", []), dtype=int)
        if actions.size == 0:
            continue
        valid_mask = actions >= 0
        if not np.any(valid_mask):
            continue
        filtered_actions = actions[valid_mask]
        if filtered_actions.size == 0:
            continue
        noop_ratio = float(np.mean(filtered_actions == 0))
        score = float(THINKER_SCORE_PLACEHOLDER.get(stem, math.nan))
        action_counts_by_step[step_key] = _merge_action_counts(
            action_counts_by_step.get(step_key),
            filtered_actions,
        )

        # Minimal container matching the interface expected by _build_step_metrics.
        thinker_files.setdefault(step_key, []).append(
            type("Metric", (), {"noop_ratio": noop_ratio, "score": score})()
        )

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
            score_diff =  sm.thinker_mean_score - sm.human_mean_score

        per_game[game_token][step_number] = (noop_diff, score_diff)

    per_game_action_stats: Dict[str, Dict[int, Tuple[float, float]]] = defaultdict(dict)
    human_action_counts = getattr(human, "action_counts", np.zeros((0,), dtype=np.int64))
    for step_key, counts in action_counts_by_step.items():
        try:
            game_token, step_token = step_key.rsplit("_", 1)
        except ValueError:
            continue
        step_number = _to_step_number(step_token)
        if step_number is None:
            continue

        js_similarity = _compute_js_similarity(counts, human_action_counts)
        noop_frequency = _compute_noop_frequency(counts)
        if np.isfinite(js_similarity) or np.isfinite(noop_frequency):
            per_game_action_stats[game_token][step_number] = (js_similarity, noop_frequency)

    return step_metrics, per_game, per_game_action_stats


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


def _plot_combined_behavior_metrics(gamename: str,
                                    action_series: Dict[int, Tuple[float, float]],
                                    score_series: Optional[Dict[int, float]],
                                    outdir: Path) -> Optional[Path]:
    steps = sorted(
        set(action_series.keys()) | (set(score_series.keys()) if score_series else set())
    )
    if not steps:
        return None

    js_values = []
    noop_values = []
    score_values = []
    for step in steps:
        js_val, noop_val = action_series.get(step, (math.nan, math.nan))
        js_values.append(js_val)
        noop_values.append(noop_val)
        if score_series is None:
            score_values.append(math.nan)
        else:
            score_values.append(score_series.get(step, math.nan))

    js_arr = np.asarray(js_values, dtype=float)
    noop_arr = np.asarray(noop_values, dtype=float)
    score_arr = np.asarray(score_values, dtype=float)

    has_js = np.isfinite(js_arr).any()
    has_noop = np.isfinite(noop_arr).any()
    has_score = np.isfinite(score_arr).any()
    if not (has_js or has_noop or has_score):
        return None

    fig, ax_left = plt.subplots(figsize=(8, 8))
    lines = []

    if has_js:
        line_js, = ax_left.plot(
            steps, js_arr, marker="o", color="tab:blue", label="JSD of Action distribution"
        )
        lines.append(line_js)
    if has_noop:
        line_noop, = ax_left.plot(
            steps, noop_arr, marker="s", color="tab:green", label="Model NOOP frequency"
        )
        lines.append(line_noop)

    ax_left.set_xlabel("Training Steps", fontsize=13)
    ax_left.set_ylabel("JS similarity / NOOP frequency", fontsize=13)
    ax_left.set_ylim(0.0, 1.0)
    ax_left.grid(True, alpha=0.3)

    if has_score:
        ax_right = ax_left.twinx()
        line_score, = ax_right.plot(
            steps,
            score_arr,
            marker="^",
            color="tab:orange",
            label="Score difference (model - human)",
        )
        ax_right.set_ylabel("Score difference (model - human)", fontsize=13)
        lines.append(line_score)

    if lines:
        ax_left.legend(lines, [line.get_label() for line in lines], loc="upper left")

    ax_left.set_title(f"{gamename}: combined behavior metrics", fontsize=15)

    out_path = outdir / f"{gamename}_combined_metrics.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


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
    step_metrics, behavior_map, action_stats = _collect_behavior_differences(
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

        fig_pong.create_step_analysis_plots(
            gamename,
            step_data,
            str(outdir),
            behavior_series=behavior_series,
            score_series=score_series,
        )

        action_series = None
        if gamename in action_stats:
            action_series = {
                step: values
                for step, values in action_stats[gamename].items()
                if np.isfinite(values[0]) or np.isfinite(values[1])
            }
        if action_series:
            combined_path = _plot_combined_behavior_metrics(
                gamename, action_series, score_series, outdir
            )
            if combined_path:
                print(f"Saved combined behavior plot to {combined_path}")

    print(f"\nAll combined plots saved to {outdir}.")


if __name__ == "__main__":
    main()
