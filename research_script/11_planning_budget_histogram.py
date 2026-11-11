#!/usr/bin/env python3
"""Plot status-2 planning budgets per complete real-step window.

The public legacy projection assigns status 0 to ``initial_root`` and
``real_transition`` events and status 2 to ``search_computation`` events.
For each native episode, a planning budget is therefore the number of search
events strictly between two consecutive real-event anchors.  The terminal
anchor is excluded because it has no following real event.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import zarr


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAPTURE_PARENT = ROOT / "logs" / "visual_behav" / "full"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "outputs"
    / "planning_budget_histogram"
    / "status2_planning_budget_by_game.png"
)
CAPTURE_TAGS = {
    "Enduro": "enduro-sub-001-allsessions-step5000000-images-all-checkpoint",
    "Pong": "pong-sub-001-allsessions-step5000000-images-all-checkpoint",
}
COLORS = {"Enduro": "#4C78A8", "Pong": "#F58518"}


def read_bool_field(
    root: zarr.hierarchy.Group,
    schema: Mapping[str, Mapping[str, object]],
    name: str,
) -> np.ndarray:
    descriptor = schema[name]
    packed = root["packs"][str(descriptor["storage_group"])]
    values = np.asarray(packed[:, int(descriptor["storage_index"])])
    return values.reshape(len(packed), -1)[:, 0].astype(np.bool_, copy=False)


def episode_budgets(store: Path) -> np.ndarray:
    attrs = json.loads((store / ".zattrs").read_text(encoding="utf-8"))
    schema = {item["name"]: item for item in attrs["field_schema"]}
    required = {"initial_root", "search_computation", "real_transition"}
    missing = required.difference(schema)
    if missing:
        raise ValueError(f"{store}: missing native status fields {sorted(missing)}")

    root = zarr.open_consolidated(str(store), mode="r")
    initial = read_bool_field(root, schema, "initial_root")
    search = read_bool_field(root, schema, "search_computation")
    real = read_bool_field(root, schema, "real_transition")
    overlap = initial.astype(np.int8) + search.astype(np.int8) + real.astype(np.int8)
    if np.any(overlap > 1):
        raise ValueError(f"{store}: native event classes overlap")

    anchors = np.flatnonzero(initial | real)
    if anchors.size == 0 or anchors[0] != 0 or np.count_nonzero(initial) != 1:
        raise ValueError(f"{store}: malformed real-event anchors")
    prefix = np.r_[0, np.cumsum(search, dtype=np.int64)]
    return np.asarray(
        [prefix[stop] - prefix[start + 1] for start, stop in zip(anchors[:-1], anchors[1:])],
        dtype=np.int64,
    )


def game_budgets(capture: Path) -> tuple[np.ndarray, int, int]:
    completed = json.loads((capture / "COMPLETED.json").read_text(encoding="utf-8"))
    if completed.get("status") != "COMPLETED":
        raise ValueError(f"capture is not complete: {capture}")
    manifest = json.loads((capture / "manifest.json").read_text(encoding="utf-8"))
    arrays = [episode_budgets(capture / episode["path"]) for episode in manifest["episodes"]]
    if not arrays:
        raise ValueError(f"capture has no episodes: {capture}")
    budgets = np.concatenate(arrays)
    source_count = len({episode["metadata"]["source_id"] for episode in manifest["episodes"]})
    return budgets, source_count, len(arrays)


def plot_histograms(data: Mapping[str, np.ndarray], output: Path) -> None:
    maximum = max(int(values.max()) for values in data.values())
    x = np.arange(maximum + 1)
    all_percent = {
        game: 100.0 * np.bincount(values, minlength=maximum + 1) / len(values)
        for game, values in data.items()
    }
    positive_percent = {}
    for game, values in data.items():
        positive = values[values > 0]
        positive_percent[game] = (
            100.0 * np.bincount(positive, minlength=maximum + 1) / len(positive)
        )

    figure, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey="row")
    for column, (game, budgets) in enumerate(data.items()):
        color = COLORS[game]
        positive = budgets[budgets > 0]

        top = axes[0, column]
        top.bar(x, all_percent[game], width=0.86, color=color, edgecolor="white", linewidth=0.45)
        top.axvline(float(budgets.mean()), color="#222222", linestyle="--", linewidth=1.3)
        top.set_title(game, fontsize=15, weight="bold", pad=9)
        top.text(
            0.98,
            0.96,
            "\n".join(
                (
                    f"complete real steps: {len(budgets):,}",
                    f"mean budget: {budgets.mean():.3f}",
                    f"planning steps: {100 * len(positive) / len(budgets):.1f}%",
                )
            ),
            transform=top.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.92},
        )

        bottom = axes[1, column]
        bottom.bar(
            x[1:],
            positive_percent[game][1:],
            width=0.86,
            color=color,
            edgecolor="white",
            linewidth=0.45,
        )
        bottom.set_yscale("log")
        bottom.text(
            0.98,
            0.96,
            f"mean | budget > 0: {positive.mean():.3f}\nmedian: {np.median(positive):.0f}   max: {positive.max():.0f}",
            transform=bottom.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.92},
        )

    axes[0, 0].set_ylabel("All complete real steps (%)")
    axes[1, 0].set_ylabel("Planned real steps (%) · log scale")
    for axis in axes[1]:
        axis.set_xlabel("Planning budget (number of status-2 events)")
    for axis in axes.flat:
        axis.set_xlim(-0.6, maximum + 0.6)
        axis.xaxis.set_major_locator(MultipleLocator(5))
        axis.xaxis.set_minor_locator(MultipleLocator(1))
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.75)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)

    figure.suptitle("Planning-budget distributions by game", fontsize=19, weight="bold", y=0.985)
    figure.text(
        0.5,
        0.948,
        "Budget = status-2 events between consecutive status-0 real anchors; terminal incomplete windows excluded",
        ha="center",
        fontsize=10.5,
        color="#444444",
    )
    figure.text(
        0.5,
        0.012,
        "Top: zero-inclusive distribution. Bottom: conditional distribution among real steps with budget ≥ 1.",
        ha="center",
        fontsize=10,
        color="#555555",
    )
    figure.tight_layout(rect=(0.035, 0.045, 0.99, 0.93), h_pad=2.0, w_pad=1.4)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-parent", type=Path, default=DEFAULT_CAPTURE_PARENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    distributions = {}
    for game, tag in CAPTURE_TAGS.items():
        budgets, sources, episodes = game_budgets(args.capture_parent / tag)
        distributions[game] = budgets
        positive = budgets[budgets > 0]
        print(
            f"{game}: sources={sources} episodes={episodes} real_steps={len(budgets)} "
            f"mean={budgets.mean():.6f} planned={100 * len(positive) / len(budgets):.3f}% "
            f"conditional_mean={positive.mean():.6f} max={budgets.max()}"
        )
    plot_histograms(distributions, args.output)
    print(f"saved {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
