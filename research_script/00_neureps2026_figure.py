#!/usr/bin/env python3
"""Render the NeurReps 2026 figure set from frozen result artifacts.

This script performs plotting-time descriptive aggregation only.  It never fits a
model, permutes labels, bootstraps observations, or invokes an analysis script.
Exactly seven figures are written as paired vector PDF and 300-dpi PNG files.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import Collection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Patch, Rectangle
from matplotlib.text import Text
import nibabel as nib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
if not (ROOT / "outputs").is_dir():
    # Allows a temporary smoke-test copy while keeping the installed script
    # relocatable inside the repository.
    ROOT = Path("/home/jeongmin/thinker/research_script")
OUT = ROOT / "outputs"
BEH = OUT / "01_behavioral_analysis" / "atari"
WIDTH = 6.0

NAVY = "#23405c"
BLUE = "#367aa8"
SKY = "#74add1"
RED = "#b54845"
ORANGE = "#dd8a35"
GREEN = "#4d8a72"
PURPLE = "#78599d"
GRAY = "#73777b"
LIGHT = "#e9edf0"
INK = "#202326"
GAME_COLORS = {1: BLUE, 2: ORANGE}
GAME_NAMES = {1: "Pong", 2: "Space Invaders"}
TRANS_NAMES = {"noop_to_active": "hold → act", "active_to_noop": "act → hold"}
OUTCOME_NAMES = {
    "exact_action": "action switch",
    "exact_switch": "action switch",
    "noop_to_active": "hold → act",
    "active_to_noop": "act → hold",
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 6.6,
            "axes.titlesize": 7.4,
            "axes.labelsize": 6.6,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "legend.fontsize": 5.8,
            "axes.linewidth": 0.55,
            "axes.edgecolor": "#666666",
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def read(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required frozen artifact is missing: {path}")
    frame = pd.read_csv(path, usecols=columns)
    if frame.empty:
        raise ValueError(f"Required frozen artifact is empty: {path}")
    return frame


def require_columns(frame: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{source}: missing columns {missing}")


def finish_ax(ax: mpl.axes.Axes, grid: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis=grid, color="#d9dde0", lw=0.45, alpha=0.8, zorder=0)


def panel(ax: mpl.axes.Axes, letter: str, x: float = -0.16, y: float = 1.08) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top", ha="left")


def save(fig: mpl.figure.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", dpi=300, bbox_inches=None)
    # Freeze any constrained-layout solution computed for the vector page so
    # the second raster export reuses it without a second layout pass.
    if fig.get_layout_engine() is not None:
        fig.set_layout_engine("none")
    fig.savefig(path.with_suffix(".png"), format="png", dpi=300, bbox_inches=None)
    plt.close(fig)
    for legacy_panel in path.parent.glob(f"{path.stem}-*.png"):
        legacy_panel.unlink()


def enforce_minimum_visual_scale(
    fig: mpl.figure.Figure,
    *,
    font_points: float = 7.0,
    line_points: float = 0.8,
) -> None:
    """Raise visible figure artists to the journal-scale readability floor."""
    # A first draw materializes tick labels and colorbar artists.
    fig.canvas.draw()
    for artist in fig.findobj(Text):
        if artist.get_visible() and artist.get_text() and artist.get_fontsize() < font_points:
            artist.set_fontsize(font_points)
    for artist in fig.findobj(lambda item: isinstance(item, (Line2D, Patch))):
        if not artist.get_visible():
            continue
        linewidth = float(artist.get_linewidth())
        if 0 < linewidth < line_points:
            artist.set_linewidth(line_points)
    for artist in fig.findobj(Collection):
        if not artist.get_visible():
            continue
        linewidths = np.asarray(artist.get_linewidths(), dtype=float)
        if linewidths.size:
            artist.set_linewidths(
                np.where((linewidths > 0) & (linewidths < line_points),
                         line_points, linewidths)
            )
    for ax in fig.axes:
        ax.tick_params(width=line_points)
    fig.canvas.draw()


def weighted_mean(group: pd.DataFrame, value: str, weight: str) -> float:
    ok = np.isfinite(group[value]) & np.isfinite(group[weight]) & (group[weight] > 0)
    if not ok.any():
        return np.nan
    return float(np.average(group.loc[ok, value], weights=group.loc[ok, weight]))


def read_json_object(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Required frozen artifact is missing: {path}")
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def validate_framework_contract() -> dict[str, int]:
    """Validate the replay/search facts stated in the framework schematic."""
    cache_root = OUT / "00_cache"
    expected = {
        cache_root / f"sub{subject:03d}_game{game}" / "manifest.json"
        for subject in range(1, 7)
        for game in (1, 2)
    }
    observed = set(cache_root.glob("sub*_game*/manifest.json"))
    if observed != expected:
        missing = sorted(str(path) for path in expected - observed)
        extra = sorted(str(path) for path in observed - expected)
        raise ValueError(
            f"Framework cache-cell contract failed; missing={missing}, extra={extra}"
        )

    expected_roles = {
        "actor_initial": "first status-2 actor_policy in the search window",
        "actor_terminal": (
            "status-0 actor_policy at the next real step; sensitivity only"
        ),
        "model_root": "tree_reps.root_policy at the preceding real step",
    }
    for path in sorted(expected):
        payload = read_json_object(path)
        subject = int(payload.get("subject", -1))
        game = int(payload.get("game", -1))
        expected_name = f"sub{subject:03d}_game{game}"
        if path.parent.name != expected_name:
            raise ValueError(
                f"{path}: subject/game identity disagrees with directory name"
            )
        components = payload.get("components")
        if not isinstance(components, dict):
            raise ValueError(f"{path}: missing components object")
        actor = components.get("actor_policy")
        control = components.get("model_control")
        persistence = components.get("persistence_action_geometry")
        rollout = components.get("rollout_value")
        if not all(isinstance(item, dict) for item in (actor, control, persistence, rollout)):
            raise ValueError(f"{path}: incomplete framework components")

        for name, component in (
            ("actor_policy", actor),
            ("model_control", control),
            ("persistence_action_geometry", persistence),
            ("rollout_value", rollout),
        ):
            if int(component.get("n_runs", -1)) != 11:
                raise ValueError(f"{path}: {name}.n_runs must equal 11")

        if float(actor.get("status2_median", np.nan)) != 29.0:
            raise ValueError(f"{path}: median valid search budget must equal 29")
        if int(actor.get("status2_max", -1)) != 29:
            raise ValueError(f"{path}: maximum valid search budget must equal 29")
        minimum = int(actor.get("status2_min", -1))
        boundary_windows = int(actor.get("cross_chunk_windows", -1))
        invalid_current_actions = int(actor.get("invalid_current_actions", -1))
        if invalid_current_actions != boundary_windows:
            raise ValueError(
                f"{path}: invalid current-action windows must equal cross-chunk windows"
            )
        if game == 1 and (minimum != 29 or boundary_windows != 0):
            raise ValueError(f"{path}: Pong windows must all contain 29 search events")
        if game == 2 and (minimum != 0 or boundary_windows <= 0):
            raise ValueError(
                f"{path}: Space Invaders must expose only boundary-empty exceptions"
            )
        if int(control.get("empty_status2_windows", -1)) != 0:
            raise ValueError(f"{path}: model-control windows cannot have empty status-2 traces")
        n_rollout_windows = int(rollout.get("n_windows", -1))
        n_prefix_pairs = int(rollout.get("n_prefix_pairs", -1))
        if n_prefix_pairs != 28 * n_rollout_windows:
            raise ValueError(f"{path}: complete rollout windows must contribute 28 prefix pairs")
        if not np.isclose(
            float(actor.get("root_policy_constancy_tolerance", np.nan)),
            1e-6,
            atol=0.0,
            rtol=0.0,
        ):
            raise ValueError(f"{path}: root-policy constancy tolerance changed")

        if not np.isclose(
            float(actor.get("root_policy_all_status_constant_fraction", np.nan)),
            1.0,
            atol=1e-12,
        ):
            raise ValueError(f"{path}: root policy is not constant over the search window")
        if not np.isclose(
            float(actor.get("root_policy_all_status_matches_real_root_fraction", np.nan)),
            1.0,
            atol=1e-12,
        ):
            raise ValueError(f"{path}: search root policy does not match the real root")
        if not np.isclose(
            float(actor.get("root_policy_all_status_max_observed_range", np.nan)),
            0.0,
            atol=1e-12,
        ):
            raise ValueError(f"{path}: root-policy range must equal zero")
        if control.get("policy_roles") != expected_roles:
            raise ValueError(f"{path}: policy-role contract changed")
        if persistence.get("policy_source") != "actor":
            raise ValueError(f"{path}: persistence policy source must be actor")
        if not np.isclose(
            float(persistence.get("policy_temperature", np.nan)), 1.0, atol=1e-12
        ):
            raise ValueError(f"{path}: persistence policy temperature must equal 1")

    return {
        "n_subjects": 6,
        "n_games": 2,
        "n_runs_per_subject_game": 11,
        "n_imagined_events_per_valid_window": 29,
    }


# ---------------------------------------------------------------------------
# Main Figure 0: decision-aligned study and Thinker framework


def framework_box(
    ax: mpl.axes.Axes,
    bounds: tuple[float, float, float, float],
    *,
    facecolor: str = "white",
    edgecolor: str = GRAY,
    linewidth: float = 0.8,
    radius: float = 0.018,
    zorder: float = 1,
) -> FancyBboxPatch:
    x, y, width, height = bounds
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=.008,rounding_size={radius}",
        transform=ax.transAxes,
        facecolor=facecolor,
        edgecolor=edgecolor,
        lw=linewidth,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def framework_arrow(
    ax: mpl.axes.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = INK,
    linewidth: float = 0.9,
    linestyle: str | tuple = "-",
    arrowstyle: str = "-|>",
    connectionstyle: str = "arc3,rad=0",
    mutation_scale: float = 7.0,
    zorder: float = 4,
) -> FancyArrowPatch:
    patch = FancyArrowPatch(
        start,
        end,
        transform=ax.transAxes,
        arrowstyle=arrowstyle,
        mutation_scale=mutation_scale,
        lw=linewidth,
        linestyle=linestyle,
        color=color,
        connectionstyle=connectionstyle,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def framework_panel_title(ax: mpl.axes.Axes, letter: str, title: str) -> None:
    ax.text(
        0.0, 1.012, letter, transform=ax.transAxes,
        ha="left", va="top", fontsize=10, fontweight="bold", color=INK,
    )
    ax.annotate(
        title, xy=(0.0, 1.012), xycoords=ax.transAxes,
        xytext=(14, 0), textcoords="offset points",
        ha="left", va="top", fontsize=8.2, fontweight="bold", color=INK,
        linespacing=1.05,
    )


def draw_game_pictograms(ax: mpl.axes.Axes) -> None:
    assets = (
        (0.05, "Pong", ROOT.parent / "atari-pong.jpg"),
        (0.55, "Space\nInvaders", ROOT.parent / "atari-spaceinvaders.png"),
    )
    for x, title, asset in assets:
        if not asset.exists():
            raise FileNotFoundError(f"Required Atari frame is missing: {asset}")
        bounds = (x, 0.805, 0.40, 0.135)
        image_ax = ax.inset_axes(bounds, transform=ax.transAxes, zorder=3)
        image_ax.imshow(plt.imread(asset), interpolation="nearest", aspect="equal")
        image_ax.set_axis_off()
        ax.add_patch(Rectangle(
            (bounds[0], bounds[1]), bounds[2], bounds[3],
            transform=ax.transAxes, fill=False, edgecolor=NAVY,
            linewidth=.8, zorder=4,
        ))
        ax.text(x + .20, .790, title, transform=ax.transAxes,
                ha="center", va="top", fontsize=7.0, color=INK,
                linespacing=.95)


def draw_continuous_framework(ax: mpl.axes.Axes) -> None:
    ax.set_axis_off()
    framework_panel_title(ax, "A", "Continuous\nhuman gameplay")
    draw_game_pictograms(ax)

    framework_box(ax, (0.04, 0.545, 0.92, 0.190),
                  facecolor="#f5f7f8", edgecolor="#aeb5ba")
    ax.text(
        .09, .640,
        "- $n=6$ participants\n"
        "- Game:\n"
        "  Pong\n"
        "  Space Invaders\n"
        "- 4 recording days\n"
        "- 11 runs / game\n"
        "- Continuous fMRI",
        transform=ax.transAxes, ha="left", va="center",
        fontsize=7.0, linespacing=1.10, color=INK,
    )

    ax.text(.50, .505, "Available transitions", transform=ax.transAxes,
            ha="center", va="center", fontsize=7.4, fontweight="bold")
    grid_x, grid_y, cell_w, cell_h = .04, .295, .46, .085
    transitions = (
        (0, 1, "$W \\rightarrow W$", "persist", BLUE, "#edf4f8"),
        (1, 1, "$W \\leftrightarrow A_i$", "start / stop", RED, "#fbefee"),
        (0, 0, "$A_i \\rightarrow A_i$", "persist", ORANGE, "#fff4e9"),
        (1, 0, "$A_i \\rightarrow A_j$", "switch", PURPLE, "#f4eff8"),
    )
    for col, row, route, label, color, face in transitions:
        x = grid_x + col * cell_w
        y = grid_y + row * cell_h
        ax.add_patch(Rectangle(
            (x, y), cell_w, cell_h, transform=ax.transAxes,
            facecolor=face, edgecolor="none", zorder=1,
        ))
        ax.text(x + cell_w / 2, y + .056, route, transform=ax.transAxes,
                ha="center", va="center", fontsize=7.3, color=color)
        ax.text(x + cell_w / 2, y + .022, label, transform=ax.transAxes,
                ha="center", va="center", fontsize=7.0,
                color=INK)
    ax.add_patch(Rectangle(
        (grid_x, grid_y), 2 * cell_w, 2 * cell_h,
        transform=ax.transAxes, fill=False, edgecolor="#8c9297",
        linewidth=.85, zorder=2,
    ))
    ax.plot([grid_x + cell_w, grid_x + cell_w],
            [grid_y, grid_y + 2 * cell_h], transform=ax.transAxes,
            color="#8c9297", lw=.85, zorder=2)
    ax.plot([grid_x, grid_x + 2 * cell_w],
            [grid_y + cell_h, grid_y + cell_h], transform=ax.transAxes,
            color="#8c9297", lw=.85, zorder=2)

def framework_node(
    ax: mpl.axes.Axes,
    x: float,
    y: float,
    *,
    size: float,
    facecolor: str,
    edgecolor: str,
    linewidth: float = .8,
    alpha: float = 1.0,
    zorder: float = 5,
) -> None:
    """Draw a point-sized circular node, independent of axes aspect ratio."""
    ax.scatter(
        [x], [y], s=size, transform=ax.transAxes,
        facecolors=facecolor, edgecolors=edgecolor,
        linewidths=linewidth, alpha=alpha, zorder=zorder,
        clip_on=False,
    )


def draw_search_burst(
    ax: mpl.axes.Axes, x: float, y: float, *, scale: float, alpha: float = 1.0
) -> None:
    root = (x, y)
    children = (
        (x - .050 * scale, y - .052 * scale),
        (x, y - .068 * scale),
        (x + .050 * scale, y - .052 * scale),
    )
    for child in children:
        framework_arrow(
            ax, root, child, color=PURPLE, linewidth=.8,
            linestyle=(0, (2.0, 1.8)), arrowstyle="-",
            mutation_scale=4, zorder=2,
        ).set_alpha(alpha)
    for px, py in (root, *children):
        framework_node(
            ax, px, py, size=30 * scale ** 2,
            facecolor="white", edgecolor=PURPLE,
            linewidth=.8, alpha=alpha, zorder=3,
        )


def draw_detailed_search_tree(ax: mpl.axes.Axes) -> None:
    nodes = {
        "root": (.465, .480),
        "a1": (.590, .560), "a2": (.590, .480), "a3": (.590, .395),
        "b1": (.705, .535), "b2": (.705, .445),
        "c1": (.830, .500), "c2": (.830, .395),
    }
    edges = (
        ("root", "a1"), ("root", "a2"), ("root", "a3"),
        ("a1", "b1"), ("a2", "b1"), ("a2", "b2"),
        ("b1", "c1"), ("b2", "c1"), ("b2", "c2"),
    )
    selected = {("root", "a2"), ("a2", "b2"), ("b2", "c2")}
    for start_key, end_key in edges:
        start, end = nodes[start_key], nodes[end_key]
        is_selected = (start_key, end_key) in selected
        framework_arrow(
            ax, start, end, color=PURPLE,
            linewidth=1.65 if is_selected else .8,
            linestyle=(0, (2.2, 1.7)), arrowstyle="-|>",
            mutation_scale=6.5 if is_selected else 5.0, zorder=3,
        )
    selected_nodes = {item for edge in selected for item in edge}
    for key, (x, y) in nodes.items():
        framework_node(
            ax, x, y, size=70 if key == "root" else 46,
            facecolor="#e8ddf2" if key == "c2" else "white",
            edgecolor=ORANGE if key in selected_nodes else PURPLE,
            linewidth=1.2 if key in selected_nodes else .8, zorder=5,
        )
    for label, x, y in (("$a_1$", .530, .535), ("$a_2$", .530, .480),
                        ("$a_3$", .530, .420)):
        ax.text(x, y, label, transform=ax.transAxes, ha="center", va="center",
                fontsize=7.0, color=PURPLE)
    ax.text(.465, .505, "root", transform=ax.transAxes, ha="center", va="bottom",
            fontsize=7.0, color=PURPLE)
    ax.text(.590, .585, "depth 1", transform=ax.transAxes, ha="center", va="bottom",
            fontsize=7.0, color=GRAY)
    ax.text(.705, .560, "depth 2", transform=ax.transAxes, ha="center", va="bottom",
            fontsize=7.0, color=GRAY)
    ax.text(.830, .525, "depth 3", transform=ax.transAxes, ha="center", va="bottom",
            fontsize=7.0, color=GRAY)
    ax.text(.860, .395, "new", transform=ax.transAxes, ha="left", va="center",
            fontsize=7.0, color=ORANGE)


def draw_coordinate_rail(ax: mpl.axes.Axes) -> None:
    boxes = (
        (
            .010, .018, .210, .172, "Pre-search",
            "policy: $\\pi^0_t(a)$\n"
            "value: $Q^0_t(a)$\n"
            "support: $M_t$",
            BLUE, "#edf4f8", .082,
        ),
        (
            .260, .018, .210, .172, "Imaginary\nevent $j$",
            "evidence: $\\mathcal{A}^{roll}_{t,j}$\n"
            "update: $\\Delta Q_{t,j}$\n"
            "branch · depth\n"
            "visits · reset",
            PURPLE, "#f4eff8", .066,
        ),
        (
            .510, .018, .230, .172, "Decision\nsummary",
            "revision: $U_t$\n"
            "allocation: $\\mathcal{G}_t$",
            GREEN, "#eef6f2", .088,
        ),
        (
            .780, .018, .210, .172, "Post-search",
            "summary: $S_t$\n"
            "readout: $\\pi_t^{post}$",
            ORANGE, "#fff4e9", .090,
        ),
    )
    for x, y, width, height, title, body, color, face, body_offset in boxes:
        framework_box(ax, (x, y, width, height), facecolor=face,
                      edgecolor=color, linewidth=.9, radius=.014)
        ax.text(x + width / 2, y + height - .028, title,
                transform=ax.transAxes, ha="center", va="top",
                fontsize=7.0, fontweight="bold", color=color,
                linespacing=.92)
        is_two_line_title = "\n" in title
        body_y = y + body_offset
        ax.text(x + width / 2, body_y, body,
                transform=ax.transAxes, ha="center", va="center",
                fontsize=7.0, linespacing=.92 if is_two_line_title else 1.05,
                color=INK)


def draw_thinker_framework(ax: mpl.axes.Axes) -> None:
    ax.set_axis_off()
    framework_panel_title(ax, "B", "Learned tree search at every real step")
    ax.text(.000, .947, "human trajectory", transform=ax.transAxes,
            ha="left", va="center", fontsize=7.0, color=NAVY, fontweight="bold")
    state_y = .885
    state_x = (.12, .50, .88)
    state_labels = ("$s_t$", "$s_{t+1}$", "$s_{t+2}$")
    for index, (x, label) in enumerate(zip(state_x, state_labels)):
        framework_node(
            ax, x, state_y, size=320,
            facecolor="#eaf2f7", edgecolor=BLUE,
            linewidth=1.1, zorder=5,
        )
        ax.text(x, state_y, label, transform=ax.transAxes,
                ha="center", va="center", fontsize=7.2, color=NAVY, zorder=6)
        framework_arrow(ax, (x, .858), (x, .842), color=PURPLE,
                        linewidth=.8, linestyle=(0, (2, 1.8)), mutation_scale=5)
        draw_search_burst(ax, x, .827, scale=.82 if index != 1 else .96,
                          alpha=.72 if index != 1 else 1.0)
        if index < 2:
            framework_arrow(ax, (x + .034, state_y), (state_x[index + 1] - .034, state_y),
                            color=NAVY, linewidth=1.15, mutation_scale=7.5)
            action_label = "$a_t^{human}$" if index == 0 else "$a_{t+1}^{human}$"
            ax.text((x + state_x[index + 1]) / 2, .903, action_label,
                    transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=7.0, color=NAVY)
    ax.text(.970, state_y, "…", transform=ax.transAxes,
            ha="center", va="center", fontsize=10, color=NAVY)
    framework_box(ax, (.01, .235, .98, .485),
                  facecolor="#faf9fb", edgecolor="#b3a5c1", linewidth=.9)
    ax.text(.025, .700, "Zoom: Imaginary search",
            transform=ax.transAxes, ha="left", va="center",
            fontsize=7.4, fontweight="bold", color=PURPLE)

    stage_x, stage_w, stage_h = .035, .370, .075
    framework_box(ax, (stage_x, .600, stage_w, stage_h),
                  facecolor="#edf4f8", edgecolor=BLUE)
    ax.text(.220, .6375, "current real state\n+ history",
            transform=ax.transAxes, ha="center", va="center", fontsize=7.0,
            linespacing=.90, color=NAVY)
    framework_arrow(ax, (.220, .594), (.220, .571), color=INK, linewidth=.8)
    framework_box(ax, (stage_x, .490, stage_w, stage_h),
                  facecolor="#edf4f8", edgecolor=BLUE)
    ax.text(.220, .5275, "pre-search root\n$\\pi^0$ fixed · $Q^0, M$",
            transform=ax.transAxes, ha="center", va="center", fontsize=7.0,
            linespacing=.90, color=NAVY)
    framework_arrow(ax, (.220, .484), (.220, .461), color=INK, linewidth=.8)
    framework_box(ax, (stage_x, .380, stage_w, stage_h),
                  facecolor="#fff3e8", edgecolor=ORANGE, linewidth=1.0)
    ax.text(.220, .4175, "learned Actor\nimagined action / reset\n← tree feedback",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=7.0, linespacing=.88, color=ORANGE, fontweight="bold")

    framework_arrow(ax, (.220, .374), (.220, .351), color=ORANGE, linewidth=.8)
    framework_box(ax, (stage_x, .270, stage_w, stage_h),
                  facecolor="#f4eff8", edgecolor=PURPLE)
    ax.text(.220, .326, "learned dual world model", transform=ax.transAxes,
            ha="center", va="center", fontsize=7.0,
            color=PURPLE, fontweight="bold")
    ax.text(.220, .292, "state–reward: $s'$ · $r$ · done\n"
            "value–policy: $\\pi$ · $V$",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=7.0, linespacing=.84, color=PURPLE)

    draw_detailed_search_tree(ax)
    framework_arrow(
        ax, (.405, .430), (.448, .477), color=ORANGE, linewidth=.9,
        linestyle=(0, (2.3, 1.8)), connectionstyle="arc3,rad=-.25",
        mutation_scale=6.5,
    )
    ax.text(.415, .463, "reset", transform=ax.transAxes,
            ha="center", va="center", fontsize=7.0, color=ORANGE)
    framework_arrow(
        ax, (.448, .465), (.407, .405), color=ORANGE, linewidth=.9,
        connectionstyle="arc3,rad=.22", mutation_scale=6.5,
    )
    framework_arrow(
        ax, (.405, .292), (.815, .395), color=PURPLE, linewidth=1.0,
        linestyle=(0, (2.3, 1.8)), connectionstyle="arc3,rad=-.12",
        mutation_scale=7.0,
    )
    ax.text(.780, .665, "expand · continue\nreset-to-root",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=7.0, linespacing=.90, color=PURPLE)
    framework_box(ax, (.610, .252, .350, .085),
                  facecolor="#fff4e9", edgecolor=ORANGE, linewidth=.8)
    ax.text(.785, .2945, "fixed search budget\n"
            "update root $Q$\n"
            "+ visit count $N$\n"
            "propagate rollout return",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=7.0, linespacing=.80, color=ORANGE)
    framework_arrow(ax, (.830, .382), (.805, .339), color=ORANGE, linewidth=.9)

    for x, color in ((.115, BLUE), (.365, PURPLE), (.625, GREEN), (.885, ORANGE)):
        framework_arrow(ax, (x, .228), (x, .198), color=color,
                        linewidth=.8, mutation_scale=5.5)
    draw_coordinate_rail(ax)


def draw_validation_framework(ax: mpl.axes.Axes) -> None:
    ax.set_axis_off()
    framework_panel_title(ax, "C", "Behavioral +\nneural tests")
    framework_box(ax, (.08, .850, .84, .075),
                  facecolor="#f4f1f7", edgecolor=PURPLE, linewidth=.9)
    ax.text(.50, .888, "decision-aligned\ncoordinates $x_t$",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=7.0, linespacing=1.0, color=PURPLE, fontweight="bold")
    ax.text(.50, .805, "tested against", transform=ax.transAxes,
            ha="center", va="center", fontsize=7.0, color=GRAY,
            bbox=dict(facecolor="white", edgecolor="none", pad=.8))
    ax.plot([.50, .50], [.848, .790], transform=ax.transAxes,
            color=GRAY, lw=.9, ls=(0, (2, 2)), zorder=2)
    ax.plot([.04, .50], [.790, .790], transform=ax.transAxes,
            color=GRAY, lw=.9, ls=(0, (2, 2)), zorder=2)
    ax.plot([.04, .04], [.790, .405], transform=ax.transAxes,
            color=GRAY, lw=.9, ls=(0, (2, 2)), zorder=2)
    ax.plot([.04, .08], [.645, .645], transform=ax.transAxes,
            color=GRAY, lw=.9, ls=(0, (2, 2)), zorder=2)
    ax.plot([.04, .08], [.405, .405], transform=ax.transAxes,
            color=GRAY, lw=.9, ls=(0, (2, 2)), zorder=2)

    framework_box(ax, (.08, .525, .84, .230),
                  facecolor="#edf4f8", edgecolor=BLUE, linewidth=.9)
    ax.text(.50, .720, "Behavioral tests", transform=ax.transAxes,
            ha="center", va="center", fontsize=7.3, color=BLUE, fontweight="bold")
    ax.text(
        .15, .630,
        "- How long does an\n"
        "  action continue?\n"
        "- When does it\n"
        "  change?\n"
        "- What action comes\n"
        "  next?\n"
        "- Can it predict an\n"
        "  unseen run?",
        transform=ax.transAxes, ha="left", va="center",
        fontsize=7.0, linespacing=1.02, color=INK,
    )

    framework_box(ax, (.08, .315, .84, .180),
                  facecolor="#eef6f2", edgecolor=GREEN, linewidth=.9)
    ax.text(.50, .462, "Neural tests", transform=ax.transAxes,
            ha="center", va="center", fontsize=7.3, color=GREEN, fontweight="bold")
    ax.text(
        .15, .395,
        "- MVPA\n"
        "- Controlled RSA\n"
        "- Nested encoding\n"
        "- Controlled GLM",
        transform=ax.transAxes, ha="left", va="center",
        fontsize=7.0, linespacing=1.10, color=INK,
    )

    framework_box(ax, (.08, .075, .84, .170),
                  facecolor="#f4f5f6", edgecolor="#aeb5ba", linewidth=.8)
    ax.text(
        .15, .160,
        "Controls\n"
        "- Visual / task\n"
        "- Decision event\n"
        "- Current / previous\n"
        "  action\n"
        "- History / time\n"
        "- $\\pi^0, M, Q^0$",
        transform=ax.transAxes, ha="left", va="center",
        fontsize=7.0, linespacing=1.02, color=GRAY,
    )


def main_figure_0(path: Path) -> None:
    fig = plt.figure(figsize=(7.2, 6.2))
    fig.subplots_adjust(left=.035, right=.988, bottom=.040, top=.965, wspace=.11)
    gs = fig.add_gridspec(1, 3, width_ratios=[.20, .59, .21])
    a = fig.add_subplot(gs[0, 0]); draw_continuous_framework(a)
    b = fig.add_subplot(gs[0, 1]); draw_thinker_framework(b)
    c = fig.add_subplot(gs[0, 2]); draw_validation_framework(c)
    enforce_minimum_visual_scale(fig)
    save(fig, path)


# ---------------------------------------------------------------------------
# Main Figure 1: behavior



def draw_action_schematic(ax: mpl.axes.Axes) -> None:
    ax.set_axis_off()
    xs = np.linspace(0.07, 0.93, 8)
    labels = [r"$a$", r"$a$", r"$a$", r"$a$", r"$a$", r"$b$", r"$b$", r"$b$"]
    colors = [BLUE] * 5 + [ORANGE] * 3
    for x, lab, col in zip(xs, labels, colors):
        ax.add_patch(Circle((x, 0.53), 0.050, transform=ax.transAxes,
                            facecolor=col, edgecolor="white", lw=0.7))
        ax.text(x, 0.53, lab, transform=ax.transAxes, color="white",
                ha="center", va="center", fontsize=8)
    ax.annotate("", xy=(0.91, 0.27), xytext=(0.09, 0.27), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", lw=0.8, color=INK))
    ax.axvline(0.69, 0.21, 0.74, color=RED, lw=1.0, ls="--")
    ax.text(0.34, 0.80, "persistence", transform=ax.transAxes,
            ha="center", color=NAVY, fontsize=6.2)
    ax.text(0.75, 0.80, "transition", transform=ax.transAxes,
            ha="center", color=RED, fontsize=6.2)
    ax.text(0.50, 0.11, "continuous gameplay", transform=ax.transAxes,
            ha="center", fontsize=6)
    ax.set_title("Action bouts", pad=2)


def draw_action_distribution(ax: mpl.axes.Axes) -> None:
    d = read(OUT / "00_learningcurve" / "results" / "subject_action_distribution_counts.csv")
    require_columns(d, ["game_id", "subject", "source", "action_id", "action_label", "proportion"], "action distribution")
    expected = {(g, s) for g in (1, 2) for s in range(1, 7)}
    got = set(zip(d.game_id.astype(int), d.subject.astype(int)))
    if got != expected:
        raise ValueError(f"action distribution pair contract failed: {sorted(got ^ expected)}")
    order = sorted(d.action_id.unique())
    x = np.arange(len(order))
    labels = [str(d.loc[d.action_id.eq(a), "action_label"].iloc[0]) for a in order]
    offsets = {1: -0.18, 2: 0.18}
    source_style = {"human_observed": ("o", "Human action"), "thinker_replay_recommendation": ("s", "Thinker action during replay")}
    for game in (1, 2):
        for source, (marker, label) in source_style.items():
            q = d[(d.game_id == game) & (d.source == source)]
            p = q.pivot(index="subject", columns="action_id", values="proportion").reindex(columns=order)
            mean = p.mean(0).to_numpy()
            sem = p.sem(0).to_numpy()
            ax.errorbar(x + offsets[game], mean, yerr=sem, fmt=marker, ms=3.0, lw=0.65,
                        capsize=1.4, color=GAME_COLORS[game], mfc=("white" if source.startswith("thinker") else GAME_COLORS[game]),
                        label=f"{GAME_NAMES[game]} · {label}", zorder=3)
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel("Action proportion")
    ax.set_ylim(bottom=0)
    ax.set_title("Action distributions", pad=2)
    finish_ax(ax)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.25), columnspacing=0.8, handletextpad=0.3)


def draw_repeatability(parent_spec, fig: mpl.figure.Figure) -> list[mpl.axes.Axes]:
    sub = parent_spec.subgridspec(1, 2, wspace=0.22)
    d = read(BEH / "1-2_noop_day_level.csv")
    icc = read(BEH / "1-2_noop_day_repeatability.csv")
    inf = read(BEH / "1-2_noop_subject_difference_inference.csv")
    axes = []
    day_colors = ["#c6dbef", "#9ecae1", "#6baed6", "#2171b5"]
    rng = np.random.default_rng(20260818)
    for col, game in enumerate((1, 2)):
        ax = fig.add_subplot(sub[0, col]); axes.append(ax)
        q = d[d.game == game].copy()
        subjects = sorted(q.subject.unique())
        vals = [q.loc[q.subject.eq(s), "noop_rate"].to_numpy() for s in subjects]
        bp = ax.boxplot(vals, positions=np.arange(1, 7), widths=0.55, patch_artist=True, showfliers=False,
                        medianprops=dict(color=INK, lw=0.9), whiskerprops=dict(color=GRAY, lw=0.6),
                        capprops=dict(color=GRAY, lw=0.6), boxprops=dict(edgecolor=GRAY, lw=0.6))
        for b in bp["boxes"]: b.set_facecolor("#edf3f7")
        for day, c in zip(sorted(q.day.unique()), day_colors):
            z = q[q.day == day]
            for _, r in z.iterrows():
                sx = int(str(r.subject).split("_")[-1])
                ax.scatter(sx + rng.normal(0, 0.035), r.noop_rate, s=9, color=c, edgecolor="white", lw=0.25, zorder=4,
                           label=f"Day {int(day)}" if sx == 1 else None)
        ii = icc[icc.game == game].iloc[0]
        ss = inf[inf.game == game].iloc[0]
        ax.set_title(f"{GAME_NAMES[game]}\nICC={ii.icc_a1:.2f}; W={ss.kendall_w:.2f} {ss.stars}", pad=2)
        ax.set_xticks(range(1, 7), [str(i) for i in range(1, 7)])
        ax.set_xlabel("Participant")
        if col == 0: ax.set_ylabel("NOOP proportion")
        finish_ax(ax)
    axes[1].legend(frameon=False, ncol=2, loc="lower right", handletextpad=0.2,
                   columnspacing=0.5, title="Day", title_fontsize=5.5)
    return axes


def draw_mixture(ax: mpl.axes.Axes) -> None:
    d = read(BEH / "1-7_noop_duration_mixture_components.csv")
    for game, marker in [(1, "o"), (2, "s")]:
        q = d[d.game == game]
        for subj in sorted(q.subject.unique()):
            s = q[q.subject == subj].sort_values("component")
            if len(s) != 2: continue
            ax.plot(s.median_steps, [int(str(subj).split("_")[-1])] * 2, color=GAME_COLORS[game], alpha=0.38, lw=0.8)
            ax.scatter(s.median_steps, [int(str(subj).split("_")[-1])] * 2,
                       s=14 + 24 * s.weight.to_numpy(), marker=marker, color=GAME_COLORS[game],
                       edgecolor="white", lw=0.4, zorder=3, label=GAME_NAMES[game] if subj == sorted(q.subject.unique())[0] else None)
    ax.set_xscale("log")
    ax.set_yticks(range(1, 7))
    ax.set_ylabel("Participant")
    ax.set_xlabel("Component median (steps; log scale)")
    ax.set_title("NOOP-bout timescales", pad=2)
    finish_ax(ax, "x")
    ax.legend(frameon=False, loc="lower right")


def main_figure_1(path: Path) -> None:
    fig = plt.figure(figsize=(WIDTH, 4.25), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.055, h_pad=0.055)
    gs = fig.add_gridspec(2, 12, height_ratios=[0.82, 1.18])
    a = fig.add_subplot(gs[0, :4]); draw_action_schematic(a); panel(a, "A", -0.10, 1.05)
    b = fig.add_subplot(gs[0, 4:]); draw_action_distribution(b); panel(b, "B", -0.06, 1.05)
    caxes = draw_repeatability(gs[1, :], fig); panel(caxes[0], "C", -0.10, 1.16)
    save(fig, path)


# ---------------------------------------------------------------------------
# Main Figure 2: process-resolved model



def draw_process_proxy(ax: mpl.axes.Axes) -> None:
    """Compact coordinate key; the full mechanism is defined in Figure 0B."""
    ax.set_axis_off()
    ax.text(
        .00, .91, "Control-state coordinates used below",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=7.5, fontweight="bold", color=INK,
    )
    items = (
        (.015, .215, "Root support", "$M_t$", BLUE, "#edf4f8"),
        (.245, .215, "Value revision", r"$U_t\,/\,Q_{\mathrm{rev},t}$", PURPLE, "#f4eff8"),
        (.500, .215, "Allocation", "$\\mathcal{G}_t$", GREEN, "#eef6f2"),
        (.720, .215, "Readout / trace", r"$\pi_t^{\mathrm{post}}\,/\,Z_t$", ORANGE, "#fff4e9"),
    )
    widths = (.200, .225, .190, .265)
    for (x, y, label, symbol, color, face), width in zip(items, widths):
        framework_box(
            ax, (x, y, width, .425), facecolor=face,
            edgecolor=color, linewidth=.9, radius=.018,
        )
        ax.text(
            x + width / 2, y + .295, label,
            transform=ax.transAxes, ha="center", va="center",
            fontsize=7.0, fontweight="bold", color=color,
        )
        ax.text(
            x + width / 2, y + .120, symbol,
            transform=ax.transAxes, ha="center", va="center",
            fontsize=7.5, color=INK,
        )
    ax.text(
        .985, .055, "Detailed model: Fig. 0",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=7.0, color=GRAY,
    )



def draw_bout_trajectories(parent_spec, fig: mpl.figure.Figure) -> list[mpl.axes.Axes]:
    sub = parent_spec.subgridspec(2, 2, wspace=.30, hspace=.58)
    axes = []
    games = {}
    limits = {}
    for game in (1, 2):
        d = read(OUT / "02_evaluation_revision_geometry" / f"group_game{game}" /
                 "results" / "same_action_bout_endpoints.csv.gz")
        games[game] = d
        pooled = []
        for tr in ("noop_to_active", "active_to_noop"):
            q = d[d.transition_class == tr]
            pooled.append(q.groupby("subject")[["start_margin", "start_revision_z",
                                                "end_margin", "end_revision_z"]].mean())
        p = pd.concat(pooled)
        xx = np.r_[p.start_margin.to_numpy(), p.end_margin.to_numpy()]
        yy = np.r_[p.start_revision_z.to_numpy(), p.end_revision_z.to_numpy()]
        xpad = max(.08, .12 * float(np.ptp(xx)))
        ypad = max(.08, .12 * float(np.ptp(yy)))
        limits[game] = ((float(np.nanmin(xx))-xpad, float(np.nanmax(xx))+xpad),
                        (float(np.nanmin(yy))-ypad, float(np.nanmax(yy))+ypad))

    for row, game in enumerate((1, 2)):
        d = games[game]
        for col, tr in enumerate(("noop_to_active", "active_to_noop")):
            ax = fig.add_subplot(sub[row, col]); axes.append(ax)
            q = d[d.transition_class == tr]
            means = q.groupby("subject")[["start_margin", "start_revision_z",
                                           "end_margin", "end_revision_z"]].mean()
            # Invisible endpoints make Matplotlib register every arrow endpoint;
            # explicit game-common limits below provide the final contract.
            ax.scatter(np.r_[means.start_margin, means.end_margin],
                       np.r_[means.start_revision_z, means.end_revision_z],
                       s=.01, alpha=0)
            for _, r in means.iterrows():
                ax.annotate("", xy=(r.end_margin, r.end_revision_z),
                            xytext=(r.start_margin, r.start_revision_z),
                            arrowprops=dict(arrowstyle="->", color=GAME_COLORS[game],
                                            alpha=.55, lw=.75))
            sm = means.mean()
            ax.annotate("", xy=(sm.end_margin, sm.end_revision_z),
                        xytext=(sm.start_margin, sm.start_revision_z),
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.5,
                                        mutation_scale=9))
            ax.axhline(0, color="#bbbbbb", lw=.45)
            ax.axvline(0, color="#bbbbbb", lw=.45)
            ax.set_xlim(*limits[game][0]); ax.set_ylim(*limits[game][1])
            ax.set_title(("Pong" if game == 1 else "Space Inv.") + " · " + TRANS_NAMES[tr],
                         pad=1, fontsize=5.8)
            if row == 1: ax.set_xlabel("incumbent margin $M$", labelpad=1)
            if col == 0: ax.set_ylabel(r"revision $Q_{\mathrm{rev}}$", labelpad=1)
            ax.tick_params(axis="both", pad=1)
            finish_ax(ax, "")
    return axes


def draw_transition_effect_summary(ax: mpl.axes.Axes) -> None:
    """Show the formal full-risk-set effects that underwrite panel C."""
    rows = []
    for game in (1, 2):
        group = read(
            OUT / "02_evaluation_revision_geometry" / f"group_game{game}" /
            "results" / "transition_specific_group_effects.csv"
        )
        for transition in ("noop_to_active", "active_to_noop"):
            one = group[group.transition_type.eq(transition)].set_index("metric")
            required = {"beta_M", "beta_V", "beta_MxV"}
            if set(one.index) != required:
                raise ValueError(
                    f"Unexpected transition-effect metrics for game={game}, "
                    f"transition={transition}: {sorted(one.index)}"
                )
            rows.append((game, transition, one))

    metrics = ("beta_M", "beta_V", "beta_MxV")
    values = np.asarray([
        [float(one.loc[metric, "mean_subject_effect"]) for metric in metrics]
        for _, _, one in rows
    ])
    q_values = np.asarray([
        [float(one.loc[metric, "bh_fdr_q"]) for metric in metrics]
        for _, _, one in rows
    ])
    limit = max(.60, float(np.nanmax(np.abs(values))))
    image = ax.imshow(
        values, cmap="RdBu_r", aspect="auto",
        norm=mpl.colors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
    )
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            marker = corrected_significance_label(q_values[row, col])
            label = f"{values[row, col]:+.2f}"
            if marker != "n.s.":
                label += f" {marker}"
            else:
                label += "  n.s."
            ax.text(
                col, row, label, ha="center", va="center", fontsize=4.55,
                color="white" if abs(values[row, col]) > .31 * limit else INK,
                fontweight="bold" if marker != "n.s." else "normal",
            )
    ax.set_xticks(
        range(3),
        [r"margin $M$", r"revision $Q_{\mathrm{rev}}$", r"$M\!\times\!Q_{\mathrm{rev}}$"],
    )
    ax.set_yticks(
        range(4),
        [f"{('Pong' if game == 1 else 'Space Inv.')} · {TRANS_NAMES[transition]}"
         for game, transition, _ in rows],
    )
    ax.tick_params(axis="both", length=0, labelsize=4.8, pad=2)
    ax.set_title("Adjusted transition effects", loc="left", pad=7)
    ax.text(
        1.0, 1.02, "standardized log-odds $\u03b2$ · BH-FDR/game",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=4.4,
        color=GRAY,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)


def load_transition_windows(game: int) -> pd.DataFrame:
    frames=[]
    cols=["subject","transition_type","transition_event","transition_margin","value_revision_robust_z"]
    base=OUT/"02_evaluation_revision_geometry"
    for s in range(1,7):
        frames.append(read(base/f"sub{s:03d}_game{game}"/"results"/"transition_specific_risk_windows.csv.gz",cols))
    return pd.concat(frames,ignore_index=True)


def frozen_transition_surface(
    subject_frames: list[pd.DataFrame],
    transition: str,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    bins: int = 34,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the frozen 02 group-display support rule without refitting."""
    x_edges = np.linspace(*x_limits, bins + 1)
    y_edges = np.linspace(*y_limits, bins + 1)
    surfaces = []
    for frame in subject_frames:
        q = frame.loc[frame.transition_type.eq(transition),
                      ["transition_margin", "value_revision_robust_z",
                       "transition_event"]].replace([np.inf, -np.inf], np.nan).dropna()
        x = q.transition_margin.to_numpy(float)
        y = q.value_revision_robust_z.to_numpy(float)
        event = q.transition_event.to_numpy(float)
        count = np.histogram2d(x, y, bins=[x_edges, y_edges])[0].T
        event_sum = np.histogram2d(
            x, y, bins=[x_edges, y_edges], weights=event
        )[0].T
        one = np.full_like(event_sum, np.nan, dtype=float)
        np.divide(event_sum, count, out=one, where=count >= 12)
        surfaces.append(one)
    stack = np.stack(surfaces)
    support = np.sum(np.isfinite(stack), axis=0)
    surface = np.full(stack.shape[1:], np.nan, dtype=float)
    np.divide(np.nansum(stack, axis=0), support, out=surface,
              where=support >= 4)
    return surface, x_edges, y_edges


def participant_equalized_marginals(
    subject_frames: list[pd.DataFrame],
    transition: str,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Average within-participant marginal probability mass over risk windows."""
    x_histograms = []
    y_histograms = []
    for frame in subject_frames:
        one = frame.loc[
            frame.transition_type.eq(transition),
            ["transition_margin", "value_revision_robust_z"],
        ].replace([np.inf, -np.inf], np.nan).dropna()
        x_count = np.histogram(one.transition_margin.to_numpy(float), bins=x_edges)[0]
        y_count = np.histogram(one.value_revision_robust_z.to_numpy(float), bins=y_edges)[0]
        if x_count.sum() and y_count.sum():
            x_histograms.append(x_count / x_count.sum())
            y_histograms.append(y_count / y_count.sum())
    if not x_histograms or not y_histograms:
        raise ValueError(f"No marginal histogram support for {transition}")
    return np.mean(x_histograms, axis=0), np.mean(y_histograms, axis=0)


def draw_transition_surfaces(parent_spec, fig: mpl.figure.Figure) -> list[mpl.axes.Axes]:
    sub = parent_spec.subgridspec(2, 2, wspace=.30, hspace=.34)
    axes = []
    cmap = mpl.colormaps["magma"].copy()
    cmap.set_bad("white")
    for row, game in enumerate((1, 2)):
        frames = []
        for subject in range(1, 7):
            cols = ["subject", "transition_type", "transition_event",
                    "transition_margin", "value_revision_robust_z"]
            frames.append(read(
                OUT / "02_evaluation_revision_geometry" /
                f"sub{subject:03d}_game{game}" / "results" /
                "transition_specific_risk_windows.csv.gz", cols
            ))
        combined = pd.concat(frames, ignore_index=True)
        margin = pd.to_numeric(combined.transition_margin, errors="coerce")
        revision = pd.to_numeric(combined.value_revision_robust_z, errors="coerce")
        x_limits = tuple(np.quantile(margin[np.isfinite(margin)], [.01, .99]))
        y_limits = tuple(np.quantile(revision[np.isfinite(revision)], [.01, .99]))
        for col, tr in enumerate(("noop_to_active", "active_to_noop")):
            cell = sub[row, col].subgridspec(
                2, 2, width_ratios=[4.0, .92], height_ratios=[4.0, .92],
                wspace=.055, hspace=.055,
            )
            ax = fig.add_subplot(cell[0, 0]); axes.append(ax)
            hist_y = fig.add_subplot(cell[0, 1], sharey=ax)
            hist_x = fig.add_subplot(cell[1, 0], sharex=ax)
            cax = fig.add_subplot(cell[1, 1])
            surface, x_edges, y_edges = frozen_transition_surface(
                frames, tr, x_limits, y_limits
            )
            finite = surface[np.isfinite(surface)]
            vmax = float(np.nanquantile(finite, .97)) if finite.size else 1.0
            vmax = max(vmax, .02 if tr == "noop_to_active" else .10)
            x_centers = (x_edges[:-1] + x_edges[1:]) / 2
            y_centers = (y_edges[:-1] + y_edges[1:]) / 2
            im = ax.pcolormesh(
                x_centers, y_centers, np.ma.masked_invalid(surface),
                shading="gouraud", cmap=cmap, vmin=0, vmax=vmax,
                rasterized=True,
            )
            ax.axvline(0, color=INK, ls="--", lw=.55)
            ax.axhline(0, color=INK, ls=":", lw=.5, alpha=.7)
            ax.set_xlim(x_limits); ax.set_ylim(y_limits)
            ax.set_title(
                ("Pong" if game == 1 else "Space Inv.") + " · " + TRANS_NAMES[tr],
                pad=1, fontsize=5.8,
            )
            if col == 0: ax.set_ylabel(r"revision $Q_{\mathrm{rev}}$", labelpad=1)
            else: ax.tick_params(labelleft=False)
            ax.tick_params(axis="both", labelbottom=False, pad=1)

            x_mass, y_mass = participant_equalized_marginals(
                frames, tr, x_edges, y_edges
            )
            hist_x.stairs(
                x_mass, x_edges, fill=True, color=GAME_COLORS[game],
                alpha=.25, linewidth=0,
            )
            hist_x.stairs(
                x_mass, x_edges, color=GAME_COLORS[game], linewidth=.65,
            )
            hist_y.stairs(
                y_mass, y_edges, orientation="horizontal", fill=True,
                color=GAME_COLORS[game], alpha=.25, linewidth=0,
            )
            hist_y.stairs(
                y_mass, y_edges, orientation="horizontal",
                color=GAME_COLORS[game], linewidth=.65,
            )
            hist_x.axvline(0, color=INK, ls="--", lw=.45, alpha=.75)
            hist_y.axhline(0, color=INK, ls=":", lw=.45, alpha=.65)
            hist_x.set_ylim(0, max(float(np.max(x_mass)) * 1.08, 1e-4))
            hist_y.set_xlim(0, max(float(np.max(y_mass)) * 1.08, 1e-4))
            hist_x.set_yticks([])
            hist_y.set_xticks([])
            hist_y.tick_params(labelleft=False)
            hist_x.tick_params(axis="x", labelsize=4.7, pad=1, length=1.6)
            hist_x.set_xlabel("transition margin $M$", fontsize=5.1, labelpad=1)
            for marginal in (hist_x, hist_y):
                marginal.spines[["top", "right"]].set_visible(False)
                marginal.spines[["left", "bottom"]].set_color("#a5a9ac")
                marginal.spines[["left", "bottom"]].set_linewidth(.4)

            cb = fig.colorbar(im, cax=cax, orientation="vertical")
            cb.ax.tick_params(labelsize=3.55, length=1.1, pad=.6)
            cb.set_label("P(trans.)", fontsize=3.9, labelpad=.5)
    return axes


def box_jitter(
    ax: mpl.axes.Axes,
    values: np.ndarray,
    x: float,
    color: str,
    rng: np.random.Generator,
    *,
    width: float = .22,
) -> None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return
    bp = ax.boxplot(
        [values], positions=[x], widths=width, patch_artist=True,
        showfliers=False, manage_ticks=False,
        boxprops=dict(facecolor=color, edgecolor=color, alpha=.22, lw=.7),
        medianprops=dict(color=color, lw=1.0),
        whiskerprops=dict(color=color, lw=.6),
        capprops=dict(color=color, lw=.6),
    )
    for artist in bp["boxes"]:
        artist.set_zorder(2)
    ax.scatter(
        x + rng.normal(0, width * .10, len(values)), values,
        s=8, color=color, alpha=.72, edgecolor="white", lw=.22, zorder=3,
    )


def corrected_significance_label(p_value: float) -> str:
    """Return a compact marker for multiplicity-corrected group inference."""
    if not np.isfinite(p_value) or p_value >= .05:
        return "n.s."
    if p_value < .001:
        return "***"
    if p_value < .01:
        return "**"
    return "*"


def annotate_corrected_significance(
    ax: mpl.axes.Axes,
    *,
    x: float,
    values: np.ndarray,
    corrected_p: float,
    span: float,
    fontsize: float,
) -> float:
    """Annotate one participant box+jitter distribution and return marker y."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return np.nan
    marker = corrected_significance_label(float(corrected_p))
    y = max(0.0, float(np.max(finite))) + .030 * span
    significant = marker != "n.s."
    if not significant:
        return y
    ax.text(
        x, y, marker, ha="center", va="bottom",
        fontsize=fontsize,
        fontweight="bold",
        color=INK,
        alpha=1.0,
        clip_on=False, zorder=7,
    )
    return y


def draw_allocation_game(
    ax: mpl.axes.Axes,
    game: int,
    *,
    show_legend: bool,
    show_xticklabels: bool = True,
) -> None:
    frames=[]
    for one_game in (1,2):
        d=read(OUT/"02_evaluation_revision_geometry"/f"group_game{one_game}"/"results"/"allocation_marginal_subject_effects.csv")
        d["game"]=one_game; frames.append(d)
    d=pd.concat(frames,ignore_index=True)
    inference=read(
        OUT/"02_evaluation_revision_geometry"/f"group_game{game}"/
        "results"/"allocation_marginal_group_effects.csv"
    )
    outcomes=["root_branch_allocation_diversity","mean_bout_terminal_depth","deep_event_fraction"]
    nice={outcomes[0]:"breadth",outcomes[1]:"depth",outcomes[2]:"deep %"}
    positions=[0,1,2,4,5,6]
    all_values=d[["beta_M","beta_Q_signed"]].to_numpy(float)
    data_min=min(0.0,float(np.nanmin(all_values)))
    data_max=max(0.0,float(np.nanmax(all_values)))
    span=max(data_max-data_min,.1)
    pred=[("beta_M",NAVY,"margin $M$"),("beta_Q_signed",GREEN,"signed $\\Delta Q$")]
    rng=np.random.default_rng(410+game)
    annotations=[]
    k=0
    for tr in ("noop_to_active","active_to_noop"):
        for out in outcomes:
            q=d[(d.game==game)&(d.transition_type==tr)&(d.outcome==out)]
            for offset,(column,color,_) in zip((-.14,.14),pred):
                values=q[column].to_numpy(float)
                x=positions[k]+offset
                box_jitter(ax,values,x,
                           color,rng,width=.22)
                cell=inference[
                    inference.transition_type.eq(tr)&
                    inference.outcome.eq(out)&
                    inference.metric.eq(column)
                ]
                if len(cell)!=1:
                    raise ValueError(
                        f"Expected one allocation inference row: game={game}, "
                        f"transition={tr}, outcome={out}, metric={column}"
                    )
                annotations.append((x,values,float(cell.iloc[0].bh_fdr_q)))
            k+=1
    marker_top=max(
        max(0.0,float(np.nanmax(values)))+.030*span
        for _,values,_ in annotations
    )
    ylimits=(data_min-.10*span,max(data_max+.20*span,marker_top+.14*span))
    ax.axhline(0,color="#777777",lw=.6)
    ax.axvline(3,color="#c4c7c9",lw=.55)
    ax.set_xlim(-.6,6.6); ax.set_ylim(*ylimits)
    for x,values,p_value in annotations:
        annotate_corrected_significance(
            ax,x=x,values=values,corrected_p=p_value,span=span,fontsize=4.5
        )
    ax.set_xticks(positions)
    if show_xticklabels:
        ax.set_xticklabels(
            [nice[o] for o in outcomes] * 2,
            rotation=60, ha="right", fontsize=5.1,
        )
    else:
        ax.set_xticklabels([])
    ax.set_ylabel("standardized slope",labelpad=1)
    ax.set_title(GAME_NAMES[game],loc="left",pad=2,
                 fontsize=6.4,fontweight="bold")
    ax.text(.22,.98,"hold → act",transform=ax.transAxes,ha="center",
            va="top",fontsize=5.4,color=BLUE)
    ax.text(.78,.98,"act → hold",transform=ax.transAxes,ha="center",
            va="top",fontsize=5.4,color=ORANGE)
    ax.tick_params(axis="both",pad=1)
    finish_ax(ax,"")
    if show_legend:
        ax.legend(
            handles=[Patch(facecolor=c,edgecolor=c,alpha=.35,label=lab)
                     for _,c,lab in pred],
            frameon=False,ncol=2,loc="upper center",
            bbox_to_anchor=(.58,1.14),columnspacing=.6,
            handlelength=.9,handletextpad=.25,
        )


def draw_allocation_summary(
    parent_spec,
    fig: mpl.figure.Figure,
) -> list[mpl.axes.Axes]:
    sub = parent_spec.subgridspec(2, 1, hspace=.31)
    axes = []
    for row, game in enumerate((1, 2)):
        ax = fig.add_subplot(sub[row, 0]); axes.append(ax)
        draw_allocation_game(
            ax, game, show_legend=(row == 0),
            show_xticklabels=(row == 1),
        )
        if row == 0:
            ax.set_ylabel("")
    return axes


def draw_behavior_validation_game(
    ax: mpl.axes.Axes,
    game: int,
    *,
    show_legend: bool,
    show_xlabel: bool,
) -> None:
    comparisons=["A1_to_A2","A2_to_A3","A2_to_A4","A2_to_A5"]
    comp_names={"A1_to_A2":"root block","A2_to_A3":"compact","A2_to_A4":"mean29","A2_to_A5":"concat29"}
    outcomes=["exact_action","noop_to_active","active_to_noop"]
    colors={"exact_action":NAVY,"noop_to_active":BLUE,"active_to_noop":ORANGE}
    offsets=np.array([-.22,0,.22])
    d=read(OUT/"05_planning_trace_behavioral_validation_v4"/f"group_game{game}"/"tables"/"subject_effects.csv")
    inference=read(
        OUT/"05_planning_trace_behavioral_validation_v4"/f"group_game{game}"/
        "tables"/"exact_inference.csv"
    )
    rng=np.random.default_rng(100+game)
    annotations=[]
    plotted_values=[]
    for ci,comp in enumerate(comparisons):
        for oi,out in enumerate(outcomes):
            vals=d[(d.comparison==comp)&(d.outcome==out)].delta_nll_positive_is_improvement.to_numpy(float)
            x=ci+offsets[oi]
            box_jitter(ax,vals,x,colors[out],rng,width=.17)
            plotted_values.append(vals)
            cell=inference[inference.effect_id.eq(f"{out}::{comp}")]
            if len(cell)!=1:
                raise ValueError(
                    f"Expected one behavioral-validation inference row: "
                    f"game={game}, outcome={out}, comparison={comp}"
                )
            annotations.append(
                (x,vals,float(cell.iloc[0].synchronized_max_t_p))
            )
    all_values=np.concatenate(plotted_values)
    data_min=min(0.0,float(np.nanmin(all_values)))
    data_max=max(0.0,float(np.nanmax(all_values)))
    span=max(data_max-data_min,1e-4)
    marker_top=max(
        max(0.0,float(np.nanmax(values)))+.030*span
        for _,values,_ in annotations
    )
    ax.set_ylim(data_min-.10*span,max(data_max+.20*span,marker_top+.14*span))
    ax.axhline(0,color="#777777",lw=.65)
    for x,values,p_value in annotations:
        annotate_corrected_significance(
            ax,x=x,values=values,corrected_p=p_value,span=span,fontsize=4.25
        )
    ax.set_xticks(range(4))
    if show_xlabel:
        ax.set_xticklabels([comp_names[c] for c in comparisons],
                           rotation=60,ha="right",fontsize=4.9)
    else:
        ax.set_xticklabels([])
    ax.set_ylabel(r"held-out $\Delta$NLL / decision",labelpad=1)
    ax.set_title(GAME_NAMES[game],loc="left",pad=1,
                 fontsize=5.8,fontweight="bold")
    ax.tick_params(axis="both",pad=1)
    finish_ax(ax,"")
    if show_legend:
        short_names = {
            "exact_action": "switch",
            "noop_to_active": "hold → act",
            "active_to_noop": "act → hold",
        }
        ax.legend(
            handles=[Patch(facecolor=colors[o],edgecolor=colors[o],alpha=.45,
                           label=short_names[o]) for o in outcomes],
            frameon=False,ncol=3,loc="lower center",
            bbox_to_anchor=(.58,1.07),columnspacing=.45,
            handlelength=.55,handletextpad=.20,borderaxespad=.1,
            fontsize=4.6,
        )


def draw_behavior_validation(
    parent_spec,
    fig: mpl.figure.Figure,
) -> list[mpl.axes.Axes]:
    sub=parent_spec.subgridspec(2,1,hspace=.38)
    axes=[]
    for row,game in enumerate((1,2)):
        ax=fig.add_subplot(sub[row,0]); axes.append(ax)
        draw_behavior_validation_game(
            ax,game,show_legend=(row==0),show_xlabel=(row==1)
        )
    return axes


def main_figure_2(path:Path)->None:
    # C is the inferential center of the figure.  The left block contains the
    # schematic, the matching full-risk-set coefficient summary, and the two
    # downstream analyses.  Preserve their physical width while widening C so
    # each of its four heatmap-plus-marginal cells has a square main plane.
    figure_width = 11.6
    fig=plt.figure(figsize=(figure_width,6.6))
    fig.subplots_adjust(left=.100,right=.961,bottom=.110,top=.955,
                        wspace=.18,hspace=.42)
    outer=fig.add_gridspec(1,2,width_ratios=[1.0,1.488],wspace=.18)
    left=outer[0,0].subgridspec(
        3,1,height_ratios=[.45,.75,1.94],hspace=.30
    )
    a=fig.add_subplot(left[0,0]); draw_process_proxy(a); panel(a,"A",-.07,1.05)
    b=fig.add_subplot(left[1,0]); draw_transition_effect_summary(b)
    panel(b,"B",-.07,1.10)

    lower=left[2,0].subgridspec(1,2,width_ratios=[1.0,1.12],wspace=.28)
    daxes=draw_allocation_summary(lower[0,0],fig)
    panel(daxes[0],"D",-.18,1.18)
    eaxes=draw_behavior_validation(lower[0,1],fig)
    panel(eaxes[0],"E",-.18,1.18)

    caxes=draw_transition_surfaces(outer[0,1],fig)
    panel(caxes[0],"C",-.25,1.12)
    fig.text(
        .985,.018,
        "* corrected $p<.05$; unmarked = n.s.  |  "
        "B/D: BH-FDR/game  |  E: synchronized max-$|t|$",
        ha="right",va="bottom",fontsize=3.4,color=GRAY,
    )
    enforce_minimum_visual_scale(fig)
    fig.canvas.draw()
    for ax in caxes:
        bounds = ax.get_position()
        physical_aspect = (
            bounds.width * fig.get_figwidth()
            / (bounds.height * fig.get_figheight())
        )
        if not .99 <= physical_aspect <= 1.01:
            raise RuntimeError(
                f"Figure 2C main plane is not square: aspect={physical_aspect:.3f}"
            )
    save(fig,path)


# ---------------------------------------------------------------------------
# Main Figure 3: corrected parcel effects on an anatomical MNI template


MNI_ATLAS_ROOT = (
    ROOT / "legacy" / "06_representational_mechanism" / "atlas" /
    "harvard_oxford" / "fsl" / "data" / "atlases"
)
MNI_TEMPLATE = Path(
    "/home/jeongmin/fmri/atari/derivatives/ants_mni/sub001-1/Session2/"
    "MNI152_T1_2.5x2.5x2.5mm_brain.nii"
)
CROSS_RED_YELLOW = mpl.colors.LinearSegmentedColormap.from_list(
    "cross_red_yellow",
    ("#C8191E", "#ED3B24", "#F57C20", "#FDBE32", "#FFE45C"),
)
HARVARD_OXFORD_SUBCORTICAL_REGIONS = (
    "Thalamus", "Caudate", "Putamen", "Pallidum",
    "Hippocampus", "Amygdala", "Accumbens",
)


def atlas_label_key(prefix: str, label: str) -> str:
    return prefix + re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def mni_projection_resources() -> dict[str, object]:
    """Load the 110-ROI atlas and a display-only 1-mm MNI T1 underlay."""
    from nibabel.processing import resample_from_to

    resources: list[tuple[np.ndarray, dict[str, int]]] = []
    brain = None
    reference = None
    contracts = (
        (
            "ho_cortical_",
            "HarvardOxford/HarvardOxford-cortl-maxprob-thr25-1mm.nii.gz",
            "HarvardOxford-Cortical-Lateralized.xml",
            False,
        ),
        (
            "ho_subcortical_",
            "HarvardOxford/HarvardOxford-sub-maxprob-thr25-1mm.nii.gz",
            "HarvardOxford-Subcortical.xml",
            True,
        ),
    )
    for prefix, nii_name, xml_name, is_subcortical in contracts:
        nii_path = MNI_ATLAS_ROOT / nii_name
        xml_path = MNI_ATLAS_ROOT / xml_name
        if not nii_path.exists() or not xml_path.exists():
            raise FileNotFoundError(f"MNI atlas resource missing: {nii_path} or {xml_path}")
        atlas_img = nib.as_closest_canonical(nib.load(str(nii_path)))
        if reference is None:
            reference = atlas_img
        elif atlas_img.shape != reference.shape or not np.allclose(
            atlas_img.affine, reference.affine
        ):
            raise ValueError("Harvard-Oxford cortical/subcortical grids differ")
        labels = {
            atlas_label_key(prefix, element.text or ""): int(element.attrib["index"]) + 1
            for element in ET.parse(xml_path).getroot().findall(".//label")
        }
        if is_subcortical:
            labels = {
                key: value for key, value in labels.items()
                if any(region.lower() in key for region in HARVARD_OXFORD_SUBCORTICAL_REGIONS)
            }
        atlas = np.asarray(atlas_img.dataobj, dtype=np.int16)
        resources.append((atlas, labels))
        brain = (atlas > 0) if brain is None else (brain | (atlas > 0))
    if reference is None or sum(len(labels) for _, labels in resources) != 110:
        raise ValueError("Expected the 110-region Harvard-Oxford analysis atlas")
    if not MNI_TEMPLATE.exists():
        raise FileNotFoundError(f"MNI anatomical template missing: {MNI_TEMPLATE}")
    anatomy_img = resample_from_to(
        nib.as_closest_canonical(nib.load(str(MNI_TEMPLATE))),
        (reference.shape, reference.affine), order=3,
    )
    anatomy = np.asarray(anatomy_img.dataobj, dtype=np.float32)
    anatomy[~np.isfinite(anatomy) | (anatomy < 0)] = 0.0
    return {
        "atlases": resources, "anatomy": anatomy,
        "brain": brain, "reference": reference,
    }


def lateral_outer_projection(volume: np.ndarray, *, reverse: bool) -> np.ndarray:
    """Project the first significant parcel encountered from the lateral surface."""
    ordered = volume[::-1] if reverse else volume
    valid = np.isfinite(ordered)
    first = np.argmax(valid, axis=0)
    projection = np.take_along_axis(ordered, first[None, ...], axis=0)[0]
    projection[~np.any(valid, axis=0)] = np.nan
    return projection


def anatomical_projection(
    anatomy: np.ndarray, brain: np.ndarray, axis: int = 0
) -> np.ndarray:
    import warnings

    values = np.where(brain & (anatomy > 0), anatomy, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        projection = np.nanpercentile(values, 70, axis=axis)
    finite = projection[np.isfinite(projection)]
    if finite.size:
        low, high = np.percentile(finite, [3, 99])
        projection = np.clip((projection - low) / max(high - low, 1e-8), 0, 1)
    return np.asarray(projection, dtype=np.float32)


def draw_mni_projection(
    ax: mpl.axes.Axes,
    effects: pd.DataFrame,
    value_col: str,
    title: str,
    norm: Normalize,
    resources: dict[str, object],
) -> mpl.cm.ScalarMappable:
    atlases = resources["atlases"]
    anatomy = resources["anatomy"]
    brain_volume = resources["brain"]
    shape = atlases[0][0].shape
    volume = np.full(shape, np.nan, dtype=np.float32)
    matched: set[str] = set()
    midpoint = shape[0] // 2
    for atlas, labels in atlases:
        for row in effects.itertuples(index=False):
            roi = str(row.roi)
            if roi not in labels:
                continue
            parcel = atlas == labels[roi]
            # Unilateral labels can include a few midline voxels.  Restrict
            # display to the named hemisphere without altering the inference.
            if "_left_" in roi:
                parcel[midpoint:] = False
            elif "_right_" in roi:
                parcel[:midpoint] = False
            volume[parcel] = float(getattr(row, value_col))
            matched.add(roi)
    expected = set(effects.roi.astype(str))
    if not expected or matched != expected:
        raise ValueError(f"Unmatched corrected ROIs for {title}: {sorted(expected-matched)}")

    bg_left = anatomical_projection(
        anatomy[:midpoint], brain_volume[:midpoint], axis=0
    ).T
    bg_right = anatomical_projection(
        anatomy[midpoint:], brain_volume[midpoint:], axis=0
    ).T
    effect_left = lateral_outer_projection(
        volume[:midpoint], reverse=False
    ).T
    effect_right = lateral_outer_projection(
        volume[midpoint:], reverse=True
    ).T
    gap = np.full((bg_left.shape[0], 8), np.nan, dtype=np.float32)
    background = np.concatenate([np.fliplr(bg_left), gap, bg_right], axis=1)
    effect = np.concatenate([np.fliplr(effect_left), gap, effect_right], axis=1)
    brain_left = np.any(brain_volume[:midpoint], axis=0).T
    brain_right = np.any(brain_volume[midpoint:], axis=0).T
    brain = np.concatenate(
        [np.fliplr(brain_left), np.zeros((brain_left.shape[0], 8), bool), brain_right],
        axis=1,
    )
    ax.imshow(
        np.ma.masked_where(~brain, background), origin="lower", cmap="Greys",
        vmin=0, vmax=1, interpolation="bicubic", alpha=.72,
    )
    image = ax.imshow(
        np.ma.masked_invalid(effect), origin="lower", cmap=CROSS_RED_YELLOW,
        norm=norm, interpolation="nearest", alpha=.96,
    )
    ax.contour(brain.astype(float), levels=[.5], colors="#62696d",
               linewidths=.30, origin="lower")
    ax.set_axis_off()
    ax.set_title(title, loc="center", pad=2, fontsize=6.4, fontweight="normal", color="black")
    ax.text(.24, .025, "L", transform=ax.transAxes, ha="center", fontsize=5, color=GRAY)
    ax.text(.76, .025, "R", transform=ax.transAxes, ha="center", fontsize=5, color=GRAY)
    return image


ROI_ORDER = ("vlPFC", "dlPFC", "OFC", "ACC", "SMA_preSMA", "V1", "HPC", "vStr")
ROI_LABELS = {
    "vlPFC": "vlPFC", "dlPFC": "dlPFC", "OFC": "OFC", "ACC": "ACC",
    "SMA_preSMA": "SMA/preSMA", "V1": "V1", "HPC": "HPC", "vStr": "vStr",
}
ROI_COLORS = dict(zip(ROI_ORDER, plt.cm.tab10(np.linspace(.05, .85, len(ROI_ORDER)))))
# Display order follows root anchor -> model-mediated search -> readout/boundary.
# It does not alter the frozen staged-control order used by 07_encoding_analysis.
RSA_EFFECTS = ("P", "Q0", "M", "E", "R", "G_joint", "A", "switch_status_interaction")
RSA_EFFECT_TITLES = {
    "P": "Pre-search policy\n$\\pi_t^0$",
    "M": "Incumbent margin\n$M_t$",
    "Q0": "Initial Q state\n$Q_t^0$",
    "R": "Value-update magnitude\n$\\log U_t$",
    "E": "Rollout advantage\n$\\mathcal{A}^{roll}_t$",
    "G_joint": "Search allocation\n$\\mathcal{G}_t$",
    "A": "Post-search Actor residual\n$A_t^{res}$",
    "switch_status_interaction": "Hold/act state change\n$Y_t^{status}$",
}


def draw_rsa_roi_distribution(
    ax: mpl.axes.Axes,
    values: pd.DataFrame,
    tests: pd.DataFrame,
    *,
    value_column: str,
    title: str,
    show_xticklabels: bool,
    roi_order: tuple[str, ...] = ROI_ORDER,
) -> None:
    """Reproduce the frozen 07 RSA ROI box-and-participant display."""
    if set(roi_order) != set(ROI_ORDER) or len(roi_order) != len(ROI_ORDER):
        raise ValueError(f"Invalid ROI display order: {roi_order}")
    x=np.arange(len(roi_order),dtype=float)
    data=[]
    for roi_key in roi_order:
        one=values[values.roi_key.eq(roi_key)][value_column].to_numpy(float)
        if len(one)!=6 or not np.isfinite(one).all():
            raise ValueError(f"Expected six finite values for {title}/{roi_key}")
        data.append(one)
    ax.boxplot(
        data,positions=x,widths=.58,patch_artist=True,showfliers=False,
        medianprops={"color":"#222426","linewidth":.8},
        whiskerprops={"color":"#6C737A","linewidth":.6},
        capprops={"color":"#6C737A","linewidth":.6},
        boxprops={"facecolor":"#E9EDF1","edgecolor":"#6C737A","linewidth":.6},
    )
    data_min=min(0.0,min(float(np.min(one)) for one in data))
    data_max=max(0.0,max(float(np.max(one)) for one in data))
    data_span=max(data_max-data_min,.02)
    marker_y=data_max+.08*data_span
    for roi_index,(roi_key,one) in enumerate(zip(roi_order,data)):
        ax.scatter(
            roi_index+np.linspace(-.16,.16,len(one)),one,s=11,
            color=ROI_COLORS[roi_key],edgecolor="white",linewidth=.25,zorder=3,
        )
        row=tests[tests.roi_key.eq(roi_key)]
        if len(row)!=1:
            raise ValueError(f"Expected one inference row for {title}/{roi_key}")
        p_corrected=float(row.iloc[0].p_max_abs_t_fwer)
        marker=corrected_significance_label(p_corrected)
        if marker != "n.s.":
            ax.text(roi_index,marker_y,marker,ha="center",va="bottom",fontsize=7)
    ax.set_ylim(data_min-.12*data_span,data_max+.30*data_span)
    ax.axhline(0,color="#8B9094",lw=.55,ls="--")
    ax.set_title(title,fontsize=7.2,fontweight="normal",pad=2)
    ax.set_xticks(x)
    if show_xticklabels:
        ax.set_xticklabels([ROI_LABELS[key] for key in roi_order],rotation=30,ha="right")
    else:
        ax.set_xticklabels([])
    ax.tick_params(labelsize=5.8)
    ax.grid(axis="y",color="#E1E4E7",lw=.45)
    ax.spines[["top","right"]].set_visible(False)


def corrected_roi_order(tests: pd.DataFrame) -> tuple[str, ...]:
    """Order ROIs by corrected p, preserving the canonical order for ties."""
    corrected = {}
    for roi_key in ROI_ORDER:
        row = tests[tests.roi_key.eq(roi_key)]
        if len(row) != 1:
            raise ValueError(f"Expected one corrected test row for {roi_key}")
        corrected[roi_key] = float(row.iloc[0].p_max_abs_t_fwer)
    canonical_index = {roi_key: index for index, roi_key in enumerate(ROI_ORDER)}
    return tuple(sorted(
        ROI_ORDER,
        key=lambda roi_key: (corrected[roi_key], canonical_index[roi_key]),
    ))


def draw_trace_roi_game(ax:mpl.axes.Axes,game:int)->None:
    result_dir=OUT/"07_RSA"/f"group_game{game}"/"planning_trace"/"results"
    values=read(result_dir/"predefined_roi_subject_effects.csv")
    tests=read(result_dir/"predefined_roi_inference.csv")
    values=values[(values.game==game)&values.trace.eq("planning_concat29")]
    tests=tests[(tests.game==game)&tests.trace.eq("planning_concat29")]
    draw_rsa_roi_distribution(
        ax,values,tests,value_column="mean_conditional_partial_r",
        title="Ordered concat29",show_xticklabels=True,
        roi_order=corrected_roi_order(tests),
    )
    ax.set_ylabel("Conditional partial Spearman $r$")


def main_figure_3(path:Path)->None:
    fig=plt.figure(figsize=(WIDTH,5.6))
    fig.subplots_adjust(left=.085,right=.990,bottom=.105,top=.925,hspace=.68)
    gs=fig.add_gridspec(2,1)
    axes=[]
    for row,game in enumerate((1,2)):
        ax=fig.add_subplot(gs[row,0]); axes.append(ax)
        draw_trace_roi_game(ax,game)
        ax.text(.5,1.18,f"{GAME_NAMES[game]} · full-trace predefined ROIs",
                transform=ax.transAxes,ha="center",va="bottom",fontsize=8.2)
        panel(ax,"A" if game==1 else "B",-.075,1.20)
    fig.text(
        .5,.025,
        "points: participants  |  ROIs ordered by corrected p; ties preserve canonical order  |  "
        "stars: max-$|t|$ FWER across 8 ROIs (* $p<.05$, ** $p<.01$, *** $p<.001$)",
        ha="center",fontsize=5.35,color=GRAY,
    )
    save(fig,path)


def draw_native_roi_game(parent_spec,fig:mpl.figure.Figure,game:int)->list[mpl.axes.Axes]:
    result_dir=OUT/"07_RSA"/f"group_game{game}"/"results"
    values=read(result_dir/"predefined_roi_subject_effects.csv")
    tests=read(result_dir/"predefined_roi_inference.csv")
    values=values[(values.game==game)&values.statistic.eq("conditional_partial_r")]
    tests=tests[(tests.game==game)&tests.statistic.eq("conditional_partial_r")]
    sub=parent_spec.subgridspec(2,4,wspace=.22,hspace=.66)
    axes=[]
    for index,effect in enumerate(RSA_EFFECTS):
        ax=fig.add_subplot(sub[index//4,index%4]); axes.append(ax)
        draw_rsa_roi_distribution(
            ax,values[values.effect.eq(effect)],tests[tests.effect.eq(effect)],
            value_column="mean_effect",title=RSA_EFFECT_TITLES[effect],
            show_xticklabels=index>=4,
        )
    return axes


def supp_figure_3(path:Path)->None:
    fig=plt.figure(figsize=(12.8,10.0))
    fig.subplots_adjust(left=.045,right=.995,bottom=.075,top=.945,hspace=.42)
    outer=fig.add_gridspec(2,1,hspace=.42)
    game_axes=[]
    for row,game in enumerate((1,2)):
        axes=draw_native_roi_game(outer[row,0],fig,game); game_axes.append(axes)
    fig.canvas.draw()
    for game,axes,letter in zip((1,2),game_axes,("A","B")):
        top=max(ax.get_position().y1 for ax in axes[:4])
        bottom=min(ax.get_position().y0 for ax in axes[4:])
        fig.text(.5,top+.032,f"{GAME_NAMES[game]} · literature-defined ROI effects",
                 ha="center",va="bottom",fontsize=8.2)
        fig.text(.018,top+.032,letter,ha="left",va="bottom",fontsize=10,fontweight="bold")
        fig.text(.5,bottom-.055,
                 "points: participants  |  stars: max-$|t|$ FWER within each 8-ROI effect family "
                 "(* $p<.05$, ** $p<.01$, *** $p<.001$)",
                 ha="center",va="top",fontsize=5.7,color=GRAY)
    save(fig,path)


def draw_hazard_generalization(ax:mpl.axes.Axes)->None:
    d=read(BEH/"1-5_day_heldout_hazard_subject.csv")
    rng=np.random.default_rng(150)
    for game,x in ((1,0),(2,1)):
        values=d[d.game==game].bout_weighted_delta_heldout_log_likelihood_per_bout.to_numpy(float)
        box_jitter(ax,values,x,GAME_COLORS[game],rng,width=.42)
    ax.axhline(0,color="#777777",lw=.65)
    ax.set_xticks([0,1],[GAME_NAMES[1],GAME_NAMES[2]])
    ax.set_xlabel("Game")
    ax.set_ylabel(r"held-out $\Delta$LL / bout")
    ax.set_title("Held-out-day generalization",pad=2)
    finish_ax(ax,"")


def km_survival(length:np.ndarray,event:np.ndarray,times:np.ndarray)->np.ndarray:
    surv=1.0; out=[]
    for t in times:
        at=np.sum(length>=t); ev=np.sum((length==t)&event)
        if at>0: surv*=1.0-ev/at
        out.append(surv)
    return np.asarray(out)


def _subject_id(value: object) -> int:
    match = re.search(r"(\d+)$", str(value))
    if match is None:
        raise ValueError(f"Cannot parse participant identifier: {value}")
    return int(match.group(1))


def draw_survival(parent_spec,fig:mpl.figure.Figure)->list[mpl.axes.Axes]:
    # Preserve the source figure's 2 game rows × 6 participant columns.
    sub=parent_spec.subgridspec(3,6,height_ratios=[.18,1,1],wspace=.27,hspace=.38)
    legend_ax=fig.add_subplot(sub[0,:]); legend_ax.axis("off")
    legend_ax.legend(
        handles=[
            Line2D([0],[0],color=INK,lw=1.0,label="observed KM"),
            Line2D([0],[0],color=RED,lw=.8,ls="--",label="constant hazard"),
            Line2D([0],[0],color=BLUE,lw=.8,label="duration-dependent"),
        ],
        loc="center",ncol=3,frameon=False,fontsize=4.2,
        handlelength=1.35,columnspacing=.65,handletextpad=.25,
    )
    bouts=read(BEH/"1-6_behavioral_bouts_with_censoring.csv")
    pars=read(BEH/"1-6_noop_duration_hazard_model_summary.csv")
    axes=[]
    for game in (1,2):
        qq=bouts[(bouts.game==game)&(bouts.bout_type=="NOOP")]
        for subject_name in sorted(qq.subject.unique(),key=_subject_id):
            subject=_subject_id(subject_name)
            row=game-1; col=subject-1
            ax=fig.add_subplot(sub[row+1,col]); axes.append(ax)
            s=qq[qq.subject==subject_name]
            r=pars[(pars.game==game)&(pars.subject==subject_name)]
            if len(r)!=1:
                raise ValueError(f"Expected one frozen hazard row for game {game}, {subject_name}")
            r=r.iloc[0]
            max_t=int(max(2,min(250,s.length.quantile(.99))))
            times=np.arange(1,max_t+1)
            observed=km_survival(s.length.to_numpy(int),(~s.censored).to_numpy(bool),times)
            geometric=np.power(float(r.geometric_q),times)
            duration=np.power(float(r.duration_q),np.power(times,float(r.duration_beta)))
            ax.step(times,observed,where="post",color=INK,lw=.85)
            ax.plot(times,geometric,color=RED,lw=.68,ls="--")
            ax.plot(times,duration,color=BLUE,lw=.72)
            ax.set_xscale("log"); ax.set_xlim(1,max_t); ax.set_ylim(0,1.02)
            ax.set_yticks([0,.5,1]); ax.set_title(f"S{subject}",pad=.5,fontsize=5.0)
            if col==0: ax.set_ylabel(f"{GAME_NAMES[game]}\nsurvival",fontsize=4.5,labelpad=1)
            else: ax.set_yticklabels([])
            if game==2: ax.set_xlabel("duration (steps)",fontsize=4.1,labelpad=.5)
            else: ax.set_xticklabels([])
            ax.tick_params(axis="both",labelsize=3.8,pad=.5,length=1.5)
            finish_ax(ax)
    if len(axes)!=12:
        raise ValueError(f"Expected 12 participant-resolved survival panels, found {len(axes)}")
    return axes


def draw_competing_risk(parent_spec,fig:mpl.figure.Figure)->list[mpl.axes.Axes]:
    sub=parent_spec.subgridspec(3,6,height_ratios=[.18,1,1],wspace=.27,hspace=.38)
    legend_ax=fig.add_subplot(sub[0,:]); legend_ax.axis("off")
    legend_ax.legend(
        handles=[
            Line2D([0],[0],color=GRAY,lw=.75,ls="--",label="NOOP persist"),
            Line2D([0],[0],color=INK,lw=.85,label="same-action persist"),
            Line2D([0],[0],color=BLUE,lw=.75,label="active → NOOP"),
            Line2D([0],[0],color=ORANGE,lw=.75,ls=":",label="active → other"),
        ],
        loc="center",ncol=2,frameon=False,fontsize=4.1,
        handlelength=1.35,columnspacing=.75,handletextpad=.25,
    )
    d=read(BEH/"1-8_action_persistence_competing_risk_curves.csv")
    require_columns(d,["game","subject","time","noop_persistence","active_persistence",
                       "cif_to_noop","cif_to_other_active"],"competing-risk curves")
    axes=[]
    for game in (1,2):
        q=d[d.game==game]
        for subject_name in sorted(q.subject.unique(),key=_subject_id):
            subject=_subject_id(subject_name)
            row=game-1; col=subject-1
            ax=fig.add_subplot(sub[row+1,col]); axes.append(ax)
            s=q[q.subject==subject_name].sort_values("time")
            max_t=int(min(300,s.time.max())); s=s[s.time<=max_t]
            ax.plot(s.time,s.noop_persistence,color=GRAY,lw=.68,ls="--")
            ax.plot(s.time,s.active_persistence,color=INK,lw=.82)
            ax.plot(s.time,s.cif_to_noop,color=BLUE,lw=.72)
            ax.plot(s.time,s.cif_to_other_active,color=ORANGE,lw=.72,ls=":")
            ax.set_xscale("log"); ax.set_xlim(1,max_t); ax.set_ylim(0,1.02)
            ax.set_yticks([0,.5,1]); ax.set_title(f"S{subject}",pad=.5,fontsize=5.0)
            if col==0: ax.set_ylabel(f"{GAME_NAMES[game]}\nprobability",fontsize=4.5,labelpad=1)
            else: ax.set_yticklabels([])
            if game==2: ax.set_xlabel("run duration (steps)",fontsize=4.1,labelpad=.5)
            else: ax.set_xticklabels([])
            ax.tick_params(axis="both",labelsize=3.8,pad=.5,length=1.5)
            finish_ax(ax)
    if len(axes)!=12:
        raise ValueError(f"Expected 12 participant-resolved competing-risk panels, found {len(axes)}")
    return axes


def supp_figure_1(path:Path)->None:
    fig=plt.figure(figsize=(WIDTH,5.15))
    fig.subplots_adjust(left=.075,right=.985,bottom=.065,top=.955,hspace=.42)
    gs=fig.add_gridspec(3,1,height_ratios=[.62,1.55,1.55])
    a=fig.add_subplot(gs[0]); draw_hazard_generalization(a)
    baxes=draw_survival(gs[1],fig)
    caxes=draw_competing_risk(gs[2],fig)
    fig.canvas.draw()
    for anchor,letter in ((a,"A"),(baxes[0],"B"),(caxes[0],"C")):
        box=anchor.get_position()
        y=.975 if letter=="A" else min(.985,box.y1+.065)
        fig.text(.035 if letter=="A" else max(.006,box.x0-.055),y,letter,
                 fontsize=10,fontweight="bold",ha="left",va="top")
    save(fig,path)


# ---------------------------------------------------------------------------
# Supplementary Figure 2: model validation




def learning_curves(parent_spec, fig: mpl.figure.Figure) -> list[mpl.axes.Axes]:
    """Two training panels plus 12 participant-resolved grouped bar panels."""
    # Training curves occupy the top-left/right cells.  The 12 action facets
    # use four rows × three columns so all six action IDs remain readable in
    # the left manuscript block.
    sub = parent_spec.subgridspec(6,12,
                                  height_ratios=[1.14,.18,.72,.72,.72,.72],
                                  wspace=.26,hspace=.58)
    axes = []
    raw = read(ROOT / "game_return.csv")
    summary = read(OUT / "00_learningcurve" / "results" /
                   "learning_curve_series_summary.csv")
    action = read(OUT / "00_learningcurve" / "results" /
                  "subject_action_distribution_counts.csv")

    for j, (key, game) in enumerate((("pong", 1), ("spaceinvaders", 2))):
        train_slice=slice(0,5) if j==0 else slice(7,12)
        ax = fig.add_subplot(sub[0,train_slice]); axes.append(ax)
        cols = sorted(
            [c for c in raw.columns if c.startswith(key + "_sub")
             and c.endswith(" - rmean_episode_return")],
            key=lambda name: int(re.search(r"_sub(\d+)_", name).group(1)),
        )
        if len(cols) != 6:
            raise ValueError(f"Expected six learning curves for {key}, found {len(cols)}")
        for c in cols:
            match = re.search(r"_sub(\d+)_", c)
            subj = int(match.group(1))
            ok = raw[c].notna() & raw.Step.le(2_000_000)
            line, = ax.plot(raw.loc[ok, "Step"] / 1e6, raw.loc[ok, c],
                            lw=.58, alpha=.86, label="_nolegend_")
            href = summary[(summary.game == key) & (summary.subject == subj)]
            if len(href) != 1:
                raise ValueError(f"Expected one human reference for {key} S{subj}")
            ax.axhline(float(href.human_mean_recording_defined_segment_return.iloc[0]),
                       color=line.get_color(), ls=(0, (3, 2)), lw=.52,
                       alpha=.74, label="_nolegend_")
        ax.set_title(f"{GAME_NAMES[game]} training", pad=2)
        ax.set_xlabel("Steps (million)", labelpad=1)
        if j == 0:
            ax.set_ylabel("running episode return")
        else:
            ax.tick_params(axis="y",left=False,labelleft=False)
        finish_ax(ax)
        ax.tick_params(axis="x", pad=1)

    action_ids = sorted(action.action_id.unique())
    expected_ids = list(range(6))
    if action_ids != expected_ids:
        raise ValueError(f"Expected action IDs 0–5, found {action_ids}")
    human_color = "#b7bdc2"
    replay_color = BLUE
    legend_ax = fig.add_subplot(sub[1, :]); legend_ax.axis("off")
    legend_ax.legend(
        handles=[
            Line2D([0],[0],color=GRAY,lw=.8,label="participant model curve"),
            Line2D([0],[0],color=GRAY,lw=.8,ls=(0,(3,2)),
                   label="participant human reference"),
            Rectangle((0,0),1,1,facecolor=human_color,edgecolor=INK,lw=.45,
                      label="Human action"),
            Rectangle((0,0),1,1,facecolor=replay_color,edgecolor=INK,lw=.45,
                      label="Thinker action during replay"),
        ],
        loc="center",ncol=4,frameon=False,fontsize=3.7,
        handlelength=1.15,columnspacing=.55,handletextpad=.25,
    )
    for game in (1,2):
        qg = action[action.game_id == game]
        ymax = min(1.0, max(.15, float(qg.proportion.max()) * 1.06))
        for col, subj in enumerate(range(1, 7)):
            action_row=(game-1)*2+(subj-1)//3
            action_col=(subj-1)%3
            ax = fig.add_subplot(sub[action_row+2,action_col*4:(action_col+1)*4]); axes.append(ax)
            qs = qg[qg.subject == subj]
            for source,color,xoff in [
                ("human_observed",human_color,-.19),
                ("thinker_replay_recommendation",replay_color,.19),
            ]:
                z = qs[qs.source == source].set_index("action_id").reindex(action_ids)
                if z.proportion.isna().any():
                    raise ValueError(f"Missing action cells: game {game}, S{subj}, {source}")
                ax.bar(np.asarray(action_ids,dtype=float)+xoff,z.proportion.to_numpy(float),
                       width=.38,color=color,edgecolor=INK,lw=.28,zorder=2)
            qa = qs.iloc[0]
            ax.text(.97,.97,
                    f"A={qa.exact_action_agreement:.2f} · TV={qa.total_variation_distance:.2f}",
                    transform=ax.transAxes,ha="right",va="top",fontsize=3.8,color=GRAY,
                    bbox=dict(facecolor="white",edgecolor="none",alpha=.72,pad=.2),zorder=4)
            ax.set_ylim(0, ymax)
            ax.set_xticks(action_ids)
            if action_row in (1,3): ax.set_xlabel("Action index",fontsize=4.1,labelpad=.5)
            if action_col == 0: ax.set_ylabel("proportion", fontsize=4.8, labelpad=1)
            else: ax.set_yticklabels([])
            game_short = "Pong" if game == 1 else "Space Inv."
            ax.set_title(f"{game_short} · S{subj}",pad=.8,fontsize=4.8)
            finish_ax(ax)
            ax.tick_params(axis="both",labelsize=3.8,pad=.5,length=1.4)
    return axes


def coefficient_panel(ax:mpl.axes.Axes,frames:list[pd.DataFrame],effects:list[str],cols:dict[str,str],title:str)->None:
    d=pd.concat(frames,ignore_index=True); rng=np.random.default_rng(314)
    for game in (1,2):
        for i,effect in enumerate(effects):
            q=d[(d.game==game)&(d.effect==effect)] if "effect" in d else d[(d.game==game)]
            col=cols[effect]
            vals=q[col].to_numpy(float)
            x=i+(game-1)*.32-.16
            box_jitter(ax,vals,x,GAME_COLORS[game],rng,width=.25)
    ax.axhline(0,color="#888",lw=.65); ax.set_xticks(range(len(effects)),[e.replace("predicted_update_scale","prediction-error scale").replace("log_prior_visits","prior visits").replace("event_progress","event position") for e in effects],rotation=25,ha="right")
    ax.set_ylabel("standardized coefficient"); ax.set_title(title,pad=2); finish_ax(ax,"")
    ax.legend(handles=[Patch(facecolor=GAME_COLORS[g],edgecolor=GAME_COLORS[g],alpha=.22,label=GAME_NAMES[g]) for g in (1,2)],frameon=False,loc="best")


def draw_actor_margin(ax:mpl.axes.Axes)->None:
    frames=[]
    for game in (1,2):
        d=read(OUT/"02_evaluation_revision_geometry"/f"group_game{game}"/"results"/"subject_effects.csv")
        d=d[d.outcome=="actor_margin_shift"].copy(); d["game"]=game; frames.append(d)
    d=pd.concat(frames); rng=np.random.default_rng(62)
    cols=[("beta_M","margin $M$"),("beta_V",r"revision $Q_{\mathrm{rev}}$"),("beta_MxV",r"$M\times Q_{\mathrm{rev}}$")]
    for game in (1,2):
        q=d[d.game==game]
        for i,(col,_) in enumerate(cols):
            vals=q[col].to_numpy(float); x=i+(game-1)*.32-.16
            box_jitter(ax,vals,x,GAME_COLORS[game],rng,width=.25)
    ax.axhline(0,color="#888",lw=.65); ax.set_xticks(range(3),[x[1] for x in cols]); ax.set_ylabel("standardized coefficient")
    ax.set_title("Actor-margin shift",pad=2); finish_ax(ax,"")
    ax.legend(handles=[Patch(facecolor=GAME_COLORS[g],edgecolor=GAME_COLORS[g],alpha=.22,label=GAME_NAMES[g]) for g in (1,2)],frameon=False)


def draw_q_mechanics(ax:mpl.axes.Axes)->None:
    frames=[]
    for game in (1,2):
        d=read(OUT/"03_actor_integration_mechanisms"/f"group_game{game}"/"tables"/"subject_effects.csv")
        d=d[d.scenario=="S3_q_update_mechanics_qc"].copy(); d["game"]=game; frames.append(d)
    coefficient_panel(ax,frames,["predicted_update_scale","log_prior_visits","depth","event_progress"],{e:"beta" for e in ["predicted_update_scale","log_prior_visits","depth","event_progress"]},"Q-update mechanics")


def draw_presearch_m(parent_spec,fig:mpl.figure.Figure)->list[mpl.axes.Axes]:
    frames=[]
    for game in (1,2):
        for s in range(1,7): frames.append(read(OUT/"05_planning_trace_behavioral_validation_v4"/f"sub{s:03d}_game{game}"/"tables"/"M_descriptive.csv.gz"))
    d=pd.concat(frames,ignore_index=True); sub=parent_spec.subgridspec(2,1,hspace=.42); axes=[]
    for j,game in enumerate((1,2)):
        ax=fig.add_subplot(sub[j,0]); axes.append(ax)
        q=d[d.game==game]
        for out,col in zip(["exact_switch","noop_to_active","active_to_noop"],[INK,BLUE,ORANGE]):
            z=q[q.outcome==out]
            agg=z.groupby("M_z_bin").apply(lambda x:pd.Series({"M":weighted_mean(x,"M_z_mean","n"),"rate":weighted_mean(x,"observed_rate","n")}),include_groups=False)
            ax.plot(agg.M,agg.rate,marker="o",ms=2.7,lw=.9,color=col,label=OUTCOME_NAMES[out])
        ax.set_title(GAME_NAMES[game],pad=2); ax.set_xlabel("pre-search margin $M$ (z)")
        if j==0: ax.set_ylabel("observed transition rate")
        finish_ax(ax)
    axes[1].legend(frameon=False,loc="upper right")
    return axes


def draw_transition_coefficients(parent_spec,fig:mpl.figure.Figure)->list[mpl.axes.Axes]:
    sub=parent_spec.subgridspec(2,2,wspace=.18,hspace=.34); axes=[]
    metrics=["beta_M","beta_V","beta_MxV"]; base_labels=["$M$",r"$Q_{\mathrm{rev}}$",r"$M\times Q_{\mathrm{rev}}$"]
    payload={}; lim=.05
    for game in (1,2):
        subject=read(OUT/"02_evaluation_revision_geometry"/f"group_game{game}"/"results"/"transition_specific_subject_effects.csv")
        group=read(OUT/"02_evaluation_revision_geometry"/f"group_game{game}"/"results"/"transition_specific_group_effects.csv")
        for tr in ("noop_to_active","active_to_noop"):
            mat=subject[subject.transition_type==tr].set_index("subject").reindex(range(1,7))[metrics].to_numpy(float)
            labels=[]
            for metric,label in zip(metrics,base_labels):
                cell=group[(group.transition_type==tr)&(group.metric==metric)]
                labels.append(label+("*" if len(cell)==1 and float(cell.bh_fdr_q.iloc[0])<.05 else ""))
            payload[(game,tr)]=(mat,labels); lim=max(lim,float(np.nanmax(np.abs(mat))))
    for row,game in enumerate((1,2)):
        for col,tr in enumerate(("noop_to_active","active_to_noop")):
            ax=fig.add_subplot(sub[row,col]); axes.append(ax)
            mat,labels=payload[(game,tr)]
            im=ax.imshow(mat,cmap="RdBu_r",vmin=-lim,vmax=lim,aspect="auto")
            ax.set_xticks(range(3),labels)
            if row==0:
                ax.set_xticklabels([])
            ax.set_yticks(range(6),[f"S{s}" for s in range(1,7)] if col==0 else [])
            game_title="Pong" if game==1 else "Space Inv."
            ax.set_title(f"{game_title}\n{TRANS_NAMES[tr]}",pad=.8,fontsize=4.7)
    cb=fig.colorbar(im,ax=axes,fraction=.015,pad=.018)
    cb.ax.tick_params(labelsize=5.4,length=2)
    cb.ax.set_title(r"$\beta$",fontsize=5.8,pad=1)
    return axes



def supp_figure_2(path:Path)->None:
    # A/B/C form the left block; D/E are stacked in the right block.  The two
    # middle columns are reserved for the external A- and C-panel legends.
    fig = plt.figure(figsize=(WIDTH, 6.35))
    fig.subplots_adjust(left=.078,right=.940,bottom=.072,top=.955,
                        hspace=.64,wspace=.60)
    gs = fig.add_gridspec(28,24)
    aaxes = learning_curves(gs[:18,:14],fig)
    b = fig.add_subplot(gs[19:23,:14]); draw_actor_margin(b)
    c = fig.add_subplot(gs[24:28,:14]); draw_q_mechanics(c); c.set_ylabel("")
    c.set_xticklabels(["prediction-error\nscale","prior visits","depth","event position"],
                      rotation=0,ha="center",fontsize=4.2)
    # B and C encode the same game identity; keep one shared legend outside C.
    handles,labels=c.get_legend_handles_labels()
    if b.get_legend() is not None: b.get_legend().remove()
    if c.get_legend() is not None: c.get_legend().remove()
    c.legend(handles,labels,frameon=False,loc="center left",bbox_to_anchor=(1.02,.50),
             fontsize=4.2,handletextpad=.25,borderaxespad=.05)

    daxes = draw_presearch_m(gs[:13,16:24],fig)
    for ax in daxes:
        ax.set_title(ax.get_title(),fontsize=5.0,pad=1)
        ax.tick_params(axis="both",labelsize=4.0,pad=.6)
        ax.set_xlabel("pre-search $M$ (z)",fontsize=4.4,labelpad=.7)
    daxes[0].set_ylabel("transition rate",fontsize=4.4,labelpad=1)
    if daxes[1].get_legend() is not None:
        daxes[1].legend(frameon=False,fontsize=3.55,loc="best",
                        handlelength=.9,handletextpad=.2,borderaxespad=.1)
    eaxes = draw_transition_coefficients(gs[15:28,16:24],fig)
    for ax in eaxes:
        ax.tick_params(axis="both",labelsize=3.8,pad=.5,length=1.3)

    # Position letters from finalized axis bounds, not from title coordinates.
    fig.canvas.draw()
    for anchor, letter in ((aaxes[0], "A"), (b, "B"), (c, "C"),
                           (daxes[0], "D"), (eaxes[0], "E")):
        box = anchor.get_position()
        if letter=="A":
            x=box.x0; y=.975
        elif letter in ("D","E"):
            x=box.x0; y=min(.985,box.y1+.018)
        else:
            x=box.x0; y=min(.985,box.y1+.035)
        fig.text(x,y,letter,
                 fontsize=10,fontweight="bold",ha="left",va="top")
    save(fig, path)


FILES = {
    "main_figure0_framework.pdf": main_figure_0,
    "main_figure1_behavior.pdf": main_figure_1,
    "main_figure2_model.pdf": main_figure_2,
    "main_figure3_neural.pdf": main_figure_3,
    "supp_figure1_behavior.pdf": supp_figure_1,
    "supp_figure2_model_validation.pdf": supp_figure_2,
    "supp_figure3_neural.pdf": supp_figure_3,
}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=OUT/"00_neureps2026_figure")
    parser.add_argument("--validate-only",action="store_true")
    args=parser.parse_args()
    setup_style()
    # Cheap contract checks precede any rendering.
    validate_framework_contract()
    read(OUT/"00_learningcurve"/"results"/"subject_action_distribution_counts.csv")
    read(BEH/"1-2_noop_day_level.csv")
    for game in (1,2):
        rsa_root=OUT/"07_RSA"/f"group_game{game}"
        read(rsa_root/"planning_trace"/"results"/"predefined_roi_subject_effects.csv")
        read(rsa_root/"planning_trace"/"results"/"predefined_roi_inference.csv")
        read(rsa_root/"results"/"predefined_roi_subject_effects.csv")
        read(rsa_root/"results"/"predefined_roi_inference.csv")
    if len(FILES) != 7:
        raise RuntimeError(f"Expected seven registered figures; found {len(FILES)}")
    if args.validate_only:
        print("Frozen artifact contract validated.")
        return
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for name,fn in FILES.items():
        fn(args.output_dir/name)
        print(args.output_dir/name)
    expected=set(FILES)|{Path(name).with_suffix(".png").name for name in FILES}
    produced={p.name for p in args.output_dir.iterdir() if p.name in expected}
    if produced!=expected:
        raise RuntimeError(f"Incomplete PDF/PNG set: {sorted(expected-produced)}")


if __name__=="__main__":
    main()
