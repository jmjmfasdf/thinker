#!/usr/bin/env python3
"""Evaluation--revision geometry for continuous action control.

This analysis separates three temporally ordered quantities that were mixed in
earlier actor-only geometries:

1. Pre-search action margin (M)
   ``log p_model(current) - max_other log p_model(other)``.
2. Q revision magnitude (Q_rev)
   the mean absolute sampled-branch root-Q update across valid status-2 events.
3. Cross-head Actor current-action margin contrast
   terminal real-head minus initial imaginary-head margin, retained as a
   diagnostic readout rather than a within-head policy update.

The allocation analyses deliberately use a different revision summary:
signed Q revision (Q_signed), the arithmetic mean sampled-branch root-Q update
across valid status-2 events.  It asks whether net upward versus downward value
revision covaries with root-branch diversity, mean terminal depth, and the
fraction of deep expansion events under the observed search budget. A separate
completed-bout analysis asks whether longer, known-onset action bouts are
associated with a deeper and narrower allocation profile after controlling
start Q_signed, margin, current/origin action, destination action, and prior
reward.  The window-level allocation analysis uses every eligible real step
and is not conditioned on a future action transition; the duration
endpoint-change analysis is explicitly restricted to completed, known-onset
transition bouts.

The first two quantities define a two-dimensional evaluation--revision plane.
Actor-margin shift and the subsequent human stay/switch are outcomes overlaid
on that plane; they are not treated as a third pre-search state coordinate.

Only shared caches produced by ``00_cache.py`` are read.  Raw traces, PCA, and
t-SNE are never recomputed.  Outputs are PNG and CSV/JSON only.

Examples
--------
python research_script/02_evaluation_revision_geometry.py --subject 3 --game 1
python research_script/02_evaluation_revision_geometry.py --game 1 \
    --aggregate-subjects 1 2 3 4 5 6
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.colors import TwoSlopeNorm
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.lines import Line2D
from scipy import stats
from scipy.ndimage import gaussian_filter

from shared_analysis_cache import (
    DEFAULT_SHARED_CACHE_ROOT,
    cache_paths,
    validate_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs" / "02_evaluation_revision_geometry"
KEYS = ["run_label", "real_step", "real_step_id"]
EPS = 1e-12
SCHEMA_VERSION = 9
FIGURE6_RENDER_VERSION = 2
FIGURE7_RENDER_VERSION = 1
N_ACTIONS = 6

PRE_SEARCH_MARGIN_LABEL = "Pre-search action margin (M)"
Q_REVISION_LABEL = "Q revision magnitude (Q_rev)"
SIGNED_Q_REVISION_LABEL = "Signed Q revision (Q_signed)"
TRANSITION_MARGIN_LABEL = "Transition margin"

ALLOCATION_METRICS: Tuple[Tuple[str, str, str], ...] = (
    ("root_branch_allocation_diversity", "Root-branch diversity", "Breadth across root actions"),
    ("mean_bout_terminal_depth", "Mean terminal depth", "Depth reached within search bouts"),
    ("deep_event_fraction", "Deep-event fraction", "Fraction of expansions at depth >= 3"),
)

INK = "#404040"
GRID = "#D9DDE3"
BLUE = "#3E6F9E"
BLUE_LIGHT = "#C9D9E8"
ORANGE = "#D7833B"
ORANGE_LIGHT = "#F1D2B7"
GOLD = "#C89B2C"
OPEN = "#F7F8FA"
MISSING = "#ECEFF2"
SIGNED_CMAP = "coolwarm"  # negative=blue, positive=red


def configure_style() -> None:
    """Apply the project's restrained static-figure style."""
    font_path = ROOT / "ARIAL.TTF"
    if font_path.exists():
        try:
            fontManager.addfont(str(font_path))
            family = FontProperties(fname=str(font_path)).get_name()
        except Exception:
            family = "DejaVu Sans"
    else:
        family = "DejaVu Sans"
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "grid.alpha": 0.55,
            "legend.frameon": False,
        }
    )


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def parse_subject(value: str) -> int:
    text = str(value).strip().lower().replace("sub", "")
    subject = int(text)
    if subject <= 0:
        raise argparse.ArgumentTypeError("subject must be positive")
    return subject


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, default=str), encoding="utf-8"
    )
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    """Return a stable content digest for a rendered artifact or input."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log(f"Saved {path}")


def bh_fdr(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    if finite.size == 0:
        return q
    order = finite[np.argsort(p[finite])]
    ranked = p[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[order] = np.clip(ranked, 0.0, 1.0)
    return q


def exact_sign_flip(values: Sequence[float]) -> Tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    observed = float(np.mean(x))
    null = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(x)):
        null.append(float(np.mean(x * np.asarray(signs))))
    null_array = np.asarray(null)
    p = float(np.mean(np.abs(null_array) >= abs(observed) - 1e-15))
    return observed, p


def mean_ci95(values: Sequence[float]) -> Tuple[float, float, float]:
    """Return the mean and a two-sided 95% t interval across independent units."""
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(data))
    if data.size < 2:
        return mean, np.nan, np.nan
    sem = float(stats.sem(data))
    critical = float(stats.t.ppf(0.975, data.size - 1))
    return mean, mean - critical * sem, mean + critical * sem


def significance_stars(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "n/a"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def draw_significance_bracket(
    axis: plt.Axes,
    left: float,
    right: float,
    height: float,
    label: str,
    *,
    span: float,
) -> None:
    tick = max(0.035 * span, EPS)
    axis.plot(
        [left, left, right, right],
        [height - tick, height, height, height - tick],
        color=INK,
        linewidth=0.9,
        clip_on=False,
    )
    axis.text(
        (left + right) / 2,
        height + 0.012 * span,
        label,
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold" if label != "ns" else "normal",
        clip_on=False,
    )


def draw_continuous_surface(
    axis: plt.Axes,
    shown: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    *,
    cmap: str,
    norm: Optional[TwoSlopeNorm] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    """Render bin estimates as a continuous-looking gradient.

    The statistic remains the prespecified bin mean/count. Gouraud shading
    only interpolates the display between supported bin centres; it does not
    add observations or change the estimator.
    """
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    image = axis.pcolormesh(
        x_centers,
        y_centers,
        np.ma.masked_invalid(shown),
        shading="gouraud",
        cmap=cmap,
        vmin=None if norm is not None else vmin,
        vmax=None if norm is not None else vmax,
        norm=norm,
        rasterized=True,
    )
    axis.set_xlim(float(x_edges[0]), float(x_edges[-1]))
    axis.set_ylim(float(y_edges[0]), float(y_edges[-1]))
    return image


def robust_z(values: pd.Series) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce").astype(float)
    finite = np.isfinite(x)
    result = pd.Series(np.nan, index=x.index, dtype=float)
    if finite.sum() < 2:
        return result
    center = float(np.nanmedian(x[finite]))
    mad = float(np.nanmedian(np.abs(x[finite] - center)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= EPS:
        scale = float(np.nanstd(x[finite], ddof=1))
    if not np.isfinite(scale) or scale <= EPS:
        return result
    result.loc[finite] = (x.loc[finite] - center) / scale
    return result


def zscore(values: pd.Series) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce").astype(float)
    result = pd.Series(np.nan, index=x.index, dtype=float)
    finite = np.isfinite(x)
    if finite.sum() < 2:
        return result
    mean = float(x.loc[finite].mean())
    sd = float(x.loc[finite].std(ddof=0))
    if not np.isfinite(sd) or sd <= EPS:
        return result
    result.loc[finite] = (x.loc[finite] - mean) / sd
    return result


def nuisance_zscore(values: pd.Series) -> pd.Series:
    """Z-score a nuisance variable, mapping a finite constant to zero."""
    x = pd.to_numeric(values, errors="coerce").astype(float)
    result = pd.Series(np.nan, index=x.index, dtype=float)
    finite = np.isfinite(x)
    if finite.sum() == 0:
        return result
    mean = float(x.loc[finite].mean())
    sd = float(x.loc[finite].std(ddof=0))
    if not np.isfinite(sd) or sd <= EPS:
        result.loc[finite] = 0.0
    else:
        result.loc[finite] = (x.loc[finite] - mean) / sd
    return result


def within_group_z(
    frame: pd.DataFrame, column: str, groups: Sequence[str]
) -> pd.Series:
    return frame.groupby(list(groups), sort=False, observed=True)[column].transform(
        zscore
    )


def within_group_demean(
    frame: pd.DataFrame, column: str, groups: Sequence[str]
) -> pd.Series:
    x = pd.to_numeric(frame[column], errors="coerce")
    return x - x.groupby([frame[name] for name in groups], sort=False).transform(
        "mean"
    )


def read_csv_columns(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    header = pd.read_csv(path, nrows=0).columns
    missing = sorted(set(columns).difference(header))
    if missing:
        raise KeyError(f"{path} lacks required columns: {missing}")
    return pd.read_csv(path, usecols=list(columns), low_memory=False)


def aggregate_revision_events(
    event_path: Path,
    *,
    chunksize: int = 750_000,
) -> pd.DataFrame:
    """Aggregate status-2 Q updates without loading the full event table."""
    usecols = KEYS + [
        "root_q_revision",
        "directed_value_revision",
        "value_revision_valid",
    ]
    pieces: List[pd.DataFrame] = []
    for chunk_index, chunk in enumerate(
        pd.read_csv(
            event_path,
            usecols=usecols,
            chunksize=chunksize,
            low_memory=False,
        )
    ):
        revision = pd.to_numeric(chunk["root_q_revision"], errors="coerce")
        directed = pd.to_numeric(
            chunk["directed_value_revision"], errors="coerce"
        )
        valid = (
            pd.to_numeric(chunk["value_revision_valid"], errors="coerce").eq(1)
            & np.isfinite(revision)
            & np.isfinite(directed)
        )
        selected = chunk.loc[valid, KEYS].copy()
        selected["revision_abs"] = revision.loc[valid].abs().to_numpy()
        selected["revision_signed"] = revision.loc[valid].to_numpy()
        selected["revision_directed"] = directed.loc[valid].to_numpy()
        grouped = (
            selected.groupby(KEYS, as_index=False, sort=False)
            .agg(
                revision_abs_sum=("revision_abs", "sum"),
                revision_signed_sum=("revision_signed", "sum"),
                revision_event_count=("revision_abs", "size"),
                revision_directed_sum=("revision_directed", "sum"),
            )
        )
        pieces.append(grouped)
        if chunk_index and chunk_index % 4 == 0:
            log(f"  revision cache chunks read: {chunk_index + 1}")
    if not pieces:
        raise RuntimeError(f"No readable Q-revision events in {event_path}")
    combined = pd.concat(pieces, ignore_index=True)
    summary = (
        combined.groupby(KEYS, as_index=False, sort=False)
        .agg(
            revision_abs_sum=("revision_abs_sum", "sum"),
            revision_signed_sum=("revision_signed_sum", "sum"),
            revision_event_count=("revision_event_count", "sum"),
            revision_directed_sum=("revision_directed_sum", "sum"),
        )
    )
    summary["value_revision_magnitude"] = (
        summary["revision_abs_sum"]
        / summary["revision_event_count"].clip(lower=1)
    )
    summary["value_revision_signed_mean"] = (
        summary["revision_signed_sum"]
        / summary["revision_event_count"].clip(lower=1)
    )
    summary["directional_revision_balance"] = (
        summary["revision_directed_sum"]
        / (summary["revision_abs_sum"] + EPS)
    )
    return summary


def load_subject_metrics(
    subject: int,
    game: int,
    cache_root: Path,
    output_root: Path,
    *,
    force_revision_summary: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    global N_ACTIONS
    paths = cache_paths(cache_root, subject, game)
    manifest = validate_manifest(paths, subject=subject, game=game)
    N_ACTIONS = int(manifest.get("num_actions", manifest.get("n_actions", 6)))
    model_columns = KEYS + [
        "subject",
        "session",
        "block",
        "game",
        "incumbent_action",
        "human_action",
        "switch",
        "bout_age",
        "prior_reward",
        "prior_done",
        "run_progress",
        "strongest_competitor_margin",
        "switch_target_uncertainty",
        "n_status2",
    ] + [f"model_root_p_{action}" for action in range(N_ACTIONS)]
    actor_columns = KEYS + [
        "actor_policy_source_real_step",
        "actor_current_action_margin_start",
        "actor_current_action_margin_terminal",
        "actor_current_action_uncertainty_start",
        "actor_current_action_uncertainty_terminal",
        "actor_current_action_coordinate_valid",
    ]
    model = read_csv_columns(paths.model_control_windows, model_columns)
    actor = read_csv_columns(paths.actor_metrics, actor_columns)
    allocation_columns = KEYS + [metric for metric, _, _ in ALLOCATION_METRICS]
    allocation = read_csv_columns(paths.persistence_windows, allocation_columns)
    if (
        model.duplicated(KEYS).any()
        or actor.duplicated(KEYS).any()
        or allocation.duplicated(KEYS).any()
    ):
        raise RuntimeError("Shared real-step cache keys are not one-to-one")

    derivative_dir = output_root / f"sub{subject:03d}_game{game}" / "results"
    derivative_path = derivative_dir / "revision_window_summary.csv.gz"
    if derivative_path.exists() and not force_revision_summary:
        revision = pd.read_csv(derivative_path, low_memory=False)
        required_revision_columns = KEYS + [
            "value_revision_magnitude",
            "value_revision_signed_mean",
        ]
        if not set(required_revision_columns).issubset(revision.columns):
            revision = aggregate_revision_events(paths.persistence_events)
    else:
        revision = aggregate_revision_events(paths.persistence_events)
    derivative_dir.mkdir(parents=True, exist_ok=True)
    revision.to_csv(derivative_path, index=False, compression="gzip")

    merged = model.merge(actor, on=KEYS, how="left", validate="one_to_one")
    merged = merged.merge(allocation, on=KEYS, how="left", validate="one_to_one")
    merged = merged.merge(revision, on=KEYS, how="left", validate="one_to_one")
    merged = merged.rename(
        columns={
            "strongest_competitor_margin": "model_current_action_margin",
            "switch_target_uncertainty": "model_alternative_uncertainty",
        }
    )
    merged["actor_current_action_margin_shift"] = (
        pd.to_numeric(
            merged["actor_current_action_margin_terminal"], errors="coerce"
        )
        - pd.to_numeric(
            merged["actor_current_action_margin_start"], errors="coerce"
        )
    )
    model_probabilities = merged[
        [f"model_root_p_{action}" for action in range(N_ACTIONS)]
    ].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    incumbent = pd.to_numeric(
        merged["incumbent_action"], errors="coerce"
    ).to_numpy(dtype=float)
    valid_incumbent = (
        np.isfinite(incumbent)
        & (incumbent >= 0)
        & (incumbent < N_ACTIONS)
    )
    current_probability = np.full(len(merged), np.nan, dtype=float)
    row_index = np.arange(len(merged))
    incumbent_integer = np.full(len(merged), -1, dtype=int)
    incumbent_integer[valid_incumbent] = incumbent[valid_incumbent].astype(int)
    current_probability[valid_incumbent] = model_probabilities[
        row_index[valid_incumbent], incumbent_integer[valid_incumbent]
    ]
    noop_probability = model_probabilities[:, 0]
    merged["model_current_vs_noop_margin"] = (
        np.log(np.clip(current_probability, EPS, None))
        - np.log(np.clip(noop_probability, EPS, None))
    )
    # Q revisions are strictly non-negative and span multiplicative scales.
    # A natural-log transform is necessary because revisions are mostly much
    # smaller than one; log1p would be effectively linear in this range.
    merged["value_revision_log"] = np.log(
        pd.to_numeric(merged["value_revision_magnitude"], errors="coerce").clip(
            lower=EPS
        )
    )
    merged["value_revision_robust_z"] = merged.groupby(
        "run_label", sort=False
    )["value_revision_log"].transform(robust_z)
    merged["value_revision_signed_robust_z"] = merged.groupby(
        "run_label", sort=False
    )["value_revision_signed_mean"].transform(robust_z)
    merged["revision_event_fraction"] = (
        pd.to_numeric(merged["revision_event_count"], errors="coerce")
        / pd.to_numeric(merged["n_status2"], errors="coerce").clip(lower=1)
    )
    merged["previous_human_action"] = pd.to_numeric(
        merged["incumbent_action"], errors="coerce"
    )
    merged["previous_reward"] = pd.to_numeric(
        merged["prior_reward"], errors="coerce"
    )
    merged["previous_done"] = pd.to_numeric(
        merged["prior_done"], errors="coerce"
    )
    merged["transition_class"] = np.select(
        [
            merged["incumbent_action"].eq(0) & merged["human_action"].ne(0),
            merged["incumbent_action"].ne(0) & merged["human_action"].eq(0),
            merged["incumbent_action"].ne(0)
            & merged["human_action"].ne(0)
            & merged["incumbent_action"].ne(merged["human_action"]),
        ],
        ["noop_to_active", "active_to_noop", "active_to_active"],
        default="stay",
    )
    finite = merged[
        [
            "model_current_action_margin",
            "value_revision_robust_z",
            "actor_current_action_margin_shift",
        ]
    ].replace([np.inf, -np.inf], np.nan)
    qa = {
        "subject": int(subject),
        "game": int(game),
        "cache_manifest": str(paths.manifest),
        "cache_components": manifest.get("components", {}),
        "n_windows": int(len(merged)),
        "n_runs": int(merged["run_label"].nunique()),
        "model_margin_coverage": float(finite.iloc[:, 0].notna().mean()),
        "revision_window_coverage": float(finite.iloc[:, 1].notna().mean()),
        "actor_shift_coverage": float(finite.iloc[:, 2].notna().mean()),
        "mean_valid_revision_events": float(
            pd.to_numeric(merged["revision_event_count"], errors="coerce").mean()
        ),
        "mean_revision_event_fraction": float(
            pd.to_numeric(merged["revision_event_fraction"], errors="coerce").mean()
        ),
        "actor_source_is_next_step_fraction": float(
            (
                pd.to_numeric(
                    merged["actor_policy_source_real_step"], errors="coerce"
                )
                == pd.to_numeric(merged["real_step"], errors="coerce") + 1
            ).mean()
        ),
        "allocation_coverage": {
            metric: float(pd.to_numeric(merged[metric], errors="coerce").notna().mean())
            for metric, _, _ in ALLOCATION_METRICS
        },
    }
    return merged, qa


def action_class(frame: pd.DataFrame, kind: str) -> pd.DataFrame:
    if kind == "noop":
        return frame.loc[frame["incumbent_action"].eq(0)].copy()
    if kind == "active":
        return frame.loc[frame["incumbent_action"].ne(0)].copy()
    return frame.copy()


def shared_limits(frames: Sequence[pd.DataFrame]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    margin = pd.concat(
        [pd.to_numeric(frame["model_current_action_margin"], errors="coerce") for frame in frames],
        ignore_index=True,
    )
    revision = pd.concat(
        [pd.to_numeric(frame["value_revision_robust_z"], errors="coerce") for frame in frames],
        ignore_index=True,
    )
    margin = margin[np.isfinite(margin)]
    revision = revision[np.isfinite(revision)]
    if margin.empty or revision.empty:
        raise RuntimeError("No finite geometry coordinates")
    x_low, x_high = np.quantile(margin, [0.01, 0.99])
    y_low, y_high = np.quantile(revision, [0.01, 0.99])
    x_pad = 0.04 * max(x_high - x_low, EPS)
    y_pad = 0.04 * max(y_high - y_low, EPS)
    return (float(x_low - x_pad), float(x_high + x_pad)), (
        float(y_low - y_pad),
        float(y_high + y_pad),
    )


def binned_surface(
    frame: pd.DataFrame,
    value: Optional[str],
    x_limits: Tuple[float, float],
    y_limits: Tuple[float, float],
    *,
    bins: int = 34,
    min_count: int = 20,
    equalize_action: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    columns = ["model_current_action_margin", "value_revision_robust_z"]
    if value is not None:
        columns.append(value)
    data = frame[columns + (["incumbent_action"] if equalize_action else [])].copy()
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=columns)
    x_edges = np.linspace(*x_limits, bins + 1)
    y_edges = np.linspace(*y_limits, bins + 1)
    x = data["model_current_action_margin"].to_numpy(dtype=float)
    y = data["value_revision_robust_z"].to_numpy(dtype=float)
    count = np.histogram2d(x, y, bins=[x_edges, y_edges])[0].T
    if value is None:
        return count, count, x_edges, y_edges

    if equalize_action:
        action_surfaces = []
        for _, group in data.groupby("incumbent_action", sort=False):
            gx = group["model_current_action_margin"].to_numpy(dtype=float)
            gy = group["value_revision_robust_z"].to_numpy(dtype=float)
            gv = group[value].to_numpy(dtype=float)
            gc = np.histogram2d(gx, gy, bins=[x_edges, y_edges])[0].T
            gs = np.histogram2d(gx, gy, bins=[x_edges, y_edges], weights=gv)[0].T
            surface = np.full_like(gs, np.nan, dtype=float)
            np.divide(gs, gc, out=surface, where=gc >= min_count)
            action_surfaces.append(surface)
        if action_surfaces:
            stack = np.stack(action_surfaces)
            support = np.sum(np.isfinite(stack), axis=0)
            summed = np.nansum(stack, axis=0)
            surface = np.full_like(summed, np.nan, dtype=float)
            np.divide(summed, support, out=surface, where=support > 0)
        else:
            surface = np.full_like(count, np.nan)
    else:
        values = data[value].to_numpy(dtype=float)
        weighted = np.histogram2d(
            x, y, bins=[x_edges, y_edges], weights=values
        )[0].T
        surface = np.full_like(weighted, np.nan, dtype=float)
        np.divide(weighted, count, out=surface, where=count >= min_count)
    return surface, count, x_edges, y_edges


def draw_surface(
    axis: plt.Axes,
    frame: pd.DataFrame,
    value: Optional[str],
    title: str,
    x_limits: Tuple[float, float],
    y_limits: Tuple[float, float],
    *,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    center: Optional[float] = None,
    equalize_action: bool = False,
    colorbar_label: str = "",
) -> None:
    surface, count, x_edges, y_edges = binned_surface(
        frame,
        value,
        x_limits,
        y_limits,
        equalize_action=equalize_action,
    )
    if value is None:
        shown = np.log10(count + 1.0)
        normalization = None
    else:
        shown = surface
        if center is not None and np.isfinite(shown).any():
            bound = max(
                abs(float(np.nanquantile(shown, 0.03))),
                abs(float(np.nanquantile(shown, 0.97))),
                EPS,
            )
            vmin = -bound if vmin is None else vmin
            vmax = bound if vmax is None else vmax
            normalization = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
        else:
            normalization = None
    image = draw_continuous_surface(
        axis,
        shown,
        x_edges,
        y_edges,
        cmap=cmap,
        norm=normalization,
        vmin=None if normalization is not None else vmin,
        vmax=None if normalization is not None else vmax,
    )
    axis.axvline(0.0, color=INK, linestyle="--", linewidth=0.9, alpha=0.75)
    axis.axhline(0.0, color=INK, linestyle=":", linewidth=0.8, alpha=0.55)
    axis.set_xlim(x_limits)
    axis.set_ylim(y_limits)
    axis.set_title(title)
    axis.set_xlabel(PRE_SEARCH_MARGIN_LABEL)
    axis.set_ylabel(Q_REVISION_LABEL)
    axis.set_box_aspect(1)
    cbar = axis.figure.colorbar(image, ax=axis, fraction=0.047, pad=0.035)
    cbar.set_label(colorbar_label)


def draw_temporal_schematic(axis: plt.Axes) -> None:
    axis.set_axis_off()
    boxes = [
        (0.02, 0.60, 0.25, 0.22, "Model evaluation\nbefore search", BLUE_LIGHT),
        (0.375, 0.60, 0.25, 0.22, "Root-Q revision\nduring search", ORANGE_LIGHT),
        (0.73, 0.60, 0.25, 0.22, "Actor readout\nat search end", OPEN),
        (0.73, 0.15, 0.25, 0.18, "Human stay / switch", OPEN),
    ]
    for x, y, width, height, label, color in boxes:
        rectangle = plt.Rectangle(
            (x, y), width, height, facecolor=color, edgecolor=INK, linewidth=1.1
        )
        axis.add_patch(rectangle)
        axis.text(x + width / 2, y + height / 2, label, ha="center", va="center")
    arrow = dict(arrowstyle="->", color=INK, linewidth=1.25, shrinkA=2, shrinkB=2)
    axis.annotate("", xy=(0.375, 0.71), xytext=(0.27, 0.71), arrowprops=arrow)
    axis.annotate("", xy=(0.73, 0.71), xytext=(0.625, 0.71), arrowprops=arrow)
    axis.annotate("", xy=(0.855, 0.33), xytext=(0.855, 0.60), arrowprops=arrow)
    axis.text(0.145, 0.50, "current-action margin", ha="center", va="center", color=BLUE)
    axis.text(0.50, 0.50, "mean absolute ΔQ", ha="center", va="center", color=ORANGE)
    axis.text(0.855, 0.50, "terminal − initial margin", ha="center", va="center")
    axis.text(
        0.50,
        0.03,
        "Arrows indicate temporal order, not an identified causal effect.",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#636B74",
    )


def plot_subject_geometry(
    frame: pd.DataFrame, output_path: Path, subject: int, game: int
) -> None:
    x_limits, y_limits = shared_limits([frame])
    noop = action_class(frame, "noop")
    active = action_class(frame, "active")
    fig, axes = plt.subplots(2, 3, figsize=(17.2, 10.8), facecolor="white")
    draw_surface(
        axes[0, 0],
        frame,
        None,
        "M–Q_rev occupancy",
        x_limits,
        y_limits,
        colorbar_label="log10(window count + 1)",
    )
    draw_surface(
        axes[0, 1],
        noop,
        "switch",
        "Switch probability · current NOOP",
        x_limits,
        y_limits,
        cmap="magma",
        vmin=0.0,
        vmax=min(0.65, max(0.15, float(noop["switch"].mean()) * 3.5)),
        colorbar_label="P(switch)",
    )
    draw_surface(
        axes[0, 2],
        active,
        "switch",
        "Switch probability · current active action",
        x_limits,
        y_limits,
        cmap="magma",
        vmin=0.0,
        vmax=min(0.65, max(0.15, float(active["switch"].mean()) * 3.5)),
        equalize_action=True,
        colorbar_label="Action-equalized P(switch)",
    )
    draw_surface(
        axes[1, 0],
        noop,
        "actor_current_action_margin_shift",
        "Actor margin shift · current NOOP",
        x_limits,
        y_limits,
        cmap=SIGNED_CMAP,
        center=0.0,
        colorbar_label="Terminal − initial actor margin",
    )
    draw_surface(
        axes[1, 1],
        active,
        "actor_current_action_margin_shift",
        "Actor margin shift · current active action",
        x_limits,
        y_limits,
        cmap=SIGNED_CMAP,
        center=0.0,
        equalize_action=True,
        colorbar_label="Action-equalized actor margin shift",
    )
    fig.delaxes(axes[1, 2])
    fig.suptitle(
        f"M–Q_rev geometry · sub{subject:03d}, game {game}",
        fontsize=17,
        x=0.5,
        y=0.995,
        ha="center",
    )
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.07, top=0.93, hspace=0.32, wspace=0.30)
    save_figure(fig, output_path)


def build_same_action_bouts(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstruct complete same-action bouts using the canonical 02 rule."""
    data = frame.sort_values(["run_label", "real_step"], kind="stable").reset_index(drop=True)
    row_records: List[pd.DataFrame] = []
    bout_records: List[Dict[str, object]] = []
    bout_id = 0
    for run_label, run in data.groupby("run_label", sort=False):
        run = run.sort_values("real_step", kind="stable").reset_index(drop=True)
        previous = run["incumbent_action"].to_numpy(dtype=int)
        current = run["human_action"].to_numpy(dtype=int)
        steps = run["real_step"].to_numpy(dtype=int)
        index = 0
        while index < len(run):
            origin = int(previous[index])
            start = index
            end = index
            destination = -1
            completed = False
            while end < len(run):
                if int(previous[end]) != origin:
                    break
                if end > start and steps[end] != steps[end - 1] + 1:
                    break
                if int(current[end]) != origin:
                    destination = int(current[end])
                    completed = True
                    break
                nxt = end + 1
                if (
                    nxt >= len(run)
                    or steps[nxt] != steps[end] + 1
                    or int(previous[nxt]) != origin
                ):
                    break
                end = nxt
            if not completed:
                while (
                    end + 1 < len(run)
                    and steps[end + 1] == steps[end] + 1
                    and int(previous[end + 1]) == origin
                    and int(current[end]) == origin
                ):
                    end += 1
            onset_known = bool(
                start > 0
                and steps[start] == steps[start - 1] + 1
                and int(current[start - 1]) == origin
                and int(previous[start - 1]) != origin
            )
            pre_bout_action = int(previous[start - 1]) if onset_known else -1
            previous_reward = float(
                pd.to_numeric(
                    pd.Series([run.iloc[start]["prior_reward"]]),
                    errors="coerce",
                ).iloc[0]
            )
            segment = run.iloc[start : end + 1].copy()
            segment["action_bout_id"] = bout_id
            segment["root_position"] = np.arange(len(segment))
            segment["onset_known"] = int(onset_known)
            segment["completed"] = int(completed)
            segment["pre_bout_action"] = pre_bout_action
            segment["previous_reward_at_bout_start"] = previous_reward
            if completed:
                transition_class = (
                    "noop_to_active"
                    if origin == 0 and destination != 0
                    else "active_to_noop"
                    if origin != 0 and destination == 0
                    else "active_to_active"
                    if origin != 0 and destination != 0 and origin != destination
                    else "stay"
                )
            else:
                transition_class = "censored"
            segment["transition_class"] = transition_class
            row_records.append(segment)
            bout_records.append(
                {
                    "action_bout_id": bout_id,
                    "run_label": run_label,
                    "origin_action": origin,
                    "destination_action": destination,
                    "transition_class": transition_class,
                    "n_risk_rows": int(len(segment)),
                    "onset_known": int(onset_known),
                    "completed": int(completed),
                    "pre_bout_action": pre_bout_action,
                    "previous_reward_at_bout_start": previous_reward,
                }
            )
            bout_id += 1
            index = end + 1
    return pd.concat(row_records, ignore_index=True), pd.DataFrame(bout_records)


def bout_endpoints(rows: pd.DataFrame) -> pd.DataFrame:
    eligible = rows.loc[
        rows["completed"].eq(1)
        & rows["onset_known"].eq(1)
        & rows["transition_class"].isin(
            ["noop_to_active", "active_to_noop", "active_to_active"]
        )
    ].copy()
    records: List[Dict[str, object]] = []
    for bout_id, bout in eligible.groupby("action_bout_id", sort=False):
        bout = bout.sort_values("root_position", kind="stable")
        first = bout.iloc[0]
        terminal = bout.iloc[-1]
        if len(bout) < 2:
            continue
        record: Dict[str, object] = {
            "action_bout_id": int(bout_id),
            "run_label": str(first["run_label"]),
            "subject": int(first["subject"]),
            "game": int(first["game"]),
            "transition_class": str(first["transition_class"]),
            "origin_action": int(first["incumbent_action"]),
            "destination_action": int(terminal["human_action"]),
            "n_risk_rows": int(len(bout)),
        }
        for prefix, row in (("start", first), ("end", terminal)):
            record[f"{prefix}_margin"] = float(row["model_current_action_margin"])
            record[f"{prefix}_revision_z"] = float(row["value_revision_robust_z"])
            record[f"{prefix}_actor_shift"] = float(
                row["actor_current_action_margin_shift"]
            )
        records.append(record)
    return pd.DataFrame(records)


def contour_points(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    limits: Tuple[Tuple[float, float], Tuple[float, float]],
    color: str,
    linestyle: str,
) -> None:
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < 20:
        return
    x_edges = np.linspace(*limits[0], 42)
    y_edges = np.linspace(*limits[1], 42)
    density = np.histogram2d(x, y, bins=[x_edges, y_edges])[0].T
    density = gaussian_filter(density, sigma=1.3)
    positive = density[density > 0]
    if positive.size == 0:
        return
    levels = np.unique(np.quantile(positive, [0.65, 0.88]))
    if levels.size == 0:
        return
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    axis.contour(
        x_centers,
        y_centers,
        density,
        levels=levels,
        colors=color,
        linestyles=linestyle,
        linewidths=1.0,
        alpha=0.75,
    )


def plot_transition_trajectories(
    endpoints: pd.DataFrame,
    output_path: Path,
    frames_for_limits: Sequence[pd.DataFrame],
    title_suffix: str,
    *,
    include_active_to_active: bool = True,
) -> None:
    classes = [
        ("noop_to_active", "NOOP → active"),
        ("active_to_noop", "Active → NOOP"),
        ("active_to_active", "Active → different active"),
    ]
    if not include_active_to_active:
        classes = classes[:2]
    available = [item for item in classes if (endpoints["transition_class"] == item[0]).sum() >= 5]
    if not available:
        log("Transition trajectory figure skipped: no comparable bouts")
        return
    x_limits, y_limits = shared_limits(frames_for_limits)
    fig, axes = plt.subplots(1, len(available), figsize=(5.7 * len(available), 5.6), squeeze=False)
    axes_flat = axes.ravel()
    for axis, (class_name, label) in zip(axes_flat, available):
        data = endpoints.loc[endpoints["transition_class"].eq(class_name)].dropna(
            subset=["start_margin", "start_revision_z", "end_margin", "end_revision_z"]
        )
        contour_points(
            axis,
            data["start_margin"].to_numpy(),
            data["start_revision_z"].to_numpy(),
            (x_limits, y_limits),
            BLUE,
            "-",
        )
        contour_points(
            axis,
            data["end_margin"].to_numpy(),
            data["end_revision_z"].to_numpy(),
            (x_limits, y_limits),
            ORANGE,
            "--",
        )
        run_means = data.groupby("run_label", as_index=False).agg(
            start_margin=("start_margin", "mean"),
            start_revision_z=("start_revision_z", "mean"),
            end_margin=("end_margin", "mean"),
            end_revision_z=("end_revision_z", "mean"),
        )
        for _, row in run_means.iterrows():
            axis.annotate(
                "",
                xy=(row["end_margin"], row["end_revision_z"]),
                xytext=(row["start_margin"], row["start_revision_z"]),
                arrowprops=dict(arrowstyle="->", color="#A9AFB6", lw=0.75, alpha=0.65),
            )
        start = run_means[["start_margin", "start_revision_z"]].mean()
        end = run_means[["end_margin", "end_revision_z"]].mean()
        axis.annotate(
            "",
            xy=(end["end_margin"], end["end_revision_z"]),
            xytext=(start["start_margin"], start["start_revision_z"]),
            arrowprops=dict(arrowstyle="->", color=INK, lw=2.0),
        )
        axis.scatter(
            start["start_margin"],
            start["start_revision_z"],
            s=90,
            marker="o",
            facecolor=BLUE,
            edgecolor=INK,
            linewidth=0.8,
            zorder=5,
            label="First comparable window",
        )
        axis.scatter(
            end["end_margin"],
            end["end_revision_z"],
            s=150,
            marker="*",
            facecolor=ORANGE,
            edgecolor=INK,
            linewidth=0.8,
            zorder=5,
            label="Transition-producing window",
        )
        axis.axvline(0, color=INK, ls="--", lw=0.85, alpha=0.7)
        axis.axhline(0, color=INK, ls=":", lw=0.75, alpha=0.55)
        axis.set_xlim(x_limits)
        axis.set_ylim(y_limits)
        axis.set_box_aspect(1)
        axis.set_title(f"{label} · N={len(data):,} bouts")
        axis.set_xlabel(PRE_SEARCH_MARGIN_LABEL)
        axis.set_ylabel(Q_REVISION_LABEL)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=10.5)
    fig.suptitle(
        f"M–Q_rev trajectories across same-action bouts · {title_suffix}",
        fontsize=16,
        x=0.5,
        y=0.99,
    )
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.16, top=0.88, wspace=0.26)
    save_figure(fig, output_path)


def fit_run_effects(frame: pd.DataFrame) -> pd.DataFrame:
    """Estimate exploratory run-wise fixed-effect linear models.

    Predictors are standardized within run × incumbent-action strata.  The
    resulting coefficients are used for sign consistency, not naïve
    observation-level population inference.
    """
    results: List[Dict[str, object]] = []
    for run_label, run in frame.groupby("run_label", sort=False):
        run = run.copy()
        run["M"] = within_group_z(
            run, "model_current_action_margin", ["incumbent_action"]
        )
        run["V"] = within_group_z(run, "value_revision_log", ["incumbent_action"])
        run["MxV"] = run["M"] * run["V"]
        run["MxV"] = within_group_z(run, "MxV", ["incumbent_action"])
        run["log_bout_age"] = np.log1p(
            pd.to_numeric(run["bout_age"], errors="coerce").clip(lower=0)
        )
        for column in ["log_bout_age", "run_progress", "revision_event_fraction"]:
            run[column + "_z"] = zscore(run[column])
        predictor_columns = [
            "M",
            "V",
            "MxV",
            "log_bout_age_z",
            "prior_reward",
            "prior_done",
            "run_progress_z",
            "revision_event_fraction_z",
        ]
        needed = predictor_columns + ["switch", "actor_current_action_margin_shift", "incumbent_action"]
        clean = run[needed].replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean) < 100:
            continue
        # Removing incumbent-action means is equivalent to action fixed effects.
        design = np.column_stack(
            [
                within_group_demean(clean, column, ["incumbent_action"]).to_numpy()
                for column in predictor_columns
            ]
        )
        for outcome, outcome_name in [
            ("switch", "human_switch"),
            ("actor_current_action_margin_shift", "actor_margin_shift"),
        ]:
            y = within_group_demean(clean, outcome, ["incumbent_action"]).to_numpy(dtype=float)
            valid = np.isfinite(y) & np.all(np.isfinite(design), axis=1)
            if valid.sum() <= design.shape[1] + 5:
                continue
            beta, _, _, _ = np.linalg.lstsq(design[valid], y[valid], rcond=None)
            record: Dict[str, object] = {
                "run_label": str(run_label),
                "subject": int(run["subject"].iloc[0]),
                "game": int(run["game"].iloc[0]),
                "outcome": outcome_name,
                "n_rows": int(valid.sum()),
            }
            for name, value in zip(predictor_columns, beta):
                record[f"beta_{name}"] = float(value)
            results.append(record)
    return pd.DataFrame(results)


def plot_effect_tiles(
    effects: pd.DataFrame, output_path: Path, title_suffix: str
) -> None:
    outcomes = [
        ("human_switch", "Human switch"),
        ("actor_margin_shift", "Actor margin shift"),
    ]
    columns = ["beta_M", "beta_V", "beta_MxV"]
    labels = [
        "Pre-search action\nmargin (M)",
        "Q revision magnitude\n(Q_rev)",
        "M × Q_rev",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.5), facecolor="white")
    for axis, (outcome, title) in zip(axes, outcomes):
        data = effects.loc[effects["outcome"].eq(outcome)].copy()
        data = data.sort_values("run_label")
        matrix = data[columns].to_numpy(dtype=float)
        bound = (
            float(np.nanquantile(np.abs(matrix), 0.98))
            if np.isfinite(matrix).any()
            else 1.0
        )
        bound = max(bound, EPS)
        image = axis.imshow(
            matrix,
            aspect="auto",
            cmap=SIGNED_CMAP,
            norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound),
        )
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = matrix[row, col]
                if np.isfinite(value):
                    axis.text(col, row, f"{value:+.2f}", ha="center", va="center", fontsize=7.5)
        axis.set_xticks(range(len(labels)), labels, rotation=18, ha="right")
        compact_runs = [
            " · ".join(
                part for part in str(value).split("_")
                if part.startswith(("ses", "block"))
            )
            for value in data["run_label"]
        ]
        axis.set_title(title)
        axis.grid(False)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.035, label="Run-wise coefficient")
    fig.suptitle(
        f"Run-level effect consistency · {title_suffix}",
        fontsize=16,
        x=0.5,
        y=0.99,
    )
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.16, top=0.88, wspace=0.34)
    save_figure(fig, output_path)


def draw_violin_at(
    axis: plt.Axes,
    values: Sequence[float],
    position: float,
    color: str,
    *,
    width: float,
) -> None:
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if data.size >= 2 and float(np.std(data, ddof=0)) > EPS:
        parts = axis.violinplot(
            [data],
            positions=[position],
            widths=width,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            points=100,
        )
        body = parts["bodies"][0]
        body.set_facecolor(color)
        body.set_edgecolor(INK)
        body.set_linewidth(0.8)
        body.set_alpha(0.58)
    elif data.size:
        axis.plot(
            [position - width / 3, position + width / 3],
            [data[0], data[0]],
            color=color,
            linewidth=5,
            alpha=0.65,
        )
    if data.size:
        jitter = np.linspace(-0.055, 0.055, data.size)
        axis.scatter(
            position + jitter,
            data,
            s=12,
            color=INK,
            alpha=0.55,
            linewidths=0,
            zorder=3,
        )


def draw_mean_ci_at(
    axis: plt.Axes,
    values: Sequence[float],
    position: float,
    color: str,
) -> Tuple[float, float, float]:
    mean, low, high = mean_ci95(values)
    if np.isfinite(low) and np.isfinite(high):
        axis.errorbar(
            position,
            mean,
            yerr=np.array([[mean - low], [high - mean]]),
            fmt="o",
            markersize=5.5,
            color=INK,
            markerfacecolor=color,
            markeredgecolor=INK,
            capsize=3,
            linewidth=1.15,
            zorder=5,
        )
    elif np.isfinite(mean):
        axis.scatter(
            position,
            mean,
            s=38,
            facecolor=color,
            edgecolor=INK,
            zorder=5,
        )
    return mean, low, high


def plot_actor_shift_pairs(
    frame: pd.DataFrame, output_path: Path, title_suffix: str, unit: str = "run_label"
) -> None:
    group_mode = unit == "subject"
    figsize = (15.2, 6.2) if group_mode else (11.4, 5.8)
    fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor="white")

    for axis, (kind, title) in zip(
        axes, [("noop", "Current NOOP"), ("active", "Current active")]
    ):
        data = action_class(frame, kind)
        axis.axhline(0, color=INK, lw=0.8, ls="--", alpha=0.7)
        if group_mode:
            run_summary = (
                data.groupby(
                    ["subject", "run_label", "switch"], as_index=False
                )["actor_current_action_margin_shift"]
                .mean()
                .pivot(
                    index=["subject", "run_label"],
                    columns="switch",
                    values="actor_current_action_margin_shift",
                )
                .dropna(subset=[0, 1])
            )
            subjects = sorted(
                int(value)
                for value in run_summary.index.get_level_values("subject").unique()
            )
            finite_values = run_summary[[0, 1]].to_numpy(dtype=float)
            finite_values = finite_values[np.isfinite(finite_values)]
            span = max(
                float(np.ptp(finite_values)) if finite_values.size else 0.0,
                0.05,
            )
            bracket_tops: List[float] = []
            for index, subject in enumerate(subjects):
                subject_rows = run_summary.xs(subject, level="subject")
                stay = subject_rows[0].to_numpy(dtype=float)
                switch = subject_rows[1].to_numpy(dtype=float)
                left, right = index - 0.18, index + 0.18
                draw_violin_at(axis, stay, left, BLUE, width=0.30)
                draw_violin_at(axis, switch, right, ORANGE, width=0.30)
                _, stay_low, stay_high = draw_mean_ci_at(axis, stay, left, BLUE)
                _, switch_low, switch_high = draw_mean_ci_at(
                    axis, switch, right, ORANGE
                )
                _, p_value = exact_sign_flip(switch - stay)
                local_top = np.nanmax(
                    np.asarray(
                        [
                            np.nanmax(stay),
                            np.nanmax(switch),
                            stay_high,
                            switch_high,
                        ],
                        dtype=float,
                    )
                )
                bracket_top = float(local_top + 0.10 * span)
                bracket_tops.append(bracket_top)
                draw_significance_bracket(
                    axis,
                    left,
                    right,
                    bracket_top,
                    significance_stars(p_value),
                    span=span,
                )
            subject_means = run_summary.groupby(level="subject")[[0, 1]].mean()
            _, group_p = exact_sign_flip(
                subject_means[1].to_numpy(dtype=float)
                - subject_means[0].to_numpy(dtype=float)
            )
            axis.set_xticks(
                range(len(subjects)),
                [f"sub{subject:03d}" for subject in subjects],
                rotation=28,
                ha="right",
            )
            axis.set_title(
                f"{title} · subject-paired exact p={group_p:.4f} "
                f"({significance_stars(group_p)})"
            )
            if finite_values.size:
                lower = float(np.nanmin(finite_values) - 0.10 * span)
                upper = max(bracket_tops) + 0.13 * span
                axis.set_ylim(lower, upper)
        else:
            summary = (
                data.groupby([unit, "switch"], as_index=False)[
                    "actor_current_action_margin_shift"
                ]
                .mean()
                .pivot(
                    index=unit,
                    columns="switch",
                    values="actor_current_action_margin_shift",
                )
                .dropna(subset=[0, 1])
            )
            stay = summary[0].to_numpy(dtype=float)
            switch = summary[1].to_numpy(dtype=float)
            draw_violin_at(axis, stay, 0.0, BLUE, width=0.72)
            draw_violin_at(axis, switch, 1.0, ORANGE, width=0.72)
            _, stay_low, stay_high = draw_mean_ci_at(axis, stay, 0.0, BLUE)
            _, switch_low, switch_high = draw_mean_ci_at(
                axis, switch, 1.0, ORANGE
            )
            _, p_value = exact_sign_flip(switch - stay)
            finite_values = np.concatenate([stay, switch])
            span = max(float(np.ptp(finite_values)), 0.05)
            top = float(
                np.nanmax(
                    np.asarray(
                        [
                            np.nanmax(finite_values),
                            stay_high,
                            switch_high,
                        ],
                        dtype=float,
                    )
                )
                + 0.10 * span
            )
            draw_significance_bracket(
                axis,
                0.0,
                1.0,
                top,
                significance_stars(p_value),
                span=span,
            )
            axis.set_ylim(
                float(np.nanmin(finite_values) - 0.10 * span),
                top + 0.14 * span,
            )
            axis.set_xticks([0, 1], ["Stay", "Switch"])
            axis.set_title(
                f"{title} · {len(summary)} runs · paired exact p={p_value:.4f}"
            )
            axis.set_box_aspect(1)

        axis.set_ylabel(
            "Cross-head current-action margin contrast\n"
            "(real terminal − imaginary initial)"
        )
        axis.text(
            0.01,
            0.99,
            "dots=unit means · point/error bar=mean ± 95% CI",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            color="#636B74",
        )

    handles = [
        plt.Line2D(
            [0], [0], marker="s", color="none", markerfacecolor=BLUE,
            markeredgecolor=INK, markersize=8, label="Stay"
        ),
        plt.Line2D(
            [0], [0], marker="s", color="none", markerfacecolor=ORANGE,
            markeredgecolor=INK, markersize=8, label="Switch"
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2)
    fig.suptitle(
        f"Cross-head Actor readout contrast for stay and switch windows · {title_suffix}",
        fontsize=16,
        x=0.5,
        y=0.99,
    )
    fig.subplots_adjust(
        left=0.075 if group_mode else 0.10,
        right=0.985,
        bottom=0.18,
        top=0.85,
        wspace=0.25,
    )
    save_figure(fig, output_path)


TRANSITION_LABELS = {
    "noop_to_active": "NOOP → active",
    "active_to_noop": "Active → NOOP",
}


def build_transition_risk_sets(frame: pd.DataFrame) -> pd.DataFrame:
    """Create transition-specific recurrent risk windows without future selection.

    NOOP-to-active retains every current-NOOP window. Active-to-NOOP retains
    same-active continuation and NOOP-onset rows; direct active-to-active
    transitions are competing terminal events and are excluded from this
    binary conditional contrast.
    """
    ordered = frame.sort_values(["run_label", "real_step"], kind="stable").copy()
    active_source = pd.to_numeric(
        ordered["incumbent_action"], errors="coerce"
    ).where(pd.to_numeric(ordered["incumbent_action"], errors="coerce").ne(0))
    ordered["pre_bout_active_action"] = (
        active_source.groupby(ordered["run_label"], sort=False)
        .ffill()
        .fillna(-1)
        .astype(int)
    )

    noop = ordered.loc[ordered["incumbent_action"].eq(0)].copy()
    noop["transition_type"] = "noop_to_active"
    noop["transition_event"] = noop["human_action"].ne(0).astype(int)
    noop["transition_margin"] = pd.to_numeric(
        noop["model_current_action_margin"], errors="coerce"
    )
    noop["competing_event"] = 0

    active_all = ordered.loc[ordered["incumbent_action"].ne(0)].copy()
    active_all["competing_event"] = (
        active_all["human_action"].ne(0)
        & active_all["human_action"].ne(active_all["incumbent_action"])
    ).astype(int)
    active = active_all.loc[active_all["competing_event"].eq(0)].copy()
    active["transition_type"] = "active_to_noop"
    active["transition_event"] = active["human_action"].eq(0).astype(int)
    active["transition_margin"] = pd.to_numeric(
        active["model_current_vs_noop_margin"], errors="coerce"
    )

    retained = [
        "subject", "session", "block", "game", *KEYS,
        "incumbent_action", "human_action", "bout_age", "prior_reward",
        "run_progress", "revision_event_fraction", "value_revision_log",
        "value_revision_robust_z", "value_revision_magnitude",
        "actor_current_action_margin_shift", "pre_bout_active_action",
        "transition_type", "transition_event", "transition_margin",
        "competing_event",
    ]
    risk = pd.concat([noop[retained], active[retained]], ignore_index=True)
    required = ["transition_margin", "value_revision_log", "transition_event"]
    risk = risk.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    return risk


def _standardized_control(values: pd.Series) -> pd.Series:
    standardized = zscore(values)
    return standardized.fillna(0.0)


def fit_transition_run_effects(risk: pd.DataFrame) -> pd.DataFrame:
    """Fit one unweighted binomial GLM per run and transition risk set."""
    records: List[Dict[str, object]] = []
    for (transition, run_label), group in risk.groupby(
        ["transition_type", "run_label"], sort=False
    ):
        data = group.copy()
        if transition == "active_to_noop":
            data["M"] = data.groupby("incumbent_action", sort=False)[
                "transition_margin"
            ].transform(zscore)
            data["V"] = data.groupby("incumbent_action", sort=False)[
                "value_revision_log"
            ].transform(zscore)
            category = "incumbent_action"
        else:
            data["M"] = zscore(data["transition_margin"])
            data["V"] = zscore(data["value_revision_log"])
            category = "pre_bout_active_action"
        data["MxV"] = data["M"] * data["V"]
        log_age = np.log1p(pd.to_numeric(data["bout_age"], errors="coerce").clip(lower=0))
        progress = pd.to_numeric(data["run_progress"], errors="coerce")
        data["log_age_z"] = _standardized_control(log_age)
        data["log_age_sq_z"] = _standardized_control(log_age ** 2)
        data["reward_z"] = _standardized_control(
            pd.to_numeric(data["prior_reward"], errors="coerce")
        )
        data["progress_z"] = _standardized_control(progress)
        data["progress_sq_z"] = _standardized_control(progress ** 2)
        data["revision_coverage_z"] = _standardized_control(
            pd.to_numeric(data["revision_event_fraction"], errors="coerce")
        )
        predictor_names = [
            "M", "V", "MxV", "log_age_z", "log_age_sq_z", "reward_z",
            "progress_z", "progress_sq_z", "revision_coverage_z",
        ]
        clean = data.replace([np.inf, -np.inf], np.nan).dropna(
            subset=predictor_names + ["transition_event", category]
        )
        n_events = int(clean["transition_event"].sum())
        if len(clean) < 100 or n_events < 5 or n_events >= len(clean) - 5:
            continue
        design = clean[predictor_names].astype(float).copy()
        dummies = pd.get_dummies(
            clean[category].astype(int).astype(str),
            prefix="action",
            drop_first=True,
            dtype=float,
        )
        if not dummies.empty:
            design = pd.concat([design, dummies], axis=1)
        varying = design.std(axis=0, ddof=0).fillna(0).gt(EPS)
        design = design.loc[:, varying]
        design = sm.add_constant(design, has_constant="add")
        converged = False
        error = ""
        coefficients = {name: np.nan for name in ("M", "V", "MxV")}
        try:
            fitted = sm.GLM(
                clean["transition_event"].astype(float),
                design.astype(float),
                family=sm.families.Binomial(),
            ).fit(maxiter=100, disp=0)
            converged = bool(getattr(fitted, "converged", True))
            for name in coefficients:
                coefficients[name] = float(fitted.params.get(name, np.nan))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        records.append(
            {
                "subject": int(clean["subject"].iloc[0]),
                "game": int(clean["game"].iloc[0]),
                "run_label": str(run_label),
                "transition_type": str(transition),
                "n_risk_windows": int(len(clean)),
                "n_events": n_events,
                "event_rate": float(n_events / len(clean)),
                "beta_M": coefficients["M"],
                "beta_V": coefficients["V"],
                "beta_MxV": coefficients["MxV"],
                "converged": int(converged),
                "error": error,
            }
        )
    return pd.DataFrame(records)


def _transition_surface(
    risk: pd.DataFrame,
    x_limits: Tuple[float, float],
    y_limits: Tuple[float, float],
    *,
    bins: int = 34,
    min_count: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = risk[
        ["transition_margin", "value_revision_robust_z", "transition_event"]
    ].replace([np.inf, -np.inf], np.nan).dropna()
    x_edges = np.linspace(*x_limits, bins + 1)
    y_edges = np.linspace(*y_limits, bins + 1)
    x = data["transition_margin"].to_numpy(dtype=float)
    y = data["value_revision_robust_z"].to_numpy(dtype=float)
    event = data["transition_event"].to_numpy(dtype=float)
    count = np.histogram2d(x, y, bins=[x_edges, y_edges])[0].T
    event_sum = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=event)[0].T
    surface = np.full_like(event_sum, np.nan, dtype=float)
    np.divide(event_sum, count, out=surface, where=count >= min_count)
    return surface, count, x_edges, y_edges


def _draw_transition_surface(
    axis: plt.Axes,
    risk: pd.DataFrame,
    transition: str,
    x_limits: Tuple[float, float],
    y_limits: Tuple[float, float],
    *,
    subject_equalized_frames: Optional[Sequence[pd.DataFrame]] = None,
    show_contours: bool = True,
) -> None:
    if subject_equalized_frames is None:
        surface, count, x_edges, y_edges = _transition_surface(
            risk, x_limits, y_limits
        )
    else:
        surfaces: List[np.ndarray] = []
        counts: List[np.ndarray] = []
        for frame in subject_equalized_frames:
            selected = frame.loc[frame["transition_type"].eq(transition)]
            one_surface, one_count, x_edges, y_edges = _transition_surface(
                selected, x_limits, y_limits, min_count=12
            )
            surfaces.append(one_surface)
            counts.append(one_count)
        stack = np.stack(surfaces)
        support = np.sum(np.isfinite(stack), axis=0)
        summed = np.nansum(stack, axis=0)
        surface = np.full_like(summed, np.nan, dtype=float)
        np.divide(summed, support, out=surface, where=support >= 4)
        count = np.sum(np.stack(counts), axis=0)
    finite = surface[np.isfinite(surface)]
    vmax = float(np.nanquantile(finite, 0.97)) if finite.size else 1.0
    vmax = max(vmax, 0.02 if transition == "noop_to_active" else 0.10)
    image = draw_continuous_surface(
        axis,
        surface,
        x_edges,
        y_edges,
        cmap="magma",
        vmin=0.0,
        vmax=vmax,
    )
    if show_contours:
        smooth = gaussian_filter(count.astype(float), sigma=1.25)
        positive = smooth[smooth > 0]
        if positive.size:
            levels = np.unique(np.quantile(positive, [0.65, 0.88]))
            if levels.size:
                x_centers = (x_edges[:-1] + x_edges[1:]) / 2
                y_centers = (y_edges[:-1] + y_edges[1:]) / 2
                axis.contour(
                    x_centers,
                    y_centers,
                    smooth,
                    levels=levels,
                    colors=INK,
                    linewidths=0.8,
                    alpha=0.65,
                )
    n_events = int(risk["transition_event"].sum())
    probability_label = (
        "Conditional Active → NOOP"
        if transition == "active_to_noop"
        else "NOOP → active"
    )
    axis.axvline(0, color=INK, ls="--", lw=0.9)
    axis.axhline(0, color=INK, ls=":", lw=0.8, alpha=0.65)
    axis.set_xlim(x_limits)
    axis.set_ylim(y_limits)
    axis.set_box_aspect(1)
    axis.set_title(
        f"{probability_label} probability · "
        f"N={len(risk):,}, events={n_events:,}"
    )
    axis.set_xlabel(TRANSITION_MARGIN_LABEL)
    axis.set_ylabel(Q_REVISION_LABEL)
    axis.figure.colorbar(
        image,
        ax=axis,
        fraction=0.047,
        pad=0.035,
        label=f"P({probability_label})",
    )


def _draw_transition_effect_tile(
    axis: plt.Axes,
    effects: pd.DataFrame,
    transition: str,
    row_field: str,
    row_formatter,
    group_summary: Optional[pd.DataFrame] = None,
) -> None:
    selected = effects.loc[effects["transition_type"].eq(transition)].copy()
    selected = selected.sort_values(row_field)
    columns = ["beta_M", "beta_V", "beta_MxV"]
    labels = [
        "Transition margin",
        "Q revision magnitude\n(Q_rev)",
        "Transition margin\n× Q_rev",
    ]
    matrix = selected[columns].to_numpy(dtype=float)
    bound = (
        max(float(np.nanquantile(np.abs(matrix), 0.98)), EPS)
        if np.isfinite(matrix).any() else 1.0
    )
    image = axis.imshow(
        matrix, aspect="auto", cmap=SIGNED_CMAP,
        norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound),
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if np.isfinite(value):
                axis.text(column, row, f"{value:+.2f}", ha="center", va="center", fontsize=8.0)
    if group_summary is not None:
        decorated = []
        for metric, label in zip(columns, labels):
            row = group_summary.loc[
                group_summary["transition_type"].eq(transition)
                & group_summary["metric"].eq(metric)
            ]
            if row.empty:
                decorated.append(label)
            else:
                decorated.append(
                    f"{label}\np={row['exact_sign_flip_p'].iloc[0]:.3f}, "
                    f"q={row['bh_fdr_q'].iloc[0]:.3f}"
                )
        labels = decorated
    axis.set_xticks(range(3), labels, rotation=0, ha="center", fontsize=8.5)
    axis.set_yticks(
        range(len(selected)), [row_formatter(value) for value in selected[row_field]]
    )
    unit_label = "Run effects" if row_field == "run_label" else "Subject effects"
    axis.set_title(f"{unit_label} · {TRANSITION_LABELS[transition]}")
    axis.grid(False)
    axis.figure.colorbar(
        image, ax=axis, fraction=0.047, pad=0.035,
        label="Log-odds coefficient per 1 SD",
    )


def plot_transition_specific_subject(
    risk: pd.DataFrame,
    effects: pd.DataFrame,
    output_path: Path,
    subject: int,
    game: int,
) -> None:
    finite_margin = pd.to_numeric(risk["transition_margin"], errors="coerce")
    finite_revision = pd.to_numeric(risk["value_revision_robust_z"], errors="coerce")
    x_limits = tuple(np.quantile(finite_margin[np.isfinite(finite_margin)], [0.01, 0.99]))
    y_limits = tuple(np.quantile(finite_revision[np.isfinite(finite_revision)], [0.01, 0.99]))
    fig, axes = plt.subplots(
        2, 2, figsize=(14.4, 11.4), facecolor="white",
        constrained_layout=True,
    )
    for column, transition in enumerate(("noop_to_active", "active_to_noop")):
        selected = risk.loc[risk["transition_type"].eq(transition)]
        _draw_transition_surface(
            axes[0, column], selected, transition, x_limits, y_limits
        )
        _draw_transition_effect_tile(
            axes[1, column], effects, transition, "run_label",
            lambda value: " · ".join(
                part for part in str(value).split("_")
                if part.startswith(("ses", "block"))
            ),
        )
    fig.suptitle(
        f"Transition margin–Q_rev geometry · sub{subject:03d}, game {game}",
        fontsize=16.5, x=0.5, y=1.035,
    )
    save_figure(fig, output_path)


def summarize_transition_group_effects(
    effects: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    subject_effects = effects.groupby(
        ["subject", "transition_type"], as_index=False
    )[["beta_M", "beta_V", "beta_MxV"]].mean()
    records: List[Dict[str, object]] = []
    for transition, data in subject_effects.groupby("transition_type", sort=False):
        for metric in ("beta_M", "beta_V", "beta_MxV"):
            mean, p = exact_sign_flip(data[metric])
            records.append(
                {
                    "transition_type": transition,
                    "metric": metric,
                    "mean_subject_effect": mean,
                    "exact_sign_flip_p": p,
                    "n_subjects": int(data[metric].notna().sum()),
                    "n_positive": int((data[metric] > 0).sum()),
                    "n_negative": int((data[metric] < 0).sum()),
                }
            )
    summary = pd.DataFrame(records)
    summary["bh_fdr_q"] = np.nan
    primary = summary["metric"].isin(["beta_M", "beta_V"])
    summary.loc[primary, "bh_fdr_q"] = bh_fdr(
        summary.loc[primary, "exact_sign_flip_p"]
    )
    interaction = summary["metric"].eq("beta_MxV")
    summary.loc[interaction, "bh_fdr_q"] = bh_fdr(
        summary.loc[interaction, "exact_sign_flip_p"]
    )
    return subject_effects, summary


def plot_transition_specific_group(
    risk_frames: Sequence[pd.DataFrame],
    subject_effects: pd.DataFrame,
    group_summary: pd.DataFrame,
    output_path: Path,
    game: int,
) -> None:
    combined = pd.concat(risk_frames, ignore_index=True)
    finite_margin = pd.to_numeric(combined["transition_margin"], errors="coerce")
    finite_revision = pd.to_numeric(combined["value_revision_robust_z"], errors="coerce")
    x_limits = tuple(np.quantile(finite_margin[np.isfinite(finite_margin)], [0.01, 0.99]))
    y_limits = tuple(np.quantile(finite_revision[np.isfinite(finite_revision)], [0.01, 0.99]))
    fig, axes = plt.subplots(
        2, 2, figsize=(14.4, 11.4), facecolor="white",
        constrained_layout=True,
    )
    for column, transition in enumerate(("noop_to_active", "active_to_noop")):
        selected = combined.loc[combined["transition_type"].eq(transition)]
        _draw_transition_surface(
            axes[0, column], selected, transition, x_limits, y_limits,
            subject_equalized_frames=risk_frames,
            show_contours=False,
        )
        _draw_transition_effect_tile(
            axes[1, column], subject_effects, transition, "subject",
            lambda value: f"sub{int(value):03d}", group_summary=group_summary,
        )
    fig.suptitle(
        f"Cross-subject transition margin–Q_rev geometry · game {game}",
        fontsize=16.5, x=0.5, y=1.035,
    )
    save_figure(fig, output_path)


def fit_allocation_run_effects(frame: pd.DataFrame) -> pd.DataFrame:
    """Fit marginal standardized allocation slopes in two source-state risks.

    Each run/outcome is fit twice: outcome z ~ margin z and outcome z ~
    signed revision z, where signed revision is mean_j(delta Q_j) over valid
    status-2 events in the same search window.  These are deliberately
    univariate, unadjusted associations: there is no mutual adjustment, action
    demeaning, interaction, or control set.  The transition labels describe
    current-action risk strata rather than selection on the future action.
    """
    records: List[Dict[str, object]] = []
    working = frame.copy()
    incumbent = pd.to_numeric(working["incumbent_action"], errors="coerce")
    working = working.loc[incumbent.between(0, N_ACTIONS - 1)].copy()
    working["transition_type"] = np.where(
        pd.to_numeric(working["incumbent_action"], errors="coerce").eq(0),
        "noop_to_active",
        "active_to_noop",
    )
    working["allocation_margin"] = np.where(
        pd.to_numeric(working["incumbent_action"], errors="coerce").eq(0),
        pd.to_numeric(working["model_current_action_margin"], errors="coerce"),
        pd.to_numeric(working["model_current_vs_noop_margin"], errors="coerce"),
    )
    for (transition_type, run_label), run in working.groupby(
        ["transition_type", "run_label"], sort=False
    ):
        run = run.copy()
        run["M"] = zscore(run["allocation_margin"])
        run["Q_signed"] = zscore(run["value_revision_signed_mean"])
        for outcome, _, _ in ALLOCATION_METRICS:
            clean = run[["M", "Q_signed", outcome]].replace(
                [np.inf, -np.inf], np.nan
            ).dropna().copy()
            if len(clean) < 100:
                continue
            clean["Y"] = zscore(clean[outcome])
            clean = clean.dropna(subset=["M", "Q_signed", "Y"])
            if len(clean) < 100:
                continue
            design_m = np.column_stack(
                [np.ones(len(clean)), clean["M"].to_numpy(dtype=float)]
            )
            design_v = np.column_stack(
                [
                    np.ones(len(clean)),
                    clean["Q_signed"].to_numpy(dtype=float),
                ]
            )
            y = clean["Y"].to_numpy(dtype=float)
            beta_m = float(np.linalg.lstsq(design_m, y, rcond=None)[0][1])
            beta_v = float(np.linalg.lstsq(design_v, y, rcond=None)[0][1])
            record: Dict[str, object] = {
                "run_label": str(run_label),
                "subject": int(run["subject"].iloc[0]),
                "game": int(run["game"].iloc[0]),
                "transition_type": str(transition_type),
                "outcome": outcome,
                "n_rows": int(len(clean)),
                "beta_M": beta_m,
                "beta_Q_signed": beta_v,
            }
            records.append(record)
    return pd.DataFrame(records)


def build_completed_bout_allocation(bout_rows: pd.DataFrame) -> pd.DataFrame:
    """Return endpoint allocation and start controls for eligible bouts."""
    eligible = bout_rows.loc[
        bout_rows["completed"].eq(1) & bout_rows["onset_known"].eq(1)
    ].copy()
    records: List[Dict[str, object]] = []
    for bout_id, bout in eligible.groupby("action_bout_id", sort=False):
        bout = bout.sort_values("root_position", kind="stable")
        origin = int(bout.iloc[0]["incumbent_action"])
        margin_column = (
            "model_current_action_margin"
            if origin == 0
            else "model_current_vs_noop_margin"
        )
        comparable_columns = [
            margin_column,
            "value_revision_signed_mean",
            *[metric for metric, _, _ in ALLOCATION_METRICS],
        ]
        comparable = bout[comparable_columns].apply(
            pd.to_numeric, errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
        if not comparable.any():
            continue
        first = bout.loc[comparable].iloc[0]
        terminal = bout.iloc[-1]
        terminal_valid = terminal[
            [metric for metric, _, _ in ALLOCATION_METRICS]
        ].apply(lambda value: np.isfinite(pd.to_numeric(value, errors="coerce"))).all()
        if not bool(terminal_valid):
            continue
        record: Dict[str, object] = {
            "action_bout_id": int(bout_id),
            "run_label": str(first["run_label"]),
            "subject": int(first["subject"]),
            "game": int(first["game"]),
            "origin_action": origin,
            "previous_action": origin,
            "pre_bout_action": int(first["pre_bout_action"]),
            "destination_action": int(terminal["human_action"]),
            "transition_class": str(first["transition_class"]),
            "bout_duration_real_steps": int(len(bout)),
            "log_bout_duration": float(np.log(max(len(bout), 1))),
            "first_comparable_root_position": int(first["root_position"]),
            "previous_reward": float(
                pd.to_numeric(first["prior_reward"], errors="coerce")
            ),
            "start_transition_margin": float(
                pd.to_numeric(first[margin_column], errors="coerce")
            ),
            "start_value_revision_signed_mean": float(
                pd.to_numeric(
                    first["value_revision_signed_mean"], errors="coerce"
                )
            ),
        }
        for metric, _, _ in ALLOCATION_METRICS:
            start = float(pd.to_numeric(first[metric], errors="coerce"))
            end = float(pd.to_numeric(terminal[metric], errors="coerce"))
            record[f"{metric}_start"] = start
            record[f"{metric}_terminal"] = end
            record[f"{metric}_delta"] = end - start
        records.append(record)
    return pd.DataFrame(records)


def fit_duration_run_effects(
    bouts: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fit adjusted endpoint-change regressions and matched duration curves."""
    records: List[Dict[str, object]] = []
    curve_records: List[Dict[str, object]] = []
    if bouts.empty:
        return pd.DataFrame(), pd.DataFrame()
    eligible = bouts.loc[
        bouts["transition_class"].isin(("noop_to_active", "active_to_noop"))
    ].copy()
    for (transition_type, run_label), run in eligible.groupby(
        ["transition_class", "run_label"], sort=False
    ):
        run = run.copy().reset_index(drop=True)
        for outcome, _, _ in ALLOCATION_METRICS:
            start_column = f"{outcome}_start"
            terminal_column = f"{outcome}_terminal"
            required = [
                "log_bout_duration",
                "bout_duration_real_steps",
                "origin_action",
                "destination_action",
                "previous_reward",
                "start_transition_margin",
                "start_value_revision_signed_mean",
                start_column,
                terminal_column,
            ]
            clean = run[required].replace([np.inf, -np.inf], np.nan).copy()
            clean["previous_reward"] = pd.to_numeric(
                clean["previous_reward"], errors="coerce"
            ).fillna(0.0)
            clean = clean.dropna()
            if len(clean) < 30:
                continue
            start = pd.to_numeric(clean[start_column], errors="coerce").to_numpy(dtype=float)
            terminal = pd.to_numeric(
                clean[terminal_column], errors="coerce"
            ).to_numpy(dtype=float)
            endpoint_scale = float(np.nanstd(np.concatenate([start, terminal])))
            if not np.isfinite(endpoint_scale) or endpoint_scale <= EPS:
                continue
            y = (terminal - start) / endpoint_scale
            d = zscore(clean["log_bout_duration"])
            m = nuisance_zscore(clean["start_transition_margin"])
            v = nuisance_zscore(clean["start_value_revision_signed_mean"])
            reward = nuisance_zscore(clean["previous_reward"])
            origin_dummies = pd.get_dummies(
                clean["origin_action"].astype(int),
                prefix="origin",
                drop_first=True,
                dtype=float,
            )
            destination_dummies = pd.get_dummies(
                clean["destination_action"].astype(int),
                prefix="destination",
                drop_first=True,
                dtype=float,
            )
            controls = pd.DataFrame(
                {
                    "previous_reward": reward,
                    "start_transition_margin": m,
                    "start_signed_q_revision": v,
                },
                index=clean.index,
            )
            controls = pd.concat(
                [
                    controls,
                    origin_dummies.set_axis(clean.index),
                    destination_dummies.set_axis(clean.index),
                ],
                axis=1,
            )
            keep_controls = [
                column
                for column in controls.columns
                if float(np.nanstd(controls[column].to_numpy(dtype=float))) > EPS
            ]
            controls = controls[keep_controls]
            design_names = ["intercept", "log_bout_duration", *keep_controls]
            design = np.column_stack(
                [
                    np.ones(len(clean), dtype=float),
                    d.to_numpy(dtype=float),
                    controls.to_numpy(dtype=float),
                ]
            )
            valid = np.isfinite(y) & np.all(np.isfinite(design), axis=1)
            fitted = design[valid]
            if (
                valid.sum() < max(30, fitted.shape[1] + 5)
                or np.linalg.matrix_rank(fitted) < fitted.shape[1]
            ):
                continue
            beta = np.linalg.lstsq(fitted, y[valid], rcond=None)[0]
            log_values = pd.to_numeric(
                clean.loc[valid, "log_bout_duration"], errors="coerce"
            ).to_numpy(dtype=float)
            log_mean = float(np.mean(log_values))
            log_sd = float(np.std(log_values))
            if not np.isfinite(log_sd) or log_sd <= EPS:
                continue
            valid_duration = pd.to_numeric(
                clean.loc[valid, "bout_duration_real_steps"], errors="coerce"
            ).to_numpy(dtype=float)
            duration_min = max(int(np.nanmin(valid_duration)), 1)
            duration_max = max(int(np.nanmax(valid_duration)), duration_min)
            coefficients = dict(zip(design_names, beta))
            record = {
                "run_label": str(run_label),
                "subject": int(run["subject"].iloc[0]),
                "game": int(run["game"].iloc[0]),
                "transition_type": str(transition_type),
                "outcome": outcome,
                "n_completed_known_onset_bouts": int(valid.sum()),
                "endpoint_pooled_start_terminal_sd": endpoint_scale,
                "duration_log_mean": log_mean,
                "duration_log_sd": log_sd,
                "duration_min": duration_min,
                "duration_max": duration_max,
                "beta_log_bout_duration": float(beta[1]),
                "beta_previous_reward": float(
                    coefficients.get("previous_reward", 0.0)
                ),
                "beta_start_transition_margin": float(
                    coefficients.get("start_transition_margin", 0.0)
                ),
                "beta_start_signed_q_revision": float(
                    coefficients.get("start_signed_q_revision", 0.0)
                ),
                "n_origin_dummy_columns": int(origin_dummies.shape[1]),
                "n_destination_dummy_columns": int(destination_dummies.shape[1]),
                "design_rank": int(np.linalg.matrix_rank(fitted)),
                "design_columns": int(fitted.shape[1]),
            }
            records.append(record)
            duration_grid = np.arange(
                duration_min, duration_max + 1, dtype=int
            )
            for duration in duration_grid:
                standardized_log_duration = (
                    np.log(float(max(duration, 1))) - log_mean
                ) / log_sd
                curve_records.append(
                    {
                        "run_label": str(run_label),
                        "subject": int(run["subject"].iloc[0]),
                        "game": int(run["game"].iloc[0]),
                        "transition_type": str(transition_type),
                        "outcome": outcome,
                        "bout_duration_real_steps": int(duration),
                        "adjusted_duration_contribution": float(
                            beta[1] * standardized_log_duration
                        ),
                    }
                )
    return pd.DataFrame(records), pd.DataFrame(curve_records)


def summarize_allocation_group_effects(
    effects: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    columns = ["beta_M", "beta_Q_signed"]
    subject_effects = effects.groupby(
        ["subject", "transition_type", "outcome"], as_index=False
    )[columns].mean()
    records: List[Dict[str, object]] = []
    for (transition_type, outcome), data in subject_effects.groupby(
        ["transition_type", "outcome"], sort=False
    ):
        for metric in columns:
            mean, p = exact_sign_flip(data[metric])
            records.append(
                {
                    "outcome": outcome,
                    "transition_type": transition_type,
                    "metric": metric,
                    "mean_subject_effect": mean,
                    "exact_sign_flip_p": p,
                    "n_subjects": int(data[metric].notna().sum()),
                    "n_positive": int((data[metric] > 0).sum()),
                    "n_negative": int((data[metric] < 0).sum()),
                }
            )
    summary = pd.DataFrame(records)
    summary["bh_fdr_q"] = bh_fdr(summary["exact_sign_flip_p"])
    return subject_effects, summary


def summarize_allocation_run_effects(effects: pd.DataFrame) -> pd.DataFrame:
    """Summarize the 12 marginal allocation tests across runs."""
    records: List[Dict[str, object]] = []
    for (transition_type, outcome), data in effects.groupby(
        ["transition_type", "outcome"], sort=False
    ):
        for metric in ("beta_M", "beta_Q_signed"):
            values = pd.to_numeric(data[metric], errors="coerce").dropna()
            mean, p = exact_sign_flip(values)
            records.append(
                {
                    "transition_type": transition_type,
                    "outcome": outcome,
                    "metric": metric,
                    "mean_run_effect": mean,
                    "exact_sign_flip_p": p,
                    "n_runs": int(len(values)),
                    "n_positive": int((values > 0).sum()),
                    "n_negative": int((values < 0).sum()),
                }
            )
    summary = pd.DataFrame(records)
    if not summary.empty:
        summary["bh_fdr_q"] = bh_fdr(summary["exact_sign_flip_p"])
    return summary


def summarize_duration_units(
    effects: pd.DataFrame,
    unit_column: str,
) -> pd.DataFrame:
    """Summarize duration coefficients over runs or subjects."""
    records: List[Dict[str, object]] = []
    for (transition_type, outcome), data in effects.groupby(
        ["transition_type", "outcome"], sort=False
    ):
        values = pd.to_numeric(
            data["beta_log_bout_duration"], errors="coerce"
        ).dropna().to_numpy(dtype=float)
        mean, p = exact_sign_flip(values)
        n = int(len(values))
        sem = float(stats.sem(values)) if n >= 2 else np.nan
        critical = float(stats.t.ppf(0.975, n - 1)) if n >= 2 else np.nan
        records.append(
            {
                "outcome": outcome,
                "transition_type": transition_type,
                "metric": "beta_log_bout_duration",
                "unit": unit_column,
                "mean_unit_effect": mean,
                "ci_low": mean - critical * sem if n >= 2 else np.nan,
                "ci_high": mean + critical * sem if n >= 2 else np.nan,
                "exact_sign_flip_p": p,
                "n_units": n,
                "n_positive": int(np.sum(values > 0)),
                "n_negative": int(np.sum(values < 0)),
            }
        )
    summary = pd.DataFrame(records)
    if not summary.empty:
        summary["bh_fdr_q"] = bh_fdr(summary["exact_sign_flip_p"])
    return summary


def summarize_duration_group_effects(
    effects: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    subject_effects = effects.groupby(
        ["subject", "transition_type", "outcome"], as_index=False
    )["beta_log_bout_duration"].mean()
    summary = summarize_duration_units(subject_effects, "subject")
    return subject_effects, summary


def plot_allocation_effects(
    effects: pd.DataFrame,
    output_path: Path,
    title_suffix: str,
    *,
    unit: str = "run_label",
    group_summary: Optional[pd.DataFrame] = None,
) -> None:
    coefficient_columns = ["beta_M", "beta_Q_signed"]
    labels = [TRANSITION_MARGIN_LABEL, SIGNED_Q_REVISION_LABEL]
    colors = [BLUE, ORANGE]
    markers = ["o", "^"]
    transition_order = ("noop_to_active", "active_to_noop")
    row_subtitles = {
        "noop_to_active": "A. NOOP → Active risk windows · current action = NOOP",
        "active_to_noop": "B. Active → NOOP risk windows · current action = active",
    }
    unit_values = sorted(
        effects[unit].dropna().unique(),
        key=(lambda value: int(value)) if unit == "subject" else (lambda value: str(value)),
    )
    positions = np.arange(len(unit_values))
    def formatter(value: object) -> str:
        if unit == "subject":
            return f"sub{int(value):03d}"
        parts = str(value).split("_")
        if len(parts) >= 3:
            return f"{parts[1]}\n{parts[2]}"
        return str(value).replace("_", "\n")
    fig, axes = plt.subplots(
        2, 3, figsize=(15.4, 9.7), facecolor="white", sharex=True, sharey="col"
    )
    for row_index, transition_type in enumerate(transition_order):
        for column_index, (outcome, title, subtitle) in enumerate(ALLOCATION_METRICS):
            axis = axes[row_index, column_index]
            selected = effects.loc[
                effects["transition_type"].eq(transition_type)
                & effects["outcome"].eq(outcome)
            ].copy()
            data = selected.groupby(unit, as_index=True)[coefficient_columns].mean().reindex(unit_values)
            for column, label, color, marker in zip(
                coefficient_columns, labels, colors, markers
            ):
                axis.plot(
                    positions, data[column], color=color, marker=marker,
                    lw=1.15, ms=5.0, label=label, alpha=0.9,
                )
            axis.axhline(0, color=INK, lw=0.9, ls="--")
            if row_index == 0:
                axis.set_title(f"{title}\n{subtitle}")
                axis.tick_params(labelbottom=False)
            else:
                axis.set_xticks(positions)
                axis.set_xticklabels(
                    [formatter(value) for value in unit_values], rotation=35, ha="right"
                )
            if column_index == 0:
                axis.set_ylabel("Standardized coefficient")
            if group_summary is not None:
                rows = group_summary.loc[
                    group_summary["transition_type"].eq(transition_type)
                    & group_summary["outcome"].eq(outcome)
                ]
                text_bits = []
                for column, short in zip(
                    coefficient_columns, ["Transition margin", "Q_signed"]
                ):
                    one = rows.loc[rows["metric"].eq(column)]
                    if not one.empty:
                        text_bits.append(
                            f"{short}: p={one['exact_sign_flip_p'].iloc[0]:.3f}, "
                            f"q={one['bh_fdr_q'].iloc[0]:.3f}"
                        )
                axis.text(
                    0.02, 0.98, "\n".join(text_bits), transform=axis.transAxes,
                    va="top", fontsize=8.1,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78},
                )
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, legend_labels, loc="lower center", ncol=3,
        bbox_to_anchor=(0.5, 0.055), frameon=False,
    )
    fig.suptitle(
        f"Transition margin/Q_signed and search allocation - {title_suffix}",
        fontsize=16, y=0.995,
    )
    fig.text(0.055, 0.905, row_subtitles["noop_to_active"], fontsize=11.5, weight="bold")
    fig.text(0.055, 0.495, row_subtitles["active_to_noop"], fontsize=11.5, weight="bold")
    fig.text(
        0.5, 0.018,
        "Each point is one run (individual) or one subject mean (group); lines "
        "only guide sign-consistency reading. Separate standardized models are "
        "Y~transition margin and Y~Q_signed, with no mutual adjustment or "
        "covariates. Q_signed/allocation are same-window.",
        ha="center", fontsize=9.3,
    )
    fig.subplots_adjust(
        left=0.055, right=0.99, bottom=0.20, top=0.85,
        hspace=0.50, wspace=0.28,
    )
    save_figure(fig, output_path)


def build_allocation_scatter_points(
    frame: pd.DataFrame,
    *,
    unit: str,
) -> pd.DataFrame:
    """Build equal-unit Figure-6 scatter points from window-level metrics.

    Individual figures use one point per run.  Group figures first average
    windows within run and then average runs within subject, so every subject
    contributes one equally weighted point regardless of window count.
    """
    if unit not in {"run_label", "subject"}:
        raise ValueError(f"Unsupported Figure-6 scatter unit: {unit}")
    working = frame.copy()
    incumbent = pd.to_numeric(working["incumbent_action"], errors="coerce")
    working = working.loc[incumbent.between(0, N_ACTIONS - 1)].copy()
    working["transition_type"] = np.where(
        pd.to_numeric(working["incumbent_action"], errors="coerce").eq(0),
        "noop_to_active",
        "active_to_noop",
    )
    working["allocation_margin"] = np.where(
        pd.to_numeric(working["incumbent_action"], errors="coerce").eq(0),
        pd.to_numeric(working["model_current_action_margin"], errors="coerce"),
        pd.to_numeric(working["model_current_vs_noop_margin"], errors="coerce"),
    )
    aggregation: Dict[str, Tuple[str, str]] = {
        "n_windows": ("allocation_margin", "size"),
        "M_raw": ("allocation_margin", "mean"),
        "Q_signed_raw": ("value_revision_signed_mean", "mean"),
    }
    for outcome, _, _ in ALLOCATION_METRICS:
        aggregation[f"{outcome}_raw"] = (outcome, "mean")
    run_points = (
        working.groupby(
            ["subject", "game", "run_label", "transition_type"],
            as_index=False,
            sort=False,
        )
        .agg(**aggregation)
    )
    if unit == "subject":
        value_columns = [
            "M_raw",
            "Q_signed_raw",
            *[f"{outcome}_raw" for outcome, _, _ in ALLOCATION_METRICS],
        ]
        subject_values = run_points.groupby(
            ["subject", "game", "transition_type"],
            as_index=False,
            sort=False,
        )[value_columns].mean()
        subject_counts = run_points.groupby(
            ["subject", "game", "transition_type"],
            as_index=False,
            sort=False,
        ).agg(
            n_runs=("run_label", "nunique"),
            n_windows=("n_windows", "sum"),
        )
        unit_points = subject_values.merge(
            subject_counts,
            on=["subject", "game", "transition_type"],
            how="left",
            validate="one_to_one",
        )
        unit_points["unit_value"] = unit_points["subject"]
    else:
        unit_points = run_points.copy()
        unit_points["n_runs"] = 1
        unit_points["unit_value"] = unit_points["run_label"]

    records: List[Dict[str, object]] = []
    predictors = (
        ("M", "M_raw", TRANSITION_MARGIN_LABEL),
        ("Q_signed", "Q_signed_raw", SIGNED_Q_REVISION_LABEL),
    )
    for row in unit_points.itertuples(index=False):
        for outcome, outcome_title, outcome_subtitle in ALLOCATION_METRICS:
            y_raw = float(getattr(row, f"{outcome}_raw"))
            for predictor, predictor_column, predictor_label in predictors:
                records.append(
                    {
                        "subject": int(row.subject),
                        "game": int(row.game),
                        "unit": unit,
                        "unit_value": getattr(row, "unit_value"),
                        "transition_type": str(row.transition_type),
                        "predictor": predictor,
                        "predictor_label": predictor_label,
                        "outcome": outcome,
                        "outcome_title": outcome_title,
                        "outcome_subtitle": outcome_subtitle,
                        "x_raw": float(getattr(row, predictor_column)),
                        "y_raw": y_raw,
                        "n_runs": int(row.n_runs),
                        "n_windows": int(row.n_windows),
                    }
                )
    points = pd.DataFrame(records)
    panel_keys = ["transition_type", "predictor", "outcome"]
    points["x_z_across_units"] = points.groupby(
        panel_keys, sort=False
    )["x_raw"].transform(zscore)
    points["y_z_across_units"] = points.groupby(
        panel_keys, sort=False
    )["y_raw"].transform(zscore)
    return points


def summarize_allocation_scatter_trends(points: pd.DataFrame) -> pd.DataFrame:
    """Fit descriptive across-unit trend lines shown in Figure 6."""
    records: List[Dict[str, object]] = []
    for (transition_type, predictor, outcome), data in points.groupby(
        ["transition_type", "predictor", "outcome"], sort=False
    ):
        clean = data[["x_z_across_units", "y_z_across_units"]].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        if len(clean) < 3:
            continue
        result = stats.linregress(
            clean["x_z_across_units"].to_numpy(dtype=float),
            clean["y_z_across_units"].to_numpy(dtype=float),
        )
        critical = float(stats.t.ppf(0.975, len(clean) - 2))
        records.append(
            {
                "unit": str(data["unit"].iloc[0]),
                "transition_type": str(transition_type),
                "predictor": str(predictor),
                "outcome": str(outcome),
                "n_units": int(len(clean)),
                "slope_standardized": float(result.slope),
                "intercept_standardized": float(result.intercept),
                "pearson_r": float(result.rvalue),
                "ols_two_sided_p_descriptive": float(result.pvalue),
                "slope_ci_low": float(result.slope - critical * result.stderr),
                "slope_ci_high": float(result.slope + critical * result.stderr),
            }
        )
    summary = pd.DataFrame(records)
    if not summary.empty:
        summary["bh_fdr_q_descriptive"] = bh_fdr(
            summary["ols_two_sided_p_descriptive"]
        )
    return summary


def plot_allocation_scatter_trends(
    points: pd.DataFrame,
    trends: pd.DataFrame,
    output_path: Path,
    title_suffix: str,
    *,
    unit: str,
) -> None:
    """Plot Figure 6 as x-y scatter panels with one point per analysis unit."""
    transition_order = ("noop_to_active", "active_to_noop")
    predictor_order = ("M", "Q_signed")
    predictor_colors = {"M": BLUE, "Q_signed": ORANGE}
    predictor_markers = {"M": "o", "Q_signed": "^"}
    row_subtitles = {
        "noop_to_active": "A. NOOP → Active risk windows · current action = NOOP",
        "active_to_noop": "B. Active → NOOP risk windows · current action = active",
    }
    grain = "run" if unit == "run_label" else "subject"
    fig, axes = plt.subplots(
        2,
        6,
        figsize=(22.0, 9.6),
        facecolor="white",
        sharex=True,
        sharey=True,
    )
    panels = [
        (outcome, title, subtitle, predictor)
        for outcome, title, subtitle in ALLOCATION_METRICS
        for predictor in predictor_order
    ]
    for row_index, transition_type in enumerate(transition_order):
        for column_index, (outcome, title, subtitle, predictor) in enumerate(panels):
            axis = axes[row_index, column_index]
            selected = points.loc[
                points["transition_type"].eq(transition_type)
                & points["outcome"].eq(outcome)
                & points["predictor"].eq(predictor)
            ].copy()
            color = predictor_colors[predictor]
            marker = predictor_markers[predictor]
            axis.scatter(
                selected["x_z_across_units"],
                selected["y_z_across_units"],
                s=38 if unit == "run_label" else 52,
                marker=marker,
                facecolor=color,
                edgecolor="white",
                linewidth=0.7,
                alpha=0.9,
                zorder=3,
            )
            one_trend = trends.loc[
                trends["transition_type"].eq(transition_type)
                & trends["outcome"].eq(outcome)
                & trends["predictor"].eq(predictor)
            ]
            if not one_trend.empty:
                trend = one_trend.iloc[0]
                x_values = selected["x_z_across_units"].to_numpy(dtype=float)
                x_grid = np.linspace(float(np.nanmin(x_values)), float(np.nanmax(x_values)), 100)
                y_grid = (
                    float(trend["intercept_standardized"])
                    + float(trend["slope_standardized"]) * x_grid
                )
                axis.plot(x_grid, y_grid, color=color, lw=2.0, zorder=2)
                axis.text(
                    0.04,
                    0.96,
                    f"trend r={trend['pearson_r']:+.2f}\nn={int(trend['n_units'])} {grain}s",
                    transform=axis.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8.0,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82},
                )
            if unit == "subject":
                for point in selected.itertuples(index=False):
                    axis.annotate(
                        str(int(point.subject)),
                        (point.x_z_across_units, point.y_z_across_units),
                        xytext=(3, 3),
                        textcoords="offset points",
                        fontsize=7.3,
                        color=INK,
                    )
            axis.axhline(0, color=GRID, lw=0.8, zorder=0)
            axis.axvline(0, color=GRID, lw=0.8, zorder=0)
            if row_index == 0:
                predictor_title = "M" if predictor == "M" else "Q_signed"
                axis.set_title(f"{title}\n{predictor_title} → allocation", fontsize=9.4)
            if row_index == 1:
                axis.set_xlabel(
                    f"{grain.capitalize()}-mean "
                    f"{'M' if predictor == 'M' else 'Q_signed'}\n(z across {grain}s)",
                    fontsize=8.5,
                )
            if column_index == 0:
                axis.set_ylabel(
                    f"{grain.capitalize()}-mean allocation\n(z across {grain}s)",
                    fontsize=9.0,
                )
    fig.suptitle(
        f"Evaluation/revision and search allocation scatter · {title_suffix}",
        fontsize=16,
        y=0.995,
    )
    fig.text(0.04, 0.905, row_subtitles["noop_to_active"], fontsize=11.5, weight="bold")
    fig.text(0.04, 0.495, row_subtitles["active_to_noop"], fontsize=11.5, weight="bold")
    fig.text(
        0.5,
        0.018,
        f"Each point is one {grain}; predictor and allocation are means of the same risk-window rows, then z-scored across displayed {grain}s. Lines are descriptive across-{grain} OLS trends, not the within-run slope/FDR inference saved separately.",
        ha="center",
        fontsize=9.2,
    )
    fig.subplots_adjust(
        left=0.045,
        right=0.995,
        bottom=0.19,
        top=0.85,
        hspace=0.48,
        wspace=0.24,
    )
    save_figure(fig, output_path)


def build_allocation_within_run_bins(
    frame: pd.DataFrame,
    *,
    n_bins: int = 9,
) -> pd.DataFrame:
    """Return run-balanced binned points for the Figure-6 within-run display.

    The standardization and complete-case rows exactly mirror
    ``fit_allocation_run_effects``. Quantile bins are formed independently in
    each run so dense runs do not dominate the displayed mean curve.
    """
    if n_bins < 3:
        raise ValueError("Figure-6 display requires at least three bins")
    working = frame.copy()
    incumbent = pd.to_numeric(working["incumbent_action"], errors="coerce")
    working = working.loc[incumbent.between(0, N_ACTIONS - 1)].copy()
    working["transition_type"] = np.where(
        pd.to_numeric(working["incumbent_action"], errors="coerce").eq(0),
        "noop_to_active",
        "active_to_noop",
    )
    working["allocation_margin"] = np.where(
        pd.to_numeric(working["incumbent_action"], errors="coerce").eq(0),
        pd.to_numeric(working["model_current_action_margin"], errors="coerce"),
        pd.to_numeric(working["model_current_vs_noop_margin"], errors="coerce"),
    )
    records: List[Dict[str, object]] = []
    for (transition_type, run_label), run in working.groupby(
        ["transition_type", "run_label"], sort=False
    ):
        run = run.copy()
        run["M"] = zscore(run["allocation_margin"])
        run["Q_signed"] = zscore(run["value_revision_signed_mean"])
        for outcome, _, _ in ALLOCATION_METRICS:
            clean = run[["M", "Q_signed", outcome]].replace(
                [np.inf, -np.inf], np.nan
            ).dropna().copy()
            if len(clean) < 100:
                continue
            clean["Y"] = zscore(clean[outcome])
            clean = clean.dropna(subset=["M", "Q_signed", "Y"])
            if len(clean) < 100:
                continue
            for predictor in ("M", "Q_signed"):
                percentile = clean[predictor].rank(
                    method="first", pct=True
                ).to_numpy(dtype=float)
                bin_index = np.minimum(
                    (percentile * n_bins).astype(int), n_bins - 1
                )
                binned = (
                    clean.assign(bin_index=bin_index)
                    .groupby("bin_index", as_index=False, sort=True)
                    .agg(
                        predictor_z=(predictor, "mean"),
                        allocation_z=("Y", "mean"),
                        n_windows=("Y", "size"),
                    )
                )
                for point in binned.itertuples(index=False):
                    records.append(
                        {
                            "subject": int(run["subject"].iloc[0]),
                            "game": int(run["game"].iloc[0]),
                            "run_label": str(run_label),
                            "transition_type": str(transition_type),
                            "outcome": str(outcome),
                            "predictor": predictor,
                            "bin_index": int(point.bin_index),
                            "predictor_z": float(point.predictor_z),
                            "allocation_z": float(point.allocation_z),
                            "n_windows": int(point.n_windows),
                        }
                    )
    return pd.DataFrame(records)


def allocation_unit_effects(
    effects: pd.DataFrame,
    *,
    unit: str,
) -> pd.DataFrame:
    """Put the existing within-run coefficients at the displayed unit grain."""
    if unit == "subject":
        base = effects.groupby(
            ["subject", "transition_type", "outcome"],
            as_index=False,
            sort=False,
        )[["beta_M", "beta_Q_signed"]].mean()
        base["unit_value"] = base["subject"]
    elif unit == "run_label":
        base = effects.copy()
        base["unit_value"] = base["run_label"]
    else:
        raise ValueError(f"Unsupported Figure-6 coefficient unit: {unit}")
    long = base.melt(
        id_vars=["subject", "transition_type", "outcome", "unit_value"],
        value_vars=["beta_M", "beta_Q_signed"],
        var_name="metric",
        value_name="beta",
    )
    long["predictor"] = long["metric"].map(
        {"beta_M": "M", "beta_Q_signed": "Q_signed"}
    )
    return long


def plot_allocation_within_run_consistency(
    frame: pd.DataFrame,
    effects: pd.DataFrame,
    summary: pd.DataFrame,
    output_path: Path,
    title_suffix: str,
    *,
    unit: str,
) -> None:
    """Plot six panels that preserve the original within-run estimand.

    M and signed Q share each panel. Thin lines are run slopes in a subject
    figure and subject slopes in a group figure. Thick lines and bands are the
    equal-unit mean and its 95% t interval. Insets show every unit coefficient.
    """
    transition_order = ("noop_to_active", "active_to_noop")
    row_subtitles = {
        "noop_to_active": "A. NOOP → Active risk windows · current action = NOOP",
        "active_to_noop": "B. Active → NOOP risk windows · current action = active",
    }
    colors = {"M": BLUE, "Q_signed": ORANGE}
    light_colors = {"M": BLUE_LIGHT, "Q_signed": ORANGE_LIGHT}
    markers = {"M": "o", "Q_signed": "^"}
    linestyles = {"M": "-", "Q_signed": "--"}
    grain = "run" if unit == "run_label" else "subject"
    binned = build_allocation_within_run_bins(frame)
    unit_effect = allocation_unit_effects(effects, unit=unit)
    if unit == "subject":
        unit_bins = (
            binned.groupby(
                [
                    "subject",
                    "transition_type",
                    "outcome",
                    "predictor",
                    "bin_index",
                ],
                as_index=False,
                sort=False,
            )[["predictor_z", "allocation_z"]]
            .mean()
        )
        unit_bins["unit_value"] = unit_bins["subject"]
    else:
        unit_bins = binned.copy()
        unit_bins["unit_value"] = unit_bins["run_label"]

    finite_beta = pd.to_numeric(unit_effect["beta"], errors="coerce")
    finite_y = pd.to_numeric(unit_bins["allocation_z"], errors="coerce")
    candidates = np.r_[
        np.abs(finite_beta[np.isfinite(finite_beta)].to_numpy()) * 2.2,
        np.abs(finite_y[np.isfinite(finite_y)].to_numpy()),
    ]
    y_bound = 0.7
    if candidates.size:
        y_bound = float(np.clip(np.nanquantile(candidates, 0.995) * 1.18, 0.7, 2.4))

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(16.8, 10.0),
        facecolor="white",
        sharex=True,
        sharey=True,
    )
    x_grid = np.linspace(-2.25, 2.25, 121)
    for row_index, transition_type in enumerate(transition_order):
        for column_index, (outcome, title, subtitle) in enumerate(ALLOCATION_METRICS):
            axis = axes[row_index, column_index]
            axis.axhline(0, color=GRID, lw=0.8, zorder=0)
            axis.axvline(0, color=GRID, lw=0.8, zorder=0)
            for predictor in ("M", "Q_signed"):
                selected_effect = unit_effect.loc[
                    unit_effect["transition_type"].eq(transition_type)
                    & unit_effect["outcome"].eq(outcome)
                    & unit_effect["predictor"].eq(predictor)
                ].copy()
                betas = pd.to_numeric(
                    selected_effect["beta"], errors="coerce"
                ).dropna().to_numpy(dtype=float)
                if not len(betas):
                    continue
                color = colors[predictor]
                for beta in betas:
                    axis.plot(
                        x_grid,
                        beta * x_grid,
                        color=color,
                        lw=0.7,
                        alpha=0.16 if unit == "run_label" else 0.22,
                        linestyle=linestyles[predictor],
                        zorder=1,
                    )
                predictions = np.outer(betas, x_grid)
                mean_prediction = predictions.mean(axis=0)
                if len(betas) >= 2:
                    critical = float(stats.t.ppf(0.975, len(betas) - 1))
                    half_width = critical * stats.sem(predictions, axis=0)
                    axis.fill_between(
                        x_grid,
                        mean_prediction - half_width,
                        mean_prediction + half_width,
                        color=light_colors[predictor],
                        alpha=0.22,
                        linewidth=0,
                        zorder=1,
                    )
                axis.plot(
                    x_grid,
                    mean_prediction,
                    color=color,
                    lw=2.5,
                    linestyle=linestyles[predictor],
                    zorder=3,
                )
                selected_bins = unit_bins.loc[
                    unit_bins["transition_type"].eq(transition_type)
                    & unit_bins["outcome"].eq(outcome)
                    & unit_bins["predictor"].eq(predictor)
                ].copy()
                bin_summary = selected_bins.groupby(
                    "bin_index", as_index=False, sort=True
                ).agg(
                    predictor_z=("predictor_z", "mean"),
                    allocation_z=("allocation_z", "mean"),
                )
                axis.scatter(
                    bin_summary["predictor_z"],
                    bin_summary["allocation_z"],
                    s=30,
                    marker=markers[predictor],
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.65,
                    alpha=0.95,
                    zorder=4,
                )

                metric = "beta_M" if predictor == "M" else "beta_Q_signed"
                one_summary = summary.loc[
                    summary["transition_type"].eq(transition_type)
                    & summary["outcome"].eq(outcome)
                    & summary["metric"].eq(metric)
                ]
                if not one_summary.empty:
                    row = one_summary.iloc[0]
                    q = float(row.get("bh_fdr_q", np.nan))
                    n_positive = int(row.get("n_positive", np.sum(betas > 0)))
                    axis.text(
                        0.03,
                        0.96 if predictor == "M" else 0.885,
                        f"{'M' if predictor == 'M' else 'Q_signed'}: "
                        f"β={np.mean(betas):+.3f}, +{n_positive}/{len(betas)}, "
                        f"q={q:.3f}",
                        transform=axis.transAxes,
                        ha="left",
                        va="top",
                        fontsize=8.2,
                        color=color,
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78},
                        zorder=6,
                    )

            inset = axis.inset_axes([0.72, 0.61, 0.25, 0.32])
            inset.set_facecolor((1, 1, 1, 0.90))
            for predictor_index, predictor in enumerate(("M", "Q_signed")):
                selected = unit_effect.loc[
                    unit_effect["transition_type"].eq(transition_type)
                    & unit_effect["outcome"].eq(outcome)
                    & unit_effect["predictor"].eq(predictor)
                ]
                values = pd.to_numeric(selected["beta"], errors="coerce").dropna().to_numpy(dtype=float)
                if not len(values):
                    continue
                offsets = np.linspace(-0.10, 0.10, len(values))
                inset.scatter(
                    predictor_index + offsets,
                    values,
                    s=13 if unit == "run_label" else 18,
                    marker=markers[predictor],
                    color=colors[predictor],
                    alpha=0.72,
                    edgecolor="white",
                    linewidth=0.35,
                    zorder=2,
                )
                mean = float(np.mean(values))
                half = 0.0
                if len(values) >= 2:
                    half = float(
                        stats.t.ppf(0.975, len(values) - 1) * stats.sem(values)
                    )
                inset.errorbar(
                    predictor_index,
                    mean,
                    yerr=half,
                    fmt=markers[predictor],
                    ms=6.0,
                    mfc=colors[predictor],
                    mec=INK,
                    mew=0.7,
                    ecolor=INK,
                    elinewidth=1.0,
                    capsize=2.2,
                    zorder=3,
                )
            inset.axhline(0, color=INK, lw=0.7, alpha=0.7)
            inset.set_xticks([0, 1], ["M", "Q"], fontsize=6.5)
            inset.tick_params(axis="y", labelsize=6.2, length=2)
            inset.set_title(f"{grain} β", fontsize=6.8, pad=1.5)
            inset.grid(False)
            for spine in inset.spines.values():
                spine.set_color(GRID)
                spine.set_linewidth(0.65)

            axis.set_xlim(-2.3, 2.3)
            axis.set_ylim(-y_bound, y_bound)
            axis.set_title(f"{title}\n{subtitle}", fontsize=10.5)
            if row_index == 1:
                axis.set_xlabel("Within-run standardized predictor", fontsize=9.5)
            if column_index == 0:
                axis.set_ylabel("Within-run standardized allocation", fontsize=9.5)

    handles = [
        Line2D([0], [0], color=BLUE, marker="o", lw=2.5, label="M"),
        Line2D([0], [0], color=ORANGE, marker="^", lw=2.5, linestyle="--", label="Signed Q revision"),
        Line2D([0], [0], color=INK, lw=0.8, alpha=0.35, label=f"Thin = one {grain}"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=3,
        fontsize=9.5,
    )
    fig.suptitle(
        f"Within-run evaluation/revision → search allocation · {title_suffix}",
        fontsize=16,
        y=0.995,
    )
    fig.text(0.055, 0.90, row_subtitles["noop_to_active"], fontsize=11.5, weight="bold")
    fig.text(0.055, 0.485, row_subtitles["active_to_noop"], fontsize=11.5, weight="bold")
    fig.text(
        0.5,
        0.018,
        f"Points are equal-{grain} means of nine within-run quantile bins; thin lines and inset points are the original marginal within-run β estimates. Thick lines and ribbons are the equal-{grain} mean and 95% t interval. q is BH-FDR across the 12 Figure-6 effects.",
        ha="center",
        fontsize=8.9,
    )
    fig.subplots_adjust(
        left=0.065,
        right=0.985,
        bottom=0.10,
        top=0.845,
        hspace=0.48,
        wspace=0.22,
    )
    save_figure(fig, output_path)


def build_duration_component_curves(
    effects: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate beta_D*z_run(log duration) only on each run's support."""
    records: List[Dict[str, object]] = []
    for row in effects.itertuples(index=False):
        duration_min = max(int(row.duration_min), 1)
        duration_max = max(int(row.duration_max), duration_min)
        duration_grid = np.arange(
            duration_min, duration_max + 1, dtype=int
        )
        for duration in duration_grid:
            standardized = (
                np.log(float(max(duration, 1))) - float(row.duration_log_mean)
            ) / float(row.duration_log_sd)
            records.append(
                {
                    "run_label": str(row.run_label),
                    "subject": int(row.subject),
                    "game": int(row.game),
                    "transition_type": str(row.transition_type),
                    "outcome": str(row.outcome),
                    "bout_duration_real_steps": int(duration),
                    "adjusted_duration_contribution": float(
                        row.beta_log_bout_duration * standardized
                    ),
                }
            )
    return pd.DataFrame(records)


def plot_completed_bout_duration(
    bouts: pd.DataFrame,
    effects: pd.DataFrame,
    summary: pd.DataFrame,
    output_path: Path,
    title_suffix: str,
    *,
    unit: str,
) -> None:
    transition_order = ("noop_to_active", "active_to_noop")
    row_subtitles = {
        "noop_to_active": "A. Completed NOOP → Active bouts",
        "active_to_noop": "B. Completed Active → NOOP bouts",
    }
    data = effects.loc[
        effects["transition_type"].isin(transition_order)
    ].copy()
    data = data.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[
            "beta_log_bout_duration",
            "duration_log_mean",
            "duration_log_sd",
            "duration_min",
            "duration_max",
        ]
    )
    data["support_low"] = (
        np.log(data["duration_min"].clip(lower=1))
        - data["duration_log_mean"]
    ) / data["duration_log_sd"]
    data["support_high"] = (
        np.log(data["duration_max"].clip(lower=1))
        - data["duration_log_mean"]
    ) / data["duration_log_sd"]
    if unit == "subject":
        display_effects = data.groupby(
            ["subject", "transition_type", "outcome"], as_index=False
        ).agg(
            beta_log_bout_duration=("beta_log_bout_duration", "mean"),
            support_low=("support_low", "max"),
            support_high=("support_high", "min"),
        )
        display_unit = "subject"
        unit_label = "subject"
    else:
        display_effects = data.copy()
        display_unit = "run_label"
        unit_label = "run"

    fig, axes = plt.subplots(
        2, 4, figsize=(18.0, 9.7), facecolor="white",
        sharex=False, sharey=False,
    )
    for row_index, transition_type in enumerate(transition_order):
        distribution_axis = axes[row_index, 0]
        row_bouts = bouts.loc[
            bouts["transition_class"].eq(transition_type)
        ].copy()
        duration = pd.to_numeric(
            row_bouts["bout_duration_real_steps"], errors="coerce"
        )
        row_bouts = row_bouts.loc[duration.ge(1)].copy()
        row_bouts["bout_duration_real_steps"] = duration.loc[
            duration.ge(1)
        ].to_numpy(dtype=float)
        if not row_bouts.empty:
            maximum = int(row_bouts["bout_duration_real_steps"].max())
            lower_edges = [1, 2, 3, 5, 9, 17, 33, 65, 129, 257, 513, 1025]
            lower_edges = [value for value in lower_edges if value <= maximum]
            labels: List[str] = []
            masks: List[np.ndarray] = []
            values = row_bouts["bout_duration_real_steps"].to_numpy(dtype=float)
            for index, lower in enumerate(lower_edges):
                upper = (
                    lower_edges[index + 1]
                    if index + 1 < len(lower_edges)
                    else None
                )
                if upper is None:
                    masks.append(values >= lower)
                    labels.append(f"{lower}+")
                else:
                    masks.append((values >= lower) & (values < upper))
                    labels.append(
                        str(lower) if upper == lower + 1 else f"{lower}–{upper - 1}"
                    )
            positions = np.arange(len(labels))
            if unit == "subject":
                proportions: List[np.ndarray] = []
                for _, one_subject in row_bouts.groupby("subject", sort=True):
                    subject_values = one_subject[
                        "bout_duration_real_steps"
                    ].to_numpy(dtype=float)
                    subject_counts = []
                    for index, lower in enumerate(lower_edges):
                        upper = (
                            lower_edges[index + 1]
                            if index + 1 < len(lower_edges)
                            else None
                        )
                        if upper is None:
                            mask = subject_values >= lower
                        else:
                            mask = (subject_values >= lower) & (subject_values < upper)
                        subject_counts.append(float(np.sum(mask)))
                    total = float(np.sum(subject_counts))
                    if total > 0:
                        proportions.append(np.asarray(subject_counts) / total * 100.0)
                matrix = np.vstack(proportions)
                mean = np.mean(matrix, axis=0)
                if len(matrix) >= 2:
                    critical = float(stats.t.ppf(0.975, len(matrix) - 1))
                    interval = critical * stats.sem(matrix, axis=0)
                else:
                    interval = np.zeros_like(mean)
                distribution_axis.bar(
                    positions, mean, color=BLUE, edgecolor="white", linewidth=0.55
                )
                distribution_axis.errorbar(
                    positions, mean, yerr=interval, fmt="none",
                    ecolor=INK, elinewidth=0.8, capsize=2.2,
                )
                distribution_axis.set_ylabel("Within-subject bout share (%)")
            else:
                counts = np.asarray([int(np.sum(mask)) for mask in masks])
                distribution_axis.bar(
                    positions, counts, color=BLUE,
                    edgecolor="white", linewidth=0.55,
                )
                if counts.max(initial=0) >= 100:
                    distribution_axis.set_yscale("log")
                distribution_axis.set_ylabel("Completed bouts")
            distribution_axis.set_xticks(
                positions, labels, rotation=32, ha="right"
            )
        else:
            distribution_axis.text(
                0.5, 0.5, "No completed bouts",
                transform=distribution_axis.transAxes,
                ha="center", va="center",
            )
        distribution_axis.set_title("Bout-duration distribution")
        distribution_axis.set_xlabel("Completed duration (real steps)")
        distribution_axis.grid(axis="y", color="#E5E7EB", lw=0.55)
        for spine in ("top", "right"):
            distribution_axis.spines[spine].set_visible(False)

        for column_index, (outcome, title, subtitle) in enumerate(
            ALLOCATION_METRICS, start=1
        ):
            axis = axes[row_index, column_index]
            selected = display_effects.loc[
                display_effects["transition_type"].eq(transition_type)
                & display_effects["outcome"].eq(outcome)
            ].copy()
            if selected.empty:
                axis.text(
                    0.5, 0.5, "Not estimable",
                    transform=axis.transAxes, ha="center", va="center",
                )
                axis.set_axis_off()
                continue
            unit_values = sorted(selected[display_unit].unique())
            if unit == "subject":
                unit_colors = plt.cm.Blues(
                    np.linspace(0.38, 0.82, len(unit_values))
                )
                line_styles: List[object] = [
                    "-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))
                ]
            else:
                unit_colors = ["#8C98A4"] * len(unit_values)
                line_styles = ["-"] * len(unit_values)
            for unit_index, unit_value in enumerate(unit_values):
                one_unit = selected.loc[selected[display_unit].eq(unit_value)].iloc[0]
                low = max(float(one_unit["support_low"]), -3.0)
                high = min(float(one_unit["support_high"]), 3.0)
                if high <= low:
                    continue
                x_unit = np.linspace(low, high, 100)
                axis.plot(
                    x_unit,
                    float(one_unit["beta_log_bout_duration"]) * x_unit,
                    color=unit_colors[unit_index],
                    alpha=0.62 if unit == "subject" else 0.28,
                    lw=1.0 if unit == "subject" else 0.85,
                    ls=line_styles[unit_index % len(line_styles)],
                    label=(
                        f"sub{int(unit_value):03d}"
                        if unit == "subject" and row_index == 0 and column_index == 1
                        else None
                    ),
                )
            common_low = float(selected["support_low"].max())
            common_high = float(selected["support_high"].min())
            display_low = max(common_low, -2.0)
            display_high = min(common_high, 2.0)
            if display_high <= display_low:
                display_low, display_high = common_low, common_high
            if display_high > display_low:
                x = np.linspace(display_low, display_high, 180)
                beta = selected["beta_log_bout_duration"].to_numpy(dtype=float)
                predictions = beta[:, None] * x[None, :]
                mean = np.mean(predictions, axis=0)
                sem = stats.sem(predictions, axis=0) if len(beta) >= 2 else np.zeros_like(mean)
                critical = (
                    float(stats.t.ppf(0.975, len(beta) - 1))
                    if len(beta) >= 2 else 0.0
                )
                interval = critical * sem
                axis.fill_between(
                    x, mean - interval, mean + interval,
                    color=BLUE_LIGHT, alpha=0.58, linewidth=0,
                )
                one = summary.loc[
                    summary["transition_type"].eq(transition_type)
                    & summary["outcome"].eq(outcome)
                ]
                q_value = float(one.iloc[0]["bh_fdr_q"]) if not one.empty else np.nan
                significant = np.isfinite(q_value) and q_value < 0.05
                axis.plot(
                    x, mean,
                    color=BLUE if significant else INK,
                    lw=2.3,
                    ls="-" if significant else "--",
                    label=(
                        f"equal-{unit_label} mean"
                        if unit == "subject" and row_index == 0 and column_index == 1
                        else None
                    ),
                )
            axis.axhline(0, color=INK, lw=0.9, ls="--")
            axis.axvline(0, color="#B8BEC5", lw=0.7, ls=":")
            axis.set_title(f"{title}\n{subtitle}")
            axis.set_xlabel(
                r"Within-run standardized log duration  $z_{run}(\log D)$"
            )
            if column_index == 1:
                axis.set_ylabel(
                    "Duration-attributable endpoint change\n(pooled endpoint SD)"
                )
            one = summary.loc[
                summary["transition_type"].eq(transition_type)
                & summary["outcome"].eq(outcome)
            ]
            if not one.empty:
                row = one.iloc[0]
                q_value = float(row["bh_fdr_q"])
                q_label = "FDR n.s." if q_value >= 0.05 else "FDR significant"
                axis.text(
                    0.02, 0.98,
                    f"mean β={row['mean_unit_effect']:+.3f}\n"
                    f"95% CI [{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]\n"
                    f"{int(row['n_positive'])}/{int(row['n_units'])} positive {unit_label}s\n"
                    f"p={row['exact_sign_flip_p']:.3f}, q={q_value:.3f} ({q_label})",
                    transform=axis.transAxes, va="top", fontsize=8.1,
                )
            axis.grid(color="#E5E7EB", lw=0.55)
            for spine in ("top", "right"):
                axis.spines[spine].set_visible(False)
            if unit == "subject" and row_index == 0 and column_index == 1:
                axis.legend(
                    loc="lower left", fontsize=7.2, ncol=2,
                    frameon=False, handlelength=2.3,
                )
    fig.suptitle(
        f"Completed-bout duration and endpoint search geometry · {title_suffix}",
        fontsize=16, y=0.995,
    )
    fig.text(0.048, 0.905, row_subtitles["noop_to_active"], fontsize=11.5, weight="bold")
    fig.text(0.048, 0.465, row_subtitles["active_to_noop"], fontsize=11.5, weight="bold")
    fig.text(
        0.5, 0.018,
        f"Thin lines are {unit_label}-specific fitted duration contributions over observed support. The bold line and 95% t interval are equal-{unit_label} summaries restricted to all-{unit_label} common support; a dashed mean denotes FDR q ≥ .05. Endpoint change is terminal minus the first comparable window. Controls: start transition margin, start Q_signed, prior reward, origin action, and destination action.",
        ha="center", fontsize=8.7,
    )
    fig.subplots_adjust(
        left=0.055, right=0.99, bottom=0.105, top=0.845,
        hspace=0.70, wspace=0.28,
    )
    save_figure(fig, output_path)


def write_subject_outputs(
    frame: pd.DataFrame,
    qa: Mapping[str, object],
    subject: int,
    game: int,
    output_root: Path,
) -> None:
    output_dir = output_root / f"sub{subject:03d}_game{game}"
    result_dir = output_dir / "results"
    figure_dir = output_dir / "figures" / "main"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    retained = [
        "subject",
        "session",
        "block",
        "game",
        *KEYS,
        "incumbent_action",
        "human_action",
        "switch",
        "transition_class",
        "bout_age",
        "prior_reward",
        "prior_done",
        "run_progress",
        "model_current_action_margin",
        "model_current_vs_noop_margin",
        "model_alternative_uncertainty",
        "value_revision_magnitude",
        "value_revision_log",
        "value_revision_robust_z",
        "value_revision_signed_mean",
        "value_revision_signed_robust_z",
        "revision_event_count",
        "revision_event_fraction",
        "directional_revision_balance",
        "actor_current_action_margin_start",
        "actor_current_action_margin_terminal",
        "actor_current_action_margin_shift",
        "actor_current_action_uncertainty_start",
        "actor_current_action_uncertainty_terminal",
        *[metric for metric, _, _ in ALLOCATION_METRICS],
    ]
    frame[retained].to_csv(
        result_dir / "evaluation_revision_window_metrics.csv.gz",
        index=False,
        compression="gzip",
    )
    bout_rows, bout_table = build_same_action_bouts(frame)
    endpoints = bout_endpoints(bout_rows)
    completed_bout_allocation = build_completed_bout_allocation(bout_rows)
    bout_table.to_csv(result_dir / "same_action_bouts.csv", index=False)
    endpoints.to_csv(result_dir / "same_action_bout_endpoints.csv.gz", index=False, compression="gzip")
    completed_bout_allocation.to_csv(
        result_dir / "completed_known_onset_bout_endpoint_changes.csv.gz",
        index=False,
        compression="gzip",
    )
    effects = fit_run_effects(frame)
    effects.to_csv(result_dir / "run_effects.csv", index=False)
    allocation_effects = fit_allocation_run_effects(frame)
    allocation_run_summary = summarize_allocation_run_effects(allocation_effects)
    allocation_effects.to_csv(
        result_dir / "allocation_marginal_run_effects.csv", index=False
    )
    allocation_run_summary.to_csv(
        result_dir / "allocation_marginal_run_summary.csv", index=False
    )
    allocation_scatter_points = build_allocation_scatter_points(
        frame, unit="run_label"
    )
    allocation_scatter_trends = summarize_allocation_scatter_trends(
        allocation_scatter_points
    )
    allocation_scatter_points.to_csv(
        result_dir / "allocation_scatter_run_points.csv", index=False
    )
    allocation_scatter_trends.to_csv(
        result_dir / "allocation_scatter_run_trends.csv", index=False
    )
    duration_effects, duration_curves = fit_duration_run_effects(
        completed_bout_allocation
    )
    duration_run_summary = summarize_duration_units(
        duration_effects, "run_label"
    )
    duration_effects.to_csv(
        result_dir / "bout_duration_endpoint_change_run_effects.csv", index=False
    )
    duration_run_summary.to_csv(
        result_dir / "bout_duration_endpoint_change_run_summary.csv", index=False
    )
    duration_curves.to_csv(
        result_dir / "bout_duration_endpoint_change_adjusted_curves.csv.gz",
        index=False,
        compression="gzip",
    )
    transition_risk = build_transition_risk_sets(frame)
    transition_effects = fit_transition_run_effects(transition_risk)
    transition_risk.to_csv(
        result_dir / "transition_specific_risk_windows.csv.gz",
        index=False, compression="gzip",
    )
    transition_effects.to_csv(
        result_dir / "transition_specific_run_effects.csv", index=False
    )

    suffix = f"sub{subject:03d}, game {game}"
    plot_subject_geometry(
        frame,
        figure_dir / "main_figure1_evaluation_revision_geometry.png",
        subject,
        game,
    )
    if not endpoints.empty:
        plot_transition_trajectories(
            endpoints,
            figure_dir / "main_figure2_same_action_bout_trajectories.png",
            [frame],
            suffix,
            include_active_to_active=False,
        )
    plot_effect_tiles(
        effects,
        figure_dir / "main_figure3_run_effect_consistency.png",
        suffix,
    )
    plot_actor_shift_pairs(
        frame,
        figure_dir / "main_figure4_actor_margin_shift_stay_switch.png",
        suffix,
    )
    plot_transition_specific_subject(
        transition_risk,
        transition_effects,
        figure_dir / "main_figure5_transition_specific_geometry.png",
        subject,
        game,
    )
    if not allocation_effects.empty:
        plot_allocation_within_run_consistency(
            frame,
            allocation_effects,
            allocation_run_summary,
            figure_dir / "main_figure6_evaluation_revision_search_allocation.png",
            suffix,
            unit="run_label",
        )
    if not duration_effects.empty and not duration_run_summary.empty:
        plot_completed_bout_duration(
            completed_bout_allocation,
            duration_effects,
            duration_run_summary,
            figure_dir / "main_figure7_completed_bout_deep_narrow_allocation.png",
            suffix,
            unit="run_label",
        )
    qa = dict(qa)
    qa["transition_specific"] = {
        transition: {
            "n_risk_windows": int(len(data)),
            "n_events": int(data["transition_event"].sum()),
            "event_rate": float(data["transition_event"].mean()),
        }
        for transition, data in transition_risk.groupby(
            "transition_type", sort=False
        )
    }
    qa["transition_specific"]["active_to_active_competing_events_excluded"] = int(
        frame["transition_class"].eq("active_to_active").sum()
    )
    qa["search_allocation"] = {
        "population": (
            "all eligible windows split by source-state risk: current NOOP for "
            "NOOP-to-active and current active for active-to-NOOP"
        ),
        "n_run_outcome_models": int(len(allocation_effects)),
        "n_figure6_marginal_tests": int(len(allocation_run_summary)),
        "n_figure6_scatter_points": int(len(allocation_scatter_points)),
        "n_figure6_scatter_trends": int(len(allocation_scatter_trends)),
        "n_completed_known_onset_bouts": int(len(completed_bout_allocation)),
        "n_duration_run_outcome_models": int(len(duration_effects)),
        "n_duration_curve_rows": int(len(duration_curves)),
        "figure6_estimand": (
            "display: one run-mean point per panel and descriptive across-run "
            "trend; inference retained separately as within-run/transition "
            "standardized univariate allocation~M and allocation~Q_signed slopes"
        ),
        "figure7_estimand": (
            "terminal transition-window minus first comparable-window allocation, "
            "scaled by the within-run pooled start/end SD; duration beta and "
            "curve use the same adjusted OLS"
        ),
        "figure7_curve_support": (
            "each run/outcome curve is restricted to that fitted sample's "
            "observed valid-duration minimum and maximum"
        ),
        "duration_selection_note": (
            "Bout-duration analysis excludes left-censored and incomplete bouts; "
            "its regression target is conditional on completed, known-onset "
            "bouts and is fit separately for NOOP-to-active and active-to-NOOP."
        ),
    }
    atomic_json(result_dir / "qa_summary.json", qa)
    manifest = {
        "script": str(Path(__file__).resolve()),
        "schema_version": SCHEMA_VERSION,
        "figure_rendering": {
            "surface_estimator": (
                "34x34 binned count/mean estimator; Gouraud interpolation is "
                "display-only and never fills unsupported cells"
            ),
            "signed_colormap": "negative=blue, zero=white, positive=red",
            "stay_switch_violin": (
                "run-mean distributions with mean 95% t CI and paired exact two-sided run sign-flip"
            ),
            "figure6_scatter": (
                "one point per run; x/y are same-risk-window run means z-scored "
                "across runs; line is a descriptive across-run OLS trend"
            ),
        },
        "subject": int(subject),
        "game": int(game),
        "primary_geometry": {
            "x": (
                "Pre-search action margin (M): model-policy current-action "
                "minus strongest-competitor log-probability margin"
            ),
            "y": (
                "Q revision magnitude (Q_rev): within-search mean absolute "
                "root-Q revision, natural log then within-run robust z for display"
            ),
            "outcomes": [
                "cross-head terminal-real minus initial-imaginary Actor margin contrast (diagnostic)",
                "human stay/switch",
            ],
        },
        "temporal_note": (
            "root_q_after_sampled is the sampled branch root-Q immediately "
            "after the same status-2 expansion, not at the next real step"
        ),
        "causal_note": (
            "The axes are temporally ordered observational measurements. The "
            "terminal-minus-initial Actor contrast spans real and imaginary heads; "
            "it is diagnostic and is neither a within-head update nor online learning."
        ),
        "search_allocation": {
            "outcomes": [metric for metric, _, _ in ALLOCATION_METRICS],
            "planning_budget": "observed n_status2 per real-step window",
            "quantity_interpretation": (
                "outcomes describe allocation geometry under the observed variable budget, "
                "not the amount of computation"
            ),
            "window_population": (
                "all eligible windows split by current source action; rows are "
                "risk strata and are not selected by the future transition"
            ),
            "model_margin_timing": (
                "pre-search; NOOP versus strongest active in the NOOP-to-active "
                "row and current active versus NOOP in the active-to-NOOP row"
            ),
            "revision_allocation_timing": "same search window; associative, not causal",
            "duration_population": (
                "completed, known-onset bouts separated into NOOP-to-active "
                "and active-to-NOOP"
            ),
            "inference_unit": "run coefficients; runs are equalized within subject for group inference",
            "figure6_display": (
                "one run-mean point per panel; x and y standardized across the "
                "11 displayed runs; descriptive across-run OLS trend"
            ),
            "figure6_inference_model": (
                "separate marginal standardized Y~transition margin and "
                "Y~Q_signed slopes; Q_signed=mean_j(delta Q_j) over valid "
                "status-2 events; no action control, mutual adjustment, "
                "interaction, or covariates"
            ),
            "figure7_model": (
                "endpoint change ~ z_run(log duration) + start transition margin + "
                "start Q_signed + prior reward + reference-coded current/origin "
                "and destination actions"
            ),
            "figure7_curve_support": (
                "each run/outcome is drawn only within its observed fitted-sample "
                "duration min-max; axes are transition-row specific"
            ),
        },
        "generated_allocation_tables": [
            "allocation_marginal_run_effects.csv",
            "allocation_marginal_run_summary.csv",
            "allocation_scatter_run_points.csv",
            "allocation_scatter_run_trends.csv",
            "completed_known_onset_bout_endpoint_changes.csv.gz",
            "bout_duration_endpoint_change_run_effects.csv",
            "bout_duration_endpoint_change_run_summary.csv",
            "bout_duration_endpoint_change_adjusted_curves.csv.gz",
        ],
        "generated_main_figures": [
            "main_figure1_evaluation_revision_geometry.png",
            *(
                ["main_figure2_same_action_bout_trajectories.png"]
                if not endpoints.empty
                else []
            ),
            "main_figure3_run_effect_consistency.png",
            "main_figure4_actor_margin_shift_stay_switch.png",
            "main_figure5_transition_specific_geometry.png",
            *(
                ["main_figure6_evaluation_revision_search_allocation.png"]
                if not allocation_effects.empty
                else []
            ),
            *(
                ["main_figure7_completed_bout_deep_narrow_allocation.png"]
                if not duration_curves.empty and not duration_run_summary.empty
                else []
            ),
        ],
        "qa": dict(qa),
    }
    atomic_json(output_dir / "analysis_manifest.json", manifest)
    log(f"Subject analysis complete -> {output_dir}")


def subject_equalized_surface(
    frames: Sequence[pd.DataFrame],
    value: str,
    x_limits: Tuple[float, float],
    y_limits: Tuple[float, float],
    *,
    kind: str,
    bins: int = 34,
    min_subjects: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    surfaces = []
    counts = []
    for frame in frames:
        selected = action_class(frame, kind)
        surface, count, x_edges, y_edges = binned_surface(
            selected,
            value,
            x_limits,
            y_limits,
            bins=bins,
            min_count=12,
            equalize_action=(kind == "active"),
        )
        surfaces.append(surface)
        counts.append(count)
    stack = np.stack(surfaces)
    support = np.sum(np.isfinite(stack), axis=0)
    summed = np.nansum(stack, axis=0)
    mean = np.full_like(summed, np.nan, dtype=float)
    np.divide(summed, support, out=mean, where=support >= min_subjects)
    return mean, np.sum(np.stack(counts), axis=0), x_edges, y_edges


def draw_group_surface(
    axis: plt.Axes,
    frames: Sequence[pd.DataFrame],
    value: str,
    title: str,
    x_limits: Tuple[float, float],
    y_limits: Tuple[float, float],
    *,
    kind: str,
    cmap: str,
    center: Optional[float] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorbar_label: str,
    show_subject_centroids: bool = True,
    label_subjects: bool = False,
) -> None:
    surface, _, x_edges, y_edges = subject_equalized_surface(
        frames, value, x_limits, y_limits, kind=kind
    )
    norm = None
    if center is not None and np.isfinite(surface).any():
        bound = max(
            abs(float(np.nanquantile(surface, 0.03))),
            abs(float(np.nanquantile(surface, 0.97))),
            EPS,
        )
        norm = TwoSlopeNorm(vmin=-bound, vcenter=center, vmax=bound)
    image = draw_continuous_surface(
        axis,
        surface,
        x_edges,
        y_edges,
        cmap=cmap,
        norm=norm,
        vmin=None if norm is not None else vmin,
        vmax=None if norm is not None else vmax,
    )
    if show_subject_centroids:
        markers = ("o", "s", "^", "D", "P", "X")
        palette = plt.get_cmap("tab10")
        for index, frame in enumerate(frames):
            selected = action_class(frame, kind)
            coordinates = selected[
                ["model_current_action_margin", "value_revision_robust_z"]
            ].replace([np.inf, -np.inf], np.nan).dropna()
            if coordinates.empty:
                continue
            subject = int(pd.to_numeric(frame["subject"], errors="coerce").dropna().iloc[0])
            axis.scatter(
                float(coordinates["model_current_action_margin"].median()),
                float(coordinates["value_revision_robust_z"].median()),
                s=46,
                marker=markers[index % len(markers)],
                facecolor="white",
                edgecolor=palette(index % 10),
                linewidth=1.4,
                zorder=6,
                label=f"sub{subject:03d}" if label_subjects else None,
            )
    axis.axvline(0, color=INK, ls="--", lw=0.9)
    axis.axhline(0, color=INK, ls=":", lw=0.8, alpha=0.65)
    axis.set_xlim(x_limits)
    axis.set_ylim(y_limits)
    axis.set_title(title)
    axis.set_xlabel(PRE_SEARCH_MARGIN_LABEL)
    axis.set_ylabel(Q_REVISION_LABEL)
    axis.set_box_aspect(1)
    axis.figure.colorbar(
        image,
        ax=axis,
        fraction=0.047,
        pad=0.035,
        label=colorbar_label,
    )
    if label_subjects:
        axis.legend(
            loc="upper left",
            ncol=2,
            fontsize=7.5,
            title="Subject median M–Q_rev location",
            title_fontsize=7.5,
        )


def plot_group_geometry(
    frames: Sequence[pd.DataFrame], output_path: Path, game: int
) -> None:
    x_limits, y_limits = shared_limits(frames)
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 11.0), facecolor="white")
    draw_group_surface(
        axes[0, 0], frames, "switch", "Switch probability · current NOOP", x_limits, y_limits,
        kind="noop", cmap="magma", vmin=0.0, vmax=0.55,
        colorbar_label="Subject-equalized P(switch)", label_subjects=True,
    )
    draw_group_surface(
        axes[0, 1], frames, "switch", "Switch probability · current active action", x_limits, y_limits,
        kind="active", cmap="magma", vmin=0.0, vmax=0.55, colorbar_label="Subject/action-equalized P(switch)"
    )
    draw_group_surface(
        axes[1, 0], frames, "actor_current_action_margin_shift", "Actor margin shift · current NOOP", x_limits, y_limits,
        kind="noop", cmap=SIGNED_CMAP, center=0.0, colorbar_label="Subject-equalized terminal − initial margin"
    )
    draw_group_surface(
        axes[1, 1], frames, "actor_current_action_margin_shift", "Actor margin shift · current active action", x_limits, y_limits,
        kind="active", cmap=SIGNED_CMAP, center=0.0, colorbar_label="Subject/action-equalized terminal − initial margin"
    )
    fig.suptitle(
        f"Cross-subject M–Q_rev geometry · game {game}",
        fontsize=17,
        x=0.5,
        y=0.995,
    )
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.07, top=0.93, hspace=0.30, wspace=0.28)
    save_figure(fig, output_path)


def summarize_group_effects(effects: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    subject_effects = (
        effects.groupby(["subject", "outcome"], as_index=False)[
            ["beta_M", "beta_V", "beta_MxV"]
        ]
        .mean()
    )
    records: List[Dict[str, object]] = []
    for outcome, group in subject_effects.groupby("outcome", sort=False):
        for metric in ["beta_M", "beta_V", "beta_MxV"]:
            mean, p = exact_sign_flip(group[metric])
            records.append(
                {
                    "outcome": outcome,
                    "metric": metric,
                    "mean_subject_effect": mean,
                    "exact_sign_flip_p": p,
                    "n_subjects": int(group[metric].notna().sum()),
                    "n_positive": int((group[metric] > 0).sum()),
                    "n_negative": int((group[metric] < 0).sum()),
                }
            )
    group_effects = pd.DataFrame(records)
    group_effects["bh_fdr_q_within_outcome"] = np.nan
    for outcome, indices in group_effects.groupby("outcome").groups.items():
        group_effects.loc[indices, "bh_fdr_q_within_outcome"] = bh_fdr(
            group_effects.loc[indices, "exact_sign_flip_p"]
        )
    return subject_effects, group_effects


def plot_group_effect_tiles(
    subject_effects: pd.DataFrame,
    group_effects: pd.DataFrame,
    output_path: Path,
    game: int,
) -> None:
    columns = ["beta_M", "beta_V", "beta_MxV"]
    base_labels = [
        "Pre-search action margin (M)",
        "Q revision magnitude (Q_rev)",
        "M × Q_rev",
    ]
    outcomes = [("human_switch", "Human switch"), ("actor_margin_shift", "Actor margin shift")]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), facecolor="white")
    for axis, (outcome, title) in zip(axes, outcomes):
        data = subject_effects.loc[subject_effects["outcome"].eq(outcome)].sort_values("subject")
        matrix = data[columns].to_numpy(dtype=float)
        bound = (
            max(float(np.nanquantile(np.abs(matrix), 0.98)), EPS)
            if np.isfinite(matrix).any()
            else 1.0
        )
        image = axis.imshow(matrix, aspect="auto", cmap=SIGNED_CMAP, norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound))
        labels = []
        for metric, label in zip(columns, base_labels):
            row = group_effects.loc[(group_effects["outcome"].eq(outcome)) & (group_effects["metric"].eq(metric))]
            if row.empty:
                labels.append(label)
            else:
                labels.append(f"{label}\np={row['exact_sign_flip_p'].iloc[0]:.3f}, q={row['bh_fdr_q_within_outcome'].iloc[0]:.3f}")
        for r in range(matrix.shape[0]):
            for c in range(matrix.shape[1]):
                value = matrix[r, c]
                if np.isfinite(value):
                    axis.text(c, r, f"{value:+.3f}", ha="center", va="center", fontsize=9)
        axis.set_xticks(range(3), labels, rotation=16, ha="right")
        axis.set_yticks(range(len(data)), [f"sub{int(v):03d}" for v in data["subject"]])
        axis.set_title(title)
        axis.grid(False)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.035, label="Across-run mean coefficient")
    fig.suptitle(
        f"Cross-subject effect consistency · game {game}",
        fontsize=16,
        x=0.5,
        y=0.995,
    )
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.23, top=0.86, wspace=0.30)
    save_figure(fig, output_path)


def aggregate_subjects(
    subjects: Sequence[int], game: int, output_root: Path
) -> None:
    frames: List[pd.DataFrame] = []
    effects: List[pd.DataFrame] = []
    endpoints: List[pd.DataFrame] = []
    transition_risks: List[pd.DataFrame] = []
    transition_effects: List[pd.DataFrame] = []
    allocation_effects: List[pd.DataFrame] = []
    completed_bouts: List[pd.DataFrame] = []
    duration_effects: List[pd.DataFrame] = []
    for subject in subjects:
        result_dir = output_root / f"sub{subject:03d}_game{game}" / "results"
        metric_path = result_dir / "evaluation_revision_window_metrics.csv.gz"
        effect_path = result_dir / "run_effects.csv"
        endpoint_path = result_dir / "same_action_bout_endpoints.csv.gz"
        transition_risk_path = result_dir / "transition_specific_risk_windows.csv.gz"
        transition_effect_path = result_dir / "transition_specific_run_effects.csv"
        allocation_effect_path = result_dir / "allocation_marginal_run_effects.csv"
        completed_bout_path = (
            result_dir / "completed_known_onset_bout_endpoint_changes.csv.gz"
        )
        duration_effect_path = (
            result_dir / "bout_duration_endpoint_change_run_effects.csv"
        )
        if not metric_path.exists() or not effect_path.exists():
            raise FileNotFoundError(
                f"Subject output missing for sub{subject:03d}, game {game}; run the subject analysis first"
            )
        frames.append(pd.read_csv(metric_path, low_memory=False))
        effects.append(pd.read_csv(effect_path, low_memory=False))
        if not transition_risk_path.exists() or not transition_effect_path.exists():
            raise FileNotFoundError(
                f"Transition-specific output missing for sub{subject:03d}, game {game}"
            )
        transition_risks.append(pd.read_csv(transition_risk_path, low_memory=False))
        transition_effects.append(pd.read_csv(transition_effect_path, low_memory=False))
        if not (
            allocation_effect_path.exists()
            and completed_bout_path.exists()
            and duration_effect_path.exists()
        ):
            raise FileNotFoundError(
                f"Allocation output missing for sub{subject:03d}, game {game}; "
                "rerun the subject analysis"
            )
        allocation_effects.append(pd.read_csv(allocation_effect_path, low_memory=False))
        completed_bouts.append(pd.read_csv(completed_bout_path, low_memory=False))
        duration_effects.append(pd.read_csv(duration_effect_path, low_memory=False))
        if endpoint_path.exists():
            endpoints.append(pd.read_csv(endpoint_path, low_memory=False))
    group_dir = output_root / f"group_game{game}"
    result_dir = group_dir / "results"
    figure_dir = group_dir / "figures" / "main"
    for obsolete_name in (
        "main_figure2_cross_subject_effect_consistency.png",
        "main_figure3_cross_subject_actor_margin_shift.png",
        "main_figure4_cross_subject_bout_trajectories.png",
    ):
        (figure_dir / obsolete_name).unlink(missing_ok=True)

    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    all_effects = pd.concat(effects, ignore_index=True)
    subject_effects, group_effects = summarize_group_effects(all_effects)
    all_effects.to_csv(result_dir / "run_effects.csv", index=False)
    subject_effects.to_csv(result_dir / "subject_effects.csv", index=False)
    group_effects.to_csv(result_dir / "group_effects.csv", index=False)
    plot_group_geometry(
        frames,
        figure_dir / "main_figure1_cross_subject_evaluation_revision_geometry.png",
        game,
    )
    plot_group_effect_tiles(
        subject_effects,
        group_effects,
        figure_dir / "main_figure3_cross_subject_effect_consistency.png",
        game,
    )
    combined = pd.concat(frames, ignore_index=True)
    plot_actor_shift_pairs(
        combined,
        figure_dir / "main_figure4_cross_subject_actor_margin_shift.png",
        f"game {game}",
        unit="subject",
    )
    if endpoints:
        all_endpoints = pd.concat(endpoints, ignore_index=True)
        all_endpoints.to_csv(result_dir / "same_action_bout_endpoints.csv.gz", index=False, compression="gzip")
        plot_transition_trajectories(
            all_endpoints,
            figure_dir / "main_figure2_cross_subject_bout_trajectories.png",
            frames,
            f"game {game}, {len(subjects)} subjects",
        )
    all_transition_effects = pd.concat(transition_effects, ignore_index=True)
    transition_subject_effects, transition_group_summary = (
        summarize_transition_group_effects(all_transition_effects)
    )
    all_transition_effects.to_csv(
        result_dir / "transition_specific_run_effects.csv", index=False
    )
    transition_subject_effects.to_csv(
        result_dir / "transition_specific_subject_effects.csv", index=False
    )
    transition_group_summary.to_csv(
        result_dir / "transition_specific_group_effects.csv", index=False
    )
    plot_transition_specific_group(
        transition_risks,
        transition_subject_effects,
        transition_group_summary,
        figure_dir / "main_figure5_cross_subject_transition_specific_geometry.png",
        game,
    )
    all_allocation_effects = pd.concat(allocation_effects, ignore_index=True)
    allocation_subject_effects, allocation_group_summary = (
        summarize_allocation_group_effects(all_allocation_effects)
    )
    all_allocation_effects.to_csv(
        result_dir / "allocation_marginal_run_effects.csv", index=False
    )
    allocation_subject_effects.to_csv(
        result_dir / "allocation_marginal_subject_effects.csv", index=False
    )
    allocation_group_summary.to_csv(
        result_dir / "allocation_marginal_group_effects.csv", index=False
    )
    allocation_scatter_points = build_allocation_scatter_points(
        combined, unit="subject"
    )
    allocation_scatter_trends = summarize_allocation_scatter_trends(
        allocation_scatter_points
    )
    allocation_scatter_points.to_csv(
        result_dir / "allocation_scatter_subject_points.csv", index=False
    )
    allocation_scatter_trends.to_csv(
        result_dir / "allocation_scatter_subject_trends.csv", index=False
    )
    plot_allocation_within_run_consistency(
        combined,
        all_allocation_effects,
        allocation_group_summary,
        figure_dir / "main_figure6_cross_subject_evaluation_revision_search_allocation.png",
        f"game {game}, {len(subjects)} subjects",
        unit="subject",
    )
    all_completed_bouts = pd.concat(completed_bouts, ignore_index=True)
    all_duration_effects = pd.concat(duration_effects, ignore_index=True)
    duration_subject_effects, duration_group_summary = summarize_duration_group_effects(
        all_duration_effects
    )
    all_duration_curves = build_duration_component_curves(all_duration_effects)
    all_completed_bouts.to_csv(
        result_dir / "completed_known_onset_bout_endpoint_changes.csv.gz",
        index=False,
        compression="gzip",
    )
    all_duration_effects.to_csv(
        result_dir / "bout_duration_endpoint_change_run_effects.csv", index=False
    )
    duration_subject_effects.to_csv(
        result_dir / "bout_duration_endpoint_change_subject_effects.csv", index=False
    )
    duration_group_summary.to_csv(
        result_dir / "bout_duration_endpoint_change_group_effects.csv", index=False
    )
    all_duration_curves.to_csv(
        result_dir / "bout_duration_endpoint_change_adjusted_curves.csv.gz",
        index=False,
        compression="gzip",
    )
    plot_completed_bout_duration(
        all_completed_bouts,
        all_duration_effects,
        duration_group_summary,
        figure_dir / "main_figure7_cross_subject_completed_bout_deep_narrow_allocation.png",
        f"game {game}, {len(subjects)} subjects",
        unit="subject",
    )
    manifest = {
        "script": str(Path(__file__).resolve()),
        "schema_version": SCHEMA_VERSION,
        "figure_rendering": {
            "surface_estimator": (
                "subject-equalized 34x34 binned means; Gouraud interpolation is display-only"
            ),
            "subject_overlay": "distinct markers show each subject's median M-Q_rev location",
            "signed_colormap": "negative=blue, zero=white, positive=red",
            "stay_switch_violin": (
                "per-subject run-mean violins and 95% t CI; subject stars use paired run sign-flips; "
                "panel title uses paired exact two-sided sign-flip across six subject means"
            ),
            "figure6_scatter": (
                "one point per subject; windows are averaged within run and runs "
                "equally averaged within subject before across-subject z scoring; "
                "line is a descriptive six-subject OLS trend"
            ),
        },
        "game": int(game),
        "subjects": [int(value) for value in subjects],
        "inference": (
            "run coefficients averaged within subject; exact two-sided subject sign-flip; "
            "BH-FDR across M and Q_rev jointly over the two transitions; "
            "for allocation, BH-FDR jointly over the 12 marginal transition-margin/Q_signed tests; "
            "duration is tested across two completed "
            "transition classes by three allocation outcomes (6 tests)"
        ),
        "search_allocation": {
            "outcomes": [metric for metric, _, _ in ALLOCATION_METRICS],
            "planning_budget": "observed n_status2 per real-step window",
            "quantity_interpretation": (
                "outcomes describe allocation geometry under the observed variable budget, "
                "not the amount of computation"
            ),
            "window_population": (
                "all eligible windows split by current source action; not selected "
                "by the future transition"
            ),
            "model_margin": (
                "NOOP versus strongest active for NOOP-to-active; current active "
                "versus NOOP for active-to-NOOP"
            ),
            "duration_population": (
                "completed, known-onset bouts separated into NOOP-to-active "
                "and active-to-NOOP"
            ),
            "same_window_caveat": (
                "Q_signed and allocation are contemporaneous associations; "
                "Q_signed averages updates across sampled branch identities"
            ),
            "figure6_display": (
                "one subject point per panel after equal-run subject averaging; "
                "descriptive across-subject OLS trend"
            ),
            "figure6_inference_model": (
                "separate marginal standardized Y~transition margin and "
                "Y~Q_signed slopes; Q_signed=mean_j(delta Q_j) over valid "
                "status-2 events; no action control, mutual adjustment, "
                "interaction, or covariates"
            ),
            "figure7_model": (
                "terminal minus first comparable allocation endpoint change, "
                "pooled-start/end-SD scaled, regressed on z_run(log duration) "
                "with start transition margin, start Q_signed, prior reward, "
                "current/origin action, and destination action controls"
            ),
            "figure7_curve_support": (
                "each run/outcome is evaluated only within its own observed "
                "fitted-sample duration min-max; axes are transition-row specific"
            ),
        },
        "generated_allocation_tables": [
            "allocation_marginal_run_effects.csv",
            "allocation_marginal_subject_effects.csv",
            "allocation_marginal_group_effects.csv",
            "allocation_scatter_subject_points.csv",
            "allocation_scatter_subject_trends.csv",
            "completed_known_onset_bout_endpoint_changes.csv.gz",
            "bout_duration_endpoint_change_run_effects.csv",
            "bout_duration_endpoint_change_subject_effects.csv",
            "bout_duration_endpoint_change_group_effects.csv",
            "bout_duration_endpoint_change_adjusted_curves.csv.gz",
        ],
        "generated_main_figures": [
            "main_figure1_cross_subject_evaluation_revision_geometry.png",
            *(
                ["main_figure2_cross_subject_bout_trajectories.png"]
                if endpoints
                else []
            ),
            "main_figure3_cross_subject_effect_consistency.png",
            "main_figure4_cross_subject_actor_margin_shift.png",
            "main_figure5_cross_subject_transition_specific_geometry.png",
            "main_figure6_cross_subject_evaluation_revision_search_allocation.png",
            "main_figure7_cross_subject_completed_bout_deep_narrow_allocation.png",
        ],
        "status": "development-sample analysis; lock definitions before new-sample confirmation",
    }
    atomic_json(group_dir / "analysis_manifest.json", manifest)
    log(f"Group analysis complete -> {group_dir}")


def write_figure6_render_manifest(
    output_dir: Path,
    *,
    unit: str,
    input_paths: Sequence[Path],
    figure_path: Path,
    subjects: Sequence[int],
    game: int,
) -> None:
    """Record byte-level provenance for an image-only Figure-6 refresh."""
    payload = {
        "figure6_render_version": FIGURE6_RENDER_VERSION,
        "analysis_schema_version": SCHEMA_VERSION,
        "rendered_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "unit": unit,
        "subjects": [int(value) for value in subjects],
        "game": int(game),
        "plot_contract": {
            "panels": "2 source-state risks x 3 allocation outcomes",
            "predictors_overlaid": {
                "M": "blue solid circles",
                "Q_signed": "orange dashed triangles",
            },
            "points": "equal-unit means of nine within-run quantile bins",
            "thin_lines_and_inset_points": (
                "original marginal within-run coefficients at the displayed unit grain"
            ),
            "thick_lines_and_ribbons": "equal-unit mean and 95% t interval",
            "inference": "existing exact sign-flip and 12-effect BH-FDR; no refit",
        },
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": file_sha256(Path(__file__).resolve()),
        },
        "inputs": [
            {"path": str(path.resolve()), "sha256": file_sha256(path)}
            for path in input_paths
        ],
        "figure": {
            "path": str(figure_path.resolve()),
            "sha256": file_sha256(figure_path),
        },
    }
    sidecar_path = output_dir / "results" / "figure6_render_manifest.json"
    atomic_json(sidecar_path, payload)
    analysis_manifest_path = output_dir / "analysis_manifest.json"
    if analysis_manifest_path.exists():
        analysis_manifest = json.loads(
            analysis_manifest_path.read_text(encoding="utf-8")
        )
        rendering = analysis_manifest.setdefault("figure_rendering", {})
        rendering.pop("figure6_scatter", None)
        rendering["figure6_within_run_consistency"] = (
            "six panels (2 source-state risks x 3 allocation outcomes); M and "
            "Q_signed are overlaid; points are equal-unit means of nine "
            "within-run quantile bins; thin lines/inset points are original "
            "within-run coefficients; thick lines/ribbons are equal-unit "
            "means and 95% t intervals"
        )
        allocation = analysis_manifest.setdefault("search_allocation", {})
        allocation["figure6_display"] = (
            "visualizes the existing within-run marginal slopes; individual "
            "figures use runs as equal display units and group figures use "
            "equal-run subject coefficients as equal display units"
        )
        allocation["figure6_render_sidecar"] = str(sidecar_path.resolve())
        atomic_json(analysis_manifest_path, analysis_manifest)


def rerender_subject_figure6(
    subject: int,
    game: int,
    output_root: Path,
) -> None:
    """Re-render one subject Figure 6 without recomputing any analysis."""
    output_dir = output_root / f"sub{subject:03d}_game{game}"
    result_dir = output_dir / "results"
    figure_path = (
        output_dir
        / "figures"
        / "main"
        / "main_figure6_evaluation_revision_search_allocation.png"
    )
    metric_path = result_dir / "evaluation_revision_window_metrics.csv.gz"
    effect_path = result_dir / "allocation_marginal_run_effects.csv"
    summary_path = result_dir / "allocation_marginal_run_summary.csv"
    input_paths = [metric_path, effect_path, summary_path]
    missing = [str(path) for path in input_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Figure-6 render inputs are missing: {missing}")
    frame = pd.read_csv(metric_path, low_memory=False)
    effects = pd.read_csv(effect_path, low_memory=False)
    summary = pd.read_csv(summary_path, low_memory=False)
    plot_allocation_within_run_consistency(
        frame,
        effects,
        summary,
        figure_path,
        f"sub{subject:03d}, game {game}",
        unit="run_label",
    )
    write_figure6_render_manifest(
        output_dir,
        unit="run_label",
        input_paths=input_paths,
        figure_path=figure_path,
        subjects=[subject],
        game=game,
    )


def rerender_group_figure6(
    subjects: Sequence[int],
    game: int,
    output_root: Path,
) -> None:
    """Re-render a group Figure 6 from frozen subject tables only."""
    frames: List[pd.DataFrame] = []
    effects: List[pd.DataFrame] = []
    input_paths: List[Path] = []
    for subject in subjects:
        result_dir = output_root / f"sub{subject:03d}_game{game}" / "results"
        metric_path = result_dir / "evaluation_revision_window_metrics.csv.gz"
        effect_path = result_dir / "allocation_marginal_run_effects.csv"
        for path in (metric_path, effect_path):
            if not path.exists():
                raise FileNotFoundError(f"Figure-6 render input is missing: {path}")
            input_paths.append(path)
        frames.append(pd.read_csv(metric_path, low_memory=False))
        effects.append(pd.read_csv(effect_path, low_memory=False))
    group_dir = output_root / f"group_game{game}"
    group_summary_path = group_dir / "results" / "allocation_marginal_group_effects.csv"
    if not group_summary_path.exists():
        raise FileNotFoundError(
            f"Figure-6 group summary is missing: {group_summary_path}"
        )
    input_paths.append(group_summary_path)
    summary = pd.read_csv(group_summary_path, low_memory=False)
    figure_path = (
        group_dir
        / "figures"
        / "main"
        / "main_figure6_cross_subject_evaluation_revision_search_allocation.png"
    )
    plot_allocation_within_run_consistency(
        pd.concat(frames, ignore_index=True),
        pd.concat(effects, ignore_index=True),
        summary,
        figure_path,
        f"game {game}, {len(subjects)} subjects",
        unit="subject",
    )
    write_figure6_render_manifest(
        group_dir,
        unit="subject",
        input_paths=input_paths,
        figure_path=figure_path,
        subjects=subjects,
        game=game,
    )


def write_figure7_render_manifest(
    output_dir: Path,
    *,
    unit: str,
    input_paths: Sequence[Path],
    figure_path: Path,
    subjects: Sequence[int],
    game: int,
) -> None:
    """Record provenance for an image-only Figure-7 refresh."""
    payload = {
        "figure7_render_version": FIGURE7_RENDER_VERSION,
        "analysis_schema_version": SCHEMA_VERSION,
        "rendered_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "unit": unit,
        "subjects": [int(value) for value in subjects],
        "game": int(game),
        "plot_contract": {
            "panels": (
                "2 completed source-to-destination transitions x "
                "(duration distribution + 3 endpoint geometry outcomes)"
            ),
            "x_axis": "within-run standardized log completed-bout duration",
            "thin_lines": f"one fitted slope per {unit}, within observed support",
            "thick_line_and_ribbon": (
                f"equal-{unit} mean and 95% t interval, restricted to the "
                f"all-{unit} common support"
            ),
            "distribution": (
                "raw completed-bout counts"
                if unit == "run_label"
                else "equal-subject mean within-subject bout share and 95% t interval"
            ),
            "inference": (
                "existing exact sign-flip and six-effect BH-FDR; no model refit"
            ),
            "nonsignificant_style": "dashed dark mean when BH-FDR q >= .05",
        },
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": file_sha256(Path(__file__).resolve()),
        },
        "inputs": [
            {"path": str(path.resolve()), "sha256": file_sha256(path)}
            for path in input_paths
        ],
        "figure": {
            "path": str(figure_path.resolve()),
            "sha256": file_sha256(figure_path),
        },
    }
    sidecar_path = output_dir / "results" / "figure7_render_manifest.json"
    atomic_json(sidecar_path, payload)
    analysis_manifest_path = output_dir / "analysis_manifest.json"
    if analysis_manifest_path.exists():
        analysis_manifest = json.loads(
            analysis_manifest_path.read_text(encoding="utf-8")
        )
        rendering = analysis_manifest.setdefault("figure_rendering", {})
        rendering["figure7_completed_bout_duration"] = (
            f"standardized-log duration slopes at the {unit} display grain; "
            f"thin unit fits over observed support; equal-{unit} mean and "
            "95% t interval over common support; existing inference unchanged"
        )
        analysis_manifest.setdefault("completed_bout_duration", {})[
            "figure7_render_sidecar"
        ] = str(sidecar_path.resolve())
        atomic_json(analysis_manifest_path, analysis_manifest)


def rerender_subject_figure7(
    subject: int,
    game: int,
    output_root: Path,
) -> None:
    """Re-render one subject Figure 7 from frozen tables only."""
    output_dir = output_root / f"sub{subject:03d}_game{game}"
    result_dir = output_dir / "results"
    bout_path = result_dir / "completed_known_onset_bout_endpoint_changes.csv.gz"
    effect_path = result_dir / "bout_duration_endpoint_change_run_effects.csv"
    summary_path = result_dir / "bout_duration_endpoint_change_run_summary.csv"
    input_paths = [bout_path, effect_path, summary_path]
    missing = [str(path) for path in input_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Figure-7 render inputs are missing: {missing}")
    bouts = pd.read_csv(bout_path, low_memory=False)
    effects = pd.read_csv(effect_path, low_memory=False)
    summary = pd.read_csv(summary_path, low_memory=False)
    figure_path = (
        output_dir / "figures" / "main"
        / "main_figure7_completed_bout_deep_narrow_allocation.png"
    )
    plot_completed_bout_duration(
        bouts,
        effects,
        summary,
        figure_path,
        f"sub{subject:03d}, game {game}",
        unit="run_label",
    )
    write_figure7_render_manifest(
        output_dir,
        unit="run_label",
        input_paths=input_paths,
        figure_path=figure_path,
        subjects=[subject],
        game=game,
    )


def rerender_group_figure7(
    subjects: Sequence[int],
    game: int,
    output_root: Path,
) -> None:
    """Re-render one group Figure 7 from frozen subject tables only."""
    bouts: List[pd.DataFrame] = []
    effects: List[pd.DataFrame] = []
    input_paths: List[Path] = []
    for subject in subjects:
        result_dir = output_root / f"sub{subject:03d}_game{game}" / "results"
        bout_path = result_dir / "completed_known_onset_bout_endpoint_changes.csv.gz"
        effect_path = result_dir / "bout_duration_endpoint_change_run_effects.csv"
        for path in (bout_path, effect_path):
            if not path.exists():
                raise FileNotFoundError(f"Figure-7 render input is missing: {path}")
            input_paths.append(path)
        bouts.append(pd.read_csv(bout_path, low_memory=False))
        effects.append(pd.read_csv(effect_path, low_memory=False))
    group_dir = output_root / f"group_game{game}"
    summary_path = (
        group_dir / "results" / "bout_duration_endpoint_change_group_effects.csv"
    )
    if not summary_path.exists():
        raise FileNotFoundError(f"Figure-7 group summary is missing: {summary_path}")
    input_paths.append(summary_path)
    summary = pd.read_csv(summary_path, low_memory=False)
    figure_path = (
        group_dir / "figures" / "main"
        / "main_figure7_cross_subject_completed_bout_deep_narrow_allocation.png"
    )
    plot_completed_bout_duration(
        pd.concat(bouts, ignore_index=True),
        pd.concat(effects, ignore_index=True),
        summary,
        figure_path,
        f"game {game}, {len(subjects)} subjects",
        unit="subject",
    )
    write_figure7_render_manifest(
        group_dir,
        unit="subject",
        input_paths=input_paths,
        figure_path=figure_path,
        subjects=subjects,
        game=game,
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--subject", type=parse_subject, default=None)
    parser.add_argument("--game", type=int, choices=[0, 1, 2], required=True)
    parser.add_argument(
        "--aggregate-subjects",
        nargs="+",
        type=parse_subject,
        default=None,
        help="Aggregate already generated subject outputs.",
    )
    parser.add_argument("--shared-cache-root", type=Path, default=DEFAULT_SHARED_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--force-revision-summary",
        action="store_true",
        help="Recompute the small derivative Q-revision summary from event cache.",
    )
    parser.add_argument(
        "--rerender-figure6-only",
        action="store_true",
        help=(
            "Read frozen window/effect tables and refresh only Figure 6; "
            "no analysis model is refit."
        ),
    )
    parser.add_argument(
        "--rerender-figure7-only",
        action="store_true",
        help=(
            "Read frozen completed-bout/effect tables and refresh only Figure 7; "
            "no analysis model is refit."
        ),
    )
    return parser


def main() -> None:
    configure_style()
    args = build_argparser().parse_args()
    if args.subject is None and not args.aggregate_subjects:
        raise ValueError("Provide --subject or --aggregate-subjects")
    if args.rerender_figure6_only:
        if args.subject is not None:
            rerender_subject_figure6(
                args.subject, args.game, args.output_root
            )
        if args.aggregate_subjects:
            rerender_group_figure6(
                args.aggregate_subjects, args.game, args.output_root
            )
        return
    if args.rerender_figure7_only:
        if args.subject is not None:
            rerender_subject_figure7(
                args.subject, args.game, args.output_root
            )
        if args.aggregate_subjects:
            rerender_group_figure7(
                args.aggregate_subjects, args.game, args.output_root
            )
        return
    if args.subject is not None:
        log(f"Loading sub{args.subject:03d}, game {args.game} from shared cache")
        frame, qa = load_subject_metrics(
            args.subject,
            args.game,
            args.shared_cache_root,
            args.output_root,
            force_revision_summary=args.force_revision_summary,
        )
        write_subject_outputs(frame, qa, args.subject, args.game, args.output_root)
    if args.aggregate_subjects:
        aggregate_subjects(args.aggregate_subjects, args.game, args.output_root)


if __name__ == "__main__":
    main()
