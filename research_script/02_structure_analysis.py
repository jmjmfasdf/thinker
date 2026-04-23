#!/usr/bin/env python3
"""
02_structure_analysis.py  –  Section 2: NOOP Bout Structure Analysis

Sub-sections:
  2-1  Bout length distribution
       KM survival + exponential null baseline, Cross2-based Short/Long split,
       onset uncertainty ~ bout length regression.
  2-2  Temporal profiles around onset and commit
       Onset/commit-aligned z-score profiles (Fig 2-2A/B), bout survival by
       pre-uncertainty 2-bin (Fig 2-2C), Short vs. Long bout commit comparison (Fig 2-2D).
  2-3  Sequential dependency and transitional structure
       First-order NOOP/action transition matrix, bout-internal entropy trajectory,
       thinker action stability within bouts.
  2-4  Within-session adaptation
       Episode-level NOOP ratio over session, early/late-half NOOP comparison.

Outputs (all in --out-dir):
  fig_2_1_bout_distribution.png
  fig_2_2_temporal_profiles.png
  fig_2_3_sequential_structure.png
  fig_2_4_session_adaptation.png
  section2_summary.csv
  bout_internal_trajectory.csv
  bout_action_stability.csv
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ─── Font setup ───────────────────────────────────────────────────────────────

_FONT_CANDIDATES = ["Noto Sans CJK KR", "NanumGothic", "Malgun Gothic", "AppleGothic"]
_FONT_PATH_CANDIDATES = [
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
]
for _p in _FONT_PATH_CANDIDATES:
    if _p.exists():
        font_manager.fontManager.addfont(str(_p))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(_p)).get_name()
        break
else:
    _AVAIL = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = next((f for f in _FONT_CANDIDATES if f in _AVAIL), "DejaVu Sans")
plt.rcParams["axes.unicode_minus"] = False

EPS = 1e-12
FRAGMENT_STATUSES = (1, 2)


# ─── Data types ───────────────────────────────────────────────────────────────

@dataclass
class FileMeta:
    subject: str
    session: str
    block: str
    game: str
    chunk: str


# ─── Utility functions ────────────────────────────────────────────────────────

def parse_file_meta(path: Path) -> FileMeta:
    m = re.match(r"(sub\d+)-ses(\d+)-block(\d+)-game(\d+)_(\d+)\.npy$", path.name)
    if m is None:
        return FileMeta("unknown", "unknown", "unknown", "unknown", path.stem)
    return FileMeta(m.group(1), m.group(2), m.group(3), m.group(4), m.group(5))


def load_npy_dict(path: Path) -> Dict[str, np.ndarray]:
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
        d = obj.item()
        if isinstance(d, dict):
            return d
    if hasattr(obj, "files"):
        return {k: obj[k] for k in obj.files}
    raise ValueError(f"Cannot parse file as dict: {path}")


def to_action_ids(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    return arr.astype(int) if arr.ndim == 1 else np.argmax(arr, axis=1).astype(int)


def softmax_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(np.clip(x, -60, 60))
    denom = ex.sum(axis=1, keepdims=True)
    return ex / np.where(denom <= 0, 1.0, denom)


def entropy_rows(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return -np.sum(p * np.log(p + EPS), axis=1)


def top2_gap_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[1] < 2:
        return np.zeros(x.shape[0])
    part = np.partition(x, -2, axis=1)[:, -2:]
    return part[:, 1] - part[:, 0]


def split_contiguous(indices: np.ndarray) -> List[np.ndarray]:
    indices = np.asarray(indices, dtype=int)
    if indices.size == 0:
        return []
    cuts = np.where(np.diff(indices) != 1)[0] + 1
    return [seg for seg in np.split(indices, cuts) if seg.size > 0]


def prev_index_of_status(status_indices: np.ndarray, ref_indices: np.ndarray) -> np.ndarray:
    if len(status_indices) == 0:
        return np.full(len(ref_indices), -1, dtype=int)
    pos = np.searchsorted(status_indices, ref_indices) - 1
    return np.where(pos >= 0, status_indices[pos], -1).astype(int)


def pick_valid_action(a: int, q_ref: np.ndarray) -> int:
    try:
        ai = int(a)
    except Exception:
        ai = -1
    if 0 <= ai < len(q_ref):
        return ai
    q = np.asarray(q_ref, dtype=float)
    q = np.where(np.isfinite(q), q, -np.inf)
    return int(np.argmax(q)) if q.size > 0 and np.isfinite(q.max()) else 0


def _mean_sem(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.nanmean(y, axis=0)
    n = np.sum(np.isfinite(y), axis=0)
    sd = np.nanstd(y, axis=0, ddof=1)
    sem = np.where(n > 0, sd / np.sqrt(np.maximum(n, 1)), np.nan)
    return mu, sem


# ─── Core data pipeline ───────────────────────────────────────────────────────

def build_real_step_table(
    data: Dict[str, np.ndarray],
    file_path: Path,
    k_rewards: Sequence[int] = (5,),
) -> pd.DataFrame:
    tree = data["tree_reps"]
    status = np.asarray(data["status"]).reshape(-1)
    human_action = to_action_ids(np.asarray(data["human_action"]))
    thinker_action = to_action_ids(np.asarray(data["thinker_action"]))
    actor_logits = np.asarray(data["actor_policy"]).reshape(len(status), -1)
    env_return = np.asarray(data["env_return"]).reshape(-1)

    cur_qs = np.asarray(tree["cur_qs_mean"], dtype=float)
    root_qs = np.asarray(tree["root_qs_mean"], dtype=float)
    cur_v = np.asarray(tree["cur_v"], dtype=float).reshape(-1)
    root_v = np.asarray(tree["root_v"], dtype=float).reshape(-1)
    root_policy = np.asarray(tree["root_policy"], dtype=float).reshape(len(status), -1)
    cur_policy = np.asarray(tree["cur_policy"], dtype=float).reshape(len(status), -1)
    rollout_return = np.asarray(tree["rollout_return"], dtype=float).reshape(-1)
    max_rollout_return = np.asarray(tree["max_rollout_return"], dtype=float).reshape(-1)

    if cur_qs.ndim == 1:
        cur_qs = cur_qs[:, None]
    if root_qs.ndim == 1:
        root_qs = root_qs[:, None]

    if "cur_action" in tree:
        ca = np.asarray(tree["cur_action"])
        cur_action_id = (
            np.argmax(ca, axis=1) if ca.ndim == 2 and ca.shape[1] == cur_qs.shape[1]
            else ca.reshape(-1)
        ).astype(int)
    else:
        cur_action_id = np.argmax(cur_qs, axis=1).astype(int)

    t = min(
        len(status), len(human_action), len(thinker_action), len(actor_logits),
        len(env_return), len(cur_qs), len(root_qs), len(cur_v), len(root_v),
        len(root_policy), len(cur_policy), len(rollout_return),
        len(max_rollout_return), len(cur_action_id),
    )
    status = status[:t]; human_action = human_action[:t]; thinker_action = thinker_action[:t]
    actor_logits = actor_logits[:t]; env_return = env_return[:t]
    cur_qs = cur_qs[:t]; root_qs = root_qs[:t]; cur_v = cur_v[:t]; root_v = root_v[:t]
    root_policy = root_policy[:t]; cur_policy = cur_policy[:t]
    rollout_return = rollout_return[:t]; max_rollout_return = max_rollout_return[:t]
    cur_action_id = cur_action_id[:t]

    probs = softmax_rows(actor_logits)
    entropy_actor = entropy_rows(probs)
    q_gap = top2_gap_rows(root_qs)
    rollout_spread = np.abs(max_rollout_return - rollout_return)

    real_idx = np.where(status == 0)[0]
    if len(real_idx) == 0:
        return pd.DataFrame()

    s2_idx = np.where(status == 2)[0]
    prev_s2_all = prev_index_of_status(s2_idx, real_idx)
    prev_s2_all = np.where(prev_s2_all >= 0, prev_s2_all, np.maximum(real_idx - 1, 0)).astype(int)

    meta = parse_file_meta(file_path)
    rows = []
    num_actions = cur_qs.shape[1]

    for i, idx_global in enumerate(real_idx):
        prev_real_global = real_idx[i - 1] if i > 0 else -1
        prev_s2 = int(prev_s2_all[i])
        target_action = int(human_action[idx_global])

        between = np.arange(prev_real_global + 1, idx_global, dtype=int)
        imag_idx = (
            between[np.isin(status[between], FRAGMENT_STATUSES)]
            if between.size > 0
            else np.array([], dtype=int)
        )

        if imag_idx.size > 0:
            if 0 <= target_action < num_actions:
                frags = split_contiguous(imag_idx)
                matched = [f for f in frags if cur_action_id[f[0]] == target_action]
                sel_idx = np.concatenate(matched) if matched else imag_idx
            else:
                sel_idx = imag_idx
        else:
            sel_idx = np.array([prev_s2], dtype=int)

        sel_idx = sel_idx[(sel_idx >= 0) & (sel_idx < t)]
        if sel_idx.size == 0:
            sel_idx = np.array([min(max(idx_global - 1, 0), t - 1)], dtype=int)

        q_prev = np.nanmean(cur_qs[sel_idx], axis=0)
        if np.any(~np.isfinite(q_prev)):
            q_fb = cur_qs[prev_s2] if 0 <= prev_s2 < t else cur_qs[min(idx_global, t - 1)]
            q_prev = np.where(np.isfinite(q_prev), q_prev, q_fb)

        v_prev = float(np.nanmean(cur_v[sel_idx]))
        if not np.isfinite(v_prev):
            v_prev = float(cur_v[prev_s2] if 0 <= prev_s2 < t else cur_v[min(idx_global, t - 1)])

        used_action = pick_valid_action(target_action, q_prev)
        next_real_global = real_idx[i + 1] if i + 1 < len(real_idx) else idx_global
        if next_real_global < root_qs.shape[0] and used_action < root_qs.shape[1]:
            q_ref = root_qs[next_real_global, used_action]
        else:
            q_ref = cur_qs[min(next_real_global, t - 1), used_action]
        if not np.isfinite(q_ref):
            q_ref = cur_qs[min(idx_global, t - 1), used_action]
        vre_abs_q = abs(float(q_prev[used_action]) - float(q_ref))

        rows.append({
            "file": str(file_path),
            "subject": meta.subject, "session": meta.session,
            "block": meta.block, "game": meta.game, "chunk": meta.chunk,
            "real_pos": i, "global_idx": int(idx_global), "prev_s2_idx": int(prev_s2),
            "human_action": int(target_action),
            "thinker_action": int(thinker_action[idx_global]),
            "cur_action_id": int(cur_action_id[idx_global]),
            "is_human_noop": int(target_action == 0),
            "is_thinker_noop": int(thinker_action[idx_global] == 0),
            "env_return": float(env_return[idx_global]) if idx_global < len(env_return) else np.nan,
            "entropy_actor": float(entropy_actor[idx_global]),
            "q_gap": float(q_gap[idx_global]),
            "rollout_spread": float(rollout_spread[idx_global]),
            "entropy_actor_prev_s2": float(entropy_actor[prev_s2]),
            "q_gap_prev_s2": float(q_gap[prev_s2]),
            "vre_abs_q": float(vre_abs_q),
            "root_v": float(root_v[idx_global]),
        })

    df = pd.DataFrame(rows).sort_values(["file", "real_pos"]).reset_index(drop=True)
    df["prev_human_action"] = df.groupby("file")["human_action"].shift(1)
    df["preceded_by_withholding"] = (
        (df["human_action"] != 0) & (df["prev_human_action"] == 0)
    ).astype(int)
    df["is_overt_action"] = (df["human_action"] != 0).astype(int)

    for k in k_rewards:
        out = np.full(len(df), np.nan)
        for _, g in df.groupby("file", sort=False):
            idxs = g.index.to_numpy()
            rewards = g["env_return"].to_numpy(dtype=float)
            for j, idx_df in enumerate(idxs):
                out[idx_df] = np.nansum(rewards[j + 1 : j + 1 + k])
        df[f"k{k}_reward"] = out

    return df


def extract_noop_bouts(df_real: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for file_id, g in df_real.groupby("file", sort=False):
        g = g.sort_values("real_pos").reset_index(drop=True)
        noop_pos = np.where(g["is_human_noop"].to_numpy(dtype=int) == 1)[0]
        for b_ix, seg in enumerate(split_contiguous(noop_pos)):
            s, e = int(seg[0]), int(seg[-1])
            pre, commit = s - 1, e + 1
            if pre < 0 or commit >= len(g):
                continue
            if int(g.loc[commit, "human_action"]) == 0:
                continue
            rows.append({
                "event_id": f"{Path(file_id).stem}::b{b_ix:04d}",
                "file": file_id,
                "start_pos": s, "end_pos": e, "pre_pos": pre, "commit_pos": commit,
                "length_real_steps": e - s + 1,
                "start_global_idx": int(g.loc[s, "global_idx"]),
                "end_global_idx": int(g.loc[e, "global_idx"]),
                "commit_global_idx": int(g.loc[commit, "global_idx"]),
                "entropy_at_pre": float(g.loc[pre, "entropy_actor"]),
                "entropy_at_onset": float(g.loc[s, "entropy_actor"]),
                "entropy_at_commit": float(g.loc[commit, "entropy_actor"]),
                "q_gap_at_pre": float(g.loc[pre, "q_gap"]),
                "q_gap_at_onset": float(g.loc[s, "q_gap"]),
            })
    return pd.DataFrame(rows).sort_values(["file", "start_pos"]).reset_index(drop=True)


def build_event_tables(
    df_real: pd.DataFrame,
    bouts: pd.DataFrame,
    window_pre: int = 6,
    window_post: int = 6,
    metric_cols: Sequence[str] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metrics = list(metric_cols or ["entropy_actor_prev_s2", "q_gap_prev_s2", "rollout_spread"])
    prepost_rows, temporal_rows = [], []
    grouped = {
        k: v.sort_values("real_pos").reset_index(drop=True)
        for k, v in df_real.groupby("file", sort=False)
    }

    for _, b in bouts.iterrows():
        g = grouped[b["file"]]
        pre_pos, onset_pos, commit_pos = int(b["pre_pos"]), int(b["start_pos"]), int(b["commit_pos"])
        event_id = b["event_id"]

        rec = {
            "event_id": event_id, "file": b["file"],
            "pre_pos": pre_pos, "onset_pos": onset_pos, "commit_pos": commit_pos,
            "length_real_steps": int(b["length_real_steps"]),
        }
        for m in metrics:
            if m not in g.columns:
                continue
            pre_val = float(g.loc[pre_pos, m])
            on_val = float(g.loc[onset_pos, m])
            com_val = float(g.loc[commit_pos, m])
            rec[f"{m}_pre"] = pre_val
            rec[f"{m}_onset"] = on_val
            rec[f"{m}_commit"] = com_val
            rec[f"{m}_delta"] = com_val - pre_val
        prepost_rows.append(rec)

        for anchor, anchor_pos in [("onset", onset_pos), ("commit", commit_pos)]:
            for rel in range(-window_pre, window_post + 1):
                pos = anchor_pos + rel
                if 0 <= pos < len(g):
                    row = {"event_id": event_id, "file": b["file"], "anchor": anchor, "rel_step": rel}
                    for m in metrics:
                        if m in g.columns:
                            row[m] = float(g.loc[pos, m])
                    temporal_rows.append(row)

    return (
        pd.DataFrame(prepost_rows).sort_values(["file", "pre_pos"]).reset_index(drop=True),
        pd.DataFrame(temporal_rows).sort_values(["anchor", "file", "rel_step"]).reset_index(drop=True),
    )


# ─── Section 2-1: Bout length distribution ───────────────────────────────────

def _km_survival(lengths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lengths = np.sort(lengths)
    max_t = int(lengths.max())
    x = np.arange(1, max_t + 1)
    surv = np.array([(lengths >= t).mean() for t in x], dtype=float)
    return x, surv


def estimate_cross2(lengths: np.ndarray) -> int:
    """
    First t where the KM curve crosses back above the exponential null baseline
    (after first dropping below it). This is Cross2 from Section 1-5 survival analysis.
    Falls back to 95th percentile if no crossing is found.
    """
    if len(lengths) < 5:
        return int(np.percentile(lengths, 95))
    lam = 1.0 / np.mean(lengths)
    x, km = _km_survival(lengths)
    exp_base = np.exp(-lam * x)
    km_above = km > exp_base

    # Cross1: first index where KM drops below exp
    if km_above[0]:
        cross1_candidates = np.where(~km_above)[0]
        cross1_idx = int(cross1_candidates[0]) if cross1_candidates.size > 0 else len(km) - 1
    else:
        cross1_idx = 0

    # Cross2: first index after Cross1 where KM rises back above exp
    search_start = cross1_idx + 1
    if search_start >= len(km):
        return int(np.percentile(lengths, 95))
    cross2_candidates = np.where(km_above[search_start:])[0]
    if cross2_candidates.size == 0:
        return int(np.percentile(lengths, 95))
    return int(x[search_start + cross2_candidates[0]])


def compute_cross2_per_file(bouts: pd.DataFrame) -> Dict[str, int]:
    cross2 = {}
    for file_id, g in bouts.groupby("file", sort=False):
        lengths = g["length_real_steps"].to_numpy(dtype=int)
        cross2[file_id] = estimate_cross2(lengths)
    return cross2


def plot_fig_2_1(bouts: pd.DataFrame, out_path: Path) -> None:
    """
    Fig 2-1: 3 panels
      A – KM survival + exponential null baseline (all bouts aggregated)
      B – Bout length histogram: Short (<Cross2) vs. Long (>=Cross2)
      C – Onset entropy vs. bout length (log scale) scatter + regression line
    """
    cross2_map = compute_cross2_per_file(bouts)
    bouts = bouts.copy()
    bouts["cross2"] = bouts["file"].map(cross2_map)
    bouts["is_long"] = (bouts["length_real_steps"] >= bouts["cross2"]).astype(int)

    lengths_all = bouts["length_real_steps"].to_numpy(dtype=int)
    lam = 1.0 / np.mean(lengths_all)
    x_km, km = _km_survival(lengths_all)
    exp_base = np.exp(-lam * x_km)
    median_c2 = int(np.median(list(cross2_map.values())))
    xlim_max = min(int(np.percentile(lengths_all, 99)), int(x_km[-1]))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Panel A: KM + exponential null
    ax = axes[0]
    ax.plot(x_km, km, lw=2, color="#1f77b4", label="KM (empirical)")
    ax.plot(x_km, exp_base, lw=1.5, ls="--", color="#d62728", label="Exponential null")
    ax.axvline(median_c2, color="#2ca02c", lw=1.2, ls=":", label=f"Median Cross2 = {median_c2}")
    ax.set_xlabel("Bout length (real NOOP steps)")
    ax.set_ylabel("Survival probability")
    ax.set_title("Fig 2-1A: KM survival vs. exponential null")
    ax.set_xlim(0, xlim_max)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)

    # Panel B: Short / Long histogram (log x-axis)
    ax = axes[1]
    short_l = bouts.loc[bouts["is_long"] == 0, "length_real_steps"].to_numpy()
    long_l = bouts.loc[bouts["is_long"] == 1, "length_real_steps"].to_numpy()
    bins = np.logspace(0, np.log10(max(int(lengths_all.max()), 2)), 40)
    ax.hist(short_l, bins=bins, color="#1f77b4", alpha=0.7, label=f"Short (<Cross2, n={len(short_l)})")
    ax.hist(long_l, bins=bins, color="#ff7f0e", alpha=0.7, label=f"Long (≥Cross2, n={len(long_l)})")
    ax.set_xscale("log")
    ax.set_xlabel("Bout length (log scale)")
    ax.set_ylabel("Count")
    ax.set_title("Fig 2-1B: Short vs. Long bout distribution")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)

    # Panel C: onset entropy vs. log(bout length)
    ax = axes[2]
    x_sc = bouts["entropy_at_onset"].to_numpy(dtype=float)
    y_sc = np.log1p(bouts["length_real_steps"].to_numpy(dtype=float))
    colors_sc = np.where(bouts["is_long"].to_numpy() == 1, "#ff7f0e", "#1f77b4")
    ax.scatter(x_sc, y_sc, s=4, alpha=0.2, c=colors_sc)
    mask = np.isfinite(x_sc) & np.isfinite(y_sc)
    if mask.sum() > 5:
        slope, intercept, r, p, _ = scipy_stats.linregress(x_sc[mask], y_sc[mask])
        xx = np.linspace(np.nanmin(x_sc), np.nanmax(x_sc), 100)
        ax.plot(xx, slope * xx + intercept, color="#d62728", lw=2,
                label=f"r = {r:.3f}, p = {p:.3g}")
        ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("Entropy at onset")
    ax.set_ylabel("log(1 + bout length)")
    ax.set_title("Fig 2-1C: Onset entropy vs. bout length")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


# ─── Section 2-2: Temporal profiles ──────────────────────────────────────────

def plot_fig_2_2(
    bouts: pd.DataFrame,
    temporal: pd.DataFrame,
    prepost: pd.DataFrame,
    out_path: Path,
    metric: str = "entropy_actor_prev_s2",
) -> None:
    """
    Fig 2-2: 4 panels
      A – Onset-aligned z-score temporal profile
      B – Commit-aligned z-score temporal profile
      C – Bout survival by pre-uncertainty (2-bin median split: Low / High)
      D – Short vs. Long bout commit-aligned profile comparison
    """
    cross2_map = compute_cross2_per_file(bouts)
    bouts_c2 = bouts.copy()
    bouts_c2["cross2"] = bouts_c2["file"].map(cross2_map)
    bouts_c2["is_long"] = (bouts_c2["length_real_steps"] >= bouts_c2["cross2"]).astype(int)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

    def _plot_anchor_profile(ax, anchor: str, title: str) -> None:
        sub = temporal[temporal["anchor"] == anchor]
        if metric not in sub.columns or len(sub) == 0:
            ax.set_axis_off()
            return
        rels = np.sort(sub["rel_step"].unique())
        pivot = sub.pivot_table(index="event_id", columns="rel_step", values=metric, aggfunc="mean")
        pivot = pivot.reindex(columns=rels)
        y = pivot.to_numpy(dtype=float)
        row_mu = np.nanmean(y, axis=1, keepdims=True)
        row_sd = np.nanstd(y, axis=1, keepdims=True) + EPS
        y_z = (y - row_mu) / row_sd
        mu, sem = _mean_sem(y_z)
        ax.plot(rels, mu, lw=2, color="#1f77b4")
        ax.fill_between(rels, mu - sem, mu + sem, alpha=0.2, color="#1f77b4")
        ax.axvline(0, color="black", lw=1, ls="--")
        ax.set_xlabel(f"Real steps from {anchor}")
        ax.set_ylabel("Within-event z-score")
        ax.set_title(title)
        ax.grid(alpha=0.3)

    _plot_anchor_profile(axes[0], "onset", "Fig 2-2A: Onset-aligned profile")
    _plot_anchor_profile(axes[1], "commit", "Fig 2-2B: Commit-aligned profile")

    # Panel C: bout survival by pre-uncertainty (2 bins)
    ax = axes[2]
    ent_col = f"{metric}_pre"
    if len(prepost) > 0 and ent_col in prepost.columns:
        ent = prepost[ent_col].to_numpy(dtype=float)
        med = np.nanquantile(ent, 0.5)
        bin_idx = np.digitize(ent, bins=[med], right=True)
        max_len = int(bouts["length_real_steps"].max())
        x = np.arange(1, max_len + 1)
        for b_val, label, color in [(0, "Low uncertainty", "#2196F3"), (1, "High uncertainty", "#F44336")]:
            lens = prepost.loc[bin_idx == b_val, "length_real_steps"].to_numpy(dtype=int)
            if len(lens) == 0:
                continue
            surv = np.array([(lens >= t).mean() for t in x], dtype=float)
            ax.plot(x, surv, lw=2, label=label, color=color)
        ax.set_xlabel("Bout length (real NOOP steps)")
        ax.set_ylabel("Survival probability")
        ax.set_title("Fig 2-2C: Survival by pre-uncertainty")
        ax.legend(frameon=False, fontsize=9)
        ax.grid(alpha=0.3)
    else:
        ax.set_axis_off()

    # Panel D: Short vs. Long bout commit-aligned comparison
    ax = axes[3]
    if len(temporal) > 0 and metric in temporal.columns:
        commit_sub = temporal[temporal["anchor"] == "commit"]
        rels = np.sort(commit_sub["rel_step"].unique())
        for is_long, label, color in [(0, "Short", "#1f77b4"), (1, "Long", "#ff7f0e")]:
            ids_in_group = set(bouts_c2.loc[bouts_c2["is_long"] == is_long, "event_id"])
            sub = commit_sub[commit_sub["event_id"].isin(ids_in_group)]
            if len(sub) == 0:
                continue
            pivot = sub.pivot_table(index="event_id", columns="rel_step", values=metric, aggfunc="mean")
            pivot = pivot.reindex(columns=rels)
            y = pivot.to_numpy(dtype=float)
            row_mu = np.nanmean(y, axis=1, keepdims=True)
            row_sd = np.nanstd(y, axis=1, keepdims=True) + EPS
            y_z = (y - row_mu) / row_sd
            mu, sem = _mean_sem(y_z)
            n_ev = int(pivot.shape[0])
            ax.plot(rels, mu, lw=2, label=f"{label} bouts (n={n_ev})", color=color)
            ax.fill_between(rels, mu - sem, mu + sem, alpha=0.15, color=color)
        ax.axvline(0, color="black", lw=1, ls="--")
        ax.set_xlabel("Real steps from commit")
        ax.set_ylabel("Within-event z-score")
        ax.set_title("Fig 2-2D: Short vs. Long commit profile")
        ax.legend(frameon=False, fontsize=9)
        ax.grid(alpha=0.3)
    else:
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


# ─── Section 2-3: Sequential structure ───────────────────────────────────────

def compute_transition_matrix(df_real: pd.DataFrame) -> np.ndarray:
    """
    First-order 2×2 transition probability matrix.
    Rows = from {action=0, NOOP=1}, columns = to {action=0, NOOP=1}.
    """
    counts = np.zeros((2, 2), dtype=float)
    for _, g in df_real.groupby("file", sort=False):
        seq = g.sort_values("real_pos")["is_human_noop"].to_numpy(dtype=int)
        for t_i in range(len(seq) - 1):
            counts[seq[t_i], seq[t_i + 1]] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    return counts / np.where(row_sums > 0, row_sums, 1.0)


def compute_bout_internal_trajectory(
    df_real: pd.DataFrame,
    bouts: pd.DataFrame,
    n_bins: int = 5,
) -> pd.DataFrame:
    """
    Sample entropy_actor at n_bins normalized positions within each bout
    (0 = onset, 1 = final NOOP step). Enables within-bout entropy trajectory.
    """
    cross2_map = compute_cross2_per_file(bouts)
    grouped = {
        k: v.sort_values("real_pos").reset_index(drop=True)
        for k, v in df_real.groupby("file", sort=False)
    }
    rows = []
    for _, b in bouts.iterrows():
        g = grouped[b["file"]]
        s, e = int(b["start_pos"]), int(b["end_pos"])
        bout_len = e - s + 1
        if bout_len < 2:
            continue
        positions = np.round(np.linspace(s, e, n_bins)).astype(int)
        positions = np.clip(positions, 0, len(g) - 1)
        ents = [float(g.loc[p, "entropy_actor"]) for p in positions]
        cross2 = cross2_map.get(b["file"], 999)
        row = {
            "event_id": b["event_id"],
            "file": b["file"],
            "bout_length": bout_len,
            "is_long": int(bout_len >= cross2),
        }
        for i, v in enumerate(ents):
            row[f"ent_pos{i}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def compute_curaction_stability(df_real: pd.DataFrame, bouts: pd.DataFrame) -> pd.DataFrame:
    """
    Per bout: how many times does thinker_action change across the NOOP steps.
    Returns change_rate = n_changes / (bout_length - 1).
    """
    cross2_map = compute_cross2_per_file(bouts)
    grouped = {
        k: v.sort_values("real_pos").reset_index(drop=True)
        for k, v in df_real.groupby("file", sort=False)
    }
    rows = []
    for _, b in bouts.iterrows():
        g = grouped[b["file"]]
        s, e = int(b["start_pos"]), int(b["end_pos"])
        actions = g.loc[s:e, "thinker_action"].to_numpy(dtype=int)
        n_changes = int(np.sum(np.diff(actions) != 0))
        bout_len = e - s + 1
        cross2 = cross2_map.get(b["file"], 999)
        rows.append({
            "event_id": b["event_id"],
            "file": b["file"],
            "bout_length": bout_len,
            "n_changes": n_changes,
            "change_rate": n_changes / max(bout_len - 1, 1),
            "is_long": int(bout_len >= cross2),
        })
    return pd.DataFrame(rows)


def plot_fig_2_3(
    df_real: pd.DataFrame,
    bouts: pd.DataFrame,
    traj: pd.DataFrame,
    stability: pd.DataFrame,
    out_path: Path,
) -> None:
    """
    Fig 2-3: 3 panels
      A – First-order transition matrix heatmap (Action / NOOP)
      B – Bout-internal entropy trajectory, Short vs. Long (normalized position)
      C – Thinker action change rate within bouts, Short vs. Long (boxplot)
    """
    trans = compute_transition_matrix(df_real)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel A: transition matrix heatmap
    ax = axes[0]
    im = ax.imshow(trans, vmin=0, vmax=1, cmap="Blues")
    tick_labels = ["Action", "NOOP"]
    ax.set_xticks([0, 1], tick_labels)
    ax.set_yticks([0, 1], tick_labels)
    ax.set_xlabel("To")
    ax.set_ylabel("From")
    ax.set_title("Fig 2-3A: NOOP/Action transition matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{trans[i, j]:.3f}", ha="center", va="center", fontsize=11,
                    color="white" if trans[i, j] > 0.6 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel B: bout-internal entropy trajectory
    ax = axes[1]
    pos_cols = [c for c in traj.columns if c.startswith("ent_pos")]
    x_pos = np.linspace(0, 1, len(pos_cols))
    for is_long, label, color in [(0, "Short", "#1f77b4"), (1, "Long", "#ff7f0e")]:
        sub = traj[traj["is_long"] == is_long][pos_cols].to_numpy(dtype=float)
        if len(sub) == 0:
            continue
        row_mu = np.nanmean(sub, axis=1, keepdims=True)
        row_sd = np.nanstd(sub, axis=1, keepdims=True) + EPS
        sub_z = (sub - row_mu) / row_sd
        mu, sem = _mean_sem(sub_z)
        ax.plot(x_pos, mu, lw=2, label=f"{label} (n={len(sub)})", color=color)
        ax.fill_between(x_pos, mu - sem, mu + sem, alpha=0.2, color=color)
    ax.set_xlabel("Normalized position within bout\n(0 = onset, 1 = last NOOP step)")
    ax.set_ylabel("Within-bout z-score entropy")
    ax.set_title("Fig 2-3B: Bout-internal entropy trajectory")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.3)

    # Panel C: thinker action stability
    ax = axes[2]
    bp_data, bp_positions, bp_labels, bp_colors = [], [], [], []
    for is_long, label, color in [(0, "Short", "#1f77b4"), (1, "Long", "#ff7f0e")]:
        vals = stability.loc[stability["is_long"] == is_long, "change_rate"].to_numpy()
        if len(vals) == 0:
            continue
        bp_data.append(vals)
        bp_positions.append(is_long)
        bp_labels.append(label)
        bp_colors.append(color)
    if bp_data:
        bps = ax.boxplot(bp_data, positions=bp_positions, widths=0.5, patch_artist=True,
                         medianprops=dict(color="black", lw=2), showfliers=False)
        for patch, color in zip(bps["boxes"], bp_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        ax.set_xticks(bp_positions, [f"{lbl} bouts" for lbl in bp_labels])
        ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel("Thinker action change rate\n(changes / NOOP step)")
    ax.set_title("Fig 2-3C: Thinker action stability within bouts")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


# ─── Section 2-4: Within-session adaptation ──────────────────────────────────

def compute_episode_stats(df_real: pd.DataFrame) -> pd.DataFrame:
    """Per-chunk (episode) aggregate stats: NOOP ratio, entropy, reward."""
    rows = []
    for file_id, g in df_real.groupby("file", sort=False):
        meta = parse_file_meta(Path(file_id))
        rows.append({
            "file": file_id,
            "subject": meta.subject,
            "game": meta.game,
            "chunk": int(meta.chunk),
            "n_steps": len(g),
            "noop_ratio": float(g["is_human_noop"].mean()),
            "mean_entropy": float(g["entropy_actor"].mean()),
            "mean_q_gap": float(g["q_gap"].mean()),
            "total_reward": float(g["env_return"].sum()),
        })
    df = pd.DataFrame(rows).sort_values(["subject", "game", "chunk"]).reset_index(drop=True)
    df["episode_idx"] = df.groupby(["subject", "game"]).cumcount()
    return df


def plot_fig_2_4(df_real: pd.DataFrame, out_path: Path) -> None:
    """
    Fig 2-4: 2 panels
      A – Episode-level NOOP ratio over session index (mean ± SEM across chunks)
      B – Early vs. Late half NOOP ratio per chunk (paired lines + boxplot)
    """
    ep = compute_episode_stats(df_real)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel A: NOOP ratio over episode index
    ax = axes[0]
    grp = ep.groupby("episode_idx")["noop_ratio"]
    ep_mu = grp.mean()
    ep_sem = grp.sem()
    x = ep_mu.index.to_numpy()
    ax.plot(x, ep_mu.to_numpy(), lw=2, color="#1f77b4")
    ax.fill_between(x, (ep_mu - ep_sem).to_numpy(), (ep_mu + ep_sem).to_numpy(),
                    alpha=0.2, color="#1f77b4")
    ax.set_xlabel("Episode (chunk) index within session")
    ax.set_ylabel("NOOP ratio")
    ax.set_title("Fig 2-4A: NOOP ratio over session")
    ax.grid(alpha=0.3)

    # Panel B: Early vs. Late half paired comparison
    ax = axes[1]
    half_rows = []
    for file_id, g in df_real.groupby("file", sort=False):
        g = g.sort_values("real_pos")
        mid = len(g) // 2
        half_rows.append({"file": file_id, "half": "Early", "noop_ratio": g.iloc[:mid]["is_human_noop"].mean()})
        half_rows.append({"file": file_id, "half": "Late", "noop_ratio": g.iloc[mid:]["is_human_noop"].mean()})
    half_df = pd.DataFrame(half_rows)

    colors_half = {"Early": "#1f77b4", "Late": "#ff7f0e"}
    pos_map = {"Early": 0, "Late": 1}
    for half, grp_h in half_df.groupby("half"):
        vals = grp_h["noop_ratio"].to_numpy()
        pos = pos_map[half]
        bp = ax.boxplot(vals, positions=[pos], widths=0.5, patch_artist=True,
                        medianprops=dict(color="black", lw=2), showfliers=False)
        bp["boxes"][0].set_facecolor(colors_half[half])
        bp["boxes"][0].set_alpha(0.65)

    pivot_half = half_df.pivot(index="file", columns="half", values="noop_ratio").dropna()
    for _, row in pivot_half.iterrows():
        ax.plot([0, 1], [row["Early"], row["Late"]], color="gray", alpha=0.4, lw=1)

    ax.set_xticks([0, 1], ["Early half", "Late half"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel("NOOP ratio")
    ax.set_title("Fig 2-4B: Early vs. Late half NOOP ratio")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


# ─── Summary stats ────────────────────────────────────────────────────────────

def compute_section2_summary(bouts: pd.DataFrame) -> pd.DataFrame:
    cross2_map = compute_cross2_per_file(bouts)
    rows = []
    for file_id, g in bouts.groupby("file", sort=False):
        lengths = g["length_real_steps"].to_numpy(dtype=int)
        c2 = cross2_map.get(file_id, np.nan)
        short_mask = lengths < c2 if not np.isnan(c2) else np.ones(len(lengths), dtype=bool)
        long_mask = ~short_mask
        rows.append({
            "file": file_id,
            "n_bouts": len(lengths),
            "mean_length": float(np.mean(lengths)),
            "median_length": float(np.median(lengths)),
            "cross2": c2,
            "n_short": int(short_mask.sum()),
            "n_long": int(long_mask.sum()),
            "pct_long": float(long_mask.sum() / max(len(lengths), 1)),
            "mean_entropy_onset_short": float(g.loc[short_mask, "entropy_at_onset"].mean()) if short_mask.sum() > 0 else np.nan,
            "mean_entropy_onset_long": float(g.loc[long_mask, "entropy_at_onset"].mean()) if long_mask.sum() > 0 else np.nan,
            "mean_entropy_commit_short": float(g.loc[short_mask, "entropy_at_commit"].mean()) if short_mask.sum() > 0 else np.nan,
            "mean_entropy_commit_long": float(g.loc[long_mask, "entropy_at_commit"].mean()) if long_mask.sum() > 0 else np.nan,
            "mean_q_gap_onset_short": float(g.loc[short_mask, "q_gap_at_onset"].mean()) if short_mask.sum() > 0 else np.nan,
            "mean_q_gap_onset_long": float(g.loc[long_mask, "q_gap_at_onset"].mean()) if long_mask.sum() > 0 else np.nan,
        })
    return pd.DataFrame(rows)


# ─── Main pipeline ────────────────────────────────────────────────────────────

def run_section2(
    input_dir: Path,
    out_dir: Path,
    window_pre: int = 6,
    window_post: int = 6,
) -> None:
    paths = sorted(input_dir.glob("*.npy"))
    if not paths:
        raise FileNotFoundError(f"No .npy files in: {input_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_real = []
    for p in paths:
        df = build_real_step_table(load_npy_dict(p), p, k_rewards=(5,))
        if len(df) > 0:
            all_real.append(df)
    if not all_real:
        raise RuntimeError("No real-step rows produced.")

    df_real = pd.concat(all_real, ignore_index=True)
    bouts = extract_noop_bouts(df_real)

    prepost, temporal = build_event_tables(
        df_real, bouts,
        window_pre=window_pre, window_post=window_post,
        metric_cols=["entropy_actor_prev_s2", "q_gap_prev_s2", "rollout_spread"],
    )
    traj = compute_bout_internal_trajectory(df_real, bouts)
    stability = compute_curaction_stability(df_real, bouts)
    summary = compute_section2_summary(bouts)

    # Save CSVs
    df_real.to_csv(out_dir / "real_step_metrics.csv", index=False)
    bouts.to_csv(out_dir / "noop_bouts.csv", index=False)
    prepost.to_csv(out_dir / "event_prepost.csv", index=False)
    temporal.to_csv(out_dir / "event_temporal.csv", index=False)
    traj.to_csv(out_dir / "bout_internal_trajectory.csv", index=False)
    stability.to_csv(out_dir / "bout_action_stability.csv", index=False)
    summary.to_csv(out_dir / "section2_summary.csv", index=False)

    # Generate figures
    plot_fig_2_1(bouts, out_dir / "fig_2_1_bout_distribution.png")
    plot_fig_2_2(bouts, temporal, prepost, out_dir / "fig_2_2_temporal_profiles.png")
    plot_fig_2_3(df_real, bouts, traj, stability, out_dir / "fig_2_3_sequential_structure.png")
    plot_fig_2_4(df_real, out_dir / "fig_2_4_session_adaptation.png")

    print(f"Section 2 analysis complete → {out_dir}")
    print(summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Section 2: NOOP Bout Structure Analysis")
    p.add_argument("--input-dir", type=Path, default=Path("test/sub001/ses-04"),
                   help="Directory containing .npy files")
    p.add_argument("--out-dir", type=Path, default=Path("analysis_outputs/section2_structure"),
                   help="Output directory for figures and CSVs")
    p.add_argument("--window-pre", type=int, default=6, help="Temporal window before onset/commit")
    p.add_argument("--window-post", type=int, default=6, help="Temporal window after onset/commit")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_section2(
        input_dir=args.input_dir,
        out_dir=args.out_dir,
        window_pre=args.window_pre,
        window_post=args.window_post,
    )


if __name__ == "__main__":
    main()
