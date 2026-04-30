#!/usr/bin/env python3
"""
02_structure_analysis.py  –  Section 2: NOOP Bout Structure Analysis

Sub-sections:
  2-1  Onset uncertainty ~ bout length regression.
  2-2  Temporal profiles around onset and commit
       Baseline-normalized onset/commit profiles (Fig 2-2A/B), uncertainty vs.
       confidence gain (Fig 2-2C), Cross2 Short vs. Long commit comparison (Fig 2-2D).
  2-3  Short/Long decision-aligned metrics
       Real-action policy entropy/margin plus pre-real-search Q-gap/rollout spread.
  2-4  Bout-internal policy trajectory
       Normalized NOOP onset → real action commit trajectories for actor entropy and margin.

Outputs (all in --out-dir):
  fig_2_1C_onset_entropy_bout_length.png
  fig_2_2_temporal_profiles.png
  fig_2_2_1_temporal_profiles_short_long.png
  fig_2_3_short_long_metrics.png
  fig_2_4_noop_commit_policy_trajectory.png
  fig_2_5_commit_hazard.png
  noop_commit_policy_trajectory.csv
  commit_hazard_steps.csv
  commit_hazard_model_summary.csv
  commit_hazard_coefficients.csv
  section2_summary.csv
  short_long_event_metrics.csv
  short_long_metric_summary.csv
  short_long_metric_tests.csv
  bout_internal_trajectory.csv
  bout_action_stability.csv
  bout_commit_dynamics.csv
  episode_stats.csv
  half_session_stats.csv
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from scipy import optimize as scipy_optimize
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
PROFILE_BASELINE_RELS = (-6, -5, -4)

METRIC_LABELS = {
    "entropy_actor": "Entropy",
    "entropy_actor_prev_s2": "Prev-s2 entropy",
    "margin_actor": "Policy margin",
    "margin_actor_prev_s2": "Prev-s2 policy margin",
    "q_gap": "Q-gap",
    "q_gap_prev_s2": "Prev-s2 Q-gap",
    "rollout_spread": "Rollout spread",
    "rollout_spread_prev_s2": "Prev-s2 rollout spread",
}


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


def normalize_subject_id(subject: object) -> str:
    digits = re.search(r"\d+", str(subject))
    return f"sub{int(digits.group(0)):03d}" if digits else str(subject)


def normalize_game_id(game: object) -> str:
    digits = re.search(r"\d+", str(game))
    return str(int(digits.group(0))) if digits else str(game)


def load_cross2_summary(path: Optional[Path]) -> Dict[Tuple[str, str], int]:
    if path is None or not path.exists():
        return {}
    table = pd.read_csv(path)
    required = {"subject", "game", "cross2"}
    if not required.issubset(table.columns):
        raise ValueError(f"Cross2 summary must contain columns {sorted(required)}: {path}")
    out: Dict[Tuple[str, str], int] = {}
    for row in table.itertuples(index=False):
        out[(normalize_subject_id(row.subject), normalize_game_id(row.game))] = int(row.cross2)
    return out


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
    sd = np.zeros_like(mu, dtype=float)
    valid = n > 1
    if np.any(valid):
        sd[valid] = np.nanstd(y[:, valid], axis=0, ddof=1)
    sem = np.where(n > 0, sd / np.sqrt(np.maximum(n, 1)), np.nan)
    return mu, sem


def _safe_int(x: object, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _safe_corr_slope(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr, y_arr = x_arr[mask], y_arr[mask]
    if x_arr.size < 3 or np.nanstd(x_arr) <= 0 or np.nanstd(y_arr) <= 0:
        return np.nan, np.nan
    z = (x_arr - np.nanmean(x_arr)) / (np.nanstd(x_arr) + EPS)
    slope, _, r, _, _ = scipy_stats.linregress(z, y_arr)
    return float(r), float(slope)


def _metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("_", " "))


def _segment_stats(values: np.ndarray) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    mean_val = float(np.mean(arr))
    final_val = float(arr[-1])
    if arr.size < 2 or np.nanstd(arr) <= 0:
        slope_val = 0.0
    else:
        x = np.linspace(0.0, 1.0, arr.size)
        slope_val = float(scipy_stats.linregress(x, arr).slope)
    return mean_val, final_val, slope_val


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
    margin_actor = top2_gap_rows(probs)
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

        imag_action_idx = imag_idx[(imag_idx >= 0) & (imag_idx < t)]
        if imag_action_idx.size == 0:
            fallback_idx = prev_s2 if 0 <= prev_s2 < t else min(max(idx_global - 1, 0), t - 1)
            imag_actions = np.array([cur_action_id[fallback_idx]], dtype=int)
        else:
            imag_actions = cur_action_id[imag_action_idx].astype(int)
        imag_vals, imag_counts = np.unique(imag_actions, return_counts=True)
        imag_probs = imag_counts / np.maximum(np.sum(imag_counts), 1)
        imag_modal_action = int(imag_vals[np.argmax(imag_counts)]) if len(imag_vals) else -1
        imag_final_action = int(imag_actions[-1]) if len(imag_actions) else -1
        imag_change_rate = float(np.sum(np.diff(imag_actions) != 0) / max(len(imag_actions) - 1, 1))
        imag_action_entropy = float(-np.sum(imag_probs * np.log(imag_probs + EPS))) if len(imag_probs) else np.nan
        imag_s2_idx = imag_idx[status[imag_idx] == 2] if imag_idx.size > 0 else np.array([], dtype=int)
        if imag_s2_idx.size == 0 and 0 <= prev_s2 < t:
            imag_s2_idx = np.array([prev_s2], dtype=int)
        q_gap_imag_mean, q_gap_imag_final, q_gap_imag_slope = _segment_stats(q_gap[imag_s2_idx])
        spread_imag_mean, spread_imag_final, spread_imag_slope = _segment_stats(rollout_spread[imag_s2_idx])

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
            "imag_cur_action_final": imag_final_action,
            "imag_cur_action_modal": imag_modal_action,
            "imag_cur_action_n_unique": int(len(imag_vals)),
            "imag_cur_action_change_rate": imag_change_rate,
            "imag_cur_action_entropy": imag_action_entropy,
            "imag_nonnoop_fraction": float(np.mean(imag_actions != 0)) if len(imag_actions) else np.nan,
            "imag_n_steps": int(len(imag_actions)),
            "imag_s2_n_steps": int(len(imag_s2_idx)),
            "q_gap_imag_mean": q_gap_imag_mean,
            "q_gap_imag_final": q_gap_imag_final,
            "q_gap_imag_slope": q_gap_imag_slope,
            "rollout_spread_imag_mean": spread_imag_mean,
            "rollout_spread_imag_final": spread_imag_final,
            "rollout_spread_imag_slope": spread_imag_slope,
            "is_human_noop": int(target_action == 0),
            "is_thinker_noop": int(thinker_action[idx_global] == 0),
            "env_return": float(env_return[idx_global]) if idx_global < len(env_return) else np.nan,
            "entropy_actor": float(entropy_actor[idx_global]),
            "margin_actor": float(margin_actor[idx_global]),
            "q_gap": float(q_gap[idx_global]),
            "rollout_spread": float(rollout_spread[idx_global]),
            "entropy_actor_prev_s2": float(entropy_actor[prev_s2]),
            "margin_actor_prev_s2": float(margin_actor[prev_s2]),
            "q_gap_prev_s2": float(q_gap[prev_s2]),
            "rollout_spread_prev_s2": float(rollout_spread[prev_s2]),
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
        meta = parse_file_meta(Path(file_id))
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
                "subject": meta.subject, "session": meta.session,
                "block": meta.block, "game": meta.game, "chunk": meta.chunk,
                "start_pos": s, "end_pos": e, "pre_pos": pre, "commit_pos": commit,
                "length_real_steps": e - s + 1,
                "start_global_idx": int(g.loc[s, "global_idx"]),
                "end_global_idx": int(g.loc[e, "global_idx"]),
                "commit_global_idx": int(g.loc[commit, "global_idx"]),
                "commit_action": int(g.loc[commit, "human_action"]),
                "entropy_at_pre": float(g.loc[pre, "entropy_actor"]),
                "entropy_at_onset": float(g.loc[s, "entropy_actor"]),
                "entropy_at_commit": float(g.loc[commit, "entropy_actor"]),
                "margin_at_pre": float(g.loc[pre, "margin_actor"]),
                "margin_at_onset": float(g.loc[s, "margin_actor"]),
                "margin_at_commit": float(g.loc[commit, "margin_actor"]),
                "q_gap_at_pre": float(g.loc[pre, "q_gap"]),
                "q_gap_at_onset": float(g.loc[s, "q_gap"]),
                "q_gap_at_commit": float(g.loc[commit, "q_gap"]),
                "rollout_spread_at_pre": float(g.loc[pre, "rollout_spread"]),
                "rollout_spread_at_onset": float(g.loc[s, "rollout_spread"]),
                "rollout_spread_at_commit": float(g.loc[commit, "rollout_spread"]),
            })
    return pd.DataFrame(rows).sort_values(["file", "start_pos"]).reset_index(drop=True)


def build_event_tables(
    df_real: pd.DataFrame,
    bouts: pd.DataFrame,
    window_pre: int = 6,
    window_post: int = 6,
    metric_cols: Sequence[str] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metrics = list(metric_cols or [
        "entropy_actor_prev_s2", "margin_actor_prev_s2", "q_gap_prev_s2",
        "rollout_spread_prev_s2", "rollout_spread",
    ])
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
            rec[f"{m}_delta_commit_pre"] = com_val - pre_val
            rec[f"{m}_delta_commit_onset"] = com_val - on_val
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


# ─── Section 2-1: Onset entropy vs. bout length ──────────────────────────────

def _km_survival(lengths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lengths = np.sort(lengths)
    max_t = int(lengths.max())
    x = np.arange(1, max_t + 1)
    surv = np.array([(lengths >= t).mean() for t in x], dtype=float)
    return x, surv


def estimate_cross2(lengths: np.ndarray) -> int:
    """
    First t where the KM curve crosses back above the exponential null baseline
    (after first dropping below it). This is the Cross2 threshold used for
    Short/Long bout splits.
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


def _with_bout_meta(bouts: pd.DataFrame) -> pd.DataFrame:
    out = bouts.copy()
    if "subject" not in out.columns or "game" not in out.columns:
        metas = [parse_file_meta(Path(f)) for f in out["file"]]
        out["subject"] = [m.subject for m in metas]
        out["game"] = [m.game for m in metas]
    out["subject"] = [normalize_subject_id(s) for s in out["subject"]]
    out["game"] = [normalize_game_id(g) for g in out["game"]]
    return out


def compute_cross2_by_subject_game(bouts: pd.DataFrame) -> Dict[Tuple[str, str], int]:
    """
    Estimate Cross2 separately for each subject × game survival curve.
    This mirrors Section 1-5 and avoids arbitrary file/chunk-specific splits.
    """
    cross2: Dict[Tuple[str, str], int] = {}
    bouts_meta = _with_bout_meta(bouts)
    for (subject, game), g in bouts_meta.groupby(["subject", "game"], sort=True):
        lengths = g["length_real_steps"].to_numpy(dtype=int)
        cross2[(str(subject), str(game))] = estimate_cross2(lengths)
    return cross2


def annotate_cross2_bouts(
    bouts: pd.DataFrame,
    cross2_override: Optional[Dict[Tuple[str, str], int]] = None,
) -> pd.DataFrame:
    out = _with_bout_meta(bouts)
    if "cross2" in out.columns and out["cross2"].notna().all() and not cross2_override:
        out["cross2"] = out["cross2"].astype(int)
    else:
        estimated = compute_cross2_by_subject_game(out)
        cross2_override = cross2_override or {}
        out["cross2"] = [
            int(cross2_override.get((str(row.subject), str(row.game)), estimated[(str(row.subject), str(row.game))]))
            for row in out.itertuples(index=False)
        ]
    out["is_long"] = (
        out["length_real_steps"].to_numpy(dtype=int) >= out["cross2"].to_numpy(dtype=int)
    ).astype(int)
    out["bout_class"] = np.where(out["is_long"] == 1, "Long", "Short")
    return out


def plot_fig_2_1c(bouts: pd.DataFrame, out_path: Path) -> None:
    """
    Fig 2-1C: Onset entropy vs. bout length (log scale) scatter + regression line.
    """
    bouts = annotate_cross2_bouts(bouts)

    fig, ax = plt.subplots(figsize=(5.8, 4.5))
    x_sc = bouts["entropy_at_onset"].to_numpy(dtype=float)
    y_sc = np.log1p(bouts["length_real_steps"].to_numpy(dtype=float))

    for is_long, label, color in [
        (0, "Short (<Cross2)", "#1f77b4"),
        (1, "Long (>=Cross2)", "#ff7f0e"),
    ]:
        sub = bouts["is_long"].to_numpy(dtype=int) == is_long
        ax.scatter(x_sc[sub], y_sc[sub], s=5, alpha=0.22, c=color, label=label)

    mask = np.isfinite(x_sc) & np.isfinite(y_sc)
    if mask.sum() > 5:
        slope, intercept, r, p, _ = scipy_stats.linregress(x_sc[mask], y_sc[mask])
        xx = np.linspace(np.nanmin(x_sc), np.nanmax(x_sc), 100)
        ax.plot(xx, slope * xx + intercept, color="#d62728", lw=2,
                label=f"r = {r:.3f}, p = {p:.3g}")
    ax.set_xlabel("Entropy at onset")
    ax.set_ylabel("log(1 + bout length)")
    ax.set_title("Fig 2-1C: Onset entropy vs. bout length")
    ax.legend(frameon=False, fontsize=8)
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
    metric: str = "entropy_actor",
    variant_label: Optional[str] = None,
) -> None:
    """
    Fig 2-2: 4 panels.
      A – Onset-aligned profile, delta from each event's pre-anchor baseline.
      B – Commit-aligned profile, delta from each event's pre-anchor baseline.
      C – Pre-withholding uncertainty vs. confidence gain.
      D – Cross2 Short vs. Long bout commit-aligned profile comparison.
    """
    bouts_c2 = annotate_cross2_bouts(bouts)
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2))
    axes = axes.ravel()
    metric_label = _metric_label(metric)
    title_suffix = f" ({variant_label})" if variant_label else ""

    def _title(panel: str, text: str) -> str:
        return f"Fig 2-2{panel}{title_suffix}: {text}"

    def _profile_matrix(sub: pd.DataFrame, rels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pivot = sub.pivot_table(index="event_id", columns="rel_step", values=metric, aggfunc="mean")
        pivot = pivot.reindex(columns=rels)
        y = pivot.to_numpy(dtype=float)
        baseline_cols = [r for r in PROFILE_BASELINE_RELS if r in pivot.columns]
        if not baseline_cols:
            baseline_cols = [r for r in pivot.columns if r < 0]
        if baseline_cols:
            base = pivot[baseline_cols].to_numpy(dtype=float)
            y = y - np.nanmean(base, axis=1, keepdims=True)
        return pivot.index.to_numpy(), y

    def _plot_anchor_profile(ax, anchor: str, title: str) -> None:
        sub = temporal[temporal["anchor"] == anchor]
        if metric not in sub.columns or len(sub) == 0:
            ax.set_axis_off()
            return
        rels = np.sort(sub["rel_step"].unique())
        _, y = _profile_matrix(sub, rels)
        mu, sem = _mean_sem(y)
        ax.plot(rels, mu, lw=2, color="#1f77b4")
        ax.fill_between(rels, mu - sem, mu + sem, alpha=0.2, color="#1f77b4")
        ax.axvline(0, color="black", lw=1, ls="--")
        ax.set_xlabel(f"Real steps from {anchor}")
        ax.set_ylabel(f"Delta {metric_label}\nfrom event baseline")
        ax.set_title(title)
        ax.grid(alpha=0.3)

    _plot_anchor_profile(axes[0], "onset", _title("A", "Onset-aligned profile"))
    _plot_anchor_profile(axes[1], "commit", _title("B", "Commit-aligned profile"))

    # Panel C: uncertainty vs. confidence gain. This keeps the useful plot from
    # withholding_analysis_sub001.py, but places it in the commit-resolution section.
    ax = axes[2]
    pre_col = f"{metric}_pre"
    delta_col = f"{metric}_delta"
    if len(prepost) > 0 and pre_col in prepost.columns and delta_col in prepost.columns:
        x = prepost[pre_col].to_numpy(dtype=float)
        y = -prepost[delta_col].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[mask], y[mask], s=9, alpha=0.22, color="#1f77b4")
        if mask.sum() > 5 and np.nanstd(x[mask]) > 0:
            slope, intercept, r, p, _ = scipy_stats.linregress(x[mask], y[mask])
            xx = np.linspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 100)
            ax.plot(xx, slope * xx + intercept, color="#d62728", lw=2,
                    label=f"r = {r:.3f}, p = {p:.3g}")
            ax.legend(frameon=False, fontsize=8)
        ax.axhline(0, color="black", lw=1, ls="--")
        ax.set_xlabel(f"Pre-withholding {metric_label.lower()}")
        ax.set_ylabel(f"Confidence gain\n(-delta {metric_label.lower()})")
        ax.set_title(_title("C", "Uncertainty vs. confidence gain"))
        ax.grid(alpha=0.3)
    else:
        ax.set_axis_off()

    # Panel D: Short vs. Long bout commit-aligned comparison using subject×game Cross2.
    ax = axes[3]
    if len(temporal) > 0 and metric in temporal.columns:
        commit_sub = temporal[temporal["anchor"] == "commit"]
        rels = np.sort(commit_sub["rel_step"].unique())
        for is_long, label, color in [(0, "Short", "#1f77b4"), (1, "Long", "#ff7f0e")]:
            ids_in_group = set(bouts_c2.loc[bouts_c2["is_long"] == is_long, "event_id"])
            sub = commit_sub[commit_sub["event_id"].isin(ids_in_group)]
            if len(sub) == 0:
                continue
            ids, y = _profile_matrix(sub, rels)
            mu, sem = _mean_sem(y)
            n_ev = int(len(ids))
            ax.plot(rels, mu, lw=2, label=f"{label} bouts (n={n_ev})", color=color)
            ax.fill_between(rels, mu - sem, mu + sem, alpha=0.15, color=color)
        ax.axvline(0, color="black", lw=1, ls="--")
        ax.set_xlabel("Real steps from commit")
        ax.set_ylabel(f"Delta {metric_label}\nfrom event baseline")
        ax.set_title(_title("D", "Cross2 Short vs. Long commit profile"))
        ax.legend(frameon=False, fontsize=9)
        ax.grid(alpha=0.3)
    else:
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_fig_2_2_1_short_long(
    bouts: pd.DataFrame,
    temporal: pd.DataFrame,
    prepost: pd.DataFrame,
    out_path: Path,
    metric: str = "entropy_actor",
) -> None:
    """
    Fig 2-2.1: same layout as Fig 2-2, but every panel is split by Short/Long bouts.
    """
    bouts_c2 = annotate_cross2_bouts(bouts)
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2))
    axes = axes.ravel()
    metric_label = _metric_label(metric)
    class_specs = [("Short", "#1f77b4"), ("Long", "#ff7f0e")]

    def _profile_matrix(sub: pd.DataFrame, rels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pivot = sub.pivot_table(index="event_id", columns="rel_step", values=metric, aggfunc="mean")
        pivot = pivot.reindex(columns=rels)
        y = pivot.to_numpy(dtype=float)
        baseline_cols = [r for r in PROFILE_BASELINE_RELS if r in pivot.columns]
        if not baseline_cols:
            baseline_cols = [r for r in pivot.columns if r < 0]
        if baseline_cols:
            base = pivot[baseline_cols].to_numpy(dtype=float)
            y = y - np.nanmean(base, axis=1, keepdims=True)
        return pivot.index.to_numpy(), y

    def _plot_anchor_profile(ax, anchor: str, title: str) -> None:
        sub_anchor = temporal[temporal["anchor"] == anchor]
        if metric not in sub_anchor.columns or len(sub_anchor) == 0:
            ax.set_axis_off()
            return
        rels = np.sort(sub_anchor["rel_step"].unique())
        for cls, color in class_specs:
            ids_in_group = set(bouts_c2.loc[bouts_c2["bout_class"] == cls, "event_id"])
            sub = sub_anchor[sub_anchor["event_id"].isin(ids_in_group)]
            if len(sub) == 0:
                continue
            ids, y = _profile_matrix(sub, rels)
            mu, sem = _mean_sem(y)
            ax.plot(rels, mu, lw=2, label=f"{cls} bouts (n={len(ids)})", color=color)
            ax.fill_between(rels, mu - sem, mu + sem, alpha=0.16, color=color)
        ax.axvline(0, color="black", lw=1, ls="--")
        ax.set_xlabel(f"Real steps from {anchor}")
        ax.set_ylabel(f"Delta {metric_label}\nfrom event baseline")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.3)

    _plot_anchor_profile(axes[0], "onset", "Fig 2-2.1A: Onset-aligned profile by class")
    _plot_anchor_profile(axes[1], "commit", "Fig 2-2.1B: Commit-aligned profile by class")

    ax = axes[2]
    pre_col = f"{metric}_pre"
    delta_col = f"{metric}_delta"
    prepost_c2 = prepost.merge(
        bouts_c2[["event_id", "bout_class"]],
        on="event_id",
        how="left",
    )
    if len(prepost_c2) > 0 and pre_col in prepost_c2.columns and delta_col in prepost_c2.columns:
        for cls, color in class_specs:
            sub = prepost_c2[prepost_c2["bout_class"] == cls]
            x = sub[pre_col].to_numpy(dtype=float)
            y = -sub[delta_col].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            ax.scatter(x[mask], y[mask], s=9, alpha=0.22, color=color, label=f"{cls} points (n={mask.sum()})")
            if mask.sum() > 5 and np.nanstd(x[mask]) > 0:
                slope, intercept, r, p, _ = scipy_stats.linregress(x[mask], y[mask])
                xx = np.linspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 100)
                ax.plot(xx, slope * xx + intercept, color=color, lw=2,
                        label=f"{cls}: r={r:.3f}, p={p:.3g}")
        ax.axhline(0, color="black", lw=1, ls="--")
        ax.set_xlabel(f"Pre-withholding {metric_label.lower()}")
        ax.set_ylabel(f"Confidence gain\n(-delta {metric_label.lower()})")
        ax.set_title("Fig 2-2.1C: Uncertainty vs. confidence gain by class")
        ax.legend(frameon=False, fontsize=7)
        ax.grid(alpha=0.3)
    else:
        ax.set_axis_off()

    _plot_anchor_profile(axes[3], "commit", "Fig 2-2.1D: Cross2 Short vs. Long commit profile")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def compute_short_long_metric_tables(
    bouts: pd.DataFrame,
    prepost: pd.DataFrame,
    metrics: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bouts_c2 = annotate_cross2_bouts(bouts)
    keep = ["event_id", "subject", "session", "game", "length_real_steps", "cross2", "is_long", "bout_class"]
    event = prepost.merge(bouts_c2[keep], on="event_id", how="left")

    summary_rows = []
    test_rows = []
    phases = ["pre", "onset", "commit", "delta"]
    for metric_name in metrics:
        for phase in phases:
            col = f"{metric_name}_{phase}"
            if col not in event.columns:
                continue
            for cls in ["Short", "Long"]:
                vals = event.loc[event["bout_class"] == cls, col].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                summary_rows.append({
                    "metric": metric_name,
                    "phase": phase,
                    "bout_class": cls,
                    "n": int(len(vals)),
                    "mean": float(np.mean(vals)) if len(vals) else np.nan,
                    "sem": float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else np.nan,
                    "median": float(np.median(vals)) if len(vals) else np.nan,
                })
            short = event.loc[event["bout_class"] == "Short", col].to_numpy(dtype=float)
            long = event.loc[event["bout_class"] == "Long", col].to_numpy(dtype=float)
            short = short[np.isfinite(short)]
            long = long[np.isfinite(long)]
            t_stat, p_val = (np.nan, np.nan)
            if len(short) > 1 and len(long) > 1:
                t_stat, p_val = scipy_stats.ttest_ind(short, long, equal_var=False, nan_policy="omit")
            test_rows.append({
                "metric": metric_name,
                "phase": phase,
                "n_short": int(len(short)),
                "n_long": int(len(long)),
                "mean_short": float(np.mean(short)) if len(short) else np.nan,
                "mean_long": float(np.mean(long)) if len(long) else np.nan,
                "long_minus_short": (
                    float(np.mean(long) - np.mean(short)) if len(short) and len(long) else np.nan
                ),
                "welch_t": float(t_stat) if np.isfinite(t_stat) else np.nan,
                "p_value": float(p_val) if np.isfinite(p_val) else np.nan,
            })
    return event, pd.DataFrame(summary_rows), pd.DataFrame(test_rows)


def plot_fig_2_3_short_long_metrics(
    event_metrics: pd.DataFrame,
    out_path: Path,
    plot_metrics: Optional[Sequence[Tuple[str, str]]] = None,
) -> None:
    """
    Fig 2-3: decision-aligned Short/Long trajectories for multiple metrics.
    Actor entropy/margin use the real-action policy row; tree metrics use the
    preceding status==2 row.
    """
    if plot_metrics is None:
        plot_metrics = [
            ("entropy_actor", "Entropy"),
            ("margin_actor", "Policy margin"),
            ("q_gap_prev_s2", "Q-gap (prev-s2)"),
            ("rollout_spread_prev_s2", "Rollout spread (prev-s2)"),
        ]
    phases = ["pre", "onset", "commit"]
    x = np.arange(len(phases))
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.0))
    axes = axes.ravel()
    for i, (ax, (metric_name, label)) in enumerate(zip(axes, plot_metrics)):
        cols = [f"{metric_name}_{phase}" for phase in phases]
        if any(c not in event_metrics.columns for c in cols):
            ax.set_axis_off()
            continue
        for cls, color in [("Short", "#1f77b4"), ("Long", "#ff7f0e")]:
            sub = event_metrics.loc[event_metrics["bout_class"] == cls, cols].to_numpy(dtype=float)
            if len(sub) == 0:
                continue
            mu, sem = _mean_sem(sub)
            ax.errorbar(x, mu, yerr=sem, marker="o", lw=2, capsize=3,
                        label=f"{cls} (n={len(sub)})", color=color)
        ax.set_xticks(x, ["Pre", "Onset", "Commit"])
        ax.set_ylabel(label)
        ax.set_title(f"Fig 2-3{chr(ord('A') + i)}: {label} by Cross2 class")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


# ─── Auxiliary sequential structure tables ───────────────────────────────────

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
    Sample metrics at n_bins normalized positions within each bout
    (0 = onset, 1 = final NOOP step). Enables within-bout dynamics.
    """
    bouts_c2 = annotate_cross2_bouts(bouts)
    grouped = {
        k: v.sort_values("real_pos").reset_index(drop=True)
        for k, v in df_real.groupby("file", sort=False)
    }
    metric_cols = ["entropy_actor", "margin_actor", "q_gap", "rollout_spread"]
    rows = []
    for _, b in bouts_c2.iterrows():
        g = grouped[b["file"]]
        s, e = int(b["start_pos"]), int(b["end_pos"])
        bout_len = e - s + 1
        if bout_len < 2:
            continue
        positions = np.round(np.linspace(s, e, n_bins)).astype(int)
        positions = np.clip(positions, 0, len(g) - 1)
        row = {
            "event_id": b["event_id"],
            "file": b["file"],
            "subject": b["subject"],
            "game": b["game"],
            "bout_length": bout_len,
            "cross2": int(b["cross2"]),
            "is_long": int(b["is_long"]),
            "bout_class": b["bout_class"],
        }
        for metric_name in metric_cols:
            for i, pos in enumerate(positions):
                row[f"{metric_name}_pos{i}"] = float(g.loc[pos, metric_name])
        rows.append(row)
    return pd.DataFrame(rows)


def compute_curaction_stability(df_real: pd.DataFrame, bouts: pd.DataFrame) -> pd.DataFrame:
    """
    Per bout: how often the internal cur_action changes across the NOOP steps,
    and whether the final/modal internal preference matches the overt commit.
    Returns change_rate = n_changes / (bout_length - 1).
    """
    bouts_c2 = annotate_cross2_bouts(bouts)
    grouped = {
        k: v.sort_values("real_pos").reset_index(drop=True)
        for k, v in df_real.groupby("file", sort=False)
    }
    rows = []
    for _, b in bouts_c2.iterrows():
        g = grouped[b["file"]]
        s, e = int(b["start_pos"]), int(b["end_pos"])
        commit = int(b["commit_pos"])
        actions = g.loc[s:e, "imag_cur_action_final"].to_numpy(dtype=int)
        n_changes = int(np.sum(np.diff(actions) != 0))
        bout_len = e - s + 1
        vals, counts = np.unique(actions, return_counts=True)
        modal_action = int(vals[np.argmax(counts)]) if len(vals) > 0 else -1
        final_action = int(actions[-1]) if len(actions) > 0 else -1
        commit_action = int(g.loc[commit, "human_action"]) if 0 <= commit < len(g) else -1
        commit_search_final = int(g.loc[commit, "imag_cur_action_final"]) if 0 <= commit < len(g) else -1
        commit_search_modal = int(g.loc[commit, "imag_cur_action_modal"]) if 0 <= commit < len(g) else -1
        rows.append({
            "event_id": b["event_id"],
            "file": b["file"],
            "subject": b["subject"],
            "game": b["game"],
            "bout_length": bout_len,
            "cross2": int(b["cross2"]),
            "is_long": int(b["is_long"]),
            "bout_class": b["bout_class"],
            "n_changes": n_changes,
            "change_rate": n_changes / max(bout_len - 1, 1),
            "n_unique_cur_actions": int(len(vals)),
            "dominant_action_fraction": float(np.max(counts) / max(len(actions), 1)) if len(vals) else np.nan,
            "final_cur_action": final_action,
            "modal_cur_action": modal_action,
            "mean_imag_nonnoop_fraction": float(g.loc[s:e, "imag_nonnoop_fraction"].mean()),
            "commit_action": commit_action,
            "commit_search_final_action": commit_search_final,
            "commit_search_modal_action": commit_search_modal,
            "commit_matches_final_cur_action": int(commit_action == final_action),
            "commit_matches_modal_cur_action": int(commit_action == modal_action),
            "commit_matches_commit_search_final": int(commit_action == commit_search_final),
            "commit_matches_commit_search_modal": int(commit_action == commit_search_modal),
        })
    return pd.DataFrame(rows)


def compute_bout_commit_dynamics(df_real: pd.DataFrame, bouts: pd.DataFrame) -> pd.DataFrame:
    """
    Stage-wise dynamics from pre-bout state through NOOP maintenance to action commit.
    This is the direct NOOP -> action-commit bridge retained as a CSV table.
    """
    bouts_c2 = annotate_cross2_bouts(bouts)
    grouped = {
        k: v.sort_values("real_pos").reset_index(drop=True)
        for k, v in df_real.groupby("file", sort=False)
    }
    metric_cols = [
        "entropy_actor", "margin_actor", "q_gap", "rollout_spread",
        "entropy_actor_prev_s2", "margin_actor_prev_s2", "q_gap_prev_s2",
    ]
    rows = []
    for _, b in bouts_c2.iterrows():
        g = grouped[b["file"]]
        s, e = int(b["start_pos"]), int(b["end_pos"])
        pre, commit = int(b["pre_pos"]), int(b["commit_pos"])
        mid = int(round((s + e) / 2))
        stages = {
            "pre": pre,
            "onset": s,
            "mid": mid,
            "final_noop": e,
            "commit": commit,
        }
        row = {
            "event_id": b["event_id"],
            "file": b["file"],
            "subject": b["subject"],
            "session": b["session"],
            "game": b["game"],
            "bout_length": int(b["length_real_steps"]),
            "cross2": int(b["cross2"]),
            "is_long": int(b["is_long"]),
            "bout_class": b["bout_class"],
            "commit_action": int(g.loc[commit, "human_action"]),
        }
        for stage, pos in stages.items():
            row[f"{stage}_pos"] = int(pos)
            row[f"{stage}_cur_action"] = int(g.loc[pos, "cur_action_id"])
            row[f"{stage}_imag_cur_action_final"] = int(g.loc[pos, "imag_cur_action_final"])
            row[f"{stage}_imag_cur_action_modal"] = int(g.loc[pos, "imag_cur_action_modal"])
            row[f"{stage}_imag_nonnoop_fraction"] = float(g.loc[pos, "imag_nonnoop_fraction"])
            for metric_name in metric_cols:
                if metric_name in g.columns:
                    row[f"{metric_name}_{stage}"] = float(g.loc[pos, metric_name])

        actions = g.loc[s:e, "imag_cur_action_final"].to_numpy(dtype=int)
        vals, counts = np.unique(actions, return_counts=True)
        modal_action = int(vals[np.argmax(counts)]) if len(vals) else -1
        final_action = int(actions[-1]) if len(actions) else -1
        row["n_cur_action_changes"] = int(np.sum(np.diff(actions) != 0))
        row["cur_action_change_rate"] = row["n_cur_action_changes"] / max(len(actions) - 1, 1)
        row["n_unique_cur_actions"] = int(len(vals))
        row["dominant_action_fraction"] = float(np.max(counts) / max(len(actions), 1)) if len(vals) else np.nan
        row["final_cur_action"] = final_action
        row["modal_cur_action"] = modal_action
        row["mean_imag_nonnoop_fraction"] = float(g.loc[s:e, "imag_nonnoop_fraction"].mean())
        row["commit_search_final_action"] = int(g.loc[commit, "imag_cur_action_final"])
        row["commit_search_modal_action"] = int(g.loc[commit, "imag_cur_action_modal"])
        row["commit_matches_final_cur_action"] = int(row["commit_action"] == final_action)
        row["commit_matches_modal_cur_action"] = int(row["commit_action"] == modal_action)
        row["commit_matches_commit_search_final"] = int(row["commit_action"] == row["commit_search_final_action"])
        row["commit_matches_commit_search_modal"] = int(row["commit_action"] == row["commit_search_modal_action"])

        for metric_name in metric_cols:
            onset_col = f"{metric_name}_onset"
            final_col = f"{metric_name}_final_noop"
            commit_col = f"{metric_name}_commit"
            pre_col = f"{metric_name}_pre"
            if all(c in row for c in [onset_col, final_col, commit_col, pre_col]):
                row[f"{metric_name}_noop_internal_delta"] = row[final_col] - row[onset_col]
                row[f"{metric_name}_commit_delta"] = row[commit_col] - row[final_col]
                row[f"{metric_name}_total_delta"] = row[commit_col] - row[pre_col]
        rows.append(row)
    return pd.DataFrame(rows)


# ─── Section 2-4: NOOP onset → commit policy trajectory ──────────────────────

def compute_episode_stats(df_real: pd.DataFrame) -> pd.DataFrame:
    """Per-chunk (episode) aggregate stats: NOOP ratio, uncertainty coupling, reward."""
    rows = []
    for file_id, g in df_real.groupby("file", sort=False):
        meta = parse_file_meta(Path(file_id))
        g = g.sort_values("real_pos")
        ent_r, ent_slope = _safe_corr_slope(g["entropy_actor"], g["is_human_noop"])
        qgap_r, qgap_slope = _safe_corr_slope(g["q_gap"], g["is_human_noop"])
        rows.append({
            "file": file_id,
            "subject": meta.subject,
            "session": meta.session,
            "block": meta.block,
            "game": meta.game,
            "chunk": _safe_int(meta.chunk),
            "n_steps": len(g),
            "noop_ratio": float(g["is_human_noop"].mean()),
            "mean_entropy": float(g["entropy_actor"].mean()),
            "mean_margin": float(g["margin_actor"].mean()),
            "mean_q_gap": float(g["q_gap"].mean()),
            "total_reward": float(g["env_return"].sum()),
            "entropy_noop_r": ent_r,
            "entropy_noop_slope": ent_slope,
            "qgap_noop_r": qgap_r,
            "qgap_noop_slope": qgap_slope,
        })
    df = pd.DataFrame(rows).sort_values(["subject", "game", "session", "block", "chunk"]).reset_index(drop=True)
    df["episode_idx"] = df.groupby(["subject", "game", "session"]).cumcount()
    df["cumulative_reward"] = df.groupby(["subject", "game", "session"])["total_reward"].cumsum()
    return df


def compute_half_session_stats(df_real: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for file_id, g in df_real.groupby("file", sort=False):
        g = g.sort_values("real_pos").reset_index(drop=True)
        meta = parse_file_meta(Path(file_id))
        mid = len(g) // 2
        for half, sub in [("Early", g.iloc[:mid]), ("Late", g.iloc[mid:])]:
            ent_r, ent_slope = _safe_corr_slope(sub["entropy_actor"], sub["is_human_noop"])
            qgap_r, qgap_slope = _safe_corr_slope(sub["q_gap"], sub["is_human_noop"])
            rows.append({
                "file": file_id,
                "subject": meta.subject,
                "session": meta.session,
                "game": meta.game,
                "half": half,
                "n_steps": int(len(sub)),
                "noop_ratio": float(sub["is_human_noop"].mean()) if len(sub) else np.nan,
                "entropy_noop_r": ent_r,
                "entropy_noop_slope": ent_slope,
                "qgap_noop_r": qgap_r,
                "qgap_noop_slope": qgap_slope,
            })
    return pd.DataFrame(rows)


def compute_noop_commit_policy_trajectory(
    df_real: pd.DataFrame,
    bouts: pd.DataFrame,
    n_bins: int = 50,
) -> pd.DataFrame:
    """
    For every NOOP bout, sample real-step actor metrics from NOOP onset through
    the first real-action commit and interpolate each bout to a 0..1 axis.
    """
    bouts_c2 = annotate_cross2_bouts(bouts)
    grouped = {
        k: v.sort_values("real_pos").reset_index(drop=True)
        for k, v in df_real.groupby("file", sort=False)
    }
    x_grid = np.linspace(0.0, 1.0, n_bins)
    rows = []
    for _, b in bouts_c2.iterrows():
        g = grouped[b["file"]]
        s, commit = int(b["start_pos"]), int(b["commit_pos"])
        if s < 0 or commit >= len(g) or commit < s:
            continue
        seg = g.loc[s:commit, ["entropy_actor", "margin_actor"]]
        if len(seg) < 2:
            continue
        x_src = np.linspace(0.0, 1.0, len(seg))
        row = {
            "event_id": b["event_id"],
            "file": b["file"],
            "subject": b["subject"],
            "game": b["game"],
            "bout_length": int(b["length_real_steps"]),
            "cross2": int(b["cross2"]),
            "is_long": int(b["is_long"]),
            "bout_class": b["bout_class"],
            "n_real_steps_onset_to_commit": int(len(seg)),
        }
        for metric_name in ["entropy_actor", "margin_actor"]:
            vals = seg[metric_name].to_numpy(dtype=float)
            finite = np.isfinite(vals)
            if finite.sum() >= 2:
                interp = np.interp(x_grid, x_src[finite], vals[finite])
            elif finite.sum() == 1:
                interp = np.full_like(x_grid, vals[finite][0], dtype=float)
            else:
                interp = np.full_like(x_grid, np.nan, dtype=float)
            for i, val in enumerate(interp):
                row[f"{metric_name}_pos{i}"] = float(val)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_fig_2_4_noop_commit_policy_trajectory(
    traj: pd.DataFrame,
    out_path: Path,
) -> None:
    """
    Fig 2-4: normalized real-step policy trajectory from NOOP onset to commit.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), sharex=True)
    metric_specs = [
        ("entropy_actor", "Actor entropy", "Fig 2-4A: Entropy trajectory"),
        ("margin_actor", "Policy margin", "Fig 2-4B: Policy margin trajectory"),
    ]

    for ax, (metric_name, label, title) in zip(axes, metric_specs):
        pos_cols = [c for c in traj.columns if c.startswith(f"{metric_name}_pos")]
        pos_cols = sorted(pos_cols, key=lambda c: int(c.rsplit("pos", 1)[1]))
        if len(traj) == 0 or not pos_cols:
            ax.set_axis_off()
            continue
        x = np.linspace(0.0, 1.0, len(pos_cols))
        for is_long, bout_label, color in [(0, "Short", "#1f77b4"), (1, "Long", "#ff7f0e")]:
            sub = traj.loc[traj["is_long"] == is_long, pos_cols].to_numpy(dtype=float)
            if len(sub) == 0:
                continue
            mu, sem = _mean_sem(sub)
            ax.plot(x, mu, lw=2, label=f"{bout_label} bouts (n={len(sub)})", color=color)
            ax.fill_between(x, mu - sem, mu + sem, alpha=0.18, color=color)
        ax.set_xlabel("Normalized real-step position\n(0 = NOOP onset, 1 = commit)")
        ax.set_ylabel(label)
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=9)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


# ─── Section 2-5: Commit hazard from actor and search features ───────────────

def compute_commit_hazard_table(df_real: pd.DataFrame, bouts: pd.DataFrame) -> pd.DataFrame:
    """
    One row per NOOP real step. The target is whether the next real step is the
    overt commit action that terminates the bout.
    """
    bouts_c2 = annotate_cross2_bouts(bouts)
    grouped = {
        k: v.sort_values("real_pos").reset_index(drop=True)
        for k, v in df_real.groupby("file", sort=False)
    }
    feature_cols = [
        "entropy_actor",
        "q_gap_prev_s2",
        "rollout_spread_prev_s2",
        "q_gap_imag_mean",
        "q_gap_imag_final",
        "q_gap_imag_slope",
        "rollout_spread_imag_mean",
        "rollout_spread_imag_final",
        "rollout_spread_imag_slope",
        "imag_cur_action_change_rate",
        "imag_cur_action_entropy",
        "imag_s2_n_steps",
    ]
    rows = []
    for _, b in bouts_c2.iterrows():
        g = grouped[b["file"]]
        s, e = int(b["start_pos"]), int(b["end_pos"])
        commit = int(b["commit_pos"])
        if s < 0 or e >= len(g) or commit >= len(g):
            continue
        bout_len = int(e - s + 1)
        for pos in range(s, e + 1):
            elapsed = int(pos - s)
            row = {
                "event_id": b["event_id"],
                "file": b["file"],
                "subject": b["subject"],
                "game": b["game"],
                "real_pos": int(pos),
                "global_idx": int(g.loc[pos, "global_idx"]),
                "bout_length": bout_len,
                "cross2": int(b["cross2"]),
                "is_long": int(b["is_long"]),
                "bout_class": b["bout_class"],
                "elapsed_noop_steps": elapsed,
                "log_elapsed_noop_steps": float(np.log1p(elapsed)),
                "commit_next": int(pos == e),
            }
            for col in feature_cols:
                row[col] = float(g.loc[pos, col]) if col in g.columns else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def _roc_auc_score_np(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    mask = np.isfinite(score)
    y, score = y[mask], score[mask]
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = pd.Series(score).rank(method="average").to_numpy()
    pos_rank_sum = float(np.sum(ranks[y == 1]))
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _log_loss_np(y: np.ndarray, prob: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    prob = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(prob) + (1 - y) * np.log(1 - prob)))


def _standardize_train_test(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = np.nanmean(x_train, axis=0)
    sd = np.nanstd(x_train, axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return (x_train - mu) / sd, (x_test - mu) / sd, mu, sd


def _fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_i = np.column_stack([np.ones(len(x)), x])

    def objective(beta: np.ndarray) -> Tuple[float, np.ndarray]:
        z = x_i @ beta
        loss = np.mean(np.logaddexp(0.0, z) - y * z)
        loss += 0.5 * l2 * np.sum(beta[1:] ** 2) / max(len(y), 1)
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))
        grad = x_i.T @ (p - y) / max(len(y), 1)
        grad[1:] += l2 * beta[1:] / max(len(y), 1)
        return float(loss), grad

    init = np.zeros(x_i.shape[1], dtype=float)
    if 0 < np.mean(y) < 1:
        init[0] = float(np.log(np.mean(y) / (1 - np.mean(y))))
    res = scipy_optimize.minimize(
        lambda b: objective(b)[0],
        init,
        jac=lambda b: objective(b)[1],
        method="BFGS",
        options={"maxiter": 500, "gtol": 1e-6},
    )
    return np.asarray(res.x, dtype=float)


def _predict_logistic(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    x_i = np.column_stack([np.ones(len(x)), x])
    z = x_i @ beta
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def compute_commit_hazard_models(
    hazard: pd.DataFrame,
    n_splits: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    models = [
        ("Elapsed only", ["log_elapsed_noop_steps"]),
        ("Actor entropy", ["log_elapsed_noop_steps", "entropy_actor"]),
        ("Prev-s2 tree", ["log_elapsed_noop_steps", "q_gap_prev_s2", "rollout_spread_prev_s2"]),
        ("Actor + prev-s2 tree", [
            "log_elapsed_noop_steps", "entropy_actor", "q_gap_prev_s2", "rollout_spread_prev_s2",
        ]),
        ("Imag search summary", [
            "log_elapsed_noop_steps",
            "q_gap_imag_mean", "q_gap_imag_final", "q_gap_imag_slope",
            "rollout_spread_imag_mean", "rollout_spread_imag_final", "rollout_spread_imag_slope",
            "imag_cur_action_change_rate", "imag_cur_action_entropy",
        ]),
        ("Actor + all search", [
            "log_elapsed_noop_steps", "entropy_actor", "q_gap_prev_s2", "rollout_spread_prev_s2",
            "q_gap_imag_mean", "q_gap_imag_final", "q_gap_imag_slope",
            "rollout_spread_imag_mean", "rollout_spread_imag_final", "rollout_spread_imag_slope",
            "imag_cur_action_change_rate", "imag_cur_action_entropy",
        ]),
    ]
    summary_rows = []
    coef_rows = []
    event_codes = pd.factorize(hazard["event_id"])[0]

    for model_name, predictors in models:
        cols = ["commit_next", "event_id"] + predictors
        data = hazard[cols].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        if len(data) == 0 or data["commit_next"].nunique() < 2:
            continue
        y = data["commit_next"].to_numpy(dtype=int)
        x = data[predictors].to_numpy(dtype=float)
        folds = pd.factorize(data["event_id"])[0] % min(n_splits, max(data["event_id"].nunique(), 1))
        pred = np.full(len(data), np.nan, dtype=float)
        for fold in np.unique(folds):
            train = folds != fold
            test = folds == fold
            if y[train].sum() == 0 or y[train].sum() == train.sum():
                continue
            x_train, x_test, _, _ = _standardize_train_test(x[train], x[test])
            beta = _fit_logistic(x_train, y[train])
            pred[test] = _predict_logistic(x_test, beta)

        valid_pred = np.isfinite(pred)
        auc = _roc_auc_score_np(y[valid_pred], pred[valid_pred])
        log_loss = _log_loss_np(y[valid_pred], pred[valid_pred])
        summary_rows.append({
            "model": model_name,
            "n": int(valid_pred.sum()),
            "n_commit_next": int(np.sum(y[valid_pred] == 1)),
            "cv_auc": auc,
            "cv_log_loss": log_loss,
            "n_predictors": len(predictors),
        })

        x_std, _, _, _ = _standardize_train_test(x, x)
        beta_full = _fit_logistic(x_std, y)
        for predictor, coef in zip(predictors, beta_full[1:]):
            coef_rows.append({
                "model": model_name,
                "predictor": predictor,
                "standardized_log_odds": float(coef),
            })

    summary = pd.DataFrame(summary_rows)
    if len(summary) > 0:
        base = summary.loc[summary["model"] == "Elapsed only", "cv_log_loss"]
        base_loss = float(base.iloc[0]) if len(base) else np.nan
        summary["log_loss_improvement_vs_elapsed"] = base_loss - summary["cv_log_loss"]
    return summary, pd.DataFrame(coef_rows)


def plot_fig_2_5_commit_hazard(
    model_summary: pd.DataFrame,
    coef_summary: pd.DataFrame,
    out_path: Path,
) -> None:
    """
    Fig 2-5: commit hazard ablation plus standardized coefficients.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.2))
    axes = axes.ravel()
    if len(model_summary) == 0:
        for ax in axes:
            ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        return

    ordered = model_summary.sort_values("cv_auc", ascending=False).reset_index(drop=True)
    colors = ["#1f77b4" if m != "Elapsed only" else "#7f7f7f" for m in ordered["model"]]

    ax = axes[0]
    ax.barh(np.arange(len(ordered)), ordered["cv_auc"], color=colors, alpha=0.8)
    ax.set_yticks(np.arange(len(ordered)), ordered["model"])
    ax.invert_yaxis()
    ax.set_xlabel("Cross-validated AUROC")
    ax.set_title("Fig 2-5A: Commit hazard model comparison")
    ax.grid(axis="x", alpha=0.3)

    ax = axes[1]
    ordered_loss = model_summary.sort_values("log_loss_improvement_vs_elapsed", ascending=True)
    ax.barh(np.arange(len(ordered_loss)), ordered_loss["log_loss_improvement_vs_elapsed"], color="#2ca02c", alpha=0.75)
    ax.set_yticks(np.arange(len(ordered_loss)), ordered_loss["model"])
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_xlabel("CV log-loss improvement vs elapsed-only")
    ax.set_title("Fig 2-5B: Added predictive value")
    ax.grid(axis="x", alpha=0.3)

    coef_labels = {
        "log_elapsed_noop_steps": "elapsed",
        "entropy_actor": "actor entropy",
        "q_gap_prev_s2": "prev-s2 Q-gap",
        "rollout_spread_prev_s2": "prev-s2 spread",
        "q_gap_imag_mean": "Q-gap mean",
        "q_gap_imag_final": "Q-gap final",
        "q_gap_imag_slope": "Q-gap slope",
        "rollout_spread_imag_mean": "spread mean",
        "rollout_spread_imag_final": "spread final",
        "rollout_spread_imag_slope": "spread slope",
        "imag_cur_action_change_rate": "action switch",
        "imag_cur_action_entropy": "action entropy",
    }

    for ax, model_name, title in [
        (axes[2], "Actor + prev-s2 tree", "Fig 2-5C: Actor + final search state"),
        (axes[3], "Imag search summary", "Fig 2-5D: Imaginary search trajectory"),
    ]:
        sub = coef_summary[coef_summary["model"] == model_name].copy()
        if len(sub) == 0:
            ax.set_axis_off()
            continue
        sub["label"] = [coef_labels.get(p, p) for p in sub["predictor"]]
        sub = sub.sort_values("standardized_log_odds")
        vals = sub["standardized_log_odds"].to_numpy(dtype=float)
        bar_colors = np.where(vals >= 0, "#d62728", "#1f77b4")
        ax.barh(np.arange(len(sub)), vals, color=bar_colors, alpha=0.75)
        ax.set_yticks(np.arange(len(sub)), sub["label"])
        ax.axvline(0, color="black", lw=1)
        ax.set_xlabel("Standardized log-odds coefficient")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


# ─── Summary stats ────────────────────────────────────────────────────────────

def compute_section2_summary(bouts: pd.DataFrame) -> pd.DataFrame:
    bouts_c2 = annotate_cross2_bouts(bouts)
    rows = []
    for (subject, game), g in bouts_c2.groupby(["subject", "game"], sort=True):
        lengths = g["length_real_steps"].to_numpy(dtype=int)
        c2 = int(g["cross2"].iloc[0])
        short_mask = lengths < c2
        long_mask = ~short_mask
        rows.append({
            "subject": subject,
            "game": game,
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
            "mean_margin_commit_short": float(g.loc[short_mask, "margin_at_commit"].mean()) if short_mask.sum() > 0 else np.nan,
            "mean_margin_commit_long": float(g.loc[long_mask, "margin_at_commit"].mean()) if long_mask.sum() > 0 else np.nan,
            "mean_q_gap_onset_short": float(g.loc[short_mask, "q_gap_at_onset"].mean()) if short_mask.sum() > 0 else np.nan,
            "mean_q_gap_onset_long": float(g.loc[long_mask, "q_gap_at_onset"].mean()) if long_mask.sum() > 0 else np.nan,
        })
    return pd.DataFrame(rows)


# ─── Main pipeline ────────────────────────────────────────────────────────────

def run_section2(
    input_dir: Path,
    out_dir: Path,
    cross2_summary_path: Optional[Path] = None,
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
    cross2_override = load_cross2_summary(cross2_summary_path)
    bouts = annotate_cross2_bouts(bouts, cross2_override=cross2_override)

    prepost, temporal = build_event_tables(
        df_real, bouts,
        window_pre=window_pre, window_post=window_post,
        metric_cols=[
            "entropy_actor", "margin_actor", "q_gap",
            "entropy_actor_prev_s2", "margin_actor_prev_s2", "q_gap_prev_s2",
            "rollout_spread", "rollout_spread_prev_s2",
        ],
    )
    traj = compute_bout_internal_trajectory(df_real, bouts)
    stability = compute_curaction_stability(df_real, bouts)
    dynamics = compute_bout_commit_dynamics(df_real, bouts)
    episode_stats = compute_episode_stats(df_real)
    half_stats = compute_half_session_stats(df_real)
    noop_commit_policy_traj = compute_noop_commit_policy_trajectory(df_real, bouts)
    commit_hazard = compute_commit_hazard_table(df_real, bouts)
    hazard_model_summary, hazard_coef_summary = compute_commit_hazard_models(commit_hazard)
    short_long_event, short_long_summary, short_long_tests = compute_short_long_metric_tables(
        bouts,
        prepost,
        metrics=[
            "entropy_actor", "margin_actor", "q_gap",
            "entropy_actor_prev_s2", "margin_actor_prev_s2", "q_gap_prev_s2",
            "rollout_spread", "rollout_spread_prev_s2",
        ],
    )
    summary = compute_section2_summary(bouts)

    # Save CSVs
    df_real.to_csv(out_dir / "real_step_metrics.csv", index=False)
    bouts.to_csv(out_dir / "noop_bouts.csv", index=False)
    prepost.to_csv(out_dir / "event_prepost.csv", index=False)
    temporal.to_csv(out_dir / "event_temporal.csv", index=False)
    short_long_event.to_csv(out_dir / "short_long_event_metrics.csv", index=False)
    short_long_summary.to_csv(out_dir / "short_long_metric_summary.csv", index=False)
    short_long_tests.to_csv(out_dir / "short_long_metric_tests.csv", index=False)
    traj.to_csv(out_dir / "bout_internal_trajectory.csv", index=False)
    stability.to_csv(out_dir / "bout_action_stability.csv", index=False)
    dynamics.to_csv(out_dir / "bout_commit_dynamics.csv", index=False)
    episode_stats.to_csv(out_dir / "episode_stats.csv", index=False)
    half_stats.to_csv(out_dir / "half_session_stats.csv", index=False)
    noop_commit_policy_traj.to_csv(out_dir / "noop_commit_policy_trajectory.csv", index=False)
    commit_hazard.to_csv(out_dir / "commit_hazard_steps.csv", index=False)
    hazard_model_summary.to_csv(out_dir / "commit_hazard_model_summary.csv", index=False)
    hazard_coef_summary.to_csv(out_dir / "commit_hazard_coefficients.csv", index=False)
    summary.to_csv(out_dir / "section2_summary.csv", index=False)

    # Generate figures
    plot_fig_2_1c(bouts, out_dir / "fig_2_1C_onset_entropy_bout_length.png")
    plot_fig_2_2(bouts, temporal, prepost, out_dir / "fig_2_2_temporal_profiles.png")
    plot_fig_2_2_1_short_long(
        bouts,
        temporal,
        prepost,
        out_dir / "fig_2_2_1_temporal_profiles_short_long.png",
    )
    plot_fig_2_3_short_long_metrics(short_long_event, out_dir / "fig_2_3_short_long_metrics.png")
    plot_fig_2_4_noop_commit_policy_trajectory(
        noop_commit_policy_traj,
        out_dir / "fig_2_4_noop_commit_policy_trajectory.png",
    )
    plot_fig_2_5_commit_hazard(
        hazard_model_summary,
        hazard_coef_summary,
        out_dir / "fig_2_5_commit_hazard.png",
    )

    print(f"Section 2 analysis complete → {out_dir}")
    print(summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Section 2: NOOP Bout Structure Analysis")
    p.add_argument("--input-dir", type=Path, default=Path("test/sub001/ses-04"),
                   help="Directory containing .npy files")
    p.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "outputs" / "02_structure_analysis",
                   help="Output directory for figures and CSVs")
    p.add_argument("--cross2-summary", type=Path,
                   default=Path(__file__).parent / "outputs" / "01_behavioral_analysis" / "1-7_bout_short_long_summary.csv",
                   help="Optional subject×game Cross2 summary from Section 1 survival analysis")
    p.add_argument("--window-pre", type=int, default=6, help="Temporal window before onset/commit")
    p.add_argument("--window-post", type=int, default=6, help="Temporal window after onset/commit")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_section2(
        input_dir=args.input_dir,
        out_dir=args.out_dir,
        cross2_summary_path=args.cross2_summary,
        window_pre=args.window_pre,
        window_post=args.window_post,
    )


if __name__ == "__main__":
    main()
