#!/usr/bin/env python3
"""
Analysis pipeline for human withholding behavior on filtered video-stat files.

Default input is limited to:
    /home/jmme425/thinker/test/sub001/ses-04

Outputs:
  - CSV tables (real-step metrics, NOOP bouts, event tables, matched-control table)
  - Figure PNGs for the Figure 1~4 schema
  - JSON summary with key aggregate numbers
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

_FONT_CANDIDATES = [
    "Noto Sans CJK KR",
    "Noto Sans CJK JP",
    "NanumGothic",
    "Malgun Gothic",
    "AppleGothic",
]
_FONT_PATH_CANDIDATES = [
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
]

for _font_path in _FONT_PATH_CANDIDATES:
    if _font_path.exists():
        font_manager.fontManager.addfont(str(_font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(
            fname=str(_font_path)
        ).get_name()
        break
else:
    _AVAILABLE_FONTS = {font.name for font in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = next(
        (font for font in _FONT_CANDIDATES if font in _AVAILABLE_FONTS),
        "DejaVu Sans",
    )
plt.rcParams["axes.unicode_minus"] = False    # 마이너스 깨짐 방지
FRAGMENT_STATUSES = (1, 2)
EPS = 1e-12


@dataclass
class FileMeta:
    subject: str
    session: str
    block: str
    game: str
    chunk: str


def parse_file_meta(path: Path) -> FileMeta:
    # e.g., sub001-ses04-block2-game2_000.npy
    m = re.match(r"(sub\d+)-ses(\d+)-block(\d+)-game(\d+)_(\d+)\.npy$", path.name)
    if m is None:
        return FileMeta("unknown", "unknown", "unknown", "unknown", path.stem)
    return FileMeta(
        subject=m.group(1),
        session=m.group(2),
        block=m.group(3),
        game=m.group(4),
        chunk=m.group(5),
    )


def load_npy_dict(path: Path) -> Dict[str, np.ndarray]:
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
        d = obj.item()
        if isinstance(d, dict):
            return d
    if hasattr(obj, "files"):
        return {k: obj[k] for k in obj.files}
    raise ValueError(f"Unable to parse file as dict npy/npz: {path}")


def to_action_ids(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim == 1:
        return arr.astype(int)
    return np.argmax(arr, axis=1).astype(int)


def softmax_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(np.clip(x, -60, 60))
    denom = np.sum(ex, axis=1, keepdims=True)
    denom = np.where(denom <= 0, 1.0, denom)
    return ex / denom


def entropy_rows(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return -np.sum(p * np.log(p + EPS), axis=1)


def top2_gap_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[1] < 2:
        return np.zeros(x.shape[0], dtype=float)
    part = np.partition(x, -2, axis=1)[:, -2:]
    return part[:, 1] - part[:, 0]


def js_divergence_rows(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = np.clip(p, EPS, None)
    q = np.clip(q, EPS, None)
    p = p / np.sum(p, axis=1, keepdims=True)
    q = q / np.sum(q, axis=1, keepdims=True)
    m = 0.5 * (p + q)
    return 0.5 * np.sum(p * np.log(p / m), axis=1) + 0.5 * np.sum(q * np.log(q / m), axis=1)


def js_divergence_1d(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = np.clip(p, EPS, None)
    q = np.clip(q, EPS, None)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def split_contiguous(indices: np.ndarray) -> List[np.ndarray]:
    indices = np.asarray(indices, dtype=int)
    if indices.size == 0:
        return []
    cuts = np.where(np.diff(indices) != 1)[0] + 1
    return [x for x in np.split(indices, cuts) if x.size > 0]


def prev_index_of_status(status_indices: np.ndarray, ref_indices: np.ndarray) -> np.ndarray:
    if len(status_indices) == 0:
        return np.full(len(ref_indices), -1, dtype=int)
    pos = np.searchsorted(status_indices, ref_indices) - 1
    out = np.where(pos >= 0, status_indices[pos], -1)
    return out.astype(int)


def pick_valid_action(a: int, q_ref: np.ndarray) -> int:
    try:
        ai = int(a)
    except Exception:
        ai = -1
    if 0 <= ai < len(q_ref):
        return ai
    q = np.asarray(q_ref, dtype=float)
    q = np.where(np.isfinite(q), q, -np.inf)
    if q.size == 0 or not np.isfinite(np.max(q)):
        return 0
    return int(np.argmax(q))


def build_real_step_table(
    data: Dict[str, np.ndarray],
    file_path: Path,
    k_rewards: Sequence[int],
    fragment_statuses: Sequence[int] = FRAGMENT_STATUSES,
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
        cur_action = np.asarray(tree["cur_action"])
        if cur_action.ndim == 2 and cur_action.shape[1] == cur_qs.shape[1]:
            cur_action_id = np.argmax(cur_action, axis=1).astype(int)
        else:
            cur_action_id = cur_action.reshape(-1).astype(int)
    else:
        cur_action_id = np.argmax(cur_qs, axis=1).astype(int)

    lengths = [
        len(status),
        len(human_action),
        len(thinker_action),
        len(actor_logits),
        len(env_return),
        len(cur_qs),
        len(root_qs),
        len(cur_v),
        len(root_v),
        len(root_policy),
        len(cur_policy),
        len(rollout_return),
        len(max_rollout_return),
        len(cur_action_id),
    ]
    t = min(lengths)
    status = status[:t]
    human_action = human_action[:t]
    thinker_action = thinker_action[:t]
    actor_logits = actor_logits[:t]
    env_return = env_return[:t]
    cur_qs = cur_qs[:t]
    root_qs = root_qs[:t]
    cur_v = cur_v[:t]
    root_v = root_v[:t]
    root_policy = root_policy[:t]
    cur_policy = cur_policy[:t]
    rollout_return = rollout_return[:t]
    max_rollout_return = max_rollout_return[:t]
    cur_action_id = cur_action_id[:t]

    probs = softmax_rows(actor_logits)
    entropy_actor = entropy_rows(probs)
    margin_actor = top2_gap_rows(probs)
    q_gap = top2_gap_rows(root_qs)
    search_disagreement = js_divergence_rows(root_policy, cur_policy)
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
        imag_idx = between[np.isin(status[between], fragment_statuses)] if between.size > 0 else np.array([], dtype=int)

        if imag_idx.size > 0:
            if 0 <= target_action < num_actions:
                frags = split_contiguous(imag_idx)
                matched = [frag for frag in frags if cur_action_id[frag[0]] == target_action]
                sel_idx = np.concatenate(matched) if len(matched) > 0 else imag_idx
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

        if idx_global < len(root_v) and np.isfinite(root_v[idx_global]):
            v_ref = float(root_v[idx_global])
        else:
            v_ref = float(cur_v[min(idx_global, t - 1)])
        vre_abs_v = abs(v_prev - v_ref)

        row = {
            "file": str(file_path),
            "subject": meta.subject,
            "session": meta.session,
            "block": meta.block,
            "game": meta.game,
            "chunk": meta.chunk,
            "real_pos": i,
            "global_idx": int(idx_global),
            "prev_s2_idx": int(prev_s2),
            "human_action": int(target_action),
            "thinker_action": int(thinker_action[idx_global]),
            "is_human_noop": int(target_action == 0),
            "is_thinker_noop": int(thinker_action[idx_global] == 0),
            "env_return": float(env_return[idx_global]) if idx_global < len(env_return) else np.nan,
            "entropy_actor": float(entropy_actor[idx_global]),
            "margin_actor": float(margin_actor[idx_global]),
            "p_noop_actor": float(probs[idx_global, 0]) if probs.shape[1] > 0 else np.nan,
            "q_gap": float(q_gap[idx_global]),
            "search_disagreement": float(search_disagreement[idx_global]),
            "rollout_spread": float(rollout_spread[idx_global]),
            # Figure 2 causal-aligned source: immediately preceding search step (status==2).
            "entropy_actor_prev_s2": float(entropy_actor[prev_s2]),
            "margin_actor_prev_s2": float(margin_actor[prev_s2]),
            "q_gap_prev_s2": float(q_gap[prev_s2]),
            "search_disagreement_prev_s2": float(search_disagreement[prev_s2]),
            "rollout_spread_prev_s2": float(rollout_spread[prev_s2]),
            "root_v": float(root_v[idx_global]),
            "cur_v": float(cur_v[idx_global]),
            "vre_abs_q": float(vre_abs_q),
            "vre_abs_v": float(vre_abs_v),
        }
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["file", "real_pos"]).reset_index(drop=True)

    # Mark whether an overt action is directly preceded by withholding.
    df["prev_human_action"] = df.groupby("file")["human_action"].shift(1)
    df["preceded_by_withholding"] = (
        (df["human_action"] != 0) & (df["prev_human_action"] == 0)
    ).astype(int)
    df["is_overt_action"] = (df["human_action"] != 0).astype(int)

    # k-step downstream reward from next real steps.
    for k in k_rewards:
        col = f"k{k}_reward"
        out = np.full(len(df), np.nan, dtype=float)
        for _, g in df.groupby("file", sort=False):
            idxs = g.index.to_numpy()
            rewards = g["env_return"].to_numpy(dtype=float)
            for j, idx_df in enumerate(idxs):
                out[idx_df] = np.nansum(rewards[j + 1 : j + 1 + k])
        df[col] = out

    return df


def extract_noop_bouts(df_real: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for file_id, g in df_real.groupby("file", sort=False):
        g = g.sort_values("real_pos").reset_index(drop=True)
        noop_pos = np.where(g["is_human_noop"].to_numpy(dtype=int) == 1)[0]
        for b_ix, seg in enumerate(split_contiguous(noop_pos)):
            s = int(seg[0])
            e = int(seg[-1])
            pre = s - 1
            commit = e + 1
            if pre < 0 or commit >= len(g):
                continue
            if int(g.loc[commit, "human_action"]) == 0:
                continue
            rows.append(
                {
                    "event_id": f"{Path(file_id).stem}::b{b_ix:04d}",
                    "file": file_id,
                    "start_pos": s,
                    "end_pos": e,
                    "pre_pos": pre,
                    "commit_pos": commit,
                    "length_real_steps": e - s + 1,
                    "start_global_idx": int(g.loc[s, "global_idx"]),
                    "end_global_idx": int(g.loc[e, "global_idx"]),
                    "commit_global_idx": int(g.loc[commit, "global_idx"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["file", "start_pos"]).reset_index(drop=True)


def build_event_tables(
    df_real: pd.DataFrame,
    bouts: pd.DataFrame,
    window_pre: int,
    window_post: int,
    metric_cols: Sequence[str] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metrics = list(
        metric_cols
        if metric_cols is not None
        else [
            "entropy_actor",
            "margin_actor",
            "q_gap",
            "search_disagreement",
            "rollout_spread",
        ]
    )
    prepost_rows = []
    temporal_rows = []

    grouped = {k: v.sort_values("real_pos").reset_index(drop=True) for k, v in df_real.groupby("file", sort=False)}

    for _, b in bouts.iterrows():
        file_id = b["file"]
        g = grouped[file_id]
        pre_pos = int(b["pre_pos"])
        onset_pos = int(b["start_pos"])
        commit_pos = int(b["commit_pos"])
        event_id = b["event_id"]

        rec = {
            "event_id": event_id,
            "file": file_id,
            "pre_pos": pre_pos,
            "onset_pos": onset_pos,
            "commit_pos": commit_pos,
            "length_real_steps": int(b["length_real_steps"]),
        }
        for m in metrics:
            pre_val = float(g.loc[pre_pos, m])
            on_val = float(g.loc[onset_pos, m])
            com_val = float(g.loc[commit_pos, m])
            rec[f"{m}_pre"] = pre_val
            rec[f"{m}_onset"] = on_val
            rec[f"{m}_commit"] = com_val
            rec[f"{m}_delta_commit_pre"] = com_val - pre_val
            rec[f"{m}_delta_commit_onset"] = com_val - on_val
        prepost_rows.append(rec)

        for rel in range(-window_pre, window_post + 1):
            pos = onset_pos + rel
            if 0 <= pos < len(g):
                row = {
                    "event_id": event_id,
                    "file": file_id,
                    "anchor": "onset",
                    "rel_step": rel,
                }
                for m in metrics:
                    row[m] = float(g.loc[pos, m])
                temporal_rows.append(row)
        for rel in range(-window_pre, window_post + 1):
            pos = commit_pos + rel
            if 0 <= pos < len(g):
                row = {
                    "event_id": event_id,
                    "file": file_id,
                    "anchor": "commit",
                    "rel_step": rel,
                }
                for m in metrics:
                    row[m] = float(g.loc[pos, m])
                temporal_rows.append(row)

    return (
        pd.DataFrame(prepost_rows).sort_values(["file", "pre_pos"]).reset_index(drop=True),
        pd.DataFrame(temporal_rows).sort_values(["anchor", "file", "rel_step"]).reset_index(drop=True),
    )


def build_matched_controls(
    df_real: pd.DataFrame,
    bouts: pd.DataFrame,
    entropy_col: str = "entropy_actor",
) -> pd.DataFrame:
    rows = []
    grouped = {k: v.sort_values("real_pos").reset_index(drop=True) for k, v in df_real.groupby("file", sort=False)}
    bouts_by_file = {k: v for k, v in bouts.groupby("file", sort=False)}

    for file_id, g in grouped.items():
        g_b = bouts_by_file.get(file_id)
        if g_b is None or len(g_b) == 0:
            continue

        n = len(g)
        blocked = np.zeros(n, dtype=bool)
        noop_flag = (g["is_human_noop"].to_numpy(dtype=int) == 1)
        for _, b in g_b.iterrows():
            blocked[int(b["start_pos"]) : int(b["end_pos"]) + 1] = True

        if entropy_col not in g.columns:
            raise KeyError(f"Requested entropy column not found: {entropy_col}")
        entropy = g[entropy_col].to_numpy(dtype=float)
        pos_frac = g["real_pos"].to_numpy(dtype=float) / max(n - 1, 1)
        # For immediate-action controls, require the next real step to be non-NOOP.
        next_is_noop = np.zeros(n, dtype=bool)
        if n > 1:
            next_is_noop[:-1] = noop_flag[1:]
            next_is_noop[-1] = True

        for _, b in g_b.iterrows():
            event_id = b["event_id"]
            pre = int(b["pre_pos"])
            commit = int(b["commit_pos"])
            event_horizon = commit - pre
            if event_horizon <= 0:
                continue

            target_entropy = entropy[pre]
            target_pos_frac = pos_frac[pre]

            # Control gain is always one-step: j -> j+1.
            control_horizon = 1
            cand_mask = np.ones(n, dtype=bool)
            cand_mask &= ~blocked
            cand_mask &= (g["is_human_noop"].to_numpy(dtype=int) == 0)
            cand_mask &= (np.arange(n) + control_horizon < n)
            cand_mask &= (np.arange(n) > 0)
            cand_mask &= ~next_is_noop
            cand_idx = np.where(cand_mask)[0]
            if cand_idx.size == 0:
                continue

            z_entropy = np.nanstd(entropy[cand_idx]) + EPS
            costs = (
                np.abs(entropy[cand_idx] - target_entropy) / z_entropy
                + 0.5 * np.abs(pos_frac[cand_idx] - target_pos_frac)
            )
            j = int(cand_idx[np.argmin(costs)])
            j_post = j + control_horizon

            rows.append(
                {
                    "event_id": event_id,
                    "file": file_id,
                    "event_pre_pos": pre,
                    "event_commit_pos": commit,
                    "control_pre_pos": j,
                    "control_post_pos": j_post,
                    "event_horizon": event_horizon,
                    "control_horizon": control_horizon,
                }
            )

    return pd.DataFrame(rows).sort_values(["file", "event_pre_pos"]).reset_index(drop=True)


def summarize_action_distributions(df_real: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for file_id, g in df_real.groupby("file", sort=False):
        n_actions = int(max(g["human_action"].max(), g["thinker_action"].max()) + 1)
        h_counts = np.bincount(g["human_action"].to_numpy(dtype=int), minlength=n_actions)
        t_counts = np.bincount(g["thinker_action"].to_numpy(dtype=int), minlength=n_actions)
        h_noop = h_counts[0] / max(h_counts.sum(), 1)
        t_noop = t_counts[0] / max(t_counts.sum(), 1)
        rows.append(
            {
                "file": file_id,
                "n_real_steps": len(g),
                "human_noop_prop": float(h_noop),
                "thinker_noop_prop": float(t_noop),
                "noop_gap": float(h_noop - t_noop),
                "jsd_human_thinker": float(js_divergence_1d(h_counts, t_counts)),
            }
        )
    return pd.DataFrame(rows).sort_values("file").reset_index(drop=True)


def _mean_sem(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # y shape: [n_samples, n_time]
    mu = np.nanmean(y, axis=0)
    n = np.sum(np.isfinite(y), axis=0)
    sd = np.nanstd(y, axis=0, ddof=1)
    sem = np.where(n > 0, sd / np.sqrt(np.maximum(n, 1)), np.nan)
    return mu, sem


def plot_figure1(df_real: pd.DataFrame, action_summary: pd.DataFrame, out_path: Path) -> None:
    n_actions = int(max(df_real["human_action"].max(), df_real["thinker_action"].max()) + 1)
    h_counts = np.bincount(df_real["human_action"].to_numpy(dtype=int), minlength=n_actions).astype(float)
    t_counts = np.bincount(df_real["thinker_action"].to_numpy(dtype=int), minlength=n_actions).astype(float)
    h_dist = h_counts / max(h_counts.sum(), 1.0)
    t_dist = t_counts / max(t_counts.sum(), 1.0)

    fig = plt.figure(figsize=(20, 4.8))
    gs = fig.add_gridspec(1, 4, width_ratios=[0.9, 1.25, 1.0, 1.0])

    # Panel 1A: withholding bout schematic (research framing)
    ax_s = fig.add_subplot(gs[0, 0])
    ax_s.plot([0, 1], [0, 0], color="black", lw=1.5)
    ax_s.plot([0.25, 0.65], [0, 0], color="#ffd166", lw=9, solid_capstyle="butt")
    ax_s.scatter([0.25, 0.65, 0.8], [0, 0, 0], color=["#ef476f", "#ef476f", "#06d6a0"], s=65)
    ax_s.text(0.25, 0.08, "지연 시작", ha="center")
    ax_s.text(0.65, 0.08, "지연 종료", ha="center")
    ax_s.text(0.8, 0.08, "행동 실행", ha="center")
    ax_s.set_xlim(0, 1)
    ax_s.set_ylim(-0.2, 0.25)
    ax_s.set_axis_off()
    ax_s.set_title("Figure 1A: Withholding bout schematic")

    # Panel 1B: action distribution bar chart
    ax1 = fig.add_subplot(gs[0, 1])
    x = np.arange(n_actions)
    w = 0.38
    ax1.bar(x - w / 2, h_dist, width=w, label="Human", color="#1f77b4")
    ax1.bar(x + w / 2, t_dist, width=w, label="Thinker", color="#ff7f0e")
    ax1.axvspan(-0.5, 0.5, color="#ffd166", alpha=0.2, lw=0)
    ax1.set_xticks(x)
    ax1.set_xlabel("Action ID")
    ax1.set_ylabel("Selection proportion (real steps)")
    ax1.set_title("Figure 1B: Action distribution (NOOP highlighted)")
    ax1.legend(loc="upper right", frameon=False)
    ax1.grid(axis="y", alpha=0.3)

    # Panel 1C: episode-level paired NOOP
    ax2 = fig.add_subplot(gs[0, 2])
    x_pair = [0, 1]
    for _, r in action_summary.iterrows():
        ax2.plot(
            x_pair,
            [r["thinker_noop_prop"], r["human_noop_prop"]],
            color="gray",
            alpha=0.6,
            lw=1.0,
        )
    ax2.scatter(np.full(len(action_summary), 0), action_summary["thinker_noop_prop"], s=18, color="#ff7f0e")
    ax2.scatter(np.full(len(action_summary), 1), action_summary["human_noop_prop"], s=18, color="#1f77b4")
    ax2.set_xticks(x_pair, ["Thinker", "Human"])
    ax2.set_ylabel("NOOP proportion")
    ax2.set_title("Figure 1C: Episode-level paired NOOP")
    ax2.grid(axis="y", alpha=0.3)

    # Panel 1D: residual distribution gap (NOOP gap vs JSD)
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.scatter(action_summary["noop_gap"], action_summary["jsd_human_thinker"], s=28, color="#2a9d8f")
    ax3.set_xlabel("NOOP gap (Human - Thinker)")
    ax3.set_ylabel("JSD(Human, Thinker)")
    ax3.set_title("Figure 1D: Residual distribution gap")
    ax3.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_figure2(
    bouts: pd.DataFrame,
    temporal: pd.DataFrame,
    prepost: pd.DataFrame,
    out_path: Path,
    metric_cols: Sequence[str] | None = None,
    uncertainty_metric: str = "entropy_actor_prev_s2",
) -> None:
    metrics = list(
        metric_cols
        if metric_cols is not None
        else ["entropy_actor_prev_s2", "margin_actor_prev_s2", "q_gap_prev_s2", "rollout_spread_prev_s2"]
    )
    colors_base = {
        "entropy_actor": "#1f77b4",
        "margin_actor": "#ff7f0e",
        "q_gap": "#2ca02c",
        "rollout_spread": "#d62728",
    }

    def _base_metric_name(m: str) -> str:
        return m.replace("_prev_s2", "")

    fig = plt.figure(figsize=(15, 5.5))
    gs = fig.add_gridspec(1, 3)

    def _plot_anchor(ax: plt.Axes, anchor: str, title: str) -> None:
        sub = temporal[temporal["anchor"] == anchor]
        rels = np.sort(sub["rel_step"].unique())
        for m in metrics:
            pivot = sub.pivot_table(index="event_id", columns="rel_step", values=m, aggfunc="mean")
            pivot = pivot.reindex(columns=rels)
            y = pivot.to_numpy(dtype=float)
            # z-score within each event to compare scales.
            row_mu = np.nanmean(y, axis=1, keepdims=True)
            row_sd = np.nanstd(y, axis=1, keepdims=True) + EPS
            y = (y - row_mu) / row_sd
            mu, sem = _mean_sem(y)
            base = _base_metric_name(m)
            label = base
            color = colors_base.get(base, "#444444")
            ax.plot(rels, mu, lw=2, label=label, color=color)
            ax.fill_between(rels, mu - sem, mu + sem, alpha=0.18, color=color)
        ax.axvline(0, color="black", lw=1, ls="--")
        ax.set_xlabel(f"Real-step offset from {anchor}")
        ax.set_ylabel("Within-event z-score")
        ax.set_title(title)
        ax.grid(alpha=0.3)

    # Panel 2A: onset-aligned temporal profile (was 2B)
    ax_a = fig.add_subplot(gs[0, 0])
    _plot_anchor(ax_a, "onset", "Figure 2A: Onset-aligned temporal profile")
    ax_a.legend(frameon=False, fontsize=9)

    # Panel 2B: commit-aligned temporal profile (was 2C)
    ax_b = fig.add_subplot(gs[0, 1])
    _plot_anchor(ax_b, "commit", "Figure 2B: Commit-aligned temporal profile")

    # Panel 2C: bout survival by pre-uncertainty (was 2D)
    ax_c = fig.add_subplot(gs[0, 2])
    if len(prepost) > 0:
        ent_col = f"{uncertainty_metric}_pre"
        if ent_col not in prepost.columns:
            ent_col = "entropy_actor_pre"
        ent = prepost[ent_col].to_numpy(dtype=float)
        q1, q2 = np.nanquantile(ent, [1 / 3, 2 / 3])
        bins = np.digitize(ent, bins=[q1, q2], right=True)
        labels = {0: "Low uncertainty", 1: "Mid uncertainty", 2: "High uncertainty"}
        max_len = int(bouts["length_real_steps"].max())
        x = np.arange(1, max_len + 1)
        for b in [0, 1, 2]:
            lens = prepost.loc[bins == b, "length_real_steps"].to_numpy(dtype=int)
            if len(lens) == 0:
                continue
            surv = np.array([(lens >= t).mean() for t in x], dtype=float)
            ax_c.plot(x, surv, lw=2, label=labels[b])
        ax_c.set_xlabel("Bout length (real NOOP steps)")
        ax_c.set_ylabel("Survival probability")
        ax_c.set_title("Figure 2C: Bout survival by pre-uncertainty")
        ax_c.grid(alpha=0.3)
        ax_c.legend(frameon=False)
    else:
        ax_c.set_axis_off()

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_figure3(
    prepost: pd.DataFrame,
    out_path: Path,
    metric_suffix: str = "",
) -> None:
    fig = plt.figure(figsize=(10, 4.8))
    gs = fig.add_gridspec(1, 2)

    ent_delta_col = f"entropy_actor{metric_suffix}_delta_commit_pre"
    mar_delta_col = f"margin_actor{metric_suffix}_delta_commit_pre"
    qgap_delta_col = f"q_gap{metric_suffix}_delta_commit_pre"
    ent_pre_col = f"entropy_actor{metric_suffix}_pre"

    needed_cols = [ent_delta_col, mar_delta_col, qgap_delta_col, ent_pre_col]
    missing = [c for c in needed_cols if c not in prepost.columns]
    if missing:
        raise KeyError(f"Figure 3 missing columns in prepost table: {missing}")

    # Panel 3A: pre->commit delta distributions.
    ax_a = fig.add_subplot(gs[0, 0])
    deltas = {
        "Delta entropy\n(commit-pre)": prepost[ent_delta_col].to_numpy(dtype=float),
        "Delta margin\n(commit-pre)": prepost[mar_delta_col].to_numpy(dtype=float),
        "Delta q-gap\n(commit-pre)": prepost[qgap_delta_col].to_numpy(dtype=float),
    }
    labels = list(deltas.keys())
    vals = [deltas[k] for k in labels]
    ax_a.boxplot(vals, tick_labels=labels, showfliers=False)
    ax_a.axhline(0, color="black", lw=1, ls="--")
    ax_a.set_ylabel("Change")
    ax_a.set_title("Figure 3A: Event-level pre/post change")
    ax_a.grid(axis="y", alpha=0.3)

    # Panel 3B: pre-uncertainty vs confidence gain.
    ax_b = fig.add_subplot(gs[0, 1])
    x = prepost[ent_pre_col].to_numpy(dtype=float)
    y = -prepost[ent_delta_col].to_numpy(dtype=float)
    ax_b.scatter(x, y, s=9, alpha=0.22, color="#1f77b4")
    if len(x) > 2 and np.nanstd(x) > 0:
        coef = np.polyfit(x, y, deg=1)
        xx = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        yy = coef[0] * xx + coef[1]
        ax_b.plot(xx, yy, color="#d62728", lw=2)
    ax_b.set_xlabel("Pre-withholding uncertainty (entropy)")
    ax_b.set_ylabel("Confidence gain (-delta entropy)")
    ax_b.set_title("Figure 3B: Uncertainty vs confidence gain")
    ax_b.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_figure4(
    df_real: pd.DataFrame,
    prepost: pd.DataFrame,
    matched: pd.DataFrame,
    out_path: Path,
    k_main: int,
    metric_suffix: str = "_prev_s2",
) -> None:
    fig = plt.figure(figsize=(20, 4.8))
    gs = fig.add_gridspec(1, 4)

    overt = df_real[df_real["is_overt_action"] == 1].copy()
    reward_col = f"k{k_main}_reward"
    if reward_col not in overt.columns:
        raise ValueError(f"Missing reward column: {reward_col}")

    # Per-file paired comparison for VRE and reward.
    paired_rows = []
    for file_id, g in overt.groupby("file", sort=False):
        g_w = g[g["preceded_by_withholding"] == 1]
        g_n = g[g["preceded_by_withholding"] == 0]
        if len(g_w) == 0 or len(g_n) == 0:
            continue
        paired_rows.append(
            {
                "file": file_id,
                "vre_with": float(g_w["vre_abs_q"].mean()),
                "vre_non": float(g_n["vre_abs_q"].mean()),
                "rew_with": float(g_w[reward_col].mean()),
                "rew_non": float(g_n[reward_col].mean()),
            }
        )
    paired = pd.DataFrame(paired_rows)

    # Panel 4A: VRE comparison
    ax_a = fig.add_subplot(gs[0, 0])
    if len(paired) > 0:
        for _, r in paired.iterrows():
            ax_a.plot([0, 1], [r["vre_with"], r["vre_non"]], color="gray", alpha=0.6, lw=1.0)
        ax_a.scatter(np.zeros(len(paired)), paired["vre_with"], color="#2ca02c", s=20)
        ax_a.scatter(np.ones(len(paired)), paired["vre_non"], color="#d62728", s=20)
        ax_a.set_xticks([0, 1], ["Withholding-preceded", "Not preceded"])
        ax_a.set_ylabel("VRE (abs Q)")
        ax_a.set_title("Figure 4A: VRE comparison")
        ax_a.grid(axis="y", alpha=0.3)
    else:
        ax_a.set_axis_off()

    # Panel 4B: downstream reward comparison
    ax_b = fig.add_subplot(gs[0, 1])
    if len(paired) > 0:
        for _, r in paired.iterrows():
            ax_b.plot([0, 1], [r["rew_with"], r["rew_non"]], color="gray", alpha=0.6, lw=1.0)
        ax_b.scatter(np.zeros(len(paired)), paired["rew_with"], color="#2ca02c", s=20)
        ax_b.scatter(np.ones(len(paired)), paired["rew_non"], color="#d62728", s=20)
        ax_b.set_xticks([0, 1], ["Withholding-preceded", "Not preceded"])
        ax_b.set_ylabel(f"k={k_main} step reward")
        ax_b.set_title("Figure 4B: Downstream reward comparison")
        ax_b.grid(axis="y", alpha=0.3)
    else:
        ax_b.set_axis_off()

    # Panel 4C: uncertainty-conditioned utility
    ax_c = fig.add_subplot(gs[0, 2])
    if len(overt) > 0:
        bins = pd.qcut(overt["entropy_actor"], q=3, labels=["Low", "Mid", "High"], duplicates="drop")
        overt = overt.assign(unc_bin=bins)
        stats = []
        for (file_id, ub), g in overt.groupby(["file", "unc_bin"], sort=False, observed=False):
            g_w = g[g["preceded_by_withholding"] == 1]
            g_n = g[g["preceded_by_withholding"] == 0]
            if len(g_w) == 0 or len(g_n) == 0:
                continue
            stats.append(
                {
                    "file": file_id,
                    "unc_bin": str(ub),
                    "vre_gain": float(g_n["vre_abs_q"].mean() - g_w["vre_abs_q"].mean()),
                    "reward_gain": float(g_w[reward_col].mean() - g_n[reward_col].mean()),
                }
            )
        s = pd.DataFrame(stats)
        if len(s) > 0:
            order = [x for x in ["Low", "Mid", "High"] if x in s["unc_bin"].unique()]
            x = np.arange(len(order))
            vre_mu = [s.loc[s["unc_bin"] == b, "vre_gain"].mean() for b in order]
            rew_mu = [s.loc[s["unc_bin"] == b, "reward_gain"].mean() for b in order]
            ax_c.plot(x, vre_mu, marker="o", lw=2, color="#1f77b4", label="VRE reduction")
            ax_c.plot(x, rew_mu, marker="s", lw=2, color="#ff7f0e", label="Reward gain")
            ax_c.axhline(0, color="black", lw=1, ls="--")
            ax_c.set_xticks(x, order)
            ax_c.set_xlabel("Uncertainty bin")
            ax_c.set_ylabel("Benefit")
            ax_c.set_title("Figure 4C: Uncertainty-conditioned utility")
            ax_c.legend(frameon=False, fontsize=9)
            ax_c.grid(alpha=0.3)
        else:
            ax_c.set_axis_off()
    else:
        ax_c.set_axis_off()

    # Panel 4D: matched-control comparison (moved from Figure 3C)
    ent_row_col = f"entropy_actor{metric_suffix}"
    ax_d = fig.add_subplot(gs[0, 3])
    if len(matched) > 0 and ent_row_col in df_real.columns:
        grouped = {k: v.sort_values("real_pos").reset_index(drop=True) for k, v in df_real.groupby("file", sort=False)}
        event_delta = []
        ctrl_delta = []
        for _, r in matched.iterrows():
            g = grouped[r["file"]]
            e_pre = int(r["event_pre_pos"])
            e_post = int(r["event_commit_pos"])
            c_pre = int(r["control_pre_pos"])
            c_post = int(r["control_post_pos"])
            e_gain = -(float(g.loc[e_post, ent_row_col]) - float(g.loc[e_pre, ent_row_col]))
            c_gain = -(float(g.loc[c_post, ent_row_col]) - float(g.loc[c_pre, ent_row_col]))
            event_delta.append(e_gain)
            ctrl_delta.append(c_gain)
        ax_d.boxplot(
            [event_delta, ctrl_delta],
            tick_labels=["Withholding\nevents", "Matched\ncontrols"],
            showfliers=False,
        )
        ax_d.axhline(0, color="black", lw=1, ls="--")
        ax_d.set_ylabel("Confidence gain (-delta entropy)")
        ax_d.set_title("Figure 4D: Matched-control comparison")
        ax_d.grid(axis="y", alpha=0.3)
    else:
        ax_d.set_axis_off()

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def run_pipeline(
    input_dir: Path,
    out_dir: Path,
    k_rewards: Sequence[int],
    k_main: int,
    window_pre: int,
    window_post: int,
) -> Dict[str, float]:
    paths = sorted(input_dir.glob("*.npy"))
    if len(paths) == 0:
        raise FileNotFoundError(f"No .npy files found in: {input_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    all_real = []
    for p in paths:
        data = load_npy_dict(p)
        df = build_real_step_table(data=data, file_path=p, k_rewards=k_rewards)
        if len(df) > 0:
            all_real.append(df)
    if len(all_real) == 0:
        raise RuntimeError("No real-step rows were produced.")

    df_real = pd.concat(all_real, axis=0, ignore_index=True)
    bouts = extract_noop_bouts(df_real)

    event_metrics_real = [
        "entropy_actor",
        "margin_actor",
        "q_gap",
        "search_disagreement",
        "rollout_spread",
    ]
    event_metrics_prev_s2 = [
        "entropy_actor_prev_s2",
        "margin_actor_prev_s2",
        "q_gap_prev_s2",
        "search_disagreement_prev_s2",
        "rollout_spread_prev_s2",
    ]

    # Build both source variants for reproducibility.
    prepost_real, temporal_real = build_event_tables(
        df_real,
        bouts,
        window_pre=window_pre,
        window_post=window_post,
        metric_cols=event_metrics_real,
    )
    prepost_prev_s2, temporal_prev_s2 = build_event_tables(
        df_real,
        bouts,
        window_pre=window_pre,
        window_post=window_post,
        metric_cols=event_metrics_prev_s2,
    )
    matched = build_matched_controls(df_real, bouts, entropy_col="entropy_actor_prev_s2")
    action_summary = summarize_action_distributions(df_real)

    df_real.to_csv(out_dir / "real_step_metrics.csv", index=False)
    bouts.to_csv(out_dir / "noop_bouts.csv", index=False)
    # Keep default names for the source used in Figure 2.
    prepost_prev_s2.to_csv(out_dir / "event_prepost.csv", index=False)
    temporal_prev_s2.to_csv(out_dir / "event_temporal.csv", index=False)
    # Save real-step source explicitly for downstream comparisons/ablation.
    prepost_real.to_csv(out_dir / "event_prepost_real.csv", index=False)
    temporal_real.to_csv(out_dir / "event_temporal_real.csv", index=False)
    matched.to_csv(out_dir / "matched_controls.csv", index=False)
    action_summary.to_csv(out_dir / "action_summary.csv", index=False)

    plot_figure1(df_real, action_summary, out_dir / "figure1_action_distribution.png")
    plot_figure2(
        bouts,
        temporal_prev_s2,
        prepost_prev_s2,
        out_dir / "figure2_temporal_profile.png",
        metric_cols=[
            "entropy_actor_prev_s2",
            "margin_actor_prev_s2",
            "q_gap_prev_s2",
            "rollout_spread_prev_s2",
        ],
        uncertainty_metric="entropy_actor_prev_s2",
    )
    plot_figure3(
        prepost_prev_s2,
        out_dir / "figure3_prepost_control.png",
        metric_suffix="_prev_s2",
    )
    plot_figure4(
        df_real,
        prepost_prev_s2,
        matched,
        out_dir / "figure4_functional_benefit.png",
        k_main=k_main,
        metric_suffix="_prev_s2",
    )

    overt = df_real[df_real["is_overt_action"] == 1].copy()
    reward_col = f"k{k_main}_reward"
    summary = {
        "n_files": int(df_real["file"].nunique()),
        "n_real_steps": int(len(df_real)),
        "n_bouts": int(len(bouts)),
        "mean_bout_length_real_steps": float(bouts["length_real_steps"].mean()) if len(bouts) > 0 else float("nan"),
        "human_noop_real_mean": float(df_real["is_human_noop"].mean()),
        "thinker_noop_real_mean": float(df_real["is_thinker_noop"].mean()),
        "mean_noop_gap_human_minus_thinker": float(
            action_summary["human_noop_prop"].mean() - action_summary["thinker_noop_prop"].mean()
        ),
        "mean_jsd_human_thinker": float(action_summary["jsd_human_thinker"].mean()),
    }
    if len(overt) > 0:
        withhold = overt[overt["preceded_by_withholding"] == 1]
        non = overt[overt["preceded_by_withholding"] == 0]
        if len(withhold) > 0 and len(non) > 0:
            summary.update(
                {
                    "vre_abs_q_withholding_preceded": float(withhold["vre_abs_q"].mean()),
                    "vre_abs_q_not_preceded": float(non["vre_abs_q"].mean()),
                    "k_reward_withholding_preceded": float(withhold[reward_col].mean()),
                    "k_reward_not_preceded": float(non[reward_col].mean()),
                }
            )

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Withholding analysis for filtered sub001 ses-04 files.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("test/sub001/ses-04"),
        help="Directory containing filtered .npy files (default: test/sub001/ses-04)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("analysis_outputs/sub001_ses04_withholding"),
        help="Output directory for csv/png/json results",
    )
    parser.add_argument(
        "--k-rewards",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10],
        help="k values for downstream reward metrics",
    )
    parser.add_argument(
        "--k-main",
        type=int,
        default=5,
        help="Main k used in Figure 4 reward panel (must be in --k-rewards)",
    )
    parser.add_argument("--window-pre", type=int, default=6, help="Temporal window before onset/commit")
    parser.add_argument("--window-post", type=int, default=6, help="Temporal window after onset/commit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.k_main not in args.k_rewards:
        raise ValueError("--k-main must be included in --k-rewards.")

    summary = run_pipeline(
        input_dir=args.input_dir,
        out_dir=args.out_dir,
        k_rewards=args.k_rewards,
        k_main=args.k_main,
        window_pre=args.window_pre,
        window_post=args.window_post,
    )
    print("Analysis complete.")
    print(json.dumps(summary, indent=2))
    print(f"Output directory: {args.out_dir}")


if __name__ == "__main__":
    main()
