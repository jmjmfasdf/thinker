#!/usr/bin/env python3
"""
09_state_complexity_gramian.py

State Complexity & Empirical Controllability Gramian Analysis.

Games:
  game_0 = Enduro   (9-action ALE)
  game_1 = Pong     (6-action ALE)
  game_2 = SpaceInvaders (6-action ALE)

Notation:
  R_t in {0..255}^128
      Raw Atari RAM vector at timestep t. Index j always denotes the original
      ALE RAM byte address j in [0, 127].
  I_g subset {0, ..., 127}
      Game-specific RAM byte subset selected for analysis. By default this is
      the annotation-based gameplay byte set listed in RAM_ANNOTATIONS below.
      A data-adaptive fallback is available via --ram-scope selected.
  x_t = R_t[I_g] in {0..255}^{d_g}
      Gameplay-relevant RAM state used by the complexity and Gramian metrics.
  z_t = x_t / 255
      Normalized RAM state.
  u_t in {0,1}^{n_actions}
      One-hot action at timestep t.
  Delta z_t = z_{t+1} - z_t
      One-step normalized RAM change aligned with action u_t.

For each game, per timestep selected state x_t:

--- State Complexity Metrics ---
  H_byte(x)    : Shannon entropy of selected-byte distribution
  delta(x)     : Hamming change rate vs previous step (fraction of selected bytes changed)
  alpha(x)     : Active byte fraction (non-zero selected bytes / d_g)
  var_byte(x)  : Variance of selected byte values
  lz(x)        : Lempel-Ziv-76 complexity of selected-byte bit string (normalized)
  eff_rank_pca : PCA effective rank over a sliding window (state space dimensionality)

--- Controllability Gramian (empirical, local linear approximation) ---
  At window t: estimate B via OLS:  Delta z_t = B @ u_t + eps
  W_c = B @ B.T  (one-step empirical Gramian, R^{d_g x d_g}, rank <= n_actions)

  Metrics:
    tr_Wc       : trace(W_c) = sum of eigenvalues = total reachable energy
    lam1        : largest eigenvalue (dominant control direction)
    lam_min_nz  : smallest non-zero eigenvalue
    cond_nz     : lam1 / lam_min_nz (condition number of non-trivial subspace)
    eff_rank_Wc : effective rank of W_c
    log_det_Wc  : log pseudo-determinant

Outputs:
  results/
    complexity_by_game.csv
    ram_byte_selection.csv
    gramian_global.csv
    gramian_sliding.csv
    complexity_gramian_corr.csv
    summary.txt
  figures/
    fig_4_1_complexity_distributions.png
    fig_4_2_gramian_eigenspectra.png
    fig_4_3_sliding_controllability.png
    fig_4_4_complexity_vs_gramian.png
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "behavioral_data_block_old"
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "09_state_complexity_gramian"

GAME_INFO = {
    0: dict(name="Enduro",        n_actions=9, color="#e67e22"),
    1: dict(name="Pong",          n_actions=6, color="#2980b9"),
    2: dict(name="SpaceInvaders", n_actions=6, color="#27ae60"),
}

# ---------------------------------------------------------------------------
# RAM annotations
# ---------------------------------------------------------------------------

# Gameplay RAM byte sets are based on OCAtari's RAM Extraction Method (REM)
# implementations, which reverse-engineer RAM bytes used to recover object
# positions, visibility, and selected game-state variables from ALE RAM.
# Source: https://github.com/k4ntz/OC_Atari/tree/master/ocatari/ram
RAM_ANNOTATIONS = {
    0: {
        "source": "OCAtari ocatari/ram/enduro.py",
        "analysis_indices": [
            27, 28, 29, 30, 31, 32, 33,  # opponent car slots/flags
            34,                          # turn / road curvature candidate
            45,                          # level/day indicator
            46,                          # player x drift / road turn
            52,                          # player y offset
            54,                          # previous x / turn history candidate
            59,                          # opponent car perspective/depth offset
            106,                         # player sinking/death y state
        ],
        "labels": {
            27: "car_slot_0_flags",
            28: "car_slot_1_flags",
            29: "car_slot_2_flags",
            30: "car_slot_3_flags",
            31: "car_slot_4_flags",
            32: "car_slot_5_flags",
            33: "car_slot_6_flags",
            34: "turn_or_road_curvature_candidate",
            45: "level_or_day_indicator",
            46: "player_x_drift_or_road_turn",
            52: "player_y_offset",
            54: "previous_player_x_or_turn_history",
            59: "car_depth_or_perspective_offset",
            106: "player_sinking_or_death_y_state",
        },
    },
    1: {
        "source": "OCAtari ocatari/ram/pong.py",
        "analysis_indices": [49, 54],  # ball_x, ball_y only (outcome-relevant)
        "labels": {
            13: "enemy_score_hud_excluded",
            14: "player_score_hud_excluded",
            49: "ball_x",
            50: "enemy_paddle_y",
            51: "player_paddle_y",
            54: "ball_y",
        },
    },
    2: {
        "source": "OCAtari ocatari/ram/spaceinvaders.py",
        "analysis_indices": [
            17, 18, 19, 20, 21, 22, 23,  # alive count + alien_bitmap_row_1~6 (outcome-relevant)
        ],
        "labels": {
            16: "alien_group_y_or_frame_y",
            17: "number_of_alive_aliens",
            18: "alien_bitmap_row_1",
            19: "alien_bitmap_row_2",
            20: "alien_bitmap_row_3",
            21: "alien_bitmap_row_4",
            22: "alien_bitmap_row_5",
            23: "alien_bitmap_row_6",
            24: "player_and_shield_visibility",
            26: "aliens_x",
            27: "shields_x_reference",
            28: "player_green_x",
            29: "player_yellow_x",
            30: "satellite_x",
            43: "left_shield_row_0",
            44: "left_shield_row_1",
            45: "left_shield_row_2",
            46: "left_shield_row_3",
            47: "left_shield_row_4",
            48: "left_shield_row_5",
            49: "left_shield_row_6",
            50: "left_shield_row_7",
            51: "left_shield_row_8",
            52: "middle_shield_row_0",
            53: "middle_shield_row_1",
            54: "middle_shield_row_2",
            55: "middle_shield_row_3",
            56: "middle_shield_row_4",
            57: "middle_shield_row_5",
            58: "middle_shield_row_6",
            59: "middle_shield_row_7",
            60: "middle_shield_row_8",
            61: "right_shield_row_0",
            62: "right_shield_row_1",
            63: "right_shield_row_2",
            64: "right_shield_row_3",
            65: "right_shield_row_4",
            66: "right_shield_row_5",
            67: "right_shield_row_6",
            68: "right_shield_row_7",
            69: "right_shield_row_8",
            71: "object_colours_excluded",
            72: "enemy_destroyed_symbol_state_excluded",
            73: "lives",
            74: "temporal_reference_excluded",
            81: "enemy_bullet_1_y",
            82: "enemy_bullet_2_y",
            83: "enemy_bullet_1_x",
            84: "enemy_bullet_2_x",
            85: "player_green_bullet_y",
            86: "player_yellow_bullet_y",
            87: "player_green_bullet_x",
            88: "player_yellow_bullet_x",
            90: "unknown_repeating_value_excluded",
            102: "player_green_score_high_hud_excluded",
            103: "player_yellow_score_high_hud_excluded",
            104: "player_green_score_low_hud_excluded",
            105: "player_yellow_score_low_hud_excluded",
            120: "lives_hud_visibility_excluded",
        },
    },
}

# ---------------------------------------------------------------------------
# Helpers: RAM parsing
# ---------------------------------------------------------------------------

def parse_ram_file(ram_path: Path) -> np.ndarray:
    """Parse RAM.txt → (T, 128) uint8 array."""
    lines = ram_path.read_text().splitlines()
    rows = []
    for line in lines:
        if ":" not in line:
            continue
        vals_str = line.split(":", 1)[1].strip().rstrip(",")
        vals = [int(v) for v in vals_str.split(",") if v.strip()]
        if len(vals) == 128:
            rows.append(vals)
    return np.array(rows, dtype=np.uint8) if rows else np.empty((0, 128), dtype=np.uint8)


def parse_npz_actions(npz_path: Path) -> np.ndarray:
    """Parse NPZ actions → (T,) int indices."""
    data = np.load(npz_path)
    return np.argmax(data["action"], axis=1).astype(np.int32)


# ---------------------------------------------------------------------------
# Helpers: State Complexity
# ---------------------------------------------------------------------------

def byte_entropy(ram: np.ndarray) -> np.ndarray:
    """Shannon entropy of selected-byte distribution per step. (T,) float."""
    # Build count matrix (T, 256) via one-hot accumulation.
    T, n_bytes = ram.shape
    if n_bytes == 0:
        return np.full(T, np.nan)
    # Use histogram2d trick: each row is a 256-bin histogram
    counts = np.zeros((T, 256), dtype=np.float32)
    np.add.at(counts, (np.repeat(np.arange(T), n_bytes), ram.ravel()), 1.0)
    p = counts / float(n_bytes)
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(p > 0, np.log2(p + 1e-300), 0.0)
    return -(p * logp).sum(axis=1)


def hamming_change_rate(ram: np.ndarray) -> np.ndarray:
    """Fraction of selected bytes that changed from previous step. (T,) float."""
    changed = np.zeros(ram.shape[0])
    if ram.shape[0] > 1:
        changed[1:] = np.mean(ram[1:] != ram[:-1], axis=1)
    return changed


def active_byte_fraction(ram: np.ndarray) -> np.ndarray:
    """Fraction of non-zero selected bytes per step. (T,) float."""
    return np.mean(ram != 0, axis=1)


def byte_variance(ram: np.ndarray) -> np.ndarray:
    """Variance of byte values per step. (T,) float."""
    return ram.astype(np.float32).var(axis=1)


def lz_complexity_fast(ram_row: np.ndarray) -> float:
    """LZ76 complexity of selected-byte RAM via string hashing. O(n log n)."""
    bits = np.unpackbits(ram_row).tolist()
    n = len(bits)
    seen = set()
    i, c, l = 0, 1, 1
    while i + l <= n:
        key = tuple(bits[i:i + l])
        if key in seen:
            l += 1
        else:
            seen.add(tuple(bits[i:i + l - 1]) if l > 1 else ())
            # add all prefixes seen so far to dictionary
            seen.add(key)
            c += 1
            i += l
            l = 1
    norm = n / (np.log2(n) + 1e-12)
    return c / norm


def lz_complexity_batch(ram: np.ndarray, max_steps: int = 2000) -> np.ndarray:
    """Batch LZ complexity — uniformly sampled then interpolated. (T,) float."""
    T = ram.shape[0]
    step = max(1, T // max_steps)
    indices = np.arange(0, T, step)
    vals = np.array([lz_complexity_fast(ram[i]) for i in indices])
    return np.interp(np.arange(T), indices, vals)


def pca_eff_rank(ram: np.ndarray, window: int = 200, step: int = 100) -> np.ndarray:
    """PCA effective rank in sliding window via covariance eigenvalues. (T,) float."""
    T = ram.shape[0]
    eff = np.full(T, np.nan)
    X = ram.astype(np.float32) / 255.0
    for t in range(0, T, step):
        lo = max(0, t - window // 2)
        hi = min(T, t + window // 2)
        if hi - lo < 20:
            continue
        Xw = X[lo:hi]
        Xw = Xw - Xw.mean(axis=0)
        # covariance (128x128) — faster than full SVD of (W,128)
        C = (Xw.T @ Xw) / (len(Xw) - 1)
        lam = np.linalg.eigvalsh(C)
        lam = np.maximum(lam, 0)
        total = lam.sum()
        if total < 1e-12:
            continue
        lam_n = lam / total
        lam_n = lam_n[lam_n > 1e-10]
        er = float(np.exp(-np.sum(lam_n * np.log(lam_n + 1e-12))))
        eff[t:min(t + step, T)] = er
    first_valid = np.where(~np.isnan(eff))[0]
    if len(first_valid) > 0:
        eff[:first_valid[0]] = eff[first_valid[0]]
    return eff


# ---------------------------------------------------------------------------
# Helpers: Controllability Gramian
# ---------------------------------------------------------------------------

def empirical_B(delta_s: np.ndarray, actions: np.ndarray, n_actions: int) -> np.ndarray:
    """
    Estimate control matrix B via OLS:  delta_s ≈ B @ u + eps
    delta_s : (T, d_g) float
    actions : (T,) int
    Returns B : (d_g, n_actions)
    """
    T = delta_s.shape[0]
    U = np.zeros((T, n_actions), dtype=np.float32)
    U[np.arange(T), actions] = 1.0
    # B = (delta_s.T @ U) @ pinv(U.T @ U)
    UtU = U.T @ U  # (n_actions, n_actions)
    UtDS = U.T @ delta_s  # (n_actions, d_g)
    try:
        B = np.linalg.solve(UtU + 1e-6 * np.eye(n_actions), UtDS).T
    except np.linalg.LinAlgError:
        B = (np.linalg.pinv(UtU) @ UtDS).T
    return B


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    """Return deterministic percentile ranks in [0, 1] for a 1D score vector."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) <= 1:
        return np.zeros_like(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, len(values))
    return ranks


def select_gameplay_ram_bytes(
    ram: np.ndarray,
    actions: np.ndarray,
    n_actions: int,
    min_bytes: int = 8,
    max_bytes: int = 64,
    max_transitions: int = 200_000,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Select a game-specific RAM byte subset I_g for gameplay/control analysis.

    The selector is intentionally data-driven because ALE RAM byte semantics are
    game-specific and not fully documented in this dataset. A byte is considered
    more relevant when it carries state variability, changes over time, and/or
    has action-coupled one-step changes under the same empirical B model used
    for the Gramian.
    """
    T, n_raw_bytes = ram.shape
    min_bytes = max(1, min(min_bytes, n_raw_bytes))
    max_bytes = max(min_bytes, min(max_bytes, n_raw_bytes))

    X = ram.astype(np.float32) / 255.0
    var_score = X.var(axis=0) if T > 0 else np.zeros(n_raw_bytes)
    if T > 1:
        change_score = np.mean(ram[1:] != ram[:-1], axis=0)
    else:
        change_score = np.zeros(n_raw_bytes)

    control_score = np.zeros(n_raw_bytes, dtype=np.float64)
    n_trans = max(0, min(T - 1, len(actions) - 1))
    if n_trans > 1:
        delta_s = np.diff(X[:n_trans + 1], axis=0)
        acts = actions[:n_trans]
        valid = (acts >= 0) & (acts < n_actions)
        idx = np.where(valid)[0]
        if len(idx) > max_transitions:
            idx = np.random.choice(idx, size=max_transitions, replace=False)
            idx.sort()
        if len(idx) > 1 and len(np.unique(acts[idx])) >= 2:
            B = empirical_B(delta_s[idx], acts[idx], n_actions)
            control_score = np.linalg.norm(B, axis=1)

    relevance = (
        0.30 * _percentile_rank(var_score)
        + 0.30 * _percentile_rank(change_score)
        + 0.40 * _percentile_rank(control_score)
    )
    dynamic = (var_score > 1e-10) | (change_score > 0.0) | (control_score > 0.0)

    ordered_all = np.argsort(relevance, kind="mergesort")[::-1]
    dynamic_set = set(np.where(dynamic)[0].tolist())
    ordered_dynamic = [idx for idx in ordered_all if idx in dynamic_set]
    if len(ordered_dynamic) < min_bytes:
        selected_by_score = ordered_dynamic + [
            idx for idx in ordered_all if idx not in set(ordered_dynamic)
        ]
        n_keep = min(max_bytes, min_bytes)
    else:
        selected_by_score = ordered_dynamic
        n_keep = min(max_bytes, len(ordered_dynamic))

    selected_by_score = np.array(selected_by_score[:n_keep], dtype=np.int32)
    selected_indices = np.sort(selected_by_score)
    rank_lookup = {int(byte): rank + 1 for rank, byte in enumerate(selected_by_score)}

    selection_df = pd.DataFrame({
        "ram_index": np.arange(n_raw_bytes, dtype=np.int32),
        "selected": [idx in rank_lookup for idx in range(n_raw_bytes)],
        "selected_rank": [rank_lookup.get(idx, np.nan) for idx in range(n_raw_bytes)],
        "var_score": var_score,
        "change_score": change_score,
        "control_score": control_score,
        "relevance_score": relevance,
    })
    return selected_indices, selection_df


def ram_annotation_metadata(game_id: int, n_raw_bytes: int = 128) -> pd.DataFrame:
    """Return per-byte annotation metadata and the annotated analysis subset."""
    spec = RAM_ANNOTATIONS.get(game_id, {})
    labels = spec.get("labels", {})
    source = spec.get("source", "")
    analysis_order = list(dict.fromkeys(spec.get("analysis_indices", [])))
    selected_rank = {int(byte): rank + 1 for rank, byte in enumerate(analysis_order)}
    selected_set = set(selected_rank)
    return pd.DataFrame({
        "ram_index": np.arange(n_raw_bytes, dtype=np.int32),
        "selected": [idx in selected_set for idx in range(n_raw_bytes)],
        "selected_rank": [selected_rank.get(idx, np.nan) for idx in range(n_raw_bytes)],
        "annotation": [labels.get(idx, "") for idx in range(n_raw_bytes)],
        "annotation_source": [
            source if idx in labels or idx in selected_set else ""
            for idx in range(n_raw_bytes)
        ],
        "var_score": np.nan,
        "change_score": np.nan,
        "control_score": np.nan,
        "relevance_score": np.nan,
    })


def gramian_metrics(B: np.ndarray) -> Dict[str, float]:
    """
    Compute controllability metrics from one-step Gramian W_c = B @ B.T.
    B : (d_g, n_actions)
    """
    Wc = B @ B.T  # (d_g, d_g)
    eigvals = np.linalg.eigvalsh(Wc)
    eigvals = np.sort(eigvals)[::-1]
    eigvals = np.maximum(eigvals, 0)

    nz = eigvals[eigvals > 1e-8 * eigvals[0] + 1e-12]
    tr = float(eigvals.sum())
    lam1 = float(eigvals[0]) if len(eigvals) > 0 else 0.0
    lam_min_nz = float(nz[-1]) if len(nz) > 0 else 0.0
    cond_nz = float(lam1 / lam_min_nz) if lam_min_nz > 0 else np.inf
    eff_rank = 0.0
    if tr > 0:
        p = eigvals / tr
        p = p[p > 1e-12]
        eff_rank = float(np.exp(-np.sum(p * np.log(p + 1e-12))))
    log_det = float(np.sum(np.log(nz + 1e-12))) if len(nz) > 0 else -np.inf

    return dict(
        tr_Wc=tr,
        lam1=lam1,
        lam_min_nz=lam_min_nz,
        cond_nz=cond_nz,
        eff_rank_Wc=eff_rank,
        log_det_Wc=log_det,
        n_eff_actions=len(nz),
    )


def sliding_gramian(
    ram: np.ndarray,
    actions: np.ndarray,
    n_actions: int,
    window: int = 300,
    step: int = 100,
) -> pd.DataFrame:
    """Compute Gramian metrics in sliding windows."""
    T = ram.shape[0]
    delta_s = np.diff(ram.astype(np.float32), axis=0) / 255.0  # (T-1, d_g)
    acts_w = actions[:-1]  # align to delta_s

    records = []
    for t in range(0, T - window, step):
        lo, hi = t, min(t + window, T - 1)
        ds = delta_s[lo:hi]
        ua = acts_w[lo:hi]
        if len(np.unique(ua)) < 2:
            continue
        B = empirical_B(ds, ua, n_actions)
        m = gramian_metrics(B)
        m["t"] = t + window // 2
        records.append(m)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_game_data(game_id: int, max_blocks: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """Load RAM and action arrays for a game across all subjects/blocks."""
    all_ram, all_acts = [], []
    count = 0
    for sub in sorted(DATA_DIR.iterdir()):
        if not sub.name.startswith("sub"):
            continue
        gpath = sub / f"game_{game_id}"
        if not gpath.exists():
            continue
        for day in sorted(gpath.iterdir()):
            if not day.is_dir():
                continue
            for block in sorted(day.iterdir()):
                if not block.is_dir():
                    continue
                ram_file = block / "RAM.txt"
                npz_files = list(block.glob("*.npz"))
                if not ram_file.exists() or not npz_files:
                    continue
                try:
                    ram = parse_ram_file(ram_file)
                    acts = parse_npz_actions(npz_files[0])
                    min_T = min(len(ram), len(acts))
                    if min_T < 50:
                        continue
                    all_ram.append(ram[:min_T])
                    all_acts.append(acts[:min_T])
                    count += 1
                    if count >= max_blocks:
                        break
                except Exception:
                    continue
            if count >= max_blocks:
                break
        if count >= max_blocks:
            break
    if not all_ram:
        return np.empty((0, 128), dtype=np.uint8), np.empty(0, dtype=np.int32)
    return np.concatenate(all_ram, axis=0), np.concatenate(all_acts, axis=0)


def load_blocks_for_subject(
    sub_path: Path, game_id: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Load all blocks for a single subject+game."""
    all_ram, all_acts = [], []
    gpath = sub_path / f"game_{game_id}"
    if not gpath.exists():
        return np.empty((0, 128), dtype=np.uint8), np.empty(0, dtype=np.int32)
    for day in sorted(gpath.iterdir()):
        if not day.is_dir():
            continue
        for block in sorted(day.iterdir()):
            if not block.is_dir():
                continue
            ram_file = block / "RAM.txt"
            npz_files = list(block.glob("*.npz"))
            if not ram_file.exists() or not npz_files:
                continue
            try:
                ram = parse_ram_file(ram_file)
                acts = parse_npz_actions(npz_files[0])
                min_T = min(len(ram), len(acts))
                if min_T < 50:
                    continue
                all_ram.append(ram[:min_T])
                all_acts.append(acts[:min_T])
            except Exception:
                continue
    if not all_ram:
        return np.empty((0, 128), dtype=np.uint8), np.empty(0, dtype=np.int32)
    return np.concatenate(all_ram, axis=0), np.concatenate(all_acts, axis=0)


def load_first_block_for_subject(
    sub_path: Path, game_id: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Load only the first available block for a single subject+game."""
    gpath = sub_path / f"game_{game_id}"
    if not gpath.exists():
        return np.empty((0, 128), dtype=np.uint8), np.empty(0, dtype=np.int32)
    for day in sorted(gpath.iterdir()):
        if not day.is_dir():
            continue
        for block in sorted(day.iterdir()):
            if not block.is_dir():
                continue
            ram_file = block / "RAM.txt"
            npz_files = list(block.glob("*.npz"))
            if not ram_file.exists() or not npz_files:
                continue
            try:
                ram = parse_ram_file(ram_file)
                acts = parse_npz_actions(npz_files[0])
                min_T = min(len(ram), len(acts))
                if min_T < 50:
                    continue
                return ram[:min_T], acts[:min_T]
            except Exception:
                continue
    return np.empty((0, 128), dtype=np.uint8), np.empty(0, dtype=np.int32)


def analyze_subject_game(
    sub_name: str,
    sub_path: Path,
    game_id: int,
    selected_indices: np.ndarray,
    n_actions: int,
) -> Dict:
    """Estimate B matrix and sliding Gramian for one subject+game."""
    ram_raw, actions = load_blocks_for_subject(sub_path, game_id)
    if len(ram_raw) == 0:
        return {}
    ram = ram_raw[:, selected_indices]
    delta_s = np.diff(ram.astype(np.float32), axis=0) / 255.0
    acts = actions[:-1]
    idx = np.arange(len(delta_s))
    if len(idx) > 200_000:
        idx = np.random.choice(idx, size=200_000, replace=False)
        idx.sort()
    B = empirical_B(delta_s[idx], acts[idx], n_actions)
    sv = np.linalg.svd(B, compute_uv=False)
    global_metrics = gramian_metrics(B)
    global_metrics["sub_name"] = sub_name
    global_metrics["game_id"] = game_id
    global_metrics["game_name"] = GAME_INFO[game_id]["name"]
    global_metrics["n_ram_bytes"] = len(selected_indices)

    ram_block_raw, acts_block = load_first_block_for_subject(sub_path, game_id)
    if len(ram_block_raw) == 0:
        sliding_df = pd.DataFrame()
    else:
        ram_block = ram_block_raw[:, selected_indices]
        sliding_df = sliding_gramian(ram_block, acts_block, n_actions, window=60, step=10)
        sliding_df["sub_name"] = sub_name
        sliding_df["game_id"] = game_id
        sliding_df["game_name"] = GAME_INFO[game_id]["name"]

    return {
        "sub_name": sub_name,
        "game_id": game_id,
        "sv_global": sv,
        "global_metrics": global_metrics,
        "sliding_df": sliding_df,
        "ram_first_block_raw": ram_block_raw,  # full 128-byte RAM for score/event overlay
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_game(
    game_id: int,
    max_blocks: int = 30,
    ram_scope: str = "annotated",
    min_ram_bytes: int = 8,
    max_ram_bytes: int = 64,
) -> Dict:
    info = GAME_INFO[game_id]
    name = info["name"]
    n_actions = info["n_actions"]
    print(f"\n{'='*60}")
    print(f"  {name} (game_{game_id}, {n_actions} actions)")
    print(f"{'='*60}")

    print("  Loading data...")
    ram_raw, actions = load_game_data(game_id, max_blocks=max_blocks)
    T = len(ram_raw)
    if T == 0:
        print("  No data found.")
        return {}
    print(f"  Loaded {T:,} steps")

    annotation_df = ram_annotation_metadata(game_id, ram_raw.shape[1])
    if ram_scope == "annotated":
        selected_indices = annotation_df.loc[
            annotation_df["selected"], "ram_index"
        ].to_numpy(dtype=np.int32)
        if len(selected_indices) == 0:
            selected_indices, ram_selection_df = select_gameplay_ram_bytes(
                ram_raw,
                actions,
                n_actions,
                min_bytes=min_ram_bytes,
                max_bytes=max_ram_bytes,
            )
            annotation_cols = ["ram_index", "annotation", "annotation_source"]
            ram_selection_df = ram_selection_df.merge(
                annotation_df[annotation_cols], on="ram_index", how="left")
            ram_scope = "selected"
        else:
            ram_selection_df = annotation_df.copy()
    elif ram_scope == "selected":
        selected_indices, ram_selection_df = select_gameplay_ram_bytes(
            ram_raw,
            actions,
            n_actions,
            min_bytes=min_ram_bytes,
            max_bytes=max_ram_bytes,
        )
        annotation_cols = ["ram_index", "annotation", "annotation_source"]
        ram_selection_df = ram_selection_df.merge(
            annotation_df[annotation_cols], on="ram_index", how="left")
    elif ram_scope == "all":
        selected_indices = np.arange(ram_raw.shape[1], dtype=np.int32)
        ram_selection_df = annotation_df.copy()
        ram_selection_df["selected"] = True
        ram_selection_df["selected_rank"] = ram_selection_df["ram_index"] + 1
    else:
        raise ValueError(f"Unknown ram_scope: {ram_scope}")

    selected_indices = np.sort(selected_indices)
    ram_selection_df.insert(0, "game_id", game_id)
    ram_selection_df.insert(1, "game_name", name)
    ram_selection_df["ram_scope"] = ram_scope
    ram_selection_df["n_selected"] = len(selected_indices)

    ram = ram_raw[:, selected_indices]
    selected_str = ",".join(str(i) for i in selected_indices.tolist())
    print(f"  RAM scope: {ram_scope} ({len(selected_indices)}/{ram_raw.shape[1]} bytes)")
    print(f"  RAM byte indices I_g: [{selected_str}]")

    # --- Step complexity metrics ---
    print("  Computing state complexity metrics...")
    H = byte_entropy(ram)
    delta = hamming_change_rate(ram)
    alpha = active_byte_fraction(ram)
    var_b = byte_variance(ram)
    lz = lz_complexity_batch(ram, max_steps=3000)
    eff_rank_pca = pca_eff_rank(ram, window=200, step=50)

    complexity_df = pd.DataFrame({
        "game_id": game_id,
        "game_name": name,
        "ram_scope": ram_scope,
        "n_ram_bytes": len(selected_indices),
        "H_byte": H,
        "delta_hamming": delta,
        "alpha_active": alpha,
        "var_byte": var_b,
        "lz_norm": lz,
        "eff_rank_pca": eff_rank_pca,
    })

    # --- Global Gramian ---
    print("  Computing global empirical Gramian...")
    delta_s_global = np.diff(ram.astype(np.float32), axis=0) / 255.0
    acts_global = actions[:-1]  # action u_t aligned to Delta z_t = z_{t+1} - z_t
    # sample to cap at 200K for speed
    idx = np.arange(len(delta_s_global))
    if len(idx) > 200_000:
        idx = np.random.choice(idx, size=200_000, replace=False)
        idx.sort()
    B_global = empirical_B(delta_s_global[idx], acts_global[idx], n_actions)
    global_metrics = gramian_metrics(B_global)
    global_metrics["game_id"] = game_id
    global_metrics["game_name"] = name
    global_metrics["ram_scope"] = ram_scope
    global_metrics["n_ram_bytes"] = len(selected_indices)
    global_metrics["ram_indices"] = selected_str
    global_metrics["n_steps"] = T

    # Eigenvalue spectrum of B for detailed plot
    sv = np.linalg.svd(B_global, compute_uv=False)

    # --- Sliding window Gramian ---
    print("  Computing sliding-window Gramian...")
    # Use first contiguous block for temporal analysis
    ram_block_raw, acts_block = load_game_data(game_id, max_blocks=1)
    ram_block = ram_block_raw[:, selected_indices]
    sliding_df = sliding_gramian(ram_block, acts_block, n_actions, window=60, step=10)
    sliding_df["game_id"] = game_id
    sliding_df["game_name"] = name
    sliding_df["ram_scope"] = ram_scope
    sliding_df["n_ram_bytes"] = len(selected_indices)

    # --- Correlation: complexity vs Gramian per window ---
    print("  Computing complexity-Gramian correlations...")
    corr_records = []
    Tw = len(ram_block)
    H_b = byte_entropy(ram_block)
    lz_b = lz_complexity_batch(ram_block, max_steps=2000)
    for _, row in sliding_df.iterrows():
        t = int(row["t"])
        lo = max(0, t - 150)
        hi = min(Tw, t + 150)
        corr_records.append({
            "game_id": game_id,
            "game_name": name,
            "ram_scope": ram_scope,
            "n_ram_bytes": len(selected_indices),
            "t": t,
            "mean_H_byte": float(H_b[lo:hi].mean()),
            "mean_lz": float(lz_b[lo:hi].mean()),
            "tr_Wc": row["tr_Wc"],
            "eff_rank_Wc": row["eff_rank_Wc"],
            "log_det_Wc": row["log_det_Wc"],
        })
    corr_df = pd.DataFrame(corr_records)

    print(f"  Done. Global metrics: tr={global_metrics['tr_Wc']:.4f}, "
          f"eff_rank={global_metrics['eff_rank_Wc']:.2f}, "
          f"cond={global_metrics['cond_nz']:.1f}")

    return {
        "complexity_df": complexity_df,
        "ram_selection_df": ram_selection_df,
        "global_metrics": global_metrics,
        "sv_global": sv,
        "sliding_df": sliding_df,
        "corr_df": corr_df,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

COLORS = {gid: GAME_INFO[gid]["color"] for gid in GAME_INFO}


def fig_eigenspectra_per_subject(
    subject_results: Dict,
    game_ids: List[int],
    out_dir: Path,
):
    """Per-subject B-matrix eigenspectrum, one panel per game."""
    n_games = len(game_ids)
    fig, axes = plt.subplots(1, n_games, figsize=(6 * n_games, 5), squeeze=False)
    axes = axes[0]

    sub_names = sorted(subject_results.keys())
    cmap = plt.get_cmap("tab10")
    sub_colors = {sub: cmap(i % 10) for i, sub in enumerate(sub_names)}

    for ax, gid in zip(axes, game_ids):
        name = GAME_INFO[gid]["name"]
        max_sv_len = 0
        for sub_name in sub_names:
            res = subject_results.get(sub_name, {}).get(gid, {})
            if not res:
                continue
            sv = res["sv_global"]
            sv_norm = sv / (sv.sum() + 1e-12)
            max_sv_len = max(max_sv_len, len(sv_norm))
            ax.plot(
                np.arange(1, len(sv_norm) + 1),
                sv_norm,
                "o-",
                color=sub_colors[sub_name],
                label=sub_name,
                linewidth=1.5,
                markersize=5,
                alpha=0.8,
            )
        ax.set_xlabel("Singular value index (action dimension)", fontsize=10)
        ax.set_ylabel("Normalized singular value", fontsize=10)
        ax.set_title(f"{name}\nSingular Values of Empirical B (per subject)", fontsize=11)
        if max_sv_len > 0:
            ax.set_xticks(np.arange(1, max_sv_len + 1))
        ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Per-Subject Controllability B-Matrix Eigenspectrum", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "fig_4_2_gramian_eigenspectra.png", dpi=150)
    plt.close(fig)
    print("  Saved fig_4_2")


def fig_sliding_per_subject(
    subject_results: Dict,
    game_ids: List[int],
    out_dir: Path,
):
    """Per-subject first-block sliding Gramian, subjects × games grid."""
    sub_names = sorted(subject_results.keys())
    n_subs = len(sub_names)
    n_games = len(game_ids)

    fig, axes = plt.subplots(
        n_subs, n_games,
        figsize=(7 * n_games, 3 * n_subs),
        squeeze=False,
    )

    for row_idx, sub_name in enumerate(sub_names):
        for col_idx, gid in enumerate(game_ids):
            ax = axes[row_idx][col_idx]
            res = subject_results.get(sub_name, {}).get(gid, {})
            if not res or not isinstance(res.get("sliding_df"), pd.DataFrame) or res["sliding_df"].empty:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", fontsize=9)
                if row_idx == 0:
                    ax.set_title(GAME_INFO[gid]["name"], fontsize=11, fontweight="bold")
                ax.set_yticks([])
            else:
                df = res["sliding_df"]
                color = COLORS[gid]
                ax.plot(df["t"], df["tr_Wc"], color=color, linewidth=1.2)
                ax.fill_between(df["t"], 0, df["tr_Wc"], color=color, alpha=0.2)
                ax.set_xlabel("Timestep", fontsize=8)
                ax.tick_params(labelsize=7)
                if row_idx == 0:
                    ax.set_title(GAME_INFO[gid]["name"], fontsize=11, fontweight="bold")

                # Overlay score / kill events from full RAM
                rb = res.get("ram_first_block_raw")
                if rb is not None and len(rb) > 1:
                    if gid == 1:  # Pong: byte 13 = player score, byte 14 = opponent score
                        player_frames = np.where(np.diff(rb[:, 13].astype(int)) > 0)[0]
                        opp_frames    = np.where(np.diff(rb[:, 14].astype(int)) > 0)[0]
                        for f in player_frames:
                            ax.axvline(f, color="#27ae60", alpha=0.7, linewidth=0.9,
                                       linestyle="--", zorder=3)
                        for f in opp_frames:
                            ax.axvline(f, color="#e74c3c", alpha=0.7, linewidth=0.9,
                                       linestyle="--", zorder=3)
                        if row_idx == 0:
                            ax.set_title(
                                f"{GAME_INFO[gid]['name']}  (-- player score / -- opp score)",
                                fontsize=10, fontweight="bold",
                            )
                    elif gid == 2:  # SpaceInvaders: byte 17 alive_count step-down
                        alive = rb[:, 17].astype(float)
                        init = alive[0] if alive[0] > 0 else 1.0
                        ax2 = ax.twinx()
                        ax2.step(np.arange(len(alive)), alive / init,
                                 where="post", color="gray", alpha=0.45,
                                 linewidth=0.8)
                        ax2.set_ylabel("alive frac.", fontsize=6, color="gray")
                        ax2.tick_params(labelsize=5, labelcolor="gray")
                        ax2.set_ylim(-0.05, 1.3)

            ax.set_ylabel(
                f"Trace(W_c)\n{sub_name}", fontsize=8,
            )

    fig.suptitle(
        "Local Controllability per Subject  (W=60 steps, first block example)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out_dir / "fig_4_3_sliding_controllability.png", dpi=150)
    plt.close(fig)
    print("  Saved fig_4_3")


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def summary_stats_per_subject(
    subject_results: Dict,
    game_ids: List[int],
    game_indices: Dict[int, np.ndarray],
) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("  STATE COMPLEXITY & CONTROLLABILITY GRAMIAN — SUMMARY")
    lines.append("=" * 70)
    lines.append("")

    # RAM byte subset
    lines.append("--- RAM Byte Subset Used for Analysis ---")
    for gid in game_ids:
        sel = game_indices.get(gid, np.array([]))
        lines.append(
            f"{GAME_INFO[gid]['name']:<16} {len(sel):>4} bytes: {sel.tolist()}"
        )
    lines.append("")

    # Per-subject, per-game global Gramian
    lines.append("--- Per-Subject Global Gramian ---")
    header = (
        f"{'Subject':<10} {'Game':<16}"
        f" {'tr(Wc)':>12} {'eff_rank':>10} {'cond':>12}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for sub_name in sorted(subject_results.keys()):
        for gid in game_ids:
            res = subject_results[sub_name].get(gid, {})
            if not res:
                continue
            gm = res["global_metrics"]
            lines.append(
                f"{sub_name:<10} {GAME_INFO[gid]['name']:<16}"
                f" {gm['tr_Wc']:>12.4e}"
                f" {gm['eff_rank_Wc']:>10.2f}"
                f" {gm['cond_nz']:>12.1f}"
            )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="State complexity & Gramian analysis")
    parser.add_argument("--games", type=int, nargs="+", default=[1, 2],
                        help="Game IDs to analyze (default: 1 2  i.e. Pong, SpaceInvaders)")
    parser.add_argument("--ram-scope", choices=["annotated", "selected", "all"],
                        default="annotated",
                        help="Use annotated, data-selected, or all 128 RAM bytes")
    parser.add_argument("--min-ram-bytes", type=int, default=8,
                        help="Minimum selected RAM bytes when --ram-scope selected")
    parser.add_argument("--max-ram-bytes", type=int, default=64,
                        help="Maximum selected RAM bytes when --ram-scope selected")
    args = parser.parse_args()

    np.random.seed(42)

    out_dir = OUT_DIR
    (out_dir / "results").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    # Step 1: determine RAM byte indices per game using pooled data
    print("Determining RAM byte indices per game...")
    game_indices: Dict[int, np.ndarray] = {}
    for gid in args.games:
        info = GAME_INFO[gid]
        ram_raw, actions = load_game_data(gid, max_blocks=5)
        if len(ram_raw) == 0:
            continue
        annotation_df = ram_annotation_metadata(gid, ram_raw.shape[1])
        if args.ram_scope == "annotated":
            sel = annotation_df.loc[annotation_df["selected"], "ram_index"].to_numpy(dtype=np.int32)
            if len(sel) == 0:
                sel, _ = select_gameplay_ram_bytes(
                    ram_raw, actions, info["n_actions"],
                    min_bytes=args.min_ram_bytes, max_bytes=args.max_ram_bytes,
                )
        elif args.ram_scope == "selected":
            sel, _ = select_gameplay_ram_bytes(
                ram_raw, actions, info["n_actions"],
                min_bytes=args.min_ram_bytes, max_bytes=args.max_ram_bytes,
            )
        else:
            sel = np.arange(ram_raw.shape[1], dtype=np.int32)
        game_indices[gid] = np.sort(sel)
        idx_str = ",".join(str(i) for i in game_indices[gid].tolist())
        print(f"  {info['name']}: {len(sel)} bytes  [{idx_str}]")

    # Step 2: per-subject analysis
    subjects = sorted(
        p for p in DATA_DIR.iterdir()
        if p.is_dir() and p.name.startswith("sub")
    )
    subject_results: Dict[str, Dict[int, Dict]] = {}

    for sub_path in subjects:
        sub_name = sub_path.name
        print(f"\n--- {sub_name} ---")
        subject_results[sub_name] = {}
        for gid in args.games:
            if gid not in game_indices:
                continue
            info = GAME_INFO[gid]
            selected_indices = game_indices[gid]
            print(f"  {info['name']} ({len(selected_indices)} bytes)...", end="", flush=True)
            res = analyze_subject_game(
                sub_name, sub_path, gid, selected_indices, info["n_actions"]
            )
            subject_results[sub_name][gid] = res
            if res:
                n_sliding = len(res["sliding_df"]) if isinstance(res.get("sliding_df"), pd.DataFrame) else 0
                print(
                    f" tr={res['global_metrics']['tr_Wc']:.3e},"
                    f" eff_rank={res['global_metrics']['eff_rank_Wc']:.2f},"
                    f" sliding_windows={n_sliding}"
                )
            else:
                print(" no data")

    # Step 3: save CSVs
    print("\nSaving results...")
    global_rows = []
    sliding_dfs = []
    for sub_name, game_res in subject_results.items():
        for gid, res in game_res.items():
            if not res:
                continue
            global_rows.append(res["global_metrics"])
            if isinstance(res.get("sliding_df"), pd.DataFrame) and not res["sliding_df"].empty:
                sliding_dfs.append(res["sliding_df"])

    if global_rows:
        pd.DataFrame(global_rows).to_csv(
            out_dir / "results" / "gramian_global_per_subject.csv", index=False)
    if sliding_dfs:
        pd.concat(sliding_dfs, ignore_index=True).to_csv(
            out_dir / "results" / "gramian_sliding_per_subject.csv", index=False)

    # Step 4: figures
    print("Generating figures...")
    fig_dir = out_dir / "figures"
    fig_eigenspectra_per_subject(subject_results, args.games, fig_dir)
    fig_sliding_per_subject(subject_results, args.games, fig_dir)

    # Step 5: summary
    summary = summary_stats_per_subject(subject_results, args.games, game_indices)
    print(summary)
    (out_dir / "results" / "summary.txt").write_text(summary)

    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
