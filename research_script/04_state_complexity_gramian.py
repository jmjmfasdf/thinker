#!/usr/bin/env python3
"""
04_state_complexity_gramian.py

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
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "04_state_complexity_gramian"

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
        "analysis_indices": [49, 50, 51, 54],
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
            16, 17, 18, 19, 20, 21, 22, 23, 24,
            26, 27, 28, 29, 30,
            43, 44, 45, 46, 47, 48, 49, 50, 51,
            52, 53, 54, 55, 56, 57, 58, 59, 60,
            61, 62, 63, 64, 65, 66, 67, 68, 69,
            73,
            81, 82, 83, 84, 85, 86, 87, 88,
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
    sliding_df = sliding_gramian(ram_block, acts_block, n_actions, window=300, step=50)
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


def fig_complexity_distributions(results: Dict, out_dir: Path):
    metrics = ["H_byte", "delta_hamming", "alpha_active", "var_byte", "lz_norm", "eff_rank_pca"]
    labels = [
        "Byte Entropy (bits)",
        "Hamming Change Rate",
        "Active Byte Fraction",
        "Byte Variance",
        "LZ Complexity (norm.)",
        "PCA Eff. Rank",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for ax, m, lbl in zip(axes, metrics, labels):
        for gid, res in results.items():
            if not res:
                continue
            df = res["complexity_df"]
            vals = df[m].dropna().values
            if len(vals) == 0:
                continue
            name = GAME_INFO[gid]["name"]
            ax.hist(vals, bins=60, alpha=0.55, color=COLORS[gid], label=name, density=True)
        ax.set_xlabel(lbl, fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=7)
        ax.set_title(lbl, fontsize=10)
    fig.suptitle("State Complexity Distributions by Game (RAM-based)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "fig_4_1_complexity_distributions.png", dpi=150)
    plt.close(fig)
    print("  Saved fig_4_1")


def fig_gramian_eigenspectra(results: Dict, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax0, ax1 = axes

    # Left: singular values of B
    for gid, res in results.items():
        if not res:
            continue
        sv = res["sv_global"]
        sv_norm = sv / (sv.sum() + 1e-12)
        name = GAME_INFO[gid]["name"]
        ax0.plot(np.arange(1, len(sv_norm) + 1), sv_norm,
                 "o-", color=COLORS[gid], label=name, linewidth=2, markersize=7)
    ax0.set_xlabel("Singular value index (action dimension)", fontsize=10)
    ax0.set_ylabel("Normalized singular value", fontsize=10)
    ax0.set_title("Singular Values of Empirical B Matrix\n(action → state change)", fontsize=11)
    ax0.legend()
    ax0.set_xticks(np.arange(1, max(len(res["sv_global"]) for res in results.values() if res) + 1))

    # Right: dual-axis bar chart because trace and effective rank have very
    # different numeric scales.
    game_ids_sorted = [gid for gid in sorted(results.keys()) if results[gid]]
    x = np.arange(len(game_ids_sorted))
    width = 0.35
    trace_vals = [results[gid]["global_metrics"]["tr_Wc"] for gid in game_ids_sorted]
    rank_vals = [results[gid]["global_metrics"]["eff_rank_Wc"] for gid in game_ids_sorted]
    labels = [GAME_INFO[gid]["name"] for gid in game_ids_sorted]

    ax1_rank = ax1.twinx()
    trace_bars = ax1.bar(
        x - width / 2,
        trace_vals,
        width,
        color="#34495e",
        alpha=0.85,
        label="Trace(W_c)",
    )
    rank_bars = ax1_rank.bar(
        x + width / 2,
        rank_vals,
        width,
        color="#8e44ad",
        alpha=0.75,
        label="Eff. Rank(W_c)",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("Trace(W_c)  [Total Reachability]", color="#34495e", fontsize=9)
    ax1_rank.set_ylabel("Eff. Rank(W_c)  [Controllable Dims]", color="#8e44ad", fontsize=9)
    ax1.tick_params(axis="y", labelcolor="#34495e")
    ax1_rank.tick_params(axis="y", labelcolor="#8e44ad")
    ax1.set_title("Global Gramian Metrics by Game", fontsize=11)
    ax1.legend([trace_bars, rank_bars], ["Trace(W_c)", "Eff. Rank(W_c)"],
               fontsize=8, loc="upper right")

    fig.suptitle("Empirical Controllability Gramian — Eigenstructure", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "fig_4_2_gramian_eigenspectra.png", dpi=150)
    plt.close(fig)
    print("  Saved fig_4_2")


def fig_sliding_controllability(results: Dict, out_dir: Path):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
    metric = "tr_Wc"
    lbl = "Trace(W_c)  [total reachable energy per window]"
    for ax, (gid, res) in zip(axes, results.items()):
        if not res or res["sliding_df"].empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            continue
        df = res["sliding_df"]
        name = GAME_INFO[gid]["name"]
        ax.plot(df["t"], df[metric], color=COLORS[gid], linewidth=1.5)
        ax.fill_between(df["t"], 0, df[metric], color=COLORS[gid], alpha=0.2)
        ax.set_ylabel(lbl, fontsize=8)
        ax.set_title(f"{name} — Local Controllability (sliding window, W=300 steps)", fontsize=10)
        ax.set_xlabel("Timestep")
    fig.suptitle("Temporal Dynamics of State Controllability", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "fig_4_3_sliding_controllability.png", dpi=150)
    plt.close(fig)
    print("  Saved fig_4_3")


def fig_complexity_vs_gramian(results: Dict, out_dir: Path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    pairs = [
        ("mean_H_byte", "tr_Wc",       "Byte Entropy",     "Trace(W_c)"),
        ("mean_H_byte", "eff_rank_Wc", "Byte Entropy",     "Eff. Rank(W_c)"),
        ("mean_H_byte", "log_det_Wc",  "Byte Entropy",     "Log Det(W_c)"),
        ("mean_lz",     "tr_Wc",       "LZ Complexity",    "Trace(W_c)"),
        ("mean_lz",     "eff_rank_Wc", "LZ Complexity",    "Eff. Rank(W_c)"),
        ("mean_lz",     "log_det_Wc",  "LZ Complexity",    "Log Det(W_c)"),
    ]
    for ax, (xm, ym, xlbl, ylbl) in zip(axes.flatten(), pairs):
        for gid, res in results.items():
            if not res or res["corr_df"].empty:
                continue
            df = res["corr_df"].replace([np.inf, -np.inf], np.nan).dropna(subset=[xm, ym])
            if len(df) < 3:
                continue
            name = GAME_INFO[gid]["name"]
            ax.scatter(df[xm], df[ym], alpha=0.5, s=20, color=COLORS[gid], label=name)
            # regression line
            r, p = scipy_stats.pearsonr(df[xm], df[ym])
            xx = np.linspace(df[xm].min(), df[xm].max(), 50)
            slope, intercept, *_ = scipy_stats.linregress(df[xm], df[ym])
            ax.plot(xx, slope * xx + intercept, "--", color=COLORS[gid], linewidth=1.5,
                    alpha=0.8, label=f"r={r:.2f} (p={p:.3f})")
        ax.set_xlabel(xlbl, fontsize=9)
        ax.set_ylabel(ylbl, fontsize=9)
        ax.legend(fontsize=6.5)
    fig.suptitle("State Complexity vs. Controllability (per sliding window)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "fig_4_4_complexity_vs_gramian.png", dpi=150)
    plt.close(fig)
    print("  Saved fig_4_4")


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def summary_stats(results: Dict) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("  STATE COMPLEXITY & CONTROLLABILITY GRAMIAN — SUMMARY")
    lines.append("=" * 70)
    lines.append("")

    # RAM byte subset table
    lines.append("--- RAM Byte Subset Used for Analysis ---")
    header0 = f"{'Game':<16} {'scope':>10} {'n_bytes':>8}  {'I_g byte indices'}"
    lines.append(header0)
    lines.append("-" * len(header0))
    for gid, res in sorted(results.items()):
        if not res:
            continue
        gm = res["global_metrics"]
        lines.append(
            f"{GAME_INFO[gid]['name']:<16}"
            f" {gm['ram_scope']:>10}"
            f" {gm['n_ram_bytes']:>8}  "
            f"{gm['ram_indices']}"
        )
    lines.append("")

    # Complexity table
    lines.append("--- Median State Complexity per Game ---")
    header = f"{'Game':<16} {'H_byte':>8} {'delta':>8} {'alpha':>8} {'var':>8} {'lz':>8} {'eff_rank':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    for gid, res in sorted(results.items()):
        if not res:
            continue
        df = res["complexity_df"]
        row = (
            f"{GAME_INFO[gid]['name']:<16}"
            f" {df['H_byte'].median():>8.3f}"
            f" {df['delta_hamming'].median():>8.4f}"
            f" {df['alpha_active'].median():>8.4f}"
            f" {df['var_byte'].median():>8.1f}"
            f" {df['lz_norm'].median():>8.4f}"
            f" {df['eff_rank_pca'].median():>8.2f}"
        )
        lines.append(row)
    lines.append("")

    # Gramian table
    lines.append("--- Global Controllability Gramian per Game ---")
    header2 = f"{'Game':<16} {'tr(Wc)':>10} {'lam1':>10} {'lam_min':>10} {'cond':>10} {'eff_rank':>10} {'log_det':>10}"
    lines.append(header2)
    lines.append("-" * len(header2))
    for gid, res in sorted(results.items()):
        if not res:
            continue
        gm = res["global_metrics"]
        row = (
            f"{GAME_INFO[gid]['name']:<16}"
            f" {gm['tr_Wc']:>10.4f}"
            f" {gm['lam1']:>10.4f}"
            f" {gm['lam_min_nz']:>10.4f}"
            f" {gm['cond_nz']:>10.1f}"
            f" {gm['eff_rank_Wc']:>10.2f}"
            f" {gm['log_det_Wc']:>10.2f}"
        )
        lines.append(row)
    lines.append("")

    # Correlation: complexity vs Gramian
    lines.append("--- Pearson r: Byte Entropy vs. Gramian Metrics (per window) ---")
    header3 = f"{'Game':<16} {'H vs tr':>10} {'H vs eff_rank':>14} {'lz vs tr':>10} {'lz vs eff':>10}"
    lines.append(header3)
    lines.append("-" * len(header3))
    for gid, res in sorted(results.items()):
        if not res or res["corr_df"].empty:
            continue
        df = res["corr_df"].replace([np.inf, -np.inf], np.nan).dropna()
        def r(a, b):
            try:
                return f"{scipy_stats.pearsonr(df[a], df[b])[0]:>10.3f}"
            except Exception:
                return f"{'N/A':>10}"
        row = (
            f"{GAME_INFO[gid]['name']:<16}"
            f" {r('mean_H_byte', 'tr_Wc')}"
            f" {r('mean_H_byte', 'eff_rank_Wc'):>14}"
            f" {r('mean_lz', 'tr_Wc')}"
            f" {r('mean_lz', 'eff_rank_Wc')}"
        )
        lines.append(row)
    lines.append("")

    # Theoretical interpretation
    lines.append("--- Theoretical Interpretation ---")
    lines.append("""
Raw RAM notation:
  R_t in {0..255}^128 is the original ALE RAM vector.
  I_g is the selected gameplay/control-relevant byte subset for game g.
      Default scope "annotated" uses RAM_ANNOTATIONS in this script, based on
      OCAtari RAM Extraction Method byte annotations. Scope "selected" uses
      the data-adaptive relevance scorer, and "all" keeps every RAM byte.
  x_t = R_t[I_g] is the selected RAM state used by all reported metrics.
  z_t = x_t / 255 and Delta z_t = z_{t+1} - z_t.

Controllability Gramian W_c = B B^T  (one-step empirical, B estimated via OLS):
  B in R^{|I_g| x n_actions} maps action one-hot u_t to Delta z_t.
  rank(W_c) <= n_actions: upper bound on independently controllable selected
                  RAM dimensions.

  tr(W_c)       = sum of eigenvalues = total "reachable energy" from control inputs.
                  High tr → actions cause large, diverse RAM changes.
  eff_rank(W_c) = exp(entropy of eigenvalue spectrum).
                  Close to n_actions → all action directions equally effective.
                  Close to 1 → one dominant direction, others nearly redundant.
  cond(W_c)     = lam_max / lam_min (non-zero).
                  Low → balanced controllability; High → ill-conditioned (some
                  directions very hard to reach).
  log_det(W_c)  = log volume of reachable state ellipsoid.
                  Higher → larger controllable subspace.

Complexity vs. Gramian:
  Positive r(H_byte, tr_Wc)   → high-entropy states tend to be more controllable.
  Negative r(H_byte, tr_Wc)   → high-entropy states are "stuck" (many active bytes
                                  but few change under action).
""")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="State complexity & Gramian analysis")
    parser.add_argument("--max-blocks", type=int, default=30,
                        help="Max blocks to load per game (default 30)")
    parser.add_argument("--games", type=int, nargs="+", default=[0, 1, 2],
                        help="Game IDs to analyze (default: 0 1 2)")
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

    results = {}
    for gid in args.games:
        results[gid] = analyze_game(
            gid,
            max_blocks=args.max_blocks,
            ram_scope=args.ram_scope,
            min_ram_bytes=args.min_ram_bytes,
            max_ram_bytes=args.max_ram_bytes,
        )

    # Save CSVs
    print("\nSaving results...")
    complexity_dfs = [r["complexity_df"] for r in results.values() if r]
    if complexity_dfs:
        pd.concat(complexity_dfs, ignore_index=True).to_csv(
            out_dir / "results" / "complexity_by_game.csv", index=False)

    selection_dfs = [r["ram_selection_df"] for r in results.values() if r]
    if selection_dfs:
        pd.concat(selection_dfs, ignore_index=True).to_csv(
            out_dir / "results" / "ram_byte_selection.csv", index=False)

    global_rows = [r["global_metrics"] for r in results.values() if r]
    if global_rows:
        pd.DataFrame(global_rows).to_csv(
            out_dir / "results" / "gramian_global.csv", index=False)

    sliding_dfs = [r["sliding_df"] for r in results.values() if r and not r["sliding_df"].empty]
    if sliding_dfs:
        pd.concat(sliding_dfs, ignore_index=True).to_csv(
            out_dir / "results" / "gramian_sliding.csv", index=False)

    corr_dfs = [r["corr_df"] for r in results.values() if r and not r["corr_df"].empty]
    if corr_dfs:
        pd.concat(corr_dfs, ignore_index=True).to_csv(
            out_dir / "results" / "complexity_gramian_corr.csv", index=False)

    # Figures
    print("Generating figures...")
    fig_dir = out_dir / "figures"
    fig_complexity_distributions(results, fig_dir)
    fig_gramian_eigenspectra(results, fig_dir)
    fig_sliding_controllability(results, fig_dir)
    fig_complexity_vs_gramian(results, fig_dir)

    # Summary
    summary = summary_stats(results)
    print(summary)
    (out_dir / "results" / "summary.txt").write_text(summary)

    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
