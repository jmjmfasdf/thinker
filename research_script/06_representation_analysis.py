#!/usr/bin/env python3
"""
Representation -> RAM decoding analysis for filtered thinker traces.

Goal
-----
Use `im_vp_vectors` from `test/sub001` to test whether imagined-state
representations can decode the aligned Atari RAM state (`RAM.txt`) at each
real step.

Key choices
-----------
1. Alignment is done at the real-step level (`status == 0`).
2. For each real step, we collect the preceding imaginary steps
   (`status == 2`) between the previous real step and the current real step.
3. `im_vp_vectors` are spatially pooled to 128 channel features, then
   summarized across all imaginary steps for that real step, or concatenated
   in full with zero-padding in `concat` mode.
4. We decode all 128 RAM addresses with out-of-fold Ridge regression and
   compare the best-decoded slots against game-state RAM addresses highlighted
   in `game_state_noop_analysis.py`.

Outputs
-------
`research_script/outputs/representation_analysis/`
  figures/
    fig_<game>_slot_scores.png
    fig_<game>_top_slots.png
    fig_<game>_key_slot_traces.png
  results/
    <game>_alignment_summary.csv
    <game>_slot_decoding_scores.csv
    <game>_key_slot_summary.csv
    <game>_key_slot_predictions.csv
    summary.txt
"""
from __future__ import annotations

import argparse
import gc
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = ROOT / "test" / "sub001"
DEFAULT_RAM_ROOT = ROOT / "behavioral_data_block_old"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "outputs" / "06_representation_analysis"

EPS = 1e-12

GAME_LABELS = {
    1: "pong",
    2: "spaceinvaders",
}
GAME_TITLES = {
    1: "Pong",
    2: "Space Invaders",
}

# Important RAM addresses from research_script/game_state_noop_analysis.py
KEY_RAM_ADDRESSES: Dict[int, Dict[int, str]] = {
    1: {
        13: "cpu_score",
        14: "player_score",
        49: "ball_x",
        50: "cpu_y",
        51: "player_y",
        54: "ball_y",
    },
    2: {
        17: "enemy_count",
        16: "enemies_y",
        28: "player_x",
        73: "num_lives",
    },
}

# Extra labels for RAM slots that are not part of the original key-state
# analysis, but have interpretable Space Invaders semantics.
ADDITIONAL_RAM_ADDRESSES: Dict[int, Dict[int, str]] = {
    2: {
        0: "internal_scan_counter",
        13: "alien_edge_x_limit",
        18: "alien_row1_bitmap",
        19: "alien_row2_bitmap",
        20: "alien_row3_bitmap",
        21: "alien_row4_bitmap",
        22: "alien_row5_bitmap",
        23: "alien_row6_bitmap",
        42: "game_state_flags",
        89: "internal_cooldown_timer",
        107: "alien_alive_column_mask",
    },
}

SHORT_RAM_SLOT_LABELS: Dict[int, Dict[int, str]] = {
    2: {
        0: "internal",
        13: "edge_x",
        18: "row1_bits",
        19: "row2_bits",
        20: "row3_bits",
        21: "row4_bits",
        22: "row5_bits",
        23: "row6_bits",
        42: "flags",
        89: "timer",
        107: "alive_cols",
    },
}

# High-quality trace plots should focus on environment-state slots. These
# labels are useful in summary plots, but are too internal for trajectory plots.
TRACE_EXCLUDED_RAM_ADDRESSES: Dict[int, set[int]] = {
    2: {0, 89},
}

DEFAULT_TRACE_MIN_R = 0.8


def ram_slot_labels(game: int) -> Dict[int, str]:
    labels = dict(ADDITIONAL_RAM_ADDRESSES.get(game, {}))
    labels.update(KEY_RAM_ADDRESSES.get(game, {}))
    return labels


def ram_slot_display_label(game: int, address: int, slot_name: str, short: bool = False) -> str:
    if short:
        return SHORT_RAM_SLOT_LABELS.get(game, {}).get(address, slot_name or str(address))
    return slot_name or str(address)


@dataclass(frozen=True)
class FileMeta:
    subject: int
    session: int
    block: int
    game: int
    chunk: int
    path: Path


def parse_file_meta(path: Path) -> FileMeta:
    match = re.match(r"sub(\d+)-ses(\d+)-block(\d+)-game(\d+)_(\d+)\.npy$", path.name)
    if match is None:
        raise ValueError(f"Unexpected file name format: {path.name}")
    return FileMeta(
        subject=int(match.group(1)),
        session=int(match.group(2)),
        block=int(match.group(3)),
        game=int(match.group(4)),
        chunk=int(match.group(5)),
        path=path,
    )


def load_npy_dict(path: Path) -> Dict[str, np.ndarray]:
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
        item = obj.item()
        if isinstance(item, dict):
            return item
    if hasattr(obj, "files"):
        return {key: obj[key] for key in obj.files}
    raise ValueError(f"Unable to parse {path} as dict-like npy/npz")


def load_ram_txt(path: Path) -> np.ndarray:
    rows: List[np.ndarray] = []
    with path.open("r") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            _, values = line.split(":", 1)
            arr = np.fromstring(values, sep=",", dtype=np.float32)
            if arr.size == 0:
                continue
            rows.append(arr[:128])
    if not rows:
        raise ValueError(f"No RAM rows parsed from {path}")
    return np.vstack(rows)


def pool_im_vp_vectors(im_vp_vectors: np.ndarray) -> np.ndarray:
    """
    Convert im_vp_vectors to per-step channel features.

    Expected shape in this dataset:
      (T, 1, 128, 6, 6) -> (T, 128) via spatial mean pooling.
    """
    arr = np.asarray(im_vp_vectors)
    if arr.ndim == 5:
        pooled = arr.mean(axis=(-1, -2))
        if pooled.shape[1] == 1:
            pooled = pooled[:, 0, :]
        else:
            pooled = pooled.reshape(pooled.shape[0], -1)
    elif arr.ndim == 4:
        pooled = arr.mean(axis=(-1, -2))
    elif arr.ndim == 3:
        pooled = arr.mean(axis=-1)
    elif arr.ndim == 2:
        pooled = arr
    else:
        pooled = arr.reshape(arr.shape[0], -1)
    return np.asarray(pooled, dtype=np.float32)


def build_step_feature(
    reps: np.ndarray,
    feature_mode: str,
    concat_k: int,
) -> np.ndarray:
    reps = np.asarray(reps, dtype=np.float32)
    if reps.ndim != 2:
        reps = reps.reshape(reps.shape[0], -1)

    if feature_mode == "last":
        feat = np.concatenate(
            [reps[-1], np.array([float(reps.shape[0])], dtype=np.float32)],
            axis=0,
        )
    elif feature_mode == "concat":
        feat_dim = reps.shape[1]
        if reps.shape[0] > concat_k:
            raise ValueError(
                f"concat padding length {concat_k} is smaller than "
                f"required sequence length {reps.shape[0]}"
            )
        padded = np.zeros((concat_k, feat_dim), dtype=np.float32)
        padded[-len(reps) :] = reps
        feat = np.concatenate(
            [padded.reshape(-1), np.array([float(reps.shape[0])], dtype=np.float32)],
            axis=0,
        )
    else:
        feat = np.concatenate(
            [
                reps.mean(axis=0),
                reps.std(axis=0),
                reps.max(axis=0),
                reps[-1],
                np.array([float(reps.shape[0])], dtype=np.float32),
            ],
            axis=0,
        )
    return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def extract_real_step_features(
    meta: FileMeta,
    feature_mode: str,
    concat_k: int,
) -> Tuple[np.ndarray, pd.DataFrame]:
    data = load_npy_dict(meta.path)
    status = np.asarray(data["status"]).reshape(-1)
    pooled = pool_im_vp_vectors(data["im_vp_vectors"])

    # Free unrelated arrays from the loaded dict early.
    del data
    gc.collect()

    t = min(len(status), len(pooled))
    status = status[:t]
    pooled = pooled[:t]

    real_idx = np.flatnonzero(status == 0)
    imag_all = np.flatnonzero(status == 2)
    if real_idx.size == 0:
        raise ValueError(f"No real steps found in {meta.path}")

    features: List[np.ndarray] = []
    rows: List[Dict[str, object]] = []
    episode_in_file = 0
    episode_step = 0

    for real_pos, idx_global in enumerate(real_idx):
        prev_real = int(real_idx[real_pos - 1]) if real_pos > 0 else -1
        between = np.arange(prev_real + 1, int(idx_global), dtype=int)
        imag_idx = between[status[between] == 2] if between.size else np.array([], dtype=int)
        if real_pos > 0 and between.size > 0 and np.isin(status[between], [1, 3]).any():
            episode_in_file += 1
            episode_step = 0

        used_fallback = 0
        if imag_idx.size == 0:
            prev_imag_pos = np.searchsorted(imag_all, idx_global) - 1
            if prev_imag_pos >= 0:
                imag_idx = np.array([imag_all[prev_imag_pos]], dtype=int)
            else:
                imag_idx = np.array([int(idx_global)], dtype=int)
            used_fallback = 1

        reps = pooled[imag_idx]
        feat = build_step_feature(reps, feature_mode=feature_mode, concat_k=concat_k)
        features.append(feat)

        rows.append(
            {
                "source_file": str(meta.path),
                "subject": meta.subject,
                "session": meta.session,
                "block": meta.block,
                "game": meta.game,
                "chunk": meta.chunk,
                "real_pos_chunk": real_pos,
                "episode_in_file": episode_in_file,
                "episode_step": episode_step,
                "status_idx": int(idx_global),
                "n_imag_steps": int(len(imag_idx)),
                "used_fallback": used_fallback,
            }
        )
        episode_step += 1

    del pooled
    gc.collect()

    return np.vstack(features).astype(np.float32), pd.DataFrame(rows)


def ram_path_from_meta(ram_root: Path, meta: FileMeta) -> Path:
    return ram_root / f"sub_{meta.subject}" / f"game_{meta.game}" / f"day_{meta.session}" / f"block_{meta.block}" / "RAM.txt"


def infer_concat_length(meta: FileMeta) -> int:
    data = load_npy_dict(meta.path)
    status = np.asarray(data["status"]).reshape(-1)
    imag_len = len(data["im_vp_vectors"])
    del data
    gc.collect()

    t = min(len(status), imag_len)
    status = status[:t]
    real_idx = np.flatnonzero(status == 0)
    if real_idx.size == 0:
        raise ValueError(f"No real steps found in {meta.path}")

    max_imag_steps = 1
    for real_pos, idx_global in enumerate(real_idx):
        prev_real = int(real_idx[real_pos - 1]) if real_pos > 0 else -1
        between = np.arange(prev_real + 1, int(idx_global), dtype=int)
        imag_count = int(np.sum(status[between] == 2)) if between.size else 0
        max_imag_steps = max(max_imag_steps, max(1, imag_count))
    return max_imag_steps


def gather_input_files(input_dir: Path, game_id: int | None = None) -> List[FileMeta]:
    files: List[FileMeta] = []
    for path in sorted(input_dir.rglob("*.npy")):
        try:
            meta = parse_file_meta(path)
        except ValueError:
            continue
        if game_id is not None and meta.game != game_id:
            continue
        files.append(meta)
    return files


def build_aligned_datasets(
    input_dir: Path,
    ram_root: Path,
    feature_mode: str,
    concat_k: int,
    game_id: int | None = None,
) -> Tuple[Dict[int, Dict[str, object]], pd.DataFrame]:
    metas = gather_input_files(input_dir, game_id=game_id)
    if not metas:
        raise FileNotFoundError(f"No matching .npy files found under {input_dir}")

    effective_concat_k = concat_k
    if feature_mode == "concat":
        concat_lengths = [infer_concat_length(meta) for meta in metas]
        effective_concat_k = max(concat_lengths) if concat_lengths else 1
        print(
            f"[concat] using all imaginary steps per real step; "
            f"padding to max_len={effective_concat_k}"
        )

    grouped: Dict[Tuple[int, int, int, int], List[FileMeta]] = {}
    for meta in metas:
        grouped.setdefault((meta.subject, meta.session, meta.block, meta.game), []).append(meta)

    per_game: Dict[int, Dict[str, List[object]]] = {}
    alignment_rows: List[Dict[str, object]] = []

    for key in sorted(grouped):
        files = sorted(grouped[key], key=lambda item: item.chunk)
        first_meta = files[0]
        ram_path = ram_path_from_meta(ram_root, first_meta)
        if not ram_path.exists():
            raise FileNotFoundError(f"Expected RAM file not found: {ram_path}")

        y_block = load_ram_txt(ram_path)
        ram_offset = 0
        block_group = (
            f"sub{first_meta.subject:03d}_ses{first_meta.session:02d}_"
            f"block{first_meta.block:02d}_game{first_meta.game}"
        )

        print(
            f"[load] subject={first_meta.subject:03d} ses={first_meta.session:02d} "
            f"block={first_meta.block} game={first_meta.game} chunks={len(files)}"
        )

        for meta in files:
            features, meta_df = extract_real_step_features(
                meta=meta,
                feature_mode=feature_mode,
                concat_k=effective_concat_k,
            )
            n_features_file = len(features)
            n_ram_total = len(y_block)
            n_ram_remaining = max(0, n_ram_total - ram_offset)
            n_used_file = min(n_features_file, n_ram_remaining)

            print(
                f"  - {meta.path.name}: real_steps={len(meta_df):,} "
                f"mean_imag={meta_df['n_imag_steps'].mean():.2f} "
                f"used_ram={n_used_file:,}"
            )

            if n_used_file > 0:
                x_file = features[:n_used_file].astype(np.float32)
                y_file = y_block[ram_offset : ram_offset + n_used_file].astype(np.float32)
                meta_file = meta_df.iloc[:n_used_file].reset_index(drop=True).copy()
                meta_file["block_group"] = block_group
                meta_file["source_file_name"] = meta.path.name
                meta_file["ram_step_block"] = np.arange(ram_offset, ram_offset + n_used_file, dtype=int)
                meta_file["aligned_real_pos_global"] = np.arange(ram_offset, ram_offset + n_used_file, dtype=int)

                per_game.setdefault(
                    first_meta.game,
                    {"X": [], "Y": [], "meta": []},
                )
                per_game[first_meta.game]["X"].append(x_file)
                per_game[first_meta.game]["Y"].append(y_file)
                per_game[first_meta.game]["meta"].append(meta_file)

            alignment_rows.append(
                {
                    "subject": first_meta.subject,
                    "session": first_meta.session,
                    "block": first_meta.block,
                    "game": first_meta.game,
                    "block_group": block_group,
                    "source_file": str(meta.path),
                    "source_file_name": meta.path.name,
                    "n_chunk_files_in_block": len(files),
                    "n_real_feature_steps": n_features_file,
                    "n_ram_steps_block": n_ram_total,
                    "ram_offset_start": ram_offset,
                    "ram_offset_stop": ram_offset + n_used_file,
                    "n_used_steps": n_used_file,
                    "trimmed_feature_steps": n_features_file - n_used_file,
                    "unused_ram_steps_after_file": n_ram_total - (ram_offset + n_used_file),
                    "mean_imag_steps": float(meta_df["n_imag_steps"].mean()),
                    "fallback_rate": float(meta_df["used_fallback"].mean()),
                    "ram_path": str(ram_path),
                }
            )

            ram_offset += n_used_file

    out: Dict[int, Dict[str, object]] = {}
    for game, pieces in per_game.items():
        out[game] = {
            "X": np.vstack(pieces["X"]).astype(np.float32),
            "Y": np.vstack(pieces["Y"]).astype(np.float32),
            "meta": pd.concat(pieces["meta"], ignore_index=True),
        }

    return out, pd.DataFrame(alignment_rows)


def build_splits(
    x: np.ndarray,
    groups: Sequence[str],
    max_splits: int,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], str]:
    groups_arr = np.asarray(groups)
    unique_groups = np.unique(groups_arr)

    if unique_groups.size >= 2:
        splitter = GroupKFold(n_splits=min(max_splits, unique_groups.size))
        splits = list(splitter.split(x, groups=groups_arr))
        return splits, f"GroupKFold(groups={unique_groups.size})"

    n_splits = min(max_splits, max(2, min(5, len(x))))
    splitter = KFold(n_splits=n_splits, shuffle=False)
    splits = list(splitter.split(x))
    return splits, f"KFold(n_splits={n_splits})"


def fit_oof_ridge(
    x: np.ndarray,
    y: np.ndarray,
    groups: Sequence[str],
    alpha: float,
    max_splits: int,
) -> Tuple[np.ndarray, str]:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    splits, splitter_name = build_splits(x, groups=groups, max_splits=max_splits)

    y_pred = np.full_like(y, np.nan, dtype=np.float32)

    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        scaler = StandardScaler(with_mean=True, with_std=True)
        x_train = scaler.fit_transform(x[train_idx])
        x_test = scaler.transform(x[test_idx])

        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(x_train, y[train_idx])
        y_pred[test_idx] = model.predict(x_test).astype(np.float32)

        print(
            f"  [fold {fold_idx}/{len(splits)}] train={len(train_idx):,} test={len(test_idx):,}"
        )

    if np.isnan(y_pred).any():
        raise RuntimeError("Out-of-fold predictions contain NaNs; split coverage failed.")

    return y_pred, splitter_name


def safe_pearsonr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0 or np.nanstd(y_true) < EPS or np.nanstd(y_pred) < EPS:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def majority_accuracy(y_true: np.ndarray) -> float:
    values, counts = np.unique(y_true.astype(np.int64), return_counts=True)
    if values.size == 0:
        return float("nan")
    return float(counts.max() / counts.sum())


def compute_slot_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    game: int,
) -> pd.DataFrame:
    key_map = KEY_RAM_ADDRESSES.get(game, {})
    label_map = ram_slot_labels(game)
    trace_excluded = TRACE_EXCLUDED_RAM_ADDRESSES.get(game, set())
    rows: List[Dict[str, object]] = []

    for address in range(y_true.shape[1]):
        yt = y_true[:, address].astype(np.float64)
        yp = y_pred[:, address].astype(np.float64)
        rounded = np.clip(np.rint(yp), 0, 255)
        value_range = float(np.max(yt) - np.min(yt))
        acc = float(np.mean(rounded == yt))
        maj_acc = majority_accuracy(yt)

        rows.append(
            {
                "address": address,
                "slot_name": label_map.get(address, ""),
                "is_key_slot": int(address in key_map),
                "is_annotated_slot": int(address in label_map),
                "trace_excluded": int(address in trace_excluded),
                "n_unique": int(np.unique(yt).size),
                "value_min": float(np.min(yt)),
                "value_max": float(np.max(yt)),
                "value_range": value_range,
                "pearson_r": safe_pearsonr(yt, yp),
                "r2": float(r2_score(yt, yp)) if value_range > 0 else float("nan"),
                "mae": float(mean_absolute_error(yt, yp)),
                "normalized_mae": float(mean_absolute_error(yt, yp) / value_range)
                if value_range > 0
                else float("nan"),
                "rounded_acc": acc,
                "majority_acc": maj_acc,
                "acc_gain": float(acc - maj_acc),
            }
        )

    df = pd.DataFrame(rows).sort_values("address").reset_index(drop=True)
    df["pearson_rank"] = (
        df["pearson_r"].fillna(-np.inf).rank(method="min", ascending=False).astype(int)
    )
    df["r2_rank"] = df["r2"].fillna(-np.inf).rank(method="min", ascending=False).astype(int)
    df["acc_gain_rank"] = (
        df["acc_gain"].fillna(-np.inf).rank(method="min", ascending=False).astype(int)
    )
    return df


def summarize_game(
    metrics_df: pd.DataFrame,
    alignment_df: pd.DataFrame,
    game: int,
    splitter_name: str,
    feature_mode: str,
    alpha: float,
) -> List[str]:
    top10 = metrics_df.sort_values("pearson_r", ascending=False).head(10)
    key_df = metrics_df[metrics_df["is_key_slot"] == 1].sort_values("pearson_rank")
    key_hits_top10 = int(top10["is_key_slot"].sum())
    n_blocks = int(alignment_df["block_group"].nunique()) if "block_group" in alignment_df.columns else len(alignment_df)
    n_files = int(alignment_df["source_file"].nunique()) if "source_file" in alignment_df.columns else len(alignment_df)

    lines = [
        f"[{GAME_TITLES.get(game, f'Game {game}')}]",
        f"- feature_mode: {feature_mode}",
        f"- decoder: Ridge(alpha={alpha}) with {splitter_name}",
        f"- aligned blocks: {n_blocks}",
        f"- aligned files: {n_files}",
        f"- used steps: {int(alignment_df['n_used_steps'].sum())}",
        f"- top-10 pearson slots containing key slots: {key_hits_top10}",
        (
            "- mean pearson_r (key vs non-key): "
            f"{key_df['pearson_r'].mean():.3f} vs "
            f"{metrics_df.loc[metrics_df['is_key_slot'] == 0, 'pearson_r'].mean():.3f}"
        ),
        "- top decoded slots by pearson_r:",
    ]

    for _, row in top10.iterrows():
        label = row["slot_name"] if row["slot_name"] else "other"
        lines.append(
            f"  addr {int(row['address']):3d} ({label}): "
            f"r={row['pearson_r']:.3f}, R2={row['r2']:.3f}, "
            f"acc={row['rounded_acc']:.3f}, acc_gain={row['acc_gain']:.3f}"
        )

    if not key_df.empty:
        lines.append("- key game-state slots:")
        for _, row in key_df.iterrows():
            lines.append(
                f"  addr {int(row['address']):3d} ({row['slot_name']}): "
                f"rank={int(row['pearson_rank'])}, "
                f"r={row['pearson_r']:.3f}, R2={row['r2']:.3f}, "
                f"acc={row['rounded_acc']:.3f}, acc_gain={row['acc_gain']:.3f}"
            )

    lines.append("")
    return lines


def plot_slot_scores(
    metrics_df: pd.DataFrame,
    game: int,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))

    annotated = metrics_df[
        (metrics_df["is_annotated_slot"] == 1) & (metrics_df["is_key_slot"] == 0)
    ]
    other = metrics_df[metrics_df["is_annotated_slot"] == 0]
    key = metrics_df[metrics_df["is_key_slot"] == 1]
    highlight = pd.concat(
        [
            metrics_df.loc[metrics_df["pearson_r"] >= DEFAULT_TRACE_MIN_R],
            key,
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["address"])

    ax.axhline(0.0, color="#9ca3af", lw=1.0, ls="--")
    ax.scatter(
        other["address"],
        other["pearson_r"],
        s=18,
        color="#9ca3af",
        alpha=0.85,
        label="Other RAM slots",
    )
    if not annotated.empty:
        ax.scatter(
            annotated["address"],
            annotated["pearson_r"],
            s=36,
            color="#2563eb",
            alpha=0.95,
            label="Annotated RAM slots",
            zorder=3,
        )
    if not key.empty:
        ax.scatter(
            key["address"],
            key["pearson_r"],
            s=48,
            color="#d97706",
            alpha=0.95,
            label="Key game-state slots",
            zorder=4,
        )

    label_offsets = {
        16: (-34, 46),
        17: (-4, 64),
        18: (28, 34),
        19: (52, 54),
        20: (58, 16),
        21: (78, 32),
        22: (96, 50),
        23: (92, 12),
    }
    for i, (_, row) in enumerate(highlight.sort_values("address").iterrows()):
        address = int(row["address"])
        slot_name = row["slot_name"] if isinstance(row["slot_name"], str) else ""
        label = ram_slot_display_label(game, address, slot_name, short=True)
        offset = label_offsets.get(address, (0, 8 + 8 * (i % 3)))
        arrowprops = (
            {
                "arrowstyle": "-",
                "color": "#6b7280",
                "lw": 0.5,
                "alpha": 0.6,
                "shrinkA": 0,
                "shrinkB": 3,
            }
            if address in label_offsets
            else None
        )
        ax.annotate(
            label,
            xy=(float(address), float(row["pearson_r"])),
            xytext=offset,
            textcoords="offset points",
            fontsize=7,
            rotation=35,
            ha="left",
            va="bottom",
            arrowprops=arrowprops,
            annotation_clip=False,
        )

    ax.set_xlim(-2, 129)
    ymin = min(-0.12, float(metrics_df["pearson_r"].min()) - 0.03)
    ax.set_ylim(ymin, 1.18)
    ax.set_xlabel("RAM address")
    ax.set_ylabel("Out-of-fold Pearson r")
    ax.set_title(
        f"{GAME_TITLES.get(game, f'Game {game}')}: decoding quality by RAM address",
        pad=18,
    )
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_top_slots(
    metrics_df: pd.DataFrame,
    game: int,
    out_path: Path,
    top_k: int = 15,
) -> None:
    top = metrics_df.sort_values(["pearson_r", "acc_gain"], ascending=False).head(top_k)
    top = top.iloc[::-1].reset_index(drop=True)

    labels = []
    for row in top.itertuples():
        slot_name = row.slot_name if isinstance(row.slot_name, str) else ""
        labels.append(
            f"{int(row.address):03d}  {slot_name}" if slot_name else f"{int(row.address):03d}"
        )
    values = top["pearson_r"].to_numpy(dtype=float)
    value_min = float(np.nanmin(values))
    value_max = float(np.nanmax(values))
    scale = (values - value_min) / max(value_max - value_min, EPS)
    cmap = plt.get_cmap("Blues")
    colors = [cmap(0.32 + 0.58 * val) for val in scale]

    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    bars = ax.barh(labels, top["pearson_r"], height=0.56, color=colors, edgecolor="none")
    for bar, value in zip(bars, values):
        ax.text(
            float(value) + 0.006,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=8.5,
            color="#111827",
        )

    ax.set_xlim(0.0, min(1.0, value_max + 0.05))
    ax.set_xlabel("Out-of-fold Pearson r")
    ax.set_title(f"{GAME_TITLES.get(game, f'Game {game}')}: top decoded RAM slots", pad=12)
    ax.xaxis.grid(True, color="#d1d5db", lw=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#9ca3af")
    fig.tight_layout()
    fig.subplots_adjust(left=0.34, right=0.96)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_key_slot_traces(
    meta_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics_df: pd.DataFrame,
    game: int,
    out_dir: Path,
    game_label: str,
    n_steps: int,
    min_pearson_r: float,
) -> None:
    trace_df = metrics_df[
        (metrics_df["is_annotated_slot"] == 1)
        & (metrics_df["trace_excluded"] == 0)
        & (metrics_df["pearson_r"] >= min_pearson_r)
        & (metrics_df["value_range"] > 0)
    ].sort_values(["pearson_r", "address"], ascending=[False, True])
    if trace_df.empty:
        return

    items = [
        (int(row["address"]), str(row["slot_name"]))
        for _, row in trace_df.iterrows()
    ]
    n_plots = len(items)
    n_cols = 3 if n_plots > 6 else 2
    n_rows = math.ceil(n_plots / n_cols)
    total_n = min(len(meta_df), len(y_true), len(y_pred))
    meta_show = meta_df.iloc[:total_n].reset_index(drop=True).copy()
    file_groups = list(meta_show.groupby("source_file", sort=False))

    for source_file, file_group in file_groups:
        if file_group.empty:
            continue

        orig_idx = file_group.sort_values("real_pos_chunk").index.to_numpy()
        x = np.arange(len(orig_idx), dtype=int)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3.8 * n_rows), squeeze=False)
        axes_flat = axes.flatten()

        for ax, (address, name) in zip(axes_flat, items):
            row = metrics_df.loc[metrics_df["address"] == address].iloc[0]
            ax.plot(x, y_true[orig_idx, address], color="#111827", lw=1.0, ls="-", label="RAM")
            ax.plot(x, y_pred[orig_idx, address], color="#2563eb", lw=1.0, ls="--", alpha=0.95, label="decoded")
            ax.set_title(
                f"addr {address}: {name}\n"
                f"r={row['pearson_r']:.3f}, R2={row['r2']:.3f}, acc={row['rounded_acc']:.3f}"
            )
            ax.set_xlabel("Real-step index within selected episode")
            ax.set_ylabel("RAM value")

        for ax in axes_flat[n_plots:]:
            ax.axis("off")

        style_handles = [
            Line2D([0], [0], color="#111827", lw=1.8, ls="-", label="RAM"),
            Line2D([0], [0], color="#2563eb", lw=1.8, ls="--", label="decoded"),
        ]
        fig.legend(
            handles=style_handles,
            frameon=False,
            loc="upper right",
            bbox_to_anchor=(0.98, 1.02),
            ncol=1,
            title="Line type",
        )
        fig.suptitle(
            f"{GAME_TITLES.get(game, f'Game {game}')}: high-quality annotated RAM traces\n"
            f"{Path(source_file).stem}; r >= {min_pearson_r:.2f}, internal slots excluded",
            y=0.995,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out_path = out_dir / f"fig_{game_label}_key_slot_trace_{Path(source_file).stem}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)


def save_key_predictions(
    meta_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    game: int,
    out_path: Path,
) -> None:
    key_map = KEY_RAM_ADDRESSES.get(game, {})
    if not key_map:
        return

    out_df = meta_df.copy()
    for address, name in key_map.items():
        out_df[f"ram_{address}_{name}_true"] = y_true[:, address]
        out_df[f"ram_{address}_{name}_pred"] = y_pred[:, address]
    out_df.to_csv(out_path, index=False)


def ensure_dirs(base_out_dir: Path) -> Tuple[Path, Path]:
    fig_dir = base_out_dir / "figures"
    res_dir = base_out_dir / "results"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir, res_dir


def run_analysis(args: argparse.Namespace) -> None:
    fig_dir, res_dir = ensure_dirs(args.output_dir)
    datasets, alignment_df = build_aligned_datasets(
        input_dir=args.input_dir,
        ram_root=args.ram_root,
        feature_mode=args.feature_mode,
        concat_k=args.concat_k,
        game_id=args.game_id,
    )

    summary_lines: List[str] = []

    for game in sorted(datasets):
        game_label = GAME_LABELS.get(game, f"game{game}")
        game_alignment = alignment_df[alignment_df["game"] == game].copy()
        x = datasets[game]["X"]
        y = datasets[game]["Y"]
        meta_df = datasets[game]["meta"].copy()
        groups = meta_df["source_file"].to_numpy()

        print(
            f"[decode] {GAME_TITLES.get(game, f'Game {game}')}: "
            f"steps={len(x):,}, features={x.shape[1]:,}, targets={y.shape[1]}"
        )
        y_pred, splitter_name = fit_oof_ridge(
            x=x,
            y=y,
            groups=groups,
            alpha=args.alpha,
            max_splits=args.max_splits,
        )

        metrics_df = compute_slot_metrics(y_true=y, y_pred=y_pred, game=game)
        key_df = metrics_df[metrics_df["is_key_slot"] == 1].copy()

        game_alignment.to_csv(res_dir / f"{game_label}_alignment_summary.csv", index=False)
        metrics_df.to_csv(res_dir / f"{game_label}_slot_decoding_scores.csv", index=False)
        key_df.to_csv(res_dir / f"{game_label}_key_slot_summary.csv", index=False)
        save_key_predictions(
            meta_df=meta_df,
            y_true=y,
            y_pred=y_pred,
            game=game,
            out_path=res_dir / f"{game_label}_key_slot_predictions.csv",
        )

        plot_slot_scores(
            metrics_df=metrics_df,
            game=game,
            out_path=fig_dir / f"fig_{game_label}_slot_scores.png",
        )
        plot_top_slots(
            metrics_df=metrics_df,
            game=game,
            out_path=fig_dir / f"fig_{game_label}_top_slots.png",
            top_k=args.top_k,
        )
        plot_key_slot_traces(
            meta_df=meta_df,
            y_true=y,
            y_pred=y_pred,
            metrics_df=metrics_df,
            game=game,
            out_dir=fig_dir,
            game_label=game_label,
            n_steps=args.trace_steps,
            min_pearson_r=args.trace_min_r,
        )

        summary_lines.extend(
            summarize_game(
                metrics_df=metrics_df,
                alignment_df=game_alignment,
                game=game,
                splitter_name=splitter_name,
                feature_mode=args.feature_mode,
                alpha=args.alpha,
            )
        )

    summary_path = res_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines).rstrip() + "\n")
    print(f"[done] wrote summary to {summary_path}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing filtered .npy files (default: test/sub001).",
    )
    parser.add_argument(
        "--ram-root",
        type=Path,
        default=DEFAULT_RAM_ROOT,
        help="Root directory containing behavioral RAM.txt files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for figures and result tables.",
    )
    parser.add_argument(
        "--game-id",
        type=int,
        default=None,
        help="Optional game filter (e.g. 2 for Space Invaders).",
    )
    parser.add_argument(
        "--feature-mode",
        choices=["moments", "last", "concat"],
        default="moments",
        help="How to summarize all imagined vectors for each real step.",
    )
    parser.add_argument(
        "--concat-k",
        type=int,
        default=40,
        help="Deprecated: concat mode now uses all imagined steps and pads to the run-wise maximum length.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=10.0,
        help="Ridge regularization strength.",
    )
    parser.add_argument(
        "--max-splits",
        type=int,
        default=5,
        help="Maximum number of CV folds.",
    )
    parser.add_argument(
        "--trace-steps",
        type=int,
        default=500,
        help="Kept for backward compatibility; key-slot trace plots now use all aligned steps.",
    )
    parser.add_argument(
        "--trace-min-r",
        type=float,
        default=DEFAULT_TRACE_MIN_R,
        help="Minimum overall Pearson r for annotated RAM slots to appear in trace plots.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="Number of RAM slots to show in the top-slot figure.",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
