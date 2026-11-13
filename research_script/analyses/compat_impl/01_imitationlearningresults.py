#!/usr/bin/env python3
"""01_imitationlearningresults.py

Plot action distribution for imitation learning results.

Usage:
    python 01_imitationlearningresults.py --subject sub001 --game 1
    python 01_imitationlearningresults.py --subject sub001 --game 2
"""
from __future__ import annotations

import argparse
import gc
import glob
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib-cache"
os.environ["XDG_CACHE_HOME"] = "/tmp"
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
DATA_ROOT = ROOT / "test"
OUT_ROOT = Path(__file__).parent / "outputs" / "01_imitationlearningresults"
DEFAULT_RAM_ROOT = ROOT / "behavioral_data_block_old"

GAME_TITLES = {1: "Pong", 2: "Space Invaders"}
GAME_KEYS = {1: "game1", 2: "game2"}

N_ACTIONS = 6
ACTION_LABELS = ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"]

HUMAN_COLOR = "#e15759"
THINKER_COLOR = "#4e79a7"
EPS = 1e-12
DEFAULT_TRACE_MIN_R = 0.8

GAME_LABELS = {1: "pong", 2: "spaceinvaders"}

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

TRACE_EXCLUDED_RAM_ADDRESSES: Dict[int, set[int]] = {
    2: {0, 89},
}

# Labels shown in the whole-RAM decoding figure.  These are deliberately
# prespecified and more restrictive than the annotations used by the key-slot
# bar and trace figures.
RAM_SCORE_LABELS: Dict[int, Dict[int, str]] = {
    1: {
        14: "player score",
        50: "cpu y",
        51: "player y",
        54: "ball y",
        49: "ball x",
    },
    2: {
        17: "enemy count",
        28: "player x",
        16: "enemies y",
    },
}

RAM_SCORE_LABEL_OFFSETS: Dict[int, Dict[int, Tuple[float, float]]] = {
    1: {
        14: (-8.0, 12.0),
        50: (-12.0, 12.0),
        51: (10.0, -18.0),
        54: (12.0, 12.0),
        49: (0.0, 12.0),
    },
    2: {
        17: (10.0, 14.0),
        28: (10.0, 12.0),
        16: (-12.0, -18.0),
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_action_ids(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim == 1:
        return arr.astype(int)
    return np.argmax(arr, axis=1).astype(int)


def accumulate_counts(files: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    human_counts = np.zeros(N_ACTIONS, dtype=np.int64)
    thinker_counts = np.zeros(N_ACTIONS, dtype=np.int64)

    for i, fpath in enumerate(files):
        print(f"  [{i + 1}/{len(files)}] loading {fpath.name} ...", flush=True)
        data = np.load(fpath, allow_pickle=True).item()

        status = np.asarray(data["status"]).reshape(-1)
        human_action = to_action_ids(np.asarray(data["human_action"]))
        thinker_action = to_action_ids(np.asarray(data["thinker_action"]))

        t = min(len(status), len(human_action), len(thinker_action))
        status = status[:t]
        human_action = human_action[:t]
        thinker_action = thinker_action[:t]

        # status == 0: real game steps (not thinking/simulation steps)
        real_mask = status == 0
        ha = human_action[real_mask]
        ta = thinker_action[real_mask]

        valid_ha = (ha >= 0) & (ha < N_ACTIONS)
        valid_ta = (ta >= 0) & (ta < N_ACTIONS)
        human_counts += np.bincount(ha[valid_ha], minlength=N_ACTIONS)
        thinker_counts += np.bincount(ta[valid_ta], minlength=N_ACTIONS)

        del data, status, human_action, thinker_action, ha, ta
        gc.collect()

    return human_counts, thinker_counts


# ---------------------------------------------------------------------------
# Representation -> RAM decoding helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FileMeta:
    subject: int
    session: int
    block: int
    game: int
    chunk: int
    path: Path


@dataclass(frozen=True)
class RawStatus2Spec:
    """Alignment information for one thinker file in raw status==2 decoding."""

    meta: FileMeta
    block_group: str
    ram_path: Path
    ram_offset_start: int
    n_real_used: int
    real_to_output: np.ndarray
    n_status2: int
    raw_dim: int


@dataclass
class RawSessionStats:
    """Raw status==2 sufficient statistics for one session."""

    n: int
    sum_x: np.ndarray
    sum_y: np.ndarray
    xtx: np.ndarray
    xty: np.ndarray


def ram_slot_labels(game: int) -> Dict[int, str]:
    labels = dict(ADDITIONAL_RAM_ADDRESSES.get(game, {}))
    labels.update(KEY_RAM_ADDRESSES.get(game, {}))
    return labels


def ram_slot_display_label(game: int, address: int, slot_name: str, short: bool = False) -> str:
    if short:
        return SHORT_RAM_SLOT_LABELS.get(game, {}).get(address, slot_name or str(address))
    return slot_name or str(address)


def parse_int_list(raw: str | None) -> set[int] | None:
    if raw is None or raw.strip() == "":
        return None
    return {int(part.strip()) for part in raw.split(",") if part.strip()}


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
            if arr.size:
                rows.append(arr[:128])
    if not rows:
        raise ValueError(f"No RAM rows parsed from {path}")
    return np.vstack(rows).astype(np.float32)


def pool_im_vp_vectors(im_vp_vectors: np.ndarray) -> np.ndarray:
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


def build_step_feature(reps: np.ndarray, feature_mode: str, concat_k: int) -> np.ndarray:
    reps = np.asarray(reps, dtype=np.float32)
    if reps.ndim != 2:
        reps = reps.reshape(reps.shape[0], -1)

    if feature_mode == "last":
        feat = np.concatenate([reps[-1], np.array([float(reps.shape[0])], dtype=np.float32)])
    elif feature_mode == "concat":
        feat_dim = reps.shape[1]
        if reps.shape[0] > concat_k:
            raise ValueError(f"concat padding length {concat_k} < required sequence length {reps.shape[0]}")
        padded = np.zeros((concat_k, feat_dim), dtype=np.float32)
        padded[-len(reps):] = reps
        feat = np.concatenate([padded.reshape(-1), np.array([float(reps.shape[0])], dtype=np.float32)])
    else:
        feat = np.concatenate([
            reps.mean(axis=0),
            reps.std(axis=0),
            reps.max(axis=0),
            reps[-1],
            np.array([float(reps.shape[0])], dtype=np.float32),
        ])
    return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def extract_real_step_features(meta: FileMeta, feature_mode: str, concat_k: int) -> Tuple[np.ndarray, pd.DataFrame]:
    data = load_npy_dict(meta.path)
    status = np.asarray(data["status"]).reshape(-1)
    pooled = pool_im_vp_vectors(data["im_vp_vectors"])
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

        features.append(build_step_feature(pooled[imag_idx], feature_mode=feature_mode, concat_k=concat_k))
        rows.append({
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
        })
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
    max_imag_steps = 1
    for real_pos, idx_global in enumerate(real_idx):
        prev_real = int(real_idx[real_pos - 1]) if real_pos > 0 else -1
        between = np.arange(prev_real + 1, int(idx_global), dtype=int)
        imag_count = int(np.sum(status[between] == 2)) if between.size else 0
        max_imag_steps = max(max_imag_steps, max(1, imag_count))
    return max_imag_steps


def gather_input_metas(input_dir: Path, game_id: int, sessions: set[int] | None) -> List[FileMeta]:
    metas: List[FileMeta] = []
    for path in sorted(input_dir.rglob("*.npy")):
        try:
            meta = parse_file_meta(path)
        except ValueError:
            continue
        if meta.game != game_id:
            continue
        if sessions is not None and meta.session not in sessions:
            continue
        metas.append(meta)
    return metas


def build_aligned_ram_dataset(
    input_dir: Path,
    ram_root: Path,
    game_id: int,
    sessions: set[int] | None,
    feature_mode: str,
    concat_k: int,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    metas = gather_input_metas(input_dir=input_dir, game_id=game_id, sessions=sessions)
    if not metas:
        raise FileNotFoundError(f"No game{game_id} .npy files found under {input_dir}")

    effective_concat_k = concat_k
    if feature_mode == "concat":
        effective_concat_k = max(infer_concat_length(meta) for meta in metas)
        print(f"[RAM decode] concat mode padding to max_len={effective_concat_k}")

    grouped: Dict[Tuple[int, int, int, int], List[FileMeta]] = {}
    for meta in metas:
        grouped.setdefault((meta.subject, meta.session, meta.block, meta.game), []).append(meta)

    x_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    meta_parts: List[pd.DataFrame] = []
    alignment_rows: List[Dict[str, object]] = []

    for key in sorted(grouped):
        files = sorted(grouped[key], key=lambda item: item.chunk)
        first_meta = files[0]
        ram_path = ram_path_from_meta(ram_root, first_meta)
        if not ram_path.exists():
            raise FileNotFoundError(f"Expected RAM file not found: {ram_path}")
        y_block = load_ram_txt(ram_path)
        ram_offset = 0
        block_group = f"sub{first_meta.subject:03d}_ses{first_meta.session:02d}_block{first_meta.block:02d}_game{first_meta.game}"
        print(
            f"[RAM decode load] sub{first_meta.subject:03d} ses{first_meta.session:02d} "
            f"block={first_meta.block} game={first_meta.game} chunks={len(files)}"
        )

        for meta in files:
            features, meta_df = extract_real_step_features(meta, feature_mode=feature_mode, concat_k=effective_concat_k)
            n_features_file = len(features)
            n_ram_total = len(y_block)
            n_used_file = min(n_features_file, max(0, n_ram_total - ram_offset))
            print(
                f"  - {meta.path.name}: real_steps={len(meta_df):,} "
                f"mean_imag={meta_df['n_imag_steps'].mean():.2f} used_ram={n_used_file:,}"
            )

            if n_used_file > 0:
                x_file = features[:n_used_file].astype(np.float32)
                y_file = y_block[ram_offset:ram_offset + n_used_file].astype(np.float32)
                meta_file = meta_df.iloc[:n_used_file].reset_index(drop=True).copy()
                meta_file["block_group"] = block_group
                meta_file["source_file_name"] = meta.path.name
                meta_file["ram_step_block"] = np.arange(ram_offset, ram_offset + n_used_file, dtype=int)
                meta_file["aligned_real_pos_global"] = np.arange(ram_offset, ram_offset + n_used_file, dtype=int)
                x_parts.append(x_file)
                y_parts.append(y_file)
                meta_parts.append(meta_file)

            alignment_rows.append({
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
            })
            ram_offset += n_used_file

    if not x_parts:
        raise RuntimeError("No aligned RAM decoding rows were produced.")

    return (
        np.vstack(x_parts).astype(np.float32),
        np.vstack(y_parts).astype(np.float32),
        pd.concat(meta_parts, ignore_index=True),
        pd.DataFrame(alignment_rows),
    )


def build_raw_status2_manifest(
    input_dir: Path,
    ram_root: Path,
    game_id: int,
    sessions: set[int] | None,
) -> Tuple[List[RawStatus2Spec], np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Align every raw status==2 activation with its following real-step RAM.

    This pass stores only RAM targets and lightweight row mappings. The raw
    activations are re-read in batches during fold-wise PCA/Ridge fitting.
    Real steps with no preceding status==2 activation are deliberately omitted;
    unlike the legacy summary features, this analysis never uses a fallback.
    """

    metas = gather_input_metas(input_dir=input_dir, game_id=game_id, sessions=sessions)
    if not metas:
        raise FileNotFoundError(f"No game{game_id} .npy files found under {input_dir}")

    grouped: Dict[Tuple[int, int, int, int], List[FileMeta]] = {}
    for meta in metas:
        grouped.setdefault((meta.subject, meta.session, meta.block, meta.game), []).append(meta)

    specs: List[RawStatus2Spec] = []
    y_parts: List[np.ndarray] = []
    meta_parts: List[pd.DataFrame] = []
    alignment_rows: List[Dict[str, object]] = []
    output_offset = 0
    expected_raw_dim: int | None = None

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
            f"[raw status2 manifest] sub{first_meta.subject:03d} "
            f"ses{first_meta.session:02d} block={first_meta.block} "
            f"game={first_meta.game} chunks={len(files)}"
        )

        for meta in files:
            data = load_npy_dict(meta.path)
            status = np.asarray(data["status"]).reshape(-1)
            reps = np.asarray(data["im_vp_vectors"])
            t = min(len(status), len(reps))
            status = status[:t]
            reps = reps[:t]
            real_idx = np.flatnonzero(status == 0)
            n_ram_remaining = max(0, len(y_block) - ram_offset)
            n_real_used = min(len(real_idx), n_ram_remaining)
            raw_dim = int(np.prod(reps.shape[1:], dtype=np.int64))

            if expected_raw_dim is None:
                expected_raw_dim = raw_dim
            elif raw_dim != expected_raw_dim:
                raise ValueError(
                    f"Raw activation size changed from {expected_raw_dim} to {raw_dim} "
                    f"in {meta.path}"
                )

            imag_idx = np.flatnonzero(status == 2)
            next_real_pos = np.searchsorted(real_idx, imag_idx, side="left")
            valid = next_real_pos < n_real_used
            next_real_pos = next_real_pos[valid]
            counts = np.bincount(next_real_pos, minlength=n_real_used).astype(np.int64)
            kept_real_pos = np.flatnonzero(counts > 0)
            n_status2 = int(counts.sum())

            real_to_output = np.full(n_real_used, -1, dtype=np.int64)
            real_to_output[kept_real_pos] = np.arange(
                output_offset, output_offset + len(kept_real_pos), dtype=np.int64
            )

            if kept_real_pos.size:
                ram_rows = ram_offset + kept_real_pos
                y_parts.append(y_block[ram_rows].astype(np.float32))
                meta_parts.append(pd.DataFrame({
                    "source_file": str(meta.path),
                    "source_file_name": meta.path.name,
                    "subject": meta.subject,
                    "session": meta.session,
                    "block": meta.block,
                    "game": meta.game,
                    "chunk": meta.chunk,
                    "real_pos_chunk": kept_real_pos.astype(int),
                    "status_idx": real_idx[kept_real_pos].astype(int),
                    "n_imag_steps": counts[kept_real_pos].astype(int),
                    "used_fallback": np.zeros(len(kept_real_pos), dtype=int),
                    "block_group": block_group,
                    "ram_step_block": ram_rows.astype(int),
                    "aligned_real_pos_global": ram_rows.astype(int),
                }))

            specs.append(RawStatus2Spec(
                meta=meta,
                block_group=block_group,
                ram_path=ram_path,
                ram_offset_start=ram_offset,
                n_real_used=n_real_used,
                real_to_output=real_to_output,
                n_status2=n_status2,
                raw_dim=raw_dim,
            ))

            alignment_rows.append({
                "subject": meta.subject,
                "session": meta.session,
                "block": meta.block,
                "game": meta.game,
                "block_group": block_group,
                "source_file": str(meta.path),
                "source_file_name": meta.path.name,
                "n_chunk_files_in_block": len(files),
                "n_real_feature_steps": len(real_idx),
                "n_real_steps_with_status2": len(kept_real_pos),
                "n_raw_status2_samples": n_status2,
                "n_ram_steps_block": len(y_block),
                "ram_offset_start": ram_offset,
                "ram_offset_stop": ram_offset + n_real_used,
                "n_used_steps": n_real_used,
                "trimmed_feature_steps": len(real_idx) - n_real_used,
                "unused_ram_steps_after_file": len(y_block) - (ram_offset + n_real_used),
                "mean_status2_per_retained_step": (
                    float(counts[kept_real_pos].mean()) if kept_real_pos.size else float("nan")
                ),
                "fallback_rate": 0.0,
                "raw_activation_dim": raw_dim,
                "ram_path": str(ram_path),
            })

            print(
                f"  - {meta.path.name}: real_steps={len(real_idx):,} "
                f"retained_real={len(kept_real_pos):,} raw_status2={n_status2:,} "
                f"raw_dim={raw_dim:,}"
            )
            output_offset += len(kept_real_pos)
            ram_offset += n_real_used
            del data, status, reps, real_idx, imag_idx, next_real_pos, counts
            gc.collect()

    if not y_parts or not meta_parts:
        raise RuntimeError("No status==2 activations could be aligned to RAM targets.")

    y = np.vstack(y_parts).astype(np.float32)
    meta_df = pd.concat(meta_parts, ignore_index=True)
    if len(y) != len(meta_df) or len(y) != output_offset:
        raise RuntimeError("Raw status==2 manifest indexing is inconsistent.")
    return specs, y, meta_df, pd.DataFrame(alignment_rows)


def iter_raw_status2_batches(
    specs: Sequence[RawStatus2Spec],
    sessions: set[int],
    batch_size: int,
):
    """Yield flattened raw status==2 activations and aligned real-step rows."""

    if batch_size <= 0:
        raise ValueError("Raw activation batch size must be positive.")

    for spec in specs:
        if spec.meta.session not in sessions or spec.n_status2 == 0:
            continue

        data = load_npy_dict(spec.meta.path)
        status = np.asarray(data["status"]).reshape(-1)
        reps = np.asarray(data["im_vp_vectors"])
        t = min(len(status), len(reps))
        status = status[:t]
        reps = reps[:t]
        real_idx = np.flatnonzero(status == 0)
        imag_idx = np.flatnonzero(status == 2)
        next_real_pos = np.searchsorted(real_idx, imag_idx, side="left")
        valid = next_real_pos < spec.n_real_used
        imag_idx = imag_idx[valid]
        next_real_pos = next_real_pos[valid]
        output_idx = spec.real_to_output[next_real_pos]
        keep = output_idx >= 0
        imag_idx = imag_idx[keep]
        output_idx = output_idx[keep]

        for start in range(0, len(imag_idx), batch_size):
            stop = min(start + batch_size, len(imag_idx))
            raw_rows = np.asarray(reps[imag_idx[start:stop]], dtype=np.float32)
            x_batch = raw_rows.reshape(len(raw_rows), -1)
            if x_batch.shape[1] != spec.raw_dim:
                raise RuntimeError(
                    f"Expected raw dim {spec.raw_dim}, got {x_batch.shape[1]} "
                    f"in {spec.meta.path}"
                )
            x_batch = np.nan_to_num(
                x_batch, nan=0.0, posinf=0.0, neginf=0.0, copy=False
            )
            yield x_batch, output_idx[start:stop]

        del data, status, reps, real_idx, imag_idx, next_real_pos, output_idx
        gc.collect()


def accumulate_raw_status2_session_stats(
    specs: Sequence[RawStatus2Spec],
    y: np.ndarray,
    n_output_steps: int,
    sessions: Sequence[int],
    batch_size: int,
) -> Tuple[Dict[int, RawSessionStats], np.ndarray]:
    """Accumulate exact session-wise raw moments with GPU matrix products.

    The session-specific moments can be added and subtracted to construct each
    held-out-session training fold without re-reading the multi-gigabyte trace
    files.  Per-real-step raw means are retained because averaging predictions
    from a linear PCA/Ridge decoder is exactly equivalent to decoding the mean
    activation preceding that real step.
    """

    if batch_size <= 0:
        raise ValueError("Raw activation batch size must be positive.")
    if not specs:
        raise ValueError("No raw status==2 specifications were provided.")

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "raw_status2_pca requires PyTorch for the exact covariance PCA path."
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "raw_status2_pca exact covariance accumulation requires a CUDA GPU."
        )

    device = torch.device("cuda:0")
    raw_dim = int(specs[0].raw_dim)
    n_targets = int(y.shape[1])
    step_mean = np.empty((n_output_steps, raw_dim), dtype=np.float32)
    step_filled = np.zeros(n_output_steps, dtype=bool)
    session_stats: Dict[int, RawSessionStats] = {}

    for session in sessions:
        session_specs = [spec for spec in specs if spec.meta.session == int(session)]
        if not session_specs:
            raise ValueError(f"No raw status==2 files were found for session {session}.")

        print(
            f"  [raw moments] ses{int(session):02d}: "
            f"{len(session_specs)} file(s), CUDA covariance accumulation"
        )
        xtx_gpu = torch.zeros((raw_dim, raw_dim), dtype=torch.float32, device=device)
        xty_gpu = torch.zeros((raw_dim, n_targets), dtype=torch.float32, device=device)
        sum_x = np.zeros(raw_dim, dtype=np.float64)
        sum_y = np.zeros(n_targets, dtype=np.float64)
        n_session = 0

        for spec_idx, spec in enumerate(session_specs, start=1):
            data = load_npy_dict(spec.meta.path)
            status = np.asarray(data["status"]).reshape(-1)
            reps = np.asarray(data["im_vp_vectors"])
            t = min(len(status), len(reps))
            status = status[:t]
            reps = reps[:t]
            real_idx = np.flatnonzero(status == 0)
            imag_idx = np.flatnonzero(status == 2)
            next_real_pos = np.searchsorted(real_idx, imag_idx, side="left")
            valid = next_real_pos < spec.n_real_used
            imag_idx = imag_idx[valid]
            next_real_pos = next_real_pos[valid]
            output_idx = spec.real_to_output[next_real_pos]
            keep = output_idx >= 0
            imag_idx = imag_idx[keep]
            output_idx = output_idx[keep]

            file_output_ids = spec.real_to_output[spec.real_to_output >= 0]
            if len(file_output_ids) == 0:
                del data, status, reps, real_idx, imag_idx, next_real_pos, output_idx
                gc.collect()
                continue
            first_output = int(file_output_ids[0])
            expected_ids = np.arange(
                first_output, first_output + len(file_output_ids), dtype=np.int64
            )
            if not np.array_equal(file_output_ids, expected_ids):
                raise RuntimeError(
                    f"Non-contiguous output mapping in {spec.meta.path}; "
                    "cannot aggregate per-step raw means safely."
                )

            file_sum_gpu = torch.zeros(
                (len(file_output_ids), raw_dim), dtype=torch.float32, device=device
            )
            file_count_gpu = torch.zeros(
                len(file_output_ids), dtype=torch.float32, device=device
            )

            for start in range(0, len(imag_idx), batch_size):
                stop = min(start + batch_size, len(imag_idx))
                raw_rows = np.asarray(reps[imag_idx[start:stop]], dtype=np.float32)
                x_np = raw_rows.reshape(len(raw_rows), -1)
                if x_np.shape[1] != raw_dim:
                    raise RuntimeError(
                        f"Expected raw dim {raw_dim}, got {x_np.shape[1]} "
                        f"in {spec.meta.path}"
                    )
                x_np = np.nan_to_num(
                    x_np, nan=0.0, posinf=0.0, neginf=0.0, copy=False
                )
                out_np = output_idx[start:stop]
                y_np = np.asarray(y[out_np], dtype=np.float32)

                x_gpu = torch.as_tensor(x_np, dtype=torch.float32, device=device)
                y_gpu = torch.as_tensor(y_np, dtype=torch.float32, device=device)
                local_gpu = torch.as_tensor(
                    out_np - first_output, dtype=torch.long, device=device
                )

                xtx_gpu.addmm_(x_gpu.T, x_gpu)
                xty_gpu.addmm_(x_gpu.T, y_gpu)
                file_sum_gpu.index_add_(0, local_gpu, x_gpu)
                file_count_gpu.index_add_(
                    0,
                    local_gpu,
                    torch.ones(len(x_gpu), dtype=torch.float32, device=device),
                )
                sum_x += x_np.sum(axis=0, dtype=np.float64)
                sum_y += y_np.sum(axis=0, dtype=np.float64)
                n_session += len(x_np)

                del x_gpu, y_gpu, local_gpu, raw_rows, x_np, y_np, out_np

            counts = file_count_gpu.cpu().numpy()
            if np.any(counts <= 0):
                raise RuntimeError(
                    f"Missing status==2 rows for {int(np.sum(counts <= 0))} "
                    f"retained real steps in {spec.meta.path}."
                )
            file_means = file_sum_gpu.cpu().numpy()
            file_means /= counts[:, None]
            step_mean[file_output_ids] = file_means
            step_filled[file_output_ids] = True
            print(
                f"    [{spec_idx}/{len(session_specs)}] {spec.meta.path.name}: "
                f"raw={len(imag_idx):,}, real={len(file_output_ids):,}"
            )

            del (
                data,
                status,
                reps,
                real_idx,
                imag_idx,
                next_real_pos,
                output_idx,
                file_sum_gpu,
                file_count_gpu,
                counts,
                file_means,
            )
            gc.collect()
            torch.cuda.empty_cache()

        session_stats[int(session)] = RawSessionStats(
            n=n_session,
            sum_x=sum_x,
            sum_y=sum_y,
            xtx=xtx_gpu.cpu().numpy().astype(np.float64),
            xty=xty_gpu.cpu().numpy().astype(np.float64),
        )
        print(
            f"    ses{int(session):02d} complete: raw samples={n_session:,}"
        )
        del xtx_gpu, xty_gpu
        torch.cuda.empty_cache()

    if not np.all(step_filled):
        raise RuntimeError(
            f"Per-step raw means are missing for "
            f"{int(np.sum(~step_filled))} retained real steps."
        )
    return session_stats, step_mean


def top_pca_basis_from_covariance(
    centered_xtx: np.ndarray,
    n_components: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the largest covariance eigenvectors using CUDA."""

    import torch

    if n_components <= 0 or n_components > centered_xtx.shape[0]:
        raise ValueError("Invalid number of PCA components.")
    device = torch.device("cuda:0")
    cov_gpu = torch.as_tensor(
        centered_xtx, dtype=torch.float32, device=device
    )
    cov_gpu = 0.5 * (cov_gpu + cov_gpu.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_gpu)
    basis = eigenvectors[:, -n_components:].flip(1).cpu().numpy().astype(np.float64)
    top_values = eigenvalues[-n_components:].flip(0).cpu().numpy().astype(np.float64)
    del cov_gpu, eigenvalues, eigenvectors
    torch.cuda.empty_cache()
    return basis, top_values


def fit_session_loro_raw_status2_cov_pca_ridge(
    specs: Sequence[RawStatus2Spec],
    y: np.ndarray,
    meta_df: pd.DataFrame,
    sessions: Sequence[int],
    n_components: int,
    batch_size: int,
    alpha: float,
) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]], str, pd.DataFrame]:
    """Exact fold-wise covariance PCA and Ridge without repeated trace reads."""

    splits, _ = build_session_loro_splits(meta_df, required_sessions=sessions)
    stats, step_mean = accumulate_raw_status2_session_stats(
        specs=specs,
        y=y,
        n_output_steps=len(meta_df),
        sessions=sessions,
        batch_size=batch_size,
    )
    y_pred = np.full_like(y, np.nan, dtype=np.float32)
    fold_rows: List[Dict[str, object]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        heldout_session = int(meta_df.iloc[test_idx[0]]["session"])
        train_sessions = [int(v) for v in sessions if int(v) != heldout_session]
        print(
            f"  [raw PCA fold {fold_idx}/{len(splits)}] "
            f"heldout=ses{heldout_session:02d}: exact PCA50 from sessions "
            f"{train_sessions}"
        )

        n_train = int(sum(stats[s].n for s in train_sessions))
        sum_x = sum((stats[s].sum_x for s in train_sessions), np.zeros_like(stats[train_sessions[0]].sum_x))
        sum_y = sum((stats[s].sum_y for s in train_sessions), np.zeros_like(stats[train_sessions[0]].sum_y))
        xtx = sum((stats[s].xtx for s in train_sessions), np.zeros_like(stats[train_sessions[0]].xtx))
        xty = sum((stats[s].xty for s in train_sessions), np.zeros_like(stats[train_sessions[0]].xty))
        mean_x = sum_x / n_train
        mean_y = sum_y / n_train
        centered_xtx = xtx - n_train * np.outer(mean_x, mean_x)
        centered_xty = xty - n_train * np.outer(mean_x, mean_y)

        basis, eigenvalues = top_pca_basis_from_covariance(
            centered_xtx=centered_xtx,
            n_components=n_components,
        )
        ztz = basis.T @ centered_xtx @ basis
        zty = basis.T @ centered_xty
        penalty = float(alpha) * np.eye(n_components, dtype=np.float64)
        try:
            coef = np.linalg.solve(ztz + penalty, zty)
        except np.linalg.LinAlgError:
            coef = np.linalg.pinv(ztz + penalty) @ zty

        z_test = (np.asarray(step_mean[test_idx], dtype=np.float64) - mean_x) @ basis
        y_pred[test_idx] = (z_test @ coef + mean_y).astype(np.float32)
        total_variance = max(float(np.trace(centered_xtx)), EPS)
        explained = float(np.maximum(eigenvalues, 0.0).sum() / total_variance)
        n_test_raw = int(stats[heldout_session].n)
        fold_rows.append({
            "heldout_session": heldout_session,
            "train_sessions": ",".join(str(v) for v in train_sessions),
            "n_pca_fit_samples": n_train,
            "n_ridge_train_samples": n_train,
            "n_test_raw_samples": n_test_raw,
            "n_test_real_steps": len(test_idx),
            "n_components": n_components,
            "pca_explained_variance_ratio_sum": explained,
            "ridge_alpha": alpha,
            "pca_solver": "exact session covariance eigendecomposition (CUDA)",
        })
        print(
            f"    heldout=ses{heldout_session:02d} train_raw={n_train:,} "
            f"test_raw={n_test_raw:,} test_real={len(test_idx):,} "
            f"PCA{n_components}_variance={explained:.3f}"
        )

        del (
            sum_x,
            sum_y,
            xtx,
            xty,
            mean_x,
            mean_y,
            centered_xtx,
            centered_xty,
            basis,
            eigenvalues,
            ztz,
            zty,
            coef,
            z_test,
        )
        gc.collect()

    if np.isnan(y_pred).any():
        raise RuntimeError("Out-of-fold raw PCA predictions contain NaNs.")
    splitter_name = (
        f"LeaveOneSessionOut(ses01-ses04), exact fold-wise covariance PCA"
        f"({n_components})"
    )
    return y_pred, splits, splitter_name, pd.DataFrame(fold_rows)


def fit_incremental_pca_on_status2(
    specs: Sequence[RawStatus2Spec],
    train_sessions: set[int],
    n_components: int,
    batch_size: int,
) -> Tuple[IncrementalPCA, int]:
    """Fit PCA on training-session raw activations without materializing them."""

    if n_components <= 0:
        raise ValueError("--raw-pca-components must be positive.")
    if batch_size < n_components:
        raise ValueError("--raw-pca-batch-size must be at least --raw-pca-components.")

    pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    pending: np.ndarray | None = None
    n_fit = 0

    for x_batch, _ in iter_raw_status2_batches(specs, train_sessions, batch_size):
        pending = (
            x_batch
            if pending is None or not len(pending)
            else np.concatenate([pending, x_batch], axis=0)
        )
        # Keep at least n_components rows pending. This lets the final short
        # fragment be combined with the preceding rows instead of being dropped.
        while len(pending) >= batch_size + n_components:
            pca.partial_fit(pending[:batch_size])
            n_fit += batch_size
            pending = pending[batch_size:].copy()

    if pending is not None and len(pending) >= n_components:
        pca.partial_fit(pending)
        n_fit += len(pending)

    if n_fit < n_components or not hasattr(pca, "components_"):
        raise RuntimeError(
            f"Only {n_fit} raw status==2 samples were available for "
            f"{n_components}-component PCA."
        )
    return pca, n_fit


def fit_streamed_pc_ridge(
    specs: Sequence[RawStatus2Spec],
    train_sessions: set[int],
    pca: IncrementalPCA,
    y: np.ndarray,
    alpha: float,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Compute the exact intercept-bearing Ridge solution from PC statistics."""

    n_components = int(pca.n_components_)
    n_targets = int(y.shape[1])
    sum_x = np.zeros(n_components, dtype=np.float64)
    sum_y = np.zeros(n_targets, dtype=np.float64)
    xtx = np.zeros((n_components, n_components), dtype=np.float64)
    xty = np.zeros((n_components, n_targets), dtype=np.float64)
    n_samples = 0

    for x_batch, output_idx in iter_raw_status2_batches(specs, train_sessions, batch_size):
        z = np.asarray(pca.transform(x_batch), dtype=np.float64)
        y_batch = np.asarray(y[output_idx], dtype=np.float64)
        sum_x += z.sum(axis=0)
        sum_y += y_batch.sum(axis=0)
        xtx += z.T @ z
        xty += z.T @ y_batch
        n_samples += len(z)

    if n_samples == 0:
        raise RuntimeError("No training status==2 samples were available for Ridge.")

    mean_x = sum_x / n_samples
    mean_y = sum_y / n_samples
    centered_xtx = xtx - n_samples * np.outer(mean_x, mean_x)
    centered_xty = xty - n_samples * np.outer(mean_x, mean_y)
    penalty = float(alpha) * np.eye(n_components, dtype=np.float64)
    try:
        coef = np.linalg.solve(centered_xtx + penalty, centered_xty)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(centered_xtx + penalty) @ centered_xty
    intercept = mean_y - mean_x @ coef
    return coef, intercept, n_samples


def fit_session_loro_raw_status2_pca_ridge(
    specs: Sequence[RawStatus2Spec],
    y: np.ndarray,
    meta_df: pd.DataFrame,
    sessions: Sequence[int],
    n_components: int,
    batch_size: int,
    alpha: float,
) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]], str, pd.DataFrame]:
    """Fold-wise raw activation -> PCA -> Ridge with real-step predictions."""

    splits, _ = build_session_loro_splits(meta_df, required_sessions=sessions)
    y_pred = np.full_like(y, np.nan, dtype=np.float32)
    fold_rows: List[Dict[str, object]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        heldout_session = int(meta_df.iloc[test_idx[0]]["session"])
        train_sessions = set(int(v) for v in sessions if int(v) != heldout_session)
        test_sessions = {heldout_session}
        print(
            f"  [raw PCA fold {fold_idx}/{len(splits)}] "
            f"heldout=ses{heldout_session:02d}: fitting PCA on sessions "
            f"{sorted(train_sessions)}"
        )
        pca, n_pca_fit = fit_incremental_pca_on_status2(
            specs=specs,
            train_sessions=train_sessions,
            n_components=n_components,
            batch_size=batch_size,
        )
        explained = float(np.nansum(pca.explained_variance_ratio_))

        print(
            f"    fitting Ridge sufficient statistics from all training "
            f"status==2 activations"
        )
        coef, intercept, n_ridge_train = fit_streamed_pc_ridge(
            specs=specs,
            train_sessions=train_sessions,
            pca=pca,
            y=y,
            alpha=alpha,
            batch_size=batch_size,
        )

        global_to_local = np.full(len(meta_df), -1, dtype=np.int64)
        global_to_local[test_idx] = np.arange(len(test_idx), dtype=np.int64)
        pred_sum = np.zeros((len(test_idx), y.shape[1]), dtype=np.float64)
        pred_count = np.zeros(len(test_idx), dtype=np.int64)
        n_test_raw = 0

        for x_batch, output_idx in iter_raw_status2_batches(specs, test_sessions, batch_size):
            z = np.asarray(pca.transform(x_batch), dtype=np.float64)
            batch_pred = z @ coef + intercept
            local_idx = global_to_local[output_idx]
            if np.any(local_idx < 0):
                raise RuntimeError("Held-out raw activation mapped outside its session fold.")
            np.add.at(pred_sum, local_idx, batch_pred)
            np.add.at(pred_count, local_idx, 1)
            n_test_raw += len(z)

        if np.any(pred_count == 0):
            raise RuntimeError(
                f"Missing raw status==2 predictions for "
                f"{int(np.sum(pred_count == 0))} held-out real steps."
            )
        # Each raw activation is decoded independently. We average predictions
        # only for evaluation when several status==2 samples map to one real step.
        y_pred[test_idx] = (pred_sum / pred_count[:, None]).astype(np.float32)
        fold_rows.append({
            "heldout_session": heldout_session,
            "train_sessions": ",".join(str(v) for v in sorted(train_sessions)),
            "n_pca_fit_samples": n_pca_fit,
            "n_ridge_train_samples": n_ridge_train,
            "n_test_raw_samples": n_test_raw,
            "n_test_real_steps": len(test_idx),
            "n_components": n_components,
            "pca_explained_variance_ratio_sum": explained,
            "ridge_alpha": alpha,
        })
        print(
            f"    heldout=ses{heldout_session:02d} train_raw={n_ridge_train:,} "
            f"test_raw={n_test_raw:,} test_real={len(test_idx):,} "
            f"PCA{n_components}_variance={explained:.3f}"
        )
        del pca, coef, intercept, global_to_local, pred_sum, pred_count
        gc.collect()

    if np.isnan(y_pred).any():
        raise RuntimeError("Out-of-fold raw PCA predictions contain NaNs.")
    splitter_name = (
        f"LeaveOneSessionOut(ses01-ses04), fold-wise IncrementalPCA({n_components})"
    )
    return y_pred, splits, splitter_name, pd.DataFrame(fold_rows)


def build_session_loro_splits(meta_df: pd.DataFrame, required_sessions: Sequence[int]) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], str]:
    sessions = np.asarray(meta_df["session"].to_numpy(), dtype=int)
    available = sorted(np.unique(sessions).tolist())
    missing = [s for s in required_sessions if s not in available]
    if missing:
        raise ValueError(f"Cannot run session LORO: missing session(s) {missing}; available={available}")
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for session in required_sessions:
        test_idx = np.flatnonzero(sessions == session)
        train_idx = np.flatnonzero(sessions != session)
        if test_idx.size == 0 or train_idx.size == 0:
            raise ValueError(f"Invalid LORO split for session {session}")
        splits.append((train_idx, test_idx))
    return splits, "LeaveOneSessionOut(ses01-ses04)"


def fit_session_loro_ridge(
    x: np.ndarray,
    y: np.ndarray,
    meta_df: pd.DataFrame,
    sessions: Sequence[int],
    alpha: float,
) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]], str]:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    splits, splitter_name = build_session_loro_splits(meta_df, required_sessions=sessions)
    y_pred = np.full_like(y, np.nan, dtype=np.float32)

    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        heldout_session = int(meta_df.iloc[test_idx[0]]["session"])
        scaler = StandardScaler(with_mean=True, with_std=True)
        x_train = scaler.fit_transform(x[train_idx])
        x_test = scaler.transform(x[test_idx])
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(x_train, y[train_idx])
        y_pred[test_idx] = model.predict(x_test).astype(np.float32)
        print(
            f"  [session fold {fold_idx}/{len(splits)}] heldout=ses{heldout_session:02d} "
            f"train={len(train_idx):,} test={len(test_idx):,}"
        )

    if np.isnan(y_pred).any():
        raise RuntimeError("Out-of-fold predictions contain NaNs; split coverage failed.")
    return y_pred, splits, splitter_name


def pearson_columns(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    yt = yt - np.nanmean(yt, axis=0, keepdims=True)
    yp = yp - np.nanmean(yp, axis=0, keepdims=True)
    den = np.sqrt(np.nansum(yt * yt, axis=0) * np.nansum(yp * yp, axis=0))
    out = np.full(yt.shape[1], np.nan, dtype=np.float32)
    valid = den > EPS
    out[valid] = (np.nansum(yt[:, valid] * yp[:, valid], axis=0) / den[valid]).astype(np.float32)
    return out


def safe_pearsonr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    vals = pearson_columns(np.asarray(y_true)[:, None], np.asarray(y_pred)[:, None])
    return float(vals[0]) if vals.size else float("nan")


def majority_accuracy(y_true: np.ndarray) -> float:
    values, counts = np.unique(y_true.astype(np.int64), return_counts=True)
    if values.size == 0:
        return float("nan")
    return float(counts.max() / counts.sum())


def fdr_bh(pvals: np.ndarray) -> np.ndarray:
    pv = np.asarray(pvals, dtype=np.float64)
    q = np.full_like(pv, np.nan)
    finite = np.isfinite(pv)
    n = int(finite.sum())
    if n == 0:
        return q.astype(np.float32)
    idx = np.where(finite)[0]
    order = np.argsort(pv[idx])
    ranked = pv[idx][order]
    q_ordered = ranked * n / np.arange(1, n + 1, dtype=np.float64)
    q_ordered = np.minimum.accumulate(q_ordered[::-1])[::-1]
    q_ordered = np.minimum(q_ordered, 1.0)
    q[idx[order]] = q_ordered
    return q.astype(np.float32)


def block_permutation_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    blocks = [np.arange(lo, min(lo + block_size, n), dtype=int) for lo in range(0, n, max(1, block_size))]
    if len(blocks) < 2:
        return np.arange(n, dtype=int)
    order = rng.permutation(len(blocks))
    return np.concatenate([blocks[i] for i in order])


def permutation_pvals_for_loro_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    n_perm: int,
    block_size: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_perm <= 0:
        raise ValueError("n_perm must be positive because FDR is mandatory for RAM decoding.")

    obs = pearson_columns(y_true, y_pred)
    ge_counts = np.zeros(y_true.shape[1], dtype=np.int32)
    null_mean_r = np.full(n_perm, np.nan, dtype=np.float32)
    rng = np.random.default_rng(seed)
    progress_step = max(1, n_perm // 5)

    for perm_i in range(n_perm):
        y_perm = np.empty_like(y_true)
        for _, test_idx in splits:
            perm_order = block_permutation_indices(len(test_idx), block_size, rng)
            y_perm[test_idx] = y_true[test_idx[perm_order]]
        perm_r = pearson_columns(y_perm, y_pred)
        valid = np.isfinite(obs) & np.isfinite(perm_r)
        ge_counts[valid] += perm_r[valid] >= obs[valid]
        null_mean_r[perm_i] = float(np.nanmean(perm_r))
        if (perm_i + 1) % progress_step == 0 or perm_i + 1 == n_perm:
            print(f"  [RAM decode perm] {perm_i + 1}/{n_perm}", flush=True)

    p_one = np.full_like(obs, np.nan, dtype=np.float32)
    valid = np.isfinite(obs)
    p_one[valid] = (1.0 + ge_counts[valid]) / (1.0 + n_perm)
    q_fdr = fdr_bh(p_one)
    return p_one, q_fdr, null_mean_r


def compute_slot_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    game: int,
    p_one_sided: np.ndarray,
    q_fdr: np.ndarray,
) -> pd.DataFrame:
    key_map = KEY_RAM_ADDRESSES.get(game, {})
    label_map = ram_slot_labels(game)
    trace_excluded = TRACE_EXCLUDED_RAM_ADDRESSES.get(game, set())
    rows: List[Dict[str, object]] = []
    pearson_r = pearson_columns(y_true, y_pred)

    for address in range(y_true.shape[1]):
        yt = y_true[:, address].astype(np.float64)
        yp = y_pred[:, address].astype(np.float64)
        rounded = np.clip(np.rint(yp), 0, 255)
        value_range = float(np.max(yt) - np.min(yt))
        acc = float(np.mean(rounded == yt))
        maj_acc = majority_accuracy(yt)
        rows.append({
            "address": address,
            "slot_name": label_map.get(address, ""),
            "is_key_slot": int(address in key_map),
            "is_annotated_slot": int(address in label_map),
            "trace_excluded": int(address in trace_excluded),
            "n_unique": int(np.unique(yt).size),
            "value_min": float(np.min(yt)),
            "value_max": float(np.max(yt)),
            "value_range": value_range,
            "pearson_r": float(pearson_r[address]),
            "p_one_sided": float(p_one_sided[address]),
            "q_fdr": float(q_fdr[address]),
            "is_fdr_sig": int(np.isfinite(q_fdr[address]) and q_fdr[address] < 0.05),
            "r2": float(r2_score(yt, yp)) if value_range > 0 else float("nan"),
            "mae": float(mean_absolute_error(yt, yp)),
            "normalized_mae": float(mean_absolute_error(yt, yp) / value_range) if value_range > 0 else float("nan"),
            "rounded_acc": acc,
            "majority_acc": maj_acc,
            "acc_gain": float(acc - maj_acc),
        })

    df = pd.DataFrame(rows).sort_values("address").reset_index(drop=True)
    df["pearson_rank"] = df["pearson_r"].fillna(-np.inf).rank(method="min", ascending=False).astype(int)
    df["q_rank"] = df["q_fdr"].fillna(np.inf).rank(method="min", ascending=True).astype(int)
    df["r2_rank"] = df["r2"].fillna(-np.inf).rank(method="min", ascending=False).astype(int)
    df["acc_gain_rank"] = df["acc_gain"].fillna(-np.inf).rank(method="min", ascending=False).astype(int)
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_action_distribution(
    human_counts: np.ndarray,
    thinker_counts: np.ndarray,
    game_title: str,
    subject: str,
    out_path: Path,
) -> None:
    total_h = human_counts.sum()
    total_t = thinker_counts.sum()
    if total_h == 0 or total_t == 0:
        print("Warning: no valid actions found; skipping plot.", file=sys.stderr)
        return

    human_prop = human_counts / total_h
    thinker_prop = thinker_counts / total_t

    x = np.arange(N_ACTIONS)
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(x - width / 2, human_prop, width,
           label="human_action", color=HUMAN_COLOR, alpha=0.85, edgecolor="white")
    ax.bar(x + width / 2, thinker_prop, width,
           label="thinker_action", color=THINKER_COLOR, alpha=0.85, edgecolor="white")

    ax.set_xlabel("Action")
    ax.set_ylabel("Proportion")
    ax.set_title(f"Action Distribution — {game_title} — {subject}")
    ax.set_xticks(x)
    ax.set_xticklabels(ACTION_LABELS, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.legend()

    # annotate totals
    ax.text(0.98, 0.97,
            f"human n={total_h:,}\nthinker n={total_t:,}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color="#555555")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_ram_slot_scores_fdr(metrics_df: pd.DataFrame, game: int, out_path: Path) -> None:
    df = metrics_df.sort_values("address").copy()
    fig, ax = plt.subplots(figsize=(13, 5.4))

    low_effect = df["pearson_r"] <= 0.8
    status_groups = (
        ("n.s.", (df["p_one_sided"] >= 0.05), "#c7c7c7"),
        (
            "p < 0.05 only",
            (df["p_one_sided"] < 0.05) & (df["q_fdr"] >= 0.05),
            "#f59e0b",
        ),
        ("FDR q < 0.05", df["q_fdr"] < 0.05, "#dc2626"),
    )
    for label, status_mask, color in status_groups:
        high_df = df[status_mask & ~low_effect]
        low_df = df[status_mask & low_effect]
        ax.scatter(
            high_df["address"],
            high_df["pearson_r"],
            s=38,
            color=color,
            alpha=0.95,
            label=label,
            zorder=3,
        )
        ax.scatter(
            low_df["address"],
            low_df["pearson_r"],
            s=7,
            color=color,
            alpha=0.55,
            zorder=2,
        )

    label_map = RAM_SCORE_LABELS.get(game, {})
    labeled = df[df["address"].isin(label_map)].copy()
    ax.scatter(
        labeled["address"],
        labeled["pearson_r"],
        s=96,
        facecolors="none",
        edgecolors="#111827",
        linewidths=1.2,
        label="labeled slot",
        zorder=4,
    )

    for address, name in label_map.items():
        row = df[df["address"] == address]
        if row.empty:
            continue
        y = float(row["pearson_r"].iloc[0])
        if np.isfinite(y):
            offset_x, offset_y = RAM_SCORE_LABEL_OFFSETS.get(game, {}).get(
                address, (0.0, 12.0)
            )
            ax.annotate(
                name,
                xy=(address, y),
                xytext=(offset_x, offset_y),
                textcoords="offset points",
                ha="right" if offset_x < 0 else "left" if offset_x > 0 else "center",
                va="top" if offset_y < 0 else "bottom",
                fontsize=8.2,
                color="#111827",
                annotation_clip=True,
                clip_on=True,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.8},
                arrowprops={"arrowstyle": "-", "color": "#9ca3af", "lw": 0.5},
                zorder=5,
            )

    ax.axhline(0, color="black", lw=0.7, ls="--")
    ax.set_xlim(-2, 129)
    finite_r = df["pearson_r"].to_numpy(dtype=float)
    finite_r = finite_r[np.isfinite(finite_r)]
    if finite_r.size:
        ax.set_ylim(
            min(-0.04, float(finite_r.min()) - 0.04),
            max(0.90, float(finite_r.max()) + 0.10),
        )
    ax.set_xlabel("RAM address")
    ax.set_ylabel("Session-LORO Pearson r")
    ax.set_title(f"{GAME_TITLES.get(game, f'Game {game}')}: RAM decoding by address with permutation FDR")
    ax.xaxis.grid(True, color="#e5e7eb", lw=0.5)
    ax.yaxis.grid(True, color="#e5e7eb", lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_ram_key_slot_fdr_bars(metrics_df: pd.DataFrame, game: int, out_path: Path) -> None:
    # This summary is intended for RAM addresses with an interpretable label.
    # Unannotated FDR-significant addresses remain available in the CSV and
    # whole-RAM figures, but are omitted from this key-slot panel.
    df = metrics_df[metrics_df["is_annotated_slot"] == 1].copy()
    df = df[np.isfinite(df["pearson_r"])].copy()
    if df.empty:
        print(f"Warning: no annotated RAM slots available; skipping {out_path.name}.", file=sys.stderr)
        return
    df = df.sort_values(["is_fdr_sig", "pearson_r"], ascending=[True, True])

    labels = []
    for row in df.itertuples():
        slot_name = str(row.slot_name) if isinstance(row.slot_name, str) else ""
        labels.append(f"{int(row.address):03d}  {slot_name}" if slot_name else f"{int(row.address):03d}")

    colors = ["#dc2626" if q < 0.05 else "#f59e0b" if p < 0.05 else "#9ca3af"
              for p, q in zip(df["p_one_sided"], df["q_fdr"])]

    fig, ax = plt.subplots(figsize=(9.8, max(5.0, 0.34 * len(df) + 1.4)))
    bars = ax.barh(labels, df["pearson_r"], color=colors, edgecolor="white", linewidth=0.5)
    for bar, row in zip(bars, df.itertuples()):
        if not bool(row.is_fdr_sig):
            continue
        value = float(row.pearson_r)
        offset = 0.008 if value >= 0 else -0.008
        ax.text(
            value + offset,
            bar.get_y() + bar.get_height() / 2,
            "*",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=11,
            fontweight="bold",
        )
    ax.axvline(0, color="black", lw=0.7, ls="--")
    ax.set_xlabel("Session-LORO Pearson r")
    ax.set_title(f"{GAME_TITLES.get(game, f'Game {game}')}: annotated RAM slots (* FDR q < 0.05)")
    ax.xaxis.grid(True, color="#d1d5db", lw=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_first_run_key_slot_traces(
    meta_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics_df: pd.DataFrame,
    game: int,
    out_path: Path,
    min_pearson_r: float,
) -> None:
    trace_df = metrics_df[
        (metrics_df["is_annotated_slot"] == 1)
        & (metrics_df["trace_excluded"] == 0)
        & (metrics_df["pearson_r"] >= min_pearson_r)
        & (metrics_df["value_range"] > 0)
    ].sort_values(["is_fdr_sig", "pearson_r", "address"], ascending=[False, False, True])
    if trace_df.empty:
        trace_df = metrics_df[
            (metrics_df["is_annotated_slot"] == 1)
            & (metrics_df["trace_excluded"] == 0)
            & (metrics_df["value_range"] > 0)
        ].sort_values(["pearson_r", "address"], ascending=[False, True]).head(6)
    if trace_df.empty:
        return

    first_source = str(meta_df.sort_values(["session", "block", "chunk", "real_pos_chunk"])["source_file"].iloc[0])
    file_group = meta_df[meta_df["source_file"] == first_source].sort_values("real_pos_chunk")
    if file_group.empty:
        return
    orig_idx = file_group.index.to_numpy()
    x = np.arange(len(orig_idx), dtype=int)

    items = [(int(row["address"]), str(row["slot_name"])) for _, row in trace_df.iterrows()]
    n_plots = len(items)
    n_cols = 3 if n_plots > 6 else 2
    n_rows = math.ceil(n_plots / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3.8 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, (address, name) in zip(axes_flat, items):
        row = metrics_df.loc[metrics_df["address"] == address].iloc[0]
        ax.plot(x, y_true[orig_idx, address], color="#111827", lw=1.0, ls="-", label="RAM")
        ax.plot(x, y_pred[orig_idx, address], color="#2563eb", lw=1.0, ls="--", alpha=0.95, label="decoded")
        ax.set_title(
            f"addr {address}: {name}\n"
            f"r={row['pearson_r']:.3f}, q={row['q_fdr']:.3g}, R2={row['r2']:.3f}"
        )
        ax.set_xlabel("Real-step index in first run")
        ax.set_ylabel("RAM value")
    for ax in axes_flat[n_plots:]:
        ax.axis("off")

    fig.legend(
        handles=[
            Line2D([0], [0], color="#111827", lw=1.8, ls="-", label="RAM"),
            Line2D([0], [0], color="#2563eb", lw=1.8, ls="--", label="decoded"),
        ],
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(0.98, 1.02),
    )
    fig.suptitle(
        f"{GAME_TITLES.get(game, f'Game {game}')}: RAM traces for first run only\n"
        f"{Path(first_source).stem}",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def save_key_predictions(
    meta_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    game: int,
    out_path: Path,
) -> None:
    key_map = ram_slot_labels(game)
    out_df = meta_df.reset_index(drop=True).copy()
    total_n = min(len(out_df), len(y_true), len(y_pred))
    out_df = out_df.iloc[:total_n].reset_index(drop=True)
    for address, name in key_map.items():
        out_df[f"ram_{address}_{name}_true"] = y_true[:total_n, address]
        out_df[f"ram_{address}_{name}_pred"] = y_pred[:total_n, address]
    out_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


def run_raw_status2_pca_analysis(
    args: argparse.Namespace,
    sub_dir: Path,
    game: int,
    fig_dir: Path,
    res_dir: Path,
) -> None:
    """Run the raw status==2 activation -> PCA -> Ridge analysis."""

    sessions = sorted(parse_int_list(args.decode_sessions) or {1, 2, 3, 4})
    if args.n_perm <= 0:
        raise ValueError("--n-perm must be positive because permutation FDR is mandatory.")
    if args.raw_pca_components != 50:
        print(
            f"[raw status2 PCA] note: using {args.raw_pca_components} PCs "
            f"instead of the default 50."
        )

    game_label = GAME_LABELS.get(game, f"game{game}")
    analysis_tag = (
        f"{game_label}_raw_status2_pca{args.raw_pca_components}_session_loro"
    )
    print(
        f"\n[raw status2 RAM decoding] game={game} | sessions={sessions} | "
        f"raw activation -> fold-wise PCA({args.raw_pca_components}) -> "
        f"Ridge(alpha={args.alpha})"
    )
    specs, y, meta_df, alignment_df = build_raw_status2_manifest(
        input_dir=sub_dir,
        ram_root=args.ram_root,
        game_id=game,
        sessions=set(sessions),
    )
    raw_dims = sorted({spec.raw_dim for spec in specs})
    total_raw = int(sum(spec.n_status2 for spec in specs))
    print(
        f"[raw status2 RAM decoding] raw samples={total_raw:,}, "
        f"retained real steps={len(y):,}, raw dim={raw_dims}, "
        f"RAM targets={y.shape[1]}"
    )
    y_pred, splits, splitter_name, fold_df = fit_session_loro_raw_status2_cov_pca_ridge(
        specs=specs,
        y=y,
        meta_df=meta_df,
        sessions=sessions,
        n_components=args.raw_pca_components,
        batch_size=args.raw_pca_batch_size,
        alpha=args.alpha,
    )
    p_one, q_fdr, null_mean_r = permutation_pvals_for_loro_predictions(
        y_true=y,
        y_pred=y_pred,
        splits=splits,
        n_perm=args.n_perm,
        block_size=args.perm_block_size,
        seed=args.perm_seed,
    )
    metrics_df = compute_slot_metrics(
        y_true=y,
        y_pred=y_pred,
        game=game,
        p_one_sided=p_one,
        q_fdr=q_fdr,
    )
    key_df = metrics_df[metrics_df["is_key_slot"] == 1].copy()
    sig_df = metrics_df[metrics_df["q_fdr"] < 0.05].copy()

    alignment_path = res_dir / f"{analysis_tag}_alignment_summary.csv"
    fold_path = res_dir / f"{analysis_tag}_fold_summary.csv"
    metrics_path = res_dir / f"{analysis_tag}_slot_decoding_scores.csv"
    key_path = res_dir / f"{analysis_tag}_key_slot_summary.csv"
    sig_path = res_dir / f"{analysis_tag}_fdr_significant_slots.csv"
    pred_path = res_dir / f"{analysis_tag}_key_slot_predictions.csv"
    null_path = res_dir / f"{analysis_tag}_permutation_nulls.npz"

    alignment_df.to_csv(alignment_path, index=False)
    fold_df.to_csv(fold_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    key_df.to_csv(key_path, index=False)
    sig_df.to_csv(sig_path, index=False)
    save_key_predictions(meta_df, y, y_pred, game, pred_path)
    np.savez_compressed(
        null_path,
        null_mean_r=null_mean_r.astype(np.float32),
        p_one_sided=p_one.astype(np.float32),
        q_fdr=q_fdr.astype(np.float32),
    )
    for saved_path in (
        alignment_path,
        fold_path,
        metrics_path,
        key_path,
        sig_path,
        null_path,
    ):
        print(f"Saved: {saved_path}")

    plot_ram_slot_scores_fdr(
        metrics_df,
        game,
        fig_dir / f"fig_{analysis_tag}_slot_scores_fdr.png",
    )
    plot_ram_key_slot_fdr_bars(
        metrics_df,
        game,
        fig_dir / f"fig_{analysis_tag}_key_fdr_bars.png",
    )
    plot_first_run_key_slot_traces(
        meta_df=meta_df,
        y_true=y,
        y_pred=y_pred,
        metrics_df=metrics_df,
        game=game,
        out_path=fig_dir / f"fig_{analysis_tag}_first_run_traces.png",
        min_pearson_r=args.trace_min_r,
    )

    summary_path = res_dir / f"{analysis_tag}_summary.txt"
    top = metrics_df.sort_values(["q_fdr", "pearson_r"], ascending=[True, False]).head(20)
    lines = [
        f"# {GAME_TITLES.get(game, f'Game {game}')} raw status==2 RAM decoding",
        "input: flattened raw im_vp_vectors at status==2 only",
        (
            f"decoder: exact fold-wise covariance PCA(n_components="
            f"{args.raw_pca_components}) -> Ridge(alpha={args.alpha})"
        ),
        f"validation: {splitter_name}",
        "PCA fitting: training sessions only within each held-out-session fold",
        (
            "evaluation: predictions from all status==2 activations preceding a "
            "real step are averaged for that real step"
        ),
        "fallback for real steps without status==2: none (those steps are omitted)",
        f"sessions: {sessions}",
        f"raw status==2 samples: {total_raw}",
        f"retained real steps: {len(y)}",
        f"raw activation dimensions: {raw_dims}",
        f"PCA components: {args.raw_pca_components}",
        f"RAM targets: {y.shape[1]}",
        f"permutations: {args.n_perm}",
        f"permutation block size: {args.perm_block_size}",
        f"FDR significant RAM slots (q<0.05): {len(sig_df)}",
        "",
        "Top slots by FDR q then Pearson r:",
        top[
            [
                "address",
                "slot_name",
                "pearson_r",
                "p_one_sided",
                "q_fdr",
                "r2",
                "rounded_acc",
            ]
        ].to_string(index=False),
    ]
    summary_path.write_text("\n".join(lines) + "\n")
    print(f"Saved: {summary_path}")


def run_ram_decoding_analysis(
    args: argparse.Namespace,
    sub_dir: Path,
    game: int,
    fig_dir: Path,
    res_dir: Path,
) -> None:
    if args.feature_mode == "raw_status2_pca":
        run_raw_status2_pca_analysis(
            args=args,
            sub_dir=sub_dir,
            game=game,
            fig_dir=fig_dir,
            res_dir=res_dir,
        )
        return

    sessions = sorted(parse_int_list(args.decode_sessions) or {1, 2, 3, 4})
    if args.n_perm <= 0:
        raise ValueError("--n-perm must be positive because permutation FDR is mandatory.")

    game_label = GAME_LABELS.get(game, f"game{game}")
    print(
        f"\n[RAM decoding] session LORO | game={game} | sessions={sessions} | "
        f"n_perm={args.n_perm} | block={args.perm_block_size}"
    )
    x, y, meta_df, alignment_df = build_aligned_ram_dataset(
        input_dir=sub_dir,
        ram_root=args.ram_root,
        game_id=game,
        sessions=set(sessions),
        feature_mode=args.feature_mode,
        concat_k=args.concat_k,
    )
    print(f"[RAM decoding] aligned steps={len(x):,}, features={x.shape[1]:,}, RAM targets={y.shape[1]}")
    y_pred, splits, splitter_name = fit_session_loro_ridge(
        x=x,
        y=y,
        meta_df=meta_df,
        sessions=sessions,
        alpha=args.alpha,
    )
    p_one, q_fdr, null_mean_r = permutation_pvals_for_loro_predictions(
        y_true=y,
        y_pred=y_pred,
        splits=splits,
        n_perm=args.n_perm,
        block_size=args.perm_block_size,
        seed=args.perm_seed,
    )
    metrics_df = compute_slot_metrics(y_true=y, y_pred=y_pred, game=game, p_one_sided=p_one, q_fdr=q_fdr)
    key_df = metrics_df[metrics_df["is_key_slot"] == 1].copy()
    sig_df = metrics_df[metrics_df["q_fdr"] < 0.05].copy()

    alignment_path = res_dir / f"{game_label}_session_loro_alignment_summary.csv"
    metrics_path = res_dir / f"{game_label}_session_loro_slot_decoding_scores.csv"
    key_path = res_dir / f"{game_label}_session_loro_key_slot_summary.csv"
    sig_path = res_dir / f"{game_label}_session_loro_fdr_significant_slots.csv"
    pred_path = res_dir / f"{game_label}_session_loro_key_slot_predictions.csv"
    null_path = res_dir / f"{game_label}_session_loro_permutation_nulls.npz"

    alignment_df.to_csv(alignment_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    key_df.to_csv(key_path, index=False)
    sig_df.to_csv(sig_path, index=False)
    save_key_predictions(meta_df, y, y_pred, game, pred_path)
    np.savez_compressed(
        null_path,
        null_mean_r=null_mean_r.astype(np.float32),
        p_one_sided=p_one.astype(np.float32),
        q_fdr=q_fdr.astype(np.float32),
    )
    print(f"Saved: {alignment_path}")
    print(f"Saved: {metrics_path}")
    print(f"Saved: {key_path}")
    print(f"Saved: {sig_path}")
    print(f"Saved: {null_path}")

    plot_ram_slot_scores_fdr(metrics_df, game, fig_dir / f"fig_{game_label}_session_loro_slot_scores_fdr.png")
    plot_ram_key_slot_fdr_bars(metrics_df, game, fig_dir / f"fig_{game_label}_session_loro_key_fdr_bars.png")
    plot_first_run_key_slot_traces(
        meta_df=meta_df,
        y_true=y,
        y_pred=y_pred,
        metrics_df=metrics_df,
        game=game,
        out_path=fig_dir / f"fig_{game_label}_session_loro_first_run_traces.png",
        min_pearson_r=args.trace_min_r,
    )

    summary_path = res_dir / f"{game_label}_session_loro_summary.txt"
    top = metrics_df.sort_values(["q_fdr", "pearson_r"], ascending=[True, False]).head(20)
    lines = [
        f"# {GAME_TITLES.get(game, f'Game {game}')} session-LORO RAM decoding",
        f"decoder: Ridge(alpha={args.alpha}) with {splitter_name}",
        f"sessions: {sessions}",
        f"steps: {len(x)}",
        f"features: {x.shape[1]}",
        f"RAM targets: {y.shape[1]}",
        f"permutations: {args.n_perm}",
        f"permutation block size: {args.perm_block_size}",
        f"FDR significant RAM slots (q<0.05): {len(sig_df)}",
        "",
        "Top slots by FDR q then Pearson r:",
        top[["address", "slot_name", "pearson_r", "p_one_sided", "q_fdr", "r2", "rounded_acc"]].to_string(index=False),
    ]
    summary_path.write_text("\n".join(lines) + "\n")
    print(f"Saved: {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Action distribution figure for imitation learning results."
    )
    parser.add_argument("--subject", required=True,
                        help="Subject ID, e.g. sub001")
    parser.add_argument("--game", required=True, type=int, choices=[1, 2],
                        help="Game number: 1=Pong, 2=Space Invaders")
    parser.add_argument("--ram-root", type=Path, default=DEFAULT_RAM_ROOT,
                        help="Root directory containing old-style RAM.txt files.")
    parser.add_argument("--decode-sessions", default="1,2,3,4",
                        help="Comma-separated sessions used as day-level LORO folds.")
    parser.add_argument(
        "--feature-mode",
        choices=["moments", "last", "concat", "raw_status2_pca"],
        default="moments",
        help=(
            "Representation analysis: legacy per-real-step summary modes, or "
            "raw status==2 activation -> fold-wise PCA -> Ridge."
        ),
    )
    parser.add_argument("--concat-k", type=int, default=40,
                        help="Fallback concat padding length; concat mode auto-expands to observed max.")
    parser.add_argument(
        "--raw-pca-components",
        type=int,
        default=50,
        help="Number of PCA components for --feature-mode raw_status2_pca.",
    )
    parser.add_argument(
        "--raw-pca-batch-size",
        type=int,
        default=8192,
        help=(
            "GPU batch size for raw covariance accumulation; "
            "must be at least --raw-pca-components."
        ),
    )
    parser.add_argument("--alpha", type=float, default=10.0,
                        help="Ridge regularization strength for RAM decoding.")
    parser.add_argument("--n-perm", type=int, default=1000,
                        help="Mandatory block permutations for RAM decoding FDR.")
    parser.add_argument("--perm-block-size", type=int, default=40,
                        help="Block size in real steps for held-out RAM permutation.")
    parser.add_argument("--perm-seed", type=int, default=0,
                        help="Random seed for permutation test.")
    parser.add_argument("--trace-min-r", type=float, default=DEFAULT_TRACE_MIN_R,
                        help="Minimum annotated-slot r for first-run trace figure.")
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument(
        "--skip-action-distribution",
        action="store_true",
        help="Skip the separate full-file pass used only for the action-distribution figure.",
    )
    run_mode.add_argument(
        "--action-distribution-only",
        action="store_true",
        help="Create only the action-distribution figure without rerunning RAM decoding.",
    )
    args = parser.parse_args()

    subject = args.subject
    game = args.game
    game_key = GAME_KEYS[game]
    game_title = GAME_TITLES[game]

    analysis_dir = OUT_ROOT / f"{subject}_game{game}"
    fig_dir = analysis_dir / "figures"
    res_dir = analysis_dir / "results"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    sub_dir = DATA_ROOT / subject
    if not sub_dir.exists():
        print(f"Error: subject directory not found: {sub_dir}", file=sys.stderr)
        return 1

    pattern = str(sub_dir / "ses-*" / f"*{game_key}_*.npy")
    files = sorted(Path(p) for p in glob.glob(pattern))
    if not files:
        print(f"Error: no files for subject={subject}, game={game} ({game_title})",
              file=sys.stderr)
        return 1

    print(f"Subject: {subject} | Game {game}: {game_title}")
    print(f"Found {len(files)} file(s):")
    for f in files:
        print(f"  {f}")

    if args.action_distribution_only:
        human_counts, thinker_counts = accumulate_counts(files)
        print(f"\nHuman action counts:  {human_counts}")
        print(f"Thinker action counts: {thinker_counts}")
        out_path = fig_dir / f"action_dist_{subject}_game{game}.png"
        plot_action_distribution(human_counts, thinker_counts, game_title, subject, out_path)
        return 0
    elif args.skip_action_distribution:
        print("\nSkipping action-distribution pass; running RAM decoding directly.")
    else:
        human_counts, thinker_counts = accumulate_counts(files)

        print(f"\nHuman action counts:  {human_counts}")
        print(f"Thinker action counts: {thinker_counts}")

        out_path = fig_dir / f"action_dist_{subject}_game{game}.png"
        plot_action_distribution(human_counts, thinker_counts, game_title, subject, out_path)
    run_ram_decoding_analysis(
        args=args,
        sub_dir=sub_dir,
        game=game,
        fig_dir=fig_dir,
        res_dir=res_dir,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
