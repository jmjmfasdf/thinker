#!/usr/bin/env python3
"""
Section 5 neural mechanism – encoding and RSA pipeline.

Self-contained: does not depend on 00_prepare.py.

For each (subject, session, block, game) run, the script:
  1. Loads Thinker trace chunks and builds three TR-aligned
     representations: tree_reps, im_vectors, im_vp_vectors.
     Each real step is described by the imaginary steps that
     preceded it (status != 0 between previous and current real step).
  2. Loads behavioral-data RAM features, trims the first/last 60 s,
     convolves with a canonical double-gamma HRF, and bins to TRs.
  3. Loads unsmoothed wfiltered_func_data.nii, extracts voxel
     patterns across 480-TR analysis window (60-vol trim each end)
     for four ROIs (L-hipp, R-hipp, bilateral hipp, PFC).
  4. Builds a hippocampus-PFC coupling DSM from an 11-TR
     sliding-window correlation time series.
  5. Pairwise Spearman RSA between all DSM pairs.
  6. Partial RSA: ROI ~ Thinker | RAM, ROI ~ RAM | Thinker, etc.
  7. Block-permutation null (40-TR blocks, n_perm=1000) with
     Benjamini-Hochberg FDR correction.
  8. Sensitivity: status == 2 (pure imaginary) representation
     alongside the primary (status != 0) representation.
  9. Optionally (--run-encoding) voxelwise ridge-regression encoding
     with within-run block CV, leave-one-run-out CV when multiple runs
     are available, and incremental-encoding comparison.

Output tree:
  outputs/07_encoding_rsa/sub{XXX}_game{G}/
    features/      (npz per run)
    dsms/          (npz per run)
    rsa/
      rsa_manifest.csv
      rsa_partial_manifest.csv
      rsa_permutation_manifest.csv
      rsa_nulls.npz
    encoding/
      encoding_manifest.csv
    figures/
    summary.md

Usage:
  python research_script/07_encoding_rsa.py --subject 1 --game 2
  python research_script/07_encoding_rsa.py --subject 1 --game 2 --sessions 3,4
  python research_script/07_encoding_rsa.py --subject 1 --game 2 --run-encoding
"""
from __future__ import annotations

import argparse
import gc
import os
import re
import resource
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib-cache"
os.environ["XDG_CACHE_HOME"] = "/tmp"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy import signal as scipy_signal
from scipy.spatial.distance import pdist, squareform

warnings.filterwarnings("ignore", category=RuntimeWarning)

_T0 = time.time()


def current_rss_gb() -> float:
    """Peak resident memory reported by the OS, in GB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1_000_000.0


def log(message: str) -> None:
    elapsed = time.time() - _T0
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] +{elapsed:8.1f}s rss={current_rss_gb():7.2f}GB {message}", flush=True)

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRACE_ROOT = ROOT / "test"
DEFAULT_BEHAV_ROOT = ROOT / "behavioral_data_block"
DEFAULT_FMRI_ROOT = Path("/home/jeongmin/fmri/atari/derivatives/ants_mni")
DEFAULT_ATLAS_ROOT = (
    Path(__file__).resolve().parent
    / "outputs"
    / "06_representational_mechanism"
    / "atlas"
    / "harvard_oxford"
    / "ants_mni_2p5mm_masks"
    / "masks"
)
DEFAULT_OUT_ROOT = Path(__file__).resolve().parent / "outputs" / "07_encoding_rsa"

# ── analysis constants ─────────────────────────────────────────────────────
FMRI_TRIM = 60          # volumes removed from each end
N_ANALYSIS = 480        # analysis window length in TRs
TR = 1.0                # seconds
COUPLING_WINDOW = 11    # TRs for sliding-window hipp-PFC correlation
PERM_BLOCK = 40         # TRs per permutation block
N_PERM = 1000
MAX_PCA_DIM = 100       # max PCA dims for high-dimensional concatenated features
RIDGE_ALPHAS = np.logspace(-2, 5, 15)
N_ENCODING_BLOCKS = 6   # block-wise CV folds
MAX_ROI_VOXELS = 4000   # cap for memory
EPS = 1e-12

ROI_MASKS: Dict[str, str] = {
    "left_hippocampus": "subcortical/roi-subcortical-010_Left-Hippocampus_mask.nii.gz",
    "right_hippocampus": "subcortical/roi-subcortical-020_Right-Hippocampus_mask.nii.gz",
    "hippocampus": "group/roi-HarvardOxford-Hippocampus_mask.nii.gz",
    "pfc": "group/roi-HarvardOxford-PFC_mask.nii.gz",
}

# ════════════════════════════════════════════════════════════════════════════
# Trace file discovery
# ════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TraceMeta:
    subject: int
    session: int
    block: int
    game: int
    chunk: int
    path: Path

    @property
    def block_id(self) -> Tuple[int, int, int, int]:
        return (self.subject, self.session, self.block, self.game)

    @property
    def fmri_subject(self) -> str:
        return f"sub{self.subject:03d}-{self.session}"

    @property
    def run_label(self) -> str:
        return f"sub{self.subject:03d}_ses{self.session:02d}_block{self.block:02d}_game{self.game}"


def _parse_trace_meta(path: Path) -> Optional[TraceMeta]:
    m = re.match(r"sub(\d+)-ses(\d+)-block(\d+)-game(\d+)_(\d+)\.npy$", path.name)
    if m:
        return TraceMeta(
            subject=int(m.group(1)), session=int(m.group(2)),
            block=int(m.group(3)), game=int(m.group(4)),
            chunk=int(m.group(5)), path=path,
        )
    # legacy: sub001-ses01-block3-game2/video_stat_000.npy
    mf = re.match(r"video_stat_(\d+)\.npy$", path.name)
    mp = re.match(r"sub(\d+)-ses(\d+)-block(\d+)-game(\d+)$", path.parent.name)
    if mf and mp:
        return TraceMeta(
            subject=int(mp.group(1)), session=int(mp.group(2)),
            block=int(mp.group(3)), game=int(mp.group(4)),
            chunk=int(mf.group(1)), path=path,
        )
    return None


def gather_trace_blocks(
    trace_root: Path,
    subject: int,
    game: int,
    sessions: Optional[set],
) -> Dict[Tuple[int, int, int, int], List[TraceMeta]]:
    blocks: Dict[Tuple, List[TraceMeta]] = {}
    for p in sorted(trace_root.rglob("*.npy")):
        meta = _parse_trace_meta(p)
        if meta is None:
            continue
        if meta.subject != subject or meta.game != game:
            continue
        if sessions is not None and meta.session not in sessions:
            continue
        blocks.setdefault(meta.block_id, []).append(meta)
    for key in blocks:
        blocks[key].sort(key=lambda m: m.chunk)
    return blocks


# ════════════════════════════════════════════════════════════════════════════
# Trace loading & concatenation
# ════════════════════════════════════════════════════════════════════════════

def _load_npy_dict(path: Path) -> dict:
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
        item = obj.item()
        if isinstance(item, dict):
            return item
    if hasattr(obj, "files"):
        return {k: obj[k] for k in obj.files}
    raise ValueError(f"Cannot parse as dict: {path}")


def _as_float_array(value) -> Optional[np.ndarray]:
    """Return a numeric float32 array, or None for empty/non-numeric values."""
    try:
        arr = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if arr.size == 0:
        return None
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def _merge_chunk_tree(carry_tree: dict, chunk_tree: dict) -> dict:
    """Merge two tree_reps dicts by concatenating matching keys."""
    merged: dict = {}
    for k in set(carry_tree) | set(chunk_tree):
        parts = []
        if k in carry_tree:
            parts.append(carry_tree[k])
        if k in chunk_tree:
            arr = _as_float_array(chunk_tree[k])
            if arr is not None:
                parts.append(arr)
        if len(parts) == 2:
            try:
                merged[k] = np.concatenate(parts, axis=0)
            except Exception:
                pass
        elif len(parts) == 1:
            merged[k] = parts[0]
    return merged


def _flatten_raw_step(raw) -> Optional[np.ndarray]:
    arr = _as_float_array(raw)
    if arr is None:
        return None
    return arr.reshape(-1)


def _flatten_tree_step(tree_arrays: Dict[str, np.ndarray], idx: int) -> Optional[np.ndarray]:
    """Flatten every non-empty tree_reps key at one step, using stable key order."""
    parts: List[np.ndarray] = []
    for key in sorted(tree_arrays):
        arr = tree_arrays[key]
        if 0 <= idx < len(arr):
            step = _as_float_array(arr[idx])
            if step is not None:
                parts.append(step.reshape(-1))
    if not parts:
        return None
    return np.concatenate(parts).astype(np.float32, copy=False)


def _concat_indexed_features(indices: np.ndarray, getter) -> np.ndarray:
    """Concatenate feature vectors for a real-step interval in temporal order."""
    parts: List[np.ndarray] = []
    for idx in indices:
        vec = getter(int(idx))
        if vec is not None and vec.size > 0:
            parts.append(np.asarray(vec, dtype=np.float32).reshape(-1))
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts).astype(np.float32, copy=False)


def _process_real_steps_in_buf(
    real_idx: np.ndarray,
    status_buf: np.ndarray,
    imvp_buf: list,
    imvec_buf: list,
    tree_arrays: Dict[str, np.ndarray],
    t: int,
    modes: Tuple[str, ...],
    rows: Dict[str, Dict[str, List[np.ndarray]]],
    start_pos: int = 0,
) -> None:
    """Accumulate feature rows for real steps into rows[mode][rep]."""
    for pos in range(start_pos, len(real_idx)):
        ridx = int(real_idx[pos])
        prev_ridx = int(real_idx[pos - 1]) if pos > 0 else -1

        for mode in modes:
            imag = _collect_imag_indices(status_buf, ridx, prev_ridx, mode)
            valid = imag[(imag >= 0) & (imag < t)]

            # tree_reps row
            if tree_arrays:
                rows[mode]["tree_reps"].append(
                    _concat_indexed_features(valid, lambda i: _flatten_tree_step(tree_arrays, i))
                )

            # im_vp_vectors row
            rows[mode]["im_vp_vectors"].append(
                _concat_indexed_features(
                    valid,
                    lambda i: _flatten_raw_step(imvp_buf[i]) if 0 <= i < len(imvp_buf) else None,
                )
            )

            # im_vectors row
            rows[mode]["im_vectors"].append(
                _concat_indexed_features(
                    valid,
                    lambda i: _flatten_raw_step(imvec_buf[i]) if 0 <= i < len(imvec_buf) else None,
                )
            )


def _process_real_steps_for_rep(
    real_idx: np.ndarray,
    status_buf: np.ndarray,
    imvp_buf: list,
    imvec_buf: list,
    tree_arrays: Dict[str, np.ndarray],
    t: int,
    mode: str,
    rep_name: str,
    rows: List[np.ndarray],
    start_pos: int = 0,
) -> None:
    """Accumulate one representation/mode at a time to keep peak RAM low."""
    for pos in range(start_pos, len(real_idx)):
        ridx = int(real_idx[pos])
        prev_ridx = int(real_idx[pos - 1]) if pos > 0 else -1
        imag = _collect_imag_indices(status_buf, ridx, prev_ridx, mode)
        valid = imag[(imag >= 0) & (imag < t)]

        if rep_name == "tree_reps":
            if not tree_arrays:
                continue
            rows.append(
                _concat_indexed_features(valid, lambda i: _flatten_tree_step(tree_arrays, i))
            )
        elif rep_name == "im_vp_vectors":
            rows.append(
                _concat_indexed_features(
                    valid,
                    lambda i: _flatten_raw_step(imvp_buf[i]) if 0 <= i < len(imvp_buf) else None,
                )
            )
        elif rep_name == "im_vectors":
            rows.append(
                _concat_indexed_features(
                    valid,
                    lambda i: _flatten_raw_step(imvec_buf[i]) if 0 <= i < len(imvec_buf) else None,
                )
            )
        else:
            raise ValueError(f"Unknown thinker representation: {rep_name}")


def _stack_pad_pca_rows(
    rows: List[np.ndarray],
    label: str,
    max_pca_dim: int = MAX_PCA_DIM,
) -> Optional[np.ndarray]:
    if not rows:
        log(f"    [trace:{label}] no rows")
        return None
    max_len = max(int(np.asarray(r).size) for r in rows)
    if max_len == 0:
        log(f"    [trace:{label}] only empty rows")
        return None

    dense_gb = len(rows) * max_len * np.dtype(np.float32).itemsize / 1e9
    log(f"    [trace:{label}] stack rows={len(rows)} max_len={max_len} dense={dense_gb:.2f}GB")
    mat = np.zeros((len(rows), max_len), dtype=np.float32)
    for i, row in enumerate(rows):
        r = np.asarray(row, dtype=np.float32).reshape(-1)
        if r.size:
            mat[i, :r.size] = r
        if (i + 1) % 500 == 0 or i + 1 == len(rows):
            log(f"    [trace:{label}] stacked {i + 1}/{len(rows)} rows")
    rows.clear()
    gc.collect()

    if max_pca_dim > 0 and mat.shape[1] > max_pca_dim:
        try:
            n_comp = min(max_pca_dim, mat.shape[0] - 1, mat.shape[1])
            if n_comp >= 1:
                from sklearn.decomposition import PCA

                log(f"    [trace:{label}] PCA {mat.shape[1]} -> {n_comp}")
                zmat = zscore_columns_float32(mat)
                mat = PCA(
                    n_components=n_comp,
                    random_state=0,
                    svd_solver="randomized",
                ).fit_transform(zmat).astype(np.float32, copy=False)
                del zmat
                gc.collect()
                log(f"    [trace:{label}] PCA done shape={mat.shape}")
            else:
                mat = mat[:, :max_pca_dim]
        except Exception as exc:
            log(f"    [warn] trace PCA failed for {label}: {exc}; truncating")
            mat = mat[:, :max_pca_dim]
    return mat.astype(np.float32, copy=False)


def build_thinker_feature_streaming(
    metas: List[TraceMeta],
    mode: str,
    rep_name: str,
    max_pca_dim: int = MAX_PCA_DIM,
    label: str = "",
) -> Optional[np.ndarray]:
    """
    Build exactly one Thinker representation/mode while streaming chunks.

    This preserves the same real-step interval logic as
    build_thinker_features_streaming(), but avoids holding all
    primary/s2_only and tree/im vectors in memory at the same time.
    """
    trace_label = label or f"{rep_name}_{mode}"
    rows: List[np.ndarray] = []

    carry_status: np.ndarray = np.array([], dtype=int)
    carry_imvp: list = []
    carry_imvec: list = []
    carry_tree: dict = {}

    need_imvp = rep_name == "im_vp_vectors"
    need_imvec = rep_name == "im_vectors"
    need_tree = rep_name == "tree_reps"

    log(f"    [trace:{trace_label}] start chunks={len(metas)}")
    for chunk_i, meta in enumerate(metas, start=1):
        try:
            size_gb = meta.path.stat().st_size / 1e9
        except OSError:
            size_gb = float("nan")
        log(f"    [trace:{trace_label}] chunk {chunk_i}/{len(metas)} load {meta.path.name} size={size_gb:.2f}GB")
        try:
            chunk = _load_npy_dict(meta.path)
        except Exception as exc:
            log(f"    [warn] chunk load failed {meta.path}: {exc}")
            continue

        c_status = np.asarray(chunk.get("status", []), dtype=int).reshape(-1)
        c_imvp = list(chunk.get("im_vp_vectors", [])) if need_imvp else []
        c_imvec = list(chunk.get("im_vectors", [])) if need_imvec else []
        c_tree: dict = {}
        if need_tree:
            raw_tree = chunk.get("tree_reps", {})
            if isinstance(raw_tree, dict):
                for k, v in raw_tree.items():
                    arr = _as_float_array(v)
                    if arr is not None:
                        c_tree[k] = arr
        del chunk
        gc.collect()

        merged_status = np.concatenate([carry_status, c_status])
        merged_imvp = carry_imvp + c_imvp if need_imvp else []
        merged_imvec = carry_imvec + c_imvec if need_imvec else []
        merged_tree = _merge_chunk_tree(carry_tree, c_tree) if need_tree else {}
        del c_status, c_imvp, c_imvec, c_tree

        t = len(merged_status)
        real_idx = np.flatnonzero(merged_status == 0)
        if len(real_idx) == 0:
            carry_status = merged_status
            carry_imvp = merged_imvp if need_imvp else []
            carry_imvec = merged_imvec if need_imvec else []
            carry_tree = merged_tree if need_tree else {}
            log(f"    [trace:{trace_label}] chunk {chunk_i}/{len(metas)} has no real steps; carry={t}")
            continue

        tree_arrays = {}
        if need_tree:
            tree_arrays = {
                key: arr[:t] for key, arr in merged_tree.items()
                if arr.ndim >= 1 and len(arr) >= t and arr.size > 0
            }

        start_pos = 1 if carry_status.size > 0 and carry_status[0] == 0 else 0
        before_rows = len(rows)
        _process_real_steps_for_rep(
            real_idx, merged_status, merged_imvp, merged_imvec,
            tree_arrays, t, mode, rep_name, rows,
            start_pos=start_pos,
        )
        last_real = int(real_idx[-1])
        carry_status = merged_status[last_real:]
        carry_imvp = merged_imvp[last_real:] if need_imvp else []
        carry_imvec = merged_imvec[last_real:] if need_imvec else []
        carry_tree = {
            k: v[last_real:] for k, v in merged_tree.items()
            if v.ndim >= 1 and last_real < len(v)
        } if need_tree else {}
        del merged_status, merged_imvp, merged_imvec, merged_tree, tree_arrays
        gc.collect()

        log(
            f"    [trace:{trace_label}] chunk {chunk_i}/{len(metas)} "
            f"real={len(real_idx)} added_rows={len(rows) - before_rows} total_rows={len(rows)} "
            f"carry={len(carry_status)}"
        )

    mat = _stack_pad_pca_rows(rows, trace_label, max_pca_dim=max_pca_dim)
    del rows, carry_status, carry_imvp, carry_imvec, carry_tree
    gc.collect()
    if mat is not None:
        log(f"    [trace:{trace_label}] done shape={mat.shape}")
    return mat


# ════════════════════════════════════════════════════════════════════════════
# Behavioral data (RAM)
# ════════════════════════════════════════════════════════════════════════════

def find_behavioral_file(
    behav_root: Path,
    subject: int,
    session: int,
    block: int,
    game: int,
) -> Optional[Path]:
    sub_str = f"sub-{subject:03d}"
    ses_str = f"ses-{session:02d}"
    fname = f"sub{subject:03d}-ses{session:02d}-block{block}-game{game}.npz"
    p = behav_root / sub_str / ses_str / fname
    if p.exists():
        return p
    return None


def load_ram_features(behav_path: Path) -> Optional[np.ndarray]:
    try:
        d = np.load(str(behav_path), allow_pickle=True)
        ram = np.asarray(d["RAM"], dtype=np.float32)
        return ram  # (N_frames, 128)
    except Exception as exc:
        log(f"  [warn] RAM load failed {behav_path}: {exc}")
        return None


# ════════════════════════════════════════════════════════════════════════════
# HRF utilities
# ════════════════════════════════════════════════════════════════════════════

def canonical_hrf(tr: float = 1.0, t_max: float = 32.0) -> np.ndarray:
    """SPM-like canonical double-gamma HRF sampled at the requested interval."""
    from scipy.stats import gamma as _gamma
    tr = max(float(tr), 1e-3)
    t = np.arange(0, t_max, tr)
    h = _gamma.pdf(t, 6, scale=1.0) - _gamma.pdf(t, 16, scale=1.0) / 6.0
    denom = np.sum(h)
    if abs(denom) < EPS:
        denom = np.sum(np.abs(h)) + EPS
    h = h / denom
    return h.astype(np.float64)


def convolve_hrf_columns(x: np.ndarray, hrf: np.ndarray) -> np.ndarray:
    """Convolve each column of x (T × D) with hrf using full convolution,
    then trim to original length."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    T = arr.shape[0]
    if T == 0 or arr.shape[1] == 0:
        return arr.astype(np.float32)
    conv = scipy_signal.fftconvolve(arr, np.asarray(hrf, dtype=np.float64)[:, None], mode="full", axes=0)
    return conv[:T].astype(np.float32)


def _trim_uniform_samples(
    x: np.ndarray,
    source_duration_s: float,
    trim_start_s: float,
    trim_end_s: float,
) -> Tuple[np.ndarray, float]:
    """Trim uniformly sampled features by seconds and return effective duration."""
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    n = arr.shape[0]
    if n == 0:
        return arr, 0.0
    duration = float(source_duration_s)
    if duration <= 0:
        duration = float(n)
    lo_t = max(0.0, float(trim_start_s))
    hi_t = max(lo_t, duration - max(0.0, float(trim_end_s)))
    lo = int(np.floor((lo_t / duration) * n))
    hi = int(np.ceil((hi_t / duration) * n))
    lo = max(0, min(n, lo))
    hi = max(lo, min(n, hi))
    return arr[lo:hi], max(0.0, hi_t - lo_t)


def bin_average_to_tr(x: np.ndarray, source_duration_s: float, n_tr: int) -> np.ndarray:
    """Average uniformly sampled features into 1 Hz/TR bins."""
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    n, d = arr.shape
    out = np.zeros((n_tr, d), dtype=np.float64)
    counts = np.zeros(n_tr, dtype=np.float64)
    if n == 0 or d == 0 or n_tr <= 0:
        return out.astype(np.float32)
    duration = float(source_duration_s)
    if duration <= 0:
        duration = float(n_tr)
    centers = (np.arange(n, dtype=np.float64) + 0.5) / n * duration
    bins = np.floor(centers / duration * n_tr).astype(int)
    bins = np.clip(bins, 0, n_tr - 1)
    np.add.at(out, bins, arr)
    np.add.at(counts, bins, 1.0)
    nonzero = counts > 0
    out[nonzero] /= counts[nonzero, None]
    if not np.all(nonzero):
        observed = np.flatnonzero(nonzero)
        missing = np.flatnonzero(~nonzero)
        if observed.size:
            for col in range(d):
                out[missing, col] = np.interp(missing, observed, out[observed, col])
    return np.nan_to_num(out).astype(np.float32)


def hrf_convolve_uniform_to_tr(
    x: np.ndarray,
    source_duration_s: float,
    n_tr: int,
    trim_start_s: float = 0.0,
    trim_end_s: float = 0.0,
) -> Optional[np.ndarray]:
    """Trim, HRF-convolve at source resolution, then average to TR bins."""
    trimmed, duration = _trim_uniform_samples(x, source_duration_s, trim_start_s, trim_end_s)
    if trimmed.shape[0] == 0 or trimmed.shape[1] == 0 or duration <= 0:
        return None
    sample_dt = duration / max(trimmed.shape[0], 1)
    hrf = canonical_hrf(sample_dt)
    conv = convolve_hrf_columns(trimmed, hrf)
    return bin_average_to_tr(conv, duration, n_tr)
    return out


def zscore_columns(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    mean = np.nanmean(arr, axis=0, keepdims=True)
    std = np.nanstd(arr, axis=0, keepdims=True)
    return np.nan_to_num((arr - mean) / np.where(std < EPS, 1.0, std)).astype(np.float32)


def zscore_columns_float32(x: np.ndarray) -> np.ndarray:
    """Column z-score for large feature matrices, in-place when possible."""
    arr = np.asarray(x, dtype=np.float32)
    mean = np.nanmean(arr, axis=0, keepdims=True).astype(np.float32, copy=False)
    std = np.nanstd(arr, axis=0, keepdims=True).astype(np.float32, copy=False)
    denom = np.where(std < EPS, np.float32(1.0), std).astype(np.float32, copy=False)
    arr -= mean
    arr /= denom
    return np.nan_to_num(arr, copy=False).astype(np.float32, copy=False)


# ════════════════════════════════════════════════════════════════════════════
# Thinker representation construction
# ════════════════════════════════════════════════════════════════════════════

def _collect_imag_indices(
    status: np.ndarray,
    real_idx: int,
    prev_real_idx: int,
    mode: str,
) -> np.ndarray:
    """Return imaginary step indices between prev_real and real_idx."""
    start = prev_real_idx + 1 if prev_real_idx >= 0 else 0
    between = np.arange(start, real_idx, dtype=int)
    if between.size == 0:
        return between
    if mode == "primary":
        return between[status[between] != 0]
    elif mode == "s2_only":
        return between[status[between] == 2]
    return between[status[between] != 0]


def build_thinker_features_streaming(
    metas: List[TraceMeta],
    modes: Tuple[str, ...] = ("primary", "s2_only"),
    max_pca_dim: int = MAX_PCA_DIM,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Process trace chunks one at a time to build Thinker features without OOM.

    Each chunk is loaded, processed, then discarded. Only a small carry buffer
    (from the last real step to end of chunk) is kept between chunks.

    Returns: {mode: {rep_name: (N_real × D) array}}
      modes: "primary" (status!=0) and/or "s2_only" (status==2)
      rep names: "tree_reps", "im_vp_vectors", "im_vectors"
    """
    all_rows: Dict[str, Dict[str, List[np.ndarray]]] = {
        mode: {"tree_reps": [], "im_vp_vectors": [], "im_vectors": []}
        for mode in modes
    }

    # Carry buffer: data from the last real step to end of previous chunk.
    carry_status: np.ndarray = np.array([], dtype=int)
    carry_imvp: list = []
    carry_imvec: list = []
    carry_tree: dict = {}

    for meta in metas:
        try:
            chunk = _load_npy_dict(meta.path)
        except Exception as exc:
            log(f"  [warn] chunk load failed {meta.path}: {exc}")
            continue

        c_status = np.asarray(chunk.get("status", []), dtype=int).reshape(-1)
        c_imvp   = list(chunk.get("im_vp_vectors", []))
        c_imvec  = list(chunk.get("im_vectors", []))
        c_tree: dict = {}
        raw_tree = chunk.get("tree_reps", {})
        if isinstance(raw_tree, dict):
            for k, v in raw_tree.items():
                arr = _as_float_array(v)
                if arr is not None:
                    c_tree[k] = arr
        del chunk  # free raw chunk immediately

        # Merge carry + current chunk
        merged_status = np.concatenate([carry_status, c_status])
        merged_imvp   = carry_imvp + c_imvp
        merged_imvec  = carry_imvec + c_imvec
        merged_tree   = _merge_chunk_tree(carry_tree, c_tree)
        del c_status, c_imvp, c_imvec, c_tree

        t = len(merged_status)
        real_idx = np.flatnonzero(merged_status == 0)

        if len(real_idx) == 0:
            # No real steps yet; carry everything to next chunk.
            carry_status = merged_status
            carry_imvp   = merged_imvp
            carry_imvec  = merged_imvec
            carry_tree   = merged_tree
            continue

        # Use every non-empty tree_reps key. Each step is flattened later,
        # then all imaginary steps in the interval are concatenated in order.
        tree_arrays = {
            key: arr[:t] for key, arr in merged_tree.items()
            if arr.ndim >= 1 and len(arr) >= t and arr.size > 0
        }

        # Process every real step whose previous-real anchor is known. If the
        # merged buffer starts with a carried anchor from the previous chunk,
        # skip that first real step because it was already paired before.
        start_pos = 1 if carry_status.size > 0 and carry_status[0] == 0 else 0
        _process_real_steps_in_buf(
            real_idx, merged_status, merged_imvp, merged_imvec,
            tree_arrays, t, modes, all_rows,
            start_pos=start_pos,
        )

        # Update carry: keep data from the last real step onwards.
        last_real = int(real_idx[-1])
        carry_status = merged_status[last_real:]
        carry_imvp   = merged_imvp[last_real:]
        carry_imvec  = merged_imvec[last_real:]
        carry_tree   = {
            k: v[last_real:] for k, v in merged_tree.items()
            if v.ndim >= 1 and last_real < len(v)
        }
        del merged_status, merged_imvp, merged_imvec, merged_tree

    def _stack_pad_pca(rows: List[np.ndarray]) -> Optional[np.ndarray]:
        if not rows:
            return None
        clean_rows = [np.asarray(r, dtype=np.float32).reshape(-1) for r in rows]
        max_len = max(r.size for r in clean_rows)
        if max_len == 0:
            return None
        mat = np.zeros((len(clean_rows), max_len), dtype=np.float32)
        for i, r in enumerate(clean_rows):
            if r.size:
                mat[i, :r.size] = r
        if max_pca_dim > 0 and mat.shape[1] > max_pca_dim:
            try:
                n_comp = min(max_pca_dim, mat.shape[0] - 1, mat.shape[1])
                if n_comp >= 1:
                    from sklearn.decomposition import PCA
                    mat = PCA(n_components=n_comp, random_state=0).fit_transform(
                        zscore_columns(mat)
                    ).astype(np.float32)
                else:
                    mat = mat[:, :max_pca_dim]
            except Exception:
                mat = mat[:, :max_pca_dim]
        return mat

    result: Dict[str, Dict[str, np.ndarray]] = {}
    for mode in modes:
        result[mode] = {}
        for rep in ("tree_reps", "im_vp_vectors", "im_vectors"):
            mat = _stack_pad_pca(all_rows[mode][rep])
            if mat is not None:
                result[mode][rep] = mat
    return result


def build_ram_tr_features(
    behav_path: Path,
    n_tr_full: int,
    n_tr_use: int,
    trim_start_s: float = FMRI_TRIM,
    trim_end_s: float = FMRI_TRIM,
) -> Optional[np.ndarray]:
    """Load RAM, trim first/last 60 s, HRF-convolve, then average to TRs."""
    ram = load_ram_features(behav_path)
    if ram is None or ram.shape[0] == 0:
        return None
    ram_tr = hrf_convolve_uniform_to_tr(
        ram.astype(np.float32),
        source_duration_s=float(n_tr_full),
        n_tr=n_tr_use,
        trim_start_s=trim_start_s,
        trim_end_s=trim_end_s,
    )
    if ram_tr is None:
        return None
    return zscore_columns(ram_tr)


# ════════════════════════════════════════════════════════════════════════════
# fMRI / ROI loading
# ════════════════════════════════════════════════════════════════════════════

def find_fmri_run(
    fmri_root: Path,
    fmri_subject: str,
    block: int,
    allow_smoothed_fallback: bool = False,
) -> Optional[Path]:
    """Return path to wfiltered_func_data.nii for this block."""
    session_dir = fmri_root / fmri_subject / f"Session{block}"
    p = session_dir / "wfiltered_func_data.nii"
    if p.exists():
        return p
    if allow_smoothed_fallback:
        p = session_dir / "s5_wfiltered_func_data.nii"
        if p.exists():
            return p
    return None


def load_roi_mask_nifti(mask_path: Path, ref_shape: Tuple) -> np.ndarray:
    import nibabel as nib
    img = nib.load(str(mask_path))
    data = np.asarray(img.get_fdata(dtype=np.float32))
    data = np.squeeze(data)
    if data.ndim != 3:
        raise ValueError(f"Mask must be 3D, got {data.shape}: {mask_path}")
    if tuple(data.shape) != tuple(ref_shape[:3]):
        raise ValueError(f"Mask shape {data.shape} != fMRI {ref_shape[:3]}")
    return data > 0


def extract_roi_patterns(
    fmri_path: Path,
    mask: np.ndarray,
    vol_start: int,
    vol_stop: int,
    max_voxels: int = MAX_ROI_VOXELS,
) -> Optional[np.ndarray]:
    """Extract (n_vols × n_voxels) ROI patterns from unsmoothed fMRI."""
    try:
        import nibabel as nib
    except ImportError:
        log("  [warn] nibabel not available")
        return None
    img = nib.load(str(fmri_path))
    coords = np.column_stack(np.where(mask)).astype(np.int32)
    if coords.size == 0:
        return None
    if max_voxels > 0 and len(coords) > max_voxels:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(coords), max_voxels, replace=False)
        idx.sort()
        coords = coords[idx]
    n_vols = vol_stop - vol_start
    patterns = np.empty((n_vols, len(coords)), dtype=np.float32)
    proxy = img.dataobj
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    for i, v in enumerate(range(vol_start, vol_stop)):
        vol = np.asarray(proxy[:, :, :, v], dtype=np.float32)
        patterns[i] = vol[x, y, z]
    # column-wise z-score across time
    patterns = zscore_columns(patterns)
    return patterns


# ════════════════════════════════════════════════════════════════════════════
# DSM construction
# ════════════════════════════════════════════════════════════════════════════

def build_dsm(x: np.ndarray) -> np.ndarray:
    """Correlation-distance DSM: DSM[i,j] = 1 - corr(x_i, x_j)."""
    arr = zscore_columns(np.asarray(x, dtype=np.float64))
    # rows with zero variance → set to zero (handled by zscore)
    dist = pdist(arr, metric="correlation")
    dist = np.nan_to_num(dist, nan=0.0, posinf=2.0, neginf=0.0)
    return dist.astype(np.float32)


def build_coupling_dsm(
    hipp_patterns: np.ndarray,
    pfc_patterns: np.ndarray,
    window: int = COUPLING_WINDOW,
) -> np.ndarray:
    """
    Hippocampus-PFC coupling DSM.

    coupling[t] = Pearson r over 11-TR window centred at t,
    between ROI mean time series.
    DSM[i,j] = |coupling[i] - coupling[j]|  (condensed form).
    """
    hipp_mean = np.nanmean(hipp_patterns, axis=1)  # (T,)
    pfc_mean = np.nanmean(pfc_patterns, axis=1)     # (T,)
    T = len(hipp_mean)
    half = window // 2
    coupling = np.full(T, np.nan, dtype=np.float64)
    for t in range(T):
        lo = max(0, t - half)
        hi = min(T, t + half + 1)
        h = hipp_mean[lo:hi]
        p = pfc_mean[lo:hi]
        if len(h) >= 3 and np.std(h) > EPS and np.std(p) > EPS:
            coupling[t] = np.corrcoef(h, p)[0, 1]
    coupling = np.nan_to_num(coupling, nan=0.0)
    # DSM[i,j] = |coupling_i - coupling_j|
    diff = np.abs(coupling[:, None] - coupling[None, :])
    idx = np.triu_indices(T, k=1)
    return diff[idx].astype(np.float32)


def build_temporal_lag_dsm(n_tr: int) -> np.ndarray:
    """Condensed DSM whose entries are absolute TR distance."""
    t = np.arange(n_tr, dtype=np.float32)
    diff = np.abs(t[:, None] - t[None, :])
    idx = np.triu_indices(n_tr, k=1)
    return diff[idx].astype(np.float32)


def upper_tri(dsm_sq: np.ndarray) -> np.ndarray:
    n = dsm_sq.shape[0]
    idx = np.triu_indices(n, k=1)
    return dsm_sq[idx]


# ════════════════════════════════════════════════════════════════════════════
# RSA
# ════════════════════════════════════════════════════════════════════════════

def spearman_rsa(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    aa, bb = np.asarray(a).reshape(-1), np.asarray(b).reshape(-1)
    n = min(len(aa), len(bb))
    aa, bb = aa[:n], bb[:n]
    valid = np.isfinite(aa) & np.isfinite(bb)
    aa, bb = aa[valid], bb[valid]
    if len(aa) < 3 or np.std(aa) < EPS or np.std(bb) < EPS:
        return np.nan, np.nan
    r, p = scipy_stats.spearmanr(aa, bb)
    return float(r), float(p)


def partial_spearman(
    a: np.ndarray,
    b: np.ndarray,
    controls: List[np.ndarray],
) -> Tuple[float, float]:
    """Partial Spearman rho of a ~ b controlling for each array in controls."""
    if not controls:
        return spearman_rsa(a, b)
    aa = np.asarray(a).reshape(-1)
    bb = np.asarray(b).reshape(-1)
    n = min(len(aa), len(bb), *(len(c) for c in controls))
    aa, bb = aa[:n], bb[:n]
    ctrl = np.vstack([np.asarray(c).reshape(-1)[:n] for c in controls]).T  # (n, k)
    valid = np.isfinite(aa) & np.isfinite(bb) & np.all(np.isfinite(ctrl), axis=1)
    aa, bb = aa[valid], bb[valid]
    ctrl = ctrl[valid]
    if len(aa) < 5:
        return np.nan, np.nan
    # rank-transform everything
    ra = scipy_stats.rankdata(aa)
    rb = scipy_stats.rankdata(bb)
    rc = np.column_stack([scipy_stats.rankdata(ctrl[:, k]) for k in range(ctrl.shape[1])])
    # partial correlation via residuals from linear regression on ranks
    from numpy.linalg import lstsq
    X = np.column_stack([np.ones(len(ra)), rc])
    res_a = ra - X @ lstsq(X, ra, rcond=None)[0]
    res_b = rb - X @ lstsq(X, rb, rcond=None)[0]
    if np.std(res_a) < EPS or np.std(res_b) < EPS:
        return np.nan, np.nan
    r, p = scipy_stats.pearsonr(res_a, res_b)
    return float(r), float(p)


# ════════════════════════════════════════════════════════════════════════════
# Block permutation
# ════════════════════════════════════════════════════════════════════════════

def block_permutation_rsa(
    model_dsm: np.ndarray,
    brain_patterns: np.ndarray,
    n_tr: int = N_ANALYSIS,
    block_size: int = PERM_BLOCK,
    n_perm: int = N_PERM,
    rng_seed: int = 0,
    label: str = "",
) -> Tuple[np.ndarray, float, float]:
    """
    Block-shuffle fMRI volumes, recompute brain DSM, compute rho with model_dsm.

    Returns (null_rho_array, p_one_sided, p_two_sided).
    """
    rng = np.random.default_rng(rng_seed)
    n_blocks = n_tr // block_size
    if n_blocks < 2:
        return np.array([]), np.nan, np.nan
    block_indices = [np.arange(i * block_size, min((i + 1) * block_size, n_tr))
                     for i in range(n_blocks)]
    # trim to even blocks
    if len(block_indices[-1]) < block_size:
        block_indices = block_indices[:-1]
        n_blocks = len(block_indices)
    if n_blocks < 2:
        return np.array([]), np.nan, np.nan

    obs_r, _ = spearman_rsa(model_dsm, build_dsm(brain_patterns))
    if not np.isfinite(obs_r):
        return np.array([]), np.nan, np.nan

    nulls = np.empty(n_perm, dtype=np.float32)
    progress_step = max(1, n_perm // 4)
    for i in range(n_perm):
        perm_order = rng.permutation(n_blocks)
        perm_idx = np.concatenate([block_indices[b] for b in perm_order])
        perm_patterns = brain_patterns[perm_idx]
        r_perm, _ = spearman_rsa(model_dsm, build_dsm(perm_patterns))
        nulls[i] = r_perm if np.isfinite(r_perm) else 0.0
        if label and ((i + 1) % progress_step == 0 or i + 1 == n_perm):
            log(f"    [perm:{label}] {i + 1}/{n_perm}")

    p_one = float((1 + np.sum(nulls >= obs_r)) / (1 + n_perm))
    p_two = float((1 + np.sum(np.abs(nulls) >= abs(obs_r))) / (1 + n_perm))
    return nulls, p_one, p_two


def block_permutation_coupling_rsa(
    model_dsm: np.ndarray,
    hipp_patterns: np.ndarray,
    pfc_patterns: np.ndarray,
    n_tr: int = N_ANALYSIS,
    block_size: int = PERM_BLOCK,
    n_perm: int = N_PERM,
    rng_seed: int = 0,
    label: str = "",
) -> Tuple[np.ndarray, float, float]:
    """Block-shuffle matched hippocampus/PFC volumes, rebuild coupling DSM."""
    rng = np.random.default_rng(rng_seed)
    n_blocks = n_tr // block_size
    if n_blocks < 2:
        return np.array([]), np.nan, np.nan
    block_indices = [
        np.arange(i * block_size, min((i + 1) * block_size, n_tr))
        for i in range(n_blocks)
    ]
    obs_dsm = build_coupling_dsm(hipp_patterns[:n_tr], pfc_patterns[:n_tr])
    obs_r, _ = spearman_rsa(model_dsm, obs_dsm)
    if not np.isfinite(obs_r):
        return np.array([]), np.nan, np.nan
    nulls = np.empty(n_perm, dtype=np.float32)
    progress_step = max(1, n_perm // 4)
    for i in range(n_perm):
        perm_order = rng.permutation(n_blocks)
        perm_idx = np.concatenate([block_indices[b] for b in perm_order])
        r_perm, _ = spearman_rsa(
            model_dsm,
            build_coupling_dsm(hipp_patterns[perm_idx], pfc_patterns[perm_idx]),
        )
        nulls[i] = r_perm if np.isfinite(r_perm) else 0.0
        if label and ((i + 1) % progress_step == 0 or i + 1 == n_perm):
            log(f"    [perm:{label}] {i + 1}/{n_perm}")
    p_one = float((1 + np.sum(nulls >= obs_r)) / (1 + n_perm))
    p_two = float((1 + np.sum(np.abs(nulls) >= abs(obs_r))) / (1 + n_perm))
    return nulls, p_one, p_two


def fdr_bh(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns q-values."""
    pv = np.asarray(pvals, dtype=float)
    finite = np.isfinite(pv)
    q = np.full_like(pv, np.nan)
    n = int(finite.sum())
    if n == 0:
        return q
    idx = np.where(finite)[0]
    order = np.argsort(pv[idx])
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    qvals = np.minimum(1.0, pv[idx] * n / ranks)
    # make monotone
    for k in range(n - 2, -1, -1):
        qvals[order[k]] = min(qvals[order[k]], qvals[order[k + 1]])
    q[idx] = qvals
    return q


# ════════════════════════════════════════════════════════════════════════════
# Encoding: voxelwise ridge regression
# ════════════════════════════════════════════════════════════════════════════

def _ridge_predict(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    alphas: np.ndarray,
) -> np.ndarray:
    """Fit ridge regression with best alpha (CV on train), return predictions."""
    from sklearn.linear_model import RidgeCV as _RidgeCV
    from sklearn.preprocessing import StandardScaler as _SS
    from sklearn.decomposition import PCA as _PCA
    sx = _SS()
    X_tr = sx.fit_transform(X_train)
    X_te = sx.transform(X_test)
    # PCA on X
    n_comp = min(MAX_PCA_DIM, X_tr.shape[1], X_tr.shape[0] - 1)
    if n_comp < 1:
        return np.zeros((X_test.shape[0], Y_train.shape[1] if Y_train.ndim > 1 else 1), dtype=np.float32)
    pca = _PCA(n_components=n_comp, random_state=0)
    Xp_tr = pca.fit_transform(X_tr)
    Xp_te = pca.transform(X_te)
    ridge = _RidgeCV(alphas=alphas, fit_intercept=True)
    if Y_train.ndim == 1:
        ridge.fit(Xp_tr, Y_train)
        return ridge.predict(Xp_te).astype(np.float32)
    ridge.fit(Xp_tr, Y_train)
    return ridge.predict(Xp_te).astype(np.float32)


def pearson_columns(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    """Pearson r between corresponding columns of Y_true and Y_pred."""
    yt = np.asarray(Y_true, dtype=np.float64)
    yp = np.asarray(Y_pred, dtype=np.float64)
    D = yt.shape[1]
    rs = np.full(D, np.nan, dtype=np.float32)
    for d in range(D):
        if np.std(yt[:, d]) < EPS or np.std(yp[:, d]) < EPS:
            continue
        rs[d] = float(np.corrcoef(yt[:, d], yp[:, d])[0, 1])
    return rs


def run_encoding(
    X: np.ndarray,
    Y: np.ndarray,
    n_folds: int = N_ENCODING_BLOCKS,
    alphas: np.ndarray = RIDGE_ALPHAS,
    label: str = "",
) -> np.ndarray:
    """
    Voxelwise ridge regression with block-wise CV.

    X: (T × n_features), Y: (T × n_voxels)
    Returns: mean held-out Pearson r per voxel (n_voxels,)
    """
    T = X.shape[0]
    fold_size = T // n_folds
    if fold_size < 2:
        return np.full(Y.shape[1], np.nan, dtype=np.float32)

    if label:
        log(f"      [encoding:{label}] start X={X.shape} Y={Y.shape} folds={n_folds}")
    voxel_rs = np.zeros(Y.shape[1], dtype=np.float64)
    fold_counts = np.zeros(Y.shape[1], dtype=int)

    for fold in range(n_folds):
        lo = fold * fold_size
        hi = lo + fold_size if fold < n_folds - 1 else T
        test_idx = np.arange(lo, hi)
        train_idx = np.concatenate([np.arange(0, lo), np.arange(hi, T)])
        if len(train_idx) < 5 or len(test_idx) < 2:
            continue
        if label:
            log(
                f"      [encoding:{label}] fold {fold + 1}/{n_folds} "
                f"train={len(train_idx)} test={len(test_idx)}"
            )
        X_tr, X_te = X[train_idx], X[test_idx]
        Y_tr, Y_te = Y[train_idx], Y[test_idx]
        Y_pred = _ridge_predict(X_tr, Y_tr, X_te, alphas)
        rs = pearson_columns(Y_te, Y_pred)
        valid = np.isfinite(rs)
        voxel_rs[valid] += rs[valid]
        fold_counts[valid] += 1

    result = np.full(Y.shape[1], np.nan, dtype=np.float32)
    has_data = fold_counts > 0
    result[has_data] = (voxel_rs[has_data] / fold_counts[has_data]).astype(np.float32)
    if label:
        log(f"      [encoding:{label}] done mean_r={float(np.nanmean(result)):.4f}")
    return result


def run_encoding_leave_one_run_out(
    X_runs: List[np.ndarray],
    Y_runs: List[np.ndarray],
    alphas: np.ndarray = RIDGE_ALPHAS,
    label: str = "",
) -> np.ndarray:
    """Voxelwise ridge encoding with one held-out run per fold."""
    if len(X_runs) < 2 or len(Y_runs) < 2 or len(X_runs) != len(Y_runs):
        return np.array([], dtype=np.float32)
    x_dims = {x.shape[1] for x in X_runs if x.ndim == 2}
    y_dims = {y.shape[1] for y in Y_runs if y.ndim == 2}
    if len(x_dims) != 1 or len(y_dims) != 1:
        return np.array([], dtype=np.float32)
    n_targets = next(iter(y_dims))
    voxel_rs = np.zeros(n_targets, dtype=np.float64)
    fold_counts = np.zeros(n_targets, dtype=int)
    for test_i in range(len(X_runs)):
        X_test, Y_test = X_runs[test_i], Y_runs[test_i]
        train_x = [x for i, x in enumerate(X_runs) if i != test_i]
        train_y = [y for i, y in enumerate(Y_runs) if i != test_i]
        if not train_x or X_test.shape[0] < 2:
            continue
        X_train = np.vstack(train_x)
        Y_train = np.vstack(train_y)
        if X_train.shape[0] < 5:
            continue
        if label:
            log(
                f"      [encoding:{label}] heldout_run={test_i + 1}/{len(X_runs)} "
                f"train={X_train.shape} test={X_test.shape}"
            )
        Y_pred = _ridge_predict(X_train, Y_train, X_test, alphas)
        rs = pearson_columns(Y_test, Y_pred)
        valid = np.isfinite(rs)
        voxel_rs[valid] += rs[valid]
        fold_counts[valid] += 1
    out = np.full(n_targets, np.nan, dtype=np.float32)
    valid = fold_counts > 0
    out[valid] = (voxel_rs[valid] / fold_counts[valid]).astype(np.float32)
    if label:
        log(f"      [encoding:{label}] done mean_r={float(np.nanmean(out)):.4f}")
    return out


def _encoding_summary_row(base: Dict, target_name: str, model_name: str, cv_scheme: str, rs: np.ndarray) -> Dict:
    return {
        **base,
        "target": target_name,
        "model": model_name,
        "roi": target_name,
        "cv_scheme": cv_scheme,
        "mean_r": float(np.nanmean(rs)) if rs.size else np.nan,
        "median_r": float(np.nanmedian(rs)) if rs.size else np.nan,
        "n_voxels": int(len(rs)),
        "n_voxels_positive": int(np.sum(rs > 0)) if rs.size else 0,
    }


def _encoding_model_map(result: Dict) -> Dict[str, np.ndarray]:
    models: Dict[str, np.ndarray] = {}
    ram_tr = result.get("ram_tr")
    thinker_feats = result.get("features", {})
    if isinstance(ram_tr, np.ndarray):
        models["ram"] = ram_tr
    for thinker_name, thinker_x in thinker_feats.items():
        if isinstance(thinker_x, np.ndarray):
            models[f"thinker_{thinker_name}"] = thinker_x
            if isinstance(ram_tr, np.ndarray):
                min_t = min(ram_tr.shape[0], thinker_x.shape[0])
                models[f"ram_plus_{thinker_name}"] = np.hstack([
                    ram_tr[:min_t],
                    thinker_x[:min_t],
                ])
    return models


def run_runwise_encoding(run_results: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Add subject/game-level leave-one-run-out encoding rows when possible."""
    if len(run_results) < 2:
        return [], []
    first_meta = run_results[0]["meta"]
    base = {
        "run_label": "ALL_RUNS",
        "subject": first_meta.subject,
        "session": -1,
        "block": -1,
        "game": first_meta.game,
    }
    model_maps = [_encoding_model_map(r) for r in run_results]
    common_models = set(model_maps[0])
    for mm in model_maps[1:]:
        common_models &= set(mm)
    rows: List[Dict] = []
    delta_rows: List[Dict] = []

    def _collect(model_name: str, target_getter) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []
        for result, models in zip(run_results, model_maps):
            X = models.get(model_name)
            Y = target_getter(result)
            if X is None or Y is None:
                return [], []
            min_t = min(X.shape[0], Y.shape[0])
            xs.append(X[:min_t])
            ys.append(Y[:min_t])
        if len({x.shape[1] for x in xs}) != 1 or len({y.shape[1] for y in ys}) != 1:
            return [], []
        return xs, ys

    # Thinker -> RAM.
    if all(isinstance(r.get("ram_tr"), np.ndarray) for r in run_results):
        for model_name in sorted(m for m in common_models if m.startswith("thinker_")):
            xs, ys = _collect(model_name, lambda r: r.get("ram_tr"))
            if xs:
                rs = run_encoding_leave_one_run_out(xs, ys, label=f"ALL_RUNS:ram:{model_name}")
                rows.append(_encoding_summary_row(base, "ram", model_name, "leave_one_run_out", rs))

    # RAM / Thinker / RAM+Thinker -> ROI BOLD.
    common_rois = set(run_results[0].get("roi_patterns", {}))
    for result in run_results[1:]:
        common_rois &= set(result.get("roi_patterns", {}))
    score_cache: Dict[Tuple[str, str], np.ndarray] = {}
    for roi_name in sorted(common_rois):
        for model_name in sorted(common_models):
            xs, ys = _collect(model_name, lambda r, roi=roi_name: r.get("roi_patterns", {}).get(roi))
            if not xs:
                continue
            rs = run_encoding_leave_one_run_out(xs, ys, label=f"ALL_RUNS:{roi_name}:{model_name}")
            score_cache[(roi_name, model_name)] = rs
            rows.append(_encoding_summary_row(base, roi_name, model_name, "leave_one_run_out", rs))

        for model_name in sorted(m for m in common_models if m.startswith("thinker_")):
            suffix = model_name.removeprefix("thinker_")
            combo_name = f"ram_plus_{suffix}"
            combo_rs = score_cache.get((roi_name, combo_name))
            ram_rs = score_cache.get((roi_name, "ram"))
            thinker_rs = score_cache.get((roi_name, model_name))
            if combo_rs is not None and ram_rs is not None and combo_rs.shape == ram_rs.shape:
                delta = combo_rs - ram_rs
                delta_rows.append({
                    **base,
                    "target": roi_name,
                    "comparison": f"{combo_name}_minus_ram",
                    "mean_delta_r": float(np.nanmean(delta)),
                    "median_delta_r": float(np.nanmedian(delta)),
                    "n_positive_delta": int(np.sum(delta > 0)),
                })
            if combo_rs is not None and thinker_rs is not None and combo_rs.shape == thinker_rs.shape:
                delta = combo_rs - thinker_rs
                delta_rows.append({
                    **base,
                    "target": roi_name,
                    "comparison": f"{combo_name}_minus_{model_name}",
                    "mean_delta_r": float(np.nanmean(delta)),
                    "median_delta_r": float(np.nanmedian(delta)),
                    "n_positive_delta": int(np.sum(delta > 0)),
                })
    return rows, delta_rows


# ════════════════════════════════════════════════════════════════════════════
# Per-run analysis
# ════════════════════════════════════════════════════════════════════════════

def analyze_run(
    run_label: str,
    meta: TraceMeta,
    metas: List[TraceMeta],
    args: argparse.Namespace,
) -> Optional[Dict]:
    """Full pipeline for one fMRI run (one block)."""

    log(f"  [run] {run_label} start chunks={len(metas)}")

    # ── 1. Check trace chunks exist ────────────────────────────────────────
    if not metas:
        log(f"    [skip] no trace chunks")
        return None

    # ── 2. Find & load fMRI ────────────────────────────────────────────────
    fmri_path = find_fmri_run(
        args.fmri_root,
        meta.fmri_subject,
        meta.block,
        allow_smoothed_fallback=args.allow_smoothed_fallback,
    )
    if fmri_path is None:
        log(f"    [skip] no fMRI: {meta.fmri_subject}/Session{meta.block}")
        return None

    try:
        import nibabel as nib
        img = nib.load(str(fmri_path))
        n_vols = img.shape[3]
        fmri_shape = img.shape
    except Exception as exc:
        log(f"    [skip] fMRI load error: {exc}")
        return None

    vol_start, vol_stop = FMRI_TRIM, n_vols - FMRI_TRIM
    n_tr = vol_stop - vol_start
    if n_tr < 60:
        log(f"    [skip] too few TRs after trim: {n_tr}")
        return None
    n_tr_use = min(n_tr, N_ANALYSIS)
    log(f"    [fmri] loaded {fmri_path} shape={fmri_shape} n_tr_use={n_tr_use}")

    # ── 3. Load ROI masks & patterns ──────────────────────────────────────
    roi_patterns: Dict[str, np.ndarray] = {}
    for roi_name, rel_path in ROI_MASKS.items():
        mask_path = args.atlas_root / rel_path
        if not mask_path.exists():
            continue
        try:
            log(f"    [roi] {roi_name} load mask/extract")
            mask = load_roi_mask_nifti(mask_path, fmri_shape)
            patterns = extract_roi_patterns(
                fmri_path, mask, vol_start, vol_start + n_tr_use,
                max_voxels=args.max_roi_voxels,
            )
            if patterns is not None:
                roi_patterns[roi_name] = patterns
                log(f"    [roi] {roi_name} patterns={patterns.shape}")
        except Exception as exc:
            log(f"    [warn] ROI {roi_name}: {exc}")

    if not roi_patterns:
        log(f"    [skip] no ROI patterns loaded")
        return None

    # ── 4. Build Thinker TR features (primary + sensitivity, streaming) ──────
    thinker_feats: Dict[str, np.ndarray] = {}
    mode_key_map = {"primary": "primary", "s2_only": "s2only"}
    for mode, mode_key in mode_key_map.items():
        for rep_name in ("tree_reps", "im_vp_vectors", "im_vectors"):
            feat_name = f"{rep_name}_{mode_key}"
            feat_mat = build_thinker_feature_streaming(
                metas,
                mode=mode,
                rep_name=rep_name,
                max_pca_dim=MAX_PCA_DIM,
                label=f"{run_label}:{feat_name}",
            )
            if feat_mat is None:
                continue
            if feat_mat.shape[0] < 5:
                log(f"    [trace:{run_label}:{feat_name}] skip too few rows={feat_mat.shape[0]}")
                del feat_mat
                gc.collect()
                continue
            log(f"    [trace:{run_label}:{feat_name}] HRF/TR input={feat_mat.shape}")
            feat_tr = hrf_convolve_uniform_to_tr(
                feat_mat,
                source_duration_s=float(n_tr_use),
                n_tr=n_tr_use,
            )
            del feat_mat
            gc.collect()
            if feat_tr is None:
                continue
            feat_tr = zscore_columns(feat_tr)
            thinker_feats[feat_name] = feat_tr
            log(f"    [trace:{run_label}:{feat_name}] TR feature={feat_tr.shape}")

    # ── 5. RAM TR features ────────────────────────────────────────────────
    ram_tr: Optional[np.ndarray] = None
    if args.behav_root is not None:
        behav_path = find_behavioral_file(
            args.behav_root, meta.subject, meta.session, meta.block, meta.game
        )
        if behav_path is not None:
            log(f"    [ram] load {behav_path}")
            ram_tr = build_ram_tr_features(behav_path, n_vols, n_tr_use)
            if ram_tr is not None:
                log(f"    [ram] TR feature={ram_tr.shape}")
        else:
            log(f"    [ram] missing behavioral file")

    # ── 6. Build DSMs ─────────────────────────────────────────────────────
    dsms: Dict[str, np.ndarray] = {}

    for feat_name, feat_mat in thinker_feats.items():
        log(f"    [dsm] thinker {feat_name} shape={feat_mat.shape}")
        dsms[feat_name] = build_dsm(feat_mat)

    if ram_tr is not None:
        log(f"    [dsm] ram shape={ram_tr.shape}")
        dsms["ram"] = build_dsm(ram_tr)

    for roi_name, patterns in roi_patterns.items():
        log(f"    [dsm] bold_{roi_name} shape={patterns.shape}")
        dsms[f"bold_{roi_name}"] = build_dsm(patterns)

    if "bold_left_hippocampus" in dsms and "bold_right_hippocampus" in dsms:
        dsms["bold_mean_hippocampus"] = (
            dsms["bold_left_hippocampus"] + dsms["bold_right_hippocampus"]
        ) / 2.0

    dsms["temporal_lag"] = build_temporal_lag_dsm(n_tr_use)

    # coupling DSM: bilateral hippocampus mean time series × PFC mean time series.
    hipp_patterns_for_coupling: Optional[np.ndarray] = None
    if "left_hippocampus" in roi_patterns and "right_hippocampus" in roi_patterns:
        hipp_patterns_for_coupling = np.hstack([
            roi_patterns["left_hippocampus"],
            roi_patterns["right_hippocampus"],
        ])
    elif "hippocampus" in roi_patterns:
        hipp_patterns_for_coupling = roi_patterns["hippocampus"]
    elif "left_hippocampus" in roi_patterns:
        hipp_patterns_for_coupling = roi_patterns["left_hippocampus"]
    elif "right_hippocampus" in roi_patterns:
        hipp_patterns_for_coupling = roi_patterns["right_hippocampus"]

    if hipp_patterns_for_coupling is not None and "pfc" in roi_patterns:
        try:
            coupling_dsm = build_coupling_dsm(
                hipp_patterns_for_coupling, roi_patterns["pfc"]
            )
            dsms["coupling_hipp_pfc"] = coupling_dsm
            log(f"    [dsm] coupling_hipp_pfc len={len(coupling_dsm)}")
        except Exception as exc:
            log(f"    [warn] coupling DSM: {exc}")

    # ── 7. RSA comparisons ────────────────────────────────────────────────
    thinker_keys = [k for k in dsms if any(k.startswith(r) for r in ("tree_reps", "im_vectors", "im_vp_vectors"))]
    ram_key = "ram" if "ram" in dsms else None
    roi_keys = [k for k in dsms if k.startswith("bold_") or k == "coupling_hipp_pfc"]

    rsa_rows: List[Dict] = []
    base = {
        "run_label": run_label,
        "subject": meta.subject,
        "session": meta.session,
        "block": meta.block,
        "game": meta.game,
    }

    def _rsa_row(dsm_a_name, dsm_b_name):
        a, b = dsms.get(dsm_a_name), dsms.get(dsm_b_name)
        if a is None or b is None:
            return
        r, p = spearman_rsa(a, b)
        rsa_rows.append({**base, "dsm_a": dsm_a_name, "dsm_b": dsm_b_name, "rho": r, "p": p})

    # Thinker × RAM
    for tk in thinker_keys:
        if ram_key:
            _rsa_row(tk, ram_key)

    # RAM × ROI
    if ram_key:
        for rk in roi_keys:
            _rsa_row(ram_key, rk)

    # Thinker × ROI
    for tk in thinker_keys:
        for rk in roi_keys:
            _rsa_row(tk, rk)

    # ── 8. Partial RSA ────────────────────────────────────────────────────
    partial_rows: List[Dict] = []

    def _partial_row(a_name, b_name, ctrl_names):
        a = dsms.get(a_name)
        b = dsms.get(b_name)
        ctrls = [dsms[c] for c in ctrl_names if c in dsms]
        if a is None or b is None or not ctrls:
            return
        r, p = partial_spearman(a, b, ctrls)
        partial_rows.append({
            **base, "dsm_a": a_name, "dsm_b": b_name,
            "controls": ",".join(ctrl_names), "rho": r, "p": p,
        })

    if ram_key:
        for tk in thinker_keys:
            for rk in roi_keys:
                _partial_row(rk, tk, ["temporal_lag"])
                _partial_row(rk, ram_key, ["temporal_lag"])
                _partial_row(rk, tk, [ram_key, "temporal_lag"])
                _partial_row(rk, ram_key, [tk, "temporal_lag"])

    # ── 9. Block permutation ──────────────────────────────────────────────
    perm_rows: List[Dict] = []
    null_store: Dict[str, np.ndarray] = {}

    if not args.skip_permutation:
        for model_name in thinker_keys + ([ram_key] if ram_key else []):
            model_dsm = dsms.get(model_name)
            if model_dsm is None:
                continue
            for roi_name, patterns in roi_patterns.items():
                key = f"{model_name}_vs_bold_{roi_name}"
                log(f"    [perm] {key} n_perm={args.n_perm}")
                nulls, p1, p2 = block_permutation_rsa(
                    model_dsm, patterns,
                    n_tr=n_tr_use,
                    block_size=PERM_BLOCK,
                    n_perm=args.n_perm,
                    label=f"{run_label}:{key}",
                )
                obs_r, _ = spearman_rsa(model_dsm, dsms.get(f"bold_{roi_name}", model_dsm))
                perm_rows.append({
                    **base, "model": model_name, "roi": roi_name,
                    "rho_obs": obs_r, "p_one_sided": p1, "p_two_sided": p2,
                })
                if nulls.size > 0:
                    null_store[key] = nulls
            if hipp_patterns_for_coupling is not None and "pfc" in roi_patterns and "coupling_hipp_pfc" in dsms:
                key = f"{model_name}_vs_coupling_hipp_pfc"
                log(f"    [perm] {key} n_perm={args.n_perm}")
                nulls, p1, p2 = block_permutation_coupling_rsa(
                    model_dsm,
                    hipp_patterns_for_coupling,
                    roi_patterns["pfc"],
                    n_tr=n_tr_use,
                    block_size=PERM_BLOCK,
                    n_perm=args.n_perm,
                    label=f"{run_label}:{key}",
                )
                obs_r, _ = spearman_rsa(model_dsm, dsms["coupling_hipp_pfc"])
                perm_rows.append({
                    **base, "model": model_name, "roi": "coupling_hipp_pfc",
                    "rho_obs": obs_r, "p_one_sided": p1, "p_two_sided": p2,
                })
                if nulls.size > 0:
                    null_store[key] = nulls

    # ── 10. Encoding ──────────────────────────────────────────────────────
    enc_rows: List[Dict] = []
    enc_delta_rows: List[Dict] = []
    if args.run_encoding and roi_patterns:
        def _append_encoding_row(target_name: str, model_name: str, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
            min_t = min(X.shape[0], Y.shape[0])
            label = f"{run_label}:{target_name}:{model_name}"
            rs = run_encoding(
                X[:min_t],
                Y[:min_t],
                n_folds=args.encoding_folds,
                label=label,
            )
            enc_rows.append({
                **base,
                "target": target_name,
                "model": model_name,
                "roi": target_name,
                "cv_scheme": "within_run_block",
                "mean_r": float(np.nanmean(rs)),
                "median_r": float(np.nanmedian(rs)),
                "n_voxels": len(rs),
                "n_voxels_positive": int(np.sum(rs > 0)),
            })
            return rs

        if ram_tr is not None:
            for thinker_name, thinker_x in thinker_feats.items():
                _append_encoding_row(
                    "ram",
                    f"thinker_{thinker_name}",
                    thinker_x,
                    ram_tr,
                )

        for roi_name, Y in roi_patterns.items():
            score_cache: Dict[str, np.ndarray] = {}
            if ram_tr is not None:
                score_cache["ram"] = _append_encoding_row(roi_name, "ram", ram_tr, Y)
            for thinker_name, thinker_x in thinker_feats.items():
                thinker_model = f"thinker_{thinker_name}"
                score_cache[thinker_model] = _append_encoding_row(roi_name, thinker_model, thinker_x, Y)
                if ram_tr is not None:
                    min_t = min(ram_tr.shape[0], thinker_x.shape[0])
                    combo_name = f"ram_plus_{thinker_name}"
                    combo_x = np.hstack([ram_tr[:min_t], thinker_x[:min_t]])
                    score_cache[combo_name] = _append_encoding_row(roi_name, combo_name, combo_x, Y)
                    del combo_x
                    gc.collect()
                    ram_rs = score_cache.get("ram")
                    combo_rs = score_cache.get(combo_name)
                    thinker_rs = score_cache.get(thinker_model)
                    if ram_rs is not None and combo_rs is not None:
                        delta = combo_rs - ram_rs
                        enc_delta_rows.append({
                            **base,
                            "target": roi_name,
                            "comparison": f"{combo_name}_minus_ram",
                            "mean_delta_r": float(np.nanmean(delta)),
                            "median_delta_r": float(np.nanmedian(delta)),
                            "n_positive_delta": int(np.sum(delta > 0)),
                        })
                    if thinker_rs is not None and combo_rs is not None:
                        delta = combo_rs - thinker_rs
                        enc_delta_rows.append({
                            **base,
                            "target": roi_name,
                            "comparison": f"{combo_name}_minus_thinker_{thinker_name}",
                            "mean_delta_r": float(np.nanmean(delta)),
                            "median_delta_r": float(np.nanmedian(delta)),
                            "n_positive_delta": int(np.sum(delta > 0)),
                        })

    gc.collect()
    log(f"  [run] {run_label} done dsms={len(dsms)} rsa={len(rsa_rows)} enc={len(enc_rows)}")
    return {
        "run_label": run_label,
        "meta": meta,
        "dsms": dsms,
        "features": thinker_feats,
        "ram_tr": ram_tr,
        "roi_patterns": roi_patterns,
        "rsa_rows": rsa_rows,
        "partial_rows": partial_rows,
        "perm_rows": perm_rows,
        "null_store": null_store,
        "enc_rows": enc_rows,
        "enc_delta_rows": enc_delta_rows,
        "n_tr": n_tr_use,
    }


# ════════════════════════════════════════════════════════════════════════════
# Figures
# ════════════════════════════════════════════════════════════════════════════

def _dsm_to_square(condensed: np.ndarray) -> np.ndarray:
    from scipy.spatial.distance import squareform
    return squareform(condensed)


def plot_dsm_panel(dsms: Dict[str, np.ndarray], run_label: str, out_path: Path) -> None:
    keys = list(dsms.keys())[:8]
    if not keys:
        return
    ncols = min(4, len(keys))
    nrows = (len(keys) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.0 * nrows))
    axes_flat = np.array(axes).reshape(-1)
    for i, key in enumerate(keys):
        dsm = dsms[key]
        try:
            sq = _dsm_to_square(dsm)
        except Exception:
            axes_flat[i].axis("off")
            continue
        im = axes_flat[i].imshow(sq, cmap="viridis", aspect="auto", interpolation="nearest")
        axes_flat[i].set_title(key, fontsize=8)
        plt.colorbar(im, ax=axes_flat[i], fraction=0.046, pad=0.04)
    for j in range(len(keys), len(axes_flat)):
        axes_flat[j].axis("off")
    fig.suptitle(run_label, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_rsa_heatmap(df: pd.DataFrame, title: str, out_path: Path) -> None:
    if df.empty or "dsm_a" not in df.columns:
        return
    pivot = df.pivot_table(index="dsm_a", columns="dsm_b", values="rho", aggfunc="mean")
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(max(6, pivot.shape[1] * 1.2), max(5, pivot.shape[0] * 0.8)))
    im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=7)
    plt.colorbar(im, ax=ax, label="Spearman rho")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_permutation_nulls(null_store: Dict[str, np.ndarray], obs_rows: pd.DataFrame, out_path: Path) -> None:
    keys = list(null_store.keys())[:6]
    if not keys:
        return
    ncols = min(3, len(keys))
    nrows = (len(keys) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes_flat = np.array(axes).reshape(-1)
    for i, key in enumerate(keys):
        nulls = null_store[key]
        ax = axes_flat[i]
        ax.hist(nulls, bins=40, color="#93c5fd", edgecolor="#1f2937", alpha=0.8)
        ax.set_title(key[:40], fontsize=7)
        ax.set_xlabel("null rho")
        # overlay observed
        if not obs_rows.empty:
            parts = key.split("_vs_bold_")
            if len(parts) == 2:
                model_name, roi_name = parts
                rows = obs_rows[
                    (obs_rows["model"] == model_name) &
                    (obs_rows["roi"] == roi_name)
                ]
                if not rows.empty:
                    obs = float(rows["rho_obs"].iloc[0])
                    ax.axvline(obs, color="#ef4444", lw=1.5, label=f"obs={obs:.3f}")
                    ax.legend(fontsize=7)
    for j in range(len(keys), len(axes_flat)):
        axes_flat[j].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_encoding_bars(df: pd.DataFrame, title: str, out_path: Path) -> None:
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ["mean_r", "median_r"]):
        pivot = df.pivot_table(index="roi", columns="model", values=metric, aggfunc="mean")
        if pivot.empty:
            ax.axis("off")
            continue
        x = np.arange(len(pivot))
        width = 0.8 / max(len(pivot.columns), 1)
        for j, col in enumerate(pivot.columns):
            ax.bar(x + j * width, pivot[col].values, width, label=col, alpha=0.85)
        ax.set_xticks(x + width * (len(pivot.columns) - 1) / 2)
        ax.set_xticklabels(pivot.index, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(metric)
        ax.legend(fontsize=8)
        ax.axhline(0, color="black", lw=0.8, ls="--")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ════════════════════════════════════════════════════════════════════════════

def run_subject_game(args: argparse.Namespace) -> None:
    subject, game = args.subject, args.game
    sessions = set(int(s) for s in args.sessions.split(",")) if args.sessions else None

    label = f"sub{subject:03d}_game{game}"
    out_root = args.output_root / label
    dirs = {
        "features": out_root / "features",
        "dsms": out_root / "dsms",
        "rsa": out_root / "rsa",
        "encoding": out_root / "encoding",
        "figures": out_root / "figures",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # Discover all blocks
    blocks = gather_trace_blocks(args.trace_root, subject, game, sessions)
    if not blocks:
        log(f"[error] no trace files for sub{subject:03d} game{game}")
        return

    log(f"[sub{subject:03d} game{game}] found {len(blocks)} blocks")

    all_rsa: List[Dict] = []
    all_partial: List[Dict] = []
    all_perm: List[Dict] = []
    all_enc: List[Dict] = []
    all_enc_delta: List[Dict] = []
    all_nulls: Dict[str, np.ndarray] = {}
    run_results: List[Dict] = []

    for block_id, metas in sorted(blocks.items()):
        sub, ses, blk, gm = block_id
        representative = metas[0]
        run_label = representative.run_label

        result = analyze_run(run_label, representative, metas, args)
        if result is None:
            continue

        run_results.append(result)
        all_rsa.extend(result["rsa_rows"])
        all_partial.extend(result["partial_rows"])
        all_perm.extend(result["perm_rows"])
        all_enc.extend(result["enc_rows"])
        all_enc_delta.extend(result.get("enc_delta_rows", []))
        for k, v in result["null_store"].items():
            all_nulls[f"{run_label}/{k}"] = v

        # per-run DSM figure
        try:
            log(f"    [save] DSM figure {run_label}")
            plot_dsm_panel(
                result["dsms"], run_label,
                dirs["figures"] / f"dsm_panel_{run_label}.png"
            )
        except Exception as exc:
            log(f"    [warn] DSM plot: {exc}")

        # save DSMs
        log(f"    [save] DSM npz {run_label}")
        np.savez_compressed(
            dirs["dsms"] / f"dsms_{run_label}.npz",
            **{k: v for k, v in result["dsms"].items() if isinstance(v, np.ndarray)},
        )

        # save features
        feat_save = {}
        for k, v in result["features"].items():
            if isinstance(v, np.ndarray):
                feat_save[k] = v
        if result["ram_tr"] is not None:
            feat_save["ram_tr"] = result["ram_tr"]
        if feat_save:
            log(f"    [save] features npz {run_label}")
            np.savez_compressed(dirs["features"] / f"features_{run_label}.npz", **feat_save)

    if not run_results:
        log(f"[{label}] no valid runs, exiting.")
        return

    if args.run_encoding and len(run_results) >= 2:
        log(f"[{label}] runwise leave-one-run-out encoding")
        runwise_rows, runwise_delta_rows = run_runwise_encoding(run_results)
        all_enc.extend(runwise_rows)
        all_enc_delta.extend(runwise_delta_rows)

    # ── Aggregate RSA ──────────────────────────────────────────────────────
    rsa_df = pd.DataFrame(all_rsa)
    partial_df = pd.DataFrame(all_partial)
    perm_df = pd.DataFrame(all_perm)
    enc_df = pd.DataFrame(all_enc)
    enc_delta_df = pd.DataFrame(all_enc_delta)

    if not perm_df.empty and "p_one_sided" in perm_df.columns:
        perm_df["q_fdr_one_sided"] = fdr_bh(perm_df["p_one_sided"].values)
        perm_df["q_fdr_two_sided"] = fdr_bh(perm_df["p_two_sided"].values)

    if not rsa_df.empty and "p" in rsa_df.columns:
        rsa_df["q_fdr"] = fdr_bh(rsa_df["p"].values)

    rsa_df.to_csv(dirs["rsa"] / "rsa_manifest.csv", index=False)
    partial_df.to_csv(dirs["rsa"] / "rsa_partial_manifest.csv", index=False)
    perm_df.to_csv(dirs["rsa"] / "rsa_permutation_manifest.csv", index=False)

    if all_nulls:
        np.savez_compressed(
            dirs["rsa"] / "rsa_nulls.npz",
            **{k.replace("/", "__"): v for k, v in all_nulls.items()},
        )

    if not enc_df.empty:
        enc_df.to_csv(dirs["encoding"] / "encoding_manifest.csv", index=False)
    if not enc_delta_df.empty:
        enc_delta_df.to_csv(dirs["encoding"] / "encoding_incremental_manifest.csv", index=False)

    # ── Aggregate figures ──────────────────────────────────────────────────
    if not rsa_df.empty:
        plot_rsa_heatmap(
            rsa_df,
            f"{label} – RSA rho (mean over runs)",
            dirs["figures"] / "rsa_heatmap_aggregate.png",
        )
    if not perm_df.empty:
        obs_repr = perm_df.rename(columns={"rho_obs": "rho"}) if "rho_obs" in perm_df.columns else perm_df
        try:
            combined_nulls = {
                k.split("/")[-1]: v for k, v in all_nulls.items()
            }
            plot_permutation_nulls(
                combined_nulls, perm_df,
                dirs["figures"] / "perm_nulls.png",
            )
        except Exception as exc:
            log(f"  [warn] perm null plot: {exc}")

    if not enc_df.empty:
        plot_encoding_bars(enc_df, f"{label} – encoding Pearson r", dirs["figures"] / "encoding_bars.png")

    # ── FDR-significant results ────────────────────────────────────────────
    sig_lines: List[str] = []
    if not perm_df.empty and "q_fdr_one_sided" in perm_df.columns:
        sig = perm_df[perm_df["q_fdr_one_sided"] < 0.05]
        if not sig.empty:
            sig_lines.append("\n### Significant RSA (FDR q < 0.05, block permutation)")
            sig_lines.append(sig[["run_label", "model", "roi", "rho_obs", "p_one_sided", "q_fdr_one_sided"]].to_string(index=False))

    summary_lines = [
        f"# {label} encoding/RSA summary",
        f"",
        f"Runs analyzed: {len(run_results)}",
        f"RSA comparisons: {len(rsa_df)}",
        f"Partial RSA comparisons: {len(partial_df)}",
        f"Permutation rows: {len(perm_df)}",
        f"Encoding rows: {len(enc_df)}",
        f"Encoding incremental rows: {len(enc_delta_df)}",
    ]
    if not rsa_df.empty:
        summary_lines.append("\n### Mean RSA rho by DSM pair (across runs)")
        if "rho" in rsa_df.columns:
            mean_rho = rsa_df.groupby(["dsm_a", "dsm_b"])["rho"].mean().reset_index()
            summary_lines.append(mean_rho.to_string(index=False))
    summary_lines.extend(sig_lines)

    (out_root / "summary.md").write_text("\n".join(summary_lines) + "\n")

    log(f"[{label}] done -> {out_root}")
    if not perm_df.empty and "q_fdr_one_sided" in perm_df.columns:
        n_sig = int((perm_df["q_fdr_one_sided"] < 0.05).sum())
        log(f"  significant (FDR q<0.05, one-sided): {n_sig}/{len(perm_df)} comparisons")


# ════════════════════════════════════════════════════════════════════════════
# Argument parser
# ════════════════════════════════════════════════════════════════════════════

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject", type=int, required=True)
    p.add_argument("--game", type=int, required=True)
    p.add_argument("--sessions", default=None,
                   help="Comma-separated session ids, e.g. 3,4")
    p.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    p.add_argument("--behav-root", type=Path, default=DEFAULT_BEHAV_ROOT)
    p.add_argument("--fmri-root", type=Path, default=DEFAULT_FMRI_ROOT)
    p.add_argument("--atlas-root", type=Path, default=DEFAULT_ATLAS_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--allow-smoothed-fallback", action="store_true",
                   help="Use s5_wfiltered_func_data.nii only if unsmoothed wfiltered_func_data.nii is absent")
    p.add_argument("--max-roi-voxels", type=int, default=MAX_ROI_VOXELS)
    p.add_argument("--n-perm", type=int, default=N_PERM)
    p.add_argument("--skip-permutation", action="store_true",
                   help="Skip block permutation test (for fast testing)")
    p.add_argument("--run-encoding", action="store_true",
                   help="Run voxelwise ridge encoding (slow, ~minutes per run)")
    p.add_argument("--encoding-folds", type=int, default=N_ENCODING_BLOCKS)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    run_subject_game(args)


if __name__ == "__main__":
    main()
