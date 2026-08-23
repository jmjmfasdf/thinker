#!/usr/bin/env python3
"""
Integrated encoding/RSA analysis.

This script is self-contained and does not import the older analysis scripts.

Pipeline:
  1. Build run-level TR-aligned Thinker and RAM features.
  2. Extract ROI BOLD patterns.
  3. Run RSA and RSA block-permutation tests.
  4. Run leave-one-run-out voxelwise encoding with alpha grid search.
  5. Generate the original summary figures, LORO encoding plots, and the
     paper-style figures.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import re
import resource
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib-cache"
os.environ["XDG_CACHE_HOME"] = "/tmp"
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy import signal as scipy_signal
from scipy import stats as scipy_stats
from scipy.spatial.distance import pdist, squareform

warnings.filterwarnings("ignore", category=RuntimeWarning)

T0 = time.time()


def current_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1_000_000.0


def log(message: str) -> None:
    elapsed = time.time() - T0
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] +{elapsed:8.1f}s rss={current_rss_gb():7.2f}GB {message}", flush=True)


def effective_n_jobs(n_jobs: int | None) -> int:
    if n_jobs is None:
        return 1
    return max(1, int(n_jobs))


# Paths
ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TRACE_ROOT = ROOT / "test"
DEFAULT_BEHAV_ROOT = ROOT / "behavioral_data_block"
DEFAULT_FMRI_ROOT = Path("/home/jeongmin/fmri/atari/derivatives/ants_mni")
DEFAULT_ATLAS_ROOT = (
    SCRIPT_DIR
    / "outputs"
    / "06_representational_mechanism"
    / "atlas"
    / "harvard_oxford"
    / "ants_mni_2p5mm_masks"
    / "masks"
)
DEFAULT_OUT_ROOT = SCRIPT_DIR / "outputs" / "07_encoding_analysis"


# Analysis constants
FMRI_TRIM = 60
N_ANALYSIS = 480
TR = 1.0
COUPLING_WINDOW = 11
PERM_BLOCK = 40
N_PERM = 1000
MAX_PCA_DIM = 100
RIDGE_ALPHAS = np.logspace(-2, 5, 15)
MAX_ROI_VOXELS = 4000
EPS = 1e-12

ROI_MASKS: Dict[str, str] = {
    "left_hippocampus": "subcortical/roi-subcortical-010_Left-Hippocampus_mask.nii.gz",
    "right_hippocampus": "subcortical/roi-subcortical-020_Right-Hippocampus_mask.nii.gz",
    "hippocampus": "group/roi-HarvardOxford-Hippocampus_mask.nii.gz",
    "pfc": "group/roi-HarvardOxford-PFC_mask.nii.gz",
}


# Trace discovery
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


def parse_run_label(run_label: str) -> Optional[Dict[str, int]]:
    m = re.match(r"sub(\d+)_ses(\d+)_block(\d+)_game(\d+)$", run_label)
    if not m:
        return None
    return {
        "subject": int(m.group(1)),
        "session": int(m.group(2)),
        "block": int(m.group(3)),
        "game": int(m.group(4)),
    }


def _parse_trace_meta(path: Path) -> Optional[TraceMeta]:
    m = re.match(r"sub(\d+)-ses(\d+)-block(\d+)-game(\d+)_(\d+)\.npy$", path.name)
    if m:
        return TraceMeta(
            subject=int(m.group(1)),
            session=int(m.group(2)),
            block=int(m.group(3)),
            game=int(m.group(4)),
            chunk=int(m.group(5)),
            path=path,
        )

    mf = re.match(r"video_stat_(\d+)\.npy$", path.name)
    mp = re.match(r"sub(\d+)-ses(\d+)-block(\d+)-game(\d+)$", path.parent.name)
    if mf and mp:
        return TraceMeta(
            subject=int(mp.group(1)),
            session=int(mp.group(2)),
            block=int(mp.group(3)),
            game=int(mp.group(4)),
            chunk=int(mf.group(1)),
            path=path,
        )
    return None


def gather_trace_blocks(
    trace_root: Path,
    subject: int,
    game: int,
    sessions: Optional[set[int]],
) -> Dict[Tuple[int, int, int, int], List[TraceMeta]]:
    blocks: Dict[Tuple[int, int, int, int], List[TraceMeta]] = {}
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


# Array utilities
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
    try:
        arr = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if arr.size == 0:
        return None
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def zscore_columns(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    mean = np.nanmean(arr, axis=0, keepdims=True)
    std = np.nanstd(arr, axis=0, keepdims=True)
    out = (arr - mean) / np.where(std < EPS, 1.0, std)
    return np.nan_to_num(out).astype(np.float32)


def zscore_columns_float32(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    mean = np.nanmean(arr, axis=0, keepdims=True).astype(np.float32, copy=False)
    std = np.nanstd(arr, axis=0, keepdims=True).astype(np.float32, copy=False)
    denom = np.where(std < EPS, np.float32(1.0), std).astype(np.float32, copy=False)
    arr -= mean
    arr /= denom
    return np.nan_to_num(arr, copy=False).astype(np.float32, copy=False)


def _merge_chunk_tree(carry_tree: dict, chunk_tree: dict) -> dict:
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
    parts: List[np.ndarray] = []
    for idx in indices:
        vec = getter(int(idx))
        if vec is not None and vec.size > 0:
            parts.append(np.asarray(vec, dtype=np.float32).reshape(-1))
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts).astype(np.float32, copy=False)


def _collect_imag_indices(status: np.ndarray, real_idx: int, prev_real_idx: int, mode: str) -> np.ndarray:
    start = prev_real_idx + 1 if prev_real_idx >= 0 else 0
    between = np.arange(start, real_idx, dtype=int)
    if between.size == 0:
        return between
    if mode == "primary":
        return between[status[between] != 0]
    if mode == "s2_only":
        return between[status[between] == 2]
    return between[status[between] != 0]


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
    for pos in range(start_pos, len(real_idx)):
        ridx = int(real_idx[pos])
        prev_ridx = int(real_idx[pos - 1]) if pos > 0 else -1
        imag = _collect_imag_indices(status_buf, ridx, prev_ridx, mode)
        valid = imag[(imag >= 0) & (imag < t)]

        if rep_name == "tree_reps":
            if not tree_arrays:
                continue
            rows.append(_concat_indexed_features(valid, lambda i: _flatten_tree_step(tree_arrays, i)))
        elif rep_name == "im_vp_vectors":
            rows.append(_concat_indexed_features(
                valid,
                lambda i: _flatten_raw_step(imvp_buf[i]) if 0 <= i < len(imvp_buf) else None,
            ))
        elif rep_name == "im_vectors":
            rows.append(_concat_indexed_features(
                valid,
                lambda i: _flatten_raw_step(imvec_buf[i]) if 0 <= i < len(imvec_buf) else None,
            ))
        else:
            raise ValueError(f"Unknown Thinker representation: {rep_name}")


def _stack_pad_pca_rows(rows: List[np.ndarray], label: str, max_pca_dim: int = MAX_PCA_DIM) -> Optional[np.ndarray]:
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
            from sklearn.decomposition import PCA

            n_comp = min(max_pca_dim, mat.shape[0] - 1, mat.shape[1])
            if n_comp >= 1:
                log(f"    [trace:{label}] PCA {mat.shape[1]} -> {n_comp}")
                zmat = zscore_columns_float32(mat)
                mat = PCA(n_components=n_comp, random_state=0, svd_solver="randomized").fit_transform(zmat)
                mat = mat.astype(np.float32, copy=False)
                del zmat
                gc.collect()
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
            real_idx,
            merged_status,
            merged_imvp,
            merged_imvec,
            tree_arrays,
            t,
            mode,
            rep_name,
            rows,
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
            f"real={len(real_idx)} added_rows={len(rows) - before_rows} "
            f"total_rows={len(rows)} carry={len(carry_status)}"
        )

    mat = _stack_pad_pca_rows(rows, trace_label, max_pca_dim=max_pca_dim)
    del rows, carry_status, carry_imvp, carry_imvec, carry_tree
    gc.collect()
    if mat is not None:
        log(f"    [trace:{trace_label}] done shape={mat.shape}")
    return mat


# Behavioral and HRF utilities
def find_behavioral_file(behav_root: Path, subject: int, session: int, block: int, game: int) -> Optional[Path]:
    sub_str = f"sub-{subject:03d}"
    ses_str = f"ses-{session:02d}"
    fname = f"sub{subject:03d}-ses{session:02d}-block{block}-game{game}.npz"
    p = behav_root / sub_str / ses_str / fname
    return p if p.exists() else None


def load_ram_features(behav_path: Path) -> Optional[np.ndarray]:
    try:
        d = np.load(str(behav_path), allow_pickle=True)
        return np.asarray(d["RAM"], dtype=np.float32)
    except Exception as exc:
        log(f"  [warn] RAM load failed {behav_path}: {exc}")
        return None


def canonical_hrf(tr: float = 1.0, t_max: float = 32.0) -> np.ndarray:
    from scipy.stats import gamma as gamma_dist

    tr = max(float(tr), 1e-3)
    t = np.arange(0, t_max, tr)
    h = gamma_dist.pdf(t, 6, scale=1.0) - gamma_dist.pdf(t, 16, scale=1.0) / 6.0
    denom = np.sum(h)
    if abs(denom) < EPS:
        denom = np.sum(np.abs(h)) + EPS
    return (h / denom).astype(np.float64)


def convolve_hrf_columns(x: np.ndarray, hrf: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    t_len = arr.shape[0]
    if t_len == 0 or arr.shape[1] == 0:
        return arr.astype(np.float32)
    conv = scipy_signal.fftconvolve(arr, np.asarray(hrf, dtype=np.float64)[:, None], mode="full", axes=0)
    return conv[:t_len].astype(np.float32)


def _trim_uniform_samples(
    x: np.ndarray,
    source_duration_s: float,
    trim_start_s: float,
    trim_end_s: float,
) -> Tuple[np.ndarray, float]:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    n = arr.shape[0]
    if n == 0:
        return arr, 0.0
    duration = float(source_duration_s) if source_duration_s > 0 else float(n)
    lo_t = max(0.0, float(trim_start_s))
    hi_t = max(lo_t, duration - max(0.0, float(trim_end_s)))
    lo = int(np.floor((lo_t / duration) * n))
    hi = int(np.ceil((hi_t / duration) * n))
    lo = max(0, min(n, lo))
    hi = max(lo, min(n, hi))
    return arr[lo:hi], max(0.0, hi_t - lo_t)


def bin_average_to_tr(x: np.ndarray, source_duration_s: float, n_tr: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    n, d = arr.shape
    out = np.zeros((n_tr, d), dtype=np.float64)
    counts = np.zeros(n_tr, dtype=np.float64)
    if n == 0 or d == 0 or n_tr <= 0:
        return out.astype(np.float32)
    duration = float(source_duration_s) if source_duration_s > 0 else float(n_tr)
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
    trimmed, duration = _trim_uniform_samples(x, source_duration_s, trim_start_s, trim_end_s)
    if trimmed.shape[0] == 0 or trimmed.shape[1] == 0 or duration <= 0:
        return None
    sample_dt = duration / max(trimmed.shape[0], 1)
    hrf = canonical_hrf(sample_dt)
    conv = convolve_hrf_columns(trimmed, hrf)
    return bin_average_to_tr(conv, duration, n_tr)


def build_ram_tr_features(behav_path: Path, n_tr_full: int, n_tr_use: int) -> Optional[np.ndarray]:
    ram = load_ram_features(behav_path)
    if ram is None or ram.shape[0] == 0:
        return None
    ram_tr = hrf_convolve_uniform_to_tr(
        ram.astype(np.float32),
        source_duration_s=float(n_tr_full),
        n_tr=n_tr_use,
        trim_start_s=FMRI_TRIM,
        trim_end_s=FMRI_TRIM,
    )
    if ram_tr is None:
        return None
    return zscore_columns(ram_tr)


# fMRI and ROI loading
def find_fmri_run(
    fmri_root: Path,
    fmri_subject: str,
    block: int,
    allow_smoothed_fallback: bool = False,
) -> Optional[Path]:
    session_dir = fmri_root / fmri_subject / f"Session{block}"
    p = session_dir / "wfiltered_func_data.nii"
    if p.exists():
        return p
    if allow_smoothed_fallback:
        p = session_dir / "s5_wfiltered_func_data.nii"
        if p.exists():
            return p
    return None


def load_roi_mask_nifti(mask_path: Path, ref_shape: Tuple[int, ...]) -> np.ndarray:
    import nibabel as nib

    img = nib.load(str(mask_path))
    data = np.squeeze(np.asarray(img.get_fdata(dtype=np.float32)))
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
    seed: int = 42,
) -> Optional[np.ndarray]:
    import nibabel as nib

    img = nib.load(str(fmri_path))
    coords = np.column_stack(np.where(mask)).astype(np.int32)
    if coords.size == 0:
        return None
    if max_voxels > 0 and len(coords) > max_voxels:
        rng = np.random.default_rng(seed)
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
    return zscore_columns(patterns)


# DSM and RSA
def build_dsm(x: np.ndarray) -> np.ndarray:
    arr = zscore_columns(np.asarray(x, dtype=np.float64))
    dist = pdist(arr, metric="correlation")
    dist = np.nan_to_num(dist, nan=0.0, posinf=2.0, neginf=0.0)
    return dist.astype(np.float32)


def build_coupling_dsm(hipp_patterns: np.ndarray, pfc_patterns: np.ndarray, window: int = COUPLING_WINDOW) -> np.ndarray:
    hipp_mean = np.nanmean(hipp_patterns, axis=1)
    pfc_mean = np.nanmean(pfc_patterns, axis=1)
    t_len = len(hipp_mean)
    half = window // 2
    coupling = np.full(t_len, np.nan, dtype=np.float64)
    for t in range(t_len):
        lo = max(0, t - half)
        hi = min(t_len, t + half + 1)
        h = hipp_mean[lo:hi]
        p = pfc_mean[lo:hi]
        if len(h) >= 3 and np.std(h) > EPS and np.std(p) > EPS:
            coupling[t] = np.corrcoef(h, p)[0, 1]
    coupling = np.nan_to_num(coupling, nan=0.0)
    diff = np.abs(coupling[:, None] - coupling[None, :])
    idx = np.triu_indices(t_len, k=1)
    return diff[idx].astype(np.float32)


def build_temporal_lag_dsm(n_tr: int) -> np.ndarray:
    t = np.arange(n_tr, dtype=np.float32)
    diff = np.abs(t[:, None] - t[None, :])
    idx = np.triu_indices(n_tr, k=1)
    return diff[idx].astype(np.float32)


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


def partial_spearman(a: np.ndarray, b: np.ndarray, controls: List[np.ndarray]) -> Tuple[float, float]:
    if not controls:
        return spearman_rsa(a, b)
    aa = np.asarray(a).reshape(-1)
    bb = np.asarray(b).reshape(-1)
    n = min(len(aa), len(bb), *(len(c) for c in controls))
    aa, bb = aa[:n], bb[:n]
    ctrl = np.vstack([np.asarray(c).reshape(-1)[:n] for c in controls]).T
    valid = np.isfinite(aa) & np.isfinite(bb) & np.all(np.isfinite(ctrl), axis=1)
    aa, bb, ctrl = aa[valid], bb[valid], ctrl[valid]
    if len(aa) < 5:
        return np.nan, np.nan
    ra = scipy_stats.rankdata(aa)
    rb = scipy_stats.rankdata(bb)
    rc = np.column_stack([scipy_stats.rankdata(ctrl[:, k]) for k in range(ctrl.shape[1])])
    from numpy.linalg import lstsq

    x = np.column_stack([np.ones(len(ra)), rc])
    res_a = ra - x @ lstsq(x, ra, rcond=None)[0]
    res_b = rb - x @ lstsq(x, rb, rcond=None)[0]
    if np.std(res_a) < EPS or np.std(res_b) < EPS:
        return np.nan, np.nan
    r, p = scipy_stats.pearsonr(res_a, res_b)
    return float(r), float(p)


def fdr_bh(pvals: np.ndarray) -> np.ndarray:
    pv = np.asarray(pvals, dtype=float)
    finite = np.isfinite(pv)
    q = np.full_like(pv, np.nan)
    n = int(finite.sum())
    if n == 0:
        return q
    idx = np.where(finite)[0]
    order = np.argsort(pv[idx])
    ranked = pv[idx][order]
    q_ordered = ranked * n / np.arange(1, n + 1, dtype=float)
    q_ordered = np.minimum.accumulate(q_ordered[::-1])[::-1]
    q_ordered = np.minimum(q_ordered, 1.0)
    q[idx[order]] = q_ordered
    return q


def block_permutation_rsa(
    model_dsm: np.ndarray,
    brain_patterns: np.ndarray,
    n_tr: int,
    block_size: int,
    n_perm: int,
    rng_seed: int,
    label: str = "",
) -> Tuple[np.ndarray, float, float]:
    rng = np.random.default_rng(rng_seed)
    n_blocks = n_tr // block_size
    if n_blocks < 2:
        return np.array([], dtype=np.float32), np.nan, np.nan
    block_indices = [np.arange(i * block_size, min((i + 1) * block_size, n_tr)) for i in range(n_blocks)]
    if len(block_indices[-1]) < block_size:
        block_indices = block_indices[:-1]
        n_blocks = len(block_indices)
    if n_blocks < 2:
        return np.array([], dtype=np.float32), np.nan, np.nan

    obs_r, _ = spearman_rsa(model_dsm, build_dsm(brain_patterns))
    if not np.isfinite(obs_r):
        return np.array([], dtype=np.float32), np.nan, np.nan

    nulls = np.empty(n_perm, dtype=np.float32)
    progress_step = max(1, n_perm // 4)
    for i in range(n_perm):
        perm_order = rng.permutation(n_blocks)
        perm_idx = np.concatenate([block_indices[b] for b in perm_order])
        r_perm, _ = spearman_rsa(model_dsm, build_dsm(brain_patterns[perm_idx]))
        nulls[i] = r_perm if np.isfinite(r_perm) else 0.0
        if label and ((i + 1) % progress_step == 0 or i + 1 == n_perm):
            log(f"    [rsa perm:{label}] {i + 1}/{n_perm}")
    p_one = float((1 + np.sum(nulls >= obs_r)) / (1 + n_perm))
    p_two = float((1 + np.sum(np.abs(nulls) >= abs(obs_r))) / (1 + n_perm))
    return nulls, p_one, p_two


def block_permutation_coupling_rsa(
    model_dsm: np.ndarray,
    hipp_patterns: np.ndarray,
    pfc_patterns: np.ndarray,
    n_tr: int,
    block_size: int,
    n_perm: int,
    rng_seed: int,
    label: str = "",
) -> Tuple[np.ndarray, float, float]:
    rng = np.random.default_rng(rng_seed)
    n_blocks = n_tr // block_size
    if n_blocks < 2:
        return np.array([], dtype=np.float32), np.nan, np.nan
    block_indices = [np.arange(i * block_size, min((i + 1) * block_size, n_tr)) for i in range(n_blocks)]
    obs_dsm = build_coupling_dsm(hipp_patterns[:n_tr], pfc_patterns[:n_tr])
    obs_r, _ = spearman_rsa(model_dsm, obs_dsm)
    if not np.isfinite(obs_r):
        return np.array([], dtype=np.float32), np.nan, np.nan
    nulls = np.empty(n_perm, dtype=np.float32)
    progress_step = max(1, n_perm // 4)
    for i in range(n_perm):
        perm_order = rng.permutation(n_blocks)
        perm_idx = np.concatenate([block_indices[b] for b in perm_order])
        r_perm, _ = spearman_rsa(model_dsm, build_coupling_dsm(hipp_patterns[perm_idx], pfc_patterns[perm_idx]))
        nulls[i] = r_perm if np.isfinite(r_perm) else 0.0
        if label and ((i + 1) % progress_step == 0 or i + 1 == n_perm):
            log(f"    [rsa perm:{label}] {i + 1}/{n_perm}")
    p_one = float((1 + np.sum(nulls >= obs_r)) / (1 + n_perm))
    p_two = float((1 + np.sum(np.abs(nulls) >= abs(obs_r))) / (1 + n_perm))
    return nulls, p_one, p_two


# LORO encoding
def pearson_columns(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    yt = np.asarray(Y_true, dtype=np.float64)
    yp = np.asarray(Y_pred, dtype=np.float64)
    if yt.ndim == 1:
        yt = yt[:, None]
    if yp.ndim == 1:
        yp = yp[:, None]
    n_cols = min(yt.shape[1], yp.shape[1])
    yt = yt[:, :n_cols]
    yp = yp[:, :n_cols]
    yt = yt - np.nanmean(yt, axis=0, keepdims=True)
    yp = yp - np.nanmean(yp, axis=0, keepdims=True)
    num = np.nansum(yt * yp, axis=0)
    den = np.sqrt(np.nansum(yt * yt, axis=0) * np.nansum(yp * yp, axis=0))
    rs = np.full(n_cols, np.nan, dtype=np.float32)
    ok = den > EPS
    rs[ok] = (num[ok] / den[ok]).astype(np.float32)
    return rs


def _preprocess_and_fit(X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, alpha: float) -> np.ndarray:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    sx = StandardScaler()
    xtr_s = sx.fit_transform(X_train)
    xte_s = sx.transform(X_test)
    n_comp = min(MAX_PCA_DIM, xtr_s.shape[1], xtr_s.shape[0] - 1)
    if n_comp < 1:
        n_out = Y_train.shape[1] if Y_train.ndim > 1 else 1
        return np.zeros((X_test.shape[0], n_out), dtype=np.float32)
    pca = PCA(n_components=n_comp, random_state=0)
    xtr_p = pca.fit_transform(xtr_s)
    xte_p = pca.transform(xte_s)
    ridge = Ridge(alpha=alpha, fit_intercept=True)
    ridge.fit(xtr_p, Y_train)
    return ridge.predict(xte_p).astype(np.float32)


def _loro_with_alpha(X_runs: List[np.ndarray], Y_runs: List[np.ndarray], alpha: float) -> np.ndarray:
    n_voxels = Y_runs[0].shape[1]
    voxel_rs = np.zeros(n_voxels, dtype=np.float64)
    counts = np.zeros(n_voxels, dtype=int)
    for test_i in range(len(X_runs)):
        X_test = X_runs[test_i]
        Y_test = Y_runs[test_i]
        X_train = np.vstack([x for i, x in enumerate(X_runs) if i != test_i])
        Y_train = np.vstack([y for i, y in enumerate(Y_runs) if i != test_i])
        if X_train.shape[0] < 5 or X_test.shape[0] < 2:
            continue
        Y_pred = _preprocess_and_fit(X_train, Y_train, X_test, alpha)
        rs = pearson_columns(Y_test, Y_pred)
        valid = np.isfinite(rs)
        voxel_rs[valid] += rs[valid]
        counts[valid] += 1
    result = np.full(n_voxels, np.nan, dtype=np.float32)
    ok = counts > 0
    result[ok] = (voxel_rs[ok] / counts[ok]).astype(np.float32)
    return result


def run_loro_encoding(
    X_runs: List[np.ndarray],
    Y_runs: List[np.ndarray],
    alphas: np.ndarray,
    label: str = "",
) -> Tuple[np.ndarray, float]:
    if len(X_runs) < 2:
        return np.array([], dtype=np.float32), float("nan")
    best_alpha = float(alphas[0])
    best_score = -np.inf
    best_rs: Optional[np.ndarray] = None
    for alpha in alphas:
        rs = _loro_with_alpha(X_runs, Y_runs, float(alpha))
        score = float(np.nanmean(rs))
        if label:
            log(f"    [alpha={alpha:.3g}] mean_r={score:.5f}")
        if score > best_score:
            best_score = score
            best_alpha = float(alpha)
            best_rs = rs
    if label:
        log(f"  [{label}] best_alpha={best_alpha:.3g} best_mean_r={best_score:.5f}")
    if best_rs is None:
        return np.array([], dtype=np.float32), float("nan")
    return best_rs, best_alpha


def block_permutation_indices(T: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    if T <= 0:
        return np.array([], dtype=int)
    block_size = max(1, int(block_size))
    blocks = [np.arange(lo, min(lo + block_size, T), dtype=int) for lo in range(0, T, block_size)]
    if len(blocks) < 2:
        return np.arange(T, dtype=int)
    order = rng.permutation(len(blocks))
    return np.concatenate([blocks[i] for i in order])


def _loro_predictions(
    X_runs: List[np.ndarray],
    Y_runs: List[np.ndarray],
    alpha: float,
    run_labels: Optional[List[str]] = None,
    label: str = "",
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], List[Dict]]:
    n_voxels = Y_runs[0].shape[1]
    voxel_rs = np.zeros(n_voxels, dtype=np.float64)
    counts = np.zeros(n_voxels, dtype=int)
    y_tests: List[np.ndarray] = []
    y_preds: List[np.ndarray] = []
    fold_rows: List[Dict] = []
    for test_i in range(len(X_runs)):
        X_test = X_runs[test_i]
        Y_test = Y_runs[test_i]
        X_train = np.vstack([x for i, x in enumerate(X_runs) if i != test_i])
        Y_train = np.vstack([y for i, y in enumerate(Y_runs) if i != test_i])
        if X_train.shape[0] < 5 or X_test.shape[0] < 2:
            continue
        if label:
            log(f"    [best-alpha fold] {test_i + 1}/{len(X_runs)} train={X_train.shape} test={X_test.shape}")
        Y_pred = _preprocess_and_fit(X_train, Y_train, X_test, alpha)
        rs = pearson_columns(Y_test, Y_pred)
        valid = np.isfinite(rs)
        voxel_rs[valid] += rs[valid]
        counts[valid] += 1
        y_tests.append(Y_test)
        y_preds.append(Y_pred)
        fold_rows.append({
            "heldout_run": run_labels[test_i] if run_labels else str(test_i),
            "fold_index": test_i,
            "mean_r": float(np.nanmean(rs)),
            "median_r": float(np.nanmedian(rs)),
            "n_voxels": int(len(rs)),
            "n_voxels_finite": int(np.isfinite(rs).sum()),
            "n_voxels_positive": int(np.sum(rs > 0)),
        })
    result = np.full(n_voxels, np.nan, dtype=np.float32)
    ok = counts > 0
    result[ok] = (voxel_rs[ok] / counts[ok]).astype(np.float32)
    return result, y_tests, y_preds, fold_rows


def run_loro_permutation(
    X_runs: List[np.ndarray],
    Y_runs: List[np.ndarray],
    alpha: float,
    n_perm: int,
    block_size: int,
    rng_seed: int,
    run_labels: Optional[List[str]] = None,
    label: str = "",
) -> Dict[str, np.ndarray | List[Dict]]:
    rs, y_tests, y_preds, fold_rows = _loro_predictions(X_runs, Y_runs, alpha, run_labels=run_labels, label=label)
    n_voxels = len(rs)
    out: Dict[str, np.ndarray | List[Dict]] = {"rs": rs, "fold_rows": fold_rows}
    if n_perm <= 0 or not y_tests:
        return out

    rng = np.random.default_rng(rng_seed)
    ge_counts = np.zeros(n_voxels, dtype=np.int32)
    abs_counts = np.zeros(n_voxels, dtype=np.int32)
    null_mean_scores = np.full(n_perm, np.nan, dtype=np.float32)
    obs_ok = np.isfinite(rs)
    progress_step = max(1, n_perm // 5)
    for perm_i in range(n_perm):
        null_sum = np.zeros(n_voxels, dtype=np.float64)
        null_counts = np.zeros(n_voxels, dtype=np.int16)
        for Y_test, Y_pred in zip(y_tests, y_preds):
            perm_idx = block_permutation_indices(Y_test.shape[0], block_size, rng)
            perm_rs = pearson_columns(Y_test[perm_idx], Y_pred)
            valid = np.isfinite(perm_rs)
            null_sum[valid] += perm_rs[valid]
            null_counts[valid] += 1
        null_avg = np.full(n_voxels, np.nan, dtype=np.float32)
        valid = null_counts > 0
        null_avg[valid] = (null_sum[valid] / null_counts[valid]).astype(np.float32)
        valid = valid & obs_ok
        ge_counts[valid] += null_avg[valid] >= rs[valid]
        abs_counts[valid] += np.abs(null_avg[valid]) >= np.abs(rs[valid])
        null_mean_scores[perm_i] = float(np.nanmean(null_avg))
        if label and ((perm_i + 1) % progress_step == 0 or perm_i + 1 == n_perm):
            log(f"    [enc perm:{label}] {perm_i + 1}/{n_perm}")

    p_one = np.full(n_voxels, np.nan, dtype=np.float32)
    p_two = np.full(n_voxels, np.nan, dtype=np.float32)
    p_one[obs_ok] = (1.0 + ge_counts[obs_ok]) / (1.0 + n_perm)
    p_two[obs_ok] = (1.0 + abs_counts[obs_ok]) / (1.0 + n_perm)
    q_one = fdr_bh(p_one).astype(np.float32)
    out.update({
        "p_one_sided": p_one,
        "p_two_sided": p_two,
        "q_fdr_one_sided": q_one,
        "null_mean_scores": null_mean_scores,
    })
    return out


def finite_stat(arr: Optional[np.ndarray], fn, default: float = float("nan")) -> float:
    if arr is None:
        return default
    vals = np.asarray(arr, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return default
    return float(fn(vals))


def build_model_map(result: Dict) -> Dict[str, np.ndarray]:
    models: Dict[str, np.ndarray] = {}
    ram_tr = result.get("ram_tr")
    features = result.get("features", {})
    if isinstance(ram_tr, np.ndarray):
        models["ram"] = ram_tr
    for feat_name, x in features.items():
        if not isinstance(x, np.ndarray):
            continue
        models[f"thinker_{feat_name}"] = x
        if isinstance(ram_tr, np.ndarray):
            min_t = min(ram_tr.shape[0], x.shape[0])
            models[f"ram_plus_{feat_name}"] = np.hstack([ram_tr[:min_t], x[:min_t]])
    return models


# Plotting helpers
def _dsm_to_square(condensed: np.ndarray) -> np.ndarray:
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
        try:
            sq = _dsm_to_square(dsms[key])
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
        if not obs_rows.empty:
            parts = key.split("_vs_bold_")
            if len(parts) == 2:
                model_name, roi_name = parts
                rows = obs_rows[(obs_rows["model"] == model_name) & (obs_rows["roi"] == roi_name)]
                if not rows.empty:
                    obs = float(rows["rho_obs"].iloc[0])
                    ax.axvline(obs, color="#ef4444", lw=1.5, label=f"obs={obs:.3f}")
                    ax.legend(fontsize=7)
    for j in range(len(keys), len(axes_flat)):
        axes_flat[j].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def model_label(name: str) -> str:
    if name == "ram":
        return "RAM"
    prefix = ""
    rest = name
    if name.startswith("ram_plus_"):
        prefix = "RAM + "
        rest = name[len("ram_plus_"):]
    elif name.startswith("thinker_"):
        prefix = "Thinker "
        rest = name[len("thinker_"):]
    rest = (
        rest.replace("tree_reps", "tree")
        .replace("im_vp_vectors", "im-vp")
        .replace("im_vectors", "im")
        .replace("_primary", " primary")
        .replace("_s2only", " s2-only")
        .replace("_", " ")
    )
    return prefix + rest


def plot_heatmap_table(
    pivot: pd.DataFrame,
    out_path: Path,
    title: str,
    cbar_label: str,
    cmap: str = "RdBu_r",
    center_zero: bool = False,
    fmt: str = ".3f",
) -> None:
    data = pivot.to_numpy(dtype=float)
    if center_zero and np.isfinite(data).any():
        vmax = float(np.nanmax(np.abs(data)))
        vmin = -vmax
    else:
        vmin = float(np.nanmin(data)) if np.isfinite(data).any() else 0.0
        vmax = float(np.nanmax(data)) if np.isfinite(data).any() else 1.0
        if abs(vmax - vmin) < EPS:
            vmax = vmin + 1.0
    fig, ax = plt.subplots(figsize=(1.35 * len(pivot.columns) + 5.8, 0.36 * len(pivot.index) + 1.8))
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(list(pivot.columns), rotation=25, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(list(pivot.index))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:{fmt}}", ha="center", va="center", fontsize=7, color="black")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03, label=cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def generate_loro_plots(enc_df: pd.DataFrame, out_dir: Path) -> None:
    if enc_df.empty:
        return
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    roi_order = [r for r in ["left_hippocampus", "right_hippocampus", "hippocampus", "pfc"] if r in set(enc_df["roi"])]
    roi_order += [r for r in sorted(enc_df["roi"].unique()) if r not in roi_order]
    roi_labels = {
        "left_hippocampus": "Left hippocampus",
        "right_hippocampus": "Right hippocampus",
        "hippocampus": "Hippocampus",
        "pfc": "PFC",
    }
    roi_display = {r: roi_labels.get(r, r) for r in roi_order}
    model_order = [
        "ram",
        "thinker_tree_reps_primary",
        "thinker_tree_reps_s2only",
        "thinker_im_vectors_primary",
        "thinker_im_vectors_s2only",
        "thinker_im_vp_vectors_primary",
        "thinker_im_vp_vectors_s2only",
        "ram_plus_tree_reps_primary",
        "ram_plus_tree_reps_s2only",
        "ram_plus_im_vectors_primary",
        "ram_plus_im_vectors_s2only",
        "ram_plus_im_vp_vectors_primary",
        "ram_plus_im_vp_vectors_s2only",
    ]
    model_order = [m for m in model_order if m in set(enc_df["model"])]
    model_order += [m for m in sorted(enc_df["model"].unique()) if m not in model_order]
    model_display = {m: model_label(m) for m in model_order}

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 140,
        "savefig.dpi": 220,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    created: List[Path] = []

    def save(fig, filename: str) -> None:
        path = plot_dir / filename
        fig.savefig(path, bbox_inches="tight")
        created.append(path)

    fig, axes = plt.subplots(1, len(roi_order), figsize=(4.2 * len(roi_order), 7.2), sharex=True)
    if len(roi_order) == 1:
        axes = [axes]
    for ax, roi in zip(axes, roi_order):
        sub = enc_df[enc_df["roi"] == roi].set_index("model").reindex(model_order).dropna(subset=["mean_r"])
        vals = sub["mean_r"].to_numpy()
        labels = [model_display[m] for m in sub.index]
        colors = [
            "#4c78a8" if m == "ram" else "#72b7b2" if m.startswith("thinker_") else "#f58518"
            for m in sub.index
        ]
        y = np.arange(len(sub))
        ax.barh(y, vals, color=colors, alpha=0.9)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(roi_display[roi])
        ax.set_yticks(y)
        ax.set_yticklabels(labels if ax is axes[0] else [])
        ax.invert_yaxis()
        ax.grid(axis="x", color="#dddddd", linewidth=0.6)
        ax.set_xlabel("Mean held-out Pearson r")
    fig.suptitle("Leave-one-run-out encoding performance", y=1.02, fontsize=13)
    fig.tight_layout()
    save(fig, "encoding_loro_mean_r_by_roi.png")
    plt.close(fig)

    pivot = enc_df.pivot(index="model", columns="roi", values="mean_r").reindex(index=model_order, columns=roi_order)
    pivot = pivot.rename(index=model_display, columns=roi_display)
    path = plot_dir / "encoding_loro_mean_r_heatmap.png"
    plot_heatmap_table(pivot, path, "Mean Pearson r", "mean r", center_zero=True, fmt=".3f")
    created.append(path)

    ram = enc_df[enc_df["model"] == "ram"].set_index("roi")["mean_r"]
    delta_rows = []
    for _, row in enc_df[enc_df["model"].str.startswith("ram_plus_")].iterrows():
        if row["roi"] in ram.index:
            delta_rows.append({
                "roi": row["roi"],
                "model": row["model"].replace("ram_plus_", ""),
                "delta_mean_r": row["mean_r"] - ram.loc[row["roi"]],
            })
    delta_df = pd.DataFrame(delta_rows)
    if not delta_df.empty:
        delta_models = [m.replace("ram_plus_", "") for m in model_order if m.startswith("ram_plus_")]
        pivot = delta_df.pivot(index="model", columns="roi", values="delta_mean_r").reindex(
            index=delta_models, columns=roi_order
        )
        pivot = pivot.rename(index={m: model_label("thinker_" + m).replace("Thinker ", "") for m in delta_models},
                             columns=roi_display)
        path = plot_dir / "encoding_loro_delta_over_ram_heatmap.png"
        plot_heatmap_table(pivot, path, "Incremental encoding over RAM", "delta mean r",
                           cmap="PiYG", center_zero=True, fmt="+.3f")
        created.append(path)

    if {"n_voxels_positive", "n_voxels"}.issubset(enc_df.columns):
        tmp = enc_df.copy()
        tmp["positive_fraction"] = tmp["n_voxels_positive"] / tmp["n_voxels"]
        pivot = tmp.pivot(index="model", columns="roi", values="positive_fraction").reindex(
            index=model_order, columns=roi_order
        )
        pivot = pivot.rename(index=model_display, columns=roi_display)
        path = plot_dir / "encoding_loro_positive_voxel_fraction.png"
        plot_heatmap_table(pivot, path, "Fraction of voxels with positive r", "positive fraction",
                           cmap="viridis", center_zero=False, fmt=".2f")
        created.append(path)

    if "best_alpha" in enc_df.columns:
        pivot = enc_df.pivot(index="model", columns="roi", values="best_alpha").reindex(index=model_order, columns=roi_order)
        pivot = np.log10(pivot.astype(float)).rename(index=model_display, columns=roi_display)
        path = plot_dir / "encoding_loro_best_alpha_heatmap.png"
        plot_heatmap_table(pivot, path, "Selected ridge alpha", "log10 alpha",
                           cmap="magma", center_zero=False, fmt=".1f")
        created.append(path)

    optional_specs = [
        ("frac_voxels_q05", "encoding_loro_fdr_q05_fraction.png", "Fraction of voxels passing FDR q < 0.05",
         "q < 0.05 fraction", "viridis", False, ".2f"),
        ("frac_voxels_p05", "encoding_loro_perm_p05_fraction.png", "Fraction of voxels with permutation p < 0.05",
         "p < 0.05 fraction", "viridis", False, ".2f"),
        ("mean_r_minus_null95_mean_r", "encoding_loro_mean_r_vs_null95_heatmap.png",
         "Observed mean r minus permutation 95th percentile", "mean r - null95", "PiYG", True, "+.3f"),
    ]
    for col, filename, title, label, cmap, center, fmt in optional_specs:
        if col in enc_df.columns and enc_df[col].notna().any():
            pivot = enc_df.pivot(index="model", columns="roi", values=col).reindex(index=model_order, columns=roi_order)
            pivot = pivot.rename(index=model_display, columns=roi_display)
            path = plot_dir / filename
            plot_heatmap_table(pivot, path, title, label, cmap=cmap, center_zero=center, fmt=fmt)
            created.append(path)

    if "min_q_fdr_one_sided" in enc_df.columns and enc_df["min_q_fdr_one_sided"].notna().any():
        tmp = enc_df.copy()
        tmp["neglog10_min_q"] = -np.log10(np.clip(tmp["min_q_fdr_one_sided"].astype(float), 1e-300, 1.0))
        pivot = tmp.pivot(index="model", columns="roi", values="neglog10_min_q").reindex(index=model_order, columns=roi_order)
        pivot = pivot.rename(index=model_display, columns=roi_display)
        path = plot_dir / "encoding_loro_min_q_neglog10_heatmap.png"
        plot_heatmap_table(pivot, path, "Strongest voxel evidence: -log10(min FDR q)", "-log10 min q",
                           cmap="magma", center_zero=False, fmt=".1f")
        created.append(path)

    if created:
        pdf_path = plot_dir / "encoding_loro_summary_plots.pdf"
        with PdfPages(pdf_path) as pdf:
            for png in created:
                img = plt.imread(png)
                fig_w = min(16, max(8, img.shape[1] / 180))
                fig_h = min(12, max(5, img.shape[0] / 180))
                fig, ax = plt.subplots(figsize=(fig_w, fig_h))
                ax.imshow(img)
                ax.axis("off")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
        created.append(pdf_path)
    for path in created:
        log(f"[plot] {path}")


def generate_paper_figures(
    perm_df: pd.DataFrame,
    enc_df: pd.DataFrame,
    enc_fold_df: pd.DataFrame,
    out_dir: Path,
    label: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if perm_df.empty:
        log("[paper figures] missing RSA permutation rows; skipping RSA figures")
    if enc_df.empty:
        log("[paper figures] missing LORO encoding rows; skipping encoding figure")

    run_labels = sorted(perm_df["run_label"].dropna().unique()) if not perm_df.empty else []
    run_short = [r.replace("sub001_", "").replace("_game2", "") for r in run_labels]
    present_models = set(perm_df["model"]) if not perm_df.empty else set()
    present_rois = set(perm_df["roi"]) if not perm_df.empty else set()

    primary_models = [m for m in ["tree_reps_primary", "im_vectors_primary", "im_vp_vectors_primary", "ram"] if m in present_models]
    focus_rois = [r for r in ["hippocampus", "right_hippocampus", "pfc", "coupling_hipp_pfc"] if r in present_rois]
    model_labels = {
        "tree_reps_primary": "tree_reps\n(primary)",
        "tree_reps_s2only": "tree_reps\n(s2only)",
        "im_vectors_primary": "im_vectors\n(primary)",
        "im_vectors_s2only": "im_vectors\n(s2only)",
        "im_vp_vectors_primary": "im_vp\n(primary)",
        "im_vp_vectors_s2only": "im_vp\n(s2only)",
        "ram": "RAM",
    }
    model_colors = {
        "tree_reps_primary": "#2166AC",
        "tree_reps_s2only": "#92C5DE",
        "im_vectors_primary": "#D6604D",
        "im_vectors_s2only": "#F4A582",
        "im_vp_vectors_primary": "#4DAC26",
        "im_vp_vectors_s2only": "#B8E186",
        "ram": "#7B3294",
    }
    roi_labels = {
        "hippocampus": "Hipp\n(bilateral)",
        "left_hippocampus": "Hipp\n(left)",
        "right_hippocampus": "Hipp\n(right)",
        "pfc": "PFC",
        "coupling_hipp_pfc": "Hipp-PFC\ncoupling",
    }

    def get_rhos(model: str, roi: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows_by_run = {
            r["run_label"]: r for _, r in perm_df[(perm_df["model"] == model) & (perm_df["roi"] == roi)].iterrows()
        }
        rhos = [float(rows_by_run[run]["rho_obs"]) if run in rows_by_run else np.nan for run in run_labels]
        ps = [float(rows_by_run[run]["p_one_sided"]) if run in rows_by_run else np.nan for run in run_labels]
        qs = [float(rows_by_run[run]["q_fdr_one_sided"]) if run in rows_by_run and "q_fdr_one_sided" in rows_by_run[run] else np.nan for run in run_labels]
        return np.asarray(rhos), np.asarray(ps), np.asarray(qs)

    if run_labels and primary_models and focus_rois:
        fig, axes = plt.subplots(1, len(focus_rois), figsize=(4 * len(focus_rois), 5), sharey=False)
        if len(focus_rois) == 1:
            axes = [axes]
        fig.suptitle(f"{label} - RSA rho per run", fontsize=13, fontweight="bold")
        x_pos = np.arange(len(primary_models))
        rng = np.random.default_rng(42)
        for ax, roi in zip(axes, focus_rois):
            for xi, model in enumerate(primary_models):
                rhos, _, qs = get_rhos(model, roi)
                valid = np.isfinite(rhos)
                jitter = (rng.random(valid.sum()) - 0.5) * 0.12
                colors = ["#D62728" if q < 0.05 else "#AAAAAA" for q in qs[valid]]
                ax.scatter(xi + jitter, rhos[valid], c=colors, s=45, zorder=3, edgecolors="none", alpha=0.85)
                ax.hlines(np.nanmean(rhos), xi - 0.28, xi + 0.28, colors=model_colors[model], linewidth=2.5, zorder=4)
            ax.set_title(roi_labels.get(roi, roi), fontsize=10, fontweight="bold")
            ax.set_xticks(x_pos)
            ax.set_xticklabels([model_labels.get(m, m) for m in primary_models], fontsize=8)
            ax.set_ylabel("Spearman rho", fontsize=8)
            ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
            ax.set_xlim(-0.6, len(primary_models) - 0.4)
        fig.legend(
            handles=[mpatches.Patch(color="#D62728", label="FDR q < 0.05"),
                     mpatches.Patch(color="#AAAAAA", label="n.s.")],
            loc="lower right",
            fontsize=9,
            title="Block perm",
            bbox_to_anchor=(1.0, 0.05),
        )
        fig.tight_layout(rect=[0, 0, 0.97, 1])
        fig.savefig(out_dir / "new_fig1_strip_plot.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        heatmap_models = [m for m in ["tree_reps_primary", "tree_reps_s2only", "im_vectors_primary", "im_vp_vectors_primary", "ram"] if m in present_models]
        heatmap_rois = focus_rois
        mean_rho_mat = np.zeros((len(heatmap_models), len(heatmap_rois)))
        sig_count = np.zeros_like(mean_rho_mat, dtype=int)
        for i, model in enumerate(heatmap_models):
            for j, roi in enumerate(heatmap_rois):
                rhos, _, qs = get_rhos(model, roi)
                mean_rho_mat[i, j] = np.nanmean(rhos)
                sig_count[i, j] = int(np.sum(qs < 0.05))
        fig, ax = plt.subplots(figsize=(max(7, 2.2 * len(heatmap_rois)), max(4, 0.8 * len(heatmap_models) + 2)))
        vmax = max(0.10, float(np.nanmax(mean_rho_mat)) if np.isfinite(mean_rho_mat).any() else 0.10)
        im = ax.imshow(mean_rho_mat, aspect="auto", cmap="Reds", vmin=0, vmax=vmax)
        plt.colorbar(im, ax=ax, label="Mean Spearman rho")
        ax.set_xticks(range(len(heatmap_rois)))
        ax.set_xticklabels([roi_labels.get(r, r).replace("\n", " ") for r in heatmap_rois], fontsize=10)
        ax.set_yticks(range(len(heatmap_models)))
        ax.set_yticklabels([model_labels.get(m, m).replace("\n", " ") for m in heatmap_models], fontsize=10)
        for i in range(len(heatmap_models)):
            for j in range(len(heatmap_rois)):
                val = mean_rho_mat[i, j]
                sig = sig_count[i, j]
                star = "" if sig == 0 else ("  *" if sig >= 2 else "  .")
                txt = f"{val:.3f}{star}\n({sig}/{len(run_labels)} sig)" if sig > 0 else f"{val:.3f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=9, color="white" if val > vmax * 0.65 else "black")
        ax.set_title(f"{label} - Mean RSA rho\n* = FDR q<0.05 in >=2 runs", fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / "new_fig2_heatmap_scaled.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        chi2_thresh = scipy_stats.chi2.ppf(0.95, df=2 * len(run_labels))
        fig, axes = plt.subplots(1, len(focus_rois), figsize=(4 * len(focus_rois), 4.5), sharey=True)
        if len(focus_rois) == 1:
            axes = [axes]
        fig.suptitle(
            f"{label} - Fisher combined chi2 (block-perm p-values, k={len(run_labels)} runs)\n"
            f"Threshold chi2(df={2 * len(run_labels)}) = {chi2_thresh:.1f}",
            fontsize=11,
            fontweight="bold",
        )
        for ax, roi in zip(axes, focus_rois):
            fstats = []
            for model in primary_models:
                _, ps, _ = get_rhos(model, roi)
                safe_ps = np.array([max(p, 1e-4) for p in ps if np.isfinite(p)])
                fstats.append(float(-2.0 * np.sum(np.log(safe_ps))) if safe_ps.size else np.nan)
            colors_bar = ["#D62728" if fs > chi2_thresh else "#AAAAAA" for fs in fstats]
            ax.bar(range(len(primary_models)), fstats, color=colors_bar, edgecolor="white", linewidth=0.5)
            ax.axhline(chi2_thresh, color="black", linewidth=1.5, linestyle="--")
            ax.set_title(roi_labels.get(roi, roi), fontsize=10, fontweight="bold")
            ax.set_xticks(range(len(primary_models)))
            ax.set_xticklabels([model_labels.get(m, m) for m in primary_models], fontsize=8)
            ax.set_ylabel("Fisher chi2 stat", fontsize=8)
            for xi, fs in enumerate(fstats):
                if np.isfinite(fs):
                    ax.text(xi, fs + 0.5, f"{fs:.0f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout(rect=[0, 0, 0.97, 1])
        fig.savefig(out_dir / "new_fig3_fisher_combined.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        detail_models = [m for m in ["tree_reps_primary", "tree_reps_s2only"] if m in present_models]
        detail_rois = [r for r in ["hippocampus", "right_hippocampus", "pfc"] if r in present_rois]
        if detail_models and detail_rois:
            fig, axes = plt.subplots(len(detail_models), len(detail_rois), figsize=(4.7 * len(detail_rois), 3.5 * len(detail_models)), sharey="row")
            axes = np.array(axes).reshape(len(detail_models), len(detail_rois))
            fig.suptitle(f"{label} - tree_reps rho per run", fontsize=12, fontweight="bold")
            for row_i, model in enumerate(detail_models):
                for col_j, roi in enumerate(detail_rois):
                    ax = axes[row_i][col_j]
                    rhos, ps, qs = get_rhos(model, roi)
                    colors = ["#D62728" if q < 0.05 else "#FF7F0E" if p < 0.05 else "#CCCCCC" for p, q in zip(ps, qs)]
                    x = np.arange(len(run_labels))
                    ax.bar(x, rhos, color=colors, edgecolor="white", linewidth=0.3)
                    mean_r = np.nanmean(rhos)
                    ax.axhline(mean_r, color="steelblue", linewidth=1.5, linestyle="--")
                    ax.axhline(0, color="black", linewidth=0.5)
                    ax.set_xticks(x)
                    ax.set_xticklabels(run_short, rotation=45, ha="right", fontsize=7)
                    ax.set_ylabel("rho_obs", fontsize=8)
                    ax.set_title(
                        f"{model_labels.get(model, model).replace(chr(10), ' ')} | "
                        f"{roi_labels.get(roi, roi).replace(chr(10), ' ')}\nmean={mean_r:.3f}",
                        fontsize=9,
                    )
                    for xi, (rho, q, p) in enumerate(zip(rhos, qs, ps)):
                        lbl = "*" if q < 0.05 else ("+" if p < 0.05 else "")
                        if lbl and np.isfinite(rho):
                            ax.text(xi, rho + 0.001, lbl, ha="center", va="bottom", fontsize=11, color="black", fontweight="bold")
            fig.legend(
                handles=[
                    mpatches.Patch(color="#D62728", label="FDR q < 0.05 (*)"),
                    mpatches.Patch(color="#FF7F0E", label="nominal p < 0.05 (+)"),
                    mpatches.Patch(color="#CCCCCC", label="n.s."),
                ],
                loc="lower right",
                fontsize=9,
                bbox_to_anchor=(1.0, 0.01),
            )
            fig.tight_layout(rect=[0, 0.03, 0.97, 1])
            fig.savefig(out_dir / "new_fig4_tree_reps_per_run.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    if not enc_df.empty:
        enc_models_order = [
            "thinker_tree_reps_primary",
            "thinker_im_vectors_primary",
            "thinker_im_vp_vectors_primary",
            "ram",
            "ram_plus_tree_reps_primary",
            "ram_plus_im_vectors_primary",
            "ram_plus_im_vp_vectors_primary",
        ]
        enc_models_order = [m for m in enc_models_order if m in set(enc_df["model"])]
        enc_target_rois = [r for r in ["hippocampus", "right_hippocampus", "pfc"] if r in set(enc_df["roi"])]
        if enc_models_order and enc_target_rois:
            enc_labels = {
                "thinker_tree_reps_primary": "Thinker\ntree_reps",
                "thinker_im_vectors_primary": "Thinker\nim_vectors",
                "thinker_im_vp_vectors_primary": "Thinker\nim_vp",
                "ram": "RAM\n(baseline)",
                "ram_plus_tree_reps_primary": "RAM +\ntree_reps",
                "ram_plus_im_vectors_primary": "RAM +\nim_vectors",
                "ram_plus_im_vp_vectors_primary": "RAM +\nim_vp",
            }
            enc_colors = {
                "thinker_tree_reps_primary": "#92C5DE",
                "thinker_im_vectors_primary": "#F4A582",
                "thinker_im_vp_vectors_primary": "#B8E186",
                "ram": "#7B3294",
                "ram_plus_tree_reps_primary": "#2166AC",
                "ram_plus_im_vectors_primary": "#D6604D",
                "ram_plus_im_vp_vectors_primary": "#4DAC26",
            }
            enc_target_labels = {
                "hippocampus": "Hippocampus (bilateral)",
                "right_hippocampus": "Right Hippocampus",
                "pfc": "PFC",
            }
            fig, axes = plt.subplots(1, len(enc_target_rois), figsize=(5.3 * len(enc_target_rois), 5))
            if len(enc_target_rois) == 1:
                axes = [axes]
            fig.suptitle(
                f"{label} - Voxelwise LORO encoding (Pearson r)\n"
                "Mean across held-out runs +/- 1 SE when fold rows are available",
                fontsize=12,
                fontweight="bold",
            )
            for ax, target_roi in zip(axes, enc_target_rois):
                means, stes = [], []
                for model in enc_models_order:
                    vals: List[float] = []
                    if not enc_fold_df.empty:
                        vals = [
                            float(v) for v in enc_fold_df[
                                (enc_fold_df["model"] == model) & (enc_fold_df["roi"] == target_roi)
                            ]["mean_r"].dropna().to_list()
                        ]
                    if vals:
                        m = float(np.mean(vals))
                        se = float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
                    else:
                        sub = enc_df[(enc_df["model"] == model) & (enc_df["roi"] == target_roi)]
                        m = float(sub["mean_r"].iloc[0]) if not sub.empty else np.nan
                        se = 0.0
                    means.append(m)
                    stes.append(se)
                x = np.arange(len(enc_models_order))
                ax.bar(
                    x,
                    means,
                    yerr=stes,
                    color=[enc_colors.get(m, "#999999") for m in enc_models_order],
                    capsize=4,
                    edgecolor="white",
                    linewidth=0.5,
                    error_kw={"linewidth": 1.2},
                )
                ax.axhline(0, color="black", linewidth=0.5)
                if "ram" in enc_models_order:
                    ram_mean = means[enc_models_order.index("ram")]
                    if np.isfinite(ram_mean):
                        ax.axhline(ram_mean, color="#7B3294", linewidth=1.2, linestyle="--", alpha=0.7)
                ax.set_xticks(x)
                ax.set_xticklabels([enc_labels.get(m, m) for m in enc_models_order], rotation=45, ha="right", fontsize=8)
                ax.set_ylabel("Mean Pearson r", fontsize=9)
                ax.set_title(enc_target_labels.get(target_roi, target_roi), fontsize=10, fontweight="bold")
            fig.tight_layout()
            fig.savefig(out_dir / "new_fig5_encoding_comparison.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    log(f"[paper figures] saved to {out_dir}")


# Per-run analysis
def analyze_run(run_label: str, meta: TraceMeta, metas: List[TraceMeta], args: argparse.Namespace) -> Optional[Dict]:
    log(f"  [run] {run_label} start chunks={len(metas)}")
    if not metas:
        log("    [skip] no trace chunks")
        return None

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

    roi_patterns: Dict[str, np.ndarray] = {}
    for roi_name, rel_path in ROI_MASKS.items():
        mask_path = args.atlas_root / rel_path
        if not mask_path.exists():
            log(f"    [warn] ROI mask missing: {mask_path}")
            continue
        try:
            log(f"    [roi] {roi_name} load mask/extract")
            mask = load_roi_mask_nifti(mask_path, fmri_shape)
            patterns = extract_roi_patterns(
                fmri_path,
                mask,
                vol_start,
                vol_start + n_tr_use,
                max_voxels=args.max_roi_voxels,
            )
            if patterns is not None:
                roi_patterns[roi_name] = patterns
                log(f"    [roi] {roi_name} patterns={patterns.shape}")
        except Exception as exc:
            log(f"    [warn] ROI {roi_name}: {exc}")

    if not roi_patterns:
        log("    [skip] no ROI patterns loaded")
        return None

    thinker_feats: Dict[str, np.ndarray] = {}
    mode_key_map = {"primary": "primary", "s2_only": "s2only"}
    for mode, mode_key in mode_key_map.items():
        for rep_name in ("tree_reps", "im_vp_vectors", "im_vectors"):
            feat_name = f"{rep_name}_{mode_key}"
            feat_mat = build_thinker_feature_streaming(
                metas,
                mode=mode,
                rep_name=rep_name,
                max_pca_dim=args.max_pca_dim,
                label=f"{run_label}:{feat_name}",
            )
            if feat_mat is None:
                continue
            if feat_mat.shape[0] < 5:
                log(f"    [trace:{run_label}:{feat_name}] skip too few rows={feat_mat.shape[0]}")
                del feat_mat
                gc.collect()
                continue
            feat_tr = hrf_convolve_uniform_to_tr(feat_mat, source_duration_s=float(n_tr_use), n_tr=n_tr_use)
            del feat_mat
            gc.collect()
            if feat_tr is None:
                continue
            thinker_feats[feat_name] = zscore_columns(feat_tr)
            log(f"    [trace:{run_label}:{feat_name}] TR feature={thinker_feats[feat_name].shape}")

    ram_tr: Optional[np.ndarray] = None
    if args.behav_root is not None:
        behav_path = find_behavioral_file(args.behav_root, meta.subject, meta.session, meta.block, meta.game)
        if behav_path is not None:
            log(f"    [ram] load {behav_path}")
            ram_tr = build_ram_tr_features(behav_path, n_vols, n_tr_use)
            if ram_tr is not None:
                log(f"    [ram] TR feature={ram_tr.shape}")
        else:
            log("    [ram] missing behavioral file")

    return {
        "run_label": run_label,
        "meta": meta,
        "features": thinker_feats,
        "ram_tr": ram_tr,
        "roi_patterns": roi_patterns,
        "n_tr": n_tr_use,
    }


def build_run_dsms(result: Dict) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray]]:
    dsms: Dict[str, np.ndarray] = {}
    for feat_name, feat_mat in result["features"].items():
        log(f"    [dsm] thinker {feat_name} shape={feat_mat.shape}")
        dsms[feat_name] = build_dsm(feat_mat)
    ram_tr = result.get("ram_tr")
    if isinstance(ram_tr, np.ndarray):
        log(f"    [dsm] ram shape={ram_tr.shape}")
        dsms["ram"] = build_dsm(ram_tr)
    roi_patterns = result["roi_patterns"]
    for roi_name, patterns in roi_patterns.items():
        log(f"    [dsm] bold_{roi_name} shape={patterns.shape}")
        dsms[f"bold_{roi_name}"] = build_dsm(patterns)
    if "bold_left_hippocampus" in dsms and "bold_right_hippocampus" in dsms:
        dsms["bold_mean_hippocampus"] = (dsms["bold_left_hippocampus"] + dsms["bold_right_hippocampus"]) / 2.0
    dsms["temporal_lag"] = build_temporal_lag_dsm(result["n_tr"])

    hipp_patterns_for_coupling: Optional[np.ndarray] = None
    if "left_hippocampus" in roi_patterns and "right_hippocampus" in roi_patterns:
        hipp_patterns_for_coupling = np.hstack([roi_patterns["left_hippocampus"], roi_patterns["right_hippocampus"]])
    elif "hippocampus" in roi_patterns:
        hipp_patterns_for_coupling = roi_patterns["hippocampus"]
    elif "left_hippocampus" in roi_patterns:
        hipp_patterns_for_coupling = roi_patterns["left_hippocampus"]
    elif "right_hippocampus" in roi_patterns:
        hipp_patterns_for_coupling = roi_patterns["right_hippocampus"]

    if hipp_patterns_for_coupling is not None and "pfc" in roi_patterns:
        try:
            dsms["coupling_hipp_pfc"] = build_coupling_dsm(hipp_patterns_for_coupling, roi_patterns["pfc"])
            log(f"    [dsm] coupling_hipp_pfc len={len(dsms['coupling_hipp_pfc'])}")
        except Exception as exc:
            log(f"    [warn] coupling DSM: {exc}")
    return dsms, hipp_patterns_for_coupling


def run_rsa_for_result(
    result: Dict,
    dsms: Dict[str, np.ndarray],
    hipp_patterns_for_coupling: Optional[np.ndarray],
    args: argparse.Namespace,
    seed_offset: int = 0,
) -> Tuple[List[Dict], List[Dict], List[Dict], Dict[str, np.ndarray]]:
    meta: TraceMeta = result["meta"]
    run_label = result["run_label"]
    base = {
        "run_label": run_label,
        "subject": meta.subject,
        "session": meta.session,
        "block": meta.block,
        "game": meta.game,
    }
    thinker_keys = [k for k in dsms if any(k.startswith(r) for r in ("tree_reps", "im_vectors", "im_vp_vectors"))]
    ram_key = "ram" if "ram" in dsms else None
    roi_keys = [k for k in dsms if k.startswith("bold_") or k == "coupling_hipp_pfc"]

    rsa_rows: List[Dict] = []

    def add_rsa(a_name: str, b_name: str) -> None:
        a, b = dsms.get(a_name), dsms.get(b_name)
        if a is None or b is None:
            return
        r, p = spearman_rsa(a, b)
        rsa_rows.append({**base, "dsm_a": a_name, "dsm_b": b_name, "rho": r, "p": p})

    for tk in thinker_keys:
        if ram_key:
            add_rsa(tk, ram_key)
    if ram_key:
        for rk in roi_keys:
            add_rsa(ram_key, rk)
    for tk in thinker_keys:
        for rk in roi_keys:
            add_rsa(tk, rk)

    partial_rows: List[Dict] = []

    def add_partial(a_name: str, b_name: str, ctrl_names: List[str]) -> None:
        a = dsms.get(a_name)
        b = dsms.get(b_name)
        ctrls = [dsms[c] for c in ctrl_names if c in dsms]
        if a is None or b is None or not ctrls:
            return
        r, p = partial_spearman(a, b, ctrls)
        partial_rows.append({**base, "dsm_a": a_name, "dsm_b": b_name, "controls": ",".join(ctrl_names), "rho": r, "p": p})

    if ram_key:
        for tk in thinker_keys:
            for rk in roi_keys:
                add_partial(rk, tk, ["temporal_lag"])
                add_partial(rk, ram_key, ["temporal_lag"])
                add_partial(rk, tk, [ram_key, "temporal_lag"])
                add_partial(rk, ram_key, [tk, "temporal_lag"])

    perm_rows: List[Dict] = []
    null_store: Dict[str, np.ndarray] = {}
    if not args.skip_rsa_permutation and args.n_rsa_perm > 0:
        roi_patterns = result["roi_patterns"]
        perm_tasks: List[Tuple[int, str, str, str, np.ndarray, Optional[np.ndarray], int]] = []
        for model_i, model_name in enumerate(thinker_keys + ([ram_key] if ram_key else [])):
            model_dsm = dsms.get(model_name)
            if model_dsm is None:
                continue
            for roi_i, (roi_name, patterns) in enumerate(roi_patterns.items()):
                perm_tasks.append((len(perm_tasks), "bold", model_name, roi_name, patterns, None, args.perm_seed + seed_offset + 1009 * model_i + 17 * roi_i))
            if hipp_patterns_for_coupling is not None and "pfc" in roi_patterns and "coupling_hipp_pfc" in dsms:
                perm_tasks.append((len(perm_tasks), "coupling", model_name, "coupling_hipp_pfc", hipp_patterns_for_coupling, roi_patterns["pfc"], args.perm_seed + seed_offset + 1009 * model_i + 501))

        def run_rsa_perm_task(task: Tuple[int, str, str, str, np.ndarray, Optional[np.ndarray], int]) -> Tuple[int, Dict, str, np.ndarray]:
            task_i, task_kind, model_name, roi_name, patterns, pfc_patterns, rng_seed = task
            model_dsm = dsms.get(model_name)
            if model_dsm is None:
                return task_i, {}, "", np.array([], dtype=np.float32)
            if task_kind == "bold":
                key = f"{model_name}_vs_bold_{roi_name}"
                log(f"    [rsa perm] {key} n_perm={args.n_rsa_perm}")
                nulls, p1, p2 = block_permutation_rsa(
                    model_dsm,
                    patterns,
                    n_tr=result["n_tr"],
                    block_size=args.rsa_perm_block_size,
                    n_perm=args.n_rsa_perm,
                    rng_seed=rng_seed,
                    label=f"{run_label}:{key}" if effective_n_jobs(args.n_jobs) == 1 else "",
                )
                obs_r, _ = spearman_rsa(model_dsm, dsms.get(f"bold_{roi_name}", model_dsm))
                row = {**base, "model": model_name, "roi": roi_name, "rho_obs": obs_r, "p_one_sided": p1, "p_two_sided": p2}
                return task_i, row, key, nulls

            if pfc_patterns is None:
                return task_i, {}, "", np.array([], dtype=np.float32)
            key = f"{model_name}_vs_coupling_hipp_pfc"
            log(f"    [rsa perm] {key} n_perm={args.n_rsa_perm}")
            nulls, p1, p2 = block_permutation_coupling_rsa(
                model_dsm,
                patterns,
                pfc_patterns,
                n_tr=result["n_tr"],
                block_size=args.rsa_perm_block_size,
                n_perm=args.n_rsa_perm,
                rng_seed=rng_seed,
                label=f"{run_label}:{key}" if effective_n_jobs(args.n_jobs) == 1 else "",
            )
            obs_r, _ = spearman_rsa(model_dsm, dsms["coupling_hipp_pfc"])
            row = {**base, "model": model_name, "roi": "coupling_hipp_pfc", "rho_obs": obs_r, "p_one_sided": p1, "p_two_sided": p2}
            return task_i, row, key, nulls

        rsa_jobs = min(effective_n_jobs(args.n_jobs), max(1, len(perm_tasks)))
        if rsa_jobs > 1 and len(perm_tasks) > 1:
            log(f"    [rsa perm] parallel tasks={len(perm_tasks)} n_jobs={rsa_jobs}")
            task_results: List[Tuple[int, Dict, str, np.ndarray]] = []
            with ThreadPoolExecutor(max_workers=rsa_jobs) as pool:
                futures = [pool.submit(run_rsa_perm_task, task) for task in perm_tasks]
                for fut in as_completed(futures):
                    task_results.append(fut.result())
        else:
            task_results = [run_rsa_perm_task(task) for task in perm_tasks]

        for _, row, key, nulls in sorted(task_results, key=lambda item: item[0]):
            if row:
                perm_rows.append(row)
            if key and nulls.size > 0:
                null_store[key] = nulls

    return rsa_rows, partial_rows, perm_rows, null_store


def run_loro_encoding_analysis(run_results: List[Dict], args: argparse.Namespace, out_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = out_root / "encoding_loro"
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(run_results) < 2:
        log("[encoding] need at least 2 runs for LORO; skipping")
        return pd.DataFrame(), pd.DataFrame()

    model_maps = [build_model_map(r) for r in run_results]
    all_models = sorted(set(k for mm in model_maps for k in mm))
    all_rois = sorted(set(k for r in run_results for k in r.get("roi_patterns", {})))
    run_labels = [r["run_label"] for r in run_results]
    first_meta: TraceMeta = run_results[0]["meta"]

    combo_tasks: List[Tuple[int, str, str, List[np.ndarray], List[np.ndarray], int]] = []
    for roi_name in all_rois:
        Y_runs = []
        for result in run_results:
            y = result.get("roi_patterns", {}).get(roi_name)
            if y is None:
                Y_runs = []
                break
            Y_runs.append(y)
        if len(Y_runs) != len(run_results):
            log(f"  [skip roi] {roi_name}: missing in some runs")
            continue
        n_vox = min(y.shape[1] for y in Y_runs)
        Y_runs = [y[:, :n_vox] for y in Y_runs]

        for model_name in all_models:
            X_runs = []
            for mm in model_maps:
                x = mm.get(model_name)
                if x is None:
                    X_runs = []
                    break
                X_runs.append(x)
            if len(X_runs) != len(run_results):
                continue

            xy_pairs = []
            for x, y in zip(X_runs, Y_runs):
                t = min(x.shape[0], y.shape[0])
                xy_pairs.append((x[:t], y[:t]))
            X_runs_aligned = [p[0] for p in xy_pairs]
            Y_runs_aligned = [p[1] for p in xy_pairs]
            combo_seed = int(args.perm_seed + 200_003 + 1009 * len(combo_tasks))
            combo_tasks.append((len(combo_tasks), roi_name, model_name, X_runs_aligned, Y_runs_aligned, combo_seed))

    def run_encoding_combo_task(task: Tuple[int, str, str, List[np.ndarray], List[np.ndarray], int]) -> Tuple[int, Optional[Dict]]:
        task_i, roi_name, model_name, X_runs_aligned, Y_runs_aligned, combo_seed = task
        label = f"{roi_name}:{model_name}"
        worker_label = label if effective_n_jobs(args.n_jobs) == 1 else ""
        log(f"\n[encoding LORO] {label}")
        rs, best_alpha = run_loro_encoding(X_runs_aligned, Y_runs_aligned, alphas=args.ridge_alphas, label=worker_label)
        if rs is None or rs.size == 0:
            return task_i, None

        if not args.skip_encoding_permutation and args.n_encoding_perm > 0:
            perm = run_loro_permutation(
                X_runs_aligned,
                Y_runs_aligned,
                best_alpha,
                n_perm=args.n_encoding_perm,
                block_size=args.encoding_perm_block_size,
                rng_seed=combo_seed,
                run_labels=run_labels,
                label=worker_label,
            )
            rs = np.asarray(perm["rs"])
        else:
            rs2, _, _, fold_rows = _loro_predictions(X_runs_aligned, Y_runs_aligned, best_alpha, run_labels=run_labels, label="")
            rs = rs2
            perm = {"rs": rs, "fold_rows": fold_rows}

        fold_rows = []
        for fold_row in list(perm.get("fold_rows", [])):
            fold_rows.append({
                "run_label": "ALL_RUNS",
                "heldout_run": fold_row["heldout_run"],
                "subject": first_meta.subject,
                "session": -1,
                "block": -1,
                "game": first_meta.game,
                "roi": roi_name,
                "model": model_name,
                "cv_scheme": "loro_alpha_cv",
                "best_alpha": best_alpha,
                **{k: v for k, v in fold_row.items() if k not in {"heldout_run"}},
            })

        p_one = perm.get("p_one_sided")
        p_two = perm.get("p_two_sided")
        q_one = perm.get("q_fdr_one_sided")
        null_mean_scores = perm.get("null_mean_scores")
        finite_rs = np.isfinite(rs)
        n_valid = int(finite_rs.sum())
        n_p05 = int(np.sum(np.isfinite(p_one) & (p_one < 0.05))) if isinstance(p_one, np.ndarray) else 0
        n_q05 = int(np.sum(np.isfinite(q_one) & (q_one < 0.05))) if isinstance(q_one, np.ndarray) else 0
        frac_p05 = float(n_p05 / n_valid) if n_valid else float("nan")
        frac_q05 = float(n_q05 / n_valid) if n_valid else float("nan")
        null_mean = finite_stat(null_mean_scores if isinstance(null_mean_scores, np.ndarray) else None, np.mean)
        null_p95 = finite_stat(null_mean_scores if isinstance(null_mean_scores, np.ndarray) else None, lambda v: np.percentile(v, 95))
        mean_r = float(np.nanmean(rs))
        median_r = float(np.nanmedian(rs))

        voxel_entries: Dict[str, np.ndarray] = {}
        voxel_key_rows: List[Dict[str, str]] = []
        if isinstance(p_one, np.ndarray) and isinstance(q_one, np.ndarray):
            safe_key = re.sub(r"[^A-Za-z0-9_]+", "_", f"{roi_name}__{model_name}")
            voxel_entries[f"{safe_key}__r"] = rs.astype(np.float32)
            voxel_entries[f"{safe_key}__p_one_sided"] = p_one.astype(np.float32)
            if isinstance(p_two, np.ndarray):
                voxel_entries[f"{safe_key}__p_two_sided"] = p_two.astype(np.float32)
            voxel_entries[f"{safe_key}__q_fdr_one_sided"] = q_one.astype(np.float32)
            voxel_key_rows.append({"roi": roi_name, "model": model_name, "key_prefix": safe_key})

        row = {
            "run_label": "ALL_RUNS",
            "subject": first_meta.subject,
            "session": -1,
            "block": -1,
            "game": first_meta.game,
            "roi": roi_name,
            "model": model_name,
            "cv_scheme": "loro_alpha_cv",
            "best_alpha": best_alpha,
            "mean_r": mean_r,
            "median_r": median_r,
            "n_voxels": int(len(rs)),
            "n_voxels_finite": n_valid,
            "n_voxels_positive": int(np.sum(rs > 0)),
            "n_perm": int(args.n_encoding_perm if isinstance(p_one, np.ndarray) else 0),
            "perm_block_size": int(args.encoding_perm_block_size if isinstance(p_one, np.ndarray) else 0),
            "n_voxels_p05": n_p05,
            "frac_voxels_p05": frac_p05,
            "n_voxels_q05": n_q05,
            "frac_voxels_q05": frac_q05,
            "min_p_one_sided": finite_stat(p_one if isinstance(p_one, np.ndarray) else None, np.min),
            "median_p_one_sided": finite_stat(p_one if isinstance(p_one, np.ndarray) else None, np.median),
            "min_q_fdr_one_sided": finite_stat(q_one if isinstance(q_one, np.ndarray) else None, np.min),
            "median_q_fdr_one_sided": finite_stat(q_one if isinstance(q_one, np.ndarray) else None, np.median),
            "null_mean_r": null_mean,
            "null_p95_mean_r": null_p95,
            "mean_r_minus_null95_mean_r": mean_r - null_p95 if np.isfinite(null_p95) else float("nan"),
        }
        log(f"  -> {label} mean_r={mean_r:.5f} best_alpha={best_alpha:.3g} q05={n_q05}/{n_valid}")
        return task_i, {
            "row": row,
            "fold_rows": fold_rows,
            "voxel_entries": voxel_entries,
            "voxel_key_rows": voxel_key_rows,
        }

    rows: List[Dict] = []
    fold_rows_all: List[Dict] = []
    voxel_stats: Dict[str, np.ndarray] = {}
    voxel_key_rows: List[Dict[str, str]] = []

    enc_jobs = min(effective_n_jobs(args.n_jobs), max(1, len(combo_tasks)))
    if enc_jobs > 1 and len(combo_tasks) > 1:
        log(f"[encoding] parallel tasks={len(combo_tasks)} n_jobs={enc_jobs}")
        task_results: List[Tuple[int, Optional[Dict]]] = []
        with ThreadPoolExecutor(max_workers=enc_jobs) as pool:
            futures = [pool.submit(run_encoding_combo_task, task) for task in combo_tasks]
            for fut in as_completed(futures):
                task_results.append(fut.result())
    else:
        task_results = [run_encoding_combo_task(task) for task in combo_tasks]

    for _, task_out in sorted(task_results, key=lambda item: item[0]):
        if task_out is None:
            continue
        rows.append(task_out["row"])
        fold_rows_all.extend(task_out["fold_rows"])
        voxel_stats.update(task_out["voxel_entries"])
        voxel_key_rows.extend(task_out["voxel_key_rows"])

    enc_df = pd.DataFrame(rows)
    fold_df = pd.DataFrame(fold_rows_all)
    if not enc_df.empty:
        enc_df.to_csv(out_dir / "encoding_loro_manifest.csv", index=False)
        compat_dir = out_root / "encoding"
        compat_dir.mkdir(parents=True, exist_ok=True)
        enc_df.to_csv(compat_dir / "encoding_manifest.csv", index=False)
    if not fold_df.empty:
        fold_df.to_csv(out_dir / "encoding_loro_fold_manifest.csv", index=False)
    if voxel_stats:
        np.savez_compressed(out_dir / "encoding_loro_voxel_stats.npz", **voxel_stats)
        pd.DataFrame(voxel_key_rows).to_csv(out_dir / "encoding_loro_voxel_stats_keys.csv", index=False)
    return enc_df, fold_df


def run_subject_game(args: argparse.Namespace) -> None:
    subject, game = args.subject, args.game
    sessions = set(int(s) for s in args.sessions.split(",")) if args.sessions else None
    label = f"sub{subject:03d}_game{game}"
    out_root = args.output_root / label
    dirs = {
        "features": out_root / "features",
        "dsms": out_root / "dsms",
        "rsa": out_root / "rsa",
        "encoding_loro": out_root / "encoding_loro",
        "figures": out_root / "figures",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    blocks = gather_trace_blocks(args.trace_root, subject, game, sessions)
    if not blocks:
        log(f"[error] no trace files for sub{subject:03d} game{game}")
        return
    log(f"[{label}] found {len(blocks)} blocks")

    run_results: List[Dict] = []
    all_rsa: List[Dict] = []
    all_partial: List[Dict] = []
    all_perm: List[Dict] = []
    all_nulls: Dict[str, np.ndarray] = {}

    for block_i, (block_id, metas) in enumerate(sorted(blocks.items())):
        representative = metas[0]
        run_label = representative.run_label
        result = analyze_run(run_label, representative, metas, args)
        if result is None:
            continue
        run_results.append(result)

        feat_save: Dict[str, np.ndarray] = {
            k: v for k, v in result["features"].items() if isinstance(v, np.ndarray)
        }
        if isinstance(result.get("ram_tr"), np.ndarray):
            feat_save["ram_tr"] = result["ram_tr"]
        if feat_save:
            log(f"    [save] features npz {run_label}")
            np.savez_compressed(dirs["features"] / f"features_{run_label}.npz", **feat_save)

        if not args.skip_rsa:
            dsms, hipp_patterns_for_coupling = build_run_dsms(result)
            result["dsms"] = dsms
            result["hipp_patterns_for_coupling"] = hipp_patterns_for_coupling
            rsa_rows, partial_rows, perm_rows, null_store = run_rsa_for_result(
                result,
                dsms,
                hipp_patterns_for_coupling,
                args,
                seed_offset=10_000 * block_i,
            )
            all_rsa.extend(rsa_rows)
            all_partial.extend(partial_rows)
            all_perm.extend(perm_rows)
            for k, v in null_store.items():
                all_nulls[f"{run_label}/{k}"] = v
            log(f"    [save] DSM npz {run_label}")
            np.savez_compressed(dirs["dsms"] / f"dsms_{run_label}.npz", **{k: v for k, v in dsms.items() if isinstance(v, np.ndarray)})
            if not args.skip_plots:
                try:
                    plot_dsm_panel(dsms, run_label, dirs["figures"] / f"dsm_panel_{run_label}.png")
                except Exception as exc:
                    log(f"    [warn] DSM plot: {exc}")

    if not run_results:
        log(f"[{label}] no valid runs, exiting.")
        return

    rsa_df = pd.DataFrame(all_rsa)
    partial_df = pd.DataFrame(all_partial)
    perm_df = pd.DataFrame(all_perm)
    if not perm_df.empty and "p_one_sided" in perm_df.columns:
        perm_df["q_fdr_one_sided"] = fdr_bh(perm_df["p_one_sided"].values)
        perm_df["q_fdr_two_sided"] = fdr_bh(perm_df["p_two_sided"].values)
    if not rsa_df.empty and "p" in rsa_df.columns:
        rsa_df["q_fdr"] = fdr_bh(rsa_df["p"].values)

    rsa_df.to_csv(dirs["rsa"] / "rsa_manifest.csv", index=False)
    partial_df.to_csv(dirs["rsa"] / "rsa_partial_manifest.csv", index=False)
    perm_df.to_csv(dirs["rsa"] / "rsa_permutation_manifest.csv", index=False)
    if all_nulls:
        np.savez_compressed(dirs["rsa"] / "rsa_nulls.npz", **{k.replace("/", "__"): v for k, v in all_nulls.items()})

    enc_df = pd.DataFrame()
    enc_fold_df = pd.DataFrame()
    if not args.skip_encoding:
        enc_df, enc_fold_df = run_loro_encoding_analysis(run_results, args, out_root)

    if not args.skip_plots:
        if not rsa_df.empty:
            plot_rsa_heatmap(rsa_df, f"{label} - RSA rho (mean over runs)", dirs["figures"] / "rsa_heatmap_aggregate.png")
        if not perm_df.empty and all_nulls:
            try:
                combined_nulls = {k.split("/")[-1]: v for k, v in all_nulls.items()}
                plot_permutation_nulls(combined_nulls, perm_df, dirs["figures"] / "perm_nulls.png")
            except Exception as exc:
                log(f"  [warn] perm null plot: {exc}")
        if not enc_df.empty:
            generate_loro_plots(enc_df, dirs["encoding_loro"])
        if not args.skip_paper_figures:
            generate_paper_figures(perm_df, enc_df, enc_fold_df, dirs["figures"] / "paper", label)

    summary_lines = [
        f"# {label} integrated encoding/RSA summary",
        "",
        f"Runs analyzed: {len(run_results)}",
        f"RSA comparisons: {len(rsa_df)}",
        f"Partial RSA comparisons: {len(partial_df)}",
        f"RSA permutation rows: {len(perm_df)}",
        f"LORO encoding rows: {len(enc_df)}",
        f"LORO encoding fold rows: {len(enc_fold_df)}",
    ]
    if not perm_df.empty and "q_fdr_one_sided" in perm_df.columns:
        sig = perm_df[perm_df["q_fdr_one_sided"] < 0.05]
        summary_lines.append(f"RSA FDR q<0.05 one-sided rows: {len(sig)}")
    if not enc_df.empty and "frac_voxels_q05" in enc_df.columns:
        summary_lines.append("")
        summary_lines.append("## LORO Encoding Mean r")
        summary_lines.append(enc_df[["roi", "model", "mean_r", "median_r", "best_alpha"]].to_string(index=False))
    (out_root / "summary.md").write_text("\n".join(summary_lines) + "\n")

    log(f"[{label}] done -> {out_root}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject", type=int, required=True)
    p.add_argument("--game", type=int, required=True)
    p.add_argument("--sessions", default=None, help="Comma-separated session ids, e.g. 3,4")
    p.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    p.add_argument("--behav-root", type=Path, default=DEFAULT_BEHAV_ROOT)
    p.add_argument("--fmri-root", type=Path, default=DEFAULT_FMRI_ROOT)
    p.add_argument("--atlas-root", type=Path, default=DEFAULT_ATLAS_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--allow-smoothed-fallback", action="store_true")
    p.add_argument("--max-roi-voxels", type=int, default=MAX_ROI_VOXELS)
    p.add_argument("--max-pca-dim", type=int, default=MAX_PCA_DIM)
    p.add_argument("--n-jobs", type=int, default=1,
                   help="Thread workers for independent RSA permutation and encoding ROI/model tasks.")
    p.add_argument("--alphas-log", nargs=2, type=float, default=[-2, 5], metavar=("LO", "HI"))
    p.add_argument("--n-alphas", type=int, default=15)
    p.add_argument("--perm-seed", type=int, default=0)
    p.add_argument("--n-rsa-perm", type=int, default=N_PERM)
    p.add_argument("--rsa-perm-block-size", type=int, default=PERM_BLOCK)
    p.add_argument("--n-encoding-perm", type=int, default=N_PERM)
    p.add_argument("--encoding-perm-block-size", type=int, default=PERM_BLOCK)
    p.add_argument("--skip-rsa", action="store_true")
    p.add_argument("--skip-encoding", action="store_true")
    p.add_argument("--skip-rsa-permutation", action="store_true")
    p.add_argument("--skip-encoding-permutation", action="store_true")
    p.add_argument("--skip-plots", action="store_true")
    p.add_argument("--skip-paper-figures", action="store_true")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    args.n_jobs = effective_n_jobs(args.n_jobs)
    args.ridge_alphas = np.logspace(args.alphas_log[0], args.alphas_log[1], args.n_alphas)
    log(f"Alpha grid: {np.round(args.ridge_alphas, 4)}")
    log(f"CPU parallel workers: n_jobs={args.n_jobs}")
    run_subject_game(args)


if __name__ == "__main__":
    main()
