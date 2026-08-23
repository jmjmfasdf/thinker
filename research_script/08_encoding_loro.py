"""
08_encoding_loro.py — Leave-one-run-out voxelwise encoding
following Logan Cross et al. (2020, Neuron).

Differences from 07_encoding_rsa.py:
  - LOO across runs as the only CV scheme
  - Alpha selection via LOO Pearson r grid search
    (not GCV/RidgeCV within each fold)

Inputs  (pre-computed by 07_encoding_rsa.py):
  <out_dir>/features/features_<run_label>.npz   — Thinker (T×100) + RAM (T×128)

Inputs  (re-loaded from original data):
  <fmri_root>/<fmri_subject>/Session<block>/wfiltered_func_data.nii
  <atlas_root>/<roi_rel_path>.nii.gz

Outputs:
  <out_dir>/encoding_loro/encoding_loro_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── paths (mirror defaults from 07_encoding_rsa.py) ─────────────────────────
SCRIPT_DIR    = Path(__file__).resolve().parent
DEFAULT_FMRI_ROOT  = Path("/home/jeongmin/fmri/atari/derivatives/ants_mni")
DEFAULT_ATLAS_ROOT = (
    SCRIPT_DIR / "outputs/06_representational_mechanism"
    "/atlas/harvard_oxford/ants_mni_2p5mm_masks/masks"
)
DEFAULT_OUT_ROOT = SCRIPT_DIR / "outputs/07_encoding_rsa"

# ── constants (mirror 07_encoding_rsa.py) ────────────────────────────────────
FMRI_TRIM     = 60
N_ANALYSIS    = 480
PERM_BLOCK    = 40
N_PERM        = 1000
MAX_ROI_VOXELS = 4000
MAX_PCA_DIM   = 100
RIDGE_ALPHAS  = np.logspace(-2, 5, 15)
EPS           = 1e-12

ROI_MASKS: Dict[str, str] = {
    "left_hippocampus":  "subcortical/roi-subcortical-010_Left-Hippocampus_mask.nii.gz",
    "right_hippocampus": "subcortical/roi-subcortical-020_Right-Hippocampus_mask.nii.gz",
    "hippocampus":       "group/roi-HarvardOxford-Hippocampus_mask.nii.gz",
    "pfc":               "group/roi-HarvardOxford-PFC_mask.nii.gz",
}


# ── logging ──────────────────────────────────────────────────────────────────
def log(msg: str) -> None:
    print(msg, flush=True)


# ── run metadata from filename ────────────────────────────────────────────────
def parse_run_label(run_label: str) -> Optional[Dict]:
    """sub001_ses01_block03_game2 → {subject, session, block, game}"""
    m = re.match(r"sub(\d+)_ses(\d+)_block(\d+)_game(\d+)$", run_label)
    if not m:
        return None
    return {
        "subject": int(m.group(1)),
        "session": int(m.group(2)),
        "block":   int(m.group(3)),
        "game":    int(m.group(4)),
    }


def fmri_subject_str(subject: int, session: int) -> str:
    """Matches TraceMeta.fmri_subject in 07_encoding_rsa.py."""
    return f"sub{subject:03d}-{session}"


# ── fMRI / ROI loading ────────────────────────────────────────────────────────
def find_fmri_path(fmri_root: Path, subject: int, session: int, block: int) -> Optional[Path]:
    fmri_subj = fmri_subject_str(subject, session)
    p = fmri_root / fmri_subj / f"Session{block}" / "wfiltered_func_data.nii"
    return p if p.exists() else None


def load_roi_mask(mask_path: Path, ref_shape: Tuple) -> np.ndarray:
    import nibabel as nib
    img  = nib.load(str(mask_path))
    data = np.squeeze(np.asarray(img.get_fdata(dtype=np.float32)))
    if data.ndim != 3:
        raise ValueError(f"Expected 3D mask, got {data.shape}")
    if tuple(data.shape) != tuple(ref_shape[:3]):
        raise ValueError(f"Mask shape {data.shape} != fMRI {ref_shape[:3]}")
    return data > 0


def zscore_columns(mat: np.ndarray) -> np.ndarray:
    m  = mat.mean(axis=0, keepdims=True)
    sd = mat.std(axis=0, keepdims=True)
    sd[sd < EPS] = 1.0
    return (mat - m) / sd


def extract_bold_patterns(
    fmri_path: Path,
    mask: np.ndarray,
    vol_start: int,
    vol_stop: int,
    max_voxels: int = MAX_ROI_VOXELS,
    seed: int = 42,
) -> Optional[np.ndarray]:
    """Returns (n_vols × n_voxels) z-scored BOLD patterns."""
    import nibabel as nib
    img    = nib.load(str(fmri_path))
    coords = np.column_stack(np.where(mask)).astype(np.int32)
    if coords.size == 0:
        return None
    if max_voxels > 0 and len(coords) > max_voxels:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(coords), max_voxels, replace=False)
        idx.sort()
        coords = coords[idx]
    n_vols = vol_stop - vol_start
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    patterns = np.empty((n_vols, len(coords)), dtype=np.float32)
    proxy = img.dataobj
    for i, v in enumerate(range(vol_start, vol_stop)):
        vol = np.asarray(proxy[:, :, :, v], dtype=np.float32)
        patterns[i] = vol[x, y, z]
    return zscore_columns(patterns).astype(np.float32)


# ── encoding core ─────────────────────────────────────────────────────────────
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


def _preprocess_and_fit(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """StandardScaler → PCA (max 100) → Ridge(alpha). Returns Y_pred."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge

    sx      = StandardScaler()
    Xtr_s   = sx.fit_transform(X_train)
    Xte_s   = sx.transform(X_test)

    n_comp  = min(MAX_PCA_DIM, Xtr_s.shape[1], Xtr_s.shape[0] - 1)
    if n_comp < 1:
        n_out = Y_train.shape[1] if Y_train.ndim > 1 else 1
        return np.zeros((X_test.shape[0], n_out), dtype=np.float32)

    pca     = PCA(n_components=n_comp, random_state=0)
    Xtr_p   = pca.fit_transform(Xtr_s)
    Xte_p   = pca.transform(Xte_s)

    ridge   = Ridge(alpha=alpha, fit_intercept=True)
    ridge.fit(Xtr_p, Y_train)
    return ridge.predict(Xte_p).astype(np.float32)


def _loro_with_alpha(
    X_runs: List[np.ndarray],
    Y_runs: List[np.ndarray],
    alpha: float,
) -> np.ndarray:
    """
    Full leave-one-run-out with a fixed alpha.
    Returns mean held-out Pearson r per voxel.
    """
    n_runs   = len(X_runs)
    n_voxels = Y_runs[0].shape[1]
    voxel_rs = np.zeros(n_voxels, dtype=np.float64)
    counts   = np.zeros(n_voxels, dtype=int)

    for test_i in range(n_runs):
        X_test = X_runs[test_i]
        Y_test = Y_runs[test_i]
        X_train = np.vstack([x for i, x in enumerate(X_runs) if i != test_i])
        Y_train = np.vstack([y for i, y in enumerate(Y_runs) if i != test_i])

        if X_train.shape[0] < 5 or X_test.shape[0] < 2:
            continue

        Y_pred = _preprocess_and_fit(X_train, Y_train, X_test, alpha)
        rs     = pearson_columns(Y_test, Y_pred)
        valid  = np.isfinite(rs)
        voxel_rs[valid] += rs[valid]
        counts[valid]   += 1

    result = np.full(n_voxels, np.nan, dtype=np.float32)
    ok     = counts > 0
    result[ok] = (voxel_rs[ok] / counts[ok]).astype(np.float32)
    return result


def run_loro_encoding(
    X_runs: List[np.ndarray],
    Y_runs: List[np.ndarray],
    alphas: np.ndarray = RIDGE_ALPHAS,
    label: str = "",
) -> Tuple[np.ndarray, float]:
    """
    Leave-one-run-out encoding with alpha selected by LOO Pearson r.

    For each alpha: run full LOO, compute mean voxel Pearson r.
    Select alpha with highest mean, return its per-voxel results.

    Returns:
        rs      — per-voxel mean Pearson r  (n_voxels,)
        best_alpha — selected regularization strength
    """
    if len(X_runs) < 2:
        return np.array([], dtype=np.float32), float('nan')

    best_alpha  = alphas[0]
    best_score  = -np.inf
    best_rs     = None

    for alpha in alphas:
        rs    = _loro_with_alpha(X_runs, Y_runs, alpha)
        score = float(np.nanmean(rs))
        if label:
            log(f"    [alpha={alpha:.3g}] mean_r={score:.5f}")
        if score > best_score:
            best_score = score
            best_alpha = alpha
            best_rs    = rs

    if label:
        log(f"  [{label}] best_alpha={best_alpha:.3g}  best_mean_r={best_score:.5f}")

    return best_rs, best_alpha


# ── encoding-specific permutation / FDR ──────────────────────────────────────
def fdr_bh(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction over finite p-values."""
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


def block_permutation_indices(T: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Shuffle a held-out run in contiguous blocks while preserving within-block order."""
    if T <= 0:
        return np.array([], dtype=int)
    block_size = max(1, int(block_size))
    blocks = [
        np.arange(lo, min(lo + block_size, T), dtype=int)
        for lo in range(0, T, block_size)
    ]
    if len(blocks) < 2:
        return np.arange(T, dtype=int)
    order = rng.permutation(len(blocks))
    return np.concatenate([blocks[i] for i in order])


def _loro_predictions(
    X_runs: List[np.ndarray],
    Y_runs: List[np.ndarray],
    alpha: float,
    label: str = "",
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """Fit best-alpha LORO folds and keep held-out predictions for permutation."""
    n_voxels = Y_runs[0].shape[1]
    voxel_rs = np.zeros(n_voxels, dtype=np.float64)
    counts = np.zeros(n_voxels, dtype=int)
    y_tests: List[np.ndarray] = []
    y_preds: List[np.ndarray] = []

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

    result = np.full(n_voxels, np.nan, dtype=np.float32)
    ok = counts > 0
    result[ok] = (voxel_rs[ok] / counts[ok]).astype(np.float32)
    return result, y_tests, y_preds


def run_loro_permutation(
    X_runs: List[np.ndarray],
    Y_runs: List[np.ndarray],
    alpha: float,
    n_perm: int = N_PERM,
    block_size: int = PERM_BLOCK,
    rng_seed: int = 0,
    label: str = "",
) -> Dict[str, np.ndarray]:
    """
    Encoding-specific block permutation.

    LORO predictions are held fixed. For each permutation, only the held-out
    BOLD time course is shuffled in 40-TR blocks, then correlated with Y_pred.
    """
    rs, y_tests, y_preds = _loro_predictions(X_runs, Y_runs, alpha, label=label)
    n_voxels = len(rs)
    out: Dict[str, np.ndarray] = {"rs": rs}
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
            Y_perm = Y_test[perm_idx]
            perm_rs = pearson_columns(Y_perm, Y_pred)
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
            log(f"    [perm:{label}] {perm_i + 1}/{n_perm}")

    p_one = np.full(n_voxels, np.nan, dtype=np.float32)
    p_two = np.full(n_voxels, np.nan, dtype=np.float32)
    p_one[obs_ok] = (1.0 + ge_counts[obs_ok]) / (1.0 + n_perm)
    p_two[obs_ok] = (1.0 + abs_counts[obs_ok]) / (1.0 + n_perm)
    q_one = fdr_bh(p_one)

    out.update({
        "p_one_sided": p_one,
        "p_two_sided": p_two,
        "q_fdr_one_sided": q_one,
        "null_mean_scores": null_mean_scores,
    })
    return out


# ── feature assembly ──────────────────────────────────────────────────────────
def load_run_features(feat_npz: Path) -> Optional[Dict[str, np.ndarray]]:
    """Load pre-computed features; build ram_plus_X combinations."""
    try:
        d = np.load(str(feat_npz))
    except Exception as exc:
        log(f"  [warn] cannot load {feat_npz}: {exc}")
        return None

    feats: Dict[str, np.ndarray] = {}
    ram_tr = d["ram_tr"] if "ram_tr" in d else None

    if ram_tr is not None:
        feats["ram"] = ram_tr

    thinker_keys = [k for k in d.files if k != "ram_tr"]
    for k in thinker_keys:
        arr = d[k]
        feats[f"thinker_{k}"] = arr
        if ram_tr is not None:
            min_t = min(ram_tr.shape[0], arr.shape[0])
            feats[f"ram_plus_{k}"] = np.hstack([ram_tr[:min_t], arr[:min_t]])

    return feats


# ── plotting ────────────────────────────────────────────────────────────────
def _model_label(name: str) -> str:
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
        rest
        .replace("tree_reps", "tree")
        .replace("im_vp_vectors", "im-vp")
        .replace("im_vectors", "im")
        .replace("_primary", " primary")
        .replace("_s2only", " s2-only")
        .replace("_", " ")
    )
    return prefix + rest


def _plot_heatmap(
    pivot,
    out_path: Path,
    title: str,
    cbar_label: str,
    cmap: str = "RdBu_r",
    center_zero: bool = False,
    fmt: str = ".3f",
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    data = pivot.to_numpy(dtype=float)
    if center_zero and np.isfinite(data).any():
        vmax = float(np.nanmax(np.abs(data)))
        vmin = -vmax
    else:
        vmin = float(np.nanmin(data)) if np.isfinite(data).any() else 0.0
        vmax = float(np.nanmax(data)) if np.isfinite(data).any() else 1.0
        if abs(vmax - vmin) < EPS:
            vmax = vmin + 1.0

    fig, ax = plt.subplots(figsize=(1.35 * len(pivot.columns) + 5.8,
                                    0.36 * len(pivot.index) + 1.8))
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


def generate_plots(csv_path: Path, out_dir: Path) -> None:
    """Create summary figures for LORO encoding and permutation/FDR outputs."""
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib-cache"
    os.environ["XDG_CACHE_HOME"] = "/tmp"
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import pandas as pd
    import numpy as np

    if not csv_path.exists():
        log(f"[plot] missing CSV: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        log(f"[plot] empty CSV: {csv_path}")
        return

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    roi_order = [r for r in ["left_hippocampus", "right_hippocampus", "hippocampus", "pfc"]
                 if r in set(df["roi"])]
    roi_order += [r for r in sorted(df["roi"].unique()) if r not in roi_order]
    roi_labels = {
        "left_hippocampus": "Left hippocampus",
        "right_hippocampus": "Right hippocampus",
        "hippocampus": "Hippocampus",
        "pfc": "PFC",
    }
    roi_display = {r: roi_labels.get(r, r) for r in roi_order}

    model_order = [
        "ram",
        "thinker_tree_reps_primary", "thinker_tree_reps_s2only",
        "thinker_im_vectors_primary", "thinker_im_vectors_s2only",
        "thinker_im_vp_vectors_primary", "thinker_im_vp_vectors_s2only",
        "ram_plus_tree_reps_primary", "ram_plus_tree_reps_s2only",
        "ram_plus_im_vectors_primary", "ram_plus_im_vectors_s2only",
        "ram_plus_im_vp_vectors_primary", "ram_plus_im_vp_vectors_s2only",
    ]
    model_order = [m for m in model_order if m in set(df["model"])]
    model_order += [m for m in sorted(df["model"].unique()) if m not in model_order]
    model_display = {m: _model_label(m) for m in model_order}

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

    # 1. Mean r: horizontal bars by ROI.
    fig, axes = plt.subplots(1, len(roi_order), figsize=(4.2 * len(roi_order), 7.2), sharex=True)
    if len(roi_order) == 1:
        axes = [axes]
    for ax, roi in zip(axes, roi_order):
        sub = df[df["roi"] == roi].set_index("model").reindex(model_order).dropna(subset=["mean_r"])
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

    # 2. Mean r heatmap.
    pivot = df.pivot(index="model", columns="roi", values="mean_r").reindex(index=model_order, columns=roi_order)
    pivot = pivot.rename(index=model_display, columns=roi_display)
    _plot_heatmap(pivot, plot_dir / "encoding_loro_mean_r_heatmap.png",
                  "Mean Pearson r", "mean r", center_zero=True, fmt=".3f")
    created.append(plot_dir / "encoding_loro_mean_r_heatmap.png")

    # 3. RAM+Thinker delta over RAM.
    ram = df[df["model"] == "ram"].set_index("roi")["mean_r"]
    delta_rows = []
    for _, row in df[df["model"].str.startswith("ram_plus_")].iterrows():
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
        pivot = pivot.rename(index={m: _model_label("thinker_" + m).replace("Thinker ", "") for m in delta_models},
                             columns=roi_display)
        _plot_heatmap(pivot, plot_dir / "encoding_loro_delta_over_ram_heatmap.png",
                      "Incremental encoding over RAM", "delta mean r",
                      cmap="PiYG", center_zero=True, fmt="+.3f")
        created.append(plot_dir / "encoding_loro_delta_over_ram_heatmap.png")

    # 4. Positive voxel fraction.
    if {"n_voxels_positive", "n_voxels"}.issubset(df.columns):
        df["positive_fraction"] = df["n_voxels_positive"] / df["n_voxels"]
        pivot = df.pivot(index="model", columns="roi", values="positive_fraction").reindex(
            index=model_order, columns=roi_order
        )
        pivot = pivot.rename(index=model_display, columns=roi_display)
        _plot_heatmap(pivot, plot_dir / "encoding_loro_positive_voxel_fraction.png",
                      "Fraction of voxels with positive r", "positive fraction",
                      cmap="viridis", center_zero=False, fmt=".2f")
        created.append(plot_dir / "encoding_loro_positive_voxel_fraction.png")

    # 5. Best alpha heatmap.
    pivot = df.pivot(index="model", columns="roi", values="best_alpha").reindex(index=model_order, columns=roi_order)
    pivot = np.log10(pivot.astype(float)).rename(index=model_display, columns=roi_display)
    _plot_heatmap(pivot, plot_dir / "encoding_loro_best_alpha_heatmap.png",
                  "Selected ridge alpha", "log10 alpha",
                  cmap="magma", center_zero=False, fmt=".1f")
    created.append(plot_dir / "encoding_loro_best_alpha_heatmap.png")

    # Additional permutation/FDR figures.
    if "frac_voxels_q05" in df.columns and df["frac_voxels_q05"].notna().any():
        pivot = df.pivot(index="model", columns="roi", values="frac_voxels_q05").reindex(
            index=model_order, columns=roi_order
        )
        pivot = pivot.rename(index=model_display, columns=roi_display)
        _plot_heatmap(pivot, plot_dir / "encoding_loro_fdr_q05_fraction.png",
                      "Fraction of voxels passing FDR q < 0.05", "q < 0.05 fraction",
                      cmap="viridis", center_zero=False, fmt=".2f")
        created.append(plot_dir / "encoding_loro_fdr_q05_fraction.png")

    if "frac_voxels_p05" in df.columns and df["frac_voxels_p05"].notna().any():
        pivot = df.pivot(index="model", columns="roi", values="frac_voxels_p05").reindex(
            index=model_order, columns=roi_order
        )
        pivot = pivot.rename(index=model_display, columns=roi_display)
        _plot_heatmap(pivot, plot_dir / "encoding_loro_perm_p05_fraction.png",
                      "Fraction of voxels with permutation p < 0.05", "p < 0.05 fraction",
                      cmap="viridis", center_zero=False, fmt=".2f")
        created.append(plot_dir / "encoding_loro_perm_p05_fraction.png")

    if "min_q_fdr_one_sided" in df.columns and df["min_q_fdr_one_sided"].notna().any():
        tmp = df.copy()
        tmp["neglog10_min_q"] = -np.log10(np.clip(tmp["min_q_fdr_one_sided"].astype(float), 1e-300, 1.0))
        pivot = tmp.pivot(index="model", columns="roi", values="neglog10_min_q").reindex(
            index=model_order, columns=roi_order
        )
        pivot = pivot.rename(index=model_display, columns=roi_display)
        _plot_heatmap(pivot, plot_dir / "encoding_loro_min_q_neglog10_heatmap.png",
                      "Strongest voxel evidence: -log10(min FDR q)", "-log10 min q",
                      cmap="magma", center_zero=False, fmt=".1f")
        created.append(plot_dir / "encoding_loro_min_q_neglog10_heatmap.png")

    if "mean_r_minus_null95_mean_r" in df.columns and df["mean_r_minus_null95_mean_r"].notna().any():
        pivot = df.pivot(index="model", columns="roi", values="mean_r_minus_null95_mean_r").reindex(
            index=model_order, columns=roi_order
        )
        pivot = pivot.rename(index=model_display, columns=roi_display)
        _plot_heatmap(pivot, plot_dir / "encoding_loro_mean_r_vs_null95_heatmap.png",
                      "Observed mean r minus permutation 95th percentile", "mean r - null95",
                      cmap="PiYG", center_zero=True, fmt="+.3f")
        created.append(plot_dir / "encoding_loro_mean_r_vs_null95_heatmap.png")

    # PDF bundle.
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


# ── main pipeline ─────────────────────────────────────────────────────────────
def run(args: argparse.Namespace) -> None:

    # ── discover feature files ────────────────────────────────────────────────
    feat_dir  = args.out_root / args.subject_game / "features"
    feat_files = sorted(feat_dir.glob("features_*.npz"))
    if not feat_files:
        log(f"[error] no feature npz found in {feat_dir}")
        sys.exit(1)
    log(f"Found {len(feat_files)} feature files")

    # ── load ROI masks once ───────────────────────────────────────────────────
    roi_masks: Dict[str, np.ndarray] = {}
    fmri_shape_ref: Optional[Tuple] = None

    # we need a reference fMRI shape to validate masks — load first available
    for ff in feat_files:
        run_label = ff.stem.replace("features_", "")
        meta      = parse_run_label(run_label)
        if meta is None:
            continue
        fp = find_fmri_path(args.fmri_root, meta["subject"], meta["session"], meta["block"])
        if fp is not None:
            import nibabel as nib
            fmri_shape_ref = nib.load(str(fp)).shape
            break

    if fmri_shape_ref is None:
        log("[error] could not load any fMRI file to get reference shape")
        sys.exit(1)

    for roi_name, rel_path in ROI_MASKS.items():
        mask_path = args.atlas_root / rel_path
        if not mask_path.exists():
            log(f"  [warn] mask not found: {mask_path}")
            continue
        try:
            roi_masks[roi_name] = load_roi_mask(mask_path, fmri_shape_ref)
            log(f"  [mask] {roi_name}: {roi_masks[roi_name].sum()} voxels")
        except Exception as exc:
            log(f"  [warn] mask load error {roi_name}: {exc}")

    if not roi_masks:
        log("[error] no ROI masks loaded")
        sys.exit(1)

    # ── load data per run ─────────────────────────────────────────────────────
    run_data: List[Dict] = []   # {run_label, meta, features, bold}

    for ff in feat_files:
        run_label = ff.stem.replace("features_", "")
        meta      = parse_run_label(run_label)
        if meta is None:
            log(f"  [skip] cannot parse run label: {run_label}")
            continue

        fp = find_fmri_path(args.fmri_root, meta["subject"], meta["session"], meta["block"])
        if fp is None:
            log(f"  [skip] fMRI not found for {run_label}")
            continue

        import nibabel as nib
        img    = nib.load(str(fp))
        n_vols = img.shape[3]
        vol_start = FMRI_TRIM
        vol_stop  = min(n_vols - FMRI_TRIM, FMRI_TRIM + N_ANALYSIS)
        n_tr_use  = vol_stop - vol_start

        if n_tr_use < 60:
            log(f"  [skip] too few TRs ({n_tr_use}) for {run_label}")
            continue

        # BOLD patterns
        bold: Dict[str, np.ndarray] = {}
        for roi_name, mask in roi_masks.items():
            try:
                patterns = extract_bold_patterns(fp, mask, vol_start, vol_stop,
                                                 max_voxels=args.max_roi_voxels)
                if patterns is not None:
                    bold[roi_name] = patterns
            except Exception as exc:
                log(f"  [warn] {run_label} {roi_name}: {exc}")

        if not bold:
            log(f"  [skip] no BOLD patterns for {run_label}")
            continue

        # features
        feats = load_run_features(ff)
        if feats is None:
            continue

        # align lengths
        feat_t = min(v.shape[0] for v in feats.values())
        bold_t = min(v.shape[0] for v in bold.values())
        t_use  = min(feat_t, bold_t)

        feats = {k: v[:t_use] for k, v in feats.items()}
        bold  = {k: v[:t_use] for k, v in bold.items()}

        run_data.append({
            "run_label": run_label,
            "meta":      meta,
            "features":  feats,
            "bold":      bold,
        })
        log(f"  [loaded] {run_label}  T={t_use}  rois={list(bold)}")

    if len(run_data) < 2:
        log("[error] need at least 2 runs for leave-one-run-out")
        sys.exit(1)

    log(f"\nReady: {len(run_data)} runs")

    # ── leave-one-run-out encoding ────────────────────────────────────────────
    out_dir = args.out_root / args.subject_game / "encoding_loro"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "encoding_loro_manifest.csv"

    first_meta = run_data[0]["meta"]
    all_models = sorted(set(k for rd in run_data for k in rd["features"]))
    all_rois   = sorted(set(k for rd in run_data for k in rd["bold"]))

    rows = []
    voxel_stats: Dict[str, np.ndarray] = {}
    voxel_key_rows: List[Dict[str, str]] = []

    for roi_name in all_rois:
        Y_runs = []
        for rd in run_data:
            if roi_name in rd["bold"]:
                Y_runs.append(rd["bold"][roi_name])
            else:
                Y_runs = []
                break
        if len(Y_runs) != len(run_data):
            log(f"  [skip roi] {roi_name}: missing in some runs")
            continue

        # ensure consistent voxel count across runs
        n_vox = min(y.shape[1] for y in Y_runs)
        Y_runs = [y[:, :n_vox] for y in Y_runs]

        for model_name in all_models:
            X_runs = []
            for rd in run_data:
                if model_name in rd["features"]:
                    X_runs.append(rd["features"][model_name])
                else:
                    X_runs = []
                    break
            if len(X_runs) != len(run_data):
                log(f"  [skip model] {model_name}: missing in some runs")
                continue

            # align T per run
            XY = []
            for x, y in zip(X_runs, Y_runs):
                t = min(x.shape[0], y.shape[0])
                XY.append((x[:t], y[:t]))
            X_runs_aligned = [xy[0] for xy in XY]
            Y_runs_aligned = [xy[1] for xy in XY]

            label = f"{roi_name}:{model_name}"
            log(f"\n[encoding] {label}")

            rs, best_alpha = run_loro_encoding(
                X_runs_aligned, Y_runs_aligned,
                alphas=RIDGE_ALPHAS,
                label=label,
            )

            if rs is None or rs.size == 0:
                continue

            perm: Dict[str, np.ndarray] = {"rs": rs}
            do_perm = (not args.skip_permutation) and args.n_perm > 0
            if do_perm:
                combo_seed = int(args.perm_seed + 1009 * len(rows))
                perm = run_loro_permutation(
                    X_runs_aligned,
                    Y_runs_aligned,
                    best_alpha,
                    n_perm=args.n_perm,
                    block_size=args.perm_block_size,
                    rng_seed=combo_seed,
                    label=label,
                )
                rs = perm["rs"]

            p_one = perm.get("p_one_sided")
            p_two = perm.get("p_two_sided")
            q_one = perm.get("q_fdr_one_sided")
            null_mean_scores = perm.get("null_mean_scores")

            finite_rs = np.isfinite(rs)
            n_valid = int(finite_rs.sum())
            n_p05 = int(np.sum(np.isfinite(p_one) & (p_one < 0.05))) if p_one is not None else 0
            n_q05 = int(np.sum(np.isfinite(q_one) & (q_one < 0.05))) if q_one is not None else 0
            frac_p05 = float(n_p05 / n_valid) if n_valid else float("nan")
            frac_q05 = float(n_q05 / n_valid) if n_valid else float("nan")

            def _finite_stat(arr: Optional[np.ndarray], fn, default: float = float("nan")) -> float:
                if arr is None:
                    return default
                vals = np.asarray(arr, dtype=np.float64)
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    return default
                return float(fn(vals))

            null_mean = _finite_stat(null_mean_scores, np.mean)
            null_p95 = _finite_stat(null_mean_scores, lambda v: np.percentile(v, 95))
            mean_r = float(np.nanmean(rs))
            median_r = float(np.nanmedian(rs))

            if do_perm and p_one is not None and q_one is not None:
                safe_key = re.sub(r"[^A-Za-z0-9_]+", "_", f"{roi_name}__{model_name}")
                voxel_stats[f"{safe_key}__r"] = rs.astype(np.float32)
                voxel_stats[f"{safe_key}__p_one_sided"] = p_one.astype(np.float32)
                if p_two is not None:
                    voxel_stats[f"{safe_key}__p_two_sided"] = p_two.astype(np.float32)
                voxel_stats[f"{safe_key}__q_fdr_one_sided"] = q_one.astype(np.float32)
                voxel_key_rows.append({
                    "roi": roi_name,
                    "model": model_name,
                    "key_prefix": safe_key,
                })

            rows.append({
                "run_label":        "ALL_RUNS",
                "subject":          first_meta["subject"],
                "session":          -1,
                "block":            -1,
                "game":             first_meta["game"],
                "roi":              roi_name,
                "model":            model_name,
                "cv_scheme":        "loro_alpha_cv",
                "best_alpha":       best_alpha,
                "mean_r":           mean_r,
                "median_r":         median_r,
                "n_voxels":         int(len(rs)),
                "n_voxels_finite":  n_valid,
                "n_voxels_positive": int(np.sum(rs > 0)),
                "n_perm":           int(args.n_perm if do_perm else 0),
                "perm_block_size":  int(args.perm_block_size if do_perm else 0),
                "n_voxels_p05":     n_p05,
                "frac_voxels_p05":  frac_p05,
                "n_voxels_q05":     n_q05,
                "frac_voxels_q05":  frac_q05,
                "min_p_one_sided":  _finite_stat(p_one, np.min),
                "median_p_one_sided": _finite_stat(p_one, np.median),
                "min_q_fdr_one_sided": _finite_stat(q_one, np.min),
                "median_q_fdr_one_sided": _finite_stat(q_one, np.median),
                "null_mean_r":      null_mean,
                "null_p95_mean_r":  null_p95,
                "mean_r_minus_null95_mean_r": (
                    mean_r - null_p95 if np.isfinite(null_p95) else float("nan")
                ),
            })
            log(
                f"  -> mean_r={rows[-1]['mean_r']:.5f}  best_alpha={best_alpha:.3g} "
                f"q05={rows[-1]['n_voxels_q05']}/{rows[-1]['n_voxels_finite']}"
            )

    # ── save CSV ──────────────────────────────────────────────────────────────
    if not rows:
        log("[warn] no results to save")
        return

    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log(f"\nSaved {len(rows)} rows → {out_csv}")

    if voxel_stats:
        stats_path = out_dir / "encoding_loro_voxel_stats.npz"
        np.savez_compressed(stats_path, **voxel_stats)
        log(f"Saved voxelwise r/p/q arrays → {stats_path}")

        keys_path = out_dir / "encoding_loro_voxel_stats_keys.csv"
        with open(keys_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["roi", "model", "key_prefix"])
            writer.writeheader()
            writer.writerows(voxel_key_rows)
        log(f"Saved voxel stats key map → {keys_path}")

    if not args.skip_plots:
        try:
            generate_plots(out_csv, out_dir)
        except Exception as exc:
            log(f"[warn] plot generation failed: {exc}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-game", default="sub001_game2",
                        help="Subdirectory under out_root, e.g. sub001_game2")
    parser.add_argument("--fmri-root", type=Path, default=DEFAULT_FMRI_ROOT)
    parser.add_argument("--atlas-root", type=Path, default=DEFAULT_ATLAS_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--max-roi-voxels", type=int, default=MAX_ROI_VOXELS)
    parser.add_argument("--alphas-log", nargs=2, type=float, default=[-2, 5],
                        metavar=("LO", "HI"),
                        help="logspace range for alpha grid (default: -2 5)")
    parser.add_argument("--n-alphas", type=int, default=15)
    parser.add_argument("--n-perm", type=int, default=N_PERM,
                        help="Encoding-specific block permutations per model/ROI (default: 1000)")
    parser.add_argument("--perm-block-size", type=int, default=PERM_BLOCK,
                        help="Held-out BOLD shuffle block size in TRs (default: 40)")
    parser.add_argument("--perm-seed", type=int, default=0)
    parser.add_argument("--skip-permutation", action="store_true",
                        help="Skip encoding-specific block permutation/FDR")
    parser.add_argument("--skip-plots", action="store_true",
                        help="Do not generate summary plots after writing CSV")
    parser.add_argument("--plot-only", action="store_true",
                        help="Regenerate plots from an existing encoding_loro_manifest.csv and exit")
    args = parser.parse_args()

    if args.plot_only:
        out_dir = args.out_root / args.subject_game / "encoding_loro"
        generate_plots(out_dir / "encoding_loro_manifest.csv", out_dir)
        return

    global RIDGE_ALPHAS
    RIDGE_ALPHAS = np.logspace(args.alphas_log[0], args.alphas_log[1], args.n_alphas)
    log(f"Alpha grid: {RIDGE_ALPHAS.round(4)}")

    run(args)


if __name__ == "__main__":
    main()
