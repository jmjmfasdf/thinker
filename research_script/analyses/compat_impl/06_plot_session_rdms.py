#!/usr/bin/env python3
"""
Plot per-session Thinker and neural representational similarity matrices.

For each fMRI analysis unit (trace session -> fMRI subject folder, block ->
fMRI Session), this script writes a 5-panel diagnostic figure:

1. Thinker im_vectors similarity matrix binned to fMRI volumes.
2. Thinker im_vp_vectors similarity matrix binned to fMRI volumes.
3. ROI neural similarity matrix across the same fMRI volumes.
4. Local correlation map: neural similarity vs im_vectors similarity.
5. Local correlation map: neural similarity vs im_vp_vectors similarity.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib-cache"
os.environ["XDG_CACHE_HOME"] = "/tmp"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import convolve


ROOT = Path(__file__).resolve().parent.parent
REP_SCRIPT = Path(__file__).resolve().parent / "06_representational_mechanism.py"
DEFAULT_OUT_DIR = (
    Path(__file__).resolve().parent
    / "outputs"
    / "06_representational_mechanism"
    / "figures"
    / "session_rdms"
)
DEFAULT_VIS_MASK = (
    Path(__file__).resolve().parent
    / "outputs"
    / "06_representational_mechanism"
    / "atlas"
    / "templateflow_schaefer2018"
    / "ants_mni_2p5mm_masks"
    / "masks"
    / "network"
    / "roi-network-Vis_mask.nii.gz"
)

EPS = 1e-12


def load_rep_module():
    spec = importlib.util.spec_from_file_location("repmech06", REP_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import helper module: {REP_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


REP = load_rep_module()


@dataclass
class RealStepReps:
    rows: pd.DataFrame
    im_vectors: np.ndarray
    im_vp_vectors: np.ndarray


def align_rows(rows: Sequence[np.ndarray]) -> np.ndarray:
    usable = [np.asarray(row, dtype=np.float32).reshape(-1) for row in rows if np.asarray(row).size > 0]
    if not usable:
        return np.empty((0, 0), dtype=np.float32)
    dim = min(row.size for row in usable)
    return np.vstack([row[:dim] for row in usable]).astype(np.float32)


def extract_file_real_step_reps(meta, max_real_steps_per_file: int | None) -> RealStepReps:
    print(f"  [load] {meta.path}", flush=True)
    data = REP.load_npy_dict(meta.path)
    status = np.asarray(data["status"]).reshape(-1)
    im_raw = data.get("im_vectors")
    im_vp_raw = data.get("im_vp_vectors")
    if im_raw is None or im_vp_raw is None:
        raise KeyError(f"Missing im_vectors or im_vp_vectors in {meta.path}")

    n = min(len(status), len(im_raw), len(im_vp_raw))
    real_idx = np.flatnonzero(status[:n] == 0)
    if max_real_steps_per_file is not None:
        real_idx = real_idx[: max_real_steps_per_file]

    rows: List[Dict[str, object]] = []
    im_rows: List[np.ndarray] = []
    im_vp_rows: List[np.ndarray] = []
    for real_pos, idx in enumerate(real_idx):
        im_vec = REP.pool_vector(im_raw[int(idx)])
        im_vp_vec = REP.pool_vector(im_vp_raw[int(idx)])
        if im_vec.size == 0 or im_vp_vec.size == 0:
            continue
        rows.append(
            {
                "file": str(meta.path),
                "subject": meta.subject,
                "session": meta.session,
                "block": meta.block,
                "game": meta.game,
                "chunk": meta.chunk,
                "fmri_subject": meta.fmri_subject,
                "fmri_session": meta.fmri_session,
                "analysis_unit": meta.analysis_unit,
                "real_pos_file": int(real_pos),
                "global_idx": int(idx),
            }
        )
        im_rows.append(im_vec)
        im_vp_rows.append(im_vp_vec)

    return RealStepReps(
        rows=pd.DataFrame(rows),
        im_vectors=align_rows(im_rows),
        im_vp_vectors=align_rows(im_vp_rows),
    )


def build_real_step_reps(metas: Sequence[object], max_real_steps_per_file: int | None) -> RealStepReps:
    frames: List[pd.DataFrame] = []
    im_blocks: List[np.ndarray] = []
    im_vp_blocks: List[np.ndarray] = []
    for meta in metas:
        reps = extract_file_real_step_reps(meta, max_real_steps_per_file)
        if reps.rows.empty:
            continue
        frames.append(reps.rows)
        im_blocks.append(reps.im_vectors)
        im_vp_blocks.append(reps.im_vp_vectors)
    if not frames:
        raise RuntimeError("No real-step im_vectors/im_vp_vectors were extracted.")

    min_im_dim = min(block.shape[1] for block in im_blocks if block.size)
    min_vp_dim = min(block.shape[1] for block in im_vp_blocks if block.size)
    df = pd.concat(frames, ignore_index=True)
    im = np.vstack([block[:, :min_im_dim] for block in im_blocks]).astype(np.float32)
    im_vp = np.vstack([block[:, :min_vp_dim] for block in im_vp_blocks]).astype(np.float32)

    df["_row"] = np.arange(len(df), dtype=int)
    df = df.sort_values(["subject", "session", "game", "block", "chunk", "real_pos_file"]).reset_index(drop=True)
    order = df["_row"].to_numpy(dtype=int)
    df = df.drop(columns=["_row"])
    return RealStepReps(rows=df, im_vectors=im[order], im_vp_vectors=im_vp[order])


def group_metas_by_analysis_unit(metas: Sequence[object]) -> Dict[str, List[object]]:
    grouped: Dict[str, List[object]] = {}
    for meta in metas:
        grouped.setdefault(meta.analysis_unit, []).append(meta)
    for key in grouped:
        grouped[key] = sorted(grouped[key], key=lambda m: (m.chunk, str(m.path)))
    return dict(sorted(grouped.items()))


def choose_rows(n_rows: int, max_rows: int) -> np.ndarray:
    if max_rows <= 0 or n_rows <= max_rows:
        return np.arange(n_rows, dtype=int)
    return np.unique(np.linspace(0, n_rows - 1, max_rows, dtype=int))


def average_patterns_to_volume_bins(patterns: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(patterns, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D pattern matrix, got {arr.shape}")
    n_rows, n_features = arr.shape
    if n_bins <= 0:
        return np.empty((0, n_features), dtype=np.float32), np.empty(0, dtype=np.int32)
    if n_rows == 0:
        return np.full((n_bins, n_features), np.nan, dtype=np.float32), np.zeros(n_bins, dtype=np.int32)

    bin_index = np.floor(np.arange(n_rows, dtype=np.float64) * n_bins / n_rows).astype(int)
    bin_index = np.clip(bin_index, 0, n_bins - 1)
    sums = np.zeros((n_bins, n_features), dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int32)
    np.add.at(sums, bin_index, np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0))
    np.add.at(counts, bin_index, 1)
    binned = sums / np.maximum(counts[:, None], 1)

    empty = counts == 0
    if np.any(empty):
        filled_x = np.flatnonzero(~empty)
        if filled_x.size == 0:
            binned[:] = 0.0
        elif filled_x.size == 1:
            binned[empty] = binned[filled_x[0]]
        else:
            x = np.arange(n_bins)
            for col in range(n_features):
                binned[empty, col] = np.interp(x[empty], filled_x, binned[filled_x, col])
    return binned.astype(np.float32), counts


def similarity_matrix(patterns: np.ndarray) -> np.ndarray:
    arr = np.asarray(patterns, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return np.empty((0, 0), dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = arr - arr.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / np.where(denom <= EPS, 1.0, denom)
    sim = arr @ arr.T
    return np.clip(sim, -1.0, 1.0).astype(np.float32)


def upper_triangle_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape or a.shape[0] < 3:
        return np.nan
    tri = np.triu_indices(a.shape[0], k=1)
    aa = a[tri].astype(np.float64)
    bb = b[tri].astype(np.float64)
    valid = np.isfinite(aa) & np.isfinite(bb)
    aa = aa[valid]
    bb = bb[valid]
    if len(aa) < 3 or np.std(aa) <= EPS or np.std(bb) <= EPS:
        return np.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def local_correlation_map(a: np.ndarray, b: np.ndarray, radius: int) -> np.ndarray:
    if a.shape != b.shape:
        raise ValueError(f"Matrix shape mismatch: {a.shape} vs {b.shape}")
    radius = max(1, int(radius))
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    valid = np.isfinite(aa) & np.isfinite(bb)
    aa = np.where(valid, aa, 0.0)
    bb = np.where(valid, bb, 0.0)
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.float64)

    count = convolve(valid.astype(np.float64), kernel, mode="constant", cval=0.0)
    sum_a = convolve(aa, kernel, mode="constant", cval=0.0)
    sum_b = convolve(bb, kernel, mode="constant", cval=0.0)
    sum_aa = convolve(aa * aa, kernel, mode="constant", cval=0.0)
    sum_bb = convolve(bb * bb, kernel, mode="constant", cval=0.0)
    sum_ab = convolve(aa * bb, kernel, mode="constant", cval=0.0)

    cov = sum_ab - (sum_a * sum_b / np.maximum(count, 1.0))
    var_a = sum_aa - (sum_a * sum_a / np.maximum(count, 1.0))
    var_b = sum_bb - (sum_b * sum_b / np.maximum(count, 1.0))
    denom = np.sqrt(np.maximum(var_a, 0.0) * np.maximum(var_b, 0.0))
    out = cov / np.where(denom <= EPS, np.nan, denom)
    out[count < 6] = np.nan
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def clean_roi_name(path: Path, roi_name: str) -> str:
    if roi_name:
        return roi_name
    stem = path.name
    for suffix in [".nii.gz", ".nii"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def plot_five_panel(
    *,
    analysis_unit: str,
    roi_name: str,
    step_labels: np.ndarray,
    im_sim: np.ndarray,
    im_vp_sim: np.ndarray,
    neural_sim: np.ndarray,
    corr_im: np.ndarray,
    corr_im_vp: np.ndarray,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(23, 4.7), constrained_layout=True)
    panels = [
        ("Thinker im_vectors RDM", im_sim, "coolwarm", -1, 1),
        ("Thinker im_vp_vectors RDM", im_vp_sim, "coolwarm", -1, 1),
        (f"{roi_name} neural RDM", neural_sim, "coolwarm", -1, 1),
        ("Local corr: neural x im_vectors", corr_im, "coolwarm", -1, 1),
        ("Local corr: neural x im_vp_vectors", corr_im_vp, "coolwarm", -1, 1),
    ]
    n = len(step_labels)
    tick_pos = np.unique(np.asarray([0, max(0, n // 2), max(0, n - 1)], dtype=int))
    tick_labels = [str(int(step_labels[i])) for i in tick_pos]
    for ax, (title, matrix, cmap, vmin, vmax) in zip(axes, panels):
        im = ax.imshow(matrix, origin="lower", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("fMRI volume / binned real steps")
        ax.set_ylabel("fMRI volume / binned real steps")
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(tick_pos)
        ax.set_yticklabels(tick_labels, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    r_im = upper_triangle_corr(neural_sim, im_sim)
    r_im_vp = upper_triangle_corr(neural_sim, im_vp_sim)
    fig.suptitle(
        f"{analysis_unit} | ROI={roi_name} | n={n} fMRI volume bins | "
        f"global r(neural, im)={r_im:.3f}, r(neural, im_vp)={r_im_vp:.3f}",
        fontsize=12,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_analysis_unit(
    df: pd.DataFrame,
    im_vectors: np.ndarray,
    im_vp_vectors: np.ndarray,
    *,
    analysis_unit: str,
    args: argparse.Namespace,
) -> Dict[str, object]:
    sg = df[df["analysis_unit"] == analysis_unit].sort_values(["chunk", "real_pos_file"])
    if sg.empty:
        return {"analysis_unit": analysis_unit, "status": "no_rows"}

    first = sg.iloc[0]
    fmri_run = REP.discover_fmri_run(
        args.fmri_root,
        str(first["fmri_subject"]),
        int(first["fmri_session"]),
        args.fmri_image,
    )
    if fmri_run is None:
        return {"analysis_unit": analysis_unit, "status": "no_matching_fmri"}

    start, stop, n_analysis, mode = REP.analysis_volume_window(int(fmri_run["n_volumes"]), args.fmri_trim_volumes)
    if n_analysis < 3:
        return {"analysis_unit": analysis_unit, "status": mode}

    row_idx_all = sg.index.to_numpy(dtype=int)
    im_binned, im_bin_counts = average_patterns_to_volume_bins(im_vectors[row_idx_all], n_analysis)
    im_vp_binned, im_vp_bin_counts = average_patterns_to_volume_bins(im_vp_vectors[row_idx_all], n_analysis)

    sample_pos = choose_rows(n_analysis, args.max_steps_per_plot)
    im_patterns = im_binned[sample_pos]
    im_vp_patterns = im_vp_binned[sample_pos]
    volume_positions = np.arange(start, stop, dtype=int)[sample_pos]
    step_labels = volume_positions
    neural_patterns, _, raw_voxels, used_voxels = REP.extract_roi_patterns(
        Path(str(fmri_run["fmri_path"])),
        roi_mask=args.roi_mask,
        roi_threshold=args.roi_threshold,
        volume_indices=volume_positions,
        max_roi_voxels=args.max_roi_voxels,
    )
    if neural_patterns.shape[0] != len(sample_pos):
        return {"analysis_unit": analysis_unit, "status": "neural_pattern_length_mismatch"}

    im_sim = similarity_matrix(im_patterns)
    im_vp_sim = similarity_matrix(im_vp_patterns)
    neural_sim = similarity_matrix(neural_patterns)
    corr_im = local_correlation_map(neural_sim, im_sim, args.local_corr_radius)
    corr_im_vp = local_correlation_map(neural_sim, im_vp_sim, args.local_corr_radius)

    roi_name = clean_roi_name(args.roi_mask, args.roi_name)
    out_name = f"rdm_5panel_{analysis_unit}_roi-{roi_name}.png"
    out_path = args.output_dir / out_name
    plot_five_panel(
        analysis_unit=analysis_unit,
        roi_name=roi_name,
        step_labels=step_labels,
        im_sim=im_sim,
        im_vp_sim=im_vp_sim,
        neural_sim=neural_sim,
        corr_im=corr_im,
        corr_im_vp=corr_im_vp,
        out_path=out_path,
    )
    return {
        "analysis_unit": analysis_unit,
        "subject": int(first["subject"]),
        "trace_session": int(first["session"]),
        "block": int(first["block"]),
        "game": int(first["game"]),
        "fmri_path": str(fmri_run["fmri_path"]),
        "roi_mask": str(args.roi_mask),
        "roi_name": roi_name,
        "n_real_steps_total": int(len(row_idx_all)),
        "n_fmri_volumes_after_trim": int(n_analysis),
        "n_fmri_volume_bins_plotted": int(len(sample_pos)),
        "min_real_steps_per_fmri_bin": int(np.min(im_bin_counts)) if im_bin_counts.size else 0,
        "max_real_steps_per_fmri_bin": int(np.max(im_bin_counts)) if im_bin_counts.size else 0,
        "empty_im_vector_bins": int(np.sum(im_bin_counts == 0)),
        "empty_im_vp_vector_bins": int(np.sum(im_vp_bin_counts == 0)),
        "n_roi_voxels_raw": int(raw_voxels),
        "n_roi_voxels_used": int(used_voxels),
        "global_corr_neural_im_vectors": upper_triangle_corr(neural_sim, im_sim),
        "global_corr_neural_im_vp_vectors": upper_triangle_corr(neural_sim, im_vp_sim),
        "output_png": str(out_path),
        "status": "ok",
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=REP.DEFAULT_INPUT_ROOT)
    parser.add_argument("--fmri-root", type=Path, default=REP.DEFAULT_FMRI_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--subjects", default=None)
    parser.add_argument("--sessions", default=None)
    parser.add_argument("--games", default=None)
    parser.add_argument("--game-session-offset", type=int, default=None)
    parser.add_argument("--fmri-image", default="s5_wfiltered_func_data.nii")
    parser.add_argument("--fmri-trim-volumes", type=int, default=REP.DEFAULT_FMRI_TRIM_VOLUMES)
    parser.add_argument("--roi-mask", type=Path, default=DEFAULT_VIS_MASK)
    parser.add_argument("--roi-name", default="")
    parser.add_argument("--roi-threshold", type=float, default=0.0)
    parser.add_argument("--max-roi-voxels", type=int, default=5000)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-real-steps-per-file", type=int, default=None)
    parser.add_argument(
        "--max-steps-per-plot",
        type=int,
        default=0,
        help=(
            "Maximum fMRI volume bins to show after binning Thinker real steps. "
            "Use 0 for all trimmed fMRI volumes, usually 480."
        ),
    )
    parser.add_argument("--local-corr-radius", type=int, default=6)
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    metas = REP.gather_trace_files(
        args.input_root,
        subjects=REP.parse_int_list(args.subjects),
        sessions=REP.parse_int_list(args.sessions),
        games=REP.parse_int_list(args.games),
        game_session_offset=args.game_session_offset,
        max_files=args.max_files,
    )
    if not metas:
        raise FileNotFoundError(f"No trace files found under {args.input_root}")
    grouped_metas = group_metas_by_analysis_unit(metas)

    rows: List[Dict[str, object]] = []
    for analysis_unit, unit_metas in grouped_metas.items():
        print(f"[unit] {analysis_unit}: {len(unit_metas)} trace file(s)", flush=True)
        reps = build_real_step_reps(unit_metas, args.max_real_steps_per_file)
        print(f"[plot] {analysis_unit}: {len(reps.rows)} real-step rows", flush=True)
        row = plot_analysis_unit(
            reps.rows,
            reps.im_vectors,
            reps.im_vp_vectors,
            analysis_unit=analysis_unit,
            args=args,
        )
        print(f"  {row.get('status')}: {row.get('output_png', '')}", flush=True)
        rows.append(row)
        del reps
        gc.collect()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary_path = args.output_dir / "session_rdm_plot_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[done] wrote {len(summary)} rows to {summary_path}")


if __name__ == "__main__":
    main()
