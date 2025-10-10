"""Aggregate exported Thinker metrics and recreate correlation plots."""

import argparse
import glob
import os
from collections import defaultdict

import numpy as np
from scipy import stats

import fig_pong


CORRELATION_KEYS = [
    "noop_freq_vs_planning_depth",
    "noop_freq_vs_image_similarity",
    "real_step_image_sim_vs_planning_depth",
    "real_step_image_sim_vs_action_diversity",
    "noop_freq_vs_action_diversity",
    "planning_depth_vs_action_diversity",
    "real_step_image_sim_vs_imagination_diversity",
    "noop_freq_vs_imagination_diversity",
    "planning_depth_vs_imagination_diversity",
]


def _parse_metrics_filename(filename: str):
    base, ext = os.path.splitext(filename)
    if ext in {".npz", ".npy"}:
        return fig_pong.parse_filename(base + ".npy")
    return fig_pong.parse_filename(filename)


def _load_metrics_file(path: str) -> dict:
    """Load a metrics dictionary saved via export_step_metrics."""
    try:
        if path.endswith(".npz"):
            with np.load(path, allow_pickle=True) as data:
                return {key: data[key] for key in data.files}
        obj = np.load(path, allow_pickle=True)
        if isinstance(obj, np.ndarray) and obj.shape == () and obj.dtype == object:
            return obj.item()
        if isinstance(obj, dict):
            return obj
    except Exception as exc:  # pragma: no cover - diagnostics
        raise RuntimeError(f"Failed to load metrics from {path}: {exc}") from exc
    raise RuntimeError(f"Unsupported metrics container in {path}")


def _ensure_1d_array(values, dtype=float) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype)
    if arr.ndim == 1:
        return arr
    return arr.reshape(-1)


def _polyfit_r2(x_vals: np.ndarray, y_vals: np.ndarray) -> float:
    if len(x_vals) < 2 or len(y_vals) < 2:
        return 0.0
    if np.allclose(x_vals, x_vals[0]) or np.allclose(y_vals, y_vals[0]):
        return 0.0
    try:
        with np.errstate(all="ignore"):
            coeffs = np.polyfit(x_vals, y_vals, 1)
            y_pred = np.poly1d(coeffs)(x_vals)
    except Exception:
        y_pred = np.full_like(y_vals, np.mean(y_vals))
    return float(fig_pong.calculate_r_squared(y_vals, y_pred))


def _sliding_window_correlation(x_data: np.ndarray, y_data: np.ndarray, window_size: int, stride: int, x_func):
    if len(x_data) < window_size or len(y_data) < window_size:
        return None
    windows_x, windows_y = fig_pong.calculate_sliding_window_correlation(
        x_data, y_data, window_size, stride, x_func
    )
    if len(windows_x) <= 1:
        return None

    finite_mask = np.isfinite(windows_x) & np.isfinite(windows_y)
    windows_x = windows_x[finite_mask]
    windows_y = windows_y[finite_mask]
    if len(windows_x) <= 1:
        return None

    if np.allclose(windows_x, windows_x[0]) or np.allclose(windows_y, windows_y[0]):
        return None

    try:
        r, p = stats.pearsonr(windows_x, windows_y)
    except Exception:
        return None

    r2 = _polyfit_r2(windows_x, windows_y)
    return {"r": float(r), "r2": float(r2), "p": float(p), "n": len(windows_x)}


def _basic_correlation(x_vals: np.ndarray, y_vals: np.ndarray):
    finite_mask = np.isfinite(x_vals) & np.isfinite(y_vals)
    x_vals = x_vals[finite_mask]
    y_vals = y_vals[finite_mask]
    if len(x_vals) <= 1:
        return None

    if np.allclose(x_vals, x_vals[0]) or np.allclose(y_vals, y_vals[0]):
        return None

    try:
        r, p = stats.pearsonr(x_vals, y_vals)
    except Exception:
        return None

    r2 = _polyfit_r2(x_vals, y_vals)
    return {"r": float(r), "r2": float(r2), "p": float(p), "n": len(x_vals)}


def calculate_metrics_correlations(metrics: dict, window_size: int, stride: int):
    planning_depth = _ensure_1d_array(metrics["planning_depth"], dtype=float)
    action_diversity = _ensure_1d_array(metrics["action_diversity"], dtype=float)
    imagination_diversity = _ensure_1d_array(metrics["imagination_diversity"], dtype=float)
    imagination_similarity = _ensure_1d_array(metrics["imagination_similarity"], dtype=float)
    actions = _ensure_1d_array(metrics["action"], dtype=float)

    lengths = {arr.shape[0] for arr in (
        planning_depth,
        action_diversity,
        imagination_diversity,
        imagination_similarity,
        actions,
    )}
    lengths.discard(0)
    if len(lengths) > 1:
        raise ValueError("Metric arrays must share the same length per file")

    valid_mask = (
        np.isfinite(planning_depth)
        & np.isfinite(action_diversity)
        & np.isfinite(imagination_diversity)
        & np.isfinite(imagination_similarity)
    )

    if not np.any(valid_mask):
        return {key: None for key in CORRELATION_KEYS}

    valid_depths = planning_depth[valid_mask]
    valid_action_div = action_diversity[valid_mask]
    valid_diversities = imagination_diversity[valid_mask]
    valid_similarities = imagination_similarity[valid_mask]
    valid_actions = actions[valid_mask]

    window_results = {}
    noop_freq_fn = lambda window: np.sum(window == 0) / len(window)

    window_results["noop_freq_vs_planning_depth"] = _sliding_window_correlation(
        valid_actions, valid_depths, window_size, stride, noop_freq_fn
    )
    window_results["noop_freq_vs_image_similarity"] = _sliding_window_correlation(
        valid_actions, valid_similarities, window_size, stride, noop_freq_fn
    )
    window_results["noop_freq_vs_action_diversity"] = _sliding_window_correlation(
        valid_actions, valid_action_div, window_size, stride, noop_freq_fn
    )

    window_results["planning_depth_vs_action_diversity"] = _sliding_window_correlation(
        valid_depths, valid_action_div, window_size, stride, np.mean
    )
    window_results["noop_freq_vs_imagination_diversity"] = _sliding_window_correlation(
        valid_actions, valid_diversities, window_size, stride, noop_freq_fn
    )
    window_results["planning_depth_vs_imagination_diversity"] = _sliding_window_correlation(
        valid_depths, valid_diversities, window_size, stride, np.mean
    )

    basic_results = {
        "real_step_image_sim_vs_planning_depth": _basic_correlation(
            valid_similarities, valid_depths
        ),
        "real_step_image_sim_vs_action_diversity": _basic_correlation(
            valid_similarities, valid_action_div
        ),
        "real_step_image_sim_vs_imagination_diversity": _basic_correlation(
            valid_similarities, valid_diversities
        ),
    }

    results = {key: window_results.get(key) for key in CORRELATION_KEYS}
    results.update(basic_results)
    return results


def _combine_fisher_stouffer(values):
    if not values:
        return None

    z_crit = stats.norm.ppf(0.975)

    def _calc_ci_from_z(z_value, weight):
        if weight <= 0:
            r_val = float(np.tanh(z_value))
            return r_val, r_val
        z_se = 1.0 / np.sqrt(weight)
        z_lower = z_value - z_crit * z_se
        z_upper = z_value + z_crit * z_se
        return float(np.tanh(z_lower)), float(np.tanh(z_upper))

    def _calc_r2_ci(r_lower, r_upper):
        lower = min(r_lower ** 2, r_upper ** 2)
        upper = max(r_lower ** 2, r_upper ** 2)
        return float(lower), float(upper)

    if len(values) == 1:
        v = values[0]
        r_single = float(v["r"])
        r_clipped = np.clip(r_single, -0.999999, 0.999999)
        z_single = float(np.arctanh(r_clipped))
        n_single = max(int(v.get("n", 0) or 0), 0)
        weight_single = float(max(n_single - 3.0, 1.0)) if n_single else 1.0
        r_low, r_high = _calc_ci_from_z(z_single, weight_single)
        r2_low, r2_high = _calc_r2_ci(r_low, r_high)
        single_r2 = float(v.get("r2", r_single ** 2))
        p_val = v.get("p")
        p_single = float(p_val) if p_val is not None else np.nan
        return {
            "r": r_single,
            "r2": single_r2,
            "p": p_single,
            "n": n_single if n_single > 0 else None,
            "count": 1,
            "fisher_z": z_single,
            "fisher_weight": weight_single,
            "r_ci_lower": r_low,
            "r_ci_upper": r_high,
            "r2_ci_lower": r2_low,
            "r2_ci_upper": r2_high,
        }

    rs = np.array([v["r"] for v in values], dtype=float)
    ps = np.array([v["p"] for v in values], dtype=float)
    ns = np.array([max(int(v.get("n", 0) or 0), 0) for v in values], dtype=float)

    mask = np.isfinite(rs) & np.isfinite(ps) & (ns >= 1)
    if not np.any(mask):
        return None

    rs = rs[mask]
    ps = ps[mask]
    ns = ns[mask]

    weights = np.maximum(ns - 3.0, 1.0)
    fisher_z = np.arctanh(np.clip(rs, -0.999999, 0.999999))
    weight_sum = float(np.sum(weights))

    z_bar = float(np.sum(weights * fisher_z) / weight_sum)
    r_bar = float(np.tanh(z_bar))
    r2_bar = float(r_bar ** 2)
    r_low, r_high = _calc_ci_from_z(z_bar, weight_sum)
    r2_low, r2_high = _calc_r2_ci(r_low, r_high)

    try:
        z_p = stats.norm.isf(ps / 2.0) * np.sign(rs)
        Z = np.sum(weights * z_p) / np.sqrt(np.sum(weights ** 2))
        p_comb = float(2.0 * stats.norm.sf(abs(Z)))
    except Exception:
        p_comb = float(np.nanmean(ps))

    n_total = int(np.sum(ns)) if float(np.sum(ns)) > 0 else None
    return {
        "r": r_bar,
        "r2": r2_bar,
        "p": p_comb,
        "n": n_total,
        "count": int(np.sum(mask)),
        "fisher_z": z_bar,
        "fisher_weight": weight_sum,
        "r_ci_lower": r_low,
        "r_ci_upper": r_high,
        "r2_ci_lower": r2_low,
        "r2_ci_upper": r2_high,
    }


def process_metrics_folder(folder: str, outdir: str, window_size: int, stride: int) -> None:
    os.makedirs(outdir, exist_ok=True)

    metric_files = sorted(
        glob.glob(os.path.join(folder, "*.npz"))
        + glob.glob(os.path.join(folder, "*.npy"))
    )
    if not metric_files:
        print(f"No metric files found in {folder}.")
        return

    file_groups = defaultdict(lambda: defaultdict(list))
    for file_path in metric_files:
        filename = os.path.basename(file_path)
        gamename, step, number = _parse_metrics_filename(filename)
        if gamename is None:
            print(f"Skipping unmatched filename pattern: {filename}")
            continue
        file_groups[gamename][step].append((file_path, number))

    for gamename, step_groups in file_groups.items():
        print(f"\n=== {gamename} ===")
        step_data = {}

        for step, files in sorted(step_groups.items()):
            print(f"  Step {step}: {len(files)} file(s)")
            correlations_by_key = {key: [] for key in CORRELATION_KEYS}

            for file_path, number in sorted(files, key=lambda item: item[1]):
                print(f"    Processing file {number} ({os.path.basename(file_path)})...")
                try:
                    metrics = _load_metrics_file(file_path)
                    correlations = calculate_metrics_correlations(metrics, window_size, stride)
                except Exception as exc:
                    print(f"      Failed: {exc}")
                    continue

                for key, value in correlations.items():
                    if key in correlations_by_key and value is not None:
                        correlations_by_key[key].append(value)

            combined = {}
            for key in CORRELATION_KEYS:
                combined[key] = _combine_fisher_stouffer(correlations_by_key[key])

            step_data[step] = combined

        fig_pong.create_step_analysis_plots(gamename, step_data, outdir)

    print(f"\nAll plots saved to {outdir}.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot correlations from exported Thinker metrics."
    )
    parser.add_argument("--folder", required=True, help="Folder containing metric files")
    parser.add_argument("--outdir", required=True, help="Output directory for plots")
    parser.add_argument(
        "--windowsize",
        type=int,
        default=150,
        help="Sliding window size for windowed correlations (default: 150)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride between sliding windows (default: 1)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.windowsize <= 0:
        raise ValueError("--windowsize must be a positive integer")
    if args.stride <= 0:
        raise ValueError("--stride must be a positive integer")

    process_metrics_folder(args.folder, args.outdir, args.windowsize, args.stride)


if __name__ == "__main__":
    main()
