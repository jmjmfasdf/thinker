import argparse
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np

BASE_DIR = Path("/home/jmme425/thinker")


def resolve_target_dir(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


def _normalize_loaded(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.dtype == object and value.shape == ():
        return value.item()
    return value


def _read_kb_value(path: Path, key: str) -> Optional[int]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith(key):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1])
    except OSError:
        return None
    return None


def get_memory_usage_kb() -> Optional[Tuple[int, int]]:
    rss_kb = _read_kb_value(Path("/proc/self/status"), "VmRSS:")
    total_kb = _read_kb_value(Path("/proc/meminfo"), "MemTotal:")
    if rss_kb is None or total_kb is None:
        return None
    return rss_kb, total_kb


def log_progress(index: int, total: int, npy_file: Path) -> None:
    usage = get_memory_usage_kb()
    if usage is None:
        print(f"[{index}/{total}] loaded {npy_file.name}")
        return
    rss_kb, total_kb = usage
    rss_mb = rss_kb / 1024
    percent = (rss_kb / total_kb) * 100
    print(f"[{index}/{total}] loaded {npy_file.name} | mem {rss_mb:.1f}MB ({percent:.1f}%)")


def load_npy_file(npy_file: Path, allow_pickle: bool) -> Any:
    try:
        loaded = np.load(npy_file, allow_pickle=allow_pickle)
        return _normalize_loaded(loaded)
    except ValueError as exc:
        if not allow_pickle and "Object arrays cannot be loaded" in str(exc):
            raise ValueError(
                f"{npy_file} contains object arrays. Re-run with --allow-pickle if you trust the source."
            ) from exc
        raise


def _as_array_for_concat(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 0:
        return np.array([array.item()])
    return array


def merge_arrays(existing: Any, new: Any, label: str) -> np.ndarray:
    left = _as_array_for_concat(existing)
    right = _as_array_for_concat(new)
    try:
        return np.concatenate([left, right], axis=0)
    except ValueError as exc:
        raise ValueError(f"Failed to concatenate {label}: {exc}") from exc


def merge_dicts(existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, np.ndarray]:
    if set(existing.keys()) != set(new.keys()):
        raise ValueError("Key mismatch between files.")

    merged: Dict[str, np.ndarray] = {}
    for key in existing.keys():
        merged[key] = merge_arrays(existing[key], new[key], f"key '{key}'")
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Concatenate .npy files in a directory and save as video_stat.npz."
    )
    parser.add_argument("path", help="Directory containing .npy files")
    parser.add_argument(
        "--allow-pickle",
        action="store_true",
        help="Allow loading object arrays from .npy files (unsafe for untrusted data).",
    )
    args = parser.parse_args()

    target_dir = resolve_target_dir(args.path)
    if not target_dir.is_dir():
        print(f"Not a directory: {target_dir}", file=sys.stderr)
        return 1

    npy_files = sorted(target_dir.glob("*.npy"))
    if not npy_files:
        print(f"No .npy files found in {target_dir}", file=sys.stderr)
        return 1

    try:
        merged = load_npy_file(npy_files[0], allow_pickle=args.allow_pickle)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    merged_is_dict = isinstance(merged, dict)
    log_progress(1, len(npy_files), npy_files[0])
    for idx, npy_file in enumerate(npy_files[1:], start=2):
        try:
            loaded = load_npy_file(npy_file, allow_pickle=args.allow_pickle)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        log_progress(idx, len(npy_files), npy_file)

        if merged_is_dict != isinstance(loaded, dict):
            print("Mixed .npy contents (dict and array) are not supported.", file=sys.stderr)
            return 1

        try:
            if merged_is_dict:
                merged = merge_dicts(merged, loaded)
            else:
                merged = merge_arrays(merged, loaded, "arrays")
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    output_path = target_dir / "video_stat.npz"
    if merged_is_dict:
        np.savez_compressed(output_path, **merged)
    else:
        np.savez_compressed(output_path, data=merged)

    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
