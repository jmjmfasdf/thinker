import argparse
from pathlib import Path
import sys
from typing import List

import numpy as np

BASE_DIR = Path("/home/jmme425/thinker")


def resolve_target_dir(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


def load_npy_files(target_dir: Path) -> List[np.ndarray]:
    npy_files = sorted(target_dir.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in {target_dir}")
    arrays = [np.load(npy_file, allow_pickle=False) for npy_file in npy_files]
    return arrays


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Concatenate .npy files in a directory and save as video_stat.npz."
    )
    parser.add_argument("path", help="Directory containing .npy files")
    args = parser.parse_args()

    target_dir = resolve_target_dir(args.path)
    if not target_dir.is_dir():
        print(f"Not a directory: {target_dir}", file=sys.stderr)
        return 1

    try:
        arrays = load_npy_files(target_dir)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if len(arrays) == 1:
        merged = arrays[0]
    else:
        try:
            merged = np.concatenate(arrays, axis=0)
        except ValueError as exc:
            print(f"Failed to concatenate arrays: {exc}", file=sys.stderr)
            return 1

    output_path = target_dir / "video_stat.npz"
    np.savez_compressed(output_path, data=merged)
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
