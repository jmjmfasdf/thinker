#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import numpy as np


def _build_output_path(data_path: Path) -> Path:
    # Expected input: data_spaceinvaders/sub002-ses03-block7-game2/XXXX.npy
    parent_name = data_path.parent.name
    m = re.search(r"(sub\d+)-?(ses\d+)", parent_name)
    if not m:
        raise ValueError(
            f"Could not parse sub/ses from parent folder name: {parent_name}"
        )
    sub_id, ses_id = m.group(1), m.group(2)
    ses_folder = f"ses-{ses_id.replace('ses', '')}"

    file_m = re.match(r".*_(\d+)$", data_path.stem)
    if not file_m:
        raise ValueError(
            f"Could not parse numeric suffix from filename: {data_path.name}"
        )
    file_suffix = file_m.group(1)

    out_dir = Path("test") / sub_id / ses_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{parent_name}_{file_suffix}.npy"


def save_filtered_npy(data_path: Path, keys: list[str]) -> Path:
    data = np.load(data_path, allow_pickle=True).item()
    print("data.keys():", data.keys())

    missing = [k for k in keys if k not in data]
    if missing:
        print("Warning: missing keys:", missing, file=sys.stderr)

    filtered = {k: data[k] for k in keys if k in data}
    out_path = _build_output_path(data_path)
    np.save(out_path, filtered, allow_pickle=True)
    print("saved:", out_path)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Filter keys from a .npy dict and save as <name>_new.npy"
    )
    parser.add_argument("data_path", help="Path to .npy file containing a dict")
    parser.add_argument(
        "keys",
        nargs="+",
        help="Keys to keep (space-separated). Example: key1 key2 key3",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Error: file not found: {data_path}", file=sys.stderr)
        return 1

    save_filtered_npy(data_path, args.keys)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
