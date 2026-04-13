#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import numpy as np


def _build_output_path(data_path: Path) -> Path:
    parent_name = data_path.parent.name
    m = re.fullmatch(r"thinker-(pong|spaceinvaders)-(\d+)", parent_name)
    if not m:
        raise ValueError(
            "Expected parent folder like 'thinker-pong-0' or "
            f"'thinker-spaceinvaders-0', got: {parent_name}"
        )

    game_slug, thinker_idx_str = m.groups()
    game_id = 1 if game_slug == "pong" else 2
    thinker_idx = int(thinker_idx_str)

    out_dir = data_path.parent.parent / f"game_{game_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"thinker_{thinker_idx:03d}.npy"


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
