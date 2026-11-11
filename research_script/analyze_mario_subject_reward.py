#!/usr/bin/env python3
"""Summarise Mario reward by subject.

The currently running Mario job applies reward clipping *after* the environment
frame skip.  With the default job settings the learner therefore receives::

    r_train[k] = clip(sum(r[k * 4 : (k + 1) * 4]), -1, 1)

The job's ``episode_return`` metric is different: it is accumulated before
that clipping.  This script reports both quantities, together with the reward
as stored in the behavioural NPZ files.

Important limitation
--------------------
``behavioral_data_block_mario`` was generated with the converter's default
sign clipping, so its ``reward`` arrays contain the stored frame-level
sign-clipped reward rather than the raw Retro reward.  The current-job values
reported here are consequently a reproducible *proxy* for the current learner
signal, not an exact replay of raw Retro rewards.  The NPZ files also do not
retain the current job's life/stage wrapper events; each NPZ is treated as one
behavioural run for the episode-level summaries.

Examples
--------
Print a subject summary using the current job defaults::

    python analyze_mario_subject_reward.py

Write the summary to CSV and use a different data root::

    python analyze_mario_subject_reward.py \
        --data-root /path/to/behavioral_data_block_mario \
        --output mario_subject_reward_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence, TextIO

import numpy as np


DEFAULT_DATA_ROOT = Path("/home/jeongmin/thinker/behavioral_data_block_mario")
DEFAULT_FRAME_SKIP = 4
DEFAULT_REWARD_CLIP = 1.0
SUBJECT_RE = re.compile(r"sub[-_]?0*(\d+)", re.IGNORECASE)


@dataclass
class SubjectAccumulator:
    """Sufficient statistics for one subject."""

    files: int = 0
    stages: set[str] = field(default_factory=set)
    stored_frames: int = 0
    train_steps: int = 0
    stored_reward_sum: float = 0.0
    current_raw_reward_sum: float = 0.0
    current_clipped_reward_sum: float = 0.0
    clipped_episode_returns: list[float] = field(default_factory=list)
    raw_episode_returns: list[float] = field(default_factory=list)

    def add(
        self,
        *,
        stored_reward: np.ndarray,
        raw_step_reward: np.ndarray,
        clipped_step_reward: np.ndarray,
        stage: str | None,
    ) -> None:
        self.files += 1
        if stage:
            self.stages.add(stage)
        self.stored_frames += int(stored_reward.size)
        self.train_steps += int(clipped_step_reward.size)
        self.stored_reward_sum += float(stored_reward.sum(dtype=np.float64))
        raw_return = float(raw_step_reward.sum(dtype=np.float64))
        clipped_return = float(clipped_step_reward.sum(dtype=np.float64))
        self.current_raw_reward_sum += raw_return
        self.current_clipped_reward_sum += clipped_return
        self.raw_episode_returns.append(raw_return)
        self.clipped_episode_returns.append(clipped_return)

    def row(self, subject: str) -> dict[str, object]:
        clipped = np.asarray(self.clipped_episode_returns, dtype=np.float64)
        raw = np.asarray(self.raw_episode_returns, dtype=np.float64)
        return {
            "subject": subject,
            "files": self.files,
            "stages": len(self.stages),
            "stored_frames": self.stored_frames,
            "train_steps": self.train_steps,
            "stored_reward_mean_per_frame": _safe_ratio(
                self.stored_reward_sum, self.stored_frames
            ),
            "current_clipped_reward_mean_per_step": _safe_ratio(
                self.current_clipped_reward_sum, self.train_steps
            ),
            "current_clipped_episode_return_mean": _nanmean(clipped),
            "current_clipped_episode_return_std": _nanstd(clipped),
            "current_raw_episode_return_mean": _nanmean(raw),
            "current_raw_episode_return_std": _nanstd(raw),
            "stored_reward_sum": self.stored_reward_sum,
            "current_clipped_reward_sum": self.current_clipped_reward_sum,
            "current_raw_reward_sum": self.current_raw_reward_sum,
        }


def _safe_ratio(numerator: float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _nanmean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else float("nan")


def _nanstd(values: np.ndarray) -> float:
    return float(np.std(values)) if values.size else float("nan")


def normalise_subject(value: object) -> str:
    """Return a canonical subject label such as ``sub001``."""

    match = SUBJECT_RE.search(str(value))
    if not match:
        raise ValueError(f"Cannot infer subject from {value!r}")
    return f"sub{int(match.group(1)):03d}"


def resolve_data_path(raw_path: str, data_root: Path) -> Path:
    """Resolve a manifest path even when it was created on another mount."""

    path = Path(raw_path)
    if path.exists():
        return path

    # The checked-in manifest uses /home/jeongmin/thinker/... paths.  When a
    # caller supplies another data root, retain the path suffix after the
    # behavioural-data directory name.
    marker = "behavioral_data_block_mario"
    parts = path.parts
    if marker in parts:
        suffix = Path(*parts[parts.index(marker) + 1 :])
        candidate = data_root / suffix
        if candidate.exists():
            return candidate

    candidate = data_root / path.name
    if candidate.exists():
        return candidate
    return path


def manifest_records(
    manifest: Path, data_root: Path
) -> Iterator[tuple[str, Path, str | None]]:
    """Yield ``(subject, npz_path, stage)`` records from a converter manifest."""

    with manifest.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"output_path", "sub"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"Manifest {manifest} is missing required columns: {sorted(missing)}"
            )
        for row in reader:
            path = resolve_data_path(row["output_path"], data_root)
            stage = None
            if row.get("world") and row.get("level"):
                stage = f"W{row['world']}-{row['level']}"
            yield normalise_subject(row["sub"]), path, stage


def scan_records(data_root: Path) -> Iterator[tuple[str, Path, str | None]]:
    """Yield records by scanning the standard ``sub-*/ses-*`` layout."""

    for path in sorted(data_root.glob("sub-*/ses-*/*.npz")):
        try:
            subject = normalise_subject(path.parts[-3])
        except ValueError:
            continue
        yield subject, path, None


def iter_records(
    *, data_root: Path, manifest: Path | None
) -> Iterator[tuple[str, Path, str | None]]:
    source = manifest_records(manifest, data_root) if manifest else scan_records(data_root)
    seen: set[Path] = set()
    for subject, path, stage in source:
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        yield subject, path, stage


def transform_reward(
    reward: np.ndarray, *, frame_skip: int, reward_clip: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(raw_step_reward, clipped_step_reward)`` for one NPZ run."""

    reward = np.asarray(reward, dtype=np.float64).reshape(-1)
    if reward.size == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty

    step_count = int(math.ceil(reward.size / frame_skip))
    padded_size = step_count * frame_skip
    if padded_size != reward.size:
        reward = np.pad(reward, (0, padded_size - reward.size), mode="constant")
    raw_step_reward = reward.reshape(step_count, frame_skip).sum(axis=1)
    if reward_clip > 0:
        clipped_step_reward = np.clip(
            raw_step_reward, -reward_clip, reward_clip
        )
    else:
        clipped_step_reward = raw_step_reward.copy()
    return raw_step_reward, clipped_step_reward


def build_summary(
    records: Iterable[tuple[str, Path, str | None]],
    *,
    frame_skip: int,
    reward_clip: float,
    subjects: set[str] | None = None,
    limit_files: int | None = None,
) -> tuple[list[dict[str, object]], int, int]:
    if frame_skip < 1:
        raise ValueError("frame_skip must be at least 1")
    if reward_clip < 0:
        raise ValueError("reward_clip must be non-negative")

    accumulators: dict[str, SubjectAccumulator] = defaultdict(SubjectAccumulator)
    processed = 0
    skipped = 0
    for subject, path, stage in records:
        if subjects and subject not in subjects:
            continue
        if limit_files is not None and processed >= limit_files:
            break
        try:
            with np.load(path, allow_pickle=False) as archive:
                if "reward" not in archive.files:
                    raise KeyError("missing reward array")
                stored_reward = np.asarray(archive["reward"], dtype=np.float64).reshape(-1)
        except (OSError, ValueError, KeyError) as exc:
            skipped += 1
            print(f"[warning] skipping {path}: {exc}", file=sys.stderr)
            continue

        raw_step_reward, clipped_step_reward = transform_reward(
            stored_reward, frame_skip=frame_skip, reward_clip=reward_clip
        )
        accumulators[subject].add(
            stored_reward=stored_reward,
            raw_step_reward=raw_step_reward,
            clipped_step_reward=clipped_step_reward,
            stage=stage,
        )
        processed += 1

    rows = [accumulators[subject].row(subject) for subject in sorted(accumulators)]
    return rows, processed, skipped


FIELDNAMES = [
    "subject",
    "files",
    "stages",
    "stored_frames",
    "train_steps",
    "stored_reward_mean_per_frame",
    "current_clipped_reward_mean_per_step",
    "current_clipped_episode_return_mean",
    "current_clipped_episode_return_std",
    "current_raw_episode_return_mean",
    "current_raw_episode_return_std",
    "stored_reward_sum",
    "current_clipped_reward_sum",
    "current_raw_reward_sum",
]


def write_csv(rows: Sequence[Mapping[str, object]], handle: TextIO) -> None:
    writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Mario NPZ root (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Converter manifest.tsv; defaults to <data-root>/manifest.tsv when present.",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=DEFAULT_FRAME_SKIP,
        help="Environment frame skip used before clipping (default: 4).",
    )
    parser.add_argument(
        "--reward-clip",
        type=float,
        default=DEFAULT_REWARD_CLIP,
        help="Clip magnitude; <=0 disables clipping (default: 1).",
    )
    parser.add_argument(
        "--subject",
        dest="subjects",
        action="append",
        help="Restrict to a subject; repeat for multiple subjects (e.g. sub001).",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="Process at most this many files (useful for a quick smoke test).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write CSV here; otherwise CSV is printed to stdout.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = args.data_root.expanduser().resolve()
    manifest = args.manifest
    if manifest is None:
        candidate = data_root / "manifest.tsv"
        manifest = candidate if candidate.exists() else None
    elif not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    subjects = None
    if args.subjects:
        subjects = {normalise_subject(subject) for subject in args.subjects}

    rows, processed, skipped = build_summary(
        iter_records(data_root=data_root, manifest=manifest),
        frame_skip=args.frame_skip,
        reward_clip=args.reward_clip,
        subjects=subjects,
        limit_files=args.limit_files,
    )
    if not rows:
        raise SystemExit(f"No usable Mario NPZ files found under {data_root}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as handle:
            write_csv(rows, handle)
        print(f"Wrote {len(rows)} subject rows to {args.output}", file=sys.stderr)
    else:
        write_csv(rows, sys.stdout)

    print(
        f"Processed {processed} files for {len(rows)} subjects; skipped {skipped}.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
