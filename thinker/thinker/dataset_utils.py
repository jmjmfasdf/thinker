from __future__ import annotations

import hashlib
import os
from typing import List, Optional, Sequence

import numpy as np

from thinker.bc_loader import FrameStackedBehavioralDataLoader


def _parse_subjects(raw_subjects: Sequence[str] | str) -> List[int]:
    if isinstance(raw_subjects, (list, tuple)):
        tokens = raw_subjects
    else:
        tokens = str(raw_subjects).split(",")
    subjects: List[int] = []
    for token in tokens:
        token = str(token).strip()
        if not token:
            continue
        try:
            subjects.append(int(token))
        except ValueError:
            continue
    return subjects


def create_behavior_loader_from_flags(flags, logger=None) -> Optional[FrameStackedBehavioralDataLoader]:
    """Instantiate FrameStackedBehavioralDataLoader based on dataset_* flags."""
    base_path = getattr(flags, "dataset_base_path", "")
    if not base_path:
        if logger:
            logger.warning("dataset_base_path is empty; dataset loader disabled.")
        return None
    subjects = _parse_subjects(getattr(flags, "dataset_subjects", ""))
    if not subjects:
        if logger:
            logger.warning("dataset_subjects is empty or invalid; dataset loader disabled.")
        return None
    try:
        game_id = int(getattr(flags, "dataset_game_id", 0))
    except (TypeError, ValueError):
        game_id = 0
    image_size = int(getattr(flags, "dataset_image_size", 84))
    target_size = (image_size, image_size)
    frame_stack = int(getattr(flags, "frame_stack_n", 4))
    grayscale = bool(getattr(flags, "dataset_grayscale", getattr(flags, "grayscale", True)))
    normalize = bool(getattr(flags, "dataset_normalize", True))
    try:
        loader = FrameStackedBehavioralDataLoader(
            base_path=os.path.abspath(base_path),
            subjects=subjects,
            game_id=game_id,
            frame_stack_n=frame_stack,
            target_size=target_size,
            grayscale=grayscale,
            normalize=normalize,
        )
    except Exception as exc:  # pragma: no cover - initialization warnings only
        if logger:
            logger.warning(f"Failed to initialise dataset loader ({base_path}): {exc}")
        return None
    if len(loader.data_files) == 0:
        if logger:
            logger.warning(
                "Dataset loader found no files at %s (subjects=%s, game_id=%s).",
                base_path,
                subjects,
                game_id,
            )
        return None
    if logger:
        logger.info(
            "Dataset loader initialised with %d files (subjects=%s, game_id=%s).",
            len(loader.data_files),
            subjects,
            game_id,
        )
    return loader


def hash_observation(obs: np.ndarray) -> str:
    """Create a deterministic hash for a stacked observation tensor."""
    arr = np.ascontiguousarray(obs)
    return hashlib.sha1(arr.tobytes()).hexdigest()
