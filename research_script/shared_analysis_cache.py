"""Shared numeric cache contract for the non-neural analyses."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pandas as pd

DEFAULT_SHARED_CACHE_ROOT = Path(__file__).resolve().parent / "outputs" / "00" / "native_cache"
SHARED_CACHE_VERSION = 3
KEYS = ["run_label", "real_step", "real_step_id"]


def cache_paths(root, subject, game):
    root = Path(root) / f"sub{subject:03d}_game{game}"
    files = {
        "manifest": "manifest.json", "actor_metrics": "actor_metrics.csv.gz",
        "root_policy_metrics": "root_policy_metrics.csv.gz", "actor_qa": "actor_qa.json",
        "persistence_windows": "persistence_windows.csv.gz",
        "persistence_bouts": "persistence_bouts.csv.gz",
        "persistence_events": "persistence_events.csv.gz", "persistence_qa": "persistence_qa.json",
        "model_control_windows": "model_control_windows.csv.gz", "model_control_qa": "model_control_qa.json",
        "rollout_windows": "rollout_windows.csv.gz", "rollout_prefix_events": "rollout_prefix_events.npz",
        "rollout_qa": "rollout_qa.json", "root_state_dir": "root_state",
        "root_state_index": "root_state/index.csv.gz", "representation_dir": "representations",
        "visual_controls": "visual_controls.npz", "visual_alignment": "visual_alignment.csv.gz",
        "native_decisions": "native_decisions.csv.gz", "native_events": "native_events.csv.gz",
        "imagination_paths": "imagination_paths.npz", "forecast_outcomes": "forecast_outcomes.csv.gz",
        "representation_summaries": "representation_summaries.csv.gz",
        "native_features_qa": "native_features_qa.json", "native_features_schema": "native_features_schema.json",
    }
    return SimpleNamespace(root=root, **{k: root / v for k, v in files.items()})


def read_manifest(paths):
    path = paths.manifest if hasattr(paths, "manifest") else Path(paths)
    return json.loads(path.read_text())


def validate_manifest(paths, *, subject=None, game=None, **kwargs):
    result = read_manifest(paths)
    for key, expected in (("subject", subject), ("game", game)):
        if expected is not None and result.get(key) != expected:
            raise ValueError(f"Cache {key} mismatch: {paths.root}")
    if result.get("status", "COMPLETED") != "COMPLETED":
        raise ValueError(f"Incomplete cache: {paths.root}")
    return result


def validate_one_to_one_real_steps(frames, keys=KEYS):
    audit = {}
    for name, frame in frames.items():
        if frame[list(keys)].isna().any().any() or frame.duplicated(list(keys)).any():
            raise ValueError(f"Invalid real-step keys in {name}")
        audit[name] = {"rows": len(frame), "runs": frame.run_label.nunique()}
    return audit


def load_actor_metrics(paths, **kwargs):
    return pd.read_csv(paths.actor_metrics, **kwargs)


def load_model_control(paths, **kwargs):
    return pd.read_csv(paths.model_control_windows, **kwargs)


def load_persistence_geometry(paths, **kwargs):
    return tuple(pd.read_csv(p, **kwargs) for p in
                 (paths.persistence_windows, paths.persistence_bouts, paths.persistence_events))


def load_rollout_value(paths, **kwargs):
    with np.load(paths.rollout_prefix_events, allow_pickle=False) as archive:
        prefix = {k: archive[k] for k in archive.files}
    return pd.read_csv(paths.rollout_windows, **kwargs), prefix


def load_representations(paths):
    directory = paths.representation_dir
    return {p.stem: np.load(p, allow_pickle=False) for p in directory.glob("*.npy")}
