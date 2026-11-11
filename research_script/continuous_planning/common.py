"""Shared contracts for the additive continuous-planning reanalysis.

No behavioral archive/pickle loading is allowed in this package. Numeric NPZ
files below are derived caches, never input behavioral archives.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

VERSION = "continuous-planning-v1.0"
HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("CONTINUOUS_REPO", "/home/jeongmin/thinker-dynamic-imitation"))
DEFAULT_OUTPUT = REPO / "research_script/outputs/continuous_planning_v1"
CAMPAIGN = REPO / "logs/visual_behav/heldout_5m_20260921"
LAGS = ((0., .125), (.125, .25), (.25, .5), (.5, 1.), (1., 2.), (2., 4.))
HORIZONS = ((1, 1), (2, 2), (3, 4), (5, 8), (9, 15), (16, 30))
MODULES = {
    "01": "01_information_timescales",
    "02": "02_maintenance_switch_destination",
    "03": "03_evidence_revision_accumulation",
    "04": "04_dynamic_computation",
    "05": "05_plan_continuity_update",
    "06": "06_urgency_modulation",
    "07": "07_neural_correspondence",
}
DEFAULT_CONFIG = dict(
    version=VERSION, seed=20260921, c_grid=[.001, .01, .1, 1., 10., 100.],
    max_iter=1000, tolerance=1e-6, pca_components=32, pca_fit_rows=12000,
    embargo_seconds=8., subjects=[1, 2, 3, 4, 5, 6], games=[0, 1],
    annotations_per_game=200, tracker_accuracy=.90, tracker_error_pixels=2.,
    verify_native_hashes=True, time_shift_seconds=8.,
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True).encode()).hexdigest()


def sources():
    return {p.name: sha(p) for p in sorted(HERE.glob("*.py"))}


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--subject", type=int, required=True, choices=range(1, 7))
    p.add_argument("--game", type=int, required=True, choices=(0, 1))
    p.add_argument("--capture-index", type=Path)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--cache-root", type=Path, help="Optional read-only existing 00_native_cache root")
    p.add_argument("--config", type=Path)
    p.add_argument("--max-episodes", type=int, help="Smoke only; cannot enter group inference")
    return p


def settings(args):
    config = dict(DEFAULT_CONFIG)
    if args.config:
        supplied = json.loads(args.config.read_text())
        unknown = set(supplied) - set(config)
        if unknown:
            raise ValueError(f"Unknown configuration keys: {unknown}")
        config.update(supplied)
    if config["version"] != VERSION:
        raise ValueError("Configuration version mismatch")
    return config


def tag(args):
    return f"sub{args.subject:03d}_game{args.game}"


def cache_dir(args):
    return (getattr(args, "cache_root", None) or args.output_root / "00_native_cache") / tag(args)


def load_cache(args, visuals=False):
    root = cache_dir(args)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["status"] != "COMPLETED" or manifest["version"] != VERSION:
        raise ValueError("A complete matching native cache is required")
    if (manifest["subject"], manifest["game"]) != (args.subject, args.game):
        raise ValueError("Cache identity mismatch")
    decisions = pd.read_csv(root / "decisions.csv.gz")
    pixels = None
    if visuals:
        with np.load(root / "visual.npz", allow_pickle=False) as data:
            pixels = data["lowres"].astype(np.float32)
        if len(pixels) != len(decisions):
            raise ValueError("Visual row alignment mismatch")
    return decisions, pixels, manifest


def start_manifest(args, module, inputs):
    root = args.output_root / module / tag(args)
    root.mkdir(parents=True, exist_ok=True)
    signature = digest(dict(config=settings(args), sources=sources(), inputs=inputs))
    path = root / "manifest.json"
    if path.exists():
        prior = json.loads(path.read_text())
        if prior["signature"] != signature:
            raise FileExistsError(f"Preserving incompatible outputs; choose a new output root: {root}")
        if prior["status"] == "COMPLETED":
            for name, expected in prior.get("artifacts", {}).items():
                if sha(root / name) != expected:
                    raise ValueError(f"Completed artifact changed: {name}")
            return root, prior, True
    manifest = dict(version=VERSION, status="RUNNING", subject=args.subject,
                    game=args.game, signature=signature, inputs=inputs,
                    config=settings(args), sources=sources(),
                    started=datetime.now(timezone.utc).isoformat(),
                    smoke=bool(args.max_episodes), analysis_kind="observational_reanalysis")
    write_json(path, manifest)
    return root, manifest, False


def finish(root, manifest, **updates):
    manifest.update(updates)
    manifest.update(status="COMPLETED", finished=datetime.now(timezone.utc).isoformat())
    manifest["artifacts"] = {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*"))
                             if p.is_file() and p.name != "manifest.json" and not p.name.endswith(".tmp")}
    write_json(root / "manifest.json", manifest)


def assert_available(origin, available, decision):
    origin, available, decision = np.broadcast_arrays(origin, available, decision)
    if np.any(~np.isfinite(available)) or np.any(origin > available + 1e-8) or np.any(available > decision + 1e-8):
        raise ValueError("Predictor availability violation / future information")


def probabilities(logits):
    return softmax(np.asarray(logits, dtype=np.float64), axis=-1)


def entropy(p):
    p = np.clip(p, 1e-12, 1.)
    return -np.sum(p * np.log(p), axis=-1)


def exact_max_t(effects):
    """Synchronized, two-sided subject sign-flips under sign-exchangeability.

    Columns with missing subjects are not tested; they are NOT silently
    averaged over a smaller sample. The all-zero contrast has p=1.
    """
    effects = np.asarray(effects, dtype=float)
    n, m = effects.shape
    complete = np.isfinite(effects).all(axis=0)
    raw = np.full(m, np.nan); adjusted = raw.copy(); observed = raw.copy()
    if n != 6:
        raise ValueError("Population inference requires all six subjects")
    if not complete.any():
        return observed, raw, adjusted
    x = effects[:, complete]
    def statistic(a):
        denominator = a.std(axis=0, ddof=1) / np.sqrt(n)
        return a.mean(axis=0) / np.maximum(denominator, 1e-15)
    obs = np.abs(statistic(x))
    null = np.array([np.abs(statistic(x * np.array(s)[:, None]))
                     for s in itertools.product((-1., 1.), repeat=n)])
    observed[complete] = statistic(x)
    raw[complete] = (null >= obs[None, :] - 1e-12).mean(axis=0)
    adjusted[complete] = (null.max(axis=1)[:, None] >= obs[None, :] - 1e-12).mean(axis=0)
    return observed, raw, adjusted


def save_figure(fig, root, name):
    root = Path(root) / "figures"
    root.mkdir(exist_ok=True, parents=True)
    fig.savefig(root / (name + ".png"), dpi=170, bbox_inches="tight")
    fig.savefig(root / (name + ".pdf"), bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
