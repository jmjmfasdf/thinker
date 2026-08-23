#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


DEFAULT_MARIO_ROOT = Path(
    "/home/jeongmin/thinker/cneuromod/mario.fmriprep/sourcedata/cneuromod.mario"
)
DEFAULT_OUTPUT_ROOT = Path("/home/jeongmin/thinker/behavioral_data_block_mario")

BK2_RE = re.compile(
    r"(?P<sub>sub-\d+)/(?P<ses>ses-\d+)/gamelogs/"
    r"(?P=sub)_(?P=ses)_task-mario_level-w(?P<world>\d+)l(?P<level>\d+)_rep-(?P<rep>\d+)\.bk2$"
)
EVENT_RE = re.compile(r"_run-(?P<run>\d+)_events\.tsv$")

BUTTON_ORDER = ["A", "Right", "Left", "Down", "B", "Select", "Start", "Up"]
DROP_FOR_FALLBACK = {"Up", "Down", "Select", "Start"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay cneuromod Mario .bk2 files and save behavioral_data_block-style .npz files. "
            "True image/reward replay requires gym-retro/stable-retro importable as `retro`."
        )
    )
    parser.add_argument("--mario-root", type=Path, default=DEFAULT_MARIO_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--action-vocab", type=Path, default=None)
    parser.add_argument("--make-manifest", action="store_true")
    parser.add_argument("--bk2", type=Path, default=None, help="Convert one .bk2 directly.")
    parser.add_argument("--array-index", type=int, default=None)
    parser.add_argument("--files-per-task", type=int, default=1)
    parser.add_argument("--num-actions", type=int, default=9)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--height", type=int, default=84)
    parser.add_argument("--width", type=int, default=84)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--datalad-get", action="store_true")
    parser.add_argument(
        "--actions-only",
        action="store_true",
        help="Debug mode: parse actions only and write zero images/rewards without emulator replay.",
    )
    parser.add_argument(
        "--unknown-action",
        choices=["nearest", "noop", "error"],
        default="nearest",
        help="How to handle button combinations outside the 9-action vocabulary.",
    )
    parser.add_argument(
        "--no-clip-reward",
        action="store_true",
        help="Keep raw environment rewards instead of clipping to -1/0/1.",
    )
    return parser.parse_args()


def default_manifest_path(output_root: Path) -> Path:
    return output_root / "manifest.tsv"


def default_vocab_path(output_root: Path) -> Path:
    return output_root / "action_vocab.json"


def abs_path(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def normalize_sub(sub: str) -> str:
    return f"sub-{int(sub.split('-')[1]):03d}"


def normalize_ses(ses: str) -> str:
    return f"ses-{int(ses.split('-')[1]):02d}"


def compact_sub(sub: str) -> str:
    return normalize_sub(sub).replace("-", "")


def compact_ses(ses: str) -> str:
    return normalize_ses(ses).replace("-", "")


def relative_posix(path: Path, root: Path) -> str:
    return abs_path(path).relative_to(abs_path(root)).as_posix()


def parse_bk2_name(path: Path, mario_root: Path) -> Dict[str, str]:
    rel = relative_posix(path, mario_root)
    match = BK2_RE.match(rel)
    if not match:
        raise ValueError(f"Unexpected bk2 path layout: {path}")
    return match.groupdict()


def event_index(mario_root: Path) -> Dict[str, Dict[str, str]]:
    index: Dict[str, Dict[str, str]] = {}
    for event_path in sorted(mario_root.glob("sub-*/ses-*/func/*_events.tsv")):
        run_match = EVENT_RE.search(event_path.name)
        if not run_match:
            continue
        run = int(run_match.group("run"))
        game_idx = 0
        with event_path.open("r", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                stim_file = (row.get("stim_file") or "").strip()
                if not stim_file.endswith(".bk2"):
                    continue
                bk2_path = abs_path(mario_root / stim_file)
                index[str(bk2_path)] = {
                    "run": str(run),
                    "game": str(game_idx),
                    "event_onset": (row.get("onset") or ""),
                    "event_sample": (row.get("sample") or ""),
                }
                game_idx += 1
    return index


def output_path_for(
    bk2_path: Path,
    mario_root: Path,
    output_root: Path,
    event_meta: Optional[Dict[str, str]],
) -> Path:
    name_meta = parse_bk2_name(bk2_path, mario_root)
    sub_dir = normalize_sub(name_meta["sub"])
    ses_dir = normalize_ses(name_meta["ses"])
    sub_token = compact_sub(name_meta["sub"])
    ses_token = compact_ses(name_meta["ses"])
    if event_meta is not None:
        block = int(event_meta["run"])
        game = int(event_meta["game"])
        filename = f"{sub_token}-{ses_token}-block{block}-game{game}.npz"
    else:
        world = name_meta["world"]
        level = name_meta["level"]
        rep = int(name_meta["rep"])
        filename = f"{sub_token}-{ses_token}-block0-gamew{world}l{level}rep{rep:03d}.npz"
    return output_root / sub_dir / ses_dir / filename


def iter_bk2(mario_root: Path) -> List[Path]:
    return sorted(mario_root.glob("sub-*/ses-*/gamelogs/*.bk2"))


def ensure_available(path: Path, mario_root: Path, datalad_get: bool) -> None:
    if zipfile.is_zipfile(path):
        return
    if not datalad_get:
        raise FileNotFoundError(
            f"{path} is not available as a readable .bk2 zip. "
            "Run datalad get first or pass --datalad-get."
        )
    subprocess.run(["datalad", "get", str(path)], cwd=mario_root, check=True)
    if not zipfile.is_zipfile(path):
        raise FileNotFoundError(f"After datalad get, {path} is still not a readable .bk2 zip.")


def read_input_log(path: Path) -> Tuple[List[str], List[str]]:
    with zipfile.ZipFile(path) as zf:
        raw = zf.read("Input Log.txt").decode("utf-8", errors="replace")

    in_input = False
    button_order: List[str] = []
    patterns: List[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if line == "[Input]":
            in_input = True
            continue
        if line == "[/Input]":
            break
        if not in_input or not line:
            continue
        if line.startswith("P1 "):
            labels = [part.strip() for part in line.split("|") if part.strip()]
            button_order = [label.split(" ", 1)[1] if " " in label else label for label in labels]
            continue
        if line.startswith("|"):
            fields = line.split("|")
            if len(fields) >= 3:
                patterns.append(fields[2])
    if not button_order:
        button_order = BUTTON_ORDER[:]
    return button_order, patterns


def pattern_to_buttons(pattern: str, button_order: Sequence[str]) -> Tuple[str, ...]:
    buttons = []
    for char, name in zip(pattern, button_order):
        if char != ".":
            buttons.append(name)
    return tuple(buttons)


def buttons_to_pattern(buttons: Iterable[str], button_order: Sequence[str]) -> str:
    selected = set(buttons)
    chars = []
    for name in button_order:
        chars.append(name[0].upper() if name in selected else ".")
    return "".join(chars)


def make_action_vocab(
    bk2_paths: Sequence[Path],
    mario_root: Path,
    output_path: Path,
    num_actions: int,
    datalad_get: bool,
) -> None:
    counts: Counter[Tuple[str, ...]] = Counter()
    button_order = BUTTON_ORDER[:]
    for bk2_path in bk2_paths:
        try:
            ensure_available(bk2_path, mario_root, datalad_get=datalad_get)
            button_order, patterns = read_input_log(bk2_path)
        except Exception as exc:
            print(f"[vocab warning] skipped {bk2_path}: {exc}", file=sys.stderr)
            continue
        counts.update(pattern_to_buttons(pattern, button_order) for pattern in patterns)

    noop = tuple()
    actions: List[Tuple[str, ...]] = [noop]
    for buttons, _count in counts.most_common():
        if buttons == noop:
            continue
        actions.append(buttons)
        if len(actions) >= num_actions:
            break

    while len(actions) < num_actions:
        actions.append((f"UNUSED_{len(actions)}",))

    payload = {
        "num_actions": num_actions,
        "button_order": button_order,
        "actions": [
            {
                "index": idx,
                "name": "NOOP" if len(buttons) == 0 else "+".join(buttons),
                "buttons": list(buttons),
                "count": int(counts.get(buttons, 0)),
            }
            for idx, buttons in enumerate(actions)
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")


def load_action_vocab(path: Optional[Path], num_actions: int) -> List[Tuple[str, ...]]:
    if path is not None and path.exists():
        payload = json.loads(path.read_text())
        actions = []
        for item in sorted(payload["actions"], key=lambda x: int(x["index"])):
            buttons = tuple(str(x) for x in item.get("buttons", []))
            actions.append(buttons)
        if len(actions) != int(payload.get("num_actions", len(actions))):
            raise ValueError(f"Malformed action vocabulary: {path}")
        return actions

    fallback = [
        tuple(),
        ("B",),
        ("Right",),
        ("Right", "B"),
        ("A", "B"),
        ("A", "Right", "B"),
        ("Left",),
        ("Left", "B"),
        ("A", "Left", "B"),
    ]
    if num_actions <= len(fallback):
        return fallback[:num_actions]
    return fallback + [(f"UNUSED_{idx}",) for idx in range(len(fallback), num_actions)]


def action_index(
    buttons: Tuple[str, ...],
    vocab: Sequence[Tuple[str, ...]],
    unknown_action: str,
) -> int:
    vocab_sets = [set(item) for item in vocab]
    selected = set(buttons)
    for idx, item in enumerate(vocab_sets):
        if selected == item:
            return idx

    collapsed = selected.difference(DROP_FOR_FALLBACK)
    for idx, item in enumerate(vocab_sets):
        if collapsed == item:
            return idx

    if unknown_action == "noop":
        return 0
    if unknown_action == "error":
        raise ValueError(f"Button combination not in vocabulary: {sorted(selected)}")

    best_idx = 0
    best_score = math.inf
    for idx, item in enumerate(vocab_sets):
        score = len(collapsed.symmetric_difference(item))
        if score < best_score:
            best_score = score
            best_idx = idx
    return best_idx


def actions_to_one_hot(
    patterns: Sequence[str],
    button_order: Sequence[str],
    vocab: Sequence[Tuple[str, ...]],
    unknown_action: str,
) -> np.ndarray:
    out = np.zeros((len(patterns), len(vocab)), dtype=np.float64)
    for idx, pattern in enumerate(patterns):
        buttons = pattern_to_buttons(pattern, button_order)
        out[idx, action_index(buttons, vocab, unknown_action)] = 1.0
    return out


def resize_frame(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=2)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    if frame.shape[:2] != (height, width):
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return frame.astype(np.uint8, copy=False)


def import_retro():
    try:
        import stable_retro as retro  # type: ignore
    except ImportError as exc:
        try:
            import retro  # type: ignore
        except ImportError:
            raise RuntimeError(
                "The `stable_retro`/`retro` Python package is not installed in this environment. "
                "Install stable-retro and import the SuperMarioBros-Nes ROM, "
                "or run with --actions-only for a parser-only smoke test."
            ) from exc
    return retro


def normalize_reset(reset_out):
    if isinstance(reset_out, tuple):
        return reset_out[0]
    return reset_out


def normalize_step(step_out):
    if len(step_out) == 5:
        obs, reward, terminated, truncated, info = step_out
        return obs, reward, bool(terminated or truncated), info
    obs, reward, done, info = step_out
    return obs, reward, bool(done), info


def replay_with_retro(
    bk2_path: Path,
    patterns: Sequence[str],
    height: int,
    width: int,
    fps: float,
    clip_reward: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    retro = import_retro()
    movie = retro.Movie(str(bk2_path))
    if not movie.step():
        raise RuntimeError(f"Movie has no initial state: {bk2_path}")

    movie_game = movie.get_game()
    players = int(getattr(movie, "players", 1) or 1)
    game_candidates = [movie_game]
    if not movie_game.endswith("-v0"):
        game_candidates.append(f"{movie_game}-v0")

    env = None
    last_file_error: Optional[FileNotFoundError] = None
    for game in game_candidates:
        try:
            env = retro.make(
                game=game,
                state=None,
                players=players,
                use_restricted_actions=retro.Actions.ALL,
                render_mode="rgb_array",
            )
            break
        except FileNotFoundError as exc:
            last_file_error = exc
    if env is None:
        raise FileNotFoundError(
            f"Retro can read the movie but cannot find the ROM for {movie_game} "
            f"(also tried: {', '.join(game_candidates[1:])}). "
            "Import the ROM first, for example: "
            "`python -m retro.import /path/to/directory/containing/the/rom`."
        ) from last_file_error
    env.initial_state = movie.get_state()
    normalize_reset(env.reset())

    images: List[np.ndarray] = []
    rewards: List[int] = []
    terminals: List[bool] = []
    times: List[float] = []

    step_idx = 0
    while movie.step():
        keys = []
        for player_idx in range(players):
            for button_idx in range(env.num_buttons):
                keys.append(bool(movie.get_key(button_idx, player_idx)))
        obs, reward, done, _info = normalize_step(env.step(keys))
        images.append(resize_frame(obs, height, width))
        reward_value = float(reward)
        if clip_reward:
            reward_value = float(np.sign(reward_value))
        rewards.append(int(reward_value))
        terminals.append(bool(done))
        times.append(float(step_idx) / fps)
        step_idx += 1
        if done:
            break

    env.close()
    if not images:
        raise RuntimeError(f"No frames produced while replaying {bk2_path}")

    return (
        np.stack(images, axis=0).astype(np.uint8, copy=False),
        np.asarray(rewards, dtype=np.int32),
        np.asarray(terminals, dtype=bool),
        np.asarray(times, dtype=np.float32),
        np.asarray(patterns[1 : 1 + len(images)]),
    )


def actions_only_arrays(
    patterns: Sequence[str],
    height: int,
    width: int,
    fps: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    length = len(patterns)
    images = np.zeros((length, height, width, 3), dtype=np.uint8)
    rewards = np.zeros(length, dtype=np.int32)
    terminals = np.zeros(length, dtype=bool)
    if length:
        terminals[-1] = True
    times = (np.arange(length, dtype=np.float32) / np.float32(fps)).astype(np.float32)
    return images, rewards, terminals, times, np.asarray(patterns)


def convert_one(
    *,
    bk2_path: Path,
    output_path: Path,
    mario_root: Path,
    action_vocab_path: Optional[Path],
    num_actions: int,
    height: int,
    width: int,
    fps: float,
    overwrite: bool,
    datalad_get: bool,
    actions_only: bool,
    unknown_action: str,
    clip_reward: bool,
) -> None:
    if output_path.exists() and not overwrite:
        print(f"[skip] {output_path}")
        return

    ensure_available(bk2_path, mario_root, datalad_get=datalad_get)
    button_order, patterns = read_input_log(bk2_path)
    vocab = load_action_vocab(action_vocab_path, num_actions=num_actions)

    if actions_only:
        images, rewards, terminals, times, used_patterns = actions_only_arrays(
            patterns, height=height, width=width, fps=fps
        )
    else:
        images, rewards, terminals, times, used_patterns = replay_with_retro(
            bk2_path,
            patterns=patterns,
            height=height,
            width=width,
            fps=fps,
            clip_reward=clip_reward,
        )

    action = actions_to_one_hot(
        used_patterns.tolist(),
        button_order=button_order,
        vocab=vocab,
        unknown_action=unknown_action,
    )
    length = len(action)
    images = images[:length]
    rewards = rewards[:length]
    terminals = terminals[:length]
    times = times[:length]
    is_first = np.zeros(length, dtype=bool)
    if length:
        is_first[0] = True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    np.savez_compressed(
        tmp_path,
        image=images.astype(np.uint8, copy=False),
        action=action.astype(np.float64, copy=False),
        reward=rewards.astype(np.int32, copy=False),
        is_first=is_first,
        is_terminal=terminals.astype(bool, copy=False),
        time=times.astype(np.float32, copy=False),
    )
    tmp_npz = tmp_path.with_suffix(tmp_path.suffix + ".npz")
    if tmp_npz.exists():
        tmp_npz.replace(output_path)
    else:
        tmp_path.replace(output_path)
    print(f"[ok] {bk2_path} -> {output_path} ({length} frames)")


def make_manifest(
    *,
    mario_root: Path,
    output_root: Path,
    manifest_path: Path,
    action_vocab_path: Path,
    num_actions: int,
    datalad_get: bool,
) -> None:
    bk2_paths = iter_bk2(mario_root)
    events = event_index(mario_root)
    rows = []
    for idx, bk2_path in enumerate(bk2_paths):
        event_meta = events.get(str(abs_path(bk2_path)))
        name_meta = parse_bk2_name(bk2_path, mario_root)
        out_path = output_path_for(
            bk2_path=bk2_path,
            mario_root=mario_root,
            output_root=output_root,
            event_meta=event_meta,
        )
        rows.append(
            {
                "index": idx,
                "bk2_path": str(bk2_path),
                "output_path": str(out_path),
                "sub": name_meta["sub"],
                "ses": name_meta["ses"],
                "run": "" if event_meta is None else event_meta["run"],
                "game": "" if event_meta is None else event_meta["game"],
                "world": name_meta["world"],
                "level": name_meta["level"],
                "rep": name_meta["rep"],
                "event_onset": "" if event_meta is None else event_meta["event_onset"],
                "event_sample": "" if event_meta is None else event_meta["event_sample"],
            }
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            delimiter="\t",
            fieldnames=[
                "index",
                "bk2_path",
                "output_path",
                "sub",
                "ses",
                "run",
                "game",
                "world",
                "level",
                "rep",
                "event_onset",
                "event_sample",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    make_action_vocab(
        bk2_paths=bk2_paths,
        mario_root=mario_root,
        output_path=action_vocab_path,
        num_actions=num_actions,
        datalad_get=datalad_get,
    )
    print(f"[manifest] wrote {len(rows)} rows to {manifest_path}")
    print(f"[vocab] wrote {action_vocab_path}")


def load_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main() -> int:
    args = parse_args()
    mario_root = abs_path(args.mario_root)
    output_root = abs_path(args.output_root)
    manifest_path = abs_path(args.manifest or default_manifest_path(output_root))
    action_vocab_path = abs_path(args.action_vocab or default_vocab_path(output_root))

    if args.make_manifest:
        make_manifest(
            mario_root=mario_root,
            output_root=output_root,
            manifest_path=manifest_path,
            action_vocab_path=action_vocab_path,
            num_actions=args.num_actions,
            datalad_get=args.datalad_get,
        )
        return 0

    if args.bk2 is not None:
        bk2_path = abs_path(args.bk2)
        event_meta = event_index(mario_root).get(str(bk2_path))
        output_path = output_path_for(
            bk2_path=bk2_path,
            mario_root=mario_root,
            output_root=output_root,
            event_meta=event_meta,
        )
        convert_one(
            bk2_path=bk2_path,
            output_path=output_path,
            mario_root=mario_root,
            action_vocab_path=action_vocab_path,
            num_actions=args.num_actions,
            height=args.height,
            width=args.width,
            fps=args.fps,
            overwrite=args.overwrite,
            datalad_get=args.datalad_get,
            actions_only=args.actions_only,
            unknown_action=args.unknown_action,
            clip_reward=not args.no_clip_reward,
        )
        return 0

    rows = load_manifest(manifest_path)
    array_index = args.array_index
    if array_index is None:
        array_index = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    start = array_index * args.files_per_task
    end = min(start + args.files_per_task, len(rows))
    if start >= len(rows):
        print(f"[done] array index {array_index} starts beyond manifest length {len(rows)}")
        return 0

    for row in rows[start:end]:
        convert_one(
            bk2_path=abs_path(Path(row["bk2_path"])),
            output_path=abs_path(Path(row["output_path"])),
            mario_root=mario_root,
            action_vocab_path=action_vocab_path,
            num_actions=args.num_actions,
            height=args.height,
            width=args.width,
            fps=args.fps,
            overwrite=args.overwrite,
            datalad_get=args.datalad_get,
            actions_only=args.actions_only,
            unknown_action=args.unknown_action,
            clip_reward=not args.no_clip_reward,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
