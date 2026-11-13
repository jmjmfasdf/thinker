"""Check seeded Atari prefix reconstruction; this is NOT a reward VOC test."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import envpool
import numpy as np


def fingerprint(result):
    h = hashlib.sha256()
    def add(value):
        if isinstance(value, dict):
            for key in sorted(value):
                h.update(key.encode()); add(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value:
                add(item)
        else:
            a = np.asarray(value)
            h.update(str((a.dtype, a.shape)).encode()); h.update(a.tobytes())
    add(result)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for game in ["Enduro-v5", "Pong-v5"]:
        for seed in [20260924, 20260925]:
            envs = [envpool.make_gymnasium(game, num_envs=1, batch_size=1, num_threads=1,
                        seed=seed, gray_scale=False, stack_num=4, episodic_life=True) for _ in range(2)]
            resets = [e.reset() for e in envs]
            reset_match = fingerprint(resets[0]) == fingerprint(resets[1])
            rng = np.random.default_rng(seed)
            actions = rng.integers(0, envs[0].action_space.n, size=128)
            exact = []
            rewards = [[], []]
            hashes = [[], []]
            for action in actions:
                results = [e.step(np.asarray([action], dtype=np.int32)) for e in envs]
                exact.append(fingerprint(results[0]) == fingerprint(results[1]))
                for i, result in enumerate(results):
                    hashes[i].append(fingerprint(result))
                    rewards[i].append(float(result[1][0]))
            row = dict(game=game, seed=seed, observation_shape=list(resets[0][0].shape),
                       num_actions=int(envs[0].action_space.n), reset_exact=reset_match,
                       exact_steps=sum(exact), tested_steps=len(exact),
                       quick_save_available=hasattr(envs[0], "quick_save"),
                       quick_load_available=hasattr(envs[0], "quick_load"),
                       actions=actions.tolist(), rewards=rewards[0], path_hashes=hashes[0])
            rows.append(row)
            print(game, seed, "paired reset", reset_match, "exact steps", sum(exact), flush=True)
            if not reset_match or not all(exact):
                raise AssertionError("Seeded online environment reconstruction failed")
            del envs
    manifest = dict(status="ENVIRONMENT_PREFIX_PASS", envpool_version=envpool.__version__, rows=rows,
                    actor_planner_pairing_tested=False, reward_voc_validated=False,
                    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    interpretation="Rebuild each online arm from the same seed and accepted action prefix. "
                      "Human NPZ observations cannot reconstruct their original emulator state. "
                      "EnvPoolWrap.reset ignores seed; the seed must be passed to envpool construction. "
                      "Installed EnvPool exposes no quick_save/quick_load; wrapper methods alone do not prove support.")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())


if __name__ == "__main__":
    main()
