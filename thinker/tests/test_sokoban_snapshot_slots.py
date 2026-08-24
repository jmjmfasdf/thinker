import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pytest


def _load_sokoban_env_without_native_constructor(monkeypatch):
    """Load psokoban while replacing only its compiled cSokoban dependency."""
    repo_root = Path(__file__).resolve().parents[2]
    env_dir = repo_root / "sokoban" / "gym_sokoban" / "envs"
    package_name = "_thinker_test_sokoban_envs"

    package = types.ModuleType(package_name)
    package.__path__ = [str(env_dir)]
    monkeypatch.setitem(sys.modules, package_name, package)

    native_module = types.ModuleType(f"{package_name}.csokoban")
    native_module.cSokoban = object
    monkeypatch.setitem(sys.modules, native_module.__name__, native_module)

    module_name = f"{package_name}.psokoban"
    spec = importlib.util.spec_from_file_location(
        module_name, env_dir / "psokoban.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module.SokobanEnv


class FakeCSokoban:
    def __init__(self):
        self.value = np.asarray([1, 2, 3], dtype=np.uint8)
        self.step_n = 0
        self.restored_states = []

    def clone_state(self):
        return {
            "value": self.value.copy(),
            "step_n": self.step_n,
        }

    def restore_state(self, state):
        self.value = state["value"].copy()
        self.step_n = state["step_n"]
        self.restored_states.append((self.value.copy(), self.step_n))


def test_sokoban_named_snapshot_slots_are_isolated_and_deletable(monkeypatch):
    SokobanEnv = _load_sokoban_env_without_native_constructor(monkeypatch)
    env = SokobanEnv.__new__(SokobanEnv)
    env.sokoban = FakeCSokoban()
    env.save_states = {}

    # The historical no-argument API remains slot zero.
    env.quick_save()
    env.sokoban.value[:] = 7
    env.sokoban.step_n = 7
    env.quick_save(11)
    env.sokoban.value[:] = 29
    env.sokoban.step_n = 29
    env.quick_save(23)

    env.sokoban.value[:] = 99
    env.sokoban.step_n = 99
    env.quick_load(11)
    np.testing.assert_array_equal(env.sokoban.value, [7, 7, 7])
    assert env.sokoban.step_n == 7

    env.quick_load(23)
    np.testing.assert_array_equal(env.sokoban.value, [29, 29, 29])
    assert env.sokoban.step_n == 29

    env.quick_load()
    np.testing.assert_array_equal(env.sokoban.value, [1, 2, 3])
    assert env.sokoban.step_n == 0

    env.quick_delete(11)
    assert 11 not in env.save_states
    with pytest.raises(ValueError, match="slot 11"):
        env.quick_load(11)
    env.quick_delete(11)  # deletion is intentionally idempotent

    with pytest.raises(ValueError, match="non-negative"):
        env.quick_save(-1)
