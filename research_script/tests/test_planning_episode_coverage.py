import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).parents[1]/'diagnostics/probe_planning_episode_coverage.py'
SPEC = importlib.util.spec_from_file_location('episode_coverage', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_imagined_terminal_does_not_advance_episode_or_real_clock():
    clock = MODULE.EpisodeClock(2)
    clock.step([False,False], [True,True], [False,False])
    assert clock.decision == 0 and clock.records == {}
    assert clock.episode.tolist() == [0,0]


def test_real_terminal_and_truncation_start_new_episode_on_next_decision():
    clock = MODULE.EpisodeClock(2)
    clock.step([True,True], [False,False], [False,False])
    clock.step([True,True], [True,False], [False,True])
    assert clock.records[(0,1)] == dict(episode=0,episode_step=1,episode_ended=True)
    assert clock.records[(1,1)] == dict(episode=0,episode_step=1,episode_ended=True)
    clock.step([True,True], [False,False], [False,False])
    assert clock.records[(0,2)] == dict(episode=1,episode_step=0,episode_ended=False)
    assert clock.decision == 3
    clock.reset()
    assert clock.records == {} and clock.decision == 0


def test_asynchronous_barrier_is_rejected_instead_of_misaligned():
    clock = MODULE.EpisodeClock(2)
    with pytest.raises(ValueError,match='synchronous'):
        clock.step([True,False], [False,False], [False,False])
