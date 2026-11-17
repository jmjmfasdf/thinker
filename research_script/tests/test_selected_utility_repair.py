import importlib.util
from pathlib import Path

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location('repair',
    Path(__file__).parents[1]/'diagnostics/repair_selected_planning_utility.py')
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def test_mandatory_draw_preserved_optional_forcing_retained():
    control = np.array([0, 2, 2, 1])
    conditional = np.array([1, 0, 1, 0])
    trace = []
    actual = M.force_pending(control, conditional, [True, True, False, True],
        [False, True, True, False], preserve=True, trace=trace)
    assert actual.tolist() == [0, 0, 2, 1]
    assert control.tolist() == [0, 2, 2, 1]
    assert trace == [dict(streams=[0, 3], natural=[0, 1], legacy=[1, 0])]


def test_legacy_comparator_reproduces_original_override():
    actual = M.force_pending(np.array([0, 2]), np.array([1, 0]), [True, True],
        [False, True], preserve=False, trace=[])
    assert actual.tolist() == [1, 0]


def test_reuse_cannot_bypass_exact_reference():
    class E:
        @staticmethod
        def verify(a, b):
            assert a['entries'] == b['entries']
    M.validate_cached({'stop': {'entries': [1]}, 'continue': {'entries': [1]}}, [1], E)
    with pytest.raises(ValueError, match='exact natural'):
        M.validate_cached({'stop': {'entries': [1]}, 'continue': {'entries': [2]}}, [1], E)
