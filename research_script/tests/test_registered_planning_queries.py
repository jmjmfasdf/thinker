import importlib.util
from pathlib import Path
import sys

import pytest

DIAGNOSTICS=Path(__file__).parents[1]/'diagnostics'
sys.path.insert(0,str(DIAGNOSTICS))
SPEC=importlib.util.spec_from_file_location('registered_queries',DIAGNOSTICS/'probe_registered_planning_queries.py')
MODULE=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(MODULE)


def reference():
    return [dict(seed=8301,stream=stream,decision=decision,length=2)
            for stream in range(8) for decision in range(128,1664)]


def test_registered_observer_requires_every_original_decision():
    rows=reference()
    assert MODULE.validate_observed(rows,list(reversed(rows)),8301)==12288
    with pytest.raises(ValueError,match='missing or duplicated'):
        MODULE.validate_observed(rows[:-1],rows,8301)
    with pytest.raises(ValueError,match='missing or duplicated'):
        MODULE.validate_observed(rows[:-1]+[rows[0]],rows,8301)


def test_changed_behavior_is_not_an_observer_only_result():
    rows=reference();changed=[dict(r) for r in rows];changed[0]['length']=3
    with pytest.raises(ValueError,match='observer changed'):
        MODULE.validate_observed(changed,rows,8301)
