import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

DIAGNOSTICS=Path(__file__).parents[1]/'diagnostics'
sys.path.insert(0,str(DIAGNOSTICS))
SPEC=importlib.util.spec_from_file_location('query_predictors',DIAGNOSTICS/'analyze_planning_query_predictors.py')
MODULE=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(MODULE)


def query(decision,count):
    return dict(seed=1,stream=0,decision=decision,count=count,probability=.8-.1*count,
        root_q_mean=[.2*count,999.],root_visits=[count,0.],rollout_return=.1*count,depth=count)


def test_decisions_have_equal_weight_and_unvisited_q_is_excluded():
    rows=[query(0,i) for i in (1,2)]+[query(1,i) for i in (1,2,3,4)]
    base,extra,y,weight,metadata,clipped=MODULE.design(rows)
    assert weight[0]==1
    assert sum(weight[1:])==pytest.approx(1)
    # Unvisited Q=999 does not enter the maximum or common-action change.
    np.testing.assert_allclose(extra[:,2],[.2,.2,.4,.6])
    assert clipped==0 and len(metadata)==4
    with pytest.raises(ValueError,match='missing or duplicated'):
        MODULE.design([query(0,1),query(0,3)])


def test_crossfit_never_uses_a_test_stream_in_training():
    rng=np.random.default_rng(1)
    x=rng.normal(size=(160,2));y=3*x[:,0]-2*x[:,1]
    group=np.repeat(['a','b','c','d'],40)
    predicted,folds=MODULE.crossfit(x,y,np.ones(160),group)
    assert np.mean((predicted-y)**2)<.02
    assert len(folds)==4
    for fold in folds:
        assert not set(fold['train_groups'])&set(fold['test_groups'])
    with pytest.raises(ValueError,match='four'):
        MODULE.crossfit(x,y,np.ones(160),np.repeat(['a','b'],80))


def test_invalid_probability_and_negative_visits_fail_closed():
    rows=[query(0,1),query(0,2)]
    with pytest.raises(ValueError,match='probability'):
        MODULE.design([rows[0],dict(rows[1],probability=float('nan'))])
    with pytest.raises(ValueError,match='visits'):
        MODULE.design([rows[0],dict(rows[1],root_visits=[-1,0])])


def test_probability_sensitivity_retains_saturated_rows_and_weights():
    rows=[dict(query(0,1),probability=1.),dict(query(0,2),probability=0.),
          dict(query(1,1),probability=0.),dict(query(1,2),probability=.4),
          dict(query(1,3),probability=1.+np.finfo(np.float32).eps)]
    base,extra,y,weight,metadata,clipped=MODULE.design(rows,'probability')
    logbase,logextra,logy,logweight,logmetadata,logclipped=MODULE.design(rows)
    np.testing.assert_allclose(y,[-1.,.4,1.])
    np.testing.assert_array_equal(weight,logweight)
    np.testing.assert_array_equal(extra,logextra)
    assert metadata==logmetadata and clipped==logclipped==3
    # Initial context changes scale too; count and structural controls stay fixed.
    np.testing.assert_allclose(base[:,28],[1.,0.,0.])
    np.testing.assert_array_equal(base[:,:28],logbase[:,:28])
    np.testing.assert_array_equal(base[:,30:],logbase[:,30:])
    with pytest.raises(ValueError,match='response scale'):
        MODULE.design(rows,'unknown')


def test_fp32_probability_sum_roundoff_preserves_raw_verifier_input():
    rows=[dict(query(0,1),probability=1+np.finfo(np.float32).eps),query(0,2)]
    stages=[dict(seed=1,stream=0,decision=0,first_probability=1.)]
    restored=MODULE.restore_allocations(rows,stages)
    assert restored[0]['first_optional_probability']==rows[0]['probability']
    with pytest.raises(ValueError,match='differs'):
        MODULE.restore_allocations(rows,[dict(stages[0],first_probability=.9)])
