import importlib.util
import math
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('td_calibration',
    Path(__file__).parents[1] / 'diagnostics/audit_planning_td_calibration.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_pooled_rmse_matches_individual_errors_with_unequal_batch_sizes():
    # Actual errors [3] and [1,-1,1]: sums are 4,6,12 over four decisions.
    rows = [dict(n=1, bias=3, mae=3, rmse=3),
            dict(n=3, bias=1/3, mae=1, rmse=1),
            dict(n=0, bias=float('nan'), mae=float('nan'), rmse=float('nan'))]
    result = MODULE.pooled_errors(rows, count_key='n', prefix='')
    assert result['count'] == 4
    assert result['bias'] == 1
    assert result['mae'] == 1.5
    assert result['rmse'] == pytest.approx(math.sqrt(3))
    assert result['rmse'] != 2  # Averaging the two batch RMSEs is incorrect.


def test_resume_reconciliation_excludes_parent_and_rejects_missing_update():
    parent = dict(count=10, error_sum=1., absolute_error_sum=10., squared_error_sum=20.)
    final = dict(count=14, error_sum=5., absolute_error_sum=16., squared_error_sum=32.)
    logged = dict(count=4, error_sum=4., absolute_error_sum=6., squared_error_sum=12.)
    MODULE.reconcile(parent, final, logged)
    with pytest.raises(ValueError, match='count'):
        MODULE.reconcile(parent, final, dict(logged, count=3))
    with pytest.raises(ValueError, match='squared_error_sum'):
        MODULE.reconcile(parent, final, dict(logged, squared_error_sum=16))


@pytest.mark.parametrize('row', [
    dict(n=-1, bias=0, mae=0, rmse=0),
    dict(n=1, bias=2, mae=1, rmse=1),
    dict(n=1, bias=0, mae=2, rmse=1),
    dict(n=1, bias=float('nan'), mae=0, rmse=0),
])
def test_invalid_supported_moments_are_rejected(row):
    with pytest.raises(ValueError):
        MODULE.pooled_errors([row], count_key='n', prefix='')
