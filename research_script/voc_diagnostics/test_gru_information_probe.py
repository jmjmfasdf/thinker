import numpy as np
import pytest

from gru_information_probe import ridge_oof, decode_report
from gru_reward_readout import verify


def test_nested_probe_recovers_signal_and_holds_out_whole_streams():
    rng = np.random.default_rng(133)
    x = rng.normal(size=(192, 4))
    y = x[:, :2]
    groups = np.repeat(np.arange(12), 16)
    pred, fold, _ = ridge_oof(x, y, groups)
    assert np.max(np.abs(pred-y)) < .005
    assert all(len(np.unique(fold[groups == g])) == 1 for g in np.unique(groups))
    null, _, _ = ridge_oof(np.zeros_like(x), y, groups)
    assert np.mean((null-y)**2) > .5


def test_decode_differentiates_preserved_and_erased_counters():
    count = np.tile(np.arange(8), 12)
    depth = count // 2
    raw = np.log1p(np.c_[count, depth])
    report, _ = decode_report(dict(count=count, depth=depth,
        stream=np.repeat(np.arange(12), 8), gru=raw, features=np.zeros_like(raw)))
    assert report['readouts']['gru']['scores']['count']['rounded_accuracy'] == 1
    assert report['readouts']['full_voc']['scores']['count']['rounded_accuracy'] < .2


@pytest.mark.parametrize('column_target', [False, True])
def test_single_target_probe_preserves_shape_and_inner_cv_loss(column_target):
    rng = np.random.default_rng(391)
    x = rng.normal(size=(192, 4))
    y = 1.7*x[:, 0] - .4*x[:, 2]
    groups = np.repeat(np.arange(12), 16)
    target = y[:, None] if column_target else y
    pred, fold, alphas = ridge_oof(x, target, groups)
    reference, reference_fold, reference_alphas = ridge_oof(x, np.c_[y, y], groups)
    assert pred.shape == (len(y), 1)
    assert np.max(np.abs(pred[:, 0]-y)) < .005
    np.testing.assert_allclose(pred[:, 0], reference[:, 0], atol=1e-10)
    np.testing.assert_array_equal(fold, reference_fold)
    assert alphas == reference_alphas == [.01]*4


def test_pair_identity_rejects_mismatched_counts_and_features():
    a = dict(entries=[dict(hash='same', count=3, depth=2, features=[.1,.2])])
    verify(a, a)
    b = dict(entries=[dict(hash='same', count=4, depth=2, features=[.1,.2])])
    with pytest.raises(AssertionError): verify(a, b)
    b = dict(entries=[dict(hash='same', count=3, depth=2, features=[.1,.200001])])
    with pytest.raises(AssertionError): verify(a, b)
