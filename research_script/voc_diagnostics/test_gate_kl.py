import numpy as np
import pytest

from gate_kl import bernoulli_scores, aggregate, METRICS


def test_known_bernoulli_kl_and_entropy_identity():
    target = np.array([0.2, 0.5, 0.8])
    delta = .05 * np.log(target / (1 - target))
    identical = bernoulli_scores(delta, target, .05)
    np.testing.assert_allclose(identical['kl_teacher_gate'], 0, atol=1e-15)
    neutral = bernoulli_scores(delta, np.full(3, .5), .05)
    expected = target*np.log(target/.5) + (1-target)*np.log((1-target)/.5)
    np.testing.assert_allclose(neutral['kl_teacher_gate'], expected, atol=1e-15)
    np.testing.assert_allclose(neutral['cross_entropy']-neutral['teacher_entropy'], expected, atol=1e-15)


def test_endpoint_probabilities_and_invalid_inputs():
    scores = bernoulli_scores(np.array([-100., 100.]), np.array([0., 1.]), .05)
    assert all(np.isfinite(v).all() for v in scores.values())
    with pytest.raises(ValueError):
        bernoulli_scores([0.], [-.1], .05)
    with pytest.raises(ValueError):
        bernoulli_scores([0.], [.5], 0.)


def test_aggregation_does_not_weight_long_runs_more():
    rows = []
    for subject, values in [(1, [0., 2.]), (2, [3.])]:
        for i, value in enumerate(values):
            rows.append(dict(game=0,subject=subject,leave_session=1,split='held_out',scope='all',run=f'{subject}-{i}',n=1000 if i==0 else 1,
                             **{m: value for m in METRICS}))
    summary = aggregate(rows)[-1]
    assert summary.loc[summary.split.eq('held_out'), 'kl_teacher_gate'].item() == 2.
