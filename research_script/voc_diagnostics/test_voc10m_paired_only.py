"""CPU checks that disabling scoring preserves paired return evaluation."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch

import eval_voc10m as evaluator


@pytest.mark.parametrize('kind', ['score', 'score_zero', 'score_sign', 'episodes'])
def test_score_tasks_rejected_before_loading_checkpoint(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(evaluator, 'load_checkpoints',
        lambda *args: pytest.fail('Disabled task loaded a checkpoint'))
    with pytest.raises(ValueError, match='Score evaluation disabled'):
        evaluator.evaluate({'kind': kind}, tmp_path)


def test_benefit_evaluation_runs_only_paired_arms(tmp_path, monkeypatch):
    actor = {'voc_ema_gate_head_state_dict': {
        'weight': torch.zeros(2, 1), 'bias': torch.zeros(2)}}
    monkeypatch.setattr(evaluator, 'load_checkpoints',
        lambda args: (SimpleNamespace(), actor, {}, {'test': True}))
    calls = []

    def arm(*args, **kwargs):
        calls.append(kwargs['arm'])
        n = len(kwargs['targets'])
        continued = kwargs['arm'] == 'continue'
        return dict(entries=[dict(features=[0.], delta=0., p_continue=.5, hash=str(i))
                             for i in range(n)],
                    returns=[dict(net_truncated=.9 if continued else 0.,
                                  task_truncated=1. if continued else 0.,
                                  current_stage_cost=.1 if continued else 0.) for _ in range(n)],
                    tail_bounds=[0.]*n)

    monkeypatch.setattr(evaluator, 'arm_rollout', arm)
    task = dict(kind='benefit', bundle=str(tmp_path/'bundle.json'),
                real_step=2_000_000, smoke=True)
    result = evaluator.evaluate(task, tmp_path)
    assert calls == ['stop', 'continue', 'stop']
    assert result['voc']['pairs'] == 8
    np.testing.assert_allclose(result['voc']['mean_net_benefit'], .9)
    assert result['budget_comparisons'] == {}
    assert result['score_evaluation'] == 'DISABLED_BY_USER'
    assert all(key.startswith('voc/') for key in result['metrics'])
    assert not list(tmp_path.glob('score*'))
    assert not (tmp_path/'calibration.json').exists()
