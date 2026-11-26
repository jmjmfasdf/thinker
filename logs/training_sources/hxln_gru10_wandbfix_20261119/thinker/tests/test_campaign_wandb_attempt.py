import sys
from types import SimpleNamespace

import pytest
from thinker import util


@pytest.mark.parametrize('subname', ['', '_model'])
def test_fresh_attempt_does_not_resume_failed_wandb_history(monkeypatch, subname):
    calls = []
    fake = SimpleNamespace(init=lambda **kwargs: calls.append(kwargs), config=SimpleNamespace(update=lambda *a, **k: None))
    monkeypatch.setitem(sys.modules, 'wandb', fake)
    monkeypatch.setenv('DI72_WANDB_ATTEMPT', '16070_46')
    flags = SimpleNamespace(xpid='dynamic-imitation-spaceinvaders-voc-sub1-leavesession1', project='thinker')
    util.Wandb(flags, subname)
    assert calls[0]['id'] == flags.xpid + subname + '--16070_46'
    assert calls[0]['resume'] == 'never'
    assert vars(flags) == dict(xpid='dynamic-imitation-spaceinvaders-voc-sub1-leavesession1', project='thinker')


def test_unrelated_runs_keep_existing_wandb_identity(monkeypatch):
    calls = []
    monkeypatch.delenv('DI72_WANDB_ATTEMPT', raising=False)
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(init=lambda **kw: calls.append(kw), config=SimpleNamespace(update=lambda *a, **k: None)))
    util.Wandb(SimpleNamespace(xpid='existing', project='thinker'))
    assert calls[0]['id'] == 'existing'
    assert calls[0]['resume'] == 'allow'
