"""Scoped opt-in autograd policy; never changes the ordinary RL graph."""
from contextlib import contextmanager
from contextvars import ContextVar

MODE = ContextVar("planning_meta_mode", default=None)


@contextmanager
def meta_scope(mode):
    token = MODE.set(mode)
    try:
        yield
    finally:
        MODE.reset(token)


def target(value):
    return value if MODE.get() == "inner" else value.detach()


def gate_features(value):
    return value if MODE.get() == "outer" else value.detach()
