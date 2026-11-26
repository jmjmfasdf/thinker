"""Visited-to-visited root-Q absolute revision helper.

V and Q analyses now live in revision_behavior.py (05 and07).
The previous ad-hoc entrypoints are preserved once in legacy/research_script.
"""
import numpy as np
from common import native_replay as trace

def updates(q0,q1,n0,n1):
    valid=(n0>0)&(n1>0)
    return np.where(valid,np.abs(q1-q0),0).sum(axis=1)
