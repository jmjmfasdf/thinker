#!/usr/bin/env python3
"""COSYNE Figure 2A–C on the dynamic replay, exported separately by game.

The retained full legacy behavioral analysis is analyses/compat_impl/01_behavioral_analysis.py.
"""
from panels.cli import main
if __name__ == '__main__':
    import sys
    if '--legacy' in sys.argv:
        sys.argv.remove('--legacy')
        from common.compat import load
        load('01_behavioral_analysis.py','01')
    else:
        main('01')
