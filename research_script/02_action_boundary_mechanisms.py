#!/usr/bin/env python3
"""Theme 02: M/U and action boundaries; results below outputs/02/."""
import sys
from common.figure_style import paper_style

@paper_style()
def entrypoint():
    if '--visualization' in sys.argv:
        sys.argv.remove('--visualization')
        from analyses.replay_themes import main
        main('02')
    elif '--replay-values' in sys.argv:
        sys.argv.remove('--replay-values')
        from analyses.value_updates import main
        main()
    else:
        from common.cli import main
        main('02')


if __name__ == '__main__':
    entrypoint()
