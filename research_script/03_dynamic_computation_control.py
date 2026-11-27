#!/usr/bin/env python3
"""Theme 03: VOC control and planning budgets; results below outputs/03/."""
import sys
from common.figure_style import paper_style

@paper_style()
def entrypoint():
    if '--visualization' in sys.argv:
        sys.argv.remove('--visualization')
        from analyses.replay_themes import main
        main('03')
    elif '--training-budget' in sys.argv:
        sys.argv.remove('--training-budget')
        from analyses.planning_budget_logs import main
        main()
    else:
        from common.cli import main
        main('03')


if __name__ == '__main__':
    entrypoint()
