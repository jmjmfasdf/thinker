"""Paper styling is local to 01 and never changes its saved numeric tables."""
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'research_script'))
from analyses import human_behavior as human


@pytest.fixture
def tables(tmp_path):
    for session in (1, 2):
        directory = tmp_path / 'sub-001' / f'ses-{session:02d}'
        directory.mkdir(parents=True)
        actions = np.tile([0] * 6 + [2] * 7 + [0] * 2 + [3] * 5 + [4] * 4, 30)
        first = np.zeros(len(actions), dtype=bool)
        first[0] = True
        np.savez(directory / f'sub001-ses{session:02d}-block1-game1.npz',
                 action=np.eye(6)[actions], time=np.arange(len(actions)) * .02,
                 is_first=first, is_terminal=np.zeros(len(actions), dtype=bool))
    result, qa = human.analyze(1, 'pong', behavior_root=tmp_path)
    assert qa['status'] == 'completed'
    return result


@pytest.mark.parametrize('group', [False, True])
def test_all_panels_have_large_fonts_and_no_titles(tmp_path, tables, monkeypatch, group):
    if group:
        tables = {name: pd.concat([frame, frame.assign(subject=2)], ignore_index=True)
                  for name, frame in tables.items()}
    original = {name: frame.copy(deep=True) for name, frame in tables.items()}
    prior = {key: matplotlib.rcParams[key] for key in
             ('font.size', 'axes.labelsize', 'xtick.labelsize', 'legend.fontsize', 'pdf.fonttype')}
    saved = []
    legend_labels = []

    def inspect_figure(fig, path, **kwargs):
        assert fig._suptitle is None or not fig._suptitle.get_text()
        assert kwargs['dpi'] == 300
        for ax in fig.axes:
            assert all(not ax.get_title(loc) for loc in ('left', 'center', 'right'))
            if not ax.axison:
                continue
            for axis in (ax.xaxis, ax.yaxis):
                if axis.label.get_text():
                    assert axis.label.get_fontsize() == 26
                for tick in axis.get_ticklabels():
                    assert tick.get_fontsize() == 22
            legend = ax.get_legend()
            if legend is not None:
                assert all(text.get_fontsize() == 22 for text in legend.get_texts())
                legend_labels.extend(text.get_text() for text in legend.get_texts())
        saved.append(Path(path).name)

    monkeypatch.setattr(Figure, 'savefig', inspect_figure)
    artifacts = human.render(tmp_path / 'result', tables, 'This title must not appear')
    assert len(saved) == len(artifacts) == 22
    assert len({Path(name).stem for name in saved}) == 11
    assert {'NOOP', 'RIGHT', 'NOOP → active', 'Active → NOOP',
            'Active → any other action', 'Any action switch'} <= set(legend_labels)
    assert all(matplotlib.rcParams[key] == value for key, value in prior.items())
    for name, frame in tables.items():
        pd.testing.assert_frame_equal(frame, original[name])


def test_style_is_restored_after_render_failure(tmp_path, monkeypatch):
    original = matplotlib.rcParams['font.size']

    def fail(*args):
        assert matplotlib.rcParams['font.size'] == 22
        raise RuntimeError('Test rendering failure')

    monkeypatch.setattr(human, '_render_figures', fail)
    with pytest.raises(RuntimeError, match='Test rendering failure'):
        human.render(tmp_path, {}, 'unused')
    assert matplotlib.rcParams['font.size'] == original
