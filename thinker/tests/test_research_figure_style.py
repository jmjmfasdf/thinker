"""Paper style applies to every numbered individual/group export, not its data."""
from pathlib import Path
from types import SimpleNamespace
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'research_script'))
from common.figure_style import PAPER_RC, paper_style
from common import research_cli as cli
from analyses import voc_relationships as voc


def assert_style(fig, **kwargs):
    assert kwargs['dpi'] == 300
    assert kwargs['bbox_inches'] == 'tight'
    assert fig._suptitle is None or not fig._suptitle.get_text()
    shared = [getattr(fig, key, None) for key in ('_supxlabel', '_supylabel')]
    assert all(not text.get_visible() or not text.get_text() or text in shared
               for text in fig.texts)
    for ax in fig.axes:
        assert all(not ax.get_title(loc) for loc in ('left', 'center', 'right'))
        for axis in (ax.xaxis, ax.yaxis):
            assert axis.label.get_fontsize() == 26
            assert axis.get_offset_text().get_fontsize() == 22
            assert all(tick.get_fontsize() == 22
                       for tick in axis.get_ticklabels(which='both'))
        legend = ax.get_legend()
        if legend is not None:
            assert all(text.get_fontsize() == 22 for text in legend.get_texts())
            assert legend.get_title().get_fontsize() == 22
    for legend in fig.legends:
        assert all(text.get_fontsize() == 22 for text in legend.get_texts())
        assert legend.get_title().get_fontsize() == 22


def legacy_figure():
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0, 1], [1e6, 2e6], label='Observed')
    ax.set(xlabel='Time (s)', ylabel='Value', title='Panel title')
    ax.set_title('Left title', loc='left')
    ax.set_title('Right title', loc='right')
    ax.tick_params(which='both', labelsize=7)
    ax.legend(fontsize=8, title='Conditions', title_fontsize=6)
    fig.legend(*ax.get_legend_handles_labels(), fontsize=6, title='Series', title_fontsize=6)
    fig.suptitle('Main title')
    fig.text(.5, -.02, 'Small bottom explanation', fontsize=7)
    ax.text(.5, -.3, 'Axes-coordinate footer', transform=ax.transAxes, fontsize=7)
    ax.text(0, -1, 'Data annotation')
    cb = fig.colorbar(ax.imshow(np.arange(4).reshape(2, 2)), ax=ax, label='Color scale')
    cb.ax.tick_params(labelsize=6)
    fig.supxlabel('Shared time (s)')
    return fig


@pytest.mark.parametrize('number', [f'{i:02d}' for i in range(1, 14)])
@pytest.mark.parametrize('group', [False, True])
def test_every_canonical_dispatch_applies_style(number, group, tmp_path, monkeypatch):
    saved = []
    def inspect(fig, path, **kwargs):
        assert_style(fig, **kwargs)
        assert any(text.get_text() == 'Data annotation' for text in fig.axes[0].texts)
        assert fig._supxlabel.get_text() == 'Shared time (s)'
        assert fig._supxlabel.get_fontsize() == 26
        assert fig.get_size_inches()[0] >= 6.5
        saved.append(path)
    monkeypatch.setattr(Figure, 'savefig', inspect)
    def draw(*args):
        fig = legacy_figure()
        fig.tight_layout(rect=(0, .15, 1, .85))
        fig.savefig(tmp_path / 'test.png', dpi=80)
        plt.close(fig)
    monkeypatch.setattr(voc, 'render', lambda *args: None)
    monkeypatch.setattr(cli, 'render_neural', draw)
    if group and number == '10':
        from analyses import control_pattern_pipeline
        monkeypatch.setattr(control_pattern_pipeline, 'render_group', draw)
    before = {key: matplotlib.rcParams[key] for key in PAPER_RC}
    (cli._render_group if group else cli.render)(number, SimpleNamespace(render=draw), tmp_path, {}, 'unused')
    assert len(saved) == 1
    assert all(matplotlib.rcParams[key] == value for key, value in before.items())
    assert Figure.savefig is inspect


def test_nested_context_restores_state_even_on_error(tmp_path, monkeypatch):
    before = {key: matplotlib.rcParams[key] for key in PAPER_RC}
    save, layout = Figure.savefig, Figure.tight_layout
    with pytest.raises(RuntimeError, match='expected'):
        with paper_style():
            outer = Figure.savefig
            with voc.render_style():
                assert Figure.savefig is outer
                assert matplotlib.rcParams['font.size'] == 22
                raise RuntimeError('expected')
    assert Figure.savefig is save and Figure.tight_layout is layout
    assert all(matplotlib.rcParams[key] == value for key, value in before.items())


def test_shared_style_only_invalidates_render_signature(monkeypatch):
    modules = {number: __import__('analyses.' + name, fromlist=['render'])
               for number, name in cli.MODULES.items()}
    from analyses import group_results
    modules['group'] = group_results
    numeric = {number: cli.code_signature(module) for number, module in modules.items()}
    drawing = {number: cli._render_signature(number if number != 'group' else '04', module)
               for number, module in modules.items()}
    original = cli.sha
    monkeypatch.setattr(cli, 'sha', lambda path: 'new-style-hash' if Path(path).name == 'figure_style.py' else original(path))
    for number, module in modules.items():
        assert cli.code_signature(module) == numeric[number]
        assert cli._render_signature(number if number != 'group' else '04', module) != drawing[number]


@pytest.mark.parametrize('number', ['02', '03', '04', '11', '12', '13'])
def test_real_renderers_from_saved_tables_only(number, tmp_path, monkeypatch):
    tables = {
        '02': {'run_scores': pd.DataFrame([dict(subject=1, population=p, accuracy=.7, log_score=-.4)
                                           for p in ('all', 'persist', 'switch')])},
        '03': {'budget_distribution': pd.DataFrame([dict(subject=1, budget=k, probability=.5) for k in (1, 2)])},
        '04': {'voc_utility_regression': pd.DataFrame([dict(protocol='paired', nominal_step=1e7,
                mean_task_reward_difference=.3, utility_mean=.2, utility_voc_slope=.1,
                slope_ci_low=.05, slope_ci_high=.15)])},
    }
    name = {'11': 'encoding', '12': 'decoding', '13': 'rsa'}.get(number)
    if name:
        tables[number] = {name + '_run_effects': pd.DataFrame([
            dict(subject=1, run_label='r1', session=1, roi=roi, term='VoC', value=.1,
                 status='ok', stage='state_controls') for roi in ('ROI A', 'ROI B')])}
    frames = tables[number]
    original = {key: frame.copy(deep=True) for key, frame in frames.items()}
    seen = []
    def inspect(fig, path, **kwargs):
        assert_style(fig, **kwargs)
        seen.append(path)
    monkeypatch.setattr(Figure, 'savefig', inspect)
    monkeypatch.setattr(cli, 'prepare_native', lambda *a, **kw: pytest.fail('Rendering reopened raw inputs'))
    module = __import__('analyses.' + cli.MODULES[number], fromlist=['render'])
    cli.render(number, module, tmp_path, frames, 'Must not appear')
    assert len(seen) >= 2
    for key, frame in frames.items():
        pd.testing.assert_frame_equal(frame, original[key])


def test_actual_exports_and_constrained_layout(tmp_path):
    with paper_style():
        fig, ax = plt.subplots(layout='constrained')
        engine = fig.get_layout_engine()
        ax.plot([0, 1], [0, 1], label='Observed')
        ax.set(xlabel='Time (s)', ylabel='Value')
        ax.legend()
        fig.suptitle('Removed')
        for suffix in ('png', 'pdf'):
            path = tmp_path / ('styled.' + suffix)
            fig.savefig(path)
            assert path.stat().st_size > 1000
        assert fig.get_layout_engine() is engine
        plt.close(fig)


def test_long_labels_wrap_without_losing_words(tmp_path, monkeypatch):
    label = 'Mean log probability of the observed human action per decision'
    def inspect(fig, path, **kwargs):
        assert_style(fig, **kwargs)
        actual = fig.axes[0].get_ylabel()
        assert '\n' in actual
        assert actual.split() == label.split()
    monkeypatch.setattr(Figure, 'savefig', inspect)
    with paper_style():
        fig, ax = plt.subplots()
        ax.set_ylabel(label)
        fig.savefig(tmp_path / 'long.png')
        plt.close(fig)
