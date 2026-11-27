"""Shared paper-figure policy, scoped to rendering rather than numeric analysis.

Apply again before layout/export: older renderers explicitly request tiny fonts
and add narrative captions. Axis labels, legends and in-panel data annotations
are retained; titles and figure-level prose belong in the accompanying report.
"""
from contextlib import contextmanager
from functools import wraps
from textwrap import fill


PAPER_RC = {
    'font.size': 22,
    'axes.labelsize': 26,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'legend.fontsize': 22,
    'legend.title_fontsize': 22,
    'axes.labelpad': 12,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.dpi': 300,
}


def _prepare_figure(fig):
    """Idempotently style even explicitly sized ticks, colorbars and legends."""
    shared_labels = [getattr(fig, name, None) for name in ('_supxlabel', '_supylabel')]
    for text in fig.texts:
        if text in shared_labels:
            text.set_fontsize(26)
        else:
            text.set_text('')
            text.set_visible(False)
            text.set_in_layout(False)
    rows = columns = 1
    for ax in fig.axes:
        for loc in ('left', 'center', 'right'):
            ax.set_title('', loc=loc)
        for name in ('xaxis', 'yaxis', 'zaxis'):
            axis = getattr(ax, name, None)
            if axis is None:
                continue
            axis.label.set_fontsize(26)
            label = axis.label.get_text()
            # Long single-line labels can extend above/below even a tight
            # export. Keep mathematical expressions intact and all wording.
            if '$' not in label:
                width = 38 if name == 'xaxis' else 32
                axis.label.set_text('\n'.join(fill(line, width=width, break_long_words=False,
                                                   break_on_hyphens=False) for line in label.split('\n')))
            axis.set_tick_params(which='both', labelsize=22)
            axis.get_offset_text().set_fontsize(22)
        # A footer can also have been attached in axes coordinates. Do not
        # remove data-coordinate annotations, even when their values are < 0.
        for text in ax.texts:
            if text.get_transform() is ax.transAxes and text.get_position()[1] < 0:
                text.set_text('')
                text.set_visible(False)
                text.set_in_layout(False)
        legend = ax.get_legend()
        if legend is not None:
            _style_legend(legend)
        if not hasattr(ax, '_colorbar') and hasattr(ax, 'get_subplotspec'):
            spec = ax.get_subplotspec()
            if spec is not None:
                grid = spec.get_topmost_subplotspec().get_gridspec()
                rows, columns = max(rows, grid.nrows), max(columns, grid.ncols)
    for legend in fig.legends:
        _style_legend(legend)
    # Keep the original aspect/layout when already spacious (notably 01).
    # Small legacy panels otherwise have no room for 22/26-pt labels.
    width, height = fig.get_size_inches()
    fig.set_size_inches(max(width, 6.5 * columns), max(height, 5.5 * rows), forward=False)


def _style_legend(legend):
    for text in legend.get_texts():
        text.set_fontsize(22)
    legend.get_title().set_fontsize(22)


@contextmanager
def paper_style():
    """Title/caption-free exports; restore all matplotlib state on exit/error.

Nestable so common dispatchers and standalone renderers can share this policy.
No raw input access or changes to numerical tables are performed here.
"""
    import matplotlib as mpl
    from matplotlib.figure import Figure
    from unittest.mock import patch

    if getattr(Figure.savefig, '_research_paper_style', False):
        yield
        return
    original_save = Figure.savefig
    original_layout = Figure.tight_layout

    @wraps(original_layout)
    def layout(fig, *args, **kwargs):
        _prepare_figure(fig)
        # Historical rects reserve space for titles/footnotes now removed.
        kwargs.pop('rect', None)
        kwargs.setdefault('pad', 1.2)
        kwargs.setdefault('w_pad', 2.)
        kwargs.setdefault('h_pad', 2.)
        return original_layout(fig, *args, **kwargs)

    @wraps(original_save)
    def save(fig, *args, **kwargs):
        _prepare_figure(fig)
        # Do not replace a renderer's constrained/compressed layout engine.
        if fig.get_layout_engine() is None:
            layout(fig)
        kwargs.update(dpi=300, bbox_inches='tight')
        return original_save(fig, *args, **kwargs)

    save._research_paper_style = True
    with mpl.rc_context(PAPER_RC), patch.object(Figure, 'savefig', save), \
            patch.object(Figure, 'tight_layout', layout):
        yield
