"""Separate game-specific scientific panels, with tables and captions alongside.

Thin lines/dots are participants; thick lines and shading are equal-participant
means and SEM, only on support shared by all requested participants. The M–U
probability surface separately follows COSYNE's >=4/6 local-support rule. Forest
plots remain diagnostics in the earlier outputs, not main research panels.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common.paths import GAMES, output, tag
from common.io import sha, write_json
from common.statistics import group_inference

COLORS = ['#367aa8', '#dd8a35', '#785b9e', '#399a87', '#cf6678', '#737d87']
POP_LABELS = {'overall': 'Any action switch', 'noop_to_active': 'No input → active',
              'active_to_noop': 'Active → no input', 'active_to_other': 'Active → other action'}
THEMES = ['01', '02', '03', '04', '05', '09', '10', '12', '13']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
    'axes.spines.right': False, 'pdf.fonttype': 42, 'savefig.facecolor': 'white'})


def table(root, name, frame):
    (root/'tables').mkdir(parents=True, exist_ok=True)
    frame.to_csv(root/'tables'/f'{name}.csv', index=False)


def save(fig, root, name, game, caption, files):
    folder = root/'figures'; folder.mkdir(parents=True, exist_ok=True)
    fig.suptitle(GAMES[game], fontsize=14, fontweight='bold')
    if fig.get_layout_engine() is None: fig.tight_layout(rect=(0, .015, 1, .95))
    for ext in ['png', 'pdf']:
        path = folder/f'{name}_game{game}.{ext}'
        fig.savefig(path, dpi=180, bbox_inches='tight'); files.append(str(path))
    (folder/f'{name}_game{game}.caption.txt').write_text(caption+'\n', encoding='utf-8')
    plt.close(fig)


def read_units(outputs, number, game, subjects):
    tables = {}; sources = {}; missing = []
    for s in subjects:
        p = output(number, outputs)/'panels'/tag(s, game)
        if not (p/'manifest.json').exists(): missing.append(s); continue
        m = json.loads((p/'manifest.json').read_text())
        if m.get('status') != 'COMPLETED': missing.append(s); continue
        sources[str(p/'manifest.json')] = sha(p/'manifest.json')
        for f in (p/'tables').glob('*.csv'):
            try: d = pd.read_csv(f)
            except pd.errors.EmptyDataError: continue
            if len(d): tables.setdefault(f.stem, []).append(d.assign(subject=s, game=game))
    return {k: pd.concat(v, ignore_index=True) for k, v in tables.items()}, sources, missing


def subject_curve(frame, x, y):
    # Averaging over available runs inside a participant precedes all group plots.
    return frame.groupby(['subject', x], as_index=False)[y].mean()


def curve(ax, frame, x, y, label, color, n_subjects=6, thin=True):
    d = subject_curve(frame, x, y)
    for s, r in d.groupby('subject'):
        if thin: ax.plot(r[x], r[y], color=color, alpha=.20, lw=.75)
    z = d.groupby(x)[y].agg(['mean', 'sem', 'count']).reset_index()
    z.loc[z['count'].ne(n_subjects), ['mean', 'sem']] = np.nan
    ax.plot(z[x], z['mean'], color=color, lw=2.2, label=label)
    ax.fill_between(z[x].to_numpy(float), (z['mean']-z['sem']).to_numpy(float),
                    (z['mean']+z['sem']).to_numpy(float), color=color, alpha=.15)
    return z


def paired(ax, data, columns, labels, ylabel):
    z = data.groupby('subject')[columns].mean().reindex(range(1, 7))
    for s, row in z.iterrows():
        ax.plot(range(len(columns)), row.to_numpy(), '-o', color=COLORS[(s-1)%6], lw=.9, ms=4, alpha=.8)
    # A dot for every participant remains visible; no distribution over decisions.
    ax.plot(range(len(columns)), z.mean().to_numpy(), 'k_', ms=15, mew=2)
    ax.set(xticks=range(len(columns)), xticklabels=labels, ylabel=ylabel)
    return z


def render01(t, root, game, files, ns):
    if 'n_original_observations' not in t['source_alignment']:
        raise ValueError('Human-only panels require original observations; run panels.raw_behavior first')
    d = t['persistence_curves']
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7))
    for active, label, color in [(0, 'No input', COLORS[0]), (1, 'Active input', COLORS[1])]:
        curve(axes[0], d.loc[d.active.eq(active)], 'seconds', 'survival', label, color, ns)
    for col, label, color in [('to_noop', 'Release input', COLORS[0]), ('to_active', 'Different active action', COLORS[1])]:
        curve(axes[1], d.loc[d.active.eq(1)], 'seconds', col, label, color, ns)
    axes[0].set(title='Action persistence', xlabel='Elapsed bout duration (s)', ylabel='Survival probability', ylim=(0, 1))
    axes[1].set(title='Active-bout exit routes', xlabel='Elapsed bout duration (s)', ylabel='Cumulative incidence', ylim=(0, 1))
    for ax in axes: ax.legend(frameon=False, fontsize=8)
    save(fig, root, 'cosyne_fig2A_action_persistence', game,
         'Censoring-aware survival and competing exits from ALL original human observations, with original source timestamps. Episode ends are censored. Each human run is included once; no checkpoint replication or downsampling. Equal runs then participants; thin lines participants, band SEM.', files)
    p = t['duration_prediction']; fig, axes = plt.subplots(1, 3, figsize=(11, 3.7))
    for ax, pop in zip(axes, ['overall', 'noop_to_active', 'active_to_noop']):
        wide = p.loc[p.population.eq(pop)].pivot_table(index=['subject', 'run_label'], columns='model', values='log_score').reset_index()
        paired(ax, wide, ['previous_action', 'action_and_duration'], ['Previous\naction', '+ Duration'], 'Held-out log score (nat/decision)')
        ax.set_title(POP_LABELS[pop], fontsize=10)
    save(fig, root, 'cosyne_fig2B_duration_prediction', game,
         'Leave-session-out prediction: previous-action model versus action-specific log-duration effects. Preprocessing uses only training sessions. Points are equal-run participant means; no independent checkpoint replicas.', files)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7)); rows = []
    for ax, pop in zip(axes, ['noop_to_active', 'active_to_other', 'active_to_noop']):
        r = t['duration_curves'].loc[t['duration_curves'].population.eq(pop)].copy()
        r.loc[r.n.lt(20), 'probability'] = np.nan
        r['seconds'] = np.sqrt(r.lower_seconds.clip(lower=.0025)*r.upper_seconds.replace(np.inf, 20))
        z = curve(ax, r, 'seconds', 'probability', 'Switch probability', COLORS[0], ns)
        ax.set(title=POP_LABELS[pop], xlabel='Elapsed action duration (s)', ylabel='Probability per recorded observation', xscale='log')
        ax.set_ylim(bottom=0)
        rows.append(z.assign(population=pop))
    table(root, 'duration_group_curves', pd.concat(rows))
    save(fig, root, 'cosyne_fig2C_duration_switch_probability', game,
         'Empirical conditional switch probabilities from ALL original human observations by common log-duration bins in source seconds. Bin edges and mean duration are saved. Each run requires 20 observations per bin; group lines require all participants. The original active→other-action endpoint and the separate active→no-input endpoint are both retained. These human-only panels use no replay-grid downsampling.', files)


def render02(t, root, game, files, ns):
    pairs = t['matched_means']; overall = pairs.loc[pairs.population.eq('overall')]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.7))
    for ax, metric, title in zip(axes, ['M_z', 'U_relative'], ['Initial action support M', 'Whole-search updating U']):
        paired(ax, overall.loc[overall.metric.eq(metric)], ['persist', 'switch'], ['Persist', 'Switch'],
               'Within-run z score' if metric == 'M_z' else 'U / run mean U')
        ax.set_title(title)
    save(fig, root, 'cosyne_fig2D_matched_MU', game,
         'Matched switch and persistence decisions: same run, preceding action, duration/time/reward strata, nearest visual descriptor; matching with replacement. M standardized within run; U_all divided by run mean. Six paired participant means. Match identities and distances are saved.', files)
    from .mu_heatmaps import append_group_files
    append_group_files(root, game, files)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    if 'k_curves' in t:
        for pop, color in zip(['noop_to_active', 'active_to_noop'], COLORS):
            part = t['k_curves'].loc[t['k_curves'].population.eq(pop)]
            curve(axes[0], part, 'percentile', 'probability', POP_LABELS[pop], color, ns)
        axes[0].legend(frameon=False, fontsize=8)
    axes[0].set(title='Adjusted transition probability', xlabel='K percentile within run', ylabel='Probability', ylim=(0, 1))
    paired(axes[1], overall.loc[overall.metric.eq('K_z')], ['persist', 'switch'], ['Persist', 'Switch'], 'K (within-run z)')
    axes[1].set_title('Matched competition-weighted update')
    save(fig, root, 'cosyne_fig3C_K_transition', game,
         'K=4 U_all sigmoid(M) sigmoid(-M), separate from learned VOC. Natural cubic percentile curves adjust visual/motion, preceding action, duration, prior reward and run intercepts; probabilities standardized over observed contexts with equal-run weights. Support: ≥200 local observations and ≥20 in ≥3 runs; group line requires all participants. Right: matched K levels. These are associations, not causal effects.', files)
    units = pairs.groupby(['subject', 'population', 'metric']).difference.mean().reset_index().rename(columns={'difference': 'value'})
    inference = group_inference(units, ['population', 'metric'], expected_subjects=ns)
    table(root, 'matched_inference', inference)


def render03(t, root, game, files, ns):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7))
    for ax, pop in zip(axes, ['start', 'continue']):
        r = t['gate_calibration'].loc[t['gate_calibration'].population.eq(pop)].copy()
        r['center'] = (r['bin']+.5)/10
        curve(ax, r, 'center', 'actual', 'Observed continuation', COLORS[0], ns)
        curve(ax, r, 'center', 'predicted', 'Mean gate probability', COLORS[1], ns, thin=False)
        ax.plot([0, 1], [0, 1], '--', color='gray', lw=.8)
        ax.set(title='Start imagination' if pop == 'start' else 'Continue imagination', xlabel='Gate probability bin', ylabel='Continuation probability', ylim=(0, 1), xlim=(0, 1)); ax.legend(fontsize=8)
    save(fig, root, 'gate_probability_execution', game,
         'Voluntary choices only, pre-choice gate probabilities versus actual continuation; RESET counts as continuation. Forced STOP is censored. Separate initiation and subsequent continuation, equal runs then participants.', files)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    b = t['budget_distribution']; curve(ax, b, 'budget', 'probability', 'Observed budget', COLORS[0], ns)
    ax.set(xlabel='Additional imagined computations', ylabel='Fraction of decisions', title='Dynamic planning budget')
    save(fig, root, 'planning_budget_distribution', game,
         'Distribution includes B=0. Equal run and participant weighting; the delivered native variable-length trace is used without padding.', files)


def render04(t, root, game, files, ns):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={'width_ratios': [2, 1]})
    d = t['update_composition']; d = d.copy(); d.loc[d.n_pairs.lt(20), 'value'] = np.nan
    u = d.groupby(['subject', 'branch', 'status2_position']).value.mean().reset_index()
    z = u.groupby(['branch', 'status2_position']).value.agg(['mean', 'count']).reset_index()
    z.loc[z['count'].ne(ns), 'mean'] = np.nan
    matrix = z.pivot(index='status2_position', columns='branch', values='mean').reindex(columns=['ongoing', 'strongest_alternative', 'other_actions'])
    limit = np.nanmax(np.abs(matrix.to_numpy())) if np.isfinite(matrix.to_numpy()).any() else 1.
    im = axes[0].imshow(matrix, aspect='auto', cmap='RdBu_r', vmin=-max(limit, 1e-8), vmax=max(limit, 1e-8))
    axes[0].set(xticks=[0, 1, 2], xticklabels=['Ongoing\naction', 'Strongest initial\nalternative', f'Other {7 if game == 0 else 4}\nactions'],
                ylabel='Imagination index (one-based)', title='Switch − persist update contribution')
    axes[0].set_yticks(range(len(matrix)), matrix.index.to_numpy(int)+1)
    fig.colorbar(im, ax=axes[0], label='Difference / run mean U')
    r = t['matched_means']; r = r.loc[r.population.eq('overall')&r.metric.eq('U_relative')]
    paired(axes[1], r, ['persist', 'switch'], ['Persist', 'Switch'], 'U / run mean U'); axes[1].set_title('Whole-search update')
    table(root, 'update_composition_group', z)
    save(fig, root, 'cosyne_fig3A_search_update_composition', game,
         'Matched switch–persist update differences by imagination index and initial branch relative to the ongoing action. Both members must have the indexed computation; unobserved later events are not zero-filled. At an observed index, other-branch contributions are zero. ≥20 matched pairs/run/index; group cells require all participants. Thus late-index effects are conditional on surviving search length, not an unconditional fixed-budget effect. Right includes B=0 in whole-search U_all.', files)
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 6), gridspec_kw={'height_ratios': [1, 1.6]})
    m = np.linspace(-4, 4, 161); from scipy.special import expit
    axes[0].plot(m, 4*expit(m)*expit(-m), color=COLORS[2], lw=2)
    axes[0].set(xlabel='Initial margin M', ylabel='w(M)', title='Competition weight: K = U × w(M)')
    if 'endpoint_sensitivity' in t:
        c = t['endpoint_sensitivity']
        curve(axes[1], c, 'M', 'observed', 'Observed local U slope', COLORS[0], ns)
        # Display the exact shape with cross-run-trained gain, not a curve fitted to K.
        g = subject_curve(c, 'M', 'theory').groupby('M').theory.agg(['mean', 'count']).reset_index()
        g.loc[g['count'].ne(ns), 'mean'] = np.nan
        axes[1].plot(g.M, g['mean'], '--', color=COLORS[2], label='w(M), gain from other runs')
        supported = c.groupby('M')['observed'].count().eq(ns).sum()
    else: supported = 0
    if not supported:
        eligible = int(t.get('endpoint_sensitivity_qa', pd.DataFrame({'n': [0]})).n.sum())
        message = 'No decisions with two valid root SEARCH choices' if eligible == 0 else 'No M range supported by all six participants'
        axes[1].text(.5, .92, message, transform=axes[1].transAxes, ha='center', va='top', fontsize=9)
    axes[1].axhline(0, color='gray', lw=.7); axes[1].set(xlabel='Initial world-model margin M', ylabel='Policy-change sensitivity\n(pp per +0.01 raw U)', title='Same-head root policy change')
    if axes[1].get_legend_handles_labels()[0]: axes[1].legend(frameon=False, fontsize=8)
    save(fig, root, 'cosyne_fig3B_policy_change_sensitivity', game,
         'Endpoint |p_last−p_first| uses only SEARCH-head choices at root, relative to the fixed strongest initial alternative. Requires ≥2 root choices; B=0 and single-root queries are undefined, not zero. Gaussian local U association uses raw M/U plus visual/action/duration/reward context and local run intercepts; original bandwidth .30 and .01 histogram approximation. K is not a predictor. Supported individual curves require ≥3 locally supported runs; group requires all six. Dashed shape has a nonnegative gain learned on other runs. This concerns imagined allocation, not final REAL choice.', files)


def prediction_panel(outputs, number, game, subjects, root, files, source_name, name):
    parts = []
    for s in subjects:
        f = output(number, outputs)/tag(s, game)/'tables'/f'{source_name}.csv'
        if f.exists():
            manifest = json.loads((f.parent.parent/'manifest.json').read_text())
            if manifest.get('status') != 'COMPLETED' or sha(f) != manifest['artifacts'][str(f.relative_to(f.parent.parent))]:
                raise ValueError('Prediction source changed: '+str(f))
            parts.append(pd.read_csv(f).assign(subject=s))
    if not parts: return
    d = pd.concat(parts, ignore_index=True)
    if 'population' not in d: d['population'] = 'Actor soft-target reconstruction'
    metric = 'log_score' if 'log_score' in d else 'cross_entropy'
    units = d.groupby(['subject', 'population', 'model'])[metric].mean().reset_index()
    table(root, name+'_subject_scores', units)
    pops = list(units.population.unique()); fig, axes = plt.subplots(1, len(pops), figsize=(4*len(pops), 4.3), squeeze=False)
    for ax, pop in zip(axes[0], pops):
        r = units.loc[units.population.eq(pop)]; models = sorted(r.model.unique())
        for s, z in r.groupby('subject'):
            ax.plot(range(len(models)), z.set_index('model')[metric].reindex(models), '-o', color=COLORS[(s-1)%6], lw=1, ms=4)
        ax.set(xticks=range(len(models)), xticklabels=[m.replace('_', ' ') for m in models], ylabel=metric+' (nat/decision)', title=POP_LABELS.get(pop, pop))
        ax.tick_params(axis='x', labelrotation=35)
        for label in ax.get_xticklabels(): label.set_ha('right'); label.set_fontsize(8)
    save(fig, root, name, game,
         'Previously validated held-out readouts, same checkpoint for train/test and omitted session as test. Each line is a participant, with equal run weights. Absolute scores and all nested model stages are displayed; predictive association is not causal evidence for human planning.', files)


def render09(t, root, game, files, ns):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8)); titles = {'root_value': 'Initial state value', 'visual_change': 'Observed visual change', 'M': 'Initial action support'}
    for ax, metric in zip(axes, titles):
        r = t['state_allocation'].loc[t['state_allocation'].metric.eq(metric)].copy(); r['percentile'] = (r['bin']+.5)*10
        curve(ax, r, 'percentile', 'budget', 'Imagined computations', COLORS[0], ns)
        ax.set(title=titles[metric], xlabel='Within-run percentile bin', ylabel='Mean additional computations')
    save(fig, root, 'state_dependent_computation', game,
         'Descriptive state–budget curves; each predictor binned independently within run, preserving ties. Equal runs then participants. These curves complement the adjusted regression, and are neither causal controllability nor a test of real reward benefit.', files)


def render12(t, root, game, files, ns):
    d = t['event_locked']; metrics = ['M_z', 'U_relative', 'K_z', 'additional_computations', 'ema_initial']
    labels = ['Initial support M (z)', 'U / run mean U', 'K (z)', 'Imagined computations', 'Pre-choice VOC']
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.8))
    for ax, metric, label in zip(axes, metrics, labels):
        for pop, color in zip(['noop_to_active', 'active_to_noop'], COLORS):
            r = d.loc[d.metric.eq(metric)&d.population.eq(pop)].copy(); r.loc[r.n.lt(10), 'value'] = np.nan
            curve(ax, r, 'lag', 'value', POP_LABELS[pop], color, ns)
        ax.axvline(0, color='gray', ls='--', lw=.8); ax.set(xlabel='Replay decisions from transition', ylabel=label)
    axes[0].legend(frameon=False, fontsize=8)
    save(fig, root, 'bidirectional_transition_trajectories', game,
         'Observed transition-centered trajectories, preserving episode/run boundaries. ±8 replay decisions; actual mean elapsed seconds are saved in the table. Overlapping windows may share observations; SEM is across participants, not transition events. Post-transition curves are descriptive and are not inputs to causal or prospective prediction.', files)


def render10(outputs, game, root, files):
    p = output(10, outputs)/'tables'
    if not (p/'state_outcomes.csv').exists(): return 'BLOCKED_INPUT'
    d = pd.read_csv(p/'state_outcomes.csv'); scores = pd.read_csv(p/'previous_pilot_scores.csv')
    names = {0: ['enduro'], 1: ['pong'], 2: ['spaceinvaders', 'space_invaders', 'space invaders']}
    def select(frame): return frame.loc[frame.game.astype(str).str.lower().isin(names[game])].copy()
    d = select(d); scores = select(scores)
    if not len(d): return 'BLOCKED_INPUT'
    table(root, 'paired_utility_states', d); table(root, 'paired_utility_scores', scores)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for protocol, r in scores.groupby('protocol'):
        r = r.sort_values('nominal_step')
        axes[0].plot(r.nominal_step/1e6, r.mean_task_reward_difference, 'o-', label=protocol)
        axes[1].plot(r.nominal_step/1e6, r.mean_net_utility, 'o-', label=protocol)
    for ax, title in zip(axes, ['Discounted task-reward difference', 'Net utility after computation cost']):
        ax.axhline(0, color='gray', lw=.7); ax.set(xlabel='Training steps (millions)', ylabel=title); ax.legend(fontsize=8)
    save(fig, root, 'paired_environment_utility', game,
         'Existing sub001 temperature-.01 paired environment pilot only. Each protocol is kept separate; no new score or environment evaluation launched. Finite-horizon outcomes are not the critic long-horizon target.', files)
    return 'COMPLETED_SUPPLEMENTARY'


def aggregate(outputs, subjects, games, numbers=None):
    reports = []
    if '02' in (numbers or THEMES):
        from .mu_heatmaps import build
        build(outputs, subjects, games)
    for number in (numbers or THEMES):
        if number == '13': continue  # Existing HSMM already has game-specific artifacts.
        for game in games:
            root = output(number, outputs)/'panels'/f'game{game}'; root.mkdir(parents=True, exist_ok=True)
            t, sources, missing = read_units(outputs, number, game, subjects); files = []
            if number == '10': status = render10(outputs, game, root, files)
            elif missing:
                status = 'INCOMPLETE_COHORT'
            else:
                for name, frame in t.items(): table(root, name, frame)
                render = {'01': render01, '02': render02, '03': render03, '04': render04,
                          '09': render09, '12': render12}.get(number)
                if render: render(t, root, game, files, len(subjects))
                if number in ['04', '05', '12']:
                    prediction_panel(outputs, number, game, subjects, root, files,
                        'actor_run_scores' if number == '04' else 'run_scores',
                        'actor_readout_comparison' if number == '04' else 'heldout_prediction_comparison')
                status = 'COMPLETED'
            write_json(root/'manifest.json', dict(status=status, game=game, game_name=GAMES[game],
                subjects=subjects, missing_subjects=missing, sources=sources,
                renderer_sha256=sha(__file__),
                heatmap_manifest_sha256=sha(root/'heatmap/manifest.json') if number=='02' else None,
                figures={str(Path(f).relative_to(root)): sha(f) for f in files},
                scope='Each panel contains one game only; all new estimates use dynamic 10M held-out caches.'))
            reports.append(dict(theme=number, game=game, status=status, n_panels=len(files)//2, path=str(root)))
    report = pd.DataFrame(reports)
    index = output(0, outputs)/'panel_campaign'; index.mkdir(parents=True, exist_ok=True)
    import html, os
    rows = []; cards = []
    for item in reports:
        folder = Path(item['path'])
        for p in sorted((folder/'figures').glob('*.png')):
            cap = p.with_suffix('.caption.txt'); caption = cap.read_text() if cap.exists() else ''
            rel = os.path.relpath(p, index); pdf = os.path.relpath(p.with_suffix('.pdf'), index)
            rows.append(dict(theme=item['theme'], game=item['game'], game_name=GAMES[item['game']],
                             panel=p.stem, png=str(p), pdf=str(p.with_suffix('.pdf')), caption=caption))
            cards.append(f'<article><h2>{item["theme"]} · {html.escape(GAMES[item["game"]])} · {html.escape(p.stem)}</h2>'
                f'<a href="{html.escape(pdf)}">Vector PDF</a><a href="{html.escape(rel)}"><img loading="lazy" src="{html.escape(rel)}"></a>'
                f'<p>{html.escape(caption)}</p></article>')
    pd.DataFrame(rows).to_csv(index/'panel_index.csv', index=False)
    (index/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Dynamic Thinker panels</title>'
        '<style>body{max-width:1400px;margin:32px auto;padding:0 24px;font:16px/1.5 system-ui;color:#203040;background:#f4f6f8}'
        'article{background:white;padding:24px;margin:24px 0;border-radius:8px}h2{font-size:18px}img{width:100%;margin-top:14px}'
        'p{font-size:14px;color:#475567}a{color:#24629b}</style>'
        '<h1>Dynamic Thinker · COSYNE panels by theme and game</h1><p>10M · temperature 0.01 · VOC–GRU connected · 6 participants. '
        'Each human session uses its held-out checkpoint. Blank regions indicate insufficient support. Neural is excluded.</p><p><a href="../../02/heatmaps/index.html">Individual and group M–U heatmaps</a> (COSYNE probability surfaces use ≥4/6 local subject support).</p>'
        +''.join(cards), encoding='utf-8')
    return report
