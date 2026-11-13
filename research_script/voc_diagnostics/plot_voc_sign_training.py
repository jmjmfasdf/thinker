"""Plot recorded continuation probabilities by EMA VOC sign over training."""
import argparse
import csv
import datetime
import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np

GAMES = [('pong', 'Pong', '#0072B2'), ('enduro', 'Enduro', '#D55E00'),
         ('spaceinvaders', 'Space Invaders', '#009E73')]


def save_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--window', type=int, default=500_000)
    parser.add_argument('--stride', type=int, default=100_000)
    args = parser.parse_args()
    if args.window <= 0 or args.stride <= 0:
        raise ValueError('Window and stride must be positive')
    campaign = args.campaign.resolve(); output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((campaign/'campaign.json').read_text())
    temperature = float(manifest['temperature'])
    cells = {c['game']: c for c in manifest['cells']}
    samples = []; curves = []; sources = []; totals = {}
    for game, _, _ in GAMES:
        source = campaign/'cells'/cells[game]['name']/'control_samples.jsonl'
        raw = source.read_bytes()
        # A running writer can leave one incomplete final line; never infer it.
        complete = raw[:raw.rfind(b'\n')+1]
        records = [json.loads(line) for line in complete.splitlines() if line]
        identity = [(r['attempt'], r['real_step'], r['stream'], r['count']) for r in records]
        if len(set(identity)) != len(identity):
            raise ValueError('Duplicate control sample identity')
        end = max(r['real_step'] for r in records)
        valid = [r for r in records if r['valid'] and not r['forced']]
        t = np.array([r['real_step'] for r in valid])
        q = np.array([r['ema_delta_q'] for r in valid])
        p = np.array([r['p_continue'] for r in valid])
        if not np.isfinite(q).all() or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
            raise ValueError('Invalid VOC or probability')
        for r in valid:
            samples.append(dict(game=game, **r))
        targets = list(range(args.window, end+1, args.stride))
        if not targets:
            continue
        if end != targets[-1]:
            if end-targets[-1] < args.stride/2:
                targets[-1] = end
            else:
                targets.append(end)
        for target in targets:
            window = (t >= target-args.window) & (t <= target)
            selected = [r for r, yes in zip(valid, window) if yes]
            row = dict(game=game, temperature=temperature, window_start=target-args.window,
                training_step=target, control_samples=int(window.sum()),
                sampled_decisions=len({(r['attempt'],r['decision'],r['stream']) for r in selected}))
            for label, sign in [('positive', 1), ('negative', -1)]:
                mask = window & (sign*q > 1e-6)
                rr = [r for r, yes in zip(valid, mask) if yes]
                row[label+'_n'] = int(mask.sum())
                row[label+'_decisions'] = len({(r['attempt'],r['decision'],r['stream']) for r in rr})
                row[label+'_probability'] = float(p[mask].mean()) if mask.any() else None
            curves.append(row)
        totals[game] = dict(last_step=end, valid_samples=len(valid),
            plotted_windows=len(targets), first_plotted_step=targets[0])
        sources.append(dict(path=str(source), snapshot_sha256=hashlib.sha256(complete).hexdigest(),
            snapshot_bytes=len(complete), incomplete_final_bytes=len(raw)-len(complete)))

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(11.6, 6.5))
    fig.subplots_adjust(left=.085, right=.98, top=.78, bottom=.2)
    fig.text(.085, .96, f'VOC-dependent continuation during training | temperature {temperature:g}',
             fontsize=17, fontweight='bold', va='top')
    fig.text(.085, .90, 'Recorded execution probabilities, averaged over the preceding 0.5M environment steps', fontsize=11)
    for game, label, color in GAMES:
        data = [r for r in curves if r['game']==game]
        x = np.array([r['training_step']/1e6 for r in data])
        for key, style in [('positive', '-'), ('negative', '--')]:
            y = np.array([r[key+'_probability'] if r[key+'_probability'] is not None else np.nan for r in data])
            ax.plot(x, y, color=color, linestyle=style, linewidth=2.3)
            ax.plot(x[-1], y[-1], 'o', color=color, markersize=5)
    game_handles = [Line2D([0], [0], color=color, lw=2.5, label=label) for _,label,color in GAMES]
    sign_handles = [Line2D([0], [0], color='#333333', lw=2.3, ls=style, label=label)
        for style,label in [('-', 'EMA VOC > 0'), ('--', 'EMA VOC < 0')]]
    fig.legend(handles=game_handles, loc='upper left', bbox_to_anchor=(.075,.86),
               ncol=3, frameon=False, columnspacing=1.5)
    fig.legend(handles=sign_handles, loc='upper right', bbox_to_anchor=(.985,.86),
               ncol=2, frameon=False, columnspacing=1.5)
    ax.axhline(.5, color='#858585', linestyle=':', linewidth=1.1, zorder=0)
    ax.set(xlim=(0,10.15), ylim=(-.02,1.02), xlabel='Training environment steps (millions)',
           ylabel='Mean probability of continuing computation')
    ax.set_xticks(np.arange(0,11,1));ax.set_yticks(np.linspace(0,1,6))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis='y', alpha=.18)
    fig.text(.085, .105, 'All valid search choices, including decisions made after computation begins; forced stops and ties excluded.', fontsize=10)
    fig.text(.085, .067, 'Stream 0 sampled every 128 decisions. Curves end at available training data; no extrapolation.', fontsize=10)
    fig.text(.085, .029, 'VOC sign is the model’s own estimate. Separation of these curves does not establish realized reward benefit.', fontsize=10)
    stem='voc_sign_continuation_training_tau'+str(temperature).replace('.','')
    for suffix in ['png','pdf','svg']:
        fig.savefig(output/(stem+'.'+suffix), dpi=210, facecolor='white')
    plt.close(fig)
    save_csv(output/(stem+'.csv'), curves)
    save_csv(output/(stem+'_samples.csv'), samples)
    qa = dict(created_at=datetime.datetime.now().astimezone().isoformat(), temperature=temperature,
        window_steps=args.window, stride_steps=args.stride, totals=totals, sources=sources,
        weighting='Equal weight per valid control choice; repeats within decisions and overlapping windows are dependent.',
        scope='Descriptive training-time gate probabilities conditioned on the sign of same-time EMA VOC, from sampled stream 0. Neither held-out utility nor game-score evaluation.',
        exclusions='Forced or invalid controls, and abs(EMA VOC) <= 1e-6. Missing sign groups remain missing.',
        initial_window='First point is at 0.5M; no shortened initial windows. Final window ends at latest complete log entry.')
    (output/(stem+'_manifest.json')).write_text(json.dumps(qa,indent=2)+'\n')
    print(json.dumps(dict(figure=str(output/(stem+'.png')), totals=totals,
        minimum_sign_counts={game:min(min(r['positive_n'],r['negative_n']) for r in curves if r['game']==game) for game,_,_ in GAMES})))


if __name__=='__main__':main()
