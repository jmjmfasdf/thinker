"""Unbinned fractional-logit regression of gate probability on measured utility."""
import argparse
import csv
import datetime
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit, logit

GAMES = [('pong', 'Pong', '#0072B2'), ('enduro', 'Enduro', '#D55E00'),
         ('spaceinvaders', 'Space Invaders', '#009E73')]


def fractional_logit(x, y):
    """Quasi-binomial fit; each distinct state has weight one, not its repetitions."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Invalid data')
    if np.any((y < 0) | (y > 1)):
        raise ValueError('Invalid probability')
    center, scale = x.mean(), x.std()
    if scale < 1e-12:
        return np.array([logit(np.clip(y.mean(), 1e-9, 1-1e-9)), 0.]), False
    if y.std() < 1e-7:
        return np.array([logit(np.clip(y.mean(), 1e-9, 1-1e-9)), 0.]), True
    design = np.c_[np.ones(len(x)), (x-center)/scale]
    b = np.array([logit(np.clip(y.mean(), 1e-9, 1-1e-9)), 0.])
    for _ in range(80):
        z = design @ b
        p = expit(z)
        gradient = design.T @ (p-y)
        if np.linalg.norm(gradient, ord=np.inf) < 1e-9:
            break
        hessian = design.T @ ((p*(1-p))[:, None] * design)
        direction = np.linalg.solve(hessian + 1e-10*np.eye(2), gradient)
        old = np.sum(np.logaddexp(0, z)-y*z)
        rate = 1.
        for _ in range(30):
            candidate = b-rate*direction
            zz = design @ candidate
            if np.sum(np.logaddexp(0, zz)-y*zz) <= old + 1e-12:
                b = candidate
                break
            rate *= .5
        else:
            break
    return np.array([b[0]-b[1]*center/scale, b[1]/scale]), True


def fit_states(states, bootstrap=1000):
    x = np.array([s['target'] for s in states])
    y = np.array([s['probability'] for s in states])
    groups = [tuple(s['key'][:2]) for s in states]
    unique = sorted(set(groups))
    indices = [np.array([i for i, g in enumerate(groups) if g == u]) for u in unique]
    b, identified = fractional_logit(x, y)
    rng = np.random.default_rng(20260928)
    draws = []
    for _ in range(bootstrap):
        ids = np.concatenate([indices[i] for i in rng.integers(len(unique), size=len(unique))])
        bb, ok = fractional_logit(x[ids], y[ids])
        if ok:
            draws.append(bb)
    draws = np.asarray(draws)
    ci = np.quantile(draws[:, 1], [.025, .975]).tolist() if len(draws) else [None, None]
    return dict(intercept=float(b[0]), slope=float(b[1]), slope_ci95=ci,
        identified=identified, n_states=len(states), n_streams=len(unique),
        n_positive=int(np.sum(x>0)), n_task_changed=sum(abs(s['target']+s['cost'])>1e-12 for s in states),
        gate_sd=float(y.std()), utility_sd=float(x.std()), valid_bootstraps=len(draws)), b, draws


def csv_write(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def campaign_results(campaign):
    registry = json.loads((campaign/'jobs.json').read_text())['evaluation']
    result = {}
    for tf in sorted((campaign/'eval_tasks').glob('*_benefit.json')):
        task = json.loads(tf.read_text())
        path = Path(registry.get(tf.stem, {}).get('output') or task['output'])/'summary.json'
        if path.exists():
            s = json.loads(path.read_text())
            if s.get('status') == 'COMPLETE':
                result[task['game'], task['real_step']] = s
    return result


def draw(results, temperature, output):
    games = [item for item in GAMES if any(key[0] == item[0] for key in results)]
    common = set.intersection(*[{step for g, step in results if g == game} for game, *_ in games])
    step = max(common)
    fig, axes = plt.subplots(len(games), 2, figsize=(11.5, 3.3*len(games)+1.9), squeeze=False)
    fig.subplots_adjust(left=.09, right=.97, top=.87, bottom=.15, hspace=.52, wspace=.24)
    fig.suptitle(f'Continuous utility–continuation relationship | temperature {temperature:g}',
                 fontsize=16, fontweight='bold', x=.09, ha='left', y=.98)
    fig.text(.09, .94, f'{step/1e6:g}M checkpoint | fractional-logit regression; no utility bins', fontsize=11)
    rows = []
    state_rows = []
    for row, (game, label, color) in enumerate(games):
        for col, count in enumerate([0, 3]):
            ax = axes[row, col]
            states = [s for s in results[game, step]['voc']['state_records'] if s['key'][3] == count]
            fit, b, draws = fit_states(states)
            rows.append(dict(temperature=temperature, game=game, checkpoint=step, count=count, **fit))
            x = np.array([s['target'] for s in states]); y = np.array([s['probability'] for s in states])
            grid = np.linspace(x.min(), x.max(), 240)
            predictions = expit(draws[:, 0, None] + draws[:, 1, None]*grid) if len(draws) else None
            if predictions is not None:
                lo, hi = np.quantile(predictions, [.025, .975], axis=0)
                ax.fill_between(grid, lo, hi, color=color, alpha=.15, linewidth=0)
            ax.plot(grid, expit(b[0]+b[1]*grid), color=color, lw=2, label='Fractional-logit fit')
            ax.scatter(x, y, color=color, s=27, alpha=.75, edgecolors='white', linewidths=.4,
                       label='State (repetitions averaged)', zorder=3)
            ax.set_ylim(-.035, 1.05)
            ax.set_title(f'{label} | ' + ('Before computation' if count == 0 else 'After 3 computations'), fontsize=11)
            ax.set_xlabel('Realized net benefit (continuous)')
            if col == 0: ax.set_ylabel('Continue probability')
            ax.ticklabel_format(axis='x', style='sci', scilimits=(-2, 3), useOffset=False)
            ax.grid(axis='y', alpha=.2)
            ax.spines[['top','right']].set_visible(False)
            ax.text(.03, .95, f"{len(states)} states / {fit['n_streams']} streams; U > 0: {fit['n_positive']}\nTask-return changes: {fit['n_task_changed']}",
                    transform=ax.transAxes, va='top', fontsize=9)
            for s in states:
                state_rows.append(dict(temperature=temperature,game=game,checkpoint=step,count=count,
                    seed=s['key'][0],stream=s['key'][1],prefix=s['key'][2],
                    utility=s['target'],continue_probability=s['probability'],cost=s['cost']))
    fig.text(.09, .087, 'Dots: distinct states. Shading: 95% bootstrap CI, resampling whole environment streams.', fontsize=10)
    fig.text(.09, .051, 'Axes use observed utility ranges separately. Sparse influential states can make the fitted curve unstable.', fontsize=10)
    stem = 'continuous_regression_tau'+str(temperature).replace('.', '')
    for ext in ['png','pdf','svg']: fig.savefig(output/(stem+'.'+ext), dpi=210, facecolor='white')
    plt.close(fig)
    csv_write(output/(stem+'_states.csv'), state_rows)
    csv_write(output/(stem+'_coefficients.csv'), rows)
    (output/(stem+'.json')).write_text(json.dumps(rows, indent=2)+'\n')
    print(json.dumps(dict(figure=str(output/(stem+'.png')), checkpoint=step, fits=rows)), flush=True)


def main():
    p = argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--temperature',type=float,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    draw(campaign_results(a.campaign),a.temperature,a.output)


if __name__ == '__main__': main()
