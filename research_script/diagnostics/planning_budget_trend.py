"""Within-run budget trends with nonoverlapping bins and HAC uncertainty."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from compare_planning_budget import HIST, STEP


def interval_table(frame, width, end, start=0):
    nominal = list(range(start, end+1, width))
    if nominal[-1] != end:
        nominal.append(end)
    edges = [0 if x == 0 else int(frame.index[frame.index <= x].max()) for x in nominal]
    k = np.arange(31)
    rows = []
    for a, b, lo, hi in zip(nominal[:-1], nominal[1:], edges[:-1], edges[1:]):
        h = frame.loc[hi, HIST].to_numpy(float)
        if lo:
            h = h-frame.loc[lo, HIST].to_numpy(float)
        n = h.sum()
        assert n == hi-lo and n > 0 and np.all(h >= 0)
        quantile = lambda q: int(np.searchsorted(np.cumsum(h), q*n))
        rows.append(dict(nominal_start=a, nominal_end=b, start_env_step=lo,
            end_env_step=hi, midpoint_m=(lo+hi)/2e6, decisions=int(n),
            mean_budget=float(h @ k/n), mean_queries_beyond_first=float(h @ np.maximum(k-1,0)/n),
            median_budget=quantile(.5), p90_budget=quantile(.9),
            one_fraction=float(h[1]/n), two_plus_fraction=float(h[2:].sum()/n),
            cap_fraction=float(h[30]/n),
            mean_below_cap=float(h[:30] @ k[:30]/h[:30].sum()),
            mean_given_two_plus=float(h[2:] @ k[2:]/h[2:].sum()),
            cap_share_of_queries=float(30*h[30]/(h @ k))))
    return pd.DataFrame(rows)


def regress(table, metric, width):
    # Half a million environment actions of lag coverage, not millions of
    # individual decisions treated as independent observations.
    lag = max(1, round(500_000/width))
    model = sm.WLS(table[metric], sm.add_constant(table.midpoint_m),
                   weights=table.decisions).fit(cov_type='HAC',
                   cov_kwds={'maxlags':lag, 'use_correction':True}, use_t=True)
    ci = model.conf_int().iloc[1]
    return dict(metric=metric, bin_width=width, bins=len(table), hac_maxlags=lag,
        slope_per_million=float(model.params.iloc[1]), ci_low=float(ci.iloc[0]),
        ci_high=float(ci.iloc[1]), p_value=float(model.pvalues.iloc[1]),
        r_squared=float(model.rsquared), intercept=float(model.params.iloc[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--counters', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    f = pd.read_csv(args.counters).set_index(STEP).sort_index()
    h = f[HIST].to_numpy()
    assert np.array_equal(h.sum(1), f.index.to_numpy())
    assert np.array_equal(h @ np.arange(31), f['collector/imagined_transitions'])
    assert np.all(np.diff(h,axis=0) >= 0)
    args.output.mkdir(parents=True, exist_ok=True)
    million = interval_table(f, 1_000_000, 6_000_000)
    million.to_csv(args.output/'million_step_intervals.csv', index=False)
    primary = interval_table(f, 100_000, 6_000_000)
    primary.to_csv(args.output/'primary_100k_intervals.csv', index=False)
    interval_table(f, 500_000, 6_500_000).to_csv(args.output/'half_million_intervals.csv', index=False)
    metrics = ['mean_budget','mean_queries_beyond_first','two_plus_fraction',
               'cap_fraction','mean_below_cap','mean_given_two_plus']
    fits = []
    for width in [50_000,100_000,250_000,500_000]:
        table = interval_table(f,width,6_000_000)
        for metric in metrics:
            fits.append(dict(analysis='bin_width_sensitivity',start=0,end=6_000_000,
                             **regress(table,metric,width)))
    for start,end in [(500_000,6_000_000),(0,6_500_000),(500_000,6_500_000)]:
        table = interval_table(f,100_000,end,start)
        fits.append(dict(analysis='window_sensitivity',start=start,end=end,
                         **regress(table,'mean_budget',100_000)))
    fits = pd.DataFrame(fits)
    fits.to_csv(args.output/'trend_regressions.csv', index=False)

    plt.rcParams.update({'font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42})
    fig, axes = plt.subplots(1,3,figsize=(12,3.8))
    for ax, metric, title, scale in zip(axes,
        ['mean_budget','two_plus_fraction','cap_fraction'],
        ['Mean imagination count','Decisions with 2+ queries','Decisions reaching 30 queries'],
        [1,100,100]):
        ax.plot(primary.midpoint_m, primary[metric]*scale, color='#999999', alpha=.5,
                linewidth=.7, marker='.', markersize=2, label='100k intervals')
        ax.plot(million.midpoint_m, million[metric]*scale, color='#186CA0',
                linewidth=1.8, marker='o', markersize=5, label='1M intervals')
        fit = regress(primary,metric,100_000)
        x = np.array([0.,6.])
        ax.plot(x,(fit['intercept']+fit['slope_per_million']*x)*scale,
                color='#D17C20',linestyle='--',linewidth=1.8,label='Linear trend')
        ax.set(title=title,xlabel='Training environment actions (million)',xlim=(0,6),
               ylabel='Queries / real decision' if scale==1 else 'Decisions (%)')
        ax.grid(axis='y',alpha=.18)
    axes[0].legend(loc='upper left',frameon=False,fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output/'planning_trend.png',dpi=180)
    fig.savefig(args.output/'planning_trend.pdf')
    plt.close(fig)
    (args.output/'manifest.json').write_text(json.dumps(dict(
        input=str(args.counters.resolve()),primary_window=[0,6_000_000],
        primary_actual_end=int(primary.end_env_step.max()),primary_bin_steps=100_000,
        method='Nonoverlapping histogram differences; decision-count-weighted linear regression; HAC lag coverage 0.5M with small-sample correction.',
        unit='Total imagined transitions per real decision, not branch depth.',
        first_query='Required in every decision, so mean beyond first = mean budget - 1.',
        limitations=['One training run and seed; inference describes within-run temporal variation.',
                     'Secondary metrics and window/bin sensitivity are exploratory and uncorrected.',
                     'On-policy state distribution changes with training; no same-state causal claim.'],
        qa=dict(physical_count_and_query_histogram_agree=True,monotone_counters=True,
                no_overlapping_primary_bins=True)),indent=2))
    print(million.to_string(index=False))
    print(fits.query('metric == "mean_budget"').to_string(index=False))


if __name__ == '__main__':
    main()
