"""Summarize paired readout sensitivity without changing planning trajectories.

Each count has its own survivor population. Changes are paired with the first
optional query of those same decisions, never with the overall first-query
mean. These are input-sensitivity measurements, not causal utility estimates.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np


def identity(row):
    return tuple(row[key] for key in ('seed', 'stream', 'decision', 'count'))


def summarize(rows, observed):
    if not rows or not observed:
        raise ValueError('empty query data')
    lookup = {identity(row): row for row in observed}
    if len(lookup) != len(observed) or len({identity(r) for r in rows}) != len(rows):
        raise ValueError('duplicate query identity')
    if set(lookup) != {identity(r) for r in rows}:
        raise ValueError('query coverage differs')
    first = {identity(row)[:3]: row for row in rows if row['count'] == 1}
    by_count = defaultdict(list)
    variants = [key for key in ('frozen_hidden', 'repeated_token', 'frozen_other',
                               'freeze_progress', 'freeze_visits', 'freeze_values')
                if key in rows[0]]
    raw = 'actual_log_odds' in rows[0]
    max_residual = 0.
    for row in rows:
        ref = lookup[identity(row)]
        if row['actual'] != ref['probability']:
            raise ValueError('production query probability changed')
        anchor = first.get(identity(row)[:3])
        if anchor is None or row['first_probability'] != anchor['actual']:
            raise ValueError('missing or inconsistent first optional query')
        if raw:
            if row['first_log_odds'] != anchor['actual_log_odds']:
                raise ValueError('first-query log-odds differ')
            for name in ('actual', *variants):
                log_odds = row[name+'_log_odds']
                if not np.isfinite(log_odds):
                    raise ValueError('nonfinite gate log-odds')
                # Stable sigmoid, including the saturated range.
                p = np.exp(-np.logaddexp(0., -log_odds))
                if not np.isclose(p, row[name], rtol=1e-5, atol=3e-7):
                    raise ValueError('gate probability/log-odds mismatch')
            if 'frozen_other' in variants:
                reconstructed = (row['frozen_hidden_log_odds']
                                 + row['frozen_other_log_odds']-row['first_log_odds'])
                residual = abs(reconstructed-row['actual_log_odds'])
                max_residual = max(max_residual, residual)
                if not np.isclose(reconstructed, row['actual_log_odds'],
                                  rtol=2e-5, atol=2e-5):
                    raise ValueError('linear feature decomposition failed')
        by_count[row['count']].append(row)

    per_count, per_stream = [], []
    for count, group in sorted(by_count.items()):
        item = dict(count=count, at_risk_decisions=len(group),
                    surviving_streams=len({(r['seed'], r['stream']) for r in group}),
                    first_probability_same_survivors=float(np.mean([r['first_probability'] for r in group])))
        for name in ('actual', *variants):
            item[name+'_mean_probability'] = float(np.mean([r[name] for r in group]))
            item[name+'_paired_probability_change'] = float(np.mean([
                r[name]-r['first_probability'] for r in group]))
            item[name+'_fraction_above_099'] = float(np.mean([r[name] > .99 for r in group]))
            if raw:
                item[name+'_mean_log_odds'] = float(np.mean([r[name+'_log_odds'] for r in group]))
                item[name+'_paired_log_odds_change'] = float(np.mean([
                    r[name+'_log_odds']-r['first_log_odds'] for r in group]))
        if raw and 'frozen_other' in variants:
            # Holding other inputs fixed isolates the additive tree term.
            item['tree_log_odds_contribution'] = item['frozen_other_paired_log_odds_change']
            item['other_log_odds_contribution'] = item['frozen_hidden_paired_log_odds_change']
        per_count.append(item)
        for seed, stream in sorted({(r['seed'], r['stream']) for r in group}):
            selected = [r for r in group if (r['seed'], r['stream']) == (seed, stream)]
            record = dict(seed=seed, stream=stream, count=count, at_risk_decisions=len(selected))
            for name in ('actual', *variants):
                key = name+'_log_odds' if raw else name
                base = 'first_log_odds' if raw else 'first_probability'
                record[name+'_paired_change'] = float(np.mean([r[key]-r[base] for r in selected]))
            per_stream.append(record)

    # One first optional query per real decision; long searches cannot receive
    # more weight in this comparison of the online critic and executed gate.
    initial = [lookup[identity(row)] for row in first.values()]
    online = np.array([r['online_delta'] > 0 for r in initial])
    gate = np.array([r['probability'] > .5 for r in initial])
    observed_agreement = float(np.mean(online == gate))
    chance = float(online.mean()*gate.mean()+(1-online.mean())*(1-gate.mean()))
    sign = dict(decisions=len(initial), online_positive_fraction=float(online.mean()),
                gate_positive_fraction=float(gate.mean()), agreement=observed_agreement,
                kappa=(observed_agreement-chance)/(1-chance) if chance < 1 else None,
                balanced_accuracy=(float(np.mean([np.mean(gate[online == v] == v) for v in (False, True)]))
                                   if len(set(online.tolist())) == 2 else None),
                online_negative_given_gate_positive=(float(np.mean(~online[gate])) if gate.any() else None),
                online_exact_zero_fraction=float(np.mean([r['online_delta'] == 0 for r in initial])),
                gate_exact_half_fraction=float(np.mean([r['probability'] == .5 for r in initial])))
    return dict(decisions=len(first), optional_queries=len(rows),
                independent_streams=len({(r['seed'], r['stream']) for r in rows}),
                raw_log_odds_available=raw, maximum_decomposition_residual=max_residual if raw else None,
                first_query_online_gate_sign=sign, by_count=per_count,
                per_stream_paired_change_units='log-odds' if raw else 'probability'), per_stream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recurrence', type=Path, required=True)
    parser.add_argument('--observed', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--conditions', nargs='+', default=['control', 'gru10', 'burn8'])
    args = parser.parse_args()
    sources, reports, stream_rows = [], [], []

    def read(path):
        raw = path.read_bytes()
        sources.append(dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest()))
        return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)

    for condition in args.conditions:
        path = args.recurrence/(condition+'_summary.json')
        if not path.exists():
            reports.append(dict(condition=condition, status='PENDING'))
            continue
        summary = read(path)
        if summary['status'] != 'COMPLETE' or summary['reference_overlap']['matched_rows'] != summary['decisions']:
            raise ValueError('probe has not completed exact trajectory validation')
        observed_summary = read(args.observed/(condition+'_summary.json'))
        if summary['checkpoint'] != observed_summary['checkpoint']:
            raise ValueError('recurrence and observed checkpoints differ')
        report, per_stream = summarize(read(args.recurrence/(condition+'_queries.json.gz')),
                                       read(args.observed/(condition+'_queries.json.gz')))
        if report['decisions'] != summary['decisions'] or report['optional_queries'] != summary['queries']:
            raise ValueError('summary/data coverage mismatch')
        reports.append(dict(condition=condition, status='COMPLETE', checkpoint=summary['checkpoint'], **report))
        stream_rows.extend(dict(condition=condition, **row) for row in per_stream)
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(checked_at=time.time(), reports=reports, sources=sources,
        limitations=[
            'Each count conditions on decisions that reached it. Paired changes use those same decisions at count1.',
            'Frozen features are input sensitivity on unchanged trajectories, not valid counterfactual search policies or utility.',
            'One trained seed and one evaluation seed; decisions within eight streams are not independent training replications.',
            'Online/gate sign disagreement does not establish which critic is accurate or an execution defect.',
            'Different trained conditions visit different states; equal evaluation seeds do not create identical between-condition states.'])
    (args.output/'evidence.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    for name, table in [('per_count.csv', [dict(condition=r['condition'], **row) for r in reports for row in r.get('by_count', [])]),
                        ('per_stream.csv', stream_rows)]:
        if table:
            with (args.output/name).open('w') as out:
                keys = list(dict.fromkeys(key for row in table for key in row))
                writer = csv.DictWriter(out, fieldnames=keys)
                writer.writeheader()
                writer.writerows(table)
    shutil.copy2(__file__, args.output/Path(__file__).name)
    for report in reports:
        print(report['condition'], report['status'], report.get('first_query_online_gate_sign'))


if __name__ == '__main__':
    main()
