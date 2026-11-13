"""Supplement validated theme results with the missing COSYNE panel analyses."""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from common.features import load_unit
from common.paths import output, tag
from common.io import sha, signature, write_json
from common.reporting import save_tables
from common import matching
from . import data

NUMBERS = ['01', '02', '03', '04', '05', '09', '12']


def run(cache, outputs, subject, game):
    source = Path(__file__).parent
    identity = dict(subject=subject, game=game, cache=str(Path(cache).resolve()),
        cache_sha256=sha(Path(cache)/'manifest.json'),
        source={**{name: sha(source/name) for name in ['data.py','estimators.py','runner.py']},
                'common/matching.py':sha(matching.__file__)},
        heldout='Each session uses only its leave-session checkpoint',
        derived={f'leave{cp:02d}': sha(output(0, outputs)/'derived'/tag(subject, game)/f'leave{cp:02d}'/'manifest.json') for cp in range(1, 5)},
        primary_result_manifests={n: sha(output(n, outputs)/tag(subject, game)/'manifest.json') for n in NUMBERS if n != '01'})
    token = signature(identity)
    # The initial runner generated replay-grid behavior before the separate
    # original-human stage was restored. Its other six themes used exactly the
    # same data/estimator functions and inputs. Accept only that known runner's
    # token, with every other identity field still required to match.
    previous_identity = dict(identity, source=dict(identity['source']))
    previous_identity['source']['runner.py'] = '3bb945d45c08f2546de786b7993061337594051f1641798f871d122bf24762a5'
    accepted_tokens = {token, signature(previous_identity)}
    for n in NUMBERS:
        if n == '01': continue
        primary = json.loads((output(n, outputs)/tag(subject, game)/'manifest.json').read_text())
        if primary.get('status') != 'COMPLETED':
            raise ValueError(f'Primary theme {n} not complete for {tag(subject,game)}')
    roots = {n: output(n, outputs)/'panels'/tag(subject, game) for n in NUMBERS}
    def completed(p):
        f = p/'manifest.json'
        if not f.exists(): return False
        m = json.loads(f.read_text())
        return m.get('status') == 'COMPLETED' and m.get('signature') in accepted_tokens and all(
            (p/name).exists() and sha(p/name) == digest for name, digest in m.get('artifacts', {}).items())
    raw_manifest = roots['01']/'manifest.json'
    if raw_manifest.exists() and all(completed(p) for n,p in roots.items() if n != '01'):
        raw = json.loads(raw_manifest.read_text())
        if raw.get('scope','').startswith('ALL original human'):
            from .raw_behavior import run as raw_run
            raw_run(cache,outputs,subject,game)
            return {'status':'REUSED','subject':subject,'game':game,'human_behavior':'original observations'}
    if all(completed(p) for p in roots.values()): return {'status': 'REUSED', 'subject': subject, 'game': game}
    for n, p in roots.items():
        p.mkdir(parents=True, exist_ok=True)
        if (p/'manifest.json').exists() and not completed(p):
            old = json.loads((p/'manifest.json').read_text())
            if old.get('status') == 'COMPLETED':
                raise ValueError('Preserve previous panel output before changing the analysis: '+str(p))
        write_json(p/'manifest.json', dict(identity, signature=token, theme=n, status='RUNNING', started_at=time.time()))
    d, e = load_unit(outputs, subject, game)
    d, timing_qa = data.timing(d, cache, subject, game); d = data.scaled(d)
    def save(n, tables, **qa):
        for df in tables.values():
            df['subject'] = subject; df['game'] = game
        save_tables(roots[n], tables)
        write_json(roots[n]/'qa.json', dict(decisions=len(d), runs=int(d.run_label.nunique()),
            zero_budget=int(d.additional_computations.eq(0).sum()), **qa))
        artifacts = {str(p.relative_to(roots[n])): sha(p) for p in (roots[n]/'tables').glob('*')}
        write_json(roots[n]/'manifest.json', dict(identity, signature=token, theme=n, status='COMPLETED',
            completed_at=time.time(), artifacts=artifacts, interpretation='Associational dynamic-model panels; no neural or environment score evaluation.'))
        print(json.dumps(dict(event='panel_theme_complete', theme=n, subject=subject, game=game)), flush=True)
    behavioral = data.behavior_panels(d); behavioral['source_alignment'] = timing_qa
    if not behavioral['duration_fit_qa'].status.eq('ok').all(): raise ValueError('Duration prediction fit failed')
    save('01', behavioral)
    pairs, paired = data.matched_pairs(d)
    k, kqa = data.k_curves(d)
    save('02', dict(matched_pairs=pairs, matched_means=paired, mu_surface=data.mu_surface(d), k_curves=k, k_curve_qa=kqa),
         k_fit_status=kqa.status.value_counts().to_dict())
    gates, budget, states = data.allocation_curves(d, e)
    save('03', dict(gate_calibration=gates, budget_distribution=budget))
    ep, events, eventqa = data.native_search(d, cache, subject, game)
    composition, coverage = data.update_composition(d, events, pairs)
    sensitivity, sensitivity_qa, cv = data.endpoint_sensitivity(ep)
    endpointcols = [*data.KEYS, 'root_choices', 'M', 'U_all', 'endpoint_change',
                    'first_preference', 'last_preference', 'fixed_alternative']
    save('04', dict(update_composition=composition, update_coverage=coverage, matched_means=paired,
        endpoint_sensitivity=sensitivity, endpoint_sensitivity_qa=sensitivity_qa, endpoint_crossvalidation=cv,
        search_endpoint_records=ep.reindex(columns=endpointcols), native_search_qa=eventqa),
        eligible_same_head_endpoints=len(ep), sensitivity_status=sensitivity_qa.status.value_counts().to_dict())
    save('05', {})  # Renderer consumes verified existing held-out predictions, no refit required.
    save('09', dict(state_allocation=states))
    save('12', dict(event_locked=data.event_locked(d)))
    return {'status': 'COMPLETED', 'subject': subject, 'game': game}
