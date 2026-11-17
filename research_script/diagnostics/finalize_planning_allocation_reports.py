"""Finalize registered allocation reports after all three GPU samples finish.

This job reads saved results on CPU; it never trains or evaluates an agent.
The existing registered comparison command remains authoritative.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def readiness(campaign, automation):
    plan = json.loads((automation / 'plan.json').read_text())
    for name, expected in plan['frozen_sha256'].items():
        if sha(automation / name) != expected:
            raise ValueError('frozen reporting source changed: ' + name)
    report_plan = Path(plan['registered_plan_path'])
    if sha(report_plan) != plan['registered_plan_sha256']:
        raise ValueError('registered comparison plan changed')
    registered = json.loads(report_plan.read_text())
    if sha(Path(registered['command'][1])) != registered['script_sha256']:
        raise ValueError('registered comparison script changed')
    spec = importlib.util.spec_from_file_location('registered_comparison', registered['command'][1])
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    ready, missing = {}, []
    for condition in ('control', 'burn8', 'gru10'):
        folder = campaign / ('allocation_1m_' + condition)
        if not (folder / 'final_binding_audit.json').exists():
            missing.append(condition)
            continue
        sample = module.load_sample([folder], condition, registered=True, final=True)
        ready[condition] = dict(decisions=sample['decisions'], checkpoint=sample['checkpoint'])
    return plan, registered, dict(ready=ready, missing=missing,
        status='READY' if not missing else 'WAITING_FOR_COMPLETE_REGISTERED_SAMPLES')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--automation', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    campaign, automation = args.campaign.resolve(), args.automation.resolve()
    plan, registered, state = readiness(campaign, automation)
    if args.check_only:
        print(json.dumps(state, indent=2))
        return
    if state['missing']:
        raise ValueError('refusing partial final comparison: ' + ','.join(state['missing']))
    completion_path = automation / 'completion.json'
    if completion_path.exists():
        raise ValueError('report already complete; do not overwrite')
    outputs = [Path(registered['output']), Path(plan['evidence_output']), Path(plan['null_output'])]
    if any(p.exists() for p in outputs):
        raise ValueError('report output exists; inspect before attempting a replacement')
    commands = [[sys.executable] + registered['command'][1:]]
    evidence = [sys.executable, str(automation / 'source/summarize_adaptive_planning_evidence.py'),
                '--campaign', str(campaign), '--output', str(outputs[1])]
    for condition in ('control', 'burn8', 'gru10'):
        evidence += ['--allocation-root', str(campaign / ('allocation_1m_' + condition))]
    commands.append(evidence)
    commands.append([sys.executable, str(automation / 'source/plot_planning_initial_gate_null.py'),
                     '--campaign', str(campaign), '--output', str(outputs[2])])
    execution = []
    for index, command in enumerate(commands):
        start = time.time()
        with (automation / ('report_' + str(index) + '.out')).open('x') as stdout, (
                automation / ('report_' + str(index) + '.err')).open('x') as stderr:
            process = subprocess.run(command, stdout=stdout, stderr=stderr)
        execution.append(dict(command=command, returncode=process.returncode,
                              elapsed_seconds=time.time() - start))
        if process.returncode:
            raise RuntimeError('reporting command failed: ' + str(index))
    comparison = json.loads((outputs[0] / 'audit.json').read_text())
    if comparison['declared_milestones'] != [250000, 500000, 1000000]:
        raise ValueError('final milestone coverage differs')
    with (outputs[0] / 'allocation_milestones.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 9 or {(r['condition'], int(r['milestone'])) for r in rows} != {
            (c, m) for c in ('control', 'burn8', 'gru10') for m in (250000, 500000, 1000000)}:
        raise ValueError('final comparison is incomplete')
    evidence_result = json.loads((outputs[1] / 'evidence.json').read_text())
    if len(evidence_result['reports']) != 3 or any(
            r['allocation_status'] != 'COMPLETE' for r in evidence_result['reports']):
        raise ValueError('allocation evidence is incomplete')
    null_result = json.loads((outputs[2] / 'audit.json').read_text())
    if null_result['missing_conditions'] or null_result['status'] != 'COMPLETE_ALLOCATION_DIAGNOSTIC':
        raise ValueError('initial-gate comparison is incomplete')
    completion = dict(status='COMPLETE_ALLOCATION_REPORTS_UTILITY_SEPARATE',
        completed_at=time.time(), input_readiness=state, execution=execution,
        outputs=[str(p) for p in outputs], analysis_only=True, gpu_requested=False,
        utility_status={r['condition']: r['utility_status'] for r in evidence_result['reports']},
        goal_complete=False,
        scope='Descriptive allocation and within-planning diagnostics. Useful adaptive planning remains a separate requirement.')
    completion_path.write_text(json.dumps(completion, indent=2) + '\n')
    print(json.dumps(completion, indent=2))


if __name__ == '__main__':
    main()
