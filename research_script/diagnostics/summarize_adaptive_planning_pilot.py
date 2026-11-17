"""Read-only scheduler/training/evaluation audit for the staged planning pilot."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import socket
import statistics
import subprocess
import time


def csv_rows(path):
    if not path.exists():
        return []
    # A final incomplete append must not masquerade as a completed update.
    return [row for row in csv.DictReader(path.open()) if None not in row.values() and None not in row]


def json_count(path):
    return len(json.loads(path.read_text())) if path.exists() else 0


def loss_health(rows):
    samples = [(r.get('real_step'), float(r['actor/icopro_nll']))
               for r in rows if r.get('actor/icopro_nll')]
    values = [value for _, value in samples]
    finite = [value for value in values if math.isfinite(value)]
    large = [(step, value) for step, value in samples if math.isfinite(value) and value > 20]
    rolling = [statistics.median(values[i-20:i]) for i in range(20, len(values)+1)
               if all(math.isfinite(value) for value in values[i-20:i])]
    peak_median = max(rolling, default=None)
    streak, longest = 0, 0
    for value in values:
        streak = streak+1 if math.isfinite(value) and value > 20 else 0
        longest = max(longest, streak)
    history = dict(
        bc_history_status=('NO_DATA' if not values else
            'NONFINITE_HISTORY' if len(finite) != len(values) else
            'HISTORICAL_SUSTAINED_LARGE_NLL' if peak_median is not None and peak_median > 20 else
            'HISTORICAL_LARGE_NLL_SPIKE' if large else 'NO_LARGE_NLL_OBSERVED'),
        maximum_bc_nll=max(finite, default=None),
        maximum_20update_median_bc_nll=peak_median,
        large_bc_update_count=len(large), longest_large_bc_streak=longest,
        first_large_bc_step=int(float(large[0][0])) if large and large[0][0] is not None else None,
        last_large_bc_step=int(float(large[-1][0])) if large and large[-1][0] is not None else None,
        nonfinite_bc_update_count=len(values)-len(finite))
    if not values:
        return dict(bc_loss_status='NO_DATA', recent_median_bc_nll=None, **history)
    recent = values[-20:]
    if not all(math.isfinite(v) for v in recent):
        return dict(bc_loss_status='NONFINITE', recent_median_bc_nll=None, **history)
    median = statistics.median(recent)
    return dict(bc_loss_status=('SUSTAINED_LARGE_NLL' if len(recent)==20 and median>20
                               else 'NO_SUSTAINED_LARGE_NLL'), recent_median_bc_nll=median, **history)


def auxiliary_jobs(campaign):
    jobs = []
    persistence = campaign/'persistence_evaluation_jobs.json'
    if persistence.exists():
        for saved in json.loads(persistence.read_text())['jobs']:
            jobs.append(dict(job_id=saved['job_id'],kind='allocation',
                label='persistence_'+str(saved['milestone']),conditions=[saved['condition']],
                output=saved['output'],expected_epsilon=0.,expected_milestone=saved['milestone'],
                expected_allocation_per_condition=24576,expected_pairs=0))
    for stem in ('allocation_100k', 'allocation_250k_long', 'allocation_250k_long_rest',
                 'allocation_250k_burn8', 'allocation_250k_burn16',
                 'allocation_500k_control', 'allocation_500k_burn8', 'allocation_500k_gru10',
                 'allocation_1m_control', 'allocation_1m_burn8', 'allocation_1m_gru10',
                 'allocation_500k_gru10_train_epsilon', 'allocation_500k_gru10_episode_coverage'):
        path = campaign/(stem+'_job.json')
        if not path.exists():
            continue
        saved = json.loads(path.read_text())
        if saved.get('superseded_by'):
            continue
        conditions = saved.get('conditions', list(saved.get('checkpoint_files', {})))
        exploration = stem.endswith('_train_epsilon')
        episode_coverage = stem.endswith('_episode_coverage')
        jobs.append(dict(job_id=saved['job_id'],
            kind=('training_policy_allocation' if exploration else
                  'episode_coverage' if episode_coverage else 'allocation'), label=stem,
            conditions=conditions, output=str(campaign/stem),
            expected_epsilon=saved.get('evaluation_epsilon', 0.),
            expected_allocation_per_condition=saved['decisions_per_condition'], expected_pairs=0))
    for stem in ('query_dynamics_250k', 'query_dynamics_1m'):
        path = campaign/(stem+'_job.json')
        if path.exists():
            saved = json.loads(path.read_text())
            jobs.append(dict(job_id=saved['job_id'], kind='query_dynamics',
                label=stem, conditions=saved['conditions'],
                output=saved.get('output',str(campaign/stem)),
                expected_allocation_per_condition=saved['decisions_per_condition'], expected_pairs=0))
    for stem in ('recurrence_250k', 'recurrence_fields_250k'):
        path = campaign/(stem+'_job.json')
        if path.exists():
            saved = json.loads(path.read_text())
            jobs.append(dict(job_id=saved['job_id'], kind='recurrence',
                label=stem, conditions=saved['conditions'], output=saved['output'],
                expected_allocation_per_condition=saved['decisions_per_condition'], expected_pairs=0))
    path = campaign/'utility_late_prefix896_jobs.json'
    if path.exists():
        saved = json.loads(path.read_text())
        for job in saved['jobs']:
            jobs.append(dict(job_id=job['job_id'], kind='late_paired_utility',
                label='utility_late_prefix896_'+job['condition'], conditions=[job['condition']],
                output=job['output'],
                expected_allocation_per_condition=(saved['allocation_decisions_per_stream']
                    *saved['batch']*len(saved['seeds'])),
                expected_pairs=saved['expected_pairs_per_condition']))
    path = campaign/'utility_continuation_jobs.json'
    if path.exists():
        for job in json.loads(path.read_text())['jobs']:
            if job.get('status') != 'SUBMITTED':
                continue
            recovery = Path(job['recovery'])
            done_path = recovery/'completion.json'
            done = json.loads(done_path.read_text()) if done_path.exists() else {}
            skipped = done.get('status') == 'ORIGINAL_COMPLETED_CONTINUATION_NOT_NEEDED'
            jobs.append(dict(job_id=job['job_id'],kind='late_paired_utility',
                label='utility_continuation_'+str(job['source_job']),conditions=[job['condition']],
                output=job['source_output'] if skipped else job['output'],
                source_job=job['source_job'],continuation_audit=str(done_path),
                continuation_plan=str(recovery/'plan.json'),
                expected_allocation_per_condition=job['expected_allocation_decisions'],
                expected_pairs=job['expected_pairs']))
    for milestone in (250, 500):
        path = campaign/f'policy_calibration_{milestone}k/matched_job.json'
        if not path.exists():
            continue
        saved = json.loads(path.read_text())
        jobs.append(dict(job_id=saved['job_id'], kind='policy_calibration',
            label=f'matched_continuation_policy_{milestone}k', conditions=[saved['condition']], output=saved['output'],
            expected_allocation_per_condition=saved['expected_allocation_decisions'],
            expected_pairs=saved['expected_pairs']))
    path = campaign/'utility_500k_gru10_job.json'
    if path.exists():
        saved = json.loads(path.read_text())
        jobs.append(dict(job_id=saved['job_id'], kind='diagnostic_paired_utility',
            label='utility_500k_gru10', conditions=saved['conditions'], output=saved['output'],
            expected_allocation_per_condition=saved['expected_allocation_decisions'],
            expected_pairs=saved['expected_pairs']))
    return jobs


def selected_utility_progress(campaign):
    folder = campaign/'selected_utility_500k'
    if not (folder/'array_job.json').exists():
        return []
    submitted = json.loads((folder/'array_job.json').read_text())
    plan_bytes = (folder/'plan.json').read_bytes()
    plan = json.loads(plan_bytes)
    proc = subprocess.run(['sacct','--array','-j',str(submitted['job_id']),'-n','-P',
        '--format=JobID,State,Elapsed,ExitCode,NodeList'],capture_output=True,text=True,check=True)
    scheduler = {}
    for line in proc.stdout.splitlines():
        job,state,elapsed,code,node = line.split('|')[:5]
        if '.' not in job:
            scheduler[job] = dict(state=state,elapsed=elapsed,exit=code,node=node)
    rows = []
    for shard in range(plan['shards']):
        job = f"{submitted['job_id']}_{shard}"
        output = folder/f'shard_{shard}'
        expected = sum(r['shard']==shard for r in plan['selected'])*plan['repetitions']
        pairs = json_count(output/'primary_pairs.json')
        complete = False
        if (output/'completion.json').exists():
            saved = json.loads((output/'completion.json').read_text())
            complete = (saved['status']=='COMPLETE' and pairs==expected
                and saved['selected_pairs']==expected and saved['checkpoint']==plan['checkpoint']
                and saved['plan_sha256']==hashlib.sha256(plan_bytes).hexdigest())
        rows.append(dict(job_id=job,**scheduler.get(job,dict(state='UNKNOWN')),
            primary_pairs=pairs,expected_primary_pairs=expected,
            natural_references_verified=(output/'natural_references.json').exists(),
            artifact_complete=complete,output=str(output)))
    recovery_registry = campaign/'selected_utility_recoveries.json'
    if recovery_registry.exists():
        for recovery in json.loads(recovery_registry.read_text())['recoveries']:
            shard = recovery['shard']
            row = rows[shard]
            if recovery['original_job_id'] != row['job_id']:
                raise ValueError('selected utility recovery names a different original shard')
            source_row = dict(row)
            job = recovery['job_id']
            proc = subprocess.run(['sacct','-j',str(job),'-X','-n','-P',
                '--format=JobID,State,Elapsed,ExitCode,NodeList'],capture_output=True,text=True,check=True)
            records = [line.split('|') for line in proc.stdout.splitlines() if line.startswith(str(job)+'|')]
            output = Path(recovery['output'])
            count = json_count(output/'primary_pairs.json')
            done_path = output/'completion.json'
            done = json.loads(done_path.read_text()) if done_path.exists() else {}
            complete = (done.get('status')=='COMPLETE' and count==row['expected_primary_pairs']
                and done.get('selected_pairs')==count and done.get('checkpoint')==plan['checkpoint']
                and done.get('plan_sha256')==hashlib.sha256(plan_bytes).hexdigest())
            if done and not complete:
                raise ValueError('recovered selected-utility artifact has mismatched protocol/coverage')
            row.update(job_id=str(job),source_job_id=recovery['original_job_id'],
                source_scheduler=source_row,output=str(output),
                primary_pairs=count if count else source_row['primary_pairs'],
                recovered_primary_pairs=count,artifact_complete=complete,
                independent_additional_sample=False)
            if records:
                _,state,elapsed,code,node = records[0][:5]
                row.update(state=state,elapsed=elapsed,exit=code,node=node)
            else:
                row.update(state='UNKNOWN')
    return rows


def real_action_progress(campaign):
    folder=campaign/'real_action_500k'
    if not (folder/'smoke_job.json').exists():
        return []
    plan_bytes=(folder/'plan.json').read_bytes()
    plan=json.loads(plan_bytes)
    registered=[('smoke',json.loads((folder/'smoke_job.json').read_text()))]
    if (folder/'job.json').exists():
        registered.append(('readout',json.loads((folder/'job.json').read_text())))
    proc=subprocess.run(['sacct','-j',','.join(str(r['job_id']) for _,r in registered),
        '-n','-P','--format=JobID,State,Elapsed,ExitCode,NodeList'],capture_output=True,text=True,check=True)
    scheduler={}
    for line in proc.stdout.splitlines():
        job,state,elapsed,code,node=line.split('|')[:5]
        if '.' not in job:scheduler[int(job)]=dict(state=state,elapsed=elapsed,exit=code,node=node)
    reports=[]
    for kind,record in registered:
        output=folder/('smoke' if kind=='smoke' else 'result')
        pairs=json_count(output/'pairs.json')
        complete=False
        summary=output/('smoke.json' if kind=='smoke' else 'summary.json')
        if summary.exists():
            saved=json.loads(summary.read_text())
            if kind=='smoke':
                complete=(saved['status']=='EXACT_UNOBSERVED_ROLLOUT_AND_REPEAT_PASS'
                          and saved['checkpoint']==plan['checkpoint'])
            else:
                complete=(saved['status']=='COMPLETE' and pairs==plan['expected_pairs']
                    and saved['paired_repetitions']==plan['expected_pairs']
                    and len(saved['states'])==plan['expected_states']
                    and saved['checkpoint']==plan['checkpoint']
                    and saved['plan_sha256']==hashlib.sha256(plan_bytes).hexdigest())
        reports.append(dict(kind=kind,job_id=record['job_id'],
            **scheduler.get(record['job_id'],dict(state='UNKNOWN')),paired_readouts=pairs,
            expected_paired_readouts=0 if kind=='smoke' else plan['expected_pairs'],
            artifact_complete=complete,output=str(output)))
    return reports


def final_report_progress(campaign, proc_root=Path('/proc')):
    batch_registry=campaign/'completion_report_job.json'
    if batch_registry.exists():
        record=json.loads(batch_registry.read_text());folder=Path(record['output'])
        proc=subprocess.run(['sacct','-j',str(record['job_id']),'-X','-n','-P',
            '--format=JobID,State,Elapsed,ExitCode,NodeList'],capture_output=True,text=True,check=True)
        rows=[line.split('|') for line in proc.stdout.splitlines()
              if line.split('|')[0]==str(record['job_id'])]
        state={'state':'UNKNOWN'}
        if rows:
            _,status,elapsed,code,node=rows[0][:5]
            state=dict(state=status,elapsed=elapsed,exit=code,node=node)
        path=folder/'manifest.json'
        complete=json.loads(path.read_text()) if path.exists() else {}
        expected_files=['training_summary.csv','allocation_summary.csv','histograms.csv',
                        'utility_summary.csv','paired_states.csv','details.json','length_distributions.png']
        ready=(complete.get('status')=='EVIDENCE_ASSEMBLED_NOT_GOAL_COMPLETION'
            and complete.get('allocation_cohorts')==6 and complete.get('utility_cohorts')==4
            and complete.get('goal_complete') is False
            and all((folder/name).is_file() for name in expected_files))
        return dict(mode='SLURM_CPU_AGGREGATION',job_id=record['job_id'],**state,
            dependencies=record['dependencies'],checked_at=time.time(),output=str(folder),
            report_complete=ready,goal_complete=False,
            note='Report readiness is separate from scientific goal completion.')
    registry_path = campaign/'final_report_watch.json'
    if not registry_path.exists():
        return None
    registry = json.loads(registry_path.read_text())
    folder = Path(registry['output'])
    identity_path = folder/'identity.json'
    identity = json.loads(identity_path.read_text()) if identity_path.exists() else {}
    local = identity.get('host') == socket.gethostname()
    alive = None
    if local and identity.get('pid') is not None:
        try:
            arguments = (proc_root/str(identity['pid'])/'cmdline').read_bytes().split(b'\0')
            decoded = [a.decode() for a in arguments if a]
            alive = (Path(registry['source']).name in [Path(a).name for a in decoded]
                     and str(folder) in decoded)
        except (OSError, UnicodeError):
            alive = False
    status_path, completion_path = folder/'status.json', folder/'completion.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    completion = json.loads(completion_path.read_text()) if completion_path.exists() else {}
    return dict(mode='LOCAL_CPU_REPORTING_ONLY',dependency_job=registry['dependency_job'],
        host=identity.get('host'),pid=identity.get('pid'),process_live_now=alive,
        checked_at=time.time(),last_dependency_observation=status,
        completion=completion,report_complete=completion.get('status')=='REPORT_COMPLETE',
        output=str(folder),note='Liveness checks the local process command, not a saved running marker.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--campaign', type=Path, required=True)
    args = ap.parse_args()
    campaign = args.campaign.resolve()
    root = campaign.parents[3]
    manifest = json.loads((campaign/'manifest.json').read_text())
    submissions = json.loads((campaign/'submissions.json').read_text())
    evaluations = json.loads((campaign/'evaluation_jobs.json').read_text())
    additional = auxiliary_jobs(campaign)
    resumes = []
    for path in sorted(campaign.glob('resume_smoke_*/submission.json')):
        resumes.append(dict(job=json.loads(path.read_text()),
            manifest=json.loads((path.parent/'manifest.json').read_text())))
    if manifest.get('smoke_runs'):
        smoke_submissions=json.loads((campaign/'smoke_submissions.json').read_text())
        for item in manifest['smoke_runs']:
            job=dict(smoke_submissions[item['condition']],wandb_url=item['wandb_url'])
            resumes.append(dict(job=job,manifest=dict(
                run_directory=item['run_directory'],xpid=item['xpid'],
                baseline=dict(real_step=item['parent_checkpoint']['actor']['real_step']),
                total_steps=item['training_steps']),
                completion_path=Path(item['config']).parent/'completion.json'))
    ids = [str(v['job_id']) for v in list(submissions.values())+list(evaluations.values())
           +additional+[item['job'] for item in resumes]]
    proc = subprocess.run(['sacct', '-j', ','.join(ids), '--noheader', '--parsable2',
        '--format=JobID,State,Elapsed,ExitCode,NodeList'], capture_output=True, text=True, check=True)
    states = {}
    for line in proc.stdout.splitlines():
        job, state, elapsed, code, node = line.split('|')[:5]
        if '.' not in job:
            states[int(job)] = dict(state=state, elapsed=elapsed, exit=code, node=node)
    records = []
    for item in manifest['runs']:
        condition = item['condition']
        if condition == 'smoke':
            continue
        run = root/'logs/thinker'/item['xpid']
        rows = csv_rows(run/'logs.csv')
        last = rows[-1] if rows else {}
        metrics_path = run/'collector_metrics.json'
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        job = submissions[condition]['job_id']
        parent_binding = item.get('parent_checkpoint', {}).get('actor', {})
        parent_step = int(parent_binding.get('real_step', 0))
        parent_rows, parent_metrics = [], {}
        if parent_binding:
            parent_run = Path(parent_binding['path']).parent
            parent_rows = csv_rows(parent_run/'logs.csv')
            if parent_rows and int(float(parent_rows[-1]['real_step'])) != parent_step:
                raise ValueError('parent log no longer matches the fixed parent checkpoint')
            parent_metrics_path = parent_run/'collector_metrics.json'
            if parent_metrics_path.exists():
                parent_metrics = json.loads(parent_metrics_path.read_text())
        step = int(float(last.get('real_step', parent_step)))
        physical = metrics.get('physical_env_steps')
        parent_physical = parent_metrics.get('physical_env_steps', 0 if not parent_binding else None)
        segment_targets = sum(float(r.get('actor/icopro_count', 0)) for r in rows)
        record = dict(condition=condition, job_id=job, **states.get(job, {'state':'UNKNOWN'}),
            **loss_health(rows),
            learner_env_steps=step, learner_progress=step/item['training_steps'],
            parent_learner_env_steps=parent_step,
            invocation_learner_env_steps=step-parent_step,
            invocation_learner_progress=(step-parent_step)/(item['training_steps']-parent_step),
            physical_env_steps=physical,
            lineage_physical_env_steps=(physical+parent_physical
                if physical is not None and parent_physical is not None else None),
            actor_updates=len(rows), bc_targets=segment_targets,
            lineage_bc_targets=(segment_targets+sum(float(r.get('actor/icopro_count', 0))
                for r in parent_rows) if parent_rows or not parent_binding else None),
            segment_counter_scope='physical_env_steps, actor_updates and bc_targets count this invocation only',
            recent_mean_length=metrics.get('window_mean_length'),
            recent_min_fraction=metrics.get('window_min_fraction'),
            recent_interior_fraction=metrics.get('window_interior_fraction'),
            recent_cap_fraction=metrics.get('window_cap_fraction'),
            recent_optional_sign_change_fraction=metrics.get('window_sign_change_fraction'),
            recent_positive_continue=metrics.get('window_ema_positive_continue_fraction'),
            recent_negative_continue=metrics.get('window_ema_negative_continue_fraction'),
            training_return_mean=metrics.get('mean_last_100_episode_return'),
            voc_gru_norm=last.get('gradient/voc_gru_applied_norm'),
            other_gru_norm=last.get('gradient/actor_gru_gradient_norm'),
            peak_actor_allocated_gib=last.get('resources/peak_allocated_gib'),
            checkpoints_actor=sorted(p.name for p in run.glob('ckp_actor.tar_step_*')),
            evaluation_job=evaluations.get(condition, {}).get('job_id'),
            evaluation_state=(states.get(evaluations[condition]['job_id'], {}).get('state','UNKNOWN')
                              if condition in evaluations else 'NOT_SCHEDULED'),
            wandb_url='https://wandb.ai/jmme425/thinker/runs/'+item['xpid'])
        outcome = campaign/f'training_outcome_{condition}.json'
        record['training_outcome'] = (json.loads(outcome.read_text())['status']
                                      if outcome.exists() else 'NOT_INDEPENDENTLY_CLASSIFIED')
        summary = campaign/('evaluation_'+condition)/'summary.json'
        record['evaluation_complete'] = summary.exists() and json.loads(summary.read_text()).get('status') == 'COMPLETE'
        record['evaluation_allocation_decisions'] = json_count(summary.parent/'allocation.json')
        record['evaluation_paired_observations'] = json_count(summary.parent/'pairs.json')
        records.append(record)
    evaluation_rows = []
    for job in additional:
        folder = Path(job['output'])
        for condition in job['conditions']:
            paired = job['kind'] in ('late_paired_utility', 'policy_calibration', 'diagnostic_paired_utility')
            allocation = folder/('allocation.json' if paired
                                  else condition+('_stages.json' if job['kind']=='query_dynamics'
                                                  else '_allocation.json'))
            summary = folder/('summary.json' if paired
                               else condition+'_summary.json')
            decisions = json_count(allocation)
            if job['kind'] == 'query_dynamics' and not allocation.exists():
                decisions = sum(json_count(path) for path in folder.glob('allocations_seed*.json'))
            pairs = json_count(folder/'pairs.json') if job['expected_pairs'] else 0
            completed = summary.exists() and json.loads(summary.read_text()).get('status')=='COMPLETE'
            if job.get('continuation_audit'):
                audit_path = Path(job['continuation_audit'])
                audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
                completed = completed and audit.get('status') in (
                    'COMPLETE_REUSED_AND_NEW_GROUPS_PASS', 'ORIGINAL_COMPLETED_CONTINUATION_NOT_NEEDED')
                completed = completed and audit.get('plan_sha256') == hashlib.sha256(
                    Path(job['continuation_plan']).read_bytes()).hexdigest()
            if completed and job['kind'] in ('allocation', 'training_policy_allocation', 'episode_coverage'):
                saved_summary = json.loads(summary.read_text())
                if saved_summary['evaluation_epsilon'] != job['expected_epsilon']:
                    raise ValueError('allocation exploration differs from registered policy')
                if job.get('expected_milestone') is not None:
                    milestone = job['expected_milestone']
                    if any(not milestone <= saved_summary['checkpoint'][kind]['real_step'] < milestone+2000
                           for kind in ('actor','model')):
                        raise ValueError('persistence evaluation used the wrong milestone')
                if job['label'].startswith('allocation_1m_'):
                    audit = folder/'final_binding_audit.json'
                    completed = audit.exists() and json.loads(audit.read_text()).get('status') == 'REGISTERED_FINAL_CHECKPOINT_AND_DECISIONS_PASS'
                if job['kind'] == 'training_policy_allocation':
                    audit = folder/'smoke.json'
                    completed = audit.exists() and json.loads(audit.read_text()).get('status') == 'EXACT_REFERENCE_AND_GATE_FORMULA_PASS'
                if job['kind'] == 'episode_coverage':
                    audit = folder/'smoke.json'
                    completed = audit.exists() and json.loads(audit.read_text()).get('status') == 'EXACT_PREFIX_WITH_EPISODE_CLOCK_PASS'
                    checks = saved_summary['exact_prefix_checks']
                    if {r['seed']:r['exact_overlap_rows'] for r in checks} != {8301:12288,8401:12288}:
                        raise ValueError('extended episode coverage lacks the exact original prefix')
            if job['kind']=='policy_calibration':
                audit = folder/'matched_prefix_audit.json'
                completed = completed and audit.exists() and json.loads(audit.read_text()).get('status') == 'SAME_INTERVENTION_ENTRIES_ACROSS_CONTINUATION_POLICIES_PASS'
            if job['kind'] == 'recurrence' and summary.exists():
                saved = json.loads(summary.read_text())
                decisions = saved['decisions']
                if saved['reference_overlap']['matched_rows'] != decisions:
                    raise ValueError('recurrence probe lacks exact reference overlap')
            correct_counts = (decisions == job['expected_allocation_per_condition']
                              and pairs == job['expected_pairs'])
            evaluation_rows.append(dict(job_id=job['job_id'],kind=job['kind'],
                condition=condition,label=job['label'],**states.get(job['job_id'],{'state':'UNKNOWN'}),
                allocation_decisions=decisions,expected_allocation_decisions=job['expected_allocation_per_condition'],
                paired_observations=pairs,expected_paired_observations=job['expected_pairs'],
                artifact_complete=bool(completed and correct_counts),reuses_source_job=job.get('source_job'),
                artifact_qa=('COUNT_MISMATCH' if completed and not correct_counts else
                             'COMPLETE' if completed else 'INCOMPLETE'),output=str(folder)))
    resume_rows = []
    for item in resumes:
        saved, job = item['manifest'], item['job']
        rows = csv_rows(Path(saved['run_directory'])/'logs.csv')
        step = int(float(rows[-1]['real_step'])) if rows else saved['baseline']['real_step']
        audit_path=item.get('completion_path')
        audit=json.loads(audit_path.read_text()) if audit_path and audit_path.exists() else {}
        resume_rows.append(dict(job_id=job['job_id'], **states.get(job['job_id'], {'state':'UNKNOWN'}),
            xpid=saved['xpid'], parent_step=saved['baseline']['real_step'],
            learner_env_steps=step, target_step=saved['total_steps'],
            observed_resumed_updates=len(rows), wandb_url=job['wandb_url'],
            **loss_health(rows),
            completion_audit_status=audit.get('status','NOT_AVAILABLE'),
            scope='Resume validation only; excluded from efficacy-condition comparisons.'))
    selected_progress = selected_utility_progress(campaign)
    action_progress = real_action_progress(campaign)
    reporting_progress = final_report_progress(campaign)
    result = dict(updated_at=time.time(), rows=records, resume_validations=resume_rows,
        auxiliary_evaluations=evaluation_rows,
        selected_utility_evaluation=selected_progress,
        real_action_readout=action_progress,
        final_allocation_report=reporting_progress,
        caution='Recent collector statistics include training exploration; they are not matched evaluation outcomes.')
    target = campaign/'progress.json'
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2)+'\n')
    temporary.replace(target)
    with (campaign/'progress.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    if evaluation_rows:
        with (campaign/'evaluation_progress.csv').open('w') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(evaluation_rows[0]))
            writer.writeheader()
            writer.writerows(evaluation_rows)
    for row in records:
        print(row['condition'], row['job_id'], row['state'], row['learner_env_steps'],
              'mean=', row['recent_mean_length'], 'cap=', row['recent_cap_fraction'],
              row['bc_loss_status'], 'history=', row['bc_history_status'])
        if row['evaluation_job'] is not None:
            print('primary evaluation',row['evaluation_job'],row['condition'],row['evaluation_state'],
                  'allocation=',row['evaluation_allocation_decisions'],
                  'pairs=',row['evaluation_paired_observations'],'complete=',row['evaluation_complete'])
    for row in evaluation_rows:
        print('evaluation',row['label'],row['condition'],row['state'],
              'allocation=',f"{row['allocation_decisions']}/{row['expected_allocation_decisions']}",
              'pairs=',f"{row['paired_observations']}/{row['expected_paired_observations']}",row['artifact_qa'])
    for row in resume_rows:
        print('resume validation', row['job_id'], row['state'],
              f"{row['parent_step']} -> {row['learner_env_steps']}/{row['target_step']}",
              'updates=', row['observed_resumed_updates'])
    for row in selected_progress:
        print('selected-state utility',row['job_id'],row['state'],
            f"{row['primary_pairs']}/{row['expected_primary_pairs']}",
            'natural_reference=',row['natural_references_verified'],'complete=',row['artifact_complete'])
    for row in action_progress:
        print('next-action readout',row['kind'],row['job_id'],row['state'],
            f"{row['paired_readouts']}/{row['expected_paired_readouts']}",
            'complete=',row['artifact_complete'])
    if reporting_progress:
        if reporting_progress['mode']=='SLURM_CPU_AGGREGATION':
            print('final allocation report',reporting_progress['job_id'],reporting_progress['state'],
                  'complete=',reporting_progress['report_complete'],'goal_complete=False')
        else:
            print('final allocation report','local_process_live=',reporting_progress['process_live_now'],
                'dependency=',reporting_progress['dependency_job'],
                'complete=',reporting_progress['report_complete'])


if __name__ == '__main__':
    main()
