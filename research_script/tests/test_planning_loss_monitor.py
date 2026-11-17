"""A recovered run must retain the evidence of earlier numerical instability."""
import importlib.util
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

PATH = Path(__file__).parents[1]/'diagnostics/summarize_adaptive_planning_pilot.py'
SPEC = importlib.util.spec_from_file_location('planning_loss_monitor', PATH)
MONITOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MONITOR)


def rows(values):
    return [dict(real_step=str((i+1)*100), **{'actor/icopro_nll':str(value)})
            for i,value in enumerate(values)]


def test_recovered_recent_loss_does_not_erase_historical_instability():
    health=MONITOR.loss_health(rows([1.]*20+[100.]*20+[1.]*20))
    assert health['bc_loss_status']=='NO_SUSTAINED_LARGE_NLL'
    assert health['bc_history_status']=='HISTORICAL_SUSTAINED_LARGE_NLL'
    assert health['maximum_bc_nll']==100.
    assert health['maximum_20update_median_bc_nll']==100.
    assert health['longest_large_bc_streak']==20
    assert health['first_large_bc_step']==2100
    assert health['last_large_bc_step']==4000


def test_isolated_spike_is_distinct_from_sustained_large_loss():
    health=MONITOR.loss_health(rows([1.]*20+[100.]+[1.]*20))
    assert health['bc_history_status']=='HISTORICAL_LARGE_NLL_SPIKE'
    assert health['large_bc_update_count']==1
    assert health['maximum_20update_median_bc_nll']==1.


def test_nonfinite_history_is_not_hidden_by_later_finite_rows():
    health=MONITOR.loss_health(rows([float('nan')]+[1.]*20))
    assert health['bc_history_status']=='NONFINITE_HISTORY'
    assert health['bc_loss_status']=='NO_SUSTAINED_LARGE_NLL'
    assert health['nonfinite_bc_update_count']==1


def test_empty_and_finite_stable_logs_do_not_report_instability():
    assert MONITOR.loss_health([])['bc_history_status']=='NO_DATA'
    health=MONITOR.loss_health(rows([0.,1.,2.]*10))
    assert health['bc_history_status']=='NO_LARGE_NLL_OBSERVED'
    assert health['large_bc_update_count']==0
    assert health['first_large_bc_step'] is None


def test_selected_recovery_replaces_shard_without_counting_old_pairs_twice(tmp_path,monkeypatch):
    folder=tmp_path/'selected_utility_500k';folder.mkdir()
    plan=dict(shards=1,repetitions=2,selected=[dict(shard=0)],checkpoint={'actor':'frozen'})
    plan_bytes=json.dumps(plan).encode();(folder/'plan.json').write_bytes(plan_bytes)
    (folder/'array_job.json').write_text(json.dumps(dict(job_id=10)))
    original=folder/'shard_0';original.mkdir()
    (original/'primary_pairs.json').write_text('[{}]')
    recovery=tmp_path/'recovered';recovery.mkdir()
    (tmp_path/'selected_utility_recoveries.json').write_text(json.dumps(dict(recoveries=[
        dict(shard=0,original_job_id='10_0',job_id=11,output=str(recovery))])))
    monkeypatch.setattr(MONITOR.subprocess,'run',lambda cmd,**kw:SimpleNamespace(
        stdout='10_0|FAILED|1:00|1:0|node\n' if '--array' in cmd else '11|RUNNING|0:01|0:0|node\n'))
    report=MONITOR.selected_utility_progress(tmp_path)
    assert len(report)==1 and report[0]['primary_pairs']==1
    assert report[0]['job_id']=='11' and not report[0]['artifact_complete']
    (recovery/'primary_pairs.json').write_text('[{},{}]')
    (recovery/'completion.json').write_text(json.dumps(dict(status='COMPLETE',selected_pairs=2,
        checkpoint=plan['checkpoint'],plan_sha256=hashlib.sha256(plan_bytes).hexdigest())))
    report=MONITOR.selected_utility_progress(tmp_path)
    assert report[0]['artifact_complete'] and report[0]['primary_pairs']==2
    assert report[0]['source_scheduler']['primary_pairs']==1
    assert report[0]['independent_additional_sample'] is False


def test_action_readout_completion_requires_plan_checkpoint_and_full_count(tmp_path,monkeypatch):
    folder=tmp_path/'real_action_500k';folder.mkdir()
    plan=dict(checkpoint={'actor':'frozen'},expected_pairs=2,expected_states=1)
    plan_bytes=json.dumps(plan).encode();(folder/'plan.json').write_bytes(plan_bytes)
    (folder/'smoke_job.json').write_text(json.dumps(dict(job_id=1)))
    (folder/'job.json').write_text(json.dumps(dict(job_id=2)))
    (folder/'smoke').mkdir();(folder/'result').mkdir()
    (folder/'smoke/smoke.json').write_text(json.dumps(dict(
        status='EXACT_UNOBSERVED_ROLLOUT_AND_REPEAT_PASS',checkpoint=plan['checkpoint'])))
    summary=dict(status='COMPLETE',checkpoint=plan['checkpoint'],paired_repetitions=2,
        states=[{}],plan_sha256=hashlib.sha256(plan_bytes).hexdigest())
    (folder/'result/summary.json').write_text(json.dumps(summary))
    (folder/'result/pairs.json').write_text(json.dumps([{}]))
    monkeypatch.setattr(MONITOR.subprocess,'run',lambda *a,**kw:SimpleNamespace(
        stdout='1|COMPLETED|1:00|0:0|node\n2|COMPLETED|1:00|0:0|node\n'))
    reports=MONITOR.real_action_progress(tmp_path)
    assert reports[0]['artifact_complete']
    assert not reports[1]['artifact_complete']
    (folder/'result/pairs.json').write_text(json.dumps([{},{}]))
    assert MONITOR.real_action_progress(tmp_path)[1]['artifact_complete']
    summary['plan_sha256']='different'
    (folder/'result/summary.json').write_text(json.dumps(summary))
    assert not MONITOR.real_action_progress(tmp_path)[1]['artifact_complete']


def test_report_watcher_checks_live_process_not_saved_status(tmp_path,monkeypatch):
    folder=tmp_path/'watch';folder.mkdir()
    registry=dict(output=str(folder),source='/frozen/watcher.py',dependency_job=7)
    (tmp_path/'final_report_watch.json').write_text(json.dumps(registry))
    (folder/'identity.json').write_text(json.dumps(dict(host='test-host',pid=123)))
    (folder/'status.json').write_text(json.dumps(dict(scheduler_state='RUNNING')))
    monkeypatch.setattr(MONITOR.socket,'gethostname',lambda:'test-host')
    proc=tmp_path/'proc';(proc/'123').mkdir(parents=True)
    # A reused PID with an unrelated command is not our live watcher.
    (proc/'123/cmdline').write_bytes(b'python\0unrelated.py\0')
    assert MONITOR.final_report_progress(tmp_path,proc)['process_live_now'] is False
    (proc/'123/cmdline').write_bytes(b'python\0/frozen/watcher.py\0--output\0'+str(folder).encode()+b'\0')
    assert MONITOR.final_report_progress(tmp_path,proc)['process_live_now'] is True
    (proc/'123/cmdline').unlink()
    result=MONITOR.final_report_progress(tmp_path,proc)
    assert result['process_live_now'] is False
    assert result['report_complete'] is False


def test_remote_report_watcher_liveness_remains_unknown(tmp_path):
    folder=tmp_path/'watch';folder.mkdir()
    (tmp_path/'final_report_watch.json').write_text(json.dumps(
        dict(output=str(folder),source='/frozen/watcher.py',dependency_job=7)))
    (folder/'identity.json').write_text(json.dumps(dict(host='not-this-host',pid=123)))
    assert MONITOR.final_report_progress(tmp_path)['process_live_now'] is None


def test_batch_report_readiness_does_not_mean_goal_complete(tmp_path,monkeypatch):
    folder=tmp_path/'result';folder.mkdir()
    (tmp_path/'completion_report_job.json').write_text(json.dumps(dict(
        job_id=19,output=str(folder),dependencies=[11,12])))
    monkeypatch.setattr(MONITOR.subprocess,'run',lambda *a,**kw:SimpleNamespace(
        stdout='19|COMPLETED|0:40|0:0|node\n'))
    assert MONITOR.final_report_progress(tmp_path)['report_complete'] is False
    (folder/'manifest.json').write_text(json.dumps(dict(status='EVIDENCE_ASSEMBLED_NOT_GOAL_COMPLETION',
        allocation_cohorts=6,utility_cohorts=4,goal_complete=False)))
    assert MONITOR.final_report_progress(tmp_path)['report_complete'] is False
    for name in ('training_summary.csv','allocation_summary.csv','histograms.csv','utility_summary.csv',
                 'paired_states.csv','details.json','length_distributions.png'):
        (folder/name).write_text('placeholder')
    report=MONITOR.final_report_progress(tmp_path)
    assert report['report_complete'] and report['goal_complete'] is False


def test_utility_continuation_keeps_original_provenance_and_skip_path(tmp_path):
    recovery=tmp_path/'recovery';recovery.mkdir()
    record=dict(status='SUBMITTED',job_id=8,source_job=7,condition='burn8',
        recovery=str(recovery),source_output=str(tmp_path/'original'),
        output=str(tmp_path/'continued'),expected_pairs=32,expected_allocation_decisions=16)
    (tmp_path/'utility_continuation_jobs.json').write_text(json.dumps(dict(jobs=[record])))
    result=MONITOR.auxiliary_jobs(tmp_path)
    assert len(result)==1 and result[0]['source_job']==7
    assert result[0]['output']==record['output']
    assert result[0]['continuation_audit']==str(recovery/'completion.json')
    (recovery/'completion.json').write_text(json.dumps(
        dict(status='ORIGINAL_COMPLETED_CONTINUATION_NOT_NEEDED')))
    assert MONITOR.auxiliary_jobs(tmp_path)[0]['output']==record['source_output']


def test_training_without_primary_utility_still_tracks_fixed_allocations(tmp_path,monkeypatch):
    campaign=tmp_path/'research_script/outputs/00/persistence';campaign.mkdir(parents=True)
    (campaign/'manifest.json').write_text(json.dumps(dict(runs=[dict(condition='gru10',xpid='resume',training_steps=1500000)])))
    (campaign/'submissions.json').write_text(json.dumps({'gru10':{'job_id':9}}))
    (campaign/'evaluation_jobs.json').write_text('{}')
    (campaign/'persistence_evaluation_jobs.json').write_text(json.dumps(dict(jobs=[
        dict(job_id=10,condition='gru10',milestone=1250000,output=str(campaign/'allocation'))])))
    monkeypatch.setattr(MONITOR.subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout='9|PENDING|0:00|0:0|None\n10|PENDING|0:00|0:0|None\n'))
    monkeypatch.setattr('sys.argv',['monitor','--campaign',str(campaign)])
    MONITOR.main()
    saved=json.loads((campaign/'progress.json').read_text())
    assert saved['rows'][0]['evaluation_job'] is None
    assert saved['rows'][0]['evaluation_state']=='NOT_SCHEDULED'
    assert saved['auxiliary_evaluations'][0]['job_id']==10
    assert saved['auxiliary_evaluations'][0]['artifact_complete'] is False
