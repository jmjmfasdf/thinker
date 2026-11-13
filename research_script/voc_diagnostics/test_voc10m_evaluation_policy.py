"""CPU-only coverage for campaign-level evaluation cancellation policy."""
import json
import sys

import pytest

import voc10m_parallel_monitor as monitor


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.mark.parametrize('game', ['pong', 'enduro', 'spaceinvaders'])
def test_disabled_episodes_never_created_at_future_milestones(tmp_path, game):
    write(tmp_path/'evaluation_policy.json', {'disabled_kinds': ['episodes']})
    (tmp_path/'eval_tasks').mkdir()
    for step in (3_000_000, 5_000_000, 10_000_000):
        monitor.queue_evaluations(tmp_path, {'name': game, 'game': game}, step, tmp_path/'bundle.json')
    tasks=[json.loads(p.read_text()) for p in (tmp_path/'eval_tasks').glob('*.json')]
    assert len(tasks)==12
    assert {p['kind'] for p in tasks}=={'score', 'score_sign', 'score_zero', 'benefit'}


def test_disabled_existing_task_cannot_be_submitted(tmp_path, monkeypatch):
    write(tmp_path/'evaluation_policy.json', {'disabled_kinds': ['episodes']})
    task=tmp_path/'existing_task.json'
    write(task, {'kind': 'episodes'})
    monkeypatch.setattr(monitor.subprocess, 'check_output', lambda *a, **k: pytest.fail('SLURM called'))
    with pytest.raises(ValueError, match='disabled'):
        monitor.submit_eval(tmp_path, task, 'bml-compute07')


def test_disabled_tasks_do_not_block_completion_or_erase_history(tmp_path, monkeypatch):
    cells=[{'name': g, 'game': g, 'node': n} for g,n in
           zip(('pong','enduro','spaceinvaders'), monitor.NODES)]
    registry={'training': {}, 'evaluation': {}}
    for i,cell in enumerate(cells):
        registry['training'][cell['name']]={'job_id': str(i+1), 'node': cell['node'], 'gpus': 2}
        run=tmp_path/'cells'/cell['name'];run.mkdir(parents=True);(run/'finish').touch()
    for name,kind,state in [('completed_episode','episodes','COMPLETED'),
                            ('cancelled_episode','episodes','CANCELLED'),
                            ('unsent_episode','episodes',None),
                            ('ordinary_score','score','COMPLETED')]:
        output=tmp_path/'evaluations'/name
        write(tmp_path/'eval_tasks'/f'{name}.json', {'game': 'pong', 'kind': kind, 'output': str(output)})
        if state:
            registry['evaluation'][name]={'job_id': name, 'node': 'bml-compute07', 'gpus': 1, 'state': state}
        if state=='COMPLETED':write(output/'summary.json', {'status': 'COMPLETE'})
    write(tmp_path/'campaign.json', {'cells': cells})
    write(tmp_path/'jobs.json', registry)
    write(tmp_path/'evaluation_policy.json', {'disabled_kinds': ['episodes'], 'reason': 'User cancelled'})
    states={j['job_id']: {'state': j.get('state', 'COMPLETED')}
            for group in registry.values() for j in group.values()}
    monkeypatch.setattr(monitor, 'job_states', lambda ids: states)
    monkeypatch.setattr(monitor, 'allocated_nodes', lambda: {})
    monkeypatch.setattr(monitor, 'training_summary', lambda run: {})
    monkeypatch.setattr(monitor, 'publish', lambda *args: tmp_path/'bundle.json')
    monkeypatch.setattr(monitor, 'queue_evaluations', lambda *args: None)
    monkeypatch.setattr(monitor, 'submit_eval', lambda *args: pytest.fail('Unexpected submission'))
    monkeypatch.setattr(sys, 'argv', ['monitor', '--campaign', str(tmp_path), '--once'])
    monitor.main()
    report=json.loads((tmp_path/'progress.json').read_text())
    assert report['phase']=='COMPLETE' and report['errors']==[]
    assert report['evaluation_complete']==report['evaluation_total']==1
    assert report['evaluation_disabled']==3
    assert report['disabled_evaluations']['unsent_episode']['state']=='NOT_SUBMITTED'
    assert json.loads((tmp_path/'jobs.json').read_text())['evaluation']==registry['evaluation']
    assert json.loads((tmp_path/'evaluations/completed_episode/summary.json').read_text())['status']=='COMPLETE'


def test_missing_policy_defaults_to_utility_at_every_milestone(tmp_path):
    assert monitor.evaluation_policy(tmp_path)['evaluation_partition']=='rtx2080'
    (tmp_path/'eval_tasks').mkdir()
    for step in monitor.MILESTONES:
        monitor.queue_evaluations(tmp_path, {'name': 'pong', 'game': 'pong'}, step, tmp_path/'bundle.json')
    tasks=[json.loads(p.read_text()) for p in (tmp_path/'eval_tasks').glob('*.json')]
    assert len(tasks)==len(monitor.MILESTONES)
    assert {p['real_step'] for p in tasks if p['kind']=='benefit'}==set(monitor.MILESTONES)
    assert {p['kind'] for p in tasks}=={'benefit'}


def test_prospective_rollout_preserves_original_benefit_tasks(tmp_path):
    write(tmp_path/'evaluation_policy.json', {
        'disabled_kinds': ['episodes'], 'benefit_from_milestone': {'pong': 7_500_000}})
    (tmp_path/'eval_tasks').mkdir()
    for step in monitor.MILESTONES:
        monitor.queue_evaluations(tmp_path, {'name': 'pong', 'game': 'pong'}, step, tmp_path/'bundle.json')
    before={p.name: p.read_bytes() for p in (tmp_path/'eval_tasks').glob('*.json')}
    for step in monitor.MILESTONES:
        monitor.queue_evaluations(tmp_path, {'name': 'pong', 'game': 'pong'}, step, tmp_path/'bundle.json')
    assert before=={p.name: p.read_bytes() for p in (tmp_path/'eval_tasks').glob('*.json')}
    tasks=[json.loads(data) for data in before.values()]
    assert {p['real_step'] for p in tasks if p['kind']=='benefit'}=={
        1_000_000, 3_000_000, 5_000_000, 7_500_000, 10_000_000}


def test_utility_can_be_disabled_explicitly():
    assert 'benefit' not in monitor.evaluation_kinds('pong', 7_500_000, {'disabled_kinds': ['benefit','episodes']})


def test_long_games_require_explicit_opt_in():
    assert 'episodes' in monitor.evaluation_kinds('pong', 5_000_000, {'disabled_kinds': []})


def test_unknown_disabled_kind_is_rejected(tmp_path):
    write(tmp_path/'evaluation_policy.json', {'disabled_kinds': ['episode']})
    with pytest.raises(ValueError, match='Invalid'):
        monitor.evaluation_policy(tmp_path)


def test_two_million_milestones_include_utility_and_no_extra_steps(tmp_path):
    steps=[2_000_000,4_000_000,6_000_000,8_000_000,10_000_000]
    assert monitor.evaluation_milestones({'evaluation_milestones':steps})==tuple(steps)
    (tmp_path/'eval_tasks').mkdir()
    for step in monitor.evaluation_milestones({'evaluation_milestones':steps}):
        monitor.queue_evaluations(tmp_path,{'name':'pong','game':'pong'},step,tmp_path/'bundle.json')
    tasks=[json.loads(p.read_text()) for p in (tmp_path/'eval_tasks').glob('*.json')]
    assert len(tasks)==5
    assert {p['real_step'] for p in tasks if p['kind']=='benefit'}==set(steps)


def test_separate_evaluations_only_request_rtx2080(tmp_path,monkeypatch):
    script=tmp_path/'evaluate.slurm';script.touch()
    write(tmp_path/'evaluation_policy.json',{'evaluation_partition':'rtx2080','evaluation_script':str(script)})
    task=tmp_path/'benefit.json';write(task,{'kind':'benefit'})
    calls=[]
    def submit(cmd,**kwargs):
        calls.append(cmd);return '12345\n'
    monkeypatch.setattr(monitor.subprocess,'check_output',submit)
    record=monitor.submit_eval(tmp_path,task,'bml-compute07')
    assert '--partition=rtx2080' in calls[0] and '--gres=gpu:rtx2080:1' in calls[0]
    assert not any(arg.startswith('--nodelist=') for arg in calls[0])
    assert record['node'] is None and record['partition']=='rtx2080'


@pytest.mark.parametrize('steps',[[2_000_000,2_000_000],[4_000_000,2_000_000],[],[0],[2.0]])
def test_invalid_milestones_rejected(steps):
    with pytest.raises(ValueError):monitor.evaluation_milestones({'evaluation_milestones':steps})
