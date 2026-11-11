#!/usr/bin/env python3
"""48 checkpoint-specific all-session captures followed by extended caches.

Host commands use only the standard library. Replay/cache execute inside the
sealed Python 3.10 runtime. No analysis/model-fitting jobs are submitted.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

REPO = Path('/home/jeongmin/thinker-dynamic-imitation')
ASSETS = Path('/home/jeongmin/visual-behav-assets')
TAG = 'allcheckpoints_allsessions_5m_20260923'
DEFAULT_OUTPUT = REPO/'research_script/outputs'/TAG
VISUAL_PARENT = REPO/'logs/visual_behav'
NODES = tuple('bml2-slave%d' % i for i in range(5,9))
FILES = (
    'research_script/00_cache.py', 'research_script/dynamic_analysis_cache.py',
    'research_script/dynamic_native_features.py', 'research_script/shared_analysis_cache.py',
    'research_script/shared_actor_policy_cache.py', 'research_script/legacy/02_persistence_action_geometry.py',
    'research_script/legacy/02_actor_policy.py', 'research_script/all_session_visual_cache.py',
    'research_script/all_session_visual_cache.pbs', 'research_script/test_dynamic_analysis.py',
    'research_script/test_dynamic_native_features.py', 'thinker/thinker/behavior_trace.py',
    'thinker/tests/test_visual_compact_storage.py',
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8*1024**2), b''):
            h.update(part)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp-%d' % os.getpid())
    temporary.write_text(json.dumps(data,indent=2)+'\n')
    temporary.replace(path)


def load(out):
    return json.loads((out/'campaign.json').read_text())


def freeze(out):
    source = out/'source'
    source.mkdir()
    for relative in FILES:
        dest = source/relative
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(REPO/relative,dest)
    for name,original in (('legacy','source-heldout-enduro-20260921-v1'),
                          ('group','source-heldout-pong-20260921-v1')):
        base = ASSETS/original
        subprocess.run(['sha256sum','-c','SHA256SUMS'],cwd=str(base),stdout=subprocess.DEVNULL,check=True)
        target = source/'runtimes'/name
        for path in sorted(base.rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts:
                continue
            dest = target/path.relative_to(base)
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,dest)
        # Overlay only storage/I/O. Actor, model, cenv, util and learning code
        # remain identical to each checkpoint's previously verified runtime.
        for relative in ('thinker/visual_behav.py','thinker/thinker/behavior_trace.py'):
            shutil.copyfile(REPO/relative,target/relative)
    checksums = []
    for path in sorted(source.rglob('*')):
        if path.is_file():
            checksums.append('%s  %s\n' % (digest(path),path.relative_to(source)))
            path.chmod(0o444)
    (source/'SHA256SUMS').write_text(''.join(checksums))
    (source/'SHA256SUMS').chmod(0o444)
    return source


def prepare(out):
    if out.exists():
        raise FileExistsError('Refusing to overwrite campaign: %s' % out)
    reference = VISUAL_PARENT/'heldout_5m_20260921'
    old = json.loads((reference/'campaign.json').read_text())
    assert len(old['tasks']) == 48
    bindings, expected_steps, old_compact_bytes = {}, {}, 0
    for task in old['tasks']:
        key = (task['subject'],task['game'])
        bindings.setdefault(key,[]).extend(task['sources'])
        capture = Path(task['capture'])
        for ep in json.loads((capture/'manifest.json').read_text())['episodes']:
            expected_steps[key] = expected_steps.get(key,0)+ep['metadata']['replay_edge_count']
            episode = json.loads((capture/ep['path']/'manifest.json').read_text())
            excluded = {p['name'] for p in episode['pack_schema'] if p['fields']==['predicted_xs']}
            old_compact_bytes += sum(f['size'] for f in episode['files']
                if not any(f['path'].startswith('packs/'+p+'/') for p in excluded))
    captures = VISUAL_PARENT/TAG
    if captures.exists():
        raise FileExistsError(captures)
    tasks = []
    # Interleave games. Each node receives 12 jobs, with scheduler-managed
    # parallelism (8 CPUs/job, at most three such jobs per 28-CPU node).
    for subject in range(1,7):
        for leave in range(1,5):
            for game in (0,1):
                old_task = next(t for t in old['tasks'] if
                    (t['subject'],t['game'],t['session']) == (subject,game,leave))
                assert digest(Path(old_task['checkpoint'])/'milestone_5000000.json') == old_task['milestone_sha256']
                sources = sorted(bindings[(subject,game)],key=lambda s:s['source_id'])
                assert len(sources) == len({s['source_id'] for s in sources}) == 11
                for s in sources:
                    path = Path(old['data_root'])/s['source_id']
                    assert path.is_file() and path.stat().st_size == s['size']
                index = len(tasks)
                name = '%s-sub%03d-leave%02d-all' % (old_task['game_name'],subject,leave)
                tasks.append(dict(index=index,subject=subject,game=game,game_name=old_task['game_name'],
                    leave_session=leave,replay_sessions=[1,2,3,4],num_actions=old_task['num_actions'],
                    checkpoint=old_task['checkpoint'],milestone_sha256=old_task['milestone_sha256'],
                    runtime='group' if old_task['dynamics_norm']=='group' else 'legacy',
                    sources=sources,capture=str(captures/'captures'/name),node=NODES[index%4],
                    expected_real_steps=expected_steps[(subject,game)],
                    cache_root=str(out/'00_dynamic_cache'/('checkpoint_leave%02d' % leave))))
    out.mkdir(parents=True)
    for name in ('reports','pbs','subjects','audit'):
        (out/name).mkdir()
    shutil.copy2(reference/'campaign.json',out/'audit/previous_heldout_campaign.json')
    source = freeze(out)
    manifest = dict(schema_version=1,status='PREPARED',created_at=time.time(),tasks=tasks,
        data_root=old['data_root'],capture_root=str(captures),output_root=str(out),source=str(source),
        source_count=132,checkpoint_count=48,checkpoint_source_pairs=528,target_step=5000000,
        compression_level=9,store_predicted_xs=False,images='all',
        include_prediction_summaries=True,include_latent_summaries=True,
        allowed_nodes=list(NODES),cpus=8,memory_gb=24,walltime_hours=168,
        full_compact_estimate_bytes=4*old_compact_bytes,
        assignment='same subject/game, every checkpoint -> all four sessions; includes training sessions',
        old_visual_children=[str(p) for p in sorted(VISUAL_PARENT.iterdir())])
    write_json(out/'campaign.json',manifest)
    print('PREPARED 48 checkpoints, 528 checkpoint/source pairs; compact baseline %.1f GiB' %
          (manifest['full_compact_estimate_bytes']/1024**3),flush=True)


def purge_old(out):
    m = load(out)
    if (out/'audit/deletion.completed.json').exists():
        raise RuntimeError('Deletion already completed')
    processes = subprocess.check_output(['ps','-u',str(os.getuid()),'-o','args='],universal_newlines=True)
    if any(line.startswith('rsync ') and 'logs/visual_behav' in line for line in processes.splitlines()):
        raise RuntimeError('A visualization rsync is active; do not delete its source')
    jobs = subprocess.check_output(['qstat','-u','jeongmin'],universal_newlines=True)
    if any('.bml2' in line for line in jobs.splitlines()):
        raise RuntimeError('Existing user PBS jobs require inspection before deletion')
    targets = [Path(s) for s in m['old_visual_children']]
    for path in targets:
        if path.parent.resolve()!=VISUAL_PARENT.resolve() or path.name==TAG or path.is_symlink():
            raise RuntimeError('Unsafe deletion target: %s' % path)
    # Keep only small provenance/log files outside the visualization directory.
    # No arrays, images, NPZ trajectories, checkpoints or numeric caches are moved/deleted here.
    backup = out/'audit/deleted_visualization_metadata'
    for target in targets:
        for directory,dirs,files in os.walk(str(target)):
            dirs[:] = [d for d in dirs if not d.endswith(('.zarr','.partial')) and not d.startswith('.')]
            for name in files:
                path = Path(directory)/name
                if path.suffix in ('.json','.md','.log','.txt','.OU','.ER') and path.stat().st_size<20*1024**2:
                    dest = backup/path.relative_to(VISUAL_PARENT)
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(path,dest)
    before = shutil.disk_usage(VISUAL_PARENT).free
    write_json(out/'audit/deletion.started.json',dict(targets=[str(p) for p in targets],started_at=time.time()))
    for path in targets:
        if path.is_dir():shutil.rmtree(path)
        elif path.exists():path.unlink()
    write_json(out/'audit/deletion.completed.json',dict(targets=[str(p) for p in targets],completed_at=time.time(),
        freed_bytes=shutil.disk_usage(VISUAL_PARENT).free-before,
        preserved='checkpoints, behavioral_data_block, existing research_script caches, frozen source snapshots'))
    print('DELETED prior visualization directories; lightweight audit retained',flush=True)


def submit(out):
    m = load(out)
    if not (out/'audit/deletion.completed.json').is_file():
        raise RuntimeError('Requested deletion has not completed')
    receipt = out/'submission.json'
    if receipt.exists():raise FileExistsError('Refusing duplicate job submission')
    free = shutil.disk_usage(out).free
    if free < m['full_compact_estimate_bytes']*1.5+80*1024**3:
        raise RuntimeError('Insufficient storage including safety allowance')
    Path(m['capture_root']).mkdir(parents=True,exist_ok=True)
    sums = ASSETS/'shared-runtime-25d766c04fe7-core.sha256'
    ledger = dict(status='SUBMITTING',created_at=time.time(),jobs=[])
    write_json(receipt,ledger)
    for task in m['tasks']:
        index = task['index']
        command = ['qsub','-N','vc-%s-s%d-l%d' % (task['game_name'][:3],task['subject'],task['leave_session']),
            '-l','select=1:ncpus=8:mem=24gb:host='+task['node'],
            '-o',str(out/'pbs'/('task-%02d.pbs.log' % index)),
            '-v','CAMPAIGN_OUTPUT=%s,CAMPAIGN_INDEX=%d,RUNTIME_SUMS_SHA=%s' % (out,index,digest(sums)),
            str(Path(m['source'])/'research_script/all_session_visual_cache.pbs')]
        job = subprocess.check_output(command,universal_newlines=True).strip()
        ledger['jobs'].append(dict(index=index,job_id=job,node=task['node'],subject=task['subject'],
                                   game=task['game'],leave_session=task['leave_session']))
        write_json(receipt,ledger)
        print('SUBMITTED',index,job,task['node'],flush=True)
    ledger['status']='SUBMITTED'
    write_json(receipt,ledger)


def monitored(command, root):
    with subprocess.Popen(command,start_new_session=True) as process:
        while True:
            try:
                code=process.wait(timeout=30)
                if code:raise subprocess.CalledProcessError(code,command)
                return
            except subprocess.TimeoutExpired:
                if shutil.disk_usage(root).free<80*1024**3:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL);process.wait()
                    raise RuntimeError('Stopped own subprocess below 80 GiB free; partial data retained')


def run(out, index, dry_only=False):
    m=load(out);task=m['tasks'][index]
    assert task['index']==index
    assert digest(Path(task['checkpoint'])/'milestone_5000000.json')==task['milestone_sha256']
    runtime=Path(m['source'])/'runtimes'/task['runtime']/'thinker'
    sys.path.insert(0,str(runtime))
    import visual_behav as visual
    assert Path(visual.__file__).resolve()==(runtime/'visual_behav.py').resolve()
    args=[sys.executable,'-B','-u',str(runtime/'visual_behav.py'),'capture',
        '--checkpoint-dir',task['checkpoint'],'--checkpoint-step','5000000',
        '--data-root',m['data_root'],'--output',task['capture'],'--devices','cpu',
        '--cpu-workers','3','--torch-threads-per-worker','2','--images','all',
        '--compression-level','9','--omit-predicted-xs','--policy-mode','checkpoint',
        '--base-seed','0','--resume']
    for source in task['sources']:args+=['--source',source['source_id']]
    with (out/'reports'/('task-%02d.dry-run.json'%index)).open('w') as f:
        subprocess.run(args+['--dry-run'],stdout=f,check=True)
    dry=json.loads((out/'reports'/('task-%02d.dry-run.json'%index)).read_text())
    semantic=dry['plan']['semantic']
    assert dry['checkpoint_payload_validation']['game_id']==task['game']
    assert {s['source_id']:s['sha256'] for s in semantic['sources']}=={s['source_id']:s['sha256'] for s in task['sources']}
    assert semantic['compression_level']==9 and semantic['store_predicted_xs'] is False
    print('PREFLIGHT_OK',index,task['game_name'],task['subject'],task['leave_session'],'11 sources / all sessions',flush=True)
    if dry_only:return
    start=time.time()
    write_json(out/'reports'/('task-%02d.running.json'%index),dict(index=index,stage='visualization',
        started_at=start,job_id=os.getenv('PBS_JOBID'),node=os.getenv('HOSTNAME'),task=task))
    monitored(args,out)
    capture=visual.validate_capture_output(task['capture'])
    assert capture['semantic']['checkpoint']['logical_xpid']==Path(task['checkpoint']).name
    assert capture['semantic']['checkpoint']['milestone_sha256']==task['milestone_sha256']
    assert {e['metadata']['session'] for e in capture['episodes']}=={1,2,3,4}
    assert sum(e['metadata']['replay_edge_count'] for e in capture['episodes'])==task['expected_real_steps']
    for ep in capture['episodes']:
        assert (ep['metadata']['subject'],ep['metadata']['game'])==(task['subject'],task['game'])
        assert ep['metadata']['store_predicted_xs'] is False and ep['metadata']['compression_level']==9
    write_json(out/'reports'/('task-%02d.visual.completed.json'%index),dict(index=index,status='COMPLETED',
        capture=task['capture'],manifest_sha256=capture['manifest_sha256'],completed_at=time.time()))
    write_json(out/'reports'/('task-%02d.running.json'%index),dict(index=index,stage='cache',
        started_at=start,job_id=os.getenv('PBS_JOBID'),task=task))
    cache_command=[sys.executable,'-B','-u',str(Path(m['source'])/'research_script/00_cache.py'),
        '--subject',str(task['subject']),'--game',str(task['game']),'--capture-dir',task['capture'],
        '--cache-root',task['cache_root'],'--include-prediction-summaries','--include-latent-summaries',
        '--native-summary-batch-size','64']
    monitored(cache_command,out)
    verify_cache(out,task,capture)
    status(out)


def verify_cache(out,task,capture):
    import pandas as pd
    cache=Path(task['cache_root'])/('sub%03d_game%d'%(task['subject'],task['game']))
    m=json.loads((cache/'manifest.json').read_text())
    assert m['status']=='COMPLETED' and m['adapter_version']==3
    assert (m['subject'],m['game'],m['num_actions'])==(task['subject'],task['game'],task['num_actions'])
    assert m['runs']==m['source_count']==11 and m['real_steps']==task['expected_real_steps']
    assert m['native_feature_options']['images'] and m['native_feature_options']['latents']
    assert m['captures'][0]['manifest_sha256']==capture['manifest_sha256']
    decisions=pd.read_csv(cache/'native_decisions.csv.gz',usecols=['real_step_id','stop','forced_stop','additional_computations'])
    assert len(decisions)==m['real_steps'] and decisions.real_step_id.is_unique
    assert (decisions.stop+decisions.forced_stop).eq(1).all()
    assert decisions.additional_computations.sum()==m['search_events']
    for name in ('native_events.csv.gz','imagination_paths.npz','forecast_outcomes.csv.gz',
                 'representation_summaries.csv.gz','native_features_schema.json','native_features_qa.json'):
        assert (cache/name).is_file() and (cache/name).stat().st_size>0
    result=dict(status='COMPLETED',index=task['index'],subject=task['subject'],game=task['game'],
        leave_session=task['leave_session'],checkpoint=task['checkpoint'],capture=task['capture'],
        capture_manifest_sha256=capture['manifest_sha256'],cache=str(cache),
        cache_manifest_sha256=digest(cache/'manifest.json'),completed_at=time.time(),
        real_steps=m['real_steps'],source_count=11,omitted_fields=['predicted_xs','predicted_xs_valid'],
        session_roles={str(s):'held_out' if s==task['leave_session'] else 'training' for s in range(1,5)})
    write_json(cache/'checkpoint_binding.json',result)
    write_json(out/'reports'/('task-%02d.completed.json'%task['index']),result)
    print('CACHE_VERIFIED',task['index'],cache,flush=True)


def status(out):
    m=load(out)
    done=[t for t in m['tasks'] if (out/'reports'/('task-%02d.completed.json'%t['index'])).is_file()]
    for subject in range(1,7):
        for game in (0,1):
            group=[t for t in done if (t['subject'],t['game'])==(subject,game)]
            if len(group)==4:
                write_json(out/'subjects'/('sub%03d_game%d.json'%(subject,game)),dict(status='COMPLETED',
                    subject=subject,game=game,checkpoints=[json.loads((out/'reports'/('task-%02d.completed.json'%t['index'])).read_text()) for t in group],
                    warning='Same human trajectories across checkpoints; not independent behavioral samples'))
    if len(done)==48:
        write_json(out/'COMPLETED.json',dict(status='COMPLETED',checkpoint_count=48,source_count=132,
            checkpoint_source_pairs=528,real_steps=sum(t['expected_real_steps'] for t in done),completed_at=time.time()))
    print('COMPLETE %d/48 visualization+cache tasks'%len(done),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','purge-old','submit','preflight','run','status'])
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--index',type=int)
    args=parser.parse_args()
    if args.command in ('run','preflight'):
        if args.index is None or not 0<=args.index<48:parser.error('--index must be 0..47')
        run(args.output,args.index,dry_only=args.command=='preflight')
    else:
        {'prepare':prepare,'purge-old':purge_old,'submit':submit,'status':status}[args.command](args.output)


if __name__=='__main__':main()
