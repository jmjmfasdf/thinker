"""Consolidate completed resumed runs without changing any checkpoint bytes.

Top-level learner CSVs describe the retained checkpoint lineage. Collector
telemetry describes all physical interactions, including the abandoned tail.
Original segment files remain byte-identical in attempts/ and external backups.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import yaml

REPO = Path('/work/jeongmin/thinker-dynamic-imitation')
LOGS = REPO / 'logs/thinker'
OUTPUTS = REPO / 'research_script/outputs'
SOURCE = OUTPUTS / 'voc10m_tau001_20260926/source'
AUDIT = OUTPUTS / 'log_consolidation_20261030'
CAMPAIGNS = [OUTPUTS / 'voc10m_tau001_allsubjects_leave1_20260929',
             OUTPUTS / 'voc10m_tau001_allsubjects_leave234_20261012']
ORDER = ['26399', '20886', '47047', '19718']
MILESTONES = [2000000, 4000000, 6000000, 8000000, 10000000]
SERVICE = 'voc54-t001-leave234-20261012.service'
PYTHON = '/home/jeongmin/miniconda-dynamic/envs/dynamic-thinker/bin/python'
ADDITIVE = ['physical_env_steps', 'episodes_completed', 'raw_reward_sum',
            'imagined_transitions', 'forced_stop_count'] + [f'count_{i}_decisions' for i in range(31)]
UNKNOWN_RATIOS = ['learned_stop_fraction', 'reset_fraction']
BC_KEYS = ['bc_targets_attempted', 'bc_targets_committed', 'actor_updates_committed']
HASH_CACHE = {}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sha(path):
    path = Path(path)
    st = path.stat()
    key = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
    if key not in HASH_CACHE:
        h = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
                h.update(chunk)
        HASH_CACHE[key] = h.hexdigest()
    return HASH_CACHE[key]


def link_file(src, dst):
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    assert not dst.exists(), dst
    os.link(src.resolve(), dst)
    assert os.path.samefile(src, dst)


def private_copy(src, dst):
    """Mutable conventional terminal names must not share a backup inode."""
    src, dst = Path(src), Path(dst)
    temporary = dst.with_name(dst.name + '.private.tmp')
    shutil.copy2(src, temporary)
    assert sha(src) == sha(temporary)
    temporary.replace(dst)


def archive_tree(src, dst):
    """Archive files by immutable hardlink; record links without following cycles."""
    entries = []
    for parent, dirs, files in os.walk(src, followlinks=False):
        parent = Path(parent)
        out = dst / parent.relative_to(src)
        out.mkdir(parents=True, exist_ok=True)
        for name in dirs[:] + files:
            p = parent / name
            rel = str(p.relative_to(src))
            if p.is_symlink():
                entries.append(dict(path=rel, kind='symlink', target=os.readlink(p)))
                if name in dirs:
                    dirs.remove(name)
            elif p.is_file():
                link_file(p, out / name)
                st = p.stat()
                entries.append(dict(path=rel, kind='file', size=st.st_size,
                                    inode=st.st_ino, device=st.st_dev, mtime_ns=st.st_mtime_ns))
    write(dst / 'archive_inventory.json', dict(source=str(src), entries=entries))
    return entries


def targets():
    found = {}
    for campaign in CAMPAIGNS:
        for cell in read(campaign / 'campaign.json')['cells']:
            if cell.get('job_id') not in ORDER:
                continue
            prov = read(cell['resume_provenance'])
            parent = Path(prov['parent_run'])
            resumed = Path(cell['original_run_dir'] if cell.get('reused') else cell['run_dir'])
            found[cell['job_id']] = dict(cell=cell, campaign=campaign, prov=prov,
                parent=parent, resumed=resumed, canonical=LOGS / cell['name'])
    assert set(found) == set(ORDER)
    return [found[j] for j in ORDER]


def combine_collector(parent, segment):
    result = dict(segment)
    for key in ADDITIVE:
        if segment.get(key) is not None and parent.get(key) is not None:
            result[key] = parent[key] + segment[key]
    steps = result.get('physical_env_steps', 0)
    if steps:
        result['raw_reward_per_1000_actions'] = 1000 * result['raw_reward_sum'] / steps
        result['computations_per_action'] = result['imagined_transitions'] / steps
    hist = [result.get(f'count_{i}_decisions') for i in range(31)]
    if all(v is not None for v in hist) and sum(hist):
        result['zero_search_fraction'] = hist[0] / sum(hist)
        result['cap_search_fraction'] = hist[-1] / sum(hist)
    best = [v for v in [parent.get('max_completed_episode_return'),
                       segment.get('max_completed_episode_return')] if v is not None]
    result['max_completed_episode_return'] = max(best) if best else None
    # Counts of valid controls/stops/resets were not persisted. Subsampled
    # control_samples cannot supply the missing population denominators.
    for key in UNKNOWN_RATIOS:
        result[key] = None
    return result


def number(text):
    if text in ('', None, 'None', 'null', 'nan'):
        return None
    if text in ('True', 'False'):
        return text == 'True'
    return float(text)


def transform_actor(row, prov, collector):
    row = dict(row)
    for key in BC_KEYS:
        field = 'training/' + key
        if row.get(field) not in (None, ''):
            row[field] = str(int(row[field]) + prov['base_training_counters'][key])
    local = {k.removeprefix('collector/'): number(v) for k, v in row.items()
             if k.startswith('collector/')}
    if local.get('physical_env_steps') is not None:
        merged = combine_collector(collector, local)
        for key, value in merged.items():
            field = 'collector/' + key
            if field in row:
                row[field] = '' if value is None else str(value)
        # Backlog means unconsumed interactions of the current segment;
        # discarded parent interactions are not waiting in the new buffer.
        local_backlog = local['physical_env_steps'] - (int(row['real_step']) - prov['resumed_actor_step'])
        row['collector/actor_backlog_env_steps'] = str(local_backlog)
        if 'max_episode_return' in row:
            row['max_episode_return'] = str(merged['max_completed_episode_return'])
    return row


def merge_csv(old, new, dst, cutoff, prov, collector, actor):
    paths = [old, new]
    headers = []
    for p in paths:
        with p.open() as f:
            headers.append(next(csv.reader(f)))
    assert headers[0] == headers[1], 'Different CSV schemas'
    stats = dict(input_rows=[0, 0], retained_rows=[0, 0], superseded_rows=0,
                 columns=len(headers[0]), cutoff=cutoff, first_step=None, last_step=None)
    fields = headers[0]
    index_path = dst.with_name(dst.name + '.provenance.csv')
    last_step = -1
    last_time = -1
    tick = 0
    with dst.open('w', newline='') as out, index_path.open('w', newline='') as idx:
        writer = csv.DictWriter(out, fieldnames=fields)
        index = csv.writer(idx)
        writer.writeheader()
        index.writerow(['merged_tick', 'segment', 'source_tick', 'real_step'])
        for segment, src in enumerate(paths):
            with src.open(newline='') as f:
                reader = csv.DictReader(f)
                for raw in reader:
                    assert None not in raw and all(v is not None for v in raw.values()), (src, reader.line_num)
                    step = int(raw['real_step'])
                    stats['input_rows'][segment] += 1
                    if segment == 0 and step > cutoff:
                        stats['superseded_rows'] += 1
                        continue
                    assert step >= last_step, (src, step, last_step)
                    assert float(raw['_time']) >= last_time
                    if segment == 1:
                        assert step >= cutoff
                    row = transform_actor(raw, prov, collector) if actor and segment else dict(raw)
                    row['_tick'] = str(tick)
                    writer.writerow(row)
                    index.writerow([tick, segment, raw['_tick'], step])
                    stats['retained_rows'][segment] += 1
                    if stats['first_step'] is None:
                        stats['first_step'] = step
                    stats['last_step'] = step
                    last_step, last_time = step, float(raw['_time'])
                    tick += 1
    assert sum(stats['input_rows']) == tick + stats['superseded_rows']
    stats['rows'] = tick
    return stats


def merge_jsonl(old, new, dst, offset, env_n, controls=False):
    count = 0
    seen = set()
    last = -1
    for segment, src in enumerate([old, new]):
        with src.open() as f, dst.open('a') as out:
            for line in f:
                r = json.loads(line)
                assert r['attempt'] == 0, 'Unexpected nested attempt'
                r['attempt'] = segment
                r['segment_real_step'] = r['real_step']
                r['real_step'] += offset if segment else 0
                assert r['real_step'] >= last
                if controls:
                    r['segment_decision'] = r['decision']
                    r['decision'] += offset // env_n if segment else 0
                    key = (r['attempt'], r['decision'], r['stream'], r['count'])
                    assert key not in seen, key
                    seen.add(key)
                # Concurrent environments can finish at the same step with
                # identical rewards. Preserve every episode row.
                out.write(json.dumps(r, allow_nan=False) + '\n')
                last = r['real_step']
                count += 1
    return dict(rows=count, final_physical_step=last, offset=offset)


def normalized_config(src, dst, canonical):
    flags = yaml.safe_load(src.read_text())
    before = dict(flags)
    flags['savedir'] = str(canonical.parent)
    flags['ckpdir'] = str(canonical)
    assert {k for k in flags if flags[k] != before.get(k)} <= {'savedir', 'ckpdir'}
    dst.write_text(yaml.safe_dump(flags, sort_keys=True))


def localize_resume_inputs(root, canonical, provenance):
    provenance = copy.deepcopy(provenance)
    for role, record in provenance['input_checkpoints'].items():
        source = Path(record['path'])
        dest = root / 'attempts/resume_inputs' / f'ckp_{role}.tar'
        if not dest.exists():
            link_file(source, dest)
        assert sha(dest) == record['sha256']
        record['historical_path'] = str(source)
        record['path'] = str(canonical / dest.relative_to(root))
    provenance['config_sha256'] = sha(root / 'config_c.yaml')
    return provenance


def stage(target):
    cell, old, new, canonical, prov = [target[k] for k in ['cell','parent','resumed','canonical','prov']]
    dest = AUDIT / 'staging' / cell['name']
    assert not dest.exists(), 'Stage exists; inspect it before rerunning'
    assert not (canonical / 'merge_manifest.json').exists(), 'Already consolidated'
    running = subprocess.check_output(['squeue','-r','-h','-u','jeongmin','-o','%i'],text=True).splitlines()
    assert cell['job_id'] not in running, 'Run still active'
    finish = read(new / 'finish')
    assert finish['status'] == 'complete'
    for name, rec in finish['checkpoint_files'].items():
        assert sha(new / name) == rec['sha256']
    flags = yaml.safe_load((new / 'config_c.yaml').read_text())
    assert flags['total_steps'] == 10000000 and flags['voc_gru_connected'] and flags['voc_gate_q_temperature'] == .01
    parent_metrics = read(old / 'collector_metrics.json')
    assert parent_metrics['physical_env_steps'] == prov['parent_collector_steps']
    assert parent_metrics['physical_env_steps'] % flags['env_n'] == 0
    dest.mkdir(parents=True)
    print(json.dumps(dict(event='stage_start',job=cell['job_id'],name=cell['name'])),flush=True)
    inventories = [archive_tree(src, dest / 'attempts' / f'attempt{i}') for i,src in enumerate([old,new])]
    # The already-imported sub001 directory has additional provenance and
    # milestone bundles. Its complete original is also retained at install.
    stats = dict(schema_version=1, status='STAGING', job_id=cell['job_id'],
        canonical=str(canonical), sources=[str(old),str(new)], source_paths_are_historical=True,
        raw_segments=[str(canonical/'attempts'/f'attempt{i}') for i in [0,1]],
        campaign=str(target['campaign']), created_at=time.time(),
        checkpoints_byte_identical=True, bitwise_continuation=False,
        source_inventory_counts=[len(x) for x in inventories],
        scientific_config_changes=[], path_config_changes=['savedir','ckpdir'],
        scopes={'logs.csv.real_step':'retained learner checkpoint lineage',
                'logs_model.csv.real_step':'retained world-model checkpoint lineage',
                'episodes.jsonl.real_step':'all physical environment actions, including abandoned tail',
                'control_samples.jsonl.real_step':'all physical environment actions, including abandoned tail',
                'collector/*':'all physical interactions across segments; rates recomputed where denominators exist',
                'training/bc_targets_*':'retained learner lineage, offset at restored checkpoint',
                'collector/actor_backlog_env_steps':'unconsumed interactions of the current runtime segment',
                '_tick':'continuous output row index; original index in *.provenance.csv',
                'training/real_actions_per_second':'original segment rate; no downtime interpolation'},
        unavailable_aggregate_fields=UNKNOWN_RATIOS,
        unavailable_reason='Population valid-control denominators were not persisted; sampled queries are insufficient.')
    stats['logs'] = {}
    for filename, role in [('logs.csv','actor'),('logs_model.csv','model')]:
        stats['logs'][filename] = merge_csv(old/filename,new/filename,dest/filename,
            prov[f'resumed_{role}_step'],prov,parent_metrics,role=='actor')
    for filename in ['episodes.jsonl','control_samples.jsonl']:
        stats['logs'][filename] = merge_jsonl(old/filename,new/filename,dest/filename,
            prov['parent_collector_steps'],flags['env_n'],filename.startswith('control'))
    merged_metrics = combine_collector(parent_metrics,read(new/'collector_metrics.json'))
    assert merged_metrics['episodes_completed'] == stats['logs']['episodes.jsonl']['rows']
    assert sum(merged_metrics[f'count_{i}_decisions'] for i in range(31)) == merged_metrics['physical_env_steps']
    write(dest/'collector_metrics.json',merged_metrics)
    for filename in ['out.log','out_model.log']:
        with (dest/filename).open('wb') as f:
            for i, src in enumerate([old,new]):
                f.write(f'\n===== Original segment {i}: {src} =====\n'.encode())
                with (src/filename).open('rb') as inp:
                    shutil.copyfileobj(inp,f)
    normalized_config(new/'config_c.yaml',dest/'config_c.yaml',canonical)
    for filename in ['meta.json','meta_model.json']:
        meta=read(new/filename)
        first=old/filename
        first_meta=read(first) if first.exists() and first.stat().st_size else read(old/'meta.json')
        meta['date_start']=first_meta.get('date_start',meta.get('date_start'))
        if isinstance(meta.get('args'),dict):
            for k in ['savedir','ckpdir']:
                meta['args'][k]=str(canonical.parent if k=='savedir' else canonical)
        meta['consolidation']=dict(manifest=str(canonical/'merge_manifest.json'),
            raw_segments=[str(canonical/'attempts'/f'attempt{i}'/filename) for i in [0,1]])
        write(dest/filename,meta)
    provenance=copy.deepcopy(prov)
    provenance.update(parent_run=str(canonical/'attempts/attempt0'),
        run_dir=str(canonical),config=str(canonical/'config_c.yaml'),
        raw_provenance=str(canonical/'attempts/attempt1/resume_provenance.json'),
        consolidated_at=time.time(),consolidation_manifest=str(canonical/'merge_manifest.json'))
    provenance['raw_collector_and_campaign_counters_scope']='Raw attempts remain segment-local; top-level files are consolidated as specified in merge_manifest.json'
    provenance['collector_statistics_scope']='Top-level collector/JSONL cumulative physical interactions; attempts retain native segment values'
    provenance=localize_resume_inputs(dest,canonical,provenance)
    write(dest/'resume_provenance.json',provenance)
    ckps=[]
    conflicts=[]
    for role in ['actor','model']:
        cutoff=prov[f'resumed_{role}_step']
        prefix=f'ckp_{role}.tar_step_'
        candidates={}
        for i,src in enumerate([old,new]):
            for file in src.glob(prefix+'*'):
                step=int(file.name.removeprefix(prefix))
                if i==0 and step>cutoff:
                    continue
                assert i==0 or step>=cutoff
                if file.name in candidates:
                    prior=candidates[file.name]
                    conflicts.append(dict(name=file.name,old_sha256=sha(prior),new_sha256=sha(file),selected='resumed'))
                candidates[file.name]=file
        inp=Path(prov['input_checkpoints'][role]['path'])
        assert sha(inp)==prov['input_checkpoints'][role]['sha256']
        input_name=prefix+str(cutoff)
        if input_name not in candidates:
            candidates[input_name]=inp
        for name,file in sorted(candidates.items()):
            link_file(file,dest/name)
            ckps.append(dict(name=name,source=str(file),size=file.stat().st_size,identity='hardlink to immutable source'))
        private_copy(new/f'ckp_{role}.tar',dest/f'ckp_{role}.tar')
    stats['checkpoint_inventory']=ckps
    stats['checkpoint_name_conflicts']=conflicts
    stats['final_checkpoint_hashes']={role:sha(dest/f'ckp_{role}.tar') for role in ['actor','model']}
    for milestone in MILESTONES:
        original=Path(cell['run_dir'])/'milestones'/str(milestone)
        payload=read(original/'bundle.json')
        out=dest/'milestones'/str(milestone);out.mkdir(parents=True)
        source_binding=copy.deepcopy(payload)
        for kind,record in payload['files'].items():
            file=Path(record['path'])
            assert sha(file)==record['sha256']
            if kind=='config':normalized_config(file,out/file.name,canonical)
            else:link_file(file,out/file.name)
            payload['files'][kind]=dict(path=str(canonical/'milestones'/str(milestone)/file.name),sha256=sha(out/file.name))
        payload['consolidation']=dict(original_bundle_sha256=sha(original/'bundle.json'),manifest=str(canonical/'merge_manifest.json'))
        write(out/'bundle.json',payload)
        write(dest/'attempts/original_bundles'/f'{milestone}.json',source_binding)
    finish=copy.deepcopy(finish)
    for name,record in finish['checkpoint_files'].items():
        record.update(sha256=sha(dest/name),size=(dest/name).stat().st_size)
    finish['consolidation']=dict(at=time.time(),original_finish_sha256=sha(new/'finish'),manifest=str(canonical/'merge_manifest.json'))
    write(dest/'finish',finish)
    marker=read(Path(cell['run_dir'])/'campaign_complete.json')
    marker.update(finish_sha256=sha(dest/'finish'),checked_at=time.time(),
                  consolidation_manifest=str(canonical/'merge_manifest.json'))
    write(dest/'campaign_complete.json',marker)
    stats['status']='STAGED'
    stats['final_collector']=dict(physical_env_steps=merged_metrics['physical_env_steps'],episodes_completed=merged_metrics['episodes_completed'])
    write(dest/'merge_manifest.json',stats)
    write(AUDIT/'stage_reports'/f'{cell["job_id"]}.json',stats)
    print(json.dumps(dict(event='staged',job=cell['job_id'],csv_rows={k:v['rows'] for k,v in stats['logs'].items()},checkpoint_files=len(ckps))),flush=True)


def verify_csv(path, actor=False):
    previous_step=-1;previous_time=-1;rows=0;previous_bc={}
    first=None;last=None
    with path.open(newline='') as f:
        reader=csv.DictReader(f)
        for row in reader:
            assert None not in row and all(v is not None for v in row.values())
            assert int(row['_tick'])==rows
            step=int(row['real_step']);timestamp=float(row['_time'])
            assert step>=previous_step and timestamp>=previous_time
            for key in BC_KEYS if actor else []:
                field='training/'+key
                assert int(row[field])>=previous_bc.get(field,0)
                previous_bc[field]=int(row[field])
            if first is None:first=step
            last=step;previous_step=step;previous_time=timestamp;rows+=1
    return dict(rows=rows,first_step=first,last_step=last,counters=previous_bc)


def validate(target, installed=False):
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0,str(SOURCE/'thinker'))
    from thinker import util
    from thinker.learn_model import assert_finite_tensors
    cell=target['cell'];canonical=LOGS/cell['name']
    root=canonical if installed else AUDIT/'staging'/cell['name']
    manifest=read(root/'merge_manifest.json')
    checks=dict(name=cell['name'],job=cell['job_id'],at=time.time(),installed=installed)
    checks['csv']={f:verify_csv(root/f,f=='logs.csv') for f in ['logs.csv','logs_model.csv']}
    for f,result in checks['csv'].items():
        assert result['rows']==manifest['logs'][f]['rows']
    for i in [0,1]:
        folder=root/'attempts'/f'attempt{i}'
        for rec in read(folder/'archive_inventory.json')['entries']:
            if rec['kind']!='file':continue
            st=(folder/rec['path']).stat()
            assert (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)==(rec['device'],rec['inode'],rec['size'],rec['mtime_ns'])
    checks['original_files_unchanged']=True
    checks['milestones']=[]
    for step in MILESTONES:
        bundle=read(root/'milestones'/str(step)/'bundle.json')
        files={kind:root/Path(record['path']).relative_to(canonical) for kind,record in bundle['files'].items()}
        for kind,p in files.items():assert sha(p)==bundle['files'][kind]['sha256']
        flags=yaml.safe_load(files['config'].read_text())
        actor=torch.load(files['actor'],map_location='cpu',weights_only=False)
        model=torch.load(files['model'],map_location='cpu',weights_only=False)
        util.validate_voc_gate_policy_checkpoint(actor,flags=argparse.Namespace(**flags),label='consolidation')
        for role,value in [('actor',actor),('model',model)]:
            assert step<=int(value['real_step'])<step+100000
            assert value['real_step']==bundle[role+'_real_step']
            for k,v in flags.items():
                if k not in {'cmd','git_revision','ckp','ckpdir','savedir'}:
                    assert value['flags'].get(k)==v,(role,k,value['flags'].get(k),v)
            for k,state in value.items():
                if k.endswith('state_dict') and isinstance(state,dict):assert_finite_tensors(state,context=role+'.'+k)
        checks['milestones'].append(step)
        del actor,model
    finish=read(root/'finish')
    for name,record in finish['checkpoint_files'].items():assert sha(root/name)==record['sha256']
    checks['final_steps']={}
    for role in ['actor','model']:
        assert sha(root/f'ckp_{role}.tar')==manifest['final_checkpoint_hashes'][role]
        data=torch.load(root/f'ckp_{role}.tar',map_location='cpu',weights_only=False)
        assert data['real_step']>=10000000
        checks['final_steps'][role]=int(data['real_step'])
        for k,state in data.items():
            if k.endswith('state_dict') and isinstance(state,dict):assert_finite_tensors(state,context=role+'.'+k)
        del data
    marker=read(root/'campaign_complete.json')
    assert marker['finish_sha256']==sha(root/'finish')
    assert not any(p.is_symlink() for p in root.rglob('*'))
    metrics=read(root/'collector_metrics.json')
    assert sum(metrics[f'count_{i}_decisions'] for i in range(31))==metrics['physical_env_steps']
    for f in ['episodes.jsonl','control_samples.jsonl']:
        count=0;last=-1;identities=set()
        with (root/f).open() as stream:
            for line in stream:
                r=json.loads(line);assert r['real_step']>=last;last=r['real_step'];count+=1
                if f=='control_samples.jsonl':
                    key=tuple(r[k] for k in ['attempt','decision','stream','count'])
                    assert key not in identities;identities.add(key)
                    assert math.isfinite(r['p_continue']) and 0<=r['p_continue']<=1
        assert count==manifest['logs'][f]['rows']
    checks.update(status='PASS',final_checkpoint_hashes=manifest['final_checkpoint_hashes'])
    write(AUDIT/('installed_validation' if installed else 'validation')/f'{cell["job_id"]}.json',checks)
    print(json.dumps(dict(event='validation',job=cell['job_id'],status='PASS',installed=installed,final_steps=checks['final_steps'])),flush=True)
    return checks


def start_monitor():
    result=subprocess.run(['systemctl','--user','start',SERVICE],capture_output=True,text=True)
    if result.returncode:
        c=CAMPAIGNS[1]
        subprocess.run(['systemd-run','--user','--unit='+SERVICE,'--property=Restart=on-failure',
            '--property=RestartSec=30','--property=CPUQuota=100%','--property=MemoryMax=4G',
            '--property=Nice=10','--property=WorkingDirectory='+str(c),
            '--property=StandardOutput=append:'+str(c/'controller.log'),
            '--property=StandardError=append:'+str(c/'controller.log'),
            PYTHON,'-B','-u',str(c/'campaign_ops.py'),'monitor'],check=True)


def install(selected):
    assert len(selected)==4,'Install the validated set together'
    assert not (AUDIT/'installation.json').exists(),'Inspect previous installation state before retry'
    for t in selected:
        assert read(AUDIT/'validation'/f'{t["cell"]["job_id"]}.json')['status']=='PASS'
        assert not (t['canonical']/'merge_manifest.json').exists()
    for c in CAMPAIGNS:
        for f in ['campaign.json','allocation.csv','progress.json','submission.json']:
            if (c/f).exists():
                dst=AUDIT/'metadata_before'/c.name/f;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(c/f,dst)
    audit=dict(started_at=time.time(),status='INSTALLING',moves=[],updated_campaigns=[])
    write(AUDIT/'installation.json',audit)
    subprocess.run(['systemctl','--user','stop',SERVICE],check=True)
    try:
        for t in selected:
            cell=t['cell'];canonical=t['canonical'];staged=AUDIT/'staging'/cell['name']
            backup=AUDIT/'backups/canonical'/cell['name'];backup.parent.mkdir(parents=True,exist_ok=True)
            canonical.rename(backup)
            try:staged.rename(canonical)
            except BaseException:
                backup.rename(canonical);raise
            move=dict(job=cell['job_id'],canonical=str(canonical),original_backup=str(backup))
            # Only the three dedicated resume directories are relocated.
            # Historical pilot source directories remain immutable in outputs.
            if t['resumed'].parent.parent==LOGS and t['resumed'].parent.name.startswith('resume_'):
                src=t['resumed'];dst=AUDIT/'backups/resumed'/cell['name'];dst.parent.mkdir(parents=True,exist_ok=True);src.rename(dst)
                move['resumed_backup']=str(dst)
                leftovers=list(src.parent.iterdir())
                assert all(x.name=='latest' and x.is_symlink() for x in leftovers),leftovers
                container=AUDIT/'backups/resume_containers'/src.parent.name;container.parent.mkdir(parents=True,exist_ok=True);src.parent.rename(container)
            audit['moves'].append(move);write(AUDIT/'installation.json',audit)
        selected_by_name={t['cell']['name']:t for t in selected}
        for campaign in CAMPAIGNS:
            data=read(campaign/'campaign.json')
            for cell in data['cells']:
                if cell['name'] not in selected_by_name:continue
                t=selected_by_name[cell['name']];dest=t['canonical']
                cell['consolidation']=dict(at=time.time(),manifest=str(dest/'merge_manifest.json'),
                    metadata_before=str(AUDIT/'metadata_before'/campaign.name/'campaign.json'))
                cell.update(run_dir=str(dest),config=str(dest/'config_c.yaml'),config_sha256=sha(dest/'config_c.yaml'),
                    resume_provenance=str(dest/'resume_provenance.json'),raw_resume_counters_scope='Raw attempts segment-local; top-level consolidated')
                for attempt in cell.get('previous_attempts',[]):attempt['run_dir']=str(dest/'attempts/attempt0')
                if cell.get('reused'):
                    cell['original_run_dir']=str(dest)
                    cell['original_initial_run_dir']=str(dest/'attempts/attempt0')
                    cell['original_bundle_root']=str(dest/'milestones')
                elif 'original_run_dir' in cell:cell['original_run_dir']=str(dest/'attempts/attempt0')
            write(campaign/'campaign.json',data)
            allocation=campaign/'allocation.csv'
            if allocation.exists():
                with allocation.open() as f:reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
                lookup={str(c.get('array_index')):c for c in data['cells'] if c.get('array_index') is not None}
                for row in rows:
                    cell=lookup.get(row.get('array_index'))
                    if cell and cell['name'] in selected_by_name:
                        for k in ['config','run_dir']:
                            if k in row:row[k]=cell[k]
                temp=allocation.with_suffix('.csv.merge.tmp')
                with temp.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
                temp.replace(allocation)
            audit['updated_campaigns'].append(str(campaign));write(AUDIT/'installation.json',audit)
        audit.update(status='INSTALLED',finished_at=time.time());write(AUDIT/'installation.json',audit)
        for t in selected:validate(t,installed=True)
        for t in selected:
            path=t['canonical']/'merge_manifest.json'
            manifest=read(path)
            manifest.update(status='COMPLETE',installed_at=time.time(),
                backup_record=str(AUDIT/'installation.json'))
            write(path,manifest)
        audit.update(status='VERIFIED',verified_at=time.time());write(AUDIT/'installation.json',audit)
    except BaseException as error:
        # Restore directories and manifests together if any operation or
        # post-install integrity check fails. No training process is touched.
        for t in reversed(selected):
            name=t['cell']['name'];canonical=t['canonical']
            backup=AUDIT/'backups/canonical'/name
            if backup.exists():
                if canonical.exists():
                    failed=AUDIT/'failed_install'/name;failed.parent.mkdir(parents=True,exist_ok=True)
                    canonical.rename(failed)
                backup.rename(canonical)
            resumed_backup=AUDIT/'backups/resumed'/name
            if resumed_backup.exists():
                parent=t['resumed'].parent
                container=AUDIT/'backups/resume_containers'/parent.name
                if container.exists():container.rename(parent)
                parent.mkdir(parents=True,exist_ok=True)
                resumed_backup.rename(t['resumed'])
        for campaign in CAMPAIGNS:
            for filename in ['campaign.json','allocation.csv','progress.json','submission.json']:
                original=AUDIT/'metadata_before'/campaign.name/filename
                if original.exists():
                    temp=campaign/(filename+'.rollback.tmp');shutil.copy2(original,temp);temp.replace(campaign/filename)
        audit.update(status='ROLLED_BACK',error=repr(error),rolled_back_at=time.time())
        write(AUDIT/'installation.json',audit)
        raise
    finally:
        start_monitor()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['stage','validate','install','validate-installed'])
    parser.add_argument('--job',choices=ORDER)
    args=parser.parse_args();AUDIT.mkdir(exist_ok=True)
    selected=[t for t in targets() if not args.job or t['cell']['job_id']==args.job]
    for t in selected:
        if args.phase=='stage':stage(t)
        elif args.phase in ['validate','validate-installed']:validate(t,args.phase=='validate-installed')
    if args.phase=='install':install(selected)


if __name__=='__main__':
    main()
