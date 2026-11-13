"""Refresh local progress/reports until all submitted pilot jobs exit.

This collector does not submit, cancel, restart, or change any experiment.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import json
from pathlib import Path
import subprocess
import time
from report_reward_learning import generate,save,read


def latest_gate_metrics(directory):
    path=directory/'actor_metrics.jsonl'
    if not path.exists():return None
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-262144))
        lines=f.read().splitlines()
    for line in reversed(lines):
        try:row=json.loads(line)
        except json.JSONDecodeError:continue
        keys=('real_step','mean_budget','search/mean_continue_probability',
            'actor/voc_delta_q','actor/voc_gate_delta_q',
            'actor/voc_q_loss','actor/voc_gate_teacher_student_kl',
            'search/reset_ratio','search/proceed_ratio')
        return {k:row.get(k) for k in keys}
    return None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--interval',type=int,default=60)
    p.add_argument('--once',action='store_true');args=p.parse_args()
    submission=read(args.root/'submission.json')
    while True:
        queue=subprocess.run(['squeue','-h','-r','-j',submission['job_array'],'-o','%i|%T|%N|%R'],
            capture_output=True,text=True,timeout=20)
        rows=[line.split('|',3) for line in queue.stdout.strip().splitlines() if line]
        states={r[0]:r for r in rows};cells=[]
        for c in submission['cells']:
            d=read(Path(c['output'])/'status.json',{})
            r=states.get(c['job']);cells.append(dict(**c,
                scheduler_state=r[1] if r else 'NOT_IN_QUEUE',phase=d.get('phase','NOT_STARTED'),
                env_steps=d.get('env_steps',0),actor_updates=d.get('actor_updates',0),
                evaluation_step=d.get('evaluation_step'),metrics=d.get('metrics'),
                error=d.get('error'),wandb_url=d.get('wandb_url'),
                gate_diagnostics=latest_gate_metrics(Path(c['output']))))
        finished=not rows and (queue.returncode==0 or 'Invalid job id' in queue.stderr)
        status=dict(checked_at=datetime.now(timezone.utc).isoformat(),cells=cells,
            queue_query_ok=queue.returncode==0,queue_stderr=queue.stderr.strip(),scheduler_finished=finished)
        save(args.root/'live_progress.json',json.dumps(status,indent=2))
        with (args.root/'.report.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);generate(args.root)
        print(json.dumps(status),flush=True)
        if finished:
            accounting=subprocess.run(['sacct','-j',submission['job_array'],'--parsable2',
                '--format=JobID,State,ExitCode,Elapsed,NodeList'],capture_output=True,text=True,timeout=20)
            save(args.root/'final_scheduler_accounting.txt',accounting.stdout+accounting.stderr)
        if args.once or finished:break
        time.sleep(max(10,args.interval))


if __name__=='__main__':main()
