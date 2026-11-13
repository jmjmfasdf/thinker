"""Read-only SLURM/artifact monitor for the GRU information experiment."""
import argparse, json, subprocess, sys, time
from pathlib import Path
from datetime import datetime, timezone


def snapshot(root, job):
    response=subprocess.run(['squeue','-j',str(job),'-h','-o','%i %T %M %R'],capture_output=True,text=True)
    state=dict(timestamp=datetime.now(timezone.utc).isoformat(),job=job,
               scheduler=response.stdout.strip(),scheduler_ok=response.returncode==0,games={})
    for game in ['enduro','pong']:
        folder=root/'reward_v2'/game
        arms=list(folder.glob('arm_*.json'))
        repeats=list(folder.glob('repeat_*.json'))
        pairfile=folder/'pairs.json'
        pair_count=len(json.loads(pairfile.read_text())) if pairfile.exists() else 0
        report=folder/'reward_readout_report.json'
        log=root/'reward_v2'/f'{game}.log'
        lines=log.read_text().splitlines() if log.exists() else []
        complete=report.exists() and json.loads(report.read_text()).get('status')=='COMPLETE'
        historical_error=any('Traceback (most recent call last)' in line for line in lines)
        refit_log=root/'reward_v2'/f'{game}_refit.log'
        if complete and historical_error and refit_log.exists():
            lines=refit_log.read_text().splitlines()
        state['games'][game]=dict(arms_completed=len(arms),arms_total=32,
            repeated_stop_checks=len(repeats),repeat_checks_total=8,
            verified_pairs=pair_count,pairs_total=128,
            report_complete=complete,
            original_run_traceback=historical_error,
            analysis_recovered=complete and historical_error and refit_log.exists(),
            unresolved_failure=historical_error and not complete,
            last_log_line=lines[-1] if lines else '')
    tmp=root/'live_progress.json.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(root/'live_progress.json')
    subprocess.run([sys.executable,str(Path(__file__).with_name('summarize_gru_information.py')),
                    '--root',str(root)],check=True,stdout=subprocess.DEVNULL)
    return state


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--job',type=int,required=True);p.add_argument('--once',action='store_true')
    a=p.parse_args();empty_checks=0
    while True:
        state=snapshot(a.root,a.job)
        print(json.dumps(state),flush=True)
        if a.once:break
        if state['scheduler_ok'] and not state['scheduler']:
            empty_checks+=1
        else:empty_checks=0
        if empty_checks>=2:break
        time.sleep(35)


if __name__=='__main__':main()
