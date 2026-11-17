"""Bounded local CPU waiter for a registered allocation-report dependency.

Never starts, restarts, or cancels a training/evaluation job. Scheduler query
errors are observations to retry, not evidence that the GPU job stopped.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--automation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--job', type=int, required=True)
    parser.add_argument('--maximum-hours', type=float, default=24)
    args = parser.parse_args()
    if args.maximum_hours <= 0:
        raise ValueError('maximum wait must be positive')
    args.output.mkdir(exist_ok=False)
    base = dict(started_at=time.time(), pid=os.getpid(), host=socket.gethostname(),
                dependency_job=args.job, mode='LOCAL_CPU_REPORTING_ONLY',
                maximum_hours=args.maximum_hours, gpu_requested=False)
    atomic(args.output / 'identity.json', base)
    deadline = time.monotonic() + args.maximum_hours * 3600
    previous = None
    while time.monotonic() < deadline:
        try:
            process = subprocess.run(['sacct', '-j', str(args.job), '--noheader', '--parsable2',
                '--format=JobID,State,ExitCode'], capture_output=True, text=True, timeout=20)
            if process.returncode:
                raise RuntimeError(process.stderr.strip())
            records = [line.split('|') for line in process.stdout.splitlines()]
            matches = [r for r in records if r[0] == str(args.job)]
            if len(matches) != 1:
                raise RuntimeError('scheduler has no unique dependency record')
            state, exit_code = matches[0][1:3]
            status = dict(base, observed_at=time.time(), scheduler_state=state, exit_code=exit_code)
            atomic(args.output / 'status.json', status)
            if state != previous:
                print(json.dumps(status), flush=True)
                previous = state
            if state == 'COMPLETED' and exit_code == '0:0':
                command = [sys.executable, str(args.automation / 'source/finalize_planning_allocation_reports.py'),
                           '--campaign', str(args.campaign), '--automation', str(args.automation)]
                with (args.output / 'report.out').open('x') as stdout, (args.output / 'report.err').open('x') as stderr:
                    report = subprocess.run(command, stdout=stdout, stderr=stderr)
                atomic(args.output / 'completion.json', dict(base, finished_at=time.time(),
                    status='REPORT_COMPLETE' if report.returncode == 0 else 'REPORT_FAILED',
                    command=command, returncode=report.returncode))
                raise SystemExit(report.returncode)
            if state.split()[0] in {'FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL',
                                    'OUT_OF_MEMORY', 'BOOT_FAIL', 'DEADLINE', 'COMPLETED'}:
                atomic(args.output / 'completion.json', dict(base, finished_at=time.time(),
                    status='DEPENDENCY_TERMINAL_WITHOUT_SUCCESS', scheduler_state=state, exit_code=exit_code))
                raise SystemExit(2)
        except (OSError, subprocess.TimeoutExpired, RuntimeError) as error:
            atomic(args.output / 'status.json', dict(base, observed_at=time.time(),
                status='OBSERVATION_ERROR_RETRYING', error=str(error)))
        time.sleep(30)
    atomic(args.output / 'completion.json', dict(base, finished_at=time.time(),
        status='LOCAL_WATCH_DEADLINE_REACHED_GPU_JOB_UNCHANGED'))


if __name__ == '__main__':
    main()
