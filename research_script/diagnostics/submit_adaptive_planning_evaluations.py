"""Queue bounded utility/allocation evaluation after each successful pilot.

Evaluation uses only RTX 2080 Ti, and final immutable step checkpoints. No
full-game scoring is scheduled. Submission IDs are recorded immediately.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time


FINAL_LAUNCH = r'''#!/bin/bash
set -euo pipefail
CAMPAIGN="${1:?campaign}"
CONDITION="${2:?condition}"
mapfile -t CHECKPOINTS < <(python3 - "$CAMPAIGN" "$CONDITION" <<'PY'
import json, sys
from pathlib import Path
c = Path(sys.argv[1])
condition = sys.argv[2]
manifest = json.loads((c/'manifest.json').read_text())
item = next(r for r in manifest['runs'] if r['condition'] == condition)
repo = c.parents[3]
run = repo/'logs/thinker'/item['xpid']
for kind in ('actor', 'model'):
    candidates = list(run.glob(f'ckp_{kind}.tar_step_*'))
    path = max(candidates, key=lambda p: int(p.name.rsplit('_', 1)[1]))
    if int(path.name.rsplit('_', 1)[1]) < item['training_steps']:
        raise RuntimeError('training did not reach its budget: '+str(path))
    print(path)
PY
)
test "${#CHECKPOINTS[@]}" -eq 2
exec bash "$CAMPAIGN/evaluate_v2.slurm" "$CAMPAIGN" "${CHECKPOINTS[0]}" "${CHECKPOINTS[1]}" "$CAMPAIGN/evaluation_$CONDITION"
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--campaign', type=Path, required=True)
    ap.add_argument('--submit', action='store_true')
    args = ap.parse_args()
    campaign = args.campaign.resolve()
    smoke = json.loads((campaign/'evaluation_smoke_v2/summary.json').read_text())
    if smoke['status'] != 'COMPLETE' or '2080 Ti' not in smoke['device']:
        raise RuntimeError('matched evaluation GPU smoke must pass first')
    launch = campaign/'evaluate_final.slurm'
    if launch.exists() and launch.read_text() != FINAL_LAUNCH:
        raise RuntimeError('refusing to change an existing evaluator launcher')
    launch.write_text(FINAL_LAUNCH)
    path = campaign/'evaluation_jobs.json'
    jobs = json.loads(path.read_text()) if path.exists() else {}
    training = json.loads((campaign/'submissions.json').read_text())
    for i, condition in enumerate(('control', 'common', 'explore', 'burn8', 'burn16', 'gru10')):
        if condition in jobs:
            print(condition, 'already submitted', jobs[condition]['job_id'])
            continue
        node = 'bml-compute13' if i % 2 == 0 else 'bml-compute01'
        command = ['sbatch', '--parsable', '--job-name=adapt-eval-'+condition,
            '--partition=rtx2080', '--nodelist='+node, '--gres=gpu:rtx2080:1',
            '--cpus-per-task=4', '--mem=16000M', '--time=1-00:00:00',
            '--dependency=afterok:'+str(training[condition]['job_id']),
            '--output='+str(campaign/('eval-'+condition+'-%j.out')),
            '--error='+str(campaign/('eval-'+condition+'-%j.err')),
            str(launch), str(campaign), condition]
        if not args.submit:
            print(command)
            continue
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError(result.stderr)
        job = int(result.stdout.strip().split(';')[0])
        jobs[condition] = dict(job_id=job, depends_on=training[condition]['job_id'],
                              node=node, submitted_at=time.time(), command=command)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(jobs, indent=2)+'\n')
        temporary.replace(path)
        print(condition, job, node, flush=True)


if __name__ == '__main__':
    main()
