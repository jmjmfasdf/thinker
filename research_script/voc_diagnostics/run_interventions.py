"""Bounded CPU campaign: all 48 checkpoints, four early held-out strata each."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
CAPTURE = ROOT / "research_script/outputs/allcheckpoints_allsessions_5m_20260923"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 7)))
    p.add_argument("--leaves", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--games", type=int, nargs="+", default=[0, 1])
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).parent / "intervention_probe.py"
    helper = Path(__file__).parent / "forced_budget_probe.py"
    identities = {s.name: hashlib.sha256(s.read_bytes()).hexdigest() for s in (script, helper)}
    tasks = []
    for original in json.loads((CAPTURE / "campaign.json").read_text())["tasks"]:
        if original["subject"] not in args.subjects or original["game"] not in args.games or original["leave_session"] not in args.leaves:
            continue
        source = next(s for s in original["sources"] if f"/ses-{original['leave_session']:02d}/" in s["source_id"])
        tasks.append(dict(subject=original["subject"], game=original["game"], leave_session=original["leave_session"],
                          checkpoint=original["checkpoint"], runtime=original["runtime"], source=source,
                          name=f"sub{original['subject']:03d}_game{original['game']}_leave{original['leave_session']:02d}"))
    spec = dict(schema=1, tasks=tasks, script_hashes=identities, budget=3, max_target_edge=256,
                selection="first of four label-defined strata in first episode of first held-out source",
                t2_seeds_per_decision=1, reward_utility_test=False)
    specfile = args.output / "campaign.json"
    if specfile.exists():
        if not args.resume or json.loads(specfile.read_text()) != spec:
            raise ValueError("Existing campaign differs or --resume is missing")
    else:
        specfile.write_text(json.dumps(spec, indent=2))
    (args.output / "analysis_plan.md").write_text(
        "# Analysis plan fixed before the 48-checkpoint campaign\n\n"
        "Primary: output influence, measured by TV and argmax change for full(3) vs STOP(0), "
        "and full(3) vs frozen-content(3) with identical search actions and recurrent update masks. "
        "No reward utility claim follows from output change.\n\n"
        "Secondary: exact human action LL and NOOP/active log score, separately for both transitions "
        "and persistence. Human labels select strata, never the preferred intervention.\n\n"
        "T2: learned vs exact EMA probability, stochastic in both arms, common uniform variates. "
        "Report actual computation counts; this first T2 comparison is NOT compute-matched.\n\n"
        "Aggregation: each stratum separately, equal checkpoints within subject, equal subjects per game. "
        "SEM across six subjects; no event-level significance tests. One early episode per checkpoint "
        "and one T2 seed per decision limit generality. Missing strata remain explicit.\n\n"
        "Reward validation remains a separate paired online-environment experiment. No fitting, "
        "threshold selection, checkpoint selection, or retraining uses these outcomes.\n")
    def run(task):
        out = args.output / task["name"]
        if (out / "manifest.json").exists():
            manifest = json.loads((out / "manifest.json").read_text())
            if args.resume and manifest["status"] == "COMPLETE" and manifest["code_sha256"] == identities[script.name] and manifest["helper_sha256"] == identities[helper.name]:
                return dict(name=task["name"], status="REUSED", exit_code=0)
            raise ValueError(f"Refusing incompatible result {out}")
        sourcepath = ROOT / "behavioral_data_block" / task["source"]["source_id"]
        if hashlib.sha256(sourcepath.read_bytes()).hexdigest() != task["source"]["sha256"]:
            raise ValueError(f"Changed source {sourcepath}")
        env = dict(os.environ, VOC_DIAGNOSTIC_RUNTIME=str(CAPTURE / "source/runtimes" / task["runtime"] / "thinker"),
                   OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        cmd = [sys.executable, str(script), "--checkpoint", task["checkpoint"], "--source", task["source"]["source_id"],
               "--budget", "3", "--max-target-edge", "256", "--threads", "1", "--output", str(out)]
        with (args.output / (task["name"] + ".log")).open("w") as log:
            done = subprocess.run(cmd, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        return dict(name=task["name"], status="COMPLETE" if done.returncode == 0 else "FAILED", exit_code=done.returncode)
    outcomes = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                outcome = future.result()
            except Exception as error:
                outcome = dict(name=task["name"], status="FAILED", error=repr(error))
            outcomes.append(outcome)
            status = dict(expected=len(tasks), finished=len(outcomes), outcomes=outcomes,
                          updated_utc=datetime.now(timezone.utc).isoformat())
            (args.output / "status.json").write_text(json.dumps(status, indent=2))
            print(f"{len(outcomes)}/{len(tasks)} {outcome['name']} {outcome['status']}", flush=True)
    for source in (script, helper):
        if hashlib.sha256(source.read_bytes()).hexdigest() != identities[source.name]:
            raise RuntimeError("Code changed during campaign")
    if any(r["status"] == "FAILED" for r in outcomes):
        raise SystemExit("Some probes failed; inspect status.json and task logs")


if __name__ == "__main__":
    main()
