"""Freeze this package and submit 12 subjects, a joint group and its supplement."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone
import hashlib

REPO = Path("/home/jeongmin/thinker-dynamic-imitation")
HERE = Path(__file__).resolve().parent
NODES = tuple("bml2-slave%d" % i for i in range(5, 9))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, default=REPO/"research_script/outputs/continuous_planning_v1")
    p.add_argument("--submit", action="store_true")
    args = p.parse_args()
    records = []
    for subject in range(1, 7):
        for game in (0, 1):
            index_path = REPO/"logs/visual_behav/heldout_5m_20260921/subjects"/f"sub{subject:03d}_game{game}.json"
            index = json.loads(index_path.read_text())
            if index["status"]!="COMPLETED" or index["source_count"]!=11:
                raise ValueError("Capture campaign is not complete")
            if (index["subject"], index["game"])!=(subject, game): raise ValueError("Capture identity mismatch")
            records.append(dict(subject=subject, game=game, node=NODES[len(records)%4], index=str(index_path)))
    print(json.dumps(dict(output=str(args.output_root), inputs=records, group="afterok all 12 subject jobs"), indent=2), flush=True)
    if not args.submit: return
    out = args.output_root.resolve()
    if out.exists(): raise FileExistsError("Refusing to overwrite an existing campaign")
    if shutil.disk_usage(out.parent).free < 40*1024**3: raise RuntimeError("Need 40 GiB free")
    (out/"jobs").mkdir(parents=True)
    snapshot = out/"source"
    paths = [p for p in HERE.iterdir() if p.suffix in (".py", ".json", ".pbs", ".md")]
    paths += [REPO/"thinker/thinker/behavior_trace.py"]
    hashes = []
    for path in sorted(paths):
        relative = path.relative_to(REPO)
        target = snapshot/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes.append(hashlib.sha256(target.read_bytes()).hexdigest()+"  "+str(relative))
        target.chmod(0o444)
    (snapshot/"SHA256SUMS").write_text("\n".join(hashes)+"\n")
    ledger = dict(status="SUBMITTING", submitted_at=datetime.now(timezone.utc).isoformat(),
                  output_root=str(out), snapshot=str(snapshot), allowed_nodes=NODES, jobs=[])
    def save():
        tmp = out/"jobs/submission.tmp"
        tmp.write_text(json.dumps(ledger, indent=2)+"\n")
        tmp.replace(out/"jobs/submission.json")
    save()
    ids = []
    for record in records+[dict(subject=0, game=0, node=NODES[0])]:
        s, g = record["subject"], record["game"]
        tag = f"sub{s:03d}_game{g}" if s else "joint_group"
        command = ["qsub", "-N", f"cp-{tag}", "-l", f"select=1:ncpus=8:mem=32gb:host={record['node']}",
                   "-o", str(out/"jobs"/(tag+".log")), "-v",
                   f"CP_OUTPUT={out},CP_SNAPSHOT={snapshot},CP_SUBJECT={s},CP_GAME={g}"]
        dependencies = [] if s else ids[:]
        if dependencies: command += ["-W", "depend=afterok:"+":".join(dependencies)]
        command += [str(snapshot/"research_script/continuous_planning/campaign.pbs")]
        job = subprocess.check_output(command, universal_newlines=True).strip()
        if s: ids.append(job)
        ledger["jobs"].append(dict(record, tag=tag, job_id=job, dependencies=dependencies))
        save(); print(f"submitted {tag}: {job} on {record['node']}", flush=True)
    ledger["status"] = "SUBMITTED"; save()
    subprocess.check_call([sys.executable, str(snapshot/"research_script/continuous_planning/submit_supplementary.py"),
                           "--output-root", str(out), "--submit"])


if __name__ == "__main__":
    main()
