"""Freeze and submit a read-only supplement after the main joint group job."""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT = Path("/home/jeongmin/thinker-dynamic-imitation/research_script/outputs/continuous_planning_v1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    out = args.output_root.resolve()
    original = json.loads((out/"jobs/submission.json").read_text())
    group = [row for row in original["jobs"] if row["tag"]=="joint_group"]
    if len(group)!=1 or original["status"]!="SUBMITTED":
        raise ValueError("Expected one submitted joint group job")
    dependency = group[0]["job_id"]
    if not re.fullmatch(r"[0-9]+\.[A-Za-z0-9_.-]+", dependency):
        raise ValueError("Unexpected PBS dependency ID")
    snapshot = out/"supplementary_source"
    ledger_path = out/"jobs/supplementary_submission.json"
    print(json.dumps(dict(output=str(out), snapshot=str(snapshot), dependency=dependency,
                          node="bml2-slave8", walltime="168:00:00"), indent=2), flush=True)
    if not args.submit:
        return
    if snapshot.exists() or ledger_path.exists():
        raise FileExistsError("Preserving existing supplementary snapshot/submission")
    base = Path(original["snapshot"])/"research_script/continuous_planning"
    paths = [base/"common.py", base/"08_group_report.py"]
    paths += [HERE/name for name in ("09_supplementary_checks.py", "test_supplementary_checks.py", "supplementary.pbs", "submit_supplementary.py")]
    if not all(p.is_file() for p in paths):
        raise FileNotFoundError("Supplementary sources are incomplete")
    snapshot.mkdir()
    checksums = []
    for path in paths:
        target = snapshot/path.name
        shutil.copy2(path, target)
        checksums.append(hashlib.sha256(target.read_bytes()).hexdigest()+"  "+path.name)
        target.chmod(0o444)
    (snapshot/"SHA256SUMS").write_text("\n".join(checksums)+"\n")
    (snapshot/"SHA256SUMS").chmod(0o444)
    record = dict(status="SUBMITTING", submitted_at=datetime.now(timezone.utc).isoformat(),
                  output_root=str(out), snapshot=str(snapshot), dependency=dependency,
                  node="bml2-slave8", cpus=8, memory="32gb", walltime="168:00:00")
    def save():
        temp = ledger_path.with_suffix(".tmp")
        temp.write_text(json.dumps(record, indent=2)+"\n")
        temp.replace(ledger_path)
    save()
    command = ["qsub", "-N", "cp-supplement", "-l", "select=1:ncpus=8:mem=32gb:host=bml2-slave8",
               "-l", "walltime=168:00:00", "-W", "depend=afterok:"+dependency,
               "-o", str(out/"jobs/supplementary.log"), "-v", f"CP_OUTPUT={out},CP_SUPPLEMENT={snapshot}",
               str(snapshot/"supplementary.pbs")]
    record["job_id"] = subprocess.check_output(command, universal_newlines=True).strip()
    record["status"] = "SUBMITTED"
    save()
    print("submitted supplement: "+record["job_id"], flush=True)


if __name__ == "__main__":
    main()
