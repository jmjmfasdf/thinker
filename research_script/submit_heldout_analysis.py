#!/usr/bin/env python3
"""Freeze sources and submit 12 subjects + 2 groups on slave5-8 (full or 04-only)."""
import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FILES = [
    "research_script/00_cache.py", "research_script/dynamic_analysis_cache.py",
    "research_script/dynamic_native_features.py",
    "research_script/shared_analysis_cache.py", "research_script/shared_actor_policy_cache.py",
    "research_script/02_evaluation_revision_geometry.py", "research_script/03_actor_integration_mechanisms.py",
    "research_script/04_search_evidence_geometry_pathway_v2.py", "research_script/verify_dynamic_analysis.py",
    "research_script/test_dynamic_analysis.py", "research_script/heldout_analysis.pbs",
    "research_script/submit_heldout_analysis.py",
    "research_script/legacy/02_persistence_action_geometry.py", "research_script/legacy/02_actor_policy.py",
    "thinker/thinker/behavior_trace.py",
]
ALLOWED_NODES = tuple(f"bml2-slave{i}" for i in range(5, 9))


def assigned_node(subject, game):
    """Spread the twelve subject jobs evenly: three 8-CPU jobs per node."""
    index = (subject - 1) * 2 + game if subject else game
    return ALLOWED_NODES[index % len(ALLOWED_NODES)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=REPO / "research_script/outputs/heldout_5m_20260921")
    parser.add_argument("--submit", action="store_true", help="Without this option only validate/print the plan")
    parser.add_argument("--only-04", action="store_true", help="Reuse the completed numeric caches and rerun only 04")
    parser.add_argument("--shared-cache-root", type=Path, help="Required with --only-04; mounted read-only")
    args = parser.parse_args()
    if args.only_04 != (args.shared_cache_root is not None):
        parser.error("--only-04 and --shared-cache-root must be supplied together")
    out = args.output_root.resolve()
    cache_root = args.shared_cache_root.resolve() if args.only_04 else out / "00_dynamic_cache"
    campaign_root = REPO / "logs/visual_behav/heldout_5m_20260921"
    completed = json.loads((campaign_root / "COMPLETED.json").read_text())
    assert completed["status"] == "COMPLETED" and completed["source_count"] == 132
    records = []
    for game in (0, 1):
        for subject in range(1, 7):
            index = json.loads((campaign_root / "subjects" / f"sub{subject:03d}_game{game}.json").read_text())
            assert index["status"] == "COMPLETED" and index["source_count"] == 11
            assert (index["subject"], index["game"]) == (subject, game)
            episodes = [e for c in index["capture_roots"] for e in json.loads((Path(c) / "manifest.json").read_text())["episodes"]]
            records.append(dict(subject=subject, game=game, source_count=11,
                episodes=len(episodes), real_steps=sum(e["metadata"]["replay_edge_count"] for e in episodes),
                events=sum(e["event_count"] for e in episodes)))
            if args.only_04:
                path = cache_root / f"sub{subject:03d}_game{game}" / "manifest.json"
                cache = json.loads(path.read_text())
                assert (cache["status"], cache["subject"], cache["game"]) == ("COMPLETED", subject, game)
                assert cache["runs"] == cache["source_count"] == 11
                assert cache["num_actions"] == {0:9, 1:6}[game]
                assert cache["real_steps"] == records[-1]["real_steps"]
                assert cache["episode_count"] == records[-1]["episodes"]
                records[-1]["cache_manifest"] = str(path)
                records[-1]["cache_manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    print(json.dumps(dict(output_root=str(out), only_04=args.only_04,
        shared_cache_root=str(cache_root), allowed_nodes=ALLOWED_NODES, inputs=records), indent=2), flush=True)
    if not args.submit:
        return
    if out.exists():
        raise FileExistsError(f"Refusing to resubmit/overwrite existing campaign: {out}")
    if shutil.disk_usage(out.parent).free < 40 * 1024 ** 3:
        raise RuntimeError("Less than 40 GiB free for numeric analysis outputs")
    out.mkdir()
    jobs = out / "dynamic_jobs"
    jobs.mkdir()
    snapshot = out / "source"
    files = FILES + (["ARIAL.TTF"] if (REPO / "ARIAL.TTF").exists() else [])
    hashes = []
    for relative in files:
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / relative, target)
        hashes.append(f"{hashlib.sha256(target.read_bytes()).hexdigest()}  {relative}\n")
        target.chmod(0o444)
    (snapshot / "SHA256SUMS").write_text("".join(hashes))
    (snapshot / "SHA256SUMS").chmod(0o444)
    ledger = dict(status="SUBMITTING", created_at=datetime.now(timezone.utc).isoformat(),
        output_root=str(out), snapshot=str(snapshot), only_04=args.only_04,
        shared_cache_root=str(cache_root), allowed_nodes=ALLOWED_NODES, inputs=records, jobs=[])
    ledger_path = jobs / "submission.json"
    def save():
        temporary = ledger_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(ledger, indent=2))
        temporary.replace(ledger_path)
    save()
    def submit(subject, game, dependencies=()):
        tag = f"sub{subject:03d}_game{game}" if subject else f"group_game{game}"
        node = assigned_node(subject, game)
        command = ["qsub", "-N", f"ana-{tag}", "-l", f"select=1:ncpus=8:mem=24gb:host={node}",
            "-o", str(jobs / f"{tag}.pbs.log"), "-v",
            f"ANALYSIS_SNAPSHOT={snapshot},ANALYSIS_OUTPUT={out},ANALYSIS_GAME={game},"
            f"ANALYSIS_SUBJECT={subject},ANALYSIS_ONLY_04={int(args.only_04)},ANALYSIS_CACHE_ROOT={cache_root}"]
        if dependencies:
            command += ["-W", "depend=afterok:" + ":".join(dependencies)]
        command += [str(snapshot / "research_script/heldout_analysis.pbs")]
        job_id = subprocess.check_output(command, universal_newlines=True).strip()
        ledger["jobs"].append(dict(tag=tag, subject=subject, game=game, job_id=job_id,
            node=node, dependencies=list(dependencies)))
        save()
        print(f"submitted {tag}: {job_id} ({node})", flush=True)
        return job_id
    ids = {0:[], 1:[]}
    # Interleave games so both have early progress.
    for subject in range(1, 7):
        for game in (0, 1):
            ids[game].append(submit(subject, game))
    for game in (0, 1):
        submit(0, game, ids[game])
    ledger["status"] = "SUBMITTED"
    save()


if __name__ == "__main__":
    main()
