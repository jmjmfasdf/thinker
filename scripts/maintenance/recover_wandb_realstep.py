#!/usr/bin/env python3
"""Recover CSV metrics with W&B _step == each observation's real_step.

Both sources are monotonic but asynchronous. Only steps strictly below BOTH
source watermarks are committed, so a lagging actor or repeated model step
cannot arrive behind W&B's append-only cursor. Repeated source observations at
one real step use their last value for charts; original CSVs remain untouched.
"""
import argparse
import ast
import csv
import fcntl
import heapq
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid

SOURCES = ("actor", "model")


class CursorPending(RuntimeError):
    """W&B acknowledged history is not visible through its read API yet."""


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def value_of(value):
    if not value.strip():
        return None
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        try:
            parsed = float(value)
        except ValueError:
            return value
    if isinstance(parsed, float) and not math.isfinite(parsed):
        return None
    return list(parsed) if isinstance(parsed, tuple) else parsed


class CsvSource:
    def __init__(self, path, source, skip=0):
        self.path, self.source = Path(path), source
        self.handle = self.path.open("rb")
        self.inode = self.path.stat().st_ino
        header = self.handle.readline()
        if not header.endswith(b"\n"):
            raise ValueError("Incomplete header: " + str(path))
        self.fields = next(csv.reader([header.decode()]))
        if len(self.fields) != len(set(self.fields)):
            raise ValueError("Duplicate CSV fields")
        self.step_column = self.fields.index("real_step")
        self.tick_column = self.fields.index("_tick")
        self.rows, self.last_step, self.last_tick = 0, -1, -1
        for _ in range(skip):
            if self.read(parse=False) is None:
                raise ValueError("Remote cursor exceeds complete source rows")

    def read(self, parse=True):
        stat = self.path.stat()
        offset = self.handle.tell()
        if stat.st_ino != self.inode or stat.st_size < offset:
            raise RuntimeError("Source CSV was replaced or truncated")
        line = self.handle.readline()
        if not line.endswith(b"\n"):
            self.handle.seek(offset)
            return None
        fields = next(csv.reader([line.decode()], strict=True))
        if len(fields) != len(self.fields):
            raise ValueError("CSV field count mismatch")
        step, tick = int(fields[self.step_column]), int(fields[self.tick_column])
        if step < self.last_step or tick <= self.last_tick:
            raise ValueError("Source steps/ticks are not monotonic")
        self.rows += 1
        self.last_step, self.last_tick = step, tick
        event = {"source": self.source, "rows": self.rows, "real_step": step,
                 "tick": tick, "offset": self.handle.tell()}
        if parse:
            data = {}
            for key, value in zip(self.fields, fields):
                if key in ("_tick", "# _tick", "_time"):
                    continue
                parsed = value_of(value)
                if parsed is None:
                    continue
                # Model's optimization counter is not actor's counter.
                if self.source == "model" and key == "step":
                    key = "model/step"
                data[key] = parsed
            event["source_time"] = float(fields[self.fields.index("_time")])
            event["data"] = data
        return event


class StepMerger:
    def __init__(self, last_step=-1, counts=None):
        self.last_step = last_step
        self.counts = dict(counts or {s: 0 for s in SOURCES})
        self.latest = {s: -1 for s in SOURCES}
        self.pending, self.heap = {}, []
        self.collapsed = {s: 0 for s in SOURCES}

    def add(self, event):
        source, step = event["source"], event["real_step"]
        if step <= self.last_step:
            raise RuntimeError("Unacknowledged row is behind the committed step")
        if step < self.latest[source]:
            raise ValueError("Source moved backwards")
        self.latest[source] = step
        if step not in self.pending:
            self.pending[step] = {}
            heapq.heappush(self.heap, step)
        if source in self.pending[step]:
            self.collapsed[source] += 1
        self.pending[step][source] = event

    def drain(self, final=False):
        watermark = math.inf if final else min(self.latest.values())
        while self.heap and self.heap[0] < watermark:
            step = heapq.heappop(self.heap)
            group = self.pending.pop(step)
            data = {}
            # Actor metadata wins any generic shared names, as in normal logger.
            for source in ("model", "actor"):
                if source not in group:
                    continue
                event = group[source]
                data.update(event["data"])
                self.counts[source] = event["rows"]
                data[f"recovery/{source}_source_time"] = event["source_time"]
                data[f"recovery/{source}_tick"] = event["tick"]
                if source == "model":
                    data["model/real_step"] = step
            data["real_step"] = step
            data.update({f"recovery/{s}_rows": self.counts[s] for s in SOURCES})
            self.last_step = step
            yield step, data


def job_active(job):
    result = subprocess.run(["squeue", "-h", "-j", job, "-o", "%T"],
                            text=True, capture_output=True, timeout=20)
    if result.returncode:
        return "Invalid job id" not in result.stderr
    return bool(result.stdout.strip())


def get_remote(api, path):
    import wandb
    try:
        return api.run(path)
    except wandb.errors.CommError as error:
        if "could not find run" in str(error).lower() or "not found" in str(error).lower():
            return None
        raise


def remote_cursor(remote):
    if remote is None:
        return -1, {s: 0 for s in SOURCES}
    if not remote.config.get("logging_recovery", {}).get("csv_realstep_v2"):
        raise RuntimeError("Remote history is not owned by real-step recovery")
    # A repeated metadata query can briefly return an older cursor after a
    # backfill. A fresh operation avoids resuming from that stale response.
    query = ("query RecoveryCursor" + uuid.uuid4().hex +
             "($project:String!,$entity:String!,$name:String!){"
             "project(name:$project,entityName:$entity){run(name:$name){historyKeys}}}")
    result = remote._exec(query)
    history_keys = (result.get("project") or {}).get("run") or {}
    last = (history_keys.get("historyKeys") or {}).get("lastStep", -1)
    if last is None or last < 0:
        return -1, {s: 0 for s in SOURCES}
    keys = ["_step", "real_step", "recovery/actor_rows", "recovery/model_rows"]
    records = list(remote.scan_history(keys=keys, min_step=last, max_step=last+1,
                                      page_size=10, use_cache=False))
    if len(records) != 1 or records[0]["_step"] != last:
        raise CursorPending("Remote history cursor is not visible yet")
    if records[0]["real_step"] != last:
        raise RuntimeError("Remote history is not aligned to real_step")
    return int(last), {s: int(records[0][f"recovery/{s}_rows"]) for s in SOURCES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--entity", default="jmme425")
    parser.add_argument("--project", default="thinker")
    parser.add_argument("--run-id")
    parser.add_argument("--poll-seconds", type=float, default=15)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.state_dir / "uploader.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(str(os.getpid()) + "\n")
    lock.flush()
    sources = {s: args.run_dir / f"logs{'_model' if s == 'model' else ''}.csv" for s in SOURCES}

    if args.audit_only:
        readers = {s: CsvSource(path, s) for s, path in sources.items()}
        merger, events, first = StepMerger(), 0, None
        while True:
            n = 0
            for reader in readers.values():
                for _ in range(128):
                    event = reader.read()
                    if event is None:
                        break
                    merger.add(event)
                    n += 1
            for step, data in merger.drain(final=(n == 0)):
                if first is None:
                    first = step
                assert data["real_step"] == step
                events += 1
            if not n:
                break
        result = {"chart_events": events, "first_step": first, "last_step": merger.last_step,
                  "source_rows": merger.counts, "repeated_step_rows_collapsed": merger.collapsed,
                  "source_files_unmodified": True}
        atomic_json(args.state_dir / "audit.json", result)
        print(json.dumps(result), flush=True)
        return

    for key in ("WANDB_SERVICE", "_WANDB_SERVICE"):
        os.environ.pop(key, None)
    os.environ["WANDB_MODE"] = "online"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import wandb
    import yaml
    run_id = args.run_id or args.run_dir.name
    path = f"{args.entity}/{args.project}/{run_id}"
    remote = get_remote(wandb.Api(timeout=45), path)
    # Historical _step rewriting is not enabled on this W&B account. Use a
    # distinct corrected ID and preserve the old run rather than deleting it.
    for attempt in range(12):
        try:
            last, counts = remote_cursor(remote)
            break
        except CursorPending:
            if attempt == 11:
                raise
            time.sleep(10)
            remote = get_remote(wandb.Api(timeout=45), path)
    config = yaml.safe_load(args.config.read_text())
    config["logging_recovery"] = {
        "csv_realstep_v2": True, "source_dir": str(args.run_dir.resolve()),
        "slurm_job": args.job, "training_unchanged": True,
        "history_index": "real_step", "actor_axis": "real_step", "model_axis": "real_step",
        "same_step_policy": "Last value per source/real_step; original CSVs and previous server history retained",
        "flush_policy": "Strictly below both source watermarks while training is active",
        "original_run_id": args.run_dir.name,
    }
    run = wandb.init(entity=args.entity, project=args.project, id=run_id,
                     name=args.run_dir.name, group=args.group, config=config,
                     dir=str(args.state_dir), tags=["csv-logging-recovery", "real-step-aligned"],
                     # Large backfills can consume future filestream rate-limit
                     # reservations in SDK 0.28.2. Keep live updates responsive
                     # after those batches; the CSV poll still bounds logging.
                     settings=wandb.Settings(init_timeout=120, x_disable_stats=True,
                                             x_file_stream_transmit_interval=1), resume="allow")
    try:
        for _ in range(12):
            if run.starting_step == last + 1:
                break
            time.sleep(10)
            try:
                last, counts = remote_cursor(get_remote(wandb.Api(timeout=45), path))
            except CursorPending:
                continue
        if run.starting_step != last + 1:
            raise RuntimeError("Resume endpoint and acknowledged history cursor disagree")
        readers = {s: CsvSource(p, s, counts[s]) for s, p in sources.items()}
        merger = StepMerger(last, counts)
        run.define_metric("real_step", hidden=True, overwrite=True)
        run.define_metric("*", step_metric="real_step", step_sync=False, overwrite=True)
        for field in readers["model"].fields:
            if field.startswith("_") or field == "real_step":
                continue
            name = "model/step" if field == "step" else field
            run.define_metric(name, step_metric="real_step", step_sync=False, overwrite=True)
        for source in SOURCES:
            for suffix in ("rows", "tick", "source_time"):
                run.define_metric(f"recovery/{source}_{suffix}", hidden=True, overwrite=True)
    except BaseException:
        run.finish(exit_code=1)
        raise
    stopped = False

    def stop(*_):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    last_change = time.monotonic()
    last_job_check, last_status = 0, 0
    active, uploaded = True, 0
    try:
        while not stopped:
            read_count = 0
            for reader in readers.values():
                for _ in range(128):
                    event = reader.read()
                    if event is None:
                        break
                    merger.add(event)
                    read_count += 1
            now = time.monotonic()
            if read_count:
                last_change = now
            if now - last_job_check >= 60:
                active = job_active(args.job)
                last_job_check = now
            final = not active and read_count == 0 and now-last_change >= 180
            for step, data in merger.drain(final=final):
                run.log(data, step=step, commit=True)
                uploaded += 1
            if now-last_status >= 15 or final:
                status = {"updated_at": time.time(), "pid": os.getpid(), "job": args.job,
                          "job_active": active, "run_url": run.url, "state": "tailing",
                          "last_queued_real_step": merger.last_step, "session_chart_events": uploaded,
                          "source_rows_committed": merger.counts,
                          "source_rows_read": {s: r.rows for s, r in readers.items()},
                          "latest_source_step": merger.latest, "pending_steps": len(merger.pending),
                          "session_repeated_step_rows_collapsed": merger.collapsed,
                          "note": "Local queue status; verify remote history separately"}
                atomic_json(args.state_dir / "status.json", status)
                print(json.dumps(status), flush=True)
                last_status = now
            if final:
                marker = args.run_dir / "finish"
                completed = marker.is_file() and json.loads(marker.read_text()).get("status") == "complete"
                run.summary["recovery/upload_complete"] = True
                run.finish(exit_code=0 if completed else 1)
                status["state"] = "upload_complete"
                atomic_json(args.state_dir / "status.json", status)
                return
            if not read_count:
                until = time.monotonic() + args.poll_seconds
                while not stopped and time.monotonic() < until:
                    time.sleep(min(1, max(0, until-time.monotonic())))
        run.finish(exit_code=1)
    except BaseException:
        run.finish(exit_code=1)
        raise


if __name__ == "__main__":
    main()
