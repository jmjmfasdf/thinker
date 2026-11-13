"""T0: audit the EMA teacher and executed soft gate on immutable native caches.

No torch/model loading or training. RESET is a CONTINUE choice. All quantities
are binary-distribution scores, in nats/control choice. Reports distinguish
training and held-out human sessions and use equal run/checkpoint/subject
weights instead of treating millions of correlated events as independent.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.special import expit
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = ROOT / "research_script/outputs/allcheckpoints_allsessions_5m_20260923/00_dynamic_cache"
METRICS = ["kl_teacher_gate", "kl_teacher_neutral", "cross_entropy", "teacher_entropy",
           "probability_abs_error", "probability_squared_error"]


def bernoulli_scores(delta_q, gate, temperature):
    """Return stable row scores; exact gate endpoints use a reported 1e-12 floor."""
    delta_q, gate = np.asarray(delta_q, float), np.asarray(gate, float)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Q temperature must be finite and positive")
    if delta_q.shape != gate.shape or not np.isfinite(delta_q).all():
        raise ValueError("Invalid delta-Q values or mismatched shapes")
    if not np.isfinite(gate).all() or np.any((gate < 0) | (gate > 1)):
        raise ValueError("Gate probabilities must be finite and in [0,1]")
    z = delta_q / temperature
    teacher = expit(z)
    entropy = teacher * np.logaddexp(0, -z) + (1 - teacher) * np.logaddexp(0, z)
    q = np.clip(gate, 1e-12, 1 - 1e-12)
    ce = -teacher * np.log(q) - (1 - teacher) * np.log1p(-q)
    kl = ce - entropy
    if np.min(kl, initial=0) < -1e-10:
        raise ArithmeticError("Negative KL beyond rounding tolerance")
    return dict(kl_teacher_gate=np.maximum(kl, 0),
                kl_teacher_neutral=np.maximum(np.log(2) - entropy, 0),
                cross_entropy=ce, teacher_entropy=entropy,
                probability_abs_error=np.abs(teacher - gate),
                probability_squared_error=(teacher - gate) ** 2)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_cache(cache):
    binding_path = cache / "checkpoint_binding.json"
    binding = json.loads(binding_path.read_text())
    if binding["status"] != "COMPLETED":
        raise ValueError(f"Incomplete cache: {cache}")
    checkpoint = Path(binding["checkpoint"]).resolve()
    config_path = checkpoint / "config_c.yaml"
    config = yaml.safe_load(config_path.read_text())
    temperature = float(config["voc_gate_q_temperature"])
    if config.get("voc_gate_epsilon_greedy_execution", False):
        raise ValueError("This audit requires the legacy soft execution gate")
    columns = ["run_label", "real_step_id", "is_control_choice", "accepted_control",
               "voc_ema_q_valid", "voc_soft_continue_probability_valid",
               "voc_ema_voc", "voc_soft_continue_probability"]
    path = cache / "native_events.csv.gz"
    pieces, total, choices, invalid = [], 0, 0, 0
    for chunk in pd.read_csv(path, usecols=columns, chunksize=250_000):
        total += len(chunk)
        part = chunk.loc[chunk.is_control_choice.eq(1)].copy()
        choices += len(part)
        valid = (part.accepted_control.isin([0, 1, 2]) & part.voc_ema_q_valid.eq(1)
                 & part.voc_soft_continue_probability_valid.eq(1))
        invalid += int((~valid).sum())
        pieces.append(part.loc[valid])
    data = pd.concat(pieces, ignore_index=True)
    if data.empty:
        raise ValueError(f"No valid control choices: {cache}")
    # Native CSV order is event order; checking contiguity detects scrambled
    # decisions before defining the first control. No downsampling is used.
    keys = data.run_label.astype(str) + "|" + data.real_step_id.astype(str)
    blocks = keys.loc[keys.ne(keys.shift())]
    if blocks.duplicated().any():
        raise ValueError(f"Non-contiguous decision keys: {cache}")
    data["choice_index"] = data.groupby(["run_label", "real_step_id"], sort=False).cumcount()
    scores = bernoulli_scores(data.voc_ema_voc, data.voc_soft_continue_probability, temperature)
    for key, value in scores.items():
        data[key] = value
    records = []
    for run, group in data.groupby("run_label", sort=False):
        match = re.search(r"_ses(\d+)_", run)
        if match is None:
            raise ValueError(f"Missing session in run key: {run}")
        session = int(match.group(1))
        split = binding["session_roles"][str(session)]
        for scope, selected in [("all", group), ("first", group[group.choice_index.eq(0)]),
                                ("later", group[group.choice_index.gt(0)])]:
            if selected.empty:
                continue
            row = dict(subject=binding["subject"], game=binding["game"],
                       leave_session=binding["leave_session"], run=run, session=session,
                       split=split, scope=scope, n=len(selected),
                       reset_count=int(selected.accepted_control.eq(1).sum()),
                       clipped_gate_count=int(((selected.voc_soft_continue_probability < 1e-12)
                                              | (selected.voc_soft_continue_probability > 1-1e-12)).sum()),
                       q_temperature=temperature, checkpoint=str(checkpoint))
            row.update({m: float(selected[m].mean()) for m in METRICS})
            records.append(row)
    qa = dict(cache=str(cache), total_events=total, control_choices=choices,
              valid_choices=len(data), invalid_choices=invalid,
              reset_count=int(data.accepted_control.eq(1).sum()),
              source_size=path.stat().st_size, source_mtime_ns=path.stat().st_mtime_ns,
              checkpoint=str(checkpoint), q_temperature=temperature,
              config_sha256=sha256(config_path), binding_sha256=sha256(binding_path),
              cache_manifest_sha256=sha256(cache / "manifest.json"),
              capture_manifest_sha256=binding["capture_manifest_sha256"])
    return records, qa


def aggregate(records):
    runs = pd.DataFrame(records)
    # All-session rows are separately named and never labelled held out.
    combined = pd.concat([runs, runs.assign(split="all_sessions")], ignore_index=True)
    keys = ["game", "subject", "leave_session", "split", "scope"]
    caches = combined.groupby(keys, as_index=False)[METRICS].mean()
    counts = combined.groupby(keys, as_index=False).agg(n=("n", "sum"), n_runs=("run", "nunique"))
    caches = caches.merge(counts, on=keys, validate="one_to_one")
    subjects = caches.groupby(["game", "subject", "split", "scope"], as_index=False)[METRICS].mean()
    summary = []
    for key, group in subjects.groupby(["game", "split", "scope"]):
        row = dict(zip(["game", "split", "scope"], key))
        row["n_subjects"] = len(group)
        for m in METRICS:
            row[m] = float(group[m].mean())
            row[m + "_sem"] = float(group[m].std(ddof=1) / np.sqrt(len(group))) if len(group) > 1 else None
        row["fraction_neutral_kl_removed"] = (1 - row["kl_teacher_gate"] / row["kl_teacher_neutral"]
                                               if row["kl_teacher_neutral"] > 1e-12 else None)
        summary.append(row)
    return runs, caches, subjects, pd.DataFrame(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 7)))
    parser.add_argument("--games", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--leaves", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Use a new output directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = [args.cache_root / f"checkpoint_leave{leave:02d}" / f"sub{subject:03d}_game{game}"
            for leave in args.leaves for subject in args.subjects for game in args.games]
    records, qa, errors = [], [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(audit_cache, job): job for job in jobs}
        for future in as_completed(pending):
            try:
                rows, quality = future.result()
                records.extend(rows)
                qa.append(quality)
                print(f"Completed {len(qa)}/{len(jobs)}: {pending[future].name}", flush=True)
            except Exception as error:
                errors.append(dict(cache=str(pending[future]), error=repr(error)))
    if records:
        runs, caches, subjects, summary = aggregate(records)
        for name, frame in [("run_scores", runs), ("cache_scores", caches),
                            ("subject_scores", subjects), ("summary", summary)]:
            frame.sort_values(list(frame.columns[:min(4, len(frame.columns))])).to_csv(args.output / f"{name}.csv", index=False)
        lines = ["# T0: EMA teacher versus gate", "",
                 "Units: nat/control choice. Lower KL is better. Neutral gate = P(CONTINUE)=0.5.",
                 "Equal run means within checkpoint, equal checkpoints within subject, then equal subjects.",
                 "SEM describes six subjects, not independent control events. KL does not validate teacher utility.", "",
                 "| Game | Human sessions | KL teacher→gate | KL teacher→neutral | Neutral KL removed |", "|---|---|---:|---:|---:|"]
        for _, row in summary[summary.scope.eq("all")].sort_values(["game", "split"]).iterrows():
            label = {0: "Enduro", 1: "Pong"}[row.game]
            lines.append(f"| {label} | {row.split} | {row.kl_teacher_gate:.6f} ± {row.kl_teacher_gate_sem:.6f} | {row.kl_teacher_neutral:.6f} | {row.fraction_neutral_kl_removed:.1%} |")
        if errors:
            lines += ["", "INCOMPLETE: see qa.json for failed caches."]
        (args.output / "summary.md").write_text("\n".join(lines) + "\n")
    (args.output / "qa.json").write_text(json.dumps(dict(expected=len(jobs), completed=len(qa), errors=errors, caches=qa), indent=2))
    manifest = dict(status="COMPLETE" if len(qa) == len(jobs) else "INCOMPLETE",
                    created_utc=datetime.now(timezone.utc).isoformat(), source_script=str(Path(__file__).resolve()),
                    script_sha256=sha256(Path(__file__)), cache_root=str(args.cache_root),
                    arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    no_checkpoint_modified=True, reset_included=True, probability_floor=1e-12,
                    weighting="run -> checkpoint -> subject; equal at each level",
                    input_identity="cache path/stat + binding/config/cache/capture manifest hashes")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if errors:
        raise SystemExit(f"Failed caches: {len(errors)}; see qa.json")


if __name__ == "__main__":
    main()
