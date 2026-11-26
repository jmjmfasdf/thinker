#!/usr/bin/env python3
"""Describe subject-consistent control patterns; no cognitive hierarchy inference.

Uses the audited original-human bouts from human_action_bout_distributions.py.
The return-motif reference preserves incoming and outgoing action counts within
each recording and current action, but makes previous/next labels independent.
It is a descriptive conditional expectation, not a significance test or a model
of game dynamics. Input mechanics and unobserved game states remain confounds.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "research_script/outputs/01/action_bout_distributions_20261123"
BUTTONS = {
    "NOOP": set(), "FIRE": {"F"}, "RIGHT": {"R"}, "LEFT": {"L"}, "DOWN": {"D"},
    "DOWNRIGHT": {"D", "R"}, "DOWNLEFT": {"D", "L"},
    "RIGHTFIRE": {"R", "F"}, "LEFTFIRE": {"L", "F"},
}
ROUTES = [
    ("Pong", "RIGHT", "NOOP"), ("Pong", "LEFT", "NOOP"),
    ("Enduro", "RIGHTFIRE", "FIRE"), ("Enduro", "LEFTFIRE", "FIRE"),
    ("Space Invaders", "FIRE", "NOOP"),
    ("Super Mario Bros.", "A+Right", "Right"),
    ("Super Mario Bros.", "A+Right+B", "Right+B"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT)
    args = parser.parse_args()
    source = args.input
    out = source / "shared_control_patterns"
    out.mkdir(exist_ok=True)
    transitions = pd.read_csv(source / "transition_counts_by_subject.csv")
    stats = pd.read_csv(source / "action_subject_summary.csv")
    stats = stats[stats.representation.eq("exact")]
    bouts = pd.read_pickle(source / "bouts.pkl.gz")
    bouts = bouts[bouts.representation.eq("exact") & ~bouts.left_censored & ~bouts.right_censored]
    transitions["button_changes"] = [
        int(a ^ b).bit_count() if game == "Super Mario Bros." else len(BUTTONS[an] ^ BUTTONS[bn])
        for game, a, b, an, bn in zip(transitions.game, transitions.source, transitions.target,
                                    transitions.source_name, transitions.target_name)
    ]
    rows = []
    for (game, subject), d in transitions.groupby(["game", "subject"]):
        rows.append(dict(game=game, subject=subject, n_switches=int(d["count"].sum()),
            single_button_change_fraction=d.loc[d.button_changes.eq(1), "count"].sum()/d["count"].sum()))
    pd.DataFrame(rows).to_csv(out / "single_button_switches.csv", index=False)
    rows = []
    for game, a, b in ROUTES:
        selected = transitions[(transitions.game == game) & (transitions.source_name == a)]
        for subject, d in selected.groupby("subject"):
            route_count = int(d.loc[d.target_name.eq(b), "count"].sum())
            highest_count = int(d["count"].max())
            rows.append(dict(game=game, subject=subject, source=a, target=b,
                n_source_switches=int(d["count"].sum()), n_route_switches=route_count,
                probability=route_count/d["count"].sum(),
                target_is_most_frequent=(route_count == highest_count)))
    routes = pd.DataFrame(rows)
    routes.to_csv(out / "control_routes_by_subject.csv", index=False)
    routes.groupby(["game", "source", "target"]).agg(
        n_subjects=("subject", "size"), n_subjects_route_is_mode=("target_is_most_frequent", "sum"),
        mean_probability=("probability", "mean"), min_probability=("probability", "min"),
        max_probability=("probability", "max")).to_csv(out / "control_route_consistency.csv")
    contrasts = [
        ("Pong", "NOOP", ["RIGHT", "LEFT"]),
        ("Enduro", "FIRE", ["RIGHTFIRE", "LEFTFIRE"]),
        ("Space Invaders", "RIGHT", ["FIRE"]),
        ("Super Mario Bros.", "Right+B", ["Right"]),
    ]
    rows = []
    for game, numerator, denominators in contrasts:
        d = stats[stats.game.eq(game)].pivot(index="subject", columns="action_name", values="median_s")
        ratio = d[numerator]/d[denominators].mean(axis=1)
        for subject, value in ratio.items():
            rows.append(dict(game=game, subject=subject, numerator=numerator,
                denominator="mean("+",".join(denominators)+")", median_duration_ratio=value))
    pd.DataFrame(rows).to_csv(out / "duration_contrasts.csv", index=False)
    rows = []
    for (game, subject), d in bouts.groupby(["game", "subject"]):
        assert (d.previous_action >= 0).all() and (d.next_action >= 0).all()
        keys = ["file_id", "action"]
        n = d.groupby(keys).size().rename("n")
        previous = d.groupby(keys+["previous_action"]).size().rename("previous_count").reset_index()
        previous = previous.rename(columns={"previous_action": "neighbor"})
        following = d.groupby(keys+["next_action"]).size().rename("next_count").reset_index()
        following = following.rename(columns={"next_action": "neighbor"})
        reference = previous.merge(following, on=keys+["neighbor"]).join(n, on=keys)
        expected = (reference.previous_count*reference.next_count/reference.n).sum()/len(d)
        observed = (d.previous_action == d.next_action).mean()
        rows.append(dict(game=game, subject=subject, n_complete_bouts=len(d),
            return_fraction=observed, conditional_reference=expected, excess=observed-expected))
    returns = pd.DataFrame(rows)
    returns.to_csv(out / "return_motifs_by_subject.csv", index=False)
    summary = returns.groupby("game").agg(
        n_subjects=("subject", "size"), n_positive=("excess", lambda x: int((x > 0).sum())),
        mean_observed=("return_fraction", "mean"), mean_reference=("conditional_reference", "mean"),
        min_excess=("excess", "min"), max_excess=("excess", "max"))
    summary.to_csv(out / "return_motif_group.csv")
    metadata = dict(status="COMPLETED",created_at=pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in
            ["transition_counts_by_subject.csv", "action_subject_summary.csv", "bouts.pkl.gz"]},
        source_data="original human recordings, exact button sets, no model simulations",
        return_motif="previous action A, current B, next A; observed onset and exit only",
        reference="within file and current action, independent previous/next labels with empirical marginals",
        significance_tests=False, caveat="Exploratory descriptive comparisons; controller mechanics and game state are not controlled.")
    (out / "provenance.json").write_text(json.dumps(metadata, indent=2))
    print(summary.to_string())
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
