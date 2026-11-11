"""Complete predeclared horizon/direct-effect contrasts from immutable outputs.

This reads completed numeric reports; it does not refit, change caches, or
overwrite 08. The enlarged 03/05 secondary-family correction here supersedes
the narrower 08 secondary p-values for those families. Primary inference is
unchanged. Path agreement is NEVER tested on a full-prefix-matched subset,
because that subset already selects agreement on the target action.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as C


def group_module():
    spec = importlib.util.spec_from_file_location("cp_group_statistics", Path(__file__).with_name("08_group_report.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, default=C.DEFAULT_OUTPUT)
    args = p.parse_args()
    out = args.output_root/"09_supplementary_checks"
    group = group_module()
    group.verify_artifacts(args.output_root/"08_group_report")
    original = pd.read_csv(args.output_root/"08_group_report/subject_effects.csv", dtype={"family": str})
    signatures = {"group": C.sha(args.output_root/"08_group_report/manifest.json")}
    effects, run_rows, curves = [], [], []
    for game in (0, 1):
        for subject in range(1, 7):
            tag = f"sub{subject:03d}_game{game}"
            roots = {m: args.output_root/C.MODULES[m]/tag for m in ("03", "05")}
            for m, root in roots.items():
                group.verify_artifacts(root); signatures[tag+"/"+m] = C.sha(root/"manifest.json")
            revision = pd.read_csv(roots["03"]/"run_effects.csv")
            names = ["incumbent_change_given_alternative", "alternative_change_given_incumbent"]
            table = revision[revision.contrast.isin(names)].pivot(index="run", columns="contrast", values="delta_ll")
            subject_effects = original[(original.subject==subject)&(original.game==game)&original.contrast.isin(names)]
            valid = (len(subject_effects)==2 and subject_effects.delta_ll.notna().all()
                     and all(n in table for n in names)
                     and len(table)>0 and np.isfinite(table[names].to_numpy()).all())
            difference = table[names[0]]-table[names[1]] if valid else pd.Series(dtype=float)
            label = "incumbent_versus_alternative_direct"
            effects.append(dict(subject=subject, game=game, family="03", primary="", contrast=label,
                delta_ll=float(difference.mean()) if valid else np.nan,
                estimable=bool(valid), runs=len(difference),
                reason="" if valid else "Both original effect estimates must be complete"))
            for run, value in difference.items(): run_rows.append(dict(subject=subject, game=game, contrast=label, run=run, delta_ll=value))
            try:
                horizon = pd.read_csv(roots["05"]/"horizon_alignment.csv")
            except pd.errors.EmptyDataError:
                horizon = pd.DataFrame(columns=["horizon", "mode", "run", "action_match", "persistence_match",
                    "root_match", "persistence_pixel_error", "pixel_error", "n", "targets"])
            for low, high in C.HORIZONS:
                h = f"{low}-{high}"
                candidates = horizon[horizon.horizon==h]
                # Baseline contrasts are defined for all sampled action paths, with
                # inverse-target weighting already applied in the per-run table.
                for metric, mode, first, second in (
                    ("path_vs_persistence", "all_paths", "action_match", "persistence_match"),
                    ("path_vs_root_policy", "all_paths", "action_match", "root_match"),
                    ("state_vs_persistence", "prefix_matched", "persistence_pixel_error", "pixel_error")):
                    rows = candidates[candidates["mode"]==mode].copy()
                    values = rows[first]-rows[second]
                    valid_rows = np.isfinite(values)
                    # Availability is a conditional estimand, not an all-run estimate.
                    # Require observations in >=2 held-out sessions for a subject effect.
                    sessions = {run.split("_ses")[1].split("_")[0] for run in rows.loc[valid_rows, "run"]}
                    valid_effect = bool(valid_rows.any() and len(sessions)>=2)
                    name = f"{metric}_h{h}"
                    effects.append(dict(subject=subject, game=game, family="05", primary="", contrast=name,
                        delta_ll=float(values[valid_rows].mean()) if valid_effect else np.nan,
                        estimable=valid_effect, runs=int(valid_rows.sum()),
                        reason="" if valid_effect else "Horizon unavailable in at least two sessions; no population test",
                        effect_unit="agreement_probability" if metric.startswith("path") else "normalized_pixel_MSE_gain"))
                    for row, value in zip(rows.itertuples(), values):
                        run_rows.append(dict(subject=subject, game=game, contrast=name, run=row.run,
                            delta_ll=value, n=row.n, targets=row.targets))
            # Descriptive computation curves supplement Figure 4. Each subject
            # contributes one curve; they are not treated as independent frames.
            cache = args.output_root/"00_native_cache"/tag
            signatures[tag+"/cache"] = C.sha(cache/"manifest.json")
            d = pd.read_csv(cache/"decisions.csv.gz", usecols=["decision_id", "root_entropy", "begin", "budget", "scored"])
            d = d[d.scored].copy()
            d["bin"] = pd.qcut(d.root_entropy, 5, labels=False, duplicates="drop")
            for b, rows in d.groupby("bin"):
                curves.append(dict(subject=subject, game=game, kind="initiation", bin=b,
                    predictor=rows.root_entropy.mean(), probability=rows.begin.mean(), count=len(rows)))
            e = pd.read_csv(cache/"search_events.csv.gz", usecols=["decision_id", "event_index", "prior_evidence_abs", "stop_choice", "forced_terminal", "control"])
            e = e[(e.event_index>0)&e.decision_id.isin(d.decision_id)].copy()
            e["bin"] = pd.qcut(e.prior_evidence_abs, 5, labels=False, duplicates="drop")
            for b, rows in e.groupby("bin"):
                curves.append(dict(subject=subject, game=game, kind="stopping", bin=b,
                    predictor=rows.prior_evidence_abs.mean(), probability=rows.stop_choice.mean(), count=len(rows)))
    additions = pd.DataFrame(effects)
    augmented = pd.concat((original[original.family.isin(["03", "05"])&original.primary.fillna("").eq("")], additions), ignore_index=True)
    identity = C.digest(dict(inputs=signatures, sources=C.sources()))
    out.mkdir(exist_ok=True)
    if (out/"manifest.json").exists():
        old = json.loads((out/"manifest.json").read_text())
        if old["signature"] != identity: raise FileExistsError("Preserving incompatible supplementary output")
        if old["status"]=="COMPLETED": group.verify_artifacts(out); return
    manifest = dict(status="RUNNING", version=C.VERSION, signature=identity, inputs=signatures)
    C.write_json(out/"manifest.json", manifest)
    additions.to_csv(out/"additional_subject_effects.csv", index=False)
    pd.DataFrame(run_rows).to_csv(out/"additional_run_effects.csv", index=False)
    results, paired = [], []
    for family, rows in augmented.groupby("family"):
        result = group.infer(rows)
        result["effect_unit"] = result.contrast.map(lambda s: "agreement_probability" if s.startswith("path_vs_") else "normalized_pixel_MSE_gain" if s.startswith("state_vs_") else "nats")
        direct = result.contrast.eq("incumbent_versus_alternative_direct")&result.p_max_t.lt(.05)&result.n_subjects.eq(6)
        result.loc[direct, "decision"] = "차이 지지됨"
        results.append(result)
        wide = rows.pivot(index=["subject", "contrast"], columns="game", values="delta_ll").reset_index()
        wide["delta_ll"] = wide[1]-wide[0]; wide["family"] = family; wide["game"] = "Pong-minus-Enduro"
        game_result = group.infer(wide)
        game_result["effect_unit"] = game_result.contrast.map(dict(zip(result.contrast, result.effect_unit)))
        game_result.loc[game_result.p_max_t.lt(.05)&game_result.n_subjects.eq(6), "decision"] = "게임 차이 지지됨"
        paired.append(game_result)
    pd.concat(results).to_csv(out/"enlarged_secondary_family_max_t.csv", index=False)
    pd.concat(paired).to_csv(out/"paired_game_contrasts.csv", index=False)
    curve = pd.DataFrame(curves)
    curve.to_csv(out/"computation_probability_curves.csv", index=False)
    fig, ax = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for game in (0, 1):
        for column, kind in enumerate(("initiation", "stopping")):
            for subject, rows in curve[(curve.game==game)&(curve.kind==kind)].groupby("subject"):
                ax[game, column].plot(rows.predictor, rows.probability, "o-", alpha=.7, label=f"S{subject}")
            ax[game, column].set(title=f"game{game}: {kind} (descriptive)",
                xlabel="Initial root entropy" if kind=="initiation" else "Evidence accumulated before control",
                ylabel="Observed initiation probability" if kind=="initiation" else "Observed voluntary STOP probability")
            ax[game, column].legend(fontsize=7)
    C.save_figure(fig, out, "figure4_computation_context_curves")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for game in (0, 1):
        for subject in range(1, 7):
            rows = additions[(additions.game==game)&(additions.subject==subject)&additions.contrast.str.startswith("path_vs_persistence")]
            axes[game, 0].plot(range(len(rows)), rows.delta_ll, "o-", alpha=.7, label=f"S{subject}")
        axes[game, 0].axhline(0, color="gray"); axes[game, 0].set_xticks(range(6), [f"{a}-{b}" for a,b in C.HORIZONS], rotation=30)
        axes[game, 0].set(title=f"game{game}: past path agreement gain", xlabel="Prediction depth", ylabel="Probability gain vs persistence")
        rr = pd.DataFrame(run_rows).reindex(columns=["game", "contrast", "targets"])
        rr = rr[(rr.game==game)&rr.contrast.str.startswith("path_vs_persistence")]
        coverage = rr.groupby("contrast", sort=False).targets.sum()
        axes[game, 1].bar(range(len(coverage)), coverage.values)
        axes[game, 1].set_xticks(range(len(coverage)), [s.split("_h")[-1] for s in coverage.index], rotation=30)
        axes[game, 1].set(title="Observed target coverage (not independent subjects)", xlabel="Prediction depth", ylabel="Target count")
    C.save_figure(fig, out, "figure5_horizon_gain_coverage")
    for metric in ("path_vs_persistence", "path_vs_root_policy", "state_vs_persistence"):
        infer_rows = pd.concat(results)
        infer_rows = infer_rows[infer_rows.contrast.str.startswith(metric)].copy()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        for game, ax in enumerate(axes):
            rows = infer_rows[infer_rows.game==game].set_index("contrast")
            for i, (low, high) in enumerate(C.HORIZONS):
                row = rows.loc[f"{metric}_h{low}-{high}"]
                vals = [row[f"sub{s:03d}"] for s in range(1, 7)]
                ax.scatter(i+np.linspace(-.12, .12, 6), vals, s=16, alpha=.65)
                if np.isfinite(row.mean_delta_ll):
                    ax.errorbar(i, row.mean_delta_ll, yerr=[[row.mean_delta_ll-row.ci95_low],
                        [row.ci95_high-row.mean_delta_ll]], fmt="D", color="black", ms=4)
                ax.text(i, .98, f"n={row.n_subjects}", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8)
            ax.axhline(0, color="gray", lw=1)
            ax.set_xticks(range(6), [f"{a}-{b}" for a, b in C.HORIZONS], rotation=30)
            ax.set(title=f"game{game}: {metric}", xlabel="Prediction depth; dots = subjects, bars = pointwise 95% t CI",
                   ylabel="Agreement probability gain" if metric.startswith("path") else "Normalized pixel MSE gain")
        C.save_figure(fig, out, metric+"_subject_effects")
    C.write_json(out/"qa.json", dict(population_unit="six subjects", primary_inference="unchanged from 08",
        correction="Use enlarged_secondary_family_max_t for complete 03/05 secondary families",
        action_agreement_selection="all_paths only; never select full-prefix agreement",
        state_error_selection="matched actual action prefix; error was available only at target observation",
        horizon_estimand="conditional on forecast availability in >=2 sessions; run-equal means",
        inference_assumption="subject-level sign-exchangeability",
        n_unestimable_subject_effects=int(additions.delta_ll.isna().sum())))
    C.finish(out, manifest, group_size=6)


if __name__ == "__main__":
    main()
