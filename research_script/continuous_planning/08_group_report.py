"""Full-cohort verification and synchronized six-subject inference across games."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common as C


EXPECTED = {0: dict(complete_real=474846, scored_real=474776, episodes=70),
            1: dict(complete_real=474853, scored_real=474787, episodes=66)}


def verify_artifacts(root):
    m = json.loads((root/"manifest.json").read_text())
    if m["status"] != "COMPLETED":
        raise ValueError(f"Incomplete module: {root}")
    for name, expected in m["artifacts"].items():
        path = root/name
        if not path.is_file() or not path.stat().st_size or C.sha(path) != expected:
            raise ValueError(f"Missing, empty or changed artifact: {path}")
    return m


def verify_subject(out, subject, game, allow_smoke=False, urgency_module=None):
    tag = f"sub{subject:03d}_game{game}"
    cache = verify_artifacts(out/"00_native_cache"/tag)
    if cache["smoke"] and not allow_smoke:
        raise ValueError("Smoke data cannot enter full-cohort inference")
    if not allow_smoke and cache["source_runs"] != 11:
        raise ValueError("Expected all 11 source runs")
    modules = {}
    for module, name in C.MODULES.items():
        if module=="06" and urgency_module: name=urgency_module
        m = verify_artifacts(out/name/tag)
        if m["smoke"] and not allow_smoke:
            raise ValueError(f"Smoke module: {module}")
        if module not in ("07",):
            if not list((out/name/tag/"figures").glob("*.png")):
                raise ValueError(f"Missing scientific figure: {name}")
        modules[module] = dict(primary_estimable=m.get("primary_estimable"),
                               urgency_inference=m.get("urgency_inference"), status=m["status"])
    return dict(subject=subject, game=game, cache=cache, modules=modules)


def infer(frame, primary=False):
    results = []
    if frame.empty: return pd.DataFrame()
    frame = frame.copy()
    frame["key"] = frame.game.astype(str)+":"+frame.contrast
    pivot = frame.pivot(index="subject", columns="key", values="delta_ll").reindex(range(1, 7))
    effects = pivot.to_numpy(float)
    stat, raw, adjusted = C.exact_max_t(effects)
    for j, key in enumerate(pivot.columns):
        meta = frame[frame.key==key].iloc[0]
        values = effects[:, j]; valid = np.isfinite(values)
        se = values.std(ddof=1)/np.sqrt(6) if valid.all() else np.nan
        mean = values.mean() if valid.all() else np.nan
        ci = student_t.ppf(.975, 5)*se
        result = dict(game=meta.game, contrast=meta.contrast, family="primary" if primary else str(meta.family),
            mean_delta_ll=mean, ci95_low=mean-ci, ci95_high=mean+ci,
            t=stat[j], p_exact=raw[j], p_max_t=adjusted[j], n_subjects=int(valid.sum()),
            decision="추정 불가" if not valid.all() else
                ("지지됨" if mean>0 and adjusted[j]<.05 else "추가 설명력 미확립"),
            reason="" if valid.all() else "At least one of the six subject effects is not estimable; no reduced-N test")
        for subject, value in enumerate(values, 1): result[f"sub{subject:03d}"] = value
        results.append(result)
    return pd.DataFrame(results)


def forest(frame, root, title):
    fig, ax = plt.subplots(figsize=(10, max(3, .46*len(frame)+1)))
    for i, row in enumerate(frame.itertuples()):
        values = np.array([getattr(row, f"sub{s:03d}") for s in range(1, 7)])
        ax.scatter(values, np.full(6, i)+np.linspace(-.12, .12, 6), s=15, alpha=.6)
        if np.isfinite(row.mean_delta_ll):
            ax.errorbar(row.mean_delta_ll, i, xerr=[[row.mean_delta_ll-row.ci95_low], [row.ci95_high-row.mean_delta_ll]],
                        fmt="D", color="black", ms=4)
    ax.set_yticks(range(len(frame)), [f"game{r.game}: {r.contrast}" for r in frame.itertuples()], fontsize=8)
    ax.axvline(0, color="gray", lw=1); ax.set_xlabel("Delta log probability; dots = six subjects; bars = pointwise 95% t CI")
    ax.set_title(title)
    C.save_figure(fig, root, "group_effects")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=C.DEFAULT_OUTPUT)
    parser.add_argument("--verify-subject", type=int)
    parser.add_argument("--game", type=int, choices=(0, 1))
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--urgency-annotations", type=Path,
                        help="Select separately preserved annotation-validated urgency variant")
    args = parser.parse_args()
    if args.verify_subject:
        report = verify_subject(args.output_root, args.verify_subject, args.game, args.allow_smoke)
        C.write_json(args.output_root/"jobs"/f"sub{args.verify_subject:03d}_game{args.game}.completed.json", report)
        print(f"Verified subject {args.verify_subject}, game {args.game}", flush=True); return
    urgency_module = C.MODULES["06"]
    if args.urgency_annotations:
        urgency_module += "_validated_"+C.sha(args.urgency_annotations)[:12]
    reports, effects, signatures = [], [], {}
    for game in (0, 1):
        game_reports = []
        for subject in range(1, 7):
            report = verify_subject(args.output_root, subject, game, urgency_module=urgency_module)
            reports.append(report); game_reports.append(report)
            for module in ("01", "02", "03", "04", "05", "06"):
                path = args.output_root/(urgency_module if module=="06" else C.MODULES[module])/f"sub{subject:03d}_game{game}"
                e = pd.read_csv(path/"effects.csv", dtype={"family": str})
                e["primary"] = e.primary.fillna("")
                effects.append(e)
                signatures[str(path)] = C.sha(path/"manifest.json")
        for key, expected in EXPECTED[game].items():
            actual = sum(r["cache"][key] for r in game_reports)
            if actual != expected:
                raise ValueError(f"Cohort acceptance failed: game{game} {key}={actual}, expected {expected}")
        if sum(r["cache"]["source_runs"] for r in game_reports) != 66:
            raise ValueError("Game source-run count mismatch")
    all_effects = pd.concat(effects, ignore_index=True)
    report_name = "08_group_report"
    if args.urgency_annotations: report_name += "_validated_"+C.sha(args.urgency_annotations)[:12]
    out = args.output_root/report_name
    out.mkdir(exist_ok=True)
    identity = C.digest(dict(signatures=signatures, sources=C.sources()))
    manifest_file = out/"manifest.json"
    if manifest_file.exists():
        previous = json.loads(manifest_file.read_text())
        if previous["signature"] != identity:
            raise FileExistsError("Preserving group output for another input/source version")
        if previous["status"] == "COMPLETED":
            verify_artifacts(out); return
    manifest = dict(version=C.VERSION, signature=identity, status="RUNNING", inputs=signatures)
    C.write_json(manifest_file, manifest)
    all_effects.to_csv(out/"subject_effects.csv", index=False)
    primary = all_effects[all_effects.primary!=""]
    if len(primary)!=36 or set(primary.primary)!={"H1", "H2", "H3"}:
        raise ValueError("Expected three primary hypotheses x two games x six subjects")
    primary_result = infer(primary, primary=True)
    primary_result.to_csv(out/"primary_max_t.csv", index=False)
    forest(primary_result, out, "Primary family: synchronized 64 sign patterns across both games")
    secondary = []
    for family, rows in all_effects[all_effects.primary==""].groupby("family"):
        result = infer(rows)
        secondary.append(result)
        for game in (0, 1):
            code = str(family).zfill(2)
            root = args.output_root/(urgency_module if code=="06" else C.MODULES[code])/f"group_game{game}"
            if args.urgency_annotations and code!="06":
                root = root/("validated_"+C.sha(args.urgency_annotations)[:12])
            root.mkdir(exist_ok=True, parents=True)
            subset = result[result.game==game]
            subset.to_csv(root/"inference.csv", index=False)
            forest(subset, root, f"Secondary family {family}: games jointly corrected")
            C.write_json(root/"manifest.json", dict(status="COMPLETED", group_size=6,
                source_group_signature=identity, correction="64 subject sign flips, max-|t| across both games and contrasts"))
    pd.concat(secondary, ignore_index=True).to_csv(out/"secondary_max_t.csv", index=False)
    # Direct paired game difference; never infer interaction from separate significance.
    paired = all_effects.pivot(index=["subject", "family", "contrast"], columns="game", values="delta_ll").reset_index()
    paired["delta_ll"] = paired[1]-paired[0]
    paired["game"] = "Pong-minus-Enduro"
    differences = []
    for family, rows in paired.groupby("family"):
        differences.append(infer(rows))
    pd.concat(differences).to_csv(out/"paired_game_contrasts.csv", index=False)
    for game in (0, 1):
        templates = [pd.read_csv(args.output_root/C.MODULES["06"]/f"sub{s:03d}_game{game}"/"annotation_template.csv") for s in range(1, 7)]
        template = pd.concat(templates, ignore_index=True)
        if template.frame_id.nunique()!=200: raise ValueError("Annotation frame sampling is not exactly 200 per game")
        template.to_csv(out/f"annotation_template_game{game}.csv", index=False)
    C.write_json(out/"qa.json", dict(full_cohort_verified=True, subjects=6, games=2, captures=48,
        sources=132, episodes=136, acceptance=EXPECTED,
        n_unestimable_subject_effects=int(all_effects.delta_ll.isna().sum()),
        urgency="Interpret only if independent annotation passed; missing validation is not a null result",
        inference_assumption="subject-level sign-exchangeability; six subjects imply coarse exact p-values"))
    lines = ["# Continuous planning v1: group summary", "", "Observational reanalysis; model computation is a proxy, not observed human thought.", ""]
    for row in primary_result.itertuples():
        lines.append(f"- game{row.game} {row.contrast}: {row.decision}; Delta LL={row.mean_delta_ll:.6g}, corrected p={row.p_max_t:.6g}, n={row.n_subjects}.")
    lines += ["", "Urgency inference is withheld without independent 200-frame-per-game annotation.",
              "Neural output is native-time regressor export only; fMRI was not fit."]
    (out/"summary.md").write_text("\n".join(lines)+"\n")
    C.finish(out, manifest, group_size=6, full_cohort=True)


if __name__ == "__main__":
    main()
